#!/usr/bin/env python3
"""Smoke the first native MTP proposal stage on the PARO+MTP-BF16 artifact.

This is not a throughput benchmark and not a full MTP proposal.  It validates the
first real execution boundary after tensor assembly:

  token embedding + target hidden -> MTP pre-fc RMSNorm concat -> mtp.fc

The output is the BF16 hidden row that will feed the one-layer MTP decoder.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from safetensors import safe_open

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.core.dtype import DType
from hipengine.core.memory import copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc
from hipengine.core.tensor import Tensor
from hipengine.kernels.hip_gfx1100.speculative.dflash_drafter import build_dflash_drafter, dflash_dense_bf16_to_bf16
from hipengine.kernels.hip_gfx1100.speculative.mtp import build_mtp_speculative, mtp_fuse_inputs_f16_bf16
from hipengine.loading import TensorInfo, load_tensor_info_to_device, load_weight_index, qwen35_paro_config_from_hf, validate_qwen35_mtp_model
from hipengine.loading.materialize import DeviceTensorAllocation
from hipengine.loading.qwen35_paro import normalize_qwen35_weight_name
from hipengine.loading.safetensors import read_tensor_storage_bytes

DEFAULT_MODEL = Path("/models/hipengine/Qwen3.6-35B-A3B-PARO-full4096-e5-packed-MTP-BF16")


def _normalized_infos(model: str | Path) -> tuple[Any, dict[str, TensorInfo]]:
    index = load_weight_index(model)
    return index, {normalize_qwen35_weight_name(name): info for name, info in index.tensors.items()}


def _load_info(infos: dict[str, TensorInfo], name: str, allocations: list[DeviceTensorAllocation]) -> Tensor:
    info = infos[name]
    alloc = load_tensor_info_to_device(info)
    allocations.append(alloc)
    return alloc.tensor


def _f32_to_bf16_bits(array: np.ndarray) -> np.ndarray:
    f32 = np.asarray(array, dtype=np.float32)
    u32 = f32.view(np.uint32).copy()
    lsb = (u32 >> 16) & np.uint32(1)
    u32 += np.uint32(0x7FFF) + lsb
    return (u32 >> 16).astype(np.uint16)


def _bf16_bits_to_f32(bits: np.ndarray) -> np.ndarray:
    u32 = np.asarray(bits, dtype=np.uint16).astype(np.uint32) << np.uint32(16)
    return u32.view(np.float32)


def _cpu_check(
    *,
    infos: dict[str, TensorInfo],
    token_ids: np.ndarray,
    target_hidden_f32: np.ndarray,
    fused_bits: np.ndarray,
    fc_out_bits: np.ndarray,
    eps: float,
) -> dict[str, Any]:
    embed_info = infos["embed_tokens.weight"]
    with safe_open(str(embed_info.shard_path), framework="numpy") as handle:
        embedding = handle.get_tensor(embed_info.name)[token_ids.astype(np.int64)].astype(np.float32)
    embed_w = _bf16_bits_to_f32(np.frombuffer(read_tensor_storage_bytes(infos["mtp.pre_fc_norm_embedding.weight"]), dtype=np.uint16).copy())
    hidden_w = _bf16_bits_to_f32(np.frombuffer(read_tensor_storage_bytes(infos["mtp.pre_fc_norm_hidden.weight"]), dtype=np.uint16).copy())
    fc_weight = _bf16_bits_to_f32(
        np.frombuffer(read_tensor_storage_bytes(infos["mtp.fc.weight"]), dtype=np.uint16).copy().reshape(fc_out_bits.shape[1], fused_bits.shape[1])
    )
    embed_norm = embedding * (1.0 / np.sqrt(np.mean(embedding * embedding, axis=1, keepdims=True) + eps)) * (1.0 + embed_w.reshape(1, -1))
    hidden_norm = target_hidden_f32 * (1.0 / np.sqrt(np.mean(target_hidden_f32 * target_hidden_f32, axis=1, keepdims=True) + eps)) * (1.0 + hidden_w.reshape(1, -1))
    fused_ref = np.concatenate([embed_norm, hidden_norm], axis=1)
    fused_ref_bits = _f32_to_bf16_bits(fused_ref)
    # Kernel writes BF16, then dense consumes BF16, so round the CPU fused input
    # before the FC reference.
    fc_ref = _bf16_bits_to_f32(fused_ref_bits).astype(np.float32) @ fc_weight.T.astype(np.float32)
    fc_ref_bits = _f32_to_bf16_bits(fc_ref)
    fused_abs = np.abs(_bf16_bits_to_f32(fused_bits) - _bf16_bits_to_f32(fused_ref_bits))
    fc_abs = np.abs(_bf16_bits_to_f32(fc_out_bits) - _bf16_bits_to_f32(fc_ref_bits))
    return {
        "fused_max_abs_diff_after_bf16_round": float(np.max(fused_abs)),
        "fused_exact_bf16": bool(np.array_equal(fused_bits, fused_ref_bits)),
        "fc_max_abs_diff_after_bf16_round": float(np.max(fc_abs)),
        "fc_close_atol_0p25": bool(np.max(fc_abs) <= 0.25),
    }


def run_smoke(model: str | Path, *, token_ids: tuple[int, ...], cpu_check: bool) -> dict[str, Any]:
    validation = validate_qwen35_mtp_model(model)
    validation.raise_for_errors()
    index, infos = _normalized_infos(model)
    config = qwen35_paro_config_from_hf(index.config)
    rows = len(token_ids)
    hidden = int(config.hidden_size)
    vocab = int(config.vocab_size)
    eps = float(config.rms_norm_eps)
    tokens = np.asarray(token_ids, dtype=np.int64)
    if rows <= 0:
        raise ValueError("at least one token id is required")
    if np.any(tokens < 0) or np.any(tokens >= vocab):
        raise ValueError(f"token ids must be in [0, {vocab})")

    # Deterministic synthetic target hidden rows.  The MTP stage only requires a
    # committed target hidden; full E2E proposal will wire real capture rows.
    rng = np.random.default_rng(1234)
    target_hidden_f32 = (rng.standard_normal((rows, hidden), dtype=np.float32) * np.float32(0.25)).astype(np.float32)
    target_hidden_bits = _f32_to_bf16_bits(target_hidden_f32)
    fused_bits = np.zeros((rows, 2 * hidden), dtype=np.uint16)
    fc_out_bits = np.zeros((rows, hidden), dtype=np.uint16)

    allocations: list[DeviceTensorAllocation] = []
    buffers = []
    try:
        embedding = _load_info(infos, "embed_tokens.weight", allocations)
        fc_weight = _load_info(infos, "mtp.fc.weight", allocations)
        embed_norm_w = _load_info(infos, "mtp.pre_fc_norm_embedding.weight", allocations)
        hidden_norm_w = _load_info(infos, "mtp.pre_fc_norm_hidden.weight", allocations)
        token_buf = malloc(tokens.nbytes)
        hidden_buf = malloc(target_hidden_bits.nbytes)
        fused_buf = malloc(fused_bits.nbytes)
        fc_buf = malloc(fc_out_bits.nbytes)
        buffers.extend([token_buf, hidden_buf, fused_buf, fc_buf])
        copy_host_to_device(token_buf, host_array_ptr(tokens), tokens.nbytes)
        copy_host_to_device(hidden_buf, host_array_ptr(target_hidden_bits), target_hidden_bits.nbytes)

        mtp_lib = build_mtp_speculative(load=True)
        dense_lib = build_dflash_drafter(load=True)
        t0 = time.perf_counter()
        mtp_fuse_inputs_f16_bf16(
            token_buf.ptr,
            embedding.ptr,
            hidden_buf.ptr,
            embed_norm_w.ptr,
            hidden_norm_w.ptr,
            fused_buf.ptr,
            rows,
            hidden,
            vocab,
            eps=eps,
            threads=256,
            library=mtp_lib,
        )
        t_fuse = time.perf_counter() - t0
        t1 = time.perf_counter()
        dflash_dense_bf16_to_bf16(
            fused_buf.ptr,
            fc_weight.ptr,
            fc_buf.ptr,
            rows,
            2 * hidden,
            hidden,
            threads=128,
            library=dense_lib,
        )
        t_fc = time.perf_counter() - t1
        copy_device_to_host(host_array_ptr(fused_bits), fused_buf, fused_bits.nbytes)
        copy_device_to_host(host_array_ptr(fc_out_bits), fc_buf, fc_out_bits.nbytes)
    finally:
        for buf in reversed(buffers):
            free(buf)
        for alloc in reversed(allocations):
            alloc.free()

    fused_f32 = _bf16_bits_to_f32(fused_bits)
    fc_f32 = _bf16_bits_to_f32(fc_out_bits)
    result: dict[str, Any] = {
        "status": "passed",
        "model": str(model),
        "tokens": [int(x) for x in token_ids],
        "rows": rows,
        "hidden_size": hidden,
        "vocab_size": vocab,
        "mtp_validation": validation.to_json_dict(),
        "fuse_seconds": t_fuse,
        "fc_seconds": t_fc,
        "fused_finite": bool(np.isfinite(fused_f32).all()),
        "fc_finite": bool(np.isfinite(fc_f32).all()),
        "fused_nonzero": bool(np.any(fused_bits != 0)),
        "fc_nonzero": bool(np.any(fc_out_bits != 0)),
        "fused_sample": [float(x) for x in fused_f32.reshape(-1)[:8]],
        "fc_sample": [float(x) for x in fc_f32.reshape(-1)[:8]],
        "note": "Smoke covers MTP input fusion + mtp.fc only; one-layer decoder/lm-head proposal remains next work.",
    }
    if cpu_check:
        result["cpu_check"] = _cpu_check(
            infos=infos,
            token_ids=tokens,
            target_hidden_f32=target_hidden_f32,
            fused_bits=fused_bits,
            fc_out_bits=fc_out_bits,
            eps=eps,
        )
    return result


def _parse_tokens(text: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    if not values:
        raise ValueError("at least one token id is required")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--tokens", default="151646,8948", help="Comma-separated token ids")
    parser.add_argument("--cpu-check", action="store_true", help="Compare GPU output with a CPU BF16-rounded reference")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    result = run_smoke(args.model, token_ids=_parse_tokens(args.tokens), cpu_check=bool(args.cpu_check))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "rows": result["rows"], "fused_finite": result["fused_finite"], "fc_finite": result["fc_finite"], "fc_seconds": result["fc_seconds"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
