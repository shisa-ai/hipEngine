#!/usr/bin/env python3
"""Diagnostic qwen35moe GGUF serial-vs-bulk prefill parity probe.

This script is intentionally a diagnostic gate, not a promoted correctness
oracle.  It compares the current public token-serial qwen35moe GGUF prefill
path against the explicit ``use_bulk=True`` path, then bisects hidden-state drift
by layer limit.  The output is meant to explain why bulk qwen35moe GGUF prefill
remains opt-in while performance work continues.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from hipengine.core.dtype import DType
from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc
from hipengine.kernels.hip_gfx1100.fused import gguf_rmsnorm_bf16_f32_weight
from hipengine.loading.qwen35_gguf import FULL_ATTENTION, LINEAR_ATTENTION
from hipengine.quant.gguf import bf16_to_float32
from hipengine.runtime.gguf_embedding import launch_gguf_embedding
from hipengine.runtime.qwen35_gguf_runner import (
    Qwen35GGUFFullStackRunner,
    Qwen35GGUFResidentSession,
    _FullStackScratch,
    _GGUFFullAttentionPrefillScratch,
)

DEFAULT_MODEL = "/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
DEFAULT_TOKEN_IDS = (760, 4087, 369, 220)
DEFAULT_LAYER_LIMITS = (0, 1, 2, 3, 4, 8, 12, 14, 20, 40)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--token-ids",
        default=",".join(str(token) for token in DEFAULT_TOKEN_IDS),
        help="Comma/space separated prompt token ids. Must contain at least four tokens for bulk prefill.",
    )
    parser.add_argument(
        "--layer-limits",
        default=",".join(str(limit) for limit in DEFAULT_LAYER_LIMITS),
        help="Comma/space separated layer-limit values to scan.",
    )
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--compiler-version-file", type=Path, default=None)
    parser.add_argument("--require-cached-build", action="store_true")
    args = parser.parse_args(argv)

    token_ids = _parse_ints(args.token_ids)
    layer_limits = tuple(_parse_ints(args.layer_limits))
    if len(token_ids) < 4:
        raise ValueError("qwen35moe GGUF bulk prefill requires at least four tokens")
    compiler_version = args.compiler_version_file.read_text() if args.compiler_version_file else None

    started = time.perf_counter()
    sample = _sample_serial_and_bulk(
        args.model,
        token_ids,
        compiler_version=compiler_version,
        require_cached_build=bool(args.require_cached_build),
    )
    layer_scan = _scan_layer_drift(args.model, token_ids, layer_limits)
    elapsed = time.perf_counter() - started

    artifact: dict[str, Any] = {
        "schema": 1,
        "status": "diagnostic",
        "model": str(args.model),
        "token_ids": token_ids,
        "elapsed_seconds": elapsed,
        "bulk_prefill_default_allowed": False,
        "summary": {
            "serial_token_id": sample["serial"]["token_id"],
            "bulk_token_id": sample["bulk"]["token_id"],
            "top1_match": sample["comparison"]["top1_match"],
            "kl_serial_to_bulk": sample["comparison"]["kl_serial_to_bulk"],
            "max_abs_logit": sample["comparison"]["max_abs_logit"],
            "first_aotriton_hidden_drift_limit": layer_scan["aotriton_full_attention"]["first_drift_limit"],
            "first_native_full_attention_hidden_drift_limit": layer_scan["native_full_attention"]["first_drift_limit"],
        },
        "sample": sample,
        "layer_scan": layer_scan,
        "decision": {
            "accepted_as_correctness_gate": False,
            "reason": (
                "Current qwen35moe GGUF use_bulk=True prefill is finite and fast but does not yet "
                "match the public token-serial path. Keep it explicit diagnostic-only until an "
                "external or stronger long-prompt oracle resolves the drift."
            ),
        },
    }
    text = json.dumps(artifact, indent=2, ensure_ascii=False)
    print(text)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n")
    return 0


def _parse_ints(text: str) -> list[int]:
    parts = text.replace(",", " ").split()
    if not parts:
        raise ValueError("expected at least one integer")
    return [int(part, 0) for part in parts]


def _sample_serial_and_bulk(
    model: str | Path,
    token_ids: list[int],
    *,
    compiler_version: str | None,
    require_cached_build: bool,
) -> dict[str, Any]:
    with Qwen35GGUFResidentSession(
        model,
        max_sequence_length=max(256, len(token_ids) + 8),
        compiler_version=compiler_version,
        require_cached_build=require_cached_build,
    ) as session:
        serial_started = time.perf_counter()
        serial = session.prefill(token_ids, use_bulk=False, return_logits=True)
        serial_seconds = time.perf_counter() - serial_started
        bulk_started = time.perf_counter()
        bulk = session.prefill(token_ids, use_bulk=True, return_logits=True)
        bulk_seconds = time.perf_counter() - bulk_started

    serial_logits = np.asarray(serial.logits, dtype=np.float32).reshape(-1)
    bulk_logits = np.asarray(bulk.logits, dtype=np.float32).reshape(-1)
    if serial_logits.shape != bulk_logits.shape:
        raise ValueError(f"serial/bulk logits shape mismatch: {serial_logits.shape} vs {bulk_logits.shape}")
    comparison = _logit_comparison(serial_logits, bulk_logits)
    return {
        "serial": {
            "token_id": int(serial.token_id),
            "logit": float(serial.logit),
            "seconds": serial_seconds,
            "finite_logits": bool(np.all(np.isfinite(serial_logits))),
        },
        "bulk": {
            "token_id": int(bulk.token_id),
            "logit": float(bulk.logit),
            "seconds": bulk_seconds,
            "finite_logits": bool(np.all(np.isfinite(bulk_logits))),
        },
        "comparison": comparison,
    }


def _scan_layer_drift(model: str | Path, token_ids: list[int], layer_limits: Iterable[int]) -> dict[str, Any]:
    runtime = get_hip_runtime()
    out: dict[str, Any] = {}
    with Qwen35GGUFFullStackRunner(model, runtime=runtime) as runner:
        max_layer = int(runner.weights.config.block_count)  # type: ignore[union-attr]
        limits = tuple(sorted(set(int(limit) for limit in layer_limits)))
        for limit in limits:
            if limit < 0 or limit > max_layer:
                raise ValueError(f"layer limit {limit} outside [0, {max_layer}]")
        serial_by_limit = {
            limit: runner.run_prompt_hidden(token_ids, layer_limit=limit)
            for limit in limits
        }
        for mode in ("aotriton", "native"):
            entries = []
            first_drift: int | None = None
            for limit in limits:
                bulk_hidden = _layerwise_bulk_hidden(runner, token_ids, layer_limit=limit, full_attention_mode=mode)
                metrics = _hidden_comparison(serial_by_limit[limit], bulk_hidden)
                layer_type = "embedding" if limit == 0 else runner.weights.config.layer_types[limit - 1]  # type: ignore[union-attr]
                entry = {
                    "layer_limit": limit,
                    "last_layer_type": layer_type,
                    **metrics,
                }
                entries.append(entry)
                if first_drift is None and not metrics["bit_equal"]:
                    first_drift = limit
            key = "aotriton_full_attention" if mode == "aotriton" else "native_full_attention"
            out[key] = {
                "first_drift_limit": first_drift,
                "entries": entries,
            }
    return out


def _layerwise_bulk_hidden(
    runner: Qwen35GGUFFullStackRunner,
    token_ids: list[int],
    *,
    layer_limit: int,
    full_attention_mode: str,
) -> np.ndarray:
    if runner.weights is None:
        raise RuntimeError("runner is closed")
    rows = int(len(token_ids))
    runtime = runner.runtime or get_hip_runtime()
    tokens = np.asarray(token_ids, dtype=np.int64)
    buffers = []
    try:
        token_buf = malloc(tokens.nbytes, runtime=runtime)
        hidden_a = malloc(rows * runner.hidden_size * DType.BF16.itemsize, runtime=runtime)
        hidden_b = malloc(rows * runner.hidden_size * DType.BF16.itemsize, runtime=runtime)
        decode_scratch = _FullStackScratch.allocate(runner, runtime=runtime, max_sequence_length=max(256, rows + 8))
        bulk_scratch = _GGUFFullAttentionPrefillScratch.allocate(runner, rows=rows, runtime=runtime)
        buffers = [token_buf, hidden_a, hidden_b, *decode_scratch.buffers, *bulk_scratch.buffers]
        decode_scratch.zero_states(runtime)
        copy_host_to_device(token_buf, host_array_ptr(tokens), tokens.nbytes, runtime=runtime)
        launch_gguf_embedding(
            runner.weights.root("token_embedding"),
            token_buf.ptr,
            hidden_a.ptr,
            rows=rows,
            hidden_size=runner.hidden_size,
            vocab_size=runner.vocab_size,
            runtime=runtime,
        )
        src = hidden_a
        dst = hidden_b
        row_nbytes = runner.hidden_size * DType.BF16.itemsize
        for layer_id, layer_type in enumerate(runner.weights.config.layer_types[:layer_limit]):
            if layer_type == LINEAR_ATTENTION:
                runner._run_linear_attention_prefill_layer_rows(
                    layer_id,
                    src.ptr,
                    dst.ptr,
                    bulk_scratch,
                    rows=rows,
                    decode_scratch=decode_scratch,
                )
            elif layer_type == FULL_ATTENTION:
                if full_attention_mode == "aotriton":
                    key_cache, value_cache = decode_scratch.full_cache(layer_id)
                    layer_scratch = replace(bulk_scratch, key_cache=key_cache, value_cache=value_cache)
                    runner._run_full_attention_prefill_layer_aotriton(layer_id, src.ptr, dst.ptr, layer_scratch)
                elif full_attention_mode == "native":
                    for row in range(rows):
                        decode_scratch.set_full_attention_position(row, runtime)
                        runner._run_full_attention_layer(
                            layer_id,
                            src.ptr + row * row_nbytes,
                            dst.ptr + row * row_nbytes,
                            decode_scratch,
                            position=row,
                        )
                else:
                    raise ValueError(f"unsupported full attention mode {full_attention_mode!r}")
            else:
                raise ValueError(f"unsupported GGUF layer type {layer_type!r}")
            src, dst = dst, src
        gguf_rmsnorm_bf16_f32_weight(
            src.ptr,
            runner.weights.root("output_norm").allocation().tensor.ptr,
            bulk_scratch.norm.ptr,
            rows=rows,
            hidden_size=runner.hidden_size,
            eps=runner.weights.config.rms_norm_eps,
            runtime=runtime,
        )
        runtime.device_synchronize()
        hidden = np.empty((rows, runner.hidden_size), dtype=np.uint16)
        copy_device_to_host(host_array_ptr(hidden), bulk_scratch.norm, hidden.nbytes, runtime=runtime)
        return hidden[-1:]
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)


def _hidden_comparison(ref_bits: np.ndarray, cand_bits: np.ndarray) -> dict[str, Any]:
    ref = bf16_to_float32(np.asarray(ref_bits, dtype=np.uint16))
    cand = bf16_to_float32(np.asarray(cand_bits, dtype=np.uint16))
    diff = ref - cand
    return {
        "bit_equal": bool(np.array_equal(ref_bits, cand_bits)),
        "max_abs": float(np.max(np.abs(diff))),
        "mean_abs": float(np.mean(np.abs(diff))),
        "nonzero_count": int(np.count_nonzero(diff)),
    }


def _logit_comparison(ref: np.ndarray, cand: np.ndarray) -> dict[str, Any]:
    diff = ref - cand
    ref_prob = _softmax64(ref)
    cand_prob = _softmax64(cand)
    kl = float(np.sum(ref_prob * (np.log(ref_prob + 1e-30) - np.log(cand_prob + 1e-30))))
    ref_argmax = int(np.argmax(ref))
    cand_argmax = int(np.argmax(cand))
    return {
        "top1_match": bool(ref_argmax == cand_argmax),
        "serial_argmax": ref_argmax,
        "bulk_argmax": cand_argmax,
        "kl_serial_to_bulk": kl,
        "max_abs_logit": float(np.max(np.abs(diff))),
        "mean_abs_logit": float(np.mean(np.abs(diff))),
        "finite": bool(np.all(np.isfinite(ref)) and np.all(np.isfinite(cand))),
    }


def _softmax64(values: np.ndarray) -> np.ndarray:
    values64 = np.asarray(values, dtype=np.float64)
    shifted = values64 - np.max(values64)
    probs = np.exp(shifted)
    return probs / np.sum(probs)


if __name__ == "__main__":
    raise SystemExit(main())
