#!/usr/bin/env python3
# ruff: noqa: E402
"""Diagnostic GGUF Q8 KV format sweep.

This is intentionally a *numerical format* probe rather than a new runtime KV
implementation.  It runs a BF16 GGUF resident session, captures the per-layer
BF16 full-attention KV caches after prefill, then quantize/dequantizes those
caches on the host with candidate Q8 layouts and replays one decode step through
the existing BF16 attention path.

The full-model replay is approximate because the newly appended decode-token KV
row is still BF16 in the replay path.  To keep the first-pass scan honest, the
script also reports exact layer-local CPU attention drift for the known-sensitive
first full-attention layer after appending the decode-token KV row.

Use this to decide which Q8 layouts are worth implementing in HIP and running
through the real BF16-vs-Q8 guard.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.benchmark.correctness import evaluate_logits
from hipengine.core.dtype import DType
from hipengine.core.hip import HipMemcpyKind
from hipengine.kvcache import FixedPagedKVPolicy
import hipengine.runtime.qwen35_gguf_runner as gguf_runtime
from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession

DEFAULT_MODEL = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
DEFAULT_FORMATS = (
    "per_token_head_fp16",
    "per_token_head_fp32",
    "q8_0_block32_fp16",
    "q8_0_block32_fp32",
    "block16_fp16",
    "block16_fp32",
    "block64_fp16",
    "block64_fp32",
    "key_bf16_value_q8_0_block32_fp16",
    "key_bf16_value_q8_0_block32_fp32",
    "key_q8_0_block32_value_bf16_fp16",
    "key_q8_0_block32_value_bf16_fp32",
)


@dataclass(frozen=True)
class QuantSpec:
    block_size: int
    scale_dtype: str = "fp16"


@dataclass(frozen=True)
class FormatSpec:
    name: str
    key: QuantSpec | None
    value: QuantSpec | None


def _parse_count(text: str) -> int:
    value = text.strip().lower()
    if value.endswith("k"):
        return int(float(value[:-1]) * 1024)
    return int(value)


def _read_compiler_version(path: Path | None) -> str | None:
    if path is None:
        return None
    return path.read_text().strip()


def _command(args: argparse.Namespace) -> str:
    parts = [
        "python3 scripts/qwen35_gguf_q8_format_sweep.py",
        f"--model {args.model}",
        f"--prompt-length {args.prompt_length_raw}",
        f"--token-id {args.token_id}",
        f"--max-sequence-length {args.max_sequence_length}",
        f"--formats {args.formats}",
        f"--bf16-prefix-full-layers {args.bf16_prefix_full_layers}",
        f"--kl-threshold {args.kl_threshold}",
        f"--top1-threshold {args.top1_threshold}",
    ]
    if args.compiler_version_file is not None:
        parts.append(f"--compiler-version-file {args.compiler_version_file}")
    if args.require_cached_build:
        parts.append("--require-cached-build")
    if args.json is not None:
        parts.append(f"--json {args.json}")
    return " ".join(parts)


def _bf16_bits_to_f32(bits: np.ndarray) -> np.ndarray:
    arr = np.asarray(bits, dtype=np.uint16)
    return (arr.astype(np.uint32) << np.uint32(16)).view(np.float32)


def _f32_to_bf16_bits(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    bits = arr.view(np.uint32)
    lsb = (bits >> np.uint32(16)) & np.uint32(1)
    rounded = bits + np.uint32(0x7FFF) + lsb
    return (rounded >> np.uint32(16)).astype(np.uint16)


def _read_array(runtime, ptr: int, shape: tuple[int, ...], dtype: np.dtype | type) -> np.ndarray:
    out = np.empty(shape, dtype=np.dtype(dtype))
    runtime.memcpy(int(out.ctypes.data), int(ptr), int(out.nbytes), HipMemcpyKind.DEVICE_TO_HOST)
    return out


def _write_array(runtime, ptr: int, array: np.ndarray) -> None:
    arr = np.ascontiguousarray(array)
    runtime.memcpy(int(ptr), int(arr.ctypes.data), int(arr.nbytes), HipMemcpyKind.HOST_TO_DEVICE)


def _read_bf16(runtime, ptr: int, numel: int) -> np.ndarray:
    bits = _read_array(runtime, ptr, (int(numel),), np.uint16)
    return _bf16_bits_to_f32(bits).astype(np.float32, copy=False)


def _compare(ref: np.ndarray, cand: np.ndarray) -> dict[str, Any]:
    diff = np.asarray(cand, dtype=np.float32) - np.asarray(ref, dtype=np.float32)
    abs_diff = np.abs(diff)
    ref_abs = np.maximum(np.abs(ref), np.float32(1.0e-8))
    return {
        "max_abs": float(np.max(abs_diff)) if abs_diff.size else 0.0,
        "mean_abs": float(np.mean(abs_diff)) if abs_diff.size else 0.0,
        "rms_abs": float(np.sqrt(np.mean(diff.astype(np.float64) ** 2))) if diff.size else 0.0,
        "max_rel": float(np.max(abs_diff / ref_abs)) if abs_diff.size else 0.0,
    }


def _logit_metrics(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    kl_threshold: float,
    top1_threshold: float,
) -> dict[str, Any]:
    metrics = evaluate_logits(
        reference.reshape(1, -1),
        candidate.reshape(1, -1),
        kl_threshold=float(kl_threshold),
        top1_threshold=float(top1_threshold),
    )
    return {
        "passed": bool(metrics.passed),
        "kl_mean": float(metrics.kl_mean),
        "kl_max": float(metrics.kl_max),
        "top1_agreement": float(metrics.top1_agreement),
        "reference_top1": int(np.argmax(reference)),
        "candidate_top1": int(np.argmax(candidate)),
        "max_abs": float(np.max(np.abs(candidate - reference))),
        "mean_abs": float(np.mean(np.abs(candidate - reference))),
    }


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x))).astype(np.float32)


def _attention_cpu(
    query: np.ndarray,
    key_rows: np.ndarray,
    value_rows: np.ndarray,
    gate: np.ndarray,
    *,
    scale: float,
) -> np.ndarray:
    num_q_heads, head_dim = query.shape
    num_kv_heads = key_rows.shape[1]
    group = num_q_heads // num_kv_heads
    out = np.empty((num_q_heads, head_dim), dtype=np.float32)
    for q_head in range(num_q_heads):
        kv_head = q_head // group
        scores = (key_rows[:, kv_head, :].astype(np.float32) @ query[q_head].astype(np.float32)) * np.float32(scale)
        scores = scores - np.max(scores)
        weights = np.exp(scores).astype(np.float32)
        weights = weights / np.sum(weights, dtype=np.float32)
        out[q_head] = weights.astype(np.float32) @ value_rows[:, kv_head, :].astype(np.float32)
    gated = out * _sigmoid(gate)
    return _bf16_bits_to_f32(_f32_to_bf16_bits(gated)).astype(np.float32, copy=False)


def _format_spec(name: str, head_dim: int) -> FormatSpec:
    lower = name.strip().lower()
    if lower == "per_token_head_fp16":
        spec = QuantSpec(block_size=head_dim, scale_dtype="fp16")
        return FormatSpec(lower, spec, spec)
    if lower == "per_token_head_fp32":
        spec = QuantSpec(block_size=head_dim, scale_dtype="fp32")
        return FormatSpec(lower, spec, spec)
    if lower == "q8_0_block32_fp16":
        spec = QuantSpec(block_size=32, scale_dtype="fp16")
        return FormatSpec(lower, spec, spec)
    if lower == "q8_0_block32_fp32":
        spec = QuantSpec(block_size=32, scale_dtype="fp32")
        return FormatSpec(lower, spec, spec)
    if lower.startswith("block") and lower.endswith(("_fp16", "_fp32")):
        block_text, scale_dtype = lower.rsplit("_", 1)
        try:
            block_size = int(block_text.removeprefix("block"))
        except ValueError as exc:
            raise ValueError(f"unknown Q8 format {name!r}") from exc
        if block_size <= 0:
            raise ValueError(f"block size must be positive in Q8 format {name!r}")
        spec = QuantSpec(block_size=block_size, scale_dtype=scale_dtype)
        return FormatSpec(lower, spec, spec)
    if lower == "key_bf16_value_q8_0_block32_fp16":
        return FormatSpec(lower, None, QuantSpec(block_size=32, scale_dtype="fp16"))
    if lower == "key_bf16_value_q8_0_block32_fp32":
        return FormatSpec(lower, None, QuantSpec(block_size=32, scale_dtype="fp32"))
    if lower == "key_q8_0_block32_value_bf16_fp16":
        return FormatSpec(lower, QuantSpec(block_size=32, scale_dtype="fp16"), None)
    if lower == "key_q8_0_block32_value_bf16_fp32":
        return FormatSpec(lower, QuantSpec(block_size=32, scale_dtype="fp32"), None)
    raise ValueError(f"unknown Q8 format {name!r}")


def _scale_itemsize(scale_dtype: str) -> int:
    if scale_dtype == "fp16":
        return 2
    if scale_dtype == "fp32":
        return 4
    raise ValueError(f"unknown scale dtype {scale_dtype!r}")


def _format_memory_bytes_per_token(spec: FormatSpec, *, num_kv_heads: int, head_dim: int) -> dict[str, Any]:
    def side_bytes(qspec: QuantSpec | None) -> dict[str, int]:
        if qspec is None:
            return {"payload": num_kv_heads * head_dim * DType.BF16.itemsize, "scale": 0}
        scale_blocks = (head_dim + qspec.block_size - 1) // qspec.block_size
        return {
            "payload": num_kv_heads * head_dim * DType.INT8.itemsize,
            "scale": num_kv_heads * scale_blocks * _scale_itemsize(qspec.scale_dtype),
        }

    key = side_bytes(spec.key)
    value = side_bytes(spec.value)
    total = key["payload"] + key["scale"] + value["payload"] + value["scale"]
    bf16_total = 2 * num_kv_heads * head_dim * DType.BF16.itemsize
    return {
        "key_payload_bytes": int(key["payload"]),
        "key_scale_bytes": int(key["scale"]),
        "value_payload_bytes": int(value["payload"]),
        "value_scale_bytes": int(value["scale"]),
        "total_bytes": int(total),
        "bf16_total_bytes": int(bf16_total),
        "pct_of_bf16": float(100.0 * total / bf16_total),
        "saves_vs_bf16_pct": float(100.0 * (bf16_total - total) / bf16_total),
    }


def _quant_dequant(values: np.ndarray, spec: QuantSpec | None) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if spec is None:
        return arr.copy()
    block_size = int(spec.block_size)
    if block_size <= 0 or arr.shape[-1] % block_size != 0:
        raise ValueError(f"head_dim={arr.shape[-1]} must be divisible by block_size={block_size}")
    original_shape = arr.shape
    reshaped = arr.reshape(*arr.shape[:-1], arr.shape[-1] // block_size, block_size)
    max_abs = np.max(np.abs(reshaped), axis=-1, keepdims=True).astype(np.float32)
    scale = max_abs / np.float32(127.0)
    if spec.scale_dtype == "fp16":
        scale = scale.astype(np.float16).astype(np.float32)
    elif spec.scale_dtype != "fp32":
        raise ValueError(f"unknown scale dtype {spec.scale_dtype!r}")
    safe_scale = np.where(scale > 0.0, scale, np.float32(1.0))
    quantized = np.rint(reshaped / safe_scale)
    quantized = np.clip(quantized, -127.0, 127.0).astype(np.int8)
    dequantized = quantized.astype(np.float32) * scale
    return dequantized.reshape(original_shape)


def _qdq_pair(key_rows: np.ndarray, value_rows: np.ndarray, spec: FormatSpec) -> tuple[np.ndarray, np.ndarray]:
    return _quant_dequant(key_rows, spec.key), _quant_dequant(value_rows, spec.value)


def _backup_state_buffers(runtime, session: Qwen35GGUFResidentSession) -> list[tuple[object, int]]:
    if session.scratch is None:
        return []
    backups: list[tuple[object, int]] = []
    for buffer in (*session.scratch.layer_conv_states, *session.scratch.layer_recurrent_states):
        if buffer is None:
            continue
        backup_ptr = runtime.malloc(buffer.nbytes)
        runtime.memcpy(backup_ptr, buffer.ptr, buffer.nbytes, HipMemcpyKind.DEVICE_TO_DEVICE)
        backups.append((buffer, backup_ptr))
    return backups


def _restore_state_buffers(runtime, backups: list[tuple[object, int]]) -> None:
    for buffer, backup_ptr in backups:
        runtime.memcpy(buffer.ptr, backup_ptr, buffer.nbytes, HipMemcpyKind.DEVICE_TO_DEVICE)


def _free_state_backups(runtime, backups: list[tuple[object, int]]) -> None:
    for _buffer, backup_ptr in backups:
        runtime.free(backup_ptr)


def _cache_shape(session: Qwen35GGUFResidentSession) -> tuple[int, int, int, int]:
    if session.scratch is None or session.runner is None or session.runner.weights is None:
        raise RuntimeError("GGUF resident session is not initialized")
    cfg = session.runner.weights.config
    blocks = int(session.scratch.max_positions // session.scratch.block_size)
    return (blocks, int(session.scratch.block_size), int(cfg.head_count_kv), int(cfg.key_length))


def _full_attention_layer_ids(session: Qwen35GGUFResidentSession) -> list[int]:
    if session.runner is None or session.runner.weights is None:
        raise RuntimeError("GGUF resident session is not initialized")
    return [
        int(layer_id)
        for layer_id, layer_type in enumerate(session.runner.weights.config.layer_types)
        if layer_type == gguf_runtime.FULL_ATTENTION
    ]


def _read_cache_backups(session: Qwen35GGUFResidentSession) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    if session.scratch is None:
        raise RuntimeError("GGUF resident session is not initialized")
    runtime = session.runtime
    shape = _cache_shape(session)
    backups: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for layer_id in _full_attention_layer_ids(session):
        key_cache, value_cache = session.scratch.full_cache(layer_id)
        backups[layer_id] = (
            _read_array(runtime, key_cache.ptr, shape, np.uint16),
            _read_array(runtime, value_cache.ptr, shape, np.uint16),
        )
    return backups


def _restore_cache_backups(session: Qwen35GGUFResidentSession, backups: dict[int, tuple[np.ndarray, np.ndarray]]) -> None:
    if session.scratch is None:
        raise RuntimeError("GGUF resident session is not initialized")
    runtime = session.runtime
    for layer_id, (key_bits, value_bits) in backups.items():
        key_cache, value_cache = session.scratch.full_cache(layer_id)
        _write_array(runtime, key_cache.ptr, key_bits)
        _write_array(runtime, value_cache.ptr, value_bits)


def _stage_qdq_caches(
    session: Qwen35GGUFResidentSession,
    backups: dict[int, tuple[np.ndarray, np.ndarray]],
    spec: FormatSpec,
    *,
    active_context: int,
    quantized_layer_ids: set[int],
) -> None:
    if session.scratch is None:
        raise RuntimeError("GGUF resident session is not initialized")
    runtime = session.runtime
    for layer_id, (key_bits, value_bits) in backups.items():
        if layer_id not in quantized_layer_ids:
            continue
        key_cache, value_cache = session.scratch.full_cache(layer_id)
        staged_key = key_bits.copy()
        staged_value = value_bits.copy()
        key_rows = _bf16_bits_to_f32(staged_key.reshape(-1, key_bits.shape[2], key_bits.shape[3])[:active_context])
        value_rows = _bf16_bits_to_f32(staged_value.reshape(-1, value_bits.shape[2], value_bits.shape[3])[:active_context])
        q_key, q_value = _qdq_pair(key_rows, value_rows, spec)
        staged_key.reshape(-1, key_bits.shape[2], key_bits.shape[3])[:active_context] = _f32_to_bf16_bits(q_key)
        staged_value.reshape(-1, value_bits.shape[2], value_bits.shape[3])[:active_context] = _f32_to_bf16_bits(q_value)
        _write_array(runtime, key_cache.ptr, staged_key)
        _write_array(runtime, value_cache.ptr, staged_value)


def _decode_once(
    session: Qwen35GGUFResidentSession,
    *,
    position: int,
    token_id: int,
    state_backups: list[tuple[object, int]],
) -> tuple[int, np.ndarray]:
    if session.scratch is None or session.runner is None or session.runner.weights is None:
        raise RuntimeError("GGUF resident session is not initialized")
    runtime = session.runtime
    _restore_state_buffers(runtime, state_backups)
    session._set_full_attention_position_device(position)
    session._set_token_id_device(token_id)
    hidden_size = int(session.runner.hidden_size)
    src = session._hidden_a
    dst = session._hidden_b
    if src is None or dst is None:
        raise RuntimeError("session hidden buffers are closed")
    for layer_id, layer_type in enumerate(session.runner.weights.config.layer_types):
        if layer_type == gguf_runtime.LINEAR_ATTENTION:
            session.runner._run_linear_attention_layer(layer_id, src.ptr, dst.ptr, session.scratch)
        elif layer_type == gguf_runtime.FULL_ATTENTION:
            session.runner._run_full_attention_layer(layer_id, src.ptr, dst.ptr, session.scratch, position=position)
        else:
            raise ValueError(f"unsupported layer type {layer_type!r}")
        src, dst = dst, src
    gguf_runtime.gguf_rmsnorm_bf16_f32_weight(
        src.ptr,
        session.runner.weights.root("output_norm").allocation().tensor.ptr,
        session.scratch.norm.ptr,
        rows=1,
        hidden_size=hidden_size,
        eps=session.runner.weights.config.rms_norm_eps,
        runtime=runtime,
    )
    result = session._sample_from_hidden(session.scratch.norm.ptr, return_logits=True)
    return int(result.token_id), np.asarray(result.logits, dtype=np.float32).reshape(-1).copy()


def _layer_local_probe(
    session: Qwen35GGUFResidentSession,
    specs: list[FormatSpec],
    *,
    layer_id: int,
    position: int,
    token_id: int,
    state_backups: list[tuple[object, int]],
) -> dict[str, Any]:
    if session.scratch is None or session.runner is None or session.runner.weights is None:
        raise RuntimeError("GGUF resident session is not initialized")
    runtime = session.runtime
    cfg = session.runner.weights.config
    cache_shape = _cache_shape(session)
    _restore_state_buffers(runtime, state_backups)
    session._set_full_attention_position_device(position)
    session._set_token_id_device(token_id)
    src = session._hidden_a
    dst = session._hidden_b
    if src is None or dst is None:
        raise RuntimeError("session hidden buffers are closed")
    for current_layer_id, layer_type in enumerate(session.runner.weights.config.layer_types):
        if current_layer_id == layer_id:
            break
        if layer_type == gguf_runtime.LINEAR_ATTENTION:
            session.runner._run_linear_attention_layer(current_layer_id, src.ptr, dst.ptr, session.scratch)
        elif layer_type == gguf_runtime.FULL_ATTENTION:
            session.runner._run_full_attention_layer(current_layer_id, src.ptr, dst.ptr, session.scratch, position=position)
        else:
            raise ValueError(f"unsupported layer type {layer_type!r}")
        src, dst = dst, src
    session.runner._run_full_attention_attn_only(layer_id, src.ptr, session.scratch.attn_out.ptr, session.scratch, position=position)
    active_context = int(position) + 1
    key_cache, value_cache = session.scratch.full_cache(layer_id)
    key_rows = _bf16_bits_to_f32(_read_array(runtime, key_cache.ptr, cache_shape, np.uint16)).reshape(
        -1,
        int(cfg.head_count_kv),
        int(cfg.key_length),
    )[:active_context]
    value_rows = _bf16_bits_to_f32(_read_array(runtime, value_cache.ptr, cache_shape, np.uint16)).reshape(
        -1,
        int(cfg.head_count_kv),
        int(cfg.key_length),
    )[:active_context]
    query = _read_array(runtime, session.scratch.full_query.ptr, (int(cfg.head_count), int(cfg.key_length)), np.float32)
    gate = _read_bf16(runtime, session.scratch.full_gate.ptr, int(cfg.head_count) * int(cfg.key_length)).reshape(
        int(cfg.head_count),
        int(cfg.key_length),
    )
    reference = _attention_cpu(query, key_rows, value_rows, gate, scale=float(int(cfg.key_length) ** -0.5))
    format_rows = []
    for spec in specs:
        q_key, q_value = _qdq_pair(key_rows, value_rows, spec)
        candidate = _attention_cpu(query, q_key, q_value, gate, scale=float(int(cfg.key_length) ** -0.5))
        format_rows.append(
            {
                "format": spec.name,
                "gated_attention_diff": _compare(reference.reshape(-1), candidate.reshape(-1)),
                "key_quantization_error": _compare(key_rows.reshape(-1), q_key.reshape(-1)),
                "value_quantization_error": _compare(value_rows.reshape(-1), q_value.reshape(-1)),
            }
        )
    return {
        "layer_id": int(layer_id),
        "active_context": int(active_context),
        "rows": format_rows,
    }


def _parse_int_list(value: str) -> list[int]:
    parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not parsed:
        raise ValueError("expected at least one integer")
    return parsed


def run(args: argparse.Namespace) -> dict[str, Any]:
    prompt_length = _parse_count(args.prompt_length_raw)
    max_sequence_length = int(args.max_sequence_length or max(8192, prompt_length + 2))
    if prompt_length + 1 > max_sequence_length:
        raise ValueError("max-sequence-length must leave room for one decode token")
    compiler_version = _read_compiler_version(args.compiler_version_file)
    prompt_tokens = [int(args.token_id)] * int(prompt_length)
    start = time.perf_counter()
    policy = FixedPagedKVPolicy(block_size=256, storage_dtype=DType.BF16)
    format_names = [item.strip() for item in args.formats.split(",") if item.strip()]
    if not format_names:
        format_names = list(DEFAULT_FORMATS)
    prefix_values = sorted(set(_parse_int_list(args.bf16_prefix_full_layers)))
    payload: dict[str, Any]
    with Qwen35GGUFResidentSession(
        args.model,
        compiler_version=compiler_version,
        require_cached_build=bool(args.require_cached_build),
        max_sequence_length=max_sequence_length,
        use_wmma_prefill=True,
        use_gemv_decode=True,
        kv_policy=policy,
    ) as session:
        if session.scratch is None or session.runner is None or session.runner.weights is None:
            raise RuntimeError("GGUF resident session is not initialized")
        cfg = session.runner.weights.config
        specs = [_format_spec(name, int(cfg.key_length)) for name in format_names]
        first = session.prefill(prompt_tokens, use_bulk=True, bulk_attention_mode="bulk", return_logits=True)
        decode_token = int(first.token_id)
        position = int(prompt_length)
        layer_ids = _full_attention_layer_ids(session)
        probe_layer = layer_ids[int(args.probe_full_attention_index)] if args.probe_layer_id is None else int(args.probe_layer_id)
        cache_backups = _read_cache_backups(session)
        state_backups = _backup_state_buffers(session.runtime, session)
        try:
            _restore_cache_backups(session, cache_backups)
            bf16_token, bf16_logits = _decode_once(
                session,
                position=position,
                token_id=decode_token,
                state_backups=state_backups,
            )
            format_results = []
            full_count = len(layer_ids)
            bf16_bytes_per_token_per_full_layer = 2 * int(cfg.head_count_kv) * int(cfg.key_length) * DType.BF16.itemsize
            for prefix in prefix_values:
                if prefix < 0 or prefix > full_count:
                    raise ValueError(f"BF16 prefix {prefix} must be within [0, {full_count}]")
                quantized_layer_ids = set(layer_ids[prefix:])
                for spec in specs:
                    _restore_cache_backups(session, cache_backups)
                    _stage_qdq_caches(
                        session,
                        cache_backups,
                        spec,
                        active_context=prompt_length,
                        quantized_layer_ids=quantized_layer_ids,
                    )
                    token, logits = _decode_once(
                        session,
                        position=position,
                        token_id=decode_token,
                        state_backups=state_backups,
                    )
                    metrics = _logit_metrics(
                        bf16_logits,
                        logits,
                        kl_threshold=float(args.kl_threshold),
                        top1_threshold=float(args.top1_threshold),
                    )
                    per_layer_memory = _format_memory_bytes_per_token(
                        spec,
                        num_kv_heads=int(cfg.head_count_kv),
                        head_dim=int(cfg.key_length),
                    )
                    hybrid_total = prefix * bf16_bytes_per_token_per_full_layer + (full_count - prefix) * int(
                        per_layer_memory["total_bytes"]
                    )
                    hybrid_bf16_total = full_count * bf16_bytes_per_token_per_full_layer
                    format_results.append(
                        {
                            "format": spec.name,
                            "bf16_prefix_full_attention_layers": int(prefix),
                            "quantized_full_attention_layers": int(full_count - prefix),
                            "token_id": int(token),
                            "memory_bytes_per_token_per_full_layer": per_layer_memory,
                            "hybrid_memory_bytes_per_token_all_full_layers": {
                                "total_bytes": int(hybrid_total),
                                "bf16_total_bytes": int(hybrid_bf16_total),
                                "pct_of_bf16": float(100.0 * hybrid_total / hybrid_bf16_total),
                                "saves_vs_bf16_pct": float(
                                    100.0 * (hybrid_bf16_total - hybrid_total) / hybrid_bf16_total
                                ),
                            },
                            "approx_decode_logit_metrics_vs_bf16": metrics,
                        }
                    )
            _restore_cache_backups(session, cache_backups)
            layer_local = _layer_local_probe(
                session,
                specs,
                layer_id=probe_layer,
                position=position,
                token_id=decode_token,
                state_backups=state_backups,
            )
        finally:
            _restore_cache_backups(session, cache_backups)
            _free_state_backups(session.runtime, state_backups)
    elapsed = time.perf_counter() - start
    gc.collect()
    payload = {
        "schema": 1,
        "mode": "qwen35_gguf_q8_format_sweep",
        "command": _command(args),
        "model": str(args.model),
        "prompt_length": int(prompt_length),
        "decode_steps": 1,
        "decode_token": int(decode_token),
        "max_sequence_length": int(max_sequence_length),
        "elapsed_seconds": float(elapsed),
        "quality_thresholds": {
            "kl_max": float(args.kl_threshold),
            "top1_agreement_min": float(args.top1_threshold),
        },
        "caveats": [
            "Full-model replay is a numerical QDQ probe through BF16 cache storage, not a HIP Q8 implementation.",
            "Full-model replay quantizes retained prefill KV rows; the newly appended decode-token KV row remains BF16.",
            "Layer-local CPU attention rows include the decode-token KV row and isolate the probed full-attention layer.",
        ],
        "reference": {
            "kv_storage": "bf16",
            "token_id": int(bf16_token),
        },
        "formats": format_results,
        "layer_local_probe": layer_local,
        "summary": {
            "approx_passed_formats": [
                {
                    "format": row["format"],
                    "bf16_prefix_full_attention_layers": row["bf16_prefix_full_attention_layers"],
                }
                for row in format_results
                if row["approx_decode_logit_metrics_vs_bf16"]["passed"]
            ],
            "best_approx_kl": min(
                (
                    {
                        "format": row["format"],
                        "bf16_prefix_full_attention_layers": row["bf16_prefix_full_attention_layers"],
                        "kl_max": row["approx_decode_logit_metrics_vs_bf16"]["kl_max"],
                    }
                    for row in format_results
                ),
                key=lambda row: row["kl_max"],
            ),
            "best_layer_local_mean_abs_format": min(
                layer_local["rows"],
                key=lambda row: row["gated_attention_diff"]["mean_abs"],
            )["format"],
        },
    }
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--prompt-length", dest="prompt_length_raw", default="4K")
    parser.add_argument("--token-id", type=int, default=9707)
    parser.add_argument("--max-sequence-length", type=int, default=8192)
    parser.add_argument("--formats", default=",".join(DEFAULT_FORMATS))
    parser.add_argument(
        "--bf16-prefix-full-layers",
        default="0",
        help="Comma-separated BF16 full-attention prefix depths to leave unquantized in the full-model replay.",
    )
    parser.add_argument("--probe-full-attention-index", type=int, default=0)
    parser.add_argument("--probe-layer-id", type=int)
    parser.add_argument("--kl-threshold", type=float, default=0.05)
    parser.add_argument("--top1-threshold", type=float, default=0.90)
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    payload = run(args)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
