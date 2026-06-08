#!/usr/bin/env python3
"""Deterministic c>N-vs-independent-c1 and GPU A/A smokes for Qwen3.5/PARO primitives."""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from hipengine.core.device import Device
from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc
from hipengine.core.tensor import Tensor
from hipengine.kernels.hip_gfx1100.attention import (
    build_qwen35_paged_attn_decode,
    build_qwen35_paged_kv_write,
    qwen35_full_attn_decode_context_bf16,
    qwen35_paged_full_attn_decode_context_bf16_batch_spans,
    qwen35_paged_full_attn_decode_context_bf16_spans,
    qwen35_write_paged_kv_mixed_value_bf16_batch_spans,
    qwen35_write_paged_kv_mixed_value_bf16_spans,
)
from hipengine.kvcache import KVLiveSpans
from hipengine.loading import float_array_to_bf16_bits
from scripts.qwen35_batch_constants import (
    RETAINED_ARTIFACT_PRIMITIVE_CORRECTNESS_NUMPY_MAX_ABS_LIMIT,
    RETAINED_ARTIFACT_REQUIRED_PRIMITIVE_CORRECTNESS_SCHEMA,
    RETAINED_ARTIFACT_REQUIRED_PRIMITIVE_CORRECTNESS_SEED,
    RETAINED_ARTIFACT_REQUIRED_PRIMITIVE_CORRECTNESS_SHAPE_FIELDS,
)

_REQUIRED_PRIMITIVE_CORRECTNESS_SCHEMA = RETAINED_ARTIFACT_REQUIRED_PRIMITIVE_CORRECTNESS_SCHEMA
_REQUIRED_PRIMITIVE_CORRECTNESS_SEED = RETAINED_ARTIFACT_REQUIRED_PRIMITIVE_CORRECTNESS_SEED
_REQUIRED_PRIMITIVE_CORRECTNESS_SHAPE_FIELDS = RETAINED_ARTIFACT_REQUIRED_PRIMITIVE_CORRECTNESS_SHAPE_FIELDS
_PRIMITIVE_CORRECTNESS_NUMPY_MAX_ABS_LIMIT = RETAINED_ARTIFACT_PRIMITIVE_CORRECTNESS_NUMPY_MAX_ABS_LIMIT


def _payload_json(payload) -> str:
    return json.dumps(payload, indent=2, allow_nan=False)


def _bf16_to_f32(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << np.uint32(16)).view(np.float32)


def _device_tensor(ptr: int, shape: tuple[int, ...], dtype: str) -> Tensor:
    return Tensor.from_handle(ptr, shape, dtype, Device("hip", 0))


class _DeviceArena:
    def __init__(self):
        self.runtime = get_hip_runtime()
        self.buffers = []

    def dev(self, array: np.ndarray):
        buf = malloc(array.nbytes, runtime=self.runtime)
        self.buffers.append(buf)
        copy_host_to_device(buf, host_array_ptr(array), runtime=self.runtime)
        return buf

    def close(self) -> None:
        for buf in reversed(self.buffers):
            free(buf, runtime=self.runtime)
        self.buffers.clear()


def _visible_hip_device_metadata(runtime) -> dict[str, object]:
    env_keys = ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL")
    visible_env: dict[str, str] = {}
    for key in env_keys:
        value = os.environ.get(key)
        if value is not None and value.strip():
            visible_env[key] = value
    metadata: dict[str, object] = {"env": visible_env}
    library = runtime.library
    try:
        library.hipGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
        library.hipGetDeviceCount.restype = ctypes.c_int
        count = ctypes.c_int()
        count_error = int(library.hipGetDeviceCount(ctypes.byref(count)))
        metadata["hipGetDeviceCount_error"] = count_error
        metadata["visible_device_count"] = int(count.value)
        if count_error != 0 or count.value <= 0:
            return metadata

        library.hipGetDevice.argtypes = [ctypes.POINTER(ctypes.c_int)]
        library.hipGetDevice.restype = ctypes.c_int
        current_device = ctypes.c_int()
        device_error = int(library.hipGetDevice(ctypes.byref(current_device)))
        metadata["hipGetDevice_error"] = device_error
        device_index = int(current_device.value) if device_error == 0 else 0
        metadata["current_device"] = device_index

        library.hipDeviceGetName.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
        library.hipDeviceGetName.restype = ctypes.c_int
        name = ctypes.create_string_buffer(256)
        name_error = int(library.hipDeviceGetName(name, ctypes.c_int(len(name)), ctypes.c_int(device_index)))
        metadata["hipDeviceGetName_error"] = name_error
        if name_error == 0:
            metadata["device_name"] = name.value.decode("utf-8", errors="replace")
    except Exception as exc:  # pragma: no cover - defensive provenance only.
        metadata["error"] = f"{type(exc).__name__}: {exc}"
    return metadata


def _numpy_attention(
    query: np.ndarray,
    key_cache: np.ndarray,
    value_cache: np.ndarray,
    context_lens: np.ndarray,
    *,
    scale: float,
) -> np.ndarray:
    rows, num_q_heads, head_dim = query.shape
    num_kv_heads = key_cache.shape[3]
    kv_group = num_q_heads // num_kv_heads
    key = _bf16_to_f32(key_cache)
    value = _bf16_to_f32(value_cache)
    out = np.zeros((rows, num_q_heads, head_dim), dtype=np.float32)
    for row in range(rows):
        context_len = int(context_lens[row])
        row_key = key[row].reshape(-1, num_kv_heads, head_dim)
        row_value = value[row].reshape(-1, num_kv_heads, head_dim)
        for q_head in range(num_q_heads):
            kv_head = q_head // kv_group
            scores = np.empty(context_len, dtype=np.float32)
            for token in range(context_len):
                scores[token] = float((query[row, q_head] * row_key[token, kv_head]).sum() * scale)
            probs = np.exp(scores - scores.max())
            probs = probs / probs.sum()
            for token, prob in enumerate(probs):
                out[row, q_head] += prob * row_value[token, kv_head]
    return out


def _primitive_correctness_passed(
    append_key_mismatch: int,
    append_value_mismatch: int,
    batch_vs_c1: float,
    batch_vs_numpy: float,
    *,
    append_batch_aa_key_mismatch: int = 0,
    append_batch_aa_value_mismatch: int = 0,
    attn_batch_aa_max_abs: float = 0.0,
) -> bool:
    return (
        append_key_mismatch == 0
        and append_value_mismatch == 0
        and append_batch_aa_key_mismatch == 0
        and append_batch_aa_value_mismatch == 0
        and float(batch_vs_c1) == 0.0
        and float(attn_batch_aa_max_abs) == 0.0
        and math.isfinite(float(batch_vs_numpy))
        and 0.0 <= float(batch_vs_numpy) <= _PRIMITIVE_CORRECTNESS_NUMPY_MAX_ABS_LIMIT
    )


def _default_context_lens(rows: int, max_context_len: int) -> np.ndarray:
    return np.asarray([(idx % max_context_len) + 1 for idx in range(rows)], dtype=np.int64)


def _parse_context_lens(text: str, *, rows: int, max_context_len: int) -> np.ndarray:
    values = [int(part) for part in text.split(",") if part.strip()]
    if len(values) != rows:
        raise ValueError("context_lens length must match rows")
    if any(value <= 0 or value > max_context_len for value in values):
        raise ValueError("context_lens values must be in 1..max_context_len")
    return np.asarray(values, dtype=np.int64)


def _max_abs_delta(lhs: np.ndarray, rhs: np.ndarray) -> dict[str, object]:
    delta = np.abs(lhs - rhs)
    flat_index = int(np.argmax(delta))
    index = tuple(int(value) for value in np.unravel_index(flat_index, delta.shape))
    return {
        "max_abs": float(delta[index]),
        "index": list(index),
        "lhs": float(lhs[index]),
        "rhs": float(rhs[index]),
    }


def _fill_context_cache_rows(
    key_cache: np.ndarray,
    value_cache: np.ndarray,
    key_cache_f32: np.ndarray,
    value_cache_f32: np.ndarray,
    context_lens: np.ndarray,
) -> None:
    rows, blocks, block_size, num_kv_heads, head_dim = key_cache.shape
    flat_tokens = blocks * block_size
    for row, context_len in enumerate(context_lens):
        if int(context_len) > flat_tokens:
            raise ValueError("context_len exceeds cache capacity")
        row_key = key_cache[row].reshape(flat_tokens, num_kv_heads, head_dim)
        row_value = value_cache[row].reshape(flat_tokens, num_kv_heads, head_dim)
        row_key_f32 = key_cache_f32[row].reshape(flat_tokens, num_kv_heads, head_dim)
        row_value_f32 = value_cache_f32[row].reshape(flat_tokens, num_kv_heads, head_dim)
        row_key[: int(context_len)] = float_array_to_bf16_bits(row_key_f32[: int(context_len)])
        row_value[: int(context_len)] = float_array_to_bf16_bits(row_value_f32[: int(context_len)])


def run(
    rows: int,
    *,
    seed: int = _REQUIRED_PRIMITIVE_CORRECTNESS_SEED,
    block_size: int = _REQUIRED_PRIMITIVE_CORRECTNESS_SHAPE_FIELDS["block_size"],
    max_context_len: int = _REQUIRED_PRIMITIVE_CORRECTNESS_SHAPE_FIELDS["max_context_len"],
    num_q_heads: int = _REQUIRED_PRIMITIVE_CORRECTNESS_SHAPE_FIELDS["num_q_heads"],
    num_kv_heads: int = _REQUIRED_PRIMITIVE_CORRECTNESS_SHAPE_FIELDS["num_kv_heads"],
    head_dim: int = _REQUIRED_PRIMITIVE_CORRECTNESS_SHAPE_FIELDS["head_dim"],
    context_lens: np.ndarray | None = None,
    include_dense_c1: bool = False,
) -> dict[str, object]:
    if rows <= 0:
        raise ValueError("rows must be positive")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if max_context_len <= 0:
        raise ValueError("max_context_len must be positive")
    if num_q_heads <= 0 or num_kv_heads <= 0 or num_q_heads % num_kv_heads != 0:
        raise ValueError("num_q_heads must be positive and divisible by num_kv_heads")
    if head_dim <= 0 or head_dim > 256:
        raise ValueError("head_dim must be in 1..256")
    rng = np.random.default_rng(seed)
    blocks = (max_context_len + block_size - 1) // block_size
    scale = 1.0 / np.sqrt(head_dim)
    if context_lens is None:
        context_lens = _default_context_lens(rows, max_context_len)
    else:
        context_lens = np.asarray(context_lens, dtype=np.int64)
        if context_lens.shape != (rows,):
            raise ValueError("context_lens shape must match rows")
        if np.any(context_lens <= 0) or np.any(context_lens > max_context_len):
            raise ValueError("context_lens values must be in 1..max_context_len")
    positions = context_lens - 1
    block_table = np.tile(np.arange(blocks, dtype=np.int32), (rows, 1))

    # Append smoke: batch append should match independent c1 append into the same row-major layout.
    append_key = rng.normal(0.0, 0.25, size=(rows, num_kv_heads, head_dim)).astype(np.float32)
    append_value_f32 = rng.normal(0.0, 0.25, size=(rows, num_kv_heads, head_dim)).astype(np.float32)
    append_value = float_array_to_bf16_bits(append_value_f32)
    batch_key_cache = np.zeros((rows, blocks, block_size, num_kv_heads, head_dim), dtype=np.uint16)
    batch_value_cache = np.zeros_like(batch_key_cache)
    c1_key_cache = np.zeros_like(batch_key_cache)
    c1_value_cache = np.zeros_like(batch_key_cache)

    arena = _DeviceArena()
    kv_lib = build_qwen35_paged_kv_write(load=True)
    attn_lib = build_qwen35_paged_attn_decode(load=True)
    try:
        bt = arena.dev(block_table)
        pos = arena.dev(positions)
        key = arena.dev(append_key)
        value = arena.dev(append_value)
        batch_key_cache_aa = np.zeros_like(batch_key_cache)
        batch_value_cache_aa = np.zeros_like(batch_value_cache)
        bkc = arena.dev(batch_key_cache)
        bvc = arena.dev(batch_value_cache)
        bkc_aa = arena.dev(batch_key_cache_aa)
        bvc_aa = arena.dev(batch_value_cache_aa)
        ckc = arena.dev(c1_key_cache)
        cvc = arena.dev(c1_value_cache)
        write_spans = KVLiveSpans.paged_uniform(
            block_table=_device_tensor(bt.ptr, block_table.shape, "int32"),
            live_counts=_device_tensor(pos.ptr, positions.shape, "int64"),
            max_live_count=int(positions.max()),
            storage_dtype="bf16",
        )
        qwen35_write_paged_kv_mixed_value_bf16_batch_spans(
            key.ptr,
            value.ptr,
            bkc.ptr,
            bvc.ptr,
            write_spans,
            rows,
            block_size,
            num_kv_heads,
            head_dim,
            library=kv_lib,
            runtime=arena.runtime,
        )
        qwen35_write_paged_kv_mixed_value_bf16_batch_spans(
            key.ptr,
            value.ptr,
            bkc_aa.ptr,
            bvc_aa.ptr,
            write_spans,
            rows,
            block_size,
            num_kv_heads,
            head_dim,
            library=kv_lib,
            runtime=arena.runtime,
        )
        row_cache_bytes = blocks * block_size * num_kv_heads * head_dim * np.dtype(np.uint16).itemsize
        row_kv_bytes = num_kv_heads * head_dim * np.dtype(np.float32).itemsize
        row_value_bytes = num_kv_heads * head_dim * np.dtype(np.uint16).itemsize
        row_table_bytes = blocks * np.dtype(np.int32).itemsize
        pos_bytes = np.dtype(np.int64).itemsize
        for row in range(rows):
            row_spans = KVLiveSpans.paged_uniform(
                block_table=_device_tensor(bt.ptr + row * row_table_bytes, (blocks,), "int32"),
                live_counts=_device_tensor(pos.ptr + row * pos_bytes, (1,), "int64"),
                max_live_count=int(positions[row]),
                storage_dtype="bf16",
            )
            qwen35_write_paged_kv_mixed_value_bf16_spans(
                key.ptr + row * row_kv_bytes,
                value.ptr + row * row_value_bytes,
                ckc.ptr + row * row_cache_bytes,
                cvc.ptr + row * row_cache_bytes,
                row_spans,
                block_size,
                num_kv_heads,
                head_dim,
                library=kv_lib,
                runtime=arena.runtime,
            )
        copy_device_to_host(host_array_ptr(batch_key_cache), bkc, runtime=arena.runtime)
        copy_device_to_host(host_array_ptr(batch_value_cache), bvc, runtime=arena.runtime)
        copy_device_to_host(host_array_ptr(batch_key_cache_aa), bkc_aa, runtime=arena.runtime)
        copy_device_to_host(host_array_ptr(batch_value_cache_aa), bvc_aa, runtime=arena.runtime)
        copy_device_to_host(host_array_ptr(c1_key_cache), ckc, runtime=arena.runtime)
        copy_device_to_host(host_array_ptr(c1_value_cache), cvc, runtime=arena.runtime)
        append_key_mismatch = int(np.count_nonzero(batch_key_cache != c1_key_cache))
        append_value_mismatch = int(np.count_nonzero(batch_value_cache != c1_value_cache))
        append_batch_aa_key_mismatch = int(np.count_nonzero(batch_key_cache != batch_key_cache_aa))
        append_batch_aa_value_mismatch = int(np.count_nonzero(batch_value_cache != batch_value_cache_aa))

        # Attention smoke: batch context decode should match independent c1 decode and NumPy oracle.
        key_cache_f32 = rng.normal(0.0, 0.25, size=(rows, blocks, block_size, num_kv_heads, head_dim)).astype(np.float32)
        value_cache_f32 = rng.normal(0.0, 0.25, size=(rows, blocks, block_size, num_kv_heads, head_dim)).astype(np.float32)
        key_cache = np.zeros_like(batch_key_cache)
        value_cache = np.zeros_like(batch_value_cache)
        _fill_context_cache_rows(key_cache, value_cache, key_cache_f32, value_cache_f32, context_lens)
        query = rng.normal(0.0, 0.25, size=(rows, num_q_heads, head_dim)).astype(np.float32)
        batch_out = np.zeros((rows, num_q_heads, head_dim), dtype=np.float32)
        batch_out_aa = np.zeros_like(batch_out)
        c1_out = np.zeros_like(batch_out)
        dense_c1_out = np.zeros_like(batch_out) if include_dense_c1 else None
        query_b = arena.dev(query)
        key_cache_b = arena.dev(key_cache)
        value_cache_b = arena.dev(value_cache)
        live_b = arena.dev(context_lens)
        batch_out_b = arena.dev(batch_out)
        batch_out_aa_b = arena.dev(batch_out_aa)
        c1_out_b = arena.dev(c1_out)
        dense_c1_out_b = arena.dev(dense_c1_out) if dense_c1_out is not None else None
        decode_spans = KVLiveSpans.paged_uniform(
            block_table=_device_tensor(bt.ptr, block_table.shape, "int32"),
            live_counts=_device_tensor(live_b.ptr, context_lens.shape, "int64"),
            max_live_count=max_context_len,
            storage_dtype="bf16",
        )
        qwen35_paged_full_attn_decode_context_bf16_batch_spans(
            query_b.ptr,
            key_cache_b.ptr,
            value_cache_b.ptr,
            batch_out_b.ptr,
            decode_spans,
            rows,
            max_context_len,
            block_size,
            num_q_heads,
            num_kv_heads,
            head_dim,
            scale,
            library=attn_lib,
            runtime=arena.runtime,
        )
        qwen35_paged_full_attn_decode_context_bf16_batch_spans(
            query_b.ptr,
            key_cache_b.ptr,
            value_cache_b.ptr,
            batch_out_aa_b.ptr,
            decode_spans,
            rows,
            max_context_len,
            block_size,
            num_q_heads,
            num_kv_heads,
            head_dim,
            scale,
            library=attn_lib,
            runtime=arena.runtime,
        )
        row_query_bytes = num_q_heads * head_dim * np.dtype(np.float32).itemsize
        row_out_bytes = row_query_bytes
        live_bytes = np.dtype(np.int64).itemsize
        for row in range(rows):
            row_spans = KVLiveSpans.paged_uniform(
                block_table=_device_tensor(bt.ptr + row * row_table_bytes, (blocks,), "int32"),
                live_counts=_device_tensor(live_b.ptr + row * live_bytes, (1,), "int64"),
                max_live_count=max_context_len,
                storage_dtype="bf16",
            )
            qwen35_paged_full_attn_decode_context_bf16_spans(
                query_b.ptr + row * row_query_bytes,
                key_cache_b.ptr + row * row_cache_bytes,
                value_cache_b.ptr + row * row_cache_bytes,
                c1_out_b.ptr + row * row_out_bytes,
                row_spans,
                max_context_len,
                block_size,
                num_q_heads,
                num_kv_heads,
                head_dim,
                scale,
                library=attn_lib,
                runtime=arena.runtime,
            )
            if dense_c1_out_b is not None:
                qwen35_full_attn_decode_context_bf16(
                    query_b.ptr + row * row_query_bytes,
                    key_cache_b.ptr + row * row_cache_bytes,
                    value_cache_b.ptr + row * row_cache_bytes,
                    dense_c1_out_b.ptr + row * row_out_bytes,
                    live_b.ptr + row * live_bytes,
                    max_context_len,
                    num_q_heads,
                    num_kv_heads,
                    head_dim,
                    scale,
                    library=attn_lib,
                    runtime=arena.runtime,
                )
        copy_device_to_host(host_array_ptr(batch_out), batch_out_b, runtime=arena.runtime)
        copy_device_to_host(host_array_ptr(batch_out_aa), batch_out_aa_b, runtime=arena.runtime)
        copy_device_to_host(host_array_ptr(c1_out), c1_out_b, runtime=arena.runtime)
        if dense_c1_out_b is not None and dense_c1_out is not None:
            copy_device_to_host(host_array_ptr(dense_c1_out), dense_c1_out_b, runtime=arena.runtime)
        expected = _numpy_attention(query, key_cache, value_cache, context_lens, scale=scale)
    finally:
        arena.close()

    batch_vs_c1_delta = _max_abs_delta(batch_out, c1_out)
    batch_vs_numpy_delta = _max_abs_delta(batch_out, expected)
    attn_batch_aa_delta = _max_abs_delta(batch_out, batch_out_aa)
    batch_vs_c1 = float(batch_vs_c1_delta["max_abs"])
    batch_vs_numpy = float(batch_vs_numpy_delta["max_abs"])
    attn_batch_aa = float(attn_batch_aa_delta["max_abs"])
    result = {
        "schema": _REQUIRED_PRIMITIVE_CORRECTNESS_SCHEMA,
        "rows": rows,
        "seed": seed,
        "block_size": block_size,
        "max_context_len": max_context_len,
        "num_q_heads": num_q_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "context_lens": context_lens.tolist(),
        "device": _visible_hip_device_metadata(arena.runtime),
        "append_key_mismatch": append_key_mismatch,
        "append_value_mismatch": append_value_mismatch,
        "append_batch_aa_key_mismatch": append_batch_aa_key_mismatch,
        "append_batch_aa_value_mismatch": append_batch_aa_value_mismatch,
        "attn_batch_vs_c1_max_abs": batch_vs_c1,
        "attn_batch_vs_c1_delta": batch_vs_c1_delta,
        "attn_batch_vs_numpy_max_abs": batch_vs_numpy,
        "attn_batch_vs_numpy_delta": batch_vs_numpy_delta,
        "attn_batch_aa_max_abs": attn_batch_aa,
        "attn_batch_aa_delta": attn_batch_aa_delta,
        "aa_passed": (
            append_batch_aa_key_mismatch == 0
            and append_batch_aa_value_mismatch == 0
            and float(attn_batch_aa) == 0.0
        ),
        "passed": _primitive_correctness_passed(
            append_key_mismatch,
            append_value_mismatch,
            batch_vs_c1,
            batch_vs_numpy,
            append_batch_aa_key_mismatch=append_batch_aa_key_mismatch,
            append_batch_aa_value_mismatch=append_batch_aa_value_mismatch,
            attn_batch_aa_max_abs=attn_batch_aa,
        ),
    }
    if dense_c1_out is not None:
        batch_vs_dense_delta = _max_abs_delta(batch_out, dense_c1_out)
        paged_c1_vs_dense_delta = _max_abs_delta(c1_out, dense_c1_out)
        dense_vs_numpy_delta = _max_abs_delta(dense_c1_out, expected)
        result.update(
            {
                "attn_batch_vs_dense_c1_max_abs": float(batch_vs_dense_delta["max_abs"]),
                "attn_batch_vs_dense_c1_delta": batch_vs_dense_delta,
                "attn_paged_c1_vs_dense_c1_max_abs": float(paged_c1_vs_dense_delta["max_abs"]),
                "attn_paged_c1_vs_dense_c1_delta": paged_c1_vs_dense_delta,
                "attn_dense_c1_vs_numpy_max_abs": float(dense_vs_numpy_delta["max_abs"]),
                "attn_dense_c1_vs_numpy_delta": dense_vs_numpy_delta,
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=2)
    parser.add_argument("--seed", type=int, default=_REQUIRED_PRIMITIVE_CORRECTNESS_SEED)
    parser.add_argument("--block-size", type=int, default=_REQUIRED_PRIMITIVE_CORRECTNESS_SHAPE_FIELDS["block_size"])
    parser.add_argument("--max-context-len", type=int, default=_REQUIRED_PRIMITIVE_CORRECTNESS_SHAPE_FIELDS["max_context_len"])
    parser.add_argument("--num-q-heads", type=int, default=_REQUIRED_PRIMITIVE_CORRECTNESS_SHAPE_FIELDS["num_q_heads"])
    parser.add_argument("--num-kv-heads", type=int, default=_REQUIRED_PRIMITIVE_CORRECTNESS_SHAPE_FIELDS["num_kv_heads"])
    parser.add_argument("--head-dim", type=int, default=_REQUIRED_PRIMITIVE_CORRECTNESS_SHAPE_FIELDS["head_dim"])
    parser.add_argument("--context-lens", help="comma-separated live counts; defaults to 1..max_context_len coverage")
    parser.add_argument("--include-dense-c1", action="store_true", help="also compare batch paged context against the dense c1 short-context kernel")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    context_lens = None
    if args.context_lens:
        context_lens = _parse_context_lens(args.context_lens, rows=args.rows, max_context_len=args.max_context_len)
    result = run(
        args.rows,
        seed=args.seed,
        block_size=args.block_size,
        max_context_len=args.max_context_len,
        num_q_heads=args.num_q_heads,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        context_lens=context_lens,
        include_dense_c1=args.include_dense_c1,
    )
    if args.json is not None:
        result["artifact_path"] = str(args.json)
    payload = _payload_json(result)
    print(payload)
    if args.json is not None:
        args.json.write_text(payload + "\n")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
