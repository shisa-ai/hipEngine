#!/usr/bin/env python3
"""Deterministic c>N-vs-independent-c1 correctness smokes for Qwen3.5/PARO primitives."""

from __future__ import annotations

import argparse
import json
import math
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
    qwen35_paged_full_attn_decode_context_bf16_batch_spans,
    qwen35_paged_full_attn_decode_context_bf16_spans,
    qwen35_write_paged_kv_mixed_value_bf16_batch_spans,
    qwen35_write_paged_kv_mixed_value_bf16_spans,
)
from hipengine.kvcache import KVLiveSpans
from hipengine.loading import float_array_to_bf16_bits


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
        for q_head in range(num_q_heads):
            kv_head = q_head // kv_group
            scores = np.empty(context_len, dtype=np.float32)
            for token in range(context_len):
                scores[token] = float((query[row, q_head] * key[row, 0, token, kv_head]).sum() * scale)
            probs = np.exp(scores - scores.max())
            probs = probs / probs.sum()
            for token, prob in enumerate(probs):
                out[row, q_head] += prob * value[row, 0, token, kv_head]
    return out


def _primitive_correctness_passed(
    append_key_mismatch: int,
    append_value_mismatch: int,
    batch_vs_c1: float,
    batch_vs_numpy: float,
) -> bool:
    return (
        append_key_mismatch == 0
        and append_value_mismatch == 0
        and float(batch_vs_c1) == 0.0
        and math.isfinite(float(batch_vs_numpy))
        and 0.0 <= float(batch_vs_numpy) <= 2e-5
    )


def run(rows: int, *, seed: int = 1234) -> dict[str, object]:
    if rows <= 0:
        raise ValueError("rows must be positive")
    rng = np.random.default_rng(seed)
    block_size = 256
    blocks = 1
    max_context_len = 4
    num_kv_heads = 1
    num_q_heads = 4
    head_dim = 8
    scale = 1.0 / np.sqrt(head_dim)
    context_lens = np.asarray([(idx % max_context_len) + 1 for idx in range(rows)], dtype=np.int64)
    positions = context_lens - 1
    block_table = np.zeros((rows, blocks), dtype=np.int32)

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
        bkc = arena.dev(batch_key_cache)
        bvc = arena.dev(batch_value_cache)
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
        copy_device_to_host(host_array_ptr(c1_key_cache), ckc, runtime=arena.runtime)
        copy_device_to_host(host_array_ptr(c1_value_cache), cvc, runtime=arena.runtime)
        append_key_mismatch = int(np.count_nonzero(batch_key_cache != c1_key_cache))
        append_value_mismatch = int(np.count_nonzero(batch_value_cache != c1_value_cache))

        # Attention smoke: batch context decode should match independent c1 decode and NumPy oracle.
        key_cache_f32 = rng.normal(0.0, 0.25, size=(rows, blocks, block_size, num_kv_heads, head_dim)).astype(np.float32)
        value_cache_f32 = rng.normal(0.0, 0.25, size=(rows, blocks, block_size, num_kv_heads, head_dim)).astype(np.float32)
        key_cache = np.zeros_like(batch_key_cache)
        value_cache = np.zeros_like(batch_value_cache)
        for row, context_len in enumerate(context_lens):
            key_cache[row, 0, :context_len] = float_array_to_bf16_bits(key_cache_f32[row, 0, :context_len])
            value_cache[row, 0, :context_len] = float_array_to_bf16_bits(value_cache_f32[row, 0, :context_len])
        query = rng.normal(0.0, 0.25, size=(rows, num_q_heads, head_dim)).astype(np.float32)
        batch_out = np.zeros((rows, num_q_heads, head_dim), dtype=np.float32)
        c1_out = np.zeros_like(batch_out)
        query_b = arena.dev(query)
        key_cache_b = arena.dev(key_cache)
        value_cache_b = arena.dev(value_cache)
        live_b = arena.dev(context_lens)
        batch_out_b = arena.dev(batch_out)
        c1_out_b = arena.dev(c1_out)
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
        copy_device_to_host(host_array_ptr(batch_out), batch_out_b, runtime=arena.runtime)
        copy_device_to_host(host_array_ptr(c1_out), c1_out_b, runtime=arena.runtime)
        expected = _numpy_attention(query, key_cache, value_cache, context_lens, scale=scale)
    finally:
        arena.close()

    batch_vs_c1 = float(np.max(np.abs(batch_out - c1_out)))
    batch_vs_numpy = float(np.max(np.abs(batch_out - expected)))
    result = {
        "schema": 1,
        "rows": rows,
        "seed": seed,
        "block_size": block_size,
        "max_context_len": max_context_len,
        "num_q_heads": num_q_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "context_lens": context_lens.tolist(),
        "append_key_mismatch": append_key_mismatch,
        "append_value_mismatch": append_value_mismatch,
        "attn_batch_vs_c1_max_abs": batch_vs_c1,
        "attn_batch_vs_numpy_max_abs": batch_vs_numpy,
        "passed": _primitive_correctness_passed(
            append_key_mismatch,
            append_value_mismatch,
            batch_vs_c1,
            batch_vs_numpy,
        ),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    result = run(args.rows, seed=args.seed)
    if args.json is not None:
        result["artifact_path"] = str(args.json)
    payload = json.dumps(result, indent=2)
    print(payload)
    if args.json is not None:
        args.json.write_text(payload + "\n")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
