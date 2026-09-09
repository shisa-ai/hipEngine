#!/usr/bin/env python3
"""Cached-only rocprofv3 child for the two-pass INT8 prefill kernel.

Builds a synthetic 8,192-token INT8 paged KV store with fp32 per-token/head
scales, then launches the two-pass tiled prefill kernel once per
full-attention layer (16) at the worst-case chunk (rows 7168..8191 attend
over all 8,192 tokens). Run under ``rocprofv3 --kernel-trace`` to read the
per-launch duration of ``qwen35_paged_full_attn_prefill_gqa_gate_int8_twopass_kernel``.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import copy_host_to_device, free, host_array_ptr, malloc
from hipengine.core.tensor import Tensor
from hipengine.kernels.hip_gfx1100.attention import (
    build_qwen35_paged_attn_decode,
    qwen35_paged_attn_prefill_int8_gqa_gate_bf16_out_flash_spans,
    qwen35_paged_attn_prefill_int8_gqa_gate_bf16_out_twopass_spans,
)
from hipengine.kvcache import KVLiveSpans, KVScaleMetadata

import os as _os
BLOCKS = int(_os.environ.get("PROBE_BLOCKS", "32"))
BLOCK_SIZE = 256
TOKENS = BLOCKS * BLOCK_SIZE
ROWS = int(_os.environ.get("PROBE_ROWS", "1024"))
NUM_Q_HEADS = 24
NUM_KV_HEADS = 4
HEAD_DIM = 256
LAYERS = 16
START = TOKENS - ROWS


def main() -> int:
    runtime = get_hip_runtime()
    lib_path = _os.environ.get("PROBE_LIB")
    if lib_path:
        import ctypes as _ctypes
        library = _ctypes.CDLL(lib_path)
    else:
        library = build_qwen35_paged_attn_decode(load=True)
    rng = np.random.default_rng(0x5EED)
    query = (rng.standard_normal((ROWS, NUM_Q_HEADS, HEAD_DIM)) * 0.15).astype(np.float32)
    key_cache = rng.integers(-12, 13, size=(BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM), dtype=np.int8)
    value_cache = rng.integers(-10, 11, size=(BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM), dtype=np.int8)
    k_scale = rng.uniform(0.008, 0.025, size=(BLOCKS, BLOCK_SIZE, NUM_KV_HEADS)).astype(np.float32)
    v_scale = rng.uniform(0.006, 0.02, size=(BLOCKS, BLOCK_SIZE, NUM_KV_HEADS)).astype(np.float32)
    gate = (rng.standard_normal((ROWS, NUM_Q_HEADS, HEAD_DIM)) * 0.25).astype(np.float16)
    block_table = np.tile(np.arange(BLOCKS, dtype=np.int32), (ROWS, 1))
    context_counts = np.full((ROWS,), TOKENS, dtype=np.int64)
    row_positions = np.arange(START, START + ROWS, dtype=np.int64)
    out = np.zeros((ROWS, NUM_Q_HEADS * HEAD_DIM), dtype=np.uint16)

    bufs = []

    def up(arr):
        buf = malloc(max(int(arr.nbytes), 4), runtime=runtime)
        bufs.append(buf)
        copy_host_to_device(buf, host_array_ptr(arr), arr.nbytes, runtime=runtime)
        return buf

    query_dev = up(query)
    key_dev = up(key_cache)
    value_dev = up(value_cache)
    k_scale_dev = up(k_scale)
    v_scale_dev = up(v_scale)
    gate_dev = up(gate)
    table_dev = up(block_table)
    counts_dev = up(context_counts)
    positions_dev = up(row_positions)
    out_dev = malloc(out.nbytes, runtime=runtime)
    bufs.append(out_dev)

    device = Device("hip", 0)
    metadata = KVScaleMetadata(
        k_scale=Tensor.from_handle(k_scale_dev.ptr, k_scale.shape, DType.FP32, device),
        v_scale=Tensor.from_handle(v_scale_dev.ptr, v_scale.shape, DType.FP32, device),
        scale_dtype=DType.FP32,
    )
    spans = KVLiveSpans.paged_uniform(
        block_table=Tensor.from_handle(table_dev.ptr, block_table.shape, DType.INT32, device),
        live_counts=Tensor.from_handle(counts_dev.ptr, context_counts.shape, DType.INT64, device),
        max_live_count=TOKENS,
        storage_dtype=DType.INT8_PER_TOKEN_HEAD,
        row_positions=Tensor.from_handle(positions_dev.ptr, row_positions.shape, DType.INT64, device),
        span_role="prefill",
        scale_metadata=metadata,
    )

    runtime.device_synchronize()
    for kernel_name, launcher in (
        ("flash", qwen35_paged_attn_prefill_int8_gqa_gate_bf16_out_flash_spans),
    ):
        # GPU-side per-launch timing via HIP events
        events = []
        for i in range(LAYERS + 1):
            events.append(runtime.event_create())
        started = time.perf_counter()
        for i in range(LAYERS):
            runtime.event_record(events[i])
            launcher(
                query_dev.ptr,
                key_dev.ptr,
                value_dev.ptr,
                k_scale_dev.ptr,
                v_scale_dev.ptr,
                gate_dev.ptr,
                out_dev.ptr,
                spans,
                ROWS,
                TOKENS,
                BLOCK_SIZE,
                NUM_Q_HEADS,
                NUM_KV_HEADS,
                HEAD_DIM,
                HEAD_DIM,
                1,
                HEAD_DIM ** -0.5,
                stream=0,
                library=library,
                runtime=runtime,
            )
            runtime.event_record(events[i + 1])
        runtime.device_synchronize()
        elapsed = time.perf_counter() - started
        gpu_times = []
        for i in range(LAYERS):
            gpu_times.append(runtime.event_elapsed_time_ms(events[i], events[i + 1]))
        print(
            f"{kernel_name}: wall {elapsed*1000:.1f} ms total ({elapsed*1000/LAYERS:.2f} ms/launch), "
            f"gpu sum {sum(gpu_times):.1f} ms ({sum(gpu_times)/LAYERS:.2f} ms/launch), "
            f"gpu per-launch min/max {min(gpu_times):.2f}/{max(gpu_times):.2f} ms"
        )
    runtime.device_synchronize()
    for buf in reversed(bufs):
        free(buf, runtime=runtime)
    print(f"launched {LAYERS} two-pass prefill kernels at {TOKENS} tokens")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
