#!/usr/bin/env python3
"""Time strided versus copied head-major BF16 K/V through AOTriton.

This is a bounded attention-subwindow companion to the full GGUF prefill wall
benchmark. It uses the production Qwen3.6 GQA shape, the registry-owned paged
copy, and the same AOTriton v3 compact-varlen ABI as the resident runner.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import (
    DeviceBuffer,
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.core.tensor import Tensor
from hipengine.kernels.hip_gfx1100.attention import (
    aotriton_attn_fwd_v3_compact_varlen,
    build_aotriton_wrap,
    build_qwen35_paged_kv_write,
    qwen35_copy_paged_kv_bf16_to_head_major_dense_prefix_spans,
)
from hipengine.kernels.hip_gfx1100.attention.aotriton_wrap import (
    tensor1,
    tensor2,
    tensor4,
)
from hipengine.kvcache import KVLiveSpans


def _bf16_bits(values: np.ndarray) -> np.ndarray:
    source = np.ascontiguousarray(values, dtype=np.float32)
    words = source.view(np.uint32)
    rounded = words + np.uint32(0x7FFF) + ((words >> np.uint32(16)) & np.uint32(1))
    return np.ascontiguousarray((rounded >> np.uint32(16)).astype(np.uint16))


def _tensor(ptr: int, shape: tuple[int, ...], dtype: DType) -> Tensor:
    return Tensor.from_handle(ptr, shape, dtype, Device("hip", 0))


def _time_ms(runtime, launch, *, warmups: int, repetitions: int) -> list[float]:
    for _ in range(warmups):
        launch()
    runtime.device_synchronize()
    start = runtime.event_create()
    stop = runtime.event_create()
    values: list[float] = []
    try:
        for _ in range(repetitions):
            runtime.event_record(start)
            launch()
            runtime.event_record(stop)
            runtime.event_synchronize(stop)
            values.append(runtime.event_elapsed_time_ms(start, stop))
    finally:
        runtime.event_destroy(stop)
        runtime.event_destroy(start)
    return values


def _stats(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "median_ms": statistics.median(values),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def _screen_context(
    context_len: int,
    *,
    query_rows: int,
    warmups: int,
    repetitions: int,
    seed: int,
    runtime,
    copy_library,
    aotriton_library,
) -> dict[str, object]:
    q_heads, kv_heads, head_dim = 16, 2, 256
    block_size = 256
    blocks = (context_len + block_size - 1) // block_size
    rng = np.random.default_rng(seed + context_len)
    key = _bf16_bits(rng.normal(0.0, 0.25, size=(context_len, kv_heads, head_dim)))
    value = _bf16_bits(rng.normal(0.0, 0.25, size=(context_len, kv_heads, head_dim)))
    query = _bf16_bits(rng.normal(0.0, 0.25, size=(query_rows, q_heads, head_dim)))
    block_table = np.arange(blocks, dtype=np.int32)
    live_count = np.asarray([context_len], dtype=np.int64)
    cu_q = np.asarray([0, query_rows], dtype=np.int32)
    cu_k = np.asarray([0, context_len], dtype=np.int32)
    zero = np.asarray([0], dtype=np.int32)
    output_shape = (query_rows, q_heads, head_dim)
    output_nbytes = int(np.prod(output_shape)) * DType.BF16.itemsize
    lse_nbytes = q_heads * query_rows * DType.FP32.itemsize
    buffers: list[DeviceBuffer] = []

    def upload(array: np.ndarray) -> DeviceBuffer:
        contiguous = np.ascontiguousarray(array)
        buffer = malloc(contiguous.nbytes, runtime=runtime)
        buffers.append(buffer)
        copy_host_to_device(buffer, host_array_ptr(contiguous), runtime=runtime)
        return buffer

    def allocate(nbytes: int) -> DeviceBuffer:
        buffer = malloc(nbytes, runtime=runtime)
        buffers.append(buffer)
        return buffer

    try:
        key_buffer = upload(key)
        value_buffer = upload(value)
        query_buffer = upload(query)
        block_table_buffer = upload(block_table)
        live_count_buffer = upload(live_count)
        cu_q_buffer = upload(cu_q)
        cu_k_buffer = upload(cu_k)
        atomic_buffer = upload(zero)
        key_head_major = allocate(key.nbytes)
        value_head_major = allocate(value.nbytes)
        output_strided = allocate(output_nbytes)
        output_head_major = allocate(output_nbytes)
        lse_strided = allocate(lse_nbytes)
        lse_head_major = allocate(lse_nbytes)
        spans = KVLiveSpans.paged_uniform(
            block_table=_tensor(block_table_buffer.ptr, block_table.shape, DType.INT32),
            live_counts=_tensor(live_count_buffer.ptr, live_count.shape, DType.INT64),
            max_live_count=context_len,
            storage_dtype=DType.BF16,
            span_role="prefill",
        )
        query_tensor = tensor4(
            query_buffer.ptr,
            (1, q_heads, query_rows, head_dim),
            (q_heads * head_dim * query_rows, head_dim, q_heads * head_dim, 1),
            DType.BF16,
        )
        strided_k = tensor4(
            key_buffer.ptr,
            (1, kv_heads, context_len, head_dim),
            (kv_heads * head_dim * context_len, head_dim, kv_heads * head_dim, 1),
            DType.BF16,
        )
        strided_v = tensor4(
            value_buffer.ptr,
            (1, kv_heads, context_len, head_dim),
            (kv_heads * head_dim * context_len, head_dim, kv_heads * head_dim, 1),
            DType.BF16,
        )
        head_major_k = tensor4(
            key_head_major.ptr,
            (1, kv_heads, context_len, head_dim),
            (kv_heads * head_dim * context_len, head_dim * context_len, head_dim, 1),
            DType.BF16,
        )
        head_major_v = tensor4(
            value_head_major.ptr,
            (1, kv_heads, context_len, head_dim),
            (kv_heads * head_dim * context_len, head_dim * context_len, head_dim, 1),
            DType.BF16,
        )
        cu_q_tensor = tensor1(cu_q_buffer.ptr, (2,), (1,), DType.INT32)
        cu_k_tensor = tensor1(cu_k_buffer.ptr, (2,), (1,), DType.INT32)
        lse_strided_tensor = tensor2(
            lse_strided.ptr,
            (q_heads, query_rows),
            (query_rows, 1),
            DType.FP32,
        )
        lse_head_major_tensor = tensor2(
            lse_head_major.ptr,
            (q_heads, query_rows),
            (query_rows, 1),
            DType.FP32,
        )
        output_strided_tensor = tensor4(
            output_strided.ptr,
            (1, q_heads, query_rows, head_dim),
            (q_heads * head_dim * query_rows, head_dim, q_heads * head_dim, 1),
            DType.BF16,
        )
        output_head_major_tensor = tensor4(
            output_head_major.ptr,
            (1, q_heads, query_rows, head_dim),
            (q_heads * head_dim * query_rows, head_dim, q_heads * head_dim, 1),
            DType.BF16,
        )

        def copy_layout() -> None:
            qwen35_copy_paged_kv_bf16_to_head_major_dense_prefix_spans(
                key_buffer.ptr,
                value_buffer.ptr,
                key_head_major.ptr,
                value_head_major.ptr,
                spans,
                context_len,
                context_len,
                block_size,
                kv_heads,
                head_dim,
                library=copy_library,
                runtime=runtime,
            )

        def attention(k_tensor, v_tensor, lse_tensor, output_tensor) -> None:
            runtime.memset(atomic_buffer.ptr, 0, atomic_buffer.nbytes)
            aotriton_attn_fwd_v3_compact_varlen(
                query_tensor,
                k_tensor,
                v_tensor,
                cu_q_tensor,
                cu_k_tensor,
                lse_tensor,
                output_tensor,
                persistent_atomic_counter_ptr=atomic_buffer.ptr,
                max_seqlen_q=query_rows,
                max_seqlen_k=context_len,
                sm_scale=head_dim**-0.5,
                is_causal=True,
                library=aotriton_library,
                runtime=runtime,
            )

        copy_layout()
        attention(strided_k, strided_v, lse_strided_tensor, output_strided_tensor)
        attention(head_major_k, head_major_v, lse_head_major_tensor, output_head_major_tensor)
        runtime.device_synchronize()
        strided_host = np.empty(output_shape, dtype=np.uint16)
        head_major_host = np.empty(output_shape, dtype=np.uint16)
        copy_device_to_host(host_array_ptr(strided_host), output_strided, runtime=runtime)
        copy_device_to_host(host_array_ptr(head_major_host), output_head_major, runtime=runtime)
        mismatches = int(np.count_nonzero(strided_host != head_major_host))

        copy_times = _time_ms(
            runtime,
            copy_layout,
            warmups=warmups,
            repetitions=repetitions,
        )
        strided_times = _time_ms(
            runtime,
            lambda: attention(strided_k, strided_v, lse_strided_tensor, output_strided_tensor),
            warmups=warmups,
            repetitions=repetitions,
        )
        head_major_times = _time_ms(
            runtime,
            lambda: attention(
                head_major_k,
                head_major_v,
                lse_head_major_tensor,
                output_head_major_tensor,
            ),
            warmups=warmups,
            repetitions=repetitions,
        )
        copy_inclusive_times = _time_ms(
            runtime,
            lambda: (
                copy_layout(),
                attention(
                    head_major_k,
                    head_major_v,
                    lse_head_major_tensor,
                    output_head_major_tensor,
                ),
            ),
            warmups=warmups,
            repetitions=repetitions,
        )
        strided_median = statistics.median(strided_times)
        head_major_median = statistics.median(head_major_times)
        inclusive_median = statistics.median(copy_inclusive_times)
        return {
            "context_len": context_len,
            "query_rows": query_rows,
            "copy_bytes": key.nbytes + value.nbytes,
            "output_bit_mismatches": mismatches,
            "strided_attention": _stats(strided_times),
            "head_major_attention": _stats(head_major_times),
            "head_major_copy": _stats(copy_times),
            "head_major_copy_inclusive_attention": _stats(copy_inclusive_times),
            "attention_speedup": strided_median / head_major_median,
            "copy_inclusive_speedup": strided_median / inclusive_median,
        }
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contexts", default="512,4096,32768,65536")
    parser.add_argument("--query-rows", type=int, default=4096)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--compiler-version-file", type=Path, required=True)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()
    contexts = tuple(int(value) for value in args.contexts.split(",") if value.strip())
    if not contexts or any(value <= 0 for value in contexts):
        raise ValueError("contexts must contain positive integers")
    if args.query_rows <= 0 or args.warmups < 0 or args.repetitions <= 0:
        raise ValueError("query_rows/repetitions must be positive and warmups non-negative")
    version = args.compiler_version_file.read_text(encoding="utf-8").strip()
    copy_library = build_qwen35_paged_kv_write(
        load=True,
        compiler_version=version,
        require_cached=args.require_cached_build,
    )
    aotriton_library = build_aotriton_wrap(
        load=True,
        compiler_version=version,
        require_cached=args.require_cached_build,
    )
    runtime = get_hip_runtime()
    rows = [
        _screen_context(
            context,
            query_rows=min(context, args.query_rows),
            warmups=args.warmups,
            repetitions=args.repetitions,
            seed=args.seed,
            runtime=runtime,
            copy_library=copy_library,
            aotriton_library=aotriton_library,
        )
        for context in contexts
    ]
    payload = {
        "schema": 1,
        "status": "diagnostic",
        "mode": "qwen35_aotriton_head_major_bf16_screen",
        "shape": {"q_heads": 16, "kv_heads": 2, "head_dim": 256},
        "warmups": args.warmups,
        "repetitions": args.repetitions,
        "passed": all(int(row["output_bit_mismatches"]) == 0 for row in rows),
        "rows": rows,
    }
    text = json.dumps(payload, indent=2)
    print(text)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n", encoding="utf-8")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
