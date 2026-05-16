#!/usr/bin/env python3
"""Synthetic HIP correctness smoke for GGUF Q8_0/Q5_K/Q6_K GEMV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.cpu_reference import gguf_q5_k_gemv, gguf_q6_k_gemv, gguf_q8_0_gemv
from hipengine.kernels.hip_gfx1100.quant.gguf_k_gemv import (
    build_gguf_k_gemv,
    gguf_q5_k_gemv_bf16_bf16_out,
    gguf_q5_k_gemv_bf16_f32_out,
    gguf_q5_k_gemv_f32_f32_out,
    gguf_q5_k_gemv_fp16_f32_out,
    gguf_q6_k_gemv_bf16_bf16_out,
    gguf_q6_k_gemv_bf16_f32_out,
    gguf_q6_k_gemv_f32_f32_out,
    gguf_q6_k_gemv_fp16_f32_out,
    gguf_q8_0_gemv_bf16_bf16_out,
    gguf_q8_0_gemv_bf16_f32_out,
    gguf_q8_0_gemv_f32_f32_out,
    gguf_q8_0_gemv_fp16_f32_out,
)
from hipengine.loading.materialize import float_array_to_bf16_bits
from hipengine.quant.gguf import bf16_to_float32

QK_K = 256
Q8_0_BLOCK_BYTES = 34
Q5_K_BLOCK_BYTES = 176
Q6_K_BLOCK_BYTES = 210


def make_q8_0_weight(out_features: int, in_features: int) -> np.ndarray:
    if in_features % 32:
        raise ValueError("in_features must be a multiple of 32")
    blocks_per_row = in_features // 32
    data = np.empty((out_features, blocks_per_row * Q8_0_BLOCK_BYTES), dtype=np.uint8)
    for out_idx in range(out_features):
        for block_idx in range(blocks_per_row):
            start = block_idx * Q8_0_BLOCK_BYTES
            data[out_idx, start : start + Q8_0_BLOCK_BYTES] = _make_q8_0_block(
                out_idx, block_idx
            )
    return data


def make_q5_k_weight(out_features: int, in_features: int) -> np.ndarray:
    return _make_k_weight(out_features, in_features, Q5_K_BLOCK_BYTES, _make_q5_k_block)


def make_q6_k_weight(out_features: int, in_features: int) -> np.ndarray:
    return _make_k_weight(out_features, in_features, Q6_K_BLOCK_BYTES, _make_q6_k_block)


def _make_k_weight(out_features: int, in_features: int, block_bytes: int, make_block) -> np.ndarray:
    if in_features % QK_K:
        raise ValueError("in_features must be a multiple of 256")
    blocks_per_row = in_features // QK_K
    data = np.empty((out_features, blocks_per_row * block_bytes), dtype=np.uint8)
    for out_idx in range(out_features):
        for block_idx in range(blocks_per_row):
            start = block_idx * block_bytes
            data[out_idx, start : start + block_bytes] = make_block(out_idx, block_idx)
    return data


def _make_q8_0_block(out_idx: int, block_idx: int) -> np.ndarray:
    d = np.float16(0.03125 * (1 + (out_idx % 5)))
    q = ((np.arange(32, dtype=np.int16) + out_idx * 7 + block_idx * 3) % 31 - 15).astype(
        np.int8
    )
    return np.concatenate([np.asarray([d], dtype=np.float16).view(np.uint8), q.view(np.uint8)])


def _make_q5_k_block(out_idx: int, block_idx: int) -> np.ndarray:
    d = np.float16(0.015625 * (1 + (out_idx % 5)))
    dmin = np.float16(0.0078125 * (1 + (block_idx % 3)))
    scales = ((np.arange(8, dtype=np.uint8) * 3 + out_idx + block_idx) % 63 + 1).astype(
        np.uint8
    )
    mins = ((np.arange(8, dtype=np.uint8) * 5 + 2 * out_idx + block_idx) % 17).astype(
        np.uint8
    )
    q = ((np.arange(QK_K, dtype=np.uint16) + out_idx * 7 + block_idx * 11) % 32).astype(
        np.uint8
    )
    q_groups = q.reshape(8, 32)
    qh = np.zeros(32, dtype=np.uint8)
    for subblock in range(8):
        qh |= ((q_groups[subblock] >> np.uint8(4)) & np.uint8(1)) << np.uint8(subblock)
    low = q_groups & np.uint8(0x0F)
    qs = np.empty(128, dtype=np.uint8)
    for pair in range(4):
        qs[pair * 32 : (pair + 1) * 32] = low[2 * pair] | (low[2 * pair + 1] << 4)
    return np.concatenate(
        [
            np.asarray([d], dtype=np.float16).view(np.uint8),
            np.asarray([dmin], dtype=np.float16).view(np.uint8),
            _pack_q4_k_scales(scales, mins),
            qh,
            qs,
        ]
    )


def _make_q6_k_block(out_idx: int, block_idx: int) -> np.ndarray:
    d = np.float16(0.0107421875 * (1 + (out_idx % 3)))
    scales = ((np.arange(16, dtype=np.int16) * 3 + out_idx - block_idx) % 31 - 15).astype(
        np.int8
    )
    q_signed = (np.arange(QK_K, dtype=np.int16) + out_idx * 5 + block_idx * 9) % 64 - 32
    q = (q_signed + 32).astype(np.uint8)
    ql = np.zeros(128, dtype=np.uint8)
    qh = np.zeros(64, dtype=np.uint8)
    for k, value in enumerate(q):
        group32 = k >> 5
        lane = k & 31
        base64 = 64 if group32 >= 4 else 0
        ql_idx = base64 + (group32 & 1) * 32 + lane
        if (group32 & 2) == 0:
            ql[ql_idx] |= value & np.uint8(0x0F)
        else:
            ql[ql_idx] |= (value & np.uint8(0x0F)) << np.uint8(4)
        qh_idx = (32 if group32 >= 4 else 0) + lane
        qh[qh_idx] |= ((value >> np.uint8(4)) & np.uint8(0x03)) << np.uint8(
            2 * (group32 & 3)
        )
    return np.concatenate(
        [ql, qh, scales.view(np.uint8), np.asarray([d], dtype=np.float16).view(np.uint8)]
    )


def _pack_q4_k_scales(scales: np.ndarray, mins: np.ndarray) -> np.ndarray:
    out = np.zeros(12, dtype=np.uint8)
    out[:4] = (scales[:4] & 0x3F) | ((scales[4:] & 0x30) << 2)
    out[4:8] = (mins[:4] & 0x3F) | ((mins[4:] & 0x30) << 2)
    out[8:12] = (scales[4:] & 0x0F) | ((mins[4:] & 0x0F) << 4)
    return out


def run_case(
    name: str,
    make_weight: Callable[[int, int], np.ndarray],
    reference: Callable[[np.ndarray, np.ndarray], np.ndarray],
    launches: dict[str, Callable],
    rows: int,
    in_features: int,
    out_features: int,
    library,
) -> dict[str, float]:
    runtime = get_hip_runtime()
    qweight = make_weight(out_features, in_features)
    x_f32 = (
        (np.arange(rows * in_features, dtype=np.float32).reshape(rows, in_features) % 19) - 9
    ) / 16.0
    x_fp16 = x_f32.astype(np.float16)
    x_bf16 = float_array_to_bf16_bits(x_f32)
    inputs = {
        "f32": (x_f32, x_f32, launches["f32"], np.float32),
        "fp16": (x_fp16, x_fp16.astype(np.float32), launches["fp16"], np.float32),
        "bf16": (x_bf16, bf16_to_float32(x_bf16), launches["bf16"], np.float32),
        "bf16_out": (x_bf16, bf16_to_float32(x_bf16), launches["bf16_out"], np.uint16),
    }
    results: dict[str, float] = {}
    for mode, (host_x, ref_x, launch, out_dtype) in inputs.items():
        expected = reference(ref_x, qweight)
        out = np.empty((rows, out_features), dtype=out_dtype)
        buffers = []
        try:
            x_dev = malloc(host_x.nbytes, runtime=runtime)
            buffers.append(x_dev)
            q_dev = malloc(qweight.nbytes, runtime=runtime)
            buffers.append(q_dev)
            out_dev = malloc(out.nbytes, runtime=runtime)
            buffers.append(out_dev)
            copy_host_to_device(
                x_dev, host_array_ptr(np.ascontiguousarray(host_x)), runtime=runtime
            )
            copy_host_to_device(
                q_dev, host_array_ptr(np.ascontiguousarray(qweight)), runtime=runtime
            )
            launch(
                x_dev.ptr,
                q_dev.ptr,
                out_dev.ptr,
                rows,
                in_features,
                out_features,
                threads=128,
                library=library,
                runtime=runtime,
            )
            runtime.device_synchronize()
            copy_device_to_host(host_array_ptr(out), out_dev, runtime=runtime)
        finally:
            for buffer in reversed(buffers):
                free(buffer, runtime=runtime)
        if out_dtype == np.uint16:
            expected_bits = float_array_to_bf16_bits(expected)
            max_abs = float(np.max(np.abs(bf16_to_float32(out) - bf16_to_float32(expected_bits))))
            bit_mismatch = int(np.count_nonzero(out != expected_bits))
            results[mode] = max_abs
            print(f"{name} {mode}_max_abs={max_abs} {mode}_bit_mismatch={bit_mismatch}")
            if bit_mismatch:
                results[mode] = float("inf")
        else:
            max_abs = float(np.max(np.abs(out - expected)))
            results[mode] = max_abs
            print(f"{name} {mode}_max_abs={max_abs}")
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=2)
    parser.add_argument("--out-features", type=int, default=7)
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--require-cached-build", action="store_true")
    args = parser.parse_args(argv)

    compiler_version = None
    if args.compiler_version_file is not None:
        compiler_version = args.compiler_version_file.read_text()
    library = build_gguf_k_gemv(
        load=True,
        compiler_version=compiler_version,
        require_cached=args.require_cached_build,
    )
    cases = [
        (
            "Q8_0",
            make_q8_0_weight,
            gguf_q8_0_gemv,
            {
                "f32": gguf_q8_0_gemv_f32_f32_out,
                "fp16": gguf_q8_0_gemv_fp16_f32_out,
                "bf16": gguf_q8_0_gemv_bf16_f32_out,
                "bf16_out": gguf_q8_0_gemv_bf16_bf16_out,
            },
            64,
        ),
        (
            "Q5_K",
            make_q5_k_weight,
            gguf_q5_k_gemv,
            {
                "f32": gguf_q5_k_gemv_f32_f32_out,
                "fp16": gguf_q5_k_gemv_fp16_f32_out,
                "bf16": gguf_q5_k_gemv_bf16_f32_out,
                "bf16_out": gguf_q5_k_gemv_bf16_bf16_out,
            },
            512,
        ),
        (
            "Q6_K",
            make_q6_k_weight,
            gguf_q6_k_gemv,
            {
                "f32": gguf_q6_k_gemv_f32_f32_out,
                "fp16": gguf_q6_k_gemv_fp16_f32_out,
                "bf16": gguf_q6_k_gemv_bf16_f32_out,
                "bf16_out": gguf_q6_k_gemv_bf16_bf16_out,
            },
            512,
        ),
    ]
    all_results = []
    for name, make_weight, reference, launches, in_features in cases:
        all_results.append(
            run_case(
                name,
                make_weight,
                reference,
                launches,
                args.rows,
                in_features,
                args.out_features,
                library,
            )
        )
    worst = max(value for result in all_results for value in result.values())
    print(f"worst_max_abs={worst}")
    return 0 if worst <= 1e-4 else 1


if __name__ == "__main__":
    raise SystemExit(main())
