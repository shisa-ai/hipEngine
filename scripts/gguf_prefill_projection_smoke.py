#!/usr/bin/env python3
"""Rows>1 GGUF projection smoke against CPU references."""

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
from hipengine.core.memory import copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc
from hipengine.kernels.cpu_reference import (
    gguf_q4_k_pack8_gemv,
    gguf_q5_k_gemv,
    gguf_q6_k_gemv,
    gguf_q8_0_gemv,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_k_gemv import (
    build_gguf_k_gemv,
    gguf_q5_k_prefill_bf16_bf16_out,
    gguf_q5_k_prefill_bf16_f32_out,
    gguf_q5_k_prefill_bf16_fp16_out,
    gguf_q6_k_prefill_bf16_bf16_out,
    gguf_q6_k_prefill_bf16_f32_out,
    gguf_q6_k_prefill_bf16_fp16_out,
    gguf_q8_0_prefill_bf16_bf16_out,
    gguf_q8_0_prefill_bf16_f32_out,
    gguf_q8_0_prefill_bf16_fp16_out,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_gemv import (
    build_gguf_q4_k_gemv,
    gguf_q4_k_pack8_prefill_bf16_bf16_out,
    gguf_q4_k_pack8_prefill_bf16_f32_out,
    gguf_q4_k_pack8_prefill_bf16_fp16_out,
)
from hipengine.loading.materialize import float_array_to_bf16_bits
from hipengine.quant.gguf import bf16_to_float32
from hipengine.quant.gguf_q4_k import repack_gguf_q4_k_pack8
from scripts.gguf_k_gemv_smoke import make_q5_k_weight, make_q6_k_weight, make_q8_0_weight
from scripts.smoke import _make_smoke_q4_k_weight


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--require-cached-build", action="store_true")
    args = parser.parse_args(argv)
    if args.rows <= 1:
        raise ValueError("--rows must be > 1 for prefill smoke")
    compiler_version = args.compiler_version_file.read_text() if args.compiler_version_file else None
    q4_lib = build_gguf_q4_k_gemv(
        load=True,
        compiler_version=compiler_version,
        require_cached=args.require_cached_build,
    )
    raw_lib = build_gguf_k_gemv(
        load=True,
        compiler_version=compiler_version,
        require_cached=args.require_cached_build,
    )
    results: list[float] = []
    results.extend(_run_q4_pack8(args.rows, q4_lib))
    raw_cases = [
        (
            "Q8_0",
            make_q8_0_weight,
            gguf_q8_0_gemv,
            gguf_q8_0_prefill_bf16_f32_out,
            gguf_q8_0_prefill_bf16_fp16_out,
            gguf_q8_0_prefill_bf16_bf16_out,
            64,
            raw_lib,
        ),
        (
            "Q5_K",
            make_q5_k_weight,
            gguf_q5_k_gemv,
            gguf_q5_k_prefill_bf16_f32_out,
            gguf_q5_k_prefill_bf16_fp16_out,
            gguf_q5_k_prefill_bf16_bf16_out,
            512,
            raw_lib,
        ),
        (
            "Q6_K",
            make_q6_k_weight,
            gguf_q6_k_gemv,
            gguf_q6_k_prefill_bf16_f32_out,
            gguf_q6_k_prefill_bf16_fp16_out,
            gguf_q6_k_prefill_bf16_bf16_out,
            512,
            raw_lib,
        ),
    ]
    for case in raw_cases:
        results.extend(_run_raw_case(args.rows, *case))
    worst = max(results)
    print(f"worst_max_abs={worst}")
    return 0 if worst <= 1e-4 else 1


def _x_bf16(rows: int, in_features: int) -> tuple[np.ndarray, np.ndarray]:
    x_f32 = ((np.arange(rows * in_features, dtype=np.float32).reshape(rows, in_features) % 19) - 9) / 16.0
    x_bf16 = float_array_to_bf16_bits(x_f32)
    return x_bf16, bf16_to_float32(x_bf16)


def _run_q4_pack8(rows: int, library) -> list[float]:
    in_features = 512
    out_features = 8
    packed = repack_gguf_q4_k_pack8(_make_smoke_q4_k_weight(out_features, in_features))
    x_host, x_ref = _x_bf16(rows, in_features)
    expected = gguf_q4_k_pack8_gemv(x_ref, packed.qweight, packed.scales, packed.mins)
    return _run_outputs(
        "Q4_K_PACK8",
        rows,
        in_features,
        out_features,
        x_host,
        (packed.qweight, packed.scales, packed.mins),
        expected,
        {
            "f32": lambda x, q, out: gguf_q4_k_pack8_prefill_bf16_f32_out(
                x, q[0], q[1], q[2], out, rows, in_features, out_features, library=library
            ),
            "fp16": lambda x, q, out: gguf_q4_k_pack8_prefill_bf16_fp16_out(
                x, q[0], q[1], q[2], out, rows, in_features, out_features, library=library
            ),
            "bf16": lambda x, q, out: gguf_q4_k_pack8_prefill_bf16_bf16_out(
                x, q[0], q[1], q[2], out, rows, in_features, out_features, library=library
            ),
        },
    )


def _run_raw_case(
    rows: int,
    name: str,
    make_weight: Callable[[int, int], np.ndarray],
    reference: Callable[[np.ndarray, np.ndarray], np.ndarray],
    launch_f32: Callable,
    launch_fp16: Callable,
    launch_bf16: Callable,
    in_features: int,
    library,
) -> list[float]:
    out_features = 7
    qweight = make_weight(out_features, in_features)
    x_host, x_ref = _x_bf16(rows, in_features)
    expected = reference(x_ref, qweight)
    return _run_outputs(
        name,
        rows,
        in_features,
        out_features,
        x_host,
        (qweight,),
        expected,
        {
            "f32": lambda x, q, out: launch_f32(
                x, q[0], out, rows, in_features, out_features, threads=128, library=library
            ),
            "fp16": lambda x, q, out: launch_fp16(
                x, q[0], out, rows, in_features, out_features, threads=128, library=library
            ),
            "bf16": lambda x, q, out: launch_bf16(
                x, q[0], out, rows, in_features, out_features, threads=128, library=library
            ),
        },
    )


def _run_outputs(
    name: str,
    rows: int,
    in_features: int,
    out_features: int,
    x_host: np.ndarray,
    q_parts: tuple[np.ndarray, ...],
    expected: np.ndarray,
    launches: dict[str, Callable[[int, tuple[object, ...], int], None]],
) -> list[float]:
    runtime = get_hip_runtime()
    out_specs = {
        "f32": (np.float32, expected.astype(np.float32)),
        "fp16": (np.float16, expected.astype(np.float16).astype(np.float32)),
        "bf16": (np.uint16, bf16_to_float32(float_array_to_bf16_bits(expected))),
    }
    results: list[float] = []
    buffers = []
    try:
        x_dev = malloc(x_host.nbytes, runtime=runtime)
        buffers.append(x_dev)
        copy_host_to_device(x_dev, host_array_ptr(np.ascontiguousarray(x_host)), runtime=runtime)
        q_devs = []
        for part in q_parts:
            dev = malloc(part.nbytes, runtime=runtime)
            buffers.append(dev)
            q_devs.append(dev)
            copy_host_to_device(dev, host_array_ptr(np.ascontiguousarray(part)), runtime=runtime)
        for mode, launch in launches.items():
            dtype, expected_mode = out_specs[mode]
            out = np.empty((rows, out_features), dtype=dtype)
            out_dev = malloc(out.nbytes, runtime=runtime)
            buffers.append(out_dev)
            launch(x_dev.ptr, tuple(dev.ptr for dev in q_devs), out_dev.ptr)
            runtime.device_synchronize()
            copy_device_to_host(host_array_ptr(out), out_dev, runtime=runtime)
            if dtype == np.uint16:
                actual = bf16_to_float32(out)
                bit_mismatch = int(np.count_nonzero(out != float_array_to_bf16_bits(expected)))
            elif dtype == np.float16:
                actual = out.astype(np.float32)
                bit_mismatch = int(np.count_nonzero(out.view(np.uint16) != expected.astype(np.float16).view(np.uint16)))
            else:
                actual = out
                bit_mismatch = 0
            max_abs = float(np.max(np.abs(actual - expected_mode)))
            results.append(max_abs if bit_mismatch == 0 else float("inf"))
            print(f"{name} prefill_bf16_{mode}_out max_abs={max_abs} bit_mismatch={bit_mismatch}")
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)
    return results


if __name__ == "__main__":
    raise SystemExit(main())
