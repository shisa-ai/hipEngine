#!/usr/bin/env python3
"""Immutable SH-D1 Qwen Q5T16 selected-down tile8 leaf screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import sys
from pathlib import Path

import numpy as np

_ROWS = 8
_IN_FEATURES = 512
_OUT_FEATURES = 2048
_TOP_K = 8
_MALL_BYTES = 32 * 1024 * 1024


def _f32_to_bf16_bits(values: np.ndarray) -> np.ndarray:
    f32 = np.ascontiguousarray(values, dtype=np.float32)
    bits = f32.view(np.uint32)
    lsb = (bits >> 16) & 1
    return ((bits + 0x7FFF + lsb) >> 16).astype(np.uint16)


def _sha256(values: np.ndarray) -> str:
    return hashlib.sha256(memoryview(np.ascontiguousarray(values))).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compiler-version-file", type=Path, default=None)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--warmups", type=int, default=80)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--burst", type=int, default=400)
    parser.add_argument("--mall-bytes", type=int, default=_MALL_BYTES)
    parser.add_argument("--trace-only", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.compiler_version_file is not None:
        os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(
            args.compiler_version_file
        )
    if args.warmups < 0 or args.samples <= 0 or args.burst <= 0:
        raise ValueError("warmups must be nonnegative; samples and burst must be positive")

    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )
    from hipengine.kernels.hip_gfx1100.quant.gguf_t16_selected_gemv import (
        build_gguf_t16_selected_gemv,
        gguf_q5_k_t16_selected_gemv_bf16_bf16_out,
        gguf_q5_k_t16_selected_qwen_tile8_gemv_bf16_bf16_out,
    )
    from hipengine.quant.gguf_t16 import repack_gguf_q5_k_tile16
    from tests._gguf_synthetic_weights import make_q5_k_weight

    runtime = get_hip_runtime()
    library = build_gguf_t16_selected_gemv(
        load=True,
        require_cached=args.require_cached_build,
    )
    rng = np.random.default_rng(20260806)
    x = _f32_to_bf16_bits(
        rng.normal(0.0, 0.3, size=(_ROWS, _IN_FEATURES)).astype(np.float32)
    )

    quant_specs = {
        "q5": (
            make_q5_k_weight,
            repack_gguf_q5_k_tile16,
            gguf_q5_k_t16_selected_gemv_bf16_bf16_out,
            gguf_q5_k_t16_selected_qwen_tile8_gemv_bf16_bf16_out,
            37,
        ),
    }
    requested = ("q5",)
    results: dict[str, object] = {}

    for quant in requested:
        make_weight, repack, control_fn, candidate_fn, calls_per_token = quant_specs[
            quant
        ]
        raw_one = np.ascontiguousarray(
            make_weight(_OUT_FEATURES, _IN_FEATURES)[None, ...]
        )
        tiles_one = np.ascontiguousarray(repack(raw_one).tiles)
        selected_matrix_bytes = int(tiles_one.nbytes * _TOP_K)
        pool_groups = max(
            2,
            (2 * int(args.mall_bytes)) // max(selected_matrix_bytes, 1) + 1,
        )
        num_experts = pool_groups * _TOP_K
        tiles = np.ascontiguousarray(
            np.tile(tiles_one, (num_experts, 1, 1, 1))
        )
        selected = np.ascontiguousarray(
            np.arange(num_experts, dtype=np.int64).reshape(pool_groups, _TOP_K)
        )
        control_host = np.zeros((_ROWS, _OUT_FEATURES), dtype=np.uint16)
        candidate_host = np.zeros_like(control_host)
        buffers = []

        def upload(host: np.ndarray):
            buffer = malloc(host.nbytes, runtime=runtime)
            copy_host_to_device(
                buffer,
                host_array_ptr(host),
                host.nbytes,
                runtime=runtime,
            )
            buffers.append(buffer)
            return buffer

        start_event = runtime.event_create()
        stop_event = runtime.event_create()
        try:
            x_buffer = upload(x)
            selected_buffer = upload(selected)
            tiles_buffer = upload(tiles)
            control_buffer = malloc(control_host.nbytes, runtime=runtime)
            candidate_buffer = malloc(candidate_host.nbytes, runtime=runtime)
            buffers.extend((control_buffer, candidate_buffer))

            def launch(fn, out_ptr: int, index: int) -> None:
                group = index % pool_groups
                selected_ptr = selected_buffer.ptr + group * _TOP_K * 8
                fn(
                    x_buffer.ptr,
                    selected_ptr,
                    tiles_buffer.ptr,
                    out_ptr,
                    _ROWS,
                    _ROWS,
                    num_experts,
                    _IN_FEATURES,
                    _OUT_FEATURES,
                    library=library,
                    runtime=runtime,
                )

            launch(control_fn, control_buffer.ptr, 0)
            launch(candidate_fn, candidate_buffer.ptr, 0)
            runtime.device_synchronize()
            copy_device_to_host(
                host_array_ptr(control_host),
                control_buffer,
                control_host.nbytes,
                runtime=runtime,
            )
            copy_device_to_host(
                host_array_ptr(candidate_host),
                candidate_buffer,
                candidate_host.nbytes,
                runtime=runtime,
            )
            exact = bool(np.array_equal(control_host, candidate_host))
            if not exact:
                raise AssertionError(f"{quant} tile8 output differs from production")

            if args.trace_only:
                results[quant] = {
                    "trace_only": True,
                    "exact": exact,
                    "control_sha256": _sha256(control_host),
                    "candidate_sha256": _sha256(candidate_host),
                }
                continue

            for index in range(args.warmups):
                launch(control_fn, control_buffer.ptr, index)
                launch(candidate_fn, candidate_buffer.ptr, index)
            runtime.device_synchronize()

            control_us: list[float] = []
            candidate_us: list[float] = []
            orders = [
                ("control", "candidate")
                if sample % 2 == 0
                else ("candidate", "control")
                for sample in range(args.samples)
            ]

            def time_variant(name: str) -> float:
                fn = control_fn if name == "control" else candidate_fn
                out_ptr = (
                    control_buffer.ptr if name == "control" else candidate_buffer.ptr
                )
                runtime.event_record(start_event)
                for index in range(args.burst):
                    launch(fn, out_ptr, index)
                runtime.event_record(stop_event)
                runtime.event_synchronize(stop_event)
                return (
                    runtime.event_elapsed_time_ms(start_event, stop_event)
                    * 1000.0
                    / args.burst
                )

            for order in orders:
                for name in order:
                    elapsed_us = time_variant(name)
                    (control_us if name == "control" else candidate_us).append(
                        elapsed_us
                    )

            control_median = statistics.median(control_us)
            candidate_median = statistics.median(candidate_us)
            speedup = control_median / candidate_median
            saving_us = control_median - candidate_median
            projection_ms = saving_us * calls_per_token / 1000.0
            results[quant] = {
                "calls_per_token": calls_per_token,
                "selected_matrix_bytes": selected_matrix_bytes,
                "pool_groups": pool_groups,
                "pool_experts": num_experts,
                "pool_bytes": int(tiles.nbytes),
                "control_us": control_us,
                "candidate_us": candidate_us,
                "control_median_us": control_median,
                "candidate_median_us": candidate_median,
                "speedup": speedup,
                "saving_us_per_call": saving_us,
                "projected_saving_ms_per_token": projection_ms,
                "gate_pass": bool(speedup >= 1.15 or projection_ms >= 0.5),
                "exact": exact,
                "control_sha256": _sha256(control_host),
                "candidate_sha256": _sha256(candidate_host),
                "orders": [list(order) for order in orders],
            }
        finally:
            runtime.event_destroy(stop_event)
            runtime.event_destroy(start_event)
            for buffer in reversed(buffers):
                free(buffer, runtime=runtime)

    command = " ".join(
        [
            f"{name}={os.environ[name]}"
            for name in (
                "GPU_MAX_HW_QUEUES",
                "HIPENGINE_HIP_ARCH",
                "HIPENGINE_COMPILER_VERSION_FILE",
            )
            if os.environ.get(name)
        ]
        + [Path(sys.executable).name, *sys.argv]
    )
    payload = {
        "schema": 1,
        "kind": "hipengine_gfx1151_gguf_sh_d1_q5_selected_down_tile8_microbench",
        "host": platform.node(),
        "hardware": "AMD Ryzen AI MAX+ 395 / Radeon 8060S",
        "arch": os.environ.get("HIPENGINE_HIP_ARCH"),
        "candidate": "tile8",
        "shape": {
            "rows": _ROWS,
            "top_k": _TOP_K,
            "in_features": _IN_FEATURES,
            "out_features": _OUT_FEATURES,
        },
        "protocol": {
            "timing": "counterbalanced HIP events",
            "warmups_per_variant": args.warmups,
            "samples_per_variant": args.samples,
            "launches_per_sample": args.burst,
            "mall_bytes": args.mall_bytes,
        },
        "results": results,
        "command": command,
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
