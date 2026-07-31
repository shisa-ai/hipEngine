#!/usr/bin/env python3
"""Screen gfx11 any-order output->router continuation at Laguna shapes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

import numpy as np

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.hip_gfx1100.linear.laguna_f16_projection import (
    build_laguna_f16_projection,
    laguna_f16w_fixedk_nontemporal_output_add_rmsnorm_signal_bf16,
    laguna_f16w_fixedk_nontemporal_output_add_rmsnorm_bf16,
)
from hipengine.kernels.hip_gfx1100.moe.router import (
    build_qwen35_router,
    qwen35_router_logits_bf16_f32w_wave0_tree,
    qwen35_router_logits_bf16_f32w_wave0_tree_anyorder,
)
from hipengine.loading.materialize import float_array_to_bf16_bits


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _median(values: list[float]) -> float:
    return float(statistics.median(values))


def _screen_shape(
    in_features: int,
    *,
    samples: int,
    warmups: int,
    burst: int,
    require_cached: bool,
    rng: np.random.Generator,
) -> dict[str, object]:
    hidden_size = 3072
    experts = 256
    x = float_array_to_bf16_bits(
        rng.normal(0.0, 0.2, size=in_features).astype(np.float32)
    )
    residual = float_array_to_bf16_bits(
        rng.normal(0.0, 0.1, size=hidden_size).astype(np.float32)
    )
    output_weight = rng.normal(
        0.0, 0.01, size=(hidden_size, in_features)
    ).astype(np.float16)
    norm_weight = rng.normal(1.0, 0.05, size=hidden_size).astype(np.float32)
    router_weight = rng.normal(
        0.0, 0.02, size=(experts, hidden_size)
    ).astype(np.float32)

    runtime = get_hip_runtime()
    linear = build_laguna_f16_projection(
        load=True, require_cached=require_cached
    )
    router = build_qwen35_router(load=True, require_cached=require_cached)
    allocations = []

    def upload(array: np.ndarray):
        contiguous = np.ascontiguousarray(array)
        buffer = malloc(contiguous.nbytes, runtime=runtime)
        allocations.append(buffer)
        copy_host_to_device(
            buffer,
            host_array_ptr(contiguous),
            contiguous.nbytes,
            runtime=runtime,
        )
        return buffer

    def allocate(nbytes: int):
        buffer = malloc(nbytes, runtime=runtime)
        allocations.append(buffer)
        return buffer

    def download(buffer, shape, dtype):
        out = np.empty(shape, dtype=dtype)
        copy_device_to_host(
            host_array_ptr(out),
            buffer,
            out.nbytes,
            runtime=runtime,
        )
        return out

    start = runtime.event_create()
    stop = runtime.event_create()
    try:
        dx = upload(x)
        doutput_weight = upload(output_weight)
        dresidual = upload(residual)
        dnorm_weight = upload(norm_weight)
        drouter_weight = upload(router_weight)
        control_projection = allocate(hidden_size * 2)
        control_norm = allocate(hidden_size * 2)
        control_residual = allocate(hidden_size * 2)
        control_logits = allocate(experts * 4)
        candidate_projection = allocate(hidden_size * 2)
        candidate_norm = allocate(hidden_size * 2)
        candidate_residual = allocate(hidden_size * 2)
        candidate_logits = allocate(experts * 4)
        control_counters = allocate(3 * 4)
        candidate_counters = allocate(3 * 4)
        runtime.memset(control_counters.ptr, 0, control_counters.nbytes)
        runtime.memset(candidate_counters.ptr, 0, candidate_counters.nbytes)

        def launch_control() -> None:
            laguna_f16w_fixedk_nontemporal_output_add_rmsnorm_bf16(
                dx.ptr,
                doutput_weight.ptr,
                control_projection.ptr,
                dresidual.ptr,
                dnorm_weight.ptr,
                control_norm.ptr,
                control_residual.ptr,
                control_counters.ptr,
                1,
                in_features,
                hidden_size,
                1.0e-6,
                library=linear,
                runtime=runtime,
            )
            qwen35_router_logits_bf16_f32w_wave0_tree(
                control_norm.ptr,
                drouter_weight.ptr,
                control_logits.ptr,
                1,
                hidden_size,
                experts,
                library=router,
                runtime=runtime,
            )

        def launch_candidate() -> None:
            laguna_f16w_fixedk_nontemporal_output_add_rmsnorm_signal_bf16(
                dx.ptr,
                doutput_weight.ptr,
                candidate_projection.ptr,
                dresidual.ptr,
                dnorm_weight.ptr,
                candidate_norm.ptr,
                candidate_residual.ptr,
                candidate_counters.ptr,
                1,
                in_features,
                hidden_size,
                1.0e-6,
                library=linear,
                runtime=runtime,
            )
            qwen35_router_logits_bf16_f32w_wave0_tree_anyorder(
                candidate_norm.ptr,
                drouter_weight.ptr,
                candidate_logits.ptr,
                candidate_counters.ptr,
                1,
                hidden_size,
                experts,
                library=router,
                runtime=runtime,
            )

        for _ in range(warmups):
            launch_control()
            launch_candidate()
        runtime.device_synchronize()

        for control, candidate, shape, dtype in (
            (
                control_projection,
                candidate_projection,
                (hidden_size,),
                np.uint16,
            ),
            (control_norm, candidate_norm, (hidden_size,), np.uint16),
            (
                control_residual,
                candidate_residual,
                (hidden_size,),
                np.uint16,
            ),
            (control_logits, candidate_logits, (experts,), np.float32),
        ):
            np.testing.assert_array_equal(
                download(candidate, shape, dtype),
                download(control, shape, dtype),
            )
        np.testing.assert_array_equal(
            download(candidate_counters, (3,), np.int32),
            np.zeros(3, dtype=np.int32),
        )

        event_ms = {"control": [], "candidate": []}
        wall_ms = {"control": [], "candidate": []}

        def measure(name: str, launch) -> None:
            wall_start = time.perf_counter_ns()
            runtime.event_record(start)
            for _ in range(burst):
                launch()
            runtime.event_record(stop)
            runtime.event_synchronize(stop)
            wall_stop = time.perf_counter_ns()
            event_ms[name].append(
                float(runtime.event_elapsed_time_ms(start, stop)) / burst
            )
            wall_ms[name].append(
                float(wall_stop - wall_start) / 1.0e6 / burst
            )

        for sample in range(samples):
            order = (
                (("control", launch_control), ("candidate", launch_candidate))
                if sample % 2 == 0
                else (("candidate", launch_candidate), ("control", launch_control))
            )
            for name, launch in order:
                measure(name, launch)

        control_event = _median(event_ms["control"])
        candidate_event = _median(event_ms["candidate"])
        control_wall = _median(wall_ms["control"])
        candidate_wall = _median(wall_ms["candidate"])
        return {
            "in_features": in_features,
            "control_event_ms": control_event,
            "candidate_event_ms": candidate_event,
            "event_delta_percent": (candidate_event / control_event - 1.0) * 100.0,
            "control_wall_ms": control_wall,
            "candidate_wall_ms": candidate_wall,
            "wall_delta_percent": (candidate_wall / control_wall - 1.0) * 100.0,
            "event_samples_ms": event_ms,
            "wall_samples_ms": wall_ms,
            "correctness": {
                "projection_bf16_bit_exact": True,
                "norm_bf16_bit_exact": True,
                "residual_bf16_bit_exact": True,
                "router_logits_f32_bit_exact": True,
                "candidate_counters_reset": True,
            },
        }
    finally:
        runtime.event_destroy(stop)
        runtime.event_destroy(start)
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=21)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--burst", type=int, default=100)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.samples, args.burst) <= 0 or args.warmups < 0:
        parser.error("samples/burst must be positive and warmups non-negative")

    rng = np.random.default_rng(20260731)
    results = [
        _screen_shape(
            in_features,
            samples=args.samples,
            warmups=args.warmups,
            burst=args.burst,
            require_cached=args.require_cached_build,
            rng=rng,
        )
        for in_features in (6144, 9216)
    ]
    payload = {
        "schema": 1,
        "kind": "laguna_output_router_anyorder_leaf",
        "protocol": {
            "samples": args.samples,
            "warmups": args.warmups,
            "burst": args.burst,
            "counterbalanced": True,
            "timers": ["HIP event", "host wall through event synchronization"],
            "synthetic_values": True,
            "natural_shapes": True,
        },
        "results": results,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"output_sha256={_sha256(args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
