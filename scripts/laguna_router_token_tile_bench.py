#!/usr/bin/env python3
"""Screen exact Laguna F32-weight router token-reuse tiles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

import numpy as np

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.hip_gfx1100.moe.router import (
    build_qwen35_router,
    qwen35_router_logits_bf16_f32w_auto_256,
    qwen35_router_logits_bf16_f32w_token_tile_8,
    qwen35_router_logits_bf16_f32w_token_tile_16,
)

_LAUNCHES = {
    "token_tile_4": qwen35_router_logits_bf16_f32w_auto_256,
    "token_tile_8": qwen35_router_logits_bf16_f32w_token_tile_8,
    "token_tile_16": qwen35_router_logits_bf16_f32w_token_tile_16,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", default="128,256,512")
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--samples", type=int, default=11)
    parser.add_argument("--burst", type=int, default=5)
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _bf16_bits(values: np.ndarray) -> np.ndarray:
    bits = values.astype(np.float32, copy=False).view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)
    return (rounded >> 16).astype(np.uint16)


def _compiler_version(path: Path | None) -> str | None:
    if path is None:
        return None
    return path.read_text().strip()


def main() -> int:
    args = _parse_args()
    rows = tuple(int(value) for value in args.rows.split(","))
    if (
        not rows
        or any(value <= 0 for value in rows)
        or args.warmups < 0
        or args.samples <= 0
        or args.burst <= 0
    ):
        raise ValueError("rows/samples/burst must be positive and warmups nonnegative")

    hidden_size = 3_072
    experts = 256
    rng = np.random.default_rng(20260726)
    hidden = _bf16_bits(
        rng.normal(
            0.0,
            0.04,
            size=(max(rows), hidden_size),
        ).astype(np.float32)
    )
    weight = rng.normal(
        0.0,
        0.03,
        size=(experts, hidden_size),
    ).astype(np.float32)
    output = np.zeros((max(rows), experts), dtype=np.float32)
    runtime = get_hip_runtime()
    library = build_qwen35_router(
        load=True,
        compiler_version=_compiler_version(args.compiler_version_file),
        require_cached=args.require_cached_build,
    )
    buffers = [
        malloc(hidden.nbytes, runtime=runtime),
        malloc(weight.nbytes, runtime=runtime),
        malloc(output.nbytes, runtime=runtime),
    ]
    start = runtime.event_create()
    stop = runtime.event_create()
    result: dict[str, object] = {
        "kind": "laguna_router_token_tile_screen",
        "hardware": {
            "backend": "hip_gfx1151",
            "architecture": "gfx1151",
            "gpu_max_hw_queues": 1,
        },
        "shape": {
            "hidden_size": hidden_size,
            "experts": experts,
            "rows": rows,
        },
        "protocol": {
            "warmups": args.warmups,
            "samples": args.samples,
            "burst": args.burst,
        },
        "results": {},
    }
    try:
        copy_host_to_device(
            buffers[0],
            host_array_ptr(hidden),
            hidden.nbytes,
            runtime=runtime,
        )
        copy_host_to_device(
            buffers[1],
            host_array_ptr(weight),
            weight.nbytes,
            runtime=runtime,
        )
        for row_count in rows:
            samples = {name: [] for name in _LAUNCHES}
            host_outputs: dict[str, np.ndarray] = {}
            for name, launch in _LAUNCHES.items():
                launch(
                    buffers[0].ptr,
                    buffers[1].ptr,
                    buffers[2].ptr,
                    row_count,
                    hidden_size,
                    experts,
                    library=library,
                    runtime=runtime,
                )
                runtime.device_synchronize()
                host_output = np.empty(
                    (row_count, experts),
                    dtype=np.float32,
                )
                copy_device_to_host(
                    host_array_ptr(host_output),
                    buffers[2],
                    host_output.nbytes,
                    runtime=runtime,
                )
                host_outputs[name] = host_output
            for candidate in ("token_tile_8", "token_tile_16"):
                if not np.array_equal(
                    host_outputs[candidate],
                    host_outputs["token_tile_4"],
                ):
                    raise AssertionError(
                        f"{candidate} differs from exact token_tile_4"
                    )
            for _ in range(args.warmups):
                for launch in _LAUNCHES.values():
                    launch(
                        buffers[0].ptr,
                        buffers[1].ptr,
                        buffers[2].ptr,
                        row_count,
                        hidden_size,
                        experts,
                        library=library,
                        runtime=runtime,
                    )
            runtime.device_synchronize()
            names = tuple(_LAUNCHES)
            for sample in range(args.samples):
                order = names[sample % len(names) :] + names[: sample % len(names)]
                for name in order:
                    launch = _LAUNCHES[name]
                    runtime.event_record(start)
                    for _ in range(args.burst):
                        launch(
                            buffers[0].ptr,
                            buffers[1].ptr,
                            buffers[2].ptr,
                            row_count,
                            hidden_size,
                            experts,
                            library=library,
                            runtime=runtime,
                        )
                    runtime.event_record(stop)
                    runtime.event_synchronize(stop)
                    samples[name].append(
                        runtime.event_elapsed_time_ms(start, stop)
                        / args.burst
                    )
            medians = {
                name: statistics.median(values)
                for name, values in samples.items()
            }
            result["results"][str(row_count)] = {
                "samples_ms": samples,
                "median_ms": medians,
                "speedup_vs_tile4": {
                    name: medians["token_tile_4"] / value
                    for name, value in medians.items()
                },
                "f32_logits_exact": True,
            }
            print(
                row_count,
                " ".join(
                    f"{name}={medians[name]:.6f}ms"
                    for name in names
                ),
                flush=True,
            )
    finally:
        runtime.event_destroy(stop)
        runtime.event_destroy(start)
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)

    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.write_text(rendered + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
