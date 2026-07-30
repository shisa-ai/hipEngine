#!/usr/bin/env python3
"""Screen exact natural-shape selected-MoE decode on actual Laguna weights."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import sys
from typing import Callable

import numpy as np

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
    memory_stats,
    reset_memory_stats,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_t16_selected_gemv import (
    build_gguf_t16_selected_gemv,
    gguf_q4_k_t16_selected_dual_gemv_bf16_bf16_out,
    gguf_q4_k_t16_selected_dual_interleaved_natural_tile8_parallel_silu_gemv_bf16_bf16_out,
    gguf_q4_k_t16_selected_dual_natural_gemv_bf16_bf16_out,
    gguf_q4_k_t16_selected_dual_natural_tile8_gemv_bf16_bf16_out,
    gguf_q4_k_t16_selected_dual_natural_tile8_parallel_gemv_bf16_bf16_out,
    gguf_q4_k_t16_selected_dual_natural_tile8_parallel_silu_gemv_bf16_bf16_out,
    gguf_q4_k_t16_selected_gemv_bf16_bf16_out,
    gguf_q4_k_t16_selected_natural_gemv_bf16_bf16_out,
    gguf_q4_k_t16_selected_natural_parallel_gemv_bf16_bf16_out,
    gguf_q6_k_t16_qmicro_planar_selected_natural_gemv_bf16_bf16_out,
    gguf_q6_k_t16_qmicro_planar_selected_natural_parallel_gemv_bf16_bf16_out,
    gguf_q6_k_t16_selected_gemv_bf16_bf16_out,
)
from hipengine.kernels.hip_gfx1100.fused.paro_silu import (
    build_paro_silu,
    silu_mul_separate_out_bf16,
)
from hipengine.quant.gguf_t16 import (
    convert_gguf_q6_k_tile16_to_qmicro_planar,
)
from hipengine.quant.gguf_q4_k import interleave_gguf_q4_k_tile16_dual

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = Path(
    "/home/lhl/models/gguf/laguna-s-2.1-Q4_K_M.hipengine-repacked-v1"
)
DEFAULT_OUTPUT = Path("/tmp/laguna-selected-natural-decode-leaf.json")
EXPERTS = 256
TOP_K = 10
Q4_TILE_BYTES = 2_368
Q6_TILE_BYTES = 3_360


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repacked-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--gate-layer", type=int, default=1)
    parser.add_argument("--q4-down-layer", type=int, default=10)
    parser.add_argument("--q6-down-layer", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--samples", type=int, default=21)
    parser.add_argument("--burst", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument(
        "--gate-candidate",
        choices=(
            "natural",
            "tile8",
            "tile8-parallel",
            "tile8-parallel-silu",
            "tile8-parallel-silu-interleaved",
        ),
        default="natural",
    )
    parser.add_argument("--gate-only", action="store_true")
    parser.add_argument(
        "--down-candidate",
        choices=("natural", "parallel-tail"),
        default="natural",
    )
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _git_revision() -> str:
    return subprocess.check_output(
        ("git", "rev-parse", "HEAD"),
        cwd=ROOT,
        text=True,
    ).strip()


def _tracked_status() -> list[str]:
    output = subprocess.check_output(
        ("git", "status", "--short", "--untracked-files=no"),
        cwd=ROOT,
        text=True,
    )
    return [line for line in output.splitlines() if line]


def _bf16_bits(values: np.ndarray) -> np.ndarray:
    f32 = np.ascontiguousarray(values, dtype=np.float32)
    bits = f32.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & np.uint32(1))
    return np.ascontiguousarray((rounded >> 16).astype(np.uint16))


def _upload(runtime, values: np.ndarray):
    array = np.asarray(values)
    if not array.flags.c_contiguous:
        array = np.ascontiguousarray(array)
    buffer = malloc(array.nbytes, runtime=runtime)
    copy_host_to_device(buffer, host_array_ptr(array), runtime=runtime)
    return buffer


def _read_bf16(runtime, buffer, shape: tuple[int, ...]) -> np.ndarray:
    result = np.empty(shape, dtype=np.uint16)
    copy_device_to_host(host_array_ptr(result), buffer, runtime=runtime)
    return result


def _event_ms(
    runtime,
    fn: Callable[[], None],
    *,
    burst: int,
) -> float:
    start = runtime.event_create()
    end = runtime.event_create()
    try:
        runtime.event_record(start)
        for _ in range(burst):
            fn()
        runtime.event_record(end)
        runtime.event_synchronize(end)
        return float(runtime.event_elapsed_time_ms(start, end)) / burst
    finally:
        runtime.event_destroy(end)
        runtime.event_destroy(start)


def _cache_tiles(
    cache_root: Path,
    manifest: dict,
    *,
    layer: int,
    slot: str,
    layout: str,
    shape: tuple[int, ...],
) -> tuple[np.ndarray, dict]:
    entry = manifest["entries"][f"layers.{layer}.{slot}"]
    allocation = entry["allocations"]["tiles"]
    tiles = np.load(cache_root / allocation["file"], mmap_mode="r")
    if (
        entry["layout"] != layout
        or tuple(tiles.shape) != shape
        or tiles.dtype != np.uint8
    ):
        raise ValueError(
            f"unexpected {slot} layer {layer}: "
            f"{entry['layout']} {tiles.shape} {tiles.dtype}"
        )
    return tiles, entry


def _screen_pair(
    *,
    runtime,
    library,
    control: Callable[..., None],
    candidate: Callable[..., None],
    x: np.ndarray,
    selected: np.ndarray,
    tiles_a: np.ndarray,
    tiles_b: np.ndarray,
    in_features: int,
    out_features: int,
    warmups: int,
    samples: int,
    burst: int,
) -> dict:
    buffers = []
    try:
        x_dev = _upload(runtime, x)
        selected_dev = _upload(runtime, selected)
        tiles_a_dev = _upload(runtime, tiles_a)
        tiles_b_dev = _upload(runtime, tiles_b)
        control_a_dev = malloc(TOP_K * out_features * 2, runtime=runtime)
        control_b_dev = malloc(TOP_K * out_features * 2, runtime=runtime)
        candidate_a_dev = malloc(TOP_K * out_features * 2, runtime=runtime)
        candidate_b_dev = malloc(TOP_K * out_features * 2, runtime=runtime)
        buffers.extend(
            (
                x_dev,
                selected_dev,
                tiles_a_dev,
                tiles_b_dev,
                control_a_dev,
                control_b_dev,
                candidate_a_dev,
                candidate_b_dev,
            )
        )

        def launch(
            wrapper: Callable[..., None],
            out_a_ptr: int,
            out_b_ptr: int,
        ) -> None:
            wrapper(
                x_dev.ptr,
                selected_dev.ptr,
                tiles_a_dev.ptr,
                tiles_b_dev.ptr,
                out_a_ptr,
                out_b_ptr,
                int(x.shape[0]),
                TOP_K,
                EXPERTS,
                in_features,
                out_features,
                library=library,
                runtime=runtime,
            )

        launchers = {
            "control": lambda: launch(
                control, control_a_dev.ptr, control_b_dev.ptr
            ),
            "candidate": lambda: launch(
                candidate, candidate_a_dev.ptr, candidate_b_dev.ptr
            ),
        }
        for _ in range(warmups):
            launchers["control"]()
            launchers["candidate"]()
        runtime.device_synchronize()
        timings = {"control": [], "candidate": []}
        for sample in range(samples):
            order = (
                ("control", "candidate")
                if sample % 2 == 0
                else ("candidate", "control")
            )
            for name in order:
                timings[name].append(
                    _event_ms(runtime, launchers[name], burst=burst)
                )
        launchers["control"]()
        launchers["candidate"]()
        runtime.device_synchronize()
        control_a = _read_bf16(
            runtime, control_a_dev, (TOP_K, out_features)
        )
        control_b = _read_bf16(
            runtime, control_b_dev, (TOP_K, out_features)
        )
        candidate_a = _read_bf16(
            runtime, candidate_a_dev, (TOP_K, out_features)
        )
        candidate_b = _read_bf16(
            runtime, candidate_b_dev, (TOP_K, out_features)
        )
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)

    medians = {
        name: statistics.median(values) for name, values in timings.items()
    }
    ratio = medians["candidate"] / medians["control"]
    return {
        "samples_ms": timings,
        "median_ms": medians,
        "candidate_over_control": ratio,
        "candidate_delta_percent": (ratio - 1.0) * 100.0,
        "bf16_mismatch_a": int(np.count_nonzero(candidate_a != control_a)),
        "bf16_mismatch_b": int(np.count_nonzero(candidate_b != control_b)),
    }


def _screen_pair_silu(
    *,
    runtime,
    library,
    silu_library,
    x: np.ndarray,
    selected: np.ndarray,
    tiles_a: np.ndarray,
    tiles_b: np.ndarray,
    in_features: int,
    out_features: int,
    warmups: int,
    samples: int,
    burst: int,
) -> dict:
    buffers = []
    try:
        x_dev = _upload(runtime, x)
        selected_dev = _upload(runtime, selected)
        tiles_a_dev = _upload(runtime, tiles_a)
        tiles_b_dev = _upload(runtime, tiles_b)
        control_gate_dev = malloc(TOP_K * out_features * 2, runtime=runtime)
        control_up_dev = malloc(TOP_K * out_features * 2, runtime=runtime)
        control_out_dev = malloc(TOP_K * out_features * 2, runtime=runtime)
        candidate_out_dev = malloc(TOP_K * out_features * 2, runtime=runtime)
        buffers.extend(
            (
                x_dev,
                selected_dev,
                tiles_a_dev,
                tiles_b_dev,
                control_gate_dev,
                control_up_dev,
                control_out_dev,
                candidate_out_dev,
            )
        )

        def control() -> None:
            gguf_q4_k_t16_selected_dual_natural_tile8_parallel_gemv_bf16_bf16_out(
                x_dev.ptr,
                selected_dev.ptr,
                tiles_a_dev.ptr,
                tiles_b_dev.ptr,
                control_gate_dev.ptr,
                control_up_dev.ptr,
                int(x.shape[0]),
                TOP_K,
                EXPERTS,
                in_features,
                out_features,
                library=library,
                runtime=runtime,
            )
            silu_mul_separate_out_bf16(
                control_gate_dev.ptr,
                control_up_dev.ptr,
                control_out_dev.ptr,
                TOP_K,
                out_features,
                library=silu_library,
                runtime=runtime,
            )

        def candidate() -> None:
            gguf_q4_k_t16_selected_dual_natural_tile8_parallel_silu_gemv_bf16_bf16_out(
                x_dev.ptr,
                selected_dev.ptr,
                tiles_a_dev.ptr,
                tiles_b_dev.ptr,
                candidate_out_dev.ptr,
                int(x.shape[0]),
                TOP_K,
                EXPERTS,
                in_features,
                out_features,
                library=library,
                runtime=runtime,
            )

        launchers = {"control": control, "candidate": candidate}
        for _ in range(warmups):
            control()
            candidate()
        runtime.device_synchronize()
        timings = {"control": [], "candidate": []}
        for sample in range(samples):
            order = (
                ("control", "candidate")
                if sample % 2 == 0
                else ("candidate", "control")
            )
            for name in order:
                timings[name].append(
                    _event_ms(runtime, launchers[name], burst=burst)
                )
        control()
        candidate()
        runtime.device_synchronize()
        control_out = _read_bf16(
            runtime, control_out_dev, (TOP_K, out_features)
        )
        candidate_out = _read_bf16(
            runtime, candidate_out_dev, (TOP_K, out_features)
        )
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)

    medians = {
        name: statistics.median(values) for name, values in timings.items()
    }
    ratio = medians["candidate"] / medians["control"]
    return {
        "samples_ms": timings,
        "median_ms": medians,
        "candidate_over_control": ratio,
        "candidate_delta_percent": (ratio - 1.0) * 100.0,
        "bf16_mismatch": int(
            np.count_nonzero(candidate_out != control_out)
        ),
        "control_launches": 2,
        "candidate_launches": 1,
    }


def _screen_pair_silu_interleaved(
    *,
    runtime,
    library,
    x: np.ndarray,
    selected: np.ndarray,
    tiles_a: np.ndarray,
    tiles_b: np.ndarray,
    tiles_dual: np.ndarray,
    in_features: int,
    out_features: int,
    warmups: int,
    samples: int,
    burst: int,
) -> dict:
    buffers = []
    try:
        x_dev = _upload(runtime, x)
        selected_dev = _upload(runtime, selected)
        tiles_a_dev = _upload(runtime, tiles_a)
        tiles_b_dev = _upload(runtime, tiles_b)
        tiles_dual_dev = _upload(runtime, tiles_dual)
        control_out_dev = malloc(TOP_K * out_features * 2, runtime=runtime)
        candidate_out_dev = malloc(TOP_K * out_features * 2, runtime=runtime)
        buffers.extend(
            (
                x_dev,
                selected_dev,
                tiles_a_dev,
                tiles_b_dev,
                tiles_dual_dev,
                control_out_dev,
                candidate_out_dev,
            )
        )

        def control() -> None:
            gguf_q4_k_t16_selected_dual_natural_tile8_parallel_silu_gemv_bf16_bf16_out(
                x_dev.ptr,
                selected_dev.ptr,
                tiles_a_dev.ptr,
                tiles_b_dev.ptr,
                control_out_dev.ptr,
                int(x.shape[0]),
                TOP_K,
                EXPERTS,
                in_features,
                out_features,
                library=library,
                runtime=runtime,
            )

        def candidate() -> None:
            gguf_q4_k_t16_selected_dual_interleaved_natural_tile8_parallel_silu_gemv_bf16_bf16_out(
                x_dev.ptr,
                selected_dev.ptr,
                tiles_dual_dev.ptr,
                tiles_dual_dev.ptr,
                candidate_out_dev.ptr,
                int(x.shape[0]),
                TOP_K,
                EXPERTS,
                in_features,
                out_features,
                library=library,
                runtime=runtime,
            )

        launchers = {"control": control, "candidate": candidate}
        for _ in range(warmups):
            control()
            candidate()
        runtime.device_synchronize()
        timings = {"control": [], "candidate": []}
        for sample in range(samples):
            order = (
                ("control", "candidate")
                if sample % 2 == 0
                else ("candidate", "control")
            )
            for name in order:
                timings[name].append(
                    _event_ms(runtime, launchers[name], burst=burst)
                )
        control()
        candidate()
        runtime.device_synchronize()
        control_out = _read_bf16(
            runtime, control_out_dev, (TOP_K, out_features)
        )
        candidate_out = _read_bf16(
            runtime, candidate_out_dev, (TOP_K, out_features)
        )
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)

    medians = {
        name: statistics.median(values) for name, values in timings.items()
    }
    ratio = medians["candidate"] / medians["control"]
    return {
        "samples_ms": timings,
        "median_ms": medians,
        "candidate_over_control": ratio,
        "candidate_delta_percent": (ratio - 1.0) * 100.0,
        "bf16_mismatch": int(
            np.count_nonzero(candidate_out != control_out)
        ),
        "control_launches": 1,
        "candidate_launches": 1,
        "control_resident_bytes": int(tiles_a.nbytes + tiles_b.nbytes),
        "candidate_resident_bytes": int(tiles_dual.nbytes),
    }


def _screen_single(
    *,
    runtime,
    library,
    control: Callable[..., None],
    candidate: Callable[..., None],
    x: np.ndarray,
    selected: np.ndarray,
    tiles: np.ndarray,
    in_features: int,
    out_features: int,
    warmups: int,
    samples: int,
    burst: int,
    control_kwargs: dict | None = None,
) -> dict:
    buffers = []
    try:
        x_dev = _upload(runtime, x)
        selected_dev = _upload(runtime, selected)
        tiles_dev = _upload(runtime, tiles)
        control_out_dev = malloc(TOP_K * out_features * 2, runtime=runtime)
        candidate_out_dev = malloc(TOP_K * out_features * 2, runtime=runtime)
        buffers.extend(
            (
                x_dev,
                selected_dev,
                tiles_dev,
                control_out_dev,
                candidate_out_dev,
            )
        )

        def launch(
            wrapper: Callable[..., None],
            out_ptr: int,
            kwargs: dict | None = None,
        ) -> None:
            wrapper(
                x_dev.ptr,
                selected_dev.ptr,
                tiles_dev.ptr,
                out_ptr,
                int(x.shape[0]),
                TOP_K,
                EXPERTS,
                in_features,
                out_features,
                library=library,
                runtime=runtime,
                **(kwargs or {}),
            )

        launchers = {
            "control": lambda: launch(
                control, control_out_dev.ptr, control_kwargs
            ),
            "candidate": lambda: launch(candidate, candidate_out_dev.ptr),
        }
        for _ in range(warmups):
            launchers["control"]()
            launchers["candidate"]()
        runtime.device_synchronize()
        timings = {"control": [], "candidate": []}
        for sample in range(samples):
            order = (
                ("control", "candidate")
                if sample % 2 == 0
                else ("candidate", "control")
            )
            for name in order:
                timings[name].append(
                    _event_ms(runtime, launchers[name], burst=burst)
                )
        launchers["control"]()
        launchers["candidate"]()
        runtime.device_synchronize()
        control_out = _read_bf16(
            runtime, control_out_dev, (TOP_K, out_features)
        )
        candidate_out = _read_bf16(
            runtime, candidate_out_dev, (TOP_K, out_features)
        )
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)

    medians = {
        name: statistics.median(values) for name, values in timings.items()
    }
    ratio = medians["candidate"] / medians["control"]
    return {
        "samples_ms": timings,
        "median_ms": medians,
        "candidate_over_control": ratio,
        "candidate_delta_percent": (ratio - 1.0) * 100.0,
        "bf16_mismatch": int(
            np.count_nonzero(candidate_out != control_out)
        ),
    }


def main() -> int:
    args = _parse_args()
    tracked_status = _tracked_status()
    if tracked_status and not args.allow_dirty:
        raise SystemExit(
            "tracked worktree must be clean; pass --allow-dirty for a screen"
        )
    if args.warmups < 0 or min(args.samples, args.burst) <= 0:
        raise ValueError("warmups must be non-negative; samples/burst positive")
    if args.compiler_version_file is not None:
        os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(
            args.compiler_version_file
        )

    manifest = json.loads(
        (args.repacked_cache / "manifest.json").read_text(encoding="utf-8")
    )
    gate, gate_entry = _cache_tiles(
        args.repacked_cache,
        manifest,
        layer=args.gate_layer,
        slot="ffn_gate_exps",
        layout="gguf_q4_k_t16_v1",
        shape=(EXPERTS, 64, 12, Q4_TILE_BYTES),
    )
    up, up_entry = _cache_tiles(
        args.repacked_cache,
        manifest,
        layer=args.gate_layer,
        slot="ffn_up_exps",
        layout="gguf_q4_k_t16_v1",
        shape=(EXPERTS, 64, 12, Q4_TILE_BYTES),
    )
    q4_down, q4_down_entry = _cache_tiles(
        args.repacked_cache,
        manifest,
        layer=args.q4_down_layer,
        slot="ffn_down_exps",
        layout="gguf_q4_k_t16_v1",
        shape=(EXPERTS, 192, 4, Q4_TILE_BYTES),
    )
    q6_down_legacy, q6_down_entry = _cache_tiles(
        args.repacked_cache,
        manifest,
        layer=args.q6_down_layer,
        slot="ffn_down_exps",
        layout="gguf_q6_k_t16_v1",
        shape=(EXPERTS, 192, 4, Q6_TILE_BYTES),
    )
    q6_down = convert_gguf_q6_k_tile16_to_qmicro_planar(
        q6_down_legacy
    ).tiles

    selected = np.asarray(
        [17, 3, 91, 42, 7, 128, 201, 55, 240, 12],
        dtype=np.int64,
    )
    rng = np.random.default_rng(args.seed)
    x_gate = _bf16_bits(
        rng.normal(0.0, 0.55, size=(1, 3072)).astype(np.float32)
    )
    x_down = _bf16_bits(
        rng.normal(0.0, 0.55, size=(TOP_K, 1024)).astype(np.float32)
    )

    runtime = get_hip_runtime()
    library = build_gguf_t16_selected_gemv(
        load=True,
        require_cached=args.require_cached_build,
    )
    silu_library = (
        build_paro_silu(
            load=True,
            require_cached=args.require_cached_build,
        )
        if args.gate_candidate == "tile8-parallel-silu"
        else None
    )
    reset_memory_stats()
    common = {
        "runtime": runtime,
        "library": library,
        "selected": selected,
        "warmups": args.warmups,
        "samples": args.samples,
        "burst": args.burst,
    }
    gate_modes = {
        "natural": (
            gguf_q4_k_t16_selected_dual_gemv_bf16_bf16_out,
            gguf_q4_k_t16_selected_dual_natural_gemv_bf16_bf16_out,
        ),
        "tile8": (
            gguf_q4_k_t16_selected_dual_natural_gemv_bf16_bf16_out,
            gguf_q4_k_t16_selected_dual_natural_tile8_gemv_bf16_bf16_out,
        ),
        "tile8-parallel": (
            gguf_q4_k_t16_selected_dual_natural_tile8_gemv_bf16_bf16_out,
            (
                gguf_q4_k_t16_selected_dual_natural_tile8_parallel_gemv_bf16_bf16_out
            ),
        ),
    }
    if args.gate_candidate == "tile8-parallel-silu":
        assert silu_library is not None
        gate_result = _screen_pair_silu(
            runtime=runtime,
            library=library,
            silu_library=silu_library,
            x=x_gate,
            selected=selected,
            tiles_a=gate,
            tiles_b=up,
            in_features=3072,
            out_features=1024,
            warmups=args.warmups,
            samples=args.samples,
            burst=args.burst,
        )
    elif args.gate_candidate == "tile8-parallel-silu-interleaved":
        gate_dual = interleave_gguf_q4_k_tile16_dual(gate, up)
        gate_result = _screen_pair_silu_interleaved(
            runtime=runtime,
            library=library,
            x=x_gate,
            selected=selected,
            tiles_a=gate,
            tiles_b=up,
            tiles_dual=gate_dual,
            in_features=3072,
            out_features=1024,
            warmups=args.warmups,
            samples=args.samples,
            burst=args.burst,
        )
    else:
        gate_control, gate_candidate = gate_modes[args.gate_candidate]
        gate_result = _screen_pair(
            control=gate_control,
            candidate=gate_candidate,
            x=x_gate,
            tiles_a=gate,
            tiles_b=up,
            in_features=3072,
            out_features=1024,
            **common,
        )
    results = {
        "q4_gate_up": gate_result,
    }
    if not args.gate_only:
        if args.down_candidate == "parallel-tail":
            q4_down_control = (
                gguf_q4_k_t16_selected_natural_gemv_bf16_bf16_out
            )
            q4_down_candidate = (
                gguf_q4_k_t16_selected_natural_parallel_gemv_bf16_bf16_out
            )
            q6_down_control = (
                gguf_q6_k_t16_qmicro_planar_selected_natural_gemv_bf16_bf16_out
            )
            q6_down_candidate = (
                gguf_q6_k_t16_qmicro_planar_selected_natural_parallel_gemv_bf16_bf16_out
            )
            q6_control_kwargs = None
        else:
            q4_down_control = gguf_q4_k_t16_selected_gemv_bf16_bf16_out
            q4_down_candidate = (
                gguf_q4_k_t16_selected_natural_gemv_bf16_bf16_out
            )
            q6_down_control = gguf_q6_k_t16_selected_gemv_bf16_bf16_out
            q6_down_candidate = (
                gguf_q6_k_t16_qmicro_planar_selected_natural_gemv_bf16_bf16_out
            )
            q6_control_kwargs = {"qmicro": True, "qmicro_planar": True}
        results.update(
            {
                "q4_down": _screen_single(
                    control=q4_down_control,
                    candidate=q4_down_candidate,
                    x=x_down,
                    tiles=q4_down,
                    in_features=1024,
                    out_features=3072,
                    **common,
                ),
                "q6_down": _screen_single(
                    control=q6_down_control,
                    candidate=q6_down_candidate,
                    x=x_down,
                    tiles=q6_down,
                    in_features=1024,
                    out_features=3072,
                    control_kwargs=q6_control_kwargs,
                    **common,
                ),
            }
        )
    exact = all(
        item.get("bf16_mismatch", 0) == 0
        and item.get("bf16_mismatch_a", 0) == 0
        and item.get("bf16_mismatch_b", 0) == 0
        for item in results.values()
    )
    all_positive = all(
        item["candidate_over_control"] < 1.0 for item in results.values()
    )
    selected_hash = hashlib.sha256(
        selected.astype("<i8").tobytes()
    ).hexdigest()
    stats = memory_stats()
    artifact = {
        "schema": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "kind": "hipengine_laguna_selected_natural_decode_leaf",
        "status": "candidate" if exact and all_positive else "rejected",
        "performance_claim": False,
        "scope": (
            "actual-weight gfx1151 c=1/top-10 selected gate/up and down leaf"
        ),
        "hardware": {
            "gpu": "AMD Radeon 8060S Graphics",
            "architecture": "gfx1151",
        },
        "workload": {
            "gate_up_x_rows": 1,
            "down_x_rows": TOP_K,
            "selected_rows": TOP_K,
            "experts": EXPERTS,
            "selected_experts": selected.tolist(),
            "selected_sha256": selected_hash,
            "gate_up": {
                "layer": args.gate_layer,
                "shape": "K3072/N1024/Q4T16 dual",
                "candidate": args.gate_candidate,
            },
            "down_candidate": args.down_candidate,
            "q4_down": {
                "layer": args.q4_down_layer,
                "shape": "K1024/N3072/Q4T16",
            },
            "q6_down": {
                "layer": args.q6_down_layer,
                "shape": "K1024/N3072/Q6T16 qmicro planar",
            },
        },
        "cache_entries": {
            "gate": gate_entry,
            "up": up_entry,
            "q4_down": q4_down_entry,
            "q6_down": q6_down_entry,
        },
        "protocol": {
            "warmups": args.warmups,
            "samples": args.samples,
            "burst": args.burst,
            "timing": "counterbalanced HIP-event elapsed time",
            "correctness": "BF16 bit equality to current production wrapper",
            "promotion_gate": (
                "each role exact and faster; whole-cycle/state/trace gates "
                "remain mandatory"
            ),
            "command": shlex.join(sys.argv),
        },
        "repo": {
            "revision": _git_revision(),
            "tracked_status": tracked_status,
        },
        "memory": {
            "tracked_peak_bytes": stats["peak_allocated_bytes"],
            "tracked_after_bytes": stats["current_allocated_bytes"],
        },
        "results": results,
        "gate": {
            "exact": exact,
            "all_roles_positive": all_positive,
            "passed": exact and all_positive,
        },
        "notes": [
            "The leaf reuses production resident T16 bytes.",
            "The Q6 layer is converted to the production qmicro-planar layout.",
            "No runtime dispatch changes are made by this artifact.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for name, result in results.items():
        print(
            f"{name}: control={result['median_ms']['control']:.6f} ms "
            f"candidate={result['median_ms']['candidate']:.6f} ms "
            f"delta={result['candidate_delta_percent']:+.3f}%"
        )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "status": artifact["status"],
                "gate": artifact["gate"],
            },
            sort_keys=True,
        )
    )
    if not exact:
        raise SystemExit("natural-shape output differs from production bits")
    if not all_positive:
        raise SystemExit("one or more natural-shape roles regress")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
