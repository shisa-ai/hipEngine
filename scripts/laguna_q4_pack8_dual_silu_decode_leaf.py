#!/usr/bin/env python3
"""Screen exact fused Q4 pack8 gate/up plus SiLU at Laguna decode shapes."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
from typing import Callable

import numpy as np

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.hip_gfx1100.fused.paro_silu import (
    build_paro_silu,
    silu_mul_separate_out_bf16,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_gemv import (
    build_gguf_q4_k_gemv,
    gguf_q4_k_pack8_dual_prefill_bf16_bf16_out,
    gguf_q4_k_pack8_dual_silu_bf16_bf16_out,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = Path(
    "/home/lhl/models/gguf/laguna-s-2.1-Q4_K_M.hipengine-repacked-v1"
)
DEFAULT_OUTPUT = Path("/tmp/laguna-q4-pack8-dual-silu-decode-leaf.json")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repacked-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--samples", type=int, default=21)
    parser.add_argument("--burst", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


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


def _event_ms(runtime, fn: Callable[[], None], *, burst: int) -> float:
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


def _load_pack8(
    cache_root: Path,
    manifest: dict,
    *,
    key: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    entry = manifest["entries"][key]
    if entry["layout"] != "q4_k_pack8":
        raise ValueError(f"{key} is not q4_k_pack8")
    arrays = []
    for name in ("qweight", "scales", "mins"):
        allocation = entry["allocations"][name]
        arrays.append(
            np.load(cache_root / allocation["file"], mmap_mode="r")
        )
    return tuple(arrays)


def _screen(
    *,
    runtime,
    q4_library,
    silu_library,
    x: np.ndarray,
    gate: tuple[np.ndarray, np.ndarray, np.ndarray],
    up: tuple[np.ndarray, np.ndarray, np.ndarray],
    in_features: int,
    out_features: int,
    warmups: int,
    samples: int,
    burst: int,
) -> dict:
    buffers = []
    try:
        x_dev = _upload(runtime, x)
        gate_dev = tuple(_upload(runtime, value) for value in gate)
        up_dev = tuple(_upload(runtime, value) for value in up)
        control_gate_dev = malloc(out_features * 2, runtime=runtime)
        control_up_dev = malloc(out_features * 2, runtime=runtime)
        control_out_dev = malloc(out_features * 2, runtime=runtime)
        candidate_out_dev = malloc(out_features * 2, runtime=runtime)
        buffers.extend(
            (
                x_dev,
                *gate_dev,
                *up_dev,
                control_gate_dev,
                control_up_dev,
                control_out_dev,
                candidate_out_dev,
            )
        )

        def control() -> None:
            gguf_q4_k_pack8_dual_prefill_bf16_bf16_out(
                x_dev.ptr,
                gate_dev[0].ptr,
                gate_dev[1].ptr,
                gate_dev[2].ptr,
                up_dev[0].ptr,
                up_dev[1].ptr,
                up_dev[2].ptr,
                control_gate_dev.ptr,
                control_up_dev.ptr,
                1,
                in_features,
                out_features,
                threads=32,
                library=q4_library,
                runtime=runtime,
            )
            silu_mul_separate_out_bf16(
                control_gate_dev.ptr,
                control_up_dev.ptr,
                control_out_dev.ptr,
                1,
                out_features,
                library=silu_library,
                runtime=runtime,
            )

        def candidate() -> None:
            gguf_q4_k_pack8_dual_silu_bf16_bf16_out(
                x_dev.ptr,
                gate_dev[0].ptr,
                gate_dev[1].ptr,
                gate_dev[2].ptr,
                up_dev[0].ptr,
                up_dev[1].ptr,
                up_dev[2].ptr,
                candidate_out_dev.ptr,
                1,
                in_features,
                out_features,
                threads=32,
                library=q4_library,
                runtime=runtime,
            )

        launchers = {"control": control, "candidate": candidate}
        for _ in range(warmups):
            control()
            candidate()
        runtime.device_synchronize()
        timings = {"control": [], "candidate": []}
        paired_wins = 0
        for sample in range(samples):
            order = (
                ("control", "candidate")
                if sample % 2 == 0
                else ("candidate", "control")
            )
            pair = {}
            for name in order:
                pair[name] = _event_ms(runtime, launchers[name], burst=burst)
                timings[name].append(pair[name])
            paired_wins += int(pair["candidate"] < pair["control"])

        control()
        candidate()
        runtime.device_synchronize()
        control_out = _read_bf16(
            runtime, control_out_dev, (1, out_features)
        )
        candidate_out = _read_bf16(
            runtime, candidate_out_dev, (1, out_features)
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
        "candidate_wins": paired_wins,
        "pair_count": samples,
        "bf16_mismatches": int(
            np.count_nonzero(candidate_out != control_out)
        ),
        "candidate_sha256": hashlib.sha256(
            candidate_out.astype("<u2").tobytes()
        ).hexdigest(),
    }


def main() -> int:
    args = _parse_args()
    if _tracked_status() and not args.allow_dirty:
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
    rng = np.random.default_rng(args.seed)
    x = _bf16_bits(
        rng.normal(0.0, 0.55, size=(1, 3072)).astype(np.float32)
    )
    runtime = get_hip_runtime()
    q4_library = build_gguf_q4_k_gemv(
        load=True,
        require_cached=args.require_cached_build,
    )
    silu_library = build_paro_silu(
        load=True,
        require_cached=args.require_cached_build,
    )
    cases = {
        "shared_m1_k3072_n1024": (
            "layers.1.ffn_gate_shexp",
            "layers.1.ffn_up_shexp",
            1024,
        ),
        "dense_m1_k3072_n12288": (
            "layers.0.ffn_gate",
            "layers.0.ffn_up",
            12288,
        ),
    }
    results = {}
    for name, (gate_key, up_key, out_features) in cases.items():
        results[name] = _screen(
            runtime=runtime,
            q4_library=q4_library,
            silu_library=silu_library,
            x=x,
            gate=_load_pack8(
                args.repacked_cache, manifest, key=gate_key
            ),
            up=_load_pack8(args.repacked_cache, manifest, key=up_key),
            in_features=3072,
            out_features=out_features,
            warmups=args.warmups,
            samples=args.samples,
            burst=args.burst,
        )

    exact = all(row["bf16_mismatches"] == 0 for row in results.values())
    all_positive = all(
        row["candidate_over_control"] < 1.0 for row in results.values()
    )
    artifact = {
        "schema": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "kind": "hipengine_laguna_q4_pack8_dual_silu_decode_leaf",
        "status": "candidate" if exact and all_positive else "rejected",
        "performance_claim": False,
        "repo_revision": subprocess.check_output(
            ("git", "rev-parse", "HEAD"), cwd=ROOT, text=True
        ).strip(),
        "hardware": {
            "gpu": "AMD Radeon 8060S Graphics",
            "architecture": "gfx1151",
        },
        "protocol": {
            "warmups": args.warmups,
            "samples": args.samples,
            "burst": args.burst,
            "timing": "counterbalanced HIP-event elapsed time",
            "weights": "actual resident pack8 layer-0 dense and layer-1 shared",
            "activation": "deterministic synthetic BF16 post-norm-shaped row",
        },
        "results": results,
        "correctness_pass": exact,
        "all_shapes_positive": all_positive,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0 if exact else 1


if __name__ == "__main__":
    raise SystemExit(main())
