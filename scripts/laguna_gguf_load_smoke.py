#!/usr/bin/env python3
"""Run an admitted Laguna GGUF full-weight load/free lifecycle smoke."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from hipengine.core.hip import HipRuntime, get_hip_runtime
from hipengine.core.memory import memory_stats, reset_memory_stats
from hipengine.loading.laguna_gguf_materialize import (
    LagunaGGUFResidentWeights,
    materialize_laguna_gguf_weights,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="completed Laguna target GGUF")
    parser.add_argument("--context-length", type=int, default=4_096)
    parser.add_argument("--backend", default="hip_gfx1151")
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--output", type=Path, help="optional JSON result path")
    parser.add_argument(
        "--sysfs-device",
        type=Path,
        help="optional /sys/class/drm/cardN/device path (auto-detected by largest GTT)",
    )
    return parser


def _read_int(path: Path | None) -> int | None:
    if path is None:
        return None
    try:
        return int(path.read_text().strip())
    except (FileNotFoundError, OSError, ValueError):
        return None


def _rss_bytes() -> int:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1_024
    return 0


def _detect_sysfs_device(explicit: Path | None) -> Path | None:
    if explicit is not None:
        return explicit
    candidates: list[tuple[int, Path]] = []
    for path in Path("/sys/class/drm").glob("card*/device"):
        total = _read_int(path / "mem_info_gtt_total")
        if total is not None:
            candidates.append((total, path))
    return max(candidates, default=(0, None), key=lambda item: item[0])[1]


def _snapshot(runtime: HipRuntime, sysfs_device: Path | None) -> dict[str, Any]:
    free_bytes, total_bytes = runtime.mem_get_info()
    return {
        "hip_free_bytes": free_bytes,
        "hip_total_bytes": total_bytes,
        "gtt_used_bytes": _read_int(
            None if sysfs_device is None else sysfs_device / "mem_info_gtt_used"
        ),
        "vram_used_bytes": _read_int(
            None if sysfs_device is None else sysfs_device / "mem_info_vram_used"
        ),
        "rss_bytes": _rss_bytes(),
        "memory_stats": memory_stats(),
    }


def main() -> int:
    args = _parser().parse_args()
    if args.context_length <= 0:
        raise SystemExit("--context-length must be positive")
    if args.progress_every <= 0:
        raise SystemExit("--progress-every must be positive")
    if not args.model.is_file():
        raise SystemExit(f"model is not a file: {args.model}")

    runtime = get_hip_runtime()
    sysfs_device = _detect_sysfs_device(args.sysfs_device)
    reset_memory_stats()
    before = _snapshot(runtime, sysfs_device)
    peak = {
        "gtt_used_bytes": before["gtt_used_bytes"],
        "rss_bytes": before["rss_bytes"],
    }
    resident: LagunaGGUFResidentWeights | None = None
    started = time.monotonic()
    print("START", json.dumps(before, sort_keys=True), flush=True)

    def progress(index: int, total: int, spec: Any) -> None:
        snapshot = _snapshot(runtime, sysfs_device)
        for key in peak:
            value = snapshot[key]
            if value is not None:
                prior = peak[key]
                peak[key] = value if prior is None else max(prior, value)
        if index == 1 or index % args.progress_every == 0 or index == total:
            print(
                f"PROGRESS {index}/{total} elapsed_s={time.monotonic() - started:.3f} "
                f"gtt={snapshot['gtt_used_bytes']} rss={snapshot['rss_bytes']} "
                f"slot={spec.slot_path} layout={spec.layout} "
                f"resident={spec.resident_nbytes}",
                flush=True,
            )

    try:
        resident = materialize_laguna_gguf_weights(
            args.model,
            context_length=args.context_length,
            available_bytes=int(before["hip_free_bytes"]),
            runtime=runtime,
            backend=args.backend,
            progress=progress,
        )
        loaded = _snapshot(runtime, sysfs_device)
        for key in peak:
            value = loaded[key]
            if value is not None:
                prior = peak[key]
                peak[key] = value if prior is None else max(prior, value)
        load_seconds = time.monotonic() - started
        resident_nbytes = resident.resident_nbytes
        print(
            "LOAD_COMPLETE",
            json.dumps(
                {
                    "seconds": load_seconds,
                    "resident_nbytes": resident_nbytes,
                    "snapshot": loaded,
                    "peak": peak,
                },
                sort_keys=True,
            ),
            flush=True,
        )

        free_started = time.monotonic()
        resident.free(runtime=runtime)
        resident = None
        free_seconds = time.monotonic() - free_started
        after = _snapshot(runtime, sysfs_device)
        result = {
            "model": str(args.model.resolve()),
            "backend": args.backend,
            "context_length": args.context_length,
            "sysfs_device": None if sysfs_device is None else str(sysfs_device),
            "before": before,
            "loaded": loaded,
            "after": after,
            "peak_sampled": peak,
            "resident_nbytes": resident_nbytes,
            "load_seconds": load_seconds,
            "free_seconds": free_seconds,
            "tracked_recovered": after["memory_stats"]["current_allocated_bytes"]
            == before["memory_stats"]["current_allocated_bytes"],
            "tracked_allocations_recovered": after["memory_stats"][
                "active_allocations"
            ]
            == before["memory_stats"]["active_allocations"],
        }
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print("FREE_COMPLETE", json.dumps(result, sort_keys=True), flush=True)
        return 0 if result["tracked_recovered"] and result["tracked_allocations_recovered"] else 1
    finally:
        if resident is not None:
            resident.free(runtime=runtime)


if __name__ == "__main__":
    raise SystemExit(main())
