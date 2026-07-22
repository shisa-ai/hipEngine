#!/usr/bin/env python3
"""Run an admitted Laguna GGUF full-weight load/free lifecycle smoke."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import json
import resource
import sys
import time
from pathlib import Path
from typing import Any

from hipengine.core.hip import HipRuntime, get_hip_runtime
from hipengine.core.memory import memory_stats, reset_memory_stats
from hipengine.loading.laguna_gguf_materialize import (
    LagunaGGUFMaterializationProfile,
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
        "--profile-tensors",
        action="store_true",
        help="record natural-path read/fault, repack, allocation, and upload telemetry",
    )
    parser.add_argument(
        "--cache-state",
        choices=("cold", "warm", "unspecified"),
        default="unspecified",
        help="label the source page-cache state; this command does not modify caches",
    )
    parser.add_argument(
        "--model-sha256",
        help="precomputed model SHA-256 provenance (the loader does not rehash 70 GiB)",
    )
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


def _status_bytes(field: str) -> int:
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith(f"{field}:"):
                return int(line.split()[1]) * 1_024
    except (FileNotFoundError, OSError, ValueError):
        pass
    return 0


def _proc_io() -> dict[str, int]:
    counters: dict[str, int] = {}
    try:
        for line in Path("/proc/self/io").read_text(encoding="utf-8").splitlines():
            name, value = line.split(":", 1)
            counters[name] = int(value.strip())
    except (FileNotFoundError, OSError, ValueError):
        pass
    return counters


def _process_snapshot() -> dict[str, Any]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "minor_faults": int(usage.ru_minflt),
        "major_faults": int(usage.ru_majflt),
        "max_rss_bytes": max(
            _status_bytes("VmRSS"),
            int(usage.ru_maxrss) * 1_024,
        ),
        "io": _proc_io(),
    }


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
        "rss_bytes": _status_bytes("VmRSS"),
        "memory_stats": memory_stats(),
        "process": _process_snapshot(),
    }


def _optional_delta(before: int | None, after: int | None) -> int | None:
    if before is None or after is None:
        return None
    return after - before


def _process_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    before_io = before.get("io", {})
    after_io = after.get("io", {})
    io_names = sorted(set(before_io) | set(after_io))
    return {
        "minor_faults": int(after["minor_faults"]) - int(before["minor_faults"]),
        "major_faults": int(after["major_faults"]) - int(before["major_faults"]),
        "io": {name: _optional_delta(before_io.get(name), after_io.get(name)) for name in io_names},
    }


def _observed_cache_state(read_bytes: int | None, source_nbytes: int) -> str:
    if read_bytes is None or source_nbytes <= 0:
        return "unknown"
    ratio = read_bytes / source_nbytes
    if ratio >= 0.8:
        return "cold_streamed"
    if ratio <= 0.1:
        return "warm_cached"
    return "partially_cached"


def _profile_summary(
    profiles: list[LagunaGGUFMaterializationProfile],
) -> dict[str, Any]:
    seconds_fields = (
        "source_map_seconds",
        "repack_seconds",
        "allocation_seconds",
        "upload_seconds",
        "other_seconds",
        "total_seconds",
    )
    integer_fields = (
        "source_nbytes",
        "resident_nbytes",
        "allocation_count",
        "upload_count",
        "allocated_nbytes",
        "uploaded_nbytes",
        "minor_faults",
        "major_faults",
    )
    grouped: dict[str, list[LagunaGGUFMaterializationProfile]] = defaultdict(list)
    for profile in profiles:
        grouped[profile.layout].append(profile)

    def summarize(rows: list[LagunaGGUFMaterializationProfile]) -> dict[str, Any]:
        result: dict[str, Any] = {"tensor_count": len(rows)}
        result.update(
            {field: sum(float(getattr(row, field)) for row in rows) for field in seconds_fields}
        )
        result.update(
            {field: sum(int(getattr(row, field)) for row in rows) for field in integer_fields}
        )
        read_values = [row.read_bytes for row in rows]
        result["read_bytes"] = (
            None if any(value is None for value in read_values) else sum(read_values)
        )
        total_seconds = float(result["total_seconds"])
        result["source_bytes_per_second"] = (
            float(result["source_nbytes"]) / total_seconds if total_seconds else None
        )
        result["resident_bytes_per_second"] = (
            float(result["resident_nbytes"]) / total_seconds if total_seconds else None
        )
        return result

    slowest = sorted(profiles, key=lambda row: row.total_seconds, reverse=True)[:20]
    return {
        "overall": summarize(profiles),
        "by_layout": {layout: summarize(rows) for layout, rows in sorted(grouped.items())},
        "slowest_tensors": [
            {
                "slot_path": row.slot_path,
                "tensor_name": row.tensor_name,
                "layout": row.layout,
                "source_nbytes": row.source_nbytes,
                "resident_nbytes": row.resident_nbytes,
                "total_seconds": row.total_seconds,
                "repack_seconds": row.repack_seconds,
                "allocation_seconds": row.allocation_seconds,
                "upload_seconds": row.upload_seconds,
                "minor_faults": row.minor_faults,
                "major_faults": row.major_faults,
                "read_bytes": row.read_bytes,
            }
            for row in slowest
        ],
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
    profiles: list[LagunaGGUFMaterializationProfile] = []
    profile_rows: list[dict[str, Any]] = []
    last_profile_snapshot: tuple[str, dict[str, Any]] | None = None
    resident: LagunaGGUFResidentWeights | None = None
    started = time.monotonic()
    print("START", json.dumps(before, sort_keys=True), flush=True)

    def update_peak(snapshot: dict[str, Any]) -> None:
        for key in peak:
            value = snapshot[key]
            if value is not None:
                prior = peak[key]
                peak[key] = value if prior is None else max(prior, value)

    def record_profile(profile: LagunaGGUFMaterializationProfile) -> None:
        nonlocal last_profile_snapshot
        snapshot = _snapshot(runtime, sysfs_device)
        update_peak(snapshot)
        profiles.append(profile)
        profile_rows.append({**asdict(profile), "snapshot_after": snapshot})
        last_profile_snapshot = (profile.slot_path, snapshot)

    def progress(index: int, total: int, spec: Any) -> None:
        if last_profile_snapshot is not None and last_profile_snapshot[0] == spec.slot_path:
            snapshot = last_profile_snapshot[1]
        else:
            snapshot = _snapshot(runtime, sysfs_device)
            update_peak(snapshot)
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
            profile=record_profile if args.profile_tensors else None,
        )
        loaded = _snapshot(runtime, sysfs_device)
        for key in peak:
            value = loaded[key]
            if value is not None:
                prior = peak[key]
                peak[key] = value if prior is None else max(prior, value)
        load_seconds = time.monotonic() - started
        resident_nbytes = resident.resident_nbytes
        source_nbytes = resident.admission.weights.source_nbytes
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
        load_process_delta = _process_delta(before["process"], loaded["process"])
        read_bytes = load_process_delta["io"].get("read_bytes")
        result = {
            "schema": 2,
            "command": [sys.executable, *sys.argv],
            "model": str(args.model.resolve()),
            "model_artifact": {
                "path": str(args.model.resolve()),
                "size_bytes": args.model.stat().st_size,
                "sha256": args.model_sha256,
            },
            "backend": args.backend,
            "cache_state_declared": args.cache_state,
            "cache_state_observed": _observed_cache_state(read_bytes, source_nbytes),
            "context_length": args.context_length,
            "sysfs_device": None if sysfs_device is None else str(sysfs_device),
            "before": before,
            "loaded": loaded,
            "after": after,
            "peak_sampled": peak,
            "resident_nbytes": resident_nbytes,
            "load_seconds": load_seconds,
            "free_seconds": free_seconds,
            "load_process_delta": load_process_delta,
            "lifecycle_process_delta": _process_delta(before["process"], after["process"]),
            "tensor_profile_enabled": args.profile_tensors,
            "tensor_profile_summary": _profile_summary(profiles),
            "tensor_profiles": profile_rows,
            "tracked_recovered": after["memory_stats"]["current_allocated_bytes"]
            == before["memory_stats"]["current_allocated_bytes"],
            "tracked_allocations_recovered": after["memory_stats"]["active_allocations"]
            == before["memory_stats"]["active_allocations"],
        }
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        console_result = {key: value for key, value in result.items() if key != "tensor_profiles"}
        console_result["tensor_profile_count"] = len(profile_rows)
        print("FREE_COMPLETE", json.dumps(console_result, sort_keys=True), flush=True)
        return 0 if result["tracked_recovered"] and result["tracked_allocations_recovered"] else 1
    finally:
        if resident is not None:
            resident.free(runtime=runtime)


if __name__ == "__main__":
    raise SystemExit(main())
