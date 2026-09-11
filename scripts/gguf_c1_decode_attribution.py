#!/usr/bin/env python3
"""Attribute C1 GGUF decode time on the shipping INT8-KV route.

Roadmap P5 asks for the C1 decode boundary to be attributed across
"kernel-family, HIP API, queue-gap, D2H/H2D, telemetry, and host CPU", and for
graph capture cost and its amortization to be stated explicitly rather than
implied. The existing attribution artifacts do not cover this: the kernel-family
artifact is a 1,500-row *prefill* trace and the HIP-API gap artifact is a
c8/specdec route.

Method. A child process runs the raw resident session (the ladder's R0 arm),
prefills, discards warmup steps, then wraps each measured decode step in a
ROCTx range. ``rocprofv3`` traces kernels, markers, HIP runtime calls, and
memory copies. The reduction slices kernels and API calls to the measured
decode-step windows, so model load, prefill, and warmup are excluded.

The decomposition is deliberately additive so it cannot quietly fail to balance:

    step wall = kernel device union + HIP API time + residual host/queue time

"Residual host/queue time" is whatever the measured window spends neither on the
device nor inside a traced HIP runtime call: Python, telemetry, scheduling, and
any queue gap that is not covered by an API call. It is reported as a residual
rather than attributed to a guessed cause.

Copy byte counts are *not* available: rocprofv3's memory-copy and HIP API traces
record direction and duration but no transfer size. Copies are reported by
direction with counts and time, and byte accounting is left to the route's own
counters rather than invented here.
"""

from __future__ import annotations

import argparse
import collections
import csv
import ctypes
import json
import os
import shlex
import shutil
import statistics
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.core import build as _hipengine_build  # noqa: E402
from hipengine.kernels.backends import hip_target_arch_for_backend  # noqa: E402

ARTIFACT_KIND = "w7900_p5_c1_decode_attribution"
DEFAULT_MODEL = Path("/models/gguf/Qwen3.8-27B-Q4_K_M.gguf")
DEFAULT_COMPILER_VERSION_FILE = Path("/tmp/hipengine-hipcc-version.txt")
DEFAULT_TRACE_ROOT = Path("/tmp/hipengine-p5-c1-decode-attribution")

# Taken from the module that reads them. A hardcoded lookalike silently does
# nothing and leaves the profiled run free to spawn hipcc.
REQUIRE_CACHED_BUILD_ENV = _hipengine_build._ENV_REQUIRE_CACHED_BUILD
COMPILER_VERSION_FILE_ENV = "HIPENGINE_COMPILER_VERSION_FILE"

STEP_MARKER_PREFIX = "hipengine_c1_decode_step_"
PROMPT_ROWS = 2048
DECODE_TOKENS = 63
WARMUP_STEPS = 8
MAX_SEQUENCE_LENGTH = 16384

_FAMILY_ORDER = (
    "gguf_q4_k_t16_dense_wmma",
    "gguf_q4_k_t16_gemv_rowtile",
    "gguf_q6_k_t16_gemv_rowtile",
    "gguf_q4_k_q8_1_mmq",
    "paged_full_attn_decode",
    "paged_attn_prefill",
    "gdn_linear_attention",
    "sampler_top1",
    "rocclr_copy",
    "other",
)


# --------------------------------------------------------------------------
# Child: run the R0 decode boundary under per-step ROCTx markers.
# --------------------------------------------------------------------------


class _Roctx:
    def __init__(self) -> None:
        try:
            self._lib = ctypes.CDLL("libroctx64.so")
        except OSError as exc:  # pragma: no cover - depends on profiler SDK
            raise RuntimeError(
                f"libroctx64.so is required for decode marker windows: {exc}"
            ) from exc
        self._push = getattr(self._lib, "roctxRangePushA", None)
        self._pop = getattr(self._lib, "roctxRangePop", None)
        if self._push is None or self._pop is None:
            raise RuntimeError(
                "libroctx64.so does not expose roctxRangePushA/roctxRangePop"
            )
        self._push.argtypes = [ctypes.c_char_p]
        self._push.restype = ctypes.c_int
        self._pop.argtypes = []
        self._pop.restype = ctypes.c_int

    def push(self, name: str) -> None:
        self._push(name.encode("utf-8"))

    def pop(self) -> None:
        self._pop()


def _prompt_tokens(rows: int) -> list[int]:
    """The parity manifest's deterministic generator (rng 20260909)."""

    import numpy as np

    rng = np.random.default_rng(20260909)
    return [int(t) for t in rng.integers(1000, 4096, size=int(rows))]


def run_child(args: argparse.Namespace) -> int:
    from hipengine.core.hip import get_hip_runtime
    from hipengine.kvcache import resolve_kv_policy
    from hipengine.runtime.prefill import PrefillConfig
    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession

    runtime = get_hip_runtime()
    policy = resolve_kv_policy("int8_per_token_head", scale_dtype="fp32")
    prompt = _prompt_tokens(int(args.prompt_rows))
    marker = _Roctx()
    walls: list[float] = []

    # A synchronous hipMemcpy blocks the host until the stream drains, so its
    # traced duration is mostly queue wait rather than transfer. The trace alone
    # cannot name the call site, so record one here for calls made inside a
    # measured step.
    sync_callsites: dict[str, dict[str, int]] = collections.defaultdict(
        lambda: {"calls": 0, "total_ns": 0}
    )
    state = {"in_step": False}
    original_memcpy = runtime.memcpy

    def traced_memcpy(dst, src, nbytes, kind):  # type: ignore[no-untyped-def]
        if not state["in_step"]:
            return original_memcpy(dst, src, nbytes, kind)
        started = time.perf_counter()
        try:
            return original_memcpy(dst, src, nbytes, kind)
        finally:
            elapsed_ns = int((time.perf_counter() - started) * 1e9)
            frames = traceback.extract_stack()
            # Skip the wrapper frames in hipengine.core so the recorded site is the
            # caller that decided to copy, not the generic helper it went through.
            caller = frames[-2]
            for frame in reversed(frames[:-1]):
                if "hipengine/core" not in frame.filename.replace("\\", "/"):
                    caller = frame
                    break
            site = f"{Path(caller.filename).name}:{caller.lineno} {caller.name}"
            entry = sync_callsites[site]
            entry["calls"] += 1
            entry["total_ns"] += elapsed_ns

    runtime.memcpy = traced_memcpy  # type: ignore[method-assign]
    try:
        with Qwen35GGUFResidentSession(
            args.model,
            runtime=runtime,
            max_sequence_length=int(args.max_sequence_length),
            prefill_config=PrefillConfig(),
            kv_policy=policy.create_policy(),
            kv_scale_dtype="fp32",
            kv_scale_granularity=policy.scale_granularity,
            use_wmma_prefill=True,
            use_gemv_decode=True,
        ) as session:
            result = session.prefill(prompt, use_bulk=True, return_logits=False)
            token = int(result.token_id)
            for _ in range(int(args.warmup_steps)):
                token = int(session.step(token, return_logits=False).token_id)
            for index in range(int(args.decode_tokens) - 1):
                marker.push(f"{STEP_MARKER_PREFIX}{index}")
                state["in_step"] = True
                started = time.perf_counter()
                try:
                    step = session.step(token, return_logits=False)
                finally:
                    state["in_step"] = False
                    marker.pop()
                walls.append(time.perf_counter() - started)
                token = int(step.token_id)
    finally:
        runtime.memcpy = original_memcpy  # type: ignore[method-assign]
    payload = {
        "kind": "gguf_c1_decode_attribution_child",
        "decode_tokens": int(args.decode_tokens),
        "warmup_steps": int(args.warmup_steps),
        "prompt_rows": len(prompt),
        "measured_steps": len(walls),
        "step_walls_ms": [round(value * 1e3, 3) for value in walls],
        "final_token_id": int(token),
        "sync_memcpy_callsites": [
            {
                "site": site,
                "calls": entry["calls"],
                "total_ns": entry["total_ns"],
                "median_us": round(entry["total_ns"] / entry["calls"] / 1e3, 2),
            }
            for site, entry in sorted(
                sync_callsites.items(), key=lambda item: -item[1]["total_ns"]
            )
        ],
    }
    print(json.dumps(payload, allow_nan=False))
    return 0


# --------------------------------------------------------------------------
# Trace parsing and reduction (pure functions over CSV paths).
# --------------------------------------------------------------------------


def classify_kernel(name: str) -> str:
    """Bucket a GGUF kernel name into a reporting family."""

    lowered = name.lower()
    if "rocclr_copybuffer" in lowered or "copybuffer" in lowered:
        return "rocclr_copy"
    if "decode_split_k_reduce" in lowered or "decode_split_k" in lowered:
        return "paged_full_attn_decode"
    if "prefill" in lowered and "attn" in lowered:
        return "paged_attn_prefill"
    if "gdn" in lowered or "ssm" in lowered or "conv" in lowered:
        return "gdn_linear_attention"
    if "sampler" in lowered or "top1" in lowered or "argmax" in lowered:
        return "sampler_top1"
    if "mmq" in lowered or "q8_1" in lowered:
        return "gguf_q4_k_q8_1_mmq"
    if "q6_k" in lowered and ("gemv" in lowered or "rowtile" in lowered):
        return "gguf_q6_k_t16_gemv_rowtile"
    if "gemv" in lowered or "rowtile" in lowered:
        return "gguf_q4_k_t16_gemv_rowtile"
    if "wmma" in lowered or "dense" in lowered:
        return "gguf_q4_k_t16_dense_wmma"
    return "other"


def read_marker_windows(path: Path) -> list[dict[str, int]]:
    """Decode-step windows from the marker trace, sorted by step index."""

    windows: list[dict[str, int]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            name = (
                row.get("Function")
                or row.get("Marker_Name")
                or row.get("Marker_Text")
                or row.get("Name")
                or ""
            ).strip()
            if not name.startswith(STEP_MARKER_PREFIX):
                continue
            try:
                index = int(name.removeprefix(STEP_MARKER_PREFIX))
                start = int(float(row["Start_Timestamp"]))
                end = int(float(row["End_Timestamp"]))
            except (KeyError, ValueError):
                continue
            if end < start:
                continue
            windows.append({"step": index, "start_ns": start, "end_ns": end})
    windows.sort(key=lambda item: item["step"])
    return windows


def read_kernels(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                start = int(float(row["Start_Timestamp"]))
                end = int(float(row["End_Timestamp"]))
            except (KeyError, ValueError):
                continue
            if end < start:
                continue
            name = (
                row.get("Kernel_Name") or row.get("KernelName") or row.get("Name") or ""
            ).strip()
            rows.append(
                {
                    "kernel": name,
                    "family": classify_kernel(name),
                    "start_ns": start,
                    "end_ns": end,
                    "duration_ns": end - start,
                }
            )
    return rows


def read_hip_api(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                start = int(float(row["Start_Timestamp"]))
                end = int(float(row["End_Timestamp"]))
            except (KeyError, ValueError):
                continue
            if end < start:
                continue
            function = str(row.get("Function") or "").strip()
            rows.append(
                {
                    "function": function.split("(")[0],
                    "correlation_id": _optional_int(row.get("Correlation_Id")),
                    "start_ns": start,
                    "end_ns": end,
                    "duration_ns": end - start,
                }
            )
    return rows


def read_memory_copies(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                start = int(float(row["Start_Timestamp"]))
                end = int(float(row["End_Timestamp"]))
            except (KeyError, ValueError):
                continue
            if end < start:
                continue
            rows.append(
                {
                    "direction": str(row.get("Direction") or "").strip(),
                    "correlation_id": _optional_int(row.get("Correlation_Id")),
                    "start_ns": start,
                    "end_ns": end,
                    "duration_ns": end - start,
                }
            )
    return rows


def _optional_int(value: object) -> int | None:
    try:
        return int(float(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _clip_to_windows(
    intervals: Iterable[tuple[int, int]], windows: Sequence[Mapping[str, int]]
) -> list[tuple[int, int]]:
    """Intersect intervals with the measured windows.

    Testing for overlap alone is not enough: an interval that starts before a
    window and ends inside it would otherwise contribute its full length and the
    additive identity could exceed the measured span. Clipping makes every
    contribution a subset of the windows by construction.
    """

    clipped: list[tuple[int, int]] = []
    for start, end in intervals:
        for window in windows:
            window_start = int(window["start_ns"])
            window_end = int(window["end_ns"])
            overlap_start = max(start, window_start)
            overlap_end = min(end, window_end)
            if overlap_end > overlap_start:
                clipped.append((overlap_start, overlap_end))
    return clipped


def compare_api_to_copies(
    hip_api: Sequence[Mapping[str, Any]], copies: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Compare synchronous ``hipMemcpy`` duration with the transfer it covers.

    A synchronous copy blocks the host until the stream drains, so its API
    duration is mostly queue wait. Pairing each API call with its memory-copy row
    by correlation id makes that gap measurable instead of assumed.
    """

    copies_by_correlation = {
        int(row["correlation_id"]): row
        for row in copies
        if row.get("correlation_id") is not None
    }
    api_ms: list[float] = []
    copy_ms: list[float] = []
    directions: collections.Counter[str] = collections.Counter()
    for row in hip_api:
        if str(row.get("function")) != "hipMemcpy":
            continue
        correlation = row.get("correlation_id")
        if correlation is None:
            continue
        match = copies_by_correlation.get(int(correlation))
        if match is None:
            continue
        api_ms.append(int(row["duration_ns"]) / 1e6)
        copy_ms.append(int(match["duration_ns"]) / 1e6)
        directions[str(match["direction"])] += 1
    if not api_ms:
        return {
            "matched_calls": 0,
            "note": "no hipMemcpy call had a matching memory-copy row",
        }
    median_api = statistics.median(api_ms)
    median_copy = statistics.median(copy_ms)
    return {
        "matched_calls": len(api_ms),
        "median_api_ms": round(median_api, 3),
        "median_transfer_ms": round(median_copy, 3),
        "api_over_transfer_ratio": (
            round(median_api / median_copy, 1) if median_copy > 0 else None
        ),
        "max_api_ms": round(max(api_ms), 3),
        "directions": dict(directions),
        "note": (
            "a synchronous hipMemcpy waits for the stream to drain, so its API "
            "duration includes queue wait that the transfer itself does not; the "
            "ratio shows how much of the call is wait rather than data movement"
        ),
    }


def _in_windows(start_ns: int, end_ns: int, windows: Sequence[Mapping[str, int]]) -> bool:
    """True when the interval overlaps any measured window at all."""

    return any(
        start_ns < int(window["end_ns"]) and end_ns > int(window["start_ns"])
        for window in windows
    )


def _union_ns(intervals: Iterable[tuple[int, int]]) -> int:
    """Total length covered by a set of intervals, counting overlaps once."""

    ordered = sorted(intervals)
    total = 0
    current_start: int | None = None
    current_end: int | None = None
    for start, end in ordered:
        if current_start is None or current_end is None:
            current_start, current_end = start, end
            continue
        if start > current_end:
            total += current_end - current_start
            current_start, current_end = start, end
        else:
            current_end = max(current_end, end)
    if current_start is not None and current_end is not None:
        total += current_end - current_start
    return total


def summarize(
    *,
    windows: Sequence[Mapping[str, int]],
    kernels: Sequence[Mapping[str, Any]],
    hip_api: Sequence[Mapping[str, Any]],
    copies: Sequence[Mapping[str, Any]],
    step_walls_ms: Sequence[float],
    sync_callsites: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Build the attribution blocks from traced intervals.

    Every interval is clipped to the measured windows, so each reported total is
    a subset of the measured span and the additive identity is a real constraint
    rather than a definition that cannot fail.
    """

    if not windows:
        raise ValueError(
            "no decode-step marker windows were traced; the child must emit "
            f"{STEP_MARKER_PREFIX}<index> ranges"
        )
    window_intervals = [
        (int(w["start_ns"]), int(w["end_ns"])) for w in windows
    ]
    window_ns = _union_ns(window_intervals)

    in_window_kernels = [
        row for row in kernels if _in_windows(row["start_ns"], row["end_ns"], windows)
    ]
    clipped_kernels: list[tuple[str, int, int]] = []
    for row in in_window_kernels:
        for start, end in _clip_to_windows(
            [(int(row["start_ns"]), int(row["end_ns"]))], windows
        ):
            clipped_kernels.append((str(row["family"]), start, end))

    kernel_time_sum_ns = sum(end - start for _family, start, end in clipped_kernels)
    device_union_ns = _union_ns((start, end) for _family, start, end in clipped_kernels)

    by_family: dict[str, dict[str, int]] = collections.defaultdict(
        lambda: {"launches": 0, "duration_ns": 0}
    )
    for family, start, end in clipped_kernels:
        entry = by_family[family]
        entry["launches"] += 1
        entry["duration_ns"] += end - start
    # Shares are of total kernel time, not of the device union: dispatch intervals
    # on this route overlap, so a share of the union can exceed 100% and read as a
    # broken table. The overlap factor is reported alongside instead.
    kernel_families = [
        {
            "family": family,
            "launches": counts["launches"],
            "duration_ns": counts["duration_ns"],
            "share_pct_of_kernel_time": (
                round(100.0 * counts["duration_ns"] / kernel_time_sum_ns, 2)
                if kernel_time_sum_ns
                else 0.0
            ),
        }
        for family, counts in sorted(
            by_family.items(), key=lambda item: (-item[1]["duration_ns"], item[0])
        )
    ]

    in_window_api = [
        row for row in hip_api if _in_windows(row["start_ns"], row["end_ns"], windows)
    ]
    api_intervals = _clip_to_windows(
        ((int(row["start_ns"]), int(row["end_ns"])) for row in in_window_api), windows
    )
    api_union_ns = _union_ns(api_intervals)
    api_by_function: dict[str, dict[str, int]] = collections.defaultdict(
        lambda: {"calls": 0, "duration_ns": 0}
    )
    for row in in_window_api:
        entry = api_by_function[str(row["function"])]
        entry["calls"] += 1
        entry["duration_ns"] += int(row["duration_ns"])
    hip_api_top = [
        {
            "function": function,
            "calls": counts["calls"],
            "duration_ns": counts["duration_ns"],
        }
        for function, counts in sorted(
            api_by_function.items(), key=lambda item: (-item[1]["duration_ns"], item[0])
        )[:12]
    ]

    copies_by_direction: dict[str, dict[str, int]] = collections.defaultdict(
        lambda: {"copies": 0, "duration_ns": 0}
    )
    in_window_copies = 0
    for row in copies:
        clipped = _clip_to_windows(
            [(int(row["start_ns"]), int(row["end_ns"]))], windows
        )
        if not clipped:
            continue
        in_window_copies += 1
        entry = copies_by_direction[str(row["direction"])]
        entry["copies"] += 1
        entry["duration_ns"] += sum(end - start for start, end in clipped)

    # window = union(device, api) + residual, where residual is genuinely
    # unaccounted time rather than a plug: it is the measured span minus the
    # union of everything observed inside it.
    combined_union_ns = _union_ns(
        [(start, end) for _family, start, end in clipped_kernels] + api_intervals
    )
    residual_ns = window_ns - combined_union_ns
    if residual_ns < 0:
        raise ValueError(
            "attribution does not balance: combined device+API union "
            f"({combined_union_ns} ns) exceeds the measured span ({window_ns} ns)"
        )
    overlap_ns = device_union_ns + api_union_ns - combined_union_ns
    api_while_gpu_idle_ns = api_union_ns - overlap_ns
    gpu_idle_ns = window_ns - device_union_ns
    if api_while_gpu_idle_ns + residual_ns != gpu_idle_ns:
        raise ValueError(
            "GPU-idle split does not balance: "
            f"api_while_idle {api_while_gpu_idle_ns} + residual {residual_ns} "
            f"!= gpu_idle {gpu_idle_ns}"
        )

    measured_wall_ms = statistics.median(step_walls_ms) if step_walls_ms else None
    return {
        "windows": {
            "measured_steps": len(windows),
            "window_total_ns": window_ns,
            "window_total_ms": round(window_ns / 1e6, 3),
            "per_step_ms": round(window_ns / 1e6 / len(windows), 3),
            "child_step_wall_median_ms": measured_wall_ms,
        },
        "device": {
            "kernel_launches": len(clipped_kernels),
            "launches_per_step": round(len(clipped_kernels) / len(windows), 2),
            "kernel_time_sum_ns": kernel_time_sum_ns,
            "device_union_ns": device_union_ns,
            "device_union_ms": round(device_union_ns / 1e6, 3),
            "device_union_pct_of_window": (
                round(100.0 * device_union_ns / window_ns, 2) if window_ns else 0.0
            ),
            "dispatch_overlap_ratio": (
                round(kernel_time_sum_ns / device_union_ns, 3)
                if device_union_ns
                else None
            ),
            "dispatch_overlap_note": (
                "sum of dispatch durations divided by their union; a value above 1 "
                "means the trace's dispatch intervals overlap, so per-family shares "
                "are reported against kernel time and not against the union"
            ),
            "gpu_idle_ns": gpu_idle_ns,
            "gpu_idle_ms": round(gpu_idle_ns / 1e6, 3),
        },
        "kernel_families": kernel_families,
        "hip_api": {
            "calls": len(in_window_api),
            "calls_per_step": round(len(in_window_api) / len(windows), 2),
            "api_union_ns": api_union_ns,
            "api_union_ms": round(api_union_ns / 1e6, 3),
            "api_union_pct_of_window": (
                round(100.0 * api_union_ns / window_ns, 2) if window_ns else 0.0
            ),
            "top_functions": hip_api_top,
        },
        "memory_copies": {
            "copies": in_window_copies,
            "copies_per_step": round(in_window_copies / len(windows), 2),
            "bytes_available": False,
            "bytes_note": (
                "rocprofv3's memory-copy and HIP API traces record direction and "
                "duration but no transfer size, so no byte figure is reported here"
            ),
            "by_direction": [
                {
                    "direction": direction,
                    "copies": counts["copies"],
                    "duration_ns": counts["duration_ns"],
                }
                for direction, counts in sorted(
                    copies_by_direction.items(),
                    key=lambda item: (-item[1]["duration_ns"], item[0]),
                )
            ],
        },
        "sync_memcpy_stall": {
            "callsites": [dict(entry) for entry in sync_callsites],
            "calls": sum(int(entry["calls"]) for entry in sync_callsites),
            "total_ns": sum(int(entry["total_ns"]) for entry in sync_callsites),
            "per_step_calls": (
                round(
                    sum(int(entry["calls"]) for entry in sync_callsites) / len(windows), 2
                )
                if windows
                else 0.0
            ),
            "per_step_ms": (
                round(
                    sum(int(entry["total_ns"]) for entry in sync_callsites)
                    / len(windows)
                    / 1e6,
                    3,
                )
                if windows
                else 0.0
            ),
            "meaning": (
                "synchronous hipMemcpy calls made inside a measured step, timed in "
                "the child and attributed to their Python call site"
            ),
        },
        "decomposition": {
            "identity": "window = union(device, api) + residual",
            "window_ns": window_ns,
            "device_union_ns": device_union_ns,
            "hip_api_union_ns": api_union_ns,
            "overlap_device_api_ns": overlap_ns,
            "combined_union_ns": combined_union_ns,
            "residual_ns": residual_ns,
            "residual_ms": round(residual_ns / 1e6, 3),
            "residual_pct_of_window": (
                round(100.0 * residual_ns / window_ns, 2) if window_ns else 0.0
            ),
            "gpu_idle_ns": gpu_idle_ns,
            "api_while_gpu_idle_ns": api_while_gpu_idle_ns,
            "gpu_idle_split": "gpu_idle = api_while_gpu_idle + residual",
            "residual_meaning": (
                "time inside the measured window that is neither device-busy nor "
                "inside a traced HIP runtime call: Python, telemetry, scheduling, "
                "and any queue gap not covered by an API call"
            ),
            "balances": combined_union_ns + residual_ns == window_ns,
        },
    }


# --------------------------------------------------------------------------
# Environment, host, and driver.
# --------------------------------------------------------------------------


def trace_environment(
    compiler_version_file: Path, backend: str
) -> dict[str, str]:
    compiler_version = compiler_version_file.read_text(encoding="utf-8").strip()
    if not compiler_version:
        raise ValueError(f"compiler version file is empty: {compiler_version_file}")
    environment = dict(os.environ)
    environment.pop("ROCR_VISIBLE_DEVICES", None)
    environment.setdefault("HIP_VISIBLE_DEVICES", "0")
    environment["HIPENGINE_HIP_ARCH"] = hip_target_arch_for_backend(backend)
    environment[COMPILER_VERSION_FILE_ENV] = str(compiler_version_file)
    environment[REQUIRE_CACHED_BUILD_ENV] = "1"
    return environment


def _cache_relevant_environment(environment: Mapping[str, str]) -> dict[str, str]:
    keys = (
        "HIP_VISIBLE_DEVICES",
        "HIPENGINE_HIP_ARCH",
        COMPILER_VERSION_FILE_ENV,
        REQUIRE_CACHED_BUILD_ENV,
    )
    return {key: environment[key] for key in keys if key in environment}


def _default_roctx_sdk() -> Path:
    """The ROCTX library that talks to rocprofv3's marker collector.

    A ``libroctx64.so`` found on the default search path is not necessarily the
    one rocprofv3 instruments, and using the wrong one yields a run with kernel
    and API traces but no marker trace at all.
    """

    python_dir = f"python{sys.version_info.major}.{sys.version_info.minor}"
    names = ("librocprofiler-sdk-roctx.so.1", "librocprofiler-sdk-roctx.so")
    candidates = [
        Path(root) / "lib" / python_dir / "site-packages" / pkg / "lib" / name
        for root in dict.fromkeys((sys.prefix, sys.base_prefix))
        for pkg in ("_rocm_sdk_core", "_rocm_sdk_devel")
        for name in names
    ]
    candidates += [Path("/opt/rocm/lib") / name for name in names]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _prepare_roctx_override(sdk_path: Path) -> tuple[Path, tuple[Path, ...]]:
    """Expose the SDK ROCTX library as ``libroctx64.so`` on a private path."""

    if not sdk_path.exists():
        raise FileNotFoundError(f"rocprofiler SDK ROCTX library not found: {sdk_path}")
    override = Path("/tmp/hipengine-roctx-sdk-override-c1-decode-attribution")
    override.mkdir(parents=True, exist_ok=True)
    symlink = override / "libroctx64.so"
    if symlink.exists() or symlink.is_symlink():
        symlink.unlink()
    symlink.symlink_to(sdk_path)
    dependency_paths: list[Path] = [sdk_path.parent]
    sysdeps = sdk_path.parent / "rocm_sysdeps" / "lib"
    if sysdeps.is_dir():
        dependency_paths.append(sysdeps)
    return override, tuple(dependency_paths)


def _with_roctx_library(
    environment: Mapping[str, str], sdk_path: Path
) -> tuple[dict[str, str], dict[str, str]]:
    """Return the child environment plus a provenance record for the SDK."""

    override, dependencies = _prepare_roctx_override(sdk_path)
    prefix = os.pathsep.join([str(override), *(str(path) for path in dependencies)])
    updated = dict(environment)
    updated["LD_LIBRARY_PATH"] = f"{prefix}:{updated.get('LD_LIBRARY_PATH', '')}"
    provenance = {
        "sdk_library": str(sdk_path),
        "override_dir": str(override),
        "dependency_dirs": [str(path) for path in dependencies],
    }
    return updated, provenance


def _single(root: Path, pattern: str) -> Path:
    matches = sorted(root.rglob(pattern))
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {pattern} under {root}, found {len(matches)}")
    return matches[0]


def _observed_device() -> str:
    executable = shutil.which("rocm-smi")
    if executable is None:
        return "unavailable"
    completed = subprocess.run(
        [executable, "--showproductname", "--json"],
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        payload = json.loads(completed.stdout or "{}")
    except json.JSONDecodeError:
        return "unavailable"
    for card in payload.values():
        if isinstance(card, Mapping):
            for key in ("Card Series", "Card Model", "Card SKU"):
                value = card.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
    return "unavailable"


def _rocprofv3_version() -> str:
    executable = shutil.which("rocprofv3")
    if executable is None:
        return "unavailable"
    completed = subprocess.run(
        [executable, "--version"], capture_output=True, text=True, check=False
    )
    text = (completed.stdout or completed.stderr or "").strip()
    if not text:
        return "unavailable"
    first = text.splitlines()[0].strip()
    prefix = "version:"
    if first.lower().startswith(prefix):
        return first[len(prefix) :].strip() or "unavailable"
    return first


def _child_command(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child-mode",
        "run",
        "--model",
        str(args.model),
        "--prompt-rows",
        str(int(args.prompt_rows)),
        "--decode-tokens",
        str(int(args.decode_tokens)),
        "--warmup-steps",
        str(int(args.warmup_steps)),
        "--max-sequence-length",
        str(int(args.max_sequence_length)),
    ]


def run(args: argparse.Namespace) -> dict[str, Any]:
    model = args.model.expanduser().resolve()
    if not model.is_file():
        raise ValueError(f"model does not exist: {model}")
    backend = str(args.backend)
    environment = trace_environment(
        args.compiler_version_file.expanduser().resolve(), backend
    )
    sdk_path = args.roctx_sdk.expanduser().resolve() if args.roctx_sdk else _default_roctx_sdk()
    roctx_provenance: dict[str, Any] = {}
    trace_root = (
        args.trace_dir.expanduser().resolve()
        if args.trace_dir
        else DEFAULT_TRACE_ROOT
    )

    child_payload: dict[str, Any] | None = None
    command: list[str] | None = None
    if args.trace_dir is None:
        if shutil.which("rocprofv3") is None:
            raise ValueError("rocprofv3 is not on PATH; pass --trace-dir to parse an existing trace")
        if trace_root.exists():
            shutil.rmtree(trace_root)
        trace_root.mkdir(parents=True, exist_ok=True)
        # The child must load the ROCTX library that rocprofv3 instruments.
        environment, roctx_provenance = _with_roctx_library(environment, sdk_path)
        if args.warmup:
            completed = subprocess.run(
                _child_command(args),
                cwd=str(REPO_ROOT),
                env=dict(environment),
                capture_output=True,
                text=True,
                check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    "unprofiled warmup child failed: "
                    f"{completed.stderr.strip()[-400:]}"
                )
        command = [
            "rocprofv3",
            "--kernel-trace",
            "--marker-trace",
            "--hip-runtime-trace",
            "--memory-copy-trace",
            "--output-format",
            "csv",
            "-d",
            str(trace_root),
            "--",
            *_child_command(args),
        ]
        completed = subprocess.run(
            command, cwd=str(REPO_ROOT), env=dict(environment), capture_output=True,
            text=True, check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"profiled child failed with exit {completed.returncode}: "
                f"{completed.stderr.strip()[-400:]}"
            )
        for line in reversed(completed.stdout.splitlines()):
            stripped = line.strip()
            if stripped.startswith("{") and stripped.endswith("}"):
                try:
                    child_payload = json.loads(stripped)
                except json.JSONDecodeError:
                    continue
                break
        if child_payload is None:
            raise RuntimeError("profiled child printed no child JSON payload")
    elif not trace_root.is_dir():
        raise ValueError(f"trace directory does not exist: {trace_root}")

    marker_csv = _single(trace_root, "*_marker_api_trace.csv")
    windows = read_marker_windows(marker_csv)
    observed = [int(window["step"]) for window in windows]
    if command is not None:
        # Only the run we drove can be checked against the requested step count;
        # a parse-only regeneration must not silently accept a partial capture
        # either, so it still requires a contiguous 0..n-1 window set.
        expected = list(range(int(args.decode_tokens) - 1))
        if observed != expected:
            raise ValueError(
                "marker windows do not match the measured steps; a partial marker "
                f"capture would silently attribute a subset. expected {len(expected)} "
                f"steps starting at 0, observed {observed[:4]}...({len(observed)}). "
                f"Check the ROCTX SDK library at {sdk_path}."
            )
    elif observed != list(range(len(observed))):
        raise ValueError(f"marker windows are not contiguous from 0: {observed[:6]}...")
    kernels = read_kernels(_single(trace_root, "*_kernel_trace.csv"))
    hip_api = read_hip_api(_single(trace_root, "*_hip_api_trace.csv"))
    copies = read_memory_copies(_single(trace_root, "*_memory_copy_trace.csv"))

    step_walls = list((child_payload or {}).get("step_walls_ms", []))
    sync_callsites = list((child_payload or {}).get("sync_memcpy_callsites", []))
    summary = summarize(
        windows=windows,
        kernels=kernels,
        hip_api=hip_api,
        copies=copies,
        step_walls_ms=step_walls,
        sync_callsites=sync_callsites,
    )
    queue_drain = compare_api_to_copies(hip_api, copies)
    if not sync_callsites:
        summary["sync_memcpy_stall"]["callsites"] = []
        summary["sync_memcpy_stall"]["callsite_note"] = (
            "call-site attribution needs the child to instrument runtime.memcpy; a "
            "parse-only reduction of a trace produced without it cannot name the site"
        )

    if command is not None:
        trace_command = " ".join(shlex.quote(part) for part in command)
        reduction = "profiled rocprofv3 run"
        recorded_environment = _cache_relevant_environment(environment)
    else:
        trace_command = str(args.trace_command or f"unavailable (parsed from {trace_root})")
        reduction = f"parse-only {trace_root}"
        recorded_environment = {}

    graph = {
        "execution_mode": "eager (no graph capture)",
        "capture_cost_ns": None,
        "capture_cost_note": (
            "this route ran eager, so no graph was captured and no capture cost is "
            "incurred inside these windows"
        ),
        "amortization_statement": (
            "no comparison here mixes an amortized graph replay with a cold request: "
            "both the decode-step windows and the child walls are eager. A graph "
            "comparison would have to state replays per capture and exclude the "
            "capture cost from the amortized rate, and report the cold request "
            "separately."
        ),
        "cross_reference": (
            "the c8 packed-graph packet records graph_capture_seconds 0.186 with one "
            "synchronized replay per logical transition; that is a different route and "
            "width and is not reused as a C1 figure here"
        ),
    }

    notes = [
        "Decode-step windows come from per-step ROCTx ranges, so model load, prefill, "
        "and warmup are excluded from every attributed total.",
        "Every interval is clipped to the measured windows before summing, so each "
        "total is a subset of the measured span and the additive identity is a real "
        "constraint: window = union(device, api) + residual, with residual >= 0.",
        "Per-family shares are of total kernel time, not of the device union: "
        "dispatch intervals on this route overlap, so a share of the union can "
        "exceed 100% and misread as a broken table. The overlap factor is reported.",
        "Copy byte counts are not available from rocprofv3 traces; copies are reported "
        "by direction, count, and time only.",
    ]
    if recorded_environment:
        notes.insert(
            0,
            "Cache-only trace: the .so was built outside the profiler and "
            f"{REQUIRE_CACHED_BUILD_ENV}=1, with the compiler version pinned by "
            f"{COMPILER_VERSION_FILE_ENV}, prevented hipcc in the profiled process. "
            "Both are recorded in trace_environment.",
        )
    else:
        notes.insert(
            0,
            "Reduced from an existing trace; the cache-only property belongs to that "
            "run, which this parse-only path cannot observe, so no cache-only claim is "
            "made here.",
        )

    return {
        "schema": 1,
        "kind": ARTIFACT_KIND,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "performance_claim": False,
        "host": {
            "observed_device": _observed_device(),
            "target_arch": hip_target_arch_for_backend(backend),
            "hip_visible_devices": environment.get("HIP_VISIBLE_DEVICES", ""),
            "rocprofv3_version": _rocprofv3_version(),
        },
        "model": {
            "path": str(model),
            "quant": "gguf_q4_k_m",
            "kv": "int8_per_token_head + fp32 scales",
        },
        "workload": {
            "arm": "r0_raw_session",
            "prompt_rows": int(args.prompt_rows),
            "decode_tokens": int(args.decode_tokens),
            "warmup_steps": int(args.warmup_steps),
            "max_sequence_length": int(args.max_sequence_length),
            "sampler": "greedy",
            "child": child_payload,
        },
        "attribution": summary,
        "queue_drain": queue_drain,
        "graph_capture": graph,
        "roctx": roctx_provenance,
        "command": trace_command,
        "reduction": reduction,
        "trace_environment": recorded_environment,
        "notes": notes,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child-mode", choices=("run",), default=None)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--backend", default="hip_gfx1100")
    parser.add_argument("--prompt-rows", type=int, default=PROMPT_ROWS)
    parser.add_argument("--decode-tokens", type=int, default=DECODE_TOKENS)
    parser.add_argument("--warmup-steps", type=int, default=WARMUP_STEPS)
    parser.add_argument("--max-sequence-length", type=int, default=MAX_SEQUENCE_LENGTH)
    parser.add_argument(
        "--trace-dir",
        type=Path,
        default=None,
        help="parse an existing rocprofv3 output directory instead of tracing",
    )
    parser.add_argument(
        "--trace-command",
        default=None,
        help="record the command that produced an existing --trace-dir",
    )
    parser.add_argument(
        "--compiler-version-file", type=Path, default=DEFAULT_COMPILER_VERSION_FILE
    )
    parser.add_argument(
        "--roctx-sdk",
        type=Path,
        default=None,
        help="rocprofiler-sdk ROCTX library to expose as libroctx64.so for the child",
    )
    parser.add_argument(
        "--no-warmup",
        dest="warmup",
        action="store_false",
        help="skip the unprofiled cache-warming child run",
    )
    parser.set_defaults(warmup=True)
    parser.add_argument("--json", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.child_mode == "run":
        return run_child(args)
    try:
        artifact = run(args)
    except (ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    text = json.dumps(artifact, indent=2, allow_nan=False) + "\n"
    if args.json:
        args.json.expanduser().resolve().write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
