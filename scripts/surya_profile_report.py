#!/usr/bin/env python3
"""Aggregate rocprofv3 traces of the Surya HIP pipeline into a profile.

Consumes the CSV traces produced by ``rocprofv3 --kernel-trace
--memory-copy-trace`` around ``scripts/surya_gpu_rocprof_driver.py`` and answers
the question the tuning decision needs: of the wall time per decode token, how
much is actually kernel execution, which kernel families own it, and how much is
host-side gap or transfer.

The driver reports how many calls it made and the CLOCK_MONOTONIC boundaries of
its phases, so per-call cost is the kernel total *inside the phase's range*
divided by the calls made *inside* that range. That distinction matters: a
decode trace also runs the vision tower and a prefill to seed the KV state, and
rocprofv3 aggregates the whole process, so dividing the raw kernel total by the
decode step count would bill that setup to decode. Kernels outside every range
are reported separately as ``outside``. Without ranges the numbers are labelled
whole-process instead of being presented as phase costs.

A per-phase host gap is ``range wall - sum(kernel durations in range)``, which
is where launch overhead, synchronisation and transfer stalls show up.

Usage:
    python3 scripts/surya_profile_report.py \\
        --trace decode=/tmp/surya-prof-decode \\
        --trace prefill=/tmp/surya-prof-prefill \\
        --out benchmarks/results/2026-09-12-gfx1151-surya-phase-attributed-profile.json
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path


def _family(kernel: str) -> str:
    """Bucket a kernel name into the Surya family it belongs to."""

    k = kernel.lower()
    # GDN is checked before EVIE because the GDN kernels live in the EVIE library
    # but belong to the linear-attention family, not to the vision elementwise set.
    if "gdn" in k or "linear_attn" in k:
        if "conv" in k:
            return "gdn_conv"
        return "gdn_recurrent"
    if k.startswith("surya_") or "surya" in k:
        if "scatter_kv" in k:
            return "surya_kv_write"
        if "attn" in k or "attention" in k:
            return "surya_attention_decode"
        if "rope" in k or "rotary" in k:
            return "surya_rope"
        return "surya_other"
    if "evie" in k:
        if "gemm" in k or "matmul" in k:
            return "evie_gemm"
        if "layernorm" in k or "rmsnorm" in k:
            return "evie_norm"
        if "gelu" in k or "silu" in k or "act" in k:
            return "evie_activation"
        if "rotary" in k or "rope" in k:
            return "evie_rope"
        if "softmax" in k:
            return "evie_softmax"
        if "add" in k or "bias" in k:
            return "evie_elementwise"
        if "attn" in k:
            return "evie_attention"
        return "evie_other"
    if "rocblas" in k or "gemm" in k or "gemv" in k or "matmul" in k:
        return "gemm_library"
    # rocBLAS dispatches to Tensile assembly kernels named Cijk_<...>; they carry
    # no "gemm" substring, so without this they land in the catch-all and hide
    # most of the profile.
    if k.startswith("cijk_") or "tensile" in k:
        return "gemm_library"
    if "elementwise" in k or "vectorized_elementwise" in k:
        return "torch_free_elementwise"
    if "copy" in k or "memcpy" in k or "fill" in k:
        return "copy_fill"
    return "other"


def _int_or_none(value) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _read_kernels(directory: Path) -> list[dict]:
    # rocprofv3 writes its CSVs into a hostname subdirectory, so the search must
    # be recursive rather than a flat glob.
    rows: list[dict] = []
    for path in sorted(directory.rglob("*kernel_trace*.csv")):
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                start = _int_or_none(row.get("Start_Timestamp"))
                end = _int_or_none(row.get("End_Timestamp"))
                if start is None or end is None or end < start:
                    continue
                name = (
                    row.get("Kernel_Name") or row.get("KernelName")
                    or row.get("Name") or ""
                ).strip()
                if not name:
                    continue
                rows.append({
                    "kernel": name,
                    "start_ns": start,
                    "duration_ns": end - start,
                    "grid": _int_or_none(row.get("Grid_Size_X")),
                    "workgroup": _int_or_none(row.get("Workgroup_Size_X")),
                })
    return rows


def _read_copies(directory: Path) -> list[dict]:
    """Memory copies from the trace.

    ``rocprofv3 --memory-copy-trace`` records direction and duration but not a
    byte count, so volume is reported as unavailable rather than guessed; the
    direction split and the time are what the trace can support.
    """

    rows: list[dict] = []
    for path in sorted(directory.rglob("*memory_copy_trace*.csv")):
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                start = _int_or_none(row.get("Start_Timestamp"))
                end = _int_or_none(row.get("End_Timestamp"))
                if start is None or end is None or end < start:
                    continue
                kind = (
                    row.get("Direction") or row.get("Copy_Operation")
                    or row.get("Kind") or "?"
                ).strip()
                # MEMORY_COPY_HOST_TO_DEVICE -> host_to_device
                kind = kind.replace("MEMORY_COPY_", "").lower()
                rows.append({
                    "kind": kind,
                    "name": (row.get("Name") or "").strip(),
                    "start_ns": start,
                    "bytes": _int_or_none(row.get("Bytes")),
                    "duration_ns": end - start,
                })
    return rows


def _summarize(kernels: list[dict], copies: list[dict], calls: int) -> dict:
    total_ns = sum(row["duration_ns"] for row in kernels)
    by_family: dict[str, dict] = {}
    by_kernel: dict[str, dict] = {}
    for row in kernels:
        for table, key in ((by_family, _family(row["kernel"])),
                           (by_kernel, row["kernel"])):
            entry = table.setdefault(key, {"calls": 0, "total_ns": 0, "max_ns": 0})
            entry["calls"] += 1
            entry["total_ns"] += row["duration_ns"]
            entry["max_ns"] = max(entry["max_ns"], row["duration_ns"])

    def ranked(table: dict) -> list[dict]:
        out = []
        for name, entry in table.items():
            out.append({
                "name": name,
                "calls": entry["calls"],
                "total_ms": entry["total_ns"] / 1e6,
                "per_call_us": (entry["total_ns"] / entry["calls"] / 1e3)
                if entry["calls"] else None,
                "max_us": entry["max_ns"] / 1e3,
                "share_pct": (100.0 * entry["total_ns"] / total_ns) if total_ns else 0.0,
            })
        out.sort(key=lambda item: item["total_ms"], reverse=True)
        return out

    copy_ns = sum(row["duration_ns"] for row in copies)
    known = [row["bytes"] for row in copies if row["bytes"] is not None]
    by_kind: dict[str, dict] = {}
    for row in copies:
        entry = by_kind.setdefault(row["kind"], {"calls": 0, "bytes": 0, "ns": 0,
                                                "bytes_known": 0})
        entry["calls"] += 1
        entry["ns"] += row["duration_ns"]
        if row["bytes"] is not None:
            entry["bytes"] += row["bytes"]
            entry["bytes_known"] += 1

    return {
        "calls_executed": calls,
        "kernel_total_ms": total_ns / 1e6,
        "kernel_per_call_ms": (total_ns / calls / 1e6) if calls else None,
        "kernel_calls": len(kernels),
        "kernel_calls_per_call": (len(kernels) / calls) if calls else None,
        "kernel_families": ranked(by_family),
        "top_kernels": ranked(by_kernel)[:20],
        "transfers": {
            "bytes_available": bool(known),
            "total_mb": (sum(known) / 1e6) if known else None,
            "total_ms": copy_ns / 1e6,
            "ms_per_call": (copy_ns / calls / 1e6) if calls else None,
            "by_kind": {
                kind: {"calls": v["calls"], "ms": v["ns"] / 1e6,
                       "mb": (v["bytes"] / 1e6) if v["bytes_known"] else None}
                for kind, v in sorted(by_kind.items(), key=lambda kv: -kv[1]["ns"])
            },
        },
    }


def _attribute(rows: list[dict], ranges: list[dict]) -> tuple[dict[str, list[dict]], list[dict]]:
    """Bucket rows into the driver's named ranges by start timestamp.

    rocprofv3 timestamps kernels with CLOCK_MONOTONIC nanoseconds, the same
    clock the driver records its range boundaries with, so a row belongs to the
    range whose ``[start_ns, end_ns)`` contains its start. A row in no range is
    returned as ``outside`` rather than dropped or assigned to a neighbour: a
    decode trace runs the vision tower and a prefill to seed the KV state, and
    counting those as decode is the error this exists to prevent.
    """

    buckets: dict[str, list[dict]] = {r["name"]: [] for r in ranges}
    outside: list[dict] = []
    ordered = sorted(ranges, key=lambda r: r["start_ns"])
    for row in rows:
        start = row.get("start_ns")
        if start is not None:
            for r in ordered:
                if r["start_ns"] <= start < r["end_ns"]:
                    buckets[r["name"]].append(row)
                    break
            else:
                outside.append(row)
        else:
            outside.append(row)
    return buckets, outside


def _load_driver_output(directory: Path) -> dict:
    for path in sorted(directory.glob("*driver*.json")) + sorted(directory.glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        if isinstance(data, dict) and "phase" in data:
            return data
    return {}


def _attribute_phases(kernels: list[dict], copies: list[dict], calls: int,
                      ranges: list[dict], phase_range_name: str) -> dict:
    """Summarize a trace as the phase range plus its setup and outside work.

    The headline numbers come from the range named after the phase, so a decode
    trace's setup kernels are not billed to decode. Every range keeps its own
    ``calls_in_range`` because that is the divisor: vision and prefill do one
    warmup outside their range, and decode's step count is known only at the
    end of the loop. ``whole_process`` is retained for reference.
    """

    whole = _summarize(kernels, copies, calls)
    if not ranges:
        summary = dict(whole)
        summary["attribution"] = "whole_process"
        summary["whole_process"] = whole
        return summary

    kernel_buckets, outside_kernels = _attribute(kernels, ranges)
    copy_buckets, outside_copies = _attribute(copies, ranges)

    def compact(summary: dict) -> dict:
        """Drop the ranked kernel table; keep it only on the headline.

        A range is reported alongside the headline, and a 20-row table of
        500-character kernel names repeated per range would dwarf the artifact.
        Families and totals carry the attribution story.
        """

        return {k: v for k, v in summary.items() if k != "top_kernels"}

    per_range = []
    headline = None
    for entry in ranges:
        range_calls = int(entry.get("calls") or 0)
        if not range_calls:
            # Older driver output has no per-range count; the phase range is
            # the counted calls, everything else ran once.
            range_calls = calls if entry["name"] == phase_range_name else 1
        sub = _summarize(kernel_buckets[entry["name"]],
                         copy_buckets[entry["name"]], range_calls)
        sub.update({
            "name": entry["name"],
            "calls_in_range": range_calls,
            "wall_ms": (entry["end_ns"] - entry["start_ns"]) / 1e6,
            "is_phase": entry["name"] == phase_range_name,
        })
        sub["host_gap_ms"] = sub["wall_ms"] - sub["kernel_total_ms"]
        if sub["is_phase"]:
            headline = sub
        per_range.append(compact(sub))

    summary = dict(whole if headline is None else headline)
    summary["attribution"] = "range" if headline else "whole_process"
    summary["calls_executed"] = calls
    summary["ranges"] = per_range
    summary["outside"] = compact(_summarize(outside_kernels, outside_copies, 1))
    summary["whole_process"] = compact(whole)
    return summary


def _context_span(driver: dict) -> dict | None:
    """The KV length and mRoPE position each decode step ran against.

    A quoted "106-token context" is the first step of a 200-step run, not a
    fixed length, and the mRoPE position is a different number again: image
    spans advance the (t, h, w) axes over the merged grid, so the last position
    is below the token count. Reporting both, with the range rather than one
    endpoint, is what keeps a per-token cost from being read as measured at a
    context it never saw. Returns ``None`` for phases that have no KV state.
    """

    if driver.get("context_start") is None:
        return None
    return {
        "context_start": driver.get("context_start"),
        "context_end": driver.get("context_end"),
        "pos_start": driver.get("pos_start"),
        "pos_end": driver.get("pos_end"),
        "decode_steps": driver.get("decode_steps"),
        "mean_context": (
            (driver["context_start"] + driver["context_end"]) / 2
            if driver.get("context_start") is not None
            and driver.get("context_end") is not None else None
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", action="append", default=[],
                        help="PHASE=DIRECTORY, repeatable")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--note", default=None)
    args = parser.parse_args()

    phases: dict[str, dict] = {}
    for spec in args.trace:
        if "=" not in spec:
            raise SystemExit(f"--trace expects PHASE=DIR, got {spec!r}")
        phase, _, directory = spec.partition("=")
        path = Path(directory)
        if not path.is_dir():
            raise SystemExit(f"trace directory not found: {path}")
        driver = _load_driver_output(path)
        calls = int(driver.get("calls_executed") or 0)
        # The range is named after the driver's --phase, which is not the
        # --trace label: a suite can trace the same phase twice (small and full
        # pages) under two labels.
        phase_range_name = f"surya-{driver.get('phase') or phase}"
        summary = _attribute_phases(
            _read_kernels(path), _read_copies(path), calls,
            list(driver.get("ranges") or []), phase_range_name,
        )
        summary["driver"] = driver
        summary["phase_range"] = phase_range_name
        summary["context"] = _context_span(driver)
        phases[phase] = summary

    artifact = {
        "date": time.strftime("%Y-%m-%d"),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": "datalab-to/surya-ocr-2",
        "protocol": {
            "profiler": "rocprofv3 --kernel-trace --memory-copy-trace",
            "jit": "libraries prebuilt outside the profiler; profiled process is cache-only",
            "attribution": (
                "phase cost = kernel total inside the driver's CLOCK_MONOTONIC "
                "range / calls the driver made inside that range; kernels outside "
                "every range are reported as 'outside'; whole_process is the "
                "unattributed trace for reference"
            ),
            "durations": "End_Timestamp - Start_Timestamp",
        },
        "phases": phases,
    }
    if args.note:
        artifact["note"] = args.note

    for phase, summary in phases.items():
        attribution = summary.get("attribution", "whole_process")
        per_call = summary["kernel_per_call_ms"]
        per_call_text = f", {per_call:.3f} ms/call [{attribution}]" if per_call is not None else ""
        print(f"\n== phase {phase}: {summary['calls_executed']} calls, "
              f"{summary['kernel_total_ms']:.2f} ms kernel total{per_call_text}")
        for entry in summary["kernel_families"][:8]:
            print(f"   {entry['name']:28s} {entry['total_ms']:8.3f} ms "
                  f"{entry['share_pct']:5.1f}%  {entry['calls']:6d} calls")
        transfers = summary["transfers"]
        mb = transfers["total_mb"]
        print(f"   transfers: {mb:.1f} MB, {transfers['total_ms']:.3f} ms"
              if mb is not None else
              f"   transfers: bytes not in trace, {transfers['total_ms']:.3f} ms")
        for entry in summary.get("ranges", []):
            mark = "<- phase" if entry["is_phase"] else ""
            print(f"   range {entry['name']:24s} {entry['calls_in_range']:5d} calls "
                  f"{entry['kernel_total_ms']:9.3f} ms kernel "
                  f"{entry['wall_ms']:9.3f} ms wall "
                  f"{entry['host_gap_ms']:7.3f} ms gap {mark}")
        outside = summary.get("outside")
        if outside and outside.get("kernel_calls"):
            print(f"   outside all ranges: {outside['kernel_calls']} kernels, "
                  f"{outside['kernel_total_ms']:.3f} ms")
        context = summary.get("context")
        if context:
            print(f"   context {context['context_start']}-{context['context_end']} "
                  f"over {context['decode_steps']} steps (mean "
                  f"{context['mean_context']:.0f}); mRoPE positions "
                  f"{context['pos_start']}-{context['pos_end']}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(artifact, indent=1))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
