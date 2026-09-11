#!/usr/bin/env python3
"""Aggregate rocprofv3 traces of the Surya HIP pipeline into a profile.

Consumes the CSV traces produced by ``rocprofv3 --kernel-trace
--memory-copy-trace`` around ``scripts/surya_gpu_rocprof_driver.py`` and answers
the question the tuning decision needs: of the wall time per decode token, how
much is actually kernel execution, which kernel families own it, and how much is
host-side gap or transfer.

The driver reports how many calls it made, so per-call cost is the kernel total
divided by that count, with no assumption about warmup. A per-phase host gap is
``wall - sum(kernel durations)``, which is where launch overhead, synchronisation
and transfer stalls show up.

Usage:
    python3 scripts/surya_profile_report.py \\
        --trace decode=/tmp/surya-prof-decode \\
        --trace prefill=/tmp/surya-prof-prefill \\
        --out benchmarks/results/2026-09-11-gfx1151-surya-post-sgemv-profile.json
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


def _load_driver_output(directory: Path) -> dict:
    for path in sorted(directory.glob("*driver*.json")) + sorted(directory.glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        if isinstance(data, dict) and "phase" in data:
            return data
    return {}


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
        summary = _summarize(_read_kernels(path), _read_copies(path), calls)
        summary["driver"] = driver
        phases[phase] = summary

    artifact = {
        "date": time.strftime("%Y-%m-%d"),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": "datalab-to/surya-ocr-2",
        "protocol": {
            "profiler": "rocprofv3 --kernel-trace --memory-copy-trace",
            "jit": "libraries prebuilt outside the profiler; profiled process is cache-only",
            "attribution": "one process per phase; per-call cost = kernel total / calls the driver made",
            "durations": "End_Timestamp - Start_Timestamp",
        },
        "phases": phases,
    }
    if args.note:
        artifact["note"] = args.note

    for phase, summary in phases.items():
        print(f"\n== phase {phase}: {summary['calls_executed']} calls, "
              f"{summary['kernel_total_ms']:.2f} ms kernel total, "
              f"{summary['kernel_per_call_ms']:.3f} ms/call")
        for entry in summary["kernel_families"][:8]:
            print(f"   {entry['name']:28s} {entry['total_ms']:8.3f} ms "
                  f"{entry['share_pct']:5.1f}%  {entry['calls']:6d} calls")
        transfers = summary["transfers"]
        mb = transfers["total_mb"]
        print(f"   transfers: {mb:.1f} MB, {transfers['total_ms']:.3f} ms"
              if mb is not None else
              f"   transfers: bytes not in trace, {transfers['total_ms']:.3f} ms")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(artifact, indent=1))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
