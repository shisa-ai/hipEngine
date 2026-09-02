#!/usr/bin/env python3
"""B1 item-3 smoke analysis: verify-owner transfer under rocprof.

Consumes the two-arm smoke run in /tmp/q38-b1-run (env-off baseline, env-on
transfer) plus the rocprof'd env-on arm. Checks, in order:

1. Correctness: both arms report mtp_self_exact passing.
2. Owner routing: launch-attributed target-pass kernel templates in the
   env-on rocprof arm are the retained prefill band bodies (wmma prefill
   family / lowvgpr / shared3r1 / one-sweep), not the per-row direct GEMVs
   measured in the B1 owner map.
3. Cost: per-pass launch-attributed kernel medians by pass shape, compared
   with the B1 owner-map medians (R20 540.6 / R28 702.0 ms at commit e5ad9975).
4. IDs: env-on generated IDs versus env-off per cell (diagnostic equality;
   retention arithmetic is settled by the B1 item-4 gates, not here).

Host-only; prints a compact summary and writes the smoke artifact JSON.
"""

from __future__ import annotations

import bisect
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

RUN = Path("/tmp/q38-b1-run")
OFF_RAW = RUN / "off-raw.json"
ON_RAW = RUN / "on-raw.json"
ON_PROF_RAW = RUN / "on-prof-raw.json"
TRACE = RUN / "rocprof/b1-on_kernel_trace.csv"
API = RUN / "rocprof/b1-on_hip_api_trace.csv"
OUT = Path("benchmarks/results/2026-09-02-gfx1151-qwen38-b1-transfer-smoke.json")

EXPECTED_OWNER_MARKERS = (
    "wmma_prefill",
    "shared3r1",
    "lowvgpr",
)
FORBIDDEN_DOMINANT = (
    "q6_k_t16_qmicro_planar_gemv_bf16_kernel",
    "q6_k_t16_gemv_kernel",
)


def _canonical(name: str) -> str:
    name = name.split("(anonymous namespace)::")[-1]
    base = name.split("<", 1)[0].split("(", 1)[0]
    return base


def _target_windows(raw: dict) -> list[dict]:
    windows = []
    seen: set[tuple[int, int]] = set()
    for cell in raw["cells"]:
        ro = cell["mtp"].get("resident_observability") or {}
        for rec in ro.get("routes", {}).get("recent_completed", []):
            if "specdec2_mtp2_cycles" not in rec:
                continue
            for s, e, r in zip(
                rec.get("specdec2_mtp2_target_pass_start_ns", []),
                rec.get("specdec2_mtp2_target_pass_end_ns", []),
                rec.get("specdec2_mtp2_target_physical_rows", []),
            ):
                if (s, e) in seen:
                    continue
                seen.add((s, e))
                windows.append(
                    {"rows": int(r), "start_ns": int(s), "end_ns": int(e)}
                )
    return windows


def _launch_attributed(windows, trace, api):
    launches = []
    with api.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["Function"] != "hipLaunchKernel":
                continue
            launches.append(
                (int(row["Start_Timestamp"]), int(row["Correlation_Id"]))
            )
    launches.sort()
    launch_starts = [l[0] for l in launches]
    kernels = {}
    with trace.open(newline="") as handle:
        for row in csv.DictReader(handle):
            kernels[int(row["Correlation_Id"])] = {
                "name": _canonical(row["Kernel_Name"]),
                "start_ns": int(row["Start_Timestamp"]),
                "end_ns": int(row["End_Timestamp"]),
            }
    by_rows: dict[int, list[dict]] = defaultdict(list)
    for w in windows:
        i = bisect.bisect_left(launch_starts, w["start_ns"])
        total = 0
        tmpl: dict[str, dict] = defaultdict(lambda: {"n": 0, "ns": 0})
        while i < len(launches) and launches[i][0] < w["end_ns"]:
            ev = kernels.get(launches[i][1])
            i += 1
            if ev is None:
                continue
            d = ev["end_ns"] - ev["start_ns"]
            total += d
            tmpl[ev["name"]]["n"] += 1
            tmpl[ev["name"]]["ns"] += d
        by_rows[w["rows"]].append(
            {
                "kernel_ms": total / 1e6,
                "templates": {
                    k: {"n": v["n"], "ms": v["ns"] / 1e6} for k, v in tmpl.items()
                },
            }
        )
    return by_rows


def _ids_from(raw: dict) -> dict[str, list[int]]:
    out = {}
    for cell in raw["cells"]:
        arm = cell["mtp"]
        ids = []
        for row in arm.get("rows", []):
            gen = row.get("generated_token_ids") or row.get("generated_ids")
            if gen:
                ids.append(list(gen))
        if ids:
            out[str(arm.get("width"))] = ids
    return out


def main() -> None:
    missing = [str(p) for p in (OFF_RAW, ON_RAW, ON_PROF_RAW, TRACE, API) if not p.exists()]
    if missing:
        print(f"missing files: {missing}")
        sys.exit(2)

    off = json.loads(OFF_RAW.read_text())
    on = json.loads(ON_RAW.read_text())
    on_prof = json.loads(ON_PROF_RAW.read_text())

    def contracts(raw):
        per_cell = []
        for cell in raw["cells"]:
            c = cell["mtp"].get("correctness", {})
            per_cell.append(
                {
                    "width": cell["mtp"].get("width"),
                    "passed": bool(c.get("passed")),
                    "exact": c.get("prompts_exact"),
                    "engaged": c.get("prompts_engaged"),
                }
            )
        return per_cell

    off_correct = contracts(off)
    on_correct = contracts(on)
    on_prof_correct = contracts(on_prof)

    windows = _target_windows(on_prof)
    by_rows = _launch_attributed(windows, TRACE, API)

    passes = {}
    for rows, plist in sorted(by_rows.items()):
        n = len(plist)
        med = sorted(p["kernel_ms"] for p in plist)
        median = med[n // 2] if n % 2 else (med[n // 2 - 1] + med[n // 2]) / 2
        agg: dict[str, dict] = defaultdict(lambda: {"n": 0, "ns": 0})
        for p in plist:
            for k, v in p["templates"].items():
                agg[k]["n"] += v["n"]
                agg[k]["ns"] += int(v["ms"] * 1e6)
        top = sorted(agg, key=lambda k: -agg[k]["ns"])[:12]
        passes[str(rows)] = {
            "passes": n,
            "median_kernel_ms": round(median, 3),
            "template_ms_mean_per_pass": {
                k: {
                    "ms": round(agg[k]["ns"] / 1e6 / n, 3),
                    "launches_per_pass": round(agg[k]["n"] / n, 2),
                }
                for k in top
            },
        }

    # Owner-routing verdict from the widest pass shape.
    wide_rows = max(by_rows, default=0)
    wide_templates = (
        passes.get(str(wide_rows), {}).get("template_ms_mean_per_pass", {})
    )
    wmma_ms = sum(
        v["ms"]
        for k, v in wide_templates.items()
        if any(m in k for m in EXPECTED_OWNER_MARKERS)
    )
    gemv_ms = sum(
        v["ms"]
        for k, v in wide_templates.items()
        if k in FORBIDDEN_DOMINANT
    )
    wide_total = passes.get(str(wide_rows), {}).get("median_kernel_ms", 0.0)

    ids_off = _ids_from(off)
    ids_on = _ids_from(on)
    id_equal = {
        w: ids_on.get(w) == ids_off.get(w) for w in sorted(set(ids_off) | set(ids_on))
    }

    baseline = {"20": 540.6, "28": 702.0}
    artifact = {
        "schema": 1,
        "kind": "gfx1151_qwen38_b1_transfer_smoke",
        "date": "2026-09-02",
        "status": "smoke",
        "performance_claim": False,
        "physical_host": "gfx1151 / Framework Desktop / AMD Radeon 8060S / 1002:1586",
        "model": "/home/lhl/models/gguf/Qwen3.8-27B-Q4_K_M.gguf",
        "prompt": "code_merge_intervals (diagnostic single prompt)",
        "runtime_profile": "production",
        "env": {"HIPENGINE_GGUF_MTP_SERVING_TARGET_WMMA_PREFILL": "1 (on arm)"},
        "correctness": {
            "off_arm": off_correct,
            "on_arm": on_correct,
            "on_prof_arm": on_prof_correct,
        },
        "target_passes_env_on": passes,
        "owner_routing": {
            "wide_pass_rows": wide_rows,
            "wmma_band_owner_ms": round(wmma_ms, 3),
            "per_row_gemv_ms": round(gemv_ms, 3),
            "wide_pass_total_ms": wide_total,
            "expected": "wmma band owners dominate; per-row GEMVs shrink to minor tails",
        },
        "generated_ids_equal_off_vs_on": id_equal,
        "baseline_from_b1_owner_map": baseline,
        "findings": [],
    }
    ok_correct = all(c["passed"] for c in off_correct + on_correct + on_prof_correct)
    artifact["findings"].append(
        f"mtp_self_exact passed on all arms: {ok_correct}."
    )
    artifact["findings"].append(
        f"Wide pass R{wide_rows}: wmma-band owners {wmma_ms:.1f} ms vs per-row "
        f"GEMV {gemv_ms:.1f} ms of {wide_total:.1f} ms total "
        f"(B1 map baseline R20/R28: 540.6/702.0 ms)."
    )
    artifact["findings"].append(
        f"Generated IDs env-on vs env-off equal per width: {id_equal} "
        "(diagnostic; retention arithmetic is settled by the B1 item-4 gates)."
    )

    OUT.write_text(json.dumps(artifact, indent=2) + "\n")
    print(f"wrote {OUT}")
    print(json.dumps(artifact["owner_routing"], indent=2))
    print(json.dumps(artifact["generated_ids_equal_off_vs_on"], indent=2))
    for r, p in passes.items():
        print(f"R{r}: {p['passes']} passes, median {p['median_kernel_ms']:.1f} ms")
        for k, v in list(p["template_ms_mean_per_pass"].items())[:6]:
            print(f"    {v['ms']:>7.2f} ms  {k[:70]}")


if __name__ == "__main__":
    main()
