"""Compact PLE measurements, preserving scope and source-verification limits."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.qwen4exp_route_pair import _mean_ci95
from scripts.qwen4exp_halo_box_campaign_ab import summarize_campaign_ab
HOST = "55ea6c509d0b49eea8de7094a1023668"
FINGERPRINT = "fb1f2fbf73d588c9ac27f24bade5663bd3da8ac1862f62ee5bf457578a88ec53"


def source_matches(data, script):
    head = data["source"]["head"]
    if not re.fullmatch("[0-9a-f]{40}", head):
        return False
    result = subprocess.run(
        ["git", "show", f"{head}:{script}"], cwd=ROOT,
        capture_output=True, check=False,
    )
    return result.returncode == 0 and hashlib.sha256(result.stdout).hexdigest() == data["script_sha256"]


def compact(path):
    raw = path.read_bytes()
    data = json.loads(raw)
    if data["status"] != "completed" or data["host"]["machine_id"] != HOST:
        raise ValueError(f"incomplete/wrong-host measurement: {path}")
    if data["model_identity"]["fingerprint"]["value"] != FINGERPRINT:
        raise ValueError("wrong model")
    result = {
        "raw_path": str(path), "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "kind": data["kind"], "source": data["source"], "host": data["host"],
        "model_identity": data["model_identity"], "command": data["command"],
        "method": data.get("method", "sorted_unique"),
        "script_sha256": data["script_sha256"],
    }
    if data["kind"] == "qwen4exp_ple_sorted_gather_screen":
        result["script_matches_recorded_commit"] = source_matches(data, "scripts/qwen4exp_ple_gather_screen.py")
        result["scope"] = data["scope"]
        result["cases"] = []
        for case in data["cases"]:
            if case["bit_exact"] is not True:
                raise ValueError("nonexact gather")
            samples = case["samples"]
            if any(not math.isfinite(s["ns"]) or s["ns"] <= 0 for s in samples):
                raise ValueError("invalid timing")
            for rep in {s["repetition"] for s in samples}:
                arms = [s["arm"] for s in samples if s["repetition"] == rep]
                if sorted(arms) != ["candidate", "parent"]:
                    raise ValueError("incomplete paired timing")
            medians = {arm: statistics.median(s["ns"] for s in samples if s["arm"] == arm)
                       for arm in ("parent", "candidate")}
            io = {arm: [s.get("process_read_bytes") for s in samples if s["arm"] == arm]
                  for arm in ("parent", "candidate")}
            result["cases"].append({
                "id": case["id"], "category": case["category"],
                "rows": case["rows"], "unique_rows": case["unique_rows"],
                "cache": case["cache_mode"], "median_ns": medians,
                "parent_over_candidate": medians["parent"] / medians["candidate"],
                "process_read_bytes": io, "bit_exact": True,
                "samples": samples,
            })
    elif data["kind"] == "qwen4exp_ple_complete_model_ab":
        result["script_matches_recorded_commit"] = source_matches(data, "scripts/qwen4exp_ple_gather_ab.py")
        samples = data["samples"]
        cases = {s["case_id"] for s in samples}
        repetitions = data["protocol"]["repetitions_per_arm"]
        expected_cases = data["protocol"].get("case_ids")
        if expected_cases is None:
            expected_cases = sorted(cases) if len(cases) == 12 else []
        if set(expected_cases) != cases or len(samples) != 2 * len(expected_cases) * repetitions:
            raise ValueError("incomplete model matrix")
        for case in cases:
            rows = [s for s in samples if s["case_id"] == case]
            for mode in ("before", "after"):
                if sum(s["mode"] == mode for s in rows) != repetitions:
                    raise ValueError("incomplete model arms")
            for key in ("output_token_ids_sha256", "final_logits_sha256", "final_state_sha256"):
                if len({s[key] for s in rows}) != 1:
                    raise ValueError(f"model output/state mismatch: {case} {key}")
        if (data["after_close"]["active_allocations"] != 0
                or data["after_close"]["current_allocated_bytes"] != 0):
            raise ValueError("ownership did not close")
        result.update(
            protocol=data["protocol"],
            summary={key: value for key, value in summarize_campaign_ab(
                samples, repetitions_per_mode=repetitions,
            ).items() if key in ("correctness", "by_shape", "by_category")},
            exact_final_logits_and_state=True,
            manifest_sha256=data["manifest_sha256"], after_close=data["after_close"],
        )
        result["complete_canonical_suite"] = len(cases) == 12
        result["pairs"] = {}
        for case in sorted(cases):
            rows = [s for s in samples if s["case_id"] == case]
            ratios = []
            for slot in range(0, 2 * repetitions, 2):
                pair = {s["mode"]: s for s in rows if s["sequence_slot"] in (slot, slot + 1)}
                if set(pair) != {"before", "after"}:
                    raise ValueError("incomplete model pair")
                ratios.append(pair["before"]["prefill_ms"] / pair["after"]["prefill_ms"])
            result["pairs"][case] = {
                "prefill_ratios": ratios, "mean": statistics.mean(ratios),
                "mean_95ci": _mean_ci95(ratios),
            }
        result["samples"] = [
            {key: row[key] for key in (
                "case_id", "category", "mode", "repetition", "sequence_slot",
                "prefill_ms", "decode_ms", "output_token_ids_sha256",
                "final_logits_sha256", "final_state_sha256", "memory_delta",
            )} for row in samples
        ]
    else:
        raise ValueError("unknown measurement")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, default=Path(__file__).with_name("artifact.json"))
    parser.add_argument("--retained", action="store_true")
    args = parser.parse_args()
    result = {
        "schema": 1, "kind": "journey_ple_evidence",
        "status": "retained" if args.retained else "in_progress",
        "performance_claim": bool(args.retained), "runtime_default_changed": bool(args.retained),
        "observations": [compact(path) for path in args.raw],
        "limits": [
            "CPU gather/staging times are not request throughput.",
            "Cold means file-scoped eviction advice, not a global cache flush.",
            "Uncommitted exploratory script snapshots are not promotion evidence.",
            "The incumbent production numerical failure is a separate unresolved gate.",
        ],
    }
    if args.retained:
        full = [r for r in result["observations"] if r.get("complete_canonical_suite")
                and r["protocol"]["repetitions_per_arm"] >= 3
                and r["script_matches_recorded_commit"]]
        if {r["protocol"]["cache"] for r in full} != {"warm", "cold"}:
            raise ValueError("retention requires full source-bound warm and cold matrices")
        cold = next(r for r in full if r["protocol"]["cache"] == "cold")
        if not all(row["after_over_before_prefill"] > 1 and row["after_over_before_decode"] > 1
                   for row in cold["summary"]["by_shape"].values()):
            raise ValueError("cold matrix did not improve")
        result["decision"] = {
            "default": "gfx1151 Qwen4Exp production PLE mapping-only random advice",
            "strict_fallback": "normal mapping advice",
            "override": "HIPENGINE_QWEN4_EXP_PLE_MAPPING_ACCESS=normal",
            "arithmetic_changed": False,
            "numerical_baseline": "incumbent failure unchanged; not a new numerical certificate",
            "warm_note": "72-row paired matrix approximately neutral; apparent Japanese4K -0.22% did not reproduce in five fresh pairs",
            "cold_note": "all12 cases improve with exact generated IDs, final logits/state and zero teardown",
        }
    args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
