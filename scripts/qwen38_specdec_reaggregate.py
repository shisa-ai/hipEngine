#!/usr/bin/env python3
"""Recompute the archived short speculative sweep without changing its sources.

The originals lack token counts, elapsed times and per-repetition AR rates.
Reconstruction assumes the configured fixed output count for every observation;
weighted prompt medians are not a substitute for pooled original timings.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path


SOURCE = Path("benchmarks/results/2026-11-12-qwen38-specdec-source")
OUTPUT = Path("benchmarks/results/2026-11-12-qwen38-specdec-reaggregation.json")


def reconstruct(path: Path) -> dict:
    artifact = json.loads(path.read_text())
    method = path.stem.split("_")[0]
    rows = artifact["rows"]
    rate = statistics.harmonic_mean(r[f"{method}_tok_s_median"] for r in rows)
    ar_rate = statistics.harmonic_mean(r["ar_tok_s_median"] for r in rows)
    return {
        "source": path.as_posix(), "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "method": method, "provenance": artifact["provenance"], "protocol": artifact["protocol"],
        "reported_prompt_median_tok_s": artifact["summary"][f"{method}_tok_s_median_of_prompts"],
        "weighted_prompt_medians_tok_s": rate, "ar_weighted_prompt_medians_tok_s": ar_rate,
        "ratio_of_weighted_prompt_medians": rate / ar_rate,
        "pooled_repetitions_tok_s_reconstructed": statistics.harmonic_mean(
            value for row in rows for value in row[f"{method}_tok_s_reps"]),
        "pooled_ar_repetitions_tok_s": None,
        "historical_exact_all": artifact["summary"]["exact_all"],
        "prompt_count": len(rows),
        "configured_token_positions_per_pass": len(rows) * artifact["protocol"]["max_new_tokens"],
        "mean_acceptance": artifact["summary"]["mean_acceptance"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    paths = sorted(args.source.glob("*.json"))
    if not paths:
        parser.error("source directory contains no JSON artifacts")
    rows = {path.stem: reconstruct(path) for path in paths}
    gains = {}
    for method in ("eagle4", "dspine", "dflash2"):
        before, after = rows[f"{method}_q8_off"], rows[f"{method}_q8_on"]
        gains[method] = {
            "prompt_median_gain_percent": (after["reported_prompt_median_tok_s"] / before["reported_prompt_median_tok_s"] - 1) * 100,
            "weighted_prompt_medians_gain_percent": (after["weighted_prompt_medians_tok_s"] / before["weighted_prompt_medians_tok_s"] - 1) * 100,
            "pooled_repetitions_gain_percent": (after["pooled_repetitions_tok_s_reconstructed"] / before["pooled_repetitions_tok_s_reconstructed"] - 1) * 100,
        }
    result = {
        "schema": 1, "kind": "historical_speculative_aggregation_correction",
        "status": "diagnostic_reconstruction", "performance_claim": False, "publishable": False,
        "command": f"python3 scripts/qwen38_specdec_reaggregate.py --source {args.source} --output {args.output}",
        "hardware_context": {
            "host": "zbook", "gpu": "Radeon 8060S", "backend": "hip_gfx1151",
            "source": "campaign worklogs; originals do not record host or clocks",
        },
        "limitations": [
            "No GPU rerun. Source JSON files are preserved byte-for-byte; stale note fields are historical, not current descriptions.",
            "Fixed output counts are inferred from the harness and configured budget, not saved counts. Historical exact_all checked only the common prefix, not lengths; token IDs were not saved.",
            "Weighted prompt medians use harmonic means of rates with equal assumed token counts. They differ from pooling every repetition's elapsed time.",
            "Per-repetition AR rates were discarded, so a fully paired pooled ratio cannot be recovered.",
            "Legacy rates count all outputs over post-prefill wall; the MTP suite counts output transitions. Timing and numerator differences prevent cross-method ranking even after reaggregation.",
            "Original artifacts record HEAD but not the working-tree diff or environment overrides. Q8 arm identity comes from file names and linked worklogs, not a complete variant manifest.",
            "Deterministic repetitions add timing observations, not independent prompt or correctness coverage. No confidence interval for depth selection is available.",
        ],
        "conclusions": {
            "depth": "The short suite does not establish that n3 is faster than n2. Keep the shipping depth pending a longer matched public-path comparison.",
            "cross_method_ranking": "Unresolved; neither EAGLE nor MTP is established as the winner by these artifacts.",
            "q8": "Within-method timing improvements survive both aggregations; public-path performance and full-length correctness need separate validation.",
        },
        "q8_gains": gains, "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(gains, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
