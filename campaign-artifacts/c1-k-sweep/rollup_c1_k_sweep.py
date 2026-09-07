#!/usr/bin/env python3
"""C1 K0-K3 sweep rollup: balanced-pair economics from the sweep JSONs.

Reads the r{1,2,3}-k{0..3} server-bench JSONs plus the k7-refusal exemplar
and prints the per-depth MTP-vs-AR ratio (per-run plus median across the
three independent runs), the true-AR baseline, and the correctness/route
gates. Exit code 1 if any gate fails: ar_exact token contract broken,
engagement or budget route expectation broken, any run status != complete,
the K0 automatic control showing engagement (automatic must select K0), or
the K7 refusal exemplar showing engagement (deeper-than-evidence requests
must refuse pre-mutation to K0).

Writes the compact benchmarks/results/ artifact when all gates pass.
"""
from __future__ import annotations

import json
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

SWEEP_DIR = Path("/tmp/he-bettermtp-raw/c1-k-sweep")
OUT_PATH = Path("benchmarks/results/2026-09-07-w7900-packed-c1-k0-k3-economics.json")
MODEL_SHA256 = "7b2aec3b9ababdfd75aa17552ee95607d866e44decf547f6f12fcef85cc89f1b"
MODEL_SIZE_BYTES = 17_106_773_984
GPU_UNIQUE_ID = "0xe282895b62c2b295"
HOST = "epyc"
SCREENING_DEPTHS = (1,)  # unlisted policy cell (explicit opt-in env)
RUNTIME_PATHS = ("hipengine/", "scripts/")


def load_run(path: Path) -> dict:
    d = json.loads(path.read_text())
    summary = d["summary"]["1"]
    fingerprint = d.get("model", {}).get("fingerprint", {})
    return {
        "name": path.stem,
        "passed": bool(d.get("passed")) and d.get("status") == "complete",
        "failure_reasons": d.get("failure_reasons", []),
        "source_commit": d.get("source", {}).get("commit"),
        "model_fingerprint": (
            f"{fingerprint.get('algorithm')}:{fingerprint.get('value')}"
            if fingerprint.get("value")
            else None
        ),
        "model_size_bytes": fingerprint.get("size_bytes"),
        "ar": summary["ar"],
        "mtp": summary["mtp"],
        "ratio": summary["mtp_vs_ar_ratio"],
        "exact_cells": summary["exact_cells"],
        "engaged_cells": summary["engaged_cells"],
        "budget_conformed_cells": summary["budget_conformed_cells"],
        "cells": summary["cells"],
        "route_expectation_passed": summary["route_expectation_passed"],
        "mtp_route_decision": (
            d["cells"][0]["mtp"]["rows"][0].get("mtp", {})
            if d.get("cells")
            else {}
        ),
    }


def main() -> int:
    failures: list[str] = []
    depths: dict[int, dict] = {}
    commits: set[str] = set()
    fingerprints: set[str] = set()
    for k in range(4):
        runs = []
        for r in (1, 2, 3):
            path = SWEEP_DIR / f"r{r}-k{k}.json"
            if not path.exists():
                failures.append(f"k{k}: missing {path.name}")
                continue
            run = load_run(path)
            runs.append(run)
            commits.add(str(run["source_commit"]))
            if run["model_fingerprint"]:
                fingerprints.add(run["model_fingerprint"])
            if not run["passed"]:
                failures.append(f"k{k} {run['name']}: status failed {run['failure_reasons']}")
            if run["cells"] != 10 or run["exact_cells"] != 10:
                failures.append(
                    f"k{k} {run['name']}: exact {run['exact_cells']}/{run['cells']}"
                )
            if not run["route_expectation_passed"]:
                failures.append(f"k{k} {run['name']}: route expectation failed")
            if k == 0 and run["engaged_cells"] != 0:
                failures.append(
                    f"k0 {run['name']}: automatic engaged {run['engaged_cells']}/10 (must be 0)"
                )
            if k > 0 and run["engaged_cells"] != 10:
                failures.append(f"k{k} {run['name']}: engaged {run['engaged_cells']}/10")
            if k > 0 and run["budget_conformed_cells"] != 10:
                failures.append(f"k{k} {run['name']}: budget {run['budget_conformed_cells']}/10")
            if run["model_fingerprint"] is None:
                failures.append(f"k{k} {run['name']}: missing model fingerprint")
            if run["model_size_bytes"] not in (None, MODEL_SIZE_BYTES):
                failures.append(f"k{k} {run['name']}: model size mismatch")
        if len(runs) != 3:
            failures.append(f"k{k}: {len(runs)}/3 runs present")
            continue
        ratios = [run["ratio"] for run in runs]
        ar_rate = statistics.median(run["ar"]["tok_s"] for run in runs)
        mtp_rate = statistics.median(run["mtp"]["tok_s"] for run in runs)
        pooled_ar = sum(run["ar"]["generated_tokens"] for run in runs) / sum(
            run["ar"]["wall_seconds"] for run in runs
        )
        pooled_mtp = sum(run["mtp"]["generated_tokens"] for run in runs) / sum(
            run["mtp"]["wall_seconds"] for run in runs
        )
        depths[k] = {
            "runs": [
                {
                    "name": run["name"],
                    "ratio": round(run["ratio"], 4),
                    "ar_tok_s": round(run["ar"]["tok_s"], 2),
                    "mtp_tok_s": round(run["mtp"]["tok_s"], 2),
                    "ar_tokens": run["ar"]["generated_tokens"],
                    "mtp_tokens": run["mtp"]["generated_tokens"],
                    "ar_wall_s": round(run["ar"]["wall_seconds"], 3),
                    "mtp_wall_s": round(run["mtp"]["wall_seconds"], 3),
                    "exact": f"{run['exact_cells']}/{run['cells']}",
                    "engaged": f"{run['engaged_cells']}/{run['cells']}",
                    "budget_conformed": f"{run['budget_conformed_cells']}/{run['cells']}",
                    "source_commit": run["source_commit"],
                }
                for run in runs
            ],
            "median_ratio": round(statistics.median(ratios), 4),
            "min_ratio": round(min(ratios), 4),
            "max_ratio": round(max(ratios), 4),
            "median_ar_tok_s": round(ar_rate, 2),
            "median_mtp_tok_s": round(mtp_rate, 2),
            "pooled_ar_tok_s": round(pooled_ar, 2),
            "pooled_mtp_tok_s": round(pooled_mtp, 2),
            "pooled_ratio": round(pooled_mtp / pooled_ar, 4),
            "policy": (
                "automatic_k0_control" if k == 0
                else "explicit_screening_cell_unqualified" if k in SCREENING_DEPTHS
                else "listed_product_cell"
            ),
        }

    # K7 refusal exemplar: deeper-than-evidence requests must refuse to K0.
    k7_refusal = None
    k7_path = SWEEP_DIR / "k7-refusal-exemplar.json"
    if k7_path.exists():
        run = load_run(k7_path)
        commits.add(str(run["source_commit"]))
        engaged = run["engaged_cells"]
        summary = run["mtp_route_decision"]
        k7_refusal = {
            "requested_candidate_budget": 7,
            "expected": "pre-mutation refusal to K0 (evidence-scope budget cap 3)",
            "engaged_cells": f"{engaged}/{run['cells']}",
            "decision_reason": summary.get("decision_reason"),
            "selection_reason": summary.get("selection_reason")
            or summary.get("decision_reason"),
            "run_passed": run["passed"],
        }
        if engaged != 0:
            failures.append(
                f"k7-refusal: engaged {engaged}/10 (deeper-than-evidence request must refuse)"
            )
    else:
        failures.append("k7-refusal: exemplar run missing")

    if len(fingerprints) > 1:
        failures.append(
            f"mixed model fingerprints across runs: {sorted(fingerprints)}"
        )

    commits = sorted(commits)
    if len(commits) > 1:
        # Docs/campaign-artifact commits between runs are acceptable when the
        # runtime surface is identical; anything else is a provenance failure.
        for before, after in zip(commits, commits[1:]):
            diff = subprocess.run(
                ["git", "diff", "--stat", before, after, "--", *RUNTIME_PATHS],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            if diff:
                failures.append(
                    f"runtime source differs between {before[:9]} and {after[:9]}: {diff}"
                )

    if failures:
        print("GATE FAILURES:")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("C1 K0-K3 balanced-pair economics (GPU0 W7900, D24, 20 ms, ar_exact):")
    print(f"{'depth':>5} {'policy':<38} {'median ratio':>12} {'range':>17} "
          f"{'ar tok/s':>9} {'mtp tok/s':>10}")
    for k in range(4):
        d = depths[k]
        rng = f"[{d['min_ratio']:.4f},{d['max_ratio']:.4f}]"
        mtp = "-" if k == 0 else f"{d['median_mtp_tok_s']:.2f}"
        print(f"{'K' + str(k):>5} {d['policy']:<38} {d['median_ratio']:>12.4f} {rng:>17} "
              f"{d['median_ar_tok_s']:>9.2f} {mtp:>10}")
    assert k7_refusal is not None
    print(f"\nK7 refusal exemplar: engaged {k7_refusal['engaged_cells']}, "
          f"decision_reason={k7_refusal['decision_reason']} (evidence cap 3 < requested 7)")
    winner = max((k for k in range(1, 4)), key=lambda k: depths[k]["median_ratio"])
    print(f"Winning qualified depth: K{winner} "
          f"(median ratio {depths[winner]['median_ratio']:.4f}, "
          f"range [{depths[winner]['min_ratio']:.4f}, {depths[winner]['max_ratio']:.4f}])")

    artifact = {
        "kind": "native_packed_c1_k0_k3_economics",
        "date": datetime.now(timezone.utc).date().isoformat(),
        "host": HOST,
        "hardware": "GPU0 AMD Radeon Pro W7900 gfx1100",
        "gpu_unique_id": GPU_UNIQUE_ID,
        "model": "/models/gguf/Qwen3.8-27B-Q4_K_M.gguf",
        "model_sha256_full_file": MODEL_SHA256,
        "model_fingerprint_sampled": sorted(fingerprints)[0] if fingerprints else None,
        "quant": "Q4_K_M",
        "execution_profile": "production",
        "kv": "BF16",
        "source_commit": sorted(commits)[0],
        "source_clean_at_launch": True,
        "protocol": (
            "scripts/gguf_mtp_c1c8_server_bench.py, width 1, resident-capacity 8, "
            "D24 greedy, 20 ms batch window, ar_exact token contract, full canonical "
            "10-prompt mtpbench-code-general-ja suite, GPU_MAX_HW_QUEUES=1, "
            "HIP_VISIBLE_DEVICES=0; one run = 10 balanced AR/MTP pairs with per-prompt "
            "arm-order alternation; three independent runs per depth in balanced "
            "rounds r1-r3; per-run ratio is the same-run suite-aggregate MTP/AR rate; "
            "aggregation reports per-run ratios plus their median and pooled arms"
        ),
        "k0_definition": (
            "true no-MTP autoregressive decode: the K0 'ar' arms "
            "(speculative_mtp=false) and the automatic control runs (automatic "
            "selects K0, engaged 0/10). No verifier-derived B0 substitute."
        ),
        "route_validity": (
            "Native packed C1 product route after the deferred static-intent repair "
            "(555fc7ef3): the realized width-miss deferral preserves the "
            "evidence-backed static eligibility, so the resident owner's fail-closed "
            "physical admission engages C1 requests (listed (1,2)/(1,3) cells; K1 via "
            "the explicit-only screening opt-in HIPENGINE_MTP2_SCREEN_UNQUALIFIED_CELLS=1)"
        ),
        "depths": {f"k{k}": depths[k] for k in range(4)},
        "k4_k7_status": {
            "measured": False,
            "reason": (
                "the serving key carries the requested candidate budget (4-7); every "
                "registered Qwen3.8 gfx1100 evidence row qualifies at most candidate "
                "budget 3, so the K4-K7 checks fail both physical_group_not_qualified "
                "and candidate_budget_not_qualified, no static eligibility override is "
                "granted, and the resident adapter then refuses the (1, K>3) cell "
                "(decline trace: cell (C1, K7) not in policy) pre-mutation to K0 by "
                "design. Measuring deeper depths on the product route requires the "
                "Packet-5/6 qualification chain (draft-chain product-route execution, "
                "service gates), not an evidence-scope bypass"
            ),
            "teacher_numerical_status": (
                "target-only calibrated fixed-teacher checks pass K1-K7 at N1 with "
                "three bit-identical repeats per depth (see "
                "benchmarks/results/2026-09-07-w7900-packed-c1-teacher-k5-repeats.json "
                "and k6-k7); they do not qualify product-route depth economics"
            ),
            "refusal_exemplar": k7_refusal,
        },
        "winning_depth": f"k{winner}",
        "winner_median_ratio": depths[winner]["median_ratio"],
        "gates": {
            "ar_exact": "10/10 prompts per run, all runs",
            "engagement": "10/10 engaged for K1-K3; K0 automatic and K7 refusal engaged 0/10",
            "budget_conformed": "10/10 for K1-K3",
            "route": "K2/K3 listed product policy; K1 explicit screening opt-in",
        },
        "evidence_scope": (
            "Economics evidence only, measured on the actual product server route. "
            "Lifecycle (K0<->MTP switch), wider-capacity isolation, sustained "
            "horizon/context, and dynamic service-owner gates remain open per the "
            "campaign doc before any C1 evidence row is re-registered or automatic "
            "promotion is considered; automatic selection stays K0."
        ),
        "commands": {
            "sweep": "bash campaign-artifacts/c1-k-sweep/run_c1_k_sweep.sh",
            "rollup": ".venv/bin/python campaign-artifacts/c1-k-sweep/rollup_c1_k_sweep.py",
            "explicit_run": (
                "HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 GPU_MAX_HW_QUEUES=1 "
                "HIPENGINE_HIP_ARCH=gfx1100 [HIPENGINE_MTP2_SCREEN_UNQUALIFIED_CELLS=1 "
                "for K1] .venv/bin/python scripts/gguf_mtp_c1c8_server_bench.py "
                "--model /models/gguf/Qwen3.8-27B-Q4_K_M.gguf --backend hip_gfx1100 "
                "--quant gguf_q4_k_m --execution-profile production "
                "--prompts benchmarks/prompts/mtpbench-code-general-ja.jsonl "
                "--mtp-request-mode explicit --widths 1 --resident-capacity 8 "
                "--expected-mtp-widths 1 --candidate-budget K --max-tokens 24 "
                "--batch-window-ms 20 --correctness-contract ar_exact --output <out>.json"
            ),
            "k0_run": (
                "same as explicit_run with --mtp-request-mode automatic "
                "--expected-mtp-widths none and no screening env"
            ),
        },
        "raw_run_dir": str(SWEEP_DIR),
    }
    OUT_PATH.write_text(json.dumps(artifact, indent=2) + "\n")
    print(f"\nWrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
