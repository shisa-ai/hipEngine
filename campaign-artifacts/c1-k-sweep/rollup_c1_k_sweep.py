#!/usr/bin/env python3
"""C1 K0-K7 sweep rollup: balanced-pair economics table from the sweep JSONs.

Reads the r{1,2,3}-k{0..7} server-bench JSONs and prints the per-depth
MTP-vs-AR ratio (per-run plus median across the three independent runs),
the true-AR baseline, and the correctness/route gates. Exit code 1 if any
gate fails: ar_exact token contract broken, engagement or budget route
expectation broken, any run status != complete, or the K0 automatic control
showing engagement (automatic must select K0).

Writes the compact benchmarks/results/ artifact when all gates pass.
"""
from __future__ import annotations

import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

SWEEP_DIR = Path("/tmp/he-bettermtp-raw/c1-k-sweep")
OUT_PATH = Path("benchmarks/results/2026-09-07-w7900-packed-c1-k0-k7-economics.json")
MODEL_SHA256 = "7b2aec3b9ababdfd75aa17552ee95607d866e44decf547f6f12fcef85cc89f1b"
GPU_UNIQUE_ID = "0xe282895b62c2b295"
HOST = "epyc"
SCREENING_DEPTHS = (1, 4, 5, 6, 7)  # unlisted policy cells (explicit opt-in env)


def load_run(path: Path) -> dict:
    d = json.loads(path.read_text())
    summary = d["summary"]["1"]
    return {
        "name": path.stem,
        "passed": bool(d.get("passed")) and d.get("status") == "complete",
        "failure_reasons": d.get("failure_reasons", []),
        "source_commit": d.get("source", {}).get("commit"),
        "model_sha256": d.get("model", {}).get("sha256"),
        "ar": summary["ar"],
        "mtp": summary["mtp"],
        "ratio": summary["mtp_vs_ar_ratio"],
        "exact_cells": summary["exact_cells"],
        "engaged_cells": summary["engaged_cells"],
        "budget_conformed_cells": summary["budget_conformed_cells"],
        "cells": summary["cells"],
        "route_expectation_passed": summary["route_expectation_passed"],
    }


def main() -> int:
    failures: list[str] = []
    depths: dict[int, dict] = {}
    for k in range(8):
        runs = []
        for r in (1, 2, 3):
            path = SWEEP_DIR / f"r{r}-k{k}.json"
            if not path.exists():
                failures.append(f"k{k}: missing {path.name}")
                continue
            run = load_run(path)
            runs.append(run)
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
            # Model identity must match the registered lane artifact.
            if run["model_sha256"] != MODEL_SHA256:
                failures.append(f"k{k} {run['name']}: model sha mismatch")
        if len(runs) != 3:
            failures.append(f"k{k}: {len(runs)}/3 runs present")
            continue
        ratios = [run["ratio"] for run in runs]
        ar_rate = statistics.median(run["ar"]["tok_s"] for run in runs)
        mtp_rate = statistics.median(run["mtp"]["tok_s"] for run in runs)
        # Pooled same-suite arms across the three runs (aggregate comparison).
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
                "automatic_k0_control" if k == 0 else
                "listed_product_cell" if k in (2, 3) else
                "explicit_screening_cell_unqualified"
            ),
        }

    if failures:
        print("GATE FAILURES:")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("C1 K0-K7 balanced-pair economics (GPU0 W7900, D24, 20 ms, ar_exact):")
    print(f"{'depth':>5} {'policy':<36} {'median ratio':>12} {'range':>17} "
          f"{'ar tok/s':>9} {'mtp tok/s':>10}")
    for k in range(8):
        d = depths[k]
        rng = f"[{d['min_ratio']:.4f},{d['max_ratio']:.4f}]"
        ar = d["median_ar_tok_s"] if k == 0 else d["median_ar_tok_s"]
        mtp = "-" if k == 0 else f"{d['median_mtp_tok_s']:.2f}"
        print(f"{'K' + str(k):>5} {d['policy']:<36} {d['median_ratio']:>12.4f} {rng:>17} "
              f"{ar:>9.2f} {mtp:>10}")

    winner = max((k for k in range(1, 8)), key=lambda k: depths[k]["median_ratio"])
    print(f"\nWinning depth: K{winner} (median ratio {depths[winner]['median_ratio']:.4f}, "
          f"range [{depths[winner]['min_ratio']:.4f}, {depths[winner]['max_ratio']:.4f}])")

    artifact = {
        "kind": "native_packed_c1_k0_k7_economics",
        "date": datetime.now(timezone.utc).date().isoformat(),
        "host": HOST,
        "hardware": "GPU0 AMD Radeon Pro W7900 gfx1100",
        "gpu_unique_id": GPU_UNIQUE_ID,
        "model": "/models/gguf/Qwen3.8-27B-Q4_K_M.gguf",
        "model_sha256": MODEL_SHA256,
        "quant": "Q4_K_M",
        "execution_profile": "production",
        "kv": "BF16",
        "source_commit": depths[0]["runs"][0]["source_commit"],
        "source_clean_at_launch": True,
        "protocol": (
            "scripts/gguf_mtp_c1c8_server_bench.py, width 1, resident-capacity 8, "
            "D24 greedy, 20 ms batch window, ar_exact token contract, full canonical "
            "10-prompt mtpbench-code-general-ja suite, GPU_MAX_HW_QUEUES=1, "
            "HIP_VISIBLE_DEVICES=0; one run = 10 balanced AR/MTP pairs with per-prompt "
            "arm-order alternation; three independent runs per depth (balanced rounds "
            "r1-r3); per-run ratio is the same-run suite-aggregate MTP/AR rate"
        ),
        "k0_definition": (
            "true no-MTP autoregressive decode: the K0 'ar' arms (speculative_mtp=false) "
            "and the automatic control runs (automatic selects K0, engaged 0/10)"
        ),
        "depths": {f"k{k}": depths[k] for k in range(8)},
        "winning_depth": f"k{winner}",
        "winner_median_ratio": depths[winner]["median_ratio"],
        "gates": {
            "ar_exact": "10/10 prompts per run, all 24 runs",
            "engagement": "10/10 engaged for K1-K7; K0 automatic engaged 0/10",
            "budget_conformed": "10/10 for K1-K7",
            "route": "K2/K3 listed product policy; K1/K4-K7 explicit screening opt-in",
        },
        "evidence_scope": (
            "Economics evidence only. Lifecycle (K0<->MTP switch), wider-capacity "
            "isolation, sustained horizon/context, and dynamic service-owner gates "
            "remain open per the campaign doc; automatic selection stays K0."
        ),
        "commands": {
            "sweep": "bash campaign-artifacts/c1-k-sweep/run_c1_k_sweep.sh",
            "rollup": ".venv/bin/python campaign-artifacts/c1-k-sweep/rollup_c1_k_sweep.py",
            "explicit_run": (
                "HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 GPU_MAX_HW_QUEUES=1 "
                "HIPENGINE_HIP_ARCH=gfx1100 [HIPENGINE_MTP2_SCREEN_UNQUALIFIED_CELLS=1 "
                "for K1/K4-K7] .venv/bin/python scripts/gguf_mtp_c1c8_server_bench.py "
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
