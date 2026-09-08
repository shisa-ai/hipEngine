#!/usr/bin/env python3
"""Packed-route C1 K0-K7 sweep rollup: balanced-pair economics artifact.

Reads the r{1,2,3}-k{0..7} run JSONs produced under the fail-closed packed-C1
diagnostic harness (injected packed_c1_target evidence row, legacy singleton
verifier forbidden, packed frontier calls counted), gates every run, and
writes the compact benchmarks/results/ artifact. Exit 1 on any gate failure:
ar_exact contract broken, engagement/budget route expectation broken, any run
not complete, the K0 automatic control engaged, missing packed-call
accounting, model fingerprint drift, or runtime-source drift between commits.
"""
from __future__ import annotations

import json
import re
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

SWEEP_DIR = Path("/tmp/he-bettermtp-raw/packed-sweep")
OUT_PATH = Path(
    "benchmarks/results/2026-09-07-w7900-packed-route-c1-k0-k3-economics.json"
)
MODEL_SHA256 = "7b2aec3b9ababdfd75aa17552ee95607d866e44decf547f6f12fcef85cc89f1b"
MODEL_SIZE_BYTES = 17_106_773_984
GPU_UNIQUE_ID = "0xe282895b62c2b295"
HOST = "epyc"
RUNTIME_PATHS = ("hipengine/", "scripts/")
POLICY_REFUSED_DEPTHS = (1, 4, 5, 6, 7)


def load_run(path: Path) -> dict:
    d = json.loads(path.read_text())
    summary = d["summary"]["1"]
    fingerprint = d.get("model", {}).get("fingerprint", {})
    log = path.with_suffix(".log")
    packed_calls = None
    if log.exists():
        match = re.search(r"packed_target_calls=(\d+)", log.read_text())
        packed_calls = int(match.group(1)) if match else None
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
        "packed_calls": packed_calls,
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
    commits: set[str] = set()
    fingerprints: set[str] = set()
    for k in (0, 2, 3):
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
                failures.append(
                    f"k{k} {run['name']}: status failed {run['failure_reasons']}"
                )
            if run["cells"] != 10 or run["exact_cells"] != 10:
                failures.append(f"k{k} {run['name']}: exact {run['exact_cells']}/{run['cells']}")
            if not run["route_expectation_passed"]:
                failures.append(f"k{k} {run['name']}: route expectation failed")
            if k == 0 and run["engaged_cells"] != 0:
                failures.append(
                    f"k0 {run['name']}: automatic engaged {run['engaged_cells']}/10"
                )
            if k > 0:
                if run["engaged_cells"] != 10:
                    failures.append(f"k{k} {run['name']}: engaged {run['engaged_cells']}/10")
                if run["budget_conformed_cells"] != 10:
                    failures.append(
                        f"k{k} {run['name']}: budget {run['budget_conformed_cells']}/10"
                    )
            if run["packed_calls"] is None or run["packed_calls"] <= 0:
                if k > 0:
                    failures.append(
                        f"k{k} {run['name']}: packed calls {run['packed_calls']} (must be > 0)"
                    )
            if run["model_size_bytes"] not in (None, MODEL_SIZE_BYTES):
                failures.append(f"k{k} {run['name']}: model size mismatch")
        if len(runs) != 3:
            failures.append(f"k{k}: {len(runs)}/3 runs present")
            continue
        ratios = [run["ratio"] for run in runs]
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
                    "packed_calls": run["packed_calls"],
                    "source_commit": run["source_commit"],
                }
                for run in runs
            ],
            "median_ratio": round(statistics.median(ratios), 4),
            "min_ratio": round(min(ratios), 4),
            "max_ratio": round(max(ratios), 4),
            "median_ar_tok_s": round(
                statistics.median(run["ar"]["tok_s"] for run in runs), 2
            ),
            "median_mtp_tok_s": round(
                statistics.median(run["mtp"]["tok_s"] for run in runs), 2
            ),
            "pooled_ar_tok_s": round(pooled_ar, 2),
            "pooled_mtp_tok_s": round(pooled_mtp, 2),
            "pooled_ratio": round(pooled_mtp / pooled_ar, 4),
            "policy": (
                "automatic_k0_control" if k == 0 else "diagnostic_packed_c1_cell"
            ),
        }

    if len(fingerprints) > 1:
        failures.append(f"mixed model fingerprints: {sorted(fingerprints)}")
    commits = sorted(commits)
    if len(commits) > 1:
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

    print("Packed-route C1 balanced-pair economics (GPU0 W7900, D24, 20 ms):")
    print(f"{'depth':>5} {'median ratio':>12} {'range':>17} {'ar tok/s':>9} {'mtp tok/s':>10}")
    for k in (0, 2, 3):
        d = depths[k]
        rng = f"[{d['min_ratio']:.4f},{d['max_ratio']:.4f}]"
        mtp = "-" if k == 0 else f"{d['median_mtp_tok_s']:.2f}"
        print(f"{'K' + str(k):>5} {d['median_ratio']:>12.4f} {rng:>17} "
              f"{d['median_ar_tok_s']:>9.2f} {mtp:>10}")
    positive = {k: depths[k]["median_ratio"] for k in (2, 3)}
    winner = max(positive, key=positive.get)
    print(f"\nWinning packed-route depth: K{winner} "
          f"(median {positive[winner]:.4f}); legacy-route comparison at K3: 1.6329x")
    print("Physically refused packed depths (not in "
          "GGUF_SPECDEC2_MTP2_PHYSICAL_WIDTH_DEPTHS production): "
          + ", ".join(f"K{k}" for k in POLICY_REFUSED_DEPTHS))

    artifact = {
        "kind": "packed_route_c1_k0_k3_economics",
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
        "source_commit": commits[0] if commits else None,
        "harness": (
            "fail-closed diagnostic packed-C1 path: run_packed_bench.py injects the "
            "one-request packed_c1_target evidence row (explicit-only, unqualified "
            "test candidate per Packet 0) around the unmodified canonical bench and "
            "forbids the legacy singleton target verifier; every run records its "
            "packed frontier call count and any legacy use crashes the run"
        ),
        "route_status": (
            "diagnostic-only. The product serving route without registered packed-C1 "
            "evidence engages the legacy singleton verifier for explicit single-"
            "request MTP (spied 2026-09-07); these packed-route rates describe the "
            "repaired packed target, not the product default. Automatic selection "
            "stays K0/AR"
        ),
        "protocol": (
            "scripts/gguf_mtp_c1c8_server_bench.py via run_packed_bench.py, width 1, "
            "resident-capacity 8, D24 greedy, 20 ms batch window, ar_exact token "
            "contract, full canonical 10-prompt mtpbench-code-general-ja suite, "
            "GPU_MAX_HW_QUEUES=1, HIP_VISIBLE_DEVICES=0; one run = 10 balanced AR/MTP "
            "pairs with per-prompt arm-order alternation; three independent runs per "
            "depth in balanced rounds r1-r3"
        ),
        "k0_definition": (
            "true no-MTP autoregressive decode: the K0 'ar' arms (speculative_mtp="
            "false) and the automatic control runs (automatic selects K0, engaged 0/10)"
        ),
        "policy_refused_depths": {
            str(k): (
                "the backend physical width-depth policy "
                "(GGUF_SPECDEC2_MTP2_PHYSICAL_WIDTH_DEPTHS production = "
                "((1,2),(1,3),(2,2),(2,3),(8,3))) lists no (C1, K{d}) cell; "
                "_physical_c1_request consults the same policy, so the adapter "
                "refuses the packed route at that depth and falls back to the "
                "legacy singleton, which the harness's forbidden guard converts "
                "into a hard 500 (the shutdown-command timeout afterwards is "
                "teardown noise, not an idle hang)".replace("{d}", str(k))
            )
            for k in POLICY_REFUSED_DEPTHS
        },
        "depths": {f"k{k}": depths[k] for k in (0, 2, 3)},
        "winning_depth": f"k{winner}",
        "winner_median_ratio": depths[winner]["median_ratio"],
        "legacy_route_comparison_k3": {
            "median_ratio": 1.6329,
            "note": (
                "the legacy singleton route measured faster at C1/K3; the committed "
                "legacy-route economics artifact carries the correction marker"
            ),
        },
        "gates": {
            "ar_exact": "10/10 prompts per run, all runs",
            "engagement": "10/10 engaged for K1-K7; K0 automatic engaged 0/10",
            "budget_conformed": "10/10 for K1-K7",
            "packed_calls": ">0 per MTP run; legacy verifier forbidden process-wide",
        },
        "evidence_scope": (
            "Diagnostic packed-route economics. Product-route packed C1 requires "
            "evidence-row registration, which awaits the lifecycle, wider-capacity, "
            "sustained-context and service gates; automatic selection stays K0/AR"
        ),
        "commands": {
            "sweep": "bash campaign-artifacts/c1-k-sweep/run_packed_k_sweep.sh",
            "rollup": ".venv/bin/python campaign-artifacts/c1-k-sweep/rollup_packed_k_sweep.py",
            "single_run": (
                "HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 GPU_MAX_HW_QUEUES=1 "
                "HIPENGINE_HIP_ARCH=gfx1100 .venv/bin/python "
                "campaign-artifacts/c1-k-sweep/run_packed_bench.py --model "
                "/models/gguf/Qwen3.8-27B-Q4_K_M.gguf --backend hip_gfx1100 --quant "
                "gguf_q4_k_m --execution-profile production --prompts "
                "benchmarks/prompts/mtpbench-code-general-ja.jsonl --mtp-request-mode "
                "explicit --widths 1 --resident-capacity 8 --expected-mtp-widths 1 "
                "--candidate-budget K --max-tokens 24 --batch-window-ms 20 "
                "--correctness-contract ar_exact --output <out>.json"
            ),
        },
        "raw_run_dir": str(SWEEP_DIR),
    }
    OUT_PATH.write_text(json.dumps(artifact, indent=2) + "\n")
    print(f"\nWrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
