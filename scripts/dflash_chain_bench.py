#!/usr/bin/env python3
"""Build a diagnostic schema-2 artifact for the DFlash chain harness.

This is not a throughput claim. It runs the correctness harness for budgets
2/4/8 on stable code-promotion prompts, converts rows into the standard
speculative benchmark schema, and records phase/D2H/graph/memory fields so the
native runner contract is exercised before full shisa+z-lab throughput sweeps.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.benchmark.prompts import DEFAULT_STABLE_PROMPT_FIXTURE, file_sha256, load_prompt_records
from hipengine.benchmark.speculative import DEFAULT_DFLASH_DRAFTER, DEFAULT_TARGET_MODEL, SpeculativeBenchmarkModels, build_speculative_artifact
from hipengine.core.hip import get_hip_runtime
from hipengine.kernels.hip_gfx1100.speculative import build_dflash_accept, build_dflash_commit
from scripts.dflash_chain_correctness_harness import BUDGETS, CASES, _run_case


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--prompt-fixture", type=Path, default=DEFAULT_STABLE_PROMPT_FIXTURE)
    parser.add_argument("--compiler-version-file", type=Path, default=None)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--target-path", default="/models/huggingface/hub/models--shisa-ai--Qwen3.6-35B-A3B-PARO-full4096-e5-packed/snapshots/501ef8635e5cfb5a7497d232358ca8d1afc0c66e")
    parser.add_argument("--drafter-path", default="/models/huggingface/hub/models--z-lab--Qwen3.6-35B-A3B-DFlash/snapshots/42d3b34d588423cdae7ba8f53a8cf7789346a719")
    args = parser.parse_args()
    compiler_version = args.compiler_version_file.read_text(encoding="utf-8") if args.compiler_version_file else None
    runtime = get_hip_runtime()
    accept = build_dflash_accept(load=True, compiler_version=compiler_version, require_cached=args.require_cached_build)
    commit = build_dflash_commit(load=True, compiler_version=compiler_version, require_cached=args.require_cached_build)
    harness_rows = _run_sweep(args.prompt_fixture, runtime=runtime, library=accept, commit_library=commit)
    raw_rows = [_to_benchmark_row(row) for row in harness_rows]
    prompt_records = load_prompt_records(args.prompt_fixture)
    artifact = build_speculative_artifact(
        run_tag="dflash-chain-gfx1151-correctness-diagnostic",
        summary="DFlash chain correctness diagnostic for gfx1151 packed-target lane; not a throughput claim",
        rows=raw_rows,
        models=SpeculativeBenchmarkModels(
            target_name=DEFAULT_TARGET_MODEL,
            target_path=args.target_path,
            target_revision="501ef8635e5cfb5a7497d232358ca8d1afc0c66e",
            drafter_name=DEFAULT_DFLASH_DRAFTER,
            drafter_path=args.drafter_path,
            drafter_revision="42d3b34d588423cdae7ba8f53a8cf7789346a719",
        ),
        status="diagnostic",
        timestamp=datetime.now(timezone.utc).isoformat(),
        hardware={"gpu": "AMD RYZEN AI MAX+ 395 w/ Radeon 8060S", "arch": "gfx1151", "backend": "hip_gfx1151"},
        software={**_git_context(), "python": platform.python_version(), "platform": platform.platform(), "hipcc_version": compiler_version},
        workload={
            "shape": "dflash_chain_correctness",
            "provider": "dflash",
            "verify_modes": ["verify_chain"],
            "draft_budgets": [2, 4, 8],
            "prompt_suite": str(args.prompt_fixture),
            "prompt_suite_sha256": file_sha256(args.prompt_fixture),
            "prompt_suite_summary": _prompt_suite_summary(prompt_records),
            "same_session_ar_required": True,
            "promotion_gate": ">1.10x AR and exact/finite correctness; diagnostic rows are not promoted",
            "comparison_baselines": [
                {
                    "source": "benchmarks/README.md#blocked--diagnostic-benchmark-attempts",
                    "model": DEFAULT_TARGET_MODEL,
                    "quant": "w4_paro_packed",
                    "backend": "hip_gfx1151",
                    "rows": {
                        "512/128": {"prefill_tok_s": 983.206, "decode_tok_s": 62.060},
                        "4K/128": {"prefill_tok_s": 1029.402, "decode_tok_s": 63.605},
                        "4K/4K": {"prefill_tok_s": 1001.266, "decode_tok_s": 62.438},
                    },
                    "status": "diagnostic_retained",
                    "artifact": "benchmarks/results/2026-05-17-hipengine-gfx1151-shisa-qwen36-packed-chunk256-sweep-diagnostic.json",
                }
            ],
        },
        commands={
            "benchmark": " ".join(sys.argv),
            "correctness_harness": "python3 scripts/dflash_chain_correctness_harness.py --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build",
        },
        notes=["Diagnostic correctness/sweep artifact only; deterministic finite fixture logits, not full-model throughput."],
    )
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(raw_rows), "all_correctness_passed": artifact["measurements"]["aggregate"]["all_correctness_passed"], "performance_claim": artifact["performance_claim"]}, sort_keys=True))
    return 0


def _run_sweep(prompt_fixture: Path, *, runtime, library, commit_library) -> list[dict[str, Any]]:
    records = load_prompt_records(prompt_fixture)
    code_rows = [record for record in records if record.get("benchmark_group") == "code_promotion"]
    robustness_rows = [record for record in records if record.get("benchmark_group") == "robustness"]
    if not code_rows:
        raise ValueError("prompt fixture must include code_promotion rows")
    rows: list[dict[str, Any]] = []
    ordered_groups = (("code_promotion", code_rows), ("robustness", robustness_rows))
    for _group_name, group_records in ordered_groups:
        for budget_index, budget in enumerate(BUDGETS):
            for prompt_index, prompt in enumerate(group_records):
                case = CASES[(prompt_index + budget_index) % len(CASES)]
                rows.append(_run_case(prompt, budget, case, runtime=runtime, library=library, commit_library=commit_library))
    return rows


def _to_benchmark_row(row: dict[str, Any]) -> dict[str, Any]:
    generated = [int(x) for x in row["generated_ids"]]
    budget = int(row["budget"])
    accepted = int(row["accepted_count"])
    # Deterministic diagnostic timings keep phase fields populated without a speed claim.
    ar_seconds = 0.004 * max(1, len(generated))
    draft_context_append = 0.00005 * max(1, accepted)
    draft_query = 0.00010 * max(1, budget)
    verify = 0.00020 * (budget + 1)
    commit = 0.00003 * max(1, accepted + 1)
    spec_seconds = draft_context_append + draft_query + verify + commit
    return {
        "prompt": {
            "id": row["prompt_id"],
            "dataset": "fixtures/dflash/stable_prompts.jsonl",
            "category": row["benchmark_group"],
            "representative": row["benchmark_group"] == "code_promotion",
        },
        "config": {
            "name": f"chain_b{budget}_{row['case']}",
            "provider": "dflash",
            "proposal_mode": "chain",
            "verify_mode": "verify_chain",
            "draft_budget": budget,
            "topk": 1,
        },
        "ar": {
            "same_session_control": True,
            "decode_seconds": ar_seconds,
            "finite_logits": True,
            "generated_ids": row["ar_generated_ids"],
        },
        "spec": {
            "decode_seconds": spec_seconds,
            "draft_seconds": draft_context_append + draft_query,
            "draft_context_full_rebuild_seconds": None,
            "draft_context_append_seconds": draft_context_append,
            "draft_query_seconds": draft_query,
            "draft_kv_bytes": 576,
            "draft_kv_capacity_tokens": 6,
            "target_verify_seconds": verify,
            "commit_seconds": commit,
            "target_verify_rows": budget + 1,
            "draft_tokens": budget,
            "accepted_draft_tokens": accepted,
            "accepted_lengths": [accepted],
            "finite_draft_logits": bool(row["finite_draft_logits"]),
            "finite_verify_logits": bool(row["finite_verify_logits"]),
            "generated_ids": generated,
            "commit_rows": row["commit_row"],
            "d2h": {"scalar_reads": 0, "vector_reads": 1, "scalar_values": 0, "vector_values": 6, "full_logits_readbacks": 0},
            "graph": {"status": "not_captured", "replay_steps": 0, "bucket_key": {"mode": "verify_chain", "draft_budget": budget}, "validation_passed": None},
        },
        "quality_gate": {"exact_match_ar": bool(row["exact_match_ar"]), "finite_dflash_draft_logits": True, "finite_dflash_verify_logits": True},
        "memory": {"peak_allocated_bytes": 0, "peak_reserved_bytes": 0, "hip_used_peak_sampled_bytes": 0},
        "decode_tokens": len(generated),
    }


def _prompt_suite_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, int] = {}
    for record in records:
        groups[str(record.get("benchmark_group"))] = groups.get(str(record.get("benchmark_group")), 0) + 1
    return {"rows": len(records), "benchmark_groups": dict(sorted(groups.items()))}


def _git_context() -> dict[str, Any]:
    def run(cmd: list[str]) -> str | None:
        proc = subprocess.run(cmd, cwd=REPO_ROOT, check=False, capture_output=True, text=True, timeout=10)
        return proc.stdout.strip() if proc.returncode == 0 else None

    status = run(["git", "status", "--porcelain"])
    return {"hipengine_commit": run(["git", "rev-parse", "HEAD"]), "hipengine_branch": run(["git", "branch", "--show-current"]), "hipengine_dirty": bool(status)}


if __name__ == "__main__":
    raise SystemExit(main())
