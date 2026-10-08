#!/usr/bin/env python3
"""Same-residency engine A/B for the per-layer MoE decode graph.

One model residency, all 12 canonical cases, production compact-capture path.
The before-arm runs the stateless per-layer MoE section eagerly
(``cache.enabled = False``); the after-arm captures and replays it through the
existing ``MoeGraphCache`` (capture-on-first-use, bit-exact self-validating
parity, eager fallback). Balanced arm order per repetition; engagement is
verified from the cache's own capture/replay/eager stats and outputs must be
bit-identical across arms. The graph replays the identical kernel sequence, so
this is a scheduling-only candidate; no default changes in this A/B.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hipengine.core.memory import memory_stats
from hipengine.execution_profiles import ExecutionProfile, resolve_runtime_profile
from hipengine.generation.qwen4_exp_gguf import Qwen4ExpGGUFTextGenerator
from hipengine.generation.qwen4_exp_profiles import (
    QWEN4_EXP_BACKEND,
    QWEN4_EXP_MODEL,
    QWEN4_EXP_QUANTS,
    register_qwen4_exp_gfx1151_profiles,
)
from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
from hipengine.loading.gguf import discover_gguf_files, load_gguf_index
from hipengine.models import resolve_model
from scripts.qwen4exp_canonical_ar_bench import (
    DEFAULT_FIXTURE,
    _capture_kwargs,
    _git_metadata,
    _hipengine_case_sample,
    _host_metadata,
    load_fixture,
)

CHUNK_SIZE = 4096
TRANSITIONS = 128


def _arm_summary(rows):
    if not rows:
        raise AssertionError("no measured samples for arm")
    prefill_ms = sum(float(row["prefill_ms"]) for row in rows)
    decode_ms = sum(float(row["decode_ms"]) for row in rows)
    prompt_tokens = sum(int(row["prompt_tokens"]) for row in rows)
    decode_tokens = sum(int(row["decode_transitions"]) for row in rows)
    return {
        "samples": len(rows),
        "prefill_tok_s": 1000.0 * prompt_tokens / prefill_ms,
        "decode_tok_s": 1000.0 * decode_tokens / decode_ms,
        "client_wall_s": (prefill_ms + decode_ms) / 1000.0,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--compiler-version-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--case-id", action="append")
    parser.add_argument("--warmups-per-mode", type=int, default=1)
    args = parser.parse_args(argv)
    if args.warmups_per_mode < 1:
        parser.error("--warmups-per-mode must be at least 1")

    os.environ["HIPENGINE_HIP_ARCH"] = "gfx1151".replace("hip_", "")
    os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(args.compiler_version_file.resolve())
    os.environ["HIPENGINE_QWEN4_EXP_MOE_GRAPH"] = "1"

    fixture, fixture_sha = load_fixture(args.fixture)
    cases = list(fixture["cases"])
    if args.case_id:
        wanted = set(args.case_id)
        cases = [case for case in cases if str(case["id"]) in wanted]
        if {str(case["id"]) for case in cases} != wanted:
            parser.error("unknown --case-id")

    register_gfx1151_kernels(replace=True)
    register_qwen4_exp_gfx1151_profiles()
    index = load_gguf_index(discover_gguf_files(args.model_root.resolve())[0])
    profile = resolve_runtime_profile(
        model=QWEN4_EXP_MODEL,
        backend=QWEN4_EXP_BACKEND,
        quant=QWEN4_EXP_QUANTS[1],
        profile=ExecutionProfile.PRODUCTION,
    )
    max_len = max(int(case["prompt_tokens"]) for case in cases) + TRANSITIONS + 8
    generator = profile.construct_generator(lambda: Qwen4ExpGGUFTextGenerator(
        model_path=args.model_root.resolve(),
        weight_index=index,
        model_plugin=resolve_model(index.architecture or ""),
        backend=QWEN4_EXP_BACKEND,
        max_sequence_length=max_len,
        prefill_chunk_size=CHUNK_SIZE,
    ))
    runner = generator.runner
    moe_cache = runner.moe_graph_cache
    if moe_cache is None or not moe_cache.enabled:
        raise AssertionError("MoE graph cache is not enabled at construction")

    def set_arm(arm: str) -> None:
        # The cache is constructed enabled; the eager arm forces the fallback
        # through the cache's own enabled flag (its run() checks it first).
        moe_cache._enabled = arm == "after"  # noqa: SLF001 - diagnostic A/B toggle

    prefill_kwargs, step_kwargs = _capture_kwargs("compact")
    report: dict = {
        "schema": 1,
        "kind": "qwen4exp_moe_graph_engine_ab",
        "status": "running",
        "performance_claim": False,
        "source": _git_metadata(ROOT),
        "host": _host_metadata(),
        "command": sys.argv,
        "model_root": str(args.model_root),
        "fixture_sha256": fixture_sha,
        "manifest_sha256": profile.manifest_sha256,
        "arms": {"before": "MoE eager", "after": "MoE graph replay"},
        "chunk_size": CHUNK_SIZE,
        "transitions": TRANSITIONS,
        "warmups_per_mode": args.warmups_per_mode,
        "cases": {},
    }
    try:
        samples = {"before": [], "after": []}
        for case_index, case in enumerate(cases):
            case_id = str(case["id"])
            case_samples: dict = {"before": [], "after": []}
            # Warm both arms; the after-arm warmup pays capture-on-first-use.
            for arm in ("after", "before"):
                set_arm(arm)
                for _ in range(args.warmups_per_mode):
                    _hipengine_case_sample(
                        runner, case=case, repetition=-1, transitions=TRANSITIONS,
                        prefill_kwargs=prefill_kwargs, step_kwargs=step_kwargs,
                    )
            for repetition in range(3):
                order = ("before", "after") if (repetition + case_index) % 2 == 0 else ("after", "before")
                for arm in order:
                    set_arm(arm)
                    stats_before = dict(moe_cache.stats)
                    row = _hipengine_case_sample(
                        runner, case=case, repetition=repetition, transitions=TRANSITIONS,
                        prefill_kwargs=prefill_kwargs, step_kwargs=step_kwargs,
                    )
                    stats_after = dict(moe_cache.stats)
                    row["replay_delta"] = stats_after.get("replay", 0) - stats_before.get("replay", 0)
                    row["eager_delta"] = stats_after.get("eager", 0) - stats_before.get("eager", 0)
                    row["capture_delta"] = stats_after.get("capture", 0) - stats_before.get("capture", 0)
                    if arm == "after" and row["replay_delta"] <= 0:
                        raise AssertionError((case_id, "MoE graph did not replay in the after arm"))
                    if arm == "before" and row["replay_delta"] != 0:
                        raise AssertionError((case_id, "MoE graph replayed in the before arm"))
                    case_samples[arm].append(row)
                    samples[arm].append(row)
            before_digests = {row["output_token_ids_sha256"] for row in case_samples["before"]}
            after_digests = {row["output_token_ids_sha256"] for row in case_samples["after"]}
            entry = {}
            for arm in ("before", "after"):
                entry[arm] = _arm_summary(case_samples[arm])
                entry[arm]["within_mode_output_deterministic"] = (
                    len({row["output_token_ids_sha256"] for row in case_samples[arm]}) == 1
                )
                entry[arm]["replay_delta"] = sum(row["replay_delta"] for row in case_samples[arm])
            entry["cross_mode_output_exact"] = before_digests == after_digests
            report["cases"][case_id] = entry
            print(json.dumps({
                "case": case_id,
                "category": case["category"],
                "prompt_tokens": case["prompt_tokens"],
                "before_prefill_tok_s": round(entry["before"]["prefill_tok_s"], 3),
                "after_prefill_tok_s": round(entry["after"]["prefill_tok_s"], 3),
                "before_decode_tok_s": round(entry["before"]["decode_tok_s"], 3),
                "after_decode_tok_s": round(entry["after"]["decode_tok_s"], 3),
                "replay_delta": entry["after"]["replay_delta"],
                "cross_mode_output_exact": entry["cross_mode_output_exact"],
            }), flush=True)
        report["summary"] = {
            "correctness": {
                "all_cases_within_mode_deterministic": all(
                    arm["within_mode_output_deterministic"]
                    for entry in report["cases"].values()
                    for arm in (entry["before"], entry["after"])
                ),
                "cross_mode_output_exact": all(
                    entry["cross_mode_output_exact"] for entry in report["cases"].values()
                ),
                "mismatched_case_ids": sorted(
                    case_id for case_id, entry in report["cases"].items()
                    if not entry["cross_mode_output_exact"]
                ),
            },
            "all": {arm: _arm_summary(samples[arm]) for arm in ("before", "after")},
            "moe_graph_stats": moe_cache.stats,
        }
        report["summary"]["all"]["after_over_before_prefill"] = (
            report["summary"]["all"]["after"]["prefill_tok_s"]
            / report["summary"]["all"]["before"]["prefill_tok_s"]
        )
        report["summary"]["all"]["after_over_before_decode"] = (
            report["summary"]["all"]["after"]["decode_tok_s"]
            / report["summary"]["all"]["before"]["decode_tok_s"]
        )
        report["summary"]["all"]["after_over_before_client_wall"] = (
            report["summary"]["all"]["after"]["client_wall_s"]
            / report["summary"]["all"]["before"]["client_wall_s"]
        )
        report["status"] = "completed"
    finally:
        generator.close()
        report["memory_after_close"] = memory_stats()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
