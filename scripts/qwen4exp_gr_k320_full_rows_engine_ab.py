#!/usr/bin/env python3
"""Complete-engine A/B for the K320/full-row GR-up DPP sibling.

One model residency, all 12 canonical cases, production compact-capture path.
Arms toggle ``HIPENGINE_QWEN4_EXP_GR_K320_FULL_ROWS`` through the runner's own
default-off selector; a passive registry counter on the candidate variant
records engagement. Balanced before/after order per repetition, full prefill
plus 128 decode transitions, exact trajectory checks, engagement counts, and
teardown accounting. The candidate is never made a default by this A/B.
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
from hipengine.kernels.registry import KernelKey, register, resolve
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

PARENT = "coltile2_branch4_rowbatch4_wave_scale_dpp_f32_exact"
CANDIDATE = "coltile2_branch4_rowbatch4_wave_scale_dpp_k320_full_rows_f32_exact"
BACKEND = "hip_gfx1151"
LAYER = "linear+gr_gated_mean"
QUANT = "gguf_q8_0"
CHUNK_SIZE = 4096
TRANSITIONS = 128
K320_ENV = "HIPENGINE_QWEN4_EXP_GR_K320_FULL_ROWS"


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
        "prefill_ms_total": prefill_ms,
        "decode_ms_total": decode_ms,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--compiler-version-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--case-id", action="append")
    parser.add_argument("--warmups-per-mode", type=int, default=1)
    parser.add_argument("--require-cached-build", action="store_true")
    args = parser.parse_args(argv)
    if args.warmups_per_mode < 1:
        parser.error("--warmups-per-mode must be at least 1")

    os.environ["HIPENGINE_HIP_ARCH"] = BACKEND.replace("hip_", "")
    os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(args.compiler_version_file.resolve())
    if args.require_cached_build:
        os.environ["HIPENGINE_REQUIRE_CACHED_BUILD"] = "1"
    os.environ["HIPENGINE_QWEN4_EXP_GR_WAVE_SCALE"] = "1"
    os.environ["HIPENGINE_QWEN4_EXP_GR_DPP_REDUCTION"] = "1"
    os.environ[K320_ENV] = "0"

    fixture, fixture_sha = load_fixture(args.fixture)
    cases = list(fixture["cases"])
    if args.case_id:
        wanted = set(args.case_id)
        cases = [case for case in cases if str(case["id"]) in wanted]
        if {str(case["id"]) for case in cases} != wanted:
            parser.error("unknown --case-id")
    if len(cases) != 12 and not args.case_id:
        raise AssertionError("publication A/B requires all 12 canonical cases")

    register_gfx1151_kernels(replace=True)
    register_qwen4_exp_gfx1151_profiles()
    index = load_gguf_index(discover_gguf_files(args.model_root.resolve())[0])
    profile = resolve_runtime_profile(
        model=QWEN4_EXP_MODEL,
        backend=QWEN4_EXP_BACKEND,
        quant=QWEN4_EXP_QUANTS[1],
        profile=ExecutionProfile.PRODUCTION,
    )
    plugin = resolve_model(index.architecture or "")
    max_sequence_length = max(int(case["prompt_tokens"]) for case in cases) + TRANSITIONS + 8
    generator = profile.construct_generator(lambda: Qwen4ExpGGUFTextGenerator(
        model_path=args.model_root.resolve(),
        weight_index=index,
        model_plugin=plugin,
        backend=BACKEND,
        max_sequence_length=max_sequence_length,
        prefill_chunk_size=CHUNK_SIZE,
    ))
    runner = generator.runner

    candidate_key = KernelKey(BACKEND, LAYER, QUANT, CANDIDATE)
    candidate = resolve(backend=BACKEND, layer=LAYER, quant=QUANT, variant=CANDIDATE)
    state = {"candidate_calls": 0, "parent_calls": 0}

    def counting_candidate(*call_args, **kwargs):
        rows, in_features, branches, hidden = map(int, call_args[5:9])
        if not (rows > 256 and rows % 4 == 0 and branches == 4
                and in_features == 320 and hidden == 2560):
            raise AssertionError(
                f"K320 selector engaged outside its window: {rows=} {in_features=} "
                f"{branches=} {hidden=}")
        state["candidate_calls"] += 1
        return candidate(*call_args, **kwargs)

    register(candidate_key, counting_candidate, replace=True)

    prefill_kwargs, step_kwargs = _capture_kwargs("compact")
    report: dict = {
        "schema": 1,
        "kind": "qwen4exp_gr_k320_full_rows_engine_ab",
        "status": "running",
        "performance_claim": False,
        "source": _git_metadata(ROOT),
        "host": _host_metadata(),
        "command": sys.argv,
        "model_root": str(args.model_root),
        "fixture_sha256": fixture_sha,
        "manifest_sha256": profile.manifest_sha256,
        "arms": {"before": {"gr_k320": "0"}, "after": {"gr_k320": "1"}},
        "arm_toggle": K320_ENV,
        "parent_variant": PARENT,
        "candidate_variant": CANDIDATE,
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
            for mode in ("before", "after"):
                for _ in range(args.warmups_per_mode):
                    os.environ[K320_ENV] = "1" if mode == "after" else "0"
                    before_calls = state["candidate_calls"]
                    _hipengine_case_sample(
                        runner, case=case, repetition=-1, transitions=TRANSITIONS,
                        prefill_kwargs=prefill_kwargs, step_kwargs=step_kwargs,
                    )
                    if mode == "after":
                        if state["candidate_calls"] - before_calls <= 0:
                            raise AssertionError((case_id, "candidate did not engage in warmup"))
            for repetition in range(3):
                order = ("before", "after") if (repetition + case_index) % 2 == 0 else ("after", "before")
                for mode in order:
                    os.environ[K320_ENV] = "1" if mode == "after" else "0"
                    before_calls = state["candidate_calls"]
                    row = _hipengine_case_sample(
                        runner, case=case, repetition=repetition, transitions=TRANSITIONS,
                        prefill_kwargs=prefill_kwargs, step_kwargs=step_kwargs,
                    )
                    row["candidate_calls"] = state["candidate_calls"] - before_calls
                    if mode == "after" and row["candidate_calls"] <= 0:
                        raise AssertionError((case_id, "candidate did not engage"))
                    if mode == "before" and row["candidate_calls"] != 0:
                        raise AssertionError((case_id, "candidate engaged in the before arm"))
                    case_samples[mode].append(row)
                    samples[mode].append(row)
            before_digests = {row["output_token_ids_sha256"] for row in case_samples["before"]}
            after_digests = {row["output_token_ids_sha256"] for row in case_samples["after"]}
            entry = {}
            for mode in ("before", "after"):
                entry[mode] = _arm_summary(case_samples[mode])
                entry[mode]["within_mode_output_deterministic"] = (
                    len({row["output_token_ids_sha256"] for row in case_samples[mode]}) == 1
                )
                entry[mode]["output_token_ids_sha256"] = sorted(
                    {row["output_token_ids_sha256"] for row in case_samples[mode]})
                entry[mode]["candidate_calls"] = [
                    row["candidate_calls"] for row in case_samples[mode]]
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
                    row["cross_mode_output_exact"] for row in report["cases"].values()
                ),
                "mismatched_case_ids": sorted(
                    case_id for case_id, row in report["cases"].items()
                    if not row["cross_mode_output_exact"]
                ),
            },
            "all": {
                mode: _arm_summary(samples[mode]) for mode in ("before", "after")
            },
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
        report["candidate_calls_total"] = state["candidate_calls"]
        report["sample_count"] = len(samples["before"]) + len(samples["after"])
        report["status"] = "completed"
        report["memory_after_close"] = None
    finally:
        os.environ[K320_ENV] = "0"
        generator.close()
        report["memory_after_close"] = memory_stats()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
