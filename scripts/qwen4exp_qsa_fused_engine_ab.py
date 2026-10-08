#!/usr/bin/env python3
"""Same-residency engine A/B for the fused QSA coefficients+values (T0-exact).

One model residency, compact-capture path, balanced arm order. The
before-arm runs the production ordered-v2 three-pass QSA route; the
after-arm sets HIPENGINE_QWEN4_EXP_QSA_ORDERED_DECODE_V2_FUSED=1 to launch
the fused coefficients+values sibling, which is bit-identical to the pair
(tests/test_gpu_qwen4exp_qsa_fused_v2.py), so cross-arm output digests must
match exactly.
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

FUSION_ENV = "HIPENGINE_QWEN4_EXP_QSA_ORDERED_DECODE_V2_FUSED"
BACKEND = "hip_gfx1151"
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
    parser.add_argument("--case-id", action="append")
    parser.add_argument("--warmups-per-mode", type=int, default=1)
    args = parser.parse_args(argv)

    os.environ["HIPENGINE_HIP_ARCH"] = "gfx1151"
    os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(args.compiler_version_file.resolve())
    os.environ[FUSION_ENV] = "0"

    fixture, fixture_sha = load_fixture(DEFAULT_FIXTURE)
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
    generator = profile.construct_generator(lambda: Qwen4ExpGGUFTextGenerator(
        model_path=args.model_root.resolve(),
        weight_index=index,
        model_plugin=resolve_model(index.architecture or ""),
        backend=QWEN4_EXP_BACKEND,
        max_sequence_length=max(int(c["prompt_tokens"]) for c in cases) + TRANSITIONS + 8,
        prefill_chunk_size=CHUNK_SIZE,
    ))

    def set_arm(arm: str) -> None:
        os.environ[FUSION_ENV] = "1" if arm == "after" else "0"

    prefill_kwargs, step_kwargs = _capture_kwargs("compact")
    report: dict = {
        "schema": 1,
        "kind": "qwen4exp_qsa_fused_engine_ab",
        "status": "running",
        "performance_claim": False,
        "source": _git_metadata(ROOT),
        "host": _host_metadata(),
        "command": sys.argv,
        "model_root": str(args.model_root),
        "fixture_sha256": fixture_sha,
        "manifest_sha256": profile.manifest_sha256,
        "arms": {
            "before": "ordered-v2 three-pass QSA route",
            "after": "fused coefficients+values sibling (bit-identical)",
        },
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
            for arm in ("after", "before"):
                set_arm(arm)
                for _ in range(args.warmups_per_mode):
                    _hipengine_case_sample(
                        runner=generator.runner, case=case, repetition=-1,
                        transitions=TRANSITIONS,
                        prefill_kwargs=prefill_kwargs, step_kwargs=step_kwargs,
                    )
            for repetition in range(3):
                order = ("before", "after") if (repetition + case_index) % 2 == 0 else ("after", "before")
                for arm in order:
                    set_arm(arm)
                    row = _hipengine_case_sample(
                        runner=generator.runner, case=case, repetition=repetition,
                        transitions=TRANSITIONS,
                        prefill_kwargs=prefill_kwargs, step_kwargs=step_kwargs,
                    )
                    case_samples[arm].append(row)
                    samples[arm].append(row)
            entry = {}
            for arm in ("before", "after"):
                entry[arm] = _arm_summary(case_samples[arm])
                entry[arm]["within_mode_output_deterministic"] = (
                    len({r["output_token_ids_sha256"] for r in case_samples[arm]}) == 1
                )
            entry["cross_mode_output_exact"] = (
                {r["output_token_ids_sha256"] for r in case_samples["before"]}
                == {r["output_token_ids_sha256"] for r in case_samples["after"]}
            )
            if not entry["cross_mode_output_exact"]:
                raise AssertionError((case_id, "fused select+cast diverged from the pair"))
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
                    entry[arm]["within_mode_output_deterministic"]
                    for entry in report["cases"].values()
                    for arm in ("before", "after")
                ),
                "cross_mode_output_exact": all(
                    entry["cross_mode_output_exact"] for entry in report["cases"].values()
                ),
            },
            "all": {arm: _arm_summary(samples[arm]) for arm in ("before", "after")},
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
        set_arm("before")
        generator.close()
        report["memory_after_close"] = memory_stats()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
