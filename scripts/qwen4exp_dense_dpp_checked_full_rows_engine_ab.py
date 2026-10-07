#!/usr/bin/env python3
"""Complete-engine A/B for the checked/full-row dense Q8 DPP sibling.

The candidate is installed only as a diagnostic registry substitution for the
same six Q8_0 dense geometries and complete four-row shapes qualified by the
natural-owner and state gates.  The production profile and selector are not
changed.  Each canonical case is measured in one model residency with balanced
parent/candidate order, full prefill plus 128 decode transitions, exact output
trajectory checks, candidate engagement counts, and teardown accounting.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

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
    _git_metadata,
    _hipengine_case_sample,
    _host_metadata,
    _write_json,
    load_fixture,
)
from scripts.qwen4exp_dense_dpp_checked_full_rows_natural_owner import (
    DECLARED_SHAPES,
    admitted_shape as natural_admitted_shape,
)

PARENT = "coltile8_rowbatch4_wave_scale_pipelined_dpp_f32_f32_out"
CANDIDATE = "coltile8_rowbatch4_wave_scale_pipelined_dpp_checked_full_rows_f32_f32_out"
BACKEND = "hip_gfx1151"
QUANT = "gguf_q8_0"
CHUNK_SIZE = 4096
TRANSITIONS = 128
_BASE_SEQUENCE = ("before", "after", "after", "before", "before", "after")


def admitted_shape(rows: int, k: int, n: int) -> bool:
    return natural_admitted_shape(rows, k, n)


def arm_sequence(case_index: int) -> tuple[str, ...]:
    return _BASE_SEQUENCE if case_index % 2 == 0 else tuple(reversed(_BASE_SEQUENCE))


def _sample_runner(
    runner: Any,
    *,
    case: Mapping[str, Any],
    transitions: int,
    prefill_kwargs: Mapping[str, Any],
    step_kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    """Run one request while resetting the model-owned request state."""
    runner.reset()
    return _hipengine_case_sample(
        runner,
        case=case,
        repetition=0,
        transitions=transitions,
        prefill_kwargs=prefill_kwargs,
        step_kwargs=step_kwargs,
    )


def weighted_rates(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    tokens = sum(int(row["prompt_tokens"]) for row in rows)
    prefill_ms = sum(float(row["prefill_ms"]) for row in rows)
    transitions = sum(int(row["decode_transitions"]) for row in rows)
    decode_ms = sum(float(row["decode_ms"]) for row in rows)
    return {
        "prefill_tok_s": 1000.0 * tokens / prefill_ms,
        "decode_tok_s": 1000.0 * transitions / decode_ms,
    }


def compare(before: Sequence[Mapping[str, Any]], after: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    b = weighted_rates(before)
    a = weighted_rates(after)
    return {
        "before_prefill_tok_s": b["prefill_tok_s"],
        "after_prefill_tok_s": a["prefill_tok_s"],
        "after_over_before_prefill": a["prefill_tok_s"] / b["prefill_tok_s"],
        "before_decode_tok_s": b["decode_tok_s"],
        "after_decode_tok_s": a["decode_tok_s"],
        "after_over_before_decode": a["decode_tok_s"] / b["decode_tok_s"],
        "before_client_wall_s": sum(float(row["client_wall_s"]) for row in before),
        "after_client_wall_s": sum(float(row["client_wall_s"]) for row in after),
    }


def summarize(samples: Sequence[Mapping[str, Any]], cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_case: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in samples:
        by_case[str(row["case_id"])].append(row)
    case_summary: dict[str, Any] = {}
    mismatches: list[str] = []
    for case in cases:
        case_id = str(case["id"])
        rows = by_case[case_id]
        before = [row for row in rows if row["mode"] == "before"]
        after = [row for row in rows if row["mode"] == "after"]
        if len(before) != 3 or len(after) != 3:
            raise ValueError(f"{case_id}: expected 3 samples per arm")
        before_ids = {str(row["output_token_ids_sha256"]) for row in before}
        after_ids = {str(row["output_token_ids_sha256"]) for row in after}
        deterministic = len(before_ids) == 1 and len(after_ids) == 1
        exact = deterministic and before_ids == after_ids
        if not exact:
            mismatches.append(case_id)
        case_summary[case_id] = {
            "category": str(case["category"]),
            "prompt_tokens": int(case["prompt_tokens"]),
            "samples_per_mode": 3,
            **compare(before, after),
            "within_mode_output_deterministic": deterministic,
            "cross_mode_output_exact": exact,
            "candidate_calls": [int(row["candidate_calls"]) for row in after],
            "output_token_ids_sha256": {
                "before": sorted(before_ids),
                "after": sorted(after_ids),
            },
        }
    if mismatches:
        raise ValueError("cross-mode output mismatch: " + ", ".join(mismatches))

    def grouped(key: str) -> dict[str, Any]:
        result: dict[str, Any] = {}
        values = sorted({int(row[key]) for row in samples})
        for value in values:
            rows = [row for row in samples if int(row[key]) == value]
            result[str(value)] = compare(
                [row for row in rows if row["mode"] == "before"],
                [row for row in rows if row["mode"] == "after"],
            )
        return result

    before_all = [row for row in samples if row["mode"] == "before"]
    after_all = [row for row in samples if row["mode"] == "after"]
    return {
        "correctness": {
            "all_cases_within_mode_deterministic": all(
                value["within_mode_output_deterministic"] for value in case_summary.values()
            ),
            "cross_mode_output_exact": not mismatches,
            "mismatched_case_ids": mismatches,
        },
        "by_case": case_summary,
        "by_prompt_tokens": grouped("prompt_tokens"),
        "by_category": {
            category: compare(
                [row for row in samples if row["category"] == category and row["mode"] == "before"],
                [row for row in samples if row["category"] == category and row["mode"] == "after"],
            )
            for category in sorted({str(row["category"]) for row in samples})
        },
        "all": compare(before_all, after_all),
        "sample_count": len(samples),
        "candidate_calls_total": sum(int(row["candidate_calls"]) for row in after_all),
        "parent_calls_total": sum(int(row["parent_calls"]) for row in samples),
    }


def main(argv: Sequence[str] | None = None) -> int:
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
    os.environ["HIPENGINE_QWEN4_EXP_Q8_DPP_REDUCTION"] = "1"

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

    # The diagnostic hook is installed before generator construction so every
    # request path resolves through the same registry entry.  The profile's
    # constructor may re-register base families, so install it again after
    # construction and retain the original callable for restoration.
    key = KernelKey(BACKEND, "linear", QUANT, PARENT)
    parent = resolve(backend=BACKEND, layer=key.layer, quant=key.quant, variant=PARENT)
    candidate = resolve(backend=BACKEND, layer=key.layer, quant=key.quant, variant=CANDIDATE)
    state = {"enabled": False, "candidate_calls": 0, "parent_calls": 0}

    def hooked(*call_args: Any, **kwargs: Any) -> Any:
        if len(call_args) < 6:
            raise AssertionError("dense Q8 registry hook received too few arguments")
        rows, k, n = map(int, call_args[3:6])
        if state["enabled"] and admitted_shape(rows, k, n):
            state["candidate_calls"] += 1
            return candidate(*call_args, **kwargs)
        state["parent_calls"] += 1
        return parent(*call_args, **kwargs)

    register(key, hooked, replace=True)

    report: dict[str, Any] = {
        "schema": 1,
        "kind": "qwen4exp_dense_dpp_checked_full_rows_engine_ab",
        "status": "running",
        "performance_claim": False,
        "source": _git_metadata(ROOT),
        "host": _host_metadata(),
        "command": [sys.executable, str(Path(__file__).resolve()), *(argv or sys.argv[1:])],
        "model": "Qwen3.8-Flash-Next UD-Q4_K_XL",
        "model_root": str(args.model_root.resolve()),
        "fixture_sha256": fixture_sha,
        "manifest_sha256": profile.manifest_sha256,
        "strict_manifest_sha256": profile.strict_manifest_sha256,
        "parent_variant": PARENT,
        "candidate_variant": CANDIDATE,
        "backend": BACKEND,
        "quant": QUANT,
        "scope": {
            "rows": [512, 1024, 4096],
            "shapes": sorted([list(shape) for shape in DECLARED_SHAPES]),
            "prefill_chunk_size": CHUNK_SIZE,
            "decode_transitions": TRANSITIONS,
        },
        "protocol": {
            "one_model_residency": True,
            "one_python_process": True,
            "case_ids": [str(case["id"]) for case in cases],
            "warmups_per_mode_per_case": args.warmups_per_mode,
            "measured_repetitions_per_mode_per_case": 3,
            "arm_sequence_even_case": list(_BASE_SEQUENCE),
            "arm_sequence_odd_case": list(tuple(reversed(_BASE_SEQUENCE))),
            "timing_boundary": "synchronized prefill plus 128 runner.step transitions and client wall",
            "correctness_boundary": "exact output-token trajectories and repeat determinism",
        },
        "arms": {"before": {"dense_q8": PARENT}, "after": {"dense_q8": CANDIDATE}},
        "samples": [],
    }
    _write_json(args.output, report)

    try:
        register(key, hooked, replace=True)
        for case_index, case in enumerate(cases):
            case_id = str(case["id"])
            warm_order = ("before", "after") if case_index % 2 == 0 else ("after", "before")
            for _ in range(args.warmups_per_mode):
                for mode in warm_order:
                    state["enabled"] = mode == "after"
                    _sample_runner(
                        generator.runner,
                        case=case,
                        transitions=TRANSITIONS,
                        prefill_kwargs={},
                        step_kwargs={},
                    )
            for slot, mode in enumerate(arm_sequence(case_index)):
                state["enabled"] = mode == "after"
                before_candidate = int(state["candidate_calls"])
                before_parent = int(state["parent_calls"])
                row = _sample_runner(
                    generator.runner,
                    case=case,
                    transitions=TRANSITIONS,
                    prefill_kwargs={},
                    step_kwargs={},
                )
                row.update(
                    {
                        "mode": mode,
                        "sequence_slot": slot,
                        "candidate_calls": int(state["candidate_calls"]) - before_candidate,
                        "parent_calls": int(state["parent_calls"]) - before_parent,
                    }
                )
                report["samples"].append(row)
                _write_json(args.output, report)
                print(
                    f"{case_id} {mode} slot={slot} candidate_calls={row['candidate_calls']} "
                    f"pp={row['prefill_tok_s']:.3f} tg={row['decode_tok_s']:.3f}",
                    flush=True,
                )
        report["summary"] = summarize(report["samples"], cases)
        report["status"] = "completed"
        report["memory_before_close"] = memory_stats()
        return 0
    except Exception as error:
        report["status"] = "failed_correctness_or_protocol"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        register(key, parent, replace=True)
        generator.close()
        report["memory_after_close"] = memory_stats()
        if report.get("status") == "completed":
            report["lifecycle_passed"] = (
                report["memory_after_close"].get("active_allocations") == 0
                and report["memory_after_close"].get("current_allocated_bytes") == 0
            )
            if not report["lifecycle_passed"]:
                report["status"] = "failed_teardown"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        _write_json(args.output, report)


if __name__ == "__main__":
    raise SystemExit(main())
