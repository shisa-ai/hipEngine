#!/usr/bin/env python3
"""Measure the retained halo-box campaign routes in one model residency.

The ``before`` arm restores the two exact owners selected at campaign commit
``0436e138b``. The ``after`` arm selects the current PF-3 Q5_1 M1 owner and
PF-1 grouped Q8_0 down owner. Every case receives three measurements per arm
in a balanced six-slot order. Adjacent cases reverse that order so each arm
occupies every timing slot equally across the four canonical categories.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.qwen4exp_canonical_ar_bench import (  # noqa: E402
    DEFAULT_FIXTURE,
    _git_metadata,
    _hipengine_case_sample,
    _host_metadata,
    _write_json,
    load_fixture,
    summarize_samples,
)


BEFORE_COMMIT = "0436e138b5fe6a43b1b1bae5df6c33fff2110148"
FORKB_ENV = "HIPENGINE_QWEN4_EXP_FORKB_GROUPED_DOWN"
Q5_M1_ENV = "HIPENGINE_QWEN4_EXP_PROFILE_Q5_1_DOWN_M1"
ROW4_ENV = "HIPENGINE_QWEN4_EXP_GROUPED_ROW4_PREFILL"
QSA_H256_ENV = "HIPENGINE_QWEN4_EXP_QSA_H256_WAVE_PREFILL"
_BASE_SEQUENCE = ("before", "after", "after", "before", "before", "after")


def arm_sequence(case_index: int) -> tuple[str, ...]:
    """Return a three-per-arm order, reversed for adjacent cases."""

    if int(case_index) % 2:
        return tuple(reversed(_BASE_SEQUENCE))
    return _BASE_SEQUENCE


def fixture_case_index(cases: Sequence[Mapping[str, Any]], case: Mapping[str, Any]) -> int:
    """Keep the full-fixture counterbalance when running a diagnostic subset."""
    return next(index for index, row in enumerate(cases) if row["id"] == case["id"])


def _weighted_rates(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    prompt_tokens = sum(int(row["prompt_tokens"]) for row in rows)
    prefill_ms = sum(float(row["prefill_ms"]) for row in rows)
    transitions = sum(int(row["decode_transitions"]) for row in rows)
    decode_ms = sum(float(row["decode_ms"]) for row in rows)
    if prefill_ms <= 0.0 or decode_ms <= 0.0:
        raise ValueError("sample timing must be positive")
    return {
        "prefill_tok_s_weighted": 1000.0 * prompt_tokens / prefill_ms,
        "decode_tok_s_weighted": 1000.0 * transitions / decode_ms,
    }


def _comparison(before: Sequence[Mapping[str, Any]], after: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    before_rates = _weighted_rates(before)
    after_rates = _weighted_rates(after)
    return {
        "before_prefill_tok_s_weighted": before_rates["prefill_tok_s_weighted"],
        "after_prefill_tok_s_weighted": after_rates["prefill_tok_s_weighted"],
        "after_over_before_prefill": (
            after_rates["prefill_tok_s_weighted"]
            / before_rates["prefill_tok_s_weighted"]
        ),
        "before_decode_tok_s_weighted": before_rates["decode_tok_s_weighted"],
        "after_decode_tok_s_weighted": after_rates["decode_tok_s_weighted"],
        "after_over_before_decode": (
            after_rates["decode_tok_s_weighted"]
            / before_rates["decode_tok_s_weighted"]
        ),
    }


def summarize_campaign_ab(
    samples: Sequence[Mapping[str, Any]], *, repetitions_per_mode: int
) -> dict[str, Any]:
    """Validate the balanced A/B packet and summarize complete timing walls."""

    expected = int(repetitions_per_mode)
    if expected <= 0:
        raise ValueError("repetitions_per_mode must be positive")
    by_case: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in samples:
        mode = row.get("mode")
        if mode not in {"before", "after"}:
            raise ValueError(f"invalid campaign A/B mode {mode!r}")
        by_case[str(row["case_id"])].append(row)
    if not by_case:
        raise ValueError("campaign A/B produced no samples")

    case_summary: dict[str, Any] = {}
    mismatches: list[str] = []
    within_mode_deterministic = True
    for case_id, rows in sorted(by_case.items()):
        modes = {
            mode: [row for row in rows if row["mode"] == mode]
            for mode in ("before", "after")
        }
        counts = {mode: len(mode_rows) for mode, mode_rows in modes.items()}
        if any(count != expected for count in counts.values()):
            raise ValueError(
                f"{case_id}: expected {expected} samples per mode, got {counts}"
            )
        digests = {
            mode: {str(row["output_token_ids_sha256"]) for row in mode_rows}
            for mode, mode_rows in modes.items()
        }
        deterministic = all(len(values) == 1 for values in digests.values())
        within_mode_deterministic = within_mode_deterministic and deterministic
        exact = deterministic and digests["before"] == digests["after"]
        if not exact:
            mismatches.append(case_id)
        case_summary[case_id] = {
            "category": str(rows[0]["category"]),
            "prompt_tokens": int(rows[0]["prompt_tokens"]),
            "samples_per_mode": expected,
            **_comparison(modes["before"], modes["after"]),
            "within_mode_deterministic": deterministic,
            "cross_mode_output_exact": exact,
            "output_token_ids_sha256": {
                mode: sorted(values) for mode, values in digests.items()
            },
        }
    if not within_mode_deterministic:
        raise ValueError("within-mode output mismatch in campaign A/B")
    if mismatches:
        raise ValueError(
            "cross-mode output mismatch in campaign A/B: " + ", ".join(mismatches)
        )

    by_shape: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    by_category: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_mode: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in samples:
        by_shape[int(row["prompt_tokens"])].append(row)
        by_category[str(row["category"])].append(row)
        by_mode[str(row["mode"])].append(row)

    def compare_group(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        before = [row for row in rows if row["mode"] == "before"]
        after = [row for row in rows if row["mode"] == "after"]
        return {
            "case_count": len({str(row["case_id"]) for row in rows}),
            "samples_per_mode": len(before),
            **_comparison(before, after),
        }

    return {
        "correctness": {
            "within_mode_deterministic": True,
            "cross_mode_output_exact": True,
            "mismatched_case_ids": [],
        },
        "by_case": case_summary,
        "by_shape": {
            str(shape): compare_group(rows)
            for shape, rows in sorted(by_shape.items())
        },
        "by_category": {
            category: compare_group(rows)
            for category, rows in sorted(by_category.items())
        },
        "before": summarize_samples(by_mode["before"]),
        "after": summarize_samples(by_mode["after"]),
    }


def _apply_mode(
    mode: str,
    *,
    environment: MutableMapping[str, str] = os.environ,
    route_package: str = "pf13",
) -> None:
    if route_package in {"q5k-row4", "qsa-h256-wave", "qsa-h256-page256"}:
        if mode not in {"before", "after"}:
            raise ValueError(f"invalid campaign A/B mode {mode!r}")
        flag = ROW4_ENV if route_package == "q5k-row4" else QSA_H256_ENV
        environment[flag] = "1" if mode == "after" else "0"
        if route_package == "qsa-h256-page256" and mode == "after":
            environment[flag] = "page256"
        return
    if mode == "before":
        environment[Q5_M1_ENV] = "0"
        environment[FORKB_ENV] = "0"
        return
    if mode == "after":
        environment[Q5_M1_ENV] = "1"
        environment[FORKB_ENV] = "1"
        return
    raise ValueError(f"invalid campaign A/B mode {mode!r}")


def validate_row4_engagement(mode: str, calls: int) -> None:
    if (mode == "before" and calls != 0) or (mode == "after" and calls <= 0):
        raise ValueError(f"invalid row4 engagement: {mode} calls={calls}")


def validate_qsa_h256_engagement(mode: str, calls: int, prompt_tokens: int) -> None:
    if prompt_tokens not in {512, 1024, 4096} or mode not in {"before", "after"}:
        raise ValueError("QSA engagement check requires a canonical shape and arm")
    expected = mode == "after" and prompt_tokens == 4096
    if (calls > 0) != expected:
        raise ValueError(f"invalid QSA engagement: {mode} p{prompt_tokens} calls={calls}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefill-chunk-size", type=int, default=512)
    parser.add_argument("--warmups-per-mode", type=int, default=1)
    parser.add_argument("--repetitions-per-mode", type=int, default=3)
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--route-package", choices=("pf13", "q5k-row4", "qsa-h256-wave", "qsa-h256-page256"), default="pf13")
    parser.add_argument("--case-id", action="append", help="Diagnostic subset; omitted for full gate")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.warmups_per_mode < 1:
        raise SystemExit("--warmups-per-mode must be at least 1")
    if args.repetitions_per_mode != 3:
        raise SystemExit("publication protocol requires --repetitions-per-mode 3")
    if args.compiler_version_file is not None:
        os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(
            args.compiler_version_file.resolve()
        )
    if args.require_cached_build:
        os.environ["HIPENGINE_REQUIRE_CACHED_BUILD"] = "1"
    os.environ.setdefault("HIPENGINE_HIP_ARCH", "gfx1151")

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

    fixture, fixture_sha256 = load_fixture(args.fixture)
    cases = fixture["cases"]
    if args.case_id:
        cases = [case for case in cases if case["id"] in args.case_id]
        if {case["id"] for case in cases} != set(args.case_id):
            raise SystemExit("unknown --case-id")
    transitions = int(fixture["decode_transitions"])
    model_root = args.model_root.resolve()
    max_sequence_length = max(int(row["prompt_tokens"]) for row in cases) + transitions + 8

    register_gfx1151_kernels(replace=True)
    register_qwen4_exp_gfx1151_profiles()
    index = load_gguf_index(discover_gguf_files(model_root)[0])
    plugin = resolve_model(index.architecture or "")
    resolved = resolve_runtime_profile(
        model=QWEN4_EXP_MODEL,
        backend=QWEN4_EXP_BACKEND,
        quant=QWEN4_EXP_QUANTS[1],
        profile=ExecutionProfile.PRODUCTION,
    )

    def factory() -> Qwen4ExpGGUFTextGenerator:
        return Qwen4ExpGGUFTextGenerator(
            model_path=model_root,
            weight_index=index,
            model_plugin=plugin,
            backend="hip_gfx1151",
            max_sequence_length=max_sequence_length,
            prefill_chunk_size=args.prefill_chunk_size,
        )

    source = _git_metadata(ROOT)
    if not source or not source.get("tracked_clean"):
        raise SystemExit("campaign publication A/B requires a tracked-clean worktree")
    command = [sys.executable, str(Path(__file__).resolve()), *(argv or sys.argv[1:])]
    artifact: dict[str, Any] = {
        "schema": 1,
        "kind": "qwen4exp_halo_box_campaign_same_residency_ab",
        "status": "running",
        "performance_claim": False,
        "host": _host_metadata(),
        "source": source,
        "commands": {"argv": command, "shell": shlex.join(command)},
        "model_root": str(model_root),
        "model_identity": fixture.get("model"),
        "fixture": str(args.fixture.resolve()),
        "fixture_sha256": fixture_sha256,
        "profile": {
            "requested": "production",
            "manifest_sha256": resolved.manifest_sha256,
            "strict_manifest_sha256": resolved.strict_manifest_sha256,
            "fell_back_to_strict": resolved.fell_back_to_strict,
        },
        "arms": {
            "before": {
                "reference_commit": BEFORE_COMMIT,
                "q5_1_down": "selected_grouped_prefill_compact_rowbatch8_out8_expertgrid64_bf16_bf16_out",
                "q8_0_down": "selected_gemv_bf16_bf16_out",
            },
            "after": {
                "q5_1_down": "selected_grouped_prefill_compact_rowbatch8_out8_expertgrid64_m1_bf16_bf16_out",
                "q8_0_down": "selected_grouped_gemv_bf16_bf16_out",
            },
        },
        "protocol": {
            "one_model_residency": True,
            "one_python_process": True,
            "case_ids": [str(row["id"]) for row in cases],
            "case_order": "fixture order",
            "case_counterbalance_indices": {
                str(row["id"]): fixture_case_index(fixture["cases"], row) for row in cases
            },
            "arm_order_even_case": list(arm_sequence(0)),
            "arm_order_odd_case": list(arm_sequence(1)),
            "warmups_per_mode_per_case": int(args.warmups_per_mode),
            "measured_repetitions_per_mode_per_case": int(args.repetitions_per_mode),
            "decode_transitions": transitions,
            "visible_output_tokens": transitions + 1,
            "prefill_chunk_size": int(args.prefill_chunk_size),
            "timing_boundary": (
                "synchronized runner.prefill including first greedy output, then "
                "exactly 128 runner.step transitions and a final device synchronize"
            ),
        },
        "warmups": [],
        "samples": [],
    }
    _write_json(args.output, artifact)

    generator = resolved.construct_generator(factory)
    row4_calls = [0]
    original_row4 = None
    if args.route_package in {"q5k-row4", "qsa-h256-wave", "qsa-h256-page256"}:
        from hipengine.kernels.registry import KernelKey, register, resolve
        row4_key = (KernelKey(
            "hip_gfx1151", "linear", "gguf_q5_k",
            "selected_grouped_row4_gemv_bf16_bf16_out")
            if args.route_package == "q5k-row4" else KernelKey(
                "hip_gfx1151", "qsa_sparse_attention", "bf16_kv",
                "strict_h256_page256_wave_rows_spans" if args.route_package == "qsa-h256-page256"
                else "strict_h256_wave_rows_spans"))
        original_row4 = resolve(
            backend=row4_key.backend, layer=row4_key.layer,
            quant=row4_key.quant, variant=row4_key.variant)

        def counted_row4(*call_args, **call_kwargs):
            row4_calls[0] += 1
            return original_row4(*call_args, **call_kwargs)

        register(row4_key, counted_row4, replace=True)
        artifact["arms"] = {
            "before": {"q5_k_gate_up": "selected_gemv_bf16_bf16_out"},
            "after": {"q5_k_gate_up": row4_key.variant},
        }
        if args.route_package.startswith("qsa-h256-"):
            artifact["arms"] = {
                "before": {"sparse_prefill": "strict_rows_spans"},
                "after": {"sparse_prefill": row4_key.variant},
            }
    artifact["route_package"] = args.route_package
    artifact["diagnostic_subset"] = bool(args.case_id)

    def sample(mode, case, repetition):
        _apply_mode(mode, route_package=args.route_package)
        start_calls = row4_calls[0]
        row = _hipengine_case_sample(
            generator.runner, case=case, repetition=repetition, transitions=transitions)
        if original_row4 is not None:
            calls = row4_calls[0] - start_calls
            if args.route_package.startswith("qsa-h256-"):
                validate_qsa_h256_engagement(mode, calls, int(case["prompt_tokens"]))
            else:
                validate_row4_engagement(mode, calls)
            row["candidate_calls"] = calls
        return row

    try:
        for case in cases:
            case_index = fixture_case_index(fixture["cases"], case)
            warmup_modes = ("before", "after") if case_index % 2 == 0 else ("after", "before")
            for warmup in range(args.warmups_per_mode):
                for mode in warmup_modes:
                    row = sample(mode, case, warmup)
                    artifact["warmups"].append(
                        {"case_id": row["case_id"], "mode": mode}
                    )
                    print(
                        f"[warmup] {mode} {row['case_id']} "
                        f"pp={row['prefill_tok_s']:.3f} tg={row['decode_tok_s']:.3f}",
                        flush=True,
                    )
            mode_counts = {"before": 0, "after": 0}
            sequence = arm_sequence(case_index)
            for slot, mode in enumerate(sequence):
                row = sample(mode, case, mode_counts[mode])
                mode_counts[mode] += 1
                row.update(
                    {
                        "mode": mode,
                        "sequence_slot": slot,
                        "case_sequence": list(sequence),
                    }
                )
                artifact["samples"].append(row)
                _write_json(args.output, artifact)
                print(
                    f"[measure {mode_counts[mode] - 1}] {mode} {row['case_id']} "
                    f"slot={slot} pp={row['prefill_tok_s']:.3f} "
                    f"tg={row['decode_tok_s']:.3f}",
                    flush=True,
                )
        try:
            artifact["summary"] = summarize_campaign_ab(
                artifact["samples"],
                repetitions_per_mode=args.repetitions_per_mode,
            )
        except ValueError as error:
            artifact["status"] = "failed_correctness_or_protocol"
            artifact["error"] = str(error)
            _write_json(args.output, artifact)
            return 2
        artifact["status"] = "completed"
        artifact["memory_before_close"] = memory_stats()
        _write_json(args.output, artifact)
        return 0
    finally:
        _apply_mode("after", route_package=args.route_package)
        if original_row4 is not None:
            register(row4_key, original_row4, replace=True)
        generator.close()
        artifact["memory_after_close"] = memory_stats()
        _write_json(args.output, artifact)


if __name__ == "__main__":
    raise SystemExit(main())
