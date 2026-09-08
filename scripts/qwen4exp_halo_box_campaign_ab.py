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
import time
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
Q4_BUNDLE_ENV = "HIPENGINE_QWEN4_EXP_Q4_BUNDLE_PREFILL"
Q51_PAIR_ENV = "HIPENGINE_QWEN4_EXP_Q51_PAIR_PREFILL"
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


def staged_slots(case_index: int, stage: int) -> list[tuple[int, str]]:
    if stage not in (0,1):
        raise ValueError("invalid screen stage")
    slots=list(enumerate(arm_sequence(case_index)))
    return slots[:2] if stage==0 else slots[2:]


def clear_screen_losses(samples, wall_increase):
    if not 0 < wall_increase < 1:
        raise ValueError("invalid screening threshold")
    summary=summarize_campaign_ab(samples,repetitions_per_mode=1)
    if len(summary["by_case"])!=12:
        raise ValueError("staged screen requires all12 cases")
    losses=[]
    for case in sorted(summary["by_case"]):
        rows={r["mode"]:r for r in samples if r["case_id"]==case}
        before,after=rows["before"],rows["after"]
        if (after["prefill_ms"]>before["prefill_ms"]*(1+wall_increase)
                and after["prefill_ms"]+after["decode_ms"]>
                (before["prefill_ms"]+before["decode_ms"])*(1+wall_increase)):
            losses.append(case)
    return losses if len(losses)>=2 else []


def measurement_sequence(case_index: int, repetitions: int) -> tuple[str, ...]:
    if repetitions not in (1,3):
        raise ValueError("only one-pair screen or three-pair publication supported")
    sequence=arm_sequence(case_index)
    return sequence[:2] if repetitions==1 else sequence


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
            "within_mode_deterministic": deterministic if expected>1 else None,
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

    result = {
        "correctness": {
            "within_mode_deterministic": True if expected>1 else None,
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
    if expected==1:
        for mode in ("before","after"):
            for case in result[mode]["cases"].values():
                for metric in case.values():
                    if isinstance(metric,dict):
                        for key in ("stddev","stdev","standard_deviation","coefficient_of_variation"):
                            if key in metric:
                                metric[key]=None
        result["uncertainty"]="Within-case repeatability/variance unmeasured; one sample per arm."
    return result


def q8_down_row4_expected_calls(prompt_tokens: int, chunk_size: int) -> int:
    if prompt_tokens < 1 or chunk_size < 1:
        raise ValueError("positive prompt and chunk sizes required")
    full, tail = divmod(prompt_tokens, chunk_size)
    return 4 * (full * (chunk_size >= 512) + (tail >= 512))


def q51_fold128_expected_calls(prompt_tokens: int, chunk_size: int) -> int:
    if prompt_tokens < 1 or chunk_size < 1:
        raise ValueError("positive prompt and chunk sizes required")
    full, tail = divmod(prompt_tokens, chunk_size)
    return 25 * (full * (chunk_size >= 64) + (tail >= 64))


def q51_fold_pair_expected_calls(prompt_tokens: int, chunk_size: int) -> int:
    if prompt_tokens < 1 or chunk_size < 1:
        raise ValueError("positive prompt and chunk sizes required")
    full, tail = divmod(prompt_tokens, chunk_size)
    return 25 * (full * (chunk_size >= 512) + (tail >= 512))


def q8_mmq_vec4_expected_calls(prompt_tokens: int, chunk_size: int) -> int:
    if prompt_tokens < 1 or chunk_size < 1:
        raise ValueError("positive prompt and chunk sizes required")
    full, tail = divmod(prompt_tokens, chunk_size)
    return 72 * (full * (chunk_size >= 64) + (tail >= 64))


def q8_mmq_raw_vector_expected_calls(prompt_tokens: int, chunk_size: int) -> int:
    if prompt_tokens < 1 or chunk_size < 1:
        raise ValueError("positive prompt and chunk sizes required")
    full,tail = divmod(prompt_tokens,chunk_size)
    return 242*(full*(chunk_size>=64)+(tail>=64))


def q8_bundle_call_in_scope(route_package: str, call_args: Sequence[object]) -> bool:
    if route_package not in {"q8-mapped-down","q8-down-bundle"}:
        return True
    if len(call_args) < 3:
        raise ValueError("Q8 bundle call requires a lane-map argument")
    mapped = bool(call_args[2])
    return mapped if route_package == "q8-mapped-down" else not mapped


def q8_mapped_down_expected_calls(prompt_tokens: int, chunk_size: int) -> int:
    if prompt_tokens < 1 or chunk_size < 1:
        raise ValueError("positive prompt and chunk sizes required")
    full,tail = divmod(prompt_tokens,chunk_size)
    return full*(chunk_size>=512)+(tail>=512)


def q8_down_register_expected_calls(tokens: int,chunk: int) -> int:
    return q8_down_row4_expected_calls(tokens,chunk)+q8_mapped_down_expected_calls(tokens,chunk)


def gdn_wave_norm_expected_calls(tokens: int, chunk: int) -> int:
    # Layers0..26 exclude six QSA layers; later GDN layers use tiled prefill.
    full, tail = divmod(tokens, chunk)
    return 21 * (full * (chunk >= 2) + (tail >= 2))

def mmq_token64_expected_calls(tokens: int, chunk: int) -> int:
    full,tail=divmod(tokens,chunk)
    return 12*(full*(chunk>=512)+(tail>=512))


def apply_chunk_mode(runner: Any, mode: str, *, allocated_chunk_size: int) -> None:
    if mode not in {"before","after"}:
        raise ValueError("invalid chunk mode")
    chunk = 512 if mode=="before" else 1024
    if allocated_chunk_size < chunk:
        raise ValueError("chunk exceeds preallocated capacity")
    runner.prefill_chunk_size = chunk


def validate_chunk_coverage(chunks: Sequence[int], tokens: int, size: int) -> None:
    if tokens < 1 or size < 1:
        raise ValueError("positive token and chunk sizes required")
    expected = [min(size,tokens-start) for start in range(0,tokens,size)]
    if list(chunks) != expected:
        raise ValueError(f"chunk coverage {list(chunks)} != {expected}")


def _apply_mode(
    mode: str,
    *,
    environment: MutableMapping[str, str] = os.environ,
    route_package: str = "pf13",
) -> None:
    if route_package == "chunk1024":
        if mode not in {"before","after"}:
            raise ValueError("invalid chunk mode")
        return
    if route_package in {"q5k-row4", "qsa-h256-wave", "qsa-h256-page256", "q4-bundle", "q51-pair", "gdn-register", "q4-pair", "q8-wave-scale", "gr-wave-scale", "q8-mmq-prepack", "q8-down-row4", "q51-fold128", "q8-down-bundle", "q51-fold-pair", "q8-mmq-vec4", "q51-register-cache", "q8-mmq-raw-vector", "q8-mapped-down", "q8-down-register", "q51-row-publish", "gdn-wave-norm", "mmq-token64", "qsa-head-pair", "qsa-head-quad", "q4-iu8-exact", "q51-iu8-exact", "q5k-iu8-exact", "qsa-ordered-v2", "gr-iu8", "gr-iu8-down", "q8-iu8-dense"}:
        if mode not in {"before", "after"}:
            raise ValueError(f"invalid campaign A/B mode {mode!r}")
        flag = ROW4_ENV if route_package == "q5k-row4" else QSA_H256_ENV
        if route_package == "q4-bundle":
            flag = Q4_BUNDLE_ENV
        if route_package == "q51-pair":
            flag = Q51_PAIR_ENV
        if route_package == "gdn-register":
            flag = "HIPENGINE_QWEN4_EXP_GDN_REGISTER_PREFILL"
        if route_package == "gdn-wave-norm":
            flag = "HIPENGINE_QWEN4_EXP_GDN_WAVE_NORM"
        if route_package == "mmq-token64":
            flag = "HIPENGINE_QWEN4_EXP_MMQ_TOKEN64"
        if route_package in {"qsa-head-pair","qsa-head-quad"}:
            flag = "HIPENGINE_QWEN4_EXP_QSA_HEAD_PAIR"
        if route_package == "q4-pair":
            flag = "HIPENGINE_QWEN4_EXP_Q4_PAIR_PREFILL"
        if route_package == "q4-iu8-exact":
            flag = "HIPENGINE_QWEN4_EXP_Q4_IU8_EXACT"
        if route_package == "q51-iu8-exact":
            flag = "HIPENGINE_QWEN4_EXP_Q51_IU8_EXACT"
        if route_package == "q5k-iu8-exact":
            flag = "HIPENGINE_QWEN4_EXP_Q5_K_IU8_EXACT"
        if route_package == "qsa-ordered-v2":
            flag = "HIPENGINE_QWEN4_EXP_QSA_ORDERED_DECODE_V2"
        if route_package == "gr-iu8":
            flag = "HIPENGINE_QWEN4_EXP_GR_IU8"
        if route_package == "gr-iu8-down":
            flag = "HIPENGINE_QWEN4_EXP_GR_IU8_DOWN"
        if route_package == "q8-iu8-dense":
            flag = "HIPENGINE_QWEN4_EXP_Q8_IU8_WMM"
        if route_package == "q8-wave-scale":
            flag = "HIPENGINE_QWEN4_EXP_Q8_WAVE_SCALE"
        if route_package == "gr-wave-scale":
            flag = "HIPENGINE_QWEN4_EXP_GR_WAVE_SCALE"
        if route_package == "q8-mmq-prepack":
            flag = "HIPENGINE_QWEN4_EXP_Q8_MMQ_PREPACK"
        if route_package == "q8-mmq-vec4":
            flag = "HIPENGINE_QWEN4_EXP_Q8_MMQ_VEC4"
        if route_package == "q8-mmq-raw-vector":
            flag = "HIPENGINE_QWEN4_EXP_Q8_MMQ_RAW_VECTOR"
        if route_package == "q8-down-row4":
            flag = "HIPENGINE_QWEN4_EXP_Q8_DOWN_ROW4_PREFILL"
        if route_package == "q51-fold128":
            flag = "HIPENGINE_QWEN4_EXP_Q51_FOLD128_PREFILL"
        if route_package == "q8-down-bundle":
            flag = "HIPENGINE_QWEN4_EXP_Q8_DOWN_BUNDLE_PREFILL"
        if route_package == "q8-mapped-down":
            flag = "HIPENGINE_QWEN4_EXP_Q8_MAPPED_DOWN"
        if route_package == "q8-down-register":
            flag = "HIPENGINE_QWEN4_EXP_Q8_DOWN_REGISTER"
        if route_package == "q51-fold-pair":
            flag = "HIPENGINE_QWEN4_EXP_Q51_FOLD_PAIR_PREFILL"
        if route_package == "q51-register-cache":
            flag = "HIPENGINE_QWEN4_EXP_Q51_REGISTER_CACHE"
        if route_package == "q51-row-publish":
            flag = "HIPENGINE_QWEN4_EXP_Q51_ROW_PUBLISH"
        environment[flag] = "1" if mode == "after" else "0"
        if route_package=="qsa-head-quad" and mode=="after":
            environment[flag]="quad"
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
    parser.add_argument("--screen-only", action="store_true",
                        help="Permit one pair/case; diagnostic only, no promotion evidence")
    parser.add_argument("--staged-screen",action="store_true",
                        help="One full-suite pair, then stop for multiple clear losses or finish remaining pairs in-residency")
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--route-package", choices=("pf13", "q5k-row4", "qsa-h256-wave", "qsa-h256-page256", "q4-bundle", "q51-pair", "gdn-register", "q4-pair", "q8-wave-scale", "gr-wave-scale", "q8-mmq-prepack", "q8-down-row4", "q51-fold128", "q8-down-bundle", "q51-fold-pair", "q8-mmq-vec4", "q51-register-cache", "q8-mmq-raw-vector", "q8-mapped-down", "chunk1024", "q8-down-register", "q51-row-publish", "gdn-wave-norm", "mmq-token64", "qsa-head-pair", "qsa-head-quad", "q4-iu8-exact", "q51-iu8-exact", "q5k-iu8-exact", "qsa-ordered-v2", "gr-iu8", "gr-iu8-down", "q8-iu8-dense"), default="pf13")
    parser.add_argument("--case-id", action="append", help="Diagnostic subset; omitted for full gate")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.warmups_per_mode < 1:
        raise SystemExit("--warmups-per-mode must be at least 1")
    if args.screen_only and args.repetitions_per_mode != 1:
        raise SystemExit("screen-only requires --repetitions-per-mode 1")
    if not args.screen_only and args.repetitions_per_mode != 3:
        raise SystemExit("publication protocol requires --repetitions-per-mode 3")
    if args.staged_screen and (args.screen_only or args.repetitions_per_mode!=3 or args.case_id):
        raise SystemExit("staged-screen requires full suite,three repeats,no screen-only")
    if args.route_package == "chunk1024" and args.prefill_chunk_size != 1024:
        raise SystemExit("chunk1024 requires --prefill-chunk-size1024 for shared allocation")
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
            "arm_order_even_case": list(measurement_sequence(0,args.repetitions_per_mode)),
            "arm_order_odd_case": list(measurement_sequence(1,args.repetitions_per_mode)),
            "qualification": "screen-one-pair" if args.screen_only else "canonical-three-pair",
            "promotion_eligible_protocol": not args.screen_only,
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
    if args.route_package == "q8-iu8-dense":
        # Registered after construction: profile construction re-registers
        # the base families and would otherwise clobber the counting hook.
        from hipengine.kernels.registry import KernelKey, register, resolve
        row4_key = KernelKey("hip_gfx1100", "linear", "gguf_q8_0",
                             "iu8_wmma_prefill_f32_f32_out")
        original_row4 = resolve(
            backend=row4_key.backend, layer=row4_key.layer,
            quant=row4_key.quant, variant=row4_key.variant)

        def counted_row4(*call_args, **call_kwargs):
            row4_calls[0] += 1
            return original_row4(*call_args, **call_kwargs)

        register(row4_key, counted_row4, replace=True)
        artifact["arms"] = {
            "before": {"q8_dense":
                "coltile8_rowbatch4_wave_scale_f32_f32_out (exact)"},
            "after": {"q8_dense":
                "iu8_wmma_prefill_f32_f32_out (T1 3-plane)"},
        }
    observed_chunks = []
    original_chunk = None
    if args.route_package == "chunk1024":
        original_chunk = generator.runner._prefill_chunk
        def counted_chunk(token_ids, **kwargs):
            observed_chunks.append(len(token_ids))
            return original_chunk(token_ids,**kwargs)
        generator.runner._prefill_chunk = counted_chunk
        artifact["arms"] = {"before":{"prefill_chunk_size":512},
                            "after":{"prefill_chunk_size":1024}}
        artifact["protocol"]["allocated_chunk_size"] = args.prefill_chunk_size
        artifact["protocol"]["prefill_chunk_size"] = "arm_specific_512_or_1024"
        artifact["protocol"]["memory_scope"] = (
            "Both arms share1024-capacity PLE/prefill/MMQ scratch, allocated before timing. "
            "Does not measure512-sized allocation vs1024-sized allocation.")
        artifact["protocol"]["correctness_scope"] = (
            "Exact generated trajectories are required by this screening harness. "
            "A mismatch requires full declared production numerical/state gates, not relaxed checks.")
    row4_calls = [0]
    register_mapped_calls = [0]
    original_row4 = None
    gr_iu8_module = None
    if args.route_package in {"gr-iu8", "gr-iu8-down"}:
        import hipengine.runtime.qwen4_exp_runner as _gr_runner_module
        gr_iu8_module = _gr_runner_module
        # The up (320->10240) and down (10240->320) legs share one kernel
        # entry; the production profile may already bind the other leg, so
        # each package counts only its own geometry.
        down_leg = args.route_package == "gr-iu8-down"

        def counted_gr_iu8(*call_args, **call_kwargs):
            in_features = call_args[4]
            out_features = call_args[5]
            if (in_features > out_features) == down_leg:
                row4_calls[0] += 1
            return original_row4(*call_args, **call_kwargs)

        original_row4 = _gr_runner_module.gguf_q8_0_iu8_wmma_prefill_f32_f32
        _gr_runner_module.gguf_q8_0_iu8_wmma_prefill_f32_f32 = counted_gr_iu8
        if down_leg:
            artifact["arms"] = {
                "before": {"gr_down":
                    "gguf_k_prefill_out_coltile_rowbatch f32 (exact coltile)"},
                "after": {"gr_down":
                    "q8_0_iu8_wmma_prefill_f32_f32 (T1 3-plane)"},
            }
        else:
            artifact["arms"] = {
                "before": {"gr_up":
                    "q8_0_gr_up_sigmoid_mean_coltile2_branch4_rowbatch4_f32 (fused exact)"},
                "after": {"gr_up":
                    "q8_0_iu8_wmma_prefill_f32_f32 + sigmoid + gated_mean (T1 3-plane)"},
            }
    if args.route_package in {"q5k-row4", "qsa-h256-wave", "qsa-h256-page256", "q4-bundle", "q51-pair", "gdn-register", "q4-pair", "q8-wave-scale", "gr-wave-scale", "q8-mmq-prepack", "q8-down-row4", "q51-fold128", "q8-down-bundle", "q51-fold-pair", "q8-mmq-vec4", "q51-register-cache", "q8-mmq-raw-vector", "q8-mapped-down", "q8-down-register", "q51-row-publish", "gdn-wave-norm", "mmq-token64", "qsa-head-pair", "qsa-head-quad", "q4-iu8-exact", "q51-iu8-exact", "q5k-iu8-exact", "qsa-ordered-v2"}:
        from hipengine.kernels.registry import KernelKey, register, resolve
        row4_key = (KernelKey(
            "hip_gfx1151", "linear", "gguf_q5_k",
            "selected_grouped_row4_gemv_bf16_bf16_out")
            if args.route_package == "q5k-row4" else KernelKey(
                "hip_gfx1151", "qsa_sparse_attention", "bf16_kv",
                "strict_h256_page256_wave_rows_spans" if args.route_package == "qsa-h256-page256"
                else "strict_h256_wave_rows_spans"))
        if args.route_package == "q4-bundle":
            row4_key = KernelKey(
                "hip_gfx1151", "moe_linear", "gguf_q4_k",
                "selected_dual_grouped_rowbatch8_out4_expertgrid64_bundle_bf16_bf16_out")
        if args.route_package == "q51-pair":
            row4_key = KernelKey(
                "hip_gfx1151", "moe_linear", "gguf_q5_1",
                "selected_grouped_prefill_pair2_bf16_bf16_out")
        if args.route_package == "gdn-register":
            row4_key = KernelKey(
                "hip_gfx1151", "gdn_recurrence_norm_gate", "f32_state",
                "qwen4exp_sigmoid_register_prefill")
        if args.route_package == "q4-pair":
            row4_key = KernelKey(
                "hip_gfx1151", "moe_linear", "gguf_q4_k",
                "selected_dual_grouped_pair2_bf16_bf16_out")
        if args.route_package == "q4-iu8-exact":
            row4_key = KernelKey(
                "hip_gfx1151", "moe_linear", "gguf_q4_k",
                "selected_dual_wmma_iu8_risk_prefill_bf16_bf16_out")
        if args.route_package == "q51-iu8-exact":
            row4_key = KernelKey(
                "hip_gfx1151", "moe_linear", "gguf_q5_1",
                "selected_wmma_iu8_risk_prefill_bf16_bf16_out")
        if args.route_package == "q5k-iu8-exact":
            row4_key = KernelKey(
                "hip_gfx1151", "moe_linear", "gguf_q5_k",
                "selected_dual_wmma_iu8_risk_prefill_bf16_bf16_out")
        if args.route_package == "qsa-ordered-v2":
            row4_key = KernelKey(
                "hip_gfx1151", "qsa_sparse_attention", "bf16_kv",
                "strict_ordered_three_pass_v2_spans")
        if args.route_package == "q8-wave-scale":
            row4_key = KernelKey(
                "hip_gfx1151", "linear", "gguf_q8_0",
                "coltile8_rowbatch4_wave_scale_f32_f32_out")
        if args.route_package == "gr-wave-scale":
            row4_key = KernelKey(
                "hip_gfx1151", "linear+gr_gated_mean", "gguf_q8_0",
                "coltile2_branch4_rowbatch4_wave_scale_f32_exact")
        if args.route_package == "q8-mmq-prepack":
            row4_key = KernelKey(
                "hip_gfx1151", "linear", "gguf_q8_0",
                "mmq128_prepacked_q8_1_d4x3_guarded_f32_f32_out")
        if args.route_package == "q8-mmq-vec4":
            row4_key = KernelKey(
                "hip_gfx1151", "linear", "gguf_q8_0",
                "mmq128_prepacked_vec4_q8_1_d4x3_guarded_f32_f32_out")
        if args.route_package == "q8-mmq-raw-vector":
            row4_key = KernelKey("hip_gfx1151","linear","gguf_q8_0",
                                "mmq128_raw_vec4_q8_1_d4x3_guarded_f32_f32_out")
        if args.route_package == "q8-down-row4":
            row4_key = KernelKey(
                "hip_gfx1151", "linear", "gguf_q8_0",
                "selected_grouped_row4_gemv_bf16_bf16_out")
        if args.route_package == "q51-fold128":
            row4_key = KernelKey(
                "hip_gfx1151", "moe_linear", "gguf_q5_1",
                "selected_grouped_prefill_pair2_fold128_bf16_bf16_out")
        if args.route_package in {"q8-down-bundle","q8-mapped-down"}:
            row4_key = KernelKey(
                "hip_gfx1151", "linear", "gguf_q8_0",
                "selected_grouped_row4_bundle_gemv_bf16_bf16_out")
        if args.route_package == "q51-fold-pair":
            row4_key = KernelKey(
                "hip_gfx1151", "moe_linear", "gguf_q5_1",
                "selected_grouped_prefill_pair2_fold128_pair_bf16_bf16_out")
        if args.route_package == "q51-register-cache":
            row4_key = KernelKey(
                "hip_gfx1151", "moe_linear", "gguf_q5_1",
                "selected_grouped_prefill_pair2_register_cache_bf16_bf16_out")
        if args.route_package == "q51-row-publish":
            row4_key = KernelKey(
                "hip_gfx1151", "moe_linear", "gguf_q5_1",
                "selected_grouped_prefill_pair2_row_publish_bf16_bf16_out")
        if args.route_package=="q8-down-register":
            row4_key=KernelKey("hip_gfx1151","linear","gguf_q8_0",
                              "selected_grouped_row4_register_gemv_bf16_bf16_out")
        if args.route_package == "gdn-wave-norm":
            row4_key = KernelKey("hip_gfx1151","gdn_recurrence_norm_gate",
                                "f32_state","qwen4exp_sigmoid_wave_norm_prefill")
        if args.route_package == "mmq-token64":
            row4_key=KernelKey("hip_gfx1151","linear","gguf_q8_0",
                              "mmq128_token64_q8_1_d4x3_guarded_f32_f32_out")
        if args.route_package=="qsa-head-pair":
            row4_key=KernelKey("hip_gfx1151","qsa_sparse_attention","bf16_kv",
                              "strict_h256_head_pair_rows_spans")
        if args.route_package=="qsa-head-quad":
            row4_key=KernelKey("hip_gfx1151","qsa_sparse_attention","bf16_kv",
                              "strict_h256_head_quad_rows_spans")
        original_row4 = resolve(
            backend=row4_key.backend, layer=row4_key.layer,
            quant=row4_key.quant, variant=row4_key.variant)

        def counted_row4(*call_args, **call_kwargs):
            if args.route_package=="q8-down-register" and call_args[2]:
                register_mapped_calls[0]+=1
            if q8_bundle_call_in_scope(args.route_package,call_args):
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
        elif args.route_package == "q4-bundle":
            artifact["arms"] = {
                "before": {"q4_gate_up": "selected_dual_grouped_rowbatch8_out4_expertgrid64_bf16_bf16_out"},
                "after": {"q4_gate_up": row4_key.variant},
            }
        elif args.route_package == "q51-pair":
            artifact["arms"] = {
                "before": {"q51_down": "selected_grouped_prefill_compact_rowbatch8_out8_expertgrid64_m1_bf16_bf16_out"},
                "after": {"q51_down": row4_key.variant},
            }
        elif args.route_package == "gdn-register":
            artifact["arms"] = {
                "before": {"serial_gdn": "qwen4exp_sigmoid_strict_prefill"},
                "after": {"serial_gdn": row4_key.variant},
            }
        elif args.route_package == "q4-pair":
            artifact["arms"] = {
                "before": {"q4_gate_up": "selected_dual_grouped_rowbatch8_out4_expertgrid64_bundle_bf16_bf16_out"},
                "after": {"q4_gate_up": row4_key.variant},
            }
        elif args.route_package == "q8-wave-scale":
            artifact["arms"] = {
                "before": {"exact_q8_prefill": "coltile8_rowbatch4_f32_f32_out"},
                "after": {"exact_q8_prefill": row4_key.variant},
            }
        elif args.route_package == "gr-wave-scale":
            artifact["arms"] = {
                "before": {"gr_up": "coltile2_branch4_rowbatch4_f32_exact"},
                "after": {"gr_up": row4_key.variant},
            }
        elif args.route_package == "q8-mmq-prepack":
            artifact["arms"] = {
                "before": {"qkv_ssm_mmq": "mmq128_prefill_q8_1_d4x3_guarded_f32_f32_out"},
                "after": {"qkv_ssm_mmq": row4_key.variant},
            }
        elif args.route_package == "q8-mmq-vec4":
            artifact["arms"] = {
                "before": {"qkv_ssm_mmq": "mmq128_prepacked_q8_1_d4x3_guarded_f32_f32_out"},
                "after": {"qkv_ssm_mmq": row4_key.variant},
            }
        elif args.route_package == "q8-mmq-raw-vector":
            artifact["arms"] = {
                "before": {"raw_mmq":"mmq128_prefill_q8_1_d4x3_guarded_f32_f32_out"},
                "after": {"raw_mmq":row4_key.variant},
            }
        elif args.route_package == "q8-down-row4":
            artifact["arms"] = {
                "before": {"grouped_q8_down": "selected_grouped_gemv_bf16_bf16_out"},
                "after": {"grouped_q8_down": row4_key.variant},
            }
        elif args.route_package == "q51-fold128":
            artifact["arms"] = {
                "before": {"q51_down": "selected_grouped_prefill_pair2_bf16_bf16_out"},
                "after": {"q51_down": row4_key.variant},
            }
        elif args.route_package == "q8-down-bundle":
            artifact["arms"] = {
                "before": {"q8_down": "selected_grouped_row4_gemv_bf16_bf16_out"},
                "after": {"q8_down": row4_key.variant},
            }
        elif args.route_package == "q8-mapped-down":
            artifact["arms"] = {
                "before": {"token_major_q8_down":"selected_gemv_bf16_bf16_out"},
                "after": {"token_major_q8_down":row4_key.variant},
            }
        elif args.route_package == "q51-fold-pair":
            artifact["arms"] = {
                "before": {"q51_down": "selected_grouped_prefill_pair2_fold128_bf16_bf16_out"},
                "after": {"q51_down": row4_key.variant},
            }
        elif args.route_package == "q51-register-cache":
            artifact["arms"] = {
                "before": {"q51_down": "selected_grouped_prefill_pair2_fold128_pair_bf16_bf16_out"},
                "after": {"q51_down": row4_key.variant},
            }
    artifact["route_package"] = args.route_package
    if args.route_package=="q8-down-register":
        artifact["arms"] = {
            "before":{"q8_down":"selected_grouped_row4_bundle_gemv_bf16_bf16_out"},
            "after":{"q8_down":"selected_grouped_row4_register_gemv_bf16_bf16_out"}}
    artifact["diagnostic_subset"] = bool(args.case_id)
    artifact["screen_only"] = args.screen_only
    if args.route_package=="qsa-head-quad":
        artifact["arms"]={
            "before":{"sparse_attention":"strict_h256_page256_wave_rows_spans"},
            "after":{"sparse_attention":"strict_h256_head_quad_rows_spans"}}
    if args.route_package=="qsa-head-pair":
        artifact["arms"]={
            "before":{"sparse_attention":"strict_h256_page256_wave_rows_spans"},
            "after":{"sparse_attention":"strict_h256_head_pair_rows_spans"}}
    if args.route_package=="qsa-ordered-v2":
        artifact["arms"]={
            "before":{"sparse_decode":"strict_ordered_three_pass_spans"},
            "after":{"sparse_decode":"strict_ordered_three_pass_v2_spans"}}
    if args.staged_screen:
        artifact["protocol"].update(
            qualification="staged-one-then-three",
            case_order="fixture order per stage; warmup once per case in first stage",
            early_stop_wall_increase=0.05,
            early_stop_minimum_cases=2,
            early_promotion=False)
    if args.route_package=="mmq-token64":
        artifact["arms"]={
            "before":{"raw_q":"mmq128_raw_vec4_q8_1_d4x3_guarded_f32_f32_out"},
            "after":{"raw_q":"mmq128_token64_q8_1_d4x3_guarded_f32_f32_out"}}
    if args.route_package == "gdn-wave-norm":
        artifact["arms"] = {
            "before":{"gdn":"qwen4exp_sigmoid_register_prefill"},
            "after":{"gdn":"qwen4exp_sigmoid_wave_norm_prefill"}}
    if args.route_package == "q51-row-publish":
        artifact["arms"] = {
            "before": {"q51_down": "selected_grouped_prefill_pair2_register_cache_bf16_bf16_out"},
            "after": {"q51_down": "selected_grouped_prefill_pair2_row_publish_bf16_bf16_out"}}

    def sample(mode, case, repetition):
        _apply_mode(mode, route_package=args.route_package)
        if args.route_package == "chunk1024":
            apply_chunk_mode(generator.runner,mode,allocated_chunk_size=args.prefill_chunk_size)
            observed_chunks.clear()
        start_calls = row4_calls[0]
        start_mapped = register_mapped_calls[0]
        row = _hipengine_case_sample(
            generator.runner, case=case, repetition=repetition, transitions=transitions)
        if args.route_package == "chunk1024":
            validate_chunk_coverage(observed_chunks,int(case["prompt_tokens"]),
                                    generator.runner.prefill_chunk_size)
            row["executed_prefill_chunks"] = list(observed_chunks)
            row["active_chunk_size"] = generator.runner.prefill_chunk_size
        if original_row4 is not None:
            calls = row4_calls[0] - start_calls
            if args.route_package=="mmq-token64":
                expected=mmq_token64_expected_calls(
                    int(case["prompt_tokens"]),args.prefill_chunk_size) if mode=="after" else 0
                assert calls==expected,(calls,expected)
            if args.route_package == "gdn-wave-norm":
                expected=gdn_wave_norm_expected_calls(
                    int(case["prompt_tokens"]),args.prefill_chunk_size) if mode=="after" else 0
                assert calls==expected,(calls,expected)
            if args.route_package=="q8-down-register":
                expected=q8_down_register_expected_calls(int(case["prompt_tokens"]),args.prefill_chunk_size) if mode=="after" else 0
                mapped=register_mapped_calls[0]-start_mapped
                assert calls==expected,(calls,expected)
                assert mapped==(q8_mapped_down_expected_calls(int(case["prompt_tokens"]),args.prefill_chunk_size) if mode=="after" else 0)
                row["mapped_candidate_calls"]=mapped
                row["compact_candidate_calls"]=calls-mapped
            elif args.route_package == "q8-mapped-down":
                expected_calls = q8_mapped_down_expected_calls(
                    int(case["prompt_tokens"]),args.prefill_chunk_size) if mode=="after" else 0
                if calls != expected_calls:
                    raise AssertionError(f"Mapped Q8 down calls {calls} != {expected_calls}")
            elif args.route_package == "q8-mmq-raw-vector":
                expected_calls = q8_mmq_raw_vector_expected_calls(
                    int(case["prompt_tokens"]),args.prefill_chunk_size) if mode=="after" else 0
                if calls != expected_calls:
                    raise AssertionError(f"Raw MMQ vector calls {calls} != {expected_calls}")
            elif args.route_package == "q8-mmq-vec4":
                expected_calls = q8_mmq_vec4_expected_calls(
                    int(case["prompt_tokens"]), args.prefill_chunk_size) if mode == "after" else 0
                if calls != expected_calls:
                    raise AssertionError(f"Q8 MMQ vec4 calls {calls} != {expected_calls}")
            elif args.route_package in {"q8-down-row4", "q8-down-bundle"}:
                expected_calls = q8_down_row4_expected_calls(
                    int(case["prompt_tokens"]), args.prefill_chunk_size) if mode == "after" else 0
                if calls != expected_calls:
                    raise AssertionError(f"Q8 down row4 calls {calls} != {expected_calls}")
            elif args.route_package == "q51-fold128":
                expected_calls = q51_fold128_expected_calls(
                    int(case["prompt_tokens"]), args.prefill_chunk_size) if mode == "after" else 0
                if calls != expected_calls:
                    raise AssertionError(f"Q51 fold128 calls {calls} != {expected_calls}")
            elif args.route_package in {"q51-fold-pair", "q51-register-cache", "q51-row-publish"}:
                expected_calls = q51_fold_pair_expected_calls(
                    int(case["prompt_tokens"]), args.prefill_chunk_size) if mode == "after" else 0
                if calls != expected_calls:
                    raise AssertionError(f"Q51 folded pair calls {calls} != {expected_calls}")
            elif args.route_package.startswith("qsa-h256-") or args.route_package in {"qsa-head-pair","qsa-head-quad"}:
                validate_qsa_h256_engagement(mode, calls, int(case["prompt_tokens"]))
            elif args.route_package == "qsa-ordered-v2":
                expected = (transitions * 12
                            if mode == "after" and int(case["prompt_tokens"]) == 4096
                            else 0)
                if calls != expected:
                    raise AssertionError(
                        f"ordered v2 decode calls {calls} != {expected}")
            elif args.route_package == "gr-iu8-down":
                prompt_tokens = int(case["prompt_tokens"])
                full_chunks, tail = divmod(prompt_tokens, args.prefill_chunk_size)
                routed_chunks = full_chunks + (1 if tail > 256 else 0)
                expected = 96 * routed_chunks if mode == "after" else 0
                if calls != expected:
                    raise AssertionError(
                        f"GR iu8 down calls {calls} != {expected}")
            elif args.route_package == "gr-iu8":
                # 96 GR up sites (48 layers x attention+ffn), one call per
                # prefill chunk with rows > 256.
                prompt_tokens = int(case["prompt_tokens"])
                full_chunks, tail = divmod(prompt_tokens, args.prefill_chunk_size)
                routed_chunks = full_chunks + (1 if tail > 256 else 0)
                expected = 96 * routed_chunks if mode == "after" else 0
                if calls != expected:
                    raise AssertionError(
                        f"GR iu8 calls {calls} != {expected}")
            elif args.route_package == "q8-iu8-dense":
                # attn_gate (36/chunk) + shared-expert down (48/chunk) plus
                # any other coltile-family dense Q8_0 projections.
                prompt_tokens = int(case["prompt_tokens"])
                full_chunks, tail = divmod(prompt_tokens, args.prefill_chunk_size)
                routed_chunks = full_chunks + (1 if tail > 256 else 0)
                expected = (84 * routed_chunks) if mode == "after" else 0
                if calls < expected:
                    raise AssertionError(
                        f"Q8 dense iu8 calls {calls} < {expected}")
            else:
                validate_row4_engagement(mode, calls)
            row["candidate_calls"] = calls
        return row

    try:
        if args.route_package == "q8-mmq-prepack":
            _apply_mode("after",route_package=args.route_package)
            prepare_start = time.perf_counter()
            generator.runner.configure_mmq_prefill_resources()
            owner = generator.runner._q8_mmq_weight_sidecars
            artifact["sidecar"] = {
                "prepare_seconds": time.perf_counter()-prepare_start,
                "bytes": owner.nbytes,
                "count": len(owner.mapping),
                "protocol": "Prepared before either timing arm; raw weights remain resident.",
            }
        stages = (0,1) if args.staged_screen else (None,)
        measured_repetitions=args.repetitions_per_mode
        for stage in stages:
          if stage==1:
            try:
                losses=clear_screen_losses(artifact["samples"],0.05)
            except ValueError as error:
                artifact["status"]="failed_correctness_or_protocol"
                artifact["error"]=str(error)
                _write_json(args.output,artifact)
                return 2
            artifact["screen_decision"]={
                "clear_loss_cases":losses,
                "action":"stop_diagnostic" if losses else "continue_same_residency",
                "first_stage_sample_count":len(artifact["samples"])}
            _write_json(args.output,artifact)
            if losses:
                artifact["screen_only"]=True
                artifact["protocol"]["promotion_eligible_protocol"]=False
                artifact["protocol"]["measured_repetitions_per_mode_per_case"]=1
                artifact["protocol"]["arm_order_even_case"]=list(measurement_sequence(0,1))
                artifact["protocol"]["arm_order_odd_case"]=list(measurement_sequence(1,1))
                measured_repetitions=1
                break
          for case in cases:
            case_index = fixture_case_index(fixture["cases"], case)
            warmup_modes = ("before", "after") if case_index % 2 == 0 else ("after", "before")
            for warmup in range(0 if stage==1 else args.warmups_per_mode):
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
            mode_counts = {"before": int(stage==1), "after": int(stage==1)}
            sequence = measurement_sequence(case_index,args.repetitions_per_mode)
            slots=staged_slots(case_index,stage) if stage is not None else enumerate(sequence)
            for slot, mode in slots:
                row = sample(mode, case, mode_counts[mode])
                mode_counts[mode] += 1
                row.update(
                    {
                        "mode": mode,
                        "sequence_slot": slot,
                        "case_sequence": list(sequence),
                        "measurement_stage":stage,
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
                repetitions_per_mode=measured_repetitions,
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
        if args.route_package == "gr-iu8":
            os.environ["HIPENGINE_QWEN4_EXP_GR_IU8"] = "0"
        elif args.route_package == "q8-iu8-dense":
            os.environ["HIPENGINE_QWEN4_EXP_Q8_IU8_WMM"] = "0"
        else:
            _apply_mode("after", route_package=args.route_package)
        if original_chunk is not None:
            generator.runner._prefill_chunk = original_chunk
            generator.runner.prefill_chunk_size = args.prefill_chunk_size
        if gr_iu8_module is not None:
            gr_iu8_module.gguf_q8_0_iu8_wmma_prefill_f32_f32 = original_row4
            if args.route_package == "gr-iu8-down":
                os.environ["HIPENGINE_QWEN4_EXP_GR_IU8_DOWN"] = "0"
        elif original_row4 is not None:
            register(row4_key, original_row4, replace=True)
        generator.close()
        artifact["memory_after_close"] = memory_stats()
        _write_json(args.output, artifact)


if __name__ == "__main__":
    raise SystemExit(main())
