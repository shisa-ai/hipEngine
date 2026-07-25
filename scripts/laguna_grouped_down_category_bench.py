#!/usr/bin/env python3
"""Gate one Laguna prefill route against the complete canonical AR suite."""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np

from hipengine.benchmark.provenance import collect_artifact_provenance
from hipengine.core.hip import HipMemcpyKind, get_hip_runtime
from hipengine.core.memory import host_array_ptr, memory_stats
from hipengine.loading.gguf import GGUFReader
from hipengine.runtime.laguna_gguf_runner import LagunaGGUFResidentSession
from hipengine.tokenization.gguf import LagunaGGUFTokenizer
from scripts.laguna_target_ar_bench import (
    DEFAULT_CACHE,
    DEFAULT_MODEL,
    DEFAULT_MODEL_SHA256,
    DEFAULT_ORACLE,
    DEFAULT_ORACLE_LOGPROBS,
    DEFAULT_PROMPTS,
    DEFAULT_TEMPLATE,
    RETAINED_HORIZONS,
    _compiler_version,
    _load_prompts,
    _normalized_log_probs,
    _oracle_gate,
    _progress,
    _repo_state,
    _session,
    _sha256_bytes,
)

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class CategoryComparison:
    """One explicit two-route category gate and its shape-screen contract."""

    name: str
    modes: tuple[str, str]
    aggregate_key: str
    screen_kind: str
    screen_status: str
    screen_decision_key: str
    require_positive_wall: bool
    execution_mode: str = "selected_down"
    require_exact_free_running: bool = True
    screen_requires_model: bool = True
    screen_candidate_variant: str | None = None
    require_shape_screen: bool = True
    require_performance_gate: bool = True


@dataclass(frozen=True)
class PrefillLaneConfiguration:
    """Explicit cumulative-prefill arithmetic and dispatch configuration."""

    f16_prefill_mode: str
    global_prefill_variant: str
    swa_prefill_variant: str
    selected_down_mode: str
    selected_gate_up_mode: str = "direct"
    f16_projection_mode: str = "retained"
    dense_q4_prefill_mode: str = "retained"


GROUPED_DOWN_COMPARISON = CategoryComparison(
    name="grouped_down",
    modes=("direct", "adaptive_grouped_smallm"),
    aggregate_key="adaptive_grouped_smallm_vs_direct",
    screen_kind="hipengine_laguna_prefill_grouped_down_ab",
    screen_status="retained",
    screen_decision_key="promotion",
    require_positive_wall=True,
)
GROUPED_COMBINE_COMPARISON = CategoryComparison(
    name="grouped_combine",
    modes=("adaptive_grouped_smallm", "adaptive_grouped_smallm_fused"),
    aggregate_key="grouped_combine_vs_grouped_smallm",
    screen_kind="hipengine_laguna_prefill_grouped_combine_ab",
    screen_status="screen_passed",
    screen_decision_key="screen",
    require_positive_wall=False,
)
F16_WMMA_COMP_SWA_COMPARISON = CategoryComparison(
    name="f16_wmma_comp_swa",
    modes=("tiled", "wmma_comp_swa"),
    aggregate_key="wmma_comp_swa_vs_tiled",
    screen_kind="hipengine_laguna_f16_wmma_screen",
    screen_status="quality_lane_admitted",
    screen_decision_key="summary",
    require_positive_wall=True,
    execution_mode="f16_prefill",
    require_exact_free_running=False,
    screen_requires_model=False,
    screen_candidate_variant="wmma_comp",
)
SWA_QROW2_COMPARISON = CategoryComparison(
    name="swa_qrow2",
    modes=("wave32_exact", "qrow2_32_exact"),
    aggregate_key="qrow2_32_exact_vs_wave32_exact",
    screen_kind="hipengine_laguna_prefill_ar_o5_swa_qrow2_ab",
    screen_status="retained",
    screen_decision_key="promotion",
    require_positive_wall=False,
    execution_mode="swa_prefill",
)
SWA_QROW2_ONLINE_COMPARISON = CategoryComparison(
    name="swa_qrow2_online",
    modes=("qrow2_32_exact", "qrow2_online"),
    aggregate_key="swa_qrow2_online_vs_qrow2_32_exact",
    screen_kind="hipengine_laguna_swa_qrow2_online_full_model_screen",
    screen_status="quality_lane_admitted",
    screen_decision_key="correctness",
    require_positive_wall=True,
    execution_mode="swa_prefill",
    require_exact_free_running=False,
    screen_candidate_variant="swa_context_rows_qrow2_online_spans",
)
GLOBAL_QROW2_ONLINE_COMPARISON = CategoryComparison(
    name="global_qrow2_online",
    modes=("global_exact", "global_qrow2_online"),
    aggregate_key="global_qrow2_online_vs_global_exact",
    screen_kind="hipengine_laguna_prefill_ar_o5_global_qrow2_online_ab",
    screen_status="quality_lane_admitted",
    screen_decision_key="promotion",
    require_positive_wall=True,
    execution_mode="global_prefill",
    require_exact_free_running=False,
)
CUMULATIVE_CONTROL_COMPARISON = CategoryComparison(
    name="cumulative_control",
    modes=("all_exact", "shipping_control"),
    aggregate_key="shipping_control_vs_all_exact",
    screen_kind="not_applicable",
    screen_status="not_applicable",
    screen_decision_key="not_applicable",
    require_positive_wall=False,
    execution_mode="cumulative_prefill",
    require_exact_free_running=False,
    screen_requires_model=False,
    require_shape_screen=False,
    require_performance_gate=False,
)
PREFILL_350_COMPARISON = CategoryComparison(
    name="prefill_350",
    modes=("shipping_control", "prefill_350_candidate"),
    aggregate_key="prefill_350_candidate_vs_shipping_control",
    screen_kind="not_applicable",
    screen_status="not_applicable",
    screen_decision_key="not_applicable",
    require_positive_wall=True,
    execution_mode="cumulative_prefill",
    require_exact_free_running=False,
    screen_requires_model=False,
    require_shape_screen=False,
)
PRODUCTION_ABSOLUTE_COMPARISON = CategoryComparison(
    name="production_absolute",
    modes=("all_exact", "production_absolute_candidate"),
    aggregate_key="production_absolute_candidate_vs_all_exact",
    screen_kind="not_applicable",
    screen_status="not_applicable",
    screen_decision_key="not_applicable",
    require_positive_wall=False,
    execution_mode="cumulative_prefill",
    require_exact_free_running=False,
    screen_requires_model=False,
    require_shape_screen=False,
    require_performance_gate=False,
)
_GLOBAL_PREFILL_VARIANTS = {
    "global_exact": "global_context_rows_spans",
    "global_qrow2_online": "global_context_rows_qrow2_online_spans",
}
_SWA_PREFILL_VARIANTS = {
    "wave32_exact": "swa_context_rows_wave32_exact_spans",
    "qrow2_32_exact": "swa_context_rows_qrow2_m128_c128_exact_spans",
    "qrow2_online": "swa_context_rows_qrow2_online_spans",
}
_PREFILL_LANE_CONFIGURATIONS = {
    "all_exact": PrefillLaneConfiguration(
        f16_prefill_mode="tiled",
        global_prefill_variant="global_context_rows_spans",
        swa_prefill_variant="swa_context_rows_qrow2_m128_c128_exact_spans",
        selected_down_mode="adaptive_grouped_smallm_fused",
    ),
    "shipping_control": PrefillLaneConfiguration(
        f16_prefill_mode="wmma_comp_swa",
        global_prefill_variant="global_context_rows_qrow2_online_spans",
        swa_prefill_variant="swa_context_rows_qrow2_online_spans",
        selected_down_mode="adaptive_grouped_smallm_fused",
    ),
    "prefill_350_candidate": PrefillLaneConfiguration(
        f16_prefill_mode="wmma_comp_swa",
        global_prefill_variant="global_context_rows_qrow2_online_spans",
        swa_prefill_variant="swa_context_rows_qrow2_online_spans",
        selected_down_mode="mmq64x32_d4_f32",
        selected_gate_up_mode="mmq128x32_d8_f32",
        f16_projection_mode="hipblaslt_scaled",
        dense_q4_prefill_mode="wmma_pack8",
    ),
    "production_absolute_candidate": PrefillLaneConfiguration(
        f16_prefill_mode="wmma_comp_swa",
        global_prefill_variant="global_context_rows_qrow2_online_spans",
        swa_prefill_variant="swa_context_rows_qrow2_online_spans",
        selected_down_mode="mmq64x32_d4_f32_rowvec",
        selected_gate_up_mode="mmq128x32_d8_f32_rowvec",
        f16_projection_mode="hipblaslt_scaled",
        dense_q4_prefill_mode="wmma_pack8",
    ),
}
_COMPARISONS = {
    comparison.name: comparison
    for comparison in (
        GROUPED_DOWN_COMPARISON,
        GROUPED_COMBINE_COMPARISON,
        F16_WMMA_COMP_SWA_COMPARISON,
        SWA_QROW2_COMPARISON,
        SWA_QROW2_ONLINE_COMPARISON,
        GLOBAL_QROW2_ONLINE_COMPARISON,
        CUMULATIVE_CONTROL_COMPARISON,
        PREFILL_350_COMPARISON,
        PRODUCTION_ABSOLUTE_COMPARISON,
    )
}
# Backward-compatible test/helper aliases for the retained grouped-down gate.
MODES = GROUPED_DOWN_COMPARISON.modes
BASELINE_MODE = MODES[0]
CANDIDATE_MODE = MODES[1]
DEFAULT_SCREEN = (
    ROOT
    / "benchmarks/results/2026-07-23-gfx1151-laguna-prefill-grouped-down-ab.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "benchmarks/results/2026-07-23-gfx1151-laguna-prefill-grouped-down-category.json"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--oracle", type=Path, default=DEFAULT_ORACLE)
    parser.add_argument("--oracle-logprobs", type=Path, default=DEFAULT_ORACLE_LOGPROBS)
    parser.add_argument("--shape-screen", type=Path, default=DEFAULT_SCREEN)
    parser.add_argument(
        "--comparison",
        choices=tuple(_COMPARISONS),
        default=GROUPED_DOWN_COMPARISON.name,
    )
    parser.add_argument("--backend", default="hip_gfx1151")
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument(
        "--output-horizons",
        type=lambda value: tuple(int(item) for item in value.split(",") if item),
        default=RETAINED_HORIZONS,
    )
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmup-output-tokens", type=int, default=2)
    parser.add_argument("--teacher-forced-tokens", type=int, default=32)
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--repacked-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--model-sha256", default=DEFAULT_MODEL_SHA256)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _mode_order(
    prompt_index: int,
    repetition: int,
    *,
    comparison: CategoryComparison = GROUPED_DOWN_COMPARISON,
) -> tuple[str, str]:
    modes = comparison.modes
    return (
        modes
        if (int(prompt_index) + int(repetition)) % 2 == 0
        else tuple(reversed(modes))
    )


@contextmanager
def _f16_prefill_mode(mode: str):
    previous = os.environ.get("HIPENGINE_LAGUNA_F16_PREFILL")
    os.environ["HIPENGINE_LAGUNA_F16_PREFILL"] = mode
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("HIPENGINE_LAGUNA_F16_PREFILL", None)
        else:
            os.environ["HIPENGINE_LAGUNA_F16_PREFILL"] = previous


def _prefill_for_mode(
    session: LagunaGGUFResidentSession,
    token_ids: Sequence[int],
    mode: str,
    comparison: CategoryComparison,
):
    f16_mode: str | None = None
    if comparison.execution_mode == "f16_prefill":
        f16_mode = mode
    elif comparison.execution_mode == "cumulative_prefill":
        f16_mode = _PREFILL_LANE_CONFIGURATIONS[mode].f16_prefill_mode
    if f16_mode is not None:
        with _f16_prefill_mode(f16_mode):
            return session.prefill(token_ids, use_bulk=True)
    return session.prefill(token_ids, use_bulk=True)


def _session_for_mode(
    owner: LagunaGGUFResidentSession,
    args: argparse.Namespace,
    mode: str,
    *,
    comparison: CategoryComparison = GROUPED_DOWN_COMPARISON,
) -> LagunaGGUFResidentSession:
    if mode not in comparison.modes:
        raise ValueError(f"unknown Laguna {comparison.name} mode {mode!r}")
    if comparison.execution_mode == "swa_prefill":
        return _session(
            owner,
            args,
            swa_prefill_variant=_SWA_PREFILL_VARIANTS[mode],
        )
    if comparison.execution_mode == "global_prefill":
        return _session(
            owner,
            args,
            global_prefill_variant=_GLOBAL_PREFILL_VARIANTS[mode],
        )
    if comparison.execution_mode == "cumulative_prefill":
        lane = _PREFILL_LANE_CONFIGURATIONS[mode]
        session = _session(
            owner,
            args,
            global_prefill_variant=lane.global_prefill_variant,
            swa_prefill_variant=lane.swa_prefill_variant,
        )
        session.set_selected_gate_up_mode(lane.selected_gate_up_mode)
        session.set_selected_down_mode(lane.selected_down_mode)
        session.set_f16_prefill_mode(lane.f16_projection_mode)
        session.set_dense_q4_prefill_mode(lane.dense_q4_prefill_mode)
        return session
    session = _session(owner, args)
    if comparison.execution_mode == "selected_down":
        session.set_selected_down_mode(mode)
    elif comparison.execution_mode not in {
        "f16_prefill",
        "global_prefill",
        "cumulative_prefill",
    }:
        raise ValueError(f"unknown Laguna execution mode {comparison.execution_mode!r}")
    return session


def _oracle_for_candidate(
    owner: LagunaGGUFResidentSession,
    args: argparse.Namespace,
    *,
    comparison: CategoryComparison = GROUPED_DOWN_COMPARISON,
) -> dict[str, Any]:
    candidate_mode = comparison.modes[1]
    if comparison.execution_mode == "f16_prefill":
        with _f16_prefill_mode(candidate_mode):
            return _oracle_gate(owner, args)
    if comparison.execution_mode == "swa_prefill":
        return _oracle_gate(
            owner,
            args,
            swa_prefill_variant=_SWA_PREFILL_VARIANTS[candidate_mode],
        )
    if comparison.execution_mode == "global_prefill":
        return _oracle_gate(
            owner,
            args,
            global_prefill_variant=_GLOBAL_PREFILL_VARIANTS[candidate_mode],
        )
    if comparison.execution_mode == "cumulative_prefill":
        lane = _PREFILL_LANE_CONFIGURATIONS[candidate_mode]

        def configure_session(session: LagunaGGUFResidentSession) -> None:
            session.set_selected_gate_up_mode(lane.selected_gate_up_mode)
            session.set_selected_down_mode(lane.selected_down_mode)
            session.set_f16_prefill_mode(lane.f16_projection_mode)
            session.set_dense_q4_prefill_mode(lane.dense_q4_prefill_mode)

        with _f16_prefill_mode(lane.f16_prefill_mode):
            return _oracle_gate(
                owner,
                args,
                global_prefill_variant=lane.global_prefill_variant,
                swa_prefill_variant=lane.swa_prefill_variant,
                session_configurator=configure_session,
            )
    return _oracle_gate(owner, args)


def _run_target_mode(
    owner: LagunaGGUFResidentSession,
    prompt: Mapping[str, Any],
    *,
    mode: str,
    horizons: Sequence[int],
    repetition: int,
    args: argparse.Namespace,
    comparison: CategoryComparison = GROUPED_DOWN_COMPARISON,
) -> dict[str, Any]:
    session = _session_for_mode(owner, args, mode, comparison=comparison)
    try:
        prefill_started = time.perf_counter()
        result = _prefill_for_mode(
            session,
            prompt["token_ids"],
            mode,
            comparison,
        )
        prefill_seconds = time.perf_counter() - prefill_started
        generated = [int(result.next_token_id)]
        decode_steps: list[float] = []
        while len(generated) < max(horizons):
            started = time.perf_counter()
            result = session.forward_token(result.next_token_id)
            decode_steps.append(time.perf_counter() - started)
            generated.append(int(result.next_token_id))
        checkpoints = {}
        for horizon in horizons:
            decode_seconds = float(sum(decode_steps[: max(0, horizon - 1)]))
            total_seconds = prefill_seconds + decode_seconds
            checkpoints[str(horizon)] = {
                "output_tokens": int(horizon),
                "generated_token_ids": generated[:horizon],
                "generated_ids_sha256": _sha256_bytes(
                    json.dumps(generated[:horizon], separators=(",", ":")).encode()
                ),
                "decode_forward_calls": max(0, int(horizon) - 1),
                "decode_seconds": decode_seconds,
                "decode_tok_s": (
                    (int(horizon) - 1) / decode_seconds if horizon > 1 else None
                ),
                "total_seconds": total_seconds,
                "e2e_output_tok_s": int(horizon) / total_seconds,
            }
        return {
            "prompt_id": prompt["id"],
            "category": prompt["category"],
            "prompt_tokens": prompt["prompt_tokens"],
            "prompt_token_ids_sha256": prompt["token_ids_sha256"],
            "mode": mode,
            "repetition": int(repetition),
            "prefill_seconds": prefill_seconds,
            "ttft_seconds": prefill_seconds,
            "prefill_tok_s": prompt["prompt_tokens"] / prefill_seconds,
            "checkpoints": checkpoints,
        }
    finally:
        session.close()


def _paired_free_running(
    rows: Sequence[Mapping[str, Any]],
    horizons: Sequence[int],
    *,
    comparison: CategoryComparison = GROUPED_DOWN_COMPARISON,
) -> dict[str, Any]:
    modes_required = comparison.modes
    baseline_mode, candidate_mode = modes_required
    grouped: dict[tuple[str, int], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        grouped[(str(row["prompt_id"]), int(row["repetition"]))][str(row["mode"])] = row
    comparisons = []
    for (prompt_id, repetition), modes in sorted(grouped.items()):
        if set(modes) != set(modes_required):
            raise ValueError(
                f"missing {comparison.name} pair for {prompt_id} repetition {repetition}"
            )
        checks = {}
        for horizon in horizons:
            baseline = modes[baseline_mode]["checkpoints"][str(horizon)][
                "generated_token_ids"
            ]
            candidate = modes[candidate_mode]["checkpoints"][str(horizon)][
                "generated_token_ids"
            ]
            checks[str(horizon)] = baseline == candidate
        comparisons.append(
            {
                "prompt_id": prompt_id,
                "repetition": repetition,
                "horizons_exact": checks,
                "pass": all(checks.values()),
            }
        )

    deterministic = True
    prompt_ids = {str(row["prompt_id"]) for row in rows}
    for mode in modes_required:
        for prompt_id in prompt_ids:
            selected = [
                row for row in rows if row["mode"] == mode and row["prompt_id"] == prompt_id
            ]
            for horizon in horizons:
                hashes = {
                    row["checkpoints"][str(horizon)]["generated_ids_sha256"]
                    for row in selected
                }
                deterministic = deterministic and len(hashes) == 1
    return {
        "all_pairs_exact": bool(all(item["pass"] for item in comparisons)),
        "same_mode_repeat_deterministic": bool(deterministic),
        "pairs": comparisons,
        "admission_role": (
            "exact mode pairs and deterministic repeats are required"
            if comparison.require_exact_free_running
            else "complete pair equality is reported; deterministic repeats are required"
        ),
    }


def _aggregate_selected(
    rows: Sequence[Mapping[str, Any]], horizons: Sequence[int]
) -> dict[str, Any]:
    prompt_tokens = sum(int(row["prompt_tokens"]) for row in rows)
    prefill_seconds = sum(float(row["prefill_seconds"]) for row in rows)
    result: dict[str, Any] = {
        "runs": len(rows),
        "prompt_tokens": prompt_tokens,
        "prefill_seconds": prefill_seconds,
        "prefill_tok_s": prompt_tokens / prefill_seconds,
        "ttft_median_seconds": statistics.median(float(row["ttft_seconds"]) for row in rows),
        "horizons": {},
    }
    for horizon in horizons:
        checkpoints = [row["checkpoints"][str(horizon)] for row in rows]
        decode_calls = sum(int(item["decode_forward_calls"]) for item in checkpoints)
        decode_seconds = sum(float(item["decode_seconds"]) for item in checkpoints)
        output_tokens = sum(int(item["output_tokens"]) for item in checkpoints)
        total_seconds = sum(float(item["total_seconds"]) for item in checkpoints)
        result["horizons"][str(horizon)] = {
            "output_tokens": output_tokens,
            "decode_forward_calls": decode_calls,
            "decode_seconds": decode_seconds,
            "decode_tok_s": decode_calls / decode_seconds,
            "total_seconds": total_seconds,
            "e2e_output_tok_s": output_tokens / total_seconds,
        }
    return result


def _aggregate(
    rows: Sequence[Mapping[str, Any]],
    horizons: Sequence[int],
    *,
    comparison: CategoryComparison = GROUPED_DOWN_COMPARISON,
) -> dict[str, Any]:
    baseline_mode, candidate_mode = comparison.modes
    result: dict[str, Any] = {}
    for mode in comparison.modes:
        selected = [row for row in rows if row["mode"] == mode]
        mode_result = _aggregate_selected(selected, horizons)
        mode_result["categories"] = {
            category: _aggregate_selected(
                [row for row in selected if row["category"] == category], horizons
            )
            for category in sorted({str(row["category"]) for row in selected})
        }
        result[mode] = mode_result

    baseline = result[baseline_mode]
    candidate = result[candidate_mode]
    comparison_result: dict[str, Any] = {
        "prefill_speedup": candidate["prefill_tok_s"] / baseline["prefill_tok_s"],
        "ttft_speedup": baseline["ttft_median_seconds"] / candidate["ttft_median_seconds"],
        "categories": {},
        "horizons": {},
    }
    for category in sorted(baseline["categories"]):
        base_category = baseline["categories"][category]
        candidate_category = candidate["categories"][category]
        comparison_result["categories"][category] = {
            "prefill_speedup": candidate_category["prefill_tok_s"]
            / base_category["prefill_tok_s"],
            "horizons": {
                str(horizon): {
                    "e2e_speedup": candidate_category["horizons"][str(horizon)][
                        "e2e_output_tok_s"
                    ]
                    / base_category["horizons"][str(horizon)]["e2e_output_tok_s"]
                }
                for horizon in horizons
            },
        }
    for horizon in horizons:
        base_checkpoint = baseline["horizons"][str(horizon)]
        candidate_checkpoint = candidate["horizons"][str(horizon)]
        comparison_result["horizons"][str(horizon)] = {
            "decode_speedup": candidate_checkpoint["decode_tok_s"]
            / base_checkpoint["decode_tok_s"],
            "e2e_speedup": candidate_checkpoint["e2e_output_tok_s"]
            / base_checkpoint["e2e_output_tok_s"],
        }
    result[comparison.aggregate_key] = comparison_result
    return result


def _teacher_forced_quality(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("teacher-forced quality requires at least one prompt")
    all_steps = [step for row in rows for step in row["steps"]]
    if not all_steps:
        raise ValueError("teacher-forced quality requires at least one step")
    max_kl = max(float(step["kl_divergence"]) for step in all_steps)
    finite = all(bool(step["finite"]) for step in all_steps)
    matches = sum(bool(step["top1_agreement"]) for step in all_steps)
    top1_agreement = matches / len(all_steps)
    categories: dict[str, Any] = {}
    failed: list[str] = []
    if not finite:
        failed.append("nonfinite_logits")
    if not math.isfinite(max_kl) or max_kl > 0.05:
        failed.append("max_kl_above_0.05")
    if top1_agreement < 0.9:
        failed.append("suite_top1_below_0.9")
    for category in sorted({str(row["category"]) for row in rows}):
        category_steps = [
            step for row in rows if row["category"] == category for step in row["steps"]
        ]
        category_matches = sum(bool(step["top1_agreement"]) for step in category_steps)
        agreement = category_matches / len(category_steps)
        category_max_kl = max(float(step["kl_divergence"]) for step in category_steps)
        category_finite = all(bool(step["finite"]) for step in category_steps)
        categories[category] = {
            "steps": len(category_steps),
            "top1_matches": category_matches,
            "top1_agreement": agreement,
            "max_kl_divergence": category_max_kl,
            "finite": category_finite,
        }
        if agreement < 0.9:
            failed.append(f"{category}_top1_below_0.9")
        if not category_finite:
            failed.append(f"{category}_nonfinite_logits")
    return {
        "pass": not failed,
        "failed_checks": failed,
        "steps": len(all_steps),
        "top1_matches": matches,
        "top1_agreement": top1_agreement,
        "max_kl_divergence": max_kl,
        "finite": finite,
        "categories": categories,
        "thresholds": {
            "max_kl_divergence": 0.05,
            "minimum_suite_top1_agreement": 0.9,
            "minimum_each_category_top1_agreement": 0.9,
        },
    }


def _copy_logits(
    session: LagunaGGUFResidentSession,
    result: Any,
    destination: np.ndarray,
) -> None:
    session.runtime.memcpy(
        host_array_ptr(destination),
        result.logits.ptr,
        destination.nbytes,
        HipMemcpyKind.DEVICE_TO_HOST,
    )


def _teacher_forced_prompt(
    owner: LagunaGGUFResidentSession,
    prompt: Mapping[str, Any],
    args: argparse.Namespace,
    *,
    comparison: CategoryComparison = GROUPED_DOWN_COMPARISON,
) -> dict[str, Any]:
    baseline_mode, candidate_mode = comparison.modes
    sessions = {
        mode: _session_for_mode(owner, args, mode, comparison=comparison)
        for mode in comparison.modes
    }
    logits = {
        mode: np.empty(sessions[mode].config.vocab_size, dtype=np.float32)
        for mode in comparison.modes
    }
    try:
        results = {
            mode: _prefill_for_mode(
                sessions[mode],
                prompt["token_ids"],
                mode,
                comparison,
            )
            for mode in comparison.modes
        }
        steps = []
        for index in range(args.teacher_forced_tokens):
            for mode in comparison.modes:
                _copy_logits(sessions[mode], results[mode], logits[mode])
            baseline_log_probs = _normalized_log_probs(logits[baseline_mode])
            candidate_log_probs = _normalized_log_probs(logits[candidate_mode])
            probabilities = np.exp(baseline_log_probs)
            kl = float(
                np.sum(probabilities * (baseline_log_probs - candidate_log_probs))
            )
            baseline_top1 = int(np.argmax(logits[baseline_mode]))
            candidate_top1 = int(np.argmax(logits[candidate_mode]))
            finite = bool(
                np.isfinite(logits[baseline_mode]).all()
                and np.isfinite(logits[candidate_mode]).all()
                and math.isfinite(kl)
            )
            steps.append(
                {
                    "index": index,
                    "teacher_token_id": baseline_top1,
                    "baseline_mode": baseline_mode,
                    "candidate_mode": candidate_mode,
                    f"{baseline_mode}_top1": baseline_top1,
                    f"{candidate_mode}_top1": candidate_top1,
                    "top1_agreement": baseline_top1 == candidate_top1,
                    "kl_divergence": kl,
                    "finite": finite,
                }
            )
            if index + 1 < args.teacher_forced_tokens:
                results = {
                    mode: sessions[mode].forward_token(baseline_top1)
                    for mode in comparison.modes
                }
    finally:
        for session in sessions.values():
            session.close()
    return {
        "prompt_id": prompt["id"],
        "category": prompt["category"],
        "prompt_tokens": prompt["prompt_tokens"],
        "prompt_token_ids_sha256": prompt["token_ids_sha256"],
        "steps": steps,
    }


def _promotion_gate(
    aggregate: Mapping[str, Any],
    free_running: Mapping[str, Any],
    teacher_forced: Mapping[str, Any],
    oracle: Mapping[str, Any],
    shape_screen: Mapping[str, Any],
    *,
    horizons: Sequence[int],
    recovered: bool,
    comparison: CategoryComparison = GROUPED_DOWN_COMPARISON,
) -> dict[str, Any]:
    comparison_result = aggregate[comparison.aggregate_key]
    failed: list[str] = []
    if not shape_screen["pass"]:
        failed.append("shape_screen_failed")
    if not teacher_forced["pass"]:
        failed.append("teacher_forced_quality_failed")
    if not oracle["pass"]:
        failed.append("poolside_oracle_failed")
    if not free_running["same_mode_repeat_deterministic"]:
        failed.append("free_running_repeat_not_deterministic")
    if comparison.require_exact_free_running and not free_running["all_pairs_exact"]:
        failed.append("free_running_pairs_not_exact")
    if not recovered:
        failed.append("tracked_lifecycle_not_recovered")
    if comparison.require_performance_gate:
        aggregate_prefill_speedup = float(comparison_result["prefill_speedup"])
        if comparison.require_positive_wall and aggregate_prefill_speedup <= 1.0:
            failed.append("aggregate_prefill_not_faster")
        elif (
            not comparison.require_positive_wall
            and aggregate_prefill_speedup < 0.995
        ):
            failed.append("aggregate_prefill_below_0.995")
        for category, values in comparison_result["categories"].items():
            prefill_speedup = float(values["prefill_speedup"])
            if comparison.require_positive_wall and prefill_speedup <= 1.0:
                failed.append(f"{category}_prefill_regressed")
            elif not comparison.require_positive_wall and prefill_speedup < 0.995:
                failed.append(f"{category}_prefill_below_0.995")
            for horizon in horizons:
                if float(values["horizons"][str(horizon)]["e2e_speedup"]) < 0.98:
                    failed.append(f"{category}_h{horizon}_e2e_below_0.98")
        for horizon in horizons:
            values = comparison_result["horizons"][str(horizon)]
            e2e_speedup = float(values["e2e_speedup"])
            if comparison.require_positive_wall and e2e_speedup <= 1.0:
                failed.append(f"h{horizon}_aggregate_e2e_not_faster")
            elif not comparison.require_positive_wall and e2e_speedup < 0.995:
                failed.append(f"h{horizon}_aggregate_e2e_below_0.995")
            decode_speedup = float(values["decode_speedup"])
            if not math.isfinite(decode_speedup) or not 0.98 <= decode_speedup <= 1.02:
                failed.append(f"h{horizon}_decode_outside_2pct")
    if not comparison.require_performance_gate:
        performance_policy = "reported; no admission threshold"
    elif comparison.require_positive_wall:
        performance_policy = (
            "aggregate and every-category prefill faster; aggregate E2E faster; "
            "each-category E2E >= 0.98x; decode within 2%"
        )
    else:
        performance_policy = (
            "aggregate/category prefill >=0.995x; aggregate E2E >=0.995x; "
            "each-category E2E >=0.98x; decode within 2%"
        )
    return {
        "pass": not failed,
        "failed_checks": failed,
        "policy": {
            "shape_screen": (
                "not applicable to the cumulative control ledger"
                if not comparison.require_shape_screen
                else (
                    "every M16-512 full/SWA family faster and M128 weighted >=2x"
                    if comparison.execution_mode == "f16_prefill"
                    else (
                        (
                            "quality-gated online SWA attention improves 512/1K/4K wall"
                            if comparison.name == SWA_QROW2_ONLINE_COMPARISON.name
                            else "exact matrix512/attention128 512/1K/4K output, KV, and wall win"
                        )
                        if comparison.execution_mode == "swa_prefill"
                        else (
                            "quality-gated online global attention improves 512/1K/4K wall"
                            if comparison.execution_mode == "global_prefill"
                            else (
                                "direct fallback >=0.995x; rows>=32 grouped shapes "
                                "and aggregate faster"
                                if comparison.require_positive_wall
                                else "each shape >=0.995x; aggregate >=0.998x; exact micro win"
                            )
                        )
                    )
                )
            ),
            "quality": "KL <= 0.05 and top-1 >= 90% suite-wide and per category",
            "free_running_ids": (
                "all mode pairs exact and same-mode repeats deterministic"
                if comparison.require_exact_free_running
                else "report complete pair equality; same-mode repeats deterministic"
            ),
            "performance": performance_policy,
            "lifecycle": "tracked allocations return exactly to baseline",
        },
    }


def _load_shape_screen(
    args: argparse.Namespace,
    *,
    comparison: CategoryComparison = GROUPED_DOWN_COMPARISON,
) -> dict[str, Any]:
    if not comparison.require_shape_screen:
        return {
            "pass": True,
            "path": None,
            "sha256": None,
            "revision": None,
            "aggregate_speedup": None,
            "grouped_min_rows": None,
            "model_sha256": args.model_sha256,
            "candidate_variant": None,
            "comparison": comparison.name,
            "role": "not_applicable_control_ledger",
        }
    artifact = json.loads(args.shape_screen.read_text(encoding="utf-8"))
    decision = artifact.get(comparison.screen_decision_key, {})
    model = artifact.get("model", {})
    candidate_variant = artifact.get("protocol", {}).get(
        "candidate_variant"
    ) or artifact.get("candidate", {}).get("variant")
    passed = bool(
        artifact.get("kind") == comparison.screen_kind
        and artifact.get("status") == comparison.screen_status
        and artifact.get("pass")
        and decision.get("pass")
        and not decision.get("regressed_rows")
        and (
            not comparison.screen_requires_model
            or model.get("sha256") == args.model_sha256
        )
        and (
            comparison.screen_candidate_variant is None
            or candidate_variant == comparison.screen_candidate_variant
        )
    )
    if comparison.require_positive_wall:
        passed = bool(passed and not decision.get("non_improving_grouped_rows"))
    return {
        "pass": passed,
        "path": str(args.shape_screen.resolve()),
        "sha256": _sha256_bytes(args.shape_screen.read_bytes()),
        "revision": artifact.get("repo", {}).get("revision"),
        "aggregate_speedup": decision.get("effective_speedup")
        or decision.get("m128_weighted_projection_sum", {}).get("speedup"),
        "grouped_min_rows": decision.get("grouped_min_rows"),
        "model_sha256": model.get("sha256"),
        "candidate_variant": candidate_variant,
        "comparison": comparison.name,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    comparison = _COMPARISONS[str(args.comparison)]
    horizons = tuple(int(value) for value in args.output_horizons)
    if horizons != RETAINED_HORIZONS:
        raise ValueError(
            f"retained {comparison.name} gate requires horizons {RETAINED_HORIZONS}"
        )
    if args.repetitions < 3:
        raise ValueError(
            f"retained {comparison.name} gate requires at least three repetitions"
        )
    if args.chunk_size != 128:
        raise ValueError(f"retained {comparison.name} gate requires chunk size 128")
    if args.teacher_forced_tokens != max(RETAINED_HORIZONS):
        raise ValueError(
            f"retained {comparison.name} gate requires 32 teacher-forced tokens"
        )
    if args.warmup_output_tokens <= 0:
        raise ValueError("warmup output tokens must be positive")
    if not args.model.is_file():
        raise FileNotFoundError(f"Laguna model not found: {args.model}")
    if comparison.require_shape_screen and not args.shape_screen.is_file():
        raise FileNotFoundError(
            f"Laguna {comparison.name} shape screen not found: {args.shape_screen}"
        )
    if not args.model_sha256:
        raise ValueError("--model-sha256 is required")
    repo = _repo_state()
    if not repo["tracked_clean"]:
        raise RuntimeError(
            f"retained {comparison.name} category gate requires a clean tracked worktree"
        )
    shape_screen = _load_shape_screen(args, comparison=comparison)
    if not shape_screen["pass"]:
        raise ValueError(f"Laguna {comparison.name} shape screen is not accepted")

    provenance = collect_artifact_provenance(
        repo_root=ROOT,
        configured_backend=args.backend,
        resolved_backend=args.backend,
        target_arch=args.backend.removeprefix("hip_"),
        model_path=args.model,
        quant="gguf_q4_k_m",
        kv_dtype="bf16",
        command=(str(Path(sys.executable).resolve()), *sys.argv),
        build_profile=f"laguna_prefill_{comparison.name}_category",
        timing_protocol=(
            f"same_owner_balanced_{comparison.modes[0]}_vs_"
            f"{comparison.modes[1]}_category_h16_h32"
        ),
        warmups=len(comparison.modes),
        repetitions=args.repetitions,
    )
    reader = GGUFReader(args.model)
    tokenizer = LagunaGGUFTokenizer.from_gguf_info(reader.info)
    prompts = _load_prompts(args.prompts, tokenizer)

    runtime = get_hip_runtime()
    gpu_free_before, gpu_total = runtime.mem_get_info()
    tracked_before = memory_stats()
    owner: LagunaGGUFResidentSession | None = None
    rows: list[dict[str, Any]] = []
    teacher_rows: list[dict[str, Any]] = []
    load_started = time.perf_counter()
    try:
        owner = LagunaGGUFResidentSession(
            args.model,
            context_length=args.context_length,
            backend=args.backend,
            runtime=runtime,
            compiler_version=_compiler_version(args.compiler_version_file),
            require_cached_build=args.require_cached_build,
            progress=_progress,
            repacked_cache=args.repacked_cache,
            model_sha256=args.model_sha256,
            prefill_chunk_size=args.chunk_size,
        )
        load_seconds = time.perf_counter() - load_started
        for mode in comparison.modes:
            _run_target_mode(
                owner,
                prompts[0],
                mode=mode,
                horizons=(int(args.warmup_output_tokens),),
                repetition=-1,
                args=args,
                comparison=comparison,
            )
        for repetition in range(args.repetitions):
            for prompt_index, prompt in enumerate(prompts):
                for mode in _mode_order(
                    prompt_index,
                    repetition,
                    comparison=comparison,
                ):
                    row = _run_target_mode(
                        owner,
                        prompt,
                        mode=mode,
                        horizons=horizons,
                        repetition=repetition,
                        args=args,
                        comparison=comparison,
                    )
                    rows.append(row)
                    print(
                        f"rep={repetition} prompt={prompt['id']} mode={mode} "
                        f"prefill={row['prefill_tok_s']:.3f} tok/s",
                        file=sys.stderr,
                        flush=True,
                    )
        for prompt in prompts:
            row = _teacher_forced_prompt(
                owner,
                prompt,
                args,
                comparison=comparison,
            )
            teacher_rows.append(row)
            matches = sum(bool(step["top1_agreement"]) for step in row["steps"])
            max_kl = max(float(step["kl_divergence"]) for step in row["steps"])
            print(
                f"teacher prompt={prompt['id']} top1={matches}/{len(row['steps'])} "
                f"max_kl={max_kl:.6g}",
                file=sys.stderr,
                flush=True,
            )
        oracle = _oracle_for_candidate(
            owner,
            args,
            comparison=comparison,
        )
        resident_nbytes = owner.resident_nbytes
    finally:
        if owner is not None:
            owner.close()
    tracked_after = memory_stats()
    gpu_free_after, gpu_total_after = runtime.mem_get_info()
    if gpu_total_after != gpu_total:
        raise RuntimeError(
            f"HIP total memory changed during {comparison.name} category gate"
        )

    free_running = _paired_free_running(
        rows,
        horizons,
        comparison=comparison,
    )
    teacher_forced = _teacher_forced_quality(teacher_rows)
    aggregate = _aggregate(rows, horizons, comparison=comparison)
    recovered = bool(
        tracked_after["current_allocated_bytes"] == tracked_before["current_allocated_bytes"]
        and tracked_after["active_allocations"] == tracked_before["active_allocations"]
    )
    promotion = _promotion_gate(
        aggregate,
        free_running,
        teacher_forced,
        oracle,
        shape_screen,
        horizons=horizons,
        recovered=recovered,
        comparison=comparison,
    )
    passed = bool(promotion["pass"])
    manifest_path = args.repacked_cache / "manifest.json"
    prompt_payload = args.prompts.read_bytes()
    return {
        "schema": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "kind": f"hipengine_laguna_prefill_{comparison.name}_category",
        "status": "retained_category_gate" if passed else "rejected_category_gate",
        "pass": passed,
        "performance_claim": bool(passed and comparison.require_positive_wall),
        "performance_claim_scope": (
            f"same-owner Laguna {comparison.modes[0]} versus {comparison.modes[1]} "
            "over all ten canonical category prompts at h16/h32; model load excluded; "
            + (
                "retained full-model performance gate"
                if comparison.require_positive_wall
                else (
                    "cumulative quality ledger; performance is reported but not gated"
                    if not comparison.require_performance_gate
                    else "quality and full-model non-regression gate only"
                )
            )
        ),
        "provenance": provenance,
        "repo": repo,
        "model": {
            "path": str(args.model.resolve()),
            "sha256": args.model_sha256,
            "quant": "Q4_K_M mixed GGUF v3",
            "repacked_cache": str(args.repacked_cache.resolve()),
            "repacked_cache_manifest_sha256": (
                _sha256_bytes(manifest_path.read_bytes()) if manifest_path.is_file() else None
            ),
        },
        "platform": {
            "backend": args.backend,
            "target_arch": args.backend.removeprefix("hip_"),
            "device_name": provenance["device_name"],
            "machine": platform.machine(),
            "hip_total_bytes": gpu_total,
        },
        "protocol": {
            "comparison": comparison.name,
            "modes": list(comparison.modes),
            "chunk_size": args.chunk_size,
            "output_horizons": list(horizons),
            "repetitions": args.repetitions,
            "warmup_output_tokens_per_mode": args.warmup_output_tokens,
            "teacher_forced_tokens_per_prompt": args.teacher_forced_tokens,
            "timed_order": (
                f"alternating {comparison.modes[0]}/{comparison.modes[1]} per prompt "
                "and reversed next repetition"
            ),
            "timing_scope": "prefill plus fixed-horizon decode; resident model load excluded",
            "prompt_suite": str(args.prompts.resolve()),
            "prompt_suite_sha256": _sha256_bytes(prompt_payload),
            "activation_quantization_included": bool(
                comparison.execution_mode
                in {"f16_prefill", "cumulative_prefill"}
            ),
            "decode_route": "identical exact c=1 path for both modes",
            "prefill_lane_configurations": (
                {
                    mode: {
                        "f16_prefill_mode": _PREFILL_LANE_CONFIGURATIONS[
                            mode
                        ].f16_prefill_mode,
                        "global_prefill_variant": _PREFILL_LANE_CONFIGURATIONS[
                            mode
                        ].global_prefill_variant,
                        "swa_prefill_variant": _PREFILL_LANE_CONFIGURATIONS[
                            mode
                        ].swa_prefill_variant,
                        "selected_down_mode": _PREFILL_LANE_CONFIGURATIONS[
                            mode
                        ].selected_down_mode,
                        "selected_gate_up_mode": _PREFILL_LANE_CONFIGURATIONS[
                            mode
                        ].selected_gate_up_mode,
                        "f16_projection_mode": _PREFILL_LANE_CONFIGURATIONS[
                            mode
                        ].f16_projection_mode,
                        "dense_q4_prefill_mode": _PREFILL_LANE_CONFIGURATIONS[
                            mode
                        ].dense_q4_prefill_mode,
                    }
                    for mode in comparison.modes
                }
                if comparison.execution_mode == "cumulative_prefill"
                else None
            ),
        },
        "shape_screen": shape_screen,
        "load": {"seconds_excluded": load_seconds, "resident_nbytes": resident_nbytes},
        "rows": rows,
        "quality": {
            "poolside_oracle": oracle,
            "teacher_forced": teacher_forced,
            "teacher_forced_prompts": teacher_rows,
            "free_running": free_running,
            "tracked_returned_to_baseline": recovered,
        },
        "aggregate": aggregate,
        "promotion": promotion,
        "memory": {
            "tracked_before": tracked_before,
            "tracked_after": tracked_after,
            "gpu_free_before": gpu_free_before,
            "gpu_free_after": gpu_free_after,
            "hip_total_bytes": gpu_total,
        },
        "command": [str(Path(sys.executable).resolve()), *sys.argv],
        "notes": [
            "Both modes share resident weights and use isolated bounded request sessions.",
            (
                "The all-exact lane freezes tiled source-F16 plus exact global and "
                "context-qualified exact SWA attention. The shipping lane freezes "
                "gfx1151 compensated source-F16 plus online global/SWA attention; "
                "both freeze the same exact grouped-down and dense/shared routes."
                if comparison.execution_mode == "cumulative_prefill"
                else (
                    "Compensated WMMA applies only to the 36 SWA layers from M16; all "
                    "12 full-attention layers stay exact tiled, M2-15 stay exact tiled, "
                    "and rows==1 stays on GEMV."
                    if comparison.execution_mode == "f16_prefill"
                    else (
                        "Online global prefill changes softmax association only on the 12 "
                        "full-attention layers; all 36 SWA layers and decode stay unchanged."
                        if comparison.execution_mode == "global_prefill"
                        else (
                            "Online SWA prefill changes softmax association only on the 36 "
                            "sliding-attention layers; global attention and decode stay unchanged."
                            if comparison.name == SWA_QROW2_ONLINE_COMPARISON.name
                            else (
                                "Adaptive grouped down stays BF16 throughout and falls back to "
                                "direct below 32 rows."
                                if comparison.require_positive_wall
                                else "The candidate preserves both BF16 boundaries while removing "
                                "one launch and the selected-output round trip for rows >=32."
                            )
                        )
                    )
                )
            ),
            "Teacher forcing feeds baseline-route top-1 IDs to both routes and compares "
            "full logits.",
            "Complete free-running IDs are reported, while KL/top-1 thresholds remain "
            "authoritative.",
            "AR decode uses the identical exact c=1 route in both modes.",
        ],
    }


def main() -> int:
    args = _parse_args()
    result = run(args)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")
    sys.stdout.write(payload)
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
