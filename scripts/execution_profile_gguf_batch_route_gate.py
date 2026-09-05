#!/usr/bin/env python3
"""Requalify GGUF packed batch routes against independent strict-c1 teachers.

The default profile retains the gfx1151 Q8T16 c4/c8 candidate protocol. The
explicit ``current_package_direct`` profile instead evaluates the current
package's direct c3/c5/c6/c7 routes and their width transitions without forcing
candidate environment overrides. Both profiles exercise sparse physical-c8
masks and exact candidate repeats. The capture is numerical and transition
evidence only: it does not fabricate full control telemetry, task verdicts, or
a runtime-resolved production-profile manifest.

``--packed-prefill-candidate`` changes only the *candidate's* prompt admission:
the group enters through packed multi-request prefill instead of one session at
a time. That route persists ``segmented_in_place_final_state`` while the strict
teacher persists per-token-exact state, so the prefill calibration drops from
byte-exact logits to token equality and every decode-step KL/top-1 threshold
still applies unchanged. The artifact records
``packed_prefill_candidate`` so a run can never be read as the canonical
serial-admission protocol.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import sys
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.benchmark.execution_profiles import (  # noqa: E402
    EvaluationThresholds,
    RowDescriptor,
    compare_profile_logits,
)
from hipengine.benchmark.provenance import collect_artifact_provenance  # noqa: E402
from scripts.execution_profile_gdn_calibration import (  # noqa: E402
    CalibrationError,
    _file_sha256,
    _matrix_sha256,
    _trajectory_arrays,
    _trajectory_sha256,
    validate_strict_baseline,
)
from scripts.gguf_gdn_semantic_gate import (  # noqa: E402
    DEFAULT_PROMPTS,
    _configure_gate_environment,
    _load_suites,
)
from scripts.gguf_gdn_trajectory_gate import (  # noqa: E402
    _gdn_mode,
    _run_logits_trajectory,
)
from scripts.gguf_mtp_bench import build_chat_prompt  # noqa: E402
from scripts.gguf_mtp_category_bench import prompt_sha256  # noqa: E402

KIND = "hipengine_execution_profile_gguf_batch_route_requalification_capture"
DEFAULT_MODEL = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
DEFAULT_HISTORICAL_SOURCE = Path(
    "benchmarks/results/2026-07-19-gfx1151-gguf-f3-q8t16-rowtile-rejected.json"
)
POLICY_CAPABILITY = "GGUF_Q8_T16_DECODE_ROWTILE_ALL"
POLICY_MIN_ROWS_CAPABILITY = "GGUF_Q8_T16_DECODE_ROWTILE_MIN_ROWS"
POLICY_ENV = "HIPENGINE_GGUF_Q8_T16_ROWTILE_ALL"
PAIR_MIN_ROWS_CAPABILITY = "GGUF_Q8_T16_DECODE_PAIR_ROWTILE_MIN_ROWS"
SELECTED_PAIR_MIN_ROWS_CAPABILITY = "GGUF_Q4_T16_SELECTED_PAIRREUSE_MIN_ROWS"
PAIR_POLICY_ENV = "HIPENGINE_GGUF_Q8_T16_PAIR_ROWTILE"
PAIR_COL8_ENV = "HIPENGINE_GGUF_Q8_T16_PAIR_COL8"
ROUTE_PROFILE_Q4_SELECTED_PAIR = "q4_selected_pair_candidate"
SELECTED_PAIR_ENV = "HIPENGINE_GGUF_T16_SELECTED_PAIRREUSE"
SELECTED_DOWN_PAIR_ENV = "HIPENGINE_GGUF_T16_SELECTED_DOWN_PAIRREUSE"
SELECTED_Q6_DOWN_PAIR_ENV = "HIPENGINE_GGUF_T16_SELECTED_Q6_DOWN_PAIRREUSE"
ROUTER_COOP_ENV = "HIPENGINE_GGUF_ROUTER_F32W_COOP"
ROUTER_PERSISTENT_ENV = "HIPENGINE_GGUF_ROUTER_F32W_PERSISTENT_COUNTER"
_router_candidate_enabled = False
DEFAULT_GDN_MODE = "chain_lds32_direct_nonvolatile"
ROUTE_PROFILE_Q8T16 = "q8t16_candidate"
ROUTE_PROFILE_Q8T16_PAIR = "q8t16_pair_candidate"
ROUTE_PROFILE_CURRENT_DIRECT = "current_package_direct"
SUPPORTED_WIDTHS = frozenset(range(1, 9))
DIRECT_STATIC_WIDTHS = frozenset({3, 5, 6, 7})
DEFAULT_DYNAMIC_SCHEDULE = ((0, 8), (6, 4), (12, 2), (18, 1))
DIRECT_DYNAMIC_SCHEDULE = ((0, 7), (6, 6), (12, 5), (18, 3))


@dataclass(frozen=True, slots=True)
class BatchRouteCapture:
    scenario_id: str
    request_id: str
    category: str
    strict: tuple[Mapping[str, object], ...]
    candidate_runs: tuple[tuple[Mapping[str, object], ...], ...]
    shapes: tuple[str, ...]
    transitions: tuple[str, ...]
    teacher_steps: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.scenario_id or not self.request_id or not self.category:
            raise ValueError("batch capture identifiers must be non-empty")
        lengths = {
            len(self.strict),
            len(self.shapes),
            len(self.transitions),
            len(self.teacher_steps),
        }
        if len(lengths) != 1 or not self.strict:
            raise ValueError("batch capture metadata must align with a non-empty strict trajectory")
        if len(self.candidate_runs) < 3:
            raise ValueError("batch route capture requires at least three candidate runs")
        if any(len(run) != len(self.strict) for run in self.candidate_runs):
            raise ValueError("candidate runs must align with strict trajectory")


def validate_width_schedule(
    schedule: Sequence[tuple[int, int]], *, decode_steps: int
) -> tuple[tuple[int, int], ...]:
    """Validate one packed-width retirement schedule."""

    values = tuple((int(step), int(width)) for step, width in schedule)
    if not values or values[0][0] != 0:
        raise ValueError("width schedule must start at step zero")
    if int(decode_steps) <= 0:
        raise ValueError("decode horizon must be positive")
    if any(step < 0 or step >= int(decode_steps) for step, _ in values):
        raise ValueError("width schedule step is outside decode horizon")
    if any(width not in SUPPORTED_WIDTHS for _, width in values):
        raise ValueError("width schedule contains an unsupported physical width")
    if any(values[index][0] >= values[index + 1][0] for index in range(len(values) - 1)):
        raise ValueError("width schedule steps must strictly increase")
    if any(values[index][1] <= values[index + 1][1] for index in range(len(values) - 1)):
        raise ValueError("width schedule widths must strictly descend")
    return values


def _trajectory_from_results(results: Sequence[Any]) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "token_id": int(result.token_id),
            "logits": np.ascontiguousarray(result.logits, dtype=np.float32),
        }
        for result in results
    )


def build_batch_route_quality(
    captures: Sequence[BatchRouteCapture],
    *,
    thresholds: EvaluationThresholds | None = None,
) -> dict[str, Any]:
    """Evaluate aligned batch-route captures with shape/transition attribution."""

    values = tuple(captures)
    if not values:
        raise ValueError("batch route quality needs at least one capture")
    rows: list[RowDescriptor] = []
    strict_chunks: list[np.ndarray] = []
    candidate_chunks: list[np.ndarray] = []
    repeat_hashes: list[list[str]] = []
    repeat_mismatches: list[dict[str, Any]] = []
    scenario_step = 0
    repeat_count: int | None = None
    for capture in values:
        if repeat_count is None:
            repeat_count = len(capture.candidate_runs)
        elif repeat_count != len(capture.candidate_runs):
            raise ValueError("candidate repeat count differs across captures")
        strict_logits, strict_ids = _trajectory_arrays(capture.strict)
        first_logits, first_ids = _trajectory_arrays(capture.candidate_runs[0])
        if strict_logits.shape != first_logits.shape:
            raise ValueError("strict and batch candidate trajectories are not aligned")
        strict_chunks.append(strict_logits)
        candidate_chunks.append(first_logits)
        hashes = [_trajectory_sha256(run) for run in capture.candidate_runs]
        repeat_hashes.append(hashes)
        for repeat_index, run in enumerate(capture.candidate_runs[1:], start=1):
            logits, token_ids = _trajectory_arrays(run)
            logits_exact = bool(np.array_equal(first_logits, logits))
            ids_exact = first_ids == token_ids
            if not logits_exact or not ids_exact:
                repeat_mismatches.append(
                    {
                        "scenario_id": capture.scenario_id,
                        "request_id": capture.request_id,
                        "repeat_index": repeat_index,
                        "logits_exact": logits_exact,
                        "selected_token_ids_exact": ids_exact,
                    }
                )
        for index, token_id in enumerate(strict_ids):
            rows.append(
                RowDescriptor(
                    scenario_id=capture.scenario_id,
                    scenario_step=scenario_step,
                    request_id=capture.request_id,
                    teacher_step=int(capture.teacher_steps[index]),
                    category=capture.category,
                    shape=capture.shapes[index],
                    transition=capture.transitions[index],
                    teacher_token_id=int(token_id),
                )
            )
            scenario_step += 1
    strict_matrix = np.concatenate(strict_chunks, axis=0)
    candidate_matrix = np.concatenate(candidate_chunks, axis=0)
    quality = compare_profile_logits(
        strict_matrix,
        candidate_matrix,
        rows,
        thresholds=thresholds,
    )
    return {
        "quality": quality,
        "repeat_determinism": {
            "runs": int(repeat_count or 0),
            "passed": not repeat_mismatches,
            "mismatches": repeat_mismatches,
            "trajectory_sha256_by_capture": repeat_hashes,
        },
        "strict_logits_sha256": _matrix_sha256(strict_matrix),
        "candidate_logits_sha256": _matrix_sha256(candidate_matrix),
        "strict_selected_token_ids_sha256": hashlib.sha256(
            np.asarray([row.teacher_token_id for row in rows], dtype="<i8").tobytes()
        ).hexdigest(),
    }


@contextlib.contextmanager
def _router_candidate_policy(candidate: bool, *, enabled: bool) -> Iterator[None]:
    if not enabled:
        yield
        return
    previous = {
        ROUTER_COOP_ENV: os.environ.get(ROUTER_COOP_ENV),
        ROUTER_PERSISTENT_ENV: os.environ.get(ROUTER_PERSISTENT_ENV),
    }
    value = "1" if candidate else "0"
    try:
        os.environ[ROUTER_COOP_ENV] = value
        os.environ[ROUTER_PERSISTENT_ENV] = value
        yield
    finally:
        for key, prior in previous.items():
            if prior is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = prior


@contextlib.contextmanager
def _rowtile_policy(enabled: bool) -> Iterator[None]:
    previous = os.environ.get(POLICY_ENV)
    if enabled and _router_candidate_enabled:
        # The production bundle uses the backend's physical-width floor.  Do
        # not force the rejected c2 all-rowtile diagnostic.
        os.environ.pop(POLICY_ENV, None)
    else:
        os.environ[POLICY_ENV] = "1" if enabled else "0"
    try:
        with _router_candidate_policy(
            enabled,
            enabled=_router_candidate_enabled,
        ):
            yield
    finally:
        if previous is None:
            os.environ.pop(POLICY_ENV, None)
        else:
            os.environ[POLICY_ENV] = previous


@contextlib.contextmanager
def _current_package_policy() -> Iterator[None]:
    """Use package defaults while restoring any caller diagnostic overrides."""

    keys = (POLICY_ENV, ROUTER_COOP_ENV, ROUTER_PERSISTENT_ENV)
    previous = {key: os.environ.get(key) for key in keys}
    try:
        for key in keys:
            os.environ.pop(key, None)
        yield
    finally:
        for key, prior in previous.items():
            if prior is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = prior


@contextlib.contextmanager
def _pair_package_policy() -> Iterator[None]:
    """Evaluate the package pair-rowtile floor with diagnostic envs unset.

    The candidate arm must reach the route exactly the way production does:
    through the backend package ``GGUF_Q8_T16_DECODE_PAIR_ROWTILE_MIN_ROWS``
    floor, never through the broad all-projection boolean or the pair env
    overrides. The env keys are *unset* rather than set to ``0`` because an
    explicit ``0`` overrides the package floor instead of deferring to it.
    """

    keys = (POLICY_ENV, PAIR_POLICY_ENV, PAIR_COL8_ENV)
    previous = {key: os.environ.get(key) for key in keys}
    try:
        for key in keys:
            os.environ.pop(key, None)
        yield
    finally:
        for key, prior in previous.items():
            if prior is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = prior


@contextlib.contextmanager
def _q4_selected_pair_package_policy() -> Iterator[None]:
    """Evaluate the package selected-pairreuse floor with diagnostic envs unset.

    Same rule as ``_pair_package_policy``: the candidate arm must reach the
    Q4 selected dual pairreuse route through the backend package
    ``GGUF_Q4_T16_SELECTED_PAIRREUSE_MIN_ROWS`` floor, never through the env
    overrides. The selected-pairreuse env keys are *unset* rather than set
    to ``0`` because an explicit ``0`` overrides the package floor.
    """

    keys = (
        POLICY_ENV,
        PAIR_POLICY_ENV,
        PAIR_COL8_ENV,
        SELECTED_PAIR_ENV,
        SELECTED_DOWN_PAIR_ENV,
        SELECTED_Q6_DOWN_PAIR_ENV,
    )
    previous = {key: os.environ.get(key) for key in keys}
    try:
        for key in keys:
            os.environ.pop(key, None)
        yield
    finally:
        for key, prior in previous.items():
            if prior is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = prior


@contextlib.contextmanager
def _candidate_route_policy(route_profile: str) -> Iterator[None]:
    if str(route_profile) == ROUTE_PROFILE_CURRENT_DIRECT:
        with _current_package_policy():
            yield
        return
    if str(route_profile) == ROUTE_PROFILE_Q8T16_PAIR:
        with _pair_package_policy():
            yield
        return
    if str(route_profile) == ROUTE_PROFILE_Q4_SELECTED_PAIR:
        with _q4_selected_pair_package_policy():
            yield
        return
    if str(route_profile) != ROUTE_PROFILE_Q8T16:
        raise ValueError(f"unsupported batch-route profile {route_profile!r}")
    with _rowtile_policy(True):
        yield


@contextlib.contextmanager
def _candidate_bundle_policy(
    candidate: bool,
    *,
    include_router_candidate: bool,
) -> Iterator[None]:
    global _router_candidate_enabled
    previous = _router_candidate_enabled
    _router_candidate_enabled = bool(include_router_candidate)
    try:
        with _rowtile_policy(candidate):
            yield
    finally:
        _router_candidate_enabled = previous


def candidate_variant_manifest(
    *,
    include_router_candidate: bool,
) -> dict[str, dict[str, str]]:
    direct = {
        "single": "t16_gemv_decode_bf16_bf16_out",
        "pair": "t16_dual_gemv_decode_bf16_bf16_out",
        "triple": "t16_triple_gemv_decode_bf16_bf16_out",
    }
    rowtile = {
        "single": "t16_gemv_decode_rowtile4_bf16_bf16_out",
        "pair": "t16_dual_gemv_decode_rowtile4_bf16_bf16_out",
        "triple": "t16_triple_gemv_decode_rowtile4_bf16_bf16_out",
    }
    c8 = {
        **rowtile,
        "pair": "t16_dual_gemv_decode_rowtile4_col8_bf16_bf16_out",
    }
    if include_router_candidate:
        return {"c1": direct, "c2": direct, "c4": rowtile, "c8": c8}
    return {"c1": direct, "c2_c4": rowtile, "c8": c8}


def _prefill_group(
    sessions: Sequence[Any],
    prompt_rows: Sequence[Mapping[str, Any]],
    prompt_tokens: Mapping[str, Sequence[int]],
    strict: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    gdn_mode: str,
    packed_candidate: bool = False,
) -> None:
    """Admit the group's prompts and calibrate against the strict trajectory.

    Serial admission requires byte-exact prefill logits. ``packed_candidate``
    admits the whole group through ``prefill_batch_native`` on ``sessions[0]`` and
    can therefore only require *token* equality: the packed route persists
    ``segmented_in_place_final_state`` while the strict teacher persists
    per-token-exact state, so their prefill outputs differ by reassociation near
    one BF16 ulp (see the 2026-08-30 packed-prefill state gate entry). Numeric
    judgment stays with the unchanged decode-step profile comparison. The packed
    candidate deliberately keeps the package-default GDN route -- the arithmetic
    production serving uses -- instead of the strict mode the teacher has.
    """

    if packed_candidate:
        for session in sessions:
            session.reset()
        prompts = [
            [int(token) for token in prompt_tokens[str(row["id"])]] for row in prompt_rows
        ]
        results = sessions[0].prefill_batch_native(prompts, sessions=tuple(sessions))
        for row, result in zip(prompt_rows, results, strict=True):
            prompt_id = str(row["id"])
            if result is None:
                raise CalibrationError(f"packed prefill returned no probe for {prompt_id}")
            expected = strict[prompt_id][0]
            if int(result.token_id) != int(expected["token_id"]):
                raise CalibrationError(
                    f"packed prefill token drift for {prompt_id}: "
                    f"candidate={result.token_id}, strict={expected['token_id']}"
                )
        return
    for session, row in zip(sessions, prompt_rows, strict=True):
        session.reset()
        prompt_id = str(row["id"])
        with _gdn_mode(gdn_mode):
            result = session.prefill(
                [int(token) for token in prompt_tokens[prompt_id]],
                use_bulk=True,
                bulk_attention_mode="bulk",
                return_logits=True,
                capture_hidden_seed_fp32=False,
            )
        expected = strict[prompt_id][0]
        if int(result.token_id) != int(expected["token_id"]):
            raise CalibrationError(
                f"serial prefill token drift for {prompt_id}: "
                f"candidate={result.token_id}, strict={expected['token_id']}"
            )
        expected_logits = np.asarray(expected["logits"], dtype=np.float32)
        actual_logits = np.asarray(result.logits, dtype=np.float32)
        if not np.array_equal(actual_logits, expected_logits):
            raise CalibrationError(f"serial prefill logits drift for {prompt_id}")


def _run_static_group(
    sessions: Sequence[Any],
    prompt_rows: Sequence[Mapping[str, Any]],
    prompt_tokens: Mapping[str, Sequence[int]],
    strict: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    decode_steps: int,
    repeat_runs: int,
    gdn_mode: str,
    route_profile: str,
    packed_candidate: bool = False,
) -> list[list[tuple[dict[str, object], ...]]]:
    all_runs: list[list[tuple[dict[str, object], ...]]] = []
    for _ in range(int(repeat_runs)):
        _prefill_group(
            sessions,
            prompt_rows,
            prompt_tokens,
            strict,
            gdn_mode=gdn_mode,
            packed_candidate=packed_candidate,
        )
        trajectories: list[list[dict[str, object]]] = [[] for _ in prompt_rows]
        with _candidate_route_policy(route_profile):
            for step in range(int(decode_steps)):
                token_ids = [
                    int(strict[str(row["id"])][step]["token_id"])
                    for row in prompt_rows
                ]
                results = sessions[0].step_batch_native(
                    token_ids,
                    sessions=tuple(sessions),
                    return_logits=True,
                    scatter_state=True,
                )
                for index, result in enumerate(results):
                    trajectories[index].append(_trajectory_from_results((result,))[0])
        all_runs.append([tuple(trajectory) for trajectory in trajectories])
    return all_runs


def _schedule_width(
    schedule: Sequence[tuple[int, int]], step: int
) -> tuple[int, bool, int | None]:
    current = int(schedule[0][1])
    previous: int | None = None
    entered = step == 0
    for start, width in schedule[1:]:
        if step < start:
            break
        previous = current
        current = int(width)
        entered = step == start
    return current, entered, previous


def _run_dynamic(
    sessions: Sequence[Any],
    prompt_rows: Sequence[Mapping[str, Any]],
    prompt_tokens: Mapping[str, Sequence[int]],
    strict: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    schedule: Sequence[tuple[int, int]],
    decode_steps: int,
    repeat_runs: int,
    gdn_mode: str,
    route_profile: str,
    packed_candidate: bool = False,
) -> tuple[
    list[dict[int, tuple[dict[str, object], ...]]],
    dict[int, tuple[str, ...]],
    dict[int, tuple[str, ...]],
]:
    survivor_order = (0, 2, 4, 6, 1, 3, 5, 7)
    run_trajectories: list[dict[int, tuple[dict[str, object], ...]]] = []
    shapes: dict[int, list[str]] = {index: [] for index in range(8)}
    transitions: dict[int, list[str]] = {index: [] for index in range(8)}
    for repeat_index in range(int(repeat_runs)):
        _prefill_group(
            sessions,
            prompt_rows,
            prompt_tokens,
            strict,
            gdn_mode=gdn_mode,
            packed_candidate=packed_candidate,
        )
        trajectories: dict[int, list[dict[str, object]]] = {index: [] for index in range(8)}
        with _candidate_route_policy(route_profile):
            for step in range(int(decode_steps)):
                width, entered, previous = _schedule_width(schedule, step)
                active_indices = survivor_order[:width]
                active_sessions = tuple(sessions[index] for index in active_indices)
                token_ids = [
                    int(strict[str(prompt_rows[index]["id"])][step]["token_id"])
                    for index in active_indices
                ]
                results = active_sessions[0].step_batch_native(
                    token_ids,
                    sessions=active_sessions,
                    return_logits=True,
                    scatter_state=True,
                )
                transition = (
                    f"enter_c{width}"
                    if entered and previous is None
                    else (
                        f"width_{previous}_to_{width}"
                        if entered
                        else "steady"
                    )
                )
                for original_index, result in zip(active_indices, results, strict=True):
                    trajectories[original_index].append(
                        _trajectory_from_results((result,))[0]
                    )
                    if repeat_index == 0:
                        shapes[original_index].append(f"c{width}")
                        transitions[original_index].append(transition)
        run_trajectories.append(
            {index: tuple(value) for index, value in trajectories.items()}
        )
    return (
        run_trajectories,
        {index: tuple(value) for index, value in shapes.items()},
        {index: tuple(value) for index, value in transitions.items()},
    )


def _run_sparse_c8(
    sessions: Sequence[Any],
    prompt_rows: Sequence[Mapping[str, Any]],
    prompt_tokens: Mapping[str, Sequence[int]],
    strict: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    decode_steps: int,
    repeat_runs: int,
    gdn_mode: str,
    route_profile: str,
    packed_candidate: bool = False,
) -> list[list[tuple[dict[str, object], ...]]]:
    active_slots = (0, 2, 5, 7)
    all_runs: list[list[tuple[dict[str, object], ...]]] = []
    for _ in range(int(repeat_runs)):
        _prefill_group(
            sessions,
            prompt_rows,
            prompt_tokens,
            strict,
            gdn_mode=gdn_mode,
            packed_candidate=packed_candidate,
        )
        trajectories: list[list[dict[str, object]]] = [[] for _ in prompt_rows]
        with _candidate_route_policy(route_profile):
            for step in range(int(decode_steps)):
                token_ids = [
                    int(strict[str(row["id"])][step]["token_id"])
                    for row in prompt_rows
                ]
                results = sessions[0].step_batch_native(
                    token_ids,
                    sessions=tuple(sessions),
                    return_logits=True,
                    scatter_state=True,
                    physical_rows=8,
                    active_slot_indices=active_slots,
                )
                for index, result in enumerate(results):
                    trajectories[index].append(_trajectory_from_results((result,))[0])
        all_runs.append([tuple(trajectory) for trajectory in trajectories])
    return all_runs


def run(args: argparse.Namespace, *, command: Sequence[str]) -> dict[str, Any]:
    if not args.model.is_file():
        raise CalibrationError(f"model does not exist: {args.model}")
    if not args.historical_source.is_file():
        raise CalibrationError(f"historical source does not exist: {args.historical_source}")
    if int(args.decode_steps) <= 0 or int(args.repeat_runs) < 3:
        raise CalibrationError("decode steps must be positive and repeat runs at least three")
    route_profile = str(args.route_profile)
    widths = tuple(int(value) for value in args.widths.split(",") if value.strip())
    if route_profile == ROUTE_PROFILE_Q8T16:
        if not widths or any(width not in {4, 8} for width in widths):
            raise CalibrationError("Q8T16 candidate widths must be a non-empty subset of 4,8")
        dynamic_schedule = DEFAULT_DYNAMIC_SCHEDULE
    elif route_profile == ROUTE_PROFILE_Q8T16_PAIR:
        if not widths or any(width not in {4, 8} for width in widths):
            raise CalibrationError("Q8T16 pair candidate widths must be a non-empty subset of 4,8")
        dynamic_schedule = DEFAULT_DYNAMIC_SCHEDULE
    elif route_profile == ROUTE_PROFILE_Q4_SELECTED_PAIR:
        if not widths or any(width not in {4, 8} for width in widths):
            raise CalibrationError("Q4 selected-pair candidate widths must be a non-empty subset of 4,8")
        dynamic_schedule = DEFAULT_DYNAMIC_SCHEDULE
    elif route_profile == ROUTE_PROFILE_CURRENT_DIRECT:
        if not widths or any(width not in DIRECT_STATIC_WIDTHS for width in widths):
            raise CalibrationError(
                "current-package direct widths must be a non-empty subset of 3,5,6,7"
            )
        if bool(args.include_router_candidate):
            raise CalibrationError(
                "current-package direct profile cannot include the router candidate"
            )
        dynamic_schedule = DIRECT_DYNAMIC_SCHEDULE
    else:
        raise CalibrationError(f"unsupported route profile {route_profile!r}")
    schedule = (
        validate_width_schedule(
            dynamic_schedule,
            decode_steps=int(args.decode_steps),
        )
        if args.dynamic
        else ()
    )
    prompt_rows = _load_suites(args.prompts)
    if args.limit is not None:
        prompt_rows = prompt_rows[: max(0, int(args.limit))]
    if not prompt_rows:
        raise CalibrationError("selected prompt suites are empty")
    selected_suites = tuple(path.resolve() for path in args.prompts)
    default_suites = tuple(path.resolve() for path in DEFAULT_PROMPTS)
    complete_suite = args.limit is None and selected_suites == default_suites
    _configure_gate_environment(decode_repack=True)
    os.environ["HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN"] = "1"

    from hipengine.loading.gguf import scan_gguf
    from hipengine.runtime.prefill import PrefillConfig
    from hipengine.runtime.qwen35_gguf_runner import (
        Qwen35GGUFResidentSession,
        _gguf_gdn_prefill_backend_exact_mode,
    )
    from hipengine.tokenization.gguf import Qwen35GGUFTokenizer

    compiler_version = None
    if args.compiler_version_file is not None:
        compiler_version = args.compiler_version_file.read_text(encoding="utf-8")
    tokenizer = Qwen35GGUFTokenizer.from_gguf_info(scan_gguf(args.model))
    prompt_tokens = {
        str(row["id"]): build_chat_prompt(tokenizer, str(row["prompt"]))
        for row in prompt_rows
    }
    max_sequence_length = max(len(value) for value in prompt_tokens.values()) + int(args.decode_steps) + 2
    package = __import__(
        f"hipengine.kernels.{args.backend}",
        fromlist=[POLICY_CAPABILITY, POLICY_MIN_ROWS_CAPABILITY],
    )
    package_value = getattr(package, POLICY_CAPABILITY)
    package_min_rows = int(getattr(package, POLICY_MIN_ROWS_CAPABILITY, 0))
    if route_profile == ROUTE_PROFILE_Q8T16_PAIR:
        pair_min_rows = int(getattr(package, PAIR_MIN_ROWS_CAPABILITY, 0))
        if pair_min_rows != 8:
            raise CalibrationError(
                f"pair candidate requires package {PAIR_MIN_ROWS_CAPABILITY} == 8, got {pair_min_rows}"
            )
    if route_profile == ROUTE_PROFILE_Q4_SELECTED_PAIR:
        selected_pair_min_rows = int(
            getattr(package, SELECTED_PAIR_MIN_ROWS_CAPABILITY, 0)
        )
        if selected_pair_min_rows != 8:
            raise CalibrationError(
                "selected-pair candidate requires package "
                f"{SELECTED_PAIR_MIN_ROWS_CAPABILITY} == 8, got {selected_pair_min_rows}"
            )
    if route_profile == ROUTE_PROFILE_Q8T16 and package_value is not False:
        raise CalibrationError(
            f"current package {POLICY_CAPABILITY} must be False, got {package_value!r}"
        )
    if args.include_router_candidate and package_min_rows != 4:
        raise CalibrationError(
            f"bundled candidate requires package rowtile min rows 4, got {package_min_rows}"
        )

    global _router_candidate_enabled
    previous_router_candidate = _router_candidate_enabled
    _router_candidate_enabled = bool(
        route_profile == ROUTE_PROFILE_Q8T16 and args.include_router_candidate
    )
    stack = ExitStack()
    captures: list[BatchRouteCapture] = []
    strict: dict[str, tuple[Mapping[str, object], ...]] = {}
    resolved_backend = str(args.backend)
    target_arch = str(args.backend).removeprefix("hip_")
    manifests: list[dict[str, Any]] = []
    try:
        owner = stack.enter_context(
            Qwen35GGUFResidentSession(
                args.model,
                backend=str(args.backend),
                compiler_version=compiler_version,
                require_cached_build=bool(args.require_cached_build),
                max_sequence_length=max_sequence_length,
                use_wmma_prefill=True,
                use_gemv_decode=True,
                prefill_config=PrefillConfig(attn_aotriton_min_tokens=512),
            )
        )
        if owner.runner is None:
            raise CalibrationError("GGUF resident session closed during setup")
        resolved_backend = str(owner.runner.backend)
        target_arch = str(owner.runner.target_arch)
        validate_strict_baseline(
            requested_mode=str(args.gdn_mode),
            backend_exact_mode=_gguf_gdn_prefill_backend_exact_mode(resolved_backend),
        )
        with _rowtile_policy(False):
            for index, row in enumerate(prompt_rows):
                prompt_id = str(row["id"])
                strict[prompt_id] = tuple(
                    _run_logits_trajectory(
                        owner,
                        prompt_ids=prompt_tokens[prompt_id],
                        mode=str(args.gdn_mode),
                        decode_steps=int(args.decode_steps),
                        bulk_attention_mode="bulk",
                    )
                )
                print(f"strict {index + 1}/{len(prompt_rows)} {prompt_id}", flush=True)
        sessions = [owner]
        while len(sessions) < 8:
            sessions.append(
                stack.enter_context(
                    Qwen35GGUFResidentSession(
                        args.model,
                        backend=str(args.backend),
                        runtime=owner.runtime,
                        shared_runner=owner.runner,
                        compiler_version=compiler_version,
                        require_cached_build=bool(args.require_cached_build),
                        max_sequence_length=max_sequence_length,
                        use_wmma_prefill=True,
                        use_gemv_decode=True,
                        prefill_config=PrefillConfig(attn_aotriton_min_tokens=512),
                    )
                )
            )
        for width in widths:
            for group_index, start in enumerate(range(0, len(prompt_rows), width)):
                actual = list(prompt_rows[start : start + width])
                padded = list(actual)
                while len(padded) < width:
                    padded.append(prompt_rows[len(padded) % len(prompt_rows)])
                runs = _run_static_group(
                    sessions[:width],
                    padded,
                    prompt_tokens,
                    strict,
                    decode_steps=int(args.decode_steps),
                    repeat_runs=int(args.repeat_runs),
                    gdn_mode=str(args.gdn_mode),
                    route_profile=route_profile,
                    packed_candidate=bool(args.packed_prefill_candidate),
                )
                for row_index, row in enumerate(actual):
                    prompt_id = str(row["id"])
                    captures.append(
                        BatchRouteCapture(
                            scenario_id=f"static_c{width}_group{group_index}",
                            request_id=prompt_id,
                            category=str(row["category"]),
                            strict=tuple(strict[prompt_id][1 : int(args.decode_steps) + 1]),
                            candidate_runs=tuple(run[row_index] for run in runs),
                            shapes=(f"c{width}",) * int(args.decode_steps),
                            transitions=("steady",) * int(args.decode_steps),
                            teacher_steps=tuple(range(1, int(args.decode_steps) + 1)),
                        )
                    )
                manifests.append(dict(owner.last_packed_execution_manifest))
                print(f"static c{width} group {group_index + 1}: captured x {args.repeat_runs}", flush=True)
        if args.dynamic and len(prompt_rows) >= 8:
            dynamic_rows = list(prompt_rows[:8])
            runs, shapes, transitions = _run_dynamic(
                sessions,
                dynamic_rows,
                prompt_tokens,
                strict,
                schedule=schedule,
                decode_steps=int(args.decode_steps),
                repeat_runs=int(args.repeat_runs),
                gdn_mode=str(args.gdn_mode),
                route_profile=route_profile,
                packed_candidate=bool(args.packed_prefill_candidate),
            )
            for index, row in enumerate(dynamic_rows):
                length = len(runs[0][index])
                if length == 0:
                    continue
                prompt_id = str(row["id"])
                captures.append(
                    BatchRouteCapture(
                        scenario_id="dynamic_width_retirement",
                        request_id=prompt_id,
                        category=str(row["category"]),
                        strict=tuple(strict[prompt_id][1 : length + 1]),
                        candidate_runs=tuple(run[index] for run in runs),
                        shapes=shapes[index],
                        transitions=transitions[index],
                        teacher_steps=tuple(range(1, length + 1)),
                    )
                )
            manifests.append(dict(owner.last_packed_execution_manifest))
            print(f"dynamic widths: captured x {args.repeat_runs}", flush=True)
        if args.sparse and len(prompt_rows) >= 4:
            sparse_rows = [prompt_rows[index] for index in (0, len(prompt_rows) // 3, 2 * len(prompt_rows) // 3, len(prompt_rows) - 1)]
            runs = _run_sparse_c8(
                sessions[:4],
                sparse_rows,
                prompt_tokens,
                strict,
                decode_steps=int(args.decode_steps),
                repeat_runs=int(args.repeat_runs),
                gdn_mode=str(args.gdn_mode),
                route_profile=route_profile,
                packed_candidate=bool(args.packed_prefill_candidate),
            )
            for index, row in enumerate(sparse_rows):
                prompt_id = str(row["id"])
                captures.append(
                    BatchRouteCapture(
                        scenario_id="sparse_physical_c8",
                        request_id=prompt_id,
                        category=str(row["category"]),
                        strict=tuple(strict[prompt_id][1 : int(args.decode_steps) + 1]),
                        candidate_runs=tuple(run[index] for run in runs),
                        shapes=("c8_sparse4",) * int(args.decode_steps),
                        transitions=("sparse_steady",) * int(args.decode_steps),
                        teacher_steps=tuple(range(1, int(args.decode_steps) + 1)),
                    )
                )
            manifests.append(dict(owner.last_packed_execution_manifest))
            print(f"sparse c8: captured x {args.repeat_runs}", flush=True)
    finally:
        stack.close()
        _router_candidate_enabled = previous_router_candidate

    evaluated = build_batch_route_quality(captures)
    provenance = collect_artifact_provenance(
        repo_root=REPO_ROOT,
        configured_backend=str(args.backend),
        resolved_backend=resolved_backend,
        target_arch=target_arch,
        model_path=args.model,
        quant="gguf_q4_k_m",
        kv_dtype="bf16",
        command=command,
        environment={
            "HIPENGINE_HIP_ARCH": os.environ.get("HIPENGINE_HIP_ARCH"),
            "HIP_VISIBLE_DEVICES": os.environ.get("HIP_VISIBLE_DEVICES"),
            POLICY_ENV: os.environ.get(POLICY_ENV),
            ROUTER_COOP_ENV: os.environ.get(ROUTER_COOP_ENV),
            ROUTER_PERSISTENT_ENV: os.environ.get(ROUTER_PERSISTENT_ENV),
        },
        build_profile=(
            "execution_profile_gguf_direct_width_batch_requalification"
            if route_profile == ROUTE_PROFILE_CURRENT_DIRECT
            else "execution_profile_gguf_q8t16_batch_requalification"
        ),
        timing_protocol="none_full_logits_only_v1",
        warmups=0,
        repetitions=int(args.repeat_runs),
        profiler={"enabled": False, "kind": None, "command": None},
    )
    required_widths = (
        DIRECT_STATIC_WIDTHS
        if route_profile == ROUTE_PROFILE_CURRENT_DIRECT
        else frozenset({4, 8})
    )
    complete_matrix = bool(
        complete_suite
        and set(widths) == required_widths
        and args.dynamic
        and args.sparse
        and len(prompt_rows) == 18
        and int(args.decode_steps) == 24
        and int(args.repeat_runs) >= 3
    )
    deterministic = bool(evaluated["repeat_determinism"]["passed"])
    measurement_valid = bool(complete_matrix and deterministic and not provenance.get("dirty"))
    historical = json.loads(args.historical_source.read_text(encoding="utf-8"))
    current_direct = route_profile == ROUTE_PROFILE_CURRENT_DIRECT
    candidate_environment = (
        {
            POLICY_ENV: "unset: current package default",
            ROUTER_COOP_ENV: "unset: current package default",
            ROUTER_PERSISTENT_ENV: "unset: current package default",
            SELECTED_PAIR_ENV: "unset: package selected-pairreuse floor",
            SELECTED_DOWN_PAIR_ENV: "unset: package selected-pairreuse floor",
            SELECTED_Q6_DOWN_PAIR_ENV: "unset: package selected-pairreuse floor",
        }
        if route_profile == ROUTE_PROFILE_Q4_SELECTED_PAIR
        else {
            POLICY_ENV: "unset: current package default",
            ROUTER_COOP_ENV: "unset: current package default",
            ROUTER_PERSISTENT_ENV: "unset: current package default",
        }
        if route_profile == ROUTE_PROFILE_Q8T16_PAIR
        else {
            POLICY_ENV: "unset: current package default",
            ROUTER_COOP_ENV: "unset: current package default",
            ROUTER_PERSISTENT_ENV: "unset: current package default",
        }
        if current_direct
        else {
            POLICY_ENV: (
                "unset: package physical-width floor"
                if args.include_router_candidate
                else "1"
            ),
            **(
                {
                    ROUTER_COOP_ENV: "1",
                    ROUTER_PERSISTENT_ENV: "1",
                }
                if args.include_router_candidate
                else {}
            ),
        }
    )
    candidate_variants: Mapping[str, Any] = (
        {
            "source": "package_selected_pairreuse_floor",
            "c1": {"pair": "t16_selected_dual_gemv_bf16_bf16_out", "single": "t16_selected_gemv_bf16_bf16_out"},
            "c2_c4": {"pair": "t16_selected_dual_gemv_bf16_bf16_out", "single": "t16_selected_gemv_bf16_bf16_out"},
            "c8": {
                "pair": "t16_selected_dual_pairreuse_gemv_bf16_bf16_out",
                "single": "t16_selected_gemv_bf16_bf16_out",
            },
        }
        if route_profile == ROUTE_PROFILE_Q4_SELECTED_PAIR
        else {
            "source": "package_pair_floor",
            "c1": {"pair": "t16_dual_gemv_decode_bf16_bf16_out", "single": "t16_gemv_decode_bf16_bf16_out", "triple": "t16_triple_gemv_decode_bf16_bf16_out"},
            "c2_c4": {"pair": "t16_dual_gemv_decode_bf16_bf16_out", "single": "t16_gemv_decode_bf16_bf16_out", "triple": "t16_triple_gemv_decode_bf16_bf16_out"},
            "c8": {
                "pair": "t16_dual_gemv_decode_rowtile4_col8_bf16_bf16_out",
                "single": "t16_gemv_decode_bf16_bf16_out",
                "triple": "t16_triple_gemv_decode_bf16_bf16_out",
            },
        }
        if route_profile == ROUTE_PROFILE_Q8T16_PAIR
        else {"source": "current_package_execution_manifests"}
        if current_direct
        else candidate_variant_manifest(
            include_router_candidate=bool(args.include_router_candidate)
        )
    )
    policy_restored = (
        all(
            key not in os.environ
            for key in (POLICY_ENV, ROUTER_COOP_ENV, ROUTER_PERSISTENT_ENV)
        )
        if current_direct
        else (
            POLICY_ENV not in os.environ
            and (
                not args.include_router_candidate
                or (
                    ROUTER_COOP_ENV not in os.environ
                    and ROUTER_PERSISTENT_ENV not in os.environ
                )
            )
        )
    )
    return {
        "schema_version": 1,
        "kind": KIND,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete" if measurement_valid else "invalid_or_screen_only",
        "measurement_valid": measurement_valid,
        "performance_claim": False,
        "profile_qualification_claim": False,
        "qualification_blockers": [
            "route manifest is evidentiary and not runtime-profile resolved",
            "adapter does not emit the complete exact control-plane record schema",
            "fresh task and BF16-relative verdicts are not available",
        ],
        "route": {
            "route_profile": route_profile,
            "policy_capability": POLICY_CAPABILITY,
            "package_value_verified": package_value,
            "package_min_rows_capability": POLICY_MIN_ROWS_CAPABILITY,
            "package_min_rows_verified": package_min_rows,
            "candidate_environment": candidate_environment,
            "strict_environment": {
                POLICY_ENV: "0",
                **(
                    {
                        ROUTER_COOP_ENV: "0",
                        ROUTER_PERSISTENT_ENV: "0",
                    }
                    if args.include_router_candidate
                    else {}
                ),
            },
            "router_candidate_included": bool(args.include_router_candidate),
            "policy_restored_after_capture": policy_restored,
            "strict_variants": {
                "c1_c2_c4": {
                    "single": "t16_gemv_decode_bf16_bf16_out",
                    "pair": "t16_dual_gemv_decode_bf16_bf16_out",
                    "triple": "t16_triple_gemv_decode_bf16_bf16_out",
                },
                "c8": {
                    "single": "t16_gemv_decode_bf16_bf16_out",
                    "pair": "t16_dual_gemv_decode_rowtile4_col8_bf16_bf16_out",
                    "triple": "t16_triple_gemv_decode_bf16_bf16_out",
                },
            },
            "candidate_variants": candidate_variants,
        },
        "protocol": {
            "route_profile": route_profile,
            "prompt_suites": [str(path.resolve()) for path in args.prompts],
            "complete_prompt_and_heldout_suite": complete_suite,
            "prompt_count": len(prompt_rows),
            "decode_steps": int(args.decode_steps),
            "packed_prefill_candidate": bool(args.packed_prefill_candidate),
            "candidate_repeat_runs": int(args.repeat_runs),
            "static_widths": list(widths),
            "dynamic_schedule": [list(value) for value in schedule] if args.dynamic else [],
            "sparse_physical_c8_active_slots": [0, 2, 5, 7] if args.sparse else [],
            "teacher_forced_rows": int(evaluated["quality"]["summary"]["rows"]),
            "same_context_rule": "every batch row consumes its independent strict-c1 teacher token",
            "strict_gdn_mode": str(args.gdn_mode),
            "thresholds_evaluated": EvaluationThresholds().to_dict(),
        },
        **evaluated,
        "execution_manifests": manifests,
        "prompts": [
            {
                "id": str(row["id"]),
                "category": str(row["category"]),
                "suite": str(row["suite"]),
                "prompt_sha256": prompt_sha256(str(row["prompt"])),
                "prompt_tokens": len(prompt_tokens[str(row["id"])]),
            }
            for row in prompt_rows
        ],
        "historical_source": {
            "path": str(args.historical_source),
            "sha256": _file_sha256(args.historical_source),
            "kind": historical.get("kind"),
            "status": historical.get("status"),
            "label_does_not_affect_fresh_verdict": True,
        },
        "provenance": provenance,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--backend", default="hip_gfx1151")
    parser.add_argument("--prompts", action="append", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--route-profile",
        choices=(
            ROUTE_PROFILE_Q8T16,
            ROUTE_PROFILE_Q8T16_PAIR,
            ROUTE_PROFILE_Q4_SELECTED_PAIR,
            ROUTE_PROFILE_CURRENT_DIRECT,
        ),
        default=ROUTE_PROFILE_Q8T16,
    )
    parser.add_argument("--widths", default="4,8")
    parser.add_argument(
        "--packed-prefill-candidate",
        action="store_true",
        help=(
            "admit candidate prompts through packed multi-request prefill instead of one"
            " session at a time; relaxes the prefill calibration from byte-exact logits to"
            " token equality (the routes persist different final-state arithmetic) and"
            " leaves the decode-step profile comparison unchanged"
        ),
    )
    parser.add_argument("--decode-steps", type=int, default=24)
    parser.add_argument("--repeat-runs", type=int, default=3)
    parser.add_argument("--dynamic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sparse", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gdn-mode", default=DEFAULT_GDN_MODE)
    parser.add_argument(
        "--include-router-candidate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Bundle the cooperative/persistent c1 router with the Q8T16 rowtile candidate.",
    )
    parser.add_argument("--historical-source", type=Path, default=DEFAULT_HISTORICAL_SOURCE)
    parser.add_argument("--compiler-version-file", type=Path, default=None)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--json", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(raw_argv)
    if args.prompts is None:
        args.prompts = list(DEFAULT_PROMPTS)
    command = [sys.executable, str(Path(__file__).relative_to(REPO_ROOT)), *raw_argv]
    try:
        artifact = run(args, command=command)
    except (CalibrationError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(args.json)
    print(
        json.dumps(
            {
                "status": artifact["status"],
                "measurement_valid": artifact["measurement_valid"],
                "hard_gate_passed": artifact["quality"]["hard_gates_passed"],
                "requires_outlier_review": artifact["quality"]["requires_outlier_review"],
                "summary": artifact["quality"]["summary"],
                "repeat_deterministic": artifact["repeat_determinism"]["passed"],
            },
            indent=2,
        )
    )
    return 0 if artifact["measurement_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
