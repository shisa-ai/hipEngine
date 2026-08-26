#!/usr/bin/env python3
"""Audit FP16 recurrent state on the actual packed c4/c8 GGUF route.

The earlier FP16-state execution-profile adapter exercises serial c1 prefill and
c1 decode only.  This companion capture covers the later packed-path kernels:
segmented in-place FP16 prefill and indexed-singleton FP16 decode at c4/c8.  It
compares the candidate with FP32 state on the same production route and same
physical schedule, using full-vocabulary teacher-forced rows.

The packet includes static c4/c8, category-diverse 512-token c8 prompts, a
c8->c4->c2->c1 retirement schedule, sparse physical-c8 slots, neighbor
substitution, and row permutation.  It is a numerical/determinism/isolation
capture, not a public-profile promotion: task/BF16-relative/serving-manifest
requirements remain separate.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.benchmark.execution_profiles import EvaluationThresholds
from hipengine.benchmark.provenance import collect_artifact_provenance
from scripts.execution_profile_gguf_batch_route_gate import (
    BatchRouteCapture,
    build_batch_route_quality,
    validate_width_schedule,
)
from scripts.execution_profile_gguf_fp16_state_gate import (
    CANDIDATE_NAME,
    FP16_STATE_ENV,
    PRODUCTION_ROUTE,
    fp16_state_environment,
    production_route_environment,
)
from scripts.gguf_gdn_semantic_gate import DEFAULT_PROMPTS, _load_suites
from scripts.gguf_mtp_bench import build_chat_prompt
from scripts.gguf_mtp_category_bench import prompt_sha256

KIND = "hipengine_execution_profile_gguf_fp16_state_batch_gate"
SCHEMA_VERSION = 1
DEFAULT_MODEL = Path("/models/gguf/Qwen3.8-27B-Q4_K_S.gguf")
SUPPORTED_WIDTHS = frozenset({4, 8})
DEFAULT_DYNAMIC_SCHEDULE = ((0, 8), (6, 4), (12, 2), (18, 1))
SPARSE_ACTIVE_SLOTS = (0, 2, 5, 7)


class GateError(RuntimeError):
    """Raised when a packed FP16-state capture cannot be evaluated safely."""


@dataclass(frozen=True, slots=True)
class StaticScenario:
    scenario_id: str
    width: int
    rows: tuple[Mapping[str, Any], ...]
    token_rows: tuple[tuple[int, ...], ...]
    actual_count: int
    decode_steps: int
    long_context: bool = False

    def __post_init__(self) -> None:
        if self.width not in SUPPORTED_WIDTHS:
            raise ValueError("static scenario width must be c4 or c8")
        if len(self.rows) != self.width or len(self.token_rows) != self.width:
            raise ValueError("static scenario rows must fill its physical width")
        if not 0 < self.actual_count <= self.width:
            raise ValueError("static scenario actual_count is outside its width")
        if self.decode_steps <= 0:
            raise ValueError("static scenario decode_steps must be positive")


def _step(result: Any) -> dict[str, object]:
    logits = np.ascontiguousarray(result.logits, dtype=np.float32)
    if logits.ndim == 2 and logits.shape[0] == 1:
        logits = np.ascontiguousarray(logits[0])
    if logits.ndim != 1 or logits.size == 0 or not np.all(np.isfinite(logits)):
        raise GateError("packed route returned invalid full-vocabulary logits")
    return {"token_id": int(result.token_id), "logits": logits}


def _trajectory_sha256(trajectory: Sequence[Mapping[str, object]]) -> str:
    digest = hashlib.sha256()
    for row in trajectory:
        digest.update(int(row["token_id"]).to_bytes(8, "little", signed=True))
        digest.update(np.ascontiguousarray(row["logits"], dtype="<f4").tobytes())
    return digest.hexdigest()


def _trajectories_exact(
    left: Sequence[Mapping[str, object]],
    right: Sequence[Mapping[str, object]],
) -> bool:
    if len(left) != len(right):
        return False
    return all(
        int(lhs["token_id"]) == int(rhs["token_id"])
        and np.array_equal(
            np.asarray(lhs["logits"], dtype=np.float32),
            np.asarray(rhs["logits"], dtype=np.float32),
        )
        for lhs, rhs in zip(left, right, strict=True)
    )


def _cycle_tokens(tokens: Sequence[int], target: int) -> tuple[int, ...]:
    values = tuple(int(token) for token in tokens)
    if not values:
        raise ValueError("cannot extend an empty prompt")
    if target <= 0:
        raise ValueError("long prompt target must be positive")
    return tuple(itertools.islice(itertools.cycle(values), int(target)))


def _category_diverse_rows(
    prompt_rows: Sequence[Mapping[str, Any]],
    *,
    per_category: int = 2,
) -> tuple[Mapping[str, Any], ...]:
    by_category: dict[str, list[Mapping[str, Any]]] = {}
    for row in prompt_rows:
        by_category.setdefault(str(row["category"]), []).append(row)
    selected: list[Mapping[str, Any]] = []
    for category in ("code", "general_en", "general_ja", "mixed_ja_en"):
        values = by_category.get(category, [])
        if len(values) < int(per_category):
            raise GateError(
                f"long-context matrix needs {per_category} prompts in {category!r}"
            )
        selected.extend(values[: int(per_category)])
    return tuple(selected)


def build_static_scenarios(
    prompt_rows: Sequence[Mapping[str, Any]],
    prompt_tokens: Mapping[str, Sequence[int]],
    *,
    widths: Sequence[int],
    decode_steps: int,
    long_prompt_tokens: int,
    long_decode_steps: int,
) -> tuple[StaticScenario, ...]:
    """Build deterministic full-suite static groups plus one long c8 group."""

    rows = tuple(prompt_rows)
    if not rows:
        raise ValueError("static scenarios need prompt rows")
    scenarios: list[StaticScenario] = []
    for width in widths:
        width = int(width)
        if width not in SUPPORTED_WIDTHS:
            raise ValueError(f"unsupported static width {width}")
        for group_index, start in enumerate(range(0, len(rows), width)):
            actual = list(rows[start : start + width])
            padded = list(actual)
            while len(padded) < width:
                padded.append(rows[len(padded) % len(rows)])
            scenarios.append(
                StaticScenario(
                    scenario_id=f"static_c{width}_group{group_index}",
                    width=width,
                    rows=tuple(padded),
                    token_rows=tuple(
                        tuple(int(token) for token in prompt_tokens[str(row["id"])])
                        for row in padded
                    ),
                    actual_count=len(actual),
                    decode_steps=int(decode_steps),
                )
            )
    if 8 in {int(width) for width in widths} and int(long_prompt_tokens) > 0:
        long_rows = _category_diverse_rows(rows)
        scenarios.append(
            StaticScenario(
                scenario_id=f"long_c8_p{int(long_prompt_tokens)}",
                width=8,
                rows=long_rows,
                token_rows=tuple(
                    _cycle_tokens(
                        prompt_tokens[str(row["id"])],
                        int(long_prompt_tokens),
                    )
                    for row in long_rows
                ),
                actual_count=8,
                decode_steps=int(long_decode_steps),
                long_context=True,
            )
        )
    return tuple(scenarios)


def _make_sessions(
    args: argparse.Namespace,
    *,
    fp16: bool,
    max_sequence_length: int,
) -> tuple[ExitStack, tuple[Any, ...], str, str]:
    from hipengine.runtime.prefill import PrefillConfig
    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession

    compiler_version = (
        None
        if args.compiler_version_file is None
        else args.compiler_version_file.read_text(encoding="utf-8")
    )
    stack = ExitStack()
    try:
        with fp16_state_environment(fp16), production_route_environment():
            owner = stack.enter_context(
                Qwen35GGUFResidentSession(
                    args.model,
                    backend=str(args.backend),
                    compiler_version=compiler_version,
                    require_cached_build=bool(args.require_cached_build),
                    max_sequence_length=int(max_sequence_length),
                    use_wmma_prefill=True,
                    use_gemv_decode=True,
                    prefill_config=PrefillConfig(attn_aotriton_min_tokens=512),
                )
            )
            if owner.runner is None:
                raise GateError("GGUF resident session closed during setup")
            if bool(owner.runner.fp16_recurrent_state) is not bool(fp16):
                raise GateError("runner did not freeze the requested recurrent-state dtype")
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
                            max_sequence_length=int(max_sequence_length),
                            use_wmma_prefill=True,
                            use_gemv_decode=True,
                            prefill_config=PrefillConfig(
                                attn_aotriton_min_tokens=512
                            ),
                        )
                    )
                )
        return (
            stack,
            tuple(sessions),
            str(owner.runner.backend),
            str(owner.runner.target_arch),
        )
    except Exception:
        stack.close()
        raise


def _reset_sessions(sessions: Sequence[Any]) -> None:
    for session in sessions:
        session.reset()


def _run_static_once(
    sessions: Sequence[Any],
    scenario: StaticScenario,
    *,
    reference: Sequence[Sequence[Mapping[str, object]]] | None,
) -> tuple[tuple[tuple[dict[str, object], ...], ...], dict[str, Any]]:
    active = tuple(sessions[: scenario.width])
    _reset_sessions(active)
    results = active[0].prefill_batch_native(
        scenario.token_rows,
        sessions=active,
        return_logits=True,
        require_logits=True,
    )
    trajectories: list[list[dict[str, object]]] = [[_step(result)] for result in results]
    prefill_manifest = dict(active[0].last_packed_execution_manifest)
    for decode_step in range(scenario.decode_steps):
        token_ids = [
            int(trajectories[index][-1]["token_id"])
            if reference is None
            else int(reference[index][decode_step]["token_id"])
            for index in range(scenario.width)
        ]
        results = active[0].step_batch_native(
            token_ids,
            sessions=active,
            return_logits=True,
            require_logits=True,
            scatter_state=True,
        )
        for index, result in enumerate(results):
            trajectories[index].append(_step(result))
    return (
        tuple(tuple(trajectory) for trajectory in trajectories),
        {
            "scenario_id": scenario.scenario_id,
            "prefill": prefill_manifest,
            "decode": dict(active[0].last_packed_execution_manifest),
        },
    )


def _schedule_width(
    schedule: Sequence[tuple[int, int]],
    step: int,
) -> tuple[int, bool, int | None]:
    current = int(schedule[0][1])
    previous: int | None = None
    entered = step == 0
    for start, width in schedule[1:]:
        if step < int(start):
            break
        previous = current
        current = int(width)
        entered = step == int(start)
    return current, entered, previous


def _run_dynamic_once(
    sessions: Sequence[Any],
    rows: Sequence[Mapping[str, Any]],
    prompt_tokens: Mapping[str, Sequence[int]],
    *,
    schedule: Sequence[tuple[int, int]],
    decode_steps: int,
    reference: Mapping[int, Sequence[Mapping[str, object]]] | None,
) -> tuple[
    dict[int, tuple[dict[str, object], ...]],
    dict[int, tuple[str, ...]],
    dict[int, tuple[str, ...]],
    dict[str, Any],
]:
    active = tuple(sessions[:8])
    _reset_sessions(active)
    token_rows = tuple(
        tuple(int(token) for token in prompt_tokens[str(row["id"])])
        for row in rows
    )
    results = active[0].prefill_batch_native(
        token_rows,
        sessions=active,
        return_logits=True,
        require_logits=True,
    )
    trajectories = {index: [_step(result)] for index, result in enumerate(results)}
    shapes = {index: ["c8_prefill"] for index in range(8)}
    transitions = {index: ["prefill_to_c8"] for index in range(8)}
    prefill_manifest = dict(active[0].last_packed_execution_manifest)
    survivor_order = (0, 2, 4, 6, 1, 3, 5, 7)
    for step in range(int(decode_steps)):
        width, entered, previous = _schedule_width(schedule, step)
        active_indices = survivor_order[:width]
        active_sessions = tuple(active[index] for index in active_indices)
        token_ids = [
            int(trajectories[index][-1]["token_id"])
            if reference is None
            else int(reference[index][step]["token_id"])
            for index in active_indices
        ]
        results = active_sessions[0].step_batch_native(
            token_ids,
            sessions=active_sessions,
            return_logits=True,
            require_logits=True,
            scatter_state=True,
        )
        transition = (
            f"enter_c{width}"
            if entered and previous is None
            else f"width_{previous}_to_{width}"
            if entered
            else "steady"
        )
        for original_index, result in zip(active_indices, results, strict=True):
            trajectories[original_index].append(_step(result))
            shapes[original_index].append(f"c{width}")
            transitions[original_index].append(transition)
    return (
        {index: tuple(value) for index, value in trajectories.items()},
        {index: tuple(value) for index, value in shapes.items()},
        {index: tuple(value) for index, value in transitions.items()},
        {
            "scenario_id": "dynamic_width_retirement",
            "prefill": prefill_manifest,
            "decode": dict(active[0].last_packed_execution_manifest),
        },
    )


def _run_sparse_once(
    sessions: Sequence[Any],
    rows: Sequence[Mapping[str, Any]],
    prompt_tokens: Mapping[str, Sequence[int]],
    *,
    decode_steps: int,
    reference: Sequence[Sequence[Mapping[str, object]]] | None,
) -> tuple[tuple[tuple[dict[str, object], ...], ...], dict[str, Any]]:
    active = tuple(sessions[:4])
    _reset_sessions(active)
    token_rows = tuple(
        tuple(int(token) for token in prompt_tokens[str(row["id"])])
        for row in rows
    )
    results = active[0].prefill_batch_native(
        token_rows,
        sessions=active,
        return_logits=True,
        require_logits=True,
    )
    trajectories: list[list[dict[str, object]]] = [[_step(result)] for result in results]
    prefill_manifest = dict(active[0].last_packed_execution_manifest)
    for step in range(int(decode_steps)):
        token_ids = [
            int(trajectories[index][-1]["token_id"])
            if reference is None
            else int(reference[index][step]["token_id"])
            for index in range(4)
        ]
        results = active[0].step_batch_native(
            token_ids,
            sessions=active,
            return_logits=True,
            require_logits=True,
            scatter_state=True,
            physical_rows=8,
            active_slot_indices=SPARSE_ACTIVE_SLOTS,
        )
        for index, result in enumerate(results):
            trajectories[index].append(_step(result))
    return (
        tuple(tuple(value) for value in trajectories),
        {
            "scenario_id": "sparse_physical_c8",
            "prefill": prefill_manifest,
            "decode": dict(active[0].last_packed_execution_manifest),
        },
    )


def _capture_isolation(
    sessions: Sequence[Any],
    prompt_rows: Sequence[Mapping[str, Any]],
    prompt_tokens: Mapping[str, Sequence[int]],
    *,
    decode_steps: int,
    repeat_runs: int,
) -> dict[str, Any]:
    rows = tuple(prompt_rows)
    if len(rows) < 8:
        raise GateError("isolation capture requires at least eight prompts")

    def scenario(name: str, selected: Sequence[Mapping[str, Any]]):
        return StaticScenario(
            scenario_id=name,
            width=4,
            rows=tuple(selected),
            token_rows=tuple(
                tuple(int(token) for token in prompt_tokens[str(row["id"])])
                for row in selected
            ),
            actual_count=4,
            decode_steps=int(decode_steps),
        )

    neighbor_a = scenario("neighbor_a", (rows[0], rows[1], rows[2], rows[3]))
    neighbor_b = scenario("neighbor_b", (rows[0], rows[5], rows[6], rows[7]))
    target_runs: list[tuple[dict[str, object], ...]] = []
    for _ in range(int(repeat_runs)):
        run_a, _ = _run_static_once(sessions, neighbor_a, reference=None)
        run_b, _ = _run_static_once(sessions, neighbor_b, reference=None)
        target_runs.extend((run_a[0], run_b[0]))
    neighbor_passed = all(
        _trajectories_exact(target_runs[0], run) for run in target_runs[1:]
    )

    base_rows = (rows[0], rows[1], rows[2], rows[3])
    permutation = (2, 0, 3, 1)
    permuted_rows = tuple(base_rows[index] for index in permutation)
    base = scenario("permutation_base", base_rows)
    permuted = scenario("permutation_changed", permuted_rows)
    base_run, _ = _run_static_once(sessions, base, reference=None)
    permuted_run, _ = _run_static_once(sessions, permuted, reference=None)
    inverse = {original: changed for changed, original in enumerate(permutation)}
    permutation_rows = []
    for original_index, row in enumerate(base_rows):
        changed_index = inverse[original_index]
        exact = _trajectories_exact(
            base_run[original_index],
            permuted_run[changed_index],
        )
        permutation_rows.append(
            {
                "prompt_id": str(row["id"]),
                "base_slot": original_index,
                "permuted_slot": changed_index,
                "exact": exact,
            }
        )
    permutation_passed = all(row["exact"] for row in permutation_rows)
    return {
        "passed": bool(neighbor_passed and permutation_passed),
        "neighbor_substitution": {
            "passed": neighbor_passed,
            "target_prompt_id": str(rows[0]["id"]),
            "trajectory_sha256": [
                _trajectory_sha256(run) for run in target_runs
            ],
        },
        "row_permutation": {
            "passed": permutation_passed,
            "rows": permutation_rows,
        },
    }


def _manifest_summary(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: manifest.get(key)
        for key in (
            "mode",
            "physical_rows",
            "active_rows",
            "active_mask",
            "linear_attention_decode_path",
            "gdn_recurrent_decode_path",
            "full_attention_decode_path",
            "moe_decode_path",
            "lm_head_decode_path",
            "sampler_decode_path",
        )
        if key in manifest
    }


def _decode_manifests_use_indexed(manifests: Sequence[Mapping[str, Any]]) -> bool:
    relevant = [
        manifest.get("decode", {})
        for manifest in manifests
        if int(manifest.get("decode", {}).get("physical_rows", 1) or 1) > 1
    ]
    return bool(relevant) and all(
        manifest.get("linear_attention_decode_path") == "indexed_batch"
        and manifest.get("gdn_recurrent_decode_path") == "indexed_singleton"
        for manifest in relevant
    )


def run(args: argparse.Namespace, *, command: Sequence[str]) -> dict[str, Any]:
    if not args.model.is_file():
        raise GateError(f"model does not exist: {args.model}")
    if int(args.decode_steps) < 24 or int(args.repeat_runs) < 3:
        raise GateError("complete gate needs at least 24 decode steps and three repeats")
    widths = tuple(int(value) for value in str(args.widths).split(",") if value)
    if set(widths) != SUPPORTED_WIDTHS:
        raise GateError("complete gate requires static widths 4,8")
    schedule = validate_width_schedule(
        DEFAULT_DYNAMIC_SCHEDULE,
        decode_steps=int(args.decode_steps),
    )
    prompt_rows = _load_suites(args.prompts)
    if args.limit is not None:
        prompt_rows = prompt_rows[: max(0, int(args.limit))]
    if not prompt_rows:
        raise GateError("selected prompt suites are empty")
    selected_suites = tuple(path.resolve() for path in args.prompts)
    default_suites = tuple(path.resolve() for path in DEFAULT_PROMPTS)
    complete_suite = args.limit is None and selected_suites == default_suites

    from hipengine.loading.gguf import scan_gguf
    from hipengine.tokenization.gguf import Qwen35GGUFTokenizer

    tokenizer = Qwen35GGUFTokenizer.from_gguf_info(scan_gguf(args.model))
    prompt_tokens = {
        str(row["id"]): tuple(
            int(token) for token in build_chat_prompt(tokenizer, str(row["prompt"]))
        )
        for row in prompt_rows
    }
    scenarios = build_static_scenarios(
        prompt_rows,
        prompt_tokens,
        widths=widths,
        decode_steps=int(args.decode_steps),
        long_prompt_tokens=int(args.long_prompt_tokens),
        long_decode_steps=int(args.long_decode_steps),
    )
    max_sequence_length = max(
        max(len(tokens) for tokens in prompt_tokens.values()),
        int(args.long_prompt_tokens),
    ) + int(args.decode_steps) + 4
    dynamic_rows = tuple(prompt_rows[:8])
    sparse_rows = tuple(
        prompt_rows[index]
        for index in (
            0,
            len(prompt_rows) // 3,
            2 * len(prompt_rows) // 3,
            len(prompt_rows) - 1,
        )
    )

    strict_static: dict[str, tuple[tuple[dict[str, object], ...], ...]] = {}
    strict_manifests: list[dict[str, Any]] = []
    strict_stack, strict_sessions, resolved_backend, target_arch = _make_sessions(
        args,
        fp16=False,
        max_sequence_length=max_sequence_length,
    )
    try:
        for index, scenario in enumerate(scenarios):
            trajectories, manifest = _run_static_once(
                strict_sessions,
                scenario,
                reference=None,
            )
            strict_static[scenario.scenario_id] = trajectories
            strict_manifests.append(manifest)
            print(
                f"strict static {index + 1}/{len(scenarios)} {scenario.scenario_id}",
                flush=True,
            )
        strict_dynamic, dynamic_shapes, dynamic_transitions, manifest = (
            _run_dynamic_once(
                strict_sessions,
                dynamic_rows,
                prompt_tokens,
                schedule=schedule,
                decode_steps=int(args.decode_steps),
                reference=None,
            )
        )
        strict_manifests.append(manifest)
        strict_sparse, manifest = _run_sparse_once(
            strict_sessions,
            sparse_rows,
            prompt_tokens,
            decode_steps=int(args.decode_steps),
            reference=None,
        )
        strict_manifests.append(manifest)
    finally:
        strict_stack.close()

    candidate_static: dict[
        str,
        list[tuple[tuple[dict[str, object], ...], ...]],
    ] = {scenario.scenario_id: [] for scenario in scenarios}
    candidate_dynamic: list[dict[int, tuple[dict[str, object], ...]]] = []
    candidate_sparse: list[tuple[tuple[dict[str, object], ...], ...]] = []
    candidate_manifests: list[dict[str, Any]] = []
    candidate_stack, candidate_sessions, candidate_backend, candidate_arch = (
        _make_sessions(
            args,
            fp16=True,
            max_sequence_length=max_sequence_length,
        )
    )
    if candidate_backend != resolved_backend or candidate_arch != target_arch:
        candidate_stack.close()
        raise GateError("strict and candidate resolved different backends")
    try:
        for repeat in range(int(args.repeat_runs)):
            for scenario in scenarios:
                trajectories, manifest = _run_static_once(
                    candidate_sessions,
                    scenario,
                    reference=strict_static[scenario.scenario_id],
                )
                candidate_static[scenario.scenario_id].append(trajectories)
                if repeat == 0:
                    candidate_manifests.append(manifest)
            dynamic, shapes, transitions, manifest = _run_dynamic_once(
                candidate_sessions,
                dynamic_rows,
                prompt_tokens,
                schedule=schedule,
                decode_steps=int(args.decode_steps),
                reference=strict_dynamic,
            )
            if shapes != dynamic_shapes or transitions != dynamic_transitions:
                raise GateError("candidate dynamic shape/transition attribution drifted")
            candidate_dynamic.append(dynamic)
            sparse, manifest_sparse = _run_sparse_once(
                candidate_sessions,
                sparse_rows,
                prompt_tokens,
                decode_steps=int(args.decode_steps),
                reference=strict_sparse,
            )
            candidate_sparse.append(sparse)
            if repeat == 0:
                candidate_manifests.extend((manifest, manifest_sparse))
            print(
                f"candidate repeat {repeat + 1}/{args.repeat_runs}: all scenarios",
                flush=True,
            )
        isolation = _capture_isolation(
            candidate_sessions,
            prompt_rows,
            prompt_tokens,
            decode_steps=int(args.isolation_decode_steps),
            repeat_runs=int(args.repeat_runs),
        )
    finally:
        candidate_stack.close()

    captures: list[BatchRouteCapture] = []
    for scenario in scenarios:
        strict_rows = strict_static[scenario.scenario_id]
        runs = candidate_static[scenario.scenario_id]
        shape_prefix = f"c{scenario.width}_prefill"
        transition_prefix = f"prefill_to_c{scenario.width}"
        if scenario.long_context:
            shape_prefix = f"c8_prefill_p{int(args.long_prompt_tokens)}"
            transition_prefix = "long_prefill_to_c8"
        for row_index, row in enumerate(scenario.rows[: scenario.actual_count]):
            captures.append(
                BatchRouteCapture(
                    scenario_id=scenario.scenario_id,
                    request_id=str(row["id"]),
                    category=str(row["category"]),
                    strict=strict_rows[row_index],
                    candidate_runs=tuple(run[row_index] for run in runs),
                    shapes=(shape_prefix,)
                    + (f"c{scenario.width}",) * scenario.decode_steps,
                    transitions=(transition_prefix,)
                    + ("steady",) * scenario.decode_steps,
                    teacher_steps=tuple(range(scenario.decode_steps + 1)),
                )
            )
    for row_index, row in enumerate(dynamic_rows):
        length = len(strict_dynamic[row_index])
        captures.append(
            BatchRouteCapture(
                scenario_id="dynamic_width_retirement",
                request_id=str(row["id"]),
                category=str(row["category"]),
                strict=strict_dynamic[row_index],
                candidate_runs=tuple(run[row_index] for run in candidate_dynamic),
                shapes=dynamic_shapes[row_index],
                transitions=dynamic_transitions[row_index],
                teacher_steps=tuple(range(length)),
            )
        )
    for row_index, row in enumerate(sparse_rows):
        captures.append(
            BatchRouteCapture(
                scenario_id="sparse_physical_c8",
                request_id=str(row["id"]),
                category=str(row["category"]),
                strict=strict_sparse[row_index],
                candidate_runs=tuple(run[row_index] for run in candidate_sparse),
                shapes=("c4_prefill",)
                + ("c8_sparse4",) * int(args.decode_steps),
                transitions=("prefill_to_sparse_c8",)
                + ("sparse_steady",) * int(args.decode_steps),
                teacher_steps=tuple(range(int(args.decode_steps) + 1)),
            )
        )

    evaluated = build_batch_route_quality(
        captures,
        thresholds=EvaluationThresholds(),
    )
    provenance = collect_artifact_provenance(
        repo_root=REPO_ROOT,
        configured_backend=str(args.backend),
        resolved_backend=resolved_backend,
        target_arch=target_arch,
        model_path=args.model,
        quant=str(args.quant_label),
        kv_dtype="bf16",
        command=command,
        environment={
            "HIPENGINE_HIP_ARCH": os.environ.get("HIPENGINE_HIP_ARCH"),
            "HIP_VISIBLE_DEVICES": os.environ.get("HIP_VISIBLE_DEVICES"),
            "strict_route_environment": {FP16_STATE_ENV: "0"},
            "candidate_route_environment": {FP16_STATE_ENV: "1"},
            "production_route": PRODUCTION_ROUTE,
        },
        build_profile="execution_profile_gguf_fp16_state_batch_gate",
        timing_protocol="none_full_logits_isolation_and_state_ownership_v1",
        warmups=0,
        repetitions=int(args.repeat_runs),
        profiler={"enabled": False, "kind": None, "command": None},
    )
    tracked_clean = not bool(
        provenance.get("staged_dirty") or provenance.get("unstaged_dirty")
    )
    complete_matrix = bool(
        complete_suite
        and len(prompt_rows) == 18
        and set(widths) == SUPPORTED_WIDTHS
        and int(args.decode_steps) >= 24
        and int(args.repeat_runs) >= 3
        and int(args.long_prompt_tokens) >= 512
        and int(args.long_decode_steps) >= 8
    )
    indexed_manifest_gate = bool(
        _decode_manifests_use_indexed(strict_manifests)
        and _decode_manifests_use_indexed(candidate_manifests)
    )
    measurement_valid = bool(
        tracked_clean
        and complete_matrix
        and evaluated["repeat_determinism"]["passed"]
        and isolation["passed"]
        and indexed_manifest_gate
    )
    passed = bool(measurement_valid and evaluated["quality"]["hard_gates_passed"])
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if passed else "failed_or_screen_only",
        "measurement_valid": measurement_valid,
        "performance_claim": False,
        "profile_qualification_claim": False,
        "qualification_blockers": [
            "capture is not a runtime-resolved public production-profile manifest",
            "fresh BF16-relative and external task-score verdicts are unavailable",
            "complete dynamic serving/SLO-goodput packet has not run",
        ],
        "candidate": {
            "name": CANDIDATE_NAME,
            "classification": "T1 lower-precision recurrent-state storage",
            "mechanism": "FP16 recurrent-state storage with FP32 accumulation",
            "strict_fallback": "FP32 recurrent-state storage on the same packed production route",
            "strict_environment": {FP16_STATE_ENV: "0"},
            "candidate_environment": {FP16_STATE_ENV: "1"},
            "production_route": PRODUCTION_ROUTE,
        },
        "protocol": {
            "model": str(args.model.resolve()),
            "prompt_suites": [str(path.resolve()) for path in args.prompts],
            "complete_prompt_and_heldout_suite": complete_suite,
            "prompt_count": len(prompt_rows),
            "static_widths": list(widths),
            "decode_steps": int(args.decode_steps),
            "long_prompt_tokens": int(args.long_prompt_tokens),
            "long_decode_steps": int(args.long_decode_steps),
            "dynamic_schedule": [list(value) for value in schedule],
            "sparse_physical_c8_active_slots": list(SPARSE_ACTIVE_SLOTS),
            "candidate_repeat_runs": int(args.repeat_runs),
            "teacher_forced_rows": int(evaluated["quality"]["summary"]["rows"]),
            "thresholds": EvaluationThresholds().to_dict(),
        },
        "quality": evaluated,
        "isolation": isolation,
        "manifest_gate": {
            "passed": indexed_manifest_gate,
            "requirement": "all c>N decode manifests use indexed_batch/indexed_singleton",
            "strict": [
                {
                    "scenario_id": row["scenario_id"],
                    "prefill": _manifest_summary(row["prefill"]),
                    "decode": _manifest_summary(row["decode"]),
                }
                for row in strict_manifests
            ],
            "candidate": [
                {
                    "scenario_id": row["scenario_id"],
                    "prefill": _manifest_summary(row["prefill"]),
                    "decode": _manifest_summary(row["decode"]),
                }
                for row in candidate_manifests
            ],
        },
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
        "provenance": provenance,
        "worktree_note": {
            "tracked_clean": tracked_clean,
            "untracked_dirty": bool(provenance.get("untracked_dirty")),
            "untracked_count": int(provenance.get("untracked_count", 0)),
            "note": "pre-existing untracked benchmark artifacts do not alter the committed code under test",
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--backend", default="hip_gfx1151")
    parser.add_argument("--quant-label", default="gguf_q4_k_s")
    parser.add_argument("--prompts", action="append", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--widths", default="4,8")
    parser.add_argument("--decode-steps", type=int, default=24)
    parser.add_argument("--repeat-runs", type=int, default=3)
    parser.add_argument("--long-prompt-tokens", type=int, default=512)
    parser.add_argument("--long-decode-steps", type=int, default=8)
    parser.add_argument("--isolation-decode-steps", type=int, default=8)
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
        payload = run(args, command=command)
    except (GateError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(args.json)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "measurement_valid": payload["measurement_valid"],
                "hard_gate_passed": payload["quality"]["quality"]["hard_gates_passed"],
                "repeat_deterministic": payload["quality"]["repeat_determinism"]["passed"],
                "isolation_passed": payload["isolation"]["passed"],
                "manifest_gate_passed": payload["manifest_gate"]["passed"],
                "summary": payload["quality"]["quality"]["summary"],
            },
            indent=2,
        )
    )
    return 0 if payload["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
