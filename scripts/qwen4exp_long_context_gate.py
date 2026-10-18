#!/usr/bin/env python3
"""Long-prompt production gate for Qwen4Exp: round-off floor plus retrieval.

The calibrated production envelope (mean/p95/p99/max KL against ``strict``)
was fitted on prompts of tens to hundreds of tokens. At 2,000 tokens and
beyond, ANY perturbation of round-off size moves the distribution by about
2e-3 mean KL against ``strict`` (an F32 summation-order change in one
projection does), so the short-prompt limits cannot tell a defect from
round-off there. This gate replaces fixed limits with two measurements that
stay meaningful at length (EXECUTION-PROFILES section 6.1):

1. **Round-off floor (numerical axis).** On the long numerical prompts,
   ``strict`` is rerun with a seeded perturbation of one BF16 ulp in one
   element of one prompt embedding row: the smallest change a single rounding
   flip in the BF16 residual stream can make. Each seed is a control arm.
   The candidate passes when its mean and p95 KL against ``strict`` do not
   exceed the median control's, its max KL does not exceed the largest
   control max, and its top-1 agreement is not below the lowest control's.
   In words: the candidate must diverge no more than a typical single
   rounding flip does. (Calibration on 2026-10-14: one flip gives all-prompt
   mean KL 4.5e-3 to 7.7e-3 on the four numerical prompts, more than the
   complete F16 route composition's 2.1e-3.)
2. **Retrieval (task axis).** On 8K and 16K retrieval prompts with answers
   stored in the suite, every arm is teacher-forced through the gold answer
   text. An answer counts as correct when the arm's argmax reproduces every
   token of that answer's number. The candidate must answer at least as many
   as ``strict``. The gold-answer NLL delta is reported alongside.

Also binding: candidate repeats are bit-identical, controls are effective (a
control that changes nothing measures no floor) and below a sanity cap.

For the AR batch arm, pass ``--candidate ar_batch --batch-width 1`` or ``2``.
Prefill stays on the c1 runner; decode is a genuine N-row ``owner.step``. Width
4 is rejected because its live bound is 2048 positions, below these fixtures.
The batch state/ownership gate and the batch variant manifest binding are
binding, and the arm needs at least three repeats.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hipengine.benchmark.execution_profiles import RowDescriptor, compare_profile_logits
from scripts.gguf_mtp_bench import build_chat_prompt
from scripts.qwen4exp_canonical_ar_bench import (
    _git_metadata,
    _host_metadata,
    _write_json,
    sha256_path,
)
from scripts.qwen4exp_layer2_profile_gate import (
    CANDIDATES,
    GateError,
    _candidate_trajectory,
    _make_generator,
    _post_construct_hook,
    _state_summary,
    _strict_trajectory,
)

KIND = "qwen4exp_long_context_profile_gate"
DEFAULT_NUMERIC = ROOT / "benchmarks" / "prompts" / "qwen4exp-long-context-gate.jsonl"
DEFAULT_RETRIEVAL = ROOT / "benchmarks" / "prompts" / "qwen4exp-long-retrieval-v1.jsonl"
# A control whose mean KL exceeds this is no longer a round-off-sized floor.
CONTROL_MEAN_KL_CAP = 1e-2
#: Long-context AR batch widths with a measured adapter path. Width 4 is
#: refused because its live runtime bound is 2048 positions, below this gate's
#: 2K-class numerical and 8K/16K retrieval fixtures; the gate must not lower
#: the context to make width 4 appear supported.
SUPPORTED_BATCH_WIDTHS = (1, 2)
#: The batch state gate compares at least three repeats.
BATCH_STATE_REPEATS = 3
#: Exact control basis: a prompt-embedding override, not a residual control.
CONTROL_BASIS = "prompt_embedding_row_one_bf16_ulp"
CONTROL_APPLICATION = "runner.prefill(embedding_overrides=...)"


def _load(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _prompt_text(row: Mapping[str, Any]) -> str:
    return str(row["messages"][0]["content"])


def _embedding_bits(runner: Any, token_id: int) -> np.ndarray:
    from hipengine.core.memory import copy_device_to_host, copy_host_to_device, host_array_ptr
    from hipengine.kernels.registry import resolve

    cfg = runner.config
    weight = runner.resident.weight("root.token_embedding")
    lookup = resolve(backend=runner.backend, layer="embedding",
                     quant=weight.spec.quant_key, variant="lookup_bf16_out")
    host = np.asarray([int(token_id)], dtype=np.int32)
    copy_host_to_device(runner.token_id_buffer, host_array_ptr(host), runtime=runner.runtime)
    lookup(runner.token_id_buffer.ptr, weight.allocation("raw").tensor.ptr,
           runner.embedding_buffer.ptr, 1, cfg.hidden_size, cfg.vocab_size,
           runtime=runner.runtime)
    bits = np.zeros(cfg.hidden_size, dtype=np.uint16)
    copy_device_to_host(host_array_ptr(bits), runner.embedding_buffer, bits.nbytes,
                        runtime=runner.runtime)
    return bits


def roundoff_override(runner: Any, prompt_ids: Sequence[int], seed: int
                      ) -> tuple[dict[int, np.ndarray], dict[str, int]]:
    """One BF16 ulp, up or down, in one element of one prompt embedding row."""

    rng = np.random.default_rng(int(seed))
    position = int(rng.integers(len(prompt_ids) - 1))
    bits = _embedding_bits(runner, int(prompt_ids[position])).copy()
    element = int(rng.integers(bits.size))
    step = 1 if int(rng.integers(2)) else -1
    bits[element] = np.uint16((int(bits[element]) + step) & 0xFFFF)
    values = (bits.astype(np.uint32) << 16).view(np.float32)
    return {position: values}, {"position": position, "element": element, "ulp_step": step}


def _forced_trajectory(runner: Any, prompt_ids: Sequence[int], forced: Sequence[int],
                       overrides: Mapping[int, np.ndarray] | None = None
                       ) -> tuple[dict[str, Any], ...]:
    if overrides is None:
        return _candidate_trajectory(runner, prompt_ids, forced)
    runner.reset()
    result = runner.prefill([int(t) for t in prompt_ids], embedding_overrides=overrides)
    out = [{"token_id": int(result.token_id),
            "logits": np.ascontiguousarray(result.logits, dtype=np.float32)}]
    for token in forced:
        result = runner.step(int(token))
        out.append({"token_id": int(result.token_id),
                    "logits": np.ascontiguousarray(result.logits, dtype=np.float32)})
    return tuple(out)


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    values = logits.astype(np.float64)
    values = values - values.max()
    return values - np.log(np.exp(values).sum())


def gold_answer_tokens(tokenizer: Any, answers: Sequence[int]) -> tuple[list[int], list[list[int]]]:
    """Gold answer text ``1: <n>\\n2: <n>...`` and the token indices of each number."""

    ids: list[int] = []
    spans: list[list[int]] = []
    for index, answer in enumerate(answers, start=1):
        prefix = tokenizer.encode(f"{index}: " if index == 1 else f"\n{index}: ")
        ids.extend(prefix)
        number = tokenizer.encode(str(int(answer)))
        spans.append(list(range(len(ids), len(ids) + len(number))))
        ids.extend(number)
    return ids, spans


def score_retrieval(trajectory: Sequence[Mapping[str, Any]], gold: Sequence[int],
                    spans: Sequence[Sequence[int]]) -> dict[str, Any]:
    """Row ``i`` of a gold-forced trajectory predicts gold token ``i``."""

    nll = [-float(_log_softmax(trajectory[i]["logits"])[int(token)])
           for i, token in enumerate(gold)]
    hits = [int(np.argmax(trajectory[i]["logits"])) == int(gold[i]) for i in range(len(gold))]
    correct = [all(hits[i] for i in span) for span in spans]
    return {"correct": correct, "correct_count": int(sum(correct)),
            "gold_nll_mean": float(np.mean(nll)),
            "answer_nll_mean": float(np.mean([nll[i] for span in spans for i in span]))}


def _summary(strict: Mapping[str, Sequence[Mapping[str, Any]]],
             arm: Mapping[str, Sequence[Mapping[str, Any]]],
             categories: Mapping[str, str], name: str) -> dict[str, Any]:
    rows: list[RowDescriptor] = []
    left, right = [], []
    for prompt_id, teacher in strict.items():
        candidate_rows = arm[prompt_id]
        if len(teacher) != len(candidate_rows):
            raise GateError(
                f"strict/candidate trajectory length mismatch for {prompt_id!r}: "
                f"{len(teacher)} != {len(candidate_rows)}"
            )
        for step, (s_row, c_row) in enumerate(zip(teacher, candidate_rows)):
            rows.append(RowDescriptor(
                scenario_id=name, scenario_step=len(rows), request_id=prompt_id,
                teacher_step=step, category=categories[prompt_id],
                shape="prefill_last" if step == 0 else "c1",
                transition="prefill_to_c1" if step == 0 else "steady",
                teacher_token_id=int(s_row["token_id"])))
            left.append(s_row["logits"])
            right.append(c_row["logits"])
    result = compare_profile_logits(np.stack(left), np.stack(right), rows)
    return dict(result["summary"])


def decide(candidate: Mapping[str, Any], controls: Sequence[Mapping[str, Any]],
           candidate_retrieval: int, strict_retrieval: int, *, deterministic: bool,
           batch_state_passed: bool | None = None,
           batch_manifest_bound: bool | None = None) -> dict[str, Any]:
    """The binding rules, separated from measurement so they can be unit-tested."""

    floor = {
        "kl_mean": float(np.median([float(c["kl_mean"]) for c in controls])),
        "kl_p95": float(np.median([float(c["kl_p95"]) for c in controls])),
        "kl_max": max(float(c["kl_max"]) for c in controls),
        "top1_matches": min(int(c["top1_matches"]) for c in controls),
    }
    checks = {
        "controls_effective": all(float(c["kl_max"]) > 0.0 for c in controls),
        "controls_round_off_sized": floor["kl_mean"] <= CONTROL_MEAN_KL_CAP,
        "mean_within_floor": float(candidate["kl_mean"]) <= floor["kl_mean"],
        "p95_within_floor": float(candidate["kl_p95"]) <= floor["kl_p95"],
        "max_within_floor": float(candidate["kl_max"]) <= floor["kl_max"],
        "top1_within_floor": int(candidate["top1_matches"]) >= floor["top1_matches"],
        "retrieval_not_below_strict": int(candidate_retrieval) >= int(strict_retrieval),
        "candidate_repeats_identical": bool(deterministic),
    }
    if batch_state_passed is not None:
        checks["batch_state_gate"] = bool(batch_state_passed)
    if batch_manifest_bound is not None:
        checks["batch_manifest_bound"] = bool(batch_manifest_bound)
    valid = checks["controls_effective"] and checks["controls_round_off_sized"]
    if batch_manifest_bound is False:
        # A batch manifest that does not name the measured path is not evidence.
        valid = False
    passed = valid and all(checks.values())
    return {"floor": floor, "checks": checks, "measurement_valid": valid,
            "status": "passed" if passed else "failed" if valid else "invalid"}


def candidate_arm_kind(spec: Any) -> str:
    """Return the execution arm for a candidate; ``ar_batch`` is never c1."""

    return "batch" if str(spec.name) == "qwen4_exp_ar_batch" else "c1"


def resolve_batch_width(args: argparse.Namespace, spec: Any) -> int | None:
    """Resolve and fail closed on the long-context batch width."""

    requested = getattr(args, "batch_width", None)
    if candidate_arm_kind(spec) != "batch":
        if requested is not None:
            raise GateError("--batch-width is only valid with --candidate ar_batch")
        return None
    if requested is None:
        raise GateError(
            "--candidate ar_batch requires --batch-width 1 or 2; "
            "the long-context gate has no c1 fallback"
        )
    width = int(requested)
    if width == 4:
        raise GateError(
            "--candidate ar_batch --batch-width 4 is unsupported by the "
            "long-context gate: the width-4 runtime is qualified only to 2048 "
            "positions, below the 2K-class/8K/16K fixtures; use --batch-width "
            "1 or 2; do not lower context"
        )
    if width not in SUPPORTED_BATCH_WIDTHS:
        raise GateError("--candidate ar_batch requires --batch-width 1 or 2")
    return width


def plan_long_batch_groups(
    numeric_requests: Sequence[Any],
    retrieval_requests: Sequence[Any],
    width: int,
) -> tuple[list[Any], dict[str, dict[str, Any]]]:
    """Plan honest N-row groups for numeric and retrieval schedules.

    Numeric prompts share the teacher-forced length. Retrieval prompts are
    bucketed by gold-answer length because the adapter requires one forced
    length per group; a bucket with one request keeps its natural width 1.
    """

    from scripts.qwen4exp_ar_batch_candidate import plan_batch_composition

    parsed_width = int(width)
    if parsed_width not in SUPPORTED_BATCH_WIDTHS:
        raise ValueError(
            f"long-context batch width must be one of {SUPPORTED_BATCH_WIDTHS}, "
            f"got {width!r}"
        )
    all_requests = tuple(numeric_requests) + tuple(retrieval_requests)
    if not all_requests:
        raise ValueError("long-context batch composition needs at least one request")
    prompt_ids = [str(request.prompt_id) for request in all_requests]
    request_ids = [int(request.request_id) for request in all_requests]
    if len(set(prompt_ids)) != len(prompt_ids):
        raise ValueError("long-context batch composition has duplicate prompt ids")
    if len(set(request_ids)) != len(request_ids):
        raise ValueError("long-context batch composition has duplicate request ids")

    groups: list[Any] = []
    schedule: dict[str, dict[str, Any]] = {}
    for schedule_name, requests in (
        ("numeric", tuple(numeric_requests)),
        ("retrieval", tuple(retrieval_requests)),
    ):
        buckets: dict[int, list[Any]] = {}
        order: list[int] = []
        for request in requests:
            forced_length = len(request.forced_ids)
            if forced_length not in buckets:
                buckets[forced_length] = []
                order.append(forced_length)
            buckets[forced_length].append(request)
        for forced_length in order:
            for group in plan_batch_composition(buckets[forced_length], parsed_width):
                groups.append(replace(group, index=len(groups)))
                for request in group.requests:
                    schedule[str(request.prompt_id)] = {
                        "schedule": schedule_name,
                        "forced_length": int(forced_length),
                    }
    return groups, schedule


def _run_batch_candidate(
    candidate_generator: Any,
    *,
    groups: Sequence[Any],
    width: int,
    repeats: int,
    candidate_mode: str,
    source_revision: str,
    owner_factory: Any | None = None,
) -> dict[str, Any]:
    """Run the existing AR batch adapter for every group and repeat."""

    from scripts.qwen4exp_ar_batch_candidate import Qwen4ExpARBatchCandidate

    adapter = Qwen4ExpARBatchCandidate(
        candidate_generator,
        width=int(width),
        label=str(candidate_mode),
        owner_factory=owner_factory,
    )
    runs: dict[str, list[tuple[Mapping[str, Any], ...]]] = {}
    states: dict[str, list[Mapping[str, Any]]] = {}
    groups_by_repeat: list[list[dict[str, Any]]] = []
    try:
        adapter.open()
        for repeat in range(int(repeats)):
            before = len(adapter.evidence["groups"])
            for group in groups:
                result = adapter.run_group(group)
                for prompt_id, trajectory in result.trajectories.items():
                    runs.setdefault(str(prompt_id), []).append(tuple(trajectory))
                for prompt_id, state in result.states.items():
                    states.setdefault(str(prompt_id), []).append(state)
                print(
                    f"candidate batch group {group.index} width {group.width} "
                    f"{list(group.prompt_ids)}",
                    flush=True,
                )
            groups_by_repeat.append(
                [
                    {**group, "repeat": repeat}
                    for group in adapter.evidence["groups"][before:]
                ]
            )
        manifest = adapter.build_variant_manifest(source_revision=str(source_revision))
        evidence = adapter.evidence
    finally:
        adapter.close()
    return {
        "runs": runs,
        "states": states,
        "groups_by_repeat": groups_by_repeat,
        "evidence": evidence,
        "manifest": manifest,
    }


def _batch_manifest_binding(
    manifest: Mapping[str, Any],
    *,
    candidate_profile: Any,
    source_revision: str,
) -> dict[str, Any]:
    """Bind the batch manifest to the live profile and source revision."""

    from hipengine.runtime.qwen4_exp_batch_manifest import (
        validate_batch_variant_manifest,
    )

    profile_manifest = candidate_profile.manifest
    return validate_batch_variant_manifest(
        manifest,
        expected_profile_manifest_sha256=candidate_profile.manifest_sha256,
        expected_source_revision=str(source_revision),
        expected_backend=profile_manifest["backend"],
        expected_model=profile_manifest["model"],
        expected_quant=profile_manifest["quant"],
    )


def run(args: argparse.Namespace, *, command: Sequence[str]) -> dict[str, Any]:
    source = _git_metadata(ROOT)
    if not source["tracked_clean"]:
        raise GateError("profile qualification requires a clean committed tracked tree")
    if args.controls < 2 or args.candidate_repeats < 2:
        raise GateError("the floor needs at least two controls; determinism two repeats")
    spec = CANDIDATES[str(args.candidate)]
    batch_width = resolve_batch_width(args, spec)
    batch_candidate = candidate_arm_kind(spec) == "batch"
    if batch_candidate and int(args.candidate_repeats) < BATCH_STATE_REPEATS:
        raise GateError(
            "the AR batch ownership/repeat gate requires at least "
            f"{BATCH_STATE_REPEATS} candidate repeats"
        )
    from hipengine.generation.qwen4_exp_profiles import register_qwen4_exp_gfx1151_profiles
    from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_selected_prefill import (
        register_gguf_q4_k_selected_prefill_kernels,
    )
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels

    register_gfx1151_kernels(replace=True)
    register_gguf_q4_k_selected_prefill_kernels(replace=True)
    register_qwen4_exp_gfx1151_profiles()
    numeric = _load(args.numeric_prompts)
    retrieval = _load(args.retrieval_prompts)
    categories = {str(r["id"]): str(r["category"]) for r in numeric}
    started = time.time()

    prompt_ids: dict[str, list[int]] = {}
    strict_numeric: dict[str, tuple] = {}
    strict_numeric_states: dict[str, dict[str, Any]] = {}
    control_runs: list[dict[str, tuple]] = [{} for _ in range(args.controls)]
    control_meta: list[dict[str, Any]] = [{} for _ in range(args.controls)]
    gold: dict[str, tuple[list[int], list[list[int]]]] = {}
    strict_retrieval: dict[str, dict[str, Any]] = {}
    strict_retrieval_states: dict[str, dict[str, Any]] = {}
    strict_gen, strict_profile, _ = _make_generator(args, "strict")
    try:
        runner, tokenizer = strict_gen.runner, strict_gen.tokenizer
        for row in numeric:
            pid = str(row["id"])
            ids = [int(t) for t in build_chat_prompt(tokenizer, _prompt_text(row))]
            prompt_ids[pid] = ids
            teacher = _strict_trajectory(runner, ids, int(args.decode_steps))
            strict_numeric[pid] = teacher
            strict_numeric_states[pid] = _state_summary(runner)
            forced = [int(step["token_id"]) for step in teacher[:-1]]
            for index in range(args.controls):
                override, meta = roundoff_override(runner, ids, args.control_seed + index)
                control_runs[index][pid] = _forced_trajectory(runner, ids, forced, override)
                control_meta[index][pid] = meta
            print(f"strict+controls {pid} ({len(ids)} tokens)", flush=True)
        for row in retrieval:
            pid = str(row["id"])
            ids = [int(t) for t in build_chat_prompt(tokenizer, _prompt_text(row))]
            prompt_ids[pid] = ids
            gold[pid] = gold_answer_tokens(tokenizer, row["answers"])
            trajectory = _forced_trajectory(runner, ids, gold[pid][0][:-1])
            strict_retrieval[pid] = {"trajectory": trajectory,
                                     **score_retrieval(trajectory, *gold[pid])}
            strict_retrieval_states[pid] = _state_summary(runner)
            print(f"strict retrieval {pid}: {strict_retrieval[pid]['correct']}", flush=True)
    finally:
        strict_gen.close()

    candidate_numeric: list[dict[str, tuple]] = [{} for _ in range(args.candidate_repeats)]
    candidate_retrieval: list[dict[str, dict[str, Any]]] = [
        {} for _ in range(args.candidate_repeats)]
    batch_result: dict[str, Any] | None = None
    batch_binding: dict[str, Any] | None = None
    batch_state_gate_result: dict[str, Any] | None = None
    batch_schedule: dict[str, dict[str, Any]] = {}
    batch_groups: tuple[Any, ...] = ()
    saved = {key: os.environ.get(key) for key in spec.environment}
    gen, candidate_profile, _ = _make_generator(args, spec.base_profile, spec.quant)
    _post_construct_hook(gen, args)
    os.environ.update(spec.environment)
    try:
        if batch_candidate:
            from scripts.qwen4exp_ar_batch_candidate import BatchCandidateRequest

            numeric_requests: list[Any] = []
            request_id = 1
            for pid, teacher in strict_numeric.items():
                forced = tuple(int(step["token_id"]) for step in teacher[:-1])
                numeric_requests.append(
                    BatchCandidateRequest(
                        prompt_id=str(pid),
                        request_id=request_id,
                        prompt_ids=tuple(int(token) for token in prompt_ids[pid]),
                        forced_ids=forced,
                    )
                )
                request_id += 1
            retrieval_requests: list[Any] = []
            for pid in strict_retrieval:
                forced = tuple(int(token) for token in gold[pid][0][:-1])
                retrieval_requests.append(
                    BatchCandidateRequest(
                        prompt_id=str(pid),
                        request_id=request_id,
                        prompt_ids=tuple(int(token) for token in prompt_ids[pid]),
                        forced_ids=forced,
                    )
                )
                request_id += 1
            try:
                planned_groups, batch_schedule = plan_long_batch_groups(
                    numeric_requests, retrieval_requests, int(batch_width)
                )
            except ValueError as error:
                raise GateError(f"batch composition rejected: {error}") from error
            batch_groups = tuple(planned_groups)
            batch_result = _run_batch_candidate(
                gen,
                groups=batch_groups,
                width=int(batch_width),
                repeats=int(args.candidate_repeats),
                candidate_mode=f"{spec.name}_w{int(batch_width)}",
                source_revision=str(source["head"] or ""),
            )
            for repeat in range(int(args.candidate_repeats)):
                for pid in strict_numeric:
                    candidate_numeric[repeat][pid] = batch_result["runs"][pid][repeat]
                for pid in strict_retrieval:
                    trajectory = batch_result["runs"][pid][repeat]
                    candidate_retrieval[repeat][pid] = {
                        "trajectory": trajectory,
                        **score_retrieval(trajectory, *gold[pid]),
                    }
                print(
                    f"candidate batch repeat {repeat + 1}/"
                    f"{args.candidate_repeats}",
                    flush=True,
                )
            batch_binding = _batch_manifest_binding(
                batch_result["manifest"],
                candidate_profile=candidate_profile,
                source_revision=str(source["head"] or ""),
            )
            strict_state_order = [
                {**strict_numeric_states[pid], "prompt_id": str(pid)}
                for pid in strict_numeric
            ] + [
                {**strict_retrieval_states[pid], "prompt_id": str(pid)}
                for pid in strict_retrieval
            ]
            candidate_state_order = [
                batch_result["states"][pid] for pid in strict_numeric
            ] + [
                batch_result["states"][pid] for pid in strict_retrieval
            ]
            from scripts.qwen4exp_ar_batch_candidate import batch_state_gate

            batch_state_gate_result = batch_state_gate(
                strict_state_order,
                candidate_state_order,
                groups=[
                    group
                    for repeat_groups in batch_result["groups_by_repeat"]
                    for group in repeat_groups
                ],
            )
        else:
            runner = gen.runner
            for repeat in range(args.candidate_repeats):
                for pid, teacher in strict_numeric.items():
                    forced = [int(step["token_id"]) for step in teacher[:-1]]
                    candidate_numeric[repeat][pid] = _forced_trajectory(
                        runner, prompt_ids[pid], forced
                    )
                for row in retrieval:
                    pid = str(row["id"])
                    trajectory = _forced_trajectory(
                        runner, prompt_ids[pid], gold[pid][0][:-1]
                    )
                    candidate_retrieval[repeat][pid] = {
                        "trajectory": trajectory,
                        **score_retrieval(trajectory, *gold[pid]),
                    }
                print(f"candidate repeat {repeat + 1}/{args.candidate_repeats}", flush=True)
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        gen.close()

    def same(a: Sequence[Mapping[str, Any]], b: Sequence[Mapping[str, Any]]) -> bool:
        return len(a) == len(b) and all(
            np.array_equal(x["logits"], y["logits"]) for x, y in zip(a, b)
        )

    deterministic = all(
        same(candidate_numeric[0][pid], run[pid])
        for run in candidate_numeric[1:] for pid in strict_numeric
    ) and all(
        same(candidate_retrieval[0][pid]["trajectory"], run[pid]["trajectory"])
        for run in candidate_retrieval[1:] for pid in strict_retrieval
    )
    controls = [_summary(strict_numeric, runs, categories, f"control_{i}")
                for i, runs in enumerate(control_runs)]
    candidate = _summary(strict_numeric, candidate_numeric[0], categories, spec.name)
    retrieval_categories = {str(r["id"]): str(r["category"]) for r in retrieval}
    retrieval_kl = _summary(
        {pid: v["trajectory"] for pid, v in strict_retrieval.items()},
        {pid: v["trajectory"] for pid, v in candidate_retrieval[0].items()},
        retrieval_categories, spec.name + "_retrieval")
    strict_correct = sum(v["correct_count"] for v in strict_retrieval.values())
    candidate_correct = sum(v["correct_count"] for v in candidate_retrieval[0].values())
    decision = decide(
        candidate,
        controls,
        candidate_correct,
        strict_correct,
        deterministic=deterministic,
        batch_state_passed=(
            bool(batch_state_gate_result["passed"])
            if batch_state_gate_result is not None
            else None
        ),
        batch_manifest_bound=(
            bool(batch_binding["valid"] and batch_binding["bound"])
            if batch_binding is not None
            else None
        ),
    )
    source = _git_metadata(ROOT)
    if not source["tracked_clean"]:
        decision["status"] = "invalid"
        decision["measurement_valid"] = False

    def retrieval_rows(table: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
        return {pid: {k: v for k, v in row.items() if k != "trajectory"}
                for pid, row in table.items()}

    return {
        "schema": 1,
        "kind": KIND,
        "status": decision["status"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": round(time.time() - started, 1),
        "performance_claim": False,
        "source": source,
        "host": _host_metadata(),
        "command": list(command),
        "candidate": {"id": str(args.candidate), "name": spec.name,
                      "arm_kind": candidate_arm_kind(spec),
                      "batch_width": batch_width,
                      "base_profile": spec.base_profile,
                      "environment": dict(spec.environment),
                      "manifest_sha256": candidate_profile.manifest_sha256,
                      "batch_manifest_sha256": (
                          batch_binding.get("manifest_sha256")
                          if batch_binding is not None else None
                      )},
        "strict_manifest_sha256": strict_profile.manifest_sha256,
        "protocol": {
            "numeric_prompts": str(args.numeric_prompts),
            "numeric_prompts_sha256": sha256_path(args.numeric_prompts),
            "retrieval_prompts": str(args.retrieval_prompts),
            "retrieval_prompts_sha256": sha256_path(args.retrieval_prompts),
            "prompt_tokens": {pid: len(ids) for pid, ids in prompt_ids.items()},
            "decode_steps": int(args.decode_steps),
            "controls": int(args.controls),
            "control_seed": int(args.control_seed),
            "control_perturbation": "one BF16 ulp in one element of one prompt embedding row",
            "control_basis": CONTROL_BASIS,
            "control_application": CONTROL_APPLICATION,
            "candidate_repeats": int(args.candidate_repeats),
            "batch_width": batch_width,
            "batch_widths_supported": list(SUPPORTED_BATCH_WIDTHS),
            "batch_state_repeats": (
                int(args.candidate_repeats) if batch_candidate else None
            ),
            "prefill_chunk_size": int(args.prefill_chunk_size),
            "max_sequence_length": int(args.max_sequence_length),
            "control_mean_kl_cap": CONTROL_MEAN_KL_CAP,
        },
        "numeric": {"candidate": candidate, "controls": controls,
                    "control_perturbations": control_meta},
        "batch": (
            {
                "width": batch_width,
                "groups": batch_result["evidence"].get("groups", []),
                "groups_by_repeat": batch_result["groups_by_repeat"],
                "schedule": batch_schedule,
                "request_count": sum(
                    len(group.requests) for group in batch_groups
                ),
                "step_widths": batch_result["evidence"].get("step_widths"),
                "imports": batch_result["evidence"].get("imports"),
                "steps": batch_result["evidence"].get("steps"),
                "prefill_source": batch_result["evidence"].get("prefill_source"),
                "manifest": batch_result["manifest"],
                "manifest_binding": batch_binding,
                "state_gate": batch_state_gate_result,
            }
            if batch_result is not None
            else None
        ),
        "retrieval": {
            "strict_correct": strict_correct,
            "candidate_correct": candidate_correct,
            "answers": sum(len(r["answers"]) for r in retrieval),
            "strict": retrieval_rows(strict_retrieval),
            "candidate": retrieval_rows(candidate_retrieval[0]),
            "gold_forced_kl": retrieval_kl,
        },
        "decision": decision,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--candidate", default="production_baseline", choices=tuple(CANDIDATES))
    parser.add_argument(
        "--batch-width",
        type=int,
        default=None,
        help=(
            "AR batch width for --candidate ar_batch; this long-context gate "
            "supports 1 or 2 and rejects width 4 rather than lowering context"
        ),
    )
    parser.add_argument("--numeric-prompts", type=Path, default=DEFAULT_NUMERIC)
    parser.add_argument("--retrieval-prompts", type=Path, default=DEFAULT_RETRIEVAL)
    parser.add_argument("--controls", type=int, default=3)
    parser.add_argument("--control-seed", type=int, default=1)
    parser.add_argument("--decode-steps", type=int, default=24)
    parser.add_argument("--candidate-repeats", type=int, default=2)
    parser.add_argument("--max-sequence-length", type=int, default=16640)
    parser.add_argument("--prefill-chunk-size", type=int, default=4096)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = run(args, command=[Path(sys.argv[0]).name, *(argv or sys.argv[1:])])
    except GateError as error:
        parser.error(str(error))
    _write_json(args.output, result)
    print(json.dumps({"status": result["status"], "checks": result["decision"]["checks"],
                      "floor": result["decision"]["floor"],
                      "candidate": {k: result["numeric"]["candidate"][k]
                                    for k in ("kl_mean", "kl_p95", "kl_max", "top1_matches")},
                      "retrieval": [result["retrieval"]["strict_correct"],
                                    result["retrieval"]["candidate_correct"]]}, indent=1))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
