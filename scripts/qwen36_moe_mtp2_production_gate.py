#!/usr/bin/env python3
"""Qualify the Qwen3.6-35B MoE GGUF Generation-2 C1/K2 production route.

The packet keeps control and arithmetic contracts separate:

* live Generation-2 requests capture the real provider drafts, native target
  graph decisions, bounded commits, repeat behavior, reverse-order isolation,
  manifests, and lifecycle drain;
* strict AR supplies the teacher token trajectory;
* the production verifier is force-fed that strict trajectory and compared at
  every full-vocabulary K1/K2 row, with three byte-stable candidate repeats;
* actual graph decisions are reconciled against the eager diagnostic verifier;
* fixed prompt-specific task features are paired against strict without making
  generated-ID equality a production gate.

Raw logits never enter Git.  The output contains row metrics and hashes only.
"""

from __future__ import annotations

import argparse
import ast
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine import LLM, SamplingParams  # noqa: E402
from hipengine.benchmark.provenance import collect_artifact_provenance  # noqa: E402
from hipengine.loading.gguf import scan_gguf  # noqa: E402
from hipengine.runtime.prefill import PrefillConfig  # noqa: E402
from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession  # noqa: E402
from hipengine.tokenization.gguf import Qwen35GGUFTokenizer  # noqa: E402
from scripts.gguf_mtp_c1c8_server_bench import load_prompt_suite  # noqa: E402
from scripts.gguf_mtp_forced_target_probe import (  # noqa: E402
    _probe_bulk_or_native,
    _probe_serial,
)
from scripts.quant_quality.metrics import per_row_metrics  # noqa: E402

KIND = "qwen36_moe_mtp2_production_gate"
MODEL_SHA256 = "0b21525e972670ed59e1812e170b27c26355381f0656ecc4e25617ece7dac58b"
THRESHOLDS = {
    "mean_kl_max": 1.0e-3,
    "p95_kl_max": 5.0e-3,
    "p99_kl_max": 2.0e-2,
    "max_kl_max": 5.0e-2,
    "top1_min": 0.99,
    "per_scope_top1_min": 0.97,
}


class GateError(RuntimeError):
    """Raised when the packet cannot be evaluated honestly."""


@dataclass(frozen=True, slots=True)
class PromptResult:
    prompt_id: str
    category: str
    heldout: bool
    token_ids: tuple[int, ...]
    text: str


@dataclass(frozen=True, slots=True)
class LiveCycle:
    cycle: int
    root_token: int
    root_position: int
    remaining_decode: int
    draft_tokens: tuple[int, ...]
    target_tokens: tuple[int, ...]
    output_tokens: tuple[int, ...]
    accepted: int
    graph: bool

    def as_replay_row(self) -> dict[str, Any]:
        return {
            "cycle": self.cycle,
            "cycle_start_seq_position": self.root_position,
            "cycle_prev_token": self.root_token,
            "draft_tokens": list(self.draft_tokens),
            "target_tokens": list(self.target_tokens),
            "output_tokens": list(self.output_tokens),
            "accepted_draft_tokens": self.accepted,
        }

    def stable_payload(self) -> dict[str, Any]:
        return {
            "cycle": self.cycle,
            "root_token": self.root_token,
            "root_position": self.root_position,
            "remaining_decode": self.remaining_decode,
            "draft_tokens": list(self.draft_tokens),
            "target_tokens": list(self.target_tokens),
            "output_tokens": list(self.output_tokens),
            "accepted": self.accepted,
            "graph": self.graph,
        }


@dataclass(frozen=True, slots=True)
class TeacherCycle:
    cycle: int
    output_index: int
    start_position: int
    root_token: int
    inputs: tuple[int, ...]
    strict_target_tokens: tuple[int, ...]
    accepted: int
    outputs: tuple[int, ...]

    @property
    def shape(self) -> str:
        return f"k{len(self.inputs) - 1}"


def _sha256_json(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str:
    completed = subprocess.run(
        ("git", *args),
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _output_ids(output: Any) -> tuple[int, ...]:
    values = getattr(output, "generated_token_ids", None)
    if values is None:
        raise GateError("generation output has no generated token IDs")
    result = tuple(int(token) for token in values)
    if not result:
        raise GateError("generation output is empty")
    return result


def _task_features(prompt_id: str, text: str) -> dict[str, bool]:
    """Fixed task features used only for paired strict non-inferiority."""

    value = str(text)
    japanese = bool(re.search(r"[\u3040-\u30ff\u3400-\u9fff]", value))
    ascii_word = bool(re.search(r"[A-Za-z]{3,}", value))
    features: dict[str, bool] = {"nonempty": bool(value.strip())}
    code_symbols = {
        "code_merge_intervals": "merge_intervals",
        "code_topological_sort": "topo_sort",
        "code_lru_cache": "LRUCache",
        "code_markdown_table": "markdown_table",
    }
    if prompt_id in code_symbols:
        stripped = value.replace("```python", "").replace("```", "")
        features.update(
            {
                "required_symbol": code_symbols[prompt_id] in value,
                "python_construct": "def " in value or "class " in value,
                "return_or_method": "return" in value or "self." in value,
            }
        )
        try:
            ast.parse(stripped)
            syntax_complete = True
        except SyntaxError:
            syntax_complete = False
        features["syntax_complete_at_d24"] = syntax_complete
    elif prompt_id == "general_en_plan":
        lower = value.lower()
        features.update(
            {
                "english": ascii_word,
                "migration_topic": any(word in lower for word in ("migrat", "postgres", "database")),
                "structured": any(mark in value for mark in ("-", "1.", "**")),
            }
        )
    elif prompt_id == "general_en_explain":
        lower = value.lower()
        features.update(
            {
                "english": ascii_word,
                "gpu_topic": any(word in lower for word in ("memory", "bandwidth", "gpu")),
                "technical": any(word in lower for word in ("weight", "token", "compute", "kernel")),
            }
        )
    elif prompt_id.startswith("general_ja"):
        features.update(
            {
                "japanese": japanese,
                "substantive": len(value.strip()) >= 16,
                "structured": any(mark in value for mark in ("・", "-", "1.", "**")),
            }
        )
    elif prompt_id == "mixed_ja_en_translate":
        features.update(
            {
                "english": ascii_word,
                "japanese": japanese,
                "bilingual": ascii_word and japanese,
            }
        )
    elif prompt_id == "mixed_ja_en_review":
        features.update(
            {
                "english": ascii_word,
                "mtp_preserved": "MTP" in value,
                "release_term_preserved": "推論" in value or "KV" in value,
            }
        )
    else:
        raise ValueError(f"unknown task prompt {prompt_id!r}")
    return features


def paired_task_verdict(
    prompt_id: str,
    strict_text: str,
    candidate_text: str,
) -> dict[str, Any]:
    strict = _task_features(prompt_id, strict_text)
    candidate = _task_features(prompt_id, candidate_text)
    strict_score = sum(bool(value) for value in strict.values())
    candidate_score = sum(bool(value) for value in candidate.values())
    regressions = [
        name for name, required in strict.items() if bool(required) and not bool(candidate.get(name))
    ]
    return {
        "prompt_id": prompt_id,
        "strict_features": strict,
        "candidate_features": candidate,
        "strict_score": strict_score,
        "candidate_score": candidate_score,
        "regressions": regressions,
        "passed": not regressions and candidate_score >= strict_score,
    }


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize zero rows")
    kls = np.asarray([float(row["kl"]) for row in rows], dtype=np.float64)
    top1 = np.asarray([bool(row["top1_equal"]) for row in rows], dtype=np.bool_)
    return {
        "rows": int(kls.size),
        "mean_kl": float(np.mean(kls)),
        "p95_kl": float(np.percentile(kls, 95.0)),
        "p99_kl": float(np.percentile(kls, 99.0)),
        "max_kl": float(np.max(kls)),
        "top1_agreement": float(np.mean(top1)),
        "top1_matches": int(np.count_nonzero(top1)),
        "top1_mismatches": int(kls.size - np.count_nonzero(top1)),
        "top5_overlap_mean": float(np.mean([float(row["top5_overlap"]) for row in rows])),
        "max_abs_logit_delta": float(max(float(row["max_abs_logit_delta"]) for row in rows)),
    }


def numerical_verdict(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    aggregate = summarize_rows(rows)
    scopes: dict[str, dict[str, Any]] = {}
    failures: list[str] = []
    for dimension in ("category", "shape", "transition"):
        groups: dict[str, Any] = {}
        for value in sorted({str(row[dimension]) for row in rows}):
            selected = [row for row in rows if str(row[dimension]) == value]
            summary = summarize_rows(selected)
            passed = bool(
                summary["mean_kl"] <= THRESHOLDS["mean_kl_max"]
                and summary["p95_kl"] <= THRESHOLDS["p95_kl_max"]
                and summary["p99_kl"] <= THRESHOLDS["p99_kl_max"]
                and summary["max_kl"] <= THRESHOLDS["max_kl_max"]
                and summary["top1_agreement"] >= THRESHOLDS["per_scope_top1_min"]
            )
            summary["passed"] = passed
            groups[value] = summary
            if not passed:
                failures.append(f"{dimension}:{value}")
        scopes[dimension] = groups
    checks = {
        "finite": bool(all(math.isfinite(float(row["kl"])) for row in rows)),
        "mean_kl": aggregate["mean_kl"] <= THRESHOLDS["mean_kl_max"],
        "p95_kl": aggregate["p95_kl"] <= THRESHOLDS["p95_kl_max"],
        "p99_kl": aggregate["p99_kl"] <= THRESHOLDS["p99_kl_max"],
        "max_kl": aggregate["max_kl"] <= THRESHOLDS["max_kl_max"],
        "top1": aggregate["top1_agreement"] >= THRESHOLDS["top1_min"],
        "per_scope": not failures,
    }
    return {
        "passed": all(checks.values()),
        "thresholds": dict(THRESHOLDS),
        "checks": checks,
        "aggregate": aggregate,
        "scopes": scopes,
        "scope_failures": failures,
    }


def _prefill(session: Any, prompt_tokens: Sequence[int]) -> int:
    session.reset()
    result = session.prefill(
        prompt_tokens,
        use_bulk=True,
        bulk_attention_mode="bulk",
        return_logits=False,
        capture_hidden_seed_fp32=True,
    )
    return int(result.token_id)


def _strict_prefix(
    session: Any,
    prompt_tokens: Sequence[int],
    output_index: int,
) -> int:
    current = _prefill(session, prompt_tokens)
    for index in range(int(output_index) - 1):
        result = session.step(
            current,
            position=len(prompt_tokens) + index,
            return_logits=False,
            capture_hidden_seed_fp32=True,
        )
        current = int(result.token_id)
    expected_position = len(prompt_tokens) + int(output_index) - 1
    if int(session.position) != expected_position:
        raise GateError(
            f"strict prefix cursor {session.position} != expected {expected_position}"
        )
    return current


def _build_teacher_and_metrics(
    session: Any,
    *,
    prompt: Mapping[str, Any],
    prompt_tokens: Sequence[int],
    teacher: Sequence[int],
    live_cycles: Sequence[LiveCycle],
    repeat_runs: int,
) -> tuple[list[TeacherCycle], list[dict[str, Any]], list[dict[str, Any]]]:
    if len(teacher) < 2 or not live_cycles:
        raise GateError("teacher metric capture requires output and provider drafts")
    schedule: list[TeacherCycle] = []
    row_metrics: list[dict[str, Any]] = []
    repeats: list[dict[str, Any]] = []
    for live_cycle in live_cycles:
        output_index = int(live_cycle.root_position) - len(prompt_tokens) + 1
        if output_index <= 0 or output_index >= len(teacher):
            raise GateError(
                f"live cycle position {live_cycle.root_position} is outside the strict trajectory"
            )
        root = _strict_prefix(session, prompt_tokens, output_index)
        remaining = len(teacher) - output_index
        drafts = tuple(int(token) for token in live_cycle.draft_tokens[: min(2, remaining)])
        if not drafts:
            raise GateError("teacher-aligned live cycle has no bounded candidate")
        inputs = (root, *drafts)
        strict_tokens, strict_logits, *_ = _probe_serial(
            session,
            list(inputs),
            capture_pre_output_norm_hidden=False,
            capture_layer_output_hidden=[],
        )
        strict_tokens_tuple = tuple(int(token) for token in strict_tokens)
        accepted = 0
        max_accepted = max(0, remaining - 1)
        while (
            accepted < min(len(drafts), max_accepted)
            and drafts[accepted] == strict_tokens_tuple[accepted]
        ):
            accepted += 1
        outputs = (*drafts[:accepted], strict_tokens_tuple[accepted])
        cycle = TeacherCycle(
            cycle=len(schedule),
            output_index=output_index,
            start_position=len(prompt_tokens) + output_index - 1,
            root_token=root,
            inputs=inputs,
            strict_target_tokens=strict_tokens_tuple,
            accepted=accepted,
            outputs=outputs,
        )
        candidate_logits_runs: list[np.ndarray] = []
        candidate_tokens_runs: list[tuple[int, ...]] = []
        candidate_hidden_finite: list[bool] = []
        for _ in range(repeat_runs):
            replay_root = _strict_prefix(session, prompt_tokens, output_index)
            if replay_root != root:
                raise GateError("candidate strict-teacher root diverged")
            candidate_tokens, candidate_logits, hidden_rows, *_ = _probe_bulk_or_native(
                session,
                list(inputs),
                mode="bulk",
                use_wmma_prefill=False,
                capture_linear_state_rows=True,
                capture_pre_output_norm_hidden=False,
                capture_layer_output_hidden=[],
                capture_layer_boundary_hidden=[],
            )
            candidate_logits_runs.append(
                np.ascontiguousarray(candidate_logits, dtype=np.float32)
            )
            candidate_tokens_runs.append(tuple(int(token) for token in candidate_tokens))
            candidate_hidden_finite.append(bool(np.isfinite(hidden_rows).all()))
        hashes = [hashlib.sha256(values.view(np.uint8)).hexdigest() for values in candidate_logits_runs]
        repeat_equal = len(set(hashes)) == 1 and len(set(candidate_tokens_runs)) == 1
        repeats.append(
            {
                "prompt_id": str(prompt["id"]),
                "cycle": cycle.cycle,
                "shape": cycle.shape,
                "candidate_logits_sha256": hashes,
                "candidate_tokens": [list(values) for values in candidate_tokens_runs],
                "hidden_finite": candidate_hidden_finite,
                "passed": repeat_equal and all(candidate_hidden_finite),
            }
        )
        labels = np.asarray(strict_tokens_tuple, dtype=np.int64)
        metrics = per_row_metrics(
            np.ascontiguousarray(strict_logits, dtype=np.float32),
            candidate_logits_runs[0],
            labels,
            top_k=5,
        )
        for row_index in range(len(inputs)):
            strict_row = strict_logits[row_index]
            top2 = np.partition(strict_row, -2)[-2:]
            row_metrics.append(
                {
                    "prompt_id": str(prompt["id"]),
                    "category": str(prompt["category"]),
                    "heldout": bool(prompt["heldout"]),
                    "cycle": cycle.cycle,
                    "row": row_index,
                    "shape": cycle.shape,
                    "transition": (
                        "prefill_to_verify" if cycle.cycle == 0 else "verify_to_verify"
                    ),
                    "position": cycle.start_position + row_index,
                    "strict_top1": strict_tokens_tuple[row_index],
                    "candidate_top1": candidate_tokens_runs[0][row_index],
                    "strict_margin": float(top2.max() - top2.min()),
                    "kl": float(metrics["kl_nats"][row_index]),
                    "top1_equal": bool(metrics["top1_equal"][row_index]),
                    "top5_overlap": float(metrics["topk_set_overlap"][row_index]),
                    "teacher_nll": float(metrics["teacher_nll_nats"][row_index]),
                    "strict_teacher_nll": float(
                        metrics["reference_teacher_nll_nats"][row_index]
                    ),
                    "delta_p": float(metrics["delta_p"][row_index]),
                    "max_abs_logit_delta": float(
                        metrics["max_abs_logit_delta"][row_index]
                    ),
                }
            )
        schedule.append(cycle)
    return schedule, row_metrics, repeats


def _replay_live_prefix(
    session: Any,
    prompt_tokens: Sequence[int],
    first_token: int,
    cycles: Sequence[LiveCycle],
) -> int:
    current = _prefill(session, prompt_tokens)
    if current != first_token:
        raise GateError("live graph/eager prefill root diverged")
    for cycle in cycles:
        if int(session.position) != cycle.root_position or current != cycle.root_token:
            raise GateError("live graph/eager prefix cursor diverged")
        result = session.verify_target_block(
            [cycle.root_token, *cycle.draft_tokens],
            bulk_attention_mode="bulk",
            use_wmma_prefill=False,
            capture_linear_state_rows=True,
            defer_linear_state_commit=True,
            record_stage_timings=False,
        )
        tokens = tuple(int(token) for token in result.token_ids)
        if tokens != cycle.target_tokens:
            raise GateError(
                f"live eager replay differs from graph at cycle {cycle.cycle}: "
                f"{tokens!r} != {cycle.target_tokens!r}"
            )
        session._commit_verify_linear_state_row(
            cycle.accepted,
            position=cycle.root_position + cycle.accepted + 1,
        )
        current = int(cycle.output_tokens[-1])
    return current


def _capture_graph_eager(
    session: Any,
    *,
    prompt: Mapping[str, Any],
    prompt_tokens: Sequence[int],
    output_ids: Sequence[int],
    cycles: Sequence[LiveCycle],
) -> dict[str, Any]:
    rows = 0
    for cycle_index, cycle in enumerate(cycles):
        current = _replay_live_prefix(
            session,
            prompt_tokens,
            int(output_ids[0]),
            cycles[:cycle_index],
        )
        if current != cycle.root_token:
            raise GateError("graph/eager current root mismatch")
        tokens, logits, hidden, *_ = _probe_bulk_or_native(
            session,
            [cycle.root_token, *cycle.draft_tokens],
            mode="bulk",
            use_wmma_prefill=False,
            capture_linear_state_rows=True,
            capture_pre_output_norm_hidden=False,
            capture_layer_output_hidden=[],
            capture_layer_boundary_hidden=[],
        )
        eager = tuple(int(token) for token in tokens)
        if eager != cycle.target_tokens:
            raise GateError(
                f"graph/eager decision mismatch for {prompt['id']} cycle {cycle_index}"
            )
        if not np.isfinite(logits).all() or not np.isfinite(hidden).all():
            raise GateError("graph/eager diagnostic produced non-finite values")
        rows += len(eager)
    return {
        "prompt_id": str(prompt["id"]),
        "cycles": len(cycles),
        "rows": rows,
        "passed": True,
    }


def _live_cycle_from_result(result: Any, kwargs: Mapping[str, Any], index: int) -> LiveCycle:
    target = result.target_result
    cycle = LiveCycle(
        cycle=index,
        root_token=int(kwargs["root_token"]),
        root_position=int(kwargs["root_position"]),
        remaining_decode=int(kwargs["remaining_decode"]),
        draft_tokens=tuple(int(token) for token in result.draft_token_ids),
        target_tokens=tuple(int(token) for token in target.target_top1),
        output_tokens=tuple(int(token) for token in result.output_token_ids),
        accepted=int(result.accepted_draft_tokens),
        graph=bool(getattr(target, "device_accept_commit", False)),
    )
    if not cycle.target_tokens or len(cycle.target_tokens) != len(cycle.draft_tokens) + 1:
        raise GateError("native graph did not return bounded target decisions")
    if len(cycle.output_tokens) != cycle.accepted + 1:
        raise GateError("native graph visible output accounting is malformed")
    return cycle


def _capture_live_production(
    *,
    model: Path,
    prompts: Sequence[Mapping[str, Any]],
    max_tokens: int,
    budget: int,
    max_sequence_length: int,
    repeat_runs: int,
    tokenizer: Qwen35GGUFTokenizer,
) -> tuple[
    dict[str, Any],
    dict[str, list[PromptResult]],
    dict[str, list[list[LiveCycle]]],
    dict[str, list[LiveCycle]],
]:
    context: dict[str, Any] = {"cycles": None}
    original = Qwen35GGUFResidentSession.run_native_spec_mtp_cycle

    def wrapped(session: Any, *args: Any, **kwargs: Any) -> Any:
        result = original(session, *args, **kwargs)
        target = context.get("cycles")
        if target is None:
            raise GateError("native MTP cycle executed outside capture ownership")
        target.append(_live_cycle_from_result(result, kwargs, len(target)))
        return result

    Qwen35GGUFResidentSession.run_native_spec_mtp_cycle = wrapped
    outputs: dict[str, list[PromptResult]] = defaultdict(list)
    traces: dict[str, list[list[LiveCycle]]] = defaultdict(list)
    isolation: dict[str, list[LiveCycle]] = {}
    llm = LLM(
        str(model),
        backend="hip_gfx1100",
        max_active_requests=1,
        max_sequence_length=max_sequence_length,
        speculative_candidate_budget=budget,
        execution_profile="production",
    )
    try:
        llm.prepare(max_sequence_length=max_sequence_length)
        sampling = SamplingParams(max_tokens=max_tokens, temperature=0.0, top_p=1.0)
        for _ in range(repeat_runs):
            for prompt in prompts:
                prompt_id = str(prompt["id"])
                cycles: list[LiveCycle] = []
                context["cycles"] = cycles
                output = llm.generate_speculative_mtp_detailed(
                    (str(prompt["rendered_prompt"]),),
                    sampling,
                )[0]
                ids = _output_ids(output)
                outputs[prompt_id].append(
                    PromptResult(
                        prompt_id=prompt_id,
                        category=str(prompt["category"]),
                        heldout=bool(prompt["heldout"]),
                        token_ids=ids,
                        text=tokenizer.decode(ids),
                    )
                )
                traces[prompt_id].append(cycles)
        for prompt in reversed(prompts):
            prompt_id = str(prompt["id"])
            cycles = []
            context["cycles"] = cycles
            output = llm.generate_speculative_mtp_detailed(
                (str(prompt["rendered_prompt"]),),
                sampling,
            )[0]
            if _output_ids(output) != outputs[prompt_id][0].token_ids:
                raise GateError(f"reverse-order isolation output mismatch for {prompt_id}")
            isolation[prompt_id] = cycles
        context["cycles"] = None
        service = llm._get_text_generator()
        snapshot = service.live_loop_snapshot()
        profile = {
            "execution_profile": getattr(llm, "execution_profile", None),
            "manifest_sha256": getattr(llm, "execution_profile_manifest_sha256", None),
            "strict_manifest_sha256": getattr(
                llm, "execution_profile_strict_manifest_sha256", None
            ),
            "manifest": getattr(llm, "execution_profile_manifest", None),
            "snapshot": snapshot,
        }
    finally:
        context["cycles"] = None
        Qwen35GGUFResidentSession.run_native_spec_mtp_cycle = original
        llm.close()
    return profile, dict(outputs), dict(traces), isolation


def _capture_strict_outputs(
    *,
    model: Path,
    prompts: Sequence[Mapping[str, Any]],
    max_tokens: int,
    max_sequence_length: int,
    tokenizer: Qwen35GGUFTokenizer,
) -> tuple[dict[str, Any], dict[str, PromptResult]]:
    llm = LLM(
        str(model),
        backend="hip_gfx1100",
        max_active_requests=1,
        max_sequence_length=max_sequence_length,
        speculative_candidate_budget=2,
        execution_profile="strict",
    )
    outputs: dict[str, PromptResult] = {}
    try:
        llm.prepare(max_sequence_length=max_sequence_length)
        sampling = SamplingParams(max_tokens=max_tokens, temperature=0.0, top_p=1.0)
        for prompt in prompts:
            prompt_id = str(prompt["id"])
            output = llm.generate_detailed((str(prompt["rendered_prompt"]),), sampling)[0]
            ids = _output_ids(output)
            outputs[prompt_id] = PromptResult(
                prompt_id=prompt_id,
                category=str(prompt["category"]),
                heldout=bool(prompt["heldout"]),
                token_ids=ids,
                text=tokenizer.decode(ids),
            )
        profile = {
            "execution_profile": getattr(llm, "execution_profile", None),
            "manifest_sha256": getattr(llm, "execution_profile_manifest_sha256", None),
            "strict_manifest_sha256": getattr(
                llm, "execution_profile_strict_manifest_sha256", None
            ),
            "manifest": getattr(llm, "execution_profile_manifest", None),
        }
    finally:
        llm.close()
    return profile, outputs


def _trace_checks(
    prompts: Sequence[Mapping[str, Any]],
    production_outputs: Mapping[str, Sequence[PromptResult]],
    traces: Mapping[str, Sequence[Sequence[LiveCycle]]],
    isolation: Mapping[str, Sequence[LiveCycle]],
    *,
    max_tokens: int,
) -> dict[str, Any]:
    prompt_rows: list[dict[str, Any]] = []
    for prompt in prompts:
        prompt_id = str(prompt["id"])
        output_rows = production_outputs[prompt_id]
        trace_rows = traces[prompt_id]
        output_repeat = len({row.token_ids for row in output_rows}) == 1
        trace_payloads = [
            [cycle.stable_payload() for cycle in run]
            for run in trace_rows
        ]
        trace_repeat = len({_sha256_json(value) for value in trace_payloads}) == 1
        isolation_payload = [cycle.stable_payload() for cycle in isolation[prompt_id]]
        isolation_equal = isolation_payload == trace_payloads[0]
        first_output = output_rows[0].token_ids
        reconstructed = [int(first_output[0])]
        expected_position = None
        bounded = True
        graph = True
        for cycle in trace_rows[0]:
            if expected_position is None:
                expected_position = cycle.root_position
            if cycle.root_position != expected_position or cycle.root_token != reconstructed[-1]:
                bounded = False
            reconstructed.extend(cycle.output_tokens)
            expected_position += len(cycle.output_tokens)
            bounded = bool(
                bounded
                and len(cycle.output_tokens) == cycle.accepted + 1
                and len(cycle.draft_tokens) in {1, 2}
                and cycle.accepted <= len(cycle.draft_tokens)
                and len(cycle.output_tokens) <= cycle.remaining_decode
            )
            graph = graph and cycle.graph
        reconstructed_equal = tuple(reconstructed) == first_output
        terminal = trace_rows[0][-1]
        terminal_exact = bool(
            terminal.remaining_decode == 1
            and len(terminal.draft_tokens) == 1
            and terminal.accepted == 0
            and len(terminal.output_tokens) == 1
        )
        passed = bool(
            output_repeat
            and trace_repeat
            and isolation_equal
            and reconstructed_equal
            and bounded
            and graph
            and terminal_exact
            and len(first_output) == max_tokens
        )
        prompt_rows.append(
            {
                "prompt_id": prompt_id,
                "output_repeat": output_repeat,
                "trace_repeat": trace_repeat,
                "reverse_order_isolation": isolation_equal,
                "reconstructed_output": reconstructed_equal,
                "bounded_control": bounded,
                "all_graph": graph,
                "terminal_zero_accept": terminal_exact,
                "cycles": len(trace_rows[0]),
                "trace_sha256": _sha256_json(trace_payloads[0]),
                "passed": passed,
            }
        )
    return {"passed": all(row["passed"] for row in prompt_rows), "prompts": prompt_rows}


def run(args: argparse.Namespace) -> dict[str, Any]:
    model = args.model.resolve()
    prompts = load_prompt_suite(args.prompts.resolve())
    if args.scope != "full":
        prompts = tuple(row for row in prompts if bool(row["heldout"]) == (args.scope == "heldout"))
    if len(prompts) < 4:
        raise GateError("production gate requires broad category prompt coverage")
    if _git("status", "--porcelain", "--untracked-files=no"):
        raise GateError("production gate requires a tracked-clean worktree")
    os.environ["HIPENGINE_GGUF_VERIFY_LM_HEAD_Q6_TOP1_DP4A"] = "0"
    os.environ["HIPENGINE_GGUF_FP16_RECURRENT_STATE"] = "0"
    os.environ["HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN"] = "1"
    os.environ["HIPENGINE_GGUF_MTP_CANDIDATE_BUDGET"] = str(args.budget)

    info = scan_gguf(model)
    tokenizer = Qwen35GGUFTokenizer.from_gguf_info(info)
    strict_profile, strict_outputs = _capture_strict_outputs(
        model=model,
        prompts=prompts,
        max_tokens=args.max_tokens,
        max_sequence_length=args.max_sequence_length,
        tokenizer=tokenizer,
    )
    production_profile, production_outputs, traces, isolation = _capture_live_production(
        model=model,
        prompts=prompts,
        max_tokens=args.max_tokens,
        budget=args.budget,
        max_sequence_length=args.max_sequence_length,
        repeat_runs=args.repeat_runs,
        tokenizer=tokenizer,
    )
    controls = _trace_checks(
        prompts,
        production_outputs,
        traces,
        isolation,
        max_tokens=args.max_tokens,
    )

    row_metrics: list[dict[str, Any]] = []
    repeat_rows: list[dict[str, Any]] = []
    schedules: dict[str, list[TeacherCycle]] = {}
    graph_eager: list[dict[str, Any]] = []
    session = Qwen35GGUFResidentSession(
        model,
        backend="hip_gfx1100",
        compiler_version=(
            None
            if args.compiler_version_file is None
            else args.compiler_version_file.read_text(encoding="utf-8")
        ),
        require_cached_build=bool(args.require_cached_build),
        max_sequence_length=args.max_sequence_length,
        use_wmma_prefill=False,
        use_gemv_decode=True,
        prefill_config=PrefillConfig(),
    )
    try:
        for prompt in prompts:
            prompt_id = str(prompt["id"])
            prompt_tokens = tokenizer.encode(str(prompt["rendered_prompt"]))
            strict = strict_outputs[prompt_id].token_ids
            live_cycles = list(traces[prompt_id][0])
            schedule, rows, repeats = _build_teacher_and_metrics(
                session,
                prompt=prompt,
                prompt_tokens=prompt_tokens,
                teacher=strict,
                live_cycles=live_cycles,
                repeat_runs=args.repeat_runs,
            )
            schedules[prompt_id] = schedule
            row_metrics.extend(rows)
            repeat_rows.extend(repeats)
            graph_eager.append(
                _capture_graph_eager(
                    session,
                    prompt=prompt,
                    prompt_tokens=prompt_tokens,
                    output_ids=production_outputs[prompt_id][0].token_ids,
                    cycles=live_cycles,
                )
            )
    finally:
        session.close()

    numerical = numerical_verdict(row_metrics)
    repeat_gate = {
        "passed": all(row["passed"] for row in repeat_rows),
        "cycles": len(repeat_rows),
        "failures": [
            {"prompt_id": row["prompt_id"], "cycle": row["cycle"]}
            for row in repeat_rows
            if not row["passed"]
        ],
    }
    graph_eager_gate = {
        "passed": all(row["passed"] for row in graph_eager),
        "prompts": graph_eager,
    }
    task_rows: list[dict[str, Any]] = []
    for prompt in prompts:
        prompt_id = str(prompt["id"])
        candidate_texts = {row.text for row in production_outputs[prompt_id]}
        if len(candidate_texts) != 1:
            raise GateError(f"production task text is not repeatable for {prompt_id}")
        task_rows.append(
            paired_task_verdict(
                prompt_id,
                strict_outputs[prompt_id].text,
                next(iter(candidate_texts)),
            )
        )
    task_gate = {"passed": all(row["passed"] for row in task_rows), "prompts": task_rows}
    profile_checks = {
        "strict_profile": strict_profile.get("execution_profile") == "strict",
        "production_profile": production_profile.get("execution_profile") == "production",
        "strict_manifest": bool(strict_profile.get("manifest_sha256")),
        "production_manifest": bool(production_profile.get("manifest_sha256")),
        "registered_strict_fallback": (
            production_profile.get("strict_manifest_sha256")
            == strict_profile.get("manifest_sha256")
        ),
    }
    snapshot = production_profile.pop("snapshot")
    engine_service = snapshot.get("engine_service", {})
    runner_routes = snapshot.get("runner", {}).get("routes", {})
    lifecycle_checks = {
        "active_children_zero": int(engine_service.get("active_children", -1)) == 0,
        "command_queue_empty": int(engine_service.get("command_queue_depth", -1)) == 0,
        "sole_driver": bool(engine_service.get("sole_driver")),
        "legacy_fallback_zero": int(
            engine_service.get("speculative_routes", {}).get("legacy_prelaunch_fallback", -1)
        )
        == 0,
        "runner_recent_failures_zero": all(
            not row.get("specdec2_mtp2_failure_reasons")
            for row in runner_routes.get("recent_completed", [])
        ),
    }
    checks = {
        "exact_control": bool(controls["passed"]),
        "numerical": bool(numerical["passed"]),
        "repeat_determinism": bool(repeat_gate["passed"]),
        "graph_eager": bool(graph_eager_gate["passed"]),
        "task_noninferiority": bool(task_gate["passed"]),
        "profiles": all(profile_checks.values()),
        "lifecycle": all(lifecycle_checks.values()),
        "full_suite": args.scope == "full" and len(prompts) == 10,
    }
    review_rows = [
        row
        for row in row_metrics
        if float(row["kl"]) > 2.0e-2 or not bool(row["top1_equal"])
    ]
    provenance = collect_artifact_provenance(
        repo_root=REPO_ROOT,
        configured_backend="hip_gfx1100",
        resolved_backend="hip_gfx1100",
        detected_arches=("gfx1100",),
        target_arch="gfx1100",
        model_path=model,
        quant="UD-Q4_K_M",
        kv_dtype="bf16",
        command=sys.argv,
        environment={
            "HIP_VISIBLE_DEVICES": os.environ.get("HIP_VISIBLE_DEVICES"),
            "ROCR_VISIBLE_DEVICES": os.environ.get("ROCR_VISIBLE_DEVICES"),
            "GPU_MAX_HW_QUEUES": os.environ.get("GPU_MAX_HW_QUEUES"),
            "HIPENGINE_HIP_ARCH": os.environ.get("HIPENGINE_HIP_ARCH"),
            "HIPENGINE_GGUF_FP16_RECURRENT_STATE": os.environ.get(
                "HIPENGINE_GGUF_FP16_RECURRENT_STATE"
            ),
        },
        timing_protocol="correctness-only strict-teacher full-logit packet",
        warmups=0,
        repetitions=args.repeat_runs,
        host_name=platform.node(),
    )
    model_sha = _file_sha256(model) if args.verify_model_sha256 else MODEL_SHA256
    model_sha_ok = model_sha == MODEL_SHA256
    checks["model_sha256"] = model_sha_ok
    return {
        "schema": 1,
        "kind": KIND,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if all(checks.values()) else "failed",
        "performance_claim": False,
        "provenance": provenance,
        "model": {
            "path": str(model),
            "size_bytes": model.stat().st_size,
            "sha256": model_sha,
            "expected_sha256": MODEL_SHA256,
            "sha256_passed": model_sha_ok,
            "quant": "UD-Q4_K_M",
            "kv": "bf16",
        },
        "workload": {
            "prompt_file": str(args.prompts.resolve()),
            "prompt_file_sha256": _file_sha256(args.prompts.resolve()),
            "prompt_ids": [str(prompt["id"]) for prompt in prompts],
            "scope": args.scope,
            "candidate_budget": args.budget,
            "max_tokens": args.max_tokens,
            "repeat_runs": args.repeat_runs,
            "concurrency": 1,
            "sampling": "raw greedy temperature=0 top_p=1",
        },
        "profiles": {
            "strict": strict_profile,
            "production": production_profile,
            "checks": profile_checks,
        },
        "checks": checks,
        "exact_control": controls,
        "numerical": numerical,
        "repeat_determinism": repeat_gate,
        "graph_eager": graph_eager_gate,
        "tasks": task_gate,
        "lifecycle": {"checks": lifecycle_checks, "engine_service": engine_service},
        "bf16_relative": {
            "applicable": False,
            "reason": (
                "The frozen artifact is selected-quant GGUF Q4_K_M; no aligned "
                "same-model BF16/full-precision weight artifact is available on the host."
            ),
        },
        "review_rows": review_rows,
        "row_metrics": row_metrics,
        "capture_hashes": {
            "strict_outputs": _sha256_json(
                {key: list(value.token_ids) for key, value in strict_outputs.items()}
            ),
            "production_outputs": _sha256_json(
                {
                    key: [list(row.token_ids) for row in values]
                    for key, values in production_outputs.items()
                }
            ),
            "teacher_schedules": _sha256_json(
                {
                    key: [
                        {
                            "cycle": row.cycle,
                            "start": row.start_position,
                            "inputs": list(row.inputs),
                            "accepted": row.accepted,
                            "outputs": list(row.outputs),
                        }
                        for row in values
                    ]
                    for key, values in schedules.items()
                }
            ),
            "row_metrics": _sha256_json(row_metrics),
        },
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        default=REPO_ROOT / "benchmarks/prompts/mtpbench-code-general-ja.jsonl",
    )
    parser.add_argument("--scope", choices=("full", "train", "heldout"), default="full")
    parser.add_argument("--budget", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=24)
    parser.add_argument("--repeat-runs", type=int, default=3)
    parser.add_argument("--max-sequence-length", type=int, default=1024)
    parser.add_argument("--compiler-version-file", type=Path, default=None)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--verify-model-sha256", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fail-on-fail", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.budget != 2:
        raise SystemExit("this production key requires --budget 2")
    if args.max_tokens != 24:
        raise SystemExit("this production key requires --max-tokens 24")
    if args.repeat_runs < 3:
        raise SystemExit("production repeat gate requires at least three runs")
    artifact = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": artifact["status"],
                "output": str(args.output.resolve()),
                "rows": artifact["numerical"]["aggregate"]["rows"],
                "mean_kl": artifact["numerical"]["aggregate"]["mean_kl"],
                "p95_kl": artifact["numerical"]["aggregate"]["p95_kl"],
                "p99_kl": artifact["numerical"]["aggregate"]["p99_kl"],
                "max_kl": artifact["numerical"]["aggregate"]["max_kl"],
                "top1": artifact["numerical"]["aggregate"]["top1_agreement"],
                "checks": artifact["checks"],
            },
            sort_keys=True,
        )
    )
    return 0 if artifact["status"] == "passed" or not args.fail_on_fail else 2


if __name__ == "__main__":
    raise SystemExit(main())
