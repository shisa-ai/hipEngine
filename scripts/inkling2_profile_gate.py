#!/usr/bin/env python3
"""Gate one Inkling 2 GGUF execution profile against the strict same-quant fallback.

Strict and candidate arithmetic run in separate model instances. The strict arm
supplies the token trajectory; every candidate row is teacher-forced onto it and
evaluated over the full vocabulary with the router-routing-scoped envelope of
docs/EXECUTION-PROFILES.md section 2.11. Three candidate repeats must be
bit-stable in logits and in resident state.

Each arm records what the session was asked at every call (token, position,
width, transaction phase, graph bucket) and the live attention and convolution
state after the trajectory. The packet compares those records with controls
derived from the schedule alone, so a candidate that drifted off the teacher's
chain, advanced out of order, or replayed a graph where the schedule says eager
fails regardless of its logits. A fresh process then replays the prompts in
reverse order to show that a request's logits do not depend on unrelated
requests, and the public generate route is checked against the captured route.

Performance is measured separately so full-logit and state downloads cannot
contaminate wall claims.
"""

from __future__ import annotations

import argparse
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.benchmark.execution_profiles import (  # noqa: E402
    ControlRecord,
    EvaluationThresholds,
    compare_control_records,
)
from hipengine.benchmark.provenance import collect_artifact_provenance  # noqa: E402
from hipengine.benchmark.quant_quality_metrics import per_row_metrics  # noqa: E402
from scripts.execution_profile_gdn_calibration import (  # noqa: E402
    PromptCalibrationCapture,
    build_candidate_quality,
)
from scripts.gguf_mtp_category_bench import prompt_sha256  # noqa: E402
from scripts.inkling2_gguf_task_check import _CASES, _PREFIX, _SUFFIX, _render  # noqa: E402

KIND = "hipengine_execution_profile_inkling2_gguf_gate"
SCHEMA_VERSION = 1
DEFAULT_SUITE = REPO_ROOT / "benchmarks/prompts/inkling2-mini-profile-gate.jsonl"
STRICT_PROFILE = "strict"
ROUTE_ENV_KEYS = (
    "HIPENGINE_INKLING2_DEVICE_ROUTES",
    "HIPENGINE_INKLING2_RESIDENT_DECODE",
    "HIPENGINE_INKLING2_DECODE_GRAPH",
)
STRICT_ENVIRONMENT = {name: "0" for name in ROUTE_ENV_KEYS}

# docs/EXECUTION-PROFILES.md section 2.11: robust statistics plus a bounded
# flip-damage budget, binding globally and per category.
MEDIAN_KL_MAX = 1.0e-3
P90_KL_MAX = 2.0e-2
DAMAGE_LINE_KL = 5.0e-2
DAMAGE_RATE_GLOBAL_MAX = 0.03
DAMAGE_RATE_CATEGORY_MAX = 0.08
TOP1_GLOBAL_MIN = 0.96
TOP1_CATEGORY_MIN = 0.94

WORK_PREFILL = "PREFILL"
WORK_DECODE = "DECODE"
BUCKET_EAGER = "c1_eager"
BUCKET_CAPTURE = "c1_graph_capture"
BUCKET_REPLAY = "c1_graph_replay"
PHASE_COMMITTED = "committed"
PHASE_PROVISIONAL = "provisional"


class GateError(RuntimeError):
    """Raised when the Inkling profile packet cannot be evaluated honestly."""


@dataclass(frozen=True, slots=True)
class Candidate:
    name: str
    profile: str
    classification: str
    mechanism: str
    strict_fallback: str
    environment: Mapping[str, str]
    graph: bool = False


CANDIDATES = {
    "fast": Candidate(
        name="fast",
        profile="fast",
        classification="T2",
        mechanism=(
            "Q8-dot NVFP4 expert projection and reduced-precision dense, controller and "
            "attention routes selected by the fast plan"
        ),
        strict_fallback="strict selected-expert F32 kernels and ordered mix",
        environment={name: "0" for name in ROUTE_ENV_KEYS},
    ),
    "fast_resident": Candidate(
        name="fast_resident",
        profile="fast",
        classification="T2",
        mechanism=(
            "one device-resident decode visit per token: the arena, router logits and "
            "expert chain of a step stay on the device between launches"
        ),
        strict_fallback="strict selected-expert F32 kernels and ordered mix",
        environment={
            "HIPENGINE_INKLING2_DEVICE_ROUTES": "1",
            "HIPENGINE_INKLING2_RESIDENT_DECODE": "1",
            "HIPENGINE_INKLING2_DECODE_GRAPH": "0",
        },
    ),
    "fast_graph": Candidate(
        name="fast_graph",
        profile="fast",
        classification="T2",
        mechanism=(
            "the resident decode visit recorded once per session and replayed as a whole-token "
            "graph, with the position carried by a device counter"
        ),
        strict_fallback="strict selected-expert F32 kernels and ordered mix",
        environment={
            "HIPENGINE_INKLING2_DEVICE_ROUTES": "1",
            "HIPENGINE_INKLING2_RESIDENT_DECODE": "1",
            "HIPENGINE_INKLING2_DECODE_GRAPH": "1",
        },
        graph=True,
    ),
}
CANDIDATE_NAMES = tuple(CANDIDATES)


def section_211_verdict(
    kl: Sequence[float],
    top1_equal: Sequence[bool],
    categories: Sequence[str],
) -> dict[str, Any]:
    """Section 2.11 binding verdict: robust statistics plus the damage budget.

    ``kl`` and ``top1_equal`` are the per-row values over the complete row set,
    so the damage rate reuses the numbers the median and p90 come from.
    """

    values = np.asarray(kl, dtype=np.float64)
    agreement = np.asarray(top1_equal, dtype=bool)
    labels = np.asarray([str(value) for value in categories])
    if values.ndim != 1 or values.size == 0 or agreement.shape != values.shape or labels.shape != values.shape:
        raise GateError("section 2.11 verdict needs aligned non-empty per-row KL, top-1 and category arrays")
    if not np.isfinite(values).all() or (values < 0).any():
        raise GateError("section 2.11 verdict needs finite non-negative KL values")

    def scope(mask: np.ndarray, *, category: bool) -> dict[str, Any]:
        scoped = values[mask]
        top1_min = TOP1_CATEGORY_MIN if category else TOP1_GLOBAL_MIN
        damage_max = DAMAGE_RATE_CATEGORY_MAX if category else DAMAGE_RATE_GLOBAL_MAX
        row = {
            "rows": int(scoped.size),
            "kl_median": float(np.median(scoped)),
            "kl_p90": float(np.percentile(scoped, 90)),
            "flip_damage_rate": float((scoped > DAMAGE_LINE_KL).mean()),
            "damage_line_kl": DAMAGE_LINE_KL,
            "top1_agreement": float(agreement[mask].mean()),
        }
        failed = []
        if row["kl_median"] > MEDIAN_KL_MAX:
            failed.append("kl_median")
        if row["kl_p90"] > P90_KL_MAX:
            failed.append("kl_p90")
        if row["flip_damage_rate"] > damage_max:
            failed.append("flip_damage_rate")
        if row["top1_agreement"] < top1_min:
            failed.append("top1_rate")
        return {
            **row,
            "thresholds": {
                "kl_median": MEDIAN_KL_MAX,
                "kl_p90": P90_KL_MAX,
                "flip_damage_rate": damage_max,
                "top1_rate": top1_min,
            },
            "failed": failed,
            "passed": not failed,
        }

    overall = scope(np.ones(values.size, dtype=bool), category=False)
    by_category = {
        str(label): scope(labels == label, category=True) for label in sorted(set(labels.tolist()))
    }
    failed_categories = [label for label, result in by_category.items() if not result["passed"]]
    return {
        "global": overall,
        "by_category": by_category,
        "failed_categories": failed_categories,
        "passed": bool(overall["passed"] and not failed_categories),
    }


def load_suite(path: Path, *, limit: int | None = None) -> list[dict[str, Any]]:
    """Load the frozen prompt suite: id, category, text, template and row count."""

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        raw = json.loads(text)
        if not isinstance(raw, dict):
            raise GateError(f"{path}:{line_no}: prompt row must be an object")
        row_id = raw.get("id")
        category = raw.get("category")
        prompt = raw.get("prompt")
        template = raw.get("template", "chat")
        steps = raw.get("decode_steps")
        if not isinstance(row_id, str) or not row_id.strip():
            raise GateError(f"{path}:{line_no}: prompt id must be a non-empty string")
        if row_id in seen:
            raise GateError(f"{path}:{line_no}: duplicate prompt id {row_id!r}")
        if not isinstance(category, str) or not category.strip():
            raise GateError(f"{path}:{line_no}: prompt category must be a non-empty string")
        if not isinstance(prompt, str) or not prompt.strip():
            raise GateError(f"{path}:{line_no}: prompt text must be a non-empty string")
        if template not in {"chat", "raw"}:
            raise GateError(f"{path}:{line_no}: template must be chat or raw")
        if type(steps) is not int or steps < 2:
            raise GateError(f"{path}:{line_no}: decode_steps must be an integer of at least 2")
        expected = raw.get("expected")
        if expected is not None and not isinstance(expected, str):
            raise GateError(f"{path}:{line_no}: expected answer must be a string")
        seen.add(row_id)
        rows.append(
            {
                "id": row_id,
                "category": category,
                "prompt": prompt,
                "template": template,
                "decode_steps": steps,
                "expected": expected,
                "requires_saturated_window": bool(raw.get("requires_saturated_window", False)),
                "crosses_window_boundary": bool(raw.get("crosses_window_boundary", False)),
            }
        )
    if limit is not None:
        rows = rows[: max(0, int(limit))]
    if not rows:
        raise GateError(f"prompt suite is empty: {path}")
    return rows


def render_prompt(row: Mapping[str, Any]) -> str:
    """Apply the chat template unless the row asks for raw continuation text."""

    if row["template"] == "raw":
        return str(row["prompt"])
    if row.get("expected") is not None:
        return _render(str(row["category"]), str(row["prompt"]))
    return _PREFIX + str(row["prompt"]) + _SUFFIX


def validate_coverage(
    requests: Sequence[Mapping[str, Any]],
    *,
    window: int,
) -> dict[str, Any]:
    """Require the suite to exercise the attention window on both sides of its limit.

    A prompt at or beyond the window starts with it full; a prompt marked as
    crossing it must fill it during the recorded rows.
    """

    if type(window) is not int or window <= 0:
        raise GateError("the model must declare a positive sliding_window_size")
    saturated = crossed = 0
    for request in requests:
        length = len(request["prompt_tokens"])
        rows = int(request["decode_steps"])
        if request["requires_saturated_window"]:
            if length < window:
                raise GateError(
                    f"{request['id']}: needs at least {window} prompt tokens, has {length}"
                )
            saturated += 1
        if request["crosses_window_boundary"]:
            if not length < window <= length + rows - 1:
                raise GateError(
                    f"{request['id']}: {length} prompt tokens and {rows} rows do not cross "
                    f"the {window}-token window"
                )
            crossed += 1
    return {
        "window_tokens": window,
        "saturated_window_prompts": saturated,
        "window_crossing_prompts": crossed,
        "prompts": len(requests),
    }


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


ACTIVE_MASK_HASH = _hash("active-mask:[true]")
MASK_MANIFEST_HASH = _hash("mask-manifest:inkling2-single-active-row-v1")
UNOBSERVED_ROUTES_HASH = _hash("routes:not-selected-on-host")


def _control(
    scenario_id: str,
    step: int,
    request_id: str,
    *,
    work_class: str,
    physical_width: int,
    execution_row: int,
    input_token_id: int,
    position: int,
    publication_ordinal: int,
    accepted_token_count: int,
    cap: int,
    route_decision_hash: str,
    route_top_k: int,
    phase: str,
    state_finite: bool,
    route_finite: bool,
    bucket: str,
) -> ControlRecord:
    live = min(position + 1, cap)
    return ControlRecord(
        scenario_id=scenario_id,
        scenario_step=step,
        work_class=work_class,
        request_id=request_id,
        physical_slot=0,
        execution_row=execution_row,
        physical_width=physical_width,
        input_token_id=input_token_id,
        position=position,
        context_length=position + 1,
        active=True,
        active_mask_hash=ACTIVE_MASK_HASH,
        mask_manifest_hash=MASK_MANIFEST_HASH,
        publication_ordinal=publication_ordinal,
        transaction_id=f"{request_id}:{work_class.lower()}:{position}",
        transaction_phase=phase,
        accepted_token_count=accepted_token_count,
        route_decision_hash=route_decision_hash,
        route_scatter_owner_hash=_hash(f"route-scatter-owner:{request_id}"),
        route_owner_request_id=request_id,
        route_top_k=route_top_k,
        kv_base_offset=0,
        kv_live_count=live,
        kv_token_position=position % cap,
        kv_evict=position >= cap,
        kv_values_finite=state_finite,
        kv_append_ordinal=position,
        state_owner_request_id=request_id,
        state_update_ordinal=position,
        state_values_finite=state_finite,
        rng_owner_request_id=request_id,
        rng_seed=0,
        rng_counter=0,
        route_values_finite=route_finite,
        graph_bucket=bucket,
    )


def _graph_bucket(model: Any, before: tuple[int, int]) -> str:
    """Name the graph route a call took, read from the model's own replay and capture counters."""

    captures, replays = int(model._graph_captures), int(model._graph_replays)
    if replays > before[1]:
        return BUCKET_REPLAY
    if captures > before[0]:
        return BUCKET_CAPTURE
    return BUCKET_EAGER


class _RouteRecorder:
    """Hash the expert ids each host-side routing call selects, one hash per session call.

    A visit whose router runs on the device makes no host call; such a call
    hashes to the declared unobserved marker instead of an empty digest.
    """

    def __init__(self, model: Any):
        self.model = model
        self.routes: list[tuple[int, tuple[int, ...]]] = []
        self._original = None
        self._owned = False

    def __enter__(self) -> "_RouteRecorder":
        self._owned = "_select_routes" in vars(self.model)
        self._original = self.model._select_routes
        self.model._select_routes = self._record
        return self

    def _record(self, visit: Any, logits: Any) -> Any:
        result = self._original(visit, logits)
        ids = np.asarray(result[0]).reshape(-1)
        self.routes.append((int(visit.weight_layer), tuple(int(value) for value in ids)))
        return result

    def take(self) -> str:
        digest = hashlib.sha256()
        for layer, ids in self.routes:
            digest.update(f"{layer}:{','.join(map(str, ids))};".encode("ascii"))
        selected = bool(self.routes)
        self.routes = []
        return digest.hexdigest() if selected else UNOBSERVED_ROUTES_HASH

    def __exit__(self, *_exc: object) -> None:
        if self._owned:
            self.model._select_routes = self._original
        else:
            del self.model._select_routes


@dataclass(frozen=True, slots=True)
class Call:
    """What one session call was asked and what the session reported around it."""

    kind: str
    tokens: tuple[int, ...]
    attention_positions: tuple[tuple[int, int], ...]
    committed: bool
    bucket: str
    route_hash: str
    finite: bool


@dataclass(frozen=True, slots=True)
class Trajectory:
    prompt_id: str
    logits: np.ndarray
    argmax: tuple[int, ...]
    fed: tuple[int, ...]
    calls: tuple[Call, ...]
    state: Mapping[str, Any]


def _download(ptr: int, nbytes: int, runtime: Any) -> np.ndarray:
    from hipengine.core.memory import DeviceBuffer, copy_device_to_host, host_array_ptr

    size = int(nbytes)
    raw = np.empty((size,), dtype=np.uint8)
    if size:
        copy_device_to_host(
            host_array_ptr(raw), DeviceBuffer(int(ptr), size), size, runtime=runtime
        )
    return raw


def state_fingerprint(session: Any) -> dict[str, Any]:
    """Hash and scan the live attention banks and convolution histories of a session."""

    attention = session.attention
    if attention is None:
        raise GateError("the gate needs a session with device attention state")
    runtime = attention.rt
    runtime.device_synchronize()
    digest = hashlib.sha256()
    finite = True
    layout: list[list[int]] = []
    for visit in sorted(attention.banks):
        keys, values, _spans = attention.banks[visit]
        cap, _nq, kv_heads, width, _extent = attention.geometry[visit]
        live = min(int(attention.positions.get(visit, 0)), int(cap))
        row_bytes = int(kv_heads) * int(width) * 4
        for name, pointer in (("keys", keys), ("values", values)):
            raw = _download(pointer, live * row_bytes, runtime)
            digest.update(raw)
            finite = finite and bool(np.isfinite(raw.view(np.float32)).all())
            layout.append([int(visit), len(name), live, int(kv_heads), int(width)])
    convolution = getattr(session, "convolution", None)
    histories = getattr(convolution, "histories", None) or {}
    for (visit, name), buffer in sorted(histories.items(), key=lambda item: (int(item[0][0]), str(item[0][1]))):
        raw = _download(buffer.ptr, buffer.nbytes, convolution.rt)
        digest.update(raw)
        finite = finite and bool(np.isfinite(raw.view(np.float32)).all())
        layout.append([int(visit), len(str(name)), raw.size // 4])
    digest.update(json.dumps(layout, separators=(",", ":")).encode("ascii"))
    return {
        "finite": bool(finite),
        "cells": len(layout),
        "min_bank_cap": int(min(int(cap) for cap, *_ in attention.geometry)),
        "state_sha256": digest.hexdigest(),
    }
def capture_trajectory(
    model: Any,
    *,
    prompt_id: str,
    prompt_tokens: Sequence[int],
    decode_rows: int,
    block_width: int,
    forced: Sequence[int] | None = None,
) -> Trajectory:
    """Run one request the way the generator does and record every session call.

    The prompt is committed in blocks of ``block_width`` tokens, a one-token
    block through ``step``, exactly as the generator schedules it. ``forced`` is
    the strict arm's chain; without it the arm follows its own greedy tokens.
    """

    prompt_tokens = [int(token) for token in prompt_tokens]
    if int(block_width) <= 0:
        raise GateError("the generator must declare a positive prefill block width")
    if forced is not None and len(forced) != decode_rows - 1:
        raise GateError(f"{prompt_id}: forced chain has {len(forced)} tokens for {decode_rows} rows")
    session = model.session(len(prompt_tokens) + decode_rows + 1)
    calls: list[Call] = []
    rows: list[np.ndarray] = []
    fed: list[int] = []
    try:
        with _RouteRecorder(model) as routes:
            for start in range(0, len(prompt_tokens), int(block_width)):
                block = prompt_tokens[start : start + int(block_width)]
                final = start + len(block) == len(prompt_tokens)
                before = (model._graph_captures, model._graph_replays)
                if len(block) == 1:
                    row = session.step(block[0], logits=final)
                else:
                    row = session.prefill_block(block, logits=final)
                calls.append(
                    _observe(WORK_PREFILL, block, session, model, before, routes.take(), row)
                )
                if final:
                    rows.append(np.asarray(row, dtype=np.float32))
            for _ in range(decode_rows - 1):
                token = int(np.argmax(rows[-1])) if forced is None else int(forced[len(fed)])
                fed.append(token)
                before = (model._graph_captures, model._graph_replays)
                row = np.asarray(session.step(token), dtype=np.float32)
                calls.append(_observe(WORK_DECODE, (token,), session, model, before, routes.take(), row))
                rows.append(row)
        state = state_fingerprint(session)
    finally:
        session.close()
    return Trajectory(
        prompt_id=prompt_id,
        logits=np.stack(rows),
        argmax=tuple(int(np.argmax(row)) for row in rows),
        fed=tuple(fed),
        calls=tuple(calls),
        state=state,
    )


def _observe(
    kind: str,
    tokens: Sequence[int],
    session: Any,
    model: Any,
    before: tuple[int, int],
    route_hash: str,
    row: Any,
) -> Call:
    return Call(
        kind=kind,
        tokens=tuple(int(token) for token in tokens),
        attention_positions=tuple(
            sorted((int(visit), int(position)) for visit, position in session.attention.positions.items())
        ),
        committed=session._undo is None,
        bucket=_graph_bucket(model, before),
        route_hash=route_hash,
        finite=row is None or bool(np.isfinite(np.asarray(row, dtype=np.float32)).all()),
    )


def _scheduled_calls(
    request: Mapping[str, Any],
    teacher: Mapping[str, Sequence[int]],
    *,
    block_width: int,
    candidate: Candidate,
) -> list[tuple[str, int, int, int, str]]:
    """List the calls the schedule demands: kind, width, position, publication ordinal, bucket."""

    prompt = [int(token) for token in request["prompt_tokens"]]
    chain = [int(token) for token in teacher[str(request["id"])]]
    rows = int(request["decode_steps"])
    if len(chain) != rows - 1:
        raise GateError(f"{request['id']}: teacher chain has {len(chain)} tokens for {rows} rows")
    plan: list[tuple[str, int, int, int, str]] = []
    committed = 0
    single_seen = False
    for start in range(0, len(prompt), block_width):
        width = min(block_width, len(prompt) - start)
        bucket = BUCKET_EAGER
        if width == 1 and candidate.graph:
            bucket = BUCKET_REPLAY if single_seen else BUCKET_CAPTURE
            single_seen = True
        plan.append((WORK_PREFILL, width, committed + width - 1, 0, bucket))
        committed += width
    for row in range(1, rows):
        bucket = BUCKET_EAGER
        if candidate.graph:
            bucket = BUCKET_REPLAY if single_seen else BUCKET_CAPTURE
            single_seen = True
        plan.append((WORK_DECODE, 1, committed + row - 1, row, bucket))
    return plan


def _planned_tokens(request: Mapping[str, Any], teacher: Mapping[str, Sequence[int]]) -> list[list[int]]:
    prompt = [int(token) for token in request["prompt_tokens"]]
    chain = [int(token) for token in teacher[str(request["id"])]]
    tokens: list[list[int]] = []
    for start in range(0, len(prompt), int(request["block_width"])):
        tokens.append(prompt[start : start + int(request["block_width"])])
    return tokens + [[token] for token in chain]


def expected_controls(
    scenario_id: str,
    requests: Sequence[Mapping[str, Any]],
    teacher: Mapping[str, Sequence[int]],
    *,
    candidate: Candidate,
    window: int,
    route_top_k: int,
    teacher_route_hashes: Mapping[str, Sequence[str]] | None = None,
) -> list[ControlRecord]:
    """Derive the scenario's controls from the schedule alone.

    ``teacher[request]`` is the strict arm's greedy chain: decode row ``k`` must
    be fed ``teacher[k - 1]``. ``teacher_route_hashes`` only fills the diagnostic
    route field with the teacher's own routing; every binding field comes from
    the schedule.
    """

    records: list[ControlRecord] = []
    step = 0
    for request in requests:
        request_id = str(request["id"])
        prompt = [int(token) for token in request["prompt_tokens"]]
        cap = min(len(prompt) + int(request["decode_steps"]) + 1, window)
        tokens = _planned_tokens(request, teacher)
        plan = _scheduled_calls(request, teacher, block_width=int(request["block_width"]), candidate=candidate)
        if len(tokens) != len(plan):
            raise GateError(f"{request_id}: schedule lists {len(plan)} calls for {len(tokens)} token blocks")
        hashes = teacher_route_hashes.get(request_id) if teacher_route_hashes else None
        for row, ((kind, width, position, publication, bucket), block) in enumerate(zip(plan, tokens, strict=True)):
            accepted = width if kind == WORK_PREFILL else 1
            records.append(
                _control(
                    scenario_id,
                    step,
                    request_id,
                    work_class=kind,
                    physical_width=width,
                    execution_row=width - 1,
                    input_token_id=block[-1],
                    position=position,
                    publication_ordinal=publication,
                    accepted_token_count=accepted,
                    cap=cap,
                    route_decision_hash=(
                        UNOBSERVED_ROUTES_HASH if hashes is None else str(hashes[row])
                    ),
                    route_top_k=route_top_k,
                    phase=PHASE_COMMITTED,
                    state_finite=True,
                    route_finite=True,
                    bucket=_listed_bucket(kind, width, bucket),
                )
            )
            step += 1
    return records


def _listed_bucket(kind: str, width: int, bucket: str) -> str:
    """A multi-row prefill block never runs a decode graph; a one-token block may."""

    return WORK_PREFILL.lower() if kind == WORK_PREFILL and width > 1 else bucket


def actual_controls(
    scenario_id: str,
    request: Mapping[str, Any],
    trajectory: Trajectory,
    *,
    route_top_k: int,
) -> list[ControlRecord]:
    """Read the controls off what the session was asked and reported, call by call."""

    request_id = str(request["id"])
    calls = trajectory.calls
    prompt_length = len(request["prompt_tokens"])
    blocks = -(-prompt_length // int(request["block_width"]))
    scheduled = blocks + int(request["decode_steps"]) - 1
    if len(calls) != scheduled:
        raise GateError(f"{request_id}: session made {len(calls)} calls for {scheduled} scheduled blocks")
    if any(call.kind != WORK_PREFILL for call in calls[:blocks]) or any(
        call.kind != WORK_DECODE for call in calls[blocks:]
    ):
        raise GateError(f"{request_id}: session call kinds do not follow the schedule")
    cap = int(trajectory.state["min_bank_cap"])
    state_finite = bool(trajectory.state["finite"])
    records: list[ControlRecord] = []
    first = int(request["first_step"])
    publication = 0
    for row, call in enumerate(calls):
        contexts = {position for _visit, position in call.attention_positions}
        if len(contexts) != 1:
            raise GateError(f"{request_id}: attention visits disagree after call {row}: {sorted(contexts)}")
        width = len(call.tokens)
        if call.kind == WORK_DECODE:
            publication += 1
        records.append(
            _control(
                scenario_id,
                first + row,
                request_id,
                work_class=call.kind,
                physical_width=width,
                execution_row=width - 1,
                input_token_id=call.tokens[-1],
                position=contexts.pop() - 1,
                publication_ordinal=publication,
                accepted_token_count=width if call.kind == WORK_PREFILL else 1,
                cap=cap,
                route_decision_hash=call.route_hash,
                route_top_k=route_top_k,
                phase=PHASE_COMMITTED if call.committed else PHASE_PROVISIONAL,
                state_finite=state_finite,
                route_finite=call.finite,
                bucket=_listed_bucket(call.kind, width, call.bucket),
            )
        )
    return records
STRICT_ARM = Candidate(
    name="strict",
    profile=STRICT_PROFILE,
    classification="T0",
    mechanism="strict selected-expert F32 kernels and ordered mix",
    strict_fallback="itself",
    environment=dict(STRICT_ENVIRONMENT),
)
TASK_MAX_TOKENS = 24
MAX_LISTED_MISMATCHES = 40
MAX_LISTED_DIAGNOSTICS = 5


def expected_flags(candidate: Candidate) -> dict[str, bool]:
    environment = candidate.environment
    return {
        "device_routes": environment["HIPENGINE_INKLING2_DEVICE_ROUTES"] == "1",
        "resident_decode": environment["HIPENGINE_INKLING2_RESIDENT_DECODE"] == "1",
        "decode_graph": environment["HIPENGINE_INKLING2_DECODE_GRAPH"] == "1",
    }


def _ids_hash(ids: Sequence[int]) -> str:
    return hashlib.sha256(np.asarray(list(ids), dtype="<i8").tobytes()).hexdigest()


@contextmanager
def _environment(values: Mapping[str, str]) -> Iterator[None]:
    """Set one arm's route flags while its model is constructed, then restore the process."""

    unset = object()
    previous = {name: os.environ.get(name, unset) for name in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is unset:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@contextmanager
def _open_llm(model: Path, backend: str, arm: Candidate, max_sequence_length: int) -> Iterator[Any]:
    """Open the public LLM for one arm under that arm's route flags.

    The generator reads the flags when it is first used, so the flags stay set
    for the arm's whole lifetime and are restored when it closes.
    """

    from hipengine import LLM

    with _environment(arm.environment):
        llm = LLM(
            model=str(model),
            backend=backend,
            execution_profile=arm.profile,
            max_sequence_length=max_sequence_length,
            prefix_cache="off",
            speculative_mtp_serving="off",
        )
        with closing(llm):
            yield llm


def _text_model(llm: Any) -> Any:
    generator = llm._get_text_generator()
    return getattr(generator, "model", None) or generator._inner.model


def _route_metadata(llm: Any, model: Any, arm: Candidate) -> dict[str, Any]:
    """Confirm the arm resolved its profile and engaged exactly the routes under test."""

    profile = str(llm.resolved_execution_profile)
    if profile != arm.profile or bool(llm.execution_profile_fell_back_to_strict):
        raise GateError(f"the {arm.name} arm resolved profile {profile!r}, expected {arm.profile!r}")
    projection = getattr(model, "projection", None)
    flags = {
        "device_routes": bool(projection.device_routes_enabled()) if projection is not None else False,
        "resident_decode": bool(getattr(model, "resident_decode", False)),
        "decode_graph": bool(getattr(model, "decode_graph", False)),
    }
    wanted = expected_flags(arm)
    if flags != wanted:
        raise GateError(f"the {arm.name} arm engaged {flags}, expected {wanted}")
    return {
        "profile": profile,
        "manifest_sha256": getattr(llm, "execution_profile_manifest_sha256", None),
        "backend": getattr(llm, "backend", None),
        "target_arch": getattr(projection, "_target_arch", None),
        "routes": flags,
    }


def _steps(trajectory: Trajectory) -> tuple[dict[str, Any], ...]:
    return tuple(
        {"logits": row, "token_id": token}
        for row, token in zip(trajectory.logits, trajectory.argmax, strict=True)
    )


def run_task_cases(llm: Any) -> dict[str, dict[str, Any]]:
    """Score the frozen known-answer cases through the public generate route."""

    from hipengine import SamplingParams

    sampling = SamplingParams(temperature=0, max_tokens=TASK_MAX_TOKENS)
    captured: dict[str, dict[str, Any]] = {}
    for key, category, prompt, expected in _CASES:
        ids = llm.tokenize(_render(category, prompt))
        (output,) = llm.generate_detailed(ids, sampling)
        answer = output.text.strip()
        captured[key] = {
            "category": category,
            "prompt_token_ids_sha256": _ids_hash(ids),
            "generated_token_ids": [int(token) for token in output.generated_token_ids],
            "answer": answer,
            "expected": expected,
            "correct": answer == expected,
        }
    return captured


def task_verdict(
    strict: Mapping[str, Mapping[str, Any]],
    candidate: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Paired criterion, fixed before the run: no case strict answers correctly may regress.

    Cases are scored independently, so an improvement cannot offset a regression.
    """

    if not strict or set(strict) != set(candidate):
        raise GateError("strict and candidate task case sets differ")
    if any(strict[key]["prompt_token_ids_sha256"] != candidate[key]["prompt_token_ids_sha256"] for key in strict):
        raise GateError("strict and candidate task inputs differed")
    regressed = sorted(
        key for key in strict if strict[key]["correct"] and not candidate[key]["correct"]
    )
    improved = sorted(
        key for key in strict if not strict[key]["correct"] and candidate[key]["correct"]
    )
    identical = sum(
        strict[key]["generated_token_ids"] == candidate[key]["generated_token_ids"]
        for key in strict
    )
    return {
        "criterion": "no case correct under strict may be wrong under the candidate",
        "cases": len(strict),
        "strict_correct": sum(bool(row["correct"]) for row in strict.values()),
        "candidate_correct": sum(bool(row["correct"]) for row in candidate.values()),
        "regressed_cases": regressed,
        "improved_cases": improved,
        "generated_ids_identical_cases": int(identical),
        "passed": not regressed,
    }


def public_route_check(llm: Any, prompt_ids: Sequence[int], free: Trajectory) -> dict[str, Any]:
    """The public generate route must reproduce the captured session route token for token."""

    from hipengine import SamplingParams

    (output,) = llm.generate_detailed(
        list(prompt_ids), SamplingParams(temperature=0, max_tokens=int(free.logits.shape[0]) - 1)
    )
    generated = [int(token) for token in output.generated_token_ids]
    chain = list(free.argmax[: len(generated)])
    matches = generated == chain
    return {
        "generated_tokens": len(generated),
        "session_chain_tokens": int(free.logits.shape[0]) - 1,
        "matches_session_chain": matches,
        "first_differing_token": next(
            (index for index, (first, second) in enumerate(zip(generated, chain)) if first != second),
            None,
        ),
        "passed": bool(generated) and matches,
    }


def repeat_verdict(runs: Mapping[str, Sequence[Trajectory]]) -> dict[str, Any]:
    """Repeat runs of one schedule must agree bit for bit in logits, calls and resident state."""

    mismatches: list[dict[str, Any]] = []
    nonfinite: list[str] = []
    for request_id, trajectories in runs.items():
        first = trajectories[0]
        for index, repeat in enumerate(trajectories[1:], start=1):
            checks = {
                "logits_exact": bool(np.array_equal(first.logits, repeat.logits)),
                "fed_tokens_exact": first.fed == repeat.fed,
                "calls_exact": first.calls == repeat.calls,
                "state_sha256_exact": first.state["state_sha256"] == repeat.state["state_sha256"],
            }
            if not all(checks.values()):
                mismatches.append({"request": request_id, "repeat_index": index, **checks})
        for repeat in trajectories:
            if not repeat.state["finite"] or not all(call.finite for call in repeat.calls):
                nonfinite.append(request_id)
    return {
        "runs_per_request": {key: len(value) for key, value in runs.items()},
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:MAX_LISTED_MISMATCHES],
        "nonfinite": sorted(set(nonfinite)),
        "passed": not mismatches and not nonfinite,
    }


def isolation_verdict(
    primary: Mapping[str, np.ndarray],
    replay: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """A request's logits must not depend on which unrelated requests ran before it."""

    rows = {
        request_id: request_id in replay
        and primary[request_id].shape == replay[request_id].shape
        and bool(np.array_equal(primary[request_id], replay[request_id]))
        for request_id in primary
    }
    mismatched = sorted(key for key, same in rows.items() if not same)
    return {
        "criterion": "bit-identical logits per request in a fresh process with the schedule reversed",
        "requests": len(rows),
        "bit_identical_requests": sum(rows.values()),
        "mismatched": mismatched,
        "passed": bool(rows) and not mismatched,
    }


def routing_agreement(
    strict: Sequence[Call],
    candidate: Sequence[Call],
) -> dict[str, Any]:
    """Count the calls whose host-selected experts equal the teacher's (diagnostic)."""

    observed = [
        (first.route_hash, second.route_hash)
        for first, second in zip(strict, candidate, strict=True)
        if first.route_hash != UNOBSERVED_ROUTES_HASH and second.route_hash != UNOBSERVED_ROUTES_HASH
    ]
    return {
        "calls_routed_on_host_in_both_arms": len(observed),
        "calls_routed_as_strict": sum(first == second for first, second in observed),
        "scope": "arms that select experts on the host; device-routed calls are not observed",
    }


def _compact_scenario(summary: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Keep the scenario's counts; its per-step sequences are reproducible from the schedule."""

    if summary is None:
        return None
    kept = {key: value for key, value in summary.items() if key in {
        "records", "steps", "request_count", "width_transition_count", "compaction_count",
        "page_boundary_count", "work_classes"}}
    kept["ragged_step_count"] = len(summary["ragged_steps"])
    kept["width_min_max"] = [min(summary["width_sequence"]), max(summary["width_sequence"])]
    return kept


def _capped(comparison: dict[str, Any]) -> dict[str, Any]:
    listed = comparison["mismatches"]
    comparison["mismatch_count"] = len(listed)
    comparison["mismatches"] = listed[:MAX_LISTED_MISMATCHES]
    comparison["diagnostic_mismatch_count"] = len(comparison["diagnostic_mismatches"])
    comparison["diagnostic_mismatches"] = comparison["diagnostic_mismatches"][:MAX_LISTED_DIAGNOSTICS]
    for key in ("expected_scenario", "actual_scenario"):
        comparison[key] = _compact_scenario(comparison[key])
    return comparison


def _run_isolation_worker(
    args: argparse.Namespace,
    plan: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Replay the schedule in a fresh process; return its logits per request and its report."""

    work = args.json.parent / f"{args.json.stem}.isolation"
    work.mkdir(parents=True, exist_ok=True)
    plan_path = work / "plan.json"
    plan_path.write_text(json.dumps(plan, sort_keys=True) + "\n", encoding="utf-8")
    worker_command = [
        sys.executable,
        str(Path(__file__).resolve()),
        str(args.model),
        "--backend",
        str(args.backend),
        "--candidate",
        str(args.candidate),
        "--max-sequence-length",
        str(args.max_sequence_length),
        "--_worker",
        "--plan",
        str(plan_path),
        "--worker-out",
        str(work),
    ]
    with (work / "worker.log").open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            worker_command, cwd=str(REPO_ROOT), stdout=log, stderr=subprocess.STDOUT, check=False
        )
    if completed.returncode != 0:
        tail = (work / "worker.log").read_text(encoding="utf-8", errors="replace")[-2000:]
        raise GateError(f"isolation worker exited with {completed.returncode}:\n{tail}")
    report = json.loads((work / "worker.json").read_text(encoding="utf-8"))
    with np.load(work / "isolation.npz") as data:
        order = [str(value) for value in data["order"].tolist()]
        logits = {request_id: data[f"rows_{index}"] for index, request_id in enumerate(order)}
    return logits, report


def run_worker(args: argparse.Namespace) -> int:
    """Fresh process: teacher-force the reversed schedule, then check the public route."""

    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    arm = CANDIDATES[str(args.candidate)]
    output_dir = Path(args.worker_out)
    output_dir.mkdir(parents=True, exist_ok=True)
    order: list[str] = []
    matrices: list[np.ndarray] = []
    public: dict[str, dict[str, Any]] = {}
    with _open_llm(Path(plan["model"]), str(args.backend), arm, int(args.max_sequence_length)) as llm:
        model = _text_model(llm)
        route = _route_metadata(llm, model, arm)
        block_width = int(getattr(llm._get_text_generator(), "prefill_block_width", 64))
        for entry in reversed(plan["requests"]):
            ids = [int(token) for token in llm.tokenize(entry["text"])]
            if _ids_hash(ids) != entry["prompt_token_ids_sha256"]:
                raise GateError(f"{entry['id']}: prompt token ids differ between processes")
            rows = int(entry["decode_rows"])
            forced = capture_trajectory(
                model,
                prompt_id=entry["id"],
                prompt_tokens=ids,
                decode_rows=rows,
                block_width=block_width,
                forced=entry["teacher"],
            )
            free = capture_trajectory(
                model,
                prompt_id=entry["id"],
                prompt_tokens=ids,
                decode_rows=rows,
                block_width=block_width,
            )
            public[entry["id"]] = public_route_check(llm, ids, free)
            order.append(entry["id"])
            matrices.append(forced.logits)
            print(f"replay {entry['id']}: isolation rows, public route {public[entry['id']]['passed']}", flush=True)
    np.savez(
        output_dir / "isolation.npz",
        order=np.asarray(order),
        **{f"rows_{index}": matrix for index, matrix in enumerate(matrices)},
    )
    (output_dir / "worker.json").write_text(
        json.dumps(
            {"route": route, "public_route": public, "order": order, "block_width": block_width},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


def run(args: argparse.Namespace, *, command: Sequence[str]) -> dict[str, Any]:
    if not args.model.is_file():
        raise GateError(f"model does not exist: {args.model}")
    if int(args.repeat_runs) < 3:
        raise GateError("repeat runs must be at least three")
    arm = CANDIDATES[str(args.candidate)]
    complete_suite = args.limit is None
    suite = load_suite(Path(args.suite), limit=args.limit)
    scenario_id = f"inkling2-{args.model.stem}-{arm.name}"
    strict_scenario_id = f"{scenario_id}-strict"
    max_length = int(args.max_sequence_length)

    # The strict arm supplies the teacher chain every other arm is forced onto.
    with _open_llm(args.model, str(args.backend), STRICT_ARM, max_length) as llm:
        model = _text_model(llm)
        block_width = int(getattr(llm._get_text_generator(), "prefill_block_width", 64))
        strict_route = _route_metadata(llm, model, STRICT_ARM)
        window = int(model.config["sliding_window_size"])
        route_top_k = int(model.mapping.layers[0].top_k)
        requests: list[dict[str, Any]] = []
        teacher: dict[str, tuple[int, ...]] = {}
        strict_runs: dict[str, Trajectory] = {}
        for index, row in enumerate(suite):
            text = render_prompt(row)
            ids = [int(token) for token in llm.tokenize(text)]
            trajectory = capture_trajectory(
                model,
                prompt_id=row["id"],
                prompt_tokens=ids,
                decode_rows=int(row["decode_steps"]),
                block_width=block_width,
            )
            teacher[row["id"]] = trajectory.argmax[:-1]
            strict_runs[row["id"]] = trajectory
            requests.append(
                {
                    **row,
                    "text": text,
                    "prompt_tokens": ids,
                    "block_width": block_width,
                    "prompt_token_ids_sha256": _ids_hash(ids),
                }
            )
            print(
                f"strict {index + 1}/{len(suite)} {row['id']}: {len(ids)} prompt tokens, "
                f"{row['decode_steps']} rows",
                flush=True,
            )
        coverage = validate_coverage(requests, window=window)
        if max(len(item["prompt_tokens"]) + int(item["decode_steps"]) + 1 for item in requests) > max_length:
            raise GateError("a request exceeds --max-sequence-length")
        strict_tasks = run_task_cases(llm)
    calls_before = 0
    for request in requests:
        request["first_step"] = calls_before
        calls_before += len(strict_runs[request["id"]].calls)

    # The candidate arm: the primary run plus its repeats, on the teacher's chain.
    with _open_llm(args.model, str(args.backend), arm, max_length) as llm:
        model = _text_model(llm)
        candidate_route = _route_metadata(llm, model, arm)
        runs: dict[str, list[Trajectory]] = {}
        for index, request in enumerate(requests):
            ids = [int(token) for token in llm.tokenize(request["text"])]
            if _ids_hash(ids) != request["prompt_token_ids_sha256"]:
                raise GateError(f"{request['id']}: prompt token ids differ between arms")
            runs[request["id"]] = [
                capture_trajectory(
                    model,
                    prompt_id=request["id"],
                    prompt_tokens=ids,
                    decode_rows=int(request["decode_steps"]),
                    block_width=block_width,
                    forced=teacher[request["id"]],
                )
                for _ in range(int(args.repeat_runs))
            ]
            print(
                f"{arm.name} {index + 1}/{len(requests)} {request['id']}: {args.repeat_runs} forced runs",
                flush=True,
            )
        candidate_tasks = run_task_cases(llm)

    replay_logits, replay_report = _run_isolation_worker(
        args,
        {
            "model": str(args.model),
            "requests": [
                {
                    "id": request["id"],
                    "text": request["text"],
                    "decode_rows": int(request["decode_steps"]),
                    "prompt_token_ids_sha256": request["prompt_token_ids_sha256"],
                    "teacher": list(teacher[request["id"]]),
                }
                for request in requests
            ],
        },
    )
    if replay_report["route"] != candidate_route:
        raise GateError("the isolation worker resolved a different route than the primary run")

    captures = tuple(
        PromptCalibrationCapture(
            prompt_id=request["id"],
            category=str(request["category"]),
            strict=_steps(strict_runs[request["id"]]),
            candidate_runs={arm.name: tuple(_steps(item) for item in runs[request["id"]])},
        )
        for request in requests
    )
    primary = {request["id"]: runs[request["id"]][0] for request in requests}

    def shape_for_step(step: int, _prompt_id: str) -> str:
        return "prefill_last" if step == 0 else "c1"

    def transition_for_step(step: int, prompt_id: str) -> str:
        if step == 0:
            return "prefill_to_c1"
        return str(primary[prompt_id].calls[step].bucket)

    dense = build_candidate_quality(
        captures,
        candidate_mode=arm.name,
        scenario_id=scenario_id,
        thresholds=EvaluationThresholds(),
        shape_for_step=shape_for_step,
        transition_for_step=transition_for_step,
    )
    strict_matrix = np.concatenate([strict_runs[request["id"]].logits for request in requests])
    candidate_matrix = np.concatenate([primary[request["id"]].logits for request in requests])
    labels = np.asarray(
        [token for request in requests for token in strict_runs[request["id"]].argmax], dtype=np.int64
    )
    vectors = per_row_metrics(strict_matrix, candidate_matrix, labels, top_k=dense["quality"]["top_k"])
    verdict = section_211_verdict(
        vectors["kl_nats"],
        vectors["top1_equal"],
        [str(request["category"]) for request in requests for _ in range(int(request["decode_steps"]))],
    )
    if args.save_logits is not None:
        args.save_logits.mkdir(parents=True, exist_ok=True)
        np.save(args.save_logits / "strict.npy", strict_matrix)
        np.save(args.save_logits / "candidate.npy", candidate_matrix)

    teacher_hashes = {
        request["id"]: [call.route_hash for call in strict_runs[request["id"]].calls] for request in requests
    }
    strict_expected = expected_controls(
        strict_scenario_id,
        requests,
        teacher,
        candidate=STRICT_ARM,
        window=window,
        route_top_k=route_top_k,
        teacher_route_hashes=teacher_hashes,
    )
    candidate_expected = expected_controls(
        scenario_id,
        requests,
        teacher,
        candidate=arm,
        window=window,
        route_top_k=route_top_k,
        teacher_route_hashes=teacher_hashes,
    )
    strict_actual: list[ControlRecord] = []
    candidate_actual: list[ControlRecord] = []
    for request in requests:
        strict_actual.extend(
            actual_controls(strict_scenario_id, request, strict_runs[request["id"]], route_top_k=route_top_k)
        )
        candidate_actual.extend(
            actual_controls(scenario_id, request, primary[request["id"]], route_top_k=route_top_k)
        )
    strict_controls = _capped(compare_control_records(strict_expected, strict_actual))
    candidate_controls = _capped(
        compare_control_records(
            candidate_expected, candidate_actual, diagnostic_fields=("route_decision_hash",)
        )
    )
    routing = routing_agreement(
        [call for request in requests for call in strict_runs[request["id"]].calls],
        [call for request in requests for call in primary[request["id"]].calls],
    )
    repeat = repeat_verdict(runs)
    isolation = isolation_verdict(
        {request["id"]: primary[request["id"]].logits for request in requests}, replay_logits
    )
    public_rows = replay_report["public_route"]
    public = {
        "criterion": "greedy generate_detailed output equals the captured session chain",
        "requests": len(public_rows),
        "requests_passed": sum(bool(row["passed"]) for row in public_rows.values()),
        "generated_tokens": sum(int(row["generated_tokens"]) for row in public_rows.values()),
        "failed": sorted(key for key, row in public_rows.items() if not row["passed"]),
        "by_request": {key: public_rows[key] for key in sorted(public_rows)
                       if not public_rows[key]["passed"]},
    }
    public["passed"] = (
        public["requests"] == public["requests_passed"] and public["generated_tokens"] > 0
    )
    tasks = task_verdict(strict_tasks, candidate_tasks)

    environment = {name: os.environ.get(name) for name in ROUTE_ENV_KEYS}
    environment.update(
        {
            "HIPENGINE_HIP_ARCH": os.environ.get("HIPENGINE_HIP_ARCH"),
            "HIP_VISIBLE_DEVICES": os.environ.get("HIP_VISIBLE_DEVICES"),
            "strict_route_environment": dict(STRICT_ARM.environment),
            "candidate_route_environment": dict(arm.environment),
        }
    )
    provenance = collect_artifact_provenance(
        repo_root=REPO_ROOT,
        configured_backend=str(args.backend),
        resolved_backend=candidate_route["backend"],
        target_arch=candidate_route["target_arch"],
        model_path=args.model,
        quant=str(args.quant),
        kv_dtype="fp32",
        command=command,
        environment=environment,
        build_profile="execution_profile_inkling2_gguf_gate",
        timing_protocol="none_full_logits_and_state_only_v1",
        warmups=0,
        repetitions=int(args.repeat_runs),
        profiler={"enabled": False, "kind": None, "command": None},
    )
    gates = {
        "quality_section_2_11": verdict["passed"],
        "control_semantics": bool(strict_controls["passed"] and candidate_controls["passed"]),
        "repeat_determinism": repeat["passed"],
        "isolation": isolation["passed"],
        "public_route": public["passed"],
        "task_quality": tasks["passed"],
    }
    measurement_valid = bool(not provenance.get("dirty") and complete_suite)
    gate_passed = bool(measurement_valid and all(gates.values()))
    # The dense-envelope scopes are reported for localization; section 2.11 carries the verdict.
    dense_quality = dense["quality"]
    dense_compact = {
        **{key: value for key, value in dense.items() if key != "quality"},
        "quality": {
            **{key: value for key, value in dense_quality.items() if key not in {
                "by_scope", "rows_over_review_boundary", "top1_mismatch_rows"}},
            "rows_over_review_boundary": len(dense_quality["rows_over_review_boundary"]),
            "top1_mismatch_rows": dense_quality["top1_mismatch_rows"][:MAX_LISTED_MISMATCHES],
        },
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "status": "passed" if gate_passed else "failed_or_screen_only",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "measurement_valid": measurement_valid,
        "complete_prompt_suite": complete_suite,
        "performance_claim": False,
        "candidate": {
            "name": arm.name,
            "classification": arm.classification,
            "mechanism": arm.mechanism,
            "strict_fallback": arm.strict_fallback,
            "strict_route_environment": dict(STRICT_ARM.environment),
            "candidate_route_environment": dict(arm.environment),
            "strict_owner": strict_route,
            "candidate_owner": candidate_route,
            "isolation_owner": replay_report["route"],
        },
        "protocol": {
            "model": str(args.model),
            "prompt_suite": str(Path(args.suite).resolve()),
            "prompt_count": len(requests),
            "teacher_forced_rows": int(strict_matrix.shape[0]),
            "session_calls_per_run": calls_before,
            "candidate_repeat_runs": int(args.repeat_runs),
            "prefill_block_width": block_width,
            "coverage": coverage,
            "route_top_k": route_top_k,
            "dense_envelope_thresholds": EvaluationThresholds().to_dict(),
            "section_2_11_thresholds": {
                "kl_median": MEDIAN_KL_MAX,
                "kl_p90": P90_KL_MAX,
                "damage_line_kl": DAMAGE_LINE_KL,
                "flip_damage_rate_global": DAMAGE_RATE_GLOBAL_MAX,
                "flip_damage_rate_category": DAMAGE_RATE_CATEGORY_MAX,
                "top1_global": TOP1_GLOBAL_MIN,
                "top1_category": TOP1_CATEGORY_MIN,
            },
        },
        "prompts": [
            {
                "id": request["id"],
                "category": request["category"],
                "prompt_sha256": prompt_sha256(request["text"]),
                "prompt_tokens": len(request["prompt_tokens"]),
                "rows": int(request["decode_steps"]),
            }
            for request in requests
        ],
        "quality": {
            "binding": "docs/EXECUTION-PROFILES.md section 2.11 (router-routing-scoped)",
            "section_2_11": verdict,
            "dense_envelope": dense_compact,
        },
        "control_semantics": {"strict": strict_controls, "candidate": candidate_controls},
        "routing_agreement": routing,
        "repeat_determinism": repeat,
        "isolation": isolation,
        "public_route": public,
        "task_quality": tasks,
        "gates": gates,
        "decision": {"passed": gate_passed},
        "provenance": provenance,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("model", type=Path)
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--candidate", choices=CANDIDATE_NAMES, required=True)
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument(
        "--limit", type=int, default=None, help="screen only: the first N prompts; cannot pass"
    )
    parser.add_argument("--repeat-runs", type=int, default=3)
    parser.add_argument("--max-sequence-length", type=int, default=4096)
    parser.add_argument("--quant", default="UD-Q4_K_M")
    parser.add_argument("--save-logits", type=Path, default=None)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--plan", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-out", type=Path, default=None, help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(raw_argv)
    if args._worker:
        if args.plan is None or args.worker_out is None:
            parser.error("--_worker needs --plan and --worker-out")
        try:
            return run_worker(args)
        except (GateError, OSError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
    if args.json is None:
        parser.error("--json is required")
    command = [sys.executable, str(Path(__file__).relative_to(REPO_ROOT)), *raw_argv]
    try:
        artifact = run(args, command=command)
    except (GateError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(args.json)
    print(
        json.dumps(
            {
                "status": artifact["status"],
                "measurement_valid": artifact["measurement_valid"],
                "gates": artifact["gates"],
                "section_2_11": artifact["quality"]["section_2_11"]["global"],
                "task_quality": {
                    key: artifact["task_quality"][key]
                    for key in ("strict_correct", "candidate_correct", "cases", "regressed_cases")
                },
            },
            indent=2,
        )
    )
    return 0 if artifact["decision"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
