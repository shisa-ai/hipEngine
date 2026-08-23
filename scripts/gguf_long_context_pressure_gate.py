#!/usr/bin/env python3
# ruff: noqa: E402
"""Validate gfx11 GGUF long-context concurrency and bounded device-KV pressure.

The packet reuses the production OpenAI/Uvicorn machinery but gives memory
lifecycle its own fail-closed protocol.  It runs concurrent 1K, 4K, 32K, mixed
1K/4K/32K, and an optional feasible longer context; then it reconfigures the
same idle owner to a deliberately tight device-KV high water.  A live 32K row
must survive while a 4K row receives retryable ``429 engine_busy`` from KV
capacity, after which the pool must shrink, regrow with fresh logical block ids,
invalidate/rebind graphs, remain c1-exact, and drain all ownership.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import json
import math
import os
import shlex
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from types import MethodType
from typing import Any, Mapping, Sequence

# Keep direct-script execution in this worktree even when another editable
# hipEngine checkout has registered the namespace-only ``scripts`` package.
_SCRIPT_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_SCRIPT_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_REPO_ROOT))

from hipengine import LLM, SamplingParams
from hipengine.benchmark.provenance import collect_artifact_provenance
from hipengine.kernels.backends import backend_package_capability
from hipengine.server import ServerConfig, create_app
from scripts.gguf_live_server_bench import (
    _artifact_backend_scope,
    _memory_snapshot,
    _run_reference,
)
from scripts.gguf_production_load_gate import (
    REPO_ROOT,
    SLOThresholds,
    WorkloadRequest,
    _EXACT_ENV,
    _HTTPTrace,
    _LocalUvicorn,
    _ReclaimedRow,
    _counter_delta,
    _evaluate_workload,
    _execute_workload,
    _http_json,
    _join_trace,
    _memory_recovery_gate,
    _metrics_snapshot,
    _parse_workload_names,
    _prompt_manifest,
    _read_compiler_version,
    _reclaimed_overrides,
    _stream_request,
    _temporary_environment,
    _wait_for_idle,
)


DEFAULT_MODEL = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
_SUPPORTED_BACKENDS = ("hip_gfx1100", "hip_gfx1151")
_REQUIRED_CONTEXTS = (1_024, 4_096, 16_384, 32_768)
# _execute_workload is shared with the production packet and intentionally
# sends this retained model id. Keep the focused pressure request identical.
_SERVED_MODEL_NAME = "qwen35-production-load"
_PROVENANCE_ENV_KEYS = (
    "HIPENGINE_BACKEND",
    "HIPENGINE_HIP_ARCH",
    "HIPENGINE_COMPILER_VERSION_FILE",
    "HIPENGINE_GGUF_FP16_RECURRENT_STATE",
    "HIPENGINE_GPU_MAX_HW_QUEUES_POLICY",
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "GPU_MAX_HW_QUEUES",
)


@dataclass(frozen=True)
class LongContextPoolPlan:
    block_size: int
    decode_tokens: int
    longer_context_tokens: int | None
    pages_by_context: dict[int, int]
    initial_pages: int
    low_water_pages: int
    mixed_high_water_pages: int
    pressure_high_water_pages: int
    chunk_pages: int

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["pages_by_context"] = {
            str(context): int(pages)
            for context, pages in sorted(self.pages_by_context.items())
        }
        return payload


def _request_pages(prompt_tokens: int, decode_tokens: int, *, block_size: int = 256) -> int:
    prompt = int(prompt_tokens)
    decode = int(decode_tokens)
    block = int(block_size)
    if prompt <= 0 or decode <= 0 or block <= 0:
        raise ValueError("prompt, decode, and block sizes must be positive")
    # Mirrors Qwen35GGUFResidentModelRunner.reserve_admission: the first output
    # token comes from prefill, so retained KV needs prompt + (decode - 1).
    return max(1, math.ceil((prompt + decode - 1) / block))


def _phase_pool_pages(request_pages: Sequence[int], *, initial_pages: int) -> int:
    del initial_pages
    # GlobalKVPoolSet admits every request from one fungible page set; capacity
    # is the simultaneous page sum, not a per-request chunk packing bound.
    return sum(int(value) for value in request_pages)


def build_pool_plan(
    *,
    decode_tokens: int,
    longer_context_tokens: int | None,
    block_size: int = 256,
) -> LongContextPoolPlan:
    decode = int(decode_tokens)
    longer = None if longer_context_tokens in (None, 0) else int(longer_context_tokens)
    contexts = [*_REQUIRED_CONTEXTS]
    if longer is not None:
        if longer <= _REQUIRED_CONTEXTS[-1]:
            raise ValueError("longer context must exceed 32K")
        contexts.append(longer)
    pages = {
        context: _request_pages(context, decode, block_size=int(block_size))
        for context in contexts
    }
    initial = pages[_REQUIRED_CONTEXTS[0]]
    phase_shapes = [
        (pages[1_024], pages[1_024]),
        (pages[4_096], pages[4_096]),
        (pages[16_384], pages[16_384]),
        (pages[32_768], pages[32_768]),
        (pages[1_024], pages[4_096], pages[16_384], pages[32_768]),
        (pages[32_768],),
    ]
    if longer is not None:
        phase_shapes.append((pages[longer], pages[longer]))
    mixed_high = max(
        _phase_pool_pages(shape, initial_pages=initial)
        for shape in phase_shapes
    )
    pressure_high = initial + pages[32_768]
    return LongContextPoolPlan(
        block_size=int(block_size),
        decode_tokens=decode,
        longer_context_tokens=longer,
        pages_by_context=pages,
        initial_pages=initial,
        low_water_pages=initial,
        mixed_high_water_pages=mixed_high,
        pressure_high_water_pages=pressure_high,
        chunk_pages=initial,
    )


def _graph_output_tokens(backend: str, decode_tokens: int) -> int:
    resolved = str(backend)
    if resolved not in _SUPPORTED_BACKENDS:
        raise ValueError(f"unsupported backend {resolved!r}")
    minimum = backend_package_capability(
        resolved,
        "GGUF_DECODE_GRAPH_MIN_REPLAY_STEPS",
    )
    if minimum is None or int(minimum) <= 0:
        raise RuntimeError(f"{resolved} does not declare GGUF c1 graph admission")
    # Prefill emits the seed output before c1 decode starts. Keep at least the
    # backend's admitted number of remaining graph transitions.
    return max(int(decode_tokens), int(minimum) + 1)


def build_workload_specs(
    *,
    decode_tokens: int,
    longer_context_tokens: int | None,
    backend: str = "hip_gfx1151",
) -> dict[str, tuple[WorkloadRequest, ...]]:
    decode = int(decode_tokens)
    if decode <= 0:
        raise ValueError("decode-tokens must be positive")
    graph_decode = _graph_output_tokens(str(backend), decode)
    workloads: dict[str, tuple[WorkloadRequest, ...]] = {
        "context_1k_c2": (
            WorkloadRequest("context-1k-a", 9707, 1_024, decode),
            WorkloadRequest("context-1k-b", 9708, 1_024, decode),
        ),
        "context_4k_c2": (
            WorkloadRequest("context-4k-a", 9707, 4_096, decode),
            WorkloadRequest("context-4k-b", 9708, 4_096, decode),
        ),
        "context_16k_c2": (
            WorkloadRequest("context-16k-a", 9707, 16_384, decode),
            WorkloadRequest("context-16k-b", 9708, 16_384, decode),
        ),
        "context_32k_c2": (
            WorkloadRequest("context-32k-a", 9707, 32_768, decode),
            WorkloadRequest("context-32k-b", 9708, 32_768, decode),
        ),
        "mixed_1k_4k_32k": (
            WorkloadRequest("mixed-1k", 9707, 1_024, decode),
            WorkloadRequest("mixed-4k", 9709, 4_096, decode),
            WorkloadRequest("mixed-32k", 9710, 32_768, decode),
        ),
    }
    longer = None if longer_context_tokens in (None, 0) else int(longer_context_tokens)
    if longer is not None:
        label = f"context_{longer // 1024}k_c2"
        workloads[label] = (
            WorkloadRequest(f"context-{longer // 1024}k-a", 9707, longer, decode),
            WorkloadRequest(f"context-{longer // 1024}k-b", 9708, longer, decode),
        )
    workloads["graph_seed_32k_c1"] = (
        WorkloadRequest("graph-seed-32k", 9709, 32_768, graph_decode),
    )
    workloads["graph_regrow_32k_c1"] = (
        WorkloadRequest("graph-regrow-blocker-1k", 9708, 1_024, graph_decode),
        WorkloadRequest(
            "graph-regrow-32k",
            9710,
            32_768,
            graph_decode,
            arrival_offset_seconds=0.1,
        ),
    )
    return workloads


def _pressure_specs(*, decode_tokens: int) -> tuple[WorkloadRequest, WorkloadRequest]:
    return (
        WorkloadRequest("pressure-live-32k", 9709, 32_768, int(decode_tokens)),
        WorkloadRequest("pressure-reject-4k", 9710, 4_096, int(decode_tokens)),
    )


def _required_admission(plan: LongContextPoolPlan) -> dict[str, Any]:
    return {
        "resource": "device_kv_pool",
        "requested_units": int(plan.pages_by_context[4_096]),
        "current_units": int(plan.pressure_high_water_pages),
        "capacity_units": int(plan.pressure_high_water_pages),
    }


def evaluate_packet(
    *,
    plan: LongContextPoolPlan,
    workloads: Mapping[str, Mapping[str, Any]],
    pressure: Mapping[str, Any],
    final_pool: Mapping[str, Any],
    graph_delta: Mapping[str, Any],
    pressure_block_ids: Sequence[int],
    regrow_block_ids: Sequence[int],
) -> dict[str, Any]:
    reasons: list[str] = []
    expected_names = tuple(
        build_workload_specs(
            decode_tokens=plan.decode_tokens,
            longer_context_tokens=plan.longer_context_tokens,
        )
    )
    if set(workloads) != set(expected_names) or any(
        workloads.get(name, {}).get("passed") is not True
        for name in expected_names
    ):
        reasons.append("long_context_workload_failed")
    if not (
        pressure.get("passed") is True
        and pressure.get("long_outcome") == "completed"
        and pressure.get("candidate_outcome") == "rejected"
        and pressure.get("candidate_error_code") == "engine_busy"
        and int(pressure.get("candidate_error_status_code", -1)) == 429
        and pressure.get("candidate_done_sentinel") is True
    ):
        reasons.append("pressure_accept_reject_contract_failed")
    if pressure.get("candidate_admission") != _required_admission(plan):
        reasons.append("pressure_admission_metadata_mismatch")
    if not (
        int(final_pool.get("current_pages", -1))
        == int(plan.pressure_high_water_pages)
        and int(final_pool.get("free_pages", -1))
        == int(plan.pressure_high_water_pages)
        and int(final_pool.get("refcounted_pages", -1)) == 0
        and int(final_pool.get("pinned_pages", -1)) == 0
        and int(final_pool.get("grow_events", -1)) == 0
        and int(final_pool.get("grow_failures", 0)) > 0
        and int(final_pool.get("shrink_events", -1)) == 0
    ):
        reasons.append("final_global_pool_lifecycle_failed")
    if not (
        int(graph_delta.get("captures", 0)) > 0
        and int(graph_delta.get("replays", 0)) > 0
        and int(graph_delta.get("invalidations", 0)) > 0
    ):
        reasons.append("graph_rebind_evidence_failed")
    pressure_ids = {int(block_id) for block_id in pressure_block_ids}
    regrow_ids = {int(block_id) for block_id in regrow_block_ids}
    if not pressure_ids or not regrow_ids:
        reasons.append("regrow_block_ids_missing")
    elif tuple(int(value) for value in pressure_block_ids) == tuple(
        int(value) for value in regrow_block_ids
    ):
        reasons.append("regrow_page_table_did_not_change")
    return {"passed": not reasons, "failure_reasons": reasons}


def _graph_totals(snapshot: Mapping[str, Any]) -> dict[str, int]:
    graph = snapshot.get("graph_buckets", {}) if isinstance(snapshot, Mapping) else {}
    return {
        "captures": int(graph.get("captures_total", 0)),
        "replays": int(graph.get("replays_total", 0)),
        "invalidations": int(graph.get("invalidations_total", 0)),
    }


def _graph_delta(before: Mapping[str, int], after: Mapping[str, int]) -> dict[str, int]:
    return {
        key: int(after.get(key, 0)) - int(before.get(key, 0))
        for key in ("captures", "replays", "invalidations")
    }


def _admission_metadata(trace: _HTTPTrace) -> dict[str, Any] | None:
    payload = trace.error_payload
    if not isinstance(payload, Mapping):
        return None
    hipengine = payload.get("hipengine")
    if not isinstance(hipengine, Mapping):
        return None
    routing = hipengine.get("routing")
    if not isinstance(routing, Mapping):
        return None
    admission = routing.get("admission")
    return copy.deepcopy(dict(admission)) if isinstance(admission, Mapping) else None


def _pool_json(runner: Any) -> dict[str, Any]:
    stats = runner.kv_pool_stats
    return {} if stats is None else stats.to_json_dict()


def _final_pool_from_idle_snapshot(final_idle: Mapping[str, Any]) -> dict[str, Any]:
    snapshot = final_idle.get("snapshot")
    snapshot = snapshot if isinstance(snapshot, Mapping) else {}
    runner = snapshot.get("runner")
    runner = runner if isinstance(runner, Mapping) else {}
    pool = runner.get("kv_pool")
    return copy.deepcopy(dict(pool)) if isinstance(pool, Mapping) else {}


def _wait_for_pressure_allocation(
    llm: LLM,
    runner: Any,
    *,
    expected_pages: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + float(timeout_seconds)
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        snapshot = llm.live_loop_snapshot() or {}
        pool = _pool_json(runner)
        active = int(snapshot.get("loop", {}).get("requests", {}).get("active", 0))
        last = {"pool": pool, "active_requests": active}
        if int(pool.get("refcounted_pages", 0)) == int(expected_pages) and active >= 1:
            return last
        time.sleep(0.01)
    raise TimeoutError(f"pressure source was not admitted at the expected page count: {last}")


def _capture_runtime(adapter: Any, runner: Any):
    reclaimed: dict[int, _ReclaimedRow] = {}
    reclaimed_lock = threading.Lock()
    original_reclaim = runner.reclaim

    def capture_reclaim(self, completed):
        request_id = int(completed.request_id)
        row = self._rows.get(request_id)
        if row is not None:
            self._flush_row_owner(row)
            allocation = row.kv_allocation
            captured = _ReclaimedRow(
                request_id=request_id,
                prompt_ids=tuple(int(token) for token in completed.prompt_tokens),
                generated_ids=(
                    ()
                    if row.slot is None
                    else tuple(int(token) for token in row.slot.generated_ids)
                ),
                finish_reason=str(completed.finish_reason),
                finish_details=completed.finish_details.to_json_dict(),
                observability=completed.observability.to_json_dict(),
                block_ids=(
                    () if allocation is None else tuple(int(block) for block in allocation.block_ids)
                ),
                completed_at=time.perf_counter(),
                pointers=(
                    () if allocation is None else tuple(int(pointer) for pointer in allocation.pointers)
                ),
            )
            with reclaimed_lock:
                reclaimed[request_id] = captured
        return original_reclaim(completed)

    runner.reclaim = MethodType(capture_reclaim, runner)
    timeline: list[dict[str, Any]] = []
    capture_state: dict[str, Any] = {"label": None}
    original_poll = adapter.poll

    def capture_poll(self, *, max_ticks=1):
        events = tuple(original_poll(max_ticks=max_ticks))
        plans: list[dict[str, Any]] = []
        for event in events:
            if (
                event.kind == "work"
                and event.work_kind is not None
                and event.work_kind.value == "decode"
            ):
                plan = runner._last_physical_group_plan
                if isinstance(plan, dict) and plan:
                    plans.append(copy.deepcopy(plan))
        snapshot = adapter.live_loop_snapshot() or {}
        loop = snapshot.get("loop", {})
        timeline.append(
            {
                "label": capture_state["label"],
                "observed_at": time.perf_counter(),
                "admitted": [
                    int(event.request_id)
                    for event in events
                    if event.kind == "admitted" and event.request_id is not None
                ],
                "tokens": [
                    int(event.request_id)
                    for event in events
                    if event.kind == "token" and event.request_id is not None
                ],
                "completed": [
                    int(event.request_id)
                    for event in events
                    if event.kind == "completed" and event.request_id is not None
                ],
                "physical_group_plans": plans,
                "physical_bucket": copy.deepcopy(loop.get("physical_bucket", {})),
                "request_counts": copy.deepcopy(loop.get("requests", {})),
            }
        )
        return events

    adapter.poll = MethodType(capture_poll, adapter)
    return reclaimed, timeline, capture_state


def _execute_pressure_workload(
    *,
    host: str,
    port: int,
    llm: LLM,
    runner: Any,
    batcher: Any,
    prompt_manifest: Mapping[str, Mapping[str, Any]],
    reference_tokens: Mapping[str, Sequence[int]],
    reclaimed: Mapping[int, _ReclaimedRow],
    capture_state: dict[str, Any],
    plan: LongContextPoolPlan,
    slos: SLOThresholds,
    idle_timeout_seconds: float,
    request_timeout_seconds: float,
) -> tuple[dict[str, Any], tuple[int, ...], tuple[int, ...]]:
    long_spec, candidate_spec = _pressure_specs(decode_tokens=plan.decode_tokens)
    before_ids = set(reclaimed)
    metrics_before = _metrics_snapshot(host, port)
    memory_before = _memory_snapshot("before_kv_pressure", runner)
    capture_state["label"] = "kv_pressure"
    start_event = threading.Event()
    workload_start = time.perf_counter() + 0.02
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            long_future = executor.submit(
                _stream_request,
                host,
                port,
                spec=long_spec,
                prompt=prompt_manifest[long_spec.oracle_key],
                start_event=start_event,
                workload_start=workload_start,
                served_model_name=_SERVED_MODEL_NAME,
                request_timeout_seconds=float(request_timeout_seconds),
            )
            start_event.set()
            admission_barrier = _wait_for_pressure_allocation(
                llm,
                runner,
                expected_pages=plan.pages_by_context[32_768],
                timeout_seconds=float(idle_timeout_seconds),
            )
            candidate_start = threading.Event()
            candidate_start.set()
            candidate_trace = executor.submit(
                _stream_request,
                host,
                port,
                spec=candidate_spec,
                prompt=prompt_manifest[candidate_spec.oracle_key],
                start_event=candidate_start,
                workload_start=time.perf_counter(),
                served_model_name=_SERVED_MODEL_NAME,
                request_timeout_seconds=float(request_timeout_seconds),
            ).result(timeout=float(request_timeout_seconds))
            long_trace = long_future.result(timeout=float(request_timeout_seconds))
        idle = _wait_for_idle(llm, batcher, timeout_seconds=float(idle_timeout_seconds))
    finally:
        capture_state["label"] = None
    traces = (long_trace, candidate_trace)
    workload_end = max(trace.completed_at for trace in traces)
    wall_seconds = max(0.0, workload_end - workload_start)
    metrics_after = _metrics_snapshot(host, port)
    memory_after = _memory_snapshot("after_kv_pressure", runner)
    new_reclaimed_ids = sorted(set(reclaimed) - before_ids)
    overrides = _reclaimed_overrides(
        traces,
        prompt_manifest=prompt_manifest,
        reclaimed_rows=reclaimed,
        new_reclaimed_ids=new_reclaimed_ids,
    )
    rows = [
        _join_trace(
            trace,
            prompt=prompt_manifest[trace.spec.oracle_key],
            expected=reference_tokens[trace.spec.oracle_key][: int(trace.spec.max_tokens)],
            reclaimed_rows=reclaimed,
            reclaimed_override=overrides.get(trace.spec.label),
        )
        for trace in traces
    ]
    summary = _evaluate_workload(
        "kv_pressure",
        rows,
        wall_seconds=wall_seconds,
        slos=slos,
        require_rejects=True,
    )
    row_by_label = {row.label: row for row in rows}
    metrics_delta = _counter_delta(metrics_before, metrics_after)
    metrics_exact = bool(
        metrics_delta.get("hipengine_requests_total", -1.0) == 2
        and metrics_delta.get("hipengine_request_completed_total", -1.0) == 1
        and metrics_delta.get("hipengine_request_rejected_total", -1.0) == 1
        and metrics_delta.get("hipengine_kv_pool_grow_failures_total", 0.0) >= 1
    )
    admission = _admission_metadata(candidate_trace)
    summary.update(
        {
            "passed": bool(
                summary["passed"]
                and metrics_exact
                and admission == _required_admission(plan)
            ),
            "long_outcome": row_by_label[long_spec.label].outcome,
            "candidate_outcome": row_by_label[candidate_spec.label].outcome,
            "candidate_error_code": candidate_trace.error_code,
            "candidate_error_status_code": candidate_trace.error_status_code,
            "candidate_done_sentinel": candidate_trace.done_sentinel,
            "candidate_admission": admission,
            "candidate_error_payload": copy.deepcopy(candidate_trace.error_payload),
            "admission_barrier": admission_barrier,
            "metrics": {
                "before": metrics_before,
                "after": metrics_after,
                "counter_delta": metrics_delta,
                "accounting_passed": metrics_exact,
            },
            "memory": {"before": memory_before, "after": memory_after},
            "ownership": {
                "new_reclaimed_request_ids": new_reclaimed_ids,
                "idle": idle,
            },
        }
    )
    if not metrics_exact:
        summary["failure_reasons"] = sorted(
            set([*summary["failure_reasons"], "pressure_server_counter_accounting_failed"])
        )
    if admission != _required_admission(plan):
        summary["failure_reasons"] = sorted(
            set([*summary["failure_reasons"], "pressure_admission_metadata_mismatch"])
        )
    long_row = overrides.get(long_spec.label)
    if long_row is None and long_trace.request_id is not None:
        long_row = reclaimed.get(int(long_trace.request_id))
    return (
        summary,
        () if long_row is None else tuple(long_row.block_ids),
        () if long_row is None else tuple(long_row.pointers),
    )


def _allocation_for_workload(
    summary: Mapping[str, Any],
    reclaimed: Mapping[int, _ReclaimedRow],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    ids = summary.get("ownership", {}).get("new_reclaimed_request_ids", ())
    candidates = [
        row
        for request_id in ids
        for row in (reclaimed.get(int(request_id)),)
        if row is not None and row.block_ids
    ]
    if not candidates:
        return (), ()
    selected = max(candidates, key=lambda row: len(row.block_ids))
    return tuple(selected.block_ids), tuple(selected.pointers)


def _llm_construction_kwargs(
    args: argparse.Namespace,
    model: Path,
) -> dict[str, Any]:
    return {
        "model": model,
        "backend": str(args.backend),
        "quant": str(args.quant),
        "max_active_requests": int(args.max_active_requests),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    model = Path(args.model).expanduser().resolve()
    if not model.is_file():
        raise ValueError(f"model does not exist: {model}")
    if int(args.max_active_requests) < 3:
        raise ValueError("max-active-requests must be at least three for the mixed-context gate")
    longer = None if int(args.longer_context_tokens) == 0 else int(args.longer_context_tokens)
    plan = build_pool_plan(
        decode_tokens=int(args.decode_tokens),
        longer_context_tokens=longer,
    )
    all_workloads = build_workload_specs(
        decode_tokens=int(args.decode_tokens),
        longer_context_tokens=longer,
        backend=str(args.backend),
    )
    workload_names = _parse_workload_names(
        args.workloads,
        available=tuple(all_workloads),
    )
    workloads = {name: all_workloads[name] for name in workload_names}
    run_pressure = not bool(args.skip_pressure)
    complete_packet = bool(
        workload_names == tuple(all_workloads)
        and run_pressure
        and longer is not None
    )
    pressure_specs = _pressure_specs(decode_tokens=int(args.decode_tokens))
    all_specs = [spec for rows in workloads.values() for spec in rows]
    if run_pressure:
        all_specs.extend(pressure_specs)
    max_prompt = max(spec.prompt_length for spec in all_specs)
    max_output = max(spec.max_tokens for spec in all_specs)
    max_sequence_length = max_prompt + max_output + 2
    compiler_version = _read_compiler_version(args.compiler_version_file)
    if args.require_cached_build and compiler_version is None:
        raise ValueError("require-cached-build requires compiler-version-file")
    slos = SLOThresholds(
        queue_p99_seconds=float(args.slo_queue_p99_seconds),
        ttft_p95_seconds=float(args.slo_ttft_p95_seconds),
        itl_p99_seconds=float(args.slo_itl_p99_seconds),
        end_to_end_p95_seconds=float(args.slo_end_to_end_p95_seconds),
    )
    env = {
        **_EXACT_ENV,
        "HIPENGINE_GGUF_GDN_PREFILL_MODE": str(args.gdn_mode),
        "HIPENGINE_PREFILL_DECODE_POLICY": "token_budget",
        "HIPENGINE_MAX_ACTIVE_REQUESTS": str(int(args.max_active_requests)),
        "HIPENGINE_MAX_PENDING_REQUESTS": str(int(args.max_pending_requests)),
        "HIPENGINE_MAX_PREFILL_CHUNK_TOKENS": str(int(args.prefill_chunk_tokens)),
        "HIPENGINE_KV_POOL_INITIAL_PAGES": str(plan.initial_pages),
        "HIPENGINE_KV_POOL_LOW_WATER_PAGES": str(plan.low_water_pages),
        "HIPENGINE_KV_POOL_HIGH_WATER_PAGES": str(plan.mixed_high_water_pages),
        "HIPENGINE_KV_POOL_CHUNK_PAGES": str(plan.chunk_pages),
        "HIPENGINE_KV_POOL_IDLE_GRACE_SECONDS": "0",
        "HIPENGINE_PREFIX_CACHE": "off",
    }
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    source_dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True
        ).strip()
    )
    started_at = time.perf_counter()
    workload_results: dict[str, dict[str, Any]] = {}
    pressure_result: dict[str, Any] = {}
    pressure_block_ids: tuple[int, ...] = ()
    pressure_pointers: tuple[int, ...] = ()
    regrow_block_ids: tuple[int, ...] = ()
    regrow_pointers: tuple[int, ...] = ()
    with _temporary_environment(env):
        llm = LLM(**_llm_construction_kwargs(args, model))
        try:
            adapter = llm._get_text_generator()
            llm.prepare(
                max_sequence_length=max_sequence_length,
                sampling_params=SamplingParams(max_tokens=max_output),
            )
            runner = adapter._runner
            prompt_rows = _prompt_manifest(runner.generator.tokenizer, all_specs)
            from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession

            reference_session = Qwen35GGUFResidentSession(
                model,
                backend=str(args.backend),
                runtime=runner._shared_runner.runtime,
                shared_runner=runner._shared_runner,
                max_sequence_length=max_sequence_length,
                use_wmma_prefill=True,
                use_gemv_decode=True,
                compiler_version=compiler_version,
                require_cached_build=bool(args.require_cached_build),
            )
            try:
                reference_runs = {
                    key: _run_reference(
                        reference_session,
                        row["token_ids"],
                        max(
                            spec.max_tokens
                            for spec in all_specs
                            if spec.oracle_key == key
                        ),
                    )
                    for key, row in sorted(prompt_rows.items())
                }
            finally:
                reference_session.close()
            reference_tokens = {
                key: tuple(int(token) for token in run.generated_tokens)
                for key, run in reference_runs.items()
            }
            reclaimed, timeline, capture_state = _capture_runtime(adapter, runner)
            app = create_app(
                ServerConfig(
                    model=str(model),
                    backend=str(args.backend),
                    quant=str(args.quant),
                    served_model_name=_SERVED_MODEL_NAME,
                    eager_load=False,
                    startup_chat_smoke=False,
                    startup_scratch_probe=False,
                    metrics="prometheus",
                    generation_batch_window_ms=float(args.batch_window_ms),
                    max_context_tokens=max_sequence_length,
                    max_active_requests=int(args.max_active_requests),
                    max_queued_requests=int(args.max_queued_requests),
                    queue_retry_after_seconds=int(args.queue_retry_after_seconds),
                    stream_queue_max_chunks=int(args.stream_queue_max_chunks),
                    shutdown_grace_seconds=10.0,
                ),
                llm=llm,
            )
            batcher = app.state.hipengine_generation_batcher
            baseline_memory = _memory_snapshot("prepared_baseline", runner)
            graph_before = _graph_totals(runner.observability_snapshot())
            with _LocalUvicorn(app) as server:
                ready = _http_json("127.0.0.1", server.port, "GET", "/ready")
                if not bool(ready.get("ready")):
                    raise RuntimeError(f"server readiness failed: {ready}")
                for name, specs in workloads.items():
                    if name == "graph_regrow_32k_c1":
                        continue
                    summary = _execute_workload(
                        name,
                        specs,
                        host="127.0.0.1",
                        port=server.port,
                        llm=llm,
                        runner=runner,
                        batcher=batcher,
                        prompt_manifest=prompt_rows,
                        reference_tokens=reference_tokens,
                        reclaimed=reclaimed,
                        timeline=timeline,
                        capture_state=capture_state,
                        slos=slos,
                        stream_queue_limit=int(args.stream_queue_max_chunks),
                        idle_timeout_seconds=float(args.idle_timeout_seconds),
                        request_timeout_seconds=float(args.request_timeout_seconds),
                    )
                    workload_results[name] = summary
                    print(
                        f"{name}: passed={summary['passed']} "
                        f"exact_tok_s={float(summary['throughput']['exact_generated_tokens_per_second'] or 0.0):.6f}",
                        file=sys.stderr,
                        flush=True,
                    )
                if run_pressure:
                    pressure_config = replace(
                        adapter._loop.config,
                        kv_pool_initial_pages=plan.initial_pages,
                        kv_pool_low_water_pages=plan.low_water_pages,
                        kv_pool_high_water_pages=plan.pressure_high_water_pages,
                        kv_pool_chunk_pages=plan.chunk_pages,
                        kv_pool_idle_grace_seconds=0.0,
                    )
                    active_driver = llm._get_text_generator()
                    reconfigure = getattr(
                        active_driver,
                        "reconfigure_engine_loop",
                        None,
                    )
                    if not callable(reconfigure):
                        raise RuntimeError(
                            "loaded engine service cannot serialize KV reconfiguration"
                        )
                    reconfigure(pressure_config)
                    pressure_result, pressure_block_ids, pressure_pointers = (
                        _execute_pressure_workload(
                            host="127.0.0.1",
                            port=server.port,
                            llm=llm,
                            runner=runner,
                            batcher=batcher,
                            prompt_manifest=prompt_rows,
                            reference_tokens=reference_tokens,
                            reclaimed=reclaimed,
                            capture_state=capture_state,
                            plan=plan,
                            slos=slos,
                            idle_timeout_seconds=float(args.idle_timeout_seconds),
                            request_timeout_seconds=float(args.request_timeout_seconds),
                        )
                    )
                    print(
                        f"kv_pressure: passed={pressure_result['passed']} "
                        f"outcomes={pressure_result['outcomes']}",
                        file=sys.stderr,
                        flush=True,
                    )
                if "graph_regrow_32k_c1" in workloads:
                    summary = _execute_workload(
                        "graph_regrow_32k_c1",
                        workloads["graph_regrow_32k_c1"],
                        host="127.0.0.1",
                        port=server.port,
                        llm=llm,
                        runner=runner,
                        batcher=batcher,
                        prompt_manifest=prompt_rows,
                        reference_tokens=reference_tokens,
                        reclaimed=reclaimed,
                        timeline=timeline,
                        capture_state=capture_state,
                        slos=slos,
                        stream_queue_limit=int(args.stream_queue_max_chunks),
                        idle_timeout_seconds=float(args.idle_timeout_seconds),
                        request_timeout_seconds=float(args.request_timeout_seconds),
                    )
                    workload_results["graph_regrow_32k_c1"] = summary
                    regrow_block_ids, regrow_pointers = _allocation_for_workload(
                        summary,
                        reclaimed,
                    )
                    print(
                        f"graph_regrow_32k_c1: passed={summary['passed']}",
                        file=sys.stderr,
                        flush=True,
                    )
                final_idle = _wait_for_idle(
                    llm,
                    batcher,
                    timeout_seconds=float(args.idle_timeout_seconds),
                )
                final_metrics = _metrics_snapshot("127.0.0.1", server.port)
            graph_after_snapshot = runner.observability_snapshot()
            graph_after = _graph_totals(graph_after_snapshot)
            final_pool = _final_pool_from_idle_snapshot(final_idle)
            final_memory = _memory_snapshot("final", runner)
            resolved_backend = str(runner.generator.backend)
            target_arch = str(runner._shared_runner.target_arch)
        finally:
            llm.close()
    memory_recovery = _memory_recovery_gate(
        baseline_memory,
        final_memory,
        tracked_tolerance_bytes=int(args.tracked_memory_tolerance_mib) * 1024 * 1024,
    )
    graph_lifecycle_delta = _graph_delta(graph_before, graph_after)
    selected_passed = bool(
        workload_results
        and set(workload_results) == set(workloads)
        and all(summary.get("passed") is True for summary in workload_results.values())
    )
    pressure_passed = bool(not run_pressure or pressure_result.get("passed") is True)
    final_ownership_passed = bool(
        int(final_idle["snapshot"]["loop"]["requests"]["active"]) == 0
        and int(final_idle["snapshot"]["loop"]["requests"]["pending"]) == 0
        and int(final_idle["snapshot"]["runner"]["model_runner"]["active_requests"]) == 0
        and int(final_idle["generation_queue_depth"]) == 0
        and int(final_idle["generation_active_requests"]) == 0
    )
    packet_gate = (
        evaluate_packet(
            plan=plan,
            workloads=workload_results,
            pressure=pressure_result,
            final_pool=final_pool,
            graph_delta=graph_lifecycle_delta,
            pressure_block_ids=pressure_block_ids,
            regrow_block_ids=regrow_block_ids,
        )
        if complete_packet
        else {"passed": True, "failure_reasons": []}
    )
    passed = bool(
        selected_passed
        and pressure_passed
        and final_ownership_passed
        and memory_recovery["passed"]
        and not source_dirty
        and packet_gate["passed"]
    )
    command = [sys.executable, "scripts/gguf_long_context_pressure_gate.py", *sys.argv[1:]]
    scope = _artifact_backend_scope(resolved_backend, target_arch)
    provenance = collect_artifact_provenance(
        repo_root=REPO_ROOT,
        configured_backend=str(args.backend),
        resolved_backend=resolved_backend,
        target_arch=target_arch,
        model_path=model,
        quant=str(args.quant),
        kv_dtype="bf16",
        command=command,
        environment={
            **{key: os.environ.get(key) for key in _PROVENANCE_ENV_KEYS},
            **env,
        },
        build_profile=f"{scope}_gguf_long_context_pressure",
        timing_protocol=(
            "one prepared model; real localhost Uvicorn SSE; independent c1 token oracle; "
            "concurrent 1K/4K/32K/mixed/longer rows; forced global-page reject; "
            "generation rebuild and changed page-table replay"
        ),
        warmups=0,
        repetitions=1,
        profiler={"used": False, "reason": "server-wall correctness/memory-pressure gate"},
        hipcc_version=compiler_version,
    )
    return {
        "schema": 1,
        "kind": f"{scope}_gguf_long_context_memory_pressure_gate",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": (
            "accepted" if passed and complete_packet else "measurement_complete" if passed else "failed"
        ),
        "passed": passed,
        "complete_packet": complete_packet,
        "performance_claim": bool(passed and complete_packet),
        "source": {"commit": source_commit, "dirty": source_dirty},
        "provenance": provenance,
        "configuration": {
            "model": str(model),
            "backend": str(args.backend),
            "quant": str(args.quant),
            "kv_dtype": "bf16",
            "decode_tokens": int(args.decode_tokens),
            "graph_decode_tokens": _graph_output_tokens(
                str(args.backend), int(args.decode_tokens)
            ),
            "max_sequence_length": max_sequence_length,
            "max_active_requests": int(args.max_active_requests),
            "prefill_decode_policy": "token_budget",
            "prefill_chunk_tokens": int(args.prefill_chunk_tokens),
            "pool_plan": plan.to_json_dict(),
            "slo_thresholds": asdict(slos),
        },
        "prompt_manifest": [
            {key: value for key, value in row.items() if key not in {"text", "token_ids"}}
            for _oracle_key, row in sorted(prompt_rows.items())
        ],
        "reference_tokens": {
            key: list(value) for key, value in sorted(reference_tokens.items())
        },
        "workloads": workload_results,
        "pressure": pressure_result,
        "gates": {
            "packet": packet_gate,
            "selected_workloads_passed": selected_passed,
            "pressure_passed": pressure_passed,
            "final_ownership_passed": final_ownership_passed,
            "memory_recovery": memory_recovery,
            "clean_source_passed": not source_dirty,
        },
        "pool_lifecycle": {
            "final": final_pool,
            "pressure_block_ids": list(pressure_block_ids),
            "pressure_pointers": list(pressure_pointers),
            "regrow_block_ids": list(regrow_block_ids),
            "regrow_pointers": list(regrow_pointers),
            "regrow_page_table_changed": bool(
                pressure_block_ids
                and regrow_block_ids
                and tuple(pressure_block_ids) != tuple(regrow_block_ids)
            ),
        },
        "graph_lifecycle": {
            "before": graph_before,
            "after": graph_after,
            "delta": graph_lifecycle_delta,
            "final_snapshot": graph_after_snapshot.get("graph_buckets", {}),
        },
        "final_metrics": final_metrics,
        "final_ownership": final_idle,
        "baseline_memory": baseline_memory,
        "final_memory": final_memory,
        "command": shlex.join(command),
        "elapsed_seconds": time.perf_counter() - started_at,
        "limitations": [
            "This packet covers greedy Q4_K_M with BF16 device KV only.",
            "Uniform INT8 and tail4 Hadamard INT8 remain fail-closed in the continuous owner because deferred dynamic KV and packed c>N kernels currently require BF16.",
            "128K is not retried: the retained gfx1151 production artifact documents a repeated-lifecycle firmware/runtime stall; 64K is the default feasible longer-context row.",
            "Generation-2 token-budget scheduling is used; sustained arrival-rate SLO selection remains the separate production-load gate.",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--backend", choices=_SUPPORTED_BACKENDS, default="hip_gfx1151")
    parser.add_argument("--quant", default="gguf_q4_k_m")
    parser.add_argument(
        "--gdn-mode",
        default="exact",
        help="GGUF GDN prefill execution mode (exact or a qualified profile route)",
    )
    parser.add_argument("--decode-tokens", type=int, default=32)
    parser.add_argument(
        "--longer-context-tokens",
        type=int,
        default=65_536,
        help="Feasible context above 32K; use 0 to disable for a diagnostic.",
    )
    default_workloads = ",".join(
        build_workload_specs(decode_tokens=32, longer_context_tokens=65_536)
    )
    parser.add_argument(
        "--workloads",
        default=default_workloads,
        help="Comma-separated workload subset; subsets are diagnostic only.",
    )
    parser.add_argument("--skip-pressure", action="store_true")
    parser.add_argument("--max-active-requests", type=int, default=3)
    parser.add_argument("--max-pending-requests", type=int, default=8)
    parser.add_argument("--max-queued-requests", type=int, default=8)
    parser.add_argument("--stream-queue-max-chunks", type=int, default=64)
    parser.add_argument("--queue-retry-after-seconds", type=int, default=1)
    parser.add_argument("--batch-window-ms", type=float, default=50.0)
    parser.add_argument("--prefill-chunk-tokens", type=int, default=256)
    parser.add_argument("--idle-timeout-seconds", type=float, default=600.0)
    parser.add_argument("--request-timeout-seconds", type=float, default=600.0)
    parser.add_argument("--slo-queue-p99-seconds", type=float, default=120.0)
    parser.add_argument("--slo-ttft-p95-seconds", type=float, default=240.0)
    parser.add_argument("--slo-itl-p99-seconds", type=float, default=5.0)
    parser.add_argument("--slo-end-to-end-p95-seconds", type=float, default=300.0)
    parser.add_argument("--tracked-memory-tolerance-mib", type=int, default=128)
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--json", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    payload = run(args)
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n", encoding="utf-8")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
