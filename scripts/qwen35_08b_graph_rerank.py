#!/usr/bin/env python3
"""Profile Qwen3.5-0.8B semantic graph roles and replay API residual."""

from __future__ import annotations

import hashlib
import json
import os
import statistics
import subprocess
import time
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np

from hipengine.benchmark.correctness import evaluate_logits
from hipengine.core.memory import copy_device_to_host, host_array_ptr
from hipengine.kernels.hip_gfx1100.runtime import wall_clock_rate_khz
from hipengine.runtime.prefill import PrefillConfig
import hipengine.runtime.gguf_decode_graph as graph_module
from hipengine.runtime.qwen35_gguf_runner import (
    Qwen35GGUFResidentSession,
    _HipWallClockStageRecorder,
)
from scripts.qwen35_gguf_bench import _memory_snapshot

COMPILER_VERSION_FILE = Path("/tmp/d08-c0/hipcc-version.txt")
PROMPT = [9707] * 512
STEPS = 128
MODELS = {
    "q4": Path("/models/gguf/Qwen3.5-0.8B-Q4_K_M.gguf"),
    "q8": Path("/models/gguf/Qwen3.5-0.8B-Q8_0.gguf"),
}
OUT = Path("/tmp/d08-review/current-graph-rerank.json")
ROOT = Path(__file__).resolve().parents[1]
PRIOR_ISOLATED_OWNER_SUM_MS = 6.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def stats(values: list[float]) -> dict[str, float | list[float]]:
    return {
        "median": float(statistics.median(values)),
        "mean": float(statistics.mean(values)),
        "min": float(min(values)),
        "max": float(max(values)),
        "p95": float(np.percentile(np.asarray(values), 95)),
        "samples": values,
    }


def compact_memory(snapshot: dict[str, object]) -> dict[str, object]:
    tracked = snapshot["tracked"]
    return {
        "tracked_current_bytes": int(tracked["current_allocated_bytes"]),
        "tracked_peak_bytes": int(tracked["peak_allocated_bytes"]),
        "owned_session_bytes": int(snapshot["owned_session_bytes"]),
        "hip_used_bytes": int(snapshot["hip"]["used_bytes"]),
    }


def prepare(session: Qwen35GGUFResidentSession):
    assert session.runtime is not None
    session.reset()
    first = session.prefill(PROMPT, use_bulk=True, bulk_attention_mode="bulk", return_logits=False)
    warm = session.step(int(first.token_id), return_logits=False)
    session.runtime.device_synchronize()
    return first, warm


def eager_run(session: Qwen35GGUFResidentSession) -> dict[str, object]:
    assert session.runtime is not None
    _, warm = prepare(session)
    current = int(warm.token_id)
    token_ids = []
    host_ms = []
    final = None
    for index in range(STEPS):
        started = time.perf_counter_ns()
        final = session.step(current, return_logits=index == STEPS - 1)
        host_ms.append((time.perf_counter_ns() - started) / 1_000_000.0)
        current = int(final.token_id)
        token_ids.append(current)
    assert final is not None and final.logits is not None
    return {
        "token_ids": token_ids,
        "final_logits": np.ascontiguousarray(final.logits, dtype=np.float32),
        "host_ms": host_ms,
    }


def graph_node_inventory(runtime, graph_handle: int) -> dict[str, object]:
    nodes = runtime.graph_nodes(graph_handle)
    edges = runtime.graph_edges(graph_handle)
    type_counts = Counter(runtime.graph_node_type(node) for node in nodes)
    kernel_names = []
    for node in nodes:
        if runtime.graph_node_type(node) != 0:
            continue
        params = runtime.graph_kernel_node_params(node)
        kernel_names.append(runtime.kernel_name_ref_by_ptr(int(params.func or 0)))
    return {
        "nodes": len(nodes),
        "edges": len(edges),
        "node_types": {str(key): value for key, value in sorted(type_counts.items())},
        "kernel_nodes": len(kernel_names),
        "kernel_name_counts": dict(sorted(Counter(kernel_names).items())),
    }


def timed_production_graph(session: Qwen35GGUFResidentSession) -> dict[str, object]:
    assert session.runtime is not None
    runtime = session.runtime
    prepare(session)
    before = compact_memory(_memory_snapshot("before_graph", runtime, session))
    started = time.perf_counter_ns()
    graph = session.capture_decode_graph(
        position=int(session.position),
        steps_per_replay=1,
        max_replay_steps=STEPS,
        record_steps=0,
    )
    capture_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    after = compact_memory(_memory_snapshot("after_graph", runtime, session))
    inventory = graph_node_inventory(runtime, graph.graph)
    host_ms = []
    launch_ms = []
    sync_ms = []
    original_launch = runtime.graph_launch
    original_sync = runtime.stream_synchronize

    def marked_launch(*args, **kwargs):
        t0 = time.perf_counter_ns()
        result = original_launch(*args, **kwargs)
        launch_ms.append((time.perf_counter_ns() - t0) / 1_000_000.0)
        return result

    def marked_sync(*args, **kwargs):
        t0 = time.perf_counter_ns()
        result = original_sync(*args, **kwargs)
        sync_ms.append((time.perf_counter_ns() - t0) / 1_000_000.0)
        return result

    runtime.graph_launch = marked_launch
    runtime.stream_synchronize = marked_sync
    try:
        for _ in range(STEPS):
            t0 = time.perf_counter_ns()
            graph.replay(1)
            host_ms.append((time.perf_counter_ns() - t0) / 1_000_000.0)
        sample = graph.read_sample(return_logits=True)
        transport = graph.transport_provenance()
    finally:
        runtime.graph_launch = original_launch
        runtime.stream_synchronize = original_sync
        graph.close()
    assert sample.logits is not None
    return {
        "capture_ms": capture_ms,
        "host_ms": host_ms,
        "graph_launch_api_ms": launch_ms,
        "stream_synchronize_api_ms": sync_ms,
        "final_token_id": int(sample.token_id),
        "final_logits": np.ascontiguousarray(sample.logits, dtype=np.float32),
        "transport": transport,
        "node_inventory": inventory,
        "memory_before": before,
        "memory_after": after,
    }


def recorded_graph_trajectory(session: Qwen35GGUFResidentSession) -> dict[str, object]:
    assert session.runtime is not None
    prepare(session)
    started = time.perf_counter_ns()
    graph = session.capture_decode_graph(
        position=int(session.position),
        steps_per_replay=1,
        max_replay_steps=STEPS,
        record_steps=STEPS,
    )
    capture_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    try:
        graph.replay(STEPS)
        token_ids = graph.read_generated_token_ids(STEPS)
        sample = graph.read_sample(return_logits=True)
        transport = graph.transport_provenance()
    finally:
        graph.close()
    assert sample.logits is not None
    return {
        "capture_ms": capture_ms,
        "token_ids": token_ids,
        "final_token_id": int(sample.token_id),
        "final_logits": np.ascontiguousarray(sample.logits, dtype=np.float32),
        "transport": transport,
    }


def recorder_stage_values(recorder: _HipWallClockStageRecorder, rate: int) -> dict[str, float]:
    assert recorder._buffer is not None
    ticks = np.empty(recorder._count, dtype=np.uint64)
    copy_device_to_host(
        host_array_ptr(ticks),
        recorder._buffer,
        nbytes=int(ticks.nbytes),
        runtime=recorder.runtime,
    )
    values: dict[str, float] = {}
    for names, start_index, stop_index in recorder._intervals:
        elapsed = (int(ticks[stop_index]) - int(ticks[start_index])) / rate
        for name in names:
            values[name] = values.get(name, 0.0) + float(elapsed)
    return values


def topological_order(runtime, graph_handle: int) -> tuple[int, ...]:
    nodes = runtime.graph_nodes(graph_handle)
    edges = runtime.graph_edges(graph_handle)
    successors: dict[int, list[int]] = {node: [] for node in nodes}
    indegree = {node: 0 for node in nodes}
    for source, destination in edges:
        successors[source].append(destination)
        indegree[destination] += 1
    ready = deque(node for node in nodes if indegree[node] == 0)
    order = []
    while ready:
        node = ready.popleft()
        order.append(node)
        for destination in successors[node]:
            indegree[destination] -= 1
            if indegree[destination] == 0:
                ready.append(destination)
    if len(order) != len(nodes):
        raise RuntimeError("captured graph is not acyclic")
    return tuple(order)


def instrumented_node_inventory(runtime, graph_handle: int, recorder: _HipWallClockStageRecorder):
    order = topological_order(runtime, graph_handle)
    per_stage: dict[str, list[dict[str, object]]] = defaultdict(list)
    marker_count = 0
    between: list[dict[str, object]] = []
    prefix: list[dict[str, object]] = []
    for node in order:
        node_type = runtime.graph_node_type(node)
        name = f"node_type_{node_type}"
        if node_type == 0:
            params = runtime.graph_kernel_node_params(node)
            name = runtime.kernel_name_ref_by_ptr(int(params.func or 0))
        row = {"type": node_type, "name": name}
        if node_type == 0 and "wall_clock_mark_u64" in name:
            if marker_count == 0:
                prefix = list(between)
            else:
                stage = recorder._intervals[marker_count - 1][0][0]
                per_stage[stage].extend(between)
            between = []
            marker_count += 1
        else:
            between.append(row)
    if marker_count != recorder._count:
        raise RuntimeError(f"expected {recorder._count} markers, observed {marker_count}")
    return {
        "nodes": len(order),
        "marker_nodes": marker_count,
        "prefix_nodes": prefix,
        "suffix_nodes": between,
        "stage_nodes": {
            stage: {
                "nodes": len(rows),
                "node_types": dict(sorted(Counter(str(row["type"]) for row in rows).items())),
                "kernel_name_counts": dict(sorted(Counter(str(row["name"]) for row in rows if row["type"] == 0).items())),
            }
            for stage, rows in per_stage.items()
        },
    }


def instrumented_graph(session: Qwen35GGUFResidentSession) -> dict[str, object]:
    assert session.runtime is not None
    runtime = session.runtime
    prepare(session)
    recorder = _HipWallClockStageRecorder(
        runtime,
        enabled=True,
        stream=0,
        library=session._runtime_state_library,
        capacity=4096,
    )
    original_set = session._set_token_embedding_from_ptr
    original_hidden = session._run_current_hidden_to_final_hidden
    original_sample = session._sample_device_from_hidden
    original_advance = graph_module.advance_decode_position_i64
    started_recorder = False

    def marked_set(token_ptr: int, *, stream: int = 0):
        nonlocal started_recorder
        recorder.stream = int(stream)
        if not started_recorder:
            recorder.start()
            recorder.mark("decode_marker_baseline")
            started_recorder = True
        result = original_set(token_ptr, stream=stream)
        recorder.mark("decode_input_setup")
        return result

    def marked_hidden(**kwargs):
        kwargs["gpu_stage_recorder"] = recorder
        return original_hidden(**kwargs)

    def marked_sample(hidden_ptr: int, *, stream: int = 0):
        result = original_sample(hidden_ptr, stream=stream)
        recorder.mark("decode_lm_head_sample")
        return result

    def marked_advance(*args, **kwargs):
        result = original_advance(*args, **kwargs)
        recorder.mark("decode_state_advance")
        return result

    session._set_token_embedding_from_ptr = marked_set
    session._run_current_hidden_to_final_hidden = marked_hidden
    session._sample_device_from_hidden = marked_sample
    graph_module.advance_decode_position_i64 = marked_advance
    try:
        started = time.perf_counter_ns()
        graph = session.capture_decode_graph(
            position=int(session.position),
            steps_per_replay=1,
            max_replay_steps=STEPS,
            record_steps=0,
        )
        capture_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    finally:
        session._set_token_embedding_from_ptr = original_set
        session._run_current_hidden_to_final_hidden = original_hidden
        session._sample_device_from_hidden = original_sample
        graph_module.advance_decode_position_i64 = original_advance
    rate = wall_clock_rate_khz(library=session._runtime_state_library, runtime=runtime)
    inventory = instrumented_node_inventory(runtime, graph.graph, recorder)
    host_ms = []
    stage_samples: dict[str, list[float]] = defaultdict(list)
    try:
        for _ in range(STEPS):
            t0 = time.perf_counter_ns()
            graph.replay(1)
            host_ms.append((time.perf_counter_ns() - t0) / 1_000_000.0)
            values = recorder_stage_values(recorder, rate)
            for name, value in values.items():
                stage_samples[name].append(value)
        sample = graph.read_sample(return_logits=True)
        transport = graph.transport_provenance()
    finally:
        graph.close()
        recorder.close()
    assert sample.logits is not None
    return {
        "capture_ms": capture_ms,
        "host_ms": host_ms,
        "stage_samples_ms": dict(stage_samples),
        "final_token_id": int(sample.token_id),
        "final_logits": np.ascontiguousarray(sample.logits, dtype=np.float32),
        "transport": transport,
        "node_inventory": inventory,
    }


def strip_arrays(row: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in row.items() if key not in {"token_ids", "final_logits", "host_ms", "graph_launch_api_ms", "stream_synchronize_api_ms", "stage_samples_ms"}}


def run_quant(label: str, model: Path) -> dict[str, object]:
    with Qwen35GGUFResidentSession(
        model,
        backend="hip_gfx1151",
        compiler_version=COMPILER_VERSION_FILE.read_text(encoding="utf-8"),
        require_cached_build=True,
        max_sequence_length=768,
        token_embedding_placement="device",
        use_wmma_prefill=True,
        use_gemv_decode=True,
        prefill_config=PrefillConfig(attn_aotriton_min_tokens=512),
    ) as session:
        eager = eager_run(session)
        recorded = recorded_graph_trajectory(session)
        production = timed_production_graph(session)
        instrumented = instrumented_graph(session)

    quality_recorded = evaluate_logits(
        eager["final_logits"], recorded["final_logits"], kl_threshold=0.05, top1_threshold=0.90
    )
    quality_production = evaluate_logits(
        eager["final_logits"], production["final_logits"], kl_threshold=0.05, top1_threshold=0.90
    )
    quality_instrumented = evaluate_logits(
        eager["final_logits"], instrumented["final_logits"], kl_threshold=0.05, top1_threshold=0.90
    )
    stage_stats = {name: stats(values) for name, values in instrumented["stage_samples_ms"].items()}
    stage_medians = {name: float(value["median"]) for name, value in stage_stats.items()}
    stage_sum = float(sum(stage_medians.values()))
    production_wall_ms = float(statistics.median(production["host_ms"]))
    launch_api_ms = float(statistics.median(production["graph_launch_api_ms"]))
    synchronize_api_ms = float(statistics.median(production["stream_synchronize_api_ms"]))
    instrumented_wall_ms = float(statistics.median(instrumented["host_ms"]))
    same_session_device_stage_ms = stage_sum
    result = {
        "model": str(model),
        "model_sha256": sha256(model),
        "eager": {**strip_arrays(eager), "host_ms": stats(eager["host_ms"])},
        "recorded_graph": {
            **strip_arrays(recorded),
            "trajectory_exact": eager["token_ids"] == recorded["token_ids"],
            "quality": {"kl_max": float(quality_recorded.kl_max), "top1_agreement": float(quality_recorded.top1_agreement)},
        },
        "production_graph": {
            **strip_arrays(production),
            "host_ms": stats(production["host_ms"]),
            "graph_launch_api_ms": stats(production["graph_launch_api_ms"]),
            "stream_synchronize_api_ms": stats(production["stream_synchronize_api_ms"]),
            "quality": {"kl_max": float(quality_production.kl_max), "top1_agreement": float(quality_production.top1_agreement)},
        },
        "instrumented_graph": {
            **strip_arrays(instrumented),
            "host_ms": stats(instrumented["host_ms"]),
            "stage_ms": stage_stats,
            "stage_median_sum_ms": stage_sum,
            "host_minus_stage_median_ms": float(instrumented_wall_ms - stage_sum),
            "quality": {"kl_max": float(quality_instrumented.kl_max), "top1_agreement": float(quality_instrumented.top1_agreement)},
        },
        "graph_replay_residual": {
            "production_wall_median_ms_per_token": production_wall_ms,
            "graph_launch_api_median_ms": launch_api_ms,
            "stream_synchronize_call_median_ms_including_device_wait": synchronize_api_ms,
            "instrumented_wall_median_ms_per_token": instrumented_wall_ms,
            "same_session_device_stage_median_sum_ms_per_token": same_session_device_stage_ms,
            "instrumented_host_minus_device_stage_ms": instrumented_wall_ms - same_session_device_stage_ms,
            "instrumented_device_stage_coverage": same_session_device_stage_ms / instrumented_wall_ms,
            "prior_k4_isolated_owner_sum_estimate_ms_per_token": PRIOR_ISOLATED_OWNER_SUM_MS,
            "production_wall_minus_prior_isolated_estimate_ms": production_wall_ms - PRIOR_ISOLATED_OWNER_SUM_MS,
            "interpretation_contract": (
                "graph_launch is asynchronous; stream_synchronize includes device execution. "
                "The host-minus-same-session-device-stage value is the API/Python residual. "
                "The prior ~6 ms isolated-owner sum is retained only to test whether its ~2.3 ms gap "
                "is real API overhead or isolated-microbench undercount."
            ),
        },
    }
    print(label, json.dumps({
        "eager_median_ms": result["eager"]["host_ms"]["median"],
        "graph_median_ms": result["production_graph"]["host_ms"]["median"],
        "graph_launch_ms": result["production_graph"]["graph_launch_api_ms"]["median"],
        "graph_sync_ms": result["production_graph"]["stream_synchronize_api_ms"]["median"],
        "instrumented_host_ms": result["instrumented_graph"]["host_ms"]["median"],
        "stage_sum_ms": stage_sum,
        "trajectory_exact": result["recorded_graph"]["trajectory_exact"],
        "nodes": result["production_graph"]["node_inventory"]["nodes"],
    }, indent=2), flush=True)
    return result


def main() -> int:
    payload = {
        "schema": 1,
        "task": "D08 post-review current graph rerank and API-residual closure",
        "status": "diagnostic",
        "commit": git("rev-parse", "HEAD"),
        "repo_status_porcelain": git("status", "--porcelain=v1"),
        "method": "128 token-level samples per route; production graph API wrappers plus graph-captured same-stream device wall-clock stage boundaries; separate recording graph for exact trajectory",
        "prompt": {"token_id": 9707, "length": 512, "decode_steps": STEPS},
        "environment": {
            "HIPENGINE_HIP_ARCH": os.environ.get("HIPENGINE_HIP_ARCH"),
            "HIPENGINE_GGUF_HOST_TOKEN_EMBEDDING": os.environ.get("HIPENGINE_GGUF_HOST_TOKEN_EMBEDDING"),
        },
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
        "compiler_version_file": str(COMPILER_VERSION_FILE),
        "compiler_version_sha256": sha256(COMPILER_VERSION_FILE),
        "prior_isolated_owner_sum_source": "worklog/entries/20260814T200358.246968Z-lhl-qwen35-08b-gfx1151-x2k4-c076e0.md",
        "results": {label: run_quant(label, model) for label, model in MODELS.items()},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
