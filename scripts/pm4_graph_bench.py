#!/usr/bin/env python3
"""Counterbalanced same-session HIP-graph/AQL/PM4 replay benchmark.

This is a focused P6 transport diagnostic, not a promotion benchmark.  It uses
one loaded GGUF model, stable graph-bound pointers, one captured executable per
transport, exact reset/rearm before every p512/dN window, and separates host
submission-call wall from synchronized replay wall.  Capture cost is reported
independently and no submit-plus-queue-recreate lifecycle stress is performed.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import statistics
import sys
import time
from contextlib import AbstractContextManager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.gguf_decode_graph_g5 import (  # noqa: E402
    _canonical_sha256,
    _capture_checkpoint,
    _prefill,
)

DEFAULT_MODEL = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
_DEFAULT_TRANSPORTS = ("hipgraph", "aql", "pm4")
_BENCHMARK_MODES = (
    *_DEFAULT_TRANSPORTS,
    "pm4_stateful",
    "pm4_stateful_local",
    "pm4_timestamps",
)
_DEFAULT_MEMORY_RECOVERY_TOLERANCE = 64 * 1024 * 1024


def _transport_spec(mode: str) -> tuple[str, bool | None, bool | None]:
    """Map one benchmark label to the production transport and PM4 encoder."""

    if mode == "pm4_stateful_local":
        return "pm4", True, True
    if mode == "pm4_stateful":
        return "pm4", True, False
    if mode in {"pm4", "pm4_timestamps"}:
        return "pm4", True, True
    if mode in {"hipgraph", "aql"}:
        return mode, None, None
    raise ValueError(f"unknown benchmark transport mode {mode!r}")


def _rotation(values: Sequence[str], index: int) -> tuple[str, ...]:
    items = tuple(str(value) for value in values)
    if not items:
        raise ValueError("transport list must be non-empty")
    offset = int(index) % len(items)
    return (*items[offset:], *items[:offset])


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("percentile values must be non-empty")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * float(quantile)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _summarize_runs(
    runs: Sequence[dict[str, Any]],
    *,
    capture_ms: float,
    prompt_length: int | None = None,
) -> dict[str, Any]:
    rows = tuple(runs)
    if not rows:
        raise ValueError("cannot summarize zero runs")
    steps = int(rows[0]["steps"])
    if steps <= 0 or any(int(row["steps"]) != steps for row in rows):
        raise ValueError("all runs must have one positive common step count")
    replay_walls = [float(row["replay_wall_ms"]) for row in rows]
    median_wall = float(statistics.median(replay_walls))
    per_run_host_ns = [
        float(statistics.median(float(value) for value in row["host_call_ns"])) for row in rows
    ]
    per_run_synced_ns = [
        float(statistics.median(float(value) for value in row["synchronized_step_ns"]))
        for row in rows
    ]
    all_synced_ns = [float(value) for row in rows for value in row["synchronized_step_ns"]]
    median_ms_per_token = median_wall / steps
    median_prefill_ms = float(statistics.median(float(row["prefill_ms"]) for row in rows))
    summary = {
        "runs": len(rows),
        "steps_per_run": steps,
        "capture_ms": float(capture_ms),
        "median_replay_wall_ms": median_wall,
        "median_replay_ms_per_token": median_ms_per_token,
        "median_replay_tok_s": 1000.0 / median_ms_per_token,
        "median_host_call_us": float(statistics.median(per_run_host_ns)) / 1000.0,
        "median_synchronized_step_ms": float(statistics.median(per_run_synced_ns)) / 1e6,
        "p95_synchronized_step_ms": _percentile(all_synced_ns, 0.95) / 1e6,
        "capture_inclusive_ms_per_token": (median_wall + float(capture_ms)) / steps,
        "capture_inclusive_tok_s": 1000.0 * steps / (median_wall + float(capture_ms)),
        "median_prefill_ms": median_prefill_ms,
        "replay_wall_ms_samples": replay_walls,
        "host_call_us_samples": [value / 1000.0 for value in per_run_host_ns],
    }
    if prompt_length is not None:
        prompt = int(prompt_length)
        if prompt <= 0:
            raise ValueError("prompt_length must be positive when provided")
        request_wall_ms = median_prefill_ms + float(capture_ms) + median_wall
        summary.update(
            {
                "request_inclusive_wall_ms": request_wall_ms,
                "request_decode_output_tok_s": 1000.0 * steps / request_wall_ms,
                "request_total_tok_s": 1000.0 * (prompt + steps) / request_wall_ms,
            }
        )
    return summary


def _setup_breakdown(capture_ms: float, provenance: dict[str, Any]) -> dict[str, float]:
    transport_context = provenance.get("transport_context", {})
    executable = provenance.get("executable", {})
    context_create_ms = float(transport_context.get("context_create_ns", 0)) / 1e6
    graph_inspection_ms = float(transport_context.get("last_graph_inspection_ns", 0)) / 1e6
    native_instantiate_ms = float(transport_context.get("last_native_instantiate_ns", 0)) / 1e6
    attributed_ms = context_create_ms + graph_inspection_ms + native_instantiate_ms
    result = {
        "capture_total_ms": float(capture_ms),
        "context_create_ms": context_create_ms,
        "graph_inspection_ms": graph_inspection_ms,
        "native_instantiate_ms": native_instantiate_ms,
        "capture_residual_ms": float(capture_ms) - attributed_ms,
    }
    for key, value in transport_context.get("last_graph_inspection_phases_ns", {}).items():
        result[f"graph_inspection_{key.removesuffix('_ns')}_ms"] = float(value) / 1e6
    for key in (
        "module_load_ns",
        "kernel_resolve_ns",
        "kernarg_stage_ns",
        "kernarg_allocate_ns",
        "aql_packet_build_ns",
        "pm4_encode_ns",
        "ib_allocate_ns",
    ):
        result[key.removesuffix("_ns") + "_ms"] = float(executable.get(key, 0)) / 1e6
    return result


def _validate_cross_transport(
    runs: dict[str, Sequence[dict[str, Any]]],
    *,
    expected_token_id: int,
) -> dict[str, Any]:
    rows = [row for mode_rows in runs.values() for row in mode_rows]
    if not rows:
        raise ValueError("correctness validation requires measured runs")
    tokens = {int(row["final_token_id"]) for row in rows}
    state_hashes = {str(row["state_sha256"]) for row in rows}
    logits_hashes = {str(row["final_logits_sha256"]) for row in rows}
    tokens_exact = tokens == {int(expected_token_id)}
    state_exact = len(state_hashes) == 1
    logits_exact = len(logits_hashes) == 1
    return {
        "passed": bool(tokens_exact and state_exact and logits_exact),
        "tokens_exact": tokens_exact,
        "state_exact": state_exact,
        "final_logits_exact": logits_exact,
        "final_token_ids": sorted(tokens),
        "state_sha256": next(iter(state_hashes)) if state_exact else None,
        "state_sha256_observed": sorted(state_hashes),
        "final_logits_sha256": next(iter(logits_hashes)) if logits_exact else None,
        "final_logits_sha256_observed": sorted(logits_hashes),
    }


def _state_sha256(session: Any, *, input_token_id: int, predicted_token_id: int) -> str:
    checkpoint = _capture_checkpoint(
        session,
        position=int(session.position),
        input_token_id=int(input_token_id),
        predicted_token_id=int(predicted_token_id),
    )
    # Production graph timing does not request the optional hidden-seed tap.
    # P5 already proves that tap bit exact.  P6 hashes only state mutated by the
    # timed production graph: cursor, every Conv/GDN pair, and all live K/V.
    payload = {
        "position": checkpoint["position"],
        "input_token_id": checkpoint["input_token_id"],
        "predicted_token_id": checkpoint["predicted_token_id"],
        "linear_states": checkpoint["linear_states"],
        "kv_states": checkpoint["kv_states"],
    }
    return _canonical_sha256(payload)


def _logits_sha256(session: Any) -> str:
    logits = np.ascontiguousarray(session._read_sample(return_logits=True).logits, dtype=np.float32)
    if logits.size == 0 or not np.isfinite(logits).all():
        raise RuntimeError("timed graph produced empty or non-finite final logits")
    return hashlib.sha256(logits.tobytes()).hexdigest()


def _timed_replay(graph: Any, *, steps: int, roctx: "_Roctx", label: str) -> dict[str, Any]:
    if graph.closed:
        raise RuntimeError("cannot time a closed graph")
    if graph.submission is None:
        raise RuntimeError("benchmark requires the registered graph submission owner")
    count = int(steps)
    if count <= 0 or count % int(graph.steps_per_replay):
        raise ValueError("steps must be positive and divisible by steps_per_replay")
    expected_position = int(graph.position) + int(graph.replayed_steps)
    if int(graph.session.position) != expected_position:
        raise RuntimeError("timed graph state generation drifted before replay")

    host_call_ns: list[int] = []
    synchronized_step_ns: list[int] = []
    runtime = graph.session.runtime
    replay_start_ns = time.perf_counter_ns()
    with roctx.range(label):
        for _ in range(count // int(graph.steps_per_replay)):
            step_start_ns = time.perf_counter_ns()
            graph.submission.launch(graph.stream)
            call_end_ns = time.perf_counter_ns()
            runtime.stream_synchronize(graph.stream)
            step_end_ns = time.perf_counter_ns()
            host_call_ns.append(call_end_ns - step_start_ns)
            synchronized_step_ns.append(step_end_ns - step_start_ns)
            graph.replayed_steps += int(graph.steps_per_replay)
            graph.session._position = int(graph.position) + int(graph.replayed_steps)
            if graph.session.scratch is not None:
                graph.session.scratch.position_host[0] = graph.session._position
                graph.session.scratch.context_host[0] = graph.session._position + 1
    replay_wall_ms = (time.perf_counter_ns() - replay_start_ns) / 1e6
    if graph.capture_hidden_seed_fp32:
        graph.session._hidden_seed_fp32_populated = True
    return {
        "steps": count,
        "replay_wall_ms": replay_wall_ms,
        "host_call_ns": host_call_ns,
        "synchronized_step_ns": synchronized_step_ns,
    }


class _NullRange(AbstractContextManager[None]):
    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class _RoctxRange(AbstractContextManager[None]):
    def __init__(self, library: Any, name: str) -> None:
        self.library = library
        self.name = name

    def __enter__(self) -> None:
        self.library.roctxRangePushA(self.name.encode("utf-8"))
        return None

    def __exit__(self, exc_type, exc, tb) -> None:
        self.library.roctxRangePop()


class _Roctx:
    def __init__(self, enabled: bool) -> None:
        self.library = None
        if enabled:
            self.library = ctypes.CDLL("libroctx64.so")
            self.library.roctxRangePushA.argtypes = [ctypes.c_char_p]
            self.library.roctxRangePushA.restype = ctypes.c_int
            self.library.roctxRangePop.argtypes = []
            self.library.roctxRangePop.restype = ctypes.c_int

    def range(self, name: str) -> AbstractContextManager[None]:
        if self.library is None:
            return _NullRange()
        return _RoctxRange(self.library, name)


def _read_compiler_version(path: Path | None) -> str | None:
    if path is None:
        return None
    value = path.expanduser().read_text(encoding="utf-8").strip()
    if not value:
        raise ValueError(f"empty compiler-version file: {path}")
    return value


def run(args: argparse.Namespace) -> dict[str, Any]:
    transports = tuple(dict.fromkeys(str(mode) for mode in args.transports))
    if not transports or any(mode not in _BENCHMARK_MODES for mode in transports):
        raise ValueError(
            "transports must be selected from hipgraph, aql, pm4, pm4_stateful, "
            "pm4_stateful_local, and pm4_timestamps"
        )
    if args.backend != "hip_gfx1100" and any(
        _transport_spec(mode)[0] != "hipgraph" for mode in transports
    ):
        raise ValueError("aql/pm4 are admitted only on hip_gfx1100")
    if min(int(args.prompt_length), int(args.steps), int(args.repetitions)) <= 0:
        raise ValueError("prompt-length, steps, and repetitions must be positive")
    if int(args.warmups) < 0:
        raise ValueError("warmups must be non-negative")
    if int(args.memory_recovery_tolerance_bytes) < 0:
        raise ValueError("memory recovery tolerance must be non-negative")

    target_arch = {
        "hip_gfx1100": "gfx1100",
        "hip_gfx1151": "gfx1151",
    }[str(args.backend)]
    os.environ["HIPENGINE_GGUF_DECODE_REPACK"] = "1"
    os.environ["HIPENGINE_GGUF_MOE_GRAPH"] = "0"
    os.environ["HIPENGINE_HIP_ARCH"] = target_arch
    compiler_version = _read_compiler_version(args.compiler_version_file)
    if args.compiler_version_file is not None:
        os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(
            args.compiler_version_file.expanduser().resolve()
        )

    from hipengine.benchmark.provenance import collect_artifact_provenance
    from hipengine.core.pm4.transport import create_graph_submission_context
    from hipengine.runtime.gguf_decode_graph import capture_qwen35_gguf_decode_graph
    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession

    model = Path(args.model).expanduser().resolve()
    if not model.is_file():
        raise FileNotFoundError(model)
    stat = model.stat()
    prompt_ids = [int(args.prompt_token_id)] * int(args.prompt_length)
    max_sequence_length = int(args.prompt_length) + int(args.steps) + 8
    roctx = _Roctx(bool(args.roctx))
    graphs: dict[str, Any] = {}
    capture_ms: dict[str, float] = {}
    warmup_rows: dict[str, list[dict[str, Any]]] = {mode: [] for mode in transports}
    measured_rows: dict[str, list[dict[str, Any]]] = {mode: [] for mode in transports}
    teardown: dict[str, Any] = {}
    custom_contexts: dict[str, Any] = {}

    with Qwen35GGUFResidentSession(
        model,
        max_sequence_length=max_sequence_length,
        compiler_version=compiler_version,
        require_cached_build=bool(args.require_cached),
        backend=str(args.backend),
        use_wmma_prefill=True,
        use_gemv_decode=True,
    ) as session:
        _prefill(session, prompt_ids)
        free_before_graphs, total_bytes = session.runtime.mem_get_info()
        try:
            for mode in transports:
                selected_transport, stateful_registers, local_cache_dependencies = _transport_spec(
                    mode
                )
                capture_start_ns = time.perf_counter_ns()
                if mode in {"pm4_stateful", "pm4_stateful_local", "pm4_timestamps"}:
                    # Comparison aliases use separate contexts so the canonical
                    # PM4 encoder and the retained global-acquire diagnostic can
                    # coexist in one counterbalanced session.
                    context = create_graph_submission_context(
                        backend=str(session.runner.backend),
                        gfx_arch=str(session.runner.target_arch),
                        runtime=session.runtime,
                        transport="pm4",
                    )
                    if context is None:
                        raise RuntimeError("stateful PM4 benchmark context was not created")
                    context.stateful_registers = bool(stateful_registers)
                    context.local_cache_dependencies = bool(local_cache_dependencies)
                    context.timestamps = mode == "pm4_timestamps"
                    custom_contexts[mode] = context
                    graph = capture_qwen35_gguf_decode_graph(
                        session,
                        position=int(args.prompt_length),
                        steps_per_replay=1,
                        max_replay_steps=int(args.steps),
                        attention_max_context_len=int(args.prompt_length) + int(args.steps),
                        submission_transport="pm4",
                        submission_context=context,
                    )
                    graphs[mode] = graph
                    session._pin_device_kv_graph(graph)
                else:
                    graphs[mode] = session.capture_decode_graph(
                        position=int(args.prompt_length),
                        steps_per_replay=1,
                        max_replay_steps=int(args.steps),
                        attention_max_context_len=int(args.prompt_length) + int(args.steps),
                        submission_transport=selected_transport,
                    )
                capture_ms[mode] = (time.perf_counter_ns() - capture_start_ns) / 1e6

            for run_index in range(int(args.warmups) + int(args.repetitions)):
                measured = run_index >= int(args.warmups)
                for mode in _rotation(transports, run_index):
                    prefill_start_ns = time.perf_counter_ns()
                    seed_token = _prefill(session, prompt_ids)
                    prefill_ms = (time.perf_counter_ns() - prefill_start_ns) / 1e6
                    graph = graphs[mode]
                    graph.rearm_replay_window()
                    session.runtime.device_synchronize()
                    row = _timed_replay(
                        graph,
                        steps=int(args.steps),
                        roctx=roctx,
                        label=f"hipengine:pm4_graph_bench:{mode}:{run_index}",
                    )
                    final = session._read_sample(return_logits=False)
                    row.update(
                        {
                            "run_index": run_index,
                            "order": list(_rotation(transports, run_index)),
                            "prefill_ms": prefill_ms,
                            "seed_token_id": int(seed_token),
                            "final_token_id": int(final.token_id),
                            "state_sha256": _state_sha256(
                                session,
                                input_token_id=int(seed_token),
                                predicted_token_id=int(final.token_id),
                            ),
                            "final_logits_sha256": _logits_sha256(session),
                        }
                    )
                    (measured_rows if measured else warmup_rows)[mode].append(row)
        finally:
            for mode in reversed(transports):
                graph = graphs.get(mode)
                if graph is not None:
                    try:
                        teardown[mode] = {"live": graph.transport_provenance()}
                    finally:
                        graph.close()
                        teardown.setdefault(mode, {})["closed"] = graph.transport_provenance()
            teardown["contexts"] = session.close_decode_graph_submission_contexts()
            for mode, context in reversed(tuple(custom_contexts.items())):
                before = context.provenance()
                context.close()
                teardown["contexts"][mode] = {
                    "before": before,
                    "after": context.provenance(),
                }
        session.runtime.device_synchronize()
        free_after_graphs, total_after = session.runtime.mem_get_info()
        pci_bdf = session.runtime.device_pci_bus_id()

    correctness_runs = {mode: [*warmup_rows[mode], *measured_rows[mode]] for mode in transports}
    correctness = _validate_cross_transport(
        correctness_runs,
        expected_token_id=int(args.expected_token_id),
    )
    summaries = {
        mode: _summarize_runs(
            measured_rows[mode],
            capture_ms=capture_ms[mode],
            prompt_length=int(args.prompt_length),
        )
        for mode in transports
    }
    native_proof = all(
        mode == "hipgraph"
        or (
            teardown[mode]["live"].get("native_fallbacks") == 0
            and teardown[mode]["live"].get("launches")
            == (int(args.warmups) + int(args.repetitions)) * int(args.steps)
            and teardown[mode]["closed"].get("closed") is True
            and teardown["contexts"][mode]["after"].get("closed") is True
            and (
                _transport_spec(mode)[1] is None
                or teardown[mode]["live"].get("stateful_registers") is _transport_spec(mode)[1]
            )
            and (
                _transport_spec(mode)[2] is None
                or teardown[mode]["live"].get("local_cache_dependencies")
                is _transport_spec(mode)[2]
            )
        )
        for mode in transports
    )
    memory_delta_bytes = int(free_before_graphs) - int(free_after_graphs)
    memory_recovered = bool(
        total_bytes == total_after
        and memory_delta_bytes <= int(args.memory_recovery_tolerance_bytes)
    )
    passed = bool(correctness["passed"] and native_proof and memory_recovered)
    provenance = collect_artifact_provenance(
        repo_root=REPO_ROOT,
        configured_backend=str(args.backend),
        resolved_backend=str(args.backend),
        target_arch=target_arch,
        model_path=model,
        quant="gguf_q4_k_m",
        kv_dtype="bf16",
        command=[str(part) for part in sys.argv],
        environment={
            "HIP_VISIBLE_DEVICES": os.environ.get("HIP_VISIBLE_DEVICES"),
            "ROCR_VISIBLE_DEVICES": os.environ.get("ROCR_VISIBLE_DEVICES"),
            "GPU_MAX_HW_QUEUES": os.environ.get("GPU_MAX_HW_QUEUES"),
            "HIPENGINE_HIP_ARCH": target_arch,
        },
        build_profile="pm4_graph_bench",
        timing_protocol=(
            "one resident model/session; one graph generation per transport; exact reset/rearm; "
            "rotating transport order; host submission call and synchronized replay timed separately"
        ),
        warmups=int(args.warmups),
        repetitions=int(args.repetitions),
        profiler={"enabled": bool(args.roctx), "ranges": "one range per transport run"},
    )
    return {
        "schema_version": 1,
        "kind": "hipengine_pm4_graph_transport_benchmark",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "diagnostic_accepted" if passed else "rejected_correctness",
        "performance_claim": False,
        "passed": passed,
        "model": {
            "path": str(model),
            "size_bytes": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        },
        "hardware": {
            "backend": str(args.backend),
            "gfx_arch": target_arch,
            "pci_bdf": pci_bdf,
        },
        "workload": {
            "prompt_token_id": int(args.prompt_token_id),
            "expected_token_id": int(args.expected_token_id),
            "prompt_length": int(args.prompt_length),
            "decode_steps": int(args.steps),
            "transports": list(transports),
            "warmups": int(args.warmups),
            "repetitions": int(args.repetitions),
            "counterbalanced": True,
            "transport_specs": {
                mode: {
                    "transport": _transport_spec(mode)[0],
                    "pm4_stateful_registers": _transport_spec(mode)[1],
                    "pm4_local_cache_dependencies": _transport_spec(mode)[2],
                }
                for mode in transports
            },
        },
        "capture_ms": capture_ms,
        "setup_breakdown_ms": {
            mode: _setup_breakdown(capture_ms[mode], teardown[mode]["live"]) for mode in transports
        },
        "summaries": summaries,
        "correctness": correctness,
        "native_transport_proof": native_proof,
        "memory": {
            "free_before_graphs": free_before_graphs,
            "free_after_graphs": free_after_graphs,
            "total_bytes": total_bytes,
            "free_delta_bytes": memory_delta_bytes,
            "recovery_tolerance_bytes": int(args.memory_recovery_tolerance_bytes),
            "recovered": memory_recovered,
        },
        "warmup_runs": warmup_rows,
        "measured_runs": measured_rows,
        "teardown": teardown,
        "provenance": provenance,
        "notes": [
            "Focused transport diagnostic; performance_claim=false until natural-suite promotion gates pass.",
            "Replay timing excludes reset/prefill, state/logit readback, and graph capture.",
            "Capture-inclusive throughput adds one measured capture/instantiate cost per decode window.",
            "No submit-plus-queue-recreate stress is performed.",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--backend", choices=("hip_gfx1100", "hip_gfx1151"), default="hip_gfx1100")
    parser.add_argument(
        "--transports",
        nargs="+",
        choices=_BENCHMARK_MODES,
        default=list(_DEFAULT_TRANSPORTS),
    )
    parser.add_argument("--prompt-token-id", type=int, default=9707)
    parser.add_argument("--expected-token-id", type=int, default=9707)
    parser.add_argument("--prompt-length", type=int, default=512)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument(
        "--compiler-version-file",
        type=Path,
        default=Path("/tmp/hipengine-hipcc-version.txt"),
    )
    parser.add_argument("--require-cached", action="store_true")
    parser.add_argument("--roctx", action="store_true")
    parser.add_argument(
        "--memory-recovery-tolerance-bytes",
        type=int,
        default=_DEFAULT_MEMORY_RECOVERY_TOLERANCE,
    )
    parser.add_argument("--json", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = run(args)
    encoded = json.dumps(payload, sort_keys=True, indent=2)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
