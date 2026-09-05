#!/usr/bin/env python3
"""Measure direct gfx11 GGUF packed graph groups and honest controls.

The default packet runs one state-bound c1, c2, c4, and native-c8 graph bucket,
an explicit chunked-c8 control made of two serial c4 groups, and a serial-c4
bridge made of four c1 groups. Every logical decode transition advances each
live row once.
Graph capture, diagnostic token readback, and final state flush are excluded
from decode throughput; each one-step replay is synchronized so the reported
inter-token distribution is a model-step completion latency rather than enqueue
latency. This harness is a raw native-c8 measurement packet, not by itself a
retained performance claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shlex
import statistics
import sys
import time
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:] = [str(REPO_ROOT), *(entry for entry in sys.path if entry != str(REPO_ROOT))]

import hipengine  # noqa: E402

from hipengine.benchmark.provenance import collect_artifact_provenance  # noqa: E402


if Path(hipengine.__file__).resolve().parents[1] != REPO_ROOT:
    raise RuntimeError("gguf_packed_ar_bench imported hipengine from another checkout")
DEFAULT_MODEL = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
SUPPORTED_BACKENDS = ("hip_gfx1100", "hip_gfx1151")
_EXACT_ENV = {
    "HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN": "1",
    "HIPENGINE_GGUF_GDN_PREFILL_MODE": "exact",
}
# Production route: no forced exact-capture env; the backend auto GDN prefill
# policy resolves (gfx1151 Q4_K_S -> chain_compact_peer_wave32). Used to measure
# production-profile variants (e.g. fp16 recurrent state) that are incompatible
# with the strict verify-capture/exact state-rows prefill writers.
_PRODUCTION_ENV: dict[str, str | None] = {
    "HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN": None,
    "HIPENGINE_GGUF_GDN_PREFILL_MODE": None,
}
_PROVENANCE_ENV_KEYS = (
    "HIPENGINE_BACKEND",
    "HIPENGINE_HIP_ARCH",
    "HIPENGINE_COMPILER_VERSION_FILE",
    "HIPENGINE_SUBMISSION_TRANSPORT",
    "HIPENGINE_GGUF_FP16_RECURRENT_STATE",
    "HIPENGINE_GGUF_Q6_LM_HEAD_MAX_CHUNK",
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "GPU_MAX_HW_QUEUES",
)


@dataclass(frozen=True)
class PackedARConfiguration:
    name: str
    logical_rows: int
    native_group_width: int
    native_group_count: int
    execution_class: str


CONFIGURATIONS: dict[str, PackedARConfiguration] = {
    "c1": PackedARConfiguration("c1", 1, 1, 1, "direct_native_group"),
    "c2": PackedARConfiguration("c2", 2, 2, 1, "direct_native_group"),
    "c3": PackedARConfiguration("c3", 3, 3, 1, "direct_native_group"),
    "c4": PackedARConfiguration("c4", 4, 4, 1, "direct_native_group"),
    "c5": PackedARConfiguration("c5", 5, 5, 1, "direct_native_group"),
    "c6": PackedARConfiguration("c6", 6, 6, 1, "direct_native_group"),
    "c7": PackedARConfiguration("c7", 7, 7, 1, "direct_native_group"),
    "native_c8": PackedARConfiguration("native_c8", 8, 8, 1, "direct_native_group"),
    "chunked_c8": PackedARConfiguration("chunked_c8", 8, 4, 2, "chunked_native_groups"),
    "serial_c4": PackedARConfiguration("serial_c4", 4, 1, 4, "serial_bridge"),
}


def _parse_configurations(raw: str) -> tuple[str, ...]:
    names = tuple(part.strip() for part in str(raw).split(",") if part.strip())
    if not names:
        raise ValueError("configurations must not be empty")
    unknown = sorted(set(names) - set(CONFIGURATIONS))
    if unknown:
        raise ValueError(f"unknown packed AR configurations: {unknown!r}")
    if len(set(names)) != len(names):
        raise ValueError("configurations must be unique")
    canonical = tuple(CONFIGURATIONS)
    if set(names) == set(canonical) and names != canonical:
        raise ValueError(
            "the complete packet must use canonical c1,c2,c4,native_c8,chunked_c8,serial_c4 order "
            "so route-specific residency grows monotonically"
        )
    return names


def _configuration_groups(config: PackedARConfiguration) -> tuple[tuple[int, ...], ...]:
    groups = tuple(
        tuple(range(start, min(start + config.native_group_width, config.logical_rows)))
        for start in range(0, config.logical_rows, config.native_group_width)
    )
    if len(groups) != config.native_group_count or any(
        len(group) != config.native_group_width for group in groups
    ):
        raise ValueError(f"invalid declared group shape for {config.name!r}")
    return groups


def _stats(values: Sequence[float]) -> dict[str, Any]:
    samples = [float(value) for value in values]
    if not samples:
        return {
            "samples": [],
            "median": None,
            "p95": None,
            "min": None,
            "max": None,
            "stdev": None,
            "stdev_pct_of_median": None,
        }
    ordered = sorted(samples)
    median = float(statistics.median(samples))
    stdev = float(statistics.stdev(samples)) if len(samples) > 1 else 0.0
    p95 = ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)]
    return {
        "samples": samples,
        "median": median,
        "p95": float(p95),
        "min": float(ordered[0]),
        "max": float(ordered[-1]),
        "stdev": stdev,
        "stdev_pct_of_median": None if median == 0.0 else 100.0 * stdev / median,
    }


def _safe_ratio(numerator: Any, denominator: Any) -> float | None:
    if not isinstance(numerator, (int, float)) or not isinstance(denominator, (int, float)):
        return None
    if not math.isfinite(float(numerator)) or not math.isfinite(float(denominator)):
        return None
    if float(denominator) <= 0.0:
        return None
    return float(numerator) / float(denominator)


def _trajectory_fingerprint(tokens: Sequence[int]) -> dict[str, Any]:
    token_ids = [int(token) for token in tokens]
    digest = hashlib.sha256()
    for token in token_ids:
        digest.update(int(token).to_bytes(8, "little", signed=True))
    return {
        "length": len(token_ids),
        "sha256": digest.hexdigest(),
        "first_token_ids": token_ids[:8],
        "last_token_ids": token_ids[-8:],
        "final_token_id": token_ids[-1] if token_ids else None,
    }


def _prompt_rows(*, rows: int, prompt_length: int, token_id: int) -> tuple[tuple[int, ...], ...]:
    prompts: list[tuple[int, ...]] = []
    for row in range(int(rows)):
        prompt = [int(token_id)] * int(prompt_length)
        # Repeat the four-row deterministic fixture for both c8 routes so each
        # half has a direct c4 trajectory oracle without token-conditioned code.
        prompt[-1] = int(token_id) + (row % 4)
        prompts.append(tuple(prompt))
    return tuple(prompts)


def _prompt_fingerprint(prompts: Sequence[Sequence[int]]) -> str:
    digest = hashlib.sha256()
    for prompt in prompts:
        digest.update(len(prompt).to_bytes(8, "little"))
        for token in prompt:
            digest.update(int(token).to_bytes(8, "little", signed=True))
    return digest.hexdigest()


def _occupancy_event(
    config: PackedARConfiguration,
    *,
    phase: str,
    elapsed_seconds: float,
) -> dict[str, Any]:
    groups = _configuration_groups(config)
    return {
        "phase": str(phase),
        "elapsed_seconds": float(elapsed_seconds),
        "logical_active_rows": config.logical_rows,
        "native_group_width": config.native_group_width,
        "native_group_count": config.native_group_count,
        "physical_bucket_widths": [len(group) for group in groups],
        "active_masks": [[True] * len(group) for group in groups],
    }


@contextmanager
def _temporary_environment(updates: Mapping[str, str | None]) -> Iterator[None]:
    prior = {key: os.environ.get(key) for key in updates}
    for key, value in updates.items():
        if value is None:
            os.environ.pop(str(key), None)
        else:
            os.environ[str(key)] = str(value)
    try:
        yield
    finally:
        for key, value in prior.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _memory_snapshot(label: str, runtime: Any) -> dict[str, Any]:
    from hipengine.core.memory import memory_stats

    free_bytes, total_bytes = runtime.mem_get_info()
    return {
        "label": str(label),
        "tracked": memory_stats(),
        "hip_free_bytes": int(free_bytes),
        "hip_total_bytes": int(total_bytes),
        "hip_used_bytes": int(total_bytes - free_bytes),
    }


def _read_compiler_version(path: Path | None) -> str | None:
    if path is None:
        return None
    text = path.expanduser().read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"compiler-version file is empty: {path}")
    return text


def _copy_json(value: Any) -> Any:
    return json.loads(json.dumps(value))


def _graph_bucket_shape_sha256(bucket_key: Mapping[str, Any]) -> str:
    payload = dict(bucket_key)
    payload.pop("buffer_identity_sha256", None)
    payload.pop("key_sha256", None)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _graph_manifest_matches_configuration(
    config: PackedARConfiguration,
    manifest: Mapping[str, Any],
    *,
    decode_steps: int,
) -> bool:
    model_step = manifest.get("model_step", {})
    graph = manifest.get("graph", {})
    movement = manifest.get("host_device_movement", {})
    return bool(
        manifest.get("mode") == "decode_graph_replay"
        and manifest.get("physical_rows") == config.native_group_width
        and manifest.get("active_rows") == config.native_group_width
        and manifest.get("active_mask") == [True] * config.native_group_width
        and graph.get("captured") is True
        and graph.get("replay_count") == int(decode_steps)
        and graph.get("replayed_steps") == int(decode_steps)
        and model_step.get("complete_c1_session_replays") == 0
        and model_step.get("complete_c1_layer_replays") == 0
        and (
            config.native_group_width == 1
            or model_step.get("host_model_row_loop_sites") == 0
        )
        and movement.get("host_to_device_total_copies") == 0
        and movement.get("device_to_host_vector_copies") == 0
    )


def _run_sample(
    *,
    config: PackedARConfiguration,
    sessions: Sequence[Any],
    prompts: Sequence[Sequence[int]],
    decode_steps: int,
    measured: bool,
    run_index: int,
    submission_transport: str | None = None,
) -> dict[str, Any]:
    groups = _configuration_groups(config)
    active_sessions = tuple(sessions[: config.logical_rows])
    runtime = active_sessions[0].runtime
    for session in active_sessions:
        session.reset()
    runtime.device_synchronize()

    memory: dict[str, Any] = {"after_reset": _memory_snapshot("after_reset", runtime)}
    sample_start = time.perf_counter()
    occupancy_timeline = [
        _occupancy_event(config, phase="admitted", elapsed_seconds=0.0)
    ]
    trajectories: list[list[int]] = [[] for _ in range(config.logical_rows)]
    current = [0] * config.logical_rows
    ttft_seconds = [0.0] * config.logical_rows
    group_prefill_seconds: list[float] = []

    prefill_start = time.perf_counter()
    for group_indices in groups:
        group_sessions = tuple(active_sessions[index] for index in group_indices)
        group_prompts = tuple(prompts[index] for index in group_indices)
        group_start = time.perf_counter()
        results = group_sessions[0].prefill_batch_native(
            group_prompts,
            sessions=group_sessions,
            return_logits=False,
        )
        group_prefill_seconds.append(time.perf_counter() - group_start)
        cumulative = time.perf_counter() - prefill_start
        for index, result in zip(group_indices, results, strict=True):
            token = int(result.token_id)
            current[index] = token
            trajectories[index].append(token)
            ttft_seconds[index] = cumulative
    prefill_seconds = time.perf_counter() - prefill_start
    memory["after_prefill"] = _memory_snapshot("after_prefill", runtime)
    occupancy_timeline.append(
        _occupancy_event(
            config,
            phase="prefill_complete",
            elapsed_seconds=time.perf_counter() - sample_start,
        )
    )

    graphs: list[Any] = []
    graph_groups: list[tuple[int, ...]] = []
    capture_start = time.perf_counter()
    try:
        for group_indices in groups:
            group_sessions = tuple(active_sessions[index] for index in group_indices)
            graph = group_sessions[0].capture_packed_decode_graph(
                [current[index] for index in group_indices],
                sessions=group_sessions,
                steps_per_replay=1,
                max_replay_steps=int(decode_steps),
                record_steps=int(decode_steps),
                submission_transport=submission_transport,
            )
            graphs.append(graph)
            graph_groups.append(group_indices)
        graph_capture_seconds = time.perf_counter() - capture_start
        memory["after_graph_capture"] = _memory_snapshot("after_graph_capture", runtime)
        occupancy_timeline.append(
            _occupancy_event(
                config,
                phase="graph_captured",
                elapsed_seconds=time.perf_counter() - sample_start,
            )
        )

        logical_step_seconds: list[float] = []
        group_step_seconds: list[list[float]] = []
        decode_start = time.perf_counter()
        for _ in range(int(decode_steps)):
            step_start = time.perf_counter()
            group_times: list[float] = []
            for graph in graphs:
                group_start = time.perf_counter()
                graph.replay(1)
                group_times.append(time.perf_counter() - group_start)
            logical_step_seconds.append(time.perf_counter() - step_start)
            group_step_seconds.append(group_times)
        decode_seconds = time.perf_counter() - decode_start
        memory["after_decode"] = _memory_snapshot("after_decode", runtime)
        occupancy_timeline.append(
            _occupancy_event(
                config,
                phase="decode_complete",
                elapsed_seconds=time.perf_counter() - sample_start,
            )
        )

        for graph, group_indices in zip(graphs, graph_groups, strict=True):
            generated = graph.read_generated_token_ids(int(decode_steps))
            for step_tokens in generated:
                for index, token in zip(group_indices, step_tokens, strict=True):
                    trajectories[index].append(int(token))
        for graph in graphs:
            graph.execution_manifest["graph"]["transport"] = graph.transport_provenance()
        graph_manifests = [_copy_json(graph.execution_manifest) for graph in graphs]
        flush_results = [bool(graph.flush_packed_state()) for graph in graphs]
        memory["after_flush"] = _memory_snapshot("after_flush", runtime)
    finally:
        for graph in reversed(graphs):
            graph.close()

    fingerprints = [_trajectory_fingerprint(tokens) for tokens in trajectories]
    expected_length = int(decode_steps) + 1
    manifests_ok = all(
        _graph_manifest_matches_configuration(
            config,
            manifest,
            decode_steps=int(decode_steps),
        )
        for manifest in graph_manifests
    )
    trajectory_lengths_ok = all(row["length"] == expected_length for row in fingerprints)
    generated_tokens = config.logical_rows * int(decode_steps)
    aggregate_decode = generated_tokens / decode_seconds if decode_seconds > 0.0 else None
    return {
        "configuration": config.name,
        "run_index": int(run_index),
        "measured": bool(measured),
        "passed": bool(manifests_ok and trajectory_lengths_ok and all(flush_results)),
        "route": {
            **asdict(config),
            "resident_session_count_at_measurement": len(sessions),
            "physical_bucket_widths": [len(group) for group in groups],
            "active_masks": [[True] * len(group) for group in groups],
            "serial_bridge": config.execution_class == "serial_bridge",
            "chunked": config.execution_class == "chunked_native_groups",
            "native_c8_claim": config.name == "native_c8",
            "requested_submission_transport": submission_transport,
        },
        "accounting": {
            "prompt_tokens": config.logical_rows * len(prompts[0]),
            "decode_transitions_per_row": int(decode_steps),
            "generated_decode_tokens": generated_tokens,
            "trajectory_tokens_including_prefill_sample": config.logical_rows * expected_length,
        },
        "timings": {
            "prefill_seconds": prefill_seconds,
            "group_prefill_seconds": group_prefill_seconds,
            "graph_capture_seconds": graph_capture_seconds,
            "decode_seconds": decode_seconds,
            "logical_step_seconds": logical_step_seconds,
            "group_step_seconds": group_step_seconds,
        },
        "throughput": {
            "prefill_tok_s_aggregate": (
                config.logical_rows * len(prompts[0]) / prefill_seconds
                if prefill_seconds > 0.0
                else None
            ),
            "decode_tok_s_aggregate": aggregate_decode,
            "decode_tok_s_per_request": (
                aggregate_decode / config.logical_rows
                if aggregate_decode is not None
                else None
            ),
        },
        "latency": {
            "ttft_seconds_per_request": ttft_seconds,
            "inter_token_model_step_seconds": logical_step_seconds,
            "contract": (
                "one synchronized graph replay per native group per logical transition; "
                "diagnostic token D2H occurs once after the measured window"
            ),
        },
        "occupancy": {
            "timeline": occupancy_timeline,
            "row_count_transitions": [],
            "constant_during_static_workload": True,
        },
        "trajectory_fingerprints": fingerprints,
        "graph_manifests": graph_manifests,
        "graph_bucket_instance_key_sha256": [
            manifest["graph"]["bucket_key"]["key_sha256"] for manifest in graph_manifests
        ],
        "graph_bucket_shape_sha256": [
            _graph_bucket_shape_sha256(manifest["graph"]["bucket_key"])
            for manifest in graph_manifests
        ],
        "flush_results": flush_results,
        "memory": memory,
    }


def _summarize_configuration(
    config: PackedARConfiguration,
    samples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    measured = [sample for sample in samples if sample.get("measured") is True]
    trajectory_hashes = [
        [str(row["sha256"]) for row in sample["trajectory_fingerprints"]]
        for sample in measured
    ]
    bucket_instance_keys = [
        list(sample["graph_bucket_instance_key_sha256"]) for sample in measured
    ]
    bucket_shape_keys = [
        list(sample["graph_bucket_shape_sha256"]) for sample in measured
    ]
    ttft = [
        float(value)
        for sample in measured
        for value in sample["latency"]["ttft_seconds_per_request"]
    ]
    itl = [
        float(value)
        for sample in measured
        for value in sample["latency"]["inter_token_model_step_seconds"]
    ]
    decode_aggregate = _stats(
        [float(sample["throughput"]["decode_tok_s_aggregate"]) for sample in measured]
    )
    tracked_current_peaks = [
        max(
            int(snapshot["tracked"]["current_allocated_bytes"])
            for snapshot in sample["memory"].values()
        )
        for sample in measured
    ]
    tracked_high_water = [
        max(
            int(snapshot["tracked"]["peak_allocated_bytes"])
            for snapshot in sample["memory"].values()
        )
        for sample in measured
    ]
    hip_used_peaks = [
        max(int(snapshot["hip_used_bytes"]) for snapshot in sample["memory"].values())
        for sample in measured
    ]
    resident_session_counts = sorted(
        {int(sample["route"]["resident_session_count_at_measurement"]) for sample in measured}
    )
    return {
        "configuration": config.name,
        "route": asdict(config),
        "sample_count": len(measured),
        "passed": bool(measured and all(sample.get("passed") is True for sample in measured)),
        "repeatable_trajectories": bool(
            trajectory_hashes and all(row == trajectory_hashes[0] for row in trajectory_hashes[1:])
        ),
        "stable_graph_bucket_shape_keys": bool(
            bucket_shape_keys
            and all(row == bucket_shape_keys[0] for row in bucket_shape_keys[1:])
        ),
        "measured_trajectory_hashes": trajectory_hashes,
        "graph_bucket_shape_sha256": bucket_shape_keys[0] if bucket_shape_keys else [],
        "graph_bucket_instance_key_sha256_by_run": bucket_instance_keys,
        "rates": {
            "prefill_tok_s_aggregate": _stats(
                [float(sample["throughput"]["prefill_tok_s_aggregate"]) for sample in measured]
            ),
            "decode_tok_s_aggregate": decode_aggregate,
            "decode_tok_s_per_request": _stats(
                [float(sample["throughput"]["decode_tok_s_per_request"]) for sample in measured]
            ),
        },
        "timings": {
            "prefill_seconds": _stats(
                [float(sample["timings"]["prefill_seconds"]) for sample in measured]
            ),
            "graph_capture_seconds": _stats(
                [float(sample["timings"]["graph_capture_seconds"]) for sample in measured]
            ),
            "decode_seconds": _stats(
                [float(sample["timings"]["decode_seconds"]) for sample in measured]
            ),
        },
        "latency": {
            "ttft_seconds_per_request": _stats(ttft),
            "inter_token_model_step_seconds": _stats(itl),
            "streaming_delivery_included": False,
        },
        "memory": {
            "resident_session_counts": resident_session_counts,
            "tracked_current_peak_bytes": _stats(tracked_current_peaks),
            "tracked_allocator_high_water_bytes": _stats(tracked_high_water),
            "hip_used_peak_sampled_bytes": _stats(hip_used_peaks),
        },
        "variance_guard": {
            "decode_stdev_pct_of_median": decode_aggregate["stdev_pct_of_median"],
            "limit_pct": 5.0,
            "passed": (
                decode_aggregate["stdev_pct_of_median"] is not None
                and float(decode_aggregate["stdev_pct_of_median"]) <= 5.0
            ),
        },
        "samples": [_copy_json(sample) for sample in samples],
    }


def _first_hashes(summary: Mapping[str, Any]) -> list[str] | None:
    rows = summary.get("measured_trajectory_hashes")
    if not isinstance(rows, list) or not rows or not isinstance(rows[0], list):
        return None
    return [str(value) for value in rows[0]]


def _cross_configuration_correctness(summaries: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    required = tuple(CONFIGURATIONS)
    missing = [name for name in required if name not in summaries]
    hashes = {name: _first_hashes(summary) for name, summary in summaries.items()}
    c4 = hashes.get("c4")
    direct_names = {
        1: "c1",
        2: "c2",
        3: "c3",
        4: "c4",
        5: "c5",
        6: "c6",
        7: "c7",
        8: "native_c8",
    }
    direct_width_exact: dict[str, bool] = {}
    for width, name in direct_names.items():
        actual = hashes.get(name)
        expected = (
            [str(c4[index % len(c4)]) for index in range(width)]
            if c4
            else None
        )
        direct_width_exact[str(width)] = bool(
            actual and expected and actual == expected
        )
    serial = hashes.get("serial_c4")
    chunked = hashes.get("chunked_c8")
    serial_exact = bool(c4 and serial and c4 == serial)
    chunked_exact = bool(
        c4
        and chunked
        and len(chunked) == 8
        and chunked == [str(c4[index % len(c4)]) for index in range(8)]
    )
    repeatable = all(
        summary.get("repeatable_trajectories") is True
        for summary in summaries.values()
    )
    all_direct_exact = all(direct_width_exact.values())
    passed = bool(
        not missing
        and all_direct_exact
        and serial_exact
        and chunked_exact
        and repeatable
    )
    return {
        "passed": passed,
        "missing_configurations": missing,
        "direct_c1_c8_match_c4_repeating_fixture": direct_width_exact,
        "all_direct_c1_c8_exact": all_direct_exact,
        "c1_c2_c3_c4_prefix_exact": all(
            direct_width_exact[str(width)] for width in range(1, 5)
        ),
        "c4_matches_serial_c4": serial_exact,
        "native_c8_rows_match_c4": direct_width_exact["8"],
        "chunked_c8_groups_match_c4": chunked_exact,
        "all_measured_runs_repeatable": repeatable,
    }


def _scaling_summary(summaries: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    def median(name: str, rate: str) -> float | None:
        summary = summaries.get(name)
        if not isinstance(summary, Mapping):
            return None
        value = summary.get("rates", {}).get(rate, {}).get("median")
        return float(value) if isinstance(value, (int, float)) else None

    direct_names = {
        1: "c1",
        2: "c2",
        3: "c3",
        4: "c4",
        5: "c5",
        6: "c6",
        7: "c7",
        8: "native_c8",
    }
    direct_aggregate = {
        str(width): median(name, "decode_tok_s_aggregate")
        for width, name in direct_names.items()
    }
    c1_aggregate = direct_aggregate["1"]
    serial_aggregate = median("serial_c4", "decode_tok_s_aggregate")
    c4_aggregate = median("c4", "decode_tok_s_aggregate")
    c4_per_request = median("c4", "decode_tok_s_per_request")
    native_aggregate = median("native_c8", "decode_tok_s_aggregate")
    native_per_request = median("native_c8", "decode_tok_s_per_request")
    chunked_aggregate = median("chunked_c8", "decode_tok_s_aggregate")
    c4_vs_c1 = _safe_ratio(c4_aggregate, c1_aggregate)
    c4_vs_serial = _safe_ratio(c4_aggregate, serial_aggregate)
    native_vs_c1 = _safe_ratio(native_aggregate, c1_aggregate)
    native_vs_chunked = _safe_ratio(native_aggregate, chunked_aggregate)
    native_vs_serial = _safe_ratio(native_aggregate, serial_aggregate)
    return {
        "direct_c1_c8_decode_tok_s_aggregate": direct_aggregate,
        "direct_c1_c8_scaling_vs_c1": {
            width: _safe_ratio(value, c1_aggregate)
            for width, value in direct_aggregate.items()
        },
        "c1_baseline_decode_tok_s": c1_aggregate,
        "serial_c4_baseline_decode_tok_s_aggregate": serial_aggregate,
        "c4_decode_tok_s_aggregate": c4_aggregate,
        "c4_decode_tok_s_per_request": c4_per_request,
        "native_c8_decode_tok_s_aggregate": native_aggregate,
        "native_c8_decode_tok_s_per_request": native_per_request,
        "chunked_c8_decode_tok_s_aggregate": chunked_aggregate,
        "ratios": {
            "c4_aggregate_vs_c1": c4_vs_c1,
            "c4_per_request_vs_c1": _safe_ratio(c4_per_request, c1_aggregate),
            "c4_aggregate_vs_serial_c4": c4_vs_serial,
            "native_c8_aggregate_vs_c1": native_vs_c1,
            "native_c8_per_request_vs_c1": _safe_ratio(native_per_request, c1_aggregate),
            "native_c8_aggregate_vs_chunked_c8": native_vs_chunked,
            "native_c8_aggregate_vs_serial_c4": native_vs_serial,
            "chunked_c8_aggregate_vs_c1": _safe_ratio(chunked_aggregate, c1_aggregate),
        },
        "c4_scaling_gate_passed": bool(
            c4_vs_c1 is not None
            and c4_vs_serial is not None
            and c4_vs_c1 > 1.0
            and c4_vs_serial > 1.0
        ),
        "native_c8_scaling_gate_passed": bool(
            native_vs_c1 is not None
            and native_vs_chunked is not None
            and native_vs_serial is not None
            and native_vs_c1 > 1.0
            and native_vs_chunked > 1.0
            and native_vs_serial > 1.0
        ),
        "native_c8_is_one_physical_group": True,
        "chunked_c8_is_native_c8": False,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if int(args.prompt_length) < 4:
        raise ValueError("prompt-length must be at least four")
    if int(args.decode_steps) <= 0:
        raise ValueError("decode-steps must be positive")
    if int(args.prompt_length) + int(args.decode_steps) >= 1024:
        raise ValueError("packed graph benchmark currently requires context < 1024")
    if int(args.warmup_runs) < 0 or int(args.measured_runs) <= 0:
        raise ValueError("warmup-runs must be non-negative and measured-runs must be positive")
    names = _parse_configurations(args.configurations)
    model = Path(args.model).expanduser().resolve()
    if not model.is_file():
        raise ValueError(f"model does not exist: {model}")
    compiler_version = _read_compiler_version(args.compiler_version_file)
    if bool(args.require_cached_build) and compiler_version is None:
        raise ValueError("require-cached-build requires compiler-version-file")

    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import reset_memory_stats
    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession

    runtime = get_hip_runtime()
    reset_memory_stats()
    memory: dict[str, Any] = {"before_load": _memory_snapshot("before_load", runtime)}
    max_rows = max(CONFIGURATIONS[name].logical_rows for name in names)
    max_sequence_length = int(args.prompt_length) + int(args.decode_steps) + 2
    prompts = _prompt_rows(
        rows=max_rows,
        prompt_length=int(args.prompt_length),
        token_id=int(args.prompt_token_id),
    )
    samples_by_name: dict[str, list[dict[str, Any]]] = {name: [] for name in names}
    resolved_backend = str(args.backend)
    target_arch = resolved_backend.removeprefix("hip_")

    stack = ExitStack()
    route_env = _EXACT_ENV if args.route == "exact" else _PRODUCTION_ENV
    try:
        with _temporary_environment(route_env):
            owner = stack.enter_context(
                Qwen35GGUFResidentSession(
                    model,
                    backend=str(args.backend),
                    compiler_version=compiler_version,
                    require_cached_build=bool(args.require_cached_build),
                    max_sequence_length=max_sequence_length,
                    use_wmma_prefill=True,
                    use_gemv_decode=True,
                )
            )
            if owner.runner is None:
                raise RuntimeError("GGUF packed benchmark owner runner is closed")
            sessions = [owner]
            resolved_backend = str(owner.backend)
            target_arch = str(owner.runner.target_arch)
            memory["after_load"] = _memory_snapshot("after_load", runtime)
            for name in names:
                config = CONFIGURATIONS[name]
                while len(sessions) < config.logical_rows:
                    sessions.append(
                        stack.enter_context(
                            Qwen35GGUFResidentSession(
                                model,
                                backend=str(args.backend),
                                runtime=owner.runtime,
                                shared_runner=owner.runner,
                                compiler_version=compiler_version,
                                require_cached_build=bool(args.require_cached_build),
                                max_sequence_length=max_sequence_length,
                                use_wmma_prefill=True,
                                use_gemv_decode=True,
                            )
                        )
                    )
                memory[f"before_{name}_with_{len(sessions)}_sessions"] = _memory_snapshot(
                    f"before_{name}_with_{len(sessions)}_sessions",
                    runtime,
                )
                for raw_index in range(int(args.warmup_runs) + int(args.measured_runs)):
                    measured = raw_index >= int(args.warmup_runs)
                    run_index = (
                        raw_index - int(args.warmup_runs) + 1
                        if measured
                        else raw_index + 1
                    )
                    sample = _run_sample(
                        config=config,
                        sessions=sessions,
                        prompts=prompts,
                        decode_steps=int(args.decode_steps),
                        measured=measured,
                        run_index=run_index,
                    )
                    samples_by_name[name].append(sample)
                    print(
                        f"{name} {'measured' if measured else 'warmup'} {run_index}: "
                        f"aggregate={sample['throughput']['decode_tok_s_aggregate']:.6f} "
                        f"per_request={sample['throughput']['decode_tok_s_per_request']:.6f}",
                        file=sys.stderr,
                        flush=True,
                    )
            memory["before_close"] = _memory_snapshot("before_close", runtime)
    finally:
        stack.close()
        memory["after_close"] = _memory_snapshot("after_close", runtime)

    summaries = {
        name: _summarize_configuration(CONFIGURATIONS[name], samples_by_name[name])
        for name in names
    }
    cross_correctness = _cross_configuration_correctness(summaries)
    scaling = _scaling_summary(summaries)
    summaries_passed = all(
        summary["passed"] is True
        and summary["repeatable_trajectories"] is True
        and summary["stable_graph_bucket_shape_keys"] is True
        and summary["variance_guard"]["passed"] is True
        for summary in summaries.values()
    )
    complete_packet = set(names) == set(CONFIGURATIONS)
    passed = bool(
        summaries_passed
        and (cross_correctness["passed"] if complete_packet else True)
    )
    command = [sys.executable, "scripts/gguf_packed_ar_bench.py", *sys.argv[1:]]
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
            **route_env,
        },
        build_profile=f"{target_arch}_gguf_packed_graph_direct_c1_c8_controls",
        timing_protocol=(
            "one shared model load; one discarded run and measured repeats per route; "
            "one synchronized graph replay per native group per logical decode transition"
        ),
        warmups=int(args.warmup_runs),
        repetitions=int(args.measured_runs),
        profiler={"used": False, "reason": "host-wall scaling and latency packet"},
        hipcc_version=compiler_version,
    )
    peak_hip_used = max(int(row["hip_used_bytes"]) for row in memory.values())
    return {
        "schema": 1,
        "kind": f"{target_arch}_gguf_native_c8_graph_scaling_packet",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": (
            "measurement_complete"
            if passed and complete_packet
            else "partial_measurement_complete"
            if passed
            else "failed"
        ),
        "passed": passed,
        "complete_packet": complete_packet,
        "performance_claim": False,
        "provenance": provenance,
        "workload": {
            "model": str(model),
            "backend": resolved_backend,
            "target_arch": target_arch,
            "quant": str(args.quant),
            "kv_dtype": "bf16",
            "prompt_length": int(args.prompt_length),
            "decode_steps": int(args.decode_steps),
            "prompt_token_id": int(args.prompt_token_id),
            "prompt_rows_sha256": _prompt_fingerprint(prompts),
            "prompt_terminal_token_ids": [int(prompt[-1]) for prompt in prompts],
            "sampling": "greedy_top1",
            "speculative_decode": False,
            "warmup_runs": int(args.warmup_runs),
            "measured_runs": int(args.measured_runs),
            "configurations": list(names),
        },
        "configuration_contracts": {name: asdict(CONFIGURATIONS[name]) for name in names},
        "summaries": summaries,
        "cross_configuration_correctness": cross_correctness,
        "scaling": scaling,
        "memory": {
            "snapshots": memory,
            "tracked_peak_allocated_bytes": max(
                int(row["tracked"]["peak_allocated_bytes"]) for row in memory.values()
            ),
            "hip_used_peak_sampled_bytes": peak_hip_used,
            "tracked_scope": "hipengine core allocator",
            "hip_scope": "hipMemGetInfo process/device-visible phase-boundary samples",
        },
        "command": shlex.join(command),
        "limitations": [
            "Raw native-c8 measurement packet; correctness and marker-profiler artifacts are joined separately before promotion.",
            "Inter-token latency is synchronized model-step completion latency; streaming token delivery and per-token D2H are Phase D.",
            "native_c8 is one physical eight-row graph; chunked_c8 remains two serial c4 groups as an honest control.",
            "The deterministic four-row raw-token fixture repeats once across both c8 halves; category diversity is gated separately.",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--backend", choices=SUPPORTED_BACKENDS, default="hip_gfx1100")
    parser.add_argument("--quant", default="gguf_q4_k_m")
    parser.add_argument(
        "--configurations",
        default="c1,c2,c3,c4,c5,c6,c7,native_c8,chunked_c8,serial_c4",
        help=(
            "Comma-separated subset of c1,c2,c3,c4,c5,c6,c7,native_c8,"
            "chunked_c8,serial_c4."
        ),
    )
    parser.add_argument("--prompt-token-id", type=int, default=9707)
    parser.add_argument("--prompt-length", type=int, default=512)
    parser.add_argument("--decode-steps", type=int, default=128)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--measured-runs", type=int, default=3)
    parser.add_argument(
        "--route",
        choices={"exact", "production"},
        default="exact",
        help="exact = strict verify-capture state-rows prefill (retained packet); "
        "production = backend auto GDN prefill route (no forced capture env).",
    )
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--json", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = run(args)
    text = json.dumps(payload, indent=2, allow_nan=False)
    print(text)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n", encoding="utf-8")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
