#!/usr/bin/env python3
"""SOL-G5 production GGUF decode-graph correctness and wall audit.

The audit drives ``Qwen35GGUFResidentSession.capture_decode_graph()`` and
compares two routes:

* one state-bound graph relaunched for every token, with explicit checks after
  the third and every later launch; and
* a conservative state-generation-keyed route that captures a fresh graph for
  every token and therefore never reuses a graph across a state transition.

Every correctness checkpoint fingerprints the FP32 output-normalized hidden
seed, every resident Conv/GDN state pair, all live BF16 full-attention K/V rows,
and the generated token.  Promotion requires exact long-replay state and a
capture-inclusive wall win over current eager decode.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_MODEL = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
KIND = "hipengine_gguf_decode_graph_sol_g5_audit"
SCHEMA_VERSION = 1


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bf16_to_f32(raw: np.ndarray) -> np.ndarray:
    words = np.ascontiguousarray(raw, dtype=np.uint8).view("<u2").astype(np.uint32)
    return np.ascontiguousarray((words << np.uint32(16)).view(np.float32))


def _fingerprint_raw(raw: np.ndarray, *, dtype: str) -> dict[str, Any]:
    raw_u8 = np.ascontiguousarray(raw, dtype=np.uint8).reshape(-1)
    if dtype == "fp32":
        if raw_u8.size % np.dtype(np.float32).itemsize:
            raise ValueError("FP32 buffer is not element-aligned")
        values = raw_u8.view("<f4")
    elif dtype == "bf16":
        if raw_u8.size % np.dtype(np.uint16).itemsize:
            raise ValueError("BF16 buffer is not element-aligned")
        values = _bf16_to_f32(raw_u8)
    else:
        raise ValueError(f"unsupported checkpoint dtype: {dtype}")
    finite = bool(np.all(np.isfinite(values)))
    values64 = values.astype(np.float64, copy=False)
    rms = float(math.sqrt(float(np.mean(values64 * values64)))) if values.size else 0.0
    max_abs = float(np.max(np.abs(values64))) if values.size else 0.0
    return {
        "nbytes": int(raw_u8.size),
        "blake2b_128": hashlib.blake2b(raw_u8.tobytes(), digest_size=16).hexdigest(),
        "finite": finite,
        "rms": rms,
        "max_abs": max_abs,
    }


def _copy_device_fingerprint(
    session: Any,
    *,
    ptr: int,
    nbytes: int,
    dtype: str,
) -> dict[str, Any]:
    from hipengine.core.memory import DeviceBuffer, copy_device_to_host, host_array_ptr

    size = int(nbytes)
    raw = np.empty((size,), dtype=np.uint8)
    if size:
        copy_device_to_host(
            host_array_ptr(raw),
            DeviceBuffer(int(ptr), size),
            size,
            runtime=session.runtime,
        )
    return _fingerprint_raw(raw, dtype=dtype)


def _capture_checkpoint(
    session: Any,
    *,
    position: int,
    input_token_id: int,
    predicted_token_id: int,
) -> dict[str, Any]:
    from hipengine.core.dtype import DType

    if session.runner is None or session.runner.weights is None or session.scratch is None:
        raise RuntimeError("GGUF resident session is closed")
    session.runtime.device_synchronize()
    runner = session.runner
    scratch = session.scratch
    hidden_seed = _copy_device_fingerprint(
        session,
        ptr=int(scratch.hidden_seed_fp32.ptr),
        nbytes=int(runner.hidden_size) * DType.FP32.itemsize,
        dtype="fp32",
    )
    linear_states: list[dict[str, Any]] = []
    for layer_id, (conv_state, recurrent_state) in enumerate(
        zip(scratch.layer_conv_states, scratch.layer_recurrent_states, strict=True)
    ):
        if conv_state is None or recurrent_state is None:
            continue
        linear_states.append(
            {
                "layer": int(layer_id),
                "conv": _copy_device_fingerprint(
                    session,
                    ptr=int(conv_state.ptr),
                    nbytes=int(conv_state.nbytes),
                    dtype="fp32",
                ),
                "recurrent": _copy_device_fingerprint(
                    session,
                    ptr=int(recurrent_state.ptr),
                    nbytes=int(recurrent_state.nbytes),
                    dtype="fp32",
                ),
            }
        )

    cfg = runner.weights.config
    live_positions = int(position)
    kv_row_nbytes = int(cfg.head_count_kv) * int(cfg.key_length) * DType.BF16.itemsize
    live_nbytes = live_positions * kv_row_nbytes
    kv_states: list[dict[str, Any]] = []
    for layer_id, (key_cache, value_cache) in enumerate(
        zip(scratch.full_key_caches, scratch.full_value_caches, strict=True)
    ):
        if key_cache is None or value_cache is None:
            continue
        checked_nbytes = min(live_nbytes, int(key_cache.nbytes), int(value_cache.nbytes))
        kv_states.append(
            {
                "layer": int(layer_id),
                "live_positions": live_positions,
                "key": _copy_device_fingerprint(
                    session,
                    ptr=int(key_cache.ptr),
                    nbytes=checked_nbytes,
                    dtype="bf16",
                ),
                "value": _copy_device_fingerprint(
                    session,
                    ptr=int(value_cache.ptr),
                    nbytes=checked_nbytes,
                    dtype="bf16",
                ),
            }
        )
    fingerprints = [
        hidden_seed,
        *(row[part] for row in linear_states for part in ("conv", "recurrent")),
        *(row[part] for row in kv_states for part in ("key", "value")),
    ]
    return {
        "position": live_positions,
        "input_token_id": int(input_token_id),
        "predicted_token_id": int(predicted_token_id),
        "finite": all(bool(fingerprint["finite"]) for fingerprint in fingerprints),
        "hidden_seed": hidden_seed,
        "linear_states": linear_states,
        "kv_states": kv_states,
    }


def _same_fingerprint(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return (
        left.get("nbytes") == right.get("nbytes")
        and left.get("blake2b_128") == right.get("blake2b_128")
    )


def _layer_map(rows: Any, *, label: str) -> dict[int, Mapping[str, Any]]:
    if not isinstance(rows, list):
        raise ValueError(f"{label} must be a list")
    result: dict[int, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or isinstance(row.get("layer"), bool):
            raise ValueError(f"{label} has an invalid layer row")
        layer = int(row["layer"])
        if layer in result:
            raise ValueError(f"{label} has duplicate layer {layer}")
        result[layer] = row
    return result


def _compare_checkpoints(reference: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    mismatches: list[dict[str, Any]] = []

    def add(component: str, *, layer: int | None, part: str | None, expected: Any, actual: Any) -> None:
        mismatches.append(
            {
                "component": component,
                "layer": layer,
                "part": part,
                "expected": expected,
                "actual": actual,
            }
        )

    for key, component in (
        ("position", "position"),
        ("input_token_id", "input_token"),
        ("predicted_token_id", "predicted_token"),
    ):
        if reference.get(key) != candidate.get(key):
            add(
                component,
                layer=None,
                part=None,
                expected=reference.get(key),
                actual=candidate.get(key),
            )
    if reference.get("finite") is not True or candidate.get("finite") is not True:
        add(
            "nonfinite",
            layer=None,
            part=None,
            expected=reference.get("finite"),
            actual=candidate.get("finite"),
        )
    if not _same_fingerprint(reference.get("hidden_seed", {}), candidate.get("hidden_seed", {})):
        add(
            "hidden_seed",
            layer=None,
            part=None,
            expected=reference.get("hidden_seed", {}).get("blake2b_128"),
            actual=candidate.get("hidden_seed", {}).get("blake2b_128"),
        )

    reference_linear = _layer_map(reference.get("linear_states"), label="reference.linear_states")
    candidate_linear = _layer_map(candidate.get("linear_states"), label="candidate.linear_states")
    for layer in sorted(set(reference_linear) | set(candidate_linear)):
        expected_row = reference_linear.get(layer)
        actual_row = candidate_linear.get(layer)
        for part in ("conv", "recurrent"):
            if expected_row is None or actual_row is None or not _same_fingerprint(
                expected_row.get(part, {}), actual_row.get(part, {})
            ):
                add(
                    "linear_state",
                    layer=layer,
                    part=part,
                    expected=None if expected_row is None else expected_row.get(part, {}).get("blake2b_128"),
                    actual=None if actual_row is None else actual_row.get(part, {}).get("blake2b_128"),
                )

    reference_kv = _layer_map(reference.get("kv_states"), label="reference.kv_states")
    candidate_kv = _layer_map(candidate.get("kv_states"), label="candidate.kv_states")
    for layer in sorted(set(reference_kv) | set(candidate_kv)):
        expected_row = reference_kv.get(layer)
        actual_row = candidate_kv.get(layer)
        if (
            expected_row is not None
            and actual_row is not None
            and expected_row.get("live_positions") != actual_row.get("live_positions")
        ):
            add(
                "full_attention_kv",
                layer=layer,
                part="live_positions",
                expected=expected_row.get("live_positions"),
                actual=actual_row.get("live_positions"),
            )
        for part in ("key", "value"):
            if expected_row is None or actual_row is None or not _same_fingerprint(
                expected_row.get(part, {}), actual_row.get(part, {})
            ):
                add(
                    "full_attention_kv",
                    layer=layer,
                    part=part,
                    expected=None if expected_row is None else expected_row.get(part, {}).get("blake2b_128"),
                    actual=None if actual_row is None else actual_row.get(part, {}).get("blake2b_128"),
                )

    component_order = {
        "position": 0,
        "input_token": 1,
        "predicted_token": 2,
        "nonfinite": 3,
        "hidden_seed": 4,
        "linear_state": 5,
        "full_attention_kv": 6,
    }
    mismatches.sort(
        key=lambda row: (
            -1 if row["layer"] is None else int(row["layer"]),
            component_order.get(str(row["component"]), 99),
            "" if row["part"] is None else str(row["part"]),
        )
    )
    first = None
    if mismatches:
        first = {
            "component": mismatches[0]["component"],
            "layer": mismatches[0]["layer"],
            "part": mismatches[0]["part"],
        }
    return {"passed": not mismatches, "mismatches": mismatches, "first_divergence": first}


def _summarize_runs(runs: Sequence[Mapping[str, Any]], *, expected_token_id: int) -> dict[str, Any]:
    if not runs:
        raise ValueError("at least one timing run is required")
    ms_per_token: list[float] = []
    tok_s: list[float] = []
    for run_index, run in enumerate(runs):
        steps = int(run["steps"])
        wall_ms = float(run["wall_ms"])
        tokens = [int(token) for token in run["generated_token_ids"]]
        if steps <= 0 or wall_ms <= 0.0 or len(tokens) != steps:
            raise ValueError(f"invalid timing run {run_index}")
        for token_index, token in enumerate(tokens):
            if token != int(expected_token_id):
                raise ValueError(
                    f"unexpected token in timing run {run_index}, step {token_index}: "
                    f"expected {expected_token_id}, observed {token}"
                )
        per_token = wall_ms / steps
        ms_per_token.append(per_token)
        tok_s.append(1000.0 / per_token)
    return {
        "runs": len(runs),
        "all_tokens_exact": True,
        "median_ms_per_token": statistics.median(ms_per_token),
        "median_tok_s": statistics.median(tok_s),
        "min_ms_per_token": min(ms_per_token),
        "max_ms_per_token": max(ms_per_token),
        "samples_ms_per_token": ms_per_token,
        "samples_tok_s": tok_s,
    }


def _classify_candidate(
    *,
    relaunch_passed: bool,
    relaunch_first_failure: int | None,
    recapture_passed: bool,
    eager_summary: Mapping[str, Any],
    recapture_summary: Mapping[str, Any],
    relaunch_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    eager_ms = float(eager_summary["median_ms_per_token"])
    candidate_summary = relaunch_summary if relaunch_passed and relaunch_summary is not None else recapture_summary
    candidate_ms = float(candidate_summary["median_ms_per_token"])
    speedup = eager_ms / candidate_ms
    delta_pct = (candidate_ms / eager_ms - 1.0) * 100.0
    common = {
        "eager_median_ms_per_token": eager_ms,
        "candidate_median_ms_per_token": candidate_ms,
        "candidate_speedup_vs_eager": speedup,
        "candidate_wall_delta_pct": delta_pct,
    }
    if not relaunch_passed:
        return {
            **common,
            "status": "rejected",
            "decision": "reject_third_or_later_relaunch_state_corruption",
            "first_failing_launch": relaunch_first_failure,
        }
    if not recapture_passed:
        return {
            **common,
            "status": "rejected",
            "decision": "reject_state_keyed_recapture_not_exact",
            "first_failing_launch": None,
        }
    if candidate_ms >= eager_ms:
        return {
            **common,
            "status": "rejected",
            "decision": "reject_no_end_to_end_wall_win",
            "first_failing_launch": None,
        }
    return {
        **common,
        "status": "accepted",
        "decision": "promote_state_bound_graph_relaunch",
        "first_failing_launch": None,
    }


def _checkpoint_summary(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    linear_states = list(checkpoint.get("linear_states", ()))
    kv_states = list(checkpoint.get("kv_states", ()))
    digest_payload = {
        "position": checkpoint.get("position"),
        "input_token_id": checkpoint.get("input_token_id"),
        "predicted_token_id": checkpoint.get("predicted_token_id"),
        "hidden_seed": checkpoint.get("hidden_seed", {}).get("blake2b_128"),
        "linear_states": [
            {
                "layer": row.get("layer"),
                "conv": row.get("conv", {}).get("blake2b_128"),
                "recurrent": row.get("recurrent", {}).get("blake2b_128"),
            }
            for row in linear_states
        ],
        "kv_states": [
            {
                "layer": row.get("layer"),
                "live_positions": row.get("live_positions"),
                "key": row.get("key", {}).get("blake2b_128"),
                "value": row.get("value", {}).get("blake2b_128"),
            }
            for row in kv_states
        ],
    }
    return {
        "position": int(checkpoint["position"]),
        "input_token_id": int(checkpoint["input_token_id"]),
        "predicted_token_id": int(checkpoint["predicted_token_id"]),
        "finite": bool(checkpoint["finite"]),
        "linear_state_pairs": len(linear_states),
        "full_attention_kv_pairs": len(kv_states),
        "state_sha256": _canonical_sha256(digest_payload),
    }


def _prefill(session: Any, prompt_ids: Sequence[int]) -> int:
    session.reset()
    result = session.prefill(
        [int(token) for token in prompt_ids],
        use_bulk=True,
        return_logits=False,
        capture_hidden_seed_fp32=True,
    )
    session.runtime.device_synchronize()
    return int(result.token_id)


def _run_eager_correctness(session: Any, *, prompt_ids: Sequence[int], steps: int) -> dict[str, Any]:
    current = _prefill(session, prompt_ids)
    checkpoints: list[dict[str, Any]] = []
    generated: list[int] = []
    for _ in range(int(steps)):
        input_token = current
        result = session.step(
            input_token,
            return_logits=False,
            capture_hidden_seed_fp32=True,
        )
        current = int(result.token_id)
        generated.append(current)
        checkpoints.append(
            _capture_checkpoint(
                session,
                position=int(session.position),
                input_token_id=input_token,
                predicted_token_id=current,
            )
        )
    return {
        "prefill_token_id": int(checkpoints[0]["input_token_id"]) if checkpoints else current,
        "generated_token_ids": generated,
        "checkpoints": checkpoints,
    }


def _run_relaunch_correctness(
    session: Any,
    *,
    prompt_ids: Sequence[int],
    steps: int,
    reference: Sequence[Mapping[str, Any]],
    submission_transport: str | None = None,
) -> dict[str, Any]:
    current = _prefill(session, prompt_ids)
    start_position = int(session.position)
    checkpoint_summaries: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    generated: list[int] = []
    with session.capture_decode_graph(
        position=start_position,
        max_replay_steps=int(steps),
        steps_per_replay=1,
        attention_max_context_len=start_position + int(steps),
        capture_hidden_seed_fp32=True,
        submission_transport=submission_transport,
    ) as graph:
        key = graph.bucket_key.as_dict()
        for launch_index in range(1, int(steps) + 1):
            input_token = current
            graph.replay(1)
            current = int(graph.read_sample(return_logits=False).token_id)
            checkpoint = _capture_checkpoint(
                session,
                position=int(session.position),
                input_token_id=input_token,
                predicted_token_id=current,
            )
            comparison = _compare_checkpoints(reference[launch_index - 1], checkpoint)
            checkpoint_summaries.append(_checkpoint_summary(checkpoint))
            comparisons.append(
                {
                    "launch": launch_index,
                    "passed": bool(comparison["passed"]),
                    "first_divergence": comparison["first_divergence"],
                    "mismatch_count": len(comparison["mismatches"]),
                    "mismatches": comparison["mismatches"],
                }
            )
            generated.append(current)
        live_transport_provenance = graph.transport_provenance()
    closed_transport_provenance = graph.transport_provenance()
    first_failure = next((row["launch"] for row in comparisons if not row["passed"]), None)
    return {
        "passed": first_failure is None,
        "first_failing_launch": first_failure,
        "third_and_later_launches_checked": max(0, int(steps) - 2),
        "graph_key": key,
        "generated_token_ids": generated,
        "transport_provenance": {
            "live": live_transport_provenance,
            "closed": closed_transport_provenance,
        },
        "comparisons": comparisons,
        "checkpoint_summaries": checkpoint_summaries,
    }


def _compact_eager_correctness(eager: Mapping[str, Any]) -> dict[str, Any]:
    checkpoints = list(eager.get("checkpoints", ()))
    return {
        "prefill_token_id": int(eager["prefill_token_id"]),
        "generated_token_ids": [int(token) for token in eager["generated_token_ids"]],
        "checkpoint_summaries": [_checkpoint_summary(checkpoint) for checkpoint in checkpoints],
    }


def _run_recapture_correctness(
    session: Any,
    *,
    prompt_ids: Sequence[int],
    steps: int,
    reference: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    current = _prefill(session, prompt_ids)
    comparisons: list[dict[str, Any]] = []
    generated: list[int] = []
    key_digests: list[str] = []
    for step_index in range(int(steps)):
        position = int(session.position)
        input_token = current
        with session.capture_decode_graph(
            position=position,
            max_replay_steps=1,
            steps_per_replay=1,
            attention_max_context_len=position + 1,
            capture_hidden_seed_fp32=True,
        ) as graph:
            key_digests.append(str(graph.bucket_key.key_sha256))
            graph.replay(1)
            current = int(graph.read_sample(return_logits=False).token_id)
        checkpoint = _capture_checkpoint(
            session,
            position=int(session.position),
            input_token_id=input_token,
            predicted_token_id=current,
        )
        comparison = _compare_checkpoints(reference[step_index], checkpoint)
        comparisons.append(
            {
                "step": step_index + 1,
                "passed": bool(comparison["passed"]),
                "first_divergence": comparison["first_divergence"],
                "mismatch_count": len(comparison["mismatches"]),
                "mismatches": comparison["mismatches"],
            }
        )
        generated.append(current)
    return {
        "passed": all(bool(row["passed"]) for row in comparisons),
        "capture_count": int(steps),
        "launches_per_capture": 1,
        "unique_key_count": len(set(key_digests)),
        "key_sha256": key_digests,
        "generated_token_ids": generated,
        "comparisons": comparisons,
    }


def _run_eager_timing(session: Any, *, prompt_ids: Sequence[int], steps: int) -> dict[str, Any]:
    current = _prefill(session, prompt_ids)
    generated: list[int] = []
    session.runtime.device_synchronize()
    start_ns = time.perf_counter_ns()
    for _ in range(int(steps)):
        current = int(session.step(current, return_logits=False).token_id)
        generated.append(current)
    session.runtime.device_synchronize()
    wall_ms = (time.perf_counter_ns() - start_ns) / 1e6
    return {"steps": int(steps), "wall_ms": wall_ms, "generated_token_ids": generated}


def _run_recapture_timing(
    session: Any,
    *,
    prompt_ids: Sequence[int],
    steps: int,
) -> dict[str, Any]:
    current = _prefill(session, prompt_ids)
    generated: list[int] = []
    capture_ms: list[float] = []
    session.runtime.device_synchronize()
    start_ns = time.perf_counter_ns()
    for _ in range(int(steps)):
        position = int(session.position)
        capture_start_ns = time.perf_counter_ns()
        graph = session.capture_decode_graph(
            position=position,
            max_replay_steps=1,
            steps_per_replay=1,
            attention_max_context_len=position + 1,
        )
        capture_ms.append((time.perf_counter_ns() - capture_start_ns) / 1e6)
        try:
            graph.replay(1)
            current = int(graph.read_sample(return_logits=False).token_id)
        finally:
            graph.close()
        generated.append(current)
    session.runtime.device_synchronize()
    wall_ms = (time.perf_counter_ns() - start_ns) / 1e6
    return {
        "steps": int(steps),
        "wall_ms": wall_ms,
        "generated_token_ids": generated,
        "capture_count": int(steps),
        "capture_ms": capture_ms,
        "capture_ms_total": sum(capture_ms),
    }


def _run_relaunch_timing(
    session: Any,
    *,
    prompt_ids: Sequence[int],
    steps: int,
) -> dict[str, Any]:
    current = _prefill(session, prompt_ids)
    start_position = int(session.position)
    inclusive_start_ns = time.perf_counter_ns()
    capture_start_ns = inclusive_start_ns
    graph = session.capture_decode_graph(
        position=start_position,
        max_replay_steps=int(steps),
        steps_per_replay=1,
        attention_max_context_len=start_position + int(steps),
    )
    capture_ms = (time.perf_counter_ns() - capture_start_ns) / 1e6
    generated: list[int] = []
    try:
        session.runtime.device_synchronize()
        start_ns = time.perf_counter_ns()
        for _ in range(int(steps)):
            graph.replay(1)
            current = int(graph.read_sample(return_logits=False).token_id)
            generated.append(current)
        session.runtime.device_synchronize()
        replay_wall_ms = (time.perf_counter_ns() - start_ns) / 1e6
    finally:
        graph.close()
    inclusive_wall_ms = (time.perf_counter_ns() - inclusive_start_ns) / 1e6
    return {
        "steps": int(steps),
        "wall_ms": inclusive_wall_ms,
        "generated_token_ids": generated,
        "capture_ms": capture_ms,
        "replay_wall_ms": replay_wall_ms,
        "replay_ms_per_token": replay_wall_ms / int(steps),
    }


def _read_compiler_version(path: Path | None) -> str | None:
    if path is None:
        return None
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"empty compiler-version file: {path}")
    return text


def _run(args: argparse.Namespace) -> int:
    os.environ["HIPENGINE_GGUF_DECODE_REPACK"] = "1"
    os.environ["HIPENGINE_GGUF_MOE_GRAPH"] = "0"
    target_arch = "gfx1151" if args.backend == "hip_gfx1151" else "gfx1100"
    os.environ["HIPENGINE_HIP_ARCH"] = target_arch
    if args.compiler_version_file is not None:
        os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(args.compiler_version_file.resolve())

    from hipengine.benchmark.provenance import collect_artifact_provenance
    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession

    model = Path(args.model).resolve()
    if not model.is_file():
        raise FileNotFoundError(model)
    model_sha256 = _file_sha256(model)
    prompt_ids = [int(args.prompt_token_id)] * int(args.prompt_length)
    max_sequence_length = (
        int(args.max_sequence_length)
        if int(args.max_sequence_length) > 0
        else int(args.prompt_length) + max(int(args.correctness_steps), int(args.timing_steps)) + 8
    )
    compiler_version = _read_compiler_version(args.compiler_version_file)

    with Qwen35GGUFResidentSession(
        model,
        max_sequence_length=max_sequence_length,
        compiler_version=compiler_version,
        require_cached_build=bool(args.require_cached),
        backend=str(args.backend),
        use_wmma_prefill=True,
        use_gemv_decode=True,
    ) as session:
        eager_correctness = _run_eager_correctness(
            session,
            prompt_ids=prompt_ids,
            steps=int(args.correctness_steps),
        )
        expected_tokens = [int(args.expected_token_id)] * int(args.correctness_steps)
        if eager_correctness["generated_token_ids"] != expected_tokens:
            raise RuntimeError(
                "current eager correctness trajectory differs from the required repeated-token oracle"
            )
        reference_checkpoints = eager_correctness["checkpoints"]
        relaunch = _run_relaunch_correctness(
            session,
            prompt_ids=prompt_ids,
            steps=int(args.correctness_steps),
            reference=reference_checkpoints,
        )
        recapture = _run_recapture_correctness(
            session,
            prompt_ids=prompt_ids,
            steps=int(args.correctness_steps),
            reference=reference_checkpoints,
        )

        eager_warmups: list[dict[str, Any]] = []
        relaunch_warmups: list[dict[str, Any]] = []
        recapture_warmups: list[dict[str, Any]] = []
        eager_runs: list[dict[str, Any]] = []
        relaunch_runs: list[dict[str, Any]] = []
        recapture_runs: list[dict[str, Any]] = []
        base_modes = ["eager", "relaunch", "recapture"] if bool(relaunch["passed"]) else ["eager", "recapture"]
        for run_index in range(int(args.warmups) + int(args.repetitions)):
            measured = run_index >= int(args.warmups)
            rotate = run_index % len(base_modes)
            modes = [*base_modes[rotate:], *base_modes[:rotate]]
            for mode in modes:
                if mode == "eager":
                    row = _run_eager_timing(session, prompt_ids=prompt_ids, steps=int(args.timing_steps))
                    (eager_runs if measured else eager_warmups).append(row)
                elif mode == "relaunch":
                    row = _run_relaunch_timing(
                        session,
                        prompt_ids=prompt_ids,
                        steps=int(args.timing_steps),
                    )
                    (relaunch_runs if measured else relaunch_warmups).append(row)
                else:
                    row = _run_recapture_timing(
                        session,
                        prompt_ids=prompt_ids,
                        steps=int(args.timing_steps),
                    )
                    (recapture_runs if measured else recapture_warmups).append(row)

    eager_summary = _summarize_runs(eager_runs, expected_token_id=int(args.expected_token_id))
    recapture_summary = _summarize_runs(recapture_runs, expected_token_id=int(args.expected_token_id))
    relaunch_summary = (
        _summarize_runs(relaunch_runs, expected_token_id=int(args.expected_token_id))
        if relaunch_runs
        else None
    )
    classification = _classify_candidate(
        relaunch_passed=bool(relaunch["passed"]),
        relaunch_first_failure=relaunch["first_failing_launch"],
        recapture_passed=bool(recapture["passed"]),
        eager_summary=eager_summary,
        recapture_summary=recapture_summary,
        relaunch_summary=relaunch_summary,
    )

    provenance = collect_artifact_provenance(
        repo_root=REPO_ROOT,
        configured_backend=str(args.backend),
        resolved_backend=str(args.backend),
        target_arch=target_arch,
        model_path=model,
        quant=str(args.quant),
        kv_dtype="bf16",
        command=[str(part) for part in sys.argv],
        environment={
            "HIPENGINE_BACKEND": os.environ.get("HIPENGINE_BACKEND"),
            "HIPENGINE_HIP_ARCH": target_arch,
            "HIPENGINE_GGUF_DECODE_REPACK": "1",
            "HIPENGINE_GGUF_MOE_GRAPH": "0",
            "HIPENGINE_GGUF_WMMA_PREFILL": "1 (constructor)",
            "HIPENGINE_GGUF_GEMV_DECODE": "1 (constructor)",
        },
        build_profile="gguf_decode_graph_sol_g5_audit",
        timing_protocol=(
            "one resident session; rotating eager, state-bound relaunch, and per-token state-keyed capture; "
            "capture/instantiate/launch/sync/destroy included for graph candidates"
        ),
        warmups=int(args.warmups),
        repetitions=int(args.repetitions),
        profiler={"enabled": False, "reason": "G5 is a correctness and host-wall promotion gate"},
    )
    evidence_valid = bool(
        not provenance["dirty"]
        and eager_summary["all_tokens_exact"]
        and recapture_summary["all_tokens_exact"]
        and bool(recapture["passed"])
        and len(relaunch["comparisons"]) == int(args.correctness_steps)
        and int(relaunch["third_and_later_launches_checked"]) >= 1
        and recapture["unique_key_count"] == int(args.correctness_steps)
    )
    artifact = {
        "kind": KIND,
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": classification["status"] if evidence_valid else "diagnostic_invalid",
        "performance_claim": bool(evidence_valid),
        "correctness_claim": bool(evidence_valid),
        "classification": {
            **classification,
            "evidence_valid": evidence_valid,
            "conclusion": (
                "The state-bound composite graph is not promotable unless its declared transition "
                "window survives third-and-later launches exactly and its capture-inclusive wall beats eager. "
                "Per-generation recapture remains a separately measured conservative fallback."
            ),
        },
        "workload": {
            "model": str(model),
            "model_sha256": model_sha256,
            "quant": str(args.quant),
            "kv_dtype": "bf16",
            "backend": str(args.backend),
            "target_arch": target_arch,
            "prompt_source": "repeated_token_id",
            "prompt_token_id": int(args.prompt_token_id),
            "expected_token_id": int(args.expected_token_id),
            "prompt_length": int(args.prompt_length),
            "correctness_steps": int(args.correctness_steps),
            "timing_steps": int(args.timing_steps),
            "max_sequence_length": max_sequence_length,
            "route": "repacked WMMA bulk prefill + GEMV decode; per-layer MoE graph off",
        },
        "graph_key_contract": {
            "description": (
                "Backend/model/quant/KV/shape/layer/weight-role/route/buffer identity plus state generation. "
                "A state-generation-keyed cache intentionally misses after every token mutation."
            ),
            "stable_relaunch_key": relaunch["graph_key"],
            "recapture_key_sha256": recapture["key_sha256"],
            "recapture_unique_key_count": recapture["unique_key_count"],
        },
        "correctness": {
            "protocol": (
                "Byte-exact FP32 hidden seed, every Conv/GDN state pair, all live BF16 K/V rows, "
                "and token after each launch versus current eager from the same resident session."
            ),
            "eager": _compact_eager_correctness(eager_correctness),
            "stable_key_relaunch": relaunch,
            "state_generation_keyed_recapture": recapture,
        },
        "timing": {
            "protocol": (
                "Rotating same-session full runs after reset/prefill; stable relaunch includes one "
                "capture/instantiate and final destroy per window, while recapture includes those costs per token."
            ),
            "warmups": {
                "eager": eager_warmups,
                "stable_key_relaunch": relaunch_warmups,
                "state_generation_keyed_recapture": recapture_warmups,
            },
            "eager": {"summary": eager_summary, "runs": eager_runs},
            "state_generation_keyed_recapture": {
                "summary": recapture_summary,
                "runs": recapture_runs,
            },
            "stable_key_relaunch": {
                "eligible": bool(relaunch["passed"]),
                "capture_included_in_summary": True,
                "summary": relaunch_summary,
                "runs": relaunch_runs,
            },
        },
        "provenance": provenance,
    }
    if args.out is not None:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
        print(f"[sol-g5] wrote {args.out}", flush=True)
    print(
        f"[sol-g5] {artifact['status']}: {classification['decision']}; "
        f"relaunch_passed={relaunch['passed']} first_failure={relaunch['first_failing_launch']}; "
        f"eager={eager_summary['median_tok_s']:.3f} tok/s "
        f"relaunch={None if relaunch_summary is None else round(relaunch_summary['median_tok_s'], 3)} tok/s "
        f"recapture={recapture_summary['median_tok_s']:.3f} tok/s",
        flush=True,
    )
    return 0 if evidence_valid else 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--backend", choices=("hip_gfx1100", "hip_gfx1151"), default="hip_gfx1151")
    parser.add_argument("--quant", default="gguf_q4_k_m")
    parser.add_argument("--prompt-token-id", type=int, default=9707)
    parser.add_argument("--expected-token-id", type=int, default=9707)
    parser.add_argument("--prompt-length", type=int, default=512)
    parser.add_argument("--correctness-steps", type=int, default=16)
    parser.add_argument("--timing-steps", type=int, default=16)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=4)
    parser.add_argument("--max-sequence-length", type=int, default=0)
    parser.add_argument("--compiler-version-file", type=Path, default=Path("/tmp/hipengine-hipcc-version.txt"))
    parser.add_argument("--require-cached", action="store_true")
    parser.add_argument("--out", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    for name in ("prompt_length", "correctness_steps", "timing_steps", "repetitions"):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if int(args.correctness_steps) < 3:
        raise ValueError("--correctness-steps must be at least 3")
    if int(args.warmups) < 0:
        raise ValueError("--warmups must be non-negative")
    if int(args.max_sequence_length) < 0:
        raise ValueError("--max-sequence-length must be non-negative")
    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
