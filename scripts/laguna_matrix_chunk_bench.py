#!/usr/bin/env python3
"""Screen Laguna matrix-chunk capacities with fixed 128-row attention tiles."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np

from hipengine.benchmark.provenance import collect_artifact_provenance
from hipengine.core.hip import HipMemcpyKind, get_hip_runtime
from hipengine.core.memory import free, host_array_ptr, malloc, memory_stats
from hipengine.loading.gguf import GGUFReader
from hipengine.loading.laguna_gguf import FULL_ATTENTION
import hipengine.runtime.laguna_gguf_runner as runner_module
from hipengine.runtime.laguna_gguf_runner import (
    LagunaGGUFResidentSession,
    LagunaHiddenCaptureTargets,
)
from hipengine.tokenization.gguf import LagunaGGUFTokenizer
from scripts.laguna_prefill_profile import _profile_token_stream
from scripts.laguna_target_ar_bench import (
    DEFAULT_CACHE,
    DEFAULT_MODEL,
    DEFAULT_MODEL_SHA256,
    DEFAULT_PROMPTS,
    _compiler_version,
    _load_prompts,
    _progress,
    _repo_state,
    _sha256_bytes,
    _sha256_json,
)

ROOT = Path(__file__).resolve().parents[1]
MATRIX_ROWS = (128, 256, 512)
ATTENTION_ROWS = 128
LENGTHS = (512, 1024, 4096)
ALL_HIDDEN_DEPTHS = tuple(range(1, 49))
_EXACT_FIELDS = (
    "next_token_id",
    "next_token_logit_hex",
    "logits_sha256",
    "final_hidden_sha256",
    "post_layer_hidden_sha256",
    "kv_sha256",
    "final_position",
)
DEFAULT_OUTPUT = Path("/tmp/laguna-matrix-chunk-screen.raw.json")


def _parse_ints(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item) for item in value.split(",") if item.strip())
    if not parsed or any(item <= 0 for item in parsed) or len(set(parsed)) != len(parsed):
        raise argparse.ArgumentTypeError("values must be positive distinct integers")
    return parsed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--backend", default="hip_gfx1151")
    parser.add_argument("--context-length", type=int, default=max(LENGTHS))
    parser.add_argument("--lengths", type=_parse_ints, default=LENGTHS)
    parser.add_argument("--matrix-rows", type=_parse_ints, default=MATRIX_ROWS)
    parser.add_argument("--attention-rows", type=int, default=ATTENTION_ROWS)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--warmup-rows", type=int, default=ATTENTION_ROWS)
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument(
        "--grouped-exact-iq",
        action="store_true",
        help="run exact expert-major IQ down reuse for every matrix capacity",
    )
    parser.add_argument("--repacked-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--model-sha256", default=DEFAULT_MODEL_SHA256)
    parser.add_argument("--quant-label", default="Q4_K_M mixed GGUF v3")
    parser.add_argument(
        "--direct-gguf",
        action="store_true",
        help="load the source GGUF directly instead of the repacked cache",
    )
    parser.add_argument("--safety-reserve-gib", type=float, default=4.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _mode_order(
    length_index: int,
    repetition: int,
    *,
    matrix_rows: Sequence[int] = MATRIX_ROWS,
) -> tuple[int, ...]:
    modes = tuple(int(value) for value in matrix_rows)
    if not modes:
        raise ValueError("matrix policy order requires at least one mode")
    shift = (int(length_index) + int(repetition)) % len(modes)
    return modes[shift:] + modes[:shift]


def _validate_protocol(
    *,
    lengths: Sequence[int],
    matrix_rows: Sequence[int],
    attention_rows: int,
    context_length: int,
    repetitions: int,
    warmup_rows: int,
) -> None:
    profiled_lengths = tuple(int(value) for value in lengths)
    modes = tuple(int(value) for value in matrix_rows)
    if not profiled_lengths or tuple(sorted(profiled_lengths)) != profiled_lengths:
        raise ValueError("matrix screen lengths must be positive ascending values")
    if len(set(profiled_lengths)) != len(profiled_lengths) or profiled_lengths[0] <= 0:
        raise ValueError("matrix screen lengths must be positive distinct values")
    if len(modes) < 2 or tuple(sorted(modes)) != modes:
        raise ValueError("matrix screen rows must be at least two ascending capacities")
    if modes[0] != 128:
        raise ValueError("matrix screen requires M128 as the exact control")
    if modes[-1] > 2_048:
        raise ValueError("matrix screen rows cannot exceed 2048")
    if profiled_lengths[0] < modes[-1]:
        raise ValueError("every matrix capacity must be exercised by the shortest length")
    if int(attention_rows) != ATTENTION_ROWS:
        raise ValueError(f"matrix screen requires attention rows {ATTENTION_ROWS}")
    if int(context_length) < max(*profiled_lengths, *modes):
        raise ValueError("largest matrix-screen shape exceeds admitted context")
    if int(repetitions) < 2:
        raise ValueError("matrix screen requires at least two repetitions")
    if int(warmup_rows) <= 0 or int(warmup_rows) > ATTENTION_ROWS:
        raise ValueError("warmup rows must fit one attention tile")


def _routing_occupancy_summary(
    selected_experts: Mapping[int, Sequence[int]],
    *,
    rows: int,
    top_k: int,
    expert_count: int,
) -> dict[str, Any]:
    parsed_rows = int(rows)
    parsed_top_k = int(top_k)
    parsed_experts = int(expert_count)
    if parsed_rows <= 0 or parsed_top_k <= 0 or parsed_experts <= 0:
        raise ValueError("routing occupancy dimensions must be positive")
    route_slots = parsed_rows * parsed_top_k
    rowbatches = (2, 4, 8, 16, 32)
    layers: dict[str, dict[str, Any]] = {}
    for layer_id, raw_selected in sorted(selected_experts.items()):
        selected = np.asarray(tuple(int(value) for value in raw_selected), dtype=np.int64)
        if selected.size != route_slots:
            raise ValueError(
                f"routing layer {layer_id} has {selected.size} lanes; expected {route_slots}"
            )
        if np.any(selected < 0) or np.any(selected >= parsed_experts):
            raise ValueError(f"routing layer {layer_id} contains an invalid expert ID")
        counts = np.bincount(selected, minlength=parsed_experts)
        active = counts[counts > 0]
        full = {
            str(rowbatch): int(np.sum((counts // rowbatch) * rowbatch))
            for rowbatch in rowbatches
        }
        tails = {
            str(rowbatch): int(np.sum(counts % rowbatch))
            for rowbatch in rowbatches
        }
        layers[str(int(layer_id))] = {
            "route_slots": route_slots,
            "active_experts": int(active.size),
            "mean_routes_per_all_expert": route_slots / parsed_experts,
            "mean_routes_per_active_expert": float(np.mean(active)),
            "median_routes_per_active_expert": float(np.median(active)),
            "max_routes_per_expert": int(np.max(active)),
            "experts_at_least_rows": {
                str(rowbatch): int(np.sum(counts >= rowbatch))
                for rowbatch in rowbatches
            },
            "full_route_slots_by_rowbatch": full,
            "tail_route_slots_by_rowbatch": tails,
            "full_route_utilization_by_rowbatch": {
                key: value / route_slots for key, value in full.items()
            },
        }
    if not layers:
        raise ValueError("routing occupancy requires at least one sparse layer")
    return {
        "rows": parsed_rows,
        "top_k": parsed_top_k,
        "expert_count": parsed_experts,
        "layers": layers,
        "aggregate": {
            "layer_count": len(layers),
            "route_slots_per_layer": route_slots,
            "mean_active_experts": statistics.mean(
                int(layer["active_experts"]) for layer in layers.values()
            ),
            "mean_routes_per_all_expert": route_slots / parsed_experts,
            "mean_routes_per_active_expert": statistics.mean(
                float(layer["mean_routes_per_active_expert"])
                for layer in layers.values()
            ),
            "max_routes_per_expert": max(
                int(layer["max_routes_per_expert"]) for layer in layers.values()
            ),
            "full_route_utilization_by_rowbatch": {
                str(rowbatch): statistics.mean(
                    float(layer["full_route_utilization_by_rowbatch"][str(rowbatch)])
                    for layer in layers.values()
                )
                for rowbatch in rowbatches
            },
            "tail_route_slots_by_rowbatch": {
                str(rowbatch): sum(
                    int(layer["tail_route_slots_by_rowbatch"][str(rowbatch)])
                    for layer in layers.values()
                )
                for rowbatch in rowbatches
            },
        },
    }


def _correctness(
    rows: Sequence[Mapping[str, Any]],
    *,
    matrix_rows: Sequence[int] = MATRIX_ROWS,
    lengths: Sequence[int] = LENGTHS,
) -> dict[str, Any]:
    modes_expected = tuple(int(value) for value in matrix_rows)
    lengths_expected = tuple(int(value) for value in lengths)
    grouped: dict[tuple[int, int], dict[int, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        grouped[(int(row["length"]), int(row["repetition"]))][
            int(row["matrix_rows"])
        ] = row
    comparisons = []
    per_mode = {str(mode): True for mode in modes_expected}
    for (length, repetition), modes in sorted(grouped.items()):
        if set(modes) != set(modes_expected):
            raise ValueError(f"missing matrix policy for length={length} rep={repetition}")
        baseline = modes[modes_expected[0]]
        checks = {}
        for mode in modes_expected:
            exact = all(modes[mode][field] == baseline[field] for field in _EXACT_FIELDS)
            checks[str(mode)] = exact
            per_mode[str(mode)] = bool(per_mode[str(mode)] and exact)
        comparisons.append(
            {
                "length": length,
                "repetition": repetition,
                "exact_by_matrix_rows": checks,
                "pass": all(checks.values()),
            }
        )
    deterministic = True
    for mode in modes_expected:
        for length in lengths_expected:
            selected = [
                row
                for row in rows
                if int(row["matrix_rows"]) == mode and int(row["length"]) == length
            ]
            for field in _EXACT_FIELDS:
                deterministic = deterministic and len({row[field] for row in selected}) == 1
    return {
        "pass": bool(all(item["pass"] for item in comparisons) and deterministic),
        "exact_by_matrix_rows": per_mode,
        "same_mode_repeat_deterministic": bool(deterministic),
        "comparisons": comparisons,
    }


def _aggregate(
    rows: Sequence[Mapping[str, Any]],
    *,
    matrix_rows: Sequence[int] = MATRIX_ROWS,
    lengths: Sequence[int] = LENGTHS,
) -> dict[str, Any]:
    modes = tuple(int(value) for value in matrix_rows)
    profiled_lengths = tuple(int(value) for value in lengths)
    result: dict[str, Any] = {}
    for mode in modes:
        selected = [row for row in rows if int(row["matrix_rows"]) == mode]
        lengths = {}
        median_sum = 0.0
        for length in profiled_lengths:
            samples = [
                float(row["prefill_seconds"])
                for row in selected
                if int(row["length"]) == length
            ]
            if not samples or any(not math.isfinite(value) or value <= 0.0 for value in samples):
                raise ValueError("every matrix-chunk timing sample must be finite and positive")
            median = statistics.median(samples)
            median_sum += median
            lengths[str(length)] = {
                "samples_seconds": samples,
                "median_seconds": median,
                "median_tok_s": length / median,
            }
        result[str(mode)] = {
            "matrix_rows": mode,
            "attention_rows": ATTENTION_ROWS,
            "median_sum_seconds": median_sum,
            "weighted_tok_s": sum(profiled_lengths) / median_sum,
            "lengths": lengths,
        }
    baseline_mode = modes[0]
    speedup_key = f"speedup_vs_{baseline_mode}"
    baseline = result[str(baseline_mode)]
    for mode in modes:
        current = result[str(mode)]
        current[speedup_key] = (
            baseline["median_sum_seconds"] / current["median_sum_seconds"]
        )
        for length in profiled_lengths:
            current["lengths"][str(length)][speedup_key] = (
                baseline["lengths"][str(length)]["median_seconds"]
                / current["lengths"][str(length)]["median_seconds"]
            )
    return result


def _decision(
    aggregate: Mapping[str, Any],
    correctness: Mapping[str, Any],
    *,
    recovered: bool,
    full_state_pass: bool = True,
    routing_prefix_pass: bool = True,
    matrix_rows: Sequence[int] = MATRIX_ROWS,
    lengths: Sequence[int] = LENGTHS,
) -> dict[str, Any]:
    modes = tuple(int(value) for value in matrix_rows)
    profiled_lengths = tuple(int(value) for value in lengths)
    baseline_mode = modes[0]
    speedup_key = f"speedup_vs_{baseline_mode}"
    eligible = []
    for mode in modes[1:]:
        summary = aggregate[str(mode)]
        exact = bool(correctness["exact_by_matrix_rows"][str(mode)])
        every_length_positive = all(
            float(summary["lengths"][str(length)][speedup_key]) > 1.0
            for length in profiled_lengths
        )
        if exact and every_length_positive and float(summary[speedup_key]) > 1.0:
            eligible.append(mode)
    selected = max(
        eligible,
        key=lambda mode: float(aggregate[str(mode)][speedup_key]),
        default=baseline_mode,
    )
    failed = []
    if not correctness["pass"]:
        failed.append("matrix_policy_outputs_or_state_not_exact")
    if not full_state_pass:
        failed.append("all_hidden_boundaries_not_exact")
    if not routing_prefix_pass:
        failed.append("shared_prefix_routing_not_exact")
    if not recovered:
        failed.append("tracked_lifecycle_not_recovered")
    if selected == baseline_mode:
        failed.append("no_larger_policy_improves_every_length")
    return {
        "pass": not failed,
        "selected_matrix_rows": selected,
        "attention_rows": ATTENTION_ROWS,
        "eligible_matrix_rows": eligible,
        "failed_checks": failed,
        "policy": (
            "all logits/48 hidden boundaries/KV/cursor fields and shared-prefix routing "
            "exact, deterministic repeats, exact lifecycle, and aggregate plus every-length "
            f"wall improvement versus M{baseline_mode}"
        ),
    }


def _normalized_log_probs(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    shifted = values - float(np.max(values))
    return shifted - float(np.log(np.exp(shifted).sum()))


def _relative_quality(
    rows: Sequence[Mapping[str, Any]],
    *,
    matrix_rows: Sequence[int],
    lengths: Sequence[int],
) -> dict[str, Any]:
    modes = tuple(int(value) for value in matrix_rows)
    profiled_lengths = tuple(int(value) for value in lengths)
    baseline_mode = modes[0]
    grouped: dict[tuple[int, int], dict[int, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        grouped[(int(row["length"]), int(row["repetition"]))][
            int(row["matrix_rows"])
        ] = row
    records: dict[int, list[dict[str, Any]]] = {mode: [] for mode in modes}
    for (length, repetition), paired in sorted(grouped.items()):
        if length not in profiled_lengths or set(paired) != set(modes):
            raise ValueError(
                f"missing quality matrix policy for length={length} rep={repetition}"
            )
        reference = np.asarray(paired[baseline_mode]["_logits"], dtype=np.float32)
        reference_log_probs = _normalized_log_probs(reference)
        reference_probabilities = np.exp(reference_log_probs)
        reference_top1 = int(np.argmax(reference))
        for mode in modes:
            candidate = np.asarray(paired[mode]["_logits"], dtype=np.float32)
            candidate_log_probs = _normalized_log_probs(candidate)
            kl = float(
                np.sum(
                    reference_probabilities
                    * (reference_log_probs - candidate_log_probs)
                )
            )
            candidate_top1 = int(np.argmax(candidate))
            records[mode].append(
                {
                    "length": length,
                    "repetition": repetition,
                    "kl_divergence": kl,
                    "reference_top1": reference_top1,
                    "candidate_top1": candidate_top1,
                    "top1_agreement": candidate_top1 == reference_top1,
                    "finite": bool(
                        np.isfinite(reference).all()
                        and np.isfinite(candidate).all()
                        and math.isfinite(kl)
                    ),
                }
            )
    by_mode = {}
    for mode in modes:
        mode_records = records[mode]
        max_kl = max(float(record["kl_divergence"]) for record in mode_records)
        top1_agreement = sum(
            bool(record["top1_agreement"]) for record in mode_records
        ) / len(mode_records)
        finite = all(bool(record["finite"]) for record in mode_records)
        by_mode[str(mode)] = {
            "pass": bool(finite and max_kl <= 0.05 and top1_agreement >= 0.9),
            "max_kl_divergence": max_kl,
            "top1_agreement": top1_agreement,
            "finite": finite,
            "records": mode_records,
        }
    return {
        "pass": all(bool(value["pass"]) for value in by_mode.values()),
        "baseline_matrix_rows": baseline_mode,
        "by_matrix_rows": by_mode,
        "thresholds": {
            "max_kl_divergence": 0.05,
            "minimum_top1_agreement": 0.9,
        },
    }


def _device_values(runtime, buffer, dtype: np.dtype) -> np.ndarray:
    values = np.empty(buffer.nbytes // np.dtype(dtype).itemsize, dtype=dtype)
    runtime.memcpy(
        host_array_ptr(values),
        buffer.ptr,
        values.nbytes,
        HipMemcpyKind.DEVICE_TO_HOST,
    )
    return values


def _device_digest(runtime, buffer, dtype: np.dtype) -> str:
    return hashlib.sha256(_device_values(runtime, buffer, dtype).tobytes()).hexdigest()


def _kv_digest(session: LagunaGGUFResidentSession) -> str:
    assert session.kv_cache is not None
    runtime = session.runtime
    digest = hashlib.sha256()
    width = session.config.head_count_kv * session.config.key_length
    for state in session.kv_cache.layers:
        offsets = np.empty(state.spans.base_offsets.numel, dtype=np.int32)
        live_counts = np.empty(state.spans.live_counts.numel, dtype=np.int64)
        positions = np.empty(state.capacity, dtype=np.int64)
        mask = np.empty(state.capacity, dtype=np.bool_)
        for tensor, destination in (
            (state.spans.base_offsets, offsets),
            (state.spans.live_counts, live_counts),
            (state.spans.token_positions, positions),
            (state.spans.evict_mask, mask),
        ):
            assert tensor is not None
            runtime.memcpy(
                host_array_ptr(destination),
                tensor.ptr,
                destination.nbytes,
                HipMemcpyKind.DEVICE_TO_HOST,
            )
        key_cache = np.empty((state.physical_capacity, width), dtype=np.uint16)
        value_cache = np.empty_like(key_cache)
        for buffer, destination in (
            (state.key_cache, key_cache),
            (state.value_cache, value_cache),
        ):
            runtime.memcpy(
                host_array_ptr(destination),
                buffer.ptr,
                destination.nbytes,
                HipMemcpyKind.DEVICE_TO_HOST,
            )
        digest.update(int(state.layer_id).to_bytes(4, "little", signed=False))
        digest.update(state.attention_type.encode("utf-8"))
        digest.update(offsets.tobytes())
        digest.update(live_counts.tobytes())
        digest.update(positions.tobytes())
        digest.update(mask.tobytes())
        for logical_slot in np.flatnonzero((~mask) & (positions >= 0)).tolist():
            if state.attention_type == FULL_ATTENTION:
                block_size = 256
                physical_slot = (
                    int(offsets[logical_slot // block_size]) * block_size
                    + logical_slot % block_size
                )
            else:
                physical_slot = int(offsets[logical_slot])
            digest.update(int(logical_slot).to_bytes(4, "little", signed=False))
            digest.update(int(positions[logical_slot]).to_bytes(8, "little", signed=True))
            digest.update(key_cache[physical_slot].tobytes())
            digest.update(value_cache[physical_slot].tobytes())
    return digest.hexdigest()


def _session(
    owner: LagunaGGUFResidentSession,
    args: argparse.Namespace,
    *,
    matrix_rows: int,
) -> LagunaGGUFResidentSession:
    assert owner.weights is not None
    session = LagunaGGUFResidentSession(
        resident_weights=owner.weights,
        context_length=args.context_length,
        backend=args.backend,
        runtime=owner.runtime,
        compiler_version=_compiler_version(args.compiler_version_file),
        require_cached_build=args.require_cached_build,
        prefill_chunk_size=matrix_rows,
        prefill_attention_chunk_size=args.attention_rows,
        prefill_global_attention_chunk_size=args.attention_rows,
    )
    if args.grouped_exact_iq:
        session.set_selected_gate_up_mode("grouped_exact")
        session.set_selected_down_mode("grouped_exact")
    return session


def _all_hidden_capture_targets(
    session: LagunaGGUFResidentSession,
) -> tuple[LagunaHiddenCaptureTargets, list[Any]]:
    depths = tuple(range(1, int(session.config.block_count) + 1))
    if depths != ALL_HIDDEN_DEPTHS:
        raise ValueError(
            f"matrix screen requires the 48-layer Laguna target; got {len(depths)} layers"
        )
    buffers: list[Any] = []
    try:
        for _ in depths:
            buffers.append(
                malloc(
                    session.config.hidden_size * np.dtype(np.uint16).itemsize,
                    runtime=session.runtime,
                )
            )
        previous_depths = runner_module.LAGUNA_DFLASH_CAPTURE_DEPTHS
        runner_module.LAGUNA_DFLASH_CAPTURE_DEPTHS = depths
        try:
            targets = LagunaHiddenCaptureTargets(
                hidden_size=session.config.hidden_size,
                buffers=dict(zip(depths, buffers, strict=True)),
                rows=1,
            )
        finally:
            runner_module.LAGUNA_DFLASH_CAPTURE_DEPTHS = previous_depths
        return targets, buffers
    except BaseException:
        for buffer in reversed(buffers):
            free(buffer, runtime=session.runtime)
        raise


def _full_state_snapshot(
    owner: LagunaGGUFResidentSession,
    token_ids: Sequence[int],
    args: argparse.Namespace,
    *,
    matrix_rows: int,
    length: int,
) -> dict[str, Any]:
    before = memory_stats()
    session = _session(owner, args, matrix_rows=matrix_rows)
    targets: LagunaHiddenCaptureTargets | None = None
    buffers: list[Any] = []
    try:
        targets, buffers = _all_hidden_capture_targets(session)
        started = time.perf_counter()
        result = session.prefill(
            token_ids[:length],
            capture_last=targets,
            use_bulk=True,
        )
        session.runtime.device_synchronize()
        elapsed = time.perf_counter() - started
        logits = _device_values(session.runtime, result.logits, np.float32)
        snapshot = {
            "matrix_rows": matrix_rows,
            "attention_rows": session.prefill_attention_chunk_size,
            "global_attention_rows": session.prefill_global_attention_chunk_size,
            "length": length,
            "seconds_diagnostic_with_captures": elapsed,
            "next_token_id": int(result.next_token_id),
            "next_token_logit_hex": float(result.next_token_logit).hex(),
            "logits_sha256": hashlib.sha256(logits.tobytes()).hexdigest(),
            "final_hidden_sha256": _device_digest(
                session.runtime, result.final_hidden, np.uint16
            ),
            "post_layer_hidden_sha256": _device_digest(
                session.runtime, result.post_layer_hidden, np.uint16
            ),
            "hidden_boundary_sha256": {
                str(depth): _device_digest(session.runtime, buffer, np.uint16)
                for depth, buffer in targets.buffers.items()
            },
            "kv_sha256": _kv_digest(session),
            "final_position": int(session.position),
        }
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=session.runtime)
        session.close()
    after = memory_stats()
    snapshot["tracked_returned_to_baseline"] = bool(
        after["current_allocated_bytes"] == before["current_allocated_bytes"]
        and after["active_allocations"] == before["active_allocations"]
    )
    return snapshot


def _full_state_gate(
    owner: LagunaGGUFResidentSession,
    token_ids: Sequence[int],
    args: argparse.Namespace,
    *,
    matrix_rows: Sequence[int],
) -> dict[str, Any]:
    modes = tuple(int(value) for value in matrix_rows)
    length = modes[-1]
    snapshots = {
        str(mode): _full_state_snapshot(
            owner,
            token_ids,
            args,
            matrix_rows=mode,
            length=length,
        )
        for mode in modes
    }
    baseline = snapshots[str(modes[0])]
    comparisons = {}
    for mode in modes:
        current = snapshots[str(mode)]
        checks = {
            field: current[field] == baseline[field]
            for field in _EXACT_FIELDS
        }
        boundary_checks = {
            depth: digest == baseline["hidden_boundary_sha256"][depth]
            for depth, digest in current["hidden_boundary_sha256"].items()
        }
        checks["all_48_hidden_boundaries"] = all(boundary_checks.values())
        checks["attention_rows_fixed"] = (
            current["attention_rows"] == ATTENTION_ROWS
            and current["global_attention_rows"] == ATTENTION_ROWS
        )
        checks["tracked_returned_to_baseline"] = bool(
            current["tracked_returned_to_baseline"]
        )
        comparisons[str(mode)] = {
            "pass": all(checks.values()),
            "checks": checks,
            "hidden_boundary_exact": boundary_checks,
        }
    return {
        "pass": all(value["pass"] for value in comparisons.values()),
        "length": length,
        "baseline_matrix_rows": modes[0],
        "snapshots": snapshots,
        "comparisons": comparisons,
    }


def _routing_probe(
    owner: LagunaGGUFResidentSession,
    token_ids: Sequence[int],
    args: argparse.Namespace,
    *,
    matrix_rows: Sequence[int],
) -> dict[str, Any]:
    modes = tuple(int(value) for value in matrix_rows)
    host_replays: dict[int, Any] = {}
    summaries: dict[str, Any] = {}
    all_recovered = True
    for mode in modes:
        before = memory_stats()
        session = _session(owner, args, matrix_rows=mode)
        try:
            started = time.perf_counter()
            replay = session.prefill_routing_replay(token_ids[:mode])
            session.runtime.device_synchronize()
            elapsed = time.perf_counter() - started
            host_replays[mode] = replay
            selected_payload = {
                str(layer_id): list(values)
                for layer_id, values in replay.selected_experts.items()
            }
            weight_payload = {
                str(layer_id): [float(value) for value in values]
                for layer_id, values in replay.routing_weights.items()
            }
            summaries[str(mode)] = {
                "matrix_rows": mode,
                "seconds_diagnostic_with_capture": elapsed,
                "next_token_id": int(replay.result.next_token_id),
                "final_position": int(session.position),
                "selected_experts_sha256": _sha256_json(selected_payload),
                "routing_weights_sha256": _sha256_json(weight_payload),
                "occupancy": _routing_occupancy_summary(
                    replay.selected_experts,
                    rows=replay.rows,
                    top_k=replay.top_k,
                    expert_count=replay.expert_count,
                ),
            }
        finally:
            session.close()
        after = memory_stats()
        recovered = bool(
            after["current_allocated_bytes"] == before["current_allocated_bytes"]
            and after["active_allocations"] == before["active_allocations"]
        )
        summaries[str(mode)]["tracked_returned_to_baseline"] = recovered
        all_recovered = bool(all_recovered and recovered)

    baseline_mode = modes[0]
    baseline = host_replays[baseline_mode]
    prefix_lanes = baseline.rows * baseline.top_k
    shared_prefix = {}
    for mode in modes:
        replay = host_replays[mode]
        ids_exact = all(
            tuple(replay.selected_experts[layer_id][:prefix_lanes])
            == tuple(baseline.selected_experts[layer_id])
            for layer_id in baseline.selected_experts
        )
        weights_exact = all(
            tuple(replay.routing_weights[layer_id][:prefix_lanes])
            == tuple(baseline.routing_weights[layer_id])
            for layer_id in baseline.routing_weights
        )
        shared_prefix[str(mode)] = {
            "selected_experts_exact": ids_exact,
            "routing_weights_exact": weights_exact,
            "pass": bool(ids_exact and weights_exact),
        }
    return {
        "pass": bool(
            all(value["pass"] for value in shared_prefix.values()) and all_recovered
        ),
        "baseline_matrix_rows": baseline_mode,
        "shared_prefix_lanes_per_layer": prefix_lanes,
        "shared_prefix": shared_prefix,
        "modes": summaries,
        "tracked_returned_to_baseline": all_recovered,
    }


def _run_one(
    owner: LagunaGGUFResidentSession,
    token_ids: Sequence[int],
    args: argparse.Namespace,
    *,
    matrix_rows: int,
    length: int,
    repetition: int,
) -> dict[str, Any]:
    before = memory_stats()
    gpu_free_before, gpu_total = owner.runtime.mem_get_info()
    session = _session(owner, args, matrix_rows=matrix_rows)
    after_allocate = memory_stats()
    gpu_free_after_allocate, gpu_total_after_allocate = owner.runtime.mem_get_info()
    try:
        if gpu_total_after_allocate != gpu_total:
            raise RuntimeError("HIP total memory changed during matrix session allocation")
        started = time.perf_counter()
        result = session.prefill(token_ids[:length], use_bulk=True)
        session.runtime.device_synchronize()
        elapsed = time.perf_counter() - started
        logits = _device_values(session.runtime, result.logits, np.float32)
        row = {
            "matrix_rows": matrix_rows,
            "attention_rows": session.prefill_attention_chunk_size,
            "global_attention_rows": session.prefill_global_attention_chunk_size,
            "raw_k_prefill_rowbatch": session.raw_k_prefill_rowbatch,
            "length": length,
            "repetition": repetition,
            "chunks": math.ceil(length / matrix_rows),
            "attention_slices_per_full_chunk": math.ceil(
                min(length, matrix_rows) / session.prefill_attention_chunk_size
            ),
            "prefill_seconds": elapsed,
            "prefill_tok_s": length / elapsed,
            "next_token_id": int(result.next_token_id),
            "next_token_logit_hex": float(result.next_token_logit).hex(),
            "logits_sha256": hashlib.sha256(logits.tobytes()).hexdigest(),
            "_logits": logits,
            "final_hidden_sha256": _device_digest(
                session.runtime, result.final_hidden, np.uint16
            ),
            "post_layer_hidden_sha256": _device_digest(
                session.runtime, result.post_layer_hidden, np.uint16
            ),
            "kv_sha256": _kv_digest(session),
            "final_position": int(session.position),
            "scratch": {
                "rows_nbytes": session.prefill_scratch_plan.rows_nbytes,
                "moe_nbytes": session.prefill_scratch_plan.moe_nbytes,
                "total_nbytes": session.prefill_scratch_plan.total_nbytes,
                "admission_nbytes": session.prefill_scratch_admission_nbytes,
            },
            "allocation_observation": {
                "tracked_delta_after_session_allocate": (
                    after_allocate["current_allocated_bytes"]
                    - before["current_allocated_bytes"]
                ),
                "gpu_free_delta_after_session_allocate": (
                    gpu_free_before - gpu_free_after_allocate
                ),
                "gpu_free_before": gpu_free_before,
                "gpu_free_after_session_allocate": gpu_free_after_allocate,
            },
        }
    finally:
        session.close()
    after = memory_stats()
    row["session_tracked_returned_to_baseline"] = bool(
        after["current_allocated_bytes"] == before["current_allocated_bytes"]
        and after["active_allocations"] == before["active_allocations"]
    )
    return row


def run(args: argparse.Namespace) -> dict[str, Any]:
    lengths = tuple(int(value) for value in args.lengths)
    matrix_rows = tuple(int(value) for value in args.matrix_rows)
    _validate_protocol(
        lengths=lengths,
        matrix_rows=matrix_rows,
        attention_rows=args.attention_rows,
        context_length=args.context_length,
        repetitions=args.repetitions,
        warmup_rows=args.warmup_rows,
    )
    if not args.model.is_file() or not args.model_sha256:
        raise ValueError("matrix screen requires the pinned model and SHA-256")
    if not math.isfinite(args.safety_reserve_gib) or args.safety_reserve_gib < 0.0:
        raise ValueError("matrix screen safety reserve must be finite and nonnegative")
    repo = _repo_state()
    if not repo["tracked_clean"] and not args.allow_dirty:
        raise RuntimeError("retained Laguna matrix screen requires a clean tracked worktree")

    provenance = collect_artifact_provenance(
        repo_root=ROOT,
        configured_backend=args.backend,
        resolved_backend=args.backend,
        target_arch=args.backend.removeprefix("hip_"),
        model_path=args.model,
        quant=args.quant_label,
        kv_dtype="bf16",
        command=(str(Path(sys.executable).resolve()), *sys.argv),
        build_profile="laguna_prefill_matrix_chunk_screen",
        timing_protocol=(
            "same_load_m"
            + "_m".join(str(value) for value in matrix_rows)
            + "_attention128_"
            + "_".join(str(value) for value in lengths)
        ),
        warmups=len(matrix_rows),
        repetitions=args.repetitions,
    )
    reader = GGUFReader(args.model)
    tokenizer = LagunaGGUFTokenizer.from_gguf_info(reader.info)
    prompts = _load_prompts(args.prompts, tokenizer)
    token_stream, token_source = _profile_token_stream(prompts, max(lengths))
    runtime = get_hip_runtime()
    tracked_before = memory_stats()
    gpu_free_before, gpu_total = runtime.mem_get_info()
    owner: LagunaGGUFResidentSession | None = None
    rows: list[dict[str, Any]] = []
    load_started = time.perf_counter()
    try:
        owner = LagunaGGUFResidentSession(
            args.model,
            context_length=args.context_length,
            backend=args.backend,
            runtime=runtime,
            compiler_version=_compiler_version(args.compiler_version_file),
            require_cached_build=args.require_cached_build,
            progress=_progress,
            repacked_cache=None if args.direct_gguf else args.repacked_cache,
            model_sha256=args.model_sha256,
            prefill_chunk_size=matrix_rows[-1],
            prefill_attention_chunk_size=ATTENTION_ROWS,
            prefill_global_attention_chunk_size=ATTENTION_ROWS,
            safety_reserve_nbytes=int(args.safety_reserve_gib * (1 << 30)),
        )
        load_seconds = time.perf_counter() - load_started
        full_state = _full_state_gate(
            owner,
            token_stream,
            args,
            matrix_rows=matrix_rows,
        )
        routing = _routing_probe(
            owner,
            token_stream,
            args,
            matrix_rows=matrix_rows,
        )
        for mode in matrix_rows:
            warmup = _session(owner, args, matrix_rows=mode)
            try:
                warmup.prefill(token_stream[: args.warmup_rows], use_bulk=True)
                runtime.device_synchronize()
            finally:
                warmup.close()
        for repetition in range(args.repetitions):
            for length_index, length in enumerate(lengths):
                for mode in _mode_order(
                    length_index,
                    repetition,
                    matrix_rows=matrix_rows,
                ):
                    row = _run_one(
                        owner,
                        token_stream,
                        args,
                        matrix_rows=mode,
                        length=length,
                        repetition=repetition,
                    )
                    rows.append(row)
                    print(
                        f"rep={repetition} length={length} matrix={mode} "
                        f"prefill={row['prefill_tok_s']:.3f} tok/s "
                        f"next={row['next_token_id']}",
                        file=sys.stderr,
                        flush=True,
                    )
        owner_resident_nbytes = owner.resident_nbytes
    finally:
        if owner is not None:
            owner.close()
    tracked_after = memory_stats()
    gpu_free_after, gpu_total_after = runtime.mem_get_info()
    if gpu_total_after != gpu_total:
        raise RuntimeError("HIP total memory changed during Laguna matrix screen")

    correctness = _correctness(
        rows,
        matrix_rows=matrix_rows,
        lengths=lengths,
    )
    aggregate = _aggregate(
        rows,
        matrix_rows=matrix_rows,
        lengths=lengths,
    )
    relative_quality = _relative_quality(
        rows,
        matrix_rows=matrix_rows,
        lengths=lengths,
    )
    recovered = bool(
        tracked_after["current_allocated_bytes"] == tracked_before["current_allocated_bytes"]
        and tracked_after["active_allocations"] == tracked_before["active_allocations"]
        and all(bool(row["session_tracked_returned_to_baseline"]) for row in rows)
    )
    decision = _decision(
        aggregate,
        correctness,
        recovered=recovered,
        full_state_pass=bool(full_state["pass"]),
        routing_prefix_pass=bool(routing["pass"]),
        matrix_rows=matrix_rows,
        lengths=lengths,
    )
    for row in rows:
        row.pop("_logits")
    manifest_path = args.repacked_cache / "manifest.json"
    return {
        "schema": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "kind": "hipengine_laguna_ar_o3_matrix_chunk_screen",
        "status": "retained_candidate" if decision["pass"] else "measured_rejected",
        "pass": bool(decision["pass"]),
        "performance_claim": bool(decision["pass"]),
        "scope": (
            "Laguna prefill-only "
            + "/".join(f"M{value}" for value in matrix_rows)
            + " matrix chunks with fixed attention128"
        ),
        "provenance": provenance,
        "repo": repo,
        "model": {
            "path": str(args.model.resolve()),
            "sha256": args.model_sha256,
            "quant": args.quant_label,
            "direct_gguf": bool(args.direct_gguf),
            "repacked_cache": (
                None if args.direct_gguf else str(args.repacked_cache.resolve())
            ),
            "repacked_cache_manifest_sha256": (
                _sha256_bytes(manifest_path.read_bytes())
                if not args.direct_gguf and manifest_path.is_file()
                else None
            ),
        },
        "platform": {
            "backend": args.backend,
            "target_arch": args.backend.removeprefix("hip_"),
            "device_name": provenance["device_name"],
            "machine": platform.machine(),
            "hip_total_bytes": gpu_total,
        },
        "protocol": {
            "lengths": list(lengths),
            "matrix_rows": list(matrix_rows),
            "attention_rows": ATTENTION_ROWS,
            "global_attention_rows": ATTENTION_ROWS,
            "grouped_exact_iq": bool(args.grouped_exact_iq),
            "repetitions": args.repetitions,
            "warmup_rows_per_mode": args.warmup_rows,
            "mode_order": "rotating Latin order by length and repetition",
            "timing_scope": (
                "borrowed session construction excluded; synchronized prefill and final "
                "projection included; hashing excluded"
            ),
            "prompt_suite": str(args.prompts.resolve()),
            "prompt_suite_sha256": _sha256_bytes(args.prompts.read_bytes()),
            "token_stream_sha256": _sha256_json(token_stream),
            "token_source": token_source,
            "direct_gguf": bool(args.direct_gguf),
            "quant_label": args.quant_label,
            "safety_reserve_gib": args.safety_reserve_gib,
            "all_hidden_boundary_gate_rows": full_state["length"],
            "routing_occupancy_rows": list(matrix_rows),
            "boundary_gate": (
                "tests/test_laguna_kv_attention.py::"
                "test_laguna_swa_resident_attention_slices_match_chunks_across_ring_wrap"
            ),
        },
        "load": {
            "seconds_excluded": load_seconds,
            "owner_resident_nbytes": owner_resident_nbytes,
        },
        "rows": rows,
        "aggregate": aggregate,
        "correctness": {
            **correctness,
            "tracked_returned_to_baseline": recovered,
        },
        "relative_quality": relative_quality,
        "full_state": full_state,
        "routing": routing,
        "decision": decision,
        "memory": {
            "tracked_before": tracked_before,
            "tracked_after": tracked_after,
            "gpu_free_before": gpu_free_before,
            "gpu_free_after": gpu_free_after,
            "hip_total_bytes": gpu_total,
        },
        "command": [str(Path(sys.executable).resolve()), *sys.argv],
        "limitations": [
            "Shape-policy screen uses one deterministic canonical-prompt-derived long token stream.",
            "The canonical prompt suite supplies a deterministic token stream extended to each requested length.",
            "No package default changes until the clean screen and required rollup are retained.",
        ],
    }


def main() -> int:
    args = _parse_args()
    result = run(args)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")
    sys.stdout.write(payload)
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
