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
from hipengine.core.memory import host_array_ptr, memory_stats
from hipengine.loading.gguf import GGUFReader
from hipengine.loading.laguna_gguf import FULL_ATTENTION
from hipengine.runtime.laguna_gguf_runner import LagunaGGUFResidentSession
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
    parser.add_argument("--repacked-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--model-sha256", default=DEFAULT_MODEL_SHA256)
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
            "all logits/hidden/KV/cursor fields exact, deterministic repeats, exact lifecycle, "
            f"and aggregate plus every-length wall improvement versus M{baseline_mode}"
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
    return LagunaGGUFResidentSession(
        resident_weights=owner.weights,
        context_length=args.context_length,
        backend=args.backend,
        runtime=owner.runtime,
        compiler_version=_compiler_version(args.compiler_version_file),
        require_cached_build=args.require_cached_build,
        prefill_chunk_size=matrix_rows,
        prefill_attention_chunk_size=args.attention_rows,
    )


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
    session = _session(owner, args, matrix_rows=matrix_rows)
    try:
        started = time.perf_counter()
        result = session.prefill(token_ids[:length], use_bulk=True)
        session.runtime.device_synchronize()
        elapsed = time.perf_counter() - started
        logits = _device_values(session.runtime, result.logits, np.float32)
        row = {
            "matrix_rows": matrix_rows,
            "attention_rows": session.prefill_attention_chunk_size,
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
    if lengths != LENGTHS:
        raise ValueError(f"matrix screen requires lengths={LENGTHS}")
    if len(matrix_rows) < 2 or tuple(sorted(matrix_rows)) != matrix_rows:
        raise ValueError("matrix screen rows must be at least two ascending capacities")
    if matrix_rows[-1] > 2_048:
        raise ValueError("matrix screen rows cannot exceed 2048")
    if args.attention_rows != ATTENTION_ROWS:
        raise ValueError(f"matrix screen requires attention rows {ATTENTION_ROWS}")
    if args.context_length < max(lengths):
        raise ValueError("largest matrix-screen length exceeds admitted context")
    if args.repetitions < 2:
        raise ValueError("matrix screen requires at least two repetitions")
    if args.warmup_rows <= 0 or args.warmup_rows > ATTENTION_ROWS:
        raise ValueError("warmup rows must fit one attention tile")
    if not args.model.is_file() or not args.model_sha256:
        raise ValueError("matrix screen requires the pinned model and SHA-256")
    repo = _repo_state()
    if not repo["tracked_clean"]:
        raise RuntimeError("retained Laguna matrix screen requires a clean tracked worktree")

    provenance = collect_artifact_provenance(
        repo_root=ROOT,
        configured_backend=args.backend,
        resolved_backend=args.backend,
        target_arch=args.backend.removeprefix("hip_"),
        model_path=args.model,
        quant="gguf_q4_k_m",
        kv_dtype="bf16",
        command=(str(Path(sys.executable).resolve()), *sys.argv),
        build_profile="laguna_prefill_matrix_chunk_screen",
        timing_protocol=(
            "same_load_m"
            + "_m".join(str(value) for value in matrix_rows)
            + "_attention128_512_1024_4096"
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
            repacked_cache=args.repacked_cache,
            model_sha256=args.model_sha256,
            prefill_chunk_size=matrix_rows[0],
            prefill_attention_chunk_size=ATTENTION_ROWS,
        )
        load_seconds = time.perf_counter() - load_started
        for mode in matrix_rows:
            warmup = _session(owner, args, matrix_rows=mode)
            try:
                warmup.prefill(token_stream[: args.warmup_rows], use_bulk=True)
                runtime.device_synchronize()
            finally:
                warmup.close()
        for repetition in range(args.repetitions):
            for length_index, length in enumerate(LENGTHS):
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
            "quant": "Q4_K_M mixed GGUF v3",
            "repacked_cache": str(args.repacked_cache.resolve()),
            "repacked_cache_manifest_sha256": (
                _sha256_bytes(manifest_path.read_bytes()) if manifest_path.is_file() else None
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
            "Canonical 68-122-token category prompts do not cross any screened matrix capacity.",
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
