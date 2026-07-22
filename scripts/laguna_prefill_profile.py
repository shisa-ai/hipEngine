#!/usr/bin/env python3
"""Profile Laguna prefill-only row shapes and replay real MoE routing occupancy."""

from __future__ import annotations

import argparse
import base64
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import struct
import sys
import time
from typing import Any, Mapping, Sequence

from hipengine.benchmark.provenance import collect_artifact_provenance
from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import memory_stats
from hipengine.loading.gguf import GGUFReader
from hipengine.runtime.laguna_gguf_runner import LagunaGGUFResidentSession
from hipengine.tokenization.gguf import LagunaGGUFTokenizer
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

DEFAULT_ROWS = (16, 32, 55, 64, 122, 128)
DEFAULT_OUTPUT = Path(
    "benchmarks/results/2026-07-23-gfx1151-laguna-prefill-lpf0-profile.json"
)


def _parse_rows(value: str) -> tuple[int, ...]:
    rows = tuple(sorted({int(item) for item in value.split(",") if item.strip()}))
    if not rows or any(item <= 1 for item in rows):
        raise argparse.ArgumentTypeError("prefill rows must be distinct integers greater than one")
    return rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--backend", default="hip_gfx1151")
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument("--rows", type=_parse_rows, default=DEFAULT_ROWS)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--routing-tile-rows", type=int, default=16)
    parser.add_argument("--skip-routing-replay", action="store_true")
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--repacked-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--model-sha256", default=DEFAULT_MODEL_SHA256)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _profile_token_stream(
    prompts: Sequence[Mapping[str, Any]],
    requested_tokens: int,
) -> tuple[tuple[int, ...], dict[str, Any]]:
    """Build one deterministic shape fixture from the longest canonical prompt."""

    count = int(requested_tokens)
    if count <= 0 or not prompts:
        raise ValueError("profile token construction requires canonical prompts and positive rows")
    source = max(
        prompts,
        key=lambda prompt: (len(tuple(prompt.get("token_ids", ()))), str(prompt.get("id", ""))),
    )
    base = tuple(int(token) for token in source.get("token_ids", ()))
    if len(base) < 2:
        raise ValueError("profile token extension source must contain at least two tokens")
    tokens = list(base[:count])
    extension = "none"
    if len(tokens) < count:
        extension = "repeat_without_leading_bos"
        continuation = base[1:]
        while len(tokens) < count:
            tokens.extend(continuation[: count - len(tokens)])
    return tuple(tokens), {
        "prompt_id": str(source["id"]),
        "category": str(source["category"]),
        "source_tokens": len(base),
        "requested_tokens": count,
        "extension": extension,
    }


def _summarize_timing_samples(samples: Sequence[float], *, rows: int) -> dict[str, Any]:
    values = [float(value) for value in samples]
    if not values or any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("every Laguna prefill timing sample must be finite and positive")
    median = statistics.median(values)
    return {
        "rows": int(rows),
        "samples_seconds": values,
        "median_seconds": median,
        "median_tok_s": int(rows) / median,
        "min_seconds": min(values),
        "max_seconds": max(values),
    }


def _summarize_routing_replay(
    selected_by_layer: Mapping[int, Sequence[int]],
    *,
    rows: int,
    top_k: int,
    expert_count: int,
    tile_rows: int = 16,
) -> dict[str, Any]:
    parsed_rows = int(rows)
    parsed_top_k = int(top_k)
    parsed_experts = int(expert_count)
    parsed_tile = int(tile_rows)
    if parsed_rows <= 0 or parsed_top_k <= 0 or parsed_experts <= 0 or parsed_tile <= 0:
        raise ValueError("routing replay dimensions must be positive")
    expected_lanes = parsed_rows * parsed_top_k
    layers: dict[str, Any] = {}
    histogram: Counter[int] = Counter()
    flattened: list[int] = []
    padded_lanes = 0
    active_groups = 0
    maximum = 0
    for raw_layer_id, raw_selected in sorted(selected_by_layer.items()):
        layer_id = int(raw_layer_id)
        selected = tuple(int(value) for value in raw_selected)
        if len(selected) != expected_lanes:
            raise ValueError(
                f"layer {layer_id} selected lanes must equal rows*top_k={expected_lanes}"
            )
        if any(value < 0 or value >= parsed_experts for value in selected):
            raise ValueError(f"layer {layer_id} expert IDs must be within [0, {parsed_experts})")
        counts = Counter(selected)
        ordered = sorted((expert_id, count) for expert_id, count in counts.items())
        for _, count in ordered:
            histogram[count] += 1
            active_groups += 1
            padded_lanes += ((count + parsed_tile - 1) // parsed_tile) * parsed_tile
            maximum = max(maximum, count)
        dense_counts = [counts.get(expert_id, 0) for expert_id in range(parsed_experts)]
        packed_counts = struct.pack(f"<{parsed_experts}H", *dense_counts)
        layers[str(layer_id)] = {
            "active_experts": len(ordered),
            "max_expert_lanes": max(counts.values()),
            "per_expert_counts_encoding": "uint16_le_dense_expert_id_order_base64",
            "per_expert_counts_u16_le_base64": base64.b64encode(packed_counts).decode("ascii"),
        }
        flattened.extend(selected)
    actual_lanes = len(flattened)
    if not flattened:
        raise ValueError("routing replay must contain at least one sparse layer")
    payload = json.dumps(flattened, separators=(",", ":")).encode("utf-8")
    padding = padded_lanes - actual_lanes
    return {
        "rows": parsed_rows,
        "top_k": parsed_top_k,
        "expert_count": parsed_experts,
        "sparse_layers": len(layers),
        "actual_lanes": actual_lanes,
        "active_expert_groups": active_groups,
        "group_size_histogram": {str(size): count for size, count in sorted(histogram.items())},
        "max_expert_lanes": maximum,
        "compact_tile_rows": parsed_tile,
        "compact_padded_lanes": padded_lanes,
        "compact_padding_lanes": padding,
        "compact_padding_overhead_ratio": padding / actual_lanes,
        "selected_ids_sha256": hashlib.sha256(payload).hexdigest(),
        "layers": layers,
    }


def _timing_order(rows: tuple[int, ...], repetition: int) -> tuple[int, ...]:
    if not rows:
        return ()
    offset = int(repetition) % len(rows)
    rotated = rows[offset:] + rows[:offset]
    return rotated if repetition % 2 == 0 else tuple(reversed(rotated))


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows = tuple(int(value) for value in args.rows)
    if tuple(sorted(set(rows))) != rows or rows != DEFAULT_ROWS:
        raise ValueError(f"retained LPF-0 profiling requires exact rows {DEFAULT_ROWS}")
    if args.repetitions < 2:
        raise ValueError("retained LPF-0 profiling requires at least two repetitions")
    if args.warmups < 0:
        raise ValueError("warmups must be non-negative")
    if args.routing_tile_rows <= 0:
        raise ValueError("routing tile rows must be positive")
    if not args.model.is_file():
        raise FileNotFoundError(f"Laguna model not found: {args.model}")
    if not args.model_sha256:
        raise ValueError("--model-sha256 is required")
    if max(rows) > args.context_length:
        raise ValueError("largest LPF-0 row shape exceeds admitted context")
    repo = _repo_state()
    if not repo["tracked_clean"]:
        raise RuntimeError("retained Laguna LPF-0 profiling requires a clean tracked worktree")

    provenance = collect_artifact_provenance(
        repo_root=Path(__file__).resolve().parents[1],
        configured_backend=args.backend,
        resolved_backend=args.backend,
        target_arch=args.backend.removeprefix("hip_"),
        model_path=args.model,
        quant="gguf_q4_k_m",
        kv_dtype="bf16",
        command=(str(Path(sys.executable).resolve()), *sys.argv),
        build_profile="laguna_prefill_lpf0",
        timing_protocol="prefill_only_one_physical_chunk_rotating_shape_order",
        warmups=args.warmups,
        repetitions=args.repetitions,
    )
    reader = GGUFReader(args.model)
    tokenizer = LagunaGGUFTokenizer.from_gguf_info(reader.info)
    prompts = _load_prompts(args.prompts, tokenizer)
    token_stream, token_source = _profile_token_stream(prompts, max(rows))

    runtime = get_hip_runtime()
    gpu_free_before, gpu_total = runtime.mem_get_info()
    tracked_before = memory_stats()
    owner: LagunaGGUFResidentSession | None = None
    timing_samples: dict[int, list[float]] = {value: [] for value in rows}
    next_tokens: dict[int, list[int]] = {value: [] for value in rows}
    routing: dict[str, Any] = {}
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
            prefill_chunk_size=max(rows),
        )
        load_seconds = time.perf_counter() - load_started
        for _ in range(args.warmups):
            for value in rows:
                owner.reset_state()
                owner.prefill(token_stream[:value], use_bulk=True)
        for repetition in range(args.repetitions):
            for value in _timing_order(rows, repetition):
                owner.reset_state()
                started = time.perf_counter()
                result = owner.prefill(token_stream[:value], use_bulk=True)
                runtime.device_synchronize()
                elapsed = time.perf_counter() - started
                timing_samples[value].append(elapsed)
                next_tokens[value].append(int(result.next_token_id))
                print(
                    f"rep={repetition} rows={value} prefill={value / elapsed:.3f} tok/s "
                    f"next={result.next_token_id}",
                    file=sys.stderr,
                    flush=True,
                )
        if not args.skip_routing_replay:
            for value in rows:
                owner.reset_state()
                replay = owner.prefill_routing_replay(token_stream[:value])
                summary = _summarize_routing_replay(
                    replay.selected_experts,
                    rows=value,
                    top_k=replay.top_k,
                    expert_count=replay.expert_count,
                    tile_rows=args.routing_tile_rows,
                )
                summary["next_token_id"] = int(replay.result.next_token_id)
                routing[str(value)] = summary
        resident_nbytes = owner.resident_nbytes
    finally:
        if owner is not None:
            owner.close()
    tracked_after = memory_stats()
    gpu_free_after, gpu_total_after = runtime.mem_get_info()
    if gpu_total_after != gpu_total:
        raise RuntimeError("HIP total memory changed during Laguna LPF-0 profiling")

    deterministic = all(len(set(values)) == 1 for values in next_tokens.values())
    routing_matches = all(
        int(summary["next_token_id"]) == next_tokens[int(value)][0]
        for value, summary in routing.items()
    )
    recovered = bool(
        tracked_after["current_allocated_bytes"] == tracked_before["current_allocated_bytes"]
        and tracked_after["active_allocations"] == tracked_before["active_allocations"]
    )
    passed = bool(deterministic and routing_matches and recovered)
    timings = {
        str(value): {
            **_summarize_timing_samples(timing_samples[value], rows=value),
            "next_token_ids": next_tokens[value],
        }
        for value in rows
    }
    prompts_payload = args.prompts.read_bytes()
    manifest_path = args.repacked_cache / "manifest.json"
    manifest_sha256 = (
        _sha256_bytes(manifest_path.read_bytes()) if manifest_path.is_file() else None
    )
    return {
        "schema": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "kind": "hipengine_laguna_prefill_lpf0_profile",
        "status": "accepted_dispatch_baseline" if passed else "rejected",
        "pass": passed,
        "performance_claim": False,
        "scope": "Laguna S 2.1 c=1 prefill-only shape baseline and real-routing replay",
        "provenance": provenance,
        "repo": repo,
        "model": {
            "path": str(args.model.resolve()),
            "sha256": args.model_sha256,
            "quant": "Q4_K_M mixed GGUF v3",
            "repacked_cache": str(args.repacked_cache.resolve()),
            "repacked_cache_manifest_sha256": manifest_sha256,
        },
        "platform": {
            "backend": args.backend,
            "target_arch": args.backend.removeprefix("hip_"),
            "device_name": provenance["device_name"],
            "machine": platform.machine(),
            "hip_total_bytes": gpu_total,
        },
        "protocol": {
            "rows": list(rows),
            "one_physical_chunk": True,
            "prefill_chunk_size": max(rows),
            "context_length": args.context_length,
            "repetitions": args.repetitions,
            "warmups_per_shape": args.warmups,
            "timed_order": "rotating and alternating direction by repetition",
            "timing_scope": "reset complete through synchronized first-token projection; load excluded",
            "routing_replay": not args.skip_routing_replay,
            "routing_tile_rows": args.routing_tile_rows,
            "prompt_suite": str(args.prompts.resolve()),
            "prompt_suite_sha256": _sha256_bytes(prompts_payload),
            "token_stream_sha256": _sha256_json(token_stream),
            "token_source": token_source,
        },
        "load": {
            "seconds_excluded": load_seconds,
            "resident_nbytes": resident_nbytes,
        },
        "timings": timings,
        "routing_replay": routing,
        "correctness": {
            "pass": passed,
            "repeat_next_token_deterministic": deterministic,
            "routing_replay_next_token_matches_timing": routing_matches,
            "tracked_returned_to_baseline": recovered,
        },
        "memory": {
            "tracked_before": tracked_before,
            "tracked_after": tracked_after,
            "gpu_free_before": gpu_free_before,
            "gpu_free_after": gpu_free_after,
            "hip_total_bytes": gpu_total,
        },
        "command": [str(Path(sys.executable).resolve()), *sys.argv],
        "notes": [
            "This artifact freezes LPF-0 shape/routing evidence and makes no speedup claim.",
            "The 128-row fixture extends the longest canonical prompt by repeating it without its leading BOS.",
            "Run rocprofv3 separately with cached builds; do not profile model JIT compilation.",
        ],
    }


def main() -> int:
    args = _parse_args()
    result = run(args)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    sys.stdout.write(payload)
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
