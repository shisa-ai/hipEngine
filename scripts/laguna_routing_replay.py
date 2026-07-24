#!/usr/bin/env python3
"""Capture natural and synthetic Laguna routing padding at frozen row sets."""

from __future__ import annotations

import argparse
import base64
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import struct
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np

from hipengine.benchmark.provenance import collect_artifact_provenance
from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import memory_stats
from hipengine.loading.gguf import GGUFReader
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
RETAINED_ROWS = (256, 512)
LAP1_ROWS = (32, 55, 64, 122, 128, 256, 512)
ROUTING_PROTOCOL_ROWS = {
    "retained": RETAINED_ROWS,
    "lap1": LAP1_ROWS,
}
DEFAULT_TILE_ROWS = (2, 4, 8, 16, 32)
DEFAULT_SEED = 20260723
DEFAULT_OUTPUT = (
    ROOT / "benchmarks/results/2026-07-23-gfx1151-laguna-routing-256-512.json"
)


def _parse_int_tuple(value: str, *, name: str) -> tuple[int, ...]:
    parsed = tuple(sorted({int(item) for item in value.split(",") if item.strip()}))
    if not parsed or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError(f"{name} must be positive comma-separated integers")
    return parsed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--backend", default="hip_gfx1151")
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument(
        "--protocol",
        choices=tuple(ROUTING_PROTOCOL_ROWS),
        default="retained",
        help="frozen row-shape protocol; historical commands default to retained",
    )
    parser.add_argument(
        "--rows",
        type=lambda value: _parse_int_tuple(value, name="rows"),
        default=RETAINED_ROWS,
    )
    parser.add_argument(
        "--tile-rows",
        type=lambda value: _parse_int_tuple(value, name="tile rows"),
        default=DEFAULT_TILE_ROWS,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--repacked-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--model-sha256", default=DEFAULT_MODEL_SHA256)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _routing_distribution_summary(
    selected_by_layer: Mapping[int, Sequence[int]],
    *,
    rows: int,
    top_k: int,
    expert_count: int,
    tile_rows: Sequence[int] = DEFAULT_TILE_ROWS,
) -> dict[str, Any]:
    parsed_rows = int(rows)
    parsed_top_k = int(top_k)
    parsed_experts = int(expert_count)
    parsed_tiles = tuple(int(value) for value in tile_rows)
    if parsed_rows <= 0 or parsed_top_k <= 0 or parsed_experts <= 0:
        raise ValueError("routing dimensions must be positive")
    if (
        not parsed_tiles
        or any(value <= 0 for value in parsed_tiles)
        or tuple(sorted(set(parsed_tiles))) != parsed_tiles
    ):
        raise ValueError("tile rows must be sorted, distinct, and positive")

    expected_lanes = parsed_rows * parsed_top_k
    histogram: Counter[int] = Counter()
    layers: dict[str, Any] = {}
    flattened: list[int] = []
    padded_by_tile = {value: 0 for value in parsed_tiles}
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
            raise ValueError(
                f"layer {layer_id} expert IDs must be within [0, {parsed_experts})"
            )
        counts = Counter(selected)
        ordered_counts = tuple(counts.get(expert, 0) for expert in range(parsed_experts))
        for count in counts.values():
            histogram[count] += 1
            active_groups += 1
            maximum = max(maximum, count)
            for tile in parsed_tiles:
                padded_by_tile[tile] += ((count + tile - 1) // tile) * tile
        packed = struct.pack(f"<{parsed_experts}H", *ordered_counts)
        layers[str(layer_id)] = {
            "active_experts": len(counts),
            "max_expert_lanes": max(counts.values()),
            "per_expert_counts_encoding": "uint16_le_dense_expert_id_order_base64",
            "per_expert_counts_u16_le_base64": base64.b64encode(packed).decode("ascii"),
        }
        flattened.extend(selected)

    if not flattened:
        raise ValueError("routing summary requires at least one sparse layer")
    actual_lanes = len(flattened)
    tiles = {}
    for tile in parsed_tiles:
        padded = padded_by_tile[tile]
        padding = padded - actual_lanes
        tiles[str(tile)] = {
            "tile_rows": tile,
            "padded_lanes": padded,
            "padding_lanes": padding,
            "padding_factor": padded / actual_lanes,
            "padding_overhead_ratio": padding / actual_lanes,
        }
    return {
        "rows": parsed_rows,
        "top_k": parsed_top_k,
        "expert_count": parsed_experts,
        "sparse_layers": len(layers),
        "actual_lanes": actual_lanes,
        "active_expert_groups": active_groups,
        "group_size_histogram": {
            str(size): count for size, count in sorted(histogram.items())
        },
        "max_expert_lanes": maximum,
        "tiles": tiles,
        "selected_ids_sha256": hashlib.sha256(
            json.dumps(flattened, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "layers": layers,
    }


def _synthetic_selected_by_layer(
    *,
    rows: int,
    top_k: int,
    expert_count: int,
    sparse_layers: int,
    pattern: str,
    seed: int,
) -> dict[int, tuple[int, ...]]:
    parsed_rows = int(rows)
    parsed_top_k = int(top_k)
    parsed_experts = int(expert_count)
    parsed_layers = int(sparse_layers)
    if (
        parsed_rows <= 0
        or parsed_top_k <= 0
        or parsed_experts <= 0
        or parsed_layers <= 0
        or parsed_top_k > parsed_experts
    ):
        raise ValueError("synthetic routing dimensions are invalid")
    if pattern not in {"hot", "zipf"}:
        raise ValueError("synthetic routing pattern must be 'hot' or 'zipf'")

    result: dict[int, tuple[int, ...]] = {}
    ranks = np.arange(1, parsed_experts + 1, dtype=np.float64)
    probabilities = np.power(ranks, -1.1)
    probabilities /= probabilities.sum()
    for layer_id in range(1, parsed_layers + 1):
        selected: list[int] = []
        if pattern == "hot":
            hot = tuple(range(parsed_top_k))
            for _ in range(parsed_rows):
                selected.extend(hot)
        else:
            rng = np.random.default_rng(int(seed) + layer_id)
            for _ in range(parsed_rows):
                token_experts = rng.choice(
                    parsed_experts,
                    size=parsed_top_k,
                    replace=False,
                    p=probabilities,
                )
                selected.extend(int(value) for value in token_experts)
        result[layer_id] = tuple(selected)
    return result


def _compact_distribution(summary: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in summary.items() if key != "layers"}


def run(args: argparse.Namespace) -> dict[str, Any]:
    protocol = str(args.protocol)
    rows = tuple(int(value) for value in args.rows)
    tiles = tuple(int(value) for value in args.tile_rows)
    expected_rows = ROUTING_PROTOCOL_ROWS[protocol]
    if rows != expected_rows:
        raise ValueError(
            f"{protocol} routing replay requires exact rows {expected_rows}"
        )
    if tiles != DEFAULT_TILE_ROWS:
        raise ValueError(
            f"{protocol} routing replay requires exact tile rows {DEFAULT_TILE_ROWS}"
        )
    if not args.model.is_file():
        raise FileNotFoundError(f"Laguna model not found: {args.model}")
    if max(rows) > args.context_length:
        raise ValueError("largest routing row shape exceeds admitted context")
    if not args.model_sha256:
        raise ValueError("--model-sha256 is required")
    repo = _repo_state()
    if not repo["tracked_clean"]:
        raise RuntimeError(
            f"{protocol} Laguna routing replay requires a clean tracked worktree"
        )

    build_profile = (
        "laguna_routing_256_512"
        if protocol == "retained"
        else "laguna_lap1_routing_shapes"
    )
    provenance = collect_artifact_provenance(
        repo_root=ROOT,
        configured_backend=args.backend,
        resolved_backend=args.backend,
        target_arch=args.backend.removeprefix("hip_"),
        model_path=args.model,
        quant="gguf_q4_k_m",
        kv_dtype="bf16",
        command=(str(Path(sys.executable).resolve()), *sys.argv),
        build_profile=build_profile,
        timing_protocol="untimed_exact_routing_replay_natural_hot_zipf",
        warmups=0,
        repetitions=1,
    )
    reader = GGUFReader(args.model)
    tokenizer = LagunaGGUFTokenizer.from_gguf_info(reader.info)
    prompts = _load_prompts(args.prompts, tokenizer)
    token_stream, token_source = _profile_token_stream(prompts, max(rows))

    runtime = get_hip_runtime()
    gpu_free_before, gpu_total = runtime.mem_get_info()
    tracked_before = memory_stats()
    owner: LagunaGGUFResidentSession | None = None
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
        selected_down_mode = owner.selected_down_mode
        for value in rows:
            owner.reset_state()
            replay = owner.prefill_routing_replay(token_stream[:value])
            natural = _routing_distribution_summary(
                replay.selected_experts,
                rows=value,
                top_k=replay.top_k,
                expert_count=replay.expert_count,
                tile_rows=tiles,
            )
            patterns: dict[str, Any] = {"natural": natural}
            for pattern in ("hot", "zipf"):
                synthetic = _synthetic_selected_by_layer(
                    rows=value,
                    top_k=replay.top_k,
                    expert_count=replay.expert_count,
                    sparse_layers=len(replay.selected_experts),
                    pattern=pattern,
                    seed=args.seed,
                )
                patterns[pattern] = _compact_distribution(
                    _routing_distribution_summary(
                        synthetic,
                        rows=value,
                        top_k=replay.top_k,
                        expert_count=replay.expert_count,
                        tile_rows=tiles,
                    )
                )
            routing[str(value)] = {
                "next_token_id": int(replay.result.next_token_id),
                "patterns": patterns,
            }
            print(
                f"rows={value} natural_groups={natural['active_expert_groups']} "
                f"m16={natural['tiles']['16']['padding_factor']:.4f}x "
                f"m32={natural['tiles']['32']['padding_factor']:.4f}x",
                file=sys.stderr,
                flush=True,
            )
        resident_nbytes = owner.resident_nbytes
    finally:
        if owner is not None:
            owner.close()
    tracked_after = memory_stats()
    gpu_free_after, gpu_total_after = runtime.mem_get_info()
    if gpu_total_after != gpu_total:
        raise RuntimeError("HIP total memory changed during Laguna routing replay")
    recovered = bool(
        tracked_after["current_allocated_bytes"]
        == tracked_before["current_allocated_bytes"]
        and tracked_after["active_allocations"] == tracked_before["active_allocations"]
    )
    valid = all(
        int(result["patterns"]["natural"]["actual_lanes"])
        == value
        * int(result["patterns"]["natural"]["top_k"])
        * int(result["patterns"]["natural"]["sparse_layers"])
        for value, result in ((int(key), value) for key, value in routing.items())
    )
    passed = bool(valid and recovered)
    manifest_path = args.repacked_cache / "manifest.json"
    return {
        "schema": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "kind": (
            "hipengine_laguna_routing_256_512"
            if protocol == "retained"
            else "hipengine_laguna_lap1_routing_shapes"
        ),
        "status": "accepted_routing_diagnostic" if passed else "rejected",
        "pass": passed,
        "performance_claim": False,
        "scope": "Laguna natural/hot/Zipf selected-expert occupancy and tile padding",
        "provenance": provenance,
        "repo": repo,
        "model": {
            "path": str(args.model.resolve()),
            "sha256": args.model_sha256,
            "quant": "Q4_K_M mixed GGUF v3",
            "repacked_cache": str(args.repacked_cache.resolve()),
            "repacked_cache_manifest_sha256": (
                _sha256_bytes(manifest_path.read_bytes())
                if manifest_path.is_file()
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
            "name": protocol,
            "rows": list(rows),
            "tile_rows": list(tiles),
            "patterns": ["natural", "hot", "zipf"],
            "synthetic_seed": args.seed,
            "synthetic_hot": "same top-k experts selected for every token",
            "synthetic_zipf": "per-token weighted sample without replacement, exponent 1.1",
            "prompt_suite": str(args.prompts.resolve()),
            "prompt_suite_sha256": _sha256_bytes(args.prompts.read_bytes()),
            "token_stream_sha256": _sha256_json(token_stream),
            "token_source": token_source,
            "selected_down_mode": selected_down_mode,
            "timing_scope": "untimed routing diagnostic; model load excluded",
        },
        "load": {"seconds_excluded": load_seconds, "resident_nbytes": resident_nbytes},
        "routing": routing,
        "correctness": {
            "pass": passed,
            "every_natural_lane_accounted": valid,
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
            "Natural routing is captured from complete model execution; hot and Zipf are deterministic scheduler controls.",
            "Per-layer natural useful-row counts are stored once as dense uint16 base64 planes.",
            "This artifact selects matrix candidates and makes no throughput claim.",
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
