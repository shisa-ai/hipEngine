#!/usr/bin/env python3
"""Benchmark Laguna selected Q4_K gate/up leaves on actual weights and routes.

The screen keeps one layer's raw GGUF and resident T16 gate/up tensors live
together only to produce a counterbalanced leaf comparison.  This temporary
one-layer allocation is not a proposed runtime sidecar.
"""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any, Callable, Sequence

import numpy as np

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
    memory_stats,
    reset_memory_stats,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_q8_1_selected_prefill import (
    build_gguf_q4_k_q8_1_selected_prefill,
    gguf_q4_k_t16_selected_dual_q8_1_ds4x3_f32_mmq64x32_prefill_compact32_bf16_bf16_out,
    gguf_q4_k_t16_selected_dual_q8_1_ds4_mmq32_prefill_compact32_bf16_bf16_out,
    gguf_q4_k_t16_selected_dual_q8_1_ds4x3_mmq32_prefill_compact32_bf16_bf16_out,
    gguf_q4_k_t16_selected_q8_1_ds4_f32_mmq128x32_wavecols_direct_doublebuf_prefill_compact32_bf16_bf16_out,
    gguf_q4_k_x8_selected_dual_q8_1_ds4_mmq32_prefill_compact32_bf16_bf16_out,
    gguf_q4_k_selected_dual_q8_1_ds4_mmq32_prefill_compact32_bf16_bf16_out,
    gguf_q8_1_mmq_ds4_pack_bf16,
    gguf_q8_1_mmq_ds4_pack_bf16_d4x3,
    gguf_q8_1_mmq_ds4_f32_pack_bf16_d4x3,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_t16_selected_prefill import (
    build_gguf_q4_k_t16_selected_prefill,
    gguf_q4_k_t16_selected_dual_wmma_prefill_compact32_bf16_bf16_out,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_t16_selected_gemv import (
    build_gguf_t16_selected_gemv,
    gguf_q4_k_t16_selected_dual_grouped_smallm_bf16_bf16_out,
    gguf_q4_k_t16_selected_dual_gemv_bf16_bf16_out,
)
from hipengine.loading.gguf import GGUFReader
from hipengine.quant.gguf_x8 import repack_gguf_q4_k_x8

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path("/home/lhl/models/gguf/laguna-s-2.1-Q4_K_M.gguf")
DEFAULT_CACHE = Path(
    "/home/lhl/models/gguf/laguna-s-2.1-Q4_K_M.hipengine-repacked-v1"
)
DEFAULT_ROUTING = Path("/tmp/laguna-lap0-routing-8d26a9562.json")
DEFAULT_OUTPUT = Path("/tmp/laguna-q4-k-mmq-leaf.raw.json")
MODEL_SHA256 = "7da520c5f44bc3c79d4eeebfd1151ba7114c5d7568e72a995638417093c5753f"
DEFAULT_MODES = ("retained-direct", "t16-wmma", "raw-mmq32")
MODES = (
    *DEFAULT_MODES,
    "t16-grouped-exact",
    "x8-mmq32",
    "t16-mmq32",
    "t16-mmq32-d4x3",
    "t16-mmq128x32-d8-f32",
    "t16-mmq128x32-d8-f32-wavecols",
    "t16-mmq128x32-d8-f32-wavecols-direct",
    "t16-mmq128x32-d8-f32-wavecols-direct-doublebuf",
    "t16-mmq128x32-role-gate-d4-up-d8",
    "t16-mmq128x32-role-gate-d8-up-d4",
)
HIDDEN = 3_072
OUT_FEATURES = 1_024
EXPERTS = 256
TOP_K = 10
Q8_DS4_BYTES_PER_128 = 144


def _parse_csv_ints(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item) for item in value.split(",") if item.strip())
    if not parsed or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    if tuple(sorted(set(parsed))) != parsed:
        raise argparse.ArgumentTypeError("values must be sorted and unique")
    return parsed


def _parse_modes(value: str) -> tuple[str, ...]:
    parsed = tuple(item.strip() for item in value.split(",") if item.strip())
    if not parsed or len(set(parsed)) != len(parsed):
        raise argparse.ArgumentTypeError("modes must be non-empty and unique")
    unknown = tuple(mode for mode in parsed if mode not in MODES)
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown modes: {unknown}")
    return parsed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--repacked-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--routing-json", type=Path, default=DEFAULT_ROUTING)
    parser.add_argument("--rows", type=_parse_csv_ints, default=(256, 512))
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--modes", type=_parse_modes, default=DEFAULT_MODES)
    parser.add_argument(
        "--mixed-thresholds",
        type=_parse_csv_ints,
        default=(),
        help=(
            "Also time whole-expert hybrids. MMQ32 owns experts with at least "
            "each threshold's row count; exact grouped-small-M owns the rest."
        ),
    )
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--burst", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ("git", "rev-parse", "HEAD"), cwd=ROOT, text=True
        ).strip()
    except Exception:
        return None


def _tracked_status() -> list[str]:
    try:
        out = subprocess.check_output(
            ("git", "status", "--short", "--untracked-files=no"),
            cwd=ROOT,
            text=True,
        )
    except Exception:
        return []
    return [line for line in out.splitlines() if line]


def _bf16_bits(values: np.ndarray) -> np.ndarray:
    bits = np.asarray(values, dtype=np.float32).view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & np.uint32(1))
    return np.ascontiguousarray((rounded >> 16).astype(np.uint16))


def _bf16_to_f32(values: np.ndarray) -> np.ndarray:
    return (np.asarray(values, dtype=np.uint16).astype(np.uint32) << 16).view(
        np.float32
    )


def _load_counts(
    payload: dict[str, Any], *, rows: int, layer: int
) -> np.ndarray:
    try:
        record = payload["routing"][str(rows)]["patterns"]["natural"]["layers"][
            str(layer)
        ]
    except KeyError as exc:
        raise ValueError(
            f"routing artifact has no natural rows={rows}, layer={layer}"
        ) from exc
    if (
        record.get("per_expert_counts_encoding")
        != "uint16_le_dense_expert_id_order_base64"
    ):
        raise ValueError("unsupported per-expert count encoding")
    counts = np.frombuffer(
        base64.b64decode(record["per_expert_counts_u16_le_base64"]),
        dtype="<u2",
    ).astype(np.int64)
    if counts.shape != (EXPERTS,):
        raise ValueError(f"expected {EXPERTS} expert counts, got {counts.shape}")
    if int(counts.sum()) != rows * TOP_K:
        raise ValueError("natural route count does not equal rows * top_k")
    return counts


def _tile_metadata(
    counts: np.ndarray, tile_rows: int
) -> tuple[np.ndarray, np.ndarray, int]:
    padded = ((counts + tile_rows - 1) // tile_rows) * tile_rows
    starts = np.zeros(counts.size + 1, dtype=np.int64)
    starts[1:] = np.cumsum(padded)
    tile_expert = np.repeat(
        np.arange(counts.size, dtype=np.int64), padded // tile_rows
    )
    total_rows = int(starts[-1])
    if tile_expert.size != total_rows // tile_rows:
        raise AssertionError("tile metadata is inconsistent")
    return starts, np.ascontiguousarray(tile_expert), total_rows


def _stable_compact_to_source(counts: np.ndarray) -> np.ndarray:
    """Realize the recorded expert degrees as stable, valid top-k source rows."""

    rows = int(counts.sum()) // TOP_K
    remaining = counts.astype(np.int64, copy=True)
    by_expert: list[list[int]] = [[] for _ in range(counts.size)]
    expert_ids = np.arange(counts.size, dtype=np.int64)
    for source_row in range(rows):
        chosen = np.lexsort((expert_ids, -remaining))[:TOP_K]
        if np.any(remaining[chosen] <= 0):
            raise ValueError("recorded expert counts do not admit a top-k replay")
        for expert in chosen:
            by_expert[int(expert)].append(source_row)
        remaining[chosen] -= 1
    if np.any(remaining):
        raise ValueError("recorded expert counts were not exhausted")
    compact_to_source = np.asarray(
        [row for expert_rows in by_expert for row in expert_rows],
        dtype=np.int64,
    )
    replayed = np.bincount(compact_to_source, minlength=rows)
    if compact_to_source.size != rows * TOP_K or np.any(replayed != TOP_K):
        raise AssertionError("reconstructed routing is not a valid top-k assignment")
    return np.ascontiguousarray(compact_to_source)


def _route_metadata(counts: np.ndarray) -> dict[str, np.ndarray | int]:
    starts = np.zeros(counts.size + 1, dtype=np.int64)
    starts[1:] = np.cumsum(counts)
    compact_rows = int(starts[-1])
    compact_experts = np.repeat(
        np.arange(counts.size, dtype=np.int64), counts
    )
    compact_to_source = _stable_compact_to_source(counts)
    starts16, tile_expert16, total16 = _tile_metadata(counts, 16)
    starts32, tile_expert32, total32 = _tile_metadata(counts, 32)
    return {
        "starts": starts,
        "compact_experts": np.ascontiguousarray(compact_experts),
        "compact_to_source": np.ascontiguousarray(compact_to_source),
        "starts16": starts16,
        "tile_expert16": tile_expert16,
        "total16": total16,
        "starts32": starts32,
        "tile_expert32": tile_expert32,
        "total32": total32,
    }


def _mixed_metadata(
    counts: np.ndarray, *, min_mmq_rows: int
) -> dict[str, np.ndarray | int]:
    """Partition active experts between whole-expert MMQ32 and exact leaves."""

    if min_mmq_rows <= 0:
        raise ValueError("min_mmq_rows must be positive")
    dense_counts = np.asarray(counts, dtype=np.int64)
    if dense_counts.ndim != 1 or np.any(dense_counts < 0):
        raise ValueError("counts must be a one-dimensional nonnegative array")

    mmq_mask = dense_counts >= min_mmq_rows
    exact_mask = (dense_counts > 0) & ~mmq_mask
    padded = np.where(
        mmq_mask,
        ((dense_counts + 31) // 32) * 32,
        0,
    )
    starts32 = np.zeros(dense_counts.size + 1, dtype=np.int64)
    starts32[1:] = np.cumsum(padded)
    tile_expert32 = np.repeat(
        np.arange(dense_counts.size, dtype=np.int64), padded // 32
    )
    total32 = int(starts32[-1])
    if tile_expert32.size != total32 // 32:
        raise AssertionError("mixed MMQ32 metadata is inconsistent")

    return {
        "starts32": np.ascontiguousarray(starts32),
        "tile_expert32": np.ascontiguousarray(tile_expert32),
        "total32": total32,
        "mmq_experts": np.ascontiguousarray(np.flatnonzero(mmq_mask)),
        "exact_experts": np.ascontiguousarray(np.flatnonzero(exact_mask)),
        "mmq_compact_rows": int(dense_counts[mmq_mask].sum()),
        "exact_compact_rows": int(dense_counts[exact_mask].sum()),
    }


def _cache_tiles(
    cache_root: Path, *, layer: int, slot: str
) -> tuple[np.ndarray, dict[str, Any]]:
    manifest = json.loads((cache_root / "manifest.json").read_text(encoding="utf-8"))
    entry = manifest["entries"][f"layers.{layer}.{slot}"]
    if entry["layout"] != "gguf_q4_k_t16_v1":
        raise ValueError(f"{slot} does not use the expected Q4_K T16 layout")
    allocation = entry["allocations"]["tiles"]
    tiles = np.load(cache_root / allocation["file"], mmap_mode="r")
    if tuple(tiles.shape) != (EXPERTS, OUT_FEATURES // 16, HIDDEN // 256, 2368):
        raise ValueError(f"unexpected {slot} T16 shape: {tiles.shape}")
    return tiles, entry


def _upload(runtime, values: np.ndarray):
    array = np.asarray(values)
    if not array.flags.c_contiguous:
        array = np.ascontiguousarray(array)
    buffer = malloc(array.nbytes, runtime=runtime)
    copy_host_to_device(buffer, host_array_ptr(array), runtime=runtime)
    return buffer


def _event_ms(runtime, fn: Callable[[], None], *, burst: int) -> float:
    start = runtime.event_create()
    end = runtime.event_create()
    try:
        runtime.event_record(start)
        for _ in range(burst):
            fn()
        runtime.event_record(end)
        runtime.event_synchronize(end)
        return float(runtime.event_elapsed_time_ms(start, end)) / burst
    finally:
        runtime.event_destroy(end)
        runtime.event_destroy(start)


def _mode_order(modes: Sequence[str], sample: int) -> tuple[str, ...]:
    offset = sample % len(modes)
    return tuple(modes[offset:]) + tuple(modes[:offset])


def main() -> None:
    args = _parse_args()
    if args.layer <= 0:
        raise SystemExit("--layer must name a sparse layer above zero")
    if args.warmups < 0 or args.samples <= 0 or args.burst <= 0:
        raise SystemExit("warmups must be non-negative; samples/burst must be positive")
    if args.compiler_version_file is not None:
        import os

        os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(
            args.compiler_version_file
        )

    routing = json.loads(args.routing_json.read_text(encoding="utf-8"))
    reader = GGUFReader(args.model)
    raw_gate = np.asarray(
        reader.tensor_data(f"blk.{args.layer}.ffn_gate_exps.weight")
    )
    raw_up = np.asarray(reader.tensor_data(f"blk.{args.layer}.ffn_up_exps.weight"))
    expected_raw_shape = (EXPERTS, OUT_FEATURES, HIDDEN * 144 // 256)
    if (
        raw_gate.shape != expected_raw_shape
        or raw_up.shape != expected_raw_shape
        or raw_gate.dtype != np.uint8
        or raw_up.dtype != np.uint8
    ):
        raise ValueError(
            f"expected raw Q4_K gate/up shape {expected_raw_shape}, got "
            f"{raw_gate.shape}/{raw_up.shape}"
        )
    needs_x8 = "x8-mmq32" in args.modes
    x8_gate = repack_gguf_q4_k_x8(raw_gate).tiles if needs_x8 else None
    x8_up = repack_gguf_q4_k_x8(raw_up).tiles if needs_x8 else None
    tiles_gate, gate_entry = _cache_tiles(
        args.repacked_cache, layer=args.layer, slot="ffn_gate_exps"
    )
    tiles_up, up_entry = _cache_tiles(
        args.repacked_cache, layer=args.layer, slot="ffn_up_exps"
    )

    runtime = get_hip_runtime()
    reset_memory_stats()
    mmq_library = build_gguf_q4_k_q8_1_selected_prefill(
        load=True, require_cached=args.require_cached_build
    )
    wmma_library = build_gguf_q4_k_t16_selected_prefill(
        load=True, require_cached=args.require_cached_build
    )
    direct_library = build_gguf_t16_selected_gemv(
        load=True, require_cached=args.require_cached_build
    )

    resident_buffers = []
    results: dict[str, Any] = {}
    try:
        raw_gate_dev = _upload(runtime, raw_gate)
        raw_up_dev = _upload(runtime, raw_up)
        tiles_gate_dev = _upload(runtime, tiles_gate)
        tiles_up_dev = _upload(runtime, tiles_up)
        resident_buffers.extend(
            (raw_gate_dev, raw_up_dev, tiles_gate_dev, tiles_up_dev)
        )
        x8_gate_dev = _upload(runtime, x8_gate) if x8_gate is not None else None
        x8_up_dev = _upload(runtime, x8_up) if x8_up is not None else None
        if x8_gate_dev is not None and x8_up_dev is not None:
            resident_buffers.extend((x8_gate_dev, x8_up_dev))

        for rows in args.rows:
            counts = _load_counts(routing, rows=rows, layer=args.layer)
            metadata = _route_metadata(counts)
            mixed_metadata = {
                threshold: _mixed_metadata(counts, min_mmq_rows=threshold)
                for threshold in args.mixed_thresholds
            }
            mixed_mode_names = tuple(
                f"mixed-ge{threshold}" for threshold in args.mixed_thresholds
            )
            effective_modes = (*args.modes, *mixed_mode_names)
            compact_rows = int(metadata["starts"][-1])
            rng = np.random.default_rng(args.seed + rows + args.layer * 1000)
            source_x = _bf16_bits(
                rng.normal(0.0, 0.55, size=(rows, HIDDEN)).astype(np.float32)
            )
            compact_x = np.ascontiguousarray(
                source_x[np.asarray(metadata["compact_to_source"])]
            )
            out_shape = (compact_rows, OUT_FEATURES)
            out_bytes = compact_rows * OUT_FEATURES * np.dtype(np.uint16).itemsize
            q8_planes = (
                3
                if "t16-mmq32-d4x3" in effective_modes
                else 2
                if any("-role-" in mode for mode in effective_modes)
                else 1
            )
            q8_block_bytes = (
                160
                if any(
                    "d8-f32" in mode or "-role-" in mode
                    for mode in effective_modes
                )
                else Q8_DS4_BYTES_PER_128
            )
            q8_bytes = (
                q8_planes
                * rows
                * (HIDDEN // 128)
                * q8_block_bytes
            )

            shape_buffers = []
            try:
                source_x_dev = _upload(runtime, source_x)
                compact_x_dev = _upload(runtime, compact_x)
                selected_dev = _upload(runtime, metadata["compact_experts"])
                starts_dev = _upload(runtime, metadata["starts"])
                starts16_dev = _upload(runtime, metadata["starts16"])
                tile_expert16_dev = _upload(runtime, metadata["tile_expert16"])
                starts32_dev = _upload(runtime, metadata["starts32"])
                tile_expert32_dev = _upload(runtime, metadata["tile_expert32"])
                compact_to_source_dev = _upload(
                    runtime, metadata["compact_to_source"]
                )
                active_experts = np.ascontiguousarray(
                    np.flatnonzero(counts > 0), dtype=np.int64
                )
                active_experts_dev = _upload(runtime, active_experts)
                active_count_dev = _upload(
                    runtime, np.asarray([active_experts.size], dtype=np.int64)
                )
                q8_dev = malloc(q8_bytes, runtime=runtime)
                out_a_dev = malloc(out_bytes, runtime=runtime)
                out_b_dev = malloc(out_bytes, runtime=runtime)
                out_dual_dev = malloc(2 * out_bytes, runtime=runtime)
                mixed_devices: dict[int, dict[str, Any]] = {}
                for threshold, hybrid in mixed_metadata.items():
                    hybrid_devices: dict[str, Any] = {
                        "starts32": _upload(runtime, hybrid["starts32"]),
                        "exact_count": _upload(
                            runtime,
                            np.asarray(
                                [len(hybrid["exact_experts"])], dtype=np.int64
                            ),
                        ),
                    }
                    if int(hybrid["total32"]) > 0:
                        hybrid_devices["tile_expert32"] = _upload(
                            runtime, hybrid["tile_expert32"]
                        )
                    if len(hybrid["exact_experts"]) > 0:
                        hybrid_devices["exact_experts"] = _upload(
                            runtime, hybrid["exact_experts"]
                        )
                    mixed_devices[threshold] = hybrid_devices
                shape_buffers.extend(
                    (
                        source_x_dev,
                        compact_x_dev,
                        selected_dev,
                        starts_dev,
                        starts16_dev,
                        tile_expert16_dev,
                        starts32_dev,
                        tile_expert32_dev,
                        compact_to_source_dev,
                        active_experts_dev,
                        active_count_dev,
                        q8_dev,
                        out_a_dev,
                        out_b_dev,
                        out_dual_dev,
                    )
                )
                shape_buffers.extend(
                    buffer
                    for hybrid_devices in mixed_devices.values()
                    for buffer in hybrid_devices.values()
                )

                def retained_direct() -> None:
                    gguf_q4_k_t16_selected_dual_gemv_bf16_bf16_out(
                        source_x_dev.ptr,
                        selected_dev.ptr,
                        tiles_gate_dev.ptr,
                        tiles_up_dev.ptr,
                        out_a_dev.ptr,
                        out_b_dev.ptr,
                        rows,
                        compact_rows,
                        EXPERTS,
                        HIDDEN,
                        OUT_FEATURES,
                        library=direct_library,
                        runtime=runtime,
                    )

                def t16_wmma() -> None:
                    gguf_q4_k_t16_selected_dual_wmma_prefill_compact32_bf16_bf16_out(
                        compact_x_dev.ptr,
                        starts_dev.ptr,
                        starts16_dev.ptr,
                        tile_expert16_dev.ptr,
                        tiles_gate_dev.ptr,
                        tiles_up_dev.ptr,
                        out_dual_dev.ptr,
                        compact_rows,
                        HIDDEN,
                        OUT_FEATURES,
                        OUT_FEATURES,
                        EXPERTS,
                        int(metadata["total16"]),
                        library=wmma_library,
                        runtime=runtime,
                    )

                def t16_grouped_exact() -> None:
                    gguf_q4_k_t16_selected_dual_grouped_smallm_bf16_bf16_out(
                        compact_x_dev.ptr,
                        starts_dev.ptr,
                        active_experts_dev.ptr,
                        active_count_dev.ptr,
                        tiles_gate_dev.ptr,
                        tiles_up_dev.ptr,
                        out_a_dev.ptr,
                        out_b_dev.ptr,
                        compact_rows,
                        HIDDEN,
                        OUT_FEATURES,
                        EXPERTS,
                        library=direct_library,
                        runtime=runtime,
                    )

                def raw_mmq32() -> None:
                    gguf_q8_1_mmq_ds4_pack_bf16(
                        source_x_dev.ptr,
                        q8_dev.ptr,
                        rows,
                        HIDDEN,
                        library=mmq_library,
                        runtime=runtime,
                    )
                    gguf_q4_k_selected_dual_q8_1_ds4_mmq32_prefill_compact32_bf16_bf16_out(
                        q8_dev.ptr,
                        compact_to_source_dev.ptr,
                        starts_dev.ptr,
                        starts32_dev.ptr,
                        tile_expert32_dev.ptr,
                        raw_gate_dev.ptr,
                        raw_up_dev.ptr,
                        out_dual_dev.ptr,
                        compact_rows,
                        HIDDEN,
                        OUT_FEATURES,
                        OUT_FEATURES,
                        EXPERTS,
                        int(metadata["total32"]),
                        library=mmq_library,
                        runtime=runtime,
                    )

                def x8_mmq32() -> None:
                    assert x8_gate_dev is not None and x8_up_dev is not None
                    gguf_q8_1_mmq_ds4_pack_bf16(
                        source_x_dev.ptr,
                        q8_dev.ptr,
                        rows,
                        HIDDEN,
                        library=mmq_library,
                        runtime=runtime,
                    )
                    gguf_q4_k_x8_selected_dual_q8_1_ds4_mmq32_prefill_compact32_bf16_bf16_out(
                        q8_dev.ptr,
                        compact_to_source_dev.ptr,
                        starts_dev.ptr,
                        starts32_dev.ptr,
                        tile_expert32_dev.ptr,
                        x8_gate_dev.ptr,
                        x8_up_dev.ptr,
                        out_dual_dev.ptr,
                        compact_rows,
                        HIDDEN,
                        OUT_FEATURES,
                        OUT_FEATURES,
                        EXPERTS,
                        int(metadata["total32"]),
                        library=mmq_library,
                        runtime=runtime,
                    )

                def t16_mmq32() -> None:
                    gguf_q8_1_mmq_ds4_pack_bf16(
                        source_x_dev.ptr,
                        q8_dev.ptr,
                        rows,
                        HIDDEN,
                        library=mmq_library,
                        runtime=runtime,
                    )
                    gguf_q4_k_t16_selected_dual_q8_1_ds4_mmq32_prefill_compact32_bf16_bf16_out(
                        q8_dev.ptr,
                        compact_to_source_dev.ptr,
                        starts_dev.ptr,
                        starts32_dev.ptr,
                        tile_expert32_dev.ptr,
                        tiles_gate_dev.ptr,
                        tiles_up_dev.ptr,
                        out_dual_dev.ptr,
                        compact_rows,
                        HIDDEN,
                        OUT_FEATURES,
                        OUT_FEATURES,
                        EXPERTS,
                        int(metadata["total32"]),
                        library=mmq_library,
                        runtime=runtime,
                    )

                def t16_mmq32_d4x3() -> None:
                    gguf_q8_1_mmq_ds4_pack_bf16_d4x3(
                        source_x_dev.ptr,
                        q8_dev.ptr,
                        rows,
                        HIDDEN,
                        library=mmq_library,
                        runtime=runtime,
                    )
                    gguf_q4_k_t16_selected_dual_q8_1_ds4x3_mmq32_prefill_compact32_bf16_bf16_out(
                        q8_dev.ptr,
                        compact_to_source_dev.ptr,
                        starts_dev.ptr,
                        starts32_dev.ptr,
                        tile_expert32_dev.ptr,
                        tiles_gate_dev.ptr,
                        tiles_up_dev.ptr,
                        out_dual_dev.ptr,
                        compact_rows,
                        rows,
                        HIDDEN,
                        OUT_FEATURES,
                        OUT_FEATURES,
                        EXPERTS,
                        int(metadata["total32"]),
                        library=mmq_library,
                        runtime=runtime,
                    )

                def t16_mmq128_d8_f32(
                    *,
                    wave_cols: bool,
                    direct_wave_decode: bool = False,
                    double_buffer_activation: bool = False,
                    split16: bool = True,
                ) -> None:
                    gguf_q8_1_mmq_ds4_f32_pack_bf16_d4x3(
                        source_x_dev.ptr,
                        q8_dev.ptr,
                        rows,
                        HIDDEN,
                        residual_passes=1,
                        split16=split16,
                        library=mmq_library,
                        runtime=runtime,
                    )
                    gguf_q4_k_t16_selected_dual_q8_1_ds4x3_f32_mmq64x32_prefill_compact32_bf16_bf16_out(
                        q8_dev.ptr,
                        compact_to_source_dev.ptr,
                        starts_dev.ptr,
                        starts32_dev.ptr,
                        tile_expert32_dev.ptr,
                        tiles_gate_dev.ptr,
                        tiles_up_dev.ptr,
                        out_dual_dev.ptr,
                        compact_rows,
                        rows,
                        HIDDEN,
                        OUT_FEATURES,
                        OUT_FEATURES,
                        EXPERTS,
                        int(metadata["total32"]),
                        residual_passes=1,
                        split16=split16,
                        rowvec=True,
                        wave_cols=wave_cols,
                        direct_wave_decode=direct_wave_decode,
                        double_buffer_activation=double_buffer_activation,
                        library=mmq_library,
                        runtime=runtime,
                    )

                def t16_mmq128_role_split(
                    *,
                    gate_split16: bool,
                ) -> None:
                    q8_plane_bytes = (
                        rows * (HIDDEN // 128) * 160
                    )
                    q8_d8_ptr = q8_dev.ptr
                    q8_d4_ptr = q8_dev.ptr + q8_plane_bytes
                    gguf_q8_1_mmq_ds4_f32_pack_bf16_d4x3(
                        source_x_dev.ptr,
                        q8_d8_ptr,
                        rows,
                        HIDDEN,
                        residual_passes=1,
                        split16=True,
                        library=mmq_library,
                        runtime=runtime,
                    )
                    gguf_q8_1_mmq_ds4_f32_pack_bf16_d4x3(
                        source_x_dev.ptr,
                        q8_d4_ptr,
                        rows,
                        HIDDEN,
                        residual_passes=1,
                        split16=False,
                        library=mmq_library,
                        runtime=runtime,
                    )
                    gguf_q4_k_t16_selected_q8_1_ds4_f32_mmq128x32_wavecols_direct_doublebuf_prefill_compact32_bf16_bf16_out(
                        q8_d8_ptr if gate_split16 else q8_d4_ptr,
                        compact_to_source_dev.ptr,
                        starts_dev.ptr,
                        starts32_dev.ptr,
                        tile_expert32_dev.ptr,
                        tiles_gate_dev.ptr,
                        out_a_dev.ptr,
                        compact_rows,
                        rows,
                        HIDDEN,
                        OUT_FEATURES,
                        EXPERTS,
                        int(metadata["total32"]),
                        split16=gate_split16,
                        library=mmq_library,
                        runtime=runtime,
                    )
                    gguf_q4_k_t16_selected_q8_1_ds4_f32_mmq128x32_wavecols_direct_doublebuf_prefill_compact32_bf16_bf16_out(
                        q8_d4_ptr if gate_split16 else q8_d8_ptr,
                        compact_to_source_dev.ptr,
                        starts_dev.ptr,
                        starts32_dev.ptr,
                        tile_expert32_dev.ptr,
                        tiles_up_dev.ptr,
                        out_b_dev.ptr,
                        compact_rows,
                        rows,
                        HIDDEN,
                        OUT_FEATURES,
                        EXPERTS,
                        int(metadata["total32"]),
                        split16=not gate_split16,
                        library=mmq_library,
                        runtime=runtime,
                    )

                launchers = {
                    "retained-direct": retained_direct,
                    "t16-wmma": t16_wmma,
                    "t16-grouped-exact": t16_grouped_exact,
                    "raw-mmq32": raw_mmq32,
                    "x8-mmq32": x8_mmq32,
                    "t16-mmq32": t16_mmq32,
                    "t16-mmq32-d4x3": t16_mmq32_d4x3,
                    "t16-mmq128x32-d8-f32": lambda: t16_mmq128_d8_f32(
                        wave_cols=False
                    ),
                    "t16-mmq128x32-d8-f32-wavecols": (
                        lambda: t16_mmq128_d8_f32(wave_cols=True)
                    ),
                    "t16-mmq128x32-d8-f32-wavecols-direct": (
                        lambda: t16_mmq128_d8_f32(
                            wave_cols=True,
                            direct_wave_decode=True,
                        )
                    ),
                    "t16-mmq128x32-d8-f32-wavecols-direct-doublebuf": (
                        lambda: t16_mmq128_d8_f32(
                            wave_cols=True,
                            direct_wave_decode=True,
                            double_buffer_activation=True,
                        )
                    ),
                    "t16-mmq128x32-role-gate-d4-up-d8": (
                        lambda: t16_mmq128_role_split(gate_split16=False)
                    ),
                    "t16-mmq128x32-role-gate-d8-up-d4": (
                        lambda: t16_mmq128_role_split(gate_split16=True)
                    ),
                }
                for threshold, hybrid in mixed_metadata.items():
                    hybrid_devices = mixed_devices[threshold]

                    def mixed_launcher(
                        *,
                        _hybrid=hybrid,
                        _devices=hybrid_devices,
                    ) -> None:
                        if int(_hybrid["total32"]) > 0:
                            gguf_q8_1_mmq_ds4_pack_bf16(
                                source_x_dev.ptr,
                                q8_dev.ptr,
                                rows,
                                HIDDEN,
                                library=mmq_library,
                                runtime=runtime,
                            )
                            gguf_q4_k_selected_dual_q8_1_ds4_mmq32_prefill_compact32_bf16_bf16_out(
                                q8_dev.ptr,
                                compact_to_source_dev.ptr,
                                starts_dev.ptr,
                                _devices["starts32"].ptr,
                                _devices["tile_expert32"].ptr,
                                raw_gate_dev.ptr,
                                raw_up_dev.ptr,
                                out_dual_dev.ptr,
                                compact_rows,
                                HIDDEN,
                                OUT_FEATURES,
                                OUT_FEATURES,
                                EXPERTS,
                                int(_hybrid["total32"]),
                                library=mmq_library,
                                runtime=runtime,
                            )
                        if len(_hybrid["exact_experts"]) > 0:
                            gguf_q4_k_t16_selected_dual_grouped_smallm_bf16_bf16_out(
                                compact_x_dev.ptr,
                                starts_dev.ptr,
                                _devices["exact_experts"].ptr,
                                _devices["exact_count"].ptr,
                                tiles_gate_dev.ptr,
                                tiles_up_dev.ptr,
                                out_a_dev.ptr,
                                out_b_dev.ptr,
                                compact_rows,
                                HIDDEN,
                                OUT_FEATURES,
                                EXPERTS,
                                library=direct_library,
                                runtime=runtime,
                            )

                    launchers[f"mixed-ge{threshold}"] = mixed_launcher
                for _ in range(args.warmups):
                    for mode in effective_modes:
                        launchers[mode]()
                runtime.device_synchronize()

                samples: dict[str, list[float]] = {
                    mode: [] for mode in effective_modes
                }
                for sample in range(args.samples):
                    for mode in _mode_order(effective_modes, sample):
                        samples[mode].append(
                            _event_ms(runtime, launchers[mode], burst=args.burst)
                        )

                sanity: dict[str, Any] = {}
                sanity_values: dict[str, np.ndarray] = {}
                for mode in effective_modes:
                    launchers[mode]()
                    runtime.device_synchronize()
                    if mode in ("retained-direct", "t16-grouped-exact") or (
                        "-role-" in mode
                    ):
                        host_a = np.empty(out_shape, dtype=np.uint16)
                        host_b = np.empty(out_shape, dtype=np.uint16)
                        copy_device_to_host(
                            host_array_ptr(host_a), out_a_dev, runtime=runtime
                        )
                        copy_device_to_host(
                            host_array_ptr(host_b), out_b_dev, runtime=runtime
                        )
                        values = np.concatenate((host_a, host_b), axis=1)
                    elif mode.startswith("mixed-ge"):
                        threshold = int(mode.removeprefix("mixed-ge"))
                        hybrid = mixed_metadata[threshold]
                        values = np.empty(
                            (compact_rows, 2 * OUT_FEATURES), dtype=np.uint16
                        )
                        copy_device_to_host(
                            host_array_ptr(values), out_dual_dev, runtime=runtime
                        )
                        if len(hybrid["exact_experts"]) > 0:
                            host_a = np.empty(out_shape, dtype=np.uint16)
                            host_b = np.empty(out_shape, dtype=np.uint16)
                            copy_device_to_host(
                                host_array_ptr(host_a), out_a_dev, runtime=runtime
                            )
                            copy_device_to_host(
                                host_array_ptr(host_b), out_b_dev, runtime=runtime
                            )
                            for expert in hybrid["exact_experts"]:
                                begin = int(metadata["starts"][expert])
                                end = int(metadata["starts"][expert + 1])
                                values[begin:end, :OUT_FEATURES] = host_a[begin:end]
                                values[begin:end, OUT_FEATURES:] = host_b[begin:end]
                    else:
                        values = np.empty(
                            (compact_rows, 2 * OUT_FEATURES), dtype=np.uint16
                        )
                        copy_device_to_host(
                            host_array_ptr(values), out_dual_dev, runtime=runtime
                        )
                    values_f32 = _bf16_to_f32(values)
                    sanity_values[mode] = values
                    sanity[mode] = {
                        "finite": bool(np.isfinite(values_f32).all()),
                        "checksum_f64": float(values_f32.astype(np.float64).sum()),
                        "max_abs": float(np.max(np.abs(values_f32))),
                    }
                exact_mode = (
                    "t16-grouped-exact"
                    if "t16-grouped-exact" in sanity_values
                    else None
                )
                if exact_mode is not None:
                    exact_values = sanity_values[exact_mode]
                    for mode, values in sanity_values.items():
                        if mode in ("retained-direct", exact_mode):
                            continue
                        mismatches = values != exact_values
                        sanity[mode]["bf16_mismatches_vs_exact"] = int(
                            np.count_nonzero(mismatches)
                        )
                        sanity[mode]["bf16_mismatch_fraction_vs_exact"] = float(
                            np.mean(mismatches)
                        )
                        sanity[mode]["exact_mode"] = exact_mode

                medians = {
                    mode: statistics.median(values)
                    for mode, values in samples.items()
                }
                shape_result: dict[str, Any] = {
                    "rows": rows,
                    "compact_rows": compact_rows,
                    "active_experts": int(np.count_nonzero(counts)),
                    "max_expert_rows": int(counts.max()),
                    "route_count_sha256": hashlib.sha256(
                        counts.astype("<u2").tobytes()
                    ).hexdigest(),
                    "padding": {
                        "t16_rows": int(metadata["total16"]),
                        "t16_factor": int(metadata["total16"]) / compact_rows,
                        "mmq32_rows": int(metadata["total32"]),
                        "mmq32_factor": int(metadata["total32"]) / compact_rows,
                    },
                    "mixed": {
                        str(threshold): {
                            "min_mmq_rows": threshold,
                            "mmq_experts": len(hybrid["mmq_experts"]),
                            "exact_experts": len(hybrid["exact_experts"]),
                            "mmq_compact_rows": int(hybrid["mmq_compact_rows"]),
                            "exact_compact_rows": int(
                                hybrid["exact_compact_rows"]
                            ),
                            "mmq32_rows": int(hybrid["total32"]),
                            "mmq32_padding_factor_over_mmq_rows": (
                                int(hybrid["total32"])
                                / int(hybrid["mmq_compact_rows"])
                                if int(hybrid["mmq_compact_rows"]) > 0
                                else None
                            ),
                        }
                        for threshold, hybrid in mixed_metadata.items()
                    },
                    "samples_ms": samples,
                    "median_ms": medians,
                    "sanity": sanity,
                }
                if "retained-direct" in medians:
                    shape_result["speedup_vs_retained"] = {
                        mode: medians["retained-direct"] / value
                        for mode, value in medians.items()
                        if mode != "retained-direct"
                    }
                results[str(rows)] = shape_result
                print(
                    f"rows={rows} active={shape_result['active_experts']} "
                    f"max={shape_result['max_expert_rows']} "
                    + " ".join(
                        f"{mode}={medians[mode]:.3f}ms"
                        for mode in effective_modes
                    ),
                    flush=True,
                )
            finally:
                for buffer in reversed(shape_buffers):
                    free(buffer, runtime=runtime)
    finally:
        for buffer in reversed(resident_buffers):
            free(buffer, runtime=runtime)

    stats = memory_stats()
    artifact = {
        "schema": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "kind": "hipengine_laguna_q4_k_mmq_actual_leaf",
        "status": "diagnostic",
        "performance_claim": False,
        "scope": "actual one-layer weights plus natural per-expert count replay",
        "model": {
            "path": str(args.model.resolve()),
            "sha256": MODEL_SHA256,
            "layer": args.layer,
            "gate_tensor": f"blk.{args.layer}.ffn_gate_exps.weight",
            "up_tensor": f"blk.{args.layer}.ffn_up_exps.weight",
        },
        "routing": {
            "artifact": str(args.routing_json.resolve()),
            "rows": list(args.rows),
            "top_k": TOP_K,
            "qualification": (
                "Per-expert counts are the recorded natural counts. The leaf "
                "reconstructs a deterministic lane order because raw selected "
                "IDs were intentionally not persisted."
            ),
        },
        "layouts": {
            "raw_q4_k_bytes": int(raw_gate.nbytes + raw_up.nbytes),
            "x8_q4_k_bytes": (
                int(x8_gate.nbytes + x8_up.nbytes)
                if x8_gate is not None and x8_up is not None
                else None
            ),
            "t16_bytes": int(tiles_gate.nbytes + tiles_up.nbytes),
            "t16_entries": {
                "gate": gate_entry,
                "up": up_entry,
            },
            "temporary_side_by_side_leaf_only": True,
        },
        "protocol": {
            "modes": [
                *args.modes,
                *(f"mixed-ge{threshold}" for threshold in args.mixed_thresholds),
            ],
            "mixed_thresholds": list(args.mixed_thresholds),
            "mixed_scope": (
                "MMQ32 for whole experts whose natural row count is greater "
                "than or equal to the global threshold; exact T16 grouped-"
                "small-M for every other active expert. Timing includes both "
                "bodies in separate diagnostic output buffers. Host sanity "
                "reconstructs disjoint expert rows; device merge/scatter cost "
                "is omitted."
            ),
            "warmups": args.warmups,
            "samples": args.samples,
            "burst": args.burst,
            "timing": "counter-rotated HIP-event elapsed time",
            "raw_mmq32_inclusive": "BF16 producer-row DS4 pack plus one dual gate/up MMQ launch",
            "x8_mmq32_inclusive": "BF16 producer-row DS4 pack plus one dual gate/up X8 MMQ launch",
            "t16_mmq32_inclusive": "BF16 producer-row DS4 pack plus one dual gate/up direct-T16 MMQ launch",
            "activation_pack": "once per producer row; compact rows index producer Q8 blocks",
        },
        "repo": {
            "revision": _git_revision(),
            "tracked_status": _tracked_status(),
        },
        "memory": {
            "tracked_peak_bytes": stats["peak_allocated_bytes"],
            "tracked_after_bytes": stats["current_allocated_bytes"],
        },
        "results": results,
        "notes": [
            "The retained-direct mode is the exact production gate/up body named by LAP-0.",
            "T16 WMMA is a diagnostic layout/body control, not the shipping route.",
            "X8 is a byte-exact, byte-neutral replacement layout; raw and X8 are resident together only for this leaf comparison.",
            "T16 MMQ reads the existing resident T16 bytes directly without a layout transpose or sidecar.",
            "Mixed modes are a temporary two-layout leaf ceiling, not a resident-layout proposal.",
            "This leaf does not select a quality policy or change runtime dispatch.",
        ],
    }
    if not all(
        mode["finite"]
        for shape in results.values()
        for mode in shape["sanity"].values()
    ):
        artifact["status"] = "failed_nonfinite"
        raise SystemExit("non-finite output in leaf screen")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(artifact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
