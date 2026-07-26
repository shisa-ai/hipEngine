#!/usr/bin/env python3
"""Benchmark byte-neutral Q6T16 qmicro on actual Laguna weights and routing."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import subprocess

import numpy as np

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
    memory_stats,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_q8_1_selected_prefill import (
    build_gguf_q4_k_q8_1_selected_prefill,
    gguf_q6_k_t16_selected_q8_1_ds4x3_f32_mmq64x32_prefill_compact32_bf16_bf16_out,
    gguf_q8_1_mmq_ds4_f32_pack_bf16_d4x3,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_t16_selected_gemv import (
    build_gguf_t16_selected_gemv,
    gguf_q6_k_t16_selected_gemv_bf16_bf16_out,
)
from hipengine.quant.gguf_t16 import convert_gguf_q6_k_tile16_to_qmicro
from scripts.laguna_q4_k_mmq_leaf_bench import (
    DEFAULT_CACHE,
    DEFAULT_ROUTING,
    _bf16_bits,
    _event_ms,
    _load_counts,
    _tile_metadata,
    _upload,
)
from scripts.laguna_target_ar_bench import _compiler_version

ROOT = Path(__file__).resolve().parents[1]
OUT_FEATURES = 3_072
IN_FEATURES = 1_024
EXPERTS = 256
ROWS = 512
TOP_K = 10
Q8_BLOCK_BYTES = 160
TOP10_EXPERTS = np.asarray(
    [17, 3, 91, 42, 7, 128, 201, 55, 240, 12],
    dtype=np.int64,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repacked-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--routing-json", type=Path, default=DEFAULT_ROUTING)
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--samples", type=int, default=11)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--prefill-burst", type=int, default=5)
    parser.add_argument("--decode-burst", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _cache_tiles(cache_root: Path, layer: int) -> tuple[np.ndarray, dict]:
    manifest = json.loads((cache_root / "manifest.json").read_text())
    entry = manifest["entries"][f"layers.{layer}.ffn_down_exps"]
    allocation = entry["allocations"]["tiles"]
    tiles = np.load(cache_root / allocation["file"], mmap_mode="r")
    expected = (EXPERTS, OUT_FEATURES // 16, IN_FEATURES // 256, 3_360)
    if entry["layout"] != "gguf_q6_k_t16_v1" or tuple(tiles.shape) != expected:
        raise ValueError("unexpected Laguna Q6T16 selected-down cache entry")
    return tiles, entry


def _read_bf16(runtime, buffer, shape: tuple[int, ...]) -> np.ndarray:
    result = np.empty(shape, dtype=np.uint16)
    copy_device_to_host(host_array_ptr(result), buffer, runtime=runtime)
    return result


def _counter_rotated(names: tuple[str, ...], sample: int) -> tuple[str, ...]:
    offset = sample % len(names)
    return names[offset:] + names[:offset]


def main() -> int:
    args = _parse_args()
    if min(
        args.samples,
        args.prefill_burst,
        args.decode_burst,
    ) <= 0 or args.warmups < 0:
        raise ValueError("samples/bursts must be positive and warmups non-negative")

    compiler_version = _compiler_version(args.compiler_version_file)
    prefill_library = build_gguf_q4_k_q8_1_selected_prefill(
        compiler_version=compiler_version,
        require_cached=args.require_cached_build,
    )
    decode_library = build_gguf_t16_selected_gemv(
        compiler_version=compiler_version,
        require_cached=args.require_cached_build,
    )
    runtime = get_hip_runtime()
    tracked_before = memory_stats()
    legacy_tiles, cache_entry = _cache_tiles(args.repacked_cache, args.layer)
    qmicro_tiles = convert_gguf_q6_k_tile16_to_qmicro(legacy_tiles).tiles
    routing = json.loads(args.routing_json.read_text())
    counts = _load_counts(routing, rows=ROWS, layer=args.layer)
    compact_starts = np.zeros(EXPERTS + 1, dtype=np.int64)
    compact_starts[1:] = np.cumsum(counts)
    padded_starts, tile_expert, padded_rows = _tile_metadata(counts, 64)
    compact_rows = int(compact_starts[-1])

    rng = np.random.default_rng(args.seed)
    prefill_x = _bf16_bits(
        rng.normal(0.0, 0.55, size=(compact_rows, IN_FEATURES)).astype(np.float32)
    )
    decode_x = _bf16_bits(
        rng.normal(0.0, 0.55, size=(1, IN_FEATURES)).astype(np.float32)
    )
    q8_nbytes = (
        compact_rows
        * (IN_FEATURES // 128)
        * Q8_BLOCK_BYTES
        * 3
    )
    prefill_out_nbytes = compact_rows * OUT_FEATURES * 2
    decode_out_nbytes = TOP_K * OUT_FEATURES * 2

    buffers = []
    samples = {
        "prefill_legacy": [],
        "prefill_qmicro": [],
        "prefill_qmicro_compact_activation": [],
        "decode_legacy": [],
        "decode_qmicro": [],
    }
    try:
        legacy_dev = _upload(runtime, legacy_tiles)
        qmicro_dev = _upload(runtime, qmicro_tiles)
        prefill_x_dev = _upload(runtime, prefill_x)
        decode_x_dev = _upload(runtime, decode_x)
        compact_starts_dev = _upload(runtime, compact_starts)
        padded_starts_dev = _upload(runtime, padded_starts)
        tile_expert_dev = _upload(runtime, tile_expert)
        selected_dev = _upload(runtime, TOP10_EXPERTS)
        q8_dev = malloc(q8_nbytes, runtime=runtime)
        prefill_legacy_out = malloc(prefill_out_nbytes, runtime=runtime)
        prefill_qmicro_out = malloc(prefill_out_nbytes, runtime=runtime)
        prefill_qmicro_compact_activation_out = malloc(
            prefill_out_nbytes,
            runtime=runtime,
        )
        decode_legacy_out = malloc(decode_out_nbytes, runtime=runtime)
        decode_qmicro_out = malloc(decode_out_nbytes, runtime=runtime)
        buffers.extend(
            (
                legacy_dev,
                qmicro_dev,
                prefill_x_dev,
                decode_x_dev,
                compact_starts_dev,
                padded_starts_dev,
                tile_expert_dev,
                selected_dev,
                q8_dev,
                prefill_legacy_out,
                prefill_qmicro_out,
                prefill_qmicro_compact_activation_out,
                decode_legacy_out,
                decode_qmicro_out,
            )
        )
        gguf_q8_1_mmq_ds4_f32_pack_bf16_d4x3(
            prefill_x_dev.ptr,
            q8_dev.ptr,
            compact_rows,
            IN_FEATURES,
            residual_passes=1,
            library=prefill_library,
            runtime=runtime,
        )

        def prefill(
            qmicro: bool,
            *,
            compact_activation: bool = False,
        ) -> None:
            gguf_q6_k_t16_selected_q8_1_ds4x3_f32_mmq64x32_prefill_compact32_bf16_bf16_out(
                q8_dev.ptr,
                compact_starts_dev.ptr,
                padded_starts_dev.ptr,
                tile_expert_dev.ptr,
                qmicro_dev.ptr if qmicro else legacy_dev.ptr,
                (
                    prefill_qmicro_compact_activation_out.ptr
                    if compact_activation
                    else prefill_qmicro_out.ptr
                    if qmicro
                    else prefill_legacy_out.ptr
                ),
                compact_rows,
                IN_FEATURES,
                OUT_FEATURES,
                EXPERTS,
                padded_rows,
                residual_passes=1,
                rowvec=True,
                tile_rows=64,
                qmicro=qmicro,
                compact_activation=compact_activation,
                library=prefill_library,
                runtime=runtime,
            )

        def decode(qmicro: bool) -> None:
            gguf_q6_k_t16_selected_gemv_bf16_bf16_out(
                decode_x_dev.ptr,
                selected_dev.ptr,
                qmicro_dev.ptr if qmicro else legacy_dev.ptr,
                decode_qmicro_out.ptr if qmicro else decode_legacy_out.ptr,
                1,
                TOP_K,
                EXPERTS,
                IN_FEATURES,
                OUT_FEATURES,
                qmicro=qmicro,
                library=decode_library,
                runtime=runtime,
            )

        launches = {
            "prefill_legacy": (lambda: prefill(False), args.prefill_burst),
            "prefill_qmicro": (lambda: prefill(True), args.prefill_burst),
            "prefill_qmicro_compact_activation": (
                lambda: prefill(True, compact_activation=True),
                args.prefill_burst,
            ),
            "decode_legacy": (lambda: decode(False), args.decode_burst),
            "decode_qmicro": (lambda: decode(True), args.decode_burst),
        }
        for _ in range(args.warmups):
            for launch, _ in launches.values():
                launch()
        runtime.device_synchronize()
        for sample in range(args.samples):
            for name in _counter_rotated(tuple(launches), sample):
                launch, burst = launches[name]
                samples[name].append(_event_ms(runtime, launch, burst=burst))

        prefill(False)
        prefill(True)
        prefill(True, compact_activation=True)
        decode(False)
        decode(True)
        runtime.device_synchronize()
        prefill_legacy_host = _read_bf16(
            runtime,
            prefill_legacy_out,
            (compact_rows, OUT_FEATURES),
        )
        prefill_qmicro_host = _read_bf16(
            runtime,
            prefill_qmicro_out,
            (compact_rows, OUT_FEATURES),
        )
        prefill_qmicro_compact_activation_host = _read_bf16(
            runtime,
            prefill_qmicro_compact_activation_out,
            (compact_rows, OUT_FEATURES),
        )
        decode_legacy_host = _read_bf16(
            runtime,
            decode_legacy_out,
            (TOP_K, OUT_FEATURES),
        )
        decode_qmicro_host = _read_bf16(
            runtime,
            decode_qmicro_out,
            (TOP_K, OUT_FEATURES),
        )
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)

    tracked_after = memory_stats()
    medians = {
        name: statistics.median(values)
        for name, values in samples.items()
    }
    prefill_mismatches = int(
        np.count_nonzero(prefill_qmicro_host != prefill_legacy_host)
    )
    prefill_compact_activation_mismatches = int(
        np.count_nonzero(
            prefill_qmicro_compact_activation_host != prefill_qmicro_host
        )
    )
    decode_mismatches = int(
        np.count_nonzero(decode_qmicro_host != decode_legacy_host)
    )
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "kind": "hipengine_laguna_q6_t16_qmicro_actual_leaf",
        "repo_revision": subprocess.check_output(
            ("git", "rev-parse", "HEAD"),
            cwd=ROOT,
            text=True,
        ).strip(),
        "tracked_changes": subprocess.check_output(
            ("git", "status", "--short", "--untracked-files=no"),
            cwd=ROOT,
            text=True,
        ).splitlines(),
        "hardware": "AMD Radeon 8060S Graphics / gfx1151",
        "queue_policy": "GPU_MAX_HW_QUEUES=1",
        "cache_entry": cache_entry,
        "shape": {
            "layer": args.layer,
            "weights": list(legacy_tiles.shape),
            "weight_nbytes": int(legacy_tiles.nbytes),
            "natural_prompt_rows": ROWS,
            "compact_rows": compact_rows,
            "padded_rows64": padded_rows,
            "decode_selected_experts": TOP10_EXPERTS.tolist(),
        },
        "protocol": {
            "samples": args.samples,
            "warmups": args.warmups,
            "prefill_burst": args.prefill_burst,
            "decode_burst": args.decode_burst,
            "order": "counter-rotated over five modes",
            "timing": "HIP events; prefill activation pack excluded",
        },
        "samples_ms": samples,
        "medians_ms": medians,
        "change_percent": {
            "prefill": (
                medians["prefill_qmicro"] / medians["prefill_legacy"] - 1.0
            )
            * 100.0,
            "prefill_compact_activation": (
                medians["prefill_qmicro_compact_activation"]
                / medians["prefill_qmicro"]
                - 1.0
            )
            * 100.0,
            "decode": (
                medians["decode_qmicro"] / medians["decode_legacy"] - 1.0
            )
            * 100.0,
        },
        "correctness": {
            "prefill_bf16_mismatches": prefill_mismatches,
            "prefill_compact_activation_bf16_mismatches": (
                prefill_compact_activation_mismatches
            ),
            "decode_bf16_mismatches": decode_mismatches,
            "prefill_checksum": int(prefill_qmicro_host.sum(dtype=np.uint64)),
            "decode_checksum": int(decode_qmicro_host.sum(dtype=np.uint64)),
        },
        "memory": {
            "tracked_before": tracked_before,
            "tracked_after": tracked_after,
        },
        "pass": bool(
            prefill_mismatches == 0
            and prefill_compact_activation_mismatches == 0
            and decode_mismatches == 0
            and medians["prefill_qmicro"] < medians["prefill_legacy"]
            and medians["prefill_qmicro_compact_activation"]
            < medians["prefill_qmicro"]
            and medians["decode_qmicro"] < medians["decode_legacy"]
            and tracked_before["current_allocated_bytes"]
            == tracked_after["current_allocated_bytes"]
            and tracked_before["active_allocations"]
            == tracked_after["active_allocations"]
        ),
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0 if payload["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
