#!/usr/bin/env python3
"""Microbench GGUF Q4_K selected-dual prefill kernel variants.

This diagnostic harness was created for the llama.cpp MMQ/Q8_1 prefill detour.
It builds a synthetic compact-selected MoE fixture and can time either:

* ``selected-wmma``: the current hipEngine Q4T16 selected-dual WMMA prefill.
* ``q8-1-dot``: a standalone raw-Q4_K x prequantized-Q8_1 integer-dot
  prototype.  This is deliberately simple and does not include activation
  quantization time in the measured loop.
* ``q8-1-ds4-dot``: the same scalar dot prototype fed by llama.cpp-style
  DS4 ``block_q8_1_mmq`` activation blocks, isolating layout effects before a
  tiled WMMA/MMQ port.
* ``q8-1-ds4-wmma``: a first wave32 integer-WMMA prototype over the DS4 layout
  and raw Q4_K nibbles. It validates the MMA decomposition before a wider
  shared-memory tiled MMQ port.
* ``q8-1-ds4-wmma32``: the same integer-WMMA math with two independent 16-column
  waves per block, reducing block-count overhead while preserving the one-wave
  fragment mapping.
* ``q8-1-ds4-wmma32-pack``: the WMMA32 prototype with the BF16->DS4 Q8_1 GPU
  activation pack kernel included in the timed loop.
* ``q8-1-ds4-t16-wmma32``: the two-wave integer-WMMA math consuming the resident
  Q4_K T16 tile layout instead of raw GGUF Q4_K weights, testing whether the
  no-raw-duplicate runtime layout preserves the DS4 headroom.
* ``q8-1-ds4-t16-wmma32-pack``: the same resident-Q4T16 prototype with the
  BF16->DS4 Q8_1 GPU activation pack kernel included in the timed loop.
* ``q8-1-ds4-wmma64``: a four-wave/64-column raw-Q4_K integer-WMMA diagnostic
  that tests whether larger output-column tiles reduce block scheduling overhead.
* ``q8-1-ds4-preview-wmma32``: the two-wave integer-WMMA math fed by a
  pre-unpacked host-side Q4_K MMQ preview layout (q4 nibbles plus FP32 scale/min
  terms), testing whether raw Q4_K metadata decode is the remaining bottleneck.
* ``q8-1-ds4-wmma32-ldspack``: the two-wave variant with the packed Q4_K qs
  payload plus scale/min terms staged into LDS before the two half-waves consume
  it.
* ``q8-1-ds4-wmma32-lds``: the two-wave variant with each Q4_K column tile
  unpacked once into LDS before the two half-waves consume it.

The script does not load a full GGUF model and does not validate model quality;
it is a same-shape kernel-design baseline/prototype harness.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from hipengine.quant.gguf_q4_k import pack_gguf_q4_k_mmq_tile16_preview, pack_q8_1_mmq_ds4_from_bf16


_GIB = 1 << 30
_Q8_1_BLOCK = 32


def _f32_to_bf16_u16(arr: np.ndarray) -> np.ndarray:
    f32 = np.ascontiguousarray(arr, dtype=np.float32)
    u32 = f32.view(np.uint32).copy()
    lsb = (u32 >> 16) & 1
    return (((u32 + 0x7FFF + lsb) >> 16).astype(np.uint16)).reshape(f32.shape)


def _bf16_u16_to_f32(arr: np.ndarray) -> np.ndarray:
    u16 = np.ascontiguousarray(arr, dtype=np.uint16)
    return (u16.astype(np.uint32) << 16).view(np.float32).reshape(u16.shape).copy()


def _quantize_q8_1_blocks(x_bf16: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return Q8_1-style ``(qs, d, sum)`` for BF16 activation rows.

    ``qs`` has shape ``[rows, hidden / 32, 32]`` and dtype ``int8``.  ``d`` and
    ``sum`` are float32 arrays with shape ``[rows, hidden / 32]``.  ``sum`` is
    the dequantized block sum ``d * sum(qs)`` used by Q4_K's min term.
    """

    x = _bf16_u16_to_f32(x_bf16).astype(np.float32, copy=False)
    if x.shape[-1] % _Q8_1_BLOCK:
        raise ValueError("hidden dimension must be divisible by 32 for Q8_1")
    blocks = x.reshape(x.shape[0], x.shape[1] // _Q8_1_BLOCK, _Q8_1_BLOCK)
    max_abs = np.max(np.abs(blocks), axis=-1)
    d = (max_abs / 127.0).astype(np.float32)
    safe_d = np.where(d > 0.0, d, 1.0).astype(np.float32)
    qs = np.rint(blocks / safe_d[..., None]).clip(-127, 127).astype(np.int8)
    qs = np.where(d[..., None] > 0.0, qs, np.zeros_like(qs)).astype(np.int8, copy=False)
    sums = (qs.astype(np.float32).sum(axis=-1) * d).astype(np.float32)
    return np.ascontiguousarray(qs), np.ascontiguousarray(d), np.ascontiguousarray(sums)


def _make_activation(rows: int, hidden: int, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Keep magnitudes modest so output finiteness is a useful sanity check and
    # the kernel is not measuring overflow behavior.
    return _f32_to_bf16_u16((rng.standard_normal((rows, hidden)) * 0.02).astype(np.float32))


def _make_uniform_compact_metadata(
    experts: int, rows_per_expert: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    counts = np.full(experts, rows_per_expert, dtype=np.int64)
    expert_start_compact = np.zeros(experts + 1, dtype=np.int64)
    expert_start_compact[1:] = np.cumsum(counts)

    padded_counts = ((counts + 15) // 16) * 16
    expert_start_wmma = np.zeros(experts + 1, dtype=np.int64)
    expert_start_wmma[1:] = np.cumsum(padded_counts)
    tile_expert = np.asarray(
        [expert for expert, padded in enumerate(padded_counts) for _ in range(int(padded) // 16)],
        dtype=np.int64,
    )
    compact_rows = int(expert_start_compact[-1])
    wmma_total_rows = int(expert_start_wmma[-1])
    return expert_start_compact, expert_start_wmma, tile_expert, compact_rows, wmma_total_rows


def _copy_to_device(arr: np.ndarray, *, runtime: Any):
    from hipengine.core.memory import copy_host_to_device, host_array_ptr, malloc

    contiguous = np.ascontiguousarray(arr)
    dev = malloc(contiguous.nbytes, runtime=runtime)
    copy_host_to_device(dev, host_array_ptr(contiguous), runtime=runtime)
    return dev, contiguous


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def _git_dirty() -> list[str]:
    try:
        out = subprocess.check_output(["git", "status", "--porcelain"], text=True)
    except Exception:
        return []
    return [line for line in out.splitlines() if line]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument(
        "--mode",
        choices=(
            "selected-wmma",
            "q8-1-dot",
            "q8-1-ds4-dot",
            "q8-1-ds4-wmma",
            "q8-1-ds4-wmma32",
            "q8-1-ds4-wmma32-pack",
            "q8-1-ds4-t16-wmma32",
            "q8-1-ds4-t16-wmma32-pack",
            "q8-1-ds4-wmma64",
            "q8-1-ds4-preview-wmma32",
            "q8-1-ds4-wmma32-ldspack",
            "q8-1-ds4-wmma32-lds",
        ),
        default="selected-wmma",
    )
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--out-features-a", type=int, default=4096)
    parser.add_argument("--out-features-b", type=int, default=4096)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--rows-per-expert", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=25)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--json", type=Path, help="Optional output JSON path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.compiler_version_file is not None:
        os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(args.compiler_version_file)
    if args.hidden % 256:
        raise SystemExit("--hidden must be divisible by 256 for Q4_K")
    if args.out_features_a % 16 or args.out_features_b % 16:
        raise SystemExit("--out-features-a/b must be divisible by 16")
    if args.experts <= 0 or args.rows_per_expert <= 0:
        raise SystemExit("--experts and --rows-per-expert must be positive")
    if args.iters <= 0 or args.warmup < 0:
        raise SystemExit("--iters must be positive and --warmup non-negative")

    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import copy_device_to_host, free, host_array_ptr, malloc, memory_stats, reset_memory_stats
    from tests._gguf_synthetic_weights import make_q4_k_weight

    runtime = get_hip_runtime()
    reset_memory_stats()

    expert_start_compact, expert_start_wmma, tile_expert, compact_rows, wmma_total_rows = _make_uniform_compact_metadata(
        args.experts, args.rows_per_expert
    )
    x_host = _make_activation(compact_rows, args.hidden, seed=args.seed)

    base_a = make_q4_k_weight(args.out_features_a, args.hidden)
    base_b = make_q4_k_weight(args.out_features_b, args.hidden)
    qweight_a = np.ascontiguousarray(
        np.stack([np.roll(base_a, shift=expert, axis=0) for expert in range(args.experts)], axis=0)
    )
    qweight_b = np.ascontiguousarray(
        np.stack([np.roll(base_b, shift=expert + 3, axis=0) for expert in range(args.experts)], axis=0)
    )
    out_host = np.zeros((compact_rows, args.out_features_a + args.out_features_b), dtype=np.uint16)

    bufs = []
    stream = runtime.stream_create()
    variant_extra: dict[str, Any] = {}
    launch: Callable[[], None]
    try:
        start_compact_dev, _ = _copy_to_device(expert_start_compact, runtime=runtime)
        start_wmma_dev, _ = _copy_to_device(expert_start_wmma, runtime=runtime)
        tile_expert_dev, _ = _copy_to_device(tile_expert, runtime=runtime)
        bufs.extend((start_compact_dev, start_wmma_dev, tile_expert_dev))

        if args.mode in {"selected-wmma", "q8-1-ds4-t16-wmma32", "q8-1-ds4-t16-wmma32-pack"}:
            from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_t16_selected_prefill import (
                build_gguf_q4_k_t16_selected_prefill,
                gguf_q4_k_t16_selected_dual_q8_1_ds4_wmma32_prefill_compact32_bf16_bf16_out,
                gguf_q4_k_t16_selected_dual_wmma_prefill_compact32_bf16_bf16_out,
            )
            from hipengine.quant.gguf_q4_k import repack_gguf_q4_k_tile16

            library = build_gguf_q4_k_t16_selected_prefill(
                load=True,
                require_cached=args.require_cached_build,
            )
            tiles_a = repack_gguf_q4_k_tile16(qweight_a).tiles
            tiles_b = repack_gguf_q4_k_tile16(qweight_b).tiles
            tiles_a_dev, _ = _copy_to_device(tiles_a, runtime=runtime)
            tiles_b_dev, _ = _copy_to_device(tiles_b, runtime=runtime)
            out_dev = malloc(out_host.nbytes, runtime=runtime)
            bufs.extend((tiles_a_dev, tiles_b_dev, out_dev))

            if args.mode in {"q8-1-ds4-t16-wmma32", "q8-1-ds4-t16-wmma32-pack"}:
                pack_in_loop = args.mode == "q8-1-ds4-t16-wmma32-pack"
                if pack_in_loop:
                    from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_q8_1_selected_prefill import (
                        build_gguf_q4_k_q8_1_selected_prefill,
                        gguf_q8_1_mmq_ds4_pack_bf16,
                    )

                    pack_library = build_gguf_q4_k_q8_1_selected_prefill(
                        load=True,
                        require_cached=args.require_cached_build,
                    )
                    q8_ds4 = np.empty((compact_rows, args.hidden // 128, 144), dtype=np.uint8)
                    q8_ds4_dev = malloc(q8_ds4.nbytes, runtime=runtime)
                    x_dev, _ = _copy_to_device(x_host, runtime=runtime)
                    bufs.extend((q8_ds4_dev, x_dev))
                else:
                    q8_ds4 = pack_q8_1_mmq_ds4_from_bf16(x_host)
                    q8_ds4_dev, _ = _copy_to_device(q8_ds4, runtime=runtime)
                    x_dev = None
                    pack_library = None
                    bufs.append(q8_ds4_dev)
                variant_extra = {
                    "host_q8_ds4_mib": q8_ds4.nbytes / (1 << 20),
                    "host_input_mib": x_host.nbytes / (1 << 20),
                    "host_tiles_a_mib": tiles_a.nbytes / (1 << 20),
                    "host_tiles_b_mib": tiles_b.nbytes / (1 << 20),
                    "activation_quantization_in_loop": pack_in_loop,
                    "activation_layout": "llama_cpp_block_q8_1_mmq_ds4",
                    "weight_layout": "gguf_q4_k_t16_v1_resident",
                    "integer_mma": True,
                    "prototype_note": (
                        "DS4 activation layout with GPU BF16->Q8_1 pack plus wave32 integer-WMMA32 dot tiles over resident Q4_K T16 repack tiles in the timed loop; diagnostic no-raw-duplicate runtime-viability probe."
                        if pack_in_loop
                        else "DS4 activation layout with wave32 integer-WMMA32 dot tiles over resident Q4_K T16 repack tiles; diagnostic no-raw-duplicate layout probe."
                    ),
                }

                def launch() -> None:
                    if pack_in_loop:
                        assert x_dev is not None and pack_library is not None
                        gguf_q8_1_mmq_ds4_pack_bf16(
                            x_dev.ptr,
                            q8_ds4_dev.ptr,
                            compact_rows,
                            args.hidden,
                            stream=stream,
                            library=pack_library,
                            runtime=runtime,
                        )
                    gguf_q4_k_t16_selected_dual_q8_1_ds4_wmma32_prefill_compact32_bf16_bf16_out(
                        q8_ds4_dev.ptr,
                        start_compact_dev.ptr,
                        start_wmma_dev.ptr,
                        tile_expert_dev.ptr,
                        tiles_a_dev.ptr,
                        tiles_b_dev.ptr,
                        out_dev.ptr,
                        compact_rows,
                        args.hidden,
                        args.out_features_a,
                        args.out_features_b,
                        args.experts,
                        wmma_total_rows,
                        stream=stream,
                        library=library,
                        runtime=runtime,
                    )

            else:
                input_dev, _ = _copy_to_device(x_host, runtime=runtime)
                bufs.append(input_dev)
                variant_extra = {
                    "host_input_mib": x_host.nbytes / (1 << 20),
                    "host_tiles_a_mib": tiles_a.nbytes / (1 << 20),
                    "host_tiles_b_mib": tiles_b.nbytes / (1 << 20),
                    "activation_quantization_in_loop": False,
                    "weight_layout": "gguf_q4_k_t16_v1",
                }

                def launch() -> None:
                    gguf_q4_k_t16_selected_dual_wmma_prefill_compact32_bf16_bf16_out(
                        input_dev.ptr,
                        start_compact_dev.ptr,
                        start_wmma_dev.ptr,
                        tile_expert_dev.ptr,
                        tiles_a_dev.ptr,
                        tiles_b_dev.ptr,
                        out_dev.ptr,
                        compact_rows,
                        args.hidden,
                        args.out_features_a,
                        args.out_features_b,
                        args.experts,
                        wmma_total_rows,
                        stream=stream,
                        library=library,
                        runtime=runtime,
                    )

        else:
            from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_q8_1_selected_prefill import (
                build_gguf_q4_k_q8_1_selected_prefill,
                gguf_q4_k_selected_dual_q8_1_ds4_prefill_compact32_bf16_bf16_out,
                gguf_q4_k_selected_dual_q8_1_ds4_preview_wmma32_prefill_compact32_bf16_bf16_out,
                gguf_q4_k_selected_dual_q8_1_ds4_wmma_prefill_compact32_bf16_bf16_out,
                gguf_q4_k_selected_dual_q8_1_ds4_wmma32_lds_prefill_compact32_bf16_bf16_out,
                gguf_q4_k_selected_dual_q8_1_ds4_wmma32_ldspack_prefill_compact32_bf16_bf16_out,
                gguf_q4_k_selected_dual_q8_1_ds4_wmma32_prefill_compact32_bf16_bf16_out,
                gguf_q4_k_selected_dual_q8_1_ds4_wmma64_prefill_compact32_bf16_bf16_out,
                gguf_q4_k_selected_dual_q8_1_prefill_compact32_bf16_bf16_out,
                gguf_q8_1_mmq_ds4_pack_bf16,
            )

            library = build_gguf_q4_k_q8_1_selected_prefill(
                load=True,
                require_cached=args.require_cached_build,
            )
            qweight_a_dev = None
            qweight_b_dev = None
            out_dev = malloc(out_host.nbytes, runtime=runtime)
            bufs.append(out_dev)
            if args.mode != "q8-1-ds4-preview-wmma32":
                qweight_a_dev, _ = _copy_to_device(qweight_a, runtime=runtime)
                qweight_b_dev, _ = _copy_to_device(qweight_b, runtime=runtime)
                bufs.extend((qweight_a_dev, qweight_b_dev))

            if args.mode == "q8-1-dot":
                q8_qs, q8_d, q8_sum = _quantize_q8_1_blocks(x_host)
                q8_qs_dev, _ = _copy_to_device(q8_qs, runtime=runtime)
                q8_d_dev, _ = _copy_to_device(q8_d, runtime=runtime)
                q8_sum_dev, _ = _copy_to_device(q8_sum, runtime=runtime)
                bufs.extend((q8_qs_dev, q8_d_dev, q8_sum_dev))
                variant_extra = {
                    "host_q8_qs_mib": q8_qs.nbytes / (1 << 20),
                    "host_q8_scale_mib": q8_d.nbytes / (1 << 20),
                    "host_q8_sum_mib": q8_sum.nbytes / (1 << 20),
                    "host_raw_qweight_a_mib": qweight_a.nbytes / (1 << 20),
                    "host_raw_qweight_b_mib": qweight_b.nbytes / (1 << 20),
                    "activation_quantization_in_loop": False,
                    "activation_layout": "separate_qs_f32_scale_f32_sum",
                    "weight_layout": "raw_gguf_q4_k",
                    "prototype_note": "Prequantized activation scalar integer-dot prototype; not a tiled MMQ implementation.",
                }

                def launch() -> None:
                    gguf_q4_k_selected_dual_q8_1_prefill_compact32_bf16_bf16_out(
                        q8_qs_dev.ptr,
                        q8_d_dev.ptr,
                        q8_sum_dev.ptr,
                        start_compact_dev.ptr,
                        start_wmma_dev.ptr,
                        tile_expert_dev.ptr,
                        qweight_a_dev.ptr,
                        qweight_b_dev.ptr,
                        out_dev.ptr,
                        compact_rows,
                        args.hidden,
                        args.out_features_a,
                        args.out_features_b,
                        args.experts,
                        wmma_total_rows,
                        stream=stream,
                        library=library,
                        runtime=runtime,
                    )

            else:
                pack_in_loop = args.mode == "q8-1-ds4-wmma32-pack"
                if pack_in_loop:
                    q8_ds4 = np.empty((compact_rows, args.hidden // 128, 144), dtype=np.uint8)
                    q8_ds4_dev = malloc(q8_ds4.nbytes, runtime=runtime)
                    x_dev, _ = _copy_to_device(x_host, runtime=runtime)
                    bufs.extend((q8_ds4_dev, x_dev))
                else:
                    q8_ds4 = pack_q8_1_mmq_ds4_from_bf16(x_host)
                    q8_ds4_dev, _ = _copy_to_device(q8_ds4, runtime=runtime)
                    x_dev = None
                    bufs.append(q8_ds4_dev)
                if args.mode == "q8-1-ds4-preview-wmma32":
                    previews_a = [pack_gguf_q4_k_mmq_tile16_preview(qweight_a[expert]) for expert in range(args.experts)]
                    previews_b = [pack_gguf_q4_k_mmq_tile16_preview(qweight_b[expert]) for expert in range(args.experts)]
                    q4_a = np.ascontiguousarray(np.stack([preview.q4 for preview in previews_a], axis=0))
                    scale_a = np.ascontiguousarray(np.stack([preview.scales for preview in previews_a], axis=0), dtype=np.float32)
                    min_a = np.ascontiguousarray(np.stack([preview.mins for preview in previews_a], axis=0), dtype=np.float32)
                    q4_b = np.ascontiguousarray(np.stack([preview.q4 for preview in previews_b], axis=0))
                    scale_b = np.ascontiguousarray(np.stack([preview.scales for preview in previews_b], axis=0), dtype=np.float32)
                    min_b = np.ascontiguousarray(np.stack([preview.mins for preview in previews_b], axis=0), dtype=np.float32)
                    q4_a_dev, _ = _copy_to_device(q4_a, runtime=runtime)
                    scale_a_dev, _ = _copy_to_device(scale_a, runtime=runtime)
                    min_a_dev, _ = _copy_to_device(min_a, runtime=runtime)
                    q4_b_dev, _ = _copy_to_device(q4_b, runtime=runtime)
                    scale_b_dev, _ = _copy_to_device(scale_b, runtime=runtime)
                    min_b_dev, _ = _copy_to_device(min_b, runtime=runtime)
                    bufs.extend((q4_a_dev, scale_a_dev, min_a_dev, q4_b_dev, scale_b_dev, min_b_dev))
                    variant_extra = {
                        "host_q8_ds4_mib": q8_ds4.nbytes / (1 << 20),
                        "host_preview_q4_a_mib": q4_a.nbytes / (1 << 20),
                        "host_preview_scale_a_mib": scale_a.nbytes / (1 << 20),
                        "host_preview_min_a_mib": min_a.nbytes / (1 << 20),
                        "host_preview_q4_b_mib": q4_b.nbytes / (1 << 20),
                        "host_preview_scale_b_mib": scale_b.nbytes / (1 << 20),
                        "host_preview_min_b_mib": min_b.nbytes / (1 << 20),
                        "activation_quantization_in_loop": False,
                        "activation_layout": "llama_cpp_block_q8_1_mmq_ds4",
                        "weight_layout": "gguf_q4_k_mmq_tile16_preview_unpacked_q4_f32_scale_min",
                        "integer_mma": True,
                        "prototype_note": "DS4 activation layout with wave32 integer-WMMA dot tiles over pre-unpacked Q4_K preview q4/scales/mins; diagnostic replacement-layout probe only.",
                    }

                    def launch() -> None:
                        gguf_q4_k_selected_dual_q8_1_ds4_preview_wmma32_prefill_compact32_bf16_bf16_out(
                            q8_ds4_dev.ptr,
                            start_compact_dev.ptr,
                            start_wmma_dev.ptr,
                            tile_expert_dev.ptr,
                            q4_a_dev.ptr,
                            scale_a_dev.ptr,
                            min_a_dev.ptr,
                            q4_b_dev.ptr,
                            scale_b_dev.ptr,
                            min_b_dev.ptr,
                            out_dev.ptr,
                            compact_rows,
                            args.hidden,
                            args.out_features_a,
                            args.out_features_b,
                            args.experts,
                            wmma_total_rows,
                            stream=stream,
                            library=library,
                            runtime=runtime,
                        )
                else:
                    use_wmma = args.mode in {
                        "q8-1-ds4-wmma",
                        "q8-1-ds4-wmma32",
                        "q8-1-ds4-wmma32-pack",
                        "q8-1-ds4-wmma64",
                        "q8-1-ds4-wmma32-ldspack",
                        "q8-1-ds4-wmma32-lds",
                    }
                    if args.mode == "q8-1-ds4-wmma32-lds":
                        ds4_launcher = gguf_q4_k_selected_dual_q8_1_ds4_wmma32_lds_prefill_compact32_bf16_bf16_out
                    elif args.mode == "q8-1-ds4-wmma32-ldspack":
                        ds4_launcher = gguf_q4_k_selected_dual_q8_1_ds4_wmma32_ldspack_prefill_compact32_bf16_bf16_out
                    elif args.mode == "q8-1-ds4-wmma64":
                        ds4_launcher = gguf_q4_k_selected_dual_q8_1_ds4_wmma64_prefill_compact32_bf16_bf16_out
                    elif args.mode in {"q8-1-ds4-wmma32", "q8-1-ds4-wmma32-pack"}:
                        ds4_launcher = gguf_q4_k_selected_dual_q8_1_ds4_wmma32_prefill_compact32_bf16_bf16_out
                    elif args.mode == "q8-1-ds4-wmma":
                        ds4_launcher = gguf_q4_k_selected_dual_q8_1_ds4_wmma_prefill_compact32_bf16_bf16_out
                    else:
                        ds4_launcher = gguf_q4_k_selected_dual_q8_1_ds4_prefill_compact32_bf16_bf16_out
                    variant_extra = {
                        "host_q8_ds4_mib": q8_ds4.nbytes / (1 << 20),
                        "host_input_mib": x_host.nbytes / (1 << 20),
                        "host_raw_qweight_a_mib": qweight_a.nbytes / (1 << 20),
                        "host_raw_qweight_b_mib": qweight_b.nbytes / (1 << 20),
                        "activation_quantization_in_loop": pack_in_loop,
                        "activation_layout": "llama_cpp_block_q8_1_mmq_ds4",
                        "weight_layout": "raw_gguf_q4_k",
                        "integer_mma": use_wmma,
                        "prototype_note": (
                            "DS4 activation layout with GPU BF16->Q8_1 pack plus wave32 integer-WMMA32 dot tiles in the timed loop; diagnostic runtime-viability probe only."
                            if pack_in_loop
                            else (
                                "DS4 activation layout with wave32 integer-WMMA dot tiles and per-block LDS-staged packed Q4_K qs bytes; still a diagnostic microbench, not a full runtime MMQ integration."
                                if args.mode == "q8-1-ds4-wmma32-ldspack"
                                else (
                                    "DS4 activation layout with wave32 integer-WMMA dot tiles and per-block LDS-staged Q4_K columns; still a diagnostic microbench, not a full runtime MMQ integration."
                                    if args.mode == "q8-1-ds4-wmma32-lds"
                                    else (
                                        "DS4 activation layout with wave32 integer-WMMA dot tiles; still uses raw Q4_K global loads rather than the full shared-memory MMQ tile."
                                        if use_wmma
                                        else "DS4 activation layout with scalar integer-dot inner loop; still not a tiled WMMA/MMQ implementation."
                                    )
                                )
                            )
                        ),
                    }

                    def launch() -> None:
                        assert qweight_a_dev is not None and qweight_b_dev is not None
                        if pack_in_loop:
                            assert x_dev is not None
                            gguf_q8_1_mmq_ds4_pack_bf16(
                                x_dev.ptr,
                                q8_ds4_dev.ptr,
                                compact_rows,
                                args.hidden,
                                stream=stream,
                                library=library,
                                runtime=runtime,
                            )
                        ds4_launcher(
                            q8_ds4_dev.ptr,
                            start_compact_dev.ptr,
                            start_wmma_dev.ptr,
                            tile_expert_dev.ptr,
                            qweight_a_dev.ptr,
                            qweight_b_dev.ptr,
                            out_dev.ptr,
                            compact_rows,
                            args.hidden,
                            args.out_features_a,
                            args.out_features_b,
                            args.experts,
                            wmma_total_rows,
                            stream=stream,
                            library=library,
                            runtime=runtime,
                        )

        for _ in range(args.warmup):
            launch()
        runtime.stream_synchronize(stream)
        start = time.perf_counter()
        for _ in range(args.iters):
            launch()
        runtime.stream_synchronize(stream)
        elapsed_s = time.perf_counter() - start
        ms_per_call = elapsed_s * 1e3 / args.iters

        launch()
        runtime.stream_synchronize(stream)
        copy_device_to_host(host_array_ptr(out_host), out_dev, runtime=runtime)
        out_f32 = _bf16_u16_to_f32(out_host)
        finite = bool(np.isfinite(out_f32).all())
        checksum = float(out_f32.astype(np.float64).sum())
        max_abs = float(np.max(np.abs(out_f32))) if out_f32.size else 0.0

        out_features_total = args.out_features_a + args.out_features_b
        logical_fma = int(compact_rows * out_features_total * args.hidden)
        logical_tflops = (2.0 * logical_fma) / (ms_per_call / 1e3) / 1e12
        result: dict[str, Any] = {
            "schema": 2,
            "status": "diagnostic_retained",
            "performance_claim": False,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_tag": f"gguf_q4_k_selected_prefill_microbench_{args.mode}",
            "kernel_mode": args.mode,
            "reason_not_promoted": "Synthetic kernel microbench/prototype only; no runtime dispatch/default change.",
            "software": {
                "hipengine_commit": _git_commit(),
                "hipengine_dirty_files": _git_dirty(),
                "compiler_version_file": str(args.compiler_version_file) if args.compiler_version_file else None,
                "python": sys.version.split()[0],
            },
            "shape": {
                "hidden": args.hidden,
                "out_features_a": args.out_features_a,
                "out_features_b": args.out_features_b,
                "out_features_total": out_features_total,
                "experts": args.experts,
                "rows_per_expert": args.rows_per_expert,
                "compact_rows": compact_rows,
                "wmma_total_rows": wmma_total_rows,
                "topology_note": "Uniform selected rows across a reduced expert set; use to compare kernel designs, not as full-model routing evidence.",
            },
            "timing": {
                "warmup": args.warmup,
                "iters": args.iters,
                "elapsed_s": elapsed_s,
                "ms_per_call": ms_per_call,
                "calls_per_s": 1000.0 / ms_per_call,
                "logical_fma": logical_fma,
                "logical_tflops": logical_tflops,
            },
            "memory": {
                **variant_extra,
                "host_output_mib": out_host.nbytes / (1 << 20),
                "tracked_peak_allocated_gib": memory_stats()["peak_allocated_bytes"] / _GIB,
            },
            "sanity": {
                "finite_output": finite,
                "output_checksum_f64": checksum,
                "output_max_abs": max_abs,
            },
        }
    finally:
        for buf in reversed(bufs):
            free(buf, runtime=runtime)
        runtime.stream_destroy(stream)

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["sanity"]["finite_output"]:
        raise SystemExit("kernel output contained non-finite values")


if __name__ == "__main__":
    main()
