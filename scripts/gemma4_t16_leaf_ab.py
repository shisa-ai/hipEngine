#!/usr/bin/env python3
"""A/B the Gemma 4 Q4T16 expert gate/up prefill leaf against its successors.

The production leaf is ``..._compact32_bf16_bf16_out``.  This probe drives it and
any candidate exported from the same library on identical inputs, checks that
every candidate's bf16 output is bit-identical to the production kernel's, and
times each one.  It exists because the in-situ census costs a 16 second model
load per iteration, while the only thing a leaf rewrite can change is the leaf.

Two expert-row distributions are offered, because the candidates pair adjacent
16-row tiles and so care whether a pair lands inside one expert:

``--distribution uniform``
    ``--rows-per-expert`` rows in every expert.  This is the fixture the DRAM
    counter unit used, and it is the easy case for a pairing kernel.

``--distribution multinomial``
    ``compact_rows`` rows drawn multinomially over the experts, which is what the
    real prefill produces: 4096 compact rows over 128 experts averages 32 rows
    per expert and pads to about 300 16-row tiles rather than 256.

Diagnostic only: no perf claim, no production path, and nothing registered.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np  # noqa: E402

from hipengine.core.hip import get_hip_runtime  # noqa: E402
from hipengine.core.memory import (  # noqa: E402
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.loading.gguf import GGUFReader  # noqa: E402

DEFAULT_ARTIFACT = (
    "/models/gguf/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf"
)

_TILE_ROWS = 16
_TILE_BYTES = 2368
_Q4_K_BLOCK_BYTES = 144
_QK_K = 256
_BLOCK_COLS = 32

# Variant name -> exported symbol. "baseline" is the production kernel as built
# from ``--baseline-library`` (or from this build when that flag is absent) and
# is the bit-identity reference.
_VARIANTS = {
    "baseline": "hipengine_gguf_q4_k_t16_selected_dual_wmma_prefill_compact32_bf16_bf16_out",
    "candidate": "hipengine_gguf_q4_k_t16_selected_dual_wmma_prefill_compact32_bf16_bf16_out",
    "shared_x": (
        "hipengine_gguf_q4_k_t16_selected_dual_wmma_prefill_compact32_shared_x_bf16_bf16_out"
    ),
    "baseline_fp16": "hipengine_gguf_q4_k_t16_selected_dual_wmma_prefill_compact32_fp16_fp16_out",
    "candidate_fp16": "hipengine_gguf_q4_k_t16_selected_dual_wmma_prefill_compact32_fp16_fp16_out",
}


def _f32_to_bf16_u16(arr: np.ndarray) -> np.ndarray:
    f32 = np.ascontiguousarray(arr, dtype=np.float32)
    u32 = f32.view(np.uint32).copy()
    lsb = (u32 >> 16) & 1
    return ((u32 + 0x7FFF + lsb) >> 16).astype(np.uint16).reshape(f32.shape)


def _metadata(counts: np.ndarray):
    """Compact-selected metadata for a per-expert row count vector."""

    counts = np.asarray(counts, dtype=np.int64)
    experts = counts.size
    start_compact = np.zeros(experts + 1, dtype=np.int64)
    start_compact[1:] = np.cumsum(counts)
    padded = ((counts + _TILE_ROWS - 1) // _TILE_ROWS) * _TILE_ROWS
    start_wmma = np.zeros(experts + 1, dtype=np.int64)
    start_wmma[1:] = np.cumsum(padded)
    tile_expert = np.asarray(
        [
            expert
            for expert, rows in enumerate(padded)
            for _ in range(int(rows) // _TILE_ROWS)
        ],
        dtype=np.int64,
    )
    return (
        start_compact,
        start_wmma,
        tile_expert,
        int(start_compact[-1]),
        int(start_wmma[-1]),
    )


def _to_device(arr: np.ndarray, runtime):
    contiguous = np.ascontiguousarray(arr)
    dev = malloc(contiguous.nbytes, runtime=runtime)
    copy_host_to_device(dev, host_array_ptr(contiguous), runtime=runtime)
    return dev


def _timer(runtime, fn, *, warmup: int, iters: int) -> list[float]:
    stream = 0
    for _ in range(warmup):
        fn()
    runtime.stream_synchronize(stream)
    start = runtime.event_create()
    stop = runtime.event_create()
    times: list[float] = []
    for _ in range(iters):
        runtime.event_record(start, stream)
        fn()
        runtime.event_record(stop, stream)
        runtime.event_synchronize(stop)
        times.append(runtime.event_elapsed_time_ms(start, stop))
    runtime.event_destroy(start)
    runtime.event_destroy(stop)
    return times


def run(args) -> dict[str, object]:
    from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_t16_selected_prefill import (
        build_gguf_q4_k_t16_selected_prefill,
    )
    from hipengine.quant.gguf_q4_k import repack_gguf_q4_k_tile16

    runtime = get_hip_runtime()
    artifact = Path(args.artifact)
    reader = GGUFReader(artifact)
    tensor_name = f"blk.{args.layer}.ffn_gate_up_exps.weight"
    tensor = reader.tensor_info(tensor_name)
    if tensor.ggml_type_name != "Q4_K":
        raise SystemExit(f"{tensor_name} is {tensor.ggml_type_name}, not Q4_K")
    raw = np.asarray(reader.tensor_data(tensor_name))
    experts, fused_rows, row_bytes = raw.shape
    intermediate = fused_rows // 2

    started = time.perf_counter()
    stacked = raw.reshape(experts, fused_rows, -1)
    tiles_gate = repack_gguf_q4_k_tile16(stacked[:, :intermediate, :]).tiles
    tiles_up = repack_gguf_q4_k_tile16(stacked[:, intermediate:, :]).tiles
    repack_s = time.perf_counter() - started

    in_features = row_bytes * _QK_K // _Q4_K_BLOCK_BYTES
    out_features_total = fused_rows

    if args.distribution == "uniform":
        counts = np.full(experts, args.rows_per_expert, dtype=np.int64)
    else:
        counts = np.random.default_rng(args.seed).multinomial(
            args.compact_rows, np.full(experts, 1.0 / experts)
        )
    start_compact, start_wmma, tile_expert, compact_rows, wmma_total_rows = _metadata(counts)

    rng = np.random.default_rng(args.seed)
    x_f32 = (rng.standard_normal((compact_rows, in_features)) * 0.02).astype(np.float32)
    x_host = (
        x_f32.astype(np.float16).view(np.uint16)
        if args.dtype == "fp16"
        else _f32_to_bf16_u16(x_f32)
    )
    out_host = np.zeros((compact_rows, out_features_total), dtype=np.uint16)

    library = (
        ctypes.CDLL(args.library)
        if args.library
        else build_gguf_q4_k_t16_selected_prefill(
            compiler_version=args.compiler_version,
            require_cached=args.require_cached_build,
            load=True,
        )
    )
    baseline_library = (
        ctypes.CDLL(args.baseline_library)
        if args.baseline_library
        else library
    )

    buffers = []
    results: dict[str, np.ndarray] = {}
    timings: dict[str, list[float]] = {}
    try:
        x_dev = _to_device(x_host, runtime)
        start_compact_dev = _to_device(start_compact, runtime)
        start_wmma_dev = _to_device(start_wmma, runtime)
        tile_expert_dev = _to_device(tile_expert, runtime)
        tiles_gate_dev = _to_device(tiles_gate, runtime)
        tiles_up_dev = _to_device(tiles_up, runtime)
        out_dev = _to_device(out_host, runtime)
        buffers.extend(
            (
                x_dev,
                start_compact_dev,
                start_wmma_dev,
                tile_expert_dev,
                tiles_gate_dev,
                tiles_up_dev,
                out_dev,
            )
        )

        for name in args.variants:
            symbol = _VARIANTS[name]
            fn = (baseline_library if name == "baseline" else library)[symbol]
            fn.argtypes = [ctypes.c_void_p] * 7 + [ctypes.c_int64] * 6 + [ctypes.c_void_p]
            fn.restype = ctypes.c_int

            def launch(fn=fn) -> None:
                rc = fn(
                    x_dev.ptr,
                    start_compact_dev.ptr,
                    start_wmma_dev.ptr,
                    tile_expert_dev.ptr,
                    tiles_gate_dev.ptr,
                    tiles_up_dev.ptr,
                    out_dev.ptr,
                    compact_rows,
                    in_features,
                    intermediate,
                    intermediate,
                    experts,
                    wmma_total_rows,
                    None,
                )
                if rc != 0:
                    raise RuntimeError(f"{name} launch rejected: {rc}")

            out_host[:] = 0
            copy_host_to_device(out_dev, host_array_ptr(out_host), runtime=runtime)
            launch()
            runtime.device_synchronize()
            copy_device_to_host(host_array_ptr(out_host), out_dev, runtime=runtime)
            results[name] = out_host.copy()
            timings[name] = _timer(runtime, launch, warmup=args.warmup, iters=args.iters)
    finally:
        for buf in buffers:
            free(buf, runtime=runtime)

    reference = results[args.reference]
    assert np.abs(reference.astype(np.float32).view(np.float32) if False else 1.0).size
    reference_f32 = (reference.astype(np.uint32) << 16).view(np.float32)
    checks = {}
    for name, arr in results.items():
        differing = int((arr != reference).sum())
        checks[name] = {
            "differing_bits": differing,
            "total_bits": int(arr.size),
            "bit_identical": differing == 0,
            "max_abs_diff_vs_baseline": float(
                np.abs((arr.astype(np.uint32) << 16).view(np.float32) - reference_f32).max()
            ),
        }

    baseline_median = float(np.median(timings[args.reference]))
    variants = {}
    for name, times in timings.items():
        median = float(np.median(times))
        variants[name] = {
            "ms_median": median,
            "ms_best": float(min(times)),
            "ms_all": times,
            "speedup_vs_baseline": baseline_median / median if median else None,
        }

    row_tiles = int(wmma_total_rows) // _TILE_ROWS
    col_tiles = (out_features_total + _BLOCK_COLS - 1) // _BLOCK_COLS
    baseline_blocks = col_tiles * row_tiles
    pair_blocks = col_tiles * ((row_tiles + 1) // 2)
    return {
        "mode": "leaf-ab",
        "artifact": str(artifact),
        "tensor": tensor_name,
        "layer": args.layer,
        "distribution": args.distribution,
        "geometry": {
            "experts": int(experts),
            "compact_rows": int(compact_rows),
            "in_features": int(in_features),
            "intermediate": int(intermediate),
            "out_features_total": int(out_features_total),
            "row_tiles": row_tiles,
            "col_tiles": col_tiles,
            "wmma_total_rows": int(wmma_total_rows),
            "baseline_blocks": baseline_blocks,
            "rowpair_blocks": pair_blocks,
            "tiles_per_expert_mean": row_tiles / experts,
        },
        "bit_checks": checks,
        "variants": variants,
        "repack_s": repack_s,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifact", default=DEFAULT_ARTIFACT)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--distribution", choices=("uniform", "multinomial"), default="multinomial")
    ap.add_argument("--rows-per-expert", type=int, default=32)
    ap.add_argument("--compact-rows", type=int, default=4096)
    ap.add_argument("--variants", default="baseline,candidate")
    ap.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    ap.add_argument("--reference", default="baseline", help="variant used for the bit-identity check")
    ap.add_argument("--iters", type=int, default=15)
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--library", default="", help="prebuilt .so to load instead of the cached build")
    ap.add_argument(
        "--baseline-library",
        default="",
        help="prebuilt .so holding the reference production kernel; defaults to --library",
    )
    ap.add_argument("--compiler-version", default=None)
    ap.add_argument("--require-cached-build", action="store_true")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()
    args.variants = [v for v in args.variants.split(",") if v]

    payload = run(args)
    text = json.dumps(payload, indent=1)
    print(text)
    if args.json:
        args.json.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
