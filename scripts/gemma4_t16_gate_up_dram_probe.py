#!/usr/bin/env python3
"""Drive the Gemma 4 Q4T16 expert gate/up prefill leaf for DRAM counter work.

The leaf (``gguf_q4_k_t16_selected_dual_wmma_prefill_compact32_bf16_bf16_out``)
is the single largest term in a 4096-token Gemma 4 26B-A4B prefill. A prior unit
priced it at 65.4 GB/s by dividing *unique* bytes -- the tile bytes once, plus
the activations once, plus the output once -- by its wall time. That denominator
excludes two re-read structures the kernel actually has:

* a 16-row tile walks every output tile of its expert, so with ``R`` compact
  rows per expert the tile bytes are demanded ``ceil(R / 16)`` times, and
* every 32-column block reads its 16 rows' full activation row twice, once per
  16-column half, and there are ``out_features / 32`` blocks, so the activation
  is demanded ``out_features / 16`` times.

Whether either reaches DRAM is a measurement, not an argument. This probe exists
to take it: it drives the leaf with the model's own weight tensor and the model's
own grid, at a controllable rows-per-expert, so the counter can be read against a
demand that is known by construction.

Two modes:

``--mode leaf``
    Load one layer's ``ffn_gate_up_exps`` Q4_K tensor from the campaign artifact,
    repack it to the production T16 tiles, build uniform compact-selected MoE
    metadata for ``--experts`` experts at ``--rows-per-expert`` rows each, and
    launch the production leaf. Prints the exact demand accounting next to the
    measurement so the counter has something to be checked against.

``--mode stream``
    Read a contiguous buffer of ``--stream-mb`` once with
    ``hipengine_g4_dram_stream_read``. The expected DRAM read traffic is the
    buffer size exactly. This is the counter's known-answer case and the
    same-session achievable-read control.

The probe measures no counters itself; run it under
``rocprofv3 --pmc GL2C_EA_RDREQ_DRAM_sum --kernel-include-regex <kernel>``.

Diagnostic only: no perf claim, no production path, and nothing registered.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import subprocess
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np  # noqa: E402

from hipengine.core.build import build_hip  # noqa: E402
from hipengine.core.hip import get_hip_runtime  # noqa: E402
from hipengine.core.memory import (  # noqa: E402
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.loading.gguf import GGUFReader  # noqa: E402

_STREAM_SOURCE = Path(__file__).with_name("gemma4_t16_gate_up_dram_probe.hip")
_STREAM_SYMBOL = "hipengine_g4_dram_stream_read"

DEFAULT_ARTIFACT = (
    "/models/gguf/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf"
)

# The leaf's compile-time geometry, read from
# ``gguf_q4_k_t16_selected_prefill.hip``: one wave32 block covers 32 output
# columns as two 16-column WMMA tiles, and one 16-row compact tile.
_T16_COLS = 16
_BLOCK_COLS = 32
_TILE_ROWS = 16
_TILE_BYTES = 2368  # Q4_T16_BLOCK_BYTES, non-lite: 2.78 percent over raw
_Q4_K_BLOCK_BYTES = 144
_QK_K = 256


def _f32_to_bf16_u16(arr: np.ndarray) -> np.ndarray:
    f32 = np.ascontiguousarray(arr, dtype=np.float32)
    u32 = f32.view(np.uint32).copy()
    lsb = (u32 >> 16) & 1
    return ((u32 + 0x7FFF + lsb) >> 16).astype(np.uint16).reshape(f32.shape)


def _uniform_metadata(experts: int, rows_per_expert: int):
    """Compact-selected metadata for a uniform expert load.

    Mirrors the production tile walk: ``expert_start_compact`` indexes the
    compact row buffer, ``expert_start_wmma`` indexes the 16-row-padded tile
    walk, and ``tile_expert`` names the expert of each 16-row tile.
    """

    counts = np.full(experts, rows_per_expert, dtype=np.int64)
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


def _git_state() -> dict[str, object]:
    def run(*args: str) -> str:
        return subprocess.check_output(args, text=True, cwd=str(_REPO_ROOT)).strip()

    try:
        commit = run("git", "rev-parse", "HEAD")
        dirty = bool(run("git", "status", "--porcelain"))
    except Exception:
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": dirty}


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


def run_stream(args) -> dict[str, object]:
    runtime = get_hip_runtime()
    library = build_hip(
        sources=[_STREAM_SOURCE],
        family="gemma4_t16_gate_up_dram_probe",
        profile="baseline",
        compiler_version=args.compiler_version,
        require_cached=args.require_cached_build,
        output_name="gemma4_t16_gate_up_dram_probe.so",
        load=True,
    )
    fn = library[_STREAM_SYMBOL]
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int

    nbytes = int(args.stream_mb) * 1024 * 1024
    nbytes -= nbytes % 16
    n4 = nbytes // 16
    src = malloc(nbytes, runtime=runtime)
    sink = malloc(16, runtime=runtime)
    # Touch every page so the read is cold DRAM rather than a fault-in.
    zeros = np.zeros(nbytes, dtype=np.uint8)
    copy_host_to_device(src, host_array_ptr(zeros), runtime=runtime)

    threads = 256
    blocks = min(8192, max(1, (n4 + threads - 1) // threads))

    def launch() -> None:
        rc = fn(src.ptr, sink.ptr, n4, blocks, threads, None)
        if rc != 0:
            raise RuntimeError(f"stream launch rejected: {rc}")

    times = _timer(runtime, launch, warmup=args.warmup, iters=args.iters)
    best = min(times)
    median = float(np.median(times))
    payload = {
        "mode": "stream",
        "buffer_bytes": nbytes,
        "blocks": blocks,
        "threads": threads,
        "iters": args.iters,
        "warmup": args.warmup,
        "ms_median": median,
        "ms_best": best,
        "gbps_median": nbytes / 1e9 / (median / 1e3),
        "gbps_best": nbytes / 1e9 / (best / 1e3),
        "expected_dram_read_bytes": nbytes,
    }
    free(src, runtime=runtime)
    free(sink, runtime=runtime)
    return payload


def run_leaf(args) -> dict[str, object]:
    from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_t16_selected_prefill import (
        build_gguf_q4_k_t16_selected_prefill,
        gguf_q4_k_t16_selected_dual_wmma_prefill_compact32_bf16_bf16_out as leaf,
    )
    from hipengine.quant.gguf_q4_k import repack_gguf_q4_k_tile16

    runtime = get_hip_runtime()
    artifact = Path(args.artifact)
    reader = GGUFReader(artifact)
    tensor_name = f"blk.{args.layer}.ffn_gate_up_exps.weight"
    tensor = reader.tensor_info(tensor_name)
    if tensor.ggml_type_name != "Q4_K":
        raise SystemExit(
            f"{tensor_name} is {tensor.ggml_type_name}; this probe's tile repack "
            "is the Q4_K one and the leaf is the Q4_K leaf"
        )
    raw = np.asarray(reader.tensor_data(tensor_name))
    experts, fused_rows, row_bytes = raw.shape
    if fused_rows % 2:
        raise SystemExit(f"{fused_rows} fused output rows do not halve")
    intermediate = fused_rows // 2
    blocks_per_row = row_bytes // _Q4_K_BLOCK_BYTES

    started = time.perf_counter()
    stacked = raw.reshape(experts, fused_rows, -1)
    tiles_gate = repack_gguf_q4_k_tile16(stacked[:, :intermediate, :]).tiles
    tiles_up = repack_gguf_q4_k_tile16(stacked[:, intermediate:, :]).tiles
    repack_s = time.perf_counter() - started
    if args.repack_only:
        # Sizes only: no device allocation, no launch. This is the cheap check
        # that the artifact's tensor is the Q4_K one and that the tile byte
        # count matches the denominator any prior price was built from.
        return {
            "mode": "leaf",
            "repack_only": True,
            "tensor": tensor_name,
            "ggml_type": tensor.ggml_type_name,
            "raw_bytes": int(raw.nbytes),
            "tiles_per_half": list(tiles_gate.shape),
            "tile_bytes_per_half": int(tiles_gate.nbytes),
            "tile_bytes_both_halves": int(tiles_gate.nbytes + tiles_up.nbytes),
            "tile_over_raw_ratio": float(
                (tiles_gate.nbytes + tiles_up.nbytes) / raw.nbytes
            ),
            "repack_s": repack_s,
        }

    (
        start_compact,
        start_wmma,
        tile_expert,
        compact_rows,
        wmma_total_rows,
    ) = _uniform_metadata(experts, args.rows_per_expert)

    in_features = row_bytes * _QK_K // _Q4_K_BLOCK_BYTES
    out_features_total = fused_rows

    rng = np.random.default_rng(args.seed)
    x_host = _f32_to_bf16_u16(
        (rng.standard_normal((compact_rows, in_features)) * 0.02).astype(np.float32)
    )
    out_host = np.zeros((compact_rows, out_features_total), dtype=np.uint16)

    library = build_gguf_q4_k_t16_selected_prefill(
        compiler_version=args.compiler_version,
        require_cached=args.require_cached_build,
        load=True,
    )

    buffers = []
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

        col_tiles = (out_features_total + _BLOCK_COLS - 1) // _BLOCK_COLS
        grid_blocks = col_tiles * (wmma_total_rows // _TILE_ROWS)

        def launch() -> None:
            leaf(
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
                library=library,
            )

        times = _timer(runtime, launch, warmup=args.warmup, iters=args.iters)
    finally:
        for buf in buffers:
            free(buf, runtime=runtime)

    # Demand, from the kernel's own structure rather than from a guess about
    # what the cache does with it.
    tiles_both_halves = int(tiles_gate.nbytes + tiles_up.nbytes)
    row_tiles_per_expert = -(-args.rows_per_expert // _TILE_ROWS)
    weight_demand = tiles_both_halves * row_tiles_per_expert
    # Per 32-column block: 2 output tiles, each walking the full K row of its 16
    # rows. One activation row is in_features * 2 bytes.
    act_per_block = 2 * _TILE_ROWS * in_features * 2
    act_demand = grid_blocks * act_per_block
    act_unique = compact_rows * in_features * 2
    out_bytes = compact_rows * out_features_total * 2

    best = min(times)
    median = float(np.median(times))
    unique_read = tiles_both_halves + act_unique
    demand_read = weight_demand + act_demand
    payload = {
        "mode": "leaf",
        "artifact": str(artifact),
        "tensor": tensor_name,
        "ggml_type": tensor.ggml_type_name,
        "layer": args.layer,
        "geometry": {
            "experts": experts,
            "rows_per_expert": args.rows_per_expert,
            "compact_rows": compact_rows,
            "in_features": in_features,
            "intermediate": intermediate,
            "out_features_total": out_features_total,
            "blocks_per_row": blocks_per_row,
            "wmma_total_rows": wmma_total_rows,
            "col_tiles": col_tiles,
            "grid_blocks": grid_blocks,
            "threads_per_block": 32,
            "row_tiles_per_expert": row_tiles_per_expert,
        },
        "bytes": {
            "raw_weight": int(raw.nbytes),
            "tiles_both_halves": tiles_both_halves,
            "weight_demand": weight_demand,
            "activation_unique": act_unique,
            "activation_demand": act_demand,
            "output_write": out_bytes,
            "unique_read_total": unique_read,
            "demand_read_total": demand_read,
        },
        "ms_median": median,
        "ms_best": best,
        "ms_all": times,
        "rates": {
            "gbps_at_unique": unique_read / 1e9 / (median / 1e3),
            "gbps_at_demand": demand_read / 1e9 / (median / 1e3),
            "gbps_at_weight_demand": weight_demand / 1e9 / (median / 1e3),
            "gbps_at_tiles_once": tiles_both_halves / 1e9 / (median / 1e3),
        },
        "repack_s": repack_s,
    }
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("leaf", "stream"), default="leaf")
    ap.add_argument("--artifact", default=DEFAULT_ARTIFACT)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--experts", type=int, default=128)
    ap.add_argument("--rows-per-expert", type=int, default=32)
    ap.add_argument("--stream-mb", type=int, default=512)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--repack-only", action="store_true")
    ap.add_argument("--compiler-version", default=None)
    ap.add_argument("--require-cached-build", action="store_true")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    if args.mode == "stream":
        payload = run_stream(args)
    else:
        payload = run_leaf(args)
    payload["git"] = _git_state()
    payload["recorded"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    print(json.dumps(payload, indent=1, default=float))
    if args.json:
        args.json.write_text(json.dumps(payload, indent=1, default=float) + "\n")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
