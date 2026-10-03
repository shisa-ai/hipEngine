"""Measure the Q8_0 T16 four-wave prefill's achieved throughput at rows=512.

This is the cost/benefit test for the loader change (worklog/entries/20260929T143000,
20260929T153000). It deliberately does NOT need the raw route's kernel identity.

The metric is TFLOPS, not GB/s. At rows=512 these shapes have an arithmetic
intensity near 963 FLOP/byte, so the GEMM is compute-bound and a bandwidth figure
is misleading -- an earlier version of this script reported GB/s and read a
compute-bound kernel as 8x slower than a bandwidth-bound one. The comparison that
matters is the dense term's in-situ rate: 189.31 ms for 1.576 TFLOP of known
shape work, i.e. 8.33 TFLOPS, of which the rows=1 lm head is a very slow part.
"""

from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, ".")

# The dense leaf shapes the dispatch actually produces at a 512-token prefill,
# from scripts/gemma4_dense_dispatch_probe.py. All are at or above the wrapper's
# out_features >= 2048 gate, which is 200 of the 206 dense calls.
SHAPES = ((2816, 2816), (2112, 2816), (4096, 2816), (8192, 2816))
ROWS = 512


def main() -> int:
    from hipengine.core import memory as mem
    from hipengine.core.hip import get_hip_runtime
    from hipengine.kernels.hip_gfx1100.quant.gguf_q8_0_t16_prefill import (
        _default_tiles,
        _four_wave_prefill_applies,
        gguf_q8_0_t16_wmma_prefill_4wave_bf16_bf16_out as four_wave_prefill,
    )
    from hipengine.quant.gguf_t16 import (
        GGUF_Q8_0_BLOCK_BYTES,
        GGUF_Q8_0_QK,
        repack_gguf_q8_0_tile16,
    )

    runtime = get_hip_runtime()
    symbol = "gguf_q8_0_t16_wmma_prefill_4wave_bf16_bf16_out"

    print(f"rows={ROWS}  four-wave wrapper: {symbol}\n")
    print("in-situ comparison: the dense term is 189.31 ms for 1.576 TFLOP of known")
    print("shape work = 8.33 TFLOPS aggregate, and that INCLUDES the rows=1 lm head,")
    print("which is far below the rows=512 rate. So the true rows=512 in-situ rate is")
    print("ABOVE 8.33, and this microbench must beat that by enough to matter.\n")
    print(f"{'out':>6} {'in':>6} {'tile':>9} {'ms':>9} {'TFLOPS':>8} {'GB/s':>7}")
    print("-" * 56)

    rng = np.random.default_rng(20260929)
    results = []
    for out_features, in_features in SHAPES:
        blocks_per_row = in_features // GGUF_Q8_0_QK
        raw = rng.integers(
            0,
            256,
            size=(out_features, blocks_per_row * GGUF_Q8_0_BLOCK_BYTES),
            dtype=np.uint8,
        )
        packed = repack_gguf_q8_0_tile16(raw)
        tiles = np.ascontiguousarray(packed.tiles, dtype=np.uint8)

        tile_m, tile_n = _default_tiles(ROWS, in_features, out_features)
        four = _four_wave_prefill_applies(
            tile_m=tile_m, tile_n=tile_n, out_features=out_features, default=True
        )
        if four:
            tile_m = 128  # what the wrapper does when the gate fires

        x = np.zeros((ROWS, in_features), dtype=np.uint16)  # bf16 activations
        out = np.zeros((ROWS, out_features), dtype=np.uint16)  # bf16 output

        xbuf = mem.malloc(x.nbytes)
        tbuf = mem.malloc(tiles.nbytes)
        obuf = mem.malloc(out.nbytes)
        mem.copy_host_to_device(xbuf, mem.host_array_ptr(x))
        mem.copy_host_to_device(tbuf, mem.host_array_ptr(tiles))

        def call():
            four_wave_prefill(
                xbuf.ptr, tbuf.ptr, obuf.ptr,
                ROWS, in_features, out_features,
                tile_m=tile_m, tile_n=tile_n,
                runtime=runtime,
            )

        try:
            for _ in range(3):
                call()
            runtime.device_synchronize()
        except Exception as exc:  # noqa: BLE001
            print(f"{out_features:6d} {in_features:6d}  launch failed: {type(exc).__name__}: {exc}")
            continue

        start = runtime.event_create()
        stop = runtime.event_create()
        iters = 20
        runtime.event_record(start)
        for _ in range(iters):
            call()
        runtime.event_record(stop)
        runtime.device_synchronize()
        ms = runtime.event_elapsed_time_ms(start, stop) / iters

        weight_bytes = out_features * in_features / GGUF_Q8_0_QK * GGUF_Q8_0_BLOCK_BYTES
        gbs = weight_bytes / (ms * 1e-3) / 1e9
        flops = 2.0 * ROWS * in_features * out_features
        tflops = flops / (ms * 1e-3) / 1e12
        results.append((out_features, in_features, ms, gbs, tflops))
        print(
            f"{out_features:6d} {in_features:6d} {tile_m:4d}x{tile_n:<4d} "
            f"{ms:9.4f} {tflops:8.1f} {gbs:7.1f}"
        )

    print()
    if not results:
        print("NO RESULT -- every launch failed, so nothing is concluded about the lever.")
        return 1
    best = max(results, key=lambda r: r[4])
    print(
        f"best: out={best[0]} in={best[1]} {best[4]:.1f} TFLOPS "
        f"({best[4] / 8.33:.2f}x the dense term's 8.33 TFLOPS AGGREGATE)"
    )
    print(
        "NOTE: this is an isolated microbench of one kernel at one row count, not an\n"
        "end-to-end claim. Because the in-situ aggregate already includes a very slow\n"
        "rows=1 lm head, the rows=512 in-situ rate is higher than 8.33, so a ratio\n"
        "computed against the aggregate OVERSTATES the lever. A like-for-like in-situ\n"
        "rows=512 measurement is required before the loader change is justified."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
