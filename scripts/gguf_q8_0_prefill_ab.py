"""Direct q8_0 GEMM comparison: the exact tiled route against the WMMA route.

Attention, the MoE router, and every other kernel are out of the picture. One
raw Q8_0 weight matrix and one bf16 activation block go through both registered
launchers, and each is compared against a CPU oracle built from the documented
Q8_0 byte layout that tests/_gguf_synthetic_weights.py generates.

The two routes have different activation contracts, so each gets its own oracle:
the tiled route reads the activation as bf16, the WMMA route narrows it to f16
first. Both contracts are exact, so a correct kernel reproduces its own oracle to
the last bit of the accumulator rounding, and a large gap against both oracles is
a kernel defect rather than a contract difference.
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "tests")

from _gguf_synthetic_weights import make_q8_0_weight  # noqa: E402
from hipengine.core.hip import get_hip_runtime  # noqa: E402
from hipengine.core.memory import (  # noqa: E402
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.hip_gfx1100.quant import gguf_k_gemv as _k  # noqa: E402,F401
from hipengine.kernels.hip_gfx1100.quant import gguf_q8_0_prefill as _p  # noqa: E402,F401
from hipengine.kernels.registry import resolve  # noqa: E402
from hipengine.loading.materialize import float_array_to_bf16_bits  # noqa: E402
from hipengine.quant.gguf import bf16_to_float32  # noqa: E402
from hipengine.kernels.registry import KernelKey  # noqa: E402

Q8_0_BLOCK = 32
Q8_0_BLOCK_BYTES = 34


def decode_q8_0(weight: np.ndarray, out_features: int, in_features: int) -> np.ndarray:
    """[out, in] f32 from the raw Q8_0 byte stream, per the documented layout."""
    blocks = in_features // Q8_0_BLOCK
    w = np.empty((out_features, in_features), dtype=np.float64)
    for c in range(out_features):
        for b in range(blocks):
            start = b * Q8_0_BLOCK_BYTES
            d = np.frombuffer(weight[c, start : start + 2].tobytes(), dtype=np.float16)[0]
            q = np.frombuffer(
                weight[c, start + 2 : start + Q8_0_BLOCK_BYTES].tobytes(), dtype=np.int8
            ).astype(np.float64)
            w[c, b * Q8_0_BLOCK : (b + 1) * Q8_0_BLOCK] = float(d) * q
    return w


def run(name: str, rows: int, in_features: int, out_features: int) -> dict:
    runtime = get_hip_runtime()
    weight = make_q8_0_weight(out_features, in_features)
    rng = np.random.default_rng(20260928)
    x_f32 = rng.normal(0, 0.35, size=(rows, in_features)).astype(np.float32)
    x_bf16 = np.ascontiguousarray(float_array_to_bf16_bits(x_f32))

    out = np.zeros((rows, out_features), dtype=np.uint16)
    buffers = []
    try:
        x_dev = malloc(x_bf16.nbytes, runtime=runtime)
        w_dev = malloc(weight.nbytes, runtime=runtime)
        out_dev = malloc(out.nbytes, runtime=runtime)
        buffers.extend((x_dev, w_dev, out_dev))
        copy_host_to_device(x_dev, host_array_ptr(x_bf16), runtime=runtime)
        copy_host_to_device(w_dev, host_array_ptr(weight), runtime=runtime)

        if name == "wmma":
            fn = resolve(
                backend="hip_gfx1100",
                layer="linear",
                quant="gguf_q8_0",
                variant="wmma_prefill_bf16_bf16_out",
            )
            fn(x_dev.ptr, w_dev.ptr, out_dev.ptr, rows, in_features, out_features)
        elif name == "tiled":
            fn = resolve(
                backend="hip_gfx1100",
                layer="linear",
                quant="gguf_q8_0",
                variant="exact_prefill_tile4x16_bf16_bf16_out",
            )
            fn(x_dev.ptr, w_dev.ptr, out_dev.ptr, rows, in_features, out_features)
        else:
            raise ValueError(name)
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(out), out_dev, runtime=runtime)
    finally:
        for buf in buffers:
            free(buf, runtime=runtime)

    got = bf16_to_float32(out).astype(np.float64)
    w = decode_q8_0(weight, out_features, in_features)

    # The tiled route reads the activation as bf16; the WMMA route narrows it to
    # f16 first. Both are exact contracts, so score against both.
    x_as_bf16 = bf16_to_float32(x_bf16).astype(np.float64)
    x_as_f16 = bf16_to_float32(x_bf16).astype(np.float16).astype(np.float64)
    o_bf16 = x_as_bf16 @ w.T
    o_f16 = x_as_f16 @ w.T

    def rel(a, b):
        scale = np.abs(b).max()
        return float(np.abs(a - b).max() / scale) if scale else 0.0

    return {
        "rows": rows,
        "vs_bf16_oracle": rel(got, o_bf16),
        "vs_f16_oracle": rel(got, o_f16),
        "got": got,
        "o_bf16": o_bf16,
        "o_f16": o_f16,
    }


if __name__ == "__main__" and len(sys.argv) == 1:
    # Real gemma4 shapes are all multiples of 64, so the sweep above cannot see a
    # tail defect. These add non-multiples on both axes.
    cases = [
        (2816, 512, (16, 32, 64, 128, 512, 1024)),
        (2816, 104, (64, 100)),     # out_features not a multiple of 64
        (2816, 128, (1, 2, 3, 5, 7, 31, 33, 63, 65, 100)),  # rows: odd and tiny
        (2816, 2112, (64, 100)),    # a wide gemma4 shape, non-multiple rows
    ]
    print(f"{'in':>6} {'out':>6} {'rows':>6} {'tiled vs oracle':>16} {'wmma vs oracle':>16} {'wmma vs tiled':>14}")
    worst = 0.0
    for in_features, out_features, row_set in cases:
        for rows in row_set:
            r = {n: run(n, rows, in_features, out_features) for n in ("tiled", "wmma")}
            a, b = r["tiled"]["got"], r["wmma"]["got"]
            scale = max(np.abs(a).max(), 1e-30)
            cross = float(np.abs(a - b).max() / scale)
            worst = max(worst, cross)
            print(
                f"{in_features:>6} {out_features:>6} {rows:>6} "
                f"{r['tiled']['vs_bf16_oracle']:>15.3e} {r['wmma']['vs_bf16_oracle']:>15.3e} "
                f"{cross:>13.3e}"
            )
    print()
    print(f"worst wmma-vs-tiled relative difference across every shape: {worst:.3e}")


def tile_sweep(in_features: int, out_features: int, rows: int) -> None:
    """Every (tile_m, tile_n) the launcher accepts, against the tiled route.

    The runtime does not call this kernel with the wrapper's default tile; it
    passes an explicit (tile_m, tile_n) from its own selection heuristic. A bare
    comparison with the default tile therefore cannot see a defect that lives in
    one particular tile.
    """
    print(f"tile sweep: in={in_features} out={out_features} rows={rows}")
    print(f"{'tile_m':>7} {'tile_n':>7} {'vs f16 oracle':>15}")
    for tile_m in (16, 32, 64):
        for tile_n in (16, 32):
            runtime = get_hip_runtime()
            weight = make_q8_0_weight(out_features, in_features)
            rng = np.random.default_rng(20260928)
            x_f32 = rng.normal(0, 0.35, size=(rows, in_features)).astype(np.float32)
            x_bf16 = np.ascontiguousarray(float_array_to_bf16_bits(x_f32))
            out = np.zeros((rows, out_features), dtype=np.uint16)
            buffers = []
            try:
                x_dev = malloc(x_bf16.nbytes, runtime=runtime)
                w_dev = malloc(weight.nbytes, runtime=runtime)
                out_dev = malloc(out.nbytes, runtime=runtime)
                buffers.extend((x_dev, w_dev, out_dev))
                copy_host_to_device(x_dev, host_array_ptr(x_bf16), runtime=runtime)
                copy_host_to_device(w_dev, host_array_ptr(weight), runtime=runtime)
                fn = resolve(
                    backend="hip_gfx1100", layer="linear", quant="gguf_q8_0",
                    variant="wmma_prefill_bf16_bf16_out",
                )
                fn(x_dev.ptr, w_dev.ptr, out_dev.ptr, rows, in_features, out_features,
                   tile_m=tile_m, tile_n=tile_n)
                runtime.device_synchronize()
                copy_device_to_host(host_array_ptr(out), out_dev, runtime=runtime)
            finally:
                for buf in buffers:
                    free(buf, runtime=runtime)
            got = bf16_to_float32(out).astype(np.float64)
            w = decode_q8_0(weight, out_features, in_features)
            x_as_f16 = bf16_to_float32(x_bf16).astype(np.float16).astype(np.float64)
            o = x_as_f16 @ w.T
            print(f"{tile_m:>7} {tile_n:>7} {float(np.abs(got - o).max()/np.abs(o).max()):>14.3e}")


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "tiles":
    for r in (64, 128, 256, 512):
        tile_sweep(2816, 512, r)
