"""PF-1d diagnostic screen: MMQ plane-count lever at production shapes.

Times the existing raw-Q8_0 MMQ128 chain (quantize + WMMA GEMM) with 1/2/3
activation planes at every QWEN4EXP_Q8_MMQ_PREFILL_POLICY production shape,
rows=512 prefill chunks, plus the guarded d4x3 production chain for
reference. No production code is changed; this sizes the plane lever before
any variant work. Output: per-shape per-plane milliseconds and ratios.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import (
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q8_0_mmq_prefill import (
    QWEN4EXP_Q8_MMQ_PREFILL_POLICY,
    build_gguf_q8_0_mmq_prefill,
    gguf_q8_0_mmq128_prefill_q8_1_d4_bf16_bf16_out,
    gguf_q8_0_mmq128_prefill_q8_1_d4x2_bf16_bf16_out,
    gguf_q8_0_mmq128_prefill_q8_1_d4x3_bf16_bf16_out,
    gguf_q8_0_mmq128_prefill_q8_1_d4x3_guarded_f32_f32_out,
    gguf_q8_0_mmq128_quantize_bf16_d4,
    gguf_q8_0_mmq128_quantize_bf16_d4x2,
    gguf_q8_0_mmq128_quantize_bf16_d4x3,
    q8_mmq_d4_nbytes,
    q8_mmq_d4x2_nbytes,
    q8_mmq_d4x3_nbytes,
)

PLANES = {
    1: (
        gguf_q8_0_mmq128_quantize_bf16_d4,
        gguf_q8_0_mmq128_prefill_q8_1_d4_bf16_bf16_out,
        q8_mmq_d4_nbytes,
    ),
    2: (
        gguf_q8_0_mmq128_quantize_bf16_d4x2,
        gguf_q8_0_mmq128_prefill_q8_1_d4x2_bf16_bf16_out,
        q8_mmq_d4x2_nbytes,
    ),
    3: (
        gguf_q8_0_mmq128_quantize_bf16_d4x3,
        gguf_q8_0_mmq128_prefill_q8_1_d4x3_bf16_bf16_out,
        q8_mmq_d4x3_nbytes,
    ),
}


def random_q8_0_weight(out_features: int, in_features: int, rng) -> np.ndarray:
    blocks = in_features // 32
    d = (rng.uniform(0.005, 0.08, size=(out_features, blocks))).astype(
        np.float16
    )
    q = rng.integers(-127, 128, size=(out_features, blocks, 32)).astype(np.int8)
    data = np.empty((out_features, blocks, 34), dtype=np.uint8)
    data[:, :, :2] = d.view(np.uint8).reshape(out_features, blocks, 2)
    data[:, :, 2:] = q.view(np.uint8)
    return np.ascontiguousarray(data.reshape(out_features, blocks * 34))


def screen_shape(
    runtime,
    library,
    rows: int,
    hidden: int,
    out_features: int,
    *,
    warmup: int,
    repeats: int,
) -> dict[int, float]:
    rng = np.random.default_rng(20260902)
    qweight = random_q8_0_weight(out_features, hidden, rng)
    x_bf16 = (
        (rng.standard_normal((rows, hidden)) * 0.5).astype(np.float32).view(
            np.uint32
        )
    )
    # pack to bf16 bits
    bits = x_bf16
    lsb = (bits >> 16) & 1
    x_bits = np.ascontiguousarray(((bits + 0x7FFF + lsb) >> 16).astype(np.uint16))

    weight_dev = malloc(qweight.nbytes, runtime=runtime)
    x_dev = malloc(x_bits.nbytes, runtime=runtime)
    out_dev = malloc(rows * out_features * 4, runtime=runtime)
    timings: dict[int, float] = {}
    try:
        copy_host_to_device(
            weight_dev, host_array_ptr(qweight), runtime=runtime
        )
        copy_host_to_device(x_dev, host_array_ptr(x_bits), runtime=runtime)
        for planes, (quantize, matmul, d4_nbytes) in PLANES.items():
            d4_dev = malloc(d4_nbytes(rows, hidden), runtime=runtime)
            try:
                for _ in range(warmup):
                    quantize(
                        x_dev.ptr,
                        d4_dev.ptr,
                        rows,
                        hidden,
                        library=library,
                        runtime=runtime,
                    )
                    matmul(
                        d4_dev.ptr,
                        weight_dev.ptr,
                        out_dev.ptr,
                        rows,
                        hidden,
                        out_features,
                        library=library,
                        runtime=runtime,
                    )
                runtime.device_synchronize()
                start = time.perf_counter()
                for _ in range(repeats):
                    quantize(
                        x_dev.ptr,
                        d4_dev.ptr,
                        rows,
                        hidden,
                        library=library,
                        runtime=runtime,
                    )
                    matmul(
                        d4_dev.ptr,
                        weight_dev.ptr,
                        out_dev.ptr,
                        rows,
                        hidden,
                        out_features,
                        library=library,
                        runtime=runtime,
                    )
                runtime.device_synchronize()
                timings[planes] = (time.perf_counter() - start) * 1e3 / repeats
            finally:
                free(d4_dev, runtime=runtime)
    finally:
        free(out_dev, runtime=runtime)
        free(x_dev, runtime=runtime)
        free(weight_dev, runtime=runtime)
    return timings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument(
        "--shapes",
        default="all",
        help="all or comma-separated hidden:out_features pairs",
    )
    args = parser.parse_args()

    shapes = sorted(QWEN4EXP_Q8_MMQ_PREFILL_POLICY.min_rows)
    if args.shapes != "all":
        shapes = [
            tuple(int(v) for v in pair.split(":")) for pair in args.shapes.split(",")
        ]

    runtime = get_hip_runtime()
    library = build_gguf_q8_0_mmq_prefill(load=True)
    print(f"rows={args.rows} warmup={args.warmup} repeats={args.repeats}")
    print(f"{'hidden':>7} {'out':>6} {'d4 ms':>10} {'d4x2 ms':>10} {'d4x3 ms':>10} {'x3/x1':>6} {'x3/x2':>6}")
    total = {1: 0.0, 2: 0.0, 3: 0.0}
    for hidden, out_features in shapes:
        timings = screen_shape(
            runtime,
            library,
            args.rows,
            hidden,
            out_features,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        for planes, ms in timings.items():
            total[planes] += ms
        print(
            f"{hidden:>7} {out_features:>6} "
            f"{timings[1]:>10.3f} {timings[2]:>10.3f} {timings[3]:>10.3f} "
            f"{timings[3] / timings[1]:>6.2f} {timings[3] / timings[2]:>6.2f}"
        )
    print(
        f"{'total':>14} {total[1]:>10.3f} {total[2]:>10.3f} {total[3]:>10.3f} "
        f"{total[3] / total[1]:>6.2f} {total[3] / total[2]:>6.2f}"
    )


if __name__ == "__main__":
    main()
