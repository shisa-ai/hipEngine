"""Compare the router-logits variants at the Gemma 4 production shape.

`gemma4_router_topk_bf16` calls `qwen35_router_logits_bf16_f32w` with the
default ``threads=512``, which dispatches to ``token_tile`` with
TOKENS_PER_BLOCK=4. At the real shape (hidden 2816, 128 experts) that leaves
threads 352..511 with no K range at all -- 31% of every block idle behind a
nine-round barrier tree, 64 FLOPs per useful thread.

Several other variants already exist in the same library and none of them is
selected by the Gemma 4 route. This measures them side by side on one shape so
the choice is made on evidence rather than on the default that shipped.
"""

from __future__ import annotations

import statistics
import time

import numpy as np

from hipengine.core.memory import (
    copy_device_to_host,
    copy_device_to_host as _c2h,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.core.hip import get_hip_runtime
from hipengine.kernels.hip_gfx1100.moe.router import (
    _ARGTYPES_ROUTER_LOGITS,
    _router_library,
)
from hipengine.kernels.hip_gfx1100.moe.router import (
    qwen35_router_logits_bf16_f32w as base_f32w,
)
from hipengine.kernels.hip_gfx1100.moe.router import (
    qwen35_router_logits_bf16_f32w_auto_256 as auto256,
)
from hipengine.kernels.hip_gfx1100.moe.router import (
    qwen35_router_logits_bf16_f32w_token_tile_8 as tt8,
)
from hipengine.kernels.hip_gfx1100.moe.router import (
    qwen35_router_logits_bf16_f32w_token_tile_16 as tt16,
)
from hipengine.kernels.hip_gfx1100.moe.router import signed_kernel_fn

TOKENS = 512
HIDDEN = 2816
EXPERTS = 128
REPEATS = 200


def _bf16_bits(values: np.ndarray) -> np.ndarray:
    u = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    u = u + ((((u >> np.uint32(16)) & np.uint32(1))) + np.uint32(0x7FFF))
    return (u >> np.uint32(16)).astype(np.uint16)


def reference(hidden: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Plain float64 matmul downcast to float32 -- accuracy yardstick only."""

    h = hidden.view(np.float16).astype(np.float32) if False else None
    del h
    from hipengine.quant.gguf_q4_k import _bf16_u16_to_f32

    h = _bf16_u16_to_f32(hidden).astype(np.float64)
    w = weight.astype(np.float64)
    return (h @ w.T).astype(np.float32)


def run(name, fn, hidden_dev, weight_dev, logits_dev, runtime, library):
    """Time `fn` over REPEATS launches and report max drift vs the reference."""

    fn(
        hidden_dev.ptr,
        weight_dev.ptr,
        logits_dev.ptr,
        TOKENS,
        HIDDEN,
        EXPERTS,
        stream=0,
        library=library,
        runtime=runtime,
    )
    runtime.device_synchronize()

    start = time.perf_counter()
    for _ in range(REPEATS):
        fn(
            hidden_dev.ptr,
            weight_dev.ptr,
            logits_dev.ptr,
            TOKENS,
            HIDDEN,
            EXPERTS,
            stream=0,
            library=library,
            runtime=runtime,
        )
    runtime.device_synchronize()
    per_call = (time.perf_counter() - start) / REPEATS * 1e3  # ms

    out = np.empty((TOKENS, EXPERTS), dtype=np.float32)
    copy_device_to_host(host_array_ptr(out), logits_dev, runtime=runtime)
    return per_call, out


def main() -> None:
    rng = np.random.default_rng(7)
    hidden = _bf16_bits(rng.standard_normal((TOKENS, HIDDEN)) * 2.0)
    weight = rng.standard_normal((EXPERTS, HIDDEN)).astype(np.float32) * 0.05

    ref = reference(hidden, weight)

    runtime = get_hip_runtime()
    library = _router_library()
    bufs = []
    try:
        h_dev = malloc(hidden.nbytes, runtime=runtime)
        w_dev = malloc(weight.nbytes, runtime=runtime)
        l_dev = malloc(ref.nbytes, runtime=runtime)
        bufs = [h_dev, w_dev, l_dev]
        copy_host_to_device(
            h_dev, host_array_ptr(np.ascontiguousarray(hidden)), runtime=runtime
        )
        copy_host_to_device(
            w_dev, host_array_ptr(np.ascontiguousarray(weight)), runtime=runtime
        )

        candidates = [
            ("base f32w (threads=512, TPB=4)", base_f32w),
            ("token_tile_8 (threads=256)", tt8),
            ("token_tile_16 (threads=256)", tt16),
            ("auto_256", auto256),
        ]

        flops = 2.0 * TOKENS * HIDDEN * EXPERTS
        print(f"shape: tokens={TOKENS} hidden={HIDDEN} experts={EXPERTS}")
        print(f"per launch: {flops / 1e6:.1f} MFLOP, {REPEATS} repeats\n")
        print(f"{'variant':34s} {'ms':>9s} {'TFLOP/s':>9s} {'maxabs':>11s}")
        results = []
        for name, fn in candidates:
            try:
                ms, out = run(name, fn, h_dev, w_dev, l_dev, runtime, library)
            except Exception as exc:  # a variant that will not launch is a result
                print(f"{name:34s} {'FAIL':>9s}  {type(exc).__name__}: {exc}")
                continue
            maxabs = float(np.max(np.abs(out - ref)))
            tflops = flops / (ms * 1e-3) / 1e12
            results.append((name, ms, tflops, maxabs))
            print(f"{name:34s} {ms:9.4f} {tflops:9.2f} {maxabs:11.3e}")

        if results:
            best = min(results, key=lambda r: r[1])
            print(f"\nfastest: {best[0]} at {best[1]:.4f} ms ({best[2]:.2f} TFLOP/s)")
    finally:
        for buf in reversed(bufs):
            free(buf, runtime=runtime)


if __name__ == "__main__":
    main()