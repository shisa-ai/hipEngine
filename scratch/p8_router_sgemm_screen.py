"""P8 remainder screen: can a bound GEMM beat the token-tile router logits?

The P8 row's remaining ask routes the router through WMMA or hipblaslt,
"blocked on a dtype".  This screen tests the unnamed alternative that keeps
the F32 weights: upcast the BF16 prescale output (bit-exact) and run the
projection through the bound rocBLAS ``sgemm_rowmajor_nt`` -- same inputs,
different accumulation order.  The row's named F16-downcast route is timed
alongside it, with the downcast leg's error reported.

Candidates, at the exact production shape (hidden 2816, 128 experts):

  A  token_tile_16, threads=128   the production prefill route (>= 32 tokens)
  A' qwen35_router_logits_bf16_f32w threads=512   the production sub-32 route
  B  bf16_to_f32 + sgemm         F32 weights, bit-exact inputs
  B' sgemm alone                 what a future prescale-direct-f32 would give
  C  bf16_to_fp16 + gemm_ex f16  the row's downcast route, lossy leg printed

Every candidate is compared against a float64 reference before timing and
timed with the same perf_counter pattern as
``scripts/gemma4_router_logits_variant_bench.py`` so the numbers stay
comparable with the P8 baseline.
"""

from __future__ import annotations

import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.core.hip import get_hip_runtime
from hipengine.core.rocblas import get_rocblas
from hipengine.kernels.hip_gfx1100.convert.cast import bf16_to_f32, bf16_to_fp16
from hipengine.kernels.hip_gfx1100.moe.router import (
    qwen35_router_logits_bf16_f32w,
    qwen35_router_logits_bf16_f32w_token_tile_16,
)

HIDDEN = 2816
EXPERTS = 128
TT16_MIN = 32
TT16_THREADS = 128
# Full ladder: production widths (1 decode / block sizes), the tier
# crossover band (256-896), and the wide block the row quotes.
SWEEP = [1, 16, 128, 256, 512, 640, 768, 896, 1024, 4096]


def _load_reference():
    spec = importlib.util.spec_from_file_location(
        "p8bench", Path("scripts/gemma4_router_logits_variant_bench.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _bf16_bits(values: np.ndarray) -> np.ndarray:
    u = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    u = u + ((((u >> np.uint32(16)) & np.uint32(1))) + np.uint32(0x7FFF))
    return (u >> np.uint32(16)).astype(np.uint16)


def repeats(tokens: int) -> int:
    if tokens <= 16:
        return 2000
    if tokens <= 128:
        return 600
    if tokens <= 1024:
        return 200
    return 60


def main() -> None:
    bench = _load_reference()
    rng = np.random.default_rng(7)
    max_tokens = max(SWEEP)

    hidden = _bf16_bits(rng.standard_normal((max_tokens, HIDDEN)) * 2.0)
    weight = rng.standard_normal((EXPERTS, HIDDEN)).astype(np.float32) * 0.05
    weight_f16 = weight.astype(np.float16)

    runtime = get_hip_runtime()
    rocblas = get_rocblas()
    library = None
    from hipengine.kernels.hip_gfx1100.moe.router import _router_library

    library = _router_library()

    bufs: list = []

    def alloc(nbytes: int) -> int:
        ptr = malloc(nbytes, runtime=runtime)
        bufs.append(ptr)
        return ptr

    h_dev = alloc(hidden.nbytes)
    w_dev = alloc(weight.nbytes)
    w16_dev = alloc(weight_f16.nbytes)
    a32_dev = alloc(max_tokens * HIDDEN * 4)
    a16_dev = alloc(max_tokens * HIDDEN * 2)
    out_dev = alloc(max_tokens * EXPERTS * 4)

    copy_host_to_device(h_dev, host_array_ptr(np.ascontiguousarray(hidden)), runtime=runtime)
    copy_host_to_device(w_dev, host_array_ptr(np.ascontiguousarray(weight)), runtime=runtime)
    copy_host_to_device(
        w16_dev, host_array_ptr(np.ascontiguousarray(weight_f16)), runtime=runtime
    )

    # Downcast leg's error: what the f16 route pays before any GEMM happens.
    w16_ref_maxabs = float(np.max(np.abs(weight_f16.astype(np.float64) - weight)))

    rows: list[dict] = []
    try:
        for tokens in SWEEP:
            n = tokens * HIDDEN
            ref = bench.reference(hidden[:tokens], weight)
            assert np.all(np.isfinite(ref)) and np.max(np.abs(ref)) > 0

            def timed(fn, repeats: int) -> float:
                fn()
                runtime.device_synchronize()
                start = time.perf_counter()
                for _ in range(repeats):
                    fn()
                runtime.device_synchronize()
                return (time.perf_counter() - start) / repeats * 1e3

            def readback() -> np.ndarray:
                out = np.empty((tokens, EXPERTS), dtype=np.float32)
                copy_device_to_host(host_array_ptr(out), out_dev, out.nbytes, runtime=runtime)
                return out

            candidates = []

            # A: the production route for this token count.
            def make_a(tokens=tokens):
                if tokens >= TT16_MIN:
                    return lambda: qwen35_router_logits_bf16_f32w_token_tile_16(
                        h_dev.ptr, w_dev.ptr, out_dev.ptr, tokens, HIDDEN, EXPERTS,
                        threads=TT16_THREADS, stream=0, library=library, runtime=runtime,
                    )
                return lambda: qwen35_router_logits_bf16_f32w(
                h_dev.ptr, w_dev.ptr, out_dev.ptr, tokens, HIDDEN, EXPERTS,
                stream=0, library=library, runtime=runtime,
            )

            name_a = "A tt16/128 (production)" if tokens >= TT16_MIN else "A' untiled 512 (production)"
            candidates.append((name_a, make_a(), None))

            # B: bit-exact upcast + F32 SGEMM -- pair cost is the real number.
            candidates.append((
                "B bf16_to_f32 + sgemm",
                lambda: (
                    bf16_to_f32(h_dev.ptr, a32_dev.ptr, n, stream=0, runtime=runtime),
                    rocblas.sgemm_rowmajor_nt(
                        a32_dev.ptr, w_dev.ptr, out_dev.ptr,
                        rows=tokens, in_features=HIDDEN, out_features=EXPERTS, stream=0,
                    ),
                ),
                None,
            ))

            # B': SGEMM alone (diagnostic for a future prescale-writes-f32).
            # Needs the upcast to exist -- reuse the buffer already filled.
            def b_prime(tokens=tokens, n=n):
                bf16_to_f32(h_dev.ptr, a32_dev.ptr, n, stream=0, runtime=runtime)
                runtime.device_synchronize()
                return lambda: rocblas.sgemm_rowmajor_nt(
                    a32_dev.ptr, w_dev.ptr, out_dev.ptr,
                    rows=tokens, in_features=HIDDEN, out_features=EXPERTS, stream=0,
                )

            # C: the row's downcast route -- bf16 -> f16 cast + f16-input GEMM.
            def make_c(n=n, tokens=tokens):
                def call():
                    bf16_to_fp16(h_dev.ptr, a16_dev.ptr, n, stream=0, runtime=runtime)
                    rocblas.gemm_ex_rowmajor_nt_fp16_f32_out(
                        a16_dev.ptr, w16_dev.ptr, out_dev.ptr,
                        rows=tokens, in_features=HIDDEN, out_features=EXPERTS, stream=0,
                    )
                return call

            # Time B' after priming the cast so only the GEMM is in the loop.
            reps = repeats(tokens)
            for name, fn, _ in list(candidates):
                if name.startswith("B "):
                    try:
                        ms = timed(fn, reps)
                        out = readback()
                        err = float(np.max(np.abs(out - ref)))
                        assert np.all(np.isfinite(out))
                        rows.append({"tokens": tokens, "candidate": name, "ms": ms, "maxabs": err})
                        print(f"{tokens:5d} {name:32s} {ms:9.4f} ms  maxabs {err:.3e}")
                    except Exception as exc:
                        print(f"{tokens:5d} {name:32s} FAIL {type(exc).__name__}: {exc}")
                        rows.append({"tokens": tokens, "candidate": name, "fail": f"{type(exc).__name__}: {exc}"})
                else:
                    try:
                        ms = timed(fn, reps)
                        out = readback()
                        err = float(np.max(np.abs(out - ref)))
                        assert np.all(np.isfinite(out))
                        rows.append({"tokens": tokens, "candidate": name, "ms": ms, "maxabs": err})
                        print(f"{tokens:5d} {name:32s} {ms:9.4f} ms  maxabs {err:.3e}")
                    except Exception as exc:
                        print(f"{tokens:5d} {name:32s} FAIL {type(exc).__name__}: {exc}")
                        rows.append({"tokens": tokens, "candidate": name, "fail": f"{type(exc).__name__}: {exc}"})

            # B' diagnostic.
            try:
                fn = b_prime()
                ms = timed(fn, reps)
                out = readback()
                err = float(np.max(np.abs(out - ref)))
                rows.append({"tokens": tokens, "candidate": "B' sgemm alone", "ms": ms, "maxabs": err})
                print(f"{tokens:5d} {'B' + chr(39) + ' sgemm alone':32s} {ms:9.4f} ms  maxabs {err:.3e}")
            except Exception as exc:
                rows.append({"tokens": tokens, "candidate": "B' sgemm alone", "fail": f"{type(exc).__name__}: {exc}"})
                print(f"{tokens:5d} B' sgemm alone FAIL {type(exc).__name__}: {exc}")

            # C diagnostic.
            try:
                fn = make_c()
                ms = timed(fn, reps)
                out = readback()
                err = float(np.max(np.abs(out - ref)))
                rows.append({"tokens": tokens, "candidate": "C f16 downcast route", "ms": ms, "maxabs": err})
                print(f"{tokens:5d} {'C f16 downcast route':32s} {ms:9.4f} ms  maxabs {err:.3e}")
            except Exception as exc:
                rows.append({"tokens": tokens, "candidate": "C f16 downcast route", "fail": f"{type(exc).__name__}: {exc}"})
                print(f"{tokens:5d} C f16 downcast route FAIL {type(exc).__name__}: {exc}")

        from scripts.gemma4_campaign_bench import _gpu_name

        artifact = {
            "kind": "gemma4-p8-router-gemm-screen",
            "gpu": _gpu_name(),
            "shape": {"hidden": HIDDEN, "experts": EXPERTS},
            "repeats": {str(t): repeats(t) for t in SWEEP},
            "weight_f16_downcast_maxabs": w16_ref_maxabs,
            "rows": rows,
        }
        out_path = (
            sys.argv[1]
            if len(sys.argv) > 1
            else "benchmarks/results/2026-09-30-gemma4-p8-router-gemm-screen.json"
        )
        Path(out_path).write_text(json.dumps(artifact, indent=2) + "\n")
        print(f"\nartifact written: {out_path}")
    finally:
        for ptr in reversed(bufs):
            free(ptr, runtime=runtime)


if __name__ == "__main__":
    main()