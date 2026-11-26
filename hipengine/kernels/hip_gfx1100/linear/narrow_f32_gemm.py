"""Narrow-output FP32 GEMM for the F32 projections of a prompt block.

The controller, sink, router and convolution projections of an Inkling prompt block have 24 to
128 outputs from 2,304 or 4,608 inputs. rocBLAS answers them at 1 to 5 TFLOP/s. This kernel keeps
the products and sums in FP32, as rocBLAS does, and changes only the order of the sum over the
inputs: a wave32 splits the inputs over its lanes and the outputs over lane groups, and the lane
partials are added in a fixed xor tree. It serves 24, 32, 64 and 128 outputs with an input count
that is a multiple of the wave's step; another shape keeps its caller's GEMM.
"""

from __future__ import annotations

import ctypes
from functools import lru_cache
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.hip import HIP_SUCCESS, capture_stream
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("narrow_f32_gemm.hip")
_FAMILY = "narrow_f32_gemm"
_OUTPUT_NAME = "narrow_f32_gemm.so"
NARROW_F32_VARIANT = "narrow_f32_f32_out"
# The outputs a kernel serves, and the multiple of the input count its wave's step needs.
_SYMBOLS = {24: "hipengine_narrow_f32_m24_f32_out", 32: "hipengine_narrow_f32_m32_f32_out",
            64: "hipengine_narrow_f32_m64_f32_out", 128: "hipengine_narrow_f32_m128_f32_out"}
_K_MULTIPLE = {24: 128, 32: 128, 64: 64, 128: 32}
# One row keeps the caller's matrix-vector product, which moves each weight once for the row
# (W7900, 4,608 inputs to 64 outputs: 20 us a row against 63 us for the kernel on any rows).
MIN_ROWS = 2


def plan_narrow_f32_gemm_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "prefill",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family=_FAMILY,
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
    )


def build_narrow_f32_gemm(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "prefill",
    dry_run: bool = False,
    load: bool = True,
    require_cached: bool = False,
) -> ctypes.CDLL | BuildArtifact:
    return build_hip(
        sources=[_SOURCE],
        family=_FAMILY,
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


@lru_cache(maxsize=4)
def _symbol(out_features: int):
    fn = getattr(build_narrow_f32_gemm(load=True), _SYMBOLS[out_features])
    fn.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int64] * 2 + [ctypes.c_void_p]
    fn.restype = ctypes.c_int
    return fn


def serves(rows: int, in_features: int, out_features: int) -> bool:
    """Whether the kernel takes this shape."""
    return (rows >= MIN_ROWS and out_features in _SYMBOLS and in_features > 0
            and in_features % _K_MULTIPLE[out_features] == 0)


def narrow_f32_gemm(x_ptr: int, weight_ptr: int, out_ptr: int, rows: int, in_features: int,
                    out_features: int, *, stream: int | None = None) -> bool:
    """out[rows, out_features] (FP32) = x[rows, in_features] @ weight[out_features, in_features].T.

    Returns False, and launches nothing, for a shape the kernel does not serve.
    """
    if not serves(rows, in_features, out_features):
        return False
    err = _symbol(out_features)(ctypes.c_void_p(x_ptr), ctypes.c_void_p(weight_ptr), ctypes.c_void_p(out_ptr),
                                rows, in_features, capture_stream() if stream is None else stream)
    if err != HIP_SUCCESS:
        raise RuntimeError(f"narrow FP32 GEMM failed with HIP error {err}")
    return True


for _backend in ("hip_gfx1100", "hip_gfx1151"):
    register(KernelKey(_backend, "dense_gemv", "f32", NARROW_F32_VARIANT), narrow_f32_gemm)
