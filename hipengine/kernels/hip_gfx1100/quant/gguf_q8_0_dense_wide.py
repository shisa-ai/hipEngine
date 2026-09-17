"""Raw-pointer wrapper for the wide-row GGUF Q8_0 prefill GEMM.

This module owns the C ABI export defined in ``gguf_q8_0_dense_wide.hip``. The
kernel is a 128-column x 256-row WMMA GEMM that stages a dequantized weight tile
and a converted activation tile in LDS, so each weight byte is read from global
memory once per 256 rows instead of once per 4-32 rows. See the header of the
``.hip`` file for the port provenance and the tiling.

The ``f32`` variants convert the activation during LDS staging and are the
registered production default for the certified lane (layers 16-47 of
``gguf_ud_q4_k_xl`` on ``hip_gfx1151``, gated in ``docs/EXECUTION-PROFILES.md``
rather than by strict bit parity). The ``f16in`` variants take an activation
already converted by :func:`f32_to_f16`, for a caller that hoists that conversion
out of the K loop; they are measurement surface, not a bound route.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.kernels import launch_census
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("gguf_q8_0_dense_wide.hip")
_OUTPUT_NAME = "gguf_q8_0_dense_wide.so"
_FAMILY = "gguf_q8_0_dense_wide"

# The kernel stages K in 64-element tiles, so in_features must be a multiple of
# 64 (Q8_0 alone only requires 32). Q8_0 model shapes satisfy this.
K_TILE = 64


def plan_gguf_q8_0_dense_wide_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family=_FAMILY,
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
    )


def build_gguf_q8_0_dense_wide(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
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


def _launch(
    symbol: str,
    x_ptr: int,
    qweight_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    threads: int = 256,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    if rows <= 0:
        raise ValueError("rows must be positive")
    if in_features <= 0 or in_features % K_TILE != 0:
        raise ValueError(f"in_features must be a positive multiple of {K_TILE}")
    if out_features <= 0:
        raise ValueError("out_features must be positive")

    library = library or build_gguf_q8_0_dense_wide(load=True)
    runtime = runtime or get_hip_runtime()
    # The launch census is the instrument a route-execution claim is checked
    # with, so this family has to record: a census that cannot see the kernel
    # cannot distinguish "did not run" from "not counted".
    launch_census.record_launch("gguf_q8_0", symbol, rows, in_features, out_features)
    fn = getattr(library, symbol)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(x_ptr),
        ctypes.c_void_p(qweight_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features),
        ctypes.c_int64(threads),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


_CAST_SYMBOL = "hipengine_gguf_q8_0_f32_to_f16"


def f32_to_f16(
    x_ptr: int,
    out_ptr: int,
    n: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Convert ``n`` f32 activation values to f16, in place of the kernel's own
    staging conversion.

    Not recorded in the launch census: the census covers dense GGUF *linears*,
    and a conversion pass is not one. A route that consumes this reports its own
    launch count and timing. The output must hold ``n`` f16 values in a
    16-byte-aligned buffer, which is what the ``*_f16in_*`` variants require.
    """

    if n <= 0:
        raise ValueError("n must be positive")
    library = library or build_gguf_q8_0_dense_wide(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _CAST_SYMBOL)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(x_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(n),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def _make_wrapper(symbol: str, variant: str):
    def wrapper(*args, **kwargs) -> None:
        _launch(symbol, *args, **kwargs)

    wrapper.__name__ = f"gguf_q8_0_dense_wide_{variant}"
    wrapper.__qualname__ = wrapper.__name__
    wrapper.__doc__ = (
        f"Launch the wide-row Q8_0 prefill GEMM (C symbol: {symbol}). "
        "Signature: (x_ptr, qweight_ptr, out_ptr, rows, in_features, "
        "out_features, threads=256, stream=0)."
    )
    return wrapper


# The tile family. The name records (columns x rows) per block; see the
# ``.hip`` header for the weight-traffic and register-pressure tradeoff. The
# ``f16in`` variants take a pre-converted f16 activation instead of f32, for a
# caller that hoists the conversion out of the K loop with :func:`f32_to_f16`.
_VARIANTS = {
    "dense_wide256_f32_f32_out": "hipengine_gguf_q8_0_dense_wide256_f32_f32_out",
    "dense_wide64x256_f32_f32_out": "hipengine_gguf_q8_0_dense_wide64x256_f32_f32_out",
    "dense_wide128x128_f32_f32_out": "hipengine_gguf_q8_0_dense_wide128x128_f32_f32_out",
    "dense_wide64x128_f32_f32_out": "hipengine_gguf_q8_0_dense_wide64x128_f32_f32_out",
    "dense_wide256_f16in_f32_f32_out": (
        "hipengine_gguf_q8_0_dense_wide256_f16in_f32_f32_out"
    ),
    "dense_wide128x128_f16in_f32_f32_out": (
        "hipengine_gguf_q8_0_dense_wide128x128_f16in_f32_f32_out"
    ),
}

_WRAPPERS = {
    variant: _make_wrapper(symbol, variant) for variant, symbol in _VARIANTS.items()
}

gguf_q8_0_dense_wide256_f32_f32_out = _WRAPPERS["dense_wide256_f32_f32_out"]
gguf_q8_0_dense_wide256_f16in_f32_f32_out = _WRAPPERS[
    "dense_wide256_f16in_f32_f32_out"
]


def register_gguf_q8_0_dense_wide_kernels(*, replace: bool = True) -> None:
    for variant, wrapper in _WRAPPERS.items():
        register(
            KernelKey("hip_gfx1100", "linear", "gguf_q8_0", variant),
            wrapper,
            replace=replace,
        )


register_gguf_q8_0_dense_wide_kernels()


__all__ = [
    "K_TILE",
    "build_gguf_q8_0_dense_wide",
    "f32_to_f16",
    "gguf_q8_0_dense_wide256_f16in_f32_f32_out",
    "gguf_q8_0_dense_wide256_f32_f32_out",
    "plan_gguf_q8_0_dense_wide_build",
    "register_gguf_q8_0_dense_wide_kernels",
]
