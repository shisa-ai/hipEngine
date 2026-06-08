"""Raw-pointer wrappers for the GGUF Q4_K batched WMMA prefill kernels.

This module owns the C ABI exports defined in ``gguf_q4_k_prefill.hip``
(see docs/GGUF.md "P8: real batched prefill GEMM" for the wider plan).
The single-output kernel is a real GEMM-style batched WMMA prefill: one
wave32 block computes a TM x TN output tile via
``__builtin_amdgcn_wmma_f32_16x16x16_f16_w32``, with raw GGUF Q4_K
``block_q4_K`` dequant in the inner K-loop.

The dual variant mirrors ``awq_fusedw4_prefill_dual_fp16_kernel``'s grid
split for dense gate+up: the first half of x-tiles writes A/gate and the
second half writes B/up. hipENGINE's GGUF dense pair ABI uses one shared
activation pointer and matching output feature count for both sides.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("gguf_q4_k_prefill.hip")
_OUTPUT_NAME = "gguf_q4_k_prefill.so"
_Q4_K_BLOCK = 256

# Allowed (tile_m, tile_n) for the WMMA prefill kernels. Mirrors the PARO
# fusedw4 prefill tile set and the Q8_0 WMMA prefill wrapper.
_ALLOWED_TILES = {
    (16, 16),
    (16, 32),
    (32, 16),
    (32, 32),
    (64, 16),
    (64, 32),
}


def plan_gguf_q4_k_prefill_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="gguf_q4_k_prefill",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
    )


def build_gguf_q4_k_prefill(
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
        family="gguf_q4_k_prefill",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def _symbol(variant: str) -> str:
    return f"hipengine_gguf_q4_k_{variant}"


def _default_tiles(rows: int, out_features: int) -> tuple[int, int]:
    """Heuristic default for (tile_m, tile_n) when caller does not override."""

    tile_n = 32 if rows >= 32 else 16
    tile_m = 32 if out_features >= 32 else 16
    return tile_m, tile_n


def _resolve_tiles(rows: int, out_features: int, tile_m: int | None, tile_n: int | None) -> tuple[int, int]:
    if tile_m is None or tile_n is None:
        tm_def, tn_def = _default_tiles(rows, out_features)
        tile_m = tm_def if tile_m is None else tile_m
        tile_n = tn_def if tile_n is None else tile_n
    if (tile_m, tile_n) not in _ALLOWED_TILES:
        allowed = ", ".join(f"({m}, {n})" for m, n in sorted(_ALLOWED_TILES))
        raise ValueError(
            f"tile (tile_m={tile_m}, tile_n={tile_n}) is not supported. "
            f"Supported tiles: {allowed}"
        )
    return tile_m, tile_n


def _validate_shape(rows: int, in_features: int, out_features: int) -> None:
    if rows <= 0:
        raise ValueError("rows must be positive")
    if in_features <= 0:
        raise ValueError("in_features must be positive")
    if out_features <= 0:
        raise ValueError("out_features must be positive")
    if in_features % _Q4_K_BLOCK != 0:
        raise ValueError("in_features must be divisible by GGUF Q4_K block size 256")


def _launch(
    symbol: str,
    x_ptr: int,
    qweight_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    tile_m: int | None = None,
    tile_n: int | None = None,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    _validate_shape(rows, in_features, out_features)
    tile_m, tile_n = _resolve_tiles(rows, out_features, tile_m, tile_n)
    library = library or build_gguf_q4_k_prefill(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, symbol)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
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
        ctypes.c_int64(tile_m),
        ctypes.c_int64(tile_n),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def _launch_dual(
    symbol: str,
    x_ptr: int,
    qweight_a_ptr: int,
    qweight_b_ptr: int,
    out_a_ptr: int,
    out_b_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    tile_m: int | None = None,
    tile_n: int | None = None,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    _validate_shape(rows, in_features, out_features)
    tile_m, tile_n = _resolve_tiles(rows, out_features, tile_m, tile_n)
    library = library or build_gguf_q4_k_prefill(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, symbol)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(x_ptr),
        ctypes.c_void_p(qweight_a_ptr),
        ctypes.c_void_p(qweight_b_ptr),
        ctypes.c_void_p(out_a_ptr),
        ctypes.c_void_p(out_b_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features),
        ctypes.c_int64(tile_m),
        ctypes.c_int64(tile_n),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def _make_wrapper(variant: str):
    sym = _symbol(variant)

    def wrapper(*args, **kwargs) -> None:
        _launch(sym, *args, **kwargs)

    wrapper.__name__ = f"gguf_q4_k_{variant}"
    wrapper.__qualname__ = wrapper.__name__
    wrapper.__doc__ = (
        f"Launch GGUF Q4_K WMMA prefill (C symbol: {sym}). Signature: "
        "(x_ptr, qweight_ptr, out_ptr, rows, in_features, out_features, "
        "tile_m=None, tile_n=None, stream=0)."
    )
    return wrapper


def _make_dual_wrapper(variant: str):
    sym = _symbol(variant)

    def wrapper(*args, **kwargs) -> None:
        _launch_dual(sym, *args, **kwargs)

    wrapper.__name__ = f"gguf_q4_k_{variant}"
    wrapper.__qualname__ = wrapper.__name__
    wrapper.__doc__ = (
        f"Launch GGUF Q4_K dual WMMA prefill (C symbol: {sym}). Signature: "
        "(x_ptr, qweight_a_ptr, qweight_b_ptr, out_a_ptr, out_b_ptr, rows, "
        "in_features, out_features, tile_m=None, tile_n=None, stream=0)."
    )
    return wrapper


# Single-output dtype matrix. Names mirror Q8_0 WMMA prefill so dispatch can
# swap prefill_* -> wmma_prefill_* by string prefix when raw Q4_K is available.
gguf_q4_k_wmma_prefill_bf16_bf16_out = _make_wrapper("wmma_prefill_bf16_bf16_out")
gguf_q4_k_wmma_prefill_bf16_fp16_out = _make_wrapper("wmma_prefill_bf16_fp16_out")
gguf_q4_k_wmma_prefill_bf16_f32_out = _make_wrapper("wmma_prefill_bf16_f32_out")
gguf_q4_k_wmma_prefill_fp16_bf16_out = _make_wrapper("wmma_prefill_fp16_bf16_out")
gguf_q4_k_wmma_prefill_fp16_fp16_out = _make_wrapper("wmma_prefill_fp16_fp16_out")
gguf_q4_k_wmma_prefill_fp16_f32_out = _make_wrapper("wmma_prefill_fp16_f32_out")
gguf_q4_k_wmma_prefill_f32_bf16_out = _make_wrapper("wmma_prefill_f32_bf16_out")
gguf_q4_k_wmma_prefill_f32_fp16_out = _make_wrapper("wmma_prefill_f32_fp16_out")
gguf_q4_k_wmma_prefill_f32_f32_out = _make_wrapper("wmma_prefill_f32_f32_out")

# GGUF runtime pair fast path uses BF16 hidden activations and BF16 outputs.
gguf_q4_k_wmma_prefill_dual_bf16_bf16_out = _make_dual_wrapper(
    "wmma_prefill_dual_bf16_bf16_out"
)


_WRAPPERS = {
    "wmma_prefill_bf16_bf16_out": gguf_q4_k_wmma_prefill_bf16_bf16_out,
    "wmma_prefill_bf16_fp16_out": gguf_q4_k_wmma_prefill_bf16_fp16_out,
    "wmma_prefill_bf16_f32_out": gguf_q4_k_wmma_prefill_bf16_f32_out,
    "wmma_prefill_fp16_bf16_out": gguf_q4_k_wmma_prefill_fp16_bf16_out,
    "wmma_prefill_fp16_fp16_out": gguf_q4_k_wmma_prefill_fp16_fp16_out,
    "wmma_prefill_fp16_f32_out": gguf_q4_k_wmma_prefill_fp16_f32_out,
    "wmma_prefill_f32_bf16_out": gguf_q4_k_wmma_prefill_f32_bf16_out,
    "wmma_prefill_f32_fp16_out": gguf_q4_k_wmma_prefill_f32_fp16_out,
    "wmma_prefill_f32_f32_out": gguf_q4_k_wmma_prefill_f32_f32_out,
}

_DUAL_WRAPPERS = {
    "wmma_prefill_dual_bf16_bf16_out": gguf_q4_k_wmma_prefill_dual_bf16_bf16_out,
}


def register_gguf_q4_k_prefill_kernels(*, replace: bool = True) -> None:
    """Register raw-Q4_K WMMA prefill wrappers in the global registry."""

    for variant, fn in _WRAPPERS.items():
        register(
            KernelKey("hip_gfx1100", "linear", "gguf_q4_k", variant),
            fn,
            replace=replace,
        )
    for variant, fn in _DUAL_WRAPPERS.items():
        register(
            KernelKey("hip_gfx1100", "linear", "gguf_q4_k", variant),
            fn,
            replace=replace,
        )


register_gguf_q4_k_prefill_kernels()


__all__ = [
    "_ALLOWED_TILES",
    "_default_tiles",
    "build_gguf_q4_k_prefill",
    "plan_gguf_q4_k_prefill_build",
    "register_gguf_q4_k_prefill_kernels",
    "gguf_q4_k_wmma_prefill_bf16_bf16_out",
    "gguf_q4_k_wmma_prefill_bf16_fp16_out",
    "gguf_q4_k_wmma_prefill_bf16_f32_out",
    "gguf_q4_k_wmma_prefill_fp16_bf16_out",
    "gguf_q4_k_wmma_prefill_fp16_fp16_out",
    "gguf_q4_k_wmma_prefill_fp16_f32_out",
    "gguf_q4_k_wmma_prefill_f32_bf16_out",
    "gguf_q4_k_wmma_prefill_f32_fp16_out",
    "gguf_q4_k_wmma_prefill_f32_f32_out",
    "gguf_q4_k_wmma_prefill_dual_bf16_bf16_out",
]
