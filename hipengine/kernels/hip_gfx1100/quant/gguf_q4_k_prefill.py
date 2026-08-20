"""Raw-pointer wrappers for GGUF K-quant batched WMMA prefill kernels.

This module owns the C ABI exports defined in ``gguf_q4_k_prefill.hip``
(see docs/GGUF.md "P8: real batched prefill GEMM" for the wider plan).
The single-output kernels are real GEMM-style batched WMMA prefill: one
wave32 block computes a TM x TN output tile via
``__builtin_amdgcn_wmma_f32_16x16x16_f16_w32``, with raw GGUF Q4_K
``block_q4_K``, resident Q4_K pack8, or raw ``block_q6_K`` dequant in the
inner K-loop.

The dual variant mirrors ``awq_fusedw4_prefill_dual_fp16_kernel``'s grid
split for dense gate+up: the first half of x-tiles writes A/gate and the
second half writes B/up. hipENGINE's GGUF dense pair ABI uses one shared
activation pointer and matching output feature count for both sides.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("gguf_q4_k_prefill.hip")
_OUTPUT_NAME = "gguf_q4_k_prefill.so"
_Q4_K_BLOCK = 256
_Q4_K_DENSE_WMMA_TILE_ENV = "HIPENGINE_GGUF_Q4_K_DENSE_WMMA_TILE"
_Q6_K_DENSE_WMMA_TILE_ENV = "HIPENGINE_GGUF_Q6_K_DENSE_WMMA_TILE"

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


def _q6_symbol(variant: str) -> str:
    return f"hipengine_gguf_q6_k_{variant}"


def _default_tiles(rows: int, out_features: int) -> tuple[int, int]:
    """Heuristic default for (tile_m, tile_n) when caller does not override."""

    tile_n = 32 if rows >= 32 else 16
    tile_m = 32 if out_features >= 32 else 16
    return tile_m, tile_n


def _default_q6_tiles(
    default: tuple[int, int] = (64, 16),
) -> tuple[int, int]:
    """Return the backend-selected Q6 dense tile or an explicit override."""

    value = os.environ.get(_Q6_K_DENSE_WMMA_TILE_ENV, "").strip().lower()
    if not value:
        return default
    try:
        tile_m_raw, tile_n_raw = value.split("x", 1)
        tile = int(tile_m_raw), int(tile_n_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{_Q6_K_DENSE_WMMA_TILE_ENV} must name a supported tile as MxN"
        ) from exc
    if tile not in _ALLOWED_TILES:
        allowed = ", ".join(f"{m}x{n}" for m, n in sorted(_ALLOWED_TILES))
        raise ValueError(
            f"{_Q6_K_DENSE_WMMA_TILE_ENV}={value!r} is unsupported; "
            f"expected one of: {allowed}"
        )
    return tile


def _default_q4_pack8_tiles(
    rows: int,
    in_features: int,
    out_features: int,
) -> tuple[int, int]:
    """Return the measured gfx1151 pack8 tile for a dense Q4 shape."""

    value = os.environ.get(_Q4_K_DENSE_WMMA_TILE_ENV, "").strip().lower()
    if value:
        try:
            tile_m_raw, tile_n_raw = value.split("x", 1)
            tile = int(tile_m_raw), int(tile_n_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{_Q4_K_DENSE_WMMA_TILE_ENV} must name a supported tile as MxN"
            ) from exc
        if tile not in _ALLOWED_TILES:
            allowed = ", ".join(f"{m}x{n}" for m, n in sorted(_ALLOWED_TILES))
            raise ValueError(
                f"{_Q4_K_DENSE_WMMA_TILE_ENV}={value!r} is unsupported; "
                f"expected one of: {allowed}"
            )
        return tile

    shape = int(rows), int(in_features), int(out_features)
    if shape == (512, 1024, 3072):
        return 64, 32
    if shape == (512, 3072, 12288):
        return 32, 32
    if shape == (512, 1024, 3584):
        return 16, 32
    if shape == (512, 3584, 1024):
        return 64, 16
    return 64, 16


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


def _launch_pack8(
    symbol: str,
    x_ptr: int,
    qweight_ptr: int,
    scales_ptr: int,
    mins_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    tile_m: int | None = None,
    tile_n: int | None = None,
    threads: int = 0,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    _validate_shape(rows, in_features, out_features)
    if out_features % 8:
        raise ValueError("pack8 out_features must be divisible by 8")
    if threads not in (0, 32):
        raise ValueError("pack8 WMMA threads must be 0 or 32")
    tile_m = 64 if tile_m is None else tile_m
    tile_n = 16 if tile_n is None else tile_n
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
        ctypes.c_void_p(qweight_ptr),
        ctypes.c_void_p(scales_ptr),
        ctypes.c_void_p(mins_ptr),
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


def gguf_q4_k_pack8_dual_wmma_prefill_silu_bf16_bf16_out(
    x_ptr: int,
    qweight_a_ptr: int,
    scales_a_ptr: int,
    mins_a_ptr: int,
    qweight_b_ptr: int,
    scales_b_ptr: int,
    mins_b_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch operation-complete resident-pack8 gate/up WMMA prefill + SiLU."""

    _validate_shape(rows, in_features, out_features)
    if out_features % 32:
        raise ValueError("pack8 dual WMMA+SiLU out_features must be divisible by 32")
    library = library or build_gguf_q4_k_prefill(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(
        library,
        _symbol("pack8_dual_wmma_prefill_silu_bf16_bf16_out"),
    )
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(x_ptr),
        ctypes.c_void_p(qweight_a_ptr),
        ctypes.c_void_p(scales_a_ptr),
        ctypes.c_void_p(mins_a_ptr),
        ctypes.c_void_p(qweight_b_ptr),
        ctypes.c_void_p(scales_b_ptr),
        ctypes.c_void_p(mins_b_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features),
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


def _make_pack8_wrapper(variant: str):
    sym = _symbol(variant)

    def wrapper(*args, **kwargs) -> None:
        _launch_pack8(sym, *args, **kwargs)

    wrapper.__name__ = f"gguf_q4_k_{variant}"
    wrapper.__qualname__ = wrapper.__name__
    wrapper.__doc__ = (
        f"Launch resident-pack8 GGUF Q4_K WMMA prefill (C symbol: {sym})."
    )
    return wrapper


def _make_pack8_gfx1151_wrapper(variant: str):
    sym = _symbol(variant)

    def wrapper(
        x_ptr: int,
        qweight_ptr: int,
        scales_ptr: int,
        mins_ptr: int,
        out_ptr: int,
        rows: int,
        in_features: int,
        out_features: int,
        *,
        tile_m: int | None = None,
        tile_n: int | None = None,
        threads: int = 0,
        stream: int = 0,
        library: ctypes.CDLL | None = None,
        runtime: HipRuntime | None = None,
    ) -> None:
        if tile_m is None and tile_n is None:
            tile_m, tile_n = _default_q4_pack8_tiles(
                rows,
                in_features,
                out_features,
            )
        _launch_pack8(
            sym,
            x_ptr,
            qweight_ptr,
            scales_ptr,
            mins_ptr,
            out_ptr,
            rows,
            in_features,
            out_features,
            tile_m=tile_m,
            tile_n=tile_n,
            threads=threads,
            stream=stream,
            library=library,
            runtime=runtime,
        )

    wrapper.__name__ = f"gguf_q4_k_{variant}_gfx1151"
    wrapper.__qualname__ = wrapper.__name__
    wrapper.__doc__ = (
        "Launch resident-pack8 GGUF Q4_K WMMA prefill with the measured "
        f"gfx1151 shape policy (C symbol: {sym})."
    )
    return wrapper


def _make_q6_wrapper(
    variant: str,
    *,
    default_tile: tuple[int, int] = (64, 16),
):
    sym = _q6_symbol(variant)

    def wrapper(*args, **kwargs) -> None:
        tile_m, tile_n = _default_q6_tiles(default_tile)
        kwargs.setdefault("tile_m", tile_m)
        kwargs.setdefault("tile_n", tile_n)
        _launch(sym, *args, **kwargs)

    wrapper.__name__ = f"gguf_q6_k_{variant}"
    wrapper.__qualname__ = wrapper.__name__
    wrapper.__doc__ = (
        f"Launch raw GGUF Q6_K WMMA prefill (C symbol: {sym})."
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
gguf_q4_k_pack8_wmma_prefill_bf16_bf16_out = _make_pack8_wrapper(
    "pack8_wmma_prefill_bf16_bf16_out"
)
gguf_q4_k_pack8_wmma_prefill_gfx1151_bf16_bf16_out = (
    _make_pack8_gfx1151_wrapper("pack8_wmma_prefill_bf16_bf16_out")
)

gguf_q6_k_wmma_prefill_bf16_bf16_out = _make_q6_wrapper(
    "wmma_prefill_bf16_bf16_out"
)
gguf_q6_k_wmma_prefill_16x32_bf16_bf16_out = _make_q6_wrapper(
    "wmma_prefill_bf16_bf16_out",
    default_tile=(16, 32),
)

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
    "pack8_wmma_prefill_bf16_bf16_out": (
        gguf_q4_k_pack8_wmma_prefill_bf16_bf16_out
    ),
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
    register(
        KernelKey(
            "hip_gfx1100",
            "linear_pair_silu",
            "gguf_q4_k",
            "pack8_dual_wmma_prefill_bf16_bf16_out",
        ),
        gguf_q4_k_pack8_dual_wmma_prefill_silu_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "linear",
            "gguf_q6_k",
            "wmma_prefill_bf16_bf16_out",
        ),
        gguf_q6_k_wmma_prefill_bf16_bf16_out,
        replace=replace,
    )


register_gguf_q4_k_prefill_kernels()


__all__ = [
    "_ALLOWED_TILES",
    "_default_q4_pack8_tiles",
    "_default_q6_tiles",
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
    "gguf_q4_k_pack8_dual_wmma_prefill_silu_bf16_bf16_out",
    "gguf_q4_k_pack8_wmma_prefill_bf16_bf16_out",
    "gguf_q4_k_pack8_wmma_prefill_gfx1151_bf16_bf16_out",
    "gguf_q4_k_wmma_prefill_dual_bf16_bf16_out",
    "gguf_q6_k_wmma_prefill_bf16_bf16_out",
    "gguf_q6_k_wmma_prefill_16x32_bf16_bf16_out",
]
