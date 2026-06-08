"""Wrappers for dense GGUF Q8_0 T16 WMMA prefill (P10.B4).

These kernels take the resident T16 tile layout from
``hipengine/quant/gguf_t16.py`` (``tiles[out_tiles16, blocks_per_row, 544]``)
and produce dense BF16 / FP16 / F32 outputs. Tile selection mirrors the
raw Q8_0 prefill kernel so the shape-aware ``_default_tiles`` policy in
``hipengine/runtime/gguf_linear.py`` can reuse the existing decision tree.
"""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Mapping

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("gguf_q8_0_t16_prefill.hip")
_OUTPUT_NAME = "gguf_q8_0_t16_prefill.so"

_Q8_0_QK = 32
_T16_COLS = 16

_ALLOWED_TILES = {(16, 16), (32, 16), (16, 32), (32, 32), (64, 16), (64, 32)}

_VARIANTS: tuple[str, ...] = (
    "wmma_prefill_bf16_bf16_out",
    "wmma_prefill_bf16_fp16_out",
    "wmma_prefill_bf16_f32_out",
    "wmma_prefill_fp16_bf16_out",
    "wmma_prefill_fp16_fp16_out",
    "wmma_prefill_fp16_f32_out",
    "wmma_prefill_f32_bf16_out",
    "wmma_prefill_f32_fp16_out",
    "wmma_prefill_f32_f32_out",
)


def plan_gguf_q8_0_t16_prefill_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "prefill",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="gguf_q8_0_t16_prefill",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        extra_flags=("-mcumode",),
        output_name=_OUTPUT_NAME,
    )


def build_gguf_q8_0_t16_prefill(
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
        family="gguf_q8_0_t16_prefill",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        extra_flags=("-mcumode",),
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def _default_tiles(rows: int, in_features: int, out_features: int) -> tuple[int, int]:
    """Reuse the raw Q8_0 prefill tile policy (see ``gguf_q8_0_prefill.py``).

    Shape-driven decisions:
    * ``out_features <= 512`` -> ``tile_m = 16``
    * ``in_features >= 4096`` and ``out_features >= 2048`` -> ``tile_m = 64``
    * ``in_features <= 2048`` and ``out_features >= 4096`` -> ``tile_m = 16``
    * ``out_features >= 32`` -> ``tile_m = 32``
    * else ``tile_m = 16``

    ``tile_n`` is 32 for ``rows >= 32`` (full prefill bulks) and 16 otherwise
    (small rows, e.g. decode warmup chunks).
    """

    tile_n = 32 if rows >= 32 else 16
    if out_features <= 512:
        tile_m = 16
    elif in_features >= 4096 and out_features >= 2048:
        tile_m = 64
    elif in_features <= 2048 and out_features >= 4096:
        tile_m = 16
    elif out_features >= 32:
        tile_m = 32
    else:
        tile_m = 16
    return tile_m, tile_n


def _symbol_for_variant(variant: str) -> str:
    return f"hipengine_gguf_q8_0_t16_{variant}"


def _make_wrapper(variant: str):
    symbol = _symbol_for_variant(variant)

    def wrapper(
        x_ptr: int,
        tiles_ptr: int,
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
        _launch(
            symbol,
            x_ptr,
            tiles_ptr,
            out_ptr,
            rows,
            in_features,
            out_features,
            tile_m=tile_m,
            tile_n=tile_n,
            stream=stream,
            library=library,
            runtime=runtime,
        )

    wrapper.__name__ = f"gguf_q8_0_t16_{variant}"
    wrapper.__qualname__ = wrapper.__name__
    wrapper.__doc__ = f"Launch Q8_0 T16 dense WMMA prefill ({variant})."
    return wrapper


_WRAPPER_CACHE: dict[str, object] = {variant: _make_wrapper(variant) for variant in _VARIANTS}


def __getattr__(name: str):
    if name.startswith("gguf_q8_0_t16_wmma_prefill_") and name.endswith("_out"):
        variant = name[len("gguf_q8_0_t16_") :]
        fn = _WRAPPER_CACHE.get(variant)
        if fn is not None:
            return fn
    raise AttributeError(f"module 'gguf_q8_0_t16_prefill' has no attribute {name!r}")


def _launch(
    symbol: str,
    x_ptr: int,
    tiles_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    tile_m: int | None,
    tile_n: int | None,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    if rows <= 0:
        raise ValueError("rows must be positive")
    if in_features <= 0:
        raise ValueError("in_features must be positive")
    if out_features <= 0:
        raise ValueError("out_features must be positive")
    if in_features % _Q8_0_QK != 0:
        raise ValueError("in_features must be divisible by Q8_0 block size 32")
    if out_features % _T16_COLS != 0:
        raise ValueError("out_features must be a multiple of 16")
    if tile_m is None or tile_n is None:
        tm_def, tn_def = _default_tiles(rows, in_features, out_features)
        tile_m = tm_def if tile_m is None else tile_m
        tile_n = tn_def if tile_n is None else tile_n
    if (tile_m, tile_n) not in _ALLOWED_TILES:
        allowed = ", ".join(f"({m}, {n})" for m, n in sorted(_ALLOWED_TILES))
        raise ValueError(
            f"tile (tile_m={tile_m}, tile_n={tile_n}) is not supported. "
            f"Supported tiles: {allowed}"
        )
    library = library or build_gguf_q8_0_t16_prefill(load=True)
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
        ctypes.c_void_p(tiles_ptr),
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


def register_gguf_q8_0_t16_prefill_kernels(*, replace: bool = True) -> None:
    """Register P10.B4 Q8_0 T16 dense WMMA prefill kernels.

    The variant spelling matches the rows>1 rewrite produced by
    ``_variant_for_rows`` in ``hipengine/runtime/gguf_linear.py``: the
    rows=1 ``t16_gemv_decode_<in>_<out>_out`` rewrites to
    ``t16_wmma_prefill_<in>_<out>_out`` at rows>1 when WMMA prefill is
    enabled. We register both the bare ``wmma_prefill_*`` name and the
    ``t16_wmma_prefill_*`` rewrite alias so dispatch can route on quant key
    alone.
    """

    for variant in _VARIANTS:
        fn = _WRAPPER_CACHE[variant]
        register(
            KernelKey(
                "hip_gfx1100",
                "linear",
                "gguf_q8_0_t16_v1",
                variant,
            ),
            fn,
            replace=replace,
        )
        # Alias under the ``t16_wmma_prefill_*`` rewrite name so the
        # _wmma_prefill_dispatch step in gguf_linear.py can route on the
        # T16 dispatch family without an if-branch on quant key.
        register(
            KernelKey(
                "hip_gfx1100",
                "linear",
                "gguf_q8_0_t16_v1",
                f"t16_{variant}",
            ),
            fn,
            replace=replace,
        )


register_gguf_q8_0_t16_prefill_kernels()


__all__ = [
    "build_gguf_q8_0_t16_prefill",
    "plan_gguf_q8_0_t16_prefill_build",
    "register_gguf_q8_0_t16_prefill_kernels",
] + [f"gguf_q8_0_t16_{variant}" for variant in _VARIANTS]
