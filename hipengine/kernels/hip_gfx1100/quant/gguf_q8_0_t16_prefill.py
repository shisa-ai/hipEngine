"""Wrappers for dense GGUF Q8_0 T16 WMMA prefill (P10.B4).

These kernels take the resident T16 tile layout from
``hipengine/quant/gguf_t16.py`` (``tiles[out_tiles16, blocks_per_row, 544]``)
and produce dense BF16 / FP16 / F32 outputs. Tile selection starts from the
raw Q8_0 prefill policy but is tuned separately for the resident T16 layout,
whose contiguous 16-column tile slabs have a different best tile balance.
"""

from __future__ import annotations

import contextlib
import ctypes
import os
from collections.abc import Iterator
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("gguf_q8_0_t16_prefill.hip")
_OUTPUT_NAME = "gguf_q8_0_t16_prefill.so"

_Q8_0_QK = 32
_T16_COLS = 16

_ALLOWED_TILES = {(16, 16), (32, 16), (16, 32), (32, 32), (64, 16), (64, 32)}

_TWO_WAVE_VARIANT = "wmma_prefill_2wave_bf16_bf16_out"
_FOUR_WAVE_VARIANT = "wmma_prefill_4wave_bf16_bf16_out"
_TWO_WAVE_ENV = "HIPENGINE_GGUF_Q8_T16_PREFILL_2WAVE"
_FOUR_WAVE_ENV = "HIPENGINE_GGUF_Q8_T16_PREFILL_4WAVE"
_wide_wave_session_enabled: bool | None = None

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
    """Tuned default for resident Q8_0 T16 WMMA prefill tiles.

    The T16 resident layout no longer matches the older raw-Q8_0 tile policy:
    synthetic qwen35moe-shape probes on GPU1 showed that the old raw defaults
    under-tiled ``in<=2048,out>=4096`` and over-tiled
    ``in>=4096,out>=2048``.  Keep the small-output and medium defaults stable,
    but use wider output tiles for the large shared/full-attention projections.
    """

    tile_n = 32 if rows >= 32 else 16
    if out_features <= 512:
        # Shared-expert gate/up (in=2048,out=512) benefits from TN16 on the
        # 768/1024-row mid/long chunks. Keep the 512-row primary gate on the
        # previous TN32 policy because the full model regressed despite the
        # isolated microbench preferring a wider TM64/16 tile.
        if rows < 32:
            tile_n = 16
            tile_m = 16
        elif rows <= 512:
            tile_m = 16
        else:
            tile_n = 16
            tile_m = 32
    elif in_features <= 512 and out_features <= 2048:
        # Shared-expert down (in=512,out=2048) also favors TN16 on the active
        # 768/1024-row chunks, while 512-row primary-gate runs stay on TN32.
        tile_m = 32
        if rows > 512:
            tile_n = 16
    elif in_features >= 4096 and out_features >= 2048:
        tile_m = 32
    elif in_features <= 2048 and out_features >= 4096:
        tile_m = 64 if rows <= 512 or (rows <= 768 and out_features < 8192) else 32
    elif out_features >= 32:
        tile_m = 32
        if rows == 1024 and in_features == 2048 and out_features == 2048:
            # The 1024-row medium-square projection benefits from TN16 in the
            # full GGUF gate; 512/768-row probes regressed and stay on TN32.
            tile_n = 16
    else:
        tile_m = 16
    return tile_m, tile_n


@contextlib.contextmanager
def q8_t16_two_wave_prefill_session(enabled: bool | None) -> Iterator[None]:
    """Temporarily scope the retained wide-wave schedule to one prefill request."""

    global _wide_wave_session_enabled
    previous = _wide_wave_session_enabled
    _wide_wave_session_enabled = None if enabled is None else bool(enabled)
    try:
        yield
    finally:
        _wide_wave_session_enabled = previous


def _wide_wave_prefill_enabled(*, default: bool) -> bool:
    """Resolve the shared GPF-5A/LCP-3 request ceiling and rollback control."""

    raw = os.environ.get(_TWO_WAVE_ENV, "").strip().lower()
    if raw:
        return raw in {"1", "true", "yes", "on"}
    if _wide_wave_session_enabled is not None:
        return _wide_wave_session_enabled
    return default


def _two_wave_prefill_applies(
    *,
    tile_m: int,
    tile_n: int,
    out_features: int,
    default: bool = False,
) -> bool:
    """Return whether the GPF-5A policy covers this tile."""

    enabled = _wide_wave_prefill_enabled(default=default)
    return enabled and tile_m in {32, 64} and tile_n == 32 and out_features >= 2048


def _four_wave_prefill_applies(
    *,
    tile_m: int,
    tile_n: int,
    out_features: int,
    default: bool = False,
) -> bool:
    """Return whether the LCP-3 four-wave schedule covers this tile.

    The `out_features == 1408` arm is the Gemma 4 26B-A4B expert gate/up
    (2 x 704): measured on the 8060S (gfx1151, 2026-10-07, in-process A/B at
    1024 routed rows: 6.83 ms four-wave vs 7.72 ms nwave, +13.2%, outputs
    identical arithmetic class) the four-wave schedule wins there too, and
    1408 is a whole multiple of the schedule's 128-column tile so it pays no
    tail waste. Only the measured width is added; every other sub-2048 width
    keeps the previous policy, and `HIPENGINE_GGUF_Q8_T16_PREFILL_2WAVE=0`
    remains the rollback for the whole wide-wave family.
    """

    raw = os.environ.get(_FOUR_WAVE_ENV, "").strip().lower()
    two_wave_raw = os.environ.get(_TWO_WAVE_ENV, "").strip().lower()
    if raw:
        enabled = raw in {"1", "true", "yes", "on"}
    elif two_wave_raw:
        # Preserve the older selector's exact meaning: explicit 2WAVE=1 asks
        # for two-wave, while 2WAVE=0 asks for production.
        enabled = False
    else:
        enabled = _wide_wave_prefill_enabled(default=default)
    covered = out_features >= 2048 or out_features == 1408
    return enabled and tile_m in {32, 64} and tile_n == 32 and covered


def _symbol_for_variant(variant: str) -> str:
    return f"hipengine_gguf_q8_0_t16_{variant}"


def _make_wrapper(
    variant: str,
    *,
    two_wave_default: bool = False,
    four_wave_default: bool = False,
):
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
        resolved_tile_m, resolved_tile_n = _default_tiles(rows, in_features, out_features)
        if tile_m is not None:
            resolved_tile_m = tile_m
        if tile_n is not None:
            resolved_tile_n = tile_n
        selected_symbol = symbol
        if variant == "wmma_prefill_bf16_bf16_out" and _four_wave_prefill_applies(
            tile_m=resolved_tile_m,
            tile_n=resolved_tile_n,
            out_features=out_features,
            default=four_wave_default,
        ):
            selected_symbol = _symbol_for_variant(_FOUR_WAVE_VARIANT)
            resolved_tile_m = 128
            f16w_ptr = (
                gguf_q8_0_t16_f16w_for(tiles_ptr, in_features, out_features)
                if resolved_tile_n == 32 else None
            )
            if f16w_ptr is not None:
                gguf_q8_0_t16_wmma_prefill_4wave_f16w_bf16_bf16_out(
                    x_ptr, tiles_ptr, f16w_ptr, out_ptr, rows, in_features, out_features,
                    stream=stream, library=library, runtime=runtime,
                )
                return
        elif variant == "wmma_prefill_bf16_bf16_out" and _two_wave_prefill_applies(
            tile_m=resolved_tile_m,
            tile_n=resolved_tile_n,
            out_features=out_features,
            default=two_wave_default,
        ):
            selected_symbol = _symbol_for_variant(_TWO_WAVE_VARIANT)
            resolved_tile_m = 64
            # The same launch over pre-converted weights, when the weight's
            # owner registered them for this shape; likewise four-wave above.
            f16w_ptr = (
                gguf_q8_0_t16_f16w_for(tiles_ptr, in_features, out_features)
                if resolved_tile_n == 32 else None
            )
            if f16w_ptr is not None:
                gguf_q8_0_t16_wmma_prefill_2wave_f16w_bf16_bf16_out(
                    x_ptr, tiles_ptr, f16w_ptr, out_ptr, rows, in_features, out_features,
                    stream=stream, library=library, runtime=runtime,
                )
                return
        _launch(
            selected_symbol,
            x_ptr,
            tiles_ptr,
            out_ptr,
            rows,
            in_features,
            out_features,
            tile_m=resolved_tile_m,
            tile_n=resolved_tile_n,
            stream=stream,
            library=library,
            runtime=runtime,
        )

    wrapper.__name__ = f"gguf_q8_0_t16_{variant}"
    wrapper.__qualname__ = wrapper.__name__
    wrapper.__doc__ = f"Launch Q8_0 T16 dense WMMA prefill ({variant})."
    return wrapper


def gguf_q8_0_t16_dual_wmma_prefill_bf16_bf16_out(
    x_ptr: int,
    tiles_a_ptr: int,
    tiles_b_ptr: int,
    out_a_ptr: int,
    out_b_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch the narrow Q8T16 dual-WMMA alpha/beta prefill leaf."""

    if rows <= 0 or in_features <= 0 or in_features % _Q8_0_QK:
        raise ValueError("rows must be positive and in_features divisible by 32")
    if out_features != _T16_COLS:
        raise ValueError("dual Q8T16 WMMA prefill requires out_features == 16")
    library = library or build_gguf_q8_0_t16_prefill(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(
        library,
        "hipengine_gguf_q8_0_t16_dual_wmma_prefill_bf16_bf16_out",
    )
    fn.argtypes = [
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
        ctypes.c_void_p(tiles_a_ptr),
        ctypes.c_void_p(tiles_b_ptr),
        ctypes.c_void_p(out_a_ptr),
        ctypes.c_void_p(out_b_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


_WRAPPER_CACHE: dict[str, object] = {variant: _make_wrapper(variant) for variant in _VARIANTS}

gguf_q8_0_t16_wmma_prefill_auto_2wave_bf16_bf16_out = _make_wrapper(
    "wmma_prefill_bf16_bf16_out",
    two_wave_default=True,
)
gguf_q8_0_t16_wmma_prefill_auto_2wave_bf16_bf16_out.__name__ = (
    "gguf_q8_0_t16_wmma_prefill_auto_2wave_bf16_bf16_out"
)
gguf_q8_0_t16_wmma_prefill_auto_2wave_bf16_bf16_out.__qualname__ = (
    gguf_q8_0_t16_wmma_prefill_auto_2wave_bf16_bf16_out.__name__
)
gguf_q8_0_t16_wmma_prefill_auto_4wave_bf16_bf16_out = _make_wrapper(
    "wmma_prefill_bf16_bf16_out",
    two_wave_default=True,
    four_wave_default=True,
)
gguf_q8_0_t16_wmma_prefill_auto_4wave_bf16_bf16_out.__name__ = (
    "gguf_q8_0_t16_wmma_prefill_auto_4wave_bf16_bf16_out"
)
gguf_q8_0_t16_wmma_prefill_auto_4wave_bf16_bf16_out.__qualname__ = (
    gguf_q8_0_t16_wmma_prefill_auto_4wave_bf16_bf16_out.__name__
)


def gguf_q8_0_t16_wmma_prefill_2wave_bf16_bf16_out(
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
    """Launch the GPF-5A two-wave activation-sharing diagnostic."""

    _launch(
        _symbol_for_variant(_TWO_WAVE_VARIANT),
        x_ptr,
        tiles_ptr,
        out_ptr,
        rows,
        in_features,
        out_features,
        tile_m=64 if tile_m is None else tile_m,
        tile_n=(32 if rows >= 32 else 16) if tile_n is None else tile_n,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_q8_0_t16_wmma_prefill_4wave_bf16_bf16_out(
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
    """Launch the LCP-3 four-wave activation-sharing diagnostic."""

    _launch(
        _symbol_for_variant(_FOUR_WAVE_VARIANT),
        x_ptr,
        tiles_ptr,
        out_ptr,
        rows,
        in_features,
        out_features,
        tile_m=128 if tile_m is None else tile_m,
        tile_n=(32 if rows >= 32 else 16) if tile_n is None else tile_n,
        stream=stream,
        library=library,
        runtime=runtime,
    )


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
    allowed_tiles = (
        {(128, 16), (128, 32)}
        if symbol == _symbol_for_variant(_FOUR_WAVE_VARIANT)
        else _ALLOWED_TILES
    )
    if (tile_m, tile_n) not in allowed_tiles:
        allowed = ", ".join(f"({m}, {n})" for m, n in sorted(allowed_tiles))
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


# --- pre-converted f16 weight tiles ("f16w") -------------------------------
#
# The multi-wave prefill kernel spends about half its time turning int8 weights
# into the half operand WMMA consumes. A weight owner with device memory to
# spare can hold that operand already converted: ``gguf_q8_0_t16_tiles_to_f16w``
# writes it from the resident T16 tiles with the kernel's own expression, and
# the owner registers the result against the tiles pointer it was built from.
# The two-wave ``wmma_prefill_bf16_bf16_out`` launch then reads it instead of
# decoding. The table is keyed by the tiles pointer because that is the only
# operand the registered launch signature carries; the owner that registers an
# entry must unregister it before freeing either buffer.
_F16W_TILES: dict[int, tuple[int, int, int]] = {}


def gguf_q8_0_t16_f16w_bytes(in_features: int, out_features: int) -> int:
    """Bytes of the f16w tiles for one ``out_features x in_features`` weight."""

    if in_features <= 0 or in_features % _Q8_0_QK or out_features <= 0 or out_features % _T16_COLS:
        raise ValueError(
            "f16w tiles need in_features divisible by 32 and out_features divisible by 16, "
            f"got in_features={in_features}, out_features={out_features}"
        )
    return int(in_features) * int(out_features) * 2


def gguf_q8_0_t16_tiles_to_f16w(
    tiles_ptr: int,
    f16w_ptr: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Write the f16w tiles for one weight from its resident T16 tiles."""

    gguf_q8_0_t16_f16w_bytes(in_features, out_features)
    library = library or build_gguf_q8_0_t16_prefill(load=True)
    runtime = runtime or get_hip_runtime()
    fn = library.hipengine_gguf_q8_0_t16_tiles_to_f16w
    fn.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(tiles_ptr), ctypes.c_void_p(f16w_ptr),
        ctypes.c_int64(in_features), ctypes.c_int64(out_features), ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def register_gguf_q8_0_t16_f16w(
    tiles_ptr: int, f16w_ptr: int, in_features: int, out_features: int
) -> None:
    """Make the wide-wave prefill read ``f16w_ptr`` for the weight at ``tiles_ptr``."""

    gguf_q8_0_t16_f16w_bytes(in_features, out_features)
    if not tiles_ptr or not f16w_ptr:
        raise ValueError("f16w registration needs non-null tiles and f16w pointers")
    _F16W_TILES[int(tiles_ptr)] = (int(f16w_ptr), int(in_features), int(out_features))


def unregister_gguf_q8_0_t16_f16w(tiles_ptr: int) -> None:
    """Drop the f16w entry for ``tiles_ptr``; a missing entry is not an error."""

    _F16W_TILES.pop(int(tiles_ptr), None)


def gguf_q8_0_t16_f16w_for(tiles_ptr: int, in_features: int, out_features: int) -> int | None:
    """The registered f16w pointer for this weight and shape, or ``None``."""

    entry = _F16W_TILES.get(int(tiles_ptr))
    if entry is None or entry[1:] != (int(in_features), int(out_features)):
        return None
    return entry[0]


def _launch_f16w(
    symbol: str,
    x_ptr: int,
    tiles_ptr: int,
    f16w_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    if rows <= 0:
        raise ValueError("rows must be positive")
    gguf_q8_0_t16_f16w_bytes(in_features, out_features)
    library = library or build_gguf_q8_0_t16_prefill(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, symbol)
    fn.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(x_ptr), ctypes.c_void_p(tiles_ptr), ctypes.c_void_p(f16w_ptr),
        ctypes.c_void_p(out_ptr), ctypes.c_int64(rows), ctypes.c_int64(in_features),
        ctypes.c_int64(out_features), ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def gguf_q8_0_t16_wmma_prefill_2wave_f16w_bf16_bf16_out(
    x_ptr: int,
    tiles_ptr: int,
    f16w_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Two-wave Q8_0 T16 WMMA prefill over pre-converted f16w tiles.

    Bit-identical to ``gguf_q8_0_t16_wmma_prefill_2wave_bf16_bf16_out`` at
    ``tile_n == 32``: the same grid, accumulation order and repair path, with
    the weight operand loaded instead of decoded. ``tiles_ptr`` is still read,
    by the non-finite repair.
    """

    _launch_f16w(
        "hipengine_gguf_q8_0_t16_wmma_prefill_2wave_f16w_bf16_bf16_out",
        x_ptr, tiles_ptr, f16w_ptr, out_ptr, rows, in_features, out_features,
        stream=stream, library=library, runtime=runtime,
    )


def gguf_q8_0_t16_wmma_prefill_4wave_f16w_bf16_bf16_out(
    x_ptr: int,
    tiles_ptr: int,
    f16w_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Four-wave form of the f16w launch.

    Bit-identical to ``gguf_q8_0_t16_wmma_prefill_4wave_bf16_bf16_out`` at
    ``tile_n == 32``.
    """

    _launch_f16w(
        "hipengine_gguf_q8_0_t16_wmma_prefill_4wave_f16w_bf16_bf16_out",
        x_ptr, tiles_ptr, f16w_ptr, out_ptr, rows, in_features, out_features,
        stream=stream, library=library, runtime=runtime,
    )


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

    register(
        KernelKey(
            "hip_gfx1100",
            "linear_pair",
            "gguf_q8_0_t16_v1",
            "t16_dual_wmma_prefill_bf16_bf16_out",
        ),
        gguf_q8_0_t16_dual_wmma_prefill_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "linear",
            "gguf_q8_0_t16_v1",
            _TWO_WAVE_VARIANT,
        ),
        gguf_q8_0_t16_wmma_prefill_2wave_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "linear",
            "gguf_q8_0_t16_v1",
            _FOUR_WAVE_VARIANT,
        ),
        gguf_q8_0_t16_wmma_prefill_4wave_bf16_bf16_out,
        replace=replace,
    )

    for variant in _VARIANTS:
        fn = (
            gguf_q8_0_t16_wmma_prefill_auto_2wave_bf16_bf16_out
            if variant == "wmma_prefill_bf16_bf16_out"
            else _WRAPPER_CACHE[variant]
        )
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
    "gguf_q8_0_t16_dual_wmma_prefill_bf16_bf16_out",
    "plan_gguf_q8_0_t16_prefill_build",
    "register_gguf_q8_0_t16_prefill_kernels",
    "q8_t16_two_wave_prefill_session",
    "gguf_q8_0_t16_wmma_prefill_2wave_bf16_bf16_out",
    "gguf_q8_0_t16_wmma_prefill_2wave_f16w_bf16_bf16_out",
    "gguf_q8_0_t16_f16w_bytes",
    "gguf_q8_0_t16_f16w_for",
    "gguf_q8_0_t16_tiles_to_f16w",
    "register_gguf_q8_0_t16_f16w",
    "unregister_gguf_q8_0_t16_f16w",
    "gguf_q8_0_t16_wmma_prefill_4wave_bf16_bf16_out",
    "gguf_q8_0_t16_wmma_prefill_4wave_f16w_bf16_bf16_out",
    "gguf_q8_0_t16_wmma_prefill_auto_2wave_bf16_bf16_out",
    "gguf_q8_0_t16_wmma_prefill_auto_4wave_bf16_bf16_out",
] + [f"gguf_q8_0_t16_{variant}" for variant in _VARIANTS]
