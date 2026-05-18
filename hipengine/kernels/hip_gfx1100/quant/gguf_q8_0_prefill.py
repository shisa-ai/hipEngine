"""Raw-pointer wrappers for the GGUF Q8_0 batched WMMA prefill kernel.

This module owns the C ABI exports defined in ``gguf_q8_0_prefill.hip``
(see docs/GGUF.md \"P8: real batched prefill GEMM\" for the wider plan).
The kernel is a real GEMM-style batched WMMA prefill: one wave32 block
computes a TM x TN output tile via
``__builtin_amdgcn_wmma_f32_16x16x16_f16_w32``, with Q8_0 dequant in the
inner K-loop. It replaces the decode-shaped ``gguf_q8_0_prefill_*`` GEMV
aliases on the rows > 1 path; the runtime dispatch in
``hipengine.runtime.gguf_linear`` opts in via a separate registry key
family (``wmma_prefill_*``).
"""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("gguf_q8_0_prefill.hip")
_OUTPUT_NAME = "gguf_q8_0_prefill.so"

# Allowed (tile_m, tile_n) for the WMMA prefill kernel. Mirrors the
# PARO fusedw4 prefill tile set. See gguf_q8_0_prefill.hip.
_ALLOWED_TILES = {
    (16, 16),
    (16, 32),
    (32, 16),
    (32, 32),
    (64, 16),
    (64, 32),
}


def plan_gguf_q8_0_prefill_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="gguf_q8_0_prefill",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
    )


def build_gguf_q8_0_prefill(
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
        family="gguf_q8_0_prefill",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def _symbol(variant: str) -> str:
    return f"hipengine_gguf_q8_0_{variant}"


def _default_tiles(rows: int, in_features: int, out_features: int) -> tuple[int, int]:
    """Heuristic default for (tile_m, tile_n) when the caller does not override.

    P9.C1 tuning (microbench on RX 7900 XTX / gfx1100, rows=512, BF16/BF16)
    is shape-specific; ``out_features`` alone was too coarse for qwen35moe:

    * ``(in=2048, out=8192)`` (linear-attention qkv and full-attn q+gate):
      ``(16,32)`` is the fastest measured tile (`~0.54 ms` synthetic). The
      previous broad ``out>=4096 -> (64,32)`` rule over-tiled this shape.
    * ``(in=2048, out=4096)`` (linear-attention gate): ``(16,32)`` slightly
      wins over ``(64,32)`` and is retained to keep the large-input family
      consistent.
    * ``(in=4096, out=2048)`` (ssm/shared down): ``(64,32)`` is best.
    * ``out<=512`` (full-attn k/v): ``(16,32)`` is best.
    * Other medium shapes keep the stable P8/P9 ``(32,32)`` default.
    * ``rows < 32``: drop ``tile_n`` to 16 (the kernel still launches but
      the bigger TN under-utilises the WMMA tile).

    See ``tests/test_gguf_q8_0_wmma_prefill.py`` for dispatch pinning tests.
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
    if rows <= 0:
        raise ValueError("rows must be positive")
    if in_features <= 0:
        raise ValueError("in_features must be positive")
    if out_features <= 0:
        raise ValueError("out_features must be positive")
    if in_features % 32 != 0:
        raise ValueError("in_features must be divisible by Q8_0 block size 32")
    if tile_m is None or tile_n is None:
        tm_def, tn_def = _default_tiles(rows, in_features, out_features)
        tile_m = tm_def if tile_m is None else tile_m
        tile_n = tn_def if tile_n is None else tile_n
    if (tile_m, tile_n) not in _ALLOWED_TILES:
        allowed = ", ".join(
            f"({m}, {n})" for m, n in sorted(_ALLOWED_TILES)
        )
        raise ValueError(
            f"tile (tile_m={tile_m}, tile_n={tile_n}) is not supported. "
            f"Supported tiles: {allowed}"
        )
    library = library or build_gguf_q8_0_prefill(load=True)
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


def _make_wrapper(variant: str):
    sym = _symbol(variant)

    def wrapper(*args, **kwargs) -> None:
        _launch(sym, *args, **kwargs)

    wrapper.__name__ = f"gguf_q8_0_{variant}"
    wrapper.__qualname__ = wrapper.__name__
    wrapper.__doc__ = (
        f"Launch GGUF Q8_0 WMMA prefill (C symbol: {sym}). Signature: "
        "(x_ptr, qweight_ptr, out_ptr, rows, in_features, out_features, "
        "tile_m=None, tile_n=None, stream=0)."
    )
    return wrapper


# Public Python entry points. Names mirror the existing gguf_q8_0_gemv_*
# wrappers so call sites can swap them by string substitution.
gguf_q8_0_wmma_prefill_bf16_bf16_out = _make_wrapper("wmma_prefill_bf16_bf16_out")
gguf_q8_0_wmma_prefill_bf16_fp16_out = _make_wrapper("wmma_prefill_bf16_fp16_out")
gguf_q8_0_wmma_prefill_bf16_f32_out = _make_wrapper("wmma_prefill_bf16_f32_out")
gguf_q8_0_wmma_prefill_fp16_bf16_out = _make_wrapper("wmma_prefill_fp16_bf16_out")
gguf_q8_0_wmma_prefill_fp16_fp16_out = _make_wrapper("wmma_prefill_fp16_fp16_out")
gguf_q8_0_wmma_prefill_fp16_f32_out = _make_wrapper("wmma_prefill_fp16_f32_out")
gguf_q8_0_wmma_prefill_f32_bf16_out = _make_wrapper("wmma_prefill_f32_bf16_out")
gguf_q8_0_wmma_prefill_f32_fp16_out = _make_wrapper("wmma_prefill_f32_fp16_out")
gguf_q8_0_wmma_prefill_f32_f32_out = _make_wrapper("wmma_prefill_f32_f32_out")


def _launch_dual(
    symbol: str,
    x_ptr: int,
    qweight_a_ptr: int,
    qweight_b_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features_a: int,
    out_features_b: int,
    *,
    tile_m: int | None = None,
    tile_n: int | None = None,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    if rows <= 0:
        raise ValueError("rows must be positive")
    if in_features <= 0:
        raise ValueError("in_features must be positive")
    if out_features_a <= 0 or out_features_b <= 0:
        raise ValueError("out_features_a and out_features_b must be positive")
    if in_features % 32 != 0:
        raise ValueError("in_features must be divisible by Q8_0 block size 32")
    if tile_m is None or tile_n is None:
        tm_def, tn_def = _default_tiles(rows, in_features, max(out_features_a, out_features_b))
        tile_m = tm_def if tile_m is None else tile_m
        tile_n = tn_def if tile_n is None else tile_n
    if (tile_m, tile_n) not in _ALLOWED_TILES:
        allowed = ", ".join(f"({m}, {n})" for m, n in sorted(_ALLOWED_TILES))
        raise ValueError(
            f"tile (tile_m={tile_m}, tile_n={tile_n}) is not supported. "
            f"Supported tiles: {allowed}"
        )
    if out_features_a % tile_m != 0:
        raise ValueError(
            f"out_features_a={out_features_a} must be a multiple of tile_m={tile_m} "
            "so a col_tile never straddles the gate/up boundary"
        )
    if out_features_b % tile_m != 0:
        raise ValueError(
            f"out_features_b={out_features_b} must be a multiple of tile_m={tile_m}"
        )
    library = library or build_gguf_q8_0_prefill(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, symbol)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
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
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features_a),
        ctypes.c_int64(out_features_b),
        ctypes.c_int64(tile_m),
        ctypes.c_int64(tile_n),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def _make_dual_wrapper(variant: str):
    sym = _symbol(variant)

    def wrapper(*args, **kwargs) -> None:
        _launch_dual(sym, *args, **kwargs)

    wrapper.__name__ = f"gguf_q8_0_{variant}"
    wrapper.__qualname__ = wrapper.__name__
    wrapper.__doc__ = (
        f"Launch GGUF Q8_0 fused dual gate+up WMMA prefill (C symbol: {sym}). "
        "Signature: (x_ptr, qweight_a_ptr, qweight_b_ptr, out_ptr, rows, "
        "in_features, out_features_a, out_features_b, tile_m=None, tile_n=None, stream=0)."
    )
    return wrapper


gguf_q8_0_wmma_prefill_dual_gate_up_bf16_bf16_out = _make_dual_wrapper(
    "wmma_prefill_dual_gate_up_bf16_bf16_out"
)
gguf_q8_0_wmma_prefill_dual_gate_up_fp16_fp16_out = _make_dual_wrapper(
    "wmma_prefill_dual_gate_up_fp16_fp16_out"
)


_WRAPPERS = {
    "wmma_prefill_bf16_bf16_out": gguf_q8_0_wmma_prefill_bf16_bf16_out,
    "wmma_prefill_bf16_fp16_out": gguf_q8_0_wmma_prefill_bf16_fp16_out,
    "wmma_prefill_bf16_f32_out": gguf_q8_0_wmma_prefill_bf16_f32_out,
    "wmma_prefill_fp16_bf16_out": gguf_q8_0_wmma_prefill_fp16_bf16_out,
    "wmma_prefill_fp16_fp16_out": gguf_q8_0_wmma_prefill_fp16_fp16_out,
    "wmma_prefill_fp16_f32_out": gguf_q8_0_wmma_prefill_fp16_f32_out,
    "wmma_prefill_f32_bf16_out": gguf_q8_0_wmma_prefill_f32_bf16_out,
    "wmma_prefill_f32_fp16_out": gguf_q8_0_wmma_prefill_f32_fp16_out,
    "wmma_prefill_f32_f32_out": gguf_q8_0_wmma_prefill_f32_f32_out,
    "wmma_prefill_dual_gate_up_bf16_bf16_out": gguf_q8_0_wmma_prefill_dual_gate_up_bf16_bf16_out,
    "wmma_prefill_dual_gate_up_fp16_fp16_out": gguf_q8_0_wmma_prefill_dual_gate_up_fp16_fp16_out,
}


def register_gguf_q8_0_prefill_kernels(*, replace: bool = True) -> None:
    """Register the WMMA prefill wrappers in the global kernel registry.

    Bound under keys::

        ("hip_gfx1100", "linear", "gguf_q8_0", "wmma_prefill_<in>_<out>_out")

    The decode-shaped ``prefill_*`` aliases in ``gguf_k_gemv.py`` are not
    touched here; the runtime dispatch in ``hipengine.runtime.gguf_linear``
    chooses between the two key families.
    """
    for variant, fn in _WRAPPERS.items():
        register(
            KernelKey("hip_gfx1100", "linear", "gguf_q8_0", variant),
            fn,
            replace=replace,
        )


register_gguf_q8_0_prefill_kernels()


__all__ = [
    "build_gguf_q8_0_prefill",
    "plan_gguf_q8_0_prefill_build",
    "register_gguf_q8_0_prefill_kernels",
    "gguf_q8_0_wmma_prefill_bf16_bf16_out",
    "gguf_q8_0_wmma_prefill_bf16_fp16_out",
    "gguf_q8_0_wmma_prefill_bf16_f32_out",
    "gguf_q8_0_wmma_prefill_fp16_bf16_out",
    "gguf_q8_0_wmma_prefill_fp16_fp16_out",
    "gguf_q8_0_wmma_prefill_fp16_f32_out",
    "gguf_q8_0_wmma_prefill_f32_bf16_out",
    "gguf_q8_0_wmma_prefill_f32_fp16_out",
    "gguf_q8_0_wmma_prefill_f32_f32_out",
    "gguf_q8_0_wmma_prefill_dual_gate_up_bf16_bf16_out",
    "gguf_q8_0_wmma_prefill_dual_gate_up_fp16_fp16_out",
]
