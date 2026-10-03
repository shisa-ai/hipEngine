"""Raw-pointer wrappers for the GGUF Q8_0 batched WMMA prefill kernel.

This module owns the C ABI exports defined in ``gguf_q8_0_prefill.hip``
(see docs/reference/GGUF.md \"P8: real batched prefill GEMM\" for the wider plan).
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
import os
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
    (16, 64),
    (32, 16),
    (32, 32),
    (32, 64),
    (64, 16),
    (64, 32),
}

# Cached single-visible-device arch for ``_default_tiles``; see ``_target_arch``.
_detected_arch: str | None = None


def _target_arch() -> str:
    """The build's target arch: ``HIPENGINE_HIP_ARCH`` when a backend scoped it,
    otherwise the (single) visible HIP device's arch, else ``""``.

    The retained Q8_0 WMMA tile differs between gfx1100 and gfx1151, so the
    default is keyed on this rather than on one flat rule. Same pattern as
    ``gguf_iq_wmma_prefill._target_arch``.
    """

    import os

    arch = (os.environ.get("HIPENGINE_HIP_ARCH") or "").strip()
    if arch:
        return arch
    global _detected_arch
    if _detected_arch is None:
        from hipengine.kernels.backends import detect_hip_target_arches

        arches = detect_hip_target_arches()
        _detected_arch = arches[0] if len(arches) == 1 else ""
    return _detected_arch


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

    The default is keyed on the target arch (``_target_arch``): the gfx1100 and
    gfx1151 sweeps below ran on different physical hosts and retained different
    tiles, so neither rule is correct for the other arch.

    gfx1100:

    ``(16, 32)`` for every shape at ``rows >= 32``, and ``(16, 16)`` below that,
    where the wider ``tile_n`` under-fills the WMMA tile.

    This replaces a shape cascade (``in >= 4096 and out >= 2048 -> tile_m 64``,
    ``in <= 2048 and out >= 4096 -> 16``, ``out >= 32 -> 32``, else 16) that was
    tuned in P9.C1. A 2026-09-27 sweep on the same GPU (RX 7900 W7900 / gfx1100,
    BF16/BF16, thirteen shapes, rows 8/31/128/256/512/1024, interleaved passes
    with per-tile medians) finds ``tile_m`` 16 fastest or tied at *every*
    measured point, and the cascade's 32/64 choices losing by:

    * ``rows >= 128``: 1.17-1.94x. Gemma 4's dense Q8_0 projections, which are
      159 ms of a 554 ms per-prefill kernel budget, were all on the losing side.
    * ``rows == 8`` or ``31``: 1.72-3.31x, so the small-row prefill shapes
      (MTP/verifier blocks) lose most of all.

    The cascade's rules that already selected 16 -- ``in <= 2048 and
    out >= 4096``, ``out <= 512``, and ``out < 32`` -- still select 16 here, so
    collapsing it changes only the cases the sweep shows were wrong. ``tile_n``
    is unchanged: 32 at ``rows >= 32`` is fastest or within 1.4% at every
    measured shape.

    ``in_features`` no longer selects anything; it stays in the signature
    because callers and the override path pass the shape as a unit. See
    ``tests/test_gpu_gguf_q8_0_wmma_prefill.py`` for the pinning tests and
    ``scripts/gemma4_dense_q8_tile_sweep.py`` for the sweep that produced this.

    gfx1151 retains a different tile, measured by an independent 2026-09-28
    sweep on that host:

    A 2026-09-28 sweep on gfx1151 (Radeon 8060S) over eleven shapes --
    gemma4's dense projections and the qwen35moe shapes the previous rules
    were written for -- found ``tile_m=64`` best or within run-to-run noise
    on every one of them, and the previous ``(32, 32)`` fallback best on
    none. A follow-up sweep added the ``tile_n=64`` points, which the first
    one never reached, and they move the family again: ``(32, 64)`` is the
    fastest tile on three of gemma4's four dense prefill geometries and the
    family total drops 4.121 -> 3.639 ms.

    ``tile_n`` is the token-side tile: a block covers ``AWQ_TILE_N`` tokens
    and the weight tile is dequantized once per (block, weight element), so
    it is the dimension that trades weight traffic against registers. It is
    NOT register-symmetric with ``tile_m``. ``my_out[TM]`` is only an index
    array, but ``my_token[TN]`` and the activation gather are per-TN, so the
    VGPR cost of the two directions differs -- measured on gfx1151:
    ``(64, 32)`` 127 VGPRs against ``(32, 64)`` 143. That is why the wider
    token tile was assumed unaffordable and never swept; the real difference
    is 16 registers, not a resource wall.

    Selected measurements from the ``tile_n=64`` sweep (rows=512, bf16 in,
    bf16 out, ms, best of 8):

    * ``(in=2816, out=4096)`` dense ffn gate/up: ``32x64`` 0.939 against
      ``64x32`` 1.116, ``32x32`` 1.307, ``16x64`` 1.370.
    * ``(in=2816, out=2816)`` o proj: ``32x64`` 0.684 against ``64x32``
      0.795, ``16x64`` 0.883, ``32x32`` 1.347.
    * ``(in=4096, out=2816)`` dense ffn down: ``32x64`` 1.482 against
      ``16x64`` 1.663, ``64x32`` 1.712, ``32x32`` 2.500.
    * ``(in=2816, out=2112)`` swa k/v: the one shape where ``64x32`` leads,
      0.498 against ``32x64`` 0.534. The gap is 7 percent on one of four
      shapes, about 1 percent of the family total, so a single tile is kept
      rather than a shape ladder.

    Earlier measurements, on shapes the ``(32, 64)`` sweep did not cover:

    * ``(in=2816, out=2048)`` swa k/v: ``64x32`` 0.459 against ``16x32``
      0.798 and ``32x32`` 0.931.
    * ``(in=2816, out=2112)`` dense ffn gate/up: ``64x32`` 0.634 against
      ``16x32`` 1.163 and ``32x32`` 1.187.
    * ``(in=4096, out=2048)`` qwen35moe ssm down: ``64x32`` 0.967 against
      ``16x32`` 1.722 and ``32x32`` 1.997.
    * ``(in=2816, out=8192)`` global q_proj: ``64x32`` 2.885 against
      ``16x32`` 3.736 and ``32x32`` 3.828.
    * ``(in=2048, out=8192)`` qwen35moe qkv: ``64x32`` 2.494 against
      ``16x32`` 2.755 -- the shape the previous rule gave ``16``, so that
      rule was picking the slower tile here too.
    * ``(in=8192, out=2816)`` and ``(in=4096, out=2816)`` were the only two
      of the eleven where a narrow tile measured ahead in the first sweep,
      by 1.1 and 0.9 percent; the ``tile_n=64`` sweep then put ``32x64``
      ahead on ``(in=4096, out=2816)`` by 15.5 percent.

    The previous rules were tuned on gfx1100 with the qwen35moe shapes and
    preferred ``(16, 32)`` for ``(in<=2048, out>=4096)`` and ``(32, 32)``
    otherwise. Both were re-measured here; the replacement is a single tile
    rather than a shape ladder because the ladder's branches were selecting
    the slower tile, and a rule that cannot be cleared by measurement is
    worse than no rule.

    All tiles produce bit-identical output on the shapes checked (three
    gemma4 dense shapes, six tiles each; then four gemma4 dense shapes, six
    tiles each, for the ``tile_n=64`` sweep -- maxdiff 0.000e+00 throughout),
    so this is a performance choice with no numerical consequence. See
    ``tests/test_gpu_gguf_q8_0_wmma_prefill.py`` for dispatch pinning tests.
    """

    override_m = os.environ.get("HIPENGINE_GGUF_Q8_0_WMMA_TILE_M")
    override_n = os.environ.get("HIPENGINE_GGUF_Q8_0_WMMA_TILE_N")
    if override_m is not None or override_n is not None:
        if override_m is None or override_n is None:
            raise ValueError("Q8_0 WMMA tile override requires both M and N")
        tile = (int(override_m), int(override_n))
        if tile not in _ALLOWED_TILES:
            raise ValueError(f"unsupported Q8_0 WMMA tile override: {tile}")
        return tile
    if _target_arch() == "gfx1151":
        tile_n = 64 if rows >= 64 else (32 if rows >= 32 else 16)
        tile_m = 32 if out_features >= 32 else 16
        return tile_m, tile_n
    # gfx1100, and any host whose arch is not the gfx1151 peer: the flat rule
    # measured on the W7900.
    return 16, (32 if rows >= 32 else 16)


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


# P1 device-driven grouped Q8_0 down owner. Signature:
# (input_ptr, expert_start_ptr, weights_ptr, output_ptr, compact_rows,
#  num_experts, in_features, out_features, stream=0).
_GROUPED_ARGTYPES = (
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_void_p,
)


def gguf_q8_0_selected_grouped_prefill_compact_bf16_bf16_out(
    input_ptr: int,
    expert_start_ptr: int,
    weights_ptr: int,
    output_ptr: int,
    compact_rows: int,
    num_experts: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch device-driven grouped Q8_0 down (no host roundtrip)."""

    for value, name in (
        (compact_rows, "compact_rows"),
        (num_experts, "num_experts"),
        (in_features, "in_features"),
        (out_features, "out_features"),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if in_features % 32:
        raise ValueError("in_features must be divisible by Q8_0 block size 32")
    library = library or build_gguf_q8_0_prefill(load=True)
    runtime = runtime or get_hip_runtime()
    fn = library.hipengine_gguf_q8_0_selected_grouped_prefill_compact_bf16_bf16_out
    fn.argtypes = _GROUPED_ARGTYPES
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(input_ptr),
        ctypes.c_void_p(expert_start_ptr),
        ctypes.c_void_p(weights_ptr),
        ctypes.c_void_p(output_ptr),
        ctypes.c_int64(compact_rows),
        ctypes.c_int64(num_experts),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


_GROUPED_WMMA_ARGTYPES = (
    ctypes.c_void_p,
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
)


def gguf_q8_0_selected_grouped_wmma_prefill_compact_bf16_bf16_out(
    input_ptr: int,
    expert_start_compact_ptr: int,
    expert_start_wmma_ptr: int,
    tile_expert_ptr: int,
    weights_ptr: int,
    output_ptr: int,
    compact_rows: int,
    num_experts: int,
    in_features: int,
    out_features: int,
    wmma_total_rows: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Run the grouped selected-expert Q8_0 down via WMMA tiles.

    #19 R9 route: the f16-WMMA dequant GEMM contract of the promoted Q5_1
    grouped WMMA down (same tile_expert / expert_start_wmma device-side
    row map), applied to Q8_0 expert weights (2-byte fp16 scale + int8 qs,
    no min term). Weight slab per expert: out_features * ((in_features/32)
    * 34) bytes, matching the raw GGUF layout.
    """

    for value, name in (
        (compact_rows, "compact_rows"),
        (num_experts, "num_experts"),
        (in_features, "in_features"),
        (out_features, "out_features"),
        (wmma_total_rows, "wmma_total_rows"),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if in_features % 32:
        raise ValueError("in_features must be divisible by Q8_0 block size 32")
    if wmma_total_rows % 16:
        raise ValueError("wmma_total_rows must be divisible by 16")
    library = library or build_gguf_q8_0_prefill(load=True)
    runtime = runtime or get_hip_runtime()
    fn = library.hipengine_gguf_q8_0_selected_grouped_wmma_prefill_compact_bf16_bf16_out
    fn.argtypes = _GROUPED_WMMA_ARGTYPES
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(input_ptr),
        ctypes.c_void_p(expert_start_compact_ptr),
        ctypes.c_void_p(expert_start_wmma_ptr),
        ctypes.c_void_p(tile_expert_ptr),
        ctypes.c_void_p(weights_ptr),
        ctypes.c_void_p(output_ptr),
        ctypes.c_int64(compact_rows),
        ctypes.c_int64(num_experts),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features),
        ctypes.c_int64(wmma_total_rows),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


_WRAPPERS = {
    "selected_grouped_wmma_prefill_bf16_bf16_out": (
        gguf_q8_0_selected_grouped_wmma_prefill_compact_bf16_bf16_out),
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

    ``selected_grouped_prefill_compact_bf16_bf16_out`` is also bound under
    ``moe_linear``. It is the same wrapper, reached through the MoE expert
    route: one MoE layer of an unsloth dynamic quant can carry Q8_0 expert
    weights while the rest of the model is Q4_K, and without this key that
    layer's projection falls through to the per-row gather.
    """
    for variant, fn in _WRAPPERS.items():
        register(
            KernelKey("hip_gfx1100", "linear", "gguf_q8_0", variant),
            fn,
            replace=replace,
        )
    # The expert path resolves the grouped WMMA down owner on the
    # ``moe_linear`` layer axis -- ``gemma4_project_experts_wmma`` builds
    # ``KernelKey(backend, "moe_linear", quant_key, _WMMA_PREFILL_VARIANT)`` --
    # and under the group owner's ``compact`` variant name rather than this
    # module's dense ``selected_grouped_wmma_prefill_*`` key. Without a binding
    # here that resolve raises ``MissingKernelError`` and the down silently
    # falls through the WMMA -> MMQ -> grouped chain to the selected GEMV,
    # which is how a single layer's Q8_0 down came to run ~44 ms while every
    # other layer's Q5_1 down reaches the WMMA owner through the matching
    # ``qwen4_exp_q5_1`` binding.
    #
    # This wrapper already carries the expert ABI: it takes
    # ``expert_start_compact_ptr`` / ``expert_start_wmma_ptr`` /
    # ``tile_expert_ptr`` and documents itself as the promoted Q5_1 grouped
    # WMMA down contract applied to Q8_0 expert weights, and
    # ``tests/test_gpu_qwen4exp_q8_0_grouped_wmma_down.py`` pins it against a
    # NumPy dequant reference through ``qwen35_moe_wmma_tile_map``, the same
    # tile map the runner uses. Only the plain form is bound: ``auto`` runs
    # ``compensated=False`` and no compensated Q8_0 wrapper exists.
    register(
        KernelKey(
            "hip_gfx1100",
            "moe_linear",
            "gguf_q8_0",
            "selected_grouped_wmma_prefill_compact_bf16_bf16_out",
        ),
        gguf_q8_0_selected_grouped_wmma_prefill_compact_bf16_bf16_out,
        replace=replace,
    )
    # The grouped-prefill expert route
    # (``gemma4_project_experts_grouped_prefill``) walks
    # ``_GROUPED_PREFILL_VARIANTS`` and calls the plain compact variant through
    # the 8-argument grouped ABI. The Q8_0 expert family registers only this
    # name, so without it one Q8_0 expert layer falls through to the per-row
    # gather. This is a distinct key from the WMMA owner above, not a
    # competing binding.
    register(
        KernelKey(
            "hip_gfx1100",
            "moe_linear",
            "gguf_q8_0",
            "selected_grouped_prefill_compact_bf16_bf16_out",
        ),
        gguf_q8_0_selected_grouped_prefill_compact_bf16_bf16_out,
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
    "gguf_q8_0_selected_grouped_prefill_compact_bf16_bf16_out",
    "gguf_q8_0_selected_grouped_wmma_prefill_compact_bf16_bf16_out",
]
