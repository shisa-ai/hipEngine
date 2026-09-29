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
