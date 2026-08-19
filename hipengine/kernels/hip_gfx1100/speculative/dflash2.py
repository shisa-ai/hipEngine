"""Raw-pointer GPU DFlash2 drafter kernels (grouped dynamic conv, top-16,
candidate selector).

Exact-math native kernels RED-pinned against ``kernels/cpu_reference/dflash2.py``
(see ``tests/test_dflash2_native_kernels.py``).  The conv reads BF16 inputs and
accumulates FP32; the selector projects hidden to the low-rank context gate and
runs the greedy walk over the top-K candidate table with BF16 codebook gathers.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("dflash2.hip")
_OUTPUT_NAME = "dflash2.so"
_SYMBOL_GROUPED_CONV = "hipengine_dflash2_grouped_conv"
_SYMBOL_TOP16_ROWS = "hipengine_dflash2_top16_rows"
_SYMBOL_SELECTOR = "hipengine_dflash2_selector"

DFLASH2_SELECTOR_MAX_TOP_K = 16
DFLASH2_SELECTOR_MAX_RANK = 256


def plan_dflash2_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="dflash2",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
    )


def build_dflash2(
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
        family="dflash2",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def dflash2_grouped_conv(
    hidden_bf16_ptr: int,
    dyn_bf16_ptr: int,
    base_bf16_ptr: int,
    out_bf16_ptr: int,
    rows: int,
    hidden_size: int,
    group_size: int,
    *,
    dyn_offset: int = 0,
    dyn_stride: int | None = None,
    threads: int = 256,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Causal grouped dynamic conv: ``out = conv(hidden, dyn, base)``.

    ``dyn`` is a strided BF16 coefficient view; ``dyn_offset``/``dyn_stride``
    select the side block within a wider projection buffer (input side: offset
    0 over a 1280-wide row; output side: offset ``2*groups`` over the same
    row).  Defaults model a compact ``(rows, 2*groups)`` buffer.  ``base`` is
    ``(2, hidden)`` BF16.
    """
    if hidden_size <= 0 or group_size <= 0 or hidden_size % group_size != 0:
        raise ValueError("grouped dynamic conv requires group_size dividing hidden_size")
    groups = hidden_size // group_size
    if dyn_stride is None:
        dyn_stride = 2 * groups
    if dyn_offset < 0 or dyn_stride <= 0:
        raise ValueError("grouped dynamic conv dyn offset/stride must be non-negative")
    library = library or build_dflash2(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_GROUPED_CONV)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(hidden_bf16_ptr),
        ctypes.c_void_p(dyn_bf16_ptr),
        ctypes.c_void_p(base_bf16_ptr),
        ctypes.c_void_p(out_bf16_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(hidden_size),
        ctypes.c_int32(group_size),
        ctypes.c_int32(groups),
        ctypes.c_int64(dyn_offset),
        ctypes.c_int64(dyn_stride),
        ctypes.c_int64(threads),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def dflash2_top16_rows(
    logits_f32_ptr: int,
    out_ids_i32_ptr: int,
    out_values_f32_ptr: int,
    rows: int,
    vocab_size: int,
    top_k: int,
    *,
    threads: int = 256,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Row-wise top-k values/ids (descending) over FP32 logits."""
    if rows <= 0:
        raise ValueError("top16 rows must be positive")
    if top_k <= 0 or top_k > DFLASH2_SELECTOR_MAX_TOP_K:
        raise ValueError(f"top_k must be in [1, {DFLASH2_SELECTOR_MAX_TOP_K}]")
    library = library or build_dflash2(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_TOP16_ROWS)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int32,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(logits_f32_ptr),
        ctypes.c_void_p(out_ids_i32_ptr),
        ctypes.c_void_p(out_values_f32_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(vocab_size),
        ctypes.c_int32(top_k),
        ctypes.c_int64(threads),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def dflash2_selector(
    hidden_bf16_ptr: int,
    hidden_projection_bf16_ptr: int,
    cand_ids_i32_ptr: int,
    cand_vals_f32_ptr: int,
    anchor_ids_i32_ptr: int,
    codebook_a_bf16_ptr: int,
    codebook_b_bf16_ptr: int,
    h_scratch_f32_ptr: int,
    path_i32_ptr: int,
    scores_f32_ptr: int,
    rows: int,
    hidden_size: int,
    rank: int,
    top_k: int,
    vocab_size: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Greedy DFlash2 candidate-selector walk over one batch of rows.

    ``hidden`` is ``(rows, hidden_size)`` BF16; ``cand_ids``/``cand_vals`` are
    ``(rows, top_k)``; ``anchor_ids`` is ``(1,)``; codebooks are
    ``(vocab_size, rank)`` BF16.  ``path`` is ``(rows,)`` i32; ``scores`` is
    ``(rows, top_k)`` f32.
    """
    if rows <= 0:
        raise ValueError("selector rows must be positive")
    if rank <= 0 or rank > DFLASH2_SELECTOR_MAX_RANK:
        raise ValueError(f"selector rank must be in [1, {DFLASH2_SELECTOR_MAX_RANK}]")
    if top_k <= 0 or top_k > DFLASH2_SELECTOR_MAX_TOP_K:
        raise ValueError(f"selector top_k must be in [1, {DFLASH2_SELECTOR_MAX_TOP_K}]")
    library = library or build_dflash2(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_SELECTOR)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
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
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(hidden_bf16_ptr),
        ctypes.c_void_p(hidden_projection_bf16_ptr),
        ctypes.c_void_p(cand_ids_i32_ptr),
        ctypes.c_void_p(cand_vals_f32_ptr),
        ctypes.c_void_p(anchor_ids_i32_ptr),
        ctypes.c_void_p(codebook_a_bf16_ptr),
        ctypes.c_void_p(codebook_b_bf16_ptr),
        ctypes.c_void_p(h_scratch_f32_ptr),
        ctypes.c_void_p(path_i32_ptr),
        ctypes.c_void_p(scores_f32_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(hidden_size),
        ctypes.c_int32(rank),
        ctypes.c_int32(top_k),
        ctypes.c_int64(vocab_size),
        ctypes.c_int64(1),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def register_dflash2_kernels(*, replace: bool = True) -> None:
    register(
        KernelKey("hip_gfx1100", "dflash2_grouped_conv", "bf16"),
        dflash2_grouped_conv,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "dflash2_selector", "bf16"),
        dflash2_selector,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "dflash2_top16_rows", "fp32"),
        dflash2_top16_rows,
        replace=replace,
    )


register_dflash2_kernels()

__all__ = [
    "DFLASH2_SELECTOR_MAX_TOP_K",
    "DFLASH2_SELECTOR_MAX_RANK",
    "build_dflash2",
    "dflash2_grouped_conv",
    "dflash2_selector",
    "dflash2_top16_rows",
    "plan_dflash2_build",
    "register_dflash2_kernels",
]
