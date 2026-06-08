"""Raw-pointer GPU lm-head + argmax wrapper."""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("lm_head.hip")
_OUTPUT_NAME = "lm_head.so"
_SYMBOL = "hipengine_lm_head_fp16_argmax_bf16"
_SYMBOL_ROWS_I32 = "hipengine_lm_head_fp16_argmax_bf16_rows_i32"
_SYMBOL_ARGMAX = "hipengine_argmax_f32"
_SYMBOL_ARGMAX_ROWS_I32 = "hipengine_argmax_f32_rows_i32"
_SYMBOL_TOPK_ROWS_I32 = "hipengine_topk_f32_rows_i32"
_SYMBOL_W8A16_LM_HEAD_ARGMAX_ROWS = "hipengine_w8a16_lm_head_argmax_rows_bf16"
_SYMBOL_BATCH_ARGMAX = "hipengine_batch_argmax_f32"
_ALLOWED_THREADS = {128, 256, 512}
_MAX_TOPK = 8


def plan_lm_head_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="lm_head",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
    )


def build_lm_head(
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
        family="lm_head",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def lm_head_fp16_argmax_bf16(
    hidden_bf16_ptr: int,
    weight_fp16_ptr: int,
    logits_f32_ptr: int,
    block_values_f32_ptr: int,
    block_indices_i64_ptr: int,
    out_index_i64_ptr: int,
    out_value_f32_ptr: int,
    hidden_size: int,
    vocab_size: int,
    *,
    threads: int = 256,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch FP16 lm-head projection and GPU argmax for one BF16 hidden row."""

    _check_shape(hidden_size, vocab_size, threads)
    library = library or build_lm_head(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL)
    fn.argtypes = [
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
        ctypes.c_void_p(hidden_bf16_ptr),
        ctypes.c_void_p(weight_fp16_ptr),
        ctypes.c_void_p(logits_f32_ptr),
        ctypes.c_void_p(block_values_f32_ptr),
        ctypes.c_void_p(block_indices_i64_ptr),
        ctypes.c_void_p(out_index_i64_ptr),
        ctypes.c_void_p(out_value_f32_ptr),
        ctypes.c_int64(hidden_size),
        ctypes.c_int64(vocab_size),
        ctypes.c_int64(threads),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def lm_head_fp16_argmax_bf16_rows_i32(
    hidden_bf16_ptr: int,
    weight_fp16_ptr: int,
    logits_f32_ptr: int,
    block_values_f32_ptr: int,
    block_indices_i32_ptr: int,
    out_indices_i32_ptr: int,
    out_values_f32_ptr: int | None,
    rows: int,
    hidden_size: int,
    vocab_size: int,
    *,
    threads: int = 256,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch FP16 lm-head projection and row-wise GPU argmax for BF16 hidden rows."""

    _check_rows(rows)
    _check_shape(hidden_size, vocab_size, threads)
    _check_i32_vocab(vocab_size)
    library = library or build_lm_head(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_ROWS_I32)
    fn.argtypes = [
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
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(hidden_bf16_ptr),
        ctypes.c_void_p(weight_fp16_ptr),
        ctypes.c_void_p(logits_f32_ptr),
        ctypes.c_void_p(block_values_f32_ptr),
        ctypes.c_void_p(block_indices_i32_ptr),
        ctypes.c_void_p(out_indices_i32_ptr),
        ctypes.c_void_p(out_values_f32_ptr) if out_values_f32_ptr is not None else ctypes.c_void_p(),
        ctypes.c_int64(rows),
        ctypes.c_int64(hidden_size),
        ctypes.c_int64(vocab_size),
        ctypes.c_int64(threads),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def argmax_f32(
    logits_f32_ptr: int,
    block_values_f32_ptr: int,
    block_indices_i64_ptr: int,
    out_index_i64_ptr: int,
    out_value_f32_ptr: int,
    vocab_size: int,
    *,
    threads: int = 256,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    _check_shape(1, vocab_size, threads)
    library = library or build_lm_head(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_ARGMAX)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(logits_f32_ptr),
        ctypes.c_void_p(block_values_f32_ptr),
        ctypes.c_void_p(block_indices_i64_ptr),
        ctypes.c_void_p(out_index_i64_ptr),
        ctypes.c_void_p(out_value_f32_ptr),
        ctypes.c_int64(vocab_size),
        ctypes.c_int64(threads),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def argmax_f32_rows_i32(
    logits_f32_ptr: int,
    block_values_f32_ptr: int,
    block_indices_i32_ptr: int,
    out_indices_i32_ptr: int,
    out_values_f32_ptr: int | None,
    rows: int,
    vocab_size: int,
    *,
    threads: int = 256,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Write row-wise top-1 ids for ``logits[rows, vocab_size]`` without host logits."""

    _check_rows(rows)
    _check_shape(1, vocab_size, threads)
    _check_i32_vocab(vocab_size)
    library = library or build_lm_head(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_ARGMAX_ROWS_I32)
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
        ctypes.c_void_p(logits_f32_ptr),
        ctypes.c_void_p(block_values_f32_ptr),
        ctypes.c_void_p(block_indices_i32_ptr),
        ctypes.c_void_p(out_indices_i32_ptr),
        ctypes.c_void_p(out_values_f32_ptr) if out_values_f32_ptr is not None else ctypes.c_void_p(),
        ctypes.c_int64(rows),
        ctypes.c_int64(vocab_size),
        ctypes.c_int64(threads),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def topk_f32_rows_i32(
    logits_f32_ptr: int,
    out_values_f32_ptr: int | None,
    out_indices_i32_ptr: int,
    rows: int,
    vocab_size: int,
    top_k: int,
    *,
    threads: int = 256,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Write row-wise top-k values/ids for ``logits[rows, vocab_size]`` on device."""

    _check_rows(rows)
    _check_shape(1, vocab_size, threads)
    _check_i32_vocab(vocab_size)
    _check_topk(top_k)
    library = library or build_lm_head(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_TOPK_ROWS_I32)
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
        ctypes.c_void_p(logits_f32_ptr),
        ctypes.c_void_p(out_values_f32_ptr) if out_values_f32_ptr is not None else ctypes.c_void_p(),
        ctypes.c_void_p(out_indices_i32_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(vocab_size),
        ctypes.c_int64(top_k),
        ctypes.c_int64(threads),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def batch_argmax_f32(
    logits_f32_ptr: int,
    block_values_f32_ptr: int,
    block_indices_i64_ptr: int,
    out_index_i64_ptr: int,
    out_value_f32_ptr: int,
    rows: int,
    vocab_size: int,
    *,
    threads: int = 256,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    _check_shape(rows, vocab_size, threads)
    library = library or build_lm_head(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_BATCH_ARGMAX)
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
        ctypes.c_void_p(logits_f32_ptr),
        ctypes.c_void_p(block_values_f32_ptr),
        ctypes.c_void_p(block_indices_i64_ptr),
        ctypes.c_void_p(out_index_i64_ptr),
        ctypes.c_void_p(out_value_f32_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(vocab_size),
        ctypes.c_int64(threads),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def lm_head_argmax_stage1_blocks(vocab_size: int, *, threads: int = 256) -> int:
    _check_shape(1, vocab_size, threads)
    return (int(vocab_size) + int(threads) * 4 - 1) // (int(threads) * 4)


def w8a16_lm_head_argmax_rows_bf16(
    hidden_bf16_ptr: int,
    weight_int8_ptr: int,
    weight_scale_f32_ptr: int,
    block_values_f32_ptr: int,
    block_indices_i32_ptr: int,
    out_indices_i32_ptr: int,
    out_values_f32_ptr: int | None,
    rows: int,
    hidden_size: int,
    vocab_size: int,
    *,
    threads: int = 256,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """R3.7 fused W8A16 LM-head + argmax-rows.

    Replaces the unfused ``w8a16_linear_bf16_f32_multi_row`` -> ``argmax_f32_rows_i32``
    pair so the per-cycle ``[rows, vocab_size]`` FP32 logits buffer is never
    materialized in HBM.  ``hidden_bf16`` is ``[rows, hidden_size]`` in bf16 bits;
    weight is W8 ``[vocab_size, hidden_size]`` INT8 with FP32 ``[vocab_size]`` scale.
    Outputs ``out_indices_i32[rows]`` (top-1 token id per row) and optional
    ``out_values_f32[rows]`` (top-1 logit value per row).  Scratch buffers
    ``block_values_f32[rows * stage1_blocks]`` / ``block_indices_i32[rows * stage1_blocks]``
    must be sized via :func:`lm_head_argmax_stage1_blocks`.
    """

    _check_rows(rows)
    _check_shape(hidden_size, vocab_size, threads)
    _check_i32_vocab(vocab_size)
    library = library or build_lm_head(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_W8A16_LM_HEAD_ARGMAX_ROWS)
    fn.argtypes = [
        ctypes.c_void_p,  # hidden bf16
        ctypes.c_void_p,  # weight int8
        ctypes.c_void_p,  # weight scale f32
        ctypes.c_void_p,  # block_values f32 scratch
        ctypes.c_void_p,  # block_indices i32 scratch
        ctypes.c_void_p,  # out_indices i32
        ctypes.c_void_p,  # out_values f32 (nullable)
        ctypes.c_int64,   # rows
        ctypes.c_int64,   # hidden_size
        ctypes.c_int64,   # vocab_size
        ctypes.c_int64,   # threads
        ctypes.c_void_p,  # stream
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(hidden_bf16_ptr),
        ctypes.c_void_p(weight_int8_ptr),
        ctypes.c_void_p(weight_scale_f32_ptr),
        ctypes.c_void_p(block_values_f32_ptr),
        ctypes.c_void_p(block_indices_i32_ptr),
        ctypes.c_void_p(out_indices_i32_ptr),
        ctypes.c_void_p(out_values_f32_ptr) if out_values_f32_ptr is not None else ctypes.c_void_p(),
        ctypes.c_int64(rows),
        ctypes.c_int64(hidden_size),
        ctypes.c_int64(vocab_size),
        ctypes.c_int64(threads),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def register_lm_head_kernels(*, replace: bool = True) -> None:
    register(
        KernelKey("hip_gfx1100", "lm_head", "w4_paro", "fp16_argmax_bf16"),
        lm_head_fp16_argmax_bf16,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "lm_head", "w4_paro", "fp16_argmax_bf16_rows_i32"),
        lm_head_fp16_argmax_bf16_rows_i32,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "argmax", "w4_paro", "f32"),
        argmax_f32,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "argmax", "w4_paro", "batch_f32"),
        batch_argmax_f32,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "argmax", "w4_paro", "f32_rows_i32"),
        argmax_f32_rows_i32,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "topk", "w4_paro", "f32_rows_i32"),
        topk_f32_rows_i32,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "lm_head_argmax", "w8a16", "bf16_rows_i32"),
        w8a16_lm_head_argmax_rows_bf16,
        replace=replace,
    )


def _check_shape(hidden_size: int, vocab_size: int, threads: int) -> None:
    if hidden_size <= 0:
        raise ValueError("hidden_size must be positive")
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    if threads not in _ALLOWED_THREADS:
        raise ValueError("threads must be one of 128, 256, or 512")


def _check_rows(rows: int) -> None:
    if rows <= 0:
        raise ValueError("rows must be positive")


def _check_i32_vocab(vocab_size: int) -> None:
    if vocab_size > 2**31 - 1:
        raise ValueError("vocab_size must fit int32 row-wise argmax outputs")


def _check_topk(top_k: int) -> None:
    if top_k <= 0 or top_k > _MAX_TOPK:
        raise ValueError(f"top_k must be in [1, {_MAX_TOPK}]")


register_lm_head_kernels()

