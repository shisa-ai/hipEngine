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
_SYMBOL_ARGMAX = "hipengine_argmax_f32"
_SYMBOL_BATCH_ARGMAX = "hipengine_batch_argmax_f32"
_ALLOWED_THREADS = {128, 256, 512}


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


def register_lm_head_kernels(*, replace: bool = True) -> None:
    register(
        KernelKey("hip_gfx1100", "lm_head", "w4_paro", "fp16_argmax_bf16"),
        lm_head_fp16_argmax_bf16,
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


def _check_shape(hidden_size: int, vocab_size: int, threads: int) -> None:
    if hidden_size <= 0:
        raise ValueError("hidden_size must be positive")
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    if threads not in _ALLOWED_THREADS:
        raise ValueError("threads must be one of 128, 256, or 512")


register_lm_head_kernels()
