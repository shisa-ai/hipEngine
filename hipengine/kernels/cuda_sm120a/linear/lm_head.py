"""Raw-pointer Moonshine fused FP16 LM-head projection + stable argmax for CUDA ``sm_120a``.

The fused kernel reproduces the exact tied FP16 projection baseline (C1c plain
256-thread ``moonshine_f16_lm_head_projection`` ordered FP32 accumulation) and
the stable lowest-index argmax fallback (C1a ``moonshine_argmax_fp16``) in a
single bounded pass over the 36,864-row weight stream.  Stage 1 emits only
per-block partial maxima; stage 2 reduces them.  No full logit plane is
materialized; scratch is ``num_blocks`` (value, index) partials plus the final
(index, value) pair.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_cuda, plan_cuda_build
from hipengine.core.ctypes_cache import signed_kernel_fn
from hipengine.core.cuda import CUDA_SUCCESS, CudaRuntime, get_cuda_runtime
from hipengine.kernels.backends import cuda_target_arch_for_backend
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("lm_head.cu")
_OUTPUT_NAME = "lm_head.so"
_BACKEND = "cuda_sm120a"
_TARGET_ARCH = cuda_target_arch_for_backend(_BACKEND)
# Stage 1 is fixed at 256 threads so each row's FP32 accumulation order matches
# the C1c plain lm-head baseline byte-for-byte.
_THREADS = 256
# Default rows per stage-1 block; a batch-timed CUDA-event screen on exclusive
# GPU0 (RTX PRO 6000 Blackwell, driver 610.43.03) closed the best bucket over
# the full 36,864x416 weight stream: rows_per_block 4/8/16 measure
# 29.90/29.04/31.52 us for the fused pass vs 39.62 us for the two-step
# projection+argmax (1.36x faster at the best bucket).
# num_blocks = ceil(vocab_size / rows_per_block).
_DEFAULT_ROWS_PER_BLOCK = 8

_LM_HEAD_ARGMAX_ARGS = (
    *(ctypes.c_void_p for _ in range(6)),
    ctypes.c_int64,  # in_features
    ctypes.c_int64,  # vocab_size
    ctypes.c_int64,  # rows_per_block
    ctypes.c_void_p,  # stream
)
# Static-B batch variant: same layout, plus an int64 ``batch`` before the stream.
_LM_HEAD_ARGMAX_BATCH_ARGS = (
    *(ctypes.c_void_p for _ in range(6)),
    ctypes.c_int64,  # in_features
    ctypes.c_int64,  # vocab_size
    ctypes.c_int64,  # rows_per_block
    ctypes.c_int64,  # batch
    ctypes.c_void_p,  # stream
)
# Fused wave8 + stable top-1 (C6/RR-8): 8 columns per block, one per warp, so
# there is no rows-per-block knob; scratch is ceil(vocab/8) partials.
_LM_HEAD_ARGMAX_WAVE8_ARGS = (
    *(ctypes.c_void_p for _ in range(6)),
    ctypes.c_int64,  # in_features
    ctypes.c_int64,  # vocab_size
    ctypes.c_void_p,  # stream
)
_LM_HEAD_ARGMAX_WAVE8_BATCH_ARGS = (
    *(ctypes.c_void_p for _ in range(6)),
    ctypes.c_int64,  # in_features
    ctypes.c_int64,  # vocab_size
    ctypes.c_int64,  # batch
    ctypes.c_void_p,  # stream
)


def plan_moonshine_lm_head_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
) -> BuildArtifact:
    return plan_cuda_build(
        sources=[_SOURCE],
        family="cuda_sm120a_moonshine_lm_head",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        target_arch=_TARGET_ARCH,
        output_name=_OUTPUT_NAME,
    )


def build_moonshine_lm_head(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
    dry_run: bool = False,
    load: bool = True,
    require_cached: bool = False,
) -> ctypes.CDLL | BuildArtifact:
    return build_cuda(
        sources=[_SOURCE],
        family="cuda_sm120a_moonshine_lm_head",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        target_arch=_TARGET_ARCH,
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def lm_head_argmax_scratch_elements(vocab_size: int, rows_per_block: int) -> int:
    """Number of (value, index) partial pairs written by the stage-1 grid."""
    return (vocab_size + rows_per_block - 1) // rows_per_block


def moonshine_lm_head_argmax_fp16(
    input_ptr: int,
    weight_ptr: int,
    block_values_ptr: int,
    block_indices_ptr: int,
    out_index_ptr: int,
    out_value_ptr: int,
    in_features: int,
    vocab_size: int,
    *,
    rows_per_block: int = _DEFAULT_ROWS_PER_BLOCK,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: CudaRuntime | None = None,
) -> None:
    """Fused FP16 tied LM-head projection + stable argmax for one hidden row.

    ``input`` is one ``[in_features]`` FP16 row; ``weight`` is the tied
    ``[vocab_size, in_features]`` FP16 matrix.  The output token (lowest index
    on ties) is written to ``out_index[0]`` (int64) and its FP16 logit value
    (as FP32) to ``out_value[0]``.  ``block_values``/``block_indices`` are the
    caller-owned bounded scratch of ``lm_head_argmax_scratch_elements`` pairs.
    """

    if in_features <= 0:
        raise ValueError("in_features must be positive")
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    if rows_per_block <= 0:
        raise ValueError("rows_per_block must be positive")
    library = library or build_moonshine_lm_head(load=True)
    runtime = runtime or get_cuda_runtime()
    _launch(
        library,
        "hipengine_cuda_sm120a_moonshine_lm_head_argmax_fp16",
        _LM_HEAD_ARGMAX_ARGS,
        (
            input_ptr,
            weight_ptr,
            block_values_ptr,
            block_indices_ptr,
            out_index_ptr,
            out_value_ptr,
            in_features,
            vocab_size,
            rows_per_block,
            stream,
        ),
        runtime,
    )


def moonshine_lm_head_argmax_batch_fp16(
    input_ptr: int,
    weight_ptr: int,
    block_values_ptr: int,
    block_indices_ptr: int,
    out_index_ptr: int,
    out_value_ptr: int,
    in_features: int,
    vocab_size: int,
    batch: int,
    *,
    rows_per_block: int = _DEFAULT_ROWS_PER_BLOCK,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: CudaRuntime | None = None,
) -> None:
    """Fused FP16 tied LM-head projection + stable argmax for ``batch`` rows.

    ``input`` is ``[batch, in_features]`` FP16; ``out_index``/``out_value`` are
    ``[batch]`` (int64 / float).  Scratch ``block_values``/``block_indices`` is
    ``[batch, num_blocks]``.  Stage 1 runs the identical ordered FP32
    accumulation per row (256 threads) and stage 2 reduces per row, so the
    output tokens are bit-exact vs B sequential single-row calls.
    """

    if in_features <= 0:
        raise ValueError("in_features must be positive")
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    if rows_per_block <= 0:
        raise ValueError("rows_per_block must be positive")
    if batch <= 0:
        raise ValueError("batch must be positive")
    library = library or build_moonshine_lm_head(load=True)
    runtime = runtime or get_cuda_runtime()
    _launch(
        library,
        "hipengine_cuda_sm120a_moonshine_lm_head_argmax_batch_fp16",
        _LM_HEAD_ARGMAX_BATCH_ARGS,
        (
            input_ptr,
            weight_ptr,
            block_values_ptr,
            block_indices_ptr,
            out_index_ptr,
            out_value_ptr,
            in_features,
            vocab_size,
            rows_per_block,
            batch,
            stream,
        ),
        runtime,
    )


def lm_head_argmax_wave8_scratch_elements(vocab_size: int) -> int:
    """(value, index) partial pairs for the fused wave8 stage-1 grid.

    One warp per vocab column, 8 warps per block -> ``ceil(vocab_size/8)``
    blocks, each writing one partial pair.
    """
    return (vocab_size + 7) // 8


def moonshine_lm_head_argmax_wave8_fp16(
    input_ptr: int,
    weight_ptr: int,
    block_values_ptr: int,
    block_indices_ptr: int,
    out_index_ptr: int,
    out_value_ptr: int,
    in_features: int,
    vocab_size: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: CudaRuntime | None = None,
) -> None:
    """Fused wave8 + stable top-1 LM head for one hidden row (C6/RR-8).

    Same signature and bounded scratch contract as
    ``moonshine_lm_head_argmax_fp16`` (scratch size from
    ``lm_head_argmax_wave8_scratch_elements``), but stage 1 uses the wave8
    arithmetic: 8 columns progress in parallel per 256-thread block (one warp
    per column, lane-stride-32 FP32 accumulation + warp butterfly, no per-column
    barrier or cross-warp serial sum), so logits differ from the exact fused
    stage at FP32-reassociation level.  The stable lowest-index tie break and
    the stage-2 reduction are identical.  This is the fused form of the
    C6-screened ``moonshine_f16_lm_head_projection_wave8`` projection.
    """

    if in_features <= 0:
        raise ValueError("in_features must be positive")
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    library = library or build_moonshine_lm_head(load=True)
    runtime = runtime or get_cuda_runtime()
    _launch(
        library,
        "hipengine_cuda_sm120a_moonshine_lm_head_argmax_wave8_fp16",
        _LM_HEAD_ARGMAX_WAVE8_ARGS,
        (
            input_ptr,
            weight_ptr,
            block_values_ptr,
            block_indices_ptr,
            out_index_ptr,
            out_value_ptr,
            in_features,
            vocab_size,
            stream,
        ),
        runtime,
    )


def moonshine_lm_head_argmax_wave8_batch_fp16(
    input_ptr: int,
    weight_ptr: int,
    block_values_ptr: int,
    block_indices_ptr: int,
    out_index_ptr: int,
    out_value_ptr: int,
    in_features: int,
    vocab_size: int,
    batch: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: CudaRuntime | None = None,
) -> None:
    """Fused wave8 + stable top-1 LM head for ``batch`` rows (C6/RR-8).

    ``input`` is ``[batch, in_features]`` FP16; ``out_index``/``out_value`` are
    ``[batch]``.  Scratch is ``[batch, ceil(vocab/8)]``.  Grid-Y is the batch
    row; each row's wave8 arithmetic, tie break, and reduction are identical to
    the single-row variant.
    """

    if in_features <= 0:
        raise ValueError("in_features must be positive")
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    if batch <= 0:
        raise ValueError("batch must be positive")
    library = library or build_moonshine_lm_head(load=True)
    runtime = runtime or get_cuda_runtime()
    _launch(
        library,
        "hipengine_cuda_sm120a_moonshine_lm_head_argmax_wave8_batch_fp16",
        _LM_HEAD_ARGMAX_WAVE8_BATCH_ARGS,
        (
            input_ptr,
            weight_ptr,
            block_values_ptr,
            block_indices_ptr,
            out_index_ptr,
            out_value_ptr,
            in_features,
            vocab_size,
            batch,
            stream,
        ),
        runtime,
    )


def _launch(
    library: ctypes.CDLL,
    symbol: str,
    argtypes,
    arguments: tuple[object, ...],
    runtime: CudaRuntime,
) -> None:
    function = signed_kernel_fn(library, symbol, argtypes, ctypes.c_int)
    error = function(*arguments)
    if int(error) != CUDA_SUCCESS:
        runtime.check(int(error))


def register_moonshine_lm_head_kernels(*, replace: bool = True) -> None:
    registrations = (
        (
            KernelKey(_BACKEND, "moonshine_lm_head", "fp16", "fused_argmax_fp32_accum"),
            moonshine_lm_head_argmax_fp16,
        ),
        (
            KernelKey(
                _BACKEND,
                "moonshine_lm_head",
                "fp16",
                "fused_argmax_fp32_accum_batch",
            ),
            moonshine_lm_head_argmax_batch_fp16,
        ),
        (
            KernelKey(
                _BACKEND,
                "moonshine_lm_head",
                "fp16",
                "fused_argmax_wave8",
            ),
            moonshine_lm_head_argmax_wave8_fp16,
        ),
        (
            KernelKey(
                _BACKEND,
                "moonshine_lm_head",
                "fp16",
                "fused_argmax_wave8_batch",
            ),
            moonshine_lm_head_argmax_wave8_batch_fp16,
        ),
    )
    for key, kernel in registrations:
        register(key, kernel, replace=replace)


register_moonshine_lm_head_kernels()

__all__ = [
    "build_moonshine_lm_head",
    "lm_head_argmax_scratch_elements",
    "lm_head_argmax_wave8_scratch_elements",
    "moonshine_lm_head_argmax_fp16",
    "moonshine_lm_head_argmax_batch_fp16",
    "moonshine_lm_head_argmax_wave8_fp16",
    "moonshine_lm_head_argmax_wave8_batch_fp16",
    "plan_moonshine_lm_head_build",
    "register_moonshine_lm_head_kernels",
]
