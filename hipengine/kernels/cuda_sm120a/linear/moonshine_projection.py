"""Raw-pointer Moonshine source-F16 projection baselines for CUDA ``sm_120a``."""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_cuda, plan_cuda_build
from hipengine.core.ctypes_cache import signed_kernel_fn
from hipengine.core.cuda import CUDA_SUCCESS, CudaRuntime, get_cuda_runtime
from hipengine.kernels.backends import cuda_target_arch_for_backend
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("moonshine_projection.cu")
_OUTPUT_NAME = "moonshine_projection.so"
_BACKEND = "cuda_sm120a"
_TARGET_ARCH = cuda_target_arch_for_backend(_BACKEND)
_ALLOWED_THREADS = {32, 64, 128, 256}
# Measured on exclusive GPU0 (RTX PRO 6000 Blackwell, driver 610.43.03) with a
# batch-timed CUDA-event screen (medians over 200-launch batches, 40 at 1,248
# rows) at threads {32,64,128,256} x rows {40,207,1248}. 64 threads is the
# best family schedule for the pair/head-major row projections across every
# bucket (1.80-2.15x faster than the inherited 256 default); single/triple at
# M=1 are flat across threads so they keep 256. The fused fc2
# (bias_residual 1664->416) is best at 256 threads for the decode M=1 bucket
# and at 64 for batch M=40.
#
# C4-R3 note: a leaf screen also selected t64 for the single/triple/bias row
# projections at rows>1 (1.79-2.00x faster than 256), but the complete-encoder
# token gate rejects it.  Forcing t64 on the encoder's single/triple/bias flips
# 45/33 tokens (audio-sumimasen / synthetic-1s-seed1234) and t128 flips 27
# (audio-konichiwa); only the t256 reduction order reproduces the exact fixture
# token stream on every file (0 mismatches, one documented borderline).  The
# encoder composition therefore keeps 256 for these families.  The t64
# pair/head-major schedule is retained: the decoder cross-K/V precompute (the
# only rows>1 pair user) is token-exact at t64.
_PAIR_THREADS = 64
_RESIDUAL_DECODE_THREADS = 256
_RESIDUAL_BATCH_THREADS = 64
_RESIDUAL_BATCH_THRESHOLD = 1


def _default_pair_threads() -> int:
    """Measured thread schedule for pair/head-major row projections.

    A batch-timed screen shows 64 threads is 1.80-2.15x faster than 256 across
    the 40/207/1,248-row production buckets on the target, so the auto-select
    is a flat 64. Explicit ``threads=`` always overrides.
    """
    return _PAIR_THREADS


def _default_residual_threads(rows: int) -> int:
    """Measured thread schedule for the fused fc2 (bias_residual) boundary.

    Batch-timed screen: 256 threads is ~1.19-1.27x faster than 64 for the
    auto-regressive decode bucket (M=1) and 64 is best at M=40. Explicit
    ``threads=`` always overrides.
    """
    if rows <= _RESIDUAL_BATCH_THRESHOLD:
        return _RESIDUAL_DECODE_THREADS
    return _RESIDUAL_BATCH_THREADS
_SINGLE_ARGS = (
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_void_p,
)
_LM_HEAD_WAVE8_ARGS = (
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_void_p,
)
_BIAS_ARGS = (
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
_BIAS_RESIDUAL_ARGS = (
    *(ctypes.c_void_p for _ in range(5)),
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_void_p,
)
_PAIR_ARGS = (
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
_PAIR_HEAD_MAJOR_ARGS = (
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
    ctypes.c_int64,
    ctypes.c_void_p,
)
_PAIR_HEAD_MAJOR_BATCH_ARGS = (
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
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_void_p,
)
_TRIPLE_ARGS = (
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
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_void_p,
)


def plan_moonshine_projection_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
) -> BuildArtifact:
    return plan_cuda_build(
        sources=[_SOURCE],
        family="cuda_sm120a_moonshine_projection",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        target_arch=_TARGET_ARCH,
        output_name=_OUTPUT_NAME,
    )


def build_moonshine_projection(
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
        family="cuda_sm120a_moonshine_projection",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        target_arch=_TARGET_ARCH,
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def _validate(
    rows: int,
    in_features: int,
    outputs: tuple[int, ...],
    threads: int,
) -> None:
    if rows <= 0:
        raise ValueError("rows must be positive")
    if in_features <= 0:
        raise ValueError("in_features must be positive")
    if any(width <= 0 for width in outputs):
        raise ValueError("out_features must be positive")
    if threads not in _ALLOWED_THREADS:
        raise ValueError("threads must be one of 32, 64, 128, 256")


def _launch(
    library: ctypes.CDLL,
    symbol: str,
    argtypes,
    arguments: tuple[object, ...],
    runtime: CudaRuntime,
) -> None:
    fn = signed_kernel_fn(library, symbol, argtypes, ctypes.c_int)
    error = fn(*arguments)
    if int(error) != CUDA_SUCCESS:
        runtime.check(int(error))


def moonshine_f16_projection(
    input_ptr: int,
    weight_ptr: int,
    output_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    threads: int = 256,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: CudaRuntime | None = None,
) -> None:
    _validate(rows, in_features, (out_features,), threads)
    library = library or build_moonshine_projection(load=True)
    runtime = runtime or get_cuda_runtime()
    _launch(
        library,
        "hipengine_cuda_sm120a_moonshine_f16_projection",
        _SINGLE_ARGS,
        (
            input_ptr,
            weight_ptr,
            output_ptr,
            rows,
            in_features,
            out_features,
            threads,
            stream,
        ),
        runtime,
    )


def moonshine_f16_lm_head_projection(
    input_ptr: int,
    weight_ptr: int,
    output_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    threads: int = 256,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: CudaRuntime | None = None,
) -> None:
    _validate(rows, in_features, (out_features,), threads)
    library = library or build_moonshine_projection(load=True)
    runtime = runtime or get_cuda_runtime()
    _launch(
        library,
        "hipengine_cuda_sm120a_moonshine_f16_lm_head_projection",
        _SINGLE_ARGS,
        (
            input_ptr,
            weight_ptr,
            output_ptr,
            rows,
            in_features,
            out_features,
            threads,
            stream,
        ),
        runtime,
    )


def moonshine_f16_lm_head_projection_wave8(
    input_ptr: int,
    weight_ptr: int,
    output_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: CudaRuntime | None = None,
) -> None:
    _validate(rows, in_features, (out_features,), 32)
    library = library or build_moonshine_projection(load=True)
    runtime = runtime or get_cuda_runtime()
    _launch(
        library,
        "hipengine_cuda_sm120a_moonshine_f16_lm_head_projection_wave8",
        _LM_HEAD_WAVE8_ARGS,
        (input_ptr, weight_ptr, output_ptr, rows, in_features, out_features, stream),
        runtime,
    )


def moonshine_f16_projection_bias(
    input_ptr: int,
    weight_ptr: int,
    bias_ptr: int,
    output_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    threads: int = 256,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: CudaRuntime | None = None,
) -> None:
    _validate(rows, in_features, (out_features,), threads)
    library = library or build_moonshine_projection(load=True)
    runtime = runtime or get_cuda_runtime()
    _launch(
        library,
        "hipengine_cuda_sm120a_moonshine_f16_projection_bias",
        _BIAS_ARGS,
        (
            input_ptr,
            weight_ptr,
            bias_ptr,
            output_ptr,
            rows,
            in_features,
            out_features,
            threads,
            stream,
        ),
        runtime,
    )


def moonshine_f16_projection_bias_gated_silu(
    input_ptr: int,
    weight_ptr: int,
    bias_ptr: int,
    output_ptr: int,
    rows: int,
    in_features: int,
    intermediate_size: int,
    *,
    threads: int = 32,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: CudaRuntime | None = None,
) -> None:
    _validate(rows, in_features, (intermediate_size,), threads)
    if threads != 32:
        raise ValueError("threads must be 32 for paired gated-SiLU projection")
    library = library or build_moonshine_projection(load=True)
    runtime = runtime or get_cuda_runtime()
    _launch(
        library,
        "hipengine_cuda_sm120a_moonshine_f16_projection_bias_gated_silu",
        _BIAS_ARGS,
        (
            input_ptr,
            weight_ptr,
            bias_ptr,
            output_ptr,
            rows,
            in_features,
            intermediate_size,
            threads,
            stream,
        ),
        runtime,
    )


def moonshine_f16_projection_bias_residual(
    input_ptr: int,
    weight_ptr: int,
    bias_ptr: int,
    residual_ptr: int,
    output_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    threads: int | None = None,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: CudaRuntime | None = None,
) -> None:
    threads = _default_residual_threads(rows) if threads is None else threads
    _validate(rows, in_features, (out_features,), threads)
    library = library or build_moonshine_projection(load=True)
    runtime = runtime or get_cuda_runtime()
    _launch(
        library,
        "hipengine_cuda_sm120a_moonshine_f16_projection_bias_residual",
        _BIAS_RESIDUAL_ARGS,
        (
            input_ptr,
            weight_ptr,
            bias_ptr,
            residual_ptr,
            output_ptr,
            rows,
            in_features,
            out_features,
            threads,
            stream,
        ),
        runtime,
    )


def moonshine_f16_projection_pair(
    input_ptr: int,
    weight_a_ptr: int,
    weight_b_ptr: int,
    output_a_ptr: int,
    output_b_ptr: int,
    rows: int,
    in_features: int,
    out_a_features: int,
    out_b_features: int,
    *,
    threads: int | None = None,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: CudaRuntime | None = None,
) -> None:
    threads = _default_pair_threads() if threads is None else threads
    _validate(rows, in_features, (out_a_features, out_b_features), threads)
    library = library or build_moonshine_projection(load=True)
    runtime = runtime or get_cuda_runtime()
    _launch(
        library,
        "hipengine_cuda_sm120a_moonshine_f16_projection_pair",
        _PAIR_ARGS,
        (
            input_ptr,
            weight_a_ptr,
            weight_b_ptr,
            output_a_ptr,
            output_b_ptr,
            rows,
            in_features,
            out_a_features,
            out_b_features,
            threads,
            stream,
        ),
        runtime,
    )


def moonshine_f16_projection_pair_head_major(
    input_ptr: int,
    weight_a_ptr: int,
    weight_b_ptr: int,
    output_a_ptr: int,
    output_b_ptr: int,
    rows: int,
    in_features: int,
    out_a_features: int,
    out_b_features: int,
    head_dim: int,
    *,
    threads: int | None = None,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: CudaRuntime | None = None,
) -> None:
    threads = _default_pair_threads() if threads is None else threads
    _validate(rows, in_features, (out_a_features, out_b_features), threads)
    if head_dim <= 0 or out_a_features % head_dim or out_b_features % head_dim:
        raise ValueError("head_dim must positively divide both output widths")
    library = library or build_moonshine_projection(load=True)
    runtime = runtime or get_cuda_runtime()
    _launch(
        library,
        "hipengine_cuda_sm120a_moonshine_f16_projection_pair_head_major",
        _PAIR_HEAD_MAJOR_ARGS,
        (
            input_ptr,
            weight_a_ptr,
            weight_b_ptr,
            output_a_ptr,
            output_b_ptr,
            rows,
            in_features,
            out_a_features,
            out_b_features,
            head_dim,
            threads,
            stream,
        ),
        runtime,
    )


def moonshine_f16_projection_pair_head_major_batch(
    input_ptr: int,
    weight_a_ptr: int,
    weight_b_ptr: int,
    output_a_ptr: int,
    output_b_ptr: int,
    batch: int,
    rows: int,
    output_frames: int,
    in_features: int,
    out_a_features: int,
    out_b_features: int,
    head_dim: int,
    *,
    threads: int | None = None,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: CudaRuntime | None = None,
) -> None:
    """Batch pair/head-major projection over ``[batch, rows, in_features]``.

    Writes batch head-major ``[B, heads, output_frames, head_dim]`` K/V
    directly into the decoder's batch-strided cross cache (C8 phase 2).
    ``rows`` is the per-plane computed frame count; ``output_frames`` is the
    decoder bucket capacity (``output_frames >= rows``, tail pre-zeroed).
    """

    if batch <= 0:
        raise ValueError("batch must be positive")
    threads = _default_pair_threads() if threads is None else threads
    _validate(rows, in_features, (out_a_features, out_b_features), threads)
    if output_frames < rows:
        raise ValueError("output_frames must be >= rows")
    if head_dim <= 0 or out_a_features % head_dim or out_b_features % head_dim:
        raise ValueError("head_dim must positively divide both output widths")
    library = library or build_moonshine_projection(load=True)
    runtime = runtime or get_cuda_runtime()
    _launch(
        library,
        "hipengine_cuda_sm120a_moonshine_f16_projection_pair_head_major_batch",
        _PAIR_HEAD_MAJOR_BATCH_ARGS,
        (
            input_ptr,
            weight_a_ptr,
            weight_b_ptr,
            output_a_ptr,
            output_b_ptr,
            batch,
            rows,
            output_frames,
            in_features,
            out_a_features,
            out_b_features,
            head_dim,
            threads,
            stream,
        ),
        runtime,
    )


def moonshine_f16_projection_triple(
    input_ptr: int,
    weight_a_ptr: int,
    weight_b_ptr: int,
    weight_c_ptr: int,
    output_a_ptr: int,
    output_b_ptr: int,
    output_c_ptr: int,
    rows: int,
    in_features: int,
    out_a_features: int,
    out_b_features: int,
    out_c_features: int,
    *,
    threads: int = 256,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: CudaRuntime | None = None,
) -> None:
    _validate(
        rows,
        in_features,
        (out_a_features, out_b_features, out_c_features),
        threads,
    )
    library = library or build_moonshine_projection(load=True)
    runtime = runtime or get_cuda_runtime()
    _launch(
        library,
        "hipengine_cuda_sm120a_moonshine_f16_projection_triple",
        _TRIPLE_ARGS,
        (
            input_ptr,
            weight_a_ptr,
            weight_b_ptr,
            weight_c_ptr,
            output_a_ptr,
            output_b_ptr,
            output_c_ptr,
            rows,
            in_features,
            out_a_features,
            out_b_features,
            out_c_features,
            threads,
            stream,
        ),
        runtime,
    )


def register_moonshine_projection_kernels(*, replace: bool = True) -> None:
    registrations = (
        (
            KernelKey(
                _BACKEND,
                "moonshine_projection",
                "fp16",
                "single_fp32_accum",
            ),
            moonshine_f16_projection,
        ),
        (
            KernelKey(
                _BACKEND,
                "moonshine_lm_head",
                "fp16",
                "tied_fp32_accum",
            ),
            moonshine_f16_lm_head_projection,
        ),
        (
            KernelKey(
                _BACKEND,
                "moonshine_lm_head",
                "fp16",
                "tied_wave8_fp32_accum",
            ),
            moonshine_f16_lm_head_projection_wave8,
        ),
        (
            KernelKey(
                _BACKEND,
                "moonshine_projection_rows",
                "fp16",
                "single_fp32_accum",
            ),
            moonshine_f16_projection,
        ),
        (
            KernelKey(
                _BACKEND,
                "moonshine_projection_bias",
                "fp16",
                "single_fp32_accum",
            ),
            moonshine_f16_projection_bias,
        ),
        (
            KernelKey(
                _BACKEND,
                "moonshine_mlp_fc1",
                "fp16",
                "bias_gated_silu_fp32_accum",
            ),
            moonshine_f16_projection_bias_gated_silu,
        ),
        (
            KernelKey(
                _BACKEND,
                "moonshine_mlp_fc2_residual",
                "fp16",
                "bias_rounded_residual_fp32_accum",
            ),
            moonshine_f16_projection_bias_residual,
        ),
        (
            KernelKey(
                _BACKEND,
                "moonshine_projection_pair",
                "fp16",
                "pair_fp32_accum",
            ),
            moonshine_f16_projection_pair,
        ),
        (
            KernelKey(
                _BACKEND,
                "moonshine_cross_kv_precompute",
                "fp16",
                "pair_head_major_fp32_accum",
            ),
            moonshine_f16_projection_pair_head_major,
        ),
        (
            KernelKey(
                _BACKEND,
                "moonshine_cross_kv_precompute",
                "fp16",
                "pair_head_major_batch_fp32_accum",
            ),
            moonshine_f16_projection_pair_head_major_batch,
        ),
        (
            KernelKey(
                _BACKEND,
                "moonshine_qkv_proj",
                "fp16",
                "triple_fp32_accum",
            ),
            moonshine_f16_projection_triple,
        ),
    )
    for key, kernel in registrations:
        register(key, kernel, replace=replace)


register_moonshine_projection_kernels()

__all__ = [
    "build_moonshine_projection",
    "moonshine_f16_lm_head_projection",
    "moonshine_f16_lm_head_projection_wave8",
    "moonshine_f16_projection",
    "moonshine_f16_projection_bias",
    "moonshine_f16_projection_bias_gated_silu",
    "moonshine_f16_projection_bias_residual",
    "moonshine_f16_projection_pair",
    "moonshine_f16_projection_pair_head_major",
    "moonshine_f16_projection_pair_head_major_batch",
    "moonshine_f16_projection_triple",
    "plan_moonshine_projection_build",
    "register_moonshine_projection_kernels",
]
