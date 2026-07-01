"""Raw-pointer wrappers for GGUF Q6_K dense pack8 GEMV decode.

P9.B4b: dense single-output decode-shaped pack8 GEMV for raw GGUF Q6_K
weights. The qwen35moe Qwen3.6-35B-A3B-UD-Q4_K_M lm-head ties to a Q6_K
output projection, so the F32 output variant is the production path the
sampler reads logits from. Mirrors the Q4_K dense kernel (P9.B4) with the
inner k loop swapped for raw GGUF Q6_K block dequant (`int8` per-16-K
scales x `fp16` super-scale, with 2 high bits per element).

Four launch entry points are registered: BF16/BF16, FP16/FP16, BF16/F32,
FP16/F32. No new ABI and no resident weight sidecar/repack.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("gguf_q6_k_pack8_gemv.hip")
_OUTPUT_NAME = "gguf_q6_k_pack8_gemv.so"
_SYM_BF16_BF16 = "hipengine_gguf_q6_k_pack8_gemv_decode_bf16_bf16_out"
_SYM_FP16_FP16 = "hipengine_gguf_q6_k_pack8_gemv_decode_fp16_fp16_out"
_SYM_BF16_F32 = "hipengine_gguf_q6_k_pack8_gemv_decode_bf16_f32_out"
_SYM_FP16_F32 = "hipengine_gguf_q6_k_pack8_gemv_decode_fp16_f32_out"
_SYM_BF16_TOP1_GATHER_F32 = "hipengine_gguf_q6_k_pack8_gemv_decode_bf16_top1_gather_f32"
_SYM_BF16_TOP1_GATHER_F32_THREADS = "hipengine_gguf_q6_k_pack8_gemv_decode_bf16_top1_gather_f32_threads"
_SYM_BF16_TOP1_STAGE1_F32 = "hipengine_gguf_q6_k_pack8_gemv_decode_bf16_top1_stage1_f32"
_SYM_Q8_1_DP4A_TOP1_GATHER_F32 = "hipengine_gguf_q6_k_pack8_gemv_decode_q8_1_dp4a_top1_gather_f32"
_SYM_Q8_1_DP4A_TOP1_GATHER_F32_THREADS = (
    "hipengine_gguf_q6_k_pack8_gemv_decode_q8_1_dp4a_top1_gather_f32_threads"
)
_SYM_Q8_1_DP4A_TOP1_STAGE1_F32 = "hipengine_gguf_q6_k_pack8_gemv_decode_q8_1_dp4a_top1_stage1_f32"
_SYM_Q8_1_DP4A_TOP1_SCALEHOIST_STAGE1_F32 = (
    "hipengine_gguf_q6_k_pack8_gemv_decode_q8_1_dp4a_top1_scalehoist_stage1_f32"
)
_SYM_Q8_1_DP4A_TOP1_SCALEHOIST_GATHER_F32_THREADS = (
    "hipengine_gguf_q6_k_pack8_gemv_decode_q8_1_dp4a_top1_scalehoist_gather_f32_threads"
)
_SYM_Q8_1_DP4A_TOP1_ROW_STAGE1_F32 = (
    "hipengine_gguf_q6_k_pack8_gemv_decode_q8_1_dp4a_top1_row_stage1_f32"
)
_SYM_Q8_1_DP4A_TOP1_ROW_GATHER_F32 = (
    "hipengine_gguf_q6_k_pack8_gemv_decode_q8_1_dp4a_top1_row_gather_f32"
)
_SYM_TOP1_STAGE2_GATHER_F32 = "hipengine_gguf_q6_k_pack8_top1_stage2_gather_f32"
_Q6_K_BLOCK = 256
_Q6_TOP1_STAGE1_THREADS_ENV = "HIPENGINE_GGUF_Q6_TOP1_STAGE1_THREADS"


def plan_gguf_q6_k_pack8_gemv_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="gguf_q6_k_pack8_gemv",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        extra_flags=("-mcumode",),
        output_name=_OUTPUT_NAME,
    )


def build_gguf_q6_k_pack8_gemv(
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
        family="gguf_q6_k_pack8_gemv",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        extra_flags=("-mcumode",),
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def _make_launch(symbol: str):
    def launch(
        x_ptr: int,
        qweight_ptr: int,
        out_ptr: int,
        rows: int,
        in_features: int,
        out_features: int,
        *,
        stream: int = 0,
        library: ctypes.CDLL | None = None,
        runtime: HipRuntime | None = None,
    ) -> None:
        _check_common(rows, in_features, out_features)
        library = library or build_gguf_q6_k_pack8_gemv(load=True)
        runtime = runtime or get_hip_runtime()
        fn = getattr(library, symbol)
        fn.argtypes = [
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
            ctypes.c_void_p(qweight_ptr),
            ctypes.c_void_p(out_ptr),
            ctypes.c_int64(rows),
            ctypes.c_int64(in_features),
            ctypes.c_int64(out_features),
            ctypes.c_void_p(stream),
        )
        if int(err) != HIP_SUCCESS:
            runtime.check(int(err))

    launch.__name__ = symbol[len("hipengine_") :]
    return launch


gguf_q6_k_pack8_gemv_decode_bf16_bf16_out = _make_launch(_SYM_BF16_BF16)
gguf_q6_k_pack8_gemv_decode_fp16_fp16_out = _make_launch(_SYM_FP16_FP16)
gguf_q6_k_pack8_gemv_decode_bf16_f32_out = _make_launch(_SYM_BF16_F32)
gguf_q6_k_pack8_gemv_decode_fp16_f32_out = _make_launch(_SYM_FP16_F32)


def _stage1_threads(value: int | None = None) -> int:
    raw = str(value if value is not None else os.environ.get(_Q6_TOP1_STAGE1_THREADS_ENV, "128")).strip()
    try:
        threads = int(raw)
    except ValueError as exc:
        raise ValueError(f"{_Q6_TOP1_STAGE1_THREADS_ENV} must be 64 or 128") from exc
    if threads not in (64, 128):
        raise ValueError("Q6 top-1 stage1 threads must be 64 or 128")
    return threads


def gguf_q6_k_pack8_gemv_decode_bf16_top1_gather_f32(
    x_ptr: int,
    qweight_ptr: int,
    block_values_f32_ptr: int,
    block_indices_i32_ptr: int,
    out_indices_i32_ptr: int,
    out_values_f32_ptr: int | None,
    embed_table_f32_ptr: int | None,
    next_embed_f32_ptr: int | None,
    rows: int,
    in_features: int,
    out_features: int,
    hidden_size: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
    stage1_threads: int | None = None,
) -> None:
    """Write top-1 ids/values for a BF16 x Q6_K head and optionally gather embeddings.

    This is an exact top-1 specialization for speculative MTP drafting.  It keeps
    the Q6_K dot-product math and tie-break semantics of the unfused
    ``bf16_f32_out -> topk_f32_rows_i32`` path, but only materializes one
    candidate per pack8 output block plus a final row winner.
    """

    _check_common(rows, in_features, out_features)
    if hidden_size < 0:
        raise ValueError("hidden_size must be non-negative")
    has_embed = embed_table_f32_ptr is not None or next_embed_f32_ptr is not None
    if has_embed and (embed_table_f32_ptr is None or next_embed_f32_ptr is None):
        raise ValueError("embed_table_f32_ptr and next_embed_f32_ptr must be provided together")
    if has_embed and hidden_size <= 0:
        raise ValueError("hidden_size must be positive when gathering embeddings")
    if out_features > 2**31 - 1:
        raise ValueError("out_features must fit in int32 for top-1 indices")
    threads = _stage1_threads(stage1_threads)
    library = library or build_gguf_q6_k_pack8_gemv(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYM_BF16_TOP1_GATHER_F32_THREADS)
    fn.argtypes = [
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
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int32,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(x_ptr),
        ctypes.c_void_p(qweight_ptr),
        ctypes.c_void_p(block_values_f32_ptr),
        ctypes.c_void_p(block_indices_i32_ptr),
        ctypes.c_void_p(out_indices_i32_ptr),
        ctypes.c_void_p(out_values_f32_ptr) if out_values_f32_ptr is not None else ctypes.c_void_p(),
        ctypes.c_void_p(embed_table_f32_ptr) if embed_table_f32_ptr is not None else ctypes.c_void_p(),
        ctypes.c_void_p(next_embed_f32_ptr) if next_embed_f32_ptr is not None else ctypes.c_void_p(),
        ctypes.c_int64(rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features),
        ctypes.c_int64(hidden_size),
        ctypes.c_int32(threads),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def gguf_q6_k_pack8_gemv_decode_q8_1_dp4a_top1_gather_f32(
    xq_ptr: int,
    qweight_ptr: int,
    block_values_f32_ptr: int,
    block_indices_i32_ptr: int,
    out_indices_i32_ptr: int,
    out_values_f32_ptr: int | None,
    embed_table_f32_ptr: int | None,
    next_embed_f32_ptr: int | None,
    rows: int,
    in_features: int,
    out_features: int,
    hidden_size: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
    stage1_threads: int | None = None,
) -> None:
    """Write approximate q8_1/dp4a Q6_K top-1 ids and optionally gather embeddings.

    ``xq_ptr`` must point at GGML-compatible q8_1 activation blocks, for example
    those produced by ``gguf_q4_k_quantize_bf16_q8_1``.  This is the
    accuracy-traded llama-compat draft lm-head path; the exact BF16 wrapper above
    remains the default path.
    """

    _check_common(rows, in_features, out_features)
    if in_features % 32 != 0:
        raise ValueError("in_features must be divisible by q8_1 block size 32")
    if hidden_size < 0:
        raise ValueError("hidden_size must be non-negative")
    has_embed = embed_table_f32_ptr is not None or next_embed_f32_ptr is not None
    if has_embed and (embed_table_f32_ptr is None or next_embed_f32_ptr is None):
        raise ValueError("embed_table_f32_ptr and next_embed_f32_ptr must be provided together")
    if has_embed and hidden_size <= 0:
        raise ValueError("hidden_size must be positive when gathering embeddings")
    if out_features > 2**31 - 1:
        raise ValueError("out_features must fit in int32 for top-1 indices")
    threads = _stage1_threads(stage1_threads)
    library = library or build_gguf_q6_k_pack8_gemv(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYM_Q8_1_DP4A_TOP1_GATHER_F32_THREADS)
    fn.argtypes = [
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
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int32,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(xq_ptr),
        ctypes.c_void_p(qweight_ptr),
        ctypes.c_void_p(block_values_f32_ptr),
        ctypes.c_void_p(block_indices_i32_ptr),
        ctypes.c_void_p(out_indices_i32_ptr),
        ctypes.c_void_p(out_values_f32_ptr) if out_values_f32_ptr is not None else ctypes.c_void_p(),
        ctypes.c_void_p(embed_table_f32_ptr) if embed_table_f32_ptr is not None else ctypes.c_void_p(),
        ctypes.c_void_p(next_embed_f32_ptr) if next_embed_f32_ptr is not None else ctypes.c_void_p(),
        ctypes.c_int64(rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features),
        ctypes.c_int64(hidden_size),
        ctypes.c_int32(threads),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def gguf_q6_k_pack8_gemv_decode_q8_1_dp4a_top1_scalehoist_gather_f32(
    xq_ptr: int,
    qweight_ptr: int,
    block_values_f32_ptr: int,
    block_indices_i32_ptr: int,
    out_indices_i32_ptr: int,
    out_values_f32_ptr: int | None,
    embed_table_f32_ptr: int | None,
    next_embed_f32_ptr: int | None,
    rows: int,
    in_features: int,
    out_features: int,
    hidden_size: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
    stage1_threads: int | None = None,
) -> None:
    """Diagnostic q8_1/dp4a Q6_K top-1 path with pack8 scale hoisting."""

    _check_common(rows, in_features, out_features)
    if in_features % 32 != 0:
        raise ValueError("in_features must be divisible by q8_1 block size 32")
    if hidden_size < 0:
        raise ValueError("hidden_size must be non-negative")
    has_embed = embed_table_f32_ptr is not None or next_embed_f32_ptr is not None
    if has_embed and (embed_table_f32_ptr is None or next_embed_f32_ptr is None):
        raise ValueError("embed_table_f32_ptr and next_embed_f32_ptr must be provided together")
    if has_embed and hidden_size <= 0:
        raise ValueError("hidden_size must be positive when gathering embeddings")
    if out_features > 2**31 - 1:
        raise ValueError("out_features must fit in int32 for top-1 indices")
    threads = _stage1_threads(stage1_threads)
    library = library or build_gguf_q6_k_pack8_gemv(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYM_Q8_1_DP4A_TOP1_SCALEHOIST_GATHER_F32_THREADS)
    fn.argtypes = [
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
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int32,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(xq_ptr),
        ctypes.c_void_p(qweight_ptr),
        ctypes.c_void_p(block_values_f32_ptr),
        ctypes.c_void_p(block_indices_i32_ptr),
        ctypes.c_void_p(out_indices_i32_ptr),
        ctypes.c_void_p(out_values_f32_ptr) if out_values_f32_ptr is not None else ctypes.c_void_p(),
        ctypes.c_void_p(embed_table_f32_ptr) if embed_table_f32_ptr is not None else ctypes.c_void_p(),
        ctypes.c_void_p(next_embed_f32_ptr) if next_embed_f32_ptr is not None else ctypes.c_void_p(),
        ctypes.c_int64(rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features),
        ctypes.c_int64(hidden_size),
        ctypes.c_int32(threads),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def gguf_q6_k_pack8_gemv_decode_bf16_top1_stage1_f32(
    x_ptr: int,
    qweight_ptr: int,
    block_values_f32_ptr: int,
    block_indices_i32_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
    stage1_threads: int | None = None,
) -> None:
    """Diagnostic stage1-only BF16 Q6_K top-1 launch."""

    _check_common(rows, in_features, out_features)
    if out_features > 2**31 - 1:
        raise ValueError("out_features must fit in int32 for top-1 indices")
    threads = _stage1_threads(stage1_threads)
    library = library or build_gguf_q6_k_pack8_gemv(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYM_BF16_TOP1_STAGE1_F32)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int32,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(x_ptr),
        ctypes.c_void_p(qweight_ptr),
        ctypes.c_void_p(block_values_f32_ptr),
        ctypes.c_void_p(block_indices_i32_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features),
        ctypes.c_int32(threads),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def gguf_q6_k_pack8_gemv_decode_q8_1_dp4a_top1_stage1_f32(
    xq_ptr: int,
    qweight_ptr: int,
    block_values_f32_ptr: int,
    block_indices_i32_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
    stage1_threads: int | None = None,
) -> None:
    """Diagnostic stage1-only q8_1/dp4a Q6_K top-1 launch."""

    _check_common(rows, in_features, out_features)
    if in_features % 32 != 0:
        raise ValueError("in_features must be divisible by q8_1 block size 32")
    if out_features > 2**31 - 1:
        raise ValueError("out_features must fit in int32 for top-1 indices")
    threads = _stage1_threads(stage1_threads)
    library = library or build_gguf_q6_k_pack8_gemv(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYM_Q8_1_DP4A_TOP1_STAGE1_F32)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int32,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(xq_ptr),
        ctypes.c_void_p(qweight_ptr),
        ctypes.c_void_p(block_values_f32_ptr),
        ctypes.c_void_p(block_indices_i32_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features),
        ctypes.c_int32(threads),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def gguf_q6_k_pack8_gemv_decode_q8_1_dp4a_top1_scalehoist_stage1_f32(
    xq_ptr: int,
    qweight_ptr: int,
    block_values_f32_ptr: int,
    block_indices_i32_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
    stage1_threads: int | None = None,
) -> None:
    """Diagnostic stage1-only q8_1/dp4a Q6_K top-1 launch with scale hoisting."""

    _check_common(rows, in_features, out_features)
    if in_features % 32 != 0:
        raise ValueError("in_features must be divisible by q8_1 block size 32")
    if out_features > 2**31 - 1:
        raise ValueError("out_features must fit in int32 for top-1 indices")
    threads = _stage1_threads(stage1_threads)
    library = library or build_gguf_q6_k_pack8_gemv(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYM_Q8_1_DP4A_TOP1_SCALEHOIST_STAGE1_F32)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int32,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(xq_ptr),
        ctypes.c_void_p(qweight_ptr),
        ctypes.c_void_p(block_values_f32_ptr),
        ctypes.c_void_p(block_indices_i32_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features),
        ctypes.c_int32(threads),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def gguf_q6_k_pack8_gemv_decode_q8_1_dp4a_top1_row_stage1_f32(
    xq_ptr: int,
    qweight_ptr: int,
    block_values_f32_ptr: int,
    block_indices_i32_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Diagnostic llama.cpp-shape q8_1/dp4a Q6_K top-1 stage1 launch."""

    _check_common(rows, in_features, out_features)
    if in_features % 32 != 0:
        raise ValueError("in_features must be divisible by q8_1 block size 32")
    if out_features > 2**31 - 1:
        raise ValueError("out_features must fit in int32 for top-1 indices")
    library = library or build_gguf_q6_k_pack8_gemv(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYM_Q8_1_DP4A_TOP1_ROW_STAGE1_F32)
    fn.argtypes = [
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
        ctypes.c_void_p(xq_ptr),
        ctypes.c_void_p(qweight_ptr),
        ctypes.c_void_p(block_values_f32_ptr),
        ctypes.c_void_p(block_indices_i32_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def gguf_q6_k_pack8_gemv_decode_q8_1_dp4a_top1_row_gather_f32(
    xq_ptr: int,
    qweight_ptr: int,
    block_values_f32_ptr: int,
    block_indices_i32_ptr: int,
    out_indices_i32_ptr: int,
    out_values_f32_ptr: int | None,
    embed_table_f32_ptr: int | None,
    next_embed_f32_ptr: int | None,
    rows: int,
    in_features: int,
    out_features: int,
    hidden_size: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Diagnostic llama.cpp-shape q8_1/dp4a Q6_K top-1 plus optional gather."""

    _check_common(rows, in_features, out_features)
    if in_features % 32 != 0:
        raise ValueError("in_features must be divisible by q8_1 block size 32")
    if hidden_size < 0:
        raise ValueError("hidden_size must be non-negative")
    has_embed = embed_table_f32_ptr is not None or next_embed_f32_ptr is not None
    if has_embed and (embed_table_f32_ptr is None or next_embed_f32_ptr is None):
        raise ValueError("embed_table_f32_ptr and next_embed_f32_ptr must be provided together")
    if has_embed and hidden_size <= 0:
        raise ValueError("hidden_size must be positive when gathering embeddings")
    if out_features > 2**31 - 1:
        raise ValueError("out_features must fit in int32 for top-1 indices")
    library = library or build_gguf_q6_k_pack8_gemv(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYM_Q8_1_DP4A_TOP1_ROW_GATHER_F32)
    fn.argtypes = [
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
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(xq_ptr),
        ctypes.c_void_p(qweight_ptr),
        ctypes.c_void_p(block_values_f32_ptr),
        ctypes.c_void_p(block_indices_i32_ptr),
        ctypes.c_void_p(out_indices_i32_ptr),
        ctypes.c_void_p(out_values_f32_ptr) if out_values_f32_ptr is not None else ctypes.c_void_p(),
        ctypes.c_void_p(embed_table_f32_ptr) if embed_table_f32_ptr is not None else ctypes.c_void_p(),
        ctypes.c_void_p(next_embed_f32_ptr) if next_embed_f32_ptr is not None else ctypes.c_void_p(),
        ctypes.c_int64(rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features),
        ctypes.c_int64(hidden_size),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def gguf_q6_k_pack8_top1_stage2_gather_f32(
    block_values_f32_ptr: int,
    block_indices_i32_ptr: int,
    out_indices_i32_ptr: int,
    out_values_f32_ptr: int | None,
    embed_table_f32_ptr: int | None,
    next_embed_f32_ptr: int | None,
    rows: int,
    num_blocks: int,
    hidden_size: int,
    vocab: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Diagnostic top-1 stage2 reduce/gather launch for split timing."""

    if rows <= 0:
        raise ValueError("rows must be positive")
    if num_blocks <= 0:
        raise ValueError("num_blocks must be positive")
    if hidden_size < 0:
        raise ValueError("hidden_size must be non-negative")
    if vocab <= 0:
        raise ValueError("vocab must be positive")
    has_embed = embed_table_f32_ptr is not None or next_embed_f32_ptr is not None
    if has_embed and (embed_table_f32_ptr is None or next_embed_f32_ptr is None):
        raise ValueError("embed_table_f32_ptr and next_embed_f32_ptr must be provided together")
    if has_embed and hidden_size <= 0:
        raise ValueError("hidden_size must be positive when gathering embeddings")
    library = library or build_gguf_q6_k_pack8_gemv(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYM_TOP1_STAGE2_GATHER_F32)
    fn.argtypes = [
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
        ctypes.c_void_p(block_values_f32_ptr),
        ctypes.c_void_p(block_indices_i32_ptr),
        ctypes.c_void_p(out_indices_i32_ptr),
        ctypes.c_void_p(out_values_f32_ptr) if out_values_f32_ptr is not None else ctypes.c_void_p(),
        ctypes.c_void_p(embed_table_f32_ptr) if embed_table_f32_ptr is not None else ctypes.c_void_p(),
        ctypes.c_void_p(next_embed_f32_ptr) if next_embed_f32_ptr is not None else ctypes.c_void_p(),
        ctypes.c_int64(rows),
        ctypes.c_int64(num_blocks),
        ctypes.c_int64(hidden_size),
        ctypes.c_int64(vocab),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def _check_common(rows: int, in_features: int, out_features: int) -> None:
    if rows <= 0:
        raise ValueError("rows must be positive")
    if in_features <= 0:
        raise ValueError("in_features must be positive")
    if out_features <= 0:
        raise ValueError("out_features must be positive")
    if in_features % _Q6_K_BLOCK != 0:
        raise ValueError("in_features must be divisible by GGUF Q6_K block size 256")
    if out_features % 8 != 0:
        raise ValueError("out_features must be a multiple of 8 (pack8 lane)")


def register_gguf_q6_k_pack8_gemv_kernels(*, replace: bool = True) -> None:
    """Register P9.B4b dense raw-Q6_K pack8 GEMV decode kernels."""

    register(
        KernelKey("hip_gfx1100", "linear", "gguf_q6_k", "pack8_gemv_decode_bf16_bf16_out"),
        gguf_q6_k_pack8_gemv_decode_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "linear", "gguf_q6_k", "pack8_gemv_decode_fp16_fp16_out"),
        gguf_q6_k_pack8_gemv_decode_fp16_fp16_out,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "linear", "gguf_q6_k", "pack8_gemv_decode_bf16_f32_out"),
        gguf_q6_k_pack8_gemv_decode_bf16_f32_out,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "linear", "gguf_q6_k", "pack8_gemv_decode_fp16_f32_out"),
        gguf_q6_k_pack8_gemv_decode_fp16_f32_out,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "linear", "gguf_q6_k", "pack8_gemv_decode_bf16_top1_gather_f32"),
        gguf_q6_k_pack8_gemv_decode_bf16_top1_gather_f32,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "linear", "gguf_q6_k", "pack8_gemv_decode_q8_1_dp4a_top1_gather_f32"),
        gguf_q6_k_pack8_gemv_decode_q8_1_dp4a_top1_gather_f32,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "linear",
            "gguf_q6_k",
            "pack8_gemv_decode_q8_1_dp4a_top1_scalehoist_gather_f32",
        ),
        gguf_q6_k_pack8_gemv_decode_q8_1_dp4a_top1_scalehoist_gather_f32,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "linear", "gguf_q6_k", "pack8_gemv_decode_bf16_top1_stage1_f32"),
        gguf_q6_k_pack8_gemv_decode_bf16_top1_stage1_f32,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "linear", "gguf_q6_k", "pack8_gemv_decode_q8_1_dp4a_top1_stage1_f32"),
        gguf_q6_k_pack8_gemv_decode_q8_1_dp4a_top1_stage1_f32,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "linear",
            "gguf_q6_k",
            "pack8_gemv_decode_q8_1_dp4a_top1_scalehoist_stage1_f32",
        ),
        gguf_q6_k_pack8_gemv_decode_q8_1_dp4a_top1_scalehoist_stage1_f32,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "linear", "gguf_q6_k", "pack8_gemv_decode_q8_1_dp4a_top1_row_stage1_f32"),
        gguf_q6_k_pack8_gemv_decode_q8_1_dp4a_top1_row_stage1_f32,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "linear", "gguf_q6_k", "pack8_gemv_decode_q8_1_dp4a_top1_row_gather_f32"),
        gguf_q6_k_pack8_gemv_decode_q8_1_dp4a_top1_row_gather_f32,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "linear", "gguf_q6_k", "pack8_top1_stage2_gather_f32"),
        gguf_q6_k_pack8_top1_stage2_gather_f32,
        replace=replace,
    )


register_gguf_q6_k_pack8_gemv_kernels()


__all__ = [
    "build_gguf_q6_k_pack8_gemv",
    "gguf_q6_k_pack8_gemv_decode_bf16_bf16_out",
    "gguf_q6_k_pack8_gemv_decode_bf16_f32_out",
    "gguf_q6_k_pack8_gemv_decode_bf16_top1_gather_f32",
    "gguf_q6_k_pack8_gemv_decode_bf16_top1_stage1_f32",
    "gguf_q6_k_pack8_gemv_decode_q8_1_dp4a_top1_gather_f32",
    "gguf_q6_k_pack8_gemv_decode_q8_1_dp4a_top1_scalehoist_gather_f32",
    "gguf_q6_k_pack8_gemv_decode_q8_1_dp4a_top1_scalehoist_stage1_f32",
    "gguf_q6_k_pack8_gemv_decode_q8_1_dp4a_top1_row_gather_f32",
    "gguf_q6_k_pack8_gemv_decode_q8_1_dp4a_top1_row_stage1_f32",
    "gguf_q6_k_pack8_gemv_decode_q8_1_dp4a_top1_stage1_f32",
    "gguf_q6_k_pack8_gemv_decode_fp16_f32_out",
    "gguf_q6_k_pack8_gemv_decode_fp16_fp16_out",
    "gguf_q6_k_pack8_top1_stage2_gather_f32",
    "plan_gguf_q6_k_pack8_gemv_build",
    "register_gguf_q6_k_pack8_gemv_kernels",
]
