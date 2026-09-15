"""VibeVoice-ASR audio front-end kernel build + launch wrappers (encoder.hip).

All launch helpers take raw device pointers and the four-axis registry
family ``vibevoice``. Storage is bf16; accumulation is fp32. Causal conv
kernels take an explicit prefix buffer (previous chunk tail rows) so
chunk-carry equals a single-pass forward exactly.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
from hipengine.loading.vibevoice_layout import f32_to_bf16_bits, conv_rows_out, transpose_conv_weight_t

from hipengine.core.build import BuildArtifact, build_hip, plan_hip_build
from hipengine.core.ctypes_cache import signed_kernel_fn
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime

_SOURCE = Path(__file__).with_name("encoder.hip")
_OUTPUT_NAME = "vibevoice_encoder"

_P = ctypes.c_void_p
_F = ctypes.c_float
_I = ctypes.c_int64
_S = ctypes.c_void_p

_ARGTYPES_RMSNORM = (_P, _P, _P, _I, _I, _F, _S)
_ARGTYPES_ADD_BIAS = (_P, _P, _P, _I, _I, _S)
_ARGTYPES_ADD_BIAS_F32 = (_P, _P, _P, _I, _I, _S)
_ARGTYPES_ELEMENTWISE_2PTR = (_P, _P, _I, _S)
_ARGTYPES_SCALE_RESIDUAL = (_P, _P, _P, _P, _I, _I, _S)
_ARGTYPES_DEPTHWISE = (_P, _P, _P, _P, _P, _P, _P, _I, _I, _I, _I, _S)
_ARGTYPES_CONV_GEMM = (_P, _P, _P, _P, _P, _I, _I, _I, _I, _I, _I, _I, _S)
_ARGTYPES_ADD_NOISE = (_P, _P, _P, _P, _I, _I, _S)
_ARGTYPES_ROPE_POS = (_P, _P, _P, _P, _P, _P, _P, _I, _I, _I, _I, _S)
_ARGTYPES_PREFILL_ATTN = (_P, _P, _P, _P, _P, _I, _I, _I, _I, _I, _F, _S)
_ARGTYPES_IM2COL = (_P, _P, _I, _I, _I, _I, _I, _I, _S)


def plan_vibevoice_encoder_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: str = "decode",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="vibevoice_encoder",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
    )


def build_vibevoice_encoder(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: str = "decode",
    dry_run: bool = False,
    load: bool = True,
    require_cached: bool = False,
) -> ctypes.CDLL | None:
    return build_hip(
        sources=[_SOURCE],
        family="vibevoice_encoder",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def _library() -> ctypes.CDLL:
    library = build_vibevoice_encoder()
    if library is None:
        raise RuntimeError("vibevoice_encoder build returned no library")
    return library




def vv_add_bias_f32(
    x_ptr: int,
    b_ptr: int,
    out_ptr: int,
    total: int,
    width: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """``out[i] = x[i] + b[i % width]`` in fp32 (GEMV bias add)."""
    library = library or _library()
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(library, "hipengine_vv_add_bias_f32", _ARGTYPES_ADD_BIAS_F32, ctypes.c_int)
    err = fn(x_ptr, b_ptr, out_ptr, total, width, stream)
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def vv_add_bias_bf16(
    x_ptr: int,
    b_ptr: int,
    out_ptr: int,
    total: int,
    width: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    library = library or _library()
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(library, "hipengine_vv_add_bias_bf16", _ARGTYPES_ADD_BIAS, ctypes.c_int)
    err = fn(x_ptr, b_ptr, out_ptr, total, width, stream)
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def vv_rmsnorm_bf16(
    x_ptr: int,
    w_ptr: int,
    out_ptr: int,
    rows: int,
    hidden: int,
    eps: float,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    library = library or _library()
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(library, "hipengine_vv_rmsnorm_bf16", _ARGTYPES_RMSNORM, ctypes.c_int)
    err = fn(x_ptr, w_ptr, out_ptr, rows, hidden, eps, stream)
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def vv_gelu_bf16(
    x_ptr: int,
    out_ptr: int,
    count: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    library = library or _library()
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(library, "hipengine_vv_gelu_bf16", _ARGTYPES_ELEMENTWISE_2PTR, ctypes.c_int)
    err = fn(x_ptr, out_ptr, count, stream)
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def vv_scale_residual_bf16(
    x_ptr: int,
    y_ptr: int,
    gamma_ptr: int,
    out_ptr: int,
    total: int,
    width: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    library = library or _library()
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(library, "hipengine_vv_scale_residual_bf16", _ARGTYPES_SCALE_RESIDUAL, ctypes.c_int)
    err = fn(x_ptr, y_ptr, gamma_ptr, out_ptr, total, width, stream)
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def vv_depthwise_conv_bf16(
    prefix_ptr: int,
    normed_ptr: int,
    resid_ptr: int,
    w_ptr: int,
    b_ptr: int,
    gamma_ptr: int,
    out_ptr: int,
    prefix_rows: int,
    rows: int,
    channels: int,
    k_len: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    if prefix_rows != k_len - 1:
        raise ValueError("prefix_rows must equal k_len - 1 for the stride-1 depthwise conv")
    library = library or _library()
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(library, "hipengine_vv_depthwise_conv_bf16", _ARGTYPES_DEPTHWISE, ctypes.c_int)
    err = fn(prefix_ptr, normed_ptr, resid_ptr, w_ptr, b_ptr, gamma_ptr, out_ptr, prefix_rows, rows, channels, k_len, stream)
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))




def vv_conv_gemm_bf16(
    prefix_ptr: int,
    x_ptr: int,
    w_t_ptr: int,
    b_ptr: int,
    out_ptr: int,
    prefix_rows: int,
    rows_out: int,
    c_in: int,
    c_out: int,
    k_len: int,
    stride: int,
    rows_in: int = -1,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Strided causal conv. ``rows_in`` bounds the taps read from ``x``.

    ``rows_in < 0`` (the default) leaves the bound off, so the caller must
    guarantee every tap is in range. Pass the true input row count to get the
    fork's right zero padding: taps past it contribute zero, which is what
    makes a non-streaming pass emit ``ceil(rows/stride)`` outputs.
    """
    if rows_out <= 0:
        raise ValueError("rows_out must be positive")
    library = library or _library()
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(library, "hipengine_vv_conv_gemm_bf16", _ARGTYPES_CONV_GEMM, ctypes.c_int)
    err = fn(prefix_ptr, x_ptr, w_t_ptr, b_ptr, out_ptr, prefix_rows, rows_out, c_in, c_out, k_len, stride, rows_in, stream)
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def vv_rope_positions_f32(
    q_ptr: int,
    k_ptr: int,
    cos_table_ptr: int,
    sin_table_ptr: int,
    positions_ptr: int,
    q_out_ptr: int,
    k_out_ptr: int,
    rows: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Batched rotate-half rope over ``rows`` tokens with explicit positions."""
    library = library or _library()
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(library, "hipengine_vv_rope_positions_f32", _ARGTYPES_ROPE_POS, ctypes.c_int)
    err = fn(q_ptr, k_ptr, cos_table_ptr, sin_table_ptr, positions_ptr, q_out_ptr, k_out_ptr,
             rows, num_q_heads, num_kv_heads, head_dim, stream)
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def vv_im2col_bf16(
    x_ptr: int,
    out_ptr: int,
    rows_out: int,
    c_in: int,
    k_len: int,
    stride: int,
    prefix_rows: int,
    rows_in: int = -1,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """im2col for strided causal convs: (rows_out, C_in*K) bf16.

    ``rows_in < 0`` leaves the input bound off; otherwise taps at or past
    ``rows_in`` are written as zero (the fork's right zero padding).
    """
    library = library or _library()
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(library, "hipengine_vv_im2col_bf16", _ARGTYPES_IM2COL, ctypes.c_int)
    err = fn(x_ptr, out_ptr, rows_out, c_in, k_len, stride, prefix_rows, rows_in, stream)
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def vv_prefill_attention_f32(
    q_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    positions_ptr: int,
    out_ptr: int,
    rows: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    max_ctx: int,
    scale: float,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Causal prefill attention over a dense contiguous bf16 KV cache."""
    if max_ctx > 16000:
        raise ValueError("max_ctx exceeds the shared-memory score window")
    library = library or _library()
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(library, "hipengine_vv_prefill_attention_f32", _ARGTYPES_PREFILL_ATTN, ctypes.c_int)
    err = fn(q_ptr, key_cache_ptr, value_cache_ptr, positions_ptr, out_ptr,
             rows, num_q_heads, num_kv_heads, head_dim, max_ctx, scale, stream)
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def vv_add_scaled_noise_bf16(
    lat_ptr: int,
    scale_ptr: int,
    noise_ptr: int,
    out_ptr: int,
    total: int,
    frames_x_hidden: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    library = library or _library()
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(library, "hipengine_vv_add_scaled_noise_bf16", _ARGTYPES_ADD_NOISE, ctypes.c_int)
    err = fn(lat_ptr, scale_ptr, noise_ptr, out_ptr, total, frames_x_hidden, stream)
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))




def _span_arguments(spans, rows):
    from hipengine.core.dtype import DType
    if (spans.spans_mode != 'uniform' or spans.storage_dtype != DType.BF16
            or spans.live_counts.dtype != DType.INT64 or spans.live_counts.numel != rows
            or spans.token_positions is None or spans.evict_mask is None or spans.row_positions is None
            or spans.token_positions.dtype != DType.INT64 or spans.row_positions.dtype != DType.INT64
            or spans.row_positions.numel != rows or spans.max_live_count != spans.base_offsets.numel
            or spans.token_positions.numel != spans.max_live_count or spans.evict_mask.numel != spans.max_live_count
            or not 0 < spans.max_live_count <= 16000):
        raise ValueError('VibeVoice requires uniform BF16 block-size-one spans with complete position/mask metadata')
    return (spans.base_offsets.ptr,spans.live_counts.ptr,spans.token_positions.ptr,
            spans.evict_mask.ptr,spans.row_positions.ptr)


def vv_kv_write_spans(key_ptr,value_ptr,key_cache_ptr,value_cache_ptr,spans,rows,kv_heads,head_dim,
                      *,stream=0,library=None,runtime=None):
    metadata = _span_arguments(spans,rows)
    library = library or _library()
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(library,'hipengine_vv_kv_write_spans',(_P,)*9+(_I,)*3+(_S,),ctypes.c_int)
    err = fn(key_ptr,value_ptr,key_cache_ptr,value_cache_ptr,*metadata,rows,kv_heads*head_dim,spans.max_live_count,stream)
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def vv_attention_spans(query_ptr,key_cache_ptr,value_cache_ptr,out_ptr,spans,rows,q_heads,kv_heads,head_dim,scale,
                       *,stream=0,library=None,runtime=None):
    metadata = _span_arguments(spans,rows)
    library = library or _library()
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(library,'hipengine_vv_attention_spans',(_P,)*9+(_I,)*5+(_F,_S),ctypes.c_int)
    err = fn(query_ptr,key_cache_ptr,value_cache_ptr,out_ptr,*metadata,rows,q_heads,kv_heads,head_dim,spans.max_live_count,scale,stream)
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def vv_depthwise_accumulate_f32(prefix, normed, w, b, accumulator, prefix_rows, rows, channels, k_len, *, stream=0, library=None, runtime=None):
    library = library or _library()
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(library, "hipengine_vv_depthwise_accumulate_f32",
        [ctypes.c_void_p]*5 + [ctypes.c_int64]*4 + [ctypes.c_void_p], ctypes.c_int)
    runtime.check(int(fn(prefix,normed,w,b,accumulator,prefix_rows,rows,channels,k_len,stream)))


def vv_depthwise_residual_bf16(accumulator, resid, gamma, out, rows, channels, *, stream=0, library=None, runtime=None):
    library = library or _library()
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(library, "hipengine_vv_depthwise_residual_bf16",
        [ctypes.c_void_p]*4 + [ctypes.c_int64]*2 + [ctypes.c_void_p], ctypes.c_int)
    runtime.check(int(fn(accumulator,resid,gamma,out,rows,channels,stream)))


def vv_depthwise_unfused_bf16(prefix, normed, resid, w, b, gamma, out, prefix_rows, rows, channels, k_len, *, stream=0, library=None, runtime=None):
    from hipengine.core.memory import malloc,free
    if rows <= 0 or channels <= 0 or k_len <= 0 or prefix_rows != k_len - 1:
        raise ValueError("invalid depthwise dimensions")
    accumulator = malloc(rows*channels*4)
    try:
        vv_depthwise_accumulate_f32(prefix,normed,w,b,accumulator.ptr,prefix_rows,rows,channels,k_len,
            stream=stream,library=library,runtime=runtime)
        vv_depthwise_residual_bf16(accumulator.ptr,resid,gamma,out,rows,channels,
            stream=stream,library=library,runtime=runtime)
    finally:
        free(accumulator)
