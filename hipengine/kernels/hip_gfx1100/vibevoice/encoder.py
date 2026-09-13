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
_ARGTYPES_ELEMENTWISE_2PTR = (_P, _P, _I, _S)
_ARGTYPES_SCALE_RESIDUAL = (_P, _P, _P, _P, _I, _I, _S)
_ARGTYPES_DEPTHWISE = (_P, _P, _P, _P, _P, _P, _P, _I, _I, _I, _I, _S)
_ARGTYPES_CONV_GEMM = (_P, _P, _P, _P, _P, _I, _I, _I, _I, _I, _I, _S)
_ARGTYPES_ADD_NOISE = (_P, _P, _P, _P, _I, _I, _S)


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


def f32_to_bf16_bits(host: np.ndarray) -> np.ndarray:
    """FP32 host array -> BF16 bits (uint16), round-to-nearest-even."""
    array = np.ascontiguousarray(host, dtype=np.float32)
    bits = array.view(np.uint32)
    rounded = (bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))) & np.uint32(0xFFFF0000)
    return (rounded >> np.uint32(16)).astype(np.uint16)


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


def conv_rows_out(prefix_rows: int, length: int, k_len: int, stride: int) -> int:
    """Valid causal outputs for a pass over ``length`` rows plus prefix."""
    padded = prefix_rows + length
    if padded < k_len:
        return 0
    return (padded - k_len) // stride + 1


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
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    if rows_out <= 0:
        raise ValueError("rows_out must be positive")
    library = library or _library()
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(library, "hipengine_vv_conv_gemm_bf16", _ARGTYPES_CONV_GEMM, ctypes.c_int)
    err = fn(prefix_ptr, x_ptr, w_t_ptr, b_ptr, out_ptr, prefix_rows, rows_out, c_in, c_out, k_len, stride, stream)
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


def transpose_conv_weight_t(w: np.ndarray) -> np.ndarray:
    """nn.Conv1d weight [C_out, C_in, K] -> kernel layout [K, C_in, C_out] bf16 bits."""
    array = np.asarray(w)
    if array.ndim != 3:
        raise ValueError("conv weight must be [C_out, C_in, K]")
    return f32_to_bf16_bits(np.ascontiguousarray(array.transpose(2, 1, 0)))
