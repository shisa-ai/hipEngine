"""YuE2 VAE decoder kernel build + launch wrappers (vae.hip).

Raw device pointers only; the host runtime converts. Everything here is FP32:
the released Oobleck decoder is FP32 end to end and the reference's arithmetic
is not reassociable into another class, so the kernels keep FP32 accumulation
and the reference's own loop order.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import build_hip, plan_hip_build
from hipengine.core.ctypes_cache import signed_kernel_fn
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime

_SOURCE = Path(__file__).with_name("vae.hip")
_OUTPUT_NAME = "yue2_vae"

_P = ctypes.c_void_p
_I = ctypes.c_int64
_S = ctypes.c_void_p

_ARGTYPES_CONV1D = (_P, _P, _P, _P, _I, _I, _I, _I, _I, _I, _I, _I, _S)
_ARGTYPES_CONV_TRANSPOSE = (_P, _P, _P, _P, _I, _I, _I, _I, _I, _I, _I, _S)
_ARGTYPES_SNAKE = (_P, _P, _P, _P, _I, _I, _S)
_ARGTYPES_ADD = (_P, _P, _P, _I, _S)


def plan_yue2_vae_build(**kwargs):
    return plan_hip_build(
        sources=[_SOURCE],
        family="yue2_vae",
        output_name=_OUTPUT_NAME,
        **kwargs,
    )


def build_yue2_vae(
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
        family="yue2_vae",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def _library() -> ctypes.CDLL:
    library = build_yue2_vae()
    if library is None:
        raise RuntimeError("yue2_vae build returned no library")
    return library


def vae_conv1d_f32(
    x_ptr: int,
    weight_ptr: int,
    bias_ptr: int,
    out_ptr: int,
    channels_in: int,
    channels_out: int,
    length: int,
    out_length: int,
    kernel: int,
    stride: int = 1,
    dilation: int = 1,
    padding: int = 0,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """FP32 Conv1d over ``[C_in, T]`` with ``[C_out, C_in, K]`` weights."""

    library = library or _library()
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(
        library, "hipengine_yue2_vae_conv1d_f32", _ARGTYPES_CONV1D, ctypes.c_int
    )
    err = fn(
        x_ptr, weight_ptr, bias_ptr, out_ptr, channels_in, channels_out, length,
        out_length, kernel, stride, dilation, padding, stream,
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def vae_conv_transpose1d_f32(
    x_ptr: int,
    weight_ptr: int,
    bias_ptr: int,
    out_ptr: int,
    channels_in: int,
    channels_out: int,
    length: int,
    out_length: int,
    kernel: int,
    stride: int,
    padding: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """FP32 ConvTranspose1d over ``[C_in, T]`` with ``[C_in, C_out, K]`` weights."""

    library = library or _library()
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(
        library, "hipengine_yue2_vae_conv_transpose1d_f32", _ARGTYPES_CONV_TRANSPOSE, ctypes.c_int
    )
    err = fn(
        x_ptr, weight_ptr, bias_ptr, out_ptr, channels_in, channels_out, length,
        out_length, kernel, stride, padding, stream,
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def vae_snake_beta_f32(
    x_ptr: int,
    alpha_ptr: int,
    beta_ptr: int,
    out_ptr: int,
    channels: int,
    length: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """FP32 SnakeBeta with log-scale alpha/beta and the released 1e-9 epsilon."""

    library = library or _library()
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(
        library, "hipengine_yue2_vae_snake_beta_f32", _ARGTYPES_SNAKE, ctypes.c_int
    )
    err = fn(x_ptr, alpha_ptr, beta_ptr, out_ptr, channels, length, stream)
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def vae_add_f32(
    x_ptr: int,
    y_ptr: int,
    out_ptr: int,
    total: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """``out = x + y`` elementwise, for residual-unit additions."""

    library = library or _library()
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(library, "hipengine_yue2_vae_add_f32", _ARGTYPES_ADD, ctypes.c_int)
    err = fn(x_ptr, y_ptr, out_ptr, total, stream)
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))
