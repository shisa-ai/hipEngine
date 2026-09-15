"""Raw-pointer wrappers for VibeVoice codec decoder fp32 kernels."""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("codec_ops.hip")
_OUTPUT_NAME = "vibevoice_codec_ops.so"


def plan_codec_ops_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="vibevoice_codec_ops",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
    )


def build_codec_ops(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
    dry_run: bool = False,
    load: bool = True,
    require_cached: bool = False,
) -> ctypes.CDLL | None:
    return build_hip(
        sources=[_SOURCE],
        family="vibevoice_codec_ops",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def _check_launch(runtime: HipRuntime, err: int) -> None:
    if err != HIP_SUCCESS:
        raise RuntimeError(f"vibevoice codec kernel launch failed with {err}")


def _fn(library: ctypes.CDLL, symbol: str, argtypes: list) -> "ctypes._FuncPtr":
    fn = getattr(library, symbol)
    fn.argtypes = argtypes
    fn.restype = ctypes.c_int
    return fn


_P = ctypes.c_void_p
_F = ctypes.c_float
_I = ctypes.c_int64
_S = ctypes.c_void_p


def _conv1d_valid_dense(
    library: ctypes.CDLL,
    x_ptr: int,
    w_ptr: int,
    b_ptr: int,
    out_ptr: int,
    in_c: int,
    out_c: int,
    k: int,
    l_out: int,
    *,
    stream: int,
    runtime: HipRuntime | None,
) -> None:
    fn = _fn(
        library,
        "hipengine_vv_conv1d_valid_dense",
        [_P, _P, _P, _P, _I, _I, _I, _I, _S],
    )
    err = fn(
        _P(x_ptr), _P(w_ptr), _P(b_ptr), _P(out_ptr),
        _I(in_c), _I(out_c), _I(k), _I(l_out), _S(stream),
    )
    _check_launch(runtime or get_hip_runtime(), err)


def _conv1d_valid_depthwise(
    library: ctypes.CDLL,
    x_ptr: int,
    w_ptr: int,
    b_ptr: int,
    out_ptr: int,
    channels: int,
    k: int,
    l_out: int,
    *,
    stream: int,
    runtime: HipRuntime | None,
) -> None:
    fn = _fn(
        library,
        "hipengine_vv_conv1d_valid_depthwise",
        [_P, _P, _P, _P, _I, _I, _I, _S],
    )
    err = fn(
        _P(x_ptr), _P(w_ptr), _P(b_ptr), _P(out_ptr),
        _I(channels), _I(k), _I(l_out), _S(stream),
    )
    _check_launch(runtime or get_hip_runtime(), err)


def _conv_transpose1d_causal(
    library: ctypes.CDLL,
    x_ptr: int,
    w_ptr: int,
    b_ptr: int,
    out_ptr: int,
    in_c: int,
    out_c: int,
    k: int,
    stride: int,
    l_out: int,
    keep_from: int,
    kept_len: int,
    *,
    stream: int,
    runtime: HipRuntime | None,
) -> None:
    fn = _fn(
        library,
        "hipengine_vv_conv_transpose1d_causal",
        [_P, _P, _P, _P, _I, _I, _I, _I, _I, _I, _I, _S],
    )
    err = fn(
        _P(x_ptr), _P(w_ptr), _P(b_ptr), _P(out_ptr),
        _I(in_c), _I(out_c), _I(k), _I(stride), _I(l_out),
        _I(keep_from), _I(kept_len), _S(stream),
    )
    _check_launch(runtime or get_hip_runtime(), err)


def _rmsnorm_channels(
    library: ctypes.CDLL,
    x_ptr: int,
    w_ptr: int,
    out_ptr: int,
    channels: int,
    length: int,
    eps: float,
    *,
    stream: int,
    runtime: HipRuntime | None,
) -> None:
    fn = _fn(
        library,
        "hipengine_vv_rmsnorm_channels",
        [_P, _P, _P, _I, _I, _F, _S],
    )
    err = fn(_P(x_ptr), _P(w_ptr), _P(out_ptr), _I(channels), _I(length), _F(eps), _S(stream))
    _check_launch(runtime or get_hip_runtime(), err)


def _gelu_erf(
    library: ctypes.CDLL,
    x_ptr: int,
    out_ptr: int,
    n: int,
    *,
    stream: int,
    runtime: HipRuntime | None,
) -> None:
    fn = _fn(library, "hipengine_vv_gelu_erf", [_P, _P, _I, _S])
    err = fn(_P(x_ptr), _P(out_ptr), _I(n), _S(stream))
    _check_launch(runtime or get_hip_runtime(), err)


def _add(
    library: ctypes.CDLL,
    x_ptr: int,
    y_ptr: int,
    out_ptr: int,
    n: int,
    *,
    stream: int,
    runtime: HipRuntime | None,
) -> None:
    fn = _fn(library, "hipengine_vv_add", [_P, _P, _P, _I, _S])
    err = fn(_P(x_ptr), _P(y_ptr), _P(out_ptr), _I(n), _S(stream))
    _check_launch(runtime or get_hip_runtime(), err)


def _channel_scale(
    library: ctypes.CDLL,
    x_ptr: int,
    scale_ptr: int,
    out_ptr: int,
    channels: int,
    length: int,
    *,
    stream: int,
    runtime: HipRuntime | None,
) -> None:
    fn = _fn(library, "hipengine_vv_channel_scale", [_P, _P, _P, _I, _I, _S])
    err = fn(
        _P(x_ptr), _P(scale_ptr), _P(out_ptr), _I(channels), _I(length), _S(stream)
    )
    _check_launch(runtime or get_hip_runtime(), err)


def _linear(
    library: ctypes.CDLL,
    x_ptr: int,
    w_ptr: int,
    b_ptr: int,
    out_ptr: int,
    in_features: int,
    out_features: int,
    length: int,
    *,
    stream: int,
    runtime: HipRuntime | None,
) -> None:
    fn = _fn(
        library,
        "hipengine_vv_linear",
        [_P, _P, _P, _P, _I, _I, _I, _S],
    )
    err = fn(
        _P(x_ptr), _P(w_ptr), _P(b_ptr), _P(out_ptr),
        _I(in_features), _I(out_features), _I(length), _S(stream),
    )
    _check_launch(runtime or get_hip_runtime(), err)


def _concat_gather(
    library: ctypes.CDLL,
    cache_ptr: int,
    x_ptr: int,
    out_ptr: int,
    channels: int,
    keep: int,
    cache_len: int,
    new_len: int,
    *,
    stream: int,
    runtime: HipRuntime | None,
) -> None:
    fn = _fn(
        library,
        "hipengine_vv_concat_gather",
        [_P, _P, _P, _I, _I, _I, _I, _S],
    )
    err = fn(
        _P(cache_ptr), _P(x_ptr), _P(out_ptr), _I(channels), _I(keep),
        _I(cache_len), _I(new_len), _S(stream),
    )
    _check_launch(runtime or get_hip_runtime(), err)


_VARIANT = "vv_codec_v1"


def register_codec_ops_kernels(*, replace: bool = True) -> None:
    """Register the codec primitives under the four-axis registry."""

    library = build_codec_ops(load=True)

    def wrap(fn, *fixed_args):
        def call(*args, **kw):
            fn(library, *args, **kw)

        return call

    register(
        KernelKey("hip_gfx1100", "conv1d", "fp32", _VARIANT),
        lambda x_ptr, w_ptr, b_ptr, out_ptr, in_c, out_c, k, l_out, **kw: _conv1d_valid_dense(
            library, x_ptr, w_ptr, b_ptr, out_ptr, in_c, out_c, k, l_out, **kw
        ),
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "depthwise_conv1d", "fp32", _VARIANT),
        lambda x_ptr, w_ptr, b_ptr, out_ptr, channels, k, l_out, **kw: _conv1d_valid_depthwise(
            library, x_ptr, w_ptr, b_ptr, out_ptr, channels, k, l_out, **kw
        ),
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "conv_transpose1d", "fp32", _VARIANT),
        lambda x_ptr, w_ptr, b_ptr, out_ptr, in_c, out_c, k, stride, l_out,
        keep_from=0, kept_len=None, **kw: _conv_transpose1d_causal(
            library, x_ptr, w_ptr, b_ptr, out_ptr, in_c, out_c, k, stride,
            l_out, keep_from, l_out if kept_len is None else kept_len, **kw
        ),
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "rmsnorm_channels", "fp32", _VARIANT),
        lambda x_ptr, w_ptr, out_ptr, channels, length, eps=1e-5, **kw: _rmsnorm_channels(
            library, x_ptr, w_ptr, out_ptr, channels, length, eps, **kw
        ),
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "gelu", "fp32", _VARIANT),
        lambda x_ptr, out_ptr, n, **kw: _gelu_erf(library, x_ptr, out_ptr, n, **kw),
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "residual_add", "fp32", _VARIANT),
        lambda x_ptr, y_ptr, out_ptr, n, **kw: _add(library, x_ptr, y_ptr, out_ptr, n, **kw),
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "channel_scale", "fp32", _VARIANT),
        lambda x_ptr, scale_ptr, out_ptr, channels, length, **kw: _channel_scale(
            library, x_ptr, scale_ptr, out_ptr, channels, length, **kw
        ),
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "linear", "fp32", _VARIANT),
        lambda x_ptr, w_ptr, b_ptr, out_ptr, in_features, out_features, length, **kw: _linear(
            library, x_ptr, w_ptr, b_ptr, out_ptr, in_features, out_features, length, **kw
        ),
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "concat_gather", "fp32", _VARIANT),
        lambda cache_ptr, x_ptr, out_ptr, channels, keep, cache_len, new_len, **kw: _concat_gather(
            library, cache_ptr, x_ptr, out_ptr, channels, keep, cache_len, new_len, **kw
        ),
        replace=replace,
    )


__all__ = [
    "build_codec_ops",
    "plan_codec_ops_build",
    "register_codec_ops_kernels",
]
