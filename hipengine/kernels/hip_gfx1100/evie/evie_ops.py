"""Raw-pointer wrappers for EVIE fp32 elementwise/norm/rope kernels."""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.ctypes_cache import signed_kernel_fn
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("evie_ops.hip")
_OUTPUT_NAME = "evie_ops.so"


def plan_evie_ops_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="evie_ops",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
    )


def build_evie_ops(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
    dry_run: bool = False,
    load: bool = True,
) -> ctypes.CDLL | None:
    return build_hip(
        sources=[_SOURCE],
        family="evie_ops",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
    )


def _check_launch(runtime: HipRuntime, err: int) -> None:
    if err != HIP_SUCCESS:
        raise RuntimeError(f"evie_ops kernel launch failed with {err}")


def _fn(library: ctypes.CDLL, symbol: str, argtypes: list) -> "ctypes._FuncPtr":
    fn = getattr(library, symbol)
    fn.argtypes = argtypes
    fn.restype = ctypes.c_int
    return fn


_P = ctypes.c_void_p
_F = ctypes.c_float
_I = ctypes.c_int64
_S = ctypes.c_void_p


def _evie_rmsnorm_f32(
    library: ctypes.CDLL,
    x_ptr: int,
    w_ptr: int,
    out_ptr: int,
    rows: int,
    hidden: int,
    eps: float,
    *,
    stream: int,
    runtime: HipRuntime | None,
) -> None:
    fn = _fn(library, "hipengine_evie_rmsnorm_f32", [_P, _P, _P, _I, _I, _F, _S])
    err = fn(_P(x_ptr), _P(w_ptr), _P(out_ptr), _I(rows), _I(hidden), _F(eps), _S(stream))
    _check_launch(runtime or get_hip_runtime(), err)


def _evie_add_f32(
    library: ctypes.CDLL,
    x_ptr: int,
    y_ptr: int,
    out_ptr: int,
    n: int,
    *,
    stream: int,
    runtime: HipRuntime | None,
) -> None:
    fn = _fn(library, "hipengine_evie_add_f32", [_P, _P, _P, _I, _S])
    err = fn(_P(x_ptr), _P(y_ptr), _P(out_ptr), _I(n), _S(stream))
    _check_launch(runtime or get_hip_runtime(), err)


def _evie_layernorm_f32(
    library: ctypes.CDLL,
    x_ptr: int,
    w_ptr: int,
    b_ptr: int,
    out_ptr: int,
    rows: int,
    hidden: int,
    eps: float,
    *,
    stream: int,
    runtime: HipRuntime | None,
) -> None:
    fn = _fn(
        library, "hipengine_evie_layernorm_f32", [_P, _P, _P, _P, _I, _I, _F, _S]
    )
    err = fn(
        _P(x_ptr), _P(w_ptr), _P(b_ptr), _P(out_ptr), _I(rows), _I(hidden),
        _F(eps), _S(stream),
    )
    _check_launch(runtime or get_hip_runtime(), err)


def _evie_silu_mul_f32(
    library: ctypes.CDLL,
    gate_ptr: int,
    up_ptr: int,
    n: int,
    *,
    stream: int,
    runtime: HipRuntime | None,
) -> None:
    fn = _fn(library, "hipengine_evie_silu_mul_f32", [_P, _P, _I, _S])
    err = fn(_P(gate_ptr), _P(up_ptr), _I(n), _S(stream))
    _check_launch(runtime or get_hip_runtime(), err)


def _evie_gelu_tanh_f32(
    library: ctypes.CDLL,
    x_ptr: int,
    out_ptr: int,
    n: int,
    *,
    stream: int,
    runtime: HipRuntime | None,
) -> None:
    fn = _fn(library, "hipengine_evie_gelu_tanh_f32", [_P, _P, _I, _S])
    err = fn(_P(x_ptr), _P(out_ptr), _I(n), _S(stream))
    _check_launch(runtime or get_hip_runtime(), err)


def _evie_gelu_erf_f32(
    library: ctypes.CDLL,
    x_ptr: int,
    out_ptr: int,
    n: int,
    *,
    stream: int,
    runtime: HipRuntime | None,
) -> None:
    fn = _fn(library, "hipengine_evie_gelu_erf_f32", [_P, _P, _I, _S])
    err = fn(_P(x_ptr), _P(out_ptr), _I(n), _S(stream))
    _check_launch(runtime or get_hip_runtime(), err)


def _evie_sigmoid_mul_f32(
    library: ctypes.CDLL,
    gate_ptr: int,
    out_ptr: int,
    n: int,
    *,
    stream: int,
    runtime: HipRuntime | None,
) -> None:
    fn = _fn(library, "hipengine_evie_sigmoid_mul_f32", [_P, _P, _I, _S])
    err = fn(_P(gate_ptr), _P(out_ptr), _I(n), _S(stream))
    _check_launch(runtime or get_hip_runtime(), err)


def _evie_l2norm_rows_f32(
    library: ctypes.CDLL,
    x_ptr: int,
    out_ptr: int,
    rows: int,
    dim: int,
    *,
    stream: int,
    runtime: HipRuntime | None,
) -> None:
    fn = _fn(library, "hipengine_evie_l2norm_rows_f32", [_P, _P, _I, _I, _S])
    err = fn(_P(x_ptr), _P(out_ptr), _I(rows), _I(dim), _S(stream))
    _check_launch(runtime or get_hip_runtime(), err)


def _evie_gdn_gates_f32(
    library: ctypes.CDLL,
    b_ptr: int,
    a_ptr: int,
    a_log_ptr: int,
    dt_bias_ptr: int,
    beta_ptr: int,
    decay_ptr: int,
    tokens: int,
    heads: int,
    *,
    stream: int,
    runtime: HipRuntime | None,
) -> None:
    fn = _fn(
        library,
        "hipengine_evie_gdn_gates_f32",
        [_P, _P, _P, _P, _P, _P, _I, _I, _S],
    )
    err = fn(
        _P(b_ptr), _P(a_ptr), _P(a_log_ptr), _P(dt_bias_ptr),
        _P(beta_ptr), _P(decay_ptr), _I(tokens), _I(heads), _S(stream),
    )
    _check_launch(runtime or get_hip_runtime(), err)


def _evie_gdn_l2norm_scale_f32(
    library: ctypes.CDLL,
    q_ptr: int,
    k_ptr: int,
    q_out_ptr: int,
    k_out_ptr: int,
    q_scale: float,
    rows: int,
    head_dim: int,
    *,
    stream: int,
    runtime: HipRuntime | None,
) -> None:
    fn = _fn(
        library,
        "hipengine_evie_gdn_l2norm_scale_f32",
        [_P, _P, _P, _P, _F, _I, _I, _S],
    )
    err = fn(
        _P(q_ptr), _P(k_ptr), _P(q_out_ptr), _P(k_out_ptr), _F(q_scale),
        _I(rows), _I(head_dim), _S(stream),
    )
    _check_launch(runtime or get_hip_runtime(), err)


def _evie_rope_f32(
    library: ctypes.CDLL,
    x_ptr: int,
    cos_ptr: int,
    sin_ptr: int,
    tokens: int,
    heads: int,
    head_dim: int,
    rotary_dim: int,
    *,
    stream: int,
    runtime: HipRuntime | None,
) -> None:
    fn = _fn(
        library, "hipengine_evie_rope_f32", [_P, _P, _P, _I, _I, _I, _I, _S]
    )
    err = fn(
        _P(x_ptr), _P(cos_ptr), _P(sin_ptr), _I(tokens), _I(heads),
        _I(head_dim), _I(rotary_dim), _S(stream),
    )
    _check_launch(runtime or get_hip_runtime(), err)


def _evie_softmax_rows_f32(
    library: ctypes.CDLL,
    scores_ptr: int,
    rows: int,
    cols: int,
    *,
    stream: int,
    runtime: HipRuntime | None,
) -> None:
    fn = _fn(library, "hipengine_evie_softmax_rows_f32", [_P, _I, _I, _S])
    err = fn(_P(scores_ptr), _I(rows), _I(cols), _S(stream))
    _check_launch(runtime or get_hip_runtime(), err)


def _evie_embed_lookup_f32(
    library: ctypes.CDLL,
    ids_ptr: int,
    table_ptr: int,
    visual_ptr: int,
    out_ptr: int,
    tokens: int,
    hidden: int,
    image_token_id: int,
    *,
    stream: int,
    runtime: HipRuntime | None,
) -> None:
    fn = _fn(
        library,
        "hipengine_evie_embed_lookup_f32",
        [_P, _P, _P, _P, _I, _I, _I, _S],
    )
    err = fn(
        _P(ids_ptr), _P(table_ptr), _P(visual_ptr), _P(out_ptr),
        _I(tokens), _I(hidden), _I(image_token_id), _S(stream),
    )
    _check_launch(runtime or get_hip_runtime(), err)


def _evie_scaled_add_f32(
    library: ctypes.CDLL,
    x_ptr: int,
    out_ptr: int,
    scale: float,
    n: int,
    *,
    stream: int,
    runtime: HipRuntime | None,
) -> None:
    fn = _fn(library, "hipengine_evie_scaled_add_f32", [_P, _P, _F, _I, _S])
    err = fn(_P(x_ptr), _P(out_ptr), _F(scale), _I(n), _S(stream))
    _check_launch(runtime or get_hip_runtime(), err)


def register_evie_ops_kernels(*, replace: bool = True) -> None:
    library = build_evie_ops(load=True)
    register(
        KernelKey("hip_gfx1100", "rmsnorm", "fp32", "evie_delta"),
        lambda x_ptr, w_ptr, out_ptr, rows, hidden, eps=1e-6, **kw: _evie_rmsnorm_f32(
            library, x_ptr, w_ptr, out_ptr, rows, hidden, eps, **kw
        ),
        replace=replace,
    )


__all__ = [
    "build_evie_ops",
    "plan_evie_ops_build",
    "register_evie_ops_kernels",
]
