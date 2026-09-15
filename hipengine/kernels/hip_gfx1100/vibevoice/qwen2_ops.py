"""Raw-pointer wrappers for the VibeVoice Qwen2 backbone f16 kernels."""

from __future__ import annotations

import ctypes
import math
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime

_SOURCE = Path(__file__).with_name("qwen2_ops.hip")
_OUTPUT_NAME = "vibevoice_qwen2_ops.so"


def plan_qwen2_ops_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="vibevoice_qwen2_ops",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
    )


def build_qwen2_ops(
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
        family="vibevoice_qwen2_ops",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


_P = ctypes.c_void_p
_F = ctypes.c_float
_I = ctypes.c_int64
_S = ctypes.c_void_p


_Fn_CACHE: dict[tuple[int, str], "ctypes._FuncPtr"] = {}


def _fn(library: ctypes.CDLL, symbol: str, argtypes: list) -> "ctypes._FuncPtr":
    key = (id(library), symbol)
    fn = _Fn_CACHE.get(key)
    if fn is None:
        fn = getattr(library, symbol)
        fn.argtypes = argtypes
        fn.restype = ctypes.c_int
        _Fn_CACHE[key] = fn
    return fn


def _launch(runtime: HipRuntime | None, library: ctypes.CDLL, symbol: str,
            argtypes: list, args: tuple) -> None:
    fn = _fn(library, symbol, argtypes)
    err = fn(*args)
    if err != HIP_SUCCESS:
        raise RuntimeError(f"vibevoice qwen2 kernel {symbol} failed with {err}")


def rmsnorm_f16(
    library: ctypes.CDLL, x_ptr: int, weight_ptr: int, out_ptr: int,
    rows: int, hidden: int, eps: float, *,
    stream: int = 0, runtime: HipRuntime | None = None) -> None:
    _launch(runtime, library, "vv_qwen2_rmsnorm_f16",
            [_P, _P, _P, _I, _I, _F, _S],
            (_P(x_ptr), _P(weight_ptr), _P(out_ptr), _I(rows), _I(hidden),
             _F(eps), _S(stream)))


def rope_f16(
    library: ctypes.CDLL, x_ptr: int, cos_ptr: int, sin_ptr: int,
    tokens: int, heads: int, head_dim: int, *,
    stream: int = 0, runtime: HipRuntime | None = None) -> None:
    _launch(runtime, library, "vv_qwen2_rope_f16",
            [_P, _P, _P, _I, _I, _I, _S],
            (_P(x_ptr), _P(cos_ptr), _P(sin_ptr),
             _I(tokens), _I(heads), _I(head_dim), _S(stream)))


def gemm_f16(
    library: ctypes.CDLL, x_ptr: int, w_ptr: int, bias_ptr: int, out_ptr: int,
    rows: int, k: int, n: int, *,
    stream: int = 0, runtime: HipRuntime | None = None) -> None:
    _launch(runtime, library, "vv_qwen2_gemm_f16",
            [_P, _P, _P, _P, _I, _I, _I, _S],
            (_P(x_ptr), _P(w_ptr), _P(bias_ptr), _P(out_ptr),
             _I(rows), _I(k), _I(n), _S(stream)))


def decode_attn_f16(
    library: ctypes.CDLL, q_ptr: int, k_cache_ptr: int, v_cache_ptr: int,
    cos_ptr: int, sin_ptr: int, out_ptr: int,
    q_heads: int, kv_heads: int, head_dim: int, capacity: int, pos: int, *,
    scale: float | None = None,
    stream: int = 0, runtime: HipRuntime | None = None) -> None:
    _launch(runtime, library, "vv_qwen2_decode_attn_f16",
            [_P, _P, _P, _P, _P, _P, _I, _I, _I, _I, _I, _F, _S],
            (_P(q_ptr), _P(k_cache_ptr), _P(v_cache_ptr),
             _P(cos_ptr), _P(sin_ptr), _P(out_ptr),
             _I(q_heads), _I(kv_heads), _I(head_dim), _I(capacity), _I(pos),
             _F(scale if scale is not None else 1.0 / math.sqrt(head_dim)),
             _S(stream)))
    # grid is (q_heads); launched via launch signature below.


def prefill_attn_f16(
    library: ctypes.CDLL, q_ptr: int, k_new_ptr: int, v_new_ptr: int,
    k_cache_ptr: int, v_cache_ptr: int, cos_ptr: int, sin_ptr: int,
    out_ptr: int, q_heads: int, kv_heads: int, head_dim: int,
    capacity: int, past: int, tokens: int, *,
    scale: float | None = None,
    stream: int = 0, runtime: HipRuntime | None = None) -> None:
    _launch(runtime, library, "vv_qwen2_prefill_attn_f16",
            [_P, _P, _P, _P, _P, _P, _P, _P, _I, _I, _I, _I, _I, _I, _F, _S],
            (_P(q_ptr), _P(k_new_ptr), _P(v_new_ptr),
             _P(k_cache_ptr), _P(v_cache_ptr), _P(cos_ptr), _P(sin_ptr),
             _P(out_ptr),
             _I(q_heads), _I(kv_heads), _I(head_dim), _I(capacity),
             _I(past), _I(tokens),
             _F(scale if scale is not None else 1.0 / math.sqrt(head_dim)),
             _S(stream)))


def add_f16(
    library: ctypes.CDLL, a_ptr: int, b_ptr: int, out_ptr: int, total: int, *,
    stream: int = 0, runtime: HipRuntime | None = None) -> None:
    _launch(runtime, library, "vv_qwen2_add_f16",
            [_P, _P, _P, _I, _S],
            (_P(a_ptr), _P(b_ptr), _P(out_ptr), _I(total), _S(stream)))


def silu_mul_f16(
    library: ctypes.CDLL, gate_ptr: int, up_ptr: int, total: int, *,
    stream: int = 0, runtime: HipRuntime | None = None) -> None:
    _launch(runtime, library, "vv_qwen2_silu_mul_f16",
            [_P, _P, _I, _S],
            (_P(gate_ptr), _P(up_ptr), _I(total), _S(stream)))


def bias_add_f16(
    library: ctypes.CDLL, out_ptr: int, bias_ptr: int, n: int, total: int, *,
    stream: int = 0, runtime: HipRuntime | None = None) -> None:
    _launch(runtime, library, "vv_qwen2_bias_add_f16",
            [_P, _P, _I, _I, _S],
            (_P(out_ptr), _P(bias_ptr), _I(n), _I(total), _S(stream)))


def append_kv_f16(
    library: ctypes.CDLL, k_ptr: int, v_ptr: int, k_cache_ptr: int,
    v_cache_ptr: int, kv_heads: int, head_dim: int, start: int, tokens: int, *,
    stream: int = 0, runtime: HipRuntime | None = None) -> None:
    _launch(runtime, library, "vv_qwen2_append_kv_f16",
            [_P, _P, _P, _P, _I, _I, _I, _I, _S],
            (_P(k_ptr), _P(v_ptr), _P(k_cache_ptr), _P(v_cache_ptr),
             _I(kv_heads), _I(head_dim), _I(start), _I(tokens), _S(stream)))
