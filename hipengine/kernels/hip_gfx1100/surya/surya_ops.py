"""Surya fp32 helper-kernel build + launch wrappers (see surya_ops.hip)."""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import build_hip, plan_hip_build
from hipengine.core.build import BuildArtifact
from hipengine.core.hip import HipRuntime, get_hip_runtime

_SOURCE = Path(__file__).with_name("surya_ops.hip")
_OUTPUT_NAME = "surya_ops"

_P = ctypes.c_void_p
_F = ctypes.c_float
_I = ctypes.c_int64
_S = ctypes.c_void_p


def plan_surya_ops_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: str = "decode",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="surya_ops",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
    )


def build_surya_ops(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: str = "decode",
    dry_run: bool = False,
    load: bool = True,
) -> ctypes.CDLL | None:
    return build_hip(
        sources=[_SOURCE],
        family="surya_ops",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
    )


def surya_split_qgate_f32(
    src_ptr: int,
    q_ptr: int,
    gate_ptr: int,
    tokens: int,
    heads: int,
    head_dim: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    runtime = runtime or get_hip_runtime()
    library = library or build_surya_ops(load=True)
    fn = _fn(library, "hipengine_surya_split_qgate_f32", [_P, _P, _P, _I, _I, _I, _S])
    err = fn(_P(src_ptr), _P(q_ptr), _P(gate_ptr), _I(tokens), _I(heads), _I(head_dim), _S(stream))
    _check(err, runtime, "surya split_qgate")


def surya_gdn_l2norm_f32(
    src_ptr: int,
    q_ptr: int,
    k_ptr: int,
    q_scale: float,
    tokens: int,
    heads: int,
    head_dim: int,
    src_stride: int,
    k_offset: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    runtime = runtime or get_hip_runtime()
    library = library or build_surya_ops(load=True)
    fn = _fn(library, "hipengine_surya_gdn_l2norm_f32",
             [_P, _P, _P, _F, _I, _I, _I, _I, _I, _S])
    err = fn(_P(src_ptr), _P(q_ptr), _P(k_ptr), _F(q_scale), _I(tokens),
             _I(heads), _I(head_dim), _I(src_stride), _I(k_offset), _S(stream))
    _check(err, runtime, "surya gdn_l2norm")


def surya_rmsnorm_f32(
    x_ptr: int,
    w_ptr: int,
    out_ptr: int,
    rows: int,
    hidden: int,
    eps: float = 1e-6,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    runtime = runtime or get_hip_runtime()
    library = library or build_surya_ops(load=True)
    fn = _fn(library, "hipengine_surya_rmsnorm_f32", [_P, _P, _P, _I, _I, _F, _S])
    err = fn(_P(x_ptr), _P(w_ptr), _P(out_ptr), _I(rows), _I(hidden), _F(eps), _S(stream))
    _check(err, runtime, "surya rmsnorm")


def surya_scatter_kv_f32(
    src_ptr: int,
    dst_ptr: int,
    tokens: int,
    token_offset: int,
    kv_heads: int,
    head_dim: int,
    max_seq: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    runtime = runtime or get_hip_runtime()
    library = library or build_surya_ops(load=True)
    fn = _fn(library, "hipengine_surya_scatter_kv_f32",
             [_P, _P, _I, _I, _I, _I, _I, _S])
    err = fn(_P(src_ptr), _P(dst_ptr), _I(tokens), _I(token_offset),
             _I(kv_heads), _I(head_dim), _I(max_seq), _S(stream))
    _check(err, runtime, "surya scatter_kv")


def surya_causal_mask_scale_f32(
    scores_ptr: int,
    scale: float,
    heads: int,
    queries: int,
    head_stride: int,
    query_offset: int = 0,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Mask and scale a causal score block.

    ``queries`` is the number of query rows in ``scores`` and ``head_stride``
    is both the key count and the row stride; entry ``(h, i, j)`` is dropped to
    ``-inf`` when ``j > query_offset + i``. A query-row tile passes the index
    of its first query as ``query_offset`` so the mask stays absolute.
    """

    runtime = runtime or get_hip_runtime()
    library = library or build_surya_ops(load=True)
    fn = _fn(library, "hipengine_surya_causal_mask_scale_f32",
             [_P, _F, _I, _I, _I, _I, _S])
    err = fn(_P(scores_ptr), _F(scale), _I(heads), _I(queries), _I(head_stride),
             _I(query_offset), _S(stream))
    _check(err, runtime, "surya causal_mask_scale")


def _fn(library: ctypes.CDLL, symbol: str, argtypes: list) -> ctypes._FuncPtr:
    fn = getattr(library, symbol, None)
    if fn is None:
        raise RuntimeError(f"missing symbol {symbol}")
    fn.argtypes = argtypes
    fn.restype = ctypes.c_int
    return fn


def _check(err: int, runtime: HipRuntime, what: str) -> None:
    if err != 0:
        raise RuntimeError(f"{what} failed: {err} ({runtime.last_error_message() if hasattr(runtime, 'last_error_message') else 'see hipGetLastError'})")
