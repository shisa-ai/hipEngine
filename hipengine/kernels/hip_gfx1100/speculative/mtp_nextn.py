"""Raw-pointer + numpy wrappers for native MTP NextN draft-head kernels (GGUF path).

M3 deliverable: a *real* GPU NextN ``eh_proj`` sub-kernel registered under
``KernelKey(backend, "mtp_nextn_eh_proj", "gguf_f32", "qwen35")`` for both
``hip_gfx1100`` and ``hip_gfx1151``.  Without this, the registry silently falls
back to the ``cpu_reference`` numpy oracle (``registry._candidate_keys`` appends
``cpu_reference`` last), so M3 had no native runtime kernel.

These F32 kernels are correctness-first and size-agnostic: they mirror
``cpu_reference`` math exactly so the M3 fixture gate runs on a real GPU.  M6
swaps the inner GEMVs for WMMA / K-quant tuned kernels on real shapes; these
remain the correctness baseline.

Importing this module registers the wrappers but does not build or load ROCm
until a wrapper is called.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.ctypes_cache import signed_kernel_fn
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("mtp_nextn.hip")
_OUTPUT_NAME = "mtp_nextn.so"
_SYMBOL_RMSNORM_F32 = "hipengine_mtp_rmsnorm_f32"
_SYMBOL_EH_PROJ_F32 = "hipengine_mtp_eh_proj_f32"

# ptr(s) + rows(int64) + hidden(int64) + eps(float) + stream
_ARGTYPES_RMSNORM_F32 = (
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int64, ctypes.c_int64,
    ctypes.c_float,
    ctypes.c_void_p,
)
# ptr(s) + rows(int64) + hidden(int64) + stream
_ARGTYPES_EH_PROJ_F32 = (
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int64, ctypes.c_int64,
    ctypes.c_void_p,
)


def plan_mtp_nextn_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="mtp_nextn",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
    )


def build_mtp_nextn(
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
        family="mtp_nextn",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def _check_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _check_launch(runtime: HipRuntime, err: int) -> None:
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def mtp_rmsnorm_f32(
    x_ptr: int,
    weight_ptr: int,
    out_ptr: int,
    rows: int,
    hidden: int,
    eps: float = 1e-6,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch F32 RMSNorm, one block per row.  ``out = x * rsqrt(mean(x^2)+eps) * weight``."""

    _check_positive("rows", rows)
    _check_positive("hidden", hidden)
    library = library or build_mtp_nextn(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(library, _SYMBOL_RMSNORM_F32, _ARGTYPES_RMSNORM_F32, ctypes.c_int)
    err = fn(x_ptr, weight_ptr, out_ptr, rows, hidden, float(eps), stream)
    _check_launch(runtime, err)


def mtp_eh_proj_f32(
    e_norm_ptr: int,
    h_norm_ptr: int,
    weight_ptr: int,
    out_ptr: int,
    rows: int,
    hidden: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch F32 eh_proj GEMV.

    ``out[row, j] = sum_k e_norm[row,k]*weight[j,k] + h_norm[row,k]*weight[j,k+hidden]``
    with ``weight`` row-major ``[hidden, 2*hidden]`` (matches ``fused @ weight.T``).
    """

    _check_positive("rows", rows)
    _check_positive("hidden", hidden)
    library = library or build_mtp_nextn(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(library, _SYMBOL_EH_PROJ_F32, _ARGTYPES_EH_PROJ_F32, ctypes.c_int)
    err = fn(e_norm_ptr, h_norm_ptr, weight_ptr, out_ptr, rows, hidden, stream)
    _check_launch(runtime, err)


# ptr(x) + ptr(weight) + ptr(out) + tokens + in_features + out_features + stream
_ARGTYPES_LINEAR_F32 = (
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
    ctypes.c_void_p,
)


def mtp_linear_f32(
    x_ptr: int,
    weight_ptr: int,
    out_ptr: int,
    tokens: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch F32 linear: ``out[t, j] = sum_k x[t,k] * weight[j,k]``.

    ``weight`` is row-major ``[out_features, in_features]`` (matches ``x @ weight.T``).
    """

    _check_positive("tokens", tokens)
    _check_positive("in_features", in_features)
    _check_positive("out_features", out_features)
    library = library or build_mtp_nextn(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(library, "hipengine_mtp_linear_f32", _ARGTYPES_LINEAR_F32, ctypes.c_int)
    err = fn(x_ptr, weight_ptr, out_ptr, tokens, in_features, out_features, stream)
    _check_launch(runtime, err)


# ptr(attn) + ptr(gate) + ptr(out) + rows + head_dim + stream
_ARGTYPES_SIGMOID_GATE_MUL_F32 = (
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int64, ctypes.c_int64,
    ctypes.c_void_p,
)


def mtp_sigmoid_gate_mul_f32(
    attn_ptr: int,
    gate_ptr: int,
    out_ptr: int,
    rows: int,
    head_dim: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch ``out[row, d] = attn[row, d] * sigmoid(gate[row, d])``."""

    _check_positive("rows", rows)
    _check_positive("head_dim", head_dim)
    library = library or build_mtp_nextn(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(
        library, "hipengine_mtp_sigmoid_gate_mul_f32", _ARGTYPES_SIGMOID_GATE_MUL_F32, ctypes.c_int
    )
    err = fn(attn_ptr, gate_ptr, out_ptr, rows, head_dim, stream)
    _check_launch(runtime, err)


# ptr(a) + ptr(b) + ptr(out) + n + stream
_ARGTYPES_ADD_F32 = (
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int64,
    ctypes.c_void_p,
)


def mtp_add_f32(
    a_ptr: int,
    b_ptr: int,
    out_ptr: int,
    n: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch elementwise ``out = a + b``."""

    _check_positive("n", n)
    library = library or build_mtp_nextn(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(library, "hipengine_mtp_add_f32", _ARGTYPES_ADD_F32, ctypes.c_int)
    err = fn(a_ptr, b_ptr, out_ptr, n, stream)
    _check_launch(runtime, err)


# query + key_cache + value_cache + positions + context_counts + out
#   + tokens + heads + kv_heads + qk_head_dim + value_head_dim + cache_tokens + scale(float) + stream
_ARGTYPES_DENSE_ATTN_F32 = (
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_float,
    ctypes.c_void_p,
)


def mtp_dense_attn_f32(
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    positions_ptr: int,
    context_counts_ptr: int,
    out_ptr: int,
    tokens: int,
    heads: int,
    kv_heads: int,
    qk_head_dim: int,
    value_head_dim: int,
    cache_tokens: int,
    scale: float,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch dense causal GQA attention (F32, correctness-first).

    See ``hipengine_mtp_dense_attn_f32_kernel`` for the math.  ``query`` is
    ``[tokens, heads, qk_head_dim]``; ``key_cache``/``value_cache`` are
    ``[cache_tokens, kv_heads, head_dim]``; ``positions``/``context_counts`` are
    ``[tokens]`` int64; ``out`` is ``[tokens, heads, value_head_dim]``.
    """

    for name, value in (
        ("tokens", tokens), ("heads", heads), ("kv_heads", kv_heads),
        ("qk_head_dim", qk_head_dim), ("value_head_dim", value_head_dim),
        ("cache_tokens", cache_tokens),
    ):
        _check_positive(name, value)
    library = library or build_mtp_nextn(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(
        library, "hipengine_mtp_dense_attn_f32", _ARGTYPES_DENSE_ATTN_F32, ctypes.c_int
    )
    err = fn(
        query_ptr, key_cache_ptr, value_cache_ptr, positions_ptr, context_counts_ptr, out_ptr,
        tokens, heads, kv_heads, qk_head_dim, value_head_dim, cache_tokens, float(scale), stream,
    )
    _check_launch(runtime, err)


# ptr(gate) + ptr(up) + ptr(out) + n + stream
_ARGTYPES_SILU_MUL_F32 = (
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int64,
    ctypes.c_void_p,
)


def mtp_silu_mul_f32(
    gate_ptr: int,
    up_ptr: int,
    out_ptr: int,
    n: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch fused ``out = silu(gate) * up`` (SiLU = x/(1+exp(-x)))."""

    _check_positive("n", n)
    library = library or build_mtp_nextn(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(library, "hipengine_mtp_silu_mul_f32", _ARGTYPES_SILU_MUL_F32, ctypes.c_int)
    err = fn(gate_ptr, up_ptr, out_ptr, n, stream)
    _check_launch(runtime, err)


# ptr(a) + ptr(b) + ptr(out) + n + stream
_ARGTYPES_MUL_F32 = (
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int64,
    ctypes.c_void_p,
)


def mtp_mul_f32(
    a_ptr: int,
    b_ptr: int,
    out_ptr: int,
    n: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch elementwise ``out = a * b``."""

    _check_positive("n", n)
    library = library or build_mtp_nextn(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(library, "hipengine_mtp_mul_f32", _ARGTYPES_MUL_F32, ctypes.c_int)
    err = fn(a_ptr, b_ptr, out_ptr, n, stream)
    _check_launch(runtime, err)


# ptr(x) + ptr(out) + scalar(float) + n + stream
_ARGTYPES_SCALE_F32 = (
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_float,
    ctypes.c_int64,
    ctypes.c_void_p,
)


def mtp_scale_f32(
    x_ptr: int,
    out_ptr: int,
    scalar: float,
    n: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch ``out = scalar * x``."""

    _check_positive("n", n)
    library = library or build_mtp_nextn(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(library, "hipengine_mtp_scale_f32", _ARGTYPES_SCALE_F32, ctypes.c_int)
    err = fn(x_ptr, out_ptr, float(scalar), n, stream)
    _check_launch(runtime, err)


# ptr(scale) + ptr(x) + ptr(out) + tokens + hidden + stream
_ARGTYPES_ROW_SCALE_F32 = (
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int64, ctypes.c_int64,
    ctypes.c_void_p,
)


def mtp_row_scale_f32(
    scale_ptr: int,
    x_ptr: int,
    out_ptr: int,
    tokens: int,
    hidden: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch broadcast row-scale ``out[t, d] = scale[t] * x[t, d]``."""

    _check_positive("tokens", tokens)
    _check_positive("hidden", hidden)
    library = library or build_mtp_nextn(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(
        library, "hipengine_mtp_row_scale_f32", _ARGTYPES_ROW_SCALE_F32, ctypes.c_int
    )
    err = fn(scale_ptr, x_ptr, out_ptr, tokens, hidden, stream)
    _check_launch(runtime, err)


def qwen35_gguf_mtp_eh_proj_f32(
    hidden_seed: "np.ndarray",
    token_embedding: "np.ndarray",
    eh_proj_weight: "np.ndarray",
    hnorm_weight: "np.ndarray",
    enorm_weight: "np.ndarray",
    eps: float = 1e-6,
) -> np.ndarray:
    """Numpy-in/out wrapper matching ``cpu_reference.qwen35_gguf_mtp_eh_proj``.

    enorm(embed) + hnorm(hidden) -> eh_proj F32 GEMV.  Returns ``[rows, hidden]`` F32.
    """

    hidden_arr = np.ascontiguousarray(hidden_seed, dtype=np.float32)
    embed_arr = np.ascontiguousarray(token_embedding, dtype=np.float32)
    weight = np.ascontiguousarray(eh_proj_weight, dtype=np.float32)
    hnorm = np.ascontiguousarray(hnorm_weight, dtype=np.float32)
    enorm = np.ascontiguousarray(enorm_weight, dtype=np.float32)
    if hidden_arr.ndim != 2:
        raise ValueError("hidden_seed must have shape [rows, hidden]")
    rows, hidden = hidden_arr.shape
    if embed_arr.shape != hidden_arr.shape:
        raise ValueError("token_embedding must match hidden_seed shape")
    if weight.shape != (hidden, hidden * 2):
        raise ValueError(
            f"eh_proj_weight must have shape [hidden, 2*hidden]=[{hidden}, {hidden * 2}]; "
            f"got {weight.shape}"
        )
    if hnorm.shape != (hidden,):
        raise ValueError("hnorm_weight must have shape [hidden]")
    if enorm.shape != (hidden,):
        raise ValueError("enorm_weight must have shape [hidden]")

    runtime = get_hip_runtime()
    e_norm_dev = malloc(embed_arr.nbytes, runtime=runtime)
    h_norm_dev = malloc(hidden_arr.nbytes, runtime=runtime)
    out_dev = malloc(hidden_arr.nbytes, runtime=runtime)
    buffers = [e_norm_dev, h_norm_dev, out_dev]
    try:
        embed_dev = malloc(embed_arr.nbytes, runtime=runtime); buffers.append(embed_dev)
        hidden_dev = malloc(hidden_arr.nbytes, runtime=runtime); buffers.append(hidden_dev)
        weight_dev = malloc(weight.nbytes, runtime=runtime); buffers.append(weight_dev)
        hnorm_dev = malloc(hnorm.nbytes, runtime=runtime); buffers.append(hnorm_dev)
        enorm_dev = malloc(enorm.nbytes, runtime=runtime); buffers.append(enorm_dev)
        copy_host_to_device(embed_dev, host_array_ptr(embed_arr), runtime=runtime)
        copy_host_to_device(hidden_dev, host_array_ptr(hidden_arr), runtime=runtime)
        copy_host_to_device(weight_dev, host_array_ptr(weight), runtime=runtime)
        copy_host_to_device(hnorm_dev, host_array_ptr(hnorm), runtime=runtime)
        copy_host_to_device(enorm_dev, host_array_ptr(enorm), runtime=runtime)
        # h_norm = rmsnorm(hidden, hnorm); e_norm = rmsnorm(embed, enorm)
        mtp_rmsnorm_f32(embed_dev.ptr, enorm_dev.ptr, e_norm_dev.ptr, rows, hidden, eps=eps,
                        runtime=runtime)
        mtp_rmsnorm_f32(hidden_dev.ptr, hnorm_dev.ptr, h_norm_dev.ptr, rows, hidden, eps=eps,
                        runtime=runtime)
        mtp_eh_proj_f32(e_norm_dev.ptr, h_norm_dev.ptr, weight_dev.ptr, out_dev.ptr,
                        rows, hidden, runtime=runtime)
        runtime.device_synchronize()
        out = np.empty((rows, hidden), dtype=np.float32)
        copy_device_to_host(host_array_ptr(out), out_dev, runtime=runtime)
        return out
    finally:
        for buf in buffers:
            free(buf, runtime=runtime)


def qwen35_gguf_mtp_attention_sublayer_f32(
    hidden: "np.ndarray",
    attn_norm_weight: "np.ndarray",
    wq_weight: "np.ndarray",
    wk_weight: "np.ndarray",
    wv_weight: "np.ndarray",
    wo_weight: "np.ndarray",
    q_norm_weight: "np.ndarray",
    k_norm_weight: "np.ndarray",
    *,
    num_heads: int,
    num_kv_heads: int,
    positions: "np.ndarray | None" = None,
    context_counts: "np.ndarray | None" = None,
    key_cache: "np.ndarray | None" = None,
    value_cache: "np.ndarray | None" = None,
    rope_cos: "np.ndarray | None" = None,
    rope_sin: "np.ndarray | None" = None,
    rotary_dim: int | None = None,
    scale: float | None = None,
    eps: float = 1e-6,
) -> np.ndarray:
    """Numpy-in/out wrapper matching ``cpu_reference.qwen35_gguf_mtp_attention_sublayer``.

    M3 scope (correctness-first): implements the DEFAULT dense attention path
    used by the F32 fixture -- ``positions=arange(tokens)``,
    ``context_counts=pos+1``, no RoPE, dense cache = the current tokens' K/V.
    The RoPE and KVLiveSpans paged-cache branches raise ``NotImplementedError``
    (M6 work).  ``value_head_dim`` must equal ``qk_head_dim`` for gated Qwen35
    attention, matching the cpu_reference contract.
    """

    x = np.ascontiguousarray(hidden, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError("hidden must have shape [tokens, hidden]")
    tokens, hidden_size = x.shape
    heads = int(num_heads)
    kv_heads = int(num_kv_heads)
    if heads <= 0 or kv_heads <= 0:
        raise ValueError("num_heads and num_kv_heads must be positive")
    if heads % kv_heads != 0:
        raise ValueError("num_heads must be divisible by num_kv_heads")

    attn_norm = np.ascontiguousarray(attn_norm_weight, dtype=np.float32)
    q_norm = np.ascontiguousarray(q_norm_weight, dtype=np.float32)
    k_norm = np.ascontiguousarray(k_norm_weight, dtype=np.float32)
    if attn_norm.shape != (hidden_size,):
        raise ValueError("attn_norm_weight must have shape [hidden]")
    if q_norm.ndim != 1:
        raise ValueError("q_norm_weight must have shape [qk_head_dim]")
    qk_head_dim = q_norm.shape[0]
    if k_norm.shape != (qk_head_dim,):
        raise ValueError("k_norm_weight must have shape [qk_head_dim]")

    wq = np.ascontiguousarray(wq_weight, dtype=np.float32)
    wk = np.ascontiguousarray(wk_weight, dtype=np.float32)
    wv = np.ascontiguousarray(wv_weight, dtype=np.float32)
    wo = np.ascontiguousarray(wo_weight, dtype=np.float32)
    if wq.shape != (heads * 2 * qk_head_dim, hidden_size):
        raise ValueError("wq_weight must have shape [num_heads * 2 * qk_head_dim, hidden]")
    if wk.shape != (kv_heads * qk_head_dim, hidden_size):
        raise ValueError("wk_weight must have shape [num_kv_heads * qk_head_dim, hidden]")
    if wv.ndim != 2 or wv.shape[1] != hidden_size or wv.shape[0] % kv_heads != 0:
        raise ValueError("wv_weight must have shape [num_kv_heads * value_head_dim, hidden]")
    value_head_dim = wv.shape[0] // kv_heads
    if value_head_dim != qk_head_dim:
        raise ValueError("value_head_dim must match qk_head_dim for gated Qwen35 attention")
    if wo.shape != (hidden_size, heads * value_head_dim):
        raise ValueError("wo_weight must have shape [hidden, num_heads * value_head_dim]")

    if rope_cos is not None or rope_sin is not None:
        raise NotImplementedError("RoPE path is M6 work; F32 M3 fixture does not exercise it")
    if key_cache is not None or value_cache is not None:
        raise NotImplementedError(
            "external KV cache path is M6 work; F32 M3 fixture uses the current-token K/V"
        )

    pos = (
        np.arange(tokens, dtype=np.int64) if positions is None
        else np.ascontiguousarray(positions, dtype=np.int64)
    )
    ctx = (
        pos + 1 if context_counts is None
        else np.ascontiguousarray(context_counts, dtype=np.int64)
    )
    scale_value = (qk_head_dim ** -0.5) if scale is None else float(scale)

    runtime = get_hip_runtime()
    buffers: list = []
    try:
        # normed = rmsnorm(hidden, attn_norm)
        normed_dev = malloc(x.nbytes, runtime=runtime); buffers.append(normed_dev)
        hidden_dev = malloc(x.nbytes, runtime=runtime); buffers.append(hidden_dev)
        attn_norm_dev = malloc(attn_norm.nbytes, runtime=runtime); buffers.append(attn_norm_dev)
        copy_host_to_device(hidden_dev, host_array_ptr(x), runtime=runtime)
        copy_host_to_device(attn_norm_dev, host_array_ptr(attn_norm), runtime=runtime)
        mtp_rmsnorm_f32(hidden_dev.ptr, attn_norm_dev.ptr, normed_dev.ptr, tokens, hidden_size,
                        eps=eps, runtime=runtime)

        # q_full = normed @ wq.T  -> [tokens, heads*2*qk_head_dim]
        q_full_dev = malloc(tokens * heads * 2 * qk_head_dim * 4, runtime=runtime)
        buffers.append(q_full_dev)
        wq_dev = malloc(wq.nbytes, runtime=runtime); buffers.append(wq_dev)
        copy_host_to_device(wq_dev, host_array_ptr(wq), runtime=runtime)
        mtp_linear_f32(normed_dev.ptr, wq_dev.ptr, q_full_dev.ptr, tokens, hidden_size,
                       heads * 2 * qk_head_dim, runtime=runtime)
        # query = rmsnorm(q_full[...,0,:], q_norm)  over qk_head_dim, rows = tokens*heads
        query_dev = malloc(tokens * heads * qk_head_dim * 4, runtime=runtime)
        buffers.append(query_dev)
        q_norm_dev = malloc(q_norm.nbytes, runtime=runtime); buffers.append(q_norm_dev)
        copy_host_to_device(q_norm_dev, host_array_ptr(q_norm), runtime=runtime)
        # q_full layout [t, h, 2, d]: the q-half is stride-2 over the (2,d) block.
        # Extract query (q_full[...,0,:]) and gate (q_full[...,1,:]) via host reshape is
        # simplest for correctness-first: D2H the small q_full, split, re-upload.
        q_full_host = np.empty((tokens, heads, 2, qk_head_dim), dtype=np.float32)
        copy_device_to_host(host_array_ptr(q_full_host), q_full_dev, runtime=runtime)
        query_host = np.ascontiguousarray(q_full_host[:, :, 0, :])
        gate_host = np.ascontiguousarray(q_full_host[:, :, 1, :])
        copy_host_to_device(query_dev, host_array_ptr(query_host), runtime=runtime)
        # per-head rmsnorm over qk_head_dim
        mtp_rmsnorm_f32(query_dev.ptr, q_norm_dev.ptr, query_dev.ptr, tokens * heads, qk_head_dim,
                        eps=eps, runtime=runtime)
        gate_dev = malloc(gate_host.nbytes, runtime=runtime); buffers.append(gate_dev)
        copy_host_to_device(gate_dev, host_array_ptr(gate_host), runtime=runtime)

        # key_cur = rmsnorm(normed @ wk.T, k_norm)  -> [tokens, kv_heads, qk_head_dim]
        key_cur_dev = malloc(tokens * kv_heads * qk_head_dim * 4, runtime=runtime)
        buffers.append(key_cur_dev)
        wk_dev = malloc(wk.nbytes, runtime=runtime); buffers.append(wk_dev)
        k_norm_dev = malloc(k_norm.nbytes, runtime=runtime); buffers.append(k_norm_dev)
        copy_host_to_device(wk_dev, host_array_ptr(wk), runtime=runtime)
        copy_host_to_device(k_norm_dev, host_array_ptr(k_norm), runtime=runtime)
        mtp_linear_f32(normed_dev.ptr, wk_dev.ptr, key_cur_dev.ptr, tokens, hidden_size,
                       kv_heads * qk_head_dim, runtime=runtime)
        mtp_rmsnorm_f32(key_cur_dev.ptr, k_norm_dev.ptr, key_cur_dev.ptr, tokens * kv_heads,
                        qk_head_dim, eps=eps, runtime=runtime)

        # value_cur = normed @ wv.T  -> [tokens, kv_heads, value_head_dim]
        value_cur_dev = malloc(tokens * kv_heads * value_head_dim * 4, runtime=runtime)
        buffers.append(value_cur_dev)
        wv_dev = malloc(wv.nbytes, runtime=runtime); buffers.append(wv_dev)
        copy_host_to_device(wv_dev, host_array_ptr(wv), runtime=runtime)
        mtp_linear_f32(normed_dev.ptr, wv_dev.ptr, value_cur_dev.ptr, tokens, hidden_size,
                       kv_heads * value_head_dim, runtime=runtime)

        # dense causal GQA: cache = current tokens (default path), cache_tokens = tokens
        pos_dev = malloc(pos.nbytes, runtime=runtime); buffers.append(pos_dev)
        ctx_dev = malloc(ctx.nbytes, runtime=runtime); buffers.append(ctx_dev)
        copy_host_to_device(pos_dev, host_array_ptr(pos), runtime=runtime)
        copy_host_to_device(ctx_dev, host_array_ptr(ctx), runtime=runtime)
        attn_dev = malloc(tokens * heads * value_head_dim * 4, runtime=runtime)
        buffers.append(attn_dev)
        mtp_dense_attn_f32(
            query_dev.ptr, key_cur_dev.ptr, value_cur_dev.ptr, pos_dev.ptr, ctx_dev.ptr,
            attn_dev.ptr, tokens, heads, kv_heads, qk_head_dim, value_head_dim, tokens,
            scale_value, runtime=runtime,
        )

        # gated = attn * sigmoid(gate)
        gated_dev = malloc(tokens * heads * value_head_dim * 4, runtime=runtime)
        buffers.append(gated_dev)
        mtp_sigmoid_gate_mul_f32(attn_dev.ptr, gate_dev.ptr, gated_dev.ptr, tokens * heads,
                                 value_head_dim, runtime=runtime)

        # wo_out = gated @ wo.T  -> [tokens, hidden]
        wo_dev = malloc(wo.nbytes, runtime=runtime); buffers.append(wo_dev)
        copy_host_to_device(wo_dev, host_array_ptr(wo), runtime=runtime)
        wo_out_dev = malloc(tokens * hidden_size * 4, runtime=runtime); buffers.append(wo_out_dev)
        mtp_linear_f32(gated_dev.ptr, wo_dev.ptr, wo_out_dev.ptr, tokens, heads * value_head_dim,
                       hidden_size, runtime=runtime)

        # out = hidden + wo_out
        out_dev = malloc(tokens * hidden_size * 4, runtime=runtime); buffers.append(out_dev)
        mtp_add_f32(hidden_dev.ptr, wo_out_dev.ptr, out_dev.ptr, tokens * hidden_size,
                    runtime=runtime)
        runtime.device_synchronize()
        out = np.empty((tokens, hidden_size), dtype=np.float32)
        copy_device_to_host(host_array_ptr(out), out_dev, runtime=runtime)
        return out
    finally:
        for buf in buffers:
            free(buf, runtime=runtime)


def qwen35_gguf_mtp_moe_routing_f32(
    hidden: "np.ndarray",
    router_weight: "np.ndarray",
    *,
    experts_used: int,
    expert_weights_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Numpy-in/out wrapper matching ``cpu_reference.qwen35_gguf_mtp_moe_routing``.

    Router linear on GPU, softmax/top-k/renorm on host (correctness-first for
    the tiny [tokens, experts] fixture; M6 GPU-accelerates the softmax/top-k).
    """

    x = np.ascontiguousarray(hidden, dtype=np.float32)
    router = np.ascontiguousarray(router_weight, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError("hidden must have shape [tokens, hidden]")
    tokens, hidden_size = x.shape
    top_k = int(experts_used)
    experts = router.shape[0]
    if top_k <= 0:
        raise ValueError("experts_used must be positive")
    if top_k > experts:
        raise ValueError("experts_used must be <= number of experts")

    runtime = get_hip_runtime()
    buffers: list = []
    try:
        x_dev = malloc(x.nbytes, runtime=runtime); buffers.append(x_dev)
        router_dev = malloc(router.nbytes, runtime=runtime); buffers.append(router_dev)
        logits_dev = malloc(tokens * experts * 4, runtime=runtime); buffers.append(logits_dev)
        copy_host_to_device(x_dev, host_array_ptr(x), runtime=runtime)
        copy_host_to_device(router_dev, host_array_ptr(router), runtime=runtime)
        mtp_linear_f32(x_dev.ptr, router_dev.ptr, logits_dev.ptr, tokens, hidden_size, experts,
                       runtime=runtime)
        logits = np.empty((tokens, experts), dtype=np.float32)
        copy_device_to_host(host_array_ptr(logits), logits_dev, runtime=runtime)
    finally:
        for buf in buffers:
            free(buf, runtime=runtime)

    # softmax + top-k + renorm on host (correctness-first)
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    probs = (exp / np.sum(exp, axis=-1, keepdims=True)).astype(np.float32)
    selected = np.argsort(-probs, axis=-1, kind="stable")[:, :top_k].astype(np.int64)
    selected_weights = np.take_along_axis(probs, selected, axis=-1).astype(np.float32)
    weight_sum = np.maximum(
        np.sum(selected_weights, axis=-1, keepdims=True), np.float32(6.103515625e-5)
    )
    selected_weights = (selected_weights / weight_sum).astype(np.float32)
    scale_value = float(expert_weights_scale)
    if scale_value != 0.0 and scale_value != 1.0:
        selected_weights = (selected_weights * np.float32(scale_value)).astype(np.float32)
    return selected, selected_weights


def qwen35_gguf_mtp_ffn_sublayer_f32(
    hidden: "np.ndarray",
    attn_post_norm_weight: "np.ndarray",
    router_weight: "np.ndarray",
    gate_qweight: "np.ndarray",
    up_qweight: "np.ndarray",
    down_qweight: "np.ndarray",
    gate_qtype: "GGMLQuantizationType",
    up_qtype: "GGMLQuantizationType",
    down_qtype: "GGMLQuantizationType",
    shared_gate_logit_weight: "np.ndarray",
    shared_gate_qweight: "np.ndarray",
    shared_up_qweight: "np.ndarray",
    shared_down_qweight: "np.ndarray",
    shared_qtype: "GGMLQuantizationType",
    *,
    experts_used: int,
    expert_weights_scale: float = 1.0,
    eps: float = 1e-6,
) -> np.ndarray:
    """Numpy-in/out wrapper matching ``cpu_reference.qwen35_gguf_mtp_ffn_sublayer``.

    M3 scope (correctness-first): F32 qtype only (all quant-gemv = x @ W.T).
    K-quant (Q4_K/Q5_K/Q8_0) expert paths raise NotImplementedError (M6).  Router
    softmax/top-k on host; expert FFN + shared expert + residual on GPU.
    """

    from hipengine.quant.gguf import GGMLQuantizationType

    for qt, name in (
        (gate_qtype, "gate_qtype"), (up_qtype, "up_qtype"), (down_qtype, "down_qtype"),
        (shared_qtype, "shared_qtype"),
    ):
        if qt != GGMLQuantizationType.F32:
            raise NotImplementedError(
                f"{name}={qt.name} not supported in M3 (F32-only); K-quant is M6 work"
            )

    x = np.ascontiguousarray(hidden, dtype=np.float32)
    norm_weight = np.ascontiguousarray(attn_post_norm_weight, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError("hidden must have shape [tokens, hidden]")
    tokens, hidden_size = x.shape
    if norm_weight.shape != (hidden_size,):
        raise ValueError("attn_post_norm_weight must have shape [hidden]")

    gate_q = np.ascontiguousarray(gate_qweight, dtype=np.float32)
    up_q = np.ascontiguousarray(up_qweight, dtype=np.float32)
    down_q = np.ascontiguousarray(down_qweight, dtype=np.float32)
    if gate_q.ndim != 3 or up_q.ndim != 3 or down_q.ndim != 3:
        raise ValueError("expert weights must be rank-3 [E, out, in]")
    num_experts = gate_q.shape[0]
    inter_dim = gate_q.shape[1]
    shared_gate_q = np.ascontiguousarray(shared_gate_qweight, dtype=np.float32)
    shared_up_q = np.ascontiguousarray(shared_up_qweight, dtype=np.float32)
    shared_down_q = np.ascontiguousarray(shared_down_qweight, dtype=np.float32)
    gate_vec = np.ascontiguousarray(shared_gate_logit_weight, dtype=np.float32)

    runtime = get_hip_runtime()
    buffers: list = []
    try:
        # normed = rmsnorm(hidden, attn_post_norm)
        normed_dev = malloc(x.nbytes, runtime=runtime); buffers.append(normed_dev)
        x_dev = malloc(x.nbytes, runtime=runtime); buffers.append(x_dev)
        norm_dev = malloc(norm_weight.nbytes, runtime=runtime); buffers.append(norm_dev)
        copy_host_to_device(x_dev, host_array_ptr(x), runtime=runtime)
        copy_host_to_device(norm_dev, host_array_ptr(norm_weight), runtime=runtime)
        mtp_rmsnorm_f32(x_dev.ptr, norm_dev.ptr, normed_dev.ptr, tokens, hidden_size, eps=eps,
                        runtime=runtime)

        # routing (router linear on GPU, softmax/topk on host)
        selected_experts, routing_weights = qwen35_gguf_mtp_moe_routing_f32(
            x.copy(),  # host copy of hidden for the host-side routing shim
            router_weight,
            experts_used=experts_used,
            expert_weights_scale=expert_weights_scale,
        )
        top_k = selected_experts.shape[1]

        # selected_out = sum over selected experts of routing_weight * down(silu(gate)*up)
        selected_out_dev = malloc(tokens * hidden_size * 4, runtime=runtime)
        buffers.append(selected_out_dev)
        # zero-init
        zero = np.zeros((tokens, hidden_size), dtype=np.float32)
        copy_host_to_device(selected_out_dev, host_array_ptr(zero), runtime=runtime)

        normed_host = np.empty((tokens, hidden_size), dtype=np.float32)
        copy_device_to_host(host_array_ptr(normed_host), normed_dev, runtime=runtime)

        for t in range(tokens):
            xt = np.ascontiguousarray(normed_host[t : t + 1])
            xt_dev = malloc(xt.nbytes, runtime=runtime); buffers.append(xt_dev)
            copy_host_to_device(xt_dev, host_array_ptr(xt), runtime=runtime)
            for k in range(top_k):
                e = int(selected_experts[t, k])
                w = float(routing_weights[t, k])
                g_w = np.ascontiguousarray(gate_q[e])  # [inter, hidden]
                u_w = np.ascontiguousarray(up_q[e])    # [inter, hidden]
                d_w = np.ascontiguousarray(down_q[e])  # [hidden, inter]
                g_dev = malloc(g_w.nbytes, runtime=runtime); buffers.append(g_dev)
                u_dev = malloc(u_w.nbytes, runtime=runtime); buffers.append(u_dev)
                d_dev = malloc(d_w.nbytes, runtime=runtime); buffers.append(d_dev)
                copy_host_to_device(g_dev, host_array_ptr(g_w), runtime=runtime)
                copy_host_to_device(u_dev, host_array_ptr(u_w), runtime=runtime)
                copy_host_to_device(d_dev, host_array_ptr(d_w), runtime=runtime)
                gate_out = malloc(1 * inter_dim * 4, runtime=runtime); buffers.append(gate_out)
                up_out = malloc(1 * inter_dim * 4, runtime=runtime); buffers.append(up_out)
                inter_out = malloc(1 * inter_dim * 4, runtime=runtime); buffers.append(inter_out)
                down_out = malloc(1 * hidden_size * 4, runtime=runtime); buffers.append(down_out)
                scaled = malloc(1 * hidden_size * 4, runtime=runtime); buffers.append(scaled)
                mtp_linear_f32(xt_dev.ptr, g_dev.ptr, gate_out.ptr, 1, hidden_size, inter_dim,
                               runtime=runtime)
                mtp_linear_f32(xt_dev.ptr, u_dev.ptr, up_out.ptr, 1, hidden_size, inter_dim,
                               runtime=runtime)
                mtp_silu_mul_f32(gate_out.ptr, up_out.ptr, inter_out.ptr, inter_dim,
                                 runtime=runtime)
                mtp_linear_f32(inter_out.ptr, d_dev.ptr, down_out.ptr, 1, inter_dim, hidden_size,
                               runtime=runtime)
                mtp_scale_f32(down_out.ptr, scaled.ptr, w, hidden_size, runtime=runtime)
                # selected_out[t] += scaled
                row_dev = malloc(hidden_size * 4, runtime=runtime); buffers.append(row_dev)
                mtp_add_f32(selected_out_dev.ptr + t * hidden_size * 4, scaled.ptr, row_dev.ptr,
                            hidden_size, runtime=runtime)
                # copy row back
                import ctypes as _ct
                runtime.memcpy(selected_out_dev.ptr + t * hidden_size * 4, row_dev.ptr,
                               hidden_size * 4, 0)  # 0 = hipMemcpyDeviceToDevice

        # shared expert
        sg_dev = malloc(shared_gate_q.nbytes, runtime=runtime); buffers.append(sg_dev)
        su_dev = malloc(shared_up_q.nbytes, runtime=runtime); buffers.append(su_dev)
        sd_dev = malloc(shared_down_q.nbytes, runtime=runtime); buffers.append(sd_dev)
        copy_host_to_device(sg_dev, host_array_ptr(shared_gate_q), runtime=runtime)
        copy_host_to_device(su_dev, host_array_ptr(shared_up_q), runtime=runtime)
        copy_host_to_device(sd_dev, host_array_ptr(shared_down_q), runtime=runtime)
        s_gate = malloc(tokens * inter_dim * 4, runtime=runtime); buffers.append(s_gate)
        s_up = malloc(tokens * inter_dim * 4, runtime=runtime); buffers.append(s_up)
        s_inter = malloc(tokens * inter_dim * 4, runtime=runtime); buffers.append(s_inter)
        s_out = malloc(tokens * hidden_size * 4, runtime=runtime); buffers.append(s_out)
        mtp_linear_f32(normed_dev.ptr, sg_dev.ptr, s_gate.ptr, tokens, hidden_size, inter_dim,
                       runtime=runtime)
        mtp_linear_f32(normed_dev.ptr, su_dev.ptr, s_up.ptr, tokens, hidden_size, inter_dim,
                       runtime=runtime)
        mtp_silu_mul_f32(s_gate.ptr, s_up.ptr, s_inter.ptr, tokens * inter_dim, runtime=runtime)
        mtp_linear_f32(s_inter.ptr, sd_dev.ptr, s_out.ptr, tokens, inter_dim, hidden_size,
                       runtime=runtime)

        # shared_gate_logit = normed @ gate_vec  -> [tokens, 1]
        gv_dev = malloc(gate_vec.nbytes, runtime=runtime); buffers.append(gv_dev)
        sgl_dev = malloc(tokens * 1 * 4, runtime=runtime); buffers.append(sgl_dev)
        copy_host_to_device(gv_dev, host_array_ptr(gate_vec), runtime=runtime)
        mtp_linear_f32(normed_dev.ptr, gv_dev.ptr, sgl_dev.ptr, tokens, hidden_size, 1,
                       runtime=runtime)
        sgl_host = np.empty((tokens, 1), dtype=np.float32)
        copy_device_to_host(host_array_ptr(sgl_host), sgl_dev, runtime=runtime)
        sigmoid_vec = (1.0 / (1.0 + np.exp(-sgl_host))).astype(np.float32).reshape(tokens)
        sig_dev = malloc(tokens * 4, runtime=runtime); buffers.append(sig_dev)
        copy_host_to_device(sig_dev, host_array_ptr(sigmoid_vec), runtime=runtime)
        gated_shared = malloc(tokens * hidden_size * 4, runtime=runtime)
        buffers.append(gated_shared)
        mtp_row_scale_f32(sig_dev.ptr, s_out.ptr, gated_shared.ptr, tokens, hidden_size,
                          runtime=runtime)

        # out = hidden + selected_out + gated_shared
        tmp = malloc(tokens * hidden_size * 4, runtime=runtime); buffers.append(tmp)
        out_dev = malloc(tokens * hidden_size * 4, runtime=runtime); buffers.append(out_dev)
        mtp_add_f32(x_dev.ptr, selected_out_dev.ptr, tmp.ptr, tokens * hidden_size,
                    runtime=runtime)
        mtp_add_f32(tmp.ptr, gated_shared.ptr, out_dev.ptr, tokens * hidden_size,
                    runtime=runtime)
        runtime.device_synchronize()
        out = np.empty((tokens, hidden_size), dtype=np.float32)
        copy_device_to_host(host_array_ptr(out), out_dev, runtime=runtime)
        return out
    finally:
        for buf in buffers:
            free(buf, runtime=runtime)


def register_mtp_nextn_kernels(*, replace: bool = True) -> None:
    for backend in ("hip_gfx1100", "hip_gfx1151"):
        register(
            KernelKey(backend, "mtp_nextn_eh_proj", "gguf_f32", "qwen35"),
            qwen35_gguf_mtp_eh_proj_f32,
            replace=replace,
        )
        register(
            KernelKey(backend, "mtp_nextn_attention", "gguf_f32", "qwen35_dense"),
            qwen35_gguf_mtp_attention_sublayer_f32,
            replace=replace,
        )
        register(
            KernelKey(backend, "mtp_nextn_moe_routing", "gguf_f32", "qwen35"),
            qwen35_gguf_mtp_moe_routing_f32,
            replace=replace,
        )
        register(
            KernelKey(backend, "mtp_nextn_ffn", "gguf_f32", "qwen35"),
            qwen35_gguf_mtp_ffn_sublayer_f32,
            replace=replace,
        )


register_mtp_nextn_kernels()


__all__ = [
    "build_mtp_nextn",
    "mtp_add_f32",
    "mtp_dense_attn_f32",
    "mtp_eh_proj_f32",
    "mtp_linear_f32",
    "mtp_mul_f32",
    "mtp_rmsnorm_f32",
    "mtp_row_scale_f32",
    "mtp_scale_f32",
    "mtp_sigmoid_gate_mul_f32",
    "mtp_silu_mul_f32",
    "plan_mtp_nextn_build",
    "qwen35_gguf_mtp_attention_sublayer_f32",
    "qwen35_gguf_mtp_eh_proj_f32",
    "qwen35_gguf_mtp_ffn_sublayer_f32",
    "qwen35_gguf_mtp_moe_routing_f32",
    "register_mtp_nextn_kernels",
]
