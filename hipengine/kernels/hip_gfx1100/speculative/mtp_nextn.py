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


register_mtp_nextn_kernels()


__all__ = [
    "build_mtp_nextn",
    "mtp_add_f32",
    "mtp_dense_attn_f32",
    "mtp_eh_proj_f32",
    "mtp_linear_f32",
    "mtp_rmsnorm_f32",
    "mtp_sigmoid_gate_mul_f32",
    "plan_mtp_nextn_build",
    "qwen35_gguf_mtp_attention_sublayer_f32",
    "qwen35_gguf_mtp_eh_proj_f32",
    "register_mtp_nextn_kernels",
]
