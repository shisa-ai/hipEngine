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

# ─── M6 weight cache: avoid re-uploading weights per step ───
_WEIGHT_CACHE: dict[str, "DeviceBuffer"] = {}

def _cached_upload(name: str, data: "np.ndarray", *, runtime=None) -> "DeviceBuffer":
    """Upload weight to device, caching the buffer for reuse.

    First call: allocates + uploads. Subsequent calls: reuses cached buffer.
    The cache is keyed by the weight name (e.g. 'blk.40.attn_q.weight').
    """
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import malloc, copy_host_to_device, host_array_ptr
    runtime = runtime or get_hip_runtime()
    if name in _WEIGHT_CACHE:
        return _WEIGHT_CACHE[name]
    buf = malloc(data.nbytes, runtime=runtime)
    copy_host_to_device(buf, host_array_ptr(np.ascontiguousarray(data)), runtime=runtime)
    _WEIGHT_CACHE[name] = buf
    return buf

def clear_weight_cache():
    """Free all cached weight buffers."""
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import free as hip_free
    runtime = get_hip_runtime()
    for buf in _WEIGHT_CACHE.values():
        hip_free(buf, runtime=runtime)
    _WEIGHT_CACHE.clear()



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


# ptr(x) + ptr(cos) + ptr(sin) + ptr(out) + tokens + heads + head_dim + rotary_dim + cos_stride + stream
_ARGTYPES_ROPE_F32 = (
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
    ctypes.c_void_p,
)


def mtp_rope_f32(
    x_ptr: int,
    cos_ptr: int,
    sin_ptr: int,
    out_ptr: int,
    tokens: int,
    heads: int,
    head_dim: int,
    rotary_dim: int,
    cos_stride: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch split-half RoPE: [x1,x2] -> [x1*cos-x2*sin, x1*sin+x2*cos]."""

    for name, value in (("tokens", tokens), ("heads", heads), ("head_dim", head_dim),
                        ("rotary_dim", rotary_dim)):
        _check_positive(name, value)
    library = library or build_mtp_nextn(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(library, "hipengine_mtp_rope_f32", _ARGTYPES_ROPE_F32, ctypes.c_int)
    err = fn(x_ptr, cos_ptr, sin_ptr, out_ptr, tokens, heads, head_dim, rotary_dim,
             cos_stride, stream)
    _check_launch(runtime, err)


# ptr(query) + ptr(key_cache) + ptr(value_cache) + ptr(block_tables) + ptr(live_counts) +
# ptr(token_positions) + ptr(out) + tokens + heads + kv_heads + qk_head_dim + value_head_dim +
# block_size + max_blocks + scale(float) + stream
_ARGTYPES_PAGED_ATTN_F32 = (
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
    ctypes.c_int64, ctypes.c_int64,
    ctypes.c_float,
    ctypes.c_void_p,
)


def mtp_paged_attn_f32(
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    block_tables_ptr: int,
    live_counts_ptr: int,
    token_positions_ptr: int,
    out_ptr: int,
    tokens: int,
    heads: int,
    kv_heads: int,
    qk_head_dim: int,
    value_head_dim: int,
    block_size: int,
    max_blocks: int,
    scale: float,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch paged KVLiveSpans causal GQA attention (F32, correctness-first)."""

    for name, value in (("tokens", tokens), ("heads", heads), ("kv_heads", kv_heads),
                        ("qk_head_dim", qk_head_dim), ("value_head_dim", value_head_dim),
                        ("block_size", block_size), ("max_blocks", max_blocks)):
        _check_positive(name, value)
    library = library or build_mtp_nextn(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(library, "hipengine_mtp_paged_attn_f32", _ARGTYPES_PAGED_ATTN_F32, ctypes.c_int)
    err = fn(query_ptr, key_cache_ptr, value_cache_ptr, block_tables_ptr, live_counts_ptr,
             token_positions_ptr, out_ptr, tokens, heads, kv_heads, qk_head_dim, value_head_dim,
             block_size, max_blocks, float(scale), stream)
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
    *,
    eh_proj_qtype: "GGMLQuantizationType | None" = None,
    eps: float = 1e-6,
) -> np.ndarray:
    """Numpy-in/out wrapper matching ``cpu_reference.qwen35_gguf_mtp_eh_proj``.

    M6: supports Q8_0 for eh_proj weight. When qtype is None/F32, uses fused F32
    GEMV; when Q8_0, splits to concat + q8_0_gemv.
    """

    hidden_arr = np.ascontiguousarray(hidden_seed, dtype=np.float32)
    embed_arr = np.ascontiguousarray(token_embedding, dtype=np.float32)
    weight = np.ascontiguousarray(eh_proj_weight)  # keep raw bytes for K-quant
    hnorm = np.ascontiguousarray(hnorm_weight, dtype=np.float32)
    enorm = np.ascontiguousarray(enorm_weight, dtype=np.float32)
    if hidden_arr.ndim != 2:
        raise ValueError("hidden_seed must have shape [rows, hidden]")
    rows, hidden = hidden_arr.shape
    if embed_arr.shape != hidden_arr.shape:
        raise ValueError("token_embedding must match hidden_seed shape")
    from hipengine.quant.gguf import GGMLQuantizationType
    if eh_proj_qtype is None or eh_proj_qtype == GGMLQuantizationType.F32:
        if weight.shape != (hidden, hidden * 2):
            raise ValueError(
                f"eh_proj_weight must have shape [hidden, 2*hidden]=[{hidden}, {hidden * 2}]; "
                f"got {weight.shape}"
            )
    # K-quant: weight is [hidden, block_bytes] (block-padded)
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
        if eh_proj_qtype is None or eh_proj_qtype == GGMLQuantizationType.F32:
            mtp_eh_proj_f32(e_norm_dev.ptr, h_norm_dev.ptr, weight_dev.ptr, out_dev.ptr,
                            rows, hidden, runtime=runtime)
        else:
            e_norm_host = np.empty((rows, hidden), dtype=np.float32)
            h_norm_host = np.empty((rows, hidden), dtype=np.float32)
            copy_device_to_host(host_array_ptr(e_norm_host), e_norm_dev, runtime=runtime)
            copy_device_to_host(host_array_ptr(h_norm_host), h_norm_dev, runtime=runtime)
            concat = np.ascontiguousarray(np.concatenate([e_norm_host, h_norm_host], axis=1),
                                          dtype=np.float32)
            concat_dev = malloc(concat.nbytes, runtime=runtime); buffers.append(concat_dev)
            copy_host_to_device(concat_dev, host_array_ptr(concat), runtime=runtime)
            if eh_proj_qtype == GGMLQuantizationType.Q8_0:
                from hipengine.kernels.hip_gfx1100.quant.gguf_k_gemv import (
                    gguf_q8_0_gemv_f32_f32_out as _hip_q8_0,
                )
                _hip_q8_0(concat_dev.ptr, weight_dev.ptr, out_dev.ptr, rows, hidden * 2,
                          hidden, runtime=runtime)
            elif eh_proj_qtype == GGMLQuantizationType.Q6_K:
                from hipengine.kernels.hip_gfx1100.quant.gguf_q6_k_pack8_gemv import (
                    gguf_q6_k_pack8_gemv_decode_fp16_f32_out as _hip_q6_k_eh,
                )
                concat_fp16 = concat.astype(np.float16)
                concat_fp16_dev = malloc(concat_fp16.nbytes, runtime=runtime)
                copy_host_to_device(concat_fp16_dev, host_array_ptr(concat_fp16), runtime=runtime)
                _hip_q6_k_eh(concat_fp16_dev.ptr, weight_dev.ptr, out_dev.ptr, rows,
                             hidden * 2, hidden, runtime=runtime)
                free(concat_fp16_dev, runtime=runtime)
            else:
                raise NotImplementedError(f"eh_proj_qtype={eh_proj_qtype.name} not supported")
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
    wq_qtype: "GGMLQuantizationType | None" = None,
    wk_qtype: "GGMLQuantizationType | None" = None,
    wv_qtype: "GGMLQuantizationType | None" = None,
    wo_qtype: "GGMLQuantizationType | None" = None,
    positions: "np.ndarray | None" = None,
    context_counts: "np.ndarray | None" = None,
    key_cache: "np.ndarray | None" = None,
    value_cache: "np.ndarray | None" = None,
    kv_base_offsets: "np.ndarray | None" = None,
    kv_live_counts: "np.ndarray | None" = None,
    kv_token_positions: "np.ndarray | None" = None,
    kv_evict_mask: "np.ndarray | None" = None,
    block_size: int | None = None,
    rope_cos: "np.ndarray | None" = None,
    rope_sin: "np.ndarray | None" = None,
    rotary_dim: int | None = None,
    scale: float | None = None,
    eps: float = 1e-6,
) -> np.ndarray:
    """Numpy-in/out wrapper matching ``cpu_reference.qwen35_gguf_mtp_attention_sublayer``.

    M6: supports K-quant (Q4_K/Q5_K/Q8_0) for wq/wk/wv/wo via _attn_dispatch_gemv.
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

    from hipengine.quant.gguf import GGMLQuantizationType
    wq = np.ascontiguousarray(wq_weight)
    wk = np.ascontiguousarray(wk_weight)
    wv = np.ascontiguousarray(wv_weight)
    wo = np.ascontiguousarray(wo_weight)
    # For F32, validate exact shapes; for K-quant, shapes are block-padded
    if (wq_qtype is None or wq_qtype == GGMLQuantizationType.F32):
        if wq.shape != (heads * 2 * qk_head_dim, hidden_size):
            raise ValueError("wq_weight must have shape [num_heads * 2 * qk_head_dim, hidden]")
    if (wk_qtype is None or wk_qtype == GGMLQuantizationType.F32):
        if wk.shape != (kv_heads * qk_head_dim, hidden_size):
            raise ValueError("wk_weight must have shape [num_kv_heads * qk_head_dim, hidden]")
    if (wv_qtype is None or wv_qtype == GGMLQuantizationType.F32):
        if wv.ndim != 2 or wv.shape[1] != hidden_size or wv.shape[0] % kv_heads != 0:
            raise ValueError("wv_weight must have shape [num_kv_heads * value_head_dim, hidden]")
        value_head_dim = wv.shape[0] // kv_heads
        if value_head_dim != qk_head_dim:
            raise ValueError("value_head_dim must match qk_head_dim for gated Qwen35 attention")
    else:
        value_head_dim = qk_head_dim  # Q8_0: inferred from model config
    if (wo_qtype is None or wo_qtype == GGMLQuantizationType.F32):
        if wo.shape != (hidden_size, heads * value_head_dim):
            raise ValueError("wo_weight must have shape [hidden, num_heads * value_head_dim]")

    from hipengine.quant.gguf import GGMLQuantizationType

    def _attn_dispatch_gemv(x_dev, weight_dev, out_dev, rows, in_features, out_features,
                            qtype, *, runtime):
        if qtype is None or qtype == GGMLQuantizationType.F32:
            mtp_linear_f32(x_dev.ptr, weight_dev.ptr, out_dev.ptr, rows, in_features,
                           out_features, runtime=runtime)
        elif qtype == GGMLQuantizationType.Q8_0:
            from hipengine.kernels.hip_gfx1100.quant.gguf_k_gemv import (
                gguf_q8_0_gemv_f32_f32_out as _hip_q8_0,
            )
            _hip_q8_0(x_dev.ptr, weight_dev.ptr, out_dev.ptr, rows, in_features,
                      out_features, runtime=runtime)
        elif qtype == GGMLQuantizationType.Q4_K:
            from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_gemv import (
                gguf_q4_k_gemv_f32_f32_out as _hip_q4_k,
            )
            _hip_q4_k(x_dev.ptr, weight_dev.ptr, out_dev.ptr, rows, in_features,
                      out_features, runtime=runtime)
        elif qtype == GGMLQuantizationType.Q5_K:
            from hipengine.kernels.hip_gfx1100.quant.gguf_k_gemv import (
                gguf_q5_k_gemv_f32_f32_out as _hip_q5_k,
            )
            _hip_q5_k(x_dev.ptr, weight_dev.ptr, out_dev.ptr, rows, in_features,
                      out_features, runtime=runtime)
        elif qtype == GGMLQuantizationType.Q6_K:
            from hipengine.kernels.hip_gfx1100.quant.gguf_q6_k_pack8_gemv import (
                gguf_q6_k_pack8_gemv_decode_fp16_f32_out as _hip_q6_k_attn,
            )
            x_f32 = np.empty((rows, in_features), dtype=np.float32)
            copy_device_to_host(host_array_ptr(x_f32), x_dev, x_f32.nbytes, runtime=runtime)
            x_fp16 = x_f32.astype(np.float16)
            x_fp16_dev = malloc(x_fp16.nbytes, runtime=runtime)
            copy_host_to_device(x_fp16_dev, host_array_ptr(x_fp16), runtime=runtime)
            _hip_q6_k_attn(x_fp16_dev.ptr, weight_dev.ptr, out_dev.ptr, rows, in_features,
                           out_features, runtime=runtime)
            free(x_fp16_dev, runtime=runtime)
        else:
            raise NotImplementedError(f"qtype={qtype.name} not supported for attention")

    # RoPE: apply if cos/sin provided
    apply_rope = rope_cos is not None and rope_sin is not None
    if (rope_cos is None) != (rope_sin is None):
        raise ValueError("rope_cos and rope_sin must be provided together")
    rot_dim = qk_head_dim if rotary_dim is None else int(rotary_dim)
    if apply_rope:
        if rot_dim % 2 != 0:
            raise ValueError("rotary_dim must be even")
        half = rot_dim // 2
        cos_arr = np.ascontiguousarray(rope_cos, dtype=np.float32)
        sin_arr = np.ascontiguousarray(rope_sin, dtype=np.float32)
        if cos_arr.shape[-1] == half * 2:
            cos_arr = np.ascontiguousarray(cos_arr[..., :half])
        elif cos_arr.shape[-1] != half:
            raise ValueError(f"rope_cos.shape[-1] must be {half} or {half*2}")
        if sin_arr.shape[-1] == half * 2:
            sin_arr = np.ascontiguousarray(sin_arr[..., :half])
        elif sin_arr.shape[-1] != half:
            raise ValueError(f"rope_sin.shape[-1] must be {half} or {half*2}")

    # Paged KVLiveSpans path: if kv_base_offsets provided, use paged attention
    use_paged = kv_base_offsets is not None
    if use_paged:
        if key_cache is None or value_cache is None:
            raise ValueError("paged KVLiveSpans attention requires key_cache and value_cache")
        if kv_live_counts is None:
            raise ValueError("kv_live_counts is required with kv_base_offsets")
        # Normalize block_tables
        bt = np.ascontiguousarray(kv_base_offsets, dtype=np.int64)
        if bt.ndim == 1:
            bt = bt[None, :]
        max_blocks = bt.shape[1]
        lc = np.ascontiguousarray(kv_live_counts, dtype=np.int64)
        tp = np.ascontiguousarray(
            positions if kv_token_positions is None else kv_token_positions, dtype=np.int64
        )
        blk = int(block_size) if block_size is not None else int(key_cache.shape[1])
    elif key_cache is not None or value_cache is not None:
        # Dense external cache (not paged) — still not supported for non-paged external cache
        raise NotImplementedError(
            "non-paged external KV cache is M6+ work; use kv_base_offsets for paged KVLiveSpans"
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
        _attn_dispatch_gemv(normed_dev, wq_dev, q_full_dev, tokens, hidden_size,
                            heads * 2 * qk_head_dim, wq_qtype, runtime=runtime)
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
        _attn_dispatch_gemv(normed_dev, wk_dev, key_cur_dev, tokens, hidden_size,
                            kv_heads * qk_head_dim, wk_qtype, runtime=runtime)
        mtp_rmsnorm_f32(key_cur_dev.ptr, k_norm_dev.ptr, key_cur_dev.ptr, tokens * kv_heads,
                        qk_head_dim, eps=eps, runtime=runtime)

        # Apply RoPE to query and key_cur if cos/sin provided
        if apply_rope:
            cos_dev = malloc(cos_arr.nbytes, runtime=runtime); buffers.append(cos_dev)
            sin_dev = malloc(sin_arr.nbytes, runtime=runtime); buffers.append(sin_dev)
            copy_host_to_device(cos_dev, host_array_ptr(cos_arr), runtime=runtime)
            copy_host_to_device(sin_dev, host_array_ptr(sin_arr), runtime=runtime)
            mtp_rope_f32(query_dev.ptr, cos_dev.ptr, sin_dev.ptr, query_dev.ptr,
                         tokens, heads, qk_head_dim, rot_dim, half, runtime=runtime)
            mtp_rope_f32(key_cur_dev.ptr, cos_dev.ptr, sin_dev.ptr, key_cur_dev.ptr,
                         tokens, kv_heads, qk_head_dim, rot_dim, half, runtime=runtime)

        # value_cur = normed @ wv.T  -> [tokens, kv_heads, value_head_dim]
        value_cur_dev = malloc(tokens * kv_heads * value_head_dim * 4, runtime=runtime)
        buffers.append(value_cur_dev)
        wv_dev = malloc(wv.nbytes, runtime=runtime); buffers.append(wv_dev)
        copy_host_to_device(wv_dev, host_array_ptr(wv), runtime=runtime)
        _attn_dispatch_gemv(normed_dev, wv_dev, value_cur_dev, tokens, hidden_size,
                            kv_heads * value_head_dim, wv_qtype, runtime=runtime)

        # dense causal GQA: cache = current tokens (default path), cache_tokens = tokens
        pos_dev = malloc(pos.nbytes, runtime=runtime); buffers.append(pos_dev)
        ctx_dev = malloc(ctx.nbytes, runtime=runtime); buffers.append(ctx_dev)
        copy_host_to_device(pos_dev, host_array_ptr(pos), runtime=runtime)
        copy_host_to_device(ctx_dev, host_array_ptr(ctx), runtime=runtime)
        attn_dev = malloc(tokens * heads * value_head_dim * 4, runtime=runtime)
        buffers.append(attn_dev)
        if use_paged:
            kc_dev = malloc(key_cache.nbytes, runtime=runtime); buffers.append(kc_dev)
            vc_dev = malloc(value_cache.nbytes, runtime=runtime); buffers.append(vc_dev)
            bt_dev = malloc(bt.nbytes, runtime=runtime); buffers.append(bt_dev)
            lc_dev = malloc(lc.nbytes, runtime=runtime); buffers.append(lc_dev)
            tp_dev = malloc(tp.nbytes, runtime=runtime); buffers.append(tp_dev)
            copy_host_to_device(kc_dev, host_array_ptr(key_cache), runtime=runtime)
            copy_host_to_device(vc_dev, host_array_ptr(value_cache), runtime=runtime)
            copy_host_to_device(bt_dev, host_array_ptr(bt), runtime=runtime)
            copy_host_to_device(lc_dev, host_array_ptr(lc), runtime=runtime)
            copy_host_to_device(tp_dev, host_array_ptr(tp), runtime=runtime)
            mtp_paged_attn_f32(
                query_dev.ptr, kc_dev.ptr, vc_dev.ptr, bt_dev.ptr, lc_dev.ptr, tp_dev.ptr,
                attn_dev.ptr, tokens, heads, kv_heads, qk_head_dim, value_head_dim,
                blk, max_blocks, scale_value, runtime=runtime,
            )
        else:
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
        _attn_dispatch_gemv(gated_dev, wo_dev, wo_out_dev, tokens, heads * value_head_dim,
                            hidden_size, wo_qtype, runtime=runtime)

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
    router = np.ascontiguousarray(router_weight)
    # BF16 router weight: dequant to F32 on host (real model has BF16 router)
    if router.dtype != np.float32:
        if router.dtype == np.uint16 or router.dtype == np.int16:
            # BF16 → F32: place uint16 bits in upper half of float32
            router_f32 = np.zeros(router.shape, dtype=np.float32)
            router_f32.view(np.uint32)[:] = router.astype(np.uint32) << 16
            router = router_f32
        else:
            router = router.astype(np.float32)
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

    def _dispatch_gemv(x_dev, weight_dev, out_dev, rows, in_features, out_features,
                       qtype, *, runtime):
        if qtype == GGMLQuantizationType.F32:
            mtp_linear_f32(x_dev.ptr, weight_dev.ptr, out_dev.ptr, rows, in_features,
                           out_features, runtime=runtime)
        elif qtype == GGMLQuantizationType.Q4_K:
            from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_gemv import (
                gguf_q4_k_gemv_f32_f32_out as _hip_q4_k,
            )
            _hip_q4_k(x_dev.ptr, weight_dev.ptr, out_dev.ptr, rows, in_features,
                      out_features, runtime=runtime)
        elif qtype == GGMLQuantizationType.Q5_K:
            from hipengine.kernels.hip_gfx1100.quant.gguf_k_gemv import (
                gguf_q5_k_gemv_f32_f32_out as _hip_q5_k,
            )
            _hip_q5_k(x_dev.ptr, weight_dev.ptr, out_dev.ptr, rows, in_features,
                      out_features, runtime=runtime)
        elif qtype == GGMLQuantizationType.Q8_0:
            from hipengine.kernels.hip_gfx1100.quant.gguf_k_gemv import (
                gguf_q8_0_gemv_f32_f32_out as _hip_q8_0,
            )
            _hip_q8_0(x_dev.ptr, weight_dev.ptr, out_dev.ptr, rows, in_features,
                      out_features, runtime=runtime)
        elif qtype == GGMLQuantizationType.Q6_K:
            from hipengine.kernels.hip_gfx1100.quant.gguf_q6_k_pack8_gemv import (
                gguf_q6_k_pack8_gemv_decode_fp16_f32_out as _hip_q6_k,
            )
            # Convert f32 x → fp16 on host (GPU cast unreliable for this use case)
            x_f32 = np.empty((rows, in_features), dtype=np.float32)
            copy_device_to_host(host_array_ptr(x_f32), x_dev, x_f32.nbytes, runtime=runtime)
            x_fp16 = x_f32.astype(np.float16)
            x_fp16_dev = malloc(x_fp16.nbytes, runtime=runtime)
            copy_host_to_device(x_fp16_dev, host_array_ptr(x_fp16), runtime=runtime)
            _hip_q6_k(x_fp16_dev.ptr, weight_dev.ptr, out_dev.ptr, rows, in_features,
                      out_features, runtime=runtime)
            free(x_fp16_dev, runtime=runtime)
        else:
            raise NotImplementedError(
                f"qtype={qtype.name} not supported (F32/Q4_K/Q5_K/Q6_K/Q8_0 only)"
            )

    for qt, name in (
        (gate_qtype, "gate_qtype"), (up_qtype, "up_qtype"), (down_qtype, "down_qtype"),
        (shared_qtype, "shared_qtype"),
    ):
        if qt not in (GGMLQuantizationType.F32, GGMLQuantizationType.Q4_K,
                      GGMLQuantizationType.Q5_K, GGMLQuantizationType.Q8_0,
                      GGMLQuantizationType.Q6_K):
            raise NotImplementedError(
                f"{name}={qt.name} not supported (F32/Q4_K/Q5_K/Q6_K/Q8_0 only)"
            )

    x = np.ascontiguousarray(hidden, dtype=np.float32)
    norm_weight = np.ascontiguousarray(attn_post_norm_weight, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError("hidden must have shape [tokens, hidden]")
    tokens, hidden_size = x.shape
    if norm_weight.shape != (hidden_size,):
        raise ValueError("attn_post_norm_weight must have shape [hidden]")

    gate_q = np.ascontiguousarray(gate_qweight)
    up_q = np.ascontiguousarray(up_qweight)
    down_q = np.ascontiguousarray(down_qweight)
    if gate_q.ndim != 3 or up_q.ndim != 3 or down_q.ndim != 3:
        raise ValueError("expert weights must be rank-3 [E, out, in/block_bytes]")
    num_experts = gate_q.shape[0]
    inter_dim = gate_q.shape[1]
    shared_gate_q = np.ascontiguousarray(shared_gate_qweight)
    shared_up_q = np.ascontiguousarray(shared_up_qweight)
    shared_down_q = np.ascontiguousarray(shared_down_qweight)
    gate_vec = np.ascontiguousarray(shared_gate_logit_weight)
    # BF16 shared_gate_logit: dequant to F32 (real model has BF16)
    if gate_vec.dtype != np.float32:
        if gate_vec.dtype == np.uint16 or gate_vec.dtype == np.int16:
            gv_f32 = np.zeros(gate_vec.shape, dtype=np.float32)
            gv_f32.view(np.uint32)[:] = gate_vec.astype(np.uint32) << 16
            gate_vec = gv_f32
        else:
            gate_vec = gate_vec.astype(np.float32)

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
                _dispatch_gemv(xt_dev, g_dev, gate_out, 1, hidden_size, inter_dim,
                               gate_qtype, runtime=runtime)
                _dispatch_gemv(xt_dev, u_dev, up_out, 1, hidden_size, inter_dim,
                               up_qtype, runtime=runtime)
                mtp_silu_mul_f32(gate_out.ptr, up_out.ptr, inter_out.ptr, inter_dim,
                                 runtime=runtime)
                _dispatch_gemv(inter_out, d_dev, down_out, 1, inter_dim, hidden_size,
                               down_qtype, runtime=runtime)
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
        _dispatch_gemv(normed_dev, sg_dev, s_gate, tokens, hidden_size, inter_dim,
                       shared_qtype, runtime=runtime)
        _dispatch_gemv(normed_dev, su_dev, s_up, tokens, hidden_size, inter_dim,
                       shared_qtype, runtime=runtime)
        mtp_silu_mul_f32(s_gate.ptr, s_up.ptr, s_inter.ptr, tokens * inter_dim, runtime=runtime)
        _dispatch_gemv(s_inter, sd_dev, s_out, tokens, inter_dim, hidden_size,
                       shared_qtype, runtime=runtime)

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


def qwen35_gguf_mtp_shared_head_logits_f32(
    nextn_hidden: "np.ndarray",
    shared_head_norm_weight: "np.ndarray",
    shared_head_weight: "np.ndarray",
    *,
    eps: float = 1e-6,
    shared_head_qtype: "GGMLQuantizationType | None" = None,
) -> np.ndarray:
    """Numpy-in/out wrapper matching ``cpu_reference.qwen35_gguf_mtp_shared_head_logits``.

    RMSNorm (GPU) + LM-head linear ``normed @ head_weight.T`` (GPU).
    """

    from hipengine.quant.gguf import GGMLQuantizationType
    hidden = np.ascontiguousarray(nextn_hidden, dtype=np.float32)
    norm_weight = np.ascontiguousarray(shared_head_norm_weight, dtype=np.float32)
    head_weight = np.ascontiguousarray(shared_head_weight)
    if hidden.ndim != 2:
        raise ValueError("nextn_hidden must have shape [rows, hidden]")
    rows, hidden_size = hidden.shape
    if norm_weight.shape != (hidden_size,):
        raise ValueError("shared_head_norm_weight must have shape [hidden]")
    if shared_head_qtype is not None and shared_head_qtype != GGMLQuantizationType.F32:
        # K-quant: raw block bytes, shape = [vocab, block_bytes_per_row]
        if head_weight.ndim != 2:
            raise ValueError("shared_head_weight must be 2D [vocab, block_bytes]")
        vocab = head_weight.shape[0]
    else:
        head_weight = head_weight.astype(np.float32)
        if head_weight.ndim != 2 or head_weight.shape[1] != hidden_size:
            raise ValueError("shared_head_weight must have shape [vocab, hidden]")
        vocab = head_weight.shape[0]

    runtime = get_hip_runtime()
    buffers: list = []
    try:
        hidden_dev = malloc(hidden.nbytes, runtime=runtime); buffers.append(hidden_dev)
        norm_dev = malloc(norm_weight.nbytes, runtime=runtime); buffers.append(norm_dev)
        normed_dev = malloc(hidden.nbytes, runtime=runtime); buffers.append(normed_dev)
        head_dev = malloc(head_weight.nbytes, runtime=runtime); buffers.append(head_dev)
        out_dev = malloc(rows * vocab * 4, runtime=runtime); buffers.append(out_dev)
        copy_host_to_device(hidden_dev, host_array_ptr(hidden), runtime=runtime)
        copy_host_to_device(norm_dev, host_array_ptr(norm_weight), runtime=runtime)
        copy_host_to_device(head_dev, host_array_ptr(head_weight), runtime=runtime)
        mtp_rmsnorm_f32(hidden_dev.ptr, norm_dev.ptr, normed_dev.ptr, rows, hidden_size, eps=eps,
                        runtime=runtime)
        if shared_head_qtype is not None and shared_head_qtype == GGMLQuantizationType.Q6_K:
            # Q6_K pack8 gemv uses fp16 input which loses too much precision
            # for the 248320-vocab shared_head. Use host-side dequant + F32 gemv.
            from hipengine.quant.gguf import dequantize_gguf_data
            head_f32 = dequantize_gguf_data(
                np.ascontiguousarray(shared_head_weight),
                GGMLQuantizationType.Q6_K,
            ).astype(np.float32)
            head_dev = malloc(head_f32.nbytes, runtime=runtime); buffers.append(head_dev)
            copy_host_to_device(head_dev, host_array_ptr(head_f32), runtime=runtime)
            mtp_linear_f32(normed_dev.ptr, head_dev.ptr, out_dev.ptr, rows, hidden_size, vocab,
                           runtime=runtime)
        elif shared_head_qtype is not None and shared_head_qtype == GGMLQuantizationType.Q8_0:
            from hipengine.kernels.hip_gfx1100.quant.gguf_k_gemv import (
                gguf_q8_0_gemv_f32_f32_out as _hip_q8_0_head,
            )
            _hip_q8_0_head(normed_dev.ptr, head_dev.ptr, out_dev.ptr, rows, hidden_size,
                           vocab, runtime=runtime)
        else:
            mtp_linear_f32(normed_dev.ptr, head_dev.ptr, out_dev.ptr, rows, hidden_size, vocab,
                           runtime=runtime)
        runtime.device_synchronize()
        out = np.empty((rows, vocab), dtype=np.float32)
        copy_device_to_host(host_array_ptr(out), out_dev, runtime=runtime)
        return out
    finally:
        for buf in buffers:
            free(buf, runtime=runtime)


def qwen35_gguf_mtp_nextn_layer_logits_f32(
    hidden_seed: "np.ndarray",
    token_embedding: "np.ndarray",
    eh_proj_weight: "np.ndarray",
    hnorm_weight: "np.ndarray",
    enorm_weight: "np.ndarray",
    attn_norm_weight: "np.ndarray",
    wq_weight: "np.ndarray",
    wk_weight: "np.ndarray",
    wv_weight: "np.ndarray",
    wo_weight: "np.ndarray",
    q_norm_weight: "np.ndarray",
    k_norm_weight: "np.ndarray",
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
    shared_head_norm_weight: "np.ndarray",
    shared_head_weight: "np.ndarray",
    *,
    num_heads: int,
    num_kv_heads: int,
    experts_used: int,
    eh_proj_qtype: "GGMLQuantizationType | None" = None,
    wq_qtype: "GGMLQuantizationType | None" = None,
    wk_qtype: "GGMLQuantizationType | None" = None,
    wv_qtype: "GGMLQuantizationType | None" = None,
    wo_qtype: "GGMLQuantizationType | None" = None,
    positions: "np.ndarray | None" = None,
    context_counts: "np.ndarray | None" = None,
    key_cache: "np.ndarray | None" = None,
    value_cache: "np.ndarray | None" = None,
    kv_base_offsets: "np.ndarray | None" = None,
    kv_live_counts: "np.ndarray | None" = None,
    kv_token_positions: "np.ndarray | None" = None,
    kv_evict_mask: "np.ndarray | None" = None,
    block_size: int | None = None,
    rope_cos: "np.ndarray | None" = None,
    rope_sin: "np.ndarray | None" = None,
    rotary_dim: int | None = None,
    scale: float | None = None,
    expert_weights_scale: float = 1.0,
    shared_head_qtype: "GGMLQuantizationType | None" = None,
    eps: float = 1e-6,
) -> np.ndarray:
    """Native GPU Qwen35 GGUF MTP NextN draft layer (M3, correctness-first).

    Composes the four GPU sub-kernels in the llama.cpp draft-only order:
    ``eh_proj`` -> attention -> ffn -> shared_head.  Signature matches
    ``cpu_reference.qwen35_gguf_mtp_nextn_layer_logits`` exactly so the M3
    fixture gate runs on a real GPU backend instead of the registry's
    cpu_reference fallback.

    M3 scope: F32 qtype, DEFAULT dense attention path (no RoPE, no KVLiveSpans
    paged cache).  K-quant, RoPE and paged-KV branches raise NotImplementedError
    (M6); the F32 M3 fixture does not exercise them.
    """

    projected = qwen35_gguf_mtp_eh_proj_f32(
        hidden_seed, token_embedding, eh_proj_weight, hnorm_weight, enorm_weight,
        eh_proj_qtype=eh_proj_qtype, eps=eps,
    )
    attended = qwen35_gguf_mtp_attention_sublayer_f32(
        projected, attn_norm_weight, wq_weight, wk_weight, wv_weight, wo_weight,
        q_norm_weight, k_norm_weight,
        num_heads=num_heads, num_kv_heads=num_kv_heads,
        wq_qtype=wq_qtype, wk_qtype=wk_qtype, wv_qtype=wv_qtype, wo_qtype=wo_qtype,
        positions=positions, context_counts=context_counts,
        key_cache=key_cache, value_cache=value_cache,
        rope_cos=rope_cos, rope_sin=rope_sin, rotary_dim=rotary_dim, scale=scale, eps=eps,
    )
    ffn_out = qwen35_gguf_mtp_ffn_sublayer_f32(
        attended, attn_post_norm_weight, router_weight,
        gate_qweight, up_qweight, down_qweight, gate_qtype, up_qtype, down_qtype,
        shared_gate_logit_weight, shared_gate_qweight, shared_up_qweight, shared_down_qweight,
        shared_qtype, experts_used=experts_used, expert_weights_scale=expert_weights_scale,
        eps=eps,
    )
    return qwen35_gguf_mtp_shared_head_logits_f32(
        ffn_out, shared_head_norm_weight, shared_head_weight, eps=eps,
        shared_head_qtype=shared_head_qtype,
    )




def qwen35_gguf_mtp_q8_0_gemv_f32(
    x: "np.ndarray",
    qweight: "np.ndarray",
) -> np.ndarray:
    """Numpy-in/out adapter for the existing hip GPU Q8_0 gemv kernel."""

    from hipengine.kernels.hip_gfx1100.quant.gguf_k_gemv import (
        gguf_q8_0_gemv_f32_f32_out as _hip_q8_0,
    )

    x_arr = np.ascontiguousarray(x, dtype=np.float32)
    qw_arr = np.ascontiguousarray(qweight)
    if x_arr.ndim != 2:
        raise ValueError("x must have shape [rows, in_features]")
    rows, in_features = x_arr.shape
    if qw_arr.ndim != 2:
        raise ValueError("qweight must be rank-2 [out_features, block_bytes_per_row]")
    out_features = qw_arr.shape[0]

    runtime = get_hip_runtime()
    buffers: list = []
    try:
        x_dev = malloc(x_arr.nbytes, runtime=runtime); buffers.append(x_dev)
        qw_dev = malloc(qw_arr.nbytes, runtime=runtime); buffers.append(qw_dev)
        out_dev = malloc(rows * out_features * 4, runtime=runtime); buffers.append(out_dev)
        copy_host_to_device(x_dev, host_array_ptr(x_arr), runtime=runtime)
        copy_host_to_device(qw_dev, host_array_ptr(qw_arr), runtime=runtime)
        _hip_q8_0(x_dev.ptr, qw_dev.ptr, out_dev.ptr, rows=rows,
                  in_features=in_features, out_features=out_features)
        runtime.device_synchronize()
        out = np.empty((rows, out_features), dtype=np.float32)
        copy_device_to_host(host_array_ptr(out), out_dev, runtime=runtime)
        return out
    finally:
        for buf in buffers:
            free(buf, runtime=runtime)


def qwen35_gguf_mtp_q4_k_gemv_f32(
    x: "np.ndarray",
    qweight: "np.ndarray",
) -> np.ndarray:
    """Numpy-in/out adapter for the existing hip GPU Q4_K gemv kernel.

    Calls the existing ``gguf_q4_k_gemv_f32_f32_out`` hip kernel with H2D/D2H.
    ``qweight`` is the raw GGUF Q4_K block bytes (not dequanted).  Returns
    ``[rows, out_features]`` F32 matching ``cpu_reference.gguf_q4_k_gemv``.
    """

    from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_gemv import (
        gguf_q4_k_gemv_f32_f32_out as _hip_q4_k_gemv,
    )

    x_arr = np.ascontiguousarray(x, dtype=np.float32)
    qw_arr = np.ascontiguousarray(qweight)
    if x_arr.ndim != 2:
        raise ValueError("x must have shape [rows, in_features]")
    rows, in_features = x_arr.shape
    if qw_arr.ndim != 2:
        raise ValueError("qweight must be rank-2 [out_features, block_bytes_per_row]")
    out_features = qw_arr.shape[0]

    runtime = get_hip_runtime()
    buffers: list = []
    try:
        x_dev = malloc(x_arr.nbytes, runtime=runtime); buffers.append(x_dev)
        qw_dev = malloc(qw_arr.nbytes, runtime=runtime); buffers.append(qw_dev)
        out_dev = malloc(rows * out_features * 4, runtime=runtime); buffers.append(out_dev)
        copy_host_to_device(x_dev, host_array_ptr(x_arr), runtime=runtime)
        copy_host_to_device(qw_dev, host_array_ptr(qw_arr), runtime=runtime)
        _hip_q4_k_gemv(
            x_dev.ptr, qw_dev.ptr, out_dev.ptr,
            rows=rows, in_features=in_features, out_features=out_features,
        )
        runtime.device_synchronize()
        out = np.empty((rows, out_features), dtype=np.float32)
        copy_device_to_host(host_array_ptr(out), out_dev, runtime=runtime)
        return out
    finally:
        for buf in buffers:
            free(buf, runtime=runtime)


def qwen35_gguf_mtp_q5_k_gemv_f32(
    x: "np.ndarray",
    qweight: "np.ndarray",
) -> np.ndarray:
    """Numpy-in/out adapter for the existing hip GPU Q5_K gemv kernel.

    Calls the existing ``gguf_q5_k_gemv_f32_f32_out`` hip kernel with H2D/D2H.
    ``qweight`` is the raw GGUF Q5_K block bytes (not dequanted).  Returns
    ``[rows, out_features]`` F32 matching ``cpu_reference.gguf_q5_k_gemv``.
    """

    from hipengine.kernels.hip_gfx1100.quant.gguf_k_gemv import (
        gguf_q5_k_gemv_f32_f32_out as _hip_q5_k_gemv,
    )

    x_arr = np.ascontiguousarray(x, dtype=np.float32)
    qw_arr = np.ascontiguousarray(qweight)
    if x_arr.ndim != 2:
        raise ValueError("x must have shape [rows, in_features]")
    rows, in_features = x_arr.shape
    if qw_arr.ndim != 2:
        raise ValueError("qweight must be rank-2 [out_features, block_bytes_per_row]")
    out_features = qw_arr.shape[0]

    runtime = get_hip_runtime()
    buffers: list = []
    try:
        x_dev = malloc(x_arr.nbytes, runtime=runtime); buffers.append(x_dev)
        qw_dev = malloc(qw_arr.nbytes, runtime=runtime); buffers.append(qw_dev)
        out_dev = malloc(rows * out_features * 4, runtime=runtime); buffers.append(out_dev)
        copy_host_to_device(x_dev, host_array_ptr(x_arr), runtime=runtime)
        copy_host_to_device(qw_dev, host_array_ptr(qw_arr), runtime=runtime)
        _hip_q5_k_gemv(
            x_dev.ptr, qw_dev.ptr, out_dev.ptr,
            rows=rows, in_features=in_features, out_features=out_features,
        )
        runtime.device_synchronize()
        out = np.empty((rows, out_features), dtype=np.float32)
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
        register(
            KernelKey(backend, "mtp_nextn_shared_head", "gguf_f32", "qwen35_dense_logits"),
            qwen35_gguf_mtp_shared_head_logits_f32,
            replace=replace,
        )
        register(
            KernelKey(backend, "mtp_nextn_layer", "w4_gguf", "qwen35_dense_logits"),
            qwen35_gguf_mtp_nextn_layer_logits_f32,
            replace=replace,
        )
        register(
            KernelKey(backend, "mtp_q8_0_gemv", "gguf_f32", "qwen35"),
            qwen35_gguf_mtp_q8_0_gemv_f32,
            replace=replace,
        )
        register(
            KernelKey(backend, "mtp_q4_k_gemv", "gguf_f32", "qwen35"),
            qwen35_gguf_mtp_q4_k_gemv_f32,
            replace=replace,
        )
        register(
            KernelKey(backend, "mtp_q5_k_gemv", "gguf_f32", "qwen35"),
            qwen35_gguf_mtp_q5_k_gemv_f32,
            replace=replace,
        )


register_mtp_nextn_kernels()


__all__ = [
    "build_mtp_nextn",
    "mtp_add_f32",
    "mtp_dense_attn_f32",
    "mtp_paged_attn_f32",
    "mtp_eh_proj_f32",
    "mtp_linear_f32",
    "mtp_mul_f32",
    "mtp_rope_f32",
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
    "qwen35_gguf_mtp_nextn_layer_logits_f32",
    "qwen35_gguf_mtp_q8_0_gemv_f32",
    "qwen35_gguf_mtp_q4_k_gemv_f32",
    "qwen35_gguf_mtp_q5_k_gemv_f32",
    "qwen35_gguf_mtp_shared_head_logits_f32",
    "register_mtp_nextn_kernels",
]
