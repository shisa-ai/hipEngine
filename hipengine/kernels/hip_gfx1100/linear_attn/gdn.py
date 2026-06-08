"""Raw-pointer wrappers for Qwen3.5 linear-attention GDN kernels."""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("gdn.hip")
_OUTPUT_NAME = "qwen35_linear_attn_gdn.so"
_SYMBOL_LOWP = "hipengine_qwen35_gdn_recurrent_rmsnorm_gate_lowp_bf16"
_SYMBOL_LOWP_FP16 = "hipengine_qwen35_gdn_recurrent_rmsnorm_gate_lowp_fp16"
_SYMBOL_TREE_TLOOP_BF16 = "hipengine_qwen35_gdn_tree_recurrent_rmsnorm_gate_lowp_tloop_bf16"
_SYMBOL_TREE_TLOOP_FP16 = "hipengine_qwen35_gdn_tree_recurrent_rmsnorm_gate_lowp_tloop_fp16"
_SYMBOL_CHAIN_TLOOP_BF16 = "hipengine_qwen35_gdn_chain_recurrent_rmsnorm_gate_lowp_tloop_bf16"
_SYMBOL_CHAIN_TLOOP_FP16 = "hipengine_qwen35_gdn_chain_recurrent_rmsnorm_gate_lowp_tloop_fp16"
_SYMBOL_PREFILL = "hipengine_qwen35_gdn_prefill_recurrent_f32"
_SYMBOL_PREFILL_K2 = "hipengine_qwen35_gdn_prefill_recurrent_k2_f32"
_SYMBOL_PREFILL_SEGMENTS_K2 = "hipengine_qwen35_gdn_prefill_recurrent_segments_k2_f32"
_SYMBOL_PREFILL_PREPARE = "hipengine_qwen35_linear_attn_prefill_prepare_f32_bf16"
_SYMBOL_PREFILL_PREPARE_FP16 = "hipengine_qwen35_linear_attn_prefill_prepare_f32_fp16"
_SYMBOL_PREFILL_RMSNORM_GATE = "hipengine_qwen35_gdn_prefill_rmsnorm_gate_bf16"
_SYMBOL_PREFILL_RMSNORM_GATE_FP16 = "hipengine_qwen35_gdn_prefill_rmsnorm_gate_fp16"
_SYMBOL_PREFILL_RMSNORM_GATE_ROTATE_FP16 = "hipengine_qwen35_gdn_prefill_rmsnorm_gate_rotate_fp16"
_SYMBOL_SEGMENTS_LOWP_FP16 = "hipengine_qwen35_gdn_recurrent_rmsnorm_gate_segments_lowp_fp16"
_SYMBOL_PREFILL_DECODE_ORDER_BF16 = "hipengine_qwen35_gdn_prefill_recurrent_rmsnorm_gate_bf16_decode_order"


def plan_qwen35_linear_attn_gdn_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="qwen35_linear_attn_gdn",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
    )


def build_qwen35_linear_attn_gdn(
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
        family="qwen35_linear_attn_gdn",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def qwen35_gdn_recurrent_rmsnorm_gate_lowp_bf16(
    conv_out_ptr: int,
    gate_ptr: int,
    a_ptr: int,
    b_ptr: int,
    dt_bias_ptr: int,
    a_log_ptr: int,
    norm_weight_ptr: int,
    recurrent_state_ptr: int,
    out_ptr: int,
    eps: float,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch BF16-gated recurrent GDN RMSNorm+gate kernel."""

    _check_gdn_shape(num_k_heads, num_v_heads, head_k_dim, head_v_dim)
    library = library or build_qwen35_linear_attn_gdn(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_LOWP)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_float,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(conv_out_ptr),
        ctypes.c_void_p(gate_ptr),
        ctypes.c_void_p(a_ptr),
        ctypes.c_void_p(b_ptr),
        ctypes.c_void_p(dt_bias_ptr),
        ctypes.c_void_p(a_log_ptr),
        ctypes.c_void_p(norm_weight_ptr),
        ctypes.c_void_p(recurrent_state_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_float(eps),
        ctypes.c_int64(num_k_heads),
        ctypes.c_int64(num_v_heads),
        ctypes.c_int64(head_k_dim),
        ctypes.c_int64(head_v_dim),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def qwen35_gdn_recurrent_rmsnorm_gate_lowp_fp16(
    conv_out_ptr: int,
    gate_ptr: int,
    a_ptr: int,
    b_ptr: int,
    dt_bias_ptr: int,
    a_log_ptr: int,
    norm_weight_ptr: int,
    recurrent_state_ptr: int,
    out_ptr: int,
    eps: float,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch FP16-gated recurrent GDN RMSNorm+gate kernel."""

    _launch_gdn_recurrent_rmsnorm_gate_lowp(
        _SYMBOL_LOWP_FP16,
        conv_out_ptr,
        gate_ptr,
        a_ptr,
        b_ptr,
        dt_bias_ptr,
        a_log_ptr,
        norm_weight_ptr,
        recurrent_state_ptr,
        out_ptr,
        eps,
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def _launch_gdn_recurrent_rmsnorm_gate_lowp(
    symbol: str,
    conv_out_ptr: int,
    gate_ptr: int,
    a_ptr: int,
    b_ptr: int,
    dt_bias_ptr: int,
    a_log_ptr: int,
    norm_weight_ptr: int,
    recurrent_state_ptr: int,
    out_ptr: int,
    eps: float,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    *,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    _check_gdn_shape(num_k_heads, num_v_heads, head_k_dim, head_v_dim)
    library = library or build_qwen35_linear_attn_gdn(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, symbol)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_float,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(conv_out_ptr),
        ctypes.c_void_p(gate_ptr),
        ctypes.c_void_p(a_ptr),
        ctypes.c_void_p(b_ptr),
        ctypes.c_void_p(dt_bias_ptr),
        ctypes.c_void_p(a_log_ptr),
        ctypes.c_void_p(norm_weight_ptr),
        ctypes.c_void_p(recurrent_state_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_float(eps),
        ctypes.c_int64(num_k_heads),
        ctypes.c_int64(num_v_heads),
        ctypes.c_int64(head_k_dim),
        ctypes.c_int64(head_v_dim),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def qwen35_gdn_tree_recurrent_rmsnorm_gate_lowp_tloop_bf16(
    conv_out_ptr: int,
    gate_ptr: int,
    a_ptr: int,
    b_ptr: int,
    dt_bias_ptr: int,
    a_log_ptr: int,
    norm_weight_ptr: int,
    base_recurrent_state_ptr: int,
    tree_recurrent_state_ptr: int,
    parent_ids_ptr: int,
    acc_buf_ptr: int,
    out_ptr: int,
    eps: float,
    max_nodes: int,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch BF16-gated parent-indexed tree GDN t-loop recurrence+finalize."""

    _launch_gdn_tree_tloop(
        _SYMBOL_TREE_TLOOP_BF16,
        conv_out_ptr,
        gate_ptr,
        a_ptr,
        b_ptr,
        dt_bias_ptr,
        a_log_ptr,
        norm_weight_ptr,
        base_recurrent_state_ptr,
        tree_recurrent_state_ptr,
        parent_ids_ptr,
        acc_buf_ptr,
        out_ptr,
        eps,
        max_nodes,
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def qwen35_gdn_tree_recurrent_rmsnorm_gate_lowp_tloop_fp16(
    conv_out_ptr: int,
    gate_ptr: int,
    a_ptr: int,
    b_ptr: int,
    dt_bias_ptr: int,
    a_log_ptr: int,
    norm_weight_ptr: int,
    base_recurrent_state_ptr: int,
    tree_recurrent_state_ptr: int,
    parent_ids_ptr: int,
    acc_buf_ptr: int,
    out_ptr: int,
    eps: float,
    max_nodes: int,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch FP16-gated parent-indexed tree GDN t-loop recurrence+finalize."""

    _launch_gdn_tree_tloop(
        _SYMBOL_TREE_TLOOP_FP16,
        conv_out_ptr,
        gate_ptr,
        a_ptr,
        b_ptr,
        dt_bias_ptr,
        a_log_ptr,
        norm_weight_ptr,
        base_recurrent_state_ptr,
        tree_recurrent_state_ptr,
        parent_ids_ptr,
        acc_buf_ptr,
        out_ptr,
        eps,
        max_nodes,
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def qwen35_gdn_chain_recurrent_rmsnorm_gate_lowp_tloop_bf16(
    conv_out_ptr: int,
    gate_ptr: int,
    a_ptr: int,
    b_ptr: int,
    dt_bias_ptr: int,
    a_log_ptr: int,
    norm_weight_ptr: int,
    base_recurrent_state_ptr: int,
    leaf_recurrent_state_ptr: int,
    acc_buf_ptr: int,
    out_ptr: int,
    eps: float,
    max_nodes: int,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch BF16-gated single-chain GDN t-loop recurrence+finalize."""

    _launch_gdn_chain_tloop(
        _SYMBOL_CHAIN_TLOOP_BF16,
        conv_out_ptr,
        gate_ptr,
        a_ptr,
        b_ptr,
        dt_bias_ptr,
        a_log_ptr,
        norm_weight_ptr,
        base_recurrent_state_ptr,
        leaf_recurrent_state_ptr,
        acc_buf_ptr,
        out_ptr,
        eps,
        max_nodes,
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def qwen35_gdn_chain_recurrent_rmsnorm_gate_lowp_tloop_fp16(
    conv_out_ptr: int,
    gate_ptr: int,
    a_ptr: int,
    b_ptr: int,
    dt_bias_ptr: int,
    a_log_ptr: int,
    norm_weight_ptr: int,
    base_recurrent_state_ptr: int,
    leaf_recurrent_state_ptr: int,
    acc_buf_ptr: int,
    out_ptr: int,
    eps: float,
    max_nodes: int,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch FP16-gated single-chain GDN t-loop recurrence+finalize."""

    _launch_gdn_chain_tloop(
        _SYMBOL_CHAIN_TLOOP_FP16,
        conv_out_ptr,
        gate_ptr,
        a_ptr,
        b_ptr,
        dt_bias_ptr,
        a_log_ptr,
        norm_weight_ptr,
        base_recurrent_state_ptr,
        leaf_recurrent_state_ptr,
        acc_buf_ptr,
        out_ptr,
        eps,
        max_nodes,
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def qwen35_gdn_prefill_recurrent_f32(
    query_ptr: int,
    key_ptr: int,
    value_ptr: int,
    beta_ptr: int,
    decay_ptr: int,
    recurrent_state_ptr: int,
    out_ptr: int,
    tokens: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch native FP32 GDN recurrent prefill kernel."""

    _launch_prefill_recurrent(
        _SYMBOL_PREFILL,
        query_ptr,
        key_ptr,
        value_ptr,
        beta_ptr,
        decay_ptr,
        recurrent_state_ptr,
        out_ptr,
        tokens,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def qwen35_gdn_prefill_recurrent_k2_f32(
    query_ptr: int,
    key_ptr: int,
    value_ptr: int,
    beta_ptr: int,
    decay_ptr: int,
    recurrent_state_ptr: int,
    out_ptr: int,
    tokens: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch native FP32 GDN recurrent prefill K2 kernel."""

    _launch_prefill_recurrent(
        _SYMBOL_PREFILL_K2,
        query_ptr,
        key_ptr,
        value_ptr,
        beta_ptr,
        decay_ptr,
        recurrent_state_ptr,
        out_ptr,
        tokens,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def qwen35_gdn_prefill_recurrent_segments_k2_f32(
    query_ptr: int,
    key_ptr: int,
    value_ptr: int,
    beta_ptr: int,
    decay_ptr: int,
    recurrent_state_ptr: int,
    out_ptr: int,
    cu_seqlens_ptr: int,
    state_indices_ptr: int,
    total_tokens: int,
    segments: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch segment-aware FP32 GDN recurrent prefill K2 kernel.

    ``cu_seqlens`` defines packed row ranges; ``state_indices`` maps each
    segment to a leading state slot in ``[state_slots, V, K, Dv]``.
    """

    _launch_prefill_recurrent_segments(
        query_ptr,
        key_ptr,
        value_ptr,
        beta_ptr,
        decay_ptr,
        recurrent_state_ptr,
        out_ptr,
        cu_seqlens_ptr,
        state_indices_ptr,
        total_tokens,
        segments,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def qwen35_linear_attn_prefill_prepare_f32_bf16(
    conv_out_ptr: int,
    a_ptr: int,
    b_ptr: int,
    dt_bias_ptr: int,
    a_log_ptr: int,
    query_ptr: int,
    key_ptr: int,
    value_ptr: int,
    beta_ptr: int,
    decay_ptr: int,
    tokens: int,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Prepare normalized Q/K, value, beta, and decay for native prefill GDN."""

    _check_prefill_shape(tokens, num_k_heads, num_v_heads, head_k_dim, head_v_dim)
    library = library or build_qwen35_linear_attn_gdn(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_PREFILL_PREPARE)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(conv_out_ptr),
        ctypes.c_void_p(a_ptr),
        ctypes.c_void_p(b_ptr),
        ctypes.c_void_p(dt_bias_ptr),
        ctypes.c_void_p(a_log_ptr),
        ctypes.c_void_p(query_ptr),
        ctypes.c_void_p(key_ptr),
        ctypes.c_void_p(value_ptr),
        ctypes.c_void_p(beta_ptr),
        ctypes.c_void_p(decay_ptr),
        ctypes.c_int64(tokens),
        ctypes.c_int64(num_k_heads),
        ctypes.c_int64(num_v_heads),
        ctypes.c_int64(head_k_dim),
        ctypes.c_int64(head_v_dim),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def _launch_linear_attn_prefill_prepare(
    symbol: str,
    conv_out_ptr: int,
    a_ptr: int,
    b_ptr: int,
    dt_bias_ptr: int,
    a_log_ptr: int,
    query_ptr: int,
    key_ptr: int,
    value_ptr: int,
    beta_ptr: int,
    decay_ptr: int,
    tokens: int,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    *,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    _check_prefill_shape(tokens, num_k_heads, num_v_heads, head_k_dim, head_v_dim)
    library = library or build_qwen35_linear_attn_gdn(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, symbol)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(conv_out_ptr),
        ctypes.c_void_p(a_ptr),
        ctypes.c_void_p(b_ptr),
        ctypes.c_void_p(dt_bias_ptr),
        ctypes.c_void_p(a_log_ptr),
        ctypes.c_void_p(query_ptr),
        ctypes.c_void_p(key_ptr),
        ctypes.c_void_p(value_ptr),
        ctypes.c_void_p(beta_ptr),
        ctypes.c_void_p(decay_ptr),
        ctypes.c_int64(tokens),
        ctypes.c_int64(num_k_heads),
        ctypes.c_int64(num_v_heads),
        ctypes.c_int64(head_k_dim),
        ctypes.c_int64(head_v_dim),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def qwen35_linear_attn_prefill_prepare_f32_fp16(
    conv_out_ptr: int,
    a_ptr: int,
    b_ptr: int,
    dt_bias_ptr: int,
    a_log_ptr: int,
    query_ptr: int,
    key_ptr: int,
    value_ptr: int,
    beta_ptr: int,
    decay_ptr: int,
    tokens: int,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Prepare native prefill GDN tensors from FP16 lowp A/B streams."""

    _launch_linear_attn_prefill_prepare(
        _SYMBOL_PREFILL_PREPARE_FP16,
        conv_out_ptr,
        a_ptr,
        b_ptr,
        dt_bias_ptr,
        a_log_ptr,
        query_ptr,
        key_ptr,
        value_ptr,
        beta_ptr,
        decay_ptr,
        tokens,
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def qwen35_gdn_prefill_rmsnorm_gate_bf16(
    recurrent_ptr: int,
    gate_ptr: int,
    norm_weight_ptr: int,
    out_ptr: int,
    eps: float,
    tokens: int,
    num_v_heads: int,
    head_v_dim: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Apply per-head RMSNorm and SiLU gate to native prefill recurrent output."""

    _check_positive(tokens, "tokens")
    _check_positive(num_v_heads, "num_v_heads")
    _check_positive(head_v_dim, "head_v_dim")
    library = library or build_qwen35_linear_attn_gdn(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_PREFILL_RMSNORM_GATE)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_float,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(recurrent_ptr),
        ctypes.c_void_p(gate_ptr),
        ctypes.c_void_p(norm_weight_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_float(eps),
        ctypes.c_int64(tokens),
        ctypes.c_int64(num_v_heads),
        ctypes.c_int64(head_v_dim),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def qwen35_gdn_prefill_rmsnorm_gate_fp16(
    recurrent_ptr: int,
    gate_ptr: int,
    norm_weight_ptr: int,
    out_ptr: int,
    eps: float,
    tokens: int,
    num_v_heads: int,
    head_v_dim: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Apply per-head RMSNorm and SiLU gate to FP16 native prefill recurrent output."""

    _check_positive(tokens, "tokens")
    _check_positive(num_v_heads, "num_v_heads")
    _check_positive(head_v_dim, "head_v_dim")
    library = library or build_qwen35_linear_attn_gdn(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_PREFILL_RMSNORM_GATE_FP16)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_float,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(recurrent_ptr),
        ctypes.c_void_p(gate_ptr),
        ctypes.c_void_p(norm_weight_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_float(eps),
        ctypes.c_int64(tokens),
        ctypes.c_int64(num_v_heads),
        ctypes.c_int64(head_v_dim),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def qwen35_gdn_prefill_rmsnorm_gate_rotate_fp16(
    recurrent_ptr: int,
    gate_ptr: int,
    norm_weight_ptr: int,
    out_rot_ptr: int,
    pairs_ptr: int,
    theta_ptr: int,
    scales_ptr: int,
    eps: float,
    tokens: int,
    num_v_heads: int,
    head_v_dim: int,
    group_size: int,
    krot: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Apply FP16 GDN RMSNorm+SiLU gate and PARO rotate1 directly to ``out_rot``.

    This P3.1 fused prefill tail is valid for Qwen3.5/PARO's natural grouping
    where each linear-attention value head is exactly one PARO rotate group.
    """

    _check_positive(tokens, "tokens")
    _check_positive(num_v_heads, "num_v_heads")
    _check_positive(head_v_dim, "head_v_dim")
    _check_positive(group_size, "group_size")
    if int(krot) < 0:
        raise ValueError("krot must be non-negative")
    if int(group_size) != int(head_v_dim):
        raise ValueError("group_size must equal head_v_dim for fused GDN rotate")
    if int(group_size) % 2:
        raise ValueError("group_size must be even for fused GDN rotate")
    library = library or build_qwen35_linear_attn_gdn(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_PREFILL_RMSNORM_GATE_ROTATE_FP16)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_float,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(recurrent_ptr),
        ctypes.c_void_p(gate_ptr),
        ctypes.c_void_p(norm_weight_ptr),
        ctypes.c_void_p(out_rot_ptr),
        ctypes.c_void_p(pairs_ptr),
        ctypes.c_void_p(theta_ptr),
        ctypes.c_void_p(scales_ptr),
        ctypes.c_float(eps),
        ctypes.c_int64(tokens),
        ctypes.c_int64(num_v_heads),
        ctypes.c_int64(head_v_dim),
        ctypes.c_int64(group_size),
        ctypes.c_int64(krot),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def qwen35_gdn_prefill_recurrent_rmsnorm_gate_bf16_decode_order(
    conv_out_ptr: int,
    gate_ptr: int,
    a_ptr: int,
    b_ptr: int,
    dt_bias_ptr: int,
    a_log_ptr: int,
    norm_weight_ptr: int,
    recurrent_state_ptr: int,
    out_ptr: int,
    eps: float,
    tokens: int,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch decode-order BF16 gated recurrent GDN prefill."""

    _check_prefill_shape(tokens, num_k_heads, num_v_heads, head_k_dim, head_v_dim)
    library = library or build_qwen35_linear_attn_gdn(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_PREFILL_DECODE_ORDER_BF16)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_float,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(conv_out_ptr),
        ctypes.c_void_p(gate_ptr),
        ctypes.c_void_p(a_ptr),
        ctypes.c_void_p(b_ptr),
        ctypes.c_void_p(dt_bias_ptr),
        ctypes.c_void_p(a_log_ptr),
        ctypes.c_void_p(norm_weight_ptr),
        ctypes.c_void_p(recurrent_state_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_float(eps),
        ctypes.c_int64(tokens),
        ctypes.c_int64(num_k_heads),
        ctypes.c_int64(num_v_heads),
        ctypes.c_int64(head_k_dim),
        ctypes.c_int64(head_v_dim),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def qwen35_gdn_recurrent_rmsnorm_gate_segments_lowp_fp16(
    conv_out_ptr: int,
    gate_ptr: int,
    a_ptr: int,
    b_ptr: int,
    dt_bias_ptr: int,
    a_log_ptr: int,
    norm_weight_ptr: int,
    recurrent_state_ptr: int,
    out_ptr: int,
    cu_seqlens_ptr: int,
    state_indices_ptr: int,
    total_tokens: int,
    segments: int,
    eps: float,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch segmented FP16-gated decode-order recurrent GDN kernel."""

    _check_prefill_shape(total_tokens, num_k_heads, num_v_heads, head_k_dim, head_v_dim)
    if segments <= 0:
        raise ValueError("segments must be positive")
    library = library or build_qwen35_linear_attn_gdn(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_SEGMENTS_LOWP_FP16)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_float,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(conv_out_ptr),
        ctypes.c_void_p(gate_ptr),
        ctypes.c_void_p(a_ptr),
        ctypes.c_void_p(b_ptr),
        ctypes.c_void_p(dt_bias_ptr),
        ctypes.c_void_p(a_log_ptr),
        ctypes.c_void_p(norm_weight_ptr),
        ctypes.c_void_p(recurrent_state_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_void_p(cu_seqlens_ptr),
        ctypes.c_void_p(state_indices_ptr),
        ctypes.c_int64(total_tokens),
        ctypes.c_int64(segments),
        ctypes.c_float(eps),
        ctypes.c_int64(num_k_heads),
        ctypes.c_int64(num_v_heads),
        ctypes.c_int64(head_k_dim),
        ctypes.c_int64(head_v_dim),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def register_qwen35_linear_attn_gdn_kernels(*, replace: bool = True) -> None:
    register(
        KernelKey("hip_gfx1100", "gdn_recurrent_rmsnorm_gate", "w4_paro", "bf16_lowp"),
        qwen35_gdn_recurrent_rmsnorm_gate_lowp_bf16,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "gdn_recurrent_rmsnorm_gate", "w4_paro", "fp16_lowp"),
        qwen35_gdn_recurrent_rmsnorm_gate_lowp_fp16,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "gdn_prefill_recurrent", "w4_paro", "f32"),
        qwen35_gdn_prefill_recurrent_f32,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "gdn_prefill_recurrent", "w4_paro", "f32_k2"),
        qwen35_gdn_prefill_recurrent_k2_f32,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "gdn_prefill_recurrent", "w4_paro", "f32_k2_segments"),
        qwen35_gdn_prefill_recurrent_segments_k2_f32,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "linear_attn_prefill_prepare", "w4_paro", "f32_bf16"),
        qwen35_linear_attn_prefill_prepare_f32_bf16,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "linear_attn_prefill_prepare", "w4_paro", "f32_fp16"),
        qwen35_linear_attn_prefill_prepare_f32_fp16,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "gdn_prefill_rmsnorm_gate", "w4_paro", "bf16"),
        qwen35_gdn_prefill_rmsnorm_gate_bf16,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "gdn_prefill_rmsnorm_gate", "w4_paro", "fp16"),
        qwen35_gdn_prefill_rmsnorm_gate_fp16,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "gdn_prefill_rmsnorm_gate_rotate", "w4_paro", "fp16"),
        qwen35_gdn_prefill_rmsnorm_gate_rotate_fp16,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "gdn_prefill_recurrent", "gguf_qwen35", "decode_order_bf16"),
        qwen35_gdn_prefill_recurrent_rmsnorm_gate_bf16_decode_order,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "gdn_prefill_recurrent", "gguf_qwen35", "f32_k2"),
        qwen35_gdn_prefill_recurrent_k2_f32,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "gdn_prefill_recurrent", "gguf_qwen35", "f32_k2_segments"),
        qwen35_gdn_prefill_recurrent_segments_k2_f32,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "linear_attn_prefill_prepare", "gguf_qwen35", "f32_bf16"),
        qwen35_linear_attn_prefill_prepare_f32_bf16,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "gdn_prefill_rmsnorm_gate", "gguf_qwen35", "bf16"),
        qwen35_gdn_prefill_rmsnorm_gate_bf16,
        replace=replace,
    )
    for backend in ("hip_gfx1100", "hip_gfx1151"):
        register(
            KernelKey(backend, "gdn_tree_recurrent_rmsnorm_gate", "w4_paro", "bf16_tloop"),
            qwen35_gdn_tree_recurrent_rmsnorm_gate_lowp_tloop_bf16,
            replace=replace,
        )
        register(
            KernelKey(backend, "gdn_tree_recurrent_rmsnorm_gate", "w4_paro", "fp16_tloop"),
            qwen35_gdn_tree_recurrent_rmsnorm_gate_lowp_tloop_fp16,
            replace=replace,
        )


def _launch_gdn_tree_tloop(
    symbol: str,
    conv_out_ptr: int,
    gate_ptr: int,
    a_ptr: int,
    b_ptr: int,
    dt_bias_ptr: int,
    a_log_ptr: int,
    norm_weight_ptr: int,
    base_recurrent_state_ptr: int,
    tree_recurrent_state_ptr: int,
    parent_ids_ptr: int,
    acc_buf_ptr: int,
    out_ptr: int,
    eps: float,
    max_nodes: int,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    *,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    _check_positive(max_nodes, "max_nodes")
    _check_gdn_shape(num_k_heads, num_v_heads, head_k_dim, head_v_dim)
    if head_k_dim > 256 or head_k_dim % 64:
        raise ValueError("tree GDN t-loop requires head_k_dim divisible by 64 and <= 256")
    library = library or build_qwen35_linear_attn_gdn(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, symbol)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_float,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(conv_out_ptr),
        ctypes.c_void_p(gate_ptr),
        ctypes.c_void_p(a_ptr),
        ctypes.c_void_p(b_ptr),
        ctypes.c_void_p(dt_bias_ptr),
        ctypes.c_void_p(a_log_ptr),
        ctypes.c_void_p(norm_weight_ptr),
        ctypes.c_void_p(base_recurrent_state_ptr),
        ctypes.c_void_p(tree_recurrent_state_ptr),
        ctypes.c_void_p(parent_ids_ptr),
        ctypes.c_void_p(acc_buf_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_float(eps),
        ctypes.c_int64(max_nodes),
        ctypes.c_int64(num_k_heads),
        ctypes.c_int64(num_v_heads),
        ctypes.c_int64(head_k_dim),
        ctypes.c_int64(head_v_dim),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def _launch_gdn_chain_tloop(
    symbol: str,
    conv_out_ptr: int,
    gate_ptr: int,
    a_ptr: int,
    b_ptr: int,
    dt_bias_ptr: int,
    a_log_ptr: int,
    norm_weight_ptr: int,
    base_recurrent_state_ptr: int,
    leaf_recurrent_state_ptr: int,
    acc_buf_ptr: int,
    out_ptr: int,
    eps: float,
    max_nodes: int,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    *,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    _check_positive(max_nodes, "max_nodes")
    _check_gdn_shape(num_k_heads, num_v_heads, head_k_dim, head_v_dim)
    if head_k_dim > 256 or head_k_dim % 64:
        raise ValueError("chain GDN t-loop requires head_k_dim divisible by 64 and <= 256")
    library = library or build_qwen35_linear_attn_gdn(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, symbol)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_float,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(conv_out_ptr),
        ctypes.c_void_p(gate_ptr),
        ctypes.c_void_p(a_ptr),
        ctypes.c_void_p(b_ptr),
        ctypes.c_void_p(dt_bias_ptr),
        ctypes.c_void_p(a_log_ptr),
        ctypes.c_void_p(norm_weight_ptr),
        ctypes.c_void_p(base_recurrent_state_ptr),
        ctypes.c_void_p(leaf_recurrent_state_ptr),
        ctypes.c_void_p(acc_buf_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_float(eps),
        ctypes.c_int64(max_nodes),
        ctypes.c_int64(num_k_heads),
        ctypes.c_int64(num_v_heads),
        ctypes.c_int64(head_k_dim),
        ctypes.c_int64(head_v_dim),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def _launch_prefill_recurrent(
    symbol: str,
    query_ptr: int,
    key_ptr: int,
    value_ptr: int,
    beta_ptr: int,
    decay_ptr: int,
    recurrent_state_ptr: int,
    out_ptr: int,
    tokens: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    *,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    _check_positive(tokens, "tokens")
    _check_positive(num_v_heads, "num_v_heads")
    _check_positive(head_k_dim, "head_k_dim")
    _check_positive(head_v_dim, "head_v_dim")
    if head_k_dim != 128:
        raise ValueError("head_k_dim must be 128 for native prefill GDN")
    library = library or build_qwen35_linear_attn_gdn(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, symbol)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(query_ptr),
        ctypes.c_void_p(key_ptr),
        ctypes.c_void_p(value_ptr),
        ctypes.c_void_p(beta_ptr),
        ctypes.c_void_p(decay_ptr),
        ctypes.c_void_p(recurrent_state_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(tokens),
        ctypes.c_int64(num_v_heads),
        ctypes.c_int64(head_k_dim),
        ctypes.c_int64(head_v_dim),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def _launch_prefill_recurrent_segments(
    query_ptr: int,
    key_ptr: int,
    value_ptr: int,
    beta_ptr: int,
    decay_ptr: int,
    recurrent_state_ptr: int,
    out_ptr: int,
    cu_seqlens_ptr: int,
    state_indices_ptr: int,
    total_tokens: int,
    segments: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    *,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    _check_positive(total_tokens, "total_tokens")
    _check_positive(segments, "segments")
    _check_positive(num_v_heads, "num_v_heads")
    _check_positive(head_k_dim, "head_k_dim")
    _check_positive(head_v_dim, "head_v_dim")
    if head_k_dim != 128:
        raise ValueError("head_k_dim must be 128 for native prefill GDN")
    library = library or build_qwen35_linear_attn_gdn(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_PREFILL_SEGMENTS_K2)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(query_ptr),
        ctypes.c_void_p(key_ptr),
        ctypes.c_void_p(value_ptr),
        ctypes.c_void_p(beta_ptr),
        ctypes.c_void_p(decay_ptr),
        ctypes.c_void_p(recurrent_state_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_void_p(cu_seqlens_ptr),
        ctypes.c_void_p(state_indices_ptr),
        ctypes.c_int64(total_tokens),
        ctypes.c_int64(segments),
        ctypes.c_int64(num_v_heads),
        ctypes.c_int64(head_k_dim),
        ctypes.c_int64(head_v_dim),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def _check_prefill_shape(
    tokens: int,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
) -> None:
    _check_positive(tokens, "tokens")
    _check_positive(num_k_heads, "num_k_heads")
    _check_positive(num_v_heads, "num_v_heads")
    if num_v_heads % num_k_heads != 0:
        raise ValueError("num_v_heads must be divisible by num_k_heads")
    _check_positive(head_k_dim, "head_k_dim")
    _check_positive(head_v_dim, "head_v_dim")


def _check_gdn_shape(
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
) -> None:
    _check_positive(num_k_heads, "num_k_heads")
    _check_positive(num_v_heads, "num_v_heads")
    if num_v_heads % num_k_heads != 0:
        raise ValueError("num_v_heads must be divisible by num_k_heads")
    _check_positive(head_k_dim, "head_k_dim")
    _check_positive(head_v_dim, "head_v_dim")
    if head_v_dim > 128:
        raise ValueError("head_v_dim must be <= 128")


def _check_positive(value: int, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _check_launch(runtime: HipRuntime, err: int) -> None:
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


register_qwen35_linear_attn_gdn_kernels()

