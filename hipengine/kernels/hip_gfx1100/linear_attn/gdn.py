"""Raw-pointer wrappers for Qwen3.5 linear-attention GDN kernels."""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.kernels.hip_gfx1100.convert.cast import f32_to_bf16
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("gdn.hip")
_OUTPUT_NAME = "qwen35_linear_attn_gdn.so"
_GROUPED_OUTPUT_NAME = "qwen35_linear_attn_gdn_grouped_heads.so"
_GROUPED_BUILD_FLAG = "-DHIPENGINE_GDN_GROUPED_HEADS=1"
_SYMBOL_LOWP = "hipengine_qwen35_gdn_recurrent_rmsnorm_gate_lowp_bf16"
_SYMBOL_LOWP_BF16_FP16STATE = (
    "hipengine_qwen35_gdn_recurrent_rmsnorm_gate_lowp_bf16_fp16state"
)
_SYMBOL_LOWP_F32_BF16 = (
    "hipengine_qwen35_gdn_recurrent_rmsnorm_gate_lowp_f32_bf16_out"
)
_SYMBOL_LOWP_F32_BF16_FP16STATE = (
    "hipengine_qwen35_gdn_recurrent_rmsnorm_gate_lowp_f32_bf16_out_fp16state"
)
_SYMBOL_LOWP_FP16 = "hipengine_qwen35_gdn_recurrent_rmsnorm_gate_lowp_fp16"
_SYMBOL_TREE_TLOOP_BF16 = "hipengine_qwen35_gdn_tree_recurrent_rmsnorm_gate_lowp_tloop_bf16"
_SYMBOL_TREE_TLOOP_FP16 = "hipengine_qwen35_gdn_tree_recurrent_rmsnorm_gate_lowp_tloop_fp16"
_SYMBOL_CHAIN_TLOOP_BF16 = "hipengine_qwen35_gdn_chain_recurrent_rmsnorm_gate_lowp_tloop_bf16"
_SYMBOL_CHAIN_C1_EXACT_TLOOP_BF16 = "hipengine_qwen35_gdn_chain_recurrent_rmsnorm_gate_lowp_c1_exact_tloop_bf16"
_SYMBOL_CHAIN_C1_EXACT_TLOOP_BF16_FP16STATE = (
    "hipengine_qwen35_gdn_chain_recurrent_rmsnorm_gate_lowp_c1_exact_tloop_bf16_fp16state"
)
_SYMBOL_CHAIN_C1_EXACT_TLOOP_F32_BF16 = (
    "hipengine_qwen35_gdn_chain_recurrent_rmsnorm_gate_lowp_"
    "c1_exact_tloop_f32_bf16_out"
)
_SYMBOL_CHAIN_C1_EXACT_SNAPSHOT_TLOOP_BF16 = (
    "hipengine_qwen35_gdn_chain_recurrent_rmsnorm_gate_lowp_"
    "c1_exact_snapshot_tloop_bf16"
)
_SYMBOL_CHAIN_C1_EXACT_SNAPSHOT_TLOOP_F32_BF16 = (
    "hipengine_qwen35_gdn_chain_recurrent_rmsnorm_gate_lowp_"
    "c1_exact_snapshot_tloop_f32_bf16_out"
)
_SYMBOL_ALPHA_BETA_GDN_CHAIN_SNAPSHOT_F32_BF16 = (
    "hipengine_qwen35_linear_attn_alpha_beta_gdn_chain_snapshot_f32_bf16_out"
)
_SYMBOL_CHAIN_TLOOP_FP16 = "hipengine_qwen35_gdn_chain_recurrent_rmsnorm_gate_lowp_tloop_fp16"
_SYMBOL_PREFILL = "hipengine_qwen35_gdn_prefill_recurrent_f32"
_SYMBOL_PREFILL_K2 = "hipengine_qwen35_gdn_prefill_recurrent_k2_f32"
_SYMBOL_PREFILL_K2_DECODE_ORDER = "hipengine_qwen35_gdn_prefill_recurrent_k2_decode_order_f32"
_SYMBOL_PREFILL_EXACT_LDS32 = (
    "hipengine_qwen35_gdn_prefill_recurrent_decode_order_exact_lds32_f32"
)
_SYMBOL_PREFILL_SEGMENTS_K2 = "hipengine_qwen35_gdn_prefill_recurrent_segments_k2_f32"
_SYMBOL_PREFILL_NORMALIZED_WAVE32_XOR = (
    "hipengine_qwen35_gdn_prefill_recurrent_normalized_wave32_xor_f32"
)
_SYMBOL_PREFILL_COMPACT_NORMALIZED_WAVE32_XOR = (
    "hipengine_qwen35_gdn_prefill_recurrent_compact_normalized_wave32_xor_f32"
)
_SYMBOL_PREFILL_COMPACT_NORMALIZED_WAVE32_XOR_FP16STATE = (
    "hipengine_qwen35_gdn_prefill_recurrent_compact_normalized_wave32_xor_fp16state"
)
_SYMBOL_PREFILL_COMPACT_NORMALIZED_SEGMENTS_WAVE32_XOR = (
    "hipengine_qwen35_gdn_prefill_recurrent_compact_normalized_segments_wave32_xor_f32"
)
_SYMBOL_PREFILL_COMPACT_NORMALIZED_SEGMENTS_WAVE32_XOR_FP16STATE = (
    "hipengine_qwen35_gdn_prefill_recurrent_compact_normalized_segments_wave32_xor_fp16state"
)
_SYMBOL_PREFILL_NORMALIZED_SEGMENTS_WAVE32_XOR = (
    "hipengine_qwen35_gdn_prefill_recurrent_normalized_segments_wave32_xor_f32"
)
_SYMBOL_PREFILL_NORMALIZED_CLUSTER8 = (
    "hipengine_qwen35_gdn_prefill_recurrent_normalized_cluster8_f32"
)
_SYMBOL_PREFILL_NORMALIZED_SEGMENTS_CLUSTER8 = (
    "hipengine_qwen35_gdn_prefill_recurrent_normalized_segments_cluster8_f32"
)
_SYMBOL_PREFILL_PREPARE = "hipengine_qwen35_linear_attn_prefill_prepare_f32_bf16"
_SYMBOL_PREFILL_PREPARE_DECODE_ORDER = (
    "hipengine_qwen35_linear_attn_prefill_prepare_decode_order_f32_bf16"
)
_SYMBOL_PREFILL_PREPARE_PEER_NORMALIZED = (
    "hipengine_qwen35_linear_attn_prefill_prepare_peer_normalized_f32_bf16"
)
_SYMBOL_PREFILL_PREPARE_COMPACT_PEER_NORMALIZED = (
    "hipengine_qwen35_linear_attn_prefill_prepare_compact_peer_normalized_f32_bf16"
)
_SYMBOL_PREFILL_PREPARE_FP16 = "hipengine_qwen35_linear_attn_prefill_prepare_f32_fp16"
_SYMBOL_PREFILL_PREPARE_RAW_SCALES = (
    "hipengine_qwen35_linear_attn_prefill_prepare_raw_scales_f32_bf16"
)
_SYMBOL_PREFILL_PREPARE_COMPACT_SCALES = (
    "hipengine_qwen35_linear_attn_prefill_prepare_compact_scales_f32_bf16"
)
_SYMBOL_PREFILL_DECODE_ORDER_EXACT = (
    "hipengine_qwen35_gdn_prefill_recurrent_decode_order_exact_f32"
)
_SYMBOL_PREFILL_DECODE_ORDER_EXACT_TILE64 = (
    "hipengine_qwen35_gdn_prefill_recurrent_decode_order_exact_tile64_f32"
)
_SYMBOL_PREFILL_DECODE_ORDER_EXACT_TILE32 = (
    "hipengine_qwen35_gdn_prefill_recurrent_decode_order_exact_tile32_f32"
)
_SYMBOL_PREFILL_DECODE_ORDER_EXACT_LDS64 = (
    "hipengine_qwen35_gdn_prefill_recurrent_decode_order_exact_lds64_f32"
)
_SYMBOL_PREFILL_DECODE_ORDER_EXACT_LDS32 = (
    "hipengine_qwen35_gdn_prefill_recurrent_decode_order_exact_lds32_f32"
)
_SYMBOL_PREFILL_DECODE_ORDER_EXACT_LDS32_DIRECT = (
    "hipengine_qwen35_gdn_prefill_recurrent_decode_order_exact_lds32_direct_f32"
)
_SYMBOL_PREFILL_DECODE_ORDER_EXACT_LDS32_DIRECT_NONVOLATILE = (
    "hipengine_qwen35_gdn_prefill_recurrent_decode_order_exact_lds32_direct_nonvolatile_f32"
)
_SYMBOL_PREFILL_DECODE_ORDER_EXACT_WAVE32 = (
    "hipengine_qwen35_gdn_prefill_recurrent_decode_order_exact_wave32_f32"
)
_SYMBOL_PREFILL_DECODE_ORDER_WAVE32_TREE = (
    "hipengine_qwen35_gdn_prefill_recurrent_decode_order_wave32_tree_f32"
)
_SYMBOL_PREFILL_DECODE_ORDER_EXACT_SEGMENTS = (
    "hipengine_qwen35_gdn_prefill_recurrent_decode_order_exact_segments_f32"
)
_SYMBOL_PREFILL_DECODE_ORDER_EXACT_SEGMENTS_TILE64 = (
    "hipengine_qwen35_gdn_prefill_recurrent_decode_order_exact_segments_tile64_f32"
)
_SYMBOL_PREFILL_DECODE_ORDER_EXACT_SEGMENTS_TILE32 = (
    "hipengine_qwen35_gdn_prefill_recurrent_decode_order_exact_segments_tile32_f32"
)
_SYMBOL_PREFILL_DECODE_ORDER_EXACT_SEGMENTS_LDS64 = (
    "hipengine_qwen35_gdn_prefill_recurrent_decode_order_exact_segments_lds64_f32"
)
_SYMBOL_PREFILL_DECODE_ORDER_EXACT_SEGMENTS_LDS32 = (
    "hipengine_qwen35_gdn_prefill_recurrent_decode_order_exact_segments_lds32_f32"
)
_SYMBOL_PREFILL_DECODE_ORDER_EXACT_SEGMENTS_LDS32_DIRECT = (
    "hipengine_qwen35_gdn_prefill_recurrent_decode_order_exact_segments_lds32_direct_f32"
)
_SYMBOL_PREFILL_DECODE_ORDER_EXACT_SEGMENTS_LDS32_DIRECT_NONVOLATILE = (
    "hipengine_qwen35_gdn_prefill_recurrent_decode_order_exact_segments_lds32_direct_nonvolatile_f32"
)
_SYMBOL_PREFILL_DECODE_ORDER_EXACT_SEGMENTS_WAVE32 = (
    "hipengine_qwen35_gdn_prefill_recurrent_decode_order_exact_segments_wave32_f32"
)
_SYMBOL_PREFILL_DECODE_ORDER_SEGMENTS_WAVE32_TREE = (
    "hipengine_qwen35_gdn_prefill_recurrent_decode_order_segments_wave32_tree_f32"
)
_SYMBOL_PREFILL_RMSNORM_GATE = "hipengine_qwen35_gdn_prefill_rmsnorm_gate_bf16"
_SYMBOL_PREFILL_RMSNORM_GATE_FP16 = "hipengine_qwen35_gdn_prefill_rmsnorm_gate_fp16"
_SYMBOL_PREFILL_RMSNORM_GATE_ROTATE_FP16 = "hipengine_qwen35_gdn_prefill_rmsnorm_gate_rotate_fp16"
_SYMBOL_INDEXED_LOWP_BF16 = "hipengine_qwen35_gdn_recurrent_rmsnorm_gate_indexed_lowp_bf16"
_SYMBOL_INDEXED_LOWP_BF16_FP16STATE = (
    "hipengine_qwen35_gdn_recurrent_rmsnorm_gate_indexed_lowp_bf16_fp16state"
)
_SYMBOL_INDEXED_SHARED_STATECACHE24_LOWP_BF16 = (
    "hipengine_qwen35_gdn_recurrent_rmsnorm_gate_indexed_shared_statecache24_lowp_bf16"
)
_SYMBOL_INDEXED_SHARED_STATECACHE24_LOWP_BF16_FP16STATE = (
    "hipengine_qwen35_gdn_recurrent_rmsnorm_gate_indexed_shared_statecache24_lowp_bf16_fp16state"
)
_SYMBOL_SEGMENTS_LOWP_BF16 = "hipengine_qwen35_gdn_recurrent_rmsnorm_gate_segments_lowp_bf16"
_SYMBOL_SEGMENTS_LOWP_STATE_ROWS_BF16 = (
    "hipengine_qwen35_gdn_recurrent_rmsnorm_gate_segments_lowp_state_rows_bf16"
)
_SYMBOL_SEGMENTS_LOWP_BF16_FP16STATE = (
    "hipengine_qwen35_gdn_recurrent_rmsnorm_gate_segments_lowp_bf16_fp16state"
)
_SYMBOL_SEGMENTS_LOWP_FP16 = "hipengine_qwen35_gdn_recurrent_rmsnorm_gate_segments_lowp_fp16"
_SYMBOL_PREFILL_DECODE_ORDER_BF16 = "hipengine_qwen35_gdn_prefill_recurrent_rmsnorm_gate_bf16_decode_order"
_SYMBOL_PREFILL_DECODE_ORDER_STATE_ROWS_BF16 = (
    "hipengine_qwen35_gdn_prefill_recurrent_rmsnorm_gate_bf16_decode_order_state_rows"
)
_SYMBOL_PREFILL_DECODE_ORDER_STATE_ROWS_NO_COPY_BF16 = (
    "hipengine_qwen35_gdn_prefill_recurrent_rmsnorm_gate_bf16_decode_order_state_rows_no_copy"
)
_SYMBOL_PREFILL_DECODE_ORDER_STATE_ROWS_NO_COPY_BF16_FP16STATE = (
    "hipengine_qwen35_gdn_prefill_recurrent_rmsnorm_gate_bf16_decode_order_state_rows_no_copy_fp16state"
)
_SYMBOL_PREFILL_DECODE_ORDER_SEGMENTS_STATE_ROWS_NO_COPY_BF16 = (
    "hipengine_qwen35_gdn_prefill_recurrent_rmsnorm_gate_bf16_decode_order_segments_state_rows_no_copy"
)
_SYMBOL_PREFILL_DECODE_ORDER_SEGMENTS_STATE_ROWS_NO_COPY_WAVE_REDUCE_BF16 = (
    "hipengine_qwen35_gdn_prefill_recurrent_rmsnorm_gate_bf16_decode_order_segments_state_rows_no_copy_wave_reduce"
)
_SYMBOL_PREFILL_DECODE_ORDER_SEGMENTS_STATE_ROWS_NO_COPY_F32_BF16 = (
    "hipengine_qwen35_gdn_prefill_recurrent_rmsnorm_gate_bf16_decode_order_segments_state_rows_no_copy_f32"
)
_SYMBOL_PREFILL_DECODE_ORDER_SEGMENTS_STATE_ROWS_NO_COPY_BF16_FP16STATE = (
    "hipengine_qwen35_gdn_prefill_recurrent_rmsnorm_gate_bf16_decode_order_segments_state_rows_no_copy_fp16state"
)
_SYMBOL_PREFILL_DECODE_ORDER_SEGMENTS_BF16 = (
    "hipengine_qwen35_gdn_prefill_recurrent_rmsnorm_gate_bf16_decode_order_segments"
)
_SYMBOL_PREFILL_DECODE_ORDER_SEGMENTS_BF16_FP16STATE = (
    "hipengine_qwen35_gdn_prefill_recurrent_rmsnorm_gate_bf16_decode_order_segments_fp16state"
)


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


def plan_qwen35_linear_attn_gdn_grouped_heads_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
) -> BuildArtifact:
    """Plan the canonical HF/PARO grouped-V-head GDN sibling."""

    return plan_hip_build(
        sources=[_SOURCE],
        family="qwen35_linear_attn_gdn_grouped_heads",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        extra_flags=(_GROUPED_BUILD_FLAG,),
        output_name=_GROUPED_OUTPUT_NAME,
    )


def build_qwen35_linear_attn_gdn_grouped_heads(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
    dry_run: bool = False,
    load: bool = True,
    require_cached: bool = False,
) -> ctypes.CDLL | BuildArtifact:
    """Build GDN kernels for canonical HF/PARO grouped V-head tensors.

    llama.cpp GGUF conversion reorders linear-attention V-head tensors to a
    tiled layout before quantization. Packed PARO safetensors retain the
    canonical Transformers ``repeat_interleave`` layout, so they require this
    separately cached compile-time sibling while sharing the same C ABI.
    """

    return build_hip(
        sources=[_SOURCE],
        family="qwen35_linear_attn_gdn_grouped_heads",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        extra_flags=(_GROUPED_BUILD_FLAG,),
        output_name=_GROUPED_OUTPUT_NAME,
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


def qwen35_gdn_recurrent_rmsnorm_gate_lowp_bf16_fp16state(
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
    """Launch BF16-gated recurrent GDN with fp16-state storage (fp32 accumulate).

    Production fp16-state route: ``recurrent_state`` is a ``half`` buffer; the
    kernel reads it, accumulates in fp32, and writes fp16 state back.  The
    FP32-state ``...lowp_bf16`` wrapper remains the registered strict fallback.
    """

    _check_gdn_shape(num_k_heads, num_v_heads, head_k_dim, head_v_dim)
    library = library or build_qwen35_linear_attn_gdn(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_LOWP_BF16_FP16STATE)
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


def qwen35_gdn_recurrent_rmsnorm_gate_lowp_f32_bf16_out(
    conv_out_ptr: int,
    gate_ptr: int,
    a_ptr: int,
    b_ptr: int,
    dt_bias_ptr: int,
    a_log_ptr: int,
    norm_weight_ptr: int,
    recurrent_state_ptr: int,
    out_ptr: int,
    out_bf16_ptr: int,
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
    """Launch recurrent GDN with exact FP32 and rounded BF16 outputs."""

    _launch_gdn_recurrent_rmsnorm_gate_lowp(
        _SYMBOL_LOWP_F32_BF16,
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
        out_bf16_ptr=out_bf16_ptr,
    )


def qwen35_gdn_recurrent_rmsnorm_gate_lowp_f32_bf16_out_fp16state(
    conv_out_ptr: int,
    gate_ptr: int,
    a_ptr: int,
    b_ptr: int,
    dt_bias_ptr: int,
    a_log_ptr: int,
    norm_weight_ptr: int,
    recurrent_state_ptr: int,
    out_ptr: int,
    out_bf16_ptr: int,
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
    """Launch fused GDN+cast with fp16-state storage.

    Production fp16-state route for the Qwen3.8 Q4_K_S topline: the dense
    ``ssm_out`` projection quantizes as ``gguf_q5_k_t16_v1`` and resolves the
    fused ``gdn_recurrent_rmsnorm_gate+cast`` owner; this wrapper launches its
    half-state instantiation (fp32 accumulate, fp16 round-trip state).
    """

    _launch_gdn_recurrent_rmsnorm_gate_lowp(
        _SYMBOL_LOWP_F32_BF16_FP16STATE,
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
        out_bf16_ptr=out_bf16_ptr,
    )


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
    out_bf16_ptr: int | None = None,
) -> None:
    _check_gdn_shape(num_k_heads, num_v_heads, head_k_dim, head_v_dim)
    library = library or build_qwen35_linear_attn_gdn(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, symbol)
    pointer_args = [
        conv_out_ptr,
        gate_ptr,
        a_ptr,
        b_ptr,
        dt_bias_ptr,
        a_log_ptr,
        norm_weight_ptr,
        recurrent_state_ptr,
        out_ptr,
    ]
    if out_bf16_ptr is not None:
        pointer_args.append(out_bf16_ptr)
    fn.argtypes = [ctypes.c_void_p] * len(pointer_args) + [
        ctypes.c_float,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        *(ctypes.c_void_p(value) for value in pointer_args),
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


def qwen35_gdn_chain_recurrent_rmsnorm_gate_lowp_c1_exact_tloop_bf16(
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
    """Launch exact BF16-gated chain GDN t-loop recurrence+FP32 finalize."""

    _launch_gdn_chain_tloop(
        _SYMBOL_CHAIN_C1_EXACT_TLOOP_BF16,
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


def qwen35_gdn_chain_recurrent_rmsnorm_gate_lowp_c1_exact_tloop_bf16_fp16state(
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
    """Launch exact BF16-gated chain GDN t-loop with fp16-state storage.

    Production fp16-state route: ``base_recurrent_state`` and
    ``leaf_recurrent_state`` are ``half`` buffers; the kernel reads/accumulates
    in fp32 and writes fp16 state back.  The FP32-state
    ``...c1_exact_tloop_bf16`` wrapper remains the registered strict fallback.
    """

    _launch_gdn_chain_tloop(
        _SYMBOL_CHAIN_C1_EXACT_TLOOP_BF16_FP16STATE,
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


def qwen35_gdn_chain_recurrent_rmsnorm_gate_lowp_c1_exact_tloop_f32_bf16_out(
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
    out_bf16_ptr: int,
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
    """Launch exact chain GDN with FP32 and rounded BF16 row outputs."""

    _launch_gdn_chain_tloop(
        _SYMBOL_CHAIN_C1_EXACT_TLOOP_F32_BF16,
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
        out_bf16_ptr=out_bf16_ptr,
    )


def qwen35_gdn_chain_recurrent_rmsnorm_gate_lowp_c1_exact_snapshot_tloop_bf16(
    conv_out_ptr: int,
    gate_ptr: int,
    a_ptr: int,
    b_ptr: int,
    dt_bias_ptr: int,
    a_log_ptr: int,
    norm_weight_ptr: int,
    base_recurrent_state_ptr: int,
    leaf_recurrent_state_ptr: int,
    initial_recurrent_state_snapshot_ptr: int,
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
    """Write exact chain rows plus the immutable initial recurrent state."""

    _launch_gdn_chain_tloop(
        _SYMBOL_CHAIN_C1_EXACT_SNAPSHOT_TLOOP_BF16,
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
        initial_state_snapshot_ptr=initial_recurrent_state_snapshot_ptr,
    )


def qwen35_gdn_chain_recurrent_rmsnorm_gate_lowp_c1_exact_snapshot_tloop_f32_bf16_out(
    conv_out_ptr: int,
    gate_ptr: int,
    a_ptr: int,
    b_ptr: int,
    dt_bias_ptr: int,
    a_log_ptr: int,
    norm_weight_ptr: int,
    base_recurrent_state_ptr: int,
    leaf_recurrent_state_ptr: int,
    initial_recurrent_state_snapshot_ptr: int,
    acc_buf_ptr: int,
    out_ptr: int,
    out_bf16_ptr: int,
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
    """Write the initial state and exact FP32/BF16 chain outputs."""

    _launch_gdn_chain_tloop(
        _SYMBOL_CHAIN_C1_EXACT_SNAPSHOT_TLOOP_F32_BF16,
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
        initial_state_snapshot_ptr=initial_recurrent_state_snapshot_ptr,
        out_bf16_ptr=out_bf16_ptr,
    )


def qwen35_linear_attn_alpha_beta_gdn_chain_snapshot_f32_bf16_out(
    norm_ptr: int,
    alpha_weight_ptr: int,
    beta_weight_ptr: int,
    alpha_out_ptr: int,
    beta_out_ptr: int,
    conv_out_ptr: int,
    gate_ptr: int,
    dt_bias_ptr: int,
    a_log_ptr: int,
    norm_weight_ptr: int,
    base_recurrent_state_ptr: int,
    leaf_recurrent_state_ptr: int,
    initial_recurrent_state_snapshot_ptr: int,
    out_ptr: int,
    out_bf16_ptr: int,
    eps: float,
    rows: int,
    in_features: int,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    *,
    threads: int = 256,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Project exact BF16 alpha/beta and consume them in snapshot chain GDN."""

    rows = int(rows)
    in_features = int(in_features)
    num_k_heads = int(num_k_heads)
    num_v_heads = int(num_v_heads)
    head_k_dim = int(head_k_dim)
    head_v_dim = int(head_v_dim)
    if rows < 1 or rows > 4:
        raise ValueError("rows must be between 1 and 4")
    if in_features != 5120:
        raise ValueError("in_features must equal 5120")
    if num_k_heads != 16:
        raise ValueError("num_k_heads must equal 16")
    if num_v_heads != 48:
        raise ValueError("num_v_heads must equal 48")
    if head_k_dim != 128:
        raise ValueError("head_k_dim must equal 128")
    if head_v_dim != 128:
        raise ValueError("head_v_dim must equal 128")
    if int(threads) != 256:
        raise ValueError("threads must equal 256")

    pointer_args = (
        norm_ptr,
        alpha_weight_ptr,
        beta_weight_ptr,
        alpha_out_ptr,
        beta_out_ptr,
        conv_out_ptr,
        gate_ptr,
        dt_bias_ptr,
        a_log_ptr,
        norm_weight_ptr,
        base_recurrent_state_ptr,
        leaf_recurrent_state_ptr,
        initial_recurrent_state_snapshot_ptr,
        out_ptr,
        out_bf16_ptr,
    )
    if any(int(pointer) <= 0 for pointer in pointer_args):
        raise ValueError("dependent alpha/beta-to-GDN pointers must be non-zero")
    library = library or build_qwen35_linear_attn_gdn(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_ALPHA_BETA_GDN_CHAIN_SNAPSHOT_F32_BF16)
    fn.argtypes = [ctypes.c_void_p] * len(pointer_args) + [
        ctypes.c_float,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        *(ctypes.c_void_p(pointer) for pointer in pointer_args),
        ctypes.c_float(eps),
        ctypes.c_int64(rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(num_k_heads),
        ctypes.c_int64(num_v_heads),
        ctypes.c_int64(head_k_dim),
        ctypes.c_int64(head_v_dim),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


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


def qwen35_gdn_prefill_recurrent_normalized_wave32_xor_f32(
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
    """Launch the llama.cpp-shaped normalized-Q/K wave32 recurrence."""

    _launch_prefill_recurrent(
        _SYMBOL_PREFILL_NORMALIZED_WAVE32_XOR,
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


def qwen35_gdn_prefill_recurrent_compact_normalized_wave32_xor_f32(
    query_ptr: int,
    key_ptr: int,
    value_ptr: int,
    beta_ptr: int,
    decay_ptr: int,
    recurrent_state_ptr: int,
    out_ptr: int,
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
    """Launch peer-wave32 recurrence over compact per-K-head Q/K."""

    _check_prefill_shape(tokens, num_k_heads, num_v_heads, head_k_dim, head_v_dim)
    library = library or build_qwen35_linear_attn_gdn(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_PREFILL_COMPACT_NORMALIZED_WAVE32_XOR)
    fn.argtypes = [ctypes.c_void_p] * 7 + [ctypes.c_int64] * 5 + [ctypes.c_void_p]
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
        ctypes.c_int64(num_k_heads),
        ctypes.c_int64(num_v_heads),
        ctypes.c_int64(head_k_dim),
        ctypes.c_int64(head_v_dim),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def qwen35_gdn_prefill_recurrent_compact_normalized_wave32_xor_fp16state(
    query_ptr: int,
    key_ptr: int,
    value_ptr: int,
    beta_ptr: int,
    decay_ptr: int,
    recurrent_state_ptr: int,
    out_ptr: int,
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
    """Launch compact peer-wave32 recurrence with fp16-state storage.

    Production fp16-state route: ``recurrent_state`` is a ``half`` buffer read
    and written through the templated kernel (fp32 accumulate inside, RNE round
    on store).  The FP32-state ``...xor_f32`` wrapper remains the registered
    strict fallback.
    """

    _check_prefill_shape(tokens, num_k_heads, num_v_heads, head_k_dim, head_v_dim)
    library = library or build_qwen35_linear_attn_gdn(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_PREFILL_COMPACT_NORMALIZED_WAVE32_XOR_FP16STATE)
    fn.argtypes = [ctypes.c_void_p] * 7 + [ctypes.c_int64] * 5 + [ctypes.c_void_p]
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
        ctypes.c_int64(num_k_heads),
        ctypes.c_int64(num_v_heads),
        ctypes.c_int64(head_k_dim),
        ctypes.c_int64(head_v_dim),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def _launch_gdn_prefill_recurrent_compact_normalized_segments_wave32_xor(
    symbol: str,
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
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    _check_prefill_shape(
        total_tokens,
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
    )
    if segments <= 0:
        raise ValueError("segments must be positive")
    library = library or build_qwen35_linear_attn_gdn(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, symbol)
    fn.argtypes = (
        [ctypes.c_void_p] * 9
        + [ctypes.c_int64] * 6
        + [ctypes.c_void_p]
    )
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
        ctypes.c_int64(num_k_heads),
        ctypes.c_int64(num_v_heads),
        ctypes.c_int64(head_k_dim),
        ctypes.c_int64(head_v_dim),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def qwen35_gdn_prefill_recurrent_compact_normalized_segments_wave32_xor_f32(
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
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch indexed compact peer-wave32 prefill with FP32 state."""

    _launch_gdn_prefill_recurrent_compact_normalized_segments_wave32_xor(
        _SYMBOL_PREFILL_COMPACT_NORMALIZED_SEGMENTS_WAVE32_XOR,
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
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def qwen35_gdn_prefill_recurrent_compact_normalized_segments_wave32_xor_fp16state(
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
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch indexed compact peer-wave32 prefill with FP16 state storage."""

    _launch_gdn_prefill_recurrent_compact_normalized_segments_wave32_xor(
        _SYMBOL_PREFILL_COMPACT_NORMALIZED_SEGMENTS_WAVE32_XOR_FP16STATE,
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
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def qwen35_gdn_prefill_recurrent_normalized_cluster8_f32(
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
    """Launch the Vulkan-shaped normalized-Q/K eight-lane recurrence."""

    _launch_prefill_recurrent(
        _SYMBOL_PREFILL_NORMALIZED_CLUSTER8,
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


def qwen35_gdn_prefill_recurrent_k2_decode_order_f32(
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
    """Launch state-parallel K2 prefill with decode-order K reduction."""

    _launch_prefill_recurrent(
        _SYMBOL_PREFILL_K2_DECODE_ORDER,
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
        _SYMBOL_PREFILL_SEGMENTS_K2,
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


def qwen35_gdn_prefill_recurrent_normalized_segments_wave32_xor_f32(
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
    """Launch the segment-aware normalized-Q/K wave32 XOR recurrence."""

    _launch_prefill_recurrent_segments(
        _SYMBOL_PREFILL_NORMALIZED_SEGMENTS_WAVE32_XOR,
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


def qwen35_gdn_prefill_recurrent_normalized_segments_cluster8_f32(
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
    """Launch the segment-aware normalized-Q/K eight-lane recurrence."""

    _launch_prefill_recurrent_segments(
        _SYMBOL_PREFILL_NORMALIZED_SEGMENTS_CLUSTER8,
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
    """Prepare output-scaled Q, normalized K, value, beta, and decay."""

    _launch_linear_attn_prefill_prepare(
        _SYMBOL_PREFILL_PREPARE,
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


def qwen35_linear_attn_prefill_prepare_peer_normalized_f32_bf16(
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
    """Prepare unit-normalized Q/K for the peer wave32 recurrence."""

    _launch_linear_attn_prefill_prepare(
        _SYMBOL_PREFILL_PREPARE_PEER_NORMALIZED,
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


def qwen35_linear_attn_prefill_prepare_compact_peer_normalized_f32_bf16(
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
    """Prepare per-K-head unit Q/K plus per-V-head value/beta/decay."""

    _launch_linear_attn_prefill_prepare(
        _SYMBOL_PREFILL_PREPARE_COMPACT_PEER_NORMALIZED,
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


def qwen35_linear_attn_prefill_prepare_decode_order_f32_bf16(
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
    """Prepare GDN tensors with the fused c=1 Q/K reduction order."""

    _launch_linear_attn_prefill_prepare(
        _SYMBOL_PREFILL_PREPARE_DECODE_ORDER,
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


def qwen35_linear_attn_prefill_prepare_raw_scales_f32_bf16(
    conv_out_ptr: int,
    a_ptr: int,
    b_ptr: int,
    dt_bias_ptr: int,
    a_log_ptr: int,
    query_raw_ptr: int,
    key_raw_ptr: int,
    value_ptr: int,
    beta_ptr: int,
    decay_ptr: int,
    query_scale_ptr: int,
    key_scale_ptr: int,
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
    """Prepare raw Q/K plus scales for byte-exact GGUF split recurrence."""

    _check_exact_prefill_shape(
        tokens, num_k_heads, num_v_heads, head_k_dim, head_v_dim
    )
    library = library or build_qwen35_linear_attn_gdn(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_PREFILL_PREPARE_RAW_SCALES)
    fn.argtypes = [ctypes.c_void_p] * 12 + [ctypes.c_int64] * 5 + [ctypes.c_void_p]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(conv_out_ptr),
        ctypes.c_void_p(a_ptr),
        ctypes.c_void_p(b_ptr),
        ctypes.c_void_p(dt_bias_ptr),
        ctypes.c_void_p(a_log_ptr),
        ctypes.c_void_p(query_raw_ptr),
        ctypes.c_void_p(key_raw_ptr),
        ctypes.c_void_p(value_ptr),
        ctypes.c_void_p(beta_ptr),
        ctypes.c_void_p(decay_ptr),
        ctypes.c_void_p(query_scale_ptr),
        ctypes.c_void_p(key_scale_ptr),
        ctypes.c_int64(tokens),
        ctypes.c_int64(num_k_heads),
        ctypes.c_int64(num_v_heads),
        ctypes.c_int64(head_k_dim),
        ctypes.c_int64(head_v_dim),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def qwen35_linear_attn_prefill_prepare_compact_scales_f32_bf16(
    conv_out_ptr: int,
    a_ptr: int,
    b_ptr: int,
    dt_bias_ptr: int,
    a_log_ptr: int,
    beta_ptr: int,
    decay_ptr: int,
    query_scale_ptr: int,
    key_scale_ptr: int,
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
    """Prepare compact per-K-head scales plus per-V-head recurrence scalars."""

    _check_exact_prefill_shape(
        tokens, num_k_heads, num_v_heads, head_k_dim, head_v_dim
    )
    library = library or build_qwen35_linear_attn_gdn(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_PREFILL_PREPARE_COMPACT_SCALES)
    fn.argtypes = [ctypes.c_void_p] * 9 + [ctypes.c_int64] * 5 + [ctypes.c_void_p]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(conv_out_ptr),
        ctypes.c_void_p(a_ptr),
        ctypes.c_void_p(b_ptr),
        ctypes.c_void_p(dt_bias_ptr),
        ctypes.c_void_p(a_log_ptr),
        ctypes.c_void_p(beta_ptr),
        ctypes.c_void_p(decay_ptr),
        ctypes.c_void_p(query_scale_ptr),
        ctypes.c_void_p(key_scale_ptr),
        ctypes.c_int64(tokens),
        ctypes.c_int64(num_k_heads),
        ctypes.c_int64(num_v_heads),
        ctypes.c_int64(head_k_dim),
        ctypes.c_int64(head_v_dim),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def _qwen35_gdn_prefill_recurrent_decode_order_exact_f32(
    symbol: str,
    query_raw_ptr: int,
    key_raw_ptr: int,
    value_ptr: int,
    beta_ptr: int,
    decay_ptr: int,
    query_scale_ptr: int,
    key_scale_ptr: int,
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
    """Launch one exact decode-order recurrence symbol."""

    _check_exact_recurrent_shape(tokens, num_v_heads, head_k_dim, head_v_dim)
    library = library or build_qwen35_linear_attn_gdn(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, symbol)
    fn.argtypes = [ctypes.c_void_p] * 9 + [ctypes.c_int64] * 4 + [ctypes.c_void_p]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(query_raw_ptr),
        ctypes.c_void_p(key_raw_ptr),
        ctypes.c_void_p(value_ptr),
        ctypes.c_void_p(beta_ptr),
        ctypes.c_void_p(decay_ptr),
        ctypes.c_void_p(query_scale_ptr),
        ctypes.c_void_p(key_scale_ptr),
        ctypes.c_void_p(recurrent_state_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(tokens),
        ctypes.c_int64(num_v_heads),
        ctypes.c_int64(head_k_dim),
        ctypes.c_int64(head_v_dim),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def qwen35_gdn_prefill_recurrent_decode_order_exact_f32(
    query_raw_ptr: int,
    key_raw_ptr: int,
    value_ptr: int,
    beta_ptr: int,
    decay_ptr: int,
    query_scale_ptr: int,
    key_scale_ptr: int,
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
    """Run the 128-column exact decode-order recurrence."""

    _qwen35_gdn_prefill_recurrent_decode_order_exact_f32(
        _SYMBOL_PREFILL_DECODE_ORDER_EXACT,
        query_raw_ptr,
        key_raw_ptr,
        value_ptr,
        beta_ptr,
        decay_ptr,
        query_scale_ptr,
        key_scale_ptr,
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


def qwen35_gdn_prefill_recurrent_decode_order_exact_tile64_f32(
    query_raw_ptr: int,
    key_raw_ptr: int,
    value_ptr: int,
    beta_ptr: int,
    decay_ptr: int,
    query_scale_ptr: int,
    key_scale_ptr: int,
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
    """Run the exact recurrence with 64 independent value columns per block."""

    _qwen35_gdn_prefill_recurrent_decode_order_exact_f32(
        _SYMBOL_PREFILL_DECODE_ORDER_EXACT_TILE64,
        query_raw_ptr,
        key_raw_ptr,
        value_ptr,
        beta_ptr,
        decay_ptr,
        query_scale_ptr,
        key_scale_ptr,
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


def qwen35_gdn_prefill_recurrent_decode_order_exact_tile32_f32(
    query_raw_ptr: int,
    key_raw_ptr: int,
    value_ptr: int,
    beta_ptr: int,
    decay_ptr: int,
    query_scale_ptr: int,
    key_scale_ptr: int,
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
    """Run the exact recurrence with 32 independent value columns per block."""

    _qwen35_gdn_prefill_recurrent_decode_order_exact_f32(
        _SYMBOL_PREFILL_DECODE_ORDER_EXACT_TILE32,
        query_raw_ptr,
        key_raw_ptr,
        value_ptr,
        beta_ptr,
        decay_ptr,
        query_scale_ptr,
        key_scale_ptr,
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


def qwen35_gdn_prefill_recurrent_decode_order_exact_lds64_f32(
    query_raw_ptr: int,
    key_raw_ptr: int,
    value_ptr: int,
    beta_ptr: int,
    decay_ptr: int,
    query_scale_ptr: int,
    key_scale_ptr: int,
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
    """Run the scalar-exact recurrence with a 64-column LDS state tile."""

    _qwen35_gdn_prefill_recurrent_decode_order_exact_f32(
        _SYMBOL_PREFILL_DECODE_ORDER_EXACT_LDS64,
        query_raw_ptr,
        key_raw_ptr,
        value_ptr,
        beta_ptr,
        decay_ptr,
        query_scale_ptr,
        key_scale_ptr,
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


def qwen35_gdn_prefill_recurrent_decode_order_exact_lds32_f32(
    query_raw_ptr: int,
    key_raw_ptr: int,
    value_ptr: int,
    beta_ptr: int,
    decay_ptr: int,
    query_scale_ptr: int,
    key_scale_ptr: int,
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
    """Run the scalar-exact recurrence with a 32-column LDS state tile."""

    _qwen35_gdn_prefill_recurrent_decode_order_exact_f32(
        _SYMBOL_PREFILL_DECODE_ORDER_EXACT_LDS32,
        query_raw_ptr,
        key_raw_ptr,
        value_ptr,
        beta_ptr,
        decay_ptr,
        query_scale_ptr,
        key_scale_ptr,
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


def qwen35_gdn_prefill_recurrent_decode_order_exact_lds32_direct_f32(
    conv_out_ptr: int,
    beta_ptr: int,
    decay_ptr: int,
    query_scale_ptr: int,
    key_scale_ptr: int,
    recurrent_state_ptr: int,
    out_ptr: int,
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
    """Run exact LDS32 recurrence by reading raw Q/K/V from conv_out."""

    _check_exact_prefill_shape(
        tokens, num_k_heads, num_v_heads, head_k_dim, head_v_dim
    )
    library = library or build_qwen35_linear_attn_gdn(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_PREFILL_DECODE_ORDER_EXACT_LDS32_DIRECT)
    fn.argtypes = [ctypes.c_void_p] * 7 + [ctypes.c_int64] * 5 + [ctypes.c_void_p]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(conv_out_ptr),
        ctypes.c_void_p(beta_ptr),
        ctypes.c_void_p(decay_ptr),
        ctypes.c_void_p(query_scale_ptr),
        ctypes.c_void_p(key_scale_ptr),
        ctypes.c_void_p(recurrent_state_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(tokens),
        ctypes.c_int64(num_k_heads),
        ctypes.c_int64(num_v_heads),
        ctypes.c_int64(head_k_dim),
        ctypes.c_int64(head_v_dim),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def qwen35_gdn_prefill_recurrent_decode_order_exact_lds32_direct_nonvolatile_f32(
    conv_out_ptr: int,
    beta_ptr: int,
    decay_ptr: int,
    query_scale_ptr: int,
    key_scale_ptr: int,
    recurrent_state_ptr: int,
    out_ptr: int,
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
    """Run exact LDS32 recurrence with compiler-cacheable state accesses."""

    _check_exact_prefill_shape(
        tokens, num_k_heads, num_v_heads, head_k_dim, head_v_dim
    )
    library = library or build_qwen35_linear_attn_gdn(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_PREFILL_DECODE_ORDER_EXACT_LDS32_DIRECT_NONVOLATILE)
    fn.argtypes = [ctypes.c_void_p] * 7 + [ctypes.c_int64] * 5 + [ctypes.c_void_p]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(conv_out_ptr),
        ctypes.c_void_p(beta_ptr),
        ctypes.c_void_p(decay_ptr),
        ctypes.c_void_p(query_scale_ptr),
        ctypes.c_void_p(key_scale_ptr),
        ctypes.c_void_p(recurrent_state_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(tokens),
        ctypes.c_int64(num_k_heads),
        ctypes.c_int64(num_v_heads),
        ctypes.c_int64(head_k_dim),
        ctypes.c_int64(head_v_dim),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def qwen35_gdn_prefill_recurrent_decode_order_exact_wave32_f32(
    query_raw_ptr: int,
    key_raw_ptr: int,
    value_ptr: int,
    beta_ptr: int,
    decay_ptr: int,
    query_scale_ptr: int,
    key_scale_ptr: int,
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
    """Run the exact recurrence with one wave32 per value column."""

    _qwen35_gdn_prefill_recurrent_decode_order_exact_f32(
        _SYMBOL_PREFILL_DECODE_ORDER_EXACT_WAVE32,
        query_raw_ptr,
        key_raw_ptr,
        value_ptr,
        beta_ptr,
        decay_ptr,
        query_scale_ptr,
        key_scale_ptr,
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


def qwen35_gdn_prefill_recurrent_decode_order_wave32_tree_f32(
    query_raw_ptr: int,
    key_raw_ptr: int,
    value_ptr: int,
    beta_ptr: int,
    decay_ptr: int,
    query_scale_ptr: int,
    key_scale_ptr: int,
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
    """Run the tree-reduced recurrence with one wave32 per value column."""

    _qwen35_gdn_prefill_recurrent_decode_order_exact_f32(
        _SYMBOL_PREFILL_DECODE_ORDER_WAVE32_TREE,
        query_raw_ptr,
        key_raw_ptr,
        value_ptr,
        beta_ptr,
        decay_ptr,
        query_scale_ptr,
        key_scale_ptr,
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


def _qwen35_gdn_prefill_recurrent_decode_order_exact_segments_f32(
    symbol: str,
    query_raw_ptr: int,
    key_raw_ptr: int,
    value_ptr: int,
    beta_ptr: int,
    decay_ptr: int,
    query_scale_ptr: int,
    key_scale_ptr: int,
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
    """Launch one segment-aware exact decode-order recurrence symbol."""

    _check_exact_recurrent_shape(
        total_tokens, num_v_heads, head_k_dim, head_v_dim
    )
    _check_positive(segments, "segments")
    library = library or build_qwen35_linear_attn_gdn(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, symbol)
    fn.argtypes = [ctypes.c_void_p] * 11 + [ctypes.c_int64] * 5 + [ctypes.c_void_p]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(query_raw_ptr),
        ctypes.c_void_p(key_raw_ptr),
        ctypes.c_void_p(value_ptr),
        ctypes.c_void_p(beta_ptr),
        ctypes.c_void_p(decay_ptr),
        ctypes.c_void_p(query_scale_ptr),
        ctypes.c_void_p(key_scale_ptr),
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


def qwen35_gdn_prefill_recurrent_decode_order_exact_segments_f32(
    query_raw_ptr: int,
    key_raw_ptr: int,
    value_ptr: int,
    beta_ptr: int,
    decay_ptr: int,
    query_scale_ptr: int,
    key_scale_ptr: int,
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
    """Run the 128-column segment-aware exact recurrence."""

    _qwen35_gdn_prefill_recurrent_decode_order_exact_segments_f32(
        _SYMBOL_PREFILL_DECODE_ORDER_EXACT_SEGMENTS,
        query_raw_ptr,
        key_raw_ptr,
        value_ptr,
        beta_ptr,
        decay_ptr,
        query_scale_ptr,
        key_scale_ptr,
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


def qwen35_gdn_prefill_recurrent_decode_order_exact_segments_tile64_f32(
    query_raw_ptr: int,
    key_raw_ptr: int,
    value_ptr: int,
    beta_ptr: int,
    decay_ptr: int,
    query_scale_ptr: int,
    key_scale_ptr: int,
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
    """Run segment-aware exact recurrence with 64 value columns per block."""

    _qwen35_gdn_prefill_recurrent_decode_order_exact_segments_f32(
        _SYMBOL_PREFILL_DECODE_ORDER_EXACT_SEGMENTS_TILE64,
        query_raw_ptr,
        key_raw_ptr,
        value_ptr,
        beta_ptr,
        decay_ptr,
        query_scale_ptr,
        key_scale_ptr,
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


def qwen35_gdn_prefill_recurrent_decode_order_exact_segments_tile32_f32(
    query_raw_ptr: int,
    key_raw_ptr: int,
    value_ptr: int,
    beta_ptr: int,
    decay_ptr: int,
    query_scale_ptr: int,
    key_scale_ptr: int,
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
    """Run segment-aware exact recurrence with 32 value columns per block."""

    _qwen35_gdn_prefill_recurrent_decode_order_exact_segments_f32(
        _SYMBOL_PREFILL_DECODE_ORDER_EXACT_SEGMENTS_TILE32,
        query_raw_ptr,
        key_raw_ptr,
        value_ptr,
        beta_ptr,
        decay_ptr,
        query_scale_ptr,
        key_scale_ptr,
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


def qwen35_gdn_prefill_recurrent_decode_order_exact_segments_lds64_f32(
    query_raw_ptr: int,
    key_raw_ptr: int,
    value_ptr: int,
    beta_ptr: int,
    decay_ptr: int,
    query_scale_ptr: int,
    key_scale_ptr: int,
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
    """Run segment-aware scalar-exact recurrence with a 64-column LDS tile."""

    _qwen35_gdn_prefill_recurrent_decode_order_exact_segments_f32(
        _SYMBOL_PREFILL_DECODE_ORDER_EXACT_SEGMENTS_LDS64,
        query_raw_ptr,
        key_raw_ptr,
        value_ptr,
        beta_ptr,
        decay_ptr,
        query_scale_ptr,
        key_scale_ptr,
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


def qwen35_gdn_prefill_recurrent_decode_order_exact_segments_lds32_f32(
    query_raw_ptr: int,
    key_raw_ptr: int,
    value_ptr: int,
    beta_ptr: int,
    decay_ptr: int,
    query_scale_ptr: int,
    key_scale_ptr: int,
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
    """Run segment-aware scalar-exact recurrence with a 32-column LDS tile."""

    _qwen35_gdn_prefill_recurrent_decode_order_exact_segments_f32(
        _SYMBOL_PREFILL_DECODE_ORDER_EXACT_SEGMENTS_LDS32,
        query_raw_ptr,
        key_raw_ptr,
        value_ptr,
        beta_ptr,
        decay_ptr,
        query_scale_ptr,
        key_scale_ptr,
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


def qwen35_gdn_prefill_recurrent_decode_order_exact_segments_lds32_direct_f32(
    conv_out_ptr: int,
    beta_ptr: int,
    decay_ptr: int,
    query_scale_ptr: int,
    key_scale_ptr: int,
    recurrent_state_ptr: int,
    out_ptr: int,
    cu_seqlens_ptr: int,
    state_indices_ptr: int,
    total_tokens: int,
    segments: int,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Run segment-aware exact LDS32 recurrence directly from conv_out."""

    _check_exact_prefill_shape(
        total_tokens, num_k_heads, num_v_heads, head_k_dim, head_v_dim
    )
    if segments <= 0:
        raise ValueError(f"segments must be positive, got {segments}")
    library = library or build_qwen35_linear_attn_gdn(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_PREFILL_DECODE_ORDER_EXACT_SEGMENTS_LDS32_DIRECT)
    fn.argtypes = [ctypes.c_void_p] * 9 + [ctypes.c_int64] * 6 + [ctypes.c_void_p]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(conv_out_ptr),
        ctypes.c_void_p(beta_ptr),
        ctypes.c_void_p(decay_ptr),
        ctypes.c_void_p(query_scale_ptr),
        ctypes.c_void_p(key_scale_ptr),
        ctypes.c_void_p(recurrent_state_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_void_p(cu_seqlens_ptr),
        ctypes.c_void_p(state_indices_ptr),
        ctypes.c_int64(total_tokens),
        ctypes.c_int64(segments),
        ctypes.c_int64(num_k_heads),
        ctypes.c_int64(num_v_heads),
        ctypes.c_int64(head_k_dim),
        ctypes.c_int64(head_v_dim),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def qwen35_gdn_prefill_recurrent_decode_order_exact_segments_lds32_direct_nonvolatile_f32(
    conv_out_ptr: int,
    beta_ptr: int,
    decay_ptr: int,
    query_scale_ptr: int,
    key_scale_ptr: int,
    recurrent_state_ptr: int,
    out_ptr: int,
    cu_seqlens_ptr: int,
    state_indices_ptr: int,
    total_tokens: int,
    segments: int,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Run segment-aware exact LDS32 recurrence with cacheable state."""

    _check_exact_prefill_shape(
        total_tokens, num_k_heads, num_v_heads, head_k_dim, head_v_dim
    )
    if segments <= 0:
        raise ValueError(f"segments must be positive, got {segments}")
    library = library or build_qwen35_linear_attn_gdn(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(
        library,
        _SYMBOL_PREFILL_DECODE_ORDER_EXACT_SEGMENTS_LDS32_DIRECT_NONVOLATILE,
    )
    fn.argtypes = [ctypes.c_void_p] * 9 + [ctypes.c_int64] * 6 + [ctypes.c_void_p]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(conv_out_ptr),
        ctypes.c_void_p(beta_ptr),
        ctypes.c_void_p(decay_ptr),
        ctypes.c_void_p(query_scale_ptr),
        ctypes.c_void_p(key_scale_ptr),
        ctypes.c_void_p(recurrent_state_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_void_p(cu_seqlens_ptr),
        ctypes.c_void_p(state_indices_ptr),
        ctypes.c_int64(total_tokens),
        ctypes.c_int64(segments),
        ctypes.c_int64(num_k_heads),
        ctypes.c_int64(num_v_heads),
        ctypes.c_int64(head_k_dim),
        ctypes.c_int64(head_v_dim),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def qwen35_gdn_prefill_recurrent_decode_order_exact_segments_wave32_f32(
    query_raw_ptr: int,
    key_raw_ptr: int,
    value_ptr: int,
    beta_ptr: int,
    decay_ptr: int,
    query_scale_ptr: int,
    key_scale_ptr: int,
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
    """Run the segment-aware exact recurrence with one wave32 per column."""

    _qwen35_gdn_prefill_recurrent_decode_order_exact_segments_f32(
        _SYMBOL_PREFILL_DECODE_ORDER_EXACT_SEGMENTS_WAVE32,
        query_raw_ptr,
        key_raw_ptr,
        value_ptr,
        beta_ptr,
        decay_ptr,
        query_scale_ptr,
        key_scale_ptr,
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


def qwen35_gdn_prefill_recurrent_decode_order_segments_wave32_tree_f32(
    query_raw_ptr: int,
    key_raw_ptr: int,
    value_ptr: int,
    beta_ptr: int,
    decay_ptr: int,
    query_scale_ptr: int,
    key_scale_ptr: int,
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
    """Run the segment-aware wave32 tree-reduced recurrence."""

    _qwen35_gdn_prefill_recurrent_decode_order_exact_segments_f32(
        _SYMBOL_PREFILL_DECODE_ORDER_SEGMENTS_WAVE32_TREE,
        query_raw_ptr,
        key_raw_ptr,
        value_ptr,
        beta_ptr,
        decay_ptr,
        query_scale_ptr,
        key_scale_ptr,
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


def qwen35_gdn_prefill_recurrent_rmsnorm_gate_bf16_decode_order_state_rows(
    conv_out_ptr: int,
    gate_ptr: int,
    a_ptr: int,
    b_ptr: int,
    dt_bias_ptr: int,
    a_log_ptr: int,
    norm_weight_ptr: int,
    recurrent_state_ptr: int,
    recurrent_state_rows_ptr: int,
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
    """Launch decode-order BF16 GDN prefill and materialize per-row states."""

    _check_prefill_shape(tokens, num_k_heads, num_v_heads, head_k_dim, head_v_dim)
    library = library or build_qwen35_linear_attn_gdn(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_PREFILL_DECODE_ORDER_STATE_ROWS_BF16)
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
        ctypes.c_void_p(recurrent_state_rows_ptr),
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


def qwen35_gdn_prefill_recurrent_rmsnorm_gate_bf16_decode_order_state_rows_no_copy(
    conv_out_ptr: int,
    gate_ptr: int,
    a_ptr: int,
    b_ptr: int,
    dt_bias_ptr: int,
    a_log_ptr: int,
    norm_weight_ptr: int,
    recurrent_state_initial_ptr: int,
    recurrent_state_rows_ptr: int,
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
    """Launch decode-order BF16 GDN prefill, writing rows without mutating state."""

    _check_prefill_shape(tokens, num_k_heads, num_v_heads, head_k_dim, head_v_dim)
    library = library or build_qwen35_linear_attn_gdn(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_PREFILL_DECODE_ORDER_STATE_ROWS_NO_COPY_BF16)
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
        ctypes.c_void_p(recurrent_state_initial_ptr),
        ctypes.c_void_p(recurrent_state_rows_ptr),
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


def qwen35_gdn_prefill_recurrent_rmsnorm_gate_bf16_decode_order_state_rows_no_copy_fp16state(
    conv_out_ptr: int,
    gate_ptr: int,
    a_ptr: int,
    b_ptr: int,
    dt_bias_ptr: int,
    a_log_ptr: int,
    norm_weight_ptr: int,
    recurrent_state_initial_ptr: int,
    recurrent_state_rows_ptr: int,
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
    """Launch fp16-state decode-order BF16 GDN prefill, writing rows.

    Identical row-state writer to the strict BF16 no-copy variant, but the
    initial state and the captured per-row state are fp16 storage with fp32
    accumulate (RNE half round-trip).
    """

    _check_prefill_shape(tokens, num_k_heads, num_v_heads, head_k_dim, head_v_dim)
    library = library or build_qwen35_linear_attn_gdn(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_PREFILL_DECODE_ORDER_STATE_ROWS_NO_COPY_BF16_FP16STATE)
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
        ctypes.c_void_p(recurrent_state_initial_ptr),
        ctypes.c_void_p(recurrent_state_rows_ptr),
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


def qwen35_gdn_prefill_recurrent_rmsnorm_gate_bf16_decode_order_segments_state_rows_no_copy(
    conv_out_ptr: int,
    gate_ptr: int,
    a_ptr: int,
    b_ptr: int,
    dt_bias_ptr: int,
    a_log_ptr: int,
    norm_weight_ptr: int,
    recurrent_state_initial_ptr: int,
    recurrent_state_rows_ptr: int,
    out_ptr: int,
    cu_seqlens_ptr: int,
    state_indices_ptr: int,
    eps: float,
    total_tokens: int,
    segments: int,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
    _symbol: str = _SYMBOL_PREFILL_DECODE_ORDER_SEGMENTS_STATE_ROWS_NO_COPY_BF16,
) -> None:
    """Launch segment-aware BF16 decode-order GDN with row-state capture.

    The input rows are packed, ``cu_seqlens`` maps each slot segment to its row
    range, and ``state_indices`` selects that segment's initial recurrent-state
    slot. The initial state is not mutated; every packed row writes its resulting
    recurrent state into ``recurrent_state_rows``.
    """

    _check_prefill_shape(total_tokens, num_k_heads, num_v_heads, head_k_dim, head_v_dim)
    if segments <= 0:
        raise ValueError("segments must be positive")
    library = library or build_qwen35_linear_attn_gdn(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _symbol)
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
        ctypes.c_void_p(recurrent_state_initial_ptr),
        ctypes.c_void_p(recurrent_state_rows_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_void_p(cu_seqlens_ptr),
        ctypes.c_void_p(state_indices_ptr),
        ctypes.c_float(eps),
        ctypes.c_int64(total_tokens),
        ctypes.c_int64(segments),
        ctypes.c_int64(num_k_heads),
        ctypes.c_int64(num_v_heads),
        ctypes.c_int64(head_k_dim),
        ctypes.c_int64(head_v_dim),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def qwen35_gdn_prefill_recurrent_rmsnorm_gate_bf16_decode_order_segments_state_rows_no_copy_wave_reduce(
    *args,
    **kwargs,
) -> None:
    """Launch the exact physical C5-C8 wave-reduced row-state candidate."""

    total_tokens = int(kwargs.get("total_tokens", args[13] if len(args) > 13 else 0))
    segments = int(kwargs.get("segments", args[14] if len(args) > 14 else 0))
    if 5 <= segments <= 8 and total_tokens in (3 * segments, 4 * segments):
        kwargs["_symbol"] = (
            _SYMBOL_PREFILL_DECODE_ORDER_SEGMENTS_STATE_ROWS_NO_COPY_WAVE_REDUCE_BF16
        )
    qwen35_gdn_prefill_recurrent_rmsnorm_gate_bf16_decode_order_segments_state_rows_no_copy(
        *args,
        **kwargs,
    )


def qwen35_gdn_prefill_recurrent_rmsnorm_gate_bf16_decode_order_segments_state_rows_no_copy_f32(
    conv_out_ptr: int, gate_ptr: int, a_ptr: int, b_ptr: int,
    dt_bias_ptr: int, a_log_ptr: int, norm_weight_ptr: int,
    recurrent_state_initial_ptr: int, recurrent_state_rows_ptr: int,
    out_ptr: int, out_f32_ptr: int, cu_seqlens_ptr: int, state_indices_ptr: int,
    eps: float, total_tokens: int, segments: int, num_k_heads: int,
    num_v_heads: int, head_k_dim: int, head_v_dim: int, *, stream: int = 0,
    library: ctypes.CDLL | None = None, runtime: HipRuntime | None = None,
) -> None:
    """Launch no-copy segmented GDN with BF16 and FP32 row outputs."""

    _check_prefill_shape(total_tokens, num_k_heads, num_v_heads, head_k_dim, head_v_dim)
    if segments <= 0:
        raise ValueError("segments must be positive")
    library = library or build_qwen35_linear_attn_gdn(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(
        library,
        _SYMBOL_PREFILL_DECODE_ORDER_SEGMENTS_STATE_ROWS_NO_COPY_F32_BF16,
    )
    fn.argtypes = [ctypes.c_void_p] * 13 + [
        ctypes.c_float, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    ptrs = (
        conv_out_ptr, gate_ptr, a_ptr, b_ptr, dt_bias_ptr, a_log_ptr,
        norm_weight_ptr, recurrent_state_initial_ptr, recurrent_state_rows_ptr,
        out_ptr, out_f32_ptr, cu_seqlens_ptr, state_indices_ptr,
    )
    err = fn(
        *(ctypes.c_void_p(value) for value in ptrs), ctypes.c_float(eps),
        ctypes.c_int64(total_tokens), ctypes.c_int64(segments),
        ctypes.c_int64(num_k_heads), ctypes.c_int64(num_v_heads),
        ctypes.c_int64(head_k_dim), ctypes.c_int64(head_v_dim),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def qwen35_gdn_prefill_recurrent_rmsnorm_gate_bf16_decode_order_segments_state_rows_no_copy_fp16state(
    conv_out_ptr: int,
    gate_ptr: int,
    a_ptr: int,
    b_ptr: int,
    dt_bias_ptr: int,
    a_log_ptr: int,
    norm_weight_ptr: int,
    recurrent_state_initial_ptr: int,
    recurrent_state_rows_ptr: int,
    out_ptr: int,
    cu_seqlens_ptr: int,
    state_indices_ptr: int,
    eps: float,
    total_tokens: int,
    segments: int,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch fp16-state segment-aware BF16 decode-order GDN with rows.

    Same packed/segment contract as the strict BF16 variant, but the initial
    and captured per-row state are fp16 storage with fp32 accumulate.
    """

    _check_prefill_shape(total_tokens, num_k_heads, num_v_heads, head_k_dim, head_v_dim)
    if segments <= 0:
        raise ValueError("segments must be positive")
    library = library or build_qwen35_linear_attn_gdn(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_PREFILL_DECODE_ORDER_SEGMENTS_STATE_ROWS_NO_COPY_BF16_FP16STATE)
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
        ctypes.c_void_p(recurrent_state_initial_ptr),
        ctypes.c_void_p(recurrent_state_rows_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_void_p(cu_seqlens_ptr),
        ctypes.c_void_p(state_indices_ptr),
        ctypes.c_float(eps),
        ctypes.c_int64(total_tokens),
        ctypes.c_int64(segments),
        ctypes.c_int64(num_k_heads),
        ctypes.c_int64(num_v_heads),
        ctypes.c_int64(head_k_dim),
        ctypes.c_int64(head_v_dim),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def qwen35_gdn_prefill_recurrent_rmsnorm_gate_bf16_decode_order_segments(
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
    eps: float,
    total_tokens: int,
    segments: int,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch segment-aware BF16 decode-order GDN and mutate per-slot state."""

    _check_prefill_shape(total_tokens, num_k_heads, num_v_heads, head_k_dim, head_v_dim)
    if segments <= 0:
        raise ValueError("segments must be positive")
    library = library or build_qwen35_linear_attn_gdn(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_PREFILL_DECODE_ORDER_SEGMENTS_BF16)
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
        ctypes.c_float(eps),
        ctypes.c_int64(total_tokens),
        ctypes.c_int64(segments),
        ctypes.c_int64(num_k_heads),
        ctypes.c_int64(num_v_heads),
        ctypes.c_int64(head_k_dim),
        ctypes.c_int64(head_v_dim),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def qwen35_gdn_prefill_recurrent_rmsnorm_gate_bf16_decode_order_segments_fp16state(
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
    eps: float,
    total_tokens: int,
    segments: int,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Segment-aware decode-order GDN, fp16 recurrent-state storage.

    Same contract as ``qwen35_gdn_prefill_recurrent_rmsnorm_gate_bf16_decode_order_segments``
    but the per-slot packed recurrent state is stored fp16 (fp32 accumulate,
    RNE round-trip) for the production fp16-state route.  The state buffer is
    half-sized relative to the strict fp32 writer.
    """

    _check_prefill_shape(total_tokens, num_k_heads, num_v_heads, head_k_dim, head_v_dim)
    if segments <= 0:
        raise ValueError("segments must be positive")
    library = library or build_qwen35_linear_attn_gdn(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_PREFILL_DECODE_ORDER_SEGMENTS_BF16_FP16STATE)
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
        ctypes.c_float(eps),
        ctypes.c_int64(total_tokens),
        ctypes.c_int64(segments),
        ctypes.c_int64(num_k_heads),
        ctypes.c_int64(num_v_heads),
        ctypes.c_int64(head_k_dim),
        ctypes.c_int64(head_v_dim),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def qwen35_gdn_recurrent_rmsnorm_gate_indexed_lowp_bf16(
    conv_out_ptr: int,
    gate_ptr: int,
    a_ptr: int,
    b_ptr: int,
    dt_bias_ptr: int,
    a_log_ptr: int,
    norm_weight_ptr: int,
    recurrent_state_ptr: int,
    out_ptr: int,
    state_indices_ptr: int,
    rows: int,
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
    """Launch one-token-per-row indexed BF16 recurrent GDN."""

    if rows <= 0:
        raise ValueError("rows must be positive")
    _check_prefill_shape(rows, num_k_heads, num_v_heads, head_k_dim, head_v_dim)
    library = library or build_qwen35_linear_attn_gdn(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_INDEXED_LOWP_BF16)
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
        ctypes.c_void_p(state_indices_ptr),
        ctypes.c_int64(rows),
        ctypes.c_float(eps),
        ctypes.c_int64(num_k_heads),
        ctypes.c_int64(num_v_heads),
        ctypes.c_int64(head_k_dim),
        ctypes.c_int64(head_v_dim),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def qwen35_gdn_recurrent_rmsnorm_gate_indexed_shared_statecache24_lowp_bf16(
    conv_out_ptr: int,
    gate_ptr: int,
    a_ptr: int,
    b_ptr: int,
    dt_bias_ptr: int,
    a_log_ptr: int,
    norm_weight_ptr: int,
    recurrent_state_ptr: int,
    out_ptr: int,
    state_indices_ptr: int,
    rows: int,
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
    """Launch gfx1151 indexed BF16 GDN with a 24-row shared state cache."""

    if rows <= 0:
        raise ValueError("rows must be positive")
    _check_prefill_shape(rows, num_k_heads, num_v_heads, head_k_dim, head_v_dim)
    if rows < 8:
        qwen35_gdn_recurrent_rmsnorm_gate_indexed_lowp_bf16(
            conv_out_ptr,
            gate_ptr,
            a_ptr,
            b_ptr,
            dt_bias_ptr,
            a_log_ptr,
            norm_weight_ptr,
            recurrent_state_ptr,
            out_ptr,
            state_indices_ptr,
            rows,
            eps,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            stream=stream,
            library=library,
            runtime=runtime,
        )
        return
    library = library or build_qwen35_linear_attn_gdn(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_INDEXED_SHARED_STATECACHE24_LOWP_BF16)
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
        ctypes.c_void_p(state_indices_ptr),
        ctypes.c_int64(rows),
        ctypes.c_float(eps),
        ctypes.c_int64(num_k_heads),
        ctypes.c_int64(num_v_heads),
        ctypes.c_int64(head_k_dim),
        ctypes.c_int64(head_v_dim),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def qwen35_gdn_recurrent_rmsnorm_gate_indexed_lowp_bf16_fp16state(
    conv_out_ptr: int,
    gate_ptr: int,
    a_ptr: int,
    b_ptr: int,
    dt_bias_ptr: int,
    a_log_ptr: int,
    norm_weight_ptr: int,
    recurrent_state_ptr: int,
    out_ptr: int,
    state_indices_ptr: int,
    rows: int,
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
    """Indexed one-token-per-row BF16 GDN with fp16 recurrent-state storage.

    Same contract as ``qwen35_gdn_recurrent_rmsnorm_gate_indexed_lowp_bf16``
    but the per-slot recurrent state is fp16 (fp32 accumulate, RNE round-trip)
    for the production fp16-state route.  The state buffer is half-sized.
    """

    if rows <= 0:
        raise ValueError("rows must be positive")
    _check_prefill_shape(rows, num_k_heads, num_v_heads, head_k_dim, head_v_dim)
    library = library or build_qwen35_linear_attn_gdn(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_INDEXED_LOWP_BF16_FP16STATE)
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
        ctypes.c_void_p(state_indices_ptr),
        ctypes.c_int64(rows),
        ctypes.c_float(eps),
        ctypes.c_int64(num_k_heads),
        ctypes.c_int64(num_v_heads),
        ctypes.c_int64(head_k_dim),
        ctypes.c_int64(head_v_dim),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def qwen35_gdn_recurrent_rmsnorm_gate_indexed_shared_statecache24_lowp_bf16_fp16state(
    conv_out_ptr: int,
    gate_ptr: int,
    a_ptr: int,
    b_ptr: int,
    dt_bias_ptr: int,
    a_log_ptr: int,
    norm_weight_ptr: int,
    recurrent_state_ptr: int,
    out_ptr: int,
    state_indices_ptr: int,
    rows: int,
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
    """gfx1151 indexed BF16 GDN with 24-row shared state cache, fp16 state.

    Same dispatch policy as the strict wrapper: rows < 8 delegate to the
    plain fp16 indexed wrapper; otherwise the shared-statecache kernel is
    used with fp16-state storage (half-sized state buffer).
    """

    if rows <= 0:
        raise ValueError("rows must be positive")
    _check_prefill_shape(rows, num_k_heads, num_v_heads, head_k_dim, head_v_dim)
    if rows < 8:
        qwen35_gdn_recurrent_rmsnorm_gate_indexed_lowp_bf16_fp16state(
            conv_out_ptr,
            gate_ptr,
            a_ptr,
            b_ptr,
            dt_bias_ptr,
            a_log_ptr,
            norm_weight_ptr,
            recurrent_state_ptr,
            out_ptr,
            state_indices_ptr,
            rows,
            eps,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            stream=stream,
            library=library,
            runtime=runtime,
        )
        return
    library = library or build_qwen35_linear_attn_gdn(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_INDEXED_SHARED_STATECACHE24_LOWP_BF16_FP16STATE)
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
        ctypes.c_void_p(state_indices_ptr),
        ctypes.c_int64(rows),
        ctypes.c_float(eps),
        ctypes.c_int64(num_k_heads),
        ctypes.c_int64(num_v_heads),
        ctypes.c_int64(head_k_dim),
        ctypes.c_int64(head_v_dim),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def qwen35_gdn_recurrent_rmsnorm_gate_segments_lowp_bf16(
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
    """Launch segmented BF16-gated decode-order recurrent GDN kernel."""

    _check_prefill_shape(total_tokens, num_k_heads, num_v_heads, head_k_dim, head_v_dim)
    if segments <= 0:
        raise ValueError("segments must be positive")
    library = library or build_qwen35_linear_attn_gdn(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_SEGMENTS_LOWP_BF16)
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


def qwen35_gdn_recurrent_rmsnorm_gate_segments_lowp_state_rows_bf16(
    conv_out_ptr: int, gate_ptr: int, a_ptr: int, b_ptr: int,
    dt_bias_ptr: int, a_log_ptr: int, norm_weight_ptr: int,
    recurrent_state_ptr: int, recurrent_state_rows_ptr: int, out_ptr: int,
    out_bf16_ptr: int, cu_seqlens_ptr: int, state_indices_ptr: int, total_tokens: int,
    segments: int, eps: float, num_k_heads: int, num_v_heads: int,
    head_k_dim: int, head_v_dim: int, *, stream: int = 0,
    library: ctypes.CDLL | None = None, runtime: HipRuntime | None = None,
) -> None:
    """Launch strict segmented GDN and publish every FP32 state row."""

    _check_prefill_shape(total_tokens, num_k_heads, num_v_heads, head_k_dim, head_v_dim)
    if segments <= 0:
        raise ValueError("segments must be positive")
    library = library or build_qwen35_linear_attn_gdn(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_SEGMENTS_LOWP_STATE_ROWS_BF16)
    fn.argtypes = [ctypes.c_void_p] * 13 + [
        ctypes.c_int64, ctypes.c_int64, ctypes.c_float,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    args = (
        conv_out_ptr, gate_ptr, a_ptr, b_ptr, dt_bias_ptr, a_log_ptr,
        norm_weight_ptr, recurrent_state_ptr, recurrent_state_rows_ptr, out_ptr,
        out_bf16_ptr, cu_seqlens_ptr, state_indices_ptr,
    )
    err = fn(
        *(ctypes.c_void_p(value) for value in args),
        ctypes.c_int64(total_tokens), ctypes.c_int64(segments), ctypes.c_float(eps),
        ctypes.c_int64(num_k_heads), ctypes.c_int64(num_v_heads),
        ctypes.c_int64(head_k_dim), ctypes.c_int64(head_v_dim),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def qwen35_gdn_recurrent_rmsnorm_gate_segments_lowp_bf16_fp16state(
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
    """Launch segmented BF16-gated decode-order recurrent GDN with fp16 state.

    Production fp16-state route: ``recurrent_state`` is a ``half`` buffer; the
    kernel reads/accumulates in fp32 and writes fp16 state back.  The
    FP32-state ``...segments_lowp_bf16`` wrapper remains the registered strict
    fallback.
    """

    _check_prefill_shape(total_tokens, num_k_heads, num_v_heads, head_k_dim, head_v_dim)
    if segments <= 0:
        raise ValueError("segments must be positive")
    library = library or build_qwen35_linear_attn_gdn(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_SEGMENTS_LOWP_BF16_FP16STATE)
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
        KernelKey(
            "hip_gfx1100",
            "gdn_recurrent_rmsnorm_gate+cast",
            "gguf_q5_k_t16_v1",
            "bf16_lowp_f32_bf16_out",
        ),
        qwen35_gdn_recurrent_rmsnorm_gate_lowp_f32_bf16_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "gdn_recurrent_rmsnorm_gate+cast",
            "gguf_q5_k_t16_v1",
            "bf16_lowp_f32_bf16_out_fp16state",
        ),
        qwen35_gdn_recurrent_rmsnorm_gate_lowp_f32_bf16_out_fp16state,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "gdn_recurrent_rmsnorm_gate",
            "gguf_qwen35",
            "bf16_fp16state",
        ),
        qwen35_gdn_recurrent_rmsnorm_gate_lowp_bf16_fp16state,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "gdn_recurrent_rmsnorm_gate", "w4_paro", "fp16_lowp"),
        qwen35_gdn_recurrent_rmsnorm_gate_lowp_fp16,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "gdn_recurrent_rmsnorm_gate", "gguf_qwen35", "bf16_segments"),
        qwen35_gdn_recurrent_rmsnorm_gate_segments_lowp_bf16,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100", "gdn_recurrent_rmsnorm_gate+state_rows",
            "gguf_qwen35", "bf16_segments_strict_order",
        ),
        qwen35_gdn_recurrent_rmsnorm_gate_segments_lowp_state_rows_bf16,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "gdn_recurrent_rmsnorm_gate",
            "gguf_qwen35",
            "bf16_segments_fp16state",
        ),
        qwen35_gdn_recurrent_rmsnorm_gate_segments_lowp_bf16_fp16state,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "gdn_chain_recurrent_rmsnorm_gate",
            "gguf_qwen35",
            "bf16_c1_exact_state_rows_tloop",
        ),
        qwen35_gdn_chain_recurrent_rmsnorm_gate_lowp_c1_exact_tloop_bf16,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "gdn_chain_recurrent_rmsnorm_gate",
            "gguf_qwen35",
            "bf16_c1_exact_state_rows_tloop_fp16state",
        ),
        qwen35_gdn_chain_recurrent_rmsnorm_gate_lowp_c1_exact_tloop_bf16_fp16state,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "gdn_chain_recurrent_rmsnorm_gate+cast",
            "gguf_q5_k_t16_v1",
            "bf16_c1_exact_state_rows_tloop_f32_bf16_out",
        ),
        qwen35_gdn_chain_recurrent_rmsnorm_gate_lowp_c1_exact_tloop_f32_bf16_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "gdn_chain_recurrent_rmsnorm_gate+snapshot",
            "gguf_qwen35",
            "bf16_c1_exact_state_rows_tloop",
        ),
        qwen35_gdn_chain_recurrent_rmsnorm_gate_lowp_c1_exact_snapshot_tloop_bf16,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "gdn_chain_recurrent_rmsnorm_gate+cast+snapshot",
            "gguf_q5_k_t16_v1",
            "bf16_c1_exact_state_rows_tloop_f32_bf16_out",
        ),
        qwen35_gdn_chain_recurrent_rmsnorm_gate_lowp_c1_exact_snapshot_tloop_f32_bf16_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "linear_attn_alpha_beta+gdn_chain_recurrent_rmsnorm_gate+cast+snapshot",
            "f32+gguf_q5_k_t16_v1",
            "bf16_k5120_n48_hk16_hv48_d128_exact_state_rows_tloop_f32_bf16_out",
        ),
        qwen35_linear_attn_alpha_beta_gdn_chain_snapshot_f32_bf16_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "gdn_recurrent_rmsnorm_gate",
            "gguf_qwen35",
            "bf16_indexed_singleton",
        ),
        qwen35_gdn_recurrent_rmsnorm_gate_indexed_lowp_bf16,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "gdn_recurrent_rmsnorm_gate",
            "gguf_qwen35",
            "bf16_indexed_singleton_fp16state",
        ),
        qwen35_gdn_recurrent_rmsnorm_gate_indexed_lowp_bf16_fp16state,
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
        KernelKey("hip_gfx1100", "gdn_prefill_recurrent", "w4_paro", "f32_k2_decode_order"),
        qwen35_gdn_prefill_recurrent_k2_decode_order_f32,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "gdn_prefill_recurrent", "w4_paro", "f32_decode_order_exact_lds32"),
        qwen35_gdn_prefill_recurrent_decode_order_exact_lds32_f32,
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
        KernelKey("hip_gfx1100", "linear_attn_prefill_prepare", "w4_paro", "f32_bf16_decode_order"),
        qwen35_linear_attn_prefill_prepare_decode_order_f32_bf16,
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
        KernelKey(
            "hip_gfx1100",
            "gdn_prefill_recurrent",
            "gguf_qwen35",
            "decode_order_bf16_state_rows_no_copy_fp16state",
        ),
        qwen35_gdn_prefill_recurrent_rmsnorm_gate_bf16_decode_order_state_rows_no_copy_fp16state,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "gdn_prefill_recurrent",
            "gguf_qwen35",
            "decode_order_bf16_segments_state_rows_no_copy",
        ),
        qwen35_gdn_prefill_recurrent_rmsnorm_gate_bf16_decode_order_segments_state_rows_no_copy,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "gdn_prefill_recurrent",
            "gguf_qwen35",
            "decode_order_bf16_segments_state_rows_no_copy_wave_reduce",
        ),
        qwen35_gdn_prefill_recurrent_rmsnorm_gate_bf16_decode_order_segments_state_rows_no_copy_wave_reduce,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "gdn_prefill_recurrent",
            "gguf_qwen35",
            "decode_order_bf16_segments_state_rows_no_copy_fp16state",
        ),
        qwen35_gdn_prefill_recurrent_rmsnorm_gate_bf16_decode_order_segments_state_rows_no_copy_fp16state,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "gdn_prefill_recurrent",
            "gguf_qwen35",
            "decode_order_bf16_segments",
        ),
        qwen35_gdn_prefill_recurrent_rmsnorm_gate_bf16_decode_order_segments,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "gdn_prefill_recurrent",
            "gguf_qwen35",
            "decode_order_bf16_segments_fp16state",
        ),
        qwen35_gdn_prefill_recurrent_rmsnorm_gate_bf16_decode_order_segments_fp16state,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "gdn_prefill_recurrent", "gguf_qwen35", "f32_k2"),
        qwen35_gdn_prefill_recurrent_k2_f32,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "gdn_prefill_recurrent", "gguf_qwen35", "f32_k2_decode_order"),
        qwen35_gdn_prefill_recurrent_k2_decode_order_f32,
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
        KernelKey(
            "hip_gfx1100",
            "linear_attn_prefill_prepare",
            "gguf_qwen35",
            "f32_peer_normalized_bf16",
        ),
        qwen35_linear_attn_prefill_prepare_peer_normalized_f32_bf16,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "linear_attn_prefill_prepare",
            "gguf_qwen35",
            "f32_compact_peer_normalized_bf16",
        ),
        qwen35_linear_attn_prefill_prepare_compact_peer_normalized_f32_bf16,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "linear_attn_prefill_prepare",
            "gguf_qwen35",
            "f32_bf16_raw_scales",
        ),
        qwen35_linear_attn_prefill_prepare_raw_scales_f32_bf16,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "linear_attn_prefill_prepare",
            "gguf_qwen35",
            "f32_bf16_compact_scales",
        ),
        qwen35_linear_attn_prefill_prepare_compact_scales_f32_bf16,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "gdn_prefill_recurrent",
            "gguf_qwen35",
            "f32_decode_order_exact",
        ),
        qwen35_gdn_prefill_recurrent_decode_order_exact_f32,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "gdn_prefill_recurrent",
            "gguf_qwen35",
            "f32_decode_order_exact_segments",
        ),
        qwen35_gdn_prefill_recurrent_decode_order_exact_segments_f32,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "gdn_prefill_recurrent",
            "gguf_qwen35",
            "f32_normalized_wave32_xor",
        ),
        qwen35_gdn_prefill_recurrent_normalized_wave32_xor_f32,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "gdn_prefill_recurrent",
            "gguf_qwen35",
            "f32_compact_normalized_wave32_xor",
        ),
        qwen35_gdn_prefill_recurrent_compact_normalized_wave32_xor_f32,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "gdn_prefill_recurrent",
            "gguf_qwen35",
            "f32_compact_normalized_wave32_xor_fp16state",
        ),
        qwen35_gdn_prefill_recurrent_compact_normalized_wave32_xor_fp16state,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "gdn_prefill_recurrent",
            "gguf_qwen35",
            "f32_compact_normalized_segments_wave32_xor",
        ),
        qwen35_gdn_prefill_recurrent_compact_normalized_segments_wave32_xor_f32,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "gdn_prefill_recurrent",
            "gguf_qwen35",
            "f32_compact_normalized_segments_wave32_xor_fp16state",
        ),
        qwen35_gdn_prefill_recurrent_compact_normalized_segments_wave32_xor_fp16state,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "gdn_prefill_recurrent",
            "gguf_qwen35",
            "f32_normalized_segments_wave32_xor",
        ),
        qwen35_gdn_prefill_recurrent_normalized_segments_wave32_xor_f32,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "gdn_prefill_recurrent",
            "gguf_qwen35",
            "f32_normalized_cluster8",
        ),
        qwen35_gdn_prefill_recurrent_normalized_cluster8_f32,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "gdn_prefill_recurrent",
            "gguf_qwen35",
            "f32_normalized_segments_cluster8",
        ),
        qwen35_gdn_prefill_recurrent_normalized_segments_cluster8_f32,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "gdn_prefill_recurrent",
            "gguf_qwen35",
            "f32_decode_order_exact_tile64",
        ),
        qwen35_gdn_prefill_recurrent_decode_order_exact_tile64_f32,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "gdn_prefill_recurrent",
            "gguf_qwen35",
            "f32_decode_order_exact_tile32",
        ),
        qwen35_gdn_prefill_recurrent_decode_order_exact_tile32_f32,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "gdn_prefill_recurrent",
            "gguf_qwen35",
            "f32_decode_order_exact_segments_tile64",
        ),
        qwen35_gdn_prefill_recurrent_decode_order_exact_segments_tile64_f32,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "gdn_prefill_recurrent",
            "gguf_qwen35",
            "f32_decode_order_exact_segments_tile32",
        ),
        qwen35_gdn_prefill_recurrent_decode_order_exact_segments_tile32_f32,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "gdn_prefill_recurrent",
            "gguf_qwen35",
            "f32_decode_order_exact_lds64",
        ),
        qwen35_gdn_prefill_recurrent_decode_order_exact_lds64_f32,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "gdn_prefill_recurrent",
            "gguf_qwen35",
            "f32_decode_order_exact_lds32",
        ),
        qwen35_gdn_prefill_recurrent_decode_order_exact_lds32_f32,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "gdn_prefill_recurrent",
            "gguf_qwen35",
            "f32_decode_order_exact_lds32_direct",
        ),
        qwen35_gdn_prefill_recurrent_decode_order_exact_lds32_direct_f32,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "gdn_prefill_recurrent",
            "gguf_qwen35",
            "f32_decode_order_exact_lds32_direct_nonvolatile",
        ),
        qwen35_gdn_prefill_recurrent_decode_order_exact_lds32_direct_nonvolatile_f32,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "gdn_prefill_recurrent",
            "gguf_qwen35",
            "f32_decode_order_exact_segments_lds64",
        ),
        qwen35_gdn_prefill_recurrent_decode_order_exact_segments_lds64_f32,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "gdn_prefill_recurrent",
            "gguf_qwen35",
            "f32_decode_order_exact_segments_lds32",
        ),
        qwen35_gdn_prefill_recurrent_decode_order_exact_segments_lds32_f32,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "gdn_prefill_recurrent",
            "gguf_qwen35",
            "f32_decode_order_exact_segments_lds32_direct",
        ),
        qwen35_gdn_prefill_recurrent_decode_order_exact_segments_lds32_direct_f32,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "gdn_prefill_recurrent",
            "gguf_qwen35",
            "f32_decode_order_exact_segments_lds32_direct_nonvolatile",
        ),
        qwen35_gdn_prefill_recurrent_decode_order_exact_segments_lds32_direct_nonvolatile_f32,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "gdn_prefill_recurrent",
            "gguf_qwen35",
            "f32_decode_order_exact_wave32",
        ),
        qwen35_gdn_prefill_recurrent_decode_order_exact_wave32_f32,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "gdn_prefill_recurrent",
            "gguf_qwen35",
            "f32_decode_order_exact_segments_wave32",
        ),
        qwen35_gdn_prefill_recurrent_decode_order_exact_segments_wave32_f32,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "gdn_prefill_recurrent",
            "gguf_qwen35",
            "f32_decode_order_wave32_tree",
        ),
        qwen35_gdn_prefill_recurrent_decode_order_wave32_tree_f32,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "gdn_prefill_recurrent",
            "gguf_qwen35",
            "f32_decode_order_segments_wave32_tree",
        ),
        qwen35_gdn_prefill_recurrent_decode_order_segments_wave32_tree_f32,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "gdn_prefill_rmsnorm_gate", "gguf_qwen35", "bf16"),
        qwen35_gdn_prefill_rmsnorm_gate_bf16,
        replace=replace,
    )
    # UD-Q3_K_M needs the split prefill chain and the retained BF16
    # recurrent-output boundary to preserve its resident decode contract.
    # Keep both on the quant plugin axis so Q4/Q8 retain the one-launch-shorter
    # F32 recurrent-output decode route.
    register(
        KernelKey(
            "hip_gfx1100",
            "gdn_decode_output_cast",
            "gguf_ud_q3_k_m",
            "f32_to_bf16_exact",
        ),
        f32_to_bf16,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "gdn_prefill_recurrent", "gguf_ud_q3_k_m", "decode_order_bf16"),
        qwen35_gdn_prefill_recurrent_rmsnorm_gate_bf16_decode_order,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "gdn_prefill_recurrent", "gguf_ud_q3_k_m", "f32_k2"),
        qwen35_gdn_prefill_recurrent_decode_order_exact_lds32_f32,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "linear_attn_prefill_prepare", "gguf_ud_q3_k_m", "f32_bf16"),
        qwen35_linear_attn_prefill_prepare_decode_order_f32_bf16,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "gdn_prefill_rmsnorm_gate", "gguf_ud_q3_k_m", "bf16"),
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
        register(
            KernelKey(backend, "gdn_chain_recurrent_rmsnorm_gate", "gguf_qwen35", "bf16_c1_exact_tloop"),
            qwen35_gdn_chain_recurrent_rmsnorm_gate_lowp_c1_exact_tloop_bf16,
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
    initial_state_snapshot_ptr: int | None = None,
    out_bf16_ptr: int | None = None,
) -> None:
    _check_positive(max_nodes, "max_nodes")
    _check_gdn_shape(num_k_heads, num_v_heads, head_k_dim, head_v_dim)
    if head_k_dim > 256 or head_k_dim % 64:
        raise ValueError("chain GDN t-loop requires head_k_dim divisible by 64 and <= 256")
    library = library or build_qwen35_linear_attn_gdn(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, symbol)
    pointer_args = [
        conv_out_ptr,
        gate_ptr,
        a_ptr,
        b_ptr,
        dt_bias_ptr,
        a_log_ptr,
        norm_weight_ptr,
        base_recurrent_state_ptr,
        leaf_recurrent_state_ptr,
    ]
    if initial_state_snapshot_ptr is not None:
        if int(initial_state_snapshot_ptr) <= 0:
            raise ValueError("initial_state_snapshot_ptr must be non-zero")
        pointer_args.append(initial_state_snapshot_ptr)
    pointer_args.extend((acc_buf_ptr, out_ptr))
    if out_bf16_ptr is not None:
        pointer_args.append(out_bf16_ptr)
    fn.argtypes = [ctypes.c_void_p] * len(pointer_args) + [
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
        *(ctypes.c_void_p(value) for value in pointer_args),
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
    symbol: str,
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


def _check_exact_prefill_shape(
    tokens: int,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
) -> None:
    _check_prefill_shape(
        tokens, num_k_heads, num_v_heads, head_k_dim, head_v_dim
    )
    _check_exact_recurrent_shape(tokens, num_v_heads, head_k_dim, head_v_dim)


def _check_exact_recurrent_shape(
    tokens: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
) -> None:
    _check_positive(tokens, "tokens")
    _check_positive(num_v_heads, "num_v_heads")
    if head_k_dim != 128:
        raise ValueError("exact decode-order GDN requires head_k_dim == 128")
    _check_positive(head_v_dim, "head_v_dim")
    if head_v_dim > 128:
        raise ValueError("head_v_dim must be <= 128")


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
