"""Laguna global and token-granular SWA ``KVLiveSpans`` kernels."""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.dtype import DType
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register
from hipengine.kvcache import KVLiveSpans

_SOURCE = Path(__file__).with_name("laguna_kv_attention.hip")
_OUTPUT_NAME = "laguna_kv_attention.so"
_SYMBOL_GLOBAL_HEAD_KV = "hipengine_laguna_global_head_rmsnorm_rope_write_kv_f32_bf16_spans"
_SYMBOL_GLOBAL_HEAD_KV_WAVE0_TREE = (
    "hipengine_laguna_global_head_rmsnorm_rope_write_kv_wave0_tree_f32_bf16_spans"
)
_SYMBOL_SWA_HEAD_KV = "hipengine_laguna_swa_head_rmsnorm_rope_write_kv_f32_bf16_spans"
_SYMBOL_GLOBAL_F16_PROJECTION_HEAD_KV = (
    "hipengine_laguna_global_f16_projection_head_kv_bf16_f32_spans"
)
_SYMBOL_SWA_F16_PROJECTION_HEAD_KV = (
    "hipengine_laguna_swa_f16_projection_head_kv_bf16_f32_spans"
)
_SYMBOL_GLOBAL_WRITE = "hipengine_laguna_global_write_kv_f32_bf16_spans"
_SYMBOL_GLOBAL_WRITE_ROWS = "hipengine_laguna_global_write_kv_rows_f32_bf16_spans"
_SYMBOL_GLOBAL_ATTENTION = "hipengine_laguna_global_attention_decode_bf16_spans"
_SYMBOL_GLOBAL_ATTENTION_SINGLE_PAGE = (
    "hipengine_laguna_global_attention_decode_single_page_bf16_spans"
)
_SYMBOL_GLOBAL_ATTENTION_SINGLE_PAGE_GATED = (
    "hipengine_laguna_global_attention_decode_single_page_softplus_gate_bf16_spans"
)
_SYMBOL_GLOBAL_ATTENTION_SPLIT_EXACT = (
    "hipengine_laguna_global_attention_decode_split_exact_bf16_spans"
)
_SYMBOL_GLOBAL_ATTENTION_SPLIT_EXACT_GATED = (
    "hipengine_laguna_global_attention_decode_split_exact_gated_bf16_spans"
)
_SYMBOL_GLOBAL_ATTENTION_SPLIT_EXACT_GATED_FIXEDSHAPE = (
    "hipengine_laguna_global_attention_decode_split_exact_gated_fixedshape_"
    "bf16_spans"
)
_SYMBOL_GLOBAL_ATTENTION_FUSED_EXACT_GATED_GQA1_FIXEDSHAPE = (
    "hipengine_laguna_global_attention_decode_fused_exact_gated_gqa1_"
    "fixedshape_bf16_spans"
)
_SYMBOL_GLOBAL_ATTENTION_FUSED_EXACT_GATED_GQA2_VSTAGE64_FIXEDSHAPE = (
    "hipengine_laguna_global_attention_decode_fused_exact_gated_gqa2_"
    "vstage64_fixedshape_bf16_spans"
)
_SYMBOL_GLOBAL_ATTENTION_FUSED_EXACT_GATED_GQA2_VSTAGE64_VEC16_FIXEDSHAPE = (
    "hipengine_laguna_global_attention_decode_fused_exact_gated_gqa2_"
    "vstage64_vec16_fixedshape_bf16_spans"
)
_SYMBOL_GLOBAL_ATTENTION_FUSED_EXACT_GATED_GQA2_VSTAGE64_VEC16_DIRECT_FIXEDSHAPE = (
    "hipengine_laguna_global_attention_decode_fused_exact_gated_gqa2_"
    "vstage64_vec16_direct_fixedshape_bf16_spans"
)
_SYMBOL_GLOBAL_ATTENTION_FUSED_EXACT_GATED_GQA2_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXEDSHAPE = (
    "hipengine_laguna_global_attention_decode_fused_exact_gated_gqa2_"
    "vstage64_vec16_direct_assume_exp_fixedshape_bf16_spans"
)
_SYMBOL_GLOBAL_ATTENTION_FUSED_EXACT_GATED_GQA2_EXP32_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXEDSHAPE = (
    "hipengine_laguna_global_attention_decode_fused_exact_gated_gqa2_exp32_"
    "vstage64_vec16_direct_assume_exp_fixedshape_bf16_spans"
)
_SYMBOL_GLOBAL_ATTENTION_FUSED_EXACT_GATED_MIXED32_EXP32_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXEDSHAPE = (
    "hipengine_laguna_global_attention_decode_fused_exact_gated_mixed32_exp32_"
    "vstage64_vec16_direct_assume_exp_fixedshape_bf16_spans"
)
_SYMBOL_GLOBAL_ATTENTION_FUSED_EXACT_GATED_MIXED32_EXP32_PRODUCER_MAX_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXEDSHAPE = (
    "hipengine_laguna_global_attention_decode_fused_exact_gated_mixed32_exp32_"
    "producer_max_vstage64_vec16_direct_assume_exp_fixedshape_bf16_spans"
)
_SYMBOL_GLOBAL_ATTENTION_FUSED_EXACT_GATED_MIXED32_EXP32_PRODUCER_MAX_DPP_QK_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXEDSHAPE = (
    "hipengine_laguna_global_attention_decode_fused_exact_gated_mixed32_exp32_"
    "producer_max_dpp_qk_vstage64_vec16_direct_assume_exp_fixedshape_bf16_spans"
)
_SYMBOL_GLOBAL_ATTENTION_FUSED_EXACT_GATED_MIXED32_EXP32_PRODUCER_MAX_DPP_QK_PROBABILITY_VEC4_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXEDSHAPE = (
    "hipengine_laguna_global_attention_decode_fused_exact_gated_mixed32_exp32_"
    "producer_max_dpp_qk_probability_vec4_vstage64_vec16_direct_assume_exp_"
    "fixedshape_bf16_spans"
)
_SYMBOL_GLOBAL_ATTENTION_FUSED_EXACT_GATED_MIXED32_EXP32_PRODUCER_MAX_DPP_QK_PROBABILITY_VEC4_PRENORM_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXEDSHAPE = (
    "hipengine_laguna_global_attention_decode_fused_exact_gated_mixed32_exp32_"
    "producer_max_dpp_qk_probability_vec4_prenorm_vstage64_vec16_direct_"
    "assume_exp_fixedshape_bf16_spans"
)
_SYMBOL_GLOBAL_ATTENTION_FUSED_EXACT_GATED_MIXED32_LOCAL512_EXP32_PRODUCER_MAX_DPP_QK_PROBABILITY_VEC4_PRENORM_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXEDSHAPE = (
    "hipengine_laguna_global_attention_decode_fused_exact_gated_mixed32_"
    "local512_exp32_producer_max_dpp_qk_probability_vec4_prenorm_vstage64_"
    "vec16_direct_assume_exp_fixedshape_bf16_spans"
)
_SYMBOL_GLOBAL_ATTENTION_FUSED_EXACT_GATED_MIXED40_LOCAL512_EXP32_PRODUCER_MAX_DPP_QK_PROBABILITY_VEC4_PRENORM_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXEDSHAPE = (
    "hipengine_laguna_global_attention_decode_fused_exact_gated_mixed40_"
    "local512_exp32_producer_max_dpp_qk_probability_vec4_prenorm_vstage64_"
    "vec16_direct_assume_exp_fixedshape_bf16_spans"
)
_SYMBOL_GLOBAL_ATTENTION_FUSED_EXACT_GATED_MIXED40_LOCAL512_EXP32_PRODUCER_MAX_DPP_QK_DENSE_PREFIX_PROBABILITY_VEC4_PRENORM_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXEDSHAPE = (
    "hipengine_laguna_global_attention_decode_fused_exact_gated_mixed40_"
    "local512_exp32_producer_max_dpp_qk_dense_prefix_probability_vec4_"
    "prenorm_vstage64_vec16_direct_assume_exp_fixedshape_bf16_spans"
)
_SYMBOL_GLOBAL_ATTENTION_FUSED_EXACT_GATED_MIXED40_LOCAL512_EXP32_PRODUCER_MAX_DPP_QK_DENSE_PREFIX_IDLE_DOUBLE_BUFFER_PROBABILITY_VEC4_PRENORM_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXEDSHAPE = (
    "hipengine_laguna_global_attention_decode_fused_exact_gated_mixed40_"
    "local512_exp32_producer_max_dpp_qk_dense_prefix_idle_double_buffer_"
    "probability_vec4_prenorm_vstage64_vec16_direct_assume_exp_fixedshape_"
    "bf16_spans"
)
_SYMBOL_GLOBAL_ATTENTION_FUSED_EXACT_GATED_MIXED40_LOCAL1024_EXP32_PRODUCER_MAX_DPP_QK_DENSE_PREFIX_IDLE_DOUBLE_BUFFER_PROBABILITY_VEC4_PRENORM_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXEDSHAPE = (
    "hipengine_laguna_global_attention_decode_fused_exact_gated_mixed40_"
    "local1024_exp32_producer_max_dpp_qk_dense_prefix_idle_double_buffer_"
    "probability_vec4_prenorm_vstage64_vec16_direct_assume_exp_fixedshape_"
    "bf16_spans"
)
_SYMBOL_GLOBAL_ATTENTION_WMMA_QK_THREE_TERM_MIXED32_EXP32_PRODUCER_MAX_EXACT_PV = (
    "hipengine_laguna_global_attention_decode_wmma_qk_three_term_mixed32_"
    "exp32_producer_max_exact_pv_bf16_spans"
)
_SYMBOL_GLOBAL_ATTENTION_WMMA_GQA6_K64_THREE_TERM_RAW_NUMERATOR = (
    "hipengine_laguna_global_attention_decode_wmma_gqa6_k64_three_term_"
    "raw_numerator_bf16_spans"
)
_SYMBOL_GLOBAL_PREFILL = "hipengine_laguna_global_attention_prefill_bf16_spans"
_SYMBOL_GLOBAL_PREFILL_QROW2_ONLINE = (
    "hipengine_laguna_global_attention_prefill_qrow2_online_bf16_spans"
)
_SYMBOL_GLOBAL_PREFILL_QROW4_ONLINE = (
    "hipengine_laguna_global_attention_prefill_qrow4_online_bf16_spans"
)
_SYMBOL_GLOBAL_PREFILL_QROW4_CACHED_ONLINE = (
    "hipengine_laguna_global_attention_prefill_qrow4_cached_online_bf16_spans"
)
_SYMBOL_GLOBAL_PREFILL_QROW4_CACHED_META_ONLINE = (
    "hipengine_laguna_global_attention_prefill_qrow4_cached_meta_online_bf16_spans"
)
_SYMBOL_GLOBAL_PREFILL_QROW6_CACHED_META_ONLINE = (
    "hipengine_laguna_global_attention_prefill_qrow6_cached_meta_online_bf16_spans"
)
_SYMBOL_GLOBAL_PREFILL_QROW4_DENSE_INITIAL_ONLINE = (
    "hipengine_laguna_global_attention_prefill_qrow4_dense_initial_online_bf16_spans"
)
_SYMBOL_GLOBAL_PREFILL_QROW6_DENSE_INITIAL_ONLINE = (
    "hipengine_laguna_global_attention_prefill_qrow6_dense_initial_online_bf16_spans"
)
_SYMBOL_SWA_WRITE = "hipengine_laguna_swa_write_kv_f32_bf16_spans"
_SYMBOL_SWA_WRITE_ROWS = "hipengine_laguna_swa_write_kv_rows_f32_bf16_spans"
_SYMBOL_SWA_ATTENTION = "hipengine_laguna_swa_attention_decode_bf16_spans"
_SYMBOL_SWA_ATTENTION_TOKEN4_EXACT = (
    "hipengine_laguna_swa_attention_decode_token4_exact_bf16_spans"
)
_SYMBOL_SWA_ATTENTION_SPLIT_EXACT = (
    "hipengine_laguna_swa_attention_decode_split_exact_bf16_spans"
)
_SYMBOL_SWA_ATTENTION_SPLIT_EXACT_GATED = (
    "hipengine_laguna_swa_attention_decode_split_exact_gated_bf16_spans"
)
_SYMBOL_SWA_ATTENTION_SPLIT_EXACT_GATED_WAVE_LOCAL = (
    "hipengine_laguna_swa_attention_decode_split_exact_gated_wave_local_bf16_spans"
)
_SYMBOL_SWA_ATTENTION_SPLIT_EXACT_GATED_GQA3_SCORES = (
    "hipengine_laguna_swa_attention_decode_split_exact_gated_gqa3_scores_bf16_spans"
)
_SYMBOL_SWA_ATTENTION_SPLIT_EXACT_GATED_WAVE_LOCAL_DIM2 = (
    "hipengine_laguna_swa_attention_decode_split_exact_gated_wave_local_dim2_bf16_spans"
)
_SYMBOL_SWA_ATTENTION_SPLIT_TILE16_EXACT = (
    "hipengine_laguna_swa_attention_decode_split_tile16_exact_bf16_spans"
)
_SYMBOL_SWA_ATTENTION_SPLIT_TILE16_EXACT_GATED = (
    "hipengine_laguna_swa_attention_decode_split_tile16_exact_gated_bf16_spans"
)
_SYMBOL_SWA_ATTENTION_SPLIT_TILE16_EXACT_GATED_WAVE_LOCAL = (
    "hipengine_laguna_swa_attention_decode_split_tile16_exact_gated_wave_local_bf16_spans"
)
_SYMBOL_SWA_ATTENTION_SPLIT_TILE16_EXACT_GATED_GQA3_SCORES = (
    "hipengine_laguna_swa_attention_decode_split_tile16_exact_gated_gqa3_scores_bf16_spans"
)
_SYMBOL_SWA_ATTENTION_SPLIT_TILE16_EXACT_GATED_GQA3_SCORES_FIXED512 = (
    "hipengine_laguna_swa_attention_decode_split_tile16_exact_gated_gqa3_scores_"
    "fixed512_bf16_spans"
)
_SYMBOL_SWA_ATTENTION_FUSED_EXACT_GATED_GQA2_FIXED512 = (
    "hipengine_laguna_swa_attention_decode_fused_exact_gated_gqa2_"
    "fixed512_bf16_spans"
)
_SYMBOL_SWA_ATTENTION_FUSED_EXACT_GATED_GQA3_LOCAL384_FIXED512 = (
    "hipengine_laguna_swa_attention_decode_fused_exact_gated_gqa3_"
    "local384_fixed512_bf16_spans"
)
_SYMBOL_SWA_ATTENTION_FUSED_EXACT_GATED_GQA3_VSTAGE64_FIXED512 = (
    "hipengine_laguna_swa_attention_decode_fused_exact_gated_gqa3_"
    "vstage64_fixed512_bf16_spans"
)
_SYMBOL_SWA_ATTENTION_FUSED_EXACT_GATED_GQA3_VSTAGE64_VEC16_FIXED512 = (
    "hipengine_laguna_swa_attention_decode_fused_exact_gated_gqa3_"
    "vstage64_vec16_fixed512_bf16_spans"
)
_SYMBOL_SWA_ATTENTION_FUSED_EXACT_GATED_GQA3_VSTAGE64_VEC16_DIRECT_FIXED512 = (
    "hipengine_laguna_swa_attention_decode_fused_exact_gated_gqa3_"
    "vstage64_vec16_direct_fixed512_bf16_spans"
)
_SYMBOL_SWA_ATTENTION_FUSED_EXACT_GATED_GQA3_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512 = (
    "hipengine_laguna_swa_attention_decode_fused_exact_gated_gqa3_"
    "vstage64_vec16_direct_assume_exp_fixed512_bf16_spans"
)
_SYMBOL_SWA_ATTENTION_FUSED_EXACT_GATED_MIXED32_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512 = (
    "hipengine_laguna_swa_attention_decode_fused_exact_gated_mixed32_"
    "vstage64_vec16_direct_assume_exp_fixed512_bf16_spans"
)
_SYMBOL_SWA_ATTENTION_FUSED_EXACT_GATED_MIXED32_EXP4_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512 = (
    "hipengine_laguna_swa_attention_decode_fused_exact_gated_mixed32_exp4_"
    "vstage64_vec16_direct_assume_exp_fixed512_bf16_spans"
)
_SYMBOL_SWA_ATTENTION_FUSED_EXACT_GATED_MIXED32_EXP8_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512 = (
    "hipengine_laguna_swa_attention_decode_fused_exact_gated_mixed32_exp8_"
    "vstage64_vec16_direct_assume_exp_fixed512_bf16_spans"
)
_SYMBOL_SWA_ATTENTION_FUSED_EXACT_GATED_MIXED32_EXP16_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512 = (
    "hipengine_laguna_swa_attention_decode_fused_exact_gated_mixed32_exp16_"
    "vstage64_vec16_direct_assume_exp_fixed512_bf16_spans"
)
_SYMBOL_SWA_ATTENTION_FUSED_EXACT_GATED_MIXED32_EXP32_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512 = (
    "hipengine_laguna_swa_attention_decode_fused_exact_gated_mixed32_exp32_"
    "vstage64_vec16_direct_assume_exp_fixed512_bf16_spans"
)
_SYMBOL_SWA_ATTENTION_FUSED_EXACT_GATED_MIXED32_EXP32_PRODUCER_MAX_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512 = (
    "hipengine_laguna_swa_attention_decode_fused_exact_gated_mixed32_exp32_"
    "producer_max_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans"
)
_SYMBOL_SWA_ATTENTION_FUSED_EXACT_GATED_MIXED32_EXP32_PRODUCER_MAX_GATE_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512 = (
    "hipengine_laguna_swa_attention_decode_fused_exact_gated_mixed32_exp32_"
    "producer_max_gate_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans"
)
_SYMBOL_SWA_ATTENTION_FUSED_EXACT_GATED_MIXED32_EXP32_PRODUCER_MAX_GATE_STAGE_PCACHE_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512 = (
    "hipengine_laguna_swa_attention_decode_fused_exact_gated_mixed32_exp32_"
    "producer_max_gate_stage_pcache_vstage64_vec16_direct_assume_exp_"
    "fixed512_bf16_spans"
)
_SYMBOL_SWA_ATTENTION_FUSED_EXACT_GATED_MIXED32_EXP32_PRODUCER_MAX_GATE_STAGE_PCACHE_IDLE_PRODUCER_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512 = (
    "hipengine_laguna_swa_attention_decode_fused_exact_gated_mixed32_exp32_"
    "producer_max_gate_stage_pcache_idle_producer_vstage64_vec16_direct_"
    "assume_exp_fixed512_bf16_spans"
)
_SYMBOL_SWA_ATTENTION_FUSED_EXACT_GATED_MIXED32_EXP32_PRODUCER_MAX_GATE_STAGE_PCACHE_VEC4_DENOM_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512 = (
    "hipengine_laguna_swa_attention_decode_fused_exact_gated_mixed32_exp32_"
    "producer_max_gate_stage_pcache_vec4_denom_vstage64_vec16_direct_"
    "assume_exp_fixed512_bf16_spans"
)
_SYMBOL_SWA_ATTENTION_FUSED_EXACT_GATED_MIXED32_EXP32_PRODUCER_MAX_GATE_STAGE_PCACHE_VEC4_DENOM_PROBABILITY_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512 = (
    "hipengine_laguna_swa_attention_decode_fused_exact_gated_mixed32_exp32_"
    "producer_max_gate_stage_pcache_vec4_denom_probability_vstage64_"
    "vec16_direct_assume_exp_fixed512_bf16_spans"
)
_SYMBOL_SWA_ATTENTION_FUSED_EXACT_GATED_MIXED32_EXP32_PRODUCER_MAX_GATE_STAGE_PCACHE_IDLE_VEC4_DENOM_PROBABILITY_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512 = (
    "hipengine_laguna_swa_attention_decode_fused_exact_gated_mixed32_exp32_"
    "producer_max_gate_stage_pcache_idle_vec4_denom_probability_vstage64_"
    "vec16_direct_assume_exp_fixed512_bf16_spans"
)
_SYMBOL_SWA_ATTENTION_FUSED_EXACT_GATED_MIXED40_EXP32_PRODUCER_MAX_GATE_STAGE_PCACHE_IDLE_VEC4_DENOM_PROBABILITY_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512 = (
    "hipengine_laguna_swa_attention_decode_fused_exact_gated_mixed40_exp32_"
    "producer_max_gate_stage_pcache_idle_vec4_denom_probability_vstage64_"
    "vec16_direct_assume_exp_fixed512_bf16_spans"
)
_SYMBOL_SWA_ATTENTION_FUSED_EXACT_GATED_MIXED40_EXP32_PRODUCER_MAX_GATE_STAGE_PCACHE_TAIL_PRODUCER_IDLE_VEC4_DENOM_PROBABILITY_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512 = (
    "hipengine_laguna_swa_attention_decode_fused_exact_gated_mixed40_exp32_"
    "producer_max_gate_stage_pcache_tail_producer_idle_vec4_denom_"
    "probability_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans"
)
_SYMBOL_SWA_ATTENTION_FUSED_EXACT_GATED_MIXED40_LOCAL512_EXP32_PRODUCER_MAX_GATE_STAGE_PCACHE_TAIL_PRODUCER_IDLE_VEC4_DENOM_PROBABILITY_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512 = (
    "hipengine_laguna_swa_attention_decode_fused_exact_gated_mixed40_local512_"
    "exp32_producer_max_gate_stage_pcache_tail_producer_idle_vec4_denom_"
    "probability_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans"
)
_SYMBOL_SWA_ATTENTION_FUSED_EXACT_GATED_MIXED40_LOCAL512_EXP32_PRODUCER_MAX_GATE_STAGE_PCACHE_TAIL_PRODUCER_VALUE_TAIL_IDLE_VEC4_DENOM_PROBABILITY_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512 = (
    "hipengine_laguna_swa_attention_decode_fused_exact_gated_mixed40_local512_"
    "exp32_producer_max_gate_stage_pcache_tail_producer_value_tail_idle_vec4_"
    "denom_probability_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans"
)
_SYMBOL_SWA_ATTENTION_FUSED_EXACT_GATED_MIXED40_LOCAL512_EXP32_PRODUCER_MAX_GATE_STAGE_PCACHE_TAIL_PRODUCER_VALUE_TAIL_IDLE_VEC4_DENOM_PROBABILITY_VSTAGE128_VEC16_DIRECT_ASSUME_EXP_FIXED512 = (
    "hipengine_laguna_swa_attention_decode_fused_exact_gated_mixed40_local512_"
    "exp32_producer_max_gate_stage_pcache_tail_producer_value_tail_idle_vec4_"
    "denom_probability_vstage128_vec16_direct_assume_exp_fixed512_bf16_spans"
)
_SYMBOL_SWA_ATTENTION_FUSED_EXACT_GATED_MIXED40_LOCAL512_EXP32_PRODUCER_MAX_GATE_STAGE_PCACHE_DUAL_TAIL_PRODUCER_VALUE_TAIL_IDLE_VEC4_DENOM_PROBABILITY_VSTAGE128_VEC16_DIRECT_ASSUME_EXP_FIXED512 = (
    "hipengine_laguna_swa_attention_decode_fused_exact_gated_mixed40_local512_"
    "exp32_producer_max_gate_stage_pcache_dual_tail_producer_value_tail_idle_"
    "vec4_denom_probability_vstage128_vec16_direct_assume_exp_fixed512_"
    "bf16_spans"
)
_SYMBOL_SWA_ATTENTION_FUSED_EXACT_GATED_MIXED40_LOCAL512_EXP32_PRODUCER_MAX_GATE_STAGE_PCACHE_OUTPUT_SHARDED_PROBABILITY_ALLWAVE_VALUE_IDLE_VEC4_DENOM_PROBABILITY_VSTAGE128_VEC16_DIRECT_ASSUME_EXP_FIXED512 = (
    "hipengine_laguna_swa_attention_decode_fused_exact_gated_mixed40_local512_"
    "exp32_producer_max_gate_stage_pcache_output_sharded_probability_allwave_"
    "value_idle_vec4_denom_probability_vstage128_vec16_direct_assume_exp_"
    "fixed512_bf16_spans"
)
_SYMBOL_SWA_ATTENTION_FUSED_EXACT_GATED_MIXED40_LOCAL512_EXP32_PRODUCER_MAX_GATE_STAGE_PCACHE_OUTPUT_SHARDED_PROBABILITY_DPP_QK_ALLWAVE_VALUE_IDLE_VEC4_DENOM_PROBABILITY_VSTAGE128_VEC16_DIRECT_ASSUME_EXP_FIXED512 = (
    "hipengine_laguna_swa_attention_decode_fused_exact_gated_mixed40_local512_"
    "exp32_producer_max_gate_stage_pcache_output_sharded_probability_dpp_qk_"
    "allwave_value_idle_vec4_denom_probability_vstage128_vec16_direct_assume_"
    "exp_fixed512_bf16_spans"
)
_SYMBOL_SWA_ATTENTION_FUSED_EXACT_GATED_MIXED40_LOCAL512_EXP32_PRODUCER_MAX_GATE_STAGE_PCACHE_OUTPUT_SHARDED_PROBABILITY_DPP_QK_DENSE_RING_ALLWAVE_VALUE_IDLE_VEC4_DENOM_PROBABILITY_VSTAGE128_VEC16_DIRECT_ASSUME_EXP_FIXED512 = (
    "hipengine_laguna_swa_attention_decode_fused_exact_gated_mixed40_local512_"
    "exp32_producer_max_gate_stage_pcache_output_sharded_probability_dpp_qk_"
    "dense_ring_allwave_value_idle_vec4_denom_probability_vstage128_vec16_"
    "direct_assume_exp_fixed512_bf16_spans"
)
_SYMBOL_SWA_ATTENTION_FUSED_EXACT_GATED_MIXED40_LOCAL1024_EXP32_PRODUCER_MAX_GATE_STAGE_PCACHE_OUTPUT_SHARDED_PROBABILITY_DPP_QK_DENSE_RING_ALLWAVE_VALUE_IDLE_VEC4_DENOM_PROBABILITY_VSTAGE128_VEC16_DIRECT_ASSUME_EXP_FIXED512 = (
    "hipengine_laguna_swa_attention_decode_fused_exact_gated_mixed40_local1024_"
    "exp32_producer_max_gate_stage_pcache_output_sharded_probability_dpp_qk_"
    "dense_ring_allwave_value_idle_vec4_denom_probability_vstage128_vec16_"
    "direct_assume_exp_fixed512_bf16_spans"
)
_SYMBOL_SWA_ATTENTION_FUSED_EXACT_GATED_MIXED40_EXP32_PRODUCER_MAX_GATE_STAGE_PCACHE_TAIL_PRODUCER_VALUE_TAIL_IDLE_VEC4_DENOM_PROBABILITY_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512 = (
    "hipengine_laguna_swa_attention_decode_fused_exact_gated_mixed40_exp32_"
    "producer_max_gate_stage_pcache_tail_producer_value_tail_idle_vec4_"
    "denom_probability_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans"
)
_SYMBOL_SWA_ATTENTION_FUSED_EXACT_GATED_MIXED40_EXP32_PRODUCER_MAX_GATE_STAGE_PCACHE_TAIL_PRODUCER_DENOM_PREFETCH4_IDLE_VEC4_PROBABILITY_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512 = (
    "hipengine_laguna_swa_attention_decode_fused_exact_gated_mixed40_exp32_"
    "producer_max_gate_stage_pcache_tail_producer_denom_prefetch4_idle_vec4_"
    "probability_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans"
)
_SYMBOL_SWA_ATTENTION_FUSED_EXACT_GATED_MIXED32_EXP32_PRODUCER_MAX_GATE_STAGE_PCACHE_DPP_QK_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512 = (
    "hipengine_laguna_swa_attention_decode_fused_exact_gated_mixed32_exp32_"
    "producer_max_gate_stage_pcache_dpp_qk_vstage64_vec16_direct_assume_exp_"
    "fixed512_bf16_spans"
)
_SYMBOL_SWA_ATTENTION_FUSED_EXACT_GATED_MIXED32_EXP32_PRODUCER_MAX_GATE_DPP_QK_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512 = (
    "hipengine_laguna_swa_attention_decode_fused_exact_gated_mixed32_exp32_"
    "producer_max_gate_dpp_qk_vstage64_vec16_direct_assume_exp_fixed512_"
    "bf16_spans"
)
_SYMBOL_SWA_ATTENTION_FUSED_EXACT_GATED_MIXED32_EXP32_PRODUCER_MAX_GATE_GATED_ONLY_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512 = (
    "hipengine_laguna_swa_attention_decode_fused_exact_gated_mixed32_exp32_"
    "producer_max_gate_gated_only_vstage64_vec16_direct_assume_exp_"
    "fixed512_bf16_spans"
)
_SYMBOL_SWA_ATTENTION_SPLIT_TILE16_EXACT_GATED_WAVE_LOCAL_DIM2 = (
    "hipengine_laguna_swa_attention_decode_split_tile16_exact_gated_wave_local_dim2_bf16_spans"
)
_SYMBOL_SWA_PREFILL = "hipengine_laguna_swa_attention_prefill_bf16_spans"
_SYMBOL_SWA_PREFILL_WAVE32_EXACT = (
    "hipengine_laguna_swa_attention_prefill_wave32_exact_bf16_spans"
)
_SYMBOL_SWA_PREFILL_QROW2_EXACT = (
    "hipengine_laguna_swa_attention_prefill_qrow2_exact_bf16_spans"
)
_SYMBOL_SWA_PREFILL_QROW2_ONLINE = (
    "hipengine_laguna_swa_attention_prefill_qrow2_online_bf16_spans"
)
_SYMBOL_SWA_PREFILL_QROW4_ONLINE = (
    "hipengine_laguna_swa_attention_prefill_qrow4_online_bf16_spans"
)
_SYMBOL_SWA_PREFILL_QROW4_SOURCEQUAL_ONLINE = (
    "hipengine_laguna_swa_attention_prefill_qrow4_sourcequal_online_bf16_spans"
)
_SYMBOL_SWA_PREFILL_QROW4_CACHED_ONLINE = (
    "hipengine_laguna_swa_attention_prefill_qrow4_cached_online_bf16_spans"
)
_SYMBOL_SWA_PREFILL_QROW4_CACHED_META_ONLINE = (
    "hipengine_laguna_swa_attention_prefill_qrow4_cached_meta_online_bf16_spans"
)
_SYMBOL_SWA_PREFILL_QROW4_DENSE_INITIAL_ONLINE = (
    "hipengine_laguna_swa_attention_prefill_qrow4_dense_initial_online_bf16_spans"
)
_SYMBOL_DENSE_INITIAL_CACHE_BF16_TO_F32 = (
    "hipengine_laguna_dense_initial_cache_bf16_to_f32_spans"
)
_SYMBOL_DENSE_INITIAL_CACHE_BLOCK_BF16_TO_F32 = (
    "hipengine_laguna_dense_initial_cache_block_bf16_to_f32_spans"
)
_SYMBOL_DENSE_INITIAL_CONTIGUOUS_CACHE_BLOCK_BF16_TO_F32 = (
    "hipengine_laguna_dense_initial_contiguous_cache_block_bf16_to_f32_spans"
)
_SYMBOL_DENSE_INITIAL_QUERY_HEAD_TRANSPOSE_F32 = (
    "hipengine_laguna_dense_initial_query_head_transpose_f32"
)
_SYMBOL_DENSE_INITIAL_CAUSAL_SOFTMAX_F32 = (
    "hipengine_laguna_dense_initial_causal_softmax_f32_spans"
)
_SYMBOL_DENSE_INITIAL_CAUSAL_SOFTMAX_WAVE_ROWS_F32 = (
    "hipengine_laguna_dense_initial_causal_softmax_wave_rows_f32_spans"
)
_SYMBOL_DENSE_INITIAL_CAUSAL_SOFTMAX_TILE_WAVE_ROWS_F32 = (
    "hipengine_laguna_dense_initial_causal_softmax_tile_wave_rows_f32_spans"
)
_SYMBOL_DENSE_INITIAL_ATTENTION_TILE_MERGE_F32 = (
    "hipengine_laguna_dense_initial_attention_tile_merge_f32"
)
_SYMBOL_SWA_UNION_BF16_TO_F32 = (
    "hipengine_laguna_swa_union_bf16_to_f32_spans"
)
_SYMBOL_SWA_UNION_SOFTMAX_WAVE_ROWS_F32 = (
    "hipengine_laguna_swa_union_softmax_wave_rows_f32"
)
_LAGUNA_KV_HEADS = 8
_LAGUNA_HEAD_DIM = 128
_GLOBAL_BLOCK_SIZE = 256
_MAX_EAGER_GLOBAL_CONTEXT = 4096


def plan_laguna_kv_attention_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="laguna_kv_attention",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
    )


def build_laguna_kv_attention(
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
        family="laguna_kv_attention",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def laguna_global_head_rmsnorm_rope_write_kv_f32_spans(
    query_ptr: int,
    key_ptr: int,
    value_ptr: int,
    q_weight_ptr: int,
    k_weight_ptr: int,
    cos_ptr: int,
    sin_ptr: int,
    query_out_ptr: int,
    key_out_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    spans: KVLiveSpans,
    eps: float,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    max_positions: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Fuse exact global head RMSNorm/RoPE with complete-span BF16 KV append."""

    capacity = _check_global_spans(spans, num_kv_heads, head_dim)
    _check_laguna_attention_shape(num_q_heads, num_kv_heads, head_dim)
    _check_head_kv_rope_shape(rotary_dim, head_dim, max_positions)
    library = library or build_laguna_kv_attention(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_GLOBAL_HEAD_KV)
    fn.argtypes = (
        [ctypes.c_void_p] * 16
        + [ctypes.c_float]
        + [ctypes.c_int64] * 8
        + [ctypes.c_void_p]
    )
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(query_ptr),
        ctypes.c_void_p(key_ptr),
        ctypes.c_void_p(value_ptr),
        ctypes.c_void_p(q_weight_ptr),
        ctypes.c_void_p(k_weight_ptr),
        ctypes.c_void_p(cos_ptr),
        ctypes.c_void_p(sin_ptr),
        ctypes.c_void_p(query_out_ptr),
        ctypes.c_void_p(key_out_ptr),
        ctypes.c_void_p(key_cache_ptr),
        ctypes.c_void_p(value_cache_ptr),
        ctypes.c_void_p(spans.base_offsets.ptr),
        ctypes.c_void_p(spans.live_counts.ptr),
        ctypes.c_void_p(spans.token_positions.ptr),
        ctypes.c_void_p(spans.evict_mask.ptr),
        ctypes.c_void_p(spans.row_positions.ptr),
        ctypes.c_float(eps),
        ctypes.c_int64(capacity),
        ctypes.c_int64(_GLOBAL_BLOCK_SIZE),
        ctypes.c_int64(spans.base_offsets.numel),
        ctypes.c_int64(num_q_heads),
        ctypes.c_int64(num_kv_heads),
        ctypes.c_int64(head_dim),
        ctypes.c_int64(rotary_dim),
        ctypes.c_int64(max_positions),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def laguna_global_head_rmsnorm_rope_write_kv_wave0_tree_f32_spans(
    query_ptr: int,
    key_ptr: int,
    value_ptr: int,
    q_weight_ptr: int,
    k_weight_ptr: int,
    cos_ptr: int,
    sin_ptr: int,
    query_out_ptr: int,
    key_out_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    spans: KVLiveSpans,
    eps: float,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    max_positions: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Run the exact wave-0 RMS tree for global current-P4 head/KV append."""

    capacity = _check_global_spans(spans, num_kv_heads, head_dim)
    _check_laguna_attention_shape(num_q_heads, num_kv_heads, head_dim)
    if int(num_q_heads) != 48:
        raise ValueError("num_q_heads must be 48 for Laguna global head/KV")
    _check_head_kv_rope_shape(rotary_dim, head_dim, max_positions)
    assert spans.token_positions is not None
    assert spans.evict_mask is not None
    assert spans.row_positions is not None
    _check_nonzero_device_pointers(
        ("query_ptr", query_ptr),
        ("key_ptr", key_ptr),
        ("value_ptr", value_ptr),
        ("q_weight_ptr", q_weight_ptr),
        ("k_weight_ptr", k_weight_ptr),
        ("cos_ptr", cos_ptr),
        ("sin_ptr", sin_ptr),
        ("query_out_ptr", query_out_ptr),
        ("key_out_ptr", key_out_ptr),
        ("key_cache_ptr", key_cache_ptr),
        ("value_cache_ptr", value_cache_ptr),
        ("base_offsets_ptr", spans.base_offsets.ptr),
        ("live_counts_ptr", spans.live_counts.ptr),
        ("token_positions_ptr", spans.token_positions.ptr),
        ("evict_mask_ptr", spans.evict_mask.ptr),
        ("row_positions_ptr", spans.row_positions.ptr),
    )
    library = library or build_laguna_kv_attention(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_GLOBAL_HEAD_KV_WAVE0_TREE)
    fn.argtypes = (
        [ctypes.c_void_p] * 16
        + [ctypes.c_float]
        + [ctypes.c_int64] * 8
        + [ctypes.c_void_p]
    )
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(query_ptr),
        ctypes.c_void_p(key_ptr),
        ctypes.c_void_p(value_ptr),
        ctypes.c_void_p(q_weight_ptr),
        ctypes.c_void_p(k_weight_ptr),
        ctypes.c_void_p(cos_ptr),
        ctypes.c_void_p(sin_ptr),
        ctypes.c_void_p(query_out_ptr),
        ctypes.c_void_p(key_out_ptr),
        ctypes.c_void_p(key_cache_ptr),
        ctypes.c_void_p(value_cache_ptr),
        ctypes.c_void_p(spans.base_offsets.ptr),
        ctypes.c_void_p(spans.live_counts.ptr),
        ctypes.c_void_p(spans.token_positions.ptr),
        ctypes.c_void_p(spans.evict_mask.ptr),
        ctypes.c_void_p(spans.row_positions.ptr),
        ctypes.c_float(eps),
        ctypes.c_int64(capacity),
        ctypes.c_int64(_GLOBAL_BLOCK_SIZE),
        ctypes.c_int64(spans.base_offsets.numel),
        ctypes.c_int64(num_q_heads),
        ctypes.c_int64(num_kv_heads),
        ctypes.c_int64(head_dim),
        ctypes.c_int64(rotary_dim),
        ctypes.c_int64(max_positions),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def laguna_swa_head_rmsnorm_rope_write_kv_f32_spans(
    query_ptr: int,
    key_ptr: int,
    value_ptr: int,
    q_weight_ptr: int,
    k_weight_ptr: int,
    cos_ptr: int,
    sin_ptr: int,
    query_out_ptr: int,
    key_out_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    spans: KVLiveSpans,
    eps: float,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    max_positions: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Fuse exact SWA head RMSNorm/RoPE with complete-ring BF16 KV append."""

    capacity = _check_swa_spans(spans, num_kv_heads, head_dim)
    _check_laguna_attention_shape(num_q_heads, num_kv_heads, head_dim)
    _check_head_kv_rope_shape(rotary_dim, head_dim, max_positions)
    library = library or build_laguna_kv_attention(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_SWA_HEAD_KV)
    fn.argtypes = (
        [ctypes.c_void_p] * 16
        + [ctypes.c_float]
        + [ctypes.c_int64] * 6
        + [ctypes.c_void_p]
    )
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(query_ptr),
        ctypes.c_void_p(key_ptr),
        ctypes.c_void_p(value_ptr),
        ctypes.c_void_p(q_weight_ptr),
        ctypes.c_void_p(k_weight_ptr),
        ctypes.c_void_p(cos_ptr),
        ctypes.c_void_p(sin_ptr),
        ctypes.c_void_p(query_out_ptr),
        ctypes.c_void_p(key_out_ptr),
        ctypes.c_void_p(key_cache_ptr),
        ctypes.c_void_p(value_cache_ptr),
        ctypes.c_void_p(spans.base_offsets.ptr),
        ctypes.c_void_p(spans.live_counts.ptr),
        ctypes.c_void_p(spans.token_positions.ptr),
        ctypes.c_void_p(spans.evict_mask.ptr),
        ctypes.c_void_p(spans.row_positions.ptr),
        ctypes.c_float(eps),
        ctypes.c_int64(capacity),
        ctypes.c_int64(num_q_heads),
        ctypes.c_int64(num_kv_heads),
        ctypes.c_int64(head_dim),
        ctypes.c_int64(rotary_dim),
        ctypes.c_int64(max_positions),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def _laguna_f16_projection_head_kv_bf16_spans(
    symbol: str,
    x_ptr: int,
    query_weight_ptr: int,
    key_weight_ptr: int,
    value_weight_ptr: int,
    gate_weight_ptr: int,
    query_ptr: int,
    key_ptr: int,
    value_ptr: int,
    gate_ptr: int,
    q_norm_weight_ptr: int,
    k_norm_weight_ptr: int,
    cos_ptr: int,
    sin_ptr: int,
    query_out_ptr: int,
    key_out_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    completion_counters_ptr: int,
    spans: KVLiveSpans,
    eps: float,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    max_positions: int,
    *,
    global_spans: bool,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    if global_spans:
        capacity = _check_global_spans(spans, num_kv_heads, head_dim)
        if int(num_q_heads) != 48:
            raise ValueError("num_q_heads must be 48 for Laguna global projection/head/KV")
    else:
        capacity = _check_swa_spans(spans, num_kv_heads, head_dim)
        if int(num_q_heads) != 72:
            raise ValueError("num_q_heads must be 72 for Laguna SWA projection/head/KV")
    _check_laguna_attention_shape(num_q_heads, num_kv_heads, head_dim)
    _check_head_kv_rope_shape(rotary_dim, head_dim, max_positions)
    assert spans.token_positions is not None
    assert spans.evict_mask is not None
    assert spans.row_positions is not None
    _check_nonzero_device_pointers(
        ("x_ptr", x_ptr),
        ("query_weight_ptr", query_weight_ptr),
        ("key_weight_ptr", key_weight_ptr),
        ("value_weight_ptr", value_weight_ptr),
        ("gate_weight_ptr", gate_weight_ptr),
        ("query_ptr", query_ptr),
        ("key_ptr", key_ptr),
        ("value_ptr", value_ptr),
        ("gate_ptr", gate_ptr),
        ("q_norm_weight_ptr", q_norm_weight_ptr),
        ("k_norm_weight_ptr", k_norm_weight_ptr),
        ("cos_ptr", cos_ptr),
        ("sin_ptr", sin_ptr),
        ("query_out_ptr", query_out_ptr),
        ("key_out_ptr", key_out_ptr),
        ("key_cache_ptr", key_cache_ptr),
        ("value_cache_ptr", value_cache_ptr),
        ("completion_counters_ptr", completion_counters_ptr),
        ("base_offsets_ptr", spans.base_offsets.ptr),
        ("live_counts_ptr", spans.live_counts.ptr),
        ("token_positions_ptr", spans.token_positions.ptr),
        ("evict_mask_ptr", spans.evict_mask.ptr),
        ("row_positions_ptr", spans.row_positions.ptr),
    )
    library = library or build_laguna_kv_attention(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, symbol)
    integer_count = 8 if global_spans else 6
    fn.argtypes = (
        [ctypes.c_void_p] * 23
        + [ctypes.c_float]
        + [ctypes.c_int64] * integer_count
        + [ctypes.c_void_p]
    )
    fn.restype = ctypes.c_int
    arguments = [
        ctypes.c_void_p(x_ptr),
        ctypes.c_void_p(query_weight_ptr),
        ctypes.c_void_p(key_weight_ptr),
        ctypes.c_void_p(value_weight_ptr),
        ctypes.c_void_p(gate_weight_ptr),
        ctypes.c_void_p(query_ptr),
        ctypes.c_void_p(key_ptr),
        ctypes.c_void_p(value_ptr),
        ctypes.c_void_p(gate_ptr),
        ctypes.c_void_p(q_norm_weight_ptr),
        ctypes.c_void_p(k_norm_weight_ptr),
        ctypes.c_void_p(cos_ptr),
        ctypes.c_void_p(sin_ptr),
        ctypes.c_void_p(query_out_ptr),
        ctypes.c_void_p(key_out_ptr),
        ctypes.c_void_p(key_cache_ptr),
        ctypes.c_void_p(value_cache_ptr),
        ctypes.c_void_p(completion_counters_ptr),
        ctypes.c_void_p(spans.base_offsets.ptr),
        ctypes.c_void_p(spans.live_counts.ptr),
        ctypes.c_void_p(spans.token_positions.ptr),
        ctypes.c_void_p(spans.evict_mask.ptr),
        ctypes.c_void_p(spans.row_positions.ptr),
        ctypes.c_float(eps),
        ctypes.c_int64(capacity),
    ]
    if global_spans:
        arguments.extend(
            (
                ctypes.c_int64(_GLOBAL_BLOCK_SIZE),
                ctypes.c_int64(spans.base_offsets.numel),
            )
        )
    arguments.extend(
        (
            ctypes.c_int64(num_q_heads),
            ctypes.c_int64(num_kv_heads),
            ctypes.c_int64(head_dim),
            ctypes.c_int64(rotary_dim),
            ctypes.c_int64(max_positions),
            ctypes.c_void_p(stream),
        )
    )
    err = fn(*arguments)
    _check_launch(runtime, err)


def laguna_global_f16_projection_head_kv_bf16_spans(
    x_ptr: int,
    query_weight_ptr: int,
    key_weight_ptr: int,
    value_weight_ptr: int,
    gate_weight_ptr: int,
    query_ptr: int,
    key_ptr: int,
    value_ptr: int,
    gate_ptr: int,
    q_norm_weight_ptr: int,
    k_norm_weight_ptr: int,
    cos_ptr: int,
    sin_ptr: int,
    query_out_ptr: int,
    key_out_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    completion_counters_ptr: int,
    spans: KVLiveSpans,
    eps: float,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    max_positions: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Fuse exact global c=1 source-F16 Q/K/V/gate, head norm, and KV append."""

    _laguna_f16_projection_head_kv_bf16_spans(
        _SYMBOL_GLOBAL_F16_PROJECTION_HEAD_KV,
        x_ptr,
        query_weight_ptr,
        key_weight_ptr,
        value_weight_ptr,
        gate_weight_ptr,
        query_ptr,
        key_ptr,
        value_ptr,
        gate_ptr,
        q_norm_weight_ptr,
        k_norm_weight_ptr,
        cos_ptr,
        sin_ptr,
        query_out_ptr,
        key_out_ptr,
        key_cache_ptr,
        value_cache_ptr,
        completion_counters_ptr,
        spans,
        eps,
        num_q_heads,
        num_kv_heads,
        head_dim,
        rotary_dim,
        max_positions,
        global_spans=True,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def laguna_swa_f16_projection_head_kv_bf16_spans(
    x_ptr: int,
    query_weight_ptr: int,
    key_weight_ptr: int,
    value_weight_ptr: int,
    gate_weight_ptr: int,
    query_ptr: int,
    key_ptr: int,
    value_ptr: int,
    gate_ptr: int,
    q_norm_weight_ptr: int,
    k_norm_weight_ptr: int,
    cos_ptr: int,
    sin_ptr: int,
    query_out_ptr: int,
    key_out_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    completion_counters_ptr: int,
    spans: KVLiveSpans,
    eps: float,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    max_positions: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Fuse exact SWA c=1 source-F16 Q/K/V/gate, head norm, and KV append."""

    _laguna_f16_projection_head_kv_bf16_spans(
        _SYMBOL_SWA_F16_PROJECTION_HEAD_KV,
        x_ptr,
        query_weight_ptr,
        key_weight_ptr,
        value_weight_ptr,
        gate_weight_ptr,
        query_ptr,
        key_ptr,
        value_ptr,
        gate_ptr,
        q_norm_weight_ptr,
        k_norm_weight_ptr,
        cos_ptr,
        sin_ptr,
        query_out_ptr,
        key_out_ptr,
        key_cache_ptr,
        value_cache_ptr,
        completion_counters_ptr,
        spans,
        eps,
        num_q_heads,
        num_kv_heads,
        head_dim,
        rotary_dim,
        max_positions,
        global_spans=False,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def laguna_global_write_kv_f32_spans(
    key_ptr: int,
    value_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    spans: KVLiveSpans,
    num_kv_heads: int,
    head_dim: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Append one F32 K/V row to complete block-256 global spans."""

    capacity = _check_global_spans(spans, num_kv_heads, head_dim)
    library = library or build_laguna_kv_attention(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_GLOBAL_WRITE)
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
        ctypes.c_void_p(key_ptr),
        ctypes.c_void_p(value_ptr),
        ctypes.c_void_p(key_cache_ptr),
        ctypes.c_void_p(value_cache_ptr),
        ctypes.c_void_p(spans.base_offsets.ptr),
        ctypes.c_void_p(spans.live_counts.ptr),
        ctypes.c_void_p(spans.token_positions.ptr),
        ctypes.c_void_p(spans.evict_mask.ptr),
        ctypes.c_void_p(spans.row_positions.ptr),
        ctypes.c_int64(capacity),
        ctypes.c_int64(_GLOBAL_BLOCK_SIZE),
        ctypes.c_int64(spans.base_offsets.numel),
        ctypes.c_int64(num_kv_heads),
        ctypes.c_int64(head_dim),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def laguna_global_write_kv_rows_f32_spans(
    key_ptr: int,
    value_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    spans: KVLiveSpans,
    rows: int,
    num_kv_heads: int,
    head_dim: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Append consecutive F32 K/V rows to complete block-256 global spans."""

    capacity = _check_global_spans(spans, num_kv_heads, head_dim)
    _check_prefill_rows(spans, rows, capacity)
    library = library or build_laguna_kv_attention(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_GLOBAL_WRITE_ROWS)
    fn.argtypes = [ctypes.c_void_p] * 9 + [ctypes.c_int64] * 6 + [ctypes.c_void_p]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(key_ptr),
        ctypes.c_void_p(value_ptr),
        ctypes.c_void_p(key_cache_ptr),
        ctypes.c_void_p(value_cache_ptr),
        ctypes.c_void_p(spans.base_offsets.ptr),
        ctypes.c_void_p(spans.live_counts.ptr),
        ctypes.c_void_p(spans.token_positions.ptr),
        ctypes.c_void_p(spans.evict_mask.ptr),
        ctypes.c_void_p(spans.row_positions.ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(capacity),
        ctypes.c_int64(_GLOBAL_BLOCK_SIZE),
        ctypes.c_int64(spans.base_offsets.numel),
        ctypes.c_int64(num_kv_heads),
        ctypes.c_int64(head_dim),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def laguna_global_attention_decode_bf16_spans(
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    spans: KVLiveSpans,
    max_context_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Run ungated block-256 global GQA through complete dense spans."""

    capacity = _check_global_spans(spans, num_kv_heads, head_dim)
    if int(max_context_len) != capacity:
        raise ValueError("max_context_len must equal the global span capacity")
    if capacity > _MAX_EAGER_GLOBAL_CONTEXT:
        raise ValueError("eager Laguna global attention currently supports at most 4096 tokens")
    _check_laguna_attention_shape(num_q_heads, num_kv_heads, head_dim)
    library = library or build_laguna_kv_attention(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_GLOBAL_ATTENTION)
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
        ctypes.c_int64,
        ctypes.c_float,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(query_ptr),
        ctypes.c_void_p(key_cache_ptr),
        ctypes.c_void_p(value_cache_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_void_p(spans.base_offsets.ptr),
        ctypes.c_void_p(spans.live_counts.ptr),
        ctypes.c_void_p(spans.token_positions.ptr),
        ctypes.c_void_p(spans.evict_mask.ptr),
        ctypes.c_void_p(spans.row_positions.ptr),
        ctypes.c_int64(capacity),
        ctypes.c_int64(_GLOBAL_BLOCK_SIZE),
        ctypes.c_int64(spans.base_offsets.numel),
        ctypes.c_int64(num_q_heads),
        ctypes.c_int64(num_kv_heads),
        ctypes.c_int64(head_dim),
        ctypes.c_float(scale),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def laguna_global_attention_decode_single_page_bf16_spans(
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    spans: KVLiveSpans,
    max_context_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Run exact scalar global GQA when the live span fits physical page zero."""

    capacity = _check_global_spans(spans, num_kv_heads, head_dim)
    if int(max_context_len) != capacity:
        raise ValueError("max_context_len must equal the global span capacity")
    if capacity > _MAX_EAGER_GLOBAL_CONTEXT:
        raise ValueError("eager Laguna global attention currently supports at most 4096 tokens")
    _check_laguna_attention_shape(num_q_heads, num_kv_heads, head_dim)
    if int(num_q_heads) != 48:
        raise ValueError("num_q_heads must be 48 for Laguna global attention")
    assert spans.token_positions is not None
    assert spans.evict_mask is not None
    assert spans.row_positions is not None
    _check_nonzero_device_pointers(
        ("query_ptr", query_ptr),
        ("key_cache_ptr", key_cache_ptr),
        ("value_cache_ptr", value_cache_ptr),
        ("out_ptr", out_ptr),
        ("base_offsets_ptr", spans.base_offsets.ptr),
        ("live_counts_ptr", spans.live_counts.ptr),
        ("token_positions_ptr", spans.token_positions.ptr),
        ("evict_mask_ptr", spans.evict_mask.ptr),
        ("row_positions_ptr", spans.row_positions.ptr),
    )
    library = library or build_laguna_kv_attention(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_GLOBAL_ATTENTION_SINGLE_PAGE)
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
        ctypes.c_int64,
        ctypes.c_float,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(query_ptr),
        ctypes.c_void_p(key_cache_ptr),
        ctypes.c_void_p(value_cache_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_void_p(spans.base_offsets.ptr),
        ctypes.c_void_p(spans.live_counts.ptr),
        ctypes.c_void_p(spans.token_positions.ptr),
        ctypes.c_void_p(spans.evict_mask.ptr),
        ctypes.c_void_p(spans.row_positions.ptr),
        ctypes.c_int64(capacity),
        ctypes.c_int64(_GLOBAL_BLOCK_SIZE),
        ctypes.c_int64(spans.base_offsets.numel),
        ctypes.c_int64(num_q_heads),
        ctypes.c_int64(num_kv_heads),
        ctypes.c_int64(head_dim),
        ctypes.c_float(scale),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def laguna_global_attention_decode_single_page_softplus_gate_bf16_spans(
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    gate_ptr: int,
    gated_out_ptr: int,
    spans: KVLiveSpans,
    max_context_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Run exact page-zero global GQA and fuse its BF16 softplus gate."""

    capacity = _check_global_spans(spans, num_kv_heads, head_dim)
    if int(max_context_len) != capacity:
        raise ValueError("max_context_len must equal the global span capacity")
    if capacity > _MAX_EAGER_GLOBAL_CONTEXT:
        raise ValueError("eager Laguna global attention currently supports at most 4096 tokens")
    _check_laguna_attention_shape(num_q_heads, num_kv_heads, head_dim)
    if int(num_q_heads) != 48:
        raise ValueError("num_q_heads must be 48 for Laguna global attention")
    assert spans.token_positions is not None
    assert spans.evict_mask is not None
    assert spans.row_positions is not None
    _check_nonzero_device_pointers(
        ("query_ptr", query_ptr),
        ("key_cache_ptr", key_cache_ptr),
        ("value_cache_ptr", value_cache_ptr),
        ("out_ptr", out_ptr),
        ("gate_ptr", gate_ptr),
        ("gated_out_ptr", gated_out_ptr),
        ("base_offsets_ptr", spans.base_offsets.ptr),
        ("live_counts_ptr", spans.live_counts.ptr),
        ("token_positions_ptr", spans.token_positions.ptr),
        ("evict_mask_ptr", spans.evict_mask.ptr),
        ("row_positions_ptr", spans.row_positions.ptr),
    )
    library = library or build_laguna_kv_attention(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_GLOBAL_ATTENTION_SINGLE_PAGE_GATED)
    fn.argtypes = [ctypes.c_void_p] * 11 + [ctypes.c_int64] * 6 + [
        ctypes.c_float,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(query_ptr),
        ctypes.c_void_p(key_cache_ptr),
        ctypes.c_void_p(value_cache_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_void_p(gate_ptr),
        ctypes.c_void_p(gated_out_ptr),
        ctypes.c_void_p(spans.base_offsets.ptr),
        ctypes.c_void_p(spans.live_counts.ptr),
        ctypes.c_void_p(spans.token_positions.ptr),
        ctypes.c_void_p(spans.evict_mask.ptr),
        ctypes.c_void_p(spans.row_positions.ptr),
        ctypes.c_int64(capacity),
        ctypes.c_int64(_GLOBAL_BLOCK_SIZE),
        ctypes.c_int64(spans.base_offsets.numel),
        ctypes.c_int64(num_q_heads),
        ctypes.c_int64(num_kv_heads),
        ctypes.c_int64(head_dim),
        ctypes.c_float(scale),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def laguna_global_attention_decode_split_exact_bf16_spans(
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    score_scratch_ptr: int,
    physical_scratch_ptr: int,
    spans: KVLiveSpans,
    scan_slots: int,
    max_context_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Run exact split-score global attention with caller-owned scratch."""

    capacity = _check_global_spans(spans, num_kv_heads, head_dim)
    if int(max_context_len) != capacity:
        raise ValueError("max_context_len must equal the global span capacity")
    parsed_scan = _check_split_scan_slots(scan_slots, capacity)
    _check_laguna_attention_shape(num_q_heads, num_kv_heads, head_dim)
    library = library or build_laguna_kv_attention(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_GLOBAL_ATTENTION_SPLIT_EXACT)
    fn.argtypes = (
        [ctypes.c_void_p] * 11
        + [ctypes.c_int64] * 8
        + [ctypes.c_float, ctypes.c_void_p]
    )
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(query_ptr),
        ctypes.c_void_p(key_cache_ptr),
        ctypes.c_void_p(value_cache_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_void_p(score_scratch_ptr),
        ctypes.c_void_p(physical_scratch_ptr),
        ctypes.c_void_p(spans.base_offsets.ptr),
        ctypes.c_void_p(spans.live_counts.ptr),
        ctypes.c_void_p(spans.token_positions.ptr),
        ctypes.c_void_p(spans.evict_mask.ptr),
        ctypes.c_void_p(spans.row_positions.ptr),
        ctypes.c_int64(capacity),
        ctypes.c_int64(_GLOBAL_BLOCK_SIZE),
        ctypes.c_int64(spans.base_offsets.numel),
        ctypes.c_int64(capacity),
        ctypes.c_int64(parsed_scan),
        ctypes.c_int64(num_q_heads),
        ctypes.c_int64(num_kv_heads),
        ctypes.c_int64(head_dim),
        ctypes.c_float(scale),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def laguna_global_attention_decode_split_exact_gated_bf16_spans(
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    gate_ptr: int,
    gated_out_ptr: int,
    score_scratch_ptr: int,
    physical_scratch_ptr: int,
    spans: KVLiveSpans,
    scan_slots: int,
    max_context_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Run exact split global attention and fuse its BF16 softplus gate."""

    capacity = _check_global_spans(spans, num_kv_heads, head_dim)
    if int(max_context_len) != capacity:
        raise ValueError("max_context_len must equal the global span capacity")
    parsed_scan = _check_split_scan_slots(scan_slots, capacity)
    _check_laguna_attention_shape(num_q_heads, num_kv_heads, head_dim)
    library = library or build_laguna_kv_attention(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_GLOBAL_ATTENTION_SPLIT_EXACT_GATED)
    fn.argtypes = (
        [ctypes.c_void_p] * 13
        + [ctypes.c_int64] * 8
        + [ctypes.c_float, ctypes.c_void_p]
    )
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(query_ptr),
        ctypes.c_void_p(key_cache_ptr),
        ctypes.c_void_p(value_cache_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_void_p(gate_ptr),
        ctypes.c_void_p(gated_out_ptr),
        ctypes.c_void_p(score_scratch_ptr),
        ctypes.c_void_p(physical_scratch_ptr),
        ctypes.c_void_p(spans.base_offsets.ptr),
        ctypes.c_void_p(spans.live_counts.ptr),
        ctypes.c_void_p(spans.token_positions.ptr),
        ctypes.c_void_p(spans.evict_mask.ptr),
        ctypes.c_void_p(spans.row_positions.ptr),
        ctypes.c_int64(capacity),
        ctypes.c_int64(_GLOBAL_BLOCK_SIZE),
        ctypes.c_int64(spans.base_offsets.numel),
        ctypes.c_int64(capacity),
        ctypes.c_int64(parsed_scan),
        ctypes.c_int64(num_q_heads),
        ctypes.c_int64(num_kv_heads),
        ctypes.c_int64(head_dim),
        ctypes.c_float(scale),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def laguna_global_attention_decode_split_exact_gated_fixedshape_bf16_spans(
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    gate_ptr: int,
    gated_out_ptr: int,
    score_scratch_ptr: int,
    physical_scratch_ptr: int,
    spans: KVLiveSpans,
    scan_slots: int,
    max_context_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Run exact split global attention with natural-shape reduction."""

    capacity = _check_global_spans(spans, num_kv_heads, head_dim)
    parsed_scan = _check_split_scan_slots(scan_slots, capacity)
    if (
        capacity != 4096
        or int(max_context_len) != 4096
        or int(num_q_heads) != 48
        or int(num_kv_heads) != 8
        or int(head_dim) != 128
    ):
        raise ValueError(
            "fixed-shape global reduction requires capacity 4096, "
            "48 query heads, 8 KV heads, and D128"
        )
    library = library or build_laguna_kv_attention(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_GLOBAL_ATTENTION_SPLIT_EXACT_GATED_FIXEDSHAPE)
    fn.argtypes = (
        [ctypes.c_void_p] * 13
        + [ctypes.c_int64] * 8
        + [ctypes.c_float, ctypes.c_void_p]
    )
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(query_ptr),
        ctypes.c_void_p(key_cache_ptr),
        ctypes.c_void_p(value_cache_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_void_p(gate_ptr),
        ctypes.c_void_p(gated_out_ptr),
        ctypes.c_void_p(score_scratch_ptr),
        ctypes.c_void_p(physical_scratch_ptr),
        ctypes.c_void_p(spans.base_offsets.ptr),
        ctypes.c_void_p(spans.live_counts.ptr),
        ctypes.c_void_p(spans.token_positions.ptr),
        ctypes.c_void_p(spans.evict_mask.ptr),
        ctypes.c_void_p(spans.row_positions.ptr),
        ctypes.c_int64(capacity),
        ctypes.c_int64(_GLOBAL_BLOCK_SIZE),
        ctypes.c_int64(spans.base_offsets.numel),
        ctypes.c_int64(capacity),
        ctypes.c_int64(parsed_scan),
        ctypes.c_int64(num_q_heads),
        ctypes.c_int64(num_kv_heads),
        ctypes.c_int64(head_dim),
        ctypes.c_float(scale),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def laguna_global_attention_decode_fused_exact_gated_gqa1_fixedshape_bf16_spans(
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    gate_ptr: int,
    gated_out_ptr: int,
    score_scratch_ptr: int,
    physical_scratch_ptr: int,
    spans: KVLiveSpans,
    scan_slots: int,
    max_context_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch exact fused one-head global attention at Laguna's natural shape."""

    capacity = _check_global_spans(spans, num_kv_heads, head_dim)
    parsed_scan = _check_split_scan_slots(scan_slots, capacity)
    if (
        capacity != 4096
        or int(max_context_len) != 4096
        or int(num_q_heads) != 48
        or int(num_kv_heads) != 8
        or int(head_dim) != 128
    ):
        raise ValueError(
            "fused GQA1 fixed-shape global attention requires capacity 4096, "
            "48 query heads, 8 KV heads, and D128"
        )
    library = library or build_laguna_kv_attention(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(
        library,
        _SYMBOL_GLOBAL_ATTENTION_FUSED_EXACT_GATED_GQA1_FIXEDSHAPE,
    )
    fn.argtypes = (
        [ctypes.c_void_p] * 13
        + [ctypes.c_int64] * 8
        + [ctypes.c_float, ctypes.c_void_p]
    )
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(query_ptr),
        ctypes.c_void_p(key_cache_ptr),
        ctypes.c_void_p(value_cache_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_void_p(gate_ptr),
        ctypes.c_void_p(gated_out_ptr),
        ctypes.c_void_p(score_scratch_ptr),
        ctypes.c_void_p(physical_scratch_ptr),
        ctypes.c_void_p(spans.base_offsets.ptr),
        ctypes.c_void_p(spans.live_counts.ptr),
        ctypes.c_void_p(spans.token_positions.ptr),
        ctypes.c_void_p(spans.evict_mask.ptr),
        ctypes.c_void_p(spans.row_positions.ptr),
        ctypes.c_int64(capacity),
        ctypes.c_int64(_GLOBAL_BLOCK_SIZE),
        ctypes.c_int64(spans.base_offsets.numel),
        ctypes.c_int64(capacity),
        ctypes.c_int64(parsed_scan),
        ctypes.c_int64(num_q_heads),
        ctypes.c_int64(num_kv_heads),
        ctypes.c_int64(head_dim),
        ctypes.c_float(scale),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def laguna_global_attention_decode_fused_exact_gated_gqa2_vstage64_fixedshape_bf16_spans(
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    gate_ptr: int,
    gated_out_ptr: int,
    score_scratch_ptr: int,
    physical_scratch_ptr: int,
    spans: KVLiveSpans,
    scan_slots: int,
    max_context_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch exact fused GQA2 global attention with 64-slot staged V."""

    capacity = _check_global_spans(spans, num_kv_heads, head_dim)
    parsed_scan = _check_split_scan_slots(scan_slots, capacity)
    if (
        capacity != 4096
        or parsed_scan > 4000
        or int(max_context_len) != 4096
        or int(num_q_heads) != 48
        or int(num_kv_heads) != 8
        or int(head_dim) != 128
    ):
        raise ValueError(
            "fused GQA2 V-stage64 fixed-shape global attention requires "
            "capacity 4096, scan slots at most 4000, 48 query heads, "
            "8 KV heads, and D128"
        )
    library = library or build_laguna_kv_attention(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(
        library,
        _SYMBOL_GLOBAL_ATTENTION_FUSED_EXACT_GATED_GQA2_VSTAGE64_FIXEDSHAPE,
    )
    fn.argtypes = (
        [ctypes.c_void_p] * 13
        + [ctypes.c_int64] * 8
        + [ctypes.c_float, ctypes.c_void_p]
    )
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(query_ptr),
        ctypes.c_void_p(key_cache_ptr),
        ctypes.c_void_p(value_cache_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_void_p(gate_ptr),
        ctypes.c_void_p(gated_out_ptr),
        ctypes.c_void_p(score_scratch_ptr),
        ctypes.c_void_p(physical_scratch_ptr),
        ctypes.c_void_p(spans.base_offsets.ptr),
        ctypes.c_void_p(spans.live_counts.ptr),
        ctypes.c_void_p(spans.token_positions.ptr),
        ctypes.c_void_p(spans.evict_mask.ptr),
        ctypes.c_void_p(spans.row_positions.ptr),
        ctypes.c_int64(capacity),
        ctypes.c_int64(_GLOBAL_BLOCK_SIZE),
        ctypes.c_int64(spans.base_offsets.numel),
        ctypes.c_int64(capacity),
        ctypes.c_int64(parsed_scan),
        ctypes.c_int64(num_q_heads),
        ctypes.c_int64(num_kv_heads),
        ctypes.c_int64(head_dim),
        ctypes.c_float(scale),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def laguna_global_attention_decode_fused_exact_gated_gqa2_vstage64_vec16_fixedshape_bf16_spans(
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    gate_ptr: int,
    gated_out_ptr: int,
    score_scratch_ptr: int,
    physical_scratch_ptr: int,
    spans: KVLiveSpans,
    scan_slots: int,
    max_context_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch exact global GQA2 with aligned 16-byte staged-V copies."""

    capacity = _check_global_spans(spans, num_kv_heads, head_dim)
    parsed_scan = _check_split_scan_slots(scan_slots, capacity)
    if (
        capacity != 4096
        or parsed_scan > 4000
        or int(max_context_len) != 4096
        or int(num_q_heads) != 48
        or int(num_kv_heads) != 8
        or int(head_dim) != 128
    ):
        raise ValueError(
            "fused global GQA2 V-stage64 vec16 requires capacity 4096, "
            "scan slots at most 4000, 48 query heads, 8 KV heads, and D128"
        )
    library = library or build_laguna_kv_attention(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(
        library,
        _SYMBOL_GLOBAL_ATTENTION_FUSED_EXACT_GATED_GQA2_VSTAGE64_VEC16_FIXEDSHAPE,
    )
    fn.argtypes = (
        [ctypes.c_void_p] * 13
        + [ctypes.c_int64] * 8
        + [ctypes.c_float, ctypes.c_void_p]
    )
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(query_ptr),
        ctypes.c_void_p(key_cache_ptr),
        ctypes.c_void_p(value_cache_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_void_p(gate_ptr),
        ctypes.c_void_p(gated_out_ptr),
        ctypes.c_void_p(score_scratch_ptr),
        ctypes.c_void_p(physical_scratch_ptr),
        ctypes.c_void_p(spans.base_offsets.ptr),
        ctypes.c_void_p(spans.live_counts.ptr),
        ctypes.c_void_p(spans.token_positions.ptr),
        ctypes.c_void_p(spans.evict_mask.ptr),
        ctypes.c_void_p(spans.row_positions.ptr),
        ctypes.c_int64(capacity),
        ctypes.c_int64(_GLOBAL_BLOCK_SIZE),
        ctypes.c_int64(spans.base_offsets.numel),
        ctypes.c_int64(capacity),
        ctypes.c_int64(parsed_scan),
        ctypes.c_int64(num_q_heads),
        ctypes.c_int64(num_kv_heads),
        ctypes.c_int64(head_dim),
        ctypes.c_float(scale),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def laguna_global_attention_decode_fused_exact_gated_gqa2_vstage64_vec16_direct_fixedshape_bf16_spans(
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    gate_ptr: int,
    gated_out_ptr: int,
    score_scratch_ptr: int,
    physical_scratch_ptr: int,
    spans: KVLiveSpans,
    scan_slots: int,
    max_context_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch exact global GQA2 with direct vec16 V-to-LDS stores."""

    capacity = _check_global_spans(spans, num_kv_heads, head_dim)
    parsed_scan = _check_split_scan_slots(scan_slots, capacity)
    if (
        capacity != 4096
        or parsed_scan > 4000
        or int(max_context_len) != 4096
        or int(num_q_heads) != 48
        or int(num_kv_heads) != 8
        or int(head_dim) != 128
    ):
        raise ValueError(
            "fused global GQA2 V-stage64 vec16-direct requires capacity 4096, "
            "scan slots at most 4000, 48 query heads, 8 KV heads, and D128"
        )
    library = library or build_laguna_kv_attention(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(
        library,
        _SYMBOL_GLOBAL_ATTENTION_FUSED_EXACT_GATED_GQA2_VSTAGE64_VEC16_DIRECT_FIXEDSHAPE,
    )
    fn.argtypes = (
        [ctypes.c_void_p] * 13
        + [ctypes.c_int64] * 8
        + [ctypes.c_float, ctypes.c_void_p]
    )
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(query_ptr),
        ctypes.c_void_p(key_cache_ptr),
        ctypes.c_void_p(value_cache_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_void_p(gate_ptr),
        ctypes.c_void_p(gated_out_ptr),
        ctypes.c_void_p(score_scratch_ptr),
        ctypes.c_void_p(physical_scratch_ptr),
        ctypes.c_void_p(spans.base_offsets.ptr),
        ctypes.c_void_p(spans.live_counts.ptr),
        ctypes.c_void_p(spans.token_positions.ptr),
        ctypes.c_void_p(spans.evict_mask.ptr),
        ctypes.c_void_p(spans.row_positions.ptr),
        ctypes.c_int64(capacity),
        ctypes.c_int64(_GLOBAL_BLOCK_SIZE),
        ctypes.c_int64(spans.base_offsets.numel),
        ctypes.c_int64(capacity),
        ctypes.c_int64(parsed_scan),
        ctypes.c_int64(num_q_heads),
        ctypes.c_int64(num_kv_heads),
        ctypes.c_int64(head_dim),
        ctypes.c_float(scale),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def laguna_global_attention_decode_fused_exact_gated_gqa2_vstage64_vec16_direct_assume_exp_fixedshape_bf16_spans(
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    gate_ptr: int,
    gated_out_ptr: int,
    score_scratch_ptr: int,
    physical_scratch_ptr: int,
    spans: KVLiveSpans,
    scan_slots: int,
    max_context_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch exact direct-vec16 global GQA2 with bounded exp domain."""

    capacity = _check_global_spans(spans, num_kv_heads, head_dim)
    parsed_scan = _check_split_scan_slots(scan_slots, capacity)
    if (
        capacity != 4096
        or parsed_scan > 4000
        or int(max_context_len) != 4096
        or int(num_q_heads) != 48
        or int(num_kv_heads) != 8
        or int(head_dim) != 128
    ):
        raise ValueError(
            "fused global GQA2 vec16-direct assume-exp requires capacity "
            "4096, scan slots at most 4000, 48 query heads, 8 KV heads, "
            "and D128"
        )
    library = library or build_laguna_kv_attention(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(
        library,
        _SYMBOL_GLOBAL_ATTENTION_FUSED_EXACT_GATED_GQA2_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXEDSHAPE,
    )
    fn.argtypes = (
        [ctypes.c_void_p] * 13
        + [ctypes.c_int64] * 8
        + [ctypes.c_float, ctypes.c_void_p]
    )
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(query_ptr),
        ctypes.c_void_p(key_cache_ptr),
        ctypes.c_void_p(value_cache_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_void_p(gate_ptr),
        ctypes.c_void_p(gated_out_ptr),
        ctypes.c_void_p(score_scratch_ptr),
        ctypes.c_void_p(physical_scratch_ptr),
        ctypes.c_void_p(spans.base_offsets.ptr),
        ctypes.c_void_p(spans.live_counts.ptr),
        ctypes.c_void_p(spans.token_positions.ptr),
        ctypes.c_void_p(spans.evict_mask.ptr),
        ctypes.c_void_p(spans.row_positions.ptr),
        ctypes.c_int64(capacity),
        ctypes.c_int64(_GLOBAL_BLOCK_SIZE),
        ctypes.c_int64(spans.base_offsets.numel),
        ctypes.c_int64(capacity),
        ctypes.c_int64(parsed_scan),
        ctypes.c_int64(num_q_heads),
        ctypes.c_int64(num_kv_heads),
        ctypes.c_int64(head_dim),
        ctypes.c_float(scale),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def laguna_global_attention_prefill_bf16_spans(
    query_ptr: int,
    current_key_ptr: int,
    current_value_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    spans: KVLiveSpans,
    rows: int,
    max_context_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Run causal block-256 global GQA over prior cache plus current rows."""

    capacity = _check_global_spans(spans, num_kv_heads, head_dim)
    _check_prefill_rows(spans, rows, capacity)
    if int(max_context_len) != capacity:
        raise ValueError("max_context_len must equal the global span capacity")
    if capacity > _MAX_EAGER_GLOBAL_CONTEXT:
        raise ValueError("Laguna global prefill currently supports at most 4096 tokens")
    _check_laguna_attention_shape(num_q_heads, num_kv_heads, head_dim)
    library = library or build_laguna_kv_attention(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_GLOBAL_PREFILL)
    fn.argtypes = (
        [ctypes.c_void_p] * 11
        + [ctypes.c_int64] * 7
        + [ctypes.c_float, ctypes.c_void_p]
    )
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(query_ptr),
        ctypes.c_void_p(current_key_ptr),
        ctypes.c_void_p(current_value_ptr),
        ctypes.c_void_p(key_cache_ptr),
        ctypes.c_void_p(value_cache_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_void_p(spans.base_offsets.ptr),
        ctypes.c_void_p(spans.live_counts.ptr),
        ctypes.c_void_p(spans.token_positions.ptr),
        ctypes.c_void_p(spans.evict_mask.ptr),
        ctypes.c_void_p(spans.row_positions.ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(capacity),
        ctypes.c_int64(_GLOBAL_BLOCK_SIZE),
        ctypes.c_int64(spans.base_offsets.numel),
        ctypes.c_int64(num_q_heads),
        ctypes.c_int64(num_kv_heads),
        ctypes.c_int64(head_dim),
        ctypes.c_float(scale),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def laguna_global_attention_prefill_qrow2_online_bf16_spans(
    query_ptr: int,
    current_key_ptr: int,
    current_value_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    spans: KVLiveSpans,
    rows: int,
    max_context_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Run two-query-row online-softmax global GQA over prior and current rows."""

    capacity = _check_global_spans(spans, num_kv_heads, head_dim)
    _check_prefill_rows(spans, rows, capacity)
    if int(max_context_len) != capacity:
        raise ValueError("max_context_len must equal the global span capacity")
    _check_laguna_attention_shape(num_q_heads, num_kv_heads, head_dim)
    library = library or build_laguna_kv_attention(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_GLOBAL_PREFILL_QROW2_ONLINE)
    fn.argtypes = (
        [ctypes.c_void_p] * 11
        + [ctypes.c_int64] * 7
        + [ctypes.c_float, ctypes.c_void_p]
    )
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(query_ptr),
        ctypes.c_void_p(current_key_ptr),
        ctypes.c_void_p(current_value_ptr),
        ctypes.c_void_p(key_cache_ptr),
        ctypes.c_void_p(value_cache_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_void_p(spans.base_offsets.ptr),
        ctypes.c_void_p(spans.live_counts.ptr),
        ctypes.c_void_p(spans.token_positions.ptr),
        ctypes.c_void_p(spans.evict_mask.ptr),
        ctypes.c_void_p(spans.row_positions.ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(capacity),
        ctypes.c_int64(_GLOBAL_BLOCK_SIZE),
        ctypes.c_int64(spans.base_offsets.numel),
        ctypes.c_int64(num_q_heads),
        ctypes.c_int64(num_kv_heads),
        ctypes.c_int64(head_dim),
        ctypes.c_float(scale),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def laguna_global_attention_prefill_qrow4_online_bf16_spans(
    query_ptr: int,
    current_key_ptr: int,
    current_value_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    spans: KVLiveSpans,
    rows: int,
    max_context_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Run four-query-row online-softmax global GQA over prior and current rows."""

    capacity = _check_global_spans(spans, num_kv_heads, head_dim)
    _check_prefill_rows(spans, rows, capacity)
    if int(max_context_len) != capacity:
        raise ValueError("max_context_len must equal the global span capacity")
    _check_laguna_attention_shape(num_q_heads, num_kv_heads, head_dim)
    library = library or build_laguna_kv_attention(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_GLOBAL_PREFILL_QROW4_ONLINE)
    fn.argtypes = (
        [ctypes.c_void_p] * 11
        + [ctypes.c_int64] * 7
        + [ctypes.c_float, ctypes.c_void_p]
    )
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(query_ptr),
        ctypes.c_void_p(current_key_ptr),
        ctypes.c_void_p(current_value_ptr),
        ctypes.c_void_p(key_cache_ptr),
        ctypes.c_void_p(value_cache_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_void_p(spans.base_offsets.ptr),
        ctypes.c_void_p(spans.live_counts.ptr),
        ctypes.c_void_p(spans.token_positions.ptr),
        ctypes.c_void_p(spans.evict_mask.ptr),
        ctypes.c_void_p(spans.row_positions.ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(capacity),
        ctypes.c_int64(_GLOBAL_BLOCK_SIZE),
        ctypes.c_int64(spans.base_offsets.numel),
        ctypes.c_int64(num_q_heads),
        ctypes.c_int64(num_kv_heads),
        ctypes.c_int64(head_dim),
        ctypes.c_float(scale),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def laguna_global_attention_prefill_qrow4_cached_online_bf16_spans(
    query_ptr: int,
    current_key_ptr: int,
    current_value_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    spans: KVLiveSpans,
    rows: int,
    max_context_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Run qrow4 global GQA after current rows are already in the BF16 cache."""

    capacity = _check_global_spans(spans, num_kv_heads, head_dim)
    _check_prefill_rows(spans, rows, capacity)
    if int(max_context_len) != capacity:
        raise ValueError("max_context_len must equal the global span capacity")
    _check_laguna_attention_shape(num_q_heads, num_kv_heads, head_dim)
    library = library or build_laguna_kv_attention(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_GLOBAL_PREFILL_QROW4_CACHED_ONLINE)
    fn.argtypes = (
        [ctypes.c_void_p] * 11
        + [ctypes.c_int64] * 7
        + [ctypes.c_float, ctypes.c_void_p]
    )
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(query_ptr),
        ctypes.c_void_p(current_key_ptr),
        ctypes.c_void_p(current_value_ptr),
        ctypes.c_void_p(key_cache_ptr),
        ctypes.c_void_p(value_cache_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_void_p(spans.base_offsets.ptr),
        ctypes.c_void_p(spans.live_counts.ptr),
        ctypes.c_void_p(spans.token_positions.ptr),
        ctypes.c_void_p(spans.evict_mask.ptr),
        ctypes.c_void_p(spans.row_positions.ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(capacity),
        ctypes.c_int64(_GLOBAL_BLOCK_SIZE),
        ctypes.c_int64(spans.base_offsets.numel),
        ctypes.c_int64(num_q_heads),
        ctypes.c_int64(num_kv_heads),
        ctypes.c_int64(head_dim),
        ctypes.c_float(scale),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def laguna_global_attention_prefill_qrow4_cached_meta_online_bf16_spans(
    query_ptr: int,
    current_key_ptr: int,
    current_value_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    spans: KVLiveSpans,
    rows: int,
    max_context_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Run qrow4 global GQA using only preappended cache visibility metadata."""

    capacity = _check_global_spans(spans, num_kv_heads, head_dim)
    _check_prefill_rows(spans, rows, capacity)
    if int(max_context_len) != capacity:
        raise ValueError("max_context_len must equal the global span capacity")
    _check_laguna_attention_shape(num_q_heads, num_kv_heads, head_dim)
    library = library or build_laguna_kv_attention(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_GLOBAL_PREFILL_QROW4_CACHED_META_ONLINE)
    fn.argtypes = (
        [ctypes.c_void_p] * 11
        + [ctypes.c_int64] * 7
        + [ctypes.c_float, ctypes.c_void_p]
    )
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(query_ptr),
        ctypes.c_void_p(current_key_ptr),
        ctypes.c_void_p(current_value_ptr),
        ctypes.c_void_p(key_cache_ptr),
        ctypes.c_void_p(value_cache_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_void_p(spans.base_offsets.ptr),
        ctypes.c_void_p(spans.live_counts.ptr),
        ctypes.c_void_p(spans.token_positions.ptr),
        ctypes.c_void_p(spans.evict_mask.ptr),
        ctypes.c_void_p(spans.row_positions.ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(capacity),
        ctypes.c_int64(_GLOBAL_BLOCK_SIZE),
        ctypes.c_int64(spans.base_offsets.numel),
        ctypes.c_int64(num_q_heads),
        ctypes.c_int64(num_kv_heads),
        ctypes.c_int64(head_dim),
        ctypes.c_float(scale),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def laguna_global_attention_prefill_qrow6_cached_meta_online_bf16_spans(
    query_ptr: int,
    current_key_ptr: int,
    current_value_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    spans: KVLiveSpans,
    rows: int,
    max_context_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Run diagnostic qrow6 global GQA from preappended cache metadata."""

    capacity = _check_global_spans(spans, num_kv_heads, head_dim)
    _check_prefill_rows(spans, rows, capacity)
    if int(max_context_len) != capacity:
        raise ValueError("max_context_len must equal the global span capacity")
    _check_laguna_attention_shape(num_q_heads, num_kv_heads, head_dim)
    library = library or build_laguna_kv_attention(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_GLOBAL_PREFILL_QROW6_CACHED_META_ONLINE)
    fn.argtypes = (
        [ctypes.c_void_p] * 11
        + [ctypes.c_int64] * 7
        + [ctypes.c_float, ctypes.c_void_p]
    )
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(query_ptr),
        ctypes.c_void_p(current_key_ptr),
        ctypes.c_void_p(current_value_ptr),
        ctypes.c_void_p(key_cache_ptr),
        ctypes.c_void_p(value_cache_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_void_p(spans.base_offsets.ptr),
        ctypes.c_void_p(spans.live_counts.ptr),
        ctypes.c_void_p(spans.token_positions.ptr),
        ctypes.c_void_p(spans.evict_mask.ptr),
        ctypes.c_void_p(spans.row_positions.ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(capacity),
        ctypes.c_int64(_GLOBAL_BLOCK_SIZE),
        ctypes.c_int64(spans.base_offsets.numel),
        ctypes.c_int64(num_q_heads),
        ctypes.c_int64(num_kv_heads),
        ctypes.c_int64(head_dim),
        ctypes.c_float(scale),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def _laguna_global_attention_prefill_dense_initial_online_bf16_spans(
    query_ptr: int,
    current_key_ptr: int,
    current_value_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    spans: KVLiveSpans,
    rows: int,
    max_context_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    symbol: str,
    start_position: int | None,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    capacity = _check_global_spans(spans, num_kv_heads, head_dim)
    _check_prefill_rows(spans, rows, capacity)
    if int(max_context_len) != capacity:
        raise ValueError("max_context_len must equal the global span capacity")
    if start_position is None or int(start_position) < 0:
        raise ValueError("dense-initial global prefill requires a start position")
    if int(start_position) + int(rows) > capacity:
        raise ValueError("dense-initial global prefill cannot exceed capacity")
    _check_laguna_attention_shape(num_q_heads, num_kv_heads, head_dim)
    library = library or build_laguna_kv_attention(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, symbol)
    fn.argtypes = (
        [ctypes.c_void_p] * 11
        + [ctypes.c_int64] * 7
        + [ctypes.c_float, ctypes.c_void_p]
    )
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(query_ptr),
        ctypes.c_void_p(current_key_ptr),
        ctypes.c_void_p(current_value_ptr),
        ctypes.c_void_p(key_cache_ptr),
        ctypes.c_void_p(value_cache_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_void_p(spans.base_offsets.ptr),
        ctypes.c_void_p(spans.live_counts.ptr),
        ctypes.c_void_p(spans.token_positions.ptr),
        ctypes.c_void_p(spans.evict_mask.ptr),
        ctypes.c_void_p(spans.row_positions.ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(capacity),
        ctypes.c_int64(_GLOBAL_BLOCK_SIZE),
        ctypes.c_int64(spans.base_offsets.numel),
        ctypes.c_int64(num_q_heads),
        ctypes.c_int64(num_kv_heads),
        ctypes.c_int64(head_dim),
        ctypes.c_float(scale),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def laguna_global_attention_prefill_qrow4_dense_initial_online_bf16_spans(
    query_ptr: int,
    current_key_ptr: int,
    current_value_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    spans: KVLiveSpans,
    rows: int,
    max_context_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    start_position: int | None = None,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Run qrow4 global attention over an identity-positioned initial fill."""

    _laguna_global_attention_prefill_dense_initial_online_bf16_spans(
        query_ptr,
        current_key_ptr,
        current_value_ptr,
        key_cache_ptr,
        value_cache_ptr,
        out_ptr,
        spans,
        rows,
        max_context_len,
        num_q_heads,
        num_kv_heads,
        head_dim,
        scale,
        symbol=_SYMBOL_GLOBAL_PREFILL_QROW4_DENSE_INITIAL_ONLINE,
        start_position=start_position,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def laguna_global_attention_decode_fused_exact_gated_gqa2_exp32_vstage64_vec16_direct_assume_exp_fixedshape_bf16_spans(
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    gate_ptr: int,
    gated_out_ptr: int,
    score_scratch_ptr: int,
    physical_scratch_ptr: int,
    spans: KVLiveSpans,
    scan_slots: int,
    max_context_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch exact global GQA2 with wave-wide exponential issue."""

    capacity = _check_global_spans(spans, num_kv_heads, head_dim)
    parsed_scan = _check_split_scan_slots(scan_slots, capacity)
    if (
        capacity != 4096
        or parsed_scan > 4000
        or int(max_context_len) != 4096
        or int(num_q_heads) != 48
        or int(num_kv_heads) != 8
        or int(head_dim) != 128
    ):
        raise ValueError(
            "fused global GQA2 exp32 vec16-direct assume-exp requires "
            "capacity 4096, scan slots at most 4000, 48 query heads, "
            "8 KV heads, and D128"
        )
    library = library or build_laguna_kv_attention(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(
        library,
        _SYMBOL_GLOBAL_ATTENTION_FUSED_EXACT_GATED_GQA2_EXP32_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXEDSHAPE,
    )
    fn.argtypes = (
        [ctypes.c_void_p] * 13
        + [ctypes.c_int64] * 8
        + [ctypes.c_float, ctypes.c_void_p]
    )
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(query_ptr),
        ctypes.c_void_p(key_cache_ptr),
        ctypes.c_void_p(value_cache_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_void_p(gate_ptr),
        ctypes.c_void_p(gated_out_ptr),
        ctypes.c_void_p(score_scratch_ptr),
        ctypes.c_void_p(physical_scratch_ptr),
        ctypes.c_void_p(spans.base_offsets.ptr),
        ctypes.c_void_p(spans.live_counts.ptr),
        ctypes.c_void_p(spans.token_positions.ptr),
        ctypes.c_void_p(spans.evict_mask.ptr),
        ctypes.c_void_p(spans.row_positions.ptr),
        ctypes.c_int64(capacity),
        ctypes.c_int64(_GLOBAL_BLOCK_SIZE),
        ctypes.c_int64(spans.base_offsets.numel),
        ctypes.c_int64(capacity),
        ctypes.c_int64(parsed_scan),
        ctypes.c_int64(num_q_heads),
        ctypes.c_int64(num_kv_heads),
        ctypes.c_int64(head_dim),
        ctypes.c_float(scale),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def laguna_global_attention_decode_fused_exact_gated_mixed32_exp32_vstage64_vec16_direct_assume_exp_fixedshape_bf16_spans(
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    gate_ptr: int,
    gated_out_ptr: int,
    score_scratch_ptr: int,
    physical_scratch_ptr: int,
    spans: KVLiveSpans,
    scan_slots: int,
    max_context_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch exact global mixed32 GQA with wave-wide exponential issue."""

    capacity = _check_global_spans(spans, num_kv_heads, head_dim)
    parsed_scan = _check_split_scan_slots(scan_slots, capacity)
    if (
        capacity != 4096
        or parsed_scan > 4000
        or int(max_context_len) != 4096
        or int(num_q_heads) != 48
        or int(num_kv_heads) != 8
        or int(head_dim) != 128
    ):
        raise ValueError(
            "fused global mixed32 exp32 vec16-direct assume-exp requires "
            "capacity 4096, scan slots at most 4000, 48 query heads, "
            "8 KV heads, and D128"
        )
    library = library or build_laguna_kv_attention(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(
        library,
        _SYMBOL_GLOBAL_ATTENTION_FUSED_EXACT_GATED_MIXED32_EXP32_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXEDSHAPE,
    )
    fn.argtypes = (
        [ctypes.c_void_p] * 13
        + [ctypes.c_int64] * 8
        + [ctypes.c_float, ctypes.c_void_p]
    )
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(query_ptr),
        ctypes.c_void_p(key_cache_ptr),
        ctypes.c_void_p(value_cache_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_void_p(gate_ptr),
        ctypes.c_void_p(gated_out_ptr),
        ctypes.c_void_p(score_scratch_ptr),
        ctypes.c_void_p(physical_scratch_ptr),
        ctypes.c_void_p(spans.base_offsets.ptr),
        ctypes.c_void_p(spans.live_counts.ptr),
        ctypes.c_void_p(spans.token_positions.ptr),
        ctypes.c_void_p(spans.evict_mask.ptr),
        ctypes.c_void_p(spans.row_positions.ptr),
        ctypes.c_int64(capacity),
        ctypes.c_int64(_GLOBAL_BLOCK_SIZE),
        ctypes.c_int64(spans.base_offsets.numel),
        ctypes.c_int64(capacity),
        ctypes.c_int64(parsed_scan),
        ctypes.c_int64(num_q_heads),
        ctypes.c_int64(num_kv_heads),
        ctypes.c_int64(head_dim),
        ctypes.c_float(scale),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def laguna_global_attention_decode_fused_exact_gated_mixed32_exp32_producer_max_vstage64_vec16_direct_assume_exp_fixedshape_bf16_spans(
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    gate_ptr: int,
    gated_out_ptr: int,
    score_scratch_ptr: int,
    physical_scratch_ptr: int,
    spans: KVLiveSpans,
    scan_slots: int,
    max_context_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch exact global mixed32 GQA with score-producer maxima."""

    capacity = _check_global_spans(spans, num_kv_heads, head_dim)
    parsed_scan = _check_split_scan_slots(scan_slots, capacity)
    if (
        capacity != 4096
        or parsed_scan > 4000
        or int(max_context_len) != 4096
        or int(num_q_heads) != 48
        or int(num_kv_heads) != 8
        or int(head_dim) != 128
    ):
        raise ValueError(
            "fused global mixed32 exp32 producer-max vec16-direct "
            "assume-exp requires capacity 4096, scan slots at most 4000, "
            "48 query heads, 8 KV heads, and D128"
        )
    library = library or build_laguna_kv_attention(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(
        library,
        _SYMBOL_GLOBAL_ATTENTION_FUSED_EXACT_GATED_MIXED32_EXP32_PRODUCER_MAX_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXEDSHAPE,
    )
    fn.argtypes = (
        [ctypes.c_void_p] * 13
        + [ctypes.c_int64] * 8
        + [ctypes.c_float, ctypes.c_void_p]
    )
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(query_ptr),
        ctypes.c_void_p(key_cache_ptr),
        ctypes.c_void_p(value_cache_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_void_p(gate_ptr),
        ctypes.c_void_p(gated_out_ptr),
        ctypes.c_void_p(score_scratch_ptr),
        ctypes.c_void_p(physical_scratch_ptr),
        ctypes.c_void_p(spans.base_offsets.ptr),
        ctypes.c_void_p(spans.live_counts.ptr),
        ctypes.c_void_p(spans.token_positions.ptr),
        ctypes.c_void_p(spans.evict_mask.ptr),
        ctypes.c_void_p(spans.row_positions.ptr),
        ctypes.c_int64(capacity),
        ctypes.c_int64(_GLOBAL_BLOCK_SIZE),
        ctypes.c_int64(spans.base_offsets.numel),
        ctypes.c_int64(capacity),
        ctypes.c_int64(parsed_scan),
        ctypes.c_int64(num_q_heads),
        ctypes.c_int64(num_kv_heads),
        ctypes.c_int64(head_dim),
        ctypes.c_float(scale),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def laguna_global_attention_decode_fused_exact_gated_mixed32_exp32_producer_max_dpp_qk_vstage64_vec16_direct_assume_exp_fixedshape_bf16_spans(
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    gate_ptr: int,
    gated_out_ptr: int,
    score_scratch_ptr: int,
    physical_scratch_ptr: int,
    spans: KVLiveSpans,
    scan_slots: int,
    max_context_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch exact global mixed32 GQA with DPP QK reduction."""

    capacity = _check_global_spans(spans, num_kv_heads, head_dim)
    parsed_scan = _check_split_scan_slots(scan_slots, capacity)
    if (
        capacity != 4096
        or parsed_scan > 4000
        or int(max_context_len) != 4096
        or int(num_q_heads) != 48
        or int(num_kv_heads) != 8
        or int(head_dim) != 128
    ):
        raise ValueError(
            "fused global mixed32 exp32 producer-max DPP-QK vec16-direct "
            "assume-exp requires capacity 4096, scan slots at most 4000, "
            "48 query heads, 8 KV heads, and D128"
        )
    library = library or build_laguna_kv_attention(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(
        library,
        _SYMBOL_GLOBAL_ATTENTION_FUSED_EXACT_GATED_MIXED32_EXP32_PRODUCER_MAX_DPP_QK_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXEDSHAPE,
    )
    fn.argtypes = (
        [ctypes.c_void_p] * 13
        + [ctypes.c_int64] * 8
        + [ctypes.c_float, ctypes.c_void_p]
    )
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(query_ptr),
        ctypes.c_void_p(key_cache_ptr),
        ctypes.c_void_p(value_cache_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_void_p(gate_ptr),
        ctypes.c_void_p(gated_out_ptr),
        ctypes.c_void_p(score_scratch_ptr),
        ctypes.c_void_p(physical_scratch_ptr),
        ctypes.c_void_p(spans.base_offsets.ptr),
        ctypes.c_void_p(spans.live_counts.ptr),
        ctypes.c_void_p(spans.token_positions.ptr),
        ctypes.c_void_p(spans.evict_mask.ptr),
        ctypes.c_void_p(spans.row_positions.ptr),
        ctypes.c_int64(capacity),
        ctypes.c_int64(_GLOBAL_BLOCK_SIZE),
        ctypes.c_int64(spans.base_offsets.numel),
        ctypes.c_int64(capacity),
        ctypes.c_int64(parsed_scan),
        ctypes.c_int64(num_q_heads),
        ctypes.c_int64(num_kv_heads),
        ctypes.c_int64(head_dim),
        ctypes.c_float(scale),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def laguna_global_attention_decode_fused_exact_gated_mixed32_exp32_producer_max_dpp_qk_probability_vec4_vstage64_vec16_direct_assume_exp_fixedshape_bf16_spans(
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    gate_ptr: int,
    gated_out_ptr: int,
    score_scratch_ptr: int,
    physical_scratch_ptr: int,
    spans: KVLiveSpans,
    scan_slots: int,
    max_context_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch exact global DPP-QK with aligned vector probability replay."""

    capacity = _check_global_spans(spans, num_kv_heads, head_dim)
    parsed_scan = _check_split_scan_slots(scan_slots, capacity)
    if (
        capacity != 4096
        or parsed_scan > 4000
        or int(max_context_len) != 4096
        or int(num_q_heads) != 48
        or int(num_kv_heads) != 8
        or int(head_dim) != 128
    ):
        raise ValueError(
            "fused global mixed32 exp32 producer-max DPP-QK probability-vec4 "
            "requires capacity 4096, scan slots at most 4000, 48 query "
            "heads, 8 KV heads, and D128"
        )
    library = library or build_laguna_kv_attention(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(
        library,
        _SYMBOL_GLOBAL_ATTENTION_FUSED_EXACT_GATED_MIXED32_EXP32_PRODUCER_MAX_DPP_QK_PROBABILITY_VEC4_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXEDSHAPE,
    )
    fn.argtypes = (
        [ctypes.c_void_p] * 13
        + [ctypes.c_int64] * 8
        + [ctypes.c_float, ctypes.c_void_p]
    )
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(query_ptr),
        ctypes.c_void_p(key_cache_ptr),
        ctypes.c_void_p(value_cache_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_void_p(gate_ptr),
        ctypes.c_void_p(gated_out_ptr),
        ctypes.c_void_p(score_scratch_ptr),
        ctypes.c_void_p(physical_scratch_ptr),
        ctypes.c_void_p(spans.base_offsets.ptr),
        ctypes.c_void_p(spans.live_counts.ptr),
        ctypes.c_void_p(spans.token_positions.ptr),
        ctypes.c_void_p(spans.evict_mask.ptr),
        ctypes.c_void_p(spans.row_positions.ptr),
        ctypes.c_int64(capacity),
        ctypes.c_int64(_GLOBAL_BLOCK_SIZE),
        ctypes.c_int64(spans.base_offsets.numel),
        ctypes.c_int64(capacity),
        ctypes.c_int64(parsed_scan),
        ctypes.c_int64(num_q_heads),
        ctypes.c_int64(num_kv_heads),
        ctypes.c_int64(head_dim),
        ctypes.c_float(scale),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def laguna_global_attention_decode_fused_exact_gated_mixed32_exp32_producer_max_dpp_qk_probability_vec4_prenorm_vstage64_vec16_direct_assume_exp_fixedshape_bf16_spans(
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    gate_ptr: int,
    gated_out_ptr: int,
    score_scratch_ptr: int,
    physical_scratch_ptr: int,
    spans: KVLiveSpans,
    scan_slots: int,
    max_context_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch exact global DPP-QK with once-normalized probabilities."""

    capacity = _check_global_spans(spans, num_kv_heads, head_dim)
    parsed_scan = _check_split_scan_slots(scan_slots, capacity)
    if (
        capacity != 4096
        or parsed_scan > 4000
        or int(max_context_len) != 4096
        or int(num_q_heads) != 48
        or int(num_kv_heads) != 8
        or int(head_dim) != 128
    ):
        raise ValueError(
            "fused global mixed32 exp32 producer-max DPP-QK probability-vec4 "
            "prenorm requires capacity 4096, scan slots at most 4000, "
            "48 query heads, 8 KV heads, and D128"
        )
    library = library or build_laguna_kv_attention(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(
        library,
        _SYMBOL_GLOBAL_ATTENTION_FUSED_EXACT_GATED_MIXED32_EXP32_PRODUCER_MAX_DPP_QK_PROBABILITY_VEC4_PRENORM_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXEDSHAPE,
    )
    fn.argtypes = (
        [ctypes.c_void_p] * 13
        + [ctypes.c_int64] * 8
        + [ctypes.c_float, ctypes.c_void_p]
    )
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(query_ptr),
        ctypes.c_void_p(key_cache_ptr),
        ctypes.c_void_p(value_cache_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_void_p(gate_ptr),
        ctypes.c_void_p(gated_out_ptr),
        ctypes.c_void_p(score_scratch_ptr),
        ctypes.c_void_p(physical_scratch_ptr),
        ctypes.c_void_p(spans.base_offsets.ptr),
        ctypes.c_void_p(spans.live_counts.ptr),
        ctypes.c_void_p(spans.token_positions.ptr),
        ctypes.c_void_p(spans.evict_mask.ptr),
        ctypes.c_void_p(spans.row_positions.ptr),
        ctypes.c_int64(capacity),
        ctypes.c_int64(_GLOBAL_BLOCK_SIZE),
        ctypes.c_int64(spans.base_offsets.numel),
        ctypes.c_int64(capacity),
        ctypes.c_int64(parsed_scan),
        ctypes.c_int64(num_q_heads),
        ctypes.c_int64(num_kv_heads),
        ctypes.c_int64(head_dim),
        ctypes.c_float(scale),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def laguna_global_attention_decode_fused_exact_gated_mixed32_local512_exp32_producer_max_dpp_qk_probability_vec4_prenorm_vstage64_vec16_direct_assume_exp_fixedshape_bf16_spans(
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    gate_ptr: int,
    gated_out_ptr: int,
    score_scratch_ptr: int,
    physical_scratch_ptr: int,
    spans: KVLiveSpans,
    scan_slots: int,
    max_context_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Screen exact global mixed32 scheduling with a local512 workgroup."""

    capacity = _check_global_spans(spans, num_kv_heads, head_dim)
    parsed_scan = _check_split_scan_slots(scan_slots, capacity)
    if (
        capacity != 4096
        or parsed_scan > 4000
        or int(max_context_len) != 4096
        or int(num_q_heads) != 48
        or int(num_kv_heads) != 8
        or int(head_dim) != 128
    ):
        raise ValueError(
            "fused global mixed32 local512 exp32 producer-max DPP-QK "
            "probability-vec4 prenorm requires capacity 4096, scan slots at "
            "most 4000, 48 query heads, 8 KV heads, and D128"
        )
    library = library or build_laguna_kv_attention(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(
        library,
        _SYMBOL_GLOBAL_ATTENTION_FUSED_EXACT_GATED_MIXED32_LOCAL512_EXP32_PRODUCER_MAX_DPP_QK_PROBABILITY_VEC4_PRENORM_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXEDSHAPE,
    )
    fn.argtypes = (
        [ctypes.c_void_p] * 13
        + [ctypes.c_int64] * 8
        + [ctypes.c_float, ctypes.c_void_p]
    )
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(query_ptr),
        ctypes.c_void_p(key_cache_ptr),
        ctypes.c_void_p(value_cache_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_void_p(gate_ptr),
        ctypes.c_void_p(gated_out_ptr),
        ctypes.c_void_p(score_scratch_ptr),
        ctypes.c_void_p(physical_scratch_ptr),
        ctypes.c_void_p(spans.base_offsets.ptr),
        ctypes.c_void_p(spans.live_counts.ptr),
        ctypes.c_void_p(spans.token_positions.ptr),
        ctypes.c_void_p(spans.evict_mask.ptr),
        ctypes.c_void_p(spans.row_positions.ptr),
        ctypes.c_int64(capacity),
        ctypes.c_int64(_GLOBAL_BLOCK_SIZE),
        ctypes.c_int64(spans.base_offsets.numel),
        ctypes.c_int64(capacity),
        ctypes.c_int64(parsed_scan),
        ctypes.c_int64(num_q_heads),
        ctypes.c_int64(num_kv_heads),
        ctypes.c_int64(head_dim),
        ctypes.c_float(scale),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def laguna_global_attention_decode_fused_exact_gated_mixed40_local512_exp32_producer_max_dpp_qk_probability_vec4_prenorm_vstage64_vec16_direct_assume_exp_fixedshape_bf16_spans(
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    gate_ptr: int,
    gated_out_ptr: int,
    score_scratch_ptr: int,
    physical_scratch_ptr: int,
    spans: KVLiveSpans,
    scan_slots: int,
    max_context_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Screen exact global local512 ownership across all 40 gfx1151 CUs."""

    capacity = _check_global_spans(spans, num_kv_heads, head_dim)
    parsed_scan = _check_split_scan_slots(scan_slots, capacity)
    if (
        capacity != 4096
        or parsed_scan > 4000
        or int(max_context_len) != 4096
        or int(num_q_heads) != 48
        or int(num_kv_heads) != 8
        or int(head_dim) != 128
    ):
        raise ValueError(
            "fused global mixed40 local512 exp32 producer-max DPP-QK "
            "probability-vec4 prenorm requires capacity 4096, scan slots at "
            "most 4000, 48 query heads, 8 KV heads, and D128"
        )
    library = library or build_laguna_kv_attention(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(
        library,
        _SYMBOL_GLOBAL_ATTENTION_FUSED_EXACT_GATED_MIXED40_LOCAL512_EXP32_PRODUCER_MAX_DPP_QK_PROBABILITY_VEC4_PRENORM_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXEDSHAPE,
    )
    fn.argtypes = (
        [ctypes.c_void_p] * 13
        + [ctypes.c_int64] * 8
        + [ctypes.c_float, ctypes.c_void_p]
    )
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(query_ptr),
        ctypes.c_void_p(key_cache_ptr),
        ctypes.c_void_p(value_cache_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_void_p(gate_ptr),
        ctypes.c_void_p(gated_out_ptr),
        ctypes.c_void_p(score_scratch_ptr),
        ctypes.c_void_p(physical_scratch_ptr),
        ctypes.c_void_p(spans.base_offsets.ptr),
        ctypes.c_void_p(spans.live_counts.ptr),
        ctypes.c_void_p(spans.token_positions.ptr),
        ctypes.c_void_p(spans.evict_mask.ptr),
        ctypes.c_void_p(spans.row_positions.ptr),
        ctypes.c_int64(capacity),
        ctypes.c_int64(_GLOBAL_BLOCK_SIZE),
        ctypes.c_int64(spans.base_offsets.numel),
        ctypes.c_int64(capacity),
        ctypes.c_int64(parsed_scan),
        ctypes.c_int64(num_q_heads),
        ctypes.c_int64(num_kv_heads),
        ctypes.c_int64(head_dim),
        ctypes.c_float(scale),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def laguna_global_attention_decode_fused_exact_gated_mixed40_local512_exp32_producer_max_dpp_qk_dense_prefix_probability_vec4_prenorm_vstage64_vec16_direct_assume_exp_fixedshape_bf16_spans(
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    gate_ptr: int,
    gated_out_ptr: int,
    score_scratch_ptr: int,
    physical_scratch_ptr: int,
    spans: KVLiveSpans,
    scan_slots: int,
    max_context_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Run exact gfx1151 global attention on a dense identity KV prefix."""

    capacity = _check_global_spans(spans, num_kv_heads, head_dim)
    parsed_scan = _check_split_scan_slots(scan_slots, capacity)
    if (
        capacity != 4096
        or parsed_scan > 4000
        or int(max_context_len) != 4096
        or int(num_q_heads) != 48
        or int(num_kv_heads) != 8
        or int(head_dim) != 128
    ):
        raise ValueError(
            "fused global mixed40 local512 exp32 producer-max DPP-QK "
            "dense-prefix probability-vec4 prenorm requires capacity 4096, "
            "scan slots at most 4000, 48 query heads, 8 KV heads, and D128"
        )
    library = library or build_laguna_kv_attention(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(
        library,
        _SYMBOL_GLOBAL_ATTENTION_FUSED_EXACT_GATED_MIXED40_LOCAL512_EXP32_PRODUCER_MAX_DPP_QK_DENSE_PREFIX_PROBABILITY_VEC4_PRENORM_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXEDSHAPE,
    )
    fn.argtypes = (
        [ctypes.c_void_p] * 13
        + [ctypes.c_int64] * 8
        + [ctypes.c_float, ctypes.c_void_p]
    )
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(query_ptr),
        ctypes.c_void_p(key_cache_ptr),
        ctypes.c_void_p(value_cache_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_void_p(gate_ptr),
        ctypes.c_void_p(gated_out_ptr),
        ctypes.c_void_p(score_scratch_ptr),
        ctypes.c_void_p(physical_scratch_ptr),
        ctypes.c_void_p(spans.base_offsets.ptr),
        ctypes.c_void_p(spans.live_counts.ptr),
        ctypes.c_void_p(spans.token_positions.ptr),
        ctypes.c_void_p(spans.evict_mask.ptr),
        ctypes.c_void_p(spans.row_positions.ptr),
        ctypes.c_int64(capacity),
        ctypes.c_int64(_GLOBAL_BLOCK_SIZE),
        ctypes.c_int64(spans.base_offsets.numel),
        ctypes.c_int64(capacity),
        ctypes.c_int64(parsed_scan),
        ctypes.c_int64(num_q_heads),
        ctypes.c_int64(num_kv_heads),
        ctypes.c_int64(head_dim),
        ctypes.c_float(scale),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def laguna_global_attention_decode_fused_exact_gated_mixed40_local512_exp32_producer_max_dpp_qk_dense_prefix_idle_double_buffer_probability_vec4_prenorm_vstage64_vec16_direct_assume_exp_fixedshape_bf16_spans(
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    gate_ptr: int,
    gated_out_ptr: int,
    score_scratch_ptr: int,
    physical_scratch_ptr: int,
    spans: KVLiveSpans,
    scan_slots: int,
    max_context_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Screen exact dense-prefix global V64 ping-pong with idle loaders."""

    capacity = _check_global_spans(spans, num_kv_heads, head_dim)
    parsed_scan = _check_split_scan_slots(scan_slots, capacity)
    if (
        capacity != 4096
        or parsed_scan > 4000
        or int(max_context_len) != 4096
        or int(num_q_heads) != 48
        or int(num_kv_heads) != 8
        or int(head_dim) != 128
    ):
        raise ValueError(
            "fused global dense-prefix idle-double-buffer requires capacity "
            "4096, scan slots at most 4000, 48 query heads, 8 KV heads, and D128"
        )
    library = library or build_laguna_kv_attention(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(
        library,
        _SYMBOL_GLOBAL_ATTENTION_FUSED_EXACT_GATED_MIXED40_LOCAL512_EXP32_PRODUCER_MAX_DPP_QK_DENSE_PREFIX_IDLE_DOUBLE_BUFFER_PROBABILITY_VEC4_PRENORM_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXEDSHAPE,
    )
    fn.argtypes = (
        [ctypes.c_void_p] * 13
        + [ctypes.c_int64] * 8
        + [ctypes.c_float, ctypes.c_void_p]
    )
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(query_ptr),
        ctypes.c_void_p(key_cache_ptr),
        ctypes.c_void_p(value_cache_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_void_p(gate_ptr),
        ctypes.c_void_p(gated_out_ptr),
        ctypes.c_void_p(score_scratch_ptr),
        ctypes.c_void_p(physical_scratch_ptr),
        ctypes.c_void_p(spans.base_offsets.ptr),
        ctypes.c_void_p(spans.live_counts.ptr),
        ctypes.c_void_p(spans.token_positions.ptr),
        ctypes.c_void_p(spans.evict_mask.ptr),
        ctypes.c_void_p(spans.row_positions.ptr),
        ctypes.c_int64(capacity),
        ctypes.c_int64(_GLOBAL_BLOCK_SIZE),
        ctypes.c_int64(spans.base_offsets.numel),
        ctypes.c_int64(capacity),
        ctypes.c_int64(parsed_scan),
        ctypes.c_int64(num_q_heads),
        ctypes.c_int64(num_kv_heads),
        ctypes.c_int64(head_dim),
        ctypes.c_float(scale),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def laguna_global_attention_decode_fused_exact_gated_mixed40_local1024_exp32_producer_max_dpp_qk_dense_prefix_idle_double_buffer_probability_vec4_prenorm_vstage64_vec16_direct_assume_exp_fixedshape_bf16_spans(
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    gate_ptr: int,
    gated_out_ptr: int,
    score_scratch_ptr: int,
    physical_scratch_ptr: int,
    spans: KVLiveSpans,
    scan_slots: int,
    max_context_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Screen exact dense-prefix global local1024 QK/V transport."""

    capacity = _check_global_spans(spans, num_kv_heads, head_dim)
    parsed_scan = _check_split_scan_slots(scan_slots, capacity)
    if (
        capacity != 4096
        or parsed_scan > 4000
        or int(max_context_len) != 4096
        or int(num_q_heads) != 48
        or int(num_kv_heads) != 8
        or int(head_dim) != 128
    ):
        raise ValueError(
            "fused global dense-prefix local1024 idle-double-buffer requires "
            "capacity 4096, scan slots at most 4000, 48 query heads, 8 KV "
            "heads, and D128"
        )
    library = library or build_laguna_kv_attention(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(
        library,
        _SYMBOL_GLOBAL_ATTENTION_FUSED_EXACT_GATED_MIXED40_LOCAL1024_EXP32_PRODUCER_MAX_DPP_QK_DENSE_PREFIX_IDLE_DOUBLE_BUFFER_PROBABILITY_VEC4_PRENORM_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXEDSHAPE,
    )
    fn.argtypes = (
        [ctypes.c_void_p] * 13
        + [ctypes.c_int64] * 8
        + [ctypes.c_float, ctypes.c_void_p]
    )
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(query_ptr),
        ctypes.c_void_p(key_cache_ptr),
        ctypes.c_void_p(value_cache_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_void_p(gate_ptr),
        ctypes.c_void_p(gated_out_ptr),
        ctypes.c_void_p(score_scratch_ptr),
        ctypes.c_void_p(physical_scratch_ptr),
        ctypes.c_void_p(spans.base_offsets.ptr),
        ctypes.c_void_p(spans.live_counts.ptr),
        ctypes.c_void_p(spans.token_positions.ptr),
        ctypes.c_void_p(spans.evict_mask.ptr),
        ctypes.c_void_p(spans.row_positions.ptr),
        ctypes.c_int64(capacity),
        ctypes.c_int64(_GLOBAL_BLOCK_SIZE),
        ctypes.c_int64(spans.base_offsets.numel),
        ctypes.c_int64(capacity),
        ctypes.c_int64(parsed_scan),
        ctypes.c_int64(num_q_heads),
        ctypes.c_int64(num_kv_heads),
        ctypes.c_int64(head_dim),
        ctypes.c_float(scale),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def laguna_global_attention_decode_wmma_qk_three_term_mixed32_exp32_producer_max_exact_pv_bf16_spans(
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    gate_ptr: int,
    gated_out_ptr: int,
    score_scratch_ptr: int,
    physical_scratch_ptr: int,
    spans: KVLiveSpans,
    scan_slots: int,
    max_context_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch mixed32 three-term WMMA QK with ordered scalar PV."""

    capacity = _check_global_spans(spans, num_kv_heads, head_dim)
    parsed_scan = _check_split_scan_slots(scan_slots, capacity)
    if (
        capacity != 4096
        or parsed_scan > 4000
        or int(max_context_len) != 4096
        or int(num_q_heads) != 48
        or int(num_kv_heads) != 8
        or int(head_dim) != 128
    ):
        raise ValueError(
            "global mixed32 three-term WMMA QK requires capacity 4096, "
            "scan slots at most 4000, 48 query heads, 8 KV heads, and D128"
        )
    library = library or build_laguna_kv_attention(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(
        library,
        _SYMBOL_GLOBAL_ATTENTION_WMMA_QK_THREE_TERM_MIXED32_EXP32_PRODUCER_MAX_EXACT_PV,
    )
    fn.argtypes = (
        [ctypes.c_void_p] * 13
        + [ctypes.c_int64] * 8
        + [ctypes.c_float, ctypes.c_void_p]
    )
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(query_ptr),
        ctypes.c_void_p(key_cache_ptr),
        ctypes.c_void_p(value_cache_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_void_p(gate_ptr),
        ctypes.c_void_p(gated_out_ptr),
        ctypes.c_void_p(score_scratch_ptr),
        ctypes.c_void_p(physical_scratch_ptr),
        ctypes.c_void_p(spans.base_offsets.ptr),
        ctypes.c_void_p(spans.live_counts.ptr),
        ctypes.c_void_p(spans.token_positions.ptr),
        ctypes.c_void_p(spans.evict_mask.ptr),
        ctypes.c_void_p(spans.row_positions.ptr),
        ctypes.c_int64(capacity),
        ctypes.c_int64(_GLOBAL_BLOCK_SIZE),
        ctypes.c_int64(spans.base_offsets.numel),
        ctypes.c_int64(capacity),
        ctypes.c_int64(parsed_scan),
        ctypes.c_int64(num_q_heads),
        ctypes.c_int64(num_kv_heads),
        ctypes.c_int64(head_dim),
        ctypes.c_float(scale),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def laguna_global_attention_decode_wmma_gqa6_k64_three_term_raw_numerator_bf16_spans(
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    gate_ptr: int,
    gated_out_ptr: int,
    score_scratch_ptr: int,
    physical_scratch_ptr: int,
    spans: KVLiveSpans,
    scan_slots: int,
    max_context_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch three-term BF16 WMMA global GQA6/K64 attention."""

    capacity = _check_global_spans(spans, num_kv_heads, head_dim)
    parsed_scan = _check_split_scan_slots(scan_slots, capacity)
    if (
        capacity != 4096
        or parsed_scan > 1024
        or int(max_context_len) != 4096
        or int(num_q_heads) != 48
        or int(num_kv_heads) != 8
        or int(head_dim) != 128
    ):
        raise ValueError(
            "global three-term WMMA GQA6/K64 requires capacity 4096, scan "
            "slots at most 1024, 48 query heads, 8 KV heads, and D128"
        )
    library = library or build_laguna_kv_attention(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(
        library,
        _SYMBOL_GLOBAL_ATTENTION_WMMA_GQA6_K64_THREE_TERM_RAW_NUMERATOR,
    )
    fn.argtypes = (
        [ctypes.c_void_p] * 13
        + [ctypes.c_int64] * 8
        + [ctypes.c_float, ctypes.c_void_p]
    )
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(query_ptr),
        ctypes.c_void_p(key_cache_ptr),
        ctypes.c_void_p(value_cache_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_void_p(gate_ptr),
        ctypes.c_void_p(gated_out_ptr),
        ctypes.c_void_p(score_scratch_ptr),
        ctypes.c_void_p(physical_scratch_ptr),
        ctypes.c_void_p(spans.base_offsets.ptr),
        ctypes.c_void_p(spans.live_counts.ptr),
        ctypes.c_void_p(spans.token_positions.ptr),
        ctypes.c_void_p(spans.evict_mask.ptr),
        ctypes.c_void_p(spans.row_positions.ptr),
        ctypes.c_int64(capacity),
        ctypes.c_int64(_GLOBAL_BLOCK_SIZE),
        ctypes.c_int64(spans.base_offsets.numel),
        ctypes.c_int64(capacity),
        ctypes.c_int64(parsed_scan),
        ctypes.c_int64(num_q_heads),
        ctypes.c_int64(num_kv_heads),
        ctypes.c_int64(head_dim),
        ctypes.c_float(scale),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def laguna_global_attention_prefill_qrow6_dense_initial_online_bf16_spans(
    query_ptr: int,
    current_key_ptr: int,
    current_value_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    spans: KVLiveSpans,
    rows: int,
    max_context_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    start_position: int | None = None,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Run qrow6 global attention over an identity-positioned initial fill."""

    _laguna_global_attention_prefill_dense_initial_online_bf16_spans(
        query_ptr,
        current_key_ptr,
        current_value_ptr,
        key_cache_ptr,
        value_cache_ptr,
        out_ptr,
        spans,
        rows,
        max_context_len,
        num_q_heads,
        num_kv_heads,
        head_dim,
        scale,
        symbol=_SYMBOL_GLOBAL_PREFILL_QROW6_DENSE_INITIAL_ONLINE,
        start_position=start_position,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def laguna_global_attention_prefill_qrow4_m128_online_bf16_spans(
    query_ptr: int,
    current_key_ptr: int,
    current_value_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    spans: KVLiveSpans,
    rows: int,
    max_context_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Use qrow4 for production M128 tiles and qrow2 for residual tiles."""

    kernel = (
        laguna_global_attention_prefill_qrow4_online_bf16_spans
        if int(rows) == 128
        else laguna_global_attention_prefill_qrow2_online_bf16_spans
    )
    kernel(
        query_ptr,
        current_key_ptr,
        current_value_ptr,
        key_cache_ptr,
        value_cache_ptr,
        out_ptr,
        spans,
        rows,
        max_context_len,
        num_q_heads,
        num_kv_heads,
        head_dim,
        scale,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def laguna_swa_write_kv_f32_spans(
    key_ptr: int,
    value_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    spans: KVLiveSpans,
    num_kv_heads: int,
    head_dim: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Append one F32 K/V row into a token-granular BF16 SWA ring."""

    capacity = _check_swa_spans(spans, num_kv_heads, head_dim)
    library = library or build_laguna_kv_attention(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_SWA_WRITE)
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
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(key_ptr),
        ctypes.c_void_p(value_ptr),
        ctypes.c_void_p(key_cache_ptr),
        ctypes.c_void_p(value_cache_ptr),
        ctypes.c_void_p(spans.base_offsets.ptr),
        ctypes.c_void_p(spans.live_counts.ptr),
        ctypes.c_void_p(spans.token_positions.ptr),
        ctypes.c_void_p(spans.evict_mask.ptr),
        ctypes.c_void_p(spans.row_positions.ptr),
        ctypes.c_int64(capacity),
        ctypes.c_int64(num_kv_heads),
        ctypes.c_int64(head_dim),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def laguna_swa_write_kv_rows_f32_spans(
    key_ptr: int,
    value_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    spans: KVLiveSpans,
    rows: int,
    num_kv_heads: int,
    head_dim: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Append consecutive F32 K/V rows into a token-granular BF16 ring."""

    capacity = _check_swa_spans(spans, num_kv_heads, head_dim)
    _check_prefill_rows(spans, rows, capacity)
    library = library or build_laguna_kv_attention(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_SWA_WRITE_ROWS)
    fn.argtypes = [ctypes.c_void_p] * 9 + [ctypes.c_int64] * 4 + [ctypes.c_void_p]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(key_ptr),
        ctypes.c_void_p(value_ptr),
        ctypes.c_void_p(key_cache_ptr),
        ctypes.c_void_p(value_cache_ptr),
        ctypes.c_void_p(spans.base_offsets.ptr),
        ctypes.c_void_p(spans.live_counts.ptr),
        ctypes.c_void_p(spans.token_positions.ptr),
        ctypes.c_void_p(spans.evict_mask.ptr),
        ctypes.c_void_p(spans.row_positions.ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(capacity),
        ctypes.c_int64(num_kv_heads),
        ctypes.c_int64(head_dim),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def laguna_swa_attention_decode_bf16_spans(
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    spans: KVLiveSpans,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    sliding_window: int | None = None,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Decode absolute-position GQA over live, non-evicted SWA ring slots."""

    capacity = _check_swa_spans(spans, num_kv_heads, head_dim)
    _check_laguna_attention_shape(num_q_heads, num_kv_heads, head_dim)
    window = capacity if sliding_window is None else int(sliding_window)
    if window <= 0 or window > capacity:
        raise ValueError("sliding_window must be in [1, ring capacity]")
    library = library or build_laguna_kv_attention(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_SWA_ATTENTION)
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
        ctypes.c_float,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(query_ptr),
        ctypes.c_void_p(key_cache_ptr),
        ctypes.c_void_p(value_cache_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_void_p(spans.base_offsets.ptr),
        ctypes.c_void_p(spans.live_counts.ptr),
        ctypes.c_void_p(spans.token_positions.ptr),
        ctypes.c_void_p(spans.evict_mask.ptr),
        ctypes.c_void_p(spans.row_positions.ptr),
        ctypes.c_int64(capacity),
        ctypes.c_int64(window),
        ctypes.c_int64(num_q_heads),
        ctypes.c_int64(num_kv_heads),
        ctypes.c_int64(head_dim),
        ctypes.c_float(scale),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def laguna_swa_attention_decode_token4_exact_bf16_spans(
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    spans: KVLiveSpans,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    sliding_window: int | None = None,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Run exact token4 score-parallel decode over SWA ``KVLiveSpans``."""

    capacity = _check_swa_spans(spans, num_kv_heads, head_dim)
    _check_laguna_attention_shape(num_q_heads, num_kv_heads, head_dim)
    window = capacity if sliding_window is None else int(sliding_window)
    if window <= 0 or window > capacity:
        raise ValueError("sliding_window must be in [1, ring capacity]")
    library = library or build_laguna_kv_attention(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_SWA_ATTENTION_TOKEN4_EXACT)
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
        ctypes.c_float,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(query_ptr),
        ctypes.c_void_p(key_cache_ptr),
        ctypes.c_void_p(value_cache_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_void_p(spans.base_offsets.ptr),
        ctypes.c_void_p(spans.live_counts.ptr),
        ctypes.c_void_p(spans.token_positions.ptr),
        ctypes.c_void_p(spans.evict_mask.ptr),
        ctypes.c_void_p(spans.row_positions.ptr),
        ctypes.c_int64(capacity),
        ctypes.c_int64(window),
        ctypes.c_int64(num_q_heads),
        ctypes.c_int64(num_kv_heads),
        ctypes.c_int64(head_dim),
        ctypes.c_float(scale),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def laguna_swa_attention_decode_split_exact_bf16_spans(
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    score_scratch_ptr: int,
    physical_scratch_ptr: int,
    spans: KVLiveSpans,
    scan_slots: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    sliding_window: int | None = None,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Run exact split-score SWA with caller-owned scratch."""

    capacity = _check_swa_spans(spans, num_kv_heads, head_dim)
    parsed_scan = _check_split_scan_slots(scan_slots, capacity)
    _check_laguna_attention_shape(num_q_heads, num_kv_heads, head_dim)
    window = capacity if sliding_window is None else int(sliding_window)
    if window <= 0 or window > capacity:
        raise ValueError("sliding_window must be in [1, ring capacity]")
    library = library or build_laguna_kv_attention(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_SWA_ATTENTION_SPLIT_EXACT)
    fn.argtypes = (
        [ctypes.c_void_p] * 11
        + [ctypes.c_int64] * 7
        + [ctypes.c_float, ctypes.c_void_p]
    )
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(query_ptr),
        ctypes.c_void_p(key_cache_ptr),
        ctypes.c_void_p(value_cache_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_void_p(score_scratch_ptr),
        ctypes.c_void_p(physical_scratch_ptr),
        ctypes.c_void_p(spans.base_offsets.ptr),
        ctypes.c_void_p(spans.live_counts.ptr),
        ctypes.c_void_p(spans.token_positions.ptr),
        ctypes.c_void_p(spans.evict_mask.ptr),
        ctypes.c_void_p(spans.row_positions.ptr),
        ctypes.c_int64(capacity),
        ctypes.c_int64(window),
        ctypes.c_int64(capacity),
        ctypes.c_int64(parsed_scan),
        ctypes.c_int64(num_q_heads),
        ctypes.c_int64(num_kv_heads),
        ctypes.c_int64(head_dim),
        ctypes.c_float(scale),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def laguna_swa_attention_decode_split_exact_gated_bf16_spans(
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    gate_ptr: int,
    gated_out_ptr: int,
    score_scratch_ptr: int,
    physical_scratch_ptr: int,
    spans: KVLiveSpans,
    scan_slots: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    sliding_window: int | None = None,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Run exact split SWA and fuse its BF16 softplus gate."""

    _laguna_swa_attention_decode_split_exact_gated_bf16_spans(
        _SYMBOL_SWA_ATTENTION_SPLIT_EXACT_GATED,
        query_ptr,
        key_cache_ptr,
        value_cache_ptr,
        out_ptr,
        gate_ptr,
        gated_out_ptr,
        score_scratch_ptr,
        physical_scratch_ptr,
        spans,
        scan_slots,
        num_q_heads,
        num_kv_heads,
        head_dim,
        scale,
        sliding_window=sliding_window,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def laguna_swa_attention_decode_split_tile16_exact_bf16_spans(
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    score_scratch_ptr: int,
    physical_scratch_ptr: int,
    spans: KVLiveSpans,
    scan_slots: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    sliding_window: int | None = None,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Run exact tile16 split-score SWA with caller-owned scratch."""

    capacity = _check_swa_spans(spans, num_kv_heads, head_dim)
    parsed_scan = _check_split_scan_slots(scan_slots, capacity)
    _check_laguna_attention_shape(num_q_heads, num_kv_heads, head_dim)
    window = capacity if sliding_window is None else int(sliding_window)
    if window <= 0 or window > capacity:
        raise ValueError("sliding_window must be in [1, ring capacity]")
    library = library or build_laguna_kv_attention(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_SWA_ATTENTION_SPLIT_TILE16_EXACT)
    fn.argtypes = (
        [ctypes.c_void_p] * 11
        + [ctypes.c_int64] * 7
        + [ctypes.c_float, ctypes.c_void_p]
    )
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(query_ptr),
        ctypes.c_void_p(key_cache_ptr),
        ctypes.c_void_p(value_cache_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_void_p(score_scratch_ptr),
        ctypes.c_void_p(physical_scratch_ptr),
        ctypes.c_void_p(spans.base_offsets.ptr),
        ctypes.c_void_p(spans.live_counts.ptr),
        ctypes.c_void_p(spans.token_positions.ptr),
        ctypes.c_void_p(spans.evict_mask.ptr),
        ctypes.c_void_p(spans.row_positions.ptr),
        ctypes.c_int64(capacity),
        ctypes.c_int64(window),
        ctypes.c_int64(capacity),
        ctypes.c_int64(parsed_scan),
        ctypes.c_int64(num_q_heads),
        ctypes.c_int64(num_kv_heads),
        ctypes.c_int64(head_dim),
        ctypes.c_float(scale),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def laguna_swa_attention_decode_split_tile16_exact_gated_bf16_spans(
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    gate_ptr: int,
    gated_out_ptr: int,
    score_scratch_ptr: int,
    physical_scratch_ptr: int,
    spans: KVLiveSpans,
    scan_slots: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    sliding_window: int | None = None,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Run exact tile16 split SWA and fuse its BF16 softplus gate."""

    _laguna_swa_attention_decode_split_exact_gated_bf16_spans(
        _SYMBOL_SWA_ATTENTION_SPLIT_TILE16_EXACT_GATED,
        query_ptr,
        key_cache_ptr,
        value_cache_ptr,
        out_ptr,
        gate_ptr,
        gated_out_ptr,
        score_scratch_ptr,
        physical_scratch_ptr,
        spans,
        scan_slots,
        num_q_heads,
        num_kv_heads,
        head_dim,
        scale,
        sliding_window=sliding_window,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def laguna_swa_attention_decode_split_exact_gated_wave_local_bf16_spans(
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    gate_ptr: int,
    gated_out_ptr: int,
    score_scratch_ptr: int,
    physical_scratch_ptr: int,
    spans: KVLiveSpans,
    scan_slots: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    sliding_window: int | None = None,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Run exact split SWA with wave-private softmax statistics."""

    _laguna_swa_attention_decode_split_exact_gated_bf16_spans(
        _SYMBOL_SWA_ATTENTION_SPLIT_EXACT_GATED_WAVE_LOCAL,
        query_ptr,
        key_cache_ptr,
        value_cache_ptr,
        out_ptr,
        gate_ptr,
        gated_out_ptr,
        score_scratch_ptr,
        physical_scratch_ptr,
        spans,
        scan_slots,
        num_q_heads,
        num_kv_heads,
        head_dim,
        scale,
        sliding_window=sliding_window,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def laguna_swa_attention_decode_split_exact_gated_gqa3_scores_bf16_spans(
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    gate_ptr: int,
    gated_out_ptr: int,
    score_scratch_ptr: int,
    physical_scratch_ptr: int,
    spans: KVLiveSpans,
    scan_slots: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    sliding_window: int | None = None,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Run exact split SWA with one value owner per three-query subgroup."""

    _laguna_swa_attention_decode_split_exact_gated_bf16_spans(
        _SYMBOL_SWA_ATTENTION_SPLIT_EXACT_GATED_GQA3_SCORES,
        query_ptr,
        key_cache_ptr,
        value_cache_ptr,
        out_ptr,
        gate_ptr,
        gated_out_ptr,
        score_scratch_ptr,
        physical_scratch_ptr,
        spans,
        scan_slots,
        num_q_heads,
        num_kv_heads,
        head_dim,
        scale,
        sliding_window=sliding_window,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def laguna_swa_attention_decode_split_exact_gated_wave_local_dim2_bf16_spans(
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    gate_ptr: int,
    gated_out_ptr: int,
    score_scratch_ptr: int,
    physical_scratch_ptr: int,
    spans: KVLiveSpans,
    scan_slots: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    sliding_window: int | None = None,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Run exact split SWA with two adjacent dimensions per thread."""

    _laguna_swa_attention_decode_split_exact_gated_bf16_spans(
        _SYMBOL_SWA_ATTENTION_SPLIT_EXACT_GATED_WAVE_LOCAL_DIM2,
        query_ptr,
        key_cache_ptr,
        value_cache_ptr,
        out_ptr,
        gate_ptr,
        gated_out_ptr,
        score_scratch_ptr,
        physical_scratch_ptr,
        spans,
        scan_slots,
        num_q_heads,
        num_kv_heads,
        head_dim,
        scale,
        sliding_window=sliding_window,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def laguna_swa_attention_decode_split_tile16_exact_gated_wave_local_bf16_spans(
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    gate_ptr: int,
    gated_out_ptr: int,
    score_scratch_ptr: int,
    physical_scratch_ptr: int,
    spans: KVLiveSpans,
    scan_slots: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    sliding_window: int | None = None,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Run exact tile16 split SWA with wave-private softmax statistics."""

    _laguna_swa_attention_decode_split_exact_gated_bf16_spans(
        _SYMBOL_SWA_ATTENTION_SPLIT_TILE16_EXACT_GATED_WAVE_LOCAL,
        query_ptr,
        key_cache_ptr,
        value_cache_ptr,
        out_ptr,
        gate_ptr,
        gated_out_ptr,
        score_scratch_ptr,
        physical_scratch_ptr,
        spans,
        scan_slots,
        num_q_heads,
        num_kv_heads,
        head_dim,
        scale,
        sliding_window=sliding_window,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def laguna_swa_attention_decode_split_tile16_exact_gated_gqa3_scores_bf16_spans(
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    gate_ptr: int,
    gated_out_ptr: int,
    score_scratch_ptr: int,
    physical_scratch_ptr: int,
    spans: KVLiveSpans,
    scan_slots: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    sliding_window: int | None = None,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Run exact tile16 split SWA with one value owner per three-query subgroup."""

    _laguna_swa_attention_decode_split_exact_gated_bf16_spans(
        _SYMBOL_SWA_ATTENTION_SPLIT_TILE16_EXACT_GATED_GQA3_SCORES,
        query_ptr,
        key_cache_ptr,
        value_cache_ptr,
        out_ptr,
        gate_ptr,
        gated_out_ptr,
        score_scratch_ptr,
        physical_scratch_ptr,
        spans,
        scan_slots,
        num_q_heads,
        num_kv_heads,
        head_dim,
        scale,
        sliding_window=sliding_window,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def laguna_swa_attention_decode_split_tile16_exact_gated_gqa3_scores_fixed512_bf16_spans(
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    gate_ptr: int,
    gated_out_ptr: int,
    score_scratch_ptr: int,
    physical_scratch_ptr: int,
    spans: KVLiveSpans,
    scan_slots: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    sliding_window: int | None = None,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Run exact tile16 GQA3 split SWA with the saturated-512 reducer."""

    capacity = _check_swa_spans(spans, num_kv_heads, head_dim)
    window = capacity if sliding_window is None else int(sliding_window)
    if (
        capacity != 512
        or window != 512
        or int(scan_slots) != 512
        or int(num_q_heads) != 72
        or int(num_kv_heads) != 8
        or int(head_dim) != 128
    ):
        raise ValueError(
            "fixed512 SWA reduction requires capacity/window/scan 512, "
            "72 query heads, 8 KV heads, and D128"
        )
    _laguna_swa_attention_decode_split_exact_gated_bf16_spans(
        _SYMBOL_SWA_ATTENTION_SPLIT_TILE16_EXACT_GATED_GQA3_SCORES_FIXED512,
        query_ptr,
        key_cache_ptr,
        value_cache_ptr,
        out_ptr,
        gate_ptr,
        gated_out_ptr,
        score_scratch_ptr,
        physical_scratch_ptr,
        spans,
        scan_slots,
        num_q_heads,
        num_kv_heads,
        head_dim,
        scale,
        sliding_window=window,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def laguna_swa_attention_decode_fused_exact_gated_gqa2_fixed512_bf16_spans(
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    gate_ptr: int,
    gated_out_ptr: int,
    score_scratch_ptr: int,
    physical_scratch_ptr: int,
    spans: KVLiveSpans,
    scan_slots: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    sliding_window: int | None = None,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch fused exact GQA2 QK/softmax/PV for saturated fixed512 SWA."""

    capacity = _check_swa_spans(spans, num_kv_heads, head_dim)
    window = capacity if sliding_window is None else int(sliding_window)
    if (
        capacity != 512
        or window != 512
        or int(scan_slots) != 512
        or int(num_q_heads) != 72
        or int(num_kv_heads) != 8
        or int(head_dim) != 128
    ):
        raise ValueError(
            "fused GQA2 fixed512 SWA requires capacity/window/scan 512, "
            "72 query heads, 8 KV heads, and D128"
        )
    _laguna_swa_attention_decode_split_exact_gated_bf16_spans(
        _SYMBOL_SWA_ATTENTION_FUSED_EXACT_GATED_GQA2_FIXED512,
        query_ptr,
        key_cache_ptr,
        value_cache_ptr,
        out_ptr,
        gate_ptr,
        gated_out_ptr,
        score_scratch_ptr,
        physical_scratch_ptr,
        spans,
        scan_slots,
        num_q_heads,
        num_kv_heads,
        head_dim,
        scale,
        sliding_window=window,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def laguna_swa_attention_decode_fused_exact_gated_gqa3_local384_fixed512_bf16_spans(
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    gate_ptr: int,
    gated_out_ptr: int,
    score_scratch_ptr: int,
    physical_scratch_ptr: int,
    spans: KVLiveSpans,
    scan_slots: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    sliding_window: int | None = None,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch exact fused-GQA3 local384 fixed512 SWA."""

    capacity = _check_swa_spans(spans, num_kv_heads, head_dim)
    window = capacity if sliding_window is None else int(sliding_window)
    if (
        capacity != 512
        or window != 512
        or int(scan_slots) != 512
        or int(num_q_heads) != 72
        or int(num_kv_heads) != 8
        or int(head_dim) != 128
    ):
        raise ValueError(
            "fused GQA3 local384 fixed512 SWA requires "
            "capacity/window/scan 512, 72 query heads, 8 KV heads, and D128"
        )
    _laguna_swa_attention_decode_split_exact_gated_bf16_spans(
        _SYMBOL_SWA_ATTENTION_FUSED_EXACT_GATED_GQA3_LOCAL384_FIXED512,
        query_ptr,
        key_cache_ptr,
        value_cache_ptr,
        out_ptr,
        gate_ptr,
        gated_out_ptr,
        score_scratch_ptr,
        physical_scratch_ptr,
        spans,
        scan_slots,
        num_q_heads,
        num_kv_heads,
        head_dim,
        scale,
        sliding_window=window,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def laguna_swa_attention_decode_fused_exact_gated_gqa3_vstage64_fixed512_bf16_spans(
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    gate_ptr: int,
    gated_out_ptr: int,
    score_scratch_ptr: int,
    physical_scratch_ptr: int,
    spans: KVLiveSpans,
    scan_slots: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    sliding_window: int | None = None,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch exact fused-GQA3 with 64-slot staged V reuse."""

    capacity = _check_swa_spans(spans, num_kv_heads, head_dim)
    window = capacity if sliding_window is None else int(sliding_window)
    if (
        capacity != 512
        or window != 512
        or int(scan_slots) != 512
        or int(num_q_heads) != 72
        or int(num_kv_heads) != 8
        or int(head_dim) != 128
    ):
        raise ValueError(
            "fused GQA3 V-stage64 fixed512 SWA requires "
            "capacity/window/scan 512, 72 query heads, 8 KV heads, and D128"
        )
    _laguna_swa_attention_decode_split_exact_gated_bf16_spans(
        _SYMBOL_SWA_ATTENTION_FUSED_EXACT_GATED_GQA3_VSTAGE64_FIXED512,
        query_ptr,
        key_cache_ptr,
        value_cache_ptr,
        out_ptr,
        gate_ptr,
        gated_out_ptr,
        score_scratch_ptr,
        physical_scratch_ptr,
        spans,
        scan_slots,
        num_q_heads,
        num_kv_heads,
        head_dim,
        scale,
        sliding_window=window,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def laguna_swa_attention_decode_fused_exact_gated_gqa3_vstage64_vec16_fixed512_bf16_spans(
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    gate_ptr: int,
    gated_out_ptr: int,
    score_scratch_ptr: int,
    physical_scratch_ptr: int,
    spans: KVLiveSpans,
    scan_slots: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    sliding_window: int | None = None,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch exact fused-GQA3 with 16-byte staged-V copies."""

    capacity = _check_swa_spans(spans, num_kv_heads, head_dim)
    window = capacity if sliding_window is None else int(sliding_window)
    if (
        capacity != 512
        or window != 512
        or int(scan_slots) != 512
        or int(num_q_heads) != 72
        or int(num_kv_heads) != 8
        or int(head_dim) != 128
    ):
        raise ValueError(
            "fused GQA3 V-stage64 vec16 fixed512 SWA requires "
            "capacity/window/scan 512, 72 query heads, 8 KV heads, and D128"
        )
    _laguna_swa_attention_decode_split_exact_gated_bf16_spans(
        _SYMBOL_SWA_ATTENTION_FUSED_EXACT_GATED_GQA3_VSTAGE64_VEC16_FIXED512,
        query_ptr,
        key_cache_ptr,
        value_cache_ptr,
        out_ptr,
        gate_ptr,
        gated_out_ptr,
        score_scratch_ptr,
        physical_scratch_ptr,
        spans,
        scan_slots,
        num_q_heads,
        num_kv_heads,
        head_dim,
        scale,
        sliding_window=window,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def laguna_swa_attention_decode_fused_exact_gated_gqa3_vstage64_vec16_direct_fixed512_bf16_spans(
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    gate_ptr: int,
    gated_out_ptr: int,
    score_scratch_ptr: int,
    physical_scratch_ptr: int,
    spans: KVLiveSpans,
    scan_slots: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    sliding_window: int | None = None,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch exact fused-GQA3 with direct vec16 V-to-LDS stores."""

    capacity = _check_swa_spans(spans, num_kv_heads, head_dim)
    window = capacity if sliding_window is None else int(sliding_window)
    if (
        capacity != 512
        or window != 512
        or int(scan_slots) != 512
        or int(num_q_heads) != 72
        or int(num_kv_heads) != 8
        or int(head_dim) != 128
    ):
        raise ValueError(
            "fused GQA3 V-stage64 vec16-direct fixed512 SWA requires "
            "capacity/window/scan 512, 72 query heads, 8 KV heads, and D128"
        )
    _laguna_swa_attention_decode_split_exact_gated_bf16_spans(
        _SYMBOL_SWA_ATTENTION_FUSED_EXACT_GATED_GQA3_VSTAGE64_VEC16_DIRECT_FIXED512,
        query_ptr,
        key_cache_ptr,
        value_cache_ptr,
        out_ptr,
        gate_ptr,
        gated_out_ptr,
        score_scratch_ptr,
        physical_scratch_ptr,
        spans,
        scan_slots,
        num_q_heads,
        num_kv_heads,
        head_dim,
        scale,
        sliding_window=window,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def laguna_swa_attention_decode_fused_exact_gated_gqa3_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans(
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    gate_ptr: int,
    gated_out_ptr: int,
    score_scratch_ptr: int,
    physical_scratch_ptr: int,
    spans: KVLiveSpans,
    scan_slots: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    sliding_window: int | None = None,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch exact direct GQA3 with an asserted softmax input domain."""

    capacity = _check_swa_spans(spans, num_kv_heads, head_dim)
    window = capacity if sliding_window is None else int(sliding_window)
    if (
        capacity != 512
        or window != 512
        or int(scan_slots) != 512
        or int(num_q_heads) != 72
        or int(num_kv_heads) != 8
        or int(head_dim) != 128
    ):
        raise ValueError(
            "fused GQA3 V-stage64 vec16-direct assume-exp fixed512 SWA "
            "requires capacity/window/scan 512, 72 query heads, "
            "8 KV heads, and D128"
        )
    _laguna_swa_attention_decode_split_exact_gated_bf16_spans(
        _SYMBOL_SWA_ATTENTION_FUSED_EXACT_GATED_GQA3_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512,
        query_ptr,
        key_cache_ptr,
        value_cache_ptr,
        out_ptr,
        gate_ptr,
        gated_out_ptr,
        score_scratch_ptr,
        physical_scratch_ptr,
        spans,
        scan_slots,
        num_q_heads,
        num_kv_heads,
        head_dim,
        scale,
        sliding_window=window,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def laguna_swa_attention_decode_fused_exact_gated_mixed32_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans(
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    gate_ptr: int,
    gated_out_ptr: int,
    score_scratch_ptr: int,
    physical_scratch_ptr: int,
    spans: KVLiveSpans,
    scan_slots: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    sliding_window: int | None = None,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch exact 32-block mixed GQA2/GQA3 fixed512 SWA."""

    capacity = _check_swa_spans(spans, num_kv_heads, head_dim)
    window = capacity if sliding_window is None else int(sliding_window)
    if (
        capacity != 512
        or window != 512
        or int(scan_slots) != 512
        or int(num_q_heads) != 72
        or int(num_kv_heads) != 8
        or int(head_dim) != 128
    ):
        raise ValueError(
            "mixed32 V-stage64 vec16-direct assume-exp fixed512 SWA "
            "requires capacity/window/scan 512, 72 query heads, "
            "8 KV heads, and D128"
        )
    _laguna_swa_attention_decode_split_exact_gated_bf16_spans(
        _SYMBOL_SWA_ATTENTION_FUSED_EXACT_GATED_MIXED32_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512,
        query_ptr,
        key_cache_ptr,
        value_cache_ptr,
        out_ptr,
        gate_ptr,
        gated_out_ptr,
        score_scratch_ptr,
        physical_scratch_ptr,
        spans,
        scan_slots,
        num_q_heads,
        num_kv_heads,
        head_dim,
        scale,
        sliding_window=window,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def laguna_swa_attention_decode_fused_exact_gated_mixed32_exp4_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans(
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    gate_ptr: int,
    gated_out_ptr: int,
    score_scratch_ptr: int,
    physical_scratch_ptr: int,
    spans: KVLiveSpans,
    scan_slots: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    sliding_window: int | None = None,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch exact mixed32 SWA with four-lane exponential batches."""

    capacity = _check_swa_spans(spans, num_kv_heads, head_dim)
    window = capacity if sliding_window is None else int(sliding_window)
    if (
        capacity != 512
        or window != 512
        or int(scan_slots) != 512
        or int(num_q_heads) != 72
        or int(num_kv_heads) != 8
        or int(head_dim) != 128
    ):
        raise ValueError(
            "mixed32 exp4 V-stage64 vec16-direct assume-exp fixed512 SWA "
            "requires capacity/window/scan 512, 72 query heads, "
            "8 KV heads, and D128"
        )
    _laguna_swa_attention_decode_split_exact_gated_bf16_spans(
        _SYMBOL_SWA_ATTENTION_FUSED_EXACT_GATED_MIXED32_EXP4_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512,
        query_ptr,
        key_cache_ptr,
        value_cache_ptr,
        out_ptr,
        gate_ptr,
        gated_out_ptr,
        score_scratch_ptr,
        physical_scratch_ptr,
        spans,
        scan_slots,
        num_q_heads,
        num_kv_heads,
        head_dim,
        scale,
        sliding_window=window,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def laguna_swa_attention_decode_fused_exact_gated_mixed32_exp8_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans(
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    gate_ptr: int,
    gated_out_ptr: int,
    score_scratch_ptr: int,
    physical_scratch_ptr: int,
    spans: KVLiveSpans,
    scan_slots: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    sliding_window: int | None = None,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch exact mixed32 SWA with eight-lane exponential batches."""

    capacity = _check_swa_spans(spans, num_kv_heads, head_dim)
    window = capacity if sliding_window is None else int(sliding_window)
    if (
        capacity != 512
        or window != 512
        or int(scan_slots) != 512
        or int(num_q_heads) != 72
        or int(num_kv_heads) != 8
        or int(head_dim) != 128
    ):
        raise ValueError(
            "mixed32 exp8 V-stage64 vec16-direct assume-exp fixed512 SWA "
            "requires capacity/window/scan 512, 72 query heads, "
            "8 KV heads, and D128"
        )
    _laguna_swa_attention_decode_split_exact_gated_bf16_spans(
        _SYMBOL_SWA_ATTENTION_FUSED_EXACT_GATED_MIXED32_EXP8_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512,
        query_ptr,
        key_cache_ptr,
        value_cache_ptr,
        out_ptr,
        gate_ptr,
        gated_out_ptr,
        score_scratch_ptr,
        physical_scratch_ptr,
        spans,
        scan_slots,
        num_q_heads,
        num_kv_heads,
        head_dim,
        scale,
        sliding_window=window,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def laguna_swa_attention_decode_fused_exact_gated_mixed32_exp16_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans(
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    gate_ptr: int,
    gated_out_ptr: int,
    score_scratch_ptr: int,
    physical_scratch_ptr: int,
    spans: KVLiveSpans,
    scan_slots: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    sliding_window: int | None = None,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch exact mixed32 SWA with sixteen-lane exponential batches."""

    capacity = _check_swa_spans(spans, num_kv_heads, head_dim)
    window = capacity if sliding_window is None else int(sliding_window)
    if (
        capacity != 512
        or window != 512
        or int(scan_slots) != 512
        or int(num_q_heads) != 72
        or int(num_kv_heads) != 8
        or int(head_dim) != 128
    ):
        raise ValueError(
            "mixed32 exp16 V-stage64 vec16-direct assume-exp fixed512 SWA "
            "requires capacity/window/scan 512, 72 query heads, "
            "8 KV heads, and D128"
        )
    _laguna_swa_attention_decode_split_exact_gated_bf16_spans(
        _SYMBOL_SWA_ATTENTION_FUSED_EXACT_GATED_MIXED32_EXP16_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512,
        query_ptr,
        key_cache_ptr,
        value_cache_ptr,
        out_ptr,
        gate_ptr,
        gated_out_ptr,
        score_scratch_ptr,
        physical_scratch_ptr,
        spans,
        scan_slots,
        num_q_heads,
        num_kv_heads,
        head_dim,
        scale,
        sliding_window=window,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def laguna_swa_attention_decode_fused_exact_gated_mixed32_exp32_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans(
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    gate_ptr: int,
    gated_out_ptr: int,
    score_scratch_ptr: int,
    physical_scratch_ptr: int,
    spans: KVLiveSpans,
    scan_slots: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    sliding_window: int | None = None,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch exact mixed32 SWA with wave-wide exponential batches."""

    capacity = _check_swa_spans(spans, num_kv_heads, head_dim)
    window = capacity if sliding_window is None else int(sliding_window)
    if (
        capacity != 512
        or window != 512
        or int(scan_slots) != 512
        or int(num_q_heads) != 72
        or int(num_kv_heads) != 8
        or int(head_dim) != 128
    ):
        raise ValueError(
            "mixed32 exp32 V-stage64 vec16-direct assume-exp fixed512 SWA "
            "requires capacity/window/scan 512, 72 query heads, "
            "8 KV heads, and D128"
        )
    _laguna_swa_attention_decode_split_exact_gated_bf16_spans(
        _SYMBOL_SWA_ATTENTION_FUSED_EXACT_GATED_MIXED32_EXP32_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512,
        query_ptr,
        key_cache_ptr,
        value_cache_ptr,
        out_ptr,
        gate_ptr,
        gated_out_ptr,
        score_scratch_ptr,
        physical_scratch_ptr,
        spans,
        scan_slots,
        num_q_heads,
        num_kv_heads,
        head_dim,
        scale,
        sliding_window=window,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def laguna_swa_attention_decode_fused_exact_gated_mixed32_exp32_producer_max_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans(
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    gate_ptr: int,
    gated_out_ptr: int,
    score_scratch_ptr: int,
    physical_scratch_ptr: int,
    spans: KVLiveSpans,
    scan_slots: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    sliding_window: int | None = None,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch exact mixed32 SWA with score-producer partial maxima."""

    capacity = _check_swa_spans(spans, num_kv_heads, head_dim)
    window = capacity if sliding_window is None else int(sliding_window)
    if (
        capacity != 512
        or window != 512
        or int(scan_slots) != 512
        or int(num_q_heads) != 72
        or int(num_kv_heads) != 8
        or int(head_dim) != 128
    ):
        raise ValueError(
            "mixed32 exp32 producer-max V-stage64 vec16-direct assume-exp "
            "fixed512 SWA requires capacity/window/scan 512, 72 query heads, "
            "8 KV heads, and D128"
        )
    _laguna_swa_attention_decode_split_exact_gated_bf16_spans(
        _SYMBOL_SWA_ATTENTION_FUSED_EXACT_GATED_MIXED32_EXP32_PRODUCER_MAX_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512,
        query_ptr,
        key_cache_ptr,
        value_cache_ptr,
        out_ptr,
        gate_ptr,
        gated_out_ptr,
        score_scratch_ptr,
        physical_scratch_ptr,
        spans,
        scan_slots,
        num_q_heads,
        num_kv_heads,
        head_dim,
        scale,
        sliding_window=window,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def laguna_swa_attention_decode_fused_exact_gated_mixed32_exp32_producer_max_gate_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans(
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    gate_ptr: int,
    gated_out_ptr: int,
    score_scratch_ptr: int,
    physical_scratch_ptr: int,
    spans: KVLiveSpans,
    scan_slots: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    sliding_window: int | None = None,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch exact SWA with producer-owned maxima and softplus gates."""

    capacity = _check_swa_spans(spans, num_kv_heads, head_dim)
    window = capacity if sliding_window is None else int(sliding_window)
    if (
        capacity != 512
        or window != 512
        or int(scan_slots) != 512
        or int(num_q_heads) != 72
        or int(num_kv_heads) != 8
        or int(head_dim) != 128
    ):
        raise ValueError(
            "mixed32 exp32 producer-max-gate V-stage64 vec16-direct "
            "assume-exp fixed512 SWA requires capacity/window/scan 512, "
            "72 query heads, 8 KV heads, and D128"
        )
    _laguna_swa_attention_decode_split_exact_gated_bf16_spans(
        _SYMBOL_SWA_ATTENTION_FUSED_EXACT_GATED_MIXED32_EXP32_PRODUCER_MAX_GATE_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512,
        query_ptr,
        key_cache_ptr,
        value_cache_ptr,
        out_ptr,
        gate_ptr,
        gated_out_ptr,
        score_scratch_ptr,
        physical_scratch_ptr,
        spans,
        scan_slots,
        num_q_heads,
        num_kv_heads,
        head_dim,
        scale,
        sliding_window=window,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def laguna_swa_attention_decode_fused_exact_gated_mixed32_exp32_producer_max_gate_dpp_qk_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans(
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    gate_ptr: int,
    gated_out_ptr: int,
    score_scratch_ptr: int,
    physical_scratch_ptr: int,
    spans: KVLiveSpans,
    scan_slots: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    sliding_window: int | None = None,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch exact producer-max/gate SWA with DPP QK lane transport."""

    capacity = _check_swa_spans(spans, num_kv_heads, head_dim)
    window = capacity if sliding_window is None else int(sliding_window)
    if (
        capacity != 512
        or window != 512
        or int(scan_slots) != 512
        or int(num_q_heads) != 72
        or int(num_kv_heads) != 8
        or int(head_dim) != 128
    ):
        raise ValueError(
            "mixed32 exp32 producer-max-gate DPP-QK V-stage64 "
            "vec16-direct assume-exp fixed512 SWA requires "
            "capacity/window/scan 512, 72 query heads, 8 KV heads, and D128"
        )
    _laguna_swa_attention_decode_split_exact_gated_bf16_spans(
        _SYMBOL_SWA_ATTENTION_FUSED_EXACT_GATED_MIXED32_EXP32_PRODUCER_MAX_GATE_DPP_QK_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512,
        query_ptr,
        key_cache_ptr,
        value_cache_ptr,
        out_ptr,
        gate_ptr,
        gated_out_ptr,
        score_scratch_ptr,
        physical_scratch_ptr,
        spans,
        scan_slots,
        num_q_heads,
        num_kv_heads,
        head_dim,
        scale,
        sliding_window=window,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def laguna_swa_attention_decode_fused_exact_gated_mixed32_exp32_producer_max_gate_stage_pcache_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans(
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    gate_ptr: int,
    gated_out_ptr: int,
    score_scratch_ptr: int,
    physical_scratch_ptr: int,
    spans: KVLiveSpans,
    scan_slots: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    sliding_window: int | None = None,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch exact SWA with per-stage shared probabilities."""

    capacity = _check_swa_spans(spans, num_kv_heads, head_dim)
    window = capacity if sliding_window is None else int(sliding_window)
    if (
        capacity != 512
        or window != 512
        or int(scan_slots) != 512
        or int(num_q_heads) != 72
        or int(num_kv_heads) != 8
        or int(head_dim) != 128
    ):
        raise ValueError(
            "mixed32 exp32 producer-max-gate stage-pcache V-stage64 "
            "vec16-direct assume-exp fixed512 SWA requires "
            "capacity/window/scan 512, 72 query heads, 8 KV heads, and D128"
        )
    _laguna_swa_attention_decode_split_exact_gated_bf16_spans(
        _SYMBOL_SWA_ATTENTION_FUSED_EXACT_GATED_MIXED32_EXP32_PRODUCER_MAX_GATE_STAGE_PCACHE_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512,
        query_ptr,
        key_cache_ptr,
        value_cache_ptr,
        out_ptr,
        gate_ptr,
        gated_out_ptr,
        score_scratch_ptr,
        physical_scratch_ptr,
        spans,
        scan_slots,
        num_q_heads,
        num_kv_heads,
        head_dim,
        scale,
        sliding_window=window,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def laguna_swa_attention_decode_fused_exact_gated_mixed32_exp32_producer_max_gate_stage_pcache_idle_producer_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans(
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    gate_ptr: int,
    gated_out_ptr: int,
    score_scratch_ptr: int,
    physical_scratch_ptr: int,
    spans: KVLiveSpans,
    scan_slots: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    sliding_window: int | None = None,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch exact stage-cache SWA with idle pair-owner producers."""

    capacity = _check_swa_spans(spans, num_kv_heads, head_dim)
    window = capacity if sliding_window is None else int(sliding_window)
    if (
        capacity != 512
        or window != 512
        or int(scan_slots) != 512
        or int(num_q_heads) != 72
        or int(num_kv_heads) != 8
        or int(head_dim) != 128
    ):
        raise ValueError(
            "mixed32 exp32 producer-max-gate stage-pcache idle-producer "
            "V-stage64 vec16-direct assume-exp fixed512 SWA requires "
            "capacity/window/scan 512, 72 query heads, 8 KV heads, and D128"
        )
    _laguna_swa_attention_decode_split_exact_gated_bf16_spans(
        _SYMBOL_SWA_ATTENTION_FUSED_EXACT_GATED_MIXED32_EXP32_PRODUCER_MAX_GATE_STAGE_PCACHE_IDLE_PRODUCER_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512,
        query_ptr,
        key_cache_ptr,
        value_cache_ptr,
        out_ptr,
        gate_ptr,
        gated_out_ptr,
        score_scratch_ptr,
        physical_scratch_ptr,
        spans,
        scan_slots,
        num_q_heads,
        num_kv_heads,
        head_dim,
        scale,
        sliding_window=window,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def laguna_swa_attention_decode_fused_exact_gated_mixed32_exp32_producer_max_gate_stage_pcache_vec4_denom_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans(
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    gate_ptr: int,
    gated_out_ptr: int,
    score_scratch_ptr: int,
    physical_scratch_ptr: int,
    spans: KVLiveSpans,
    scan_slots: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    sliding_window: int | None = None,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch exact stage-cache SWA with vectorized denominator replay."""

    capacity = _check_swa_spans(spans, num_kv_heads, head_dim)
    window = capacity if sliding_window is None else int(sliding_window)
    if (
        capacity != 512
        or window != 512
        or int(scan_slots) != 512
        or int(num_q_heads) != 72
        or int(num_kv_heads) != 8
        or int(head_dim) != 128
    ):
        raise ValueError(
            "mixed32 exp32 producer-max-gate stage-pcache vec4-denom "
            "V-stage64 vec16-direct assume-exp fixed512 SWA requires "
            "capacity/window/scan 512, 72 query heads, 8 KV heads, and D128"
        )
    _laguna_swa_attention_decode_split_exact_gated_bf16_spans(
        _SYMBOL_SWA_ATTENTION_FUSED_EXACT_GATED_MIXED32_EXP32_PRODUCER_MAX_GATE_STAGE_PCACHE_VEC4_DENOM_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512,
        query_ptr,
        key_cache_ptr,
        value_cache_ptr,
        out_ptr,
        gate_ptr,
        gated_out_ptr,
        score_scratch_ptr,
        physical_scratch_ptr,
        spans,
        scan_slots,
        num_q_heads,
        num_kv_heads,
        head_dim,
        scale,
        sliding_window=window,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def laguna_swa_attention_decode_fused_exact_gated_mixed32_exp32_producer_max_gate_stage_pcache_vec4_denom_probability_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans(
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    gate_ptr: int,
    gated_out_ptr: int,
    score_scratch_ptr: int,
    physical_scratch_ptr: int,
    spans: KVLiveSpans,
    scan_slots: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    sliding_window: int | None = None,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch exact stage-cache SWA with vectorized denominator and PV reads."""

    capacity = _check_swa_spans(spans, num_kv_heads, head_dim)
    window = capacity if sliding_window is None else int(sliding_window)
    if (
        capacity != 512
        or window != 512
        or int(scan_slots) != 512
        or int(num_q_heads) != 72
        or int(num_kv_heads) != 8
        or int(head_dim) != 128
    ):
        raise ValueError(
            "mixed32 exp32 producer-max-gate stage-pcache vec4-denom-"
            "probability V-stage64 vec16-direct assume-exp fixed512 SWA "
            "requires capacity/window/scan 512, 72 query heads, 8 KV heads, "
            "and D128"
        )
    _laguna_swa_attention_decode_split_exact_gated_bf16_spans(
        _SYMBOL_SWA_ATTENTION_FUSED_EXACT_GATED_MIXED32_EXP32_PRODUCER_MAX_GATE_STAGE_PCACHE_VEC4_DENOM_PROBABILITY_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512,
        query_ptr,
        key_cache_ptr,
        value_cache_ptr,
        out_ptr,
        gate_ptr,
        gated_out_ptr,
        score_scratch_ptr,
        physical_scratch_ptr,
        spans,
        scan_slots,
        num_q_heads,
        num_kv_heads,
        head_dim,
        scale,
        sliding_window=window,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def laguna_swa_attention_decode_fused_exact_gated_mixed32_exp32_producer_max_gate_stage_pcache_idle_vec4_denom_probability_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans(
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    gate_ptr: int,
    gated_out_ptr: int,
    score_scratch_ptr: int,
    physical_scratch_ptr: int,
    spans: KVLiveSpans,
    scan_slots: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    sliding_window: int | None = None,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch exact stage-cache SWA with pair-owner idle denominator replay."""

    capacity = _check_swa_spans(spans, num_kv_heads, head_dim)
    window = capacity if sliding_window is None else int(sliding_window)
    if (
        capacity != 512
        or window != 512
        or int(scan_slots) != 512
        or int(num_q_heads) != 72
        or int(num_kv_heads) != 8
        or int(head_dim) != 128
    ):
        raise ValueError(
            "mixed32 exp32 producer-max-gate stage-pcache idle-vec4-denom-"
            "probability V-stage64 vec16-direct assume-exp fixed512 SWA "
            "requires capacity/window/scan 512, 72 query heads, 8 KV heads, "
            "and D128"
        )
    _laguna_swa_attention_decode_split_exact_gated_bf16_spans(
        _SYMBOL_SWA_ATTENTION_FUSED_EXACT_GATED_MIXED32_EXP32_PRODUCER_MAX_GATE_STAGE_PCACHE_IDLE_VEC4_DENOM_PROBABILITY_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512,
        query_ptr,
        key_cache_ptr,
        value_cache_ptr,
        out_ptr,
        gate_ptr,
        gated_out_ptr,
        score_scratch_ptr,
        physical_scratch_ptr,
        spans,
        scan_slots,
        num_q_heads,
        num_kv_heads,
        head_dim,
        scale,
        sliding_window=window,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def laguna_swa_attention_decode_fused_exact_gated_mixed40_exp32_producer_max_gate_stage_pcache_idle_vec4_denom_probability_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans(
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    gate_ptr: int,
    gated_out_ptr: int,
    score_scratch_ptr: int,
    physical_scratch_ptr: int,
    spans: KVLiveSpans,
    scan_slots: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    sliding_window: int | None = None,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch exact mixed40 SWA with idle-wave vector denominator replay."""

    capacity = _check_swa_spans(spans, num_kv_heads, head_dim)
    window = capacity if sliding_window is None else int(sliding_window)
    if (
        capacity != 512
        or window != 512
        or int(scan_slots) != 512
        or int(num_q_heads) != 72
        or int(num_kv_heads) != 8
        or int(head_dim) != 128
    ):
        raise ValueError(
            "mixed40 exp32 producer-max-gate stage-pcache idle-vec4-denom-"
            "probability V-stage64 vec16-direct assume-exp fixed512 SWA "
            "requires capacity/window/scan 512, 72 query heads, 8 KV heads, "
            "and D128"
        )
    _laguna_swa_attention_decode_split_exact_gated_bf16_spans(
        _SYMBOL_SWA_ATTENTION_FUSED_EXACT_GATED_MIXED40_EXP32_PRODUCER_MAX_GATE_STAGE_PCACHE_IDLE_VEC4_DENOM_PROBABILITY_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512,
        query_ptr,
        key_cache_ptr,
        value_cache_ptr,
        out_ptr,
        gate_ptr,
        gated_out_ptr,
        score_scratch_ptr,
        physical_scratch_ptr,
        spans,
        scan_slots,
        num_q_heads,
        num_kv_heads,
        head_dim,
        scale,
        sliding_window=window,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def laguna_swa_attention_decode_fused_exact_gated_mixed40_exp32_producer_max_gate_stage_pcache_tail_producer_idle_vec4_denom_probability_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans(
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    gate_ptr: int,
    gated_out_ptr: int,
    score_scratch_ptr: int,
    physical_scratch_ptr: int,
    spans: KVLiveSpans,
    scan_slots: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    sliding_window: int | None = None,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch mixed40 SWA with separate tail exp and denominator waves."""

    capacity = _check_swa_spans(spans, num_kv_heads, head_dim)
    window = capacity if sliding_window is None else int(sliding_window)
    if (
        capacity != 512
        or window != 512
        or int(scan_slots) != 512
        or int(num_q_heads) != 72
        or int(num_kv_heads) != 8
        or int(head_dim) != 128
    ):
        raise ValueError(
            "mixed40 exp32 producer-max-gate stage-pcache tail-producer "
            "idle-vec4-denom probability V-stage64 vec16-direct assume-exp "
            "fixed512 SWA requires capacity/window/scan 512, 72 query heads, "
            "8 KV heads, and D128"
        )
    _laguna_swa_attention_decode_split_exact_gated_bf16_spans(
        _SYMBOL_SWA_ATTENTION_FUSED_EXACT_GATED_MIXED40_EXP32_PRODUCER_MAX_GATE_STAGE_PCACHE_TAIL_PRODUCER_IDLE_VEC4_DENOM_PROBABILITY_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512,
        query_ptr,
        key_cache_ptr,
        value_cache_ptr,
        out_ptr,
        gate_ptr,
        gated_out_ptr,
        score_scratch_ptr,
        physical_scratch_ptr,
        spans,
        scan_slots,
        num_q_heads,
        num_kv_heads,
        head_dim,
        scale,
        sliding_window=window,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def laguna_swa_attention_decode_fused_exact_gated_mixed40_local512_exp32_producer_max_gate_stage_pcache_tail_producer_idle_vec4_denom_probability_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans(
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    gate_ptr: int,
    gated_out_ptr: int,
    score_scratch_ptr: int,
    physical_scratch_ptr: int,
    spans: KVLiveSpans,
    scan_slots: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    sliding_window: int | None = None,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch the exact mixed40 local512 scheduling specialization."""

    capacity = _check_swa_spans(spans, num_kv_heads, head_dim)
    window = capacity if sliding_window is None else int(sliding_window)
    if (
        capacity != 512
        or window != 512
        or int(scan_slots) != 512
        or int(num_q_heads) != 72
        or int(num_kv_heads) != 8
        or int(head_dim) != 128
    ):
        raise ValueError(
            "mixed40 local512 fixed512 SWA requires capacity/window/scan "
            "512, 72 query heads, 8 KV heads, and D128"
        )
    _laguna_swa_attention_decode_split_exact_gated_bf16_spans(
        _SYMBOL_SWA_ATTENTION_FUSED_EXACT_GATED_MIXED40_LOCAL512_EXP32_PRODUCER_MAX_GATE_STAGE_PCACHE_TAIL_PRODUCER_IDLE_VEC4_DENOM_PROBABILITY_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512,
        query_ptr,
        key_cache_ptr,
        value_cache_ptr,
        out_ptr,
        gate_ptr,
        gated_out_ptr,
        score_scratch_ptr,
        physical_scratch_ptr,
        spans,
        scan_slots,
        num_q_heads,
        num_kv_heads,
        head_dim,
        scale,
        sliding_window=window,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def laguna_swa_attention_decode_fused_exact_gated_mixed40_local512_exp32_producer_max_gate_stage_pcache_tail_producer_value_tail_idle_vec4_denom_probability_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans(
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    gate_ptr: int,
    gated_out_ptr: int,
    score_scratch_ptr: int,
    physical_scratch_ptr: int,
    spans: KVLiveSpans,
    scan_slots: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    sliding_window: int | None = None,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch local512 mixed40 SWA with producer-wave staged-V tail copies."""

    capacity = _check_swa_spans(spans, num_kv_heads, head_dim)
    window = capacity if sliding_window is None else int(sliding_window)
    if (
        capacity != 512
        or window != 512
        or int(scan_slots) != 512
        or int(num_q_heads) != 72
        or int(num_kv_heads) != 8
        or int(head_dim) != 128
    ):
        raise ValueError(
            "mixed40 local512 tail-producer value-tail fixed512 SWA requires "
            "capacity/window/scan 512, 72 query heads, 8 KV heads, and D128"
        )
    _laguna_swa_attention_decode_split_exact_gated_bf16_spans(
        _SYMBOL_SWA_ATTENTION_FUSED_EXACT_GATED_MIXED40_LOCAL512_EXP32_PRODUCER_MAX_GATE_STAGE_PCACHE_TAIL_PRODUCER_VALUE_TAIL_IDLE_VEC4_DENOM_PROBABILITY_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512,
        query_ptr,
        key_cache_ptr,
        value_cache_ptr,
        out_ptr,
        gate_ptr,
        gated_out_ptr,
        score_scratch_ptr,
        physical_scratch_ptr,
        spans,
        scan_slots,
        num_q_heads,
        num_kv_heads,
        head_dim,
        scale,
        sliding_window=window,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def laguna_swa_attention_decode_fused_exact_gated_mixed40_exp32_producer_max_gate_stage_pcache_tail_producer_denom_prefetch4_idle_vec4_probability_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans(
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    gate_ptr: int,
    gated_out_ptr: int,
    score_scratch_ptr: int,
    physical_scratch_ptr: int,
    spans: KVLiveSpans,
    scan_slots: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    sliding_window: int | None = None,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch mixed40 SWA with four-vector denominator LDS prefetch."""

    capacity = _check_swa_spans(spans, num_kv_heads, head_dim)
    window = capacity if sliding_window is None else int(sliding_window)
    if (
        capacity != 512
        or window != 512
        or int(scan_slots) != 512
        or int(num_q_heads) != 72
        or int(num_kv_heads) != 8
        or int(head_dim) != 128
    ):
        raise ValueError(
            "mixed40 exp32 producer-max-gate stage-pcache tail-producer "
            "denom-prefetch4 idle-vec4 probability V-stage64 vec16-direct "
            "assume-exp fixed512 SWA requires capacity/window/scan 512, "
            "72 query heads, 8 KV heads, and D128"
        )
    _laguna_swa_attention_decode_split_exact_gated_bf16_spans(
        _SYMBOL_SWA_ATTENTION_FUSED_EXACT_GATED_MIXED40_EXP32_PRODUCER_MAX_GATE_STAGE_PCACHE_TAIL_PRODUCER_DENOM_PREFETCH4_IDLE_VEC4_PROBABILITY_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512,
        query_ptr,
        key_cache_ptr,
        value_cache_ptr,
        out_ptr,
        gate_ptr,
        gated_out_ptr,
        score_scratch_ptr,
        physical_scratch_ptr,
        spans,
        scan_slots,
        num_q_heads,
        num_kv_heads,
        head_dim,
        scale,
        sliding_window=window,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def laguna_swa_attention_decode_fused_exact_gated_mixed40_local512_exp32_producer_max_gate_stage_pcache_tail_producer_value_tail_idle_vec4_denom_probability_vstage128_vec16_direct_assume_exp_fixed512_bf16_spans(
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    gate_ptr: int,
    gated_out_ptr: int,
    score_scratch_ptr: int,
    physical_scratch_ptr: int,
    spans: KVLiveSpans,
    scan_slots: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    sliding_window: int | None = None,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch local512 mixed40 SWA with 128-slot staged-V replay."""

    capacity = _check_swa_spans(spans, num_kv_heads, head_dim)
    window = capacity if sliding_window is None else int(sliding_window)
    if (
        capacity != 512
        or window != 512
        or int(scan_slots) != 512
        or int(num_q_heads) != 72
        or int(num_kv_heads) != 8
        or int(head_dim) != 128
    ):
        raise ValueError(
            "mixed40 local512 tail-producer value-tail V-stage128 fixed512 "
            "SWA requires capacity/window/scan 512, 72 query heads, "
            "8 KV heads, and D128"
        )
    _laguna_swa_attention_decode_split_exact_gated_bf16_spans(
        _SYMBOL_SWA_ATTENTION_FUSED_EXACT_GATED_MIXED40_LOCAL512_EXP32_PRODUCER_MAX_GATE_STAGE_PCACHE_TAIL_PRODUCER_VALUE_TAIL_IDLE_VEC4_DENOM_PROBABILITY_VSTAGE128_VEC16_DIRECT_ASSUME_EXP_FIXED512,
        query_ptr,
        key_cache_ptr,
        value_cache_ptr,
        out_ptr,
        gate_ptr,
        gated_out_ptr,
        score_scratch_ptr,
        physical_scratch_ptr,
        spans,
        scan_slots,
        num_q_heads,
        num_kv_heads,
        head_dim,
        scale,
        sliding_window=window,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def laguna_swa_attention_decode_fused_exact_gated_mixed40_local512_exp32_producer_max_gate_stage_pcache_dual_tail_producer_value_tail_idle_vec4_denom_probability_vstage128_vec16_direct_assume_exp_fixed512_bf16_spans(
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    gate_ptr: int,
    gated_out_ptr: int,
    score_scratch_ptr: int,
    physical_scratch_ptr: int,
    spans: KVLiveSpans,
    scan_slots: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    sliding_window: int | None = None,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch local512 mixed40 SWA with two probability waves per query."""

    capacity = _check_swa_spans(spans, num_kv_heads, head_dim)
    window = capacity if sliding_window is None else int(sliding_window)
    if (
        capacity != 512
        or window != 512
        or int(scan_slots) != 512
        or int(num_q_heads) != 72
        or int(num_kv_heads) != 8
        or int(head_dim) != 128
    ):
        raise ValueError(
            "mixed40 local512 dual tail-producer V-stage128 fixed512 "
            "SWA requires capacity/window/scan 512, 72 query heads, "
            "8 KV heads, and D128"
        )
    _laguna_swa_attention_decode_split_exact_gated_bf16_spans(
        _SYMBOL_SWA_ATTENTION_FUSED_EXACT_GATED_MIXED40_LOCAL512_EXP32_PRODUCER_MAX_GATE_STAGE_PCACHE_DUAL_TAIL_PRODUCER_VALUE_TAIL_IDLE_VEC4_DENOM_PROBABILITY_VSTAGE128_VEC16_DIRECT_ASSUME_EXP_FIXED512,
        query_ptr,
        key_cache_ptr,
        value_cache_ptr,
        out_ptr,
        gate_ptr,
        gated_out_ptr,
        score_scratch_ptr,
        physical_scratch_ptr,
        spans,
        scan_slots,
        num_q_heads,
        num_kv_heads,
        head_dim,
        scale,
        sliding_window=window,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def laguna_swa_attention_decode_fused_exact_gated_mixed40_local512_exp32_producer_max_gate_stage_pcache_output_sharded_probability_allwave_value_idle_vec4_denom_probability_vstage128_vec16_direct_assume_exp_fixed512_bf16_spans(
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    gate_ptr: int,
    gated_out_ptr: int,
    score_scratch_ptr: int,
    physical_scratch_ptr: int,
    spans: KVLiveSpans,
    scan_slots: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    sliding_window: int | None = None,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch V128 SWA with probability shards on the output waves."""

    capacity = _check_swa_spans(spans, num_kv_heads, head_dim)
    window = capacity if sliding_window is None else int(sliding_window)
    if (
        capacity != 512
        or window != 512
        or int(scan_slots) != 512
        or int(num_q_heads) != 72
        or int(num_kv_heads) != 8
        or int(head_dim) != 128
    ):
        raise ValueError(
            "mixed40 local512 output-sharded probability V-stage128 "
            "fixed512 SWA requires capacity/window/scan 512, 72 query "
            "heads, 8 KV heads, and D128"
        )
    _laguna_swa_attention_decode_split_exact_gated_bf16_spans(
        _SYMBOL_SWA_ATTENTION_FUSED_EXACT_GATED_MIXED40_LOCAL512_EXP32_PRODUCER_MAX_GATE_STAGE_PCACHE_OUTPUT_SHARDED_PROBABILITY_ALLWAVE_VALUE_IDLE_VEC4_DENOM_PROBABILITY_VSTAGE128_VEC16_DIRECT_ASSUME_EXP_FIXED512,
        query_ptr,
        key_cache_ptr,
        value_cache_ptr,
        out_ptr,
        gate_ptr,
        gated_out_ptr,
        score_scratch_ptr,
        physical_scratch_ptr,
        spans,
        scan_slots,
        num_q_heads,
        num_kv_heads,
        head_dim,
        scale,
        sliding_window=window,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def laguna_swa_attention_decode_fused_exact_gated_mixed40_local512_exp32_producer_max_gate_stage_pcache_output_sharded_probability_dpp_qk_allwave_value_idle_vec4_denom_probability_vstage128_vec16_direct_assume_exp_fixed512_bf16_spans(
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    gate_ptr: int,
    gated_out_ptr: int,
    score_scratch_ptr: int,
    physical_scratch_ptr: int,
    spans: KVLiveSpans,
    scan_slots: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    sliding_window: int | None = None,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch output-sharded V128 SWA with exact DPP QK transport."""

    capacity = _check_swa_spans(spans, num_kv_heads, head_dim)
    window = capacity if sliding_window is None else int(sliding_window)
    if (
        capacity != 512
        or window != 512
        or int(scan_slots) != 512
        or int(num_q_heads) != 72
        or int(num_kv_heads) != 8
        or int(head_dim) != 128
    ):
        raise ValueError(
            "mixed40 local512 output-sharded probability DPP-QK V-stage128 "
            "fixed512 SWA requires capacity/window/scan 512, 72 query "
            "heads, 8 KV heads, and D128"
        )
    _laguna_swa_attention_decode_split_exact_gated_bf16_spans(
        _SYMBOL_SWA_ATTENTION_FUSED_EXACT_GATED_MIXED40_LOCAL512_EXP32_PRODUCER_MAX_GATE_STAGE_PCACHE_OUTPUT_SHARDED_PROBABILITY_DPP_QK_ALLWAVE_VALUE_IDLE_VEC4_DENOM_PROBABILITY_VSTAGE128_VEC16_DIRECT_ASSUME_EXP_FIXED512,
        query_ptr,
        key_cache_ptr,
        value_cache_ptr,
        out_ptr,
        gate_ptr,
        gated_out_ptr,
        score_scratch_ptr,
        physical_scratch_ptr,
        spans,
        scan_slots,
        num_q_heads,
        num_kv_heads,
        head_dim,
        scale,
        sliding_window=window,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def laguna_swa_attention_decode_fused_exact_gated_mixed40_local512_exp32_producer_max_gate_stage_pcache_output_sharded_probability_dpp_qk_dense_ring_allwave_value_idle_vec4_denom_probability_vstage128_vec16_direct_assume_exp_fixed512_bf16_spans(
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    gate_ptr: int,
    gated_out_ptr: int,
    score_scratch_ptr: int,
    physical_scratch_ptr: int,
    spans: KVLiveSpans,
    scan_slots: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    sliding_window: int | None = None,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch exact metadata-free output-sharded SWA for a dense full ring."""

    capacity = _check_swa_spans(spans, num_kv_heads, head_dim)
    window = capacity if sliding_window is None else int(sliding_window)
    if (
        capacity != 512
        or window != 512
        or int(scan_slots) != 512
        or int(num_q_heads) != 72
        or int(num_kv_heads) != 8
        or int(head_dim) != 128
    ):
        raise ValueError(
            "mixed40 local512 output-sharded probability DPP-QK dense-ring "
            "V-stage128 fixed512 SWA requires capacity/window/scan 512, "
            "72 query heads, 8 KV heads, and D128"
        )
    _laguna_swa_attention_decode_split_exact_gated_bf16_spans(
        _SYMBOL_SWA_ATTENTION_FUSED_EXACT_GATED_MIXED40_LOCAL512_EXP32_PRODUCER_MAX_GATE_STAGE_PCACHE_OUTPUT_SHARDED_PROBABILITY_DPP_QK_DENSE_RING_ALLWAVE_VALUE_IDLE_VEC4_DENOM_PROBABILITY_VSTAGE128_VEC16_DIRECT_ASSUME_EXP_FIXED512,
        query_ptr,
        key_cache_ptr,
        value_cache_ptr,
        out_ptr,
        gate_ptr,
        gated_out_ptr,
        score_scratch_ptr,
        physical_scratch_ptr,
        spans,
        scan_slots,
        num_q_heads,
        num_kv_heads,
        head_dim,
        scale,
        sliding_window=window,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def laguna_swa_attention_decode_fused_exact_gated_mixed40_local1024_exp32_producer_max_gate_stage_pcache_output_sharded_probability_dpp_qk_dense_ring_allwave_value_idle_vec4_denom_probability_vstage128_vec16_direct_assume_exp_fixed512_bf16_spans(
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    gate_ptr: int,
    gated_out_ptr: int,
    score_scratch_ptr: int,
    physical_scratch_ptr: int,
    spans: KVLiveSpans,
    scan_slots: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    sliding_window: int | None = None,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch exact dense-ring SWA with the full 32-wave gfx1151 block."""

    capacity = _check_swa_spans(spans, num_kv_heads, head_dim)
    window = capacity if sliding_window is None else int(sliding_window)
    if (
        capacity != 512
        or window != 512
        or int(scan_slots) != 512
        or int(num_q_heads) != 72
        or int(num_kv_heads) != 8
        or int(head_dim) != 128
    ):
        raise ValueError(
            "mixed40 local1024 output-sharded probability DPP-QK dense-ring "
            "V-stage128 fixed512 SWA requires capacity/window/scan 512, "
            "72 query heads, 8 KV heads, and D128"
        )
    _laguna_swa_attention_decode_split_exact_gated_bf16_spans(
        _SYMBOL_SWA_ATTENTION_FUSED_EXACT_GATED_MIXED40_LOCAL1024_EXP32_PRODUCER_MAX_GATE_STAGE_PCACHE_OUTPUT_SHARDED_PROBABILITY_DPP_QK_DENSE_RING_ALLWAVE_VALUE_IDLE_VEC4_DENOM_PROBABILITY_VSTAGE128_VEC16_DIRECT_ASSUME_EXP_FIXED512,
        query_ptr,
        key_cache_ptr,
        value_cache_ptr,
        out_ptr,
        gate_ptr,
        gated_out_ptr,
        score_scratch_ptr,
        physical_scratch_ptr,
        spans,
        scan_slots,
        num_q_heads,
        num_kv_heads,
        head_dim,
        scale,
        sliding_window=window,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def laguna_swa_attention_decode_fused_exact_gated_mixed40_exp32_producer_max_gate_stage_pcache_tail_producer_value_tail_idle_vec4_denom_probability_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans(
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    gate_ptr: int,
    gated_out_ptr: int,
    score_scratch_ptr: int,
    physical_scratch_ptr: int,
    spans: KVLiveSpans,
    scan_slots: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    sliding_window: int | None = None,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch mixed40 SWA with producer waves copying the staged-V tail."""

    capacity = _check_swa_spans(spans, num_kv_heads, head_dim)
    window = capacity if sliding_window is None else int(sliding_window)
    if (
        capacity != 512
        or window != 512
        or int(scan_slots) != 512
        or int(num_q_heads) != 72
        or int(num_kv_heads) != 8
        or int(head_dim) != 128
    ):
        raise ValueError(
            "mixed40 exp32 producer-max-gate stage-pcache tail-producer "
            "value-tail idle-vec4-denom probability V-stage64 vec16-direct "
            "assume-exp fixed512 SWA requires capacity/window/scan 512, "
            "72 query heads, 8 KV heads, and D128"
        )
    _laguna_swa_attention_decode_split_exact_gated_bf16_spans(
        _SYMBOL_SWA_ATTENTION_FUSED_EXACT_GATED_MIXED40_EXP32_PRODUCER_MAX_GATE_STAGE_PCACHE_TAIL_PRODUCER_VALUE_TAIL_IDLE_VEC4_DENOM_PROBABILITY_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512,
        query_ptr,
        key_cache_ptr,
        value_cache_ptr,
        out_ptr,
        gate_ptr,
        gated_out_ptr,
        score_scratch_ptr,
        physical_scratch_ptr,
        spans,
        scan_slots,
        num_q_heads,
        num_kv_heads,
        head_dim,
        scale,
        sliding_window=window,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def laguna_swa_attention_decode_fused_exact_gated_mixed32_exp32_producer_max_gate_stage_pcache_dpp_qk_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans(
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    gate_ptr: int,
    gated_out_ptr: int,
    score_scratch_ptr: int,
    physical_scratch_ptr: int,
    spans: KVLiveSpans,
    scan_slots: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    sliding_window: int | None = None,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch exact stage-probability-cache SWA with DPP QK transport."""

    capacity = _check_swa_spans(spans, num_kv_heads, head_dim)
    window = capacity if sliding_window is None else int(sliding_window)
    if (
        capacity != 512
        or window != 512
        or int(scan_slots) != 512
        or int(num_q_heads) != 72
        or int(num_kv_heads) != 8
        or int(head_dim) != 128
    ):
        raise ValueError(
            "mixed32 exp32 producer-max-gate stage-pcache DPP-QK V-stage64 "
            "vec16-direct assume-exp fixed512 SWA requires "
            "capacity/window/scan 512, 72 query heads, 8 KV heads, and D128"
        )
    _laguna_swa_attention_decode_split_exact_gated_bf16_spans(
        _SYMBOL_SWA_ATTENTION_FUSED_EXACT_GATED_MIXED32_EXP32_PRODUCER_MAX_GATE_STAGE_PCACHE_DPP_QK_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512,
        query_ptr,
        key_cache_ptr,
        value_cache_ptr,
        out_ptr,
        gate_ptr,
        gated_out_ptr,
        score_scratch_ptr,
        physical_scratch_ptr,
        spans,
        scan_slots,
        num_q_heads,
        num_kv_heads,
        head_dim,
        scale,
        sliding_window=window,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def laguna_swa_attention_decode_fused_exact_gated_mixed32_exp32_producer_max_gate_gated_only_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans(
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    gate_ptr: int,
    gated_out_ptr: int,
    score_scratch_ptr: int,
    physical_scratch_ptr: int,
    spans: KVLiveSpans,
    scan_slots: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    sliding_window: int | None = None,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch exact producer-max/gate SWA without the dead F32 context store."""

    capacity = _check_swa_spans(spans, num_kv_heads, head_dim)
    window = capacity if sliding_window is None else int(sliding_window)
    if (
        capacity != 512
        or window != 512
        or int(scan_slots) != 512
        or int(num_q_heads) != 72
        or int(num_kv_heads) != 8
        or int(head_dim) != 128
    ):
        raise ValueError(
            "mixed32 exp32 producer-max-gate gated-only V-stage64 "
            "vec16-direct assume-exp fixed512 SWA requires "
            "capacity/window/scan 512, 72 query heads, 8 KV heads, and D128"
        )
    _laguna_swa_attention_decode_split_exact_gated_bf16_spans(
        _SYMBOL_SWA_ATTENTION_FUSED_EXACT_GATED_MIXED32_EXP32_PRODUCER_MAX_GATE_GATED_ONLY_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512,
        query_ptr,
        key_cache_ptr,
        value_cache_ptr,
        out_ptr,
        gate_ptr,
        gated_out_ptr,
        score_scratch_ptr,
        physical_scratch_ptr,
        spans,
        scan_slots,
        num_q_heads,
        num_kv_heads,
        head_dim,
        scale,
        sliding_window=window,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def laguna_swa_attention_decode_split_tile16_exact_gated_wave_local_dim2_bf16_spans(
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    gate_ptr: int,
    gated_out_ptr: int,
    score_scratch_ptr: int,
    physical_scratch_ptr: int,
    spans: KVLiveSpans,
    scan_slots: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    sliding_window: int | None = None,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Run exact tile16 split SWA with two adjacent dimensions per thread."""

    _laguna_swa_attention_decode_split_exact_gated_bf16_spans(
        _SYMBOL_SWA_ATTENTION_SPLIT_TILE16_EXACT_GATED_WAVE_LOCAL_DIM2,
        query_ptr,
        key_cache_ptr,
        value_cache_ptr,
        out_ptr,
        gate_ptr,
        gated_out_ptr,
        score_scratch_ptr,
        physical_scratch_ptr,
        spans,
        scan_slots,
        num_q_heads,
        num_kv_heads,
        head_dim,
        scale,
        sliding_window=sliding_window,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def _laguna_swa_attention_decode_split_exact_gated_bf16_spans(
    symbol: str,
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    gate_ptr: int,
    gated_out_ptr: int,
    score_scratch_ptr: int,
    physical_scratch_ptr: int,
    spans: KVLiveSpans,
    scan_slots: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    sliding_window: int | None,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    capacity = _check_swa_spans(spans, num_kv_heads, head_dim)
    parsed_scan = _check_split_scan_slots(scan_slots, capacity)
    _check_laguna_attention_shape(num_q_heads, num_kv_heads, head_dim)
    window = capacity if sliding_window is None else int(sliding_window)
    if window <= 0 or window > capacity:
        raise ValueError("sliding_window must be in [1, ring capacity]")
    library = library or build_laguna_kv_attention(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, symbol)
    fn.argtypes = (
        [ctypes.c_void_p] * 13
        + [ctypes.c_int64] * 7
        + [ctypes.c_float, ctypes.c_void_p]
    )
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(query_ptr),
        ctypes.c_void_p(key_cache_ptr),
        ctypes.c_void_p(value_cache_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_void_p(gate_ptr),
        ctypes.c_void_p(gated_out_ptr),
        ctypes.c_void_p(score_scratch_ptr),
        ctypes.c_void_p(physical_scratch_ptr),
        ctypes.c_void_p(spans.base_offsets.ptr),
        ctypes.c_void_p(spans.live_counts.ptr),
        ctypes.c_void_p(spans.token_positions.ptr),
        ctypes.c_void_p(spans.evict_mask.ptr),
        ctypes.c_void_p(spans.row_positions.ptr),
        ctypes.c_int64(capacity),
        ctypes.c_int64(window),
        ctypes.c_int64(capacity),
        ctypes.c_int64(parsed_scan),
        ctypes.c_int64(num_q_heads),
        ctypes.c_int64(num_kv_heads),
        ctypes.c_int64(head_dim),
        ctypes.c_float(scale),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def laguna_swa_attention_prefill_bf16_spans(
    query_ptr: int,
    current_key_ptr: int,
    current_value_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    spans: KVLiveSpans,
    rows: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    sliding_window: int | None = None,
    start_position: int | None = None,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Run causal GQA over prior ring state plus current consecutive rows."""

    capacity = _check_swa_spans(spans, num_kv_heads, head_dim)
    _check_prefill_rows(spans, rows, capacity)
    _check_laguna_attention_shape(num_q_heads, num_kv_heads, head_dim)
    window = capacity if sliding_window is None else int(sliding_window)
    if window <= 0 or window > capacity:
        raise ValueError("sliding_window must be in [1, ring capacity]")
    library = library or build_laguna_kv_attention(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_SWA_PREFILL)
    fn.argtypes = (
        [ctypes.c_void_p] * 11
        + [ctypes.c_int64] * 6
        + [ctypes.c_float, ctypes.c_void_p]
    )
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(query_ptr),
        ctypes.c_void_p(current_key_ptr),
        ctypes.c_void_p(current_value_ptr),
        ctypes.c_void_p(key_cache_ptr),
        ctypes.c_void_p(value_cache_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_void_p(spans.base_offsets.ptr),
        ctypes.c_void_p(spans.live_counts.ptr),
        ctypes.c_void_p(spans.token_positions.ptr),
        ctypes.c_void_p(spans.evict_mask.ptr),
        ctypes.c_void_p(spans.row_positions.ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(capacity),
        ctypes.c_int64(window),
        ctypes.c_int64(num_q_heads),
        ctypes.c_int64(num_kv_heads),
        ctypes.c_int64(head_dim),
        ctypes.c_float(scale),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def laguna_swa_attention_prefill_wave32_exact_bf16_spans(
    query_ptr: int,
    current_key_ptr: int,
    current_value_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    spans: KVLiveSpans,
    rows: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    sliding_window: int | None = None,
    start_position: int | None = None,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Run reduction-order-exact causal SWA with one wave32 per query head."""

    capacity = _check_swa_spans(spans, num_kv_heads, head_dim)
    _check_prefill_rows(spans, rows, capacity)
    _check_laguna_attention_shape(num_q_heads, num_kv_heads, head_dim)
    window = capacity if sliding_window is None else int(sliding_window)
    if window <= 0 or window > capacity:
        raise ValueError("sliding_window must be in [1, ring capacity]")
    library = library or build_laguna_kv_attention(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_SWA_PREFILL_WAVE32_EXACT)
    fn.argtypes = (
        [ctypes.c_void_p] * 11
        + [ctypes.c_int64] * 6
        + [ctypes.c_float, ctypes.c_void_p]
    )
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(query_ptr),
        ctypes.c_void_p(current_key_ptr),
        ctypes.c_void_p(current_value_ptr),
        ctypes.c_void_p(key_cache_ptr),
        ctypes.c_void_p(value_cache_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_void_p(spans.base_offsets.ptr),
        ctypes.c_void_p(spans.live_counts.ptr),
        ctypes.c_void_p(spans.token_positions.ptr),
        ctypes.c_void_p(spans.evict_mask.ptr),
        ctypes.c_void_p(spans.row_positions.ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(capacity),
        ctypes.c_int64(window),
        ctypes.c_int64(num_q_heads),
        ctypes.c_int64(num_kv_heads),
        ctypes.c_int64(head_dim),
        ctypes.c_float(scale),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def laguna_swa_attention_prefill_qrow2_exact_bf16_spans(
    query_ptr: int,
    current_key_ptr: int,
    current_value_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    spans: KVLiveSpans,
    rows: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    sliding_window: int | None = None,
    start_position: int | None = None,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Run exact SWA while reusing each K/V row across two query rows."""

    capacity = _check_swa_spans(spans, num_kv_heads, head_dim)
    _check_prefill_rows(spans, rows, capacity)
    _check_laguna_attention_shape(num_q_heads, num_kv_heads, head_dim)
    window = capacity if sliding_window is None else int(sliding_window)
    if window <= 0 or window > capacity:
        raise ValueError("sliding_window must be in [1, ring capacity]")
    library = library or build_laguna_kv_attention(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_SWA_PREFILL_QROW2_EXACT)
    fn.argtypes = (
        [ctypes.c_void_p] * 11
        + [ctypes.c_int64] * 6
        + [ctypes.c_float, ctypes.c_void_p]
    )
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(query_ptr),
        ctypes.c_void_p(current_key_ptr),
        ctypes.c_void_p(current_value_ptr),
        ctypes.c_void_p(key_cache_ptr),
        ctypes.c_void_p(value_cache_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_void_p(spans.base_offsets.ptr),
        ctypes.c_void_p(spans.live_counts.ptr),
        ctypes.c_void_p(spans.token_positions.ptr),
        ctypes.c_void_p(spans.evict_mask.ptr),
        ctypes.c_void_p(spans.row_positions.ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(capacity),
        ctypes.c_int64(window),
        ctypes.c_int64(num_q_heads),
        ctypes.c_int64(num_kv_heads),
        ctypes.c_int64(head_dim),
        ctypes.c_float(scale),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def laguna_swa_attention_prefill_qrow2_online_bf16_spans(
    query_ptr: int,
    current_key_ptr: int,
    current_value_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    spans: KVLiveSpans,
    rows: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    sliding_window: int | None = None,
    start_position: int | None = None,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Run two-query-row online-softmax SWA over prior and current rows."""

    capacity = _check_swa_spans(spans, num_kv_heads, head_dim)
    _check_prefill_rows(spans, rows, capacity)
    _check_laguna_attention_shape(num_q_heads, num_kv_heads, head_dim)
    window = capacity if sliding_window is None else int(sliding_window)
    if window <= 0 or window > capacity:
        raise ValueError("sliding_window must be in [1, ring capacity]")
    library = library or build_laguna_kv_attention(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_SWA_PREFILL_QROW2_ONLINE)
    fn.argtypes = (
        [ctypes.c_void_p] * 11
        + [ctypes.c_int64] * 6
        + [ctypes.c_float, ctypes.c_void_p]
    )
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(query_ptr),
        ctypes.c_void_p(current_key_ptr),
        ctypes.c_void_p(current_value_ptr),
        ctypes.c_void_p(key_cache_ptr),
        ctypes.c_void_p(value_cache_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_void_p(spans.base_offsets.ptr),
        ctypes.c_void_p(spans.live_counts.ptr),
        ctypes.c_void_p(spans.token_positions.ptr),
        ctypes.c_void_p(spans.evict_mask.ptr),
        ctypes.c_void_p(spans.row_positions.ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(capacity),
        ctypes.c_int64(window),
        ctypes.c_int64(num_q_heads),
        ctypes.c_int64(num_kv_heads),
        ctypes.c_int64(head_dim),
        ctypes.c_float(scale),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def laguna_swa_attention_prefill_qrow4_online_bf16_spans(
    query_ptr: int,
    current_key_ptr: int,
    current_value_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    spans: KVLiveSpans,
    rows: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    sliding_window: int | None = None,
    start_position: int | None = None,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Run four-query-row online-softmax SWA over prior and current rows."""

    capacity = _check_swa_spans(spans, num_kv_heads, head_dim)
    _check_prefill_rows(spans, rows, capacity)
    _check_laguna_attention_shape(num_q_heads, num_kv_heads, head_dim)
    window = capacity if sliding_window is None else int(sliding_window)
    if window <= 0 or window > capacity:
        raise ValueError("sliding_window must be in [1, ring capacity]")
    library = library or build_laguna_kv_attention(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_SWA_PREFILL_QROW4_ONLINE)
    fn.argtypes = (
        [ctypes.c_void_p] * 11
        + [ctypes.c_int64] * 6
        + [ctypes.c_float, ctypes.c_void_p]
    )
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(query_ptr),
        ctypes.c_void_p(current_key_ptr),
        ctypes.c_void_p(current_value_ptr),
        ctypes.c_void_p(key_cache_ptr),
        ctypes.c_void_p(value_cache_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_void_p(spans.base_offsets.ptr),
        ctypes.c_void_p(spans.live_counts.ptr),
        ctypes.c_void_p(spans.token_positions.ptr),
        ctypes.c_void_p(spans.evict_mask.ptr),
        ctypes.c_void_p(spans.row_positions.ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(capacity),
        ctypes.c_int64(window),
        ctypes.c_int64(num_q_heads),
        ctypes.c_int64(num_kv_heads),
        ctypes.c_int64(head_dim),
        ctypes.c_float(scale),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def laguna_swa_attention_prefill_qrow4_sourcequal_online_bf16_spans(
    query_ptr: int,
    current_key_ptr: int,
    current_value_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    spans: KVLiveSpans,
    rows: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    sliding_window: int | None = None,
    start_position: int | None = None,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Skip K/V sources unused by every visible row in a qrow4 group."""

    capacity = _check_swa_spans(spans, num_kv_heads, head_dim)
    _check_prefill_rows(spans, rows, capacity)
    _check_laguna_attention_shape(num_q_heads, num_kv_heads, head_dim)
    window = capacity if sliding_window is None else int(sliding_window)
    if window <= 0 or window > capacity:
        raise ValueError("sliding_window must be in [1, ring capacity]")
    library = library or build_laguna_kv_attention(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_SWA_PREFILL_QROW4_SOURCEQUAL_ONLINE)
    fn.argtypes = (
        [ctypes.c_void_p] * 11
        + [ctypes.c_int64] * 6
        + [ctypes.c_float, ctypes.c_void_p]
    )
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(query_ptr),
        ctypes.c_void_p(current_key_ptr),
        ctypes.c_void_p(current_value_ptr),
        ctypes.c_void_p(key_cache_ptr),
        ctypes.c_void_p(value_cache_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_void_p(spans.base_offsets.ptr),
        ctypes.c_void_p(spans.live_counts.ptr),
        ctypes.c_void_p(spans.token_positions.ptr),
        ctypes.c_void_p(spans.evict_mask.ptr),
        ctypes.c_void_p(spans.row_positions.ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(capacity),
        ctypes.c_int64(window),
        ctypes.c_int64(num_q_heads),
        ctypes.c_int64(num_kv_heads),
        ctypes.c_int64(head_dim),
        ctypes.c_float(scale),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def laguna_swa_attention_prefill_qrow4_cached_online_bf16_spans(
    query_ptr: int,
    current_key_ptr: int,
    current_value_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    spans: KVLiveSpans,
    rows: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    sliding_window: int | None = None,
    start_position: int | None = None,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Run qrow4 SWA after a pre-wrap tile is already in the BF16 ring."""

    capacity = _check_swa_spans(spans, num_kv_heads, head_dim)
    _check_prefill_rows(spans, rows, capacity)
    _check_laguna_attention_shape(num_q_heads, num_kv_heads, head_dim)
    window = capacity if sliding_window is None else int(sliding_window)
    if window <= 0 or window > capacity:
        raise ValueError("sliding_window must be in [1, ring capacity]")
    if start_position is None or int(start_position) < 0:
        raise ValueError("cached SWA prefill requires a non-negative start_position")
    if int(start_position) + int(rows) > capacity:
        raise ValueError("cached SWA prefill cannot overwrite a wrapped ring tile")
    library = library or build_laguna_kv_attention(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_SWA_PREFILL_QROW4_CACHED_ONLINE)
    fn.argtypes = (
        [ctypes.c_void_p] * 11
        + [ctypes.c_int64] * 6
        + [ctypes.c_float, ctypes.c_void_p]
    )
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(query_ptr),
        ctypes.c_void_p(current_key_ptr),
        ctypes.c_void_p(current_value_ptr),
        ctypes.c_void_p(key_cache_ptr),
        ctypes.c_void_p(value_cache_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_void_p(spans.base_offsets.ptr),
        ctypes.c_void_p(spans.live_counts.ptr),
        ctypes.c_void_p(spans.token_positions.ptr),
        ctypes.c_void_p(spans.evict_mask.ptr),
        ctypes.c_void_p(spans.row_positions.ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(capacity),
        ctypes.c_int64(window),
        ctypes.c_int64(num_q_heads),
        ctypes.c_int64(num_kv_heads),
        ctypes.c_int64(head_dim),
        ctypes.c_float(scale),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def laguna_swa_attention_prefill_qrow4_cached_meta_online_bf16_spans(
    query_ptr: int,
    current_key_ptr: int,
    current_value_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    spans: KVLiveSpans,
    rows: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    sliding_window: int | None = None,
    start_position: int | None = None,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Run pre-wrap qrow4 SWA using only preappended cache visibility metadata."""

    capacity = _check_swa_spans(spans, num_kv_heads, head_dim)
    _check_prefill_rows(spans, rows, capacity)
    _check_laguna_attention_shape(num_q_heads, num_kv_heads, head_dim)
    window = capacity if sliding_window is None else int(sliding_window)
    if window <= 0 or window > capacity:
        raise ValueError("sliding_window must be in [1, ring capacity]")
    if start_position is None or int(start_position) < 0:
        raise ValueError("cached SWA prefill requires a non-negative start_position")
    if int(start_position) + int(rows) > capacity:
        raise ValueError("cached SWA prefill cannot overwrite a wrapped ring tile")
    library = library or build_laguna_kv_attention(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_SWA_PREFILL_QROW4_CACHED_META_ONLINE)
    fn.argtypes = (
        [ctypes.c_void_p] * 11
        + [ctypes.c_int64] * 6
        + [ctypes.c_float, ctypes.c_void_p]
    )
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(query_ptr),
        ctypes.c_void_p(current_key_ptr),
        ctypes.c_void_p(current_value_ptr),
        ctypes.c_void_p(key_cache_ptr),
        ctypes.c_void_p(value_cache_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_void_p(spans.base_offsets.ptr),
        ctypes.c_void_p(spans.live_counts.ptr),
        ctypes.c_void_p(spans.token_positions.ptr),
        ctypes.c_void_p(spans.evict_mask.ptr),
        ctypes.c_void_p(spans.row_positions.ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(capacity),
        ctypes.c_int64(window),
        ctypes.c_int64(num_q_heads),
        ctypes.c_int64(num_kv_heads),
        ctypes.c_int64(head_dim),
        ctypes.c_float(scale),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def laguna_swa_attention_prefill_qrow4_dense_initial_online_bf16_spans(
    query_ptr: int,
    current_key_ptr: int,
    current_value_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    spans: KVLiveSpans,
    rows: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    sliding_window: int | None = None,
    start_position: int | None = None,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Run qrow4 SWA over an identity-positioned initial ring fill."""

    capacity = _check_swa_spans(spans, num_kv_heads, head_dim)
    _check_prefill_rows(spans, rows, capacity)
    _check_laguna_attention_shape(num_q_heads, num_kv_heads, head_dim)
    window = capacity if sliding_window is None else int(sliding_window)
    if window <= 0 or window > capacity:
        raise ValueError("sliding_window must be in [1, ring capacity]")
    if start_position is None or int(start_position) < 0:
        raise ValueError("dense-initial SWA prefill requires a non-negative start_position")
    if int(start_position) + int(rows) > capacity:
        raise ValueError("dense-initial SWA prefill cannot cross the first ring wrap")
    library = library or build_laguna_kv_attention(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_SWA_PREFILL_QROW4_DENSE_INITIAL_ONLINE)
    fn.argtypes = (
        [ctypes.c_void_p] * 11
        + [ctypes.c_int64] * 6
        + [ctypes.c_float, ctypes.c_void_p]
    )
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(query_ptr),
        ctypes.c_void_p(current_key_ptr),
        ctypes.c_void_p(current_value_ptr),
        ctypes.c_void_p(key_cache_ptr),
        ctypes.c_void_p(value_cache_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_void_p(spans.base_offsets.ptr),
        ctypes.c_void_p(spans.live_counts.ptr),
        ctypes.c_void_p(spans.token_positions.ptr),
        ctypes.c_void_p(spans.evict_mask.ptr),
        ctypes.c_void_p(spans.row_positions.ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(capacity),
        ctypes.c_int64(window),
        ctypes.c_int64(num_q_heads),
        ctypes.c_int64(num_kv_heads),
        ctypes.c_int64(head_dim),
        ctypes.c_float(scale),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def laguna_swa_attention_prefill_qrow4_m128_online_bf16_spans(
    query_ptr: int,
    current_key_ptr: int,
    current_value_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    spans: KVLiveSpans,
    rows: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    sliding_window: int | None = None,
    start_position: int | None = None,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Use source-qualified qrow4 for M128 tiles and qrow2 for residuals."""

    kernel = (
        laguna_swa_attention_prefill_qrow4_sourcequal_online_bf16_spans
        if int(rows) == 128
        else laguna_swa_attention_prefill_qrow2_online_bf16_spans
    )
    kernel(
        query_ptr,
        current_key_ptr,
        current_value_ptr,
        key_cache_ptr,
        value_cache_ptr,
        out_ptr,
        spans,
        rows,
        num_q_heads,
        num_kv_heads,
        head_dim,
        scale,
        sliding_window=sliding_window,
        start_position=start_position,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def laguna_swa_attention_prefill_qrow2_m128_c128_exact_bf16_spans(
    query_ptr: int,
    current_key_ptr: int,
    current_value_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    spans: KVLiveSpans,
    rows: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    *,
    sliding_window: int | None = None,
    start_position: int | None = None,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Pair query rows only after both measured row/context crossovers."""

    kernel = (
        laguna_swa_attention_prefill_qrow2_exact_bf16_spans
        if int(rows) == 128
        and start_position is not None
        and int(start_position) >= 128
        else laguna_swa_attention_prefill_wave32_exact_bf16_spans
    )
    kernel(
        query_ptr,
        current_key_ptr,
        current_value_ptr,
        key_cache_ptr,
        value_cache_ptr,
        out_ptr,
        spans,
        rows,
        num_q_heads,
        num_kv_heads,
        head_dim,
        scale,
        sliding_window=sliding_window,
        start_position=start_position,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def laguna_dense_initial_cache_bf16_to_f32_spans(
    key_cache_ptr: int,
    value_cache_ptr: int,
    key_f32_ptr: int,
    value_f32_ptr: int,
    spans: KVLiveSpans,
    context: int,
    num_kv_heads: int,
    head_dim: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Widen one initial identity-mapped BF16 K/V prefix for BLAS attention."""

    if spans.spans_mode == "uniform":
        capacity = _check_global_spans(spans, num_kv_heads, head_dim)
        block_size = _GLOBAL_BLOCK_SIZE
        global_layout = 1
    else:
        capacity = _check_swa_spans(spans, num_kv_heads, head_dim)
        block_size = 1
        global_layout = 0
    parsed_context = int(context)
    if parsed_context <= 0 or parsed_context > int(capacity):
        raise ValueError("dense-initial BLAS context must be within capacity")
    library = library or build_laguna_kv_attention(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_DENSE_INITIAL_CACHE_BF16_TO_F32)
    fn.argtypes = (
        [ctypes.c_void_p] * 8
        + [ctypes.c_int64] * 6
        + [ctypes.c_int, ctypes.c_void_p]
    )
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(key_cache_ptr),
        ctypes.c_void_p(value_cache_ptr),
        ctypes.c_void_p(key_f32_ptr),
        ctypes.c_void_p(value_f32_ptr),
        ctypes.c_void_p(spans.base_offsets.ptr),
        ctypes.c_void_p(spans.live_counts.ptr),
        ctypes.c_void_p(spans.token_positions.ptr),
        ctypes.c_void_p(spans.evict_mask.ptr),
        ctypes.c_int64(parsed_context),
        ctypes.c_int64(capacity),
        ctypes.c_int64(block_size),
        ctypes.c_int64(spans.base_offsets.numel),
        ctypes.c_int64(num_kv_heads),
        ctypes.c_int64(head_dim),
        ctypes.c_int(global_layout),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def laguna_dense_initial_cache_block_bf16_to_f32_spans(
    key_cache_ptr: int,
    value_cache_ptr: int,
    key_f32_ptr: int,
    value_f32_ptr: int,
    spans: KVLiveSpans,
    logical_start: int,
    count: int,
    context: int,
    num_kv_heads: int,
    head_dim: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Widen one initial identity-mapped BF16 K/V block for tiled BLAS."""

    if spans.spans_mode == "uniform":
        capacity = _check_global_spans(spans, num_kv_heads, head_dim)
        block_size = _GLOBAL_BLOCK_SIZE
        global_layout = 1
    else:
        capacity = _check_swa_spans(spans, num_kv_heads, head_dim)
        block_size = 1
        global_layout = 0
    parsed_start = int(logical_start)
    parsed_count = int(count)
    parsed_context = int(context)
    if (
        parsed_start < 0
        or parsed_count <= 0
        or parsed_start + parsed_count > parsed_context
        or parsed_context > int(capacity)
    ):
        raise ValueError("dense-initial BLAS cache block is out of range")
    library = library or build_laguna_kv_attention(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(
        library,
        _SYMBOL_DENSE_INITIAL_CACHE_BLOCK_BF16_TO_F32,
    )
    fn.argtypes = (
        [ctypes.c_void_p] * 8
        + [ctypes.c_int64] * 8
        + [ctypes.c_int, ctypes.c_void_p]
    )
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(key_cache_ptr),
        ctypes.c_void_p(value_cache_ptr),
        ctypes.c_void_p(key_f32_ptr),
        ctypes.c_void_p(value_f32_ptr),
        ctypes.c_void_p(spans.base_offsets.ptr),
        ctypes.c_void_p(spans.live_counts.ptr),
        ctypes.c_void_p(spans.token_positions.ptr),
        ctypes.c_void_p(spans.evict_mask.ptr),
        ctypes.c_int64(parsed_start),
        ctypes.c_int64(parsed_count),
        ctypes.c_int64(parsed_context),
        ctypes.c_int64(capacity),
        ctypes.c_int64(block_size),
        ctypes.c_int64(spans.base_offsets.numel),
        ctypes.c_int64(num_kv_heads),
        ctypes.c_int64(head_dim),
        ctypes.c_int(global_layout),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def laguna_dense_initial_contiguous_cache_block_bf16_to_f32_spans(
    key_cache_ptr: int,
    value_cache_ptr: int,
    key_f32_ptr: int,
    value_f32_ptr: int,
    spans: KVLiveSpans,
    logical_start: int,
    count: int,
    context: int,
    num_kv_heads: int,
    head_dim: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Widen one qualified identity-mapped global BF16 K/V block."""

    if spans.spans_mode != "uniform":
        raise ValueError("dense contiguous cache widening requires global spans")
    capacity = _check_global_spans(spans, num_kv_heads, head_dim)
    block_size = _GLOBAL_BLOCK_SIZE
    parsed_start = int(logical_start)
    parsed_count = int(count)
    parsed_context = int(context)
    if (
        parsed_start < 0
        or parsed_count <= 0
        or parsed_start + parsed_count > parsed_context
        or parsed_context > int(capacity)
    ):
        raise ValueError("dense contiguous cache block is out of range")
    library = library or build_laguna_kv_attention(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(
        library,
        _SYMBOL_DENSE_INITIAL_CONTIGUOUS_CACHE_BLOCK_BF16_TO_F32,
    )
    fn.argtypes = (
        [ctypes.c_void_p] * 8
        + [ctypes.c_int64] * 8
        + [ctypes.c_int, ctypes.c_void_p]
    )
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(key_cache_ptr),
        ctypes.c_void_p(value_cache_ptr),
        ctypes.c_void_p(key_f32_ptr),
        ctypes.c_void_p(value_f32_ptr),
        ctypes.c_void_p(spans.base_offsets.ptr),
        ctypes.c_void_p(spans.live_counts.ptr),
        ctypes.c_void_p(spans.token_positions.ptr),
        ctypes.c_void_p(spans.evict_mask.ptr),
        ctypes.c_int64(parsed_start),
        ctypes.c_int64(parsed_count),
        ctypes.c_int64(parsed_context),
        ctypes.c_int64(capacity),
        ctypes.c_int64(block_size),
        ctypes.c_int64(spans.base_offsets.numel),
        ctypes.c_int64(num_kv_heads),
        ctypes.c_int64(head_dim),
        ctypes.c_int(1),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def laguna_dense_initial_query_head_transpose_f32(
    input_ptr: int,
    output_ptr: int,
    rows: int,
    num_q_heads: int,
    head_dim: int,
    *,
    to_head_major: bool,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Transpose dense-initial F32 query/output rows around batched BLAS."""

    if int(rows) <= 0 or int(rows) > 2_048:
        raise ValueError("dense-initial BLAS rows must be within [1, 2048]")
    _check_laguna_attention_shape(
        num_q_heads,
        _LAGUNA_KV_HEADS,
        head_dim,
    )
    library = library or build_laguna_kv_attention(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_DENSE_INITIAL_QUERY_HEAD_TRANSPOSE_F32)
    fn.argtypes = (
        [ctypes.c_void_p] * 2
        + [ctypes.c_int64] * 3
        + [ctypes.c_int, ctypes.c_void_p]
    )
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(input_ptr),
        ctypes.c_void_p(output_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(num_q_heads),
        ctypes.c_int64(head_dim),
        ctypes.c_int(int(bool(to_head_major))),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def laguna_dense_initial_causal_softmax_f32_spans(
    scores_ptr: int,
    spans: KVLiveSpans,
    rows: int,
    context: int,
    num_q_heads: int,
    start_position: int,
    scale: float,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Apply initial-fill causal softmax to head-major BLAS score matrices."""

    if spans.spans_mode == "uniform":
        capacity = _check_global_spans(spans, _LAGUNA_KV_HEADS, _LAGUNA_HEAD_DIM)
    else:
        capacity = _check_swa_spans(spans, _LAGUNA_KV_HEADS, _LAGUNA_HEAD_DIM)
    _check_prefill_rows(spans, rows, capacity)
    parsed_context = int(context)
    parsed_start = int(start_position)
    if parsed_context <= 0 or parsed_context > min(int(capacity), 512):
        raise ValueError("dense-initial BLAS context must be within [1, 512]")
    if parsed_start < 0 or parsed_start + int(rows) != parsed_context:
        raise ValueError("dense-initial BLAS rows must end at the context boundary")
    if int(num_q_heads) <= 0:
        raise ValueError("num_q_heads must be positive")
    library = library or build_laguna_kv_attention(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_DENSE_INITIAL_CAUSAL_SOFTMAX_F32)
    fn.argtypes = (
        [ctypes.c_void_p] * 5
        + [ctypes.c_int64] * 4
        + [ctypes.c_float, ctypes.c_void_p]
    )
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(scores_ptr),
        ctypes.c_void_p(spans.live_counts.ptr),
        ctypes.c_void_p(spans.token_positions.ptr),
        ctypes.c_void_p(spans.evict_mask.ptr),
        ctypes.c_void_p(spans.row_positions.ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(parsed_context),
        ctypes.c_int64(num_q_heads),
        ctypes.c_int64(parsed_start),
        ctypes.c_float(scale),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def laguna_dense_initial_causal_softmax_wave_rows_f32_spans(
    scores_ptr: int,
    spans: KVLiveSpans,
    rows: int,
    context: int,
    num_q_heads: int,
    start_position: int,
    scale: float,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Apply causal softmax with one independent wave32 per score row."""

    if spans.spans_mode == "uniform":
        capacity = _check_global_spans(spans, _LAGUNA_KV_HEADS, _LAGUNA_HEAD_DIM)
    else:
        capacity = _check_swa_spans(spans, _LAGUNA_KV_HEADS, _LAGUNA_HEAD_DIM)
    _check_prefill_rows(spans, rows, capacity)
    parsed_context = int(context)
    parsed_start = int(start_position)
    if parsed_context <= 0 or parsed_context > int(capacity):
        raise ValueError("dense-initial BLAS context must be within capacity")
    if parsed_start < 0 or parsed_start + int(rows) != parsed_context:
        raise ValueError("dense-initial BLAS rows must end at the context boundary")
    if int(num_q_heads) <= 0:
        raise ValueError("num_q_heads must be positive")
    library = library or build_laguna_kv_attention(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(
        library,
        _SYMBOL_DENSE_INITIAL_CAUSAL_SOFTMAX_WAVE_ROWS_F32,
    )
    fn.argtypes = (
        [ctypes.c_void_p] * 5
        + [ctypes.c_int64] * 4
        + [ctypes.c_float, ctypes.c_void_p]
    )
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(scores_ptr),
        ctypes.c_void_p(spans.live_counts.ptr),
        ctypes.c_void_p(spans.token_positions.ptr),
        ctypes.c_void_p(spans.evict_mask.ptr),
        ctypes.c_void_p(spans.row_positions.ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(parsed_context),
        ctypes.c_int64(num_q_heads),
        ctypes.c_int64(parsed_start),
        ctypes.c_float(scale),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def laguna_dense_initial_causal_softmax_tile_wave_rows_f32_spans(
    scores_ptr: int,
    row_max_ptr: int,
    row_sum_ptr: int,
    merge_scales_ptr: int,
    spans: KVLiveSpans,
    rows: int,
    tile_start: int,
    tile_count: int,
    context: int,
    num_q_heads: int,
    start_position: int,
    scale: float,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Update online softmax state for one head-major score tile."""

    if spans.spans_mode == "uniform":
        capacity = _check_global_spans(
            spans,
            _LAGUNA_KV_HEADS,
            _LAGUNA_HEAD_DIM,
        )
    else:
        capacity = _check_swa_spans(
            spans,
            _LAGUNA_KV_HEADS,
            _LAGUNA_HEAD_DIM,
        )
    _check_prefill_rows(spans, rows, capacity)
    parsed_start = int(tile_start)
    parsed_count = int(tile_count)
    parsed_context = int(context)
    parsed_row_start = int(start_position)
    if (
        parsed_start < 0
        or parsed_count <= 0
        or parsed_start + parsed_count > parsed_context
        or parsed_context > int(capacity)
    ):
        raise ValueError("dense-initial BLAS score tile is out of range")
    if parsed_row_start < 0 or parsed_row_start + int(rows) != parsed_context:
        raise ValueError("dense-initial BLAS rows must end at the context boundary")
    if int(num_q_heads) <= 0:
        raise ValueError("num_q_heads must be positive")
    library = library or build_laguna_kv_attention(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(
        library,
        _SYMBOL_DENSE_INITIAL_CAUSAL_SOFTMAX_TILE_WAVE_ROWS_F32,
    )
    fn.argtypes = (
        [ctypes.c_void_p] * 8
        + [ctypes.c_int64] * 6
        + [ctypes.c_float, ctypes.c_void_p]
    )
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(scores_ptr),
        ctypes.c_void_p(row_max_ptr),
        ctypes.c_void_p(row_sum_ptr),
        ctypes.c_void_p(merge_scales_ptr),
        ctypes.c_void_p(spans.live_counts.ptr),
        ctypes.c_void_p(spans.token_positions.ptr),
        ctypes.c_void_p(spans.evict_mask.ptr),
        ctypes.c_void_p(spans.row_positions.ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(parsed_start),
        ctypes.c_int64(parsed_count),
        ctypes.c_int64(parsed_context),
        ctypes.c_int64(num_q_heads),
        ctypes.c_int64(parsed_row_start),
        ctypes.c_float(scale),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def laguna_dense_initial_attention_tile_merge_f32(
    accumulator_ptr: int,
    tile_output_ptr: int,
    final_output_ptr: int,
    row_sum_ptr: int,
    merge_scales_ptr: int,
    rows: int,
    num_q_heads: int,
    head_dim: int,
    *,
    first_tile: bool,
    final_tile: bool,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Merge one PV numerator tile and normalize the final head-major output."""

    if int(rows) <= 0 or int(rows) > 2_048:
        raise ValueError("dense-initial BLAS rows must be within [1, 2048]")
    _check_laguna_attention_shape(
        num_q_heads,
        _LAGUNA_KV_HEADS,
        head_dim,
    )
    library = library or build_laguna_kv_attention(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(
        library,
        _SYMBOL_DENSE_INITIAL_ATTENTION_TILE_MERGE_F32,
    )
    fn.argtypes = (
        [ctypes.c_void_p] * 5
        + [ctypes.c_int64] * 3
        + [ctypes.c_int] * 2
        + [ctypes.c_void_p]
    )
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(accumulator_ptr),
        ctypes.c_void_p(tile_output_ptr),
        ctypes.c_void_p(final_output_ptr),
        ctypes.c_void_p(row_sum_ptr),
        ctypes.c_void_p(merge_scales_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(num_q_heads),
        ctypes.c_int64(head_dim),
        ctypes.c_int(int(bool(first_tile))),
        ctypes.c_int(int(bool(final_tile))),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def laguna_swa_union_bf16_to_f32_spans(
    current_key_ptr: int,
    current_value_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    key_f32_ptr: int,
    value_f32_ptr: int,
    spans: KVLiveSpans,
    rows: int,
    num_kv_heads: int,
    head_dim: int,
    start_position: int,
    *,
    sliding_window: int = 512,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Gather the 511 historical plus 128 current rows for rolling SWA."""

    capacity = _check_swa_spans(spans, num_kv_heads, head_dim)
    _check_prefill_rows(spans, rows, capacity)
    if (
        int(rows) != 128
        or int(capacity) != 512
        or int(sliding_window) != 512
        or int(start_position) < int(sliding_window)
    ):
        raise ValueError("tensorized SWA requires M128 at rolling window 512")
    library = library or build_laguna_kv_attention(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_SWA_UNION_BF16_TO_F32)
    fn.argtypes = (
        [ctypes.c_void_p] * 11
        + [ctypes.c_int64] * 6
        + [ctypes.c_void_p]
    )
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(current_key_ptr),
        ctypes.c_void_p(current_value_ptr),
        ctypes.c_void_p(key_cache_ptr),
        ctypes.c_void_p(value_cache_ptr),
        ctypes.c_void_p(key_f32_ptr),
        ctypes.c_void_p(value_f32_ptr),
        ctypes.c_void_p(spans.base_offsets.ptr),
        ctypes.c_void_p(spans.live_counts.ptr),
        ctypes.c_void_p(spans.token_positions.ptr),
        ctypes.c_void_p(spans.evict_mask.ptr),
        ctypes.c_void_p(spans.row_positions.ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(capacity),
        ctypes.c_int64(sliding_window),
        ctypes.c_int64(num_kv_heads),
        ctypes.c_int64(head_dim),
        ctypes.c_int64(start_position),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def laguna_swa_union_softmax_wave_rows_f32(
    scores_ptr: int,
    rows: int,
    num_q_heads: int,
    scale: float,
    *,
    union_context: int = 639,
    sliding_window: int = 512,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Normalize a fixed 639-key rolling-SWA score union in place."""

    if (
        int(rows) != 128
        or int(num_q_heads) != 72
        or int(union_context) != 639
        or int(sliding_window) != 512
    ):
        raise ValueError("tensorized SWA softmax requires [72,128,639]")
    library = library or build_laguna_kv_attention(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_SWA_UNION_SOFTMAX_WAVE_ROWS_F32)
    fn.argtypes = [ctypes.c_void_p] + [ctypes.c_int64] * 4 + [
        ctypes.c_float,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(scores_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(union_context),
        ctypes.c_int64(sliding_window),
        ctypes.c_int64(num_q_heads),
        ctypes.c_float(scale),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def register_laguna_kv_attention_kernels(*, replace: bool = True) -> None:
    for variant, kernel in (
        ("global_f32_bf16_spans", laguna_global_head_rmsnorm_rope_write_kv_f32_spans),
        (
            "global_wave0_tree_f32_bf16_spans",
            laguna_global_head_rmsnorm_rope_write_kv_wave0_tree_f32_spans,
        ),
        ("swa_f32_bf16_spans", laguna_swa_head_rmsnorm_rope_write_kv_f32_spans),
    ):
        register(
            KernelKey(
                "hip_gfx1100",
                "head_rmsnorm+partial_rotary+kv_write",
                "laguna_f32_weight",
                variant,
            ),
            kernel,
            replace=replace,
        )

    for variant, kernel in (
        (
            "global_fixedk_bf16_f32_spans",
            laguna_global_f16_projection_head_kv_bf16_spans,
        ),
        (
            "swa_fixedk_bf16_f32_spans",
            laguna_swa_f16_projection_head_kv_bf16_spans,
        ),
    ):
        register(
            KernelKey(
                "hip_gfx1100",
                "attention_projection+head_rmsnorm+partial_rotary+kv_write",
                "fp16_weight+laguna_f32_weight",
                variant,
            ),
            kernel,
            replace=replace,
        )

    registrations = (
        (
            "laguna_kv_write",
            "global_f32_spans",
            laguna_global_write_kv_f32_spans,
        ),
        (
            "laguna_kv_write",
            "swa_f32_spans",
            laguna_swa_write_kv_f32_spans,
        ),
        (
            "laguna_kv_write",
            "global_f32_rows_spans",
            laguna_global_write_kv_rows_f32_spans,
        ),
        (
            "laguna_kv_write",
            "swa_f32_rows_spans",
            laguna_swa_write_kv_rows_f32_spans,
        ),
        (
            "laguna_attention_decode",
            "global_context_spans",
            laguna_global_attention_decode_bf16_spans,
        ),
        (
            "laguna_attention_decode",
            "global_context_single_page_spans",
            laguna_global_attention_decode_single_page_bf16_spans,
        ),
        (
            "laguna_attention_decode+attention_gate",
            "global_single_page_softplus_bf16_spans",
            laguna_global_attention_decode_single_page_softplus_gate_bf16_spans,
        ),
        (
            "laguna_attention_decode",
            "global_context_split_exact_spans",
            laguna_global_attention_decode_split_exact_bf16_spans,
        ),
        (
            "laguna_attention_decode",
            "global_context_split_exact_gated_spans",
            laguna_global_attention_decode_split_exact_gated_bf16_spans,
        ),
        (
            "laguna_attention_decode",
            "global_context_split_exact_gated_fixedshape_spans",
            (
                laguna_global_attention_decode_split_exact_gated_fixedshape_bf16_spans
            ),
        ),
        (
            "laguna_attention_decode",
            "global_context_fused_exact_gated_gqa1_fixedshape_spans",
            laguna_global_attention_decode_fused_exact_gated_gqa1_fixedshape_bf16_spans,
        ),
        (
            "laguna_attention_decode",
            "global_context_fused_exact_gated_gqa2_vstage64_fixedshape_spans",
            laguna_global_attention_decode_fused_exact_gated_gqa2_vstage64_fixedshape_bf16_spans,
        ),
        (
            "laguna_attention_decode",
            "global_context_fused_exact_gated_gqa2_vstage64_vec16_fixedshape_spans",
            laguna_global_attention_decode_fused_exact_gated_gqa2_vstage64_vec16_fixedshape_bf16_spans,
        ),
        (
            "laguna_attention_decode",
            "global_context_fused_exact_gated_gqa2_vstage64_vec16_direct_fixedshape_spans",
            laguna_global_attention_decode_fused_exact_gated_gqa2_vstage64_vec16_direct_fixedshape_bf16_spans,
        ),
        (
            "laguna_attention_decode",
            "global_context_fused_exact_gated_gqa2_vstage64_vec16_direct_assume_exp_fixedshape_spans",
            laguna_global_attention_decode_fused_exact_gated_gqa2_vstage64_vec16_direct_assume_exp_fixedshape_bf16_spans,
        ),
        (
            "laguna_attention_decode",
            "global_context_fused_exact_gated_gqa2_exp32_vstage64_vec16_direct_assume_exp_fixedshape_spans",
            laguna_global_attention_decode_fused_exact_gated_gqa2_exp32_vstage64_vec16_direct_assume_exp_fixedshape_bf16_spans,
        ),
        (
            "laguna_attention_decode",
            "global_context_fused_exact_gated_mixed32_exp32_vstage64_vec16_direct_assume_exp_fixedshape_spans",
            laguna_global_attention_decode_fused_exact_gated_mixed32_exp32_vstage64_vec16_direct_assume_exp_fixedshape_bf16_spans,
        ),
        (
            "laguna_attention_decode",
            "global_context_fused_exact_gated_mixed32_exp32_producer_max_vstage64_vec16_direct_assume_exp_fixedshape_spans",
            laguna_global_attention_decode_fused_exact_gated_mixed32_exp32_producer_max_vstage64_vec16_direct_assume_exp_fixedshape_bf16_spans,
        ),
        (
            "laguna_attention_decode",
            "global_context_fused_exact_gated_mixed32_exp32_producer_max_dpp_qk_vstage64_vec16_direct_assume_exp_fixedshape_spans",
            laguna_global_attention_decode_fused_exact_gated_mixed32_exp32_producer_max_dpp_qk_vstage64_vec16_direct_assume_exp_fixedshape_bf16_spans,
        ),
        (
            "laguna_attention_decode",
            "global_context_fused_exact_gated_mixed32_exp32_producer_max_dpp_qk_probability_vec4_vstage64_vec16_direct_assume_exp_fixedshape_spans",
            laguna_global_attention_decode_fused_exact_gated_mixed32_exp32_producer_max_dpp_qk_probability_vec4_vstage64_vec16_direct_assume_exp_fixedshape_bf16_spans,
        ),
        (
            "laguna_attention_decode",
            "global_context_fused_exact_gated_mixed32_exp32_producer_max_dpp_qk_probability_vec4_prenorm_vstage64_vec16_direct_assume_exp_fixedshape_spans",
            laguna_global_attention_decode_fused_exact_gated_mixed32_exp32_producer_max_dpp_qk_probability_vec4_prenorm_vstage64_vec16_direct_assume_exp_fixedshape_bf16_spans,
        ),
        (
            "laguna_attention_decode",
            "global_context_fused_exact_gated_mixed32_local512_exp32_producer_max_dpp_qk_probability_vec4_prenorm_vstage64_vec16_direct_assume_exp_fixedshape_spans",
            laguna_global_attention_decode_fused_exact_gated_mixed32_local512_exp32_producer_max_dpp_qk_probability_vec4_prenorm_vstage64_vec16_direct_assume_exp_fixedshape_bf16_spans,
        ),
        (
            "laguna_attention_decode",
            "global_context_fused_exact_gated_mixed40_local512_exp32_producer_max_dpp_qk_probability_vec4_prenorm_vstage64_vec16_direct_assume_exp_fixedshape_spans",
            laguna_global_attention_decode_fused_exact_gated_mixed40_local512_exp32_producer_max_dpp_qk_probability_vec4_prenorm_vstage64_vec16_direct_assume_exp_fixedshape_bf16_spans,
        ),
        (
            "laguna_attention_decode",
            "global_context_fused_exact_gated_mixed40_local512_exp32_producer_max_dpp_qk_dense_prefix_probability_vec4_prenorm_vstage64_vec16_direct_assume_exp_fixedshape_spans",
            laguna_global_attention_decode_fused_exact_gated_mixed40_local512_exp32_producer_max_dpp_qk_dense_prefix_probability_vec4_prenorm_vstage64_vec16_direct_assume_exp_fixedshape_bf16_spans,
        ),
        (
            "laguna_attention_decode",
            "global_context_fused_exact_gated_mixed40_local512_exp32_producer_max_dpp_qk_dense_prefix_idle_double_buffer_probability_vec4_prenorm_vstage64_vec16_direct_assume_exp_fixedshape_spans",
            laguna_global_attention_decode_fused_exact_gated_mixed40_local512_exp32_producer_max_dpp_qk_dense_prefix_idle_double_buffer_probability_vec4_prenorm_vstage64_vec16_direct_assume_exp_fixedshape_bf16_spans,
        ),
        (
            "laguna_attention_decode",
            "global_context_fused_exact_gated_mixed40_local1024_exp32_producer_max_dpp_qk_dense_prefix_idle_double_buffer_probability_vec4_prenorm_vstage64_vec16_direct_assume_exp_fixedshape_spans",
            laguna_global_attention_decode_fused_exact_gated_mixed40_local1024_exp32_producer_max_dpp_qk_dense_prefix_idle_double_buffer_probability_vec4_prenorm_vstage64_vec16_direct_assume_exp_fixedshape_bf16_spans,
        ),
        (
            "laguna_attention_decode",
            "global_context_wmma_qk_three_term_mixed32_exp32_producer_max_exact_pv_spans",
            laguna_global_attention_decode_wmma_qk_three_term_mixed32_exp32_producer_max_exact_pv_bf16_spans,
        ),
        (
            "laguna_attention_decode",
            "global_context_wmma_gqa6_k64_three_term_raw_numerator_spans",
            laguna_global_attention_decode_wmma_gqa6_k64_three_term_raw_numerator_bf16_spans,
        ),
        (
            "laguna_attention_decode",
            "swa_context_spans",
            laguna_swa_attention_decode_bf16_spans,
        ),
        (
            "laguna_attention_decode",
            "swa_context_token4_exact_spans",
            laguna_swa_attention_decode_token4_exact_bf16_spans,
        ),
        (
            "laguna_attention_decode",
            "swa_context_split_exact_spans",
            laguna_swa_attention_decode_split_exact_bf16_spans,
        ),
        (
            "laguna_attention_decode",
            "swa_context_split_exact_gated_spans",
            laguna_swa_attention_decode_split_exact_gated_bf16_spans,
        ),
        (
            "laguna_attention_decode",
            "swa_context_split_exact_gated_wave_local_spans",
            laguna_swa_attention_decode_split_exact_gated_wave_local_bf16_spans,
        ),
        (
            "laguna_attention_decode",
            "swa_context_split_exact_gated_gqa3_scores_spans",
            laguna_swa_attention_decode_split_exact_gated_gqa3_scores_bf16_spans,
        ),
        (
            "laguna_attention_decode",
            "swa_context_split_exact_gated_wave_local_dim2_spans",
            laguna_swa_attention_decode_split_exact_gated_wave_local_dim2_bf16_spans,
        ),
        (
            "laguna_attention_decode",
            "swa_context_split_tile16_exact_spans",
            laguna_swa_attention_decode_split_tile16_exact_bf16_spans,
        ),
        (
            "laguna_attention_decode",
            "swa_context_split_tile16_exact_gated_spans",
            laguna_swa_attention_decode_split_tile16_exact_gated_bf16_spans,
        ),
        (
            "laguna_attention_decode",
            "swa_context_split_tile16_exact_gated_wave_local_spans",
            laguna_swa_attention_decode_split_tile16_exact_gated_wave_local_bf16_spans,
        ),
        (
            "laguna_attention_decode",
            "swa_context_split_tile16_exact_gated_gqa3_scores_spans",
            laguna_swa_attention_decode_split_tile16_exact_gated_gqa3_scores_bf16_spans,
        ),
        (
            "laguna_attention_decode",
            "swa_context_split_tile16_exact_gated_gqa3_scores_fixed512_spans",
            laguna_swa_attention_decode_split_tile16_exact_gated_gqa3_scores_fixed512_bf16_spans,
        ),
        (
            "laguna_attention_decode",
            "swa_context_fused_exact_gated_gqa2_fixed512_spans",
            laguna_swa_attention_decode_fused_exact_gated_gqa2_fixed512_bf16_spans,
        ),
        (
            "laguna_attention_decode",
            "swa_context_fused_exact_gated_gqa3_local384_fixed512_spans",
            laguna_swa_attention_decode_fused_exact_gated_gqa3_local384_fixed512_bf16_spans,
        ),
        (
            "laguna_attention_decode",
            "swa_context_fused_exact_gated_gqa3_vstage64_fixed512_spans",
            laguna_swa_attention_decode_fused_exact_gated_gqa3_vstage64_fixed512_bf16_spans,
        ),
        (
            "laguna_attention_decode",
            "swa_context_fused_exact_gated_gqa3_vstage64_vec16_fixed512_spans",
            laguna_swa_attention_decode_fused_exact_gated_gqa3_vstage64_vec16_fixed512_bf16_spans,
        ),
        (
            "laguna_attention_decode",
            "swa_context_fused_exact_gated_gqa3_vstage64_vec16_direct_fixed512_spans",
            laguna_swa_attention_decode_fused_exact_gated_gqa3_vstage64_vec16_direct_fixed512_bf16_spans,
        ),
        (
            "laguna_attention_decode",
            "swa_context_fused_exact_gated_gqa3_vstage64_vec16_direct_assume_exp_fixed512_spans",
            laguna_swa_attention_decode_fused_exact_gated_gqa3_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans,
        ),
        (
            "laguna_attention_decode",
            "swa_context_fused_exact_gated_mixed32_vstage64_vec16_direct_assume_exp_fixed512_spans",
            laguna_swa_attention_decode_fused_exact_gated_mixed32_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans,
        ),
        (
            "laguna_attention_decode",
            "swa_context_fused_exact_gated_mixed32_exp4_vstage64_vec16_direct_assume_exp_fixed512_spans",
            laguna_swa_attention_decode_fused_exact_gated_mixed32_exp4_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans,
        ),
        (
            "laguna_attention_decode",
            "swa_context_fused_exact_gated_mixed32_exp8_vstage64_vec16_direct_assume_exp_fixed512_spans",
            laguna_swa_attention_decode_fused_exact_gated_mixed32_exp8_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans,
        ),
        (
            "laguna_attention_decode",
            "swa_context_fused_exact_gated_mixed32_exp16_vstage64_vec16_direct_assume_exp_fixed512_spans",
            laguna_swa_attention_decode_fused_exact_gated_mixed32_exp16_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans,
        ),
        (
            "laguna_attention_decode",
            "swa_context_fused_exact_gated_mixed32_exp32_vstage64_vec16_direct_assume_exp_fixed512_spans",
            laguna_swa_attention_decode_fused_exact_gated_mixed32_exp32_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans,
        ),
        (
            "laguna_attention_decode",
            "swa_context_fused_exact_gated_mixed32_exp32_producer_max_vstage64_vec16_direct_assume_exp_fixed512_spans",
            laguna_swa_attention_decode_fused_exact_gated_mixed32_exp32_producer_max_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans,
        ),
        (
            "laguna_attention_decode",
            "swa_context_fused_exact_gated_mixed32_exp32_producer_max_gate_vstage64_vec16_direct_assume_exp_fixed512_spans",
            laguna_swa_attention_decode_fused_exact_gated_mixed32_exp32_producer_max_gate_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans,
        ),
        (
            "laguna_attention_decode",
            "swa_context_fused_exact_gated_mixed32_exp32_producer_max_gate_stage_pcache_vstage64_vec16_direct_assume_exp_fixed512_spans",
            laguna_swa_attention_decode_fused_exact_gated_mixed32_exp32_producer_max_gate_stage_pcache_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans,
        ),
        (
            "laguna_attention_decode",
            "swa_context_fused_exact_gated_mixed32_exp32_producer_max_gate_stage_pcache_idle_producer_vstage64_vec16_direct_assume_exp_fixed512_spans",
            laguna_swa_attention_decode_fused_exact_gated_mixed32_exp32_producer_max_gate_stage_pcache_idle_producer_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans,
        ),
        (
            "laguna_attention_decode",
            "swa_context_fused_exact_gated_mixed32_exp32_producer_max_gate_stage_pcache_vec4_denom_vstage64_vec16_direct_assume_exp_fixed512_spans",
            laguna_swa_attention_decode_fused_exact_gated_mixed32_exp32_producer_max_gate_stage_pcache_vec4_denom_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans,
        ),
        (
            "laguna_attention_decode",
            "swa_context_fused_exact_gated_mixed32_exp32_producer_max_gate_stage_pcache_vec4_denom_probability_vstage64_vec16_direct_assume_exp_fixed512_spans",
            laguna_swa_attention_decode_fused_exact_gated_mixed32_exp32_producer_max_gate_stage_pcache_vec4_denom_probability_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans,
        ),
        (
            "laguna_attention_decode",
            "swa_context_fused_exact_gated_mixed32_exp32_producer_max_gate_stage_pcache_idle_vec4_denom_probability_vstage64_vec16_direct_assume_exp_fixed512_spans",
            laguna_swa_attention_decode_fused_exact_gated_mixed32_exp32_producer_max_gate_stage_pcache_idle_vec4_denom_probability_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans,
        ),
        (
            "laguna_attention_decode",
            "swa_context_fused_exact_gated_mixed40_exp32_producer_max_gate_stage_pcache_idle_vec4_denom_probability_vstage64_vec16_direct_assume_exp_fixed512_spans",
            laguna_swa_attention_decode_fused_exact_gated_mixed40_exp32_producer_max_gate_stage_pcache_idle_vec4_denom_probability_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans,
        ),
        (
            "laguna_attention_decode",
            "swa_context_fused_exact_gated_mixed40_exp32_producer_max_gate_stage_pcache_tail_producer_idle_vec4_denom_probability_vstage64_vec16_direct_assume_exp_fixed512_spans",
            laguna_swa_attention_decode_fused_exact_gated_mixed40_exp32_producer_max_gate_stage_pcache_tail_producer_idle_vec4_denom_probability_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans,
        ),
        (
            "laguna_attention_decode",
            "swa_context_fused_exact_gated_mixed40_local512_exp32_producer_max_gate_stage_pcache_tail_producer_idle_vec4_denom_probability_vstage64_vec16_direct_assume_exp_fixed512_spans",
            laguna_swa_attention_decode_fused_exact_gated_mixed40_local512_exp32_producer_max_gate_stage_pcache_tail_producer_idle_vec4_denom_probability_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans,
        ),
        (
            "laguna_attention_decode",
            "swa_context_fused_exact_gated_mixed40_local512_exp32_producer_max_gate_stage_pcache_tail_producer_value_tail_idle_vec4_denom_probability_vstage64_vec16_direct_assume_exp_fixed512_spans",
            laguna_swa_attention_decode_fused_exact_gated_mixed40_local512_exp32_producer_max_gate_stage_pcache_tail_producer_value_tail_idle_vec4_denom_probability_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans,
        ),
        (
            "laguna_attention_decode",
            "swa_context_fused_exact_gated_mixed40_local512_exp32_producer_max_gate_stage_pcache_tail_producer_value_tail_idle_vec4_denom_probability_vstage128_vec16_direct_assume_exp_fixed512_spans",
            laguna_swa_attention_decode_fused_exact_gated_mixed40_local512_exp32_producer_max_gate_stage_pcache_tail_producer_value_tail_idle_vec4_denom_probability_vstage128_vec16_direct_assume_exp_fixed512_bf16_spans,
        ),
        (
            "laguna_attention_decode",
            "swa_context_fused_exact_gated_mixed40_local512_exp32_producer_max_gate_stage_pcache_dual_tail_producer_value_tail_idle_vec4_denom_probability_vstage128_vec16_direct_assume_exp_fixed512_spans",
            laguna_swa_attention_decode_fused_exact_gated_mixed40_local512_exp32_producer_max_gate_stage_pcache_dual_tail_producer_value_tail_idle_vec4_denom_probability_vstage128_vec16_direct_assume_exp_fixed512_bf16_spans,
        ),
        (
            "laguna_attention_decode",
            "swa_context_fused_exact_gated_mixed40_local512_exp32_producer_max_gate_stage_pcache_output_sharded_probability_allwave_value_idle_vec4_denom_probability_vstage128_vec16_direct_assume_exp_fixed512_spans",
            laguna_swa_attention_decode_fused_exact_gated_mixed40_local512_exp32_producer_max_gate_stage_pcache_output_sharded_probability_allwave_value_idle_vec4_denom_probability_vstage128_vec16_direct_assume_exp_fixed512_bf16_spans,
        ),
        (
            "laguna_attention_decode",
            "swa_context_fused_exact_gated_mixed40_local512_exp32_producer_max_gate_stage_pcache_output_sharded_probability_dpp_qk_allwave_value_idle_vec4_denom_probability_vstage128_vec16_direct_assume_exp_fixed512_spans",
            laguna_swa_attention_decode_fused_exact_gated_mixed40_local512_exp32_producer_max_gate_stage_pcache_output_sharded_probability_dpp_qk_allwave_value_idle_vec4_denom_probability_vstage128_vec16_direct_assume_exp_fixed512_bf16_spans,
        ),
        (
            "laguna_attention_decode",
            "swa_context_fused_exact_gated_mixed40_local512_exp32_producer_max_gate_stage_pcache_output_sharded_probability_dpp_qk_dense_ring_allwave_value_idle_vec4_denom_probability_vstage128_vec16_direct_assume_exp_fixed512_spans",
            laguna_swa_attention_decode_fused_exact_gated_mixed40_local512_exp32_producer_max_gate_stage_pcache_output_sharded_probability_dpp_qk_dense_ring_allwave_value_idle_vec4_denom_probability_vstage128_vec16_direct_assume_exp_fixed512_bf16_spans,
        ),
        (
            "laguna_attention_decode",
            "swa_context_fused_exact_gated_mixed40_local1024_exp32_producer_max_gate_stage_pcache_output_sharded_probability_dpp_qk_dense_ring_allwave_value_idle_vec4_denom_probability_vstage128_vec16_direct_assume_exp_fixed512_spans",
            laguna_swa_attention_decode_fused_exact_gated_mixed40_local1024_exp32_producer_max_gate_stage_pcache_output_sharded_probability_dpp_qk_dense_ring_allwave_value_idle_vec4_denom_probability_vstage128_vec16_direct_assume_exp_fixed512_bf16_spans,
        ),
        (
            "laguna_attention_decode",
            "swa_context_fused_exact_gated_mixed40_exp32_producer_max_gate_stage_pcache_tail_producer_value_tail_idle_vec4_denom_probability_vstage64_vec16_direct_assume_exp_fixed512_spans",
            laguna_swa_attention_decode_fused_exact_gated_mixed40_exp32_producer_max_gate_stage_pcache_tail_producer_value_tail_idle_vec4_denom_probability_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans,
        ),
        (
            "laguna_attention_decode",
            "swa_context_fused_exact_gated_mixed40_exp32_producer_max_gate_stage_pcache_tail_producer_denom_prefetch4_idle_vec4_probability_vstage64_vec16_direct_assume_exp_fixed512_spans",
            laguna_swa_attention_decode_fused_exact_gated_mixed40_exp32_producer_max_gate_stage_pcache_tail_producer_denom_prefetch4_idle_vec4_probability_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans,
        ),
        (
            "laguna_attention_decode",
            "swa_context_fused_exact_gated_mixed32_exp32_producer_max_gate_stage_pcache_dpp_qk_vstage64_vec16_direct_assume_exp_fixed512_spans",
            laguna_swa_attention_decode_fused_exact_gated_mixed32_exp32_producer_max_gate_stage_pcache_dpp_qk_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans,
        ),
        (
            "laguna_attention_decode",
            "swa_context_fused_exact_gated_mixed32_exp32_producer_max_gate_dpp_qk_vstage64_vec16_direct_assume_exp_fixed512_spans",
            laguna_swa_attention_decode_fused_exact_gated_mixed32_exp32_producer_max_gate_dpp_qk_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans,
        ),
        (
            "laguna_attention_decode",
            "swa_context_fused_exact_gated_mixed32_exp32_producer_max_gate_gated_only_vstage64_vec16_direct_assume_exp_fixed512_spans",
            laguna_swa_attention_decode_fused_exact_gated_mixed32_exp32_producer_max_gate_gated_only_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans,
        ),
        (
            "laguna_attention_decode",
            "swa_context_split_tile16_exact_gated_wave_local_dim2_spans",
            laguna_swa_attention_decode_split_tile16_exact_gated_wave_local_dim2_bf16_spans,
        ),
        (
            "laguna_attention_prefill",
            "global_context_rows_spans",
            laguna_global_attention_prefill_bf16_spans,
        ),
        (
            "laguna_attention_prefill",
            "global_context_rows_qrow2_online_spans",
            laguna_global_attention_prefill_qrow2_online_bf16_spans,
        ),
        (
            "laguna_attention_prefill",
            "global_context_rows_qrow4_online_spans",
            laguna_global_attention_prefill_qrow4_online_bf16_spans,
        ),
        (
            "laguna_attention_prefill",
            "global_context_rows_qrow4_cached_meta_online_spans",
            laguna_global_attention_prefill_qrow4_cached_meta_online_bf16_spans,
        ),
        (
            "laguna_attention_prefill",
            "global_context_rows_qrow4_dense_initial_online_spans",
            laguna_global_attention_prefill_qrow4_dense_initial_online_bf16_spans,
        ),
        (
            "laguna_attention_prefill",
            "global_context_rows_qrow6_cached_meta_online_spans",
            laguna_global_attention_prefill_qrow6_cached_meta_online_bf16_spans,
        ),
        (
            "laguna_attention_prefill",
            "global_context_rows_qrow6_dense_initial_online_spans",
            laguna_global_attention_prefill_qrow6_dense_initial_online_bf16_spans,
        ),
        (
            "laguna_attention_prefill",
            "global_context_rows_qrow4_cached_online_spans",
            laguna_global_attention_prefill_qrow4_cached_online_bf16_spans,
        ),
        (
            "laguna_attention_prefill",
            "global_context_rows_qrow4_m128_online_spans",
            laguna_global_attention_prefill_qrow4_m128_online_bf16_spans,
        ),
        (
            "laguna_attention_prefill",
            "swa_context_rows_spans",
            laguna_swa_attention_prefill_bf16_spans,
        ),
        (
            "laguna_attention_prefill",
            "swa_context_rows_wave32_exact_spans",
            laguna_swa_attention_prefill_wave32_exact_bf16_spans,
        ),
        (
            "laguna_attention_prefill",
            "swa_context_rows_qrow2_exact_spans",
            laguna_swa_attention_prefill_qrow2_exact_bf16_spans,
        ),
        (
            "laguna_attention_prefill",
            "swa_context_rows_qrow2_m128_c128_exact_spans",
            laguna_swa_attention_prefill_qrow2_m128_c128_exact_bf16_spans,
        ),
        (
            "laguna_attention_prefill",
            "swa_context_rows_qrow2_online_spans",
            laguna_swa_attention_prefill_qrow2_online_bf16_spans,
        ),
        (
            "laguna_attention_prefill",
            "swa_context_rows_qrow4_online_spans",
            laguna_swa_attention_prefill_qrow4_online_bf16_spans,
        ),
        (
            "laguna_attention_prefill",
            "swa_context_rows_qrow4_sourcequal_online_spans",
            laguna_swa_attention_prefill_qrow4_sourcequal_online_bf16_spans,
        ),
        (
            "laguna_attention_prefill",
            "swa_context_rows_qrow4_cached_meta_online_spans",
            laguna_swa_attention_prefill_qrow4_cached_meta_online_bf16_spans,
        ),
        (
            "laguna_attention_prefill",
            "swa_context_rows_qrow4_dense_initial_online_spans",
            laguna_swa_attention_prefill_qrow4_dense_initial_online_bf16_spans,
        ),
        (
            "laguna_attention_prefill",
            "swa_context_rows_qrow4_cached_online_spans",
            laguna_swa_attention_prefill_qrow4_cached_online_bf16_spans,
        ),
        (
            "laguna_attention_prefill",
            "swa_context_rows_qrow4_m128_online_spans",
            laguna_swa_attention_prefill_qrow4_m128_online_bf16_spans,
        ),
    )
    for layer, variant, kernel in registrations:
        register(
            KernelKey("hip_gfx1100", layer, "bf16", variant),
            kernel,
            replace=replace,
        )


def _check_global_spans(
    spans: KVLiveSpans,
    num_kv_heads: int,
    head_dim: int,
) -> int:
    if spans.spans_mode != "uniform":
        raise ValueError("Laguna global KV requires uniform spans")
    if spans.storage_dtype != DType.BF16:
        raise ValueError("Laguna global KV requires bf16 storage")
    if spans.token_positions is None or spans.evict_mask is None:
        raise ValueError("Laguna global KV requires token_positions and evict_mask")
    if spans.row_positions is None:
        raise ValueError("Laguna global KV requires absolute row_positions")
    capacity = spans.token_positions.numel
    if spans.evict_mask.numel != capacity or spans.max_live_count != capacity:
        raise ValueError("Laguna global span metadata must match its logical capacity")
    if (_GLOBAL_BLOCK_SIZE * spans.base_offsets.numel) < capacity:
        raise ValueError("Laguna global block table is too short for its logical capacity")
    _check_laguna_kv_shape(num_kv_heads, head_dim)
    return capacity


def _check_swa_spans(
    spans: KVLiveSpans,
    num_kv_heads: int,
    head_dim: int,
) -> int:
    if spans.spans_mode != "sliding_ring":
        raise ValueError("Laguna SWA requires sliding_ring spans")
    if spans.storage_dtype != DType.BF16:
        raise ValueError("Laguna SWA requires bf16 storage")
    if spans.token_positions is None or spans.evict_mask is None:
        raise ValueError("Laguna SWA requires token_positions and evict_mask")
    if spans.row_positions is None:
        raise ValueError("Laguna SWA requires absolute row_positions")
    _check_laguna_kv_shape(num_kv_heads, head_dim)
    return spans.base_offsets.numel


def _check_prefill_rows(spans: KVLiveSpans, rows: int, capacity: int) -> None:
    parsed = int(rows)
    if parsed <= 0 or parsed > int(capacity):
        raise ValueError("rows must be within [1, span capacity]")
    assert spans.row_positions is not None
    if spans.row_positions.numel != 1:
        raise ValueError("row_positions must contain the consecutive chunk start scalar")


def _check_split_scan_slots(scan_slots: int, capacity: int) -> int:
    parsed = int(scan_slots)
    if parsed <= 0 or parsed > int(capacity):
        raise ValueError("scan_slots must be within [1, span capacity]")
    return parsed


def _check_nonzero_device_pointers(*pointers: tuple[str, int]) -> None:
    for name, pointer in pointers:
        if int(pointer) == 0:
            raise ValueError(f"{name} must be non-zero")


def _check_laguna_kv_shape(num_kv_heads: int, head_dim: int) -> None:
    if int(num_kv_heads) != _LAGUNA_KV_HEADS:
        raise ValueError(f"num_kv_heads must be {_LAGUNA_KV_HEADS} for Laguna")
    if int(head_dim) != _LAGUNA_HEAD_DIM:
        raise ValueError(f"head_dim must be {_LAGUNA_HEAD_DIM} for Laguna")


def _check_laguna_attention_shape(
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> None:
    _check_laguna_kv_shape(num_kv_heads, head_dim)
    if int(num_q_heads) % int(num_kv_heads):
        raise ValueError("num_q_heads must be divisible by num_kv_heads")
    if int(num_q_heads) not in {48, 72}:
        raise ValueError("num_q_heads must be a Laguna production width (48 or 72)")


def _check_head_kv_rope_shape(
    rotary_dim: int,
    head_dim: int,
    max_positions: int,
) -> None:
    parsed_rotary = int(rotary_dim)
    if parsed_rotary <= 0 or parsed_rotary > int(head_dim):
        raise ValueError("rotary_dim must be within [1, head_dim]")
    if parsed_rotary % 2:
        raise ValueError("rotary_dim must be even")
    if int(max_positions) <= 0:
        raise ValueError("max_positions must be positive")


def _check_launch(runtime: HipRuntime, err: int) -> None:
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


register_laguna_kv_attention_kernels()

__all__ = [
    "build_laguna_kv_attention",
    "laguna_dense_initial_cache_bf16_to_f32_spans",
    "laguna_dense_initial_cache_block_bf16_to_f32_spans",
    "laguna_dense_initial_contiguous_cache_block_bf16_to_f32_spans",
    "laguna_dense_initial_attention_tile_merge_f32",
    "laguna_dense_initial_causal_softmax_f32_spans",
    "laguna_dense_initial_causal_softmax_tile_wave_rows_f32_spans",
    "laguna_dense_initial_causal_softmax_wave_rows_f32_spans",
    "laguna_dense_initial_query_head_transpose_f32",
    "laguna_global_attention_decode_bf16_spans",
    "laguna_global_attention_decode_single_page_bf16_spans",
    "laguna_global_attention_decode_single_page_softplus_gate_bf16_spans",
    "laguna_global_attention_decode_split_exact_bf16_spans",
    "laguna_global_attention_decode_split_exact_gated_bf16_spans",
    "laguna_global_attention_decode_split_exact_gated_fixedshape_bf16_spans",
    "laguna_global_attention_decode_fused_exact_gated_gqa1_fixedshape_bf16_spans",
    "laguna_global_attention_decode_fused_exact_gated_gqa2_vstage64_fixedshape_bf16_spans",
    "laguna_global_attention_decode_fused_exact_gated_gqa2_vstage64_vec16_fixedshape_bf16_spans",
    "laguna_global_attention_decode_fused_exact_gated_gqa2_vstage64_vec16_direct_fixedshape_bf16_spans",
    "laguna_global_attention_decode_fused_exact_gated_gqa2_vstage64_vec16_direct_assume_exp_fixedshape_bf16_spans",
    "laguna_global_attention_decode_fused_exact_gated_gqa2_exp32_vstage64_vec16_direct_assume_exp_fixedshape_bf16_spans",
    "laguna_global_attention_decode_fused_exact_gated_mixed32_exp32_vstage64_vec16_direct_assume_exp_fixedshape_bf16_spans",
    "laguna_global_attention_decode_fused_exact_gated_mixed32_exp32_producer_max_vstage64_vec16_direct_assume_exp_fixedshape_bf16_spans",
    "laguna_global_attention_decode_fused_exact_gated_mixed32_exp32_producer_max_dpp_qk_vstage64_vec16_direct_assume_exp_fixedshape_bf16_spans",
    "laguna_global_attention_decode_fused_exact_gated_mixed32_exp32_producer_max_dpp_qk_probability_vec4_vstage64_vec16_direct_assume_exp_fixedshape_bf16_spans",
    "laguna_global_attention_decode_fused_exact_gated_mixed32_exp32_producer_max_dpp_qk_probability_vec4_prenorm_vstage64_vec16_direct_assume_exp_fixedshape_bf16_spans",
    "laguna_global_attention_decode_fused_exact_gated_mixed32_local512_exp32_producer_max_dpp_qk_probability_vec4_prenorm_vstage64_vec16_direct_assume_exp_fixedshape_bf16_spans",
    "laguna_global_attention_decode_fused_exact_gated_mixed40_local512_exp32_producer_max_dpp_qk_probability_vec4_prenorm_vstage64_vec16_direct_assume_exp_fixedshape_bf16_spans",
    "laguna_global_attention_decode_fused_exact_gated_mixed40_local512_exp32_producer_max_dpp_qk_dense_prefix_probability_vec4_prenorm_vstage64_vec16_direct_assume_exp_fixedshape_bf16_spans",
    "laguna_global_attention_decode_fused_exact_gated_mixed40_local512_exp32_producer_max_dpp_qk_dense_prefix_idle_double_buffer_probability_vec4_prenorm_vstage64_vec16_direct_assume_exp_fixedshape_bf16_spans",
    "laguna_global_attention_decode_fused_exact_gated_mixed40_local1024_exp32_producer_max_dpp_qk_dense_prefix_idle_double_buffer_probability_vec4_prenorm_vstage64_vec16_direct_assume_exp_fixedshape_bf16_spans",
    "laguna_global_attention_decode_wmma_qk_three_term_mixed32_exp32_producer_max_exact_pv_bf16_spans",
    "laguna_global_attention_decode_wmma_gqa6_k64_three_term_raw_numerator_bf16_spans",
    "laguna_global_attention_prefill_bf16_spans",
    "laguna_global_attention_prefill_qrow2_online_bf16_spans",
    "laguna_global_attention_prefill_qrow4_cached_meta_online_bf16_spans",
    "laguna_global_attention_prefill_qrow4_cached_online_bf16_spans",
    "laguna_global_attention_prefill_qrow4_dense_initial_online_bf16_spans",
    "laguna_global_attention_prefill_qrow4_online_bf16_spans",
    "laguna_global_attention_prefill_qrow4_m128_online_bf16_spans",
    "laguna_global_attention_prefill_qrow6_cached_meta_online_bf16_spans",
    "laguna_global_attention_prefill_qrow6_dense_initial_online_bf16_spans",
    "laguna_global_f16_projection_head_kv_bf16_spans",
    "laguna_global_head_rmsnorm_rope_write_kv_f32_spans",
    "laguna_global_head_rmsnorm_rope_write_kv_wave0_tree_f32_spans",
    "laguna_global_write_kv_f32_spans",
    "laguna_global_write_kv_rows_f32_spans",
    "laguna_swa_attention_decode_bf16_spans",
    "laguna_swa_attention_decode_fused_exact_gated_gqa2_fixed512_bf16_spans",
    "laguna_swa_attention_decode_fused_exact_gated_gqa3_local384_fixed512_bf16_spans",
    "laguna_swa_attention_decode_fused_exact_gated_gqa3_vstage64_fixed512_bf16_spans",
    "laguna_swa_attention_decode_fused_exact_gated_gqa3_vstage64_vec16_fixed512_bf16_spans",
    "laguna_swa_attention_decode_fused_exact_gated_gqa3_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans",
    "laguna_swa_attention_decode_fused_exact_gated_gqa3_vstage64_vec16_direct_fixed512_bf16_spans",
    "laguna_swa_attention_decode_fused_exact_gated_mixed32_exp4_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans",
    "laguna_swa_attention_decode_fused_exact_gated_mixed32_exp8_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans",
    "laguna_swa_attention_decode_fused_exact_gated_mixed32_exp16_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans",
    "laguna_swa_attention_decode_fused_exact_gated_mixed32_exp32_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans",
    "laguna_swa_attention_decode_fused_exact_gated_mixed32_exp32_producer_max_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans",
    "laguna_swa_attention_decode_fused_exact_gated_mixed32_exp32_producer_max_gate_dpp_qk_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans",
    "laguna_swa_attention_decode_fused_exact_gated_mixed32_exp32_producer_max_gate_stage_pcache_dpp_qk_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans",
    "laguna_swa_attention_decode_fused_exact_gated_mixed32_exp32_producer_max_gate_stage_pcache_idle_producer_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans",
    "laguna_swa_attention_decode_fused_exact_gated_mixed32_exp32_producer_max_gate_stage_pcache_idle_vec4_denom_probability_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans",
    "laguna_swa_attention_decode_fused_exact_gated_mixed40_exp32_producer_max_gate_stage_pcache_idle_vec4_denom_probability_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans",
    "laguna_swa_attention_decode_fused_exact_gated_mixed40_exp32_producer_max_gate_stage_pcache_tail_producer_denom_prefetch4_idle_vec4_probability_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans",
    "laguna_swa_attention_decode_fused_exact_gated_mixed40_exp32_producer_max_gate_stage_pcache_tail_producer_idle_vec4_denom_probability_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans",
    "laguna_swa_attention_decode_fused_exact_gated_mixed40_local512_exp32_producer_max_gate_stage_pcache_tail_producer_idle_vec4_denom_probability_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans",
    "laguna_swa_attention_decode_fused_exact_gated_mixed40_local512_exp32_producer_max_gate_stage_pcache_tail_producer_value_tail_idle_vec4_denom_probability_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans",
    "laguna_swa_attention_decode_fused_exact_gated_mixed40_local512_exp32_producer_max_gate_stage_pcache_tail_producer_value_tail_idle_vec4_denom_probability_vstage128_vec16_direct_assume_exp_fixed512_bf16_spans",
    "laguna_swa_attention_decode_fused_exact_gated_mixed40_local512_exp32_producer_max_gate_stage_pcache_dual_tail_producer_value_tail_idle_vec4_denom_probability_vstage128_vec16_direct_assume_exp_fixed512_bf16_spans",
    "laguna_swa_attention_decode_fused_exact_gated_mixed40_local512_exp32_producer_max_gate_stage_pcache_output_sharded_probability_allwave_value_idle_vec4_denom_probability_vstage128_vec16_direct_assume_exp_fixed512_bf16_spans",
    "laguna_swa_attention_decode_fused_exact_gated_mixed40_local512_exp32_producer_max_gate_stage_pcache_output_sharded_probability_dpp_qk_allwave_value_idle_vec4_denom_probability_vstage128_vec16_direct_assume_exp_fixed512_bf16_spans",
    "laguna_swa_attention_decode_fused_exact_gated_mixed40_local512_exp32_producer_max_gate_stage_pcache_output_sharded_probability_dpp_qk_dense_ring_allwave_value_idle_vec4_denom_probability_vstage128_vec16_direct_assume_exp_fixed512_bf16_spans",
    "laguna_swa_attention_decode_fused_exact_gated_mixed40_local1024_exp32_producer_max_gate_stage_pcache_output_sharded_probability_dpp_qk_dense_ring_allwave_value_idle_vec4_denom_probability_vstage128_vec16_direct_assume_exp_fixed512_bf16_spans",
    "laguna_swa_attention_decode_fused_exact_gated_mixed40_exp32_producer_max_gate_stage_pcache_tail_producer_value_tail_idle_vec4_denom_probability_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans",
    "laguna_swa_attention_decode_fused_exact_gated_mixed32_exp32_producer_max_gate_stage_pcache_vec4_denom_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans",
    "laguna_swa_attention_decode_fused_exact_gated_mixed32_exp32_producer_max_gate_stage_pcache_vec4_denom_probability_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans",
    "laguna_swa_attention_decode_fused_exact_gated_mixed32_exp32_producer_max_gate_stage_pcache_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans",
    "laguna_swa_attention_decode_fused_exact_gated_mixed32_exp32_producer_max_gate_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans",
    "laguna_swa_attention_decode_fused_exact_gated_mixed32_exp32_producer_max_gate_gated_only_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans",
    "laguna_swa_attention_decode_fused_exact_gated_mixed32_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans",
    "laguna_swa_attention_decode_token4_exact_bf16_spans",
    "laguna_swa_attention_decode_split_exact_bf16_spans",
    "laguna_swa_attention_decode_split_exact_gated_bf16_spans",
    "laguna_swa_attention_decode_split_exact_gated_gqa3_scores_bf16_spans",
    "laguna_swa_attention_decode_split_exact_gated_wave_local_bf16_spans",
    "laguna_swa_attention_decode_split_exact_gated_wave_local_dim2_bf16_spans",
    "laguna_swa_attention_decode_split_tile16_exact_bf16_spans",
    "laguna_swa_attention_decode_split_tile16_exact_gated_bf16_spans",
    "laguna_swa_attention_decode_split_tile16_exact_gated_gqa3_scores_bf16_spans",
    "laguna_swa_attention_decode_split_tile16_exact_gated_gqa3_scores_fixed512_bf16_spans",
    "laguna_swa_attention_decode_split_tile16_exact_gated_wave_local_bf16_spans",
    "laguna_swa_attention_decode_split_tile16_exact_gated_wave_local_dim2_bf16_spans",
    "laguna_swa_attention_prefill_bf16_spans",
    "laguna_swa_attention_prefill_qrow2_m128_c128_exact_bf16_spans",
    "laguna_swa_attention_prefill_qrow2_exact_bf16_spans",
    "laguna_swa_attention_prefill_qrow2_online_bf16_spans",
    "laguna_swa_attention_prefill_qrow4_cached_meta_online_bf16_spans",
    "laguna_swa_attention_prefill_qrow4_dense_initial_online_bf16_spans",
    "laguna_swa_attention_prefill_qrow4_cached_online_bf16_spans",
    "laguna_swa_attention_prefill_qrow4_online_bf16_spans",
    "laguna_swa_attention_prefill_qrow4_sourcequal_online_bf16_spans",
    "laguna_swa_attention_prefill_qrow4_m128_online_bf16_spans",
    "laguna_swa_attention_prefill_wave32_exact_bf16_spans",
    "laguna_swa_union_bf16_to_f32_spans",
    "laguna_swa_union_softmax_wave_rows_f32",
    "laguna_swa_f16_projection_head_kv_bf16_spans",
    "laguna_swa_head_rmsnorm_rope_write_kv_f32_spans",
    "laguna_swa_write_kv_f32_spans",
    "laguna_swa_write_kv_rows_f32_spans",
    "plan_laguna_kv_attention_build",
    "register_laguna_kv_attention_kernels",
]
