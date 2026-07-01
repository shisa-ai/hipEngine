"""Qwen3.5 GGUF runtime bring-up probes."""

from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.hip import HipMemcpyKind, HipRuntime, get_hip_runtime
from hipengine.core.memory import DeviceBuffer, copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc
from hipengine.core.tensor import Tensor
from hipengine.runtime.moe_graph import MoeGraphCache
from hipengine.kernels.hip_gfx1100.attention import (
    aotriton_attn_fwd_compact_varlen,
    aotriton_attn_fwd_v3_compact_varlen,
    build_aotriton_wrap,
)
from hipengine.kernels.hip_gfx1100.attention.aotriton_wrap import (
    tensor1 as aotriton_tensor1,
    tensor2 as aotriton_tensor2,
    tensor4 as aotriton_tensor4,
)
from hipengine.kernels.hip_gfx1100.attention.paged_attn_decode import (
    build_qwen35_paged_attn_decode,
    qwen35_full_attn_gate_mul_bf16,
    qwen35_full_attn_gate_mul_bf16_to_bf16,
    qwen35_paged_full_attn_decode_context_bf16_batch_c1_exact_spans,
    qwen35_paged_full_attn_decode_context_bf16_spans,
    qwen35_paged_full_attn_decode_split_k_gate_bf16_spans,
    qwen35_paged_full_attn_decode_split_k_gqa_gate_bf16_spans,
    qwen35_paged_full_attn_decode_split_k_warp_gate_bf16_spans,
    qwen35_paged_full_attn_prefill_gqa_gate_bf16_spans,
)
from hipengine.kernels.hip_gfx1100.attention.paged_kv_write import (
    build_qwen35_paged_kv_write,
    qwen35_write_paged_kv_mixed_value_bf16_prompt_spans,
    qwen35_write_paged_kv_mixed_value_bf16_spans,
)
from hipengine.kernels.hip_gfx1100.convert import bf16_to_f32, build_cast, f32_to_bf16
from hipengine.kernels.hip_gfx1100.fused import (
    gguf_add_rmsnorm_bf16_f32_weight,
    gguf_bf16_add,
    gguf_qwen35_head_rmsnorm_partial_rotary_position_f32_weight,
    gguf_qwen35_head_rmsnorm_partial_rotary_positions_f32_weight,
    gguf_rmsnorm_bf16_f32_weight,
    gguf_rmsnorm_bf16_f32_weight_out_f32,
    register_paro_combine_kernels,
    register_paro_silu_kernels,
    shared_gate_combine_residual_batch_out_bf16,
    silu_mul_dual_out_bf16,
    silu_mul_separate_out_bf16,
    weighted_lanes_sum_out_bf16_f32w,
)
from hipengine.kernels.hip_gfx1100.fused.paro_combine import (
    weighted_sum_shared_gate_combine_residual_batch_out_bf16_f32w,
    weighted_sum_shared_gate_combine_residual_out_bf16_f32w,
)
from hipengine.kernels.hip_gfx1100.linear.lm_head import (
    argmax_f32,
    argmax_f32_rows_i32,
    build_lm_head,
    lm_head_argmax_stage1_blocks,
)
from hipengine.kernels.hip_gfx1100.rotary.qwen35_rotary import qwen35_split_qgate_bf16
from hipengine.kernels.hip_gfx1100.runtime import (
    advance_decode_position_i64,
    build_runtime_state,
    record_f32_row_indexed,
    record_i64_scalar_indexed,
    set_decode_position_i64,
    set_i64_scalar,
)
from hipengine.kernels.hip_gfx1100.speculative import (
    build_dflash_commit,
    linear_state_pair_commit_chunked_i32,
    linear_state_pair_commit_i32,
)
from hipengine.kvcache import KVLiveSpans
from hipengine.kernels.hip_gfx1100.linear_attn.conv import (
    qwen35_linear_attn_chain_conv_decode_bf16_tloop,
    qwen35_linear_attn_conv_decode_bf16,
    qwen35_linear_attn_conv_prefill_f32,
)
from hipengine.kernels.hip_gfx1100.linear_attn.gdn import (
    qwen35_gdn_chain_recurrent_rmsnorm_gate_lowp_c1_exact_tloop_bf16,
    qwen35_gdn_prefill_recurrent_k2_f32,
    qwen35_gdn_prefill_recurrent_rmsnorm_gate_bf16_decode_order,
    qwen35_gdn_prefill_recurrent_segments_k2_f32,
    qwen35_gdn_prefill_rmsnorm_gate_bf16,
    qwen35_gdn_recurrent_rmsnorm_gate_lowp_bf16,
    qwen35_linear_attn_prefill_prepare_f32_bf16,
    register_qwen35_linear_attn_gdn_kernels,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_expert_pack8_gemv import (
    build_gguf_expert_pack8_gemv,
    register_gguf_expert_pack8_gemv_kernels,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_k_selected_prefill import (
    register_gguf_k_selected_prefill_kernels,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_k_selected_pack8_gemv import (
    register_gguf_k_selected_pack8_gemv_kernels,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_selected_prefill import (
    register_gguf_q4_k_selected_prefill_kernels,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_t16_selected_prefill import (
    register_gguf_q4_k_t16_selected_prefill_kernels,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_k_t16_selected_prefill import (
    register_gguf_k_t16_selected_prefill_kernels,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_selected_pack8_gemv import (
    register_gguf_q4_k_selected_pack8_gemv_kernels,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_t16_selected_gemv import (
    gguf_q4_k_t16_selected_dual_gemv_bf16_bf16_out,
    gguf_q4_k_t16_selected_dual_q8_1_dp4a_gemv_bf16_bf16_out,
    gguf_q4_k_t16_selected_dual_silu_gemv_bf16_bf16_out,
    gguf_q4_k_t16_selected_gemv_bf16_bf16_out,
    gguf_q5_k_t16_selected_gemv_bf16_bf16_out,
    gguf_q5_k_t16_selected_q8_1_dp4a_gemv_bf16_bf16_out,
    gguf_q6_k_t16_selected_gemv_bf16_bf16_out,
    register_gguf_t16_selected_gemv_kernels,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_x8_selected_gemv import (
    gguf_q4_k_x8_selected_dual_q8_1_dp4a_gemv_bf16_bf16_out,
    gguf_q5_k_x8_selected_q8_1_dp4a_gemv_bf16_bf16_out,
    gguf_q5_k_x8_selected_q8_1_dp4a_gemv_decode_compact_bf16_bf16_out,
    gguf_q6_k_x8_selected_q8_1_dp4a_gemv_bf16_bf16_out,
    gguf_q6_k_x8_selected_q8_1_dp4a_gemv_decode_compact_bf16_bf16_out,
    register_gguf_x8_selected_gemv_kernels,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_k_gemv import (
    gguf_q5_k_selected_gemv_bf16_bf16_out,
    gguf_q5_k_selected_pack8_gemv_bf16_bf16_out,
    gguf_q5_k_selected_pack8_q8_1_dp4a_gemv_bf16_bf16_out,
    gguf_q6_k_selected_gemv_bf16_bf16_out,
    gguf_q6_k_selected_pack8_gemv_bf16_bf16_out,
    gguf_q6_k_selected_pack8_q8_1_dp4a_gemv_bf16_bf16_out,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_gemv import (
    gguf_q4_k_quantize_bf16_q8_1,
    gguf_q4_k_selected_dual_gemv_bf16_bf16_out,
    gguf_q4_k_selected_dual_q8_1_dp4a_gemv_bf16_bf16_out,
    gguf_q4_k_selected_gemv_bf16_bf16_out,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q6_k_pack8_gemv import (
    build_gguf_q6_k_pack8_gemv,
    gguf_q6_k_x8_gemv_decode_q8_1_dp4a_top1_gather_f32,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q8_0_dp4a_gemv import (
    gguf_q8_0_dp4a_dual_split_rowtile4_gemv_bf16_bf16_out,
    gguf_q8_0_dp4a_rowtile4_gemv_bf16_bf16_out,
    gguf_q8_0_dp4a_triple_split_rowtile4_gemv_bf16_bf16_out,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_moe_ffn_fused import (
    gguf_q4_k_selected_ffn_fused_bf16_bf16_out,
)
from hipengine.kernels.registry import KernelKey, resolve
from hipengine.kernels.hip_gfx1100.moe.group_scatter import (
    qwen35_moe_group_count,
    qwen35_moe_group_prefix,
    qwen35_moe_group_scatter_gather_lowp,
    qwen35_moe_wmma_tile_map,
    register_qwen35_moe_group_scatter_kernels,
)
from hipengine.kernels.hip_gfx1100.moe.router import (
    qwen35_router_select,
    qwen35_router_topk_split_shared_coop_out_bf16,
)
from hipengine.loading.gguf import GGUFReader
from hipengine.loading.qwen35_gguf import FULL_ATTENTION, LINEAR_ATTENTION, build_qwen35_gguf_tensor_map
from hipengine.loading.qwen35_gguf_expert_sidecar import (
    GGUFExpertPackedTensor,
    build_packed_expert_tensor_from_reader,
    expert_sidecar_cache_path,
    load_packed_expert_tensor,
    save_packed_expert_tensor,
)
from hipengine.loading.qwen35_gguf_materialize import (
    Qwen35GGUFDeviceWeight,
    Qwen35GGUFResidentWeights,
    gguf_decode_repack_enabled,
    materialize_qwen35_gguf_weights,
)
from hipengine.quant.gguf import bf16_to_float32
from hipengine.runtime.gguf_embedding import launch_gguf_embedding
from hipengine.runtime.gguf_linear import (
    GGUF_ACTIVATION_F32,
    GGUF_OUTPUT_F32,
    gemv_decode_session,
    gguf_gemv_decode_enabled,
    gguf_wmma_prefill_enabled,
    launch_gguf_linear,
    launch_gguf_linear_pair,
    launch_gguf_linear_pair_concat,
    launch_gguf_linear_raw_ptr,
    launch_gguf_linear_triple,
    wmma_prefill_session,
)
from hipengine.runtime.prefill import PrefillConfig, resolve_prefill_config_for_sequence


def _add_sync_stage_timing(
    timings: dict[str, float] | None,
    name: str,
    ms: float,
) -> None:
    if timings is None:
        return
    if ms < 0.0:
        raise RuntimeError(f"negative GGUF sync stage timing for {name}: {ms}")
    timings[name] = timings.get(name, 0.0) + float(ms)


def _mark_sync_stage(
    runtime: HipRuntime,
    timings: dict[str, float] | None,
    enabled: bool,
    name: str,
    t0: float,
) -> float:
    if not enabled or timings is None:
        return t0
    runtime.device_synchronize()
    _add_sync_stage_timing(timings, name, (time.perf_counter() - t0) * 1000)
    return time.perf_counter()


@dataclass(frozen=True)
class Qwen35GGUFNextTokenProbeResult:
    token_id: int
    logit: float
    logits: np.ndarray


@dataclass(frozen=True)
class Qwen35GGUFMTPDraftSeed:
    """Ready target seed descriptor for one GGUF MTP draft step."""

    token_id: int
    position: int
    hidden_ptr: int
    hidden_contract: "Qwen35GGUFHiddenSeedContract"

    def __post_init__(self) -> None:
        if self.token_id < 0:
            raise ValueError("MTP draft seed token_id must be non-negative")
        if self.position < 0:
            raise ValueError("MTP draft seed position must be non-negative")
        if self.hidden_ptr <= 0:
            raise ValueError(
                "MTP draft seed hidden_ptr must be a non-zero device pointer"
            )
        if not self.hidden_contract.ready_for_mtp:
            raise ValueError("MTP draft seed requires a ready fp32 hidden contract")
        if self.hidden_contract.rows != 1:
            raise ValueError("MTP draft seed currently supports exactly one hidden row")

    def as_dict(self) -> dict[str, object]:
        return {
            "token_id": self.token_id,
            "position": self.position,
            "hidden_ptr": self.hidden_ptr,
            "hidden_contract": self.hidden_contract.as_dict(),
        }


@dataclass(frozen=True)
class Qwen35GGUFHiddenSeedContract:
    """Contract for GGUF MTP target hidden seeds.

    llama.cpp's Qwen35MoE MTP seed is the post-output_norm hidden row exposed as
    fp32.  The current GGUF resident decode path computes the right provenance
    but stores it in a BF16 scratch row; M2.5 must upgrade that tap before M3 can
    consume it.
    """

    provenance: str
    dtype: DType
    rows: int
    hidden_size: int
    source_buffer: str
    populated_by_decode: bool
    llama_cpp_compatible: bool

    def __post_init__(self) -> None:
        if self.provenance != "post_output_norm":
            raise ValueError("GGUF MTP hidden seed provenance must be post_output_norm")
        if self.rows <= 0:
            raise ValueError("GGUF MTP hidden seed rows must be positive")
        if self.hidden_size <= 0:
            raise ValueError("GGUF MTP hidden seed hidden_size must be positive")
        expected = self.dtype is DType.FP32 and self.populated_by_decode
        if self.llama_cpp_compatible != expected:
            raise ValueError(
                "llama_cpp_compatible must reflect whether the seed is FP32 and populated by decode"
            )

    @property
    def requires_fp32_tap(self) -> bool:
        return self.dtype is not DType.FP32

    @property
    def ready_for_mtp(self) -> bool:
        return self.llama_cpp_compatible

    def as_dict(self) -> dict[str, object]:
        return {
            "provenance": self.provenance,
            "dtype": self.dtype.name,
            "rows": self.rows,
            "hidden_size": self.hidden_size,
            "source_buffer": self.source_buffer,
            "populated_by_decode": self.populated_by_decode,
            "llama_cpp_compatible": self.llama_cpp_compatible,
            "requires_fp32_tap": self.requires_fp32_tap,
            "ready_for_mtp": self.ready_for_mtp,
        }


def qwen35_gguf_fp32_hidden_seed_contract(
    hidden_size: int,
    *,
    rows: int = 1,
    populated_by_decode: bool = False,
) -> Qwen35GGUFHiddenSeedContract:
    """Describe the M2.5 fp32 GGUF hidden seed target buffer."""

    return Qwen35GGUFHiddenSeedContract(
        provenance="post_output_norm",
        dtype=DType.FP32,
        rows=int(rows),
        hidden_size=int(hidden_size),
        source_buffer="Qwen35GGUFResidentSession.scratch.hidden_seed_fp32",
        populated_by_decode=bool(populated_by_decode),
        llama_cpp_compatible=bool(populated_by_decode),
    )


def qwen35_gguf_fp32_verify_hidden_seed_contract(
    hidden_size: int,
    *,
    rows: int = 1,
    populated_by_decode: bool = False,
) -> Qwen35GGUFHiddenSeedContract:
    """Describe the fp32 verifier-row hidden-seed staging buffer."""

    return Qwen35GGUFHiddenSeedContract(
        provenance="post_output_norm",
        dtype=DType.FP32,
        rows=int(rows),
        hidden_size=int(hidden_size),
        source_buffer="Qwen35GGUFResidentSession._verify_hidden_seed_buf",
        populated_by_decode=bool(populated_by_decode),
        llama_cpp_compatible=bool(populated_by_decode),
    )


def qwen35_gguf_current_hidden_seed_contract(
    hidden_size: int,
    *,
    rows: int = 1,
) -> Qwen35GGUFHiddenSeedContract:
    """Describe the current GGUF AR decode hidden seed tap.

    This is a contract descriptor, not a data read.  It intentionally reports the
    current BF16 scratch pointer as non-llama-compatible so later M2.5 runtime
    work has a precise RED target.
    """

    return Qwen35GGUFHiddenSeedContract(
        provenance="post_output_norm",
        dtype=DType.BF16,
        rows=int(rows),
        hidden_size=int(hidden_size),
        source_buffer="Qwen35GGUFResidentSession.scratch.norm",
        populated_by_decode=True,
        llama_cpp_compatible=False,
    )


@dataclass(frozen=True)
class Qwen35GGUFFullAttentionPrefillResult:
    """Host-visible result for a GGUF full-attention layer prefill probe."""

    hidden_bits: np.ndarray
    mode: str
    used_aotriton: bool


@dataclass(frozen=True)
class Qwen35GGUFBlockVerifyResult:
    """Host-visible target block verification result.

    ``token_ids[row]`` is the greedy target token produced after consuming
    ``input_token_ids[row]`` at ``start_position + row``.  ``hidden_seeds[row]``
    is the FP32 post-output_norm row for that consumed target input, matching the
    MTP seed contract used by llama.cpp's shifted draft model.
    """

    input_token_ids: list[int]
    token_ids: list[int]
    hidden_seeds: np.ndarray
    start_position: int
    linear_state_rows_captured: bool = False

    def __post_init__(self) -> None:
        if self.start_position < 0:
            raise ValueError("start_position must be non-negative")
        if len(self.input_token_ids) == 0:
            raise ValueError("input_token_ids must be non-empty")
        if len(self.token_ids) != len(self.input_token_ids):
            raise ValueError("token_ids must match input_token_ids length")
        if self.hidden_seeds.shape[0] != len(self.input_token_ids):
            raise ValueError("hidden_seeds rows must match input_token_ids length")
        if self.hidden_seeds.dtype != np.float32:
            raise ValueError("hidden_seeds must be float32")


@dataclass(frozen=True)
class Qwen35GGUFLinearAttentionBoundaryCapture:
    """Host-visible diagnostic snapshot for one GGUF linear-attention boundary."""

    layer_id: int
    token_id: int
    position: int
    hidden_size: int
    ssm_time_step_rank: int
    linear_qkv_width: int
    ssm_inner_size: int
    attn_norm_f32: np.ndarray
    linear_qkv_f32: np.ndarray
    linear_z_f32: np.ndarray
    ssm_alpha_f32: np.ndarray
    ssm_beta_f32: np.ndarray
    conv_out_f32: np.ndarray
    recurrent_out_f32: np.ndarray
    recurrent_bf16_f32: np.ndarray
    attn_out_f32: np.ndarray

    def as_summary_dict(self) -> dict[str, object]:
        return {
            "layer_id": int(self.layer_id),
            "token_id": int(self.token_id),
            "position": int(self.position),
            "hidden_size": int(self.hidden_size),
            "ssm_time_step_rank": int(self.ssm_time_step_rank),
            "linear_qkv_width": int(self.linear_qkv_width),
            "ssm_inner_size": int(self.ssm_inner_size),
            "attn_norm_shape": list(self.attn_norm_f32.shape),
            "linear_qkv_shape": list(self.linear_qkv_f32.shape),
            "linear_z_shape": list(self.linear_z_f32.shape),
            "ssm_alpha_shape": list(self.ssm_alpha_f32.shape),
            "ssm_beta_shape": list(self.ssm_beta_f32.shape),
            "conv_out_shape": list(self.conv_out_f32.shape),
            "recurrent_out_shape": list(self.recurrent_out_f32.shape),
            "recurrent_bf16_shape": list(self.recurrent_bf16_f32.shape),
            "attn_out_shape": list(self.attn_out_f32.shape),
            "finite": bool(
                np.all(np.isfinite(self.attn_norm_f32))
                and np.all(np.isfinite(self.linear_qkv_f32))
                and np.all(np.isfinite(self.linear_z_f32))
                and np.all(np.isfinite(self.ssm_alpha_f32))
                and np.all(np.isfinite(self.ssm_beta_f32))
                and np.all(np.isfinite(self.conv_out_f32))
                and np.all(np.isfinite(self.recurrent_out_f32))
                and np.all(np.isfinite(self.recurrent_bf16_f32))
                and np.all(np.isfinite(self.attn_out_f32))
            ),
        }


@dataclass(frozen=True)
class Qwen35GGUFLinearAttentionLayerCapture:
    """Host-visible diagnostic snapshot for a full attention layer."""

    layer_id: int
    layer_type: str
    token_id: int
    position: int
    hidden_size: int
    is_moe: bool
    top_k: int
    preceding_layer_count: int
    hidden_in_f32: np.ndarray
    attn_out_f32: np.ndarray
    post_norm_f32: np.ndarray
    residual_f32: np.ndarray
    ffn_or_moe_down_f32: np.ndarray
    layer_out_f32: np.ndarray
    moe_shared_out_f32: np.ndarray | None = None
    moe_routing_weights_f32: np.ndarray | None = None
    moe_shared_gate_f32: np.ndarray | None = None
    moe_selected_experts_i64: np.ndarray | None = None

    def as_summary_dict(self) -> dict[str, object]:
        optional_finite = True
        if self.moe_shared_out_f32 is not None:
            optional_finite = bool(np.all(np.isfinite(self.moe_shared_out_f32)))
        if self.moe_routing_weights_f32 is not None:
            optional_finite = optional_finite and bool(
                np.all(np.isfinite(self.moe_routing_weights_f32))
            )
        if self.moe_shared_gate_f32 is not None:
            optional_finite = optional_finite and bool(
                np.all(np.isfinite(self.moe_shared_gate_f32))
            )
        return {
            "layer_id": int(self.layer_id),
            "layer_type": str(self.layer_type),
            "token_id": int(self.token_id),
            "position": int(self.position),
            "hidden_size": int(self.hidden_size),
            "is_moe": bool(self.is_moe),
            "top_k": int(self.top_k),
            "preceding_layer_count": int(self.preceding_layer_count),
            "hidden_in_shape": list(self.hidden_in_f32.shape),
            "attn_out_shape": list(self.attn_out_f32.shape),
            "post_norm_shape": list(self.post_norm_f32.shape),
            "residual_shape": list(self.residual_f32.shape),
            "ffn_or_moe_down_shape": list(self.ffn_or_moe_down_f32.shape),
            "moe_shared_out_shape": (
                None
                if self.moe_shared_out_f32 is None
                else list(self.moe_shared_out_f32.shape)
            ),
            "moe_routing_weights_shape": (
                None
                if self.moe_routing_weights_f32 is None
                else list(self.moe_routing_weights_f32.shape)
            ),
            "moe_shared_gate_shape": (
                None
                if self.moe_shared_gate_f32 is None
                else list(self.moe_shared_gate_f32.shape)
            ),
            "moe_selected_experts_shape": (
                None
                if self.moe_selected_experts_i64 is None
                else list(self.moe_selected_experts_i64.shape)
            ),
            "layer_out_shape": list(self.layer_out_f32.shape),
            "finite": bool(
                np.all(np.isfinite(self.hidden_in_f32))
                and np.all(np.isfinite(self.attn_out_f32))
                and np.all(np.isfinite(self.post_norm_f32))
                and np.all(np.isfinite(self.residual_f32))
                and np.all(np.isfinite(self.ffn_or_moe_down_f32))
                and np.all(np.isfinite(self.layer_out_f32))
                and optional_finite
            ),
        }


def _launch_qwen35_router_logits_bf16_hidden(
    hidden_ptr: int,
    weight: Qwen35GGUFDeviceWeight,
    logits_ptr: int,
    tokens: int,
    hidden_size: int,
    num_rows: int,
    *,
    stream: int = 0,
    runtime=None,
) -> None:
    """Launch BF16-hidden router logits through the kernel registry.

    Qwen3.6 stores the MoE router and shared-gate vectors as GGUF F32 tensors;
    llama.cpp consumes them as F32.  Older hipEngine builds contracted those
    tensors to BF16 and called the BF16-weight router kernel.  Keeping this as a
    tiny registry adapter avoids quant/layout branches at the call sites while
    preserving the BF16-weight path for fixtures and legacy materializations.
    """

    fn = resolve(
        backend="hip_gfx1100",
        layer="router_logits",
        quant=weight.spec.quant_key,
        variant="bf16_hidden",
    )
    fn(
        hidden_ptr,
        weight.allocation().tensor.ptr,
        logits_ptr,
        tokens,
        hidden_size,
        num_rows,
        stream=stream,
        runtime=runtime,
    )


@dataclass(frozen=True)
class _DeviceExpertPackedTensor:
    quant_key: str
    qweight_low: DeviceBuffer
    scales: DeviceBuffer
    qweight_high: DeviceBuffer | None
    mins: DeviceBuffer | None
    num_experts: int
    in_features: int
    out_features: int
    buffers: tuple[DeviceBuffer, ...]

    @classmethod
    def from_host(cls, packed: GGUFExpertPackedTensor, *, runtime: HipRuntime) -> "_DeviceExpertPackedTensor":
        buffers: list[DeviceBuffer] = []
        try:
            qweight_low = _copy_sidecar_array_to_device(packed.qweight_low, runtime=runtime)
            buffers.append(qweight_low)
            scales = _copy_sidecar_array_to_device(packed.scales, runtime=runtime)
            buffers.append(scales)
            qweight_high = None
            if packed.qweight_high is not None:
                qweight_high = _copy_sidecar_array_to_device(packed.qweight_high, runtime=runtime)
                buffers.append(qweight_high)
            mins = None
            if packed.mins is not None:
                mins = _copy_sidecar_array_to_device(packed.mins, runtime=runtime)
                buffers.append(mins)
            return cls(
                quant_key=packed.quant_key,
                qweight_low=qweight_low,
                scales=scales,
                qweight_high=qweight_high,
                mins=mins,
                num_experts=packed.num_experts,
                in_features=packed.in_features,
                out_features=packed.out_features,
                buffers=tuple(buffers),
            )
        except Exception:
            for buffer in reversed(buffers):
                free(buffer, runtime=runtime)
            raise

    def free(self, *, runtime: HipRuntime) -> None:
        for buffer in reversed(self.buffers):
            free(buffer, runtime=runtime)


@dataclass(frozen=True)
class _DeviceExpertLayerSidecar:
    tensors: dict[str, _DeviceExpertPackedTensor]

    def tensor(self, slot: str) -> _DeviceExpertPackedTensor:
        return self.tensors[slot]

    def free(self, *, runtime: HipRuntime) -> None:
        for tensor in reversed(tuple(self.tensors.values())):
            tensor.free(runtime=runtime)


@dataclass
class Qwen35GGUFOneLayerProbe:
    """Minimal resident GGUF one-layer projection probe.

    This is not yet the full Qwen3.5 layer. It is the first live runtime wiring
    that starts from a Q6_K token embedding, applies the layer RMSNorm, then
    launches GGUF linear projections through the registry adapter to produce a
    hidden-size BF16 output. The full layer runner will replace this probe once
    conv/SSM/attention/residual/MLP are wired.
    """

    model_path: str | Path
    layer_id: int = 0
    runtime: HipRuntime | None = None
    weights: Qwen35GGUFResidentWeights | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.runtime = self.runtime or get_hip_runtime()
        selected = (
            "root.token_embedding",
            "root.lm_head",
            f"layers.{self.layer_id}.attn_norm",
            f"layers.{self.layer_id}.attn_gate",
            f"layers.{self.layer_id}.ssm_out",
        )
        self.weights = materialize_qwen35_gguf_weights(
            self.model_path,
            selected_slots=selected,
            runtime=self.runtime,
        )
        if self.weights.config.layer_types[self.layer_id] != LINEAR_ATTENTION:
            raise ValueError(f"layer {self.layer_id} is not a linear_attention layer")

    @property
    def hidden_size(self) -> int:
        assert self.weights is not None
        return self.weights.config.hidden_size

    @property
    def ssm_inner_size(self) -> int:
        assert self.weights is not None
        return self.weights.config.ssm_inner_size

    @property
    def vocab_size(self) -> int:
        assert self.weights is not None
        return self.weights.config.vocab_size

    def run_token(self, token_id: int) -> np.ndarray:
        """Run the one-layer projection probe and return BF16 bits on host."""

        assert self.weights is not None
        runtime = self.runtime or get_hip_runtime()
        token_ids = np.asarray([int(token_id)], dtype=np.int64)
        out_bits = np.empty((1, self.hidden_size), dtype=np.uint16)
        buffers = []
        try:
            token_buf = malloc(token_ids.nbytes, runtime=runtime)
            hidden_buf = malloc(out_bits.nbytes, runtime=runtime)
            norm_buf = malloc(out_bits.nbytes, runtime=runtime)
            gate_buf = malloc(2 * self.ssm_inner_size, runtime=runtime)
            out_buf = malloc(out_bits.nbytes, runtime=runtime)
            buffers.extend((token_buf, hidden_buf, norm_buf, gate_buf, out_buf))
            copy_host_to_device(token_buf, host_array_ptr(token_ids), runtime=runtime)

            launch_gguf_embedding(
                self.weights.root("token_embedding"),
                token_buf.ptr,
                hidden_buf.ptr,
                rows=1,
                hidden_size=self.hidden_size,
                vocab_size=self.vocab_size,
                runtime=runtime,
            )
            gguf_rmsnorm_bf16_f32_weight(
                hidden_buf.ptr,
                self.weights.layer(self.layer_id).weight("attn_norm").allocation().tensor.ptr,
                norm_buf.ptr,
                rows=1,
                hidden_size=self.hidden_size,
                eps=self.weights.config.rms_norm_eps,
                runtime=runtime,
            )
            launch_gguf_linear(
                self.weights.layer(self.layer_id).weight("attn_gate"),
                norm_buf.ptr,
                gate_buf.ptr,
                rows=1,
                in_features=self.hidden_size,
                out_features=self.ssm_inner_size,
                runtime=runtime,
            )
            launch_gguf_linear(
                self.weights.layer(self.layer_id).weight("ssm_out"),
                gate_buf.ptr,
                out_buf.ptr,
                rows=1,
                in_features=self.ssm_inner_size,
                out_features=self.hidden_size,
                runtime=runtime,
            )
            runtime.device_synchronize()
            copy_device_to_host(host_array_ptr(out_bits), out_buf, runtime=runtime)
        finally:
            for buffer in reversed(buffers):
                free(buffer, runtime=runtime)
        return out_bits

    def run_token_f32(self, token_id: int) -> np.ndarray:
        return bf16_to_float32(self.run_token(token_id))

    def logits_from_hidden_bits(self, hidden_bits: np.ndarray) -> np.ndarray:
        """Run the tied Q6_K lm-head and return FP32 logits on host."""

        assert self.weights is not None
        runtime = self.runtime or get_hip_runtime()
        hidden = np.ascontiguousarray(hidden_bits, dtype=np.uint16)
        if hidden.shape != (1, self.hidden_size):
            raise ValueError(f"hidden_bits must have shape (1, {self.hidden_size})")
        logits = np.empty((1, self.vocab_size), dtype=np.float32)
        buffers = []
        try:
            hidden_buf = malloc(hidden.nbytes, runtime=runtime)
            logits_buf = malloc(logits.nbytes, runtime=runtime)
            buffers.extend((hidden_buf, logits_buf))
            copy_host_to_device(hidden_buf, host_array_ptr(hidden), runtime=runtime)
            launch_gguf_linear(
                self.weights.root("lm_head"),
                hidden_buf.ptr,
                logits_buf.ptr,
                rows=1,
                in_features=self.hidden_size,
                out_features=self.vocab_size,
                output_dtype=GGUF_OUTPUT_F32,
                runtime=runtime,
            )
            runtime.device_synchronize()
            copy_device_to_host(host_array_ptr(logits), logits_buf, runtime=runtime)
        finally:
            for buffer in reversed(buffers):
                free(buffer, runtime=runtime)
        return logits

    def sample_next_token(self, token_id: int) -> Qwen35GGUFNextTokenProbeResult:
        logits = self.logits_from_hidden_bits(self.run_token(token_id))
        if not np.all(np.isfinite(logits)):
            raise FloatingPointError("GGUF lm-head logits contain NaN or Inf")
        flat = logits.reshape(-1)
        next_id = int(np.argmax(flat))
        return Qwen35GGUFNextTokenProbeResult(
            token_id=next_id,
            logit=float(flat[next_id]),
            logits=logits,
        )

    def close(self) -> None:
        if self.weights is not None:
            self.weights.free(runtime=self.runtime)
            self.weights = None

    def __enter__(self) -> "Qwen35GGUFOneLayerProbe":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


@dataclass
class Qwen35GGUFFullStackRunner:
    """GGUF Qwen3.5 full-stack primitive runner over resident native weights.

    The public generator uses :class:`Qwen35GGUFResidentSession` so decode state
    persists across tokens.  This lower-level runner remains as a deterministic
    compatibility/probe surface and still provides ``sample_next_token`` for
    tests that intentionally compare against the old full-context replay path.
    """

    model_path: str | Path
    runtime: HipRuntime | None = None
    compiler_version: str | None = None
    require_cached_build: bool = False
    weights: Qwen35GGUFResidentWeights | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.runtime = self.runtime or get_hip_runtime()
        self.require_cached_build = bool(self.require_cached_build)
        self.weights = materialize_qwen35_gguf_weights(self.model_path, runtime=self.runtime)

    def _aotriton_prefill_library(self):
        """Return the cached AOTriton prefill shim handle."""

        library = getattr(self, "_aotriton_library", None)
        if library is None:
            library = build_aotriton_wrap(
                load=True,
                compiler_version=self.compiler_version,
                require_cached=self.require_cached_build,
            )
            self._aotriton_library = library
        return library

    def _paged_attn_decode_library(self):
        """Return the cached native paged-attention/gate shim handle."""

        library = getattr(self, "_paged_attn_decode_library_handle", None)
        if library is None:
            library = build_qwen35_paged_attn_decode(
                load=True,
                compiler_version=self.compiler_version,
                require_cached=self.require_cached_build,
            )
            self._paged_attn_decode_library_handle = library
        return library

    def _cast_library(self):
        """Return the cached dtype-cast shim handle."""

        library = getattr(self, "_cast_library_handle", None)
        if library is None:
            library = build_cast(
                load=True,
                compiler_version=self.compiler_version,
                require_cached=self.require_cached_build,
            )
            self._cast_library_handle = library
        return library

    def _paged_kv_write_library(self):
        """Return the cached paged-KV write shim handle."""

        library = getattr(self, "_paged_kv_write_library_handle", None)
        if library is None:
            library = build_qwen35_paged_kv_write(
                load=True,
                compiler_version=self.compiler_version,
                require_cached=self.require_cached_build,
            )
            self._paged_kv_write_library_handle = library
        return library

    def _gdn_prefill_plan(self) -> _GGUFGDNPrefillPlan:
        """Return the cached qwen35 GGUF GDN prefill plan.

        Resolved once per runner via the kernel registry. Falls back to the
        legacy fused decode-order kernel when the chained path is incomplete.
        """

        plan = getattr(self, "_gguf_gdn_prefill_plan_cache", None)
        if plan is None:
            plan = _resolve_gguf_gdn_prefill_plan()
            self._gguf_gdn_prefill_plan_cache = plan
        return plan

    def _run_gdn_prefill(
        self,
        *,
        layer,
        scratch,
        cfg,
        rows: int,
        recurrent_state,
        stream: int,
        runtime: HipRuntime,
    ) -> None:
        """Dispatch the qwen35 GGUF GDN prefill chain (or fused fallback).

        Plugin-style: the kernel chain is resolved via the kernel registry
        keyed by ``(hip_gfx1100, ..., gguf_qwen35, ...)``. Whether the
        single-segment k2 or multi-segment k2_segments recurrent kernel runs
        is a perf-tuning decision controlled by
        ``HIPENGINE_GGUF_GDN_PREFILL_SEGMENT_THRESHOLD`` (default 256), not a
        per-quant/per-backend branch.
        """

        plan = self._gdn_prefill_plan()
        if plan.has_fused:
            # Correctness-first fallback: the split prepare+k2+rmsnorm chain is
            # still registered for tests and future perf work, but real GGUF
            # prompt parity currently matches the token-serial path only through
            # the legacy decode-order fused kernel.  Keep bulk prefill on that
            # path until the chain is re-certified against the same target AR
            # trace.
            plan.fused_decode_order(
                scratch.conv_out.ptr,
                scratch.linear_z.ptr,
                scratch.linear_alpha.ptr,
                scratch.linear_beta.ptr,
                layer.weight("ssm_dt_bias").allocation().tensor.ptr,
                layer.weight("ssm_a").allocation().tensor.ptr,
                layer.weight("ssm_norm").allocation().tensor.ptr,
                recurrent_state.ptr,
                scratch.recurrent_bf16.ptr,
                cfg.rms_norm_eps,
                rows,
                cfg.ssm_group_count,
                cfg.ssm_time_step_rank,
                cfg.ssm_state_size,
                self.ssm_value_dim,
                stream=stream,
                runtime=runtime,
            )
            return
        if plan.has_chain:
            plan.prepare(
                scratch.conv_out.ptr,
                scratch.linear_alpha.ptr,
                scratch.linear_beta.ptr,
                layer.weight("ssm_dt_bias").allocation().tensor.ptr,
                layer.weight("ssm_a").allocation().tensor.ptr,
                scratch.prefill_query.ptr,
                scratch.prefill_key.ptr,
                scratch.prefill_value.ptr,
                scratch.prefill_beta.ptr,
                scratch.prefill_decay.ptr,
                rows,
                cfg.ssm_group_count,
                cfg.ssm_time_step_rank,
                cfg.ssm_state_size,
                self.ssm_value_dim,
                stream=stream,
                runtime=runtime,
            )
            segment_threshold = _gguf_gdn_prefill_segment_threshold()
            use_segments = (
                plan.recurrent_segments is not None
                and rows >= segment_threshold
                and getattr(scratch, "gdn_cu_seqlens", None) is not None
                and getattr(scratch, "gdn_state_indices", None) is not None
            )
            if use_segments:
                plan.recurrent_segments(
                    scratch.prefill_query.ptr,
                    scratch.prefill_key.ptr,
                    scratch.prefill_value.ptr,
                    scratch.prefill_beta.ptr,
                    scratch.prefill_decay.ptr,
                    recurrent_state.ptr,
                    scratch.recurrent_out.ptr,
                    scratch.gdn_cu_seqlens.ptr,
                    scratch.gdn_state_indices.ptr,
                    rows,
                    1,
                    cfg.ssm_time_step_rank,
                    cfg.ssm_state_size,
                    self.ssm_value_dim,
                    stream=stream,
                    runtime=runtime,
                )
            else:
                plan.recurrent(
                    scratch.prefill_query.ptr,
                    scratch.prefill_key.ptr,
                    scratch.prefill_value.ptr,
                    scratch.prefill_beta.ptr,
                    scratch.prefill_decay.ptr,
                    recurrent_state.ptr,
                    scratch.recurrent_out.ptr,
                    rows,
                    cfg.ssm_time_step_rank,
                    cfg.ssm_state_size,
                    self.ssm_value_dim,
                    stream=stream,
                    runtime=runtime,
                )
            plan.rmsnorm_gate(
                scratch.recurrent_out.ptr,
                scratch.linear_z.ptr,
                layer.weight("ssm_norm").allocation().tensor.ptr,
                scratch.recurrent_bf16.ptr,
                cfg.rms_norm_eps,
                rows,
                cfg.ssm_time_step_rank,
                self.ssm_value_dim,
                stream=stream,
                runtime=runtime,
            )
            return
        raise RuntimeError(
            "no qwen35 GGUF GDN prefill kernels are registered; "
            "call register_qwen35_linear_attn_gdn_kernels() before prefill"
        )

    @property
    def hidden_size(self) -> int:
        assert self.weights is not None
        return self.weights.config.hidden_size

    @property
    def vocab_size(self) -> int:
        assert self.weights is not None
        return self.weights.config.vocab_size

    @property
    def ffn_size(self) -> int:
        assert self.weights is not None
        return self.weights.config.feed_forward_length

    @property
    def expert_count(self) -> int:
        assert self.weights is not None
        return self.weights.config.expert_count

    @property
    def top_k(self) -> int:
        assert self.weights is not None
        return self.weights.config.expert_used_count

    @property
    def shared_ffn_size(self) -> int:
        assert self.weights is not None
        return self.weights.config.expert_shared_feed_forward_length

    @property
    def q_width(self) -> int:
        assert self.weights is not None
        return self.weights.config.head_count * self.weights.config.key_length

    @property
    def kv_width(self) -> int:
        assert self.weights is not None
        return self.weights.config.head_count_kv * self.weights.config.value_length

    @property
    def linear_qkv_width(self) -> int:
        assert self.weights is not None
        cfg = self.weights.config
        return 2 * cfg.ssm_group_count * cfg.ssm_state_size + cfg.ssm_inner_size

    @property
    def ssm_value_dim(self) -> int:
        assert self.weights is not None
        return self.weights.config.ssm_inner_size // self.weights.config.ssm_time_step_rank

    def run_prompt_hidden(
        self,
        token_ids: list[int] | tuple[int, ...],
        *,
        layer_limit: int | None = None,
    ) -> np.ndarray:
        """Run prompt tokens sequentially and return final BF16 hidden bits."""

        if not token_ids:
            raise ValueError("token_ids must be non-empty")
        assert self.weights is not None
        runtime = self.runtime or get_hip_runtime()
        layer_count = self.weights.config.block_count if layer_limit is None else int(layer_limit)
        if layer_count < 0 or layer_count > self.weights.config.block_count:
            raise ValueError("layer_limit must be between 0 and block_count")
        hidden_bits = np.empty((1, self.hidden_size), dtype=np.uint16)
        token_arr = np.empty((1,), dtype=np.int64)
        buffers = []
        try:
            token_buf = malloc(token_arr.nbytes, runtime=runtime)
            hidden_a = malloc(hidden_bits.nbytes, runtime=runtime)
            hidden_b = malloc(hidden_bits.nbytes, runtime=runtime)
            scratch = _FullStackScratch.allocate(self, runtime=runtime)
            buffers.extend((token_buf, hidden_a, hidden_b, *scratch.buffers))
            scratch.zero_states(runtime)
            src = hidden_a
            dst = hidden_b
            for position, token_id in enumerate(token_ids):
                scratch.set_full_attention_position(position, runtime)
                token_arr[0] = int(token_id)
                copy_host_to_device(token_buf, host_array_ptr(token_arr), runtime=runtime)
                launch_gguf_embedding(
                    self.weights.root("token_embedding"),
                    token_buf.ptr,
                    hidden_a.ptr,
                    rows=1,
                    hidden_size=self.hidden_size,
                    vocab_size=self.vocab_size,
                    runtime=runtime,
                )
                src = hidden_a
                dst = hidden_b
                for layer_id, layer_type in enumerate(self.weights.config.layer_types[:layer_count]):
                    if layer_type == LINEAR_ATTENTION:
                        self._run_linear_attention_layer(layer_id, src.ptr, dst.ptr, scratch)
                    elif layer_type == FULL_ATTENTION:
                        self._run_full_attention_layer(layer_id, src.ptr, dst.ptr, scratch, position=position)
                    else:
                        raise ValueError(f"unsupported GGUF layer type {layer_type!r}")
                    src, dst = dst, src
            gguf_rmsnorm_bf16_f32_weight(
                src.ptr,
                self.weights.root("output_norm").allocation().tensor.ptr,
                scratch.norm.ptr,
                rows=1,
                hidden_size=self.hidden_size,
                eps=self.weights.config.rms_norm_eps,
                runtime=runtime,
            )
            runtime.device_synchronize()
            copy_device_to_host(host_array_ptr(hidden_bits), scratch.norm, runtime=runtime)
        finally:
            for buffer in reversed(buffers):
                free(buffer, runtime=runtime)
        return hidden_bits

    def run_token_hidden(self, token_id: int, *, layer_limit: int | None = None) -> np.ndarray:
        """Run all layers for one token and return BF16 hidden bits on host."""

        return self.run_prompt_hidden([int(token_id)], layer_limit=layer_limit)

    def logits_from_hidden_bits(self, hidden_bits: np.ndarray) -> np.ndarray:
        assert self.weights is not None
        runtime = self.runtime or get_hip_runtime()
        hidden = np.ascontiguousarray(hidden_bits, dtype=np.uint16)
        if hidden.shape != (1, self.hidden_size):
            raise ValueError(f"hidden_bits must have shape (1, {self.hidden_size})")
        logits = np.empty((1, self.vocab_size), dtype=np.float32)
        buffers = []
        try:
            hidden_buf = malloc(hidden.nbytes, runtime=runtime)
            logits_buf = malloc(logits.nbytes, runtime=runtime)
            buffers.extend((hidden_buf, logits_buf))
            copy_host_to_device(hidden_buf, host_array_ptr(hidden), runtime=runtime)
            launch_gguf_linear(
                self.weights.root("lm_head"),
                hidden_buf.ptr,
                logits_buf.ptr,
                rows=1,
                in_features=self.hidden_size,
                out_features=self.vocab_size,
                output_dtype=GGUF_OUTPUT_F32,
                runtime=runtime,
            )
            runtime.device_synchronize()
            copy_device_to_host(host_array_ptr(logits), logits_buf, runtime=runtime)
        finally:
            for buffer in reversed(buffers):
                free(buffer, runtime=runtime)
        return logits

    def sample_next_token(self, token_ids: list[int] | tuple[int, ...]) -> Qwen35GGUFNextTokenProbeResult:
        logits = self.logits_from_hidden_bits(self.run_prompt_hidden(token_ids))
        if not np.all(np.isfinite(logits)):
            raise FloatingPointError("GGUF full-stack lm-head logits contain NaN or Inf")
        flat = logits.reshape(-1)
        next_id = int(np.argmax(flat))
        return Qwen35GGUFNextTokenProbeResult(
            token_id=next_id,
            logit=float(flat[next_id]),
            logits=logits,
        )

    def run_full_attention_prefill_layer(
        self,
        layer_id: int,
        hidden_bits: np.ndarray,
        *,
        prefill_config: PrefillConfig | None = None,
        attn_aotriton_min_tokens: int | None = None,
    ) -> Qwen35GGUFFullAttentionPrefillResult:
        """Run one GGUF full-attention layer over multiple prompt rows.

        This is the layer-level native prefill path used to validate the GGUF
        full-attention prefill wiring before the full-model scheduler is
        promoted. Rows below the threshold use the existing resident one-token
        path in a loop; rows at/above the threshold use the batched prefill
        path after GGUF Q/K/V projection and GPU q/k norm+RoPE. The batched
        path dispatches AOTriton at the retained 512-token crossover and uses
        the native causal GQA kernel below that crossover.
        """

        if self.weights is None:
            raise RuntimeError("GGUF runner is closed")
        if self.weights.config.layer_types[layer_id] != FULL_ATTENTION:
            raise ValueError(f"layer {layer_id} is not a full_attention layer")
        hidden = np.ascontiguousarray(hidden_bits, dtype=np.uint16)
        if hidden.ndim != 2 or hidden.shape[1] != self.hidden_size:
            raise ValueError(f"hidden_bits must have shape (rows, {self.hidden_size})")
        rows = int(hidden.shape[0])
        if rows <= 0:
            raise ValueError("hidden_bits must contain at least one row")
        config = prefill_config or PrefillConfig()
        threshold = int(config.attn_aotriton_min_tokens if attn_aotriton_min_tokens is None else attn_aotriton_min_tokens)
        if threshold < 0:
            raise ValueError("attn_aotriton_min_tokens must be non-negative")
        use_aotriton = threshold > 0 and rows >= threshold
        runtime = self.runtime or get_hip_runtime()
        output = np.empty_like(hidden)
        buffers = []
        try:
            hidden_buf = malloc(hidden.nbytes, runtime=runtime)
            out_buf = malloc(output.nbytes, runtime=runtime)
            buffers.extend((hidden_buf, out_buf))
            copy_host_to_device(hidden_buf, host_array_ptr(hidden), runtime=runtime)
            if use_aotriton:
                prefill_scratch = _GGUFFullAttentionPrefillScratch.allocate(self, rows=rows, runtime=runtime)
                buffers.extend(prefill_scratch.buffers)
                used_aotriton = self._run_full_attention_prefill_layer_aotriton(
                    layer_id,
                    hidden_buf.ptr,
                    out_buf.ptr,
                    prefill_scratch,
                )
                mode = f"aotriton_{_gguf_aotriton_prefill_mode(0, rows, rows)}" if used_aotriton else "native_gqa_bf16"
            else:
                scratch = _FullStackScratch.allocate(self, runtime=runtime)
                buffers.extend(scratch.buffers)
                scratch.zero_states(runtime)
                hidden_row_nbytes = self.hidden_size * 2
                for row in range(rows):
                    scratch.set_full_attention_position(row, runtime)
                    self._run_full_attention_layer(
                        layer_id,
                        hidden_buf.ptr + row * hidden_row_nbytes,
                        out_buf.ptr + row * hidden_row_nbytes,
                        scratch,
                        position=row,
                    )
                mode = "native_sequential"
            runtime.device_synchronize()
            copy_device_to_host(host_array_ptr(output), out_buf, runtime=runtime)
        finally:
            for buffer in reversed(buffers):
                free(buffer, runtime=runtime)
        return Qwen35GGUFFullAttentionPrefillResult(
            hidden_bits=output,
            mode=mode,
            used_aotriton=mode.startswith("aotriton"),
        )

    def _run_full_attention_prefill_layer_aotriton(
        self,
        layer_id: int,
        hidden_ptr: int,
        out_ptr: int,
        scratch,
        *,
        stream: int = 0,
        expert_sidecar: _DeviceExpertLayerSidecar | None = None,
    ) -> bool:
        assert self.weights is not None
        layer = self.weights.layer(layer_id)
        cfg = self.weights.config
        runtime = self.runtime or get_hip_runtime()
        rows = scratch.rows
        cast_library = self._cast_library()
        kv_write_library = self._paged_kv_write_library()
        gguf_rmsnorm_bf16_f32_weight(
            hidden_ptr,
            layer.weight("attn_norm").allocation().tensor.ptr,
            scratch.norm.ptr,
            rows=rows,
            hidden_size=self.hidden_size,
            eps=cfg.rms_norm_eps,
            stream=stream,
            runtime=runtime,
        )
        if not launch_gguf_linear_triple(
            layer.weight("attn_q"),
            layer.weight("attn_k"),
            layer.weight("attn_v"),
            scratch.norm.ptr,
            scratch.full_q.ptr,
            scratch.full_k.ptr,
            scratch.full_v.ptr,
            rows=rows,
            in_features=self.hidden_size,
            out_features=2 * self.q_width,
            out_features_b=self.kv_width,
            out_features_c=self.kv_width,
            stream=stream,
            runtime=runtime,
        ):
            launch_gguf_linear(
                layer.weight("attn_q"),
                scratch.norm.ptr,
                scratch.full_q.ptr,
                rows=rows,
                in_features=self.hidden_size,
                out_features=2 * self.q_width,
                stream=stream,
                runtime=runtime,
            )
            if not launch_gguf_linear_pair(
                layer.weight("attn_k"),
                layer.weight("attn_v"),
                scratch.norm.ptr,
                scratch.full_k.ptr,
                scratch.full_v.ptr,
                rows=rows,
                in_features=self.hidden_size,
                out_features=self.kv_width,
                stream=stream,
                runtime=runtime,
            ):
                launch_gguf_linear(
                    layer.weight("attn_k"),
                    scratch.norm.ptr,
                    scratch.full_k.ptr,
                    rows=rows,
                    in_features=self.hidden_size,
                    out_features=self.kv_width,
                    stream=stream,
                    runtime=runtime,
                )
                launch_gguf_linear(
                    layer.weight("attn_v"),
                    scratch.norm.ptr,
                    scratch.full_v.ptr,
                    rows=rows,
                    in_features=self.hidden_size,
                    out_features=self.kv_width,
                    stream=stream,
                    runtime=runtime,
                )
        qwen35_split_qgate_bf16(
            scratch.full_q.ptr,
            scratch.full_query_raw.ptr,
            scratch.full_gate.ptr,
            rows,
            cfg.head_count,
            cfg.key_length,
            stream=stream,
            runtime=runtime,
        )
        bf16_to_f32(
            scratch.full_k.ptr,
            scratch.full_key_raw.ptr,
            rows * self.kv_width,
            stream=stream,
            library=cast_library,
            runtime=runtime,
        )
        gguf_qwen35_head_rmsnorm_partial_rotary_positions_f32_weight(
            scratch.full_query_raw.ptr,
            scratch.full_key_raw.ptr,
            layer.weight("attn_q_norm").allocation().tensor.ptr,
            layer.weight("attn_k_norm").allocation().tensor.ptr,
            scratch.cos_table.ptr,
            scratch.sin_table.ptr,
            scratch.positions_tensor.ptr,
            scratch.full_query.ptr,
            scratch.full_key.ptr,
            cfg.rms_norm_eps,
            rows,
            cfg.head_count,
            cfg.head_count_kv,
            cfg.key_length,
            cfg.rope_dimension_count,
            scratch.max_positions,
            stream=stream,
            runtime=runtime,
        )
        qwen35_write_paged_kv_mixed_value_bf16_prompt_spans(
            scratch.full_key.ptr,
            scratch.full_v.ptr,
            scratch.key_cache.ptr,
            scratch.value_cache.ptr,
            scratch.append_spans,
            rows,
            scratch.block_size,
            cfg.head_count_kv,
            cfg.key_length,
            stream=stream,
            library=kv_write_library,
            runtime=runtime,
        )
        threshold = int(PrefillConfig().attn_aotriton_min_tokens)
        use_aotriton = threshold > 0 and rows >= threshold
        paged_attn_library = self._paged_attn_decode_library()
        end = scratch.start + rows
        if scratch.key_cache is None or scratch.value_cache is None:
            raise RuntimeError(
                "GGUF full-attention prefill requires cache-backed key/value buffers; "
                "resident bulk prefill should replace scratch caches with _FullStackScratch.full_cache(layer_id)"
            )

        if use_aotriton:
            aotriton_library = self._aotriton_prefill_library()

            # Convert FP32 query to BF16 for AOTriton
            f32_to_bf16(
                scratch.full_query.ptr,
                scratch.full_query_bf16.ptr,
                rows * self.q_width,
                stream=stream,
                library=cast_library,
                runtime=runtime,
            )

            k_tensor = aotriton_tensor4(
                scratch.key_cache.ptr,
                (1, cfg.head_count_kv, end, cfg.key_length),
                (self.kv_width * end, cfg.key_length, self.kv_width, 1),
                DType.BF16,
            )
            v_tensor = aotriton_tensor4(
                scratch.value_cache.ptr,
                (1, cfg.head_count_kv, end, cfg.key_length),
                (self.kv_width * end, cfg.key_length, self.kv_width, 1),
                DType.BF16,
            )

            q_tensor = aotriton_tensor4(
                scratch.full_query_bf16.ptr,
                (1, cfg.head_count, rows, cfg.key_length),
                (cfg.head_count * cfg.key_length * rows, cfg.key_length, cfg.head_count * cfg.key_length, 1),
                DType.BF16,
            )
            cu_q_tensor = aotriton_tensor1(scratch.cu_q.ptr, (2,), (1,), DType.INT32)
            cu_k_tensor = aotriton_tensor1(scratch.cu_k.ptr, (2,), (1,), DType.INT32)
            lse_tensor = aotriton_tensor2(scratch.softmax_lse.ptr, (cfg.head_count, rows), (rows, 1), DType.FP32)
            out_tensor = aotriton_tensor4(
                scratch.full_attn_bf16.ptr,
                (1, cfg.head_count, rows, cfg.key_length),
                (cfg.head_count * cfg.key_length * rows, cfg.key_length, cfg.head_count * cfg.key_length, 1),
                DType.BF16,
            )
            aotriton_mode = _gguf_aotriton_prefill_mode(scratch.start, rows, end)
            if aotriton_mode == "v2":
                aotriton_attn_fwd_compact_varlen(
                    q_tensor,
                    k_tensor,
                    v_tensor,
                    cu_q_tensor,
                    cu_k_tensor,
                    lse_tensor,
                    out_tensor,
                    max_seqlen_q=rows,
                    max_seqlen_k=end,
                    sm_scale=cfg.key_length ** -0.5,
                    is_causal=True,
                    stream=stream,
                    library=aotriton_library,
                    runtime=runtime,
                )
            else:
                aotriton_attn_fwd_v3_compact_varlen(
                    q_tensor,
                    k_tensor,
                    v_tensor,
                    cu_q_tensor,
                    cu_k_tensor,
                    lse_tensor,
                    out_tensor,
                    persistent_atomic_counter_ptr=scratch.atomic.ptr,
                    max_seqlen_q=rows,
                    max_seqlen_k=end,
                    sm_scale=cfg.key_length ** -0.5,
                    is_causal=True,
                    stream=stream,
                    library=aotriton_library,
                    runtime=runtime,
                )

            qwen35_full_attn_gate_mul_bf16_to_bf16(
                scratch.full_attn_bf16.ptr,
                scratch.full_gate.ptr,
                scratch.full_gated.ptr,
                rows * cfg.head_count * cfg.key_length,
                stream=stream,
                library=paged_attn_library,
                runtime=runtime,
            )
        else:
            qwen35_paged_full_attn_prefill_gqa_gate_bf16_spans(
                scratch.full_query.ptr,
                scratch.key_cache.ptr,
                scratch.value_cache.ptr,
                scratch.full_gate.ptr,
                scratch.full_gated.ptr,
                scratch.prefill_spans,
                rows,
                end,
                scratch.block_size,
                cfg.head_count,
                cfg.head_count_kv,
                cfg.key_length,
                cfg.key_length,
                1,
                cfg.key_length ** -0.5,
                stream=stream,
                library=paged_attn_library,
                runtime=runtime,
            )
        used_aotriton = use_aotriton
        if not _try_launch_dense_q8_single_dp4a(
            layer.weight("attn_output"),
            scratch.full_gated.ptr,
            scratch.attn_out.ptr,
            scratch,
            rows=rows,
            in_features=self.q_width,
            out_features=self.hidden_size,
            stream=stream,
            runtime=runtime,
        ):
            launch_gguf_linear(
                layer.weight("attn_output"),
                scratch.full_gated.ptr,
                scratch.attn_out.ptr,
                rows=rows,
                in_features=self.q_width,
                out_features=self.hidden_size,
                stream=stream,
                runtime=runtime,
            )
        self._run_post_attention_ffn_rows(
            layer_id,
            hidden_ptr,
            scratch.attn_out.ptr,
            out_ptr,
            scratch,
            rows=rows,
            stream=stream,
            expert_sidecar=expert_sidecar,
        )
        return used_aotriton

    def _run_full_attention_decode_batch_layer_rows(
        self,
        layer_id: int,
        hidden_ptr: int,
        out_ptr: int,
        scratch,
        *,
        stream: int = 0,
        expert_sidecar: _DeviceExpertLayerSidecar | None = None,
        stage_timings: dict[str, float] | None = None,
        sync_stage_timings: bool = False,
        stage_prefix: str = "target_block_full_attn",
    ) -> None:
        """Run row-bulk QKV/KV write with the c1-equivalent batch decode attention."""

        assert self.weights is not None
        layer = self.weights.layer(layer_id)
        cfg = self.weights.config
        runtime = self.runtime or get_hip_runtime()
        rows = int(scratch.rows)
        if rows <= 0:
            raise ValueError("rows must be positive")
        if scratch.key_cache is None or scratch.value_cache is None:
            raise RuntimeError("GGUF full-attention decode batch requires cache-backed key/value buffers")
        end = int(scratch.start) + rows
        if end >= 1024:
            raise ValueError("GGUF full-attention decode batch verifier path currently requires context < 1024")
        cast_library = self._cast_library()
        kv_write_library = self._paged_kv_write_library()
        paged_attn_library = self._paged_attn_decode_library()
        sync_stages = bool(sync_stage_timings and stage_timings is not None)
        t_stage = time.perf_counter() if sync_stages else 0.0
        gguf_rmsnorm_bf16_f32_weight(
            hidden_ptr,
            layer.weight("attn_norm").allocation().tensor.ptr,
            scratch.norm.ptr,
            rows=rows,
            hidden_size=self.hidden_size,
            eps=cfg.rms_norm_eps,
            stream=stream,
            runtime=runtime,
        )
        if not _try_launch_dense_q8_triple_dp4a(
            layer.weight("attn_q"),
            layer.weight("attn_k"),
            layer.weight("attn_v"),
            scratch.norm.ptr,
            scratch.full_q.ptr,
            scratch.full_k.ptr,
            scratch.full_v.ptr,
            scratch,
            rows=rows,
            in_features=self.hidden_size,
            out_features_a=2 * self.q_width,
            out_features_b=self.kv_width,
            out_features_c=self.kv_width,
            stream=stream,
            runtime=runtime,
        ) and not launch_gguf_linear_triple(
            layer.weight("attn_q"),
            layer.weight("attn_k"),
            layer.weight("attn_v"),
            scratch.norm.ptr,
            scratch.full_q.ptr,
            scratch.full_k.ptr,
            scratch.full_v.ptr,
            rows=rows,
            in_features=self.hidden_size,
            out_features=2 * self.q_width,
            out_features_b=self.kv_width,
            out_features_c=self.kv_width,
            stream=stream,
            runtime=runtime,
        ):
            launch_gguf_linear(
                layer.weight("attn_q"),
                scratch.norm.ptr,
                scratch.full_q.ptr,
                rows=rows,
                in_features=self.hidden_size,
                out_features=2 * self.q_width,
                stream=stream,
                runtime=runtime,
            )
            if not launch_gguf_linear_pair(
                layer.weight("attn_k"),
                layer.weight("attn_v"),
                scratch.norm.ptr,
                scratch.full_k.ptr,
                scratch.full_v.ptr,
                rows=rows,
                in_features=self.hidden_size,
                out_features=self.kv_width,
                stream=stream,
                runtime=runtime,
            ):
                launch_gguf_linear(
                    layer.weight("attn_k"),
                    scratch.norm.ptr,
                    scratch.full_k.ptr,
                    rows=rows,
                    in_features=self.hidden_size,
                    out_features=self.kv_width,
                    stream=stream,
                    runtime=runtime,
                )
                launch_gguf_linear(
                    layer.weight("attn_v"),
                    scratch.norm.ptr,
                    scratch.full_v.ptr,
                    rows=rows,
                    in_features=self.hidden_size,
                    out_features=self.kv_width,
                    stream=stream,
                    runtime=runtime,
                )
        qwen35_split_qgate_bf16(
            scratch.full_q.ptr,
            scratch.full_query_raw.ptr,
            scratch.full_gate.ptr,
            rows,
            cfg.head_count,
            cfg.key_length,
            stream=stream,
            runtime=runtime,
        )
        bf16_to_f32(
            scratch.full_k.ptr,
            scratch.full_key_raw.ptr,
            rows * self.kv_width,
            stream=stream,
            library=cast_library,
            runtime=runtime,
        )
        t_stage = _mark_sync_stage(
            runtime,
            stage_timings,
            sync_stages,
            f"{stage_prefix}_norm_qkv_split",
            t_stage,
        )
        gguf_qwen35_head_rmsnorm_partial_rotary_positions_f32_weight(
            scratch.full_query_raw.ptr,
            scratch.full_key_raw.ptr,
            layer.weight("attn_q_norm").allocation().tensor.ptr,
            layer.weight("attn_k_norm").allocation().tensor.ptr,
            scratch.cos_table.ptr,
            scratch.sin_table.ptr,
            scratch.positions_tensor.ptr,
            scratch.full_query.ptr,
            scratch.full_key.ptr,
            cfg.rms_norm_eps,
            rows,
            cfg.head_count,
            cfg.head_count_kv,
            cfg.key_length,
            cfg.rope_dimension_count,
            scratch.max_positions,
            stream=stream,
            runtime=runtime,
        )
        t_stage = _mark_sync_stage(
            runtime,
            stage_timings,
            sync_stages,
            f"{stage_prefix}_head_norm_rope",
            t_stage,
        )
        qwen35_write_paged_kv_mixed_value_bf16_prompt_spans(
            scratch.full_key.ptr,
            scratch.full_v.ptr,
            scratch.key_cache.ptr,
            scratch.value_cache.ptr,
            scratch.append_spans,
            rows,
            scratch.block_size,
            cfg.head_count_kv,
            cfg.key_length,
            stream=stream,
            library=kv_write_library,
            runtime=runtime,
        )
        t_stage = _mark_sync_stage(
            runtime,
            stage_timings,
            sync_stages,
            f"{stage_prefix}_kv_write",
            t_stage,
        )
        qwen35_paged_full_attn_decode_context_bf16_batch_c1_exact_spans(
            scratch.full_query.ptr,
            scratch.key_cache.ptr,
            scratch.value_cache.ptr,
            scratch.full_query_raw.ptr,
            scratch.prefill_spans,
            rows,
            end,
            scratch.block_size,
            cfg.head_count,
            cfg.head_count_kv,
            cfg.key_length,
            cfg.key_length ** -0.5,
            stream=stream,
            library=paged_attn_library,
            runtime=runtime,
        )
        t_stage = _mark_sync_stage(
            runtime,
            stage_timings,
            sync_stages,
            f"{stage_prefix}_paged_attn",
            t_stage,
        )
        qwen35_full_attn_gate_mul_bf16(
            scratch.full_query_raw.ptr,
            scratch.full_gate.ptr,
            scratch.full_gated.ptr,
            rows * self.q_width,
            stream=stream,
            library=paged_attn_library,
            runtime=runtime,
        )
        launch_gguf_linear(
            layer.weight("attn_output"),
            scratch.full_gated.ptr,
            scratch.attn_out.ptr,
            rows=rows,
            in_features=self.q_width,
            out_features=self.hidden_size,
            stream=stream,
            runtime=runtime,
        )
        t_stage = _mark_sync_stage(
            runtime,
            stage_timings,
            sync_stages,
            f"{stage_prefix}_gate_output",
            t_stage,
        )
        self._run_post_attention_ffn_rows(
            layer_id,
            hidden_ptr,
            scratch.attn_out.ptr,
            out_ptr,
            scratch,
            rows=rows,
            stream=stream,
            expert_sidecar=expert_sidecar,
            stage_timings=stage_timings,
            sync_stage_timings=sync_stage_timings,
            stage_prefix=f"{stage_prefix}_ffn",
        )

    def _run_linear_attention_layer(self, layer_id: int, hidden_ptr: int, out_ptr: int, scratch, *, stream: int = 0) -> None:
        self._run_linear_attention_attn_only(layer_id, hidden_ptr, scratch.attn_out.ptr, scratch, stream=stream)
        self._run_post_attention_ffn(layer_id, hidden_ptr, scratch.attn_out.ptr, out_ptr, scratch, stream=stream)

    def _run_linear_attention_attn_only(
        self,
        layer_id: int,
        hidden_ptr: int,
        attn_out_ptr: int,
        scratch,
        *,
        stream: int = 0,
    ) -> None:
        assert self.weights is not None
        layer = self.weights.layer(layer_id)
        cfg = self.weights.config
        runtime = self.runtime or get_hip_runtime()
        conv_state = scratch.layer_conv_states[layer_id]
        recurrent_state = scratch.layer_recurrent_states[layer_id]
        if conv_state is None or recurrent_state is None:
            raise ValueError(f"layer {layer_id} has no linear-attention state")
        gguf_rmsnorm_bf16_f32_weight(
            hidden_ptr,
            layer.weight("attn_norm").allocation().tensor.ptr,
            scratch.norm.ptr,
            rows=1,
            hidden_size=self.hidden_size,
            eps=cfg.rms_norm_eps,
            stream=stream,
            runtime=runtime,
        )
        pair_fused = launch_gguf_linear_pair(
            layer.weight("attn_qkv"),
            layer.weight("attn_gate"),
            scratch.norm.ptr,
            scratch.linear_qkv.ptr,
            scratch.linear_z.ptr,
            rows=1,
            in_features=self.hidden_size,
            out_features=self.linear_qkv_width,
            out_features_b=cfg.ssm_inner_size,
            stream=stream,
            runtime=runtime,
        )
        if not pair_fused:
            launch_gguf_linear(
                layer.weight("attn_qkv"),
                scratch.norm.ptr,
                scratch.linear_qkv.ptr,
                rows=1,
                in_features=self.hidden_size,
                out_features=self.linear_qkv_width,
                stream=stream,
                runtime=runtime,
            )
            launch_gguf_linear(
                layer.weight("attn_gate"),
                scratch.norm.ptr,
                scratch.linear_z.ptr,
                rows=1,
                in_features=self.hidden_size,
                out_features=cfg.ssm_inner_size,
                stream=stream,
                runtime=runtime,
            )
        linear_alpha_ptr = scratch.linear_alpha.ptr
        linear_beta_ptr = scratch.linear_beta.ptr
        if cfg.is_moe:
            launch_gguf_linear(
                layer.weight("ssm_alpha"),
                scratch.norm.ptr,
                scratch.linear_alpha.ptr,
                rows=1,
                in_features=self.hidden_size,
                out_features=cfg.ssm_time_step_rank,
                stream=stream,
                runtime=runtime,
            )
            launch_gguf_linear(
                layer.weight("ssm_beta"),
                scratch.norm.ptr,
                scratch.linear_beta.ptr,
                rows=1,
                in_features=self.hidden_size,
                out_features=cfg.ssm_time_step_rank,
                stream=stream,
                runtime=runtime,
            )
        elif not launch_gguf_linear_pair(
            layer.weight("ssm_alpha"),
            layer.weight("ssm_beta"),
            scratch.norm.ptr,
            scratch.linear_alpha.ptr,
            scratch.linear_beta.ptr,
            rows=1,
            in_features=self.hidden_size,
            out_features=cfg.ssm_time_step_rank,
            stream=stream,
            runtime=runtime,
        ):
            launch_gguf_linear(
                layer.weight("ssm_alpha"),
                scratch.norm.ptr,
                scratch.linear_alpha.ptr,
                rows=1,
                in_features=self.hidden_size,
                out_features=cfg.ssm_time_step_rank,
                stream=stream,
                runtime=runtime,
            )
            launch_gguf_linear(
                layer.weight("ssm_beta"),
                scratch.norm.ptr,
                scratch.linear_beta.ptr,
                rows=1,
                in_features=self.hidden_size,
                out_features=cfg.ssm_time_step_rank,
                stream=stream,
                runtime=runtime,
            )
        qwen35_linear_attn_conv_decode_bf16(
            scratch.linear_qkv.ptr,
            conv_state.ptr,
            layer.weight("ssm_conv1d").allocation().tensor.ptr,
            scratch.conv_out.ptr,
            self.linear_qkv_width,
            cfg.ssm_conv_kernel,
            stream=stream,
            runtime=runtime,
        )
        qwen35_gdn_recurrent_rmsnorm_gate_lowp_bf16(
            scratch.conv_out.ptr,
            scratch.linear_z.ptr,
            linear_alpha_ptr,
            linear_beta_ptr,
            layer.weight("ssm_dt_bias").allocation().tensor.ptr,
            layer.weight("ssm_a").allocation().tensor.ptr,
            layer.weight("ssm_norm").allocation().tensor.ptr,
            recurrent_state.ptr,
            scratch.recurrent_out.ptr,
            cfg.rms_norm_eps,
            cfg.ssm_group_count,
            cfg.ssm_time_step_rank,
            cfg.ssm_state_size,
            self.ssm_value_dim,
            stream=stream,
            runtime=runtime,
        )
        launch_gguf_linear(
            layer.weight("ssm_out"),
            scratch.recurrent_out.ptr,
            attn_out_ptr,
            rows=1,
            in_features=cfg.ssm_inner_size,
            out_features=self.hidden_size,
            activation_dtype=GGUF_ACTIVATION_F32,
            stream=stream,
            runtime=runtime,
        )

    def _run_native_attention_bulk_ffn_layer_rows(
        self,
        layer_id: int,
        layer_type: str,
        hidden_ptr: int,
        out_ptr: int,
        scratch,
        *,
        rows: int,
        decode_scratch,
        start_position: int = 0,
        linear_state_rows: tuple[object, object] | None = None,
        stream: int = 0,
    ) -> None:
        """Run row-serial attention followed by the row-bulk GGUF FFN/MoE path.

        This parity-safe scheduler preserves the resident token-serial attention
        kernels/state updates while still exercising the multi-row MoE path. It
        is slower than the fully bulk prefill scheduler, but gives a correctness
        baseline for qwen35moe GGUF bulk MoE work.
        """

        if rows <= 0:
            raise ValueError("rows must be positive")
        start_position = int(start_position)
        if start_position < 0:
            raise ValueError("start_position must be non-negative")
        row_nbytes = self.hidden_size * DType.BF16.itemsize
        runtime = self.runtime or get_hip_runtime()
        for row in range(rows):
            position = start_position + row
            hidden_row = hidden_ptr + row * row_nbytes
            attn_row = scratch.attn_out.ptr + row * row_nbytes
            if layer_type == LINEAR_ATTENTION:
                self._run_linear_attention_attn_only(layer_id, hidden_row, attn_row, decode_scratch, stream=stream)
                if linear_state_rows is not None:
                    conv_state = decode_scratch.layer_conv_states[layer_id]
                    recurrent_state = decode_scratch.layer_recurrent_states[layer_id]
                    if conv_state is None or recurrent_state is None:
                        raise ValueError(f"layer {layer_id} has no linear-attention state")
                    conv_state_rows, recurrent_state_rows = linear_state_rows
                    runtime.memcpy_async(
                        conv_state_rows.ptr + row * int(conv_state.nbytes),
                        conv_state.ptr,
                        int(conv_state.nbytes),
                        HipMemcpyKind.DEVICE_TO_DEVICE,
                        stream,
                    )
                    runtime.memcpy_async(
                        recurrent_state_rows.ptr + row * int(recurrent_state.nbytes),
                        recurrent_state.ptr,
                        int(recurrent_state.nbytes),
                        HipMemcpyKind.DEVICE_TO_DEVICE,
                        stream,
                    )
            elif layer_type == FULL_ATTENTION:
                decode_scratch.set_full_attention_position(position, runtime)
                self._run_full_attention_attn_only(
                    layer_id,
                    hidden_row,
                    attn_row,
                    decode_scratch,
                    position=position,
                    stream=stream,
                )
            else:
                raise ValueError(f"unsupported GGUF layer type {layer_type!r}")
        self._run_post_attention_ffn_rows(layer_id, hidden_ptr, scratch.attn_out.ptr, out_ptr, scratch, rows=rows, stream=stream)

    def _run_linear_attention_prefill_layer_rows(
        self,
        layer_id: int,
        hidden_ptr: int,
        out_ptr: int,
        scratch,
        *,
        rows: int,
        decode_scratch,
        stream: int = 0,
        expert_sidecar: _DeviceExpertLayerSidecar | None = None,
        linear_state_rows: tuple[object, object] | None = None,
        commit_final_linear_state: bool = True,
        stage_timings: dict[str, float] | None = None,
        sync_stage_timings: bool = False,
        stage_prefix: str = "target_block_linear_attn",
    ) -> None:
        assert self.weights is not None
        if rows <= 0:
            raise ValueError("rows must be positive")
        layer = self.weights.layer(layer_id)
        cfg = self.weights.config
        runtime = self.runtime or get_hip_runtime()
        cast_library = self._cast_library()
        conv_state = decode_scratch.layer_conv_states[layer_id]
        recurrent_state = decode_scratch.layer_recurrent_states[layer_id]
        if conv_state is None or recurrent_state is None:
            raise ValueError(f"layer {layer_id} has no linear-attention state")
        sync_stages = bool(sync_stage_timings and stage_timings is not None)
        t_stage = time.perf_counter() if sync_stages else 0.0
        t_norm_qkv_gate_ms = 0.0
        gguf_rmsnorm_bf16_f32_weight(
            hidden_ptr,
            layer.weight("attn_norm").allocation().tensor.ptr,
            scratch.norm.ptr,
            rows=rows,
            hidden_size=self.hidden_size,
            eps=cfg.rms_norm_eps,
            stream=stream,
            runtime=runtime,
        )
        if sync_stages:
            runtime.device_synchronize()
            t_now = time.perf_counter()
            norm_ms = (t_now - t_stage) * 1000
            t_norm_qkv_gate_ms += norm_ms
            _add_sync_stage_timing(stage_timings, f"{stage_prefix}_attn_norm", norm_ms)
            t_stage = time.perf_counter()
        qkv_gate_route = "pair"
        pair_fused = _try_launch_dense_q8_pair_dp4a(
            layer.weight("attn_qkv"),
            layer.weight("attn_gate"),
            scratch.norm.ptr,
            scratch.linear_qkv.ptr,
            scratch.linear_z.ptr,
            scratch,
            rows=rows,
            in_features=self.hidden_size,
            out_features_a=self.linear_qkv_width,
            out_features_b=cfg.ssm_inner_size,
            stream=stream,
            runtime=runtime,
        )
        if pair_fused:
            qkv_gate_route = "dense_q8_dp4a"
        else:
            pair_fused = launch_gguf_linear_pair(
                layer.weight("attn_qkv"),
                layer.weight("attn_gate"),
                scratch.norm.ptr,
                scratch.linear_qkv.ptr,
                scratch.linear_z.ptr,
                rows=rows,
                in_features=self.hidden_size,
                out_features=self.linear_qkv_width,
                out_features_b=cfg.ssm_inner_size,
                stream=stream,
                runtime=runtime,
            )
        if not pair_fused:
            qkv_gate_route = "fallback"
            launch_gguf_linear(
                layer.weight("attn_qkv"),
                scratch.norm.ptr,
                scratch.linear_qkv.ptr,
                rows=rows,
                in_features=self.hidden_size,
                out_features=self.linear_qkv_width,
                stream=stream,
                runtime=runtime,
            )
            launch_gguf_linear(
                layer.weight("attn_gate"),
                scratch.norm.ptr,
                scratch.linear_z.ptr,
                rows=rows,
                in_features=self.hidden_size,
                out_features=cfg.ssm_inner_size,
                stream=stream,
                runtime=runtime,
            )
        if sync_stages:
            runtime.device_synchronize()
            t_now = time.perf_counter()
            qkv_gate_ms = (t_now - t_stage) * 1000
            t_norm_qkv_gate_ms += qkv_gate_ms
            _add_sync_stage_timing(stage_timings, f"{stage_prefix}_attn_qkv_gate_{qkv_gate_route}", qkv_gate_ms)
            _add_sync_stage_timing(stage_timings, f"{stage_prefix}_norm_qkv_gate", t_norm_qkv_gate_ms)
            t_stage = time.perf_counter()
        if cfg.is_moe:
            # The small dense time-step projections feed the recurrent update.
            # Use the registry-dispatched GGUF linear path so qwen35moe's GGUF
            # F32 alpha/beta tensors are consumed as F32, matching llama.cpp's
            # materialized weights while keeping the existing BF16 stream ABI.
            launch_gguf_linear(
                layer.weight("ssm_alpha"),
                scratch.norm.ptr,
                scratch.linear_alpha.ptr,
                rows=rows,
                in_features=self.hidden_size,
                out_features=cfg.ssm_time_step_rank,
                stream=stream,
                runtime=runtime,
            )
            launch_gguf_linear(
                layer.weight("ssm_beta"),
                scratch.norm.ptr,
                scratch.linear_beta.ptr,
                rows=rows,
                in_features=self.hidden_size,
                out_features=cfg.ssm_time_step_rank,
                stream=stream,
                runtime=runtime,
            )
        else:
            launch_gguf_linear(
                layer.weight("ssm_alpha"),
                scratch.norm.ptr,
                scratch.linear_alpha.ptr,
                rows=rows,
                in_features=self.hidden_size,
                out_features=cfg.ssm_time_step_rank,
                stream=stream,
                runtime=runtime,
            )
            launch_gguf_linear(
                layer.weight("ssm_beta"),
                scratch.norm.ptr,
                scratch.linear_beta.ptr,
                rows=rows,
                in_features=self.hidden_size,
                out_features=cfg.ssm_time_step_rank,
                stream=stream,
                runtime=runtime,
            )
        t_stage = _mark_sync_stage(
            runtime,
            stage_timings,
            sync_stages,
            f"{stage_prefix}_alpha_beta",
            t_stage,
        )
        if linear_state_rows is not None:
            conv_state_rows, recurrent_state_rows = linear_state_rows
            qwen35_linear_attn_chain_conv_decode_bf16_tloop(
                scratch.linear_qkv.ptr,
                conv_state.ptr,
                conv_state_rows.ptr,
                layer.weight("ssm_conv1d").allocation().tensor.ptr,
                scratch.conv_out.ptr,
                rows,
                self.linear_qkv_width,
                cfg.ssm_conv_kernel,
                stream=stream,
                runtime=runtime,
            )
            t_stage = _mark_sync_stage(
                runtime,
                stage_timings,
                sync_stages,
                f"{stage_prefix}_chain_conv",
                t_stage,
            )
            qwen35_gdn_chain_recurrent_rmsnorm_gate_lowp_c1_exact_tloop_bf16(
                scratch.conv_out.ptr,
                scratch.linear_z.ptr,
                scratch.linear_alpha.ptr,
                scratch.linear_beta.ptr,
                layer.weight("ssm_dt_bias").allocation().tensor.ptr,
                layer.weight("ssm_a").allocation().tensor.ptr,
                layer.weight("ssm_norm").allocation().tensor.ptr,
                recurrent_state.ptr,
                recurrent_state_rows.ptr,
                scratch.recurrent_out.ptr,
                scratch.recurrent_out.ptr,
                cfg.rms_norm_eps,
                rows,
                cfg.ssm_group_count,
                cfg.ssm_time_step_rank,
                cfg.ssm_state_size,
                self.ssm_value_dim,
                stream=stream,
                runtime=runtime,
            )
            t_stage = _mark_sync_stage(
                runtime,
                stage_timings,
                sync_stages,
                f"{stage_prefix}_chain_gdn",
                t_stage,
            )
            if commit_final_linear_state:
                runtime.memcpy_async(
                    conv_state.ptr,
                    conv_state_rows.ptr + (rows - 1) * int(conv_state.nbytes),
                    int(conv_state.nbytes),
                    HipMemcpyKind.DEVICE_TO_DEVICE,
                    stream,
                )
                runtime.memcpy_async(
                    recurrent_state.ptr,
                    recurrent_state_rows.ptr + (rows - 1) * int(recurrent_state.nbytes),
                    int(recurrent_state.nbytes),
                    HipMemcpyKind.DEVICE_TO_DEVICE,
                    stream,
                )
                t_stage = _mark_sync_stage(
                    runtime,
                    stage_timings,
                    sync_stages,
                    f"{stage_prefix}_final_state_copy",
                    t_stage,
                )
            launch_gguf_linear(
                layer.weight("ssm_out"),
                scratch.recurrent_out.ptr,
                scratch.attn_out.ptr,
                rows=rows,
                in_features=cfg.ssm_inner_size,
                out_features=self.hidden_size,
                activation_dtype=GGUF_ACTIVATION_F32,
                stream=stream,
                runtime=runtime,
            )
            t_stage = _mark_sync_stage(
                runtime,
                stage_timings,
                sync_stages,
                f"{stage_prefix}_ssm_out",
                t_stage,
            )
            self._run_post_attention_ffn_rows(
                layer_id,
                hidden_ptr,
                scratch.attn_out.ptr,
                out_ptr,
                scratch,
                rows=rows,
                stream=stream,
                expert_sidecar=expert_sidecar,
                stage_timings=stage_timings,
                sync_stage_timings=sync_stage_timings,
                stage_prefix=f"{stage_prefix}_ffn",
            )
            return
        bf16_to_f32(
            scratch.linear_qkv.ptr,
            scratch.linear_qkv_f32.ptr,
            rows * self.linear_qkv_width,
            stream=stream,
            library=cast_library,
            runtime=runtime,
        )
        t_stage = _mark_sync_stage(
            runtime,
            stage_timings,
            sync_stages,
            f"{stage_prefix}_qkv_bf16_to_f32",
            t_stage,
        )
        qwen35_linear_attn_conv_prefill_f32(
            scratch.linear_qkv_f32.ptr,
            conv_state.ptr,
            layer.weight("ssm_conv1d").allocation().tensor.ptr,
            scratch.conv_out.ptr,
            rows,
            self.linear_qkv_width,
            cfg.ssm_conv_kernel,
            stream=stream,
            runtime=runtime,
        )
        t_stage = _mark_sync_stage(
            runtime,
            stage_timings,
            sync_stages,
            f"{stage_prefix}_prefill_conv",
            t_stage,
        )
        self._run_gdn_prefill(
            layer=layer,
            scratch=scratch,
            cfg=cfg,
            rows=rows,
            recurrent_state=recurrent_state,
            stream=stream,
            runtime=runtime,
        )
        t_stage = _mark_sync_stage(
            runtime,
            stage_timings,
            sync_stages,
            f"{stage_prefix}_prefill_gdn",
            t_stage,
        )
        if not _try_launch_dense_q8_single_dp4a(
            layer.weight("ssm_out"),
            scratch.recurrent_bf16.ptr,
            scratch.attn_out.ptr,
            scratch,
            rows=rows,
            in_features=cfg.ssm_inner_size,
            out_features=self.hidden_size,
            stream=stream,
            runtime=runtime,
        ):
            launch_gguf_linear(
                layer.weight("ssm_out"),
                scratch.recurrent_bf16.ptr,
                scratch.attn_out.ptr,
                rows=rows,
                in_features=cfg.ssm_inner_size,
                out_features=self.hidden_size,
                stream=stream,
                runtime=runtime,
            )
        t_stage = _mark_sync_stage(
            runtime,
            stage_timings,
            sync_stages,
            f"{stage_prefix}_ssm_out",
            t_stage,
        )
        self._run_post_attention_ffn_rows(
            layer_id,
            hidden_ptr,
            scratch.attn_out.ptr,
            out_ptr,
            scratch,
            rows=rows,
            stream=stream,
            expert_sidecar=expert_sidecar,
            stage_timings=stage_timings,
            sync_stage_timings=sync_stage_timings,
            stage_prefix=f"{stage_prefix}_ffn",
        )

    def _run_full_attention_layer(
        self,
        layer_id: int,
        hidden_ptr: int,
        out_ptr: int,
        scratch,
        *,
        position: int,
        stream: int = 0,
        attention_max_context_len: int | None = None,
    ) -> None:
        self._run_full_attention_attn_only(
            layer_id,
            hidden_ptr,
            scratch.attn_out.ptr,
            scratch,
            position=position,
            stream=stream,
            attention_max_context_len=attention_max_context_len,
        )
        self._run_post_attention_ffn(layer_id, hidden_ptr, scratch.attn_out.ptr, out_ptr, scratch, stream=stream)

    def _run_full_attention_attn_only(
        self,
        layer_id: int,
        hidden_ptr: int,
        attn_out_ptr: int,
        scratch,
        *,
        position: int,
        stream: int = 0,
        attention_max_context_len: int | None = None,
    ) -> None:
        assert self.weights is not None
        layer = self.weights.layer(layer_id)
        cfg = self.weights.config
        runtime = self.runtime or get_hip_runtime()
        cast_library = self._cast_library()
        kv_write_library = self._paged_kv_write_library()
        if int(scratch.position_host[0]) != int(position):
            scratch.set_full_attention_position(position, runtime)
        gguf_rmsnorm_bf16_f32_weight(
            hidden_ptr,
            layer.weight("attn_norm").allocation().tensor.ptr,
            scratch.norm.ptr,
            rows=1,
            hidden_size=self.hidden_size,
            eps=cfg.rms_norm_eps,
            stream=stream,
            runtime=runtime,
        )
        if not launch_gguf_linear_triple(
            layer.weight("attn_q"),
            layer.weight("attn_k"),
            layer.weight("attn_v"),
            scratch.norm.ptr,
            scratch.full_q.ptr,
            scratch.full_k.ptr,
            scratch.full_v.ptr,
            rows=1,
            in_features=self.hidden_size,
            out_features=2 * self.q_width,
            out_features_b=self.kv_width,
            out_features_c=self.kv_width,
            stream=stream,
            runtime=runtime,
        ):
            launch_gguf_linear(
                layer.weight("attn_q"),
                scratch.norm.ptr,
                scratch.full_q.ptr,
                rows=1,
                in_features=self.hidden_size,
                out_features=2 * self.q_width,
                stream=stream,
                runtime=runtime,
            )
            if not launch_gguf_linear_pair(
                layer.weight("attn_k"),
                layer.weight("attn_v"),
                scratch.norm.ptr,
                scratch.full_k.ptr,
                scratch.full_v.ptr,
                rows=1,
                in_features=self.hidden_size,
                out_features=self.kv_width,
                stream=stream,
                runtime=runtime,
            ):
                launch_gguf_linear(
                    layer.weight("attn_k"),
                    scratch.norm.ptr,
                    scratch.full_k.ptr,
                    rows=1,
                    in_features=self.hidden_size,
                    out_features=self.kv_width,
                    stream=stream,
                    runtime=runtime,
                )
                launch_gguf_linear(
                    layer.weight("attn_v"),
                    scratch.norm.ptr,
                    scratch.full_v.ptr,
                    rows=1,
                    in_features=self.hidden_size,
                    out_features=self.kv_width,
                    stream=stream,
                    runtime=runtime,
                )
        qwen35_split_qgate_bf16(
            scratch.full_q.ptr,
            scratch.full_query_raw.ptr,
            scratch.full_gate.ptr,
            1,
            cfg.head_count,
            cfg.key_length,
            stream=stream,
            runtime=runtime,
        )
        bf16_to_f32(
            scratch.full_k.ptr,
            scratch.full_key_raw.ptr,
            self.kv_width,
            stream=stream,
            library=cast_library,
            runtime=runtime,
        )
        gguf_qwen35_head_rmsnorm_partial_rotary_position_f32_weight(
            scratch.full_query_raw.ptr,
            scratch.full_key_raw.ptr,
            layer.weight("attn_q_norm").allocation().tensor.ptr,
            layer.weight("attn_k_norm").allocation().tensor.ptr,
            scratch.cos_table.ptr,
            scratch.sin_table.ptr,
            scratch.position_tensor.ptr,
            scratch.full_query.ptr,
            scratch.full_key.ptr,
            cfg.rms_norm_eps,
            cfg.head_count,
            cfg.head_count_kv,
            cfg.key_length,
            cfg.rope_dimension_count,
            scratch.max_positions,
            stream=stream,
            runtime=runtime,
        )
        key_cache, value_cache = scratch.full_cache(layer_id)
        paged_attn_library = self._paged_attn_decode_library()
        qwen35_write_paged_kv_mixed_value_bf16_spans(
            scratch.full_key.ptr,
            scratch.full_v.ptr,
            key_cache.ptr,
            value_cache.ptr,
            scratch.append_spans,
            scratch.block_size,
            cfg.head_count_kv,
            cfg.key_length,
            stream=stream,
            library=kv_write_library,
            runtime=runtime,
        )
        active_context = int(position) + 1
        attention_context_cap = active_context if attention_max_context_len is None else int(attention_max_context_len)
        if attention_context_cap < active_context:
            raise ValueError("attention_max_context_len must cover the current decode position")
        if attention_context_cap > int(scratch.max_positions):
            raise ValueError("attention_max_context_len exceeds GGUF resident cache capacity")
        if _use_gguf_full_attention_split_decode(attention_context_cap):
            chunk_size = int(scratch.block_size)
            num_splits = min(
                int(scratch.full_attn_split_count),
                max(1, (attention_context_cap + chunk_size - 1) // chunk_size),
            )
            split_gate_fn = _gguf_full_attention_split_gate_bf16_fn(
                cfg,
                block_size=scratch.block_size,
                num_splits=num_splits,
                active_context=attention_context_cap,
            )
            split_gate_fn(
                scratch.full_query.ptr,
                key_cache.ptr,
                value_cache.ptr,
                scratch.full_gate.ptr,
                scratch.full_gated.ptr,
                scratch.full_attn_split_partial.ptr,
                scratch.full_attn_split_m.ptr,
                scratch.full_attn_split_l.ptr,
                scratch.decode_spans,
                chunk_size,
                num_splits,
                scratch.block_size,
                cfg.head_count,
                cfg.head_count_kv,
                cfg.key_length,
                cfg.key_length,
                1,
                cfg.key_length ** -0.5,
                stream=stream,
                library=paged_attn_library,
                runtime=runtime,
            )
        else:
            qwen35_paged_full_attn_decode_context_bf16_spans(
                scratch.full_query.ptr,
                key_cache.ptr,
                value_cache.ptr,
                scratch.full_attn_context.ptr,
                scratch.decode_spans,
                attention_context_cap,
                scratch.block_size,
                cfg.head_count,
                cfg.head_count_kv,
                cfg.key_length,
                cfg.key_length ** -0.5,
                stream=stream,
                library=paged_attn_library,
                runtime=runtime,
            )
            qwen35_full_attn_gate_mul_bf16(
                scratch.full_attn_context.ptr,
                scratch.full_gate.ptr,
                scratch.full_gated.ptr,
                self.q_width,
                stream=stream,
                library=paged_attn_library,
                runtime=runtime,
            )
        launch_gguf_linear(
            layer.weight("attn_output"),
            scratch.full_gated.ptr,
            attn_out_ptr,
            rows=1,
            in_features=self.q_width,
            out_features=self.hidden_size,
            stream=stream,
            runtime=runtime,
        )

    def _run_post_attention_ffn(self, layer_id: int, hidden_ptr: int, attn_out_ptr: int, out_ptr: int, scratch, *, stream: int = 0) -> None:
        self._run_post_attention_ffn_rows(layer_id, hidden_ptr, attn_out_ptr, out_ptr, scratch, rows=1, stream=stream)

    def _run_post_attention_ffn_rows(
        self,
        layer_id: int,
        hidden_ptr: int,
        attn_out_ptr: int,
        out_ptr: int,
        scratch,
        *,
        rows: int,
        stream: int = 0,
        expert_sidecar: _DeviceExpertLayerSidecar | None = None,
        stage_timings: dict[str, float] | None = None,
        sync_stage_timings: bool = False,
        stage_prefix: str = "target_block_ffn",
    ) -> None:
        assert self.weights is not None
        layer = self.weights.layer(layer_id)
        runtime = self.runtime or get_hip_runtime()
        sync_stages = bool(sync_stage_timings and stage_timings is not None)
        t_stage = time.perf_counter() if sync_stages else 0.0
        gguf_add_rmsnorm_bf16_f32_weight(
            hidden_ptr,
            attn_out_ptr,
            layer.weight("post_attention_norm").allocation().tensor.ptr,
            scratch.post_norm.ptr,
            scratch.residual.ptr,
            rows=rows,
            hidden_size=self.hidden_size,
            eps=self.weights.config.rms_norm_eps,
            stream=stream,
            runtime=runtime,
        )
        t_stage = _mark_sync_stage(
            runtime,
            stage_timings,
            sync_stages,
            f"{stage_prefix}_post_norm_residual",
            t_stage,
        )
        if self.weights.config.is_moe:
            if rows == 1:
                self._run_post_attention_moe_c1(layer_id, out_ptr, scratch, stream=stream)
            else:
                self._run_post_attention_moe_rows(
                    layer_id,
                    out_ptr,
                    scratch,
                    rows=rows,
                    stream=stream,
                    expert_sidecar=expert_sidecar,
                    stage_timings=stage_timings,
                    sync_stage_timings=sync_stage_timings,
                    stage_prefix=f"{stage_prefix}_moe",
                )
            return
        if not launch_gguf_linear_pair(
            layer.weight("ffn_gate"),
            layer.weight("ffn_up"),
            scratch.post_norm.ptr,
            scratch.ffn_gate_up.ptr,
            scratch.ffn_gate_up.ptr + self.ffn_size * rows * 2,
            rows=rows,
            in_features=self.hidden_size,
            out_features=self.ffn_size,
            stream=stream,
            runtime=runtime,
        ):
            launch_gguf_linear(
                layer.weight("ffn_gate"),
                scratch.post_norm.ptr,
                scratch.ffn_gate_up.ptr,
                rows=rows,
                in_features=self.hidden_size,
                out_features=self.ffn_size,
                stream=stream,
                runtime=runtime,
            )
            launch_gguf_linear(
                layer.weight("ffn_up"),
                scratch.post_norm.ptr,
                scratch.ffn_gate_up.ptr + self.ffn_size * rows * 2,
                rows=rows,
                in_features=self.hidden_size,
                out_features=self.ffn_size,
                stream=stream,
                runtime=runtime,
            )
        t_stage = _mark_sync_stage(
            runtime,
            stage_timings,
            sync_stages,
            f"{stage_prefix}_gate_up",
            t_stage,
        )
        silu_mul_separate_out_bf16(
            scratch.ffn_gate_up.ptr,
            scratch.ffn_gate_up.ptr + self.ffn_size * rows * 2,
            scratch.ffn_intermediate.ptr,
            rows=rows,
            features=self.ffn_size,
            stream=stream,
            runtime=runtime,
        )
        t_stage = _mark_sync_stage(
            runtime,
            stage_timings,
            sync_stages,
            f"{stage_prefix}_silu",
            t_stage,
        )
        launch_gguf_linear(
            layer.weight("ffn_down"),
            scratch.ffn_intermediate.ptr,
            scratch.ffn_down.ptr,
            rows=rows,
            in_features=self.ffn_size,
            out_features=self.hidden_size,
            stream=stream,
            runtime=runtime,
        )
        gguf_bf16_add(
            scratch.residual.ptr,
            scratch.ffn_down.ptr,
            out_ptr,
            rows * self.hidden_size,
            stream=stream,
            runtime=runtime,
        )
        _mark_sync_stage(
            runtime,
            stage_timings,
            sync_stages,
            f"{stage_prefix}_down_residual",
            t_stage,
        )

    def _run_post_attention_moe_c1(self, layer_id: int, out_ptr: int, scratch, *, stream: int = 0) -> None:
        assert self.weights is not None
        cfg = self.weights.config
        if not cfg.is_moe:
            raise ValueError("MoE path requires qwen35moe GGUF config")
        layer = self.weights.layer(layer_id)
        runtime = self.runtime or get_hip_runtime()
        top_k = int(cfg.expert_used_count)
        if top_k <= 0:
            raise ValueError("qwen35moe GGUF expert_used_count must be positive")
        if top_k > scratch.moe_selected_host.shape[0]:
            raise ValueError("qwen35moe scratch top-k capacity is too small")

        # llama.cpp keeps the qwen35moe router and shared-gate tensors in F32.
        # Compute expert logits and the adjacent shared-gate logit separately via
        # the registry-resolved router adapter so GGUF F32 weights do not get
        # silently contracted to BF16 on the correctness-first decode path.
        _launch_qwen35_router_logits_bf16_hidden(
            scratch.post_norm.ptr,
            layer.weight("ffn_gate_inp"),
            scratch.moe_router_logits.ptr,
            1,
            self.hidden_size,
            cfg.expert_count,
            stream=stream,
            runtime=runtime,
        )
        _launch_qwen35_router_logits_bf16_hidden(
            scratch.post_norm.ptr,
            layer.weight("ffn_gate_inp_shexp"),
            scratch.moe_router_logits.ptr + cfg.expert_count * DType.FP32.itemsize,
            1,
            self.hidden_size,
            1,
            stream=stream,
            runtime=runtime,
        )
        qwen35_router_select(
            scratch.moe_router_logits.ptr,
            scratch.moe_selected_experts.ptr,
            scratch.moe_routing_weights.ptr,
            1,
            cfg.expert_count,
            cfg.expert_count,
            top_k,
            threads=256,
            stream=stream,
            runtime=runtime,
        )

        gate_weight = layer.weight("ffn_gate_exps")
        up_weight = layer.weight("ffn_up_exps")
        down_weight = layer.weight("ffn_down_exps")
        if (
            _env_flag(_GGUF_COMPACT_MOE_C1_ENV, False)
            and _try_run_post_attention_moe_c1_compact_gemv(
                self,
                layer,
                gate_weight,
                up_weight,
                down_weight,
                out_ptr,
                scratch,
                top_k=top_k,
                stream=stream,
                runtime=runtime,
            )
        ):
            return
        selected_rows = top_k
        if not (
            _env_flag(_GGUF_FUSED_MOE_FFN_ENV, False)
            and _try_run_post_attention_moe_c1_fused_ffn(
                self,
                layer,
                gate_weight,
                up_weight,
                down_weight,
                scratch,
                top_k=top_k,
                stream=stream,
                runtime=runtime,
            )
        ):
            self._run_post_attention_moe_c1_unfused_selected_ffn(
                gate_weight,
                up_weight,
                down_weight,
                scratch,
                selected_rows=selected_rows,
                stream=stream,
                runtime=runtime,
            )

        if launch_gguf_linear_pair_concat(
            layer.weight("ffn_gate_shexp"),
            layer.weight("ffn_up_shexp"),
            scratch.post_norm.ptr,
            scratch.ffn_gate_up.ptr,
            rows=1,
            in_features=self.hidden_size,
            out_features=cfg.expert_shared_feed_forward_length,
            stream=stream,
            runtime=runtime,
        ):
            silu_mul_dual_out_bf16(
                scratch.ffn_gate_up.ptr,
                scratch.moe_shared_intermediate.ptr,
                rows=1,
                features=cfg.expert_shared_feed_forward_length,
                stream=stream,
                runtime=runtime,
            )
        else:
            if not launch_gguf_linear_pair(
                layer.weight("ffn_gate_shexp"),
                layer.weight("ffn_up_shexp"),
                scratch.post_norm.ptr,
                scratch.moe_shared_gate.ptr,
                scratch.moe_shared_up.ptr,
                rows=1,
                in_features=self.hidden_size,
                out_features=cfg.expert_shared_feed_forward_length,
                stream=stream,
                runtime=runtime,
            ):
                launch_gguf_linear(
                    layer.weight("ffn_gate_shexp"),
                    scratch.post_norm.ptr,
                    scratch.moe_shared_gate.ptr,
                    rows=1,
                    in_features=self.hidden_size,
                    out_features=cfg.expert_shared_feed_forward_length,
                    stream=stream,
                    runtime=runtime,
                )
                launch_gguf_linear(
                    layer.weight("ffn_up_shexp"),
                    scratch.post_norm.ptr,
                    scratch.moe_shared_up.ptr,
                    rows=1,
                    in_features=self.hidden_size,
                    out_features=cfg.expert_shared_feed_forward_length,
                    stream=stream,
                    runtime=runtime,
                )
            silu_mul_separate_out_bf16(
                scratch.moe_shared_gate.ptr,
                scratch.moe_shared_up.ptr,
                scratch.moe_shared_intermediate.ptr,
                rows=1,
                features=cfg.expert_shared_feed_forward_length,
                stream=stream,
                runtime=runtime,
            )
        launch_gguf_linear(
            layer.weight("ffn_down_shexp"),
            scratch.moe_shared_intermediate.ptr,
            scratch.moe_shared_out.ptr,
            rows=1,
            in_features=cfg.expert_shared_feed_forward_length,
            out_features=self.hidden_size,
            stream=stream,
            runtime=runtime,
        )
        weighted_sum_shared_gate_combine_residual_out_bf16_f32w(
            scratch.moe_down_out.ptr,
            scratch.moe_routing_weights.ptr,
            scratch.moe_shared_out.ptr,
            scratch.moe_router_logits.ptr + cfg.expert_count * 4,
            scratch.residual.ptr,
            out_ptr,
            top_k,
            self.hidden_size,
            stream=stream,
            runtime=runtime,
        )

    def _run_post_attention_moe_c1_unfused_selected_ffn(
        self,
        gate_weight,
        up_weight,
        down_weight,
        scratch,
        *,
        selected_rows: int,
        stream: int,
        runtime: HipRuntime,
    ) -> None:
        """Unfused selected-expert FFN: gate_up GEMV -> silu*mul -> down GEMV into
        ``scratch.moe_down_out``. Numerically-equivalent fallback for the fused B1
        megakernel (architectural invariant), and the default rows==1 path."""
        cfg = self.weights.config
        gate_rows_nbytes = selected_rows * cfg.expert_feed_forward_length * DType.BF16.itemsize
        expert_silu_ready = _launch_selected_raw_gguf_moe_pair_silu(
            gate_weight,
            up_weight,
            scratch.post_norm.ptr,
            scratch.moe_selected_experts.ptr,
            scratch.ffn_intermediate.ptr,
            x_rows=1,
            rows=selected_rows,
            num_experts=cfg.expert_count,
            in_features=self.hidden_size,
            out_features=cfg.expert_feed_forward_length,
            stream=stream,
            runtime=runtime,
        )
        if not expert_silu_ready:
            q8_1_workspace_ptr = _optional_q8_1_workspace_ptr(
                scratch,
                1,
                self.hidden_size,
                enabled=(
                    _gguf_q4k_selected_dual_dp4a_enabled()
                    or _gguf_t16_selected_dp4a_enabled()
                    or _gguf_raw_selected_dp4a_enabled()
                    or _selected_pair_requires_q8_1_input(gate_weight, up_weight)
                ),
            )
            if not _launch_selected_raw_gguf_moe_pair(
                gate_weight,
                up_weight,
                scratch.post_norm.ptr,
                scratch.moe_selected_experts.ptr,
                scratch.ffn_gate_up.ptr,
                scratch.ffn_gate_up.ptr + gate_rows_nbytes,
                x_rows=1,
                rows=selected_rows,
                num_experts=cfg.expert_count,
                in_features=self.hidden_size,
                out_features=cfg.expert_feed_forward_length,
                q8_1_workspace_ptr=q8_1_workspace_ptr,
                stream=stream,
                runtime=runtime,
            ):
                _launch_selected_raw_gguf_moe_linear(
                    gate_weight,
                    scratch.post_norm.ptr,
                    scratch.moe_selected_experts.ptr,
                    scratch.ffn_gate_up.ptr,
                    x_rows=1,
                    rows=selected_rows,
                    num_experts=cfg.expert_count,
                    in_features=self.hidden_size,
                    out_features=cfg.expert_feed_forward_length,
                    stream=stream,
                    runtime=runtime,
                )
                _launch_selected_raw_gguf_moe_linear(
                    up_weight,
                    scratch.post_norm.ptr,
                    scratch.moe_selected_experts.ptr,
                    scratch.ffn_gate_up.ptr + gate_rows_nbytes,
                    x_rows=1,
                    rows=selected_rows,
                    num_experts=cfg.expert_count,
                    in_features=self.hidden_size,
                    out_features=cfg.expert_feed_forward_length,
                    stream=stream,
                    runtime=runtime,
                )
            silu_mul_separate_out_bf16(
                scratch.ffn_gate_up.ptr,
                scratch.ffn_gate_up.ptr + gate_rows_nbytes,
                scratch.ffn_intermediate.ptr,
                rows=selected_rows,
                features=cfg.expert_feed_forward_length,
                stream=stream,
                runtime=runtime,
            )
        _launch_selected_raw_gguf_moe_linear(
            down_weight,
            scratch.ffn_intermediate.ptr,
            scratch.moe_selected_experts.ptr,
            scratch.moe_down_out.ptr,
            x_rows=selected_rows,
            rows=selected_rows,
            num_experts=cfg.expert_count,
            in_features=cfg.expert_feed_forward_length,
            out_features=self.hidden_size,
            q8_1_workspace_ptr=_optional_q8_1_workspace_ptr(
                scratch,
                selected_rows,
                cfg.expert_feed_forward_length,
                enabled=(
                    _gguf_t16_selected_dp4a_enabled()
                    or _gguf_raw_selected_dp4a_enabled()
                    or _selected_gemv_requires_q8_1_input(down_weight)
                ),
            ),
            stream=stream,
            runtime=runtime,
        )

    def _run_post_attention_moe_rows(
        self,
        layer_id: int,
        out_ptr: int,
        scratch,
        *,
        rows: int,
        stream: int = 0,
        expert_sidecar: _DeviceExpertLayerSidecar | None = None,
        stage_timings: dict[str, float] | None = None,
        sync_stage_timings: bool = False,
        stage_prefix: str = "target_block_ffn_moe",
    ) -> None:
        assert self.weights is not None
        cfg = self.weights.config
        if not cfg.is_moe:
            raise ValueError("MoE path requires qwen35moe GGUF config")
        if rows <= 1:
            raise ValueError("bulk MoE rows path requires rows > 1")
        if not hasattr(scratch, "moe_shared_gate_logits"):
            raise ValueError("qwen35moe bulk MoE scratch is missing shared-gate logits")
        layer = self.weights.layer(layer_id)
        runtime = self.runtime or get_hip_runtime()
        sync_stages = bool(sync_stage_timings and stage_timings is not None)
        t_stage = time.perf_counter() if sync_stages else 0.0
        top_k = int(cfg.expert_used_count)
        if top_k <= 0:
            raise ValueError("qwen35moe GGUF expert_used_count must be positive")
        selected_rows = rows * top_k

        _launch_qwen35_router_logits_bf16_hidden(
            scratch.post_norm.ptr,
            layer.weight("ffn_gate_inp"),
            scratch.moe_router_logits.ptr,
            rows,
            self.hidden_size,
            cfg.expert_count,
            stream=stream,
            runtime=runtime,
        )
        _launch_qwen35_router_logits_bf16_hidden(
            scratch.post_norm.ptr,
            layer.weight("ffn_gate_inp_shexp"),
            scratch.moe_shared_gate_logits.ptr,
            rows,
            self.hidden_size,
            1,
            stream=stream,
            runtime=runtime,
        )
        qwen35_router_select(
            scratch.moe_router_logits.ptr,
            scratch.moe_selected_experts.ptr,
            scratch.moe_routing_weights.ptr,
            rows,
            cfg.expert_count,
            cfg.expert_count,
            top_k,
            stream=stream,
            runtime=runtime,
        )
        t_stage = _mark_sync_stage(
            runtime,
            stage_timings,
            sync_stages,
            f"{stage_prefix}_router",
            t_stage,
        )

        gate_weight = layer.weight("ffn_gate_exps")
        up_weight = layer.weight("ffn_up_exps")
        down_weight = layer.weight("ffn_down_exps")
        if _try_run_post_attention_moe_rows_compact_wmma(
            self,
            layer,
            gate_weight,
            up_weight,
            down_weight,
            out_ptr,
            scratch,
            rows=rows,
            selected_rows=selected_rows,
            top_k=top_k,
            stream=stream,
            runtime=runtime,
        ):
            _mark_sync_stage(
                runtime,
                stage_timings,
                sync_stages,
                f"{stage_prefix}_compact_wmma",
                t_stage,
            )
            return
        if _gguf_row_compact_gemv_enabled() and _try_run_post_attention_moe_rows_compact_gemv(
            self,
            layer,
            gate_weight,
            up_weight,
            down_weight,
            out_ptr,
            scratch,
            rows=rows,
            selected_rows=selected_rows,
            top_k=top_k,
            stream=stream,
            runtime=runtime,
        ):
            _mark_sync_stage(
                runtime,
                stage_timings,
                sync_stages,
                f"{stage_prefix}_compact_gemv",
                t_stage,
            )
            return
        gate_rows_nbytes = selected_rows * cfg.expert_feed_forward_length * DType.BF16.itemsize
        expert_silu_ready = False
        if expert_sidecar is not None and _launch_selected_expert_pack8_moe_pair(
            expert_sidecar.tensor("ffn_gate_exps"),
            expert_sidecar.tensor("ffn_up_exps"),
            scratch.post_norm.ptr,
            scratch.moe_selected_experts.ptr,
            scratch.ffn_gate_up.ptr,
            scratch.ffn_gate_up.ptr + gate_rows_nbytes,
            x_rows=rows,
            rows=selected_rows,
            num_experts=cfg.expert_count,
            in_features=self.hidden_size,
            out_features=cfg.expert_feed_forward_length,
            stream=stream,
            runtime=runtime,
            library=getattr(self, "_expert_pack8_library", None),
        ):
            pass
        elif expert_sidecar is not None:
            _launch_selected_expert_pack8_moe_linear(
                expert_sidecar.tensor("ffn_gate_exps"),
                scratch.post_norm.ptr,
                scratch.moe_selected_experts.ptr,
                scratch.ffn_gate_up.ptr,
                x_rows=rows,
                rows=selected_rows,
                num_experts=cfg.expert_count,
                in_features=self.hidden_size,
                out_features=cfg.expert_feed_forward_length,
                stream=stream,
                runtime=runtime,
                library=getattr(self, "_expert_pack8_library", None),
            )
            _launch_selected_expert_pack8_moe_linear(
                expert_sidecar.tensor("ffn_up_exps"),
                scratch.post_norm.ptr,
                scratch.moe_selected_experts.ptr,
                scratch.ffn_gate_up.ptr + gate_rows_nbytes,
                x_rows=rows,
                rows=selected_rows,
                num_experts=cfg.expert_count,
                in_features=self.hidden_size,
                out_features=cfg.expert_feed_forward_length,
                stream=stream,
                runtime=runtime,
                library=getattr(self, "_expert_pack8_library", None),
            )
        else:
            # The Q4T16 dual+SiLU fusion is decode-only for now.  In rows>1
            # bulk prefill the extra exp/rounding work in the GEMV accumulator
            # did not pay for the removed SiLU launch, so keep the split path.
            q8_1_workspace_ptr = _optional_q8_1_workspace_ptr(
                scratch,
                rows,
                self.hidden_size,
                enabled=(
                    _gguf_q4k_selected_dual_dp4a_enabled()
                    or _gguf_t16_selected_dp4a_enabled()
                    or _gguf_raw_selected_dp4a_enabled()
                    or _selected_pair_requires_q8_1_input(gate_weight, up_weight)
                ),
            )
            if not _launch_selected_raw_gguf_moe_pair(
                gate_weight,
                up_weight,
                scratch.post_norm.ptr,
                scratch.moe_selected_experts.ptr,
                scratch.ffn_gate_up.ptr,
                scratch.ffn_gate_up.ptr + gate_rows_nbytes,
                x_rows=rows,
                rows=selected_rows,
                num_experts=cfg.expert_count,
                in_features=self.hidden_size,
                out_features=cfg.expert_feed_forward_length,
                q8_1_workspace_ptr=q8_1_workspace_ptr,
                stream=stream,
                runtime=runtime,
                stage_timings=stage_timings,
                sync_stage_timings=sync_stages,
                stage_prefix=f"{stage_prefix}_expert_gate_up",
            ):
                _launch_selected_raw_gguf_moe_linear(
                    gate_weight,
                    scratch.post_norm.ptr,
                    scratch.moe_selected_experts.ptr,
                    scratch.ffn_gate_up.ptr,
                    x_rows=rows,
                    rows=selected_rows,
                    num_experts=cfg.expert_count,
                    in_features=self.hidden_size,
                    out_features=cfg.expert_feed_forward_length,
                    stream=stream,
                    runtime=runtime,
                )
                _launch_selected_raw_gguf_moe_linear(
                    up_weight,
                    scratch.post_norm.ptr,
                    scratch.moe_selected_experts.ptr,
                    scratch.ffn_gate_up.ptr + gate_rows_nbytes,
                    x_rows=rows,
                    rows=selected_rows,
                    num_experts=cfg.expert_count,
                    in_features=self.hidden_size,
                    out_features=cfg.expert_feed_forward_length,
                    stream=stream,
                    runtime=runtime,
                )
        t_stage = _mark_sync_stage(
            runtime,
            stage_timings,
            sync_stages,
            f"{stage_prefix}_expert_gate_up",
            t_stage,
        )
        if not expert_silu_ready:
            silu_mul_separate_out_bf16(
                scratch.ffn_gate_up.ptr,
                scratch.ffn_gate_up.ptr + gate_rows_nbytes,
                scratch.ffn_intermediate.ptr,
                rows=selected_rows,
                features=cfg.expert_feed_forward_length,
                stream=stream,
                runtime=runtime,
            )
            t_stage = _mark_sync_stage(
                runtime,
                stage_timings,
                sync_stages,
                f"{stage_prefix}_expert_silu",
                t_stage,
            )
        if expert_sidecar is not None:
            _launch_selected_expert_pack8_moe_linear(
                expert_sidecar.tensor("ffn_down_exps"),
                scratch.ffn_intermediate.ptr,
                scratch.moe_selected_experts.ptr,
                scratch.moe_down_out.ptr,
                x_rows=selected_rows,
                rows=selected_rows,
                num_experts=cfg.expert_count,
                in_features=cfg.expert_feed_forward_length,
                out_features=self.hidden_size,
                stream=stream,
                runtime=runtime,
                library=getattr(self, "_expert_pack8_library", None),
            )
        else:
            _launch_selected_raw_gguf_moe_linear(
                down_weight,
                scratch.ffn_intermediate.ptr,
                scratch.moe_selected_experts.ptr,
                scratch.moe_down_out.ptr,
                x_rows=selected_rows,
                rows=selected_rows,
                num_experts=cfg.expert_count,
                in_features=cfg.expert_feed_forward_length,
                out_features=self.hidden_size,
                q8_1_workspace_ptr=_optional_q8_1_workspace_ptr(
                    scratch,
                    selected_rows,
                    cfg.expert_feed_forward_length,
                    enabled=(
                        _gguf_t16_selected_dp4a_enabled()
                        or _gguf_raw_selected_dp4a_enabled()
                        or _selected_gemv_requires_q8_1_input(down_weight)
                    ),
                ),
                stream=stream,
                runtime=runtime,
                stage_timings=stage_timings,
                sync_stage_timings=sync_stages,
                stage_prefix=f"{stage_prefix}_expert_down",
            )
        t_stage = _mark_sync_stage(
            runtime,
            stage_timings,
            sync_stages,
            f"{stage_prefix}_expert_down",
            t_stage,
        )

        shared_q8_dp4a_enabled = _gguf_dense_q8_dp4a_shared_enabled()
        shared_gate_up_dp4a = shared_q8_dp4a_enabled and _try_launch_dense_q8_pair_dp4a(
            layer.weight("ffn_gate_shexp"),
            layer.weight("ffn_up_shexp"),
            scratch.post_norm.ptr,
            scratch.moe_shared_gate.ptr,
            scratch.moe_shared_up.ptr,
            scratch,
            rows=rows,
            in_features=self.hidden_size,
            out_features_a=cfg.expert_shared_feed_forward_length,
            out_features_b=cfg.expert_shared_feed_forward_length,
            stream=stream,
            runtime=runtime,
        )
        if shared_gate_up_dp4a:
            silu_mul_separate_out_bf16(
                scratch.moe_shared_gate.ptr,
                scratch.moe_shared_up.ptr,
                scratch.moe_shared_intermediate.ptr,
                rows=rows,
                features=cfg.expert_shared_feed_forward_length,
                stream=stream,
                runtime=runtime,
            )
        elif launch_gguf_linear_pair_concat(
            layer.weight("ffn_gate_shexp"),
            layer.weight("ffn_up_shexp"),
            scratch.post_norm.ptr,
            scratch.ffn_gate_up.ptr,
            rows=rows,
            in_features=self.hidden_size,
            out_features=cfg.expert_shared_feed_forward_length,
            stream=stream,
            runtime=runtime,
        ):
            silu_mul_dual_out_bf16(
                scratch.ffn_gate_up.ptr,
                scratch.moe_shared_intermediate.ptr,
                rows=rows,
                features=cfg.expert_shared_feed_forward_length,
                stream=stream,
                runtime=runtime,
            )
        else:
            if not launch_gguf_linear_pair(
                layer.weight("ffn_gate_shexp"),
                layer.weight("ffn_up_shexp"),
                scratch.post_norm.ptr,
                scratch.moe_shared_gate.ptr,
                scratch.moe_shared_up.ptr,
                rows=rows,
                in_features=self.hidden_size,
                out_features=cfg.expert_shared_feed_forward_length,
                stream=stream,
                runtime=runtime,
            ):
                launch_gguf_linear(
                    layer.weight("ffn_gate_shexp"),
                    scratch.post_norm.ptr,
                    scratch.moe_shared_gate.ptr,
                    rows=rows,
                    in_features=self.hidden_size,
                    out_features=cfg.expert_shared_feed_forward_length,
                    stream=stream,
                    runtime=runtime,
                )
                launch_gguf_linear(
                    layer.weight("ffn_up_shexp"),
                    scratch.post_norm.ptr,
                    scratch.moe_shared_up.ptr,
                    rows=rows,
                    in_features=self.hidden_size,
                    out_features=cfg.expert_shared_feed_forward_length,
                    stream=stream,
                    runtime=runtime,
                )
            silu_mul_separate_out_bf16(
                scratch.moe_shared_gate.ptr,
                scratch.moe_shared_up.ptr,
                scratch.moe_shared_intermediate.ptr,
                rows=rows,
                features=cfg.expert_shared_feed_forward_length,
                stream=stream,
                runtime=runtime,
            )
        t_stage = _mark_sync_stage(
            runtime,
            stage_timings,
            sync_stages,
            f"{stage_prefix}_shared_gate_up_silu",
            t_stage,
        )
        if not (
            shared_q8_dp4a_enabled and _try_launch_dense_q8_single_dp4a(
                layer.weight("ffn_down_shexp"),
                scratch.moe_shared_intermediate.ptr,
                scratch.moe_shared_out.ptr,
                scratch,
                rows=rows,
                in_features=cfg.expert_shared_feed_forward_length,
                out_features=self.hidden_size,
                stream=stream,
                runtime=runtime,
            )
        ):
            launch_gguf_linear(
                layer.weight("ffn_down_shexp"),
                scratch.moe_shared_intermediate.ptr,
                scratch.moe_shared_out.ptr,
                rows=rows,
                in_features=cfg.expert_shared_feed_forward_length,
                out_features=self.hidden_size,
                stream=stream,
                runtime=runtime,
            )
        t_stage = _mark_sync_stage(
            runtime,
            stage_timings,
            sync_stages,
            f"{stage_prefix}_shared_down",
            t_stage,
        )
        weighted_sum_shared_gate_combine_residual_batch_out_bf16_f32w(
            scratch.moe_down_out.ptr,
            scratch.moe_routing_weights.ptr,
            scratch.moe_shared_out.ptr,
            scratch.moe_shared_gate_logits.ptr,
            scratch.residual.ptr,
            out_ptr,
            rows,
            top_k,
            self.hidden_size,
            1,
            stream=stream,
            runtime=runtime,
        )
        _mark_sync_stage(
            runtime,
            stage_timings,
            sync_stages,
            f"{stage_prefix}_combine_residual",
            t_stage,
        )

    def close(self) -> None:
        if self.weights is not None:
            self.weights.free(runtime=self.runtime)
            self.weights = None

    def __enter__(self) -> "Qwen35GGUFFullStackRunner":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


_QWEN35MOE_UNSAFE_FASTPATH_ENV = "HIPENGINE_GGUF_ALLOW_UNSAFE_QWEN35MOE_FASTPATHS"
_GGUF_AOTRITON_PREFILL_ENV = "HIPENGINE_GGUF_AOTRITON_PREFILL"
_GGUF_FULL_ATTN_DECODE_SPLIT_MIN_CONTEXT_ENV = "HIPENGINE_GGUF_FULL_ATTN_DECODE_PAGED_MIN_CONTEXT"
_GGUF_FULL_ATTN_DECODE_SPLIT_MIN_CONTEXT_DEFAULT = 1024
_GGUF_COMPACT_MOE_C1_ENV = "HIPENGINE_GGUF_COMPACT_MOE_C1"
# B2: opt-in fused selected-expert MoE FFN megakernel for rows==1 raw-Q4_K decode.
_GGUF_FUSED_MOE_FFN_ENV = "HIPENGINE_GGUF_FUSED_MOE_FFN"
_GGUF_Q4K_SELECTED_DUAL_DP4A_ENV = "HIPENGINE_GGUF_Q4K_SELECTED_DUAL_DP4A"
_GGUF_T16_SELECTED_DP4A_ENV = "HIPENGINE_GGUF_T16_SELECTED_DP4A"
_GGUF_RAW_SELECTED_DP4A_ENV = "HIPENGINE_GGUF_RAW_SELECTED_DP4A"
_GGUF_DENSE_Q8_DP4A_ENV = "HIPENGINE_GGUF_DENSE_Q8_DP4A"
_GGUF_DENSE_Q8_DP4A_ALL_ENV = "HIPENGINE_GGUF_DENSE_Q8_DP4A_ALL"
_GGUF_DENSE_Q8_DP4A_SHARED_ENV = "HIPENGINE_GGUF_DENSE_Q8_DP4A_SHARED"
_GGUF_ROW_COMPACT_GEMV_ENV = "HIPENGINE_GGUF_ROW_COMPACT_GEMV"
_GGUF_VERIFY_ROW_LM_HEAD_ENV = "HIPENGINE_GGUF_VERIFY_ROW_LM_HEAD"
_GGUF_VERIFY_LM_HEAD_Q6_TOP1_DP4A_ENV = "HIPENGINE_GGUF_VERIFY_LM_HEAD_Q6_TOP1_DP4A"
_GGUF_MOE_GRAPH_ENV = "HIPENGINE_GGUF_MOE_GRAPH"
_Q8_1_BLOCK = 32
_Q8_1_BLOCK_BYTES = 36


@dataclass(frozen=True)
class Qwen35GGUFFastPathSafety:
    """Effective qwen35moe GGUF fast-path state after correctness gating."""

    is_qwen35moe: bool
    allow_unsafe_qwen35moe_fastpaths: bool
    requested_wmma_prefill: bool
    requested_gemv_decode: bool
    effective_wmma_prefill: bool
    effective_gemv_decode: bool
    disabled_wmma_prefill: bool
    disabled_gemv_decode: bool
    reason: str | None = None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _env_value(name: str, *aliases: str) -> str | None:
    for key in (name, *aliases):
        raw = os.environ.get(key)
        if raw is not None and raw.strip() != "":
            return raw.strip()
    return None


def _env_truthy(name: str) -> bool:
    raw = os.environ.get(name, "")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_flag(name: str, default: bool, *aliases: str) -> bool:
    raw = _env_value(name, *aliases)
    if raw is None:
        return default
    return raw.lower() not in {"0", "false", "off", "no"}


def _env_int(name: str, default: int, *aliases: str) -> int:
    raw = _env_value(name, *aliases)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _gguf_q4k_selected_dual_dp4a_enabled() -> bool:
    return _env_flag(_GGUF_Q4K_SELECTED_DUAL_DP4A_ENV, False)


def _gguf_t16_selected_dp4a_enabled() -> bool:
    return _env_flag(_GGUF_T16_SELECTED_DP4A_ENV, False)


def _gguf_raw_selected_dp4a_enabled() -> bool:
    return _env_flag(_GGUF_RAW_SELECTED_DP4A_ENV, False)


def _gguf_dense_q8_dp4a_enabled() -> bool:
    return _env_flag(_GGUF_DENSE_Q8_DP4A_ENV, False) or _gguf_dense_q8_dp4a_all_enabled()


def _gguf_dense_q8_dp4a_all_enabled() -> bool:
    return _env_flag(_GGUF_DENSE_Q8_DP4A_ALL_ENV, False)


def _gguf_dense_q8_dp4a_shared_enabled() -> bool:
    return _env_flag(_GGUF_DENSE_Q8_DP4A_SHARED_ENV, False)


def _gguf_row_compact_gemv_enabled() -> bool:
    return _env_flag(_GGUF_ROW_COMPACT_GEMV_ENV, False)


def _gguf_verify_row_lm_head_enabled() -> bool:
    return _env_flag(_GGUF_VERIFY_ROW_LM_HEAD_ENV, False)


def _gguf_verify_lm_head_q6_top1_dp4a_enabled() -> bool:
    return _env_flag(_GGUF_VERIFY_LM_HEAD_Q6_TOP1_DP4A_ENV, False)


def _gguf_moe_graph_enabled() -> bool:
    # Default off: per-layer MoE FFN graph capture/replay for the rows==1 resident
    # decode path. Experimental launch-count-reduction probe (task #15); promote to
    # default only after the B3 acceptance + AR-tok/s gate (docs/REFACTOR.md).
    return _env_flag(_GGUF_MOE_GRAPH_ENV, False)


def _q8_1_workspace_bytes(rows: int, in_features: int) -> int:
    rows = int(rows)
    in_features = int(in_features)
    if rows <= 0 or in_features <= 0 or in_features % _Q8_1_BLOCK != 0:
        raise ValueError("q8_1 workspace requires positive rows and in_features divisible by 32")
    return rows * (in_features // _Q8_1_BLOCK) * _Q8_1_BLOCK_BYTES


def _optional_q8_1_workspace_ptr(scratch, rows: int, in_features: int, *, enabled: bool | None = None) -> int | None:
    if enabled is None:
        enabled = (
            _gguf_q4k_selected_dual_dp4a_enabled()
            or _gguf_t16_selected_dp4a_enabled()
            or _gguf_raw_selected_dp4a_enabled()
        )
    if not enabled:
        return None
    workspace = getattr(scratch, "moe_q8_1", None)
    if workspace is None:
        return None
    required = _q8_1_workspace_bytes(rows, in_features)
    if int(getattr(workspace, "nbytes", required)) < required:
        raise ValueError(
            f"GGUF q8_1 workspace is too small: need {required} bytes, "
            f"got {getattr(workspace, 'nbytes', 'unknown')}"
        )
    return int(workspace.ptr)


def _dense_q8_raw_ptr(weight: Qwen35GGUFDeviceWeight) -> int | None:
    if weight.spec.quant_key != "gguf_q8_0_t16_v1":
        return None
    try:
        return int(weight.allocation("raw").tensor.ptr)
    except KeyError:
        return None


def _dense_q8_workspace_ptr(scratch, rows: int, in_features: int) -> int | None:
    try:
        return _optional_q8_1_workspace_ptr(scratch, rows, in_features, enabled=True)
    except ValueError:
        return None


def _try_launch_dense_q8_single_dp4a(
    weight: Qwen35GGUFDeviceWeight,
    x_ptr: int,
    out_ptr: int,
    scratch,
    *,
    rows: int,
    in_features: int,
    out_features: int,
    stream: int,
    runtime: HipRuntime,
) -> bool:
    """Opt-in raw-Q8 q8_1/dp4a singleton route for verifier-shaped dense projections."""

    if not _gguf_dense_q8_dp4a_all_enabled() or int(rows) <= 1:
        return False
    raw = _dense_q8_raw_ptr(weight)
    if raw is None:
        return False
    q8_1_workspace_ptr = _dense_q8_workspace_ptr(scratch, rows, in_features)
    if q8_1_workspace_ptr is None:
        return False
    gguf_q4_k_quantize_bf16_q8_1(
        x_ptr,
        q8_1_workspace_ptr,
        rows,
        in_features,
        stream=stream,
        runtime=runtime,
    )
    gguf_q8_0_dp4a_rowtile4_gemv_bf16_bf16_out(
        q8_1_workspace_ptr,
        raw,
        out_ptr,
        rows,
        in_features,
        out_features,
        stream=stream,
        runtime=runtime,
    )
    return True


def _try_launch_dense_q8_triple_dp4a(
    weight_a: Qwen35GGUFDeviceWeight,
    weight_b: Qwen35GGUFDeviceWeight,
    weight_c: Qwen35GGUFDeviceWeight,
    x_ptr: int,
    out_a_ptr: int,
    out_b_ptr: int,
    out_c_ptr: int,
    scratch,
    *,
    rows: int,
    in_features: int,
    out_features_a: int,
    out_features_b: int,
    out_features_c: int,
    stream: int,
    runtime: HipRuntime,
) -> bool:
    """Opt-in raw-Q8 q8_1/dp4a triple route for verifier-shaped Q/K/V projections."""

    if not _gguf_dense_q8_dp4a_all_enabled() or int(rows) <= 1:
        return False
    raw_a = _dense_q8_raw_ptr(weight_a)
    raw_b = _dense_q8_raw_ptr(weight_b)
    raw_c = _dense_q8_raw_ptr(weight_c)
    if raw_a is None or raw_b is None or raw_c is None:
        return False
    q8_1_workspace_ptr = _dense_q8_workspace_ptr(scratch, rows, in_features)
    if q8_1_workspace_ptr is None:
        return False
    gguf_q4_k_quantize_bf16_q8_1(
        x_ptr,
        q8_1_workspace_ptr,
        rows,
        in_features,
        stream=stream,
        runtime=runtime,
    )
    gguf_q8_0_dp4a_triple_split_rowtile4_gemv_bf16_bf16_out(
        q8_1_workspace_ptr,
        raw_a,
        raw_b,
        raw_c,
        out_a_ptr,
        out_b_ptr,
        out_c_ptr,
        rows,
        in_features,
        out_features_a,
        out_features_b,
        out_features_c,
        stream=stream,
        runtime=runtime,
    )
    return True


def _try_launch_dense_q8_pair_dp4a(
    weight_a: Qwen35GGUFDeviceWeight,
    weight_b: Qwen35GGUFDeviceWeight,
    x_ptr: int,
    out_a_ptr: int,
    out_b_ptr: int,
    scratch,
    *,
    rows: int,
    in_features: int,
    out_features_a: int,
    out_features_b: int,
    stream: int,
    runtime: HipRuntime,
) -> bool:
    """Opt-in llama-compat Q8_0 q8_1/dp4a route for verifier block projections."""

    if not _gguf_dense_q8_dp4a_enabled() or int(rows) <= 1:
        return False
    raw_a = _dense_q8_raw_ptr(weight_a)
    raw_b = _dense_q8_raw_ptr(weight_b)
    if raw_a is None or raw_b is None:
        return False
    q8_1_workspace_ptr = _dense_q8_workspace_ptr(scratch, rows, in_features)
    if q8_1_workspace_ptr is None:
        return False
    gguf_q4_k_quantize_bf16_q8_1(
        x_ptr,
        q8_1_workspace_ptr,
        rows,
        in_features,
        stream=stream,
        runtime=runtime,
    )
    gguf_q8_0_dp4a_dual_split_rowtile4_gemv_bf16_bf16_out(
        q8_1_workspace_ptr,
        raw_a,
        raw_b,
        out_a_ptr,
        out_b_ptr,
        rows,
        in_features,
        out_features_a,
        out_features_b,
        stream=stream,
        runtime=runtime,
    )
    return True


def _gguf_aotriton_prefill_mode(start: int, rows: int, key_rows: int) -> str:
    """Resolve the GGUF AOTriton prefill wrapper for the current query window.

    V3 is the standardized default and is required whenever the query rows are
    only a suffix of the key rows.  That case needs AOTriton's bottom-right
    causal window semantics.  The legacy V2 wrapper is still useful as a
    diagnostic for full-context prefill where ``start == 0`` and
    ``rows == key_rows``; forcing V2 outside that shape would apply the wrong
    causal mask, so reject it rather than silently producing bad logits.
    """

    start = int(start)
    rows = int(rows)
    key_rows = int(key_rows)
    full_context = start == 0 and rows == key_rows
    raw = (_env_value(_GGUF_AOTRITON_PREFILL_ENV) or "v3").strip().lower().replace("_", "-")
    aliases = {
        "default": "v3",
        "standard": "v3",
        "v3": "v3",
        "legacy": "v2",
        "v2": "v2",
        "auto": "auto",
        "v2-if-safe": "auto",
        "safe-v2": "auto",
    }
    mode = aliases.get(raw)
    if mode is None:
        raise ValueError(
            f"{_GGUF_AOTRITON_PREFILL_ENV} must be one of v3, v2, or auto/v2-if-safe; got {raw!r}"
        )
    if mode == "auto":
        return "v2" if full_context else "v3"
    if mode == "v2" and not full_context:
        raise ValueError(
            f"{_GGUF_AOTRITON_PREFILL_ENV}=v2 is only valid for full-context prefill; "
            f"got start={start}, rows={rows}, key_rows={key_rows}. Use v3 or auto for chunked prefill."
        )
    return mode


def resolve_qwen35moe_fastpath_safety(
    *,
    is_qwen35moe: bool,
    use_wmma_prefill: bool | None,
    use_gemv_decode: bool | None,
) -> Qwen35GGUFFastPathSafety:
    """Resolve correctness-safe qwen35moe WMMA/GEMV opt-in state.

    P9.E2 showed that the raw-GGUF (non-repack) qwen35moe full-model path fails
    the formal KL/top-1 contract when WMMA prefill and GEMV decode are BOTH
    enabled.  The T16 decode-repack path (P9.H3/D1-D18) passes E2E correctness
    with both opt-ins active, but T16 has no WMMA prefill kernels yet, so
    prefill stays on the slow GEMV fallback.  Raw-GGUF WMMA prefill without
    GEMV decode is deterministic (P9.C11) and the fastest correct-ish prefill
    path available today.  The unsafe combination we block is specifically
    raw weights + WMMA prefill + GEMV decode together.
    """

    requested_wmma = gguf_wmma_prefill_enabled(use_wmma_prefill)
    requested_gemv = gguf_gemv_decode_enabled(use_gemv_decode)
    decode_repack = gguf_decode_repack_enabled(None)
    allow_unsafe = _env_truthy(_QWEN35MOE_UNSAFE_FASTPATH_ENV)
    # Block the proven-unsafe raw-GGUF combo: WMMA + GEMV decode without T16.
    disabled_wmma = bool(is_qwen35moe and requested_wmma and requested_gemv and not decode_repack and not allow_unsafe)
    disabled_gemv = bool(is_qwen35moe and requested_gemv and not allow_unsafe and not decode_repack)
    reason = None
    if disabled_wmma or disabled_gemv:
        reason = (
            "qwen35moe raw-GGUF GEMV decode is disabled by default because P9.E2 "
            "rejected the raw-GGUF opt-in; raw-GGUF WMMA prefill + GEMV decode is "
            "also blocked as the unsafe combo. Enable resident T16 decode repack "
            f"(HIPENGINE_GGUF_DECODE_REPACK=1) or set {_QWEN35MOE_UNSAFE_FASTPATH_ENV}=1."
        )
    return Qwen35GGUFFastPathSafety(
        is_qwen35moe=bool(is_qwen35moe),
        allow_unsafe_qwen35moe_fastpaths=allow_unsafe,
        requested_wmma_prefill=requested_wmma,
        requested_gemv_decode=requested_gemv,
        effective_wmma_prefill=bool(requested_wmma and not disabled_wmma),
        effective_gemv_decode=bool(requested_gemv and not disabled_gemv),
        disabled_wmma_prefill=disabled_wmma,
        disabled_gemv_decode=disabled_gemv,
        reason=reason,
    )


def _chunk_ranges(total: int, chunk_size: int, *, min_chunk_size: int = 1) -> tuple[tuple[int, int], ...]:
    total = int(total)
    if total <= 0:
        raise ValueError("total must be positive")
    size = int(chunk_size)
    min_rows = max(1, int(min_chunk_size))
    if size <= 0 or total <= size:
        return ((0, total),)
    ranges = [(start, min(start + size, total)) for start in range(0, total, size)]
    while len(ranges) >= 2 and ranges[-1][1] - ranges[-1][0] < min_rows:
        ranges[-2] = (ranges[-2][0], ranges[-1][1])
        ranges.pop()
    return tuple(ranges)


@dataclass
class Qwen35GGUFResidentSession:
    """Persistent GGUF Qwen3.5 session for public greedy generation.

    The session materializes GGUF weights once, owns reusable device scratch, and
    carries linear-attention recurrent state plus paged full-attention K/V cache
    across decode steps. Full-attention q/k norm, RoPE, KV append, softmax, gate
    application, lm-head argmax, full-model bulk prefill, and one-step decode
    graph replay stay on GPU for the resident path.
    """

    model_path: str | Path
    runtime: HipRuntime | None = None
    compiler_version: str | None = None
    require_cached_build: bool = False
    max_sequence_length: int | None = None
    use_expert_sidecar: bool = False
    expert_sidecar_cache_dir: str | Path | None = None
    require_expert_sidecar: bool = False
    preload_expert_sidecars: bool = True
    use_wmma_prefill: bool | None = None
    use_gemv_decode: bool | None = None
    prefill_chunk_size: int = 0
    prefill_config: PrefillConfig | None = None
    runner: Qwen35GGUFFullStackRunner | None = field(default=None, init=False)
    scratch: object | None = field(default=None, init=False)
    _token_buf: object | None = field(default=None, init=False)
    _hidden_a: object | None = field(default=None, init=False)
    _hidden_b: object | None = field(default=None, init=False)
    _logits_buf: object | None = field(default=None, init=False)
    _lm_block_values: object | None = field(default=None, init=False)
    _lm_block_indices: object | None = field(default=None, init=False)
    _lm_out_index: object | None = field(default=None, init=False)
    _lm_out_value: object | None = field(default=None, init=False)
    _verify_hidden_seed_buf: object | None = field(default=None, init=False)
    _verify_token_ids_i64: object | None = field(default=None, init=False)
    _verify_token_counter_i64: object | None = field(default=None, init=False)
    _verify_block_rows_capacity: int = field(default=0, init=False)
    _verify_hidden_seed_rows_populated: int = field(default=0, init=False)
    _verify_logits_buf: object | None = field(default=None, init=False)
    _verify_lm_block_values: object | None = field(default=None, init=False)
    _verify_lm_block_indices_i32: object | None = field(default=None, init=False)
    _verify_lm_out_indices_i32: object | None = field(default=None, init=False)
    _verify_lm_out_values: object | None = field(default=None, init=False)
    _verify_lm_q8_1: object | None = field(default=None, init=False)
    _verify_lm_rows_capacity: int = field(default=0, init=False)
    _verify_linear_conv_state_rows: tuple[object | None, ...] = field(default=(), init=False)
    _verify_linear_recurrent_state_rows: tuple[object | None, ...] = field(default=(), init=False)
    _verify_linear_state_rows_capacity: int = field(default=0, init=False)
    _verify_linear_state_src_conv_table_buf: object | None = field(default=None, init=False)
    _verify_linear_state_src_recurrent_table_buf: object | None = field(default=None, init=False)
    _verify_linear_state_dst_conv_table_buf: object | None = field(default=None, init=False)
    _verify_linear_state_dst_recurrent_table_buf: object | None = field(default=None, init=False)
    _verify_linear_state_commit_row_i32_buf: object | None = field(default=None, init=False)
    _verify_linear_state_src_conv_host: np.ndarray | None = field(default=None, init=False)
    _verify_linear_state_src_recurrent_host: np.ndarray | None = field(default=None, init=False)
    _verify_linear_state_src_conv_cached: np.ndarray | None = field(default=None, init=False)
    _verify_linear_state_src_recurrent_cached: np.ndarray | None = field(default=None, init=False)
    _verify_linear_state_dst_conv_host: np.ndarray | None = field(default=None, init=False)
    _verify_linear_state_dst_recurrent_host: np.ndarray | None = field(default=None, init=False)
    _verify_linear_state_conv_row_nbytes: int = field(default=0, init=False)
    _verify_linear_state_recurrent_row_nbytes: int = field(default=0, init=False)
    _verify_linear_state_layer_count: int = field(default=0, init=False)
    _prefill_token_buf: object | None = field(default=None, init=False)
    _prefill_hidden_a: object | None = field(default=None, init=False)
    _prefill_hidden_b: object | None = field(default=None, init=False)
    _bulk_prefill_scratch: object | None = field(default=None, init=False)
    _linear_state_snapshot_backups: tuple[object, ...] = field(default=(), init=False)
    _runtime_state_library: object | None = field(default=None, init=False)
    _lm_head_library: object | None = field(default=None, init=False)
    _dflash_commit_library: object | None = field(default=None, init=False)
    _expert_pack8_library: object | None = field(default=None, init=False)
    _q6_pack8_library: object | None = field(default=None, init=False)
    _expert_sidecar_reader: GGUFReader | None = field(default=None, init=False)
    _expert_sidecar_model_map: object | None = field(default=None, init=False)
    _expert_sidecar_host_layers: dict[int, dict[str, GGUFExpertPackedTensor]] | None = field(default=None, init=False)
    _token_host: np.ndarray = field(default_factory=lambda: np.empty((1,), dtype=np.int64), init=False)
    _logits_host: np.ndarray | None = field(default=None, init=False)
    _buffers: tuple[object, ...] = field(default=(), init=False)
    _position: int = field(default=0, init=False)
    _hidden_seed_fp32_populated: bool = field(default=False, init=False)
    _lm_head_threads: int = field(default=128, init=False)
    _lm_head_stage1_blocks: int = field(default=0, init=False)
    last_verify_stage_timings_ms: dict[str, float] = field(default_factory=dict, init=False)
    fastpath_safety: Qwen35GGUFFastPathSafety | None = field(default=None, init=False)
    prefill_chunk_tuning: dict[str, object] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.runtime = self.runtime or get_hip_runtime()
        self.runner = Qwen35GGUFFullStackRunner(
            self.model_path,
            runtime=self.runtime,
            compiler_version=self.compiler_version,
            require_cached_build=self.require_cached_build,
        )
        if self.runner.weights is None:
            raise RuntimeError("GGUF full-stack runner did not materialize weights")
        self.fastpath_safety = resolve_qwen35moe_fastpath_safety(
            is_qwen35moe=self.runner.weights.config.is_moe,
            use_wmma_prefill=self.use_wmma_prefill,
            use_gemv_decode=self.use_gemv_decode,
        )
        if self.runner.weights.config.is_moe:
            self.use_wmma_prefill = self.fastpath_safety.effective_wmma_prefill
            self.use_gemv_decode = self.fastpath_safety.effective_gemv_decode
        runtime = self.runtime or get_hip_runtime()
        build_kwargs = {
            "load": True,
            "compiler_version": self.compiler_version,
            "require_cached": self.require_cached_build,
        }
        self._runtime_state_library = build_runtime_state(**build_kwargs)
        self._lm_head_library = build_lm_head(**build_kwargs)
        if _gguf_verify_lm_head_q6_top1_dp4a_enabled():
            self._q6_pack8_library = build_gguf_q6_k_pack8_gemv(**build_kwargs)
        if self.use_expert_sidecar:
            self._expert_pack8_library = build_gguf_expert_pack8_gemv(**build_kwargs)
            setattr(self.runner, "_expert_pack8_library", self._expert_pack8_library)
            self._expert_sidecar_reader = GGUFReader(self.model_path)
            self._expert_sidecar_model_map = build_qwen35_gguf_tensor_map(self._expert_sidecar_reader.info)
            if self.preload_expert_sidecars:
                self._expert_sidecar_host_layers = {
                    layer_id: self._load_expert_sidecar_host_layer(layer_id)
                    for layer_id in range(self.runner.weights.config.block_count)
                }
        self.scratch = _FullStackScratch.allocate(
            self.runner,
            runtime=runtime,
            max_sequence_length=self.max_sequence_length,
        )
        total_memory_bytes = 0
        try:
            _free_bytes, total_memory_bytes = runtime.mem_get_info()
        except Exception:
            total_memory_bytes = 0
        self.prefill_config, self.prefill_chunk_tuning = resolve_prefill_config_for_sequence(
            self.prefill_config or PrefillConfig(),
            max_sequence_length=int(self.scratch.max_positions),
            total_memory_bytes=int(total_memory_bytes),
        )
        self._token_buf = malloc(self._token_host.nbytes, runtime=runtime)
        hidden_bytes = self.runner.hidden_size * 2
        self._hidden_a = malloc(hidden_bytes, runtime=runtime)
        self._hidden_b = malloc(hidden_bytes, runtime=runtime)
        self._logits_host = np.empty((1, self.runner.vocab_size), dtype=np.float32)
        self._logits_buf = malloc(self._logits_host.nbytes, runtime=runtime)
        self._lm_head_threads = 128
        self._lm_head_stage1_blocks = lm_head_argmax_stage1_blocks(self.runner.vocab_size, threads=self._lm_head_threads)
        self._lm_block_values = malloc(self._lm_head_stage1_blocks * DType.FP32.itemsize, runtime=runtime)
        self._lm_block_indices = malloc(self._lm_head_stage1_blocks * DType.INT64.itemsize, runtime=runtime)
        self._lm_out_index = malloc(DType.INT64.itemsize, runtime=runtime)
        self._lm_out_value = malloc(DType.FP32.itemsize, runtime=runtime)
        prefill_capacity = int(self.scratch.max_positions)
        self._prefill_token_buf = malloc(prefill_capacity * DType.INT64.itemsize, runtime=runtime)
        self._prefill_hidden_a = malloc(prefill_capacity * hidden_bytes, runtime=runtime)
        self._prefill_hidden_b = malloc(prefill_capacity * hidden_bytes, runtime=runtime)
        prefill_rows = self._prefill_scratch_rows(prefill_capacity)
        self._bulk_prefill_scratch = _GGUFFullAttentionPrefillScratch.allocate(
            self.runner,
            rows=prefill_rows,
            capacity=prefill_capacity,
            allocate_kv_cache=False,
            runtime=runtime,
        )
        self._buffers = (
            self._token_buf,
            self._hidden_a,
            self._hidden_b,
            self._logits_buf,
            self._lm_block_values,
            self._lm_block_indices,
            self._lm_out_index,
            self._lm_out_value,
            self._prefill_token_buf,
            self._prefill_hidden_a,
            self._prefill_hidden_b,
            *self._bulk_prefill_scratch.buffers,
        )
        # Lazily-created per-layer MoE FFN graph cache (rows==1 resident decode),
        # gated by HIPENGINE_GGUF_MOE_GRAPH. None until first graphed decode.
        self._moe_graph: MoeGraphCache | None = None
        self.reset()

    @property
    def position(self) -> int:
        """Next token position that will be consumed by :meth:`step`."""

        return int(self._position)

    def hidden_seed_contract(self, *, rows: int = 1) -> Qwen35GGUFHiddenSeedContract:
        """Return the current GGUF MTP hidden-seed contract for this session.

        The method is metadata-only: it does not read device memory or change
        generation state.  It exposes the current BF16 post-output_norm scratch
        row contract so M2.5 can replace it with an fp32-compatible tap.
        """

        if self.runner is None:
            raise RuntimeError("GGUF resident session is closed")
        return qwen35_gguf_current_hidden_seed_contract(
            self.runner.hidden_size,
            rows=rows,
        )

    def fp32_hidden_seed_contract(self, *, rows: int = 1) -> Qwen35GGUFHiddenSeedContract:
        """Return the allocated fp32 M2.5 hidden-seed target contract."""

        if self.runner is None or self.scratch is None:
            raise RuntimeError("GGUF resident session is closed")
        return qwen35_gguf_fp32_hidden_seed_contract(
            self.runner.hidden_size,
            rows=rows,
            populated_by_decode=self._hidden_seed_fp32_populated,
        )

    def fp32_hidden_seed_ptr(self) -> int:
        """Return the populated fp32 hidden-seed device pointer.

        The pointer is only valid after a prefill/decode step with
        ``capture_hidden_seed_fp32=True``.  This guard prevents future MTP code
        from consuming the allocated scratch row before it is populated.
        """

        if self.runner is None or self.scratch is None:
            raise RuntimeError("GGUF resident session is closed")
        if not self.fp32_hidden_seed_contract().ready_for_mtp:
            raise RuntimeError(
                "GGUF fp32 hidden seed is not populated; "
                "call prefill(..., capture_hidden_seed_fp32=True) or "
                "step(..., capture_hidden_seed_fp32=True) first"
            )
        return int(self.scratch.hidden_seed_fp32.ptr)

    def mtp_draft_seed(self, *, token_id: int, position: int) -> Qwen35GGUFMTPDraftSeed:
        """Package the populated fp32 hidden seed for future NextN draft work."""

        return Qwen35GGUFMTPDraftSeed(
            token_id=int(token_id),
            position=int(position),
            hidden_ptr=self.fp32_hidden_seed_ptr(),
            hidden_contract=self.fp32_hidden_seed_contract(rows=1),
        )

    def fp32_verify_hidden_seed_contract(self, *, rows: int = 1) -> Qwen35GGUFHiddenSeedContract:
        """Return the fp32 hidden-row staging contract for target verifier rows."""

        if self.runner is None:
            raise RuntimeError("GGUF resident session is closed")
        return qwen35_gguf_fp32_verify_hidden_seed_contract(
            self.runner.hidden_size,
            rows=int(rows),
            populated_by_decode=int(self._verify_hidden_seed_rows_populated) >= int(rows),
        )

    def fp32_verify_hidden_seed_ptr(self, row_index: int = 0) -> int:
        """Return a staged fp32 verifier hidden-row pointer."""

        if self.runner is None:
            raise RuntimeError("GGUF resident session is closed")
        if self._verify_hidden_seed_buf is None:
            raise RuntimeError("GGUF verifier hidden seed rows are not allocated")
        row = int(row_index)
        if row < 0 or row >= int(self._verify_hidden_seed_rows_populated):
            raise RuntimeError("GGUF verifier hidden seed row is not populated")
        row_nbytes = self.runner.hidden_size * DType.FP32.itemsize
        return int(self._verify_hidden_seed_buf.ptr + row * row_nbytes)

    def stage_current_hidden_seed_as_verify_row(
        self,
        *,
        row_index: int,
        token_id: int,
        position: int,
        rows_capacity: int | None = None,
        stream: int = 0,
    ) -> Qwen35GGUFMTPDraftSeed:
        """Copy the current fp32 target hidden seed into verifier-row staging.

        This is the device-resident counterpart to the llama.cpp ``verify_h``
        rows: serial target verification can preserve each row with a D2D copy
        instead of reading hidden rows back to the host before MTP draft-KV
        commit.
        """

        if self.runner is None or self.scratch is None:
            raise RuntimeError("GGUF resident session is closed")
        if not self.fp32_hidden_seed_contract().ready_for_mtp:
            raise RuntimeError(
                "GGUF fp32 hidden seed is not populated; "
                "call prefill(..., capture_hidden_seed_fp32=True) or "
                "step(..., capture_hidden_seed_fp32=True) first"
            )
        row = int(row_index)
        if row < 0:
            raise ValueError("row_index must be non-negative")
        capacity = max(row + 1, int(rows_capacity or 0))
        runtime = self.runtime or get_hip_runtime()
        self._ensure_verify_block_buffers(capacity, runtime=runtime)
        if self._verify_hidden_seed_buf is None:
            raise RuntimeError("GGUF verifier hidden seed rows are not allocated")
        row_nbytes = self.runner.hidden_size * DType.FP32.itemsize
        runtime.memcpy_async(
            self._verify_hidden_seed_buf.ptr + row * row_nbytes,
            self.scratch.hidden_seed_fp32.ptr,
            row_nbytes,
            HipMemcpyKind.DEVICE_TO_DEVICE,
            stream,
        )
        self._verify_hidden_seed_rows_populated = max(
            int(self._verify_hidden_seed_rows_populated),
            row + 1,
        )
        return Qwen35GGUFMTPDraftSeed(
            token_id=int(token_id),
            position=int(position),
            hidden_ptr=self.fp32_verify_hidden_seed_ptr(row),
            hidden_contract=self.fp32_verify_hidden_seed_contract(rows=1),
        )

    def reset(self) -> None:
        """Reset sequence state without freeing resident weights or scratch."""

        if self.scratch is None:
            raise RuntimeError("GGUF resident session is closed")
        runtime = self.runtime or get_hip_runtime()
        self.scratch.zero_states(runtime)
        self._position = 0
        self._hidden_seed_fp32_populated = False
        self._verify_hidden_seed_rows_populated = 0

    def _linear_state_snapshot(self) -> list[tuple[DeviceBuffer, DeviceBuffer]]:
        """D2D-copy linear-attention state for rollback-safe block verification.

        Snapshot buffers are allocated lazily and reused across verifier cycles;
        allocating/freeing them per cycle cost several milliseconds in the B3
        verifier path and is independent of the model math we want to measure.
        """

        if self.scratch is None:
            raise RuntimeError("GGUF resident session is closed")
        runtime = self.runtime or get_hip_runtime()
        states: list[DeviceBuffer] = []
        for conv_state, recurrent_state in zip(
            self.scratch.layer_conv_states,
            self.scratch.layer_recurrent_states,
            strict=True,
        ):
            for state in (conv_state, recurrent_state):
                if state is not None:
                    states.append(state)
        backups = list(self._linear_state_snapshot_backups)
        if len(backups) != len(states) or any(
            int(backup.nbytes) != int(state.nbytes)
            for state, backup in zip(states, backups, strict=False)
        ):
            for backup in reversed(backups):
                free(backup, runtime=runtime)
            backups = [malloc(state.nbytes, runtime=runtime) for state in states]
            self._linear_state_snapshot_backups = tuple(backups)
        snapshot = list(zip(states, backups, strict=True))
        for state, backup in snapshot:
            runtime.memcpy(backup.ptr, state.ptr, state.nbytes, HipMemcpyKind.DEVICE_TO_DEVICE)
        return snapshot

    def _restore_linear_state_snapshot(
        self,
        snapshot: list[tuple[DeviceBuffer, DeviceBuffer]],
        *,
        position: int,
    ) -> None:
        if self.scratch is None:
            raise RuntimeError("GGUF resident session is closed")
        runtime = self.runtime or get_hip_runtime()
        for state, backup in snapshot:
            runtime.memcpy(state.ptr, backup.ptr, state.nbytes, HipMemcpyKind.DEVICE_TO_DEVICE)
        self._position = int(position)
        self.scratch.position_host[0] = int(position)
        self.scratch.context_host[0] = int(position) + 1
        set_decode_position_i64(
            self.scratch.position_buf.ptr,
            self.scratch.context_buf.ptr,
            int(position),
            library=self._runtime_state_library,
            runtime=runtime,
        )
        self._hidden_seed_fp32_populated = False

    def _free_linear_state_snapshot(self, snapshot: list[tuple[DeviceBuffer, DeviceBuffer]]) -> None:
        # Backing buffers are persistent and freed with the session.
        del snapshot

    @staticmethod
    def _smallest_positive_or_total(total: int, *sizes: int) -> int:
        positives = [int(size) for size in sizes if int(size) > 0]
        return int(total) if not positives else min(int(total), min(positives))

    def _manual_prefill_chunk_size(self) -> int:
        return max(0, int(self.prefill_chunk_size or 0))

    def _linear_prefill_layer_chunk_size(self, tokens: int) -> int:
        tokens = int(tokens)
        min_rows = int(getattr(self.runner.weights.config, "ssm_conv_kernel", 1)) if self.runner and self.runner.weights else 1
        manual = self._manual_prefill_chunk_size()
        if manual > 0:
            return min(tokens, max(manual, min_rows)) if tokens >= min_rows else tokens
        config = self.prefill_config or PrefillConfig()
        size = self._smallest_positive_or_total(tokens, config.linear_chunk_size, config.moe_chunk_size)
        return min(tokens, max(size, min_rows)) if tokens >= min_rows else tokens

    def _full_attention_prefill_layer_chunk_size(self, tokens: int) -> int:
        tokens = int(tokens)
        manual = self._manual_prefill_chunk_size()
        if manual > 0:
            return min(tokens, max(manual, 2)) if tokens > 1 else tokens
        config = self.prefill_config or PrefillConfig()
        if int(config.full_attn_query_chunk_size) > 0:
            size = min(tokens, int(config.full_attn_query_chunk_size))
        else:
            size = self._smallest_positive_or_total(
                tokens,
                config.full_attn_post_chunk_size,
                config.full_attn_rope_chunk_size,
                config.moe_chunk_size,
            )
        return 2 if tokens > 1 and size == 1 else size

    def _prefill_scratch_rows(self, capacity: int) -> int:
        capacity = int(capacity)
        if capacity <= 0:
            raise ValueError("prefill capacity must be positive")
        return max(
            1,
            min(
                capacity,
                max(
                    self._linear_prefill_layer_chunk_size(capacity),
                    self._full_attention_prefill_layer_chunk_size(capacity),
                ),
            ),
        )

    def prefill(
        self,
        token_ids: list[int] | tuple[int, ...],
        *,
        use_bulk: bool | None = None,
        bulk_attention_mode: str = "bulk",
        return_logits: bool = True,
        capture_hidden_seed_fp32: bool = False,
    ) -> Qwen35GGUFNextTokenProbeResult:
        """Consume prompt tokens once and return the greedy next token.

        Prompts at least as long as the linear-attention convolution kernel use
        bulk prefill by default. qwen35moe now defaults to the fast fully
        bulk attention+MoE scheduler after the full-attention and recurrent
        parity fixes; pass ``bulk_attention_mode='native'`` to keep row-serial
        attention as a diagnostic fallback. Short prompts keep the token-serial
        path as a correctness/bisect fallback. Set
        ``return_logits=False`` for public generation paths that only need the
        sampled token and should avoid copying full logits back to the host.
        Set ``capture_hidden_seed_fp32=True`` to populate the M2.5
        post-output_norm fp32 seed row for the final prompt token.
        """

        if not token_ids:
            raise ValueError("token_ids must be non-empty")
        if self.runner is None or self.runner.weights is None:
            raise RuntimeError("GGUF resident session is closed")
        min_bulk_tokens = int(self.runner.weights.config.ssm_conv_kernel)
        selected_bulk_attention_mode = bulk_attention_mode
        run_bulk = len(token_ids) >= min_bulk_tokens if use_bulk is None else bool(use_bulk)
        if run_bulk:
            if len(token_ids) < min_bulk_tokens:
                raise ValueError(
                    f"GGUF bulk prefill requires at least {min_bulk_tokens} tokens; got {len(token_ids)}"
                )
            with wmma_prefill_session(self.use_wmma_prefill), gemv_decode_session(self.use_gemv_decode):
                return self._run_bulk_prefill_and_sample(
                    token_ids,
                    bulk_attention_mode=selected_bulk_attention_mode,
                    return_logits=return_logits,
                    capture_hidden_seed_fp32=bool(capture_hidden_seed_fp32),
                )

        self.reset()
        hidden_ptr = None
        final_index = len(token_ids) - 1
        for index, token_id in enumerate(token_ids):
            hidden_ptr = self._run_token_to_final_hidden(
                int(token_id),
                position=self._position,
                capture_hidden_seed_fp32=bool(capture_hidden_seed_fp32) and index == final_index,
            )
            self._position += 1
        assert hidden_ptr is not None
        return self._sample_from_hidden(hidden_ptr, return_logits=return_logits)

    def _run_bulk_prefill_and_sample(
        self,
        token_ids: list[int] | tuple[int, ...],
        *,
        stream: int = 0,
        bulk_attention_mode: str = "bulk",
        return_logits: bool = True,
        capture_hidden_seed_fp32: bool = False,
    ) -> Qwen35GGUFNextTokenProbeResult:
        if self.runner is None or self.runner.weights is None or self.scratch is None:
            raise RuntimeError("GGUF resident session is closed")
        if self._prefill_token_buf is None or self._prefill_hidden_a is None or self._prefill_hidden_b is None:
            raise RuntimeError("GGUF resident bulk prefill buffers are closed")
        if self._bulk_prefill_scratch is None:
            raise RuntimeError("GGUF resident bulk prefill scratch is closed")
        rows = int(len(token_ids))
        if rows <= 0:
            raise ValueError("token_ids must be non-empty")
        if rows > self.scratch.max_positions:
            raise ValueError(f"GGUF bulk prefill rows {rows} exceed cache capacity {self.scratch.max_positions}")
        if bulk_attention_mode not in {"bulk", "native"}:
            raise ValueError("bulk_attention_mode must be 'bulk' or 'native'")
        runtime = self.runtime or get_hip_runtime()
        tokens = np.asarray([int(token) for token in token_ids], dtype=np.int64)
        for token in tokens.tolist():
            if token < 0 or token >= self.runner.vocab_size:
                raise ValueError(f"token_id {token} outside [0, {self.runner.vocab_size})")
        self.reset()
        copy_host_to_device(self._prefill_token_buf, host_array_ptr(tokens), tokens.nbytes, runtime=runtime)
        launch_gguf_embedding(
            self.runner.weights.root("token_embedding"),
            self._prefill_token_buf.ptr,
            self._prefill_hidden_a.ptr,
            rows=rows,
            hidden_size=self.runner.hidden_size,
            vocab_size=self.runner.vocab_size,
            stream=stream,
            runtime=runtime,
        )
        src = self._prefill_hidden_a
        dst = self._prefill_hidden_b
        use_wmma_prefill = gguf_wmma_prefill_enabled(None)
        linear_min_rows = int(self.runner.weights.config.ssm_conv_kernel)
        for layer_id, layer_type in enumerate(self.runner.weights.config.layer_types):
            expert_sidecar = None
            if (
                self.use_expert_sidecar
                and bulk_attention_mode == "bulk"
                and self.runner.weights.config.is_moe
                and not use_wmma_prefill
            ):
                expert_sidecar = self._load_expert_sidecar_device_layer(layer_id, runtime=runtime)
            try:
                if layer_type == LINEAR_ATTENTION:
                    chunk_size = self._linear_prefill_layer_chunk_size(rows)
                    ranges = _chunk_ranges(rows, chunk_size, min_chunk_size=linear_min_rows)
                elif layer_type == FULL_ATTENTION:
                    chunk_size = self._full_attention_prefill_layer_chunk_size(rows)
                    ranges = _chunk_ranges(rows, chunk_size, min_chunk_size=2)
                else:
                    raise ValueError(f"unsupported GGUF layer type {layer_type!r}")
                for start, end in ranges:
                    chunk_rows = end - start
                    src_chunk_ptr = src.ptr + start * self.runner.hidden_size * 2
                    dst_chunk_ptr = dst.ptr + start * self.runner.hidden_size * 2
                    bulk_scratch = self._bulk_prefill_scratch.for_chunk(
                        start, chunk_rows, total_tokens=rows, runtime=runtime, stream=stream
                    )
                    if bulk_attention_mode == "native":
                        self.runner._run_native_attention_bulk_ffn_layer_rows(
                            layer_id,
                            layer_type,
                            src_chunk_ptr,
                            dst_chunk_ptr,
                            bulk_scratch,
                            rows=chunk_rows,
                            start_position=start,
                            stream=stream,
                            decode_scratch=self.scratch,
                        )
                    elif layer_type == LINEAR_ATTENTION:
                        self.runner._run_linear_attention_prefill_layer_rows(
                            layer_id,
                            src_chunk_ptr,
                            dst_chunk_ptr,
                            bulk_scratch,
                            rows=chunk_rows,
                            stream=stream,
                            decode_scratch=self.scratch,
                            expert_sidecar=expert_sidecar,
                        )
                    elif layer_type == FULL_ATTENTION:
                        key_cache, value_cache = self.scratch.full_cache(layer_id)
                        layer_scratch = replace(bulk_scratch, key_cache=key_cache, value_cache=value_cache)
                        self.runner._run_full_attention_prefill_layer_aotriton(
                            layer_id,
                            src_chunk_ptr,
                            dst_chunk_ptr,
                            layer_scratch,
                            stream=stream,
                            expert_sidecar=expert_sidecar,
                        )
                    else:
                        raise ValueError(f"unsupported GGUF layer type {layer_type!r}")
            finally:
                if expert_sidecar is not None:
                    expert_sidecar.free(runtime=runtime)
            src, dst = dst, src
        last_bulk_scratch = self._bulk_prefill_scratch.for_chunk(
            rows - 1, 1, total_tokens=rows, runtime=runtime, stream=stream
        )
        last_src_ptr = src.ptr + (rows - 1) * self.runner.hidden_size * 2
        self._run_output_norm_hidden(
            last_src_ptr,
            last_bulk_scratch.norm.ptr,
            stream=stream,
            capture_hidden_seed_fp32=bool(capture_hidden_seed_fp32),
        )
        self._position = rows
        self.scratch.position_host[0] = rows
        self.scratch.context_host[0] = rows + 1
        set_decode_position_i64(
            self.scratch.position_buf.ptr,
            self.scratch.context_buf.ptr,
            rows,
            stream=stream,
            library=self._runtime_state_library,
            runtime=runtime,
        )
        last_hidden_ptr = last_bulk_scratch.norm.ptr
        return self._sample_from_hidden(last_hidden_ptr, return_logits=return_logits)

    def verify_target_block(
        self,
        input_token_ids: list[int] | tuple[int, ...],
        *,
        bulk_attention_mode: str = "bulk",
        use_wmma_prefill: bool | None = None,
        stream: int = 0,
        advance_state_only: bool = False,
        capture_linear_state_rows: bool = False,
        record_stage_timings: bool = False,
        sync_stage_timings: bool = False,
        defer_linear_state_commit: bool = False,
    ) -> Qwen35GGUFBlockVerifyResult:
        """Consume a continuation block and return greedy target rows.

        The current resident decode state is treated as the prefix state.  The
        method consumes ``input_token_ids`` at absolute positions beginning at
        :attr:`position`, runs the existing row-bulk target path over that
        continuation block, records FP32 post-output_norm hidden rows, samples
        greedy target IDs row-wise on device, and advances the resident cursor by
        the block length.

        Callers that need partial-accept rollback should snapshot linear state
        with :meth:`_linear_state_snapshot` before this call, restore on mismatch,
        and replay the accepted prefix.  Full-attention KV rows beyond the
        restored cursor are ignored by live-count metadata and overwritten later.

        ``advance_state_only`` skips the per-row LM-head vocab GEMV + greedy
        sampling (~16% of the forward).  Use it for the accepted-prefix REPLAY,
        whose target tokens are already known from the first full-block pass and
        are discarded here: only the linear/KV state advance and the FP32 hidden
        rows (returned and used for decode continuity) are needed.  ``token_ids``
        in the result echoes the input in this mode and must not be consumed.

        ``capture_linear_state_rows`` materializes per-row linear-attention
        Conv/GDN states for a later :meth:`_commit_verify_linear_state_row`
        call.  This is the llama.cpp-style accept-row lifecycle used by strict
        block verifier diagnostics to avoid accepted-prefix replay.
        """

        if self.runner is None or self.runner.weights is None or self.scratch is None:
            raise RuntimeError("GGUF resident session is closed")
        stage_timings: dict[str, float] | None = {} if record_stage_timings else None
        self.last_verify_stage_timings_ms = stage_timings if stage_timings is not None else {}
        sync_stages = bool(sync_stage_timings and stage_timings is not None)

        def add_verify_stage(name: str, ms: float) -> None:
            if stage_timings is None:
                return
            if ms < 0.0:
                raise RuntimeError(f"negative target block stage timing for {name}: {ms}")
            stage_timings[name] = stage_timings.get(name, 0.0) + float(ms)

        if self._prefill_token_buf is None or self._prefill_hidden_a is None or self._prefill_hidden_b is None:
            raise RuntimeError("GGUF resident bulk prefill buffers are closed")
        if self._bulk_prefill_scratch is None:
            raise RuntimeError("GGUF resident bulk prefill scratch is closed")
        if bulk_attention_mode not in {"bulk", "native"}:
            raise ValueError("bulk_attention_mode must be 'bulk' or 'native'")
        rows = int(len(input_token_ids))
        if rows <= 0:
            raise ValueError("input_token_ids must be non-empty")
        if rows > int(self._bulk_prefill_scratch.rows):
            raise ValueError(
                f"target block rows {rows} exceed resident bulk scratch rows {self._bulk_prefill_scratch.rows}"
            )
        start = int(self._position)
        end = start + rows
        if end > int(self.scratch.max_positions):
            raise ValueError(f"target block end position {end} exceeds cache capacity {self.scratch.max_positions}")
        tokens = np.asarray([int(token) for token in input_token_ids], dtype=np.int64)
        for token in tokens.tolist():
            if token < 0 or token >= self.runner.vocab_size:
                raise ValueError(f"token_id {token} outside [0, {self.runner.vocab_size})")
        runtime = self.runtime or get_hip_runtime()
        row_nbytes = self.runner.hidden_size * DType.BF16.itemsize
        t_setup0 = time.perf_counter() if stage_timings is not None else 0.0
        self._ensure_verify_block_buffers(rows, runtime=runtime)
        if capture_linear_state_rows:
            self._ensure_verify_linear_state_row_buffers(rows, runtime=runtime)
        if self._verify_hidden_seed_buf is None:
            raise RuntimeError("GGUF verifier hidden-seed buffer is closed")
        hidden_seed_buf = self._verify_hidden_seed_buf
        token_ids_buf = self._verify_token_ids_i64
        token_counter_buf = self._verify_token_counter_i64
        if token_ids_buf is None or token_counter_buf is None:
            raise RuntimeError("GGUF verifier token buffers are closed")
        zero_index = np.zeros((1,), dtype=np.int64)
        copy_host_to_device(token_counter_buf, host_array_ptr(zero_index), zero_index.nbytes, runtime=runtime)
        copy_host_to_device(self._prefill_token_buf, host_array_ptr(tokens), tokens.nbytes, runtime=runtime)
        add_verify_stage("target_block_setup", (time.perf_counter() - t_setup0) * 1000 if stage_timings is not None else 0.0)
        try:
            t_embedding0 = time.perf_counter() if stage_timings is not None else 0.0
            launch_gguf_embedding(
                self.runner.weights.root("token_embedding"),
                self._prefill_token_buf.ptr,
                self._prefill_hidden_a.ptr + start * row_nbytes,
                rows=rows,
                hidden_size=self.runner.hidden_size,
                vocab_size=self.runner.vocab_size,
                stream=stream,
                runtime=runtime,
            )
            if sync_stages:
                runtime.device_synchronize()
            add_verify_stage(
                "target_block_embedding",
                (time.perf_counter() - t_embedding0) * 1000 if stage_timings is not None else 0.0,
            )
            src = self._prefill_hidden_a
            dst = self._prefill_hidden_b
            block_wmma_prefill = gguf_wmma_prefill_enabled(
                self.use_wmma_prefill if use_wmma_prefill is None else use_wmma_prefill
            )
            with wmma_prefill_session(block_wmma_prefill), gemv_decode_session(self.use_gemv_decode):
                for layer_id, layer_type in enumerate(self.runner.weights.config.layer_types):
                    if sync_stages:
                        runtime.device_synchronize()
                    t_layer0 = time.perf_counter() if stage_timings is not None else 0.0
                    expert_sidecar = None
                    if (
                        self.use_expert_sidecar
                        and bulk_attention_mode == "bulk"
                        and self.runner.weights.config.is_moe
                        and not block_wmma_prefill
                    ):
                        expert_sidecar = self._load_expert_sidecar_device_layer(layer_id, runtime=runtime)
                    try:
                        bulk_scratch = self._bulk_prefill_scratch.for_chunk(
                            start,
                            rows,
                            total_tokens=end,
                            runtime=runtime,
                            stream=stream,
                        )
                        src_chunk_ptr = src.ptr + start * row_nbytes
                        dst_chunk_ptr = dst.ptr + start * row_nbytes
                        if bulk_attention_mode == "native":
                            self.runner._run_native_attention_bulk_ffn_layer_rows(
                                layer_id,
                                layer_type,
                                src_chunk_ptr,
                                dst_chunk_ptr,
                                bulk_scratch,
                                rows=rows,
                                start_position=start,
                                stream=stream,
                                decode_scratch=self.scratch,
                                linear_state_rows=(
                                    self._verify_linear_state_row_pair(layer_id)
                                    if capture_linear_state_rows
                                    else None
                                ),
                            )
                        elif layer_type == LINEAR_ATTENTION:
                            self.runner._run_linear_attention_prefill_layer_rows(
                                layer_id,
                                src_chunk_ptr,
                                dst_chunk_ptr,
                                bulk_scratch,
                                rows=rows,
                                stream=stream,
                                decode_scratch=self.scratch,
                                expert_sidecar=expert_sidecar,
                                linear_state_rows=(
                                    self._verify_linear_state_row_pair(layer_id)
                                    if capture_linear_state_rows
                                    else None
                                ),
                                commit_final_linear_state=not bool(defer_linear_state_commit),
                                stage_timings=stage_timings,
                                sync_stage_timings=sync_stage_timings,
                                stage_prefix="target_block_linear_attn",
                            )
                        elif layer_type == FULL_ATTENTION:
                            key_cache, value_cache = self.scratch.full_cache(layer_id)
                            layer_scratch = replace(bulk_scratch, key_cache=key_cache, value_cache=value_cache)
                            if end < 1024:
                                self.runner._run_full_attention_decode_batch_layer_rows(
                                    layer_id,
                                    src_chunk_ptr,
                                    dst_chunk_ptr,
                                    layer_scratch,
                                    stream=stream,
                                    expert_sidecar=expert_sidecar,
                                    stage_timings=stage_timings,
                                    sync_stage_timings=sync_stage_timings,
                                    stage_prefix="target_block_full_attn",
                                )
                            else:
                                self.runner._run_full_attention_prefill_layer_aotriton(
                                    layer_id,
                                    src_chunk_ptr,
                                    dst_chunk_ptr,
                                    layer_scratch,
                                    stream=stream,
                                    expert_sidecar=expert_sidecar,
                                )
                        else:
                            raise ValueError(f"unsupported GGUF layer type {layer_type!r}")
                    finally:
                        if expert_sidecar is not None:
                            expert_sidecar.free(runtime=runtime)
                        if stage_timings is not None:
                            layer_ms = (time.perf_counter() - t_layer0) * 1000
                            if layer_type == LINEAR_ATTENTION:
                                add_verify_stage("target_block_linear_attn_layers", layer_ms)
                            elif layer_type == FULL_ATTENTION:
                                add_verify_stage("target_block_full_attn_layers", layer_ms)
                            else:
                                add_verify_stage("target_block_other_layers", layer_ms)
                            add_verify_stage("target_block_layer_total", layer_ms)
                    src, dst = dst, src
                t_output0 = time.perf_counter() if stage_timings is not None else 0.0
                final_scratch = self._bulk_prefill_scratch.for_chunk(
                    start,
                    rows,
                    total_tokens=end,
                    runtime=runtime,
                    stream=stream,
                )
                output_norm_weight_ptr = self.runner.weights.root("output_norm").allocation().tensor.ptr
                gguf_rmsnorm_bf16_f32_weight(
                    src.ptr + start * row_nbytes,
                    output_norm_weight_ptr,
                    final_scratch.norm.ptr,
                    rows=rows,
                    hidden_size=self.runner.hidden_size,
                    eps=self.runner.weights.config.rms_norm_eps,
                    stream=stream,
                    runtime=runtime,
                )
                gguf_rmsnorm_bf16_f32_weight_out_f32(
                    src.ptr + start * row_nbytes,
                    output_norm_weight_ptr,
                    hidden_seed_buf.ptr,
                    rows=rows,
                    hidden_size=self.runner.hidden_size,
                    eps=self.runner.weights.config.rms_norm_eps,
                    stream=stream,
                    runtime=runtime,
                )
                runtime.memcpy_async(
                    self.scratch.hidden_seed_fp32.ptr,
                    hidden_seed_buf.ptr + (rows - 1) * self.runner.hidden_size * DType.FP32.itemsize,
                    self.runner.hidden_size * DType.FP32.itemsize,
                    HipMemcpyKind.DEVICE_TO_DEVICE,
                    stream,
                )
                if sync_stages:
                    runtime.device_synchronize()
                add_verify_stage(
                    "target_block_output_norm_hidden",
                    (time.perf_counter() - t_output0) * 1000 if stage_timings is not None else 0.0,
                )
            if advance_state_only:
                # Accepted-prefix replay: target tokens already known from the
                # first full-block pass and discarded here. Skip the per-row
                # LM-head vocab GEMV + greedy sampling; keep the FP32 hidden rows
                # (decode continuity) and the linear/KV state advance above.
                t_sample0 = time.perf_counter() if stage_timings is not None else 0.0
                token_host = np.ascontiguousarray(tokens, dtype=np.int64)
                runtime.device_synchronize()
                add_verify_stage(
                    "target_block_lm_head_sample",
                    (time.perf_counter() - t_sample0) * 1000 if stage_timings is not None else 0.0,
                )
            else:
                # rows 2-6: default to the batched lm-head path so the Q6_K t16
                # rowtile GEMV reads the 417MB head once across all block rows
                # (vs the per-row loop re-reading it per row). Bit-exact; the env
                # flag can still force it for other row counts.
                t_sample0 = time.perf_counter() if stage_timings is not None else 0.0
                row_lm_head = _gguf_verify_row_lm_head_enabled() or (2 <= rows <= 6)
                if row_lm_head:
                    token_host = self._sample_target_block_rows_from_hidden(
                        final_scratch.norm.ptr,
                        rows,
                        stream=stream,
                    )
                else:
                    token_host = np.empty((rows,), dtype=np.int64)
                    for row in range(rows):
                        self._sample_device_from_hidden(
                            final_scratch.norm.ptr + row * row_nbytes,
                            stream=stream,
                        )
                        record_i64_scalar_indexed(
                            self._lm_out_index.ptr,
                            token_ids_buf.ptr,
                            token_counter_buf.ptr,
                            rows,
                            stream=stream,
                            library=self._runtime_state_library,
                            runtime=runtime,
                        )
                runtime.device_synchronize()
                if not row_lm_head:
                    copy_device_to_host(host_array_ptr(token_host), token_ids_buf, token_host.nbytes, runtime=runtime)
                add_verify_stage(
                    "target_block_lm_head_sample",
                    (time.perf_counter() - t_sample0) * 1000 if stage_timings is not None else 0.0,
                )
            t_hidden0 = time.perf_counter() if stage_timings is not None else 0.0
            hidden_host = np.empty((rows, self.runner.hidden_size), dtype=np.float32)
            copy_device_to_host(host_array_ptr(hidden_host), hidden_seed_buf, hidden_host.nbytes, runtime=runtime)
            add_verify_stage(
                "target_block_hidden_readback",
                (time.perf_counter() - t_hidden0) * 1000 if stage_timings is not None else 0.0,
            )
        finally:
            pass
        t_cursor0 = time.perf_counter() if stage_timings is not None else 0.0
        self._verify_hidden_seed_rows_populated = rows
        self._position = end
        self.scratch.position_host[0] = end
        self.scratch.context_host[0] = end + 1
        set_decode_position_i64(
            self.scratch.position_buf.ptr,
            self.scratch.context_buf.ptr,
            end,
            library=self._runtime_state_library,
            runtime=runtime,
        )
        self._hidden_seed_fp32_populated = True
        add_verify_stage(
            "target_block_cursor_update",
            (time.perf_counter() - t_cursor0) * 1000 if stage_timings is not None else 0.0,
        )
        return Qwen35GGUFBlockVerifyResult(
            input_token_ids=[int(token) for token in tokens.tolist()],
            token_ids=[int(token) for token in token_host.tolist()],
            hidden_seeds=np.ascontiguousarray(hidden_host, dtype=np.float32),
            start_position=start,
            linear_state_rows_captured=bool(capture_linear_state_rows),
        )

    def verify_target_block_serial_exact(
        self,
        input_token_ids: list[int] | tuple[int, ...],
        *,
        capture_linear_state_rows: bool = False,
        stream: int = 0,
    ) -> Qwen35GGUFBlockVerifyResult:
        """Consume a continuation block with the token-serial decode path.

        This is a correctness baseline for rollback-slot work.  It uses the
        same per-token kernels as :meth:`step`, then stages each hidden row and,
        optionally, each Conv/GDN state row for direct commit.  It deliberately
        does not amortize target weight loads.
        """

        if stream != 0:
            raise ValueError("serial-exact target block currently supports only the default stream")
        if self.runner is None or self.runner.weights is None or self.scratch is None:
            raise RuntimeError("GGUF resident session is closed")
        rows = int(len(input_token_ids))
        if rows <= 0:
            raise ValueError("input_token_ids must be non-empty")
        start = int(self._position)
        end = start + rows
        if end > int(self.scratch.max_positions):
            raise ValueError(f"target block end position {end} exceeds cache capacity {self.scratch.max_positions}")
        tokens = np.asarray([int(token) for token in input_token_ids], dtype=np.int64)
        for token in tokens.tolist():
            if token < 0 or token >= self.runner.vocab_size:
                raise ValueError(f"token_id {token} outside [0, {self.runner.vocab_size})")

        runtime = self.runtime or get_hip_runtime()
        self._ensure_verify_block_buffers(rows, runtime=runtime)
        if capture_linear_state_rows:
            self._ensure_verify_linear_state_row_buffers(rows, runtime=runtime)
        if self._verify_hidden_seed_buf is None:
            raise RuntimeError("GGUF verifier hidden-seed buffer is closed")

        token_host = np.empty((rows,), dtype=np.int64)
        hidden_row_nbytes = self.runner.hidden_size * DType.FP32.itemsize
        for row, token in enumerate(tokens.tolist()):
            result = self.step(
                int(token),
                return_logits=False,
                capture_hidden_seed_fp32=True,
            )
            token_host[row] = int(result.token_id)
            runtime.memcpy_async(
                self._verify_hidden_seed_buf.ptr + row * hidden_row_nbytes,
                self.scratch.hidden_seed_fp32.ptr,
                hidden_row_nbytes,
                HipMemcpyKind.DEVICE_TO_DEVICE,
                stream,
            )
            if capture_linear_state_rows:
                self._record_current_linear_state_row(row, stream=stream)

        runtime.device_synchronize()
        hidden_host = np.empty((rows, self.runner.hidden_size), dtype=np.float32)
        copy_device_to_host(host_array_ptr(hidden_host), self._verify_hidden_seed_buf, hidden_host.nbytes, runtime=runtime)
        self._verify_hidden_seed_rows_populated = rows
        self._hidden_seed_fp32_populated = True
        return Qwen35GGUFBlockVerifyResult(
            input_token_ids=[int(token) for token in tokens.tolist()],
            token_ids=[int(token) for token in token_host.tolist()],
            hidden_seeds=np.ascontiguousarray(hidden_host, dtype=np.float32),
            start_position=start,
            linear_state_rows_captured=bool(capture_linear_state_rows),
        )

    def _load_expert_sidecar_host_layer(self, layer_id: int) -> dict[str, GGUFExpertPackedTensor]:
        if self._expert_sidecar_reader is None or self._expert_sidecar_model_map is None:
            raise RuntimeError("GGUF expert sidecar loading was not enabled for this session")
        layer_map = self._expert_sidecar_model_map.layer(layer_id)  # type: ignore[attr-defined]
        tensors: dict[str, GGUFExpertPackedTensor] = {}
        for slot in ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"):
            tensor_info = layer_map.tensor(slot)
            cache_path = expert_sidecar_cache_path(
                self._expert_sidecar_reader.info,
                tensor_info,
                cache_dir=self.expert_sidecar_cache_dir,
            )
            if cache_path.exists():
                packed = load_packed_expert_tensor(cache_path)
            else:
                if self.require_expert_sidecar:
                    raise FileNotFoundError(f"missing cached GGUF expert sidecar for {tensor_info.name}: {cache_path}")
                packed = build_packed_expert_tensor_from_reader(self._expert_sidecar_reader, tensor_info, slot=slot)
                save_packed_expert_tensor(cache_path, packed)
            tensors[slot] = packed
        return tensors

    def _load_expert_sidecar_device_layer(self, layer_id: int, *, runtime: HipRuntime) -> _DeviceExpertLayerSidecar:
        host_tensors = (
            self._expert_sidecar_host_layers[layer_id]
            if self._expert_sidecar_host_layers is not None
            else self._load_expert_sidecar_host_layer(layer_id)
        )
        tensors: dict[str, _DeviceExpertPackedTensor] = {}
        try:
            for slot, packed in host_tensors.items():
                tensors[slot] = _DeviceExpertPackedTensor.from_host(packed, runtime=runtime)
            return _DeviceExpertLayerSidecar(tensors=tensors)
        except Exception:
            for tensor in reversed(tuple(tensors.values())):
                tensor.free(runtime=runtime)
            raise

    def step(
        self,
        token_id: int,
        position: int | None = None,
        *,
        return_logits: bool = True,
        capture_hidden_seed_fp32: bool = False,
    ) -> Qwen35GGUFNextTokenProbeResult:
        """Consume one generated token and return the next greedy token.

        ``position`` is optional because the session tracks its own decode
        cursor.  When supplied, it is validated to catch caller/context drift.

        Decode runs inside ``gemv_decode_session(self.use_gemv_decode)`` so
        the P9.B6 opt-in routes ``rows == 1`` GGUF projections through the
        ``pack8_gemv_decode_*`` family when registered (default off).
        """

        if position is not None and int(position) != self._position:
            raise ValueError(f"position {position} does not match session cursor {self._position}")
        with gemv_decode_session(self.use_gemv_decode):
            hidden_ptr = self._run_token_to_final_hidden(
                int(token_id),
                position=self._position,
                capture_hidden_seed_fp32=bool(capture_hidden_seed_fp32),
            )
            self._position += 1
            return self._sample_from_hidden(hidden_ptr, return_logits=return_logits)

    def _run_token_to_final_hidden(
        self,
        token_id: int,
        *,
        position: int,
        stream: int = 0,
        capture_hidden_seed_fp32: bool = False,
    ) -> int:
        if self._token_buf is None:
            raise RuntimeError("GGUF resident session buffers are closed")
        self._set_full_attention_position_device(position, stream=stream)
        self._set_token_id_device(int(token_id), stream=stream)
        return self._run_current_hidden_to_final_hidden(
            position=position,
            stream=stream,
            capture_hidden_seed_fp32=bool(capture_hidden_seed_fp32),
        )

    def capture_linear_attention_boundary(
        self,
        token_id: int,
        *,
        position: int,
        layer_id: int = 0,
        stream: int = 0,
    ) -> Qwen35GGUFLinearAttentionBoundaryCapture:
        """Run one decode linear-attention boundary and copy diagnostic buffers.

        This is a correctness-debug tap, not a generation fast path.  It mutates
        the resident decode state exactly like ``_run_linear_attention_attn_only``
        for the selected token/layer so captured buffers can be compared against
        CPU or llama.cpp boundary oracles.
        """

        if self.runner is None or self.runner.weights is None or self.scratch is None:
            raise RuntimeError("GGUF resident session is closed")
        if self._hidden_a is None:
            raise RuntimeError("GGUF resident session buffers are closed")
        if position < 0:
            raise ValueError("position must be non-negative")
        layer_types = self.runner.weights.config.layer_types
        if layer_id < 0 or layer_id >= len(layer_types):
            raise ValueError(f"layer_id {layer_id} outside resident layer range")
        if layer_types[layer_id] != LINEAR_ATTENTION:
            raise ValueError(f"layer {layer_id} is not a linear_attention layer")

        runtime = self.runtime or get_hip_runtime()
        self._hidden_seed_fp32_populated = False
        self._set_full_attention_position_device(position, stream=stream)
        self._set_token_id_device(int(token_id), stream=stream)
        self.runner._run_linear_attention_attn_only(
            layer_id,
            self._hidden_a.ptr,
            self.scratch.attn_out.ptr,
            self.scratch,
            stream=stream,
        )
        if stream:
            runtime.stream_synchronize(stream)
        else:
            runtime.device_synchronize()

        cfg = self.runner.weights.config
        rank = int(cfg.ssm_time_step_rank)
        hidden_size = int(self.runner.hidden_size)
        linear_qkv_width = int(self.runner.linear_qkv_width)
        ssm_inner_size = int(cfg.ssm_inner_size)
        alpha_ptr = int(self.scratch.linear_alpha.ptr)
        beta_ptr = int(self.scratch.linear_beta.ptr)

        return Qwen35GGUFLinearAttentionBoundaryCapture(
            layer_id=int(layer_id),
            token_id=int(token_id),
            position=int(position),
            hidden_size=hidden_size,
            ssm_time_step_rank=rank,
            linear_qkv_width=linear_qkv_width,
            ssm_inner_size=ssm_inner_size,
            attn_norm_f32=_copy_bf16_ptr_to_host_f32(
                int(self.scratch.norm.ptr), hidden_size, runtime=runtime
            ),
            linear_qkv_f32=_copy_bf16_ptr_to_host_f32(
                int(self.scratch.linear_qkv.ptr), linear_qkv_width, runtime=runtime
            ),
            linear_z_f32=_copy_bf16_ptr_to_host_f32(
                int(self.scratch.linear_z.ptr), ssm_inner_size, runtime=runtime
            ),
            ssm_alpha_f32=_copy_bf16_ptr_to_host_f32(alpha_ptr, rank, runtime=runtime),
            ssm_beta_f32=_copy_bf16_ptr_to_host_f32(beta_ptr, rank, runtime=runtime),
            conv_out_f32=_copy_f32_ptr_to_host(
                int(self.scratch.conv_out.ptr), linear_qkv_width, runtime=runtime
            ),
            recurrent_out_f32=_copy_f32_ptr_to_host(
                int(self.scratch.recurrent_out.ptr), ssm_inner_size, runtime=runtime
            ),
            recurrent_bf16_f32=_copy_bf16_ptr_to_host_f32(
                int(self.scratch.recurrent_bf16.ptr), ssm_inner_size, runtime=runtime
            ),
            attn_out_f32=_copy_bf16_ptr_to_host_f32(
                int(self.scratch.attn_out.ptr), hidden_size, runtime=runtime
            ),
        )

    def capture_attention_layer(
        self,
        token_id: int,
        *,
        position: int,
        layer_id: int = 0,
        stream: int = 0,
        run_preceding_layers: bool = False,
    ) -> Qwen35GGUFLinearAttentionLayerCapture:
        """Run one full attention layer and copy post-FFN boundary buffers.

        When ``run_preceding_layers`` is true, the selected token is first run
        through layers ``[0, layer_id)`` so the captured boundary reflects the
        in-stack path instead of applying ``layer_id`` directly to the token
        embedding.
        """

        if self.runner is None or self.runner.weights is None or self.scratch is None:
            raise RuntimeError("GGUF resident session is closed")
        if self._hidden_a is None or self._hidden_b is None:
            raise RuntimeError("GGUF resident session buffers are closed")
        if position < 0:
            raise ValueError("position must be non-negative")
        layer_types = self.runner.weights.config.layer_types
        if layer_id < 0 or layer_id >= len(layer_types):
            raise ValueError(f"layer_id {layer_id} outside resident layer range")
        layer_type = str(layer_types[layer_id])
        if layer_type not in (LINEAR_ATTENTION, FULL_ATTENTION):
            raise ValueError(f"unsupported GGUF layer type {layer_type!r}")

        runtime = self.runtime or get_hip_runtime()
        self._hidden_seed_fp32_populated = False
        self._set_full_attention_position_device(position, stream=stream)
        self._set_token_id_device(int(token_id), stream=stream)
        src = self._hidden_a
        dst = self._hidden_b
        if run_preceding_layers:
            for prev_layer_id, prev_layer_type in enumerate(layer_types[:layer_id]):
                if prev_layer_type == LINEAR_ATTENTION:
                    self.runner._run_linear_attention_layer(
                        prev_layer_id,
                        src.ptr,
                        dst.ptr,
                        self.scratch,
                        stream=stream,
                    )
                elif prev_layer_type == FULL_ATTENTION:
                    self.runner._run_full_attention_layer(
                        prev_layer_id,
                        src.ptr,
                        dst.ptr,
                        self.scratch,
                        position=position,
                        stream=stream,
                    )
                else:
                    raise ValueError(f"unsupported GGUF layer type {prev_layer_type!r}")
                src, dst = dst, src
        target_src_ptr = int(src.ptr)
        target_dst_ptr = int(dst.ptr)
        if layer_type == LINEAR_ATTENTION:
            self.runner._run_linear_attention_layer(
                layer_id,
                target_src_ptr,
                target_dst_ptr,
                self.scratch,
                stream=stream,
            )
        else:
            self.runner._run_full_attention_layer(
                layer_id,
                target_src_ptr,
                target_dst_ptr,
                self.scratch,
                position=position,
                stream=stream,
            )
        if stream:
            runtime.stream_synchronize(stream)
        else:
            runtime.device_synchronize()

        cfg = self.runner.weights.config
        hidden_size = int(self.runner.hidden_size)
        top_k = int(cfg.expert_used_count) if cfg.is_moe else 1
        down_ptr = int(
            self.scratch.moe_down_out.ptr if cfg.is_moe else self.scratch.ffn_down.ptr
        )
        down_elements = hidden_size * top_k if cfg.is_moe else hidden_size
        moe_shared_out = None
        moe_routing_weights = None
        moe_shared_gate = None
        moe_selected_experts = None
        if cfg.is_moe:
            moe_shared_out = _copy_bf16_ptr_to_host_f32(
                int(self.scratch.moe_shared_out.ptr), hidden_size, runtime=runtime
            )
            moe_routing_weights = _copy_f32_ptr_to_host(
                int(self.scratch.moe_routing_weights.ptr), top_k, runtime=runtime
            )
            shared_gate_ptr = (
                int(self.scratch.moe_router_logits.ptr)
                + int(cfg.expert_count) * DType.FP32.itemsize
            )
            moe_shared_gate = _copy_f32_ptr_to_host(shared_gate_ptr, 1, runtime=runtime)
            moe_selected_experts = _copy_i64_ptr_to_host(
                int(self.scratch.moe_selected_experts.ptr), top_k, runtime=runtime
            )

        return Qwen35GGUFLinearAttentionLayerCapture(
            layer_id=int(layer_id),
            layer_type=layer_type,
            token_id=int(token_id),
            position=int(position),
            hidden_size=hidden_size,
            is_moe=bool(cfg.is_moe),
            top_k=top_k,
            preceding_layer_count=int(layer_id) if run_preceding_layers else 0,
            hidden_in_f32=_copy_bf16_ptr_to_host_f32(
                target_src_ptr, hidden_size, runtime=runtime
            ),
            attn_out_f32=_copy_bf16_ptr_to_host_f32(
                int(self.scratch.attn_out.ptr), hidden_size, runtime=runtime
            ),
            post_norm_f32=_copy_bf16_ptr_to_host_f32(
                int(self.scratch.post_norm.ptr), hidden_size, runtime=runtime
            ),
            residual_f32=_copy_bf16_ptr_to_host_f32(
                int(self.scratch.residual.ptr), hidden_size, runtime=runtime
            ),
            ffn_or_moe_down_f32=_copy_bf16_ptr_to_host_f32(
                down_ptr, down_elements, runtime=runtime
            ),
            layer_out_f32=_copy_bf16_ptr_to_host_f32(
                target_dst_ptr, hidden_size, runtime=runtime
            ),
            moe_shared_out_f32=moe_shared_out,
            moe_routing_weights_f32=moe_routing_weights,
            moe_shared_gate_f32=moe_shared_gate,
            moe_selected_experts_i64=moe_selected_experts,
        )

    def capture_linear_attention_layer(
        self,
        token_id: int,
        *,
        position: int,
        layer_id: int = 0,
        stream: int = 0,
    ) -> Qwen35GGUFLinearAttentionLayerCapture:
        """Run one full linear-attention layer and copy post-FFN boundary buffers."""

        if self.runner is None or self.runner.weights is None:
            raise RuntimeError("GGUF resident session is closed")
        layer_types = self.runner.weights.config.layer_types
        if layer_id < 0 or layer_id >= len(layer_types):
            raise ValueError(f"layer_id {layer_id} outside resident layer range")
        if layer_types[layer_id] != LINEAR_ATTENTION:
            raise ValueError(f"layer {layer_id} is not a linear_attention layer")
        return self.capture_attention_layer(
            token_id,
            position=position,
            layer_id=layer_id,
            stream=stream,
        )

    def _run_current_hidden_to_final_hidden(
        self,
        *,
        position: int,
        stream: int = 0,
        attention_max_context_len: int | None = None,
        capture_hidden_seed_fp32: bool = False,
    ) -> int:
        if self.runner is None or self.scratch is None:
            raise RuntimeError("GGUF resident session is closed")
        if self._hidden_a is None or self._hidden_b is None:
            raise RuntimeError("GGUF resident session buffers are closed")
        assert self.runner.weights is not None
        self._hidden_seed_fp32_populated = False
        self.scratch.position_host[0] = int(position)
        self.scratch.context_host[0] = int(position) + 1
        src = self._hidden_a
        dst = self._hidden_b
        moe_graph = self._moe_graph_for_decode()
        for layer_id, layer_type in enumerate(self.runner.weights.config.layer_types):
            if moe_graph is not None:
                self._run_decode_layer_graphed(
                    layer_id,
                    layer_type,
                    src.ptr,
                    dst.ptr,
                    moe_graph,
                    position=position,
                    stream=stream,
                    attention_max_context_len=attention_max_context_len,
                )
            elif layer_type == LINEAR_ATTENTION:
                self.runner._run_linear_attention_layer(layer_id, src.ptr, dst.ptr, self.scratch, stream=stream)
            elif layer_type == FULL_ATTENTION:
                self.runner._run_full_attention_layer(
                    layer_id,
                    src.ptr,
                    dst.ptr,
                    self.scratch,
                    position=position,
                    stream=stream,
                    attention_max_context_len=attention_max_context_len,
                )
            else:
                raise ValueError(f"unsupported GGUF layer type {layer_type!r}")
            src, dst = dst, src
        return self._run_output_norm_hidden(
            src.ptr,
            self.scratch.norm.ptr,
            stream=stream,
            capture_hidden_seed_fp32=bool(capture_hidden_seed_fp32),
        )

    def _moe_graph_for_decode(self) -> MoeGraphCache | None:
        """Return the rows==1 MoE FFN graph cache, lazily created when enabled.

        Returns None (eager decode) unless HIPENGINE_GGUF_MOE_GRAPH is set.  The
        cache is bound to this session's resident buffers and torn down in
        :meth:`close` while those buffers are still alive.
        """
        if not _gguf_moe_graph_enabled():
            return None
        if self._moe_graph is None:
            self._moe_graph = MoeGraphCache(
                self.runtime or get_hip_runtime(), enabled=True
            )
        return self._moe_graph

    def _run_decode_layer_graphed(
        self,
        layer_id: int,
        layer_type: str,
        src_ptr: int,
        dst_ptr: int,
        moe_graph: MoeGraphCache,
        *,
        position: int,
        stream: int = 0,
        attention_max_context_len: int | None = None,
    ) -> None:
        """Run one decode layer with the stateful attention eager and the
        stateless MoE FFN routed through the capture/replay cache.

        The attention (GDN conv/recurrent or full-attn paged-KV write) is
        position-dependent and stateful, so it MUST stay eager; only the FFN
        (rmsnorm -> router -> selected experts -> shared expert -> combine) is
        graph-captured.  ``key`` is ``(layer_id, src_ptr, dst_ptr)``: stable per
        layer across tokens because the decode loop ping-pongs ``hidden_a``/
        ``hidden_b`` with fixed parity and scratch is session-resident.
        """
        runner = self.runner
        scratch = self.scratch
        attn_out_ptr = scratch.attn_out.ptr
        if layer_type == LINEAR_ATTENTION:
            runner._run_linear_attention_attn_only(
                layer_id, src_ptr, attn_out_ptr, scratch, stream=stream
            )
        elif layer_type == FULL_ATTENTION:
            runner._run_full_attention_attn_only(
                layer_id,
                src_ptr,
                attn_out_ptr,
                scratch,
                position=position,
                stream=stream,
                attention_max_context_len=attention_max_context_len,
            )
        else:
            raise ValueError(f"unsupported GGUF layer type {layer_type!r}")

        def _ffn(capture_stream: int) -> None:
            runner._run_post_attention_ffn(
                layer_id, src_ptr, attn_out_ptr, dst_ptr, scratch, stream=capture_stream
            )

        moe_graph.run(
            (layer_id, int(src_ptr), int(dst_ptr)),
            eager=_ffn,
            out_ptr=int(dst_ptr),
            out_nbytes=runner.hidden_size * DType.BF16.itemsize,
            stream=stream,
        )

    def _run_output_norm_hidden(
        self,
        src_ptr: int,
        out_ptr: int,
        *,
        stream: int = 0,
        capture_hidden_seed_fp32: bool = False,
    ) -> int:
        if self.runner is None or self.runner.weights is None or self.scratch is None:
            raise RuntimeError("GGUF resident session is closed")
        runtime = self.runtime or get_hip_runtime()
        self._hidden_seed_fp32_populated = False
        output_norm_weight_ptr = self.runner.weights.root("output_norm").allocation().tensor.ptr
        gguf_rmsnorm_bf16_f32_weight(
            src_ptr,
            output_norm_weight_ptr,
            out_ptr,
            rows=1,
            hidden_size=self.runner.hidden_size,
            eps=self.runner.weights.config.rms_norm_eps,
            stream=stream,
            runtime=runtime,
        )
        if capture_hidden_seed_fp32:
            gguf_rmsnorm_bf16_f32_weight_out_f32(
                src_ptr,
                output_norm_weight_ptr,
                self.scratch.hidden_seed_fp32.ptr,
                rows=1,
                hidden_size=self.runner.hidden_size,
                eps=self.runner.weights.config.rms_norm_eps,
                stream=stream,
                runtime=runtime,
            )
            self._hidden_seed_fp32_populated = True
        return int(out_ptr)

    def _set_token_id_device(self, token_id: int, *, stream: int = 0) -> None:
        if self.runner is None or self._token_buf is None:
            raise RuntimeError("GGUF resident session is closed")
        if token_id < 0 or token_id >= self.runner.vocab_size:
            raise ValueError(f"token_id {token_id} outside [0, {self.runner.vocab_size})")
        set_i64_scalar(
            self._token_buf.ptr,
            int(token_id),
            stream=stream,
            library=self._runtime_state_library,
            runtime=self.runtime or get_hip_runtime(),
        )
        self._set_token_embedding_from_ptr(self._token_buf.ptr, stream=stream)

    def _set_token_embedding_from_ptr(self, token_id_ptr: int, *, stream: int = 0) -> None:
        if self.runner is None or self._hidden_a is None:
            raise RuntimeError("GGUF resident session is closed")
        assert self.runner.weights is not None
        launch_gguf_embedding(
            self.runner.weights.root("token_embedding"),
            token_id_ptr,
            self._hidden_a.ptr,
            rows=1,
            hidden_size=self.runner.hidden_size,
            vocab_size=self.runner.vocab_size,
            stream=stream,
            runtime=self.runtime or get_hip_runtime(),
        )

    def _set_full_attention_position_device(self, position: int, *, stream: int = 0) -> None:
        if self.scratch is None:
            raise RuntimeError("GGUF resident session is closed")
        if position < 0 or position >= self.scratch.max_positions:
            raise ValueError(f"GGUF resident full-attention position {position} exceeds cache capacity {self.scratch.max_positions}")
        self.scratch.position_host[0] = int(position)
        self.scratch.context_host[0] = int(position) + 1
        set_decode_position_i64(
            self.scratch.position_buf.ptr,
            self.scratch.context_buf.ptr,
            int(position),
            stream=stream,
            library=self._runtime_state_library,
            runtime=self.runtime or get_hip_runtime(),
        )

    def _sample_device_from_hidden(self, hidden_ptr: int, *, stream: int = 0) -> None:
        if (
            self.runner is None
            or self._logits_buf is None
            or self._lm_block_values is None
            or self._lm_block_indices is None
            or self._lm_out_index is None
            or self._lm_out_value is None
        ):
            raise RuntimeError("GGUF resident session is closed")
        assert self.runner.weights is not None
        runtime = self.runtime or get_hip_runtime()
        launch_gguf_linear(
            self.runner.weights.root("lm_head"),
            hidden_ptr,
            self._logits_buf.ptr,
            rows=1,
            in_features=self.runner.hidden_size,
            out_features=self.runner.vocab_size,
            output_dtype=GGUF_OUTPUT_F32,
            stream=stream,
            runtime=runtime,
        )
        argmax_f32(
            self._logits_buf.ptr,
            self._lm_block_values.ptr,
            self._lm_block_indices.ptr,
            self._lm_out_index.ptr,
            self._lm_out_value.ptr,
            self.runner.vocab_size,
            threads=self._lm_head_threads,
            stream=stream,
            library=self._lm_head_library,
            runtime=runtime,
        )

    def _ensure_verify_block_buffers(self, rows: int, *, runtime: HipRuntime) -> None:
        rows = int(rows)
        if rows <= 0:
            raise ValueError("verify block rows must be positive")
        if rows <= int(self._verify_block_rows_capacity):
            return
        for buffer in (
            self._verify_token_counter_i64,
            self._verify_token_ids_i64,
            self._verify_hidden_seed_buf,
        ):
            if buffer is not None:
                free(buffer, runtime=runtime)
        if self.runner is None:
            raise RuntimeError("GGUF resident session is closed")
        self._verify_hidden_seed_buf = malloc(
            rows * self.runner.hidden_size * DType.FP32.itemsize,
            runtime=runtime,
        )
        self._verify_token_ids_i64 = malloc(rows * DType.INT64.itemsize, runtime=runtime)
        self._verify_token_counter_i64 = malloc(DType.INT64.itemsize, runtime=runtime)
        self._verify_block_rows_capacity = rows
        self._verify_hidden_seed_rows_populated = 0

    def _free_verify_linear_state_row_buffers(self, *, runtime: HipRuntime) -> None:
        for buffer in (
            *self._verify_linear_recurrent_state_rows,
            *self._verify_linear_conv_state_rows,
        ):
            if buffer is not None:
                free(buffer, runtime=runtime)
        self._verify_linear_conv_state_rows = ()
        self._verify_linear_recurrent_state_rows = ()
        self._verify_linear_state_rows_capacity = 0

    def _ensure_verify_linear_state_row_buffers(self, rows: int, *, runtime: HipRuntime) -> None:
        rows = int(rows)
        if rows <= 0:
            raise ValueError("verify linear-state rows must be positive")
        if rows <= int(self._verify_linear_state_rows_capacity):
            return
        self._free_verify_linear_state_row_buffers(runtime=runtime)
        if self.runner is None or self.scratch is None:
            raise RuntimeError("GGUF resident session is closed")
        conv_rows: list[object | None] = []
        recurrent_rows: list[object | None] = []
        for conv_state, recurrent_state in zip(
            self.scratch.layer_conv_states,
            self.scratch.layer_recurrent_states,
            strict=True,
        ):
            if conv_state is None or recurrent_state is None:
                conv_rows.append(None)
                recurrent_rows.append(None)
                continue
            conv_rows.append(malloc(rows * int(conv_state.nbytes), runtime=runtime))
            recurrent_rows.append(malloc(rows * int(recurrent_state.nbytes), runtime=runtime))
        self._verify_linear_conv_state_rows = tuple(conv_rows)
        self._verify_linear_recurrent_state_rows = tuple(recurrent_rows)
        self._verify_linear_state_rows_capacity = rows

    def _verify_linear_state_row_pair(self, layer_id: int) -> tuple[object, object] | None:
        if (
            not self._verify_linear_conv_state_rows
            or not self._verify_linear_recurrent_state_rows
        ):
            return None
        conv_rows = self._verify_linear_conv_state_rows[layer_id]
        recurrent_rows = self._verify_linear_recurrent_state_rows[layer_id]
        if conv_rows is None or recurrent_rows is None:
            return None
        return conv_rows, recurrent_rows

    def _record_current_linear_state_row(
        self,
        row_index: int,
        *,
        stream: int = 0,
    ) -> None:
        """Stage the current resident Conv/GDN states into verifier row storage."""

        if self.scratch is None:
            raise RuntimeError("GGUF resident session is closed")
        row_index = int(row_index)
        if row_index < 0 or row_index >= int(self._verify_linear_state_rows_capacity):
            raise ValueError("row_index is outside captured linear-state rows")
        runtime = self.runtime or get_hip_runtime()
        for layer_id, (conv_state, recurrent_state) in enumerate(
            zip(self.scratch.layer_conv_states, self.scratch.layer_recurrent_states, strict=True)
        ):
            if conv_state is None or recurrent_state is None:
                continue
            pair = self._verify_linear_state_row_pair(layer_id)
            if pair is None:
                raise RuntimeError(f"linear-state rows for layer {layer_id} were not captured")
            conv_rows, recurrent_rows = pair
            runtime.memcpy_async(
                conv_rows.ptr + row_index * int(conv_state.nbytes),
                conv_state.ptr,
                int(conv_state.nbytes),
                HipMemcpyKind.DEVICE_TO_DEVICE,
                stream,
            )
            runtime.memcpy_async(
                recurrent_rows.ptr + row_index * int(recurrent_state.nbytes),
                recurrent_state.ptr,
                int(recurrent_state.nbytes),
                HipMemcpyKind.DEVICE_TO_DEVICE,
                stream,
            )

    def _fused_linear_state_commit_enabled(self) -> bool:
        value = os.environ.get("HIPENGINE_FUSED_LINEAR_STATE_COMMIT")
        if value is None:
            return True
        return value.strip().lower() not in {"0", "false", "no", "off"}

    def _chunked_linear_state_commit_enabled(self) -> bool:
        value = os.environ.get("HIPENGINE_LINEAR_STATE_COMMIT_CHUNKED")
        if value is None:
            return True
        return value.strip().lower() not in {"0", "false", "no", "off"}

    def _ensure_verify_linear_state_commit_tables(self, *, runtime: HipRuntime) -> bool:
        if self.scratch is None:
            raise RuntimeError("GGUF resident session is closed")
        conv_sources: list[int] = []
        recurrent_sources: list[int] = []
        conv_dests: list[int] = []
        recurrent_dests: list[int] = []
        conv_row_nbytes = 0
        recurrent_row_nbytes = 0
        for layer_id, (conv_state, recurrent_state) in enumerate(
            zip(self.scratch.layer_conv_states, self.scratch.layer_recurrent_states, strict=True)
        ):
            if conv_state is None or recurrent_state is None:
                continue
            pair = self._verify_linear_state_row_pair(layer_id)
            if pair is None:
                return False
            conv_rows, recurrent_rows = pair
            conv_nbytes = int(conv_state.nbytes)
            recurrent_nbytes = int(recurrent_state.nbytes)
            if conv_row_nbytes not in {0, conv_nbytes}:
                return False
            if recurrent_row_nbytes not in {0, recurrent_nbytes}:
                return False
            conv_row_nbytes = conv_nbytes
            recurrent_row_nbytes = recurrent_nbytes
            conv_sources.append(int(conv_rows.ptr))
            recurrent_sources.append(int(recurrent_rows.ptr))
            conv_dests.append(int(conv_state.ptr))
            recurrent_dests.append(int(recurrent_state.ptr))
        n_layers = len(conv_sources)
        if n_layers <= 0:
            return False
        tables_ready = (
            self._verify_linear_state_src_conv_table_buf is not None
            and self._verify_linear_state_src_recurrent_table_buf is not None
            and self._verify_linear_state_dst_conv_table_buf is not None
            and self._verify_linear_state_dst_recurrent_table_buf is not None
            and self._verify_linear_state_commit_row_i32_buf is not None
            and int(self._verify_linear_state_layer_count) == n_layers
            and int(self._verify_linear_state_conv_row_nbytes) == conv_row_nbytes
            and int(self._verify_linear_state_recurrent_row_nbytes) == recurrent_row_nbytes
        )
        if not tables_ready:
            table_nbytes = n_layers * np.dtype(np.uint64).itemsize
            new_buffers = (
                malloc(table_nbytes, runtime=runtime),
                malloc(table_nbytes, runtime=runtime),
                malloc(table_nbytes, runtime=runtime),
                malloc(table_nbytes, runtime=runtime),
                malloc(DType.INT32.itemsize, runtime=runtime),
            )
            self._verify_linear_state_src_conv_table_buf = new_buffers[0]
            self._verify_linear_state_src_recurrent_table_buf = new_buffers[1]
            self._verify_linear_state_dst_conv_table_buf = new_buffers[2]
            self._verify_linear_state_dst_recurrent_table_buf = new_buffers[3]
            self._verify_linear_state_commit_row_i32_buf = new_buffers[4]
            self._verify_linear_state_src_conv_host = np.zeros((n_layers,), dtype=np.uint64)
            self._verify_linear_state_src_recurrent_host = np.zeros((n_layers,), dtype=np.uint64)
            self._verify_linear_state_src_conv_cached = np.zeros((n_layers,), dtype=np.uint64)
            self._verify_linear_state_src_recurrent_cached = np.zeros((n_layers,), dtype=np.uint64)
            self._verify_linear_state_dst_conv_host = np.asarray(conv_dests, dtype=np.uint64)
            self._verify_linear_state_dst_recurrent_host = np.asarray(recurrent_dests, dtype=np.uint64)
            self._verify_linear_state_conv_row_nbytes = conv_row_nbytes
            self._verify_linear_state_recurrent_row_nbytes = recurrent_row_nbytes
            self._verify_linear_state_layer_count = n_layers
            self._buffers = (*self._buffers, *new_buffers)
            copy_host_to_device(
                self._verify_linear_state_dst_conv_table_buf,
                host_array_ptr(self._verify_linear_state_dst_conv_host),
                self._verify_linear_state_dst_conv_host.nbytes,
                runtime=runtime,
            )
            copy_host_to_device(
                self._verify_linear_state_dst_recurrent_table_buf,
                host_array_ptr(self._verify_linear_state_dst_recurrent_host),
                self._verify_linear_state_dst_recurrent_host.nbytes,
                runtime=runtime,
            )

        assert self._verify_linear_state_src_conv_host is not None
        assert self._verify_linear_state_src_recurrent_host is not None
        assert self._verify_linear_state_src_conv_cached is not None
        assert self._verify_linear_state_src_recurrent_cached is not None
        self._verify_linear_state_src_conv_host[:] = np.asarray(conv_sources, dtype=np.uint64)
        self._verify_linear_state_src_recurrent_host[:] = np.asarray(recurrent_sources, dtype=np.uint64)
        if not np.array_equal(self._verify_linear_state_src_conv_host, self._verify_linear_state_src_conv_cached):
            copy_host_to_device(
                self._verify_linear_state_src_conv_table_buf,
                host_array_ptr(self._verify_linear_state_src_conv_host),
                self._verify_linear_state_src_conv_host.nbytes,
                runtime=runtime,
            )
            np.copyto(self._verify_linear_state_src_conv_cached, self._verify_linear_state_src_conv_host)
        if not np.array_equal(
            self._verify_linear_state_src_recurrent_host,
            self._verify_linear_state_src_recurrent_cached,
        ):
            copy_host_to_device(
                self._verify_linear_state_src_recurrent_table_buf,
                host_array_ptr(self._verify_linear_state_src_recurrent_host),
                self._verify_linear_state_src_recurrent_host.nbytes,
                runtime=runtime,
            )
            np.copyto(self._verify_linear_state_src_recurrent_cached, self._verify_linear_state_src_recurrent_host)
        return True

    def _commit_verify_linear_state_row(
        self,
        row_index: int,
        *,
        position: int,
        stream: int = 0,
    ) -> None:
        """Commit a previously captured verifier row as the resident state."""

        if self.runner is None or self.scratch is None:
            raise RuntimeError("GGUF resident session is closed")
        if self._verify_hidden_seed_buf is None:
            raise RuntimeError("GGUF verifier hidden-seed buffer is closed")
        row_index = int(row_index)
        if row_index < 0 or row_index >= int(self._verify_linear_state_rows_capacity):
            raise ValueError("row_index is outside captured linear-state rows")
        runtime = self.runtime or get_hip_runtime()
        used_fused_commit = False
        if self._fused_linear_state_commit_enabled() and self._ensure_verify_linear_state_commit_tables(runtime=runtime):
            assert self._verify_linear_state_commit_row_i32_buf is not None
            commit_row = np.asarray([row_index], dtype=np.int32)
            copy_host_to_device(
                self._verify_linear_state_commit_row_i32_buf,
                host_array_ptr(commit_row),
                commit_row.nbytes,
                runtime=runtime,
            )
            linear_commit = (
                linear_state_pair_commit_chunked_i32
                if self._chunked_linear_state_commit_enabled()
                else linear_state_pair_commit_i32
            )
            if self._dflash_commit_library is None:
                self._dflash_commit_library = build_dflash_commit(
                    load=True,
                    compiler_version=self.compiler_version,
                    require_cached=self.require_cached_build,
                )
            linear_commit(
                self._verify_linear_state_src_conv_table_buf.ptr,
                self._verify_linear_state_dst_conv_table_buf.ptr,
                int(self._verify_linear_state_conv_row_nbytes),
                self._verify_linear_state_src_recurrent_table_buf.ptr,
                self._verify_linear_state_dst_recurrent_table_buf.ptr,
                int(self._verify_linear_state_recurrent_row_nbytes),
                self._verify_linear_state_commit_row_i32_buf.ptr,
                int(self._verify_linear_state_layer_count),
                stream=stream,
                library=self._dflash_commit_library,
                runtime=runtime,
            )
            used_fused_commit = True
        if not used_fused_commit:
            for layer_id, (conv_state, recurrent_state) in enumerate(
                zip(self.scratch.layer_conv_states, self.scratch.layer_recurrent_states, strict=True)
            ):
                if conv_state is None or recurrent_state is None:
                    continue
                pair = self._verify_linear_state_row_pair(layer_id)
                if pair is None:
                    raise RuntimeError(f"linear-state rows for layer {layer_id} were not captured")
                conv_rows, recurrent_rows = pair
                runtime.memcpy_async(
                    conv_state.ptr,
                    conv_rows.ptr + row_index * int(conv_state.nbytes),
                    int(conv_state.nbytes),
                    HipMemcpyKind.DEVICE_TO_DEVICE,
                    stream,
                )
                runtime.memcpy_async(
                    recurrent_state.ptr,
                    recurrent_rows.ptr + row_index * int(recurrent_state.nbytes),
                    int(recurrent_state.nbytes),
                    HipMemcpyKind.DEVICE_TO_DEVICE,
                    stream,
                )
        hidden_row_nbytes = self.runner.hidden_size * DType.FP32.itemsize
        runtime.memcpy_async(
            self.scratch.hidden_seed_fp32.ptr,
            self._verify_hidden_seed_buf.ptr + row_index * hidden_row_nbytes,
            hidden_row_nbytes,
            HipMemcpyKind.DEVICE_TO_DEVICE,
            stream,
        )
        end = int(position)
        self._position = end
        self.scratch.position_host[0] = end
        self.scratch.context_host[0] = end + 1
        set_decode_position_i64(
            self.scratch.position_buf.ptr,
            self.scratch.context_buf.ptr,
            end,
            library=self._runtime_state_library,
            runtime=runtime,
        )
        self._hidden_seed_fp32_populated = True

    def _ensure_verify_lm_head_buffers(self, rows: int, *, runtime: HipRuntime) -> None:
        rows = int(rows)
        if rows <= 0:
            raise ValueError("verify lm-head rows must be positive")
        if rows <= int(self._verify_lm_rows_capacity):
            return
        for buffer in (
            self._verify_lm_out_values,
            self._verify_lm_out_indices_i32,
            self._verify_lm_block_indices_i32,
            self._verify_lm_block_values,
            self._verify_lm_q8_1,
            self._verify_logits_buf,
        ):
            if buffer is not None:
                free(buffer, runtime=runtime)
        stage1_blocks = int(self._lm_head_stage1_blocks)
        vocab_size = int(self.runner.vocab_size if self.runner is not None else 0)
        if stage1_blocks <= 0 or vocab_size <= 0:
            raise RuntimeError("GGUF verifier lm-head buffers require initialized session state")
        self._verify_logits_buf = malloc(rows * vocab_size * DType.FP32.itemsize, runtime=runtime)
        block_capacity = max(stage1_blocks, vocab_size)
        self._verify_lm_block_values = malloc(rows * block_capacity * DType.FP32.itemsize, runtime=runtime)
        self._verify_lm_block_indices_i32 = malloc(rows * block_capacity * DType.INT32.itemsize, runtime=runtime)
        self._verify_lm_out_indices_i32 = malloc(rows * DType.INT32.itemsize, runtime=runtime)
        self._verify_lm_out_values = malloc(rows * DType.FP32.itemsize, runtime=runtime)
        self._verify_lm_q8_1 = malloc(_q8_1_workspace_bytes(rows, self.runner.hidden_size), runtime=runtime)
        self._verify_lm_rows_capacity = rows

    def _verify_lm_head_rowtile(
        self, hidden_ptr: int, out_ptr: int, rows: int, *, stream: int = 0, runtime=None
    ) -> bool:
        """Weight-amortized small-B (rows 2-6) verify lm-head GEMV.

        Reads the Q6_K t16 lm-head tiles ONCE across all block rows instead of
        re-reading the 417MB head per row (the per-row decode kernel's small-B
        over-read). Bit-exact vs the per-row decode kernel
        (tests/test_gguf_q6_k_t16_rowtile_gemv.py). Returns True if it handled the
        GEMV; False means the caller should fall back to launch_gguf_linear.
        """

        if rows < 2 or rows > 6 or self.runner is None or self.runner.weights is None:
            return False
        from hipengine.runtime.gguf_linear import (
            GGUF_ACTIVATION_BF16,
            resolve_gguf_linear_dispatch,
        )
        from hipengine.kernels.hip_gfx1100.quant.gguf_q6_k_t16_gemv import (
            gguf_q6_k_t16_gemv_rowtile_bf16_f32_out,
        )

        weight = self.runner.weights.root("lm_head")
        try:
            dispatch = resolve_gguf_linear_dispatch(
                weight,
                activation_dtype=GGUF_ACTIVATION_BF16,
                output_dtype=GGUF_OUTPUT_F32,
                backend="hip_gfx1100",
                rows=rows,
            )
            if dispatch.key.quant != "gguf_q6_k_t16_v1" or dispatch.abi != "t16":
                return False
            tiles_ptr = weight.allocation("tiles").tensor.ptr
        except Exception:
            return False
        gguf_q6_k_t16_gemv_rowtile_bf16_f32_out(
            hidden_ptr,
            tiles_ptr,
            out_ptr,
            rows,
            self.runner.hidden_size,
            self.runner.vocab_size,
            stream=stream,
            runtime=runtime,
        )
        return True

    def _verify_lm_head_q6_top1_dp4a(
        self, hidden_ptr: int, rows: int, *, stream: int = 0, runtime=None
    ) -> bool:
        """Accuracy-traded verifier lm-head path matching the llama-compat top-1 class."""

        if not _gguf_verify_lm_head_q6_top1_dp4a_enabled():
            return False
        if self.runner is None or self.runner.weights is None:
            raise RuntimeError("GGUF resident session is closed")
        if (
            self._verify_lm_q8_1 is None
            or self._verify_lm_block_values is None
            or self._verify_lm_block_indices_i32 is None
            or self._verify_lm_out_indices_i32 is None
            or self._verify_lm_out_values is None
        ):
            raise RuntimeError("GGUF verifier Q6 top-1 buffers are closed")
        weight = self.runner.weights.root("lm_head")
        try:
            tiles_ptr = weight.allocation("x8").tensor.ptr
        except Exception as exc:
            raise RuntimeError(
                "GGUF verifier Q6 top-1 dp4a requires "
                "HIPENGINE_GGUF_LM_HEAD_Q6_X8_SIDECAR=1 before materialization"
            ) from exc
        runtime = runtime or (self.runtime or get_hip_runtime())
        library = self._q6_pack8_library
        if library is None:
            library = build_gguf_q6_k_pack8_gemv(
                load=True,
                compiler_version=self.compiler_version,
                require_cached=self.require_cached_build,
            )
            self._q6_pack8_library = library
        gguf_q4_k_quantize_bf16_q8_1(
            hidden_ptr,
            self._verify_lm_q8_1.ptr,
            rows,
            self.runner.hidden_size,
            stream=stream,
            runtime=runtime,
        )
        gguf_q6_k_x8_gemv_decode_q8_1_dp4a_top1_gather_f32(
            self._verify_lm_q8_1.ptr,
            tiles_ptr,
            self._verify_lm_block_values.ptr,
            self._verify_lm_block_indices_i32.ptr,
            self._verify_lm_out_indices_i32.ptr,
            self._verify_lm_out_values.ptr,
            None,
            None,
            rows,
            self.runner.hidden_size,
            self.runner.vocab_size,
            0,
            stream=stream,
            library=library,
            runtime=runtime,
        )
        return True

    def _sample_target_block_rows_from_hidden(self, hidden_ptr: int, rows: int, *, stream: int = 0) -> np.ndarray:
        if self.runner is None or self.runner.weights is None:
            raise RuntimeError("GGUF resident session is closed")
        runtime = self.runtime or get_hip_runtime()
        rows = int(rows)
        self._ensure_verify_lm_head_buffers(rows, runtime=runtime)
        if (
            self._verify_logits_buf is None
            or self._verify_lm_block_values is None
            or self._verify_lm_block_indices_i32 is None
            or self._verify_lm_out_indices_i32 is None
            or self._verify_lm_out_values is None
        ):
            raise RuntimeError("GGUF verifier lm-head buffers are closed")
        direct_top1 = self._verify_lm_head_q6_top1_dp4a(hidden_ptr, rows, stream=stream, runtime=runtime)
        if not direct_top1:
            if not self._verify_lm_head_rowtile(
                hidden_ptr, self._verify_logits_buf.ptr, rows, stream=stream, runtime=runtime
            ):
                launch_gguf_linear(
                    self.runner.weights.root("lm_head"),
                    hidden_ptr,
                    self._verify_logits_buf.ptr,
                    rows=rows,
                    in_features=self.runner.hidden_size,
                    out_features=self.runner.vocab_size,
                    output_dtype=GGUF_OUTPUT_F32,
                    stream=stream,
                    runtime=runtime,
                )
            argmax_f32_rows_i32(
                self._verify_logits_buf.ptr,
                self._verify_lm_block_values.ptr,
                self._verify_lm_block_indices_i32.ptr,
                self._verify_lm_out_indices_i32.ptr,
                self._verify_lm_out_values.ptr,
                rows,
                self.runner.vocab_size,
                threads=self._lm_head_threads,
                stream=stream,
                library=self._lm_head_library,
                runtime=runtime,
            )
        if stream:
            runtime.stream_synchronize(stream)
        else:
            runtime.device_synchronize()
        token_i32 = np.empty((rows,), dtype=np.int32)
        copy_device_to_host(
            host_array_ptr(token_i32),
            DeviceBuffer(self._verify_lm_out_indices_i32.ptr, token_i32.nbytes),
            token_i32.nbytes,
            runtime=runtime,
        )
        token_i64 = token_i32.astype(np.int64, copy=False)
        if self._lm_out_index is not None and rows > 0:
            set_i64_scalar(
                self._lm_out_index.ptr,
                int(token_i64[-1]),
                library=self._runtime_state_library,
                runtime=runtime,
            )
        return np.ascontiguousarray(token_i64, dtype=np.int64)

    def _sample_from_hidden(self, hidden_ptr: int, *, return_logits: bool = True) -> Qwen35GGUFNextTokenProbeResult:
        self._sample_device_from_hidden(hidden_ptr)
        (self.runtime or get_hip_runtime()).device_synchronize()
        return self._read_sample(return_logits=return_logits)

    def _read_sample(self, *, return_logits: bool = True) -> Qwen35GGUFNextTokenProbeResult:
        if self.runner is None or self._logits_buf is None or self._logits_host is None:
            raise RuntimeError("GGUF resident session is closed")
        if self._lm_out_index is None or self._lm_out_value is None:
            raise RuntimeError("GGUF resident lm-head buffers are closed")
        runtime = self.runtime or get_hip_runtime()
        index_host = np.empty((1,), dtype=np.int64)
        copy_device_to_host(host_array_ptr(index_host), self._lm_out_index, runtime=runtime)
        logits = np.empty((0,), dtype=np.float32)
        logit = 0.0
        if return_logits:
            value_host = np.empty((1,), dtype=np.float32)
            copy_device_to_host(host_array_ptr(value_host), self._lm_out_value, runtime=runtime)
            logit = float(value_host[0])
            copy_device_to_host(host_array_ptr(self._logits_host), self._logits_buf, runtime=runtime)
            if not np.all(np.isfinite(self._logits_host)):
                raise FloatingPointError("GGUF resident lm-head logits contain NaN or Inf")
            logits = self._logits_host.copy()
        token_id = int(index_host[0])
        return Qwen35GGUFNextTokenProbeResult(
            token_id=token_id,
            logit=logit,
            logits=logits,
        )

    def close(self) -> None:
        runtime = self.runtime or get_hip_runtime()
        if self._moe_graph is not None:
            self._moe_graph.close()
            self._moe_graph = None
        for buffer in (
            self._verify_lm_out_values,
            self._verify_lm_out_indices_i32,
            self._verify_lm_block_indices_i32,
            self._verify_lm_block_values,
            self._verify_lm_q8_1,
            self._verify_logits_buf,
            self._verify_token_counter_i64,
            self._verify_token_ids_i64,
            self._verify_hidden_seed_buf,
        ):
            if buffer is not None:
                free(buffer, runtime=runtime)
        self._verify_hidden_seed_buf = None
        self._verify_token_ids_i64 = None
        self._verify_token_counter_i64 = None
        self._verify_block_rows_capacity = 0
        self._verify_hidden_seed_rows_populated = 0
        self._verify_lm_out_values = None
        self._verify_lm_out_indices_i32 = None
        self._verify_lm_block_indices_i32 = None
        self._verify_lm_block_values = None
        self._verify_lm_q8_1 = None
        self._verify_logits_buf = None
        self._verify_lm_rows_capacity = 0
        self._free_verify_linear_state_row_buffers(runtime=runtime)
        for buffer in reversed(self._buffers):
            if buffer is not None:
                free(buffer, runtime=runtime)
        self._buffers = ()
        for buffer in reversed(self._linear_state_snapshot_backups):
            if buffer is not None:
                free(buffer, runtime=runtime)
        self._linear_state_snapshot_backups = ()
        if self.scratch is not None:
            for buffer in reversed(self.scratch.buffers):
                free(buffer, runtime=runtime)
            self.scratch = None
        if self.runner is not None:
            self.runner.close()
            self.runner = None
        self._token_buf = None
        self._hidden_a = None
        self._hidden_b = None
        self._logits_buf = None
        self._lm_block_values = None
        self._lm_block_indices = None
        self._lm_out_index = None
        self._lm_out_value = None
        self._prefill_token_buf = None
        self._prefill_hidden_a = None
        self._prefill_hidden_b = None
        self._bulk_prefill_scratch = None
        self._logits_host = None
        self._expert_sidecar_host_layers = None
        self._expert_sidecar_reader = None
        self._expert_sidecar_model_map = None

    def __enter__(self) -> "Qwen35GGUFResidentSession":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


@dataclass(frozen=True)
class _GGUFFullAttentionPrefillScratch:
    rows: int
    norm: object
    full_q: object
    full_k: object
    full_v: object
    linear_qkv: object
    linear_qkv_f32: object
    linear_z: object
    linear_alpha: object
    linear_beta: object
    conv_out: object
    prefill_query: object
    prefill_key: object
    prefill_value: object
    prefill_beta: object
    prefill_decay: object
    recurrent_out: object
    recurrent_bf16: object
    gdn_cu_seqlens: object
    gdn_state_indices: object
    full_query_raw: object
    full_key_raw: object
    full_query: object
    full_key: object
    full_query_bf16: object
    full_gate: object
    full_attn_bf16: object
    full_gated: object
    attn_out: object
    post_norm: object
    residual: object
    ffn_gate_up: object
    ffn_intermediate: object
    ffn_down: object
    moe_q8_1: object
    moe_router_logits: object
    moe_shared_gate_logits: object
    moe_selected_experts: object
    moe_routing_weights: object
    moe_down_out: object
    moe_group_counts: object
    moe_padded_counts: object
    moe_scatter_offsets: object
    moe_expert_start_compact: object
    moe_expert_start_wmma: object
    moe_total_compact: object
    moe_wmma_total: object
    moe_tile_expert: object
    moe_sorted_lanes: object
    moe_sorted_experts: object
    moe_sorted_weights: object
    moe_lane_to_row: object
    moe_shared_gate: object
    moe_shared_up: object
    moe_shared_intermediate: object
    moe_shared_out: object
    key_cache: object | None
    value_cache: object | None
    block_table: object
    positions: object
    context_counts: object
    cos_table_buf: object
    sin_table_buf: object
    cu_q: object
    cu_k: object
    softmax_lse: object
    atomic: object
    block_table_tensor: Tensor
    positions_tensor: Tensor
    context_counts_tensor: Tensor
    append_spans: KVLiveSpans
    prefill_spans: KVLiveSpans
    cos_table: Tensor
    sin_table: Tensor
    block_size: int
    blocks: int
    max_positions: int
    moe_group_counts_zero: np.ndarray
    moe_scatter_offsets_zero: np.ndarray
    moe_wmma_total_host: np.ndarray
    moe_selected_host: np.ndarray
    moe_selected_rows_capacity: int
    moe_wmma_rows_capacity: int
    buffers: tuple[object, ...]
    start: int = 0

    @classmethod
    def allocate(
        cls,
        runner: Qwen35GGUFFullStackRunner,
        *,
        rows: int,
        capacity: int | None = None,
        allocate_kv_cache: bool = True,
        runtime: HipRuntime,
    ):
        if rows <= 0:
            raise ValueError("rows must be positive")
        capacity = int(rows) if capacity is None else int(capacity)
        if capacity < rows:
            raise ValueError(f"capacity {capacity} must be >= rows {rows}")
        assert runner.weights is not None
        cfg = runner.weights.config
        device = Device("hip", 0)
        block_size = 256
        blocks = (capacity + block_size - 1) // block_size
        max_positions = blocks * block_size

        def buf(nbytes: int):
            return malloc(nbytes, runtime=runtime)

        hidden_bytes = rows * runner.hidden_size * 2
        q_proj_bytes = rows * 2 * runner.q_width * 2
        kv_bf16_bytes = rows * runner.kv_width * 2
        q_f32_bytes = rows * runner.q_width * 4
        kv_f32_bytes = rows * runner.kv_width * 4
        ffn_bytes = rows * runner.ffn_size * 2
        moe_lane_count = max(1, int(cfg.expert_used_count)) if cfg.is_moe else 1
        moe_top_k = max(1, int(cfg.expert_used_count))
        moe_experts = max(1, int(cfg.expert_count))
        moe_selected_rows_capacity = rows * moe_top_k
        moe_wmma_rows_capacity = moe_selected_rows_capacity + 16 * moe_experts
        moe_tile_capacity = max(1, (moe_wmma_rows_capacity + 15) // 16)
        moe_group_counts_zero = np.zeros((moe_experts,), dtype=np.int32)
        moe_scatter_offsets_zero = np.zeros((moe_experts,), dtype=np.int32)
        moe_wmma_total_host = np.empty((1,), dtype=np.int64)
        moe_shared_ffn = max(1, int(cfg.expert_shared_feed_forward_length or runner.ffn_size or 1))
        q8_1_gate_blocks = rows * ((runner.hidden_size + _Q8_1_BLOCK - 1) // _Q8_1_BLOCK)
        q8_1_down_blocks = moe_selected_rows_capacity * ((runner.ffn_size + _Q8_1_BLOCK - 1) // _Q8_1_BLOCK)
        q8_1_moe_bytes = max(q8_1_gate_blocks, q8_1_down_blocks) * _Q8_1_BLOCK_BYTES
        linear_qkv_bf16_bytes = rows * runner.linear_qkv_width * 2
        linear_qkv_f32_bytes = rows * runner.linear_qkv_width * 4
        linear_z_bytes = rows * cfg.ssm_inner_size * 2
        linear_ab_bytes = rows * cfg.ssm_time_step_rank * 2
        recurrent_f32_bytes = rows * cfg.ssm_inner_size * 4
        prefill_scalar_bytes = rows * cfg.ssm_time_step_rank * 4
        cache_nbytes = max_positions * cfg.head_count_kv * cfg.key_length * 2 if allocate_kv_cache else 0
        block_table_arr = np.tile(np.arange(blocks, dtype=np.int32), (capacity, 1))
        positions_arr = np.arange(capacity, dtype=np.int64)
        context_arr = positions_arr + np.int64(1)
        cu_arr = np.asarray([0, rows], dtype=np.int32)
        atomic_arr = np.asarray([0], dtype=np.int32)
        cos_arr, sin_arr = _rope_tables(
            max_positions=max_positions,
            rotary_dim=cfg.rope_dimension_count,
            base=cfg.rope_freq_base,
        )
        fields = {
            "norm": buf(hidden_bytes),
            "full_q": buf(q_proj_bytes),
            "full_k": buf(kv_bf16_bytes),
            "full_v": buf(kv_bf16_bytes),
            "linear_qkv": buf(linear_qkv_bf16_bytes),
            "linear_qkv_f32": buf(linear_qkv_f32_bytes),
            "linear_z": buf(linear_z_bytes),
            "linear_alpha": buf(linear_ab_bytes),
            "linear_beta": buf(linear_ab_bytes),
            "conv_out": buf(linear_qkv_f32_bytes),
            "prefill_query": buf(recurrent_f32_bytes),
            "prefill_key": buf(recurrent_f32_bytes),
            "prefill_value": buf(recurrent_f32_bytes),
            "prefill_beta": buf(prefill_scalar_bytes),
            "prefill_decay": buf(prefill_scalar_bytes),
            "recurrent_out": buf(recurrent_f32_bytes),
            "recurrent_bf16": buf(linear_z_bytes),
            "gdn_cu_seqlens": buf(2 * DType.INT32.itemsize),
            "gdn_state_indices": buf(DType.INT64.itemsize),
            "full_query_raw": buf(q_f32_bytes),
            "full_key_raw": buf(kv_f32_bytes),
            "full_query": buf(q_f32_bytes),
            "full_key": buf(kv_f32_bytes),
            "full_query_bf16": buf(rows * runner.q_width * 2),
            "full_gate": buf(rows * runner.q_width * 2),
            "full_attn_bf16": buf(rows * runner.q_width * 2),
            "full_gated": buf(rows * runner.q_width * 2),
            "attn_out": buf(hidden_bytes),
            "post_norm": buf(hidden_bytes),
            "residual": buf(hidden_bytes),
            "ffn_gate_up": buf(2 * ffn_bytes * moe_lane_count),
            "ffn_intermediate": buf(ffn_bytes * moe_lane_count),
            "ffn_down": buf(hidden_bytes),
            "moe_q8_1": buf(q8_1_moe_bytes),
            "moe_router_logits": buf(rows * moe_experts * DType.FP32.itemsize),
            "moe_shared_gate_logits": buf(rows * DType.FP32.itemsize),
            "moe_selected_experts": buf(rows * moe_top_k * DType.INT64.itemsize),
            "moe_routing_weights": buf(rows * moe_top_k * DType.FP32.itemsize),
            "moe_down_out": buf(moe_top_k * hidden_bytes),
            "moe_group_counts": buf(moe_group_counts_zero.nbytes),
            "moe_padded_counts": buf(moe_group_counts_zero.nbytes),
            "moe_scatter_offsets": buf(moe_scatter_offsets_zero.nbytes),
            "moe_expert_start_compact": buf((moe_experts + 1) * DType.INT64.itemsize),
            "moe_expert_start_wmma": buf((moe_experts + 1) * DType.INT64.itemsize),
            "moe_total_compact": buf(DType.INT64.itemsize),
            "moe_wmma_total": buf(DType.INT64.itemsize),
            "moe_tile_expert": buf(moe_tile_capacity * DType.INT64.itemsize),
            "moe_sorted_lanes": buf(moe_selected_rows_capacity * DType.INT64.itemsize),
            "moe_sorted_experts": buf(moe_selected_rows_capacity * DType.INT64.itemsize),
            "moe_sorted_weights": buf(moe_selected_rows_capacity * DType.FP32.itemsize),
            "moe_lane_to_row": buf(moe_selected_rows_capacity * DType.INT64.itemsize),
            "moe_shared_gate": buf(rows * moe_shared_ffn * DType.BF16.itemsize),
            "moe_shared_up": buf(rows * moe_shared_ffn * DType.BF16.itemsize),
            "moe_shared_intermediate": buf(rows * moe_shared_ffn * DType.BF16.itemsize),
            "moe_shared_out": buf(hidden_bytes),
            "key_cache": buf(cache_nbytes) if allocate_kv_cache else None,
            "value_cache": buf(cache_nbytes) if allocate_kv_cache else None,
            "block_table": buf(block_table_arr.nbytes),
            "positions": buf(positions_arr.nbytes),
            "context_counts": buf(context_arr.nbytes),
            "cos_table_buf": buf(cos_arr.nbytes),
            "sin_table_buf": buf(sin_arr.nbytes),
            "cu_q": buf(cu_arr.nbytes),
            "cu_k": buf(cu_arr.nbytes),
            "softmax_lse": buf(cfg.head_count * rows * 4),
            "atomic": buf(atomic_arr.nbytes),
        }
        copy_host_to_device(fields["block_table"], host_array_ptr(block_table_arr), runtime=runtime)
        copy_host_to_device(fields["positions"], host_array_ptr(positions_arr), runtime=runtime)
        copy_host_to_device(fields["context_counts"], host_array_ptr(context_arr), runtime=runtime)
        copy_host_to_device(fields["cos_table_buf"], host_array_ptr(cos_arr), runtime=runtime)
        copy_host_to_device(fields["sin_table_buf"], host_array_ptr(sin_arr), runtime=runtime)
        copy_host_to_device(fields["cu_q"], host_array_ptr(cu_arr), runtime=runtime)
        copy_host_to_device(fields["cu_k"], host_array_ptr(cu_arr), runtime=runtime)
        copy_host_to_device(fields["atomic"], host_array_ptr(atomic_arr), runtime=runtime)
        gdn_state_indices_arr = np.zeros((1,), dtype=np.int64)
        copy_host_to_device(
            fields["gdn_cu_seqlens"], host_array_ptr(cu_arr), cu_arr.nbytes, runtime=runtime
        )
        copy_host_to_device(
            fields["gdn_state_indices"],
            host_array_ptr(gdn_state_indices_arr),
            gdn_state_indices_arr.nbytes,
            runtime=runtime,
        )
        block_table_tensor = Tensor.from_handle(fields["block_table"].ptr, block_table_arr.shape, DType.INT32, device)
        positions_tensor = Tensor.from_handle(fields["positions"].ptr, positions_arr.shape, DType.INT64, device)
        context_tensor = Tensor.from_handle(fields["context_counts"].ptr, context_arr.shape, DType.INT64, device)
        append_spans = KVLiveSpans.paged_uniform(
            block_table=block_table_tensor,
            live_counts=positions_tensor,
            max_live_count=rows - 1,
            storage_dtype=DType.BF16,
            row_positions=positions_tensor,
            span_role="prefill",
        )
        prefill_spans = KVLiveSpans.paged_uniform(
            block_table=block_table_tensor,
            live_counts=context_tensor,
            max_live_count=rows,
            storage_dtype=DType.BF16,
            row_positions=positions_tensor,
            span_role="prefill",
        )
        cos_table = Tensor.from_handle(fields["cos_table_buf"].ptr, cos_arr.shape, DType.FP32, device)
        sin_table = Tensor.from_handle(fields["sin_table_buf"].ptr, sin_arr.shape, DType.FP32, device)
        return cls(
            **fields,
            rows=rows,
            block_table_tensor=block_table_tensor,
            positions_tensor=positions_tensor,
            context_counts_tensor=context_tensor,
            append_spans=append_spans,
            prefill_spans=prefill_spans,
            cos_table=cos_table,
            sin_table=sin_table,
            block_size=block_size,
            blocks=blocks,
            max_positions=max_positions,
            moe_group_counts_zero=moe_group_counts_zero,
            moe_scatter_offsets_zero=moe_scatter_offsets_zero,
            moe_wmma_total_host=moe_wmma_total_host,
            moe_selected_host=np.empty((moe_top_k,), dtype=np.int64),
            moe_selected_rows_capacity=moe_selected_rows_capacity,
            moe_wmma_rows_capacity=moe_wmma_rows_capacity,
            buffers=tuple(value for value in fields.values() if value is not None),
        )

    def for_chunk(self, start: int, rows: int, total_tokens: int, *, runtime: HipRuntime, stream: int = 0):
        start = int(start)
        rows = int(rows)
        total_tokens = int(total_tokens)
        if start < 0 or rows <= 0 or start + rows > self.max_positions:
            raise ValueError(f"chunk bounds [{start}, {start+rows}) must be within [0, {self.max_positions})")
        if total_tokens <= 0 or total_tokens > self.max_positions or start + rows > total_tokens:
            raise ValueError(
                f"chunk bounds [{start}, {start+rows}) must be within total_tokens={total_tokens} and max_positions={self.max_positions}"
            )
        cu_q_arr = np.asarray([0, rows], dtype=np.int32)
        cu_k_arr = np.asarray([0, start + rows], dtype=np.int32)
        atomic_arr = np.asarray([0], dtype=np.int32)
        copy_host_to_device(self.cu_q, host_array_ptr(cu_q_arr), cu_q_arr.nbytes, runtime=runtime)
        copy_host_to_device(self.cu_k, host_array_ptr(cu_k_arr), cu_k_arr.nbytes, runtime=runtime)
        copy_host_to_device(self.atomic, host_array_ptr(atomic_arr), atomic_arr.nbytes, runtime=runtime)
        copy_host_to_device(
            self.gdn_cu_seqlens, host_array_ptr(cu_q_arr), cu_q_arr.nbytes, runtime=runtime
        )
        _ = stream
        block_table = Tensor.from_handle(
            self.block_table.ptr + start * self.blocks * DType.INT32.itemsize,
            (rows, self.blocks),
            DType.INT32,
            self.block_table_tensor.device,
        )
        positions = Tensor.from_handle(
            self.positions.ptr + start * DType.INT64.itemsize,
            (rows,),
            DType.INT64,
            self.positions_tensor.device,
        )
        context_counts = Tensor.from_handle(
            self.context_counts.ptr + start * DType.INT64.itemsize,
            (rows,),
            DType.INT64,
            self.context_counts_tensor.device,
        )
        append_spans = KVLiveSpans.paged_uniform(
            block_table=block_table,
            live_counts=positions,
            max_live_count=total_tokens - 1,
            storage_dtype=DType.BF16,
            row_positions=positions,
            span_role="prefill",
        )
        prefill_spans = KVLiveSpans.paged_uniform(
            block_table=block_table,
            live_counts=context_counts,
            max_live_count=total_tokens,
            storage_dtype=DType.BF16,
            row_positions=positions,
            span_role="prefill",
        )
        return replace(
            self,
            start=start,
            rows=rows,
            block_table_tensor=block_table,
            positions_tensor=positions,
            context_counts_tensor=context_counts,
            append_spans=append_spans,
            prefill_spans=prefill_spans,
        )

    def for_rows(self, rows: int, *, runtime: HipRuntime, stream: int = 0):
        return self.for_chunk(start=0, rows=rows, total_tokens=rows, runtime=runtime, stream=stream)


@dataclass(frozen=True)
class _FullStackScratch:
    norm: object
    hidden_seed_fp32: object
    post_norm: object
    residual: object
    attn_out: object
    linear_qkv: object
    linear_z: object
    linear_alpha: object
    linear_beta: object
    linear_alpha_beta: object
    conv_out: object
    recurrent_out: object
    recurrent_bf16: object
    layer_conv_states: tuple[object | None, ...]
    layer_recurrent_states: tuple[object | None, ...]
    conv_zero: np.ndarray
    recurrent_zero: np.ndarray
    full_q: object
    full_k: object
    full_v: object
    full_query_raw: object
    full_key_raw: object
    full_query: object
    full_key: object
    full_gate: object
    full_attn_context: object
    full_attn_split_partial: object
    full_attn_split_m: object
    full_attn_split_l: object
    full_attn_split_count: int
    full_gated: object
    full_key_caches: tuple[object | None, ...]
    full_value_caches: tuple[object | None, ...]
    block_table: object
    position_buf: object
    context_buf: object
    cos_table_buf: object
    sin_table_buf: object
    block_table_tensor: Tensor
    position_tensor: Tensor
    context_tensor: Tensor
    append_spans: KVLiveSpans
    decode_spans: KVLiveSpans
    cos_table: Tensor
    sin_table: Tensor
    block_size: int
    max_positions: int
    position_host: np.ndarray
    context_host: np.ndarray
    ffn_gate_up: object
    ffn_intermediate: object
    ffn_down: object
    moe_q8_1: object
    moe_router_logits: object
    moe_selected_experts: object
    moe_routing_weights: object
    moe_down_out: object
    moe_group_counts: object
    moe_padded_counts: object
    moe_scatter_offsets: object
    moe_expert_start_compact: object
    moe_total_compact: object
    moe_sorted_lanes: object
    moe_sorted_experts: object
    moe_sorted_weights: object
    moe_lane_to_row: object
    moe_shared_gate: object
    moe_shared_up: object
    moe_shared_intermediate: object
    moe_shared_out: object
    moe_shared_gate_logits: object
    moe_selected_host: np.ndarray
    moe_group_counts_zero: np.ndarray
    moe_scatter_offsets_zero: np.ndarray
    moe_selected_rows_capacity: int
    buffers: tuple[object, ...]

    @classmethod
    def allocate(
        cls,
        runner: Qwen35GGUFFullStackRunner,
        *,
        runtime: HipRuntime,
        max_sequence_length: int | None = None,
    ):
        def buf(nbytes: int):
            return malloc(nbytes, runtime=runtime)

        assert runner.weights is not None
        cfg = runner.weights.config
        device = Device("hip", 0)
        block_size = 256
        requested_positions = block_size if max_sequence_length is None else int(max_sequence_length)
        if requested_positions <= 0:
            raise ValueError("max_sequence_length must be positive")
        if requested_positions > int(cfg.context_length):
            raise ValueError(
                f"max_sequence_length {requested_positions} exceeds GGUF context length {cfg.context_length}"
            )
        block_count = (requested_positions + block_size - 1) // block_size
        max_positions = min(int(cfg.context_length), block_count * block_size)
        hidden_bytes = runner.hidden_size * 2
        hidden_fp32_bytes = runner.hidden_size * DType.FP32.itemsize
        ffn_bytes = runner.ffn_size * 2
        moe_lane_count = max(1, int(cfg.expert_used_count)) if cfg.is_moe else 1
        moe_top_k = max(1, int(cfg.expert_used_count))
        moe_experts = max(1, int(cfg.expert_count))
        moe_shared_ffn = max(1, int(cfg.expert_shared_feed_forward_length or runner.ffn_size or 1))
        q8_1_gate_blocks = (runner.hidden_size + _Q8_1_BLOCK - 1) // _Q8_1_BLOCK
        q8_1_down_blocks = moe_top_k * ((runner.ffn_size + _Q8_1_BLOCK - 1) // _Q8_1_BLOCK)
        q8_1_moe_bytes = max(q8_1_gate_blocks, q8_1_down_blocks) * _Q8_1_BLOCK_BYTES
        linear_qkv_bytes = runner.linear_qkv_width * 2
        ssm_inner_bytes = cfg.ssm_inner_size * 2
        alpha_bytes = cfg.ssm_time_step_rank * 2
        q_proj_bytes = 2 * runner.q_width * 2
        kv_bf16_bytes = runner.kv_width * 2
        q_f32_bytes = runner.q_width * 4
        kv_f32_bytes = runner.kv_width * 4
        full_attn_split_count = (max_positions + block_size - 1) // block_size
        full_attn_split_partial_bytes = runner.q_width * full_attn_split_count * 4
        full_attn_split_stat_bytes = cfg.head_count * full_attn_split_count * 4
        conv_zero = np.zeros((runner.linear_qkv_width, cfg.ssm_conv_kernel), dtype=np.float32)
        recurrent_zero = np.zeros((cfg.ssm_time_step_rank, cfg.ssm_state_size, runner.ssm_value_dim), dtype=np.float32)
        layer_conv_states: list[object | None] = []
        layer_recurrent_states: list[object | None] = []
        full_key_caches: list[object | None] = []
        full_value_caches: list[object | None] = []
        state_buffers: list[object] = []
        cache_buffers: list[object] = []
        cache_nbytes = max_positions * cfg.head_count_kv * cfg.key_length * 2
        for layer_type in cfg.layer_types:
            if layer_type == LINEAR_ATTENTION:
                conv_state = buf(conv_zero.nbytes)
                recurrent_state = buf(recurrent_zero.nbytes)
                state_buffers.extend((conv_state, recurrent_state))
                layer_conv_states.append(conv_state)
                layer_recurrent_states.append(recurrent_state)
                full_key_caches.append(None)
                full_value_caches.append(None)
            else:
                key_cache = buf(cache_nbytes)
                value_cache = buf(cache_nbytes)
                cache_buffers.extend((key_cache, value_cache))
                layer_conv_states.append(None)
                layer_recurrent_states.append(None)
                full_key_caches.append(key_cache)
                full_value_caches.append(value_cache)
        block_table_arr = np.arange(block_count, dtype=np.int32)
        position_host = np.asarray([0], dtype=np.int64)
        context_host = np.asarray([1], dtype=np.int64)
        cos_arr, sin_arr = _rope_tables(
            max_positions=max_positions,
            rotary_dim=cfg.rope_dimension_count,
            base=cfg.rope_freq_base,
        )
        block_table = buf(block_table_arr.nbytes)
        position_buf = buf(position_host.nbytes)
        context_buf = buf(context_host.nbytes)
        cos_table_buf = buf(cos_arr.nbytes)
        sin_table_buf = buf(sin_arr.nbytes)
        copy_host_to_device(block_table, host_array_ptr(block_table_arr), runtime=runtime)
        copy_host_to_device(position_buf, host_array_ptr(position_host), runtime=runtime)
        copy_host_to_device(context_buf, host_array_ptr(context_host), runtime=runtime)
        copy_host_to_device(cos_table_buf, host_array_ptr(cos_arr), runtime=runtime)
        copy_host_to_device(sin_table_buf, host_array_ptr(sin_arr), runtime=runtime)
        block_table_tensor = Tensor.from_handle(block_table.ptr, block_table_arr.shape, DType.INT32, device)
        position_tensor = Tensor.from_handle(position_buf.ptr, position_host.shape, DType.INT64, device)
        context_tensor = Tensor.from_handle(context_buf.ptr, context_host.shape, DType.INT64, device)
        append_spans = KVLiveSpans.paged_uniform(
            block_table=block_table_tensor,
            live_counts=position_tensor,
            max_live_count=max_positions - 1,
            storage_dtype=DType.BF16,
        )
        decode_spans = KVLiveSpans.paged_uniform(
            block_table=block_table_tensor,
            live_counts=context_tensor,
            max_live_count=max_positions,
            storage_dtype=DType.BF16,
        )
        cos_table = Tensor.from_handle(cos_table_buf.ptr, cos_arr.shape, DType.FP32, device)
        sin_table = Tensor.from_handle(sin_table_buf.ptr, sin_arr.shape, DType.FP32, device)
        fields = {
            "norm": buf(hidden_bytes),
            "hidden_seed_fp32": buf(hidden_fp32_bytes),
            "post_norm": buf(hidden_bytes),
            "residual": buf(hidden_bytes),
            "attn_out": buf(hidden_bytes),
            "linear_qkv": buf(linear_qkv_bytes),
            "linear_z": buf(ssm_inner_bytes),
            "linear_alpha": buf(alpha_bytes),
            "linear_beta": buf(alpha_bytes),
            "linear_alpha_beta": buf(2 * alpha_bytes),
            "conv_out": buf(runner.linear_qkv_width * 4),
            "recurrent_out": buf(cfg.ssm_inner_size * 4),
            "recurrent_bf16": buf(ssm_inner_bytes),
            "full_q": buf(q_proj_bytes),
            "full_k": buf(kv_bf16_bytes),
            "full_v": buf(kv_bf16_bytes),
            "full_query_raw": buf(q_f32_bytes),
            "full_key_raw": buf(kv_f32_bytes),
            "full_query": buf(q_f32_bytes),
            "full_key": buf(kv_f32_bytes),
            "full_gate": buf(runner.q_width * 2),
            "full_attn_context": buf(q_f32_bytes),
            "full_attn_split_partial": buf(full_attn_split_partial_bytes),
            "full_attn_split_m": buf(full_attn_split_stat_bytes),
            "full_attn_split_l": buf(full_attn_split_stat_bytes),
            "full_gated": buf(runner.q_width * 2),
            "ffn_gate_up": buf(2 * ffn_bytes * moe_lane_count),
            "ffn_intermediate": buf(ffn_bytes * moe_lane_count),
            "ffn_down": buf(hidden_bytes),
            "moe_q8_1": buf(q8_1_moe_bytes),
            "moe_router_logits": buf((moe_experts + 1) * DType.FP32.itemsize),
            "moe_selected_experts": buf(moe_top_k * DType.INT64.itemsize),
            "moe_routing_weights": buf(moe_top_k * DType.FP32.itemsize),
            "moe_down_out": buf(moe_top_k * hidden_bytes),
            "moe_group_counts": buf(moe_experts * DType.INT32.itemsize),
            "moe_padded_counts": buf(moe_experts * DType.INT32.itemsize),
            "moe_scatter_offsets": buf(moe_experts * DType.INT32.itemsize),
            "moe_expert_start_compact": buf((moe_experts + 1) * DType.INT64.itemsize),
            "moe_total_compact": buf(DType.INT64.itemsize),
            "moe_sorted_lanes": buf(moe_top_k * DType.INT64.itemsize),
            "moe_sorted_experts": buf(moe_top_k * DType.INT64.itemsize),
            "moe_sorted_weights": buf(moe_top_k * DType.FP32.itemsize),
            "moe_lane_to_row": buf(moe_top_k * DType.INT64.itemsize),
            "moe_shared_gate": buf(moe_shared_ffn * DType.BF16.itemsize),
            "moe_shared_up": buf(moe_shared_ffn * DType.BF16.itemsize),
            "moe_shared_intermediate": buf(moe_shared_ffn * DType.BF16.itemsize),
            "moe_shared_out": buf(hidden_bytes),
            "moe_shared_gate_logits": buf(DType.FP32.itemsize),
        }
        moe_group_counts_zero = np.zeros((moe_experts,), dtype=np.int32)
        moe_scatter_offsets_zero = np.zeros((moe_experts,), dtype=np.int32)
        metadata_buffers = (block_table, position_buf, context_buf, cos_table_buf, sin_table_buf)
        return cls(
            **fields,
            full_attn_split_count=full_attn_split_count,
            full_key_caches=tuple(full_key_caches),
            full_value_caches=tuple(full_value_caches),
            block_table=block_table,
            position_buf=position_buf,
            context_buf=context_buf,
            cos_table_buf=cos_table_buf,
            sin_table_buf=sin_table_buf,
            block_table_tensor=block_table_tensor,
            position_tensor=position_tensor,
            context_tensor=context_tensor,
            append_spans=append_spans,
            decode_spans=decode_spans,
            cos_table=cos_table,
            sin_table=sin_table,
            block_size=block_size,
            max_positions=max_positions,
            position_host=position_host,
            context_host=context_host,
            layer_conv_states=tuple(layer_conv_states),
            layer_recurrent_states=tuple(layer_recurrent_states),
            conv_zero=conv_zero,
            recurrent_zero=recurrent_zero,
            moe_selected_host=np.empty((moe_top_k,), dtype=np.int64),
            moe_group_counts_zero=moe_group_counts_zero,
            moe_scatter_offsets_zero=moe_scatter_offsets_zero,
            moe_selected_rows_capacity=moe_top_k,
            buffers=tuple(fields.values()) + tuple(state_buffers) + tuple(cache_buffers) + metadata_buffers,
        )

    def full_cache(self, layer_id: int) -> tuple[object, object]:
        key_cache = self.full_key_caches[layer_id]
        value_cache = self.full_value_caches[layer_id]
        if key_cache is None or value_cache is None:
            raise ValueError(f"layer {layer_id} has no full-attention KV cache")
        return key_cache, value_cache

    def set_full_attention_position(self, position: int, runtime: HipRuntime) -> None:
        if position < 0 or position >= self.max_positions:
            raise ValueError(f"GGUF resident full-attention position {position} exceeds cache capacity {self.max_positions}")
        self.position_host[0] = int(position)
        self.context_host[0] = int(position) + 1
        copy_host_to_device(self.position_buf, host_array_ptr(self.position_host), runtime=runtime)
        copy_host_to_device(self.context_buf, host_array_ptr(self.context_host), runtime=runtime)

    def zero_states(self, runtime: HipRuntime) -> None:
        for conv_state, recurrent_state in zip(self.layer_conv_states, self.layer_recurrent_states, strict=True):
            if conv_state is not None:
                _zero(runtime, conv_state, self.conv_zero)
            if recurrent_state is not None:
                _zero(runtime, recurrent_state, self.recurrent_zero)
        self.set_full_attention_position(0, runtime)


def _expert_raw_ptr(weight: Qwen35GGUFDeviceWeight, expert_id: int) -> int:
    """Return the raw GGUF row pointer for one rank-3 MoE expert tensor."""

    source = weight.spec.source
    if len(source.shape) != 3 or len(source.byte_shape) != 3:
        raise ValueError(f"GGUF expert tensor {source.name!r} must be rank-3, got {source.shape}")
    experts, rows, row_bytes = source.byte_shape
    if expert_id < 0 or expert_id >= experts:
        raise ValueError(f"expert_id {expert_id} outside [0, {experts}) for {source.name}")
    return weight.allocation("raw").tensor.ptr + int(expert_id) * int(rows) * int(row_bytes)


_EXPERT_PACK8_SELECTED_KEYS = {
    "gguf_q4_k": KernelKey("hip_gfx1100", "moe_linear", "gguf_q4_k", "expert_pack8_selected_bf16_bf16_out"),
    "gguf_q5_k": KernelKey("hip_gfx1100", "moe_linear", "gguf_q5_k", "expert_pack8_selected_bf16_bf16_out"),
    "gguf_q6_k": KernelKey("hip_gfx1100", "moe_linear", "gguf_q6_k", "expert_pack8_selected_bf16_bf16_out"),
}
_EXPERT_PACK8_DUAL_KEYS = {
    ("gguf_q4_k", "gguf_q4_k"): KernelKey(
        "hip_gfx1100",
        "moe_linear",
        "gguf_q4_k",
        "expert_pack8_dual_selected_bf16_bf16_out",
    ),
}
_COMPACT_MOE_SCHEDULER_KEYS = (
    KernelKey("hip_gfx1100", "moe_group_count", "w4_paro", "qwen35"),
    KernelKey("hip_gfx1100", "moe_group_prefix", "w4_paro", "qwen35"),
    KernelKey("hip_gfx1100", "moe_group_scatter_gather", "w4_paro", "qwen35_lowp"),
    KernelKey("hip_gfx1100", "moe_wmma_tile_map", "w4_paro", "qwen35"),
)
_COMPACT_MOE_FUSED_KEYS = (
    KernelKey("hip_gfx1100", "weighted_lanes_sum", "w4_paro", "out"),
    KernelKey("hip_gfx1100", "shared_gate_combine+residual", "w4_paro", "batch_out"),
)
_COMPACT_MOE_Q4_DUAL_KEYS = {
    ("gguf_q4_k", "gguf_q4_k"): KernelKey(
        "hip_gfx1100",
        "moe_linear",
        "gguf_q4_k",
        "selected_dual_wmma_prefill_compact_bf16_bf16_out",
    ),
    # P10.B1: T16 decode-repack mode reuses the same compact selected dual
    # WMMA prefill ABI, with tile bytes consumed in place of raw GGUF bytes.
    # The kernel below is registered by
    # ``register_gguf_q4_k_t16_selected_prefill_kernels`` under the same
    # ``selected_dual_wmma_prefill_compact_*`` alias spelling so dispatch can
    # route on ``quant_key`` alone (no backend / quant branch).
    ("gguf_q4_k_t16_v1", "gguf_q4_k_t16_v1"): KernelKey(
        "hip_gfx1100",
        "moe_linear",
        "gguf_q4_k_t16_v1",
        "selected_dual_wmma_prefill_compact_bf16_bf16_out",
    ),
}
_COMPACT_MOE_DOWN_KEYS = {
    # Q4_K_S stores selected down experts as Q4_K.  In decode-repack mode those
    # tensors use the same single-output compact WMMA ABI as Q5/Q6 T16.
    "gguf_q4_k_t16_v1": KernelKey(
        "hip_gfx1100",
        "moe_linear",
        "gguf_q4_k_t16_v1",
        "selected_wmma_prefill_compact_bf16_bf16_out",
    ),
    "gguf_q5_k": KernelKey(
        "hip_gfx1100",
        "moe_linear",
        "gguf_q5_k",
        "selected_wmma_prefill_compact_bf16_bf16_out",
    ),
    "gguf_q6_k": KernelKey(
        "hip_gfx1100",
        "moe_linear",
        "gguf_q6_k",
        "selected_wmma_prefill_compact_bf16_bf16_out",
    ),
    # P10.B2: Q5T16 selected single-output WMMA prefill.
    "gguf_q5_k_t16_v1": KernelKey(
        "hip_gfx1100",
        "moe_linear",
        "gguf_q5_k_t16_v1",
        "selected_wmma_prefill_compact_bf16_bf16_out",
    ),
    # P10.B3: Q6T16 selected single-output WMMA prefill.
    "gguf_q6_k_t16_v1": KernelKey(
        "hip_gfx1100",
        "moe_linear",
        "gguf_q6_k_t16_v1",
        "selected_wmma_prefill_compact_bf16_bf16_out",
    ),
}
_COMPACT_MOE_Q4_DUAL_GEMV_KEYS = {
    ("gguf_q4_k", "gguf_q4_k"): KernelKey(
        "hip_gfx1100",
        "moe_linear",
        "gguf_q4_k",
        "selected_dual_pack8_gemv_decode_compact_bf16_bf16_out",
    ),
    ("gguf_q4_k_t16_v1", "gguf_q4_k_t16_v1"): KernelKey(
        "hip_gfx1100",
        "moe_linear",
        "gguf_q4_k_t16_v1",
        "selected_dual_t16_gemv_decode_compact_bf16_bf16_out",
    ),
}
_COMPACT_MOE_DOWN_GEMV_KEYS = {
    "gguf_q5_k": KernelKey(
        "hip_gfx1100",
        "moe_linear",
        "gguf_q5_k",
        "selected_pack8_gemv_decode_compact_bf16_bf16_out",
    ),
    "gguf_q6_k": KernelKey(
        "hip_gfx1100",
        "moe_linear",
        "gguf_q6_k",
        "selected_pack8_gemv_decode_compact_bf16_bf16_out",
    ),
    "gguf_q5_k_t16_v1": KernelKey(
        "hip_gfx1100",
        "moe_linear",
        "gguf_q5_k_t16_v1",
        "selected_t16_gemv_decode_compact_bf16_bf16_out",
    ),
    "gguf_q6_k_t16_v1": KernelKey(
        "hip_gfx1100",
        "moe_linear",
        "gguf_q6_k_t16_v1",
        "selected_t16_gemv_decode_compact_bf16_bf16_out",
    ),
    "gguf_q5_k_x8_v1": KernelKey(
        "hip_gfx1100",
        "moe_linear",
        "gguf_q5_k_x8_v1",
        "selected_x8_q8_1_dp4a_gemv_decode_compact_bf16_bf16_out",
    ),
    "gguf_q6_k_x8_v1": KernelKey(
        "hip_gfx1100",
        "moe_linear",
        "gguf_q6_k_x8_v1",
        "selected_x8_q8_1_dp4a_gemv_decode_compact_bf16_bf16_out",
    ),
}
_COMPACT_MOE_GEMV_DECODE_SCRATCH = (
    "moe_group_counts",
    "moe_padded_counts",
    "moe_scatter_offsets",
    "moe_expert_start_compact",
    "moe_total_compact",
    "moe_sorted_lanes",
    "moe_sorted_experts",
    "moe_sorted_weights",
    "moe_lane_to_row",
    "moe_group_counts_zero",
    "moe_scatter_offsets_zero",
)
_COMPACT_MOE_REQUIRED_SCRATCH = (
    "moe_group_counts",
    "moe_padded_counts",
    "moe_scatter_offsets",
    "moe_expert_start_compact",
    "moe_expert_start_wmma",
    "moe_total_compact",
    "moe_wmma_total",
    "moe_tile_expert",
    "moe_sorted_lanes",
    "moe_sorted_experts",
    "moe_sorted_weights",
    "moe_lane_to_row",
    "moe_group_counts_zero",
    "moe_scatter_offsets_zero",
    "moe_wmma_total_host",
)

_GDN_PREFILL_PREPARE_KEY = KernelKey(
    "hip_gfx1100", "linear_attn_prefill_prepare", "gguf_qwen35", "f32_bf16"
)
_GDN_PREFILL_RECURRENT_K2_KEY = KernelKey(
    "hip_gfx1100", "gdn_prefill_recurrent", "gguf_qwen35", "f32_k2"
)
_GDN_PREFILL_RECURRENT_SEGMENTS_K2_KEY = KernelKey(
    "hip_gfx1100", "gdn_prefill_recurrent", "gguf_qwen35", "f32_k2_segments"
)
_GDN_PREFILL_RMSNORM_GATE_BF16_KEY = KernelKey(
    "hip_gfx1100", "gdn_prefill_rmsnorm_gate", "gguf_qwen35", "bf16"
)
_GDN_PREFILL_DECODE_ORDER_BF16_KEY = KernelKey(
    "hip_gfx1100", "gdn_prefill_recurrent", "gguf_qwen35", "decode_order_bf16"
)
_GDN_PREFILL_SEGMENT_THRESHOLD_DEFAULT = 256


@dataclass(frozen=True)
class _CompactMoeGemvPlan:
    gate_up_fn: object
    down_fn: object
    gate_allocation: str
    up_allocation: str
    down_allocation: str


@dataclass(frozen=True)
class _CompactMoeWmmaPlan:
    gate_up_fn: object
    down_fn: object
    gate_allocation: str
    up_allocation: str
    down_allocation: str


@dataclass(frozen=True)
class _GGUFGDNPrefillPlan:
    """Resolved kernel set for the qwen35 GGUF GDN prefill path.

    ``recurrent_segments`` is optional and only consulted when the runtime
    decides the prefill row count meets the multi-segment threshold; for the
    current single-sequence prefill it is always called with ``segments=1``,
    so the parent ``segments_k2`` kernel is only useful for batched prefill.
    The chain falls back to ``fused_decode_order`` when any of the chain
    members is not registered.
    """

    prepare: object | None
    recurrent: object | None
    recurrent_segments: object | None
    rmsnorm_gate: object | None
    fused_decode_order: object | None

    @property
    def has_chain(self) -> bool:
        return (
            self.prepare is not None
            and self.recurrent is not None
            and self.rmsnorm_gate is not None
        )

    @property
    def has_fused(self) -> bool:
        return self.fused_decode_order is not None


def _gguf_gdn_prefill_segment_threshold() -> int:
    raw = os.environ.get("HIPENGINE_GGUF_GDN_PREFILL_SEGMENT_THRESHOLD")
    if not raw:
        return _GDN_PREFILL_SEGMENT_THRESHOLD_DEFAULT
    try:
        value = int(raw)
    except ValueError:
        return _GDN_PREFILL_SEGMENT_THRESHOLD_DEFAULT
    return max(1, value)


def _gguf_full_attention_split_decode_min_context() -> int:
    return max(
        0,
        _env_int(
            _GGUF_FULL_ATTN_DECODE_SPLIT_MIN_CONTEXT_ENV,
            _GGUF_FULL_ATTN_DECODE_SPLIT_MIN_CONTEXT_DEFAULT,
            "NANOVLLM_GGUF_FULL_ATTN_DECODE_PAGED_MIN_CONTEXT",
        ),
    )


def _use_gguf_full_attention_split_decode(active_context: int) -> bool:
    threshold = _gguf_full_attention_split_decode_min_context()
    return threshold > 0 and int(active_context) >= threshold


def _gguf_paged_attn_gqa_grouped_min_splits() -> int:
    return max(1, _env_int("HIPENGINE_PAGED_ATTN_GQA_GROUPED_MIN_SPLITS", 64))


def _gguf_paged_attn_gqa_grouped_min_context() -> int:
    return max(0, _env_int("HIPENGINE_PAGED_ATTN_GQA_GROUPED_MIN_CONTEXT", 4096))


def _gguf_paged_attn_gqa_grouped_enabled() -> bool:
    return _env_flag(
        "HIPENGINE_PAGED_ATTN_GQA_GROUPED_CTX",
        True,
        "NANOVLLM_AMD_PAGED_ATTN_GQA_GROUPED_CTX",
    )


def _gguf_paged_attn_warp_split_enabled() -> bool:
    return _env_flag(
        "HIPENGINE_PAGED_ATTN_WARP_SPLIT_CTX",
        True,
        "NANOVLLM_AMD_PAGED_ATTN_WARP_SPLIT_CTX",
    )


def _gguf_qwen35_gqa_decode_shape(config, *, block_size: int) -> bool:
    return (
        int(block_size) == 256
        and int(config.head_count) == 16
        and int(config.head_count_kv) == 2
        and int(config.key_length) == 256
    )


def _use_gguf_paged_attn_gqa_grouped(active_context: int, num_splits: int) -> bool:
    if not _gguf_paged_attn_gqa_grouped_enabled():
        return False
    return int(num_splits) >= _gguf_paged_attn_gqa_grouped_min_splits() or int(
        active_context
    ) >= _gguf_paged_attn_gqa_grouped_min_context()


def _gguf_full_attention_split_gate_bf16_fn(
    config,
    *,
    block_size: int,
    num_splits: int,
    active_context: int,
):
    if _gguf_qwen35_gqa_decode_shape(config, block_size=block_size):
        if _use_gguf_paged_attn_gqa_grouped(active_context, num_splits):
            return qwen35_paged_full_attn_decode_split_k_gqa_gate_bf16_spans
        if _gguf_paged_attn_warp_split_enabled():
            return qwen35_paged_full_attn_decode_split_k_warp_gate_bf16_spans
    return qwen35_paged_full_attn_decode_split_k_gate_bf16_spans


def _resolve_gguf_gdn_prefill_plan() -> _GGUFGDNPrefillPlan:
    register_qwen35_linear_attn_gdn_kernels()

    def _resolve(key: KernelKey):
        return resolve(
            backend=key.backend,
            layer=key.layer,
            quant=key.quant,
            variant=key.variant,
            missing="none",
        )

    return _GGUFGDNPrefillPlan(
        prepare=_resolve(_GDN_PREFILL_PREPARE_KEY),
        recurrent=_resolve(_GDN_PREFILL_RECURRENT_K2_KEY),
        recurrent_segments=_resolve(_GDN_PREFILL_RECURRENT_SEGMENTS_K2_KEY),
        rmsnorm_gate=_resolve(_GDN_PREFILL_RMSNORM_GATE_BF16_KEY),
        fused_decode_order=_resolve(_GDN_PREFILL_DECODE_ORDER_BF16_KEY),
    )


def _copy_sidecar_array_to_device(array: np.ndarray, *, runtime: HipRuntime) -> DeviceBuffer:
    contiguous = np.ascontiguousarray(array)
    buffer = malloc(contiguous.nbytes, runtime=runtime)
    copy_host_to_device(buffer, host_array_ptr(contiguous), runtime=runtime)
    return buffer


def _launch_selected_expert_pack8_moe_pair(
    weight_a: _DeviceExpertPackedTensor,
    weight_b: _DeviceExpertPackedTensor,
    x_ptr: int,
    selected_ptr: int,
    out_a_ptr: int,
    out_b_ptr: int,
    *,
    x_rows: int,
    rows: int,
    num_experts: int,
    in_features: int,
    out_features: int,
    stream: int,
    runtime: HipRuntime,
    library: object | None = None,
) -> bool:
    key = _EXPERT_PACK8_DUAL_KEYS.get((weight_a.quant_key, weight_b.quant_key))
    if key is None:
        return False
    _validate_expert_pack8_shape(
        weight_a,
        num_experts=num_experts,
        in_features=in_features,
        out_features=out_features,
    )
    _validate_expert_pack8_shape(
        weight_b,
        num_experts=num_experts,
        in_features=in_features,
        out_features=out_features,
    )
    register_gguf_expert_pack8_gemv_kernels()
    fn = resolve(backend=key.backend, layer=key.layer, quant=key.quant, variant=key.variant)
    fn(
        x_ptr,
        selected_ptr,
        weight_a.qweight_low.ptr,
        weight_a.scales.ptr,
        _required_ptr(weight_a.mins, "mins", weight_a.quant_key),
        weight_b.qweight_low.ptr,
        weight_b.scales.ptr,
        _required_ptr(weight_b.mins, "mins", weight_b.quant_key),
        out_a_ptr,
        out_b_ptr,
        x_rows=x_rows,
        rows=rows,
        num_experts=num_experts,
        in_features=in_features,
        out_features=out_features,
        stream=stream,
        runtime=runtime,
        library=library,
    )
    return True


def _launch_selected_expert_pack8_moe_linear(
    weight: _DeviceExpertPackedTensor,
    x_ptr: int,
    selected_ptr: int,
    out_ptr: int,
    *,
    x_rows: int,
    rows: int,
    num_experts: int,
    in_features: int,
    out_features: int,
    stream: int,
    runtime: HipRuntime,
    library: object | None = None,
) -> None:
    try:
        key = _EXPERT_PACK8_SELECTED_KEYS[weight.quant_key]
    except KeyError as exc:
        raise ValueError(f"unsupported expert pack8 quant {weight.quant_key!r}") from exc
    _validate_expert_pack8_shape(
        weight,
        num_experts=num_experts,
        in_features=in_features,
        out_features=out_features,
    )
    register_gguf_expert_pack8_gemv_kernels()
    fn = resolve(backend=key.backend, layer=key.layer, quant=key.quant, variant=key.variant)
    fn(
        x_ptr,
        selected_ptr,
        weight.qweight_low.ptr,
        0 if weight.qweight_high is None else weight.qweight_high.ptr,
        weight.scales.ptr,
        0 if weight.mins is None else weight.mins.ptr,
        out_ptr,
        x_rows=x_rows,
        rows=rows,
        num_experts=num_experts,
        in_features=in_features,
        out_features=out_features,
        stream=stream,
        runtime=runtime,
        library=library,
    )


def _validate_expert_pack8_shape(
    weight: _DeviceExpertPackedTensor,
    *,
    num_experts: int,
    in_features: int,
    out_features: int,
) -> None:
    if weight.num_experts != num_experts or weight.in_features != in_features or weight.out_features != out_features:
        raise ValueError(
            "expert sidecar shape does not match launch: "
            f"sidecar=({weight.num_experts}, {weight.out_features}, {weight.in_features}), "
            f"launch=({num_experts}, {out_features}, {in_features})"
        )


def _required_ptr(buffer: DeviceBuffer | None, name: str, quant_key: str) -> int:
    if buffer is None:
        raise ValueError(f"expert pack8 {quant_key} requires {name}")
    return buffer.ptr


def _try_run_post_attention_moe_rows_compact_wmma(
    runner: Qwen35GGUFFullStackRunner,
    layer,
    gate_weight: Qwen35GGUFDeviceWeight,
    up_weight: Qwen35GGUFDeviceWeight,
    down_weight: Qwen35GGUFDeviceWeight,
    out_ptr: int,
    scratch,
    *,
    rows: int,
    selected_rows: int,
    top_k: int,
    stream: int,
    runtime: HipRuntime,
) -> bool:
    """Run the opt-in P8.6 compact grouped-MoE WMMA path when available."""

    if not gguf_wmma_prefill_enabled(None):
        return False
    if not _scratch_has_compact_moe_fields(scratch):
        return False
    cfg = runner.weights.config if runner.weights is not None else None
    if cfg is None:
        return False
    plan = _resolve_compact_moe_wmma_kernels(gate_weight, up_weight, down_weight)
    if plan is None:
        return False
    gate_up_fn = plan.gate_up_fn
    down_fn = plan.down_fn
    num_experts = int(cfg.expert_count)
    hidden_size = int(runner.hidden_size)
    expert_ffn = int(cfg.expert_feed_forward_length)
    if selected_rows <= 0 or selected_rows > int(getattr(scratch, "moe_selected_rows_capacity", selected_rows)):
        return False
    if hidden_size % 256 != 0 or expert_ffn % 256 != 0 or expert_ffn % 16 != 0:
        return False
    _validate_raw_rank3_expert_weight(
        gate_weight,
        num_experts=num_experts,
        in_features=hidden_size,
        out_features=expert_ffn,
    )
    _validate_raw_rank3_expert_weight(
        up_weight,
        num_experts=num_experts,
        in_features=hidden_size,
        out_features=expert_ffn,
    )
    _validate_raw_rank3_expert_weight(
        down_weight,
        num_experts=num_experts,
        in_features=expert_ffn,
        out_features=hidden_size,
    )

    _zero(runtime, scratch.moe_group_counts, scratch.moe_group_counts_zero, stream=stream)
    qwen35_moe_group_count(
        scratch.moe_selected_experts.ptr,
        scratch.moe_group_counts.ptr,
        selected_rows,
        num_experts,
        stream=stream,
        runtime=runtime,
    )
    qwen35_moe_group_prefix(
        scratch.moe_group_counts.ptr,
        scratch.moe_padded_counts.ptr,
        scratch.moe_expert_start_compact.ptr,
        scratch.moe_total_compact.ptr,
        num_experts,
        1,
        stream=stream,
        runtime=runtime,
    )
    _zero(runtime, scratch.moe_scatter_offsets, scratch.moe_scatter_offsets_zero, stream=stream)
    qwen35_moe_group_scatter_gather_lowp(
        scratch.post_norm.ptr,
        scratch.moe_selected_experts.ptr,
        scratch.moe_routing_weights.ptr,
        scratch.moe_expert_start_compact.ptr,
        scratch.moe_scatter_offsets.ptr,
        scratch.moe_sorted_lanes.ptr,
        scratch.moe_sorted_experts.ptr,
        scratch.moe_sorted_weights.ptr,
        scratch.moe_down_out.ptr,
        selected_rows,
        num_experts,
        top_k,
        hidden_size,
        stream=stream,
        runtime=runtime,
    )
    qwen35_moe_wmma_tile_map(
        scratch.moe_expert_start_compact.ptr,
        scratch.moe_expert_start_wmma.ptr,
        scratch.moe_tile_expert.ptr,
        scratch.moe_wmma_total.ptr,
        num_experts,
        stream=stream,
        runtime=runtime,
    )
    wmma_total_rows = _read_i64_device_scalar(
        scratch.moe_wmma_total,
        scratch.moe_wmma_total_host,
        stream=stream,
        runtime=runtime,
    )
    if wmma_total_rows <= 0 or wmma_total_rows > int(getattr(scratch, "moe_wmma_rows_capacity", wmma_total_rows)):
        return False

    gate_up_fn(
        scratch.moe_down_out.ptr,
        scratch.moe_expert_start_compact.ptr,
        scratch.moe_expert_start_wmma.ptr,
        scratch.moe_tile_expert.ptr,
        gate_weight.allocation(plan.gate_allocation).tensor.ptr,
        up_weight.allocation(plan.up_allocation).tensor.ptr,
        scratch.ffn_gate_up.ptr,
        selected_rows,
        hidden_size,
        expert_ffn,
        expert_ffn,
        num_experts,
        wmma_total_rows,
        stream=stream,
        runtime=runtime,
    )
    silu_mul_dual_out_bf16(
        scratch.ffn_gate_up.ptr,
        scratch.ffn_intermediate.ptr,
        rows=selected_rows,
        features=expert_ffn,
        stream=stream,
        runtime=runtime,
    )
    down_fn(
        scratch.ffn_intermediate.ptr,
        scratch.moe_expert_start_compact.ptr,
        scratch.moe_expert_start_wmma.ptr,
        scratch.moe_tile_expert.ptr,
        down_weight.allocation(plan.down_allocation).tensor.ptr,
        scratch.moe_down_out.ptr,
        selected_rows,
        expert_ffn,
        hidden_size,
        num_experts,
        wmma_total_rows,
        stream=stream,
        runtime=runtime,
    )
    weighted_lanes_sum_out_bf16_f32w(
        scratch.moe_down_out.ptr,
        scratch.moe_sorted_weights.ptr,
        scratch.moe_sorted_lanes.ptr,
        scratch.moe_lane_to_row.ptr,
        scratch.ffn_down.ptr,
        rows,
        top_k,
        hidden_size,
        stream=stream,
        runtime=runtime,
    )

    shared_ffn = int(cfg.expert_shared_feed_forward_length)
    if launch_gguf_linear_pair_concat(
        layer.weight("ffn_gate_shexp"),
        layer.weight("ffn_up_shexp"),
        scratch.post_norm.ptr,
        scratch.ffn_gate_up.ptr,
        rows=rows,
        in_features=hidden_size,
        out_features=shared_ffn,
        stream=stream,
        runtime=runtime,
    ):
        silu_mul_dual_out_bf16(
            scratch.ffn_gate_up.ptr,
            scratch.moe_shared_intermediate.ptr,
            rows=rows,
            features=shared_ffn,
            stream=stream,
            runtime=runtime,
        )
    else:
        if not launch_gguf_linear_pair(
            layer.weight("ffn_gate_shexp"),
            layer.weight("ffn_up_shexp"),
            scratch.post_norm.ptr,
            scratch.moe_shared_gate.ptr,
            scratch.moe_shared_up.ptr,
            rows=rows,
            in_features=hidden_size,
            out_features=shared_ffn,
            stream=stream,
            runtime=runtime,
        ):
            launch_gguf_linear(
                layer.weight("ffn_gate_shexp"),
                scratch.post_norm.ptr,
                scratch.moe_shared_gate.ptr,
                rows=rows,
                in_features=hidden_size,
                out_features=shared_ffn,
                stream=stream,
                runtime=runtime,
            )
            launch_gguf_linear(
                layer.weight("ffn_up_shexp"),
                scratch.post_norm.ptr,
                scratch.moe_shared_up.ptr,
                rows=rows,
                in_features=hidden_size,
                out_features=shared_ffn,
                stream=stream,
                runtime=runtime,
            )
        silu_mul_separate_out_bf16(
            scratch.moe_shared_gate.ptr,
            scratch.moe_shared_up.ptr,
            scratch.moe_shared_intermediate.ptr,
            rows=rows,
            features=shared_ffn,
            stream=stream,
            runtime=runtime,
        )
    launch_gguf_linear(
        layer.weight("ffn_down_shexp"),
        scratch.moe_shared_intermediate.ptr,
        scratch.moe_shared_out.ptr,
        rows=rows,
        in_features=int(cfg.expert_shared_feed_forward_length),
        out_features=hidden_size,
        stream=stream,
        runtime=runtime,
    )
    shared_gate_combine_residual_batch_out_bf16(
        scratch.ffn_down.ptr,
        scratch.moe_shared_out.ptr,
        scratch.moe_shared_gate_logits.ptr,
        scratch.residual.ptr,
        out_ptr,
        rows,
        hidden_size,
        1,
        stream=stream,
        runtime=runtime,
    )
    return True


def _scratch_has_compact_moe_fields(scratch) -> bool:
    return all(hasattr(scratch, name) for name in _COMPACT_MOE_REQUIRED_SCRATCH)


def _try_run_post_attention_moe_rows_compact_gemv(
    runner: Qwen35GGUFFullStackRunner,
    layer,
    gate_weight: Qwen35GGUFDeviceWeight,
    up_weight: Qwen35GGUFDeviceWeight,
    down_weight: Qwen35GGUFDeviceWeight,
    out_ptr: int,
    scratch,
    *,
    rows: int,
    selected_rows: int,
    top_k: int,
    stream: int,
    runtime: HipRuntime,
) -> bool:
    """Run the resident T16 compact GEMV path for row-bulk MoE prefill.

    Decode-repack mode replaces selected expert raw bytes with T16 tiles.  The
    existing row-bulk fallback cannot consume those replacements, so use the
    same compact scheduler as the WMMA path and the T16 selected GEMV kernels
    for all routed lanes.  Raw pack8 GEMV is intentionally not enabled here;
    this branch is a correctness-preserving replacement-layout fallback for
    bulk prefill, not a promotion of the old unsafe raw-GGUF decode kernels.
    """

    if not gguf_gemv_decode_enabled(None):
        return False
    if not _scratch_has_compact_moe_gemv_fields(scratch):
        return False
    cfg = runner.weights.config if runner.weights is not None else None
    if cfg is None:
        return False
    plan = _resolve_compact_moe_gemv_kernels(gate_weight, up_weight, down_weight)
    if plan is None:
        return False
    if (plan.gate_allocation, plan.up_allocation, plan.down_allocation) != ("tiles", "tiles", "tiles"):
        return False
    gate_up_fn = plan.gate_up_fn
    down_fn = plan.down_fn
    num_experts = int(cfg.expert_count)
    hidden_size = int(runner.hidden_size)
    expert_ffn = int(cfg.expert_feed_forward_length)
    if rows <= 0 or selected_rows <= 0:
        return False
    if selected_rows > int(getattr(scratch, "moe_selected_rows_capacity", selected_rows)):
        return False
    if hidden_size % 256 != 0 or expert_ffn % 256 != 0 or expert_ffn % 16 != 0:
        return False
    _validate_raw_rank3_expert_weight(
        gate_weight,
        num_experts=num_experts,
        in_features=hidden_size,
        out_features=expert_ffn,
    )
    _validate_raw_rank3_expert_weight(
        up_weight,
        num_experts=num_experts,
        in_features=hidden_size,
        out_features=expert_ffn,
    )
    _validate_raw_rank3_expert_weight(
        down_weight,
        num_experts=num_experts,
        in_features=expert_ffn,
        out_features=hidden_size,
    )

    _zero(runtime, scratch.moe_group_counts, scratch.moe_group_counts_zero, stream=stream)
    qwen35_moe_group_count(
        scratch.moe_selected_experts.ptr,
        scratch.moe_group_counts.ptr,
        selected_rows,
        num_experts,
        stream=stream,
        runtime=runtime,
    )
    qwen35_moe_group_prefix(
        scratch.moe_group_counts.ptr,
        scratch.moe_padded_counts.ptr,
        scratch.moe_expert_start_compact.ptr,
        scratch.moe_total_compact.ptr,
        num_experts,
        1,
        stream=stream,
        runtime=runtime,
    )
    _zero(runtime, scratch.moe_scatter_offsets, scratch.moe_scatter_offsets_zero, stream=stream)
    qwen35_moe_group_scatter_gather_lowp(
        scratch.post_norm.ptr,
        scratch.moe_selected_experts.ptr,
        scratch.moe_routing_weights.ptr,
        scratch.moe_expert_start_compact.ptr,
        scratch.moe_scatter_offsets.ptr,
        scratch.moe_sorted_lanes.ptr,
        scratch.moe_sorted_experts.ptr,
        scratch.moe_sorted_weights.ptr,
        scratch.moe_down_out.ptr,
        selected_rows,
        num_experts,
        top_k,
        hidden_size,
        stream=stream,
        runtime=runtime,
    )
    gate_up_fn(
        scratch.moe_down_out.ptr,
        scratch.moe_expert_start_compact.ptr,
        gate_weight.allocation(plan.gate_allocation).tensor.ptr,
        up_weight.allocation(plan.up_allocation).tensor.ptr,
        scratch.ffn_gate_up.ptr,
        selected_rows,
        hidden_size,
        expert_ffn,
        expert_ffn,
        num_experts,
        stream=stream,
        runtime=runtime,
    )
    silu_mul_dual_out_bf16(
        scratch.ffn_gate_up.ptr,
        scratch.ffn_intermediate.ptr,
        rows=selected_rows,
        features=expert_ffn,
        stream=stream,
        runtime=runtime,
    )
    down_input_ptr = scratch.ffn_intermediate.ptr
    if _selected_gemv_requires_q8_1_input(down_weight):
        q8_1_workspace_ptr = _optional_q8_1_workspace_ptr(
            scratch,
            selected_rows,
            expert_ffn,
            enabled=True,
        )
        if q8_1_workspace_ptr is None:
            return False
        gguf_q4_k_quantize_bf16_q8_1(
            scratch.ffn_intermediate.ptr,
            q8_1_workspace_ptr,
            selected_rows,
            expert_ffn,
            stream=stream,
            runtime=runtime,
        )
        down_input_ptr = q8_1_workspace_ptr
    down_fn(
        down_input_ptr,
        scratch.moe_expert_start_compact.ptr,
        down_weight.allocation(plan.down_allocation).tensor.ptr,
        scratch.moe_down_out.ptr,
        selected_rows,
        expert_ffn,
        hidden_size,
        num_experts,
        stream=stream,
        runtime=runtime,
    )
    weighted_lanes_sum_out_bf16_f32w(
        scratch.moe_down_out.ptr,
        scratch.moe_sorted_weights.ptr,
        scratch.moe_sorted_lanes.ptr,
        scratch.moe_lane_to_row.ptr,
        scratch.ffn_down.ptr,
        rows,
        top_k,
        hidden_size,
        stream=stream,
        runtime=runtime,
    )

    shared_ffn = int(cfg.expert_shared_feed_forward_length)
    if launch_gguf_linear_pair_concat(
        layer.weight("ffn_gate_shexp"),
        layer.weight("ffn_up_shexp"),
        scratch.post_norm.ptr,
        scratch.ffn_gate_up.ptr,
        rows=rows,
        in_features=hidden_size,
        out_features=shared_ffn,
        stream=stream,
        runtime=runtime,
    ):
        silu_mul_dual_out_bf16(
            scratch.ffn_gate_up.ptr,
            scratch.moe_shared_intermediate.ptr,
            rows=rows,
            features=shared_ffn,
            stream=stream,
            runtime=runtime,
        )
    else:
        if not launch_gguf_linear_pair(
            layer.weight("ffn_gate_shexp"),
            layer.weight("ffn_up_shexp"),
            scratch.post_norm.ptr,
            scratch.moe_shared_gate.ptr,
            scratch.moe_shared_up.ptr,
            rows=rows,
            in_features=hidden_size,
            out_features=shared_ffn,
            stream=stream,
            runtime=runtime,
        ):
            launch_gguf_linear(
                layer.weight("ffn_gate_shexp"),
                scratch.post_norm.ptr,
                scratch.moe_shared_gate.ptr,
                rows=rows,
                in_features=hidden_size,
                out_features=shared_ffn,
                stream=stream,
                runtime=runtime,
            )
            launch_gguf_linear(
                layer.weight("ffn_up_shexp"),
                scratch.post_norm.ptr,
                scratch.moe_shared_up.ptr,
                rows=rows,
                in_features=hidden_size,
                out_features=shared_ffn,
                stream=stream,
                runtime=runtime,
            )
        silu_mul_separate_out_bf16(
            scratch.moe_shared_gate.ptr,
            scratch.moe_shared_up.ptr,
            scratch.moe_shared_intermediate.ptr,
            rows=rows,
            features=shared_ffn,
            stream=stream,
            runtime=runtime,
        )
    launch_gguf_linear(
        layer.weight("ffn_down_shexp"),
        scratch.moe_shared_intermediate.ptr,
        scratch.moe_shared_out.ptr,
        rows=rows,
        in_features=shared_ffn,
        out_features=hidden_size,
        stream=stream,
        runtime=runtime,
    )
    shared_gate_combine_residual_batch_out_bf16(
        scratch.ffn_down.ptr,
        scratch.moe_shared_out.ptr,
        scratch.moe_shared_gate_logits.ptr,
        scratch.residual.ptr,
        out_ptr,
        rows,
        hidden_size,
        1,
        stream=stream,
        runtime=runtime,
    )
    return True



def _try_run_post_attention_moe_c1_fused_ffn(
    runner: Qwen35GGUFFullStackRunner,
    layer,
    gate_weight: Qwen35GGUFDeviceWeight,
    up_weight: Qwen35GGUFDeviceWeight,
    down_weight: Qwen35GGUFDeviceWeight,
    scratch,
    *,
    top_k: int,
    stream: int,
    runtime: HipRuntime,
) -> bool:
    """Opt-in B1 fused selected-expert MoE FFN megakernel for rows==1 decode.

    Replaces the gate_up GEMV -> silu*mul -> down GEMV chain (3 launches) with a
    single fused launch writing ``scratch.moe_down_out`` (per-selected-row down);
    the shared expert + routing-weighted combine run unchanged afterward. Applies
    only when all three expert tensors are raw ``gguf_q4_k`` (the megakernel reads
    raw Q4_K bytes); returns ``False`` for T16/repacked layouts so the caller
    falls back to the unfused chain.
    """

    cfg = runner.weights.config if runner.weights is not None else None
    if cfg is None:
        return False
    for weight in (gate_weight, up_weight, down_weight):
        if weight.spec.quant_key != "gguf_q4_k":
            return False
    num_experts = int(cfg.expert_count)
    hidden_size = int(runner.hidden_size)
    expert_ffn = int(cfg.expert_feed_forward_length)
    if hidden_size % 256 != 0 or expert_ffn % 256 != 0:
        return False
    if top_k <= 0 or top_k > int(getattr(scratch, "moe_selected_rows_capacity", top_k)):
        return False
    _validate_raw_rank3_expert_weight(
        gate_weight, num_experts=num_experts, in_features=hidden_size, out_features=expert_ffn
    )
    _validate_raw_rank3_expert_weight(
        up_weight, num_experts=num_experts, in_features=hidden_size, out_features=expert_ffn
    )
    _validate_raw_rank3_expert_weight(
        down_weight, num_experts=num_experts, in_features=expert_ffn, out_features=hidden_size
    )
    gguf_q4_k_selected_ffn_fused_bf16_bf16_out(
        scratch.post_norm.ptr,
        scratch.moe_selected_experts.ptr,
        gate_weight.allocation("raw").tensor.ptr,
        up_weight.allocation("raw").tensor.ptr,
        down_weight.allocation("raw").tensor.ptr,
        scratch.moe_down_out.ptr,
        1,
        top_k,
        num_experts,
        hidden_size,
        expert_ffn,
        stream=stream,
        runtime=runtime,
    )
    return True


def _try_run_post_attention_moe_c1_compact_gemv(
    runner: Qwen35GGUFFullStackRunner,
    layer,
    gate_weight: Qwen35GGUFDeviceWeight,
    up_weight: Qwen35GGUFDeviceWeight,
    down_weight: Qwen35GGUFDeviceWeight,
    out_ptr: int,
    scratch,
    *,
    top_k: int,
    stream: int,
    runtime: HipRuntime,
) -> bool:
    """Run the opt-in P9.B6 compact grouped-MoE GEMV decode path when available.

    Mirrors :func:`_try_run_post_attention_moe_rows_compact_wmma` but for
    ``rows == 1`` decode. Differences from the bulk WMMA path:

    * The compact scheduler runs without ``qwen35_moe_wmma_tile_map`` (GEMV
      does not consume the WMMA tile space).
    * Inner kernels are P9.B1 / P9.B2 ``selected_*_pack8_gemv_decode_compact``
      instead of P8.4 / P8.5 WMMA prefill variants.
    * The compact-row count equals ``top_k`` (one active lane per expert per
      decode token).

    Returns ``False`` when any of the gating conditions fails so the caller
    falls back to the legacy per-row selected GEMV path.
    """

    if not gguf_gemv_decode_enabled(None):
        return False
    if not _scratch_has_compact_moe_gemv_fields(scratch):
        return False
    cfg = runner.weights.config if runner.weights is not None else None
    if cfg is None:
        return False
    plan = _resolve_compact_moe_gemv_kernels(gate_weight, up_weight, down_weight)
    if plan is None:
        return False
    if (plan.gate_allocation, plan.up_allocation, plan.down_allocation) == ("tiles", "tiles", "tiles"):
        return False
    gate_up_fn = plan.gate_up_fn
    down_fn = plan.down_fn
    num_experts = int(cfg.expert_count)
    hidden_size = int(runner.hidden_size)
    expert_ffn = int(cfg.expert_feed_forward_length)
    if top_k <= 0 or top_k > int(getattr(scratch, "moe_selected_rows_capacity", top_k)):
        return False
    if hidden_size % 256 != 0 or expert_ffn % 256 != 0 or expert_ffn % 8 != 0:
        return False
    _validate_raw_rank3_expert_weight(
        gate_weight,
        num_experts=num_experts,
        in_features=hidden_size,
        out_features=expert_ffn,
    )
    _validate_raw_rank3_expert_weight(
        up_weight,
        num_experts=num_experts,
        in_features=hidden_size,
        out_features=expert_ffn,
    )
    _validate_raw_rank3_expert_weight(
        down_weight,
        num_experts=num_experts,
        in_features=expert_ffn,
        out_features=hidden_size,
    )

    _zero(runtime, scratch.moe_group_counts, scratch.moe_group_counts_zero, stream=stream)
    qwen35_moe_group_count(
        scratch.moe_selected_experts.ptr,
        scratch.moe_group_counts.ptr,
        top_k,
        num_experts,
        stream=stream,
        runtime=runtime,
    )
    qwen35_moe_group_prefix(
        scratch.moe_group_counts.ptr,
        scratch.moe_padded_counts.ptr,
        scratch.moe_expert_start_compact.ptr,
        scratch.moe_total_compact.ptr,
        num_experts,
        1,
        stream=stream,
        runtime=runtime,
    )
    _zero(runtime, scratch.moe_scatter_offsets, scratch.moe_scatter_offsets_zero, stream=stream)
    qwen35_moe_group_scatter_gather_lowp(
        scratch.post_norm.ptr,
        scratch.moe_selected_experts.ptr,
        scratch.moe_routing_weights.ptr,
        scratch.moe_expert_start_compact.ptr,
        scratch.moe_scatter_offsets.ptr,
        scratch.moe_sorted_lanes.ptr,
        scratch.moe_sorted_experts.ptr,
        scratch.moe_sorted_weights.ptr,
        scratch.moe_down_out.ptr,
        top_k,
        num_experts,
        top_k,
        hidden_size,
        stream=stream,
        runtime=runtime,
    )
    gate_up_fn(
        scratch.moe_down_out.ptr,
        scratch.moe_expert_start_compact.ptr,
        gate_weight.allocation(plan.gate_allocation).tensor.ptr,
        up_weight.allocation(plan.up_allocation).tensor.ptr,
        scratch.ffn_gate_up.ptr,
        top_k,
        hidden_size,
        expert_ffn,
        expert_ffn,
        num_experts,
        stream=stream,
        runtime=runtime,
    )
    silu_mul_dual_out_bf16(
        scratch.ffn_gate_up.ptr,
        scratch.ffn_intermediate.ptr,
        rows=top_k,
        features=expert_ffn,
        stream=stream,
        runtime=runtime,
    )
    down_fn(
        scratch.ffn_intermediate.ptr,
        scratch.moe_expert_start_compact.ptr,
        down_weight.allocation(plan.down_allocation).tensor.ptr,
        scratch.moe_down_out.ptr,
        top_k,
        expert_ffn,
        hidden_size,
        num_experts,
        stream=stream,
        runtime=runtime,
    )
    weighted_lanes_sum_out_bf16_f32w(
        scratch.moe_down_out.ptr,
        scratch.moe_sorted_weights.ptr,
        scratch.moe_sorted_lanes.ptr,
        scratch.moe_lane_to_row.ptr,
        scratch.ffn_down.ptr,
        1,
        top_k,
        hidden_size,
        stream=stream,
        runtime=runtime,
    )

    shared_ffn = int(cfg.expert_shared_feed_forward_length)
    if launch_gguf_linear_pair_concat(
        layer.weight("ffn_gate_shexp"),
        layer.weight("ffn_up_shexp"),
        scratch.post_norm.ptr,
        scratch.ffn_gate_up.ptr,
        rows=1,
        in_features=hidden_size,
        out_features=shared_ffn,
        stream=stream,
        runtime=runtime,
    ):
        silu_mul_dual_out_bf16(
            scratch.ffn_gate_up.ptr,
            scratch.moe_shared_intermediate.ptr,
            rows=1,
            features=shared_ffn,
            stream=stream,
            runtime=runtime,
        )
    else:
        if not launch_gguf_linear_pair(
            layer.weight("ffn_gate_shexp"),
            layer.weight("ffn_up_shexp"),
            scratch.post_norm.ptr,
            scratch.moe_shared_gate.ptr,
            scratch.moe_shared_up.ptr,
            rows=1,
            in_features=hidden_size,
            out_features=shared_ffn,
            stream=stream,
            runtime=runtime,
        ):
            launch_gguf_linear(
                layer.weight("ffn_gate_shexp"),
                scratch.post_norm.ptr,
                scratch.moe_shared_gate.ptr,
                rows=1,
                in_features=hidden_size,
                out_features=shared_ffn,
                stream=stream,
                runtime=runtime,
            )
            launch_gguf_linear(
                layer.weight("ffn_up_shexp"),
                scratch.post_norm.ptr,
                scratch.moe_shared_up.ptr,
                rows=1,
                in_features=hidden_size,
                out_features=shared_ffn,
                stream=stream,
                runtime=runtime,
            )
        silu_mul_separate_out_bf16(
            scratch.moe_shared_gate.ptr,
            scratch.moe_shared_up.ptr,
            scratch.moe_shared_intermediate.ptr,
            rows=1,
            features=shared_ffn,
            stream=stream,
            runtime=runtime,
        )
    launch_gguf_linear(
        layer.weight("ffn_down_shexp"),
        scratch.moe_shared_intermediate.ptr,
        scratch.moe_shared_out.ptr,
        rows=1,
        in_features=int(cfg.expert_shared_feed_forward_length),
        out_features=hidden_size,
        stream=stream,
        runtime=runtime,
    )
    shared_gate_combine_residual_batch_out_bf16(
        scratch.ffn_down.ptr,
        scratch.moe_shared_out.ptr,
        scratch.moe_router_logits.ptr + num_experts * 4,
        scratch.residual.ptr,
        out_ptr,
        1,
        hidden_size,
        1,
        stream=stream,
        runtime=runtime,
    )
    return True


def _resolve_compact_moe_wmma_kernels(
    gate_weight: Qwen35GGUFDeviceWeight,
    up_weight: Qwen35GGUFDeviceWeight,
    down_weight: Qwen35GGUFDeviceWeight,
):
    """Resolve the compact selected MoE WMMA prefill chain.

    Returns a :class:`_CompactMoeWmmaPlan` carrying the resolved gate+up /
    down callables and the per-weight allocation name (``"raw"`` for raw
    GGUF, ``"tiles"`` for T16 decode-repack). Mirrors
    :func:`_resolve_compact_moe_gemv_kernels` so the same plan structure
    can route to either layout family without a quant branch in the caller.

    Falls back to ``None`` when any required kernel is missing so the
    runtime can transparently use the slower per-row fallback paths.
    """

    gate_up_key = _COMPACT_MOE_Q4_DUAL_KEYS.get((gate_weight.spec.quant_key, up_weight.spec.quant_key))
    down_key = _COMPACT_MOE_DOWN_KEYS.get(down_weight.spec.quant_key)
    if gate_up_key is None or down_key is None:
        return None
    required = (*_COMPACT_MOE_SCHEDULER_KEYS, *_COMPACT_MOE_FUSED_KEYS, gate_up_key, down_key)
    resolved = _resolve_compact_moe_required_keys(required)
    if any(fn is None for fn in resolved):
        _ensure_compact_moe_wmma_registered()
        resolved = _resolve_compact_moe_required_keys(required)
    if any(fn is None for fn in resolved):
        return None
    return _CompactMoeWmmaPlan(
        gate_up_fn=resolved[-2],
        down_fn=resolved[-1],
        gate_allocation=_selected_wmma_allocation_name(gate_weight),
        up_allocation=_selected_wmma_allocation_name(up_weight),
        down_allocation=_selected_wmma_allocation_name(down_weight),
    )


def _selected_wmma_allocation_name(weight: Qwen35GGUFDeviceWeight) -> str:
    """Return the allocation name for the WMMA prefill chain.

    Raw-layout quant keys carry their bytes in the ``"raw"`` allocation
    (single contiguous rank-3 buffer). T16 decode-repack quant keys keep
    the byte-lossless tiles under ``"tiles"`` (see
    ``docs/GGUF_DECODE_REPACK.md``). The compact WMMA prefill kernels
    accept whichever layout was materialized via the same compact ABI;
    dispatch picks the allocation name here so the runner stays
    quant-agnostic.
    """

    return "tiles" if weight.spec.quant_key.endswith("_t16_v1") else "raw"


def _resolve_compact_moe_required_keys(keys: tuple[KernelKey, ...]):
    return [
        resolve(
            backend=key.backend,
            layer=key.layer,
            quant=key.quant,
            variant=key.variant,
            missing="none",
        )
        for key in keys
    ]


def _ensure_compact_moe_wmma_registered() -> None:
    register_qwen35_moe_group_scatter_kernels()
    register_paro_silu_kernels()
    register_paro_combine_kernels()
    register_gguf_q4_k_selected_prefill_kernels()
    register_gguf_q4_k_t16_selected_prefill_kernels()
    register_gguf_k_selected_prefill_kernels()
    register_gguf_k_t16_selected_prefill_kernels()


def _ensure_compact_moe_gemv_registered() -> None:
    """Register P9.B kernels referenced by the compact GEMV decode path."""

    register_qwen35_moe_group_scatter_kernels()
    register_paro_silu_kernels()
    register_paro_combine_kernels()
    register_gguf_q4_k_selected_pack8_gemv_kernels()
    register_gguf_k_selected_pack8_gemv_kernels()
    register_gguf_t16_selected_gemv_kernels()
    register_gguf_x8_selected_gemv_kernels()


def _scratch_has_compact_moe_gemv_fields(scratch) -> bool:
    return all(hasattr(scratch, name) for name in _COMPACT_MOE_GEMV_DECODE_SCRATCH)


def _resolve_compact_moe_gemv_kernels(
    gate_weight: Qwen35GGUFDeviceWeight,
    up_weight: Qwen35GGUFDeviceWeight,
    down_weight: Qwen35GGUFDeviceWeight,
):
    """Resolve P9.B1/B2 compact selected GEMV kernels for the decode chain.

    Mirrors :func:`_resolve_compact_moe_wmma_kernels` but for the rows=1
    decode path. The compact scheduler keys remain shared with the prefill
    WMMA path (group_count/prefix/scatter_gather); ``wmma_tile_map`` is
    deliberately not required here because GEMV does not consume the WMMA
    tile space. Falls back to ``None`` when any required kernel is missing
    so the runtime can transparently use the legacy per-row selected GEMV.
    """

    gate_up_key = _COMPACT_MOE_Q4_DUAL_GEMV_KEYS.get(
        (gate_weight.spec.quant_key, up_weight.spec.quant_key)
    )
    down_key = _COMPACT_MOE_DOWN_GEMV_KEYS.get(down_weight.spec.quant_key)
    if gate_up_key is None or down_key is None:
        return None
    scheduler_keys = (
        KernelKey("hip_gfx1100", "moe_group_count", "w4_paro", "qwen35"),
        KernelKey("hip_gfx1100", "moe_group_prefix", "w4_paro", "qwen35"),
        KernelKey("hip_gfx1100", "moe_group_scatter_gather", "w4_paro", "qwen35_lowp"),
    )
    required = (*scheduler_keys, *_COMPACT_MOE_FUSED_KEYS, gate_up_key, down_key)
    resolved = _resolve_compact_moe_required_keys(required)
    if any(fn is None for fn in resolved):
        _ensure_compact_moe_gemv_registered()
        resolved = _resolve_compact_moe_required_keys(required)
    if any(fn is None for fn in resolved):
        return None
    return _CompactMoeGemvPlan(
        gate_up_fn=resolved[-2],
        down_fn=resolved[-1],
        gate_allocation=_selected_gemv_allocation_name(gate_weight),
        up_allocation=_selected_gemv_allocation_name(up_weight),
        down_allocation=_selected_gemv_allocation_name(down_weight),
    )


def _selected_gemv_allocation_name(weight: Qwen35GGUFDeviceWeight) -> str:
    quant_key = weight.spec.quant_key
    return "tiles" if quant_key.endswith("_t16_v1") or quant_key.endswith("_x8_v1") else "raw"


def _selected_gemv_requires_q8_1_input(weight: Qwen35GGUFDeviceWeight) -> bool:
    return weight.spec.quant_key in {"gguf_q5_k_x8_v1", "gguf_q6_k_x8_v1"}


def _selected_pair_requires_q8_1_input(
    weight_a: Qwen35GGUFDeviceWeight,
    weight_b: Qwen35GGUFDeviceWeight,
) -> bool:
    return weight_a.spec.quant_key == "gguf_q4_k_x8_v1" and weight_b.spec.quant_key == "gguf_q4_k_x8_v1"


def _validate_raw_rank3_expert_weight(
    weight: Qwen35GGUFDeviceWeight,
    *,
    num_experts: int,
    in_features: int,
    out_features: int,
) -> None:
    source = weight.spec.source
    if len(source.shape) != 3 or len(source.byte_shape) != 3:
        raise ValueError(f"GGUF expert tensor {source.name!r} must be rank-3, got {source.shape}")
    experts, rows, row_bytes = (int(v) for v in source.byte_shape)
    if experts != int(num_experts) or rows != int(out_features):
        raise ValueError(
            "GGUF compact expert tensor shape does not match launch: "
            f"tensor=({experts}, {rows}, {row_bytes}), "
            f"launch=({num_experts}, {out_features}, in_features={in_features})"
        )
    if row_bytes <= 0 or int(in_features) <= 0:
        raise ValueError(f"invalid GGUF expert tensor shape for {source.name!r}")


def _copy_bf16_ptr_to_host_f32(ptr: int, elements: int, *, runtime: HipRuntime) -> np.ndarray:
    elements = int(elements)
    if elements <= 0:
        raise ValueError("elements must be positive")
    bits = np.empty((elements,), dtype=np.uint16)
    copy_device_to_host(
        host_array_ptr(bits),
        DeviceBuffer(int(ptr), elements * DType.BF16.itemsize),
        bits.nbytes,
        runtime=runtime,
    )
    return bf16_to_float32(bits)


def _copy_f32_ptr_to_host(ptr: int, elements: int, *, runtime: HipRuntime) -> np.ndarray:
    elements = int(elements)
    if elements <= 0:
        raise ValueError("elements must be positive")
    values = np.empty((elements,), dtype=np.float32)
    copy_device_to_host(
        host_array_ptr(values),
        DeviceBuffer(int(ptr), values.nbytes),
        values.nbytes,
        runtime=runtime,
    )
    return values


def _copy_i64_ptr_to_host(ptr: int, elements: int, *, runtime: HipRuntime) -> np.ndarray:
    elements = int(elements)
    if elements <= 0:
        raise ValueError("elements must be positive")
    values = np.empty((elements,), dtype=np.int64)
    copy_device_to_host(
        host_array_ptr(values),
        DeviceBuffer(int(ptr), values.nbytes),
        values.nbytes,
        runtime=runtime,
    )
    return values


def _read_i64_device_scalar(buffer, host: np.ndarray, *, stream: int = 0, runtime: HipRuntime) -> int:
    if stream:
        runtime.stream_synchronize(stream)
    copy_device_to_host(host_array_ptr(host), buffer, host.nbytes, runtime=runtime)
    return int(host[0])


def _launch_selected_raw_gguf_moe_pair_silu(
    weight_a: Qwen35GGUFDeviceWeight,
    weight_b: Qwen35GGUFDeviceWeight,
    x_ptr: int,
    selected_ptr: int,
    out_ptr: int,
    *,
    x_rows: int,
    rows: int,
    num_experts: int,
    in_features: int,
    out_features: int,
    stream: int,
    runtime: HipRuntime,
) -> bool:
    if weight_a.spec.quant_key == "gguf_q4_k_t16_v1" and weight_b.spec.quant_key == "gguf_q4_k_t16_v1":
        # The q8_1+sudot4 fused-SiLU T16 diagnostic is callable, but the
        # production c1 trace regressed it on gfx1151. Keep c1 on the exact
        # float-dequant fused kernel and reserve dp4a routing for rows>1 split
        # gate/up where it measured faster.
        gguf_q4_k_t16_selected_dual_silu_gemv_bf16_bf16_out(
            x_ptr,
            selected_ptr,
            weight_a.allocation("tiles").tensor.ptr,
            weight_b.allocation("tiles").tensor.ptr,
            out_ptr,
            x_rows,
            rows,
            num_experts,
            in_features,
            out_features,
            stream=stream,
            runtime=runtime,
        )
        return True
    return False


def _launch_selected_raw_gguf_moe_pair(
    weight_a: Qwen35GGUFDeviceWeight,
    weight_b: Qwen35GGUFDeviceWeight,
    x_ptr: int,
    selected_ptr: int,
    out_a_ptr: int,
    out_b_ptr: int,
    *,
    x_rows: int,
    rows: int,
    num_experts: int,
    in_features: int,
    out_features: int,
    q8_1_workspace_ptr: int | None = None,
    stream: int,
    runtime: HipRuntime,
    stage_timings: dict[str, float] | None = None,
    sync_stage_timings: bool = False,
    stage_prefix: str | None = None,
) -> bool:
    sync_stages = bool(sync_stage_timings and stage_timings is not None and stage_prefix)
    t_stage = time.perf_counter() if sync_stages else 0.0
    if weight_a.spec.quant_key == "gguf_q4_k" and weight_b.spec.quant_key == "gguf_q4_k":
        if q8_1_workspace_ptr is not None and (
            _gguf_q4k_selected_dual_dp4a_enabled() or _gguf_raw_selected_dp4a_enabled()
        ):
            gguf_q4_k_quantize_bf16_q8_1(
                x_ptr,
                q8_1_workspace_ptr,
                x_rows,
                in_features,
                stream=stream,
                runtime=runtime,
            )
            t_stage = _mark_sync_stage(
                runtime,
                stage_timings,
                sync_stages,
                f"{stage_prefix}_q8_quantize",
                t_stage,
            )
            gguf_q4_k_selected_dual_q8_1_dp4a_gemv_bf16_bf16_out(
                q8_1_workspace_ptr,
                selected_ptr,
                weight_a.allocation("raw").tensor.ptr,
                weight_b.allocation("raw").tensor.ptr,
                out_a_ptr,
                out_b_ptr,
                x_rows,
                rows,
                num_experts,
                in_features,
                out_features,
                stream=stream,
                runtime=runtime,
            )
            _mark_sync_stage(
                runtime,
                stage_timings,
                sync_stages,
                f"{stage_prefix}_gemv",
                t_stage,
            )
        else:
            gguf_q4_k_selected_dual_gemv_bf16_bf16_out(
                x_ptr,
                selected_ptr,
                weight_a.allocation("raw").tensor.ptr,
                weight_b.allocation("raw").tensor.ptr,
                out_a_ptr,
                out_b_ptr,
                x_rows,
                rows,
                num_experts,
                in_features,
                out_features,
                stream=stream,
                runtime=runtime,
            )
            _mark_sync_stage(
                runtime,
                stage_timings,
                sync_stages,
                f"{stage_prefix}_gemv",
                t_stage,
            )
        return True
    if weight_a.spec.quant_key == "gguf_q4_k_t16_v1" and weight_b.spec.quant_key == "gguf_q4_k_t16_v1":
        if q8_1_workspace_ptr is not None and (
            _gguf_q4k_selected_dual_dp4a_enabled() or _gguf_t16_selected_dp4a_enabled()
        ):
            gguf_q4_k_quantize_bf16_q8_1(
                x_ptr,
                q8_1_workspace_ptr,
                x_rows,
                in_features,
                stream=stream,
                runtime=runtime,
            )
            t_stage = _mark_sync_stage(
                runtime,
                stage_timings,
                sync_stages,
                f"{stage_prefix}_q8_quantize",
                t_stage,
            )
            gguf_q4_k_t16_selected_dual_q8_1_dp4a_gemv_bf16_bf16_out(
                q8_1_workspace_ptr,
                selected_ptr,
                weight_a.allocation("tiles").tensor.ptr,
                weight_b.allocation("tiles").tensor.ptr,
                out_a_ptr,
                out_b_ptr,
                x_rows,
                rows,
                num_experts,
                in_features,
                out_features,
                stream=stream,
                runtime=runtime,
            )
            _mark_sync_stage(
                runtime,
                stage_timings,
                sync_stages,
                f"{stage_prefix}_gemv",
                t_stage,
            )
        else:
            gguf_q4_k_t16_selected_dual_gemv_bf16_bf16_out(
                x_ptr,
                selected_ptr,
                weight_a.allocation("tiles").tensor.ptr,
                weight_b.allocation("tiles").tensor.ptr,
                out_a_ptr,
                out_b_ptr,
                x_rows,
                rows,
                num_experts,
                in_features,
                out_features,
                stream=stream,
                runtime=runtime,
            )
            _mark_sync_stage(
                runtime,
                stage_timings,
                sync_stages,
                f"{stage_prefix}_gemv",
                t_stage,
            )
        return True
    if weight_a.spec.quant_key == "gguf_q4_k_x8_v1" and weight_b.spec.quant_key == "gguf_q4_k_x8_v1":
        if q8_1_workspace_ptr is None:
            raise ValueError("gguf_q4_k_x8_v1 selected-dual GEMV requires q8_1 workspace")
        gguf_q4_k_quantize_bf16_q8_1(
            x_ptr,
            q8_1_workspace_ptr,
            x_rows,
            in_features,
            stream=stream,
            runtime=runtime,
        )
        t_stage = _mark_sync_stage(
            runtime,
            stage_timings,
            sync_stages,
            f"{stage_prefix}_q8_quantize",
            t_stage,
        )
        gguf_q4_k_x8_selected_dual_q8_1_dp4a_gemv_bf16_bf16_out(
            q8_1_workspace_ptr,
            selected_ptr,
            weight_a.allocation("tiles").tensor.ptr,
            weight_b.allocation("tiles").tensor.ptr,
            out_a_ptr,
            out_b_ptr,
            x_rows,
            rows,
            num_experts,
            in_features,
            out_features,
            stream=stream,
            runtime=runtime,
        )
        _mark_sync_stage(
            runtime,
            stage_timings,
            sync_stages,
            f"{stage_prefix}_gemv",
            t_stage,
        )
        return True
    return False


def _launch_selected_raw_gguf_moe_linear(
    weight: Qwen35GGUFDeviceWeight,
    x_ptr: int,
    selected_ptr: int,
    out_ptr: int,
    *,
    x_rows: int,
    rows: int,
    num_experts: int,
    in_features: int,
    out_features: int,
    q8_1_workspace_ptr: int | None = None,
    stream: int,
    runtime: HipRuntime,
    stage_timings: dict[str, float] | None = None,
    sync_stage_timings: bool = False,
    stage_prefix: str | None = None,
) -> None:
    quant_key = weight.spec.quant_key
    allocation = _selected_gemv_allocation_name(weight)
    use_q8_1_input = False
    sync_stages = bool(sync_stage_timings and stage_timings is not None and stage_prefix)
    t_stage = time.perf_counter() if sync_stages else 0.0
    if (
        q8_1_workspace_ptr is not None
        and quant_key == "gguf_q5_k_t16_v1"
        and _gguf_t16_selected_dp4a_enabled()
    ):
        gguf_q4_k_quantize_bf16_q8_1(
            x_ptr,
            q8_1_workspace_ptr,
            x_rows,
            in_features,
            stream=stream,
            runtime=runtime,
        )
        t_stage = _mark_sync_stage(
            runtime,
            stage_timings,
            sync_stages,
            f"{stage_prefix}_q8_quantize",
            t_stage,
        )
        fn = gguf_q5_k_t16_selected_q8_1_dp4a_gemv_bf16_bf16_out
        use_q8_1_input = True
    elif quant_key == "gguf_q5_k_x8_v1":
        if q8_1_workspace_ptr is None:
            raise ValueError("gguf_q5_k_x8_v1 selected GEMV requires q8_1 workspace")
        gguf_q4_k_quantize_bf16_q8_1(
            x_ptr,
            q8_1_workspace_ptr,
            x_rows,
            in_features,
            stream=stream,
            runtime=runtime,
        )
        t_stage = _mark_sync_stage(
            runtime,
            stage_timings,
            sync_stages,
            f"{stage_prefix}_q8_quantize",
            t_stage,
        )
        fn = gguf_q5_k_x8_selected_q8_1_dp4a_gemv_bf16_bf16_out
        use_q8_1_input = True
    elif quant_key == "gguf_q6_k_x8_v1":
        if q8_1_workspace_ptr is None:
            raise ValueError("gguf_q6_k_x8_v1 selected GEMV requires q8_1 workspace")
        gguf_q4_k_quantize_bf16_q8_1(
            x_ptr,
            q8_1_workspace_ptr,
            x_rows,
            in_features,
            stream=stream,
            runtime=runtime,
        )
        t_stage = _mark_sync_stage(
            runtime,
            stage_timings,
            sync_stages,
            f"{stage_prefix}_q8_quantize",
            t_stage,
        )
        fn = gguf_q6_k_x8_selected_q8_1_dp4a_gemv_bf16_bf16_out
        use_q8_1_input = True
    elif (
        q8_1_workspace_ptr is not None
        and quant_key == "gguf_q5_k"
        and out_features % 8 == 0
        and _gguf_raw_selected_dp4a_enabled()
    ):
        gguf_q4_k_quantize_bf16_q8_1(
            x_ptr,
            q8_1_workspace_ptr,
            x_rows,
            in_features,
            stream=stream,
            runtime=runtime,
        )
        t_stage = _mark_sync_stage(
            runtime,
            stage_timings,
            sync_stages,
            f"{stage_prefix}_q8_quantize",
            t_stage,
        )
        fn = gguf_q5_k_selected_pack8_q8_1_dp4a_gemv_bf16_bf16_out
        use_q8_1_input = True
    elif (
        q8_1_workspace_ptr is not None
        and quant_key == "gguf_q6_k"
        and out_features % 8 == 0
        and _gguf_raw_selected_dp4a_enabled()
    ):
        gguf_q4_k_quantize_bf16_q8_1(
            x_ptr,
            q8_1_workspace_ptr,
            x_rows,
            in_features,
            stream=stream,
            runtime=runtime,
        )
        t_stage = _mark_sync_stage(
            runtime,
            stage_timings,
            sync_stages,
            f"{stage_prefix}_q8_quantize",
            t_stage,
        )
        fn = gguf_q6_k_selected_pack8_q8_1_dp4a_gemv_bf16_bf16_out
        use_q8_1_input = True
    elif quant_key == "gguf_q4_k":
        fn = gguf_q4_k_selected_gemv_bf16_bf16_out
    elif quant_key == "gguf_q5_k" and out_features % 8 == 0:
        fn = gguf_q5_k_selected_pack8_gemv_bf16_bf16_out
    elif quant_key == "gguf_q5_k":
        fn = gguf_q5_k_selected_gemv_bf16_bf16_out
    elif quant_key == "gguf_q6_k" and out_features % 8 == 0:
        fn = gguf_q6_k_selected_pack8_gemv_bf16_bf16_out
    elif quant_key == "gguf_q6_k":
        fn = gguf_q6_k_selected_gemv_bf16_bf16_out
    elif quant_key == "gguf_q4_k_t16_v1":
        fn = gguf_q4_k_t16_selected_gemv_bf16_bf16_out
    elif quant_key == "gguf_q5_k_t16_v1":
        fn = gguf_q5_k_t16_selected_gemv_bf16_bf16_out
    elif quant_key == "gguf_q6_k_t16_v1":
        fn = gguf_q6_k_t16_selected_gemv_bf16_bf16_out
    else:
        raise ValueError(f"unsupported selected GGUF MoE quant {quant_key!r} for {weight.spec.source.name}")
    fn(
        q8_1_workspace_ptr if use_q8_1_input else x_ptr,
        selected_ptr,
        weight.allocation(allocation).tensor.ptr,
        out_ptr,
        x_rows,
        rows,
        num_experts,
        in_features,
        out_features,
        stream=stream,
        runtime=runtime,
    )
    _mark_sync_stage(
        runtime,
        stage_timings,
        sync_stages,
        f"{stage_prefix}_gemv",
        t_stage,
    )


def _zero(runtime: HipRuntime, buffer, zeros: np.ndarray, *, stream: int = 0) -> None:
    if zeros.nbytes == buffer.nbytes and bool(np.all(zeros == 0)):
        if stream:
            runtime.memset_async(buffer.ptr, 0, buffer.nbytes, stream)
        else:
            runtime.memset(buffer.ptr, 0, buffer.nbytes)
        return
    copy_host_to_device(buffer, host_array_ptr(zeros), runtime=runtime)


def _rope_tables(*, max_positions: int, rotary_dim: int, base: float) -> tuple[np.ndarray, np.ndarray]:
    positions = np.arange(max_positions, dtype=np.float32)[:, None]
    dims = np.arange(rotary_dim // 2, dtype=np.float32)[None, :]
    inv_freq = np.power(np.float32(base), -2.0 * dims / np.float32(rotary_dim))
    freqs = positions * inv_freq
    cos_half = np.cos(freqs).astype(np.float32, copy=False)
    sin_half = np.sin(freqs).astype(np.float32, copy=False)
    cos = np.concatenate([cos_half, cos_half], axis=1).astype(np.float32, copy=False)
    sin = np.concatenate([sin_half, sin_half], axis=1).astype(np.float32, copy=False)
    return np.ascontiguousarray(cos), np.ascontiguousarray(sin)


__all__ = [
    "Qwen35GGUFFullAttentionPrefillResult",
    "Qwen35GGUFLinearAttentionBoundaryCapture",
    "Qwen35GGUFLinearAttentionLayerCapture",
    "Qwen35GGUFFullStackRunner",
    "Qwen35GGUFHiddenSeedContract",
    "Qwen35GGUFMTPDraftSeed",
    "Qwen35GGUFNextTokenProbeResult",
    "Qwen35GGUFOneLayerProbe",
    "Qwen35GGUFFastPathSafety",
    "Qwen35GGUFResidentSession",
    "qwen35_gguf_current_hidden_seed_contract",
    "qwen35_gguf_fp32_hidden_seed_contract",
    "qwen35_gguf_fp32_verify_hidden_seed_contract",
    "resolve_qwen35moe_fastpath_safety",
]
