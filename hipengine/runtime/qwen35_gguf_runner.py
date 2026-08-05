"""Qwen3.5 GGUF runtime bring-up probes."""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.hip import HipError, HipMemcpyKind, HipRuntime, get_hip_runtime
from hipengine.core.memory import (
    DeviceBuffer,
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.core.tensor import Tensor
from hipengine.runtime.gguf_packed_manifest import build_packed_decode_execution_manifest
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
    qwen35_paged_attn_decode_int8_block16_gqa_splitk_gate_bf16_spans,
    qwen35_paged_attn_decode_int8_hadamard_group32_gqa_splitk_gate_bf16_spans,
    qwen35_paged_attn_decode_int8_key_bf16_value_gqa_splitk_gate_bf16_spans,
    qwen35_paged_attn_decode_int8_gqa_splitk_gate_bf16_spans,
    qwen35_paged_full_attn_decode_context_bf16_batch_c1_exact_spans,
    qwen35_paged_full_attn_decode_context_bf16_spans,
    qwen35_paged_full_attn_decode_split_k_gate_bf16_spans,
    qwen35_paged_full_attn_decode_split_k_gqa_gate_bf16_batch_spans,
    qwen35_paged_full_attn_decode_split_k_gqa_gate_bf16_parallel_reduce_spans,
    qwen35_paged_full_attn_decode_split_k_gqa_gate_bf16_spans,
    qwen35_paged_full_attn_decode_split_k_warp_gate_bf16_spans,
    qwen35_paged_attn_prefill_int8_hadamard_group32_gqa_gate_fp16_spans,
    qwen35_paged_full_attn_prefill_gqa_gate_bf16_spans,
)
from hipengine.kernels.hip_gfx1100.attention.paged_kv_write import (
    build_qwen35_paged_kv_write,
    qwen35_write_paged_kv_int8_block16_prompt_spans,
    qwen35_write_paged_kv_int8_block16_spans,
    qwen35_write_paged_kv_int8_hadamard_group32_prompt_spans,
    qwen35_write_paged_kv_int8_hadamard_group32_spans,
    qwen35_write_paged_kv_int8_key_bf16_value_prompt_spans,
    qwen35_write_paged_kv_int8_key_bf16_value_spans,
    qwen35_write_paged_kv_int8_per_token_head_prompt_spans,
    qwen35_write_paged_kv_int8_per_token_head_spans,
    qwen35_write_paged_kv_mixed_value_bf16_prompt_spans,
    qwen35_write_paged_kv_mixed_value_bf16_spans,
)
from hipengine.kernels.hip_gfx1100.convert import bf16_to_f32, build_cast, f32_to_bf16
from hipengine.kernels.hip_gfx1100.fused import (
    gguf_add_rmsnorm_bf16_f32_weight,
    gguf_add_rmsnorm_f32_bf16_f32_weight,
    gguf_add_rmsnorm_f32_f32_f32_weight,
    gguf_bf16_add,
    gguf_f32_bf16_add_out_f32,
    gguf_qwen35_head_rmsnorm_partial_rotary_position_f32_weight,
    gguf_qwen35_head_rmsnorm_partial_rotary_positions_f32_weight,
    gguf_rmsnorm_bf16_f32_weight,
    gguf_rmsnorm_bf16_f32_weight_out_f32,
    gguf_rmsnorm_f32_f32_weight,
    gguf_rmsnorm_f32_f32_weight_out_f32,
    register_paro_combine_kernels,
    register_paro_silu_kernels,
    shared_gate_combine_residual_batch_out_bf16,
    shared_gate_combine_residual_out_bf16,
    shared_gate_combine_residual_rmsnorm_gguf_bf16_out,
    silu_mul_dual_out_bf16,
    silu_mul_separate_out_f32,
    silu_mul_separate_out_bf16,
    weighted_lanes_sum_out_bf16_f32w,
)
from hipengine.kernels.hip_gfx1100.fused.paro_combine import (
    weighted_sum_f32_shared_f32_gate_combine_residual_batch_out_f32_accum_f32w,
    weighted_sum_f32_shared_f32_gate_combine_residual_out_f32_accum_f32w,
    weighted_sum_f32_shared_gate_combine_residual_batch_out_f32_accum_f32w,
    weighted_sum_f32_shared_gate_combine_residual_out_f32_accum_f32w,
    weighted_sum_shared_gate_combine_residual_batch_out_bf16_f32w,
    weighted_sum_shared_gate_combine_residual_batch_out_f32_accum_f32w,
    weighted_sum_shared_gate_combine_residual_batch_out_f32_f32w,
    weighted_sum_shared_gate_combine_residual_out_bf16_f32w,
    weighted_sum_shared_gate_combine_residual_out_f32_accum_f32w,
    weighted_sum_shared_gate_combine_residual_out_f32_f32w,
)
from hipengine.kernels.hip_gfx1100.linear.dense_gemv import dense_gemv_out_bf16
from hipengine.kernels.hip_gfx1100.linear.lm_head import (
    argmax_f32,
    argmax_f32_rows_i32,
    build_lm_head,
    lm_head_argmax_stage1_blocks,
)
from hipengine.kernels.hip_gfx1100.sampling import build_sampler
from hipengine.kernels.hip_gfx1100.rotary.qwen35_rotary import qwen35_split_qgate_bf16
from hipengine.kernels.hip_gfx1100.runtime import (
    advance_decode_position_i64,
    build_runtime_state,
    copy_i32_to_i64,
    prepare_prefill_chunk_metadata,
    record_i64_scalar_indexed,
    set_decode_position_i64,
    set_i64_scalar,
)
from hipengine.kvcache import (
    DeviceChunkedKVPool,
    DeviceKVPoolAllocation,
    FixedPagedKVPolicy,
    KVLiveSpans,
    KVScaleMetadata,
)
from hipengine.runtime.native_sampler import NativeSamplerWorkspace
from hipengine.runtime.qwen35_paro import AotritonPrefillStreamBridge
from hipengine.kernels.hip_gfx1100.speculative import (
    build_dflash_commit,
    linear_state_pair_commit_chunked_i32,
    linear_state_pair_commit_i32,
)
from hipengine.kernels.hip_gfx1100.linear_attn.conv import (
    qwen35_linear_attn_chain_conv_decode_bf16_tloop,
    qwen35_linear_attn_chain_conv_decode_f32_tloop,
    qwen35_linear_attn_conv_decode_bf16,
    qwen35_linear_attn_conv_decode_indexed_bf16,
    qwen35_linear_attn_conv_prefill_f32,
    qwen35_linear_attn_conv_prefill_f32_state_rows,
    qwen35_linear_attn_conv_prefill_segments_f32,
    qwen35_linear_attn_conv_prefill_segments_f32_state_rows,
)
from hipengine.kernels.hip_gfx1100.linear_attn.gdn import (
    qwen35_gdn_chain_recurrent_rmsnorm_gate_lowp_c1_exact_tloop_bf16,
    qwen35_gdn_chain_recurrent_rmsnorm_gate_lowp_tloop_bf16,
    qwen35_gdn_prefill_recurrent_rmsnorm_gate_bf16_decode_order,
    qwen35_gdn_prefill_recurrent_rmsnorm_gate_bf16_decode_order_segments,
    qwen35_gdn_prefill_recurrent_rmsnorm_gate_bf16_decode_order_segments_state_rows_no_copy,
    qwen35_gdn_prefill_recurrent_rmsnorm_gate_bf16_decode_order_state_rows_no_copy,
    qwen35_gdn_recurrent_rmsnorm_gate_lowp_bf16,
    qwen35_gdn_recurrent_rmsnorm_gate_segments_lowp_bf16,
    register_qwen35_linear_attn_gdn_kernels,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_expert_pack8_gemv import (
    build_gguf_expert_pack8_gemv,
    register_gguf_expert_pack8_gemv_kernels,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_iq_selected_prefill import (
    register_gguf_iq_selected_prefill_kernels,
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
from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_q8_1_selected_prefill import (
    gguf_q8_1_mmq_ds4_pack_bf16,
    register_gguf_q4_k_q8_1_selected_prefill_kernels,
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
    gguf_q4_k_t16_selected_dual_pairreuse_gemv_bf16_bf16_out,
    gguf_q4_k_t16_selected_dual_q8_1_dp4a_gemv_bf16_bf16_out,
    gguf_q4_k_t16_selected_dual_silu_gemv_bf16_bf16_out,
    gguf_q4_k_t16_selected_gemv_bf16_bf16_out,
    gguf_q5_k_t16_selected_gemv_bf16_bf16_out,
    gguf_q5_k_t16_selected_qwen_tile8_gemv_bf16_bf16_out,
    gguf_q5_k_t16_selected_pairreuse_gemv_bf16_bf16_out,
    gguf_q5_k_t16_selected_q8_1_dp4a_gemv_bf16_bf16_out,
    gguf_q6_k_t16_selected_gemv_bf16_bf16_out,
    gguf_q6_k_t16_selected_pairreuse_gemv_bf16_bf16_out,
    register_gguf_t16_selected_gemv_kernels,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_x8_selected_gemv import (
    gguf_q4_k_x8_selected_dual_q8_1_dp4a_gemv_bf16_bf16_out,
    gguf_q5_k_x8_selected_q8_1_dp4a_gemv_bf16_bf16_out,
    gguf_q5_k_x8_selected_q8_1_dp4a_gemv_bf16_f32_out,
    gguf_q6_k_x8_selected_q8_1_dp4a_gemv_bf16_bf16_out,
    gguf_q6_k_x8_selected_q8_1_dp4a_gemv_bf16_f32_out,
    register_gguf_x8_selected_gemv_kernels,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_k_gemv import (
    gguf_q8_0_gemv_bf16_f32_out,
    gguf_q8_0_gemv_rowtile_bf16_f32_out,
    gguf_q5_k_selected_gemv_bf16_bf16_out,
    gguf_q5_k_selected_pack8_gemv_bf16_bf16_out,
    gguf_q5_k_selected_pack8_q8_1_dp4a_gemv_bf16_bf16_out,
    gguf_q6_k_selected_gemv_bf16_bf16_out,
    gguf_q6_k_selected_pack8_gemv_bf16_bf16_out,
    gguf_q6_k_selected_pack8_q8_1_dp4a_gemv_bf16_bf16_out,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q8_0_mmq_prefill import (
    build_gguf_q8_0_mmq_prefill,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_gemv import (
    gguf_q4_k_quantize_bf16_q8_1,
    gguf_q4_k_quantize_f32_q8_1,
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
    gguf_q8_0_dp4a_dual_split_rowtile4_gemv_f32_f32_out,
    gguf_q8_0_dp4a_rowtile4_gemv_f32_f32_out,
    gguf_q8_0_dp4a_rowtile4_gemv_bf16_bf16_out,
    gguf_q8_0_dp4a_triple_split_rowtile4_gemv_bf16_bf16_out,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_moe_ffn_fused import (
    gguf_q4_k_selected_ffn_fused_bf16_bf16_out,
)
from hipengine.kernels.registry import KernelKey, is_registered, resolve
from hipengine.kernels.backends import (
    backend_package_capability,
    hip_target_arch_environment,
    hip_target_arch_for_backend,
    load_backend_kernel_package,
    resolve_backend,
)
from hipengine.kernels.hip_gfx1100.moe.group_scatter import (
    qwen35_moe_group_count,
    qwen35_moe_group_prefix,
    qwen35_moe_group_scatter_gather_lowp,
    qwen35_moe_wmma_tile_map,
    register_qwen35_moe_group_scatter_kernels,
)
from hipengine.kernels.hip_gfx1100.moe.router import qwen35_router_select
from hipengine.loading.gguf import GGUFReader
from hipengine.loading.materialize import float_array_to_bf16_bits
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
from hipengine.quant.gguf import bf16_to_float32, dequantize_gguf_data
from hipengine.kernels.hip_gfx1100.quant.gguf_q8_0_t16_prefill import (
    q8_t16_two_wave_prefill_session,
)
from hipengine.runtime.gguf_embedding import launch_gguf_embedding
from hipengine.runtime.gguf_linear import (
    GGUF_ACTIVATION_BF16,
    GGUF_ACTIVATION_F32,
    GGUF_OUTPUT_F32,
    gemv_decode_session,
    gguf_gemv_decode_enabled,
    gguf_wmma_prefill_enabled,
    launch_gguf_linear,
    launch_gguf_linear_pair,
    launch_gguf_linear_pair_concat,
    launch_gguf_linear_triple,
    native_batch_decode_session,
    q8_mmq_prefill_session,
    q8_t16_pair_rowtile_min_rows_session,
    q8_t16_rowtile_all_session,
    resolve_gguf_linear_dispatch,
    resolve_q8_mmq_prefill_policy,
    wmma_prefill_session,
)
from hipengine.runtime.prefill import PrefillConfig, resolve_prefill_config_for_sequence
from hipengine.speculative import TargetVerifyBatch
from hipengine.runtime.prefill_flight_recorder import (
    FlightRecorderPhase,
    PrefillFlightRecorder,
)


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


class _HipEventStageRecorder:
    """HIP-event stage timing for queued work without per-stage synchronizes."""

    def __init__(self, runtime: HipRuntime, *, enabled: bool, stream: int = 0) -> None:
        self.runtime = runtime
        self.enabled = bool(enabled)
        self.stream = int(stream)
        self._last_event: int | None = None
        self._events: list[int] = []
        self._intervals: list[tuple[tuple[str, ...], int, int]] = []

    def start(self) -> None:
        if not self.enabled:
            return
        self._last_event = self._new_event()
        self.runtime.event_record(self._last_event, self.stream)

    def mark(self, name: str, *aliases: str) -> None:
        if not self.enabled:
            return
        if self._last_event is None:
            self.start()
        assert self._last_event is not None
        event = self._new_event()
        self.runtime.event_record(event, self.stream)
        names = (str(name), *(str(alias) for alias in aliases))
        self._intervals.append((names, self._last_event, event))
        self._last_event = event

    def resolve_into(self, timings: dict[str, float] | None) -> None:
        if not self.enabled:
            return
        try:
            for names, start, stop in self._intervals:
                self.runtime.event_synchronize(stop)
                ms = self.runtime.event_elapsed_time_ms(start, stop)
                if ms < 0.0:
                    _add_sync_stage_timing(timings, f"{names[0]}_negative_event_elapsed", -ms)
                    _add_sync_stage_timing(timings, "packed_verify_gpu_negative_event_elapsed", -ms)
                    ms = 0.0
                for name in names:
                    _add_sync_stage_timing(timings, name, ms)
        finally:
            self.close()

    def close(self) -> None:
        while self._events:
            self.runtime.event_destroy(self._events.pop())
        self._last_event = None
        self._intervals.clear()

    def _new_event(self) -> int:
        event = self.runtime.event_create()
        self._events.append(event)
        return event


@dataclass(frozen=True)
class Qwen35GGUFResidentTargetLayout:
    """Batch-shaped resident storage contract for GGUF target rows."""

    max_batch_size: int
    hidden_size: int
    vocab_size: int
    max_sequence_length: int
    block_size: int = 256

    def __post_init__(self) -> None:
        for name in ("max_batch_size", "hidden_size", "vocab_size", "max_sequence_length", "block_size"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")

    @property
    def blocks_per_slot(self) -> int:
        return (int(self.max_sequence_length) + int(self.block_size) - 1) // int(self.block_size)

    @property
    def token_shape(self) -> tuple[int, ...]:
        return (int(self.max_batch_size),)

    @property
    def position_shape(self) -> tuple[int, ...]:
        return self.token_shape

    @property
    def hidden_shape(self) -> tuple[int, ...]:
        return (int(self.max_batch_size), int(self.hidden_size))

    @property
    def slot0_hidden_shape(self) -> tuple[int, ...]:
        return (1, int(self.hidden_size))

    @property
    def logits_shape(self) -> tuple[int, ...]:
        return (int(self.max_batch_size), int(self.vocab_size))

    @property
    def block_table_shape(self) -> tuple[int, ...]:
        return (int(self.max_batch_size), self.blocks_per_slot)


@dataclass(frozen=True)
class Qwen35GGUFTargetRowsResult:
    """Host-visible outputs from the correctness-first GGUF target executor."""

    token_ids: tuple[int, ...]
    positions: tuple[int, ...]
    slot_indices: tuple[int, ...]
    span_role: str
    logits: np.ndarray
    layer_hidden_bits: dict[int, np.ndarray]
    execution_paths: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        rows = len(self.token_ids)
        if rows <= 0:
            raise ValueError("target-row result must contain at least one row")
        if len(self.positions) != rows or len(self.slot_indices) != rows:
            raise ValueError("target-row result metadata must align")
        if self.span_role not in {"decode", "verify_chain", "verify_tree"}:
            raise ValueError("target-row result span_role is invalid")
        if self.logits.ndim != 2 or self.logits.shape[0] != rows:
            raise ValueError("target-row logits must be rank-2 with one row per result")
        for hidden in self.layer_hidden_bits.values():
            if hidden.ndim != 2 or hidden.shape[0] != rows or hidden.dtype != np.uint16:
                raise ValueError("target-row layer captures must be row-aligned uint16 arrays")


@dataclass(frozen=True)
class Qwen35GGUFNextTokenProbeResult:
    token_id: int
    logit: float
    logits: np.ndarray


@dataclass(frozen=True)
class Qwen35GGUFPackedPrefillResult:
    """Host-visible result for one slot in a packed prompt prefill pass."""

    input_token_ids: list[int]
    token_id: int
    hidden_seeds: np.ndarray
    start_position: int

    def __post_init__(self) -> None:
        if self.start_position < 0:
            raise ValueError("start_position must be non-negative")
        if len(self.input_token_ids) == 0:
            raise ValueError("input_token_ids must be non-empty")
        if self.hidden_seeds.shape[0] != len(self.input_token_ids):
            raise ValueError("hidden_seeds rows must match input_token_ids length")
        if self.hidden_seeds.dtype != np.float32:
            raise ValueError("hidden_seeds must be float32")


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
    source_buffer: str = "Qwen35GGUFResidentSession._verify_hidden_seed_buf",
) -> Qwen35GGUFHiddenSeedContract:
    """Describe an fp32 verifier-row hidden-seed buffer."""

    return Qwen35GGUFHiddenSeedContract(
        provenance="post_output_norm",
        dtype=DType.FP32,
        rows=int(rows),
        hidden_size=int(hidden_size),
        source_buffer=str(source_buffer),
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
    pre_output_norm_hidden: np.ndarray | None = None
    layer_output_hidden: dict[int, np.ndarray] | None = None
    layer_boundary_hidden: dict[int, dict[str, np.ndarray]] | None = None
    lm_head_logits_f32: np.ndarray | None = None
    linear_state_rows_captured: bool = False
    final_linear_state_committed: bool = False
    deferred_packed_state: object | None = None

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
        if self.pre_output_norm_hidden is not None:
            if self.pre_output_norm_hidden.shape != self.hidden_seeds.shape:
                raise ValueError("pre_output_norm_hidden shape must match hidden_seeds")
            if self.pre_output_norm_hidden.dtype != np.float32:
                raise ValueError("pre_output_norm_hidden must be float32")
        if self.layer_output_hidden is not None:
            for layer_id, hidden in self.layer_output_hidden.items():
                if int(layer_id) < 0:
                    raise ValueError("layer_output_hidden keys must be non-negative layer IDs")
                if hidden.shape != self.hidden_seeds.shape:
                    raise ValueError("layer_output_hidden rows must match hidden_seeds")
                if hidden.dtype != np.float32:
                    raise ValueError("layer_output_hidden values must be float32")
        if self.layer_boundary_hidden is not None:
            for layer_id, arrays in self.layer_boundary_hidden.items():
                if int(layer_id) < 0:
                    raise ValueError("layer_boundary_hidden keys must be non-negative layer IDs")
                if not isinstance(arrays, dict):
                    raise ValueError("layer_boundary_hidden values must be dictionaries")
                for name, array in arrays.items():
                    if array.shape[0] != len(self.input_token_ids):
                        raise ValueError(
                            f"layer_boundary_hidden[{layer_id!r}][{name!r}] rows must match input_token_ids"
                        )
                    if array.dtype not in (np.float32, np.int64):
                        raise ValueError(
                            f"layer_boundary_hidden[{layer_id!r}][{name!r}] must be float32 or int64"
                        )
        if self.lm_head_logits_f32 is not None:
            if self.lm_head_logits_f32.ndim != 2:
                raise ValueError("lm_head_logits_f32 must be rank-2")
            if self.lm_head_logits_f32.shape[0] != len(self.input_token_ids):
                raise ValueError("lm_head_logits_f32 rows must match input_token_ids length")
            if self.lm_head_logits_f32.dtype != np.float32:
                raise ValueError("lm_head_logits_f32 must be float32")


@dataclass(frozen=True)
class _GGUFPackedVerifySlotBlock:
    """One speculative target-verifier block for a packed multi-slot pass."""

    input_token_ids: tuple[int, ...]
    start_position: int
    active: bool = True

    def __post_init__(self) -> None:
        if not self.input_token_ids:
            raise ValueError("packed verify slot block input_token_ids must be non-empty")
        if self.active and int(self.start_position) < 0:
            raise ValueError("packed verify slot block start_position must be non-negative")
        if not self.active and (
            int(self.start_position) != -1 or len(self.input_token_ids) != 1
        ):
            raise ValueError("inactive packed verify slots require one dummy row at position -1")


@dataclass(frozen=True)
class _GGUFPackedVerifyLayout:
    """CPU-side row/state layout for a future packed multi-slot target verifier."""

    input_token_ids: np.ndarray
    row_slot_indices: np.ndarray
    row_positions: np.ndarray
    row_offsets_in_slot: np.ndarray
    live_counts: np.ndarray
    block_table: np.ndarray
    cu_seqlens: np.ndarray
    state_indices: np.ndarray
    active_mask: np.ndarray
    blocks_per_slot: int
    block_size: int
    max_live_count: int
    total_physical_positions: int

    @property
    def rows(self) -> int:
        return int(self.input_token_ids.shape[0])

    @property
    def slot_count(self) -> int:
        return int(self.state_indices.shape[0])


@dataclass(frozen=True)
class _GGUFPackedARPrefillLinearStatePlan:
    """Linear-state storage contract for an all-accepted packed AR prompt."""

    route: str
    state_slots: int
    transient_state_rows: int
    capture_token_state_rows: bool
    commit_captured_state_rows: bool


@dataclass(frozen=True)
class _GGUFPackedARPrefillChunk:
    """One row-bounded packed-prefill round over every still-active slot."""

    slot_indices: tuple[int, ...]
    start_offsets: tuple[int, ...]
    prompt_token_ids: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        slot_count = len(self.slot_indices)
        if slot_count <= 0:
            raise ValueError("packed AR prefill chunk must contain at least one slot")
        if len(self.start_offsets) != slot_count or len(self.prompt_token_ids) != slot_count:
            raise ValueError("packed AR prefill chunk metadata must match its slot count")
        if len(set(int(index) for index in self.slot_indices)) != slot_count:
            raise ValueError("packed AR prefill chunk slot indices must be unique")
        if min(int(index) for index in self.slot_indices) < 0:
            raise ValueError("packed AR prefill chunk slot indices must be non-negative")
        if min(int(offset) for offset in self.start_offsets) < 0:
            raise ValueError("packed AR prefill chunk offsets must be non-negative")
        if any(not tokens for tokens in self.prompt_token_ids):
            raise ValueError("packed AR prefill chunk token segments must be non-empty")

    @property
    def rows(self) -> int:
        return sum(len(tokens) for tokens in self.prompt_token_ids)


def _plan_packed_ar_prefill_chunks(
    prompt_token_ids: tuple[tuple[int, ...], ...],
    *,
    row_capacity: int,
) -> tuple[_GGUFPackedARPrefillChunk, ...]:
    """Plan slot-fair packed rounds without exceeding the resident row slab.

    Every unfinished slot appears in each round.  The planner therefore fails
    closed instead of silently processing only a subset when the slab cannot
    represent at least one row for every active slot.
    """

    prompts = tuple(tuple(int(token) for token in prompt) for prompt in prompt_token_ids)
    if not prompts:
        raise ValueError("packed AR prefill chunk plan requires at least one prompt")
    if any(not prompt for prompt in prompts):
        raise ValueError("packed AR prefill chunk plan requires non-empty prompts")
    capacity = int(row_capacity)
    if capacity <= 0:
        raise ValueError("packed AR prefill row capacity must be positive")
    total_rows = sum(len(prompt) for prompt in prompts)
    if total_rows <= capacity:
        return (
            _GGUFPackedARPrefillChunk(
                slot_indices=tuple(range(len(prompts))),
                start_offsets=tuple(0 for _ in prompts),
                prompt_token_ids=prompts,
            ),
        )

    cursors = [0 for _ in prompts]
    chunks: list[_GGUFPackedARPrefillChunk] = []
    while any(cursor < len(prompt) for cursor, prompt in zip(cursors, prompts, strict=True)):
        active = tuple(
            slot_index
            for slot_index, (cursor, prompt) in enumerate(zip(cursors, prompts, strict=True))
            if cursor < len(prompt)
        )
        if capacity < len(active):
            raise ValueError(
                f"packed AR prefill row capacity {capacity} cannot represent all {len(active)} active slots"
            )
        rows_per_slot = max(1, capacity // len(active))
        starts: list[int] = []
        segments: list[tuple[int, ...]] = []
        for slot_index in active:
            start = int(cursors[slot_index])
            end = min(len(prompts[slot_index]), start + rows_per_slot)
            starts.append(start)
            segments.append(prompts[slot_index][start:end])
            cursors[slot_index] = end
        chunks.append(
            _GGUFPackedARPrefillChunk(
                slot_indices=active,
                start_offsets=tuple(starts),
                prompt_token_ids=tuple(segments),
            )
        )
    return tuple(chunks)


def _packed_ar_prefill_linear_state_plan(
    layout: _GGUFPackedVerifyLayout,
) -> _GGUFPackedARPrefillLinearStatePlan:
    """Keep one in-place final Conv/GDN state per slot, never per prompt row.

    The speculative verifier needs every candidate row for later accept-row
    selection. AR prompt prefill accepts every row, so its segmented Conv/GDN
    kernels can update the packed per-slot state directly. Materializing
    ``layout.rows`` recurrent snapshots would scale as O(tokens * layers) and
    exhaust VRAM for the B2 ragged 512/64/64/64 gate.
    """

    return _GGUFPackedARPrefillLinearStatePlan(
        route="segmented_in_place_final_state",
        state_slots=int(layout.slot_count),
        transient_state_rows=0,
        capture_token_state_rows=False,
        commit_captured_state_rows=False,
    )


def _scatter_packed_layer_output_hidden(
    sessions: tuple[object, ...],
    *,
    layer_id: int,
    hidden_rows: np.ndarray,
    row_indices: tuple[int, ...] | None = None,
) -> None:
    """Expose selected packed BF16 layer rows through each session's c1 tap."""

    rows = np.asarray(hidden_rows)
    if rows.ndim != 2 or rows.dtype != np.float32:
        raise ValueError("packed layer-output hidden rows must be rank-2 float32")
    selected = tuple(range(len(sessions))) if row_indices is None else tuple(row_indices)
    if len(selected) != len(sessions):
        raise ValueError("packed layer-output row indices must match session count")
    for session, row_index in zip(sessions, selected, strict=True):
        if row_index < 0 or row_index >= int(rows.shape[0]):
            raise ValueError(f"packed layer-output row index {row_index} is out of range")
        captured = getattr(session, "_last_layer_output_hidden", None)
        if not isinstance(captured, dict):
            raise RuntimeError("packed layer-output session capture map is unavailable")
        captured[int(layer_id)] = np.ascontiguousarray(
            rows[row_index : row_index + 1],
            dtype=np.float32,
        )


@dataclass(frozen=True)
class _GGUFPackedVerifyDeferredState:
    """Owner-side packed verifier state kept live until accept-row commit."""

    owner: object
    packed_state: object
    slot_index: int
    row_start: int
    row_end: int
    start_position: int
    end_position: int


def _build_gguf_packed_verify_layout(
    slot_blocks: list[_GGUFPackedVerifySlotBlock] | tuple[_GGUFPackedVerifySlotBlock, ...],
    *,
    block_size: int = 256,
    slot_capacity: int | None = None,
) -> _GGUFPackedVerifyLayout:
    """Build per-row KVLiveSpans/linear-state metadata for packed slot verify.

    Rows are packed slot-major. Each slot gets a disjoint physical paged-KV
    block range while keeping local per-slot RoPE positions and live counts.
    Linear-attention segment metadata mirrors the same slot-major packing:
    ``cu_seqlens`` defines each slot's row range and ``state_indices`` maps that
    segment to the slot's Conv/GDN recurrent-state slab.
    """

    blocks = tuple(slot_blocks)
    if not blocks:
        raise ValueError("packed verify layout requires at least one slot block")
    if not any(block.active for block in blocks):
        raise ValueError("packed verify layout requires at least one active slot")
    block_size = int(block_size)
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    row_count = sum(len(block.input_token_ids) for block in blocks)
    max_live_count = max(
        int(block.start_position) + len(block.input_token_ids)
        for block in blocks
        if block.active
    )
    slot_capacity = max_live_count if slot_capacity is None else int(slot_capacity)
    if slot_capacity < max_live_count:
        raise ValueError(
            f"slot_capacity {slot_capacity} must be >= max_live_count {max_live_count}"
        )
    blocks_per_slot = max(1, (slot_capacity + block_size - 1) // block_size)
    input_token_ids = np.empty((row_count,), dtype=np.int64)
    row_slot_indices = np.empty((row_count,), dtype=np.int32)
    row_positions = np.empty((row_count,), dtype=np.int64)
    row_offsets_in_slot = np.empty((row_count,), dtype=np.int32)
    live_counts = np.empty((row_count,), dtype=np.int64)
    block_table = np.empty((row_count, blocks_per_slot), dtype=np.int32)
    cu_seqlens = np.empty((len(blocks) + 1,), dtype=np.int32)
    state_indices = np.arange(len(blocks), dtype=np.int64)
    active_mask = np.asarray([bool(block.active) for block in blocks], dtype=np.bool_)

    row_cursor = 0
    cu_seqlens[0] = 0
    for slot_index, block in enumerate(blocks):
        slot_rows = len(block.input_token_ids)
        slot_start = int(block.start_position)
        slot_block_base = slot_index * blocks_per_slot
        slot_block_table = np.arange(
            slot_block_base,
            slot_block_base + blocks_per_slot,
            dtype=np.int32,
        )
        for row_offset, token_id in enumerate(block.input_token_ids):
            row = row_cursor + row_offset
            position = slot_start + row_offset if block.active else -1
            input_token_ids[row] = int(token_id)
            row_slot_indices[row] = int(slot_index)
            row_positions[row] = int(position)
            row_offsets_in_slot[row] = int(row_offset)
            live_counts[row] = int(position) + 1
            block_table[row, :] = slot_block_table if block.active else -1
        row_cursor += slot_rows
        cu_seqlens[slot_index + 1] = row_cursor

    return _GGUFPackedVerifyLayout(
        input_token_ids=input_token_ids,
        row_slot_indices=row_slot_indices,
        row_positions=row_positions,
        row_offsets_in_slot=row_offsets_in_slot,
        live_counts=live_counts,
        block_table=block_table,
        cu_seqlens=cu_seqlens,
        state_indices=state_indices,
        active_mask=active_mask,
        blocks_per_slot=blocks_per_slot,
        block_size=block_size,
        max_live_count=max_live_count,
        total_physical_positions=len(blocks) * blocks_per_slot * block_size,
    )


def _packed_ar_slot_capacity(
    max_live_count: int,
    *,
    block_size: int = 256,
    minimum: int = 1024,
) -> int:
    """Round packed-AR capacity so deferred state survives ordinary token steps."""

    max_live_count = int(max_live_count)
    block_size = int(block_size)
    minimum = int(minimum)
    if max_live_count <= 0:
        raise ValueError("max_live_count must be positive")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if minimum <= 0:
        raise ValueError("minimum must be positive")
    rounded = ((max_live_count + block_size - 1) // block_size) * block_size
    return max(minimum, rounded)


def _packed_decode_metadata_device_eligible(
    layout: _GGUFPackedVerifyLayout,
) -> bool:
    """Return whether the c4 device kernel can rebuild this layout exactly."""

    rows = int(layout.rows)
    if (
        rows <= 0
        or rows > 4
        or int(layout.slot_count) != rows
        or not bool(np.all(layout.active_mask))
    ):
        return False
    expected_rows = np.arange(rows, dtype=np.int64)
    expected_blocks = np.arange(
        rows * int(layout.blocks_per_slot),
        dtype=np.int32,
    ).reshape(rows, int(layout.blocks_per_slot))
    return bool(
        np.array_equal(layout.row_slot_indices, expected_rows.astype(np.int32))
        and np.array_equal(layout.row_offsets_in_slot, np.zeros(rows, dtype=np.int32))
        and np.array_equal(layout.live_counts, layout.row_positions + np.int64(1))
        and np.array_equal(layout.block_table, expected_blocks)
        and np.array_equal(layout.cu_seqlens, np.arange(rows + 1, dtype=np.int32))
        and np.array_equal(layout.state_indices, expected_rows)
    )


def _packed_prefill_requires_slot_local_full_attention(
    layout: _GGUFPackedVerifyLayout,
    *,
    aotriton_threshold: int | None = None,
) -> bool:
    """Return whether a packed slab contains a c1 AOTriton-sized slot."""

    threshold = (
        int(PrefillConfig().attn_aotriton_min_tokens)
        if aotriton_threshold is None
        else int(aotriton_threshold)
    )
    return bool(
        threshold > 0
        and any(
            int(layout.cu_seqlens[index + 1]) - int(layout.cu_seqlens[index])
            >= threshold
            for index in range(int(layout.slot_count))
        )
    )


def _validate_packed_ar_prefill_context(
    layout: _GGUFPackedVerifyLayout,
    *,
    slot_local_full_prefill: bool,
) -> None:
    """Keep long contexts on the per-session full-attention cache path."""

    if int(layout.max_live_count) >= 1024 and not slot_local_full_prefill:
        raise NotImplementedError(
            "packed paged AR prefill currently requires context < 1024"
        )


@dataclass(frozen=True)
class _GGUFPackedARAttentionWorkspace:
    """Split-K intermediates sized for one physical packed-AR decode width."""

    rows: int
    chunk_size: int
    num_splits: int
    partial_out: object
    partial_m: object
    partial_l: object
    buffers: tuple[object, ...]

    @classmethod
    def allocate(
        cls,
        runner: "Qwen35GGUFFullStackRunner",
        *,
        rows: int,
        max_context_len: int,
        runtime: HipRuntime,
        chunk_size: int = 256,
    ) -> "_GGUFPackedARAttentionWorkspace":
        rows = int(rows)
        max_context_len = int(max_context_len)
        chunk_size = int(chunk_size)
        if rows <= 0:
            raise ValueError("packed AR attention rows must be positive")
        if max_context_len <= 0:
            raise ValueError("packed AR attention max_context_len must be positive")
        if chunk_size <= 0:
            raise ValueError("packed AR attention chunk_size must be positive")
        if runner.weights is None:
            raise RuntimeError("GGUF packed AR attention requires materialized weights")
        num_splits = (max_context_len + chunk_size - 1) // chunk_size
        num_q_heads = int(runner.weights.config.head_count)
        partial_out = malloc(
            rows * int(runner.q_width) * num_splits * DType.FP32.itemsize,
            runtime=runtime,
        )
        partial_m = malloc(
            rows * num_q_heads * num_splits * DType.FP32.itemsize,
            runtime=runtime,
        )
        partial_l = malloc(
            rows * num_q_heads * num_splits * DType.FP32.itemsize,
            runtime=runtime,
        )
        return cls(
            rows=rows,
            chunk_size=chunk_size,
            num_splits=num_splits,
            partial_out=partial_out,
            partial_m=partial_m,
            partial_l=partial_l,
            buffers=(partial_out, partial_m, partial_l),
        )


@dataclass(frozen=True)
class _GGUFPackedTargetState:
    """Per-slot recurrent state plus policy-shaped packed KV backing."""

    slot_count: int
    max_sequence_length: int
    block_size: int
    blocks_per_slot: int
    total_positions: int
    kv_layout: Qwen35GGUFKVChunkLayout
    layer_conv_states: tuple[object | None, ...]
    layer_recurrent_states: tuple[object | None, ...]
    full_key_caches: tuple[object | None, ...]
    full_value_caches: tuple[object | None, ...]
    full_bf16_mirror_key_caches: tuple[object | None, ...]
    full_bf16_mirror_value_caches: tuple[object | None, ...]
    full_k_scale_caches: tuple[object | None, ...]
    full_v_scale_caches: tuple[object | None, ...]
    full_kv_scale_metadata: tuple[KVScaleMetadata | None, ...]
    buffers: tuple[object, ...]

    @classmethod
    def allocate(
        cls,
        runner: Qwen35GGUFFullStackRunner,
        *,
        slot_count: int,
        max_sequence_length: int,
        runtime: HipRuntime,
        block_size: int = 256,
        kv_layout: Qwen35GGUFKVChunkLayout | None = None,
    ) -> "_GGUFPackedTargetState":
        slot_count = int(slot_count)
        max_sequence_length = int(max_sequence_length)
        block_size = int(block_size)
        if slot_count <= 0:
            raise ValueError("slot_count must be positive")
        if max_sequence_length <= 0:
            raise ValueError("max_sequence_length must be positive")
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        if block_size != 256:
            raise ValueError("packed GGUF KV state currently requires block_size 256")
        assert runner.weights is not None
        cfg = runner.weights.config
        if kv_layout is None:
            kv_layout = Qwen35GGUFKVChunkLayout(
                storage_dtype=DType.BF16,
                storage_layout="uniform",
                scale_dtype=DType.FP16,
                scale_granularity="per_token_head",
                int8_kv_value_bf16=False,
                layer_storage_dtypes=tuple(
                    None if layer_type == LINEAR_ATTENTION else DType.BF16
                    for layer_type in cfg.layer_types
                ),
            )
        blocks_per_slot = (max_sequence_length + block_size - 1) // block_size
        total_pages = slot_count * blocks_per_slot
        total_positions = total_pages * block_size

        def buf(nbytes: int):
            return malloc(nbytes, runtime=runtime)

        conv_state_nbytes = (
            slot_count
            * int(runner.linear_qkv_width)
            * int(cfg.ssm_conv_kernel)
            * DType.FP32.itemsize
        )
        recurrent_state_nbytes = (
            slot_count
            * int(cfg.ssm_time_step_rank)
            * int(cfg.ssm_state_size)
            * int(runner.ssm_value_dim)
            * DType.FP32.itemsize
        )
        layer_conv_states: list[object | None] = []
        layer_recurrent_states: list[object | None] = []
        state_buffers: list[object] = []
        try:
            for layer_type in cfg.layer_types:
                if layer_type == LINEAR_ATTENTION:
                    conv_state = buf(conv_state_nbytes)
                    recurrent_state = buf(recurrent_state_nbytes)
                    memset = getattr(runtime, "memset", None)
                    if callable(memset):
                        memset(conv_state.ptr, 0, conv_state.nbytes)
                        memset(recurrent_state.ptr, 0, recurrent_state.nbytes)
                    state_buffers.extend((conv_state, recurrent_state))
                    layer_conv_states.append(conv_state)
                    layer_recurrent_states.append(recurrent_state)
                elif layer_type == FULL_ATTENTION:
                    layer_conv_states.append(None)
                    layer_recurrent_states.append(None)
                else:
                    raise ValueError(f"unsupported GGUF layer type {layer_type!r}")
            kv_backing = _allocate_qwen35_gguf_kv_chunk(
                runner,
                runtime=runtime,
                start_block_id=0,
                pages=total_pages,
                layout=kv_layout,
            )
        except Exception:
            for buffer in reversed(state_buffers):
                free(buffer, runtime=runtime)
            raise
        return cls(
            slot_count=slot_count,
            max_sequence_length=max_sequence_length,
            block_size=block_size,
            blocks_per_slot=blocks_per_slot,
            total_positions=total_positions,
            kv_layout=kv_layout,
            layer_conv_states=tuple(layer_conv_states),
            layer_recurrent_states=tuple(layer_recurrent_states),
            full_key_caches=kv_backing.full_key_caches,
            full_value_caches=kv_backing.full_value_caches,
            full_bf16_mirror_key_caches=kv_backing.full_bf16_mirror_key_caches,
            full_bf16_mirror_value_caches=kv_backing.full_bf16_mirror_value_caches,
            full_k_scale_caches=kv_backing.full_k_scale_caches,
            full_v_scale_caches=kv_backing.full_v_scale_caches,
            full_kv_scale_metadata=kv_backing.full_kv_scale_metadata,
            buffers=(*tuple(state_buffers), *kv_backing.buffers),
        )

    def linear_state_pair(self, layer_id: int) -> tuple[object, object]:
        conv_state = self.layer_conv_states[int(layer_id)]
        recurrent_state = self.layer_recurrent_states[int(layer_id)]
        if conv_state is None or recurrent_state is None:
            raise ValueError(f"layer {layer_id} has no packed linear-attention state")
        return conv_state, recurrent_state

    def full_cache(self, layer_id: int) -> tuple[object, object]:
        key_cache = self.full_key_caches[int(layer_id)]
        value_cache = self.full_value_caches[int(layer_id)]
        if key_cache is None or value_cache is None:
            raise ValueError(f"layer {layer_id} has no packed full-attention KV cache")
        return key_cache, value_cache

    def full_bf16_mirror_cache(self, layer_id: int) -> tuple[object, object] | None:
        key_cache = self.full_bf16_mirror_key_caches[int(layer_id)]
        value_cache = self.full_bf16_mirror_value_caches[int(layer_id)]
        if key_cache is None or value_cache is None:
            return None
        return key_cache, value_cache

    def full_scale_metadata(self, layer_id: int) -> KVScaleMetadata | None:
        return self.full_kv_scale_metadata[int(layer_id)]


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
    attn_norm_f32: np.ndarray
    attn_out_f32: np.ndarray
    post_norm_f32: np.ndarray
    residual_f32: np.ndarray
    ffn_or_moe_down_f32: np.ndarray
    layer_out_f32: np.ndarray
    moe_router_logits_f32: np.ndarray | None = None
    moe_selected_intermediate_f32: np.ndarray | None = None
    moe_shared_intermediate_f32: np.ndarray | None = None
    moe_shared_out_f32: np.ndarray | None = None
    moe_routing_weights_f32: np.ndarray | None = None
    moe_shared_gate_f32: np.ndarray | None = None
    moe_selected_experts_i64: np.ndarray | None = None
    linear_qkv_f32: np.ndarray | None = None
    linear_z_f32: np.ndarray | None = None
    ssm_alpha_f32: np.ndarray | None = None
    ssm_beta_f32: np.ndarray | None = None
    conv_out_f32: np.ndarray | None = None
    recurrent_out_f32: np.ndarray | None = None
    recurrent_bf16_f32: np.ndarray | None = None
    post_norm_source: str = "bf16_scratch.post_norm"

    def as_summary_dict(self) -> dict[str, object]:
        optional_finite = True
        optional_arrays = (
            self.moe_router_logits_f32,
            self.moe_selected_intermediate_f32,
            self.moe_shared_intermediate_f32,
            self.moe_shared_out_f32,
            self.moe_routing_weights_f32,
            self.moe_shared_gate_f32,
            self.linear_qkv_f32,
            self.linear_z_f32,
            self.ssm_alpha_f32,
            self.ssm_beta_f32,
            self.conv_out_f32,
            self.recurrent_out_f32,
            self.recurrent_bf16_f32,
        )
        for array in optional_arrays:
            if array is not None:
                optional_finite = optional_finite and bool(np.all(np.isfinite(array)))
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
            "attn_norm_shape": list(self.attn_norm_f32.shape),
            "attn_out_shape": list(self.attn_out_f32.shape),
            "post_norm_shape": list(self.post_norm_f32.shape),
            "post_norm_source": str(self.post_norm_source),
            "residual_shape": list(self.residual_f32.shape),
            "ffn_or_moe_down_shape": list(self.ffn_or_moe_down_f32.shape),
            "moe_router_logits_shape": (
                None
                if self.moe_router_logits_f32 is None
                else list(self.moe_router_logits_f32.shape)
            ),
            "moe_selected_intermediate_shape": (
                None
                if self.moe_selected_intermediate_f32 is None
                else list(self.moe_selected_intermediate_f32.shape)
            ),
            "moe_shared_intermediate_shape": (
                None
                if self.moe_shared_intermediate_f32 is None
                else list(self.moe_shared_intermediate_f32.shape)
            ),
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
            "linear_qkv_shape": (
                None if self.linear_qkv_f32 is None else list(self.linear_qkv_f32.shape)
            ),
            "linear_z_shape": (
                None if self.linear_z_f32 is None else list(self.linear_z_f32.shape)
            ),
            "ssm_alpha_shape": (
                None if self.ssm_alpha_f32 is None else list(self.ssm_alpha_f32.shape)
            ),
            "ssm_beta_shape": (
                None if self.ssm_beta_f32 is None else list(self.ssm_beta_f32.shape)
            ),
            "conv_out_shape": (
                None if self.conv_out_f32 is None else list(self.conv_out_f32.shape)
            ),
            "recurrent_out_shape": (
                None
                if self.recurrent_out_f32 is None
                else list(self.recurrent_out_f32.shape)
            ),
            "recurrent_bf16_shape": (
                None
                if self.recurrent_bf16_f32 is None
                else list(self.recurrent_bf16_f32.shape)
            ),
            "layer_out_shape": list(self.layer_out_f32.shape),
            "finite": bool(
                np.all(np.isfinite(self.hidden_in_f32))
                and np.all(np.isfinite(self.attn_norm_f32))
                and np.all(np.isfinite(self.attn_out_f32))
                and np.all(np.isfinite(self.post_norm_f32))
                and np.all(np.isfinite(self.residual_f32))
                and np.all(np.isfinite(self.ffn_or_moe_down_f32))
                and np.all(np.isfinite(self.layer_out_f32))
                and optional_finite
            ),
        }


@dataclass(frozen=True)
class Qwen35GGUFRouterTraceLayerCapture:
    """Host-visible router snapshot for one decoded target layer."""

    layer_id: int
    layer_type: str
    token_id: int
    position: int
    hidden_size: int
    expert_count: int
    top_k: int
    hidden_in_f32: np.ndarray
    layer_out_f32: np.ndarray
    moe_router_logits_f32: np.ndarray
    moe_routing_weights_f32: np.ndarray
    moe_shared_gate_f32: np.ndarray
    moe_selected_experts_i64: np.ndarray

    def as_summary_dict(self) -> dict[str, object]:
        return {
            "layer_id": int(self.layer_id),
            "layer_type": str(self.layer_type),
            "token_id": int(self.token_id),
            "position": int(self.position),
            "hidden_size": int(self.hidden_size),
            "expert_count": int(self.expert_count),
            "top_k": int(self.top_k),
            "hidden_in_shape": list(self.hidden_in_f32.shape),
            "layer_out_shape": list(self.layer_out_f32.shape),
            "moe_router_logits_shape": list(self.moe_router_logits_f32.shape),
            "moe_routing_weights_shape": list(self.moe_routing_weights_f32.shape),
            "moe_shared_gate_shape": list(self.moe_shared_gate_f32.shape),
            "moe_selected_experts_shape": list(self.moe_selected_experts_i64.shape),
            "finite": bool(
                np.all(np.isfinite(self.hidden_in_f32))
                and np.all(np.isfinite(self.layer_out_f32))
                and np.all(np.isfinite(self.moe_router_logits_f32))
                and np.all(np.isfinite(self.moe_routing_weights_f32))
                and np.all(np.isfinite(self.moe_shared_gate_f32))
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
        backend=weight.backend,
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


def _launch_qwen35_router_logits_f32_hidden(
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
    """Launch F32-hidden router logits through the kernel registry."""

    fn = resolve(
        backend=weight.backend,
        layer="router_logits",
        quant=weight.spec.quant_key,
        variant="f32_hidden",
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


def _try_launch_qwen35_router_topk_split_shared_bf16_f32w(
    hidden_ptr: int,
    expert_weight: Qwen35GGUFDeviceWeight,
    shared_weight: Qwen35GGUFDeviceWeight,
    logits_ptr: int,
    selected_ptr: int,
    routing_ptr: int,
    completion_counter_ptr: int,
    *,
    hidden_size: int,
    num_experts: int,
    top_k: int,
    persistent_counter: bool = False,
    stream: int = 0,
    runtime=None,
) -> bool:
    """Try the exact c=1 cooperative router registered for split F32 weights."""

    expert_backend = getattr(expert_weight, "backend", None)
    shared_backend = getattr(shared_weight, "backend", None)
    if expert_backend is None or expert_backend != shared_backend:
        return False
    if expert_weight.spec.quant_key != shared_weight.spec.quant_key:
        return False
    variant = (
        "coop_out_bf16_hidden_persistent"
        if persistent_counter
        else "coop_out_bf16_hidden"
    )
    fn = resolve(
        backend=expert_backend,
        layer="router_topk_split_shared",
        quant=expert_weight.spec.quant_key,
        variant=variant,
        missing="none",
    )
    if fn is None:
        return False
    args = (
        hidden_ptr,
        expert_weight.allocation().tensor.ptr,
        shared_weight.allocation().tensor.ptr,
        logits_ptr,
        selected_ptr,
        routing_ptr,
    )
    if persistent_counter:
        args = (*args, completion_counter_ptr)
    fn(
        *args,
        1,
        hidden_size,
        num_experts,
        top_k,
        threads=256,
        stream=stream,
        runtime=runtime,
    )
    return True


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
    backend: str = "auto"
    resident_weights: Qwen35GGUFResidentWeights | None = field(default=None, repr=False)
    owns_resident_weights: bool = False
    target_arch: str = field(default="", init=False)
    weights: Qwen35GGUFResidentWeights | None = field(default=None, init=False)
    _paged_attn_context_batch: object | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.backend = resolve_backend(self.backend)
        try:
            self.target_arch = hip_target_arch_for_backend(self.backend)
        except ValueError as exc:
            raise RuntimeError("Qwen35GGUFFullStackRunner requires a HIP backend") from exc
        load_backend_kernel_package(self.backend)
        self._paged_attn_context_batch = resolve(
            backend=self.backend,
            layer="paged_attn_decode",
            quant="w4_paro",
            variant="bf16_context_batch_paged_c1_exact_spans",
        )
        self.runtime = self.runtime or get_hip_runtime()
        self.require_cached_build = bool(self.require_cached_build)
        self.weights = self.resident_weights
        if self.weights is None:
            self.weights = materialize_qwen35_gguf_weights(
                self.model_path,
                runtime=self.runtime,
                backend=self.backend,
            )
            self.owns_resident_weights = True
        elif self.weights.backend != self.backend:
            raise ValueError(
                "resident GGUF weight backend does not match runner backend: "
                f"{self.weights.backend!r} != {self.backend!r}"
            )

    def select_prefill_quant(self, quant: str) -> None:
        """Select the registry quant axis for GGUF prefill plugins."""

        selected = str(quant).strip()
        if not selected:
            raise ValueError("prefill quant must be non-empty")
        self._gguf_prefill_quant = selected
        self._gguf_q8_mmq_prefill_policy = resolve_q8_mmq_prefill_policy(selected)
        self.__dict__.pop("_gguf_gdn_prefill_plan_cache", None)
        self.__dict__.pop("_gguf_gdn_decode_output_cast_fn_cache", None)
        self.__dict__.pop("_gguf_full_attn_decode_batch_native_fn_cache", None)
        self.__dict__.pop("_gguf_full_attn_prefill_native_fn_cache", None)

    def _gdn_decode_output_cast_fn(self):
        """Resolve the selected quant plugin's decode-output width contract."""

        missing = object()
        fn = getattr(self, "_gguf_gdn_decode_output_cast_fn_cache", missing)
        if fn is missing:
            quant = getattr(self, "_gguf_prefill_quant", "gguf_qwen35")
            fn = resolve(
                backend=self.backend,
                layer="gdn_decode_output_cast",
                quant=quant,
                variant="f32_to_bf16_exact",
                missing="none",
            )
            self._gguf_gdn_decode_output_cast_fn_cache = fn
        return fn

    def _full_attn_decode_batch_native_fn(self):
        """Return the exact compact-row attention leaf on the quant axis."""

        missing = object()
        fn = getattr(self, "_gguf_full_attn_decode_batch_native_fn_cache", missing)
        if fn is missing:
            quant = getattr(self, "_gguf_prefill_quant", "gguf_qwen35")
            fn = resolve(
                backend=self.backend,
                layer="paged_attn_decode",
                quant=quant,
                variant="bf16_context_batch_native_exact_spans",
                missing="none",
            )
            if fn is None:
                fn = qwen35_paged_full_attn_decode_context_bf16_batch_c1_exact_spans
            self._gguf_full_attn_decode_batch_native_fn_cache = fn
        return fn

    def _full_attn_prefill_native_fn(self):
        """Return the native GQA prefill kernel on the selected quant axis."""

        fn = getattr(self, "_gguf_full_attn_prefill_native_fn_cache", None)
        if fn is None:
            quant = getattr(self, "_gguf_prefill_quant", "gguf_qwen35")
            fn = resolve(
                backend=self.backend,
                layer="full_attn_prefill",
                quant=quant,
                variant="causal_gqa_gate_bf16",
                missing="none",
            )
            if fn is None:
                fn = qwen35_paged_full_attn_prefill_gqa_gate_bf16_spans
            self._gguf_full_attn_prefill_native_fn_cache = fn
        return fn

    def _aotriton_prefill_library(self):
        """Return the cached AOTriton prefill shim handle."""

        library = getattr(self, "_aotriton_library", None)
        if library is None:
            with hip_target_arch_environment(self.target_arch):
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
            with hip_target_arch_environment(self.target_arch):
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
            with hip_target_arch_environment(self.target_arch):
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
            with hip_target_arch_environment(self.target_arch):
                library = build_qwen35_paged_kv_write(
                    load=True,
                    compiler_version=self.compiler_version,
                    require_cached=self.require_cached_build,
                )
            self._paged_kv_write_library_handle = library
        return library

    def _gdn_prefill_plan(self) -> _GGUFGDNPrefillPlan:
        """Return the cached qwen35 GGUF GDN prefill plan.

        Resolved once per runner via the kernel registry. The runtime selector
        chooses the fused or chained implementation independently of registry
        resolution.
        """

        plan = getattr(self, "_gguf_gdn_prefill_plan_cache", None)
        if plan is None:
            plan = _resolve_gguf_gdn_prefill_plan(self.backend)
            self._gguf_gdn_prefill_plan_cache = plan
        return plan

    def _linear_attn_conv_prefill_kernel(self):
        """Resolve and cache the registered GGUF convolution prefill schedule."""

        kernel = getattr(self, "_gguf_linear_attn_conv_prefill_kernel_cache", None)
        if kernel is None:
            load_backend_kernel_package(self.backend)
            mode = _gguf_linear_attn_conv_prefill_mode(self.backend)
            variant = {
                "baseline": "f32_baseline",
                "tile32x128": "f32_tile32x128",
            }[mode]
            kernel = resolve(
                backend=self.backend,
                layer="linear_attn_conv_prefill",
                quant="gguf_qwen35",
                variant=variant,
            )
            self._gguf_linear_attn_conv_prefill_kernel_cache = kernel
        return kernel

    def _linear_attention_decode_batch_plan(self) -> _GGUFLinearAttentionDecodeBatchPlan:
        """Resolve the optional exact indexed Conv/GDN decode capability."""

        plan = getattr(self, "_gguf_linear_attention_decode_batch_plan_cache", None)
        if not isinstance(plan, _GGUFLinearAttentionDecodeBatchPlan):
            plan = _resolve_gguf_linear_attention_decode_batch_plan(self.backend)
            self._gguf_linear_attention_decode_batch_plan_cache = plan
        return plan

    def _packed_decode_metadata_kernel(self):
        """Resolve the optional copy-free packed c4 metadata producer."""

        missing = object()
        kernel = getattr(self, "_gguf_packed_decode_metadata_kernel_cache", missing)
        if kernel is missing:
            load_backend_kernel_package(self.backend)
            kernel = resolve(
                backend=self.backend,
                layer=_PACKED_DECODE_METADATA_KEY.layer,
                quant=_PACKED_DECODE_METADATA_KEY.quant,
                variant=_PACKED_DECODE_METADATA_KEY.variant,
                missing="none",
            )
            self._gguf_packed_decode_metadata_kernel_cache = kernel
        return kernel

    def _packed_ar_attention_batch_kernel(self):
        """Resolve the backend's BF16 long-context packed-AR attention kernel."""

        missing = object()
        kernel = getattr(self, "_gguf_packed_ar_attention_batch_kernel_cache", missing)
        if kernel is missing:
            load_backend_kernel_package(self.backend)
            kernel = resolve(
                backend=self.backend,
                layer=_PACKED_AR_ATTN_BATCH_KEY.layer,
                quant=_PACKED_AR_ATTN_BATCH_KEY.quant,
                variant=_PACKED_AR_ATTN_BATCH_KEY.variant,
                missing="none",
            )
            self._gguf_packed_ar_attention_batch_kernel_cache = kernel
        return kernel

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
        """Dispatch the selected qwen35 GGUF GDN prefill implementation.

        Plugin-style: the kernel chain is resolved via the kernel registry
        keyed by ``(resolved_backend, ..., gguf_qwen35, ...)``.
        ``HIPENGINE_GGUF_GDN_PREFILL_MODE`` selects the fused route, the
        128-column exact chain, or its registered value-tiled/wave diagnostic
        schedules; ``auto`` resolves the architecture-scoped production policy,
        while ``exact`` resolves the architecture-scoped strict-exact
        rollback/oracle. Automatic production falls back to the
        correctness-certified fused route when its preferred schedule is
        unavailable; the explicit exact route fails closed. The raw-scale
        chains keep raw Q/K and normalization scales separate so their
        recurrent kernels preserve fused decode-order arithmetic. Whether the matching single-
        sequence or segment-aware recurrence runs is controlled by
        ``HIPENGINE_GGUF_GDN_PREFILL_SEGMENT_THRESHOLD`` (default 1025), not a
        per-quant/per-backend branch.
        """

        plan = self._gdn_prefill_plan()
        requested_mode = _gguf_gdn_prefill_mode()
        if requested_mode == "auto":
            mode = plan.auto_mode
        elif requested_mode == "exact":
            mode = _gguf_gdn_prefill_backend_exact_mode(self.backend)
        else:
            mode = requested_mode
        if requested_mode == "auto" and not _gguf_gdn_prefill_plan_has_mode(
            plan, mode
        ):
            if plan.has_fused:
                mode = "fused"
            elif plan.has_chain:
                mode = "chain"
            else:
                mode = "auto"
        elif requested_mode == "exact" and not _gguf_gdn_prefill_plan_has_mode(
            plan, mode
        ):
            raise RuntimeError(
                "backend GGUF GDN prefill exact mode is unavailable; "
                f"required route {mode!r} is not fully registered for {self.backend!r}"
            )
        exact_recurrent = plan.exact_recurrent
        exact_recurrent_segments = plan.exact_recurrent_segments
        use_normalized_chain = mode in {
            "chain_k2",
            "chain_peer_wave32",
            "chain_peer_cluster8",
        }
        normalized_prepare = plan.prepare
        normalized_recurrent = plan.recurrent
        normalized_recurrent_segments = plan.recurrent_segments
        if mode == "chain_peer_wave32":
            normalized_prepare = plan.prepare_peer_normalized
            normalized_recurrent = plan.recurrent_peer_wave32
            normalized_recurrent_segments = plan.recurrent_segments_peer_wave32
        elif mode == "chain_peer_cluster8":
            normalized_prepare = plan.prepare_peer_normalized
            normalized_recurrent = plan.recurrent_peer_cluster8
            normalized_recurrent_segments = plan.recurrent_segments_peer_cluster8
        use_direct_lds32 = mode in {
            "chain_lds32_direct",
            "chain_lds32_direct_nonvolatile",
        }
        direct_recurrent = plan.exact_recurrent_lds32_direct
        direct_recurrent_segments = plan.exact_recurrent_segments_lds32_direct
        direct_route_available = plan.has_exact_chain_lds32_direct
        if mode == "chain_lds32_direct_nonvolatile":
            direct_recurrent = plan.exact_recurrent_lds32_direct_nonvolatile
            direct_recurrent_segments = (
                plan.exact_recurrent_segments_lds32_direct_nonvolatile
            )
            direct_route_available = plan.has_exact_chain_lds32_direct_nonvolatile
        if mode == "chain_tile64":
            exact_recurrent = plan.exact_recurrent_tile64
            exact_recurrent_segments = plan.exact_recurrent_segments_tile64
        elif mode == "chain_tile32":
            exact_recurrent = plan.exact_recurrent_tile32
            exact_recurrent_segments = plan.exact_recurrent_segments_tile32
        elif mode == "chain_lds64":
            exact_recurrent = plan.exact_recurrent_lds64
            exact_recurrent_segments = plan.exact_recurrent_segments_lds64
        elif mode == "chain_lds32":
            exact_recurrent = plan.exact_recurrent_lds32
            exact_recurrent_segments = plan.exact_recurrent_segments_lds32
        elif mode == "chain_wave32":
            exact_recurrent = plan.exact_recurrent_wave32
            exact_recurrent_segments = plan.exact_recurrent_segments_wave32
        elif mode == "chain_wave32_tree":
            exact_recurrent = plan.recurrent_wave32_tree
            exact_recurrent_segments = plan.recurrent_segments_wave32_tree
        if requested_mode == "fused" and not plan.has_fused:
            raise RuntimeError(
                "explicit GGUF GDN prefill mode 'fused' is unavailable; "
                "the fused decode-order kernel is not registered"
            )
        if requested_mode == "chain" and not plan.has_chain:
            raise RuntimeError(
                "explicit GGUF GDN prefill mode 'chain' is unavailable; "
                "the prepare, recurrent, and RMSNorm-gate kernels must all be registered"
            )
        if requested_mode == "chain_k2" and not plan.has_chain_k2:
            raise RuntimeError(
                "explicit GGUF GDN prefill mode 'chain_k2' is unavailable; "
                "the normalized prepare, K2 recurrent, and RMSNorm-gate kernels "
                "must all be registered"
            )
        if requested_mode == "chain_peer_wave32" and not plan.has_chain_peer_wave32:
            raise RuntimeError(
                "explicit GGUF GDN prefill mode 'chain_peer_wave32' is unavailable; "
                "the normalized prepare, peer wave32 recurrent, and RMSNorm-gate "
                "kernels must all be registered"
            )
        if requested_mode == "chain_peer_cluster8" and not plan.has_chain_peer_cluster8:
            raise RuntimeError(
                "explicit GGUF GDN prefill mode 'chain_peer_cluster8' is unavailable; "
                "the peer-normalized prepare, clustered8 recurrent, and "
                "RMSNorm-gate kernels must all be registered"
            )
        if requested_mode == "chain_tile64" and not plan.has_exact_chain_tile64:
            raise RuntimeError(
                "explicit GGUF GDN prefill mode 'chain_tile64' is unavailable; "
                "the exact prepare, tile64 recurrent, and RMSNorm-gate kernels "
                "must all be registered"
            )
        if requested_mode == "chain_tile32" and not plan.has_exact_chain_tile32:
            raise RuntimeError(
                "explicit GGUF GDN prefill mode 'chain_tile32' is unavailable; "
                "the exact prepare, tile32 recurrent, and RMSNorm-gate kernels "
                "must all be registered"
            )
        if requested_mode == "chain_lds64" and not plan.has_exact_chain_lds64:
            raise RuntimeError(
                "explicit GGUF GDN prefill mode 'chain_lds64' is unavailable; "
                "the exact prepare, LDS64 recurrent, and RMSNorm-gate kernels "
                "must all be registered"
            )
        if requested_mode == "chain_lds32" and not plan.has_exact_chain_lds32:
            raise RuntimeError(
                "explicit GGUF GDN prefill mode 'chain_lds32' is unavailable; "
                "the exact prepare, LDS32 recurrent, and RMSNorm-gate kernels "
                "must all be registered"
            )
        if (
            requested_mode == "chain_lds32_direct"
            and not plan.has_exact_chain_lds32_direct
        ):
            raise RuntimeError(
                "explicit GGUF GDN prefill mode 'chain_lds32_direct' is unavailable; "
                "the compact-scale prepare, direct LDS32 recurrent, and "
                "RMSNorm-gate kernels must all be registered"
            )
        if (
            requested_mode == "chain_lds32_direct_nonvolatile"
            and not plan.has_exact_chain_lds32_direct_nonvolatile
        ):
            raise RuntimeError(
                "explicit GGUF GDN prefill mode "
                "'chain_lds32_direct_nonvolatile' is unavailable; "
                "the compact-scale prepare, nonvolatile direct LDS32 recurrent, "
                "and RMSNorm-gate kernels must all be registered"
            )
        if requested_mode == "chain_wave32" and not plan.has_exact_chain_wave32:
            raise RuntimeError(
                "explicit GGUF GDN prefill mode 'chain_wave32' is unavailable; "
                "the exact prepare, wave32 recurrent, and RMSNorm-gate kernels "
                "must all be registered"
            )
        if requested_mode == "chain_wave32_tree" and not plan.has_chain_wave32_tree:
            raise RuntimeError(
                "explicit GGUF GDN prefill mode 'chain_wave32_tree' is unavailable; "
                "the exact prepare, tree-reduced wave32 recurrent, and "
                "RMSNorm-gate kernels must all be registered"
            )
        use_fused = plan.has_fused and mode == "fused"
        use_chain = mode in {
            "chain",
            "chain_k2",
            "chain_peer_wave32",
            "chain_peer_cluster8",
            "chain_tile64",
            "chain_tile32",
            "chain_lds64",
            "chain_lds32",
            "chain_lds32_direct",
            "chain_lds32_direct_nonvolatile",
            "chain_wave32",
            "chain_wave32_tree",
        } or (plan.has_chain and not use_fused)
        if use_fused:
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
        if use_chain:
            if use_direct_lds32 and direct_route_available:
                plan.exact_prepare_compact(
                    scratch.conv_out.ptr,
                    scratch.linear_alpha.ptr,
                    scratch.linear_beta.ptr,
                    layer.weight("ssm_dt_bias").allocation().tensor.ptr,
                    layer.weight("ssm_a").allocation().tensor.ptr,
                    scratch.prefill_beta.ptr,
                    scratch.prefill_decay.ptr,
                    scratch.prefill_query_scale.ptr,
                    scratch.prefill_key_scale.ptr,
                    rows,
                    cfg.ssm_group_count,
                    cfg.ssm_time_step_rank,
                    cfg.ssm_state_size,
                    self.ssm_value_dim,
                    stream=stream,
                    runtime=runtime,
                )
                segment_threshold = _gguf_gdn_prefill_segment_threshold()
                use_direct_segments = (
                    direct_recurrent_segments is not None
                    and rows >= segment_threshold
                    and getattr(scratch, "gdn_cu_seqlens", None) is not None
                    and getattr(scratch, "gdn_state_indices", None) is not None
                )
                if use_direct_segments:
                    direct_recurrent_segments(
                        scratch.conv_out.ptr,
                        scratch.prefill_beta.ptr,
                        scratch.prefill_decay.ptr,
                        scratch.prefill_query_scale.ptr,
                        scratch.prefill_key_scale.ptr,
                        recurrent_state.ptr,
                        scratch.recurrent_out.ptr,
                        scratch.gdn_cu_seqlens.ptr,
                        scratch.gdn_state_indices.ptr,
                        rows,
                        1,
                        cfg.ssm_group_count,
                        cfg.ssm_time_step_rank,
                        cfg.ssm_state_size,
                        self.ssm_value_dim,
                        stream=stream,
                        runtime=runtime,
                    )
                else:
                    direct_recurrent(
                        scratch.conv_out.ptr,
                        scratch.prefill_beta.ptr,
                        scratch.prefill_decay.ptr,
                        scratch.prefill_query_scale.ptr,
                        scratch.prefill_key_scale.ptr,
                        recurrent_state.ptr,
                        scratch.recurrent_out.ptr,
                        rows,
                        cfg.ssm_group_count,
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
            if (
                not use_normalized_chain
                and plan.exact_prepare is not None
                and exact_recurrent is not None
                and plan.rmsnorm_gate is not None
            ):
                plan.exact_prepare(
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
                    scratch.prefill_query_scale.ptr,
                    scratch.prefill_key_scale.ptr,
                    rows,
                    cfg.ssm_group_count,
                    cfg.ssm_time_step_rank,
                    cfg.ssm_state_size,
                    self.ssm_value_dim,
                    stream=stream,
                    runtime=runtime,
                )
                segment_threshold = _gguf_gdn_prefill_segment_threshold()
                use_exact_segments = (
                    exact_recurrent_segments is not None
                    and rows >= segment_threshold
                    and getattr(scratch, "gdn_cu_seqlens", None) is not None
                    and getattr(scratch, "gdn_state_indices", None) is not None
                )
                if use_exact_segments:
                    exact_recurrent_segments(
                        scratch.prefill_query.ptr,
                        scratch.prefill_key.ptr,
                        scratch.prefill_value.ptr,
                        scratch.prefill_beta.ptr,
                        scratch.prefill_decay.ptr,
                        scratch.prefill_query_scale.ptr,
                        scratch.prefill_key_scale.ptr,
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
                    exact_recurrent(
                        scratch.prefill_query.ptr,
                        scratch.prefill_key.ptr,
                        scratch.prefill_value.ptr,
                        scratch.prefill_beta.ptr,
                        scratch.prefill_decay.ptr,
                        scratch.prefill_query_scale.ptr,
                        scratch.prefill_key_scale.ptr,
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
            normalized_prepare(
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
                normalized_recurrent_segments is not None
                and rows >= segment_threshold
                and getattr(scratch, "gdn_cu_seqlens", None) is not None
                and getattr(scratch, "gdn_state_indices", None) is not None
            )
            if use_segments:
                normalized_recurrent_segments(
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
                normalized_recurrent(
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
                active_layer_types = self.weights.config.layer_types[:layer_count]
                chain_next_rms = self.weights.config.is_moe and _gguf_moe_tail_next_rms_enabled()
                input_norm_ptr: int | None = None
                for layer_id, layer_type in enumerate(active_layer_types):
                    next_norm_weight_ptr = None
                    next_norm_out_ptr = None
                    if chain_next_rms and layer_id + 1 < len(active_layer_types):
                        next_norm_weight_ptr = (
                            self.weights.layer(layer_id + 1).weight("attn_norm").allocation().tensor.ptr
                        )
                        next_norm_out_ptr = scratch.norm.ptr
                    if layer_type == LINEAR_ATTENTION:
                        self._run_linear_attention_layer(
                            layer_id,
                            src.ptr,
                            dst.ptr,
                            scratch,
                            input_norm_ptr=input_norm_ptr,
                            next_norm_weight_ptr=next_norm_weight_ptr,
                            next_norm_out_ptr=next_norm_out_ptr,
                        )
                    elif layer_type == FULL_ATTENTION:
                        self._run_full_attention_layer(
                            layer_id,
                            src.ptr,
                            dst.ptr,
                            scratch,
                            position=position,
                            input_norm_ptr=input_norm_ptr,
                            next_norm_weight_ptr=next_norm_weight_ptr,
                            next_norm_out_ptr=next_norm_out_ptr,
                        )
                    else:
                        raise ValueError(f"unsupported GGUF layer type {layer_type!r}")
                    input_norm_ptr = next_norm_out_ptr
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
                    cos_table_ptr=prefill_scratch.cos_table.ptr,
                    sin_table_ptr=prefill_scratch.sin_table.ptr,
                    max_positions=rows,
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
        cos_table_ptr: int,
        sin_table_ptr: int,
        max_positions: int,
        attn_aotriton_min_tokens: int | None = None,
        stream: int = 0,
        aotriton_bridge: AotritonPrefillStreamBridge | None = None,
        expert_sidecar: _DeviceExpertLayerSidecar | None = None,
        allow_aotriton: bool = True,
        aotriton_min_tokens: int | None = None,
        paged_max_context_len: int | None = None,
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
            cos_table_ptr,
            sin_table_ptr,
            scratch.positions_tensor.ptr,
            scratch.full_query.ptr,
            scratch.full_key.ptr,
            cfg.rms_norm_eps,
            rows,
            cfg.head_count,
            cfg.head_count_kv,
            cfg.key_length,
            cfg.rope_dimension_count,
            max_positions,
            stream=stream,
            runtime=runtime,
        )
        if scratch.key_cache is None or scratch.value_cache is None:
            raise RuntimeError(
                "GGUF full-attention prefill requires cache-backed key/value buffers; "
                "resident bulk prefill should provide either retained BF16 caches or a BF16 oracle cache for INT8 retention"
            )
        append_metadata = scratch.append_spans.scale_metadata
        direct_hadamard_int8 = (
            scratch.append_spans.storage_dtype == DType.INT8_PER_TOKEN_HEAD
            and append_metadata is not None
            and append_metadata.granularity == "hadamard_group32"
        )
        if direct_hadamard_int8:
            if (
                scratch.prefill_spans.storage_dtype != DType.INT8_PER_TOKEN_HEAD
                or scratch.prefill_spans.scale_metadata is not append_metadata
            ):
                raise RuntimeError("GGUF direct Hadamard INT8 prefill requires matching append/attention metadata")
            bf16_to_f32(
                scratch.full_v.ptr,
                scratch.full_key_raw.ptr,
                rows * self.kv_width,
                stream=stream,
                library=cast_library,
                runtime=runtime,
            )
            int8_prompt_write_fn = _gguf_int8_kv_prompt_write_fn(append_metadata)
            int8_prompt_write_fn(
                scratch.full_key.ptr,
                scratch.full_key_raw.ptr,
                scratch.key_cache.ptr,
                scratch.value_cache.ptr,
                append_metadata.k_scale.ptr,
                append_metadata.v_scale.ptr,
                scratch.append_spans,
                rows,
                scratch.block_size,
                cfg.head_count_kv,
                cfg.key_length,
                stream=stream,
                library=kv_write_library,
                runtime=runtime,
            )
        else:
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
            if scratch.retained_key_cache is not None or scratch.retained_value_cache is not None:
                if scratch.retained_key_cache is None or scratch.retained_value_cache is None:
                    raise RuntimeError("GGUF INT8 retained prefill requires both key and value caches")
                retained_spans = scratch.retained_append_spans
                if retained_spans is None or retained_spans.scale_metadata is None:
                    raise RuntimeError("GGUF INT8 retained prefill requires append spans with scale metadata")
                bf16_to_f32(
                    scratch.full_v.ptr,
                    scratch.full_key_raw.ptr,
                    rows * self.kv_width,
                    stream=stream,
                    library=cast_library,
                    runtime=runtime,
                )
                if getattr(scratch, "int8_kv_value_bf16", False):
                    qwen35_write_paged_kv_int8_key_bf16_value_prompt_spans(
                        scratch.full_key.ptr,
                        scratch.full_key_raw.ptr,
                        scratch.retained_key_cache.ptr,
                        scratch.retained_value_cache.ptr,
                        retained_spans.scale_metadata.k_scale.ptr,
                        retained_spans,
                        rows,
                        scratch.block_size,
                        cfg.head_count_kv,
                        cfg.key_length,
                        stream=stream,
                        library=kv_write_library,
                        runtime=runtime,
                    )
                else:
                    int8_prompt_write_fn = _gguf_int8_kv_prompt_write_fn(retained_spans.scale_metadata)
                    int8_prompt_write_fn(
                        scratch.full_key.ptr,
                        scratch.full_key_raw.ptr,
                        scratch.retained_key_cache.ptr,
                        scratch.retained_value_cache.ptr,
                        retained_spans.scale_metadata.k_scale.ptr,
                        retained_spans.scale_metadata.v_scale.ptr,
                        retained_spans,
                        rows,
                        scratch.block_size,
                        cfg.head_count_kv,
                        cfg.key_length,
                        stream=stream,
                        library=kv_write_library,
                        runtime=runtime,
                    )
        threshold = int(
            PrefillConfig().attn_aotriton_min_tokens
            if aotriton_min_tokens is None
            else aotriton_min_tokens
        )
        if threshold < 0:
            raise ValueError("aotriton_min_tokens must be non-negative")
        use_aotriton = bool(
            allow_aotriton and not direct_hadamard_int8 and threshold > 0 and rows >= threshold
        )
        paged_attn_library = self._paged_attn_decode_library()
        end = scratch.start + rows
        paged_context_len = end if paged_max_context_len is None else int(paged_max_context_len)
        if paged_context_len <= 0 or paged_context_len > int(scratch.max_positions):
            raise ValueError(
                "paged_max_context_len must be positive and within the prefill scratch capacity"
            )

        if direct_hadamard_int8:
            qwen35_paged_attn_prefill_int8_hadamard_group32_gqa_gate_fp16_spans(
                scratch.full_query.ptr,
                scratch.key_cache.ptr,
                scratch.value_cache.ptr,
                append_metadata.k_scale.ptr,
                append_metadata.v_scale.ptr,
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
        elif use_aotriton:
            aotriton_library = self._aotriton_prefill_library()

            # Keep pre/post work on the caller stream. Only the high-scratch
            # AOTriton dispatch moves to the event-linked isolated queue.
            aotriton_stream = stream

            # Convert FP32 query to BF16 for AOTriton
            f32_to_bf16(
                scratch.full_query.ptr,
                scratch.full_query_bf16.ptr,
                rows * self.q_width,
                stream=stream,
                library=cast_library,
                runtime=runtime,
            )

            kv_key_ptr = scratch.key_cache.ptr
            kv_value_ptr = scratch.value_cache.ptr
            kv_strides = (self.kv_width * end, cfg.key_length, self.kv_width, 1)
            head_major_buffers = _gguf_aotriton_head_major_buffers(
                scratch,
                context_len=end,
            )
            if head_major_buffers is not None:
                head_major_key, head_major_value, head_major_capacity = head_major_buffers
                copy_variant = (
                    "head_major_dense_prefix_spans"
                    if bool(getattr(scratch, "head_major_kv_dense_prefix", False))
                    else "head_major_spans"
                )
                copy_head_major = resolve(
                    backend=scratch.backend,
                    layer="paged_kv_copy",
                    quant="bf16",
                    variant=copy_variant,
                    missing="none",
                )
                if copy_head_major is not None:
                    copy_head_major(
                        scratch.key_cache.ptr,
                        scratch.value_cache.ptr,
                        head_major_key.ptr,
                        head_major_value.ptr,
                        scratch.prefill_spans,
                        end,
                        head_major_capacity,
                        scratch.block_size,
                        cfg.head_count_kv,
                        cfg.key_length,
                        stream=stream,
                        library=kv_write_library,
                        runtime=runtime,
                    )
                    kv_key_ptr = head_major_key.ptr
                    kv_value_ptr = head_major_value.ptr
                    kv_strides = (
                        self.kv_width * head_major_capacity,
                        cfg.key_length * head_major_capacity,
                        cfg.key_length,
                        1,
                    )

            k_tensor = aotriton_tensor4(
                kv_key_ptr,
                (1, cfg.head_count_kv, end, cfg.key_length),
                kv_strides,
                DType.BF16,
            )
            v_tensor = aotriton_tensor4(
                kv_value_ptr,
                (1, cfg.head_count_kv, end, cfg.key_length),
                kv_strides,
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
            if aotriton_bridge is not None:
                aotriton_bridge.wait_for_inputs(runtime, stream)
                aotriton_stream = aotriton_bridge.stream
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
                    stream=aotriton_stream,
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
                    stream=aotriton_stream,
                    library=aotriton_library,
                    runtime=runtime,
                )

            if aotriton_bridge is not None:
                aotriton_bridge.release_output(runtime, stream)
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
            self._full_attn_prefill_native_fn()(
                scratch.full_query.ptr,
                scratch.key_cache.ptr,
                scratch.value_cache.ptr,
                scratch.full_gate.ptr,
                scratch.full_gated.ptr,
                scratch.prefill_spans,
                rows,
                paged_context_len,
                scratch.block_size,
                cfg.head_count,
                cfg.head_count_kv,
                cfg.key_length,
                cfg.key_length,
                1,
                cfg.key_length ** -0.5,
                split_partial_out_ptr=scratch.full_attn_split_partial.ptr,
                split_partial_m_ptr=scratch.full_attn_split_m.ptr,
                split_partial_l_ptr=scratch.full_attn_split_l.ptr,
                split_batch_rows=scratch.full_attn_split_batch_rows,
                split_count=scratch.full_attn_split_count,
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

    def _run_attention_norm_rows(
        self,
        *,
        hidden_ptr: int,
        hidden_f32_ptr: int | None,
        weight_ptr: int,
        out_ptr: int,
        out_f32_ptr: int | None = None,
        rows: int,
        eps: float,
        stream: int,
        runtime: HipRuntime,
    ) -> int | None:
        if hidden_f32_ptr is None:
            gguf_rmsnorm_bf16_f32_weight(
                hidden_ptr,
                weight_ptr,
                out_ptr,
                rows=rows,
                hidden_size=self.hidden_size,
                eps=eps,
                stream=stream,
                runtime=runtime,
            )
            return None
        if _gguf_verify_f32_attention_norm_enabled() and out_f32_ptr is not None:
            gguf_rmsnorm_f32_f32_weight_out_f32(
                int(hidden_f32_ptr),
                weight_ptr,
                int(out_f32_ptr),
                rows=rows,
                hidden_size=self.hidden_size,
                eps=eps,
                stream=stream,
                runtime=runtime,
            )
            f32_to_bf16(
                int(out_f32_ptr),
                out_ptr,
                rows * self.hidden_size,
                stream=stream,
                library=self._cast_library(),
                runtime=runtime,
            )
            return int(out_f32_ptr)
        else:
            gguf_rmsnorm_f32_f32_weight(
                int(hidden_f32_ptr),
                weight_ptr,
                out_ptr,
                rows=rows,
                hidden_size=self.hidden_size,
                eps=eps,
                stream=stream,
                runtime=runtime,
            )
            return None

    def _run_linear_attention_alpha_beta_rows(
        self,
        layer,
        norm_ptr: int,
        norm_f32_ptr: int | None,
        scratch,
        *,
        rows: int,
        stream: int,
        runtime: HipRuntime,
    ) -> str:
        cfg = self.weights.config
        if (
            norm_f32_ptr is not None
            and _gguf_verify_f32_linear_projections_enabled()
            and hasattr(scratch, "linear_alpha_f32")
            and hasattr(scratch, "linear_beta_f32")
        ):
            if (
                _gguf_linear_supports_f32_activation_f32_output(layer.weight("ssm_alpha"), rows=rows)
                and _gguf_linear_supports_f32_activation_f32_output(layer.weight("ssm_beta"), rows=rows)
            ):
                launch_gguf_linear(
                    layer.weight("ssm_alpha"),
                    int(norm_f32_ptr),
                    scratch.linear_alpha_f32.ptr,
                    rows=rows,
                    in_features=self.hidden_size,
                    out_features=cfg.ssm_time_step_rank,
                    activation_dtype=GGUF_ACTIVATION_F32,
                    output_dtype=GGUF_OUTPUT_F32,
                    stream=stream,
                    runtime=runtime,
                )
                launch_gguf_linear(
                    layer.weight("ssm_beta"),
                    int(norm_f32_ptr),
                    scratch.linear_beta_f32.ptr,
                    rows=rows,
                    in_features=self.hidden_size,
                    out_features=cfg.ssm_time_step_rank,
                    activation_dtype=GGUF_ACTIVATION_F32,
                    output_dtype=GGUF_OUTPUT_F32,
                    stream=stream,
                    runtime=runtime,
                )
                cast_library = self._cast_library()
                f32_to_bf16(
                    scratch.linear_alpha_f32.ptr,
                    scratch.linear_alpha.ptr,
                    rows * cfg.ssm_time_step_rank,
                    stream=stream,
                    library=cast_library,
                    runtime=runtime,
                )
                f32_to_bf16(
                    scratch.linear_beta_f32.ptr,
                    scratch.linear_beta.ptr,
                    rows * cfg.ssm_time_step_rank,
                    stream=stream,
                    library=cast_library,
                    runtime=runtime,
                )
                return "f32_singletons_f32_out"
            if _try_launch_dense_q8_pair_dp4a_f32_out(
                layer.weight("ssm_alpha"),
                layer.weight("ssm_beta"),
                int(norm_f32_ptr),
                scratch.linear_alpha_f32.ptr,
                scratch.linear_beta_f32.ptr,
                scratch,
                rows=rows,
                in_features=self.hidden_size,
                out_features_a=cfg.ssm_time_step_rank,
                out_features_b=cfg.ssm_time_step_rank,
                stream=stream,
                runtime=runtime,
            ):
                cast_library = self._cast_library()
                f32_to_bf16(
                    scratch.linear_alpha_f32.ptr,
                    scratch.linear_alpha.ptr,
                    rows * cfg.ssm_time_step_rank,
                    stream=stream,
                    library=cast_library,
                    runtime=runtime,
                )
                f32_to_bf16(
                    scratch.linear_beta_f32.ptr,
                    scratch.linear_beta.ptr,
                    rows * cfg.ssm_time_step_rank,
                    stream=stream,
                    library=cast_library,
                    runtime=runtime,
                )
                return "dense_q8_dp4a_f32_out"
        if norm_f32_ptr is not None and _gguf_verify_f32_alpha_beta_enabled():
            if _try_launch_dense_q8_pair_dp4a_f32(
                layer.weight("ssm_alpha"),
                layer.weight("ssm_beta"),
                int(norm_f32_ptr),
                scratch.linear_alpha.ptr,
                scratch.linear_beta.ptr,
                scratch,
                rows=rows,
                in_features=self.hidden_size,
                out_features_a=cfg.ssm_time_step_rank,
                out_features_b=cfg.ssm_time_step_rank,
                stream=stream,
                runtime=runtime,
            ):
                return "dense_q8_dp4a_f32"
            if (
                _gguf_linear_supports_f32_activation(layer.weight("ssm_alpha"))
                and _gguf_linear_supports_f32_activation(layer.weight("ssm_beta"))
            ):
                launch_gguf_linear(
                    layer.weight("ssm_alpha"),
                    int(norm_f32_ptr),
                    scratch.linear_alpha.ptr,
                    rows=rows,
                    in_features=self.hidden_size,
                    out_features=cfg.ssm_time_step_rank,
                    activation_dtype=GGUF_ACTIVATION_F32,
                    stream=stream,
                    runtime=runtime,
                )
                launch_gguf_linear(
                    layer.weight("ssm_beta"),
                    int(norm_f32_ptr),
                    scratch.linear_beta.ptr,
                    rows=rows,
                    in_features=self.hidden_size,
                    out_features=cfg.ssm_time_step_rank,
                    activation_dtype=GGUF_ACTIVATION_F32,
                    stream=stream,
                    runtime=runtime,
                )
                return "f32_singletons"
        if cfg.is_moe:
            launch_gguf_linear(
                layer.weight("ssm_alpha"),
                norm_ptr,
                scratch.linear_alpha.ptr,
                rows=rows,
                in_features=self.hidden_size,
                out_features=cfg.ssm_time_step_rank,
                stream=stream,
                runtime=runtime,
            )
            launch_gguf_linear(
                layer.weight("ssm_beta"),
                norm_ptr,
                scratch.linear_beta.ptr,
                rows=rows,
                in_features=self.hidden_size,
                out_features=cfg.ssm_time_step_rank,
                stream=stream,
                runtime=runtime,
            )
            return "singletons"
        if launch_gguf_linear_pair(
            layer.weight("ssm_alpha"),
            layer.weight("ssm_beta"),
            norm_ptr,
            scratch.linear_alpha.ptr,
            scratch.linear_beta.ptr,
            rows=rows,
            in_features=self.hidden_size,
            out_features=cfg.ssm_time_step_rank,
            stream=stream,
            runtime=runtime,
        ):
            return "pair"
        launch_gguf_linear(
            layer.weight("ssm_alpha"),
            norm_ptr,
            scratch.linear_alpha.ptr,
            rows=rows,
            in_features=self.hidden_size,
            out_features=cfg.ssm_time_step_rank,
            stream=stream,
            runtime=runtime,
        )
        launch_gguf_linear(
            layer.weight("ssm_beta"),
            norm_ptr,
            scratch.linear_beta.ptr,
            rows=rows,
            in_features=self.hidden_size,
            out_features=cfg.ssm_time_step_rank,
            stream=stream,
            runtime=runtime,
        )
        return "fallback"

    def _run_full_attention_decode_batch_layer_rows(
        self,
        layer_id: int,
        hidden_ptr: int,
        out_ptr: int,
        scratch,
        *,
        stream: int = 0,
        expert_sidecar: _DeviceExpertLayerSidecar | None = None,
        hidden_f32_ptr: int | None = None,
        out_f32_ptr: int | None = None,
        stage_timings: dict[str, float] | None = None,
        sync_stage_timings: bool = False,
        stage_prefix: str = "target_block_full_attn",
        split_workspace: _GGUFPackedARAttentionWorkspace | None = None,
    ) -> str:
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
        max_context_len = int(getattr(scratch.prefill_spans, "max_live_count", end))
        cast_library = self._cast_library()
        kv_write_library = self._paged_kv_write_library()
        paged_attn_library = self._paged_attn_decode_library()
        sync_stages = bool(sync_stage_timings and stage_timings is not None)
        t_stage = time.perf_counter() if sync_stages else 0.0
        attn_norm_f32_ptr = self._run_attention_norm_rows(
            hidden_ptr=hidden_ptr,
            hidden_f32_ptr=hidden_f32_ptr,
            weight_ptr=layer.weight("attn_norm").allocation().tensor.ptr,
            out_ptr=scratch.norm.ptr,
            out_f32_ptr=(scratch.post_norm_f32.ptr if hasattr(scratch, "post_norm_f32") else None),
            rows=rows,
            eps=cfg.rms_norm_eps,
            stream=stream,
            runtime=runtime,
        )
        if not (
            attn_norm_f32_ptr is not None
            and _try_launch_dense_q8_triple_dp4a_f32(
                layer.weight("attn_q"),
                layer.weight("attn_k"),
                layer.weight("attn_v"),
                int(attn_norm_f32_ptr),
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
            )
        ) and not _try_launch_dense_q8_triple_dp4a(
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
        retained_spans = getattr(scratch, "retained_append_spans", None)
        if retained_spans is None:
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
        else:
            metadata = retained_spans.scale_metadata
            retained_key_cache = getattr(scratch, "retained_key_cache", None)
            retained_value_cache = getattr(scratch, "retained_value_cache", None)
            if metadata is None or retained_key_cache is None or retained_value_cache is None:
                raise RuntimeError("packed INT8 KV write requires retained payload and scale metadata")
            bf16_to_f32(
                scratch.full_v.ptr,
                scratch.full_key_raw.ptr,
                rows * self.kv_width,
                stream=stream,
                library=cast_library,
                runtime=runtime,
            )
            if bool(getattr(scratch, "int8_kv_value_bf16", False)):
                qwen35_write_paged_kv_int8_key_bf16_value_prompt_spans(
                    scratch.full_key.ptr,
                    scratch.full_key_raw.ptr,
                    retained_key_cache.ptr,
                    retained_value_cache.ptr,
                    metadata.k_scale.ptr,
                    retained_spans,
                    rows,
                    scratch.block_size,
                    cfg.head_count_kv,
                    cfg.key_length,
                    stream=stream,
                    library=kv_write_library,
                    runtime=runtime,
                )
            else:
                _gguf_int8_kv_prompt_write_fn(metadata)(
                    scratch.full_key.ptr,
                    scratch.full_key_raw.ptr,
                    retained_key_cache.ptr,
                    retained_value_cache.ptr,
                    metadata.k_scale.ptr,
                    metadata.v_scale.ptr,
                    retained_spans,
                    rows,
                    scratch.block_size,
                    cfg.head_count_kv,
                    cfg.key_length,
                    stream=stream,
                    library=kv_write_library,
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
        t_stage = _mark_sync_stage(
            runtime,
            stage_timings,
            sync_stages,
            f"{stage_prefix}_kv_write",
            t_stage,
        )
        if max_context_len < 1024:
            context_batch = self._paged_attn_context_batch
            if not callable(context_batch):  # pragma: no cover - registry resolve is fail-closed
                raise RuntimeError("paged context-batch attention kernel is unavailable")
            context_batch(
                scratch.full_query.ptr,
                scratch.key_cache.ptr,
                scratch.value_cache.ptr,
                scratch.full_query_raw.ptr,
                scratch.prefill_spans,
                rows,
                max_context_len,
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
                scratch.full_query_raw.ptr,
                scratch.full_gate.ptr,
                scratch.full_gated.ptr,
                rows * self.q_width,
                stream=stream,
                library=paged_attn_library,
                runtime=runtime,
            )
        else:
            num_splits = (max_context_len + int(scratch.block_size) - 1) // int(
                scratch.block_size
            )
            if (
                split_workspace is None
                or int(split_workspace.rows) < rows
                or int(split_workspace.num_splits) < num_splits
            ):
                raise NotImplementedError(
                    "long-context packed AR decode requires a row-sized split-K workspace"
                )
            split_batch = self._packed_ar_attention_batch_kernel()
            if not callable(split_batch):
                raise NotImplementedError(
                    "long-context packed AR BF16 batch attention kernel is unavailable"
                )
            split_batch(
                scratch.full_query.ptr,
                scratch.key_cache.ptr,
                scratch.value_cache.ptr,
                scratch.full_gate.ptr,
                scratch.full_gated.ptr,
                split_workspace.partial_out.ptr,
                split_workspace.partial_m.ptr,
                split_workspace.partial_l.ptr,
                scratch.prefill_spans,
                rows,
                int(scratch.block_size),
                num_splits,
                int(scratch.block_size),
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
        t_stage = _mark_sync_stage(
            runtime,
            stage_timings,
            sync_stages,
            f"{stage_prefix}_paged_attn",
            t_stage,
        )
        attn_out_f32_ptr = self._run_full_attention_output_rows(
            layer,
            scratch.full_gated.ptr,
            scratch.attn_out.ptr,
            scratch,
            rows=rows,
            hidden_f32_ptr=hidden_f32_ptr,
            out_f32_ptr=out_f32_ptr,
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
            hidden_f32_ptr=hidden_f32_ptr,
            out_f32_ptr=out_f32_ptr,
            attn_out_f32_ptr=attn_out_f32_ptr,
            stage_timings=stage_timings,
            sync_stage_timings=sync_stage_timings,
            stage_prefix=f"{stage_prefix}_ffn",
        )
        return "kv_live_spans_batch"

    def _run_linear_attention_layer(
        self,
        layer_id: int,
        hidden_ptr: int,
        out_ptr: int,
        scratch,
        *,
        input_norm_ptr: int | None = None,
        next_norm_weight_ptr: int | None = None,
        next_norm_out_ptr: int | None = None,
        stream: int = 0,
    ) -> None:
        self._run_linear_attention_attn_only(
            layer_id,
            hidden_ptr,
            scratch.attn_out.ptr,
            scratch,
            input_norm_ptr=input_norm_ptr,
            stream=stream,
        )
        self._run_post_attention_ffn(
            layer_id,
            hidden_ptr,
            scratch.attn_out.ptr,
            out_ptr,
            scratch,
            next_norm_weight_ptr=next_norm_weight_ptr,
            next_norm_out_ptr=next_norm_out_ptr,
            stream=stream,
        )

    def _run_linear_attention_attn_only(
        self,
        layer_id: int,
        hidden_ptr: int,
        attn_out_ptr: int,
        scratch,
        *,
        hidden_f32_ptr: int | None = None,
        input_norm_ptr: int | None = None,
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
        if input_norm_ptr is None:
            attn_norm_f32_ptr = self._run_attention_norm_rows(
                hidden_ptr=hidden_ptr,
                hidden_f32_ptr=hidden_f32_ptr,
                weight_ptr=layer.weight("attn_norm").allocation().tensor.ptr,
                out_ptr=scratch.norm.ptr,
                out_f32_ptr=(scratch.post_norm_f32.ptr if hasattr(scratch, "post_norm_f32") else None),
                rows=1,
                eps=cfg.rms_norm_eps,
                stream=stream,
                runtime=runtime,
            )
        else:
            if int(input_norm_ptr) != int(scratch.norm.ptr):
                raise ValueError("prefused linear-attention input norm must use scratch.norm")
            attn_norm_f32_ptr = None
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
        self._run_linear_attention_alpha_beta_rows(
            layer,
            scratch.norm.ptr,
            attn_norm_f32_ptr,
            scratch,
            rows=1,
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
        ssm_out_input_ptr = scratch.recurrent_out.ptr
        ssm_out_activation_dtype = GGUF_ACTIVATION_F32
        output_cast = self._gdn_decode_output_cast_fn()
        if output_cast is not None:
            output_cast(
                scratch.recurrent_out.ptr,
                scratch.recurrent_bf16.ptr,
                cfg.ssm_inner_size,
                stream=stream,
                library=self._cast_library(),
                runtime=runtime,
            )
            ssm_out_input_ptr = scratch.recurrent_bf16.ptr
            ssm_out_activation_dtype = GGUF_ACTIVATION_BF16
        launch_gguf_linear(
            layer.weight("ssm_out"),
            ssm_out_input_ptr,
            attn_out_ptr,
            rows=1,
            in_features=cfg.ssm_inner_size,
            out_features=self.hidden_size,
            activation_dtype=ssm_out_activation_dtype,
            stream=stream,
            runtime=runtime,
        )

    def _run_linear_attention_attn_rows_indexed_exact(
        self,
        layer_id: int,
        hidden_ptr: int,
        attn_out_ptr: int,
        scratch,
        *,
        rows: int,
        decode_scratch,
        batch_plan,
        gdn_cu_seqlens_ptr: int,
        state_indices_ptr: int,
        hidden_f32_ptr: int | None = None,
        stream: int = 0,
    ) -> None:
        """Run independent packed decode rows without rounding GDN output.

        Norm and GGUF projections use their existing row-shaped launch ABIs.
        Conv indexes the canonical packed state slab directly, while segmented
        GDN preserves the scalar kernel's FP32 output for the ``ssm_out``
        projection.  Every segment contains exactly one decode token.
        """

        if rows <= 0:
            raise ValueError("rows must be positive")
        if not bool(getattr(batch_plan, "available", False)):
            raise ValueError("indexed linear-attention decode plan is incomplete")
        if int(gdn_cu_seqlens_ptr) == 0 or int(state_indices_ptr) == 0:
            raise ValueError("indexed linear-attention decode requires segment metadata")
        if self.weights is None:
            raise RuntimeError("GGUF weights are not materialized")
        layer = self.weights.layer(layer_id)
        cfg = self.weights.config
        runtime = self.runtime or get_hip_runtime()
        conv_state = decode_scratch.layer_conv_states[layer_id]
        recurrent_state = decode_scratch.layer_recurrent_states[layer_id]
        if conv_state is None or recurrent_state is None:
            raise ValueError(f"layer {layer_id} has no linear-attention state")

        attn_norm_f32_ptr = self._run_attention_norm_rows(
            hidden_ptr=hidden_ptr,
            hidden_f32_ptr=hidden_f32_ptr,
            weight_ptr=layer.weight("attn_norm").allocation().tensor.ptr,
            out_ptr=scratch.norm.ptr,
            out_f32_ptr=(
                scratch.post_norm_f32.ptr
                if hasattr(scratch, "post_norm_f32")
                else None
            ),
            rows=rows,
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
            rows=rows,
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
        self._run_linear_attention_alpha_beta_rows(
            layer,
            scratch.norm.ptr,
            attn_norm_f32_ptr,
            scratch,
            rows=rows,
            stream=stream,
            runtime=runtime,
        )
        batch_plan.conv_indexed(
            scratch.linear_qkv.ptr,
            conv_state.ptr,
            layer.weight("ssm_conv1d").allocation().tensor.ptr,
            scratch.conv_out.ptr,
            state_indices_ptr,
            rows,
            self.linear_qkv_width,
            cfg.ssm_conv_kernel,
            stream=stream,
            runtime=runtime,
        )
        if callable(batch_plan.gdn_indexed_singleton):
            batch_plan.gdn_indexed_singleton(
                scratch.conv_out.ptr,
                scratch.linear_z.ptr,
                scratch.linear_alpha.ptr,
                scratch.linear_beta.ptr,
                layer.weight("ssm_dt_bias").allocation().tensor.ptr,
                layer.weight("ssm_a").allocation().tensor.ptr,
                layer.weight("ssm_norm").allocation().tensor.ptr,
                recurrent_state.ptr,
                scratch.recurrent_out.ptr,
                state_indices_ptr,
                rows,
                cfg.rms_norm_eps,
                cfg.ssm_group_count,
                cfg.ssm_time_step_rank,
                cfg.ssm_state_size,
                self.ssm_value_dim,
                stream=stream,
                runtime=runtime,
            )
        else:
            batch_plan.gdn_segments(
                scratch.conv_out.ptr,
                scratch.linear_z.ptr,
                scratch.linear_alpha.ptr,
                scratch.linear_beta.ptr,
                layer.weight("ssm_dt_bias").allocation().tensor.ptr,
                layer.weight("ssm_a").allocation().tensor.ptr,
                layer.weight("ssm_norm").allocation().tensor.ptr,
                recurrent_state.ptr,
                scratch.recurrent_out.ptr,
                gdn_cu_seqlens_ptr,
                state_indices_ptr,
                rows,
                rows,
                cfg.rms_norm_eps,
                cfg.ssm_group_count,
                cfg.ssm_time_step_rank,
                cfg.ssm_state_size,
                self.ssm_value_dim,
                stream=stream,
                runtime=runtime,
            )
        ssm_out_input_ptr = scratch.recurrent_out.ptr
        ssm_out_activation_dtype = GGUF_ACTIVATION_F32
        output_cast = self._gdn_decode_output_cast_fn()
        if output_cast is not None:
            output_cast(
                scratch.recurrent_out.ptr,
                scratch.recurrent_bf16.ptr,
                rows * cfg.ssm_inner_size,
                stream=stream,
                library=self._cast_library(),
                runtime=runtime,
            )
            ssm_out_input_ptr = scratch.recurrent_bf16.ptr
            ssm_out_activation_dtype = GGUF_ACTIVATION_BF16
        launch_gguf_linear(
            layer.weight("ssm_out"),
            ssm_out_input_ptr,
            attn_out_ptr,
            rows=rows,
            in_features=cfg.ssm_inner_size,
            out_features=self.hidden_size,
            activation_dtype=ssm_out_activation_dtype,
            stream=stream,
            runtime=runtime,
        )

    def _run_linear_attention_decode_slot_rows_exact(
        self,
        layer_id: int,
        hidden_ptr: int,
        out_ptr: int,
        scratch,
        *,
        rows: int,
        state_indices: tuple[int, ...],
        decode_scratch,
        batch_plan=None,
        gdn_cu_seqlens_ptr: int | None = None,
        state_indices_ptr: int | None = None,
        stream: int = 0,
        expert_sidecar: _DeviceExpertLayerSidecar | None = None,
        hidden_f32_ptr: int | None = None,
        out_f32_ptr: int | None = None,
        stage_timings: dict[str, float] | None = None,
        sync_stage_timings: bool = False,
        stage_prefix: str = "ar_batch_linear_attn",
    ) -> str:
        """Run exact indexed linear attention or its scalar row fallback."""

        if rows <= 0:
            raise ValueError("rows must be positive")
        if len(state_indices) != rows:
            raise ValueError("state_indices must contain one entry per decode row")
        if self.weights is None:
            raise RuntimeError("GGUF weights are not materialized")
        cfg = self.weights.config
        base_conv = decode_scratch.layer_conv_states[layer_id]
        base_recurrent = decode_scratch.layer_recurrent_states[layer_id]
        if base_conv is None or base_recurrent is None:
            raise ValueError(f"layer {layer_id} has no linear-attention state")
        conv_row_nbytes = (
            int(self.linear_qkv_width)
            * int(cfg.ssm_conv_kernel)
            * DType.FP32.itemsize
        )
        recurrent_row_nbytes = (
            int(cfg.ssm_time_step_rank)
            * int(cfg.ssm_state_size)
            * int(self.ssm_value_dim)
            * DType.FP32.itemsize
        )
        max_state_index = max(int(index) for index in state_indices)
        if min(int(index) for index in state_indices) < 0:
            raise ValueError("state_indices must be non-negative")
        if int(base_conv.nbytes) < (max_state_index + 1) * conv_row_nbytes:
            raise ValueError("packed Conv state slab is smaller than the requested slot rows")
        if int(base_recurrent.nbytes) < (max_state_index + 1) * recurrent_row_nbytes:
            raise ValueError("packed recurrent state slab is smaller than the requested slot rows")

        use_indexed_batch = rows > 1 and bool(
            getattr(batch_plan, "available", False)
        )
        if use_indexed_batch:
            if gdn_cu_seqlens_ptr is None or state_indices_ptr is None:
                raise ValueError(
                    "indexed linear-attention decode requires segment metadata pointers"
                )
            self._run_linear_attention_attn_rows_indexed_exact(
                layer_id,
                hidden_ptr,
                scratch.attn_out.ptr,
                scratch,
                rows=rows,
                decode_scratch=decode_scratch,
                batch_plan=batch_plan,
                gdn_cu_seqlens_ptr=gdn_cu_seqlens_ptr,
                state_indices_ptr=state_indices_ptr,
                hidden_f32_ptr=hidden_f32_ptr,
                stream=stream,
            )
            execution_path = "indexed_batch"
        else:
            hidden_row_nbytes = int(self.hidden_size) * DType.BF16.itemsize
            hidden_f32_row_nbytes = int(self.hidden_size) * DType.FP32.itemsize
            for row, state_index in enumerate(state_indices):
                conv_states = list(decode_scratch.layer_conv_states)
                recurrent_states = list(decode_scratch.layer_recurrent_states)
                conv_states[layer_id] = DeviceBuffer(
                    int(base_conv.ptr) + int(state_index) * conv_row_nbytes,
                    conv_row_nbytes,
                )
                recurrent_states[layer_id] = DeviceBuffer(
                    int(base_recurrent.ptr) + int(state_index) * recurrent_row_nbytes,
                    recurrent_row_nbytes,
                )
                row_scratch = replace(
                    decode_scratch,
                    layer_conv_states=tuple(conv_states),
                    layer_recurrent_states=tuple(recurrent_states),
                )
                self._run_linear_attention_attn_only(
                    layer_id,
                    int(hidden_ptr) + row * hidden_row_nbytes,
                    int(scratch.attn_out.ptr) + row * hidden_row_nbytes,
                    row_scratch,
                    hidden_f32_ptr=(
                        None
                        if hidden_f32_ptr is None
                        else int(hidden_f32_ptr) + row * hidden_f32_row_nbytes
                    ),
                    stream=stream,
                )
            execution_path = "exact_row_local"

        self._run_post_attention_ffn_rows(
            layer_id,
            hidden_ptr,
            scratch.attn_out.ptr,
            out_ptr,
            scratch,
            rows=rows,
            stream=stream,
            expert_sidecar=expert_sidecar,
            hidden_f32_ptr=hidden_f32_ptr,
            out_f32_ptr=out_f32_ptr,
            stage_timings=stage_timings,
            sync_stage_timings=sync_stage_timings,
            stage_prefix=f"{stage_prefix}_ffn",
        )
        return execution_path

    def _run_linear_attention_decode_rows_native(
        self,
        layer_id: int,
        hidden_ptr: int,
        out_ptr: int,
        scratch,
        *,
        rows: int,
        cu_seqlens_ptr: int,
        state_indices_ptr: int,
        stream: int = 0,
    ) -> str:
        """Run independent decode rows through the Q3 indexed-state contract."""

        assert self.weights is not None
        if rows <= 1:
            raise ValueError("native linear-attention batch requires rows > 1")
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
            rows=rows,
            hidden_size=self.hidden_size,
            eps=cfg.rms_norm_eps,
            stream=stream,
            runtime=runtime,
        )
        if not launch_gguf_linear_pair(
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
        ):
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

        # Keep the scalar dense reduction association on the two rank-16
        # projections; the indexed GDN ABI consumes separate row-major arrays.
        dense_gemv_out_bf16(
            scratch.norm.ptr,
            layer.weight("ssm_alpha").allocation("raw").tensor.ptr,
            scratch.linear_alpha.ptr,
            rows,
            self.hidden_size,
            cfg.ssm_time_step_rank,
            stream=stream,
            runtime=runtime,
        )
        dense_gemv_out_bf16(
            scratch.norm.ptr,
            layer.weight("ssm_beta").allocation("raw").tensor.ptr,
            scratch.linear_beta.ptr,
            rows,
            self.hidden_size,
            cfg.ssm_time_step_rank,
            stream=stream,
            runtime=runtime,
        )
        qwen35_linear_attn_conv_decode_indexed_bf16(
            scratch.linear_qkv.ptr,
            conv_state.ptr,
            layer.weight("ssm_conv1d").allocation().tensor.ptr,
            scratch.conv_out.ptr,
            state_indices_ptr,
            rows,
            self.linear_qkv_width,
            cfg.ssm_conv_kernel,
            stream=stream,
            runtime=runtime,
        )
        qwen35_gdn_recurrent_rmsnorm_gate_segments_lowp_bf16(
            scratch.conv_out.ptr,
            scratch.linear_z.ptr,
            scratch.linear_alpha.ptr,
            scratch.linear_beta.ptr,
            layer.weight("ssm_dt_bias").allocation().tensor.ptr,
            layer.weight("ssm_a").allocation().tensor.ptr,
            layer.weight("ssm_norm").allocation().tensor.ptr,
            recurrent_state.ptr,
            scratch.recurrent_out.ptr,
            cu_seqlens_ptr,
            state_indices_ptr,
            rows,
            rows,
            cfg.rms_norm_eps,
            cfg.ssm_group_count,
            cfg.ssm_time_step_rank,
            cfg.ssm_state_size,
            self.ssm_value_dim,
            stream=stream,
            runtime=runtime,
        )
        f32_to_bf16(
            scratch.recurrent_out.ptr,
            scratch.recurrent_bf16.ptr,
            rows * cfg.ssm_inner_size,
            stream=stream,
            library=self._cast_library(),
            runtime=runtime,
        )
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
        self._run_post_attention_ffn_rows(
            layer_id,
            hidden_ptr,
            scratch.attn_out.ptr,
            out_ptr,
            scratch,
            rows=rows,
            stream=stream,
        )
        return "indexed_conv_gdn"

    def _run_full_attention_decode_rows_native(
        self,
        layer_id: int,
        hidden_ptr: int,
        out_ptr: int,
        scratch,
        *,
        rows: int,
        stream: int = 0,
    ) -> str:
        """Run one compact row-batched full-attention decode layer."""

        assert self.weights is not None
        if rows <= 1:
            raise ValueError("native full-attention batch requires rows > 1")
        if scratch.kv_storage_dtype != DType.BF16:
            raise NotImplementedError("native GGUF target rows currently require BF16 KV")
        layer = self.weights.layer(layer_id)
        cfg = self.weights.config
        runtime = self.runtime or get_hip_runtime()
        cast_library = self._cast_library()
        kv_write_library = self._paged_kv_write_library()
        paged_attn_library = self._paged_attn_decode_library()

        self._run_attention_norm_rows(
            hidden_ptr=hidden_ptr,
            hidden_f32_ptr=None,
            weight_ptr=layer.weight("attn_norm").allocation().tensor.ptr,
            out_ptr=scratch.norm.ptr,
            rows=rows,
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
            scratch.position_tensor.ptr,
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
        key_cache, value_cache = scratch.full_cache(layer_id)
        qwen35_write_paged_kv_mixed_value_bf16_prompt_spans(
            scratch.full_key.ptr,
            scratch.full_v.ptr,
            key_cache.ptr,
            value_cache.ptr,
            scratch.append_spans,
            rows,
            scratch.block_size,
            cfg.head_count_kv,
            cfg.key_length,
            stream=stream,
            library=kv_write_library,
            runtime=runtime,
        )
        max_context_len = int(scratch.decode_spans.max_live_count)
        if max_context_len < _GGUF_FULL_ATTN_DECODE_SPLIT_MIN_CONTEXT_DEFAULT:
            self._full_attn_decode_batch_native_fn()(
                scratch.full_query.ptr,
                key_cache.ptr,
                value_cache.ptr,
                scratch.full_query_raw.ptr,
                scratch.decode_spans,
                rows,
                max_context_len,
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
                scratch.full_query_raw.ptr,
                scratch.full_gate.ptr,
                scratch.full_gated.ptr,
                rows * self.q_width,
                stream=stream,
                library=paged_attn_library,
                runtime=runtime,
            )
            attention_path = "kv_live_spans_batch_c1_exact"
        else:
            chunk_size = int(scratch.block_size)
            num_splits = min(
                int(scratch.full_attn_split_count),
                max(1, (max_context_len + chunk_size - 1) // chunk_size),
            )
            qwen35_paged_full_attn_decode_split_k_gqa_gate_bf16_batch_spans(
                scratch.full_query.ptr,
                key_cache.ptr,
                value_cache.ptr,
                scratch.full_gate.ptr,
                scratch.full_gated.ptr,
                scratch.full_attn_split_partial.ptr,
                scratch.full_attn_split_m.ptr,
                scratch.full_attn_split_l.ptr,
                scratch.decode_spans,
                rows,
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
            attention_path = "kv_live_spans_batch_split_gqa"
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
        )
        return attention_path

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
        hidden_f32_ptr: int | None = None,
        out_f32_ptr: int | None = None,
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
        row_f32_nbytes = self.hidden_size * DType.FP32.itemsize
        runtime = self.runtime or get_hip_runtime()
        for row in range(rows):
            position = start_position + row
            hidden_row = hidden_ptr + row * row_nbytes
            hidden_f32_row = None if hidden_f32_ptr is None else int(hidden_f32_ptr) + row * row_f32_nbytes
            attn_row = scratch.attn_out.ptr + row * row_nbytes
            if layer_type == LINEAR_ATTENTION:
                self._run_linear_attention_attn_only(
                    layer_id,
                    hidden_row,
                    attn_row,
                    decode_scratch,
                    hidden_f32_ptr=hidden_f32_row,
                    stream=stream,
                )
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
                    hidden_f32_ptr=hidden_f32_row,
                    stream=stream,
                )
            else:
                raise ValueError(f"unsupported GGUF layer type {layer_type!r}")
        self._run_post_attention_ffn_rows(
            layer_id,
            hidden_ptr,
            scratch.attn_out.ptr,
            out_ptr,
            scratch,
            rows=rows,
            stream=stream,
            hidden_f32_ptr=hidden_f32_ptr,
            out_f32_ptr=out_f32_ptr,
        )

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
        hidden_f32_ptr: int | None = None,
        out_f32_ptr: int | None = None,
        stage_timings: dict[str, float] | None = None,
        sync_stage_timings: bool = False,
        stage_prefix: str = "target_block_linear_attn",
        gpu_stage_recorder: _HipEventStageRecorder | None = None,
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
        attn_norm_f32_ptr = self._run_attention_norm_rows(
            hidden_ptr=hidden_ptr,
            hidden_f32_ptr=hidden_f32_ptr,
            weight_ptr=layer.weight("attn_norm").allocation().tensor.ptr,
            out_ptr=scratch.norm.ptr,
            out_f32_ptr=(scratch.post_norm_f32.ptr if hasattr(scratch, "post_norm_f32") else None),
            rows=rows,
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
        if gpu_stage_recorder is not None:
            gpu_stage_recorder.mark(f"{stage_prefix}_attn_norm")
        qkv_gate_route = "pair"
        pair_fused = False
        linear_qkv_f32_ready = False
        linear_z_f32_ready = False
        use_f32_linear_projections = (
            attn_norm_f32_ptr is not None
            and _gguf_verify_f32_linear_projections_enabled()
            and hasattr(scratch, "linear_qkv_f32")
            and hasattr(scratch, "linear_z_f32")
        )
        if use_f32_linear_projections:
            pair_fused = _try_launch_dense_q8_pair_dp4a_f32_out(
                layer.weight("attn_qkv"),
                layer.weight("attn_gate"),
                int(attn_norm_f32_ptr),
                scratch.linear_qkv_f32.ptr,
                scratch.linear_z_f32.ptr,
                scratch,
                rows=rows,
                in_features=self.hidden_size,
                out_features_a=self.linear_qkv_width,
                out_features_b=cfg.ssm_inner_size,
                stream=stream,
                runtime=runtime,
            )
            if pair_fused:
                cast_library = self._cast_library()
                f32_to_bf16(
                    scratch.linear_qkv_f32.ptr,
                    scratch.linear_qkv.ptr,
                    rows * self.linear_qkv_width,
                    stream=stream,
                    library=cast_library,
                    runtime=runtime,
                )
                f32_to_bf16(
                    scratch.linear_z_f32.ptr,
                    scratch.linear_z.ptr,
                    rows * cfg.ssm_inner_size,
                    stream=stream,
                    library=cast_library,
                    runtime=runtime,
                )
                qkv_gate_route = "dense_q8_dp4a_f32_out"
                linear_qkv_f32_ready = True
                linear_z_f32_ready = True
        if attn_norm_f32_ptr is not None:
            if not pair_fused:
                pair_fused = _try_launch_dense_q8_pair_dp4a_f32(
                    layer.weight("attn_qkv"),
                    layer.weight("attn_gate"),
                    int(attn_norm_f32_ptr),
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
                if not linear_qkv_f32_ready:
                    qkv_gate_route = "dense_q8_dp4a_f32"
            elif (
                _gguf_linear_supports_f32_activation(layer.weight("attn_qkv"))
                and _gguf_linear_supports_f32_activation(layer.weight("attn_gate"))
            ):
                launch_gguf_linear(
                    layer.weight("attn_qkv"),
                    int(attn_norm_f32_ptr),
                    scratch.linear_qkv.ptr,
                    rows=rows,
                    in_features=self.hidden_size,
                    out_features=self.linear_qkv_width,
                    activation_dtype=GGUF_ACTIVATION_F32,
                    stream=stream,
                    runtime=runtime,
                )
                launch_gguf_linear(
                    layer.weight("attn_gate"),
                    int(attn_norm_f32_ptr),
                    scratch.linear_z.ptr,
                    rows=rows,
                    in_features=self.hidden_size,
                    out_features=cfg.ssm_inner_size,
                    activation_dtype=GGUF_ACTIVATION_F32,
                    stream=stream,
                    runtime=runtime,
                )
                pair_fused = True
                qkv_gate_route = "f32_singletons"
        if not pair_fused:
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
        if not pair_fused:
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
        if use_f32_linear_projections:
            if not linear_qkv_f32_ready:
                bf16_to_f32(
                    scratch.linear_qkv.ptr,
                    scratch.linear_qkv_f32.ptr,
                    rows * self.linear_qkv_width,
                    stream=stream,
                    library=cast_library,
                    runtime=runtime,
                )
                linear_qkv_f32_ready = True
            if not linear_z_f32_ready:
                bf16_to_f32(
                    scratch.linear_z.ptr,
                    scratch.linear_z_f32.ptr,
                    rows * cfg.ssm_inner_size,
                    stream=stream,
                    library=cast_library,
                    runtime=runtime,
                )
                linear_z_f32_ready = True
        if sync_stages:
            runtime.device_synchronize()
            t_now = time.perf_counter()
            qkv_gate_ms = (t_now - t_stage) * 1000
            t_norm_qkv_gate_ms += qkv_gate_ms
            _add_sync_stage_timing(stage_timings, f"{stage_prefix}_attn_qkv_gate_{qkv_gate_route}", qkv_gate_ms)
            _add_sync_stage_timing(stage_timings, f"{stage_prefix}_norm_qkv_gate", t_norm_qkv_gate_ms)
            t_stage = time.perf_counter()
        if gpu_stage_recorder is not None:
            gpu_stage_recorder.mark(
                f"{stage_prefix}_attn_qkv_gate_{qkv_gate_route}",
                f"{stage_prefix}_attn_qkv_gate",
            )
        alpha_beta_route = self._run_linear_attention_alpha_beta_rows(
            layer,
            scratch.norm.ptr,
            attn_norm_f32_ptr,
            scratch,
            rows=rows,
            stream=stream,
            runtime=runtime,
        )
        if (
            use_f32_linear_projections
            and not _linear_attention_alpha_beta_f32_outputs_ready(alpha_beta_route)
            and hasattr(scratch, "linear_alpha_f32")
            and hasattr(scratch, "linear_beta_f32")
        ):
            bf16_to_f32(
                scratch.linear_alpha.ptr,
                scratch.linear_alpha_f32.ptr,
                rows * cfg.ssm_time_step_rank,
                stream=stream,
                library=cast_library,
                runtime=runtime,
            )
            bf16_to_f32(
                scratch.linear_beta.ptr,
                scratch.linear_beta_f32.ptr,
                rows * cfg.ssm_time_step_rank,
                stream=stream,
                library=cast_library,
                runtime=runtime,
            )
        if sync_stages and stage_timings is not None:
            runtime.device_synchronize()
            alpha_beta_ms = (time.perf_counter() - t_stage) * 1000
            _add_sync_stage_timing(stage_timings, f"{stage_prefix}_alpha_beta", alpha_beta_ms)
            _add_sync_stage_timing(stage_timings, f"{stage_prefix}_alpha_beta_{alpha_beta_route}", alpha_beta_ms)
            t_stage = time.perf_counter()
        else:
            t_stage = _mark_sync_stage(
                runtime,
                stage_timings,
                sync_stages,
                f"{stage_prefix}_alpha_beta",
                t_stage,
            )
        if gpu_stage_recorder is not None:
            gpu_stage_recorder.mark(
                f"{stage_prefix}_alpha_beta_{alpha_beta_route}",
                f"{stage_prefix}_alpha_beta",
            )
        active_segments = int(getattr(scratch, "gdn_active_segments", 1))
        if linear_state_rows is not None:
            conv_state_rows, recurrent_state_rows = linear_state_rows
            use_prefill_gdn_capture = _gguf_verify_capture_prefill_gdn_enabled()
            use_prefill_gdn_chain_conv = (
                use_prefill_gdn_capture
                and _gguf_verify_capture_prefill_gdn_chain_conv_enabled()
            )
            use_prefill_score_capture = _gguf_verify_capture_score_prefill_enabled()
            use_f32_chain_conv = (
                _gguf_verify_capture_f32_chain_conv_enabled() or use_prefill_gdn_capture
            )
            if use_prefill_gdn_capture and not use_prefill_gdn_chain_conv:
                if not linear_qkv_f32_ready:
                    bf16_to_f32(
                        scratch.linear_qkv.ptr,
                        scratch.linear_qkv_f32.ptr,
                        rows * self.linear_qkv_width,
                        stream=stream,
                        library=cast_library,
                        runtime=runtime,
                    )
                    linear_qkv_f32_ready = True
                if active_segments > 1:
                    qwen35_linear_attn_conv_prefill_segments_f32_state_rows(
                        scratch.linear_qkv_f32.ptr,
                        conv_state.ptr,
                        conv_state_rows.ptr,
                        layer.weight("ssm_conv1d").allocation().tensor.ptr,
                        scratch.conv_out.ptr,
                        scratch.gdn_cu_seqlens.ptr,
                        scratch.gdn_state_indices.ptr,
                        rows,
                        active_segments,
                        self.linear_qkv_width,
                        cfg.ssm_conv_kernel,
                        stream=stream,
                        runtime=runtime,
                    )
                else:
                    qwen35_linear_attn_conv_prefill_f32_state_rows(
                        scratch.linear_qkv_f32.ptr,
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
            elif use_f32_chain_conv:
                if not linear_qkv_f32_ready:
                    bf16_to_f32(
                        scratch.linear_qkv.ptr,
                        scratch.linear_qkv_f32.ptr,
                        rows * self.linear_qkv_width,
                        stream=stream,
                        library=cast_library,
                        runtime=runtime,
                    )
                    linear_qkv_f32_ready = True
                qwen35_linear_attn_chain_conv_decode_f32_tloop(
                    scratch.linear_qkv_f32.ptr,
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
            else:
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
            chain_conv_stage_name = (
                f"{stage_prefix}_prefill_conv_state_rows"
                if use_prefill_gdn_capture and not use_prefill_gdn_chain_conv
                else (
                    f"{stage_prefix}_prefill_gdn_chain_conv_f32"
                    if use_prefill_gdn_chain_conv
                    else f"{stage_prefix}_chain_conv_f32"
                    if use_f32_chain_conv
                    else f"{stage_prefix}_chain_conv"
                )
            )
            t_stage = _mark_sync_stage(
                runtime,
                stage_timings,
                sync_stages,
                chain_conv_stage_name,
                t_stage,
            )
            if gpu_stage_recorder is not None:
                gpu_stage_recorder.mark(chain_conv_stage_name)
            if use_prefill_gdn_capture:
                if active_segments > 1:
                    qwen35_gdn_prefill_recurrent_rmsnorm_gate_bf16_decode_order_segments_state_rows_no_copy(
                        scratch.conv_out.ptr,
                        scratch.linear_z.ptr,
                        scratch.linear_alpha.ptr,
                        scratch.linear_beta.ptr,
                        layer.weight("ssm_dt_bias").allocation().tensor.ptr,
                        layer.weight("ssm_a").allocation().tensor.ptr,
                        layer.weight("ssm_norm").allocation().tensor.ptr,
                        recurrent_state.ptr,
                        recurrent_state_rows.ptr,
                        scratch.recurrent_bf16.ptr,
                        scratch.gdn_cu_seqlens.ptr,
                        scratch.gdn_state_indices.ptr,
                        cfg.rms_norm_eps,
                        rows,
                        active_segments,
                        cfg.ssm_group_count,
                        cfg.ssm_time_step_rank,
                        cfg.ssm_state_size,
                        self.ssm_value_dim,
                        stream=stream,
                        runtime=runtime,
                    )
                else:
                    qwen35_gdn_prefill_recurrent_rmsnorm_gate_bf16_decode_order_state_rows_no_copy(
                        scratch.conv_out.ptr,
                        scratch.linear_z.ptr,
                        scratch.linear_alpha.ptr,
                        scratch.linear_beta.ptr,
                        layer.weight("ssm_dt_bias").allocation().tensor.ptr,
                        layer.weight("ssm_a").allocation().tensor.ptr,
                        layer.weight("ssm_norm").allocation().tensor.ptr,
                        recurrent_state.ptr,
                        recurrent_state_rows.ptr,
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
                t_stage = _mark_sync_stage(
                    runtime,
                    stage_timings,
                    sync_stages,
                    f"{stage_prefix}_prefill_gdn_state_rows",
                    t_stage,
                )
                if gpu_stage_recorder is not None:
                    gpu_stage_recorder.mark(f"{stage_prefix}_prefill_gdn_state_rows")
            else:
                chain_gdn = (
                    qwen35_gdn_chain_recurrent_rmsnorm_gate_lowp_tloop_bf16
                    if _gguf_verify_capture_regular_chain_gdn_enabled()
                    else qwen35_gdn_chain_recurrent_rmsnorm_gate_lowp_c1_exact_tloop_bf16
                )
                chain_gdn(
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
                chain_gdn_stage_name = (
                    f"{stage_prefix}_chain_gdn_regular"
                    if _gguf_verify_capture_regular_chain_gdn_enabled()
                    else f"{stage_prefix}_chain_gdn"
                )
                t_stage = _mark_sync_stage(
                    runtime,
                    stage_timings,
                    sync_stages,
                    chain_gdn_stage_name,
                    t_stage,
                )
                if gpu_stage_recorder is not None:
                    gpu_stage_recorder.mark(chain_gdn_stage_name)
            prefill_score_ready = False
            if use_prefill_score_capture and not use_prefill_gdn_capture:
                runtime.memcpy_async(
                    scratch.linear_conv_state_tmp.ptr,
                    conv_state.ptr,
                    int(conv_state.nbytes),
                    HipMemcpyKind.DEVICE_TO_DEVICE,
                    stream,
                )
                runtime.memcpy_async(
                    scratch.linear_recurrent_state_tmp.ptr,
                    recurrent_state.ptr,
                    int(recurrent_state.nbytes),
                    HipMemcpyKind.DEVICE_TO_DEVICE,
                    stream,
                )
                if not linear_qkv_f32_ready:
                    bf16_to_f32(
                        scratch.linear_qkv.ptr,
                        scratch.linear_qkv_f32.ptr,
                        rows * self.linear_qkv_width,
                        stream=stream,
                        library=cast_library,
                        runtime=runtime,
                    )
                    linear_qkv_f32_ready = True
                qwen35_linear_attn_conv_prefill_f32(
                    scratch.linear_qkv_f32.ptr,
                    scratch.linear_conv_state_tmp.ptr,
                    layer.weight("ssm_conv1d").allocation().tensor.ptr,
                    scratch.conv_out.ptr,
                    rows,
                    self.linear_qkv_width,
                    cfg.ssm_conv_kernel,
                    stream=stream,
                    runtime=runtime,
                )
                qwen35_gdn_prefill_recurrent_rmsnorm_gate_bf16_decode_order(
                    scratch.conv_out.ptr,
                    scratch.linear_z.ptr,
                    scratch.linear_alpha.ptr,
                    scratch.linear_beta.ptr,
                    layer.weight("ssm_dt_bias").allocation().tensor.ptr,
                    layer.weight("ssm_a").allocation().tensor.ptr,
                    layer.weight("ssm_norm").allocation().tensor.ptr,
                    scratch.linear_recurrent_state_tmp.ptr,
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
                prefill_score_ready = True
                t_stage = _mark_sync_stage(
                    runtime,
                    stage_timings,
                    sync_stages,
                    f"{stage_prefix}_prefill_score",
                    t_stage,
                )
                if gpu_stage_recorder is not None:
                    gpu_stage_recorder.mark(f"{stage_prefix}_prefill_score")
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
                if gpu_stage_recorder is not None:
                    gpu_stage_recorder.mark(f"{stage_prefix}_final_state_copy")
            attn_out_f32_ptr: int | None = None
            f32_attn_out_ready = False
            ssm_out_input_ptr = (
                scratch.recurrent_bf16.ptr
                if (use_prefill_gdn_capture or prefill_score_ready)
                else scratch.recurrent_out.ptr
            )
            ssm_out_activation_dtype = (
                GGUF_ACTIVATION_BF16
                if (use_prefill_gdn_capture or prefill_score_ready)
                else GGUF_ACTIVATION_F32
            )
            if _gguf_verify_capture_bf16_gdn_out_enabled() and not use_prefill_gdn_capture:
                f32_to_bf16(
                    scratch.recurrent_out.ptr,
                    scratch.recurrent_bf16.ptr,
                    rows * cfg.ssm_inner_size,
                    stream=stream,
                    library=cast_library,
                    runtime=runtime,
                )
                ssm_out_input_ptr = scratch.recurrent_bf16.ptr
                ssm_out_activation_dtype = GGUF_ACTIVATION_BF16
                t_stage = _mark_sync_stage(
                    runtime,
                    stage_timings,
                    sync_stages,
                    f"{stage_prefix}_chain_gdn_out_bf16",
                    t_stage,
                )
                if gpu_stage_recorder is not None:
                    gpu_stage_recorder.mark(f"{stage_prefix}_chain_gdn_out_bf16")
            if (
                _gguf_verify_f32_attn_out_enabled()
                and ssm_out_activation_dtype == GGUF_ACTIVATION_F32
                and hidden_f32_ptr is not None
                and out_f32_ptr is not None
                and getattr(scratch, "conv_out", None) is not None
                and int(getattr(scratch.conv_out, "nbytes", 0)) >= rows * self.hidden_size * DType.FP32.itemsize
            ):
                f32_attn_out_ready = _try_launch_dense_q8_single_dp4a_f32_out(
                    layer.weight("ssm_out"),
                    ssm_out_input_ptr,
                    scratch.conv_out.ptr,
                    scratch,
                    rows=rows,
                    in_features=cfg.ssm_inner_size,
                    out_features=self.hidden_size,
                    stream=stream,
                    runtime=runtime,
                )
                if f32_attn_out_ready:
                    attn_out_f32_ptr = scratch.conv_out.ptr
                    f32_to_bf16(
                        int(attn_out_f32_ptr),
                        scratch.attn_out.ptr,
                        rows * self.hidden_size,
                        stream=stream,
                        library=cast_library,
                        runtime=runtime,
                    )
            ssm_out_q8_ready = False
            if not f32_attn_out_ready:
                if ssm_out_activation_dtype == GGUF_ACTIVATION_F32:
                    ssm_out_q8_ready = _try_launch_dense_q8_single_dp4a_f32(
                        layer.weight("ssm_out"),
                        ssm_out_input_ptr,
                        scratch.attn_out.ptr,
                        scratch,
                        rows=rows,
                        in_features=cfg.ssm_inner_size,
                        out_features=self.hidden_size,
                        stream=stream,
                        runtime=runtime,
                    )
                else:
                    ssm_out_q8_ready = _try_launch_dense_q8_single_dp4a(
                        layer.weight("ssm_out"),
                        ssm_out_input_ptr,
                        scratch.attn_out.ptr,
                        scratch,
                        rows=rows,
                        in_features=cfg.ssm_inner_size,
                        out_features=self.hidden_size,
                        stream=stream,
                        runtime=runtime,
                    )
            if not f32_attn_out_ready and not ssm_out_q8_ready:
                launch_gguf_linear(
                    layer.weight("ssm_out"),
                    ssm_out_input_ptr,
                    scratch.attn_out.ptr,
                    rows=rows,
                    in_features=cfg.ssm_inner_size,
                    out_features=self.hidden_size,
                    activation_dtype=ssm_out_activation_dtype,
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
            if gpu_stage_recorder is not None:
                gpu_stage_recorder.mark(f"{stage_prefix}_ssm_out")
            self._run_post_attention_ffn_rows(
                layer_id,
                hidden_ptr,
                scratch.attn_out.ptr,
                out_ptr,
                scratch,
                rows=rows,
                stream=stream,
                expert_sidecar=expert_sidecar,
                hidden_f32_ptr=hidden_f32_ptr,
                out_f32_ptr=out_f32_ptr,
                attn_out_f32_ptr=attn_out_f32_ptr,
                stage_timings=stage_timings,
                sync_stage_timings=sync_stage_timings,
                stage_prefix=f"{stage_prefix}_ffn",
                gpu_stage_recorder=gpu_stage_recorder,
            )
            if gpu_stage_recorder is not None:
                gpu_stage_recorder.mark(f"{stage_prefix}_ffn_total")
            return
        if not linear_qkv_f32_ready:
            bf16_to_f32(
                scratch.linear_qkv.ptr,
                scratch.linear_qkv_f32.ptr,
                rows * self.linear_qkv_width,
                stream=stream,
                library=cast_library,
                runtime=runtime,
            )
            linear_qkv_f32_ready = True
        t_stage = _mark_sync_stage(
            runtime,
            stage_timings,
            sync_stages,
            f"{stage_prefix}_qkv_bf16_to_f32",
            t_stage,
        )
        if gpu_stage_recorder is not None:
            gpu_stage_recorder.mark(f"{stage_prefix}_qkv_bf16_to_f32")
        if active_segments > 1:
            qwen35_linear_attn_conv_prefill_segments_f32(
                scratch.linear_qkv_f32.ptr,
                conv_state.ptr,
                layer.weight("ssm_conv1d").allocation().tensor.ptr,
                scratch.conv_out.ptr,
                scratch.gdn_cu_seqlens.ptr,
                scratch.gdn_state_indices.ptr,
                rows,
                active_segments,
                self.linear_qkv_width,
                cfg.ssm_conv_kernel,
                stream=stream,
                runtime=runtime,
            )
        else:
            self._linear_attn_conv_prefill_kernel()(
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
            f"{stage_prefix}_prefill_conv_segments" if active_segments > 1 else f"{stage_prefix}_prefill_conv",
            t_stage,
        )
        if gpu_stage_recorder is not None:
            gpu_stage_recorder.mark(
                f"{stage_prefix}_prefill_conv_segments" if active_segments > 1 else f"{stage_prefix}_prefill_conv"
            )
        if active_segments > 1:
            qwen35_gdn_prefill_recurrent_rmsnorm_gate_bf16_decode_order_segments(
                scratch.conv_out.ptr,
                scratch.linear_z.ptr,
                scratch.linear_alpha.ptr,
                scratch.linear_beta.ptr,
                layer.weight("ssm_dt_bias").allocation().tensor.ptr,
                layer.weight("ssm_a").allocation().tensor.ptr,
                layer.weight("ssm_norm").allocation().tensor.ptr,
                recurrent_state.ptr,
                scratch.recurrent_bf16.ptr,
                scratch.gdn_cu_seqlens.ptr,
                scratch.gdn_state_indices.ptr,
                cfg.rms_norm_eps,
                rows,
                active_segments,
                cfg.ssm_group_count,
                cfg.ssm_time_step_rank,
                cfg.ssm_state_size,
                self.ssm_value_dim,
                stream=stream,
                runtime=runtime,
            )
        else:
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
            f"{stage_prefix}_prefill_gdn_segments" if active_segments > 1 else f"{stage_prefix}_prefill_gdn",
            t_stage,
        )
        if gpu_stage_recorder is not None:
            gpu_stage_recorder.mark(
                f"{stage_prefix}_prefill_gdn_segments" if active_segments > 1 else f"{stage_prefix}_prefill_gdn"
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
        if gpu_stage_recorder is not None:
            gpu_stage_recorder.mark(f"{stage_prefix}_ssm_out")
        self._run_post_attention_ffn_rows(
            layer_id,
            hidden_ptr,
            scratch.attn_out.ptr,
            out_ptr,
            scratch,
            rows=rows,
            stream=stream,
            expert_sidecar=expert_sidecar,
            hidden_f32_ptr=hidden_f32_ptr,
            out_f32_ptr=out_f32_ptr,
            stage_timings=stage_timings,
            sync_stage_timings=sync_stage_timings,
            stage_prefix=f"{stage_prefix}_ffn",
            gpu_stage_recorder=gpu_stage_recorder,
        )
        if gpu_stage_recorder is not None:
            gpu_stage_recorder.mark(f"{stage_prefix}_ffn_total")

    def _run_full_attention_layer(
        self,
        layer_id: int,
        hidden_ptr: int,
        out_ptr: int,
        scratch,
        *,
        position: int,
        max_context_len: int | None = None,
        input_norm_ptr: int | None = None,
        next_norm_weight_ptr: int | None = None,
        next_norm_out_ptr: int | None = None,
        stream: int = 0,
        attention_max_context_len: int | None = None,
    ) -> None:
        if max_context_len is None:
            max_context_len = attention_max_context_len
        elif attention_max_context_len is not None and int(max_context_len) != int(attention_max_context_len):
            raise ValueError("conflicting full-attention context limits")
        self._run_full_attention_attn_only(
            layer_id,
            hidden_ptr,
            scratch.attn_out.ptr,
            scratch,
            position=position,
            input_norm_ptr=input_norm_ptr,
            stream=stream,
            attention_max_context_len=max_context_len,
        )
        self._run_post_attention_ffn(
            layer_id,
            hidden_ptr,
            scratch.attn_out.ptr,
            out_ptr,
            scratch,
            next_norm_weight_ptr=next_norm_weight_ptr,
            next_norm_out_ptr=next_norm_out_ptr,
            stream=stream,
        )

    def _run_full_attention_attn_only(
        self,
        layer_id: int,
        hidden_ptr: int,
        attn_out_ptr: int,
        scratch,
        *,
        position: int,
        hidden_f32_ptr: int | None = None,
        input_norm_ptr: int | None = None,
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
        if input_norm_ptr is None:
            self._run_attention_norm_rows(
                hidden_ptr=hidden_ptr,
                hidden_f32_ptr=hidden_f32_ptr,
                weight_ptr=layer.weight("attn_norm").allocation().tensor.ptr,
                out_ptr=scratch.norm.ptr,
                rows=1,
                eps=cfg.rms_norm_eps,
                stream=stream,
                runtime=runtime,
            )
        elif int(input_norm_ptr) != int(scratch.norm.ptr):
            raise ValueError("prefused full-attention input norm must use scratch.norm")
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
        append_spans = scratch.append_spans_for_layer(layer_id)
        decode_spans = scratch.decode_spans_for_layer(layer_id)
        paged_attn_library = self._paged_attn_decode_library()
        bf16_mirror_cache = None
        full_bf16_mirror_cache = getattr(scratch, "full_bf16_mirror_cache", None)
        if full_bf16_mirror_cache is not None:
            bf16_mirror_cache = full_bf16_mirror_cache(layer_id)
        layer_uses_int8_kv = scratch.kv_storage_dtype == DType.INT8_PER_TOKEN_HEAD and append_spans.scale_metadata is not None
        if layer_uses_int8_kv:
            metadata = append_spans.scale_metadata
            bf16_to_f32(
                scratch.full_v.ptr,
                scratch.full_key_raw.ptr,
                self.kv_width,
                stream=stream,
                library=cast_library,
                runtime=runtime,
            )
            if getattr(scratch, "int8_kv_value_bf16", False):
                qwen35_write_paged_kv_int8_key_bf16_value_spans(
                    scratch.full_key.ptr,
                    scratch.full_key_raw.ptr,
                    key_cache.ptr,
                    value_cache.ptr,
                    metadata.k_scale.ptr,
                    append_spans,
                    scratch.block_size,
                    cfg.head_count_kv,
                    cfg.key_length,
                    stream=stream,
                    library=kv_write_library,
                    runtime=runtime,
                )
            else:
                int8_append_write_fn = _gguf_int8_kv_append_write_fn(metadata)
                int8_append_write_fn(
                    scratch.full_key.ptr,
                    scratch.full_key_raw.ptr,
                    key_cache.ptr,
                    value_cache.ptr,
                    metadata.k_scale.ptr,
                    metadata.v_scale.ptr,
                    append_spans,
                    scratch.block_size,
                    cfg.head_count_kv,
                    cfg.key_length,
                    stream=stream,
                    library=kv_write_library,
                    runtime=runtime,
                )
            if bf16_mirror_cache is not None:
                mirror_key_cache, mirror_value_cache = bf16_mirror_cache
                qwen35_write_paged_kv_mixed_value_bf16_spans(
                    scratch.full_key.ptr,
                    scratch.full_v.ptr,
                    mirror_key_cache.ptr,
                    mirror_value_cache.ptr,
                    scratch.append_spans,
                    scratch.block_size,
                    cfg.head_count_kv,
                    cfg.key_length,
                    stream=stream,
                    library=kv_write_library,
                    runtime=runtime,
                )
        else:
            qwen35_write_paged_kv_mixed_value_bf16_spans(
                scratch.full_key.ptr,
                scratch.full_v.ptr,
                key_cache.ptr,
                value_cache.ptr,
                append_spans,
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
        if layer_uses_int8_kv and bf16_mirror_cache is None:
            metadata = decode_spans.scale_metadata
            if metadata is None:
                raise RuntimeError("GGUF INT8 full-attention decode requires scale metadata")
            chunk_size = int(scratch.block_size)
            num_splits = min(
                int(scratch.full_attn_split_count),
                max(1, (attention_context_cap + chunk_size - 1) // chunk_size),
            )
            if getattr(scratch, "int8_kv_value_bf16", False):
                qwen35_paged_attn_decode_int8_key_bf16_value_gqa_splitk_gate_bf16_spans(
                    scratch.full_query.ptr,
                    key_cache.ptr,
                    value_cache.ptr,
                    metadata.k_scale.ptr,
                    scratch.full_gate.ptr,
                    scratch.full_gated.ptr,
                    scratch.full_attn_split_partial.ptr,
                    scratch.full_attn_split_m.ptr,
                    scratch.full_attn_split_l.ptr,
                    decode_spans,
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
                int8_decode_gate_fn = _gguf_int8_kv_decode_gate_fn(metadata)
                int8_decode_gate_fn(
                    scratch.full_query.ptr,
                    key_cache.ptr,
                    value_cache.ptr,
                    metadata.k_scale.ptr,
                    metadata.v_scale.ptr,
                    scratch.full_gate.ptr,
                    scratch.full_gated.ptr,
                    scratch.full_attn_split_partial.ptr,
                    scratch.full_attn_split_m.ptr,
                    scratch.full_attn_split_l.ptr,
                    decode_spans,
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
            if bf16_mirror_cache is not None:
                key_cache, value_cache = bf16_mirror_cache
                decode_spans = scratch.decode_spans
            if _use_gguf_full_attention_split_decode(attention_context_cap):
                chunk_size = int(scratch.block_size)
                num_splits = min(
                    int(scratch.full_attn_split_count),
                    max(1, (attention_context_cap + chunk_size - 1) // chunk_size),
                )
                split_gate_fn = _gguf_full_attention_split_gate_bf16_fn(
                    cfg,
                    backend=self.backend,
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
                    decode_spans,
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
                    decode_spans,
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

    def _run_full_attention_output_rows(
        self,
        layer,
        gated_ptr: int,
        attn_out_ptr: int,
        scratch,
        *,
        rows: int,
        hidden_f32_ptr: int | None,
        out_f32_ptr: int | None,
        stream: int,
        runtime: HipRuntime,
    ) -> int | None:
        attn_out_f32_ptr: int | None = None
        if (
            _gguf_verify_f32_attn_out_enabled()
            and hidden_f32_ptr is not None
            and out_f32_ptr is not None
            and hasattr(scratch, "post_norm_f32")
        ):
            candidate_ptr = int(scratch.post_norm_f32.ptr)
            if _try_launch_gguf_linear_bf16_f32_output(
                layer.weight("attn_output"),
                gated_ptr,
                candidate_ptr,
                rows=rows,
                in_features=self.q_width,
                out_features=self.hidden_size,
                stream=stream,
                runtime=runtime,
            ):
                f32_to_bf16(
                    candidate_ptr,
                    attn_out_ptr,
                    rows * self.hidden_size,
                    stream=stream,
                    library=self._cast_library(),
                    runtime=runtime,
                )
                attn_out_f32_ptr = candidate_ptr
        if attn_out_f32_ptr is None:
            launch_gguf_linear(
                layer.weight("attn_output"),
                gated_ptr,
                attn_out_ptr,
                rows=rows,
                in_features=self.q_width,
                out_features=self.hidden_size,
                stream=stream,
                runtime=runtime,
            )
        return attn_out_f32_ptr

    def _run_post_attention_ffn(
        self,
        layer_id: int,
        hidden_ptr: int,
        attn_out_ptr: int,
        out_ptr: int,
        scratch,
        *,
        next_norm_weight_ptr: int | None = None,
        next_norm_out_ptr: int | None = None,
        stream: int = 0,
    ) -> None:
        self._run_post_attention_ffn_rows(
            layer_id,
            hidden_ptr,
            attn_out_ptr,
            out_ptr,
            scratch,
            rows=1,
            next_norm_weight_ptr=next_norm_weight_ptr,
            next_norm_out_ptr=next_norm_out_ptr,
            stream=stream,
        )

    def _run_post_attention_ffn_rows(
        self,
        layer_id: int,
        hidden_ptr: int,
        attn_out_ptr: int,
        out_ptr: int,
        scratch,
        *,
        rows: int,
        next_norm_weight_ptr: int | None = None,
        next_norm_out_ptr: int | None = None,
        stream: int = 0,
        expert_sidecar: _DeviceExpertLayerSidecar | None = None,
        hidden_f32_ptr: int | None = None,
        out_f32_ptr: int | None = None,
        attn_out_f32_ptr: int | None = None,
        stage_timings: dict[str, float] | None = None,
        sync_stage_timings: bool = False,
        stage_prefix: str = "target_block_ffn",
        gpu_stage_recorder: _HipEventStageRecorder | None = None,
    ) -> None:
        assert self.weights is not None
        layer = self.weights.layer(layer_id)
        runtime = self.runtime or get_hip_runtime()
        sync_stages = bool(sync_stage_timings and stage_timings is not None)
        t_stage = time.perf_counter() if sync_stages else 0.0
        f32_residual = hidden_f32_ptr is not None or out_f32_ptr is not None
        post_norm_f32_ptr: int | None = None
        if f32_residual:
            if hidden_f32_ptr is None or out_f32_ptr is None:
                raise ValueError("hidden_f32_ptr and out_f32_ptr must be provided together")
            post_attention_norm_ptr = layer.weight("post_attention_norm").allocation().tensor.ptr
            if attn_out_f32_ptr is not None and _gguf_verify_f32_attn_out_enabled():
                gguf_add_rmsnorm_f32_f32_f32_weight(
                    int(hidden_f32_ptr),
                    int(attn_out_f32_ptr),
                    post_attention_norm_ptr,
                    scratch.post_norm.ptr,
                    int(out_f32_ptr),
                    rows=rows,
                    hidden_size=self.hidden_size,
                    eps=self.weights.config.rms_norm_eps,
                    stream=stream,
                    runtime=runtime,
                )
            else:
                gguf_add_rmsnorm_f32_bf16_f32_weight(
                    int(hidden_f32_ptr),
                    attn_out_ptr,
                    post_attention_norm_ptr,
                    scratch.post_norm.ptr,
                    int(out_f32_ptr),
                    rows=rows,
                    hidden_size=self.hidden_size,
                    eps=self.weights.config.rms_norm_eps,
                    stream=stream,
                    runtime=runtime,
                )
            if _gguf_verify_f32_post_norm_enabled():
                if not hasattr(scratch, "post_norm_f32"):
                    raise ValueError("scratch is missing post_norm_f32 for verifier F32 post-norm diagnostic")
                gguf_rmsnorm_f32_f32_weight_out_f32(
                    int(out_f32_ptr),
                    post_attention_norm_ptr,
                    scratch.post_norm_f32.ptr,
                    rows=rows,
                    hidden_size=self.hidden_size,
                    eps=self.weights.config.rms_norm_eps,
                    stream=stream,
                    runtime=runtime,
                )
                post_norm_f32_ptr = scratch.post_norm_f32.ptr
        else:
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
        if gpu_stage_recorder is not None:
            gpu_stage_recorder.mark(f"{stage_prefix}_post_norm_residual")
        if self.weights.config.is_moe:
            if rows == 1:
                self._run_post_attention_moe_c1(
                    layer_id,
                    out_ptr,
                    scratch,
                    stream=stream,
                    residual_f32_ptr=out_f32_ptr if f32_residual else None,
                    out_f32_ptr=out_f32_ptr if f32_residual else None,
                    post_norm_f32_ptr=post_norm_f32_ptr,
                    next_norm_weight_ptr=next_norm_weight_ptr,
                    next_norm_out_ptr=next_norm_out_ptr,
                )
            else:
                self._run_post_attention_moe_rows(
                    layer_id,
                    out_ptr,
                    scratch,
                    rows=rows,
                    stream=stream,
                    expert_sidecar=expert_sidecar,
                    residual_f32_ptr=out_f32_ptr if f32_residual else None,
                    out_f32_ptr=out_f32_ptr if f32_residual else None,
                    post_norm_f32_ptr=post_norm_f32_ptr,
                    stage_timings=stage_timings,
                    sync_stage_timings=sync_stage_timings,
                    stage_prefix=f"{stage_prefix}_moe",
                    gpu_stage_recorder=gpu_stage_recorder,
                )
            return
        if next_norm_weight_ptr is not None:
            raise ValueError("MoE-tail next RMSNorm fusion requires an MoE layer")
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
        if gpu_stage_recorder is not None:
            gpu_stage_recorder.mark(f"{stage_prefix}_gate_up")
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
        if gpu_stage_recorder is not None:
            gpu_stage_recorder.mark(f"{stage_prefix}_silu")
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
        if f32_residual:
            count = rows * self.hidden_size
            gguf_f32_bf16_add_out_f32(
                int(out_f32_ptr),
                scratch.ffn_down.ptr,
                int(out_f32_ptr),
                count,
                stream=stream,
                runtime=runtime,
            )
            f32_to_bf16(
                int(out_f32_ptr),
                out_ptr,
                count,
                stream=stream,
                library=self._cast_library(),
                runtime=runtime,
            )
        else:
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
        if gpu_stage_recorder is not None:
            gpu_stage_recorder.mark(f"{stage_prefix}_down_residual")

    def _run_post_attention_moe_c1(
        self,
        layer_id: int,
        out_ptr: int,
        scratch,
        *,
        stream: int = 0,
        residual_f32_ptr: int | None = None,
        out_f32_ptr: int | None = None,
        post_norm_f32_ptr: int | None = None,
        next_norm_weight_ptr: int | None = None,
        next_norm_out_ptr: int | None = None,
    ) -> None:
        assert self.weights is not None
        cfg = self.weights.config
        if (next_norm_weight_ptr is None) != (next_norm_out_ptr is None):
            raise ValueError("next norm weight and output pointers must be provided together")
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
        router_f32_ptr = (
            post_norm_f32_ptr
            if post_norm_f32_ptr is not None and _gguf_verify_f32_post_norm_router_enabled()
            else None
        )
        selected_f32_ptr = (
            post_norm_f32_ptr
            if post_norm_f32_ptr is not None and _gguf_verify_f32_post_norm_selected_q8_enabled()
            else None
        )
        shared_f32_ptr = (
            post_norm_f32_ptr
            if post_norm_f32_ptr is not None and _gguf_verify_f32_post_norm_shared_q8_enabled()
            else None
        )
        router_fn = (
            _launch_qwen35_router_logits_f32_hidden
            if router_f32_ptr is not None
            else _launch_qwen35_router_logits_bf16_hidden
        )
        router_hidden_ptr = int(router_f32_ptr) if router_f32_ptr is not None else scratch.post_norm.ptr
        router_fused = (
            router_f32_ptr is None
            and _gguf_router_f32w_coop_enabled()
            and _try_launch_qwen35_router_topk_split_shared_bf16_f32w(
                router_hidden_ptr,
                layer.weight("ffn_gate_inp"),
                layer.weight("ffn_gate_inp_shexp"),
                scratch.moe_router_logits.ptr,
                scratch.moe_selected_experts.ptr,
                scratch.moe_routing_weights.ptr,
                scratch.moe_router_counter.ptr,
                hidden_size=self.hidden_size,
                num_experts=cfg.expert_count,
                top_k=top_k,
                persistent_counter=_gguf_router_f32w_persistent_counter_enabled(),
                stream=stream,
                runtime=runtime,
            )
        )
        if not router_fused:
            router_fn(
                router_hidden_ptr,
                layer.weight("ffn_gate_inp"),
                scratch.moe_router_logits.ptr,
                1,
                self.hidden_size,
                cfg.expert_count,
                stream=stream,
                runtime=runtime,
            )
            router_fn(
                router_hidden_ptr,
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
        f32_residual = residual_f32_ptr is not None or out_f32_ptr is not None
        if f32_residual and (residual_f32_ptr is None or out_f32_ptr is None):
            raise ValueError("residual_f32_ptr and out_f32_ptr must be provided together")

        if (
            not f32_residual
            and _env_flag(_GGUF_COMPACT_MOE_C1_ENV, False)
            and _try_run_post_attention_moe_c1_compact_gemv(
                self,
                layer,
                gate_weight,
                up_weight,
                down_weight,
                out_ptr,
                scratch,
                top_k=top_k,
                next_norm_weight_ptr=next_norm_weight_ptr,
                next_norm_out_ptr=next_norm_out_ptr,
                stream=stream,
                runtime=runtime,
            )
        ):
            return
        selected_rows = top_k
        selected_down_is_f32 = False
        expert_down_weighted = False
        prefer_f32_selected_down = _gguf_use_f32_selected_down(down_weight, scratch, f32_residual)
        if not (
            (not prefer_f32_selected_down)
            and _env_flag(_GGUF_FUSED_MOE_FFN_ENV, False)
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
            selected_down_is_f32, expert_down_weighted = self._run_post_attention_moe_c1_unfused_selected_ffn(
                gate_weight,
                up_weight,
                down_weight,
                scratch,
                selected_rows=selected_rows,
                post_norm_f32_ptr=selected_f32_ptr,
                prefer_f32_selected_down=prefer_f32_selected_down,
                stream=stream,
                runtime=runtime,
            )

        if _try_launch_shared_gate_up_from_f32_post_norm(
            layer.weight("ffn_gate_shexp"),
            layer.weight("ffn_up_shexp"),
            shared_f32_ptr,
            scratch.moe_shared_gate.ptr,
            scratch.moe_shared_up.ptr,
            rows=1,
            hidden_size=self.hidden_size,
            shared_ffn=cfg.expert_shared_feed_forward_length,
            stream=stream,
            runtime=runtime,
        ):
            silu_mul_separate_out_bf16(
                scratch.moe_shared_gate.ptr,
                scratch.moe_shared_up.ptr,
                scratch.moe_shared_intermediate.ptr,
                rows=1,
                features=cfg.expert_shared_feed_forward_length,
                stream=stream,
                runtime=runtime,
            )
        elif launch_gguf_linear_pair_concat(
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
        shared_down_is_f32 = False
        if (
            _gguf_use_f32_shared_down(scratch, f32_residual, selected_down_is_f32)
            and _try_launch_gguf_linear_bf16_f32_output(
                layer.weight("ffn_down_shexp"),
                scratch.moe_shared_intermediate.ptr,
                scratch.moe_shared_out_f32.ptr,
                rows=1,
                in_features=cfg.expert_shared_feed_forward_length,
                out_features=self.hidden_size,
                stream=stream,
                runtime=runtime,
            )
        ):
            f32_to_bf16(
                scratch.moe_shared_out_f32.ptr,
                scratch.moe_shared_out.ptr,
                self.hidden_size,
                stream=stream,
                library=self._cast_library(),
                runtime=runtime,
            )
            shared_down_is_f32 = True
        else:
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
        if f32_residual:
            if selected_down_is_f32 and shared_down_is_f32:
                weighted_sum_f32_shared_f32_gate_combine_residual_out_f32_accum_f32w(
                    scratch.moe_down_out_f32.ptr,
                    scratch.moe_routing_weights.ptr,
                    scratch.moe_shared_out_f32.ptr,
                    scratch.moe_router_logits.ptr + cfg.expert_count * 4,
                    int(residual_f32_ptr),
                    int(out_f32_ptr),
                    top_k,
                    self.hidden_size,
                    stream=stream,
                    runtime=runtime,
                )
            elif selected_down_is_f32:
                weighted_sum_f32_shared_gate_combine_residual_out_f32_accum_f32w(
                    scratch.moe_down_out_f32.ptr,
                    scratch.moe_routing_weights.ptr,
                    scratch.moe_shared_out.ptr,
                    scratch.moe_router_logits.ptr + cfg.expert_count * 4,
                    int(residual_f32_ptr),
                    int(out_f32_ptr),
                    top_k,
                    self.hidden_size,
                    stream=stream,
                    runtime=runtime,
                )
            else:
                _gguf_f32_moe_combine_out_fn()(
                    scratch.moe_down_out.ptr,
                    scratch.moe_routing_weights.ptr,
                    scratch.moe_shared_out.ptr,
                    scratch.moe_router_logits.ptr + cfg.expert_count * 4,
                    int(residual_f32_ptr),
                    int(out_f32_ptr),
                    top_k,
                    self.hidden_size,
                    stream=stream,
                    runtime=runtime,
                )
            f32_to_bf16(
                int(out_f32_ptr),
                out_ptr,
                self.hidden_size,
                stream=stream,
                library=self._cast_library(),
                runtime=runtime,
            )
            if next_norm_weight_ptr is not None:
                gguf_rmsnorm_bf16_f32_weight(
                    out_ptr,
                    next_norm_weight_ptr,
                    next_norm_out_ptr,
                    rows=1,
                    hidden_size=self.hidden_size,
                    eps=cfg.rms_norm_eps,
                    stream=stream,
                    runtime=runtime,
                )
        else:
            if expert_down_weighted:
                if next_norm_weight_ptr is None:
                    shared_gate_combine_residual_out_bf16(
                        scratch.moe_down_out.ptr,
                        scratch.moe_shared_out.ptr,
                        scratch.moe_router_logits.ptr + cfg.expert_count * 4,
                        scratch.residual.ptr,
                        out_ptr,
                        self.hidden_size,
                        stream=stream,
                        runtime=runtime,
                    )
                else:
                    shared_gate_combine_residual_rmsnorm_gguf_bf16_out(
                        scratch.moe_down_out.ptr,
                        scratch.moe_shared_out.ptr,
                        scratch.moe_router_logits.ptr + cfg.expert_count * 4,
                        scratch.residual.ptr,
                        next_norm_weight_ptr,
                        next_norm_out_ptr,
                        out_ptr,
                        1,
                        self.hidden_size,
                        1,
                        eps=cfg.rms_norm_eps,
                        stream=stream,
                        runtime=runtime,
                    )
            else:
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
                if next_norm_weight_ptr is not None:
                    gguf_rmsnorm_bf16_f32_weight(
                        out_ptr,
                        next_norm_weight_ptr,
                        next_norm_out_ptr,
                        rows=1,
                        hidden_size=self.hidden_size,
                        eps=cfg.rms_norm_eps,
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
        post_norm_f32_ptr: int | None = None,
        prefer_f32_selected_down: bool = False,
        stream: int,
        runtime: HipRuntime,
    ) -> tuple[bool, bool]:
        """Unfused selected-expert FFN: gate_up GEMV -> silu*mul -> down GEMV into
        ``scratch.moe_down_out``. Numerically-equivalent fallback for the fused B1
        megakernel (architectural invariant), and the default rows==1 path."""
        cfg = self.weights.config
        gate_rows_nbytes = selected_rows * cfg.expert_feed_forward_length * DType.BF16.itemsize
        f32_selected_intermediate = _gguf_use_f32_selected_intermediate(
            scratch,
            bool(prefer_f32_selected_down),
        )
        expert_silu_ready = False
        if post_norm_f32_ptr is None and not f32_selected_intermediate:
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
                x_f32_ptr=post_norm_f32_ptr,
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
                    x_f32_ptr=post_norm_f32_ptr,
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
                    x_f32_ptr=post_norm_f32_ptr,
                    stream=stream,
                    runtime=runtime,
                )
            if f32_selected_intermediate:
                silu_mul_separate_out_f32(
                    scratch.ffn_gate_up.ptr,
                    scratch.ffn_gate_up.ptr + gate_rows_nbytes,
                    scratch.ffn_intermediate_f32.ptr,
                    rows=selected_rows,
                    features=cfg.expert_feed_forward_length,
                    stream=stream,
                    runtime=runtime,
                )
                f32_to_bf16(
                    scratch.ffn_intermediate_f32.ptr,
                    scratch.ffn_intermediate.ptr,
                    selected_rows * cfg.expert_feed_forward_length,
                    stream=stream,
                    library=self._cast_library(),
                    runtime=runtime,
                )
            else:
                silu_mul_separate_out_bf16(
                    scratch.ffn_gate_up.ptr,
                    scratch.ffn_gate_up.ptr + gate_rows_nbytes,
                    scratch.ffn_intermediate.ptr,
                    rows=selected_rows,
                    features=cfg.expert_feed_forward_length,
                    stream=stream,
                    runtime=runtime,
                )
        f32_selected_down = bool(prefer_f32_selected_down)
        expert_down_weighted = False
        if not f32_selected_down:
            expert_down_weighted = _launch_weighted_selected_raw_gguf_moe_linear(
                down_weight,
                scratch.ffn_intermediate.ptr,
                scratch.moe_selected_experts.ptr,
                scratch.moe_routing_weights.ptr,
                scratch.moe_down_out.ptr,
                tokens=1,
                top_k=selected_rows,
                num_experts=cfg.expert_count,
                in_features=cfg.expert_feed_forward_length,
                out_features=self.hidden_size,
                stream=stream,
                runtime=runtime,
            )
        if not expert_down_weighted:
            _launch_selected_raw_gguf_moe_linear(
                down_weight,
                scratch.ffn_intermediate.ptr,
                scratch.moe_selected_experts.ptr,
                scratch.moe_down_out_f32.ptr if f32_selected_down else scratch.moe_down_out.ptr,
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
                x_f32_ptr=scratch.ffn_intermediate_f32.ptr if f32_selected_intermediate else None,
                prefer_f32_out=f32_selected_down,
                backend=self.backend,
                stream=stream,
                runtime=runtime,
            )
        return f32_selected_down, expert_down_weighted

    def _run_post_attention_moe_rows(
        self,
        layer_id: int,
        out_ptr: int,
        scratch,
        *,
        rows: int,
        stream: int = 0,
        expert_sidecar: _DeviceExpertLayerSidecar | None = None,
        residual_f32_ptr: int | None = None,
        out_f32_ptr: int | None = None,
        post_norm_f32_ptr: int | None = None,
        stage_timings: dict[str, float] | None = None,
        sync_stage_timings: bool = False,
        stage_prefix: str = "target_block_ffn_moe",
        gpu_stage_recorder: _HipEventStageRecorder | None = None,
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
        f32_residual = residual_f32_ptr is not None or out_f32_ptr is not None
        if f32_residual and (residual_f32_ptr is None or out_f32_ptr is None):
            raise ValueError("residual_f32_ptr and out_f32_ptr must be provided together")

        router_f32_ptr = (
            post_norm_f32_ptr
            if post_norm_f32_ptr is not None and _gguf_verify_f32_post_norm_router_enabled()
            else None
        )
        selected_f32_ptr = (
            post_norm_f32_ptr
            if post_norm_f32_ptr is not None and _gguf_verify_f32_post_norm_selected_q8_enabled()
            else None
        )
        shared_f32_ptr = (
            post_norm_f32_ptr
            if post_norm_f32_ptr is not None and _gguf_verify_f32_post_norm_shared_q8_enabled()
            else None
        )
        router_fn = (
            _launch_qwen35_router_logits_f32_hidden
            if router_f32_ptr is not None
            else _launch_qwen35_router_logits_bf16_hidden
        )
        router_hidden_ptr = int(router_f32_ptr) if router_f32_ptr is not None else scratch.post_norm.ptr
        router_fn(
            router_hidden_ptr,
            layer.weight("ffn_gate_inp"),
            scratch.moe_router_logits.ptr,
            rows,
            self.hidden_size,
            cfg.expert_count,
            stream=stream,
            runtime=runtime,
        )
        router_fn(
            router_hidden_ptr,
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
            threads=_gguf_prefill_router_select_threads(self.backend),
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
        if gpu_stage_recorder is not None:
            gpu_stage_recorder.mark(f"{stage_prefix}_router")

        gate_weight = layer.weight("ffn_gate_exps")
        up_weight = layer.weight("ffn_up_exps")
        down_weight = layer.weight("ffn_down_exps")
        prefer_f32_selected_down = (
            False
            if expert_sidecar is not None
            else _gguf_use_f32_selected_down(down_weight, scratch, f32_residual)
        )
        f32_selected_intermediate = _gguf_use_f32_selected_intermediate(
            scratch,
            prefer_f32_selected_down,
        )
        if (not f32_residual) and _try_run_post_attention_moe_rows_compact_wmma(
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
            gpu_stage_recorder=gpu_stage_recorder,
            stage_prefix=f"{stage_prefix}_compact_wmma",
        ):
            _mark_sync_stage(
                runtime,
                stage_timings,
                sync_stages,
                f"{stage_prefix}_compact_wmma",
                t_stage,
            )
            return
        if (
            (not f32_residual)
            and _gguf_row_compact_gemv_enabled()
            and _try_run_post_attention_moe_rows_compact_gemv(
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
            )
        ):
            _mark_sync_stage(
                runtime,
                stage_timings,
                sync_stages,
                f"{stage_prefix}_compact_gemv",
                t_stage,
            )
            if gpu_stage_recorder is not None:
                gpu_stage_recorder.mark(f"{stage_prefix}_compact_gemv")
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
            backend=self.backend,
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
                backend=self.backend,
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
                backend=self.backend,
                x_rows=rows,
                rows=selected_rows,
                num_experts=cfg.expert_count,
                in_features=self.hidden_size,
                out_features=cfg.expert_feed_forward_length,
                stream=stream,
                runtime=runtime,
                library=getattr(self, "_expert_pack8_library", None),
            )
        elif _launch_selected_raw_gguf_moe_pair_silu(
            gate_weight,
            up_weight,
            scratch.post_norm.ptr,
            scratch.moe_selected_experts.ptr,
            scratch.ffn_intermediate.ptr,
            x_rows=rows,
            rows=selected_rows,
            num_experts=cfg.expert_count,
            in_features=self.hidden_size,
            out_features=cfg.expert_feed_forward_length,
            stream=stream,
            runtime=runtime,
            allow_legacy=False,
        ):
            expert_silu_ready = True
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
                x_f32_ptr=selected_f32_ptr,
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
                    x_f32_ptr=selected_f32_ptr,
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
                    x_f32_ptr=selected_f32_ptr,
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
        if gpu_stage_recorder is not None:
            gpu_stage_recorder.mark(f"{stage_prefix}_expert_gate_up")
        if not expert_silu_ready:
            if f32_selected_intermediate:
                silu_mul_separate_out_f32(
                    scratch.ffn_gate_up.ptr,
                    scratch.ffn_gate_up.ptr + gate_rows_nbytes,
                    scratch.ffn_intermediate_f32.ptr,
                    rows=selected_rows,
                    features=cfg.expert_feed_forward_length,
                    stream=stream,
                    runtime=runtime,
                )
                f32_to_bf16(
                    scratch.ffn_intermediate_f32.ptr,
                    scratch.ffn_intermediate.ptr,
                    selected_rows * cfg.expert_feed_forward_length,
                    stream=stream,
                    library=self._cast_library(),
                    runtime=runtime,
                )
            else:
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
            if gpu_stage_recorder is not None:
                gpu_stage_recorder.mark(f"{stage_prefix}_expert_silu")
        selected_down_is_f32 = False
        expert_down_weighted = False
        if expert_sidecar is not None:
            _launch_selected_expert_pack8_moe_linear(
                expert_sidecar.tensor("ffn_down_exps"),
                scratch.ffn_intermediate.ptr,
                scratch.moe_selected_experts.ptr,
                scratch.moe_down_out.ptr,
                backend=self.backend,
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
            selected_down_is_f32 = prefer_f32_selected_down
            if not selected_down_is_f32 and not f32_residual:
                expert_down_weighted = _launch_weighted_selected_raw_gguf_moe_linear(
                    down_weight,
                    scratch.ffn_intermediate.ptr,
                    scratch.moe_selected_experts.ptr,
                    scratch.moe_routing_weights.ptr,
                    scratch.moe_down_out.ptr,
                    tokens=rows,
                    top_k=top_k,
                    num_experts=cfg.expert_count,
                    in_features=cfg.expert_feed_forward_length,
                    out_features=self.hidden_size,
                    stream=stream,
                    runtime=runtime,
                )
            if not expert_down_weighted:
                _launch_selected_raw_gguf_moe_linear(
                    down_weight,
                    scratch.ffn_intermediate.ptr,
                    scratch.moe_selected_experts.ptr,
                    scratch.moe_down_out_f32.ptr if selected_down_is_f32 else scratch.moe_down_out.ptr,
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
                    x_f32_ptr=scratch.ffn_intermediate_f32.ptr if f32_selected_intermediate else None,
                    prefer_f32_out=selected_down_is_f32,
                    backend=self.backend,
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
        if gpu_stage_recorder is not None:
            gpu_stage_recorder.mark(f"{stage_prefix}_expert_down")

        shared_q8_dp4a_enabled = _gguf_dense_q8_dp4a_shared_enabled()
        if shared_q8_dp4a_enabled and shared_f32_ptr is not None:
            shared_gate_up_dp4a = _try_launch_dense_q8_pair_dp4a_f32(
                layer.weight("ffn_gate_shexp"),
                layer.weight("ffn_up_shexp"),
                int(shared_f32_ptr),
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
        else:
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
        elif _try_launch_shared_gate_up_from_f32_post_norm(
            layer.weight("ffn_gate_shexp"),
            layer.weight("ffn_up_shexp"),
            shared_f32_ptr,
            scratch.moe_shared_gate.ptr,
            scratch.moe_shared_up.ptr,
            rows=rows,
            hidden_size=self.hidden_size,
            shared_ffn=cfg.expert_shared_feed_forward_length,
            stream=stream,
            runtime=runtime,
        ):
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
        if gpu_stage_recorder is not None:
            gpu_stage_recorder.mark(f"{stage_prefix}_shared_gate_up_silu")
        shared_down_is_f32 = False
        if (
            _gguf_use_f32_shared_down(scratch, f32_residual, selected_down_is_f32)
            and _try_launch_gguf_linear_bf16_f32_output(
                layer.weight("ffn_down_shexp"),
                scratch.moe_shared_intermediate.ptr,
                scratch.moe_shared_out_f32.ptr,
                rows=rows,
                in_features=cfg.expert_shared_feed_forward_length,
                out_features=self.hidden_size,
                stream=stream,
                runtime=runtime,
            )
        ):
            f32_to_bf16(
                scratch.moe_shared_out_f32.ptr,
                scratch.moe_shared_out.ptr,
                rows * self.hidden_size,
                stream=stream,
                library=self._cast_library(),
                runtime=runtime,
            )
            shared_down_is_f32 = True
        elif not (
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
        if gpu_stage_recorder is not None:
            gpu_stage_recorder.mark(f"{stage_prefix}_shared_down")
        if f32_residual:
            if selected_down_is_f32 and shared_down_is_f32:
                weighted_sum_f32_shared_f32_gate_combine_residual_batch_out_f32_accum_f32w(
                    scratch.moe_down_out_f32.ptr,
                    scratch.moe_routing_weights.ptr,
                    scratch.moe_shared_out_f32.ptr,
                    scratch.moe_shared_gate_logits.ptr,
                    int(residual_f32_ptr),
                    int(out_f32_ptr),
                    rows,
                    top_k,
                    self.hidden_size,
                    1,
                    stream=stream,
                    runtime=runtime,
                )
            elif selected_down_is_f32:
                weighted_sum_f32_shared_gate_combine_residual_batch_out_f32_accum_f32w(
                    scratch.moe_down_out_f32.ptr,
                    scratch.moe_routing_weights.ptr,
                    scratch.moe_shared_out.ptr,
                    scratch.moe_shared_gate_logits.ptr,
                    int(residual_f32_ptr),
                    int(out_f32_ptr),
                    rows,
                    top_k,
                    self.hidden_size,
                    1,
                    stream=stream,
                    runtime=runtime,
                )
            else:
                _gguf_f32_moe_combine_batch_out_fn()(
                    scratch.moe_down_out.ptr,
                    scratch.moe_routing_weights.ptr,
                    scratch.moe_shared_out.ptr,
                    scratch.moe_shared_gate_logits.ptr,
                    int(residual_f32_ptr),
                    int(out_f32_ptr),
                    rows,
                    top_k,
                    self.hidden_size,
                    1,
                    stream=stream,
                    runtime=runtime,
                )
            f32_to_bf16(
                int(out_f32_ptr),
                out_ptr,
                rows * self.hidden_size,
                stream=stream,
                library=self._cast_library(),
                runtime=runtime,
            )
        else:
            if expert_down_weighted:
                shared_gate_combine_residual_batch_out_bf16(
                    scratch.moe_down_out.ptr,
                    scratch.moe_shared_out.ptr,
                    scratch.moe_shared_gate_logits.ptr,
                    scratch.residual.ptr,
                    out_ptr,
                    rows,
                    self.hidden_size,
                    1,
                    stream=stream,
                    runtime=runtime,
                )
            else:
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
        if gpu_stage_recorder is not None:
            gpu_stage_recorder.mark(f"{stage_prefix}_combine_residual")

    def allocate_scratch(
        self,
        *,
        max_sequence_length: int | None = None,
        max_batch_size: int = 1,
    ) -> "_FullStackScratch":
        """Allocate batch-shaped state/KV scratch for AR or draft execution."""

        return _FullStackScratch.allocate(
            self,
            runtime=self.runtime or get_hip_runtime(),
            max_sequence_length=max_sequence_length,
            max_batch_size=max_batch_size,
        )

    def close(self) -> None:
        if self.weights is not None:
            if self.owns_resident_weights:
                self.weights.free(runtime=self.runtime)
            self.weights = None

    def __enter__(self) -> "Qwen35GGUFFullStackRunner":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


_QWEN35MOE_UNSAFE_FASTPATH_ENV = "HIPENGINE_GGUF_ALLOW_UNSAFE_QWEN35MOE_FASTPATHS"
_GGUF_AOTRITON_PREFILL_ENV = "HIPENGINE_GGUF_AOTRITON_PREFILL"
_GGUF_AOTRITON_HEAD_MAJOR_KV_ENV = "HIPENGINE_GGUF_AOTRITON_HEAD_MAJOR_KV"
_GGUF_AOTRITON_HEAD_MAJOR_KV_MAX_BYTES_ENV = (
    "HIPENGINE_GGUF_AOTRITON_HEAD_MAJOR_KV_MAX_BYTES"
)
_GGUF_AOTRITON_HEAD_MAJOR_KV_MAX_TOKENS_ENV = (
    "HIPENGINE_GGUF_AOTRITON_HEAD_MAJOR_KV_MAX_TOKENS"
)
_GGUF_AOTRITON_HEAD_MAJOR_KV_MAX_BYTES_DEFAULT = 512 * 1024 * 1024
_GGUF_AOTRITON_HEAD_MAJOR_KV_MAX_TOKENS_DEFAULT = 65_792
_GGUF_FULL_ATTN_DECODE_SPLIT_MIN_CONTEXT_ENV = "HIPENGINE_GGUF_FULL_ATTN_DECODE_PAGED_MIN_CONTEXT"
_GGUF_FULL_ATTN_DECODE_SPLIT_MIN_CONTEXT_DEFAULT = 1024
_GGUF_FULL_ATTN_PREFILL_SPLIT_BATCH_ROWS = 16
_GGUF_COMPACT_MOE_C1_ENV = "HIPENGINE_GGUF_COMPACT_MOE_C1"
# Keep explicit INT8-KV short gates on the exact BF16 decode path. Long-context
# sweeps after the layer-local BF16 prefill-oracle fix found that prefix 8/10
# full-attention layers passes 128K/128 while prefix 7 fails a 128K/16 top-1
# guard. Prefix 8 is therefore the admitted GGUF Q4_K_M long-context hybrid.
_GGUF_INT8_SHORT_BF16_MIRROR_MAX_POSITIONS = 8192
_GGUF_INT8_LONG_BF16_PREFIX_FULL_ATTENTION_LAYERS = 8
_GGUF_INT8_BF16_PREFIX_FULL_ATTENTION_ENV = "HIPENGINE_GGUF_INT8_KV_BF16_PREFIX_FULL_LAYERS"
_GGUF_INT8_BF16_FULL_ATTENTION_LAYERS_ENV = "HIPENGINE_GGUF_INT8_KV_BF16_FULL_LAYERS"
_GGUF_INT8_ALLOW_UNVERIFIED_LONG_ENV = "HIPENGINE_GGUF_INT8_KV_ALLOW_UNVERIFIED_LONG"
_GGUF_INT8_KV_KEY_ONLY_ENV = "HIPENGINE_GGUF_INT8_KV_KEY_ONLY"
_GGUF_INT8_KV_BLOCK16_ENV = "HIPENGINE_GGUF_INT8_KV_BLOCK16"
# B2: opt-in fused selected-expert MoE FFN megakernel for rows==1 raw-Q4_K decode.
_GGUF_FUSED_MOE_FFN_ENV = "HIPENGINE_GGUF_FUSED_MOE_FFN"
_GGUF_HOST_TOKEN_EMBEDDING_ENV = "HIPENGINE_GGUF_HOST_TOKEN_EMBEDDING"
_GGUF_IQ_GROUPED_PREFILL_ENV = "HIPENGINE_GGUF_IQ_GROUPED_PREFILL"
_GGUF_MOE_TAIL_NEXT_RMS_ENV = "HIPENGINE_GGUF_MOE_TAIL_NEXT_RMS"
_GGUF_Q4K_SELECTED_DUAL_DP4A_ENV = "HIPENGINE_GGUF_Q4K_SELECTED_DUAL_DP4A"
_GGUF_T16_SELECTED_DP4A_ENV = "HIPENGINE_GGUF_T16_SELECTED_DP4A"
_GGUF_T16_SELECTED_PAIRREUSE_ENV = "HIPENGINE_GGUF_T16_SELECTED_PAIRREUSE"
_GGUF_T16_SELECTED_DOWN_PAIRREUSE_ENV = "HIPENGINE_GGUF_T16_SELECTED_DOWN_PAIRREUSE"
_GGUF_T16_SELECTED_Q6_DOWN_PAIRREUSE_ENV = "HIPENGINE_GGUF_T16_SELECTED_Q6_DOWN_PAIRREUSE"
_gguf_t16_selected_pairreuse_min_rows_session: int | None = None
_gguf_t16_selected_down_pairreuse_min_rows_session: int | None = None
_gguf_t16_selected_q6_down_pairreuse_min_rows_session: int | None = None
_GGUF_T16_DS4_PREFILL_ENV = "HIPENGINE_GGUF_T16_DS4_PREFILL"
_GGUF_RAW_SELECTED_DP4A_ENV = "HIPENGINE_GGUF_RAW_SELECTED_DP4A"
_GGUF_DENSE_Q8_DP4A_ENV = "HIPENGINE_GGUF_DENSE_Q8_DP4A"
_GGUF_DENSE_Q8_DP4A_ALL_ENV = "HIPENGINE_GGUF_DENSE_Q8_DP4A_ALL"
_GGUF_DENSE_Q8_DP4A_SHARED_ENV = "HIPENGINE_GGUF_DENSE_Q8_DP4A_SHARED"
_GGUF_DENSE_Q8_DP4A_F32_ENV = "HIPENGINE_GGUF_DENSE_Q8_DP4A_F32"
_GGUF_ROW_COMPACT_GEMV_ENV = "HIPENGINE_GGUF_ROW_COMPACT_GEMV"
_GGUF_VERIFY_ROW_LM_HEAD_ENV = "HIPENGINE_GGUF_VERIFY_ROW_LM_HEAD"
_GGUF_VERIFY_LM_HEAD_Q6_TOP1_DP4A_ENV = "HIPENGINE_GGUF_VERIFY_LM_HEAD_Q6_TOP1_DP4A"
_GGUF_VERIFY_F32_RESIDUAL_ENV = "HIPENGINE_GGUF_VERIFY_F32_RESIDUAL"
_GGUF_VERIFY_F32_TOKEN_EMBEDDING_ENV = "HIPENGINE_GGUF_VERIFY_F32_TOKEN_EMBEDDING"
_GGUF_VERIFY_F32_ATTENTION_NORM_ENV = "HIPENGINE_GGUF_VERIFY_F32_ATTENTION_NORM"
_GGUF_VERIFY_F32_LINEAR_PROJECTIONS_ENV = "HIPENGINE_GGUF_VERIFY_F32_LINEAR_PROJECTIONS"
_GGUF_VERIFY_F32_ALPHA_BETA_ENV = "HIPENGINE_GGUF_VERIFY_F32_ALPHA_BETA"
_GGUF_VERIFY_F32_ATTN_OUT_ENV = "HIPENGINE_GGUF_VERIFY_F32_ATTN_OUT"
_GGUF_VERIFY_F32_MOE_COMBINE_ENV = "HIPENGINE_GGUF_VERIFY_F32_MOE_COMBINE"
_GGUF_VERIFY_F32_SELECTED_DOWN_ENV = "HIPENGINE_GGUF_VERIFY_F32_SELECTED_DOWN"
_GGUF_VERIFY_F32_SELECTED_INTERMEDIATE_ENV = "HIPENGINE_GGUF_VERIFY_F32_SELECTED_INTERMEDIATE"
_GGUF_VERIFY_F32_SHARED_DOWN_ENV = "HIPENGINE_GGUF_VERIFY_F32_SHARED_DOWN"
_GGUF_VERIFY_F32_POST_NORM_ENV = "HIPENGINE_GGUF_VERIFY_F32_POST_NORM"
_GGUF_VERIFY_F32_POST_NORM_ROUTER_ENV = "HIPENGINE_GGUF_VERIFY_F32_POST_NORM_ROUTER"
_GGUF_ROUTER_F32W_COOP_ENV = "HIPENGINE_GGUF_ROUTER_F32W_COOP"
_GGUF_ROUTER_F32W_PERSISTENT_COUNTER_ENV = "HIPENGINE_GGUF_ROUTER_F32W_PERSISTENT_COUNTER"
_GGUF_VERIFY_F32_POST_NORM_SELECTED_Q8_ENV = "HIPENGINE_GGUF_VERIFY_F32_POST_NORM_SELECTED_Q8"
_GGUF_VERIFY_F32_POST_NORM_SHARED_Q8_ENV = "HIPENGINE_GGUF_VERIFY_F32_POST_NORM_SHARED_Q8"
_GGUF_VERIFY_CAPTURE_F32_CHAIN_CONV_ENV = "HIPENGINE_GGUF_VERIFY_CAPTURE_F32_CHAIN_CONV"
_GGUF_VERIFY_CAPTURE_REGULAR_CHAIN_GDN_ENV = "HIPENGINE_GGUF_VERIFY_CAPTURE_REGULAR_CHAIN_GDN"
_GGUF_VERIFY_CAPTURE_BF16_GDN_OUT_ENV = "HIPENGINE_GGUF_VERIFY_CAPTURE_BF16_GDN_OUT"
_GGUF_VERIFY_CAPTURE_PREFILL_GDN_ENV = "HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN"
_GGUF_VERIFY_CAPTURE_PREFILL_GDN_CHAIN_CONV_ENV = "HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN_CHAIN_CONV"
_GGUF_VERIFY_CAPTURE_SCORE_PREFILL_ENV = "HIPENGINE_GGUF_VERIFY_CAPTURE_SCORE_PREFILL"
_GGUF_PACKED_VERIFY_GPU_STAGE_TIMINGS_ENV = "HIPENGINE_GGUF_PACKED_VERIFY_GPU_STAGE_TIMINGS"
_GGUF_COMPACT_WMMA_NO_READ_MAX_SELECTED_ROWS_ENV = "HIPENGINE_GGUF_COMPACT_WMMA_NO_READ_MAX_SELECTED_ROWS"
_GGUF_MOE_GRAPH_ENV = "HIPENGINE_GGUF_MOE_GRAPH"
_GGUF_PREFILL_DEVICE_METADATA_ENV = "HIPENGINE_GGUF_PREFILL_DEVICE_METADATA"
_GGUF_PREFILL_ROUTER_SELECT_THREADS_ENV = "HIPENGINE_GGUF_PREFILL_ROUTER_SELECT_THREADS"
_QWEN35_AOTRITON_ISOLATED_PREFILL_STREAM_ENV = "HIPENGINE_QWEN35_AOTRITON_ISOLATED_PREFILL_STREAM"
_GGUF_LINEAR_ATTN_CONV_PREFILL_MODE_ENV = "HIPENGINE_GGUF_LINEAR_ATTN_CONV_PREFILL_MODE"
_GGUF_LINEAR_ATTN_CONV_PREFILL_MODES = frozenset({"baseline", "tile32x128"})
_GGUF_AOTRITON_ISOLATED_PREFILL_MIN_QUERY_ROWS = 512
_GGUF_TOKEN_EMBEDDING_TENSOR = "token_embd.weight"
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


def _normalize_prefill_queue_drain(value: str) -> str:
    mode = str(value).strip().lower()
    if mode not in {"none", "chunk", "layer"}:
        raise ValueError(
            f"prefill_queue_drain must be one of none, chunk, layer; got {value!r}"
        )
    return mode


def _iq_grouped_prefill_enabled() -> bool:
    return _env_flag(_GGUF_IQ_GROUPED_PREFILL_ENV, True)


def _gguf_moe_tail_next_rms_enabled() -> bool:
    return _env_flag(_GGUF_MOE_TAIL_NEXT_RMS_ENV, True)


def _gguf_prefill_router_select_threads(backend: str) -> int:
    raw = _env_value(_GGUF_PREFILL_ROUTER_SELECT_THREADS_ENV)
    value = (
        backend_package_capability(
            backend,
            "GGUF_PREFILL_ROUTER_SELECT_THREADS",
            512,
        )
        if raw is None
        else raw
    )
    try:
        threads = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"prefill router-select threads must be one of 64, 128, 256, 512, got {value!r}"
        ) from exc
    if threads not in {64, 128, 256, 512}:
        raise ValueError(
            f"prefill router-select threads must be one of 64, 128, 256, 512, got {threads}"
        )
    return threads


def _gguf_prefill_device_metadata_enabled(
    *,
    backend: str | None = None,
    prompt_tokens: int | None = None,
) -> bool:
    raw = _env_value(_GGUF_PREFILL_DEVICE_METADATA_ENV)
    if raw is not None:
        return _env_flag(_GGUF_PREFILL_DEVICE_METADATA_ENV, False)
    if backend is None or prompt_tokens is None:
        return False
    try:
        max_tokens = int(
            backend_package_capability(
                backend,
                "GGUF_PREFILL_DEVICE_METADATA_MAX_TOKENS",
                0,
            )
        )
    except (TypeError, ValueError):
        return False
    return max_tokens > 0 and 0 < int(prompt_tokens) <= max_tokens


def _gguf_linear_attn_conv_prefill_mode(backend: str) -> str:
    """Resolve the explicit or architecture-scoped GGUF convolution schedule."""

    raw = _env_value(_GGUF_LINEAR_ATTN_CONV_PREFILL_MODE_ENV)
    mode = "auto" if raw is None else raw.strip().lower()
    if mode == "auto":
        mode = str(
            backend_package_capability(
                backend,
                "GGUF_LINEAR_ATTN_CONV_PREFILL_AUTO_MODE",
                "baseline",
            )
        ).strip().lower()
    if mode not in _GGUF_LINEAR_ATTN_CONV_PREFILL_MODES:
        valid = ", ".join(sorted(_GGUF_LINEAR_ATTN_CONV_PREFILL_MODES))
        raise ValueError(
            "unsupported GGUF linear-attention convolution prefill mode "
            f"{mode!r}; expected auto, {valid}"
        )
    return mode


def _gguf_q8_t16_two_wave_prefill_applies(backend: str, prompt_tokens: int) -> bool:
    """Resolve the backend package's request-scoped GPF-5A ceiling."""

    raw = backend_package_capability(
        backend,
        "GGUF_Q8_T16_PREFILL_TWO_WAVE_MAX_TOKENS",
        0,
    )
    try:
        max_tokens = int(raw)
    except (TypeError, ValueError):
        return False
    return max_tokens > 0 and 0 < int(prompt_tokens) <= max_tokens


def _gguf_aotriton_isolated_prefill_stream_applies(backend: str, query_rows: int) -> bool:
    if int(query_rows) < _GGUF_AOTRITON_ISOLATED_PREFILL_MIN_QUERY_ROWS:
        return False
    if _env_value(_QWEN35_AOTRITON_ISOLATED_PREFILL_STREAM_ENV) is not None:
        return _env_flag(_QWEN35_AOTRITON_ISOLATED_PREFILL_STREAM_ENV, False)
    return bool(
        backend_package_capability(
            backend,
            "GGUF_AOTRITON_ISOLATED_PREFILL_STREAM",
            False,
        )
    )


def _env_int(name: str, default: int, *aliases: str) -> int:
    raw = _env_value(name, *aliases)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _gguf_aotriton_head_major_kv_enabled(backend: str) -> bool:
    """Resolve the architecture package default with an explicit rollback."""

    if _env_value(_GGUF_AOTRITON_HEAD_MAJOR_KV_ENV) is not None:
        return _env_flag(_GGUF_AOTRITON_HEAD_MAJOR_KV_ENV, False)
    return bool(
        backend_package_capability(
            backend,
            "GGUF_AOTRITON_HEAD_MAJOR_KV",
            False,
        )
    )


def _try_allocate_gguf_aotriton_head_major_kv_scratch(
    *,
    backend: str,
    capacity_tokens: int,
    kv_width: int,
    runtime: HipRuntime,
) -> tuple[DeviceBuffer, DeviceBuffer] | None:
    """Best-effort tracked allocation for one cross-layer BF16 K/V pair.

    The gfx1151 default is bounded to the validated 64K allocation class.
    Token/byte capacity or HIP allocation denial is an exact fallback signal,
    not a session-construction failure. If the second allocation fails, the
    first is released before returning.
    """

    if not _gguf_aotriton_head_major_kv_enabled(backend):
        return None
    capacity_tokens = int(capacity_tokens)
    kv_width = int(kv_width)
    if capacity_tokens <= 0 or kv_width <= 0:
        raise ValueError("head-major KV scratch dimensions must be positive")
    max_tokens = _env_int(
        _GGUF_AOTRITON_HEAD_MAJOR_KV_MAX_TOKENS_ENV,
        _GGUF_AOTRITON_HEAD_MAJOR_KV_MAX_TOKENS_DEFAULT,
    )
    if max_tokens <= 0 or capacity_tokens > max_tokens:
        return None
    per_buffer_bytes = capacity_tokens * kv_width * DType.BF16.itemsize
    total_bytes = 2 * per_buffer_bytes
    max_bytes = _env_int(
        _GGUF_AOTRITON_HEAD_MAJOR_KV_MAX_BYTES_ENV,
        _GGUF_AOTRITON_HEAD_MAJOR_KV_MAX_BYTES_DEFAULT,
    )
    if max_bytes <= 0 or total_bytes > max_bytes:
        return None
    key_buffer: DeviceBuffer | None = None
    try:
        key_buffer = malloc(per_buffer_bytes, runtime=runtime)
        value_buffer = malloc(per_buffer_bytes, runtime=runtime)
    except (HipError, MemoryError):
        if key_buffer is not None:
            free(key_buffer, runtime=runtime)
        return None
    return key_buffer, value_buffer


def _gguf_aotriton_head_major_buffers(
    scratch,
    *,
    context_len: int,
) -> tuple[object, object, int] | None:
    """Return an admitted complete scratch pair or select the strided fallback."""

    if not bool(getattr(scratch, "head_major_kv_admitted", False)):
        return None
    key_buffer = getattr(scratch, "head_major_key_cache", None)
    value_buffer = getattr(scratch, "head_major_value_cache", None)
    capacity = int(getattr(scratch, "head_major_kv_capacity", 0))
    if key_buffer is None or value_buffer is None or capacity < int(context_len):
        return None
    return key_buffer, value_buffer, capacity


def _gguf_host_token_embedding_requested() -> bool:
    return _env_flag(_GGUF_HOST_TOKEN_EMBEDDING_ENV, False)


def _gguf_int8_kv_value_bf16_enabled(*, kv_storage_dtype: DType) -> bool:
    return kv_storage_dtype == DType.INT8_PER_TOKEN_HEAD and _env_flag(_GGUF_INT8_KV_KEY_ONLY_ENV, False)


def _gguf_int8_kv_scale_granularity(
    *,
    kv_storage_dtype: DType,
    requested_granularity: str = "per_token_head",
) -> str:
    requested = str(requested_granularity or "per_token_head").strip().lower()
    if requested not in {"per_token_head", "block16", "hadamard_group32"}:
        raise ValueError(
            "GGUF resident INT8 KV scale granularity must be per_token_head, block16, or hadamard_group32"
        )
    if kv_storage_dtype != DType.INT8_PER_TOKEN_HEAD:
        return "per_token_head"
    if _env_flag(_GGUF_INT8_KV_BLOCK16_ENV, False):
        return "block16"
    return requested


def _gguf_int8_bf16_prefix_full_attention_layers(*, kv_storage_dtype: DType, max_positions: int) -> int:
    if kv_storage_dtype != DType.INT8_PER_TOKEN_HEAD:
        return 0
    if int(max_positions) <= _GGUF_INT8_SHORT_BF16_MIRROR_MAX_POSITIONS:
        return 0
    allow_unverified = _env_flag(_GGUF_INT8_ALLOW_UNVERIFIED_LONG_ENV, False)
    default_prefix = 0 if allow_unverified else _GGUF_INT8_LONG_BF16_PREFIX_FULL_ATTENTION_LAYERS
    return max(
        0,
        _env_int(
            _GGUF_INT8_BF16_PREFIX_FULL_ATTENTION_ENV,
            default_prefix,
        ),
    )


def _parse_gguf_int8_full_attention_layer_indices(raw: str, *, full_attention_layers: int) -> tuple[int, ...]:
    value = raw.strip().lower()
    count = int(full_attention_layers)
    if count < 0:
        raise ValueError("full_attention_layers must be non-negative")
    if value in {"none", "empty", "-"}:
        return ()
    if value == "all":
        return tuple(range(count))
    indices: set[int] = set()
    for item in value.replace(";", ",").split(","):
        token = item.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start = int(start_text.strip())
            end = int(end_text.strip())
            if end < start:
                raise ValueError(f"invalid GGUF INT8 BF16 full-layer range {token!r}")
            indices.update(range(start, end + 1))
        else:
            indices.add(int(token))
    if not indices:
        return ()
    bad = sorted(idx for idx in indices if idx < 0 or idx >= count)
    if bad:
        raise ValueError(
            f"{_GGUF_INT8_BF16_FULL_ATTENTION_LAYERS_ENV} index/indices {bad} outside [0, {count})"
        )
    return tuple(sorted(indices))


def _gguf_int8_bf16_full_attention_layer_indices(
    *,
    kv_storage_dtype: DType,
    max_positions: int,
    full_attention_layers: int,
) -> tuple[int, ...]:
    if kv_storage_dtype != DType.INT8_PER_TOKEN_HEAD:
        return ()
    if int(max_positions) <= _GGUF_INT8_SHORT_BF16_MIRROR_MAX_POSITIONS:
        return ()
    override = _env_value(_GGUF_INT8_BF16_FULL_ATTENTION_LAYERS_ENV)
    if override is not None:
        return _parse_gguf_int8_full_attention_layer_indices(
            override,
            full_attention_layers=int(full_attention_layers),
        )
    prefix = _gguf_int8_bf16_prefix_full_attention_layers(
        kv_storage_dtype=kv_storage_dtype,
        max_positions=max_positions,
    )
    return tuple(range(min(max(0, int(prefix)), int(full_attention_layers))))


def _gguf_int8_effective_scale_dtype(
    *,
    kv_storage_dtype: DType,
    max_positions: int,
    requested_scale_dtype: DType,
    bf16_prefix_full_attention_layers: int,
    bf16_full_attention_layer_count: int | None = None,
    scale_granularity: str = "per_token_head",
) -> DType:
    hybrid_bf16_layers = (
        int(bf16_prefix_full_attention_layers)
        if bf16_full_attention_layer_count is None
        else int(bf16_full_attention_layer_count)
    )
    if (
        kv_storage_dtype == DType.INT8_PER_TOKEN_HEAD
        and int(max_positions) > _GGUF_INT8_SHORT_BF16_MIRROR_MAX_POSITIONS
        and hybrid_bf16_layers > 0
        and scale_granularity != "hadamard_group32"
    ):
        return DType.FP32
    return requested_scale_dtype


def _validate_gguf_int8_kv_context(
    *,
    kv_storage_dtype: DType,
    max_positions: int,
    bf16_prefix_full_attention_layers: int = 0,
    bf16_full_attention_layer_indices: tuple[int, ...] | None = None,
    storage_layout: str = "uniform",
) -> None:
    if kv_storage_dtype != DType.INT8_PER_TOKEN_HEAD:
        return
    if storage_layout == "tail4_hadamard_group32":
        return
    if storage_layout != "uniform":
        raise ValueError(f"unsupported GGUF resident KV storage layout {storage_layout!r}")
    if int(max_positions) <= _GGUF_INT8_SHORT_BF16_MIRROR_MAX_POSITIONS:
        return
    if _env_flag(_GGUF_INT8_ALLOW_UNVERIFIED_LONG_ENV, False):
        return
    admitted_prefix = tuple(range(_GGUF_INT8_LONG_BF16_PREFIX_FULL_ATTENTION_LAYERS))
    if bf16_full_attention_layer_indices is not None:
        if tuple(sorted(int(idx) for idx in bf16_full_attention_layer_indices)) == admitted_prefix:
            return
        raise ValueError(
            "GGUF int8_per_token_head KV custom BF16 full-attention layer sets are diagnostic-only. "
            f"Set {_GGUF_INT8_ALLOW_UNVERIFIED_LONG_ENV}=1 to test {_GGUF_INT8_BF16_FULL_ATTENTION_LAYERS_ENV}."
        )
    if int(bf16_prefix_full_attention_layers) >= _GGUF_INT8_LONG_BF16_PREFIX_FULL_ATTENTION_LAYERS:
        return
    raise ValueError(
        "GGUF int8_per_token_head KV is correctness-admitted for long contexts only with "
        f"at least {_GGUF_INT8_LONG_BF16_PREFIX_FULL_ATTENTION_LAYERS} BF16-prefix full-attention layers; "
        f"got rounded max context {int(max_positions)} and prefix={int(bf16_prefix_full_attention_layers)}. "
        "Prefixes below that failed BF16-vs-hybrid long-context logit gates and are diagnostic-only. Set "
        f"{_GGUF_INT8_ALLOW_UNVERIFIED_LONG_ENV}=1 only to reproduce blocked capacity diagnostics."
    )


def _gguf_int8_kv_prompt_write_fn(metadata: KVScaleMetadata):
    granularity = getattr(metadata, "granularity", "per_token_head")
    if granularity == "block16":
        return qwen35_write_paged_kv_int8_block16_prompt_spans
    if granularity == "per_token_head":
        return qwen35_write_paged_kv_int8_per_token_head_prompt_spans
    if granularity == "hadamard_group32":
        return qwen35_write_paged_kv_int8_hadamard_group32_prompt_spans
    raise ValueError(
        "GGUF INT8 KV scale granularity must be per_token_head, block16, or hadamard_group32"
    )


def _gguf_int8_kv_append_write_fn(metadata: KVScaleMetadata):
    granularity = getattr(metadata, "granularity", "per_token_head")
    if granularity == "block16":
        return qwen35_write_paged_kv_int8_block16_spans
    if granularity == "per_token_head":
        return qwen35_write_paged_kv_int8_per_token_head_spans
    if granularity == "hadamard_group32":
        return qwen35_write_paged_kv_int8_hadamard_group32_spans
    raise ValueError(
        "GGUF INT8 KV scale granularity must be per_token_head, block16, or hadamard_group32"
    )


def _gguf_int8_kv_decode_gate_fn(metadata: KVScaleMetadata):
    granularity = getattr(metadata, "granularity", "per_token_head")
    if granularity == "block16":
        return qwen35_paged_attn_decode_int8_block16_gqa_splitk_gate_bf16_spans
    if granularity == "per_token_head":
        return qwen35_paged_attn_decode_int8_gqa_splitk_gate_bf16_spans
    if granularity == "hadamard_group32":
        return qwen35_paged_attn_decode_int8_hadamard_group32_gqa_splitk_gate_bf16_spans
    raise ValueError(
        "GGUF INT8 KV scale granularity must be per_token_head, block16, or hadamard_group32"
    )


def _q8_0_embedding_rows_to_bf16(
    raw_embedding: np.ndarray,
    token_ids: np.ndarray,
    *,
    hidden_size: int,
    cache: dict[int, np.ndarray] | None = None,
) -> np.ndarray:
    """Dequantize selected Q8_0 token-embedding rows to BF16 bits on host."""

    raw = np.asarray(raw_embedding)
    if raw.ndim != 2:
        raise ValueError(f"Q8_0 token embedding raw bytes must be rank-2, got shape {raw.shape}")
    hidden = int(hidden_size)
    if hidden <= 0 or hidden % 32 != 0:
        raise ValueError(f"hidden_size must be a positive multiple of 32, got {hidden_size}")
    blocks_per_row = hidden // 32
    expected_row_bytes = blocks_per_row * 34
    if int(raw.shape[1]) != expected_row_bytes:
        raise ValueError(
            f"Q8_0 token embedding row bytes mismatch: expected {expected_row_bytes}, got {raw.shape[1]}"
        )

    tokens = np.asarray(token_ids, dtype=np.int64).reshape(-1)
    if tokens.size == 0:
        return np.empty((0, hidden), dtype=np.uint16)
    min_token = int(tokens.min())
    max_token = int(tokens.max())
    if min_token < 0 or max_token >= int(raw.shape[0]):
        raise ValueError(f"token_id outside [0, {raw.shape[0]}): min={min_token}, max={max_token}")

    row_cache = cache if cache is not None else {}

    def one_row(token: int) -> np.ndarray:
        cached = row_cache.get(token)
        if cached is not None:
            return cached
        blocks = np.asarray(raw[token], dtype=np.uint8).reshape(blocks_per_row, 34)
        scales = blocks[:, :2].copy().view(np.float16).astype(np.float32).reshape(blocks_per_row, 1)
        q = blocks[:, 2:].view(np.int8).astype(np.float32)
        bf16 = float_array_to_bf16_bits((q * scales).reshape(hidden))
        row_cache[token] = bf16
        return bf16

    out = np.empty((int(tokens.size), hidden), dtype=np.uint16)
    unique, inverse = np.unique(tokens, return_inverse=True)
    if unique.size == 1:
        out[...] = one_row(int(unique[0]))
        return out
    for index, token in enumerate(unique.tolist()):
        out[inverse == index] = one_row(int(token))
    return out


def _gguf_q4k_selected_dual_dp4a_enabled() -> bool:
    return _env_flag(_GGUF_Q4K_SELECTED_DUAL_DP4A_ENV, False)


def _gguf_t16_selected_dp4a_enabled() -> bool:
    return _env_flag(_GGUF_T16_SELECTED_DP4A_ENV, False)


@contextmanager
def _gguf_t16_selected_pairreuse_min_rows_scope(min_rows: int | None):
    """Apply a backend-certified physical-width floor during packed enqueue."""

    global _gguf_t16_selected_pairreuse_min_rows_session
    previous = _gguf_t16_selected_pairreuse_min_rows_session
    _gguf_t16_selected_pairreuse_min_rows_session = (
        None if min_rows is None else int(min_rows)
    )
    try:
        yield
    finally:
        _gguf_t16_selected_pairreuse_min_rows_session = previous


def _gguf_t16_selected_pairreuse_enabled() -> bool:
    raw = os.environ.get(_GGUF_T16_SELECTED_PAIRREUSE_ENV, "")
    if raw:
        return _env_flag(_GGUF_T16_SELECTED_PAIRREUSE_ENV, False)
    min_rows = _gguf_t16_selected_pairreuse_min_rows_session
    return min_rows is not None and min_rows > 0


@contextmanager
def _gguf_t16_selected_down_pairreuse_min_rows_scope(min_rows: int | None):
    """Apply a backend-certified selected-down reuse width floor."""

    global _gguf_t16_selected_down_pairreuse_min_rows_session
    previous = _gguf_t16_selected_down_pairreuse_min_rows_session
    _gguf_t16_selected_down_pairreuse_min_rows_session = (
        None if min_rows is None else int(min_rows)
    )
    try:
        yield
    finally:
        _gguf_t16_selected_down_pairreuse_min_rows_session = previous


def _gguf_t16_selected_down_pairreuse_enabled() -> bool:
    raw = os.environ.get(_GGUF_T16_SELECTED_DOWN_PAIRREUSE_ENV, "")
    if raw:
        return _env_flag(_GGUF_T16_SELECTED_DOWN_PAIRREUSE_ENV, False)
    min_rows = _gguf_t16_selected_down_pairreuse_min_rows_session
    return min_rows is not None and min_rows > 0


def _gguf_q5_t16_selected_qwen_tile8_enabled(backend: str | None) -> bool:
    if backend is None:
        return False
    return bool(
        backend_package_capability(
            backend,
            "GGUF_Q5_T16_SELECTED_QWEN_TILE8",
            False,
        )
    )


@contextmanager
def _gguf_t16_selected_q6_down_pairreuse_min_rows_scope(min_rows: int | None):
    """Apply a backend-certified Q6 selected-down reuse width floor."""

    global _gguf_t16_selected_q6_down_pairreuse_min_rows_session
    previous = _gguf_t16_selected_q6_down_pairreuse_min_rows_session
    _gguf_t16_selected_q6_down_pairreuse_min_rows_session = (
        None if min_rows is None else int(min_rows)
    )
    try:
        yield
    finally:
        _gguf_t16_selected_q6_down_pairreuse_min_rows_session = previous


def _gguf_t16_selected_q6_down_pairreuse_enabled() -> bool:
    raw = os.environ.get(_GGUF_T16_SELECTED_Q6_DOWN_PAIRREUSE_ENV, "")
    if raw:
        return _env_flag(_GGUF_T16_SELECTED_Q6_DOWN_PAIRREUSE_ENV, False)
    min_rows = _gguf_t16_selected_q6_down_pairreuse_min_rows_session
    return min_rows is not None and min_rows > 0


def _gguf_t16_ds4_prefill_enabled() -> bool:
    return _env_flag(_GGUF_T16_DS4_PREFILL_ENV, False)


def _gguf_raw_selected_dp4a_enabled() -> bool:
    return _env_flag(_GGUF_RAW_SELECTED_DP4A_ENV, False)


def _gguf_dense_q8_dp4a_enabled() -> bool:
    return _env_flag(_GGUF_DENSE_Q8_DP4A_ENV, False) or _gguf_dense_q8_dp4a_all_enabled()


def _gguf_dense_q8_dp4a_all_enabled() -> bool:
    return _env_flag(_GGUF_DENSE_Q8_DP4A_ALL_ENV, False)


def _gguf_dense_q8_dp4a_shared_enabled() -> bool:
    return _env_flag(_GGUF_DENSE_Q8_DP4A_SHARED_ENV, False)


def _gguf_dense_q8_dp4a_f32_enabled() -> bool:
    return _env_flag(_GGUF_DENSE_Q8_DP4A_F32_ENV, False)


def _gguf_row_compact_gemv_enabled() -> bool:
    return _env_flag(_GGUF_ROW_COMPACT_GEMV_ENV, False)


def _gguf_verify_row_lm_head_enabled() -> bool:
    return _env_flag(_GGUF_VERIFY_ROW_LM_HEAD_ENV, False)


def _gguf_verify_lm_head_q6_top1_dp4a_enabled() -> bool:
    return _env_flag(_GGUF_VERIFY_LM_HEAD_Q6_TOP1_DP4A_ENV, False)


def _gguf_verify_f32_residual_enabled() -> bool:
    return _env_flag(_GGUF_VERIFY_F32_RESIDUAL_ENV, False)


def _gguf_verify_f32_token_embedding_enabled() -> bool:
    return _env_flag(_GGUF_VERIFY_F32_TOKEN_EMBEDDING_ENV, False)


def _gguf_verify_f32_attention_norm_enabled() -> bool:
    return _env_flag(_GGUF_VERIFY_F32_ATTENTION_NORM_ENV, False)


def _gguf_verify_f32_linear_projections_enabled() -> bool:
    return _env_flag(_GGUF_VERIFY_F32_LINEAR_PROJECTIONS_ENV, False)


def _gguf_verify_f32_alpha_beta_enabled() -> bool:
    return _env_flag(_GGUF_VERIFY_F32_ALPHA_BETA_ENV, False)


def _gguf_verify_f32_attn_out_enabled() -> bool:
    return _env_flag(_GGUF_VERIFY_F32_ATTN_OUT_ENV, False)


def _gguf_verify_f32_moe_combine_enabled() -> bool:
    return _env_flag(_GGUF_VERIFY_F32_MOE_COMBINE_ENV, False)


def _gguf_verify_f32_selected_down_enabled() -> bool:
    return _env_flag(_GGUF_VERIFY_F32_SELECTED_DOWN_ENV, False)


def _gguf_verify_f32_selected_intermediate_enabled() -> bool:
    return _env_flag(_GGUF_VERIFY_F32_SELECTED_INTERMEDIATE_ENV, False)


def _gguf_verify_f32_shared_down_enabled() -> bool:
    return _env_flag(_GGUF_VERIFY_F32_SHARED_DOWN_ENV, False)


def _gguf_verify_f32_post_norm_enabled() -> bool:
    return _env_flag(_GGUF_VERIFY_F32_POST_NORM_ENV, False)


def _gguf_verify_f32_post_norm_router_enabled() -> bool:
    return _env_flag(_GGUF_VERIFY_F32_POST_NORM_ROUTER_ENV, True)


def _gguf_router_f32w_coop_enabled() -> bool:
    return _env_flag(_GGUF_ROUTER_F32W_COOP_ENV, True)


def _gguf_router_f32w_persistent_counter_enabled() -> bool:
    return _env_flag(_GGUF_ROUTER_F32W_PERSISTENT_COUNTER_ENV, True)


def _gguf_verify_f32_post_norm_selected_q8_enabled() -> bool:
    return _env_flag(_GGUF_VERIFY_F32_POST_NORM_SELECTED_Q8_ENV, True)


def _gguf_verify_f32_post_norm_shared_q8_enabled() -> bool:
    return _env_flag(_GGUF_VERIFY_F32_POST_NORM_SHARED_Q8_ENV, True)


def _gguf_verify_capture_f32_chain_conv_enabled() -> bool:
    return _env_flag(_GGUF_VERIFY_CAPTURE_F32_CHAIN_CONV_ENV, False)


def _gguf_verify_capture_regular_chain_gdn_enabled() -> bool:
    return _env_flag(_GGUF_VERIFY_CAPTURE_REGULAR_CHAIN_GDN_ENV, False)


def _gguf_verify_capture_bf16_gdn_out_enabled() -> bool:
    return _env_flag(_GGUF_VERIFY_CAPTURE_BF16_GDN_OUT_ENV, False)


def _gguf_verify_capture_prefill_gdn_enabled() -> bool:
    return _env_flag(_GGUF_VERIFY_CAPTURE_PREFILL_GDN_ENV, False)


def _gguf_verify_capture_prefill_gdn_chain_conv_enabled() -> bool:
    return _env_flag(_GGUF_VERIFY_CAPTURE_PREFILL_GDN_CHAIN_CONV_ENV, False)


def _gguf_verify_capture_score_prefill_enabled() -> bool:
    return _env_flag(_GGUF_VERIFY_CAPTURE_SCORE_PREFILL_ENV, False)


def _gguf_packed_verify_gpu_stage_timings_enabled() -> bool:
    return _env_flag(_GGUF_PACKED_VERIFY_GPU_STAGE_TIMINGS_ENV, False)


def _gguf_compact_wmma_no_read_max_selected_rows(backend: str) -> int:
    raw_default = backend_package_capability(
        backend,
        "GGUF_COMPACT_WMMA_NO_READ_MAX_SELECTED_ROWS",
        0,
    )
    try:
        default = int(raw_default)
    except (TypeError, ValueError):
        default = 0
    return max(
        0,
        _env_int(
            _GGUF_COMPACT_WMMA_NO_READ_MAX_SELECTED_ROWS_ENV,
            default,
        ),
    )


def _compact_wmma_static_upper_bound(
    selected_rows: int,
    num_experts: int,
) -> tuple[int, int]:
    """Return the tight routing-independent padded-row/tile upper bound.

    Compact WMMA pads each active expert to a 16-row tile.  For ``S`` selected
    rows and ``A=min(S, E)`` potentially active experts, assigning one row to
    every active expert creates ``A`` tiles.  Every further tile requires 16
    additional rows, so the exact worst-case tile count is
    ``A + floor((S-A)/16)``.  Unused tiles are initialized to expert ``-1`` and
    rejected by every compact selected kernel.
    """

    selected_rows = int(selected_rows)
    num_experts = int(num_experts)
    if selected_rows <= 0:
        raise ValueError("selected_rows must be positive")
    if num_experts <= 0:
        raise ValueError("num_experts must be positive")
    active_experts = min(selected_rows, num_experts)
    upper_tiles = active_experts + (selected_rows - active_experts) // 16
    return upper_tiles * 16, upper_tiles


def _gguf_token_embedding_rows_f32(
    reader: GGUFReader,
    token_ids: object,
    *,
    hidden_size: int,
    vocab_size: int,
    tensor_name: str = _GGUF_TOKEN_EMBEDDING_TENSOR,
) -> np.ndarray:
    tokens = np.asarray(token_ids, dtype=np.int64).reshape(-1)
    hidden_size = int(hidden_size)
    vocab_size = int(vocab_size)
    info = reader.tensor_info(tensor_name)
    shape = tuple(int(dim) for dim in info.shape)
    if shape != (vocab_size, hidden_size):
        raise ValueError(
            f"{tensor_name} shape {shape} does not match expected {(vocab_size, hidden_size)}"
        )
    raw = reader.tensor_data(tensor_name)
    rows = np.empty((int(tokens.size), hidden_size), dtype=np.float32)
    for row_index, token in enumerate(tokens.tolist()):
        token = int(token)
        if token < 0 or token >= vocab_size:
            raise ValueError(f"token_id {token} outside [0, {vocab_size})")
        dequantized = np.asarray(dequantize_gguf_data(raw[token], info.ggml_type), dtype=np.float32).reshape(-1)
        if int(dequantized.size) != hidden_size:
            raise ValueError(
                f"{tensor_name} token {token} dequantized to {dequantized.size} values, "
                f"expected {hidden_size}"
            )
        rows[row_index, :] = dequantized
    return np.ascontiguousarray(rows, dtype=np.float32)


def _gguf_f32_moe_combine_out_fn():
    if _gguf_verify_f32_moe_combine_enabled():
        return weighted_sum_shared_gate_combine_residual_out_f32_accum_f32w
    return weighted_sum_shared_gate_combine_residual_out_f32_f32w


def _gguf_f32_moe_combine_batch_out_fn():
    if _gguf_verify_f32_moe_combine_enabled():
        return weighted_sum_shared_gate_combine_residual_batch_out_f32_accum_f32w
    return weighted_sum_shared_gate_combine_residual_batch_out_f32_f32w


def _gguf_selected_down_supports_f32_output(weight: Qwen35GGUFDeviceWeight) -> bool:
    return weight.spec.quant_key in {"gguf_q5_k_x8_v1", "gguf_q6_k_x8_v1"}


def _gguf_use_f32_selected_down(weight: Qwen35GGUFDeviceWeight, scratch, f32_residual: bool) -> bool:
    return (
        f32_residual
        and _gguf_verify_f32_moe_combine_enabled()
        and _gguf_verify_f32_selected_down_enabled()
        and _gguf_selected_down_supports_f32_output(weight)
        and getattr(scratch, "moe_down_out_f32", None) is not None
    )


def _gguf_use_f32_shared_down(scratch, f32_residual: bool, selected_down_is_f32: bool) -> bool:
    return (
        f32_residual
        and selected_down_is_f32
        and _gguf_verify_f32_moe_combine_enabled()
        and _gguf_verify_f32_selected_down_enabled()
        and _gguf_verify_f32_shared_down_enabled()
        and getattr(scratch, "moe_shared_out_f32", None) is not None
    )


def _gguf_use_f32_selected_intermediate(scratch, prefer_f32_selected_down: bool) -> bool:
    return (
        prefer_f32_selected_down
        and _gguf_verify_f32_moe_combine_enabled()
        and _gguf_verify_f32_selected_down_enabled()
        and _gguf_verify_f32_selected_intermediate_enabled()
        and getattr(scratch, "ffn_intermediate_f32", None) is not None
    )


def _gguf_linear_supports_f32_activation(weight: Qwen35GGUFDeviceWeight) -> bool:
    try:
        resolve_gguf_linear_dispatch(weight, activation_dtype=GGUF_ACTIVATION_F32)
    except ValueError:
        return False
    return True


def _gguf_linear_supports_f32_activation_f32_output(
    weight: Qwen35GGUFDeviceWeight,
    *,
    rows: int,
) -> bool:
    try:
        resolve_gguf_linear_dispatch(
            weight,
            activation_dtype=GGUF_ACTIVATION_F32,
            output_dtype=GGUF_OUTPUT_F32,
            rows=rows,
        )
    except ValueError:
        return False
    return True


def _linear_attention_alpha_beta_f32_outputs_ready(route: str) -> bool:
    return route in {"dense_q8_dp4a_f32_out", "f32_singletons_f32_out"}


def _try_launch_gguf_linear_bf16_f32_output(
    weight: Qwen35GGUFDeviceWeight,
    x_ptr: int,
    out_ptr: int,
    *,
    rows: int,
    in_features: int,
    out_features: int,
    stream: int,
    runtime: HipRuntime,
) -> bool:
    try:
        resolve_gguf_linear_dispatch(weight, output_dtype=GGUF_OUTPUT_F32, rows=rows)
    except ValueError:
        pass
    else:
        launch_gguf_linear(
            weight,
            x_ptr,
            out_ptr,
            rows=rows,
            in_features=in_features,
            out_features=out_features,
            output_dtype=GGUF_OUTPUT_F32,
            stream=stream,
            runtime=runtime,
        )
        return True
    return _try_launch_dense_q8_raw_bf16_f32_output(
        weight,
        x_ptr,
        out_ptr,
        rows=rows,
        in_features=in_features,
        out_features=out_features,
        stream=stream,
        runtime=runtime,
    )


def _try_launch_dense_q8_raw_bf16_f32_output(
    weight: Qwen35GGUFDeviceWeight,
    x_ptr: int,
    out_ptr: int,
    *,
    rows: int,
    in_features: int,
    out_features: int,
    stream: int,
    runtime: HipRuntime,
) -> bool:
    raw = _dense_q8_raw_ptr(weight)
    if raw is None:
        return False
    rows = int(rows)
    if 2 <= rows <= 8:
        gguf_q8_0_gemv_rowtile_bf16_f32_out(
            x_ptr,
            raw,
            out_ptr,
            rows,
            in_features,
            out_features,
            stream=stream,
            runtime=runtime,
        )
    else:
        gguf_q8_0_gemv_bf16_f32_out(
            x_ptr,
            raw,
            out_ptr,
            rows,
            in_features,
            out_features,
            stream=stream,
            runtime=runtime,
        )
    return True


def _try_launch_shared_gate_up_from_f32_post_norm(
    gate_weight: Qwen35GGUFDeviceWeight,
    up_weight: Qwen35GGUFDeviceWeight,
    post_norm_f32_ptr: int | None,
    gate_out_ptr: int,
    up_out_ptr: int,
    *,
    rows: int,
    hidden_size: int,
    shared_ffn: int,
    stream: int,
    runtime: HipRuntime,
) -> bool:
    if post_norm_f32_ptr is None:
        return False
    if not (
        _gguf_linear_supports_f32_activation(gate_weight)
        and _gguf_linear_supports_f32_activation(up_weight)
    ):
        return False
    launch_gguf_linear(
        gate_weight,
        int(post_norm_f32_ptr),
        gate_out_ptr,
        rows=rows,
        in_features=hidden_size,
        out_features=shared_ffn,
        activation_dtype=GGUF_ACTIVATION_F32,
        stream=stream,
        runtime=runtime,
    )
    launch_gguf_linear(
        up_weight,
        int(post_norm_f32_ptr),
        up_out_ptr,
        rows=rows,
        in_features=hidden_size,
        out_features=shared_ffn,
        activation_dtype=GGUF_ACTIVATION_F32,
        stream=stream,
        runtime=runtime,
    )
    return True


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


def _quantize_activation_q8_1(
    x_ptr: int,
    q8_1_workspace_ptr: int,
    rows: int,
    in_features: int,
    *,
    x_f32_ptr: int | None = None,
    stream: int,
    runtime: HipRuntime,
) -> None:
    if x_f32_ptr is None:
        gguf_q4_k_quantize_bf16_q8_1(
            x_ptr,
            q8_1_workspace_ptr,
            rows,
            in_features,
            stream=stream,
            runtime=runtime,
        )
    else:
        gguf_q4_k_quantize_f32_q8_1(
            int(x_f32_ptr),
            q8_1_workspace_ptr,
            rows,
            in_features,
            stream=stream,
            runtime=runtime,
        )


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


def _try_launch_dense_q8_single_dp4a_f32(
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
    """Opt-in raw-Q8 q8_1/dp4a singleton route for F32 verifier activations."""

    if not _gguf_dense_q8_dp4a_f32_enabled() or int(rows) <= 1:
        return False
    raw = _dense_q8_raw_ptr(weight)
    if raw is None:
        return False
    q8_1_workspace_ptr = _dense_q8_workspace_ptr(scratch, rows, in_features)
    if q8_1_workspace_ptr is None:
        return False
    gguf_q4_k_quantize_f32_q8_1(
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


def _try_launch_dense_q8_single_dp4a_f32_out(
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
    """Opt-in raw-Q8 q8_1/dp4a singleton route for F32 verifier outputs."""

    if not _gguf_dense_q8_dp4a_f32_enabled() or int(rows) <= 1:
        return False
    raw = _dense_q8_raw_ptr(weight)
    if raw is None:
        return False
    q8_1_workspace_ptr = _dense_q8_workspace_ptr(scratch, rows, in_features)
    if q8_1_workspace_ptr is None:
        return False
    gguf_q4_k_quantize_f32_q8_1(
        x_ptr,
        q8_1_workspace_ptr,
        rows,
        in_features,
        stream=stream,
        runtime=runtime,
    )
    gguf_q8_0_dp4a_rowtile4_gemv_f32_f32_out(
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


def _try_launch_dense_q8_triple_dp4a_f32(
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
    """Opt-in raw-Q8 q8_1/dp4a triple route for F32 verifier activations."""

    if not _gguf_dense_q8_dp4a_f32_enabled() or int(rows) <= 1:
        return False
    raw_a = _dense_q8_raw_ptr(weight_a)
    raw_b = _dense_q8_raw_ptr(weight_b)
    raw_c = _dense_q8_raw_ptr(weight_c)
    if raw_a is None or raw_b is None or raw_c is None:
        return False
    q8_1_workspace_ptr = _dense_q8_workspace_ptr(scratch, rows, in_features)
    if q8_1_workspace_ptr is None:
        return False
    gguf_q4_k_quantize_f32_q8_1(
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


def _try_launch_dense_q8_pair_dp4a_f32(
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
    """Opt-in raw-Q8 q8_1/dp4a dual route for F32 verifier activations."""

    if not _gguf_dense_q8_dp4a_enabled() or int(rows) <= 1:
        return False
    raw_a = _dense_q8_raw_ptr(weight_a)
    raw_b = _dense_q8_raw_ptr(weight_b)
    if raw_a is None or raw_b is None:
        return False
    q8_1_workspace_ptr = _dense_q8_workspace_ptr(scratch, rows, in_features)
    if q8_1_workspace_ptr is None:
        return False
    gguf_q4_k_quantize_f32_q8_1(
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


def _try_launch_dense_q8_pair_dp4a_f32_out(
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
    """Opt-in raw-Q8 q8_1/dp4a dual route for F32 verifier outputs."""

    if (
        not _gguf_verify_f32_linear_projections_enabled()
        or not _gguf_dense_q8_dp4a_enabled()
        or int(rows) <= 1
    ):
        return False
    raw_a = _dense_q8_raw_ptr(weight_a)
    raw_b = _dense_q8_raw_ptr(weight_b)
    if raw_a is None or raw_b is None:
        return False
    q8_1_workspace_ptr = _dense_q8_workspace_ptr(scratch, rows, in_features)
    if q8_1_workspace_ptr is None:
        return False
    gguf_q4_k_quantize_f32_q8_1(
        x_ptr,
        q8_1_workspace_ptr,
        rows,
        in_features,
        stream=stream,
        runtime=runtime,
    )
    gguf_q8_0_dp4a_dual_split_rowtile4_gemv_f32_f32_out(
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


def _small_b_rowtile_chunks(rows: int, *, max_chunk: int = 6) -> tuple[int, ...]:
    rows = int(rows)
    max_chunk = int(max_chunk)
    if rows <= 0:
        raise ValueError("rows must be positive")
    if max_chunk < 2:
        raise ValueError("max_chunk must be at least 2")
    if rows <= max_chunk:
        return (rows,)
    chunks: list[int] = []
    remaining = rows
    while remaining > 0:
        if remaining <= max_chunk:
            if remaining == 1 and chunks:
                chunks[-1] -= 1
                chunks.append(2)
            else:
                chunks.append(remaining)
            break
        take = max_chunk
        if remaining - take == 1:
            take -= 1
        chunks.append(take)
        remaining -= take
    return tuple(chunks)


@dataclass(frozen=True)
class Qwen35GGUFKVChunkLayout:
    """Complete policy identity for one scheduler-owned GGUF KV chunk.

    ``layer_storage_dtypes`` is aligned with model layers and uses ``None`` for
    linear-attention layers.  Keeping the resolved per-layer layout beside the
    backing lets resident sessions reject payload/scale mismatches before any
    device metadata is changed.
    """

    storage_dtype: DType
    storage_layout: str
    scale_dtype: DType
    scale_granularity: str
    int8_kv_value_bf16: bool
    layer_storage_dtypes: tuple[DType | None, ...]
    bf16_mirror_layer_indices: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        storage = DType.parse(self.storage_dtype)
        scale = DType.parse(self.scale_dtype)
        layers = tuple(
            None if layer_storage is None else DType.parse(layer_storage)
            for layer_storage in self.layer_storage_dtypes
        )
        mirrors = tuple(sorted({int(index) for index in self.bf16_mirror_layer_indices}))
        object.__setattr__(self, "storage_dtype", storage)
        object.__setattr__(self, "scale_dtype", scale)
        object.__setattr__(self, "layer_storage_dtypes", layers)
        object.__setattr__(self, "bf16_mirror_layer_indices", mirrors)
        if storage not in {DType.BF16, DType.INT8_PER_TOKEN_HEAD}:
            raise ValueError("GGUF dynamic KV storage must be bf16 or int8_per_token_head")
        if scale not in {DType.FP16, DType.FP32}:
            raise ValueError("GGUF dynamic KV scales must use fp16 or fp32")
        if self.storage_layout not in {"uniform", "tail4_hadamard_group32"}:
            raise ValueError("unsupported GGUF dynamic KV storage layout")
        if self.scale_granularity not in {"per_token_head", "block16", "hadamard_group32"}:
            raise ValueError("unsupported GGUF dynamic KV scale granularity")
        if not layers or not any(layer_storage is not None for layer_storage in layers):
            raise ValueError("GGUF dynamic KV layout requires a full-attention layer")
        if any(
            layer_storage not in {None, DType.BF16, DType.INT8_PER_TOKEN_HEAD}
            for layer_storage in layers
        ):
            raise ValueError("unsupported per-layer GGUF dynamic KV storage dtype")
        int8_layers = tuple(
            index
            for index, layer_storage in enumerate(layers)
            if layer_storage == DType.INT8_PER_TOKEN_HEAD
        )
        if storage == DType.BF16 and int8_layers:
            raise ValueError("BF16 GGUF dynamic KV layout cannot contain INT8 layers")
        if storage == DType.INT8_PER_TOKEN_HEAD and not int8_layers:
            raise ValueError("INT8 GGUF dynamic KV layout must contain an INT8 layer")
        if self.storage_layout == "tail4_hadamard_group32" and (
            storage != DType.INT8_PER_TOKEN_HEAD
            or self.scale_granularity != "hadamard_group32"
            or len(int8_layers) != 4
        ):
            raise ValueError("tail4_hadamard_group32 requires exactly four INT8 layers")
        if self.int8_kv_value_bf16 and (
            storage != DType.INT8_PER_TOKEN_HEAD
            or self.scale_granularity != "per_token_head"
        ):
            raise ValueError("BF16-value INT8 GGUF dynamic KV requires per-token-head scales")
        if any(index < 0 or index >= len(layers) for index in mirrors):
            raise ValueError("GGUF dynamic KV mirror layer index is outside the model")
        if any(layers[index] != DType.INT8_PER_TOKEN_HEAD for index in mirrors):
            raise ValueError("GGUF dynamic KV mirrors are valid only for INT8 layers")
        if mirrors and self.scale_granularity == "hadamard_group32":
            raise ValueError("Hadamard-group32 GGUF dynamic KV does not use BF16 mirrors")


@dataclass(frozen=True)
class Qwen35GGUFKVChunkBacking:
    """Payload and scale backing for one contiguous chunk of logical KV pages."""

    layout: Qwen35GGUFKVChunkLayout
    start_block_id: int
    pages: int
    full_key_caches: tuple[DeviceBuffer | None, ...]
    full_value_caches: tuple[DeviceBuffer | None, ...]
    full_bf16_mirror_key_caches: tuple[DeviceBuffer | None, ...]
    full_bf16_mirror_value_caches: tuple[DeviceBuffer | None, ...]
    full_k_scale_caches: tuple[DeviceBuffer | None, ...]
    full_v_scale_caches: tuple[DeviceBuffer | None, ...]
    full_kv_scale_metadata: tuple[KVScaleMetadata | None, ...]
    buffers: tuple[DeviceBuffer, ...]

    def __post_init__(self) -> None:
        if int(self.start_block_id) < 0:
            raise ValueError("GGUF KV chunk start_block_id must be non-negative")
        if int(self.pages) <= 0:
            raise ValueError("GGUF KV chunk pages must be positive")
        layer_count = len(self.layout.layer_storage_dtypes)
        aligned = (
            self.full_key_caches,
            self.full_value_caches,
            self.full_bf16_mirror_key_caches,
            self.full_bf16_mirror_value_caches,
            self.full_k_scale_caches,
            self.full_v_scale_caches,
            self.full_kv_scale_metadata,
        )
        if any(len(values) != layer_count for values in aligned):
            raise ValueError("GGUF KV chunk layer tuples must align with the layout")
        mirror_layers = frozenset(self.layout.bf16_mirror_layer_indices)
        for layer_id, layer_storage in enumerate(self.layout.layer_storage_dtypes):
            key_cache = self.full_key_caches[layer_id]
            value_cache = self.full_value_caches[layer_id]
            mirror_key = self.full_bf16_mirror_key_caches[layer_id]
            mirror_value = self.full_bf16_mirror_value_caches[layer_id]
            k_scale = self.full_k_scale_caches[layer_id]
            v_scale = self.full_v_scale_caches[layer_id]
            metadata = self.full_kv_scale_metadata[layer_id]
            if layer_storage is None:
                if any(
                    value is not None
                    for value in (key_cache, value_cache, mirror_key, mirror_value, k_scale, v_scale, metadata)
                ):
                    raise ValueError("linear-attention GGUF KV chunk entries must be empty")
                continue
            if key_cache is None or value_cache is None:
                raise ValueError("full-attention GGUF KV chunk payload is incomplete")
            if (mirror_key is None) != (mirror_value is None):
                raise ValueError("GGUF KV chunk BF16 mirror payload must be paired")
            if (layer_id in mirror_layers) != (mirror_key is not None):
                raise ValueError("GGUF KV chunk BF16 mirror payload does not match its layout")
            if layer_storage == DType.INT8_PER_TOKEN_HEAD:
                if k_scale is None or v_scale is None or metadata is None:
                    raise ValueError("INT8 GGUF KV chunk requires paired scale payload and metadata")
                if int(metadata.k_scale.ptr) != int(k_scale.ptr) or int(metadata.v_scale.ptr) != int(v_scale.ptr):
                    raise ValueError("GGUF KV chunk scale metadata does not reference its backing")
                if metadata.scale_dtype != self.layout.scale_dtype:
                    raise ValueError("GGUF KV chunk scale dtype does not match its layout")
                if metadata.granularity != self.layout.scale_granularity:
                    raise ValueError("GGUF KV chunk scale granularity does not match its layout")
                if int(metadata.k_scale.shape[0]) != int(self.pages):
                    raise ValueError("GGUF KV chunk scale metadata does not cover every page")
            elif any(value is not None for value in (k_scale, v_scale, metadata)):
                raise ValueError("BF16 GGUF KV chunk layer cannot carry INT8 scales")
        if any(int(buffer.nbytes) <= 0 or int(buffer.nbytes) % int(self.pages) for buffer in self.buffers):
            raise ValueError("GGUF KV chunk buffers must contain whole positive pages")

    @property
    def total_nbytes(self) -> int:
        return sum(int(buffer.nbytes) for buffer in self.buffers)

    @property
    def page_bytes(self) -> int:
        return self.total_nbytes // int(self.pages)

    def page_pointer(self, local_page: int) -> int:
        page = int(local_page)
        if page < 0 or page >= self.pages:
            raise IndexError("local KV page is outside the chunk")
        first = next((buffer for buffer in self.full_key_caches if buffer is not None), None)
        if first is None:
            raise RuntimeError("GGUF KV chunk has no full-attention layer backing")
        return int(first.ptr) + page * (int(first.nbytes) // int(self.pages))

    def validate_bound_blocks(self, block_ids: tuple[int, ...]) -> None:
        """Validate request-local page ids without slicing the shared backing."""

        blocks = tuple(int(block_id) for block_id in block_ids)
        if not blocks:
            raise ValueError("bound GGUF KV allocation must contain pages")
        if len(set(blocks)) != len(blocks):
            raise ValueError("bound GGUF KV allocation pages must be unique")
        local_pages = tuple(block_id - int(self.start_block_id) for block_id in blocks)
        if any(local_page < 0 or local_page >= self.pages for local_page in local_pages):
            raise ValueError("GGUF KV allocation is outside its backing chunk")


def _qwen35_gguf_kv_scale_shape(
    cfg: object,
    *,
    pages: int,
    granularity: str,
) -> tuple[int, ...]:
    base = (int(pages), 256, int(cfg.head_count_kv))
    if granularity == "per_token_head":
        return base
    group_size = 16 if granularity == "block16" else 32
    if int(cfg.key_length) % group_size:
        raise ValueError(f"GGUF {granularity} KV requires key_length divisible by {group_size}")
    groups = int(cfg.key_length) // group_size
    if granularity == "block16" and groups != 16:
        raise ValueError("GGUF INT8 KV block16 scales require head_dim/key_length 256")
    return (*base, groups)


def _qwen35_gguf_kv_page_bytes(cfg: object, layout: Qwen35GGUFKVChunkLayout) -> int:
    payload_elements = 256 * int(cfg.head_count_kv) * int(cfg.key_length)
    scale_elements = int(np.prod(_qwen35_gguf_kv_scale_shape(cfg, pages=1, granularity=layout.scale_granularity)))
    mirror_layers = frozenset(layout.bf16_mirror_layer_indices)
    page_bytes = 0
    for layer_id, layer_storage in enumerate(layout.layer_storage_dtypes):
        if layer_storage is None:
            continue
        key_dtype = DType.INT8 if layer_storage == DType.INT8_PER_TOKEN_HEAD else DType.BF16
        value_dtype = (
            DType.BF16
            if layer_storage == DType.INT8_PER_TOKEN_HEAD and layout.int8_kv_value_bf16
            else key_dtype
        )
        page_bytes += payload_elements * (key_dtype.itemsize + value_dtype.itemsize)
        if layer_id in mirror_layers:
            page_bytes += 2 * payload_elements * DType.BF16.itemsize
        if layer_storage == DType.INT8_PER_TOKEN_HEAD:
            page_bytes += 2 * scale_elements * layout.scale_dtype.itemsize
    return int(page_bytes)


def _allocate_qwen35_gguf_kv_chunk(
    runner: Qwen35GGUFFullStackRunner,
    *,
    runtime: HipRuntime,
    start_block_id: int,
    pages: int,
    layout: Qwen35GGUFKVChunkLayout,
) -> Qwen35GGUFKVChunkBacking:
    if runner.weights is None:
        raise RuntimeError("GGUF full-stack runner is closed")
    cfg = runner.weights.config
    if len(layout.layer_storage_dtypes) != len(cfg.layer_types):
        raise ValueError("GGUF KV chunk layout does not match the model layer count")
    payload_elements = int(pages) * 256 * int(cfg.head_count_kv) * int(cfg.key_length)
    scale_shape = _qwen35_gguf_kv_scale_shape(
        cfg,
        pages=int(pages),
        granularity=layout.scale_granularity,
    )
    scale_nbytes = int(np.prod(scale_shape)) * layout.scale_dtype.itemsize
    mirror_layers = frozenset(layout.bf16_mirror_layer_indices)
    device = Device("hip", 0)
    key_caches: list[DeviceBuffer | None] = []
    value_caches: list[DeviceBuffer | None] = []
    mirror_key_caches: list[DeviceBuffer | None] = []
    mirror_value_caches: list[DeviceBuffer | None] = []
    k_scale_caches: list[DeviceBuffer | None] = []
    v_scale_caches: list[DeviceBuffer | None] = []
    scale_metadata: list[KVScaleMetadata | None] = []
    buffers: list[DeviceBuffer] = []
    try:
        for layer_id, (layer_type, layer_storage) in enumerate(
            zip(cfg.layer_types, layout.layer_storage_dtypes, strict=True)
        ):
            if layer_type == LINEAR_ATTENTION:
                if layer_storage is not None:
                    raise ValueError("GGUF KV chunk layout assigns storage to a linear-attention layer")
                key_caches.append(None)
                value_caches.append(None)
                mirror_key_caches.append(None)
                mirror_value_caches.append(None)
                k_scale_caches.append(None)
                v_scale_caches.append(None)
                scale_metadata.append(None)
                continue
            if layer_storage is None:
                raise ValueError("GGUF KV chunk layout omits a full-attention layer")
            key_dtype = DType.INT8 if layer_storage == DType.INT8_PER_TOKEN_HEAD else DType.BF16
            value_dtype = (
                DType.BF16
                if layer_storage == DType.INT8_PER_TOKEN_HEAD and layout.int8_kv_value_bf16
                else key_dtype
            )
            key_cache = malloc(payload_elements * key_dtype.itemsize, runtime=runtime)
            value_cache = malloc(payload_elements * value_dtype.itemsize, runtime=runtime)
            buffers.extend((key_cache, value_cache))
            key_caches.append(key_cache)
            value_caches.append(value_cache)
            if layer_id in mirror_layers:
                mirror_nbytes = payload_elements * DType.BF16.itemsize
                mirror_key = malloc(mirror_nbytes, runtime=runtime)
                mirror_value = malloc(mirror_nbytes, runtime=runtime)
                buffers.extend((mirror_key, mirror_value))
                mirror_key_caches.append(mirror_key)
                mirror_value_caches.append(mirror_value)
            else:
                mirror_key_caches.append(None)
                mirror_value_caches.append(None)
            if layer_storage == DType.INT8_PER_TOKEN_HEAD:
                k_scale = malloc(scale_nbytes, runtime=runtime)
                v_scale = malloc(scale_nbytes, runtime=runtime)
                buffers.extend((k_scale, v_scale))
                k_scale_caches.append(k_scale)
                v_scale_caches.append(v_scale)
                scale_metadata.append(
                    KVScaleMetadata(
                        k_scale=Tensor.from_handle(k_scale.ptr, scale_shape, layout.scale_dtype, device),
                        v_scale=Tensor.from_handle(v_scale.ptr, scale_shape, layout.scale_dtype, device),
                        scale_dtype=layout.scale_dtype,
                        granularity=layout.scale_granularity,
                    )
                )
            else:
                k_scale_caches.append(None)
                v_scale_caches.append(None)
                scale_metadata.append(None)
    except Exception:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)
        raise
    try:
        backing = Qwen35GGUFKVChunkBacking(
            layout=layout,
            start_block_id=int(start_block_id),
            pages=int(pages),
            full_key_caches=tuple(key_caches),
            full_value_caches=tuple(value_caches),
            full_bf16_mirror_key_caches=tuple(mirror_key_caches),
            full_bf16_mirror_value_caches=tuple(mirror_value_caches),
            full_k_scale_caches=tuple(k_scale_caches),
            full_v_scale_caches=tuple(v_scale_caches),
            full_kv_scale_metadata=tuple(scale_metadata),
            buffers=tuple(buffers),
        )
        expected_page_bytes = _qwen35_gguf_kv_page_bytes(cfg, layout)
        if backing.page_bytes != expected_page_bytes:
            raise RuntimeError("GGUF KV chunk byte accounting drift")
    except Exception:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)
        raise
    return backing


def _free_qwen35_gguf_kv_chunk(
    backing: Qwen35GGUFKVChunkBacking,
    *,
    runtime: HipRuntime,
) -> None:
    for buffer in reversed(backing.buffers):
        free(buffer, runtime=runtime)


def _qwen35_gguf_session_kv_chunk_layout(
    session: "Qwen35GGUFResidentSession",
) -> Qwen35GGUFKVChunkLayout:
    if session.runner is None or session.runner.weights is None or session.scratch is None:
        raise RuntimeError("GGUF resident session is closed")
    layer_storage: list[DType | None] = []
    bf16_full_indices = frozenset(int(index) for index in session.int8_bf16_full_attention_layer_indices)
    full_attention_index = 0
    for layer_type in session.runner.weights.config.layer_types:
        if layer_type == LINEAR_ATTENTION:
            layer_storage.append(None)
            continue
        layer_storage.append(
            DType.BF16
            if session.kv_storage_dtype == DType.BF16 or full_attention_index in bf16_full_indices
            else DType.INT8_PER_TOKEN_HEAD
        )
        full_attention_index += 1
    mirror_layers = (
        tuple(
            layer_id
            for layer_id, storage in enumerate(layer_storage)
            if storage == DType.INT8_PER_TOKEN_HEAD
        )
        if session.kv_storage_dtype == DType.INT8_PER_TOKEN_HEAD
        and session.kv_scale_granularity != "hadamard_group32"
        and int(session.scratch.max_positions) <= _GGUF_INT8_SHORT_BF16_MIRROR_MAX_POSITIONS
        else ()
    )
    return Qwen35GGUFKVChunkLayout(
        storage_dtype=session.kv_storage_dtype,
        storage_layout=session.kv_storage_layout,
        scale_dtype=session.kv_scale_dtype,
        scale_granularity=session.kv_scale_granularity,
        int8_kv_value_bf16=bool(session.int8_kv_value_bf16),
        layer_storage_dtypes=tuple(layer_storage),
        bf16_mirror_layer_indices=mirror_layers,
    )


def _gguf_device_kv_contiguous_base_row(
    session: object,
    *,
    block_size: int = 256,
) -> int | None:
    """Return the raw-cache row offset for one contiguous device-KV allocation."""

    page_tokens = int(block_size)
    if page_tokens <= 0:
        raise ValueError("GGUF device KV block size must be positive")
    allocation = getattr(session, "_device_kv_allocation", None)
    if allocation is None:
        return 0
    block_ids = tuple(int(block_id) for block_id in allocation.block_ids)
    if not block_ids:
        raise ValueError("GGUF device KV allocation must contain pages")
    if block_ids != tuple(block_ids[0] + index for index in range(len(block_ids))):
        return None
    local_page = block_ids[0] - int(allocation.chunk_start_block_id)
    if local_page < 0:
        raise ValueError("GGUF device KV block precedes its backing chunk")
    return local_page * page_tokens


def _gguf_device_kv_contiguous_cache_view(
    session: object,
    cache: DeviceBuffer,
    *,
    row_nbytes: int,
    block_size: int = 256,
) -> DeviceBuffer | None:
    """Rebase a raw contiguous cache to the request's first physical page."""

    row_bytes = int(row_nbytes)
    if row_bytes <= 0:
        raise ValueError("GGUF device KV row bytes must be positive")
    base_row = _gguf_device_kv_contiguous_base_row(
        session,
        block_size=block_size,
    )
    if base_row is None:
        return None
    offset = int(base_row) * row_bytes
    if offset == 0:
        return cache
    if offset < 0 or offset >= int(cache.nbytes):
        raise ValueError("GGUF contiguous device KV view exceeds its backing cache")
    return DeviceBuffer(
        ptr=int(cache.ptr) + offset,
        nbytes=int(cache.nbytes) - offset,
    )


def _gguf_device_kv_copy_segments(
    session: object,
    *,
    start_position: int,
    rows: int,
    block_size: int = 256,
) -> tuple[tuple[int, int, int], ...]:
    """Map logical K/V rows to physical backing rows for D2D gather/scatter."""

    start = int(start_position)
    remaining = int(rows)
    page_tokens = int(block_size)
    if start < 0 or remaining < 0:
        raise ValueError("GGUF device KV copy range must be non-negative")
    if page_tokens <= 0:
        raise ValueError("GGUF device KV block size must be positive")
    allocation = getattr(session, "_device_kv_allocation", None)
    if allocation is None or remaining == 0:
        return () if remaining == 0 else ((start, start, remaining),)
    block_ids = tuple(int(block_id) for block_id in allocation.block_ids)
    chunk_start = int(allocation.chunk_start_block_id)
    segments: list[tuple[int, int, int]] = []
    logical = start
    while remaining:
        logical_page, in_page = divmod(logical, page_tokens)
        if logical_page >= len(block_ids):
            raise ValueError("GGUF device KV allocation does not cover the copy range")
        local_page = block_ids[logical_page] - chunk_start
        if local_page < 0:
            raise ValueError("GGUF device KV block precedes its backing chunk")
        take = min(remaining, page_tokens - in_page)
        physical = local_page * page_tokens + in_page
        segments.append((logical, physical, take))
        logical += take
        remaining -= take
    return tuple(segments)


@dataclass
class Qwen35GGUFPrefixStateSnapshot:
    """Cache-owned device snapshot for one exact GGUF hybrid-state boundary."""

    runtime: HipRuntime
    runner: object
    kv_storage_dtype: DType
    kv_storage_layout: str
    position: int
    block_ids: tuple[int, ...]
    backing: object
    layer_conv_states: tuple[DeviceBuffer | None, ...]
    layer_recurrent_states: tuple[DeviceBuffer | None, ...]
    closed: bool = False

    @property
    def nbytes(self) -> int:
        return sum(
            int(buffer.nbytes)
            for buffer in (*self.layer_conv_states, *self.layer_recurrent_states)
            if buffer is not None
        )

    def close(self) -> None:
        if self.closed:
            return
        for buffer in reversed((*self.layer_conv_states, *self.layer_recurrent_states)):
            if buffer is not None:
                free(buffer, runtime=self.runtime)
        self.closed = True


@dataclass
class Qwen35GGUFResidentSession:
    """Persistent GGUF Qwen3.5 session for public greedy generation.

    The session materializes GGUF weights once, owns reusable device scratch, and
    carries linear-attention recurrent state plus paged full-attention K/V cache
    across decode steps. Full-attention q/k norm, RoPE, KV append, softmax, gate
    application, lm-head argmax, and full-model bulk prefill stay on GPU for the
    resident path. Backends may additionally admit state-bound decode replay for
    measured long greedy windows.
    """

    model_path: str | Path
    runtime: HipRuntime | None = None
    compiler_version: str | None = None
    require_cached_build: bool = False
    backend: str = "auto"
    shared_runner: Qwen35GGUFFullStackRunner | None = None
    max_sequence_length: int | None = None
    max_batch_size: int = 1
    use_expert_sidecar: bool = False
    expert_sidecar_cache_dir: str | Path | None = None
    require_expert_sidecar: bool = False
    preload_expert_sidecars: bool = True
    use_wmma_prefill: bool | None = None
    use_gemv_decode: bool | None = None
    prefill_chunk_size: int = 0
    prefill_config: PrefillConfig | None = None
    prefill_flight_recorder_path: str | Path | None = None
    prefill_flight_recorder_granularity: str = "chunk"
    prefill_queue_drain: str = "none"
    kv_policy: FixedPagedKVPolicy | None = None
    kv_scale_dtype: str | DType = DType.FP16
    kv_scale_granularity: str = "per_token_head"
    defer_kv_allocation: bool = False
    runner: Qwen35GGUFFullStackRunner | None = field(default=None, init=False)
    scratch: object | None = field(default=None, init=False)
    _target_scratch_owner: object | None = field(default=None, init=False)
    _target_layout: Qwen35GGUFResidentTargetLayout | None = field(default=None, init=False)
    _owns_runner: bool = field(default=True, init=False)
    _token_buf: object | None = field(default=None, init=False)
    _hidden_a: object | None = field(default=None, init=False)
    _hidden_b: object | None = field(default=None, init=False)
    _logits_buf: object | None = field(default=None, init=False)
    _native_cu_seqlens_buf: object | None = field(default=None, init=False)
    _native_state_indices_buf: object | None = field(default=None, init=False)
    _native_token_ids_host: np.ndarray | None = field(default=None, init=False)
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
    _verify_hidden_f32_a: object | None = field(default=None, init=False)
    _verify_hidden_f32_b: object | None = field(default=None, init=False)
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
    _packed_verify_state: _GGUFPackedTargetState | None = field(default=None, init=False)
    _packed_verify_scratch: object | None = field(default=None, init=False)
    _packed_verify_session_ids: tuple[int, ...] = field(default=(), init=False)
    _packed_verify_max_written_positions: tuple[int, ...] = field(default=(), init=False)
    _packed_decode_sessions: tuple["Qwen35GGUFResidentSession | None", ...] = field(default=(), init=False)
    _packed_decode_last_layout: _GGUFPackedVerifyLayout | None = field(default=None, init=False)
    _packed_decode_state_dirty: bool = field(default=False, init=False)
    _packed_decode_session_ids: tuple[int, ...] = field(default=(), init=False)
    _packed_decode_positions: tuple[int, ...] = field(default=(), init=False)
    _packed_ar_attention_workspace: _GGUFPackedARAttentionWorkspace | None = field(
        default=None,
        init=False,
    )
    _prefill_token_buf: object | None = field(default=None, init=False)
    _prefill_hidden_a: object | None = field(default=None, init=False)
    _prefill_hidden_b: object | None = field(default=None, init=False)
    _bulk_prefill_scratch: object | None = field(default=None, init=False)
    _q8_mmq_prefill_library: object | None = field(default=None, init=False)
    _q8_mmq_risk_count: object | None = field(default=None, init=False)
    _q8_mmq_risk_indices: object | None = field(default=None, init=False)
    _prefill_flight_recorder: PrefillFlightRecorder | None = field(default=None, init=False)
    _prefill_aotriton_stream: int = field(default=0, init=False)
    _prefill_aotriton_input_ready_event: int = field(default=0, init=False)
    _prefill_aotriton_output_ready_event: int = field(default=0, init=False)
    _int8_prefill_oracle_buffers: dict[int, tuple[DeviceBuffer, DeviceBuffer]] = field(default_factory=dict, init=False)
    _linear_state_snapshot_backups: tuple[object, ...] = field(default=(), init=False)
    _runtime_state_library: object | None = field(default=None, init=False)
    _lm_head_library: object | None = field(default=None, init=False)
    _native_sampler_workspace: NativeSamplerWorkspace | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _dflash_commit_library: object | None = field(default=None, init=False)
    _expert_pack8_library: object | None = field(default=None, init=False)
    _q6_pack8_library: object | None = field(default=None, init=False)
    _expert_sidecar_reader: GGUFReader | None = field(default=None, init=False)
    _verify_token_embedding_reader: GGUFReader | None = field(default=None, init=False)
    _expert_sidecar_model_map: object | None = field(default=None, init=False)
    _expert_sidecar_host_layers: dict[int, dict[str, GGUFExpertPackedTensor]] | None = field(default=None, init=False)
    _host_token_embedding_reader: GGUFReader | None = field(default=None, init=False)
    _host_token_embedding_raw: np.ndarray | None = field(default=None, init=False)
    _host_token_embedding_cache: dict[int, np.ndarray] = field(default_factory=dict, init=False)
    host_token_embedding_enabled: bool = field(default=False, init=False)
    host_token_embedding_reason: str | None = field(default=None, init=False)
    _owns_runner: bool = field(default=True, init=False)
    _token_host: np.ndarray = field(default_factory=lambda: np.empty((1,), dtype=np.int64), init=False)
    _logits_host: np.ndarray | None = field(default=None, init=False)
    _buffers: tuple[object, ...] = field(default=(), init=False)
    _position: int = field(default=0, init=False)
    _last_target_hidden_ptr: int = field(default=0, init=False)
    _reset_current_slot_only: bool = field(default=False, init=False)
    _hidden_seed_fp32_populated: bool = field(default=False, init=False)
    _last_pre_output_norm_hidden: np.ndarray | None = field(default=None, init=False)
    _last_layer_output_hidden: dict[int, np.ndarray] = field(default_factory=dict, init=False)
    _lm_head_threads: int = field(default=128, init=False)
    _lm_head_stage1_blocks: int = field(default=0, init=False)
    last_verify_stage_timings_ms: dict[str, float] = field(default_factory=dict, init=False)
    last_packed_verify_stage_timings_ms: dict[str, float] = field(default_factory=dict, init=False)
    last_native_spec_target_submitted: bool = field(default=False, init=False)
    last_native_spec_target_fallback_reason: str | None = field(default=None, init=False)
    last_native_spec_target_capture_ms: float = field(default=0.0, init=False)
    last_native_spec_target_submit_ms: float = field(default=0.0, init=False)
    last_native_spec_target_readback_ms: float = field(default=0.0, init=False)
    fastpath_safety: Qwen35GGUFFastPathSafety | None = field(default=None, init=False)
    prefill_chunk_tuning: dict[str, object] = field(default_factory=dict, init=False)
    last_packed_prefill_plan: dict[str, object] = field(default_factory=dict, init=False)
    last_packed_execution_manifest: dict[str, object] = field(default_factory=dict, init=False)
    _last_packed_lm_head_decode_path: str = field(default="unobserved", init=False)
    _last_packed_sampler_decode_path: str = field(default="unobserved", init=False)
    kv_storage_dtype: DType = field(default=DType.BF16, init=False)
    kv_storage_layout: str = field(default="uniform", init=False)
    int8_kv_value_bf16: bool = field(default=False, init=False)
    int8_bf16_prefix_full_attention_layers: int = field(default=0, init=False)
    int8_bf16_full_attention_layer_indices: tuple[int, ...] = field(default=(), init=False)
    _decode_graphs: list[object] = field(default_factory=list, init=False)
    _decode_graph_min_replay_steps_cache: int | None = field(default=None, init=False)
    _native_spec_b1_target_graph: object | None = field(default=None, init=False, repr=False)
    _native_spec_b2_target_graph: object | None = field(default=None, init=False, repr=False)
    _native_spec_b1_target_graph_n2: object | None = field(default=None, init=False, repr=False)
    _native_spec_b2_target_graph_n2: object | None = field(default=None, init=False, repr=False)
    _device_kv_pool: DeviceChunkedKVPool | None = field(default=None, init=False, repr=False)
    _device_kv_allocation: DeviceKVPoolAllocation | None = field(default=None, init=False, repr=False)
    _device_kv_layout: Qwen35GGUFKVChunkLayout | None = field(default=None, init=False, repr=False)
    _device_kv_graph_handles: dict[int, object] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self.prefill_queue_drain = _normalize_prefill_queue_drain(
            self.prefill_queue_drain
        )
        self.runtime = self.runtime or get_hip_runtime()
        if self.shared_runner is None:
            self.runner = Qwen35GGUFFullStackRunner(
                self.model_path,
                runtime=self.runtime,
                compiler_version=self.compiler_version,
                require_cached_build=self.require_cached_build,
                backend=self.backend,
            )
            self._owns_runner = True
        else:
            self.runner = self.shared_runner
            self.runtime = self.runner.runtime or self.runtime
            self._owns_runner = False
        # ``auto`` is a load-time selector, never a registry/capability key.
        # The full-stack runner has already resolved it against the detected
        # device, so keep the resident session on that concrete identity too.
        self.backend = self.runner.backend
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
        self.kv_policy = self.kv_policy or FixedPagedKVPolicy(block_size=256, storage_dtype=DType.BF16)
        policy_block_size = int(getattr(self.kv_policy, "block_size", 256))
        if policy_block_size != 256:
            raise ValueError("GGUF resident KV policy block_size must be 256")
        self.kv_storage_dtype = DType.parse(getattr(self.kv_policy, "storage_dtype", DType.BF16))
        if self.kv_storage_dtype not in {DType.BF16, DType.INT8_PER_TOKEN_HEAD}:
            raise ValueError("GGUF resident full-attention KV storage must be bf16 or int8_per_token_head")
        self.kv_storage_layout = str(getattr(self.kv_policy, "storage_layout", "uniform"))
        self.int8_kv_value_bf16 = _gguf_int8_kv_value_bf16_enabled(kv_storage_dtype=self.kv_storage_dtype)
        self.kv_scale_dtype = DType.parse(self.kv_scale_dtype)
        if self.kv_scale_dtype not in {DType.FP16, DType.FP32}:
            raise ValueError("GGUF resident INT8 KV scales must use fp16 or fp32")
        requested_granularity = (
            str(getattr(self.kv_policy, "scale_granularity", self.kv_scale_granularity))
            if self.kv_storage_layout != "uniform"
            else self.kv_scale_granularity
        )
        if self.kv_storage_layout == "uniform":
            self.kv_scale_granularity = _gguf_int8_kv_scale_granularity(
                kv_storage_dtype=self.kv_storage_dtype,
                requested_granularity=requested_granularity,
            )
        else:
            self.kv_scale_granularity = str(requested_granularity).strip().lower()
            if self.kv_scale_granularity not in {"per_token_head", "block16", "hadamard_group32"}:
                raise ValueError("unsupported GGUF resident INT8 KV scale granularity")
        if self.int8_kv_value_bf16 and self.kv_scale_granularity != "per_token_head":
            raise ValueError("GGUF grouped INT8 KV scales are not supported with the key-only diagnostic")
        requested_positions = 256 if self.max_sequence_length is None else int(self.max_sequence_length)
        rounded_positions = min(
            int(self.runner.weights.config.context_length),
            ((requested_positions + 255) // 256) * 256,
        )
        full_attention_layer_count = sum(
            1 for layer_type in self.runner.weights.config.layer_types if layer_type == FULL_ATTENTION
        )
        if self.kv_storage_layout == "uniform":
            self.int8_bf16_prefix_full_attention_layers = _gguf_int8_bf16_prefix_full_attention_layers(
                kv_storage_dtype=self.kv_storage_dtype,
                max_positions=rounded_positions,
            )
            self.int8_bf16_full_attention_layer_indices = _gguf_int8_bf16_full_attention_layer_indices(
                kv_storage_dtype=self.kv_storage_dtype,
                max_positions=rounded_positions,
                full_attention_layers=full_attention_layer_count,
            )
        else:
            selector = getattr(self.kv_policy, "full_attention_storage_dtype", None)
            if selector is None:
                raise ValueError("mixed GGUF KV policy must select per-layer full-attention storage")
            self.int8_bf16_full_attention_layer_indices = tuple(
                index
                for index in range(full_attention_layer_count)
                if DType.parse(selector(index, full_attention_layer_count)) == DType.BF16
            )
            self.int8_bf16_prefix_full_attention_layers = next(
                (
                    index
                    for index in range(full_attention_layer_count)
                    if index not in self.int8_bf16_full_attention_layer_indices
                ),
                full_attention_layer_count,
            )
        custom_bf16_layers = (
            self.kv_storage_layout == "uniform"
            and _env_value(_GGUF_INT8_BF16_FULL_ATTENTION_LAYERS_ENV) is not None
        )
        self.kv_scale_dtype = _gguf_int8_effective_scale_dtype(
            kv_storage_dtype=self.kv_storage_dtype,
            max_positions=rounded_positions,
            requested_scale_dtype=self.kv_scale_dtype,
            bf16_prefix_full_attention_layers=self.int8_bf16_prefix_full_attention_layers,
            bf16_full_attention_layer_count=len(self.int8_bf16_full_attention_layer_indices),
            scale_granularity=self.kv_scale_granularity,
        )
        _validate_gguf_int8_kv_context(
            kv_storage_dtype=self.kv_storage_dtype,
            max_positions=rounded_positions,
            bf16_prefix_full_attention_layers=self.int8_bf16_prefix_full_attention_layers,
            bf16_full_attention_layer_indices=(
                self.int8_bf16_full_attention_layer_indices if custom_bf16_layers else None
            ),
            storage_layout=self.kv_storage_layout,
        )
        runtime = self.runtime or get_hip_runtime()
        if _gguf_host_token_embedding_requested():
            self._offload_token_embedding_to_host(runtime=runtime)
        build_kwargs = {
            "load": True,
            "compiler_version": self.compiler_version,
            "require_cached": self.require_cached_build,
        }
        with hip_target_arch_environment(self.runner.target_arch):
            self._runtime_state_library = build_runtime_state(**build_kwargs)
            self._lm_head_library = build_lm_head(**build_kwargs)
            if _gguf_verify_lm_head_q6_top1_dp4a_enabled():
                self._q6_pack8_library = build_gguf_q6_k_pack8_gemv(**build_kwargs)
            if self.use_expert_sidecar:
                self._expert_pack8_library = build_gguf_expert_pack8_gemv(**build_kwargs)
        if self.use_expert_sidecar:
            setattr(self.runner, "_expert_pack8_library", self._expert_pack8_library)
            self._expert_sidecar_reader = GGUFReader(self.model_path)
            self._expert_sidecar_model_map = build_qwen35_gguf_tensor_map(self._expert_sidecar_reader.info)
            if self.preload_expert_sidecars:
                self._expert_sidecar_host_layers = {
                    layer_id: self._load_expert_sidecar_host_layer(layer_id)
                    for layer_id in range(self.runner.weights.config.block_count)
                }
        self._target_scratch_owner = _FullStackScratch.allocate(
            self.runner,
            runtime=runtime,
            max_sequence_length=self.max_sequence_length,
            max_batch_size=self.max_batch_size,
            kv_storage_dtype=self.kv_storage_dtype,
            kv_storage_layout=self.kv_storage_layout,
            kv_scale_dtype=self.kv_scale_dtype,
            kv_scale_granularity=self.kv_scale_granularity,
            int8_kv_value_bf16=self.int8_kv_value_bf16,
            int8_bf16_prefix_full_attention_layers=self.int8_bf16_prefix_full_attention_layers,
            int8_bf16_full_attention_layer_indices=self.int8_bf16_full_attention_layer_indices,
            allocate_kv_cache=not bool(self.defer_kv_allocation),
        )
        self.scratch = self._target_scratch_owner.for_slot(0)
        self._target_layout = Qwen35GGUFResidentTargetLayout(
            max_batch_size=self.max_batch_size,
            hidden_size=self.runner.hidden_size,
            vocab_size=self.runner.vocab_size,
            max_sequence_length=int(self.scratch.max_positions),
            block_size=int(self.scratch.block_size),
        )
        self._device_kv_layout = _qwen35_gguf_session_kv_chunk_layout(self)
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
        self._token_host = np.empty((self.max_batch_size,), dtype=np.int64)
        self._token_buf = malloc(self._token_host.nbytes, runtime=runtime)
        hidden_bytes = self.runner.hidden_size * 2
        self._hidden_a = malloc(self.max_batch_size * hidden_bytes, runtime=runtime)
        self._hidden_b = malloc(self.max_batch_size * hidden_bytes, runtime=runtime)
        self._logits_host = np.empty((1, self.runner.vocab_size), dtype=np.float32)
        self._logits_buf = malloc(
            self.max_batch_size * self.runner.vocab_size * DType.FP32.itemsize,
            runtime=runtime,
        )
        native_cu = np.arange(self.max_batch_size + 1, dtype=np.int32)
        native_state_indices = np.arange(self.max_batch_size, dtype=np.int64)
        self._native_cu_seqlens_buf = malloc(native_cu.nbytes, runtime=runtime)
        self._native_state_indices_buf = malloc(native_state_indices.nbytes, runtime=runtime)
        copy_host_to_device(
            self._native_cu_seqlens_buf,
            host_array_ptr(native_cu),
            native_cu.nbytes,
            runtime=runtime,
        )
        copy_host_to_device(
            self._native_state_indices_buf,
            host_array_ptr(native_state_indices),
            native_state_indices.nbytes,
            runtime=runtime,
        )
        self._native_token_ids_host = np.empty((self.max_batch_size,), dtype=np.int32)
        self._lm_head_threads = 128
        self._lm_head_stage1_blocks = lm_head_argmax_stage1_blocks(self.runner.vocab_size, threads=self._lm_head_threads)
        self._lm_block_values = malloc(
            self.max_batch_size * self._lm_head_stage1_blocks * DType.FP32.itemsize,
            runtime=runtime,
        )
        self._lm_block_indices = malloc(
            self.max_batch_size * self._lm_head_stage1_blocks * DType.INT64.itemsize,
            runtime=runtime,
        )
        self._lm_out_index = malloc(self.max_batch_size * DType.INT64.itemsize, runtime=runtime)
        self._lm_out_value = malloc(self.max_batch_size * DType.FP32.itemsize, runtime=runtime)
        prefill_capacity = int(self.scratch.max_positions)
        prefill_rows = self._prefill_scratch_rows(prefill_capacity)
        alloc_capacity = prefill_capacity if self.use_expert_sidecar else prefill_rows
        self._prefill_token_buf = malloc(alloc_capacity * DType.INT64.itemsize, runtime=runtime)
        self._prefill_hidden_a = malloc(alloc_capacity * hidden_bytes, runtime=runtime)
        self._prefill_hidden_b = malloc(alloc_capacity * hidden_bytes, runtime=runtime)
        self._bulk_prefill_scratch = _GGUFFullAttentionPrefillScratch.allocate(
            self.runner,
            rows=prefill_rows,
            capacity=prefill_capacity,
            allocate_kv_cache=False,
            runtime=runtime,
            runtime_state_library=self._runtime_state_library,
        )
        head_major_pair = _try_allocate_gguf_aotriton_head_major_kv_scratch(
            backend=self.backend,
            capacity_tokens=prefill_capacity,
            kv_width=self.runner.kv_width,
            runtime=runtime,
        )
        if head_major_pair is not None:
            head_major_key_cache, head_major_value_cache = head_major_pair
            self._bulk_prefill_scratch = replace(
                self._bulk_prefill_scratch,
                head_major_key_cache=head_major_key_cache,
                head_major_value_cache=head_major_value_cache,
                head_major_kv_capacity=prefill_capacity,
                buffers=(*self._bulk_prefill_scratch.buffers, *head_major_pair),
            )
        self._buffers = (
            self._token_buf,
            self._hidden_a,
            self._hidden_b,
            self._logits_buf,
            self._native_cu_seqlens_buf,
            self._native_state_indices_buf,
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
        if self.prefill_flight_recorder_path is not None:
            self._prefill_flight_recorder = PrefillFlightRecorder(
                self.prefill_flight_recorder_path,
                runtime=runtime,
                marker_library=self._runtime_state_library,
                granularity=self.prefill_flight_recorder_granularity,
            )
        self._decode_graph_min_replay_steps_cache = self._resolve_decode_graph_min_replay_steps()

    @property
    def position(self) -> int:
        """Next token position that will be consumed by :meth:`step`."""

        return int(self._position)

    @property
    def device_kv_allocation(self) -> DeviceKVPoolAllocation | None:
        return self._device_kv_allocation

    @property
    def device_kv_capacity_tokens(self) -> int:
        allocation = self._device_kv_allocation
        if allocation is not None:
            return len(allocation.block_ids) * 256
        if self.scratch is None:
            return 0
        return 0 if self.defer_kv_allocation else int(self.scratch.max_positions)

    def bind_device_kv_allocation(
        self,
        pool: DeviceChunkedKVPool,
        allocation: DeviceKVPoolAllocation,
    ) -> None:
        """Bind one scheduler-owned allocation to this otherwise resident session."""

        if not self.defer_kv_allocation:
            raise RuntimeError("GGUF session was not created for deferred KV allocation")
        if self.scratch is None:
            raise RuntimeError("GGUF resident session is closed")
        if self._device_kv_allocation is not None:
            raise RuntimeError("GGUF resident session already has a device KV allocation")
        backing = allocation.backing
        expected_layout = self._device_kv_layout
        if expected_layout is None:
            expected_layout = _qwen35_gguf_session_kv_chunk_layout(self)
            self._device_kv_layout = expected_layout
        if getattr(backing, "layout", None) != expected_layout:
            raise TypeError("GGUF device KV allocation layout does not match the resident session")
        if not isinstance(backing, Qwen35GGUFKVChunkBacking):
            raise TypeError("GGUF device KV allocation has incompatible backing")
        if len(allocation.block_ids) > int(self.scratch.block_table_tensor.numel):
            raise ValueError("GGUF device KV allocation exceeds the session block-table capacity")
        if int(allocation.chunk_start_block_id) != int(backing.start_block_id):
            raise ValueError("GGUF device KV allocation backing identity mismatch")
        backing.validate_bound_blocks(allocation.block_ids)
        local_block_table = np.zeros(self.scratch.block_table_tensor.shape, dtype=np.int32)
        local_block_table[: len(allocation.block_ids)] = np.asarray(
            [
                int(block_id) - int(backing.start_block_id)
                for block_id in allocation.block_ids
            ],
            dtype=np.int32,
        )
        copy_host_to_device(
            self.scratch.block_table,
            host_array_ptr(local_block_table),
            local_block_table.nbytes,
            runtime=self.runtime or get_hip_runtime(),
        )
        self.scratch = replace(
            self.scratch,
            full_key_caches=backing.full_key_caches,
            full_value_caches=backing.full_value_caches,
            full_bf16_mirror_key_caches=backing.full_bf16_mirror_key_caches,
            full_bf16_mirror_value_caches=backing.full_bf16_mirror_value_caches,
            full_k_scale_caches=backing.full_k_scale_caches,
            full_v_scale_caches=backing.full_v_scale_caches,
            full_kv_scale_metadata=backing.full_kv_scale_metadata,
        )
        self._device_kv_pool = pool
        self._device_kv_allocation = allocation

    def capture_prefix_state_snapshot(
        self,
        *,
        position: int | None = None,
        stream: int = 0,
    ) -> Qwen35GGUFPrefixStateSnapshot:
        """Capture cache-owned Conv/GDN state at the exact current KV boundary."""

        if self.scratch is None:
            raise RuntimeError("GGUF prefix snapshot requires a live resident session")
        boundary = int(self.position if position is None else position)
        if boundary != int(self.position):
            raise ValueError("GGUF prefix snapshot position must equal the current boundary")
        if boundary <= 0 or boundary % 256 != 0:
            raise ValueError("GGUF prefix snapshot requires a positive 256-token-aligned boundary")
        if boundary >= int(self.scratch.max_positions):
            raise ValueError("GGUF prefix snapshot boundary leaves no continuation capacity")
        if bool(getattr(self, "_packed_decode_state_dirty", False)):
            raise RuntimeError("GGUF prefix snapshot source has unflushed packed state")
        allocation = self._device_kv_allocation
        if allocation is None:
            raise RuntimeError("GGUF prefix snapshot requires bound device KV")
        prefix_pages = boundary // 256
        block_ids = tuple(int(block_id) for block_id in allocation.block_ids[:prefix_pages])
        if len(block_ids) != prefix_pages:
            raise ValueError("GGUF prefix snapshot allocation does not cover its boundary")

        runtime = self.runtime or get_hip_runtime()
        conv_backups: list[DeviceBuffer | None] = []
        recurrent_backups: list[DeviceBuffer | None] = []
        allocated: list[DeviceBuffer] = []
        try:
            for layer_id, (conv_state, recurrent_state) in enumerate(
                zip(
                    self.scratch.layer_conv_states,
                    self.scratch.layer_recurrent_states,
                    strict=True,
                )
            ):
                if (conv_state is None) != (recurrent_state is None):
                    raise ValueError(
                        f"GGUF prefix snapshot partial linear state at layer {layer_id}"
                    )
                if conv_state is None:
                    conv_backups.append(None)
                    recurrent_backups.append(None)
                    continue
                assert recurrent_state is not None
                conv_backup = malloc(conv_state.nbytes, runtime=runtime)
                allocated.append(conv_backup)
                recurrent_backup = malloc(recurrent_state.nbytes, runtime=runtime)
                allocated.append(recurrent_backup)
                conv_backups.append(conv_backup)
                recurrent_backups.append(recurrent_backup)
                runtime.memcpy_async(
                    conv_backup.ptr,
                    conv_state.ptr,
                    conv_state.nbytes,
                    HipMemcpyKind.DEVICE_TO_DEVICE,
                    int(stream),
                )
                runtime.memcpy_async(
                    recurrent_backup.ptr,
                    recurrent_state.ptr,
                    recurrent_state.nbytes,
                    HipMemcpyKind.DEVICE_TO_DEVICE,
                    int(stream),
                )
            if stream:
                runtime.stream_synchronize(int(stream))
            else:
                runtime.device_synchronize()
        except Exception:
            for buffer in reversed(allocated):
                free(buffer, runtime=runtime)
            raise
        return Qwen35GGUFPrefixStateSnapshot(
            runtime=runtime,
            runner=self.runner,
            kv_storage_dtype=self.kv_storage_dtype,
            kv_storage_layout=self.kv_storage_layout,
            position=boundary,
            block_ids=block_ids,
            backing=allocation.backing,
            layer_conv_states=tuple(conv_backups),
            layer_recurrent_states=tuple(recurrent_backups),
        )

    def clone_prefix_state_from_snapshot(
        self,
        snapshot: Qwen35GGUFPrefixStateSnapshot,
        *,
        stream: int = 0,
    ) -> int:
        """Restore an exact historical hybrid boundary beside shared KV pages."""

        if snapshot.closed:
            raise RuntimeError("GGUF prefix state snapshot is closed")
        if self.scratch is None:
            raise RuntimeError("GGUF prefix snapshot clone requires a live destination")
        boundary = int(snapshot.position)
        if boundary <= 0 or boundary % 256 != 0:
            raise ValueError("GGUF prefix clone requires a positive 256-token-aligned boundary")
        if boundary >= int(self.scratch.max_positions):
            raise ValueError("GGUF prefix clone boundary exceeds destination capacity")
        runtime = self.runtime or get_hip_runtime()
        if snapshot.runner is not self.runner or snapshot.runtime is not runtime:
            raise ValueError("GGUF prefix snapshot clone requires one runner and HIP runtime")
        if snapshot.kv_storage_dtype != self.kv_storage_dtype:
            raise ValueError("GGUF prefix snapshot clone requires matching KV storage dtype")
        if snapshot.kv_storage_layout != self.kv_storage_layout:
            raise ValueError("GGUF prefix snapshot clone requires matching KV storage layout")
        if int(self.position) != 0:
            raise ValueError("GGUF prefix clone destination must be reset")
        if bool(getattr(self, "_packed_decode_state_dirty", False)):
            raise RuntimeError("GGUF prefix clone destination has unflushed packed state")
        if self._device_kv_graph_handles or any(
            not bool(getattr(graph, "closed", False))
            for graph in tuple(self._decode_graphs)
        ):
            raise RuntimeError("GGUF prefix clone destination still owns live graphs")
        allocation = self._device_kv_allocation
        if allocation is None:
            raise RuntimeError("GGUF prefix snapshot clone requires bound device KV")
        if (
            tuple(int(block_id) for block_id in allocation.reused_block_ids)
            != snapshot.block_ids
        ):
            raise ValueError("GGUF prefix clone allocation does not share the snapshot boundary")
        if (
            tuple(
                int(block_id)
                for block_id in allocation.block_ids[: len(snapshot.block_ids)]
            )
            != snapshot.block_ids
        ):
            raise ValueError("GGUF prefix clone block table does not match the snapshot")
        if allocation.backing is not snapshot.backing:
            raise ValueError("GGUF prefix snapshot and destination must share one backing")

        destination_conv = tuple(self.scratch.layer_conv_states)
        destination_recurrent = tuple(self.scratch.layer_recurrent_states)
        if not (
            len(destination_conv)
            == len(destination_recurrent)
            == len(snapshot.layer_conv_states)
            == len(snapshot.layer_recurrent_states)
        ):
            raise ValueError("GGUF prefix snapshot linear-state layer count mismatch")
        copied_bytes = 0
        for layer_id, (dst_conv, dst_recurrent, src_conv, src_recurrent) in enumerate(
            zip(
                destination_conv,
                destination_recurrent,
                snapshot.layer_conv_states,
                snapshot.layer_recurrent_states,
                strict=True,
            )
        ):
            if (dst_conv is None) != (src_conv is None) or (
                (dst_recurrent is None) != (src_recurrent is None)
            ):
                raise ValueError(
                    f"GGUF prefix snapshot linear-state layout mismatch at layer {layer_id}"
                )
            if dst_conv is None:
                if not (
                    dst_recurrent is None and src_conv is None and src_recurrent is None
                ):
                    raise ValueError(
                        f"GGUF prefix snapshot partial linear state at layer {layer_id}"
                    )
                continue
            assert dst_recurrent is not None
            assert src_conv is not None and src_recurrent is not None
            if int(dst_conv.nbytes) != int(src_conv.nbytes):
                raise ValueError(f"GGUF prefix snapshot Conv size mismatch at layer {layer_id}")
            if int(dst_recurrent.nbytes) != int(src_recurrent.nbytes):
                raise ValueError(
                    f"GGUF prefix snapshot recurrent size mismatch at layer {layer_id}"
                )
            runtime.memcpy_async(
                dst_conv.ptr,
                src_conv.ptr,
                src_conv.nbytes,
                HipMemcpyKind.DEVICE_TO_DEVICE,
                int(stream),
            )
            runtime.memcpy_async(
                dst_recurrent.ptr,
                src_recurrent.ptr,
                src_recurrent.nbytes,
                HipMemcpyKind.DEVICE_TO_DEVICE,
                int(stream),
            )
            copied_bytes += int(src_conv.nbytes) + int(src_recurrent.nbytes)
        self._commit_prefix_state_clone(boundary, stream=int(stream))
        if stream:
            runtime.stream_synchronize(int(stream))
        else:
            runtime.device_synchronize()
        return copied_bytes

    def clone_prefix_state_from(
        self,
        source: "Qwen35GGUFResidentSession",
        *,
        position: int | None = None,
        stream: int = 0,
    ) -> int:
        """Clone exact current-boundary hybrid state from a shared-KV source.

        Full-attention K/V pages are shared by the device pool; this method
        copies the Conv/GDN state that is not page-addressable.  Only the
        source's current complete-page boundary is valid.  Historical prefixes
        require separately retained per-boundary linear-state snapshots.
        """

        if source is self:
            raise ValueError("GGUF prefix state source and destination must differ")
        if self.scratch is None or source.scratch is None:
            raise RuntimeError("GGUF prefix state clone requires live resident sessions")
        boundary = int(source.position if position is None else position)
        if boundary != int(source.position):
            raise ValueError("GGUF prefix clone position must equal the current source boundary")
        if boundary <= 0 or boundary % 256 != 0:
            raise ValueError("GGUF prefix clone requires a positive 256-token-aligned boundary")
        if boundary >= int(self.scratch.max_positions):
            raise ValueError("GGUF prefix clone boundary exceeds destination capacity")
        if self.runner is not source.runner:
            raise ValueError("GGUF prefix clone requires sessions from one shared runner")
        if self.runtime is not source.runtime:
            raise ValueError("GGUF prefix clone requires one shared HIP runtime")
        if self.kv_storage_dtype != source.kv_storage_dtype:
            raise ValueError("GGUF prefix clone requires matching KV storage dtype")
        if self.kv_storage_layout != source.kv_storage_layout:
            raise ValueError("GGUF prefix clone requires matching KV storage layout")
        if int(self.position) != 0:
            raise ValueError("GGUF prefix clone destination must be reset")
        if bool(getattr(source, "_packed_decode_state_dirty", False)):
            raise RuntimeError("GGUF prefix clone source has unflushed packed state")
        if bool(getattr(self, "_packed_decode_state_dirty", False)):
            raise RuntimeError("GGUF prefix clone destination has unflushed packed state")
        if self._device_kv_graph_handles or any(
            not bool(getattr(graph, "closed", False))
            for graph in tuple(self._decode_graphs)
        ):
            raise RuntimeError("GGUF prefix clone destination still owns live graphs")

        source_allocation = source._device_kv_allocation
        destination_allocation = self._device_kv_allocation
        if source_allocation is None or destination_allocation is None:
            raise RuntimeError("GGUF prefix clone requires bound device KV allocations")
        prefix_pages = boundary // 256
        source_prefix = tuple(int(block_id) for block_id in source_allocation.block_ids[:prefix_pages])
        destination_prefix = tuple(
            int(block_id) for block_id in destination_allocation.reused_block_ids
        )
        if len(source_prefix) != prefix_pages or destination_prefix != source_prefix:
            raise ValueError("GGUF prefix clone allocation does not share the exact source boundary")
        if tuple(int(block_id) for block_id in destination_allocation.block_ids[:prefix_pages]) != source_prefix:
            raise ValueError("GGUF prefix clone block-table prefix does not match the source")
        if destination_allocation.backing is not source_allocation.backing:
            raise ValueError("GGUF prefix clone allocations must share one backing")

        source_conv = tuple(source.scratch.layer_conv_states)
        source_recurrent = tuple(source.scratch.layer_recurrent_states)
        destination_conv = tuple(self.scratch.layer_conv_states)
        destination_recurrent = tuple(self.scratch.layer_recurrent_states)
        if not (
            len(source_conv)
            == len(source_recurrent)
            == len(destination_conv)
            == len(destination_recurrent)
        ):
            raise ValueError("GGUF prefix clone linear-state layer count mismatch")
        copies: list[tuple[DeviceBuffer, DeviceBuffer]] = []
        for layer_id, (src_conv, src_recurrent, dst_conv, dst_recurrent) in enumerate(
            zip(
                source_conv,
                source_recurrent,
                destination_conv,
                destination_recurrent,
                strict=True,
            )
        ):
            source_missing = src_conv is None or src_recurrent is None
            destination_missing = dst_conv is None or dst_recurrent is None
            if source_missing != destination_missing:
                raise ValueError(f"GGUF prefix clone linear-state layout mismatch at layer {layer_id}")
            if source_missing:
                if not (src_conv is None and src_recurrent is None and dst_conv is None and dst_recurrent is None):
                    raise ValueError(f"GGUF prefix clone partial linear state at layer {layer_id}")
                continue
            assert src_conv is not None and src_recurrent is not None
            assert dst_conv is not None and dst_recurrent is not None
            if int(src_conv.nbytes) != int(dst_conv.nbytes):
                raise ValueError(f"GGUF prefix clone Conv state size mismatch at layer {layer_id}")
            if int(src_recurrent.nbytes) != int(dst_recurrent.nbytes):
                raise ValueError(f"GGUF prefix clone recurrent state size mismatch at layer {layer_id}")
            copies.extend(((dst_conv, src_conv), (dst_recurrent, src_recurrent)))

        runtime = self.runtime or get_hip_runtime()
        copied_bytes = 0
        for destination, origin in copies:
            runtime.memcpy_async(
                destination.ptr,
                origin.ptr,
                origin.nbytes,
                HipMemcpyKind.DEVICE_TO_DEVICE,
                int(stream),
            )
            copied_bytes += int(origin.nbytes)
        self._commit_prefix_state_clone(boundary, stream=int(stream))
        return copied_bytes

    def _commit_prefix_state_clone(self, boundary: int, *, stream: int) -> None:
        self._set_full_attention_position_device(int(boundary), stream=int(stream))
        self._position = int(boundary)
        self._hidden_seed_fp32_populated = False
        self._last_pre_output_norm_hidden = None
        self._last_layer_output_hidden = {}
        self._verify_hidden_seed_rows_populated = 0
        self._packed_verify_session_ids = ()
        self._packed_verify_max_written_positions = ()
        self._packed_decode_sessions = ()
        self._packed_decode_last_layout = None
        self._packed_decode_state_dirty = False
        self._packed_decode_session_ids = ()
        self._packed_decode_positions = ()
        self.last_packed_execution_manifest = {}
        self._last_packed_lm_head_decode_path = "unobserved"
        self._last_packed_sampler_decode_path = "unobserved"

    def invalidate_device_kv_graphs(self) -> int:
        """Close every graph that pins this session's current KV allocation."""

        handles = {
            id(handle): handle
            for handle in (*tuple(self._device_kv_graph_handles.values()), *tuple(self._decode_graphs))
        }
        closed = 0
        for handle in tuple(handles.values()):
            close = getattr(handle, "close", None)
            if callable(close) and not bool(getattr(handle, "closed", False)):
                close()
                closed += 1
        return closed

    def unbind_device_kv_allocation(self) -> DeviceKVPoolAllocation:
        """Detach scheduler-owned KV after all graph pins have been invalidated."""

        if self.scratch is None:
            raise RuntimeError("GGUF resident session is closed")
        allocation = self._device_kv_allocation
        if allocation is None:
            raise RuntimeError("GGUF resident session has no device KV allocation")
        if self._device_kv_graph_handles:
            raise RuntimeError("cannot detach GGUF device KV while graphs remain pinned")
        empty = tuple(None for _ in self.scratch.full_key_caches)
        self.scratch = replace(
            self.scratch,
            full_key_caches=empty,
            full_value_caches=empty,
            full_bf16_mirror_key_caches=empty,
            full_bf16_mirror_value_caches=empty,
            full_k_scale_caches=empty,
            full_v_scale_caches=empty,
            full_kv_scale_metadata=empty,
        )
        block_table = np.zeros(self.scratch.block_table_tensor.shape, dtype=np.int32)
        copy_host_to_device(
            self.scratch.block_table,
            host_array_ptr(block_table),
            block_table.nbytes,
            runtime=self.runtime or get_hip_runtime(),
        )
        self._device_kv_allocation = None
        self._device_kv_pool = None
        return allocation

    def _pin_device_kv_graph(self, graph: object) -> None:
        allocation = self._device_kv_allocation
        pool = self._device_kv_pool
        if allocation is None or pool is None:
            return
        key = id(graph)
        if key in self._device_kv_graph_handles:
            return
        pool.pin(allocation.block_ids)
        self._device_kv_graph_handles[key] = graph

    def _unpin_device_kv_graph(self, graph: object) -> None:
        key = id(graph)
        if key not in self._device_kv_graph_handles:
            return
        allocation = self._device_kv_allocation
        pool = self._device_kv_pool
        if allocation is None or pool is None:
            raise RuntimeError("GGUF device KV graph pin outlived its allocation")
        pool.unpin(allocation.block_ids)
        self._device_kv_graph_handles.pop(key, None)

    def create_device_kv_pool(
        self,
        *,
        initial_pages: int,
        low_water_pages: int,
        high_water_pages: int | None,
        chunk_pages: int,
        idle_grace_seconds: float,
    ) -> DeviceChunkedKVPool:
        """Create the scheduler-owned policy-shaped pool from this session plan."""

        if not self.defer_kv_allocation:
            raise RuntimeError("GGUF session was not created for deferred KV allocation")
        if self.runner is None or self.runner.weights is None or self.scratch is None:
            raise RuntimeError("GGUF resident session is closed")
        runtime = self.runtime or get_hip_runtime()
        cfg = self.runner.weights.config
        layout = self._device_kv_layout
        if layout is None:
            layout = _qwen35_gguf_session_kv_chunk_layout(self)
            self._device_kv_layout = layout
        page_bytes = _qwen35_gguf_kv_page_bytes(cfg, layout)

        def allocate_chunk(start_block_id: int, pages: int):
            return _allocate_qwen35_gguf_kv_chunk(
                self.runner,
                runtime=runtime,
                start_block_id=int(start_block_id),
                pages=int(pages),
                layout=layout,
            )

        return DeviceChunkedKVPool(
            page_bytes=page_bytes,
            initial_pages=int(initial_pages),
            low_water_pages=int(low_water_pages),
            high_water_pages=(None if high_water_pages is None else int(high_water_pages)),
            chunk_pages=int(chunk_pages),
            idle_grace_seconds=float(idle_grace_seconds),
            allocate_chunk=allocate_chunk,
            free_chunk=lambda backing: _free_qwen35_gguf_kv_chunk(
                backing, runtime=runtime
            ),
            page_pointer=lambda backing, local_page: backing.page_pointer(local_page),
        )

    def decode_graph_min_replay_steps(self) -> int | None:
        """Return this backend package's admitted graph break-even, if any."""

        return self._decode_graph_min_replay_steps_cache

    def _resolve_decode_graph_min_replay_steps(self) -> int | None:
        """Resolve backend graph capability once after resident initialization."""

        if (
            self.runner is None
            or self.runner.weights is None
            or self.scratch is None
            or self.host_token_embedding_enabled
            or self.kv_storage_dtype != DType.BF16
            or not bool(self.use_gemv_decode)
            or _gguf_moe_graph_enabled()
        ):
            return None
        quant_keys = [str(weight.spec.quant_key) for weight in self.runner.weights.weights]
        if not any(key.endswith("_t16_v1") or key.endswith("_x8_v1") for key in quant_keys):
            return None
        raw = backend_package_capability(
            self.runner.backend,
            "GGUF_DECODE_GRAPH_MIN_REPLAY_STEPS",
        )
        if raw is None:
            return None
        minimum = int(raw)
        if minimum <= 0:
            raise RuntimeError("backend GGUF decode graph minimum must be positive")
        return minimum

    @property
    def last_pre_output_norm_hidden(self) -> np.ndarray | None:
        """Return the last diagnostic pre-output_norm decode row, if captured."""

        if self._last_pre_output_norm_hidden is None:
            return None
        return np.ascontiguousarray(self._last_pre_output_norm_hidden, dtype=np.float32)

    @property
    def last_layer_output_hidden(self) -> dict[int, np.ndarray]:
        """Return diagnostic post-layer decode rows captured by the last step."""

        return {
            int(layer_id): np.ascontiguousarray(hidden, dtype=np.float32)
            for layer_id, hidden in self._last_layer_output_hidden.items()
        }

    def _normalize_layer_output_capture(
        self,
        layer_ids: list[int] | tuple[int, ...] | set[int] | None,
    ) -> set[int]:
        if layer_ids is None:
            return set()
        if self.runner is None or self.runner.weights is None:
            raise RuntimeError("GGUF resident session is closed")
        layer_count = len(self.runner.weights.config.layer_types)
        normalized: set[int] = set()
        for layer_id in layer_ids:
            value = int(layer_id)
            if value < 0 or value >= layer_count:
                raise ValueError(f"layer_id {value} outside resident layer range [0, {layer_count})")
            normalized.add(value)
        return normalized

    def _capture_verify_layer_boundary_rows(
        self,
        layer_id: int,
        layer_type: str,
        *,
        hidden_in_ptr: int,
        hidden_in_f32_ptr: int | None,
        layer_out_ptr: int,
        layer_out_f32_ptr: int | None,
        scratch,
        rows: int,
        runtime: HipRuntime,
    ) -> dict[str, np.ndarray]:
        """Copy scored bulk-verifier layer internals for parity diagnostics.

        This is deliberately host-copy heavy and opt-in.  It captures the
        tensors produced by the same bulk/direct-state verifier pass that scored
        the block, unlike the isolated single-row layer probes.
        """

        if self.runner is None or self.runner.weights is None:
            raise RuntimeError("GGUF resident session is closed")
        cfg = self.runner.weights.config
        hidden_size = int(self.runner.hidden_size)
        row_count = int(rows)
        if row_count <= 0:
            raise ValueError("rows must be positive")

        def copy_bf16(name: str, ptr: int, shape: tuple[int, ...]) -> tuple[str, np.ndarray]:
            elements = int(np.prod(shape))
            return name, _copy_bf16_ptr_to_host_f32(int(ptr), elements, runtime=runtime).reshape(shape)

        def copy_f32(name: str, ptr: int, shape: tuple[int, ...]) -> tuple[str, np.ndarray]:
            elements = int(np.prod(shape))
            return name, _copy_f32_ptr_to_host(int(ptr), elements, runtime=runtime).reshape(shape)

        def copy_i64(name: str, ptr: int, shape: tuple[int, ...]) -> tuple[str, np.ndarray]:
            elements = int(np.prod(shape))
            return name, _copy_i64_ptr_to_host(int(ptr), elements, runtime=runtime).reshape(shape)

        arrays: dict[str, np.ndarray] = {}
        if hidden_in_f32_ptr is not None:
            key, value = copy_f32("hidden_in", hidden_in_f32_ptr, (row_count, hidden_size))
        else:
            key, value = copy_bf16("hidden_in", hidden_in_ptr, (row_count, hidden_size))
        arrays[key] = np.ascontiguousarray(value, dtype=np.float32)
        key, value = copy_bf16("attn_norm", int(scratch.norm.ptr), (row_count, hidden_size))
        arrays[key] = np.ascontiguousarray(value, dtype=np.float32)
        if (
            hidden_in_f32_ptr is not None
            and hasattr(scratch, "post_norm_f32")
            and _gguf_verify_f32_attention_norm_enabled()
            and not _gguf_verify_f32_post_norm_enabled()
        ):
            key, value = copy_f32(
                "attn_norm_f32_scratch",
                int(scratch.post_norm_f32.ptr),
                (row_count, hidden_size),
            )
            arrays[key] = np.ascontiguousarray(value, dtype=np.float32)

        if str(layer_type) == LINEAR_ATTENTION:
            use_f32_linear_projection_capture = (
                _gguf_verify_f32_linear_projections_enabled()
                and hasattr(scratch, "linear_qkv_f32")
                and hasattr(scratch, "linear_z_f32")
                and hasattr(scratch, "linear_alpha_f32")
                and hasattr(scratch, "linear_beta_f32")
            )
            linear_qkv_shape = (row_count, int(self.runner.linear_qkv_width))
            linear_z_shape = (row_count, int(cfg.ssm_inner_size))
            linear_ab_shape = (row_count, int(cfg.ssm_time_step_rank))
            if use_f32_linear_projection_capture:
                key, value = copy_f32("linear_qkv", int(scratch.linear_qkv_f32.ptr), linear_qkv_shape)
                arrays[key] = np.ascontiguousarray(value, dtype=np.float32)
                key, value = copy_bf16(
                    "linear_qkv_bf16_mirror",
                    int(scratch.linear_qkv.ptr),
                    linear_qkv_shape,
                )
                arrays[key] = np.ascontiguousarray(value, dtype=np.float32)
            else:
                key, value = copy_bf16("linear_qkv", int(scratch.linear_qkv.ptr), linear_qkv_shape)
                arrays[key] = np.ascontiguousarray(value, dtype=np.float32)
            if use_f32_linear_projection_capture:
                key, value = copy_f32("linear_z", int(scratch.linear_z_f32.ptr), linear_z_shape)
                arrays[key] = np.ascontiguousarray(value, dtype=np.float32)
                key, value = copy_bf16(
                    "linear_z_bf16_mirror",
                    int(scratch.linear_z.ptr),
                    linear_z_shape,
                )
                arrays[key] = np.ascontiguousarray(value, dtype=np.float32)
            else:
                key, value = copy_bf16("linear_z", int(scratch.linear_z.ptr), linear_z_shape)
                arrays[key] = np.ascontiguousarray(value, dtype=np.float32)
            if use_f32_linear_projection_capture:
                key, value = copy_f32("ssm_alpha", int(scratch.linear_alpha_f32.ptr), linear_ab_shape)
                arrays[key] = np.ascontiguousarray(value, dtype=np.float32)
                key, value = copy_bf16(
                    "ssm_alpha_bf16_mirror",
                    int(scratch.linear_alpha.ptr),
                    linear_ab_shape,
                )
                arrays[key] = np.ascontiguousarray(value, dtype=np.float32)
            else:
                key, value = copy_bf16("ssm_alpha", int(scratch.linear_alpha.ptr), linear_ab_shape)
                arrays[key] = np.ascontiguousarray(value, dtype=np.float32)
            if use_f32_linear_projection_capture:
                key, value = copy_f32("ssm_beta", int(scratch.linear_beta_f32.ptr), linear_ab_shape)
                arrays[key] = np.ascontiguousarray(value, dtype=np.float32)
                key, value = copy_bf16(
                    "ssm_beta_bf16_mirror",
                    int(scratch.linear_beta.ptr),
                    linear_ab_shape,
                )
                arrays[key] = np.ascontiguousarray(value, dtype=np.float32)
            else:
                key, value = copy_bf16("ssm_beta", int(scratch.linear_beta.ptr), linear_ab_shape)
                arrays[key] = np.ascontiguousarray(value, dtype=np.float32)
            if not _gguf_verify_f32_attn_out_enabled():
                key, value = copy_f32(
                    "conv_out",
                    int(scratch.conv_out.ptr),
                    (row_count, int(self.runner.linear_qkv_width)),
                )
                arrays[key] = np.ascontiguousarray(value, dtype=np.float32)
            if not (
                _gguf_verify_capture_prefill_gdn_enabled()
                or _gguf_verify_capture_score_prefill_enabled()
                or _gguf_verify_capture_bf16_gdn_out_enabled()
            ):
                key, value = copy_f32(
                    "recurrent_out",
                    int(scratch.recurrent_out.ptr),
                    (row_count, int(cfg.ssm_inner_size)),
                )
                arrays[key] = np.ascontiguousarray(value, dtype=np.float32)
            if (
                _gguf_verify_capture_prefill_gdn_enabled()
                or _gguf_verify_capture_score_prefill_enabled()
                or _gguf_verify_capture_bf16_gdn_out_enabled()
            ):
                key, value = copy_bf16(
                    "recurrent_bf16",
                    int(scratch.recurrent_bf16.ptr),
                    (row_count, int(cfg.ssm_inner_size)),
                )
                arrays[key] = np.ascontiguousarray(value, dtype=np.float32)

        key, value = copy_bf16("attn_out", int(scratch.attn_out.ptr), (row_count, hidden_size))
        arrays[key] = np.ascontiguousarray(value, dtype=np.float32)
        if layer_out_f32_ptr is None:
            key, value = copy_bf16("attn_residual", int(scratch.residual.ptr), (row_count, hidden_size))
            arrays[key] = np.ascontiguousarray(value, dtype=np.float32)
        key, value = copy_bf16("attn_post_norm_bf16", int(scratch.post_norm.ptr), (row_count, hidden_size))
        arrays[key] = np.ascontiguousarray(value, dtype=np.float32)
        arrays["attn_post_norm"] = arrays["attn_post_norm_bf16"]
        if (
            layer_out_f32_ptr is not None
            and hasattr(scratch, "post_norm_f32")
            and _gguf_verify_f32_post_norm_enabled()
        ):
            key, value = copy_f32("attn_post_norm_f32", int(scratch.post_norm_f32.ptr), (row_count, hidden_size))
            arrays[key] = np.ascontiguousarray(value, dtype=np.float32)

        if cfg.is_moe:
            top_k = int(cfg.expert_used_count)
            expert_count = int(cfg.expert_count)
            key, value = copy_f32(
                "moe_router_logits",
                int(scratch.moe_router_logits.ptr),
                (row_count, expert_count),
            )
            arrays[key] = np.ascontiguousarray(value, dtype=np.float32)
            if hasattr(scratch, "moe_shared_gate_logits"):
                key, value = copy_f32(
                    "moe_shared_gate",
                    int(scratch.moe_shared_gate_logits.ptr),
                    (row_count, 1),
                )
            else:
                shared_gate_ptr = int(scratch.moe_router_logits.ptr) + expert_count * DType.FP32.itemsize
                key, value = copy_f32("moe_shared_gate", shared_gate_ptr, (row_count, 1))
            arrays[key] = np.ascontiguousarray(value, dtype=np.float32)
            key, value = copy_i64(
                "moe_selected_experts",
                int(scratch.moe_selected_experts.ptr),
                (row_count, top_k),
            )
            arrays[key] = np.ascontiguousarray(value, dtype=np.int64)
            key, value = copy_f32(
                "moe_routing_weights",
                int(scratch.moe_routing_weights.ptr),
                (row_count, top_k),
            )
            arrays[key] = np.ascontiguousarray(value, dtype=np.float32)
            selected_shape = (row_count, top_k, int(cfg.expert_feed_forward_length))
            key, value = copy_bf16("moe_selected_swiglu", int(scratch.ffn_intermediate.ptr), selected_shape)
            arrays[key] = np.ascontiguousarray(value, dtype=np.float32)
            if (
                _gguf_verify_f32_selected_intermediate_enabled()
                and getattr(scratch, "ffn_intermediate_f32", None) is not None
            ):
                key, value = copy_f32(
                    "moe_selected_swiglu_f32_scratch",
                    int(scratch.ffn_intermediate_f32.ptr),
                    selected_shape,
                )
                arrays[key] = np.ascontiguousarray(value, dtype=np.float32)
            down_shape = (row_count, top_k, hidden_size)
            key, value = copy_bf16("ffn_or_moe_down", int(scratch.moe_down_out.ptr), down_shape)
            arrays[key] = np.ascontiguousarray(value, dtype=np.float32)
            if (
                _gguf_verify_f32_selected_down_enabled()
                and _gguf_verify_f32_moe_combine_enabled()
                and getattr(scratch, "moe_down_out_f32", None) is not None
            ):
                key, value = copy_f32("ffn_or_moe_down_f32_scratch", int(scratch.moe_down_out_f32.ptr), down_shape)
                arrays[key] = np.ascontiguousarray(value, dtype=np.float32)
            key, value = copy_bf16(
                "moe_shared_intermediate",
                int(scratch.moe_shared_intermediate.ptr),
                (row_count, int(cfg.expert_shared_feed_forward_length)),
            )
            arrays[key] = np.ascontiguousarray(value, dtype=np.float32)
            key, value = copy_bf16("moe_shared_out", int(scratch.moe_shared_out.ptr), (row_count, hidden_size))
            arrays[key] = np.ascontiguousarray(value, dtype=np.float32)
            if (
                _gguf_verify_f32_shared_down_enabled()
                and _gguf_verify_f32_selected_down_enabled()
                and _gguf_verify_f32_moe_combine_enabled()
                and getattr(scratch, "moe_shared_out_f32", None) is not None
            ):
                key, value = copy_f32(
                    "moe_shared_out_f32_scratch",
                    int(scratch.moe_shared_out_f32.ptr),
                    (row_count, hidden_size),
                )
                arrays[key] = np.ascontiguousarray(value, dtype=np.float32)

        if layer_out_f32_ptr is not None:
            key, value = copy_f32("layer_out", layer_out_f32_ptr, (row_count, hidden_size))
            arrays[key] = np.ascontiguousarray(value, dtype=np.float32)
            key, value = copy_bf16("layer_out_bf16", layer_out_ptr, (row_count, hidden_size))
            arrays[key] = np.ascontiguousarray(value, dtype=np.float32)
        else:
            key, value = copy_bf16("layer_out", layer_out_ptr, (row_count, hidden_size))
            arrays[key] = np.ascontiguousarray(value, dtype=np.float32)
        arrays["layer_type_id"] = np.full((row_count, 1), 0 if str(layer_type) == LINEAR_ATTENTION else 1, dtype=np.int64)
        return arrays

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

    def mtp_verify_seed(
        self,
        row_index: int,
        *,
        token_id: int,
        position: int,
        hidden_seed_base_ptr: int | None = None,
        hidden_seed_row_count: int | None = None,
    ) -> Qwen35GGUFMTPDraftSeed:
        """Describe one already-populated FP32 verifier row without copying it."""

        row = int(row_index)
        if hidden_seed_base_ptr is None:
            hidden_ptr = self.fp32_verify_hidden_seed_ptr(row)
            contract = self.fp32_verify_hidden_seed_contract(rows=1)
        else:
            base_ptr = int(hidden_seed_base_ptr)
            row_count = int(hidden_seed_row_count or 0)
            if base_ptr <= 0:
                raise ValueError("hidden_seed_base_ptr must be a non-zero device pointer")
            if row < 0 or row >= row_count:
                raise ValueError("row_index is outside external hidden rows")
            hidden_ptr = base_ptr + row * self.runner.hidden_size * DType.FP32.itemsize
            contract = qwen35_gguf_fp32_verify_hidden_seed_contract(
                self.runner.hidden_size,
                rows=1,
                populated_by_decode=True,
                source_buffer="Qwen35GGUFNativeAcceptCommitResult.hidden_seed_rows_ptr",
            )
        return Qwen35GGUFMTPDraftSeed(
            token_id=int(token_id),
            position=int(position),
            hidden_ptr=hidden_ptr,
            hidden_contract=contract,
        )

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

    @property
    def last_target_hidden(self) -> Tensor:
        """Final trunk hidden row that produced the most recent target sample.

        Target-attached draft models consume the pre-output-norm trunk row
        together with the next root token. The pointer is owned by the resident
        target session and is valid until the next target forward overwrites
        its scratch.
        """

        if self.runner is None or self._last_target_hidden_ptr == 0:
            raise RuntimeError("GGUF resident session has no completed target hidden row")
        return Tensor.from_handle(
            self._last_target_hidden_ptr,
            (1, self.runner.hidden_size),
            DType.BF16,
            Device("hip", 0),
        )

    @property
    def row_positions(self) -> tuple[int, ...]:
        """Next target-token position for every resident physical row."""

        if self._target_scratch_owner is None:
            raise RuntimeError("GGUF resident session is closed")
        positions = [int(value) for value in self._target_scratch_owner.position_host.tolist()]
        if positions:
            positions[0] = int(self._position)
        return tuple(positions)

    @property
    def target_layout(self) -> Qwen35GGUFResidentTargetLayout:
        if self._target_layout is None:
            raise RuntimeError("GGUF resident session is closed")
        return self._target_layout

    def target_spans(
        self,
        *,
        slot_indices: tuple[int, ...] | list[int] | None = None,
        span_role: str = "decode",
    ) -> KVLiveSpans:
        """Return contiguous per-row spans for the resident target executor."""

        if self._target_scratch_owner is None:
            raise RuntimeError("GGUF resident session is closed")
        owner = self._target_scratch_owner
        slots = (
            tuple(range(int(owner.slot_count)))
            if slot_indices is None
            else tuple(int(slot) for slot in slot_indices)
        )
        if not slots:
            raise ValueError("slot_indices must contain at least one slot")
        if slots != tuple(range(slots[0], slots[0] + len(slots))):
            raise ValueError("target spans currently require contiguous physical slots")
        if slots[0] < 0 or slots[-1] >= int(owner.slot_count):
            raise ValueError("target span slot outside resident capacity")
        if span_role not in {"decode", "verify_chain", "verify_tree"}:
            raise ValueError("span_role must be decode, verify_chain, or verify_tree")
        rows = len(slots)
        blocks = int(owner.blocks_per_slot)
        block_ptr = int(owner.block_table.ptr) + slots[0] * blocks * DType.INT32.itemsize
        position_ptr = int(owner.position_buf.ptr) + slots[0] * DType.INT64.itemsize
        context_ptr = int(owner.context_buf.ptr) + slots[0] * DType.INT64.itemsize
        block_table = Tensor.from_handle(
            block_ptr,
            (rows, blocks),
            DType.INT32,
            owner.block_table_tensor.device,
        )
        positions = Tensor.from_handle(
            position_ptr,
            (rows,),
            DType.INT64,
            owner.position_tensor.device,
        )
        contexts = Tensor.from_handle(
            context_ptr,
            (rows,),
            DType.INT64,
            owner.context_tensor.device,
        )
        max_live_count = max(int(owner.context_host[slot]) for slot in slots)
        return KVLiveSpans.paged_uniform(
            block_table=block_table,
            live_counts=contexts,
            max_live_count=max_live_count,
            storage_dtype=owner.kv_storage_dtype,
            row_positions=positions,
            span_role=span_role,
        )

    def select_prefill_quant(self, quant: str) -> None:
        """Select the four-axis GGUF prefill plugins for this session."""

        if self.runner is None:
            raise RuntimeError("GGUF resident session is closed")
        self.runner.select_prefill_quant(quant)

    def reset(self, *, stream: int = 0) -> None:
        """Reset resident target state without freeing weights or scratch."""

        if self.scratch is None or self._target_scratch_owner is None:
            raise RuntimeError("GGUF resident session is closed")
        runtime = self.runtime or get_hip_runtime()
        if self._reset_current_slot_only:
            self.scratch.zero_states(runtime, stream=stream, set_position=False)
            self._set_full_attention_position_device(0, stream=stream)
        else:
            self._target_scratch_owner.zero_states(runtime, stream=stream)
        self._position = 0
        self._last_target_hidden_ptr = 0
        self._hidden_seed_fp32_populated = False
        self._last_pre_output_norm_hidden = None
        self._last_layer_output_hidden = {}
        self._verify_hidden_seed_rows_populated = 0
        self._packed_verify_session_ids = ()
        self._packed_verify_max_written_positions = ()
        self._packed_decode_sessions = ()
        self._packed_decode_last_layout = None
        self._packed_decode_state_dirty = False
        self._packed_decode_session_ids = ()
        self._packed_decode_positions = ()
        self.last_packed_execution_manifest = {}
        self._last_packed_lm_head_decode_path = "unobserved"
        self._last_packed_sampler_decode_path = "unobserved"

    def compact_target_slots(
        self,
        source_slots: tuple[int, ...] | list[int],
        *,
        stream: int = 0,
    ) -> tuple[tuple[int, int], ...]:
        """Pack live target rows into the physical prefix and reclaim the tail."""

        if self._target_scratch_owner is None:
            raise RuntimeError("GGUF resident session is closed")
        owner = self._target_scratch_owner
        sources = tuple(int(slot) for slot in source_slots)
        if len(sources) > int(owner.slot_count):
            raise ValueError("target compaction exceeds resident slot capacity")
        if len(set(sources)) != len(sources):
            raise ValueError("target compaction source slots must be unique")
        if any(slot < 0 or slot >= int(owner.slot_count) for slot in sources):
            raise ValueError("target compaction source slot outside resident capacity")
        if sources != tuple(sorted(sources)):
            raise ValueError("target compaction currently requires stable ascending source slots")

        runtime = self.runtime or get_hip_runtime()
        old_positions = list(self.row_positions)
        moves = tuple((source, destination) for destination, source in enumerate(sources))

        def copy_slot(buffer: object | None, source: int, destination: int, *, live_only: bool) -> None:
            if buffer is None or source == destination:
                return
            slot_nbytes = int(buffer.nbytes) // int(owner.slot_count)
            copy_nbytes = slot_nbytes
            if live_only:
                row_nbytes, remainder = divmod(slot_nbytes, int(owner.max_positions))
                if remainder:
                    raise ValueError("resident target KV slot is not position-major")
                copy_nbytes = min(slot_nbytes, int(old_positions[source]) * row_nbytes)
            if copy_nbytes <= 0:
                return
            runtime.memcpy_async(
                int(buffer.ptr) + destination * slot_nbytes,
                int(buffer.ptr) + source * slot_nbytes,
                copy_nbytes,
                HipMemcpyKind.DEVICE_TO_DEVICE,
                int(stream),
            )

        state_buffers = (*owner.layer_conv_states, *owner.layer_recurrent_states)
        cache_buffers = (
            *owner.full_key_caches,
            *owner.full_value_caches,
            *owner.full_bf16_mirror_key_caches,
            *owner.full_bf16_mirror_value_caches,
            *owner.full_k_scale_caches,
            *owner.full_v_scale_caches,
        )
        copied = False
        for source, destination in moves:
            if source == destination:
                continue
            copied = True
            for buffer in state_buffers:
                copy_slot(buffer, source, destination, live_only=False)
            for buffer in cache_buffers:
                copy_slot(buffer, source, destination, live_only=True)
        if copied:
            if stream:
                runtime.stream_synchronize(int(stream))
            else:
                runtime.device_synchronize()

        compact_positions = [old_positions[source] for source in sources]
        compact_positions.extend([0] * (int(owner.slot_count) - len(compact_positions)))
        owner.set_full_attention_positions(tuple(compact_positions), runtime)
        self._position = int(compact_positions[0]) if compact_positions else 0
        return moves

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

    def _offload_token_embedding_to_host(self, *, runtime: HipRuntime) -> None:
        """Release the resident Q8_0 token embedding and serve lookups from host."""

        if self.runner is None or self.runner.weights is None:
            raise RuntimeError("GGUF resident session is closed")
        weights = self.runner.weights
        token_weight = weights.root("token_embedding")
        if token_weight.spec.layout != "raw_gguf" or token_weight.spec.quant_key != "gguf_q8_0":
            raise ValueError(
                f"{_GGUF_HOST_TOKEN_EMBEDDING_ENV}=1 requires a raw Q8_0 token embedding; "
                f"got layout={token_weight.spec.layout!r}, quant={token_weight.spec.quant_key!r}"
            )
        if "raw" not in token_weight.allocations:
            raise ValueError("token embedding has no raw device allocation to offload")
        aliased_slots = [
            slot for slot, weight in weights.root_weights.items() if slot != "token_embedding" and weight is token_weight
        ]
        if aliased_slots:
            aliases = ", ".join(sorted(aliased_slots))
            raise ValueError(
                f"{_GGUF_HOST_TOKEN_EMBEDDING_ENV}=1 cannot offload token_embedding because it is "
                f"aliased by device-resident root slot(s): {aliases}"
            )

        reader = GGUFReader(self.model_path)
        raw = reader.tensor_data(token_weight.spec.source.name)
        for allocation in reversed(tuple(token_weight.allocations.values())):
            allocation.free(runtime=runtime)
        host_weight = Qwen35GGUFDeviceWeight(
            spec=token_weight.spec,
            allocations=MappingProxyType({}),
            backend=token_weight.backend,
        )
        root_weights = dict(weights.root_weights)
        root_weights["token_embedding"] = host_weight
        self.runner.weights = Qwen35GGUFResidentWeights(
            config=weights.config,
            root_weights=MappingProxyType(root_weights),
            layers=weights.layers,
            backend=weights.backend,
        )
        self._host_token_embedding_reader = reader
        self._host_token_embedding_raw = raw
        self._host_token_embedding_cache = {}
        self.host_token_embedding_enabled = True
        self.host_token_embedding_reason = f"{_GGUF_HOST_TOKEN_EMBEDDING_ENV}=1"

    def _copy_token_embeddings_to_device(
        self,
        token_ids: np.ndarray,
        out_ptr: int,
        *,
        rows: int,
        token_ids_device_ptr: int | None = None,
        stream: int = 0,
    ) -> None:
        if self.runner is None or self.runner.weights is None:
            raise RuntimeError("GGUF resident session is closed")
        runtime = self.runtime or get_hip_runtime()
        token_arr = np.asarray(token_ids, dtype=np.int64).reshape(-1)
        if int(token_arr.size) != int(rows):
            raise ValueError(f"token row count mismatch: got {token_arr.size}, expected {rows}")
        if self.host_token_embedding_enabled:
            if stream != 0:
                raise RuntimeError("host token embedding is not compatible with non-default stream decode/graph capture")
            if self._host_token_embedding_raw is None:
                raise RuntimeError("host token embedding was enabled without host raw bytes")
            hidden = _q8_0_embedding_rows_to_bf16(
                self._host_token_embedding_raw,
                token_arr,
                hidden_size=self.runner.hidden_size,
                cache=self._host_token_embedding_cache,
            )
            nbytes = int(hidden.nbytes)
            copy_host_to_device(DeviceBuffer(int(out_ptr), nbytes), host_array_ptr(hidden), nbytes, runtime=runtime)
            return

        if token_ids_device_ptr is None:
            raise ValueError("token_ids_device_ptr is required for device token embedding")
        token_ids_device = DeviceBuffer(int(token_ids_device_ptr), int(token_arr.nbytes))
        if stream:
            runtime.memcpy_async(
                token_ids_device.ptr,
                host_array_ptr(token_arr),
                int(token_arr.nbytes),
                HipMemcpyKind.HOST_TO_DEVICE,
                stream,
            )
        else:
            copy_host_to_device(
                token_ids_device,
                host_array_ptr(token_arr),
                int(token_arr.nbytes),
                runtime=runtime,
            )
        launch_gguf_embedding(
            self.runner.weights.root("token_embedding"),
            int(token_ids_device_ptr),
            int(out_ptr),
            rows=int(rows),
            hidden_size=self.runner.hidden_size,
            vocab_size=self.runner.vocab_size,
            stream=stream,
            runtime=runtime,
        )

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

    def _ensure_prefill_aotriton_bridge(self) -> AotritonPrefillStreamBridge:
        runtime = self.runtime or get_hip_runtime()
        stream = int(getattr(self, "_prefill_aotriton_stream", 0))
        input_ready = int(getattr(self, "_prefill_aotriton_input_ready_event", 0))
        output_ready = int(getattr(self, "_prefill_aotriton_output_ready_event", 0))
        if stream and input_ready and output_ready:
            return AotritonPrefillStreamBridge(stream, input_ready, output_ready)
        if stream or input_ready or output_ready:
            raise RuntimeError("partial GGUF AOTriton prefill stream bridge state")

        new_stream = runtime.stream_create()
        new_input_ready = 0
        new_output_ready = 0
        try:
            new_input_ready = runtime.event_create()
            new_output_ready = runtime.event_create()
        except Exception:
            if new_output_ready:
                runtime.event_destroy(new_output_ready)
            if new_input_ready:
                runtime.event_destroy(new_input_ready)
            if new_stream:
                runtime.stream_destroy(new_stream)
            raise
        self._prefill_aotriton_stream = int(new_stream)
        self._prefill_aotriton_input_ready_event = int(new_input_ready)
        self._prefill_aotriton_output_ready_event = int(new_output_ready)
        return AotritonPrefillStreamBridge(new_stream, new_input_ready, new_output_ready)

    def _release_prefill_aotriton_bridge(self) -> None:
        runtime = self.runtime or get_hip_runtime()
        output_ready = int(getattr(self, "_prefill_aotriton_output_ready_event", 0))
        input_ready = int(getattr(self, "_prefill_aotriton_input_ready_event", 0))
        stream = int(getattr(self, "_prefill_aotriton_stream", 0))
        if output_ready:
            runtime.event_destroy(output_ready)
        if input_ready:
            runtime.event_destroy(input_ready)
        if stream:
            runtime.stream_destroy(stream)
        self._prefill_aotriton_output_ready_event = 0
        self._prefill_aotriton_input_ready_event = 0
        self._prefill_aotriton_stream = 0

    def _prefill_aotriton_bridge_for_rows(
        self,
        query_rows: int,
    ) -> AotritonPrefillStreamBridge | None:
        if self.runner is None or not _gguf_aotriton_isolated_prefill_stream_applies(
            self.runner.backend,
            query_rows,
        ):
            return None
        return self._ensure_prefill_aotriton_bridge()

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

    def _full_attention_prefill_scratch_for_layer(self, bulk_scratch, layer_id: int):
        if self.scratch is None:
            raise RuntimeError("GGUF resident session is closed")
        metadata = None
        if self.kv_storage_dtype == DType.INT8_PER_TOKEN_HEAD:
            metadata = self.scratch.full_scale_metadata(layer_id)
        if self.kv_storage_dtype != DType.INT8_PER_TOKEN_HEAD or metadata is None:
            key_cache, value_cache = self.scratch.full_cache(layer_id)
            return replace(
                bulk_scratch,
                key_cache=key_cache,
                value_cache=value_cache,
                retained_key_cache=None,
                retained_value_cache=None,
                retained_append_spans=None,
                int8_kv_value_bf16=False,
            )
        bf16_mirror_cache = None
        full_bf16_mirror_cache = getattr(self.scratch, "full_bf16_mirror_cache", None)
        if full_bf16_mirror_cache is not None:
            bf16_mirror_cache = full_bf16_mirror_cache(layer_id)
        if bf16_mirror_cache is None:
            oracle_key_cache, oracle_value_cache = self._int8_prefill_oracle_cache_for_layer(layer_id)
        else:
            oracle_key_cache, oracle_value_cache = bf16_mirror_cache
        retained_key_cache, retained_value_cache = self.scratch.full_cache(layer_id)
        retained_append_spans = replace(
            bulk_scratch.append_spans,
            storage_dtype=DType.INT8_PER_TOKEN_HEAD,
            scale_metadata=metadata,
        )
        return replace(
            bulk_scratch,
            key_cache=oracle_key_cache,
            value_cache=oracle_value_cache,
            retained_key_cache=retained_key_cache,
            retained_value_cache=retained_value_cache,
            retained_append_spans=retained_append_spans,
            int8_kv_value_bf16=getattr(self, "int8_kv_value_bf16", False),
        )

    def _packed_full_attention_scratch_for_layer(
        self,
        packed_scratch,
        packed_state: _GGUFPackedTargetState,
        layer_id: int,
    ):
        key_cache, value_cache = packed_state.full_cache(layer_id)
        metadata = packed_state.full_scale_metadata(layer_id)
        if metadata is None:
            return replace(
                packed_scratch,
                key_cache=key_cache,
                value_cache=value_cache,
                retained_key_cache=None,
                retained_value_cache=None,
                retained_append_spans=None,
                int8_kv_value_bf16=False,
            )
        mirror = packed_state.full_bf16_mirror_cache(layer_id)
        if mirror is None:
            raise NotImplementedError(
                "packed AR direct INT8 attention is not admitted without a bounded BF16 mirror"
            )
        retained_append_spans = replace(
            packed_scratch.append_spans,
            storage_dtype=DType.INT8_PER_TOKEN_HEAD,
            scale_metadata=metadata,
        )
        return replace(
            packed_scratch,
            key_cache=mirror[0],
            value_cache=mirror[1],
            retained_key_cache=key_cache,
            retained_value_cache=value_cache,
            retained_append_spans=retained_append_spans,
            int8_kv_value_bf16=bool(packed_state.kv_layout.int8_kv_value_bf16),
        )

    def _int8_prefill_oracle_cache_for_layer(self, layer_id: int) -> tuple[DeviceBuffer, DeviceBuffer]:
        """Return a per-layer BF16 oracle cache for INT8-retained GGUF prefill.

        Chunk-outer bulk prefill revisits each layer once per prompt chunk. A
        single shared oracle cache is therefore unsafe when more than one
        full-attention layer is INT8-retained: later layers overwrite earlier
        layers' previous chunks before those earlier layers process the next
        chunk. Keep the temporary BF16 oracle layer-local for the duration of
        one prefill, then release it before decode.
        """

        if self.scratch is None or self.runner is None or self.runner.weights is None:
            raise RuntimeError("GGUF resident session is closed")
        layer = int(layer_id)
        cached = self._int8_prefill_oracle_buffers.get(layer)
        if cached is not None:
            return cached
        cfg = self.runner.weights.config
        nbytes = int(self.scratch.max_positions) * int(cfg.head_count_kv) * int(cfg.key_length) * DType.BF16.itemsize
        runtime = self.runtime or get_hip_runtime()
        key_cache = malloc(nbytes, runtime=runtime)
        value_cache = malloc(nbytes, runtime=runtime)
        cached = (key_cache, value_cache)
        self._int8_prefill_oracle_buffers[layer] = cached
        return cached

    def _release_int8_prefill_oracle_buffers(self) -> None:
        if not self._int8_prefill_oracle_buffers:
            return
        runtime = self.runtime or get_hip_runtime()
        for key_cache, value_cache in reversed(tuple(self._int8_prefill_oracle_buffers.values())):
            free(value_cache, runtime=runtime)
            free(key_cache, runtime=runtime)
        self._int8_prefill_oracle_buffers.clear()

    def _q8_mmq_prefill_context(self):
        """Return the bounded Q8 MMQ context selected by the generator plugin."""

        if self.runner is None or self._bulk_prefill_scratch is None:
            raise RuntimeError("GGUF resident bulk prefill scratch is closed")
        policy = getattr(self.runner, "_gguf_q8_mmq_prefill_policy", None)
        if policy is not None and getattr(self, "_q8_mmq_prefill_library", None) is None:
            self._q8_mmq_prefill_library = build_gguf_q8_0_mmq_prefill(
                load=True,
                compiler_version=self.compiler_version,
                require_cached=self.require_cached_build,
            )
        if policy is not None and getattr(self, "_q8_mmq_risk_count", None) is None:
            runtime = self.runtime or get_hip_runtime()
            risk_rows = min(int(self.max_sequence_length), int(policy.max_rows))
            risk_count = malloc(DType.INT32.itemsize, runtime=runtime)
            try:
                risk_indices = malloc(
                    policy.risk_indices_nbytes(risk_rows),
                    runtime=runtime,
                )
            except Exception:
                free(risk_count, runtime=runtime)
                raise
            self._q8_mmq_risk_count = risk_count
            self._q8_mmq_risk_indices = risk_indices
            self._buffers = (*self._buffers, risk_count, risk_indices)
        workspace = self._bulk_prefill_scratch.linear_qkv_f32
        risk_count = getattr(self, "_q8_mmq_risk_count", None)
        risk_indices = getattr(self, "_q8_mmq_risk_indices", None)
        return q8_mmq_prefill_session(
            workspace_ptr=workspace.ptr,
            workspace_nbytes=workspace.nbytes,
            risk_count_ptr=0 if risk_count is None else risk_count.ptr,
            risk_count_nbytes=0 if risk_count is None else risk_count.nbytes,
            risk_indices_ptr=0 if risk_indices is None else risk_indices.ptr,
            risk_indices_nbytes=0 if risk_indices is None else risk_indices.nbytes,
            policy=policy,
            library=getattr(self, "_q8_mmq_prefill_library", None),
        )

    def _drain_prefill_queue(
        self,
        boundary: str,
        *,
        runtime: HipRuntime,
        stream: int,
    ) -> None:
        """Optionally bound outstanding prefill work at a host boundary."""

        if boundary not in {"chunk", "layer"}:
            raise ValueError(f"unsupported prefill queue-drain boundary {boundary!r}")
        if getattr(self, "prefill_queue_drain", "none") == boundary:
            runtime.stream_synchronize(stream)

    def prefill(
        self,
        token_ids: list[int] | tuple[int, ...],
        *,
        use_bulk: bool | None = None,
        bulk_attention_mode: str | None = None,
        return_logits: bool = True,
        capture_hidden_seed_fp32: bool = False,
        capture_layer_output_hidden: list[int] | tuple[int, ...] | set[int] | None = None,
    ) -> Qwen35GGUFNextTokenProbeResult:
        """Consume prompt tokens once and return the greedy next token.

        Prompts at least as long as the linear-attention convolution kernel use
        bulk prefill by default. The generator plugin selects the certified
        default bulk scheduler for its quant preset; callers can pass
        ``bulk_attention_mode='bulk'`` or ``'native'`` explicitly for
        diagnostics. Short prompts keep the token-serial
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
        selected_bulk_attention_mode = (
            getattr(self, "default_bulk_attention_mode", "bulk")
            if bulk_attention_mode is None
            else bulk_attention_mode
        )
        run_bulk = len(token_ids) >= min_bulk_tokens if use_bulk is None else bool(use_bulk)
        if run_bulk:
            if len(token_ids) < min_bulk_tokens:
                raise ValueError(
                    f"GGUF bulk prefill requires at least {min_bulk_tokens} tokens; got {len(token_ids)}"
                )
            with (
                q8_t16_two_wave_prefill_session(
                    _gguf_q8_t16_two_wave_prefill_applies(
                        getattr(self.runner, "backend", "hip_gfx1100"),
                        len(token_ids),
                    )
                ),
                wmma_prefill_session(self.use_wmma_prefill),
                gemv_decode_session(self.use_gemv_decode),
                self._q8_mmq_prefill_context(),
            ):
                bulk_kwargs: dict[str, object] = {}
                if capture_hidden_seed_fp32:
                    bulk_kwargs["capture_hidden_seed_fp32"] = True
                if capture_layer_output_hidden is not None:
                    bulk_kwargs["capture_layer_output_hidden"] = capture_layer_output_hidden
                return self._run_bulk_prefill_and_sample(
                    token_ids,
                    bulk_attention_mode=selected_bulk_attention_mode,
                    return_logits=return_logits,
                    **bulk_kwargs,
                )

        self.reset()
        hidden_ptr = None
        final_index = len(token_ids) - 1
        for index, token_id in enumerate(token_ids):
            token_kwargs: dict[str, object] = {}
            if index == final_index and capture_layer_output_hidden is not None:
                token_kwargs["capture_layer_output_hidden"] = capture_layer_output_hidden
            hidden_ptr = self._run_token_to_final_hidden(
                int(token_id),
                position=self._position,
                capture_hidden_seed_fp32=bool(capture_hidden_seed_fp32) and index == final_index,
                **token_kwargs,
            )
            self._position += 1
        assert hidden_ptr is not None
        return self._sample_from_hidden(hidden_ptr, return_logits=return_logits)

    def prefill_async_top1(
        self,
        token_ids: list[int] | tuple[int, ...],
        *,
        use_bulk: bool | None = None,
        bulk_attention_mode: str = "bulk",
        capture_hidden_seed_fp32: bool = False,
        stream: int,
    ) -> None:
        """Launch prompt prefill and top-1 sampling on ``stream`` without host readback."""

        if int(stream) == 0:
            raise ValueError("prefill_async_top1 requires a non-default stream")
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
            with (
                q8_t16_two_wave_prefill_session(
                    _gguf_q8_t16_two_wave_prefill_applies(
                        getattr(self.runner, "backend", "hip_gfx1100"),
                        len(token_ids),
                    )
                ),
                wmma_prefill_session(self.use_wmma_prefill),
                gemv_decode_session(self.use_gemv_decode),
                self._q8_mmq_prefill_context(),
            ):
                self._run_bulk_prefill_and_sample(
                    token_ids,
                    stream=int(stream),
                    bulk_attention_mode=selected_bulk_attention_mode,
                    return_logits=False,
                    capture_hidden_seed_fp32=bool(capture_hidden_seed_fp32),
                    enqueue_sample_only=True,
                )
            return

        self.reset(stream=int(stream))
        hidden_ptr = None
        final_index = len(token_ids) - 1
        for index, token_id in enumerate(token_ids):
            hidden_ptr = self._run_token_to_final_hidden(
                int(token_id),
                position=self._position,
                stream=int(stream),
                capture_hidden_seed_fp32=bool(capture_hidden_seed_fp32) and index == final_index,
            )
            self._position += 1
        assert hidden_ptr is not None
        self._sample_device_from_hidden(hidden_ptr, stream=int(stream))

    def _run_bulk_prefill_and_sample(
        self,
        token_ids: list[int] | tuple[int, ...],
        *,
        stream: int = 0,
        bulk_attention_mode: str = "bulk",
        return_logits: bool = True,
        capture_hidden_seed_fp32: bool = False,
        capture_layer_output_hidden: list[int] | tuple[int, ...] | set[int] | None = None,
        enqueue_sample_only: bool = False,
    ) -> Qwen35GGUFNextTokenProbeResult | None:
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
        capture_layer_ids = self._normalize_layer_output_capture(
            capture_layer_output_hidden
        )
        if capture_layer_ids:
            self._last_layer_output_hidden = {}
        tokens = np.asarray([int(token) for token in token_ids], dtype=np.int64)
        for token in tokens.tolist():
            if token < 0 or token >= self.runner.vocab_size:
                raise ValueError(f"token_id {token} outside [0, {self.runner.vocab_size})")
        hidden_seed_buf = None
        if capture_hidden_seed_fp32:
            self._ensure_verify_block_buffers(rows, runtime=runtime)
            if self._verify_hidden_seed_buf is None:
                raise RuntimeError("GGUF verifier hidden-seed buffer is closed")
            hidden_seed_buf = self._verify_hidden_seed_buf
        recorder = self._prefill_flight_recorder
        prefill_id = 0
        reset_sequence = 0
        if recorder is not None:
            prefill_id = recorder.begin_prefill(total_rows=rows, stream=stream)
            reset_sequence = recorder.submit(
                phase=FlightRecorderPhase.RESET,
                prefill_id=prefill_id,
                chunk_start=0,
                chunk_end=rows,
                layer_id=-1,
                layer_type=0,
                stream=stream,
            )
        self.reset(stream=stream)
        if recorder is not None:
            recorder.complete(reset_sequence, stream=stream)
        alloc_capacity = self._prefill_hidden_a.nbytes // (self.runner.hidden_size * 2)
        chunk_outer = alloc_capacity < rows
        if hidden_seed_buf is not None and chunk_outer:
            raise ValueError("capture_hidden_seed_fp32 is not supported with chunked outer GGUF prefill")

        linear_min_rows = int(self.runner.weights.config.ssm_conv_kernel)
        use_wmma_prefill = gguf_wmma_prefill_enabled(None)

        try:
            if chunk_outer:
                chunk_size = self._prefill_scratch_rows(rows)
                ranges = _chunk_ranges(rows, chunk_size, min_chunk_size=linear_min_rows)
                last_src_ptr = 0
                last_bulk_scratch = None
                for chunk_start, chunk_end in ranges:
                    chunk_rows = chunk_end - chunk_start
                    chunk_tokens = tokens[chunk_start:chunk_end]
                    chunk_sequence = 0
                    embedding_sequence = 0
                    if recorder is not None:
                        chunk_sequence = recorder.submit(
                            phase=FlightRecorderPhase.CHUNK,
                            prefill_id=prefill_id,
                            chunk_start=chunk_start,
                            chunk_end=chunk_end,
                            layer_id=-1,
                            layer_type=0,
                            stream=stream,
                        )
                        embedding_sequence = recorder.submit(
                            phase=FlightRecorderPhase.EMBEDDING,
                            prefill_id=prefill_id,
                            chunk_start=chunk_start,
                            chunk_end=chunk_end,
                            layer_id=-1,
                            layer_type=0,
                            stream=stream,
                        )
                    self._copy_token_embeddings_to_device(
                        chunk_tokens,
                        self._prefill_hidden_a.ptr,
                        rows=chunk_rows,
                        token_ids_device_ptr=self._prefill_token_buf.ptr,
                        stream=stream,
                    )
                    if recorder is not None and recorder.should_complete_layers:
                        recorder.complete(embedding_sequence, stream=stream)
                    src = self._prefill_hidden_a
                    dst = self._prefill_hidden_b
                    # Chunk metadata is request/chunk scoped, not layer scoped.
                    # Re-uploading the same six host arrays before every layer
                    # serialized the default stream and created hundreds of
                    # copy dispatches in pp512.
                    bulk_scratch = self._bulk_prefill_scratch.for_chunk(
                        chunk_start,
                        chunk_rows,
                        total_tokens=rows,
                        runtime=runtime,
                        stream=stream,
                    )
                    for layer_id, layer_type in enumerate(self.runner.weights.config.layer_types):
                        layer_sequence = 0
                        if recorder is not None:
                            layer_sequence = recorder.submit(
                                phase=(
                                    FlightRecorderPhase.LINEAR_ATTENTION_LAYER
                                    if layer_type == LINEAR_ATTENTION
                                    else FlightRecorderPhase.FULL_ATTENTION_LAYER
                                ),
                                prefill_id=prefill_id,
                                chunk_start=chunk_start,
                                chunk_end=chunk_end,
                                layer_id=layer_id,
                                layer_type=(1 if layer_type == LINEAR_ATTENTION else 2),
                                stream=stream,
                            )
                        if bulk_attention_mode == "native":
                            self.runner._run_native_attention_bulk_ffn_layer_rows(
                                layer_id,
                                layer_type,
                                src.ptr,
                                dst.ptr,
                                bulk_scratch,
                                rows=chunk_rows,
                                start_position=chunk_start,
                                stream=stream,
                                decode_scratch=self.scratch,
                            )
                        elif layer_type == LINEAR_ATTENTION:
                            self.runner._run_linear_attention_prefill_layer_rows(
                                layer_id,
                                src.ptr,
                                dst.ptr,
                                bulk_scratch,
                                rows=chunk_rows,
                                stream=stream,
                                decode_scratch=self.scratch,
                                expert_sidecar=None,
                            )
                        elif layer_type == FULL_ATTENTION:
                            layer_scratch = self._full_attention_prefill_scratch_for_layer(bulk_scratch, layer_id)
                            self.runner._run_full_attention_prefill_layer_aotriton(
                                layer_id,
                                src.ptr,
                                dst.ptr,
                                layer_scratch,
                                cos_table_ptr=self.scratch.cos_table_buf.ptr,
                                sin_table_ptr=self.scratch.sin_table_buf.ptr,
                                max_positions=int(self.scratch.max_positions),
                                stream=stream,
                                aotriton_bridge=self._prefill_aotriton_bridge_for_rows(chunk_rows),
                                expert_sidecar=None,
                            )
                        else:
                            raise ValueError(f"unsupported GGUF layer type {layer_type!r}")
                        if recorder is not None and recorder.should_complete_layers:
                            recorder.complete(layer_sequence, stream=stream)
                        self._drain_prefill_queue(
                            "layer",
                            runtime=runtime,
                            stream=stream,
                        )
                        src, dst = dst, src
                        if layer_id in capture_layer_ids and chunk_end == rows:
                            final_row_ptr = (
                                src.ptr
                                + (chunk_rows - 1)
                                * self.runner.hidden_size
                                * DType.BF16.itemsize
                            )
                            self._last_layer_output_hidden[int(layer_id)] = (
                                _copy_bf16_rows_to_host_f32(
                                    final_row_ptr,
                                    1,
                                    self.runner.hidden_size,
                                    runtime=runtime,
                                )
                            )
                    if recorder is not None and not recorder.should_complete_layers:
                        recorder.complete(chunk_sequence, stream=stream)
                    self._drain_prefill_queue(
                        "chunk",
                        runtime=runtime,
                        stream=stream,
                    )
                    last_bulk_scratch = bulk_scratch
                    last_src_ptr = src.ptr + (chunk_rows - 1) * self.runner.hidden_size * DType.BF16.itemsize
                if last_bulk_scratch is None:
                    raise RuntimeError("GGUF chunked prefill did not process any chunks")
            else:
                chunk_sequence = 0
                embedding_sequence = 0
                if recorder is not None:
                    chunk_sequence = recorder.submit(
                        phase=FlightRecorderPhase.CHUNK,
                        prefill_id=prefill_id,
                        chunk_start=0,
                        chunk_end=rows,
                        layer_id=-1,
                        layer_type=0,
                        stream=stream,
                    )
                    embedding_sequence = recorder.submit(
                        phase=FlightRecorderPhase.EMBEDDING,
                        prefill_id=prefill_id,
                        chunk_start=0,
                        chunk_end=rows,
                        layer_id=-1,
                        layer_type=0,
                        stream=stream,
                    )
                self._copy_token_embeddings_to_device(
                    tokens,
                    self._prefill_hidden_a.ptr,
                    rows=rows,
                    token_ids_device_ptr=self._prefill_token_buf.ptr,
                    stream=stream,
                )
                if recorder is not None and recorder.should_complete_layers:
                    recorder.complete(embedding_sequence, stream=stream)
                src = self._prefill_hidden_a
                dst = self._prefill_hidden_b
                active_chunk_key: tuple[int, int, int] | None = None
                active_bulk_scratch = None
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
                            layer_chunk_size = self._linear_prefill_layer_chunk_size(rows)
                            layer_ranges = _chunk_ranges(rows, layer_chunk_size, min_chunk_size=linear_min_rows)
                        elif layer_type == FULL_ATTENTION:
                            layer_chunk_size = self._full_attention_prefill_layer_chunk_size(rows)
                            layer_ranges = _chunk_ranges(rows, layer_chunk_size, min_chunk_size=2)
                        else:
                            raise ValueError(f"unsupported GGUF layer type {layer_type!r}")
                        for start, end in layer_ranges:
                            chunk_rows = end - start
                            layer_sequence = 0
                            if recorder is not None:
                                layer_sequence = recorder.submit(
                                    phase=(
                                        FlightRecorderPhase.LINEAR_ATTENTION_LAYER
                                        if layer_type == LINEAR_ATTENTION
                                        else FlightRecorderPhase.FULL_ATTENTION_LAYER
                                    ),
                                    prefill_id=prefill_id,
                                    chunk_start=start,
                                    chunk_end=end,
                                    layer_id=layer_id,
                                    layer_type=(1 if layer_type == LINEAR_ATTENTION else 2),
                                    stream=stream,
                                )
                            src_chunk_ptr = src.ptr + start * self.runner.hidden_size * DType.BF16.itemsize
                            dst_chunk_ptr = dst.ptr + start * self.runner.hidden_size * DType.BF16.itemsize
                            chunk_key = (int(start), int(chunk_rows), int(rows))
                            if chunk_key != active_chunk_key:
                                active_bulk_scratch = self._bulk_prefill_scratch.for_chunk(
                                    start,
                                    chunk_rows,
                                    total_tokens=rows,
                                    runtime=runtime,
                                    stream=stream,
                                )
                                active_chunk_key = chunk_key
                            if active_bulk_scratch is None:
                                raise RuntimeError("GGUF bulk prefill chunk metadata was not prepared")
                            bulk_scratch = active_bulk_scratch
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
                                layer_scratch = self._full_attention_prefill_scratch_for_layer(bulk_scratch, layer_id)
                                self.runner._run_full_attention_prefill_layer_aotriton(
                                    layer_id,
                                    src_chunk_ptr,
                                    dst_chunk_ptr,
                                    layer_scratch,
                                    cos_table_ptr=self.scratch.cos_table_buf.ptr,
                                    sin_table_ptr=self.scratch.sin_table_buf.ptr,
                                    max_positions=int(self.scratch.max_positions),
                                    stream=stream,
                                    aotriton_bridge=self._prefill_aotriton_bridge_for_rows(chunk_rows),
                                    expert_sidecar=expert_sidecar,
                                )
                            else:
                                raise ValueError(f"unsupported GGUF layer type {layer_type!r}")
                            if recorder is not None and recorder.should_complete_layers:
                                recorder.complete(layer_sequence, stream=stream)
                        self._drain_prefill_queue(
                            "layer",
                            runtime=runtime,
                            stream=stream,
                        )
                    finally:
                        if expert_sidecar is not None:
                            expert_sidecar.free(runtime=runtime)
                    src, dst = dst, src
                    if layer_id in capture_layer_ids:
                        final_row_ptr = (
                            src.ptr
                            + (rows - 1)
                            * self.runner.hidden_size
                            * DType.BF16.itemsize
                        )
                        self._last_layer_output_hidden[int(layer_id)] = (
                            _copy_bf16_rows_to_host_f32(
                                final_row_ptr,
                                1,
                                self.runner.hidden_size,
                                runtime=runtime,
                            )
                        )
                if recorder is not None and not recorder.should_complete_layers:
                    recorder.complete(chunk_sequence, stream=stream)
                self._drain_prefill_queue(
                    "chunk",
                    runtime=runtime,
                    stream=stream,
                )
                last_bulk_scratch = active_bulk_scratch
                if last_bulk_scratch is None:
                    # Empty synthetic layer stacks still need the shared norm
                    # workspace used by the output head.
                    last_bulk_scratch = self._bulk_prefill_scratch.for_chunk(
                        rows - 1,
                        1,
                        total_tokens=rows,
                        runtime=runtime,
                        stream=stream,
                    )
                last_src_ptr = src.ptr + (rows - 1) * self.runner.hidden_size * DType.BF16.itemsize

            finalize_sequence = 0
            if recorder is not None:
                finalize_sequence = recorder.submit(
                    phase=FlightRecorderPhase.PREFILL_FINALIZE,
                    prefill_id=prefill_id,
                    chunk_start=rows - 1,
                    chunk_end=rows,
                    layer_id=-1,
                    layer_type=0,
                    stream=stream,
                )
            if hidden_seed_buf is not None:
                final_scratch = self._bulk_prefill_scratch.for_chunk(
                    0,
                    rows,
                    total_tokens=rows,
                    runtime=runtime,
                    stream=stream,
                )
                output_norm_weight_ptr = self.runner.weights.root("output_norm").allocation().tensor.ptr
                gguf_rmsnorm_bf16_f32_weight(
                    src.ptr,
                    output_norm_weight_ptr,
                    final_scratch.norm.ptr,
                    rows=rows,
                    hidden_size=self.runner.hidden_size,
                    eps=self.runner.weights.config.rms_norm_eps,
                    stream=stream,
                    runtime=runtime,
                )
                gguf_rmsnorm_bf16_f32_weight_out_f32(
                    src.ptr,
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
                self._verify_hidden_seed_rows_populated = rows
                self._hidden_seed_fp32_populated = True
                last_hidden_ptr = final_scratch.norm.ptr + (rows - 1) * self.runner.hidden_size * DType.BF16.itemsize
            else:
                last_hidden_ptr = self._run_output_norm_hidden(
                    last_src_ptr,
                    last_bulk_scratch.norm.ptr,
                    stream=stream,
                    capture_hidden_seed_fp32=False,
                )
            self._last_target_hidden_ptr = int(last_src_ptr)
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
            if recorder is not None:
                recorder.complete(finalize_sequence, stream=stream)
                sample_sequence = recorder.submit(
                    phase=FlightRecorderPhase.SAMPLE,
                    prefill_id=prefill_id,
                    chunk_start=rows - 1,
                    chunk_end=rows,
                    layer_id=-1,
                    layer_type=0,
                    stream=stream,
                )
            else:
                sample_sequence = 0
            if enqueue_sample_only:
                self._sample_device_from_hidden(last_hidden_ptr, stream=stream)
                if recorder is not None:
                    recorder.complete(sample_sequence, stream=stream)
                return None
            result = self._sample_from_hidden(last_hidden_ptr, return_logits=return_logits, stream=stream)
            if recorder is not None:
                recorder.complete(sample_sequence, stream=stream)
            return result
        finally:
            self._release_int8_prefill_oracle_buffers()

    def _verify_token_embedding_reader_cached(self) -> GGUFReader:
        reader = self._verify_token_embedding_reader
        if reader is None:
            reader = GGUFReader(self.model_path)
            self._verify_token_embedding_reader = reader
        return reader

    def _seed_verify_hidden_f32_from_token_embedding(
        self,
        token_ids: np.ndarray,
        *,
        runtime: HipRuntime,
    ) -> None:
        if self.runner is None:
            raise RuntimeError("GGUF resident session is closed")
        if self._verify_hidden_f32_a is None:
            raise RuntimeError("GGUF verifier F32 residual buffer is closed")
        rows_f32 = _gguf_token_embedding_rows_f32(
            self._verify_token_embedding_reader_cached(),
            token_ids,
            hidden_size=self.runner.hidden_size,
            vocab_size=self.runner.vocab_size,
        )
        copy_host_to_device(
            self._verify_hidden_f32_a,
            host_array_ptr(rows_f32),
            rows_f32.nbytes,
            runtime=runtime,
        )

    def verify_target_blocks_batch(
        self,
        jobs: list[dict[str, object]] | tuple[dict[str, object], ...],
        *,
        stream: int = 0,
    ) -> list[Qwen35GGUFBlockVerifyResult]:
        """Run one packed target-verifier pass for multiple resident sessions.

        This is the llama.cpp-style serving verifier path: rows from independent
        slots are packed slot-major, full-attention K/V writes go through one
        contiguous packed target cache, and Conv/GDN row-state captures are
        split back into each slot session for the later accept-row commit.

        The first production slice is intentionally bounded to the server MTP
        shape: shared runner, BF16 KV, bulk verifier, no-copy prefill-GDN state
        capture, and context < 1024 where the existing c1-equivalent
        full-attention batch decoder is valid. Unsupported shapes raise
        ``NotImplementedError`` so the scheduler can fall back to per-slot
        verification.
        """

        stage_timings: dict[str, float] = {}
        self.last_packed_verify_stage_timings_ms = stage_timings
        total_start = time.perf_counter()

        def add_stage(name: str, start: float) -> float:
            now = time.perf_counter()
            stage_timings[name] = stage_timings.get(name, 0.0) + (now - start) * 1000.0
            return now

        setup_start = time.perf_counter()
        job_list = list(jobs)
        if len(job_list) <= 1:
            raise NotImplementedError("packed target verifier requires at least two jobs")
        if self.runner is None or self.runner.weights is None or self.scratch is None:
            raise RuntimeError("GGUF resident session is closed")
        if self._prefill_token_buf is None or self._prefill_hidden_a is None or self._prefill_hidden_b is None:
            raise RuntimeError("GGUF resident packed verifier buffers are closed")
        if self._bulk_prefill_scratch is None:
            raise RuntimeError("GGUF resident bulk prefill scratch is closed")
        if self.kv_storage_dtype != DType.BF16:
            raise NotImplementedError("packed target verifier currently supports BF16 KV only")
        if self.use_expert_sidecar:
            raise NotImplementedError("packed target verifier does not support expert sidecars yet")
        if _gguf_verify_f32_residual_enabled():
            raise NotImplementedError("packed target verifier does not support F32 residual diagnostics")
        if not _gguf_verify_capture_prefill_gdn_enabled():
            raise NotImplementedError("packed target verifier requires prefill-GDN no-copy capture")

        capture_linear_state_rows = bool(job_list[0].get("capture_linear_state_rows", False))
        defer_linear_state_commit = bool(job_list[0].get("defer_linear_state_commit", False))
        defer_state_scatter = bool(job_list[0].get("defer_state_scatter", False))
        if capture_linear_state_rows and not defer_linear_state_commit:
            raise NotImplementedError("packed target verifier captured-row mode requires deferred linear-state commit")
        if not capture_linear_state_rows and defer_linear_state_commit:
            raise NotImplementedError("packed target verifier cannot defer uncaptured linear-state commit")
        if defer_state_scatter and (not capture_linear_state_rows or not defer_linear_state_commit):
            raise NotImplementedError("packed target verifier can defer scatter only for captured/deferred rows")

        slot_blocks: list[_GGUFPackedVerifySlotBlock] = []
        for job in job_list:
            session = job.get("session")
            if not isinstance(session, Qwen35GGUFResidentSession):
                raise NotImplementedError("packed target verifier requires resident GGUF sessions")
            if session.runner is not self.runner:
                raise NotImplementedError("packed target verifier requires shared runner sessions")
            if session.scratch is None:
                raise RuntimeError("packed verifier job session is closed")
            if session.kv_storage_dtype != DType.BF16:
                raise NotImplementedError("packed target verifier currently supports BF16 KV only")
            if str(job.get("bulk_attention_mode", "bulk")) != "bulk":
                raise NotImplementedError("packed target verifier supports bulk attention mode only")
            if bool(job.get("capture_linear_state_rows", False)) != capture_linear_state_rows:
                raise NotImplementedError("packed target verifier requires uniform linear-state capture mode")
            if bool(job.get("defer_linear_state_commit", False)) != defer_linear_state_commit:
                raise NotImplementedError("packed target verifier requires uniform linear-state commit mode")
            if bool(job.get("defer_state_scatter", False)) != defer_state_scatter:
                raise NotImplementedError("packed target verifier requires uniform state-scatter mode")
            input_token_ids = tuple(int(token) for token in job.get("input_token_ids", ()))
            if not input_token_ids:
                raise ValueError("packed target verifier jobs require non-empty input_token_ids")
            for token in input_token_ids:
                if token < 0 or token >= int(self.runner.vocab_size):
                    raise ValueError(f"token_id {token} outside [0, {self.runner.vocab_size})")
            slot_blocks.append(
                _GGUFPackedVerifySlotBlock(
                    input_token_ids=input_token_ids,
                    start_position=int(session.position),
                )
            )

        max_live_count = max(
            int(block.start_position) + len(block.input_token_ids)
            for block in slot_blocks
        )
        slot_capacity = max(1024, max_live_count)
        layout = _build_gguf_packed_verify_layout(slot_blocks, slot_capacity=slot_capacity)
        if int(layout.max_live_count) >= 1024:
            raise NotImplementedError("packed target verifier currently requires context < 1024")
        rows = int(layout.rows)
        if rows > int(self._bulk_prefill_scratch.rows):
            raise NotImplementedError(
                f"packed target rows {rows} exceed resident hidden-buffer capacity {self._bulk_prefill_scratch.rows}"
            )
        runtime = self.runtime or get_hip_runtime()
        packed_state, packed_scratch_base = self._ensure_packed_verify_workspace(
            slot_count=int(layout.slot_count),
            rows=rows,
            max_sequence_length=slot_capacity,
            runtime=runtime,
        )
        self._ensure_verify_block_buffers(rows, runtime=runtime)
        if capture_linear_state_rows:
            self._ensure_verify_linear_state_row_buffers(rows, runtime=runtime)
        add_stage("packed_verify_setup", setup_start)
        sync_state_start = time.perf_counter()
        self._sync_packed_verify_initial_state(
            job_list,
            layout,
            packed_state,
            runtime=runtime,
            stream=stream,
        )
        add_stage("packed_verify_sync_initial_state", sync_state_start)
        token_upload_start = time.perf_counter()
        packed_scratch = packed_scratch_base.for_packed_verify_layout(layout, runtime=runtime, stream=stream)
        token_ids = np.ascontiguousarray(layout.input_token_ids, dtype=np.int64)
        copy_host_to_device(self._prefill_token_buf, host_array_ptr(token_ids), token_ids.nbytes, runtime=runtime)
        add_stage("packed_verify_token_upload", token_upload_start)
        gpu_stage_recorder = (
            _HipEventStageRecorder(runtime, enabled=True, stream=stream)
            if _gguf_packed_verify_gpu_stage_timings_enabled()
            else None
        )
        if gpu_stage_recorder is not None:
            gpu_stage_recorder.start()

        embedding_start = time.perf_counter()
        hidden_seed_buf = self._verify_hidden_seed_buf
        if hidden_seed_buf is None:
            raise RuntimeError("GGUF packed verifier hidden-seed buffer is closed")
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
        add_stage("packed_verify_embedding", embedding_start)
        if gpu_stage_recorder is not None:
            gpu_stage_recorder.mark("packed_verify_gpu_embedding")
        src = self._prefill_hidden_a
        dst = self._prefill_hidden_b
        linear_decode_scratch = replace(
            self.scratch,
            layer_conv_states=packed_state.layer_conv_states,
            layer_recurrent_states=packed_state.layer_recurrent_states,
        )
        block_wmma_prefill = bool(job_list[0].get("use_wmma_prefill", True))
        with wmma_prefill_session(block_wmma_prefill), gemv_decode_session(self.use_gemv_decode):
            for layer_id, layer_type in enumerate(self.runner.weights.config.layer_types):
                layer_start = time.perf_counter()
                if layer_type == LINEAR_ATTENTION:
                    self.runner._run_linear_attention_prefill_layer_rows(
                        layer_id,
                        src.ptr,
                        dst.ptr,
                        packed_scratch,
                        rows=rows,
                        stream=stream,
                        decode_scratch=linear_decode_scratch,
                        expert_sidecar=None,
                        linear_state_rows=(
                            self._verify_linear_state_row_pair(layer_id)
                            if capture_linear_state_rows
                            else None
                        ),
                        commit_final_linear_state=not defer_linear_state_commit,
                        hidden_f32_ptr=None,
                        out_f32_ptr=None,
                        stage_timings=None,
                        sync_stage_timings=False,
                        stage_prefix="packed_verify_gpu_linear_attn",
                        gpu_stage_recorder=gpu_stage_recorder,
                    )
                    add_stage("packed_verify_linear_attn_layers", layer_start)
                elif layer_type == FULL_ATTENTION:
                    key_cache, value_cache = packed_state.full_cache(layer_id)
                    layer_scratch = replace(
                        packed_scratch,
                        key_cache=key_cache,
                        value_cache=value_cache,
                        cos_table=self.scratch.cos_table,
                        sin_table=self.scratch.sin_table,
                    )
                    self.runner._run_full_attention_decode_batch_layer_rows(
                        layer_id,
                        src.ptr,
                        dst.ptr,
                        layer_scratch,
                        stream=stream,
                        expert_sidecar=None,
                        stage_timings=None,
                        sync_stage_timings=False,
                        stage_prefix="target_block_packed_full_attn",
                    )
                    if gpu_stage_recorder is not None:
                        gpu_stage_recorder.mark("packed_verify_gpu_full_attn_layers")
                    add_stage("packed_verify_full_attn_layers", layer_start)
                else:
                    raise ValueError(f"unsupported GGUF layer type {layer_type!r}")
                src, dst = dst, src

            output_norm_start = time.perf_counter()
            output_norm_weight_ptr = self.runner.weights.root("output_norm").allocation().tensor.ptr
            gguf_rmsnorm_bf16_f32_weight(
                src.ptr,
                output_norm_weight_ptr,
                packed_scratch.norm.ptr,
                rows=rows,
                hidden_size=self.runner.hidden_size,
                eps=self.runner.weights.config.rms_norm_eps,
                stream=stream,
                runtime=runtime,
            )
            gguf_rmsnorm_bf16_f32_weight_out_f32(
                src.ptr,
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
            add_stage("packed_verify_output_norm_hidden", output_norm_start)
            if gpu_stage_recorder is not None:
                gpu_stage_recorder.mark("packed_verify_gpu_output_norm_hidden")
            sample_start = time.perf_counter()
            token_host = self._sample_target_block_rows_from_hidden(
                packed_scratch.norm.ptr,
                rows,
                activation_dtype=GGUF_ACTIVATION_BF16,
                stream=stream,
            )
            add_stage("packed_verify_lm_head_sample", sample_start)
            if gpu_stage_recorder is not None:
                gpu_stage_recorder.mark("packed_verify_gpu_lm_head_sample")
        hidden_readback_start = time.perf_counter()
        hidden_host = np.empty((rows, self.runner.hidden_size), dtype=np.float32)
        copy_device_to_host(host_array_ptr(hidden_host), hidden_seed_buf, hidden_host.nbytes, runtime=runtime)
        add_stage("packed_verify_hidden_readback", hidden_readback_start)
        scatter_start = time.perf_counter()
        results = self._scatter_packed_verify_outputs(
            job_list,
            layout,
            packed_state,
            hidden_host,
            token_host,
            runtime=runtime,
            stream=stream,
            linear_state_rows_captured=capture_linear_state_rows,
            final_linear_state_committed=not defer_linear_state_commit,
            defer_state_scatter=defer_state_scatter,
        )
        add_stage("packed_verify_scatter_outputs", scatter_start)
        sync_start = time.perf_counter()
        if stream:
            runtime.stream_synchronize(stream)
        else:
            runtime.device_synchronize()
        add_stage("packed_verify_final_sync", sync_start)
        if gpu_stage_recorder is not None:
            gpu_stage_recorder.resolve_into(stage_timings)
        stage_timings["packed_verify_total"] = (time.perf_counter() - total_start) * 1000.0
        return results

    def verify_target_block(
        self,
        input_token_ids: list[int] | tuple[int, ...],
        *,
        bulk_attention_mode: str = "bulk",
        use_wmma_prefill: bool | None = None,
        stream: int = 0,
        advance_state_only: bool = False,
        capture_linear_state_rows: bool = False,
        capture_pre_output_norm_hidden: bool = False,
        capture_layer_output_hidden: list[int] | tuple[int, ...] | set[int] | None = None,
        capture_layer_boundary_hidden: list[int] | tuple[int, ...] | set[int] | None = None,
        capture_lm_head_logits: bool = False,
        record_stage_timings: bool = False,
        sync_stage_timings: bool = False,
        defer_linear_state_commit: bool = False,
        _pre_staged_token_ids_ptr: int | None = None,
        _target_top1_i64_ptr: int | None = None,
        _target_top1_i32_ptr: int | None = None,
        _enqueue_only: bool = False,
        _prebuilt_bulk_scratch: object | None = None,
        _dynamic_cursor_advance: bool = False,
        _graph_hidden_seed_buf: object | None = None,
        _graph_hidden_f32_a: object | None = None,
        _graph_hidden_f32_b: object | None = None,
    ) -> Qwen35GGUFBlockVerifyResult | None:
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
        if _enqueue_only:
            if stream == 0:
                raise ValueError("enqueue-only target verification requires a non-default capture stream")
            if _pre_staged_token_ids_ptr is None or int(_pre_staged_token_ids_ptr) <= 0:
                raise ValueError("enqueue-only target verification requires pre-staged token ids")
            has_i64_top1 = _target_top1_i64_ptr is not None and int(_target_top1_i64_ptr) > 0
            has_i32_top1 = _target_top1_i32_ptr is not None and int(_target_top1_i32_ptr) > 0
            if has_i64_top1 == has_i32_top1:
                raise ValueError(
                    "enqueue-only target verification requires exactly one int32/int64 target-top1 destination"
                )
            if rows not in {2, 3}:
                raise ValueError("enqueue-only native target verification supports B1/B2 rows=2-3")
            if advance_state_only or capture_pre_output_norm_hidden or capture_lm_head_logits:
                raise ValueError("enqueue-only target verification does not support host diagnostic outputs")
            if capture_layer_output_hidden or capture_layer_boundary_hidden:
                raise ValueError("enqueue-only target verification does not support layer host captures")
            if record_stage_timings or sync_stage_timings:
                raise ValueError("enqueue-only target verification does not support host stage timings")
            graph_hidden = (
                _graph_hidden_seed_buf,
                _graph_hidden_f32_a,
                _graph_hidden_f32_b,
            )
            if _prebuilt_bulk_scratch is not None:
                if int(getattr(_prebuilt_bulk_scratch, "start", -1)) != 0:
                    raise ValueError("reusable target graph scratch must use fixed row offset zero")
                if int(getattr(_prebuilt_bulk_scratch, "rows", 0)) != rows:
                    raise ValueError("reusable target graph scratch rows must match verifier rows")
                if not _dynamic_cursor_advance:
                    raise ValueError("reusable target graph scratch requires device-driven cursor advance")
                if any(buffer is None for buffer in graph_hidden):
                    raise ValueError("reusable target graph requires graph-owned hidden buffers")
            elif _dynamic_cursor_advance or any(buffer is not None for buffer in graph_hidden):
                raise ValueError("device-driven graph state requires reusable target graph scratch")
        elif (
            _pre_staged_token_ids_ptr is not None
            or _target_top1_i64_ptr is not None
            or _target_top1_i32_ptr is not None
            or _prebuilt_bulk_scratch is not None
            or _dynamic_cursor_advance
            or _graph_hidden_seed_buf is not None
            or _graph_hidden_f32_a is not None
            or _graph_hidden_f32_b is not None
        ):
            raise ValueError("private target graph controls require enqueue-only mode")
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
        pre_output_norm_hidden_host = None
        lm_head_logits_host = None
        capture_layer_ids = self._normalize_layer_output_capture(capture_layer_output_hidden)
        capture_layer_boundary_ids = self._normalize_layer_output_capture(capture_layer_boundary_hidden)
        layer_output_hidden_host: dict[int, np.ndarray] = {}
        layer_boundary_hidden_host: dict[int, dict[str, np.ndarray]] = {}
        t_setup0 = time.perf_counter() if stage_timings is not None else 0.0
        self._ensure_verify_block_buffers(rows, runtime=runtime)
        if capture_linear_state_rows:
            self._ensure_verify_linear_state_row_buffers(rows, runtime=runtime)
        hidden_seed_buf = (
            self._verify_hidden_seed_buf
            if _graph_hidden_seed_buf is None
            else _graph_hidden_seed_buf
        )
        if hidden_seed_buf is None:
            raise RuntimeError("GGUF verifier hidden-seed buffer is closed")
        verify_hidden_f32_a = (
            self._verify_hidden_f32_a
            if _graph_hidden_f32_a is None
            else _graph_hidden_f32_a
        )
        verify_hidden_f32_b = (
            self._verify_hidden_f32_b
            if _graph_hidden_f32_b is None
            else _graph_hidden_f32_b
        )
        use_f32_residual = _gguf_verify_f32_residual_enabled()
        if use_f32_residual and (verify_hidden_f32_a is None or verify_hidden_f32_b is None):
            raise RuntimeError("GGUF verifier F32 residual buffers are closed")
        token_ids_buf = self._verify_token_ids_i64
        token_counter_buf = self._verify_token_counter_i64
        if token_ids_buf is None or token_counter_buf is None:
            raise RuntimeError("GGUF verifier token buffers are closed")
        if not _enqueue_only:
            zero_index = np.zeros((1,), dtype=np.int64)
            copy_host_to_device(token_counter_buf, host_array_ptr(zero_index), zero_index.nbytes, runtime=runtime)
            copy_host_to_device(self._prefill_token_buf, host_array_ptr(tokens), tokens.nbytes, runtime=runtime)
        input_token_ids_ptr = (
            self._prefill_token_buf.ptr
            if _pre_staged_token_ids_ptr is None
            else int(_pre_staged_token_ids_ptr)
        )
        add_verify_stage("target_block_setup", (time.perf_counter() - t_setup0) * 1000 if stage_timings is not None else 0.0)
        scratch_row_start = 0 if _prebuilt_bulk_scratch is not None else start
        try:
            t_embedding0 = time.perf_counter() if stage_timings is not None else 0.0
            launch_gguf_embedding(
                self.runner.weights.root("token_embedding"),
                input_token_ids_ptr,
                self._prefill_hidden_a.ptr + scratch_row_start * row_nbytes,
                rows=rows,
                hidden_size=self.runner.hidden_size,
                vocab_size=self.runner.vocab_size,
                stream=stream,
                runtime=runtime,
            )
            if use_f32_residual:
                if _gguf_verify_f32_token_embedding_enabled():
                    t_f32_embedding_seed0 = time.perf_counter() if stage_timings is not None else 0.0
                    self._seed_verify_hidden_f32_from_token_embedding(tokens, runtime=runtime)
                    add_verify_stage(
                        "target_block_f32_token_embedding_seed",
                        (time.perf_counter() - t_f32_embedding_seed0) * 1000
                        if stage_timings is not None
                        else 0.0,
                    )
                else:
                    bf16_to_f32(
                        self._prefill_hidden_a.ptr + scratch_row_start * row_nbytes,
                        verify_hidden_f32_a.ptr,
                        rows * self.runner.hidden_size,
                        stream=stream,
                        library=self.runner._cast_library(),
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
            src_f32 = verify_hidden_f32_a if use_f32_residual else None
            dst_f32 = verify_hidden_f32_b if use_f32_residual else None
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
                        bulk_scratch = (
                            _prebuilt_bulk_scratch
                            if _prebuilt_bulk_scratch is not None
                            else self._bulk_prefill_scratch.for_chunk(
                                start,
                                rows,
                                total_tokens=end,
                                runtime=runtime,
                                stream=stream,
                            )
                        )
                        src_chunk_ptr = src.ptr + scratch_row_start * row_nbytes
                        dst_chunk_ptr = dst.ptr + scratch_row_start * row_nbytes
                        src_f32_chunk_ptr = None if src_f32 is None else int(src_f32.ptr)
                        dst_f32_chunk_ptr = None if dst_f32 is None else int(dst_f32.ptr)
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
                                hidden_f32_ptr=src_f32_chunk_ptr,
                                out_f32_ptr=dst_f32_chunk_ptr,
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
                                hidden_f32_ptr=src_f32_chunk_ptr,
                                out_f32_ptr=dst_f32_chunk_ptr,
                                stage_timings=stage_timings,
                                sync_stage_timings=sync_stage_timings,
                                stage_prefix="target_block_linear_attn",
                            )
                        elif layer_type == FULL_ATTENTION:
                            key_cache, value_cache = self.scratch.full_cache(layer_id)
                            layer_scratch = replace(
                                bulk_scratch,
                                key_cache=key_cache,
                                value_cache=value_cache,
                                cos_table=self.scratch.cos_table,
                                sin_table=self.scratch.sin_table,
                            )
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
                                    hidden_f32_ptr=src_f32_chunk_ptr,
                                    out_f32_ptr=dst_f32_chunk_ptr,
                                )
                            else:
                                if use_f32_residual:
                                    raise RuntimeError(
                                        "HIPENGINE_GGUF_VERIFY_F32_RESIDUAL currently supports "
                                        "the full-attention decode-batch verifier path only (context < 1024)"
                                    )
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
                    if layer_id in capture_layer_boundary_ids:
                        layer_boundary_hidden_host[int(layer_id)] = self._capture_verify_layer_boundary_rows(
                            int(layer_id),
                            str(layer_type),
                            hidden_in_ptr=src_chunk_ptr,
                            hidden_in_f32_ptr=src_f32_chunk_ptr,
                            layer_out_ptr=dst_chunk_ptr,
                            layer_out_f32_ptr=dst_f32_chunk_ptr,
                            scratch=bulk_scratch,
                            rows=rows,
                            runtime=runtime,
                        )
                    src, dst = dst, src
                    if use_f32_residual:
                        src_f32, dst_f32 = dst_f32, src_f32
                    if layer_id in capture_layer_ids:
                        if use_f32_residual:
                            layer_output_hidden_host[int(layer_id)] = _copy_f32_ptr_to_host(
                                int(src_f32.ptr),
                                rows * self.runner.hidden_size,
                                runtime=runtime,
                            ).reshape(rows, self.runner.hidden_size)
                        else:
                            layer_output_hidden_host[int(layer_id)] = _copy_bf16_rows_to_host_f32(
                                src.ptr + start * row_nbytes,
                                rows,
                                self.runner.hidden_size,
                                runtime=runtime,
                            )
                t_output0 = time.perf_counter() if stage_timings is not None else 0.0
                final_scratch = (
                    _prebuilt_bulk_scratch
                    if _prebuilt_bulk_scratch is not None
                    else self._bulk_prefill_scratch.for_chunk(
                        start,
                        rows,
                        total_tokens=end,
                        runtime=runtime,
                        stream=stream,
                    )
                )
                output_norm_weight_ptr = self.runner.weights.root("output_norm").allocation().tensor.ptr
                if capture_pre_output_norm_hidden:
                    if use_f32_residual:
                        pre_output_norm_hidden_host = _copy_f32_ptr_to_host(
                            int(src_f32.ptr),
                            rows * self.runner.hidden_size,
                            runtime=runtime,
                        ).reshape(rows, self.runner.hidden_size)
                    else:
                        pre_output_norm_hidden_host = _copy_bf16_rows_to_host_f32(
                            src.ptr + scratch_row_start * row_nbytes,
                            rows,
                            self.runner.hidden_size,
                            runtime=runtime,
                        )
                if use_f32_residual:
                    gguf_rmsnorm_f32_f32_weight(
                        int(src_f32.ptr),
                        output_norm_weight_ptr,
                        final_scratch.norm.ptr,
                        rows=rows,
                        hidden_size=self.runner.hidden_size,
                        eps=self.runner.weights.config.rms_norm_eps,
                        stream=stream,
                        runtime=runtime,
                    )
                    gguf_rmsnorm_f32_f32_weight_out_f32(
                        int(src_f32.ptr),
                        output_norm_weight_ptr,
                        hidden_seed_buf.ptr,
                        rows=rows,
                        hidden_size=self.runner.hidden_size,
                        eps=self.runner.weights.config.rms_norm_eps,
                        stream=stream,
                        runtime=runtime,
                    )
                else:
                    gguf_rmsnorm_bf16_f32_weight(
                        src.ptr + scratch_row_start * row_nbytes,
                        output_norm_weight_ptr,
                        final_scratch.norm.ptr,
                        rows=rows,
                        hidden_size=self.runner.hidden_size,
                        eps=self.runner.weights.config.rms_norm_eps,
                        stream=stream,
                        runtime=runtime,
                    )
                    gguf_rmsnorm_bf16_f32_weight_out_f32(
                        src.ptr + scratch_row_start * row_nbytes,
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
                direct_top1 = False
                if row_lm_head:
                    direct_top1 = _gguf_verify_lm_head_q6_top1_dp4a_enabled()
                    sample_hidden_ptr = hidden_seed_buf.ptr if direct_top1 else final_scratch.norm.ptr
                    sample_dtype = GGUF_ACTIVATION_F32 if direct_top1 else GGUF_ACTIVATION_BF16
                    if _enqueue_only:
                        self._enqueue_target_block_rows_from_hidden(
                            sample_hidden_ptr,
                            rows,
                            activation_dtype=sample_dtype,
                            stream=stream,
                        )
                        if self._verify_lm_out_indices_i32 is None:
                            raise RuntimeError("GGUF verifier int32 top1 rows are closed")
                        if _target_top1_i64_ptr is not None:
                            copy_i32_to_i64(
                                self._verify_lm_out_indices_i32.ptr,
                                int(_target_top1_i64_ptr),
                                rows,
                                stream=stream,
                                library=self._runtime_state_library,
                                runtime=runtime,
                            )
                        elif int(_target_top1_i32_ptr) != int(self._verify_lm_out_indices_i32.ptr):
                            runtime.memcpy_async(
                                int(_target_top1_i32_ptr),
                                self._verify_lm_out_indices_i32.ptr,
                                rows * DType.INT32.itemsize,
                                HipMemcpyKind.DEVICE_TO_DEVICE,
                                stream,
                            )
                        if self._lm_out_index is None:
                            raise RuntimeError("GGUF resident lm-head token buffer is closed")
                        copy_i32_to_i64(
                            self._verify_lm_out_indices_i32.ptr + (rows - 1) * DType.INT32.itemsize,
                            self._lm_out_index.ptr,
                            1,
                            stream=stream,
                            library=self._runtime_state_library,
                            runtime=runtime,
                        )
                        token_host = None
                    else:
                        token_host = self._sample_target_block_rows_from_hidden(
                            sample_hidden_ptr,
                            rows,
                            activation_dtype=sample_dtype,
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
                if not _enqueue_only:
                    runtime.device_synchronize()
                    if not row_lm_head:
                        copy_device_to_host(host_array_ptr(token_host), token_ids_buf, token_host.nbytes, runtime=runtime)
                if (
                    not _enqueue_only
                    and capture_lm_head_logits
                    and row_lm_head
                    and not direct_top1
                    and self._verify_logits_buf is not None
                ):
                    lm_head_logits_host = np.empty((rows, self.runner.vocab_size), dtype=np.float32)
                    copy_device_to_host(
                        host_array_ptr(lm_head_logits_host),
                        DeviceBuffer(self._verify_logits_buf.ptr, lm_head_logits_host.nbytes),
                        lm_head_logits_host.nbytes,
                        runtime=runtime,
                    )
                add_verify_stage(
                    "target_block_lm_head_sample",
                    (time.perf_counter() - t_sample0) * 1000 if stage_timings is not None else 0.0,
                )
            if _enqueue_only:
                hidden_host = None
            else:
                t_hidden0 = time.perf_counter() if stage_timings is not None else 0.0
                hidden_host = np.empty((rows, self.runner.hidden_size), dtype=np.float32)
                copy_device_to_host(host_array_ptr(hidden_host), hidden_seed_buf, hidden_host.nbytes, runtime=runtime)
                add_verify_stage(
                    "target_block_hidden_readback",
                    (time.perf_counter() - t_hidden0) * 1000 if stage_timings is not None else 0.0,
                )
        finally:
            pass
        if _enqueue_only:
            if _dynamic_cursor_advance:
                for _ in range(rows):
                    advance_decode_position_i64(
                        self.scratch.position_buf.ptr,
                        self.scratch.context_buf.ptr,
                        stream=stream,
                        library=self._runtime_state_library,
                        runtime=runtime,
                    )
            else:
                set_decode_position_i64(
                    self.scratch.position_buf.ptr,
                    self.scratch.context_buf.ptr,
                    end,
                    stream=stream,
                    library=self._runtime_state_library,
                    runtime=runtime,
                )
            return None
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
            pre_output_norm_hidden=(
                None
                if pre_output_norm_hidden_host is None
                else np.ascontiguousarray(pre_output_norm_hidden_host, dtype=np.float32)
            ),
            layer_output_hidden=(
                None
                if not capture_layer_ids
                else {
                    int(layer_id): np.ascontiguousarray(layer_output_hidden_host[int(layer_id)], dtype=np.float32)
                    for layer_id in sorted(capture_layer_ids)
                }
            ),
            layer_boundary_hidden=(
                None
                if not capture_layer_boundary_ids
                else {
                    int(layer_id): {
                        str(name): np.ascontiguousarray(value, dtype=value.dtype)
                        for name, value in sorted(layer_boundary_hidden_host[int(layer_id)].items())
                    }
                    for layer_id in sorted(capture_layer_boundary_ids)
                }
            ),
            lm_head_logits_f32=(
                None if lm_head_logits_host is None else np.ascontiguousarray(lm_head_logits_host, dtype=np.float32)
            ),
            linear_state_rows_captured=bool(capture_linear_state_rows),
            final_linear_state_committed=not bool(defer_linear_state_commit),
        )

    def verify_target_block_serial_exact(
        self,
        input_token_ids: list[int] | tuple[int, ...],
        *,
        advance_state_only: bool = False,
        capture_linear_state_rows: bool = False,
        capture_pre_output_norm_hidden: bool = False,
        capture_layer_output_hidden: list[int] | tuple[int, ...] | set[int] | None = None,
        stream: int = 0,
    ) -> Qwen35GGUFBlockVerifyResult:
        """Consume a continuation block with the token-serial decode path.

        This is a correctness baseline for rollback-slot work.  It uses the
        same per-token kernels as :meth:`step`, then stages each hidden row and,
        optionally, each Conv/GDN state row for direct commit.  It deliberately
        does not amortize target weight loads.

        ``advance_state_only`` keeps the exact serial hidden/state path but
        skips the LM-head sampling used only to re-derive already-known block
        target tokens during accepted-prefix replay.  ``token_ids`` echoes the
        inputs in this mode and must not be used for scoring.
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
        pre_output_norm_rows: list[np.ndarray] = []
        capture_layer_ids = self._normalize_layer_output_capture(capture_layer_output_hidden)
        layer_output_rows: dict[int, list[np.ndarray]] = {int(layer_id): [] for layer_id in capture_layer_ids}
        with gemv_decode_session(self.use_gemv_decode):
            for row, token in enumerate(tokens.tolist()):
                if advance_state_only:
                    self._run_token_to_final_hidden(
                        int(token),
                        position=self._position,
                        stream=stream,
                        capture_hidden_seed_fp32=True,
                        capture_pre_output_norm_hidden=bool(capture_pre_output_norm_hidden),
                        capture_layer_output_hidden=capture_layer_ids,
                    )
                    self._position += 1
                    token_host[row] = int(token)
                else:
                    result = self.step(
                        int(token),
                        return_logits=False,
                        capture_hidden_seed_fp32=True,
                        capture_pre_output_norm_hidden=bool(capture_pre_output_norm_hidden),
                        capture_layer_output_hidden=capture_layer_ids,
                    )
                    token_host[row] = int(result.token_id)
                if capture_pre_output_norm_hidden:
                    pre_hidden = self.last_pre_output_norm_hidden
                    if pre_hidden is None:
                        raise RuntimeError("pre-output_norm hidden was requested but not captured")
                    pre_output_norm_rows.append(pre_hidden.reshape(-1))
                if capture_layer_ids:
                    last_layer_hidden = self.last_layer_output_hidden
                    for layer_id in sorted(capture_layer_ids):
                        layer_hidden = last_layer_hidden.get(int(layer_id))
                        if layer_hidden is None:
                            raise RuntimeError(f"layer {layer_id} output hidden was requested but not captured")
                        layer_output_rows[int(layer_id)].append(layer_hidden.reshape(-1))
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
            pre_output_norm_hidden=(
                np.ascontiguousarray(np.stack(pre_output_norm_rows, axis=0), dtype=np.float32)
                if capture_pre_output_norm_hidden
                else None
            ),
            layer_output_hidden=(
                None
                if not capture_layer_ids
                else {
                    int(layer_id): np.ascontiguousarray(np.stack(rows_for_layer, axis=0), dtype=np.float32)
                    for layer_id, rows_for_layer in sorted(layer_output_rows.items())
                }
            ),
            linear_state_rows_captured=bool(capture_linear_state_rows),
            final_linear_state_committed=True,
        )

    def prefill_slot(
        self,
        token_ids: list[int] | tuple[int, ...],
        *,
        slot: int,
        use_bulk: bool | None = None,
        bulk_attention_mode: str | None = None,
        return_logits: bool = False,
    ) -> Qwen35GGUFNextTokenProbeResult:
        """Prefill one resident physical slot without resetting sibling rows."""

        if self._target_scratch_owner is None or self.scratch is None:
            raise RuntimeError("GGUF resident session is closed")
        slot = int(slot)
        if slot < 0 or slot >= int(self.max_batch_size):
            raise ValueError("prefill slot outside resident max_batch_size")
        if int(self._target_scratch_owner.position_host[slot]) != 0:
            raise ValueError("prefill slot must be empty")
        if self._bulk_prefill_scratch is None:
            raise RuntimeError("GGUF resident bulk prefill scratch is closed")
        owner = self._target_scratch_owner
        if owner.kv_storage_dtype != DType.BF16:
            raise NotImplementedError("native GGUF slot prefill currently requires BF16 KV")
        runtime = self.runtime or get_hip_runtime()
        blocks = int(owner.blocks_per_slot)
        local_blocks = np.arange(blocks, dtype=np.int32)
        block_rows = int(self._bulk_prefill_scratch.rows)
        block_table = np.tile(local_blocks, (block_rows, 1))
        copy_host_to_device(
            self._bulk_prefill_scratch.block_table,
            host_array_ptr(block_table),
            block_table.nbytes,
            runtime=runtime,
        )

        def cache_slot(buffer):
            if buffer is None:
                return None
            slot_nbytes = int(buffer.nbytes) // int(owner.slot_count)
            return DeviceBuffer(int(buffer.ptr) + slot * slot_nbytes, slot_nbytes)

        saved_scratch = self.scratch
        saved_position = int(self._position)
        saved_slot_reset = bool(self._reset_current_slot_only)
        min_bulk_tokens = int(self.runner.weights.config.ssm_conv_kernel)
        run_bulk = len(token_ids) >= min_bulk_tokens if use_bulk is None else bool(use_bulk)
        slot_scratch = owner.for_slot(slot)
        if run_bulk:
            # Bulk prefill uses a slot-local block table, so its cache pointers
            # must be rebased to the same slot. Token-serial prefill keeps the
            # physical block ids and therefore must keep owner-level caches.
            slot_scratch = replace(
                slot_scratch,
                full_key_caches=tuple(cache_slot(cache) for cache in owner.full_key_caches),
                full_value_caches=tuple(cache_slot(cache) for cache in owner.full_value_caches),
            )
        self.scratch = slot_scratch
        self._position = 0
        self._reset_current_slot_only = True
        completed_position: int | None = None
        try:
            result = self.prefill(
                token_ids,
                use_bulk=use_bulk,
                bulk_attention_mode=bulk_attention_mode,
                return_logits=return_logits,
            )
            completed_position = int(self._position)
            return result
        finally:
            self._reset_current_slot_only = saved_slot_reset
            self.scratch = saved_scratch
            if completed_position is not None:
                positions = list(owner.position_host.tolist())
                positions[slot] = completed_position
                owner.set_full_attention_positions(tuple(positions), runtime)
            self._position = int(owner.position_host[0])
            if slot == 0 and self._position == 0:
                self._position = saved_position

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

    def _native_compact_scratch(
        self,
        rows: int,
        *,
        span_role: str,
        max_context_len: int | None = None,
    ):
        if self._target_scratch_owner is None:
            raise RuntimeError("GGUF resident session is closed")
        slots = tuple(range(int(rows)))
        decode_spans = self.target_spans(slot_indices=slots, span_role=span_role)
        if decode_spans.row_positions is None:
            raise RuntimeError("native target spans require row positions")
        positions = decode_spans.row_positions
        current_max_context = int(decode_spans.max_live_count)
        if max_context_len is not None:
            requested = int(max_context_len)
            if requested < current_max_context or requested > int(self.target_layout.max_sequence_length):
                raise ValueError("native graph context bound is outside resident capacity")
            decode_spans = replace(decode_spans, max_live_count=requested)
        append_spans = replace(
            decode_spans,
            live_counts=positions,
            max_live_count=max(int(self.row_positions[slot]) for slot in slots),
        )
        return replace(
            self._target_scratch_owner,
            block_table_tensor=decode_spans.base_offsets,
            position_tensor=positions,
            context_tensor=decode_spans.live_counts,
            append_spans=append_spans,
            decode_spans=decode_spans,
        )

    def _enqueue_native_rows_model(
        self,
        scratch,
        *,
        rows: int,
        stream: int,
        embedding_ready: bool,
        capture_ids: tuple[int, ...] = (),
    ) -> tuple[dict[str, str], dict[int, np.ndarray]]:
        """Enqueue one compact native model step without host token/state updates."""

        if (
            self.runner is None
            or self.runner.weights is None
            or self._token_buf is None
            or self._hidden_a is None
            or self._hidden_b is None
            or self._logits_buf is None
            or self._native_cu_seqlens_buf is None
            or self._native_state_indices_buf is None
            or self._lm_block_values is None
            or self._lm_block_indices is None
            or self._lm_out_index is None
            or self._lm_out_value is None
        ):
            raise RuntimeError("GGUF resident native-row buffers are closed")
        runtime = self.runtime or get_hip_runtime()
        if not embedding_ready:
            launch_gguf_embedding(
                self.runner.weights.root("token_embedding"),
                self._token_buf.ptr,
                self._hidden_a.ptr,
                rows=rows,
                hidden_size=self.runner.hidden_size,
                vocab_size=self.runner.vocab_size,
                stream=stream,
                runtime=runtime,
            )
        src = self._hidden_a
        dst = self._hidden_b
        captures: dict[int, np.ndarray] = {}
        execution_paths: dict[str, str] = {
            "linear_attention": "not_applicable",
            "full_attention": "not_applicable",
            "moe": "selected_rows_batch" if self.runner.weights.config.is_moe else "dense_ffn_rows",
            "lm_head": "row_linear_f32",
        }
        with (
            wmma_prefill_session(False),
            gemv_decode_session(self.use_gemv_decode),
            native_batch_decode_session(True),
        ):
            for layer_id, layer_type in enumerate(self.runner.weights.config.layer_types):
                if layer_type == LINEAR_ATTENTION:
                    execution_paths["linear_attention"] = self.runner._run_linear_attention_decode_rows_native(
                        layer_id,
                        src.ptr,
                        dst.ptr,
                        scratch,
                        rows=rows,
                        cu_seqlens_ptr=self._native_cu_seqlens_buf.ptr,
                        state_indices_ptr=self._native_state_indices_buf.ptr,
                        stream=stream,
                    )
                elif layer_type == FULL_ATTENTION:
                    execution_paths["full_attention"] = self.runner._run_full_attention_decode_rows_native(
                        layer_id,
                        src.ptr,
                        dst.ptr,
                        scratch,
                        rows=rows,
                        stream=stream,
                    )
                else:
                    raise ValueError(f"unsupported GGUF layer type {layer_type!r}")
                src, dst = dst, src
                if layer_id in capture_ids:
                    if stream:
                        runtime.stream_synchronize(stream)
                    hidden = np.empty((rows, self.runner.hidden_size), dtype=np.uint16)
                    copy_device_to_host(
                        host_array_ptr(hidden),
                        DeviceBuffer(src.ptr, hidden.nbytes),
                        hidden.nbytes,
                        runtime=runtime,
                    )
                    captures[layer_id] = hidden

            gguf_rmsnorm_bf16_f32_weight(
                src.ptr,
                self.runner.weights.root("output_norm").allocation().tensor.ptr,
                scratch.norm.ptr,
                rows=rows,
                hidden_size=self.runner.hidden_size,
                eps=self.runner.weights.config.rms_norm_eps,
                stream=stream,
                runtime=runtime,
            )
            launch_gguf_linear(
                self.runner.weights.root("lm_head"),
                scratch.norm.ptr,
                self._logits_buf.ptr,
                rows=rows,
                in_features=self.runner.hidden_size,
                out_features=self.runner.vocab_size,
                output_dtype=GGUF_OUTPUT_F32,
                stream=stream,
                runtime=runtime,
            )
            argmax_f32_rows_i32(
                self._logits_buf.ptr,
                self._lm_block_values.ptr,
                self._lm_block_indices.ptr,
                self._lm_out_index.ptr,
                self._lm_out_value.ptr,
                rows,
                self.runner.vocab_size,
                threads=self._lm_head_threads,
                stream=stream,
                library=self._lm_head_library,
                runtime=runtime,
            )
        return execution_paths, captures

    def step_rows_native(
        self,
        token_ids: tuple[int, ...] | list[int],
        positions: tuple[int, ...] | list[int] | None = None,
        *,
        slot_indices: tuple[int, ...] | list[int] | None = None,
        span_role: str = "decode",
        return_logits: bool = True,
        capture_layer_ids: tuple[int, ...] | list[int] | set[int] = (),
        stream: int = 0,
    ) -> Qwen35GGUFTargetRowsResult:
        """Advance compact c>1 rows with native state, attention, and MoE families."""

        if self.runner is None or self.runner.weights is None or self._target_scratch_owner is None:
            raise RuntimeError("GGUF resident session is closed")
        if (
            self._token_buf is None
            or self._hidden_a is None
            or self._hidden_b is None
            or self._logits_buf is None
            or self._native_cu_seqlens_buf is None
            or self._native_state_indices_buf is None
            or self._native_token_ids_host is None
        ):
            raise RuntimeError("GGUF resident native-row buffers are closed")
        tokens = tuple(int(token) for token in token_ids)
        rows = len(tokens)
        if rows <= 1:
            raise ValueError("native GGUF target execution requires at least two rows")
        if rows > int(self.max_batch_size):
            raise ValueError("native target rows exceed resident max_batch_size")
        slots = tuple(range(rows)) if slot_indices is None else tuple(int(slot) for slot in slot_indices)
        if slots != tuple(range(rows)):
            raise NotImplementedError("native GGUF target execution currently requires compact physical slots")
        if span_role not in {"decode", "verify_chain", "verify_tree"}:
            raise ValueError("span_role must be decode, verify_chain, or verify_tree")
        for token in tokens:
            if token < 0 or token >= int(self.runner.vocab_size):
                raise ValueError(f"token_id {token} outside [0, {self.runner.vocab_size})")

        current_positions = list(self.row_positions)
        consumed_positions = tuple(current_positions[slot] for slot in slots)
        if positions is not None and tuple(int(position) for position in positions) != consumed_positions:
            raise ValueError("native target row positions do not match resident cursors")
        owner = self._target_scratch_owner
        owner.set_full_attention_positions(tuple(current_positions), self.runtime or get_hip_runtime())
        scratch = self._native_compact_scratch(rows, span_role=span_role)
        capture_ids = tuple(sorted({int(layer_id) for layer_id in capture_layer_ids}))
        layer_count = len(self.runner.weights.config.layer_types)
        if any(layer_id < 0 or layer_id >= layer_count for layer_id in capture_ids):
            raise ValueError("capture_layer_ids contains an out-of-range layer")

        runtime = self.runtime or get_hip_runtime()
        self._copy_token_embeddings_to_device(
            np.asarray(tokens, dtype=np.int64),
            self._hidden_a.ptr,
            rows=rows,
            token_ids_device_ptr=self._token_buf.ptr,
            stream=stream,
        )
        execution_paths, captures = self._enqueue_native_rows_model(
            scratch,
            rows=rows,
            stream=stream,
            embedding_ready=True,
            capture_ids=capture_ids,
        )

        if return_logits:
            if stream:
                runtime.stream_synchronize(stream)
            else:
                runtime.device_synchronize()
            logits = np.empty((rows, self.runner.vocab_size), dtype=np.float32)
            copy_device_to_host(
                host_array_ptr(logits),
                DeviceBuffer(self._logits_buf.ptr, logits.nbytes),
                logits.nbytes,
                runtime=runtime,
            )
            if not np.all(np.isfinite(logits)):
                raise FloatingPointError("GGUF native target-row logits contain NaN or Inf")
            next_tokens = tuple(int(np.argmax(row)) for row in logits)
            execution_paths["sampler"] = "host_argmax_full_logits"
        else:
            if stream:
                runtime.stream_synchronize(stream)
            else:
                runtime.device_synchronize()
            copy_device_to_host(
                host_array_ptr(self._native_token_ids_host),
                DeviceBuffer(self._lm_out_index.ptr, rows * DType.INT32.itemsize),
                rows * DType.INT32.itemsize,
                runtime=runtime,
            )
            next_tokens = tuple(int(token) for token in self._native_token_ids_host[:rows])
            logits = np.empty((rows, 0), dtype=np.float32)
            execution_paths["sampler"] = "argmax_rows_i32"

        for slot in slots:
            current_positions[slot] += 1
        owner.set_full_attention_positions(tuple(current_positions), runtime)
        self._position = int(current_positions[0])
        return Qwen35GGUFTargetRowsResult(
            token_ids=next_tokens,
            positions=consumed_positions,
            slot_indices=slots,
            span_role=span_role,
            logits=logits,
            layer_hidden_bits=captures,
            execution_paths=execution_paths,
        )

    def capture_native_rows_graph(
        self,
        *,
        rows: int,
        max_context_len: int,
        span_role: str = "decode",
    ) -> "Qwen35GGUFNativeRowsGraph":
        """Capture one compact c>1 target step for host-fed graph replay."""

        if self.runner is None or self.runner.weights is None or self._target_scratch_owner is None:
            raise RuntimeError("GGUF resident session is closed")
        if self.host_token_embedding_enabled:
            raise RuntimeError("host token embedding is not compatible with native row graphs")
        rows = int(rows)
        if rows <= 1 or rows > int(self.max_batch_size):
            raise ValueError("native row graph rows must be within [2, max_batch_size]")
        if span_role not in {"decode", "verify_chain", "verify_tree"}:
            raise ValueError("span_role must be decode, verify_chain, or verify_tree")
        if self._token_buf is None:
            raise RuntimeError("GGUF resident native token buffer is closed")
        current_positions = list(self.row_positions)
        if max(current_positions[:rows]) >= int(max_context_len):
            raise ValueError("native row graph context bound must exceed current positions")
        runtime = self.runtime or get_hip_runtime()
        self._target_scratch_owner.set_full_attention_positions(tuple(current_positions), runtime)
        scratch = self._native_compact_scratch(
            rows,
            span_role=span_role,
            max_context_len=int(max_context_len),
        )
        graph = 0
        stream = runtime.stream_create()
        try:
            runtime.stream_begin_capture(stream)
            try:
                execution_paths, _ = self._enqueue_native_rows_model(
                    scratch,
                    rows=rows,
                    stream=stream,
                    embedding_ready=False,
                )
            except Exception:
                try:
                    runtime.stream_end_capture(stream)
                except Exception:
                    pass
                raise
            graph = runtime.stream_end_capture(stream)
            graph_exec = runtime.graph_instantiate(graph)
        except Exception:
            if graph:
                try:
                    runtime.graph_destroy(graph)
                except Exception:
                    pass
            runtime.stream_destroy(stream)
            raise
        return Qwen35GGUFNativeRowsGraph(
            session=self,
            graph=graph,
            graph_exec=graph_exec,
            stream=stream,
            rows=rows,
            max_context_len=int(max_context_len),
            span_role=span_role,
            execution_paths={**execution_paths, "sampler": "argmax_rows_i32_graph"},
        )

    def step_rows(
        self,
        token_ids: tuple[int, ...] | list[int],
        positions: tuple[int, ...] | list[int] | None = None,
        *,
        slot_indices: tuple[int, ...] | list[int] | None = None,
        span_role: str = "decode",
        return_logits: bool = True,
        capture_layer_ids: tuple[int, ...] | list[int] | set[int] = (),
        stream: int = 0,
    ) -> Qwen35GGUFTargetRowsResult:
        """Run target rows through the c=1-exact fallback over batch storage.

        The ABI and all resident ownership are row-shaped. This first executor
        deliberately invokes the retained c=1 layer path once per active row;
        Task #29 may replace individual families with c-aware kernels without
        changing token/state/KV metadata or the verification entrypoint.
        """

        if self.runner is None or self.runner.weights is None or self._target_scratch_owner is None:
            raise RuntimeError("GGUF resident session is closed")
        if self._token_buf is None or self._hidden_a is None or self._hidden_b is None or self._logits_buf is None:
            raise RuntimeError("GGUF resident target-row buffers are closed")
        tokens = tuple(int(token) for token in token_ids)
        if not tokens:
            raise ValueError("token_ids must contain at least one row")
        slots = tuple(range(len(tokens))) if slot_indices is None else tuple(int(slot) for slot in slot_indices)
        if len(slots) != len(tokens):
            raise ValueError("slot_indices must align with token_ids")
        if len(set(slots)) != len(slots):
            raise ValueError("slot_indices must be unique")
        if min(slots) < 0 or max(slots) >= int(self.max_batch_size):
            raise ValueError("target row slot outside resident max_batch_size")
        if span_role not in {"decode", "verify_chain", "verify_tree"}:
            raise ValueError("span_role must be decode, verify_chain, or verify_tree")
        for token in tokens:
            if token < 0 or token >= int(self.runner.vocab_size):
                raise ValueError(f"token_id {token} outside [0, {self.runner.vocab_size})")

        owner = self._target_scratch_owner
        current_positions = list(self.row_positions)
        consumed_positions = tuple(current_positions[slot] for slot in slots)
        if positions is not None:
            requested_positions = tuple(int(position) for position in positions)
            if requested_positions != consumed_positions:
                raise ValueError(
                    f"target row positions {requested_positions!r} do not match resident cursors "
                    f"{consumed_positions!r}"
                )
        owner.set_full_attention_positions(tuple(current_positions), self.runtime or get_hip_runtime())
        capture_ids = tuple(sorted({int(layer_id) for layer_id in capture_layer_ids}))
        layer_count = len(self.runner.weights.config.layer_types)
        if any(layer_id < 0 or layer_id >= layer_count for layer_id in capture_ids):
            raise ValueError("capture_layer_ids contains an out-of-range layer")

        runtime = self.runtime or get_hip_runtime()
        hidden_row_nbytes = int(self.runner.hidden_size) * DType.BF16.itemsize
        logits_row_nbytes = int(self.runner.vocab_size) * DType.FP32.itemsize
        slot_scratch = {
            slot: owner.for_slot(slot, span_role=span_role)
            for slot in slots
        }
        for token, slot in zip(tokens, slots, strict=True):
            self._copy_token_embeddings_to_device(
                np.asarray([token], dtype=np.int64),
                int(self._hidden_a.ptr) + slot * hidden_row_nbytes,
                rows=1,
                token_ids_device_ptr=int(self._token_buf.ptr) + slot * DType.INT64.itemsize,
                stream=stream,
            )

        src = self._hidden_a
        dst = self._hidden_b
        captures: dict[int, np.ndarray] = {}
        with gemv_decode_session(self.use_gemv_decode):
            for layer_id, layer_type in enumerate(self.runner.weights.config.layer_types):
                for slot in slots:
                    scratch = slot_scratch[slot]
                    src_ptr = int(src.ptr) + slot * hidden_row_nbytes
                    dst_ptr = int(dst.ptr) + slot * hidden_row_nbytes
                    if layer_type == LINEAR_ATTENTION:
                        self.runner._run_linear_attention_layer(
                            layer_id,
                            src_ptr,
                            dst_ptr,
                            scratch,
                            stream=stream,
                        )
                    elif layer_type == FULL_ATTENTION:
                        self.runner._run_full_attention_layer(
                            layer_id,
                            src_ptr,
                            dst_ptr,
                            scratch,
                            position=consumed_positions[slots.index(slot)],
                            stream=stream,
                        )
                    else:
                        raise ValueError(f"unsupported GGUF layer type {layer_type!r}")
                src, dst = dst, src
                if layer_id in capture_ids:
                    hidden = np.empty((len(slots), self.runner.hidden_size), dtype=np.uint16)
                    for row, slot in enumerate(slots):
                        copy_device_to_host(
                            host_array_ptr(hidden[row]),
                            DeviceBuffer(int(src.ptr) + slot * hidden_row_nbytes, hidden_row_nbytes),
                            runtime=runtime,
                        )
                    captures[layer_id] = hidden

            for slot in slots:
                scratch = slot_scratch[slot]
                src_ptr = int(src.ptr) + slot * hidden_row_nbytes
                gguf_rmsnorm_bf16_f32_weight(
                    src_ptr,
                    self.runner.weights.root("output_norm").allocation().tensor.ptr,
                    scratch.norm.ptr,
                    rows=1,
                    hidden_size=self.runner.hidden_size,
                    eps=self.runner.weights.config.rms_norm_eps,
                    stream=stream,
                    runtime=runtime,
                )
                launch_gguf_linear(
                    self.runner.weights.root("lm_head"),
                    scratch.norm.ptr,
                    int(self._logits_buf.ptr) + slot * logits_row_nbytes,
                    rows=1,
                    in_features=self.runner.hidden_size,
                    out_features=self.runner.vocab_size,
                    output_dtype=GGUF_OUTPUT_F32,
                    stream=stream,
                    runtime=runtime,
                )

        if stream:
            runtime.stream_synchronize(stream)
        else:
            runtime.device_synchronize()
        logits = np.empty((len(slots), self.runner.vocab_size), dtype=np.float32)
        for row, slot in enumerate(slots):
            copy_device_to_host(
                host_array_ptr(logits[row]),
                DeviceBuffer(int(self._logits_buf.ptr) + slot * logits_row_nbytes, logits_row_nbytes),
                runtime=runtime,
            )
        if not np.all(np.isfinite(logits)):
            raise FloatingPointError("GGUF target-row lm-head logits contain NaN or Inf")
        next_tokens = tuple(int(np.argmax(row)) for row in logits)

        for slot in slots:
            current_positions[slot] += 1
        owner.set_full_attention_positions(tuple(current_positions), runtime)
        self._position = int(current_positions[0])
        return Qwen35GGUFTargetRowsResult(
            token_ids=next_tokens,
            positions=consumed_positions,
            slot_indices=slots,
            span_role=span_role,
            logits=logits if return_logits else np.empty((len(slots), 0), dtype=np.float32),
            layer_hidden_bits=captures,
        )

    def verify_rows(
        self,
        batch: TargetVerifyBatch,
        *,
        return_logits: bool = True,
        capture_layer_ids: tuple[int, ...] | list[int] | set[int] = (),
        stream: int = 0,
    ) -> Qwen35GGUFTargetRowsResult:
        """Execute a correctness-first linear target chain over shared row state.

        This fallback mutates its dedicated resident target slots in row order;
        scheduler-owned transactional import/commit remains the integration
        boundary for Task #31. Tree branches fail closed until parent-state
        snapshots are wired.
        """

        if batch.mode != "verify_chain":
            raise NotImplementedError("GGUF target-row fallback currently supports verify_chain only")
        if len(batch.request_ids) > int(self.max_batch_size):
            raise ValueError("target verify requests exceed resident max_batch_size")
        if not all(batch.active_mask):
            raise NotImplementedError("inactive target verify rows are not supported by the fallback")
        slot_by_request = {
            request_id: slot
            for slot, request_id in enumerate(batch.request_ids)
        }
        last_row_by_request: dict[int, int] = {}
        row_results: list[Qwen35GGUFTargetRowsResult] = []
        for row, (token, position, request_id, parent_row) in enumerate(
            zip(
                batch.tokens,
                batch.positions,
                batch.row_to_request,
                batch.parent_rows,
                strict=True,
            )
        ):
            if row in set(batch.root_rows):
                if parent_row != -1:
                    raise ValueError("target verify root rows must have parent -1")
            elif parent_row != last_row_by_request.get(request_id):
                raise NotImplementedError(
                    "branched target verification requires parent-state snapshots"
                )
            slot = slot_by_request[request_id]
            row_results.append(
                self.step_rows(
                    (int(token),),
                    positions=(int(position),),
                    slot_indices=(slot,),
                    span_role="verify_chain",
                    return_logits=return_logits,
                    capture_layer_ids=capture_layer_ids,
                    stream=stream,
                )
            )
            last_row_by_request[request_id] = row

        capture_ids = tuple(sorted({int(layer_id) for layer_id in capture_layer_ids}))
        return Qwen35GGUFTargetRowsResult(
            token_ids=tuple(result.token_ids[0] for result in row_results),
            positions=tuple(int(position) for position in batch.positions),
            slot_indices=tuple(slot_by_request[request_id] for request_id in batch.row_to_request),
            span_role="verify_chain",
            logits=np.concatenate([result.logits for result in row_results], axis=0),
            layer_hidden_bits={
                layer_id: np.concatenate(
                    [result.layer_hidden_bits[layer_id] for result in row_results],
                    axis=0,
                )
                for layer_id in capture_ids
            },
        )

    def step(
        self,
        token_id: int,
        position: int | None = None,
        *,
        return_logits: bool = True,
        span_role: str = "decode",
        capture_hidden_seed_fp32: bool = False,
        capture_pre_output_norm_hidden: bool = False,
        capture_layer_output_hidden: list[int] | tuple[int, ...] | set[int] | None = None,
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
                span_role=span_role,
                capture_hidden_seed_fp32=bool(capture_hidden_seed_fp32),
                capture_pre_output_norm_hidden=bool(capture_pre_output_norm_hidden),
                capture_layer_output_hidden=capture_layer_output_hidden,
            )
            self._position += 1
            return self._sample_from_hidden(hidden_ptr, return_logits=return_logits)

    def step_async_top1(
        self,
        token_id: int,
        position: int | None = None,
        *,
        stream: int,
    ) -> None:
        """Launch one exact scalar decode step on ``stream`` without host sync."""

        if position is not None and int(position) != self._position:
            raise ValueError(f"position {position} does not match session cursor {self._position}")
        with gemv_decode_session(self.use_gemv_decode):
            hidden_ptr = self._run_token_to_final_hidden(
                int(token_id),
                position=self._position,
                stream=stream,
            )
            self._position += 1
            self._sample_device_from_hidden(hidden_ptr, stream=stream)

    def read_top1_sample(self) -> Qwen35GGUFNextTokenProbeResult:
        """Read the token produced by ``step_async_top1`` after stream sync."""

        return self._read_sample(return_logits=False)

    def _packed_ar_kv_layout_for_sessions(
        self,
        sessions: tuple["Qwen35GGUFResidentSession", ...],
    ) -> Qwen35GGUFKVChunkLayout:
        layout = self._device_kv_layout
        if layout is None:
            layout = _qwen35_gguf_session_kv_chunk_layout(self)
            self._device_kv_layout = layout
        for session in sessions:
            session_layout = session._device_kv_layout
            if session_layout is None:
                session_layout = _qwen35_gguf_session_kv_chunk_layout(session)
                session._device_kv_layout = session_layout
            if session_layout != layout:
                raise NotImplementedError("packed AR requires identical KV layouts across resident sessions")
        int8_layers = tuple(
            layer_id
            for layer_id, storage in enumerate(layout.layer_storage_dtypes)
            if storage == DType.INT8_PER_TOKEN_HEAD
        )
        mirror_layers = frozenset(layout.bf16_mirror_layer_indices)
        if any(layer_id not in mirror_layers for layer_id in int8_layers):
            raise NotImplementedError(
                "packed AR direct INT8 attention is not admitted without a bounded BF16 mirror"
            )
        return layout

    def prefill_batch_native(
        self,
        prompt_token_ids: list[list[int] | tuple[int, ...]] | tuple[list[int] | tuple[int, ...], ...],
        *,
        sessions: list["Qwen35GGUFResidentSession"] | tuple["Qwen35GGUFResidentSession", ...] | None = None,
        full_prompt_lengths: list[int] | tuple[int, ...] | None = None,
        return_logits: bool = False,
        return_hidden_seeds: bool = False,
        sample_output: bool = True,
        require_logits: bool = False,
        capture_layer_output_hidden: list[int] | tuple[int, ...] | set[int] | None = None,
        stream: int = 0,
    ) -> list[Qwen35GGUFNextTokenProbeResult | Qwen35GGUFPackedPrefillResult | None]:
        """Consume one prompt per session through row-bounded packed rounds.

        A prompt slab that fits runs unchanged.  Larger slabs are split fairly
        across every unfinished slot, so capacity pressure never turns a c>N
        prompt into undeclared slot-at-a-time prefill.  Conv/GDN and paged-KV
        continuity crosses rounds through the normal packed state contract.
        ``full_prompt_lengths`` preserves the full-prompt attention route when
        an outer scheduler supplies one logical prompt over multiple calls.
        ``sample_output=False`` commits model/KV state without output norm or
        LM-head work and returns one ``None`` per prompt. Chunk-tail samples are
        internal; only each prompt's final result is returned when sampling.
        ``return_hidden_seeds=True`` concatenates all per-round FP32
        rows for the llama-compatible MTP draft catch-up path.
        """

        prompt_tuple = tuple(tuple(int(token) for token in prompt) for prompt in prompt_token_ids)
        session_tuple = (self,) if sessions is None else tuple(sessions)
        if not prompt_tuple:
            raise ValueError("prompt_token_ids must be non-empty")
        if return_hidden_seeds and not sample_output:
            raise ValueError("return_hidden_seeds requires sample_output")
        if len(prompt_tuple) != len(session_tuple):
            raise ValueError("prompt_token_ids and sessions must have the same length")
        logical_lengths_supplied = full_prompt_lengths is not None
        logical_prompt_lengths = (
            tuple(len(prompt) for prompt in prompt_tuple)
            if full_prompt_lengths is None
            else tuple(int(length) for length in full_prompt_lengths)
        )
        if len(logical_prompt_lengths) != len(prompt_tuple):
            raise ValueError("full_prompt_lengths must have one entry per prompt")
        if any(length <= 0 for length in logical_prompt_lengths):
            raise ValueError("full_prompt_lengths must be positive")
        if logical_lengths_supplied and any(
            int(session.position) + len(prompt) > full_length
            for session, prompt, full_length in zip(
                session_tuple,
                prompt_tuple,
                logical_prompt_lengths,
                strict=True,
            )
        ):
            raise ValueError("prefill chunk extends beyond its declared full prompt length")
        if self._bulk_prefill_scratch is None:
            raise RuntimeError("GGUF resident bulk prefill scratch is closed")
        row_capacity = int(self._bulk_prefill_scratch.rows)
        chunks = _plan_packed_ar_prefill_chunks(
            prompt_tuple,
            row_capacity=row_capacity,
        )
        aotriton_threshold = int(PrefillConfig().attn_aotriton_min_tokens)
        aotriton_eligible_slots = tuple(
            slot_index
            for slot_index, prompt_length in enumerate(logical_prompt_lengths)
            if aotriton_threshold > 0 and prompt_length >= aotriton_threshold
        )
        self.last_packed_prefill_plan = {
            "route": "slot_fair_bounded_rounds",
            "row_capacity": row_capacity,
            "total_rows": sum(len(prompt) for prompt in prompt_tuple),
            "full_prompt_lengths": list(logical_prompt_lengths),
            "chunk_count": len(chunks),
            "chunk_rows": [int(chunk.rows) for chunk in chunks],
            "slot_indices": [list(chunk.slot_indices) for chunk in chunks],
            "start_offsets": [list(chunk.start_offsets) for chunk in chunks],
            "slot_rows": [
                [len(tokens) for tokens in chunk.prompt_token_ids]
                for chunk in chunks
            ],
            "all_active_slots_represented": True,
            "slot_serial_fallback": False,
            "aotriton_threshold": aotriton_threshold,
            "aotriton_eligible_slots": list(aotriton_eligible_slots),
            "aotriton_eligibility_preserved_across_chunks": bool(
                aotriton_eligible_slots
                and (
                    len(chunks) > 1
                    or any(
                        full_length > len(prompt)
                        for full_length, prompt in zip(
                            logical_prompt_lengths,
                            prompt_tuple,
                            strict=True,
                        )
                    )
                )
            ),
            "intermediate_tail_samples": max(
                0,
                sum(len(chunk.slot_indices) for chunk in chunks) - len(prompt_tuple),
            ),
            "sample_output": bool(sample_output),
            "device_logits_required": bool(require_logits),
            "output_norm_rows": 0,
            "lm_head_sample_rows": 0,
        }
        if len(chunks) == 1:
            return self._prefill_batch_native_single_slab(
                prompt_tuple,
                sessions=session_tuple,
                return_logits=return_logits,
                return_hidden_seeds=return_hidden_seeds,
                sample_output=sample_output,
                require_logits=require_logits,
                capture_layer_output_hidden=capture_layer_output_hidden,
                stream=stream,
                _slot_local_full_attention=bool(aotriton_eligible_slots),
                _force_aotriton_slot_indices=aotriton_eligible_slots,
            )

        initial_positions = tuple(int(session.position) for session in session_tuple)
        final_results: list[Qwen35GGUFNextTokenProbeResult | Qwen35GGUFPackedPrefillResult | None] = [
            None for _ in prompt_tuple
        ]
        hidden_parts: list[list[np.ndarray]] = [[] for _ in prompt_tuple]
        aotriton_eligible_set = set(aotriton_eligible_slots)
        for chunk in chunks:
            chunk_sessions = tuple(session_tuple[index] for index in chunk.slot_indices)
            force_aotriton_slot_indices = tuple(
                local_index
                for local_index, slot_index in enumerate(chunk.slot_indices)
                if slot_index in aotriton_eligible_set
            )
            chunk_results = self._prefill_batch_native_single_slab(
                chunk.prompt_token_ids,
                sessions=chunk_sessions,
                return_logits=return_logits,
                return_hidden_seeds=return_hidden_seeds,
                sample_output=sample_output,
                require_logits=require_logits,
                capture_layer_output_hidden=capture_layer_output_hidden,
                stream=stream,
                _slot_local_full_attention=bool(force_aotriton_slot_indices),
                _force_aotriton_slot_indices=force_aotriton_slot_indices,
            )
            if len(chunk_results) != len(chunk.slot_indices):
                raise RuntimeError("packed AR prefill chunk result count does not match active slots")
            for slot_index, result in zip(chunk.slot_indices, chunk_results, strict=True):
                final_results[slot_index] = result
                if return_hidden_seeds:
                    if not isinstance(result, Qwen35GGUFPackedPrefillResult):
                        raise RuntimeError("packed AR prefill chunk did not return hidden seeds")
                    hidden_parts[slot_index].append(
                        np.ascontiguousarray(result.hidden_seeds, dtype=np.float32)
                    )
        if not sample_output:
            return [None for _ in prompt_tuple]
        if any(result is None for result in final_results):
            raise RuntimeError("packed AR prefill did not produce a final result for every slot")
        if not return_hidden_seeds:
            return [
                result
                for result in final_results
                if result is not None
            ]

        combined: list[Qwen35GGUFPackedPrefillResult] = []
        for slot_index, result in enumerate(final_results):
            if not isinstance(result, Qwen35GGUFPackedPrefillResult):
                raise RuntimeError("packed AR prefill final result did not include hidden seeds")
            hidden_seeds = np.ascontiguousarray(
                np.concatenate(hidden_parts[slot_index], axis=0),
                dtype=np.float32,
            )
            combined.append(
                Qwen35GGUFPackedPrefillResult(
                    input_token_ids=[int(token) for token in prompt_tuple[slot_index]],
                    token_id=int(result.token_id),
                    hidden_seeds=hidden_seeds,
                    start_position=int(initial_positions[slot_index]),
                )
            )
        return combined

    def _prefill_batch_native_single_slab(
        self,
        prompt_token_ids: list[list[int] | tuple[int, ...]] | tuple[list[int] | tuple[int, ...], ...],
        *,
        sessions: list["Qwen35GGUFResidentSession"] | tuple["Qwen35GGUFResidentSession", ...] | None = None,
        return_logits: bool = False,
        return_hidden_seeds: bool = False,
        sample_output: bool = True,
        require_logits: bool = False,
        capture_layer_output_hidden: list[int] | tuple[int, ...] | set[int] | None = None,
        stream: int = 0,
        _slot_local_full_attention: bool | None = None,
        _force_aotriton_slot_indices: tuple[int, ...] = (),
    ) -> list[Qwen35GGUFNextTokenProbeResult | Qwen35GGUFPackedPrefillResult | None]:
        """Execute one packed prompt slab that already fits resident scratch."""

        prompt_tuple = tuple(tuple(int(token) for token in prompt) for prompt in prompt_token_ids)
        session_tuple = (self,) if sessions is None else tuple(sessions)
        if not prompt_tuple:
            raise ValueError("prompt_token_ids must be non-empty")
        if len(prompt_tuple) != len(session_tuple):
            raise ValueError("prompt_token_ids and sessions must have the same length")
        if return_logits and return_hidden_seeds:
            raise ValueError("packed AR prefill cannot return logits and hidden seeds together")
        if return_logits and not sample_output:
            raise ValueError("return_logits requires sample_output")
        if return_hidden_seeds and not sample_output:
            raise ValueError("return_hidden_seeds requires sample_output")
        if self.runner is None or self.runner.weights is None or self.scratch is None:
            raise RuntimeError("GGUF resident session is closed")
        if self._prefill_token_buf is None or self._prefill_hidden_a is None or self._prefill_hidden_b is None:
            raise RuntimeError("GGUF resident packed prefill buffers are closed")
        if self._bulk_prefill_scratch is None:
            raise RuntimeError("GGUF resident bulk prefill scratch is closed")
        if self.use_expert_sidecar:
            raise NotImplementedError("packed AR prefill does not support expert sidecars yet")
        if self.host_token_embedding_enabled:
            raise NotImplementedError("packed AR prefill does not support host token embedding")
        if not _gguf_verify_capture_prefill_gdn_enabled():
            raise NotImplementedError("packed AR prefill requires segmented prefill-GDN state rows")

        for prompt in prompt_tuple:
            if not prompt:
                raise ValueError("packed AR prefill prompts must be non-empty")
            for token in prompt:
                if token < 0 or token >= int(self.runner.vocab_size):
                    raise ValueError(f"token_id {token} outside [0, {self.runner.vocab_size})")
        for session in session_tuple:
            if not isinstance(session, Qwen35GGUFResidentSession):
                raise NotImplementedError("packed AR prefill requires resident GGUF sessions")
            if session.runner is not self.runner:
                raise NotImplementedError("packed AR prefill requires shared runner sessions")
            if session.scratch is None:
                raise RuntimeError("packed AR prefill job session is closed")
        self._packed_ar_kv_layout_for_sessions(session_tuple)

        slot_blocks = tuple(
            _GGUFPackedVerifySlotBlock(
                input_token_ids=prompt,
                start_position=int(session.position),
            )
            for prompt, session in zip(prompt_tuple, session_tuple, strict=True)
        )
        max_live_count = max(
            int(block.start_position) + len(block.input_token_ids)
            for block in slot_blocks
        )
        slot_capacity = max(1024, max_live_count)
        layout = _build_gguf_packed_verify_layout(slot_blocks, slot_capacity=slot_capacity)
        slot_local_full_prefill = (
            _packed_prefill_requires_slot_local_full_attention(layout)
            if _slot_local_full_attention is None
            else bool(_slot_local_full_attention)
        )
        force_aotriton_slots = set(int(index) for index in _force_aotriton_slot_indices)
        if any(index < 0 or index >= len(session_tuple) for index in force_aotriton_slots):
            raise ValueError("forced AOTriton slot index is outside the packed slab")
        device_kv_contiguous_base_rows = tuple(
            _gguf_device_kv_contiguous_base_row(session)
            for session in session_tuple
        )
        device_kv_nonidentity_scatter = any(
            base_row is None for base_row in device_kv_contiguous_base_rows
        )
        if device_kv_nonidentity_scatter:
            # Slot-local prefill cannot represent a shared prefix plus a
            # non-contiguous COW suffix. Keep those sessions in packed scratch
            # and scatter through their scheduler-owned block table below.
            slot_local_full_prefill = False
            force_aotriton_slots.clear()
        self.last_packed_prefill_plan["device_kv_nonidentity_scatter"] = bool(
            device_kv_nonidentity_scatter
        )
        self.last_packed_prefill_plan["device_kv_contiguous_base_rows"] = [
            None if base_row is None else int(base_row)
            for base_row in device_kv_contiguous_base_rows
        ]
        self.last_packed_prefill_plan["device_kv_shifted_contiguous_rebase"] = any(
            base_row not in {None, 0}
            for base_row in device_kv_contiguous_base_rows
        )
        if force_aotriton_slots and not slot_local_full_prefill:
            raise ValueError("forced AOTriton slots require slot-local full attention")
        _validate_packed_ar_prefill_context(
            layout,
            slot_local_full_prefill=slot_local_full_prefill,
        )
        capture_layer_ids = self._normalize_layer_output_capture(
            capture_layer_output_hidden
        )
        capture_row_indices = tuple(
            int(layout.cu_seqlens[slot_index + 1]) - 1
            for slot_index in range(int(layout.slot_count))
        )
        if capture_layer_ids:
            for session in session_tuple:
                session._last_layer_output_hidden = {}
        rows = int(layout.rows)
        if rows > int(self._bulk_prefill_scratch.rows):
            raise NotImplementedError(
                f"packed AR prefill rows {rows} exceed resident hidden-buffer capacity {self._bulk_prefill_scratch.rows}"
            )

        runtime = self.runtime or get_hip_runtime()
        packed_state, packed_scratch_base = self._ensure_packed_verify_workspace(
            slot_count=int(layout.slot_count),
            rows=rows,
            max_sequence_length=slot_capacity,
            runtime=runtime,
        )
        hidden_seed_buf = None
        if return_hidden_seeds:
            self._ensure_verify_block_buffers(rows, runtime=runtime)
            if self._verify_hidden_seed_buf is None:
                raise RuntimeError("GGUF packed prefill hidden-seed buffer is closed")
            hidden_seed_buf = self._verify_hidden_seed_buf
        self._sync_packed_decode_initial_state(
            session_tuple,
            layout,
            packed_state,
            runtime=runtime,
            stream=stream,
        )
        packed_scratch = packed_scratch_base.for_packed_verify_layout(layout, runtime=runtime, stream=stream)
        linear_state_plan = _packed_ar_prefill_linear_state_plan(layout)
        if linear_state_plan.capture_token_state_rows:
            self._ensure_verify_linear_state_row_buffers(
                int(linear_state_plan.transient_state_rows),
                runtime=runtime,
            )
        token_array = np.ascontiguousarray(layout.input_token_ids, dtype=np.int64)
        copy_host_to_device(self._prefill_token_buf, host_array_ptr(token_array), token_array.nbytes, runtime=runtime)

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
        linear_decode_scratch = replace(
            self.scratch,
            layer_conv_states=packed_state.layer_conv_states,
            layer_recurrent_states=packed_state.layer_recurrent_states,
        )
        full_kv_row_nbytes = self._packed_full_kv_row_nbytes()
        with wmma_prefill_session(self.use_wmma_prefill), gemv_decode_session(self.use_gemv_decode):
            for layer_id, layer_type in enumerate(self.runner.weights.config.layer_types):
                if layer_type == LINEAR_ATTENTION:
                    self.runner._run_linear_attention_prefill_layer_rows(
                        layer_id,
                        src.ptr,
                        dst.ptr,
                        packed_scratch,
                        rows=rows,
                        stream=stream,
                        decode_scratch=linear_decode_scratch,
                        expert_sidecar=None,
                        linear_state_rows=(
                            self._verify_linear_state_row_pair(layer_id)
                            if linear_state_plan.capture_token_state_rows
                            else None
                        ),
                        commit_final_linear_state=bool(
                            linear_state_plan.commit_captured_state_rows
                        ),
                        hidden_f32_ptr=None,
                        out_f32_ptr=None,
                        stage_timings=None,
                        sync_stage_timings=False,
                        stage_prefix="ar_prefill_batch_linear_attn",
                    )
                elif layer_type == FULL_ATTENTION:
                    if slot_local_full_prefill:
                        for slot_index, session in enumerate(session_tuple):
                            if session.scratch is None or session._bulk_prefill_scratch is None:
                                raise RuntimeError("packed prefill slot scratch is closed")
                            row_start = int(layout.cu_seqlens[slot_index])
                            row_end = int(layout.cu_seqlens[slot_index + 1])
                            slot_rows = row_end - row_start
                            start_position = int(layout.row_positions[row_start])
                            end_position = start_position + slot_rows
                            slot_scratch = session._bulk_prefill_scratch.for_chunk(
                                start_position,
                                slot_rows,
                                total_tokens=end_position,
                                runtime=runtime,
                                stream=stream,
                            )
                            layer_scratch = session._full_attention_prefill_scratch_for_layer(
                                slot_scratch,
                                layer_id,
                            )
                            key_cache_view = _gguf_device_kv_contiguous_cache_view(
                                session,
                                layer_scratch.key_cache,
                                row_nbytes=full_kv_row_nbytes,
                            )
                            value_cache_view = _gguf_device_kv_contiguous_cache_view(
                                session,
                                layer_scratch.value_cache,
                                row_nbytes=full_kv_row_nbytes,
                            )
                            if key_cache_view is None or value_cache_view is None:
                                raise RuntimeError(
                                    "slot-local GGUF prefill requires contiguous device KV"
                                )
                            layer_scratch = replace(
                                layer_scratch,
                                key_cache=key_cache_view,
                                value_cache=value_cache_view,
                            )
                            row_nbytes = self.runner.hidden_size * DType.BF16.itemsize
                            self.runner._run_full_attention_prefill_layer_aotriton(
                                layer_id,
                                src.ptr + row_start * row_nbytes,
                                dst.ptr + row_start * row_nbytes,
                                layer_scratch,
                                cos_table_ptr=int(session.scratch.cos_table.ptr),
                                sin_table_ptr=int(session.scratch.sin_table.ptr),
                                max_positions=int(session.scratch.max_positions),
                                stream=stream,
                                expert_sidecar=None,
                                aotriton_min_tokens=(
                                    1 if slot_index in force_aotriton_slots else None
                                ),
                            )
                    else:
                        layer_scratch = replace(
                            self._packed_full_attention_scratch_for_layer(
                                packed_scratch,
                                packed_state,
                                layer_id,
                            ),
                            cos_table=self.scratch.cos_table,
                            sin_table=self.scratch.sin_table,
                        )
                        # Packed rows are separate slot-major sequences. The
                        # paged prefill kernel consumes row-local spans and
                        # matches the c1 reduction order below the AOTriton
                        # threshold; its compact wrapper describes one sequence.
                        self.runner._run_full_attention_prefill_layer_aotriton(
                            layer_id,
                            src.ptr,
                            dst.ptr,
                            layer_scratch,
                            cos_table_ptr=int(self.scratch.cos_table.ptr),
                            sin_table_ptr=int(self.scratch.sin_table.ptr),
                            max_positions=int(self.scratch.max_positions),
                            stream=stream,
                            expert_sidecar=None,
                            allow_aotriton=False,
                            paged_max_context_len=int(layout.max_live_count),
                        )
                else:
                    raise ValueError(f"unsupported GGUF layer type {layer_type!r}")
                src, dst = dst, src
                if layer_id in capture_layer_ids:
                    hidden_row_nbytes = self.runner.hidden_size * DType.BF16.itemsize
                    selected_hidden_rows = np.ascontiguousarray(
                        np.concatenate(
                            [
                                _copy_bf16_rows_to_host_f32(
                                    src.ptr + row_index * hidden_row_nbytes,
                                    1,
                                    self.runner.hidden_size,
                                    runtime=runtime,
                                )
                                for row_index in capture_row_indices
                            ],
                            axis=0,
                        ),
                        dtype=np.float32,
                    )
                    _scatter_packed_layer_output_hidden(
                        session_tuple,
                        layer_id=layer_id,
                        hidden_rows=selected_hidden_rows,
                    )

            if linear_state_plan.commit_captured_state_rows:
                self._commit_packed_decode_linear_state_rows(
                    layout,
                    packed_state,
                    runtime=runtime,
                    stream=stream,
                )
            token_host: np.ndarray | None = None
            if sample_output:
                output_norm_weight_ptr = (
                    self.runner.weights.root("output_norm").allocation().tensor.ptr
                )
                row_nbytes = self.runner.hidden_size * DType.BF16.itemsize
                if hidden_seed_buf is not None:
                    gguf_rmsnorm_bf16_f32_weight(
                        src.ptr,
                        output_norm_weight_ptr,
                        packed_scratch.norm.ptr,
                        rows=rows,
                        hidden_size=self.runner.hidden_size,
                        eps=self.runner.weights.config.rms_norm_eps,
                        stream=stream,
                        runtime=runtime,
                    )
                    gguf_rmsnorm_bf16_f32_weight_out_f32(
                        src.ptr,
                        output_norm_weight_ptr,
                        hidden_seed_buf.ptr,
                        rows=rows,
                        hidden_size=self.runner.hidden_size,
                        eps=self.runner.weights.config.rms_norm_eps,
                        stream=stream,
                        runtime=runtime,
                    )
                    for slot_index in range(int(layout.slot_count)):
                        final_row = int(layout.cu_seqlens[slot_index + 1]) - 1
                        if final_row < int(layout.cu_seqlens[slot_index]):
                            raise RuntimeError(
                                "packed AR prefill slot has no final row to sample"
                            )
                        runtime.memcpy_async(
                            self._prefill_hidden_a.ptr + slot_index * row_nbytes,
                            packed_scratch.norm.ptr + final_row * row_nbytes,
                            row_nbytes,
                            HipMemcpyKind.DEVICE_TO_DEVICE,
                            stream,
                        )
                    output_norm_rows = rows
                else:
                    # RMSNorm is row-independent. Gather only the raw slot tails
                    # and normalize the rows that the LM head will consume.
                    for slot_index in range(int(layout.slot_count)):
                        final_row = int(layout.cu_seqlens[slot_index + 1]) - 1
                        if final_row < int(layout.cu_seqlens[slot_index]):
                            raise RuntimeError(
                                "packed AR prefill slot has no final row to sample"
                            )
                        runtime.memcpy_async(
                            packed_scratch.norm.ptr + slot_index * row_nbytes,
                            src.ptr + final_row * row_nbytes,
                            row_nbytes,
                            HipMemcpyKind.DEVICE_TO_DEVICE,
                            stream,
                        )
                    gguf_rmsnorm_bf16_f32_weight(
                        packed_scratch.norm.ptr,
                        output_norm_weight_ptr,
                        self._prefill_hidden_a.ptr,
                        rows=int(layout.slot_count),
                        hidden_size=self.runner.hidden_size,
                        eps=self.runner.weights.config.rms_norm_eps,
                        stream=stream,
                        runtime=runtime,
                    )
                    output_norm_rows = int(layout.slot_count)
                self.last_packed_prefill_plan["output_norm_rows"] = int(
                    output_norm_rows
                )
                self.last_packed_prefill_plan["lm_head_sample_rows"] = int(
                    layout.slot_count
                )
                self._enqueue_target_block_rows_from_hidden(
                    self._prefill_hidden_a.ptr,
                    int(layout.slot_count),
                    activation_dtype=GGUF_ACTIVATION_BF16,
                    stream=stream,
                    require_logits=bool(return_logits or require_logits),
                )
                token_host = self._read_target_block_row_tokens(
                    int(layout.slot_count),
                    stream=stream,
                )

        logits_host = None
        if return_logits:
            if self._verify_logits_buf is None:
                raise RuntimeError("GGUF packed AR prefill logits buffer is closed")
            logits_host = np.empty(
                (int(layout.slot_count), self.runner.vocab_size),
                dtype=np.float32,
            )
            copy_device_to_host(
                host_array_ptr(logits_host),
                DeviceBuffer(self._verify_logits_buf.ptr, logits_host.nbytes),
                logits_host.nbytes,
                runtime=runtime,
            )
            if not np.all(np.isfinite(logits_host)):
                raise FloatingPointError("GGUF packed AR prefill lm-head logits contain NaN or Inf")
        self.last_packed_prefill_plan["host_logits_d2h"] = bool(return_logits)
        self.last_packed_prefill_plan["host_logits_d2h_bytes"] = (
            0 if logits_host is None else int(logits_host.nbytes)
        )

        hidden_host = None
        if hidden_seed_buf is not None:
            hidden_host = np.empty((rows, self.runner.hidden_size), dtype=np.float32)
            copy_device_to_host(host_array_ptr(hidden_host), hidden_seed_buf, hidden_host.nbytes, runtime=runtime)
        self._scatter_packed_decode_state(
            session_tuple,
            layout,
            packed_state,
            runtime=runtime,
            stream=stream,
            copy_kv=not slot_local_full_prefill,
        )
        if hidden_seed_buf is not None:
            hidden_row_nbytes = self.runner.hidden_size * DType.FP32.itemsize
            for slot_index, session in enumerate(session_tuple):
                if session.scratch is None:
                    raise RuntimeError("packed prefill job session is closed")
                row_start = int(layout.cu_seqlens[slot_index])
                row_end = int(layout.cu_seqlens[slot_index + 1])
                slot_rows = row_end - row_start
                final_row = row_end - 1
                session._ensure_verify_block_buffers(slot_rows, runtime=runtime)
                if session._verify_hidden_seed_buf is None:
                    raise RuntimeError("packed prefill slot hidden-seed buffer is closed")
                if session is not self:
                    runtime.memcpy_async(
                        session._verify_hidden_seed_buf.ptr,
                        hidden_seed_buf.ptr + row_start * hidden_row_nbytes,
                        slot_rows * hidden_row_nbytes,
                        HipMemcpyKind.DEVICE_TO_DEVICE,
                        stream,
                    )
                    final_src_ptr = session._verify_hidden_seed_buf.ptr + (slot_rows - 1) * hidden_row_nbytes
                else:
                    final_src_ptr = hidden_seed_buf.ptr + final_row * hidden_row_nbytes
                runtime.memcpy_async(
                    session.scratch.hidden_seed_fp32.ptr,
                    final_src_ptr,
                    hidden_row_nbytes,
                    HipMemcpyKind.DEVICE_TO_DEVICE,
                    stream,
                )
                session._verify_hidden_seed_rows_populated = slot_rows
                session._hidden_seed_fp32_populated = True
        self._packed_decode_sessions = ()
        self._packed_decode_last_layout = None
        self._packed_decode_state_dirty = False
        self._packed_decode_session_ids = (
            ()
            if slot_local_full_prefill
            else tuple(id(session) for session in session_tuple)
        )
        self._packed_decode_positions = (
            ()
            if slot_local_full_prefill
            else tuple(int(session.position) for session in session_tuple)
        )
        if stream:
            runtime.stream_synchronize(stream)
        else:
            runtime.device_synchronize()
        if not sample_output:
            return [None for _ in range(int(layout.slot_count))]
        if token_host is None:
            raise RuntimeError("packed prefill sampling did not produce token IDs")
        if return_hidden_seeds:
            if hidden_host is None:
                raise RuntimeError("packed prefill hidden rows were not captured")
            return [
                Qwen35GGUFPackedPrefillResult(
                    input_token_ids=[int(token) for token in layout.input_token_ids[
                        int(layout.cu_seqlens[slot_index]):int(layout.cu_seqlens[slot_index + 1])
                    ].tolist()],
                    token_id=int(token_host[slot_index]),
                    hidden_seeds=np.ascontiguousarray(
                        hidden_host[
                            int(layout.cu_seqlens[slot_index]):int(layout.cu_seqlens[slot_index + 1])
                        ],
                        dtype=np.float32,
                    ),
                    start_position=int(layout.row_positions[int(layout.cu_seqlens[slot_index])]),
                )
                for slot_index in range(int(layout.slot_count))
            ]
        results: list[Qwen35GGUFNextTokenProbeResult] = []
        for slot_index, token in enumerate(token_host.tolist()):
            token_id = int(token)
            row_logits = (
                np.empty((0,), dtype=np.float32)
                if logits_host is None
                else np.ascontiguousarray(logits_host[slot_index : slot_index + 1])
            )
            results.append(
                Qwen35GGUFNextTokenProbeResult(
                    token_id=token_id,
                    logit=(
                        0.0
                        if logits_host is None
                        else float(logits_host[slot_index, token_id])
                    ),
                    logits=row_logits,
                )
            )
        return results

    def _enqueue_packed_decode_model_step(
        self,
        *,
        rows: int,
        state_indices: tuple[int, ...],
        packed_scratch,
        packed_state,
        linear_decode_scratch,
        stream: int,
        split_workspace: _GGUFPackedARAttentionWorkspace | None = None,
        layer_output_callback=None,
        require_logits: bool = False,
    ) -> tuple[set[str], set[str]]:
        """Enqueue one c-aware packed model step without host synchronization."""

        if self.runner is None or self.runner.weights is None or self.scratch is None:
            raise RuntimeError("GGUF resident session is closed")
        if self._prefill_token_buf is None or self._prefill_hidden_a is None or self._prefill_hidden_b is None:
            raise RuntimeError("GGUF resident packed decode buffers are closed")
        runtime = self.runtime or get_hip_runtime()
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
        linear_attention_decode_paths: set[str] = set()
        full_attention_decode_paths: set[str] = set()
        linear_attention_decode_batch_plan = self.runner._linear_attention_decode_batch_plan()
        q8_t16_rowtile_all = bool(
            backend_package_capability(
                self.runner.backend,
                "GGUF_Q8_T16_DECODE_ROWTILE_ALL",
                False,
            )
        )
        q8_t16_pair_rowtile_min_rows = int(
            backend_package_capability(
                self.runner.backend,
                "GGUF_Q8_T16_DECODE_PAIR_ROWTILE_MIN_ROWS",
                0,
            )
        )
        selected_pairreuse_min_rows = int(
            backend_package_capability(
                self.runner.backend,
                "GGUF_Q4_T16_SELECTED_PAIRREUSE_MIN_ROWS",
                0,
            )
        )
        selected_down_pairreuse_min_rows = int(
            backend_package_capability(
                self.runner.backend,
                "GGUF_Q5_T16_SELECTED_PAIRREUSE_MIN_ROWS",
                0,
            )
        )
        selected_q6_down_pairreuse_min_rows = int(
            backend_package_capability(
                self.runner.backend,
                "GGUF_Q6_T16_SELECTED_PAIRREUSE_MIN_ROWS",
                0,
            )
        )
        with (
            wmma_prefill_session(False),
            gemv_decode_session(self.use_gemv_decode),
            _gguf_t16_selected_pairreuse_min_rows_scope(selected_pairreuse_min_rows),
            _gguf_t16_selected_down_pairreuse_min_rows_scope(selected_down_pairreuse_min_rows),
            _gguf_t16_selected_q6_down_pairreuse_min_rows_scope(
                selected_q6_down_pairreuse_min_rows
            ),
            q8_t16_pair_rowtile_min_rows_session(q8_t16_pair_rowtile_min_rows),
            q8_t16_rowtile_all_session(q8_t16_rowtile_all),
        ):
            for layer_id, layer_type in enumerate(self.runner.weights.config.layer_types):
                if layer_type == LINEAR_ATTENTION:
                    linear_attention_decode_paths.add(
                        self.runner._run_linear_attention_decode_slot_rows_exact(
                            layer_id,
                            src.ptr,
                            dst.ptr,
                            packed_scratch,
                            rows=rows,
                            state_indices=state_indices,
                            stream=stream,
                            decode_scratch=linear_decode_scratch,
                            batch_plan=linear_attention_decode_batch_plan,
                            gdn_cu_seqlens_ptr=packed_scratch.gdn_cu_seqlens.ptr,
                            state_indices_ptr=packed_scratch.gdn_state_indices.ptr,
                            expert_sidecar=None,
                            hidden_f32_ptr=None,
                            out_f32_ptr=None,
                            stage_timings=None,
                            sync_stage_timings=False,
                            stage_prefix="ar_batch_linear_attn",
                        )
                    )
                elif layer_type == FULL_ATTENTION:
                    layer_scratch = replace(
                        self._packed_full_attention_scratch_for_layer(
                            packed_scratch,
                            packed_state,
                            layer_id,
                        ),
                        cos_table=self.scratch.cos_table,
                        sin_table=self.scratch.sin_table,
                    )
                    full_attention_decode_paths.add(
                        self.runner._run_full_attention_decode_batch_layer_rows(
                            layer_id,
                            src.ptr,
                            dst.ptr,
                            layer_scratch,
                            stream=stream,
                            expert_sidecar=None,
                            stage_timings=None,
                            sync_stage_timings=False,
                            stage_prefix="ar_batch_full_attn",
                            split_workspace=split_workspace,
                        )
                    )
                else:
                    raise ValueError(f"unsupported GGUF layer type {layer_type!r}")
                src, dst = dst, src
                if layer_output_callback is not None:
                    layer_output_callback(layer_id, src.ptr)

            output_norm_weight_ptr = self.runner.weights.root("output_norm").allocation().tensor.ptr
            gguf_rmsnorm_bf16_f32_weight(
                src.ptr,
                output_norm_weight_ptr,
                packed_scratch.norm.ptr,
                rows=rows,
                hidden_size=self.runner.hidden_size,
                eps=self.runner.weights.config.rms_norm_eps,
                stream=stream,
                runtime=runtime,
            )
            self._enqueue_target_block_rows_from_hidden(
                packed_scratch.norm.ptr,
                rows,
                activation_dtype=GGUF_ACTIVATION_BF16,
                stream=stream,
                require_logits=bool(require_logits),
            )
        return linear_attention_decode_paths, full_attention_decode_paths

    def step_batch_native(
        self,
        token_ids: list[int] | tuple[int, ...],
        *,
        sessions: list["Qwen35GGUFResidentSession"] | tuple["Qwen35GGUFResidentSession", ...] | None = None,
        positions: list[int] | tuple[int, ...] | None = None,
        return_logits: bool = False,
        require_logits: bool = False,
        stream: int = 0,
        scatter_state: bool = True,
        capture_layer_output_hidden: list[int] | tuple[int, ...] | set[int] | None = None,
        physical_rows: int | None = None,
        active_slot_indices: list[int] | tuple[int, ...] | None = None,
    ) -> list[Qwen35GGUFNextTokenProbeResult]:
        """Consume one decode token for each resident session in one packed pass.

        This is the AR-serving counterpart to ``verify_target_blocks_batch``.
        Linear attention is c1-exact per slot while MoE/FFN and full attention
        remain row-batched. Every input row is committed, so the path skips
        verifier hidden-row D2H. Consecutive calls with the same session tuple
        keep the packed workspace canonical and avoid re-importing per-slot
        Conv/GDN and full-attention history.
        """

        token_tuple = tuple(int(token) for token in token_ids)
        session_tuple = (self,) if sessions is None else tuple(sessions)
        if not token_tuple:
            raise ValueError("token_ids must be non-empty")
        if len(token_tuple) != len(session_tuple):
            raise ValueError("token_ids and sessions must have the same length")
        if self.runner is None or self.runner.weights is None or self.scratch is None:
            raise RuntimeError("GGUF resident session is closed")
        if self._prefill_token_buf is None or self._prefill_hidden_a is None or self._prefill_hidden_b is None:
            raise RuntimeError("GGUF resident packed decode buffers are closed")
        if self._bulk_prefill_scratch is None:
            raise RuntimeError("GGUF resident bulk prefill scratch is closed")
        if self.use_expert_sidecar:
            raise NotImplementedError("packed AR decode does not support expert sidecars yet")
        if self.host_token_embedding_enabled:
            raise NotImplementedError("packed AR decode does not support host token embedding")
        if not _gguf_verify_capture_prefill_gdn_enabled():
            raise NotImplementedError("packed AR decode requires segmented prefill-GDN state rows")

        position_tuple = tuple(int(session.position) for session in session_tuple)
        if positions is not None:
            expected = tuple(int(position) for position in positions)
            if expected != position_tuple:
                raise ValueError(f"packed AR positions {expected!r} do not match session cursors {position_tuple!r}")
        for token in token_tuple:
            if token < 0 or token >= int(self.runner.vocab_size):
                raise ValueError(f"token_id {token} outside [0, {self.runner.vocab_size})")
        for session in session_tuple:
            if not isinstance(session, Qwen35GGUFResidentSession):
                raise NotImplementedError("packed AR decode requires resident GGUF sessions")
            if session.runner is not self.runner:
                raise NotImplementedError("packed AR decode requires shared runner sessions")
            if session.scratch is None:
                raise RuntimeError("packed AR decode job session is closed")
        self._packed_ar_kv_layout_for_sessions(session_tuple)

        active_rows = len(session_tuple)
        row_count = active_rows if physical_rows is None else int(physical_rows)
        if row_count <= 0 or row_count > 8 or row_count < active_rows:
            raise ValueError("physical_rows must be in [active_rows, 8]")
        active_slots = (
            tuple(range(active_rows))
            if active_slot_indices is None
            else tuple(int(index) for index in active_slot_indices)
        )
        if len(active_slots) != active_rows:
            raise ValueError("active_slot_indices must align with token_ids and sessions")
        if (
            len(set(active_slots)) != len(active_slots)
            or any(index < 0 or index >= row_count for index in active_slots)
        ):
            raise ValueError("active_slot_indices must be unique lanes within physical_rows")
        physical_sessions: list[Qwen35GGUFResidentSession | None] = [None] * row_count
        physical_tokens = [0] * row_count
        physical_positions = [-1] * row_count
        for token, session, position, slot_index in zip(
            token_tuple,
            session_tuple,
            position_tuple,
            active_slots,
            strict=True,
        ):
            physical_sessions[slot_index] = session
            physical_tokens[slot_index] = token
            physical_positions[slot_index] = position
        physical_session_tuple = tuple(physical_sessions)
        active_mask = tuple(session is not None for session in physical_session_tuple)
        slot_blocks = tuple(
            _GGUFPackedVerifySlotBlock(
                input_token_ids=(physical_tokens[index],),
                start_position=physical_positions[index],
                active=active_mask[index],
            )
            for index in range(row_count)
        )
        max_live_count = max(
            int(block.start_position) + len(block.input_token_ids)
            for block in slot_blocks
            if block.active
        )
        slot_capacity = _packed_ar_slot_capacity(max_live_count)
        layout = _build_gguf_packed_verify_layout(slot_blocks, slot_capacity=slot_capacity)
        capture_layer_ids = self._normalize_layer_output_capture(
            capture_layer_output_hidden
        )
        if capture_layer_ids:
            for session in session_tuple:
                session._last_layer_output_hidden = {}
        rows = int(layout.rows)
        if rows != row_count:
            raise RuntimeError("packed AR physical layout row count drift")
        if rows > int(self._bulk_prefill_scratch.rows):
            raise NotImplementedError(
                f"packed AR rows {rows} exceed resident hidden-buffer capacity {self._bulk_prefill_scratch.rows}"
            )

        runtime = self.runtime or get_hip_runtime()
        packed_state, packed_scratch_base = self._ensure_packed_verify_workspace(
            slot_count=int(layout.slot_count),
            rows=rows,
            max_sequence_length=slot_capacity,
            runtime=runtime,
            stream=stream,
        )
        split_workspace = (
            self._ensure_packed_ar_attention_workspace(
                rows=rows,
                max_context_len=int(layout.max_live_count),
                runtime=runtime,
            )
            if int(layout.max_live_count) >= 1024
            else None
        )
        imported_slot_indices = self._sync_packed_decode_initial_state(
            physical_session_tuple,
            layout,
            packed_state,
            runtime=runtime,
            stream=stream,
        )
        packed_scratch = packed_scratch_base.for_packed_verify_layout(
            layout,
            runtime=runtime,
            stream=stream,
            metadata_prepare_fn=self.runner._packed_decode_metadata_kernel(),
        )
        token_array = np.ascontiguousarray(layout.input_token_ids, dtype=np.int64)
        copy_host_to_device(self._prefill_token_buf, host_array_ptr(token_array), token_array.nbytes, runtime=runtime)

        linear_decode_scratch = replace(
            self.scratch,
            layer_conv_states=packed_state.layer_conv_states,
            layer_recurrent_states=packed_state.layer_recurrent_states,
        )

        def capture_layer_output(layer_id: int, hidden_ptr: int) -> None:
            if layer_id not in capture_layer_ids:
                return
            hidden_rows = _copy_bf16_rows_to_host_f32(
                hidden_ptr,
                rows,
                self.runner.hidden_size,
                runtime=runtime,
            )
            _scatter_packed_layer_output_hidden(
                session_tuple,
                layer_id=layer_id,
                hidden_rows=hidden_rows,
                row_indices=active_slots,
            )

        linear_attention_decode_paths, full_attention_decode_paths = (
            self._enqueue_packed_decode_model_step(
                rows=rows,
                state_indices=tuple(
                    int(index) for index in layout.row_slot_indices.tolist()
                ),
                packed_scratch=packed_scratch,
                packed_state=packed_state,
                linear_decode_scratch=linear_decode_scratch,
                stream=stream,
                split_workspace=split_workspace,
                layer_output_callback=(
                    capture_layer_output if capture_layer_ids else None
                ),
                require_logits=bool(return_logits or require_logits),
            )
        )
        token_host = self._read_target_block_row_tokens(rows, stream=stream)
        logits_host = None
        if return_logits:
            if self._verify_logits_buf is None:
                raise RuntimeError("GGUF packed AR logits buffer is closed")
            logits_host = np.empty((rows, self.runner.vocab_size), dtype=np.float32)
            copy_device_to_host(
                host_array_ptr(logits_host),
                DeviceBuffer(self._verify_logits_buf.ptr, logits_host.nbytes),
                logits_host.nbytes,
                runtime=runtime,
            )
            if not np.all(np.isfinite(logits_host[np.asarray(active_slots, dtype=np.intp)])):
                raise FloatingPointError("GGUF packed AR lm-head logits contain NaN or Inf")

        session_ids = tuple(
            0 if session is None else id(session)
            for session in physical_session_tuple
        )
        if scatter_state:
            self._scatter_packed_decode_state(
                physical_session_tuple,
                layout,
                packed_state,
                runtime=runtime,
                stream=stream,
            )
            self._packed_decode_sessions = ()
            self._packed_decode_last_layout = None
            self._packed_decode_state_dirty = False
        else:
            self._advance_packed_decode_session_cursors(physical_session_tuple, layout)
            self._packed_decode_sessions = physical_session_tuple
            self._packed_decode_last_layout = layout
            self._packed_decode_state_dirty = True
        self._packed_decode_session_ids = session_ids
        self._packed_decode_positions = tuple(
            -1 if session is None else int(session.position)
            for session in physical_session_tuple
        )
        if stream:
            runtime.stream_synchronize(stream)
        else:
            runtime.device_synchronize()
        if len(linear_attention_decode_paths) > 1:
            raise RuntimeError(
                "packed linear-attention layers used inconsistent decode paths: "
                f"{sorted(linear_attention_decode_paths)!r}"
            )
        linear_attention_decode_path = (
            next(iter(linear_attention_decode_paths))
            if linear_attention_decode_paths
            else "not_applicable"
        )
        if len(full_attention_decode_paths) > 1:
            raise RuntimeError(
                "packed full-attention layers used inconsistent decode paths: "
                f"{sorted(full_attention_decode_paths)!r}"
            )
        full_attention_decode_path = (
            next(iter(full_attention_decode_paths))
            if full_attention_decode_paths
            else "not_applicable"
        )
        self.last_packed_execution_manifest = build_packed_decode_execution_manifest(
            rows=rows,
            layer_types=self.runner.weights.config.layer_types,
            imported_slot_indices=imported_slot_indices,
            import_positions=tuple(physical_positions),
            scatter_state=bool(scatter_state),
            active_mask=active_mask,
            blocks_per_slot=int(layout.blocks_per_slot),
            capture_layer_count=len(capture_layer_ids),
            linear_attention_decode_path=linear_attention_decode_path,
            gdn_recurrent_decode_path=(
                self.runner._linear_attention_decode_batch_plan().gdn_decode_path
                if linear_attention_decode_path == "indexed_batch"
                else None
            ),
            full_attention_decode_path=full_attention_decode_path,
            moe_decode_path=(
                "selected_rows_batch"
                if self.runner.weights.config.is_moe
                else "dense_ffn_rows"
            ),
            moe_top_k=(
                int(self.runner.weights.config.expert_used_count)
                if self.runner.weights.config.is_moe
                else 0
            ),
            lm_head_decode_path=self._last_packed_lm_head_decode_path,
            sampler_decode_path=self._last_packed_sampler_decode_path,
            metadata_prepare_path=str(packed_scratch.metadata_prepare_path),
        )
        self.last_packed_execution_manifest["active_slot_indices"] = list(active_slots)
        self.last_packed_execution_manifest["host_logits_d2h"] = bool(return_logits)
        self.last_packed_execution_manifest["host_logits_d2h_bytes"] = (
            0
            if logits_host is None
            else int(rows * self.runner.vocab_size * DType.FP32.itemsize)
        )
        results: list[Qwen35GGUFNextTokenProbeResult] = []
        for slot_index in active_slots:
            token_id = int(token_host[slot_index])
            row_logits = (
                np.empty((0,), dtype=np.float32)
                if logits_host is None
                else np.ascontiguousarray(logits_host[slot_index : slot_index + 1])
            )
            results.append(
                Qwen35GGUFNextTokenProbeResult(
                    token_id=token_id,
                    logit=(
                        0.0
                        if logits_host is None
                        else float(logits_host[slot_index, token_id])
                    ),
                    logits=row_logits,
                )
            )
        return results

    def _run_token_to_final_hidden(
        self,
        token_id: int,
        *,
        position: int,
        span_role: str = "decode",
        stream: int = 0,
        capture_hidden_seed_fp32: bool = False,
        capture_pre_output_norm_hidden: bool = False,
        capture_layer_output_hidden: list[int] | tuple[int, ...] | set[int] | None = None,
    ) -> int:
        if self._token_buf is None or self.scratch is None:
            raise RuntimeError("GGUF resident session buffers are closed")
        scratch = self.scratch.for_slot(0, span_role=span_role)
        self._set_full_attention_position_device(position, stream=stream, scratch=scratch)
        self._set_token_id_device(int(token_id), stream=stream)
        return self._run_current_hidden_to_final_hidden(
            position=position,
            stream=stream,
            scratch=scratch,
            capture_hidden_seed_fp32=bool(capture_hidden_seed_fp32),
            capture_pre_output_norm_hidden=bool(capture_pre_output_norm_hidden),
            capture_layer_output_hidden=capture_layer_output_hidden,
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
        moe_router_logits = None
        moe_selected_intermediate = None
        moe_shared_intermediate = None
        linear_qkv = None
        linear_z = None
        ssm_alpha = None
        ssm_beta = None
        conv_out = None
        recurrent_out = None
        recurrent_bf16 = None
        if layer_type == LINEAR_ATTENTION:
            linear_qkv_width = int(self.runner.linear_qkv_width)
            ssm_inner_size = int(cfg.ssm_inner_size)
            ssm_time_step_rank = int(cfg.ssm_time_step_rank)
            linear_qkv = _copy_bf16_ptr_to_host_f32(
                int(self.scratch.linear_qkv.ptr),
                linear_qkv_width,
                runtime=runtime,
            )
            linear_z = _copy_bf16_ptr_to_host_f32(
                int(self.scratch.linear_z.ptr),
                ssm_inner_size,
                runtime=runtime,
            )
            ssm_alpha = _copy_bf16_ptr_to_host_f32(
                int(self.scratch.linear_alpha.ptr),
                ssm_time_step_rank,
                runtime=runtime,
            )
            ssm_beta = _copy_bf16_ptr_to_host_f32(
                int(self.scratch.linear_beta.ptr),
                ssm_time_step_rank,
                runtime=runtime,
            )
            conv_out = _copy_f32_ptr_to_host(
                int(self.scratch.conv_out.ptr),
                linear_qkv_width,
                runtime=runtime,
            )
            recurrent_out = _copy_f32_ptr_to_host(
                int(self.scratch.recurrent_out.ptr),
                ssm_inner_size,
                runtime=runtime,
            )
            recurrent_bf16 = _copy_bf16_ptr_to_host_f32(
                int(self.scratch.recurrent_bf16.ptr),
                ssm_inner_size,
                runtime=runtime,
            )
        if cfg.is_moe:
            moe_router_logits = _copy_f32_ptr_to_host(
                int(self.scratch.moe_router_logits.ptr),
                int(cfg.expert_count),
                runtime=runtime,
            )
            moe_selected_intermediate = _copy_bf16_ptr_to_host_f32(
                int(self.scratch.ffn_intermediate.ptr),
                top_k * int(cfg.expert_feed_forward_length),
                runtime=runtime,
            )
            moe_shared_intermediate = _copy_bf16_ptr_to_host_f32(
                int(self.scratch.moe_shared_intermediate.ptr),
                int(cfg.expert_shared_feed_forward_length),
                runtime=runtime,
            )
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
            attn_norm_f32=_copy_bf16_ptr_to_host_f32(
                int(self.scratch.norm.ptr), hidden_size, runtime=runtime
            ),
            attn_out_f32=_copy_bf16_ptr_to_host_f32(
                int(self.scratch.attn_out.ptr), hidden_size, runtime=runtime
            ),
            post_norm_f32=_copy_bf16_ptr_to_host_f32(
                int(self.scratch.post_norm.ptr), hidden_size, runtime=runtime
            ),
            post_norm_source="bf16_scratch.post_norm",
            residual_f32=_copy_bf16_ptr_to_host_f32(
                int(self.scratch.residual.ptr), hidden_size, runtime=runtime
            ),
            ffn_or_moe_down_f32=_copy_bf16_ptr_to_host_f32(
                down_ptr, down_elements, runtime=runtime
            ),
            layer_out_f32=_copy_bf16_ptr_to_host_f32(
                target_dst_ptr, hidden_size, runtime=runtime
            ),
            moe_router_logits_f32=moe_router_logits,
            moe_selected_intermediate_f32=moe_selected_intermediate,
            moe_shared_intermediate_f32=moe_shared_intermediate,
            moe_shared_out_f32=moe_shared_out,
            moe_routing_weights_f32=moe_routing_weights,
            moe_shared_gate_f32=moe_shared_gate,
            moe_selected_experts_i64=moe_selected_experts,
            linear_qkv_f32=linear_qkv,
            linear_z_f32=linear_z,
            ssm_alpha_f32=ssm_alpha,
            ssm_beta_f32=ssm_beta,
            conv_out_f32=conv_out,
            recurrent_out_f32=recurrent_out,
            recurrent_bf16_f32=recurrent_bf16,
        )

    def capture_attention_router_trace(
        self,
        token_id: int,
        *,
        position: int,
        stream: int = 0,
    ) -> list[Qwen35GGUFRouterTraceLayerCapture]:
        """Run one decode row and capture per-layer MoE router state.

        This diagnostic tap mutates the resident decode state like ``step`` for
        the supplied token/position, but stops before output_norm and sampling.
        It is intended for cross-engine router/top-k bisection, not generation.
        """

        if self.runner is None or self.runner.weights is None or self.scratch is None:
            raise RuntimeError("GGUF resident session is closed")
        if self._hidden_a is None or self._hidden_b is None:
            raise RuntimeError("GGUF resident session buffers are closed")
        if position < 0:
            raise ValueError("position must be non-negative")
        cfg = self.runner.weights.config
        if not cfg.is_moe:
            raise ValueError("router trace requires a MoE GGUF model")

        runtime = self.runtime or get_hip_runtime()
        self._hidden_seed_fp32_populated = False
        self._set_full_attention_position_device(position, stream=stream)
        self._set_token_id_device(int(token_id), stream=stream)
        src = self._hidden_a
        dst = self._hidden_b
        hidden_size = int(self.runner.hidden_size)
        expert_count = int(cfg.expert_count)
        top_k = int(cfg.expert_used_count)
        if top_k <= 0:
            raise ValueError("qwen35moe GGUF expert_used_count must be positive")

        captures: list[Qwen35GGUFRouterTraceLayerCapture] = []
        for layer_id, layer_type in enumerate(cfg.layer_types):
            hidden_in = _copy_bf16_ptr_to_host_f32(
                int(src.ptr), hidden_size, runtime=runtime
            )
            if layer_type == LINEAR_ATTENTION:
                self.runner._run_linear_attention_layer(
                    layer_id,
                    src.ptr,
                    dst.ptr,
                    self.scratch,
                    stream=stream,
                )
            elif layer_type == FULL_ATTENTION:
                self.runner._run_full_attention_layer(
                    layer_id,
                    src.ptr,
                    dst.ptr,
                    self.scratch,
                    position=position,
                    stream=stream,
                )
            else:
                raise ValueError(f"unsupported GGUF layer type {layer_type!r}")
            if stream:
                runtime.stream_synchronize(stream)
            else:
                runtime.device_synchronize()
            shared_gate_ptr = (
                int(self.scratch.moe_router_logits.ptr)
                + expert_count * DType.FP32.itemsize
            )
            captures.append(
                Qwen35GGUFRouterTraceLayerCapture(
                    layer_id=int(layer_id),
                    layer_type=str(layer_type),
                    token_id=int(token_id),
                    position=int(position),
                    hidden_size=hidden_size,
                    expert_count=expert_count,
                    top_k=top_k,
                    hidden_in_f32=np.ascontiguousarray(hidden_in, dtype=np.float32),
                    layer_out_f32=_copy_bf16_ptr_to_host_f32(
                        int(dst.ptr), hidden_size, runtime=runtime
                    ),
                    moe_router_logits_f32=_copy_f32_ptr_to_host(
                        int(self.scratch.moe_router_logits.ptr),
                        expert_count,
                        runtime=runtime,
                    ),
                    moe_routing_weights_f32=_copy_f32_ptr_to_host(
                        int(self.scratch.moe_routing_weights.ptr),
                        top_k,
                        runtime=runtime,
                    ),
                    moe_shared_gate_f32=_copy_f32_ptr_to_host(
                        shared_gate_ptr,
                        1,
                        runtime=runtime,
                    ),
                    moe_selected_experts_i64=_copy_i64_ptr_to_host(
                        int(self.scratch.moe_selected_experts.ptr),
                        top_k,
                        runtime=runtime,
                    ),
                )
            )
            src, dst = dst, src
        return captures

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
        max_context_len: int | None = None,
        attention_max_context_len: int | None = None,
        stream: int = 0,
        scratch=None,
        capture_hidden_seed_fp32: bool = False,
        capture_pre_output_norm_hidden: bool = False,
        capture_layer_output_hidden: list[int] | tuple[int, ...] | set[int] | None = None,
    ) -> int:
        if self.runner is None or self.scratch is None:
            raise RuntimeError("GGUF resident session is closed")
        scratch = self.scratch if scratch is None else scratch
        if self._hidden_a is None or self._hidden_b is None:
            raise RuntimeError("GGUF resident session buffers are closed")
        assert self.runner.weights is not None
        if max_context_len is None:
            max_context_len = attention_max_context_len
        elif attention_max_context_len is not None and int(max_context_len) != int(attention_max_context_len):
            raise ValueError("conflicting GGUF attention context limits")
        self._hidden_seed_fp32_populated = False
        capture_layer_ids = self._normalize_layer_output_capture(capture_layer_output_hidden)
        self._last_layer_output_hidden = {}
        scratch.position_host[0] = int(position)
        scratch.context_host[0] = int(position) + 1
        src = self._hidden_a
        dst = self._hidden_b
        layer_types = self.runner.weights.config.layer_types
        chain_next_rms = self.runner.weights.config.is_moe and _gguf_moe_tail_next_rms_enabled()
        moe_graph = self._moe_graph_for_decode() if scratch is self.scratch and not chain_next_rms else None
        input_norm_ptr: int | None = None
        for layer_id, layer_type in enumerate(layer_types):
            next_norm_weight_ptr = None
            next_norm_out_ptr = None
            if chain_next_rms and layer_id + 1 < len(layer_types):
                next_norm_weight_ptr = (
                    self.runner.weights.layer(layer_id + 1).weight("attn_norm").allocation().tensor.ptr
                )
                next_norm_out_ptr = scratch.norm.ptr
            if moe_graph is not None:
                self._run_decode_layer_graphed(
                    layer_id,
                    layer_type,
                    src.ptr,
                    dst.ptr,
                    moe_graph,
                    position=position,
                    stream=stream,
                    attention_max_context_len=max_context_len,
                )
            elif layer_type == LINEAR_ATTENTION:
                self.runner._run_linear_attention_layer(
                    layer_id,
                    src.ptr,
                    dst.ptr,
                    scratch,
                    input_norm_ptr=input_norm_ptr,
                    next_norm_weight_ptr=next_norm_weight_ptr,
                    next_norm_out_ptr=next_norm_out_ptr,
                    stream=stream,
                )
            elif layer_type == FULL_ATTENTION:
                self.runner._run_full_attention_layer(
                    layer_id,
                    src.ptr,
                    dst.ptr,
                    scratch,
                    position=position,
                    max_context_len=max_context_len,
                    input_norm_ptr=input_norm_ptr,
                    next_norm_weight_ptr=next_norm_weight_ptr,
                    next_norm_out_ptr=next_norm_out_ptr,
                    stream=stream,
                )
            else:
                raise ValueError(f"unsupported GGUF layer type {layer_type!r}")
            input_norm_ptr = next_norm_out_ptr
            src, dst = dst, src
            if layer_id in capture_layer_ids:
                self._last_layer_output_hidden[int(layer_id)] = _copy_bf16_rows_to_host_f32(
                    src.ptr,
                    1,
                    self.runner.hidden_size,
                    runtime=self.runtime or get_hip_runtime(),
                )
        self._last_target_hidden_ptr = int(src.ptr)
        if capture_pre_output_norm_hidden:
            self._last_pre_output_norm_hidden = _copy_bf16_rows_to_host_f32(
                src.ptr,
                1,
                self.runner.hidden_size,
                runtime=self.runtime or get_hip_runtime(),
            )
        else:
            self._last_pre_output_norm_hidden = None
        return self._run_output_norm_hidden(
            src.ptr,
            scratch.norm.ptr,
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

    def _seed_decode_graph_input_token(self, token_id: int) -> None:
        """Publish the authoritative host token before cross-stream graph capture."""

        if self.runner is None or self._lm_out_index is None:
            raise RuntimeError("GGUF resident session is closed")
        token_id = int(token_id)
        if token_id < 0 or token_id >= self.runner.vocab_size:
            raise ValueError(f"token_id {token_id} outside [0, {self.runner.vocab_size})")
        runtime = self.runtime or get_hip_runtime()
        set_i64_scalar(
            self._lm_out_index.ptr,
            token_id,
            stream=0,
            library=self._runtime_state_library,
            runtime=runtime,
        )
        # Whole-step capture uses a newly-created nonblocking stream. A packed
        # width transition leaves each row's authoritative token on the host,
        # so make the default-stream seed visible before that stream captures.
        runtime.device_synchronize()

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
        if self.host_token_embedding_enabled:
            self._copy_token_embeddings_to_device(
                np.asarray([int(token_id)], dtype=np.int64),
                self._hidden_a.ptr,
                rows=1,
                token_ids_device_ptr=self._token_buf.ptr,
                stream=stream,
            )
        else:
            self._set_token_embedding_from_ptr(self._token_buf.ptr, stream=stream)

    def _set_token_embedding_from_ptr(self, token_id_ptr: int, *, stream: int = 0) -> None:
        if self.runner is None or self._hidden_a is None:
            raise RuntimeError("GGUF resident session is closed")
        assert self.runner.weights is not None
        if self.host_token_embedding_enabled:
            raise RuntimeError("host token embedding cannot embed from a device token pointer")
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

    def _set_full_attention_position_device(
        self,
        position: int,
        *,
        stream: int = 0,
        scratch=None,
    ) -> None:
        if self.scratch is None:
            raise RuntimeError("GGUF resident session is closed")
        scratch = self.scratch if scratch is None else scratch
        if position < 0 or position >= scratch.max_positions:
            raise ValueError(f"GGUF resident full-attention position {position} exceeds cache capacity {scratch.max_positions}")
        scratch.position_host[0] = int(position)
        scratch.context_host[0] = int(position) + 1
        set_decode_position_i64(
            scratch.position_buf.ptr,
            scratch.context_buf.ptr,
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
            self._verify_hidden_f32_a,
            self._verify_hidden_f32_b,
        ):
            if buffer is not None:
                free(buffer, runtime=runtime)
        if self.runner is None:
            raise RuntimeError("GGUF resident session is closed")
        self._verify_hidden_seed_buf = malloc(
            rows * self.runner.hidden_size * DType.FP32.itemsize,
            runtime=runtime,
        )
        self._verify_hidden_f32_a = malloc(
            rows * self.runner.hidden_size * DType.FP32.itemsize,
            runtime=runtime,
        )
        self._verify_hidden_f32_b = malloc(
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

    def packed_workspace_nbytes(self) -> int:
        """Return owner-only packed workspace bytes retained by this session."""

        seen: set[int] = set()
        total = 0
        for workspace in (
            getattr(self, "_packed_ar_attention_workspace", None),
            self._packed_verify_scratch,
            self._packed_verify_state,
        ):
            if workspace is None:
                continue
            for buffer in workspace.buffers:
                if buffer is None or int(buffer.ptr) == 0 or int(buffer.ptr) in seen:
                    continue
                seen.add(int(buffer.ptr))
                total += int(buffer.nbytes)
        return total

    def release_idle_packed_workspace(self) -> int:
        """Release packed owner scratch after state scatter and graph teardown."""

        if bool(getattr(self, "_packed_decode_state_dirty", False)):
            raise RuntimeError("cannot release idle workspace with unflushed packed state")
        graphs = {
            id(graph): graph
            for graph in (
                *tuple(getattr(self, "_decode_graphs", ())),
                *tuple(getattr(self, "_device_kv_graph_handles", {}).values()),
            )
        }
        if any(not bool(getattr(graph, "closed", False)) for graph in graphs.values()):
            raise RuntimeError("cannot release idle workspace while a live graph still binds it")
        released_bytes = self.packed_workspace_nbytes()
        self._free_packed_verify_workspace(runtime=self.runtime or get_hip_runtime())
        return released_bytes

    def _free_packed_verify_workspace(self, *, runtime: HipRuntime) -> None:
        attention_workspace = getattr(self, "_packed_ar_attention_workspace", None)
        if attention_workspace is not None:
            for buffer in reversed(attention_workspace.buffers):
                if buffer is not None:
                    free(buffer, runtime=runtime)
        self._packed_ar_attention_workspace = None
        if self._packed_verify_scratch is not None:
            for buffer in reversed(self._packed_verify_scratch.buffers):
                if buffer is not None:
                    free(buffer, runtime=runtime)
        self._packed_verify_scratch = None
        if self._packed_verify_state is not None:
            for buffer in reversed(self._packed_verify_state.buffers):
                if buffer is not None:
                    free(buffer, runtime=runtime)
        self._packed_verify_state = None
        self._packed_verify_session_ids = ()
        self._packed_verify_max_written_positions = ()
        self._packed_decode_sessions = ()
        self._packed_decode_last_layout = None
        self._packed_decode_state_dirty = False
        self._packed_decode_session_ids = ()
        self._packed_decode_positions = ()

    def _ensure_packed_ar_attention_workspace(
        self,
        *,
        rows: int,
        max_context_len: int,
        runtime: HipRuntime,
    ) -> _GGUFPackedARAttentionWorkspace:
        if self.runner is None:
            raise RuntimeError("GGUF resident session is closed")
        rows = int(rows)
        max_context_len = int(max_context_len)
        required_splits = (max_context_len + 255) // 256
        workspace = getattr(self, "_packed_ar_attention_workspace", None)
        if (
            workspace is None
            or int(workspace.rows) < rows
            or int(workspace.num_splits) < required_splits
        ):
            if workspace is not None:
                for buffer in reversed(workspace.buffers):
                    free(buffer, runtime=runtime)
            workspace = _GGUFPackedARAttentionWorkspace.allocate(
                self.runner,
                rows=rows,
                max_context_len=max_context_len,
                runtime=runtime,
            )
            self._packed_ar_attention_workspace = workspace
        return workspace

    def _ensure_packed_verify_workspace(
        self,
        *,
        slot_count: int,
        rows: int,
        max_sequence_length: int,
        runtime: HipRuntime,
        stream: int = 0,
    ) -> tuple[_GGUFPackedTargetState, object]:
        if self.runner is None:
            raise RuntimeError("GGUF resident session is closed")
        slot_count = int(slot_count)
        rows = int(rows)
        max_sequence_length = int(max_sequence_length)
        if slot_count <= 0 or rows <= 0 or max_sequence_length <= 0:
            raise ValueError("packed verify workspace dimensions must be positive")
        kv_layout = getattr(self, "_device_kv_layout", None)
        if kv_layout is None and getattr(self.runner, "weights", None) is not None and self.scratch is not None:
            kv_layout = _qwen35_gguf_session_kv_chunk_layout(self)
            self._device_kv_layout = kv_layout
        state_ready = (
            self._packed_verify_state is not None
            and int(self._packed_verify_state.slot_count) >= slot_count
            and int(self._packed_verify_state.max_sequence_length) >= max_sequence_length
            and getattr(self._packed_verify_state, "kv_layout", kv_layout) == kv_layout
        )
        scratch_ready = (
            self._packed_verify_scratch is not None
            and int(self._packed_verify_scratch.rows) >= rows
            and int(self._packed_verify_scratch.max_positions) >= max_sequence_length
            and int(self._packed_verify_scratch.gdn_segment_capacity) >= slot_count
        )
        if not state_ready or not scratch_ready:
            if getattr(self, "_packed_decode_state_dirty", False):
                if not self.flush_packed_decode_state(stream=stream):
                    raise RuntimeError(
                        "cannot resize packed workspace before deferred state is scattered"
                    )
            self._free_packed_verify_workspace(runtime=runtime)
            self._packed_verify_state = _GGUFPackedTargetState.allocate(
                self.runner,
                slot_count=slot_count,
                max_sequence_length=max_sequence_length,
                runtime=runtime,
                kv_layout=kv_layout,
            )
            self._packed_verify_scratch = _GGUFFullAttentionPrefillScratch.allocate(
                self.runner,
                rows=rows,
                capacity=max_sequence_length,
                allocate_kv_cache=False,
                segments=slot_count,
                runtime=runtime,
            )
        if self._packed_verify_state is None or self._packed_verify_scratch is None:
            raise RuntimeError("packed verify workspace allocation failed")
        return self._packed_verify_state, self._packed_verify_scratch

    def _packed_full_kv_row_nbytes(self) -> int:
        if self.runner is None or self.runner.weights is None:
            raise RuntimeError("GGUF resident session is closed")
        cfg = self.runner.weights.config
        return int(cfg.head_count_kv) * int(cfg.key_length) * DType.BF16.itemsize

    def _packed_kv_copy_planes(
        self,
        source: object,
        destination: object,
        layer_id: int,
    ) -> tuple[tuple[object, object, int], ...]:
        if self.runner is None or self.runner.weights is None:
            raise RuntimeError("GGUF resident session is closed")
        layout = getattr(source, "kv_layout", None) or getattr(destination, "kv_layout", None)
        if layout is None:
            layout = getattr(self, "_device_kv_layout", None)
        if layout is None:
            cfg = self.runner.weights.config
            layout = Qwen35GGUFKVChunkLayout(
                storage_dtype=DType.BF16,
                storage_layout="uniform",
                scale_dtype=DType.FP16,
                scale_granularity="per_token_head",
                int8_kv_value_bf16=False,
                layer_storage_dtypes=tuple(
                    None if layer_type == LINEAR_ATTENTION else DType.BF16
                    for layer_type in cfg.layer_types
                ),
            )
        storage = layout.layer_storage_dtypes[int(layer_id)]
        if storage is None:
            raise ValueError(f"layer {layer_id} has no full-attention KV copy planes")
        cfg = self.runner.weights.config
        payload_elements = int(cfg.head_count_kv) * int(cfg.key_length)
        key_row_nbytes = payload_elements * (
            DType.INT8.itemsize if storage == DType.INT8_PER_TOKEN_HEAD else DType.BF16.itemsize
        )
        value_row_nbytes = payload_elements * (
            DType.BF16.itemsize
            if storage == DType.INT8_PER_TOKEN_HEAD and layout.int8_kv_value_bf16
            else (DType.INT8.itemsize if storage == DType.INT8_PER_TOKEN_HEAD else DType.BF16.itemsize)
        )
        plane_specs: list[tuple[str, int]] = [
            ("full_key_caches", key_row_nbytes),
            ("full_value_caches", value_row_nbytes),
        ]
        if int(layer_id) in frozenset(layout.bf16_mirror_layer_indices):
            mirror_row_nbytes = payload_elements * DType.BF16.itemsize
            plane_specs.extend(
                (
                    ("full_bf16_mirror_key_caches", mirror_row_nbytes),
                    ("full_bf16_mirror_value_caches", mirror_row_nbytes),
                )
            )
        if storage == DType.INT8_PER_TOKEN_HEAD:
            group_factor = 1
            if layout.scale_granularity == "block16":
                group_factor = int(cfg.key_length) // 16
            elif layout.scale_granularity == "hadamard_group32":
                group_factor = int(cfg.key_length) // 32
            scale_row_nbytes = int(cfg.head_count_kv) * group_factor * layout.scale_dtype.itemsize
            plane_specs.extend(
                (
                    ("full_k_scale_caches", scale_row_nbytes),
                    ("full_v_scale_caches", scale_row_nbytes),
                )
            )
        planes: list[tuple[object, object, int]] = []
        for field_name, row_nbytes in plane_specs:
            if field_name in {"full_key_caches", "full_value_caches"} and not hasattr(source, field_name):
                source_buffer = source.full_cache(int(layer_id))[
                    0 if field_name == "full_key_caches" else 1
                ]
            else:
                source_buffer = getattr(source, field_name)[int(layer_id)]
            if field_name in {"full_key_caches", "full_value_caches"} and not hasattr(destination, field_name):
                destination_buffer = destination.full_cache(int(layer_id))[
                    0 if field_name == "full_key_caches" else 1
                ]
            else:
                destination_buffer = getattr(destination, field_name)[int(layer_id)]
            if source_buffer is None or destination_buffer is None:
                raise RuntimeError(
                    f"GGUF packed KV copy plane {field_name} is missing at layer {layer_id}"
                )
            planes.append((source_buffer, destination_buffer, int(row_nbytes)))
        return tuple(planes)

    def _copy_packed_kv_rows(
        self,
        source: object,
        destination: object,
        layer_id: int,
        *,
        source_start: int,
        destination_start: int,
        rows: int,
        runtime: HipRuntime,
        stream: int,
    ) -> None:
        if int(rows) <= 0:
            return
        for source_buffer, destination_buffer, row_nbytes in self._packed_kv_copy_planes(
            source,
            destination,
            layer_id,
        ):
            runtime.memcpy_async(
                int(destination_buffer.ptr) + int(destination_start) * row_nbytes,
                int(source_buffer.ptr) + int(source_start) * row_nbytes,
                int(rows) * row_nbytes,
                HipMemcpyKind.DEVICE_TO_DEVICE,
                int(stream),
            )

    def _sync_packed_verify_initial_state(
        self,
        jobs: list[dict[str, object]],
        layout: _GGUFPackedVerifyLayout,
        packed_state: _GGUFPackedTargetState,
        *,
        runtime: HipRuntime,
        stream: int,
    ) -> None:
        if self.runner is None or self.runner.weights is None:
            raise RuntimeError("GGUF resident session is closed")
        session_ids = tuple(id(job["session"]) for job in jobs)
        if self._packed_verify_session_ids != session_ids:
            self._packed_verify_session_ids = session_ids
            self._packed_verify_max_written_positions = tuple(0 for _ in jobs)
        if len(self._packed_verify_max_written_positions) != len(jobs):
            self._packed_verify_max_written_positions = tuple(0 for _ in jobs)
        written_positions = list(self._packed_verify_max_written_positions)
        cfg = self.runner.weights.config
        for slot_index, job in enumerate(jobs):
            session = job["session"]
            if not isinstance(session, Qwen35GGUFResidentSession) or session.scratch is None:
                raise NotImplementedError("packed verifier requires resident GGUF sessions")
            if session.runner is not self.runner:
                raise NotImplementedError("packed verifier requires shared runner sessions")
            start_position = int(session.position)
            if start_position != int(layout.row_positions[int(layout.cu_seqlens[slot_index])]):
                raise NotImplementedError("packed verifier job start does not match session position")
            for layer_id, layer_type in enumerate(cfg.layer_types):
                if layer_type == LINEAR_ATTENTION:
                    src_conv = session.scratch.layer_conv_states[layer_id]
                    src_recurrent = session.scratch.layer_recurrent_states[layer_id]
                    if src_conv is None or src_recurrent is None:
                        raise RuntimeError(f"session layer {layer_id} missing linear state")
                    dst_conv, dst_recurrent = packed_state.linear_state_pair(layer_id)
                    runtime.memcpy_async(
                        dst_conv.ptr + slot_index * int(src_conv.nbytes),
                        src_conv.ptr,
                        int(src_conv.nbytes),
                        HipMemcpyKind.DEVICE_TO_DEVICE,
                        stream,
                    )
                    runtime.memcpy_async(
                        dst_recurrent.ptr + slot_index * int(src_recurrent.nbytes),
                        src_recurrent.ptr,
                        int(src_recurrent.nbytes),
                        HipMemcpyKind.DEVICE_TO_DEVICE,
                        stream,
                    )
                elif layer_type == FULL_ATTENTION:
                    if start_position <= int(written_positions[slot_index]):
                        continue
                    physical_base = slot_index * int(packed_state.blocks_per_slot) * int(packed_state.block_size)
                    for logical_start, physical_start, copy_rows in _gguf_device_kv_copy_segments(
                        session,
                        start_position=0,
                        rows=start_position,
                    ):
                        self._copy_packed_kv_rows(
                            session.scratch,
                            packed_state,
                            layer_id,
                            source_start=physical_start,
                            destination_start=physical_base + logical_start,
                            rows=copy_rows,
                            runtime=runtime,
                            stream=stream,
                        )
                else:
                    raise ValueError(f"unsupported GGUF layer type {layer_type!r}")
            written_positions[slot_index] = max(int(written_positions[slot_index]), start_position)
        self._packed_verify_max_written_positions = tuple(written_positions)

    def _sync_packed_decode_initial_state(
        self,
        sessions: tuple["Qwen35GGUFResidentSession | None", ...],
        layout: _GGUFPackedVerifyLayout,
        packed_state: _GGUFPackedTargetState,
        *,
        runtime: HipRuntime,
        stream: int,
    ) -> tuple[int, ...]:
        if self.runner is None or self.runner.weights is None:
            raise RuntimeError("GGUF resident session is closed")
        session_ids = tuple(0 if session is None else id(session) for session in sessions)
        prior_positions = self._packed_decode_positions
        can_reuse = (
            self._packed_decode_session_ids == session_ids
            and len(prior_positions) == len(sessions)
        )
        cfg = self.runner.weights.config
        imported_slot_indices: list[int] = []
        for slot_index, session in enumerate(sessions):
            if session is None:
                continue
            if session.scratch is None:
                raise RuntimeError("packed AR decode job session is closed")
            row_start = int(layout.cu_seqlens[slot_index])
            start_position = int(session.position)
            if start_position != int(layout.row_positions[row_start]):
                raise NotImplementedError("packed AR decode job start does not match session position")
            slot_is_current = can_reuse and int(prior_positions[slot_index]) == start_position
            if slot_is_current:
                continue
            imported_slot_indices.append(int(slot_index))
            for layer_id, layer_type in enumerate(cfg.layer_types):
                if layer_type == LINEAR_ATTENTION:
                    src_conv = session.scratch.layer_conv_states[layer_id]
                    src_recurrent = session.scratch.layer_recurrent_states[layer_id]
                    if src_conv is None or src_recurrent is None:
                        raise RuntimeError(f"session layer {layer_id} missing linear state")
                    dst_conv, dst_recurrent = packed_state.linear_state_pair(layer_id)
                    runtime.memcpy_async(
                        dst_conv.ptr + slot_index * int(src_conv.nbytes),
                        src_conv.ptr,
                        int(src_conv.nbytes),
                        HipMemcpyKind.DEVICE_TO_DEVICE,
                        stream,
                    )
                    runtime.memcpy_async(
                        dst_recurrent.ptr + slot_index * int(src_recurrent.nbytes),
                        src_recurrent.ptr,
                        int(src_recurrent.nbytes),
                        HipMemcpyKind.DEVICE_TO_DEVICE,
                        stream,
                    )
                elif layer_type == FULL_ATTENTION:
                    if start_position <= 0:
                        continue
                    physical_base = slot_index * int(packed_state.blocks_per_slot) * int(packed_state.block_size)
                    for logical_start, physical_start, copy_rows in _gguf_device_kv_copy_segments(
                        session,
                        start_position=0,
                        rows=start_position,
                    ):
                        self._copy_packed_kv_rows(
                            session.scratch,
                            packed_state,
                            layer_id,
                            source_start=physical_start,
                            destination_start=physical_base + logical_start,
                            rows=copy_rows,
                            runtime=runtime,
                            stream=stream,
                        )
                else:
                    raise ValueError(f"unsupported GGUF layer type {layer_type!r}")
        return tuple(imported_slot_indices)

    def _commit_packed_decode_linear_state_rows(
        self,
        layout: _GGUFPackedVerifyLayout,
        packed_state: _GGUFPackedTargetState,
        *,
        runtime: HipRuntime,
        stream: int,
    ) -> None:
        if self.runner is None or self.runner.weights is None or self.scratch is None:
            raise RuntimeError("GGUF resident session is closed")
        cfg = self.runner.weights.config
        used_fused_commit = False
        if self._fused_linear_state_commit_enabled():
            conv_sources: list[int] = []
            recurrent_sources: list[int] = []
            conv_dests: list[int] = []
            recurrent_dests: list[int] = []
            conv_row_nbytes = 0
            recurrent_row_nbytes = 0
            for layer_id, layer_type in enumerate(cfg.layer_types):
                if layer_type != LINEAR_ATTENTION:
                    continue
                src_pair = self._verify_linear_state_row_pair(layer_id)
                if src_pair is None:
                    raise RuntimeError(f"linear-state rows for layer {layer_id} were not captured")
                src_conv_rows, src_recurrent_rows = src_pair
                dst_conv, dst_recurrent = packed_state.linear_state_pair(layer_id)
                single_conv_state = self.scratch.layer_conv_states[layer_id]
                single_recurrent_state = self.scratch.layer_recurrent_states[layer_id]
                if single_conv_state is None or single_recurrent_state is None:
                    raise RuntimeError(f"session layer {layer_id} missing linear state")
                conv_nbytes = int(single_conv_state.nbytes)
                recurrent_nbytes = int(single_recurrent_state.nbytes)
                if conv_row_nbytes not in {0, conv_nbytes}:
                    conv_sources = []
                    break
                if recurrent_row_nbytes not in {0, recurrent_nbytes}:
                    conv_sources = []
                    break
                conv_row_nbytes = conv_nbytes
                recurrent_row_nbytes = recurrent_nbytes
                for slot_index in range(int(layout.slot_count)):
                    final_row = int(layout.cu_seqlens[slot_index + 1]) - 1
                    if final_row < int(layout.cu_seqlens[slot_index]):
                        raise RuntimeError("packed AR decode slot has no final row to commit")
                    conv_sources.append(int(src_conv_rows.ptr) + final_row * conv_nbytes)
                    recurrent_sources.append(int(src_recurrent_rows.ptr) + final_row * recurrent_nbytes)
                    conv_dests.append(int(dst_conv.ptr) + slot_index * conv_nbytes)
                    recurrent_dests.append(int(dst_recurrent.ptr) + slot_index * recurrent_nbytes)
            n_entries = len(conv_sources)
            if n_entries > 0 and n_entries == len(recurrent_sources):
                tables_ready = (
                    self._verify_linear_state_src_conv_table_buf is not None
                    and self._verify_linear_state_src_recurrent_table_buf is not None
                    and self._verify_linear_state_dst_conv_table_buf is not None
                    and self._verify_linear_state_dst_recurrent_table_buf is not None
                    and self._verify_linear_state_commit_row_i32_buf is not None
                    and int(self._verify_linear_state_layer_count) == n_entries
                    and int(self._verify_linear_state_conv_row_nbytes) == conv_row_nbytes
                    and int(self._verify_linear_state_recurrent_row_nbytes) == recurrent_row_nbytes
                )
                if not tables_ready:
                    table_nbytes = n_entries * np.dtype(np.uint64).itemsize
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
                    self._verify_linear_state_src_conv_host = np.zeros((n_entries,), dtype=np.uint64)
                    self._verify_linear_state_src_recurrent_host = np.zeros((n_entries,), dtype=np.uint64)
                    self._verify_linear_state_src_conv_cached = np.zeros((n_entries,), dtype=np.uint64)
                    self._verify_linear_state_src_recurrent_cached = np.zeros((n_entries,), dtype=np.uint64)
                    self._verify_linear_state_dst_conv_host = np.zeros((n_entries,), dtype=np.uint64)
                    self._verify_linear_state_dst_recurrent_host = np.zeros((n_entries,), dtype=np.uint64)
                    self._verify_linear_state_conv_row_nbytes = conv_row_nbytes
                    self._verify_linear_state_recurrent_row_nbytes = recurrent_row_nbytes
                    self._verify_linear_state_layer_count = n_entries
                    self._buffers = (*self._buffers, *new_buffers)
                assert self._verify_linear_state_src_conv_host is not None
                assert self._verify_linear_state_src_recurrent_host is not None
                assert self._verify_linear_state_src_conv_cached is not None
                assert self._verify_linear_state_src_recurrent_cached is not None
                assert self._verify_linear_state_dst_conv_host is not None
                assert self._verify_linear_state_dst_recurrent_host is not None
                self._verify_linear_state_src_conv_host[:] = np.asarray(conv_sources, dtype=np.uint64)
                self._verify_linear_state_src_recurrent_host[:] = np.asarray(recurrent_sources, dtype=np.uint64)
                self._verify_linear_state_dst_conv_host[:] = np.asarray(conv_dests, dtype=np.uint64)
                self._verify_linear_state_dst_recurrent_host[:] = np.asarray(recurrent_dests, dtype=np.uint64)
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
                assert self._verify_linear_state_commit_row_i32_buf is not None
                commit_row = np.asarray([0], dtype=np.int32)
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
                    int(conv_row_nbytes),
                    self._verify_linear_state_src_recurrent_table_buf.ptr,
                    self._verify_linear_state_dst_recurrent_table_buf.ptr,
                    int(recurrent_row_nbytes),
                    self._verify_linear_state_commit_row_i32_buf.ptr,
                    int(n_entries),
                    stream=stream,
                    library=self._dflash_commit_library,
                    runtime=runtime,
                )
                used_fused_commit = True
        if used_fused_commit:
            return
        for layer_id, layer_type in enumerate(cfg.layer_types):
            if layer_type != LINEAR_ATTENTION:
                continue
            src_pair = self._verify_linear_state_row_pair(layer_id)
            if src_pair is None:
                raise RuntimeError(f"linear-state rows for layer {layer_id} were not captured")
            src_conv_rows, src_recurrent_rows = src_pair
            dst_conv, dst_recurrent = packed_state.linear_state_pair(layer_id)
            single_conv_state = self.scratch.layer_conv_states[layer_id]
            single_recurrent_state = self.scratch.layer_recurrent_states[layer_id]
            if single_conv_state is None or single_recurrent_state is None:
                raise RuntimeError(f"session layer {layer_id} missing linear state")
            conv_nbytes = int(single_conv_state.nbytes)
            recurrent_nbytes = int(single_recurrent_state.nbytes)
            for slot_index in range(int(layout.slot_count)):
                final_row = int(layout.cu_seqlens[slot_index + 1]) - 1
                if final_row < int(layout.cu_seqlens[slot_index]):
                    raise RuntimeError("packed AR decode slot has no final row to commit")
                runtime.memcpy_async(
                    dst_conv.ptr + slot_index * conv_nbytes,
                    src_conv_rows.ptr + final_row * conv_nbytes,
                    conv_nbytes,
                    HipMemcpyKind.DEVICE_TO_DEVICE,
                    stream,
                )
                runtime.memcpy_async(
                    dst_recurrent.ptr + slot_index * recurrent_nbytes,
                    src_recurrent_rows.ptr + final_row * recurrent_nbytes,
                    recurrent_nbytes,
                    HipMemcpyKind.DEVICE_TO_DEVICE,
                    stream,
                )

    def _scatter_packed_decode_state(
        self,
        sessions: tuple["Qwen35GGUFResidentSession | None", ...],
        layout: _GGUFPackedVerifyLayout,
        packed_state: _GGUFPackedTargetState,
        *,
        runtime: HipRuntime,
        stream: int,
        copy_full_kv: bool = False,
        copy_kv: bool = True,
    ) -> None:
        if self.runner is None or self.runner.weights is None:
            raise RuntimeError("GGUF resident session is closed")
        cfg = self.runner.weights.config
        for slot_index, session in enumerate(sessions):
            if session is None:
                continue
            if session.scratch is None:
                raise RuntimeError("packed AR decode job session is closed")
            row_start = int(layout.cu_seqlens[slot_index])
            row_end = int(layout.cu_seqlens[slot_index + 1])
            slot_rows = row_end - row_start
            start_position = int(layout.row_positions[row_start])
            end_position = start_position + slot_rows
            for layer_id, layer_type in enumerate(cfg.layer_types):
                if layer_type == LINEAR_ATTENTION:
                    src_conv, src_recurrent = packed_state.linear_state_pair(layer_id)
                    dst_conv = session.scratch.layer_conv_states[layer_id]
                    dst_recurrent = session.scratch.layer_recurrent_states[layer_id]
                    if dst_conv is None or dst_recurrent is None:
                        raise RuntimeError(f"session layer {layer_id} missing linear state")
                    runtime.memcpy_async(
                        dst_conv.ptr,
                        src_conv.ptr + slot_index * int(dst_conv.nbytes),
                        int(dst_conv.nbytes),
                        HipMemcpyKind.DEVICE_TO_DEVICE,
                        stream,
                    )
                    runtime.memcpy_async(
                        dst_recurrent.ptr,
                        src_recurrent.ptr + slot_index * int(dst_recurrent.nbytes),
                        int(dst_recurrent.nbytes),
                        HipMemcpyKind.DEVICE_TO_DEVICE,
                        stream,
                    )
                elif layer_type == FULL_ATTENTION and copy_kv:
                    physical_base = slot_index * int(packed_state.blocks_per_slot) * int(packed_state.block_size)
                    copy_start = 0 if copy_full_kv else start_position
                    copy_rows = end_position if copy_full_kv else slot_rows
                    for logical_start, physical_start, segment_rows in _gguf_device_kv_copy_segments(
                        session,
                        start_position=copy_start,
                        rows=copy_rows,
                    ):
                        self._copy_packed_kv_rows(
                            packed_state,
                            session.scratch,
                            layer_id,
                            source_start=physical_base + logical_start,
                            destination_start=physical_start,
                            rows=segment_rows,
                            runtime=runtime,
                            stream=stream,
                        )
                elif layer_type == FULL_ATTENTION:
                    continue
                else:
                    raise ValueError(f"unsupported GGUF layer type {layer_type!r}")
            session._position = end_position
            session.scratch.position_host[0] = end_position
            session.scratch.context_host[0] = end_position + 1
            set_decode_position_i64(
                session.scratch.position_buf.ptr,
                session.scratch.context_buf.ptr,
                end_position,
                stream=stream,
                library=session._runtime_state_library,
                runtime=runtime,
            )

    def _advance_packed_decode_session_cursors(
        self,
        sessions: tuple["Qwen35GGUFResidentSession | None", ...],
        layout: _GGUFPackedVerifyLayout,
    ) -> None:
        for slot_index, session in enumerate(sessions):
            if session is None:
                continue
            if session.scratch is None:
                raise RuntimeError("packed AR decode job session is closed")
            row_start = int(layout.cu_seqlens[slot_index])
            row_end = int(layout.cu_seqlens[slot_index + 1])
            slot_rows = row_end - row_start
            start_position = int(layout.row_positions[row_start])
            end_position = start_position + slot_rows
            session._position = end_position
            session.scratch.position_host[0] = end_position
            session.scratch.context_host[0] = end_position + 1

    def discard_packed_decode_state(self) -> bool:
        """Forget deferred packed-AR state after every bound session is terminal."""

        if not self._packed_decode_state_dirty:
            return False
        self._packed_decode_sessions = ()
        self._packed_decode_last_layout = None
        self._packed_decode_state_dirty = False
        self._packed_decode_session_ids = ()
        self._packed_decode_positions = ()
        return True

    def flush_packed_decode_state(self, *, stream: int = 0) -> bool:
        """Scatter deferred packed-AR state back into the slot sessions.

        ``step_batch_native(..., scatter_state=False)`` keeps the packed
        multi-slot state canonical across decode cycles. Call this before any
        scalar/session-local fallback needs the per-session scratch state.
        """

        if not self._packed_decode_state_dirty:
            return False
        if self._packed_verify_state is None or self._packed_decode_last_layout is None:
            return False
        if not self._packed_decode_sessions:
            return False
        runtime = self.runtime or get_hip_runtime()
        self._scatter_packed_decode_state(
            self._packed_decode_sessions,
            self._packed_decode_last_layout,
            self._packed_verify_state,
            runtime=runtime,
            stream=stream,
            copy_full_kv=True,
        )
        if stream:
            runtime.stream_synchronize(stream)
        else:
            runtime.device_synchronize()
        self._packed_decode_positions = tuple(
            -1 if session is None else int(session.position)
            for session in self._packed_decode_sessions
        )
        self._packed_decode_state_dirty = False
        return True

    def _scatter_packed_verify_outputs(
        self,
        jobs: list[dict[str, object]],
        layout: _GGUFPackedVerifyLayout,
        packed_state: _GGUFPackedTargetState,
        hidden_host: np.ndarray,
        token_host: np.ndarray,
        *,
        runtime: HipRuntime,
        stream: int,
        linear_state_rows_captured: bool = True,
        final_linear_state_committed: bool = False,
        defer_state_scatter: bool = False,
    ) -> list[Qwen35GGUFBlockVerifyResult]:
        if self.runner is None or self.runner.weights is None:
            raise RuntimeError("GGUF resident session is closed")
        cfg = self.runner.weights.config
        row_nbytes = self._packed_full_kv_row_nbytes()
        hidden_row_nbytes = self.runner.hidden_size * DType.FP32.itemsize
        written_positions = list(self._packed_verify_max_written_positions)
        results: list[Qwen35GGUFBlockVerifyResult] = []
        for slot_index, job in enumerate(jobs):
            session = job["session"]
            if not isinstance(session, Qwen35GGUFResidentSession) or session.scratch is None:
                raise RuntimeError("packed verifier job lost resident session")
            row_start = int(layout.cu_seqlens[slot_index])
            row_end = int(layout.cu_seqlens[slot_index + 1])
            slot_rows = row_end - row_start
            start_position = int(layout.row_positions[row_start])
            end_position = start_position + slot_rows
            deferred_state = None
            if defer_state_scatter:
                deferred_state = _GGUFPackedVerifyDeferredState(
                    owner=self,
                    packed_state=packed_state,
                    slot_index=int(slot_index),
                    row_start=row_start,
                    row_end=row_end,
                    start_position=start_position,
                    end_position=end_position,
                )
            session._ensure_verify_block_buffers(slot_rows, runtime=runtime)
            if linear_state_rows_captured and not defer_state_scatter:
                session._ensure_verify_linear_state_row_buffers(slot_rows, runtime=runtime)
            if session is not self and not defer_state_scatter:
                if self._verify_hidden_seed_buf is None or session._verify_hidden_seed_buf is None:
                    raise RuntimeError("packed verifier hidden buffers are closed")
                runtime.memcpy_async(
                    session._verify_hidden_seed_buf.ptr,
                    self._verify_hidden_seed_buf.ptr + row_start * hidden_row_nbytes,
                    slot_rows * hidden_row_nbytes,
                    HipMemcpyKind.DEVICE_TO_DEVICE,
                    stream,
                )
            for layer_id, layer_type in enumerate(cfg.layer_types):
                if layer_type == LINEAR_ATTENTION:
                    conv_state, recurrent_state = session.scratch.layer_conv_states[layer_id], session.scratch.layer_recurrent_states[layer_id]
                    if conv_state is None or recurrent_state is None:
                        raise RuntimeError(f"session layer {layer_id} missing linear state")
                    if defer_state_scatter:
                        if not linear_state_rows_captured:
                            raise RuntimeError("deferred packed verifier scatter requires captured linear rows")
                    elif linear_state_rows_captured:
                        src_pair = self._verify_linear_state_row_pair(layer_id)
                        dst_pair = session._verify_linear_state_row_pair(layer_id)
                        if src_pair is None or dst_pair is None:
                            raise RuntimeError(f"linear-state rows for layer {layer_id} were not captured")
                        src_conv_rows, src_recurrent_rows = src_pair
                        dst_conv_rows, dst_recurrent_rows = dst_pair
                        if session is not self:
                            runtime.memcpy_async(
                                dst_conv_rows.ptr,
                                src_conv_rows.ptr + row_start * int(conv_state.nbytes),
                                slot_rows * int(conv_state.nbytes),
                                HipMemcpyKind.DEVICE_TO_DEVICE,
                                stream,
                            )
                            runtime.memcpy_async(
                                dst_recurrent_rows.ptr,
                                src_recurrent_rows.ptr + row_start * int(recurrent_state.nbytes),
                                slot_rows * int(recurrent_state.nbytes),
                                HipMemcpyKind.DEVICE_TO_DEVICE,
                                stream,
                            )
                    elif final_linear_state_committed:
                        packed_conv, packed_recurrent = packed_state.linear_state_pair(layer_id)
                        runtime.memcpy_async(
                            conv_state.ptr,
                            packed_conv.ptr + slot_index * int(conv_state.nbytes),
                            int(conv_state.nbytes),
                            HipMemcpyKind.DEVICE_TO_DEVICE,
                            stream,
                        )
                        runtime.memcpy_async(
                            recurrent_state.ptr,
                            packed_recurrent.ptr + slot_index * int(recurrent_state.nbytes),
                            int(recurrent_state.nbytes),
                            HipMemcpyKind.DEVICE_TO_DEVICE,
                            stream,
                        )
                    else:
                        raise RuntimeError("packed verifier produced neither captured nor final linear state")
                elif layer_type == FULL_ATTENTION:
                    if defer_state_scatter:
                        continue
                    src_key, src_value = packed_state.full_cache(layer_id)
                    dst_key, dst_value = session.scratch.full_cache(layer_id)
                    physical_base = slot_index * int(packed_state.blocks_per_slot) * int(packed_state.block_size)
                    nbytes = slot_rows * row_nbytes
                    runtime.memcpy_async(
                        dst_key.ptr + start_position * row_nbytes,
                        src_key.ptr + (physical_base + start_position) * row_nbytes,
                        nbytes,
                        HipMemcpyKind.DEVICE_TO_DEVICE,
                        stream,
                    )
                    runtime.memcpy_async(
                        dst_value.ptr + start_position * row_nbytes,
                        src_value.ptr + (physical_base + start_position) * row_nbytes,
                        nbytes,
                        HipMemcpyKind.DEVICE_TO_DEVICE,
                        stream,
                    )
                else:
                    raise ValueError(f"unsupported GGUF layer type {layer_type!r}")
            if not defer_state_scatter:
                session._verify_hidden_seed_rows_populated = slot_rows
            if final_linear_state_committed and not defer_state_scatter:
                if self._verify_hidden_seed_buf is None:
                    raise RuntimeError("packed verifier hidden buffers are closed")
                runtime.memcpy_async(
                    session.scratch.hidden_seed_fp32.ptr,
                    self._verify_hidden_seed_buf.ptr + (row_end - 1) * hidden_row_nbytes,
                    hidden_row_nbytes,
                    HipMemcpyKind.DEVICE_TO_DEVICE,
                    stream,
                )
            if not defer_state_scatter:
                session._position = end_position
                session.scratch.position_host[0] = end_position
                session.scratch.context_host[0] = end_position + 1
                set_decode_position_i64(
                    session.scratch.position_buf.ptr,
                    session.scratch.context_buf.ptr,
                    end_position,
                    library=session._runtime_state_library,
                    runtime=runtime,
                )
                session._hidden_seed_fp32_populated = True
                written_positions[slot_index] = max(int(written_positions[slot_index]), end_position)
            results.append(
                Qwen35GGUFBlockVerifyResult(
                    input_token_ids=[int(token) for token in layout.input_token_ids[row_start:row_end].tolist()],
                    token_ids=[int(token) for token in token_host[row_start:row_end].tolist()],
                    hidden_seeds=np.ascontiguousarray(hidden_host[row_start:row_end], dtype=np.float32),
                    start_position=start_position,
                    linear_state_rows_captured=bool(linear_state_rows_captured),
                    final_linear_state_committed=bool(final_linear_state_committed),
                    deferred_packed_state=deferred_state,
                )
            )
        self._packed_verify_max_written_positions = tuple(written_positions)
        return results

    def _commit_deferred_packed_verify_state(
        self,
        deferred_state: object,
        destination_session: "Qwen35GGUFResidentSession",
        *,
        commit_row_index: int,
        position: int,
        hidden_rows: int,
        stream: int = 0,
    ) -> None:
        """Commit one accepted packed-verifier row after deferred scatter."""

        if self.runner is None or self.runner.weights is None:
            raise RuntimeError("GGUF resident session is closed")
        if getattr(deferred_state, "owner", None) is not self:
            raise RuntimeError("deferred packed verifier state belongs to a different owner")
        if not isinstance(destination_session, Qwen35GGUFResidentSession) or destination_session.scratch is None:
            raise RuntimeError("deferred packed verifier destination session is closed")
        if destination_session.runner is not self.runner:
            raise RuntimeError("deferred packed verifier destination must share the owner runner")
        if self._verify_hidden_seed_buf is None:
            raise RuntimeError("packed verifier hidden buffers are closed")
        packed_state = getattr(deferred_state, "packed_state", None)
        if not isinstance(packed_state, _GGUFPackedTargetState):
            raise RuntimeError("deferred packed verifier state is invalid")
        row_start = int(getattr(deferred_state, "row_start"))
        row_end = int(getattr(deferred_state, "row_end"))
        slot_index = int(getattr(deferred_state, "slot_index"))
        start_position = int(getattr(deferred_state, "start_position"))
        slot_rows = row_end - row_start
        row = int(commit_row_index)
        if row < 0 or row >= slot_rows:
            raise ValueError("commit_row_index is outside deferred packed verifier rows")
        end_position = int(position)
        consumed_rows = end_position - start_position
        if consumed_rows <= 0 or consumed_rows > slot_rows:
            raise ValueError("deferred packed verifier commit position is outside slot rows")
        hidden_count = int(hidden_rows)
        if hidden_count <= 0 or hidden_count > slot_rows:
            raise ValueError("hidden_rows is outside deferred packed verifier rows")
        if row >= hidden_count:
            raise ValueError("hidden_rows must include the committed verifier row")

        runtime = self.runtime or get_hip_runtime()
        cfg = self.runner.weights.config
        destination_session._ensure_verify_block_buffers(hidden_count, runtime=runtime)
        if destination_session._verify_hidden_seed_buf is None:
            raise RuntimeError("destination verifier hidden buffers are closed")
        hidden_row_nbytes = self.runner.hidden_size * DType.FP32.itemsize
        runtime.memcpy_async(
            destination_session._verify_hidden_seed_buf.ptr,
            self._verify_hidden_seed_buf.ptr + row_start * hidden_row_nbytes,
            hidden_count * hidden_row_nbytes,
            HipMemcpyKind.DEVICE_TO_DEVICE,
            stream,
        )
        runtime.memcpy_async(
            destination_session.scratch.hidden_seed_fp32.ptr,
            self._verify_hidden_seed_buf.ptr + (row_start + row) * hidden_row_nbytes,
            hidden_row_nbytes,
            HipMemcpyKind.DEVICE_TO_DEVICE,
            stream,
        )

        full_row_nbytes = self._packed_full_kv_row_nbytes()
        for layer_id, layer_type in enumerate(cfg.layer_types):
            if layer_type == LINEAR_ATTENTION:
                src_pair = self._verify_linear_state_row_pair(layer_id)
                if src_pair is None:
                    raise RuntimeError(f"linear-state rows for layer {layer_id} were not captured")
                src_conv_rows, src_recurrent_rows = src_pair
                dst_conv = destination_session.scratch.layer_conv_states[layer_id]
                dst_recurrent = destination_session.scratch.layer_recurrent_states[layer_id]
                if dst_conv is None or dst_recurrent is None:
                    raise RuntimeError(f"destination layer {layer_id} missing linear state")
                runtime.memcpy_async(
                    dst_conv.ptr,
                    src_conv_rows.ptr + (row_start + row) * int(dst_conv.nbytes),
                    int(dst_conv.nbytes),
                    HipMemcpyKind.DEVICE_TO_DEVICE,
                    stream,
                )
                runtime.memcpy_async(
                    dst_recurrent.ptr,
                    src_recurrent_rows.ptr + (row_start + row) * int(dst_recurrent.nbytes),
                    int(dst_recurrent.nbytes),
                    HipMemcpyKind.DEVICE_TO_DEVICE,
                    stream,
                )
            elif layer_type == FULL_ATTENTION:
                src_key, src_value = packed_state.full_cache(layer_id)
                dst_key, dst_value = destination_session.scratch.full_cache(layer_id)
                physical_base = slot_index * int(packed_state.blocks_per_slot) * int(packed_state.block_size)
                nbytes = consumed_rows * full_row_nbytes
                runtime.memcpy_async(
                    dst_key.ptr + start_position * full_row_nbytes,
                    src_key.ptr + (physical_base + start_position) * full_row_nbytes,
                    nbytes,
                    HipMemcpyKind.DEVICE_TO_DEVICE,
                    stream,
                )
                runtime.memcpy_async(
                    dst_value.ptr + start_position * full_row_nbytes,
                    src_value.ptr + (physical_base + start_position) * full_row_nbytes,
                    nbytes,
                    HipMemcpyKind.DEVICE_TO_DEVICE,
                    stream,
                )
            else:
                raise ValueError(f"unsupported GGUF layer type {layer_type!r}")

        destination_session._verify_hidden_seed_rows_populated = hidden_count
        destination_session._position = end_position
        destination_session.scratch.position_host[0] = end_position
        destination_session.scratch.context_host[0] = end_position + 1
        set_decode_position_i64(
            destination_session.scratch.position_buf.ptr,
            destination_session.scratch.context_buf.ptr,
            end_position,
            library=destination_session._runtime_state_library,
            runtime=runtime,
        )
        destination_session._hidden_seed_fp32_populated = True
        if slot_index < len(self._packed_verify_max_written_positions):
            written_positions = list(self._packed_verify_max_written_positions)
            written_positions[slot_index] = max(int(written_positions[slot_index]), end_position)
            self._packed_verify_max_written_positions = tuple(written_positions)

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

    def _verify_lm_head_rowtile_chunked(
        self, hidden_ptr: int, out_ptr: int, rows: int, *, stream: int = 0, runtime=None
    ) -> bool:
        rows = int(rows)
        if rows <= 0:
            raise ValueError("rows must be positive")
        if rows <= 6:
            return self._verify_lm_head_rowtile(
                hidden_ptr,
                out_ptr,
                rows,
                stream=stream,
                runtime=runtime,
            )
        if self.runner is None:
            raise RuntimeError("GGUF resident session is closed")
        hidden_row_nbytes = int(self.runner.hidden_size) * DType.BF16.itemsize
        logits_row_nbytes = int(self.runner.vocab_size) * DType.FP32.itemsize
        row_offset = 0
        max_chunk_raw = os.environ.get("HIPENGINE_GGUF_Q6_LM_HEAD_MAX_CHUNK", "")
        max_chunk = (
            int(max_chunk_raw)
            if max_chunk_raw
            else int(
                backend_package_capability(
                    self.runner.backend,
                    "GGUF_Q6_LM_HEAD_MAX_CHUNK",
                    6,
                )
            )
        )
        if max_chunk < 2 or max_chunk > 6:
            raise ValueError("HIPENGINE_GGUF_Q6_LM_HEAD_MAX_CHUNK must be in [2, 6]")
        for chunk_rows in _small_b_rowtile_chunks(rows, max_chunk=max_chunk):
            if int(chunk_rows) < 2:
                return False
            handled = self._verify_lm_head_rowtile(
                hidden_ptr + row_offset * hidden_row_nbytes,
                out_ptr + row_offset * logits_row_nbytes,
                int(chunk_rows),
                stream=stream,
                runtime=runtime,
            )
            if not handled:
                return False
            row_offset += int(chunk_rows)
        return True

    def _verify_lm_head_q6_top1_dp4a(
        self,
        hidden_ptr: int,
        rows: int,
        *,
        activation_dtype: str = GGUF_ACTIVATION_BF16,
        stream: int = 0,
        runtime=None,
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
        if activation_dtype == GGUF_ACTIVATION_F32:
            gguf_q4_k_quantize_f32_q8_1(
                hidden_ptr,
                self._verify_lm_q8_1.ptr,
                rows,
                self.runner.hidden_size,
                stream=stream,
                runtime=runtime,
            )
        elif activation_dtype == GGUF_ACTIVATION_BF16:
            gguf_q4_k_quantize_bf16_q8_1(
                hidden_ptr,
                self._verify_lm_q8_1.ptr,
                rows,
                self.runner.hidden_size,
                stream=stream,
                runtime=runtime,
            )
        else:
            raise ValueError(f"unsupported verifier lm-head activation dtype: {activation_dtype!r}")
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

    def _enqueue_target_block_rows_from_hidden(
        self,
        hidden_ptr: int,
        rows: int,
        *,
        activation_dtype: str = GGUF_ACTIVATION_BF16,
        stream: int = 0,
        require_logits: bool = False,
    ) -> None:
        """Enqueue packed lm-head and sampler work without host synchronization."""

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
        direct_top1 = False
        if not require_logits:
            direct_top1 = self._verify_lm_head_q6_top1_dp4a(
                hidden_ptr,
                rows,
                activation_dtype=activation_dtype,
                stream=stream,
                runtime=runtime,
            )
        if direct_top1:
            self._last_packed_lm_head_decode_path = "direct_top1_rows"
            self._last_packed_sampler_decode_path = "fused_top1_i32_rows"
            return
        if activation_dtype != GGUF_ACTIVATION_BF16:
            raise ValueError(
                "non-dp4a verifier lm-head fallback expects BF16 hidden rows, "
                f"got {activation_dtype!r}"
            )
        rowtile = self._verify_lm_head_rowtile_chunked(
            hidden_ptr, self._verify_logits_buf.ptr, rows, stream=stream, runtime=runtime
        )
        if not rowtile:
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
        self._last_packed_lm_head_decode_path = (
            "q6_rowtile_f32_logits" if rowtile else "row_linear_f32_logits"
        )
        self._last_packed_sampler_decode_path = "argmax_i32_rows"
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

    def _read_target_block_row_tokens(self, rows: int, *, stream: int = 0) -> np.ndarray:
        """Synchronize and read one sampled int32 token per packed row."""

        runtime = self.runtime or get_hip_runtime()
        rows = int(rows)
        if self._verify_lm_out_indices_i32 is None:
            raise RuntimeError("GGUF verifier lm-head token buffer is closed")
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

    def _sample_target_block_rows_from_hidden(
        self,
        hidden_ptr: int,
        rows: int,
        *,
        activation_dtype: str = GGUF_ACTIVATION_BF16,
        stream: int = 0,
    ) -> np.ndarray:
        self._enqueue_target_block_rows_from_hidden(
            hidden_ptr,
            rows,
            activation_dtype=activation_dtype,
            stream=stream,
        )
        return self._read_target_block_row_tokens(rows, stream=stream)

    def sample_native_from_last_logits(
        self,
        params: object,
        state: object,
        *,
        stream: int = 0,
    ):
        """Select from the latest c1 FP32 logits without full-vocab readback."""

        if self._logits_buf is None:
            raise RuntimeError("GGUF resident logits buffer is closed")
        if self._lm_out_index is None or self._lm_out_value is None:
            raise RuntimeError("GGUF resident lm-head outputs are closed")
        workspace = self._native_sampler()
        return workspace.sample(
            self._logits_buf.ptr,
            params,
            state,
            out_index_i64_ptr=self._lm_out_index.ptr,
            out_value_f32_ptr=self._lm_out_value.ptr,
            stream=int(stream),
        )

    def sample_native_from_packed_logits_rows(
        self,
        physical_rows: list[int] | tuple[int, ...],
        params_rows: list[object] | tuple[object, ...],
        states: list[object] | tuple[object, ...],
        *,
        stream: int = 0,
    ):
        """Select a contiguous compatible packed row group in one launch."""

        row_tuple = tuple(int(row) for row in physical_rows)
        params_tuple = tuple(params_rows)
        state_tuple = tuple(states)
        if not row_tuple:
            raise ValueError("packed native sampler rows must be non-empty")
        if len(row_tuple) != len(params_tuple) or len(row_tuple) != len(state_tuple):
            raise ValueError("packed native sampler rows, params, and states must align")
        if row_tuple != tuple(range(row_tuple[0], row_tuple[0] + len(row_tuple))):
            raise NotImplementedError("packed native sampler rows must be contiguous")
        if row_tuple[0] < 0 or row_tuple[-1] >= int(self._verify_lm_rows_capacity):
            raise ValueError(
                f"packed native sampler rows {row_tuple!r} exceed capacity "
                f"{self._verify_lm_rows_capacity}"
            )
        if self._verify_logits_buf is None:
            raise RuntimeError("GGUF packed logits buffer is closed")
        logits_ptr = self._verify_logits_buf.ptr + (
            row_tuple[0] * int(self.runner.vocab_size) * DType.FP32.itemsize
        )
        samples = self._native_sampler().sample_rows(
            logits_ptr,
            params_tuple,
            state_tuple,
            stream=int(stream),
        )
        self._last_packed_sampler_decode_path = "native_gpu_sampler_rows"
        if self.last_packed_execution_manifest:
            self.last_packed_execution_manifest["sampler_decode_path"] = (
                "native_gpu_sampler_rows"
            )
            self.last_packed_execution_manifest["native_sampler_rows"] = len(
                row_tuple
            )
        if self.last_packed_prefill_plan:
            self.last_packed_prefill_plan["sampler_path"] = (
                "native_gpu_sampler_rows"
            )
            self.last_packed_prefill_plan["native_sampler_rows"] = len(row_tuple)
        return samples

    def sample_native_from_packed_logits(
        self,
        physical_row: int,
        params: object,
        state: object,
        *,
        output_session: "Qwen35GGUFResidentSession",
        stream: int = 0,
    ):
        """Select one active physical row from the latest packed FP32 logits."""

        row = int(physical_row)
        if row < 0 or row >= int(self._verify_lm_rows_capacity):
            raise ValueError(
                f"packed native sampler row {row} exceeds capacity "
                f"{self._verify_lm_rows_capacity}"
            )
        if self._verify_logits_buf is None:
            raise RuntimeError("GGUF packed logits buffer is closed")
        if output_session._lm_out_index is None or output_session._lm_out_value is None:
            raise RuntimeError("GGUF packed output session lm-head buffers are closed")
        logits_ptr = self._verify_logits_buf.ptr + (
            row * int(self.runner.vocab_size) * DType.FP32.itemsize
        )
        workspace = self._native_sampler()
        sample = workspace.sample(
            logits_ptr,
            params,
            state,
            out_index_i64_ptr=output_session._lm_out_index.ptr,
            out_value_f32_ptr=output_session._lm_out_value.ptr,
            stream=int(stream),
        )
        self._last_packed_sampler_decode_path = "native_gpu_sampler_row"
        if self.last_packed_execution_manifest:
            self.last_packed_execution_manifest["sampler_decode_path"] = (
                "native_gpu_sampler_row"
            )
            self.last_packed_execution_manifest["native_sampler_rows"] = 1
        if self.last_packed_prefill_plan:
            self.last_packed_prefill_plan["sampler_path"] = (
                "native_gpu_sampler_row"
            )
            self.last_packed_prefill_plan["native_sampler_rows"] = 1
        return sample

    def _native_sampler(self) -> NativeSamplerWorkspace:
        workspace = self._native_sampler_workspace
        if workspace is not None and not workspace.closed:
            return workspace
        if self.runner is None or self._lm_head_library is None:
            raise RuntimeError("GGUF resident session is closed")
        build_kwargs = {
            "load": True,
            "compiler_version": self.compiler_version,
            "require_cached": self.require_cached_build,
        }
        with hip_target_arch_environment(self.runner.target_arch):
            sampler_library = build_sampler(**build_kwargs)
        workspace = NativeSamplerWorkspace(
            runtime=self.runtime or get_hip_runtime(),
            vocab_size=self.runner.vocab_size,
            sampler_library=sampler_library,
        )
        self._native_sampler_workspace = workspace
        return workspace

    def _sample_from_hidden(
        self,
        hidden_ptr: int,
        *,
        return_logits: bool = True,
        stream: int = 0,
    ) -> Qwen35GGUFNextTokenProbeResult:
        self._sample_device_from_hidden(hidden_ptr, stream=stream)
        runtime = self.runtime or get_hip_runtime()
        if stream:
            runtime.stream_synchronize(stream)
        else:
            runtime.device_synchronize()
        return self._read_sample(return_logits=return_logits)

    def _read_sample(self, *, return_logits: bool = True) -> Qwen35GGUFNextTokenProbeResult:
        if self.runner is None or self._logits_buf is None or self._logits_host is None:
            raise RuntimeError("GGUF resident session is closed")
        if self._lm_out_index is None or self._lm_out_value is None:
            raise RuntimeError("GGUF resident lm-head buffers are closed")
        runtime = self.runtime or get_hip_runtime()
        index_host = np.empty((1,), dtype=np.int64)
        copy_device_to_host(
            host_array_ptr(index_host),
            self._lm_out_index,
            index_host.nbytes,
            runtime=runtime,
        )
        logits = np.empty((0,), dtype=np.float32)
        logit = 0.0
        if return_logits:
            value_host = np.empty((1,), dtype=np.float32)
            copy_device_to_host(
                host_array_ptr(value_host),
                self._lm_out_value,
                value_host.nbytes,
                runtime=runtime,
            )
            logit = float(value_host[0])
            copy_device_to_host(
                host_array_ptr(self._logits_host),
                self._logits_buf,
                self._logits_host.nbytes,
                runtime=runtime,
            )
            if not np.all(np.isfinite(self._logits_host)):
                raise FloatingPointError("GGUF resident lm-head logits contain NaN or Inf")
            logits = self._logits_host.copy()
        token_id = int(index_host[0])
        return Qwen35GGUFNextTokenProbeResult(
            token_id=token_id,
            logit=logit,
            logits=logits,
        )

    def capture_native_spec_target_graph(
        self,
        input_token_ids: list[int] | tuple[int, ...],
        *,
        cycle_id: int = 0,
        transaction_id: int = 0,
        request_id: int = 0,
        bulk_attention_mode: str = "bulk",
        use_wmma_prefill: bool = False,
        capture_linear_state_rows: bool = False,
        defer_linear_state_commit: bool = False,
        device_accept_commit: bool = False,
    ):
        """Capture a reusable B1/B2 N1 or N2 native target graph."""

        from hipengine.runtime.gguf_native_spec_cycle import (
            capture_qwen35_gguf_native_b2_target_graph,
        )

        return capture_qwen35_gguf_native_b2_target_graph(
            self,
            input_token_ids,
            cycle_id=int(cycle_id),
            transaction_id=int(transaction_id),
            request_id=int(request_id),
            bulk_attention_mode=bulk_attention_mode,
            use_wmma_prefill=bool(use_wmma_prefill),
            capture_linear_state_rows=bool(capture_linear_state_rows),
            defer_linear_state_commit=bool(defer_linear_state_commit),
            device_accept_commit=bool(device_accept_commit),
        )

    def verify_target_block_native_cycle(
        self,
        input_token_ids: list[int] | tuple[int, ...],
        *,
        fallback: bool = True,
        cycle_id: int = 0,
        transaction_id: int = 0,
        request_id: int = 0,
        bulk_attention_mode: str = "bulk",
        use_wmma_prefill: bool = False,
        capture_linear_state_rows: bool = False,
        capture_lm_head_logits: bool = False,
        record_stage_timings: bool = False,
        sync_stage_timings: bool = False,
        defer_linear_state_commit: bool = False,
        device_accept_commit: bool = False,
        remaining_decode: int | None = None,
    ):
        """Run reusable B1/B2 N1/N2 submission or preserve the eager fallback."""

        from hipengine.runtime.gguf_native_spec_cycle import (
            verify_qwen35_gguf_native_b2_target,
        )

        return verify_qwen35_gguf_native_b2_target(
            self,
            input_token_ids,
            fallback=bool(fallback),
            cycle_id=int(cycle_id),
            transaction_id=int(transaction_id),
            request_id=int(request_id),
            bulk_attention_mode=bulk_attention_mode,
            use_wmma_prefill=bool(use_wmma_prefill),
            capture_linear_state_rows=bool(capture_linear_state_rows),
            capture_lm_head_logits=bool(capture_lm_head_logits),
            record_stage_timings=bool(record_stage_timings),
            sync_stage_timings=bool(sync_stage_timings),
            defer_linear_state_commit=bool(defer_linear_state_commit),
            device_accept_commit=bool(device_accept_commit),
            remaining_decode=(None if remaining_decode is None else int(remaining_decode)),
        )

    def run_native_spec_mtp_cycle(
        self,
        resident_draft,
        resident_context,
        *,
        root_token: int,
        root_position: int,
        candidate_budget: int,
        remaining_decode: int,
        rope_cos: np.ndarray,
        rope_sin: np.ndarray,
        draft_key_cache: DeviceBuffer,
        draft_value_cache: DeviceBuffer,
        draft_cache_len: int,
        cycle_id: int = 0,
        transaction_id: int = 0,
        request_id: int = 0,
        record_stage_timings: bool = False,
        native_proposal_graph: bool = False,
    ):
        """Run one complete strict GGUF MTP cycle through the N3 adapter."""

        from hipengine.runtime.gguf_native_spec_cycle import (
            run_qwen35_gguf_native_mtp_cycle,
        )

        return run_qwen35_gguf_native_mtp_cycle(
            self,
            resident_draft,
            resident_context,
            root_token=int(root_token),
            root_position=int(root_position),
            candidate_budget=int(candidate_budget),
            remaining_decode=int(remaining_decode),
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            draft_key_cache=draft_key_cache,
            draft_value_cache=draft_value_cache,
            draft_cache_len=int(draft_cache_len),
            cycle_id=int(cycle_id),
            transaction_id=int(transaction_id),
            request_id=int(request_id),
            record_stage_timings=bool(record_stage_timings),
            native_proposal_graph=bool(native_proposal_graph),
        )

    def capture_decode_graph(
        self,
        *,
        position: int,
        steps_per_replay: int = 1,
        max_replay_steps: int | None = None,
        record_steps: int = 0,
        attention_max_context_len: int | None = None,
        capture_hidden_seed_fp32: bool = False,
        record_hidden_seeds: bool = False,
        input_token_id: int | None = None,
    ):
        """Capture one session-bound GGUF decode transition window.

        The returned graph owns a full shape/state key and refuses replay after
        any cursor drift. ``input_token_id`` re-seeds graph device feedback
        after packed-width execution, where the per-session scalar is not the
        authoritative row token. Whole-step capture is incompatible with the
        optional nested per-layer MoE graph diagnostic.
        """

        if _gguf_moe_graph_enabled():
            raise RuntimeError("whole-step GGUF graph cannot nest the per-layer MoE graph diagnostic")
        from hipengine.runtime.gguf_decode_graph import capture_qwen35_gguf_decode_graph

        if input_token_id is not None:
            self._seed_decode_graph_input_token(int(input_token_id))
        graph = capture_qwen35_gguf_decode_graph(
            self,
            position=int(position),
            steps_per_replay=int(steps_per_replay),
            max_replay_steps=max_replay_steps,
            record_steps=int(record_steps),
            attention_max_context_len=attention_max_context_len,
            capture_hidden_seed_fp32=bool(capture_hidden_seed_fp32),
            record_hidden_seeds=bool(record_hidden_seeds),
        )
        self._pin_device_kv_graph(graph)
        return graph

    def capture_packed_decode_graph(
        self,
        token_ids: list[int] | tuple[int, ...],
        *,
        sessions: list["Qwen35GGUFResidentSession"] | tuple["Qwen35GGUFResidentSession", ...] | None = None,
        physical_rows: int | None = None,
        active_slot_indices: list[int] | tuple[int, ...] | None = None,
        steps_per_replay: int = 1,
        max_replay_steps: int | None = None,
        record_steps: int = 0,
        record_layer_output_hidden: list[int] | tuple[int, ...] | set[int] = (),
    ):
        """Capture one fixed-width packed decode bucket with device feedback."""

        from hipengine.runtime.gguf_packed_decode_graph import (
            capture_qwen35_gguf_packed_decode_graph,
        )

        session_tuple = (self,) if sessions is None else tuple(sessions)
        graph = capture_qwen35_gguf_packed_decode_graph(
            self,
            token_ids=tuple(int(token) for token in token_ids),
            sessions=session_tuple,
            physical_rows=physical_rows,
            active_slot_indices=active_slot_indices,
            steps_per_replay=int(steps_per_replay),
            max_replay_steps=max_replay_steps,
            record_steps=int(record_steps),
            record_layer_output_hidden=tuple(
                int(layer_id) for layer_id in record_layer_output_hidden
            ),
        )
        for session in session_tuple:
            session._pin_device_kv_graph(graph)
        return graph

    def close(self) -> None:
        runtime = self.runtime or get_hip_runtime()
        recorder = self._prefill_flight_recorder
        if recorder is not None:
            # Never unmap the GPU-visible cursor until all queued markers retire.
            # If synchronization fails, leave the mapping registered and abort
            # teardown rather than creating a device use-after-unmap.
            runtime.device_synchronize()
            recorder.close()
            self._prefill_flight_recorder = None
        if (
            int(getattr(self, "_prefill_aotriton_stream", 0))
            or int(getattr(self, "_prefill_aotriton_input_ready_event", 0))
            or int(getattr(self, "_prefill_aotriton_output_ready_event", 0))
        ):
            runtime.device_synchronize()
            self._release_prefill_aotriton_bridge()
        self._release_int8_prefill_oracle_buffers()
        for graph in tuple(self._decode_graphs):
            graph.close()
        self._decode_graphs.clear()
        if self._device_kv_allocation is not None:
            pool = self._device_kv_pool
            allocation = self.unbind_device_kv_allocation()
            if pool is None:
                raise RuntimeError("GGUF session lost its device KV pool during close")
            pool.release(allocation.request_id, now_seconds=time.monotonic())
        if self._moe_graph is not None:
            self._moe_graph.close()
            self._moe_graph = None
        native_sampler = self._native_sampler_workspace
        if native_sampler is not None:
            native_sampler.close()
            self._native_sampler_workspace = None
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
            self._verify_hidden_f32_a,
            self._verify_hidden_f32_b,
        ):
            if buffer is not None:
                free(buffer, runtime=runtime)
        self._verify_hidden_seed_buf = None
        self._verify_hidden_f32_a = None
        self._verify_hidden_f32_b = None
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
        self._free_packed_verify_workspace(runtime=runtime)
        for buffer in reversed(self._buffers):
            if buffer is not None:
                free(buffer, runtime=runtime)
        self._buffers = ()
        for buffer in reversed(self._linear_state_snapshot_backups):
            if buffer is not None:
                free(buffer, runtime=runtime)
        self._linear_state_snapshot_backups = ()
        if self._target_scratch_owner is not None:
            for buffer in reversed(self._target_scratch_owner.buffers):
                free(buffer, runtime=runtime)
            self._target_scratch_owner = None
            self.scratch = None
        self._target_layout = None
        if self.runner is not None and self._owns_runner:
            self.runner.close()
        self.runner = None
        self._token_buf = None
        self._hidden_a = None
        self._hidden_b = None
        self._logits_buf = None
        self._native_cu_seqlens_buf = None
        self._native_state_indices_buf = None
        self._native_token_ids_host = None
        self._lm_block_values = None
        self._lm_block_indices = None
        self._lm_out_index = None
        self._lm_out_value = None
        self._prefill_token_buf = None
        self._prefill_hidden_a = None
        self._prefill_hidden_b = None
        self._bulk_prefill_scratch = None
        self._q8_mmq_risk_count = None
        self._q8_mmq_risk_indices = None
        self._logits_host = None
        self._expert_sidecar_host_layers = None
        self._expert_sidecar_reader = None
        self._expert_sidecar_model_map = None
        self._host_token_embedding_reader = None
        self._host_token_embedding_raw = None
        self._host_token_embedding_cache = {}
        self.host_token_embedding_enabled = False
        self.host_token_embedding_reason = None

    def __enter__(self) -> "Qwen35GGUFResidentSession":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


_GGUF_PREFILL_SCRATCH_EMPTY = DeviceBuffer(ptr=0, nbytes=0)
_GGUF_PREFILL_SCRATCH_COMPACT_DISABLED_FIELDS = frozenset(
    {
        "linear_z_f32",
        "linear_alpha_f32",
        "linear_beta_f32",
        "prefill_query",
        "prefill_key",
        "prefill_value",
        "linear_conv_state_tmp",
        "linear_recurrent_state_tmp",
        "post_norm_f32",
        "ffn_intermediate_f32",
        "moe_down_out_f32",
        "moe_shared_out_f32",
    }
)
_GGUF_PREFILL_SCRATCH_PEER_DISABLED_FIELDS = frozenset(
    _GGUF_PREFILL_SCRATCH_COMPACT_DISABLED_FIELDS
    - {"prefill_query", "prefill_key", "prefill_value"}
)


def _both_prefill_routes(start: int, end: int) -> tuple[tuple[str, int, int], ...]:
    return (("linear", start, end), ("full", start, end))


# Logical per-layer lifetimes for the production Qwen3.6 MoE bulk-prefill route.
# Linear- and full-attention temporaries live in mutually-exclusive routes;
# attention outputs hand off to one common post-attention MoE chain. Intervals
# are conservative unions across the compact-WMMA and unfused MoE fallbacks.
_GGUF_PREFILL_SCRATCH_LIFETIMES: Mapping[str, tuple[tuple[str, int, int], ...]] = MappingProxyType(
    {
        "norm": _both_prefill_routes(0, 1),
        "linear_qkv": (("linear", 0, 2),),
        "linear_qkv_f32": (("linear", 1, 3),),
        "linear_z": (("linear", 0, 5),),
        "linear_alpha": (("linear", 0, 4),),
        "linear_beta": (("linear", 0, 4),),
        # The exact recurrence reads the full convolution sequence while
        # writing recurrent_out, so those ranges must remain disjoint.
        "conv_out": (("linear", 2, 5),),
        # Normalized peer GDN materializes Q/K/V after convolution and keeps
        # them live only through the recurrent output handoff.
        "prefill_query": (("linear", 3, 5),),
        "prefill_key": (("linear", 3, 5),),
        "prefill_value": (("linear", 3, 5),),
        "prefill_beta": (("linear", 3, 5),),
        "prefill_decay": (("linear", 3, 5),),
        "prefill_query_scale": (("linear", 3, 5),),
        "prefill_key_scale": (("linear", 3, 5),),
        "recurrent_out": (("linear", 4, 5),),
        "recurrent_bf16": (("linear", 5, 6),),
        "full_q": (("full", 0, 2),),
        "full_k": (("full", 0, 3),),
        "full_v": (("full", 0, 4),),
        "full_query_raw": (("full", 1, 3),),
        "full_key_raw": (("full", 1, 3),),
        "full_query": (("full", 2, 4),),
        "full_key": (("full", 2, 4),),
        "full_query_bf16": (("full", 3, 5),),
        "full_gate": (("full", 1, 5),),
        "full_attn_bf16": (("full", 4, 5),),
        "full_gated": (("full", 5, 6),),
        "attn_out": _both_prefill_routes(5, 7),
        "post_norm": _both_prefill_routes(7, 15),
        "residual": _both_prefill_routes(7, 17),
        "ffn_gate_up": _both_prefill_routes(9, 15),
        "ffn_intermediate": _both_prefill_routes(10, 12),
        "ffn_down": _both_prefill_routes(12, 17),
        # q8_1 is optional in the current default but remains provisioned for
        # replacement layouts. Its union covers dense and selected routes.
        "moe_q8_1": _both_prefill_routes(0, 17),
        "moe_router_logits": _both_prefill_routes(7, 8),
        "moe_shared_gate_logits": _both_prefill_routes(7, 17),
        "moe_selected_experts": _both_prefill_routes(7, 12),
        "moe_routing_weights": _both_prefill_routes(7, 17),
        "moe_down_out": _both_prefill_routes(8, 17),
        "moe_group_counts": _both_prefill_routes(8, 12),
        "moe_padded_counts": _both_prefill_routes(8, 12),
        "moe_scatter_offsets": _both_prefill_routes(8, 12),
        "moe_expert_start_compact": _both_prefill_routes(8, 12),
        "moe_expert_start_wmma": _both_prefill_routes(8, 12),
        "moe_total_compact": _both_prefill_routes(8, 12),
        "moe_wmma_total": _both_prefill_routes(8, 12),
        "moe_tile_expert": _both_prefill_routes(8, 12),
        "moe_sorted_lanes": _both_prefill_routes(8, 13),
        "moe_sorted_experts": _both_prefill_routes(8, 13),
        "moe_sorted_weights": _both_prefill_routes(8, 13),
        "moe_lane_to_row": _both_prefill_routes(8, 13),
        "moe_shared_gate": _both_prefill_routes(13, 15),
        "moe_shared_up": _both_prefill_routes(13, 15),
        "moe_shared_intermediate": _both_prefill_routes(14, 16),
        "moe_shared_out": _both_prefill_routes(15, 17),
    }
)


def _prefill_scratch_lifetimes_overlap(
    lhs: tuple[tuple[str, int, int], ...],
    rhs: tuple[tuple[str, int, int], ...],
) -> bool:
    return any(
        lhs_route == rhs_route and lhs_start < rhs_end and rhs_start < lhs_end
        for lhs_route, lhs_start, lhs_end in lhs
        for rhs_route, rhs_start, rhs_end in rhs
    )


_GGUF_PREFILL_SCRATCH_COLOR_BYTES = 64 * 1024


def _align_prefill_scratch(value: int, alignment: int = 256) -> int:
    return (int(value) + alignment - 1) // alignment * alignment


def _allocate_prefill_scratch_liveness_arena(
    sizes: Mapping[str, int],
    *,
    runtime: HipRuntime,
) -> tuple[DeviceBuffer, dict[str, DeviceBuffer], Mapping[str, tuple[int, int]]]:
    missing = sorted(set(sizes) - set(_GGUF_PREFILL_SCRATCH_LIFETIMES))
    if missing:
        raise ValueError(f"bulk-prefill liveness plan is missing fields: {missing}")
    offsets: dict[str, tuple[int, int]] = {}
    for name, size in sorted(sizes.items(), key=lambda item: (-int(item[1]), item[0])):
        size = int(size)
        if size <= 0:
            raise ValueError(f"bulk-prefill liveness field {name} must be positive")
        candidates = {0}
        # Avoid exact power-of-two separation between simultaneously-live
        # weight/activation streams; on RDNA3 that can map both ranges onto the
        # same L2/TLB colors and made the otherwise-exact 512 route slower.
        candidates.update(
            _align_prefill_scratch(offset + allocated + _GGUF_PREFILL_SCRATCH_COLOR_BYTES)
            for offset, allocated in offsets.values()
        )
        for candidate in sorted(candidates):
            conflict = False
            for other, (other_offset, other_size) in offsets.items():
                ranges_overlap = candidate < other_offset + other_size and other_offset < candidate + size
                if ranges_overlap and _prefill_scratch_lifetimes_overlap(
                    _GGUF_PREFILL_SCRATCH_LIFETIMES[name],
                    _GGUF_PREFILL_SCRATCH_LIFETIMES[other],
                ):
                    conflict = True
                    break
            if not conflict:
                offsets[name] = (candidate, size)
                break
        else:  # pragma: no cover - candidate set always includes the current end
            raise RuntimeError(f"failed to place bulk-prefill liveness field {name}")
    arena_nbytes = _align_prefill_scratch(max(offset + size for offset, size in offsets.values()))
    arena = malloc(arena_nbytes, runtime=runtime)
    views = {
        name: DeviceBuffer(ptr=arena.ptr + offset, nbytes=size)
        for name, (offset, size) in offsets.items()
    }
    return arena, views, MappingProxyType(offsets)


def _gguf_prefill_scratch_liveness_disabled_fields(
    runner: object,
) -> frozenset[str] | None:
    cfg = getattr(getattr(runner, "weights", None), "config", None)
    backend = getattr(runner, "backend", None)
    if cfg is None or not bool(getattr(cfg, "is_moe", False)) or not isinstance(backend, str):
        return None
    try:
        admitted = bool(
            backend_package_capability(
                backend,
                "GGUF_PREFILL_SCRATCH_LIVENESS_ALIAS",
                False,
            )
        )
    except ValueError:
        return None
    if not admitted:
        return None
    requested_mode = _gguf_gdn_prefill_mode()
    if requested_mode == "auto":
        effective_mode = _gguf_gdn_prefill_backend_auto_mode(backend)
    elif requested_mode == "exact":
        effective_mode = _gguf_gdn_prefill_backend_exact_mode(backend)
    else:
        effective_mode = requested_mode
    if effective_mode in {
        "chain_lds32_direct",
        "chain_lds32_direct_nonvolatile",
    }:
        disabled_fields = _GGUF_PREFILL_SCRATCH_COMPACT_DISABLED_FIELDS
    elif effective_mode == "chain_peer_wave32":
        disabled_fields = _GGUF_PREFILL_SCRATCH_PEER_DISABLED_FIELDS
    else:
        return None
    # F32/capture diagnostics intentionally retain independently-owned buffers
    # so post-layer inspection can observe every intermediate concurrently.
    if any(
        _env_flag(name, False)
        for name in os.environ
        if name.startswith("HIPENGINE_GGUF_VERIFY_")
    ):
        return None
    return disabled_fields


@dataclass(frozen=True)
class _GGUFFullAttentionPrefillScratch:
    rows: int
    backend: str
    norm: object
    full_q: object
    full_k: object
    full_v: object
    linear_qkv: object
    linear_qkv_f32: object
    linear_z: object
    linear_z_f32: object
    linear_alpha: object
    linear_alpha_f32: object
    linear_beta: object
    linear_beta_f32: object
    conv_out: object
    prefill_query: object
    prefill_key: object
    prefill_value: object
    prefill_beta: object
    prefill_decay: object
    prefill_query_scale: object
    prefill_key_scale: object
    recurrent_out: object
    recurrent_bf16: object
    linear_conv_state_tmp: object
    linear_recurrent_state_tmp: object
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
    full_attn_split_partial: object
    full_attn_split_m: object
    full_attn_split_l: object
    attn_out: object
    post_norm: object
    post_norm_f32: object
    residual: object
    ffn_gate_up: object
    ffn_intermediate: object
    ffn_intermediate_f32: object
    ffn_down: object
    moe_q8_1: object
    moe_router_logits: object
    moe_router_counter: object
    moe_shared_gate_logits: object
    moe_selected_experts: object
    moe_routing_weights: object
    moe_down_out: object
    moe_down_out_f32: object
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
    moe_shared_out_f32: object
    key_cache: object | None
    value_cache: object | None
    retained_key_cache: object | None
    retained_value_cache: object | None
    retained_append_spans: KVLiveSpans | None
    block_table: object
    positions: object
    context_counts: object
    cu_q: object
    cu_k: object
    softmax_lse: object
    atomic: object
    block_table_tensor: Tensor
    positions_tensor: Tensor
    context_counts_tensor: Tensor
    append_spans: KVLiveSpans
    prefill_spans: KVLiveSpans
    block_size: int
    blocks: int
    max_positions: int
    full_attn_split_batch_rows: int
    full_attn_split_count: int
    moe_group_counts_zero: np.ndarray
    moe_scatter_offsets_zero: np.ndarray
    moe_wmma_total_host: np.ndarray
    moe_selected_host: np.ndarray
    moe_selected_rows_capacity: int
    moe_wmma_rows_capacity: int
    buffers: tuple[object, ...]
    runtime_state_library: object | None = None
    metadata_prepare_path: str = "host_upload"
    allocation_mode: str = "dedicated"
    allocation_offsets: Mapping[str, tuple[int, int]] = field(
        default_factory=lambda: MappingProxyType({})
    )
    allocation_lifetimes: Mapping[str, tuple[tuple[str, int, int], ...]] = field(
        default_factory=lambda: MappingProxyType({})
    )
    cos_table: object | None = None
    sin_table: object | None = None
    int8_kv_value_bf16: bool = False
    head_major_key_cache: object | None = None
    head_major_value_cache: object | None = None
    head_major_kv_capacity: int = 0
    head_major_kv_admitted: bool = False
    head_major_kv_dense_prefix: bool = True
    start: int = 0
    gdn_segment_capacity: int = 1
    gdn_active_segments: int = 1

    @classmethod
    def allocate(
        cls,
        runner: Qwen35GGUFFullStackRunner,
        *,
        rows: int,
        capacity: int | None = None,
        allocate_kv_cache: bool = True,
        segments: int = 1,
        runtime: HipRuntime,
        runtime_state_library: object | None = None,
    ):
        if rows <= 0:
            raise ValueError("rows must be positive")
        segments = int(segments)
        if segments <= 0:
            raise ValueError("segments must be positive")
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
        hidden_f32_bytes = rows * runner.hidden_size * DType.FP32.itemsize
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
        linear_z_f32_bytes = rows * cfg.ssm_inner_size * 4
        linear_ab_bytes = rows * cfg.ssm_time_step_rank * 2
        linear_ab_f32_bytes = rows * cfg.ssm_time_step_rank * 4
        recurrent_f32_bytes = rows * cfg.ssm_inner_size * 4
        conv_state_bytes = runner.linear_qkv_width * cfg.ssm_conv_kernel * DType.FP32.itemsize
        recurrent_state_bytes = (
            cfg.ssm_time_step_rank
            * cfg.ssm_state_size
            * runner.ssm_value_dim
            * DType.FP32.itemsize
        )
        prefill_scalar_bytes = rows * cfg.ssm_time_step_rank * 4
        cache_nbytes = max_positions * cfg.head_count_kv * cfg.key_length * 2 if allocate_kv_cache else 0
        full_attn_split_count = (capacity + block_size - 1) // block_size
        full_attn_split_batch_rows = min(rows, _GGUF_FULL_ATTN_PREFILL_SPLIT_BATCH_ROWS)
        full_attn_split_partial_bytes = (
            full_attn_split_batch_rows
            * runner.q_width
            * full_attn_split_count
            * DType.FP32.itemsize
        )
        full_attn_split_stat_bytes = (
            full_attn_split_batch_rows
            * cfg.head_count
            * full_attn_split_count
            * DType.FP32.itemsize
        )
        block_table_arr = np.tile(np.arange(blocks, dtype=np.int32), (rows, 1))
        positions_arr = np.arange(rows, dtype=np.int64)
        context_arr = positions_arr + np.int64(1)
        cu_arr = np.zeros((segments + 1,), dtype=np.int32)
        cu_arr[1:] = rows
        atomic_arr = np.asarray([0], dtype=np.int32)
        cos_arr, sin_arr = _rope_tables(
            max_positions=rows,
            rotary_dim=cfg.rope_dimension_count,
            base=cfg.rope_freq_base,
        )
        field_sizes = {
            "norm": hidden_bytes,
            "full_q": q_proj_bytes,
            "full_k": kv_bf16_bytes,
            "full_v": kv_bf16_bytes,
            "linear_qkv": linear_qkv_bf16_bytes,
            "linear_qkv_f32": linear_qkv_f32_bytes,
            "linear_z": linear_z_bytes,
            "linear_z_f32": linear_z_f32_bytes,
            "linear_alpha": linear_ab_bytes,
            "linear_alpha_f32": linear_ab_f32_bytes,
            "linear_beta": linear_ab_bytes,
            "linear_beta_f32": linear_ab_f32_bytes,
            "conv_out": linear_qkv_f32_bytes,
            "prefill_query": recurrent_f32_bytes,
            "prefill_key": recurrent_f32_bytes,
            "prefill_value": recurrent_f32_bytes,
            "prefill_beta": prefill_scalar_bytes,
            "prefill_decay": prefill_scalar_bytes,
            "prefill_query_scale": prefill_scalar_bytes,
            "prefill_key_scale": prefill_scalar_bytes,
            "recurrent_out": recurrent_f32_bytes,
            "recurrent_bf16": linear_z_bytes,
            "linear_conv_state_tmp": conv_state_bytes,
            "linear_recurrent_state_tmp": recurrent_state_bytes,
            "full_query_raw": q_f32_bytes,
            "full_key_raw": kv_f32_bytes,
            "full_query": q_f32_bytes,
            "full_key": kv_f32_bytes,
            "full_query_bf16": rows * runner.q_width * 2,
            "full_gate": rows * runner.q_width * 2,
            "full_attn_bf16": rows * runner.q_width * 2,
            "full_gated": rows * runner.q_width * 2,
            "attn_out": hidden_bytes,
            "post_norm": hidden_bytes,
            "post_norm_f32": hidden_f32_bytes,
            "residual": hidden_bytes,
            "ffn_gate_up": 2 * ffn_bytes * moe_lane_count,
            "ffn_intermediate": ffn_bytes * moe_lane_count,
            "ffn_intermediate_f32": moe_selected_rows_capacity * runner.ffn_size * DType.FP32.itemsize,
            "ffn_down": hidden_bytes,
            "moe_q8_1": q8_1_moe_bytes,
            "moe_router_logits": rows * moe_experts * DType.FP32.itemsize,
            "moe_shared_gate_logits": rows * DType.FP32.itemsize,
            "moe_selected_experts": rows * moe_top_k * DType.INT64.itemsize,
            "moe_routing_weights": rows * moe_top_k * DType.FP32.itemsize,
            "moe_down_out": moe_top_k * hidden_bytes,
            "moe_down_out_f32": moe_selected_rows_capacity * runner.hidden_size * DType.FP32.itemsize,
            "moe_group_counts": moe_group_counts_zero.nbytes,
            "moe_padded_counts": moe_group_counts_zero.nbytes,
            "moe_scatter_offsets": moe_scatter_offsets_zero.nbytes,
            "moe_expert_start_compact": (moe_experts + 1) * DType.INT64.itemsize,
            "moe_expert_start_wmma": (moe_experts + 1) * DType.INT64.itemsize,
            "moe_total_compact": DType.INT64.itemsize,
            "moe_wmma_total": DType.INT64.itemsize,
            "moe_tile_expert": moe_tile_capacity * DType.INT64.itemsize,
            "moe_sorted_lanes": moe_selected_rows_capacity * DType.INT64.itemsize,
            "moe_sorted_experts": moe_selected_rows_capacity * DType.INT64.itemsize,
            "moe_sorted_weights": moe_selected_rows_capacity * DType.FP32.itemsize,
            "moe_lane_to_row": moe_selected_rows_capacity * DType.INT64.itemsize,
            "moe_shared_gate": rows * moe_shared_ffn * DType.BF16.itemsize,
            "moe_shared_up": rows * moe_shared_ffn * DType.BF16.itemsize,
            "moe_shared_intermediate": rows * moe_shared_ffn * DType.BF16.itemsize,
            "moe_shared_out": hidden_bytes,
            "moe_shared_out_f32": hidden_f32_bytes,
        }
        allocation_mode = "dedicated"
        allocation_offsets: Mapping[str, tuple[int, int]] = MappingProxyType({})
        allocation_lifetimes: Mapping[str, tuple[tuple[str, int, int], ...]] = MappingProxyType({})
        owners: list[DeviceBuffer] = []
        liveness_disabled_fields = _gguf_prefill_scratch_liveness_disabled_fields(runner)
        if liveness_disabled_fields is not None:
            active_sizes = {
                name: int(nbytes)
                for name, nbytes in field_sizes.items()
                if name not in liveness_disabled_fields
            }
            arena, active_fields, allocation_offsets = _allocate_prefill_scratch_liveness_arena(
                active_sizes,
                runtime=runtime,
            )
            fields = {
                name: (
                    _GGUF_PREFILL_SCRATCH_EMPTY
                    if name in liveness_disabled_fields
                    else active_fields[name]
                )
                for name in field_sizes
            }
            owners.append(arena)
            allocation_mode = "liveness_aliased"
            allocation_lifetimes = MappingProxyType(
                {name: _GGUF_PREFILL_SCRATCH_LIFETIMES[name] for name in active_sizes}
            )
        else:
            fields = {name: buf(nbytes) for name, nbytes in field_sizes.items()}
            owners.extend(fields.values())

        dedicated_fields = {
            "gdn_cu_seqlens": buf((segments + 1) * DType.INT32.itemsize),
            "gdn_state_indices": buf(segments * DType.INT64.itemsize),
            "key_cache": buf(cache_nbytes) if allocate_kv_cache else None,
            "value_cache": buf(cache_nbytes) if allocate_kv_cache else None,
            "block_table": buf(block_table_arr.nbytes),
            "positions": buf(positions_arr.nbytes),
            "context_counts": buf(context_arr.nbytes),
            "cu_q": buf(cu_arr.nbytes),
            "cu_k": buf(cu_arr.nbytes),
            "softmax_lse": buf(cfg.head_count * rows * 4),
            "full_attn_split_partial": buf(full_attn_split_partial_bytes),
            "full_attn_split_m": buf(full_attn_split_stat_bytes),
            "full_attn_split_l": buf(full_attn_split_stat_bytes),
            "atomic": buf(atomic_arr.nbytes),
            "moe_router_counter": buf(DType.INT32.itemsize),
        }
        fields.update(dedicated_fields)
        owners.extend(value for value in dedicated_fields.values() if value is not None)
        copy_host_to_device(fields["block_table"], host_array_ptr(block_table_arr), runtime=runtime)
        copy_host_to_device(fields["positions"], host_array_ptr(positions_arr), runtime=runtime)
        copy_host_to_device(fields["context_counts"], host_array_ptr(context_arr), runtime=runtime)
        copy_host_to_device(fields["cu_q"], host_array_ptr(cu_arr), runtime=runtime)
        copy_host_to_device(fields["cu_k"], host_array_ptr(cu_arr), runtime=runtime)
        copy_host_to_device(fields["atomic"], host_array_ptr(atomic_arr), runtime=runtime)
        copy_host_to_device(fields["moe_router_counter"], host_array_ptr(atomic_arr), runtime=runtime)
        gdn_state_indices_arr = np.arange(segments, dtype=np.int64)
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
        return cls(
            **fields,
            rows=rows,
            backend=runner.backend,
            block_table_tensor=block_table_tensor,
            positions_tensor=positions_tensor,
            context_counts_tensor=context_tensor,
            retained_key_cache=None,
            retained_value_cache=None,
            retained_append_spans=None,
            append_spans=append_spans,
            prefill_spans=prefill_spans,
            block_size=block_size,
            blocks=blocks,
            max_positions=capacity,
            full_attn_split_batch_rows=full_attn_split_batch_rows,
            full_attn_split_count=full_attn_split_count,
            moe_group_counts_zero=moe_group_counts_zero,
            moe_scatter_offsets_zero=moe_scatter_offsets_zero,
            moe_wmma_total_host=moe_wmma_total_host,
            moe_selected_host=np.empty((moe_top_k,), dtype=np.int64),
            moe_selected_rows_capacity=moe_selected_rows_capacity,
            moe_wmma_rows_capacity=moe_wmma_rows_capacity,
            buffers=tuple(owners),
            runtime_state_library=runtime_state_library,
            allocation_mode=allocation_mode,
            allocation_offsets=allocation_offsets,
            allocation_lifetimes=allocation_lifetimes,
            gdn_segment_capacity=segments,
            gdn_active_segments=1,
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
        if _gguf_prefill_device_metadata_enabled(
            backend=self.backend,
            prompt_tokens=total_tokens,
        ):
            prepare_prefill_chunk_metadata(
                self.cu_q.ptr,
                self.cu_k.ptr,
                self.atomic.ptr,
                self.gdn_cu_seqlens.ptr,
                self.positions.ptr,
                self.context_counts.ptr,
                start,
                rows,
                stream=stream,
                library=self.runtime_state_library,
                runtime=runtime,
            )
        else:
            cu_q_arr = np.asarray([0, rows], dtype=np.int32)
            cu_k_arr = np.asarray([0, start + rows], dtype=np.int32)
            atomic_arr = np.asarray([0], dtype=np.int32)
            copy_host_to_device(self.cu_q, host_array_ptr(cu_q_arr), cu_q_arr.nbytes, runtime=runtime)
            copy_host_to_device(self.cu_k, host_array_ptr(cu_k_arr), cu_k_arr.nbytes, runtime=runtime)
            copy_host_to_device(self.atomic, host_array_ptr(atomic_arr), atomic_arr.nbytes, runtime=runtime)
            copy_host_to_device(
                self.gdn_cu_seqlens, host_array_ptr(cu_q_arr), cu_q_arr.nbytes, runtime=runtime
            )
            positions_arr = np.arange(start, start + rows, dtype=np.int64)
            context_arr = positions_arr + np.int64(1)
            copy_host_to_device(self.positions, host_array_ptr(positions_arr), positions_arr.nbytes, runtime=runtime)
            copy_host_to_device(self.context_counts, host_array_ptr(context_arr), context_arr.nbytes, runtime=runtime)

        block_table = Tensor.from_handle(
            self.block_table.ptr,
            (rows, self.blocks),
            DType.INT32,
            self.block_table_tensor.device,
        )
        positions = Tensor.from_handle(
            self.positions.ptr,
            (rows,),
            DType.INT64,
            self.positions_tensor.device,
        )
        context_counts = Tensor.from_handle(
            self.context_counts.ptr,
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
            head_major_kv_admitted=(
                self.head_major_key_cache is not None
                and self.head_major_value_cache is not None
                and self.head_major_kv_capacity >= total_tokens
            ),
            block_table_tensor=block_table,
            positions_tensor=positions,
            context_counts_tensor=context_counts,
            append_spans=append_spans,
            prefill_spans=prefill_spans,
            gdn_active_segments=1,
        )

    def for_packed_verify_layout(
        self,
        layout: _GGUFPackedVerifyLayout,
        *,
        runtime: HipRuntime,
        stream: int = 0,
        metadata_prepare_fn=None,
    ):
        rows = int(layout.rows)
        if rows <= 0 or rows > int(self.rows):
            raise ValueError(f"packed verify rows {rows} exceed scratch row capacity {self.rows}")
        if int(layout.slot_count) > int(self.gdn_segment_capacity):
            raise ValueError(
                f"packed verify slots {layout.slot_count} exceed GDN segment capacity {self.gdn_segment_capacity}"
            )
        if int(layout.blocks_per_slot) > int(self.blocks):
            raise ValueError(
                f"packed verify blocks_per_slot {layout.blocks_per_slot} exceed scratch block capacity {self.blocks}"
            )
        if (
            self.key_cache is not None
            and int(layout.total_physical_positions) > int(self.blocks) * int(self.block_size)
        ):
            raise ValueError("packed verify physical KV span exceeds scratch KV capacity")
        device_prepare = callable(metadata_prepare_fn) and _packed_decode_metadata_device_eligible(layout)
        if device_prepare:
            metadata_prepare_fn(
                self.block_table.ptr,
                self.positions.ptr,
                self.context_counts.ptr,
                self.cu_q.ptr,
                self.cu_k.ptr,
                self.atomic.ptr,
                self.gdn_cu_seqlens.ptr,
                self.gdn_state_indices.ptr,
                tuple(int(position) for position in layout.row_positions.tolist()),
                int(layout.blocks_per_slot),
                stream=stream,
                library=self.runtime_state_library,
                runtime=runtime,
            )
            metadata_prepare_path = "device_prepare_persistent"
        else:
            atomic_arr = np.asarray([0], dtype=np.int32)
            cu_q_arr = np.asarray([0, rows], dtype=np.int32)
            cu_k_arr = np.asarray([0, int(layout.max_live_count)], dtype=np.int32)
            copy_host_to_device(self.block_table, host_array_ptr(layout.block_table), layout.block_table.nbytes, runtime=runtime)
            copy_host_to_device(self.positions, host_array_ptr(layout.row_positions), layout.row_positions.nbytes, runtime=runtime)
            copy_host_to_device(self.context_counts, host_array_ptr(layout.live_counts), layout.live_counts.nbytes, runtime=runtime)
            copy_host_to_device(self.cu_q, host_array_ptr(cu_q_arr), cu_q_arr.nbytes, runtime=runtime)
            copy_host_to_device(self.cu_k, host_array_ptr(cu_k_arr), cu_k_arr.nbytes, runtime=runtime)
            copy_host_to_device(self.atomic, host_array_ptr(atomic_arr), atomic_arr.nbytes, runtime=runtime)
            copy_host_to_device(
                self.gdn_cu_seqlens,
                host_array_ptr(layout.cu_seqlens),
                layout.cu_seqlens.nbytes,
                runtime=runtime,
            )
            copy_host_to_device(
                self.gdn_state_indices,
                host_array_ptr(layout.state_indices),
                layout.state_indices.nbytes,
                runtime=runtime,
            )
            metadata_prepare_path = "host_upload"
        block_table = Tensor.from_handle(
            self.block_table.ptr,
            layout.block_table.shape,
            DType.INT32,
            self.block_table_tensor.device,
        )
        positions = Tensor.from_handle(
            self.positions.ptr,
            layout.row_positions.shape,
            DType.INT64,
            self.positions_tensor.device,
        )
        context_counts = Tensor.from_handle(
            self.context_counts.ptr,
            layout.live_counts.shape,
            DType.INT64,
            self.context_counts_tensor.device,
        )
        append_spans = KVLiveSpans.paged_uniform(
            block_table=block_table,
            live_counts=positions,
            max_live_count=max(0, int(layout.max_live_count) - 1),
            storage_dtype=DType.BF16,
            row_positions=positions,
            span_role="prefill",
        )
        prefill_spans = KVLiveSpans.paged_uniform(
            block_table=block_table,
            live_counts=context_counts,
            max_live_count=int(layout.max_live_count),
            storage_dtype=DType.BF16,
            row_positions=positions,
            span_role="prefill",
        )
        return replace(
            self,
            start=0,
            rows=rows,
            block_table_tensor=block_table,
            positions_tensor=positions,
            context_counts_tensor=context_counts,
            append_spans=append_spans,
            prefill_spans=prefill_spans,
            gdn_active_segments=int(layout.slot_count),
            metadata_prepare_path=metadata_prepare_path,
        )

    def for_rows(self, rows: int, *, runtime: HipRuntime, stream: int = 0):
        return self.for_chunk(start=0, rows=rows, total_tokens=rows, runtime=runtime, stream=stream)


@dataclass(frozen=True)
class _FullStackScratch:
    norm: object
    hidden_seed_fp32: object
    post_norm: object
    post_norm_f32: object
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
    linear_conv_state_tmp: object
    linear_recurrent_state_tmp: object
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
    full_bf16_mirror_key_caches: tuple[object | None, ...]
    full_bf16_mirror_value_caches: tuple[object | None, ...]
    full_k_scale_caches: tuple[object | None, ...]
    full_v_scale_caches: tuple[object | None, ...]
    full_kv_scale_metadata: tuple[KVScaleMetadata | None, ...]
    kv_storage_dtype: DType
    kv_storage_layout: str
    kv_scale_dtype: DType
    kv_scale_granularity: str
    int8_kv_value_bf16: bool
    int8_bf16_full_attention_layer_indices: tuple[int, ...]
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
    ffn_intermediate_f32: object
    ffn_down: object
    moe_q8_1: object
    moe_router_logits: object
    moe_router_counter: object
    moe_selected_experts: object
    moe_routing_weights: object
    moe_down_out: object
    moe_down_out_f32: object
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
    moe_shared_out_f32: object
    moe_shared_gate_logits: object
    moe_selected_host: np.ndarray
    moe_group_counts_zero: np.ndarray
    moe_scatter_offsets_zero: np.ndarray
    moe_selected_rows_capacity: int
    buffers: tuple[object, ...]
    slot_count: int = 1
    blocks_per_slot: int = 1

    @classmethod
    def allocate(
        cls,
        runner: Qwen35GGUFFullStackRunner,
        *,
        runtime: HipRuntime,
        max_sequence_length: int | None = None,
        max_batch_size: int = 1,
        kv_storage_dtype: str | DType = DType.BF16,
        kv_storage_layout: str = "uniform",
        kv_scale_dtype: str | DType = DType.FP16,
        kv_scale_granularity: str = "per_token_head",
        int8_kv_value_bf16: bool = False,
        int8_bf16_prefix_full_attention_layers: int = 0,
        int8_bf16_full_attention_layer_indices: tuple[int, ...] | None = None,
        allocate_kv_cache: bool = True,
    ):
        def buf(nbytes: int):
            return malloc(nbytes, runtime=runtime)

        assert runner.weights is not None
        cfg = runner.weights.config
        device = Device("hip", 0)
        block_size = 256
        kv_storage = DType.parse(kv_storage_dtype)
        if kv_storage not in {DType.BF16, DType.INT8_PER_TOKEN_HEAD}:
            raise ValueError("GGUF resident full-attention KV storage must be bf16 or int8_per_token_head")
        scale_dtype = DType.parse(kv_scale_dtype)
        if scale_dtype not in {DType.FP16, DType.FP32}:
            raise ValueError("GGUF INT8 KV scales must use fp16 or fp32")
        kv_storage_layout = str(kv_storage_layout or "uniform").strip().lower()
        if kv_storage_layout not in {"uniform", "tail4_hadamard_group32"}:
            raise ValueError(f"unsupported GGUF resident KV storage layout {kv_storage_layout!r}")
        kv_scale_granularity = str(kv_scale_granularity or "per_token_head").strip().lower()
        if kv_scale_granularity not in {"per_token_head", "block16", "hadamard_group32"}:
            raise ValueError(
                "GGUF INT8 KV scale granularity must be per_token_head, block16, or hadamard_group32"
            )
        if kv_storage_layout == "tail4_hadamard_group32" and (
            kv_storage != DType.INT8_PER_TOKEN_HEAD or kv_scale_granularity != "hadamard_group32"
        ):
            raise ValueError("tail4_hadamard_group32 requires Hadamard-group32 INT8 storage")
        int8_kv_value_bf16 = bool(int8_kv_value_bf16 and kv_storage == DType.INT8_PER_TOKEN_HEAD)
        if int8_kv_value_bf16 and kv_scale_granularity != "per_token_head":
            raise ValueError("GGUF grouped INT8 KV scales are not supported with the key-only diagnostic")
        requested_positions = block_size if max_sequence_length is None else int(max_sequence_length)
        if requested_positions <= 0:
            raise ValueError("max_sequence_length must be positive")
        if requested_positions > int(cfg.context_length):
            raise ValueError(
                f"max_sequence_length {requested_positions} exceeds GGUF context length {cfg.context_length}"
            )
        slot_count = int(max_batch_size)
        if slot_count <= 0:
            raise ValueError("max_batch_size must be positive")
        block_count = (requested_positions + block_size - 1) // block_size
        max_positions = min(int(cfg.context_length), block_count * block_size)
        total_blocks = slot_count * block_count
        total_positions = slot_count * max_positions
        hidden_bytes = slot_count * runner.hidden_size * 2
        hidden_fp32_bytes = runner.hidden_size * DType.FP32.itemsize
        ffn_bytes = slot_count * runner.ffn_size * 2
        moe_lane_count = max(1, int(cfg.expert_used_count)) if cfg.is_moe else 1
        moe_top_k = max(1, int(cfg.expert_used_count))
        moe_experts = max(1, int(cfg.expert_count))
        moe_shared_ffn = max(1, int(cfg.expert_shared_feed_forward_length or runner.ffn_size or 1))
        q8_1_gate_blocks = (runner.hidden_size + _Q8_1_BLOCK - 1) // _Q8_1_BLOCK
        q8_1_down_blocks = moe_top_k * ((runner.ffn_size + _Q8_1_BLOCK - 1) // _Q8_1_BLOCK)
        q8_1_moe_bytes = max(q8_1_gate_blocks, q8_1_down_blocks) * _Q8_1_BLOCK_BYTES
        linear_qkv_bytes = slot_count * runner.linear_qkv_width * 2
        ssm_inner_bytes = slot_count * cfg.ssm_inner_size * 2
        alpha_bytes = slot_count * cfg.ssm_time_step_rank * 2
        q_proj_bytes = slot_count * 2 * runner.q_width * 2
        kv_bf16_bytes = slot_count * runner.kv_width * 2
        q_f32_bytes = slot_count * runner.q_width * 4
        kv_f32_bytes = slot_count * runner.kv_width * 4
        full_attn_split_count = (max_positions + block_size - 1) // block_size
        full_attn_split_partial_bytes = slot_count * runner.q_width * full_attn_split_count * 4
        full_attn_split_stat_bytes = slot_count * cfg.head_count * full_attn_split_count * 4
        conv_zero = np.zeros(
            (slot_count, runner.linear_qkv_width, cfg.ssm_conv_kernel), dtype=np.float32
        )
        recurrent_zero = np.zeros(
            (slot_count, cfg.ssm_time_step_rank, cfg.ssm_state_size, runner.ssm_value_dim),
            dtype=np.float32,
        )
        layer_conv_states: list[object | None] = []
        layer_recurrent_states: list[object | None] = []
        full_key_caches: list[object | None] = []
        full_value_caches: list[object | None] = []
        full_bf16_mirror_key_caches: list[object | None] = []
        full_bf16_mirror_value_caches: list[object | None] = []
        full_k_scale_caches: list[object | None] = []
        full_v_scale_caches: list[object | None] = []
        full_kv_scale_metadata: list[KVScaleMetadata | None] = []
        state_buffers: list[object] = []
        cache_buffers: list[object] = []
        int8_bf16_prefix_full_attention_layers = max(0, int(int8_bf16_prefix_full_attention_layers))
        if int8_bf16_full_attention_layer_indices is None:
            bf16_full_attention_indices = tuple(range(int8_bf16_prefix_full_attention_layers))
        else:
            bf16_full_attention_indices = tuple(sorted({int(idx) for idx in int8_bf16_full_attention_layer_indices}))
        full_attention_count = sum(1 for layer_type in cfg.layer_types if layer_type == FULL_ATTENTION)
        bad_bf16_indices = [idx for idx in bf16_full_attention_indices if idx < 0 or idx >= full_attention_count]
        if bad_bf16_indices:
            raise ValueError(
                f"GGUF INT8 BF16 full-attention layer indices {bad_bf16_indices} outside [0, {full_attention_count})"
            )
        bf16_full_attention_index_set = frozenset(bf16_full_attention_indices)
        int8_cache_nbytes = total_positions * cfg.head_count_kv * cfg.key_length * DType.INT8.itemsize
        bf16_cache_nbytes = total_positions * cfg.head_count_kv * cfg.key_length * DType.BF16.itemsize
        mirror_bf16_nbytes = bf16_cache_nbytes
        short_int8_bf16_mirror = (
            kv_storage == DType.INT8_PER_TOKEN_HEAD
            and kv_scale_granularity != "hadamard_group32"
            and max_positions <= _GGUF_INT8_SHORT_BF16_MIRROR_MAX_POSITIONS
        )
        if kv_scale_granularity == "block16":
            scale_dim_blocks = int(cfg.key_length) // 16
            if int(cfg.key_length) % 16 != 0 or scale_dim_blocks != 16:
                raise ValueError("GGUF INT8 KV block16 scales require head_dim/key_length 256")
            scale_shape = (total_blocks, block_size, cfg.head_count_kv, scale_dim_blocks)
        elif kv_scale_granularity == "hadamard_group32":
            if int(cfg.key_length) % 32:
                raise ValueError("GGUF Hadamard-group32 KV requires head_dim/key_length divisible by 32")
            scale_shape = (total_blocks, block_size, cfg.head_count_kv, int(cfg.key_length) // 32)
        else:
            scale_shape = (total_blocks, block_size, cfg.head_count_kv)
        scale_nbytes = int(np.prod(scale_shape)) * scale_dtype.itemsize
        full_attention_index = 0
        for layer_type in cfg.layer_types:
            if layer_type == LINEAR_ATTENTION:
                conv_state = buf(conv_zero.nbytes)
                recurrent_state = buf(recurrent_zero.nbytes)
                state_buffers.extend((conv_state, recurrent_state))
                layer_conv_states.append(conv_state)
                layer_recurrent_states.append(recurrent_state)
                full_key_caches.append(None)
                full_value_caches.append(None)
                full_bf16_mirror_key_caches.append(None)
                full_bf16_mirror_value_caches.append(None)
                full_k_scale_caches.append(None)
                full_v_scale_caches.append(None)
                full_kv_scale_metadata.append(None)
            else:
                if not allocate_kv_cache:
                    layer_conv_states.append(None)
                    layer_recurrent_states.append(None)
                    full_key_caches.append(None)
                    full_value_caches.append(None)
                    full_bf16_mirror_key_caches.append(None)
                    full_bf16_mirror_value_caches.append(None)
                    full_k_scale_caches.append(None)
                    full_v_scale_caches.append(None)
                    full_kv_scale_metadata.append(None)
                    full_attention_index += 1
                    continue
                layer_uses_int8 = kv_storage == DType.INT8_PER_TOKEN_HEAD and (
                    full_attention_index not in bf16_full_attention_index_set
                )
                key_cache_nbytes = int8_cache_nbytes if layer_uses_int8 else bf16_cache_nbytes
                value_cache_nbytes = (
                    bf16_cache_nbytes if layer_uses_int8 and int8_kv_value_bf16 else key_cache_nbytes
                )
                key_cache = buf(key_cache_nbytes)
                value_cache = buf(value_cache_nbytes)
                cache_buffers.extend((key_cache, value_cache))
                layer_conv_states.append(None)
                layer_recurrent_states.append(None)
                full_key_caches.append(key_cache)
                full_value_caches.append(value_cache)
                if short_int8_bf16_mirror and layer_uses_int8:
                    mirror_key_cache = buf(mirror_bf16_nbytes)
                    mirror_value_cache = buf(mirror_bf16_nbytes)
                    cache_buffers.extend((mirror_key_cache, mirror_value_cache))
                    full_bf16_mirror_key_caches.append(mirror_key_cache)
                    full_bf16_mirror_value_caches.append(mirror_value_cache)
                else:
                    full_bf16_mirror_key_caches.append(None)
                    full_bf16_mirror_value_caches.append(None)
                if layer_uses_int8:
                    k_scale = buf(scale_nbytes)
                    v_scale = buf(scale_nbytes)
                    cache_buffers.extend((k_scale, v_scale))
                    full_k_scale_caches.append(k_scale)
                    full_v_scale_caches.append(v_scale)
                    full_kv_scale_metadata.append(
                        KVScaleMetadata(
                            k_scale=Tensor.from_handle(k_scale.ptr, scale_shape, scale_dtype, device),
                            v_scale=Tensor.from_handle(v_scale.ptr, scale_shape, scale_dtype, device),
                            scale_dtype=scale_dtype,
                            granularity=kv_scale_granularity,
                        )
                    )
                else:
                    full_k_scale_caches.append(None)
                    full_v_scale_caches.append(None)
                    full_kv_scale_metadata.append(None)
                full_attention_index += 1
        if slot_count == 1:
            block_table_arr = np.arange(block_count, dtype=np.int32)
        else:
            block_table_arr = np.arange(total_blocks, dtype=np.int32).reshape(slot_count, block_count)
        position_host = np.zeros((slot_count,), dtype=np.int64)
        context_host = np.ones((slot_count,), dtype=np.int64)
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
            "post_norm_f32": buf(hidden_fp32_bytes),
            "residual": buf(hidden_bytes),
            "attn_out": buf(hidden_bytes),
            "linear_qkv": buf(linear_qkv_bytes),
            "linear_z": buf(ssm_inner_bytes),
            "linear_alpha": buf(alpha_bytes),
            "linear_beta": buf(alpha_bytes),
            "linear_alpha_beta": buf(2 * alpha_bytes),
            "conv_out": buf(slot_count * runner.linear_qkv_width * 4),
            "recurrent_out": buf(slot_count * cfg.ssm_inner_size * 4),
            "recurrent_bf16": buf(ssm_inner_bytes),
            "linear_conv_state_tmp": buf(conv_zero.nbytes),
            "linear_recurrent_state_tmp": buf(recurrent_zero.nbytes),
            "full_q": buf(q_proj_bytes),
            "full_k": buf(kv_bf16_bytes),
            "full_v": buf(kv_bf16_bytes),
            "full_query_raw": buf(q_f32_bytes),
            "full_key_raw": buf(kv_f32_bytes),
            "full_query": buf(q_f32_bytes),
            "full_key": buf(kv_f32_bytes),
            "full_gate": buf(slot_count * runner.q_width * 2),
            "full_attn_context": buf(q_f32_bytes),
            "full_attn_split_partial": buf(full_attn_split_partial_bytes),
            "full_attn_split_m": buf(full_attn_split_stat_bytes),
            "full_attn_split_l": buf(full_attn_split_stat_bytes),
            "full_gated": buf(slot_count * runner.q_width * 2),
            "ffn_gate_up": buf(2 * ffn_bytes * moe_lane_count),
            "ffn_intermediate": buf(ffn_bytes * moe_lane_count),
            "ffn_intermediate_f32": buf(moe_top_k * runner.ffn_size * DType.FP32.itemsize),
            "ffn_down": buf(hidden_bytes),
            "moe_q8_1": buf(q8_1_moe_bytes),
            "moe_router_logits": buf(slot_count * (moe_experts + 1) * DType.FP32.itemsize),
            "moe_router_counter": buf(DType.INT32.itemsize),
            "moe_selected_experts": buf(slot_count * moe_top_k * DType.INT64.itemsize),
            "moe_routing_weights": buf(slot_count * moe_top_k * DType.FP32.itemsize),
            "moe_down_out": buf(moe_top_k * hidden_bytes),
            "moe_down_out_f32": buf(moe_top_k * runner.hidden_size * DType.FP32.itemsize),
            "moe_group_counts": buf(slot_count * moe_experts * DType.INT32.itemsize),
            "moe_padded_counts": buf(slot_count * moe_experts * DType.INT32.itemsize),
            "moe_scatter_offsets": buf(slot_count * moe_experts * DType.INT32.itemsize),
            "moe_expert_start_compact": buf(slot_count * (moe_experts + 1) * DType.INT64.itemsize),
            "moe_total_compact": buf(slot_count * DType.INT64.itemsize),
            "moe_sorted_lanes": buf(slot_count * moe_top_k * DType.INT64.itemsize),
            "moe_sorted_experts": buf(slot_count * moe_top_k * DType.INT64.itemsize),
            "moe_sorted_weights": buf(slot_count * moe_top_k * DType.FP32.itemsize),
            "moe_lane_to_row": buf(slot_count * moe_top_k * DType.INT64.itemsize),
            "moe_shared_gate": buf(slot_count * moe_shared_ffn * DType.BF16.itemsize),
            "moe_shared_up": buf(slot_count * moe_shared_ffn * DType.BF16.itemsize),
            "moe_shared_intermediate": buf(slot_count * moe_shared_ffn * DType.BF16.itemsize),
            "moe_shared_out": buf(hidden_bytes),
            "moe_shared_out_f32": buf(hidden_fp32_bytes),
            "moe_shared_gate_logits": buf(slot_count * DType.FP32.itemsize),
        }
        moe_group_counts_zero = np.zeros((moe_experts,), dtype=np.int32)
        moe_scatter_offsets_zero = np.zeros((moe_experts,), dtype=np.int32)
        router_counter_zero = np.zeros((1,), dtype=np.int32)
        copy_host_to_device(
            fields["moe_router_counter"],
            host_array_ptr(router_counter_zero),
            router_counter_zero.nbytes,
            runtime=runtime,
        )
        metadata_buffers = (block_table, position_buf, context_buf, cos_table_buf, sin_table_buf)
        return cls(
            **fields,
            full_attn_split_count=full_attn_split_count,
            full_key_caches=tuple(full_key_caches),
            full_value_caches=tuple(full_value_caches),
            full_bf16_mirror_key_caches=tuple(full_bf16_mirror_key_caches),
            full_bf16_mirror_value_caches=tuple(full_bf16_mirror_value_caches),
            full_k_scale_caches=tuple(full_k_scale_caches),
            full_v_scale_caches=tuple(full_v_scale_caches),
            full_kv_scale_metadata=tuple(full_kv_scale_metadata),
            kv_storage_dtype=kv_storage,
            kv_storage_layout=kv_storage_layout,
            kv_scale_dtype=scale_dtype,
            kv_scale_granularity=kv_scale_granularity,
            int8_kv_value_bf16=int8_kv_value_bf16,
            int8_bf16_full_attention_layer_indices=bf16_full_attention_indices,
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
            moe_selected_host=(
                np.empty((moe_top_k,), dtype=np.int64)
                if slot_count == 1
                else np.empty((slot_count, moe_top_k), dtype=np.int64)
            ),
            moe_group_counts_zero=moe_group_counts_zero,
            moe_scatter_offsets_zero=moe_scatter_offsets_zero,
            moe_selected_rows_capacity=slot_count * moe_top_k,
            buffers=tuple(fields.values()) + tuple(state_buffers) + tuple(cache_buffers) + metadata_buffers,
            slot_count=slot_count,
            blocks_per_slot=block_count,
        )

    def for_slot(self, slot: int, *, span_role: str = "decode") -> "_FullStackScratch":
        """Return a c=1-compatible view over one row of batch-shaped storage."""

        slot = int(slot)
        if slot < 0 or slot >= int(self.slot_count):
            raise ValueError(f"slot {slot} outside [0, {self.slot_count})")
        if span_role not in {"decode", "verify_chain", "verify_tree"}:
            raise ValueError("span_role must be decode, verify_chain, or verify_tree")
        if int(self.slot_count) == 1:
            if self.append_spans.span_role == span_role and self.decode_spans.span_role == span_role:
                return self
            position_tensor = self.position_tensor
            return replace(
                self,
                append_spans=replace(
                    self.append_spans,
                    row_positions=position_tensor,
                    span_role=span_role,
                ),
                decode_spans=replace(
                    self.decode_spans,
                    row_positions=position_tensor,
                    span_role=span_role,
                ),
            )

        def row(buffer) -> DeviceBuffer:
            row_nbytes = int(buffer.nbytes) // int(self.slot_count)
            return DeviceBuffer(int(buffer.ptr) + slot * row_nbytes, row_nbytes)

        block_row_nbytes = int(self.blocks_per_slot) * DType.INT32.itemsize
        block_table = DeviceBuffer(
            int(self.block_table.ptr) + slot * block_row_nbytes,
            block_row_nbytes,
        )
        position_buf = DeviceBuffer(
            int(self.position_buf.ptr) + slot * DType.INT64.itemsize,
            DType.INT64.itemsize,
        )
        context_buf = DeviceBuffer(
            int(self.context_buf.ptr) + slot * DType.INT64.itemsize,
            DType.INT64.itemsize,
        )
        block_table_tensor = Tensor.from_handle(
            block_table.ptr,
            (int(self.blocks_per_slot),),
            DType.INT32,
            self.block_table_tensor.device,
        )
        position_tensor = Tensor.from_handle(
            position_buf.ptr,
            (1,),
            DType.INT64,
            self.position_tensor.device,
        )
        context_tensor = Tensor.from_handle(
            context_buf.ptr,
            (1,),
            DType.INT64,
            self.context_tensor.device,
        )
        append_spans = KVLiveSpans.paged_uniform(
            block_table=block_table_tensor,
            live_counts=position_tensor,
            max_live_count=int(self.max_positions) - 1,
            storage_dtype=DType.BF16,
            row_positions=position_tensor,
            span_role=span_role,
        )
        decode_spans = KVLiveSpans.paged_uniform(
            block_table=block_table_tensor,
            live_counts=context_tensor,
            max_live_count=int(self.max_positions),
            storage_dtype=DType.BF16,
            row_positions=position_tensor,
            span_role=span_role,
        )
        layer_conv_states = tuple(None if state is None else row(state) for state in self.layer_conv_states)
        layer_recurrent_states = tuple(
            None if state is None else row(state) for state in self.layer_recurrent_states
        )
        selected_host = (
            self.moe_selected_host
            if self.moe_selected_host.ndim == 1
            else self.moe_selected_host[slot]
        )
        return replace(
            self,
            norm=row(self.norm),
            post_norm=row(self.post_norm),
            residual=row(self.residual),
            attn_out=row(self.attn_out),
            linear_qkv=row(self.linear_qkv),
            linear_z=row(self.linear_z),
            linear_alpha=row(self.linear_alpha),
            linear_beta=row(self.linear_beta),
            linear_alpha_beta=row(self.linear_alpha_beta),
            conv_out=row(self.conv_out),
            recurrent_out=row(self.recurrent_out),
            recurrent_bf16=row(self.recurrent_bf16),
            layer_conv_states=layer_conv_states,
            layer_recurrent_states=layer_recurrent_states,
            conv_zero=self.conv_zero[slot],
            recurrent_zero=self.recurrent_zero[slot],
            full_q=row(self.full_q),
            full_k=row(self.full_k),
            full_v=row(self.full_v),
            full_query_raw=row(self.full_query_raw),
            full_key_raw=row(self.full_key_raw),
            full_query=row(self.full_query),
            full_key=row(self.full_key),
            full_gate=row(self.full_gate),
            full_attn_context=row(self.full_attn_context),
            full_attn_split_partial=row(self.full_attn_split_partial),
            full_attn_split_m=row(self.full_attn_split_m),
            full_attn_split_l=row(self.full_attn_split_l),
            full_gated=row(self.full_gated),
            block_table=block_table,
            position_buf=position_buf,
            context_buf=context_buf,
            block_table_tensor=block_table_tensor,
            position_tensor=position_tensor,
            context_tensor=context_tensor,
            append_spans=append_spans,
            decode_spans=decode_spans,
            position_host=self.position_host[slot : slot + 1],
            context_host=self.context_host[slot : slot + 1],
            ffn_gate_up=row(self.ffn_gate_up),
            ffn_intermediate=row(self.ffn_intermediate),
            ffn_down=row(self.ffn_down),
            moe_router_logits=row(self.moe_router_logits),
            moe_selected_experts=row(self.moe_selected_experts),
            moe_routing_weights=row(self.moe_routing_weights),
            moe_down_out=row(self.moe_down_out),
            moe_group_counts=row(self.moe_group_counts),
            moe_padded_counts=row(self.moe_padded_counts),
            moe_scatter_offsets=row(self.moe_scatter_offsets),
            moe_expert_start_compact=row(self.moe_expert_start_compact),
            moe_total_compact=row(self.moe_total_compact),
            moe_sorted_lanes=row(self.moe_sorted_lanes),
            moe_sorted_experts=row(self.moe_sorted_experts),
            moe_sorted_weights=row(self.moe_sorted_weights),
            moe_lane_to_row=row(self.moe_lane_to_row),
            moe_shared_gate=row(self.moe_shared_gate),
            moe_shared_up=row(self.moe_shared_up),
            moe_shared_intermediate=row(self.moe_shared_intermediate),
            moe_shared_out=row(self.moe_shared_out),
            moe_shared_gate_logits=row(self.moe_shared_gate_logits),
            moe_selected_host=selected_host,
            buffers=(),
            slot_count=1,
        )

    def full_cache(self, layer_id: int) -> tuple[object, object]:
        key_cache = self.full_key_caches[layer_id]
        value_cache = self.full_value_caches[layer_id]
        if key_cache is None or value_cache is None:
            raise ValueError(f"layer {layer_id} has no full-attention KV cache")
        return key_cache, value_cache

    def full_bf16_mirror_cache(self, layer_id: int) -> tuple[object, object] | None:
        key_cache = self.full_bf16_mirror_key_caches[layer_id]
        value_cache = self.full_bf16_mirror_value_caches[layer_id]
        if key_cache is None or value_cache is None:
            return None
        return key_cache, value_cache

    def full_scale_metadata(self, layer_id: int) -> KVScaleMetadata | None:
        return self.full_kv_scale_metadata[layer_id]

    def append_spans_for_layer(self, layer_id: int) -> KVLiveSpans:
        metadata = self.full_scale_metadata(layer_id)
        if self.kv_storage_dtype != DType.INT8_PER_TOKEN_HEAD or metadata is None:
            return self.append_spans
        return replace(
            self.append_spans,
            storage_dtype=DType.INT8_PER_TOKEN_HEAD,
            scale_metadata=metadata,
        )

    def decode_spans_for_layer(self, layer_id: int) -> KVLiveSpans:
        metadata = self.full_scale_metadata(layer_id)
        if self.kv_storage_dtype != DType.INT8_PER_TOKEN_HEAD or metadata is None:
            return self.decode_spans
        return replace(
            self.decode_spans,
            storage_dtype=DType.INT8_PER_TOKEN_HEAD,
            scale_metadata=metadata,
        )

    def set_full_attention_position(self, position: int, runtime: HipRuntime) -> None:
        if position < 0 or position >= self.max_positions:
            raise ValueError(f"GGUF resident full-attention position {position} exceeds cache capacity {self.max_positions}")
        self.position_host[0] = int(position)
        self.context_host[0] = int(position) + 1
        copy_host_to_device(self.position_buf, host_array_ptr(self.position_host), runtime=runtime)
        copy_host_to_device(self.context_buf, host_array_ptr(self.context_host), runtime=runtime)

    def set_full_attention_positions(
        self,
        positions: tuple[int, ...] | list[int] | np.ndarray,
        runtime: HipRuntime,
    ) -> None:
        values = np.asarray(positions, dtype=np.int64).reshape(-1)
        if values.size != int(self.slot_count):
            raise ValueError("positions must contain one value per resident target slot")
        if values.size == 0 or int(values.min()) < 0 or int(values.max()) >= int(self.max_positions):
            raise ValueError("resident target positions exceed cache capacity")
        self.position_host[:] = values
        self.context_host[:] = values + np.int64(1)
        copy_host_to_device(self.position_buf, host_array_ptr(self.position_host), runtime=runtime)
        copy_host_to_device(self.context_buf, host_array_ptr(self.context_host), runtime=runtime)

    def zero_states(self, runtime: HipRuntime, *, stream: int = 0, set_position: bool = True) -> None:
        for conv_state, recurrent_state in zip(self.layer_conv_states, self.layer_recurrent_states, strict=True):
            if conv_state is not None:
                _zero(runtime, conv_state, self.conv_zero, stream=stream)
            if recurrent_state is not None:
                _zero(runtime, recurrent_state, self.recurrent_zero, stream=stream)
        if set_position:
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
_COMPACT_MOE_GROUPED_SCHEDULER_KEYS = (
    KernelKey("hip_gfx1100", "moe_group_count", "w4_paro", "qwen35"),
    KernelKey("hip_gfx1100", "moe_group_prefix", "w4_paro", "qwen35"),
    KernelKey("hip_gfx1100", "moe_group_scatter_gather", "w4_paro", "qwen35_lowp"),
)
_COMPACT_MOE_SCHEDULER_KEYS = (
    *_COMPACT_MOE_GROUPED_SCHEDULER_KEYS,
    KernelKey("hip_gfx1100", "moe_wmma_tile_map", "w4_paro", "qwen35"),
)
_COMPACT_MOE_FUSED_KEYS = (
    KernelKey("hip_gfx1100", "weighted_lanes_sum", "w4_paro", "out"),
    KernelKey("hip_gfx1100", "shared_gate_combine+residual", "w4_paro", "batch_out"),
)
_COMPACT_MOE_IQ_GROUPED_DUAL_KEYS = {
    ("gguf_iq2_xs", "gguf_iq2_xs"): KernelKey(
        "hip_gfx1100",
        "moe_linear",
        "gguf_iq2_xs",
        "selected_dual_grouped_prefill_compact_auto_bf16_bf16_out",
    ),
    ("gguf_iq3_xxs", "gguf_iq3_xxs"): KernelKey(
        "hip_gfx1100",
        "moe_linear",
        "gguf_iq3_xxs",
        "selected_dual_grouped_prefill_compact_auto_bf16_bf16_out",
    ),
    ("gguf_iq4_xs", "gguf_iq4_xs"): KernelKey(
        "hip_gfx1100",
        "moe_linear",
        "gguf_iq4_xs",
        "selected_dual_grouped_prefill_compact_bf16_bf16_out",
    ),
}
_COMPACT_MOE_IQ_GROUPED_DOWN_KEYS = {
    "gguf_iq4_xs": KernelKey(
        "hip_gfx1100",
        "moe_linear",
        "gguf_iq4_xs",
        "selected_grouped_prefill_compact_auto_bf16_bf16_out",
    )
}
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
_COMPACT_MOE_Q4_DUAL_MODE_KEYS = {
    ("gguf_q4_k_t16_v1", "gguf_q4_k_t16_v1"): {
        "baseline": _COMPACT_MOE_Q4_DUAL_KEYS[
            ("gguf_q4_k_t16_v1", "gguf_q4_k_t16_v1")
        ],
        "shared_x": KernelKey(
            "hip_gfx1100",
            "moe_linear",
            "gguf_q4_k_t16_v1",
            "selected_dual_wmma_prefill_compact32_shared_x_bf16_bf16_out",
        ),
    },
}
_COMPACT_MOE_Q4_DUAL_DS4_KEYS = {
    ("gguf_q4_k_t16_v1", "gguf_q4_k_t16_v1"): KernelKey(
        "hip_gfx1100",
        "moe_linear",
        "gguf_q4_k_t16_v1",
        "selected_dual_q8_1_ds4_wmma32_prefill_compact32_bf16_bf16_out",
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

_PACKED_DECODE_METADATA_KEY = KernelKey(
    "hip_gfx1100",
    "decode_metadata",
    "gguf_qwen35",
    "packed_c4_i64",
)
_PACKED_AR_ATTN_BATCH_KEY = KernelKey(
    "hip_gfx1100",
    "paged_attn_decode",
    "w4_paro",
    "bf16_split_k_gqa_gate_bf16_batch_spans",
)
_LINEAR_ATTN_DECODE_INDEXED_BF16_KEY = KernelKey(
    "hip_gfx1100",
    "linear_attn_conv_decode",
    "gguf_qwen35",
    "bf16_indexed",
)
_GDN_DECODE_SEGMENTS_BF16_KEY = KernelKey(
    "hip_gfx1100",
    "gdn_recurrent_rmsnorm_gate",
    "gguf_qwen35",
    "bf16_segments",
)
_GDN_DECODE_INDEXED_SINGLETON_BF16_KEY = KernelKey(
    "hip_gfx1100",
    "gdn_recurrent_rmsnorm_gate",
    "gguf_qwen35",
    "bf16_indexed_singleton",
)
_GDN_PREFILL_PREPARE_KEY = KernelKey(
    "hip_gfx1100", "linear_attn_prefill_prepare", "gguf_qwen35", "f32_bf16"
)
_GDN_PREFILL_PREPARE_PEER_NORMALIZED_KEY = KernelKey(
    "hip_gfx1100",
    "linear_attn_prefill_prepare",
    "gguf_qwen35",
    "f32_peer_normalized_bf16",
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
_GDN_PREFILL_EXACT_PREPARE_KEY = KernelKey(
    "hip_gfx1100",
    "linear_attn_prefill_prepare",
    "gguf_qwen35",
    "f32_bf16_raw_scales",
)
_GDN_PREFILL_EXACT_PREPARE_COMPACT_KEY = KernelKey(
    "hip_gfx1100",
    "linear_attn_prefill_prepare",
    "gguf_qwen35",
    "f32_bf16_compact_scales",
)
_GDN_PREFILL_EXACT_RECURRENT_KEY = KernelKey(
    "hip_gfx1100",
    "gdn_prefill_recurrent",
    "gguf_qwen35",
    "f32_decode_order_exact",
)
_GDN_PREFILL_EXACT_RECURRENT_SEGMENTS_KEY = KernelKey(
    "hip_gfx1100",
    "gdn_prefill_recurrent",
    "gguf_qwen35",
    "f32_decode_order_exact_segments",
)
_GDN_PREFILL_EXACT_RECURRENT_TILE64_KEY = KernelKey(
    "hip_gfx1100",
    "gdn_prefill_recurrent",
    "gguf_qwen35",
    "f32_decode_order_exact_tile64",
)
_GDN_PREFILL_EXACT_RECURRENT_SEGMENTS_TILE64_KEY = KernelKey(
    "hip_gfx1100",
    "gdn_prefill_recurrent",
    "gguf_qwen35",
    "f32_decode_order_exact_segments_tile64",
)
_GDN_PREFILL_EXACT_RECURRENT_TILE32_KEY = KernelKey(
    "hip_gfx1100",
    "gdn_prefill_recurrent",
    "gguf_qwen35",
    "f32_decode_order_exact_tile32",
)
_GDN_PREFILL_EXACT_RECURRENT_SEGMENTS_TILE32_KEY = KernelKey(
    "hip_gfx1100",
    "gdn_prefill_recurrent",
    "gguf_qwen35",
    "f32_decode_order_exact_segments_tile32",
)
_GDN_PREFILL_EXACT_RECURRENT_LDS64_KEY = KernelKey(
    "hip_gfx1100",
    "gdn_prefill_recurrent",
    "gguf_qwen35",
    "f32_decode_order_exact_lds64",
)
_GDN_PREFILL_EXACT_RECURRENT_SEGMENTS_LDS64_KEY = KernelKey(
    "hip_gfx1100",
    "gdn_prefill_recurrent",
    "gguf_qwen35",
    "f32_decode_order_exact_segments_lds64",
)
_GDN_PREFILL_EXACT_RECURRENT_LDS32_KEY = KernelKey(
    "hip_gfx1100",
    "gdn_prefill_recurrent",
    "gguf_qwen35",
    "f32_decode_order_exact_lds32",
)
_GDN_PREFILL_EXACT_RECURRENT_SEGMENTS_LDS32_KEY = KernelKey(
    "hip_gfx1100",
    "gdn_prefill_recurrent",
    "gguf_qwen35",
    "f32_decode_order_exact_segments_lds32",
)
_GDN_PREFILL_EXACT_RECURRENT_LDS32_DIRECT_KEY = KernelKey(
    "hip_gfx1100",
    "gdn_prefill_recurrent",
    "gguf_qwen35",
    "f32_decode_order_exact_lds32_direct",
)
_GDN_PREFILL_EXACT_RECURRENT_SEGMENTS_LDS32_DIRECT_KEY = KernelKey(
    "hip_gfx1100",
    "gdn_prefill_recurrent",
    "gguf_qwen35",
    "f32_decode_order_exact_segments_lds32_direct",
)
_GDN_PREFILL_EXACT_RECURRENT_LDS32_DIRECT_NONVOLATILE_KEY = KernelKey(
    "hip_gfx1100",
    "gdn_prefill_recurrent",
    "gguf_qwen35",
    "f32_decode_order_exact_lds32_direct_nonvolatile",
)
_GDN_PREFILL_EXACT_RECURRENT_SEGMENTS_LDS32_DIRECT_NONVOLATILE_KEY = KernelKey(
    "hip_gfx1100",
    "gdn_prefill_recurrent",
    "gguf_qwen35",
    "f32_decode_order_exact_segments_lds32_direct_nonvolatile",
)
_GDN_PREFILL_EXACT_RECURRENT_WAVE32_KEY = KernelKey(
    "hip_gfx1100",
    "gdn_prefill_recurrent",
    "gguf_qwen35",
    "f32_decode_order_exact_wave32",
)
_GDN_PREFILL_EXACT_RECURRENT_SEGMENTS_WAVE32_KEY = KernelKey(
    "hip_gfx1100",
    "gdn_prefill_recurrent",
    "gguf_qwen35",
    "f32_decode_order_exact_segments_wave32",
)
_GDN_PREFILL_RECURRENT_WAVE32_TREE_KEY = KernelKey(
    "hip_gfx1100",
    "gdn_prefill_recurrent",
    "gguf_qwen35",
    "f32_decode_order_wave32_tree",
)
_GDN_PREFILL_RECURRENT_SEGMENTS_WAVE32_TREE_KEY = KernelKey(
    "hip_gfx1100",
    "gdn_prefill_recurrent",
    "gguf_qwen35",
    "f32_decode_order_segments_wave32_tree",
)
_GDN_PREFILL_RECURRENT_PEER_WAVE32_KEY = KernelKey(
    "hip_gfx1100",
    "gdn_prefill_recurrent",
    "gguf_qwen35",
    "f32_normalized_wave32_xor",
)
_GDN_PREFILL_RECURRENT_SEGMENTS_PEER_WAVE32_KEY = KernelKey(
    "hip_gfx1100",
    "gdn_prefill_recurrent",
    "gguf_qwen35",
    "f32_normalized_segments_wave32_xor",
)
_GDN_PREFILL_RECURRENT_PEER_CLUSTER8_KEY = KernelKey(
    "hip_gfx1100",
    "gdn_prefill_recurrent",
    "gguf_qwen35",
    "f32_normalized_cluster8",
)
_GDN_PREFILL_RECURRENT_SEGMENTS_PEER_CLUSTER8_KEY = KernelKey(
    "hip_gfx1100",
    "gdn_prefill_recurrent",
    "gguf_qwen35",
    "f32_normalized_segments_cluster8",
)
_GDN_PREFILL_SEGMENT_THRESHOLD_DEFAULT = 256
_GGUF_GDN_PREFILL_MODE_ENV = "HIPENGINE_GGUF_GDN_PREFILL_MODE"
_GGUF_GDN_PREFILL_MODES = frozenset(
    {
        "auto",
        "exact",
        "fused",
        "chain",
        "chain_k2",
        "chain_peer_wave32",
        "chain_peer_cluster8",
        "chain_tile64",
        "chain_tile32",
        "chain_lds64",
        "chain_lds32",
        "chain_lds32_direct",
        "chain_lds32_direct_nonvolatile",
        "chain_wave32",
        "chain_wave32_tree",
    }
)
_GGUF_GDN_PREFILL_EXACT_MODES = frozenset(
    {
        "fused",
        "chain",
        "chain_tile64",
        "chain_tile32",
        "chain_lds64",
        "chain_lds32",
        "chain_lds32_direct",
        "chain_lds32_direct_nonvolatile",
        "chain_wave32",
    }
)
_GGUF_Q4_T16_SELECTED_PREFILL_MODE_ENV = (
    "HIPENGINE_GGUF_Q4_T16_SELECTED_PREFILL_MODE"
)
_GGUF_Q4_T16_SELECTED_PREFILL_MODES = frozenset(
    {"auto", "baseline", "shared_x"}
)


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
    gate_up_requires_ds4_input: bool = False


@dataclass(frozen=True)
class _GGUFLinearAttentionDecodeBatchPlan:
    """Optional c-aware decode state kernels resolved through the registry."""

    conv_indexed: object | None
    gdn_segments: object | None
    gdn_indexed_singleton: object | None

    @property
    def available(self) -> bool:
        return callable(self.conv_indexed) and (
            callable(self.gdn_indexed_singleton) or callable(self.gdn_segments)
        )

    @property
    def gdn_decode_path(self) -> str:
        if callable(self.gdn_indexed_singleton):
            return "indexed_singleton"
        if callable(self.gdn_segments):
            return "segments"
        return "unavailable"


@dataclass(frozen=True)
class _CompactMoeGroupedPlan:
    gate_up_fn: object
    down_fn: object | None
    gate_allocation: str
    up_allocation: str
    down_allocation: str


@dataclass(frozen=True)
class _GGUFGDNPrefillPlan:
    """Resolved kernel set for the qwen35 GGUF GDN prefill path.

    The segment-aware members are optional and only consulted when the runtime
    decides the prefill row count meets the threshold. For the current
    single-sequence prefill they are called with ``segments=1``; the same ABI
    also supports future packed segments. ``auto_mode`` comes from backend-
    package capability metadata; explicit diagnostic selections fail closed
    when their required members are not registered. When present, the raw-
    scale exact members supersede the legacy normalized-Q/K k2 members for
    explicit ``chain`` dispatch.
    """

    prepare: object | None
    recurrent: object | None
    recurrent_segments: object | None
    rmsnorm_gate: object | None
    fused_decode_order: object | None
    prepare_peer_normalized: object | None = None
    exact_prepare: object | None = None
    exact_prepare_compact: object | None = None
    exact_recurrent: object | None = None
    exact_recurrent_segments: object | None = None
    exact_recurrent_tile64: object | None = None
    exact_recurrent_segments_tile64: object | None = None
    exact_recurrent_tile32: object | None = None
    exact_recurrent_segments_tile32: object | None = None
    exact_recurrent_lds64: object | None = None
    exact_recurrent_segments_lds64: object | None = None
    exact_recurrent_lds32: object | None = None
    exact_recurrent_segments_lds32: object | None = None
    exact_recurrent_lds32_direct: object | None = None
    exact_recurrent_segments_lds32_direct: object | None = None
    exact_recurrent_lds32_direct_nonvolatile: object | None = None
    exact_recurrent_segments_lds32_direct_nonvolatile: object | None = None
    exact_recurrent_wave32: object | None = None
    exact_recurrent_segments_wave32: object | None = None
    recurrent_wave32_tree: object | None = None
    recurrent_segments_wave32_tree: object | None = None
    recurrent_peer_wave32: object | None = None
    recurrent_segments_peer_wave32: object | None = None
    recurrent_peer_cluster8: object | None = None
    recurrent_segments_peer_cluster8: object | None = None
    auto_mode: str = "fused"

    @property
    def has_chain(self) -> bool:
        return self.has_exact_chain or self.has_chain_k2

    @property
    def has_chain_k2(self) -> bool:
        return (
            self.prepare is not None
            and self.recurrent is not None
            and self.rmsnorm_gate is not None
        )

    @property
    def has_chain_peer_wave32(self) -> bool:
        return (
            self.prepare_peer_normalized is not None
            and self.recurrent_peer_wave32 is not None
            and self.rmsnorm_gate is not None
        )

    @property
    def has_chain_peer_cluster8(self) -> bool:
        return (
            self.prepare_peer_normalized is not None
            and self.recurrent_peer_cluster8 is not None
            and self.rmsnorm_gate is not None
        )

    @property
    def has_fused(self) -> bool:
        return self.fused_decode_order is not None

    @property
    def has_exact_chain(self) -> bool:
        return (
            self.exact_prepare is not None
            and self.exact_recurrent is not None
            and self.rmsnorm_gate is not None
        )

    @property
    def has_exact_chain_tile64(self) -> bool:
        return (
            self.exact_prepare is not None
            and self.exact_recurrent_tile64 is not None
            and self.rmsnorm_gate is not None
        )

    @property
    def has_exact_chain_tile32(self) -> bool:
        return (
            self.exact_prepare is not None
            and self.exact_recurrent_tile32 is not None
            and self.rmsnorm_gate is not None
        )

    @property
    def has_exact_chain_lds64(self) -> bool:
        return (
            self.exact_prepare is not None
            and self.exact_recurrent_lds64 is not None
            and self.rmsnorm_gate is not None
        )

    @property
    def has_exact_chain_lds32(self) -> bool:
        return (
            self.exact_prepare is not None
            and self.exact_recurrent_lds32 is not None
            and self.rmsnorm_gate is not None
        )

    @property
    def has_exact_chain_lds32_direct(self) -> bool:
        return (
            self.exact_prepare_compact is not None
            and self.exact_recurrent_lds32_direct is not None
            and self.rmsnorm_gate is not None
        )

    @property
    def has_exact_chain_lds32_direct_nonvolatile(self) -> bool:
        return (
            self.exact_prepare_compact is not None
            and self.exact_recurrent_lds32_direct_nonvolatile is not None
            and self.rmsnorm_gate is not None
        )

    @property
    def has_exact_chain_wave32(self) -> bool:
        return (
            self.exact_prepare is not None
            and self.exact_recurrent_wave32 is not None
            and self.rmsnorm_gate is not None
        )

    @property
    def has_chain_wave32_tree(self) -> bool:
        return (
            self.exact_prepare is not None
            and self.recurrent_wave32_tree is not None
            and self.rmsnorm_gate is not None
        )


def _gguf_gdn_prefill_plan_has_mode(plan: _GGUFGDNPrefillPlan, mode: str) -> bool:
    """Return whether *plan* contains the complete named GDN prefill route."""

    return {
        "fused": plan.has_fused,
        "chain": plan.has_chain,
        "chain_k2": plan.has_chain_k2,
        "chain_peer_wave32": plan.has_chain_peer_wave32,
        "chain_peer_cluster8": plan.has_chain_peer_cluster8,
        "chain_tile64": plan.has_exact_chain_tile64,
        "chain_tile32": plan.has_exact_chain_tile32,
        "chain_lds64": plan.has_exact_chain_lds64,
        "chain_lds32": plan.has_exact_chain_lds32,
        "chain_lds32_direct": plan.has_exact_chain_lds32_direct,
        "chain_lds32_direct_nonvolatile": plan.has_exact_chain_lds32_direct_nonvolatile,
        "chain_wave32": plan.has_exact_chain_wave32,
        "chain_wave32_tree": plan.has_chain_wave32_tree,
    }.get(str(mode), False)


def _gguf_gdn_prefill_backend_auto_mode(backend: str) -> str:
    """Resolve and validate one backend package's automatic GDN policy."""

    raw = backend_package_capability(
        backend,
        "GGUF_GDN_PREFILL_AUTO_MODE",
        "fused",
    )
    mode = str(raw).strip().lower()
    if mode in {"auto", "exact"} or mode not in _GGUF_GDN_PREFILL_MODES:
        choices = "|".join(sorted(_GGUF_GDN_PREFILL_MODES - {"auto", "exact"}))
        raise RuntimeError(
            "backend GGUF GDN prefill automatic mode must be one of "
            f"{choices}, got {mode!r} for {backend!r}"
        )
    return mode


def _gguf_gdn_prefill_backend_exact_mode(backend: str) -> str:
    """Resolve and validate one backend package's strict-exact GDN route."""

    raw = backend_package_capability(
        backend,
        "GGUF_GDN_PREFILL_EXACT_MODE",
        "chain",
    )
    mode = str(raw).strip().lower()
    if mode not in _GGUF_GDN_PREFILL_EXACT_MODES:
        choices = "|".join(sorted(_GGUF_GDN_PREFILL_EXACT_MODES))
        raise RuntimeError(
            "backend GGUF GDN prefill exact mode must be one of "
            f"{choices}, got {mode!r} for {backend!r}"
        )
    return mode


def _gguf_q4_t16_selected_prefill_requested_mode() -> str:
    """Return and validate the process-level Q4T16 schedule request."""

    mode = os.environ.get(
        _GGUF_Q4_T16_SELECTED_PREFILL_MODE_ENV, "auto"
    ).strip().lower()
    if not mode:
        mode = "auto"
    if mode not in _GGUF_Q4_T16_SELECTED_PREFILL_MODES:
        choices = "|".join(sorted(_GGUF_Q4_T16_SELECTED_PREFILL_MODES))
        raise ValueError(
            "Q4 T16 selected prefill mode "
            f"{_GGUF_Q4_T16_SELECTED_PREFILL_MODE_ENV} must be one of "
            f"{choices}, got {mode!r}"
        )
    return mode


def _gguf_q4_t16_selected_prefill_mode(backend: str) -> str:
    """Resolve the fail-closed Q4T16 selected-prefill schedule."""

    mode = _gguf_q4_t16_selected_prefill_requested_mode()
    if mode == "auto":
        raw = backend_package_capability(
            backend,
            "GGUF_Q4_T16_SELECTED_PREFILL_AUTO_MODE",
            "baseline",
        )
        mode = str(raw).strip().lower()
    if mode == "auto" or mode not in _GGUF_Q4_T16_SELECTED_PREFILL_MODES:
        choices = "|".join(
            sorted(_GGUF_Q4_T16_SELECTED_PREFILL_MODES - {"auto"})
        )
        raise RuntimeError(
            "backend Q4 T16 selected prefill automatic mode must be one of "
            f"{choices}, got {mode!r} for {backend!r}"
        )
    return mode


def _gguf_gdn_prefill_segment_threshold() -> int:
    raw = os.environ.get("HIPENGINE_GGUF_GDN_PREFILL_SEGMENT_THRESHOLD")
    if not raw:
        return _GDN_PREFILL_SEGMENT_THRESHOLD_DEFAULT
    try:
        value = int(raw)
    except ValueError:
        return _GDN_PREFILL_SEGMENT_THRESHOLD_DEFAULT
    return max(1, value)


def _gguf_gdn_prefill_mode() -> str:
    """Return the fail-closed GGUF GDN prefill diagnostic selection."""

    mode = os.environ.get(_GGUF_GDN_PREFILL_MODE_ENV, "auto").strip().lower()
    if not mode:
        mode = "auto"
    if mode not in _GGUF_GDN_PREFILL_MODES:
        choices = "|".join(sorted(_GGUF_GDN_PREFILL_MODES))
        raise ValueError(
            f"{_GGUF_GDN_PREFILL_MODE_ENV} must be one of {choices}, got {mode!r}"
        )
    return mode


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


def _gguf_paged_attn_parallel_reduce_enabled(
    backend: str,
    active_context: int,
) -> bool:
    default_enabled = bool(
        backend_package_capability(
            backend,
            "GGUF_PAGED_ATTN_PARALLEL_REDUCE",
            False,
        )
    )
    default_min_context = int(
        backend_package_capability(
            backend,
            "GGUF_PAGED_ATTN_PARALLEL_REDUCE_MIN_CONTEXT",
            32768,
        )
    )
    min_context = max(
        0,
        _env_int(
            "HIPENGINE_GGUF_PAGED_ATTN_PARALLEL_REDUCE_MIN_CONTEXT",
            default_min_context,
        ),
    )
    return (
        _env_flag(
            "HIPENGINE_GGUF_PAGED_ATTN_PARALLEL_REDUCE",
            default_enabled,
        )
        and int(active_context) >= min_context
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
    backend: str,
    block_size: int,
    num_splits: int,
    active_context: int,
):
    if _gguf_qwen35_gqa_decode_shape(config, block_size=block_size):
        if _use_gguf_paged_attn_gqa_grouped(active_context, num_splits):
            if _gguf_paged_attn_parallel_reduce_enabled(backend, active_context):
                return qwen35_paged_full_attn_decode_split_k_gqa_gate_bf16_parallel_reduce_spans
            return qwen35_paged_full_attn_decode_split_k_gqa_gate_bf16_spans
        if _gguf_paged_attn_warp_split_enabled():
            return qwen35_paged_full_attn_decode_split_k_warp_gate_bf16_spans
    return qwen35_paged_full_attn_decode_split_k_gate_bf16_spans


def _resolve_gguf_linear_attention_decode_batch_plan(
    backend: str = "hip_gfx1100",
) -> _GGUFLinearAttentionDecodeBatchPlan:
    load_backend_kernel_package(backend)

    def _resolve(key: KernelKey):
        return resolve(
            backend=backend,
            layer=key.layer,
            quant=key.quant,
            variant=key.variant,
            missing="none",
        )

    indexed_singleton_enabled = bool(
        backend_package_capability(
            backend,
            "GGUF_GDN_INDEXED_SINGLETON_DECODE",
            False,
        )
    )
    return _GGUFLinearAttentionDecodeBatchPlan(
        conv_indexed=_resolve(_LINEAR_ATTN_DECODE_INDEXED_BF16_KEY),
        gdn_segments=_resolve(_GDN_DECODE_SEGMENTS_BF16_KEY),
        gdn_indexed_singleton=(
            _resolve(_GDN_DECODE_INDEXED_SINGLETON_BF16_KEY)
            if indexed_singleton_enabled
            else None
        ),
    )


def _resolve_gguf_gdn_prefill_plan(
    backend: str = "hip_gfx1100",
) -> _GGUFGDNPrefillPlan:
    register_qwen35_linear_attn_gdn_kernels()
    load_backend_kernel_package(backend)

    def _resolve(key: KernelKey):
        return resolve(
            backend=backend,
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
        prepare_peer_normalized=_resolve(_GDN_PREFILL_PREPARE_PEER_NORMALIZED_KEY),
        exact_prepare=_resolve(_GDN_PREFILL_EXACT_PREPARE_KEY),
        exact_prepare_compact=_resolve(_GDN_PREFILL_EXACT_PREPARE_COMPACT_KEY),
        exact_recurrent=_resolve(_GDN_PREFILL_EXACT_RECURRENT_KEY),
        exact_recurrent_segments=_resolve(_GDN_PREFILL_EXACT_RECURRENT_SEGMENTS_KEY),
        exact_recurrent_tile64=_resolve(_GDN_PREFILL_EXACT_RECURRENT_TILE64_KEY),
        exact_recurrent_segments_tile64=_resolve(
            _GDN_PREFILL_EXACT_RECURRENT_SEGMENTS_TILE64_KEY
        ),
        exact_recurrent_tile32=_resolve(_GDN_PREFILL_EXACT_RECURRENT_TILE32_KEY),
        exact_recurrent_segments_tile32=_resolve(
            _GDN_PREFILL_EXACT_RECURRENT_SEGMENTS_TILE32_KEY
        ),
        exact_recurrent_lds64=_resolve(_GDN_PREFILL_EXACT_RECURRENT_LDS64_KEY),
        exact_recurrent_segments_lds64=_resolve(
            _GDN_PREFILL_EXACT_RECURRENT_SEGMENTS_LDS64_KEY
        ),
        exact_recurrent_lds32=_resolve(_GDN_PREFILL_EXACT_RECURRENT_LDS32_KEY),
        exact_recurrent_segments_lds32=_resolve(
            _GDN_PREFILL_EXACT_RECURRENT_SEGMENTS_LDS32_KEY
        ),
        exact_recurrent_lds32_direct=_resolve(
            _GDN_PREFILL_EXACT_RECURRENT_LDS32_DIRECT_KEY
        ),
        exact_recurrent_segments_lds32_direct=_resolve(
            _GDN_PREFILL_EXACT_RECURRENT_SEGMENTS_LDS32_DIRECT_KEY
        ),
        exact_recurrent_lds32_direct_nonvolatile=_resolve(
            _GDN_PREFILL_EXACT_RECURRENT_LDS32_DIRECT_NONVOLATILE_KEY
        ),
        exact_recurrent_segments_lds32_direct_nonvolatile=_resolve(
            _GDN_PREFILL_EXACT_RECURRENT_SEGMENTS_LDS32_DIRECT_NONVOLATILE_KEY
        ),
        exact_recurrent_wave32=_resolve(_GDN_PREFILL_EXACT_RECURRENT_WAVE32_KEY),
        exact_recurrent_segments_wave32=_resolve(
            _GDN_PREFILL_EXACT_RECURRENT_SEGMENTS_WAVE32_KEY
        ),
        recurrent_wave32_tree=_resolve(_GDN_PREFILL_RECURRENT_WAVE32_TREE_KEY),
        recurrent_segments_wave32_tree=_resolve(
            _GDN_PREFILL_RECURRENT_SEGMENTS_WAVE32_TREE_KEY
        ),
        recurrent_peer_wave32=_resolve(_GDN_PREFILL_RECURRENT_PEER_WAVE32_KEY),
        recurrent_segments_peer_wave32=_resolve(
            _GDN_PREFILL_RECURRENT_SEGMENTS_PEER_WAVE32_KEY
        ),
        recurrent_peer_cluster8=_resolve(_GDN_PREFILL_RECURRENT_PEER_CLUSTER8_KEY),
        recurrent_segments_peer_cluster8=_resolve(
            _GDN_PREFILL_RECURRENT_SEGMENTS_PEER_CLUSTER8_KEY
        ),
        auto_mode=_gguf_gdn_prefill_backend_auto_mode(backend),
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
    backend: str,
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
    fn = resolve(
        backend=backend,
        layer=key.layer,
        quant=key.quant,
        variant=key.variant,
        missing="none",
    )
    if fn is None:
        register_gguf_expert_pack8_gemv_kernels()
        load_backend_kernel_package(backend)
        fn = resolve(backend=backend, layer=key.layer, quant=key.quant, variant=key.variant)
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
    backend: str,
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
    fn = resolve(
        backend=backend,
        layer=key.layer,
        quant=key.quant,
        variant=key.variant,
        missing="none",
    )
    if fn is None:
        register_gguf_expert_pack8_gemv_kernels()
        load_backend_kernel_package(backend)
        fn = resolve(backend=backend, layer=key.layer, quant=key.quant, variant=key.variant)
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
    gpu_stage_recorder: _HipEventStageRecorder | None = None,
    stage_prefix: str = "target_block_ffn_moe_compact_wmma",
) -> bool:
    """Run an enabled compact grouped-MoE prefill plan when available.

    Raw-IQ exact grouping uses package-local leaf crossovers. Otherwise the
    established WMMA prefill switch selects the compact WMMA plan. Both share
    the resident count/prefix/scatter ABI; small route sets stay on the direct
    selected fallback because they cannot amortize expert grouping.
    """

    if not _scratch_has_compact_moe_fields(scratch):
        return False
    cfg = runner.weights.config if runner.weights is not None else None
    if cfg is None:
        return False
    num_experts = int(cfg.expert_count)
    grouped_plan = None
    if _iq_grouped_prefill_enabled() and selected_rows >= num_experts:
        grouped_plan = _resolve_compact_moe_grouped_kernels(gate_weight, up_weight, down_weight)
    wmma_plan = None
    if grouped_plan is None and gguf_wmma_prefill_enabled(None):
        wmma_plan = _resolve_compact_moe_wmma_kernels(gate_weight, up_weight, down_weight)
    plan = grouped_plan if grouped_plan is not None else wmma_plan
    if plan is None:
        return False
    use_wmma = wmma_plan is not None
    gate_up_fn = plan.gate_up_fn
    down_fn = plan.down_fn
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
    if use_wmma:
        wmma_rows_capacity = int(getattr(scratch, "moe_wmma_rows_capacity", 0))
        wmma_total_upper_rows, wmma_total_upper_tiles = _compact_wmma_static_upper_bound(
            selected_rows,
            num_experts,
        )
        use_wmma_total_upper_bound = (
            selected_rows <= _gguf_compact_wmma_no_read_max_selected_rows(runner.backend)
            and wmma_total_upper_rows <= wmma_rows_capacity
        )
        qwen35_moe_wmma_tile_map(
            scratch.moe_expert_start_compact.ptr,
            scratch.moe_expert_start_wmma.ptr,
            scratch.moe_tile_expert.ptr,
            scratch.moe_wmma_total.ptr,
            num_experts,
            tile_capacity=wmma_total_upper_tiles if use_wmma_total_upper_bound else 0,
            stream=stream,
            runtime=runtime,
        )
        if use_wmma_total_upper_bound:
            wmma_total_rows = wmma_total_upper_rows
            if gpu_stage_recorder is not None:
                gpu_stage_recorder.mark(f"{stage_prefix}_wmma_total_upper_bound", stage_prefix)
        else:
            wmma_total_rows = _read_i64_device_scalar(
                scratch.moe_wmma_total,
                scratch.moe_wmma_total_host,
                stream=stream,
                runtime=runtime,
            )
            if gpu_stage_recorder is not None:
                gpu_stage_recorder.mark(f"{stage_prefix}_read_wmma_total", stage_prefix)
            if wmma_total_rows <= 0 or wmma_total_rows > wmma_rows_capacity:
                return False
    if gpu_stage_recorder is not None:
        gpu_stage_recorder.mark(f"{stage_prefix}_scheduler", stage_prefix)

    if use_wmma:
        gate_up_input_ptr = scratch.moe_down_out.ptr
        if getattr(plan, "gate_up_requires_ds4_input", False):
            if not hasattr(scratch, "moe_q8_1_ds4"):
                return False
            gguf_q8_1_mmq_ds4_pack_bf16(
                scratch.moe_down_out.ptr,
                scratch.moe_q8_1_ds4.ptr,
                selected_rows,
                hidden_size,
                stream=stream,
                runtime=runtime,
            )
            gate_up_input_ptr = scratch.moe_q8_1_ds4.ptr
            if gpu_stage_recorder is not None:
                gpu_stage_recorder.mark(f"{stage_prefix}_gate_up_ds4_pack", stage_prefix)
        gate_up_fn(
            gate_up_input_ptr,
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
    else:
        gate_up_fn(
            scratch.moe_down_out.ptr,
            scratch.moe_expert_start_compact.ptr,
            gate_weight.allocation(plan.gate_allocation).tensor.ptr,
            up_weight.allocation(plan.up_allocation).tensor.ptr,
            scratch.ffn_gate_up.ptr,
            compact_rows=selected_rows,
            in_features=hidden_size,
            out_features=expert_ffn,
            num_experts=num_experts,
            stream=stream,
            runtime=runtime,
        )
    if gpu_stage_recorder is not None:
        gpu_stage_recorder.mark(f"{stage_prefix}_selected_gate_up", stage_prefix)
    silu_mul_dual_out_bf16(
        scratch.ffn_gate_up.ptr,
        scratch.ffn_intermediate.ptr,
        rows=selected_rows,
        features=expert_ffn,
        stream=stream,
        runtime=runtime,
    )
    if gpu_stage_recorder is not None:
        gpu_stage_recorder.mark(f"{stage_prefix}_selected_silu", stage_prefix)
    if use_wmma:
        assert down_fn is not None
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
    elif down_fn is not None:
        down_fn(
            scratch.ffn_intermediate.ptr,
            scratch.moe_expert_start_compact.ptr,
            down_weight.allocation(plan.down_allocation).tensor.ptr,
            scratch.moe_down_out.ptr,
            compact_rows=selected_rows,
            in_features=expert_ffn,
            out_features=hidden_size,
            num_experts=num_experts,
            stream=stream,
            runtime=runtime,
        )
    else:
        _launch_selected_raw_gguf_moe_linear(
            down_weight,
            scratch.ffn_intermediate.ptr,
            scratch.moe_sorted_experts.ptr,
            scratch.moe_down_out.ptr,
            x_rows=selected_rows,
            rows=selected_rows,
            num_experts=num_experts,
            in_features=expert_ffn,
            out_features=hidden_size,
            backend=runner.backend,
            stream=stream,
            runtime=runtime,
        )
    if gpu_stage_recorder is not None:
        gpu_stage_recorder.mark(f"{stage_prefix}_selected_down", stage_prefix)
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
    if gpu_stage_recorder is not None:
        gpu_stage_recorder.mark(f"{stage_prefix}_selected_sum", stage_prefix)

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
        if gpu_stage_recorder is not None:
            gpu_stage_recorder.mark(f"{stage_prefix}_shared_gate_up_silu", stage_prefix)
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
        if gpu_stage_recorder is not None:
            gpu_stage_recorder.mark(f"{stage_prefix}_shared_gate_up_silu", stage_prefix)
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
    if gpu_stage_recorder is not None:
        gpu_stage_recorder.mark(f"{stage_prefix}_shared_down", stage_prefix)
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
    if gpu_stage_recorder is not None:
        gpu_stage_recorder.mark(f"{stage_prefix}_combine_residual", stage_prefix)
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
    next_norm_weight_ptr: int | None = None,
    next_norm_out_ptr: int | None = None,
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
    if (next_norm_weight_ptr is None) != (next_norm_out_ptr is None):
        raise ValueError("next norm weight and output pointers must be provided together")
    if next_norm_weight_ptr is None:
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
    else:
        shared_gate_combine_residual_rmsnorm_gguf_bf16_out(
            scratch.ffn_down.ptr,
            scratch.moe_shared_out.ptr,
            scratch.moe_router_logits.ptr + num_experts * 4,
            scratch.residual.ptr,
            next_norm_weight_ptr,
            next_norm_out_ptr,
            out_ptr,
            1,
            hidden_size,
            1,
            eps=cfg.rms_norm_eps,
            stream=stream,
            runtime=runtime,
        )
    return True


def _resolve_compact_moe_grouped_kernels(
    gate_weight: Qwen35GGUFDeviceWeight,
    up_weight: Qwen35GGUFDeviceWeight,
    down_weight: Qwen35GGUFDeviceWeight,
):
    """Resolve raw-IQ expert-major scalar kernels on the compact scheduler.

    A missing grouped down kernel intentionally uses the registered direct
    selected primitive on the already sorted compact lanes.  This keeps the
    three Q6_K down layers on their exact fallback while grouping their IQ4_XS
    gate/up projections without adding a format branch to the caller.
    """

    backend = getattr(gate_weight, "backend", "hip_gfx1100")
    if (
        getattr(up_weight, "backend", backend) != backend
        or getattr(down_weight, "backend", backend) != backend
    ):
        raise ValueError("GGUF compact MoE weights must share one backend")
    gate_up_key = _COMPACT_MOE_IQ_GROUPED_DUAL_KEYS.get(
        (gate_weight.spec.quant_key, up_weight.spec.quant_key)
    )
    if gate_up_key is None:
        return None
    down_key = _COMPACT_MOE_IQ_GROUPED_DOWN_KEYS.get(down_weight.spec.quant_key)
    kernel_keys = (gate_up_key,) if down_key is None else (gate_up_key, down_key)
    required = (*_COMPACT_MOE_GROUPED_SCHEDULER_KEYS, *_COMPACT_MOE_FUSED_KEYS, *kernel_keys)
    resolved = _resolve_compact_moe_required_keys(required, backend=backend)
    if any(fn is None for fn in resolved):
        _ensure_compact_moe_wmma_registered()
        load_backend_kernel_package(backend)
        resolved = _resolve_compact_moe_required_keys(required, backend=backend)
    if any(fn is None for fn in resolved):
        return None
    return _CompactMoeGroupedPlan(
        gate_up_fn=resolved[-1] if down_key is None else resolved[-2],
        down_fn=None if down_key is None else resolved[-1],
        gate_allocation="raw",
        up_allocation="raw",
        down_allocation="raw",
    )


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

    backend = getattr(gate_weight, "backend", "hip_gfx1100")
    if (
        getattr(up_weight, "backend", backend) != backend
        or getattr(down_weight, "backend", backend) != backend
    ):
        raise ValueError("GGUF compact MoE weights must share one backend")
    gate_up_pair = (gate_weight.spec.quant_key, up_weight.spec.quant_key)
    ds4_gate_up_key = _COMPACT_MOE_Q4_DUAL_DS4_KEYS.get(gate_up_pair)
    use_ds4_gate_up = _gguf_t16_ds4_prefill_enabled() and ds4_gate_up_key is not None
    mode_keys = _COMPACT_MOE_Q4_DUAL_MODE_KEYS.get(gate_up_pair)
    selected_requested_mode = None
    selected_mode = None
    if mode_keys is not None:
        selected_requested_mode = _gguf_q4_t16_selected_prefill_requested_mode()
        selected_mode = _gguf_q4_t16_selected_prefill_mode(backend)
        selected_gate_up_key = mode_keys.get(selected_mode)
        if selected_gate_up_key is None:
            raise RuntimeError(
                "the selected Q4 T16 prefill mode has no registered key: "
                f"{selected_mode!r}"
            )
        if use_ds4_gate_up and selected_mode != "baseline":
            raise RuntimeError(
                "Q4 T16 shared-X and DS4 selected-prefill diagnostics are "
                "mutually exclusive"
            )
    else:
        selected_gate_up_key = _COMPACT_MOE_Q4_DUAL_KEYS.get(gate_up_pair)
    gate_up_key = ds4_gate_up_key if use_ds4_gate_up else selected_gate_up_key
    down_key = _COMPACT_MOE_DOWN_KEYS.get(down_weight.spec.quant_key)
    if gate_up_key is None or down_key is None:
        return None
    required = (*_COMPACT_MOE_SCHEDULER_KEYS, *_COMPACT_MOE_FUSED_KEYS, gate_up_key, down_key)
    resolved = _resolve_compact_moe_required_keys(required, backend=backend)
    if any(fn is None for fn in resolved):
        _ensure_compact_moe_wmma_registered()
        load_backend_kernel_package(backend)
        resolved = _resolve_compact_moe_required_keys(required, backend=backend)
    if any(fn is None for fn in resolved):
        if selected_requested_mode not in {None, "auto"}:
            raise RuntimeError(
                "explicit Q4 T16 selected prefill mode is unavailable: "
                f"{selected_requested_mode!r} on {backend!r}"
            )
        if mode_keys is None or selected_mode == "baseline":
            return None
        gate_up_key = mode_keys["baseline"]
        required = (
            *_COMPACT_MOE_SCHEDULER_KEYS,
            *_COMPACT_MOE_FUSED_KEYS,
            gate_up_key,
            down_key,
        )
        resolved = _resolve_compact_moe_required_keys(required, backend=backend)
        if any(fn is None for fn in resolved):
            return None
    return _CompactMoeWmmaPlan(
        gate_up_fn=resolved[-2],
        down_fn=resolved[-1],
        gate_allocation=_selected_wmma_allocation_name(gate_weight),
        up_allocation=_selected_wmma_allocation_name(up_weight),
        down_allocation=_selected_wmma_allocation_name(down_weight),
        gate_up_requires_ds4_input=use_ds4_gate_up,
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


def _resolve_compact_moe_required_keys(
    keys: tuple[KernelKey, ...],
    *,
    backend: str,
):
    return [
        resolve(
            backend=backend,
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
    register_gguf_iq_selected_prefill_kernels()
    register_gguf_q4_k_selected_prefill_kernels()
    register_gguf_q4_k_q8_1_selected_prefill_kernels()
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

    backend = getattr(gate_weight, "backend", "hip_gfx1100")
    if (
        getattr(up_weight, "backend", backend) != backend
        or getattr(down_weight, "backend", backend) != backend
    ):
        raise ValueError("GGUF compact MoE weights must share one backend")
    gate_up_key = _COMPACT_MOE_Q4_DUAL_GEMV_KEYS.get(
        (gate_weight.spec.quant_key, up_weight.spec.quant_key)
    )
    down_key = _COMPACT_MOE_DOWN_GEMV_KEYS.get(down_weight.spec.quant_key)
    if gate_up_key is None or down_key is None:
        return None
    scheduler_keys = (
        KernelKey(backend, "moe_group_count", "w4_paro", "qwen35"),
        KernelKey(backend, "moe_group_prefix", "w4_paro", "qwen35"),
        KernelKey(backend, "moe_group_scatter_gather", "w4_paro", "qwen35_lowp"),
    )
    required = (*scheduler_keys, *_COMPACT_MOE_FUSED_KEYS, gate_up_key, down_key)
    resolved = _resolve_compact_moe_required_keys(required, backend=backend)
    if any(fn is None for fn in resolved):
        _ensure_compact_moe_gemv_registered()
        load_backend_kernel_package(backend)
        resolved = _resolve_compact_moe_required_keys(required, backend=backend)
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


def _copy_bf16_rows_to_host_f32(ptr: int, rows: int, hidden_size: int, *, runtime: HipRuntime) -> np.ndarray:
    rows = int(rows)
    hidden_size = int(hidden_size)
    if rows <= 0:
        raise ValueError("rows must be positive")
    if hidden_size <= 0:
        raise ValueError("hidden_size must be positive")
    bits = np.empty((rows, hidden_size), dtype=np.uint16)
    copy_device_to_host(
        host_array_ptr(bits),
        DeviceBuffer(int(ptr), bits.nbytes),
        bits.nbytes,
        runtime=runtime,
    )
    return np.ascontiguousarray(bf16_to_float32(bits), dtype=np.float32)


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


_SELECTED_MOE_SINGLE_VARIANT = "selected_gemv_decode_bf16_bf16_out"
_SELECTED_MOE_DUAL_SILU_VARIANT = "selected_dual_silu_gemv_decode_bf16_bf16_out"
_SELECTED_MOE_WEIGHTED_DOWN_VARIANT = "selected_weighted_down_gemv_decode_bf16_bf16_out"


def _resolve_exact_selected_moe_kernel(quant_key: str, variant: str):
    key = KernelKey("hip_gfx1100", "moe_linear", quant_key, variant)
    if not is_registered(key):
        return None
    return resolve(
        backend=key.backend,
        layer=key.layer,
        quant=key.quant,
        variant=key.variant,
    )


def _raw_selected_moe_weight_ptr(weight: Qwen35GGUFDeviceWeight) -> int:
    try:
        return int(weight.allocation("raw").tensor.ptr)
    except KeyError as exc:
        raise ValueError(
            f"registered raw selected-MoE kernel requires a 'raw' allocation for {weight.spec.source.name!r}"
        ) from exc


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
    allow_legacy: bool = True,
) -> bool:
    if weight_a.spec.quant_key == weight_b.spec.quant_key:
        fn = _resolve_exact_selected_moe_kernel(
            weight_a.spec.quant_key,
            _SELECTED_MOE_DUAL_SILU_VARIANT,
        )
        if fn is not None:
            _validate_raw_rank3_expert_weight(
                weight_a,
                num_experts=num_experts,
                in_features=in_features,
                out_features=out_features,
            )
            _validate_raw_rank3_expert_weight(
                weight_b,
                num_experts=num_experts,
                in_features=in_features,
                out_features=out_features,
            )
            fn(
                x_ptr,
                selected_ptr,
                _raw_selected_moe_weight_ptr(weight_a),
                _raw_selected_moe_weight_ptr(weight_b),
                out_ptr,
                x_rows=x_rows,
                rows=rows,
                num_experts=num_experts,
                in_features=in_features,
                out_features=out_features,
                stream=stream,
                runtime=runtime,
            )
            return True
    if not allow_legacy:
        return False
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
    x_f32_ptr: int | None = None,
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
            _quantize_activation_q8_1(
                x_ptr,
                q8_1_workspace_ptr,
                x_rows,
                in_features,
                x_f32_ptr=x_f32_ptr,
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
            _quantize_activation_q8_1(
                x_ptr,
                q8_1_workspace_ptr,
                x_rows,
                in_features,
                x_f32_ptr=x_f32_ptr,
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
            selected_dual_fn = (
                gguf_q4_k_t16_selected_dual_pairreuse_gemv_bf16_bf16_out
                if _gguf_t16_selected_pairreuse_enabled() and x_rows == 8 and rows == 64
                else gguf_q4_k_t16_selected_dual_gemv_bf16_bf16_out
            )
            selected_dual_fn(
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
        _quantize_activation_q8_1(
            x_ptr,
            q8_1_workspace_ptr,
            x_rows,
            in_features,
            x_f32_ptr=x_f32_ptr,
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
    x_f32_ptr: int | None = None,
    prefer_f32_out: bool = False,
    backend: str | None = None,
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
    fn = _resolve_exact_selected_moe_kernel(quant_key, _SELECTED_MOE_SINGLE_VARIANT)
    if fn is not None:
        _validate_raw_rank3_expert_weight(
            weight,
            num_experts=num_experts,
            in_features=in_features,
            out_features=out_features,
        )
        fn(
            x_ptr,
            selected_ptr,
            _raw_selected_moe_weight_ptr(weight),
            out_ptr,
            x_rows=x_rows,
            rows=rows,
            num_experts=num_experts,
            in_features=in_features,
            out_features=out_features,
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
        return
    if (
        q8_1_workspace_ptr is not None
        and quant_key == "gguf_q5_k_t16_v1"
        and _gguf_t16_selected_dp4a_enabled()
    ):
        _quantize_activation_q8_1(
            x_ptr,
            q8_1_workspace_ptr,
            x_rows,
            in_features,
            x_f32_ptr=x_f32_ptr,
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
        _quantize_activation_q8_1(
            x_ptr,
            q8_1_workspace_ptr,
            x_rows,
            in_features,
            x_f32_ptr=x_f32_ptr,
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
        fn = (
            gguf_q5_k_x8_selected_q8_1_dp4a_gemv_bf16_f32_out
            if prefer_f32_out
            else gguf_q5_k_x8_selected_q8_1_dp4a_gemv_bf16_bf16_out
        )
        use_q8_1_input = True
    elif quant_key == "gguf_q6_k_x8_v1":
        if q8_1_workspace_ptr is None:
            raise ValueError("gguf_q6_k_x8_v1 selected GEMV requires q8_1 workspace")
        _quantize_activation_q8_1(
            x_ptr,
            q8_1_workspace_ptr,
            x_rows,
            in_features,
            x_f32_ptr=x_f32_ptr,
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
        fn = (
            gguf_q6_k_x8_selected_q8_1_dp4a_gemv_bf16_f32_out
            if prefer_f32_out
            else gguf_q6_k_x8_selected_q8_1_dp4a_gemv_bf16_bf16_out
        )
        use_q8_1_input = True
    elif (
        q8_1_workspace_ptr is not None
        and quant_key == "gguf_q5_k"
        and out_features % 8 == 0
        and _gguf_raw_selected_dp4a_enabled()
    ):
        _quantize_activation_q8_1(
            x_ptr,
            q8_1_workspace_ptr,
            x_rows,
            in_features,
            x_f32_ptr=x_f32_ptr,
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
        _quantize_activation_q8_1(
            x_ptr,
            q8_1_workspace_ptr,
            x_rows,
            in_features,
            x_f32_ptr=x_f32_ptr,
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
        if (
            not prefer_f32_out
            and _gguf_q5_t16_selected_qwen_tile8_enabled(backend)
            and x_rows == 8
            and rows == 8
            and in_features == 512
            and out_features == 2048
        ):
            fn = gguf_q5_k_t16_selected_qwen_tile8_gemv_bf16_bf16_out
        else:
            fn = (
                gguf_q5_k_t16_selected_pairreuse_gemv_bf16_bf16_out
                if _gguf_t16_selected_down_pairreuse_enabled()
                and x_rows == 64
                and rows == 64
                else gguf_q5_k_t16_selected_gemv_bf16_bf16_out
            )
    elif quant_key == "gguf_q6_k_t16_v1":
        fn = (
            gguf_q6_k_t16_selected_pairreuse_gemv_bf16_bf16_out
            if _gguf_t16_selected_q6_down_pairreuse_enabled()
            and x_rows == 64
            and rows == 64
            else gguf_q6_k_t16_selected_gemv_bf16_bf16_out
        )
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


def _launch_weighted_selected_raw_gguf_moe_linear(
    weight: Qwen35GGUFDeviceWeight,
    x_ptr: int,
    selected_ptr: int,
    routing_weights_ptr: int,
    out_ptr: int,
    *,
    tokens: int,
    top_k: int,
    num_experts: int,
    in_features: int,
    out_features: int,
    stream: int,
    runtime: HipRuntime,
) -> bool:
    fn = _resolve_exact_selected_moe_kernel(
        weight.spec.quant_key,
        _SELECTED_MOE_WEIGHTED_DOWN_VARIANT,
    )
    if fn is None:
        return False
    _validate_raw_rank3_expert_weight(
        weight,
        num_experts=num_experts,
        in_features=in_features,
        out_features=out_features,
    )
    fn(
        x_ptr,
        selected_ptr,
        routing_weights_ptr,
        _raw_selected_moe_weight_ptr(weight),
        out_ptr,
        tokens=tokens,
        top_k=top_k,
        num_experts=num_experts,
        in_features=in_features,
        out_features=out_features,
        stream=stream,
        runtime=runtime,
    )
    return True


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


@dataclass
class Qwen35GGUFNativeRowsGraph:
    """One-step native c>N HIP graph with host-fed tokens and row cursors."""

    session: Qwen35GGUFResidentSession
    graph: int
    graph_exec: int
    stream: int
    rows: int
    max_context_len: int
    span_role: str
    execution_paths: dict[str, str]
    closed: bool = False

    def step(self, token_ids: tuple[int, ...] | list[int]) -> Qwen35GGUFTargetRowsResult:
        if self.closed:
            raise RuntimeError("GGUF native row graph is closed")
        if self.session._token_buf is None or self.session._native_token_ids_host is None:
            raise RuntimeError("GGUF resident native-row buffers are closed")
        tokens = tuple(int(token) for token in token_ids)
        if len(tokens) != int(self.rows):
            raise ValueError("native row graph token count must match captured rows")
        if self.session.runner is None:
            raise RuntimeError("GGUF resident session is closed")
        for token in tokens:
            if token < 0 or token >= int(self.session.runner.vocab_size):
                raise ValueError("native row graph token id is out of range")
        current_positions = list(self.session.row_positions)
        consumed = tuple(current_positions[row] for row in range(self.rows))
        if max(consumed) >= int(self.max_context_len):
            raise ValueError("native row graph replay exceeds captured context bound")
        runtime = self.session.runtime or get_hip_runtime()
        token_host = np.asarray(tokens, dtype=np.int64)
        copy_host_to_device(
            self.session._token_buf,
            host_array_ptr(token_host),
            token_host.nbytes,
            runtime=runtime,
        )
        if self.session._target_scratch_owner is None:
            raise RuntimeError("GGUF resident target scratch is closed")
        self.session._target_scratch_owner.set_full_attention_positions(
            tuple(current_positions),
            runtime,
        )
        runtime.graph_launch(self.graph_exec, self.stream)
        runtime.stream_synchronize(self.stream)
        copy_device_to_host(
            host_array_ptr(self.session._native_token_ids_host),
            DeviceBuffer(
                self.session._lm_out_index.ptr,
                self.rows * DType.INT32.itemsize,
            ),
            self.rows * DType.INT32.itemsize,
            runtime=runtime,
        )
        next_tokens = tuple(
            int(token) for token in self.session._native_token_ids_host[: self.rows]
        )
        for row in range(self.rows):
            current_positions[row] += 1
        self.session._target_scratch_owner.set_full_attention_positions(
            tuple(current_positions),
            runtime,
        )
        self.session._position = int(current_positions[0])
        return Qwen35GGUFTargetRowsResult(
            token_ids=next_tokens,
            positions=consumed,
            slot_indices=tuple(range(self.rows)),
            span_role=self.span_role,
            logits=np.empty((self.rows, 0), dtype=np.float32),
            layer_hidden_bits={},
            execution_paths=dict(self.execution_paths),
        )

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        runtime = self.session.runtime or get_hip_runtime()
        runtime.graph_exec_destroy(self.graph_exec)
        runtime.graph_destroy(self.graph)
        runtime.stream_destroy(self.stream)

    def __enter__(self) -> "Qwen35GGUFNativeRowsGraph":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


__all__ = [
    "Qwen35GGUFFullAttentionPrefillResult",
    "Qwen35GGUFLinearAttentionBoundaryCapture",
    "Qwen35GGUFLinearAttentionLayerCapture",
    "Qwen35GGUFRouterTraceLayerCapture",
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
