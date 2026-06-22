"""Qwen3.5 GGUF runtime bring-up probes."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from types import MappingProxyType

import numpy as np

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.hip import HipRuntime, get_hip_runtime
from hipengine.core.memory import DeviceBuffer, copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc
from hipengine.core.tensor import Tensor
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
    qwen35_paged_attn_decode_int8_gqa_splitk_gate_bf16_spans,
    qwen35_paged_full_attn_decode_context_bf16_spans,
    qwen35_paged_full_attn_decode_split_k_gate_bf16_spans,
    qwen35_paged_full_attn_decode_split_k_gqa_gate_bf16_spans,
    qwen35_paged_full_attn_decode_split_k_warp_gate_bf16_spans,
    qwen35_paged_full_attn_prefill_gqa_gate_bf16_spans,
)
from hipengine.kernels.hip_gfx1100.attention.paged_kv_write import (
    build_qwen35_paged_kv_write,
    qwen35_write_paged_kv_int8_per_token_head_prompt_spans,
    qwen35_write_paged_kv_int8_per_token_head_spans,
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
from hipengine.kernels.hip_gfx1100.linear.dense_gemv import (
    dense_dual_gemv_out_bf16,
    dense_gemv_out_bf16,
)
from hipengine.kernels.hip_gfx1100.linear.lm_head import argmax_f32, build_lm_head, lm_head_argmax_stage1_blocks
from hipengine.kernels.hip_gfx1100.rotary.qwen35_rotary import qwen35_split_qgate_bf16
from hipengine.kernels.hip_gfx1100.runtime import (
    advance_decode_position_i64,
    build_runtime_state,
    record_i64_scalar_indexed,
    set_decode_position_i64,
    set_i64_scalar,
)
from hipengine.kvcache import FixedPagedKVPolicy, KVLiveSpans, KVScaleMetadata
from hipengine.kernels.hip_gfx1100.linear_attn.conv import (
    qwen35_linear_attn_conv_decode_bf16,
    qwen35_linear_attn_conv_prefill_f32,
)
from hipengine.kernels.hip_gfx1100.linear_attn.gdn import (
    qwen35_gdn_recurrent_rmsnorm_gate_lowp_bf16,
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
    gguf_q4_k_t16_selected_dual_silu_gemv_bf16_bf16_out,
    gguf_q4_k_t16_selected_gemv_bf16_bf16_out,
    gguf_q5_k_t16_selected_gemv_bf16_bf16_out,
    gguf_q6_k_t16_selected_gemv_bf16_bf16_out,
    register_gguf_t16_selected_gemv_kernels,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_k_gemv import (
    gguf_q5_k_selected_gemv_bf16_bf16_out,
    gguf_q5_k_selected_pack8_gemv_bf16_bf16_out,
    gguf_q6_k_selected_gemv_bf16_bf16_out,
    gguf_q6_k_selected_pack8_gemv_bf16_bf16_out,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_gemv import (
    gguf_q4_k_selected_dual_gemv_bf16_bf16_out,
    gguf_q4_k_selected_gemv_bf16_bf16_out,
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
    qwen35_router_logits_bf16,
    qwen35_router_select,
    qwen35_router_topk_split_shared_coop_out_bf16,
)
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
from hipengine.quant.gguf import bf16_to_float32
from hipengine.runtime.gguf_embedding import launch_gguf_embedding
from hipengine.runtime.gguf_linear import (
    GGUF_OUTPUT_F32,
    gemv_decode_session,
    gguf_gemv_decode_enabled,
    gguf_wmma_prefill_enabled,
    launch_gguf_linear,
    launch_gguf_linear_pair,
    launch_gguf_linear_pair_concat,
    launch_gguf_linear_triple,
    wmma_prefill_session,
)
from hipengine.runtime.prefill import PrefillConfig, resolve_prefill_config_for_sequence


@dataclass(frozen=True)
class Qwen35GGUFNextTokenProbeResult:
    token_id: int
    logit: float
    logits: np.ndarray


@dataclass(frozen=True)
class Qwen35GGUFFullAttentionPrefillResult:
    """Host-visible result for a GGUF full-attention layer prefill probe."""

    hidden_bits: np.ndarray
    mode: str
    used_aotriton: bool


@dataclass(frozen=True)
class Qwen35GGUFDecodeGraphWeightRole:
    """Small, serialisable description of one resident weight's decode role."""

    slot_path: str
    quant_key: str
    rank: int


@dataclass(frozen=True)
class Qwen35GGUFDecodeGraphBucketKey:
    """Shape/dispatch bucket used for one resident GGUF decode graph capture.

    The c=1 GGUF graph is still captured directly by
    :meth:`Qwen35GGUFResidentSession.capture_decode_graph`; this key records the
    replay budget and active kernel-symbol groups that must be present in the
    captured graph's rocprof trace before a P9 graph-replay row is retained.
    """

    active_c: int
    context_bucket: int
    replay_context_limit: int
    block_size: int
    replay_steps: int
    max_replay_steps: int
    use_gemv_decode: bool
    decode_repack: bool
    active_symbol_groups: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def qwen35_gguf_decode_graph_weight_roles(
    weights: Qwen35GGUFResidentWeights,
) -> tuple[Qwen35GGUFDecodeGraphWeightRole, ...]:
    """Return the lightweight weight-role inventory used by graph buckets."""

    return tuple(
        Qwen35GGUFDecodeGraphWeightRole(
            slot_path=str(weight.spec.slot_path),
            quant_key=str(weight.spec.quant_key),
            rank=len(weight.spec.source.shape),
        )
        for weight in weights.weights
    )


def qwen35_gguf_decode_graph_active_symbol_groups(
    *,
    is_moe: bool,
    layer_types: tuple[str, ...],
    weight_roles: tuple[Qwen35GGUFDecodeGraphWeightRole, ...],
    use_gemv_decode: bool,
) -> tuple[str, ...]:
    """Derive the active decode kernel-symbol groups for a GGUF graph trace.

    Groups are intentionally semantic rather than exact mangled symbols: the
    graph coverage smoke maps these names to the current T16 and pack8 symbol
    spellings. Only groups implied by the materialized weights are included, so
    optional families such as dense Q4_K are required only when active.
    """

    groups: list[str] = []

    def add(group: str) -> None:
        if group not in groups:
            groups.append(group)

    if any(layer_type == LINEAR_ATTENTION for layer_type in layer_types):
        add("gdn_decode")
    if any(layer_type == FULL_ATTENTION for layer_type in layer_types):
        add("paged_kv_write")
        add("paged_full_attention_decode")

    if is_moe:
        if _has_role(weight_roles, quant_key="gguf_q4_k_t16_v1", slot_contains="ffn_gate_exps"):
            add("moe_q4_k_selected_dual")
        elif use_gemv_decode and _has_role(weight_roles, quant_key="gguf_q4_k", rank=3, slot_contains="ffn_gate_exps"):
            add("moe_q4_k_selected_dual")

        if _has_role(weight_roles, quant_key="gguf_q5_k_t16_v1", slot_contains="ffn_down_exps"):
            add("moe_q5_k_selected")
        elif use_gemv_decode and _has_role(weight_roles, quant_key="gguf_q5_k", rank=3, slot_contains="ffn_down_exps"):
            add("moe_q5_k_selected")

        if _has_role(weight_roles, quant_key="gguf_q6_k_t16_v1", slot_contains="ffn_down_exps"):
            add("moe_q6_k_selected")
        elif use_gemv_decode and _has_role(weight_roles, quant_key="gguf_q6_k", rank=3, slot_contains="ffn_down_exps"):
            add("moe_q6_k_selected")

    if _has_role(weight_roles, quant_key="gguf_q8_0_t16_v1", rank=2):
        add("dense_q8_0_single")
    elif use_gemv_decode and _has_role(weight_roles, quant_key="gguf_q8_0", rank=2):
        add("dense_q8_0_single")
    if (
        _has_role(weight_roles, quant_key="gguf_q8_0_t16_v1", rank=2, slot_contains="ffn_gate_shexp")
        and _has_role(weight_roles, quant_key="gguf_q8_0_t16_v1", rank=2, slot_contains="ffn_up_shexp")
    ) or (
        use_gemv_decode
        and _has_role(weight_roles, quant_key="gguf_q8_0", rank=2, slot_contains="ffn_gate_shexp")
        and _has_role(weight_roles, quant_key="gguf_q8_0", rank=2, slot_contains="ffn_up_shexp")
    ):
        add("dense_q8_0_dual")

    if use_gemv_decode and _has_role(weight_roles, quant_key="gguf_q4_k", rank=2):
        add("dense_q4_k")
    if _has_role(weight_roles, quant_key="gguf_q6_k_t16_v1", rank=2, slot_contains="root.lm_head"):
        add("dense_q6_k_lm_head")
    elif use_gemv_decode and _has_role(weight_roles, quant_key="gguf_q6_k", rank=2, slot_contains="root.lm_head"):
        add("dense_q6_k_lm_head")
    return tuple(groups)


def build_qwen35_gguf_decode_graph_bucket_key(
    *,
    position: int,
    steps_per_replay: int,
    max_replay_steps: int,
    block_size: int,
    max_positions: int,
    is_moe: bool,
    layer_types: tuple[str, ...],
    weight_roles: tuple[Qwen35GGUFDecodeGraphWeightRole, ...],
    use_gemv_decode: bool,
) -> Qwen35GGUFDecodeGraphBucketKey:
    """Build the c=1 GGUF decode graph bucket key for a replay budget."""

    if position < 0:
        raise ValueError("position must be non-negative")
    if steps_per_replay <= 0:
        raise ValueError("steps_per_replay must be positive")
    if max_replay_steps <= 0:
        raise ValueError("max_replay_steps must be positive")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    replay_context_limit = int(position) + int(max_replay_steps)
    context_bucket = ((replay_context_limit + int(block_size) - 1) // int(block_size)) * int(block_size)
    if context_bucket > int(max_positions):
        raise ValueError("decode graph bucket exceeds resident cache capacity")
    decode_repack = any(role.quant_key.endswith("_t16_v1") for role in weight_roles)
    return Qwen35GGUFDecodeGraphBucketKey(
        active_c=1,
        context_bucket=context_bucket,
        replay_context_limit=replay_context_limit,
        block_size=int(block_size),
        replay_steps=int(steps_per_replay),
        max_replay_steps=int(max_replay_steps),
        use_gemv_decode=bool(use_gemv_decode),
        decode_repack=bool(decode_repack),
        active_symbol_groups=qwen35_gguf_decode_graph_active_symbol_groups(
            is_moe=bool(is_moe),
            layer_types=tuple(layer_types),
            weight_roles=tuple(weight_roles),
            use_gemv_decode=bool(use_gemv_decode),
        ),
    )


def _has_role(
    roles: tuple[Qwen35GGUFDecodeGraphWeightRole, ...],
    *,
    quant_key: str,
    rank: int | None = None,
    slot_contains: str | None = None,
) -> bool:
    for role in roles:
        if role.quant_key != quant_key:
            continue
        if rank is not None and role.rank != rank:
            continue
        if slot_contains is not None and slot_contains not in role.slot_path:
            continue
        return True
    return False


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
        if plan.has_fused:
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
            qwen35_write_paged_kv_int8_per_token_head_prompt_spans(
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
        threshold = int(PrefillConfig().attn_aotriton_min_tokens)
        use_aotriton = threshold > 0 and rows >= threshold
        paged_attn_library = self._paged_attn_decode_library()
        end = scratch.start + rows

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
        if not launch_gguf_linear_pair(
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
        ):
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
            linear_alpha_ptr = scratch.linear_alpha_beta.ptr
            linear_beta_ptr = (
                scratch.linear_alpha_beta.ptr + cfg.ssm_time_step_rank * DType.BF16.itemsize
            )
            dense_dual_gemv_out_bf16(
                scratch.norm.ptr,
                layer.weight("ssm_alpha").allocation("raw").tensor.ptr,
                layer.weight("ssm_beta").allocation("raw").tensor.ptr,
                scratch.linear_alpha_beta.ptr,
                1,
                self.hidden_size,
                cfg.ssm_time_step_rank,
                cfg.ssm_time_step_rank,
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
        f32_to_bf16(
            scratch.recurrent_out.ptr,
            scratch.recurrent_bf16.ptr,
            cfg.ssm_inner_size,
            stream=stream,
            runtime=runtime,
        )
        launch_gguf_linear(
            layer.weight("ssm_out"),
            scratch.recurrent_bf16.ptr,
            attn_out_ptr,
            rows=1,
            in_features=cfg.ssm_inner_size,
            out_features=self.hidden_size,
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
        row_nbytes = self.hidden_size * DType.BF16.itemsize
        for row in range(rows):
            hidden_row = hidden_ptr + row * row_nbytes
            attn_row = scratch.attn_out.ptr + row * row_nbytes
            if layer_type == LINEAR_ATTENTION:
                self._run_linear_attention_attn_only(layer_id, hidden_row, attn_row, decode_scratch, stream=stream)
            elif layer_type == FULL_ATTENTION:
                decode_scratch.set_full_attention_position(row, self.runtime or get_hip_runtime())
                self._run_full_attention_attn_only(
                    layer_id,
                    hidden_row,
                    attn_row,
                    decode_scratch,
                    position=row,
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
        if cfg.is_moe:
            # The small dense time-step projections feed the recurrent update.
            # Use the GEMV-order dense kernel even for multi-row qwen35moe prefill
            # so BF16 alpha/beta bits match the token-serial path exactly.
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
        bf16_to_f32(
            scratch.linear_qkv.ptr,
            scratch.linear_qkv_f32.ptr,
            rows * self.linear_qkv_width,
            stream=stream,
            library=cast_library,
            runtime=runtime,
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
        self._run_gdn_prefill(
            layer=layer,
            scratch=scratch,
            cfg=cfg,
            rows=rows,
            recurrent_state=recurrent_state,
            stream=stream,
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
            expert_sidecar=expert_sidecar,
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
    ) -> None:
        self._run_full_attention_attn_only(layer_id, hidden_ptr, scratch.attn_out.ptr, scratch, position=position, stream=stream)
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
            qwen35_write_paged_kv_int8_per_token_head_spans(
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
        if layer_uses_int8_kv and bf16_mirror_cache is None:
            metadata = decode_spans.scale_metadata
            if metadata is None:
                raise RuntimeError("GGUF INT8 full-attention decode requires scale metadata")
            chunk_size = int(scratch.block_size)
            num_splits = min(
                int(scratch.full_attn_split_count),
                max(1, (active_context + chunk_size - 1) // chunk_size),
            )
            qwen35_paged_attn_decode_int8_gqa_splitk_gate_bf16_spans(
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
            if _use_gguf_full_attention_split_decode(active_context):
                chunk_size = int(scratch.block_size)
                num_splits = min(
                    int(scratch.full_attn_split_count),
                    max(1, (active_context + chunk_size - 1) // chunk_size),
                )
                split_gate_fn = _gguf_full_attention_split_gate_bf16_fn(
                    cfg,
                    block_size=scratch.block_size,
                    num_splits=num_splits,
                    active_context=active_context,
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
                    active_context,
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
    ) -> None:
        assert self.weights is not None
        layer = self.weights.layer(layer_id)
        runtime = self.runtime or get_hip_runtime()
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
        silu_mul_separate_out_bf16(
            scratch.ffn_gate_up.ptr,
            scratch.ffn_gate_up.ptr + self.ffn_size * rows * 2,
            scratch.ffn_intermediate.ptr,
            rows=rows,
            features=self.ffn_size,
            stream=stream,
            runtime=runtime,
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

        # Router weights are GGUF F32 tensors converted to BF16 at materialization time.
        # Decode fuses expert logits, shared-gate logit, and top-k selection into one
        # cooperative launch while preserving the existing logits scratch ABI.
        qwen35_router_topk_split_shared_coop_out_bf16(
            scratch.post_norm.ptr,
            layer.weight("ffn_gate_inp").allocation().tensor.ptr,
            layer.weight("ffn_gate_inp_shexp").allocation().tensor.ptr,
            scratch.moe_router_logits.ptr,
            scratch.moe_selected_experts.ptr,
            scratch.moe_routing_weights.ptr,
            1,
            self.hidden_size,
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
        top_k = int(cfg.expert_used_count)
        if top_k <= 0:
            raise ValueError("qwen35moe GGUF expert_used_count must be positive")
        selected_rows = rows * top_k

        qwen35_router_logits_bf16(
            scratch.post_norm.ptr,
            layer.weight("ffn_gate_inp").allocation().tensor.ptr,
            scratch.moe_router_logits.ptr,
            rows,
            self.hidden_size,
            cfg.expert_count,
            stream=stream,
            runtime=runtime,
        )
        qwen35_router_logits_bf16(
            scratch.post_norm.ptr,
            layer.weight("ffn_gate_inp_shexp").allocation().tensor.ptr,
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
                stream=stream,
                runtime=runtime,
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
                stream=stream,
                runtime=runtime,
            )

        if launch_gguf_linear_pair_concat(
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
# Keep explicit INT8-KV short gates on the exact BF16 decode path. For longer
# contexts, the first full-attention layers stay BF16 by default because layer
# probes show early-layer INT8 value quantization is amplified by downstream
# routing. The env below exists only for reproducing the rejected INT8-only path.
_GGUF_INT8_SHORT_BF16_MIRROR_MAX_POSITIONS = 8192
_GGUF_INT8_LONG_BF16_PREFIX_FULL_ATTENTION_LAYERS = 3
_GGUF_INT8_BF16_PREFIX_FULL_ATTENTION_ENV = "HIPENGINE_GGUF_INT8_KV_BF16_PREFIX_FULL_LAYERS"
_GGUF_INT8_ALLOW_UNVERIFIED_LONG_ENV = "HIPENGINE_GGUF_INT8_KV_ALLOW_UNVERIFIED_LONG"
# B2: opt-in fused selected-expert MoE FFN megakernel for rows==1 raw-Q4_K decode.
_GGUF_FUSED_MOE_FFN_ENV = "HIPENGINE_GGUF_FUSED_MOE_FFN"
_GGUF_HOST_TOKEN_EMBEDDING_ENV = "HIPENGINE_GGUF_HOST_TOKEN_EMBEDDING"


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


def _gguf_host_token_embedding_requested() -> bool:
    return _env_flag(_GGUF_HOST_TOKEN_EMBEDDING_ENV, False)


def _gguf_int8_bf16_prefix_full_attention_layers(*, kv_storage_dtype: DType, max_positions: int) -> int:
    if kv_storage_dtype != DType.INT8_PER_TOKEN_HEAD:
        return 0
    if int(max_positions) <= _GGUF_INT8_SHORT_BF16_MIRROR_MAX_POSITIONS:
        return 0
    if _env_flag(_GGUF_INT8_ALLOW_UNVERIFIED_LONG_ENV, False):
        return 0
    return max(
        0,
        _env_int(
            _GGUF_INT8_BF16_PREFIX_FULL_ATTENTION_ENV,
            _GGUF_INT8_LONG_BF16_PREFIX_FULL_ATTENTION_LAYERS,
        ),
    )


def _gguf_int8_effective_scale_dtype(
    *,
    kv_storage_dtype: DType,
    max_positions: int,
    requested_scale_dtype: DType,
    bf16_prefix_full_attention_layers: int,
) -> DType:
    if (
        kv_storage_dtype == DType.INT8_PER_TOKEN_HEAD
        and int(max_positions) > _GGUF_INT8_SHORT_BF16_MIRROR_MAX_POSITIONS
        and int(bf16_prefix_full_attention_layers) > 0
        and not _env_flag(_GGUF_INT8_ALLOW_UNVERIFIED_LONG_ENV, False)
    ):
        return DType.FP32
    return requested_scale_dtype


def _validate_gguf_int8_kv_context(
    *,
    kv_storage_dtype: DType,
    max_positions: int,
    bf16_prefix_full_attention_layers: int = 0,
) -> None:
    if kv_storage_dtype != DType.INT8_PER_TOKEN_HEAD:
        return
    if int(max_positions) <= _GGUF_INT8_SHORT_BF16_MIRROR_MAX_POSITIONS:
        return
    if int(bf16_prefix_full_attention_layers) > 0:
        return
    if _env_flag(_GGUF_INT8_ALLOW_UNVERIFIED_LONG_ENV, False):
        return
    raise ValueError(
        "GGUF int8_per_token_head KV is correctness-admitted for long contexts only with "
        f"a BF16 full-attention prefix (default {_GGUF_INT8_LONG_BF16_PREFIX_FULL_ATTENTION_LAYERS} layers); "
        f"got rounded max context {int(max_positions)} and prefix={int(bf16_prefix_full_attention_layers)}. "
        "The INT8-only long-context path failed BF16-vs-INT8 logit gates and is diagnostic-only. Set "
        f"{_GGUF_INT8_ALLOW_UNVERIFIED_LONG_ENV}=1 only to reproduce blocked capacity diagnostics."
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
    kv_policy: FixedPagedKVPolicy | None = None
    kv_scale_dtype: str | DType = DType.FP16
    kv_scale_granularity: str = "per_token_head"
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
    _prefill_token_buf: object | None = field(default=None, init=False)
    _prefill_hidden_a: object | None = field(default=None, init=False)
    _prefill_hidden_b: object | None = field(default=None, init=False)
    _bulk_prefill_scratch: object | None = field(default=None, init=False)
    _runtime_state_library: object | None = field(default=None, init=False)
    _lm_head_library: object | None = field(default=None, init=False)
    _expert_pack8_library: object | None = field(default=None, init=False)
    _expert_sidecar_reader: GGUFReader | None = field(default=None, init=False)
    _expert_sidecar_model_map: object | None = field(default=None, init=False)
    _expert_sidecar_host_layers: dict[int, dict[str, GGUFExpertPackedTensor]] | None = field(default=None, init=False)
    _host_token_embedding_reader: GGUFReader | None = field(default=None, init=False)
    _host_token_embedding_raw: np.ndarray | None = field(default=None, init=False)
    _host_token_embedding_cache: dict[int, np.ndarray] = field(default_factory=dict, init=False)
    host_token_embedding_enabled: bool = field(default=False, init=False)
    host_token_embedding_reason: str | None = field(default=None, init=False)
    _token_host: np.ndarray = field(default_factory=lambda: np.empty((1,), dtype=np.int64), init=False)
    _logits_host: np.ndarray | None = field(default=None, init=False)
    _buffers: tuple[object, ...] = field(default=(), init=False)
    _position: int = field(default=0, init=False)
    _lm_head_threads: int = field(default=128, init=False)
    _lm_head_stage1_blocks: int = field(default=0, init=False)
    fastpath_safety: Qwen35GGUFFastPathSafety | None = field(default=None, init=False)
    prefill_chunk_tuning: dict[str, object] = field(default_factory=dict, init=False)
    kv_storage_dtype: DType = field(default=DType.BF16, init=False)
    int8_bf16_prefix_full_attention_layers: int = field(default=0, init=False)

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
        self.kv_policy = self.kv_policy or FixedPagedKVPolicy(block_size=256, storage_dtype=DType.BF16)
        policy_block_size = int(getattr(self.kv_policy, "block_size", 256))
        if policy_block_size != 256:
            raise ValueError("GGUF resident KV policy block_size must be 256")
        self.kv_storage_dtype = DType.parse(getattr(self.kv_policy, "storage_dtype", DType.BF16))
        if self.kv_storage_dtype not in {DType.BF16, DType.INT8_PER_TOKEN_HEAD}:
            raise ValueError("GGUF resident full-attention KV storage must be bf16 or int8_per_token_head")
        self.kv_scale_dtype = DType.parse(self.kv_scale_dtype)
        if self.kv_scale_dtype not in {DType.FP16, DType.FP32}:
            raise ValueError("GGUF resident INT8 KV scales must use fp16 or fp32")
        if self.kv_scale_granularity != "per_token_head":
            raise ValueError("GGUF resident INT8 KV scale granularity must be per_token_head")
        requested_positions = 256 if self.max_sequence_length is None else int(self.max_sequence_length)
        rounded_positions = min(
            int(self.runner.weights.config.context_length),
            ((requested_positions + 255) // 256) * 256,
        )
        self.int8_bf16_prefix_full_attention_layers = _gguf_int8_bf16_prefix_full_attention_layers(
            kv_storage_dtype=self.kv_storage_dtype,
            max_positions=rounded_positions,
        )
        self.kv_scale_dtype = _gguf_int8_effective_scale_dtype(
            kv_storage_dtype=self.kv_storage_dtype,
            max_positions=rounded_positions,
            requested_scale_dtype=self.kv_scale_dtype,
            bf16_prefix_full_attention_layers=self.int8_bf16_prefix_full_attention_layers,
        )
        _validate_gguf_int8_kv_context(
            kv_storage_dtype=self.kv_storage_dtype,
            max_positions=rounded_positions,
            bf16_prefix_full_attention_layers=self.int8_bf16_prefix_full_attention_layers,
        )
        runtime = self.runtime or get_hip_runtime()
        if _gguf_host_token_embedding_requested():
            self._offload_token_embedding_to_host(runtime=runtime)
        build_kwargs = {
            "load": True,
            "compiler_version": self.compiler_version,
            "require_cached": self.require_cached_build,
        }
        self._runtime_state_library = build_runtime_state(**build_kwargs)
        self._lm_head_library = build_lm_head(**build_kwargs)
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
            kv_storage_dtype=self.kv_storage_dtype,
            kv_scale_dtype=self.kv_scale_dtype,
            kv_scale_granularity=self.kv_scale_granularity,
            int8_bf16_prefix_full_attention_layers=self.int8_bf16_prefix_full_attention_layers,
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
        prefill_rows = self._prefill_scratch_rows(prefill_capacity)
        alloc_capacity = prefill_capacity if self.use_expert_sidecar else prefill_rows
        self._prefill_token_buf = malloc(alloc_capacity * DType.INT64.itemsize, runtime=runtime)
        self._prefill_hidden_a = malloc(alloc_capacity * hidden_bytes, runtime=runtime)
        self._prefill_hidden_b = malloc(alloc_capacity * hidden_bytes, runtime=runtime)
        self._bulk_prefill_scratch = _GGUFFullAttentionPrefillScratch.allocate(
            self.runner,
            rows=prefill_rows,
            capacity=prefill_capacity,
            allocate_kv_cache=self.kv_storage_dtype == DType.INT8_PER_TOKEN_HEAD,
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
        self.reset()

    @property
    def position(self) -> int:
        """Next token position that will be consumed by :meth:`step`."""

        return int(self._position)

    def reset(self) -> None:
        """Reset sequence state without freeing resident weights or scratch."""

        if self.scratch is None:
            raise RuntimeError("GGUF resident session is closed")
        runtime = self.runtime or get_hip_runtime()
        self.scratch.zero_states(runtime)
        self._position = 0

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
        host_weight = Qwen35GGUFDeviceWeight(spec=token_weight.spec, allocations=MappingProxyType({}))
        root_weights = dict(weights.root_weights)
        root_weights["token_embedding"] = host_weight
        self.runner.weights = Qwen35GGUFResidentWeights(
            config=weights.config,
            root_weights=MappingProxyType(root_weights),
            layers=weights.layers,
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
        copy_host_to_device(
            DeviceBuffer(int(token_ids_device_ptr), int(token_arr.nbytes)),
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
            )
        bf16_mirror_cache = None
        full_bf16_mirror_cache = getattr(self.scratch, "full_bf16_mirror_cache", None)
        if full_bf16_mirror_cache is not None:
            bf16_mirror_cache = full_bf16_mirror_cache(layer_id)
        if bf16_mirror_cache is None:
            if bulk_scratch.key_cache is None or bulk_scratch.value_cache is None:
                raise RuntimeError("GGUF INT8 retained prefill requires a BF16 oracle cache in bulk scratch")
            oracle_key_cache, oracle_value_cache = bulk_scratch.key_cache, bulk_scratch.value_cache
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
        )

    def prefill(
        self,
        token_ids: list[int] | tuple[int, ...],
        *,
        use_bulk: bool | None = None,
        bulk_attention_mode: str = "bulk",
        return_logits: bool = True,
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
                )

        self.reset()
        hidden_ptr = None
        for token_id in token_ids:
            hidden_ptr = self._run_token_to_final_hidden(int(token_id), position=self._position)
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
        alloc_capacity = self._prefill_hidden_a.nbytes // (self.runner.hidden_size * 2)
        chunk_outer = alloc_capacity < rows

        linear_min_rows = int(self.runner.weights.config.ssm_conv_kernel)
        use_wmma_prefill = gguf_wmma_prefill_enabled(None)

        if chunk_outer:
            chunk_size = self._prefill_scratch_rows(rows)
            ranges = _chunk_ranges(rows, chunk_size, min_chunk_size=linear_min_rows)
            last_chunk_rows = 0
            for chunk_start, chunk_end in ranges:
                chunk_rows = chunk_end - chunk_start
                last_chunk_rows = chunk_rows
                chunk_tokens = tokens[chunk_start:chunk_end]
                self._copy_token_embeddings_to_device(
                    chunk_tokens,
                    self._prefill_hidden_a.ptr,
                    rows=chunk_rows,
                    token_ids_device_ptr=self._prefill_token_buf.ptr,
                    stream=stream,
                )
                src = self._prefill_hidden_a
                dst = self._prefill_hidden_b
                for layer_id, layer_type in enumerate(self.runner.weights.config.layer_types):
                    bulk_scratch = self._bulk_prefill_scratch.for_chunk(
                        chunk_start, chunk_rows, total_tokens=rows, runtime=runtime, stream=stream
                    )
                    expert_sidecar = None
                    if bulk_attention_mode == "native":
                        self.runner._run_native_attention_bulk_ffn_layer_rows(
                            layer_id, layer_type, src.ptr, dst.ptr, bulk_scratch, rows=chunk_rows, stream=stream, decode_scratch=self.scratch
                        )
                    elif layer_type == LINEAR_ATTENTION:
                        self.runner._run_linear_attention_prefill_layer_rows(
                            layer_id, src.ptr, dst.ptr, bulk_scratch, rows=chunk_rows, stream=stream, decode_scratch=self.scratch, expert_sidecar=expert_sidecar
                        )
                    elif layer_type == FULL_ATTENTION:
                        layer_scratch = self._full_attention_prefill_scratch_for_layer(bulk_scratch, layer_id)
                        self.runner._run_full_attention_prefill_layer_aotriton(
                            layer_id, src.ptr, dst.ptr, layer_scratch, cos_table_ptr=self.scratch.cos_table_buf.ptr, sin_table_ptr=self.scratch.sin_table_buf.ptr, max_positions=int(self.scratch.max_positions), stream=stream, expert_sidecar=expert_sidecar
                        )
                    else:
                        raise ValueError(f"unsupported GGUF layer type {layer_type!r}")
                    src, dst = dst, src
            
            last_bulk_scratch = self._bulk_prefill_scratch.for_chunk(
                rows - 1, 1, total_tokens=rows, runtime=runtime, stream=stream
            )
            last_src_ptr = src.ptr + (last_chunk_rows - 1) * self.runner.hidden_size * 2
        else:
            self._copy_token_embeddings_to_device(
                tokens,
                self._prefill_hidden_a.ptr,
                rows=rows,
                token_ids_device_ptr=self._prefill_token_buf.ptr,
                stream=stream,
            )
            src = self._prefill_hidden_a
            dst = self._prefill_hidden_b
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
        gguf_rmsnorm_bf16_f32_weight(
            last_src_ptr,
            self.runner.weights.root("output_norm").allocation().tensor.ptr,
            last_bulk_scratch.norm.ptr,
            rows=1,
            hidden_size=self.runner.hidden_size,
            eps=self.runner.weights.config.rms_norm_eps,
            stream=stream,
            runtime=runtime,
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
            hidden_ptr = self._run_token_to_final_hidden(int(token_id), position=self._position)
            self._position += 1
            return self._sample_from_hidden(hidden_ptr, return_logits=return_logits)

    def _run_token_to_final_hidden(self, token_id: int, *, position: int, stream: int = 0) -> int:
        if self._token_buf is None:
            raise RuntimeError("GGUF resident session buffers are closed")
        self._set_full_attention_position_device(position, stream=stream)
        self._set_token_id_device(int(token_id), stream=stream)
        return self._run_current_hidden_to_final_hidden(position=position, stream=stream)

    def _run_current_hidden_to_final_hidden(self, *, position: int, stream: int = 0) -> int:
        if self.runner is None or self.scratch is None:
            raise RuntimeError("GGUF resident session is closed")
        if self._hidden_a is None or self._hidden_b is None:
            raise RuntimeError("GGUF resident session buffers are closed")
        assert self.runner.weights is not None
        runtime = self.runtime or get_hip_runtime()
        self.scratch.position_host[0] = int(position)
        self.scratch.context_host[0] = int(position) + 1
        src = self._hidden_a
        dst = self._hidden_b
        for layer_id, layer_type in enumerate(self.runner.weights.config.layer_types):
            if layer_type == LINEAR_ATTENTION:
                self.runner._run_linear_attention_layer(layer_id, src.ptr, dst.ptr, self.scratch, stream=stream)
            elif layer_type == FULL_ATTENTION:
                self.runner._run_full_attention_layer(layer_id, src.ptr, dst.ptr, self.scratch, position=position, stream=stream)
            else:
                raise ValueError(f"unsupported GGUF layer type {layer_type!r}")
            src, dst = dst, src
        gguf_rmsnorm_bf16_f32_weight(
            src.ptr,
            self.runner.weights.root("output_norm").allocation().tensor.ptr,
            self.scratch.norm.ptr,
            rows=1,
            hidden_size=self.runner.hidden_size,
            eps=self.runner.weights.config.rms_norm_eps,
            stream=stream,
            runtime=runtime,
        )
        return self.scratch.norm.ptr

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

    def capture_decode_graph(
        self,
        *,
        position: int,
        steps_per_replay: int = 1,
        max_replay_steps: int | None = None,
        record_steps: int = 0,
    ) -> "Qwen35GGUFDecodeGraph":
        """Capture one-step resident GGUF decode for graph replay.

        The captured step consumes the current device greedy token in
        ``_lm_out_index``, gathers the next token embedding from the tied GGUF
        Q6_K embedding on-device, advances resident linear/KV state once,
        runs the GGUF Q6_K lm-head into device logits, writes the next greedy
        token back to ``_lm_out_index``, and advances the device position/context
        scalar.  Optional recording appends generated token IDs to a device
        int64 buffer so graph/eager correctness gates do not need host sampling
        between replayed steps.
        """

        if self.runner is None or self.scratch is None:
            raise RuntimeError("GGUF resident session is closed")
        if self.host_token_embedding_enabled:
            raise RuntimeError("host token embedding is not compatible with GGUF HIP decode graph replay")
        if steps_per_replay <= 0:
            raise ValueError("steps_per_replay must be positive")
        if max_replay_steps is not None and max_replay_steps <= 0:
            raise ValueError("max_replay_steps must be positive")
        if record_steps < 0:
            raise ValueError("record_steps must be non-negative")
        replay_span = int(max_replay_steps) if max_replay_steps is not None else int(steps_per_replay)
        if position < 0 or position + replay_span - 1 >= self.scratch.max_positions:
            raise ValueError("decode graph replay span exceeds GGUF resident cache capacity")
        if position + steps_per_replay - 1 >= self.scratch.max_positions:
            raise ValueError("decode graph capture span exceeds GGUF resident cache capacity")
        bucket_key = build_qwen35_gguf_decode_graph_bucket_key(
            position=int(position),
            steps_per_replay=int(steps_per_replay),
            max_replay_steps=int(replay_span),
            block_size=int(self.scratch.block_size),
            max_positions=int(self.scratch.max_positions),
            is_moe=bool(self.runner.weights.config.is_moe),
            layer_types=tuple(self.runner.weights.config.layer_types),
            weight_roles=qwen35_gguf_decode_graph_weight_roles(self.runner.weights),
            use_gemv_decode=bool(self.use_gemv_decode),
        )

        runtime = self.runtime or get_hip_runtime()
        generated_buf: DeviceBuffer | None = None
        generated_index_buf: DeviceBuffer | None = None
        if record_steps:
            generated_buf = malloc(int(record_steps) * DType.INT64.itemsize, runtime=runtime)
            generated_index_buf = malloc(DType.INT64.itemsize, runtime=runtime)
            runtime.memset(generated_buf.ptr, 0xFF, generated_buf.nbytes)
            zero = np.zeros((1,), dtype=np.int64)
            copy_host_to_device(generated_index_buf, host_array_ptr(zero), runtime=runtime)

        graph = 0
        stream = runtime.stream_create()
        try:
            self._set_full_attention_position_device(position, stream=stream)
            runtime.stream_synchronize(stream)
            runtime.stream_begin_capture(stream)
            try:
                with gemv_decode_session(self.use_gemv_decode):
                    for offset in range(steps_per_replay):
                        self._step_from_device_token(
                            position=position + offset,
                            advance_position=True,
                            stream=stream,
                            record_output_ptr=None if generated_buf is None else generated_buf.ptr,
                            record_index_ptr=None if generated_index_buf is None else generated_index_buf.ptr,
                            record_capacity=record_steps,
                        )
                graph = runtime.stream_end_capture(stream)
            except Exception:
                try:
                    runtime.stream_end_capture(stream)
                except Exception:
                    pass
                raise
            graph_exec = runtime.graph_instantiate(graph)
        except Exception:
            if graph:
                try:
                    runtime.graph_destroy(graph)
                except Exception:
                    pass
            runtime.stream_destroy(stream)
            if generated_index_buf is not None:
                free(generated_index_buf, runtime=runtime)
            if generated_buf is not None:
                free(generated_buf, runtime=runtime)
            raise
        return Qwen35GGUFDecodeGraph(
            session=self,
            graph=graph,
            graph_exec=graph_exec,
            stream=stream,
            position=int(position),
            steps_per_replay=int(steps_per_replay),
            max_replay_steps=replay_span,
            generated=generated_buf,
            generated_index=generated_index_buf,
            record_steps=int(record_steps),
            bucket_key=bucket_key,
        )

    def _step_from_device_token(
        self,
        *,
        position: int,
        advance_position: bool,
        stream: int,
        record_output_ptr: int | None = None,
        record_index_ptr: int | None = None,
        record_capacity: int = 0,
    ) -> None:
        if self._lm_out_index is None:
            raise RuntimeError("GGUF resident lm-head buffers are closed")
        if self.scratch is None:
            raise RuntimeError("GGUF resident session is closed")
        if position < 0 or position >= self.scratch.max_positions:
            raise ValueError(f"position {position} outside GGUF resident cache capacity {self.scratch.max_positions}")
        # Keep host-side guards in the layer helpers aligned with the graph step.
        # The device scalar is advanced by the captured runtime-state kernel.
        self.scratch.position_host[0] = int(position)
        self.scratch.context_host[0] = int(position) + 1
        self._set_token_embedding_from_ptr(self._lm_out_index.ptr, stream=stream)
        hidden_ptr = self._run_current_hidden_to_final_hidden(position=position, stream=stream)
        self._sample_device_from_hidden(hidden_ptr, stream=stream)
        if record_output_ptr is not None:
            if record_index_ptr is None:
                raise ValueError("record_index_ptr is required when recording GGUF decode graph outputs")
            record_i64_scalar_indexed(
                self._lm_out_index.ptr,
                record_output_ptr,
                record_index_ptr,
                int(record_capacity),
                stream=stream,
                library=self._runtime_state_library,
                runtime=self.runtime or get_hip_runtime(),
            )
        if advance_position:
            advance_decode_position_i64(
                self.scratch.position_buf.ptr,
                self.scratch.context_buf.ptr,
                stream=stream,
                library=self._runtime_state_library,
                runtime=self.runtime or get_hip_runtime(),
            )

    def close(self) -> None:
        runtime = self.runtime or get_hip_runtime()
        for buffer in reversed(self._buffers):
            if buffer is not None:
                free(buffer, runtime=runtime)
        self._buffers = ()
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
        self._host_token_embedding_reader = None
        self._host_token_embedding_raw = None
        self._host_token_embedding_cache = {}
        self.host_token_embedding_enabled = False
        self.host_token_embedding_reason = None

    def __enter__(self) -> "Qwen35GGUFResidentSession":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


@dataclass
class Qwen35GGUFDecodeGraph:
    session: Qwen35GGUFResidentSession
    graph: int
    graph_exec: int
    stream: int
    position: int
    steps_per_replay: int = 1
    max_replay_steps: int = 1
    generated: DeviceBuffer | None = None
    generated_index: DeviceBuffer | None = None
    record_steps: int = 0
    bucket_key: Qwen35GGUFDecodeGraphBucketKey | None = None
    closed: bool = False

    def replay(self, steps: int) -> None:
        if self.closed:
            raise RuntimeError("GGUF decode graph is closed")
        if steps < 0:
            raise ValueError("steps must be non-negative")
        if self.steps_per_replay <= 0:
            raise ValueError("steps_per_replay must be positive")
        if steps > self.max_replay_steps:
            raise ValueError("steps exceed captured max_replay_steps")
        if self.record_steps and steps > self.record_steps:
            raise ValueError("steps exceed decode graph record capacity")
        if steps % self.steps_per_replay != 0:
            raise ValueError("steps must be divisible by steps_per_replay")
        launches = steps // self.steps_per_replay
        for _ in range(launches):
            self.session.runtime.graph_launch(self.graph_exec, self.stream)  # type: ignore[union-attr]
        self.session.runtime.stream_synchronize(self.stream)  # type: ignore[union-attr]
        self.session._position = self.position + steps
        if self.session.scratch is not None:
            self.session.scratch.position_host[0] = self.session._position
            self.session.scratch.context_host[0] = self.session._position + 1

    def read_sample(self) -> Qwen35GGUFNextTokenProbeResult:
        if self.closed:
            raise RuntimeError("GGUF decode graph is closed")
        return self.session._read_sample()

    def read_generated_token_ids(self, count: int | None = None) -> list[int]:
        if self.closed:
            raise RuntimeError("GGUF decode graph is closed")
        if self.generated is None:
            raise RuntimeError("GGUF decode graph was captured without generated-token recording")
        rows = int(self.record_steps if count is None else count)
        if rows < 0 or rows > self.record_steps:
            raise ValueError("count outside decode graph record capacity")
        host = np.empty((rows,), dtype=np.int64)
        copy_device_to_host(
            host_array_ptr(host),
            DeviceBuffer(self.generated.ptr, rows * DType.INT64.itemsize),
            runtime=self.session.runtime or get_hip_runtime(),
        )
        return [int(item) for item in host.tolist()]

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        runtime = self.session.runtime or get_hip_runtime()
        runtime.graph_exec_destroy(self.graph_exec)
        runtime.graph_destroy(self.graph)
        if self.stream:
            runtime.stream_destroy(self.stream)
        if self.generated_index is not None:
            free(self.generated_index, runtime=runtime)
            self.generated_index = None
        if self.generated is not None:
            free(self.generated, runtime=runtime)
            self.generated = None

    def __enter__(self) -> "Qwen35GGUFDecodeGraph":
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
    moe_group_counts_zero: np.ndarray
    moe_scatter_offsets_zero: np.ndarray
    moe_wmma_total_host: np.ndarray
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
        linear_qkv_bf16_bytes = rows * runner.linear_qkv_width * 2
        linear_qkv_f32_bytes = rows * runner.linear_qkv_width * 4
        linear_z_bytes = rows * cfg.ssm_inner_size * 2
        linear_ab_bytes = rows * cfg.ssm_time_step_rank * 2
        recurrent_f32_bytes = rows * cfg.ssm_inner_size * 4
        prefill_scalar_bytes = rows * cfg.ssm_time_step_rank * 4
        cache_nbytes = max_positions * cfg.head_count_kv * cfg.key_length * 2 if allocate_kv_cache else 0
        block_table_arr = np.tile(np.arange(blocks, dtype=np.int32), (rows, 1))
        positions_arr = np.arange(rows, dtype=np.int64)
        context_arr = positions_arr + np.int64(1)
        cu_arr = np.asarray([0, rows], dtype=np.int32)
        atomic_arr = np.asarray([0], dtype=np.int32)
        cos_arr, sin_arr = _rope_tables(
            max_positions=rows,
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
            "cu_q": buf(cu_arr.nbytes),
            "cu_k": buf(cu_arr.nbytes),
            "softmax_lse": buf(cfg.head_count * rows * 4),
            "atomic": buf(atomic_arr.nbytes),
        }
        copy_host_to_device(fields["block_table"], host_array_ptr(block_table_arr), runtime=runtime)
        copy_host_to_device(fields["positions"], host_array_ptr(positions_arr), runtime=runtime)
        copy_host_to_device(fields["context_counts"], host_array_ptr(context_arr), runtime=runtime)
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
        return cls(
            **fields,
            rows=rows,
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
            moe_group_counts_zero=moe_group_counts_zero,
            moe_scatter_offsets_zero=moe_scatter_offsets_zero,
            moe_wmma_total_host=moe_wmma_total_host,
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
    full_bf16_mirror_key_caches: tuple[object | None, ...]
    full_bf16_mirror_value_caches: tuple[object | None, ...]
    full_k_scale_caches: tuple[object | None, ...]
    full_v_scale_caches: tuple[object | None, ...]
    full_kv_scale_metadata: tuple[KVScaleMetadata | None, ...]
    kv_storage_dtype: DType
    kv_scale_dtype: DType
    kv_scale_granularity: str
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
        kv_storage_dtype: str | DType = DType.BF16,
        kv_scale_dtype: str | DType = DType.FP16,
        kv_scale_granularity: str = "per_token_head",
        int8_bf16_prefix_full_attention_layers: int = 0,
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
        if kv_scale_granularity != "per_token_head":
            raise ValueError("GGUF INT8 KV scale granularity must be per_token_head")
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
        ffn_bytes = runner.ffn_size * 2
        moe_lane_count = max(1, int(cfg.expert_used_count)) if cfg.is_moe else 1
        moe_top_k = max(1, int(cfg.expert_used_count))
        moe_experts = max(1, int(cfg.expert_count))
        moe_shared_ffn = max(1, int(cfg.expert_shared_feed_forward_length or runner.ffn_size or 1))
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
        full_bf16_mirror_key_caches: list[object | None] = []
        full_bf16_mirror_value_caches: list[object | None] = []
        full_k_scale_caches: list[object | None] = []
        full_v_scale_caches: list[object | None] = []
        full_kv_scale_metadata: list[KVScaleMetadata | None] = []
        state_buffers: list[object] = []
        cache_buffers: list[object] = []
        int8_bf16_prefix_full_attention_layers = max(0, int(int8_bf16_prefix_full_attention_layers))
        int8_cache_nbytes = max_positions * cfg.head_count_kv * cfg.key_length * DType.INT8.itemsize
        bf16_cache_nbytes = max_positions * cfg.head_count_kv * cfg.key_length * DType.BF16.itemsize
        mirror_bf16_nbytes = bf16_cache_nbytes
        short_int8_bf16_mirror = (
            kv_storage == DType.INT8_PER_TOKEN_HEAD
            and max_positions <= _GGUF_INT8_SHORT_BF16_MIRROR_MAX_POSITIONS
        )
        scale_shape = (block_count, block_size, cfg.head_count_kv)
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
                layer_uses_int8 = kv_storage == DType.INT8_PER_TOKEN_HEAD and (
                    full_attention_index >= int8_bf16_prefix_full_attention_layers
                )
                cache_nbytes = int8_cache_nbytes if layer_uses_int8 else bf16_cache_nbytes
                key_cache = buf(cache_nbytes)
                value_cache = buf(cache_nbytes)
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
            full_bf16_mirror_key_caches=tuple(full_bf16_mirror_key_caches),
            full_bf16_mirror_value_caches=tuple(full_bf16_mirror_value_caches),
            full_k_scale_caches=tuple(full_k_scale_caches),
            full_v_scale_caches=tuple(full_v_scale_caches),
            full_kv_scale_metadata=tuple(full_kv_scale_metadata),
            kv_storage_dtype=kv_storage,
            kv_scale_dtype=scale_dtype,
            kv_scale_granularity=kv_scale_granularity,
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
    down_fn(
        scratch.ffn_intermediate.ptr,
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
    return "tiles" if weight.spec.quant_key.endswith("_t16_v1") else "raw"


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
    stream: int,
    runtime: HipRuntime,
) -> bool:
    if weight_a.spec.quant_key == "gguf_q4_k" and weight_b.spec.quant_key == "gguf_q4_k":
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
        return True
    if weight_a.spec.quant_key == "gguf_q4_k_t16_v1" and weight_b.spec.quant_key == "gguf_q4_k_t16_v1":
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
    stream: int,
    runtime: HipRuntime,
) -> None:
    quant_key = weight.spec.quant_key
    if quant_key == "gguf_q4_k":
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
    allocation = "tiles" if quant_key.endswith("_t16_v1") else "raw"
    fn(
        x_ptr,
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
    "Qwen35GGUFDecodeGraph",
    "Qwen35GGUFDecodeGraphBucketKey",
    "Qwen35GGUFDecodeGraphWeightRole",
    "Qwen35GGUFFullAttentionPrefillResult",
    "Qwen35GGUFFullStackRunner",
    "Qwen35GGUFNextTokenProbeResult",
    "Qwen35GGUFOneLayerProbe",
    "Qwen35GGUFFastPathSafety",
    "Qwen35GGUFResidentSession",
    "build_qwen35_gguf_decode_graph_bucket_key",
    "qwen35_gguf_decode_graph_active_symbol_groups",
    "qwen35_gguf_decode_graph_weight_roles",
    "resolve_qwen35moe_fastpath_safety",
]
