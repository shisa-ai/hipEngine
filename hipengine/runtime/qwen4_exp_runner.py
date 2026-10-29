"""Dedicated Qwen4Exp GGUF runtime composition.

This module begins with the strict gated-residual read owner used by every
Qwen3.8-Flash-Next token mixer and MoE. It is model-specific composition over
model-neutral GGUF projections and registered gfx11 primitives; it does not add
architecture branches to the engine or dispatch layer.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field, replace
import contextvars
import math
import os
import time
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.hip import HipMemcpyKind, HipRuntime, get_hip_runtime
from hipengine.core.memory import (
    DeviceBuffer,
    copy_device_to_host,
    copy_host_array_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.core.tensor import Tensor
from hipengine.kernels.backends import (
    backend_package_capability,
    load_backend_kernel_package,
)
from hipengine.kernels.hip_gfx1100.attention.qwen4_exp_qsa_flash import (
    qwen4_exp_qsa_flash_prefill,
)
from hipengine.kernels.hip_gfx1100.attention.paged_attn_decode import (
    qwen35_paged_full_attn_decode_context_bf16_batch_fixed256_spans,
    qwen35_paged_full_attn_decode_context_bf16_batch_fixed256_promoted_spans,
    qwen35_paged_full_attn_decode_context_bf16_batch_spans,
    qwen35_paged_full_attn_decode_context_bf16_spans,
)
from hipengine.kernels.hip_gfx1100.attention.paged_kv_write import (
    qwen35_write_paged_kv_f32_batch_spans,
    qwen35_write_paged_kv_f32_spans,
)
from hipengine.kernels.hip_gfx1100.attention.qwen4_exp_qsa import (
    qwen4_exp_qsa_gate_context_f32,
    qwen4_exp_qsa_gate_context_strided_f32,
    qwen4_exp_qsa_norm_rope_f32,
    qwen4_exp_qsa_norm_rope_rows_f32,
    qwen4_exp_qsa_norm_mrope_f32,
    qwen4_exp_qsa_norm_mrope_rows_f32,
    qwen4_exp_qsa_pool_norm_rope_f32,
    qwen4_exp_qsa_score_f32,
    qwen4_exp_qsa_score_rows_f32,
    qwen4_exp_qsa_topk_mask_rows_f32,
    qwen4_exp_qsa_scatter_index_key_device_position_f32,
    qwen4_exp_qsa_scatter_index_keys_f32,
    qwen4_exp_qsa_sparse_attention_paged_bf16_f32,
    qwen4_exp_qsa_sparse_attention_paged_bf16_ordered_f32,
    qwen4_exp_qsa_sparse_attention_paged_bf16_ordered_v2_f32,
    qwen4_exp_qsa_sparse_attention_paged_bf16_fused_v2_f32,
    qwen4_exp_qsa_sparse_attention_paged_bf16_rows_f32,
    qwen4_exp_qsa_sparse_attention_paged_bf16_wave32_f32,
    qwen4_exp_qsa_sparse_attention_paged_bf16_rows_wave32_f32,
    qwen4_exp_qsa_split_norm_rope_f32,
    qwen4_exp_qsa_split_norm_rope_rows_f32,
    qwen4_exp_qsa_split_norm_mrope_f32,
    qwen4_exp_qsa_split_norm_mrope_rows_f32,
    qwen4_exp_qsa_topk_expand_f32_i64,
    qwen4_exp_qsa_topk_expand_rows_f32_i64,
)
from hipengine.kernels.hip_gfx1100.convert.cast import f32_to_bf16
from hipengine.kernels.hip_gfx1100.fused.paro_combine import (
    shared_gate_combine_f32_shared_out_bf16,
    weighted_sum_batch_out_bf16_f32w,
    weighted_lanes_sum_out_bf16_f32w,
    weighted_lanes_sum_shared_gate_combine_batch_out_bf16_f32w,
    weighted_sum_out_bf16_f32w,
)
from hipengine.kernels.hip_gfx1100.fused.paro_silu import (
    silu_mul_dual_out_bf16,
    silu_mul_separate_out_bf16,
)
from hipengine.kernels.hip_gfx1100.moe.group_scatter import (
    qwen35_moe_group_count,
    qwen35_moe_group_prefix,
    qwen35_moe_group_prefix_active,
    qwen35_moe_group_scatter_gather_lowp,
    qwen35_moe_group_scatter,
    qwen35_moe_wmma_tile_map,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_t16_selected_prefill import (
    gguf_q4_k_t16_selected_dual_wmma_prefill_compact32_bf16_bf16_out,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_k_t16_selected_prefill import (
    gguf_q5_1_t16_selected_expert_major_wmma_comp_bf16_bf16_out,
)
from hipengine.kernels.hip_gfx1100.linear.lm_head import (
    argmax_f32,
    lm_head_argmax_stage1_blocks,
)
from hipengine.kernels.hip_gfx1100.linear_attn.conv import (
    qwen35_linear_attn_conv_decode_f32,
)
from hipengine.kernels.hip_gfx1100.attention.qwen4_exp_qsa_wmma import (
    qsa_wmma_geometry_supported,
    qsa_wmma_mask_words,
    qwen4_exp_qsa_wmma_attention_f32,
    qwen4_exp_qsa_wmma_gather_kv_f16,
    qwen4_exp_qsa_wmma_selection_mask,
)
from hipengine.kernels.hip_gfx1100.quant.qwen4_exp_routed_f16 import (
    ROUTED_F16_GATE_UP_VARIANT,
    ROUTED_F16_LINEAR_VARIANT,
    qwen4_exp_routed_f16_tile_map,
    routed_f16_tile_capacity,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_gemv import (
    gguf_q4_k_quantize_bf16_q8_1,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_selected_prefill import (
    gguf_q4_k_selected_dual_grouped_rowbatch8_bf16_bf16_out,
    gguf_q4_k_selected_dual_grouped_rowbatch8_out4_bf16_bf16_out,
    gguf_q4_k_selected_dual_grouped_rowbatch8_out4_expertgrid64_bf16_bf16_out,
    gguf_q4_k_selected_dual_sparse_exact_repair_bf16,
    gguf_q4_k_selected_dual_wmma_iu8_prefill_bf16_bf16_out,
    gguf_q4_k_selected_dual_wmma_iu8_risk_prefill_bf16_bf16_out,
    gguf_q4_k_selected_dual_wmma_prefill_compact_bf16_bf16_out,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_k_selected_prefill import (
    gguf_q5_k_selected_wmma_prefill_compact_bf16_bf16_out,
)
from hipengine.kernels.hip_gfx1100.quant.qwen4_exp_q5_1 import (
    qwen4_exp_gather_bf16_lanes,
    qwen4_exp_q5_1_selected_grouped_prefill_compact_rowbatch8_bf16_bf16_out,
    qwen4_exp_q5_1_selected_grouped_prefill_compact_rowbatch8_out8_bf16_bf16_out,
    qwen4_exp_q5_1_selected_grouped_prefill_compact_rowbatch8_out8_expertgrid64_bf16_bf16_out,
    qwen4_exp_q5_1_selected_grouped_wmma_prefill_compact_bf16_bf16_out,
    qwen4_exp_q5_1_selected_sparse_exact_repair_row_publish_bf16,
    qwen4_exp_q5_1_selected_wmma_iu8_risk_prefill_bf16_bf16_out,
)
from hipengine.kernels.hip_gfx1100.linear_attn.qwen4_exp_gdn import (
    qwen4_exp_gdn_decode_f32,
    qwen4_exp_gdn_peer_prefill_f32,
    qwen4_exp_gdn_prefill_columnwarps_f32,
    qwen4_exp_gdn_prefill_f32,
    qwen4_exp_gdn_prefill_tiled16_f32,
)
from hipengine.kernels.hip_gfx1100.fused.qwen4_exp_gr import (
    qwen4_exp_gated_mean_f32,
    qwen4_exp_gated_mean_sigmoid_f32,
    qwen4_exp_gr_inject_bf16_f32,
    qwen4_exp_gr_write_bf16_bf16,
    qwen4_exp_grouped_rmsnorm_bf16_f16_scale,
    qwen4_exp_grouped_rmsnorm_bf16_f16_scale_inject,
    qwen4_exp_gr_write_norm_inject,
    qwen4_exp_grouped_rmsnorm_bf16_f32,
    _check_norm_publication,
    qwen4_exp_grouped_rmsnorm_f32,
    qwen4_exp_gr_write_bf16_f32,
    qwen4_exp_repeat_bf16_branches,
    qwen4_exp_scaled_silu_f32,
    qwen4_exp_silu_mul_f32,
    qwen4_exp_sigmoid_f32,
)
from hipengine.kernels.hip_gfx1100.fused.qwen4_exp_ple import (
    qwen4_exp_ple_add_delta_bf16_f32,
    qwen4_exp_ple_dilated_depthwise_conv_f32,
    qwen4_exp_ple_repeat_gated_value_f32,
    qwen4_exp_ple_signed_sqrt_gate_f32,
)
from hipengine.loading.materialize import float_array_to_bf16_bits
from hipengine.runtime.gguf_linear import (
    GGUF_ACTIVATION_F32,
    _invalidate_wide_f16_publication,
    _q8_dense_wide_enabled,
    _wide_f16_consumer_key,
    dense_wide_tail_session,
    norm_f16_publication_session,
    wide_f16_activation_pair,
    wide_f16_activation_enabled,
    wide_f16_activation_publication,
    wide_f16_activation_session,
    wide_f16_activation_workspace,
    wide_f16_pair_enabled,
    wide_f16_pair_session,
    wide_f16_publication_keys,
    raw_k_prefill_rowbatch_session,
    raw_k_prefill_variant_session,
    wmma_prefill_session,
    wmma_prefill_weight_filter_session,
    GGUF_OUTPUT_F32,
    launch_gguf_linear,
    launch_gguf_linear_decode_rows,
    q8_mmq_lane_session,
    q8_mmq_prefill_session,
    resolve_q8_mmq_prefill_policy,
)
from hipengine.kernels.cpu_reference.qwen4_exp import (
    PLEHashState,
    QSASelection,
    ple_hash_rows,
    qsa_index_scores,
    qsa_prepare_index_keys,
    qsa_select_positions,
)
from hipengine.kernels.registry import KernelKey, is_registered, resolve
from hipengine.kernels.registry import generation as _registry_generation
from hipengine.kernels.hip_gfx1100.linear.f16_mirror import (
    declare_f16_mirror,
    invalidate_f16_mirror,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_k_gemv import (
    qwen4_exp_gr_down_pack8_decode,
    qwen4_exp_gr_up_mean_pack8_decode,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q8_0_dense_wide import (
    _WRAPPERS as _GGUF_Q8_0_DENSE_WIDE_WRAPPERS,
    f32_to_f16 as gguf_q8_0_f32_to_f16,
    qwen4_exp_gr_up_mean_f16in_f32_out,
    qwen4_exp_gr_up_mean_residual_f16in_f32_out,
    gguf_q8_0_dense_wide256_f32_f32_out,
)
gguf_q8_0_dense_wide128x128_f16in_f32_f32_out = _GGUF_Q8_0_DENSE_WIDE_WRAPPERS[
    "dense_wide128x128_f16in_f32_f32_out"
]
from hipengine.kernels.hip_gfx1100.quant.gguf_k_gemv import (
    gguf_q8_0_iu8_wmma_prefill_f32_f32,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q8_0_prefill import (
    normalize_wmma_tile_pair,
    wmma_tile_policy,
)
from hipengine.kvcache import KVLiveSpans
from hipengine.loading.qwen4_exp_materialize import Qwen4ExpResidentWeights
from hipengine.loading.qwen4_exp_gguf import Qwen4ExpGGUFConfig
from hipengine.loading.qwen4_exp_scratch import qwen4_exp_wide_f16_activation_nbytes
from hipengine.runtime.gguf_weight import GGUFDeviceWeight
from hipengine.runtime.moe_graph import MoeGraphCache


def _launch_qwen4_exp_dense_pair(
    weight_a, weight_b, x_ptr, out_a_ptr, out_b_ptr, rows, in_features,
    out_features_a, out_features_b, *, stream=0, runtime=None,
) -> None:
    """Keep normal dispatch; optionally loan staging to this immediate pair."""
    scope = (
        wide_f16_activation_pair(weight_a, weight_b, x_ptr, out_a_ptr, out_b_ptr,
            rows=rows, in_features=in_features, out_features_a=out_features_a,
            out_features_b=out_features_b, stream=stream, runtime=runtime)
        if rows > 1 and wide_f16_pair_enabled()
        else contextlib.nullcontext()
    )
    with scope:
        for weight, output, n in ((weight_a, out_a_ptr, out_features_a),
                                  (weight_b, out_b_ptr, out_features_b)):
            launch_gguf_linear(weight, x_ptr, output, rows, in_features, n,
                activation_dtype=GGUF_ACTIVATION_F32, output_dtype=GGUF_OUTPUT_F32,
                stream=stream, runtime=runtime)


def _qwen4_exp_layer_scope_contains(slot_path: str, scope: str | None) -> bool:
    """Does one weight's layer fall inside a ``"35-47"``/``"all"`` scope?

    The parser is shared by every layer-gated selector, so an empty or absent
    scope means "every layer" here. A selector whose absence means "off" owns
    that reading itself (see :func:`_qwen4_exp_gdn_peer_layer_allowed`), which
    keeps the parse and the gate from disagreeing.
    """

    parts = slot_path.split(".")
    if len(parts) <= 2 or parts[0] != "layers":
        return False
    try:
        layer = int(parts[1])
    except ValueError:
        return False
    raw = "" if scope is None else scope
    if raw in {"", "all"}:
        return True
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            low, high = token.split("-", 1)
            if int(low) <= layer <= int(high):
                return True
        elif int(token) == layer:
            return True
    return False


def _qwen4_exp_layer_allowed(
    weight: GGUFDeviceWeight,
    *,
    env_name: str,
    default: str,
) -> bool:
    return _qwen4_exp_layer_scope_contains(
        weight.spec.slot_path, os.environ.get(env_name, default)
    )


def _qwen4_exp_q4_iu8_layer_allowed(weights: Mapping[str, GGUFDeviceWeight]) -> bool:
    """iu8-WMMA selected-dual prefill layer set (dp4a class, screened suffix)."""

    return _qwen4_exp_layer_allowed(
        weights["expert_gate"],
        env_name="HIPENGINE_QWEN4_EXP_Q4_IU8_LAYERS",
        default="35-47",
    )


#: The certified global layer scope of the GDN peer prefill owner. The
#: production plan publishes it on the owning runner (see
#: :func:`gdn_peer_prefill`); ``True`` restores it for a diagnostic arm.
GDN_PEER_CERTIFIED_SCOPE = ",".join(map(str, range(35, 48)))


def qwen4_exp_gdn_peer_prefill_selected(
    *, head_dim: int, published: str | None,
) -> bool:
    """Is the peer prefill owner in scope for this chunk's head width?

    Geometry and the published gate only. ``published`` is the owning runner's
    layer scope (see :func:`gdn_peer_prefill`), so an unbound runner or a plan
    that publishes no scope keeps the serial sigmoid owners. The layer test is
    :func:`_qwen4_exp_gdn_peer_layer_allowed`, and the columnwarp parent stays
    ahead of both inside its own gate.
    """

    return head_dim == 128 and bool(published)


def _qwen4_exp_gdn_peer_layer_allowed(
    weights: Mapping[str, GGUFDeviceWeight],
    *,
    published: str | None,
) -> bool:
    """Does this layer sit inside the peer owner's published scope?

    Unlike the shared parser, an empty or absent scope means "no layer": the
    published scope *is* the gate, so an unbound runner binds the serial owners
    rather than every layer.
    """

    if not published:
        return False
    return _qwen4_exp_layer_scope_contains(
        weights["attn_qkv"].spec.slot_path, published
    )


def qwen4_exp_gdn_register_prefill_selected(
    *, backend: str, rows: int, num_k_heads: int, num_v_heads: int, head_dim: int,
) -> bool:
    return (
        rows >= 2 and (num_k_heads, num_v_heads, head_dim) == (16, 48, 128)
        and os.environ.get("HIPENGINE_QWEN4_EXP_GDN_REGISTER_PREFILL", "0")
        not in {"", "0", "false", "False"}
        and is_registered(KernelKey(
            backend, "gdn_recurrence_norm_gate", "f32_state",
            "qwen4exp_sigmoid_register_prefill",
        ))
    )


def qwen4_exp_gdn_tile16_prefill_selected(
    *,
    rows: int,
    num_k_heads: int,
    num_v_heads: int,
    head_dim: int,
) -> bool:
    """Does the binding Hk16/Hv48/D128 envelope fit the tile-16 owner?

    Geometry only. Which owner serves the chunk is the owning runner's
    ``gdn_tile16_prefill`` decision (see :func:`gdn_tile16_prefill`), so a
    diagnostic arm binds the columnwarp parent on the runner it measures
    instead of writing a process-wide switch. Non-envelope shapes keep that
    parent either way, and the serial strict route stays registered below both.
    """

    return (
        rows >= 16
        and num_k_heads == 16
        and num_v_heads in (32, 48)
        and head_dim == 128
    )


def qwen4_exp_gdn_tile16_owner_admitted(
    *,
    rows: int,
    num_k_heads: int,
    num_v_heads: int,
    head_dim: int,
    supports_any_rows: bool,
    enabled: bool = True,
) -> bool:
    """Does the tile-16 branch own this chunk, or the columnwarp parent?

    The envelope predicate covers the tile-16 bodies, which need at least 16
    rows. A row-split owner serves any width from two rows up, so it also takes
    verifier and small-batch chunks where the columnwarp parent costs 147 us per
    layer at four rows. ``enabled`` is the owning runner's decision: False
    leaves every shape to that parent, including the row-split widths.
    """

    if not enabled:
        return False
    if qwen4_exp_gdn_tile16_prefill_selected(
        rows=rows,
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
        head_dim=head_dim,
    ):
        return True
    return rows >= 2 and head_dim == 128 and bool(supports_any_rows)


def stage_qwen4_exp_ple_rows(
    staging: object,
    rows: Sequence[Sequence[int]],
    *,
    hidden: int,
) -> np.ndarray:
    """Gather one prompt chunk into a single owned PLE row matrix."""

    width = int(hidden)
    if width <= 0:
        raise ValueError("PLE hidden width must be positive")
    count = len(rows)
    if count == 0:
        return np.empty((0, width), dtype=np.float32)
    indices = np.asarray(rows, dtype=np.int64)
    if indices.ndim != 2 or indices.shape[0] != count:
        raise ValueError("PLE prompt row indices must have shape [rows, heads]")
    staged = np.asarray(staging.stage(indices.reshape(-1)), dtype=np.float32)
    if staged.size != count * width:
        raise ValueError("PLE staged values do not match prompt rows × hidden")
    # The result aliases the active pinned ring buffer until the next stage().
    # The sole runtime consumer performs its synchronous H2D copy immediately.
    return staged.reshape(count, width)


@dataclass(frozen=True)
class Qwen4ExpHostQSAIndexSnapshot:
    raw_keys: np.ndarray
    count: int


@dataclass
class Qwen4ExpHostQSAIndexState:
    raw_keys: np.ndarray
    compression_ratio: int
    block_budget: int
    count: int = 0

    @classmethod
    def allocate(
        cls,
        *,
        capacity: int,
        index_dim: int,
        compression_ratio: int,
        block_budget: int,
    ) -> "Qwen4ExpHostQSAIndexState":
        if capacity <= 0 or index_dim <= 0:
            raise ValueError("QSA index capacity and dimension must be positive")
        if compression_ratio <= 0 or block_budget <= 0:
            raise ValueError("QSA compression ratio and budget must be positive")
        return cls(
            np.empty((capacity, index_dim), dtype=np.float32),
            int(compression_ratio),
            int(block_budget),
        )

    @property
    def capacity(self) -> int:
        return int(self.raw_keys.shape[0])

    @property
    def index_dim(self) -> int:
        return int(self.raw_keys.shape[1])

    def append(self, key: object, *, position: int) -> None:
        if int(position) != self.count:
            raise ValueError("QSA index position must equal contiguous count")
        if self.count >= self.capacity:
            raise ValueError("QSA index capacity exceeded")
        value = np.asarray(key, dtype=np.float32)
        if value.shape != (self.index_dim,):
            raise ValueError("QSA index key must have shape [index_dim]")
        self.raw_keys[self.count] = value
        self.count += 1

    def snapshot(self) -> Qwen4ExpHostQSAIndexSnapshot:
        return Qwen4ExpHostQSAIndexSnapshot(self.raw_keys[: self.count].copy(), self.count)

    def restore(self, snapshot: Qwen4ExpHostQSAIndexSnapshot) -> None:
        if snapshot.count < 0 or snapshot.count > self.capacity:
            raise ValueError("QSA index snapshot count exceeds capacity")
        if snapshot.raw_keys.shape != (snapshot.count, self.index_dim):
            raise ValueError("QSA index snapshot shape mismatch")
        self.raw_keys[: snapshot.count] = snapshot.raw_keys
        self.count = int(snapshot.count)

    def select(
        self,
        prepared_query: object,
        *,
        query_position: int,
        key_norm_weight: object,
        rotary_dim: int,
        theta: float,
        eps: float = 1e-6,
    ) -> QSASelection:
        if self.count <= 0 or int(query_position) != self.count - 1:
            raise ValueError("QSA query position must identify the latest index key")
        pooled = qsa_prepare_index_keys(
            self.raw_keys[: self.count],
            np.arange(self.count, dtype=np.int64),
            key_norm_weight,
            compression_ratio=self.compression_ratio,
            rotary_dim=rotary_dim,
            theta=theta,
            eps=eps,
        )
        query = np.asarray(prepared_query, dtype=np.float32)
        if query.ndim != 2 or query.shape[1] != self.index_dim:
            raise ValueError("prepared QSA query must have shape [heads, index_dim]")
        scores = qsa_index_scores(query[None], pooled.keys)
        return qsa_select_positions(
            scores,
            pooled.block_starts,
            query_positions=[query_position],
            available_positions=np.arange(self.count),
            compression_ratio=self.compression_ratio,
            block_budget=self.block_budget,
        )


@dataclass(frozen=True)
class Qwen4ExpDecodeStateSnapshot:
    buffers: Mapping[str, np.ndarray]


@dataclass
class Qwen4ExpDecodeStateDeviceSnapshot:
    buffers: Mapping[str, DeviceBuffer]
    runtime: HipRuntime
    closed: bool = False

    @classmethod
    def allocate_like(
        cls, state: "Qwen4ExpDecodeState"
    ) -> "Qwen4ExpDecodeStateDeviceSnapshot":
        state._require_open()
        buffers: dict[str, DeviceBuffer] = {}
        try:
            for name, source in state.owned_buffers.items():
                buffers[name] = malloc(source.nbytes, runtime=state.runtime)
        except Exception:
            for buffer in reversed(tuple(buffers.values())):
                free(buffer, runtime=state.runtime)
            raise
        return cls(MappingProxyType(buffers), state.runtime)

    @property
    def nbytes_by_owner(self) -> Mapping[str, int]:
        return MappingProxyType({
            name: buffer.nbytes for name, buffer in self.buffers.items()
        })

    def close(self) -> None:
        if self.closed:
            return
        for buffer in reversed(tuple(self.buffers.values())):
            free(buffer, runtime=self.runtime)
        self.closed = True


@dataclass
class Qwen4ExpDecodeState:
    gdn_matrix: DeviceBuffer
    gdn_conv: DeviceBuffer
    ple_conv: DeviceBuffer
    residual: DeviceBuffer
    runtime: HipRuntime
    closed: bool = False

    @classmethod
    def allocate(
        cls,
        *,
        gdn_layers: int,
        gdn_value_heads: int,
        gdn_head_dim: int,
        gdn_conv_channels: int,
        gdn_conv_kernel: int,
        residual_branches: int,
        hidden: int,
        ple_conv_kernel: int,
        ple_dilation: int,
        runtime: HipRuntime | None = None,
    ) -> "Qwen4ExpDecodeState":
        dimensions = (
            gdn_layers,
            gdn_value_heads,
            gdn_head_dim,
            gdn_conv_channels,
            gdn_conv_kernel,
            residual_branches,
            hidden,
            ple_conv_kernel,
            ple_dilation,
        )
        if any(int(value) <= 0 for value in dimensions):
            raise ValueError("Qwen4Exp state dimensions must be positive")
        active_runtime = runtime or get_hip_runtime()
        sizes = (
            gdn_layers * gdn_value_heads * gdn_head_dim * gdn_head_dim * 4,
            gdn_layers * gdn_conv_kernel * gdn_conv_channels * 4,
            (ple_conv_kernel - 1) * ple_dilation * residual_branches * hidden * 4,
            residual_branches * hidden * 2,
        )
        buffers: list[DeviceBuffer] = []
        try:
            for size in sizes:
                buffers.append(malloc(size, runtime=active_runtime))
        except Exception:
            for buffer in reversed(buffers):
                free(buffer, runtime=active_runtime)
            raise
        state = cls(*buffers, runtime=active_runtime)
        state.zero()
        return state

    @property
    def owned_buffers(self) -> Mapping[str, DeviceBuffer]:
        return MappingProxyType(
            {
                "gdn_matrix": self.gdn_matrix,
                "gdn_conv": self.gdn_conv,
                "ple_conv": self.ple_conv,
                "residual": self.residual,
            }
        )

    @property
    def nbytes_by_owner(self) -> Mapping[str, int]:
        return MappingProxyType(
            {name: buffer.nbytes for name, buffer in self.owned_buffers.items()}
        )

    def zero(self) -> None:
        self._require_open()
        for buffer in self.owned_buffers.values():
            self.runtime.memset(buffer.ptr, 0, buffer.nbytes)

    def device_snapshot(
        self,
        snapshot: Qwen4ExpDecodeStateDeviceSnapshot | None = None,
    ) -> Qwen4ExpDecodeStateDeviceSnapshot:
        self._require_open()
        if snapshot is None:
            snapshot = Qwen4ExpDecodeStateDeviceSnapshot.allocate_like(self)
        elif snapshot.closed:
            raise RuntimeError("Qwen4Exp device state snapshot is closed")
        elif snapshot.nbytes_by_owner != self.nbytes_by_owner:
            raise ValueError("Qwen4Exp reusable device snapshot shape does not match")
        for name, source in self.owned_buffers.items():
            self.runtime.memcpy(
                snapshot.buffers[name].ptr,
                source.ptr,
                source.nbytes,
                HipMemcpyKind.DEVICE_TO_DEVICE,
            )
        return snapshot

    def restore_device_snapshot(
        self, snapshot: Qwen4ExpDecodeStateDeviceSnapshot
    ) -> None:
        self._require_open()
        if snapshot.closed:
            raise RuntimeError("Qwen4Exp device state snapshot is closed")
        if set(snapshot.buffers) != set(self.owned_buffers):
            raise ValueError("Qwen4Exp device state snapshot owner set does not match")
        for name, destination in self.owned_buffers.items():
            source = snapshot.buffers[name]
            if source.nbytes != destination.nbytes:
                raise ValueError(
                    f"Qwen4Exp device state snapshot size mismatch for {name}"
                )
            self.runtime.memcpy(
                destination.ptr,
                source.ptr,
                destination.nbytes,
                HipMemcpyKind.DEVICE_TO_DEVICE,
            )

    def snapshot(self) -> Qwen4ExpDecodeStateSnapshot:
        self._require_open()
        buffers: dict[str, np.ndarray] = {}
        for name, buffer in self.owned_buffers.items():
            host = np.empty(buffer.nbytes, dtype=np.uint8)
            copy_device_to_host(host_array_ptr(host), buffer, runtime=self.runtime)
            buffers[name] = host
        return Qwen4ExpDecodeStateSnapshot(MappingProxyType(buffers))

    def restore(self, snapshot: Qwen4ExpDecodeStateSnapshot) -> None:
        self._require_open()
        if set(snapshot.buffers) != set(self.owned_buffers):
            raise ValueError("Qwen4Exp state snapshot owner set does not match")
        for name, buffer in self.owned_buffers.items():
            host = np.ascontiguousarray(snapshot.buffers[name], dtype=np.uint8)
            if host.nbytes != buffer.nbytes:
                raise ValueError(f"Qwen4Exp state snapshot size mismatch for {name}")
            copy_host_array_to_device(buffer, host, runtime=self.runtime)

    def close(self) -> None:
        if self.closed:
            return
        for buffer in reversed(tuple(self.owned_buffers.values())):
            free(buffer, runtime=self.runtime)
        self.closed = True

    def _require_open(self) -> None:
        if self.closed:
            raise RuntimeError("Qwen4Exp decode state is closed")


@dataclass
class Qwen4ExpDenseAttentionState:
    key_cache: DeviceBuffer
    value_cache: DeviceBuffer
    block_table: DeviceBuffer
    position: DeviceBuffer
    context: DeviceBuffer
    append_spans: KVLiveSpans
    decode_spans: KVLiveSpans
    block_host: np.ndarray
    position_host: np.ndarray
    context_host: np.ndarray
    block_size: int
    max_positions: int
    runtime: HipRuntime
    closed: bool = False
    owns_position_context: bool = True

    @classmethod
    def allocate(
        cls,
        *,
        max_positions: int,
        block_size: int,
        kv_heads: int,
        head_dim: int,
        runtime: HipRuntime | None = None,
        position_view: DeviceBuffer | None = None,
        context_view: DeviceBuffer | None = None,
    ) -> "Qwen4ExpDenseAttentionState":
        if max_positions <= 0 or block_size <= 0 or kv_heads <= 0 or head_dim <= 0:
            raise ValueError("attention state dimensions must be positive")
        active_runtime = runtime or get_hip_runtime()
        blocks = (max_positions + block_size - 1) // block_size
        block_host = np.arange(blocks, dtype=np.int32)
        position_host = np.zeros(1, dtype=np.int64)
        context_host = np.ones(1, dtype=np.int64)
        buffers: list[DeviceBuffer] = []
        try:
            key_cache = malloc(
                blocks * block_size * kv_heads * head_dim * 2,
                runtime=active_runtime,
            )
            buffers.append(key_cache)
            value_cache = malloc(key_cache.nbytes, runtime=active_runtime)
            buffers.append(value_cache)
            block_table = malloc(block_host.nbytes, runtime=active_runtime)
            buffers.append(block_table)
            owns_position_context = True
            if position_view is None or context_view is None:
                position = malloc(position_host.nbytes, runtime=active_runtime)
                buffers.append(position)
                context = malloc(context_host.nbytes, runtime=active_runtime)
                buffers.append(context)
            else:
                if (
                    position_view.nbytes != position_host.nbytes
                    or context_view.nbytes != context_host.nbytes
                ):
                    raise ValueError(
                        "external QSA position/context views must be 8-byte int64 slots"
                    )
                position = position_view
                context = context_view
                owns_position_context = False
            copy_host_array_to_device(block_table, block_host, runtime=active_runtime)
            copy_host_array_to_device(position, position_host, runtime=active_runtime)
            copy_host_array_to_device(context, context_host, runtime=active_runtime)
            active_runtime.memset(key_cache.ptr, 0, key_cache.nbytes)
            active_runtime.memset(value_cache.ptr, 0, value_cache.nbytes)
        except Exception:
            for buffer in reversed(buffers):
                free(buffer, runtime=active_runtime)
            raise
        device = Device("hip", 0)
        block_tensor = Tensor.from_handle(
            block_table.ptr, block_host.shape, DType.INT32, device
        )
        position_tensor = Tensor.from_handle(
            position.ptr, position_host.shape, DType.INT64, device
        )
        context_tensor = Tensor.from_handle(
            context.ptr, context_host.shape, DType.INT64, device
        )
        append_spans = KVLiveSpans.paged_uniform(
            block_table=block_tensor,
            live_counts=position_tensor,
            max_live_count=max_positions - 1,
            storage_dtype=DType.BF16,
        )
        decode_spans = KVLiveSpans.paged_uniform(
            block_table=block_tensor,
            live_counts=context_tensor,
            max_live_count=max_positions,
            storage_dtype=DType.BF16,
        )
        return cls(
            key_cache, value_cache, block_table, position, context,
            append_spans, decode_spans, block_host, position_host, context_host,
            block_size, max_positions, active_runtime,
            owns_position_context=owns_position_context,
        )

    def written_cache_bytes(self) -> int:
        """Bytes of K/V storage that may hold data from the current request.

        Clearing K/V to the full allocated capacity costs time proportional to
        capacity rather than to work done, so at native capacity a short request
        pays to clear storage it never touched (RT-4: 29.182 ms at 128K, 99.6%
        wasted).  This returns the exact bound instead.

        Writes only ever occur below ``position``, so everything at or above
        ``round_up(position, block_size)`` has never been written and is already
        zero - either from the allocation memset or from an earlier bounded clear.
        Rounding up to a whole block is REQUIRED rather than conservative: the
        final partial block does contain written bytes.

        ``position_host`` is host-resident, so this needs no device read.
        """
        written = int(self.position_host[0])
        if written <= 0:
            return 0
        blocks = int(self.block_host.size)
        if blocks <= 0 or self.block_size <= 0:
            return 0
        bytes_per_block = self.key_cache.nbytes // blocks
        written_blocks = (written + self.block_size - 1) // self.block_size
        return min(written_blocks * bytes_per_block, self.key_cache.nbytes)

    def clear_written_cache(self) -> None:
        """Zero only the K/V storage this request could have written.

        Equivalent to clearing the full capacity by the invariant documented on
        :meth:`written_cache_bytes`, but proportional to work done.
        """
        bound = self.written_cache_bytes()
        if bound <= 0:
            return
        self.runtime.memset(self.key_cache.ptr, 0, bound)
        self.runtime.memset(self.value_cache.ptr, 0, bound)

    def set_position(self, value: int) -> None:
        if self.closed:
            raise RuntimeError("Qwen4Exp attention state is closed")
        position = int(value)
        if not 0 <= position < self.max_positions:
            raise ValueError("attention position exceeds state capacity")
        self.position_host[0] = position
        self.context_host[0] = position + 1
        copy_host_array_to_device(
            self.position, self.position_host, runtime=self.runtime
        )
        copy_host_array_to_device(
            self.context, self.context_host, runtime=self.runtime
        )

    def close(self) -> None:
        if self.closed:
            return
        owned: tuple[DeviceBuffer, ...] = (
            self.key_cache, self.value_cache, self.block_table,
        )
        if self.owns_position_context:
            owned = owned + (self.position, self.context)
        for buffer in reversed(owned):
            free(buffer, runtime=self.runtime)
        self.closed = True


def _qwen4_exp_bounded_kv_reset_enabled() -> bool:
    # RT-4: clear only the K/V bytes this request could have written instead of the
    # full allocated capacity.  Sound by the invariant on
    # ``Qwen4ExpDenseAttentionState.written_cache_bytes``: allocation zeroes the
    # whole cache and writes only ever land below ``position``, so bytes above the
    # bound are already zero.
    #
    # PROMOTED to default (2026-09-30).  Measured at 128K capacity: reset
    # 15.689/15.519 ms -> 1.133/1.741 ms, 14.17 ms saved per request, decode
    # identical in both regimes.  Exactness proven by parent parity on
    # ``scripts/qwen4exp_layer2_profile_gate.py``: production_baseline and
    # bounded_kv_reset produce an identical candidate_logits_sha256, with
    # identical strict hashes across the two runs showing the harness itself is
    # reproducible; the raw K/V buffers are byte-identical at all 12 attention
    # states across three arms (flag off, off, on).  Set
    # HIPENGINE_QWEN4_EXP_BOUNDED_KV_RESET=0 to restore the full-capacity clear
    # for rollback or bisection.
    return os.environ.get("HIPENGINE_QWEN4_EXP_BOUNDED_KV_RESET", "1") not in {
        "",
        "0",
        "false",
        "False",
    }


def _qwen4_exp_batched_position_enabled() -> bool:
    # Promoted (R7 wrapper-host route): one shared 192B H2D replaces the 24
    # per-layer 8B blocking set_position copies per decode step. Bit-exact
    # token streams; TG median -0.77 ms/token (~1.3%). Set
    # HIPENGINE_QWEN4_EXP_BATCHED_POSITION=0 to restore per-layer copies.
    return os.environ.get("HIPENGINE_QWEN4_EXP_BATCHED_POSITION", "1") != "0"


@dataclass
class Qwen4ExpQSAPrefillMetadata:
    block_tables: DeviceBuffer
    positions: DeviceBuffer
    context_counts: DeviceBuffer
    selected_positions: DeviceBuffer
    selected_counts: DeviceBuffer
    scores: DeviceBuffer
    block_tables_host: np.ndarray
    positions_host: np.ndarray
    context_counts_host: np.ndarray
    selected_positions_host: np.ndarray
    selected_counts_host: np.ndarray
    rows: int
    block_table_len: int
    selection_capacity: int
    score_blocks: int
    max_positions: int
    runtime: HipRuntime
    closed: bool = False
    _uploaded_contiguous: tuple[int, int] | None = None

    @classmethod
    def allocate(
        cls,
        attention_state: Qwen4ExpDenseAttentionState,
        *,
        rows: int,
        selection_capacity: int,
        score_blocks: int | None = None,
    ) -> "Qwen4ExpQSAPrefillMetadata":
        if attention_state.closed:
            raise RuntimeError("QSA prefill metadata requires an open attention state")
        count = int(rows)
        selected = int(selection_capacity)
        score_capacity = (
            attention_state.max_positions
            if score_blocks is None
            else int(score_blocks)
        )
        if count <= 0 or selected <= 0 or score_capacity <= 0:
            raise ValueError(
                "QSA prefill rows, selection capacity, and score blocks must be positive"
            )
        tables = np.ascontiguousarray(
            np.tile(attention_state.block_host, (count, 1)), dtype=np.int32
        )
        positions = np.zeros(count, dtype=np.int64)
        contexts = np.ones(count, dtype=np.int64)
        selected_positions = np.full((count, selected), -1, dtype=np.int64)
        selected_counts = np.zeros(count, dtype=np.int32)
        hosts = (tables, positions, contexts, selected_positions, selected_counts)
        buffers: list[DeviceBuffer] = []
        try:
            for host in hosts:
                buffer = malloc(host.nbytes, runtime=attention_state.runtime)
                buffers.append(buffer)
                copy_host_array_to_device(
                    buffer, host, runtime=attention_state.runtime
                )
            buffers.append(
                malloc(
                    count * score_capacity * DType.FP32.itemsize,
                    runtime=attention_state.runtime,
                )
            )
        except Exception:
            for buffer in reversed(buffers):
                free(buffer, runtime=attention_state.runtime)
            raise
        return cls(
            *buffers,
            *hosts,
            count,
            int(attention_state.block_host.size),
            selected,
            score_capacity,
            attention_state.max_positions,
            attention_state.runtime,
        )

    def set_contiguous(self, start_position: int, rows: int) -> None:
        if self.closed:
            raise RuntimeError("QSA prefill metadata is closed")
        start = int(start_position)
        count = int(rows)
        if count <= 0 or count > self.rows:
            raise ValueError("active QSA prefill rows exceed metadata capacity")
        if start < 0 or start + count > self.max_positions:
            raise ValueError("QSA prefill positions exceed attention capacity")
        self.positions_host[:count] = np.arange(start, start + count, dtype=np.int64)
        self.context_counts_host[:count] = self.positions_host[:count] + 1
        self.selected_positions_host[:count] = -1
        self.selected_counts_host[:count] = 0
        # Layers share these read-only position planes. Selection buffers are
        # layer-local scratch, so their host reset above must still happen.
        if self._uploaded_contiguous == (start, count):
            return
        # A partial copy failure invalidates even the previous cached pair.
        self._uploaded_contiguous = None
        copy_host_array_to_device(
            self.positions,
            self.positions_host,
            count * np.dtype(np.int64).itemsize,
            runtime=self.runtime,
        )
        copy_host_array_to_device(
            self.context_counts,
            self.context_counts_host,
            count * np.dtype(np.int64).itemsize,
            runtime=self.runtime,
        )
        self._uploaded_contiguous = (start, count)

    def spans(self, *, start_row: int, rows: int, decode: bool) -> KVLiveSpans:
        if self.closed:
            raise RuntimeError("QSA prefill metadata is closed")
        start = int(start_row)
        count = int(rows)
        if start < 0 or count <= 0 or start + count > self.rows:
            raise ValueError("QSA prefill span slice is outside metadata rows")
        device = Device("hip", 0)
        table_offset = start * self.block_table_len * np.dtype(np.int32).itemsize
        live_buffer = self.context_counts if decode else self.positions
        live_offset = start * np.dtype(np.int64).itemsize
        return KVLiveSpans.paged_uniform(
            block_table=Tensor.from_handle(
                self.block_tables.ptr + table_offset,
                (count, self.block_table_len),
                DType.INT32,
                device,
            ),
            live_counts=Tensor.from_handle(
                live_buffer.ptr + live_offset,
                (count,),
                DType.INT64,
                device,
            ),
            max_live_count=self.max_positions if decode else self.max_positions - 1,
            storage_dtype=DType.BF16,
        )

    def upload_selections(self, rows: int) -> None:
        count = int(rows)
        if count <= 0 or count > self.rows:
            raise ValueError("selection rows exceed QSA prefill metadata capacity")
        copy_host_array_to_device(
            self.selected_positions,
            self.selected_positions_host,
            count * self.selection_capacity * np.dtype(np.int64).itemsize,
            runtime=self.runtime,
        )
        copy_host_array_to_device(
            self.selected_counts,
            self.selected_counts_host,
            count * np.dtype(np.int32).itemsize,
            runtime=self.runtime,
        )

    def close(self) -> None:
        if self.closed:
            return
        for buffer in reversed(
            (
                self.block_tables,
                self.positions,
                self.context_counts,
                self.selected_positions,
                self.selected_counts,
                self.scores,
            )
        ):
            free(buffer, runtime=self.runtime)
        self.closed = True


def _qsa_select_starts_host(
    scores: object,
    block_starts: object,
    *,
    budget: int,
) -> np.ndarray:
    """Exact host control ordering for QSA blocks: score descending, then start.

    Retained as the readable statement of the tie contract the device radix
    selector must match (``qsa_topk_expand_f32_i64`` compacts the lowest block
    indices at an exact threshold). No engine path selects it: the promoted
    route is the device selector, and the CPU reference
    (``qsa_select_positions``) is what the gates compare against.
    """

    score = np.asarray(scores, dtype=np.float32)
    starts = np.asarray(block_starts, dtype=np.int64)
    count = int(budget)
    if score.ndim != 1 or starts.shape != score.shape:
        raise ValueError("QSA scores and block starts must have the same 1D shape")
    if count <= 0 or count > score.size:
        raise ValueError("QSA selection budget must be in 1..block count")
    if not np.all(np.isfinite(score)):
        raise ValueError("QSA scores must be finite")
    ranking = np.lexsort((starts, -score))
    return np.sort(starts[ranking[:count]]).astype(np.int64)


@dataclass
class Qwen4ExpQSAIndexDeviceState:
    raw_keys: DeviceBuffer
    member_indices: DeviceBuffer
    block_starts: DeviceBuffer
    pooled_keys: DeviceBuffer
    scores: DeviceBuffer
    selected_starts: DeviceBuffer
    selected_count: DeviceBuffer
    query_position: DeviceBuffer
    selected_positions: DeviceBuffer
    member_host: np.ndarray
    scores_host: np.ndarray
    physical_positions_host: np.ndarray
    capacity: int
    index_heads: int
    index_dim: int
    compression_ratio: int
    block_budget: int
    runtime: HipRuntime
    count: int = 0
    pooled_count: int = 0
    closed: bool = False

    @classmethod
    def allocate(
        cls,
        *,
        attention_state: Qwen4ExpDenseAttentionState,
        index_heads: int,
        index_dim: int,
        compression_ratio: int,
        block_budget: int,
        runtime: HipRuntime | None = None,
    ) -> "Qwen4ExpQSAIndexDeviceState":
        if attention_state.closed:
            raise RuntimeError("QSA index state requires an open attention state")
        active_runtime = runtime or attention_state.runtime
        if active_runtime is not attention_state.runtime:
            raise ValueError("runtime must match the mirrored attention state")
        heads = int(index_heads)
        dimension = int(index_dim)
        ratio = int(compression_ratio)
        budget = int(block_budget)
        if heads <= 0 or dimension <= 0 or ratio <= 0 or budget <= 0:
            raise ValueError("QSA index heads, dimension, ratio, and budget must be positive")
        capacity = int(attention_state.max_positions)
        complete_blocks = capacity // ratio
        if complete_blocks <= 0:
            raise ValueError("QSA index capacity must contain one complete block")
        logical_positions = np.arange(capacity, dtype=np.int64)
        logical_blocks = logical_positions // attention_state.block_size
        physical_positions = (
            attention_state.block_host[logical_blocks].astype(np.int64)
            * attention_state.block_size
            + logical_positions % attention_state.block_size
        )
        member_host = np.ascontiguousarray(
            physical_positions[: complete_blocks * ratio].reshape(complete_blocks, ratio),
            dtype=np.int32,
        )
        # Upload sources for the device buffers whose host mirrors were only
        # ever read by the retired host selection route. They stay locals so the
        # device buffers keep their initial contents (the stateful graph probe
        # snapshots and restores them).
        block_starts_init = np.arange(complete_blocks, dtype=np.int64) * ratio
        selected_starts_init = np.full(budget, -1, dtype=np.int64)
        selected_count_init = np.zeros(1, dtype=np.int32)
        query_position_init = np.zeros(1, dtype=np.int64)
        # ``scores_host`` is the retrieval gate's D2H target for its final score
        # row, so it stays a field even though no engine path reads it.
        scores_host = np.empty(complete_blocks, dtype=np.float32)
        sizes = (
            capacity * dimension * DType.FP32.itemsize,
            member_host.nbytes,
            block_starts_init.nbytes,
            complete_blocks * dimension * DType.FP32.itemsize,
            complete_blocks * DType.FP32.itemsize,
            selected_starts_init.nbytes,
            selected_count_init.nbytes,
            query_position_init.nbytes,
            (budget * ratio + ratio - 1) * DType.INT64.itemsize,
        )
        buffers: list[DeviceBuffer] = []
        try:
            for size in sizes:
                buffers.append(malloc(size, runtime=active_runtime))
            copy_host_array_to_device(buffers[1], member_host, runtime=active_runtime)
            copy_host_array_to_device(
                buffers[2], block_starts_init, runtime=active_runtime
            )
            copy_host_array_to_device(
                buffers[5], selected_starts_init, runtime=active_runtime
            )
            copy_host_array_to_device(
                buffers[6], selected_count_init, runtime=active_runtime
            )
            copy_host_array_to_device(
                buffers[7], query_position_init, runtime=active_runtime
            )
            active_runtime.memset(buffers[0].ptr, 0, buffers[0].nbytes)
        except Exception:
            for buffer in reversed(buffers):
                free(buffer, runtime=active_runtime)
            raise
        return cls(
            *buffers,
            member_host,
            scores_host,
            physical_positions,
            capacity,
            heads,
            dimension,
            ratio,
            budget,
            active_runtime,
        )

    @property
    def dense_equivalent_limit(self) -> int:
        return self.block_budget * self.compression_ratio + self.compression_ratio - 1

    def append(self, raw_key_ptr: int, *, position: int, stream: int = 0) -> None:
        if self.closed:
            raise RuntimeError("QSA index state is closed")
        logical = int(position)
        if logical != self.count:
            raise ValueError("QSA index position must equal contiguous count")
        if logical >= self.capacity:
            raise ValueError("QSA index capacity exceeded")
        physical = int(self.physical_positions_host[logical])
        self.runtime.memcpy_async(
            self.raw_keys.ptr + physical * self.index_dim * DType.FP32.itemsize,
            int(raw_key_ptr),
            self.index_dim * DType.FP32.itemsize,
            HipMemcpyKind.DEVICE_TO_DEVICE,
            int(stream),
        )
        self.count += 1

    def append_device_position(
        self,
        raw_key_ptr: int,
        *,
        position_ptr: int,
        block_table_ptr: int,
        block_size: int,
        block_table_len: int,
        stream: int = 0,
    ) -> None:
        """Append one raw key at a request-owned device position.

        This exact graph primitive intentionally leaves the host ``count``
        mirror unchanged; the graph owner advances that mirror after replay.
        """

        if self.closed:
            raise RuntimeError("QSA index state is closed")
        qwen4_exp_qsa_scatter_index_key_device_position_f32(
            int(raw_key_ptr),
            self.raw_keys.ptr,
            int(block_table_ptr),
            int(position_ptr),
            int(block_size),
            int(block_table_len),
            self.index_dim,
            stream=int(stream),
            runtime=self.runtime,
        )

    def append_rows(
        self,
        raw_keys_ptr: int,
        *,
        start_position: int,
        rows: int,
        block_table_ptr: int,
        block_size: int,
        stream: int = 0,
    ) -> None:
        if self.closed:
            raise RuntimeError("QSA index state is closed")
        start = int(start_position)
        count = int(rows)
        if start != self.count:
            raise ValueError("QSA index row start must equal contiguous count")
        if count <= 0 or start + count > self.capacity:
            raise ValueError("QSA index row append exceeds state capacity")
        qwen4_exp_qsa_scatter_index_keys_f32(
            int(raw_keys_ptr),
            self.raw_keys.ptr,
            int(block_table_ptr),
            start,
            count,
            int(block_size),
            self.index_dim,
            stream=int(stream),
            runtime=self.runtime,
        )
        self.count += count

    def prepare_complete_blocks(
        self,
        blocks: int,
        *,
        key_norm_weight_ptr: int,
        rotary_dim: int,
        theta: float,
        eps: float = 1e-6,
        stream: int = 0,
    ) -> None:
        """Append only newly completed normalized/RoPE index-key blocks."""

        if self.closed:
            raise RuntimeError("QSA index state is closed")
        complete = int(blocks)
        maximum = self.capacity // self.compression_ratio
        if complete < 0 or complete > maximum:
            raise ValueError("QSA complete-block count exceeds index capacity")
        first = self.pooled_count
        if complete <= first:
            return
        count = complete - first
        qwen4_exp_qsa_pool_norm_rope_f32(
            self.raw_keys.ptr,
            self.member_indices.ptr
            + first * self.compression_ratio * DType.INT32.itemsize,
            self.block_starts.ptr + first * DType.INT64.itemsize,
            int(key_norm_weight_ptr),
            self.pooled_keys.ptr + first * self.index_dim * DType.FP32.itemsize,
            count,
            self.compression_ratio,
            self.index_dim,
            rotary_dim,
            theta,
            eps,
            stream=stream,
            runtime=self.runtime,
        )
        self.pooled_count = complete

    def select_positions_device(
        self,
        prepared_query_ptr: int,
        *,
        query_position: int,
        output_positions_ptr: int | None = None,
        output_count_ptr: int | None = None,
        key_norm_weight_ptr: int,
        rotary_dim: int,
        theta: float,
        eps: float = 1e-6,
        stream: int = 0,
    ) -> int:
        """Score and select entirely on device with strict tie/order semantics."""

        if self.closed:
            raise RuntimeError("QSA index state is closed")
        position = int(query_position)
        if position < 0 or position >= self.count:
            raise ValueError("QSA query position must identify an appended index key")
        blocks = (position + 1) // self.compression_ratio
        if blocks <= self.block_budget:
            raise ValueError("native QSA selection requires more blocks than the budget")
        self.prepare_complete_blocks(
            blocks,
            key_norm_weight_ptr=key_norm_weight_ptr,
            rotary_dim=rotary_dim,
            theta=theta,
            eps=eps,
            stream=stream,
        )
        qwen4_exp_qsa_score_f32(
            int(prepared_query_ptr),
            self.pooled_keys.ptr,
            self.scores.ptr,
            1,
            blocks,
            self.index_heads,
            self.index_dim,
            stream=stream,
            runtime=self.runtime,
        )
        positions_ptr = (
            self.selected_positions.ptr
            if output_positions_ptr is None
            else int(output_positions_ptr)
        )
        count_ptr = (
            self.selected_count.ptr
            if output_count_ptr is None
            else int(output_count_ptr)
        )
        qwen4_exp_qsa_topk_expand_f32_i64(
            self.scores.ptr,
            positions_ptr,
            count_ptr,
            blocks,
            position,
            self.compression_ratio,
            self.block_budget,
            stream=stream,
            runtime=self.runtime,
        )
        tail = position + 1 - blocks * self.compression_ratio
        return self.block_budget * self.compression_ratio + tail

    def select(
        self,
        prepared_query_ptr: int,
        *,
        query_position: int,
        key_norm_weight_ptr: int,
        rotary_dim: int,
        theta: float,
        eps: float = 1e-6,
        stream: int = 0,
    ) -> tuple[int, DeviceBuffer]:
        position = int(query_position)
        if position != self.count - 1:
            raise ValueError("QSA query position must identify the latest index key")
        selected_count = self.select_positions_device(
            prepared_query_ptr,
            query_position=position,
            key_norm_weight_ptr=key_norm_weight_ptr,
            rotary_dim=rotary_dim,
            theta=theta,
            eps=eps,
            stream=stream,
        )
        return selected_count, self.selected_positions

    def restore_count(self, count: int) -> None:
        if self.closed:
            raise RuntimeError("QSA index state is closed")
        restored = int(count)
        if restored < 0 or restored > self.capacity:
            raise ValueError("QSA index restore count exceeds capacity")
        self.count = restored
        # Dense rows append raw keys without preparing pooled blocks. Restoring
        # a cursor cannot make those uncomputed blocks valid; sparse selection
        # prepares the missing suffix when it is first needed.
        self.pooled_count = min(self.pooled_count, restored // self.compression_ratio)

    def reset(self) -> None:
        self.restore_count(0)

    def close(self) -> None:
        if self.closed:
            return
        for buffer in reversed(
            (
                self.raw_keys,
                self.member_indices,
                self.block_starts,
                self.pooled_keys,
                self.scores,
                self.selected_starts,
                self.selected_count,
                self.query_position,
                self.selected_positions,
            )
        ):
            free(buffer, runtime=self.runtime)
        self.closed = True


@dataclass(frozen=True)
class Qwen4ExpGRDeviceWeights:
    norm_weight_ptr: int
    down: GGUFDeviceWeight
    up: GGUFDeviceWeight
    inject: GGUFDeviceWeight | None


@dataclass(frozen=True)
class Qwen4ExpGDNMixerDeviceWeights:
    projections: Mapping[str, GGUFDeviceWeight]
    conv_weight_ptr: int
    dt_bias_ptr: int
    a_ptr: int
    norm_weight_ptr: int


@dataclass(frozen=True)
class Qwen4ExpQSAMixerDeviceWeights:
    projections: Mapping[str, GGUFDeviceWeight]
    q_norm_weight_ptr: int
    k_norm_weight_ptr: int
    index_q_norm_weight_ptr: int = 0
    index_k_norm_weight_ptr: int = 0


@dataclass(frozen=True)
class Qwen4ExpQSALayerDeviceWeights:
    attention_gr: Qwen4ExpGRDeviceWeights
    mixer: Qwen4ExpQSAMixerDeviceWeights
    ffn_gr: Qwen4ExpGRDeviceWeights
    moe: Mapping[str, GGUFDeviceWeight]
    layer_id: int = -1
    layer_type: str = "qsa"
    qsa_state_index: int = -1


@dataclass(frozen=True)
class Qwen4ExpGDNLayerDeviceWeights:
    attention_gr: Qwen4ExpGRDeviceWeights
    mixer: Qwen4ExpGDNMixerDeviceWeights
    ffn_gr: Qwen4ExpGRDeviceWeights
    moe: Mapping[str, GGUFDeviceWeight]
    layer_id: int = -1
    layer_type: str = "gdn"
    gdn_state_index: int = -1
    has_ple: bool = False


@dataclass
class Qwen4ExpQSAScratch:
    q_projected: DeviceBuffer
    key_projected: DeviceBuffer
    value_projected: DeviceBuffer
    index_q_projected: DeviceBuffer
    index_k_projected: DeviceBuffer
    index_query: DeviceBuffer
    query: DeviceBuffer
    key: DeviceBuffer
    gate: DeviceBuffer
    context: DeviceBuffer
    gated: DeviceBuffer
    output: DeviceBuffer
    runtime: HipRuntime
    closed: bool = False

    @classmethod
    def allocate(
        cls,
        *,
        rows: int,
        hidden: int,
        query_heads: int,
        kv_heads: int,
        head_dim: int,
        index_heads: int = 1,
        index_dim: int = 1,
        runtime: HipRuntime | None = None,
    ) -> "Qwen4ExpQSAScratch":
        dimensions = (rows, hidden, query_heads, kv_heads, head_dim, index_heads, index_dim)
        if any(int(value) <= 0 for value in dimensions):
            raise ValueError("Qwen4Exp QSA scratch dimensions must be positive")
        active_runtime = runtime or get_hip_runtime()
        q_width = query_heads * head_dim
        kv_width = kv_heads * head_dim
        elements = (
            rows * q_width * 2,
            rows * kv_width,
            rows * kv_width,
            rows * index_heads * index_dim,
            rows * index_dim,
            rows * index_heads * index_dim,
            rows * q_width,
            rows * kv_width,
            rows * q_width,
            rows * q_width,
            rows * q_width,
            rows * hidden,
        )
        buffers: list[DeviceBuffer] = []
        try:
            for count in elements:
                buffers.append(malloc(count * 4, runtime=active_runtime))
        except Exception:
            for buffer in reversed(buffers):
                free(buffer, runtime=active_runtime)
            raise
        return cls(*buffers, runtime=active_runtime)

    def ordered_attention_scratch(
        self, *, query_heads: int, selected_count: int
    ) -> tuple[DeviceBuffer, DeviceBuffer]:
        if self.closed:
            raise RuntimeError("Qwen4Exp QSA scratch is closed")
        required = int(query_heads) * int(selected_count) * DType.FP32.itemsize
        if required <= 0:
            raise ValueError("ordered attention scratch dimensions must be positive")
        scores = getattr(self, "ordered_scores", None)
        coefficients = getattr(self, "ordered_coefficients", None)
        if scores is None or scores.nbytes < required:
            if scores is not None:
                free(scores, runtime=self.runtime)
                free(coefficients, runtime=self.runtime)
            scores = malloc(required, runtime=self.runtime)
            coefficients = malloc(2 * required, runtime=self.runtime)
            self.ordered_scores = scores
            self.ordered_coefficients = coefficients
        return scores, coefficients

    def close(self) -> None:
        if self.closed:
            return
        for buffer in reversed(
            (
                self.q_projected, self.key_projected, self.value_projected,
                self.index_q_projected, self.index_k_projected, self.index_query,
                self.query, self.key, self.gate, self.context, self.gated, self.output,
            )
        ):
            free(buffer, runtime=self.runtime)
        for lazy_key in (
            "flash_k_scratch", "flash_v_scratch", "ordered_scores",
            "wmma_k_f16", "wmma_v_f16", "wmma_mask",
            "ordered_coefficients",
        ):
            lazy = getattr(self, lazy_key, None)
            if lazy is not None:
                free(lazy, runtime=self.runtime)
                setattr(self, lazy_key, None)
        self.closed = True


@dataclass
class Qwen4ExpQSALayerScratch:
    attention_gr: Qwen4ExpGRScratch
    qsa: Qwen4ExpQSAScratch
    ffn_gr: Qwen4ExpGRScratch
    moe: Qwen4ExpMoEScratch
    after_attention: DeviceBuffer
    moe_f32: DeviceBuffer
    output: DeviceBuffer
    runtime: HipRuntime
    closed: bool = False

    @classmethod
    def allocate(
        cls,
        *,
        rows: int,
        branches: int,
        hidden: int,
        low_rank: int,
        query_heads: int,
        kv_heads: int,
        head_dim: int,
        ffn: int,
        experts: int,
        top_k: int,
        index_heads: int = 1,
        index_dim: int = 1,
        runtime: HipRuntime | None = None,
    ) -> "Qwen4ExpQSALayerScratch":
        active_runtime = runtime or get_hip_runtime()
        owners: list[object] = []
        try:
            attention_gr = Qwen4ExpGRScratch.allocate(
                rows=rows, branches=branches, hidden=hidden, low_rank=low_rank,
                runtime=active_runtime,
            )
            owners.append(attention_gr)
            qsa = Qwen4ExpQSAScratch.allocate(
                rows=rows, hidden=hidden, query_heads=query_heads,
                kv_heads=kv_heads, head_dim=head_dim,
                index_heads=index_heads, index_dim=index_dim, runtime=active_runtime,
            )
            owners.append(qsa)
            ffn_gr = Qwen4ExpGRScratch.allocate(
                rows=rows, branches=branches, hidden=hidden, low_rank=low_rank,
                runtime=active_runtime,
            )
            owners.append(ffn_gr)
            moe = Qwen4ExpMoEScratch.allocate(
                rows=rows, hidden=hidden, ffn=ffn, experts=experts, top_k=top_k,
                runtime=active_runtime,
            )
            owners.append(moe)
            after_attention = malloc(rows * branches * hidden * 2, runtime=active_runtime)
            owners.append(after_attention)
            moe_f32 = malloc(rows * hidden * 4, runtime=active_runtime)
            owners.append(moe_f32)
            output = malloc(rows * branches * hidden * 2, runtime=active_runtime)
            owners.append(output)
        except Exception:
            for owner in reversed(owners):
                close = getattr(owner, "close", None)
                if callable(close):
                    close()
                else:
                    free(owner, runtime=active_runtime)
            raise
        return cls(
            attention_gr, qsa, ffn_gr, moe,
            after_attention, moe_f32, output, active_runtime,
        )

    def close(self) -> None:
        if self.closed:
            return
        free(self.output, runtime=self.runtime)
        free(self.moe_f32, runtime=self.runtime)
        free(self.after_attention, runtime=self.runtime)
        self.moe.close()
        self.ffn_gr.close()
        self.qsa.close()
        self.attention_gr.close()
        self.closed = True


@dataclass
class Qwen4ExpGDNScratch:
    qkv: DeviceBuffer
    gate: DeviceBuffer
    alpha: DeviceBuffer
    beta: DeviceBuffer
    conv: DeviceBuffer
    core: DeviceBuffer
    output: DeviceBuffer
    runtime: HipRuntime
    closed: bool = False

    @classmethod
    def allocate(
        cls,
        *,
        rows: int,
        qkv_width: int,
        core_width: int,
        scalar_width: int,
        hidden: int,
        runtime: HipRuntime | None = None,
    ) -> "Qwen4ExpGDNScratch":
        dimensions = (rows, qkv_width, core_width, scalar_width, hidden)
        if any(int(value) <= 0 for value in dimensions):
            raise ValueError("Qwen4Exp GDN scratch dimensions must be positive")
        active_runtime = runtime or get_hip_runtime()
        elements = (
            rows * qkv_width,
            rows * core_width,
            rows * scalar_width,
            rows * scalar_width,
            rows * qkv_width,
            rows * core_width,
            rows * hidden,
        )
        buffers: list[DeviceBuffer] = []
        try:
            for count in elements:
                buffers.append(malloc(count * 4, runtime=active_runtime))
        except Exception:
            for buffer in reversed(buffers):
                free(buffer, runtime=active_runtime)
            raise
        return cls(*buffers, runtime=active_runtime)

    def close(self) -> None:
        if self.closed:
            return
        for buffer in reversed(
            (self.qkv, self.gate, self.alpha, self.beta, self.conv, self.core, self.output)
        ):
            free(buffer, runtime=self.runtime)
        self.closed = True


@dataclass
class Qwen4ExpGDNLayerScratch:
    attention_gr: Qwen4ExpGRScratch
    gdn: Qwen4ExpGDNScratch
    ffn_gr: Qwen4ExpGRScratch
    moe: Qwen4ExpMoEScratch
    after_attention: DeviceBuffer
    moe_f32: DeviceBuffer
    output: DeviceBuffer
    runtime: HipRuntime
    closed: bool = False

    @classmethod
    def allocate(
        cls,
        *,
        rows: int,
        branches: int,
        hidden: int,
        low_rank: int,
        qkv_width: int,
        core_width: int,
        scalar_width: int,
        ffn: int,
        experts: int,
        top_k: int,
        runtime: HipRuntime | None = None,
    ) -> "Qwen4ExpGDNLayerScratch":
        active_runtime = runtime or get_hip_runtime()
        owners: list[object] = []
        try:
            attention_gr = Qwen4ExpGRScratch.allocate(
                rows=rows, branches=branches, hidden=hidden, low_rank=low_rank,
                runtime=active_runtime,
            )
            owners.append(attention_gr)
            gdn = Qwen4ExpGDNScratch.allocate(
                rows=rows, qkv_width=qkv_width, core_width=core_width,
                scalar_width=scalar_width, hidden=hidden, runtime=active_runtime,
            )
            owners.append(gdn)
            ffn_gr = Qwen4ExpGRScratch.allocate(
                rows=rows, branches=branches, hidden=hidden, low_rank=low_rank,
                runtime=active_runtime,
            )
            owners.append(ffn_gr)
            moe = Qwen4ExpMoEScratch.allocate(
                rows=rows, hidden=hidden, ffn=ffn, experts=experts, top_k=top_k,
                runtime=active_runtime,
            )
            owners.append(moe)
            after_attention = malloc(rows * branches * hidden * 2, runtime=active_runtime)
            owners.append(after_attention)
            moe_f32 = malloc(rows * hidden * 4, runtime=active_runtime)
            owners.append(moe_f32)
            output = malloc(rows * branches * hidden * 2, runtime=active_runtime)
            owners.append(output)
        except Exception:
            for owner in reversed(owners):
                close = getattr(owner, "close", None)
                if callable(close):
                    close()
                else:
                    free(owner, runtime=active_runtime)
            raise
        return cls(
            attention_gr, gdn, ffn_gr, moe,
            after_attention, moe_f32, output, active_runtime,
        )

    def close(self) -> None:
        if self.closed:
            return
        free(self.output, runtime=self.runtime)
        free(self.moe_f32, runtime=self.runtime)
        free(self.after_attention, runtime=self.runtime)
        self.moe.close()
        self.ffn_gr.close()
        self.gdn.close()
        self.attention_gr.close()
        self.closed = True


@dataclass
class Qwen4ExpPLEScratch:
    key: DeviceBuffer
    value: DeviceBuffer
    query: DeviceBuffer
    gate: DeviceBuffer
    gated_value: DeviceBuffer
    normalized: DeviceBuffer
    conv: DeviceBuffer
    output: DeviceBuffer
    runtime: HipRuntime
    closed: bool = False

    @classmethod
    def allocate(
        cls,
        *,
        rows: int,
        branches: int,
        hidden: int,
        runtime: HipRuntime | None = None,
    ) -> "Qwen4ExpPLEScratch":
        if rows <= 0 or branches <= 0 or hidden <= 0:
            raise ValueError("rows, branches, and hidden must be positive")
        active_runtime = runtime or get_hip_runtime()
        channels = branches * hidden
        byte_sizes = (
            rows * channels * 4,
            rows * hidden * 4,
            rows * channels * 4,
            rows * branches * 4,
            rows * channels * 4,
            rows * channels * 4,
            rows * channels * 4,
            rows * channels * 2,
        )
        buffers: list[DeviceBuffer] = []
        try:
            for size in byte_sizes:
                buffers.append(malloc(size, runtime=active_runtime))
        except Exception:
            for buffer in reversed(buffers):
                free(buffer, runtime=active_runtime)
            raise
        return cls(*buffers, runtime=active_runtime)

    def close(self) -> None:
        if self.closed:
            return
        for buffer in reversed(
            (
                self.key,
                self.value,
                self.query,
                self.gate,
                self.gated_value,
                self.normalized,
                self.conv,
                self.output,
            )
        ):
            free(buffer, runtime=self.runtime)
        self.closed = True


@dataclass
class Qwen4ExpMoEScratch:
    router_logits: DeviceBuffer
    selected: DeviceBuffer
    routing: DeviceBuffer
    hidden_bf16: DeviceBuffer
    expert_gate: DeviceBuffer
    expert_up: DeviceBuffer
    expert_intermediate: DeviceBuffer
    expert_down: DeviceBuffer
    routed: DeviceBuffer
    shared_gate: DeviceBuffer
    shared_up: DeviceBuffer
    shared_intermediate: DeviceBuffer
    shared_down: DeviceBuffer
    shared_down_bf16: DeviceBuffer
    shared_gate_logits: DeviceBuffer
    output: DeviceBuffer
    group_counts: DeviceBuffer
    group_padded_counts: DeviceBuffer
    group_expert_start: DeviceBuffer
    group_scatter_offsets: DeviceBuffer
    group_sorted_lanes: DeviceBuffer
    group_sorted_experts: DeviceBuffer
    group_sorted_weights: DeviceBuffer
    group_wmma_expert_start: DeviceBuffer
    group_tile_expert: DeviceBuffer
    group_wmma_total: DeviceBuffer
    group_gate_up: DeviceBuffer
    group_lane_to_row: DeviceBuffer
    group_active_experts: DeviceBuffer
    group_active_count: DeviceBuffer
    runtime: HipRuntime
    closed: bool = False
    group_risk_count: DeviceBuffer | None = None
    group_risk_indices: DeviceBuffer | None = None

    @classmethod
    def allocate(
        cls,
        *,
        rows: int,
        hidden: int,
        ffn: int,
        experts: int,
        top_k: int,
        runtime: HipRuntime | None = None,
    ) -> "Qwen4ExpMoEScratch":
        dimensions = (rows, hidden, ffn, experts, top_k)
        if any(int(value) <= 0 for value in dimensions) or top_k > experts:
            raise ValueError("Qwen4Exp MoE dimensions must be positive with top_k <= experts")
        active_runtime = runtime or get_hip_runtime()
        compact = rows * top_k
        tile_capacity = (compact + 15 * experts + 15) // 16
        byte_sizes = (
            rows * experts * 4,
            compact * DType.INT64.itemsize,
            compact * 4,
            rows * hidden * 2,
            compact * ffn * 2,
            compact * ffn * 2,
            compact * ffn * 2,
            compact * hidden * 2,
            rows * hidden * 2,
            rows * ffn * 4,
            rows * ffn * 4,
            rows * ffn * 4,
            rows * hidden * 4,
            rows * hidden * 2,
            rows * 4,
            rows * hidden * 2,
            experts * DType.INT32.itemsize,
            experts * DType.INT32.itemsize,
            (experts + 1) * DType.INT64.itemsize,
            experts * DType.INT32.itemsize,
            compact * DType.INT64.itemsize,
            compact * DType.INT64.itemsize,
            compact * DType.FP32.itemsize,
            (experts + 1) * DType.INT64.itemsize,
            tile_capacity * DType.INT64.itemsize,
            DType.INT64.itemsize,
            compact * 2 * ffn * DType.BF16.itemsize,
            compact * DType.INT64.itemsize,
            experts * DType.INT64.itemsize,
            DType.INT64.itemsize,
        )
        buffers: list[DeviceBuffer] = []
        try:
            for size in byte_sizes:
                buffers.append(malloc(size, runtime=active_runtime))
        except Exception:
            for buffer in reversed(buffers):
                free(buffer, runtime=active_runtime)
            raise
        return cls(*buffers, runtime=active_runtime)

    def close(self) -> None:
        if self.closed:
            return
        for buffer in reversed(
            (
                self.router_logits,
                self.selected,
                self.routing,
                self.hidden_bf16,
                self.expert_gate,
                self.expert_up,
                self.expert_intermediate,
                self.expert_down,
                self.routed,
                self.shared_gate,
                self.shared_up,
                self.shared_intermediate,
                self.shared_down,
                self.shared_down_bf16,
                self.shared_gate_logits,
                self.output,
                self.group_counts,
                self.group_padded_counts,
                self.group_expert_start,
                self.group_scatter_offsets,
                self.group_sorted_lanes,
                self.group_sorted_experts,
                self.group_sorted_weights,
                self.group_wmma_expert_start,
                self.group_tile_expert,
                self.group_wmma_total,
                self.group_gate_up,
                self.group_lane_to_row,
                self.group_active_experts,
                self.group_active_count,
            )
        ):
            free(buffer, runtime=self.runtime)
        if self.group_risk_indices is not None:
            free(self.group_risk_indices, runtime=self.runtime)
            self.group_risk_indices = None
        if self.group_risk_count is not None:
            free(self.group_risk_count, runtime=self.runtime)
            self.group_risk_count = None
        self.closed = True

    def ensure_group_risk_buffers(
        self, *, compact_rows: int, out_features_total: int
    ) -> tuple[DeviceBuffer, DeviceBuffer]:
        """Lazily allocate the bounded risk counter/index queue.

        The index capacity is the full worst case (every output at risk) so
        the exact repair route can never silently drop a queued index.
        """

        if self.closed:
            raise RuntimeError("Qwen4Exp MoE scratch is closed")
        capacity = int(compact_rows) * int(out_features_total)
        if capacity <= 0 or capacity > 2**31 - 1:
            raise ValueError("risk index capacity must be a positive int32 count")
        if (
            self.group_risk_count is None
            or self.group_risk_indices is None
            or self.group_risk_indices.nbytes < capacity * DType.INT32.itemsize
        ):
            if self.group_risk_indices is not None:
                free(self.group_risk_indices, runtime=self.runtime)
            if self.group_risk_count is not None:
                free(self.group_risk_count, runtime=self.runtime)
            self.group_risk_count = malloc(
                DType.INT32.itemsize, runtime=self.runtime)
            self.group_risk_indices = malloc(
                capacity * DType.INT32.itemsize, runtime=self.runtime)
        return self.group_risk_count, self.group_risk_indices


@dataclass(frozen=True)
class Qwen4ExpMoEDeviceResult:
    output: DeviceBuffer
    selected: DeviceBuffer
    routing: DeviceBuffer


@dataclass
class Qwen4ExpGRScratch:
    normalized: DeviceBuffer
    low_rank: DeviceBuffer
    gate: DeviceBuffer
    inject_logits: DeviceBuffer
    mixed: DeviceBuffer
    runtime: HipRuntime
    closed: bool = False

    @classmethod
    def allocate(
        cls,
        *,
        rows: int,
        branches: int,
        hidden: int,
        low_rank: int,
        runtime: HipRuntime | None = None,
    ) -> "Qwen4ExpGRScratch":
        if rows <= 0 or branches <= 0 or hidden <= 0 or low_rank <= 0:
            raise ValueError("rows, branches, hidden, and low_rank must be positive")
        active_runtime = runtime or get_hip_runtime()
        buffers: list[DeviceBuffer] = []
        try:
            for elements in (
                rows * branches * hidden,
                rows * low_rank,
                rows * branches * hidden,
                rows * branches,
                rows * hidden,
            ):
                buffers.append(malloc(elements * 4, runtime=active_runtime))
        except Exception:
            for buffer in reversed(buffers):
                free(buffer, runtime=active_runtime)
            raise
        return cls(*buffers, runtime=active_runtime)

    def close(self) -> None:
        if self.closed:
            return
        for buffer in reversed(
            (
                self.normalized,
                self.low_rank,
                self.gate,
                self.inject_logits,
                self.mixed,
            )
        ):
            free(buffer, runtime=self.runtime)
        self.closed = True


@dataclass(frozen=True)
class Qwen4ExpGRReadDeviceResult:
    normalized: DeviceBuffer
    gate: DeviceBuffer
    mixed: DeviceBuffer
    inject_logits: DeviceBuffer


def bind_qwen4_exp_qsa_layer(
    resident: object,
    layer_id: int,
) -> Qwen4ExpQSALayerDeviceWeights:
    """Bind one validated resident QSA layer to dense F5 physical roles."""

    plan = getattr(resident, "plan")
    config = plan.config
    layer = int(layer_id)
    if not 0 <= layer < config.block_count:
        raise ValueError(f"layer_id must be in [0, {config.block_count}), got {layer}")
    if config.layer_types[layer] != "qsa":
        raise ValueError(f"Qwen4Exp layer {layer} is not QSA")

    def weight(slot: str) -> GGUFDeviceWeight:
        return resident.weight(f"layers.{layer}.{slot}")

    def pointer(slot: str) -> int:
        return int(weight(slot).allocation("raw").tensor.ptr)

    def gr(prefix: str) -> Qwen4ExpGRDeviceWeights:
        return Qwen4ExpGRDeviceWeights(
            norm_weight_ptr=pointer(f"hc_{prefix}_norm"),
            down=weight(f"hc_{prefix}_down"),
            up=weight(f"hc_{prefix}_up"),
            inject=weight(f"hc_{prefix}_inject"),
        )

    projection_slots = (
        "attn_q", "attn_k", "attn_v", "attn_output", "index_q", "index_k"
    )
    moe_slots = {
        "router": "router", "expert_gate": "expert_gate",
        "expert_up": "expert_up", "expert_down": "expert_down",
        "shared_gate": "shared_gate", "shared_up": "shared_up",
        "shared_down": "shared_down", "shared_gate_weight": "shared_expert_gate",
    }
    return Qwen4ExpQSALayerDeviceWeights(
        attention_gr=gr("attn"),
        mixer=Qwen4ExpQSAMixerDeviceWeights(
            projections=MappingProxyType(
                {slot: weight(slot) for slot in projection_slots}
            ),
            q_norm_weight_ptr=pointer("attn_q_norm"),
            k_norm_weight_ptr=pointer("attn_k_norm"),
            index_q_norm_weight_ptr=pointer("index_q_norm"),
            index_k_norm_weight_ptr=pointer("index_k_norm"),
        ),
        ffn_gr=gr("ffn"),
        moe=MappingProxyType(
            {target: weight(source) for target, source in moe_slots.items()}
        ),
        layer_id=layer,
        layer_type="qsa",
        qsa_state_index=sum(1 for kind in config.layer_types[:layer] if kind == "qsa"),
    )


def bind_qwen4_exp_gdn_layer(
    resident: object,
    layer_id: int,
) -> Qwen4ExpGDNLayerDeviceWeights:
    """Bind one validated resident GDN layer to physical runner roles."""

    plan = getattr(resident, "plan")
    config = plan.config
    layer = int(layer_id)
    if not 0 <= layer < config.block_count:
        raise ValueError(f"layer_id must be in [0, {config.block_count}), got {layer}")
    if config.layer_types[layer] != "gdn":
        raise ValueError(f"Qwen4Exp layer {layer} is not GDN")

    def weight(slot: str) -> GGUFDeviceWeight:
        return resident.weight(f"layers.{layer}.{slot}")

    def pointer(slot: str) -> int:
        return int(weight(slot).allocation("raw").tensor.ptr)

    def gr(prefix: str) -> Qwen4ExpGRDeviceWeights:
        return Qwen4ExpGRDeviceWeights(
            norm_weight_ptr=pointer(f"hc_{prefix}_norm"),
            down=weight(f"hc_{prefix}_down"),
            up=weight(f"hc_{prefix}_up"),
            inject=weight(f"hc_{prefix}_inject"),
        )

    mixer_slots = ("attn_qkv", "attn_gate", "ssm_alpha", "ssm_beta", "ssm_out")
    moe_slots = {
        "router": "router",
        "expert_gate": "expert_gate",
        "expert_up": "expert_up",
        "expert_down": "expert_down",
        "shared_gate": "shared_gate",
        "shared_up": "shared_up",
        "shared_down": "shared_down",
        "shared_gate_weight": "shared_expert_gate",
    }
    return Qwen4ExpGDNLayerDeviceWeights(
        attention_gr=gr("attn"),
        mixer=Qwen4ExpGDNMixerDeviceWeights(
            projections=MappingProxyType({slot: weight(slot) for slot in mixer_slots}),
            conv_weight_ptr=pointer("ssm_conv1d"),
            dt_bias_ptr=pointer("ssm_dt_bias"),
            a_ptr=pointer("ssm_a"),
            norm_weight_ptr=pointer("ssm_norm"),
        ),
        ffn_gr=gr("ffn"),
        moe=MappingProxyType(
            {target: weight(source) for target, source in moe_slots.items()}
        ),
        layer_id=layer,
        layer_type="gdn",
        gdn_state_index=sum(1 for kind in config.layer_types[:layer] if kind == "gdn"),
        has_ple=layer in config.ple_layers,
    )


def run_qwen4_exp_dense_qsa_token_mixer(
    mixed_ptr: int,
    weights: Qwen4ExpQSAMixerDeviceWeights,
    *,
    attention_state: Qwen4ExpDenseAttentionState,
    scratch: Qwen4ExpQSAScratch,
    position: int,
    rows: int,
    hidden: int,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    theta: float,
    index_state: Qwen4ExpQSAIndexDeviceState | None = None,
    index_heads: int = 0,
    index_dim: int = 0,
    index_rotary_dim: int = 0,
    rope_positions_ptr: int | None = None,
    position_prepared: bool = False,
    device_position_owned: bool = False,
    attention_context_limit: int | None = None,
    cache_only: bool = False,
    qsa_ordered_decode_v2: bool = False,
    qsa_wave32: bool = True,
    eps: float = 1e-6,
    stream: int = 0,
    runtime: HipRuntime | None = None,
) -> DeviceBuffer | None:
    """Append exact scalar cache state; optionally omit the unused attention output.

    ``qsa_ordered_decode_v2`` is the owning runner's plan-published ordered-v2
    sparse-decode decision (see :func:`qsa_ordered_decode_v2`). It only selects
    the v2 three-pass leaf inside the ordered route's own guards - the v1 switch
    must be on, ``head_dim`` must be 256, and the selection must be 1..4096 rows
    - and never on a cache-only call or on decode positions the index state has
    not made sparse. An unbound caller keeps the v1 ordered parent.

    ``qsa_wave32`` is the owning runner's H128 sparse-attention decision (see
    :func:`qsa_wave32`) and defaults to the production wave32 owner, so an
    unbound H128 caller keeps that arithmetic. It selects the wave32 paged
    owner only for ``head_dim`` 128 on the non-ordered sparse arm; every other
    geometry keeps the wave64 parent, which is also what an explicit ``False``
    restores.
    """

    if rows != 1:
        raise ValueError("strict dense QSA decode currently requires rows == 1")
    if scratch.closed or attention_state.closed:
        raise RuntimeError("Qwen4Exp QSA scratch/state is closed")
    active_runtime = runtime or scratch.runtime
    if active_runtime is not scratch.runtime or active_runtime is not attention_state.runtime:
        raise ValueError("runtime must match QSA scratch and attention state owners")
    required = {"attn_q", "attn_k", "attn_v", "attn_output"}
    missing = sorted(required - set(weights.projections))
    if missing:
        raise ValueError("missing Qwen4Exp QSA weights: " + ", ".join(missing))
    if device_position_owned and not position_prepared:
        raise ValueError("device-owned QSA position must be prepared before launch")
    if not position_prepared:
        attention_state.set_position(position)
    context_limit = position + 1 if attention_context_limit is None else int(attention_context_limit)
    if context_limit < position + 1 or context_limit > attention_state.max_positions:
        raise ValueError("QSA attention context limit must cover position within capacity")
    q_width = query_heads * head_dim
    kv_width = kv_heads * head_dim
    for slot, output, out_features in (
        ("attn_q", scratch.q_projected, q_width * 2),
        ("attn_k", scratch.key_projected, kv_width),
        ("attn_v", scratch.value_projected, kv_width),
    ):
        launch_gguf_linear(
            weights.projections[slot], mixed_ptr, output.ptr,
            rows, hidden, out_features,
            activation_dtype=GGUF_ACTIVATION_F32,
            output_dtype=GGUF_OUTPUT_F32,
            stream=stream, runtime=active_runtime,
        )
    selected_count = 0
    selected_positions: DeviceBuffer | None = None
    if index_state is not None:
        index_required = {"index_q", "index_k"}
        index_missing = sorted(index_required - set(weights.projections))
        if index_missing:
            raise ValueError("missing Qwen4Exp QSA index weights: " + ", ".join(index_missing))
        if index_heads != index_state.index_heads or index_dim != index_state.index_dim:
            raise ValueError("QSA index geometry must match its state owner")
        if weights.index_q_norm_weight_ptr == 0 or weights.index_k_norm_weight_ptr == 0:
            raise ValueError("QSA index norm weights are required with index state")
        launch_gguf_linear(
            weights.projections["index_k"], mixed_ptr, scratch.index_k_projected.ptr,
            rows, hidden, index_dim,
            activation_dtype=GGUF_ACTIVATION_F32,
            output_dtype=GGUF_OUTPUT_F32,
            stream=stream, runtime=active_runtime,
        )
        if device_position_owned:
            if position + 1 > index_state.dense_equivalent_limit:
                raise ValueError("device-owned dense QSA capture cannot cross sparse selection")
            index_state.append_device_position(
                scratch.index_k_projected.ptr,
                position_ptr=attention_state.position.ptr,
                block_table_ptr=attention_state.block_table.ptr,
                block_size=attention_state.block_size,
                block_table_len=int(attention_state.block_host.size),
                stream=stream,
            )
        else:
            index_state.append(
                scratch.index_k_projected.ptr, position=position, stream=stream
            )
        if not cache_only and not device_position_owned and index_state.count > index_state.dense_equivalent_limit:
            launch_gguf_linear(
                weights.projections["index_q"], mixed_ptr, scratch.index_q_projected.ptr,
                rows, hidden, index_heads * index_dim,
                activation_dtype=GGUF_ACTIVATION_F32,
                output_dtype=GGUF_OUTPUT_F32,
                stream=stream, runtime=active_runtime,
            )
            index_rope = (
                qwen4_exp_qsa_norm_mrope_f32
                if rope_positions_ptr is not None
                else qwen4_exp_qsa_norm_rope_f32
            )
            index_rope(
                scratch.index_q_projected.ptr,
                weights.index_q_norm_weight_ptr,
                attention_state.position.ptr
                if rope_positions_ptr is None
                else int(rope_positions_ptr),
                scratch.index_query.ptr,
                index_heads,
                index_dim,
                index_rotary_dim,
                theta,
                eps,
                stream=stream,
                runtime=active_runtime,
            )
            selected_count, selected_positions = index_state.select(
                scratch.index_query.ptr,
                query_position=position,
                key_norm_weight_ptr=weights.index_k_norm_weight_ptr,
                rotary_dim=index_rotary_dim,
                theta=theta,
                eps=eps,
                stream=stream,
            )
    split_rope = (
        qwen4_exp_qsa_split_norm_mrope_f32
        if rope_positions_ptr is not None
        else qwen4_exp_qsa_split_norm_rope_f32
    )
    split_rope(
        scratch.q_projected.ptr,
        scratch.key_projected.ptr,
        weights.q_norm_weight_ptr,
        weights.k_norm_weight_ptr,
        attention_state.position.ptr
        if rope_positions_ptr is None
        else int(rope_positions_ptr),
        scratch.query.ptr,
        scratch.key.ptr,
        scratch.gate.ptr,
        query_heads,
        kv_heads,
        head_dim,
        rotary_dim,
        theta,
        eps,
        stream=stream,
        runtime=active_runtime,
    )
    qwen35_write_paged_kv_f32_spans(
        scratch.key.ptr,
        scratch.value_projected.ptr,
        attention_state.key_cache.ptr,
        attention_state.value_cache.ptr,
        attention_state.append_spans,
        attention_state.block_size,
        kv_heads,
        head_dim,
        stream=stream,
        runtime=active_runtime,
    )
    if cache_only:
        return None
    if selected_positions is None:
        qwen35_paged_full_attn_decode_context_bf16_spans(
            scratch.query.ptr,
            attention_state.key_cache.ptr,
            attention_state.value_cache.ptr,
            scratch.context.ptr,
            attention_state.decode_spans,
            context_limit,
            attention_state.block_size,
            query_heads,
            kv_heads,
            head_dim,
            head_dim ** -0.5,
            stream=stream,
            runtime=active_runtime,
        )
    else:
        ordered_decode = (
            head_dim == 256
            and os.environ.get("HIPENGINE_QWEN4_EXP_QSA_ORDERED_DECODE", "0")
            not in {"", "0", "false", "False"}
        )
        # The v2 leaf is the owning runner's plan-published decision, not a
        # process-wide switch: the v1 switch above stays the prerequisite that
        # decides whether any ordered decode runs at all, the H256/count guards
        # stay dispatch-time, and an unbound caller keeps the v1 parent. The
        # fused sibling below remains a separate diagnostic over this decision.
        if ordered_decode:
            scores, coefficients = scratch.ordered_attention_scratch(
                query_heads=query_heads, selected_count=selected_count
            )

            def _resolve_ordered_v2():
                v2_key = KernelKey(
                    str(weights.projections["attn_q"].backend),
                    "qsa_sparse_attention", "bf16_kv",
                    "strict_ordered_three_pass_v2_spans")
                if not is_registered(v2_key):
                    return None
                return resolve(backend=v2_key.backend, layer=v2_key.layer,
                               quant=v2_key.quant, variant=v2_key.variant)

            attention_fn = _qwen4_exp_qsa_ordered_decode_leaf(
                ordered=ordered_decode, head_dim=head_dim,
                selected_count=selected_count,
                published=qsa_ordered_decode_v2,
                fused=os.environ.get(
                    "HIPENGINE_QWEN4_EXP_QSA_ORDERED_DECODE_V2_FUSED", "0"
                ) not in {"", "0", "false", "False"},
                resolve_v2=_resolve_ordered_v2,
            )
            attention_fn(
                scratch.query.ptr,
                attention_state.key_cache.ptr,
                attention_state.value_cache.ptr,
                selected_positions.ptr,
                scores.ptr,
                coefficients.ptr,
                scratch.context.ptr,
                attention_state.decode_spans,
                selected_count=selected_count,
                block_size=attention_state.block_size,
                query_heads=query_heads,
                kv_heads=kv_heads,
                head_dim=head_dim,
                scale=head_dim ** -0.5,
                stream=stream,
                runtime=active_runtime,
            )
        else:
            sparse_attention = qwen4_exp_qsa_sparse_attention_paged_bf16_f32
            if _qwen4_exp_qsa_wave32_selected(
                head_dim=head_dim, published=qsa_wave32
            ):
                sparse_attention = qwen4_exp_qsa_sparse_attention_paged_bf16_wave32_f32
            sparse_attention(
                scratch.query.ptr,
                attention_state.key_cache.ptr,
                attention_state.value_cache.ptr,
                selected_positions.ptr,
                scratch.context.ptr,
                attention_state.decode_spans,
                selected_count=selected_count,
                block_size=attention_state.block_size,
                query_heads=query_heads,
                kv_heads=kv_heads,
                head_dim=head_dim,
                scale=head_dim ** -0.5,
                stream=stream,
                runtime=active_runtime,
            )
    qwen4_exp_qsa_gate_context_f32(
        scratch.context.ptr,
        scratch.gate.ptr,
        scratch.gated.ptr,
        rows * q_width,
        stream=stream,
        runtime=active_runtime,
    )
    launch_gguf_linear(
        weights.projections["attn_output"],
        scratch.gated.ptr,
        scratch.output.ptr,
        rows,
        q_width,
        hidden,
        activation_dtype=GGUF_ACTIVATION_F32,
        output_dtype=GGUF_OUTPUT_F32,
        stream=stream,
        runtime=active_runtime,
    )
    return scratch.output


# H256 prefill head grouping for the *scalar* sparse-prefill rows owner. The
# execution profile publishes the decision to the owning runner ("quad" on
# production lanes); "pair" exists only for explicit diagnostics and is never
# published. An unbound runner, an unknown value, or a shape the leaf does not
# cover keeps the single-head parent. This leaf is resolved here and launched
# only where WMMA attention does not own the chunk: the production profiles also
# bind HIPENGINE_QWEN4_EXP_QSA_WMMA=1, so on those lanes the grouping is the
# fallback selection rather than the kernel that runs.
_QSA_HEAD_VARIANTS = {
    "pair": "strict_h256_head_pair_rows_spans",
    "quad": "strict_h256_head_quad_rows_spans",
}
_QSA_HEAD_PARENT = "parent"


@contextlib.contextmanager
def qsa_prefill_head_mode(runner: Any, mode: str | None = None):
    """Bind an explicit head mode for a diagnostic arm.

    The request's own decision is published on the runner by the profile binder;
    a harness runs outside that request, so it selects its arm here and the
    scope puts the published value back on exit. ``None`` leaves it in force.
    """

    prior = getattr(runner, "qsa_prefill_head_mode", _QSA_HEAD_PARENT)
    if mode is not None:
        runner.qsa_prefill_head_mode = str(mode)
    try:
        yield prior
    finally:
        runner.qsa_prefill_head_mode = prior


# QSA chunk selection owner. Production takes the batched path (one score pass,
# the top-k mask, one expansion per row) and keeps the strict per-row device
# selection below it for a diagnostic arm. It is request-local state: the
# prefill chunk path forwards the owning runner's decision to the token mixer,
# so a harness binds the fallback on the runner it measures instead of writing a
# process-wide switch.
_QSA_BATCHED_SELECTION_UNSET = object()


@contextlib.contextmanager
def qsa_batched_selection(
    runner: Any, enabled: bool | object = _QSA_BATCHED_SELECTION_UNSET
):
    """Bind the QSA chunk selection owner for a diagnostic arm.

    ``False`` pins the per-row fallback for the measured window and ``True``
    restores the batched owner; leaving ``enabled`` unset keeps whatever the
    runner carries. The prior value is restored on exit, including on an
    exception.
    """

    prior = getattr(runner, "qsa_batched_selection", True)
    if enabled is not _QSA_BATCHED_SELECTION_UNSET:
        runner.qsa_batched_selection = bool(enabled)
    try:
        yield prior
    finally:
        runner.qsa_batched_selection = prior


# QSA H128 sparse-attention selection. The promoted production arithmetic for
# head_dim 128 is the wave32 paged owner, on both the decode dispatch and the
# prefill rows fallback below the H256 wave-prefill owner; the wave64 parents
# stay the registered strict fallback. It is request-local state: the c1 decode
# path and the prefill chunk path forward the owning runner's decision, so a
# harness binds the parent on the runner it measures instead of writing a
# process-wide switch.
_QSA_WAVE32_UNSET = object()


@contextlib.contextmanager
def qsa_wave32(runner: Any, enabled: bool | object = _QSA_WAVE32_UNSET):
    """Bind the QSA H128 sparse-attention owner for a diagnostic arm.

    ``False`` pins the wave64 parents for the measured window and ``True``
    restores the wave32 owners; leaving ``enabled`` unset keeps whatever the
    runner carries. The prior value is restored on exit, including on an
    exception.
    """

    prior = getattr(runner, "qsa_wave32", True)
    if enabled is not _QSA_WAVE32_UNSET:
        runner.qsa_wave32 = bool(enabled)
    try:
        yield prior
    finally:
        runner.qsa_wave32 = prior


def _qwen4_exp_qsa_wave32_selected(*, head_dim: int, published: bool) -> bool:
    """The dispatch-time guard on the H128 wave32 sparse-attention owners.

    The selection covers the head_dim 128 geometry only; every other geometry
    keeps its own owner, and the wave64 parents remain the registered strict
    fallback. The decision is the owning runner's, not a process switch.
    """

    return bool(head_dim == 128 and published)


# QSA ordered-v2 sparse-decode selection. The production plan publishes the
# ordered-v2 leaf for the c1 H256 indexed sparse decode scope; the binder
# publishes that decision to the owning runner, and the dispatch-time selector
# keeps its guards - the v1 ordered switch as prerequisite, head_dim 256, and a
# selection of 1..4096 positions - with the registered v1 parent as fallback.
# The fused coefficients+values sibling stays a separate env diagnostic over
# this decision. It is request-local state, so a diagnostic arm binds it on the
# runner it measures instead of writing a process-wide switch.
_QSA_ORDERED_DECODE_V2_UNSET = object()


@contextlib.contextmanager
def qsa_ordered_decode_v2(
    runner: Any, enabled: bool | object = _QSA_ORDERED_DECODE_V2_UNSET,
):
    """Bind the QSA ordered-v2 sparse-decode selection for a diagnostic arm.

    ``True`` selects the v2 three-pass leaf inside the ordered route's guards
    and ``False`` keeps the v1 ordered parent; leaving ``enabled`` unset keeps
    whatever the plan published. The binding is a mutable runner field, so it
    stays in force for as long as the scope is open and the published value is
    restored on exit, including on an exception.
    """

    prior = getattr(runner, "qsa_ordered_decode_v2", False)
    if enabled is not _QSA_ORDERED_DECODE_V2_UNSET:
        runner.qsa_ordered_decode_v2 = bool(enabled)
    try:
        yield prior
    finally:
        runner.qsa_ordered_decode_v2 = prior


def _qwen4_exp_qsa_ordered_decode_v2(
    *, ordered: bool, head_dim: int, selected_count: int, published: bool
) -> bool:
    """The dispatch-time guard on the ordered-v2 sparse decode leaf.

    The leaf is the c1 H256 indexed sparse decode owner, so it needs the ordered
    v1 route as its prerequisite, ``head_dim`` 256, a selection of 1..4096
    positions and the owning runner's published decision. Anything else keeps
    the registered v1 ordered parent.
    """

    return bool(
        ordered
        and head_dim == 256
        and 0 < selected_count <= 4096
        and published
    )


def _qwen4_exp_qsa_ordered_decode_leaf(
    *, ordered: bool, head_dim: int, selected_count: int, published: bool,
    fused: bool, resolve_v2=None,
):
    """Pick the sparse-decode attention kernel for one ordered decode step.

    ``None`` means the ordered route is off, so the caller keeps the
    non-ordered sparse owner. ``fused`` is the separate coefficients+values
    diagnostic and only selects its bit-identical sibling on top of an eligible
    v2 step; an unregistered v2 kernel falls back to the registered v1 ordered
    parent rather than to the non-ordered owner.
    """

    if not ordered:
        return None
    if not _qwen4_exp_qsa_ordered_decode_v2(
        ordered=ordered, head_dim=head_dim, selected_count=selected_count,
        published=published,
    ):
        return qwen4_exp_qsa_sparse_attention_paged_bf16_ordered_f32
    if fused:
        return qwen4_exp_qsa_sparse_attention_paged_bf16_fused_v2_f32
    resolved = None if resolve_v2 is None else resolve_v2()
    return resolved or qwen4_exp_qsa_sparse_attention_paged_bf16_ordered_f32


def _qwen4_exp_qsa_prefill_head_key(
    key: KernelKey, *, query_heads: int, kv_heads: int, mode: str | None
) -> KernelKey:
    if (key.variant != "strict_h256_page256_wave_rows_spans"
            or query_heads != 24 or kv_heads != 2):
        return key
    variant = _QSA_HEAD_VARIANTS.get(str(mode))
    if variant is None:
        return key
    candidate = KernelKey(key.backend, key.layer, key.quant, variant)
    return candidate if is_registered(candidate) else key


@contextlib.contextmanager
def qsa_h256_prefill_mode(runner: Any, mode: str | None = None):
    """Bind parent, generic wave or page256 H256 prefill on one runner."""

    if mode is not None and mode not in {"parent", "wave", "page256"}:
        raise ValueError(f"invalid H256 prefill mode {mode!r}")
    prior = getattr(runner, "qsa_h256_prefill_mode", "parent")
    if mode is not None:
        runner.qsa_h256_prefill_mode = mode
    try:
        yield prior
    finally:
        runner.qsa_h256_prefill_mode = prior


def qwen4_exp_qsa_h256_wave_prefill_variant(published: str = "parent") -> str:
    return (
        "strict_h256_page256_wave_rows_spans"
        if published == "page256"
        else "strict_h256_wave_rows_spans"
    )


def qwen4_exp_qsa_h256_wave_prefill_selected(
    *, head_dim: int, rows: int, backend: str, block_size: int = 256,
    published: str = "parent",
) -> bool:
    return (
        head_dim == 256 and rows >= 16
        and published in {"wave", "page256"}
        and (published != "page256" or block_size == 256)
        and is_registered(KernelKey(
            backend, "qsa_sparse_attention", "bf16_kv",
            qwen4_exp_qsa_h256_wave_prefill_variant(published)))
    )


def run_qwen4_exp_qsa_prefill_token_mixer(
    mixed_ptr: int,
    weights: Qwen4ExpQSAMixerDeviceWeights,
    *,
    attention_state: Qwen4ExpDenseAttentionState,
    index_state: Qwen4ExpQSAIndexDeviceState,
    scratch: Qwen4ExpQSAScratch,
    metadata: Qwen4ExpQSAPrefillMetadata,
    start_position: int,
    rows: int,
    hidden: int,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    theta: float,
    index_heads: int,
    index_dim: int,
    index_rotary_dim: int,
    rope_positions_ptr: int | None = None,
    cache_only: bool = False,
    h256_mode: str = "parent",
    head_mode: str | None = _QSA_HEAD_PARENT,
    batched_selection: bool = True,
    qsa_wave32: bool = True,
    eps: float = 1e-6,
    stream: int = 0,
    runtime: HipRuntime | None = None,
) -> DeviceBuffer | None:
    """Append a contiguous QSA chunk, optionally omitting unused attention.

    ``h256_mode`` is the owning runner's scalar H256 sparse-prefill choice;
    page256 keeps its block-size guard and the generic wave leaf is diagnostic.
    Head grouping and WMMA precedence are separate decisions. Kernel arithmetic
    and the ordered decode route are unchanged.

    ``head_mode`` is the owning runner's H256 prefill head grouping decision
    for the scalar rows owner (see :func:`qsa_prefill_head_mode`). It selects
    the fallback leaf: when ``qsa_wmma`` owns the chunk the grouping is resolved
    but that owner is not launched. It is ignored by the ``cache_only`` path,
    which returns before any attention dispatch.

    ``batched_selection`` is the owning runner's QSA chunk selection decision
    (see :func:`qsa_batched_selection`): True runs the batched score/top-k/expand
    owner, False the strict per-row fallback. The shape and count guards below
    apply to whichever arm runs.

    ``qsa_wave32`` is the owning runner's H128 sparse-attention decision (see
    :func:`qsa_wave32`) and defaults to the production wave32 owner, so an
    unbound H128 caller keeps that arithmetic. It selects the wave32 rows owner
    as the fallback below the H256 wave-prefill owner, only for ``head_dim``
    128 and only where WMMA attention does not own the chunk; every other
    geometry, an explicit ``False``, and a chunk WMMA owns keep the wave64 rows
    parent.
    """

    count = int(rows)
    start = int(start_position)
    if count <= 0 or count > metadata.rows:
        raise ValueError("QSA prefill rows must fit metadata capacity")
    if start != index_state.count:
        raise ValueError("QSA prefill start must equal the contiguous index cursor")
    if start + count > attention_state.max_positions:
        raise ValueError("QSA prefill chunk exceeds attention capacity")
    if scratch.closed or attention_state.closed or index_state.closed or metadata.closed:
        raise RuntimeError("Qwen4Exp QSA prefill scratch/state is closed")
    active_runtime = runtime or scratch.runtime
    if any(
        owner is not active_runtime
        for owner in (attention_state.runtime, index_state.runtime, metadata.runtime)
    ) or scratch.runtime is not active_runtime:
        raise ValueError("runtime must match all QSA prefill owners")
    required = {"attn_q", "attn_k", "attn_v", "attn_output", "index_q", "index_k"}
    missing = sorted(required - set(weights.projections))
    if missing:
        raise ValueError("missing Qwen4Exp QSA prefill weights: " + ", ".join(missing))
    if index_heads != index_state.index_heads or index_dim != index_state.index_dim:
        raise ValueError("QSA index geometry must match its state owner")
    if weights.index_q_norm_weight_ptr == 0 or weights.index_k_norm_weight_ptr == 0:
        raise ValueError("QSA index norm weights are required for prefill")
    metadata.set_contiguous(start, count)
    q_width = query_heads * head_dim
    kv_width = kv_heads * head_dim
    for slot, output, out_features in (
        ("attn_q", scratch.q_projected, q_width * 2),
        ("attn_k", scratch.key_projected, kv_width),
        ("attn_v", scratch.value_projected, kv_width),
        ("index_q", scratch.index_q_projected, index_heads * index_dim),
        ("index_k", scratch.index_k_projected, index_dim),
    ):
        if cache_only and slot in {"attn_q", "index_q"}:
            continue
        project = launch_gguf_linear_decode_rows if cache_only else launch_gguf_linear
        project(
            weights.projections[slot],
            mixed_ptr,
            output.ptr,
            count,
            hidden,
            out_features,
            activation_dtype=GGUF_ACTIVATION_F32,
            output_dtype=GGUF_OUTPUT_F32,
            stream=stream,
            runtime=active_runtime,
        )
    # The gate multiply reads the gate half of the q projection in place, so
    # the split kernel does not copy it out (bit-identical).
    strided_gate = _qwen4_exp_qsa_strided_gate_enabled()
    split_rope_rows = (
        qwen4_exp_qsa_split_norm_mrope_rows_f32
        if rope_positions_ptr is not None
        else qwen4_exp_qsa_split_norm_rope_rows_f32
    )
    norm_rope_rows = (
        qwen4_exp_qsa_norm_mrope_rows_f32
        if rope_positions_ptr is not None
        else qwen4_exp_qsa_norm_rope_rows_f32
    )
    positions_ptr = (
        metadata.positions.ptr if rope_positions_ptr is None else int(rope_positions_ptr)
    )
    if cache_only:
        # Prompt cache construction never consumes queries or gates. The
        # standalone norm/rope uses the split kernel's per-key reduction.
        norm_rope_rows(
            scratch.key_projected.ptr,
            weights.k_norm_weight_ptr,
            positions_ptr,
            scratch.key.ptr,
            count,
            kv_heads,
            head_dim,
            rotary_dim,
            theta,
            eps,
            stream=stream,
            runtime=active_runtime,
        )
    else:
        split_rope_rows(
            scratch.q_projected.ptr,
            scratch.key_projected.ptr,
            weights.q_norm_weight_ptr,
            weights.k_norm_weight_ptr,
            positions_ptr,
            scratch.query.ptr,
            scratch.key.ptr,
            0 if strided_gate else scratch.gate.ptr,
            count,
            query_heads,
            kv_heads,
            head_dim,
            rotary_dim,
            theta,
            eps,
            stream=stream,
            runtime=active_runtime,
        )
        norm_rope_rows(
            scratch.index_q_projected.ptr,
            weights.index_q_norm_weight_ptr,
            metadata.positions.ptr
            if rope_positions_ptr is None
            else int(rope_positions_ptr),
            scratch.index_query.ptr,
            count,
            index_heads,
            index_dim,
            index_rotary_dim,
            theta,
            eps,
            stream=stream,
            runtime=active_runtime,
        )
    index_state.append_rows(
        scratch.index_k_projected.ptr,
        start_position=start,
        rows=count,
        block_table_ptr=attention_state.block_table.ptr,
        block_size=attention_state.block_size,
        stream=stream,
    )
    qwen35_write_paged_kv_f32_batch_spans(
        scratch.key.ptr,
        scratch.value_projected.ptr,
        attention_state.key_cache.ptr,
        attention_state.value_cache.ptr,
        metadata.spans(start_row=0, rows=count, decode=False),
        count,
        attention_state.block_size,
        kv_heads,
        head_dim,
        stream=stream,
        runtime=active_runtime,
    )
    if cache_only:
        return None
    dense_rows = max(0, min(count, index_state.dense_equivalent_limit - start))
    qsa_flash = (
        dense_rows > 0
        and head_dim == 256
        and kv_heads * head_dim == 512
        and os.environ.get("HIPENGINE_QWEN4_EXP_QSA_FLASH_PREFILL", "0")
        not in {"", "0", "false", "False"}
        and _qwen4_exp_layer_allowed(
            weights.projections["attn_q"],
            env_name="HIPENGINE_QWEN4_EXP_QSA_FLASH_LAYERS",
            default="all",
        )
    )
    # WMMA attention (fast profile): F16 K/V gathered once per chunk for
    # this layer, then one dense launch and one masked sparse launch.
    qsa_wmma = (
        os.environ.get(_QSA_WMMA_ENV, "0") not in {"", "0", "false", "False"}
        and qsa_wmma_geometry_supported(
            query_heads=query_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            compression_ratio=index_state.compression_ratio,
            context_tokens=start + count,
        )
    )
    if qsa_wmma:
        context_len = start + count
        kv_bytes = context_len * kv_heads * head_dim * DType.FP16.itemsize
        mask_words = qsa_wmma_mask_words(context_len)
        mask_bytes = count * mask_words * DType.INT32.itemsize
        for name, nbytes in (
            ("wmma_k_f16", kv_bytes),
            ("wmma_v_f16", kv_bytes),
            ("wmma_mask", mask_bytes),
        ):
            owned = getattr(scratch, name, None)
            if owned is None or owned.nbytes < nbytes:
                if owned is not None:
                    free(owned, runtime=active_runtime)
                setattr(scratch, name, malloc(nbytes, runtime=active_runtime))
        qwen4_exp_qsa_wmma_gather_kv_f16(
            attention_state.key_cache.ptr,
            attention_state.value_cache.ptr,
            metadata.spans(start_row=0, rows=1, decode=True),
            scratch.wmma_k_f16.ptr,
            scratch.wmma_v_f16.ptr,
            block_size=attention_state.block_size,
            first_position=0,
            context_len=context_len,
            stream=stream,
            runtime=active_runtime,
        )
        if dense_rows:
            qwen4_exp_qsa_wmma_attention_f32(
                scratch.query.ptr,
                scratch.wmma_k_f16.ptr,
                scratch.wmma_v_f16.ptr,
                scratch.context.ptr,
                rows=dense_rows,
                start_position=start,
                stream=stream,
                runtime=active_runtime,
            )
    elif qsa_flash:
        context_len = start + dense_rows
        kv_elems = context_len * kv_heads * head_dim
        kv_bytes = kv_elems * DType.BF16.itemsize
        k_scratch = getattr(scratch, "flash_k_scratch", None)
        if k_scratch is None or k_scratch.nbytes < kv_bytes:
            if k_scratch is not None:
                free(k_scratch, runtime=active_runtime)
                free(getattr(scratch, "flash_v_scratch"), runtime=active_runtime)
            k_scratch = malloc(kv_bytes, runtime=active_runtime)
            scratch.flash_k_scratch = k_scratch
            scratch.flash_v_scratch = malloc(kv_bytes, runtime=active_runtime)
        block_table_ptr = (
            metadata.block_tables.ptr
            + 0 * metadata.block_table_len * DType.INT32.itemsize
        )
        qwen4_exp_qsa_flash_prefill(
            scratch.query.ptr,
            attention_state.key_cache.ptr,
            attention_state.value_cache.ptr,
            block_table_ptr,
            metadata.positions.ptr,
            scratch.flash_k_scratch.ptr,
            scratch.flash_v_scratch.ptr,
            scratch.context.ptr,
            dense_rows,
            query_heads,
            kv_heads,
            head_dim,
            attention_state.block_size,
            metadata.block_table_len,
            context_len,
            head_dim ** -0.5,
            stream=stream,
            runtime=active_runtime,
        )
    elif dense_rows:
        if _qwen4_exp_qsa_dense_fixed256_enabled(dense_rows):
            if _qwen4_exp_fullattn_context_legacy_enabled():
                # Temporary A/B opt-out back to the pre-promotion route. See
                # docs/REFACTOR.md for the removal trigger.
                dense_attention = (
                    qwen35_paged_full_attn_decode_context_bf16_batch_fixed256_spans
                )
                kv_capacity_tokens = 0
            else:
                # The promoted route needs the KV capacity to decide whether the
                # per-token offset table provably fits int32. The buffer is one
                # layer of bf16 KV, so tokens = bytes / (kv_heads * head_dim * 2).
                dense_attention = (
                    qwen35_paged_full_attn_decode_context_bf16_batch_fixed256_promoted_spans
                )
                kv_capacity_tokens = attention_state.key_cache.nbytes // (
                    kv_heads * head_dim * 2
                )
        else:
            dense_attention = qwen35_paged_full_attn_decode_context_bf16_batch_spans
            kv_capacity_tokens = 0
        dense_attention(
            scratch.query.ptr,
            attention_state.key_cache.ptr,
            attention_state.value_cache.ptr,
            scratch.context.ptr,
            metadata.spans(start_row=0, rows=dense_rows, decode=True),
            dense_rows,
            start + dense_rows,
            attention_state.block_size,
            query_heads,
            kv_heads,
            head_dim,
            head_dim ** -0.5,
            *((kv_capacity_tokens,) if kv_capacity_tokens else ()),
            stream=stream,
            runtime=active_runtime,
        )
    sparse_rows = count - dense_rows
    if sparse_rows:
        index_state.prepare_complete_blocks(
            (start + count) // index_state.compression_ratio,
            key_norm_weight_ptr=weights.index_k_norm_weight_ptr,
            rotary_dim=index_rotary_dim,
            theta=theta,
            eps=eps,
            stream=stream,
        )
        mask_direct = False
        if batched_selection:
            score_blocks = (start + count) // index_state.compression_ratio
            if score_blocks > metadata.score_blocks:
                raise ValueError("QSA score width exceeds prefill metadata capacity")
            direct_select = (
                index_heads == 4 and index_dim == 128
                and _qwen4_exp_qsa_direct_select_enabled()
            )
            sparse_query_ptr = (
                scratch.index_query.ptr
                + dense_rows * index_heads * index_dim * DType.FP32.itemsize
            )
            if direct_select:
                # Scores only the complete blocks each row can see (the only
                # scores top-k reads); bit-identical to the tiled scorer.
                qwen4_exp_qsa_score_rows_f32(
                    sparse_query_ptr,
                    index_state.pooled_keys.ptr,
                    metadata.scores.ptr,
                    sparse_rows,
                    score_blocks,
                    start + dense_rows,
                    index_state.compression_ratio,
                    stream=stream,
                    runtime=active_runtime,
                )
            else:
                qwen4_exp_qsa_score_f32(
                    sparse_query_ptr,
                    index_state.pooled_keys.ptr,
                    metadata.scores.ptr,
                    sparse_rows,
                    score_blocks,
                    index_heads,
                    index_dim,
                    stream=stream,
                    runtime=active_runtime,
                )
            # With the WMMA attention only the selection mask and the last
            # row's positions (for decode) are consumed, so the mask is
            # written straight from the scores and only the last row expands.
            mask_direct = direct_select and qsa_wmma
            expand_first = sparse_rows - 1 if mask_direct else 0
            if mask_direct:
                qwen4_exp_qsa_topk_mask_rows_f32(
                    metadata.scores.ptr,
                    scratch.wmma_mask.ptr,
                    sparse_rows,
                    start + dense_rows,
                    score_blocks,
                    mask_words,
                    index_state.compression_ratio,
                    index_state.block_budget,
                    stream=stream,
                    runtime=active_runtime,
                )
            expand_row = dense_rows + expand_first
            qwen4_exp_qsa_topk_expand_rows_f32_i64(
                metadata.scores.ptr
                + expand_first * score_blocks * DType.FP32.itemsize,
                metadata.positions.ptr + expand_row * DType.INT64.itemsize,
                metadata.selected_positions.ptr
                + expand_row * metadata.selection_capacity * DType.INT64.itemsize,
                metadata.selected_counts.ptr + expand_row * DType.INT32.itemsize,
                sparse_rows - expand_first,
                score_blocks,
                metadata.selection_capacity,
                index_state.compression_ratio,
                index_state.block_budget,
                stream=stream,
                runtime=active_runtime,
            )
        else:
            for local_row in range(dense_rows, count):
                position = start + local_row
                selected_count = index_state.select_positions_device(
                    scratch.index_query.ptr
                    + local_row * index_heads * index_dim * DType.FP32.itemsize,
                    query_position=position,
                    output_positions_ptr=metadata.selected_positions.ptr
                    + local_row
                    * metadata.selection_capacity
                    * DType.INT64.itemsize,
                    output_count_ptr=metadata.selected_counts.ptr
                    + local_row * DType.INT32.itemsize,
                    key_norm_weight_ptr=weights.index_k_norm_weight_ptr,
                    rotary_dim=index_rotary_dim,
                    theta=theta,
                    eps=eps,
                    stream=stream,
                )
                if selected_count > metadata.selection_capacity:
                    raise ValueError("QSA selection exceeds prefill metadata capacity")
        last_row = count - 1
        active_runtime.memcpy_async(
            index_state.selected_positions.ptr,
            metadata.selected_positions.ptr
            + last_row * metadata.selection_capacity * DType.INT64.itemsize,
            index_state.selected_positions.nbytes,
            HipMemcpyKind.DEVICE_TO_DEVICE,
            stream,
        )
        active_runtime.memcpy_async(
            index_state.selected_count.ptr,
            metadata.selected_counts.ptr + last_row * DType.INT32.itemsize,
            DType.INT32.itemsize,
            HipMemcpyKind.DEVICE_TO_DEVICE,
            stream,
        )
        if qsa_wmma:
            selected_offset = (
                dense_rows * metadata.selection_capacity * DType.INT64.itemsize
            )
            if not mask_direct:
                qwen4_exp_qsa_wmma_selection_mask(
                    metadata.selected_positions.ptr + selected_offset,
                    metadata.selected_counts.ptr + dense_rows * DType.INT32.itemsize,
                    scratch.wmma_mask.ptr,
                    rows=sparse_rows,
                    selected_stride=metadata.selection_capacity,
                    mask_words=mask_words,
                    stream=stream,
                    runtime=active_runtime,
                )
            qwen4_exp_qsa_wmma_attention_f32(
                scratch.query.ptr + dense_rows * q_width * DType.FP32.itemsize,
                scratch.wmma_k_f16.ptr,
                scratch.wmma_v_f16.ptr,
                scratch.context.ptr + dense_rows * q_width * DType.FP32.itemsize,
                rows=sparse_rows,
                start_position=start + dense_rows,
                mask_ptr=scratch.wmma_mask.ptr,
                mask_words=mask_words,
                stream=stream,
                runtime=active_runtime,
            )
        sparse_attention_rows = qwen4_exp_qsa_sparse_attention_paged_bf16_rows_f32
        backend = str(weights.projections["attn_q"].backend)
        if qwen4_exp_qsa_h256_wave_prefill_selected(
            head_dim=head_dim, rows=sparse_rows, backend=backend,
            block_size=attention_state.block_size, published=h256_mode,
        ):
            sparse_key=_qwen4_exp_qsa_prefill_head_key(
                KernelKey(backend,"qsa_sparse_attention","bf16_kv",
                          qwen4_exp_qsa_h256_wave_prefill_variant(h256_mode)),
                query_heads=query_heads,kv_heads=kv_heads,mode=head_mode)
            sparse_attention_rows = resolve(
                backend=sparse_key.backend,layer=sparse_key.layer,
                quant=sparse_key.quant,variant=sparse_key.variant)
        elif _qwen4_exp_qsa_wave32_selected(
            head_dim=head_dim, published=qsa_wave32
        ):
            sparse_attention_rows = (
                qwen4_exp_qsa_sparse_attention_paged_bf16_rows_wave32_f32
            )
        if not qsa_wmma:
            sparse_attention_rows(
                scratch.query.ptr + dense_rows * q_width * DType.FP32.itemsize,
                attention_state.key_cache.ptr,
                attention_state.value_cache.ptr,
                metadata.selected_positions.ptr
                + dense_rows * metadata.selection_capacity * DType.INT64.itemsize,
                metadata.selected_counts.ptr + dense_rows * DType.INT32.itemsize,
                scratch.context.ptr + dense_rows * q_width * DType.FP32.itemsize,
                metadata.spans(start_row=dense_rows, rows=sparse_rows, decode=True),
                rows=sparse_rows,
                selected_stride=metadata.selection_capacity,
                block_size=attention_state.block_size,
                query_heads=query_heads,
                kv_heads=kv_heads,
                head_dim=head_dim,
                scale=head_dim ** -0.5,
                stream=stream,
                runtime=active_runtime,
            )
    gated_f16 = 0
    if strided_gate:
        # The gated context's F16 image goes to the wide-route workspace and
        # is declared, so the output projection skips its own conversion.
        workspace = wide_f16_activation_workspace()
        if workspace is not None and workspace.nbytes >= count * q_width * 2:
            _invalidate_wide_f16_publication()
            gated_f16 = workspace.ptr
        qwen4_exp_qsa_gate_context_strided_f32(
            scratch.context.ptr,
            scratch.q_projected.ptr,
            scratch.gated.ptr,
            count * q_width,
            head_dim,
            output_f16_ptr=gated_f16,
            stream=stream,
            runtime=active_runtime,
        )
        if gated_f16:
            declare_f16_mirror(scratch.gated.ptr, count * q_width, gated_f16)
    else:
        qwen4_exp_qsa_gate_context_f32(
            scratch.context.ptr,
            scratch.gate.ptr,
            scratch.gated.ptr,
            count * q_width,
            stream=stream,
            runtime=active_runtime,
        )
    launch_gguf_linear(
        weights.projections["attn_output"],
        scratch.gated.ptr,
        scratch.output.ptr,
        count,
        q_width,
        hidden,
        activation_dtype=GGUF_ACTIVATION_F32,
        output_dtype=GGUF_OUTPUT_F32,
        stream=stream,
        runtime=active_runtime,
    )
    if gated_f16:
        invalidate_f16_mirror()
    attention_state.set_position(start + count - 1)
    return scratch.output


def _qwen4_exp_gr_pending_write(
    rows: int, source_ptr: int, block_ptr: int, logits_ptr: int,
) -> tuple[int, int, int, bool] | None:
    """Defer a prefill residual write into the next read's norm launch.

    Bit-identical (`tests/test_gpu_qwen4exp_gr_write_norm.py`). Chunks of at
    least 1024 rows only: the fused launch has one workgroup per row, and at
    512 rows the separate per-branch write fills the GPU better (code-p512
    1027.9 against 1021.7 tok/s). HIPENGINE_QWEN4_EXP_GR_WRITE_NORM=0 rolls
    back (docs/REFACTOR.md).
    """

    if rows < 1024 or os.environ.get(
        "HIPENGINE_QWEN4_EXP_GR_WRITE_NORM", "1"
    ) in {"", "0", "false", "False"}:
        return None
    return (int(source_ptr), int(block_ptr), int(logits_ptr), False)


def _qwen4_exp_gr_defer_write(
    defer_write: list | None, rows: int, source_ptr: int, block_ptr: int,
    logits_ptr: int,
) -> bool:
    """Hand a layer's final BF16 residual write to the next layer's read."""

    if defer_write is None or _qwen4_exp_gr_pending_write(rows, 1, 1, 1) is None:
        return False
    defer_write.append((int(source_ptr), int(block_ptr), int(logits_ptr), True))
    return True


def _qwen4_exp_gr_flush_write(
    pending_write: tuple[int, int, int, bool], output_ptr: int, rows: int,
    branches: int, hidden: int, *, stream: int, runtime: HipRuntime,
) -> None:
    """Launch a deferred residual write on its own."""

    source_ptr, block_ptr, logits_ptr, block_bf16 = pending_write
    write = qwen4_exp_gr_write_bf16_bf16 if block_bf16 else qwen4_exp_gr_write_bf16_f32
    write(source_ptr, block_ptr, logits_ptr, output_ptr, rows, branches, hidden,
          stream=stream, runtime=runtime)


def _run_qwen4_exp_gr_norm_down(
    residual_ptr, norm_weight_ptr, down_weight, scratch, *,
    rows, branches, hidden, low_rank, eps, stream, runtime,
    scale_ptr: int = 0,
    inject_weight_ptr: int = 0,
    inject_out_ptr: int = 0,
    pending_write: tuple[int, int, int, bool] | None = None,
) -> bool:
    """Publish only for a qualified immediate HC-down consumer; else parent.

    ``pending_write`` is a residual write ``(source, block, inject_logits,
    block_bf16)`` whose output is ``residual_ptr``; it runs before the norm
    reads ``residual_ptr``, fused into the norm launch where it can be.

    With ``scale_ptr`` the direct launch may skip the FP32 normalized plane
    and store ``inv_rms[rows, branches]`` there instead; the return value says
    whether it did, in which case ``scratch.normalized`` is not written.
    """
    width = branches * hidden
    count = rows * width
    iu8_down = (
        _qwen4_exp_gr_iu8_down_enabled(rows)
        and down_weight.spec.quant_key == "gguf_q8_0"
        and width % 32 == 0
    )
    workspace = wide_f16_activation_workspace()
    if (_qwen4_exp_gr_f16_direct_enabled(rows, low_rank) and not iu8_down
            and workspace is not None and workspace.nbytes >= count * 2
            and down_weight.spec.quant_key == "gguf_q8_0"):
        producer = KernelKey(str(down_weight.backend), "gr_grouped_rmsnorm",
                             "bf16_f32", "f32_f16_publish")
        # Only where the ordinary dispatch already runs this projection on the
        # F16 wide tile: the direct launch then changes the tile shape and who
        # converts the activation, not a single output bit.
        if is_registered(producer) and _wide_f16_consumer_key(
                down_weight, rows=rows, in_features=width,
                out_features=low_rank) is not None:
            _invalidate_wide_f16_publication()
            if pending_write is not None and not (scale_ptr and inject_out_ptr):
                _qwen4_exp_gr_flush_write(
                    pending_write, residual_ptr, rows, branches, hidden,
                    stream=stream, runtime=runtime)
                pending_write = None
            if scale_ptr and inject_out_ptr and pending_write is not None:
                # The residual write and the norm in one launch: the written
                # row is normalized from LDS instead of being read back.
                source_ptr, block_ptr, logits_ptr, block_bf16 = pending_write
                qwen4_exp_gr_write_norm_inject(
                    source_ptr, block_ptr, logits_ptr, residual_ptr,
                    norm_weight_ptr, inject_weight_ptr, workspace.ptr,
                    scale_ptr, inject_out_ptr, rows, branches, hidden, eps,
                    block_bf16=block_bf16, stream=stream, runtime=runtime,
                )
                pending_write = None
            elif scale_ptr and inject_out_ptr:
                # One block per row: the norm and the inject logits from a
                # single read of the residual.
                qwen4_exp_grouped_rmsnorm_bf16_f16_scale_inject(
                    residual_ptr, norm_weight_ptr, inject_weight_ptr,
                    workspace.ptr, scale_ptr, inject_out_ptr,
                    rows, branches, hidden, eps, stream=stream, runtime=runtime,
                )
            elif scale_ptr:
                qwen4_exp_grouped_rmsnorm_bf16_f16_scale(
                    residual_ptr, norm_weight_ptr, workspace.ptr, scale_ptr,
                    rows, branches, hidden, eps, stream=stream, runtime=runtime,
                )
            else:
                resolve(backend=producer.backend, layer=producer.layer,
                        quant=producer.quant, variant=producer.variant)(
                    residual_ptr, norm_weight_ptr, scratch.normalized.ptr,
                    workspace.ptr, rows, branches, hidden, eps,
                    stream=stream, runtime=runtime,
                )
            gguf_q8_0_dense_wide128x128_f16in_f32_f32_out(
                workspace.ptr, down_weight.allocation("raw").tensor.ptr,
                scratch.low_rank.ptr, rows, width, low_rank,
                stream=stream, runtime=runtime,
            )
            return bool(scale_ptr)
    if pending_write is not None:
        _qwen4_exp_gr_flush_write(
            pending_write, residual_ptr, rows, branches, hidden,
            stream=stream, runtime=runtime)
    keys = None
    unsafe_workspace = False
    if (workspace is not None and workspace.nbytes >= count * 2
            and (rows, branches, hidden, low_rank) == (4096, 4, 2560, 320)
            and not iu8_down):
        keys = wide_f16_publication_keys(
            down_weight, rows=rows, in_features=width, out_features=low_rank,
        )
    if keys is not None:
        try:
            # Validate every producer range before a write; refused admission
            # uses the ordinary parent operation, not a partial publication.
            _check_norm_publication(residual_ptr, norm_weight_ptr,
                scratch.normalized.ptr, workspace.ptr, rows, branches, hidden)
            raw = down_weight.allocation("raw")
            ranges = ((scratch.low_rank.ptr, rows * low_rank * 4),
                      (raw.tensor.ptr, raw.buffer.nbytes))
            if any(workspace.ptr < ptr + size and ptr < workspace.ptr + count * 2
                   for ptr, size in ranges):
                keys = None
                unsafe_workspace = True
        except ValueError:
            keys = None
            unsafe_workspace = True
    if keys is not None:
        producer, consumer = keys
        # Enter before publication so nesting invalidates an earlier loan
        # before overwriting it. Only the immediately following launch can read.
        with wide_f16_activation_publication(
            scratch.normalized.ptr, count, stream=stream, runtime=runtime,
            consumer_key=consumer,
        ):
            resolve(backend=producer.backend, layer=producer.layer,
                    quant=producer.quant, variant=producer.variant)(
                residual_ptr, norm_weight_ptr, scratch.normalized.ptr,
                workspace.ptr, rows, branches, hidden, eps,
                stream=stream, runtime=runtime,
            )
            launch_gguf_linear(down_weight, scratch.normalized.ptr, scratch.low_rank.ptr,
                rows, width, low_rank, activation_dtype=GGUF_ACTIVATION_F32,
                output_dtype=GGUF_OUTPUT_F32, stream=stream, runtime=runtime)
        return False
    qwen4_exp_grouped_rmsnorm_bf16_f32(
        residual_ptr, norm_weight_ptr, scratch.normalized.ptr, rows, branches, hidden, eps,
        stream=stream, runtime=runtime,
    )
    if iu8_down:
        gguf_q8_0_iu8_wmma_prefill_f32_f32(
            scratch.normalized.ptr, down_weight.allocation("raw").tensor.ptr,
            scratch.low_rank.ptr, rows, width, low_rank, stream=stream, runtime=runtime,
        )
    else:
        # Publication refusal must not let ordinary staging write the same
        # unsafe alias. Shadow only this launch; the session owner is restored.
        with (wide_f16_activation_session(False) if unsafe_workspace else contextlib.nullcontext()):
            launch_gguf_linear(down_weight, scratch.normalized.ptr, scratch.low_rank.ptr,
                rows, width, low_rank, activation_dtype=GGUF_ACTIVATION_F32,
                output_dtype=GGUF_OUTPUT_F32, stream=stream, runtime=runtime)
    return False


def run_qwen4_exp_gr_read(
    residual_ptr: int,
    norm_weight_ptr: int,
    down_weight: GGUFDeviceWeight,
    up_weight: GGUFDeviceWeight,
    inject_weight: GGUFDeviceWeight | None,
    scratch: Qwen4ExpGRScratch,
    *,
    rows: int,
    branches: int,
    hidden: int,
    low_rank: int,
    eps: float = 1e-6,
    stream: int = 0,
    runtime: HipRuntime | None = None,
    pending_write: tuple[int, int, int, bool] | None = None,
    decode_rows: bool = False,
) -> Qwen4ExpGRReadDeviceResult:
    """Execute strict grouped GR read using model-neutral resident projections.

    ``pending_write`` = ``(source, block, inject_logits, block_bf16)`` is the
    residual write that produces ``residual_ptr``; the read launches it first,
    or inside its norm launch where that is bit-identical.
    """

    if scratch.closed:
        raise RuntimeError("Qwen4Exp GR scratch is closed")
    if rows <= 0 or branches <= 0 or hidden <= 0 or low_rank <= 0:
        raise ValueError("rows, branches, hidden, and low_rank must be positive")
    active_runtime = runtime or scratch.runtime
    if active_runtime is not scratch.runtime:
        raise ValueError("runtime must match the GR scratch owner")
    # Every read rewrites ``mixed``; an image declared by an earlier read must
    # not outlive it.
    invalidate_f16_mirror()
    residual_width = branches * hidden
    if (
        _qwen4_exp_gr_decode_fused_enabled(1 if decode_rows else rows, branches, hidden, low_rank)
        and down_weight.spec.quant_key == "gguf_q8_0"
        and up_weight.spec.quant_key == "gguf_q8_0"
        and "raw" in down_weight.allocations
        and "raw" in up_weight.allocations
        and (inject_weight is None or (
            inject_weight.spec.quant_key == "f32"
            and "raw" in inject_weight.allocations
        ))
    ):
        if pending_write is not None:
            _qwen4_exp_gr_flush_write(
                pending_write, residual_ptr, rows, branches, hidden,
                stream=stream, runtime=active_runtime)
        # Three launches for the whole read, bit-identical to the six-launch
        # chain: the grouped norm, then the down projection with the scaled
        # SiLU and the inject logits, then the up projection with the gated
        # mean. ``scratch.gate`` is not written.
        qwen4_exp_grouped_rmsnorm_bf16_f32(
            residual_ptr, norm_weight_ptr, scratch.normalized.ptr, rows, branches,
            hidden, eps, stream=stream, runtime=active_runtime,
        )
        qwen4_exp_gr_down_pack8_decode(
            scratch.normalized.ptr,
            down_weight.allocation("raw").tensor.ptr,
            0 if inject_weight is None else inject_weight.allocation("raw").tensor.ptr,
            scratch.low_rank.ptr,
            scratch.inject_logits.ptr,
            rows,
            residual_width,
            low_rank,
            1.0 / branches,
            stream=stream,
            runtime=active_runtime,
        )
        qwen4_exp_gr_up_mean_pack8_decode(
            scratch.low_rank.ptr,
            up_weight.allocation("raw").tensor.ptr,
            scratch.normalized.ptr,
            scratch.mixed.ptr,
            rows,
            branches,
            hidden,
            low_rank,
            stream=stream,
            runtime=active_runtime,
        )
        return Qwen4ExpGRReadDeviceResult(
            normalized=scratch.normalized,
            gate=scratch.gate,
            mixed=scratch.mixed,
            inject_logits=scratch.inject_logits,
        )
    if decode_rows:
        # Preserve the scalar fallback when the row-independent fused kernels
        # do not cover the geometry or their rollback switch is selected.
        if pending_write is not None:
            _qwen4_exp_gr_flush_write(pending_write, residual_ptr, rows, branches,
                hidden, stream=stream, runtime=active_runtime)
        widths = {"normalized": residual_width, "low_rank": low_rank,
                  "gate": residual_width, "inject_logits": branches, "mixed": hidden}
        for row in range(rows):
            views = {name: DeviceBuffer(getattr(scratch, name).ptr + row * width * 4, width * 4)
                     for name, width in widths.items()}
            run_qwen4_exp_gr_read(residual_ptr + row * residual_width * 2,
                norm_weight_ptr, down_weight, up_weight, inject_weight,
                Qwen4ExpGRScratch(**views, runtime=active_runtime), rows=1,
                branches=branches, hidden=hidden, low_rank=low_rank, eps=eps,
                stream=stream, runtime=active_runtime)
        return Qwen4ExpGRReadDeviceResult(normalized=scratch.normalized,
            gate=scratch.gate, mixed=scratch.mixed, inject_logits=scratch.inject_logits)
    # Without the FP32 normalized plane: the up epilogue and the inject logits
    # recompute bf16(residual) * inv_rms * weight, the value the norm stores,
    # so nothing downstream changes. Requires the fused up epilogue and an
    # inject projection the residual kernel covers; ``scratch.gate`` is unused
    # on this path and holds inv_rms[rows, branches].
    scale_only = (
        _qwen4_exp_gr_f16_direct_enabled(rows, low_rank)
        and _qwen4_exp_gr_wide_enabled(rows, low_rank)
        and not _qwen4_exp_gr_iu8_enabled(rows)
        and branches == 4
        and hidden % 32 == 0
        and residual_width % 2048 == 0
        and up_weight.spec.quant_key == "gguf_q8_0"
        and (inject_weight is None or inject_weight.spec.quant_key == "f32")
        and os.environ.get("HIPENGINE_QWEN4_EXP_GR_SCALE_ONLY", "1")
        not in {"", "0", "false", "False"}
    )
    # The norm launch also computes the inject logits when it runs
    # scale-only and there is an inject projection; bit-identical to the
    # separate launch. HIPENGINE_QWEN4_EXP_GR_NORM_INJECT=0 keeps them apart.
    norm_inject = (
        scale_only
        and inject_weight is not None
        and os.environ.get("HIPENGINE_QWEN4_EXP_GR_NORM_INJECT", "1")
        not in {"", "0", "false", "False"}
    )
    scale_only = _run_qwen4_exp_gr_norm_down(
        residual_ptr, norm_weight_ptr, down_weight, scratch,
        rows=rows, branches=branches, hidden=hidden, low_rank=low_rank,
        eps=eps, stream=stream, runtime=active_runtime,
        scale_ptr=scratch.gate.ptr if scale_only else 0,
        inject_weight_ptr=(
            inject_weight.allocation("raw").tensor.ptr if norm_inject else 0
        ),
        inject_out_ptr=scratch.inject_logits.ptr if norm_inject else 0,
        pending_write=pending_write,
    )
    norm_inject = norm_inject and scale_only
    qwen4_exp_scaled_silu_f32(
        scratch.low_rank.ptr,
        scratch.low_rank.ptr,
        rows * low_rank,
        1.0 / branches,
        stream=stream,
        runtime=active_runtime,
    )
    fused_up_key = KernelKey(
        str(up_weight.backend),
        "linear+gr_gated_mean",
        up_weight.spec.quant_key,
        "coltile2_branch4_rowbatch4_f32_exact",
    )
    fused_up = (
        _qwen4_exp_gr_up_sigmoid_mean_enabled(rows)
        and branches == 4
        and is_registered(fused_up_key)
    )
    gr_iu8 = fused_up and _qwen4_exp_gr_iu8_enabled(rows)
    gr_wide = fused_up and not gr_iu8 and _qwen4_exp_gr_wide_enabled(rows, low_rank)
    if scale_only and not gr_wide:
        # ``scratch.normalized`` was not written; only the fused wide up
        # epilogue can run without it.
        raise RuntimeError("Qwen4Exp GR scale-only norm requires the wide up route")
    if gr_wide:
        # Measured 4.38x faster than the fused exact kernel at rows=1024,
        # K=320, N=10240 (1.4925 ms against 6.5379 ms, real weights). The
        # projection is the same tile the dense route already ships. The
        # epilogue applies the sigmoid inside the gated-mean pass, which is
        # bit-identical to the separate pair and does not write the gate:
        # no caller reads it, and ``scratch.gate`` keeps the logits here.
        workspace = wide_f16_activation_workspace()
        if (_qwen4_exp_gr_f16_direct_enabled(rows, low_rank)
                and workspace is not None
                and workspace.nbytes >= rows * low_rank * 2
                and hidden % 32 == 0):
            # The F32-input tile converts the activation to F16 while staging
            # it; converting once up front feeds the same bytes to the same
            # products. The fused kernel then applies the sigmoid gated mean
            # to the logits while they are still in registers, so neither the
            # logits nor the gate reach ``scratch.gate``; no caller reads it.
            _invalidate_wide_f16_publication()
            gguf_q8_0_f32_to_f16(
                scratch.low_rank.ptr, workspace.ptr, rows * low_rank,
                stream=stream, runtime=active_runtime,
            )
            if scale_only:
                # ``scratch.normalized`` is unwritten on this path; it holds
                # half(mixed), declared so the F16-operand projections that
                # read ``mixed`` next skip their own conversion.
                mirror = _qwen4_exp_gr_f16_mirror_enabled()
                qwen4_exp_gr_up_mean_residual_f16in_f32_out(
                    workspace.ptr,
                    up_weight.allocation("raw").tensor.ptr,
                    residual_ptr,
                    scratch.gate.ptr,
                    norm_weight_ptr,
                    scratch.mixed.ptr,
                    rows,
                    low_rank,
                    branches,
                    hidden,
                    out_f16_ptr=scratch.normalized.ptr if mirror else 0,
                    stream=stream,
                    runtime=active_runtime,
                )
                if mirror:
                    declare_f16_mirror(
                        scratch.mixed.ptr, rows * hidden, scratch.normalized.ptr
                    )
            else:
                qwen4_exp_gr_up_mean_f16in_f32_out(
                    workspace.ptr,
                    up_weight.allocation("raw").tensor.ptr,
                    scratch.normalized.ptr,
                    scratch.mixed.ptr,
                    rows,
                    low_rank,
                    branches,
                    hidden,
                    stream=stream,
                    runtime=active_runtime,
                )
        elif scale_only:
            raise RuntimeError(
                "Qwen4Exp GR scale-only norm ran without the fused up epilogue"
            )
        else:
            gguf_q8_0_dense_wide256_f32_f32_out(
                scratch.low_rank.ptr,
                up_weight.allocation("raw").tensor.ptr,
                scratch.gate.ptr,
                rows,
                low_rank,
                residual_width,
                stream=stream,
                runtime=active_runtime,
            )
            qwen4_exp_gated_mean_sigmoid_f32(
                scratch.normalized.ptr,
                scratch.gate.ptr,
                scratch.mixed.ptr,
                rows,
                branches,
                hidden,
                stream=stream,
                runtime=active_runtime,
                store_gate=False,
            )
    elif gr_iu8:
        # T1 candidate: iu8-WMMA projection + separate sigmoid/gated-mean
        # epilogue (the unfused epilogue kernels are the retained exact ones).
        gguf_q8_0_iu8_wmma_prefill_f32_f32(
            scratch.low_rank.ptr,
            up_weight.allocation("raw").tensor.ptr,
            scratch.gate.ptr,
            rows,
            low_rank,
            residual_width,
            stream=stream,
            runtime=active_runtime,
        )
        qwen4_exp_sigmoid_f32(
            scratch.gate.ptr,
            scratch.gate.ptr,
            rows * residual_width,
            stream=stream,
            runtime=active_runtime,
        )
        qwen4_exp_gated_mean_f32(
            scratch.normalized.ptr,
            scratch.gate.ptr,
            scratch.mixed.ptr,
            rows,
            branches,
            hidden,
            stream=stream,
            runtime=active_runtime,
        )
    elif fused_up:
        fused_up_key = _qwen4_exp_gr_wave_scale_key(fused_up_key, rows=rows, branches=branches)
        fused_up_key = _qwen4_exp_gr_k320_full_rows_key(
            fused_up_key, rows=rows, branches=branches, low_rank=low_rank,
            hidden=hidden,
        )
        resolve(
            backend=fused_up_key.backend,
            layer=fused_up_key.layer,
            quant=fused_up_key.quant,
            variant=fused_up_key.variant,
        )(
            scratch.low_rank.ptr,
            up_weight.allocation("raw").tensor.ptr,
            scratch.normalized.ptr,
            scratch.gate.ptr,
            scratch.mixed.ptr,
            rows,
            low_rank,
            branches,
            hidden,
            stream=stream,
            runtime=active_runtime,
        )
    else:
        launch_gguf_linear(
            up_weight,
            scratch.low_rank.ptr,
            scratch.gate.ptr,
            rows,
            low_rank,
            residual_width,
            activation_dtype=GGUF_ACTIVATION_F32,
            output_dtype=GGUF_OUTPUT_F32,
            stream=stream,
            runtime=active_runtime,
        )
    fused_sigmoid_mean = not fused_up and _qwen4_exp_gr_sigmoid_mean_fused(rows)
    if not fused_up and not fused_sigmoid_mean:
        qwen4_exp_sigmoid_f32(
            scratch.gate.ptr,
            scratch.gate.ptr,
            rows * residual_width,
            stream=stream,
            runtime=active_runtime,
        )
    if inject_weight is None:
        if stream:
            active_runtime.memset_async(
                scratch.inject_logits.ptr, 0, rows * branches * 4, stream
            )
        else:
            active_runtime.memset(scratch.inject_logits.ptr, 0, rows * branches * 4)
    elif norm_inject:
        pass  # written by the norm launch
    elif scale_only:
        qwen4_exp_gr_inject_bf16_f32(
            residual_ptr,
            norm_weight_ptr,
            scratch.gate.ptr,
            inject_weight.allocation("raw").tensor.ptr,
            scratch.inject_logits.ptr,
            rows,
            branches,
            hidden,
            stream=stream,
            runtime=active_runtime,
        )
    else:
        launch_gguf_linear(
            inject_weight,
            scratch.normalized.ptr,
            scratch.inject_logits.ptr,
            rows,
            residual_width,
            branches,
            activation_dtype=GGUF_ACTIVATION_F32,
            output_dtype=GGUF_OUTPUT_F32,
            stream=stream,
            runtime=active_runtime,
        )
    if not fused_up:
        gated_mean = (
            qwen4_exp_gated_mean_sigmoid_f32
            if fused_sigmoid_mean
            else qwen4_exp_gated_mean_f32
        )
        gated_mean(
            scratch.normalized.ptr,
            scratch.gate.ptr,
            scratch.mixed.ptr,
            rows,
            branches,
            hidden,
            stream=stream,
            runtime=active_runtime,
        )
    return Qwen4ExpGRReadDeviceResult(
        normalized=scratch.normalized,
        gate=scratch.gate,
        mixed=scratch.mixed,
        inject_logits=scratch.inject_logits,
    )


def _qwen4_exp_qsa_strided_gate_enabled() -> bool:
    """Prefill gate multiply reads the gate from the q projection and writes
    the output projection's F16 image. Bit-identical; the env switch is a
    rollback (docs/REFACTOR.md)."""

    return os.environ.get(
        "HIPENGINE_QWEN4_EXP_QSA_STRIDED_GATE", "1"
    ) not in {"", "0", "false", "False"}


def _qwen4_exp_qsa_direct_select_enabled() -> bool:
    """Causal-only block scoring and, with the WMMA attention, a selection
    mask written straight from the scores. Bit-identical selection; the env
    switch is a rollback (docs/REFACTOR.md)."""

    return os.environ.get(
        "HIPENGINE_QWEN4_EXP_QSA_DIRECT_SELECT", "1"
    ) not in {"", "0", "false", "False"}


def _qwen4_exp_qsa_dense_fixed256_enabled(rows: int) -> bool:
    return rows >= 2


def _qwen4_exp_fullattn_context_legacy_enabled() -> bool:
    """Temporary A/B opt-out back to the pre-promotion context route.

    Set ``HIPENGINE_QWEN4_EXP_FULLATTN_CONTEXT_LEGACY=1`` to measure the shipped
    fixed256 route in the same session as the promoted route. Removal trigger is
    recorded in docs/REFACTOR.md.
    """

    return os.environ.get(
        "HIPENGINE_QWEN4_EXP_FULLATTN_CONTEXT_LEGACY", "0"
    ) not in {"", "0", "false", "False"}


def _qwen4_exp_q51_register_cache_key(
    key: KernelKey, *, rows: int, in_features: int, published: bool = False,
) -> KernelKey:
    """Select the Q5_1 register-cache sibling of the folded-pair owner.

    ``published`` is the owning runner's plan-published decision (see
    :func:`q51_register_cache`), not a process-wide switch: the production plan
    publishes the register-cache fallback for the rows>=512 K640 scope, and
    every plan that keeps the folded-pair parent publishes ``False``. The
    guards stay dispatch-time - rows, in_features, the folded-pair parent
    variant, and a registry that must actually carry the leaf - so an unbound
    runner or a missing registration keeps the parent. The downstream
    row-publish sibling is a separate decision that reads this leaf's variant.
    """

    if (rows < 512 or in_features != 640
            or key.variant != "selected_grouped_prefill_pair2_fold128_pair_bf16_bf16_out"
            or not published):
        return key
    candidate = KernelKey(key.backend,key.layer,key.quant,
                          "selected_grouped_prefill_pair2_register_cache_bf16_bf16_out")
    return candidate if is_registered(candidate) else key


# GDN serial-prefill norm/gate owner. The execution profile publishes the
# decision to the owning runner: the narrow checked-index leaf inside its
# Hk16/Hv48/D128 2..4096-row bounds, the wide wave-norm leaf wherever the
# wave-norm admission applies, and ``None`` (the register parent) everywhere
# else. An unbound runner, an unknown value, or a leaf the registry does not
# carry keeps the parent, and rows outside the narrow bound still select the
# wide leaf. Diagnostic arms bind the value on the runner they measure, so
# neither arm writes a process-wide switch.
GDN_PREFILL_PARENT_VARIANT = "qwen4exp_sigmoid_register_prefill"
GDN_PREFILL_WIDE_VARIANT = "qwen4exp_sigmoid_wave_norm_prefill"
GDN_PREFILL_NARROW_VARIANT = "qwen4exp_sigmoid_wave_norm_narrow_prefill"
_GDN_PREFILL_UNSET = object()


@contextlib.contextmanager
def gdn_prefill_variant(runner: Any, variant: str | None | object = _GDN_PREFILL_UNSET):
    """Bind an explicit serial-prefill owner for a diagnostic arm.

    ``None`` selects the register parent and a wide/narrow name selects that
    leaf; leaving ``variant`` unset keeps whatever the profile published. The
    binding is a mutable runner field, so it stays in force for as long as the
    scope is open - a prefill that suspends between layer yields keeps the arm
    bound - and the published value is restored on exit, including on an
    exception.
    """

    prior = getattr(runner, "gdn_prefill_variant", None)
    if variant is not _GDN_PREFILL_UNSET:
        runner.gdn_prefill_variant = variant
    try:
        yield prior
    finally:
        runner.gdn_prefill_variant = prior


# GDN tile-16 prefill selection. The production plan opens the columnwarp gate
# and the runner's own decision admits the tile-16 owner inside it for the
# binding Hk16/Hv48/D128 envelope; the columnwarp parent serves everything else
# and remains the immediate rollback, with the serial strict route registered
# below both. Diagnostics bind the parent on the runner they measure instead
# of changing process-wide state.
_GDN_TILE16_UNSET = object()


@contextlib.contextmanager
def gdn_tile16_prefill(runner: Any, enabled: bool | object = _GDN_TILE16_UNSET):
    """Bind the tile-16 GDN prefill branch for a diagnostic arm.

    ``False`` pins the columnwarp parent for the measured window and ``True``
    restores the production owner; leaving ``enabled`` unset keeps whatever the
    runner carries. The binding is a mutable runner field, so it stays in force
    across a prefill that suspends between layer yields, and the prior value is
    restored on exit, including on an exception.
    """

    prior = getattr(runner, "gdn_tile16_prefill", True)
    if enabled is not _GDN_TILE16_UNSET:
        runner.gdn_tile16_prefill = bool(enabled)
    try:
        yield prior
    finally:
        runner.gdn_tile16_prefill = prior


# GDN peer prefill ownership. The production plan publishes the certified global
# layer scope on the owning runner instead of a process-wide switch: inside that
# scope the peer owner reads the compact Q/K/V projections straight out of the
# pre-projection buffer, every layer outside it keeps the columnwarp parent, and
# the serial sigmoid owner stays registered below both. Strict, and every plan
# that falls back to the strict binder, publishes no scope at all. Diagnostics
# bind the scope on the runner they measure - `False` for the serial owners,
# `"all"` or a numeric set for an explicit one - which is the same state the
# plan publishes.
_GDN_PEER_UNSET = object()


def _gdn_peer_scope_value(scope: bool | str | None) -> str | None:
    if scope is None or scope is False:
        return None
    if scope is True:
        return GDN_PEER_CERTIFIED_SCOPE
    return str(scope)


@contextlib.contextmanager
def gdn_peer_prefill(
    runner: Any, scope: bool | str | None | object = _GDN_PEER_UNSET
):
    """Bind the GDN peer prefill owner's layer scope for a diagnostic arm.

    ``False`` or ``None`` binds the serial sigmoid owners for the measured
    window, ``True`` restores the certified global layers 35-47, and a string
    names an explicit scope (``"all"`` for every layer, ``"0-34"`` or
    ``"35,40,47"`` for a numeric set). Leaving ``scope`` unset keeps whatever
    the runner carries. The binding is a mutable runner field, so it stays in
    force across a prefill that suspends between layer yields, and the prior
    value is restored on exit, including on an exception.
    """

    prior = getattr(runner, "gdn_peer_prefill_layers", None)
    if scope is not _GDN_PEER_UNSET:
        runner.gdn_peer_prefill_layers = _gdn_peer_scope_value(scope)
    try:
        yield prior
    finally:
        runner.gdn_peer_prefill_layers = prior


# GDN conv prefill ownership. Production runs the exact bulk conv owner for
# every chunk wider than one row and keeps the per-row decode-conv loop below it
# as the arithmetic reference. It is request-local state: the prefill chunk path
# forwards the owning runner's decision to the token mixer, so a diagnostic that
# needs the serial comparison binds it on the runner it measures.
_GDN_EXACT_CONV_UNSET = object()


@contextlib.contextmanager
def gdn_exact_bulk_conv(
    runner: Any, enabled: bool | object = _GDN_EXACT_CONV_UNSET
):
    """Bind the GDN conv prefill owner for a diagnostic arm.

    ``False`` pins the per-row serial fallback for the measured window and
    ``True`` restores the exact bulk owner; leaving ``enabled`` unset keeps
    whatever the runner carries. The prior value is restored on exit, including
    on an exception.
    """

    prior = getattr(runner, "gdn_exact_bulk_conv", True)
    if enabled is not _GDN_EXACT_CONV_UNSET:
        runner.gdn_exact_bulk_conv = bool(enabled)
    try:
        yield prior
    finally:
        runner.gdn_exact_bulk_conv = prior


# Q5_1 row-publish selection. The production plan declares the row-publish
# owner for the rows>=512 K640 grouped Q5_1 down scope; the binder publishes
# that decision to the owning runner, and the dispatch-time selector keeps its
# rows/K640/parent-variant/registration guards. It is request-local state, so a
# diagnostic arm binds it on the runner it measures instead of writing a
# process-wide switch.
# Q5_1 register-cache selection. The production plan publishes the
# register-cache fallback owner for the rows>=512 K640 grouped Q5_1 down scope;
# the binder publishes that decision to the owning runner, and the dispatch-time
# selector keeps its rows/K640/folded-pair-parent/registration guards. It is the
# parent of the row-publish sibling, which replaces it wherever that plan
# publishes the row-publish decision. It is request-local state, so a diagnostic
# arm binds it on the runner it measures instead of writing a process-wide
# switch.
_Q51_REGISTER_CACHE_UNSET = object()


@contextlib.contextmanager
def q51_register_cache(
    runner: Any, enabled: bool | object = _Q51_REGISTER_CACHE_UNSET,
):
    """Bind the Q5_1 register-cache selection for a diagnostic arm.

    ``True`` selects the register-cache sibling of the folded-pair owner inside
    its rows>=512 K640 scope and ``False`` keeps that scalar parent; leaving
    ``enabled`` unset keeps whatever the plan published. The binding is a
    mutable runner field, so it stays in force for as long as the scope is open
    and the published value is restored on exit, including on an exception.
    """

    prior = getattr(runner, "q51_register_cache", False)
    if enabled is not _Q51_REGISTER_CACHE_UNSET:
        runner.q51_register_cache = bool(enabled)
    try:
        yield prior
    finally:
        runner.q51_register_cache = prior


_Q51_ROW_PUBLISH_UNSET = object()


@contextlib.contextmanager
def q51_row_publish(runner: Any, enabled: bool | object = _Q51_ROW_PUBLISH_UNSET):
    """Bind the Q5_1 row-publish selection for a diagnostic arm.

    ``True`` selects the row-publish sibling inside its rows>=512 K640 scope and
    ``False`` keeps the register-cache parent; leaving ``enabled`` unset keeps
    whatever the plan published. The binding is a mutable runner field, so it
    stays in force for as long as the scope is open and the published value is
    restored on exit, including on an exception.
    """

    prior = getattr(runner, "q51_row_publish", False)
    if enabled is not _Q51_ROW_PUBLISH_UNSET:
        runner.q51_row_publish = bool(enabled)
    try:
        yield prior
    finally:
        runner.q51_row_publish = prior


# Q8_0 down register selection. The production plan declares the register GEMV
# down owner for the rows>=512 K640 Q8_0 expert-down scope; the binder publishes
# that decision to the owning runner, and the dispatch-time selector keeps its
# rows/K640/bundle-parent/registration guards. One decision serves both MoE down
# paths, which resolve the same bundle key. It is request-local state, so a
# diagnostic arm binds it on the runner it measures instead of writing a
# process-wide switch.
_Q8_DOWN_REGISTER_UNSET = object()


@contextlib.contextmanager
def q8_down_register(runner: Any, enabled: bool | object = _Q8_DOWN_REGISTER_UNSET):
    """Bind the Q8_0 down register selection for a diagnostic arm.

    ``True`` selects the register GEMV sibling of the row4 bundle owner inside
    its rows>=512 K640 scope and ``False`` keeps the bundle parent; leaving
    ``enabled`` unset keeps whatever the plan published. The sorted grouped
    chain and the token-major mapped path resolve the same bundle key, so one
    binding moves both. The binding is a mutable runner field, so it stays in
    force for as long as the scope is open and the published value is restored
    on exit, including on an exception.
    """

    prior = getattr(runner, "q8_down_register", False)
    if enabled is not _Q8_DOWN_REGISTER_UNSET:
        runner.q8_down_register = bool(enabled)
    try:
        yield prior
    finally:
        runner.q8_down_register = prior


# Q5_1 exact grouped down gate. Production runs the exact grouped Q5_1 down
# chain for every multi-row prefill chunk; the binder publishes that decision to
# the owning runner, and the dispatch-time gate keeps its rows>=2, quant and
# production-MoE-preemptor guards. It is the gate of the chain the Q5_1 ladder
# sits inside, so closing it falls through to the sorted/WMMA owners below and
# makes every rung of that ladder unreachable. It is request-local state, so a
# diagnostic arm binds it on the runner it measures instead of writing a
# process-wide switch.
_Q51_EXACT_GROUPED_DOWN_UNSET = object()


@contextlib.contextmanager
def q51_exact_grouped_down(
    runner: Any, enabled: bool | object = _Q51_EXACT_GROUPED_DOWN_UNSET,
):
    """Bind the Q5_1 exact grouped down chain gate for a diagnostic arm.

    ``False`` skips the exact grouped Q5_1 down chain for the measured window so
    the dispatch falls through to the sorted/WMMA owners below it, and ``True``
    restores the production chain; leaving ``enabled`` unset keeps whatever the
    plan published. The gate is the chain's own, so a closed gate also makes
    every Q5_1 ladder rung inside it unreachable and, with the Q4 exact gate's
    ``_ALL`` sibling cleared, takes the Q4_K exact gate/up predicate with it.
    The binding is a mutable runner field, so it stays in force for as long as
    the scope is open and the published value is restored on exit, including on
    an exception.
    """

    prior = getattr(runner, "q51_exact_grouped_down", True)
    if enabled is not _Q51_EXACT_GROUPED_DOWN_UNSET:
        runner.q51_exact_grouped_down = bool(enabled)
    try:
        yield prior
    finally:
        runner.q51_exact_grouped_down = prior


# Q5_1 F16 WMMA down owner admission. Production publishes it off: the owner is
# the down-only MX-1 diagnostic below the exact grouped Q5_1 down chain, and the
# production grouped MoE route admits it on the Q5K gate/up window by itself. It
# is request-local state, so a diagnostic arm binds it on the runner it measures
# instead of writing a process-wide switch.
_Q51_WMMA_DOWN_UNSET = object()


@contextlib.contextmanager
def q51_wmma_down(
    runner: Any, enabled: bool | object = _Q51_WMMA_DOWN_UNSET,
):
    """Bind the Q5_1 F16 WMMA down owner admission for a diagnostic arm.

    ``True`` admits the F16 WMMA compact grouped Q5_1 down owner on the chunks
    the production grouped MoE route does not own, and ``False`` keeps the
    production admission alone; leaving ``enabled`` unset keeps whatever the
    plan published. The field only decides below the exact grouped Q5_1 down
    chain and below the T16 owner, and the routed-F16 route still owns the whole
    down leg ahead of the Q5_1 branch, so an arm that names this owner closes
    those first. The binding is a mutable runner field, so it stays in force for
    as long as the scope is open and the published value is restored on exit,
    including on an exception.
    """

    prior = getattr(runner, "q51_wmma_down", False)
    if enabled is not _Q51_WMMA_DOWN_UNSET:
        runner.q51_wmma_down = bool(enabled)
    try:
        yield prior
    finally:
        runner.q51_wmma_down = prior


# Compact F16 Q4_K gate/up tile pair. The binder publishes 16/16 to the owning
# runner, and an unbound runner keeps the dispatch heuristic; a diagnostic arm
# binds a pair on the runner it measures instead of writing a process-wide
# switch. The pair only shapes the compact F16 fallback owner below the exact
# grouped Q4_K chain and the iu8 owner, never their arithmetic.
_Q4_TILE_UNSET = object()


@contextlib.contextmanager
def q4_tile_m(runner: Any, enabled: int | None | object = _Q4_TILE_UNSET):
    """Bind the compact F16 Q4_K gate/up ``tile_m`` for a diagnostic arm.

    An integer binds that tile height, and ``None`` restores the unbound
    heuristic (64 from 512 rows up, 16 below); leaving ``enabled`` unset keeps
    whatever the plan published. The binding is a mutable runner field, so it
    stays in force for as long as the scope is open and the published value is
    restored on exit, including on an exception.
    """

    prior = getattr(runner, "q4_tile_m", None)
    if enabled is not _Q4_TILE_UNSET:
        runner.q4_tile_m = enabled
    try:
        yield prior
    finally:
        runner.q4_tile_m = prior


@contextlib.contextmanager
def q4_tile_n(runner: Any, enabled: int | None | object = _Q4_TILE_UNSET):
    """Bind the compact F16 Q4_K gate/up ``tile_n`` for a diagnostic arm.

    Same protocol as :func:`q4_tile_m`; ``None`` restores the 16-wide default.
    """

    prior = getattr(runner, "q4_tile_n", None)
    if enabled is not _Q4_TILE_UNSET:
        runner.q4_tile_n = enabled
    try:
        yield prior
    finally:
        runner.q4_tile_n = prior


# Q8_0 dense WMMA prefill tile pair. The binder publishes the plan's pair on the
# owning runner and the runner binds it for each scope that can launch that
# family, so the value is execution-owned rather than process-wide; a diagnostic
# arm rebinds the pair on the runner it measures.
_Q8_WMMA_TILES_UNSET = object()


@contextlib.contextmanager
def q8_wmma_tiles(runner: Any, pair: object = _Q8_WMMA_TILES_UNSET):
    """Bind the Q8_0 dense WMMA prefill tile pair for a diagnostic arm.

    ``(tile_m, tile_n)`` binds that pair, ``None`` restores the unbound arch
    heuristic, and leaving ``pair`` unset keeps whatever the plan published. The
    pair is validated against the owner's allowed tiles when the scope is
    entered, so an unsupported or half-specified pair fails here rather than at
    the first launch. Restored on exit, including on an exception.
    """

    prior = getattr(runner, "q8_wmma_tiles", None)
    if pair is not _Q8_WMMA_TILES_UNSET:
        if pair is not None:
            pair = normalize_wmma_tile_pair(*pair)
        runner.q8_wmma_tiles = pair
    try:
        yield prior
    finally:
        runner.q8_wmma_tiles = prior


def _qwen4_exp_q8_tile_scope(owner: Any):
    """Bind the owner's Q8_0 tile pair for one execution entry.

    Every Qwen4Exp entry that can reach the Q8_0 dense WMMA prefill under an
    existing opt-in enters this, so the pair never depends on which entry the
    scheduler happened to call. An owner that published nothing binds
    ``(None, None)``: the pair is *ownership*, not an optional override, so a
    nested runner must mask whatever pair its caller had bound rather than
    inherit it.
    """

    pair = getattr(owner, "q8_wmma_tiles", None)
    return wmma_tile_policy(*(pair if pair is not None else (None, None)))


# Q4 exact grouped gate/up gate. Production runs the exact grouped Q4_K gate/up
# chain for every multi-row Q4_K/Q4_K chunk; the binder publishes that decision to
# the owning runner, and the dispatch-time gate keeps its rows>=2, quant,
# production-MoE-preemptor and joint `_ALL` guards. Closing it is the only way to
# reach the F16 compact and plain-iu8 owners below it, because the chain is tested
# before both. It is request-local state, so a diagnostic arm binds it on the
# runner it measures instead of writing a process-wide switch.
_Q4_EXACT_GROUPED_GATE_UNSET = object()


@contextlib.contextmanager
def q4_exact_grouped_gate(
    runner: Any, enabled: bool | object = _Q4_EXACT_GROUPED_GATE_UNSET,
):
    """Bind the Q4_K exact grouped gate/up chain gate for a diagnostic arm.

    ``False`` skips the exact grouped Q4_K gate/up chain for the measured window
    so the dispatch falls through to the plain-iu8 or F16 compact owner below it,
    and ``True`` restores the production chain; leaving ``enabled`` unset keeps
    whatever the plan published. The gate is the chain's own, so a closed gate
    also makes every rung of the Q4 ladder inside it unreachable - including the
    out4/expertgrid owners and the pair2/bundle preemptors - and it changes the
    grouped Q5_1 down chain's input layout, because that chain reads the combined
    gate/up buffer only while this gate is on. The binding is a mutable runner
    field, so it stays in force for as long as the scope is open and the
    published value is restored on exit, including on an exception.
    """

    prior = getattr(runner, "q4_exact_grouped_gate", True)
    if enabled is not _Q4_EXACT_GROUPED_GATE_UNSET:
        runner.q4_exact_grouped_gate = bool(enabled)
    try:
        yield prior
    finally:
        runner.q4_exact_grouped_gate = prior


# Q4 exact grouped `_ALL` joint term. The chain's predicate is
# `gate and (exact_grouped_down or all)`, so `_ALL` only matters where the exact
# grouped Q5_1 down chain is off: with it on, the Q4 chain runs on every Q4_K
# multi-row chunk; with it off, the Q4 chain is restricted to chunks the Q5_1
# down chain already owns. Production keeps it on. It is request-local state, so
# a diagnostic arm binds it on the runner it measures.
_Q4_EXACT_GROUPED_ALL_UNSET = object()


@contextlib.contextmanager
def q4_exact_grouped_all(
    runner: Any, enabled: bool | object = _Q4_EXACT_GROUPED_ALL_UNSET,
):
    """Bind the Q4_K exact grouped chain's `_ALL` joint term for an arm.

    ``False`` restricts the Q4 chain to chunks where the exact grouped Q5_1 down
    chain also runs, and ``True`` extends it to every Q4_K multi-row chunk;
    leaving ``enabled`` unset keeps whatever the plan published. It is a joint
    term, not a gate: it can only narrow a chain the gate already opened, and
    with the gate off or the down chain on it changes nothing. The binding is a
    mutable runner field, so it stays in force for as long as the scope is open
    and the published value is restored on exit, including on an exception.
    """

    prior = getattr(runner, "q4_exact_grouped_all", True)
    if enabled is not _Q4_EXACT_GROUPED_ALL_UNSET:
        runner.q4_exact_grouped_all = bool(enabled)
    try:
        yield prior
    finally:
        runner.q4_exact_grouped_all = prior


# Q5_1 out8 fallback selection. It is one rung of the exact grouped Q5_1 down
# ladder: the fallback below the exact expert-grid family, choosing the out8
# rowbatch8 owner over the output1 one. The expert-grid family, the exact
# iu8-risk+repair chain above it, the pair2 preemptor and the fold siblings below
# it are separate decisions; this one only moves that rung. It is request-local
# state, so a diagnostic arm binds it on the runner it measures instead of
# writing a process-wide switch.
_Q51_OUT8_UNSET = object()


@contextlib.contextmanager
def q51_out8(runner: Any, enabled: bool | object = _Q51_OUT8_UNSET):
    """Bind the Q5_1 out8 fallback rung for a diagnostic arm.

    ``False`` pins the output1 rowbatch8 base for the measured window and
    ``True`` restores the out8 owner; leaving ``enabled`` unset keeps whatever
    the plan published. The rung only decides the non-expertgrid fallback, so an
    arm that also needs the output1 leaf to launch has to move the exact expert
    grid family (and the exact iu8-risk+repair chain above it) out of the way.
    The binding is a mutable runner field, so it stays in force for as long as
    the scope is open and the published value is restored on exit, including on
    an exception.
    """

    prior = getattr(runner, "q51_out8", True)
    if enabled is not _Q51_OUT8_UNSET:
        runner.q51_out8 = bool(enabled)
    try:
        yield prior
    finally:
        runner.q51_out8 = prior


# Q4 out4 fallback selection. It is one rung of the exact grouped Q4_K gate/up
# ladder: the fallback below the exact expert-grid family, choosing the out4
# rowbatch8 owner over the strict rowbatch8 base. The routed-F16 owner, the
# chain's own switches, the exact iu8-risk+repair chain and the expert-grid
# family with its pair2/bundle preemptors are separate decisions; this one only
# moves that rung. It is request-local state, so a diagnostic arm binds it on the
# runner it measures instead of writing a process-wide switch.
_Q4_OUT4_UNSET = object()


@contextlib.contextmanager
def q4_out4(runner: Any, enabled: bool | object = _Q4_OUT4_UNSET):
    """Bind the Q4 out4 fallback rung for a diagnostic arm.

    ``False`` pins the strict rowbatch8 base for the measured window and ``True``
    restores the out4 owner; leaving ``enabled`` unset keeps whatever the plan
    published. The rung only decides the non-expertgrid fallback, so an arm that
    also needs the strict base to launch has to move the exact expert grid family
    and the exact iu8-risk+repair chain above it out of the way. The binding is a
    mutable runner field, so it stays in force for as long as the scope is open
    and the published value is restored on exit, including on an exception.
    """

    prior = getattr(runner, "q4_out4", True)
    if enabled is not _Q4_OUT4_UNSET:
        runner.q4_out4 = bool(enabled)
    try:
        yield prior
    finally:
        runner.q4_out4 = prior


# Q5_1 down M1 selection. The production plan declares the M1 owner for the
# exact grouped Q5_1 down chain's small-chunk scope; the binder publishes that
# decision to the owning runner, and the dispatch-time selector keeps its exact
# expert-grid gate and the pair2/fold ladder that preempts it at rows>=64. It is
# the parent rung of that ladder, so every fold plan reads the variant this
# decision selects. It is request-local state, so a diagnostic arm binds it on
# the runner it measures instead of writing a process-wide switch.
_Q51_DOWN_M1_UNSET = object()


@contextlib.contextmanager
def q51_down_m1(
    runner: Any, enabled: bool | object = _Q51_DOWN_M1_UNSET,
):
    """Bind the Q5_1 down M1 selection for a diagnostic arm.

    ``True`` selects the M1 sibling of the expertgrid64 exact grouped Q5_1 down
    owner inside its small-chunk scope and ``False`` keeps the preceding
    expertgrid64 owner; leaving ``enabled`` unset keeps whatever the plan
    published. The pair2/fold ladder above it still preempts it at rows>=64 and
    the exact expert-grid gate still decides whether that family is selected at
    all. The binding is a mutable runner field, so it stays in force for as long
    as the scope is open and the published value is restored on exit, including
    on an exception.
    """

    prior = getattr(runner, "q51_down_m1", False)
    if enabled is not _Q51_DOWN_M1_UNSET:
        runner.q51_down_m1 = bool(enabled)
    try:
        yield prior
    finally:
        runner.q51_down_m1 = prior


# Q5_1 fold128 selection. The production plan declares the fold128 owner for
# the grouped pair2 chain's rows>=64 scope; the binder publishes that decision
# to the owning runner, and the dispatch-time selector keeps its rows/
# pair2-parent/registration guards. It is the parent of the folded-pair,
# register-cache and row-publish siblings, so every one of those plans reads the
# variant this decision selects. It is request-local state, so a diagnostic arm
# binds it on the runner it measures instead of writing a process-wide switch.
_Q51_FOLD128_PREFILL_UNSET = object()


@contextlib.contextmanager
def q51_fold128_prefill(
    runner: Any, enabled: bool | object = _Q51_FOLD128_PREFILL_UNSET,
):
    """Bind the Q5_1 fold128 selection for a diagnostic arm.

    ``True`` selects the fold128 sibling of the grouped pair2 prefill owner
    inside its rows>=64 scope and ``False`` keeps that pair2 parent; leaving
    ``enabled`` unset keeps whatever the plan published. The binding is a
    mutable runner field, so it stays in force for as long as the scope is open
    and the published value is restored on exit, including on an exception.
    """

    prior = getattr(runner, "q51_fold128_prefill", False)
    if enabled is not _Q51_FOLD128_PREFILL_UNSET:
        runner.q51_fold128_prefill = bool(enabled)
    try:
        yield prior
    finally:
        runner.q51_fold128_prefill = prior


# Q5_1 folded-pair selection. The production plan declares the folded-pair
# owner for the fold128 ladder's rows>=512 scope; the binder publishes that
# decision to the owning runner, and the dispatch-time selector keeps its rows/
# fold128-parent/registration guards. It is request-local state, so a
# diagnostic arm binds it on the runner it measures instead of writing a
# process-wide switch.
_Q51_FOLD_PAIR_PREFILL_UNSET = object()


@contextlib.contextmanager
def q51_fold_pair_prefill(
    runner: Any, enabled: bool | object = _Q51_FOLD_PAIR_PREFILL_UNSET,
):
    """Bind the Q5_1 folded-pair selection for a diagnostic arm.

    ``True`` selects the folded-pair sibling of the fold128 owner inside its
    rows>=512 scope and ``False`` keeps the fold128 parent; leaving ``enabled``
    unset keeps whatever the plan published. The binding is a mutable runner
    field, so it stays in force for as long as the scope is open and the
    published value is restored on exit, including on an exception.
    """

    prior = getattr(runner, "q51_fold_pair_prefill", False)
    if enabled is not _Q51_FOLD_PAIR_PREFILL_UNSET:
        runner.q51_fold_pair_prefill = bool(enabled)
    try:
        yield prior
    finally:
        runner.q51_fold_pair_prefill = prior


# PF-1 fork-b grouped Q8_0 down selection. The production plan declares the
# grouped selected-gemv owner for the sorted chain's Q8_0 down scope; the binder
# publishes that decision to the owning runner, and the dispatch-time selector
# keeps its position in the chain - the Q8_0 WMMA down owner, when a plan binds
# one, preempts it outright. It is request-local state, so a diagnostic arm
# binds it on the runner it measures instead of writing a process-wide switch.
_FORKB_GROUPED_DOWN_UNSET = object()


@contextlib.contextmanager
def forkb_grouped_down(
    runner: Any, enabled: bool | object = _FORKB_GROUPED_DOWN_UNSET,
):
    """Bind the PF-1 fork-b grouped Q8_0 down selection for a diagnostic arm.

    ``True`` routes the sorted chain's Q8_0 down through the grouped selected
    gemv (one block per (expert, out_col) pair) and ``False`` keeps the strict
    per-expert selected gemv; leaving ``enabled`` unset keeps whatever the plan
    published. It only selects when no Q8_0 WMMA down owner is bound and the
    grouped chain is active, so a plan that binds that owner leaves it inert.
    The binding is a mutable runner field, so it stays in force for as long as
    the scope is open and the published value is restored on exit, including on
    an exception.
    """

    prior = getattr(runner, "forkb_grouped_down", False)
    if enabled is not _FORKB_GROUPED_DOWN_UNSET:
        runner.forkb_grouped_down = bool(enabled)
    try:
        yield prior
    finally:
        runner.forkb_grouped_down = prior


# Q8_0 down row4 selection. The production plan declares the row4 GEMV owner
# for the sorted grouped chain's rows>=512 scope; the binder publishes that
# decision to the owning runner, and the dispatch-time selector keeps its rows/
# parent-variant/registration guards. It is the parent of the bundle and
# register siblings, which read the variant it selects. It is request-local
# state, so a diagnostic arm binds it on the runner it measures instead of
# writing a process-wide switch.
_Q8_DOWN_ROW4_PREFILL_UNSET = object()


@contextlib.contextmanager
def q8_down_row4_prefill(
    runner: Any, enabled: bool | object = _Q8_DOWN_ROW4_PREFILL_UNSET,
):
    """Bind the Q8_0 down row4 selection for a diagnostic arm.

    ``True`` selects the row4 GEMV sibling of the sorted grouped chain inside
    its rows>=512 scope and ``False`` keeps that parent; leaving ``enabled``
    unset keeps whatever the plan published. The binding is a mutable runner
    field, so it stays in force for as long as the scope is open and the
    published value is restored on exit, including on an exception.
    """

    prior = getattr(runner, "q8_down_row4_prefill", False)
    if enabled is not _Q8_DOWN_ROW4_PREFILL_UNSET:
        runner.q8_down_row4_prefill = bool(enabled)
    try:
        yield prior
    finally:
        runner.q8_down_row4_prefill = prior


# Q8_0 down bundle selection. The production plan declares the bundle GEMV
# owner for the row4 ladder's rows>=512 scope; the binder publishes that
# decision to the owning runner, and the dispatch-time selector keeps its rows/
# row4-parent/registration guards. The register sibling downstream replaces it
# in turn. It is request-local state, so a diagnostic arm binds it on the
# runner it measures instead of writing a process-wide switch.
_Q8_DOWN_BUNDLE_PREFILL_UNSET = object()


@contextlib.contextmanager
def q8_down_bundle_prefill(
    runner: Any, enabled: bool | object = _Q8_DOWN_BUNDLE_PREFILL_UNSET,
):
    """Bind the Q8_0 down bundle selection for a diagnostic arm.

    ``True`` selects the bundle GEMV sibling of the row4 owner inside its
    rows>=512 scope and ``False`` keeps the row4 parent; leaving ``enabled``
    unset keeps whatever the plan published. The binding is a mutable runner
    field, so it stays in force for as long as the scope is open and the
    published value is restored on exit, including on an exception.
    """

    prior = getattr(runner, "q8_down_bundle_prefill", False)
    if enabled is not _Q8_DOWN_BUNDLE_PREFILL_UNSET:
        runner.q8_down_bundle_prefill = bool(enabled)
    try:
        yield prior
    finally:
        runner.q8_down_bundle_prefill = prior


# The token-major mapped Q8_0 expert-down owner is the same kind of
# request-local decision: the plan publishes whether the bundle leaf may serve
# the rows>=512 scope, the dispatch-time selector keeps the current call's map,
# the row count and the registration guards, and a diagnostic arm binds the
# field on the runner it measures instead of writing a process-wide switch.
_Q8_MAPPED_DOWN_UNSET = object()


@contextlib.contextmanager
def q8_mapped_down(runner: Any, enabled: bool | object = _Q8_MAPPED_DOWN_UNSET):
    """Bind the Q8_0 token-major mapped down owner for a diagnostic arm.

    ``True`` admits the row4 bundle leaf through the token-major mapped path
    when this call built the Q5_K row4 map and ``rows>=512``; ``False`` keeps
    the token-major parent, and leaving ``enabled`` unset keeps whatever the
    plan published. The binding is a mutable runner field, so it stays in force
    for as long as the scope is open and the published value is restored on
    exit, including on an exception.
    """

    prior = getattr(runner, "q8_mapped_down", False)
    if enabled is not _Q8_MAPPED_DOWN_UNSET:
        runner.q8_mapped_down = bool(enabled)
    try:
        yield prior
    finally:
        runner.q8_mapped_down = prior


def _qwen4_exp_gdn_prefill_key(
    key: KernelKey, *, rows: int = 0, num_k_heads: int = 0,
    num_v_heads: int = 0, head_dim: int = 0, variant: str | None = None,
) -> KernelKey:
    if key.variant != GDN_PREFILL_PARENT_VARIANT:
        return key
    variant = str(variant or "")
    if variant not in (GDN_PREFILL_WIDE_VARIANT, GDN_PREFILL_NARROW_VARIANT):
        return key
    if (variant == GDN_PREFILL_NARROW_VARIANT and 2 <= rows <= 4096
            and (num_k_heads, num_v_heads, head_dim) == (16, 48, 128)):
        narrow = KernelKey(key.backend, key.layer, key.quant,
                           GDN_PREFILL_NARROW_VARIANT)
        if is_registered(narrow):
            return narrow
    wide = KernelKey(key.backend, key.layer, key.quant, GDN_PREFILL_WIDE_VARIANT)
    return wide if is_registered(wide) else key


def _qwen4_exp_q51_row_publish_key(
    key: KernelKey, *, rows: int, in_features: int, published: bool = False,
) -> KernelKey:
    """Select the Q5_1 row-publish sibling of the register-cache prefill owner.

    ``published`` is the owning runner's plan-published decision (see
    :func:`q51_row_publish`), not a process-wide switch: the production plan
    declares the row-publish owner for the rows>=512 K640 scope, and every plan
    that keeps the register-cache parent publishes ``False``. The guards stay
    dispatch-time - rows, in_features, the register-cache parent variant, and a
    registry that must actually carry the leaf - so an unbound runner or a
    missing registration keeps the parent.
    """

    if (rows < 512 or in_features != 640
            or key.variant != "selected_grouped_prefill_pair2_register_cache_bf16_bf16_out"
            or not published):
        return key
    candidate = KernelKey(key.backend, key.layer, key.quant,
                         "selected_grouped_prefill_pair2_row_publish_bf16_bf16_out")
    return candidate if is_registered(candidate) else key


def _qwen4_exp_q51_down_key(
    *,
    backend: str,
    quant: str,
    rows: int,
    in_features: int,
    q51_down_m1: bool = False,
    q51_out8: bool = True,
    q51_pair_prefill: bool = False,
    q51_fold128_prefill: bool = False,
    q51_fold_pair_prefill: bool = False,
    q51_register_cache: bool = False,
    q51_row_publish: bool = False,
) -> KernelKey:
    """The exact grouped Q5_1 down owner, before its own resolve.

    This is the ladder the chain's guards select, in dispatch order: the exact
    expert-grid family (``EXACT_EXPERT_GRID``), the out8 fallback rung
    (``q51_out8``, see :func:`q51_out8`), the output1 base, then the pair2
    preemptor, then the fold128/folded-pair/register-cache/row-publish siblings.
    The caller owns everything above it - ``production_grouped_moe``, the routed
    F16 and T16 owners, the chain gate and the exact iu8-risk+repair switch - so
    this function is only reached by a chunk that already runs the chain.

    Every input is either a dispatch-time guard (rows, in_features, the
    registry) or the owning runner's published decision, so the same call on the
    same runner and chunk always returns the same owner. The fold siblings only
    replace the pair2 parent variant, so they are inert while the pair2
    preemptor keeps the parent - but they are read here rather than assumed.
    """

    if os.environ.get(
        "HIPENGINE_QWEN4_EXP_EXACT_EXPERT_GRID", "64"
    ) in {"64", "q5"}:
        variant = (
            "selected_grouped_prefill_compact_rowbatch8_out8_"
            "expertgrid64_m1_bf16_bf16_out"
            if q51_down_m1
            else "selected_grouped_prefill_compact_rowbatch8_out8_"
            "expertgrid64_bf16_bf16_out"
        )
    elif q51_out8:
        variant = "selected_grouped_prefill_compact_rowbatch8_out8_bf16_bf16_out"
    else:
        variant = "selected_grouped_prefill_compact_rowbatch8_bf16_bf16_out"
    if qwen4_exp_q51_pair_prefill_selected(
        backend, quant, rows=rows, in_features=in_features,
        published=q51_pair_prefill,
    ):
        variant = "selected_grouped_prefill_pair2_bf16_bf16_out"
    key = _qwen4_exp_q51_fold128_key(
        KernelKey(backend, "moe_linear", quant, variant), rows=rows,
        published=q51_fold128_prefill)
    key = _qwen4_exp_q51_fold_pair_key(
        key, rows=rows, published=q51_fold_pair_prefill)
    key = _qwen4_exp_q51_register_cache_key(
        key, rows=rows, in_features=in_features,
        published=q51_register_cache)
    return _qwen4_exp_q51_row_publish_key(
        key, rows=rows, in_features=in_features,
        published=q51_row_publish)


def _qwen4_exp_q51_fold_pair_key(
    key: KernelKey, *, rows: int, published: bool = False,
) -> KernelKey:
    """Select the Q5_1 folded-pair sibling of the fold128 prefill owner.

    ``published`` is the owning runner's plan-published decision (see
    :func:`q51_fold_pair_prefill`), not a process-wide switch: the production
    plan declares the folded-pair owner for the rows>=512 scope, and every plan
    that keeps the fold128 parent publishes ``False``. The guards stay
    dispatch-time - rows, the fold128 parent variant, and a registry that must
    actually carry the leaf - so an unbound runner or a missing registration
    keeps the parent.
    """

    if (rows < 512 or key.variant != "selected_grouped_prefill_pair2_fold128_bf16_bf16_out"
            or not published):
        return key
    candidate = KernelKey(key.backend, key.layer, key.quant,
                          "selected_grouped_prefill_pair2_fold128_pair_bf16_bf16_out")
    return candidate if is_registered(candidate) else key


def _qwen4_exp_q51_fold128_key(
    key: KernelKey, *, rows: int, published: bool = False,
) -> KernelKey:
    """Select the Q5_1 fold128 sibling of the grouped pair2 prefill owner.

    ``published`` is the owning runner's plan-published decision (see
    :func:`q51_fold128_prefill`), not a process-wide switch: the production
    plan declares the fold128 owner for the rows>=64 scope, and every plan that
    keeps the pair2 parent publishes ``False``. The guards stay dispatch-time -
    rows, the pair2 parent variant, and a registry that must actually carry the
    leaf - so an unbound runner or a missing registration keeps the parent. The
    downstream folded-pair and register-cache/row-publish siblings are separate
    decisions that read this leaf's variant.
    """

    if (rows < 64 or key.variant != "selected_grouped_prefill_pair2_bf16_bf16_out"
            or not published):
        return key
    candidate = KernelKey(key.backend, key.layer, key.quant,
                          "selected_grouped_prefill_pair2_fold128_bf16_bf16_out")
    return candidate if is_registered(candidate) else key


def _qwen4_exp_mapped_down_key(*, backend: str, quant: str, rows: int,
                              map_ready: bool, published: bool = False) -> KernelKey | None:
    """Select the token-major mapped Q8_0 expert-down owner for this call.

    ``published`` is the owning runner's plan-published decision (see
    :func:`q8_mapped_down`), not a process-wide switch: the production plan
    declares the mapped owner for the rows>=512 token-major scope, and every
    plan that keeps the token-major parent publishes ``False``. ``map_ready``
    is this invocation's own ownership proof - the Q5_K row4 map it just built -
    so allocated scratch never admits the key. The guards stay dispatch-time -
    rows, the current call's map, and a registry that must actually carry the
    leaf - so an unbound runner or a missing registration keeps the parent.
    """

    if not published or not map_ready or rows < 512:
        return None
    key = KernelKey(backend,"linear",quant,"selected_grouped_row4_bundle_gemv_bf16_bf16_out")
    return key if is_registered(key) else None


def _qwen4_exp_q8_down_register_key(
    key: KernelKey, *, rows: int, in_features: int, published: bool = False,
) -> KernelKey:
    """Select the register GEMV sibling of the row4 bundle Q8_0 down owner.

    ``published`` is the owning runner's plan-published decision (see
    :func:`q8_down_register`), not a process-wide switch: the production plan
    declares the register owner for the rows>=512 K640 scope, and every plan
    that keeps the bundle parent publishes ``False``. Both MoE down paths
    resolve the same bundle key - the sorted grouped chain through
    :func:`_qwen4_exp_q8_down_bundle_key`, the token-major mapped path through
    :func:`_qwen4_exp_mapped_down_key` - so one published value serves both.
    The guards stay dispatch-time - rows, in_features, the bundle parent
    variant, and a registry that must actually carry the leaf - so an unbound
    runner or a missing registration keeps the parent.
    """

    if (rows < 512 or in_features != 640
            or key.variant != "selected_grouped_row4_bundle_gemv_bf16_bf16_out"
            or not published):
        return key
    candidate = KernelKey(key.backend,key.layer,key.quant,
                          "selected_grouped_row4_register_gemv_bf16_bf16_out")
    return candidate if is_registered(candidate) else key


def _qwen4_exp_q8_down_bundle_key(
    key: KernelKey, *, rows: int, published: bool = False,
) -> KernelKey:
    """Select the bundle GEMV sibling of the row4 Q8_0 down owner.

    ``published`` is the owning runner's plan-published decision (see
    :func:`q8_down_bundle_prefill`), not a process-wide switch: the production
    plan declares the bundle owner for the rows>=512 scope, and every plan that
    keeps the row4 parent publishes ``False``. The guards stay dispatch-time -
    rows, the row4 parent variant, and a registry that must actually carry the
    leaf - so an unbound runner or a missing registration keeps the parent. The
    downstream register decision is separate and reads this leaf's variant.
    """

    if (rows < 512 or key.variant != "selected_grouped_row4_gemv_bf16_bf16_out"
            or not published):
        return key
    candidate = KernelKey(key.backend, key.layer, key.quant,
                          "selected_grouped_row4_bundle_gemv_bf16_bf16_out")
    return candidate if is_registered(candidate) else key


def _qwen4_exp_q8_down_row4_key(
    key: KernelKey, *, rows: int, published: bool = False,
) -> KernelKey:
    """Select the row4 GEMV sibling of the sorted grouped Q8_0 down owner.

    ``published`` is the owning runner's plan-published decision (see
    :func:`q8_down_row4_prefill`), not a process-wide switch: the production
    plan declares the row4 owner for the rows>=512 sorted-chain scope, and
    every plan that keeps the parent publishes ``False``. The guards stay
    dispatch-time - rows, the parent variant, and a registry that must actually
    carry the leaf - so an unbound runner or a missing registration keeps the
    parent. The downstream bundle and register decisions are separate and read
    this leaf's variant.
    """

    if (rows < 512 or key.variant != "selected_grouped_gemv_bf16_bf16_out"
            or not published):
        return key
    candidate = KernelKey(key.backend, key.layer, key.quant,
                          "selected_grouped_row4_gemv_bf16_bf16_out")
    return candidate if is_registered(candidate) else key


def _qwen4_exp_gr_wave_scale_key(key: KernelKey, *, rows: int, branches: int) -> KernelKey:
    if rows <= 256 or branches != 4:
        return key
    wave_scale_enabled = os.environ.get(
        "HIPENGINE_QWEN4_EXP_GR_WAVE_SCALE", "0"
    ) not in {"", "0", "false", "False"}
    dpp_enabled = os.environ.get(
        "HIPENGINE_QWEN4_EXP_GR_DPP_REDUCTION", "0"
    ) not in {"", "0", "false", "False"}
    if not wave_scale_enabled and not dpp_enabled:
        return key
    if dpp_enabled:
        dpp = KernelKey(key.backend, key.layer, key.quant,
                        "coltile2_branch4_rowbatch4_wave_scale_dpp_f32_exact")
        if is_registered(dpp):
            return dpp
    # DPP is an exact replacement for the wave-scale owner. Keep that owner as
    # the first fallback even when a recovery profile has closed its legacy
    # gate; otherwise a missing DPP registration would silently lose the
    # profile's exact wave-scale fallback.
    candidate = KernelKey(key.backend, key.layer, key.quant,
                          "coltile2_branch4_rowbatch4_wave_scale_f32_exact")
    return candidate if is_registered(candidate) else key




def _qwen4_exp_gr_k320_full_rows_key(
    key: KernelKey, *, rows: int, branches: int, low_rank: int, hidden: int = 2560
) -> KernelKey:
    """Select the registered K320 specialization of the exact DPP owner.

    The candidate is narrower than the promoted DPP owner: it is valid only for
    K=320, H=2560, four branches, and complete ROW_BATCH=4 groups. All other
    shapes keep the already-selected DPP owner or its parent fallback.
    """
    if (
        key.variant != "coltile2_branch4_rowbatch4_wave_scale_dpp_f32_exact"
        or rows <= 256
        or rows % 4 != 0
        or branches != 4
        or low_rank != 320
        or hidden != 2560
    ):
        return key
    candidate = KernelKey(
        key.backend, key.layer, key.quant,
        "coltile2_branch4_rowbatch4_wave_scale_dpp_k320_full_rows_f32_exact",
    )
    return candidate if is_registered(candidate) else key


def _qwen4_exp_gr_iu8_down_enabled(rows: int) -> bool:
    """Experimental iu8-WMMA GR down route, disabled by production recovery.

    The three-plane fp32 projection failed its standalone depth gate against
    strict. A new projection correction and full requalification are required;
    see docs/REFACTOR.md, "Isolated GR-Down IU8".
    """

    return (
        os.environ.get("HIPENGINE_QWEN4_EXP_GR_IU8_DOWN", "0")
        not in {"", "0", "false", "False"}
        and rows > 256
    )


def _qwen4_exp_gr_wide_enabled(rows: int, low_rank: int) -> bool:
    """Wide-tile GR up route selected by the production F16 composition.

    The profile binder enables this after arithmetic recovery for UD production
    and fast profiles. Its composition gate is distinct from the historical
    standalone rejection; the prefill-boundary KL screen is not an admission
    gate. See docs/REFACTOR.md, "GR-Up Wide Tile".
    """
    return (
        rows > 256
        and low_rank % 64 == 0
        and os.environ.get("HIPENGINE_QWEN4_EXP_GR_WIDE", "0")
        not in {"", "0", "false", "False"}
    )


def _qwen4_exp_gr_f16_direct_enabled(rows: int, low_rank: int) -> bool:
    """Direct F16 launches for the hyper-connection read projections.

    Bit-identical to the routes it replaces wherever those already run the F16
    wide tile. Set ``HIPENGINE_QWEN4_EXP_GR_F16_DIRECT=0`` to return to the
    ordinary dispatch for bisection; see docs/REFACTOR.md.
    """
    return (
        rows > 256
        and low_rank % 64 == 0
        and os.environ.get("HIPENGINE_QWEN4_EXP_GR_F16_DIRECT", "0")
        not in {"", "0", "false", "False"}
    )


def _qwen4_exp_moe_decode_warp_enabled(leg: str) -> bool:
    """Warp-per-expert decode owners for the routed experts (``GATE_UP``, ``DOWN``).

    Selected by the composition's binder through
    ``HIPENGINE_QWEN4_EXP_MOE_DECODE_WARP_<leg>=1``; see docs/REFACTOR.md.
    """
    return os.environ.get(
        f"HIPENGINE_QWEN4_EXP_MOE_DECODE_WARP_{leg}", "0"
    ) not in {"", "0", "false", "False"}


def _qwen4_exp_gr_decode_fused_enabled(
    rows: int, branches: int, hidden: int, low_rank: int,
) -> bool:
    """Three-launch hyper-connection read at one row.

    Bit-identical to the six-launch chain at rows == 1, where that chain runs
    the 256- and 128-thread pack8 GEMVs. ``HIPENGINE_QWEN4_EXP_GR_DECODE_FUSED=0``
    restores the chain; see docs/REFACTOR.md.
    """
    return (
        rows == 1
        and branches == 4
        and hidden % 2 == 0
        and (4 * hidden) % 32 == 0
        and low_rank > 0
        and low_rank % 32 == 0
        and os.environ.get("HIPENGINE_QWEN4_EXP_GR_DECODE_FUSED", "1")
        not in {"", "0", "false", "False"}
    )


def _qwen4_exp_gr_f16_mirror_enabled() -> bool:
    """Whether the scale-only read also publishes half(mixed).

    ``HIPENGINE_QWEN4_EXP_GR_F16_MIRROR=0`` keeps the per-projection
    conversions for bisection; see docs/REFACTOR.md.
    """
    return os.environ.get("HIPENGINE_QWEN4_EXP_GR_F16_MIRROR", "1") not in {
        "", "0", "false", "False",
    }


def _qwen4_exp_gr_up_sigmoid_mean_enabled(rows: int) -> bool:
    return rows > 256


def _qwen4_exp_gr_iu8_enabled(rows: int) -> bool:
    """Experimental iu8-WMMA GR up route, disabled by production recovery.

    Three-plane fp32 staging preserves the raw Q8_0 weight path but failed the
    standalone depth gate against strict. Keep the exact fused parent as the
    fallback pending a new correction and requalification; see docs/REFACTOR.md,
    "Isolated GR-Up IU8".
    """

    return (
        os.environ.get("HIPENGINE_QWEN4_EXP_GR_IU8", "0")
        not in {"", "0", "false", "False"}
        and rows > 256
    )


def _qwen4_exp_gr_sigmoid_mean_fused(rows: int) -> bool:
    """Select the exact launch-contracted GR epilogue in its measured row range."""

    return rows <= 256


def _qwen4_exp_q8_mmq_policy(policy):
    """Add the P3 attention-gate shape only for an explicit candidate run.

    Also applies the #28 R12 plane-count candidate env per prefill (the
    base policy resolves once at construction; the session re-applies
    route candidates per chunk so an A/B can switch arms in residency).
    """

    if policy is None:
        return None
    if os.environ.get(
        "HIPENGINE_QWEN4_EXP_Q8_MMQ_ATTN_GATE", "0"
    ) in {"", "0", "false", "False"}:
        pass
    else:
        minimums = dict(policy.min_rows)
        minimums[(2560, 6144)] = 64
        policy = replace(policy, min_rows=minimums)
    planes_env = os.environ.get("HIPENGINE_QWEN4_EXP_Q8_MMQ_PLANES", "")
    if planes_env in {"2", "3"} and int(planes_env) != policy.f32_planes:
        policy = replace(policy, f32_planes=int(planes_env))
    return policy


_ROUTER_PRODUCER_VARIANT_ENV = "HIPENGINE_QWEN4_EXP_ROUTER_VARIANT"
_ROUTER_PRODUCER_TILE4_VARIANT = "f32_hidden_token_tile4_dense_exact"
_ROUTER_PRODUCER_WARP_REDUCE_VARIANT = "f32_hidden_warp_reduce_dense_exact"
_ROUTER_PRODUCER_DPP_VARIANT = "f32_hidden_token_tile4_dpp_exact"
# F16-operand WMMA GEMM producer (UD production plan): operands rounded to
# F16, so near-tied expert selections can differ from the FP32 producers.
_ROUTER_PRODUCER_F16_GEMM_VARIANT = "f32_hidden_f16_wmma_gemm"
_ROUTER_PRODUCER_VARIANTS = (
    _ROUTER_PRODUCER_TILE4_VARIANT,
    _ROUTER_PRODUCER_WARP_REDUCE_VARIANT,
    _ROUTER_PRODUCER_DPP_VARIANT,
    _ROUTER_PRODUCER_F16_GEMM_VARIANT,
)


_ROUTER_F16_GEMM_MIN_ROWS = 64


def _qwen4_exp_router_producer_variant(rows: int) -> str | None:
    """Return the multirow F32 router producer variant, or None for the linear path.

    The named profile binder publishes the resolved producer: the UD
    production plan selects the F16-operand WMMA GEMM, ordered DPP is the
    arithmetic-recovery default and the named rollback, and strict and Q4_K_M
    keep tile4. The environment reader remains for the named rollback and the
    gate arms; see docs/REFACTOR.md. Unknown values raise rather than silently
    running an inert measurement arm.
    """

    if rows < 2:
        return None
    requested = os.environ.get(_ROUTER_PRODUCER_VARIANT_ENV, "").strip()
    if not requested:
        return _ROUTER_PRODUCER_TILE4_VARIANT
    if requested not in _ROUTER_PRODUCER_VARIANTS:
        raise ValueError(
            f"{_ROUTER_PRODUCER_VARIANT_ENV} must be one of "
            + ", ".join(_ROUTER_PRODUCER_VARIANTS)
            + f"; got {requested!r}"
        )
    if (
        requested == _ROUTER_PRODUCER_F16_GEMM_VARIANT
        and rows < _ROUTER_F16_GEMM_MIN_ROWS
    ):
        # The F16 WMMA GEMM stages 128-row tiles; at verifier and
        # small-batch widths its tiles are almost empty (4 rows: 184 us per
        # layer against 26 us for one row on the DPP producer). The exact
        # DPP producer serves those widths.
        return _ROUTER_PRODUCER_DPP_VARIANT
    return requested


def _qwen4_exp_production_moe_prefill_enabled(
    weight: GGUFDeviceWeight, *, rows: int
) -> bool:
    if rows < 16 or os.environ.get(
        "HIPENGINE_QWEN4_EXP_PRODUCTION_MOE_PREFILL", "0"
    ) in {"", "0", "false", "False"}:
        return False
    parts = weight.spec.slot_path.split(".")
    if len(parts) < 3 or parts[0] != "layers":
        return False
    admitted = backend_package_capability(
        str(weight.backend),
        "QWEN4_EXP_PRODUCTION_MOE_PREFILL_LAYERS",
        (),
    )
    return int(parts[1]) in admitted


def _qwen4_exp_q51_wmma_down_enabled(
    production_grouped_moe: bool, published: bool,
) -> bool:
    """Whether the F16 WMMA compact grouped Q5_1 down owner takes the chunk.

    Two admissions, one decision. The production grouped MoE route owns the Q5_1
    down leg for the Q5K gate/up window on its own, so the owner runs there
    whatever the plan published; everywhere else the plan-published runner field
    decides. Neither term narrows the other, so a plan that keeps the field off
    still gets the production owner on the chunks the production route owns.
    """

    return bool(production_grouped_moe or published)


def _qwen4_exp_q4_compact_tiles(
    *, rows: int, tile_m: int | None, tile_n: int | None,
) -> tuple[int, int]:
    """Resolve the compact F16 Q4_K dual gate/up tile pair for one dispatch.

    Two plan-published runner fields, one launch shape. The profile binder
    publishes 16/16 for every plan, which is what the retired process keys were
    pinned to; an unbound runner (no binder) keeps the historical heuristic -
    64 rows of work per tile from 512 rows up and 16 below it, with a 16-wide N
    tile. Each side resolves on its own, exactly like the two keys it replaces,
    and the owner's allowed-tile validation still rejects anything unsupported.
    """

    resolved_m = 64 if rows >= 512 else 16
    resolved_n = 16
    return (
        resolved_m if tile_m is None else tile_m,
        resolved_n if tile_n is None else tile_n,
    )


_Q5K_ROW4_PREFILL_UNSET = object()


@contextlib.contextmanager
def q5k_row4_prefill(runner: Any, enabled: bool | object = _Q5K_ROW4_PREFILL_UNSET):
    """Bind the runner's Q5_K gate/up row4 choice and restore it on exit."""

    prior = getattr(runner, "q5k_row4_prefill", False)
    if enabled is not _Q5K_ROW4_PREFILL_UNSET:
        runner.q5k_row4_prefill = bool(enabled)
    try:
        yield prior
    finally:
        runner.q5k_row4_prefill = prior


def qwen4_exp_grouped_row4_prefill_selected(
    weights: Mapping[str, GGUFDeviceWeight], *, rows: int, backend: str,
    published: bool = False,
) -> bool:
    if rows < 64 or not published:
        return False
    return all(
        is_registered(KernelKey(
            backend, "linear", weights[name].spec.quant_key,
            "selected_grouped_row4_gemv_bf16_bf16_out",
        ))
        for name in ("expert_gate", "expert_up")
    )


@contextlib.contextmanager
def q4_bundle_prefill(runner: Any, enabled: bool):
    """Temporarily bind bundled Q4 publication on one owning runner."""
    prior = getattr(runner, "q4_bundle_prefill", False)
    runner.q4_bundle_prefill = bool(enabled)
    try:
        yield prior
    finally:
        runner.q4_bundle_prefill = prior


def qwen4_exp_q4_bundle_prefill_selected(
    backend: str, quant: str, *, published: bool = False,
) -> bool:
    return (
        published
        and is_registered(KernelKey(
            backend, "moe_linear", quant,
            "selected_dual_grouped_rowbatch8_out4_expertgrid64_bundle_bf16_bf16_out"))
    )


def qwen4_exp_q4_pair_prefill_selected(
    backend: str, quant: str, *, rows: int, in_features: int
) -> bool:
    return (
        rows >= 64 and 0 < in_features <= 4096 and in_features % 256 == 0
        and os.environ.get("HIPENGINE_QWEN4_EXP_Q4_PAIR_PREFILL", "0")
        not in {"", "0", "false", "False"}
        and is_registered(KernelKey(
            backend, "moe_linear", quant, "selected_dual_grouped_pair2_bf16_bf16_out"))
    )


def _qwen4_exp_q4_exact_grouped_chain(
    *,
    production_grouped_moe: bool,
    rows: int,
    gate_quant: str,
    up_quant: str,
    exact_grouped_gate: bool,
    exact_grouped_all: bool,
    exact_grouped_down: bool,
) -> bool:
    """The exact grouped Q4_K gate/up chain's joint predicate.

    The chain runs on a chunk when every one of these holds, in this order:

    * the production grouped-MoE owner does not preempt it;
    * the chunk is multi-row (``rows >= 2``; decode never runs it);
    * the gate *and* up weights are Q4_K;
    * the owning runner published the chain's gate
      (``q4_exact_grouped_gate``, see :func:`q4_exact_grouped_gate`);
    * and the chunk is either one the exact grouped Q5_1 down chain already
      owns, or the runner published the ``_ALL`` joint term
      (``q4_exact_grouped_all``, see :func:`q4_exact_grouped_all`).

    Only the last two are runner decisions; the rest are dispatch-time state, so
    an arm cannot widen the chain past them. ``exact_grouped_down`` is the Q5_1
    chain's own local, already gated by ``q51_exact_grouped_down``.
    """

    return (
        not production_grouped_moe
        and rows >= 2
        and gate_quant == "gguf_q4_k"
        and up_quant == "gguf_q4_k"
        and exact_grouped_gate
        and (exact_grouped_down or exact_grouped_all)
    )


def _qwen4_exp_q4_gate_up_key(
    *,
    backend: str,
    quant: str,
    rows: int,
    in_features: int,
    q4_out4: bool = True,
    q4_bundle_prefill: bool = False,
) -> KernelKey:
    """The exact grouped Q4_K gate/up owner, before its own launch.

    This is the ladder the chain's guards select, in dispatch order: the exact
    expert-grid family (``EXACT_EXPERT_GRID``), the out4 fallback rung
    (``q4_out4``, see :func:`q4_out4`), the strict rowbatch8 base, then the pair2
    and bundle preemptors inside the expert-grid family. The caller owns
    everything above it - the routed-F16 owner, ``production_grouped_moe``, the
    chain's ``q4_exact_grouped_gate``/``q4_exact_grouped_all`` decisions and the
    iu8-risk+repair switch - so this function is only reached by a chunk that
    already runs the chain.

    Every input is either a dispatch-time guard (rows, in_features, the
    registry) or the owning runner's published decision, so the same call on the
    same runner and chunk always returns the same owner. The pair2 and bundle
    preemptors are gated on the expert-grid family, so they cannot replace this
    rung's owner while that family is closed - but they are read here rather
    than assumed. Every arm resolves through the registry, so a diagnostic
    wrapper registered on one of these keys sees the launch the dispatch makes.
    """

    expert_grid64 = os.environ.get(
        "HIPENGINE_QWEN4_EXP_EXACT_EXPERT_GRID", "64"
    ) in {"64", "q4"}
    if expert_grid64:
        variant = "selected_dual_grouped_rowbatch8_out4_expertgrid64_bf16_bf16_out"
    elif q4_out4:
        variant = "selected_dual_grouped_rowbatch8_out4_bf16_bf16_out"
    else:
        variant = "selected_dual_grouped_rowbatch8_bf16_bf16_out"
    if expert_grid64 and qwen4_exp_q4_pair_prefill_selected(
        backend, quant, rows=rows, in_features=in_features,
    ):
        variant = "selected_dual_grouped_pair2_bf16_bf16_out"
    elif expert_grid64 and qwen4_exp_q4_bundle_prefill_selected(
        backend, quant, published=q4_bundle_prefill,
    ):
        variant = "selected_dual_grouped_rowbatch8_out4_expertgrid64_bundle_bf16_bf16_out"
    return KernelKey(backend, "moe_linear", quant, variant)


def _qwen4_exp_q5_k_iu8_risk_multiplier() -> float:
    """Kahan-bound multiplier for the exact Q5_K iu8 risk criterion.

    Mirrors the Q5_1 down route's conservative default (16.0 keeps a
    measured ~1.3x margin over the flip-escape floor at the production
    multiplier screen); the Q5_K gate/up activations share the same
    magnitude class.
    """

    raw = os.environ.get("HIPENGINE_QWEN4_EXP_Q5_K_IU8_RISK_MULT", "16.0")
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(
            "HIPENGINE_QWEN4_EXP_Q5_K_IU8_RISK_MULT must be a positive float"
        )
    if not (value > 0.0) or value != value:
        raise ValueError(
            "HIPENGINE_QWEN4_EXP_Q5_K_IU8_RISK_MULT must be a positive float"
        )
    return value


def _qwen4_exp_q4_iu8_exact_enabled() -> bool:
    """Default-off exact iu8-risk+repair selected Q4 gate/up route."""

    return os.environ.get(
        "HIPENGINE_QWEN4_EXP_Q4_IU8_EXACT", "0"
    ) not in {"", "0", "false", "False"}


def _qwen4_exp_q51_iu8_exact_enabled() -> bool:
    """Default-off exact iu8-risk+repair selected Q5_1 down route."""

    return os.environ.get(
        "HIPENGINE_QWEN4_EXP_Q51_IU8_EXACT", "0"
    ) not in {"", "0", "false", "False"}


# T16 gate/up admission (2026-10-08): per-weight T16 tile cache keyed by the
# gate weight's raw device pointer. Populated by the integration helper (the
# same GGUF read + repack_gguf_q4_k_tile16 path the natural-owner screen
# validated); the runner branch falls back to the production iu8 chain on a
# cache miss, so the default path is untouched.
_T16_GATE_UP_TILES: dict[int, tuple[object, object]] = {}
_T16_GATE_UP_ENV = "HIPENGINE_QWEN4_EXP_T16_GATE_UP"


# T16 Q5_1 down route (2026-10-09 down-leg build, step 4): the same
# env-gated pattern as the gate/up T16 route. The A/B harness populates
# the tile cache (repack_gguf_q5_1_tile16_packed on the GGUF down weights,
# uploaded and keyed by the resident down weight's raw device pointer);
# the runner branch falls back to the production down chain on a cache
# miss, so the default path is untouched and the strict fallback is
# structural.
_T16_DOWN_TILES: dict[int, object] = {}
_T16_DOWN_ENV = "HIPENGINE_QWEN4_EXP_T16_DOWN"


# Wide-tile routed F16 GEMMs over the raw expert weights (fast profile): the
# gate/up pair with a SwiGLU epilogue and the down projection, each decoding a
# weight tile once per macro tile of grouped rows. The binder publishes the
# selector; without it the grouped chains below are untouched.
# WMMA QSA prefill attention (fast profile); see run_qwen4_exp_qsa_prefill_token_mixer.
_QSA_WMMA_ENV = "HIPENGINE_QWEN4_EXP_QSA_WMMA"
_ROUTED_F16_ENV = "HIPENGINE_QWEN4_EXP_ROUTED_F16"
# Grouped prefill row counts each gate/up quant routes from, matching the
# scopes the fast plan declares.
_ROUTED_F16_MIN_ROWS = {"gguf_q4_k": 2, "gguf_q5_k": 64}
_ROUTED_F16_GATE_UP_QUANTS = frozenset({"gguf_q4_k", "gguf_q5_k"})
_ROUTED_F16_DOWN_QUANTS = frozenset({"gguf_q5_1", "gguf_q8_0"})


def _routed_f16_enabled() -> bool:
    return os.environ.get(_ROUTED_F16_ENV, "0") not in {"", "0", "false", "False"}


def _routed_f16_gate_up_tile_rows(compact_rows: int, experts: int) -> int:
    """Macro tile for the paired gate/up GEMM from the mean expert bucket."""

    return 128 if compact_rows >= 64 * experts else 64


def _routed_f16_down_tile_rows(compact_rows: int, experts: int) -> int:
    """Macro tile for the down GEMM from the mean expert bucket."""

    if compact_rows >= 32 * experts:
        return 64
    return 48 if compact_rows >= 12 * experts else 16


def _routed_f16_tile_map(
    scratch: "Qwen4ExpMoEScratch",
    *,
    compact: int,
    experts: int,
    tile_rows: int,
    stream: int,
    runtime: HipRuntime,
) -> int:
    """Build the routed tile map in the grouped scratch; returns its capacity.

    The int32 entries and count reuse the grouped 16-row tile buffers, which
    are sized for at least as many int64 entries.
    """

    capacity = routed_f16_tile_capacity(compact, experts, tile_rows)
    if capacity * DType.INT32.itemsize > scratch.group_tile_expert.nbytes:
        raise RuntimeError("Qwen4Exp routed F16 tile capacity is invalid")
    qwen4_exp_routed_f16_tile_map(
        scratch.group_expert_start.ptr,
        scratch.group_tile_expert.ptr,
        scratch.group_wmma_total.ptr,
        experts,
        tile_rows=tile_rows,
        tile_capacity=capacity,
        stream=stream,
        runtime=runtime,
    )
    return capacity


def _t16_down_enabled() -> bool:
    """Default-off T16 Q5_1 expert-major down route."""

    return os.environ.get(
        _T16_DOWN_ENV, "0"
    ) not in {"", "0", "false", "False"}


def _t16_gate_up_enabled() -> bool:
    """Default-off T16 dual compensated gate/up route."""

    return os.environ.get(
        _T16_GATE_UP_ENV, "0"
    ) not in {"", "0", "false", "False"}


def _run_qwen4_exp_q51_wmma_down(
    *,
    input_ptr: int,
    expert_start_ptr: int,
    wmma_expert_start_ptr: int,
    tile_expert_ptr: int,
    weights_ptr: int,
    output_ptr: int,
    wmma_total_buffer: DeviceBuffer,
    compact_rows: int,
    num_experts: int,
    in_features: int,
    out_features: int,
    tile_capacity: int,
    stream: int,
    runtime: HipRuntime,
) -> int:
    """Run the F16 Q5_1 grouped down owner on a current 16-row WMMA tile map.

    The owner consumes a 16-row tile map and requires a positive multiple-of-16
    host row count. Only the non-exact gate/up paths build that map for the
    current invocation, so a down-only configuration (production exact gate/up
    with the F16 down owner) would otherwise pass a stale zero. Rebuilding the
    map here from this invocation's compact expert
    starts makes the map and count current by construction, independent of
    which gate/up route ran. The extra map launch and D2H readback are charged
    only on this F16 down route.
    """

    qwen35_moe_wmma_tile_map(
        expert_start_ptr,
        wmma_expert_start_ptr,
        tile_expert_ptr,
        wmma_total_buffer.ptr,
        num_experts,
        tile_capacity=tile_capacity,
        stream=stream,
        runtime=runtime,
    )
    wmma_total_host = np.empty(1, dtype=np.int64)
    if stream:
        runtime.stream_synchronize(stream)
    copy_device_to_host(
        host_array_ptr(wmma_total_host),
        wmma_total_buffer,
        DType.INT64.itemsize,
        runtime=runtime,
    )
    wmma_total_rows = int(wmma_total_host[0])
    if (
        wmma_total_rows <= 0
        or wmma_total_rows % 16
        or wmma_total_rows > tile_capacity * 16
    ):
        raise RuntimeError("Qwen4Exp F16 Q5_1 down tile row count is invalid")
    qwen4_exp_q5_1_selected_grouped_wmma_prefill_compact_bf16_bf16_out(
        input_ptr,
        expert_start_ptr,
        wmma_expert_start_ptr,
        tile_expert_ptr,
        weights_ptr,
        output_ptr,
        compact_rows,
        num_experts,
        in_features,
        out_features,
        wmma_total_rows,
        stream=stream,
        runtime=runtime,
    )
    return wmma_total_rows


def _qwen4_exp_q8_static_tile_count(compact_rows: int, num_experts: int) -> int:
    """Upper bound on 16-row WMMA tiles for the compact expert layout.

    Each active expert contributes at least one tile; the remaining compact
    rows contribute at most one additional tile per 16 rows. The tile-map
    kernel fills every capacity slot with ``-1`` before writing the actual
    experts, and the grouped WMMA kernels skip negative tile ids, so launching
    the upper bound is safe. This is the same bound the exact iu8 routes use.
    """

    active = min(int(compact_rows), int(num_experts))
    return active + (int(compact_rows) - active) // 16


def _run_qwen4_exp_q8_wmma_down(
    *,
    input_ptr: int,
    expert_start_ptr: int,
    wmma_expert_start_ptr: int,
    tile_expert_ptr: int,
    weights_ptr: int,
    output_ptr: int,
    wmma_total_buffer: DeviceBuffer,
    compact_rows: int,
    num_experts: int,
    in_features: int,
    out_features: int,
    tile_capacity: int,
    stream: int,
    runtime: HipRuntime,
    owner: Any,
    risk_kwargs: Mapping[str, Any],
) -> int:
    """Run the Q8_0 grouped WMMA down owner on a current 16-row tile map.

    ``scratch.group_wmma_total`` is shared by the grouped prefill path. The
    group-prefix kernel writes the compact padded row total there before the
    gate/up leg; the exact Q4/Q5 gate/up owners overwrite it with a 16-row tile
    map, but the strict Q4 owner does not. A Q8_0 down owner that reads the
    shared buffer after strict Q4 therefore sees the group-prefix total (for a
    1023-row chunk with top-k 10, 10230) rather than a tile row count, and the
    guarded wrapper rejects it because 10230 is not a multiple of 16.

    Rebuild the map here from this invocation's compact expert starts. Pass the
    static tile upper bound to the owner instead of reading the map's actual
    count back: the map kernel initializes the capacity window to ``-1`` and
    the grouped WMMA kernels skip negative tiles. This avoids a D2H readback
    and matches the existing exact iu8 tile-map launches. The map and launch
    bound are current by construction, independent of which gate/up route ran.
    """

    static_tiles = _qwen4_exp_q8_static_tile_count(compact_rows, num_experts)
    if static_tiles <= 0 or static_tiles > tile_capacity:
        raise RuntimeError("Qwen4Exp Q8_0 WMMA down tile count is invalid")
    qwen35_moe_wmma_tile_map(
        expert_start_ptr,
        wmma_expert_start_ptr,
        tile_expert_ptr,
        wmma_total_buffer.ptr,
        num_experts,
        tile_capacity=static_tiles,
        stream=stream,
        runtime=runtime,
    )
    wmma_total_rows = static_tiles * 16
    owner(
        input_ptr,
        expert_start_ptr,
        wmma_expert_start_ptr,
        tile_expert_ptr,
        weights_ptr,
        output_ptr,
        compact_rows,
        num_experts,
        in_features,
        out_features,
        wmma_total_rows,
        **dict(risk_kwargs),
        stream=stream,
        runtime=runtime,
    )
    return wmma_total_rows


def _qwen4_exp_q51_iu8_risk_multiplier() -> float:
    """Kahan-bound multiplier for the exact Q5_1 iu8 risk criterion.

    Screened on captured actual down-input activations across four cases
    and five layers: flip escapes at 4.0 (up to 7 per chunk-layer) are gone
    by 12.0; the default 16.0 keeps a ~1.3x margin over the measured floor
    at ~1.3x operation-complete speedup. Under the production-correctness
    contract a rare escape is bounded drift, not a contract violation;
    16.0 is chosen so the route stays bit-identical to the strict parent
    in practice, keeping every existing gate green.
    """

    raw = os.environ.get("HIPENGINE_QWEN4_EXP_Q51_IU8_RISK_MULT", "16.0")
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(
            "HIPENGINE_QWEN4_EXP_Q51_IU8_RISK_MULT must be a positive float"
        )
    if not (value > 0.0) or value != value:
        raise ValueError(
            "HIPENGINE_QWEN4_EXP_Q51_IU8_RISK_MULT must be a positive float"
        )
    return value


def _qwen4_exp_q4_iu8_risk_multiplier() -> float:
    """Kahan-bound multiplier for the exact iu8 risk criterion.

    Screened floor on actual weights is between 0.5 and 1 (below it, BF16
    flips escape the repair); the default 4.0 keeps a >=4x margin at
    1.50-1.82x operation-complete speedup.
    """

    raw = os.environ.get("HIPENGINE_QWEN4_EXP_Q4_IU8_RISK_MULT", "4.0")
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(
            "HIPENGINE_QWEN4_EXP_Q4_IU8_RISK_MULT must be a positive float"
        )
    if not (value > 0.0) or value != value:
        raise ValueError(
            "HIPENGINE_QWEN4_EXP_Q4_IU8_RISK_MULT must be a positive float"
        )
    return value


# Repair-rate instrumentation for the exact iu8 risk+repair routes.
#
# The risk criterion decides, per output element, whether the iu8-WMMA result
# is close enough to the exact parent to keep or must be recomputed by the
# sparse exact-repair kernel. Its trigger rate is the number that decides
# whether the pattern is worth extending to another family: a low rate keeps
# the speedup, a rate near 1.0 means the fast kernel only pays for repair it
# then discards. Nothing here runs unless the opt-in env var is set, because
# the readback synchronizes the caller's stream.
RISK_DIAGNOSTICS_ENV = "HIPENGINE_QWEN4_EXP_RISK_DIAGNOSTICS"
_RISK_DIAGNOSTIC_LIMIT = 8192
_risk_diagnostic_records: list[dict[str, Any]] = []


def qwen4_exp_risk_diagnostics_enabled() -> bool:
    """Whether the opt-in repair-rate readback is active."""

    return os.environ.get(RISK_DIAGNOSTICS_ENV, "0") not in {
        "", "0", "false", "False",
    }


def reset_qwen4_exp_risk_diagnostics() -> None:
    _risk_diagnostic_records.clear()


def qwen4_exp_risk_diagnostics() -> tuple[dict[str, Any], ...]:
    """Raw per-call repair observations, in execution order."""

    return tuple(_risk_diagnostic_records)


def _observe_qwen4_exp_risk_queue(
    *,
    runtime: HipRuntime,
    stream: int,
    risk_count: DeviceBuffer,
    route: str,
    role: str,
    layer: str,
    rows: int,
    compact_rows: int,
    out_features_total: int,
    experts: int,
    risk_capacity: int,
) -> None:
    """Read one risk-queue count back to host and record the observation.

    Diagnostic only. It synchronizes ``stream`` before the device-to-host
    copy, which is why the caller gates it behind the opt-in env var.
    """

    if not qwen4_exp_risk_diagnostics_enabled():
        return
    if len(_risk_diagnostic_records) >= _RISK_DIAGNOSTIC_LIMIT:
        return
    observed = np.empty(1, dtype=np.int32)
    if stream:
        runtime.stream_synchronize(stream)
    copy_device_to_host(
        host_array_ptr(observed),
        risk_count,
        DType.INT32.itemsize,
        runtime=runtime,
    )
    risk = int(observed[0])
    outputs = int(compact_rows) * int(out_features_total)
    _risk_diagnostic_records.append(
        {
            "route": route,
            "role": role,
            "layer": layer,
            "rows": int(rows),
            "compact_rows": int(compact_rows),
            "out_features_total": int(out_features_total),
            "experts": int(experts),
            "risk": risk,
            "outputs": outputs,
            "risk_capacity": int(risk_capacity),
            "repair_rate": (float(risk) / float(outputs)) if outputs > 0 else None,
            "over_capacity": bool(risk > int(risk_capacity)),
        }
    )


def summarize_qwen4_exp_risk_diagnostics(
    records: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Aggregate repair observations by route/role and by route/role/layer."""

    rows = list(_risk_diagnostic_records if records is None else records)

    def bucket() -> dict[str, Any]:
        return {
            "calls": 0,
            "outputs": 0,
            "risk": 0,
            "max_repair_rate": 0.0,
            "over_capacity_calls": 0,
        }

    def accumulate(target: dict[str, Any], record: Mapping[str, Any]) -> None:
        target["calls"] += 1
        target["outputs"] += int(record.get("outputs") or 0)
        target["risk"] += int(record.get("risk") or 0)
        rate = record.get("repair_rate")
        if isinstance(rate, (int, float)) and rate > target["max_repair_rate"]:
            target["max_repair_rate"] = float(rate)
        if record.get("over_capacity"):
            target["over_capacity_calls"] += 1

    by_role: dict[tuple[str, str], dict[str, Any]] = {}
    by_layer: dict[tuple[str, str, str], dict[str, Any]] = {}
    for record in rows:
        accumulate(by_role.setdefault(
            (str(record.get("route")), str(record.get("role"))), bucket()
        ), record)
        accumulate(by_layer.setdefault(
            (str(record.get("route")), str(record.get("role")),
             str(record.get("layer"))), bucket()
        ), record)

    def finish(entry: dict[str, Any]) -> dict[str, Any]:
        outputs = entry["outputs"]
        entry["aggregate_repair_rate"] = (
            float(entry["risk"]) / float(outputs) if outputs > 0 else None
        )
        return entry

    return {
        "calls": len(rows),
        "by_role": {
            f"{route}:{role}": finish(entry)
            for (route, role), entry in sorted(by_role.items())
        },
        "by_layer": {
            f"{route}:{role}:{layer}": finish(entry)
            for (route, role, layer), entry in sorted(by_layer.items())
        },
    }


@contextlib.contextmanager
def q51_pair_prefill(runner: Any, enabled: bool):
    """Bind pair2 selection on one runner and restore it on exit."""
    prior = getattr(runner, "q51_pair_prefill", False)
    runner.q51_pair_prefill = bool(enabled)
    try:
        yield prior
    finally:
        runner.q51_pair_prefill = prior


def qwen4_exp_q51_pair_prefill_selected(
    backend: str, quant: str, *, rows: int, in_features: int,
    published: bool = False,
) -> bool:
    return (
        rows >= 64 and in_features <= 4096
        and published
        and is_registered(KernelKey(
            backend, "moe_linear", quant, "selected_grouped_prefill_pair2_bf16_bf16_out"))
    )


_MOE_BACKEND_REGISTRY_GENERATIONS: dict[str, int] = {}


def _ensure_qwen4_exp_moe_backend(backend: str) -> None:
    generation = _registry_generation()
    if _MOE_BACKEND_REGISTRY_GENERATIONS.get(backend) == generation:
        return
    load_backend_kernel_package(backend)
    # Do not certify a generation that changed during the refresh.
    if _registry_generation() == generation:
        _MOE_BACKEND_REGISTRY_GENERATIONS[backend] = generation
    else:
        _MOE_BACKEND_REGISTRY_GENERATIONS.pop(backend, None)


def _qwen4_exp_selected_gemv_variant(quant_key: str) -> str:
    """The registered selected-GEMV owner for one expert projection.

    Q5_1 uses the logical-256 t64 reduction. The wave64 variant remains
    registered for direct numerical diagnostics, not selected by this dispatch.
    """

    if quant_key == "gguf_q5_1":
        return "selected_gemv_logical256_t64_bf16_bf16_out"
    return "selected_gemv_bf16_bf16_out"


def run_qwen4_exp_moe(
    mixed_ptr: int,
    weights: Mapping[str, GGUFDeviceWeight],
    *,
    q4_bundle_prefill: bool = False,
    q5k_row4_prefill: bool = False,
    scratch: Qwen4ExpMoEScratch,
    rows: int,
    hidden: int,
    ffn: int,
    experts: int,
    top_k: int,
    stream: int = 0,
    runtime: HipRuntime | None = None,
    graph_cache: MoeGraphCache | None = None,
    graph_key: object | None = None,
    q51_down_m1: bool = False,
    q51_pair_prefill: bool = False,
    q51_fold128_prefill: bool = False,
    q51_fold_pair_prefill: bool = False,
    q51_register_cache: bool = False,
    q51_row_publish: bool = False,
    forkb_grouped_down: bool = False,
    q8_down_row4_prefill: bool = False,
    q8_down_bundle_prefill: bool = False,
    q8_down_register: bool = False,
    q8_mapped_down: bool = False,
    q51_exact_grouped_down: bool = True,
    q51_wmma_down: bool = False,
    q51_out8: bool = True,
    q4_exact_grouped_gate: bool = True,
    q4_exact_grouped_all: bool = True,
    q4_out4: bool = True,
    q4_tile_m: int | None = None,
    q4_tile_n: int | None = None,
) -> Qwen4ExpMoEDeviceResult:
    """Run normalized softmax top-k routed experts plus gated shared expert.

    ``q5k_row4_prefill`` is the owning runner's gate/up row4 choice. It keeps
    the rows>=64 and registered-kernel guards, below grouped-chain preemptors.

    ``q51_down_m1`` is the owning runner's plan-published Q5_1 down M1 decision
    (see :func:`q51_down_m1`); it only selects inside the exact grouped Q5_1
    down chain's small-chunk scope, below the pair2/fold ladder that preempts it
    at rows>=64, and never on decode.

    ``forkb_grouped_down`` is the owning runner's plan-published PF-1 fork-b
    decision (see :func:`forkb_grouped_down`); it only selects the sorted
    chain's Q8_0 down owner when no Q8_0 WMMA down owner is bound, and never on
    decode.

    ``q51_fold128_prefill`` is the owning runner's plan-published Q5_1 fold128
    decision (see :func:`q51_fold128_prefill`); it only selects inside the
    grouped pair2 chain's rows>=64 scope and never on decode.

    ``q51_fold_pair_prefill`` is the owning runner's plan-published Q5_1
    folded-pair decision (see :func:`q51_fold_pair_prefill`); it only selects
    inside the fold128 ladder's rows>=512 scope and never on decode.

    ``q8_down_row4_prefill`` is the owning runner's plan-published Q8_0 down
    row4 decision (see :func:`q8_down_row4_prefill`); it only selects inside the
    sorted grouped chain's rows>=512 scope and never on decode.

    ``q8_down_bundle_prefill`` is the owning runner's plan-published Q8_0 down
    bundle decision (see :func:`q8_down_bundle_prefill`); it only selects inside
    the row4 ladder's rows>=512 scope and never on decode.

    ``q51_register_cache`` is the owning runner's plan-published Q5_1
    register-cache fallback decision (see :func:`q51_register_cache`); it only
    selects inside the grouped pair2 chain's rows>=512 K640 scope and never on
    decode.

    ``q51_row_publish`` is the owning runner's plan-published Q5_1 row-publish
    decision (see :func:`q51_row_publish`); it only selects inside the grouped
    pair2 chain's rows>=512 K640 scope and never on decode.

    ``q51_out8`` is the owning runner's out8 fallback decision for the exact
    grouped Q5_1 down chain (see :func:`q51_out8`); it only picks between the
    out8 rowbatch8 owner and the output1 base on the chain's non-expertgrid
    rung, so the exact expert-grid family above it, the exact iu8-risk+repair
    chain above that, and the pair2/fold siblings below it are unaffected, and
    it never selects on decode.

    ``q4_exact_grouped_gate`` is the owning runner's gate for the exact grouped
    Q4_K gate/up chain (see :func:`q4_exact_grouped_gate`), and
    ``q4_exact_grouped_all`` is that chain's joint `_ALL` term (see
    :func:`q4_exact_grouped_all`). The chain's other guards - rows>=2, a Q4_K/Q4_K
    gate/up pair and the ``production_grouped_moe`` preemptor - are dispatch-time
    state, not runner decisions, so an arm cannot widen the chain past them.
    Closing the gate falls through to the plain-iu8 or F16 compact owner below it
    and makes every rung of the Q4 ladder unreachable.

    ``q4_out4`` is the owning runner's out4 fallback decision for the exact
    grouped Q4_K gate/up chain (see :func:`q4_out4`); it only picks between the
    out4 rowbatch8 owner and the strict rowbatch8 base on the chain's
    non-expertgrid rung, so the routed-F16 owner, the chain's own switches, the
    exact iu8-risk+repair chain and the expert-grid family's pair2/bundle
    preemptors are unaffected, and it never selects on decode.

    ``q8_down_register`` is the owning runner's plan-published Q8_0 down
    register decision (see :func:`q8_down_register`); it selects inside the
    sorted grouped chain and the token-major mapped path for the same
    rows>=512 K640 scope, and never on decode.

    ``q8_mapped_down`` is the owning runner's plan-published Q8_0 token-major
    mapped down decision (see :func:`q8_mapped_down`); it only admits the
    bundle leaf when this call built the Q5_K row4 map and rows>=512, and never
    on decode.

    ``q51_exact_grouped_down`` is the owning runner's plan-published Q5_1 exact
    grouped down decision (see :func:`q51_exact_grouped_down`); it is the gate
    of the exact grouped Q5_1 down chain itself, so ``False`` skips that chain
    for the whole multi-row prefill and falls through to the sorted/WMMA owners
    below it. It never selects on decode.

    ``q4_tile_m`` / ``q4_tile_n`` are the owning runner's plan-published tile
    pair for the compact F16 Q4_K dual gate/up fallback owner (see
    :func:`q4_tile_m`); ``None`` on either side keeps the dispatch heuristic
    for that side. The pair only shapes that owner's launch, below the exact
    grouped Q4_K chain and the iu8 owner.

    ``q51_wmma_down`` is the owning runner's plan-published admission of the F16
    WMMA compact grouped Q5_1 down owner (see :func:`q51_wmma_down`). It only
    decides below the exact grouped Q5_1 down chain, and the production grouped
    MoE route admits that owner on its own: the dispatch runs it whenever
    ``production_grouped_moe`` owns the chunk, whatever this field says, so the
    field only widens the route to the chunks the production route does not own.
    """

    if scratch.closed:
        raise RuntimeError("Qwen4Exp MoE scratch is closed")
    if rows <= 0:
        raise ValueError("Qwen4Exp MoE rows must be positive")
    active_runtime = runtime or scratch.runtime
    if active_runtime is not scratch.runtime:
        raise ValueError("runtime must match the MoE scratch owner")
    if graph_cache is not None:
        if rows != 1 or graph_key is None:
            raise ValueError("Qwen4Exp MoE graph requires c1 and an explicit key")

        def eager(graph_stream: int) -> None:
            run_qwen4_exp_moe(
                mixed_ptr,
                weights,
                q4_bundle_prefill=q4_bundle_prefill,
                q5k_row4_prefill=q5k_row4_prefill,
                scratch=scratch,
                rows=rows,
                hidden=hidden,
                ffn=ffn,
                experts=experts,
                top_k=top_k,
                stream=graph_stream,
                runtime=active_runtime,
                q51_down_m1=q51_down_m1,
                q51_pair_prefill=q51_pair_prefill,
                q51_fold128_prefill=q51_fold128_prefill,
                q51_fold_pair_prefill=q51_fold_pair_prefill,
                q51_register_cache=q51_register_cache,
                q51_row_publish=q51_row_publish,
                q51_exact_grouped_down=q51_exact_grouped_down,
                q51_wmma_down=q51_wmma_down,
                q51_out8=q51_out8,
                q4_exact_grouped_gate=q4_exact_grouped_gate,
                q4_exact_grouped_all=q4_exact_grouped_all,
                q4_out4=q4_out4,
                q4_tile_m=q4_tile_m,
                q4_tile_n=q4_tile_n,
                forkb_grouped_down=forkb_grouped_down,
                q8_down_row4_prefill=q8_down_row4_prefill,
                q8_down_bundle_prefill=q8_down_bundle_prefill,
                q8_down_register=q8_down_register,
                q8_mapped_down=q8_mapped_down,
            )

        graph_cache.run(
            graph_key,
            eager=eager,
            out_ptr=scratch.output.ptr,
            out_nbytes=rows * hidden * DType.BF16.itemsize,
            stream=stream,
        )
        return Qwen4ExpMoEDeviceResult(
            scratch.output, scratch.selected, scratch.routing
        )
    required = {
        "router",
        "expert_gate",
        "expert_up",
        "expert_down",
        "shared_gate",
        "shared_up",
        "shared_down",
        "shared_gate_weight",
    }
    missing = sorted(required - set(weights))
    if missing:
        raise ValueError("missing Qwen4Exp MoE weights: " + ", ".join(missing))
    backend = str(weights["expert_gate"].backend)
    _ensure_qwen4_exp_moe_backend(backend)
    router_variant = _qwen4_exp_router_producer_variant(rows)
    if router_variant is not None:
        router_logits = resolve(
            backend=backend,
            layer="router_logits",
            quant="f32",
            variant=router_variant,
        )
        router_logits(
            mixed_ptr,
            weights["router"].allocation("raw").tensor.ptr,
            scratch.router_logits.ptr,
            rows,
            hidden,
            experts,
            stream=stream,
            runtime=active_runtime,
        )
    else:
        launch_gguf_linear(
            weights["router"], mixed_ptr, scratch.router_logits.ptr,
            rows, hidden, experts,
            activation_dtype=GGUF_ACTIVATION_F32,
            output_dtype=GGUF_OUTPUT_F32,
            stream=stream, runtime=active_runtime,
        )
    select_cast = resolve(
        backend=backend, layer="router_select+cast", quant="fp32",
    )
    select_cast(
        scratch.router_logits.ptr, scratch.selected.ptr, scratch.routing.ptr,
        mixed_ptr, scratch.hidden_bf16.ptr, rows, experts, experts, top_k, hidden,
        stream=stream, runtime=active_runtime,
    )
    compact = rows * top_k

    def selected_projection(
        slot: str,
        input_ptr: int,
        output_ptr: int,
        x_rows: int,
        selected_rows: int,
        in_features: int,
        out_features: int,
        *,
        selected_ptr: int | None = None,
    ) -> None:
        weight = weights[slot]
        variant = _qwen4_exp_selected_gemv_variant(weight.spec.quant_key)
        function = resolve(
            backend=backend,
            layer="linear",
            quant=weight.spec.quant_key,
            variant=variant,
        )
        function(
            input_ptr,
            scratch.selected.ptr if selected_ptr is None else selected_ptr,
            weight.allocation("raw").tensor.ptr,
            output_ptr,
            x_rows,
            selected_rows,
            experts,
            in_features,
            out_features,
            stream=stream,
            runtime=active_runtime,
        )

    production_grouped_moe = _qwen4_exp_production_moe_prefill_enabled(
        weights["expert_gate"], rows=rows
    )
    exact_grouped_down = (
        not production_grouped_moe
        and weights["expert_down"].spec.quant_key == "gguf_q5_1"
        and rows >= 2
        and q51_exact_grouped_down
    )
    # T16 down route (down-leg build step 4): env-gated, tile-cache-keyed.
    # A cache miss falls through to the chain below, so the default path is
    # untouched.
    t16_down = (
        _t16_down_enabled()
        and weights["expert_down"].spec.quant_key == "gguf_q5_1"
        and rows >= 2
        and ffn % 32 == 0
        and hidden % 16 == 0
    )
    t16_down_tiles = None
    if t16_down:
        down_w = weights["expert_down"]
        down_allocs = getattr(down_w, "allocations", {})
        if "t16" in down_allocs:
            # Binding-owned tile-residency stage
            # (docs/QWEN4EXP-T16-RESIDENCY.md): the tiles live on the
            # weight, not the module sidecar.
            t16_down_tiles = down_allocs["t16"].buffer
        else:
            t16_down_raw_ptr = int(
                down_w.allocation("raw").tensor.ptr
            )
            t16_down_tiles = _T16_DOWN_TILES.get(t16_down_raw_ptr)
        t16_down = t16_down_tiles is not None
    exact_grouped_q4_gate = _qwen4_exp_q4_exact_grouped_chain(
        production_grouped_moe=production_grouped_moe,
        rows=rows,
        gate_quant=weights["expert_gate"].spec.quant_key,
        up_quant=weights["expert_up"].spec.quant_key,
        exact_grouped_gate=q4_exact_grouped_gate,
        exact_grouped_all=q4_exact_grouped_all,
        exact_grouped_down=exact_grouped_down,
    )
    # #15 Q5_K bundle KL-shaving variant: default-off exact iu8-risk+repair
    # selected Q5_K gate/up route (the layer-2 Q5_K layer).
    q5_k_iu8_exact = (
        os.environ.get("HIPENGINE_QWEN4_EXP_Q5_K_IU8_EXACT", "0")
        not in {"", "0", "false", "False"}
        and rows >= 2
        and hidden % 256 == 0
        and ffn % 128 == 0
    )
    gate_quants = {
        weights["expert_gate"].spec.quant_key,
        weights["expert_up"].spec.quant_key,
    }
    # The down GEMM reads the grouped SwiGLU rows the paired gate/up writes,
    # so the two are selected together.
    routed_f16 = (
        _routed_f16_enabled()
        and len(gate_quants) == 1
        and gate_quants <= _ROUTED_F16_GATE_UP_QUANTS
        and rows >= _ROUTED_F16_MIN_ROWS[weights["expert_gate"].spec.quant_key]
        and hidden % 256 == 0
        and weights["expert_down"].spec.quant_key in _ROUTED_F16_DOWN_QUANTS
        and ffn % 64 == 0
    )
    if routed_f16:
        t16_down = False
    grouped_prefill = (
        routed_f16
        or exact_grouped_down or exact_grouped_q4_gate or q5_k_iu8_exact or (
            production_grouped_moe
            and (
                weights["expert_gate"].spec.quant_key,
                weights["expert_up"].spec.quant_key,
            )
            in {
                ("gguf_q4_k", "gguf_q4_k"),
                ("gguf_q5_k", "gguf_q5_k"),
            }
        )
    )
    if grouped_prefill:
        active_runtime.memset_async(
            scratch.group_counts.ptr, 0, scratch.group_counts.nbytes, stream
        )
        qwen35_moe_group_count(
            scratch.selected.ptr,
            scratch.group_counts.ptr,
            compact,
            experts,
            stream=stream,
            runtime=active_runtime,
        )
        qwen35_moe_group_prefix(
            scratch.group_counts.ptr,
            scratch.group_padded_counts.ptr,
            scratch.group_expert_start.ptr,
            scratch.group_wmma_total.ptr,
            experts,
            1,
            stream=stream,
            runtime=active_runtime,
        )
        if t16_down:
            # The expert-major down kernel consumes a device-resident list of
            # nonempty experts; this extra single-block launch is charged to
            # the T16 down route. The standard prefix above still feeds every
            # other consumer of the grouped map.
            qwen35_moe_group_prefix_active(
                scratch.group_counts.ptr,
                scratch.group_expert_start.ptr,
                scratch.group_active_experts.ptr,
                scratch.group_active_count.ptr,
                experts,
                stream=stream,
                runtime=active_runtime,
            )
        active_runtime.memset_async(
            scratch.group_scatter_offsets.ptr,
            0,
            scratch.group_scatter_offsets.nbytes,
            stream,
        )
        # The routed gate/up kernel can read token rows through the lane map,
        # so its grouped copy of the activations (rows * top_k * hidden BF16)
        # is not built. HIPENGINE_QWEN4_EXP_ROUTED_F16_INDEXED=0 restores the
        # gathered copy; see docs/REFACTOR.md.
        routed_indexed = routed_f16 and os.environ.get(
            "HIPENGINE_QWEN4_EXP_ROUTED_F16_INDEXED", "1"
        ) not in {"", "0", "false", "False"}
        if routed_indexed:
            qwen35_moe_group_scatter(
                scratch.selected.ptr,
                scratch.routing.ptr,
                scratch.group_expert_start.ptr,
                scratch.group_scatter_offsets.ptr,
                scratch.group_sorted_lanes.ptr,
                scratch.group_sorted_experts.ptr,
                scratch.group_sorted_weights.ptr,
                compact,
                experts,
                stream=stream,
                runtime=active_runtime,
            )
        else:
            qwen35_moe_group_scatter_gather_lowp(
                scratch.hidden_bf16.ptr,
                scratch.selected.ptr,
                scratch.routing.ptr,
                scratch.group_expert_start.ptr,
                scratch.group_scatter_offsets.ptr,
                scratch.group_sorted_lanes.ptr,
                scratch.group_sorted_experts.ptr,
                scratch.group_sorted_weights.ptr,
                scratch.expert_down.ptr,
                compact,
                experts,
                top_k,
                hidden,
                stream=stream,
                runtime=active_runtime,
            )
        tile_capacity = scratch.group_tile_expert.nbytes // DType.INT64.itemsize
        wmma_total_rows = 0
        if not exact_grouped_down and not exact_grouped_q4_gate:
            qwen35_moe_wmma_tile_map(
                scratch.group_expert_start.ptr,
                scratch.group_wmma_expert_start.ptr,
                scratch.group_tile_expert.ptr,
                scratch.group_wmma_total.ptr,
                experts,
                tile_capacity=tile_capacity,
                stream=stream,
                runtime=active_runtime,
            )
            wmma_total_host = np.empty(1, dtype=np.int64)
            if stream:
                active_runtime.stream_synchronize(stream)
            copy_device_to_host(
                host_array_ptr(wmma_total_host),
                scratch.group_wmma_total,
                DType.INT64.itemsize,
                runtime=active_runtime,
            )
            wmma_total_rows = int(wmma_total_host[0])
            if wmma_total_rows <= 0 or wmma_total_rows > tile_capacity * 16:
                raise RuntimeError("Qwen4Exp grouped MoE tile row count is invalid")
        if routed_f16:
            gate_quant = weights["expert_gate"].spec.quant_key
            gate_up_tile_rows = _routed_f16_gate_up_tile_rows(compact, experts)
            gate_up_capacity = _routed_f16_tile_map(
                scratch,
                compact=compact,
                experts=experts,
                tile_rows=gate_up_tile_rows,
                stream=stream,
                runtime=active_runtime,
            )
            resolve(
                backend=backend,
                layer="moe_linear",
                quant=gate_quant,
                variant=ROUTED_F16_GATE_UP_VARIANT,
            )(
                scratch.hidden_bf16.ptr if routed_indexed else scratch.expert_down.ptr,
                scratch.group_expert_start.ptr,
                scratch.group_tile_expert.ptr,
                scratch.group_wmma_total.ptr,
                weights["expert_gate"].allocation("raw").tensor.ptr,
                weights["expert_up"].allocation("raw").tensor.ptr,
                scratch.expert_intermediate.ptr,
                hidden,
                ffn,
                tile_rows=gate_up_tile_rows,
                tile_capacity=gate_up_capacity,
                row_lanes_ptr=scratch.group_sorted_lanes.ptr if routed_indexed else 0,
                top_k=top_k,
                stream=stream,
                runtime=active_runtime,
            )
        elif exact_grouped_q4_gate:
            q4_iu8_exact = (
                _qwen4_exp_q4_iu8_exact_enabled()
                and rows >= 2
                and hidden % 256 == 0
                and ffn % 128 == 0
                and weights["expert_gate"].spec.quant_key == "gguf_q4_k"
                and weights["expert_up"].spec.quant_key == "gguf_q4_k"
            )
            if q4_iu8_exact:
                # T16 gate/up branch (2026-10-08 admission, env-gated, default
                # off): the bit-lossless T16 repack decoded in registers, F16
                # WMMA with compensated accumulate, no repair by construction.
                # Tiles come from a module-level cache keyed by the gate
                # weight's raw device pointer; a cache miss falls through to
                # the production iu8-risk+repair chain, so the strict fallback
                # is structural and the default path is untouched.
                t16_tiles = None
                if _t16_gate_up_enabled():
                    gate_w = weights["expert_gate"]
                    up_w = weights["expert_up"]
                    gate_allocs = getattr(gate_w, "allocations", {})
                    up_allocs = getattr(up_w, "allocations", {})
                    if "t16" in gate_allocs and "t16" in up_allocs:
                        # Binding-owned tile-residency stage
                        # (docs/QWEN4EXP-T16-RESIDENCY.md): the tiles live on
                        # the weight, not the module sidecar.
                        t16_tiles = (
                            gate_allocs["t16"].buffer,
                            up_allocs["t16"].buffer,
                        )
                    else:
                        gate_raw_ptr = int(
                            gate_w.allocation("raw").tensor.ptr
                        )
                        t16_tiles = _T16_GATE_UP_TILES.get(gate_raw_ptr)
                if t16_tiles is not None:
                    tile_capacity = (
                        scratch.group_tile_expert.nbytes // DType.INT64.itemsize
                    )
                    active = min(compact, experts)
                    static_tiles = active + (compact - active) // 16
                    if static_tiles > tile_capacity:
                        raise RuntimeError(
                            "Qwen4Exp grouped MoE tile capacity is invalid"
                        )
                    qwen35_moe_wmma_tile_map(
                        scratch.group_expert_start.ptr,
                        scratch.group_wmma_expert_start.ptr,
                        scratch.group_tile_expert.ptr,
                        scratch.group_wmma_total.ptr,
                        experts,
                        tile_capacity=static_tiles,
                        stream=stream,
                        runtime=active_runtime,
                    )
                    gguf_q4_k_t16_selected_dual_wmma_prefill_compact32_bf16_bf16_out(
                        scratch.expert_down.ptr,
                        scratch.group_expert_start.ptr,
                        scratch.group_wmma_expert_start.ptr,
                        scratch.group_tile_expert.ptr,
                        t16_tiles[0].ptr,
                        t16_tiles[1].ptr,
                        scratch.group_gate_up.ptr,
                        compact,
                        hidden,
                        ffn,
                        ffn,
                        experts,
                        static_tiles * 16,
                        stream=stream,
                        runtime=active_runtime,
                    )
                    silu_mul_dual_out_bf16(
                        scratch.group_gate_up.ptr,
                        scratch.expert_intermediate.ptr,
                        rows=compact,
                        features=ffn,
                        stream=stream,
                        runtime=active_runtime,
                    )
                else:
                    # Exact iu8-risk+repair route: GPU tile map with the static
                    # upper bound (no device read), risk-collecting iu8-WMMA
                    # gate/up, sparse exact repair reproducing the pair2 parent,
                    # then the combined-layout SiLU (bit-identical element-wise
                    # to the separate-layout SiLU used by the pair2 path).
                    tile_capacity = (
                        scratch.group_tile_expert.nbytes // DType.INT64.itemsize
                    )
                    active = min(compact, experts)
                    static_tiles = active + (compact - active) // 16
                    if static_tiles > tile_capacity:
                        raise RuntimeError(
                            "Qwen4Exp grouped MoE tile capacity is invalid"
                        )
                    qwen35_moe_wmma_tile_map(
                        scratch.group_expert_start.ptr,
                        scratch.group_wmma_expert_start.ptr,
                        scratch.group_tile_expert.ptr,
                        scratch.group_wmma_total.ptr,
                        experts,
                        tile_capacity=static_tiles,
                        stream=stream,
                        runtime=active_runtime,
                    )
                    risk_count, risk_indices = scratch.ensure_group_risk_buffers(
                        compact_rows=compact, out_features_total=2 * ffn
                    )
                    active_runtime.memset_async(
                        risk_count.ptr, 0, DType.INT32.itemsize, stream
                    )
                    risk_capacity = compact * 2 * ffn
                    iu8_gate_up_variant = (
                        "selected_dual_wmma_iu8_risk_p2_prefill_bf16_bf16_out"
                        if os.environ.get("HIPENGINE_QWEN4_EXP_Q4_IU8_PLANES", "3") == "2"
                        else "selected_dual_wmma_iu8_risk_prefill_bf16_bf16_out")
                    iu8_risk_gate_up = resolve(
                        backend=backend, layer="moe_linear", quant="gguf_q4_k",
                        variant=iu8_gate_up_variant,
                    )
                    sparse_exact_repair = resolve(
                        backend=backend, layer="moe_linear", quant="gguf_q4_k",
                        variant="selected_dual_sparse_exact_repair_bf16",
                    )
                    iu8_risk_gate_up(
                        scratch.expert_down.ptr,
                        scratch.group_expert_start.ptr,
                        scratch.group_wmma_expert_start.ptr,
                        scratch.group_tile_expert.ptr,
                        weights["expert_gate"].allocation("raw").tensor.ptr,
                        weights["expert_up"].allocation("raw").tensor.ptr,
                        scratch.group_gate_up.ptr,
                        risk_count.ptr,
                        risk_indices.ptr,
                        risk_capacity,
                        _qwen4_exp_q4_iu8_risk_multiplier(),
                        compact,
                        hidden,
                        ffn,
                        ffn,
                        experts,
                        static_tiles * 16,
                        stream=stream,
                        runtime=active_runtime,
                    )
                    sparse_exact_repair(
                        scratch.expert_down.ptr,
                        scratch.group_expert_start.ptr,
                        weights["expert_gate"].allocation("raw").tensor.ptr,
                        weights["expert_up"].allocation("raw").tensor.ptr,
                        scratch.group_gate_up.ptr,
                        risk_count.ptr,
                        risk_indices.ptr,
                        risk_capacity,
                        compact,
                        hidden,
                        ffn,
                        ffn,
                        experts,
                        stream=stream,
                        runtime=active_runtime,
                    )
                    _observe_qwen4_exp_risk_queue(
                        runtime=active_runtime,
                        stream=stream,
                        risk_count=risk_count,
                        route="q4_iu8_exact",
                        role="expert_gate_up",
                        layer=weights["expert_gate"].spec.slot_path,
                        rows=rows,
                        compact_rows=compact,
                        out_features_total=2 * ffn,
                        experts=experts,
                        risk_capacity=risk_capacity,
                    )
                    silu_mul_dual_out_bf16(
                        scratch.group_gate_up.ptr,
                        scratch.expert_intermediate.ptr,
                        rows=compact,
                        features=ffn,
                        stream=stream,
                        runtime=active_runtime,
                    )
            else:
                grouped_q4_key = _qwen4_exp_q4_gate_up_key(
                    backend=backend,
                    quant=weights["expert_gate"].spec.quant_key,
                    rows=rows, in_features=hidden,
                    q4_out4=q4_out4,
                    q4_bundle_prefill=q4_bundle_prefill,
                )
                grouped_q4_gate = resolve(
                    backend=grouped_q4_key.backend, layer=grouped_q4_key.layer,
                    quant=grouped_q4_key.quant, variant=grouped_q4_key.variant)
                grouped_q4_gate(
                    scratch.expert_down.ptr,
                    scratch.group_expert_start.ptr,
                    weights["expert_gate"].allocation("raw").tensor.ptr,
                    weights["expert_up"].allocation("raw").tensor.ptr,
                    scratch.expert_gate.ptr,
                    scratch.expert_up.ptr,
                    compact,
                    experts,
                    hidden,
                    ffn,
                    stream=stream,
                    runtime=active_runtime,
                )
                silu_mul_separate_out_bf16(
                    scratch.expert_gate.ptr,
                    scratch.expert_up.ptr,
                    scratch.expert_intermediate.ptr,
                    compact,
                    ffn,
                    stream=stream,
                    runtime=active_runtime,
                )
        elif exact_grouped_down:
            selected_projection(
                "expert_gate", scratch.hidden_bf16.ptr, scratch.expert_gate.ptr,
                rows, compact, hidden, ffn,
            )
            selected_projection(
                "expert_up", scratch.hidden_bf16.ptr, scratch.expert_up.ptr,
                rows, compact, hidden, ffn,
            )
            silu_mul_separate_out_bf16(
                scratch.expert_gate.ptr,
                scratch.expert_up.ptr,
                scratch.expert_intermediate.ptr,
                compact,
                ffn,
                stream=stream,
                runtime=active_runtime,
            )
            qwen4_exp_gather_bf16_lanes(
                scratch.expert_intermediate.ptr,
                scratch.group_sorted_lanes.ptr,
                scratch.expert_gate.ptr,
                compact,
                ffn,
                stream=stream,
                runtime=active_runtime,
            )
        elif weights["expert_gate"].spec.quant_key == "gguf_q4_k":
            q4_iu8 = (
                os.environ.get("HIPENGINE_QWEN4_EXP_Q4_IU8_PREFILL", "0")
                not in {"", "0", "false", "False"}
                and rows >= 2
                and hidden % 256 == 0
                and ffn % 128 == 0
                and _qwen4_exp_q4_iu8_layer_allowed(weights)
            )
            if q4_iu8:
                gguf_q4_k_selected_dual_wmma_iu8_prefill_bf16_bf16_out(
                    scratch.expert_down.ptr,
                    scratch.group_expert_start.ptr,
                    scratch.group_wmma_expert_start.ptr,
                    scratch.group_tile_expert.ptr,
                    weights["expert_gate"].allocation("raw").tensor.ptr,
                    weights["expert_up"].allocation("raw").tensor.ptr,
                    scratch.group_gate_up.ptr,
                    compact,
                    hidden,
                    ffn,
                    ffn,
                    experts,
                    wmma_total_rows,
                    stream=stream,
                    runtime=active_runtime,
                )
            else:
                compact_tile_m, compact_tile_n = _qwen4_exp_q4_compact_tiles(
                    rows=rows, tile_m=q4_tile_m, tile_n=q4_tile_n
                )
                gguf_q4_k_selected_dual_wmma_prefill_compact_bf16_bf16_out(
                    scratch.expert_down.ptr,
                    scratch.group_expert_start.ptr,
                    scratch.group_wmma_expert_start.ptr,
                    scratch.group_tile_expert.ptr,
                    weights["expert_gate"].allocation("raw").tensor.ptr,
                    weights["expert_up"].allocation("raw").tensor.ptr,
                    scratch.group_gate_up.ptr,
                    compact,
                    hidden,
                    ffn,
                    ffn,
                    experts,
                    wmma_total_rows,
                    tile_m=compact_tile_m,
                    tile_n=compact_tile_n,
                    stream=stream,
                    runtime=active_runtime,
                )
            silu_mul_dual_out_bf16(
                scratch.group_gate_up.ptr,
                scratch.expert_intermediate.ptr,
                rows=compact,
                features=ffn,
                stream=stream,
                runtime=active_runtime,
            )
        elif (
            weights["expert_gate"].spec.quant_key == "gguf_q5_k"
            and weights["expert_up"].spec.quant_key == "gguf_q5_k"
            and q5_k_iu8_exact
        ):
            # #15 Q5_K KL-shaving variant: exact iu8-risk+repair gate/up for
            # the Q5_K layer(s), mirroring the Q4_K production route. The
            # 3-plane activation quantization, Kahan-bounded risk collection
            # and sparse repair reproduce the strict row4 parent bit-exactly
            # (128-thread strided k ownership + shuffle/wave tree), so the
            # published gate/up arithmetic is unchanged; only the kernel
            # executes through the iu8-WMMA datapath. The lazy registration
            # is idempotent and keeps the route resolvable on every path
            # (the gfx1151 alias pass may run before any harness-side
            # registration).
            from hipengine.kernels.hip_gfx1100.quant.gguf_q5_k_q8_1_selected_prefill import (
                register_gguf_q5_k_q8_1_selected_prefill_kernels as _register_q5k_iu8,
            )
            try:
                _register_q5k_iu8(replace=False)
            except Exception:
                pass  # already registered by the import-time or probe path
            risk_count, risk_indices = scratch.ensure_group_risk_buffers(
                compact_rows=compact, out_features_total=2 * ffn
            )
            active_runtime.memset_async(
                risk_count.ptr, 0, DType.INT32.itemsize, stream
            )
            risk_capacity = compact * 2 * ffn
            i8_risk_gate_up = resolve(
                backend=backend, layer="moe_linear", quant="gguf_q5_k",
                variant="selected_dual_wmma_iu8_risk_prefill_bf16_bf16_out",
            )
            sparse_repair = resolve(
                backend=backend, layer="moe_linear", quant="gguf_q5_k",
                variant="selected_dual_sparse_exact_repair_bf16",
            )
            i8_risk_gate_up(
                scratch.expert_down.ptr,
                scratch.group_expert_start.ptr,
                scratch.group_wmma_expert_start.ptr,
                scratch.group_tile_expert.ptr,
                weights["expert_gate"].allocation("raw").tensor.ptr,
                weights["expert_up"].allocation("raw").tensor.ptr,
                scratch.group_gate_up.ptr,
                risk_count.ptr,
                risk_indices.ptr,
                risk_capacity,
                _qwen4_exp_q5_k_iu8_risk_multiplier(),
                compact,
                hidden,
                ffn,
                ffn,
                experts,
                wmma_total_rows,
                stream=stream,
                runtime=active_runtime,
            )
            sparse_repair(
                scratch.expert_down.ptr,
                scratch.group_expert_start.ptr,
                weights["expert_gate"].allocation("raw").tensor.ptr,
                weights["expert_up"].allocation("raw").tensor.ptr,
                scratch.group_gate_up.ptr,
                risk_count.ptr,
                risk_indices.ptr,
                risk_capacity,
                compact,
                hidden,
                ffn,
                ffn,
                experts,
                stream=stream,
                runtime=active_runtime,
            )
            silu_mul_dual_out_bf16(
                scratch.group_gate_up.ptr,
                scratch.expert_intermediate.ptr,
                rows=compact,
                features=ffn,
                stream=stream,
                runtime=active_runtime,
            )
        else:
            for slot, output in (
                ("expert_gate", scratch.expert_gate),
                ("expert_up", scratch.expert_up),
            ):
                gguf_q5_k_selected_wmma_prefill_compact_bf16_bf16_out(
                    scratch.expert_down.ptr,
                    scratch.group_expert_start.ptr,
                    scratch.group_wmma_expert_start.ptr,
                    scratch.group_tile_expert.ptr,
                    weights[slot].allocation("raw").tensor.ptr,
                    output.ptr,
                    compact,
                    hidden,
                    ffn,
                    experts,
                    wmma_total_rows,
                    stream=stream,
                    runtime=active_runtime,
                )
            silu_mul_separate_out_bf16(
                scratch.expert_gate.ptr,
                scratch.expert_up.ptr,
                scratch.expert_intermediate.ptr,
                compact,
                ffn,
                stream=stream,
                runtime=active_runtime,
            )
        if routed_f16:
            down_tile_rows = _routed_f16_down_tile_rows(compact, experts)
            down_capacity = _routed_f16_tile_map(
                scratch,
                compact=compact,
                experts=experts,
                tile_rows=down_tile_rows,
                stream=stream,
                runtime=active_runtime,
            )
            resolve(
                backend=backend,
                layer="moe_linear",
                quant=weights["expert_down"].spec.quant_key,
                variant=ROUTED_F16_LINEAR_VARIANT,
            )(
                scratch.expert_intermediate.ptr,
                scratch.group_expert_start.ptr,
                scratch.group_tile_expert.ptr,
                scratch.group_wmma_total.ptr,
                weights["expert_down"].allocation("raw").tensor.ptr,
                scratch.expert_down.ptr,
                ffn,
                hidden,
                tile_rows=down_tile_rows,
                tile_capacity=down_capacity,
                stream=stream,
                runtime=active_runtime,
            )
        elif weights["expert_down"].spec.quant_key == "gguf_q5_1":
            q5_wmma = _qwen4_exp_q51_wmma_down_enabled(
                production_grouped_moe, q51_wmma_down
            )
            if t16_down:
                # T16 Q5_1 expert-major down (down-leg build step 4):
                # F16 WMMA over the bit-lossless Q5_1 tiles with the
                # compensated accumulate. Changes arithmetic vs the
                # production down chain; the binding correctness authority
                # is the layer2 production gate, and the env flag
                # plus the tile-cache key keep the default path untouched.
                down_input_ptr = (
                    scratch.expert_intermediate.ptr
                    if (exact_grouped_q4_gate or production_grouped_moe)
                    else scratch.expert_gate.ptr
                )
                gguf_q5_1_t16_selected_expert_major_wmma_comp_bf16_bf16_out(
                    down_input_ptr,
                    scratch.group_expert_start.ptr,
                    scratch.group_active_experts.ptr,
                    scratch.group_active_count.ptr,
                    t16_down_tiles.ptr,
                    scratch.expert_down.ptr,
                    compact,
                    ffn,
                    hidden,
                    experts,
                    stream=stream,
                    runtime=active_runtime,
                )
            elif exact_grouped_down:
                # PF-3 production: fused single-loop logical256 Q5_1 down
                # prefill, bit-exact vs the strict owner. Production selects
                # M1 after the one-process/one-residency canonical gate;
                # strict selects the preceding expertgrid64 owner.
                q51_iu8_exact = (
                    _qwen4_exp_q51_iu8_exact_enabled()
                    and rows >= 2
                    and ffn % 128 == 0
                )
                if q51_iu8_exact:
                    # Default-off exact iu8-risk+repair Q5_1 down route:
                    # weight-exact iu8-WMMA with the Kahan risk criterion,
                    # then the sparse exact repair reproducing the pair2
                    # row-publish arithmetic.
                    tile_capacity = (
                        scratch.group_tile_expert.nbytes // DType.INT64.itemsize
                    )
                    active = min(compact, experts)
                    static_tiles = active + (compact - active) // 16
                    if static_tiles > tile_capacity:
                        raise RuntimeError(
                            "Qwen4Exp grouped MoE tile capacity is invalid"
                        )
                    qwen35_moe_wmma_tile_map(
                        scratch.group_expert_start.ptr,
                        scratch.group_wmma_expert_start.ptr,
                        scratch.group_tile_expert.ptr,
                        scratch.group_wmma_total.ptr,
                        experts,
                        tile_capacity=static_tiles,
                        stream=stream,
                        runtime=active_runtime,
                    )
                    risk_count, risk_indices = scratch.ensure_group_risk_buffers(
                        compact_rows=compact, out_features_total=hidden
                    )
                    active_runtime.memset_async(
                        risk_count.ptr, 0, DType.INT32.itemsize, stream
                    )
                    risk_capacity = compact * hidden
                    down_input_ptr = (
                        scratch.expert_intermediate.ptr
                        if exact_grouped_q4_gate
                        else scratch.expert_gate.ptr
                    )
                    q51_iu8_risk_down = resolve(
                        backend=backend, layer="moe_linear",
                        quant="gguf_q5_1",
                        variant="selected_wmma_iu8_risk_prefill_bf16_bf16_out",
                    )
                    q51_sparse_repair = resolve(
                        backend=backend, layer="moe_linear",
                        quant="gguf_q5_1",
                        variant="selected_sparse_exact_repair_row_publish_bf16",
                    )
                    q51_iu8_risk_down(
                        down_input_ptr,
                        scratch.group_expert_start.ptr,
                        scratch.group_wmma_expert_start.ptr,
                        scratch.group_tile_expert.ptr,
                        weights["expert_down"].allocation("raw").tensor.ptr,
                        scratch.expert_down.ptr,
                        risk_count.ptr,
                        risk_indices.ptr,
                        risk_capacity,
                        _qwen4_exp_q51_iu8_risk_multiplier(),
                        compact,
                        ffn,
                        hidden,
                        experts,
                        static_tiles * 16,
                        stream=stream,
                        runtime=active_runtime,
                    )
                    q51_sparse_repair(
                        down_input_ptr,
                        scratch.group_expert_start.ptr,
                        weights["expert_down"].allocation("raw").tensor.ptr,
                        scratch.expert_down.ptr,
                        risk_count.ptr,
                        risk_indices.ptr,
                        risk_capacity,
                        compact,
                        ffn,
                        hidden,
                        experts,
                        stream=stream,
                        runtime=active_runtime,
                    )
                else:
                    grouped_q5_key = _qwen4_exp_q51_down_key(
                        backend=backend,
                        quant=weights["expert_down"].spec.quant_key,
                        rows=rows, in_features=ffn,
                        q51_down_m1=q51_down_m1,
                        q51_out8=q51_out8,
                        q51_pair_prefill=q51_pair_prefill,
                        q51_fold128_prefill=q51_fold128_prefill,
                        q51_fold_pair_prefill=q51_fold_pair_prefill,
                        q51_register_cache=q51_register_cache,
                        q51_row_publish=q51_row_publish,
                    )
                    grouped_q5_down = resolve(
                        backend=grouped_q5_key.backend, layer=grouped_q5_key.layer,
                        quant=grouped_q5_key.quant, variant=grouped_q5_key.variant)
                    grouped_q5_down(
                        (
                            scratch.expert_intermediate.ptr
                            if exact_grouped_q4_gate
                            else scratch.expert_gate.ptr
                        ),
                        scratch.group_expert_start.ptr,
                        weights["expert_down"].allocation("raw").tensor.ptr,
                        scratch.expert_down.ptr,
                        compact,
                        experts,
                        ffn,
                        hidden,
                        stream=stream,
                        runtime=active_runtime,
                    )
            elif q5_wmma:
                # The F16 Q5_1 down owner needs a current 16-row WMMA tile map
                # and its host row count. Only the non-exact gate/up paths build
                # that map, so the down-only configuration (production exact
                # gate/up) would otherwise pass a stale zero; rebuild it here.
                _run_qwen4_exp_q51_wmma_down(
                    input_ptr=scratch.expert_intermediate.ptr,
                    expert_start_ptr=scratch.group_expert_start.ptr,
                    wmma_expert_start_ptr=scratch.group_wmma_expert_start.ptr,
                    tile_expert_ptr=scratch.group_tile_expert.ptr,
                    weights_ptr=weights["expert_down"].allocation("raw").tensor.ptr,
                    output_ptr=scratch.expert_down.ptr,
                    wmma_total_buffer=scratch.group_wmma_total,
                    compact_rows=compact,
                    num_experts=experts,
                    in_features=ffn,
                    out_features=hidden,
                    tile_capacity=tile_capacity,
                    stream=stream,
                    runtime=active_runtime,
                )
            else:
                qwen4_exp_q5_1_selected_grouped_prefill_compact_rowbatch8_bf16_bf16_out(
                    scratch.expert_intermediate.ptr,
                    scratch.group_expert_start.ptr,
                    weights["expert_down"].allocation("raw").tensor.ptr,
                    scratch.expert_down.ptr,
                    compact,
                    experts,
                    ffn,
                    hidden,
                    stream=stream,
                    runtime=active_runtime,
                )
        elif weights["expert_down"].spec.quant_key == "gguf_q8_0":
            # PF-1 fork (b) T0 candidate: grouped selected down, one
            # block per (expert, out_col) pair serving every lane of the pair so
            # the expert's weight row is read once. Per-output arithmetic is
            # identical to the strict per-expert selected gemv (bit-exact,
            # RED-pinned in tests/test_gpu_qwen4exp_pf1_forkb_selected_down.py);
            # kernel-level paired A/B at the p4096 compact shape measured
            # 32.42 vs 47.42 ms median (-31.6%). The incumbent strict selected
            # gemv and the legacy P1 grouped/WMMA owners stay available below.
            # Production selects the grouped owner after the exact
            # one-process/one-residency canonical gate; strict keeps the
            # per-expert selected gemv.
            if os.environ.get(
                "HIPENGINE_QWEN4_EXP_Q8_0_SELECTED_WMMA_DOWN", "0"
            ) not in {"", "0", "false", "False"}:
                # #19 R9 candidate: the Q5_1-style grouped WMMA down for the
                # Q8_0 expert-down layers (2, 4, 30, 46, 47). Same f16-WMMA
                # dequant arithmetic class as the promoted Q5_1 down route
                # (NOT bit-exact vs the row4 grouped parent); qualified by
                # the production numerical envelope. Resolved through the
                # registry so counted wrappers (state gate) can intercept
                # the route. The shared ``group_wmma_total`` buffer is also
                # written by ``qwen35_moe_group_prefix`` (the compact padded
                # row total) and only overwritten with a 16-row tile map by
                # the exact Q4/Q5 gate/up owners. A strict Q4 gate/up leaves
                # the prefix total there, so the Q8_0 owner must rebuild the
                # current map rather than read the shared buffer. The helper
                # rebuilds and validates the map for this invocation; a
                # missing/invalid map raises instead of silently falling back
                # to a different registered owner.
                tile_capacity = (
                    scratch.group_tile_expert.nbytes // DType.INT64.itemsize
                )
                q8_variant = os.environ.get(
                    "HIPENGINE_QWEN4_EXP_Q8_DOWN_VARIANT",
                    "selected_grouped_wmma_prefill_bf16_bf16_out")
                q8_wmma_down = resolve(
                    backend=backend,
                    layer="linear",
                    quant=weights["expert_down"].spec.quant_key,
                    variant=q8_variant,
                )
                q8_kwargs = {}
                if q8_variant == "selected_grouped_blockscale_guarded_prefill_bf16_bf16_out":
                    count, indices = scratch.ensure_group_risk_buffers(
                        compact_rows=compact, out_features_total=hidden)
                    q8_kwargs = dict(risk_count_ptr=count.ptr, risk_indices_ptr=indices.ptr,
                                     risk_capacity=compact * hidden)
                _run_qwen4_exp_q8_wmma_down(
                    input_ptr=scratch.expert_intermediate.ptr,
                    expert_start_ptr=scratch.group_expert_start.ptr,
                    wmma_expert_start_ptr=scratch.group_wmma_expert_start.ptr,
                    tile_expert_ptr=scratch.group_tile_expert.ptr,
                    weights_ptr=weights["expert_down"].allocation("raw").tensor.ptr,
                    output_ptr=scratch.expert_down.ptr,
                    wmma_total_buffer=scratch.group_wmma_total,
                    compact_rows=compact,
                    num_experts=experts,
                    in_features=ffn,
                    out_features=hidden,
                    tile_capacity=tile_capacity,
                    stream=stream,
                    runtime=active_runtime,
                    owner=q8_wmma_down,
                    risk_kwargs=q8_kwargs,
                )
            elif forkb_grouped_down:
                # x (expert_intermediate) and out (expert_down) are already in
                # sorted-lane order here, so the kernel serves row = sorted
                # position directly (lane_to_row=nullptr).
                grouped_key = _qwen4_exp_q8_down_row4_key(
                    KernelKey(backend, "linear", weights["expert_down"].spec.quant_key,
                              "selected_grouped_gemv_bf16_bf16_out"), rows=rows,
                    published=q8_down_row4_prefill)
                grouped_key = _qwen4_exp_q8_down_bundle_key(
                    grouped_key, rows=rows, published=q8_down_bundle_prefill)
                grouped_key = _qwen4_exp_q8_down_register_key(
                    grouped_key, rows=rows, in_features=ffn,
                    published=q8_down_register)
                grouped_q8_down = resolve(
                    backend=grouped_key.backend, layer=grouped_key.layer,
                    quant=grouped_key.quant, variant=grouped_key.variant)
                grouped_q8_down(
                    scratch.expert_intermediate.ptr,
                    scratch.group_expert_start.ptr,
                    None,
                    weights["expert_down"].allocation("raw").tensor.ptr,
                    scratch.expert_down.ptr,
                    compact,
                    compact,
                    experts,
                    ffn,
                    hidden,
                    stream=stream,
                    runtime=active_runtime,
                )
            else:
                # Strict fallback: every shipped Q8_0 down owner is off, so the
                # incumbent strict per-expert selected gemv computes the Q8_0
                # down layers (layer 2, 4, 30, 46, 47). The two legacy grouped
                # opt-ins that used to sit above this arm are retired; the
                # grouped kernel stays registered for the Gemma4 expert route
                # and the direct harnesses.
                selected_projection(
                    "expert_down",
                    scratch.expert_intermediate.ptr,
                    scratch.expert_down.ptr,
                    compact,
                    compact,
                    ffn,
                    hidden,
                    selected_ptr=scratch.group_sorted_experts.ptr,
                )
        else:
            selected_projection(
                "expert_down",
                scratch.expert_intermediate.ptr,
                scratch.expert_down.ptr,
                compact,
                compact,
                ffn,
                hidden,
                selected_ptr=scratch.group_sorted_experts.ptr,
            )
        if rows == 1 or os.environ.get("HIPENGINE_QWEN4_EXP_FUSED_COMBINE", "") != "1":
            # Grouped rows==1, or fused PF-4 lever-2 combine disabled:
            # keep the unfused sum. The process-separated timing diagnostic
            # did not establish a binding whole-model rejection.
            weighted_lanes_sum_out_bf16_f32w(
                scratch.expert_down.ptr,
                scratch.group_sorted_weights.ptr,
                scratch.group_sorted_lanes.ptr,
                scratch.group_lane_to_row.ptr,
                scratch.routed.ptr,
                rows,
                top_k,
                hidden,
                stream=stream,
                runtime=active_runtime,
            )
        # rows > 1 with HIPENGINE_QWEN4_EXP_FUSED_COMBINE=1: the fused
        # PF-4 lever-2 candidate replaces this sum and the
        # shared_gate_combine_batch tail in one kernel (see combine site
        # below). The unfused chain stays registered as the strict
        # fallback. Default is the unfused chain; the historical timing
        # artifact requires a one-process/one-residency comparison.
    else:
        gate_weight = weights["expert_gate"]
        up_weight = weights["expert_up"]
        dual_key = KernelKey(
            backend,
            "linear",
            gate_weight.spec.quant_key,
            "selected_dual_gemv_bf16_bf16_out",
        )
        fused_key = KernelKey(
            backend,
            "linear",
            gate_weight.spec.quant_key,
            "selected_dual_silu_logical128_t64_gemv_bf16_bf16_out",
        )
        dp4a_key = KernelKey(
            backend,
            "linear",
            gate_weight.spec.quant_key,
            "selected_dual_q8_1_dp4a_silu_logical128_t64_gemv_bf16_bf16_out",
        )
        dp4a_layers_raw = os.environ.get("HIPENGINE_QWEN4_EXP_Q4_DP4A64_LAYERS", "")
        dp4a_layers = {
            int(value) for value in dp4a_layers_raw.split(",") if value.strip()
        }
        gate_parts = gate_weight.spec.slot_path.split(".")
        gate_layer = (
            int(gate_parts[1])
            if len(gate_parts) >= 3 and gate_parts[0] == "layers"
            else -1
        )
        use_dp4a = bool(
            rows == 1
            and gate_weight.spec.quant_key == up_weight.spec.quant_key
            and os.environ.get("HIPENGINE_QWEN4_EXP_Q4_DP4A64", "")
            not in {"", "0", "false", "False"}
            and (not dp4a_layers or gate_layer in dp4a_layers)
            and is_registered(dp4a_key)
        )
        fused_silu = bool(
            rows == 1
            and gate_weight.spec.quant_key == up_weight.spec.quant_key
            and is_registered(fused_key)
        )
        mapped_gate_up_ready = False
        if use_dp4a:
            gguf_q4_k_quantize_bf16_q8_1(
                scratch.hidden_bf16.ptr,
                scratch.group_gate_up.ptr,
                rows,
                hidden,
                stream=stream,
                runtime=active_runtime,
            )
            # The warp-per-expert DP4A owner (one block per output column,
            # one wave per selected expert, no barriers) when the profile
            # selects it and the row count fits; same per-group integer
            # arithmetic as the logical128/t64 owner, a different reduction
            # order. History of the earlier rejection: docs/REFACTOR.md.
            warp_key = KernelKey(
                dp4a_key.backend, dp4a_key.layer, dp4a_key.quant,
                "selected_dual_q8_1_dp4a_silu_warp256_gemv_bf16_bf16_out",
            )
            if (
                compact <= 12
                and _qwen4_exp_moe_decode_warp_enabled("GATE_UP")
                and is_registered(warp_key)
            ):
                dp4a_key = warp_key
            resolve(
                backend=dp4a_key.backend,
                layer=dp4a_key.layer,
                quant=dp4a_key.quant,
                variant=dp4a_key.variant,
            )(
                scratch.group_gate_up.ptr,
                scratch.selected.ptr,
                gate_weight.allocation("raw").tensor.ptr,
                up_weight.allocation("raw").tensor.ptr,
                scratch.expert_intermediate.ptr,
                rows,
                compact,
                experts,
                hidden,
                ffn,
                stream=stream,
                runtime=active_runtime,
            )
        elif fused_silu:
            resolve(
                backend=fused_key.backend,
                layer=fused_key.layer,
                quant=fused_key.quant,
                variant=fused_key.variant,
            )(
                scratch.hidden_bf16.ptr,
                scratch.selected.ptr,
                gate_weight.allocation("raw").tensor.ptr,
                up_weight.allocation("raw").tensor.ptr,
                scratch.expert_intermediate.ptr,
                rows,
                compact,
                experts,
                hidden,
                ffn,
                stream=stream,
                runtime=active_runtime,
            )
        elif (
            rows == 1
            and gate_weight.spec.quant_key == up_weight.spec.quant_key
            and is_registered(dual_key)
        ):
            resolve(
                backend=dual_key.backend,
                layer=dual_key.layer,
                quant=dual_key.quant,
                variant=dual_key.variant,
            )(
                scratch.hidden_bf16.ptr,
                scratch.selected.ptr,
                gate_weight.allocation("raw").tensor.ptr,
                up_weight.allocation("raw").tensor.ptr,
                scratch.expert_gate.ptr,
                scratch.expert_up.ptr,
                rows,
                compact,
                experts,
                hidden,
                ffn,
                stream=stream,
                runtime=active_runtime,
            )
        else:
            row4 = qwen4_exp_grouped_row4_prefill_selected(
                weights, rows=rows, backend=backend, published=q5k_row4_prefill)
            if row4:
                # Map only: projections publish original token-major lanes,
                # leaving SiLU/down/ordered combine ownership unchanged.
                active_runtime.memset_async(
                    scratch.group_counts.ptr, 0, scratch.group_counts.nbytes, stream)
                qwen35_moe_group_count(
                    scratch.selected.ptr, scratch.group_counts.ptr, compact,
                    experts, stream=stream, runtime=active_runtime)
                qwen35_moe_group_prefix(
                    scratch.group_counts.ptr, scratch.group_padded_counts.ptr,
                    scratch.group_expert_start.ptr, scratch.group_wmma_total.ptr,
                    experts, 1, stream=stream, runtime=active_runtime)
                active_runtime.memset_async(
                    scratch.group_scatter_offsets.ptr, 0,
                    scratch.group_scatter_offsets.nbytes, stream)
                qwen35_moe_group_scatter(
                    scratch.selected.ptr, scratch.routing.ptr,
                    scratch.group_expert_start.ptr, scratch.group_scatter_offsets.ptr,
                    scratch.group_sorted_lanes.ptr, scratch.group_sorted_experts.ptr,
                    scratch.group_sorted_weights.ptr, compact, experts,
                    stream=stream, runtime=active_runtime)
                mapped_gate_up_ready = True
                for name, output in (
                    ("expert_gate", scratch.expert_gate),
                    ("expert_up", scratch.expert_up),
                ):
                    weight = weights[name]
                    resolve(
                        backend=backend, layer="linear", quant=weight.spec.quant_key,
                        variant="selected_grouped_row4_gemv_bf16_bf16_out",
                    )(
                        scratch.hidden_bf16.ptr, scratch.group_expert_start.ptr,
                        scratch.group_sorted_lanes.ptr, weight.allocation("raw").tensor.ptr,
                        output.ptr, rows, compact, experts, hidden, ffn,
                        stream=stream, runtime=active_runtime,
                    )
            else:
                selected_projection(
                    "expert_gate", scratch.hidden_bf16.ptr, scratch.expert_gate.ptr,
                    rows, compact, hidden, ffn,
                )
                selected_projection(
                    "expert_up", scratch.hidden_bf16.ptr, scratch.expert_up.ptr,
                    rows, compact, hidden, ffn,
                )
        if not fused_silu and not use_dp4a:
            silu_mul_separate_out_bf16(
                scratch.expert_gate.ptr,
                scratch.expert_up.ptr,
                scratch.expert_intermediate.ptr,
                compact,
                ffn,
                stream=stream,
                runtime=active_runtime,
            )
        down_weight = weights["expert_down"]
        fused_down_key = KernelKey(
            backend,
            "linear",
            down_weight.spec.quant_key,
            "selected_weighted_sum_logical256_t64_bf16_bf16_out",
        )
        fused_down = bool(rows == 1 and is_registered(fused_down_key))
        warp_down_key = KernelKey(
            backend, "linear", down_weight.spec.quant_key,
            "selected_weighted_sum_warp256_fast_bf16_bf16_out",
        )
        if (
            fused_down
            and compact <= 12
            and _qwen4_exp_moe_decode_warp_enabled("DOWN")
            and is_registered(warp_down_key)
        ):
            # Vectorized half-block loads, one wave per selected expert; the
            # intra-dot reduction order differs from the logical256/t64 owner.
            fused_down_key = warp_down_key
        if fused_down:
            resolve(
                backend=fused_down_key.backend,
                layer=fused_down_key.layer,
                quant=fused_down_key.quant,
                variant=fused_down_key.variant,
            )(
                scratch.expert_intermediate.ptr,
                scratch.selected.ptr,
                down_weight.allocation("raw").tensor.ptr,
                scratch.routing.ptr,
                scratch.routed.ptr,
                compact,
                experts,
                ffn,
                hidden,
                stream=stream,
                runtime=active_runtime,
            )
        else:
            mapped_key = _qwen4_exp_mapped_down_key(
                backend=backend,quant=down_weight.spec.quant_key,rows=rows,
                map_ready=mapped_gate_up_ready,published=q8_mapped_down)
            if mapped_key is not None:
                mapped_key = _qwen4_exp_q8_down_register_key(
                    mapped_key, rows=rows, in_features=ffn,
                    published=q8_down_register)
                resolve(backend=mapped_key.backend,layer=mapped_key.layer,
                        quant=mapped_key.quant,variant=mapped_key.variant)(
                    scratch.expert_intermediate.ptr,scratch.group_expert_start.ptr,
                    scratch.group_sorted_lanes.ptr,down_weight.allocation("raw").tensor.ptr,
                    scratch.expert_down.ptr,compact,compact,experts,ffn,hidden,
                    stream=stream,runtime=active_runtime)
            else:
                selected_projection(
                    "expert_down", scratch.expert_intermediate.ptr,
                    scratch.expert_down.ptr, compact, compact, ffn, hidden,
                )
            if rows == 1:
                weighted_sum_out_bf16_f32w(
                    scratch.expert_down.ptr,
                    scratch.routing.ptr,
                    scratch.routed.ptr,
                    compact,
                    hidden,
                    stream=stream,
                    runtime=active_runtime,
                )
            else:
                weighted_sum_batch_out_bf16_f32w(
                    scratch.expert_down.ptr,
                    scratch.routing.ptr,
                    scratch.routed.ptr,
                    rows,
                    top_k,
                    hidden,
                    stream=stream,
                    runtime=active_runtime,
                )
    _launch_qwen4_exp_dense_pair(
        weights["shared_gate"], weights["shared_up"], mixed_ptr,
        scratch.shared_gate.ptr, scratch.shared_up.ptr, rows, hidden, ffn, ffn,
        stream=stream, runtime=active_runtime,
    )
    qwen4_exp_silu_mul_f32(
        scratch.shared_gate.ptr,
        scratch.shared_up.ptr,
        scratch.shared_intermediate.ptr,
        rows * ffn,
        stream=stream,
        runtime=active_runtime,
    )
    launch_gguf_linear(
        weights["shared_down"], scratch.shared_intermediate.ptr, scratch.shared_down.ptr,
        rows, ffn, hidden,
        activation_dtype=GGUF_ACTIVATION_F32,
        output_dtype=GGUF_OUTPUT_F32,
        stream=stream, runtime=active_runtime,
    )
    launch_gguf_linear(
        weights["shared_gate_weight"], mixed_ptr, scratch.shared_gate_logits.ptr,
        rows, hidden, 1,
        activation_dtype=GGUF_ACTIVATION_F32,
        output_dtype=GGUF_OUTPUT_F32,
        stream=stream, runtime=active_runtime,
    )
    fused_lane_combine = grouped_prefill and os.environ.get(
        "HIPENGINE_QWEN4_EXP_FUSED_COMBINE", ""
    ) == "1"
    if not fused_lane_combine:
        # The combine rounds the FP32 shared output to BF16 itself, which is
        # bit-identical to the separate cast followed by the BF16 combine.
        shared_gate_combine_f32_shared_out_bf16(
            scratch.routed.ptr,
            scratch.shared_down.ptr,
            scratch.shared_gate_logits.ptr,
            scratch.output.ptr,
            rows,
            hidden,
            stream=stream,
            runtime=active_runtime,
        )
        return Qwen4ExpMoEDeviceResult(scratch.output, scratch.selected, scratch.routing)
    f32_to_bf16(
        scratch.shared_down.ptr,
        scratch.shared_down_bf16.ptr,
        rows * hidden,
        stream=stream,
        runtime=active_runtime,
    )
    if fused_lane_combine:
        # PF-4 lever 2 T0: fused routed-sum + gated shared combine (bit-exact
        # vs the unfused weighted_lanes_sum -> shared_gate_combine chain;
        # kernel-level A/B 455e176ab). Opt-in only: the historical whole-model
        # timing was process-separated and remains provisional. The unfused
        # chain stays the default and registered strict fallback.
        weighted_lanes_sum_shared_gate_combine_batch_out_bf16_f32w(
            scratch.expert_down.ptr,
            scratch.group_sorted_weights.ptr,
            scratch.group_sorted_lanes.ptr,
            scratch.group_lane_to_row.ptr,
            scratch.shared_down_bf16.ptr,
            scratch.shared_gate_logits.ptr,
            scratch.output.ptr,
            rows,
            top_k,
            hidden,
            stream=stream,
            runtime=active_runtime,
        )
    return Qwen4ExpMoEDeviceResult(scratch.output, scratch.selected, scratch.routing)


def run_qwen4_exp_dense_qsa_layer(
    residual_ptr: int,
    weights: Qwen4ExpQSALayerDeviceWeights,
    *,
    q4_bundle_prefill: bool = False,
    q5k_row4_prefill: bool = False,
    attention_state: Qwen4ExpDenseAttentionState,
    scratch: Qwen4ExpQSALayerScratch,
    position: int,
    rows: int,
    branches: int,
    hidden: int,
    low_rank: int,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    theta: float,
    ffn: int,
    experts: int,
    top_k: int,
    index_state: Qwen4ExpQSAIndexDeviceState | None = None,
    index_heads: int = 0,
    index_dim: int = 0,
    index_rotary_dim: int = 0,
    rope_positions_ptr: int | None = None,
    position_prepared: bool = False,
    device_position_owned: bool = False,
    attention_context_limit: int | None = None,
    stream: int = 0,
    runtime: HipRuntime | None = None,
    moe_graph_cache: MoeGraphCache | None = None,
    moe_graph_key: object | None = None,
    cache_only: bool = False,
    qsa_ordered_decode_v2: bool = False,
    qsa_wave32: bool = True,
    q51_down_m1: bool = False,
    q51_pair_prefill: bool = False,
    q51_fold128_prefill: bool = False,
    q51_fold_pair_prefill: bool = False,
    q51_register_cache: bool = False,
    q51_row_publish: bool = False,
    q51_exact_grouped_down: bool = True,
    q51_wmma_down: bool = False,
    q51_out8: bool = True,
    q4_exact_grouped_gate: bool = True,
    q4_exact_grouped_all: bool = True,
    q4_out4: bool = True,
    q4_tile_m: int | None = None,
    q4_tile_n: int | None = None,
    forkb_grouped_down: bool = False,
    q8_down_row4_prefill: bool = False,
    q8_down_bundle_prefill: bool = False,
    q8_down_register: bool = False,
    q8_mapped_down: bool = False,
) -> DeviceBuffer | None:
    """Execute a QSA+MoE layer, or only its independent cache-state writes.

    ``qsa_ordered_decode_v2`` is the owning runner's plan-published ordered-v2
    sparse-decode decision, forwarded to the token mixer (see
    :func:`qsa_ordered_decode_v2`); it selects nothing on a cache-only call and
    only inside the ordered route's own guards.

    ``qsa_wave32`` is the owning runner's H128 sparse-attention decision,
    forwarded to the token mixer (see :func:`qsa_wave32`); it selects nothing on
    a cache-only call and only for the head_dim 128 geometry.

    The six expert-down decisions (``q51_down_m1``, ``q51_fold128_prefill``,
    ``q51_fold_pair_prefill``, ``forkb_grouped_down``, ``q8_down_row4_prefill``,
    ``q8_down_bundle_prefill``) and the register/row-publish/mapped siblings are
    the owning runner's plan-published MoE down selections, forwarded to the MoE
    dispatch unchanged; ``q51_exact_grouped_down`` is the exact grouped Q5_1
    down chain's own gate and is forwarded the same way. ``q51_out8`` is that
    chain's non-expertgrid fallback rung (see :func:`q51_out8`): it picks the
    out8 rowbatch8 owner over the output1 base and selects nothing on a
    cache-only call. ``q4_out4`` is the Q4 gate/up ladder's equivalent rung (see
    :func:`q4_out4`): it picks the out4 rowbatch8 owner over the strict base and
    likewise selects nothing on a cache-only call.
    """

    if scratch.closed:
        raise RuntimeError("Qwen4Exp QSA layer scratch is closed")
    active_runtime = runtime or scratch.runtime
    if active_runtime is not scratch.runtime:
        raise ValueError("runtime must match QSA layer scratch owner")
    attention_read = run_qwen4_exp_gr_read(
        residual_ptr,
        weights.attention_gr.norm_weight_ptr,
        weights.attention_gr.down,
        weights.attention_gr.up,
        weights.attention_gr.inject,
        scratch.attention_gr,
        rows=rows, branches=branches, hidden=hidden, low_rank=low_rank,
        stream=stream, runtime=active_runtime,
    )
    mixer = run_qwen4_exp_dense_qsa_token_mixer(
        attention_read.mixed.ptr,
        weights.mixer,
        attention_state=attention_state,
        scratch=scratch.qsa,
        position=position,
        rows=rows, hidden=hidden, query_heads=query_heads, kv_heads=kv_heads,
        head_dim=head_dim, rotary_dim=rotary_dim, theta=theta,
        index_state=index_state, index_heads=index_heads, index_dim=index_dim,
        index_rotary_dim=index_rotary_dim,
        rope_positions_ptr=rope_positions_ptr,
        position_prepared=position_prepared,
        device_position_owned=device_position_owned,
        attention_context_limit=attention_context_limit,
        cache_only=cache_only,
        qsa_ordered_decode_v2=qsa_ordered_decode_v2,
        qsa_wave32=qsa_wave32,
        stream=stream, runtime=active_runtime,
    )
    if cache_only:
        return None
    pending_write = _qwen4_exp_gr_pending_write(
        rows, residual_ptr, mixer.ptr, attention_read.inject_logits.ptr)
    if pending_write is None:
        qwen4_exp_gr_write_bf16_f32(
            residual_ptr, mixer.ptr, attention_read.inject_logits.ptr,
            scratch.after_attention.ptr, rows, branches, hidden,
            stream=stream, runtime=active_runtime,
        )
    ffn_read = run_qwen4_exp_gr_read(
        scratch.after_attention.ptr,
        weights.ffn_gr.norm_weight_ptr,
        weights.ffn_gr.down,
        weights.ffn_gr.up,
        weights.ffn_gr.inject,
        scratch.ffn_gr,
        rows=rows, branches=branches, hidden=hidden, low_rank=low_rank,
        stream=stream, runtime=active_runtime,
        pending_write=pending_write,
    )
    moe = run_qwen4_exp_moe(
        ffn_read.mixed.ptr, weights.moe, scratch=scratch.moe,
        q4_bundle_prefill=q4_bundle_prefill,
        q5k_row4_prefill=q5k_row4_prefill,
        rows=rows, hidden=hidden, ffn=ffn, experts=experts, top_k=top_k,
        stream=stream, runtime=active_runtime,
        graph_cache=moe_graph_cache, graph_key=moe_graph_key,
        q51_down_m1=q51_down_m1,
        q51_pair_prefill=q51_pair_prefill,
        q51_fold128_prefill=q51_fold128_prefill,
        q51_fold_pair_prefill=q51_fold_pair_prefill,
        q51_register_cache=q51_register_cache,
        q51_row_publish=q51_row_publish,
        q51_exact_grouped_down=q51_exact_grouped_down,
        q51_wmma_down=q51_wmma_down,
        q51_out8=q51_out8,
        q4_exact_grouped_gate=q4_exact_grouped_gate,
        q4_exact_grouped_all=q4_exact_grouped_all,
        q4_out4=q4_out4,
        q4_tile_m=q4_tile_m,
        q4_tile_n=q4_tile_n,
        forkb_grouped_down=forkb_grouped_down,
        q8_down_row4_prefill=q8_down_row4_prefill,
        q8_down_bundle_prefill=q8_down_bundle_prefill,
        q8_down_register=q8_down_register,
        q8_mapped_down=q8_mapped_down,
    )
    qwen4_exp_gr_write_bf16_bf16(
        scratch.after_attention.ptr,
        moe.output.ptr,
        ffn_read.inject_logits.ptr, scratch.output.ptr,
        rows, branches, hidden, stream=stream, runtime=active_runtime,
    )
    return scratch.output


def run_qwen4_exp_qsa_prefill_layer(
    residual_ptr: int,
    weights: Qwen4ExpQSALayerDeviceWeights,
    *,
    q4_bundle_prefill: bool = False,
    q5k_row4_prefill: bool = False,
    attention_state: Qwen4ExpDenseAttentionState,
    index_state: Qwen4ExpQSAIndexDeviceState,
    scratch: Qwen4ExpQSALayerScratch,
    metadata: Qwen4ExpQSAPrefillMetadata,
    start_position: int,
    rows: int,
    branches: int,
    hidden: int,
    low_rank: int,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    theta: float,
    index_heads: int,
    index_dim: int,
    index_rotary_dim: int,
    ffn: int,
    experts: int,
    top_k: int,
    rope_positions_ptr: int | None = None,
    head_mode: str | None = _QSA_HEAD_PARENT,
    h256_mode: str = "parent",
    batched_selection: bool = True,
    qsa_wave32: bool = True,
    stream: int = 0,
    runtime: HipRuntime | None = None,
    pending_write: tuple[int, int, int, bool] | None = None,
    defer_write: list | None = None,
    q51_down_m1: bool = False,
    q51_pair_prefill: bool = False,
    q51_fold128_prefill: bool = False,
    q51_fold_pair_prefill: bool = False,
    q51_register_cache: bool = False,
    q51_row_publish: bool = False,
    q51_exact_grouped_down: bool = True,
    q51_wmma_down: bool = False,
    q51_out8: bool = True,
    q4_exact_grouped_gate: bool = True,
    q4_exact_grouped_all: bool = True,
    q4_out4: bool = True,
    q4_tile_m: int | None = None,
    q4_tile_n: int | None = None,
    forkb_grouped_down: bool = False,
    q8_down_row4_prefill: bool = False,
    q8_down_bundle_prefill: bool = False,
    q8_down_register: bool = False,
    q8_mapped_down: bool = False,
) -> DeviceBuffer:
    """Execute one complete Qwen4Exp QSA+MoE prompt chunk.

    ``head_mode`` is the owning runner's H256 prefill head grouping decision for
    the scalar rows owner and is forwarded unchanged to the token mixer, which
    resolves that leaf but launches it only when WMMA attention does not own the
    chunk.

    ``batched_selection`` is the owning runner's QSA chunk selection decision,
    forwarded unchanged to the token mixer (see :func:`qsa_batched_selection`).

    ``qsa_wave32`` is the owning runner's H128 sparse-attention decision,
    forwarded unchanged to the token mixer (see :func:`qsa_wave32`); the mixer
    applies it only as the fallback below the H256 wave-prefill owner and only
    for the head_dim 128 geometry.

    ``q51_down_m1`` is the owning runner's plan-published Q5_1 down M1 decision,
    forwarded to the MoE down dispatch; it only selects inside the exact grouped
    Q5_1 down chain's small-chunk scope, below the pair2/fold ladder.

    ``q51_fold128_prefill`` is the owning runner's plan-published Q5_1 fold128
    decision, forwarded to the MoE down dispatch; it only selects inside the
    grouped pair2 chain's rows>=64 scope.

    ``q51_fold_pair_prefill`` is the owning runner's plan-published Q5_1
    folded-pair decision, forwarded to the same dispatch; it only selects
    inside the fold128 ladder's rows>=512 scope.

    ``forkb_grouped_down`` is the owning runner's plan-published PF-1 fork-b
    decision, forwarded to the same dispatch; it only selects the sorted chain's
    Q8_0 down owner when no Q8_0 WMMA down owner is bound.

    ``q8_down_row4_prefill`` is the owning runner's plan-published Q8_0 down
    row4 decision, forwarded to the same dispatch; it only selects inside the
    sorted grouped chain's rows>=512 scope.

    ``q8_down_bundle_prefill`` is the owning runner's plan-published Q8_0 down
    bundle decision, forwarded to the same dispatch; it only selects inside the
    row4 ladder's rows>=512 scope.

    ``q51_register_cache`` is the owning runner's plan-published Q5_1
    register-cache fallback decision, forwarded to the MoE down dispatch; it
    only selects inside the grouped pair2 chain's rows>=512 K640 scope.

    ``q51_row_publish`` is the owning runner's plan-published Q5_1 row-publish
    decision, forwarded to the MoE down dispatch; it only selects inside the
    grouped pair2 chain's rows>=512 K640 scope.

    ``q8_down_register`` is the owning runner's plan-published Q8_0 down
    register decision, forwarded to the same dispatch for the sorted and
    mapped paths.

    ``q8_mapped_down`` is the owning runner's plan-published Q8_0 token-major
    mapped down decision, forwarded to the same dispatch.

    ``pending_write`` is the previous layer's deferred residual write (its
    output is ``residual_ptr``). With ``defer_write`` this layer's own final
    write may be appended there instead of launched; the returned buffer is
    then written by whoever consumes the pending write.
    """

    if scratch.closed:
        raise RuntimeError("Qwen4Exp QSA prefill layer scratch is closed")
    active_runtime = runtime or scratch.runtime
    if active_runtime is not scratch.runtime:
        raise ValueError("runtime must match QSA prefill layer scratch")
    attention_read = run_qwen4_exp_gr_read(
        residual_ptr,
        weights.attention_gr.norm_weight_ptr,
        weights.attention_gr.down,
        weights.attention_gr.up,
        weights.attention_gr.inject,
        scratch.attention_gr,
        rows=rows,
        branches=branches,
        hidden=hidden,
        low_rank=low_rank,
        stream=stream,
        runtime=active_runtime,
        pending_write=pending_write,
    )
    mixer = run_qwen4_exp_qsa_prefill_token_mixer(
        attention_read.mixed.ptr,
        weights.mixer,
        attention_state=attention_state,
        index_state=index_state,
        scratch=scratch.qsa,
        metadata=metadata,
        start_position=start_position,
        rows=rows,
        hidden=hidden,
        query_heads=query_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        theta=theta,
        index_heads=index_heads,
        index_dim=index_dim,
        index_rotary_dim=index_rotary_dim,
        rope_positions_ptr=rope_positions_ptr,
        head_mode=head_mode,
        h256_mode=h256_mode,
        batched_selection=batched_selection,
        qsa_wave32=qsa_wave32,
        stream=stream,
        runtime=active_runtime,
    )
    pending_write = _qwen4_exp_gr_pending_write(
        rows, residual_ptr, mixer.ptr, attention_read.inject_logits.ptr)
    if pending_write is None:
        qwen4_exp_gr_write_bf16_f32(
            residual_ptr,
            mixer.ptr,
            attention_read.inject_logits.ptr,
            scratch.after_attention.ptr,
            rows,
            branches,
            hidden,
            stream=stream,
            runtime=active_runtime,
        )
    ffn_read = run_qwen4_exp_gr_read(
        scratch.after_attention.ptr,
        weights.ffn_gr.norm_weight_ptr,
        weights.ffn_gr.down,
        weights.ffn_gr.up,
        weights.ffn_gr.inject,
        scratch.ffn_gr,
        rows=rows,
        branches=branches,
        hidden=hidden,
        low_rank=low_rank,
        stream=stream,
        runtime=active_runtime,
        pending_write=pending_write,
    )
    moe = run_qwen4_exp_moe(
        ffn_read.mixed.ptr,
        weights.moe,
        q4_bundle_prefill=q4_bundle_prefill,
        q5k_row4_prefill=q5k_row4_prefill,
        scratch=scratch.moe,
        rows=rows,
        hidden=hidden,
        ffn=ffn,
        experts=experts,
        top_k=top_k,
        stream=stream,
        runtime=active_runtime,
        q51_down_m1=q51_down_m1,
        q51_pair_prefill=q51_pair_prefill,
        q51_fold128_prefill=q51_fold128_prefill,
        q51_fold_pair_prefill=q51_fold_pair_prefill,
        q51_register_cache=q51_register_cache,
        q51_row_publish=q51_row_publish,
        q51_exact_grouped_down=q51_exact_grouped_down,
        q51_wmma_down=q51_wmma_down,
        q51_out8=q51_out8,
        q4_exact_grouped_gate=q4_exact_grouped_gate,
        q4_exact_grouped_all=q4_exact_grouped_all,
        q4_out4=q4_out4,
        q4_tile_m=q4_tile_m,
        q4_tile_n=q4_tile_n,
        forkb_grouped_down=forkb_grouped_down,
        q8_down_row4_prefill=q8_down_row4_prefill,
        q8_down_bundle_prefill=q8_down_bundle_prefill,
        q8_down_register=q8_down_register,
        q8_mapped_down=q8_mapped_down,
    )
    if not _qwen4_exp_gr_defer_write(
        defer_write, rows, scratch.after_attention.ptr, moe.output.ptr,
        ffn_read.inject_logits.ptr,
    ):
        qwen4_exp_gr_write_bf16_bf16(
            scratch.after_attention.ptr,
            moe.output.ptr,
            ffn_read.inject_logits.ptr,
            scratch.output.ptr,
            rows,
            branches,
            hidden,
            stream=stream,
            runtime=active_runtime,
        )
    return scratch.output


def run_qwen4_exp_gdn_layer(
    residual_ptr: int,
    weights: Qwen4ExpGDNLayerDeviceWeights,
    *,
    q4_bundle_prefill: bool = False,
    q5k_row4_prefill: bool = False,
    conv_state_ptr: int,
    recurrent_state_ptr: int,
    scratch: Qwen4ExpGDNLayerScratch,
    rows: int,
    branches: int,
    hidden: int,
    low_rank: int,
    num_k_heads: int,
    num_v_heads: int,
    head_dim: int,
    conv_kernel: int,
    ffn: int,
    experts: int,
    top_k: int,
    stream: int = 0,
    runtime: HipRuntime | None = None,
    moe_graph_cache: MoeGraphCache | None = None,
    moe_graph_key: object | None = None,
    layer_graph_cache: MoeGraphCache | None = None,
    layer_graph_key: object | None = None,
    pending_write: tuple[int, int, int, bool] | None = None,
    defer_write: list | None = None,
    prefill_variant: str | None = None,
    gdn_tile16_prefill: bool = True,
    gdn_peer_prefill_layers: str | None = None,
    gdn_exact_bulk_conv: bool = True,
    q51_down_m1: bool = False,
    q51_pair_prefill: bool = False,
    q51_fold128_prefill: bool = False,
    q51_fold_pair_prefill: bool = False,
    q51_register_cache: bool = False,
    q51_row_publish: bool = False,
    q51_exact_grouped_down: bool = True,
    q51_wmma_down: bool = False,
    q51_out8: bool = True,
    q4_exact_grouped_gate: bool = True,
    q4_exact_grouped_all: bool = True,
    q4_out4: bool = True,
    q4_tile_m: int | None = None,
    q4_tile_n: int | None = None,
    forkb_grouped_down: bool = False,
    q8_down_row4_prefill: bool = False,
    q8_down_bundle_prefill: bool = False,
    q8_down_register: bool = False,
    q8_mapped_down: bool = False,
) -> DeviceBuffer:
    """Execute one complete strict Qwen4Exp GDN+MoE physical layer.

    ``prefill_variant`` is the owning runner's GDN serial-prefill owner (see
    :func:`gdn_prefill_variant`): it selects the narrow/wide wave-norm leaf for
    multi-row chunks and leaves the register parent and decode untouched.

    ``gdn_tile16_prefill`` is the owning runner's tile-16 prefill decision (see
    :func:`gdn_tile16_prefill`), forwarded to the token mixer.

    ``gdn_peer_prefill_layers`` is the owning runner's peer prefill scope (see
    :func:`gdn_peer_prefill`), forwarded to the token mixer.

    ``gdn_exact_bulk_conv`` is the owning runner's conv prefill decision (see
    :func:`gdn_exact_bulk_conv`), forwarded to the token mixer.

    ``q51_down_m1`` is the owning runner's plan-published Q5_1 down M1
    decision, forwarded to the MoE down dispatch.

    ``q51_fold128_prefill`` is the owning runner's plan-published Q5_1 fold128
    decision, forwarded to the MoE down dispatch.

    ``q51_fold_pair_prefill`` is the owning runner's plan-published Q5_1
    folded-pair decision, forwarded to the same dispatch.

    ``forkb_grouped_down`` is the owning runner's plan-published PF-1 fork-b
    decision, forwarded to the same dispatch.

    ``q8_down_row4_prefill`` is the owning runner's plan-published Q8_0 down
    row4 decision, forwarded to the same dispatch.

    ``q8_down_bundle_prefill`` is the owning runner's plan-published Q8_0 down
    bundle decision, forwarded to the same dispatch.

    ``q51_register_cache`` is the owning runner's plan-published Q5_1
    register-cache fallback decision, forwarded to the MoE down dispatch.

    ``q51_row_publish`` is the owning runner's plan-published Q5_1 row-publish
    decision, forwarded to the MoE down dispatch.

    ``q8_down_register`` is the owning runner's plan-published Q8_0 down
    register decision, forwarded to the same dispatch.

    ``q8_mapped_down`` is the owning runner's plan-published Q8_0 token-major
    mapped down decision, forwarded to the same dispatch.

    ``pending_write`` / ``defer_write`` as for the QSA prefill layer; only
    multi-row (uncaptured) chunks use them.
    """

    if scratch.closed:
        raise RuntimeError("Qwen4Exp GDN layer scratch is closed")
    active_runtime = runtime or scratch.runtime
    if active_runtime is not scratch.runtime:
        raise ValueError("runtime must match the GDN layer scratch owner")
    layer_graph = (
        layer_graph_cache
        if (layer_graph_cache is not None and rows == 1 and layer_graph_key is not None)
        else None
    )
    if layer_graph is not None and pending_write is not None:
        raise ValueError("a deferred residual write cannot enter a captured layer")
    if layer_graph is not None and (
        weights.attention_gr.inject is None or weights.ffn_gr.inject is None
    ):
        # Inject-free GR reads memset inject_logits; keep that rare layout on
        # the plain path until it is needed.
        layer_graph = None
    if layer_graph is not None:
        # #20 R10: graph-capture the three stateless segments around the
        # stateful GDN conv/recurrent mixer (the retired whole-layer decode
        # graph machinery died on the mixer's in-place state hazard; these
        # segments only recompute from fixed scratch pointers, so they replay
        # bit-exact and the self-validating cache rejects any unit that does
        # not).
        def _segment_gr_read_attn(graph_stream: int) -> None:
            run_qwen4_exp_gr_read(
                residual_ptr,
                weights.attention_gr.norm_weight_ptr,
                weights.attention_gr.down,
                weights.attention_gr.up,
                weights.attention_gr.inject,
                scratch.attention_gr,
                rows=rows, branches=branches, hidden=hidden, low_rank=low_rank,
                stream=graph_stream, runtime=active_runtime,
            )

        layer_graph.run(
            (layer_graph_key, "gr_read_attn"),
            eager=_segment_gr_read_attn,
            out_ptr=scratch.attention_gr.mixed.ptr,
            out_nbytes=scratch.attention_gr.mixed.nbytes,
            out_regions=(
                (
                    scratch.attention_gr.inject_logits.ptr,
                    scratch.attention_gr.inject_logits.nbytes,
                ),
            ),
            stream=stream,
        )
        attention_read = Qwen4ExpGRReadDeviceResult(
            normalized=scratch.attention_gr.normalized,
            gate=scratch.attention_gr.gate,
            mixed=scratch.attention_gr.mixed,
            inject_logits=scratch.attention_gr.inject_logits,
        )
    else:
        attention_read = run_qwen4_exp_gr_read(
            residual_ptr,
            weights.attention_gr.norm_weight_ptr,
            weights.attention_gr.down,
            weights.attention_gr.up,
            weights.attention_gr.inject,
            scratch.attention_gr,
            rows=rows, branches=branches, hidden=hidden, low_rank=low_rank,
            stream=stream, runtime=active_runtime,
            pending_write=pending_write,
        )
    mixer_output = run_qwen4_exp_gdn_token_mixer(
        attention_read.mixed.ptr,
        weights.mixer.projections,
        conv_weight_ptr=weights.mixer.conv_weight_ptr,
        dt_bias_ptr=weights.mixer.dt_bias_ptr,
        a_ptr=weights.mixer.a_ptr,
        norm_weight_ptr=weights.mixer.norm_weight_ptr,
        conv_state_ptr=conv_state_ptr,
        recurrent_state_ptr=recurrent_state_ptr,
        scratch=scratch.gdn,
        rows=rows, hidden=hidden,
        num_k_heads=num_k_heads, num_v_heads=num_v_heads, head_dim=head_dim,
        conv_kernel=conv_kernel, stream=stream, runtime=active_runtime,
        prefill_variant=prefill_variant,
        gdn_tile16_prefill=gdn_tile16_prefill,
        gdn_peer_prefill_layers=gdn_peer_prefill_layers,
        gdn_exact_bulk_conv=gdn_exact_bulk_conv,
    )
    if layer_graph is not None:
        def _segment_gr_write_ffn_read(graph_stream: int) -> None:
            qwen4_exp_gr_write_bf16_f32(
                residual_ptr, mixer_output.ptr, attention_read.inject_logits.ptr,
                scratch.after_attention.ptr, rows, branches, hidden,
                stream=graph_stream, runtime=active_runtime,
            )
            run_qwen4_exp_gr_read(
                scratch.after_attention.ptr,
                weights.ffn_gr.norm_weight_ptr,
                weights.ffn_gr.down,
                weights.ffn_gr.up,
                weights.ffn_gr.inject,
                scratch.ffn_gr,
                rows=rows, branches=branches, hidden=hidden, low_rank=low_rank,
                stream=graph_stream, runtime=active_runtime,
            )

        layer_graph.run(
            (layer_graph_key, "gr_write_ffn_read"),
            eager=_segment_gr_write_ffn_read,
            out_ptr=scratch.ffn_gr.mixed.ptr,
            out_nbytes=scratch.ffn_gr.mixed.nbytes,
            out_regions=(
                (
                    scratch.ffn_gr.inject_logits.ptr,
                    scratch.ffn_gr.inject_logits.nbytes,
                ),
            ),
            stream=stream,
        )
        ffn_read = Qwen4ExpGRReadDeviceResult(
            normalized=scratch.ffn_gr.normalized,
            gate=scratch.ffn_gr.gate,
            mixed=scratch.ffn_gr.mixed,
            inject_logits=scratch.ffn_gr.inject_logits,
        )
    else:
        pending_write = _qwen4_exp_gr_pending_write(
        rows, residual_ptr, mixer_output.ptr, attention_read.inject_logits.ptr)
        if pending_write is None:
            qwen4_exp_gr_write_bf16_f32(
                residual_ptr, mixer_output.ptr, attention_read.inject_logits.ptr,
                scratch.after_attention.ptr, rows, branches, hidden,
                stream=stream, runtime=active_runtime,
            )
        ffn_read = run_qwen4_exp_gr_read(
            scratch.after_attention.ptr,
            weights.ffn_gr.norm_weight_ptr,
            weights.ffn_gr.down,
            weights.ffn_gr.up,
            weights.ffn_gr.inject,
            scratch.ffn_gr,
            rows=rows, branches=branches, hidden=hidden, low_rank=low_rank,
            stream=stream, runtime=active_runtime,
            pending_write=pending_write,
        )
    moe_output = run_qwen4_exp_moe(
        ffn_read.mixed.ptr,
        weights.moe,
        q4_bundle_prefill=q4_bundle_prefill,
        q5k_row4_prefill=q5k_row4_prefill,
        scratch=scratch.moe,
        rows=rows, hidden=hidden, ffn=ffn, experts=experts, top_k=top_k,
        stream=stream, runtime=active_runtime,
        graph_cache=moe_graph_cache, graph_key=moe_graph_key,
        q51_down_m1=q51_down_m1,
        q51_pair_prefill=q51_pair_prefill,
        q51_fold128_prefill=q51_fold128_prefill,
        q51_fold_pair_prefill=q51_fold_pair_prefill,
        q51_register_cache=q51_register_cache,
        q51_row_publish=q51_row_publish,
        q51_exact_grouped_down=q51_exact_grouped_down,
        q51_wmma_down=q51_wmma_down,
        q51_out8=q51_out8,
        q4_exact_grouped_gate=q4_exact_grouped_gate,
        q4_exact_grouped_all=q4_exact_grouped_all,
        q4_out4=q4_out4,
        q4_tile_m=q4_tile_m,
        q4_tile_n=q4_tile_n,
        forkb_grouped_down=forkb_grouped_down,
        q8_down_row4_prefill=q8_down_row4_prefill,
        q8_down_bundle_prefill=q8_down_bundle_prefill,
        q8_down_register=q8_down_register,
        q8_mapped_down=q8_mapped_down,
    )
    if layer_graph is not None:
        def _segment_epilogue(graph_stream: int) -> None:
            qwen4_exp_gr_write_bf16_bf16(
                scratch.after_attention.ptr,
                moe_output.output.ptr,
                ffn_read.inject_logits.ptr,
                scratch.output.ptr,
                rows,
                branches,
                hidden,
                stream=graph_stream,
                runtime=active_runtime,
            )

        layer_graph.run(
            (layer_graph_key, "epilogue"),
            eager=_segment_epilogue,
            out_ptr=scratch.output.ptr,
            out_nbytes=scratch.output.nbytes,
            stream=stream,
        )
    elif not _qwen4_exp_gr_defer_write(
        defer_write, rows, scratch.after_attention.ptr, moe_output.output.ptr,
        ffn_read.inject_logits.ptr,
    ):
        qwen4_exp_gr_write_bf16_bf16(
            scratch.after_attention.ptr,
            moe_output.output.ptr,
            ffn_read.inject_logits.ptr,
            scratch.output.ptr,
            rows,
            branches,
            hidden,
            stream=stream,
            runtime=active_runtime,
        )
    return scratch.output


def run_qwen4_exp_ple(
    residual_ptr: int,
    embedding_ptr: int,
    weights: Mapping[str, GGUFDeviceWeight],
    *,
    norm_key_ptr: int,
    norm_query_ptr: int,
    norm_conv_ptr: int,
    conv_weight_ptr: int,
    conv_history_ptr: int,
    scratch: Qwen4ExpPLEScratch,
    rows: int,
    branches: int,
    hidden: int,
    conv_kernel: int,
    dilation: int,
    eps: float = 1e-6,
    stream: int = 0,
    runtime: HipRuntime | None = None,
) -> DeviceBuffer:
    """Execute strict PLE projections, branch gate, Conv, and residual injection."""

    if scratch.closed:
        raise RuntimeError("Qwen4Exp PLE scratch is closed")
    active_runtime = runtime or scratch.runtime
    if active_runtime is not scratch.runtime:
        raise ValueError("runtime must match the PLE scratch owner")
    required = {"ple_key", "ple_value"}
    missing = sorted(required - set(weights))
    if missing:
        raise ValueError("missing Qwen4Exp PLE weights: " + ", ".join(missing))
    channels = branches * hidden
    launch_gguf_linear(
        weights["ple_key"],
        embedding_ptr,
        scratch.key.ptr,
        rows,
        hidden,
        channels,
        activation_dtype=GGUF_ACTIVATION_F32,
        output_dtype=GGUF_OUTPUT_F32,
        stream=stream,
        runtime=active_runtime,
    )
    launch_gguf_linear(
        weights["ple_value"],
        embedding_ptr,
        scratch.value.ptr,
        rows,
        hidden,
        hidden,
        activation_dtype=GGUF_ACTIVATION_F32,
        output_dtype=GGUF_OUTPUT_F32,
        stream=stream,
        runtime=active_runtime,
    )
    qwen4_exp_grouped_rmsnorm_f32(
        scratch.key.ptr,
        norm_key_ptr,
        scratch.key.ptr,
        rows,
        branches,
        hidden,
        eps,
        stream=stream,
        runtime=active_runtime,
    )
    qwen4_exp_grouped_rmsnorm_bf16_f32(
        residual_ptr,
        norm_query_ptr,
        scratch.query.ptr,
        rows,
        branches,
        hidden,
        eps,
        stream=stream,
        runtime=active_runtime,
    )
    qwen4_exp_ple_signed_sqrt_gate_f32(
        scratch.key.ptr,
        scratch.query.ptr,
        scratch.gate.ptr,
        rows,
        branches,
        hidden,
        stream=stream,
        runtime=active_runtime,
    )
    qwen4_exp_ple_repeat_gated_value_f32(
        scratch.value.ptr,
        scratch.gate.ptr,
        scratch.gated_value.ptr,
        rows,
        branches,
        hidden,
        stream=stream,
        runtime=active_runtime,
    )
    qwen4_exp_grouped_rmsnorm_f32(
        scratch.gated_value.ptr,
        norm_conv_ptr,
        scratch.normalized.ptr,
        rows,
        branches,
        hidden,
        eps,
        stream=stream,
        runtime=active_runtime,
    )
    qwen4_exp_ple_dilated_depthwise_conv_f32(
        scratch.normalized.ptr,
        conv_weight_ptr,
        conv_history_ptr,
        scratch.conv.ptr,
        rows,
        channels,
        conv_kernel,
        dilation,
        stream=stream,
        runtime=active_runtime,
    )
    qwen4_exp_scaled_silu_f32(
        scratch.conv.ptr,
        scratch.conv.ptr,
        rows * channels,
        1.0,
        stream=stream,
        runtime=active_runtime,
    )
    qwen4_exp_ple_add_delta_bf16_f32(
        residual_ptr,
        scratch.gated_value.ptr,
        scratch.conv.ptr,
        scratch.output.ptr,
        rows * channels,
        stream=stream,
        runtime=active_runtime,
    )
    return scratch.output


# Per-token state capture for a speculative verify pass: maps a GDN layer's
# recurrent-state pointer to (recurrent state rows, conv state rows) buffers.
# Set only around ``Qwen4ExpGGUFResidentModelRunner.verify_target_block_state_rows``.
_GDN_STATE_ROWS_CAPTURE: contextvars.ContextVar[Mapping[int, tuple[int, int]] | None] = (
    contextvars.ContextVar("qwen4_exp_gdn_state_rows_capture", default=None)
)


def run_qwen4_exp_gdn_token_mixer(
    mixed_ptr: int,
    weights: Mapping[str, GGUFDeviceWeight],
    *,
    conv_weight_ptr: int,
    dt_bias_ptr: int,
    a_ptr: int,
    norm_weight_ptr: int,
    conv_state_ptr: int,
    recurrent_state_ptr: int,
    scratch: Qwen4ExpGDNScratch,
    rows: int,
    hidden: int,
    num_k_heads: int,
    num_v_heads: int,
    head_dim: int,
    conv_kernel: int,
    stream: int = 0,
    runtime: HipRuntime | None = None,
    prefill_variant: str | None = None,
    gdn_tile16_prefill: bool = True,
    gdn_peer_prefill_layers: str | None = None,
    gdn_exact_bulk_conv: bool = True,
) -> DeviceBuffer:
    """Execute strict Qwen4Exp GDN projections, Conv, recurrence, and output.

    ``prefill_variant`` is the owning runner's GDN serial-prefill owner (see
    :func:`gdn_prefill_variant`); it selects the narrow/wide wave-norm leaf for
    multi-row chunks and leaves the register parent and decode untouched.
    ``gdn_tile16_prefill`` is the owning runner's tile-16 decision (see
    :func:`gdn_tile16_prefill`): False pins the columnwarp parent inside the
    columnwarp gate for a diagnostic arm.
    ``gdn_peer_prefill_layers`` is the owning runner's peer prefill scope (see
    :func:`gdn_peer_prefill`): the peer owner takes the layers inside it and
    everything else, including an unbound runner, keeps the serial owners.
    ``gdn_exact_bulk_conv`` is the owning runner's conv prefill decision (see
    :func:`gdn_exact_bulk_conv`): False runs the per-row serial fallback for
    chunks wider than one row. Decode always runs the decode conv kernel.
    """

    if scratch.closed:
        raise RuntimeError("Qwen4Exp GDN scratch is closed")
    if rows <= 0:
        raise ValueError("Qwen4Exp GDN rows must be positive")
    active_runtime = runtime or scratch.runtime
    if active_runtime is not scratch.runtime:
        raise ValueError("runtime must match the GDN scratch owner")
    qkv_width = 2 * num_k_heads * head_dim + num_v_heads * head_dim
    core_width = num_v_heads * head_dim
    required = {"attn_qkv", "attn_gate", "ssm_alpha", "ssm_beta", "ssm_out"}
    missing = sorted(required - set(weights))
    if missing:
        raise ValueError("missing Qwen4Exp GDN weights: " + ", ".join(missing))
    capture_map = _GDN_STATE_ROWS_CAPTURE.get()
    state_rows_capture = (
        capture_map.get(int(recurrent_state_ptr)) if capture_map else None
    )
    if state_rows_capture and not gdn_exact_bulk_conv:
        # The per-row loop runs the decode owner once per row and has nowhere to
        # publish per-token state; only the bulk owner takes state_rows_ptr.
        raise RuntimeError(
            "GDN state-row capture needs the exact bulk conv; the per-row "
            "serial fallback cannot publish state rows"
        )
    captured: set[str] = set()
    _launch_qwen4_exp_dense_pair(
        weights["attn_qkv"], weights["attn_gate"], mixed_ptr,
        scratch.qkv.ptr, scratch.gate.ptr, rows, hidden, qkv_width, core_width,
        stream=stream, runtime=active_runtime,
    )
    for slot, output, out_features in (
        ("ssm_alpha", scratch.alpha, num_v_heads),
        ("ssm_beta", scratch.beta, num_v_heads),
    ):
        launch_gguf_linear(
            weights[slot],
            mixed_ptr,
            output.ptr,
            rows,
            hidden,
            out_features,
            activation_dtype=GGUF_ACTIVATION_F32,
            output_dtype=GGUF_OUTPUT_F32,
            stream=stream,
            runtime=active_runtime,
        )
    if rows == 1:
        qwen35_linear_attn_conv_decode_f32(
            scratch.qkv.ptr,
            conv_state_ptr,
            conv_weight_ptr,
            scratch.conv.ptr,
            qkv_width,
            conv_kernel,
            stream=stream,
            runtime=active_runtime,
        )
        decode_step = qwen4_exp_gdn_decode_f32
        rowsplit_decode_key = KernelKey(
            weights["attn_qkv"].backend, "gdn_recurrence_norm_gate", "f32_state",
            "qwen4exp_gdn_rowsplit_decode",
        )
        if (
            head_dim == 128
            and os.environ.get("HIPENGINE_QWEN4_EXP_GDN_DECODE_ROWSPLIT", "0")
            not in {"", "0", "false", "False"}
            and is_registered(rowsplit_decode_key)
        ):
            # One read and one write of the state with both dot products in
            # registers; FP32 summation order differs from the decode owner.
            decode_step = resolve(
                backend=rowsplit_decode_key.backend,
                layer=rowsplit_decode_key.layer,
                quant=rowsplit_decode_key.quant,
                variant=rowsplit_decode_key.variant,
            )
        decode_step(
            scratch.conv.ptr,
            scratch.gate.ptr,
            scratch.alpha.ptr,
            scratch.beta.ptr,
            dt_bias_ptr,
            a_ptr,
            norm_weight_ptr,
            recurrent_state_ptr,
            scratch.core.ptr,
            num_k_heads,
            num_v_heads,
            head_dim,
            head_dim,
            stream=stream,
            runtime=active_runtime,
        )
    else:
        if gdn_exact_bulk_conv:
            resolve(
                backend=str(weights["attn_qkv"].backend),
                layer="linear_attn_conv_prefill",
                quant="qwen4_exp",
                variant="f32_decode_exact_k4",
            )(
                scratch.qkv.ptr,
                conv_state_ptr,
                conv_weight_ptr,
                scratch.conv.ptr,
                rows,
                qkv_width,
                conv_kernel,
                stream=stream,
                runtime=active_runtime,
                **({"state_rows_ptr": state_rows_capture[1]}
                   if state_rows_capture else {}),
            )
            captured.add("conv")
        else:
            for row in range(rows):
                qwen35_linear_attn_conv_decode_f32(
                    scratch.qkv.ptr + row * qkv_width * DType.FP32.itemsize,
                    conv_state_ptr,
                    conv_weight_ptr,
                    scratch.conv.ptr + row * qkv_width * DType.FP32.itemsize,
                    qkv_width,
                    conv_kernel,
                    stream=stream,
                    runtime=active_runtime,
                )
        peer_prefill = (
            qwen4_exp_gdn_peer_prefill_selected(
                head_dim=head_dim, published=gdn_peer_prefill_layers
            )
            and _qwen4_exp_gdn_peer_layer_allowed(
                weights, published=gdn_peer_prefill_layers
            )
        )
        colwarps_prefill = (
            head_dim == 128
            and os.environ.get(
                "HIPENGINE_QWEN4_EXP_GDN_COLWARPS_PREFILL", "0"
            )
            not in {"", "0", "false", "False"}
            and _qwen4_exp_layer_allowed(
                weights["attn_qkv"],
                env_name="HIPENGINE_QWEN4_EXP_GDN_COLWARPS_LAYERS",
                default="all",
            )
        )
        if colwarps_prefill:
            tile16_variant = os.environ.get("HIPENGINE_QWEN4_EXP_GDN_TILE16_VARIANT")
            tile16 = (
                resolve(
                    backend=weights["attn_qkv"].backend,
                    layer="gdn_recurrence_norm_gate", quant="f32_state",
                    variant=tile16_variant,
                )
                if tile16_variant else qwen4_exp_gdn_prefill_tiled16_f32
            )
            if qwen4_exp_gdn_tile16_owner_admitted(
                rows=rows,
                num_k_heads=num_k_heads,
                num_v_heads=num_v_heads,
                head_dim=head_dim,
                supports_any_rows=getattr(tile16, "supports_any_rows", False),
                enabled=gdn_tile16_prefill,
            ):
                # PF-5 production: exact token-tile-16 owner, bit-exact to
                # the columnwarp parent at the binding Hk16/Hv48/D128 shape.
                # A recurrence owner that can publish half(core) writes it
                # into the wide F16 stage, where the output projection (the
                # next launch, and the only reader of ``core``) finds it in
                # place of its own conversion.
                core_image_kwargs = {}
                core_count = rows * num_v_heads * head_dim
                image_workspace = wide_f16_activation_workspace()
                if (
                    getattr(tile16, "publishes_f16_image", False)
                    and _qwen4_exp_gr_f16_mirror_enabled()
                    and image_workspace is not None
                    and image_workspace.nbytes >= core_count * 2
                ):
                    _invalidate_wide_f16_publication()
                    core_image_kwargs["core_f16_ptr"] = image_workspace.ptr
                state_rows_kwargs = {}
                if state_rows_capture:
                    if not getattr(tile16, "supports_state_rows", False):
                        raise RuntimeError(
                            "GDN state-row capture needs the row-split recurrence")
                    state_rows_kwargs["state_rows_ptr"] = state_rows_capture[0]
                    captured.add("recurrence")
                tile16(
                    scratch.conv.ptr,
                    scratch.gate.ptr,
                    scratch.alpha.ptr,
                    scratch.beta.ptr,
                    dt_bias_ptr,
                    a_ptr,
                    norm_weight_ptr,
                    recurrent_state_ptr,
                    scratch.core.ptr,
                    rows,
                    num_k_heads,
                    num_v_heads,
                    head_dim,
                    head_dim,
                    stream=stream,
                    runtime=active_runtime,
                    **core_image_kwargs,
                    **state_rows_kwargs,
                )
                if core_image_kwargs:
                    declare_f16_mirror(
                        scratch.core.ptr, core_count, image_workspace.ptr
                    )
            else:
                qwen4_exp_gdn_prefill_columnwarps_f32(
                    scratch.conv.ptr,
                    scratch.gate.ptr,
                    scratch.alpha.ptr,
                    scratch.beta.ptr,
                    dt_bias_ptr,
                    a_ptr,
                    norm_weight_ptr,
                    recurrent_state_ptr,
                    scratch.core.ptr,
                    rows,
                    num_k_heads,
                    num_v_heads,
                    head_dim,
                    head_dim,
                    stream=stream,
                    runtime=active_runtime,
                )
        elif peer_prefill:
            compact_width = rows * num_k_heads * head_dim
            query_ptr = scratch.qkv.ptr
            key_ptr = query_ptr + compact_width * DType.FP32.itemsize
            value_ptr = key_ptr + compact_width * DType.FP32.itemsize
            qwen4_exp_gdn_peer_prefill_f32(
                scratch.conv.ptr,
                scratch.gate.ptr,
                scratch.alpha.ptr,
                scratch.beta.ptr,
                dt_bias_ptr,
                a_ptr,
                norm_weight_ptr,
                recurrent_state_ptr,
                query_ptr,
                key_ptr,
                value_ptr,
                scratch.core.ptr,
                rows,
                num_k_heads,
                num_v_heads,
                head_dim,
                head_dim,
                stream=stream,
                runtime=active_runtime,
            )
        else:
            serial_prefill = qwen4_exp_gdn_prefill_f32
            backend = str(weights["attn_qkv"].backend)
            if qwen4_exp_gdn_register_prefill_selected(
                backend=backend, rows=rows, num_k_heads=num_k_heads,
                num_v_heads=num_v_heads, head_dim=head_dim,
            ):
                serial_key = _qwen4_exp_gdn_prefill_key(
                    KernelKey(backend, "gdn_recurrence_norm_gate", "f32_state",
                              GDN_PREFILL_PARENT_VARIANT),
                    rows=rows, num_k_heads=num_k_heads,
                    num_v_heads=num_v_heads, head_dim=head_dim,
                    variant=prefill_variant,
                )
                serial_prefill = resolve(
                    backend=serial_key.backend, layer=serial_key.layer,
                    quant=serial_key.quant, variant=serial_key.variant)
            serial_prefill(
                scratch.conv.ptr,
                scratch.gate.ptr,
                scratch.alpha.ptr,
                scratch.beta.ptr,
                dt_bias_ptr,
                a_ptr,
                norm_weight_ptr,
                recurrent_state_ptr,
                scratch.core.ptr,
                rows,
                num_k_heads,
                num_v_heads,
                head_dim,
                head_dim,
                stream=stream,
                runtime=active_runtime,
            )
    if state_rows_capture and captured != {"conv", "recurrence"}:
        raise RuntimeError(
            "GDN state-row capture requested but this chunk ran "
            f"{sorted(captured) or 'no capturing owner'}")
    launch_gguf_linear(
        weights["ssm_out"],
        scratch.core.ptr,
        scratch.output.ptr,
        rows,
        core_width,
        hidden,
        activation_dtype=GGUF_ACTIVATION_F32,
        output_dtype=GGUF_OUTPUT_F32,
        stream=stream,
        runtime=active_runtime,
    )
    # The image of ``core`` lives in the shared wide F16 stage, which the next
    # staged conversion overwrites; retire the declaration with its one reader.
    invalidate_f16_mirror()
    return scratch.output


@dataclass
class Qwen4ExpTargetVerifyOutput:
    """Bounded request-owned rows<=8 storage for future target verification."""

    rows_capacity: int
    residual_rows: DeviceBuffer
    logits_rows: DeviceBuffer
    token_ids: DeviceBuffer
    head_scratch: Qwen4ExpGRScratch
    runtime: HipRuntime
    closed: bool = False

    @classmethod
    def allocate(
        cls,
        *,
        rows: int,
        branches: int,
        hidden: int,
        low_rank: int,
        vocab: int,
        runtime: HipRuntime | None = None,
    ) -> "Qwen4ExpTargetVerifyOutput":
        capacity = int(rows)
        if not 1 <= capacity <= 8:
            raise ValueError("Qwen4Exp target verify rows must be in 1..8")
        if min(int(branches), int(hidden), int(low_rank), int(vocab)) <= 0:
            raise ValueError("Qwen4Exp target verify dimensions must be positive")
        active_runtime = runtime or get_hip_runtime()
        buffers: list[DeviceBuffer] = []
        head_scratch = None
        try:
            buffers = [
                malloc(capacity * branches * hidden * DType.BF16.itemsize, runtime=active_runtime),
                malloc(capacity * vocab * DType.FP32.itemsize, runtime=active_runtime),
                malloc(capacity * DType.INT64.itemsize, runtime=active_runtime),
            ]
            head_scratch = Qwen4ExpGRScratch.allocate(
                rows=capacity,
                branches=branches,
                hidden=hidden,
                low_rank=low_rank,
                runtime=active_runtime,
            )
        except Exception:
            if head_scratch is not None:
                head_scratch.close()
            for buffer in reversed(buffers):
                free(buffer, runtime=active_runtime)
            raise
        return cls(capacity, *buffers, head_scratch, active_runtime)

    @property
    def owned_buffers(self) -> Mapping[str, DeviceBuffer]:
        return MappingProxyType({
            "residual_rows": self.residual_rows,
            "logits_rows": self.logits_rows,
            "token_ids": self.token_ids,
        })

    @property
    def nbytes_by_owner(self) -> Mapping[str, int]:
        return MappingProxyType({
            name: buffer.nbytes for name, buffer in self.owned_buffers.items()
        })

    def require_rows(self, rows: int) -> None:
        if self.closed:
            raise RuntimeError("Qwen4Exp target verify output is closed")
        count = int(rows)
        if not 1 <= count <= self.rows_capacity:
            raise ValueError(
                f"Qwen4Exp target verify rows must be in 1..{self.rows_capacity}"
            )

    def close(self) -> None:
        if self.closed:
            return
        self.head_scratch.close()
        for buffer in reversed(tuple(self.owned_buffers.values())):
            free(buffer, runtime=self.runtime)
        self.closed = True


def qwen4_exp_moe_graph_enabled(
    backend: str, explicit: bool | None = None
) -> bool:
    """Admit the captured c1 MoE decode section for ``backend``.

    ``explicit`` is the harness/test control: a caller that must compare the
    captured and eager profiles on one residency passes the arm it wants, and
    every other caller leaves it ``None`` so the backend package's own
    capability decides. There is deliberately no process env override; the
    retired ``HIPENGINE_QWEN4_EXP_MOE_GRAPH`` switch made a fixture's eager
    isolation silently inert once the read was gone.
    """

    if explicit is not None:
        return bool(explicit)
    return bool(backend_package_capability(backend, "QWEN4_EXP_MOE_GRAPH", False))


@dataclass(frozen=True)
class Qwen4ExpRunnerSnapshot:
    decode_state: Qwen4ExpDecodeStateSnapshot
    position: int
    ple_hash_states: Mapping[int, PLEHashState]


@dataclass
class Qwen4ExpRunnerDeviceTransaction:
    """One all-or-nothing device-state and cursor transaction."""

    decode_state: Qwen4ExpDecodeStateDeviceSnapshot
    position: int
    ple_hash_states: Mapping[int, PLEHashState]
    reusable_snapshot: bool = False
    committed: bool = False
    rolled_back: bool = False
    closed: bool = False

    def require_active(self) -> None:
        if self.committed or self.rolled_back or self.closed:
            raise RuntimeError("Qwen4Exp device transaction is already finalized")

    def close(self) -> None:
        if self.closed:
            return
        if not self.reusable_snapshot:
            self.decode_state.close()
        self.closed = True


@dataclass(frozen=True)
class Qwen4ExpTokenResult:
    token_id: int
    logits: np.ndarray | None
    hidden_seeds: np.ndarray | None = None

    @property
    def hidden_seed(self) -> np.ndarray | None:
        if self.hidden_seeds is None:
            return None
        return self.hidden_seeds[-1]


@dataclass(frozen=True)
class Qwen4ExpTargetVerifyResult:
    token_ids: tuple[int, ...]
    logits: tuple[np.ndarray, ...] | None
    hidden_seeds: np.ndarray | None


@dataclass(frozen=True)
class Qwen4ExpTargetAcceptResult:
    verify: Qwen4ExpTargetVerifyResult
    accepted: int
    consumed: int
    replayed: bool


@dataclass(eq=False)
class Qwen4ExpPrefillWorkspace:
    """Own prefill storage independently of decode state and borrowed views."""

    config: Qwen4ExpGGUFConfig
    runtime: HipRuntime
    resident: Qwen4ExpResidentWeights | None
    backend: str | None
    max_sequence_length: int
    prefill_chunk_size: int
    gdn_prefill_scratch: Qwen4ExpGDNLayerScratch
    qsa_prefill_scratch: Qwen4ExpQSALayerScratch
    ple_prefill_scratch: Qwen4ExpPLEScratch
    qsa_prefill_metadata: Qwen4ExpQSAPrefillMetadata
    _prefill_buffers: list[DeviceBuffer] = field(default_factory=list)
    closed: bool = False

    @classmethod
    def allocate_scratch(cls, runner, chunk_size):
        cfg = runner.config
        rows = min(int(chunk_size), runner.max_sequence_length)
        if rows <= 0:
            raise ValueError("prefill workspace chunk must be positive")
        owners = []
        try:
            gdn = Qwen4ExpGDNLayerScratch.allocate(
                rows=rows, branches=cfg.residual_branch_count, hidden=cfg.hidden_size,
                low_rank=cfg.residual_low_rank,
                qkv_width=2 * cfg.gdn_group_count * cfg.gdn_state_size + cfg.gdn_inner_size,
                core_width=cfg.gdn_inner_size, scalar_width=cfg.gdn_time_step_rank,
                ffn=cfg.expert_feed_forward_length, experts=cfg.expert_count,
                top_k=cfg.expert_used_count, runtime=runner.runtime)
            owners.append(gdn)
            qsa = Qwen4ExpQSALayerScratch.allocate(
                rows=rows, branches=cfg.residual_branch_count, hidden=cfg.hidden_size,
                low_rank=cfg.residual_low_rank, query_heads=cfg.attention_head_count,
                kv_heads=cfg.attention_kv_head_count, head_dim=cfg.attention_key_length,
                ffn=cfg.expert_feed_forward_length, experts=cfg.expert_count,
                top_k=cfg.expert_used_count, index_heads=cfg.indexer_head_count,
                index_dim=cfg.indexer_key_length, runtime=runner.runtime)
            owners.append(qsa)
            ple = Qwen4ExpPLEScratch.allocate(
                rows=rows, branches=cfg.residual_branch_count,
                hidden=cfg.hidden_size, runtime=runner.runtime)
            owners.append(ple)
            metadata = Qwen4ExpQSAPrefillMetadata.allocate(
                runner.attention_states[0], rows=rows,
                selection_capacity=cfg.qsa_dense_equivalent_max_tokens,
                score_blocks=(runner.max_sequence_length + cfg.qsa_compression_ratio - 1)
                // cfg.qsa_compression_ratio)
            owners.append(metadata)
            return cls(cfg, runner.runtime, getattr(runner, "resident", None),
                       getattr(runner, "backend", None), runner.max_sequence_length,
                       int(chunk_size), gdn, qsa, ple, metadata)
        except BaseException:
            for owner in reversed(owners):
                owner.close()
            raise

    def allocate_inputs(self):
        if self.closed or self._prefill_buffers:
            raise RuntimeError("prefill inputs already allocated or owner closed")
        cfg = self.config
        rows = min(self.prefill_chunk_size, self.max_sequence_length)
        try:
            for nbytes in (
                rows * np.dtype(np.int64).itemsize,
                rows * cfg.hidden_size * DType.BF16.itemsize,
                rows * cfg.hidden_size * DType.FP32.itemsize,
                rows * cfg.residual_branch_count * cfg.hidden_size * DType.BF16.itemsize,
                3 * rows * np.dtype(np.int64).itemsize,
            ):
                self._prefill_buffers.append(malloc(nbytes, runtime=self.runtime))
        except BaseException:
            self.close_inputs()
            raise

    def close_inputs(self):
        for buffer in reversed(self._prefill_buffers):
            free(buffer, runtime=self.runtime)
        self._prefill_buffers.clear()

    def close(self):
        if self.closed:
            return
        self.close_inputs()
        for owner in (self.qsa_prefill_metadata, self.ple_prefill_scratch,
                      self.qsa_prefill_scratch, self.gdn_prefill_scratch):
            owner.close()
        self.closed = True


class Qwen4ExpGGUFResidentModelRunner:
    """Strict c1 text runner for the complete 48-layer Qwen4Exp target."""

    # The execution-profile binder publishes DP-2's lane decision here, per
    # runner; an unbound runner keeps the parent route. This is the request's
    # own decision, not the effective one: a diagnostic scope may outrank it.
    dense_f16_pair_enabled: bool = False
    dense_wide_tail_variant: str | None = None
    norm_f16_publication_enabled: bool = False
    # Same shape for the Q8 MMQ lane: the profile publishes which variant
    # siblings this runner's prefill may select. Publication is independent of
    # the MMQ parent's own switch, so a diagnostic that re-enables
    # HIPENGINE_QWEN4_EXP_Q8_MMQ_PREFILL after the bind still gets the children
    # the production plan selected.
    dense_mmq_token64_enabled: bool = False
    dense_mmq_raw_vector_enabled: bool = False
    dense_mmq_vec4_enabled: bool = False
    # H256 QSA prefill head grouping, published from the plan's own selection
    # ("quad" on the UD lane). "pair" is diagnostic-only and never published; an
    # unbound runner keeps the single-head parent.
    qsa_prefill_head_mode: str = "parent"
    qsa_h256_prefill_mode: str = "parent"
    # QSA chunk selection owner: True runs the batched score/top-k/expand path,
    # False the strict per-row fallback. The prefill chunk path is its only
    # reader and forwards it to the token mixer, so a diagnostic binds the
    # fallback on the one runner it measures.
    qsa_batched_selection: bool = True
    # QSA H128 sparse-attention owner: True runs the wave32 paged kernels (the
    # decode dispatch and the prefill rows fallback below the H256 wave-prefill
    # owner), the promoted production arithmetic for head_dim 128; False keeps
    # the wave64 parents, which stay the registered strict fallback. The
    # geometry guard stays dispatch-time and the H256 owners keep their own
    # preemption. Both the c1 decode path and the prefill chunk path forward it,
    # so a diagnostic binds the parent on the one runner it measures.
    qsa_wave32: bool = True
    # QSA ordered-v2 sparse decode, published by the profile binder from the
    # plan's own c1 H256 indexed sparse decode row: True on a production plan,
    # False on every plan that keeps the ordered v1 parent. The dispatch-time
    # selector keeps the v1 switch, H256 and 1..4096 guards. A diagnostic scope
    # may outrank it.
    qsa_ordered_decode_v2: bool = False
    # GDN serial-prefill owner, published by the profile binder from the plan's
    # own selection: the narrow checked-index leaf on the UD lane, the wide
    # wave-norm leaf where that admission applies, and None (the register
    # parent) everywhere else. A diagnostic scope may outrank it.
    gdn_prefill_variant: str | None = None
    # GDN tile-16 prefill owner inside the columnwarp gate, for the binding
    # Hk16/Hv48/D128 envelope. True on every runner the profile constructs, and
    # the prefill chunk path is its only reader: decode runs one row, and the
    # batch GDN mixer is decode-only. A diagnostic scope binds the columnwarp
    # parent on the one runner it measures; a fresh runner or a serving clone
    # inherits this default, never that binding.
    gdn_tile16_prefill: bool = True
    # GDN peer prefill owner's layer scope, published by the profile binder as a
    # request-local decision rather than a process-wide switch. ``None`` (an
    # unbound runner, strict, and every plan that falls back to the strict
    # binder) binds the serial sigmoid owners; a scope string such as the
    # certified layers 35-47 lets the peer owner take those layers inside the
    # prefill chunk path. Decode runs one row and the batch GDN mixer is
    # decode-only, so neither reads it. A diagnostic scope binds this field on
    # the one runner it measures; a fresh runner or a serving clone inherits the
    # published value, never that binding.
    gdn_peer_prefill_layers: str | None = None
    # GDN conv prefill owner: True runs the exact bulk conv for chunks wider
    # than one row, False the per-row serial fallback. Decode always runs the
    # decode conv kernel. The prefill chunk path is its only reader and forwards
    # it to the token mixer, so a diagnostic binds the serial arm on the one
    # runner it measures.
    gdn_exact_bulk_conv: bool = True
    # Q5_1 exact grouped down gate, published by the profile binder for every
    # plan: True runs the exact grouped Q5_1 down chain on each multi-row
    # prefill chunk, and False skips that chain so the dispatch falls through to
    # the sorted/WMMA owners below it. It is the chain's own gate, so every Q5_1
    # ladder rung inside it is unreachable while it is closed. Decode never
    # selects the chain. A diagnostic scope may outrank it.
    q51_exact_grouped_down: bool = True
    # Q5_1 F16 WMMA down owner admission, published off by the profile binder for
    # every plan: it is the down-only MX-1 diagnostic below the exact grouped
    # Q5_1 down chain, and the production grouped MoE route admits that owner on
    # the Q5K gate/up window by itself. The dispatch ORs the two, so this field
    # only widens the route to chunks the production route does not own. Decode
    # never reaches the production term, so a decode capture that ran this owner
    # is keyed on it. A diagnostic scope may outrank it.
    q51_wmma_down: bool = False
    # Q4_K exact grouped gate/up gate and its `_ALL` joint term, published by the
    # profile binder for every plan: the chain runs on every Q4_K/Q4_K multi-row
    # chunk unless an arm closes the gate, and `_ALL` only narrows it to chunks
    # the exact grouped Q5_1 down chain already owns. Closing the gate is the
    # route to the plain-iu8 and F16 compact gate/up owners below the chain.
    q4_exact_grouped_gate: bool = True
    q4_exact_grouped_all: bool = True
    # Q5_1 out8 fallback selection, published by the profile binder for every
    # plan: True runs the out8 rowbatch8 owner as the exact grouped Q5_1 down
    # chain's non-expertgrid fallback, and False keeps the output1 rowbatch8
    # base. It is one rung: the exact expert-grid family above it keeps its own
    # out8 variants, and the pair2/fold rungs below it replace the variant
    # either way. A diagnostic scope may outrank it.
    q51_out8: bool = True
    # Q4 out4 fallback selection, published by the profile binder for every
    # plan: True runs the out4 rowbatch8 owner as the exact grouped Q4_K gate/up
    # chain's non-expertgrid fallback, and False keeps the strict rowbatch8 base.
    # It is one rung: the expert-grid family above it keeps its own out4 variant
    # and the pair2/bundle preemptors inside it, and the iu8 chain above that is
    # unaffected. A diagnostic scope may outrank it.
    q4_out4: bool = True
    q4_bundle_prefill: bool = False
    # The compact F16 Q4_K dual gate/up tile pair, published 16/16 by the
    # profile binder for every plan and ``None`` (the dispatch heuristic) on an
    # unbound runner: M is 64 from 512 rows up and 16 below, N is 16. The pair
    # only shapes that fallback owner's launch below the exact grouped Q4_K
    # chain and the iu8 owner. Multi-row batch capture identity includes the
    # pair. A diagnostic scope may outrank the publication.
    q4_tile_m: int | None = None
    q4_tile_n: int | None = None
    # The Q8_0 dense WMMA prefill tile pair, published (64, 32) by the profile
    # binder and ``None`` on an unbound runner (the arch heuristic). It is bound
    # around each execution entry that can reach that family, so a co-resident
    # model in the same process keeps its own default. A diagnostic arm rebinds
    # it on the runner it measures.
    q8_wmma_tiles: tuple[int, int] | None = None
    # Q5_1 down M1 selection, published by the profile binder for the exact
    # grouped Q5_1 down chain's small-chunk scope: True on a production plan,
    # False on every plan that keeps the preceding expertgrid64 owner. The
    # pair2/fold ladder above it preempts it at rows>=64, and the exact expert
    # grid gate still decides whether that family is selected at all. A
    # diagnostic scope may outrank it.
    q51_down_m1: bool = False
    # Pair2 selection is profile-owned; rows, input width and registry
    # capability still guard the selected variant at dispatch.
    q51_pair_prefill: bool = False
    # Q5_1 fold128 selection, published by the profile binder for the grouped
    # pair2 chain's rows>=64 scope: True on a production plan, False on every
    # plan that keeps the pair2 parent. It is the parent of the folded-pair,
    # register-cache and row-publish siblings, so each of those plans reads the
    # variant this selects. A diagnostic scope may outrank it.
    q51_fold128_prefill: bool = False
    # Q5_1 folded-pair selection, published by the profile binder for the
    # fold128 ladder's rows>=512 scope: True on a production plan, False on
    # every plan that keeps the fold128 parent. A diagnostic scope may outrank
    # it.
    q51_fold_pair_prefill: bool = False
    # Q5_1 register-cache fallback selection, published by the profile binder for
    # the rows>=512 K640 grouped Q5_1 down scope: True on a production plan,
    # False on every plan that keeps the folded-pair parent. It is the parent of
    # the row-publish sibling, so a plan that publishes row-publish replaces the
    # leaf this selects. A diagnostic scope may outrank it.
    q51_register_cache: bool = False
    # Q5_1 row-publish selection, published by the profile binder from the plan's
    # own rows>=512 K640 row: True where that plan declares the row-publish
    # sibling of the register-cache prefill owner, False on every plan that
    # keeps the parent. A diagnostic scope may outrank it.
    q51_row_publish: bool = False
    # PF-1 fork-b grouped Q8_0 down selection, published by the profile binder
    # for the sorted chain's Q8_0 down scope: True on a production plan, False
    # on every plan that keeps the strict per-expert selected gemv. A plan that
    # binds the Q8_0 WMMA down owner preempts it, so it is inert there. A
    # diagnostic scope may outrank it.
    forkb_grouped_down: bool = False
    # Q8_0 down row4 selection, published by the profile binder for the sorted
    # grouped chain's rows>=512 scope: True on a production plan, False on every
    # plan that keeps the sorted parent. It is the parent of the bundle and
    # register siblings, which read the variant it selects. A diagnostic scope
    # may outrank it.
    q8_down_row4_prefill: bool = False
    # Q5_K token-major gate/up row4, published by the profile binder.
    q5k_row4_prefill: bool = False
    # Q8_0 down bundle selection, published by the profile binder for the row4
    # ladder's rows>=512 scope: True on a production plan, False on every plan
    # that keeps the row4 parent. The register sibling downstream replaces it in
    # turn. A diagnostic scope may outrank it.
    q8_down_bundle_prefill: bool = False
    # Q8_0 down register selection, published by the profile binder for the
    # rows>=512 K640 Q8_0 expert-down scope: True on a production plan, False on
    # every plan that keeps the row4 bundle parent. The sorted grouped chain and
    # the token-major mapped path both read it. A diagnostic scope may outrank
    # it.
    q8_down_register: bool = False
    # Q8_0 token-major mapped down owner, published by the profile binder for
    # the rows>=512 scope: True on a production plan, False on every plan that
    # keeps the token-major parent. It admits the row4 bundle leaf only when the
    # current call built the Q5_K row4 map. A diagnostic scope may outrank it.
    q8_mapped_down: bool = False

    def __init__(
        self,
        resident: Qwen4ExpResidentWeights,
        *,
        max_sequence_length: int = 2_051,
        prefill_chunk_size: int = 512,
        backend: str = "hip_gfx1151",
        runtime: HipRuntime | None = None,
        moe_graph_enabled: bool | None = None,
    ) -> None:
        self.resident = resident
        self.config = resident.plan.config
        self.backend = str(backend)
        if not 0 < int(max_sequence_length) <= self.config.context_length:
            raise ValueError(
                "Qwen4Exp runner max_sequence_length must be in "
                f"1..{self.config.context_length}"
            )
        self.runtime = runtime or get_hip_runtime()
        # R7 wrapper-host route: populate the PLE table into page cache so
        # decode-step row gathers hit warm pages instead of storage faults
        # (measured 1.4-27.7 ms/step cold, case-dependent). Pure cache hint;
        # opt-in via env until gated.
        # OPT-IN (R7 wrapper-host route, demoted from default per resource
        # qualification 2026-09-10): one mmap-touch sweep (~28.8 GB, +15.4 s
        # and ~6,015 major faults per runner construction) removes the
        # cold-cache PLE gather fault cost (measured 1.4-27.7 ms/step,
        # case-dependent; TG median -4.8 ms/token). Bit-exact. Amortization
        # is 577-15,400 generated tokens (median ~2,050) - short-lived
        # runners never amortize, and the 28.8 GB page-cache residency is
        # unqualified on memory-constrained shared hosts (capture windows
        # showed reclaim pressure). Enable via HIPENGINE_QWEN4_EXP_PLE_WARM=1
        # for long-lived serving.
        if os.environ.get("HIPENGINE_QWEN4_EXP_PLE_WARM", "0") == "1":
            self.resident.ple_table.warm_page_cache()
        self.max_sequence_length = int(max_sequence_length)
        self.prefill_chunk_size = int(prefill_chunk_size)
        if self.prefill_chunk_size <= 0:
            raise ValueError("Qwen4Exp prefill_chunk_size must be positive")
        load_backend_kernel_package(self.backend)
        self.rebind_resident_weights()
        self.state: Qwen4ExpDecodeState | None = None
        self.gdn_scratch: Qwen4ExpGDNLayerScratch | None = None
        self.qsa_scratch: Qwen4ExpQSALayerScratch | None = None
        self.ple_scratch: Qwen4ExpPLEScratch | None = None
        self.head_scratch: Qwen4ExpGRScratch | None = None
        self._target_verify_output: Qwen4ExpTargetVerifyOutput | None = None
        self._device_transaction_snapshot: Qwen4ExpDecodeStateDeviceSnapshot | None = None
        self._device_transaction_lease = False
        # Batched verify with per-row state capture (MTP): device buffers sized
        # for the largest block seen, and the block awaiting its commit.
        self._verify_rows_buffers: dict[str, DeviceBuffer] = {}
        self._verify_rows_capacity = 0
        self._pending_verify_block: tuple[Any, ...] | None = None
        self.gdn_prefill_scratch: Qwen4ExpGDNLayerScratch | None = None
        self.qsa_prefill_scratch: Qwen4ExpQSALayerScratch | None = None
        self.ple_prefill_scratch: Qwen4ExpPLEScratch | None = None
        self.qsa_prefill_metadata: Qwen4ExpQSAPrefillMetadata | None = None
        self.attention_states: tuple[Qwen4ExpDenseAttentionState, ...] = ()
        self.index_states: tuple[Qwen4ExpQSAIndexDeviceState, ...] = ()
        self._buffers: list[DeviceBuffer] = []
        self._prefill_buffers: list[DeviceBuffer] = []
        self._prefill_workspaces: list[Qwen4ExpPrefillWorkspace] = []
        self._q8_mmq_weight_sidecars = None
        self._ple_hash_states: dict[int, PLEHashState] = {}
        self.moe_graph_cache: MoeGraphCache | None = None
        self.position = 0
        self.closed = False
        try:
            self._allocate()
            self.moe_graph_cache = MoeGraphCache(
                self.runtime,
                enabled=qwen4_exp_moe_graph_enabled(
                    self.backend, moe_graph_enabled
                ),
            )
            layer_graph_override = os.environ.get(
                "HIPENGINE_QWEN4_EXP_LAYER_GRAPHS"
            )
            layer_graph_enabled = bool(
                backend_package_capability(
                    self.backend, "QWEN4_EXP_LAYER_GRAPHS", False
                )
                if layer_graph_override is None
                else layer_graph_override not in {"", "0", "false", "False"}
            )
            self.layer_graph_cache: MoeGraphCache | None = MoeGraphCache(
                self.runtime, enabled=layer_graph_enabled
            )
        except Exception:
            self.close()
            raise

    def _allocate(self) -> None:
        cfg = self.config
        qkv_width = 2 * cfg.gdn_group_count * cfg.gdn_state_size + cfg.gdn_inner_size
        self.state = Qwen4ExpDecodeState.allocate(
            gdn_layers=cfg.layer_types.count("gdn"),
            gdn_value_heads=cfg.gdn_time_step_rank,
            gdn_head_dim=cfg.gdn_state_size,
            gdn_conv_channels=qkv_width,
            gdn_conv_kernel=cfg.gdn_conv_kernel,
            residual_branches=cfg.residual_branch_count,
            hidden=cfg.hidden_size,
            ple_conv_kernel=cfg.ple_conv_kernel,
            ple_dilation=cfg.ple_ngram_size,
            runtime=self.runtime,
        )
        self.gdn_scratch = Qwen4ExpGDNLayerScratch.allocate(
            rows=1, branches=cfg.residual_branch_count, hidden=cfg.hidden_size,
            low_rank=cfg.residual_low_rank, qkv_width=qkv_width,
            core_width=cfg.gdn_inner_size, scalar_width=cfg.gdn_time_step_rank,
            ffn=cfg.expert_feed_forward_length, experts=cfg.expert_count,
            top_k=cfg.expert_used_count, runtime=self.runtime,
        )
        self.qsa_scratch = Qwen4ExpQSALayerScratch.allocate(
            rows=1, branches=cfg.residual_branch_count, hidden=cfg.hidden_size,
            low_rank=cfg.residual_low_rank, query_heads=cfg.attention_head_count,
            kv_heads=cfg.attention_kv_head_count, head_dim=cfg.attention_key_length,
            ffn=cfg.expert_feed_forward_length, experts=cfg.expert_count,
            top_k=cfg.expert_used_count, index_heads=cfg.indexer_head_count,
            index_dim=cfg.indexer_key_length, runtime=self.runtime,
        )
        self.qsa_scratch.qsa.ordered_attention_scratch(
            query_heads=cfg.attention_head_count,
            selected_count=cfg.qsa_dense_equivalent_max_tokens,
        )
        self.ple_scratch = Qwen4ExpPLEScratch.allocate(
            rows=1, branches=cfg.residual_branch_count, hidden=cfg.hidden_size,
            runtime=self.runtime,
        )
        self.head_scratch = Qwen4ExpGRScratch.allocate(
            rows=1, branches=cfg.residual_branch_count, hidden=cfg.hidden_size,
            low_rank=cfg.residual_low_rank, runtime=self.runtime,
        )
        self._batched_position = _qwen4_exp_batched_position_enabled()
        self._shared_position_context: DeviceBuffer | None = None
        self._shared_position_context_host: np.ndarray | None = None
        if self._batched_position:
            state_count = cfg.layer_types.count("qsa")
            # One interleaved [position, context] int64 pair per QSA state; a
            # single blocking H2D copy per decode step replaces the 24
            # per-layer set_position copies (R7 wrapper-host screen).
            self._shared_position_context = malloc(
                2 * state_count * DType.INT64.itemsize, runtime=self.runtime
            )
            self._shared_position_context_host = np.zeros(
                2 * state_count, dtype=np.int64
            )
            self._shared_position_context_host[1::2] = 1
            copy_host_array_to_device(
                self._shared_position_context,
                self._shared_position_context_host,
                runtime=self.runtime,
            )
        self.attention_states = tuple(
            Qwen4ExpDenseAttentionState.allocate(
                max_positions=self.max_sequence_length,
                block_size=256,
                kv_heads=cfg.attention_kv_head_count,
                head_dim=cfg.attention_key_length,
                runtime=self.runtime,
                position_view=(
                    DeviceBuffer(
                        self._shared_position_context.ptr + 16 * index,
                        DType.INT64.itemsize,
                    )
                    if self._batched_position
                    else None
                ),
                context_view=(
                    DeviceBuffer(
                        self._shared_position_context.ptr + 16 * index + 8,
                        DType.INT64.itemsize,
                    )
                    if self._batched_position
                    else None
                ),
            )
            for index in range(cfg.layer_types.count("qsa"))
        )
        self.index_states = tuple(
            Qwen4ExpQSAIndexDeviceState.allocate(
                attention_state=attention,
                index_heads=cfg.indexer_head_count,
                index_dim=cfg.indexer_key_length,
                compression_ratio=cfg.qsa_compression_ratio,
                block_budget=cfg.qsa_block_budget,
                runtime=self.runtime,
            )
            for attention in self.attention_states
        )
        workspace = Qwen4ExpPrefillWorkspace.allocate_scratch(self, self.prefill_chunk_size)
        self._prefill_workspaces = [workspace]
        self.gdn_prefill_scratch = workspace.gdn_prefill_scratch
        self.qsa_prefill_scratch = workspace.qsa_prefill_scratch
        self.ple_prefill_scratch = workspace.ple_prefill_scratch
        self.qsa_prefill_metadata = workspace.qsa_prefill_metadata
        argmax_blocks = lm_head_argmax_stage1_blocks(cfg.vocab_size, threads=256)
        for nbytes in (
            np.dtype(np.int64).itemsize,
            cfg.hidden_size * 2,
            cfg.hidden_size * 4,
            cfg.vocab_size * 4,
            cfg.residual_width * DType.BF16.itemsize,
            3 * np.dtype(np.int64).itemsize,
            argmax_blocks * DType.FP32.itemsize,
            argmax_blocks * DType.INT64.itemsize,
            DType.FP32.itemsize,
        ):
            self._buffers.append(malloc(nbytes, runtime=self.runtime))
        # Keep primary input allocations after decode I/O, preserving allocation order.
        workspace.allocate_inputs()
        self._prefill_buffers = workspace._prefill_buffers
        self._q8_mmq_policy = None
        self._q8_mmq_library = None
        self._q8_mmq_buffers: tuple[DeviceBuffer, ...] = ()
        self._configure_q8_mmq_prefill_resources()
        self._configure_q8_mmq_weight_sidecars()

    def _wide_f16_activation_context(self):
        """Bind stable runner-owned staging only during chunked prefill.

        All consumers enqueue on the prefill stream. Each launch overwrites
        the scratch only after the previous consumer, so no input cache or
        cross-request invalidation is needed. The original workspace capacity
        bounds the allocation even while a smaller prefill owner is borrowed.
        """
        self._require_open()
        # Qualified against the production F32-input owner; retain an opt-out.
        if not _q8_dense_wide_enabled() or not wide_f16_activation_enabled(True):
            return self._wide_f16_pair_scopes(
                wide_f16_activation_session(False), enabled=False)
        buffer = getattr(self, "_wide_f16_activation_buffer", None)
        if buffer is None:
            nbytes = qwen4_exp_wide_f16_activation_nbytes(
                self.config, context_tokens=self.max_sequence_length,
                prefill_chunk_size=self._prefill_workspaces[0].prefill_chunk_size)
            buffer = malloc(nbytes, runtime=self.runtime)
            self._buffers.append(buffer)
            self._wide_f16_activation_buffer = buffer
        return self._wide_f16_pair_scopes(
            wide_f16_activation_session(
                workspace_ptr=buffer.ptr, workspace_nbytes=buffer.nbytes),
            enabled=bool(self.dense_f16_pair_enabled),
        )

    @contextlib.contextmanager
    def _wide_f16_pair_scopes(self, workspace_scope, *, enabled: bool):
        """Bind this runner's pair decision around the workspace it owns.

        Both scopes open on ``__enter__`` and unwind in reverse order, so a
        context that is constructed and dropped leaves no workspace or pair
        state behind, and an exception inside the body restores both.
        """

        with (workspace_scope, wide_f16_pair_session(enabled),
              dense_wide_tail_session(self.dense_wide_tail_variant),
              norm_f16_publication_session(self.norm_f16_publication_enabled)):
            yield

    def _allocate_extra_prefill_workspace(self, chunk_size):
        """Prepare an inactive owner; callers must admit its additional memory."""
        self._require_open()
        if self._q8_mmq_buffers or self._q8_mmq_weight_sidecars is not None:
            raise ValueError("extra prefill workspaces require no global MMQ resources")
        workspace = Qwen4ExpPrefillWorkspace.allocate_scratch(self, chunk_size)
        try:
            workspace.allocate_inputs()
        except BaseException:
            workspace.close()
            raise
        self._prefill_workspaces.append(workspace)
        return workspace

    def _configure_q8_mmq_weight_sidecars(self) -> None:
        if self._q8_mmq_weight_sidecars is not None or os.environ.get(
            "HIPENGINE_QWEN4_EXP_Q8_MMQ_PREPACK", "0"
        ) != "1" or self._q8_mmq_library is None:
            return
        from hipengine.runtime.gguf_q8_mmq_sidecars import Q8MMQWeightSidecars
        owner = Q8MMQWeightSidecars(runtime=self.runtime,library=self._q8_mmq_library)
        try:
            for binding in self.gdn_bindings.values():
                for slot in ("attn_qkv", "ssm_out"):
                    weight = binding.mixer.projections[slot]
                    n,k = map(int,weight.spec.source.shape)
                    if (k,n) not in {(2560,10240), (6144,2560)}:
                        continue
                    owner.prepare(weight)
        except BaseException:
            owner.close()
            raise
        self._q8_mmq_weight_sidecars = owner

    def _configure_q8_mmq_prefill_resources(self) -> None:
        if self._q8_mmq_buffers or os.environ.get(
            "HIPENGINE_QWEN4_EXP_Q8_MMQ_PREFILL", "0"
        ) in {"", "0", "false", "False"}:
            return
        policy = resolve_q8_mmq_prefill_policy("gguf_ud_q4_k_xl")
        if policy is None:
            return
        from hipengine.kernels.hip_gfx1100.quant.gguf_q8_0_mmq_prefill import (
            build_gguf_q8_0_mmq_prefill,
            q8_mmq_d4x3_nbytes,
        )

        rows_cap = min(int(self.prefill_chunk_size), int(policy.max_rows))
        hidden_cap = max(int(hidden) for hidden, _ in policy.min_rows)
        self._q8_mmq_policy = policy
        self._q8_mmq_library = build_gguf_q8_0_mmq_prefill(load=True)
        workspace = malloc(
            q8_mmq_d4x3_nbytes(rows_cap, hidden_cap), runtime=self.runtime
        )
        risk_count = malloc(DType.INT32.itemsize, runtime=self.runtime)
        risk_indices = malloc(
            policy.risk_indices_nbytes(rows_cap), runtime=self.runtime
        )
        self._q8_mmq_buffers = (workspace, risk_count, risk_indices)
        self._buffers.extend(self._q8_mmq_buffers)

    def rebind_resident_weights(self) -> None:
        """Bind every layer to the resident set's current weight objects.

        The layer bindings hold weight objects, not slots, so a weight
        published through ``swap_weight`` after construction (a new
        allocation stage) is only dispatched once the layers are rebound.
        """

        resident = self.resident
        self.gdn_bindings = {
            layer: bind_qwen4_exp_gdn_layer(resident, layer)
            for layer, kind in enumerate(self.config.layer_types)
            if kind == "gdn"
        }
        self.qsa_bindings = {
            layer: bind_qwen4_exp_qsa_layer(resident, layer)
            for layer, kind in enumerate(self.config.layer_types)
            if kind == "qsa"
        }

    def configure_mmq_prefill_resources(self) -> None:
        """Allocate profile-selected MMQ resources after cold-path binding."""

        if self.closed:
            raise RuntimeError("Qwen4Exp runner is closed")
        self._configure_q8_mmq_prefill_resources()
        self._configure_q8_mmq_weight_sidecars()

    def _q8_mmq_prefill_context(self):
        """Expose the guarded Q8 MMQ session while a chunked prefill runs."""

        enabled = os.environ.get(
            "HIPENGINE_QWEN4_EXP_Q8_MMQ_PREFILL", "0"
        ) not in {"", "0", "false", "False"}
        workspace, risk_count, risk_indices = (
            self._q8_mmq_buffers
            if enabled and self._q8_mmq_buffers
            else (None, None, None)
        )
        prepacked = None
        if enabled and os.environ.get("HIPENGINE_QWEN4_EXP_Q8_MMQ_PREPACK", "0") == "1":
            if self._q8_mmq_weight_sidecars is None:
                raise RuntimeError("Q8 MMQ prepacked route requires prepared sidecar ownership")
            prepacked = self._q8_mmq_weight_sidecars.mapping
        return self._q8_mmq_scopes(
            q8_mmq_prefill_session(
                workspace_ptr=0 if workspace is None else workspace.ptr,
                workspace_nbytes=0 if workspace is None else workspace.nbytes,
                risk_count_ptr=0 if risk_count is None else risk_count.ptr,
                risk_count_nbytes=0 if risk_count is None else risk_count.nbytes,
                risk_indices_ptr=0 if risk_indices is None else risk_indices.ptr,
                risk_indices_nbytes=0 if risk_indices is None else risk_indices.nbytes,
                policy=(
                    _qwen4_exp_q8_mmq_policy(self._q8_mmq_policy)
                    if enabled
                    else None
                ),
                library=self._q8_mmq_library if enabled else None,
                prepacked_weights=prepacked,
            ),
            token64=bool(self.dense_mmq_token64_enabled),
            raw_vector=bool(self.dense_mmq_raw_vector_enabled),
            vec4=bool(self.dense_mmq_vec4_enabled),
        )

    @contextlib.contextmanager
    def _q8_mmq_scopes(
        self,
        session_scope,
        *,
        token64: bool,
        raw_vector: bool,
        vec4: bool,
    ):
        """Bind this runner's MMQ lane decisions around the workspace it owns.

        Both scopes open on ``__enter__`` and unwind in reverse order, so a
        context that is constructed and dropped leaves no state behind, and an
        exception inside the body restores both.
        """

        with session_scope, q8_mmq_lane_session(
            token64=token64, raw_vector=raw_vector, vec4=vec4,
        ):
            yield

    def target_verify_output(self, rows: int) -> Qwen4ExpTargetVerifyOutput:
        """Return lazily allocated rows<=8 verifier output storage."""

        self._require_open()
        count = int(rows)
        capacity = min(8, self.max_sequence_length)
        if not 1 <= count <= capacity:
            raise ValueError(f"Qwen4Exp target verify rows must be in 1..{capacity}")
        if self._target_verify_output is None:
            cfg = self.config
            self._target_verify_output = Qwen4ExpTargetVerifyOutput.allocate(
                rows=capacity,
                branches=cfg.residual_branch_count,
                hidden=cfg.hidden_size,
                low_rank=cfg.residual_low_rank,
                vocab=cfg.vocab_size,
                runtime=self.runtime,
            )
        self._target_verify_output.require_rows(count)
        return self._target_verify_output

    @property
    def token_id_buffer(self) -> DeviceBuffer:
        return self._buffers[0]

    @property
    def embedding_buffer(self) -> DeviceBuffer:
        return self._buffers[1]

    @property
    def ple_embedding_buffer(self) -> DeviceBuffer:
        return self._buffers[2]

    @property
    def logits_buffer(self) -> DeviceBuffer:
        return self._buffers[3]

    @property
    def argmax_block_values(self) -> DeviceBuffer:
        return self._buffers[6]

    @property
    def argmax_block_indices(self) -> DeviceBuffer:
        return self._buffers[7]

    @property
    def argmax_value(self) -> DeviceBuffer:
        return self._buffers[8]

    @property
    def last_target_hidden(self) -> DeviceBuffer:
        """Authoritative pre-final-mix widened BF16 target row for MTP."""

        return self._buffers[4]

    @property
    def rope_position_buffer(self) -> DeviceBuffer:
        return self._buffers[5]

    @property
    def prefill_token_ids(self) -> DeviceBuffer:
        return self._prefill_buffers[0]

    @property
    def prefill_embeddings(self) -> DeviceBuffer:
        return self._prefill_buffers[1]

    @property
    def prefill_ple_embeddings(self) -> DeviceBuffer:
        return self._prefill_buffers[2]

    @property
    def prefill_residual(self) -> DeviceBuffer:
        return self._prefill_buffers[3]

    @property
    def prefill_rope_positions(self) -> DeviceBuffer:
        return self._prefill_buffers[4]

    def verify_target_block_serial_exact(
        self,
        input_token_ids: Sequence[int],
        *,
        capture_logits: bool = True,
        capture_hidden_seeds: bool = True,
    ) -> Qwen4ExpTargetVerifyResult:
        """Execute the rows<=8 verifier oracle through serial target steps."""

        self._require_open()
        tokens = tuple(int(token) for token in input_token_ids)
        if not 1 <= len(tokens) <= 8:
            raise ValueError("Qwen4Exp target verify rows must be in 1..8")
        results = [
            self.step(
                token,
                capture_hidden_seed=capture_hidden_seeds,
                capture_logits=capture_logits,
                capture_target_hidden=True,
            )
            for token in tokens
        ]
        return Qwen4ExpTargetVerifyResult(
            token_ids=tuple(int(result.token_id) for result in results),
            logits=(
                tuple(np.asarray(result.logits, dtype=np.float32) for result in results)
                if capture_logits
                else None
            ),
            hidden_seeds=(
                np.concatenate(
                    [
                        np.asarray(result.hidden_seeds, dtype=np.float32)
                        for result in results
                    ],
                    axis=0,
                )
                if capture_hidden_seeds
                else None
            ),
        )

    def verify_target_block_deferred_head(
        self,
        input_token_ids: Sequence[int],
    ) -> Qwen4ExpTargetVerifyResult:
        """Run serial target bodies and score their captured rows together."""

        self._require_open()
        # The per-row bodies are serial decode, but the deferred head read is
        # one rows>1 pass over the captured rows, so this entry owns the plan's
        # Q8_0 tile pair for its whole body.
        with _qwen4_exp_q8_tile_scope(self):
            return self._deferred_head_rows(input_token_ids)

    def _deferred_head_rows(
        self,
        input_token_ids: Sequence[int],
    ) -> Qwen4ExpTargetVerifyResult:
        tokens = tuple(int(token) for token in input_token_ids)
        output = self.target_verify_output(len(tokens))
        cfg = self.config
        row_bytes = cfg.residual_width * DType.BF16.itemsize
        hidden_rows: list[np.ndarray] = []
        for row, token in enumerate(tokens):
            result = self.step(
                token,
                capture_hidden_seed=True,
                capture_logits=False,
                capture_target_hidden=True,
                deferred_head_output_ptr=output.residual_rows.ptr + row * row_bytes,
            )
            assert result.hidden_seeds is not None
            hidden_rows.append(result.hidden_seeds)
        head = run_qwen4_exp_gr_read(
            output.residual_rows.ptr,
            self.resident.weight("root.head_hc_norm").allocation("raw").tensor.ptr,
            self.resident.weight("root.head_hc_down"),
            self.resident.weight("root.head_hc_up"),
            None,
            output.head_scratch,
            rows=len(tokens),
            branches=cfg.residual_branch_count,
            hidden=cfg.hidden_size,
            low_rank=cfg.residual_low_rank,
            runtime=self.runtime,
        )
        launch_gguf_linear(
            self.resident.weight("root.lm_head"),
            head.mixed.ptr,
            output.logits_rows.ptr,
            len(tokens),
            cfg.hidden_size,
            cfg.vocab_size,
            activation_dtype=GGUF_ACTIVATION_F32,
            output_dtype=GGUF_OUTPUT_F32,
            runtime=self.runtime,
        )
        logits = np.empty((len(tokens), cfg.vocab_size), dtype=np.float32)
        copy_device_to_host(
            host_array_ptr(logits), output.logits_rows, logits.nbytes, runtime=self.runtime
        )
        return Qwen4ExpTargetVerifyResult(
            token_ids=tuple(int(token) for token in np.argmax(logits, axis=1)),
            logits=tuple(row for row in logits),
            hidden_seeds=np.concatenate(hidden_rows, axis=0),
        )

    def _state_row_strides(self) -> tuple[int, int]:
        cfg = self.config
        matrix = (
            cfg.gdn_time_step_rank * cfg.gdn_state_size * cfg.gdn_state_size
            * DType.FP32.itemsize
        )
        conv = (
            (2 * cfg.gdn_group_count * cfg.gdn_state_size + cfg.gdn_inner_size)
            * cfg.gdn_conv_kernel * DType.FP32.itemsize
        )
        return matrix, conv

    def _verify_rows_workspace(self, rows: int) -> Mapping[str, DeviceBuffer]:
        if rows <= self._verify_rows_capacity:
            return self._verify_rows_buffers
        assert self.state is not None
        cfg = self.config
        matrix, conv = self._state_row_strides()
        gdn_layers = len(self.gdn_bindings)
        sizes = {
            "gdn_matrix_rows": gdn_layers * rows * matrix,
            "gdn_conv_rows": gdn_layers * rows * conv,
            "ple_conv": self.state.ple_conv.nbytes,
            "ple_out": rows * cfg.residual_width * DType.FP32.itemsize,
            "argmax_ids": rows * DType.INT64.itemsize,
        }
        for buffer in self._verify_rows_buffers.values():
            free(buffer, runtime=self.runtime)
        self._verify_rows_buffers = {}
        self._verify_rows_capacity = 0
        for name, nbytes in sizes.items():
            self._verify_rows_buffers[name] = malloc(nbytes, runtime=self.runtime)
        self._verify_rows_capacity = rows
        return self._verify_rows_buffers

    def verify_target_block_state_rows(
        self,
        input_token_ids: Sequence[int],
        *,
        capture_logits: bool = False,
    ) -> Qwen4ExpTargetVerifyResult:
        """Verify 2..8 rows in one prefill-shaped pass, keeping each row's state.

        The pass advances the target by every input row, like
        :meth:`verify_target_block_batched_body`, but the GDN recurrence and
        conv also write the state after every row. :meth:`commit_target_block_rows`
        must follow: it keeps the first ``consumed`` rows by copying that
        row's GDN and conv states back and rewinding the attention, index,
        PLE and position cursors, with no replay. Per-row logits come from
        prefill-shaped kernels (production arithmetic), not from the serial
        decode step.
        """

        self._require_open()
        # The pass launches rows>1 dense projections both in the prefill body
        # and in the deferred head read, so it owns the plan's Q8_0 tile pair
        # for the whole entry rather than for the prefill chunk alone.
        with _qwen4_exp_q8_tile_scope(self):
            return self._verify_state_rows(
                input_token_ids, capture_logits=capture_logits
            )

    def _verify_state_rows(
        self,
        input_token_ids: Sequence[int],
        *,
        capture_logits: bool = False,
    ) -> Qwen4ExpTargetVerifyResult:
        if self._pending_verify_block is not None:
            raise RuntimeError("commit the previous Qwen4Exp verify block first")
        tokens = tuple(int(token) for token in input_token_ids)
        count = len(tokens)
        if not 2 <= count <= 8:
            raise ValueError("Qwen4Exp state-row verify needs 2..8 rows")
        assert self.state is not None
        cfg = self.config
        buffers = self._verify_rows_workspace(count)
        matrix, conv = self._state_row_strides()
        capture: dict[int, tuple[int, int]] = {}
        for binding in self.gdn_bindings.values():
            index = int(binding.gdn_state_index)
            capture[self.state.gdn_matrix.ptr + index * matrix] = (
                buffers["gdn_matrix_rows"].ptr + index * count * matrix,
                buffers["gdn_conv_rows"].ptr + index * count * conv,
            )
        self.runtime.memcpy_async(
            buffers["ple_conv"].ptr, self.state.ple_conv.ptr,
            self.state.ple_conv.nbytes, HipMemcpyKind.DEVICE_TO_DEVICE, 0,
        )
        start = int(self.position)
        hash_states = dict(self._ple_hash_states)
        output = self.target_verify_output(count)
        token = _GDN_STATE_ROWS_CAPTURE.set(MappingProxyType(capture))
        try:
            last_residual_ptr, hidden_rows = self._prefill_chunk(
                tokens, capture_hidden_seeds=True
            )
        finally:
            _GDN_STATE_ROWS_CAPTURE.reset(token)
        self._pending_verify_block = (tokens, start, hash_states, count)
        row_bytes = cfg.residual_width * DType.BF16.itemsize
        head = run_qwen4_exp_gr_read(
            int(last_residual_ptr) - (count - 1) * row_bytes,
            self.resident.weight("root.head_hc_norm").allocation("raw").tensor.ptr,
            self.resident.weight("root.head_hc_down"),
            self.resident.weight("root.head_hc_up"),
            None,
            output.head_scratch,
            rows=count,
            branches=cfg.residual_branch_count,
            hidden=cfg.hidden_size,
            low_rank=cfg.residual_low_rank,
            runtime=self.runtime,
        )
        launch_gguf_linear(
            self.resident.weight("root.lm_head"),
            head.mixed.ptr,
            output.logits_rows.ptr,
            count,
            cfg.hidden_size,
            cfg.vocab_size,
            activation_dtype=GGUF_ACTIVATION_F32,
            output_dtype=GGUF_OUTPUT_F32,
            runtime=self.runtime,
        )
        assert hidden_rows is not None
        if capture_logits:
            logits = np.empty((count, cfg.vocab_size), dtype=np.float32)
            copy_device_to_host(
                host_array_ptr(logits), output.logits_rows, logits.nbytes,
                runtime=self.runtime,
            )
            return Qwen4ExpTargetVerifyResult(
                token_ids=tuple(int(token) for token in np.argmax(logits, axis=1)),
                logits=tuple(row for row in logits),
                hidden_seeds=np.asarray(hidden_rows, dtype=np.float32),
            )
        # The decode step's device argmax per row; only the ids come back.
        ids_buffer = buffers["argmax_ids"]
        row_logits = cfg.vocab_size * DType.FP32.itemsize
        for row in range(count):
            argmax_f32(
                output.logits_rows.ptr + row * row_logits,
                self.argmax_block_values.ptr,
                self.argmax_block_indices.ptr,
                ids_buffer.ptr + row * DType.INT64.itemsize,
                self.argmax_value.ptr,
                cfg.vocab_size,
                runtime=self.runtime,
            )
        ids = np.empty(count, dtype=np.int64)
        copy_device_to_host(host_array_ptr(ids), ids_buffer, ids.nbytes, runtime=self.runtime)
        return Qwen4ExpTargetVerifyResult(
            token_ids=tuple(int(token) for token in ids),
            logits=(),
            hidden_seeds=np.asarray(hidden_rows, dtype=np.float32),
        )

    def commit_target_block_rows(self, consumed: int) -> None:
        """Keep the first ``consumed`` rows of the pending state-row verify."""

        self._require_open()
        if self._pending_verify_block is None:
            raise RuntimeError("no Qwen4Exp verify block is pending")
        tokens, start, hash_states, count = self._pending_verify_block
        consumed = int(consumed)
        if not 1 <= consumed <= count:
            raise ValueError("consumed rows must be in 1..block rows")
        self._pending_verify_block = None
        if consumed == count:
            return
        assert self.state is not None and self.ple_prefill_scratch is not None
        cfg = self.config
        buffers = self._verify_rows_buffers
        matrix, conv = self._state_row_strides()
        last = consumed - 1
        for binding in self.gdn_bindings.values():
            index = int(binding.gdn_state_index)
            self.runtime.memcpy_async(
                self.state.gdn_matrix.ptr + index * matrix,
                buffers["gdn_matrix_rows"].ptr + (index * count + last) * matrix,
                matrix, HipMemcpyKind.DEVICE_TO_DEVICE, 0,
            )
            self.runtime.memcpy_async(
                self.state.gdn_conv.ptr + index * conv,
                buffers["gdn_conv_rows"].ptr + (index * count + last) * conv,
                conv, HipMemcpyKind.DEVICE_TO_DEVICE, 0,
            )
        # The PLE conv history a prefill of the consumed rows leaves: restore
        # the history from before the block and rerun the conv on those rows
        # (its history update only copies inputs).
        self.runtime.memcpy_async(
            self.state.ple_conv.ptr, buffers["ple_conv"].ptr,
            self.state.ple_conv.nbytes, HipMemcpyKind.DEVICE_TO_DEVICE, 0,
        )
        for layer in cfg.ple_layers:
            qwen4_exp_ple_dilated_depthwise_conv_f32(
                self.ple_prefill_scratch.normalized.ptr,
                self.resident.weight(f"layers.{layer}.ple_conv1d")
                .allocation("raw").tensor.ptr,
                self.state.ple_conv.ptr,
                buffers["ple_out"].ptr,
                consumed,
                cfg.residual_width,
                cfg.ple_conv_kernel,
                cfg.ple_ngram_size,
                runtime=self.runtime,
            )
        position = start + consumed
        _, accepted_hash_states = ple_hash_rows(
            list(tokens[:consumed]),
            positions=list(range(start, position)),
            sequence_ids=[0] * consumed,
            states=hash_states,
            eos_token_id=cfg.ple_eos_token_id,
            layer_multipliers=cfg.ple_layer_multipliers,
            head_offsets=cfg.ple_head_offsets,
            head_vocab_sizes=cfg.ple_head_vocab_sizes,
            heads_per_ngram=cfg.ple_heads_per_ngram,
            ngram_size=cfg.ple_ngram_size,
        )
        self.adopt_prefix_cursor(position, accepted_hash_states)

    def verify_target_block_with_acceptance(
        self,
        input_token_ids: Sequence[int],
        candidate_token_ids: Sequence[int],
    ) -> Qwen4ExpTargetAcceptResult:
        """Verify a fixed chain and replay only the committed rejection prefix."""

        inputs = tuple(int(token) for token in input_token_ids)
        candidates = tuple(int(token) for token in candidate_token_ids)
        if not inputs or len(inputs) != len(candidates) or len(inputs) > 8:
            raise ValueError(
                "Qwen4Exp target verify inputs/candidates must have equal rows in 1..8"
            )
        transaction = self.begin_device_transaction(reuse_snapshot=True)
        try:
            block = self.verify_target_block_deferred_head(inputs)
        except Exception:
            self.rollback_device_transaction(transaction)
            raise
        accepted = 0
        for truth, candidate in zip(block.token_ids, candidates, strict=True):
            if truth != candidate:
                break
            accepted += 1
        consumed = min(accepted + 1, len(inputs))
        if consumed == len(inputs):
            self.commit_device_transaction(transaction)
            committed = block
            replayed = False
        else:
            self.rollback_device_transaction(transaction)
            committed = self.verify_target_block_serial_exact(
                inputs[:consumed],
                capture_logits=True,
                capture_hidden_seeds=True,
            )
            replayed = True
        return Qwen4ExpTargetAcceptResult(
            verify=committed,
            accepted=accepted,
            consumed=consumed,
            replayed=replayed,
        )

    def begin_device_transaction(
        self, *, reuse_snapshot: bool = False
    ) -> Qwen4ExpRunnerDeviceTransaction:
        """Capture mutable state and all append-only cursor ownership."""

        self._require_open()
        assert self.state is not None
        if reuse_snapshot:
            if self._device_transaction_lease:
                raise RuntimeError("Qwen4Exp reusable device snapshot is already leased")
            if self._device_transaction_snapshot is None:
                self._device_transaction_snapshot = self.state.device_snapshot()
            else:
                self.state.device_snapshot(self._device_transaction_snapshot)
            decode_state = self._device_transaction_snapshot
            self._device_transaction_lease = True
        else:
            decode_state = self.state.device_snapshot()
        return Qwen4ExpRunnerDeviceTransaction(
            decode_state=decode_state,
            position=int(self.position),
            ple_hash_states=MappingProxyType(dict(self._ple_hash_states)),
            reusable_snapshot=bool(reuse_snapshot),
        )

    def rollback_device_transaction(
        self, transaction: Qwen4ExpRunnerDeviceTransaction
    ) -> None:
        """Restore device state, QSA cursors, and host PLE hash state."""

        self._require_open()
        transaction.require_active()
        assert self.state is not None
        self.state.restore_device_snapshot(transaction.decode_state)
        self.adopt_prefix_cursor(transaction.position, transaction.ple_hash_states)
        transaction.rolled_back = True
        transaction.close()
        if transaction.reusable_snapshot:
            self._device_transaction_lease = False

    def commit_device_transaction(
        self, transaction: Qwen4ExpRunnerDeviceTransaction
    ) -> None:
        """Keep current state and release the rollback snapshot."""

        self._require_open()
        transaction.require_active()
        transaction.committed = True
        transaction.close()
        if transaction.reusable_snapshot:
            self._device_transaction_lease = False

    def snapshot(self) -> Qwen4ExpRunnerSnapshot:
        """Capture mutable non-append state plus the shared KV/index cursor."""

        self._require_open()
        assert self.state is not None
        return Qwen4ExpRunnerSnapshot(
            self.state.snapshot(),
            self.position,
            MappingProxyType(dict(self._ple_hash_states)),
        )

    def restore(self, snapshot: Qwen4ExpRunnerSnapshot) -> None:
        """Rollback request-local state; append-only K/V is bounded by the cursor."""

        self._require_open()
        assert self.state is not None
        position = int(snapshot.position)
        if position < 0 or position > self.max_sequence_length:
            raise ValueError("Qwen4Exp runner snapshot position exceeds capacity")
        self.state.restore(snapshot.decode_state)
        self.adopt_prefix_cursor(position, snapshot.ple_hash_states)

    def adopt_prefix_cursor(
        self, position: int, ple_hash_states: Mapping[int, PLEHashState]
    ) -> None:
        """Point the KV/index cursors and PLE history at ``position``.

        The caller has already written the K/V rows, index keys, pooled index
        blocks and recurrent state that describe that position. The attention
        and index states bound-check the cursor.

        With shared position/context staging the QSA cursor moves with one
        interleaved copy instead of two 8-byte copies per state. That copy
        writes every state at once, so each state's closed flag and position
        bound is checked first and a refusal leaves the device untouched.
        """

        self._require_open()
        position = int(position)
        cursor = max(position - 1, 0)
        compact = (
            self._shared_position_context is not None
            and self._shared_position_context_host is not None
        )
        if compact:
            for attention in self.attention_states:
                if attention.closed:
                    raise RuntimeError("Qwen4Exp attention state is closed")
                if not 0 <= cursor < attention.max_positions:
                    raise ValueError("attention position exceeds state capacity")
            for index in self.index_states:
                if index.closed:
                    raise RuntimeError("QSA index state is closed")
                if position < 0 or position > index.capacity:
                    raise ValueError("QSA index restore count exceeds capacity")
            self._prepare_decode_positions(cursor)
        else:
            for attention in self.attention_states:
                attention.set_position(cursor)
        for index in self.index_states:
            index.restore_count(position)
        self._ple_hash_states = dict(ple_hash_states)
        self.position = position

    def reset(self) -> None:
        self._require_open()
        pending = getattr(self, "_pending_prefill_append", None)
        if pending is not None:
            context, steps, _ = pending
            context.run(steps.close)
            self._pending_prefill_append = None
        assert self.state is not None
        self.state.zero()
        bounded = _qwen4_exp_bounded_kv_reset_enabled()
        for attention in self.attention_states:
            if bounded:
                # RT-4: clear only what this request could have written.  Sound by
                # the invariant on ``written_cache_bytes`` - allocation zeroes the
                # whole cache and writes only ever land below ``position`` - so the
                # bytes above the bound are already zero and clearing them is pure
                # waste (29.182 ms at 128K, 99.6% of it spent on untouched bytes).
                attention.clear_written_cache()
            else:
                self.runtime.memset(attention.key_cache.ptr, 0, attention.key_cache.nbytes)
                self.runtime.memset(attention.value_cache.ptr, 0, attention.value_cache.nbytes)
            attention.set_position(0)
        for index in self.index_states:
            index.reset()
        self._ple_hash_states = {}
        self.position = 0

    @property
    def ple_hash_states(self) -> Mapping[int, PLEHashState]:
        """Request-local PLE n-gram history, keyed by sequence id.

        Read-only copy for a hand-off owner (the batch runner's
        ``import_prefilled``) that must not alias this runner's history.
        """

        return MappingProxyType(dict(self._ple_hash_states))

    def _finish_lm_head(
        self,
        *,
        capture_logits: bool,
        hidden_seeds: np.ndarray | None,
    ) -> Qwen4ExpTokenResult:
        if capture_logits:
            self.runtime.device_synchronize()
            logits = np.empty(self.config.vocab_size, dtype=np.float32)
            copy_device_to_host(
                host_array_ptr(logits), self.logits_buffer, runtime=self.runtime
            )
            return Qwen4ExpTokenResult(
                int(np.argmax(logits)), logits, hidden_seeds=hidden_seeds
            )
        argmax_f32(
            self.logits_buffer.ptr,
            self.argmax_block_values.ptr,
            self.argmax_block_indices.ptr,
            self.token_id_buffer.ptr,
            self.argmax_value.ptr,
            self.config.vocab_size,
            runtime=self.runtime,
        )
        token = np.empty(1, dtype=np.int64)
        copy_device_to_host(
            host_array_ptr(token),
            self.token_id_buffer,
            token.nbytes,
            runtime=self.runtime,
        )
        return Qwen4ExpTokenResult(int(token[0]), None, hidden_seeds=hidden_seeds)

    def _prepare_decode_positions(self, position: int) -> None:
        """Batched set_position for every QSA state via one 8B-per-state H2D copy."""
        if self._shared_position_context is None or self._shared_position_context_host is None:
            raise RuntimeError("batched QSA positions requested without shared staging")
        host = self._shared_position_context_host
        host[0::2] = position
        host[1::2] = position + 1
        copy_host_array_to_device(
            self._shared_position_context,
            host,
            runtime=self.runtime,
        )
        for state in self.attention_states:
            state.position_host[0] = position
            state.context_host[0] = position + 1

    def step(
        self,
        token_id: int,
        *,
        capture_hidden_seed: bool = False,
        capture_logits: bool = True,
        capture_target_hidden: bool = True,
        token_id_resident: bool = False,
        rope_positions: tuple[int, int, int] | None = None,
        deferred_head_output_ptr: int | None = None,
    ) -> Qwen4ExpTokenResult:
        self._require_open()
        if self.position >= self.max_sequence_length:
            raise ValueError("Qwen4Exp dense runner sequence capacity exceeded")
        cfg = self.config
        assert self.state is not None
        assert self.gdn_scratch is not None
        assert self.qsa_scratch is not None
        assert self.ple_scratch is not None
        assert self.head_scratch is not None
        token_host = np.asarray([int(token_id)], dtype=np.int64)
        rope_positions_ptr = None
        if rope_positions is not None:
            rope_host = np.ascontiguousarray(rope_positions, dtype=np.int64)
            if rope_host.shape != (3,) or np.any(rope_host < 0):
                raise ValueError(
                    "Qwen4Exp decode MRoPE positions must be three nonnegative values"
                )
            copy_host_array_to_device(
                self.rope_position_buffer,
                rope_host,
                rope_host.nbytes,
                runtime=self.runtime,
            )
            rope_positions_ptr = self.rope_position_buffer.ptr
        if token_host[0] < 0 or token_host[0] >= cfg.vocab_size:
            raise ValueError("token_id is outside Qwen4Exp vocabulary")
        if not token_id_resident:
            copy_host_array_to_device(
                self.token_id_buffer,
                token_host,
                runtime=self.runtime,
            )
        token_weight = self.resident.weight("root.token_embedding")
        embedding = resolve(
            backend=self.backend,
            layer="embedding",
            quant=token_weight.spec.quant_key,
            variant="lookup_bf16_out",
        )
        embedding(
            self.token_id_buffer.ptr,
            token_weight.allocation("raw").tensor.ptr,
            self.embedding_buffer.ptr,
            1,
            cfg.hidden_size,
            cfg.vocab_size,
            runtime=self.runtime,
        )
        qwen4_exp_repeat_bf16_branches(
            self.embedding_buffer.ptr,
            self.state.residual.ptr,
            cfg.residual_branch_count,
            cfg.hidden_size,
            runtime=self.runtime,
        )
        # The PLE row is first read at the first PLE layer. With a pinned
        # staging ring its host hash, gather and copy run after the layers
        # before it are enqueued, so they overlap GPU work; the copy is then
        # asynchronous on the same stream (bit-identical).
        overlap_ple = (
            self.resident.ple_staging.pinned
            and min(cfg.ple_layers, default=0) > 0
            and os.environ.get("HIPENGINE_QWEN4_EXP_PLE_OVERLAP", "1")
            not in {"", "0", "false", "False"}
        )

        def stage_ple() -> None:
            ple_rows, self._ple_hash_states = ple_hash_rows(
                [int(token_id)],
                positions=[self.position],
                sequence_ids=[0],
                states=self._ple_hash_states,
                eos_token_id=cfg.ple_eos_token_id,
                layer_multipliers=cfg.ple_layer_multipliers,
                head_offsets=cfg.ple_head_offsets,
                head_vocab_sizes=cfg.ple_head_vocab_sizes,
                heads_per_ngram=cfg.ple_heads_per_ngram,
                ngram_size=cfg.ple_ngram_size,
            )
            staged = self.resident.ple_staging.stage(ple_rows[0]).reshape(1, cfg.hidden_size)
            ple_telemetry = self.resident.ple_table.telemetry() is not None
            ple_h2d_started = time.perf_counter_ns() if ple_telemetry else 0
            if overlap_ple:
                self.runtime.memcpy_async(
                    self.ple_embedding_buffer.ptr,
                    host_array_ptr(staged),
                    staged.nbytes,
                    HipMemcpyKind.HOST_TO_DEVICE,
                    0,
                )
            else:
                copy_host_array_to_device(
                    self.ple_embedding_buffer,
                    staged,
                    staged.nbytes,
                    runtime=self.runtime,
                )
            if ple_h2d_started:
                self.resident.ple_staging.record_h2d(
                    nbytes=staged.nbytes,
                    wall_ns=time.perf_counter_ns() - ple_h2d_started,
                )

        if not overlap_ple:
            stage_ple()
        residual_ptr = self.state.residual.ptr
        gdn_conv_row_bytes = (
            (2 * cfg.gdn_group_count * cfg.gdn_state_size + cfg.gdn_inner_size)
            * cfg.gdn_conv_kernel * 4
        )
        gdn_matrix_row_bytes = (
            cfg.gdn_time_step_rank * cfg.gdn_state_size * cfg.gdn_state_size * 4
        )
        for layer, kind in enumerate(cfg.layer_types):
            if layer == 0 and self._batched_position:
                self._prepare_decode_positions(self.position)
            if layer in cfg.ple_layers:
                if overlap_ple and layer == min(cfg.ple_layers):
                    stage_ple()
                layer_prefix = f"layers.{layer}."
                residual_ptr = run_qwen4_exp_ple(
                    residual_ptr,
                    self.ple_embedding_buffer.ptr,
                    {
                        "ple_key": self.resident.weight(layer_prefix + "ple_key"),
                        "ple_value": self.resident.weight(layer_prefix + "ple_value"),
                    },
                    norm_key_ptr=self.resident.weight(
                        layer_prefix + "ple_norm_key"
                    ).allocation("raw").tensor.ptr,
                    norm_query_ptr=self.resident.weight(
                        layer_prefix + "ple_norm_query"
                    ).allocation("raw").tensor.ptr,
                    norm_conv_ptr=self.resident.weight(
                        layer_prefix + "ple_norm_conv"
                    ).allocation("raw").tensor.ptr,
                    conv_weight_ptr=self.resident.weight(
                        layer_prefix + "ple_conv1d"
                    ).allocation("raw").tensor.ptr,
                    conv_history_ptr=self.state.ple_conv.ptr,
                    scratch=self.ple_scratch,
                    rows=1, branches=cfg.residual_branch_count, hidden=cfg.hidden_size,
                    conv_kernel=cfg.ple_conv_kernel, dilation=cfg.ple_ngram_size,
                    runtime=self.runtime,
                ).ptr
            if kind == "gdn":
                binding = self.gdn_bindings[layer]
                residual_ptr = run_qwen4_exp_gdn_layer(
                    residual_ptr,
                    binding,
                    conv_state_ptr=(
                        self.state.gdn_conv.ptr
                        + binding.gdn_state_index * gdn_conv_row_bytes
                    ),
                    recurrent_state_ptr=(
                        self.state.gdn_matrix.ptr
                        + binding.gdn_state_index * gdn_matrix_row_bytes
                    ),
                    scratch=self.gdn_scratch,
                    q4_bundle_prefill=self.q4_bundle_prefill,
                    rows=1, branches=cfg.residual_branch_count, hidden=cfg.hidden_size,
                    low_rank=cfg.residual_low_rank,
                    num_k_heads=cfg.gdn_group_count,
                    num_v_heads=cfg.gdn_time_step_rank,
                    head_dim=cfg.gdn_state_size,
                    conv_kernel=cfg.gdn_conv_kernel,
                    ffn=cfg.expert_feed_forward_length,
                    experts=cfg.expert_count,
                    top_k=cfg.expert_used_count,
                    runtime=self.runtime,
                    q5k_row4_prefill=self.q5k_row4_prefill,
                    moe_graph_cache=self.moe_graph_cache,
                    moe_graph_key=(
                        "gdn", layer,
                        os.environ.get("HIPENGINE_QWEN4_EXP_Q4_DP4A64", ""),
                        os.environ.get(
                            "HIPENGINE_QWEN4_EXP_Q4_DP4A64_LAYERS", ""
                        ),
                    ),
                    layer_graph_cache=self.layer_graph_cache,
                    layer_graph_key=(
                        "gdn", layer,
                        os.environ.get("HIPENGINE_QWEN4_EXP_Q4_DP4A64", ""),
                        os.environ.get(
                            "HIPENGINE_QWEN4_EXP_Q4_DP4A64_LAYERS", ""
                        ),
                    ),
                    q51_down_m1=self.q51_down_m1,
                    q51_pair_prefill=self.q51_pair_prefill,
                    q51_fold128_prefill=self.q51_fold128_prefill,
                    q51_fold_pair_prefill=self.q51_fold_pair_prefill,
                    q51_register_cache=self.q51_register_cache,
                    q51_row_publish=self.q51_row_publish,
                    q51_exact_grouped_down=self.q51_exact_grouped_down,
                    q51_wmma_down=self.q51_wmma_down,
                    q51_out8=self.q51_out8,
                    q4_exact_grouped_gate=self.q4_exact_grouped_gate,
                    q4_exact_grouped_all=self.q4_exact_grouped_all,
                    q4_out4=self.q4_out4,
                    q4_tile_m=self.q4_tile_m,
                    q4_tile_n=self.q4_tile_n,
                    forkb_grouped_down=self.forkb_grouped_down,
                    q8_down_row4_prefill=self.q8_down_row4_prefill,
                    q8_down_bundle_prefill=self.q8_down_bundle_prefill,
                    q8_down_register=self.q8_down_register,
                    q8_mapped_down=self.q8_mapped_down,
                ).ptr
            else:
                binding = self.qsa_bindings[layer]
                residual_ptr = run_qwen4_exp_dense_qsa_layer(
                    residual_ptr,
                    binding,
                    q4_bundle_prefill=self.q4_bundle_prefill,
                    attention_state=self.attention_states[binding.qsa_state_index],
                    index_state=self.index_states[binding.qsa_state_index],
                    scratch=self.qsa_scratch,
                    position=self.position,
                    position_prepared=self._batched_position,
                    rows=1, branches=cfg.residual_branch_count, hidden=cfg.hidden_size,
                    low_rank=cfg.residual_low_rank,
                    query_heads=cfg.attention_head_count,
                    kv_heads=cfg.attention_kv_head_count,
                    head_dim=cfg.attention_key_length,
                    rotary_dim=cfg.rope_dimension_count,
                    theta=cfg.rope_freq_base,
                    index_heads=cfg.indexer_head_count,
                    index_dim=cfg.indexer_key_length,
                    index_rotary_dim=cfg.rope_dimension_count,
                    rope_positions_ptr=rope_positions_ptr,
                    ffn=cfg.expert_feed_forward_length,
                    experts=cfg.expert_count,
                    top_k=cfg.expert_used_count,
                    runtime=self.runtime,
                    q5k_row4_prefill=self.q5k_row4_prefill,
                    moe_graph_cache=self.moe_graph_cache,
                    moe_graph_key=(
                        "qsa", layer,
                        os.environ.get("HIPENGINE_QWEN4_EXP_Q4_DP4A64", ""),
                        os.environ.get(
                            "HIPENGINE_QWEN4_EXP_Q4_DP4A64_LAYERS", ""
                        ),
                    ),
                    cache_only=False,
                    qsa_ordered_decode_v2=self.qsa_ordered_decode_v2,
                    qsa_wave32=self.qsa_wave32,
                    q51_down_m1=self.q51_down_m1,
                    q51_pair_prefill=self.q51_pair_prefill,
                    q51_fold128_prefill=self.q51_fold128_prefill,
                    q51_fold_pair_prefill=self.q51_fold_pair_prefill,
                    q51_register_cache=self.q51_register_cache,
                    q51_row_publish=self.q51_row_publish,
                    q51_exact_grouped_down=self.q51_exact_grouped_down,
                    q51_wmma_down=self.q51_wmma_down,
                    q51_out8=self.q51_out8,
                    q4_exact_grouped_gate=self.q4_exact_grouped_gate,
                    q4_exact_grouped_all=self.q4_exact_grouped_all,
                    q4_out4=self.q4_out4,
                    q4_tile_m=self.q4_tile_m,
                    q4_tile_n=self.q4_tile_n,
                    forkb_grouped_down=self.forkb_grouped_down,
                    q8_down_row4_prefill=self.q8_down_row4_prefill,
                    q8_down_bundle_prefill=self.q8_down_bundle_prefill,
                    q8_down_register=self.q8_down_register,
                    q8_mapped_down=self.q8_mapped_down,
                ).ptr
        if capture_target_hidden:
            self.runtime.memcpy(
                self.last_target_hidden.ptr,
                residual_ptr,
                self.last_target_hidden.nbytes,
                HipMemcpyKind.DEVICE_TO_DEVICE,
            )
        hidden_seeds = (
            self._read_hidden_seed_rows(
                self.last_target_hidden.ptr if capture_target_hidden else residual_ptr,
                1,
            )
            if capture_hidden_seed
            else None
        )
        if deferred_head_output_ptr is not None:
            destination = int(deferred_head_output_ptr)
            if destination <= 0:
                raise ValueError("deferred Qwen4Exp head output pointer must be positive")
            self.runtime.memcpy(
                destination,
                residual_ptr,
                cfg.residual_width * DType.BF16.itemsize,
                HipMemcpyKind.DEVICE_TO_DEVICE,
            )
            self.position += 1
            return Qwen4ExpTokenResult(-1, None, hidden_seeds=hidden_seeds)
        head_read = run_qwen4_exp_gr_read(
            residual_ptr,
            self.resident.weight("root.head_hc_norm").allocation("raw").tensor.ptr,
            self.resident.weight("root.head_hc_down"),
            self.resident.weight("root.head_hc_up"),
            None,
            self.head_scratch,
            rows=1, branches=cfg.residual_branch_count, hidden=cfg.hidden_size,
            low_rank=cfg.residual_low_rank, runtime=self.runtime,
        )
        launch_gguf_linear(
            self.resident.weight("root.lm_head"),
            head_read.mixed.ptr,
            self.logits_buffer.ptr,
            1,
            cfg.hidden_size,
            cfg.vocab_size,
            activation_dtype=GGUF_ACTIVATION_F32,
            output_dtype=GGUF_OUTPUT_F32,
            runtime=self.runtime,
        )
        result = self._finish_lm_head(
            capture_logits=capture_logits,
            hidden_seeds=hidden_seeds,
        )
        self.position += 1
        return result

    def _read_hidden_seed_rows(self, source_ptr: int, rows: int) -> np.ndarray:
        count = int(rows)
        if count <= 0 or int(source_ptr) <= 0:
            raise ValueError("target hidden capture requires positive rows and pointer")
        bits = np.empty((count, self.config.residual_width), dtype=np.uint16)
        copy_device_to_host(
            host_array_ptr(bits),
            DeviceBuffer(int(source_ptr), bits.nbytes),
            bits.nbytes,
            runtime=self.runtime,
        )
        return np.ascontiguousarray(
            (bits.astype(np.uint32) << 16).view(np.float32)
        )

    def _prefill_chunk(self, token_ids, **kwargs):
        with _qwen4_exp_q8_tile_scope(self):
            steps = self._prefill_chunk_steps(token_ids, **kwargs)
            while True:
                try:
                    next(steps)
                except StopIteration as done:
                    return done.value

    def _prefill_chunk_steps(
        self,
        token_ids: tuple[int, ...],
        *,
        capture_hidden_seeds: bool = False,
        embedding_overrides: Mapping[int, np.ndarray] | None = None,
        mrope_positions: np.ndarray | None = None,
        yield_layers: bool = False,
    ):
        count = len(token_ids)
        invalidate_f16_mirror()
        if count <= 0 or count > min(self.prefill_chunk_size, self.max_sequence_length):
            raise ValueError("Qwen4Exp prefill chunk has invalid row count")
        if self.position + count > self.max_sequence_length:
            raise ValueError("Qwen4Exp prefill chunk exceeds sequence capacity")
        cfg = self.config
        assert self.state is not None
        assert self.gdn_prefill_scratch is not None
        assert self.qsa_prefill_scratch is not None
        assert self.ple_prefill_scratch is not None
        assert self.qsa_prefill_metadata is not None
        token_host = np.ascontiguousarray(token_ids, dtype=np.int64)
        if np.any(token_host < 0) or np.any(token_host >= cfg.vocab_size):
            raise ValueError("token_id is outside Qwen4Exp vocabulary")
        copy_host_array_to_device(
            self.prefill_token_ids,
            token_host,
            token_host.nbytes,
            runtime=self.runtime,
        )
        rope_positions_ptr = None
        if mrope_positions is not None:
            rope_host = np.ascontiguousarray(mrope_positions, dtype=np.int64)
            if rope_host.shape != (3, count) or np.any(rope_host < 0):
                raise ValueError(
                    "Qwen4Exp prompt MRoPE positions must have shape [3, rows] "
                    "and be nonnegative"
                )
            copy_host_array_to_device(
                self.prefill_rope_positions,
                rope_host,
                rope_host.nbytes,
                runtime=self.runtime,
            )
            rope_positions_ptr = self.prefill_rope_positions.ptr
        token_weight = self.resident.weight("root.token_embedding")
        embedding = resolve(
            backend=self.backend,
            layer="embedding",
            quant=token_weight.spec.quant_key,
            variant="lookup_bf16_out",
        )
        embedding(
            self.prefill_token_ids.ptr,
            token_weight.allocation("raw").tensor.ptr,
            self.prefill_embeddings.ptr,
            count,
            cfg.hidden_size,
            cfg.vocab_size,
            runtime=self.runtime,
        )
        start = self.position
        if embedding_overrides:
            row_nbytes = cfg.hidden_size * DType.BF16.itemsize
            for absolute_position, values in embedding_overrides.items():
                position = int(absolute_position)
                if not start <= position < start + count:
                    continue
                row = np.asarray(values, dtype=np.float32).reshape(-1)
                if row.shape != (cfg.hidden_size,):
                    raise ValueError("Qwen4Exp embedding override must have hidden_size values")
                bits = np.ascontiguousarray(float_array_to_bf16_bits(row), dtype=np.uint16)
                copy_host_array_to_device(
                    DeviceBuffer(
                        self.prefill_embeddings.ptr
                        + (position - start) * row_nbytes,
                        row_nbytes,
                    ),
                    bits,
                    bits.nbytes,
                    runtime=self.runtime,
                )
        qwen4_exp_repeat_bf16_branches(
            self.prefill_embeddings.ptr,
            self.prefill_residual.ptr,
            cfg.residual_branch_count,
            cfg.hidden_size,
            rows=count,
            runtime=self.runtime,
        )
        positions = list(range(start, start + count))
        rows, self._ple_hash_states = ple_hash_rows(
            token_host.tolist(),
            positions=positions,
            sequence_ids=[0] * count,
            states=self._ple_hash_states,
            eos_token_id=cfg.ple_eos_token_id,
            layer_multipliers=cfg.ple_layer_multipliers,
            head_offsets=cfg.ple_head_offsets,
            head_vocab_sizes=cfg.ple_head_vocab_sizes,
            heads_per_ngram=cfg.ple_heads_per_ngram,
            ngram_size=cfg.ple_ngram_size,
        )
        staged = stage_qwen4_exp_ple_rows(
            self.resident.ple_staging,
            rows,
            hidden=cfg.hidden_size,
        )
        ple_telemetry = self.resident.ple_table.telemetry() is not None
        ple_h2d_started = time.perf_counter_ns() if ple_telemetry else 0
        copy_host_array_to_device(
            self.prefill_ple_embeddings,
            staged,
            staged.nbytes,
            runtime=self.runtime,
        )
        if ple_h2d_started:
            # The decode path records its copy; without this the PLE telemetry
            # counts only decode copies and understates the request H2D payload
            # (5 MiB prefill vs 1.25 MiB of decode copies at p512).
            self.resident.ple_staging.record_h2d(
                nbytes=staged.nbytes,
                wall_ns=time.perf_counter_ns() - ple_h2d_started,
            )
        residual_ptr = self.prefill_residual.ptr
        gdn_conv_row_bytes = (
            (2 * cfg.gdn_group_count * cfg.gdn_state_size + cfg.gdn_inner_size)
            * cfg.gdn_conv_kernel
            * DType.FP32.itemsize
        )
        gdn_matrix_row_bytes = (
            cfg.gdn_time_step_rank
            * cfg.gdn_state_size
            * cfg.gdn_state_size
            * DType.FP32.itemsize
        )
        # A layer may hand its final residual write to the next layer's read
        # (bit-identical); it is flushed before anything else reads the
        # residual.
        pending_write = None

        def flush_pending(output_ptr: int) -> None:
            nonlocal pending_write
            if pending_write is not None:
                _qwen4_exp_gr_flush_write(
                    pending_write, output_ptr, count, cfg.residual_branch_count,
                    cfg.hidden_size, stream=0, runtime=self.runtime)
                pending_write = None

        for layer, kind in enumerate(cfg.layer_types):
            if yield_layers and layer:
                # Complete the residual before returning to the scheduler.
                flush_pending(residual_ptr)
                self.runtime.device_synchronize()
                yield None
                invalidate_f16_mirror()
            deferred: list = []
            if layer in cfg.ple_layers:
                flush_pending(residual_ptr)
                layer_prefix = f"layers.{layer}."
                residual_ptr = run_qwen4_exp_ple(
                    residual_ptr,
                    self.prefill_ple_embeddings.ptr,
                    {
                        "ple_key": self.resident.weight(layer_prefix + "ple_key"),
                        "ple_value": self.resident.weight(layer_prefix + "ple_value"),
                    },
                    norm_key_ptr=self.resident.weight(
                        layer_prefix + "ple_norm_key"
                    ).allocation("raw").tensor.ptr,
                    norm_query_ptr=self.resident.weight(
                        layer_prefix + "ple_norm_query"
                    ).allocation("raw").tensor.ptr,
                    norm_conv_ptr=self.resident.weight(
                        layer_prefix + "ple_norm_conv"
                    ).allocation("raw").tensor.ptr,
                    conv_weight_ptr=self.resident.weight(
                        layer_prefix + "ple_conv1d"
                    ).allocation("raw").tensor.ptr,
                    conv_history_ptr=self.state.ple_conv.ptr,
                    scratch=self.ple_prefill_scratch,
                    rows=count,
                    branches=cfg.residual_branch_count,
                    hidden=cfg.hidden_size,
                    conv_kernel=cfg.ple_conv_kernel,
                    dilation=cfg.ple_ngram_size,
                    runtime=self.runtime,
                ).ptr
            if kind == "gdn":
                binding = self.gdn_bindings[layer]
                residual_ptr = run_qwen4_exp_gdn_layer(
                    residual_ptr,
                    binding,
                    conv_state_ptr=(
                        self.state.gdn_conv.ptr
                        + binding.gdn_state_index * gdn_conv_row_bytes
                    ),
                    recurrent_state_ptr=(
                        self.state.gdn_matrix.ptr
                        + binding.gdn_state_index * gdn_matrix_row_bytes
                    ),
                    scratch=self.gdn_prefill_scratch,
                    q4_bundle_prefill=self.q4_bundle_prefill,
                    rows=count,
                    branches=cfg.residual_branch_count,
                    hidden=cfg.hidden_size,
                    low_rank=cfg.residual_low_rank,
                    num_k_heads=cfg.gdn_group_count,
                    num_v_heads=cfg.gdn_time_step_rank,
                    head_dim=cfg.gdn_state_size,
                    conv_kernel=cfg.gdn_conv_kernel,
                    ffn=cfg.expert_feed_forward_length,
                    experts=cfg.expert_count,
                    top_k=cfg.expert_used_count,
                    runtime=self.runtime,
                    pending_write=pending_write,
                    defer_write=deferred,
                    prefill_variant=self.gdn_prefill_variant,
                    gdn_tile16_prefill=self.gdn_tile16_prefill,
                    gdn_peer_prefill_layers=self.gdn_peer_prefill_layers,
                    gdn_exact_bulk_conv=self.gdn_exact_bulk_conv,
                    q5k_row4_prefill=self.q5k_row4_prefill,
                    q51_down_m1=self.q51_down_m1,
                    q51_pair_prefill=self.q51_pair_prefill,
                    q51_fold128_prefill=self.q51_fold128_prefill,
                    q51_fold_pair_prefill=self.q51_fold_pair_prefill,
                    q51_register_cache=self.q51_register_cache,
                    q51_row_publish=self.q51_row_publish,
                    q51_exact_grouped_down=self.q51_exact_grouped_down,
                    q51_wmma_down=self.q51_wmma_down,
                    q51_out8=self.q51_out8,
                    q4_exact_grouped_gate=self.q4_exact_grouped_gate,
                    q4_exact_grouped_all=self.q4_exact_grouped_all,
                    q4_out4=self.q4_out4,
                    q4_tile_m=self.q4_tile_m,
                    q4_tile_n=self.q4_tile_n,
                    forkb_grouped_down=self.forkb_grouped_down,
                    q8_down_row4_prefill=self.q8_down_row4_prefill,
                    q8_down_bundle_prefill=self.q8_down_bundle_prefill,
                    q8_down_register=self.q8_down_register,
                    q8_mapped_down=self.q8_mapped_down,
                ).ptr
            else:
                binding = self.qsa_bindings[layer]
                residual_ptr = run_qwen4_exp_qsa_prefill_layer(
                    residual_ptr,
                    binding,
                    q4_bundle_prefill=self.q4_bundle_prefill,
                    attention_state=self.attention_states[binding.qsa_state_index],
                    index_state=self.index_states[binding.qsa_state_index],
                    scratch=self.qsa_prefill_scratch,
                    metadata=self.qsa_prefill_metadata,
                    start_position=start,
                    rows=count,
                    branches=cfg.residual_branch_count,
                    hidden=cfg.hidden_size,
                    low_rank=cfg.residual_low_rank,
                    query_heads=cfg.attention_head_count,
                    kv_heads=cfg.attention_kv_head_count,
                    head_dim=cfg.attention_key_length,
                    rotary_dim=cfg.rope_dimension_count,
                    theta=cfg.rope_freq_base,
                    index_heads=cfg.indexer_head_count,
                    index_dim=cfg.indexer_key_length,
                    index_rotary_dim=cfg.rope_dimension_count,
                    rope_positions_ptr=rope_positions_ptr,
                    head_mode=self.qsa_prefill_head_mode,
                    h256_mode=self.qsa_h256_prefill_mode,
                    q5k_row4_prefill=self.q5k_row4_prefill,
                    batched_selection=self.qsa_batched_selection,
                    qsa_wave32=self.qsa_wave32,
                    q51_down_m1=self.q51_down_m1,
                    q51_pair_prefill=self.q51_pair_prefill,
                    q51_fold128_prefill=self.q51_fold128_prefill,
                    q51_fold_pair_prefill=self.q51_fold_pair_prefill,
                    q51_register_cache=self.q51_register_cache,
                    q51_row_publish=self.q51_row_publish,
                    q51_exact_grouped_down=self.q51_exact_grouped_down,
                    q51_wmma_down=self.q51_wmma_down,
                    q51_out8=self.q51_out8,
                    q4_exact_grouped_gate=self.q4_exact_grouped_gate,
                    q4_exact_grouped_all=self.q4_exact_grouped_all,
                    q4_out4=self.q4_out4,
                    q4_tile_m=self.q4_tile_m,
                    q4_tile_n=self.q4_tile_n,
                    forkb_grouped_down=self.forkb_grouped_down,
                    q8_down_row4_prefill=self.q8_down_row4_prefill,
                    q8_down_bundle_prefill=self.q8_down_bundle_prefill,
                    q8_down_register=self.q8_down_register,
                    q8_mapped_down=self.q8_mapped_down,
                    ffn=cfg.expert_feed_forward_length,
                    experts=cfg.expert_count,
                    top_k=cfg.expert_used_count,
                    runtime=self.runtime,
                    pending_write=pending_write,
                    defer_write=deferred,
                ).ptr
            pending_write = deferred[0] if deferred else None
        flush_pending(residual_ptr)
        if yield_layers:
            # Keep final hidden readback, head projection and batch import out
            # of the last layer's decode-stall window, even when that tail used
            # less than the target time share.
            self.runtime.device_synchronize()
            yield "prefill_layers_complete"
            invalidate_f16_mirror()
        hidden_seeds = (
            self._read_hidden_seed_rows(residual_ptr, count)
            if capture_hidden_seeds
            else None
        )
        self.position += count
        return (
            residual_ptr
            + (count - 1)
            * cfg.residual_width
            * DType.BF16.itemsize,
            hidden_seeds,
        )

    def prefill_serial(
        self,
        token_ids: list[int] | tuple[int, ...],
        *,
        capture_hidden_seeds: bool = False,
        capture_logits: bool = True,
        capture_target_hidden: bool = True,
    ) -> Qwen4ExpTokenResult:
        if not token_ids:
            raise ValueError("Qwen4Exp prefill requires at least one token")
        self.reset()
        result = None
        captured: list[np.ndarray] = []
        for index, token in enumerate(token_ids):
            result = self.step(
                int(token),
                capture_hidden_seed=capture_hidden_seeds,
                capture_logits=(capture_logits if index + 1 == len(token_ids) else False),
                capture_target_hidden=capture_target_hidden,
            )
            if result.hidden_seeds is not None:
                captured.append(result.hidden_seeds)
        assert result is not None
        if not capture_hidden_seeds:
            return result
        return Qwen4ExpTokenResult(
            result.token_id,
            result.logits,
            hidden_seeds=np.concatenate(captured, axis=0),
        )

    def _prefill_chunked_impl(self, token_ids, **kwargs):
        steps = self._prefill_chunked_steps(token_ids, **kwargs)
        while True:
            try:
                next(steps)
            except StopIteration as done:
                return done.value

    def _prefill_chunked_steps(
        self,
        token_ids: list[int] | tuple[int, ...],
        *,
        capture_hidden_seeds: bool = False,
        capture_logits: bool = True,
        capture_target_hidden: bool = True,
        embedding_overrides: Mapping[int, np.ndarray] | None = None,
        mrope_positions: np.ndarray | None = None,
        reset: bool = True,
        yield_layers: bool = False,
        hidden_seed_consumer=None,
    ):
        if not token_ids:
            raise ValueError("Qwen4Exp prefill requires at least one token")
        start_position = 0 if reset else self.position
        if start_position + len(token_ids) > self.max_sequence_length:
            raise ValueError("Qwen4Exp prefill exceeds sequence capacity")
        if reset:
            self.reset()
        last_residual_ptr = 0
        captured: list[np.ndarray] = []
        values = tuple(int(token) for token in token_ids)
        rope_values = None
        if mrope_positions is not None:
            rope_values = np.ascontiguousarray(mrope_positions, dtype=np.int64)
            if rope_values.shape != (3, len(values)) or np.any(rope_values < 0):
                raise ValueError(
                    "Qwen4Exp prompt MRoPE positions must have shape [3, tokens]"
                )
        raw_rowbatch = os.environ.get("HIPENGINE_QWEN4_EXP_RAW_ROWBATCH", "32")
        if raw_rowbatch in {"", "0", "false", "False"}:
            selected_raw_rowbatch = 0
        elif raw_rowbatch in {"1", "true", "True"}:
            selected_raw_rowbatch = 32
        else:
            selected_raw_rowbatch = int(raw_rowbatch)
        selected_raw_variant = os.environ.get(
            "HIPENGINE_QWEN4_EXP_RAW_VARIANT", "coltile8"
        )
        if selected_raw_variant == "coltile8" and os.environ.get(
            "HIPENGINE_QWEN4_EXP_Q8_WAVE_SCALE", "0"
        ) not in {"", "0", "false", "False"}:
            selected_raw_variant = "coltile8_wave_scale"
        q8_wmma_layers_raw = os.environ.get(
            "HIPENGINE_QWEN4_EXP_Q8_WMMA_LAYERS", ""
        )
        q8_wmma_layers = {
            int(value)
            for value in q8_wmma_layers_raw.split(",")
            if value.strip()
        }
        def q8_wmma_weight_filter(weight: GGUFDeviceWeight) -> bool:
            parts = weight.spec.slot_path.split(".")
            return bool(
                weight.spec.quant_key == "gguf_q8_0"
                and len(parts) >= 3
                and parts[0] == "layers"
                and int(parts[1]) in q8_wmma_layers
            )

        for start in range(0, len(values), self.prefill_chunk_size):
            with (
                raw_k_prefill_rowbatch_session(selected_raw_rowbatch),
                raw_k_prefill_variant_session(selected_raw_variant),
                wmma_prefill_session(True if q8_wmma_layers else None),
                wmma_prefill_weight_filter_session(
                    q8_wmma_weight_filter if q8_wmma_layers else None
                ),
                self._q8_mmq_prefill_context(),
                self._wide_f16_activation_context(),
                _qwen4_exp_q8_tile_scope(self),
            ):
                chunk_kwargs = dict(
                    capture_hidden_seeds=capture_hidden_seeds and hidden_seed_consumer is None,
                    embedding_overrides=embedding_overrides,
                    mrope_positions=(
                        None
                        if rope_values is None
                        else np.ascontiguousarray(
                            rope_values[:, start : start + self.prefill_chunk_size]
                        )
                    ),
                )
                chunk = values[start : start + self.prefill_chunk_size]
                if yield_layers:
                    last_residual_ptr, hidden_rows = yield from self._prefill_chunk_steps(
                        chunk, yield_layers=True, **chunk_kwargs,
                    )
                else:
                    last_residual_ptr, hidden_rows = self._prefill_chunk(chunk, **chunk_kwargs)
            if hidden_seed_consumer is not None:
                # Borrow only until this synchronous callback returns, before
                # another chunk can reuse the target scratch. Restore target
                # projection contexts first; the predictor has its own policy.
                stride = self.config.residual_width * DType.BF16.itemsize
                view = DeviceBuffer(last_residual_ptr - (len(chunk) - 1) * stride,
                                    len(chunk) * stride)
                last_seed = hidden_seed_consumer(chunk, view)
                if capture_hidden_seeds:
                    captured[:] = [np.asarray(last_seed, dtype=np.float32).reshape(1, -1)]
            elif hidden_rows is not None:
                captured.append(hidden_rows)
        cfg = self.config
        assert self.head_scratch is not None
        if capture_target_hidden:
            self.runtime.memcpy(
                self.last_target_hidden.ptr,
                last_residual_ptr,
                self.last_target_hidden.nbytes,
                HipMemcpyKind.DEVICE_TO_DEVICE,
            )
        head_read = run_qwen4_exp_gr_read(
            last_residual_ptr,
            self.resident.weight("root.head_hc_norm").allocation("raw").tensor.ptr,
            self.resident.weight("root.head_hc_down"),
            self.resident.weight("root.head_hc_up"),
            None,
            self.head_scratch,
            rows=1,
            branches=cfg.residual_branch_count,
            hidden=cfg.hidden_size,
            low_rank=cfg.residual_low_rank,
            runtime=self.runtime,
        )
        launch_gguf_linear(
            self.resident.weight("root.lm_head"),
            head_read.mixed.ptr,
            self.logits_buffer.ptr,
            1,
            cfg.hidden_size,
            cfg.vocab_size,
            activation_dtype=GGUF_ACTIVATION_F32,
            output_dtype=GGUF_OUTPUT_F32,
            runtime=self.runtime,
        )
        return self._finish_lm_head(
            capture_logits=capture_logits,
            hidden_seeds=(
                np.concatenate(captured, axis=0)
                if capture_hidden_seeds
                else None
            ),
        )

    def prefill_append_resumable(self, token_ids, *, time_budget_seconds=None, **kwargs):
        """Advance prepared prompt work to a quiescent layer boundary.

        The saved Python context isolates dispatch sessions while suspended.
        Device state and scratch remain owned by this request's staging runner.
        A layer cannot be interrupted; the time budget is a target, not a bound.
        """
        tokens = tuple(int(token) for token in token_ids)
        if time_budget_seconds is None:
            # Narrow prompt tails use the non-wide dispatch family. Spend less
            # time before peer decode there; preserve longer shares for dense
            # chunks to avoid making prompt completion switch-bound.
            time_budget_seconds = 0.08 if len(tokens) < 257 else 0.15
        if not math.isfinite(time_budget_seconds) or time_budget_seconds <= 0:
            raise ValueError("prefill time budget must be finite and positive")
        self._require_open()
        pending = getattr(self, "_pending_prefill_append", None)
        if pending is None:
            context = contextvars.copy_context()
            steps = self._prefill_chunked_steps(tokens, reset=False, yield_layers=True, **kwargs)
            pending = (context, steps, tokens)
            self._pending_prefill_append = pending
        context, steps, expected = pending
        if tokens != expected:
            raise ValueError("resumed prefill must name the same prompt chunk")
        started = time.perf_counter()
        try:
            while True:
                boundary = context.run(next, steps)
                if (boundary == "prefill_layers_complete"
                        or time.perf_counter() - started >= time_budget_seconds):
                    return None
        except StopIteration as done:
            self._pending_prefill_append = None
            return done.value
        except BaseException:
            context.run(steps.close)
            self._pending_prefill_append = None
            raise

    def prefill_append(
        self,
        token_ids: list[int] | tuple[int, ...],
        *,
        capture_hidden_seeds: bool = False,
        capture_logits: bool = True,
        capture_target_hidden: bool = True,
        hidden_seed_consumer=None,
    ) -> Qwen4ExpTokenResult:
        """Append prompt work to the current request using its prepared workspace.

        The caller owns reset, chunk boundaries and request retirement. Each
        call restores dispatch contexts before returning to the scheduler; no
        context manager or temporary workspace binding spans another request.
        A hidden consumer synchronously receives each chunk's borrowed BF16
        rows and returns its final host seed. With a consumer, hidden capture
        returns only that final row rather than downloading the entire prompt.
        """
        return self._prefill_chunked_impl(
            token_ids,
            capture_hidden_seeds=capture_hidden_seeds,
            capture_logits=capture_logits,
            capture_target_hidden=capture_target_hidden,
            hidden_seed_consumer=hidden_seed_consumer,
            reset=False,
        )

    def prefill(
        self,
        token_ids: list[int] | tuple[int, ...],
        *,
        capture_hidden_seeds: bool = False,
        capture_logits: bool = True,
        capture_target_hidden: bool = True,
        embedding_overrides: Mapping[int, np.ndarray] | None = None,
        mrope_positions: np.ndarray | None = None,
        hidden_seed_consumer=None,
    ) -> Qwen4ExpTokenResult:
        return self._prefill_chunked_impl(
            token_ids,
            capture_hidden_seeds=capture_hidden_seeds,
            capture_logits=capture_logits,
            capture_target_hidden=capture_target_hidden,
            embedding_overrides=embedding_overrides,
            mrope_positions=mrope_positions,
            hidden_seed_consumer=hidden_seed_consumer,
        )

    def generate(
        self,
        token_ids: list[int] | tuple[int, ...],
        *,
        max_new_tokens: int,
    ) -> tuple[int, ...]:
        count = int(max_new_tokens)
        if count <= 0:
            raise ValueError("max_new_tokens must be positive")
        result = self.prefill(
            token_ids,
            capture_logits=False,
            capture_target_hidden=False,
        )
        output: list[int] = []
        for index in range(count):
            output.append(result.token_id)
            if index + 1 < count:
                result = self.step(
                    result.token_id,
                    capture_logits=False,
                    capture_target_hidden=False,
                    token_id_resident=True,
                )
        return tuple(output)

    def close(self) -> None:
        if self.closed:
            return
        pending = getattr(self, "_pending_prefill_append", None)
        if pending is not None:
            context, steps, _ = pending
            context.run(steps.close)
            self._pending_prefill_append = None
        if self.moe_graph_cache is not None:
            self.moe_graph_cache.close()
            self.moe_graph_cache = None
        if getattr(self, "layer_graph_cache", None) is not None:
            self.layer_graph_cache.close()
            self.layer_graph_cache = None
        if self._target_verify_output is not None:
            self._target_verify_output.close()
            self._target_verify_output = None
        if self._device_transaction_snapshot is not None:
            self._device_transaction_snapshot.close()
            self._device_transaction_snapshot = None
            self._device_transaction_lease = False
        for buffer in getattr(self, "_verify_rows_buffers", {}).values():
            free(buffer, runtime=self.runtime)
        self._verify_rows_buffers = {}
        self._verify_rows_capacity = 0
        self._pending_verify_block = None
        if self._q8_mmq_weight_sidecars is not None:
            self._q8_mmq_weight_sidecars.close()
            self._q8_mmq_weight_sidecars = None
        self._q8_mmq_policy = None
        self._q8_mmq_library = None
        self._q8_mmq_buffers = ()
        for workspace in reversed(getattr(self, "_prefill_workspaces", ())):
            workspace.close_inputs()
        self._prefill_buffers = []
        for buffer in reversed(self._buffers):
            free(buffer, runtime=self.runtime)
        self._buffers = []
        self._wide_f16_activation_buffer = None
        for state in reversed(self.index_states):
            state.close()
        self.index_states = ()
        for state in reversed(self.attention_states):
            state.close()
        self.attention_states = ()
        if self._shared_position_context is not None:
            free(self._shared_position_context, runtime=self.runtime)
            self._shared_position_context = None
            self._shared_position_context_host = None
        for workspace in reversed(getattr(self, "_prefill_workspaces", ())):
            workspace.close()
        self._prefill_workspaces = []
        for owner in (
            self.head_scratch,
            self.ple_scratch,
            self.qsa_scratch,
            self.gdn_scratch,
            self.state,
        ):
            if owner is not None:
                owner.close()
        self.closed = True

    def _require_open(self) -> None:
        if self.closed:
            raise RuntimeError("Qwen4Exp resident model runner is closed")


__all__ = [
    "Qwen4ExpDecodeState",
    "bind_qwen4_exp_gdn_layer",
    "bind_qwen4_exp_qsa_layer",
    "Qwen4ExpDecodeStateSnapshot",
    "Qwen4ExpDecodeStateDeviceSnapshot",
    "Qwen4ExpDenseAttentionState",
    "Qwen4ExpGDNLayerDeviceWeights",
    "Qwen4ExpGDNLayerScratch",
    "Qwen4ExpGDNMixerDeviceWeights",
    "Qwen4ExpGDNScratch",
    "Qwen4ExpGGUFResidentModelRunner",
    "Qwen4ExpHostQSAIndexSnapshot",
    "Qwen4ExpHostQSAIndexState",
    "Qwen4ExpGRDeviceWeights",
    "Qwen4ExpGRReadDeviceResult",
    "Qwen4ExpMoEDeviceResult",
    "Qwen4ExpMoEScratch",
    "Qwen4ExpGRScratch",
    "Qwen4ExpPLEScratch",
    "Qwen4ExpQSAIndexDeviceState",
    "Qwen4ExpQSAPrefillMetadata",
    "Qwen4ExpRunnerSnapshot",
    "Qwen4ExpRunnerDeviceTransaction",
    "Qwen4ExpTokenResult",
    "Qwen4ExpTargetAcceptResult",
    "Qwen4ExpTargetVerifyOutput",
    "Qwen4ExpTargetVerifyResult",
    "Qwen4ExpQSALayerDeviceWeights",
    "Qwen4ExpQSALayerScratch",
    "Qwen4ExpQSAMixerDeviceWeights",
    "Qwen4ExpQSAScratch",
    "run_qwen4_exp_dense_qsa_layer",
    "run_qwen4_exp_dense_qsa_token_mixer",
    "run_qwen4_exp_qsa_prefill_token_mixer",
    "stage_qwen4_exp_ple_rows",
    "run_qwen4_exp_gdn_layer",
    "run_qwen4_exp_qsa_prefill_layer",
    "run_qwen4_exp_gdn_token_mixer",
    "run_qwen4_exp_moe",
    "run_qwen4_exp_ple",
    "run_qwen4_exp_gr_read",
]
