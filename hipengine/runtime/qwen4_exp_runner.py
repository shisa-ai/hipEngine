"""Dedicated Qwen4Exp GGUF runtime composition.

This module begins with the strict gated-residual read owner used by every
Qwen3.8-Flash-Next token mixer and MoE. It is model-specific composition over
model-neutral GGUF projections and registered gfx11 primitives; it does not add
architecture branches to the engine or dispatch layer.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import os
import time
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.hip import HipMemcpyKind, HipRuntime, get_hip_runtime
from hipengine.core.memory import (
    DeviceBuffer,
    copy_device_to_host,
    copy_host_to_device,
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
    qwen35_paged_full_attn_decode_context_bf16_batch_spans,
    qwen35_paged_full_attn_decode_context_bf16_spans,
)
from hipengine.kernels.hip_gfx1100.attention.paged_kv_write import (
    qwen35_write_paged_kv_f32_batch_spans,
    qwen35_write_paged_kv_f32_spans,
)
from hipengine.kernels.hip_gfx1100.attention.qwen4_exp_qsa import (
    qwen4_exp_qsa_gate_context_f32,
    qwen4_exp_qsa_norm_rope_f32,
    qwen4_exp_qsa_norm_rope_rows_f32,
    qwen4_exp_qsa_norm_mrope_f32,
    qwen4_exp_qsa_norm_mrope_rows_f32,
    qwen4_exp_qsa_pool_norm_rope_f32,
    qwen4_exp_qsa_score_f32,
    qwen4_exp_qsa_scatter_index_key_device_position_f32,
    qwen4_exp_qsa_scatter_index_keys_f32,
    qwen4_exp_qsa_sparse_attention_paged_bf16_f32,
    qwen4_exp_qsa_sparse_attention_paged_bf16_ordered_f32,
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
from hipengine.kernels.hip_gfx1100.convert.cast import bf16_to_f32, f32_to_bf16
from hipengine.kernels.hip_gfx1100.fused.paro_combine import (
    shared_gate_combine_batch_out_bf16,
    shared_gate_combine_out_bf16,
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
    qwen35_moe_group_scatter_gather_lowp,
    qwen35_moe_group_scatter,
    qwen35_moe_mmq32_tile_map,
    qwen35_moe_wmma_tile_map,
)
from hipengine.kernels.hip_gfx1100.moe.router import qwen35_router_select
from hipengine.kernels.hip_gfx1100.linear.lm_head import (
    argmax_f32,
    lm_head_argmax_stage1_blocks,
)
from hipengine.kernels.hip_gfx1100.linear_attn.conv import (
    qwen35_linear_attn_conv_decode_f32,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_gemv import (
    gguf_q4_k_quantize_bf16_q8_1,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_selected_prefill import (
    gguf_q4_k_selected_dual_grouped_rowbatch8_bf16_bf16_out,
    gguf_q4_k_selected_dual_grouped_rowbatch8_out4_bf16_bf16_out,
    gguf_q4_k_selected_dual_grouped_rowbatch8_out4_expertgrid64_bf16_bf16_out,
    gguf_q4_k_selected_dual_wmma_iu8_prefill_bf16_bf16_out,
    gguf_q4_k_selected_dual_wmma_prefill_compact_bf16_bf16_out,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_k_selected_prefill import (
    gguf_q5_k_selected_wmma_prefill_compact_bf16_bf16_out,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q8_0_prefill import (
    gguf_q8_0_selected_grouped_prefill_compact_bf16_bf16_out,
    gguf_q8_0_wmma_prefill_bf16_bf16_out,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_q8_1_selected_prefill import (
    gguf_q4_k_selected_dual_q8_1_ds4_mmq32_prefill_compact32_bf16_bf16_out,
    gguf_q8_1_mmq_ds4_pack_bf16 as gguf_q4_k_q8_1_mmq_ds4_pack_bf16,
    gguf_q8_1_mmq_ds4_pack_bf16_d4x3 as gguf_q8_1_mmq_ds4_pack_bf16,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q5_1_mmq_selected_prefill import (
    gguf_q5_1_mmq_ds4_selected_prefill_bf16_bf16_out,
)
from hipengine.kernels.hip_gfx1100.quant.qwen4_exp_q5_1 import (
    qwen4_exp_gather_bf16_lanes,
    qwen4_exp_q5_1_selected_grouped_prefill_compact_rowbatch8_bf16_bf16_out,
    qwen4_exp_q5_1_selected_grouped_prefill_compact_rowbatch8_out8_bf16_bf16_out,
    qwen4_exp_q5_1_selected_grouped_prefill_compact_rowbatch8_out8_expertgrid64_bf16_bf16_out,
    qwen4_exp_q5_1_selected_grouped_wmma_prefill_compact_bf16_bf16_out,
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
    qwen4_exp_grouped_rmsnorm_bf16_f32,
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
    raw_k_prefill_rowbatch_session,
    raw_k_prefill_variant_session,
    wmma_prefill_session,
    wmma_prefill_weight_filter_session,
    GGUF_OUTPUT_F32,
    launch_gguf_linear,
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
from hipengine.kvcache import KVLiveSpans
from hipengine.loading.qwen4_exp_materialize import Qwen4ExpResidentWeights
from hipengine.runtime.gguf_weight import GGUFDeviceWeight
from hipengine.runtime.moe_graph import MoeGraphCache


def _qwen4_exp_layer_allowed(
    weight: GGUFDeviceWeight,
    *,
    env_name: str,
    default: str,
) -> bool:
    parts = weight.spec.slot_path.split(".")
    if len(parts) <= 2 or parts[0] != "layers":
        return False
    try:
        layer = int(parts[1])
    except ValueError:
        return False
    raw = os.environ.get(env_name, default)
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


def _qwen4_exp_q5_1_mmq_layer_allowed(weights: Mapping[str, GGUFDeviceWeight]) -> bool:
    """Certified Q5_1-MMQ prefill layer set (suffix 32-47 by default)."""

    return _qwen4_exp_layer_allowed(
        weights["expert_gate"],
        env_name="HIPENGINE_QWEN4_EXP_Q5_1_MMQ_LAYERS",
        default="32-47",
    )


def _qwen4_exp_q4_k_mmq_layer_allowed(weights: Mapping[str, GGUFDeviceWeight]) -> bool:
    """Certified Q4_K-MMQ prefill layer set (suffix 35-47 by default)."""

    return _qwen4_exp_layer_allowed(
        weights["expert_gate"],
        env_name="HIPENGINE_QWEN4_EXP_Q4_K_MMQ_LAYERS",
        default="35-47",
    )


def _qwen4_exp_q4_iu8_layer_allowed(weights: Mapping[str, GGUFDeviceWeight]) -> bool:
    """iu8-WMMA selected-dual prefill layer set (dp4a class, screened suffix)."""

    return _qwen4_exp_layer_allowed(
        weights["expert_gate"],
        env_name="HIPENGINE_QWEN4_EXP_Q4_IU8_LAYERS",
        default="35-47",
    )


def _qwen4_exp_gdn_peer_layer_allowed(
    weights: Mapping[str, GGUFDeviceWeight],
) -> bool:
    return _qwen4_exp_layer_allowed(
        weights["attn_qkv"],
        env_name="HIPENGINE_QWEN4_EXP_GDN_PEER_PREFILL_LAYERS",
        default="35-47",
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
    """Select the exact tile-16 GDN prefill owner inside the colwarps gate.

    PF-5 production default after the engaged one-process/one-residency
    canonical A/B (2026-09-05, 72/72 cross-mode exact, prefill
    +0.49%/+0.47%/+0.37% at p512/p1024/p4096). Non-envelope shapes and the
    ``HIPENGINE_QWEN4_EXP_GDN_TILE16_PREFILL=0`` opt-out use the columnwarp
    parent; the serial strict route stays registered below both.
    """

    if os.environ.get("HIPENGINE_QWEN4_EXP_GDN_TILE16_PREFILL", "1") in {
        "",
        "0",
        "false",
        "False",
    }:
        return False
    return (
        rows >= 16
        and num_k_heads == 16
        and num_v_heads in (32, 48)
        and head_dim == 128
    )


def _configure_qwen4_exp_moe_mmq_scratch(
    moe: object,
    *,
    rows: int,
    hidden: int,
    ffn: int,
    top_k: int,
    runtime: HipRuntime,
) -> None:
    compact_capacity = rows * top_k
    if (
        os.environ.get("HIPENGINE_QWEN4_EXP_Q5_1_MMQ_PREFILL", "0")
        not in {"", "0", "false", "False"}
        and ffn % 128 == 0
        and getattr(moe, "q5_1_mmq_ds4_workspace", None) is None
    ):
        from hipengine.kernels.hip_gfx1100.quant.gguf_q5_1_mmq_selected_prefill import (
            build_gguf_q5_1_mmq_selected_prefill,
            ds4_workspace_nbytes,
        )

        moe.q5_1_mmq_ds4_workspace = malloc(
            ds4_workspace_nbytes(compact_capacity, ffn), runtime=runtime
        )
        moe.q5_1_mmq_library = build_gguf_q5_1_mmq_selected_prefill(load=True)
    if (
        os.environ.get("HIPENGINE_QWEN4_EXP_Q4_K_MMQ_PREFILL", "0")
        not in {"", "0", "false", "False"}
        and hidden % 128 == 0
        and getattr(moe, "q4_k_mmq_ds4_workspace", None) is None
    ):
        from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_q8_1_selected_prefill import (
            build_gguf_q4_k_q8_1_selected_prefill,
        )

        moe.q4_k_mmq_ds4_workspace = malloc(
            compact_capacity * (hidden // 128) * 144, runtime=runtime
        )
        moe.q4_k_mmq_identity = malloc(
            compact_capacity * DType.INT64.itemsize, runtime=runtime
        )
        identity = np.arange(compact_capacity, dtype=np.int64)
        copy_host_to_device(
            moe.q4_k_mmq_identity,
            host_array_ptr(identity),
            identity.nbytes,
            runtime=runtime,
        )
        moe.q4_k_mmq_library = build_gguf_q4_k_q8_1_selected_prefill(load=True)


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
            copy_host_to_device(buffer, host_array_ptr(host), runtime=self.runtime)

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

    @classmethod
    def allocate(
        cls,
        *,
        max_positions: int,
        block_size: int,
        kv_heads: int,
        head_dim: int,
        runtime: HipRuntime | None = None,
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
            position = malloc(position_host.nbytes, runtime=active_runtime)
            buffers.append(position)
            context = malloc(context_host.nbytes, runtime=active_runtime)
            buffers.append(context)
            copy_host_to_device(block_table, host_array_ptr(block_host), runtime=active_runtime)
            copy_host_to_device(position, host_array_ptr(position_host), runtime=active_runtime)
            copy_host_to_device(context, host_array_ptr(context_host), runtime=active_runtime)
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
        )

    def set_position(self, value: int) -> None:
        if self.closed:
            raise RuntimeError("Qwen4Exp attention state is closed")
        position = int(value)
        if not 0 <= position < self.max_positions:
            raise ValueError("attention position exceeds state capacity")
        self.position_host[0] = position
        self.context_host[0] = position + 1
        copy_host_to_device(
            self.position, host_array_ptr(self.position_host), runtime=self.runtime
        )
        copy_host_to_device(
            self.context, host_array_ptr(self.context_host), runtime=self.runtime
        )

    def close(self) -> None:
        if self.closed:
            return
        for buffer in reversed(
            (self.key_cache, self.value_cache, self.block_table, self.position, self.context)
        ):
            free(buffer, runtime=self.runtime)
        self.closed = True


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
                copy_host_to_device(
                    buffer, host_array_ptr(host), runtime=attention_state.runtime
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
        copy_host_to_device(
            self.positions,
            host_array_ptr(self.positions_host),
            count * np.dtype(np.int64).itemsize,
            runtime=self.runtime,
        )
        copy_host_to_device(
            self.context_counts,
            host_array_ptr(self.context_counts_host),
            count * np.dtype(np.int64).itemsize,
            runtime=self.runtime,
        )

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
        copy_host_to_device(
            self.selected_positions,
            host_array_ptr(self.selected_positions_host),
            count * self.selection_capacity * np.dtype(np.int64).itemsize,
            runtime=self.runtime,
        )
        copy_host_to_device(
            self.selected_counts,
            host_array_ptr(self.selected_counts_host),
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
    """Select exact QSA blocks without the quadratic single-thread GPU fallback."""

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
    block_starts_host: np.ndarray
    scores_host: np.ndarray
    selected_starts_host: np.ndarray
    selected_count_host: np.ndarray
    query_position_host: np.ndarray
    selected_positions_host: np.ndarray
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
        block_starts_host = np.arange(complete_blocks, dtype=np.int64) * ratio
        scores_host = np.empty(complete_blocks, dtype=np.float32)
        selected_starts_host = np.full(budget, -1, dtype=np.int64)
        selected_count_host = np.zeros(1, dtype=np.int32)
        query_position_host = np.zeros(1, dtype=np.int64)
        selected_positions_host = np.empty(budget * ratio + ratio - 1, dtype=np.int64)
        sizes = (
            capacity * dimension * DType.FP32.itemsize,
            member_host.nbytes,
            block_starts_host.nbytes,
            complete_blocks * dimension * DType.FP32.itemsize,
            complete_blocks * DType.FP32.itemsize,
            selected_starts_host.nbytes,
            selected_count_host.nbytes,
            query_position_host.nbytes,
            selected_positions_host.nbytes,
        )
        buffers: list[DeviceBuffer] = []
        try:
            for size in sizes:
                buffers.append(malloc(size, runtime=active_runtime))
            copy_host_to_device(buffers[1], host_array_ptr(member_host), runtime=active_runtime)
            copy_host_to_device(
                buffers[2], host_array_ptr(block_starts_host), runtime=active_runtime
            )
            copy_host_to_device(
                buffers[5], host_array_ptr(selected_starts_host), runtime=active_runtime
            )
            copy_host_to_device(
                buffers[6], host_array_ptr(selected_count_host), runtime=active_runtime
            )
            copy_host_to_device(
                buffers[7], host_array_ptr(query_position_host), runtime=active_runtime
            )
            active_runtime.memset(buffers[0].ptr, 0, buffers[0].nbytes)
        except Exception:
            for buffer in reversed(buffers):
                free(buffer, runtime=active_runtime)
            raise
        return cls(
            *buffers,
            member_host,
            block_starts_host,
            scores_host,
            selected_starts_host,
            selected_count_host,
            query_position_host,
            selected_positions_host,
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

    def select_positions_host(
        self,
        prepared_query_ptr: int,
        *,
        query_position: int,
        key_norm_weight_ptr: int,
        rotary_dim: int,
        theta: float,
        eps: float = 1e-6,
        stream: int = 0,
    ) -> np.ndarray:
        """Return exact selected logical positions for any appended sparse query."""

        if self.closed:
            raise RuntimeError("QSA index state is closed")
        position = int(query_position)
        if position < 0 or position >= self.count:
            raise ValueError("QSA query position must identify an appended index key")
        blocks = (position + 1) // self.compression_ratio
        if blocks <= self.block_budget:
            raise ValueError("native QSA selection requires more blocks than the budget")
        self.query_position_host[0] = position
        copy_host_to_device(
            self.query_position,
            host_array_ptr(self.query_position_host),
            runtime=self.runtime,
        )
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
        if stream:
            self.runtime.stream_synchronize(stream)
        copy_device_to_host(
            host_array_ptr(self.scores_host),
            self.scores,
            blocks * DType.FP32.itemsize,
            runtime=self.runtime,
        )
        chosen = _qsa_select_starts_host(
            self.scores_host[:blocks],
            self.block_starts_host[:blocks],
            budget=self.block_budget,
        )
        self.selected_starts_host[:] = chosen
        self.selected_count_host[0] = self.block_budget
        output_count = 0
        for start in chosen.tolist():
            stop = output_count + self.compression_ratio
            self.selected_positions_host[output_count:stop] = np.arange(
                int(start), int(start) + self.compression_ratio, dtype=np.int64
            )
            output_count = stop
        if position % self.compression_ratio != self.compression_ratio - 1:
            tail_start = position // self.compression_ratio * self.compression_ratio
            tail = np.arange(tail_start, position + 1, dtype=np.int64)
            self.selected_positions_host[output_count : output_count + tail.size] = tail
            output_count += int(tail.size)
        selected = self.selected_positions_host[:output_count]
        selected.sort()
        return selected.copy()

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
        self.pooled_count = restored // self.compression_ratio

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
            _configure_qwen4_exp_moe_mmq_scratch(
                moe,
                rows=rows,
                hidden=hidden,
                ffn=ffn,
                top_k=top_k,
                runtime=active_runtime,
            )
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
            _configure_qwen4_exp_moe_mmq_scratch(
                moe,
                rows=rows,
                hidden=hidden,
                ffn=ffn,
                top_k=top_k,
                runtime=active_runtime,
            )
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
    runtime: HipRuntime
    closed: bool = False
    q5_1_mmq_ds4_workspace: DeviceBuffer | None = None
    q5_1_mmq_library: object | None = None
    q4_k_mmq_ds4_workspace: DeviceBuffer | None = None
    q4_k_mmq_identity: DeviceBuffer | None = None
    q4_k_mmq_library: object | None = None

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
            )
        ):
            free(buffer, runtime=self.runtime)
        if self.q5_1_mmq_ds4_workspace is not None:
            free(self.q5_1_mmq_ds4_workspace, runtime=self.runtime)
            self.q5_1_mmq_ds4_workspace = None
        if self.q4_k_mmq_identity is not None:
            free(self.q4_k_mmq_identity, runtime=self.runtime)
            self.q4_k_mmq_identity = None
        if self.q4_k_mmq_ds4_workspace is not None:
            free(self.q4_k_mmq_ds4_workspace, runtime=self.runtime)
            self.q4_k_mmq_ds4_workspace = None
        self.closed = True


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
    eps: float = 1e-6,
    stream: int = 0,
    runtime: HipRuntime | None = None,
) -> DeviceBuffer:
    """Execute dense-equivalent QSA decode below the sparse-selection budget."""

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
        if not device_position_owned and index_state.count > index_state.dense_equivalent_limit:
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
        if ordered_decode:
            scores, coefficients = scratch.ordered_attention_scratch(
                query_heads=query_heads, selected_count=selected_count
            )
            qwen4_exp_qsa_sparse_attention_paged_bf16_ordered_f32(
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
            if (
                head_dim == 128
                and os.environ.get("HIPENGINE_QWEN4_EXP_QSA_WAVE32", "1")
                not in {"", "0", "false", "False"}
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


def qwen4_exp_qsa_h256_wave_prefill_variant() -> str:
    return (
        "strict_h256_page256_wave_rows_spans"
        if os.environ.get("HIPENGINE_QWEN4_EXP_QSA_H256_WAVE_PREFILL") == "page256"
        else "strict_h256_wave_rows_spans"
    )


def qwen4_exp_qsa_h256_wave_prefill_selected(
    *, head_dim: int, rows: int, backend: str, block_size: int = 256
) -> bool:
    return (
        head_dim == 256 and rows >= 16
        and os.environ.get("HIPENGINE_QWEN4_EXP_QSA_H256_WAVE_PREFILL", "0")
        not in {"", "0", "false", "False"}
        and (os.environ.get("HIPENGINE_QWEN4_EXP_QSA_H256_WAVE_PREFILL") != "page256"
             or block_size == 256)
        and is_registered(KernelKey(
            backend, "qsa_sparse_attention", "bf16_kv", qwen4_exp_qsa_h256_wave_prefill_variant()))
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
    eps: float = 1e-6,
    stream: int = 0,
    runtime: HipRuntime | None = None,
) -> DeviceBuffer:
    """Execute one contiguous QSA prompt chunk over shared paged state."""

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
        launch_gguf_linear(
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
    split_rope_rows = (
        qwen4_exp_qsa_split_norm_mrope_rows_f32
        if rope_positions_ptr is not None
        else qwen4_exp_qsa_split_norm_rope_rows_f32
    )
    split_rope_rows(
        scratch.q_projected.ptr,
        scratch.key_projected.ptr,
        weights.q_norm_weight_ptr,
        weights.k_norm_weight_ptr,
        metadata.positions.ptr
        if rope_positions_ptr is None
        else int(rope_positions_ptr),
        scratch.query.ptr,
        scratch.key.ptr,
        scratch.gate.ptr,
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
    index_rope_rows = (
        qwen4_exp_qsa_norm_mrope_rows_f32
        if rope_positions_ptr is not None
        else qwen4_exp_qsa_norm_rope_rows_f32
    )
    index_rope_rows(
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
    if qsa_flash:
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
        dense_attention = (
            qwen35_paged_full_attn_decode_context_bf16_batch_fixed256_spans
            if _qwen4_exp_qsa_dense_fixed256_enabled(dense_rows)
            else qwen35_paged_full_attn_decode_context_bf16_batch_spans
        )
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
        batched_selection = os.environ.get(
            "HIPENGINE_QWEN4_EXP_QSA_BATCHED_SELECTION", "1"
        ) not in {"", "0", "false", "False"}
        if batched_selection:
            score_blocks = (start + count) // index_state.compression_ratio
            if score_blocks > metadata.score_blocks:
                raise ValueError("QSA score width exceeds prefill metadata capacity")
            qwen4_exp_qsa_score_f32(
                scratch.index_query.ptr
                + dense_rows * index_heads * index_dim * DType.FP32.itemsize,
                index_state.pooled_keys.ptr,
                metadata.scores.ptr,
                sparse_rows,
                score_blocks,
                index_heads,
                index_dim,
                stream=stream,
                runtime=active_runtime,
            )
            qwen4_exp_qsa_topk_expand_rows_f32_i64(
                metadata.scores.ptr,
                metadata.positions.ptr + dense_rows * DType.INT64.itemsize,
                metadata.selected_positions.ptr
                + dense_rows * metadata.selection_capacity * DType.INT64.itemsize,
                metadata.selected_counts.ptr + dense_rows * DType.INT32.itemsize,
                sparse_rows,
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
        sparse_attention_rows = qwen4_exp_qsa_sparse_attention_paged_bf16_rows_f32
        backend = str(weights.projections["attn_q"].backend)
        if qwen4_exp_qsa_h256_wave_prefill_selected(
            head_dim=head_dim, rows=sparse_rows, backend=backend,
            block_size=attention_state.block_size,
        ):
            sparse_attention_rows = resolve(
                backend=backend, layer="qsa_sparse_attention", quant="bf16_kv",
                variant=qwen4_exp_qsa_h256_wave_prefill_variant())
        elif (
            head_dim == 128
            and os.environ.get("HIPENGINE_QWEN4_EXP_QSA_WAVE32", "1")
            not in {"", "0", "false", "False"}
        ):
            sparse_attention_rows = (
                qwen4_exp_qsa_sparse_attention_paged_bf16_rows_wave32_f32
            )
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
    attention_state.set_position(start + count - 1)
    return scratch.output


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
) -> Qwen4ExpGRReadDeviceResult:
    """Execute strict grouped GR read using model-neutral resident projections."""

    if scratch.closed:
        raise RuntimeError("Qwen4Exp GR scratch is closed")
    if rows <= 0 or branches <= 0 or hidden <= 0 or low_rank <= 0:
        raise ValueError("rows, branches, hidden, and low_rank must be positive")
    active_runtime = runtime or scratch.runtime
    if active_runtime is not scratch.runtime:
        raise ValueError("runtime must match the GR scratch owner")
    residual_width = branches * hidden
    qwen4_exp_grouped_rmsnorm_bf16_f32(
        residual_ptr,
        norm_weight_ptr,
        scratch.normalized.ptr,
        rows,
        branches,
        hidden,
        eps,
        stream=stream,
        runtime=active_runtime,
    )
    launch_gguf_linear(
        down_weight,
        scratch.normalized.ptr,
        scratch.low_rank.ptr,
        rows,
        residual_width,
        low_rank,
        activation_dtype=GGUF_ACTIVATION_F32,
        output_dtype=GGUF_OUTPUT_F32,
        stream=stream,
        runtime=active_runtime,
    )
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
    if fused_up:
        fused_up_key = _qwen4_exp_gr_wave_scale_key(fused_up_key, rows=rows, branches=branches)
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


def _qwen4_exp_qsa_dense_fixed256_enabled(rows: int) -> bool:
    return rows >= 2


def _qwen4_exp_gr_wave_scale_key(key: KernelKey, *, rows: int, branches: int) -> KernelKey:
    if rows <= 256 or branches != 4 or os.environ.get(
        "HIPENGINE_QWEN4_EXP_GR_WAVE_SCALE", "0"
    ) in {"", "0", "false", "False"}:
        return key
    candidate = KernelKey(key.backend, key.layer, key.quant,
                          "coltile2_branch4_rowbatch4_wave_scale_f32_exact")
    return candidate if is_registered(candidate) else key


def _qwen4_exp_gr_up_sigmoid_mean_enabled(rows: int) -> bool:
    return rows > 256


def _qwen4_exp_gr_sigmoid_mean_fused(rows: int) -> bool:
    """Select the exact launch-contracted GR epilogue in its measured row range."""

    return rows <= 256


def _qwen4_exp_q8_mmq_policy(policy):
    """Add the P3 attention-gate shape only for an explicit candidate run."""

    if policy is None or os.environ.get(
        "HIPENGINE_QWEN4_EXP_Q8_MMQ_ATTN_GATE", "0"
    ) in {"", "0", "false", "False"}:
        return policy
    minimums = dict(policy.min_rows)
    minimums[(2560, 6144)] = 64
    return replace(policy, min_rows=minimums)


def _qwen4_exp_router_f32_tile4_enabled(rows: int) -> bool:
    """Select the exact four-token F32 router producer for multirow work."""

    return rows >= 2


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


def qwen4_exp_grouped_row4_prefill_selected(
    weights: Mapping[str, GGUFDeviceWeight], *, rows: int, backend: str
) -> bool:
    if rows < 64 or os.environ.get(
        "HIPENGINE_QWEN4_EXP_GROUPED_ROW4_PREFILL", "0"
    ) in {"", "0", "false", "False"}:
        return False
    return all(
        is_registered(KernelKey(
            backend, "linear", weights[name].spec.quant_key,
            "selected_grouped_row4_gemv_bf16_bf16_out",
        ))
        for name in ("expert_gate", "expert_up")
    )


def qwen4_exp_q4_bundle_prefill_selected(backend: str, quant: str) -> bool:
    return (
        os.environ.get("HIPENGINE_QWEN4_EXP_Q4_BUNDLE_PREFILL", "0")
        not in {"", "0", "false", "False"}
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


def qwen4_exp_q51_pair_prefill_selected(
    backend: str, quant: str, *, rows: int, in_features: int
) -> bool:
    return (
        rows >= 64 and in_features <= 4096
        and os.environ.get("HIPENGINE_QWEN4_EXP_Q51_PAIR_PREFILL", "0")
        not in {"", "0", "false", "False"}
        and is_registered(KernelKey(
            backend, "moe_linear", quant, "selected_grouped_prefill_pair2_bf16_bf16_out"))
    )


def run_qwen4_exp_moe(
    mixed_ptr: int,
    weights: Mapping[str, GGUFDeviceWeight],
    *,
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
) -> Qwen4ExpMoEDeviceResult:
    """Run normalized softmax top-k routed experts plus gated shared expert."""

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
                scratch=scratch,
                rows=rows,
                hidden=hidden,
                ffn=ffn,
                experts=experts,
                top_k=top_k,
                stream=graph_stream,
                runtime=active_runtime,
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
    load_backend_kernel_package(backend)
    router_tile4 = _qwen4_exp_router_f32_tile4_enabled(rows)
    if router_tile4:
        router_logits = resolve(
            backend=backend,
            layer="router_logits",
            quant="f32",
            variant="f32_hidden_token_tile4_dense_exact",
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
    qwen35_router_select(
        scratch.router_logits.ptr,
        scratch.selected.ptr,
        scratch.routing.ptr,
        rows,
        experts,
        experts,
        top_k,
        stream=stream,
        runtime=active_runtime,
    )
    f32_to_bf16(
        mixed_ptr,
        scratch.hidden_bf16.ptr,
        rows * hidden,
        stream=stream,
        runtime=active_runtime,
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
        variant = "selected_gemv_bf16_bf16_out"
        if weight.spec.quant_key == "gguf_q5_1":
            variant = "selected_gemv_logical256_t64_bf16_bf16_out"
            if os.environ.get("HIPENGINE_QWEN4_EXP_Q5_1_WAVE64", "") not in {
                "", "0", "false", "False",
            }:
                variant = "selected_gemv_wave64_bf16_bf16_out"
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
    q4_k_mmq_prefill = (
        os.environ.get("HIPENGINE_QWEN4_EXP_Q4_K_MMQ_PREFILL", "0")
        not in {"", "0", "false", "False"}
        and scratch.q4_k_mmq_ds4_workspace is not None
        and scratch.q4_k_mmq_identity is not None
        and rows >= 2
        and hidden % 256 == 0
        and ffn % 32 == 0
        and weights["expert_gate"].spec.quant_key == "gguf_q4_k"
        and weights["expert_up"].spec.quant_key == "gguf_q4_k"
        and _qwen4_exp_q4_k_mmq_layer_allowed(weights)
    )
    exact_grouped_down = (
        not production_grouped_moe
        and weights["expert_down"].spec.quant_key == "gguf_q5_1"
        and rows >= 2
        and os.environ.get("HIPENGINE_QWEN4_EXP_EXACT_GROUPED_DOWN", "1")
        not in {"", "0", "false", "False"}
    )
    exact_grouped_q4_gate = (
        not production_grouped_moe
        and rows >= 2
        and weights["expert_gate"].spec.quant_key == "gguf_q4_k"
        and weights["expert_up"].spec.quant_key == "gguf_q4_k"
        and os.environ.get("HIPENGINE_QWEN4_EXP_EXACT_GROUPED_Q4", "1")
        not in {"", "0", "false", "False"}
        and (
            exact_grouped_down
            or os.environ.get("HIPENGINE_QWEN4_EXP_EXACT_GROUPED_Q4_ALL", "1")
            not in {"", "0", "false", "False"}
        )
    )
    grouped_prefill = exact_grouped_down or exact_grouped_q4_gate or (
        (production_grouped_moe or (
            rows >= 16
            and os.environ.get("HIPENGINE_QWEN4_EXP_GROUPED_MOE_PREFILL", "")
            not in {"", "0", "false", "False"}
        ))
        and (
            weights["expert_gate"].spec.quant_key,
            weights["expert_up"].spec.quant_key,
        )
        in {
            ("gguf_q4_k", "gguf_q4_k"),
            ("gguf_q5_k", "gguf_q5_k"),
        }
    )
    if grouped_prefill:
        active_runtime.memset(
            scratch.group_counts.ptr, 0, scratch.group_counts.nbytes
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
        active_runtime.memset(
            scratch.group_scatter_offsets.ptr,
            0,
            scratch.group_scatter_offsets.nbytes,
        )
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
        if q4_k_mmq_prefill:
            qwen35_moe_mmq32_tile_map(
                scratch.group_expert_start.ptr,
                scratch.group_wmma_expert_start.ptr,
                scratch.group_tile_expert.ptr,
                scratch.group_wmma_total.ptr,
                experts,
                tile_capacity=tile_capacity,
                stream=stream,
                runtime=active_runtime,
            )
        elif not exact_grouped_down and not exact_grouped_q4_gate:
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
        if q4_k_mmq_prefill or (
            not exact_grouped_down and not exact_grouped_q4_gate
        ):
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
            tile_rows = 32 if q4_k_mmq_prefill else 16
            if wmma_total_rows <= 0 or wmma_total_rows > tile_capacity * tile_rows:
                raise RuntimeError("Qwen4Exp grouped MoE tile row count is invalid")
        if q4_k_mmq_prefill:
            gguf_q4_k_q8_1_mmq_ds4_pack_bf16(
                scratch.expert_down.ptr,
                scratch.q4_k_mmq_ds4_workspace.ptr,
                compact,
                hidden,
                stream=stream,
                runtime=active_runtime,
                library=scratch.q4_k_mmq_library,
            )
            gguf_q4_k_selected_dual_q8_1_ds4_mmq32_prefill_compact32_bf16_bf16_out(
                scratch.q4_k_mmq_ds4_workspace.ptr,
                scratch.q4_k_mmq_identity.ptr,
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
                library=scratch.q4_k_mmq_library,
            )
            silu_mul_dual_out_bf16(
                scratch.group_gate_up.ptr,
                scratch.expert_intermediate.ptr,
                rows=compact,
                features=ffn,
                stream=stream,
                runtime=active_runtime,
            )
        elif exact_grouped_q4_gate:
            expert_grid_mode = os.environ.get(
                "HIPENGINE_QWEN4_EXP_EXACT_EXPERT_GRID", "64"
            )
            expert_grid64 = expert_grid_mode in {"64", "q4"}
            grouped_q4_gate = (
                gguf_q4_k_selected_dual_grouped_rowbatch8_out4_expertgrid64_bf16_bf16_out
                if expert_grid64
                else gguf_q4_k_selected_dual_grouped_rowbatch8_out4_bf16_bf16_out
                if os.environ.get("HIPENGINE_QWEN4_EXP_Q4_OUT4", "1")
                not in {"", "0", "false", "False"}
                else gguf_q4_k_selected_dual_grouped_rowbatch8_bf16_bf16_out
            )
            if expert_grid64 and qwen4_exp_q4_pair_prefill_selected(
                backend, weights["expert_gate"].spec.quant_key,
                rows=rows, in_features=hidden,
            ):
                grouped_q4_gate = resolve(
                    backend=backend, layer="moe_linear",
                    quant=weights["expert_gate"].spec.quant_key,
                    variant="selected_dual_grouped_pair2_bf16_bf16_out")
            elif expert_grid64 and qwen4_exp_q4_bundle_prefill_selected(
                backend, weights["expert_gate"].spec.quant_key
            ):
                grouped_q4_gate = resolve(
                    backend=backend, layer="moe_linear",
                    quant=weights["expert_gate"].spec.quant_key,
                    variant="selected_dual_grouped_rowbatch8_out4_expertgrid64_bundle_bf16_bf16_out")
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
                    tile_m=int(
                        os.environ.get(
                            "HIPENGINE_QWEN4_EXP_Q4_TILE_M",
                            "64" if rows >= 512 else "16",
                        )
                    ),
                    tile_n=int(os.environ.get("HIPENGINE_QWEN4_EXP_Q4_TILE_N", "16")),
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
        if weights["expert_down"].spec.quant_key == "gguf_q5_1":
            q5_wmma = production_grouped_moe or os.environ.get(
                "HIPENGINE_QWEN4_EXP_Q5_1_WMMA", ""
            ) not in {"", "0", "false", "False"}
            q5_mmq = (
                os.environ.get("HIPENGINE_QWEN4_EXP_Q5_1_MMQ_PREFILL", "0")
                not in {"", "0", "false", "False"}
                and scratch.q5_1_mmq_ds4_workspace is not None
                and rows >= 2
                and ffn % 128 == 0
                and _qwen4_exp_q5_1_mmq_layer_allowed(weights)
            )
            if q5_mmq:
                down_input_ptr = (
                    scratch.expert_intermediate.ptr
                    if (
                        exact_grouped_q4_gate
                        or production_grouped_moe
                        or q4_k_mmq_prefill
                    )
                    else scratch.expert_gate.ptr
                )
                gguf_q8_1_mmq_ds4_pack_bf16(
                    down_input_ptr,
                    scratch.q5_1_mmq_ds4_workspace.ptr,
                    compact,
                    ffn,
                    stream=stream,
                    runtime=active_runtime,
                )
                gguf_q5_1_mmq_ds4_selected_prefill_bf16_bf16_out(
                    scratch.q5_1_mmq_ds4_workspace.ptr,
                    scratch.group_expert_start.ptr,
                    weights["expert_down"].allocation("raw").tensor.ptr,
                    scratch.expert_down.ptr,
                    compact,
                    experts,
                    ffn,
                    hidden,
                    3,
                    stream=stream,
                    runtime=active_runtime,
                    library=scratch.q5_1_mmq_library,
                )
            elif exact_grouped_down:
                # PF-3 production: fused single-loop logical256 Q5_1 down
                # prefill, bit-exact vs the strict owner. Production selects
                # M1 after the one-process/one-residency canonical gate;
                # strict selects the preceding expertgrid64 owner.
                if os.environ.get(
                    "HIPENGINE_QWEN4_EXP_EXACT_EXPERT_GRID", "64"
                ) in {"64", "q5"}:
                    grouped_q5_variant = (
                        "selected_grouped_prefill_compact_rowbatch8_out8_"
                        "expertgrid64_m1_bf16_bf16_out"
                        if os.environ.get(
                            "HIPENGINE_QWEN4_EXP_PROFILE_Q5_1_DOWN_M1", "0"
                        )
                        not in {"", "0", "false", "False"}
                        else "selected_grouped_prefill_compact_rowbatch8_out8_"
                        "expertgrid64_bf16_bf16_out"
                    )
                elif os.environ.get(
                    "HIPENGINE_QWEN4_EXP_Q5_1_OUT8", "1"
                ) not in {"", "0", "false", "False"}:
                    grouped_q5_variant = (
                        "selected_grouped_prefill_compact_rowbatch8_"
                        "out8_bf16_bf16_out"
                    )
                else:
                    grouped_q5_variant = (
                        "selected_grouped_prefill_compact_rowbatch8_"
                        "bf16_bf16_out"
                    )
                if qwen4_exp_q51_pair_prefill_selected(
                    backend, weights["expert_down"].spec.quant_key,
                    rows=rows, in_features=ffn,
                ):
                    grouped_q5_variant = "selected_grouped_prefill_pair2_bf16_bf16_out"
                grouped_q5_down = resolve(
                    backend=backend,
                    layer="moe_linear",
                    quant=weights["expert_down"].spec.quant_key,
                    variant=grouped_q5_variant,
                )
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
                qwen4_exp_q5_1_selected_grouped_wmma_prefill_compact_bf16_bf16_out(
                    scratch.expert_intermediate.ptr,
                    scratch.group_expert_start.ptr,
                    scratch.group_wmma_expert_start.ptr,
                    scratch.group_tile_expert.ptr,
                    weights["expert_down"].allocation("raw").tensor.ptr,
                    scratch.expert_down.ptr,
                    compact,
                    experts,
                    ffn,
                    hidden,
                    wmma_total_rows,
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
            # RED-pinned in tests/test_qwen4exp_pf1_forkb_selected_down.py);
            # kernel-level paired A/B at the p4096 compact shape measured
            # 32.42 vs 47.42 ms median (-31.6%). The incumbent strict selected
            # gemv and the legacy P1 grouped/WMMA owners stay available below.
            # Production selects the grouped owner after the exact
            # one-process/one-residency canonical gate; strict keeps the
            # per-expert selected gemv.
            if os.environ.get("HIPENGINE_QWEN4_EXP_FORKB_GROUPED_DOWN", "0") not in {
                "", "0", "false", "False",
            }:
                # x (expert_intermediate) and out (expert_down) are already in
                # sorted-lane order here, so the kernel serves row = sorted
                # position directly (lane_to_row=nullptr).
                grouped_q8_down = resolve(
                    backend=backend,
                    layer="linear",
                    quant=weights["expert_down"].spec.quant_key,
                    variant="selected_grouped_gemv_bf16_bf16_out",
                )
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
            # P1: device-driven grouped Q8_0 down owner (no D2H, no Python loop
            # over 512 experts). Reads expert_start on device via a fixed worker
            # grid. The legacy per-expert D2H+WMMA loop and the strict per-expert
            # selected gemv remain available as explicit opt-outs for rollback /
            # bisection until the P1 full packet gate passes.
            elif os.environ.get("HIPENGINE_QWEN4_EXP_Q8_0_GROUPED_WMMA", "") not in {"", "0", "false", "False"}:
                expert_start_host = np.empty(experts + 1, dtype=np.int64)
                copy_device_to_host(
                    host_array_ptr(expert_start_host),
                    scratch.group_expert_start,
                    expert_start_host.nbytes,
                    runtime=active_runtime,
                )
                expert_weight_bytes = hidden * (ffn // 32) * 34
                for expert in range(experts):
                    start_row = int(expert_start_host[expert])
                    expert_rows = int(expert_start_host[expert + 1]) - start_row
                    if expert_rows <= 0:
                        continue
                    gguf_q8_0_wmma_prefill_bf16_bf16_out(
                        scratch.expert_intermediate.ptr
                        + start_row * ffn * DType.BF16.itemsize,
                        weights["expert_down"].allocation("raw").tensor.ptr
                        + expert * expert_weight_bytes,
                        scratch.expert_down.ptr
                        + start_row * hidden * DType.BF16.itemsize,
                        expert_rows,
                        ffn,
                        hidden,
                        stream=stream,
                        runtime=active_runtime,
                    )
            elif os.environ.get("HIPENGINE_QWEN4_EXP_Q8_0_GROUPED", "") not in {"", "0", "false", "False"}:
                gguf_q8_0_selected_grouped_prefill_compact_bf16_bf16_out(
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
            else:
                # Strict fallback: neither opt-in Q8_0 grouped owner is selected
                # (both Q8_0_GROUPED_WMMA and Q8_0_GROUPED are off). Preserve the
                # incumbent strict per-expert selected gemv so Q8_0 down layers
                # (layer 2, 4, 30, 46, 47) are still computed correctly.
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
            # Grouped rows==1, or fused PF-4 lever-2 combine disabled
            # (default after the whole-model A/B measured loss): keep the
            # unfused sum; the rows==1 combine tail uses the single-token
            # shared_gate_combine_out below.
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
        # fallback. Default is the unfused chain: whole-model counterbalanced
        # A/B measured pp −1.69% mean (all 12 cases negative), tg −0.19%.
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
        if use_dp4a:
            gguf_q4_k_quantize_bf16_q8_1(
                scratch.hidden_bf16.ptr,
                scratch.group_gate_up.ptr,
                rows,
                hidden,
                stream=stream,
                runtime=active_runtime,
            )
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
                weights, rows=rows, backend=backend)
            if row4:
                # Map only: projections publish original token-major lanes,
                # leaving SiLU/down/ordered combine ownership unchanged.
                active_runtime.memset(scratch.group_counts.ptr, 0, scratch.group_counts.nbytes)
                qwen35_moe_group_count(
                    scratch.selected.ptr, scratch.group_counts.ptr, compact,
                    experts, stream=stream, runtime=active_runtime)
                qwen35_moe_group_prefix(
                    scratch.group_counts.ptr, scratch.group_padded_counts.ptr,
                    scratch.group_expert_start.ptr, scratch.group_wmma_total.ptr,
                    experts, 1, stream=stream, runtime=active_runtime)
                active_runtime.memset(
                    scratch.group_scatter_offsets.ptr, 0, scratch.group_scatter_offsets.nbytes)
                qwen35_moe_group_scatter(
                    scratch.selected.ptr, scratch.routing.ptr,
                    scratch.group_expert_start.ptr, scratch.group_scatter_offsets.ptr,
                    scratch.group_sorted_lanes.ptr, scratch.group_sorted_experts.ptr,
                    scratch.group_sorted_weights.ptr, compact, experts,
                    stream=stream, runtime=active_runtime)
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
    for slot, output in (
        ("shared_gate", scratch.shared_gate),
        ("shared_up", scratch.shared_up),
    ):
        launch_gguf_linear(
            weights[slot], mixed_ptr, output.ptr,
            rows, hidden, ffn,
            activation_dtype=GGUF_ACTIVATION_F32,
            output_dtype=GGUF_OUTPUT_F32,
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
    f32_to_bf16(
        scratch.shared_down.ptr,
        scratch.shared_down_bf16.ptr,
        rows * hidden,
        stream=stream,
        runtime=active_runtime,
    )
    if rows == 1:
        shared_gate_combine_out_bf16(
            scratch.routed.ptr,
            scratch.shared_down_bf16.ptr,
            scratch.shared_gate_logits.ptr,
            scratch.output.ptr,
            hidden,
            stream=stream,
            runtime=active_runtime,
        )
    elif grouped_prefill and os.environ.get(
        "HIPENGINE_QWEN4_EXP_FUSED_COMBINE", ""
    ) == "1":
        # PF-4 lever 2 T0: fused routed-sum + gated shared combine (bit-exact
        # vs the unfused weighted_lanes_sum -> shared_gate_combine chain;
        # kernel-level A/B 455e176ab). Opt-in only: whole-model A/B measured
        # a prefill loss, so the unfused chain remains the default and the
        # strict fallback via weighted_lanes_sum+shared_add registry keys.
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
    else:
        shared_gate_combine_batch_out_bf16(
            scratch.routed.ptr,
            scratch.shared_down_bf16.ptr,
            scratch.shared_gate_logits.ptr,
            scratch.output.ptr,
            rows,
            hidden,
            stream=stream,
            runtime=active_runtime,
        )
    return Qwen4ExpMoEDeviceResult(scratch.output, scratch.selected, scratch.routing)


def run_qwen4_exp_dense_qsa_layer(
    residual_ptr: int,
    weights: Qwen4ExpQSALayerDeviceWeights,
    *,
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
) -> DeviceBuffer:
    """Execute one complete dense-equivalent Qwen4Exp QSA+MoE layer."""

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
        stream=stream, runtime=active_runtime,
    )
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
    )
    moe = run_qwen4_exp_moe(
        ffn_read.mixed.ptr, weights.moe, scratch=scratch.moe,
        rows=rows, hidden=hidden, ffn=ffn, experts=experts, top_k=top_k,
        stream=stream, runtime=active_runtime,
        graph_cache=moe_graph_cache, graph_key=moe_graph_key,
    )
    bf16_to_f32(
        moe.output.ptr, scratch.moe_f32.ptr, rows * hidden,
        stream=stream, runtime=active_runtime,
    )
    qwen4_exp_gr_write_bf16_f32(
        scratch.after_attention.ptr, scratch.moe_f32.ptr,
        ffn_read.inject_logits.ptr, scratch.output.ptr,
        rows, branches, hidden, stream=stream, runtime=active_runtime,
    )
    return scratch.output


def run_qwen4_exp_qsa_prefill_layer(
    residual_ptr: int,
    weights: Qwen4ExpQSALayerDeviceWeights,
    *,
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
    stream: int = 0,
    runtime: HipRuntime | None = None,
) -> DeviceBuffer:
    """Execute one complete Qwen4Exp QSA+MoE prompt chunk."""

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
        stream=stream,
        runtime=active_runtime,
    )
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
    )
    moe = run_qwen4_exp_moe(
        ffn_read.mixed.ptr,
        weights.moe,
        scratch=scratch.moe,
        rows=rows,
        hidden=hidden,
        ffn=ffn,
        experts=experts,
        top_k=top_k,
        stream=stream,
        runtime=active_runtime,
    )
    bf16_to_f32(
        moe.output.ptr,
        scratch.moe_f32.ptr,
        rows * hidden,
        stream=stream,
        runtime=active_runtime,
    )
    qwen4_exp_gr_write_bf16_f32(
        scratch.after_attention.ptr,
        scratch.moe_f32.ptr,
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
) -> DeviceBuffer:
    """Execute one complete strict Qwen4Exp GDN+MoE physical layer."""

    if scratch.closed:
        raise RuntimeError("Qwen4Exp GDN layer scratch is closed")
    active_runtime = runtime or scratch.runtime
    if active_runtime is not scratch.runtime:
        raise ValueError("runtime must match the GDN layer scratch owner")
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
    )
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
    )
    moe_output = run_qwen4_exp_moe(
        ffn_read.mixed.ptr,
        weights.moe,
        scratch=scratch.moe,
        rows=rows, hidden=hidden, ffn=ffn, experts=experts, top_k=top_k,
        stream=stream, runtime=active_runtime,
        graph_cache=moe_graph_cache, graph_key=moe_graph_key,
    )
    bf16_to_f32(
        moe_output.output.ptr,
        scratch.moe_f32.ptr,
        rows * hidden,
        stream=stream,
        runtime=active_runtime,
    )
    qwen4_exp_gr_write_bf16_f32(
        scratch.after_attention.ptr,
        scratch.moe_f32.ptr,
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
) -> DeviceBuffer:
    """Execute strict Qwen4Exp GDN projections, Conv, recurrence, and output."""

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
    for slot, output, out_features in (
        ("attn_qkv", scratch.qkv, qkv_width),
        ("attn_gate", scratch.gate, core_width),
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
        qwen4_exp_gdn_decode_f32(
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
        exact_bulk_conv = os.environ.get(
            "HIPENGINE_QWEN4_EXP_EXACT_CONV_PREFILL", "1"
        ) not in {"", "0", "false", "False"}
        if exact_bulk_conv:
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
            )
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
            head_dim == 128
            and os.environ.get("HIPENGINE_QWEN4_EXP_GDN_PEER_PREFILL", "")
            not in {"", "0", "false", "False"}
            and _qwen4_exp_gdn_peer_layer_allowed(weights)
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
            if qwen4_exp_gdn_tile16_prefill_selected(
                rows=rows,
                num_k_heads=num_k_heads,
                num_v_heads=num_v_heads,
                head_dim=head_dim,
            ):
                # PF-5 production: exact token-tile-16 owner, bit-exact to
                # the columnwarp parent at the binding Hk16/Hv48/D128 shape.
                qwen4_exp_gdn_prefill_tiled16_f32(
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
                serial_prefill = resolve(
                    backend=backend, layer="gdn_recurrence_norm_gate",
                    quant="f32_state", variant="qwen4exp_sigmoid_register_prefill",
                )
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


class Qwen4ExpGGUFResidentModelRunner:
    """Strict c1 text runner for the complete 48-layer Qwen4Exp target."""

    def __init__(
        self,
        resident: Qwen4ExpResidentWeights,
        *,
        max_sequence_length: int = 2_051,
        prefill_chunk_size: int = 512,
        backend: str = "hip_gfx1151",
        runtime: HipRuntime | None = None,
    ) -> None:
        self.resident = resident
        self.config = resident.plan.config
        self.backend = str(backend)
        self.runtime = runtime or get_hip_runtime()
        self.max_sequence_length = int(max_sequence_length)
        self.prefill_chunk_size = int(prefill_chunk_size)
        if self.prefill_chunk_size <= 0:
            raise ValueError("Qwen4Exp prefill_chunk_size must be positive")
        if not 0 < self.max_sequence_length <= self.config.context_length:
            raise ValueError(
                "Qwen4Exp runner max_sequence_length must be in "
                f"1..{self.config.context_length}"
            )
        load_backend_kernel_package(self.backend)
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
        self.state: Qwen4ExpDecodeState | None = None
        self.gdn_scratch: Qwen4ExpGDNLayerScratch | None = None
        self.qsa_scratch: Qwen4ExpQSALayerScratch | None = None
        self.ple_scratch: Qwen4ExpPLEScratch | None = None
        self.head_scratch: Qwen4ExpGRScratch | None = None
        self._target_verify_output: Qwen4ExpTargetVerifyOutput | None = None
        self._device_transaction_snapshot: Qwen4ExpDecodeStateDeviceSnapshot | None = None
        self._device_transaction_lease = False
        self.gdn_prefill_scratch: Qwen4ExpGDNLayerScratch | None = None
        self.qsa_prefill_scratch: Qwen4ExpQSALayerScratch | None = None
        self.ple_prefill_scratch: Qwen4ExpPLEScratch | None = None
        self.qsa_prefill_metadata: Qwen4ExpQSAPrefillMetadata | None = None
        self.attention_states: tuple[Qwen4ExpDenseAttentionState, ...] = ()
        self.index_states: tuple[Qwen4ExpQSAIndexDeviceState, ...] = ()
        self._buffers: list[DeviceBuffer] = []
        self._prefill_buffers: list[DeviceBuffer] = []
        self._ple_hash_states: dict[int, PLEHashState] = {}
        self.moe_graph_cache: MoeGraphCache | None = None
        self.position = 0
        self.closed = False
        try:
            self._allocate()
            graph_override = os.environ.get("HIPENGINE_QWEN4_EXP_MOE_GRAPH")
            graph_enabled = bool(
                backend_package_capability(
                    self.backend, "QWEN4_EXP_MOE_GRAPH", False
                )
                if graph_override is None
                else graph_override not in {"", "0", "false", "False"}
            )
            self.moe_graph_cache = MoeGraphCache(
                self.runtime, enabled=graph_enabled
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
        self.attention_states = tuple(
            Qwen4ExpDenseAttentionState.allocate(
                max_positions=self.max_sequence_length,
                block_size=256,
                kv_heads=cfg.attention_kv_head_count,
                head_dim=cfg.attention_key_length,
                runtime=self.runtime,
            )
            for _ in range(cfg.layer_types.count("qsa"))
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
        prefill_rows = min(self.prefill_chunk_size, self.max_sequence_length)
        self.gdn_prefill_scratch = Qwen4ExpGDNLayerScratch.allocate(
            rows=prefill_rows, branches=cfg.residual_branch_count, hidden=cfg.hidden_size,
            low_rank=cfg.residual_low_rank, qkv_width=qkv_width,
            core_width=cfg.gdn_inner_size, scalar_width=cfg.gdn_time_step_rank,
            ffn=cfg.expert_feed_forward_length, experts=cfg.expert_count,
            top_k=cfg.expert_used_count, runtime=self.runtime,
        )
        self.qsa_prefill_scratch = Qwen4ExpQSALayerScratch.allocate(
            rows=prefill_rows, branches=cfg.residual_branch_count, hidden=cfg.hidden_size,
            low_rank=cfg.residual_low_rank, query_heads=cfg.attention_head_count,
            kv_heads=cfg.attention_kv_head_count, head_dim=cfg.attention_key_length,
            ffn=cfg.expert_feed_forward_length, experts=cfg.expert_count,
            top_k=cfg.expert_used_count, index_heads=cfg.indexer_head_count,
            index_dim=cfg.indexer_key_length, runtime=self.runtime,
        )
        self.ple_prefill_scratch = Qwen4ExpPLEScratch.allocate(
            rows=prefill_rows, branches=cfg.residual_branch_count,
            hidden=cfg.hidden_size, runtime=self.runtime,
        )
        self.qsa_prefill_metadata = Qwen4ExpQSAPrefillMetadata.allocate(
            self.attention_states[0],
            rows=prefill_rows,
            selection_capacity=cfg.qsa_dense_equivalent_max_tokens,
            score_blocks=(
                self.max_sequence_length + cfg.qsa_compression_ratio - 1
            )
            // cfg.qsa_compression_ratio,
        )
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
        for nbytes in (
            prefill_rows * np.dtype(np.int64).itemsize,
            prefill_rows * cfg.hidden_size * DType.BF16.itemsize,
            prefill_rows * cfg.hidden_size * DType.FP32.itemsize,
            prefill_rows * cfg.residual_branch_count * cfg.hidden_size * DType.BF16.itemsize,
            3 * prefill_rows * np.dtype(np.int64).itemsize,
        ):
            self._prefill_buffers.append(malloc(nbytes, runtime=self.runtime))
        self._q8_mmq_policy = None
        self._q8_mmq_library = None
        self._q8_mmq_buffers: tuple[DeviceBuffer, ...] = ()
        self._configure_q8_mmq_prefill_resources()

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

    def configure_mmq_prefill_resources(self) -> None:
        """Allocate profile-selected MMQ resources after cold-path binding."""

        if self.closed:
            raise RuntimeError("Qwen4Exp runner is closed")
        self._configure_q8_mmq_prefill_resources()
        cfg = self.config
        prefill_rows = min(self.prefill_chunk_size, self.max_sequence_length)
        for owner in (self.gdn_prefill_scratch, self.qsa_prefill_scratch):
            if owner is None:
                continue
            _configure_qwen4_exp_moe_mmq_scratch(
                owner.moe,
                rows=prefill_rows,
                hidden=cfg.hidden_size,
                ffn=cfg.expert_feed_forward_length,
                top_k=cfg.expert_used_count,
                runtime=self.runtime,
            )

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
        return q8_mmq_prefill_session(
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
        )

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
        position = int(transaction.position)
        cursor = max(position - 1, 0)
        for attention in self.attention_states:
            attention.set_position(cursor)
        for index in self.index_states:
            index.restore_count(position)
        self._ple_hash_states = dict(transaction.ple_hash_states)
        self.position = position
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
        cursor = max(position - 1, 0)
        for attention in self.attention_states:
            attention.set_position(cursor)
        for index in self.index_states:
            index.restore_count(position)
        self._ple_hash_states = dict(snapshot.ple_hash_states)
        self.position = position

    def reset(self) -> None:
        self._require_open()
        assert self.state is not None
        self.state.zero()
        for attention in self.attention_states:
            self.runtime.memset(attention.key_cache.ptr, 0, attention.key_cache.nbytes)
            self.runtime.memset(attention.value_cache.ptr, 0, attention.value_cache.nbytes)
            attention.set_position(0)
        for index in self.index_states:
            index.reset()
        self._ple_hash_states = {}
        self.position = 0

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
            copy_host_to_device(
                self.rope_position_buffer,
                host_array_ptr(rope_host),
                rope_host.nbytes,
                runtime=self.runtime,
            )
            rope_positions_ptr = self.rope_position_buffer.ptr
        if token_host[0] < 0 or token_host[0] >= cfg.vocab_size:
            raise ValueError("token_id is outside Qwen4Exp vocabulary")
        if not token_id_resident:
            copy_host_to_device(
                self.token_id_buffer,
                host_array_ptr(token_host),
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
        rows, self._ple_hash_states = ple_hash_rows(
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
        staged = self.resident.ple_staging.stage(rows[0]).reshape(1, cfg.hidden_size)
        ple_telemetry = self.resident.ple_table.telemetry() is not None
        ple_h2d_started = time.perf_counter_ns() if ple_telemetry else 0
        copy_host_to_device(
            self.ple_embedding_buffer,
            host_array_ptr(staged),
            staged.nbytes,
            runtime=self.runtime,
        )
        if ple_h2d_started:
            self.resident.ple_staging.record_h2d(
                nbytes=staged.nbytes,
                wall_ns=time.perf_counter_ns() - ple_h2d_started,
            )
        residual_ptr = self.state.residual.ptr
        gdn_conv_row_bytes = (
            (2 * cfg.gdn_group_count * cfg.gdn_state_size + cfg.gdn_inner_size)
            * cfg.gdn_conv_kernel * 4
        )
        gdn_matrix_row_bytes = (
            cfg.gdn_time_step_rank * cfg.gdn_state_size * cfg.gdn_state_size * 4
        )
        for layer, kind in enumerate(cfg.layer_types):
            if layer in cfg.ple_layers:
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
                    moe_graph_cache=self.moe_graph_cache,
                    moe_graph_key=(
                        "gdn", layer,
                        os.environ.get("HIPENGINE_QWEN4_EXP_Q4_DP4A64", ""),
                        os.environ.get(
                            "HIPENGINE_QWEN4_EXP_Q4_DP4A64_LAYERS", ""
                        ),
                    ),
                ).ptr
            else:
                binding = self.qsa_bindings[layer]
                residual_ptr = run_qwen4_exp_dense_qsa_layer(
                    residual_ptr,
                    binding,
                    attention_state=self.attention_states[binding.qsa_state_index],
                    index_state=self.index_states[binding.qsa_state_index],
                    scratch=self.qsa_scratch,
                    position=self.position,
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
                    moe_graph_cache=self.moe_graph_cache,
                    moe_graph_key=(
                        "qsa", layer,
                        os.environ.get("HIPENGINE_QWEN4_EXP_Q4_DP4A64", ""),
                        os.environ.get(
                            "HIPENGINE_QWEN4_EXP_Q4_DP4A64_LAYERS", ""
                        ),
                    ),
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

    def _prefill_chunk(
        self,
        token_ids: tuple[int, ...],
        *,
        capture_hidden_seeds: bool = False,
        embedding_overrides: Mapping[int, np.ndarray] | None = None,
        mrope_positions: np.ndarray | None = None,
    ) -> tuple[int, np.ndarray | None]:
        count = len(token_ids)
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
        copy_host_to_device(
            self.prefill_token_ids,
            host_array_ptr(token_host),
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
            copy_host_to_device(
                self.prefill_rope_positions,
                host_array_ptr(rope_host),
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
                copy_host_to_device(
                    DeviceBuffer(
                        self.prefill_embeddings.ptr
                        + (position - start) * row_nbytes,
                        row_nbytes,
                    ),
                    host_array_ptr(bits),
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
        copy_host_to_device(
            self.prefill_ple_embeddings,
            host_array_ptr(staged),
            staged.nbytes,
            runtime=self.runtime,
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
        for layer, kind in enumerate(cfg.layer_types):
            if layer in cfg.ple_layers:
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
                ).ptr
            else:
                binding = self.qsa_bindings[layer]
                residual_ptr = run_qwen4_exp_qsa_prefill_layer(
                    residual_ptr,
                    binding,
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
                    ffn=cfg.expert_feed_forward_length,
                    experts=cfg.expert_count,
                    top_k=cfg.expert_used_count,
                    runtime=self.runtime,
                ).ptr
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

    def prefill_chunked(
        self,
        token_ids: list[int] | tuple[int, ...],
        *,
        capture_hidden_seeds: bool = False,
        capture_logits: bool = True,
        capture_target_hidden: bool = True,
        embedding_overrides: Mapping[int, np.ndarray] | None = None,
        mrope_positions: np.ndarray | None = None,
    ) -> Qwen4ExpTokenResult:
        if not token_ids:
            raise ValueError("Qwen4Exp prefill requires at least one token")
        if len(token_ids) > self.max_sequence_length:
            raise ValueError("Qwen4Exp prefill exceeds sequence capacity")
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

        with (
            raw_k_prefill_rowbatch_session(selected_raw_rowbatch),
            raw_k_prefill_variant_session(selected_raw_variant),
            wmma_prefill_session(True if q8_wmma_layers else None),
            wmma_prefill_weight_filter_session(
                q8_wmma_weight_filter if q8_wmma_layers else None
            ),
            self._q8_mmq_prefill_context(),
        ):
            for start in range(0, len(values), self.prefill_chunk_size):
                last_residual_ptr, hidden_rows = self._prefill_chunk(
                    values[start : start + self.prefill_chunk_size],
                    capture_hidden_seeds=capture_hidden_seeds,
                    embedding_overrides=embedding_overrides,
                    mrope_positions=(
                        None
                        if rope_values is None
                        else np.ascontiguousarray(
                            rope_values[:, start : start + self.prefill_chunk_size]
                        )
                    ),
                )
                if hidden_rows is not None:
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

    def prefill(
        self,
        token_ids: list[int] | tuple[int, ...],
        *,
        capture_hidden_seeds: bool = False,
        capture_logits: bool = True,
        capture_target_hidden: bool = True,
        embedding_overrides: Mapping[int, np.ndarray] | None = None,
        mrope_positions: np.ndarray | None = None,
    ) -> Qwen4ExpTokenResult:
        return self.prefill_chunked(
            token_ids,
            capture_hidden_seeds=capture_hidden_seeds,
            capture_logits=capture_logits,
            capture_target_hidden=capture_target_hidden,
            embedding_overrides=embedding_overrides,
            mrope_positions=mrope_positions,
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
        if self.moe_graph_cache is not None:
            self.moe_graph_cache.close()
            self.moe_graph_cache = None
        if self._target_verify_output is not None:
            self._target_verify_output.close()
            self._target_verify_output = None
        if self._device_transaction_snapshot is not None:
            self._device_transaction_snapshot.close()
            self._device_transaction_snapshot = None
            self._device_transaction_lease = False
        self._q8_mmq_policy = None
        self._q8_mmq_library = None
        self._q8_mmq_buffers = ()
        for buffer in reversed(self._prefill_buffers):
            free(buffer, runtime=self.runtime)
        self._prefill_buffers = []
        for buffer in reversed(self._buffers):
            free(buffer, runtime=self.runtime)
        self._buffers = []
        for state in reversed(self.index_states):
            state.close()
        self.index_states = ()
        for state in reversed(self.attention_states):
            state.close()
        self.attention_states = ()
        for owner in (
            self.qsa_prefill_metadata,
            self.ple_prefill_scratch,
            self.qsa_prefill_scratch,
            self.gdn_prefill_scratch,
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
