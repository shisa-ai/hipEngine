"""Dedicated Qwen4Exp GGUF runtime composition.

This module begins with the strict gated-residual read owner used by every
Qwen3.8-Flash-Next token mixer and MoE. It is model-specific composition over
model-neutral GGUF projections and registered gfx11 primitives; it does not add
architecture branches to the engine or dispatch layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

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
from hipengine.kernels.backends import load_backend_kernel_package
from hipengine.kernels.hip_gfx1100.attention.paged_attn_decode import (
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
    qwen4_exp_qsa_pool_norm_rope_f32,
    qwen4_exp_qsa_score_f32,
    qwen4_exp_qsa_sparse_attention_paged_bf16_f32,
    qwen4_exp_qsa_sparse_attention_paged_bf16_rows_f32,
    qwen4_exp_qsa_split_norm_rope_f32,
    qwen4_exp_qsa_split_norm_rope_rows_f32,
)
from hipengine.kernels.hip_gfx1100.convert.cast import bf16_to_f32, f32_to_bf16
from hipengine.kernels.hip_gfx1100.fused.paro_combine import (
    shared_gate_combine_batch_out_bf16,
    shared_gate_combine_out_bf16,
    weighted_sum_batch_out_bf16_f32w,
    weighted_sum_out_bf16_f32w,
)
from hipengine.kernels.hip_gfx1100.fused.paro_silu import (
    silu_mul_separate_out_bf16,
)
from hipengine.kernels.hip_gfx1100.moe.router import qwen35_router_select
from hipengine.kernels.hip_gfx1100.linear_attn.conv import (
    qwen35_linear_attn_conv_decode_f32,
    qwen35_linear_attn_conv_prefill_f32,
)
from hipengine.kernels.hip_gfx1100.linear_attn.qwen4_exp_gdn import (
    qwen4_exp_gdn_decode_f32,
    qwen4_exp_gdn_prefill_f32,
)
from hipengine.kernels.hip_gfx1100.fused.qwen4_exp_gr import (
    qwen4_exp_gated_mean_f32,
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
from hipengine.runtime.gguf_linear import (
    GGUF_ACTIVATION_F32,
    GGUF_OUTPUT_F32,
    launch_gguf_linear,
)
from hipengine.kernels.cpu_reference.qwen4_exp import (
    PLEHashState,
    QSASelection,
    ple_hash_rows,
    qsa_index_scores,
    qsa_prepare_index_keys,
    qsa_select_positions,
)
from hipengine.kernels.registry import resolve
from hipengine.kvcache import KVLiveSpans
from hipengine.loading.qwen4_exp_materialize import Qwen4ExpResidentWeights
from hipengine.runtime.gguf_weight import GGUFDeviceWeight


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
    block_tables_host: np.ndarray
    positions_host: np.ndarray
    context_counts_host: np.ndarray
    selected_positions_host: np.ndarray
    selected_counts_host: np.ndarray
    rows: int
    block_table_len: int
    selection_capacity: int
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
    ) -> "Qwen4ExpQSAPrefillMetadata":
        if attention_state.closed:
            raise RuntimeError("QSA prefill metadata requires an open attention state")
        count = int(rows)
        selected = int(selection_capacity)
        if count <= 0 or selected <= 0:
            raise ValueError("QSA prefill rows and selection capacity must be positive")
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
        qwen4_exp_qsa_pool_norm_rope_f32(
            self.raw_keys.ptr,
            self.member_indices.ptr,
            self.block_starts.ptr,
            int(key_norm_weight_ptr),
            self.pooled_keys.ptr,
            blocks,
            self.compression_ratio,
            self.index_dim,
            rotary_dim,
            theta,
            eps,
            stream=stream,
            runtime=self.runtime,
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
        selected = self.select_positions_host(
            prepared_query_ptr,
            query_position=position,
            key_norm_weight_ptr=key_norm_weight_ptr,
            rotary_dim=rotary_dim,
            theta=theta,
            eps=eps,
            stream=stream,
        )
        copy_host_to_device(
            self.selected_starts,
            host_array_ptr(self.selected_starts_host),
            runtime=self.runtime,
        )
        copy_host_to_device(
            self.selected_count,
            host_array_ptr(self.selected_count_host),
            runtime=self.runtime,
        )
        copy_host_to_device(
            self.selected_positions,
            host_array_ptr(selected),
            selected.nbytes,
            runtime=self.runtime,
        )
        return int(selected.size), self.selected_positions

    def restore_count(self, count: int) -> None:
        if self.closed:
            raise RuntimeError("QSA index state is closed")
        restored = int(count)
        if restored < 0 or restored > self.capacity:
            raise ValueError("QSA index restore count exceeds capacity")
        self.count = restored

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
    runtime: HipRuntime
    closed: bool = False

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
            )
        ):
            free(buffer, runtime=self.runtime)
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
    attention_state.set_position(position)
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
        index_state.append(scratch.index_k_projected.ptr, position=position, stream=stream)
        if index_state.count > index_state.dense_equivalent_limit:
            launch_gguf_linear(
                weights.projections["index_q"], mixed_ptr, scratch.index_q_projected.ptr,
                rows, hidden, index_heads * index_dim,
                activation_dtype=GGUF_ACTIVATION_F32,
                output_dtype=GGUF_OUTPUT_F32,
                stream=stream, runtime=active_runtime,
            )
            qwen4_exp_qsa_norm_rope_f32(
                scratch.index_q_projected.ptr,
                weights.index_q_norm_weight_ptr,
                attention_state.position.ptr,
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
    qwen4_exp_qsa_split_norm_rope_f32(
        scratch.q_projected.ptr,
        scratch.key_projected.ptr,
        weights.q_norm_weight_ptr,
        weights.k_norm_weight_ptr,
        attention_state.position.ptr,
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
            position + 1,
            attention_state.block_size,
            query_heads,
            kv_heads,
            head_dim,
            head_dim ** -0.5,
            stream=stream,
            runtime=active_runtime,
        )
    else:
        qwen4_exp_qsa_sparse_attention_paged_bf16_f32(
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
    qwen4_exp_qsa_split_norm_rope_rows_f32(
        scratch.q_projected.ptr,
        scratch.key_projected.ptr,
        weights.q_norm_weight_ptr,
        weights.k_norm_weight_ptr,
        metadata.positions.ptr,
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
    qwen4_exp_qsa_norm_rope_rows_f32(
        scratch.index_q_projected.ptr,
        weights.index_q_norm_weight_ptr,
        metadata.positions.ptr,
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
    for row in range(count):
        index_state.append(
            scratch.index_k_projected.ptr + row * index_dim * DType.FP32.itemsize,
            position=start + row,
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
    if dense_rows:
        qwen35_paged_full_attn_decode_context_bf16_batch_spans(
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
        for local_row in range(dense_rows, count):
            position = start + local_row
            selected = index_state.select_positions_host(
                scratch.index_query.ptr
                + local_row * index_heads * index_dim * DType.FP32.itemsize,
                query_position=position,
                key_norm_weight_ptr=weights.index_k_norm_weight_ptr,
                rotary_dim=index_rotary_dim,
                theta=theta,
                eps=eps,
                stream=stream,
            )
            if selected.size > metadata.selection_capacity:
                raise ValueError("QSA selection exceeds prefill metadata capacity")
            metadata.selected_positions_host[local_row, : selected.size] = selected
            metadata.selected_counts_host[local_row] = selected.size
        metadata.upload_selections(count)
        qwen4_exp_qsa_sparse_attention_paged_bf16_rows_f32(
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
    qwen4_exp_sigmoid_f32(
        scratch.gate.ptr,
        scratch.gate.ptr,
        rows * residual_width,
        stream=stream,
        runtime=active_runtime,
    )
    if inject_weight is None:
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
    return Qwen4ExpGRReadDeviceResult(
        normalized=scratch.normalized,
        gate=scratch.gate,
        mixed=scratch.mixed,
        inject_logits=scratch.inject_logits,
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
) -> Qwen4ExpMoEDeviceResult:
    """Run normalized softmax top-k routed experts plus gated shared expert."""

    if scratch.closed:
        raise RuntimeError("Qwen4Exp MoE scratch is closed")
    if rows <= 0:
        raise ValueError("Qwen4Exp MoE rows must be positive")
    active_runtime = runtime or scratch.runtime
    if active_runtime is not scratch.runtime:
        raise ValueError("runtime must match the MoE scratch owner")
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
    ) -> None:
        weight = weights[slot]
        function = resolve(
            backend=backend,
            layer="linear",
            quant=weight.spec.quant_key,
            variant="selected_gemv_bf16_bf16_out",
        )
        function(
            input_ptr,
            scratch.selected.ptr,
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
    selected_projection(
        "expert_down", scratch.expert_intermediate.ptr, scratch.expert_down.ptr,
        compact, compact, ffn, hidden,
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
    stream: int = 0,
    runtime: HipRuntime | None = None,
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
        qwen35_linear_attn_conv_prefill_f32(
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
        qwen4_exp_gdn_prefill_f32(
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


@dataclass(frozen=True)
class Qwen4ExpRunnerSnapshot:
    decode_state: Qwen4ExpDecodeStateSnapshot
    position: int
    ple_hash_states: Mapping[int, PLEHashState]


@dataclass(frozen=True)
class Qwen4ExpTokenResult:
    token_id: int
    logits: np.ndarray


class Qwen4ExpGGUFResidentModelRunner:
    """Strict c1 text runner for the complete 48-layer Qwen4Exp target."""

    def __init__(
        self,
        resident: Qwen4ExpResidentWeights,
        *,
        max_sequence_length: int = 2_051,
        prefill_chunk_size: int = 2,
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
        self.gdn_prefill_scratch: Qwen4ExpGDNLayerScratch | None = None
        self.qsa_prefill_scratch: Qwen4ExpQSALayerScratch | None = None
        self.ple_prefill_scratch: Qwen4ExpPLEScratch | None = None
        self.qsa_prefill_metadata: Qwen4ExpQSAPrefillMetadata | None = None
        self.attention_states: tuple[Qwen4ExpDenseAttentionState, ...] = ()
        self.index_states: tuple[Qwen4ExpQSAIndexDeviceState, ...] = ()
        self._buffers: list[DeviceBuffer] = []
        self._prefill_buffers: list[DeviceBuffer] = []
        self._ple_hash_states: dict[int, PLEHashState] = {}
        self.position = 0
        self.closed = False
        try:
            self._allocate()
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
        )
        for nbytes in (
            np.dtype(np.int64).itemsize,
            cfg.hidden_size * 2,
            cfg.hidden_size * 4,
            cfg.vocab_size * 4,
        ):
            self._buffers.append(malloc(nbytes, runtime=self.runtime))
        for nbytes in (
            prefill_rows * np.dtype(np.int64).itemsize,
            prefill_rows * cfg.hidden_size * DType.BF16.itemsize,
            prefill_rows * cfg.hidden_size * DType.FP32.itemsize,
            prefill_rows * cfg.residual_branch_count * cfg.hidden_size * DType.BF16.itemsize,
        ):
            self._prefill_buffers.append(malloc(nbytes, runtime=self.runtime))

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

    def step(self, token_id: int) -> Qwen4ExpTokenResult:
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
        if token_host[0] < 0 or token_host[0] >= cfg.vocab_size:
            raise ValueError("token_id is outside Qwen4Exp vocabulary")
        copy_host_to_device(
            self.token_id_buffer, host_array_ptr(token_host), runtime=self.runtime
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
        copy_host_to_device(
            self.ple_embedding_buffer,
            host_array_ptr(staged),
            staged.nbytes,
            runtime=self.runtime,
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
                    ffn=cfg.expert_feed_forward_length,
                    experts=cfg.expert_count,
                    top_k=cfg.expert_used_count,
                    runtime=self.runtime,
                ).ptr
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
        self.runtime.device_synchronize()
        logits = np.empty(cfg.vocab_size, dtype=np.float32)
        copy_device_to_host(
            host_array_ptr(logits), self.logits_buffer, runtime=self.runtime
        )
        result = Qwen4ExpTokenResult(int(np.argmax(logits)), logits)
        self.position += 1
        return result

    def _prefill_chunk(self, token_ids: tuple[int, ...]) -> int:
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
        qwen4_exp_repeat_bf16_branches(
            self.prefill_embeddings.ptr,
            self.prefill_residual.ptr,
            cfg.residual_branch_count,
            cfg.hidden_size,
            rows=count,
            runtime=self.runtime,
        )
        start = self.position
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
        staged = np.ascontiguousarray(
            np.stack([self.resident.ple_staging.stage(row) for row in rows]),
            dtype=np.float32,
        ).reshape(count, cfg.hidden_size)
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
                    ffn=cfg.expert_feed_forward_length,
                    experts=cfg.expert_count,
                    top_k=cfg.expert_used_count,
                    runtime=self.runtime,
                ).ptr
        self.position += count
        return residual_ptr + (
            (count - 1)
            * cfg.residual_branch_count
            * cfg.hidden_size
            * DType.BF16.itemsize
        )

    def prefill_serial(self, token_ids: list[int] | tuple[int, ...]) -> Qwen4ExpTokenResult:
        if not token_ids:
            raise ValueError("Qwen4Exp prefill requires at least one token")
        self.reset()
        result = None
        for token in token_ids:
            result = self.step(int(token))
        assert result is not None
        return result

    def prefill_chunked(self, token_ids: list[int] | tuple[int, ...]) -> Qwen4ExpTokenResult:
        if not token_ids:
            raise ValueError("Qwen4Exp prefill requires at least one token")
        if len(token_ids) > self.max_sequence_length:
            raise ValueError("Qwen4Exp prefill exceeds sequence capacity")
        self.reset()
        last_residual_ptr = 0
        values = tuple(int(token) for token in token_ids)
        for start in range(0, len(values), self.prefill_chunk_size):
            last_residual_ptr = self._prefill_chunk(
                values[start : start + self.prefill_chunk_size]
            )
        cfg = self.config
        assert self.head_scratch is not None
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
        self.runtime.device_synchronize()
        logits = np.empty(cfg.vocab_size, dtype=np.float32)
        copy_device_to_host(
            host_array_ptr(logits), self.logits_buffer, runtime=self.runtime
        )
        return Qwen4ExpTokenResult(int(np.argmax(logits)), logits)

    def prefill(self, token_ids: list[int] | tuple[int, ...]) -> Qwen4ExpTokenResult:
        return self.prefill_serial(token_ids)

    def generate(
        self,
        token_ids: list[int] | tuple[int, ...],
        *,
        max_new_tokens: int,
    ) -> tuple[int, ...]:
        count = int(max_new_tokens)
        if count <= 0:
            raise ValueError("max_new_tokens must be positive")
        result = self.prefill(token_ids)
        output: list[int] = []
        for index in range(count):
            output.append(result.token_id)
            if index + 1 < count:
                result = self.step(result.token_id)
        return tuple(output)

    def close(self) -> None:
        if self.closed:
            return
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
    "Qwen4ExpTokenResult",
    "Qwen4ExpQSALayerDeviceWeights",
    "Qwen4ExpQSALayerScratch",
    "Qwen4ExpQSAMixerDeviceWeights",
    "Qwen4ExpQSAScratch",
    "run_qwen4_exp_dense_qsa_layer",
    "run_qwen4_exp_dense_qsa_token_mixer",
    "run_qwen4_exp_qsa_prefill_token_mixer",
    "run_qwen4_exp_gdn_layer",
    "run_qwen4_exp_qsa_prefill_layer",
    "run_qwen4_exp_gdn_token_mixer",
    "run_qwen4_exp_moe",
    "run_qwen4_exp_ple",
    "run_qwen4_exp_gr_read",
]
