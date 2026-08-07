"""Resident torch-free Maple ternary decode runner."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import numpy as np

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.hip import HipRuntime, get_hip_runtime
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
    hip_target_arch_environment,
    hip_target_arch_for_backend,
    load_backend_kernel_package,
    resolve_backend,
)
from hipengine.kernels.hip_gfx1100.attention.maple_attention import (
    build_maple_attention,
    maple_attention_decode_batched_bf16,
    maple_attention_decode_bf16,
    maple_attention_fused_qknorm_decode_bf16,
    maple_attention_prefill_ring_bf16,
    maple_attention_prefill_ring_gqa4_bf16,
    maple_kv_span_update,
    maple_kv_span_update_batched,
    maple_qknorm_rope_kv_write_batched_bf16,
    maple_qknorm_rope_kv_write_batched_decode_bf16,
    maple_qknorm_rope_kv_write_bf16,
)
from hipengine.kernels.hip_gfx1100.linear.lm_head import (
    argmax_f32,
    argmax_f32_rows_i32,
    build_lm_head,
)
from hipengine.kernels.hip_gfx1100.moe.group_scatter import (
    build_qwen35_moe_group_scatter,
    qwen35_moe_group_compact_active_i32_parallel,
)
from hipengine.kernels.hip_gfx1100.moe.maple_moe import (
    build_maple_moe,
    maple_clamped_swiglu_bf16,
    maple_router_topk_parallel_batched_bf16,
    maple_router_topk_parallel_bf16,
    maple_router_topk_single_dispatch_bf16,
    maple_weighted_residual_batched_bf16,
    maple_weighted_residual_bf16,
)
from hipengine.kernels.hip_gfx1100.norm.rmsnorm import (
    build_qwen35_rmsnorm,
    paro_add_rmsnorm_out_bf16,
    paro_rmsnorm_out_bf16,
)
from hipengine.kernels.hip_gfx1100.quant.maple_ternary import (
    build_maple_ternary,
    maple_affine4_embed_batched_bf16,
    maple_affine4_embed_bf16,
    maple_affine4_gemv_batched_f32,
    maple_affine4_gemv_batched_rowreuse_exact_f32,
    maple_affine4_gemv_f32,
    maple_affine4_gemv_wave32_exact_f32,
    maple_moe_dual_swiglu_bf16,
    maple_selected_ternary_dual_gemv_batched_bf16,
    maple_selected_ternary_dual_gemv_bf16,
    maple_selected_ternary_dual_grouped_bf16,
    maple_selected_ternary_gemv_batched_bf16,
    maple_selected_ternary_gemv_bf16,
    maple_selected_ternary_grouped_bf16,
    maple_ternary_gemm_bf16,
    maple_ternary_gemv_bf16,
    maple_ternary_qkv_gemm_bf16,
    maple_ternary_qkv_gemv_bf16,
)
from hipengine.kvcache import KVLiveSpans
from hipengine.loading.maple import (
    MapleCheckpoint,
    MapleDeviceWeights,
    materialize_maple_weights,
)
from hipengine.models.maple import MapleModelSpec
from hipengine.runtime.maple_graph import MapleGraphCache


def _maple_fuse_moe() -> bool:
    # Default off: the fused maple_moe_dual_swiglu_bf16 is bit-exact and cuts
    # decode launches 295->271, but the fused kernel is ~9% slower per MoE
    # layer than the unfused dual+swiglu pair (interleaved micro-benchmark).
    # That kernel-efficiency regression matters in the optional M1 c1 hipGraph
    # path where launch overhead is amortized, so the unfused chain stays the
    # default and the fusion is opt-in via HIPENGINE_MAPLE_FUSE_MOE=1 pending an
    # efficiency fix. The M6 helper does not use this graph. See docs/REFACTOR.md.
    return os.environ.get("HIPENGINE_MAPLE_FUSE_MOE", "0") != "0"


def _maple_fuse_qkattn() -> bool:
    # Opt-in: fuse the per-layer qknorm_rope_kv_write + attention_decode pair
    # into one kernel (maple_attention_fused_qknorm_decode_bf16). Default off;
    # see docs/REFACTOR.md.
    return os.environ.get("HIPENGINE_MAPLE_FUSE_QKATTN", "0") != "0"


def _maple_router_single_dispatch() -> bool:
    # D0 exact last-block router composite is the c1 default after the complete
    # byte-state and same-resident wall gates. Preserve =0 as the two-dispatch
    # rollback.
    return os.environ.get("HIPENGINE_MAPLE_ROUTER_SINGLE_DISPATCH", "1") != "0"


def _maple_affine4_wave32_exact() -> bool:
    # D0 exact one-wave head is the c1/final-row default after the complete
    # category/state and same-resident wall gates. Preserve =0 as the exact
    # group64 rollback.
    return os.environ.get("HIPENGINE_MAPLE_AFFINE4_WAVE32_EXACT", "1") != "0"


def _maple_batch_affine4_rowreuse_exact() -> bool:
    # D1 exact row-reuse head is the c2/c4/c8 default after the complete width,
    # natural/heldout, sparse/reclaimed-slot, lifecycle, and wall gates.
    # Preserve =0 as the original all-row exact rollback.
    return os.environ.get(
        "HIPENGINE_MAPLE_BATCH_AFFINE4_ROWREUSE_EXACT", "1"
    ) != "0"


def _maple_batch_affine4_head(rows: int):
    if rows in (2, 4, 8) and _maple_batch_affine4_rowreuse_exact():
        return maple_affine4_gemv_batched_rowreuse_exact_f32
    return maple_affine4_gemv_batched_f32


def _maple_prefill_grouped_moe() -> bool:
    # P1 exact expert-major prefill is the default. Preserve the original
    # row/route-gather chain as HIPENGINE_MAPLE_PREFILL_GROUPED_MOE=0 rollback.
    return os.environ.get("HIPENGINE_MAPLE_PREFILL_GROUPED_MOE", "1") != "0"


def _maple_prefill_gqa4() -> bool:
    # P2 exact wave32 GQA4 is the prefill default after the complete byte-state
    # and physical profile gates. Preserve local128 as =0 rollback.
    return os.environ.get("HIPENGINE_MAPLE_PREFILL_GQA4", "1") != "0"


def _maple_graph_enabled() -> bool:
    # Default off: a controlled c1 review measured whole-step graph replay at
    # only 1.0047x (+0.47%, 0.033 ms), bit-exact and within noise. Keep eager as
    # the default and expose the graph as an opt-in (HIPENGINE_MAPLE_GRAPH=1).
    # The M6 helper has a separate
    # eager-batched chain. The KVLiveSpans device-pointer ABI makes this c1 step
    # stateless across tokens, so a single graph stays valid across positions.
    return os.environ.get("HIPENGINE_MAPLE_GRAPH", "0") != "0"


@dataclass(frozen=True)
class MapleRunnerLibraries:
    ternary: object
    attention: object
    moe: object
    norm: object
    lm_head: object
    group_scatter: object | None = None


# Batched prefill chunk size (rows) per docs/TUNING-gfx1151.md Lesson 0.
PREFILL_CHUNK = 256


@dataclass
class _PrefillBuffers:
    """T-row scratch buffers for the batched prefill path."""

    hidden: DeviceBuffer
    normalized: DeviceBuffer
    residual: DeviceBuffer
    qkv: DeviceBuffer
    attention: DeviceBuffer
    projection: DeviceBuffer
    selected_ids: DeviceBuffer
    routing_weights: DeviceBuffer
    router_logits: DeviceBuffer
    expert_gate: DeviceBuffer
    expert_up: DeviceBuffer
    expert_intermediate: DeviceBuffer
    expert_down: DeviceBuffer
    expert_start: DeviceBuffer
    active_experts: DeviceBuffer
    active_count: DeviceBuffer
    sorted_lanes: DeviceBuffer
    sorted_experts: DeviceBuffer
    sorted_weights: DeviceBuffer
    token_ids: DeviceBuffer

    def __init__(
        self,
        *,
        owner: _BufferOwner,
        spec: MapleModelSpec,
        top_k: int,
        intermediate: int,
        T: int,
    ) -> None:
        h = spec.hidden_size
        qkv = spec.q_size + 2 * spec.kv_size
        self.hidden = owner.allocate(T * h * 2)
        self.normalized = owner.allocate(T * h * 2)
        self.residual = owner.allocate(T * h * 2)
        self.qkv = owner.allocate(T * qkv * 2)
        self.attention = owner.allocate(T * spec.q_size * 2)
        self.projection = owner.allocate(T * h * 2)
        self.selected_ids = owner.allocate(T * top_k * 4)
        self.routing_weights = owner.allocate(T * top_k * 4)
        self.router_logits = owner.allocate(T * spec.num_experts * 4)
        self.expert_gate = owner.allocate(T * top_k * intermediate * 2)
        self.expert_up = owner.allocate(T * top_k * intermediate * 2)
        self.expert_intermediate = owner.allocate(T * top_k * intermediate * 2)
        self.expert_down = owner.allocate(T * top_k * h * 2)
        total_lanes = T * top_k
        self.expert_start = owner.allocate((spec.num_experts + 1) * 8)
        self.active_experts = owner.allocate(spec.num_experts * 8)
        self.active_count = owner.allocate(8)
        self.sorted_lanes = owner.allocate(total_lanes * 8)
        self.sorted_experts = owner.allocate(total_lanes * 8)
        self.sorted_weights = owner.allocate(total_lanes * 4)
        self.token_ids = owner.allocate(T * 8)


@dataclass
class _BatchBuffers(_PrefillBuffers):
    """T-row runtime buffers including the all-row sampling tail."""

    logits: DeviceBuffer
    argmax_block_values: DeviceBuffer
    argmax_block_indices: DeviceBuffer
    argmax_index: DeviceBuffer
    argmax_value: DeviceBuffer

    def __init__(
        self,
        *,
        owner: _BufferOwner,
        spec: MapleModelSpec,
        top_k: int,
        intermediate: int,
        T: int,
    ) -> None:
        super().__init__(
            owner=owner,
            spec=spec,
            top_k=top_k,
            intermediate=intermediate,
            T=T,
        )
        self.logits = owner.allocate(T * spec.vocab_size * 4)
        argmax_blocks = (spec.vocab_size + 1_023) // 1_024
        self.argmax_block_values = owner.allocate(T * argmax_blocks * 4)
        self.argmax_block_indices = owner.allocate(T * argmax_blocks * 8)
        self.argmax_index = owner.allocate(T * 4)
        self.argmax_value = owner.allocate(T * 4)


@dataclass(frozen=True)
class MapleKVLayer:
    key_cache: DeviceBuffer
    value_cache: DeviceBuffer
    spans: KVLiveSpans


@dataclass(frozen=True)
class MapleStepResult:
    position: int
    token_id: int
    top_logit: float
    elapsed_ms: float


class _BufferOwner:
    def __init__(self, runtime: HipRuntime) -> None:
        self.runtime = runtime
        self.buffers: list[DeviceBuffer] = []
        self.closed = False

    def allocate(self, nbytes: int) -> DeviceBuffer:
        if self.closed:
            raise RuntimeError("Maple buffer owner is closed")
        buffer = malloc(int(nbytes), runtime=self.runtime)
        self.buffers.append(buffer)
        return buffer

    def put(self, array: np.ndarray) -> DeviceBuffer:
        host = np.ascontiguousarray(array)
        buffer = self.allocate(host.nbytes)
        copy_host_to_device(
            buffer,
            host_array_ptr(host),
            runtime=self.runtime,
        )
        return buffer

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        for buffer in reversed(self.buffers):
            free(buffer, runtime=self.runtime)
        self.buffers.clear()


class MapleSpanOwner:
    """One resettable token-granular span set shared by like-capacity layers."""

    def __init__(self, owner: _BufferOwner, capacity: int, device: Device) -> None:
        if capacity <= 0:
            raise ValueError("Maple span capacity must be positive")
        self.owner = owner
        self.capacity = int(capacity)
        self.device = device
        self._base_host = np.arange(self.capacity, dtype=np.int32)
        self._live_host = np.zeros(1, dtype=np.int64)
        self._token_host = np.full(self.capacity, -1, dtype=np.int64)
        self._evict_host = np.ones(self.capacity, dtype=np.bool_)
        self._row_host = np.full(1, -1, dtype=np.int64)
        self.base_offsets = owner.put(self._base_host)
        self.live_counts = owner.put(self._live_host)
        self.token_positions = owner.put(self._token_host)
        self.evict_mask = owner.put(self._evict_host)
        self.row_positions = owner.put(self._row_host)
        self.spans = KVLiveSpans.sliding_ring(
            base_offsets=Tensor.from_handle(
                self.base_offsets.ptr, (self.capacity,), DType.INT32, device
            ),
            live_counts=Tensor.from_handle(self.live_counts.ptr, (1,), DType.INT64, device),
            token_positions=Tensor.from_handle(
                self.token_positions.ptr, (self.capacity,), DType.INT64, device
            ),
            evict_mask=Tensor.from_handle(
                self.evict_mask.ptr, (self.capacity,), DType.BOOL, device
            ),
            row_positions=Tensor.from_handle(self.row_positions.ptr, (1,), DType.INT64, device),
            capacity=self.capacity,
        )

    def reset(self) -> None:
        self._live_host.fill(0)
        self._token_host.fill(-1)
        self._evict_host.fill(True)
        self._row_host.fill(-1)
        for buffer, host in (
            (self.live_counts, self._live_host),
            (self.token_positions, self._token_host),
            (self.evict_mask, self._evict_host),
            (self.row_positions, self._row_host),
        ):
            copy_host_to_device(
                buffer,
                host_array_ptr(host),
                runtime=self.owner.runtime,
            )


class MapleBatchSpanOwner:
    """Per-request ring spans for one batch attention capacity class."""

    def __init__(
        self,
        owner: _BufferOwner,
        *,
        batch_size: int,
        per_request_capacity: int,
        device: Device,
    ) -> None:
        if batch_size <= 0 or per_request_capacity <= 0:
            raise ValueError("Maple batch span dimensions must be positive")
        self.owner = owner
        self.runtime = owner.runtime
        self.batch_size = int(batch_size)
        self.per_request_capacity = int(per_request_capacity)
        self.capacity = self.batch_size * self.per_request_capacity
        self.device = device
        self.base_host = np.arange(self.capacity, dtype=np.int32)
        self.live_host = np.zeros(self.batch_size, dtype=np.int64)
        self.token_host = np.full(self.capacity, -1, dtype=np.int64)
        self.evict_host = np.ones(self.capacity, dtype=np.bool_)
        self.row_host = np.full(self.batch_size, -1, dtype=np.int64)
        self.row_base_host = (
            np.arange(self.batch_size, dtype=np.int64) * self.per_request_capacity
        )
        self.base_offsets = owner.put(self.base_host)
        self.live_counts = owner.put(self.live_host)
        self.token_positions = owner.put(self.token_host)
        self.evict_mask = owner.put(self.evict_host)
        self.row_positions = owner.put(self.row_host)
        self.row_base_offsets = owner.put(self.row_base_host)
        self.spans = KVLiveSpans(
            base_offsets=Tensor.from_handle(
                self.base_offsets.ptr, (self.capacity,), DType.INT32, self.device
            ),
            live_counts=Tensor.from_handle(
                self.live_counts.ptr, (self.batch_size,), DType.INT64, self.device
            ),
            max_live_count=self.capacity,
            token_positions=Tensor.from_handle(
                self.token_positions.ptr, (self.capacity,), DType.INT64, self.device
            ),
            evict_mask=Tensor.from_handle(
                self.evict_mask.ptr, (self.capacity,), DType.BOOL, self.device
            ),
            storage_dtype=DType.BF16,
            spans_mode="uniform",
            row_positions=Tensor.from_handle(
                self.row_positions.ptr, (self.batch_size,), DType.INT64, self.device
            ),
        )

    @staticmethod
    def _view(buffer: DeviceBuffer, offset: int, nbytes: int) -> DeviceBuffer:
        return DeviceBuffer(ptr=buffer.ptr + offset, nbytes=nbytes)

    def publish(self, positions: np.ndarray, active_mask: tuple[bool, ...]) -> None:
        if len(positions) != self.batch_size or len(active_mask) != self.batch_size:
            raise ValueError("Maple batch span publication shape mismatch")
        for request, active in enumerate(active_mask):
            if not active:
                self.live_host[request] = 0
                self.row_host[request] = -1
                continue
            position = int(positions[request])
            if position < 0:
                raise ValueError("Maple batch positions must be non-negative")
            local_slot = position % self.per_request_capacity
            slot = request * self.per_request_capacity + local_slot
            self.token_host[slot] = position
            self.evict_host[slot] = False
            self.live_host[request] = min(position + 1, self.per_request_capacity)
            self.row_host[request] = position
            copy_host_to_device(
                self._view(self.token_positions, slot * 8, 8),
                host_array_ptr(self.token_host[slot : slot + 1]),
                nbytes=8,
                runtime=self.runtime,
            )
            copy_host_to_device(
                self._view(self.evict_mask, slot, 1),
                host_array_ptr(self.evict_host[slot : slot + 1]),
                nbytes=1,
                runtime=self.runtime,
            )
        for buffer, host in (
            (self.live_counts, self.live_host),
            (self.row_positions, self.row_host),
        ):
            copy_host_to_device(buffer, host_array_ptr(host), runtime=self.runtime)

    def reset(self) -> None:
        self.live_host.fill(0)
        self.token_host.fill(-1)
        self.evict_host.fill(True)
        self.row_host.fill(-1)
        for buffer, host in (
            (self.live_counts, self.live_host),
            (self.token_positions, self.token_host),
            (self.evict_mask, self.evict_host),
            (self.row_positions, self.row_host),
        ):
            copy_host_to_device(buffer, host_array_ptr(host), runtime=self.runtime)

    def reset_request(self, request: int) -> None:
        if not 0 <= request < self.batch_size:
            raise ValueError(f"request {request} out of range")
        lo = request * self.per_request_capacity
        hi = lo + self.per_request_capacity
        self.token_host[lo:hi] = -1
        self.evict_host[lo:hi] = True
        self.live_host[request] = 0
        self.row_host[request] = -1
        copies = (
            (
                self._view(self.token_positions, lo * 8, (hi - lo) * 8),
                self.token_host[lo:hi],
            ),
            (
                self._view(self.evict_mask, lo, hi - lo),
                self.evict_host[lo:hi],
            ),
            (
                self._view(self.live_counts, request * 8, 8),
                self.live_host[request : request + 1],
            ),
            (
                self._view(self.row_positions, request * 8, 8),
                self.row_host[request : request + 1],
            ),
        )
        for buffer, host in copies:
            copy_host_to_device(
                buffer,
                host_array_ptr(host),
                nbytes=host.nbytes,
                runtime=self.runtime,
            )


class MapleRuntimeBuffers:
    def __init__(
        self,
        *,
        checkpoint: MapleCheckpoint,
        owner: _BufferOwner,
        max_context: int,
        device: Device,
    ) -> None:
        spec = checkpoint.spec
        h = spec.hidden_size
        qkv = spec.q_size + 2 * spec.kv_size
        top_k = spec.num_experts_per_tok
        intermediate = spec.moe_intermediate_size
        self.owner = owner
        self.hidden = owner.allocate(h * 2)
        self.normalized = owner.allocate(h * 2)
        self.residual = owner.allocate(h * 2)
        self.qkv = owner.allocate(qkv * 2)
        self.attention = owner.allocate(spec.q_size * 2)
        self.projection = owner.allocate(h * 2)
        self.selected_ids = owner.allocate(top_k * 4)
        self.routing_weights = owner.allocate(top_k * 4)
        self.router_logits = owner.allocate(spec.num_experts * 4)
        self.router_counter = owner.put(np.zeros(1, dtype=np.uint32))
        self.expert_gate = owner.allocate(top_k * intermediate * 2)
        self.expert_up = owner.allocate(top_k * intermediate * 2)
        self.expert_intermediate = owner.allocate(top_k * intermediate * 2)
        self.expert_down = owner.allocate(top_k * h * 2)
        self.logits = owner.allocate(spec.vocab_size * 4)
        argmax_blocks = (spec.vocab_size + 1_023) // 1_024
        self.argmax_block_values = owner.allocate(argmax_blocks * 4)
        self.argmax_block_indices = owner.allocate(argmax_blocks * 8)
        self.argmax_index = owner.allocate(8)
        self.argmax_value = owner.allocate(4)

        # Batched prefill scratch (T = PREFILL_CHUNK rows).
        T = PREFILL_CHUNK
        self.pf = _PrefillBuffers(
            owner=owner,
            spec=spec,
            top_k=top_k,
            intermediate=intermediate,
            T=T,
        )

        self.sliding_span_owner = MapleSpanOwner(
            owner,
            min(spec.sliding_window, max_context),
            device,
        )
        self.global_span_owner = MapleSpanOwner(owner, max_context, device)
        layers: list[MapleKVLayer] = []
        for layer, kind in enumerate(spec.layer_types):
            spans = (
                self.sliding_span_owner.spans
                if kind == "sliding_attention"
                else self.global_span_owner.spans
            )
            cache_bytes = spans.max_live_count * spec.kv_size * 2
            layers.append(
                MapleKVLayer(
                    key_cache=owner.allocate(cache_bytes),
                    value_cache=owner.allocate(cache_bytes),
                    spans=spans,
                )
            )
        self.layers = tuple(layers)

    def reset(self) -> None:
        self.owner.runtime.memset(
            self.router_counter.ptr, 0, self.router_counter.nbytes
        )
        self.sliding_span_owner.reset()
        self.global_span_owner.reset()


class MapleRunner:
    """Single-sequence resident Maple decode runner for exact greedy bring-up."""

    def __init__(
        self,
        *,
        checkpoint: MapleCheckpoint,
        weights: MapleDeviceWeights,
        buffers: MapleRuntimeBuffers,
        libraries: MapleRunnerLibraries,
        owner: _BufferOwner,
        backend: str,
        max_context: int,
        runtime: HipRuntime,
    ) -> None:
        self.checkpoint = checkpoint
        self.weights = weights
        self.buffers = buffers
        self.libraries = libraries
        self.owner = owner
        self.backend = backend
        self.max_context = int(max_context)
        self.runtime = runtime
        self.position = 0
        self.last_hidden_states: tuple[np.ndarray, ...] = ()
        self._graph: MapleGraphCache | None = None
        self.closed = False

    @classmethod
    def load(
        cls,
        checkpoint: MapleCheckpoint,
        *,
        backend: str = "auto",
        max_context: int = 4_096,
        runtime: HipRuntime | None = None,
    ) -> MapleRunner:
        backend = resolve_backend(backend)
        target_arch = hip_target_arch_for_backend(backend)
        load_backend_kernel_package(backend)
        parsed_context = int(max_context)
        if parsed_context <= 0 or parsed_context > checkpoint.spec.max_position_embeddings:
            raise ValueError(
                "Maple max_context must be positive and not exceed max_position_embeddings"
            )
        runtime = runtime or get_hip_runtime()
        with hip_target_arch_environment(target_arch):
            libraries = MapleRunnerLibraries(
                ternary=build_maple_ternary(load=True),
                attention=build_maple_attention(load=True),
                moe=build_maple_moe(load=True),
                norm=build_qwen35_rmsnorm(load=True),
                lm_head=build_lm_head(load=True),
                group_scatter=(
                    build_qwen35_moe_group_scatter(load=True)
                    if _maple_prefill_grouped_moe()
                    else None
                ),
            )
        weights: MapleDeviceWeights | None = None
        owner = _BufferOwner(runtime)
        try:
            weights = materialize_maple_weights(checkpoint, runtime=runtime)
            buffers = MapleRuntimeBuffers(
                checkpoint=checkpoint,
                owner=owner,
                max_context=parsed_context,
                device=Device("hip", 0),
            )
            return cls(
                checkpoint=checkpoint,
                weights=weights,
                buffers=buffers,
                libraries=libraries,
                owner=owner,
                backend=backend,
                max_context=parsed_context,
                runtime=runtime,
            )
        except Exception:
            owner.close()
            if weights is not None:
                weights.free(runtime=runtime)
            raise

    def _graph_cache(self) -> MapleGraphCache | None:
        if not _maple_graph_enabled():
            return None
        if self._graph is None:
            self._graph = MapleGraphCache(self.runtime, enabled=True)
        return self._graph

    def reset(self) -> None:
        self._require_open()
        self.runtime.device_synchronize()
        self.buffers.reset()
        self.position = 0
        self.last_hidden_states = ()

    def step(self, token_id: int, *, capture_hidden: bool = False) -> MapleStepResult:
        self._require_open()
        spec = self.checkpoint.spec
        token = int(token_id)
        if token < 0 or token >= spec.vocab_size:
            raise ValueError(f"Maple token_id must be in [0, {spec.vocab_size})")
        if self.position >= self.max_context:
            raise ValueError(f"Maple context capacity {self.max_context} exceeded")
        started = time.perf_counter()
        position = self.position
        captured: list[np.ndarray] = []
        b = self.buffers
        libs = self.libraries
        self._publish_span_position(position)

        maple_affine4_embed_bf16(
            self.weights.embeddings.weight.ptr,
            self.weights.embeddings.scales.ptr,
            self.weights.embeddings.biases.ptr,
            b.hidden.ptr,
            token,
            spec.hidden_size,
            library=libs.ternary,
            runtime=self.runtime,
        )
        if capture_hidden:
            captured.append(self._copy_bf16(b.hidden, spec.hidden_size))
        single_dispatch_router = _maple_router_single_dispatch()
        affine4_wave32_exact = _maple_affine4_wave32_exact()
        fuse_qkattn = _maple_fuse_qkattn()
        fuse_moe = _maple_fuse_moe()

        def _decode_body(stream: int) -> None:
            _decode_layers_and_tail(stream)

        def _decode_layers_and_tail(stream: int) -> None:
            for layer_id, (layer_weights, kv_layer) in enumerate(
                zip(self.weights.layers, b.layers)
            ):
                paro_rmsnorm_out_bf16(
                    b.hidden.ptr,
                    layer_weights.input_layernorm.ptr,
                    b.normalized.ptr,
                    1,
                    spec.hidden_size,
                    spec.rms_norm_eps,
                    library=libs.norm,
                    runtime=self.runtime, stream=stream,
                )
                maple_ternary_qkv_gemv_bf16(
                    b.normalized.ptr,
                    layer_weights.q_proj.weight.ptr,
                    layer_weights.q_proj.row_alpha.ptr,
                    layer_weights.k_proj.weight.ptr,
                    layer_weights.k_proj.row_alpha.ptr,
                    layer_weights.v_proj.weight.ptr,
                    layer_weights.v_proj.row_alpha.ptr,
                    b.qkv.ptr,
                    spec.hidden_size,
                    spec.q_size,
                    spec.kv_size,
                    library=libs.ternary,
                    runtime=self.runtime, stream=stream,
                )
                rope_dim = spec.rotary_dim if spec.uses_rope(layer_id) else 0
                if fuse_qkattn:
                    maple_attention_fused_qknorm_decode_bf16(
                        b.qkv.ptr,
                        layer_weights.q_norm.ptr,
                        layer_weights.k_norm.ptr,
                        kv_layer.key_cache.ptr,
                        kv_layer.value_cache.ptr,
                        b.attention.ptr,
                        kv_layer.spans,
                        q_heads=spec.num_attention_heads,
                        kv_heads=spec.num_key_value_heads,
                        head_dim=spec.head_dim,
                        rope_dim=rope_dim,
                        eps=spec.rms_norm_eps,
                        rope_theta=spec.rope_theta,
                        scale=spec.head_dim**-0.5,
                        library=libs.attention,
                        runtime=self.runtime, stream=stream,
                    )
                else:
                    maple_qknorm_rope_kv_write_bf16(
                        b.qkv.ptr,
                        layer_weights.q_norm.ptr,
                        layer_weights.k_norm.ptr,
                        kv_layer.key_cache.ptr,
                        kv_layer.value_cache.ptr,
                        kv_layer.spans,
                        q_heads=spec.num_attention_heads,
                        kv_heads=spec.num_key_value_heads,
                        head_dim=spec.head_dim,
                        rope_dim=rope_dim,
                        eps=spec.rms_norm_eps,
                        rope_theta=spec.rope_theta,
                        library=libs.attention,
                        runtime=self.runtime, stream=stream,
                    )
                    maple_attention_decode_bf16(
                        b.qkv.ptr,
                        kv_layer.key_cache.ptr,
                        kv_layer.value_cache.ptr,
                        b.attention.ptr,
                        kv_layer.spans,
                        q_heads=spec.num_attention_heads,
                        kv_heads=spec.num_key_value_heads,
                        head_dim=spec.head_dim,
                        scale=spec.head_dim**-0.5,
                        library=libs.attention,
                        runtime=self.runtime, stream=stream,
                    )
                maple_ternary_gemv_bf16(
                    b.attention.ptr,
                    layer_weights.o_proj.weight.ptr,
                    layer_weights.o_proj.row_alpha.ptr,
                    b.projection.ptr,
                    spec.q_size,
                    spec.hidden_size,
                    library=libs.ternary,
                    runtime=self.runtime, stream=stream,
                )
                paro_add_rmsnorm_out_bf16(
                    b.hidden.ptr,
                    b.projection.ptr,
                    layer_weights.post_attention_layernorm.ptr,
                    b.normalized.ptr,
                    b.residual.ptr,
                    1,
                    spec.hidden_size,
                    spec.rms_norm_eps,
                    library=libs.norm,
                    runtime=self.runtime, stream=stream,
                )
                if single_dispatch_router:
                    maple_router_topk_single_dispatch_bf16(
                        b.normalized.ptr,
                        layer_weights.router.ptr,
                        b.selected_ids.ptr,
                        b.routing_weights.ptr,
                        b.router_logits.ptr,
                        b.router_counter.ptr,
                        spec.hidden_size,
                        spec.num_experts,
                        spec.num_experts_per_tok,
                        library=libs.moe,
                        runtime=self.runtime,
                        stream=stream,
                    )
                else:
                    maple_router_topk_parallel_bf16(
                        b.normalized.ptr,
                        layer_weights.router.ptr,
                        b.selected_ids.ptr,
                        b.routing_weights.ptr,
                        b.router_logits.ptr,
                        spec.hidden_size,
                        spec.num_experts,
                        spec.num_experts_per_tok,
                        library=libs.moe,
                        runtime=self.runtime,
                        stream=stream,
                    )
                if fuse_moe:
                    maple_moe_dual_swiglu_bf16(
                        b.normalized.ptr,
                        layer_weights.expert_gate_proj.weight.ptr,
                        layer_weights.expert_gate_proj.row_alpha.ptr,
                        layer_weights.expert_up_proj.weight.ptr,
                        layer_weights.expert_up_proj.row_alpha.ptr,
                        b.selected_ids.ptr,
                        b.expert_intermediate.ptr,
                        spec.num_experts,
                        spec.num_experts_per_tok,
                        spec.hidden_size,
                        spec.moe_intermediate_size,
                        library=libs.ternary,
                        runtime=self.runtime, stream=stream,
                    )
                    maple_selected_ternary_gemv_bf16(
                        b.expert_intermediate.ptr,
                        layer_weights.expert_down_proj.weight.ptr,
                        layer_weights.expert_down_proj.row_alpha.ptr,
                        b.selected_ids.ptr,
                        b.expert_down.ptr,
                        spec.num_experts,
                        spec.num_experts_per_tok,
                        spec.moe_intermediate_size,
                        spec.hidden_size,
                        library=libs.ternary,
                        runtime=self.runtime, stream=stream,
                    )
                    maple_weighted_residual_bf16(
                        b.residual.ptr,
                        b.expert_down.ptr,
                        b.routing_weights.ptr,
                        b.hidden.ptr,
                        spec.num_experts_per_tok,
                        spec.hidden_size,
                        library=libs.moe,
                        runtime=self.runtime, stream=stream,
                    )
                else:
                    maple_selected_ternary_dual_gemv_bf16(
                        b.normalized.ptr,
                        layer_weights.expert_gate_proj.weight.ptr,
                        layer_weights.expert_gate_proj.row_alpha.ptr,
                        layer_weights.expert_up_proj.weight.ptr,
                        layer_weights.expert_up_proj.row_alpha.ptr,
                        b.selected_ids.ptr,
                        b.expert_gate.ptr,
                        b.expert_up.ptr,
                        spec.num_experts,
                        spec.num_experts_per_tok,
                        spec.hidden_size,
                        spec.moe_intermediate_size,
                        library=libs.ternary,
                        runtime=self.runtime, stream=stream,
                    )
                    maple_clamped_swiglu_bf16(
                        b.expert_gate.ptr,
                        b.expert_up.ptr,
                        b.expert_intermediate.ptr,
                        spec.num_experts_per_tok,
                        spec.moe_intermediate_size,
                        library=libs.moe,
                        runtime=self.runtime, stream=stream,
                    )
                    maple_selected_ternary_gemv_bf16(
                        b.expert_intermediate.ptr,
                        layer_weights.expert_down_proj.weight.ptr,
                        layer_weights.expert_down_proj.row_alpha.ptr,
                        b.selected_ids.ptr,
                        b.expert_down.ptr,
                        spec.num_experts,
                        spec.num_experts_per_tok,
                        spec.moe_intermediate_size,
                        spec.hidden_size,
                        library=libs.ternary,
                        runtime=self.runtime, stream=stream,
                    )
                    maple_weighted_residual_bf16(
                        b.residual.ptr,
                        b.expert_down.ptr,
                        b.routing_weights.ptr,
                        b.hidden.ptr,
                        spec.num_experts_per_tok,
                        spec.hidden_size,
                        library=libs.moe,
                        runtime=self.runtime, stream=stream,
                    )
                if capture_hidden:
                    captured.append(self._copy_bf16(b.hidden, spec.hidden_size))

            paro_rmsnorm_out_bf16(
                b.hidden.ptr,
                self.weights.final_norm.ptr,
                b.normalized.ptr,
                1,
                spec.hidden_size,
                spec.rms_norm_eps,
                library=libs.norm,
                runtime=self.runtime, stream=stream,
            )
            if capture_hidden:
                captured.append(self._copy_bf16(b.normalized, spec.hidden_size))
            affine4_head = (
                maple_affine4_gemv_wave32_exact_f32
                if affine4_wave32_exact
                else maple_affine4_gemv_f32
            )
            affine4_head(
                b.normalized.ptr,
                self.weights.lm_head.weight.ptr,
                self.weights.lm_head.scales.ptr,
                self.weights.lm_head.biases.ptr,
                b.logits.ptr,
                spec.hidden_size,
                spec.vocab_size,
                library=libs.ternary,
                runtime=self.runtime, stream=stream,
            )
            argmax_f32(
                b.logits.ptr,
                b.argmax_block_values.ptr,
                b.argmax_block_indices.ptr,
                b.argmax_index.ptr,
                b.argmax_value.ptr,
                spec.vocab_size,
                library=libs.lm_head,
                runtime=self.runtime, stream=stream,
            )

        if capture_hidden:
            _decode_layers_and_tail(0)
        else:
            graph = self._graph_cache()
            if graph is not None and graph.enabled:
                graph.run(
                    (),
                    eager=_decode_body,
                    argmax_index_ptr=b.argmax_index.ptr,
                    argmax_value_ptr=b.argmax_value.ptr,
                    mutable_inputs=((b.hidden.ptr, spec.hidden_size * 2),),
                    stream=0,
                )
            else:
                _decode_layers_and_tail(0)
        index_host = np.empty(1, dtype=np.int64)
        value_host = np.empty(1, dtype=np.float32)
        copy_device_to_host(
            host_array_ptr(index_host),
            b.argmax_index,
            runtime=self.runtime,
        )
        copy_device_to_host(
            host_array_ptr(value_host),
            b.argmax_value,
            runtime=self.runtime,
        )
        self.last_hidden_states = tuple(captured)
        self.position += 1
        return MapleStepResult(
            position=position,
            token_id=int(index_host[0]),
            top_logit=float(value_host[0]),
            elapsed_ms=(time.perf_counter() - started) * 1_000.0,
        )

    def prefill(self, token_ids: tuple[int, ...] | list[int]) -> MapleStepResult:
        tokens = tuple(int(token) for token in token_ids)
        if not tokens:
            raise ValueError("Maple prompt token IDs must not be empty")
        if self.position + len(tokens) > self.max_context:
            raise ValueError("Maple prompt exceeds runner context capacity")
        result: MapleStepResult | None = None
        for token in tokens:
            result = self.step(token)
        assert result is not None
        return result

    def prefill_native(
        self,
        token_ids: tuple[int, ...] | list[int],
        *,
        chunk_size: int = PREFILL_CHUNK,
    ) -> MapleStepResult:
        """Batched bulk prefill over all layers (P4), chunked into bounded rows.

        Uses the batched prefill kernel chain (embed, QKV GEMM, qknorm ring
        write, ring attention, o_proj GEMM, router, stable expert compaction,
        expert-major MoE, and weighted residual), then samples only the final
        prompt row with the proven c1 norm/head/argmax tail. Returns that row's
        next-token prediction and top logit while preserving the token-serial
        continuation contract. The row/route-gather MoE remains an explicit
        rollback path.
        """

        self._require_open()
        try:
            parsed_chunk = int(chunk_size)
        except (TypeError, ValueError) as exc:
            raise ValueError("Maple chunk_size must be an integer") from exc
        if (
            isinstance(chunk_size, bool)
            or parsed_chunk != chunk_size
            or not 1 <= parsed_chunk <= PREFILL_CHUNK
        ):
            raise ValueError(f"Maple chunk_size must be in [1, {PREFILL_CHUNK}]")
        tokens = tuple(int(token) for token in token_ids)
        if not tokens:
            raise ValueError("Maple prompt token IDs must not be empty")
        spec = self.checkpoint.spec
        invalid_token = next(
            (token for token in tokens if token < 0 or token >= spec.vocab_size),
            None,
        )
        if invalid_token is not None:
            raise ValueError(f"Maple token_id must be in [0, {spec.vocab_size})")
        end_position = self.position + len(tokens)
        if end_position > self.max_context:
            raise ValueError("Maple prompt exceeds runner context capacity")
        if end_position > spec.sliding_window:
            raise NotImplementedError(
                "Maple native prefill currently supports at most one sliding-window "
                f"capacity ({spec.sliding_window} positions); use serial prefill beyond it"
            )
        b = self.buffers
        pf = b.pf
        libs = self.libraries
        h = spec.hidden_size
        q_size = spec.q_size
        kv_size = spec.kv_size
        top_k = spec.num_experts_per_tok
        intermediate = spec.moe_intermediate_size
        grouped_moe = _maple_prefill_grouped_moe()
        gqa4_attention = (
            _maple_prefill_gqa4()
            and spec.num_attention_heads == spec.num_key_value_heads * 4
            and spec.head_dim == 128
        )
        n = len(tokens)
        started = time.perf_counter()
        start_position = self.position
        last_index = np.empty(1, dtype=np.int64)
        last_value = np.empty(1, dtype=np.float32)
        for c0 in range(0, n, parsed_chunk):
            ids = tokens[c0 : c0 + parsed_chunk]
            rows = len(ids)
            pos = start_position + c0
            for span_owner in (b.sliding_span_owner, b.global_span_owner):
                maple_kv_span_update_batched(
                    span_owner.spans,
                    start=pos,
                    rows=rows,
                    library=libs.attention,
                    runtime=self.runtime,
                )
            ids_arr = np.asarray(ids, dtype=np.int64)
            copy_host_to_device(
                pf.token_ids,
                host_array_ptr(ids_arr),
                nbytes=ids_arr.nbytes,
                runtime=self.runtime,
            )
            maple_affine4_embed_batched_bf16(
                self.weights.embeddings.weight.ptr,
                self.weights.embeddings.scales.ptr,
                self.weights.embeddings.biases.ptr,
                pf.token_ids.ptr,
                pf.hidden.ptr,
                rows,
                h,
                library=libs.ternary,
                runtime=self.runtime,
            )
            for layer_id, (layer_weights, kv_layer) in enumerate(
                zip(self.weights.layers, b.layers)
            ):
                paro_rmsnorm_out_bf16(
                    pf.hidden.ptr,
                    layer_weights.input_layernorm.ptr,
                    pf.normalized.ptr,
                    rows,
                    h,
                    spec.rms_norm_eps,
                    library=libs.norm,
                    runtime=self.runtime,
                )
                maple_ternary_qkv_gemm_bf16(
                    pf.normalized.ptr,
                    layer_weights.q_proj.weight.ptr,
                    layer_weights.q_proj.row_alpha.ptr,
                    layer_weights.k_proj.weight.ptr,
                    layer_weights.k_proj.row_alpha.ptr,
                    layer_weights.v_proj.weight.ptr,
                    layer_weights.v_proj.row_alpha.ptr,
                    pf.qkv.ptr,
                    rows,
                    h,
                    q_size,
                    kv_size,
                    library=libs.ternary,
                    runtime=self.runtime,
                )
                rope_dim = spec.rotary_dim if spec.uses_rope(layer_id) else 0
                maple_qknorm_rope_kv_write_batched_bf16(
                    pf.qkv.ptr,
                    layer_weights.q_norm.ptr,
                    layer_weights.k_norm.ptr,
                    kv_layer.key_cache.ptr,
                    kv_layer.value_cache.ptr,
                    kv_layer.spans,
                    q_heads=spec.num_attention_heads,
                    kv_heads=spec.num_key_value_heads,
                    head_dim=spec.head_dim,
                    rope_dim=rope_dim,
                    eps=spec.rms_norm_eps,
                    rope_theta=spec.rope_theta,
                    start=pos,
                    rows=rows,
                    library=libs.attention,
                    runtime=self.runtime,
                )
                attention_prefill = (
                    maple_attention_prefill_ring_gqa4_bf16
                    if gqa4_attention
                    else maple_attention_prefill_ring_bf16
                )
                attention_prefill(
                    pf.qkv.ptr,
                    kv_layer.key_cache.ptr,
                    kv_layer.value_cache.ptr,
                    pf.attention.ptr,
                    kv_layer.spans,
                    rows=rows,
                    q_heads=spec.num_attention_heads,
                    kv_heads=spec.num_key_value_heads,
                    head_dim=spec.head_dim,
                    scale=spec.head_dim**-0.5,
                    start=pos,
                    library=libs.attention,
                    runtime=self.runtime,
                )
                maple_ternary_gemm_bf16(
                    pf.attention.ptr,
                    layer_weights.o_proj.weight.ptr,
                    layer_weights.o_proj.row_alpha.ptr,
                    pf.projection.ptr,
                    rows,
                    q_size,
                    h,
                    library=libs.ternary,
                    runtime=self.runtime,
                )
                paro_add_rmsnorm_out_bf16(
                    pf.hidden.ptr,
                    pf.projection.ptr,
                    layer_weights.post_attention_layernorm.ptr,
                    pf.normalized.ptr,
                    pf.residual.ptr,
                    rows,
                    h,
                    spec.rms_norm_eps,
                    library=libs.norm,
                    runtime=self.runtime,
                )
                maple_router_topk_parallel_batched_bf16(
                    pf.normalized.ptr,
                    layer_weights.router.ptr,
                    pf.selected_ids.ptr,
                    pf.routing_weights.ptr,
                    pf.router_logits.ptr,
                    rows,
                    h,
                    spec.num_experts,
                    top_k,
                    library=libs.moe,
                    runtime=self.runtime,
                )
                if grouped_moe:
                    if libs.group_scatter is None:
                        raise RuntimeError(
                            "Maple grouped-MoE metadata library is unavailable"
                        )
                    qwen35_moe_group_compact_active_i32_parallel(
                        pf.selected_ids.ptr,
                        pf.routing_weights.ptr,
                        pf.expert_start.ptr,
                        pf.active_experts.ptr,
                        pf.active_count.ptr,
                        pf.sorted_lanes.ptr,
                        pf.sorted_experts.ptr,
                        pf.sorted_weights.ptr,
                        rows * top_k,
                        spec.num_experts,
                        library=libs.group_scatter,
                        runtime=self.runtime,
                    )
                    maple_selected_ternary_dual_grouped_bf16(
                        pf.normalized.ptr,
                        layer_weights.expert_gate_proj.weight.ptr,
                        layer_weights.expert_gate_proj.row_alpha.ptr,
                        layer_weights.expert_up_proj.weight.ptr,
                        layer_weights.expert_up_proj.row_alpha.ptr,
                        pf.expert_start.ptr,
                        pf.sorted_lanes.ptr,
                        pf.expert_gate.ptr,
                        pf.expert_up.ptr,
                        rows,
                        spec.num_experts,
                        top_k,
                        h,
                        intermediate,
                        library=libs.ternary,
                        runtime=self.runtime,
                    )
                else:
                    maple_selected_ternary_dual_gemv_batched_bf16(
                        pf.normalized.ptr,
                        layer_weights.expert_gate_proj.weight.ptr,
                        layer_weights.expert_gate_proj.row_alpha.ptr,
                        layer_weights.expert_up_proj.weight.ptr,
                        layer_weights.expert_up_proj.row_alpha.ptr,
                        pf.selected_ids.ptr,
                        pf.expert_gate.ptr,
                        pf.expert_up.ptr,
                        rows,
                        spec.num_experts,
                        top_k,
                        h,
                        intermediate,
                        library=libs.ternary,
                        runtime=self.runtime,
                    )
                maple_clamped_swiglu_bf16(
                    pf.expert_gate.ptr,
                    pf.expert_up.ptr,
                    pf.expert_intermediate.ptr,
                    rows * top_k,
                    intermediate,
                    library=libs.moe,
                    runtime=self.runtime,
                )
                if grouped_moe:
                    maple_selected_ternary_grouped_bf16(
                        pf.expert_intermediate.ptr,
                        layer_weights.expert_down_proj.weight.ptr,
                        layer_weights.expert_down_proj.row_alpha.ptr,
                        pf.expert_start.ptr,
                        pf.sorted_lanes.ptr,
                        pf.expert_down.ptr,
                        rows,
                        spec.num_experts,
                        top_k,
                        intermediate,
                        h,
                        library=libs.ternary,
                        runtime=self.runtime,
                    )
                else:
                    maple_selected_ternary_gemv_batched_bf16(
                        pf.expert_intermediate.ptr,
                        layer_weights.expert_down_proj.weight.ptr,
                        layer_weights.expert_down_proj.row_alpha.ptr,
                        pf.selected_ids.ptr,
                        pf.expert_down.ptr,
                        rows,
                        spec.num_experts,
                        top_k,
                        intermediate,
                        h,
                        library=libs.ternary,
                        runtime=self.runtime,
                    )
                maple_weighted_residual_batched_bf16(
                    pf.residual.ptr,
                    pf.expert_down.ptr,
                    pf.routing_weights.ptr,
                    pf.hidden.ptr,
                    rows,
                    top_k,
                    h,
                    library=libs.moe,
                    runtime=self.runtime,
                )
        final_hidden_ptr = pf.hidden.ptr + (rows - 1) * h * 2
        paro_rmsnorm_out_bf16(
            final_hidden_ptr,
            self.weights.final_norm.ptr,
            b.normalized.ptr,
            1,
            h,
            spec.rms_norm_eps,
            library=libs.norm,
            runtime=self.runtime,
        )
        affine4_head = (
            maple_affine4_gemv_wave32_exact_f32
            if _maple_affine4_wave32_exact()
            else maple_affine4_gemv_f32
        )
        affine4_head(
            b.normalized.ptr,
            self.weights.lm_head.weight.ptr,
            self.weights.lm_head.scales.ptr,
            self.weights.lm_head.biases.ptr,
            b.logits.ptr,
            h,
            spec.vocab_size,
            library=libs.ternary,
            runtime=self.runtime,
        )
        argmax_f32(
            b.logits.ptr,
            b.argmax_block_values.ptr,
            b.argmax_block_indices.ptr,
            b.argmax_index.ptr,
            b.argmax_value.ptr,
            spec.vocab_size,
            library=libs.lm_head,
            runtime=self.runtime,
        )
        copy_device_to_host(
            host_array_ptr(last_index),
            b.argmax_index,
            runtime=self.runtime,
        )
        copy_device_to_host(
            host_array_ptr(last_value),
            b.argmax_value,
            runtime=self.runtime,
        )
        self.position = start_position + n
        return MapleStepResult(
            position=self.position - 1,
            token_id=int(last_index[0]),
            top_logit=float(last_value[0]),
            elapsed_ms=(time.perf_counter() - started) * 1_000.0,
        )

    def _publish_span_position(self, position: int) -> None:
        for span_owner in (
            self.buffers.sliding_span_owner,
            self.buffers.global_span_owner,
        ):
            maple_kv_span_update(
                span_owner.spans,
                position=position,
                library=self.libraries.attention,
                runtime=self.runtime,
            )

    def _copy_bf16(self, buffer: DeviceBuffer, elements: int) -> np.ndarray:
        bits = np.empty(int(elements), dtype=np.uint16)
        copy_device_to_host(
            host_array_ptr(bits),
            buffer,
            nbytes=bits.nbytes,
            runtime=self.runtime,
        )
        return (bits.astype(np.uint32) << np.uint32(16)).view(np.float32)

    def copy_logits(self) -> np.ndarray:
        self._require_open()
        logits = np.empty(self.checkpoint.spec.vocab_size, dtype=np.float32)
        copy_device_to_host(
            host_array_ptr(logits),
            self.buffers.logits,
            runtime=self.runtime,
        )
        return logits

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self._graph is not None:
            self._graph.close()
            self._graph = None
        self.runtime.device_synchronize()
        self.owner.close()
        self.weights.free(runtime=self.runtime)

    def _require_open(self) -> None:
        if self.closed:
            raise RuntimeError("Maple runner is closed")


class MapleBatchRunner:
    """Multi-request (c>1) Maple decode runner for M6 batch decode.

    Each of the ``batch_size`` requests owns a disjoint position range
    ``[r*per_cap, (r+1)*per_cap)`` in a shared identity KV arena, so the
    batched decode kernels (maple_qknorm_rope_kv_write_batched_decode_bf16,
    maple_attention_decode_batched_bf16) operate on independent per-row spans
    via per-row live_counts/row_positions. Bit-exact with running c serial c1
    decode steps.
    """

    def __init__(
        self,
        *,
        checkpoint: MapleCheckpoint,
        weights: MapleDeviceWeights,
        libraries: MapleRunnerLibraries,
        owner: _BufferOwner,
        backend: str,
        batch_size: int,
        per_capacity: int,
        runtime: HipRuntime,
    ) -> None:
        self.checkpoint = checkpoint
        self.weights = weights
        self.libraries = libraries
        self.owner = owner
        self.backend = backend
        self.runtime = runtime
        self.batch_size = int(batch_size)
        self.per_capacity = int(per_capacity)
        spec = checkpoint.spec
        if self.batch_size <= 0 or self.per_capacity <= 0:
            raise ValueError("batch_size and per_capacity must be positive")
        self.capacity = self.batch_size * self.per_capacity
        self.device = Device("hip", 0)
        self._requests = np.zeros(self.batch_size, dtype=np.int64)
        self.closed = False

        kv_size = spec.kv_size
        top_k = spec.num_experts_per_tok
        intermediate = spec.moe_intermediate_size
        T = self.batch_size
        self.pf = _BatchBuffers(
            owner=owner, spec=spec, top_k=top_k, intermediate=intermediate, T=T
        )

        # SWA and global layers retain independent per-request ring capacities.
        self.sliding_span_owner = MapleBatchSpanOwner(
            owner,
            batch_size=self.batch_size,
            per_request_capacity=min(self.per_capacity, spec.sliding_window),
            device=self.device,
        )
        self.global_span_owner = MapleBatchSpanOwner(
            owner,
            batch_size=self.batch_size,
            per_request_capacity=self.per_capacity,
            device=self.device,
        )

        layers: list[MapleKVLayer] = []
        for kind in spec.layer_types:
            span_owner = (
                self.sliding_span_owner
                if kind == "sliding_attention"
                else self.global_span_owner
            )
            cache_bytes = span_owner.capacity * kv_size * 2
            layers.append(
                MapleKVLayer(
                    key_cache=owner.allocate(cache_bytes),
                    value_cache=owner.allocate(cache_bytes),
                    spans=span_owner.spans,
                )
            )
        self.layers = tuple(layers)

    @classmethod
    def load(
        cls,
        checkpoint: MapleCheckpoint,
        *,
        backend: str = "auto",
        batch_size: int = 4,
        per_capacity: int = 64,
        runtime: HipRuntime | None = None,
    ) -> MapleBatchRunner:
        backend = resolve_backend(backend)
        target_arch = hip_target_arch_for_backend(backend)
        load_backend_kernel_package(backend)
        runtime = runtime or get_hip_runtime()
        with hip_target_arch_environment(target_arch):
            libraries = MapleRunnerLibraries(
                ternary=build_maple_ternary(load=True),
                attention=build_maple_attention(load=True),
                moe=build_maple_moe(load=True),
                norm=build_qwen35_rmsnorm(load=True),
                lm_head=build_lm_head(load=True),
            )
        weights: MapleDeviceWeights | None = None
        owner = _BufferOwner(runtime)
        try:
            weights = materialize_maple_weights(checkpoint, runtime=runtime)
            return cls(
                checkpoint=checkpoint,
                weights=weights,
                libraries=libraries,
                owner=owner,
                backend=backend,
                batch_size=int(batch_size),
                per_capacity=int(per_capacity),
                runtime=runtime,
            )
        except Exception:
            owner.close()
            if weights is not None:
                weights.free(runtime=runtime)
            raise

    def batch_step(
        self,
        token_ids: list[int] | tuple[int, ...],
        *,
        active_mask: list[bool] | tuple[bool, ...] | None = None,
    ) -> list[int]:
        """Decode one token per active request; inactive rows return ``-1``."""
        self._require_open()
        ids = [int(t) for t in token_ids]
        if len(ids) != self.batch_size:
            raise ValueError(
                f"batch_step expects {self.batch_size} token ids, got {len(ids)}"
            )
        active = (
            tuple(True for _ in range(self.batch_size))
            if active_mask is None
            else tuple(bool(value) for value in active_mask)
        )
        if len(active) != self.batch_size:
            raise ValueError(
                f"batch_step expects {self.batch_size} active-mask values, got {len(active)}"
            )
        if not any(active):
            raise ValueError("batch_step requires at least one active request")
        spec = self.checkpoint.spec
        invalid_token = next(
            (token for token in ids if token < 0 or token >= spec.vocab_size),
            None,
        )
        if invalid_token is not None:
            raise ValueError(f"Maple token_id must be in [0, {spec.vocab_size})")
        b = self.pf
        libs = self.libraries
        h = spec.hidden_size
        q_size = spec.q_size
        kv_size = spec.kv_size
        top_k = spec.num_experts_per_tok
        intermediate = spec.moe_intermediate_size
        rows = self.batch_size
        affine4_head = _maple_batch_affine4_head(rows)

        # Publish each active request's current position into both capacity classes.
        for request, is_active in enumerate(active):
            if is_active and self._requests[request] >= self.per_capacity:
                raise ValueError(
                    f"request {request} exceeds per-request capacity {self.per_capacity}"
                )
        for span_owner in (self.sliding_span_owner, self.global_span_owner):
            span_owner.publish(self._requests, active)

        ids_arr = np.asarray(ids, dtype=np.int64)
        copy_host_to_device(
            b.token_ids, host_array_ptr(ids_arr), runtime=self.runtime
        )
        maple_affine4_embed_batched_bf16(
            self.weights.embeddings.weight.ptr,
            self.weights.embeddings.scales.ptr,
            self.weights.embeddings.biases.ptr,
            b.token_ids.ptr,
            b.hidden.ptr,
            rows,
            h,
            library=libs.ternary,
            runtime=self.runtime,
        )
        for layer_id, (layer_weights, kv_layer, kind) in enumerate(
            zip(self.weights.layers, self.layers, spec.layer_types)
        ):
            span_owner = (
                self.sliding_span_owner
                if kind == "sliding_attention"
                else self.global_span_owner
            )
            paro_rmsnorm_out_bf16(
                b.hidden.ptr,
                layer_weights.input_layernorm.ptr,
                b.normalized.ptr,
                rows,
                h,
                spec.rms_norm_eps,
                library=libs.norm,
                runtime=self.runtime,
            )
            maple_ternary_qkv_gemm_bf16(
                b.normalized.ptr,
                layer_weights.q_proj.weight.ptr,
                layer_weights.q_proj.row_alpha.ptr,
                layer_weights.k_proj.weight.ptr,
                layer_weights.k_proj.row_alpha.ptr,
                layer_weights.v_proj.weight.ptr,
                layer_weights.v_proj.row_alpha.ptr,
                b.qkv.ptr,
                rows,
                h,
                q_size,
                kv_size,
                library=libs.ternary,
                runtime=self.runtime,
            )
            rope_dim = spec.rotary_dim if spec.uses_rope(layer_id) else 0
            maple_qknorm_rope_kv_write_batched_decode_bf16(
                b.qkv.ptr,
                layer_weights.q_norm.ptr,
                layer_weights.k_norm.ptr,
                kv_layer.key_cache.ptr,
                kv_layer.value_cache.ptr,
                kv_layer.spans,
                row_base_offsets=span_owner.row_base_offsets.ptr,
                rows=rows,
                q_heads=spec.num_attention_heads,
                kv_heads=spec.num_key_value_heads,
                head_dim=spec.head_dim,
                rope_dim=rope_dim,
                eps=spec.rms_norm_eps,
                rope_theta=spec.rope_theta,
                library=libs.attention,
                runtime=self.runtime,
            )
            maple_attention_decode_batched_bf16(
                b.qkv.ptr,
                kv_layer.key_cache.ptr,
                kv_layer.value_cache.ptr,
                b.attention.ptr,
                kv_layer.spans,
                row_base_offsets=span_owner.row_base_offsets.ptr,
                rows=rows,
                q_heads=spec.num_attention_heads,
                kv_heads=spec.num_key_value_heads,
                head_dim=spec.head_dim,
                scale=spec.head_dim**-0.5,
                library=libs.attention,
                runtime=self.runtime,
            )
            maple_ternary_gemm_bf16(
                b.attention.ptr,
                layer_weights.o_proj.weight.ptr,
                layer_weights.o_proj.row_alpha.ptr,
                b.projection.ptr,
                rows,
                q_size,
                h,
                library=libs.ternary,
                runtime=self.runtime,
            )
            paro_add_rmsnorm_out_bf16(
                b.hidden.ptr,
                b.projection.ptr,
                layer_weights.post_attention_layernorm.ptr,
                b.normalized.ptr,
                b.residual.ptr,
                rows,
                h,
                spec.rms_norm_eps,
                library=libs.norm,
                runtime=self.runtime,
            )
            maple_router_topk_parallel_batched_bf16(
                b.normalized.ptr,
                layer_weights.router.ptr,
                b.selected_ids.ptr,
                b.routing_weights.ptr,
                b.router_logits.ptr,
                rows,
                h,
                spec.num_experts,
                top_k,
                library=libs.moe,
                runtime=self.runtime,
            )
            maple_selected_ternary_dual_gemv_batched_bf16(
                b.normalized.ptr,
                layer_weights.expert_gate_proj.weight.ptr,
                layer_weights.expert_gate_proj.row_alpha.ptr,
                layer_weights.expert_up_proj.weight.ptr,
                layer_weights.expert_up_proj.row_alpha.ptr,
                b.selected_ids.ptr,
                b.expert_gate.ptr,
                b.expert_up.ptr,
                rows,
                spec.num_experts,
                top_k,
                h,
                intermediate,
                library=libs.ternary,
                runtime=self.runtime,
            )
            maple_clamped_swiglu_bf16(
                b.expert_gate.ptr,
                b.expert_up.ptr,
                b.expert_intermediate.ptr,
                rows * top_k,
                intermediate,
                library=libs.moe,
                runtime=self.runtime,
            )
            maple_selected_ternary_gemv_batched_bf16(
                b.expert_intermediate.ptr,
                layer_weights.expert_down_proj.weight.ptr,
                layer_weights.expert_down_proj.row_alpha.ptr,
                b.selected_ids.ptr,
                b.expert_down.ptr,
                rows,
                spec.num_experts,
                top_k,
                intermediate,
                h,
                library=libs.ternary,
                runtime=self.runtime,
            )
            maple_weighted_residual_batched_bf16(
                b.residual.ptr,
                b.expert_down.ptr,
                b.routing_weights.ptr,
                b.hidden.ptr,
                rows,
                top_k,
                h,
                library=libs.moe,
                runtime=self.runtime,
            )
        paro_rmsnorm_out_bf16(
            b.hidden.ptr,
            self.weights.final_norm.ptr,
            b.normalized.ptr,
            rows,
            h,
            spec.rms_norm_eps,
            library=libs.norm,
            runtime=self.runtime,
        )
        affine4_head(
            b.normalized.ptr,
            self.weights.lm_head.weight.ptr,
            self.weights.lm_head.scales.ptr,
            self.weights.lm_head.biases.ptr,
            b.logits.ptr,
            rows,
            h,
            spec.vocab_size,
            library=libs.ternary,
            runtime=self.runtime,
        )
        argmax_f32_rows_i32(
            b.logits.ptr,
            b.argmax_block_values.ptr,
            b.argmax_block_indices.ptr,
            b.argmax_index.ptr,
            None,
            rows,
            spec.vocab_size,
            library=libs.lm_head,
            runtime=self.runtime,
        )
        out = np.empty(rows, dtype=np.int32)
        copy_device_to_host(
            host_array_ptr(out),
            b.argmax_index,
            nbytes=out.nbytes,
            runtime=self.runtime,
        )
        for request, is_active in enumerate(active):
            if is_active:
                self._requests[request] += 1
        return [int(token) if active[row] else -1 for row, token in enumerate(out)]

    def reset(self) -> None:
        self._require_open()
        self.runtime.device_synchronize()
        self._requests.fill(0)
        self.sliding_span_owner.reset()
        self.global_span_owner.reset()

    def reset_request(self, request: int) -> None:
        """Reclaim one request slot and clear both of its KV span regions."""
        self._require_open()
        if not 0 <= request < self.batch_size:
            raise ValueError(f"request {request} out of range")
        self._requests[request] = 0
        self.sliding_span_owner.reset_request(request)
        self.global_span_owner.reset_request(request)
        self.runtime.device_synchronize()

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.runtime.device_synchronize()
        self.owner.close()
        self.weights.free(runtime=self.runtime)

    def _require_open(self) -> None:
        if self.closed:
            raise RuntimeError("Maple batch runner is closed")


__all__ = [
    "MapleBatchRunner",
    "MapleBatchSpanOwner",
    "MapleKVLayer",
    "MapleRunner",
    "MapleRunnerLibraries",
    "MapleRuntimeBuffers",
    "MapleSpanOwner",
    "MapleStepResult",
]
