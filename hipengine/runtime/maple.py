"""Resident torch-free Maple ternary decode runner."""

from __future__ import annotations

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
    maple_attention_decode_bf16,
    maple_kv_span_update,
    maple_qknorm_rope_kv_write_bf16,
)
from hipengine.kernels.hip_gfx1100.linear.lm_head import argmax_f32, build_lm_head
from hipengine.kernels.hip_gfx1100.moe.maple_moe import (
    build_maple_moe,
    maple_clamped_swiglu_bf16,
    maple_router_topk_bf16,
    maple_weighted_residual_bf16,
)
from hipengine.kernels.hip_gfx1100.norm.rmsnorm import (
    build_qwen35_rmsnorm,
    paro_add_rmsnorm_out_bf16,
    paro_rmsnorm_out_bf16,
)
from hipengine.kernels.hip_gfx1100.quant.maple_ternary import (
    build_maple_ternary,
    maple_affine4_embed_bf16,
    maple_affine4_gemv_f32,
    maple_selected_ternary_dual_gemv_bf16,
    maple_selected_ternary_gemv_bf16,
    maple_ternary_gemv_bf16,
    maple_ternary_qkv_gemv_bf16,
)
from hipengine.kvcache import KVLiveSpans
from hipengine.loading.maple import (
    MapleCheckpoint,
    MapleDeviceWeights,
    materialize_maple_weights,
)


@dataclass(frozen=True)
class MapleRunnerLibraries:
    ternary: object
    attention: object
    moe: object
    norm: object
    lm_head: object


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

    def reset(self) -> None:
        self._require_open()
        self.runtime.device_synchronize()
        self.buffers.reset()
        self.position = 0

    def step(self, token_id: int) -> MapleStepResult:
        self._require_open()
        spec = self.checkpoint.spec
        token = int(token_id)
        if token < 0 or token >= spec.vocab_size:
            raise ValueError(f"Maple token_id must be in [0, {spec.vocab_size})")
        if self.position >= self.max_context:
            raise ValueError(f"Maple context capacity {self.max_context} exceeded")
        started = time.perf_counter()
        position = self.position
        b = self.buffers
        libs = self.libraries
        maple_kv_span_update(
            b.sliding_span_owner.spans,
            position=position,
            library=libs.attention,
            runtime=self.runtime,
        )
        if b.global_span_owner.capacity != b.sliding_span_owner.capacity:
            maple_kv_span_update(
                b.global_span_owner.spans,
                position=position,
                library=libs.attention,
                runtime=self.runtime,
            )

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
                runtime=self.runtime,
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
                runtime=self.runtime,
            )
            rope_dim = spec.rotary_dim if spec.uses_rope(layer_id) else 0
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
                runtime=self.runtime,
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
                runtime=self.runtime,
            )
            maple_ternary_gemv_bf16(
                b.attention.ptr,
                layer_weights.o_proj.weight.ptr,
                layer_weights.o_proj.row_alpha.ptr,
                b.projection.ptr,
                spec.q_size,
                spec.hidden_size,
                library=libs.ternary,
                runtime=self.runtime,
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
                runtime=self.runtime,
            )
            maple_router_topk_bf16(
                b.normalized.ptr,
                layer_weights.router.ptr,
                b.selected_ids.ptr,
                b.routing_weights.ptr,
                spec.hidden_size,
                spec.num_experts,
                spec.num_experts_per_tok,
                library=libs.moe,
                runtime=self.runtime,
            )
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
                runtime=self.runtime,
            )
            maple_clamped_swiglu_bf16(
                b.expert_gate.ptr,
                b.expert_up.ptr,
                b.expert_intermediate.ptr,
                spec.num_experts_per_tok,
                spec.moe_intermediate_size,
                library=libs.moe,
                runtime=self.runtime,
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
                runtime=self.runtime,
            )
            maple_weighted_residual_bf16(
                b.residual.ptr,
                b.expert_down.ptr,
                b.routing_weights.ptr,
                b.hidden.ptr,
                spec.num_experts_per_tok,
                spec.hidden_size,
                library=libs.moe,
                runtime=self.runtime,
            )

        paro_rmsnorm_out_bf16(
            b.hidden.ptr,
            self.weights.final_norm.ptr,
            b.normalized.ptr,
            1,
            spec.hidden_size,
            spec.rms_norm_eps,
            library=libs.norm,
            runtime=self.runtime,
        )
        maple_affine4_gemv_f32(
            b.normalized.ptr,
            self.weights.lm_head.weight.ptr,
            self.weights.lm_head.scales.ptr,
            self.weights.lm_head.biases.ptr,
            b.logits.ptr,
            spec.hidden_size,
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
        self.runtime.device_synchronize()
        self.owner.close()
        self.weights.free(runtime=self.runtime)

    def _require_open(self) -> None:
        if self.closed:
            raise RuntimeError("Maple runner is closed")


__all__ = [
    "MapleKVLayer",
    "MapleRunner",
    "MapleRunnerLibraries",
    "MapleRuntimeBuffers",
    "MapleSpanOwner",
    "MapleStepResult",
]
