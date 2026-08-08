"""CUDA sm_120a resident c1 runner for Maple's packed checkpoint."""

from __future__ import annotations

import time

import numpy as np

from hipengine.core.cuda import CudaRuntime, get_cuda_runtime
from hipengine.core.device import Device
from hipengine.core.memory import copy_device_to_host, host_array_ptr
from hipengine.kernels.backends import (
    cuda_target_arch_for_backend,
    load_backend_kernel_package,
    resolve_backend,
)
from hipengine.kernels.cuda_sm120a.attention.maple_attention import (
    build_maple_attention,
    maple_attention_decode_bf16,
    maple_attention_fused_qknorm_decode_bf16,
    maple_kv_span_update,
    maple_qknorm_rope_kv_write_bf16,
)
from hipengine.kernels.cuda_sm120a.linear.maple_lm_head import (
    argmax_f32,
    build_lm_head,
)
from hipengine.kernels.cuda_sm120a.moe.maple_moe import (
    build_maple_moe,
    maple_clamped_swiglu_bf16,
    maple_router_topk_single_dispatch_bf16,
    maple_weighted_residual_bf16,
)
from hipengine.kernels.cuda_sm120a.norm.maple_rmsnorm import (
    build_qwen35_rmsnorm,
    paro_add_rmsnorm_out_bf16,
    paro_rmsnorm_out_bf16,
)
from hipengine.kernels.cuda_sm120a.quant.maple_ternary import (
    build_maple_ternary,
    maple_affine4_embed_bf16,
    maple_affine4_gemv_wave32_exact_f32,
    maple_moe_dual_swiglu_bf16,
    maple_selected_ternary_dual_gemv_bf16,
    maple_selected_ternary_gemv_bf16,
    maple_ternary_gemv_bf16,
    maple_ternary_qkv_gemv_bf16,
)
from hipengine.loading.maple import (
    MapleCheckpoint,
    MapleDeviceWeights,
    materialize_maple_weights,
)
from hipengine.runtime.maple import (
    PREFILL_CHUNK,
    MapleRunner,
    MapleRunnerLibraries,
    MapleRuntimeBuffers,
    MapleStepResult,
    _BufferOwner,
    _maple_fuse_moe,
    _maple_fuse_qkattn,
)


class MapleCudaRunner(MapleRunner):
    """Correctness-first CUDA c1 runner using the peer sm_120a kernels.

    Decode is fully device resident. Initial prompt admission deliberately uses
    the token-serial c1 path until the CUDA grouped-prefill metadata family is
    independently ported and gated.
    """

    @classmethod
    def load(
        cls,
        checkpoint: MapleCheckpoint,
        *,
        backend: str = "cuda_sm120a",
        max_context: int = 4_096,
        runtime: CudaRuntime | None = None,
    ) -> MapleCudaRunner:
        backend = resolve_backend(backend)
        cuda_target_arch_for_backend(backend)
        load_backend_kernel_package(backend)
        parsed_context = int(max_context)
        if (
            parsed_context <= 0
            or parsed_context > checkpoint.spec.max_position_embeddings
        ):
            raise ValueError(
                "Maple max_context must be positive and not exceed "
                "max_position_embeddings"
            )
        runtime = runtime or get_cuda_runtime()
        runtime.set_device(0)
        libraries = MapleRunnerLibraries(
            ternary=build_maple_ternary(load=True),
            attention=build_maple_attention(load=True),
            moe=build_maple_moe(load=True),
            norm=build_qwen35_rmsnorm(load=True),
            lm_head=build_lm_head(load=True),
            group_scatter=None,
        )
        weights: MapleDeviceWeights | None = None
        owner = _BufferOwner(runtime)
        try:
            device = Device("cuda", 0)
            weights = materialize_maple_weights(
                checkpoint,
                device=device,
                runtime=runtime,
            )
            buffers = MapleRuntimeBuffers(
                checkpoint=checkpoint,
                owner=owner,
                max_context=parsed_context,
                device=device,
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
            maple_affine4_gemv_wave32_exact_f32(
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

    def prefill_native(
        self,
        token_ids: tuple[int, ...] | list[int],
        *,
        chunk_size: int = PREFILL_CHUNK,
    ) -> MapleStepResult:
        """Correctness-first serial prompt admission through CUDA c1 kernels."""

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
        return self.prefill(token_ids)

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


__all__ = ["MapleCudaRunner"]
