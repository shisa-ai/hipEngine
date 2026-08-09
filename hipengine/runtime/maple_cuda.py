"""CUDA sm_120a resident c1 runner for Maple's packed checkpoint."""

from __future__ import annotations

import time

import numpy as np

from hipengine.core.cuda import CudaRuntime, get_cuda_runtime
from hipengine.core.device import Device
from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    host_array_ptr,
)
from hipengine.kernels.backends import (
    cuda_target_arch_for_backend,
    load_backend_kernel_package,
    resolve_backend,
)
from hipengine.kernels.cuda_sm120a.attention.maple_attention import (
    build_maple_attention,
    maple_attention_decode_wave32_exact_bf16,
    maple_attention_fused_qknorm_decode_bf16,
    maple_attention_prefill_ring_bf16,
    maple_attention_prefill_ring_gqa4_bf16,
    maple_kv_span_update,
    maple_kv_span_update_batched,
    maple_qknorm_rope_kv_write_batched_bf16,
    maple_qknorm_rope_kv_write_bf16,
)
from hipengine.kernels.cuda_sm120a.linear.maple_lm_head import (
    argmax_f32,
    build_lm_head,
)
from hipengine.kernels.cuda_sm120a.moe.group_scatter import (
    build_qwen35_moe_group_scatter,
    qwen35_moe_group_compact_active_i32_parallel,
)
from hipengine.kernels.cuda_sm120a.moe.maple_moe import (
    build_maple_moe,
    maple_clamped_swiglu_bf16,
    maple_router_topk_parallel_batched_bf16,
    maple_router_topk_single_dispatch_bf16,
    maple_weighted_residual_batched_bf16,
    maple_weighted_residual_bf16,
)
from hipengine.kernels.cuda_sm120a.norm.maple_rmsnorm import (
    build_qwen35_rmsnorm,
    paro_add_rmsnorm_out_bf16,
    paro_rmsnorm_out_bf16,
)
from hipengine.kernels.cuda_sm120a.quant.maple_ternary import (
    build_maple_ternary,
    maple_affine4_embed_batched_bf16,
    maple_affine4_embed_bf16,
    maple_affine4_gemv_wave32_exact_f32,
    maple_moe_dual_swiglu_bf16,
    maple_selected_ternary_dual_gemv_bf16,
    maple_selected_ternary_dual_grouped_bf16,
    maple_selected_ternary_gemv_bf16,
    maple_selected_ternary_grouped_bf16,
    maple_ternary_gemm_bf16,
    maple_ternary_gemv_bf16,
    maple_ternary_qkv_gemm_bf16,
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
    _maple_prefill_swa_segments,
)


class MapleCudaRunner(MapleRunner):
    """Correctness-first CUDA c1 runner using the peer sm_120a kernels.

    Decode and grouped native c1 prefill are fully device resident. The serial
    c1 chain remains the independent state/logit oracle; SWA-wrap attention is
    segmented without falling back to HIP builders.
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
            group_scatter=build_qwen35_moe_group_scatter(load=True),
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
                    maple_attention_decode_wave32_exact_bf16(
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
        """Run the complete grouped CUDA bulk-prefill chain over bounded rows."""

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

        b = self.buffers
        pf = b.pf
        libs = self.libraries
        if libs.group_scatter is None:
            raise RuntimeError("Maple CUDA grouped-MoE metadata library is unavailable")
        h = spec.hidden_size
        q_size = spec.q_size
        kv_size = spec.kv_size
        top_k = spec.num_experts_per_tok
        intermediate = spec.moe_intermediate_size
        gqa4_attention = (
            spec.num_attention_heads == spec.num_key_value_heads * 4
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
            swa_segments = _maple_prefill_swa_segments(
                start=pos,
                rows=rows,
                capacity=b.sliding_span_owner.capacity,
            )
            replay_swa_per_layer = len(swa_segments) > 1
            maple_kv_span_update_batched(
                b.global_span_owner.spans,
                start=pos,
                rows=rows,
                library=libs.attention,
                runtime=self.runtime,
            )
            if not replay_swa_per_layer:
                maple_kv_span_update_batched(
                    b.sliding_span_owner.spans,
                    start=pos,
                    rows=rows,
                    library=libs.attention,
                    runtime=self.runtime,
                )
            ids_array = np.asarray(ids, dtype=np.int64)
            copy_host_to_device(
                pf.token_ids,
                host_array_ptr(ids_array),
                nbytes=ids_array.nbytes,
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

            for layer_id, (layer_weights, kv_layer, layer_kind) in enumerate(
                zip(self.weights.layers, b.layers, spec.layer_types)
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
                attention_prefill = (
                    maple_attention_prefill_ring_gqa4_bf16
                    if gqa4_attention
                    else maple_attention_prefill_ring_bf16
                )
                layer_segments = (
                    swa_segments
                    if layer_kind == "sliding_attention" and replay_swa_per_layer
                    else ((0, rows),)
                )
                if layer_kind == "sliding_attention" and replay_swa_per_layer:
                    prefix_rows = min(pos, b.sliding_span_owner.capacity)
                    if prefix_rows:
                        maple_kv_span_update_batched(
                            b.sliding_span_owner.spans,
                            start=pos - prefix_rows,
                            rows=prefix_rows,
                            library=libs.attention,
                            runtime=self.runtime,
                        )
                qkv_row_bytes = (q_size + 2 * kv_size) * 2
                attention_row_bytes = q_size * 2
                for row_offset, segment_rows in layer_segments:
                    segment_start = pos + row_offset
                    if layer_kind == "sliding_attention" and replay_swa_per_layer:
                        maple_kv_span_update_batched(
                            b.sliding_span_owner.spans,
                            start=segment_start,
                            rows=segment_rows,
                            library=libs.attention,
                            runtime=self.runtime,
                        )
                    qkv_ptr = pf.qkv.ptr + row_offset * qkv_row_bytes
                    attention_ptr = (
                        pf.attention.ptr + row_offset * attention_row_bytes
                    )
                    maple_qknorm_rope_kv_write_batched_bf16(
                        qkv_ptr,
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
                        start=segment_start,
                        rows=segment_rows,
                        library=libs.attention,
                        runtime=self.runtime,
                    )
                    attention_prefill(
                        qkv_ptr,
                        kv_layer.key_cache.ptr,
                        kv_layer.value_cache.ptr,
                        attention_ptr,
                        kv_layer.spans,
                        rows=segment_rows,
                        q_heads=spec.num_attention_heads,
                        kv_heads=spec.num_key_value_heads,
                        head_dim=spec.head_dim,
                        scale=spec.head_dim**-0.5,
                        start=segment_start,
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
                maple_clamped_swiglu_bf16(
                    pf.expert_gate.ptr,
                    pf.expert_up.ptr,
                    pf.expert_intermediate.ptr,
                    rows * top_k,
                    intermediate,
                    library=libs.moe,
                    runtime=self.runtime,
                )
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
        maple_affine4_gemv_wave32_exact_f32(
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


__all__ = ["MapleCudaRunner"]
