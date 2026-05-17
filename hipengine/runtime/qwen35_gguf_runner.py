"""Qwen3.5 GGUF runtime bring-up probes."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.hip import HipRuntime, get_hip_runtime
from hipengine.core.memory import copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc
from hipengine.core.tensor import Tensor
from hipengine.kernels.hip_gfx1100.attention.aotriton_wrap import (
    aotriton_attn_fwd_v3_compact_varlen,
    tensor1 as aotriton_tensor1,
    tensor2 as aotriton_tensor2,
    tensor4 as aotriton_tensor4,
)
from hipengine.kernels.hip_gfx1100.attention.paged_attn_decode import (
    qwen35_full_attn_gate_mul_bf16,
    qwen35_paged_full_attn_decode_context_bf16_spans,
)
from hipengine.kernels.hip_gfx1100.attention.paged_kv_write import (
    qwen35_write_paged_kv_mixed_value_bf16_prompt_spans,
    qwen35_write_paged_kv_mixed_value_bf16_spans,
)
from hipengine.kernels.hip_gfx1100.convert import bf16_to_f32, f32_to_bf16
from hipengine.kernels.hip_gfx1100.fused import (
    gguf_add_rmsnorm_bf16_f32_weight,
    gguf_bf16_add,
    gguf_gate_mul_bf16,
    gguf_qwen35_head_rmsnorm_partial_rotary_position_f32_weight,
    gguf_qwen35_head_rmsnorm_partial_rotary_positions_f32_weight,
    gguf_rmsnorm_bf16_f32_weight,
    silu_mul_dual_out_bf16,
)
from hipengine.kernels.hip_gfx1100.rotary.qwen35_rotary import qwen35_split_qgate_bf16
from hipengine.kvcache import KVLiveSpans
from hipengine.kernels.hip_gfx1100.linear_attn.conv import qwen35_linear_attn_conv_decode_bf16
from hipengine.kernels.hip_gfx1100.linear_attn.gdn import qwen35_gdn_recurrent_rmsnorm_gate_lowp_bf16
from hipengine.kernels.hip_gfx1100.quant.gguf_q6_k_embedding import gguf_q6_k_embedding_bf16_out
from hipengine.loading.qwen35_gguf import FULL_ATTENTION, LINEAR_ATTENTION
from hipengine.loading.qwen35_gguf_materialize import (
    Qwen35GGUFResidentWeights,
    materialize_qwen35_gguf_weights,
)
from hipengine.quant.gguf import bf16_to_float32
from hipengine.runtime.gguf_linear import GGUF_OUTPUT_F32, launch_gguf_linear
from hipengine.runtime.prefill import PrefillConfig


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

            gguf_q6_k_embedding_bf16_out(
                token_buf.ptr,
                self.weights.root("token_embedding").allocation().tensor.ptr,
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
    weights: Qwen35GGUFResidentWeights | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.runtime = self.runtime or get_hip_runtime()
        self.weights = materialize_qwen35_gguf_weights(self.model_path, runtime=self.runtime)

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
                gguf_q6_k_embedding_bf16_out(
                    token_buf.ptr,
                    self.weights.root("token_embedding").allocation().tensor.ptr,
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
        AOTriton V3 wiring before the full model prefill scheduler is promoted.
        Rows below the threshold use the existing resident one-token path in a
        loop; rows at/above the threshold use the compact-varlen AOTriton V3
        attention path after GGUF Q/K/V projection and GPU q/k norm+RoPE.
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
                self._run_full_attention_prefill_layer_aotriton(layer_id, hidden_buf.ptr, out_buf.ptr, prefill_scratch)
                mode = "aotriton_v3"
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
            used_aotriton=use_aotriton,
        )

    def _run_full_attention_prefill_layer_aotriton(
        self,
        layer_id: int,
        hidden_ptr: int,
        out_ptr: int,
        scratch,
        *,
        stream: int = 0,
    ) -> None:
        assert self.weights is not None
        layer = self.weights.layer(layer_id)
        cfg = self.weights.config
        runtime = self.runtime or get_hip_runtime()
        rows = scratch.rows
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
            runtime=runtime,
        )
        f32_to_bf16(
            scratch.full_query.ptr,
            scratch.full_query_bf16.ptr,
            rows * self.q_width,
            stream=stream,
            runtime=runtime,
        )
        aotriton_attn_fwd_v3_compact_varlen(
            aotriton_tensor4(
                scratch.full_query_bf16.ptr,
                (1, cfg.head_count, rows, cfg.key_length),
                (self.q_width * rows, cfg.key_length, self.q_width, 1),
                DType.BF16,
            ),
            aotriton_tensor4(
                scratch.key_cache.ptr,
                (1, cfg.head_count_kv, rows, cfg.key_length),
                (self.kv_width * rows, cfg.key_length, self.kv_width, 1),
                DType.BF16,
            ),
            aotriton_tensor4(
                scratch.value_cache.ptr,
                (1, cfg.head_count_kv, rows, cfg.key_length),
                (self.kv_width * rows, cfg.key_length, self.kv_width, 1),
                DType.BF16,
            ),
            aotriton_tensor1(scratch.cu_q.ptr, (2,), (1,), DType.INT32),
            aotriton_tensor1(scratch.cu_k.ptr, (2,), (1,), DType.INT32),
            aotriton_tensor2(scratch.softmax_lse.ptr, (cfg.head_count, rows), (rows, 1), DType.FP32),
            aotriton_tensor4(
                scratch.full_attn_bf16.ptr,
                (1, cfg.head_count, rows, cfg.key_length),
                (self.q_width * rows, cfg.key_length, self.q_width, 1),
                DType.BF16,
            ),
            persistent_atomic_counter_ptr=scratch.atomic.ptr,
            max_seqlen_q=rows,
            max_seqlen_k=rows,
            sm_scale=cfg.key_length ** -0.5,
            is_causal=True,
            stream=stream,
            runtime=runtime,
        )
        gguf_gate_mul_bf16(
            scratch.full_attn_bf16.ptr,
            scratch.full_gate.ptr,
            scratch.full_gated.ptr,
            rows * self.q_width,
            stream=stream,
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
        self._run_post_attention_ffn_rows(layer_id, hidden_ptr, scratch.attn_out.ptr, out_ptr, scratch, rows=rows, stream=stream)

    def _run_linear_attention_layer(self, layer_id: int, hidden_ptr: int, out_ptr: int, scratch) -> None:
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
            runtime=runtime,
        )
        launch_gguf_linear(
            layer.weight("attn_qkv"),
            scratch.norm.ptr,
            scratch.linear_qkv.ptr,
            rows=1,
            in_features=self.hidden_size,
            out_features=self.linear_qkv_width,
            runtime=runtime,
        )
        launch_gguf_linear(
            layer.weight("attn_gate"),
            scratch.norm.ptr,
            scratch.linear_z.ptr,
            rows=1,
            in_features=self.hidden_size,
            out_features=cfg.ssm_inner_size,
            runtime=runtime,
        )
        launch_gguf_linear(
            layer.weight("ssm_alpha"),
            scratch.norm.ptr,
            scratch.linear_alpha.ptr,
            rows=1,
            in_features=self.hidden_size,
            out_features=cfg.ssm_time_step_rank,
            runtime=runtime,
        )
        launch_gguf_linear(
            layer.weight("ssm_beta"),
            scratch.norm.ptr,
            scratch.linear_beta.ptr,
            rows=1,
            in_features=self.hidden_size,
            out_features=cfg.ssm_time_step_rank,
            runtime=runtime,
        )
        qwen35_linear_attn_conv_decode_bf16(
            scratch.linear_qkv.ptr,
            conv_state.ptr,
            layer.weight("ssm_conv1d").allocation().tensor.ptr,
            scratch.conv_out.ptr,
            self.linear_qkv_width,
            cfg.ssm_conv_kernel,
            runtime=runtime,
        )
        qwen35_gdn_recurrent_rmsnorm_gate_lowp_bf16(
            scratch.conv_out.ptr,
            scratch.linear_z.ptr,
            scratch.linear_alpha.ptr,
            scratch.linear_beta.ptr,
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
            runtime=runtime,
        )
        f32_to_bf16(
            scratch.recurrent_out.ptr,
            scratch.recurrent_bf16.ptr,
            cfg.ssm_inner_size,
            runtime=runtime,
        )
        launch_gguf_linear(
            layer.weight("ssm_out"),
            scratch.recurrent_bf16.ptr,
            scratch.attn_out.ptr,
            rows=1,
            in_features=cfg.ssm_inner_size,
            out_features=self.hidden_size,
            runtime=runtime,
        )
        self._run_post_attention_ffn(layer_id, hidden_ptr, scratch.attn_out.ptr, out_ptr, scratch)

    def _run_full_attention_layer(
        self,
        layer_id: int,
        hidden_ptr: int,
        out_ptr: int,
        scratch,
        *,
        position: int,
    ) -> None:
        assert self.weights is not None
        layer = self.weights.layer(layer_id)
        cfg = self.weights.config
        runtime = self.runtime or get_hip_runtime()
        if int(scratch.position_host[0]) != int(position):
            scratch.set_full_attention_position(position, runtime)
        gguf_rmsnorm_bf16_f32_weight(
            hidden_ptr,
            layer.weight("attn_norm").allocation().tensor.ptr,
            scratch.norm.ptr,
            rows=1,
            hidden_size=self.hidden_size,
            eps=cfg.rms_norm_eps,
            runtime=runtime,
        )
        launch_gguf_linear(
            layer.weight("attn_q"),
            scratch.norm.ptr,
            scratch.full_q.ptr,
            rows=1,
            in_features=self.hidden_size,
            out_features=2 * self.q_width,
            runtime=runtime,
        )
        launch_gguf_linear(
            layer.weight("attn_k"),
            scratch.norm.ptr,
            scratch.full_k.ptr,
            rows=1,
            in_features=self.hidden_size,
            out_features=self.kv_width,
            runtime=runtime,
        )
        launch_gguf_linear(
            layer.weight("attn_v"),
            scratch.norm.ptr,
            scratch.full_v.ptr,
            rows=1,
            in_features=self.hidden_size,
            out_features=self.kv_width,
            runtime=runtime,
        )
        qwen35_split_qgate_bf16(
            scratch.full_q.ptr,
            scratch.full_query_raw.ptr,
            scratch.full_gate.ptr,
            1,
            cfg.head_count,
            cfg.key_length,
            runtime=runtime,
        )
        bf16_to_f32(
            scratch.full_k.ptr,
            scratch.full_key_raw.ptr,
            self.kv_width,
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
            runtime=runtime,
        )
        key_cache, value_cache = scratch.full_cache(layer_id)
        qwen35_write_paged_kv_mixed_value_bf16_spans(
            scratch.full_key.ptr,
            scratch.full_v.ptr,
            key_cache.ptr,
            value_cache.ptr,
            scratch.append_spans,
            scratch.block_size,
            cfg.head_count_kv,
            cfg.key_length,
            runtime=runtime,
        )
        qwen35_paged_full_attn_decode_context_bf16_spans(
            scratch.full_query.ptr,
            key_cache.ptr,
            value_cache.ptr,
            scratch.full_attn_context.ptr,
            scratch.decode_spans,
            scratch.max_positions,
            scratch.block_size,
            cfg.head_count,
            cfg.head_count_kv,
            cfg.key_length,
            cfg.key_length ** -0.5,
            runtime=runtime,
        )
        qwen35_full_attn_gate_mul_bf16(
            scratch.full_attn_context.ptr,
            scratch.full_gate.ptr,
            scratch.full_gated.ptr,
            self.q_width,
            runtime=runtime,
        )
        launch_gguf_linear(
            layer.weight("attn_output"),
            scratch.full_gated.ptr,
            scratch.attn_out.ptr,
            rows=1,
            in_features=self.q_width,
            out_features=self.hidden_size,
            runtime=runtime,
        )
        self._run_post_attention_ffn(layer_id, hidden_ptr, scratch.attn_out.ptr, out_ptr, scratch)

    def _run_post_attention_ffn(self, layer_id: int, hidden_ptr: int, attn_out_ptr: int, out_ptr: int, scratch) -> None:
        self._run_post_attention_ffn_rows(layer_id, hidden_ptr, attn_out_ptr, out_ptr, scratch, rows=1)

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
        silu_mul_dual_out_bf16(
            scratch.ffn_gate_up.ptr,
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

    def close(self) -> None:
        if self.weights is not None:
            self.weights.free(runtime=self.runtime)
            self.weights = None

    def __enter__(self) -> "Qwen35GGUFFullStackRunner":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


@dataclass
class Qwen35GGUFResidentSession:
    """Persistent GGUF Qwen3.5 session for public greedy generation.

    The session materializes GGUF weights once, owns reusable device scratch, and
    carries linear-attention recurrent state plus paged full-attention K/V cache
    across decode steps.  Full-attention q/k norm, RoPE, KV append, softmax, and
    gate application now stay on GPU for the one-token resident path; rows>1
    prefill and AOTriton are still follow-up work.
    """

    model_path: str | Path
    runtime: HipRuntime | None = None
    runner: Qwen35GGUFFullStackRunner | None = field(default=None, init=False)
    scratch: object | None = field(default=None, init=False)
    _token_buf: object | None = field(default=None, init=False)
    _hidden_a: object | None = field(default=None, init=False)
    _hidden_b: object | None = field(default=None, init=False)
    _logits_buf: object | None = field(default=None, init=False)
    _token_host: np.ndarray = field(default_factory=lambda: np.empty((1,), dtype=np.int64), init=False)
    _logits_host: np.ndarray | None = field(default=None, init=False)
    _buffers: tuple[object, ...] = field(default=(), init=False)
    _position: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.runtime = self.runtime or get_hip_runtime()
        self.runner = Qwen35GGUFFullStackRunner(self.model_path, runtime=self.runtime)
        runtime = self.runtime or get_hip_runtime()
        self.scratch = _FullStackScratch.allocate(self.runner, runtime=runtime)
        self._token_buf = malloc(self._token_host.nbytes, runtime=runtime)
        hidden_bytes = self.runner.hidden_size * 2
        self._hidden_a = malloc(hidden_bytes, runtime=runtime)
        self._hidden_b = malloc(hidden_bytes, runtime=runtime)
        self._logits_host = np.empty((1, self.runner.vocab_size), dtype=np.float32)
        self._logits_buf = malloc(self._logits_host.nbytes, runtime=runtime)
        self._buffers = (self._token_buf, self._hidden_a, self._hidden_b, self._logits_buf)
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

    def prefill(self, token_ids: list[int] | tuple[int, ...]) -> Qwen35GGUFNextTokenProbeResult:
        """Consume prompt tokens once and return greedy next-token logits."""

        if not token_ids:
            raise ValueError("token_ids must be non-empty")
        self.reset()
        hidden_ptr = None
        for token_id in token_ids:
            hidden_ptr = self._run_token_to_final_hidden(int(token_id), position=self._position)
            self._position += 1
        assert hidden_ptr is not None
        return self._sample_from_hidden(hidden_ptr)

    def step(self, token_id: int, position: int | None = None) -> Qwen35GGUFNextTokenProbeResult:
        """Consume one generated token and return the next greedy token.

        ``position`` is optional because the session tracks its own decode
        cursor.  When supplied, it is validated to catch caller/context drift.
        """

        if position is not None and int(position) != self._position:
            raise ValueError(f"position {position} does not match session cursor {self._position}")
        hidden_ptr = self._run_token_to_final_hidden(int(token_id), position=self._position)
        self._position += 1
        return self._sample_from_hidden(hidden_ptr)

    def _run_token_to_final_hidden(self, token_id: int, *, position: int) -> int:
        if self.runner is None or self.scratch is None:
            raise RuntimeError("GGUF resident session is closed")
        if self._token_buf is None or self._hidden_a is None or self._hidden_b is None:
            raise RuntimeError("GGUF resident session buffers are closed")
        assert self.runner.weights is not None
        runtime = self.runtime or get_hip_runtime()
        self.scratch.set_full_attention_position(position, runtime)
        self._token_host[0] = int(token_id)
        copy_host_to_device(self._token_buf, host_array_ptr(self._token_host), runtime=runtime)
        gguf_q6_k_embedding_bf16_out(
            self._token_buf.ptr,
            self.runner.weights.root("token_embedding").allocation().tensor.ptr,
            self._hidden_a.ptr,
            rows=1,
            hidden_size=self.runner.hidden_size,
            vocab_size=self.runner.vocab_size,
            runtime=runtime,
        )
        src = self._hidden_a
        dst = self._hidden_b
        for layer_id, layer_type in enumerate(self.runner.weights.config.layer_types):
            if layer_type == LINEAR_ATTENTION:
                self.runner._run_linear_attention_layer(layer_id, src.ptr, dst.ptr, self.scratch)
            elif layer_type == FULL_ATTENTION:
                self.runner._run_full_attention_layer(layer_id, src.ptr, dst.ptr, self.scratch, position=position)
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
            runtime=runtime,
        )
        return self.scratch.norm.ptr

    def _sample_from_hidden(self, hidden_ptr: int) -> Qwen35GGUFNextTokenProbeResult:
        if self.runner is None or self._logits_buf is None or self._logits_host is None:
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
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(self._logits_host), self._logits_buf, runtime=runtime)
        if not np.all(np.isfinite(self._logits_host)):
            raise FloatingPointError("GGUF resident lm-head logits contain NaN or Inf")
        flat = self._logits_host.reshape(-1)
        next_id = int(np.argmax(flat))
        return Qwen35GGUFNextTokenProbeResult(
            token_id=next_id,
            logit=float(flat[next_id]),
            logits=self._logits_host.copy(),
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
        self._logits_host = None

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
    key_cache: object
    value_cache: object
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
    buffers: tuple[object, ...]

    @classmethod
    def allocate(cls, runner: Qwen35GGUFFullStackRunner, *, rows: int, runtime: HipRuntime):
        if rows <= 0:
            raise ValueError("rows must be positive")
        assert runner.weights is not None
        cfg = runner.weights.config
        device = Device("hip", 0)
        block_size = 256
        blocks = (int(rows) + block_size - 1) // block_size
        max_positions = blocks * block_size

        def buf(nbytes: int):
            return malloc(nbytes, runtime=runtime)

        hidden_bytes = rows * runner.hidden_size * 2
        q_proj_bytes = rows * 2 * runner.q_width * 2
        kv_bf16_bytes = rows * runner.kv_width * 2
        q_f32_bytes = rows * runner.q_width * 4
        kv_f32_bytes = rows * runner.kv_width * 4
        ffn_bytes = rows * runner.ffn_size * 2
        cache_nbytes = max_positions * cfg.head_count_kv * cfg.key_length * 2
        block_table_arr = np.tile(np.arange(blocks, dtype=np.int32), (rows, 1))
        positions_arr = np.arange(rows, dtype=np.int64)
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
            "ffn_gate_up": buf(2 * ffn_bytes),
            "ffn_intermediate": buf(ffn_bytes),
            "ffn_down": buf(hidden_bytes),
            "key_cache": buf(cache_nbytes),
            "value_cache": buf(cache_nbytes),
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
            buffers=tuple(fields.values()),
        )


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
    buffers: tuple[object, ...]

    @classmethod
    def allocate(cls, runner: Qwen35GGUFFullStackRunner, *, runtime: HipRuntime):
        def buf(nbytes: int):
            return malloc(nbytes, runtime=runtime)

        assert runner.weights is not None
        cfg = runner.weights.config
        device = Device("hip", 0)
        block_size = 256
        max_positions = min(int(cfg.context_length), block_size)
        hidden_bytes = runner.hidden_size * 2
        ffn_bytes = runner.ffn_size * 2
        linear_qkv_bytes = runner.linear_qkv_width * 2
        ssm_inner_bytes = cfg.ssm_inner_size * 2
        alpha_bytes = cfg.ssm_time_step_rank * 2
        q_proj_bytes = 2 * runner.q_width * 2
        kv_bf16_bytes = runner.kv_width * 2
        q_f32_bytes = runner.q_width * 4
        kv_f32_bytes = runner.kv_width * 4
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
        block_table_arr = np.asarray([0], dtype=np.int32)
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
            "full_gated": buf(runner.q_width * 2),
            "ffn_gate_up": buf(2 * ffn_bytes),
            "ffn_intermediate": buf(ffn_bytes),
            "ffn_down": buf(hidden_bytes),
        }
        metadata_buffers = (block_table, position_buf, context_buf, cos_table_buf, sin_table_buf)
        return cls(
            **fields,
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


def _zero(runtime: HipRuntime, buffer, zeros: np.ndarray) -> None:
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
    "Qwen35GGUFFullStackRunner",
    "Qwen35GGUFNextTokenProbeResult",
    "Qwen35GGUFOneLayerProbe",
    "Qwen35GGUFResidentSession",
]
