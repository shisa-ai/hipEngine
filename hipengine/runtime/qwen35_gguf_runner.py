"""Qwen3.5 GGUF runtime bring-up probes."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from hipengine.core.hip import HipRuntime, get_hip_runtime
from hipengine.core.memory import copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc
from hipengine.kernels.hip_gfx1100.convert import f32_to_bf16
from hipengine.kernels.hip_gfx1100.fused import (
    gguf_add_rmsnorm_bf16_f32_weight,
    gguf_bf16_add,
    gguf_gate_repeat_value_bf16,
    gguf_rmsnorm_bf16_f32_weight,
    silu_mul_dual_out_bf16,
)
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


@dataclass(frozen=True)
class Qwen35GGUFNextTokenProbeResult:
    token_id: int
    logit: float
    logits: np.ndarray


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
    """One-token GGUF Qwen3.5 decode stack over resident native GGUF weights.

    This runner executes all mapped layers for a single selected token with zeroed
    decode state.  It is the task-34 replacement for the one-layer projection
    probe: every layer runs input RMSNorm, the linear-attention or full-attention
    projection path, residuals, dense FFN, and the final output RMSNorm.  It is
    still a one-token decode bring-up path; prompt prefill/state carry-over for
    llama.cpp parity lands in the follow-up E2E task.
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
        if not token_ids:
            raise ValueError("token_ids must be non-empty")
        return self.run_token_hidden(int(token_ids[-1]), layer_limit=layer_limit)

    def run_token_hidden(self, token_id: int, *, layer_limit: int | None = None) -> np.ndarray:
        """Run all layers plus final norm and return BF16 hidden bits on host."""

        assert self.weights is not None
        runtime = self.runtime or get_hip_runtime()
        token_ids = np.asarray([int(token_id)], dtype=np.int64)
        hidden_bits = np.empty((1, self.hidden_size), dtype=np.uint16)
        buffers = []
        try:
            token_buf = malloc(token_ids.nbytes, runtime=runtime)
            hidden_a = malloc(hidden_bits.nbytes, runtime=runtime)
            hidden_b = malloc(hidden_bits.nbytes, runtime=runtime)
            scratch = _FullStackScratch.allocate(self, runtime=runtime)
            buffers.extend((token_buf, hidden_a, hidden_b, *scratch.buffers))
            copy_host_to_device(token_buf, host_array_ptr(token_ids), runtime=runtime)
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
            layer_count = self.weights.config.block_count if layer_limit is None else int(layer_limit)
            if layer_count < 0 or layer_count > self.weights.config.block_count:
                raise ValueError("layer_limit must be between 0 and block_count")
            for layer_id, layer_type in enumerate(self.weights.config.layer_types[:layer_count]):
                if layer_type == LINEAR_ATTENTION:
                    self._run_linear_attention_layer(layer_id, src.ptr, dst.ptr, scratch)
                elif layer_type == FULL_ATTENTION:
                    self._run_full_attention_layer(layer_id, src.ptr, dst.ptr, scratch)
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

    def _run_linear_attention_layer(self, layer_id: int, hidden_ptr: int, out_ptr: int, scratch) -> None:
        assert self.weights is not None
        layer = self.weights.layer(layer_id)
        cfg = self.weights.config
        runtime = self.runtime or get_hip_runtime()
        _zero(runtime, scratch.conv_state, scratch.conv_zero)
        _zero(runtime, scratch.recurrent_state, scratch.recurrent_zero)
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
            scratch.conv_state.ptr,
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
            scratch.recurrent_state.ptr,
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

    def _run_full_attention_layer(self, layer_id: int, hidden_ptr: int, out_ptr: int, scratch) -> None:
        assert self.weights is not None
        layer = self.weights.layer(layer_id)
        cfg = self.weights.config
        runtime = self.runtime or get_hip_runtime()
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
        # K is computed to exercise the projection path; this one-token decode
        # stack uses V directly because a length-1 attention softmax is 1.0.
        gguf_gate_repeat_value_bf16(
            scratch.full_q.ptr + self.q_width * 2,
            scratch.full_v.ptr,
            scratch.full_gated.ptr,
            cfg.head_count,
            cfg.head_count_kv,
            cfg.value_length,
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
        assert self.weights is not None
        layer = self.weights.layer(layer_id)
        runtime = self.runtime or get_hip_runtime()
        gguf_add_rmsnorm_bf16_f32_weight(
            hidden_ptr,
            attn_out_ptr,
            layer.weight("post_attention_norm").allocation().tensor.ptr,
            scratch.post_norm.ptr,
            scratch.residual.ptr,
            rows=1,
            hidden_size=self.hidden_size,
            eps=self.weights.config.rms_norm_eps,
            runtime=runtime,
        )
        launch_gguf_linear(
            layer.weight("ffn_gate"),
            scratch.post_norm.ptr,
            scratch.ffn_gate_up.ptr,
            rows=1,
            in_features=self.hidden_size,
            out_features=self.ffn_size,
            runtime=runtime,
        )
        launch_gguf_linear(
            layer.weight("ffn_up"),
            scratch.post_norm.ptr,
            scratch.ffn_gate_up.ptr + self.ffn_size * 2,
            rows=1,
            in_features=self.hidden_size,
            out_features=self.ffn_size,
            runtime=runtime,
        )
        silu_mul_dual_out_bf16(
            scratch.ffn_gate_up.ptr,
            scratch.ffn_intermediate.ptr,
            rows=1,
            features=self.ffn_size,
            runtime=runtime,
        )
        launch_gguf_linear(
            layer.weight("ffn_down"),
            scratch.ffn_intermediate.ptr,
            scratch.ffn_down.ptr,
            rows=1,
            in_features=self.ffn_size,
            out_features=self.hidden_size,
            runtime=runtime,
        )
        gguf_bf16_add(scratch.residual.ptr, scratch.ffn_down.ptr, out_ptr, self.hidden_size, runtime=runtime)

    def close(self) -> None:
        if self.weights is not None:
            self.weights.free(runtime=self.runtime)
            self.weights = None

    def __enter__(self) -> "Qwen35GGUFFullStackRunner":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


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
    conv_state: object
    recurrent_state: object
    conv_zero: np.ndarray
    recurrent_zero: np.ndarray
    full_q: object
    full_k: object
    full_v: object
    full_gated: object
    ffn_gate_up: object
    ffn_intermediate: object
    ffn_down: object
    buffers: tuple[object, ...]

    @classmethod
    def allocate(cls, runner: Qwen35GGUFFullStackRunner, *, runtime: HipRuntime):
        def buf(nbytes: int):
            return malloc(nbytes, runtime=runtime)

        hidden_bytes = runner.hidden_size * 2
        ffn_bytes = runner.ffn_size * 2
        linear_qkv_bytes = runner.linear_qkv_width * 2
        ssm_inner_bytes = runner.weights.config.ssm_inner_size * 2
        alpha_bytes = runner.weights.config.ssm_time_step_rank * 2
        q_bytes = 2 * runner.q_width * 2
        kv_bytes = runner.kv_width * 2
        conv_zero = np.zeros(
            (runner.linear_qkv_width, runner.weights.config.ssm_conv_kernel),
            dtype=np.float32,
        )
        recurrent_zero = np.zeros(
            (
                runner.weights.config.ssm_time_step_rank,
                runner.weights.config.ssm_state_size,
                runner.ssm_value_dim,
            ),
            dtype=np.float32,
        )
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
            "recurrent_out": buf(runner.weights.config.ssm_inner_size * 4),
            "recurrent_bf16": buf(ssm_inner_bytes),
            "conv_state": buf(conv_zero.nbytes),
            "recurrent_state": buf(recurrent_zero.nbytes),
            "full_q": buf(q_bytes),
            "full_k": buf(kv_bytes),
            "full_v": buf(kv_bytes),
            "full_gated": buf(runner.q_width * 2),
            "ffn_gate_up": buf(2 * ffn_bytes),
            "ffn_intermediate": buf(ffn_bytes),
            "ffn_down": buf(hidden_bytes),
        }
        return cls(
            **fields,
            conv_zero=conv_zero,
            recurrent_zero=recurrent_zero,
            buffers=tuple(fields.values()),
        )


def _zero(runtime: HipRuntime, buffer, zeros: np.ndarray) -> None:
    copy_host_to_device(buffer, host_array_ptr(zeros), runtime=runtime)


__all__ = [
    "Qwen35GGUFFullStackRunner",
    "Qwen35GGUFNextTokenProbeResult",
    "Qwen35GGUFOneLayerProbe",
]
