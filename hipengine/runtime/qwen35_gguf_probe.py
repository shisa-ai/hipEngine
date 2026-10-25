"""Diagnostic one-layer projection probe over resident Qwen3.5 GGUF weights.

The probe is a partial projection utility, not full-stack execution: starting
from one token embedding it applies a single layer's RMSNorm and the layer's
``attn_gate``/``ssm_out`` GGUF linear projections, then optionally runs the tied
``lm_head`` to produce logits. Conv, SSM, attention, residual, and MLP state live
in the resident session and full-stack runner, not here.

The result contract itself is independent of this diagnostic module. The legacy
runner import path still re-exports the probe for compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from hipengine.core.hip import HipRuntime, get_hip_runtime
from hipengine.core.memory import (
    DeviceBuffer,
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.hip_gfx1100.fused.gguf_ops import gguf_rmsnorm_bf16_f32_weight
from hipengine.loading.qwen35_gguf import LINEAR_ATTENTION
from hipengine.loading.qwen35_gguf_materialize import (
    Qwen35GGUFResidentWeights,
    materialize_qwen35_gguf_weights,
)
from hipengine.quant.gguf import bf16_to_float32
from hipengine.runtime.gguf_embedding import launch_gguf_embedding
from hipengine.runtime.gguf_linear import GGUF_OUTPUT_F32, launch_gguf_linear
from hipengine.runtime.qwen35_gguf_results import Qwen35GGUFNextTokenProbeResult


@dataclass
class Qwen35GGUFOneLayerProbe:
    """Diagnostic one-layer projection probe over resident GGUF weights.

    A diagnostic partial projection, not the Qwen3.5 layer and not full-stack
    execution. It materializes only the embedding, the tied ``lm_head``, and
    one linear_attention layer's ``attn_norm``/``attn_gate``/``ssm_out``
    projections, then exposes the resulting hidden-size BF16 bits and tied-head
    logits. Resident conv/SSM/attention state, residual, and MLP execution
    belong to the resident session and full-stack runner.
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
        weights = materialize_qwen35_gguf_weights(
            self.model_path,
            selected_slots=selected,
            # This probe executes projections, not any MoE expert call.
            selected_call_intents=(),
            runtime=self.runtime,
        )
        try:
            if weights.config.layer_types[self.layer_id] != LINEAR_ATTENTION:
                raise ValueError(f"layer {self.layer_id} is not a linear_attention layer")
        except Exception:
            weights.free(runtime=self.runtime)
            raise
        self.weights = weights

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
        buffers: list[DeviceBuffer] = []
        try:
            token_buf = malloc(token_ids.nbytes, runtime=runtime)
            buffers.append(token_buf)
            hidden_buf = malloc(out_bits.nbytes, runtime=runtime)
            buffers.append(hidden_buf)
            norm_buf = malloc(out_bits.nbytes, runtime=runtime)
            buffers.append(norm_buf)
            gate_buf = malloc(2 * self.ssm_inner_size, runtime=runtime)
            buffers.append(gate_buf)
            out_buf = malloc(out_bits.nbytes, runtime=runtime)
            buffers.append(out_buf)
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
        buffers: list[DeviceBuffer] = []
        try:
            hidden_buf = malloc(hidden.nbytes, runtime=runtime)
            buffers.append(hidden_buf)
            logits_buf = malloc(logits.nbytes, runtime=runtime)
            buffers.append(logits_buf)
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


__all__ = ["Qwen35GGUFOneLayerProbe"]
