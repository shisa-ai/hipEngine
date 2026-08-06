"""Torch-free fixed-address Moonshine FP16 encoder runtime for CUDA ``sm_120a`` (C4).

The C4 standalone encoder owns every fixed-address object needed to turn one
padded audio bucket into the resident encoder hidden state and the downsampled
int32 encoder attention mask, without PyTorch and without any timed
allocation.  The pipeline mirrors the HF ``MoonshineEncoder`` forward exactly:

  conv1(127, stride 64) + tanh -> GroupNorm(1, 416) -> conv2(7, stride 3)
  + GELU -> conv3(3, stride 2) + GELU -> [batch, frames, 416] -> eight
  MoonshineEncoderLayer blocks (input LayerNorm -> non-causal full-sequence
  self-attention with partial RoPE -> residual -> post-attention LayerNorm ->
  fc1 + GELU + fc2 -> residual) -> final LayerNorm.

Conv/GELU/GroupNorm/RoPE/attention are the C4b encoder primitives; LayerNorm,
the head-major Q/K/V (and o-) projections, and the MLP re-use the measured
C1b/C1c/C1d kernels.  A single row-to-head-major transpose bridges the
row-major projection layout to the head-major layout consumed by RoPE and the
encoder self-attention kernel.  The finished hidden + mask are handed off to a
``MoonshineCudaResidentRuntime`` through ``set_encoder_state_from_device``
followed by ``precompute_cross_kv``, closing the full torch-free encoder ->
decoder ASR path.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from hipengine.core.cuda import CudaRuntime, get_cuda_runtime
from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.memory import copy_host_to_device, host_array_ptr, memory_stats
from hipengine.core.tensor import Tensor
from hipengine.kernels.cpu_reference.moonshine import moonshine_rope_tables
from hipengine.loading.moonshine import MoonshineLoadedModel, load_moonshine_model
from hipengine.runtime.workspace import RuntimeWorkspace

# Moonshine conv chain (pinned model contract): conv1 kernel 127 stride 64,
# conv2 kernel 7 stride 3, conv3 kernel 3 stride 2.  The encoder frame count is
# the output of the valid-conv chain over the raw audio length, and the
# attention mask downsample stride is the product of the three strides (384).
_CONV1_KERNEL = 127
_CONV1_STRIDE = 64
_CONV2_KERNEL = 7
_CONV2_STRIDE = 3
_CONV3_KERNEL = 3
_CONV3_STRIDE = 2
_DOWNSAMPLE_STRIDE = _CONV1_STRIDE * _CONV2_STRIDE * _CONV3_STRIDE
_ENCODER_HEADS = 8
_ENCODER_HEAD_DIM = 52
_GROUPNORM_EPS = 1.0e-5

# Measured C4b schedule: conv1/groupnorm/gelu/rope/transpose use t256, conv2
# uses t832 (one thread per output channel), conv3 uses t416, and the encoder
# self-attention uses one 32-lane wave per (query, head).  The conv thread
# counts are enforced inside the conv2/conv3 wrappers.


def moonshine_encoder_frames_from_audio(audio_samples: int) -> int:
    """Return the exact Moonshine encoder frame count for ``audio_samples``."""

    if isinstance(audio_samples, bool) or not isinstance(audio_samples, int):
        raise ValueError("audio_samples must be an integer")
    if audio_samples <= 0:
        raise ValueError("audio_samples must be positive")
    length = (audio_samples - _CONV1_KERNEL) // _CONV1_STRIDE + 1
    if length <= 0:
        raise ValueError("audio_samples too short for the conv1 kernel")
    length = (length - _CONV2_KERNEL) // _CONV2_STRIDE + 1
    if length <= 0:
        raise ValueError("audio is too short after conv1 for the conv2 kernel")
    length = (length - _CONV3_KERNEL) // _CONV3_STRIDE + 1
    if length <= 0:
        raise ValueError("audio is too short after conv2 for the conv3 kernel")
    return int(length)


@dataclass(frozen=True)
class MoonshineCudaEncoderLibraries:
    """Prebuilt code objects used by the CUDA Moonshine encoder chain."""

    encoder: object
    layernorm: object
    projection: object


class MoonshineCudaEncoderRuntime:
    """Own every fixed-address object needed to encode one Moonshine audio bucket."""

    _SCRATCH_NAMES = (
        "audio",
        "conv1_out",
        "conv2_out",
        "groupnorm_partial",
        "groupnorm_mean_rstd",
        "hidden",
        "normalized",
        "query_row",
        "key_row",
        "value_row",
        "query",
        "key",
        "value",
        "attention",
        "projection",
        "mlp_fc1",
        "mlp_gelu",
        "encoder_output",
        "encoder_attention_mask",
    )

    def __init__(
        self,
        *,
        audio_samples: int,
        model_path: str | Path | None = None,
        loaded_model: MoonshineLoadedModel | None = None,
        device: Device | None = None,
        runtime: CudaRuntime | None = None,
        owns_weights: bool = True,
    ) -> None:
        if (model_path is None) == (loaded_model is None):
            raise ValueError("provide exactly one of model_path or loaded_model")
        self.runtime = runtime or get_cuda_runtime()
        self.device = device or Device("cuda", 0)
        self.loaded_model = loaded_model
        self.weights = loaded_model.weights if loaded_model is not None else None
        self.spec = loaded_model.spec if loaded_model is not None else None
        self.owns_weights = bool(owns_weights)
        self.audio_samples = int(audio_samples)
        self.encoder_frames = moonshine_encoder_frames_from_audio(self.audio_samples)
        self.workspace = RuntimeWorkspace(device=self.device, runtime=self.runtime)
        self.stream = 0
        self.encoder_libraries: MoonshineCudaEncoderLibraries | None = None
        self.closed = False
        self.teardown_returned_to_baseline: bool | None = None
        self._allocation_baseline = memory_stats()["current_allocated_bytes"]
        try:
            if self.loaded_model is None:
                self.loaded_model = load_moonshine_model(
                    model_path,
                    device=self.device,
                    runtime=self.runtime,
                )
                self.weights = self.loaded_model.weights
                self.spec = self.loaded_model.spec
            assert self.loaded_model is not None
            assert self.weights is not None
            assert self.spec is not None
            if self.audio_samples <= 0:
                raise ValueError("audio_samples must be positive")
            self.stream = self.runtime.stream_create(nonblocking=True)
            self._reserve_workspace()
            self._initialize_workspace()
        except Exception:
            self.close()
            raise

    def _reserve_workspace(self) -> None:
        assert self.spec is not None
        spec = self.spec
        frames = self.encoder_frames
        hidden = spec.hidden_size
        intermediate = spec.intermediate_size
        reserve = self.workspace.reserve_tensor
        reserve("rope_cos", (spec.max_positions, spec.rotary_dim // 2), DType.FP16)
        reserve("rope_sin", (spec.max_positions, spec.rotary_dim // 2), DType.FP16)
        reserve("audio", (1, self.audio_samples), DType.FP16)
        length = (self.audio_samples - _CONV1_KERNEL) // _CONV1_STRIDE + 1
        reserve("conv1_out", (hidden, length), DType.FP16)
        length = (length - _CONV2_KERNEL) // _CONV2_STRIDE + 1
        reserve("conv2_out", (2 * hidden, length), DType.FP16)
        reserve("groupnorm_partial", (2 * hidden,), DType.FP32)
        reserve("groupnorm_mean_rstd", (2,), DType.FP32)
        reserve("hidden", (1, frames, hidden), DType.FP16)
        reserve("normalized", (1, frames, hidden), DType.FP16)
        for name in ("query_row", "key_row", "value_row"):
            reserve(name, (1, frames, hidden), DType.FP16)
        for name in ("query", "key", "value"):
            reserve(name, (_ENCODER_HEADS, frames, spec.head_dim), DType.FP16)
        reserve("attention", (1, frames, hidden), DType.FP16)
        reserve("projection", (1, frames, hidden), DType.FP16)
        reserve("mlp_fc1", (1, frames, intermediate), DType.FP16)
        reserve("mlp_gelu", (1, frames, intermediate), DType.FP16)
        reserve("encoder_output", (1, frames, hidden), DType.FP16)
        reserve("encoder_attention_mask", (1, frames), DType.INT32)

    def _initialize_workspace(self) -> None:
        assert self.spec is not None
        for name in self.workspace.names:
            allocation = self.workspace.allocation(name)
            self.runtime.memset_async(
                allocation.buffer.ptr, 0, allocation.buffer.nbytes, self.stream
            )
        self.runtime.stream_synchronize(self.stream)
        cos, sin = moonshine_rope_tables(
            self.spec.max_positions,
            rotary_dim=self.spec.rotary_dim,
            theta=self.spec.rope_theta,
        )
        copy_host_to_device(
            self.workspace.allocation("rope_cos").buffer,
            host_array_ptr(cos),
            runtime=self.runtime,
        )
        copy_host_to_device(
            self.workspace.allocation("rope_sin").buffer,
            host_array_ptr(sin),
            runtime=self.runtime,
        )

    def prepare_encoder_kernels(
        self,
        *,
        libraries: MoonshineCudaEncoderLibraries | None = None,
        compiler_version: str | None = None,
        require_cached: bool = False,
    ) -> MoonshineCudaEncoderLibraries:
        """Load every code object before the timed, no-allocation encode region."""

        if self.closed:
            raise RuntimeError("Moonshine encoder runtime is closed")
        if libraries is None:
            from hipengine.kernels.cuda_sm120a.encoder.moonshine_encoder import (
                build_moonshine_encoder,
            )
            from hipengine.kernels.cuda_sm120a.linear.moonshine_projection import (
                build_moonshine_projection,
            )
            from hipengine.kernels.cuda_sm120a.norm.moonshine_layernorm import (
                build_moonshine_layernorm,
            )

            arguments = {
                "compiler_version": compiler_version,
                "load": True,
                "require_cached": require_cached,
            }
            libraries = MoonshineCudaEncoderLibraries(
                encoder=build_moonshine_encoder(**arguments),
                layernorm=build_moonshine_layernorm(**arguments),
                projection=build_moonshine_projection(**arguments),
            )
        self.encoder_libraries = libraries
        return libraries

    def tensor(self, name: str) -> Tensor:
        return self.workspace.allocation(name).tensor

    def encoder_output(self) -> Tensor:
        """The finished row-major ``[1, frames, hidden]`` FP16 encoder hidden state."""

        return self.tensor("encoder_output")

    def attention_mask(self) -> Tensor:
        """The downsampled int32 ``[1, frames]`` encoder attention mask."""

        return self.tensor("encoder_attention_mask")

    def encode(
        self,
        input_values: np.ndarray,
        attention_mask: np.ndarray | None = None,
        *,
        stream: int | None = None,
    ) -> None:
        """Upload one audio bucket and run the full fixed-address encoder DAG.

        ``input_values`` must be a finite ``[1, audio_samples]`` float32 (or
        float16) array and ``attention_mask`` (optional) a ``[1, audio_samples]``
        int64/int32 mask.  On return the device-resident ``encoder_output`` and
        ``encoder_attention_mask`` tensors are ready for a decoder handoff.
        """

        if self.closed:
            raise RuntimeError("Moonshine encoder runtime is closed")
        audio = np.asarray(input_values)
        if audio.ndim != 2 or audio.shape != (1, self.audio_samples):
            raise ValueError(
                f"input_values must have shape (1, {self.audio_samples})"
            )
        if audio.dtype == np.float32:
            values = audio.astype(np.float16)
        elif audio.dtype == np.float16:
            values = audio
        else:
            raise ValueError("input_values must be float32 or float16")
        if not bool(np.isfinite(values.astype(np.float32)).all()):
            raise ValueError("input_values must contain only finite values")
        values = np.ascontiguousarray(values)
        copy_host_to_device(
            self.workspace.allocation("audio").buffer,
            host_array_ptr(values),
            runtime=self.runtime,
        )
        self._upload_mask(attention_mask)
        use_stream = self.stream if stream is None else int(stream)
        self._enqueue_encode(stream=use_stream)
        self.runtime.stream_synchronize(use_stream)

    def _upload_mask(self, attention_mask: np.ndarray | None) -> None:
        if attention_mask is None:
            mask_values = np.ones((1, self.audio_samples), dtype=np.int64)
        else:
            mask_values = np.asarray(attention_mask)
            if mask_values.shape != (1, self.audio_samples):
                raise ValueError(
                    f"attention_mask must have shape (1, {self.audio_samples})"
                )
        if not bool(((mask_values == 0) | (mask_values == 1)).all()):
            raise ValueError("attention_mask must be binary")
        output = (
            mask_values[..., ::_DOWNSAMPLE_STRIDE][..., : self.encoder_frames]
            .astype(np.int32)
            .copy()
        )
        copy_host_to_device(
            self.workspace.allocation("encoder_attention_mask").buffer,
            host_array_ptr(output),
            runtime=self.runtime,
        )

    def _enqueue_encode(self, *, stream: int) -> None:
        """Enqueue the fixed-address encoder DAG without changing host-owned state."""

        if self.closed or self.spec is None or self.weights is None:
            raise RuntimeError("Moonshine encoder runtime is closed")
        libraries = self.encoder_libraries
        if libraries is None:
            raise RuntimeError("Moonshine encoder kernels are not prepared")
        from hipengine.kernels.cuda_sm120a.encoder.moonshine_encoder import (
            moonshine_conv1_tanh_fp16,
            moonshine_conv2_gelu_fp16,
            moonshine_conv3_gelu_fp16,
            moonshine_encoder_attention_fp16,
            moonshine_encoder_rope_fp16,
            moonshine_encoder_transpose_head_major_fp16,
            moonshine_gelu_fp16,
            moonshine_groupnorm_fp16,
        )
        from hipengine.kernels.cuda_sm120a.linear.moonshine_projection import (
            moonshine_f16_projection,
            moonshine_f16_projection_bias,
            moonshine_f16_projection_bias_residual,
            moonshine_f16_projection_triple,
        )
        from hipengine.kernels.cuda_sm120a.norm.moonshine_layernorm import (
            moonshine_layernorm_fp16,
            moonshine_residual_layernorm_fp16,
        )

        spec = self.spec
        frames = self.encoder_frames
        hidden = spec.hidden_size
        intermediate = spec.intermediate_size
        heads = _ENCODER_HEADS
        head_dim = spec.head_dim
        common = {"stream": stream, "runtime": self.runtime}

        # ---- conv front end -------------------------------------------------
        length = (self.audio_samples - _CONV1_KERNEL) // _CONV1_STRIDE + 1
        moonshine_conv1_tanh_fp16(
            self.tensor("audio").ptr,
            self.weights["model.encoder.conv1.weight"].ptr,
            self.tensor("conv1_out").ptr,
            self.audio_samples,
            length,
            library=libraries.encoder,
            **common,
        )
        moonshine_groupnorm_fp16(
            self.tensor("conv1_out").ptr,
            self.weights["model.encoder.groupnorm.weight"].ptr,
            self.weights["model.encoder.groupnorm.bias"].ptr,
            self.tensor("conv1_out").ptr,
            self.tensor("groupnorm_partial").ptr,
            self.tensor("groupnorm_mean_rstd").ptr,
            hidden,
            length,
            eps=_GROUPNORM_EPS,
            library=libraries.encoder,
            **common,
        )
        conv1_length = length
        conv2_length = (length - _CONV2_KERNEL) // _CONV2_STRIDE + 1
        moonshine_conv2_gelu_fp16(
            self.tensor("conv1_out").ptr,
            self.weights["model.encoder.conv2.weight"].ptr,
            self.weights["model.encoder.conv2.bias"].ptr,
            self.tensor("conv2_out").ptr,
            conv1_length,
            conv2_length,
            library=libraries.encoder,
            **common,
        )
        conv3_length = (conv2_length - _CONV3_KERNEL) // _CONV3_STRIDE + 1
        assert conv3_length == frames
        moonshine_conv3_gelu_fp16(
            self.tensor("conv2_out").ptr,
            self.weights["model.encoder.conv3.weight"].ptr,
            self.weights["model.encoder.conv3.bias"].ptr,
            self.tensor("hidden").ptr,
            conv2_length,
            conv3_length,
            library=libraries.encoder,
            **common,
        )

        # ---- eight encoder layers -------------------------------------------
        mask = self.tensor("encoder_attention_mask").ptr
        cos = self.tensor("rope_cos").ptr
        sin = self.tensor("rope_sin").ptr
        scale = head_dim**-0.5
        for layer in range(spec.encoder_layers):
            prefix = f"model.encoder.layers.{layer}"
            moonshine_layernorm_fp16(
                self.tensor("hidden").ptr,
                self.weights[f"{prefix}.input_layernorm.weight"].ptr,
                self.tensor("normalized").ptr,
                frames,
                hidden,
                eps=spec.layer_norm_epsilon,
                library=libraries.layernorm,
                **common,
            )
            moonshine_f16_projection_triple(
                self.tensor("normalized").ptr,
                self.weights[f"{prefix}.self_attn.q_proj.weight"].ptr,
                self.weights[f"{prefix}.self_attn.k_proj.weight"].ptr,
                self.weights[f"{prefix}.self_attn.v_proj.weight"].ptr,
                self.tensor("query_row").ptr,
                self.tensor("key_row").ptr,
                self.tensor("value_row").ptr,
                frames,
                hidden,
                hidden,
                hidden,
                hidden,
                library=libraries.projection,
                **common,
            )
            for row_name, head_name in (
                ("query_row", "query"),
                ("key_row", "key"),
                ("value_row", "value"),
            ):
                moonshine_encoder_transpose_head_major_fp16(
                    self.tensor(row_name).ptr,
                    self.tensor(head_name).ptr,
                    frames,
                    heads,
                    head_dim,
                    library=libraries.encoder,
                    **common,
                )
            moonshine_encoder_rope_fp16(
                self.tensor("query").ptr,
                self.tensor("key").ptr,
                cos,
                sin,
                self.tensor("query").ptr,
                self.tensor("key").ptr,
                heads,
                frames,
                head_dim,
                spec.rotary_dim,
                spec.max_positions,
                library=libraries.encoder,
                **common,
            )
            moonshine_encoder_attention_fp16(
                self.tensor("query").ptr,
                self.tensor("key").ptr,
                self.tensor("value").ptr,
                mask,
                self.tensor("attention").ptr,
                heads,
                head_dim,
                frames,
                scale=scale,
                library=libraries.encoder,
                **common,
            )
            moonshine_f16_projection(
                self.tensor("attention").ptr,
                self.weights[f"{prefix}.self_attn.o_proj.weight"].ptr,
                self.tensor("projection").ptr,
                frames,
                hidden,
                hidden,
                library=libraries.projection,
                **common,
            )
            moonshine_residual_layernorm_fp16(
                self.tensor("hidden").ptr,
                self.tensor("projection").ptr,
                self.weights[f"{prefix}.post_attention_layernorm.weight"].ptr,
                self.tensor("hidden").ptr,
                self.tensor("normalized").ptr,
                frames,
                hidden,
                eps=spec.layer_norm_epsilon,
                library=libraries.layernorm,
                **common,
            )
            moonshine_f16_projection_bias(
                self.tensor("normalized").ptr,
                self.weights[f"{prefix}.mlp.fc1.weight"].ptr,
                self.weights[f"{prefix}.mlp.fc1.bias"].ptr,
                self.tensor("mlp_fc1").ptr,
                frames,
                hidden,
                intermediate,
                library=libraries.projection,
                **common,
            )
            moonshine_gelu_fp16(
                self.tensor("mlp_fc1").ptr,
                self.tensor("mlp_gelu").ptr,
                frames * intermediate,
                library=libraries.encoder,
                **common,
            )
            moonshine_f16_projection_bias_residual(
                self.tensor("mlp_gelu").ptr,
                self.weights[f"{prefix}.mlp.fc2.weight"].ptr,
                self.weights[f"{prefix}.mlp.fc2.bias"].ptr,
                self.tensor("hidden").ptr,
                self.tensor("hidden").ptr,
                frames,
                intermediate,
                hidden,
                library=libraries.projection,
                **common,
            )
        moonshine_layernorm_fp16(
            self.tensor("hidden").ptr,
            self.weights["model.encoder.layer_norm.weight"].ptr,
            self.tensor("encoder_output").ptr,
            frames,
            hidden,
            eps=spec.layer_norm_epsilon,
            library=libraries.layernorm,
            **common,
        )

    def handoff_to(self, decoder: "object") -> None:
        """Hand the resident hidden + mask to a ``MoonshineCudaResidentRuntime``.

        Copies the finished encoder output into the decoder's fixed padded
        bucket (D2D) and precomputes all eight head-major cross K/V caches so
        the decoder is ready to decode from position 0.
        """

        from hipengine.runtime.moonshine_cuda import MoonshineCudaResidentRuntime

        if not isinstance(decoder, MoonshineCudaResidentRuntime):
            raise TypeError("decoder must be a MoonshineCudaResidentRuntime")
        decoder.set_encoder_state_from_device(
            hidden_fp16_ptr=self.tensor("encoder_output").ptr,
            attention_mask_int32_ptr=self.tensor("encoder_attention_mask").ptr,
            source_frames=self.encoder_frames,
        )
        decoder.precompute_cross_kv()

    def close(self) -> None:
        """Free workspace, weights, and stream, and report teardown parity."""

        if self.closed:
            return
        self.closed = True
        try:
            if self.workspace is not None:
                self.workspace.free()
            if self.owns_weights and self.loaded_model is not None and self.weights is not None:
                self.loaded_model.weights.free(runtime=self.runtime)
            if self.stream:
                self.runtime.stream_destroy(self.stream)
                self.stream = 0
        finally:
            after = memory_stats()["current_allocated_bytes"]
            self.teardown_returned_to_baseline = after <= self._allocation_baseline

    def __enter__(self) -> "MoonshineCudaEncoderRuntime":
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        self.close()
