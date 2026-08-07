"""Torch-free fixed-address Moonshine FP16 batch encoder runtime for CUDA ``sm_120a`` (C8 phase 2).

A static homogeneous-B extension of ``MoonshineCudaEncoderRuntime``: every
workspace object is batch-strided ``[B, ...]`` at a fixed address and every row
processes the same audio length (``audio_samples``), so the batch encoder is
bit-exact against B independent batch-one encoder sessions.  The conv front end
(conv1+tanh, GroupNorm, conv2/conv3+gelu), the head-major transpose, the
full-sequence partial RoPE, and the non-causal self-attention each gained a
batch-plane grid dimension; LayerNorm, the QKV/o/fc1/fc2 projections, and the
exact-erf GELU reuse the row-generic kernels at ``rows = B * frames``.

``handoff_to`` projects all B rows into a ``MoonshineCudaBatchRuntime``'s
batch-strided cross cache on device (zeroing the bucket tail) and copies the
downsampled per-row int32 masks, closing the torch-free batch encoder -> batch
decoder path without any host-side cross-cache round trip.
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
from hipengine.runtime.moonshine_encoder_cuda import (
    _CONV1_KERNEL,
    _CONV1_STRIDE,
    _CONV2_KERNEL,
    _CONV2_STRIDE,
    _CONV3_KERNEL,
    _CONV3_STRIDE,
    _DOWNSAMPLE_STRIDE,
    _ENCODER_HEAD_DIM,
    _ENCODER_HEADS,
    _GROUPNORM_EPS,
    MoonshineCudaEncoderLibraries,
    moonshine_encoder_frames_from_audio,
)
from hipengine.runtime.workspace import RuntimeWorkspace

# Re-exported production encoder-frame buckets for callers that size the batch
# decoder from the same shared length.
MOONSHINE_CUDA_ENC_BUCKETS = (40, 207, 1248)


@dataclass
class MoonshineCudaBatchEncoderChainGraph:
    """One captured fixed-address batch encoder + handoff + cross-KV graph.

    Mirrors ``MoonshineCudaBatchTokenGraph`` on the encoder side of the
    static-B chain: the capture records the decoder's fresh-generation memsets
    (self/cross KV + mask), the full batch encoder DAG (conv front end, eight
    encoder layers, final LayerNorm), and the batch cross-KV handoff
    projection + mask D2D copy on the decoder's stream, so one
    :meth:`~MoonshineCudaBatchEncoderRuntime.graph_encode_and_handoff` replay
    replaces the per-request Python dispatch of ~101 encoder kernels plus the
    eight cross-KV projections and the mask copy.  The graph is bound to this
    runtime's fixed ``(max_batch, audio_samples)`` encoder bucket.
    """

    owner: "MoonshineCudaBatchEncoderRuntime"
    graph: int
    graph_exec: int
    capture_wall_ms: float
    instantiate_wall_ms: float
    replay_count: int = 0
    closed: bool = False

    def close(self) -> None:
        """Destroy the graph and detach it from the owner runtime."""

        if self.closed:
            return
        self.closed = True
        self.owner.runtime.graph_exec_destroy(self.graph_exec)
        self.owner.runtime.graph_destroy(self.graph)
        if self.owner._enc_chain_graph is self:
            self.owner._enc_chain_graph = None


class MoonshineCudaBatchEncoderRuntime:
    """Own every fixed-address object needed to encode B Moonshine audio rows.

    All B rows share one ``audio_samples`` length (static homogeneous B).  The
    buffers are exact-size for that length, so every batch kernel's plane
    stride equals its process length and the composition is bit-exact against
    B independent ``MoonshineCudaEncoderRuntime`` sessions.
    """

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
        max_batch: int,
        audio_samples: int,
        model_path: str | Path | None = None,
        loaded_model: MoonshineLoadedModel | None = None,
        device: Device | None = None,
        runtime: CudaRuntime | None = None,
        owns_weights: bool = True,
        projection_route: str = "custom",
        long_bucket_gemm_rows: int = 768,
        conv_route: str = "custom",
        long_bucket_conv_frames: int = 768,
    ) -> None:
        if (model_path is None) == (loaded_model is None):
            raise ValueError("provide exactly one of model_path or loaded_model")
        if isinstance(max_batch, bool) or not isinstance(max_batch, int):
            raise ValueError("max_batch must be a positive integer")
        if max_batch <= 0:
            raise ValueError("max_batch must be a positive integer")
        if projection_route not in ("custom", "cublaslt"):
            raise ValueError("projection_route must be 'custom' or 'cublaslt'")
        if conv_route not in ("custom", "cudnn"):
            raise ValueError("conv_route must be 'custom' or 'cudnn'")
        if isinstance(long_bucket_gemm_rows, bool) or not isinstance(long_bucket_gemm_rows, int):
            raise ValueError("long_bucket_gemm_rows must be a positive integer")
        if long_bucket_gemm_rows <= 0:
            raise ValueError("long_bucket_gemm_rows must be a positive integer")
        if isinstance(long_bucket_conv_frames, bool) or not isinstance(long_bucket_conv_frames, int):
            raise ValueError("long_bucket_conv_frames must be a positive integer")
        if long_bucket_conv_frames <= 0:
            raise ValueError("long_bucket_conv_frames must be a positive integer")
        self.runtime = runtime or get_cuda_runtime()
        self.device = device or Device("cuda", 0)
        self.loaded_model = loaded_model
        self.weights = loaded_model.weights if loaded_model is not None else None
        self.spec = loaded_model.spec if loaded_model is not None else None
        self.owns_weights = bool(owns_weights)
        self.max_batch = int(max_batch)
        self.audio_samples = int(audio_samples)
        self.projection_route = str(projection_route)
        self.long_bucket_gemm_rows = int(long_bucket_gemm_rows)
        self.conv_route = str(conv_route)
        self.long_bucket_conv_frames = int(long_bucket_conv_frames)
        self.encoder_frames = moonshine_encoder_frames_from_audio(self.audio_samples)
        self.workspace = RuntimeWorkspace(device=self.device, runtime=self.runtime)
        self.stream = 0
        self.encoder_libraries: MoonshineCudaEncoderLibraries | None = None
        self.cublaslt: object | None = None
        self._lt_problems: dict | None = None
        self._lt_epilogue_library: object | None = None
        self.cudnn: object | None = None
        self._conv_problems: dict | None = None
        self._conv_lengths: dict | None = None
        self._cudnn_epilogue_library: object | None = None
        self._enc_chain_graph: MoonshineCudaBatchEncoderChainGraph | None = None
        self.closed = False
        self.teardown_returned_to_baseline: bool | None = None
        self._input_uploaded = False
        self._last_real_frames: int | None = None
        self._last_real_samples: int | None = None
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
        batch = self.max_batch
        frames = self.encoder_frames
        hidden = spec.hidden_size
        intermediate = spec.intermediate_size
        reserve = self.workspace.reserve_tensor
        reserve("rope_cos", (frames, spec.rotary_dim // 2), DType.FP16)
        reserve("rope_sin", (frames, spec.rotary_dim // 2), DType.FP16)
        reserve("audio", (batch, self.audio_samples), DType.FP16)
        length = (self.audio_samples - _CONV1_KERNEL) // _CONV1_STRIDE + 1
        reserve("conv1_out", (batch, hidden, length), DType.FP16)
        length = (length - _CONV2_KERNEL) // _CONV2_STRIDE + 1
        reserve("conv2_out", (batch, 2 * hidden, length), DType.FP16)
        reserve("groupnorm_partial", (batch, 2 * hidden, 2), DType.FP32)
        reserve("groupnorm_mean_rstd", (batch, 2), DType.FP32)
        reserve("hidden", (batch, frames, hidden), DType.FP16)
        reserve("normalized", (batch, frames, hidden), DType.FP16)
        for name in ("query_row", "key_row", "value_row"):
            reserve(name, (batch, frames, hidden), DType.FP16)
        for name in ("query", "key", "value"):
            reserve(name, (batch, _ENCODER_HEADS, frames, spec.head_dim), DType.FP16)
        reserve("attention", (batch, frames, hidden), DType.FP16)
        reserve("projection", (batch, frames, hidden), DType.FP16)
        reserve("mlp_fc1", (batch, frames, intermediate), DType.FP16)
        reserve("mlp_gelu", (batch, frames, intermediate), DType.FP16)
        reserve("encoder_output", (batch, frames, hidden), DType.FP16)
        reserve("encoder_attention_mask", (batch, frames), DType.INT32)
        if self.projection_route == "cublaslt":
            # FP32 GEMM boundary for the fc1/fc2 bias/residual epilogues.
            reserve("gemm_f32", (batch, frames, intermediate), DType.FP32)
        if self.conv_route == "cudnn":
            # cuDNN conv3 outputs NCHW [B, 416, frames]; the row-major epilogue
            # transposes it into the encoder "hidden" [B, frames, 416] layout.
            reserve("conv3_out", (batch, hidden, frames), DType.FP16)

    def _initialize_workspace(self) -> None:
        assert self.spec is not None
        for name in self.workspace.names:
            allocation = self.workspace.allocation(name)
            self.runtime.memset_async(
                allocation.buffer.ptr, 0, allocation.buffer.nbytes, self.stream
            )
        self.runtime.stream_synchronize(self.stream)
        cos, sin = moonshine_rope_tables(
            self.encoder_frames,
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
            raise RuntimeError("Moonshine batch encoder runtime is closed")
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
        self._prepare_long_bucket_gemm()
        self._prepare_long_bucket_conv()
        if self._use_cublaslt():
            from hipengine.kernels.cuda_sm120a.encoder.moonshine_encoder_lt import (
                build_moonshine_encoder_lt,
            )

            self._lt_epilogue_library = build_moonshine_encoder_lt(load=True)
        if self._use_cudnn_conv():
            from hipengine.kernels.cuda_sm120a.encoder.moonshine_encoder_cudnn import (
                build_moonshine_encoder_cudnn,
            )

            self._cudnn_epilogue_library = build_moonshine_encoder_cudnn(load=True)
        return libraries

    def _use_cublaslt(self) -> bool:
        """True when the long-bucket cuBLASLt route is armed for this batch."""

        return (
            self.projection_route == "cublaslt"
            and self.cublaslt is not None
            and self.max_batch * self.encoder_frames >= self.long_bucket_gemm_rows
        )

    def _use_cudnn_conv(self) -> bool:
        """True when the long-bucket cuDNN conv route is armed for this batch."""

        return (
            self.conv_route == "cudnn"
            and self.cudnn is not None
            and self.encoder_frames >= self.long_bucket_conv_frames
        )

    def _prepare_long_bucket_conv(self) -> None:
        """Create the three fixed-shape cuDNN conv problems (outside timing)."""

        if self.conv_route != "cudnn":
            return
        if self.spec is None or self.weights is None:
            return
        if self.encoder_frames < self.long_bucket_conv_frames:
            return
        from hipengine.core.cudnn import Cudnn

        batch = self.max_batch
        hidden = self.spec.hidden_size
        conv1_len = (self.audio_samples - _CONV1_KERNEL) // _CONV1_STRIDE + 1
        conv2_len = (conv1_len - _CONV2_KERNEL) // _CONV2_STRIDE + 1
        conv3_len = (conv2_len - _CONV3_KERNEL) // _CONV3_STRIDE + 1
        assert conv3_len == self.encoder_frames
        owner = Cudnn(runtime=self.runtime)
        try:
            convs = {
                "conv1": owner.conv(
                    batch, 1, hidden, _CONV1_KERNEL, _CONV1_STRIDE,
                    self.audio_samples,
                ),
                "conv2": owner.conv(
                    batch, hidden, 2 * hidden, _CONV2_KERNEL, _CONV2_STRIDE,
                    conv1_len,
                ),
                "conv3": owner.conv(
                    batch, 2 * hidden, hidden, _CONV3_KERNEL, _CONV3_STRIDE,
                    conv2_len,
                ),
            }
        except Exception:
            owner.close()
            raise
        self.cudnn = owner
        self._conv_problems = convs
        self._conv_lengths = {
            "conv1": conv1_len,
            "conv2": conv2_len,
            "conv3": conv3_len,
        }

    def _enqueue_conv_frontend_cudnn(
        self, *, stream: int, conv1_len: int, conv2_len: int
    ) -> None:
        """cuDNN conv front end + retained FP16-rounding activation epilogues.

        conv1+tanh, GroupNorm, conv2+gelu, conv3+gelu-with-transpose into the
        encoder ``hidden`` layout, using the fixed cuDNN problems created by
        :meth:`_prepare_long_bucket_conv` and the element-wise epilogue
        kernels of ``moonshine_encoder_cudnn``.  The only divergence from the
        exact custom route is the FP32-reassociation/ULP rounding of the
        cuDNN conv output before the epilogue (the accepted C8 re-derived
        numerical gate).
        """

        if self.spec is None or self.weights is None or self.cudnn is None:
            raise RuntimeError("Moonshine batch encoder runtime is closed")
        from hipengine.kernels.cuda_sm120a.encoder.moonshine_encoder import (
            moonshine_groupnorm_batch_fp16,
        )
        from hipengine.kernels.cuda_sm120a.encoder.moonshine_encoder_cudnn import (
            moonshine_bias_gelu_apply_fp16,
            moonshine_bias_gelu_apply_rowmajor_fp16,
            moonshine_tanh_apply_fp16,
        )

        batch = self.max_batch
        hidden = self.spec.hidden_size
        convs = self._conv_problems
        assert convs is not None
        lib = self._cudnn_epilogue_library
        if lib is None:
            raise RuntimeError("Moonshine batch encoder cuDNN epilogues are not prepared")
        common = {"stream": stream, "runtime": self.runtime}
        epilogue = {"stream": stream, "library": lib, "runtime": self.runtime}
        weights = self.weights
        # conv1: [B, 1, samples] -> [B, hidden, conv1_len] fp16 output, then
        # fp16(tanhf(fp32(in))) in place on conv1_out.
        convs["conv1"].forward(
            self.tensor("audio").ptr,
            weights["model.encoder.conv1.weight"].ptr,
            self.tensor("conv1_out").ptr,
            workspace_ptr=0,
            stream=stream,
        )
        moonshine_tanh_apply_fp16(
            self.tensor("conv1_out").ptr,
            self.tensor("conv1_out").ptr,
            batch * hidden * conv1_len,
            **epilogue,
        )
        moonshine_groupnorm_batch_fp16(
            self.tensor("conv1_out").ptr,
            weights["model.encoder.groupnorm.weight"].ptr,
            weights["model.encoder.groupnorm.bias"].ptr,
            self.tensor("conv1_out").ptr,
            self.tensor("groupnorm_partial").ptr,
            self.tensor("groupnorm_mean_rstd").ptr,
            batch,
            hidden,
            conv1_len,
            eps=_GROUPNORM_EPS,
            library=self.encoder_libraries.encoder,
            **common,
        )
        # conv2: [B, hidden, conv1_len] -> [B, 2*hidden, conv2_len] fp16
        # output, then bias+gelu in place on conv2_out.
        convs["conv2"].forward(
            self.tensor("conv1_out").ptr,
            weights["model.encoder.conv2.weight"].ptr,
            self.tensor("conv2_out").ptr,
            workspace_ptr=0,
            stream=stream,
        )
        moonshine_bias_gelu_apply_fp16(
            self.tensor("conv2_out").ptr,
            weights["model.encoder.conv2.bias"].ptr,
            self.tensor("conv2_out").ptr,
            batch * 2 * hidden * conv2_len,
            2 * hidden,
            conv2_len,
            **epilogue,
        )
        # conv3: [B, 2*hidden, conv2_len] -> NCHW fp16 scratch, then row-major
        # bias+gelu into the encoder "hidden" [B, frames, hidden] layout.
        convs["conv3"].forward(
            self.tensor("conv2_out").ptr,
            weights["model.encoder.conv3.weight"].ptr,
            self.tensor("conv3_out").ptr,
            workspace_ptr=0,
            stream=stream,
        )
        moonshine_bias_gelu_apply_rowmajor_fp16(
            self.tensor("conv3_out").ptr,
            weights["model.encoder.conv3.bias"].ptr,
            self.tensor("hidden").ptr,
            batch,
            hidden,
            self.encoder_frames,
            **epilogue,
        )

    def _prepare_long_bucket_gemm(self) -> None:
        """Create the per-shape cuBLASLt problems (outside any timed region)."""

        if self.projection_route != "cublaslt":
            return
        if self.spec is None or self.weights is None:
            return
        rows = self.max_batch * self.encoder_frames
        if rows < self.long_bucket_gemm_rows:
            return
        from hipengine.core.cublaslt import CUDA_R_16F, CUDA_R_32F, CublasLt

        owner = CublasLt(runtime=self.runtime)
        hidden = self.spec.hidden_size
        intermediate = self.spec.intermediate_size
        try:
            problems = {
                "qkv": [
                    owner.problem(rows, hidden, hidden, output_dtype=CUDA_R_16F)
                    for _ in range(3)
                ],
                "output": owner.problem(
                    rows, hidden, hidden, output_dtype=CUDA_R_16F
                ),
                "fc1": owner.problem(
                    rows, hidden, intermediate, output_dtype=CUDA_R_32F
                ),
                "fc2": owner.problem(
                    rows, intermediate, hidden, output_dtype=CUDA_R_32F
                ),
            }
        except Exception:
            owner.close()
            raise
        self.cublaslt = owner
        self._lt_problems = problems

    def _enqueue_lt_qkv(self, *, layer: int, stream: int) -> None:
        """Enqueue the QKV projections as three FP16 cuBLASLt GEMMs."""

        if self.spec is None or self.weights is None or self.cublaslt is None:
            raise RuntimeError("Moonshine batch encoder runtime is closed")
        problems = self._lt_problems["qkv"]
        weight = self.weights
        prefix = f"model.encoder.layers.{layer}"
        for index, name in enumerate(("q_proj", "k_proj", "v_proj")):
            problems[index].launch(
                self.tensor("normalized").ptr,
                weight[f"{prefix}.self_attn.{name}.weight"].ptr,
                self.tensor(("query_row", "key_row", "value_row")[index]).ptr,
                stream=stream,
            )

    def _enqueue_lt_output(self, *, layer: int, stream: int) -> None:
        """Enqueue the output projection as one FP16 cuBLASLt GEMM."""

        if self.spec is None or self.weights is None or self.cublaslt is None:
            raise RuntimeError("Moonshine batch encoder runtime is closed")
        prefix = f"model.encoder.layers.{layer}"
        self._lt_problems["output"].launch(
            self.tensor("attention").ptr,
            self.weights[f"{prefix}.self_attn.o_proj.weight"].ptr,
            self.tensor("projection").ptr,
            stream=stream,
        )

    def _enqueue_lt_fc1(self, *, layer: int, stream: int, rows: int) -> None:
        """Enqueue fc1 as an FP32 GEMM + the retained FP16 bias epilogue."""

        if self.spec is None or self.weights is None or self.cublaslt is None:
            raise RuntimeError("Moonshine batch encoder runtime is closed")
        from hipengine.kernels.cuda_sm120a.encoder.moonshine_encoder_lt import (
            moonshine_f16_bias_round_fp32,
        )

        prefix = f"model.encoder.layers.{layer}"
        intermediate = self.spec.intermediate_size
        self._lt_problems["fc1"].launch(
            self.tensor("normalized").ptr,
            self.weights[f"{prefix}.mlp.fc1.weight"].ptr,
            self.tensor("gemm_f32").ptr,
            stream=stream,
        )
        moonshine_f16_bias_round_fp32(
            self.tensor("gemm_f32").ptr,
            self.weights[f"{prefix}.mlp.fc1.bias"].ptr,
            self.tensor("mlp_fc1").ptr,
            rows,
            intermediate,
            library=self._lt_epilogue_library,
            stream=stream,
        )

    def _enqueue_lt_fc2(self, *, layer: int, stream: int, rows: int) -> None:
        """Enqueue fc2 as an FP32 GEMM + the retained FP16 bias/residual epilogue."""

        if self.spec is None or self.weights is None or self.cublaslt is None:
            raise RuntimeError("Moonshine batch encoder runtime is closed")
        from hipengine.kernels.cuda_sm120a.encoder.moonshine_encoder_lt import (
            moonshine_f16_bias_residual_round_fp32,
        )

        prefix = f"model.encoder.layers.{layer}"
        hidden = self.spec.hidden_size
        self._lt_problems["fc2"].launch(
            self.tensor("mlp_gelu").ptr,
            self.weights[f"{prefix}.mlp.fc2.weight"].ptr,
            self.tensor("gemm_f32").ptr,
            stream=stream,
        )
        moonshine_f16_bias_residual_round_fp32(
            self.tensor("gemm_f32").ptr,
            self.weights[f"{prefix}.mlp.fc2.bias"].ptr,
            self.tensor("hidden").ptr,
            self.tensor("hidden").ptr,
            rows,
            hidden,
            library=self._lt_epilogue_library,
            stream=stream,
        )

    def tensor(self, name: str) -> Tensor:
        return self.workspace.allocation(name).tensor

    def encoder_output(self) -> Tensor:
        """The finished row-major ``[B, frames, hidden]`` FP16 batch encoder hidden state."""

        return self.tensor("encoder_output")

    def attention_mask(self) -> Tensor:
        """The downsampled int32 ``[B, frames]`` batch encoder attention mask."""

        return self.tensor("encoder_attention_mask")

    @property
    def real_frames(self) -> int:
        """The frame count of the last uploaded audio (== ``encoder_frames``)."""

        if self._last_real_frames is None:
            raise RuntimeError("Moonshine batch encoder input is not uploaded")
        return self._last_real_frames

    def upload_input(
        self,
        input_values: np.ndarray,
        attention_mask: np.ndarray | None = None,
    ) -> None:
        """Upload B fixed-length audio planes and the per-row masks.

        ``input_values`` must be a finite ``[B, real_samples]`` float32 (or
        float16) array with ``real_samples == audio_samples`` (static
        homogeneous B; every row processes the same length so GroupNorm
        statistics and every batch-plane kernel are bit-exact against B
        independent batch-one sessions).  ``attention_mask`` (optional) is the
        ``[B, real_samples]`` int64/int32 mask.  After upload the batch is
        resident and :meth:`run_encode` can run the fixed-address DAG.
        """

        if self.closed:
            raise RuntimeError("Moonshine batch encoder runtime is closed")
        audio = np.asarray(input_values)
        if audio.ndim != 2 or audio.shape[0] != self.max_batch:
            raise ValueError(
                f"input_values must have shape ({self.max_batch}, real_samples)"
            )
        real_samples = int(audio.shape[1])
        if real_samples != self.audio_samples:
            raise ValueError(
                f"real_samples {real_samples} must equal audio_samples "
                f"{self.audio_samples} (static homogeneous batch encoder)"
            )
        try:
            real_frames = moonshine_encoder_frames_from_audio(real_samples)
        except ValueError as error:
            raise ValueError(f"audio too short for the encoder bucket: {error}") from error
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
            nbytes=values.nbytes,
            runtime=self.runtime,
        )
        self._upload_mask(attention_mask, real_samples=real_samples)
        self._last_real_frames = real_frames
        self._last_real_samples = real_samples
        self._input_uploaded = True

    def _upload_mask(
        self,
        attention_mask: np.ndarray | None,
        *,
        real_samples: int,
    ) -> None:
        if attention_mask is None:
            mask_values = np.ones(
                (self.max_batch, real_samples), dtype=np.int64
            )
        else:
            mask_values = np.asarray(attention_mask)
            if mask_values.ndim != 2 or mask_values.shape[0] != self.max_batch:
                raise ValueError(
                    f"attention_mask must have shape ({self.max_batch}, {real_samples})"
                )
        if not bool(((mask_values == 0) | (mask_values == 1)).all()):
            raise ValueError("attention_mask must be binary")
        real_frames = moonshine_encoder_frames_from_audio(real_samples)
        downsampled = (
            mask_values[..., ::_DOWNSAMPLE_STRIDE][..., :real_frames]
            .astype(np.int32)
            .reshape(self.max_batch, -1)
        )
        output = np.zeros((self.max_batch, self.encoder_frames), dtype=np.int32)
        output[:, :real_frames] = downsampled
        copy_host_to_device(
            self.workspace.allocation("encoder_attention_mask").buffer,
            host_array_ptr(output),
            runtime=self.runtime,
        )

    def run_encode(
        self,
        *,
        stream: int | None = None,
        synchronize: bool = True,
    ) -> None:
        """Run the fixed-address batch encoder DAG (optionally without a terminal sync)."""

        if not self._input_uploaded:
            raise RuntimeError("Moonshine batch encoder input is not uploaded")
        use_stream = self.stream if stream is None else int(stream)
        self._enqueue_encode(stream=use_stream)
        if synchronize:
            self.runtime.stream_synchronize(use_stream)

    def encode(
        self,
        input_values: np.ndarray,
        attention_mask: np.ndarray | None = None,
        *,
        stream: int | None = None,
    ) -> None:
        """Upload B audio planes and run the full fixed-address batch encoder DAG."""

        self.upload_input(input_values, attention_mask)
        self.run_encode(stream=stream)

    def _enqueue_encode(self, *, stream: int) -> None:
        """Enqueue the fixed-address batch encoder DAG without changing host state."""

        if self.closed or self.spec is None or self.weights is None:
            raise RuntimeError("Moonshine batch encoder runtime is closed")
        libraries = self.encoder_libraries
        if libraries is None:
            raise RuntimeError("Moonshine batch encoder kernels are not prepared")
        from hipengine.kernels.cuda_sm120a.encoder.moonshine_encoder import (
            moonshine_conv1_tanh_batch_fp16,
            moonshine_conv2_gelu_batch_fp16,
            moonshine_conv3_gelu_batch_fp16,
            moonshine_encoder_attention_batch_fp16,
            moonshine_encoder_rope_batch_fp16,
            moonshine_encoder_transpose_head_major_batch_fp16,
            moonshine_gelu_fp16,
            moonshine_groupnorm_batch_fp16,
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
        batch = self.max_batch
        real_samples = self._last_real_samples
        frames = self._last_real_frames
        hidden = spec.hidden_size
        intermediate = spec.intermediate_size
        heads = _ENCODER_HEADS
        head_dim = spec.head_dim
        common = {"stream": stream, "runtime": self.runtime}

        # ---- conv front end -------------------------------------------------
        length = (real_samples - _CONV1_KERNEL) // _CONV1_STRIDE + 1
        conv2_length = (length - _CONV2_KERNEL) // _CONV2_STRIDE + 1
        if self._use_cudnn_conv():
            self._enqueue_conv_frontend_cudnn(
                stream=stream,
                conv1_len=length,
                conv2_len=conv2_length,
            )
        else:
            moonshine_conv1_tanh_batch_fp16(
                self.tensor("audio").ptr,
                self.weights["model.encoder.conv1.weight"].ptr,
                self.tensor("conv1_out").ptr,
                batch,
                real_samples,
                length,
                library=libraries.encoder,
                **common,
            )
            moonshine_groupnorm_batch_fp16(
                self.tensor("conv1_out").ptr,
                self.weights["model.encoder.groupnorm.weight"].ptr,
                self.weights["model.encoder.groupnorm.bias"].ptr,
                self.tensor("conv1_out").ptr,
                self.tensor("groupnorm_partial").ptr,
                self.tensor("groupnorm_mean_rstd").ptr,
                batch,
                hidden,
                length,
                eps=_GROUPNORM_EPS,
                library=libraries.encoder,
                **common,
            )
            moonshine_conv2_gelu_batch_fp16(
                self.tensor("conv1_out").ptr,
                self.weights["model.encoder.conv2.weight"].ptr,
                self.weights["model.encoder.conv2.bias"].ptr,
                self.tensor("conv2_out").ptr,
                batch,
                length,
                conv2_length,
                library=libraries.encoder,
                **common,
            )
        conv3_length = (conv2_length - _CONV3_KERNEL) // _CONV3_STRIDE + 1
        assert conv3_length == frames
        if self._use_cudnn_conv():
            # The cuDNN branch already wrote the encoder ``hidden`` tensor.
            pass
        else:
            moonshine_conv3_gelu_batch_fp16(
                self.tensor("conv2_out").ptr,
                self.weights["model.encoder.conv3.weight"].ptr,
                self.weights["model.encoder.conv3.bias"].ptr,
                self.tensor("hidden").ptr,
                batch,
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
        total_rows = batch * frames
        use_long_bucket_gemm = self._use_cublaslt()
        for layer in range(spec.encoder_layers):
            prefix = f"model.encoder.layers.{layer}"
            moonshine_layernorm_fp16(
                self.tensor("hidden").ptr,
                self.weights[f"{prefix}.input_layernorm.weight"].ptr,
                self.tensor("normalized").ptr,
                total_rows,
                hidden,
                eps=spec.layer_norm_epsilon,
                library=libraries.layernorm,
                **common,
            )
            if use_long_bucket_gemm:
                self._enqueue_lt_qkv(layer=layer, stream=stream)
            else:
                moonshine_f16_projection_triple(
                    self.tensor("normalized").ptr,
                    self.weights[f"{prefix}.self_attn.q_proj.weight"].ptr,
                    self.weights[f"{prefix}.self_attn.k_proj.weight"].ptr,
                    self.weights[f"{prefix}.self_attn.v_proj.weight"].ptr,
                    self.tensor("query_row").ptr,
                    self.tensor("key_row").ptr,
                    self.tensor("value_row").ptr,
                    total_rows,
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
                moonshine_encoder_transpose_head_major_batch_fp16(
                    self.tensor(row_name).ptr,
                    self.tensor(head_name).ptr,
                    batch,
                    frames,
                    heads,
                    head_dim,
                    library=libraries.encoder,
                    **common,
                )
            moonshine_encoder_rope_batch_fp16(
                self.tensor("query").ptr,
                self.tensor("key").ptr,
                cos,
                sin,
                self.tensor("query").ptr,
                self.tensor("key").ptr,
                batch,
                heads,
                frames,
                head_dim,
                spec.rotary_dim,
                self.encoder_frames,
                library=libraries.encoder,
                **common,
            )
            moonshine_encoder_attention_batch_fp16(
                self.tensor("query").ptr,
                self.tensor("key").ptr,
                self.tensor("value").ptr,
                mask,
                self.tensor("attention").ptr,
                batch,
                heads,
                head_dim,
                frames,
                scale=scale,
                library=libraries.encoder,
                **common,
            )
            if use_long_bucket_gemm:
                self._enqueue_lt_output(layer=layer, stream=stream)
            else:
                moonshine_f16_projection(
                    self.tensor("attention").ptr,
                    self.weights[f"{prefix}.self_attn.o_proj.weight"].ptr,
                    self.tensor("projection").ptr,
                    total_rows,
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
                total_rows,
                hidden,
                eps=spec.layer_norm_epsilon,
                library=libraries.layernorm,
                **common,
            )
            if use_long_bucket_gemm:
                self._enqueue_lt_fc1(layer=layer, stream=stream, rows=total_rows)
            else:
                moonshine_f16_projection_bias(
                    self.tensor("normalized").ptr,
                    self.weights[f"{prefix}.mlp.fc1.weight"].ptr,
                    self.weights[f"{prefix}.mlp.fc1.bias"].ptr,
                    self.tensor("mlp_fc1").ptr,
                    total_rows,
                    hidden,
                    intermediate,
                    library=libraries.projection,
                    **common,
                )
            moonshine_gelu_fp16(
                self.tensor("mlp_fc1").ptr,
                self.tensor("mlp_gelu").ptr,
                total_rows * intermediate,
                library=libraries.encoder,
                **common,
            )
            if use_long_bucket_gemm:
                self._enqueue_lt_fc2(layer=layer, stream=stream, rows=total_rows)
            else:
                moonshine_f16_projection_bias_residual(
                    self.tensor("mlp_gelu").ptr,
                    self.weights[f"{prefix}.mlp.fc2.weight"].ptr,
                    self.weights[f"{prefix}.mlp.fc2.bias"].ptr,
                    self.tensor("hidden").ptr,
                    self.tensor("hidden").ptr,
                    total_rows,
                    intermediate,
                    hidden,
                    library=libraries.projection,
                    **common,
                )
        moonshine_layernorm_fp16(
            self.tensor("hidden").ptr,
            self.weights["model.encoder.layer_norm.weight"].ptr,
            self.tensor("encoder_output").ptr,
            total_rows,
            hidden,
            eps=spec.layer_norm_epsilon,
            library=libraries.layernorm,
            **common,
        )

    def capture_encoder_chain(
        self, decoder: object
    ) -> MoonshineCudaBatchEncoderChainGraph:
        """Capture the batch encoder DAG + reset + cross-KV handoff as one graph.

        Requires a prepared ``MoonshineCudaBatchRuntime`` decoder and a
        resident input (``upload_input``).  The capture runs on the decoder's
        stream and records, in order: the decoder's fresh-generation memsets
        (self/cross KV + mask), the full batch encoder DAG, and the cross-KV
        handoff projection + mask D2D copy.  Each replay of the resulting
        graph is therefore a complete, self-contained new-generation
        encode+handoff (no host reset or per-request kernel dispatch), and the
        decoder is left with a valid cross cache ready to decode from position
        0.  The graph is bound to this runtime's fixed
        ``(max_batch, audio_samples)`` encoder bucket; capture twice returns
        the existing capture.  Capturing an already-armed cuBLASLt route
        records the ``cublasLtMatmul`` calls into the same graph (they enqueue
        on the capturing stream).
        """

        import time

        from hipengine.runtime.moonshine_cuda_batch import MoonshineCudaBatchRuntime

        if not isinstance(decoder, MoonshineCudaBatchRuntime):
            raise TypeError("decoder must be a MoonshineCudaBatchRuntime")
        if self.closed or self.spec is None or self.weights is None:
            raise RuntimeError("Moonshine batch encoder runtime is closed")
        if self.encoder_libraries is None:
            raise RuntimeError("Moonshine batch encoder kernels are not prepared")
        if not self._input_uploaded:
            raise RuntimeError("Moonshine batch encoder input is not uploaded")
        if decoder.decoder_libraries is None:
            raise RuntimeError("Moonshine batch decoder kernels are not prepared")
        existing = self._enc_chain_graph
        if existing is not None and not existing.closed:
            return existing

        stream = decoder.stream
        # Both sides must be idle before capture records a stable DAG.
        self.runtime.stream_synchronize(self.stream)
        self.runtime.stream_synchronize(stream)
        # Host-side fresh generation; the device memsets run inside the capture.
        decoder._reset_generation_host_state(clear_cross_cache=True)
        graph = 0
        capture_start = time.perf_counter_ns()
        self.runtime.stream_begin_capture(stream)
        try:
            decoder._enqueue_generation_reset(
                stream=stream,
                clear_cross_cache=True,
            )
            self._enqueue_encode(stream=stream)
            decoder._enqueue_cross_kv_handoff(
                encoder_hidden_ptr=self.tensor("encoder_output").ptr,
                attention_mask_ptr=self.tensor("encoder_attention_mask").ptr,
                source_frames=self.real_frames,
                stream=stream,
            )
            graph = self.runtime.stream_end_capture(stream)
        except Exception:
            try:
                leaked = self.runtime.stream_end_capture(stream)
                if leaked:
                    self.runtime.graph_destroy(leaked)
            except Exception:
                pass
            raise
        capture_wall_ms = (time.perf_counter_ns() - capture_start) / 1.0e6
        instantiate_start = time.perf_counter_ns()
        try:
            graph_exec = self.runtime.graph_instantiate(graph)
        except Exception:
            self.runtime.graph_destroy(graph)
            raise
        instantiate_wall_ms = (time.perf_counter_ns() - instantiate_start) / 1.0e6
        # The captured graph (and its memsets) leave a valid cross cache on replay.
        decoder.cross_cache_valid = True
        decoder.encoder_state_valid = True
        capture = MoonshineCudaBatchEncoderChainGraph(
            owner=self,
            graph=graph,
            graph_exec=graph_exec,
            capture_wall_ms=float(capture_wall_ms),
            instantiate_wall_ms=float(instantiate_wall_ms),
        )
        self._enc_chain_graph = capture
        return capture

    def graph_encode_and_handoff(self, decoder: object) -> None:
        """Replay the captured encoder+reset+handoff+cross-KV graph.

        The graph encodes the resident audio and hands the cross K/V into
        ``decoder`` on its stream, leaving a fresh generation ready to decode
        from position 0.  No host reset or per-request kernel dispatch is
        needed; the caller still uploads new audio with :meth:`upload_input`
        between replays (the graph reads the fixed audio buffer).
        """

        from hipengine.runtime.moonshine_cuda_batch import MoonshineCudaBatchRuntime

        if not isinstance(decoder, MoonshineCudaBatchRuntime):
            raise TypeError("decoder must be a MoonshineCudaBatchRuntime")
        if self.closed:
            raise RuntimeError("Moonshine batch encoder runtime is closed")
        capture = self._enc_chain_graph
        if capture is None or capture.closed:
            raise RuntimeError(
                "Moonshine batch encoder chain graph is not captured; "
                "call capture_encoder_chain first"
            )
        self.runtime.graph_launch(capture.graph_exec, decoder.stream)
        capture.replay_count += 1
        # The graph's memsets + handoff establish a fresh valid generation.
        decoder.self_cache_length = 0
        decoder.decode_position = None
        decoder.cross_cache_valid = True
        decoder.encoder_state_valid = True

    def encoder_chain_graph_contract(self) -> dict[str, object]:
        """Current encoder-chain graph capture/replay contract for reporting."""

        capture = self._enc_chain_graph
        captured = capture is not None and not capture.closed
        return {
            "captured": captured,
            "graph": 0 if not captured else capture.graph,
            "graph_exec": 0 if not captured else capture.graph_exec,
            "capture_wall_ms": 0.0 if not captured else capture.capture_wall_ms,
            "instantiate_wall_ms": 0.0 if not captured else capture.instantiate_wall_ms,
            "replay_count": 0 if not captured else capture.replay_count,
        }

    def handoff_to(self, decoder: object, *, synchronize: bool = True) -> None:
        """Hand the resident B encoder rows + masks to a ``MoonshineCudaBatchRuntime``.

        Zeroes the decoder's cross cache + mask, projects all B rows into the
        batch-strided cross cache on device (zero-padding the decoder bucket
        tail), and copies the downsampled per-row masks so the batch decoder is
        ready to decode all rows from position 0.  With ``synchronize=False``
        the work is queued on the decoder's stream without a terminal host sync.
        """

        from hipengine.runtime.moonshine_cuda_batch import MoonshineCudaBatchRuntime

        if not isinstance(decoder, MoonshineCudaBatchRuntime):
            raise TypeError("decoder must be a MoonshineCudaBatchRuntime")
        decoder.set_encoder_state_from_batch_encoder(
            encoder_hidden_ptr=self.tensor("encoder_output").ptr,
            attention_mask_ptr=self.tensor("encoder_attention_mask").ptr,
            source_frames=self.real_frames,
            synchronize=synchronize,
        )

    def close(self) -> None:
        """Free workspace, weights, cuBLASLt, and stream, and report teardown parity."""

        if self.closed:
            return
        self.closed = True
        try:
            if self._enc_chain_graph is not None:
                self._enc_chain_graph.close()
                self._enc_chain_graph = None
            if self.cublaslt is not None:
                self.cublaslt.close()
                self.cublaslt = None
            if self.cudnn is not None:
                self.cudnn.close()
                self.cudnn = None
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

    def __enter__(self) -> "MoonshineCudaBatchEncoderRuntime":
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        self.close()
