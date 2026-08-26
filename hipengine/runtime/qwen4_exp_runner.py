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

from hipengine.core.hip import HipRuntime, get_hip_runtime
from hipengine.core.memory import (
    DeviceBuffer,
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.hip_gfx1100.linear_attn.conv import (
    qwen35_linear_attn_conv_decode_f32,
)
from hipengine.kernels.hip_gfx1100.linear_attn.qwen4_exp_gdn import (
    qwen4_exp_gdn_decode_f32,
)
from hipengine.kernels.hip_gfx1100.fused.qwen4_exp_gr import (
    qwen4_exp_gated_mean_f32,
    qwen4_exp_grouped_rmsnorm_bf16_f32,
    qwen4_exp_scaled_silu_f32,
    qwen4_exp_sigmoid_f32,
)
from hipengine.runtime.gguf_linear import (
    GGUF_ACTIVATION_F32,
    GGUF_OUTPUT_F32,
    launch_gguf_linear,
)
from hipengine.runtime.gguf_weight import GGUFDeviceWeight


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


def run_qwen4_exp_gr_read(
    residual_ptr: int,
    norm_weight_ptr: int,
    down_weight: GGUFDeviceWeight,
    up_weight: GGUFDeviceWeight,
    inject_weight: GGUFDeviceWeight,
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


def run_qwen4_exp_gdn_token_mixer(
    mixed_ptr: int,
    weights: Mapping[str, GGUFDeviceWeight],
    *,
    conv_weight_ptr: int,
    dt_bias_ptr: int,
    a_log_ptr: int,
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
    if rows != 1:
        raise ValueError("strict Qwen4Exp GDN decode currently requires rows == 1")
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
        a_log_ptr,
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


__all__ = [
    "Qwen4ExpDecodeState",
    "Qwen4ExpDecodeStateSnapshot",
    "Qwen4ExpGDNScratch",
    "Qwen4ExpGRReadDeviceResult",
    "Qwen4ExpGRScratch",
    "run_qwen4_exp_gdn_token_mixer",
    "run_qwen4_exp_gr_read",
]
