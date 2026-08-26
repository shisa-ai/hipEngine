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
from hipengine.kernels.backends import load_backend_kernel_package
from hipengine.kernels.hip_gfx1100.convert.cast import f32_to_bf16
from hipengine.kernels.hip_gfx1100.fused.paro_combine import (
    shared_gate_combine_out_bf16,
    weighted_sum_out_bf16_f32w,
)
from hipengine.kernels.hip_gfx1100.fused.paro_silu import (
    silu_mul_separate_out_bf16,
)
from hipengine.kernels.hip_gfx1100.moe.router import qwen35_router_select
from hipengine.kernels.hip_gfx1100.linear_attn.conv import (
    qwen35_linear_attn_conv_decode_f32,
)
from hipengine.kernels.hip_gfx1100.linear_attn.qwen4_exp_gdn import (
    qwen4_exp_gdn_decode_f32,
)
from hipengine.kernels.hip_gfx1100.fused.qwen4_exp_gr import (
    qwen4_exp_gated_mean_f32,
    qwen4_exp_grouped_rmsnorm_bf16_f32,
    qwen4_exp_grouped_rmsnorm_f32,
    qwen4_exp_scaled_silu_f32,
    qwen4_exp_silu_mul_f32,
    qwen4_exp_sigmoid_f32,
)
from hipengine.kernels.hip_gfx1100.fused.qwen4_exp_ple import (
    qwen4_exp_ple_add_delta_bf16_f32,
    qwen4_exp_ple_dilated_depthwise_conv_f32,
    qwen4_exp_ple_repeat_gated_value_f32,
    qwen4_exp_ple_signed_sqrt_gate_f32,
)
from hipengine.runtime.gguf_linear import (
    GGUF_ACTIVATION_F32,
    GGUF_OUTPUT_F32,
    launch_gguf_linear,
)
from hipengine.kernels.registry import resolve
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
class Qwen4ExpPLEScratch:
    key: DeviceBuffer
    value: DeviceBuffer
    query: DeviceBuffer
    gate: DeviceBuffer
    gated_value: DeviceBuffer
    normalized: DeviceBuffer
    conv: DeviceBuffer
    output: DeviceBuffer
    runtime: HipRuntime
    closed: bool = False

    @classmethod
    def allocate(
        cls,
        *,
        rows: int,
        branches: int,
        hidden: int,
        runtime: HipRuntime | None = None,
    ) -> "Qwen4ExpPLEScratch":
        if rows <= 0 or branches <= 0 or hidden <= 0:
            raise ValueError("rows, branches, and hidden must be positive")
        active_runtime = runtime or get_hip_runtime()
        channels = branches * hidden
        byte_sizes = (
            rows * channels * 4,
            rows * hidden * 4,
            rows * channels * 4,
            rows * branches * 4,
            rows * channels * 4,
            rows * channels * 4,
            rows * channels * 4,
            rows * channels * 2,
        )
        buffers: list[DeviceBuffer] = []
        try:
            for size in byte_sizes:
                buffers.append(malloc(size, runtime=active_runtime))
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
                self.key,
                self.value,
                self.query,
                self.gate,
                self.gated_value,
                self.normalized,
                self.conv,
                self.output,
            )
        ):
            free(buffer, runtime=self.runtime)
        self.closed = True


@dataclass
class Qwen4ExpMoEScratch:
    router_logits: DeviceBuffer
    selected: DeviceBuffer
    routing: DeviceBuffer
    hidden_bf16: DeviceBuffer
    expert_gate: DeviceBuffer
    expert_up: DeviceBuffer
    expert_intermediate: DeviceBuffer
    expert_down: DeviceBuffer
    routed: DeviceBuffer
    shared_gate: DeviceBuffer
    shared_up: DeviceBuffer
    shared_intermediate: DeviceBuffer
    shared_down: DeviceBuffer
    shared_down_bf16: DeviceBuffer
    shared_gate_logits: DeviceBuffer
    output: DeviceBuffer
    runtime: HipRuntime
    closed: bool = False

    @classmethod
    def allocate(
        cls,
        *,
        rows: int,
        hidden: int,
        ffn: int,
        experts: int,
        top_k: int,
        runtime: HipRuntime | None = None,
    ) -> "Qwen4ExpMoEScratch":
        dimensions = (rows, hidden, ffn, experts, top_k)
        if any(int(value) <= 0 for value in dimensions) or top_k > experts:
            raise ValueError("Qwen4Exp MoE dimensions must be positive with top_k <= experts")
        active_runtime = runtime or get_hip_runtime()
        compact = rows * top_k
        byte_sizes = (
            rows * experts * 4,
            compact * 4,
            compact * 4,
            rows * hidden * 2,
            compact * ffn * 2,
            compact * ffn * 2,
            compact * ffn * 2,
            compact * hidden * 2,
            rows * hidden * 2,
            rows * ffn * 4,
            rows * ffn * 4,
            rows * ffn * 4,
            rows * hidden * 4,
            rows * hidden * 2,
            rows * 4,
            rows * hidden * 2,
        )
        buffers: list[DeviceBuffer] = []
        try:
            for size in byte_sizes:
                buffers.append(malloc(size, runtime=active_runtime))
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
                self.router_logits,
                self.selected,
                self.routing,
                self.hidden_bf16,
                self.expert_gate,
                self.expert_up,
                self.expert_intermediate,
                self.expert_down,
                self.routed,
                self.shared_gate,
                self.shared_up,
                self.shared_intermediate,
                self.shared_down,
                self.shared_down_bf16,
                self.shared_gate_logits,
                self.output,
            )
        ):
            free(buffer, runtime=self.runtime)
        self.closed = True


@dataclass(frozen=True)
class Qwen4ExpMoEDeviceResult:
    output: DeviceBuffer
    selected: DeviceBuffer
    routing: DeviceBuffer


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


def run_qwen4_exp_moe(
    mixed_ptr: int,
    weights: Mapping[str, GGUFDeviceWeight],
    *,
    scratch: Qwen4ExpMoEScratch,
    rows: int,
    hidden: int,
    ffn: int,
    experts: int,
    top_k: int,
    stream: int = 0,
    runtime: HipRuntime | None = None,
) -> Qwen4ExpMoEDeviceResult:
    """Run normalized softmax top-k routed experts plus gated shared expert."""

    if scratch.closed:
        raise RuntimeError("Qwen4Exp MoE scratch is closed")
    if rows != 1:
        raise ValueError("strict Qwen4Exp MoE decode currently requires rows == 1")
    active_runtime = runtime or scratch.runtime
    if active_runtime is not scratch.runtime:
        raise ValueError("runtime must match the MoE scratch owner")
    required = {
        "router",
        "expert_gate",
        "expert_up",
        "expert_down",
        "shared_gate",
        "shared_up",
        "shared_down",
        "shared_gate_weight",
    }
    missing = sorted(required - set(weights))
    if missing:
        raise ValueError("missing Qwen4Exp MoE weights: " + ", ".join(missing))
    backend = str(weights["expert_gate"].backend)
    load_backend_kernel_package(backend)
    launch_gguf_linear(
        weights["router"], mixed_ptr, scratch.router_logits.ptr,
        rows, hidden, experts,
        activation_dtype=GGUF_ACTIVATION_F32,
        output_dtype=GGUF_OUTPUT_F32,
        stream=stream, runtime=active_runtime,
    )
    qwen35_router_select(
        scratch.router_logits.ptr,
        scratch.selected.ptr,
        scratch.routing.ptr,
        rows,
        experts,
        experts,
        top_k,
        stream=stream,
        runtime=active_runtime,
    )
    f32_to_bf16(
        mixed_ptr,
        scratch.hidden_bf16.ptr,
        rows * hidden,
        stream=stream,
        runtime=active_runtime,
    )
    compact = rows * top_k

    def selected_projection(
        slot: str,
        input_ptr: int,
        output_ptr: int,
        x_rows: int,
        selected_rows: int,
        in_features: int,
        out_features: int,
    ) -> None:
        weight = weights[slot]
        function = resolve(
            backend=backend,
            layer="linear",
            quant=weight.spec.quant_key,
            variant="selected_gemv_bf16_bf16_out",
        )
        function(
            input_ptr,
            scratch.selected.ptr,
            weight.allocation("raw").tensor.ptr,
            output_ptr,
            x_rows,
            selected_rows,
            experts,
            in_features,
            out_features,
            stream=stream,
            runtime=active_runtime,
        )

    selected_projection(
        "expert_gate", scratch.hidden_bf16.ptr, scratch.expert_gate.ptr,
        rows, compact, hidden, ffn,
    )
    selected_projection(
        "expert_up", scratch.hidden_bf16.ptr, scratch.expert_up.ptr,
        rows, compact, hidden, ffn,
    )
    silu_mul_separate_out_bf16(
        scratch.expert_gate.ptr,
        scratch.expert_up.ptr,
        scratch.expert_intermediate.ptr,
        compact,
        ffn,
        stream=stream,
        runtime=active_runtime,
    )
    selected_projection(
        "expert_down", scratch.expert_intermediate.ptr, scratch.expert_down.ptr,
        compact, compact, ffn, hidden,
    )
    weighted_sum_out_bf16_f32w(
        scratch.expert_down.ptr,
        scratch.routing.ptr,
        scratch.routed.ptr,
        compact,
        hidden,
        stream=stream,
        runtime=active_runtime,
    )
    for slot, output in (
        ("shared_gate", scratch.shared_gate),
        ("shared_up", scratch.shared_up),
    ):
        launch_gguf_linear(
            weights[slot], mixed_ptr, output.ptr,
            rows, hidden, ffn,
            activation_dtype=GGUF_ACTIVATION_F32,
            output_dtype=GGUF_OUTPUT_F32,
            stream=stream, runtime=active_runtime,
        )
    qwen4_exp_silu_mul_f32(
        scratch.shared_gate.ptr,
        scratch.shared_up.ptr,
        scratch.shared_intermediate.ptr,
        rows * ffn,
        stream=stream,
        runtime=active_runtime,
    )
    launch_gguf_linear(
        weights["shared_down"], scratch.shared_intermediate.ptr, scratch.shared_down.ptr,
        rows, ffn, hidden,
        activation_dtype=GGUF_ACTIVATION_F32,
        output_dtype=GGUF_OUTPUT_F32,
        stream=stream, runtime=active_runtime,
    )
    launch_gguf_linear(
        weights["shared_gate_weight"], mixed_ptr, scratch.shared_gate_logits.ptr,
        rows, hidden, 1,
        activation_dtype=GGUF_ACTIVATION_F32,
        output_dtype=GGUF_OUTPUT_F32,
        stream=stream, runtime=active_runtime,
    )
    f32_to_bf16(
        scratch.shared_down.ptr,
        scratch.shared_down_bf16.ptr,
        rows * hidden,
        stream=stream,
        runtime=active_runtime,
    )
    shared_gate_combine_out_bf16(
        scratch.routed.ptr,
        scratch.shared_down_bf16.ptr,
        scratch.shared_gate_logits.ptr,
        scratch.output.ptr,
        hidden,
        stream=stream,
        runtime=active_runtime,
    )
    return Qwen4ExpMoEDeviceResult(scratch.output, scratch.selected, scratch.routing)


def run_qwen4_exp_ple(
    residual_ptr: int,
    embedding_ptr: int,
    weights: Mapping[str, GGUFDeviceWeight],
    *,
    norm_key_ptr: int,
    norm_query_ptr: int,
    norm_conv_ptr: int,
    conv_weight_ptr: int,
    conv_history_ptr: int,
    scratch: Qwen4ExpPLEScratch,
    rows: int,
    branches: int,
    hidden: int,
    conv_kernel: int,
    dilation: int,
    eps: float = 1e-6,
    stream: int = 0,
    runtime: HipRuntime | None = None,
) -> DeviceBuffer:
    """Execute strict PLE projections, branch gate, Conv, and residual injection."""

    if scratch.closed:
        raise RuntimeError("Qwen4Exp PLE scratch is closed")
    active_runtime = runtime or scratch.runtime
    if active_runtime is not scratch.runtime:
        raise ValueError("runtime must match the PLE scratch owner")
    required = {"ple_key", "ple_value"}
    missing = sorted(required - set(weights))
    if missing:
        raise ValueError("missing Qwen4Exp PLE weights: " + ", ".join(missing))
    channels = branches * hidden
    launch_gguf_linear(
        weights["ple_key"],
        embedding_ptr,
        scratch.key.ptr,
        rows,
        hidden,
        channels,
        activation_dtype=GGUF_ACTIVATION_F32,
        output_dtype=GGUF_OUTPUT_F32,
        stream=stream,
        runtime=active_runtime,
    )
    launch_gguf_linear(
        weights["ple_value"],
        embedding_ptr,
        scratch.value.ptr,
        rows,
        hidden,
        hidden,
        activation_dtype=GGUF_ACTIVATION_F32,
        output_dtype=GGUF_OUTPUT_F32,
        stream=stream,
        runtime=active_runtime,
    )
    qwen4_exp_grouped_rmsnorm_f32(
        scratch.key.ptr,
        norm_key_ptr,
        scratch.key.ptr,
        rows,
        branches,
        hidden,
        eps,
        stream=stream,
        runtime=active_runtime,
    )
    qwen4_exp_grouped_rmsnorm_bf16_f32(
        residual_ptr,
        norm_query_ptr,
        scratch.query.ptr,
        rows,
        branches,
        hidden,
        eps,
        stream=stream,
        runtime=active_runtime,
    )
    qwen4_exp_ple_signed_sqrt_gate_f32(
        scratch.key.ptr,
        scratch.query.ptr,
        scratch.gate.ptr,
        rows,
        branches,
        hidden,
        stream=stream,
        runtime=active_runtime,
    )
    qwen4_exp_ple_repeat_gated_value_f32(
        scratch.value.ptr,
        scratch.gate.ptr,
        scratch.gated_value.ptr,
        rows,
        branches,
        hidden,
        stream=stream,
        runtime=active_runtime,
    )
    qwen4_exp_grouped_rmsnorm_f32(
        scratch.gated_value.ptr,
        norm_conv_ptr,
        scratch.normalized.ptr,
        rows,
        branches,
        hidden,
        eps,
        stream=stream,
        runtime=active_runtime,
    )
    qwen4_exp_ple_dilated_depthwise_conv_f32(
        scratch.normalized.ptr,
        conv_weight_ptr,
        conv_history_ptr,
        scratch.conv.ptr,
        rows,
        channels,
        conv_kernel,
        dilation,
        stream=stream,
        runtime=active_runtime,
    )
    qwen4_exp_scaled_silu_f32(
        scratch.conv.ptr,
        scratch.conv.ptr,
        rows * channels,
        1.0,
        stream=stream,
        runtime=active_runtime,
    )
    qwen4_exp_ple_add_delta_bf16_f32(
        residual_ptr,
        scratch.gated_value.ptr,
        scratch.conv.ptr,
        scratch.output.ptr,
        rows * channels,
        stream=stream,
        runtime=active_runtime,
    )
    return scratch.output


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
    "Qwen4ExpMoEDeviceResult",
    "Qwen4ExpMoEScratch",
    "Qwen4ExpGRScratch",
    "Qwen4ExpPLEScratch",
    "run_qwen4_exp_gdn_token_mixer",
    "run_qwen4_exp_moe",
    "run_qwen4_exp_ple",
    "run_qwen4_exp_gr_read",
]
