"""Eager c=1 Laguna sigmoid-MoE execution over resident GGUF weights."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Callable

from hipengine.core.hip import HipRuntime
from hipengine.core.memory import DeviceBuffer, free, malloc
from hipengine.kernels.backends import load_backend_kernel_package
from hipengine.kernels.registry import KernelKey, is_registered, resolve
from hipengine.loading.laguna_gguf import LagunaGGUFConfig, SPARSE_MOE
from hipengine.loading.laguna_gguf_materialize import (
    LAYOUT_DENSE_F32,
    LAYOUT_GGUF_Q4_K_T16,
    LAYOUT_GGUF_Q6_K_T16,
    LAYOUT_Q4_K_PACK8,
    LAYOUT_RAW_GGUF,
    LagunaGGUFResidentLayerWeights,
)
from hipengine.quant.gguf_q4_k import GGUF_Q4_K_TILE16_BLOCK_BYTES
from hipengine.quant.gguf_t16 import GGUF_Q6_K_T16_BLOCK_BYTES
from hipengine.runtime.gguf_linear import launch_gguf_linear

_QK_K = 256
_T16_COLUMNS = 16
_BF16_NBYTES = 2
_F32_NBYTES = 4
_I64_NBYTES = 8

_ROUTER_LOGITS_VARIANT = "bf16_hidden"
_ROUTER_SELECT_VARIANT = "correction_bias"
_SELECTED_DUAL_VARIANT = "selected_dual_t16_gemv_decode_bf16_bf16_out"
_SELECTED_DOWN_VARIANT = "selected_t16_gemv_decode_bf16_bf16_out"
_SILU_VARIANT = "out"
_WEIGHTED_SUM_VARIANT = "out"
_ADD_VARIANT = "add"


@dataclass(frozen=True)
class LagunaMoEKernelPlan:
    """Resolved registry plan and exact eager Laguna MoE dimensions."""

    backend: str
    hidden_size: int
    expert_count: int
    top_k: int
    expert_ffn_size: int
    shared_ffn_size: int
    routed_scaling_factor: float
    router_logits_key: KernelKey
    router_select_key: KernelKey
    selected_gate_up_key: KernelKey
    selected_silu_key: KernelKey
    selected_down_key: KernelKey
    selected_down_keys: Mapping[str, KernelKey]
    routed_sum_key: KernelKey
    shared_silu_key: KernelKey
    add_key: KernelKey
    router_logits: Callable
    router_select: Callable
    selected_gate_up: Callable
    selected_silu: Callable
    selected_down: Callable
    selected_downs: Mapping[str, Callable]
    routed_sum: Callable
    shared_silu: Callable
    add: Callable

    @property
    def kernel_keys(self) -> tuple[KernelKey, ...]:
        return (
            self.router_logits_key,
            self.router_select_key,
            self.selected_gate_up_key,
            self.selected_silu_key,
            *tuple(self.selected_down_keys.values()),
            self.routed_sum_key,
            self.shared_silu_key,
            self.add_key,
        )


@dataclass(frozen=True)
class LagunaMoEScratch:
    """Owned caller-visible buffers for the exact unfused c=1 MoE chain."""

    plan: LagunaMoEKernelPlan
    router_logits: DeviceBuffer
    routing_scores: DeviceBuffer
    selection_scores: DeviceBuffer
    selected_experts: DeviceBuffer
    routing_weights: DeviceBuffer
    scaled_routing_weights: DeviceBuffer
    expert_gate: DeviceBuffer
    expert_up: DeviceBuffer
    expert_intermediate: DeviceBuffer
    expert_down: DeviceBuffer
    routed_output: DeviceBuffer
    shared_gate: DeviceBuffer
    shared_up: DeviceBuffer
    shared_intermediate: DeviceBuffer
    shared_output: DeviceBuffer
    output: DeviceBuffer

    @property
    def buffers(self) -> tuple[DeviceBuffer, ...]:
        return (
            self.router_logits,
            self.routing_scores,
            self.selection_scores,
            self.selected_experts,
            self.routing_weights,
            self.scaled_routing_weights,
            self.expert_gate,
            self.expert_up,
            self.expert_intermediate,
            self.expert_down,
            self.routed_output,
            self.shared_gate,
            self.shared_up,
            self.shared_intermediate,
            self.shared_output,
            self.output,
        )

    @property
    def nbytes(self) -> int:
        return sum(buffer.nbytes for buffer in self.buffers)

    def free(self, *, runtime: HipRuntime | None = None) -> None:
        for buffer in reversed(self.buffers):
            free(buffer, runtime=runtime)


def resolve_laguna_moe_plan(
    config: LagunaGGUFConfig,
    *,
    backend: str,
) -> LagunaMoEKernelPlan:
    """Resolve Laguna's eager MoE stages without backend/quant branches."""

    if config.expert_gating_func != "sigmoid":
        raise ValueError("Laguna MoE plan requires sigmoid expert gating")
    if not config.expert_weights_norm:
        raise ValueError("Laguna MoE plan requires normalized selected probabilities")
    if config.expert_count <= 0 or config.expert_count > 256:
        raise ValueError("Laguna MoE plan supports between 1 and 256 experts")
    if config.expert_used_count <= 0 or config.expert_used_count > 16:
        raise ValueError("Laguna MoE plan supports between top-1 and top-16 routing")
    if config.expert_used_count > config.expert_count:
        raise ValueError("Laguna MoE top-k cannot exceed the expert count")
    if not math.isfinite(config.expert_weights_scale) or config.expert_weights_scale <= 0.0:
        raise ValueError("Laguna routed scaling factor must be finite and positive")
    if config.hidden_size % _QK_K:
        raise ValueError("Laguna hidden_size must be divisible by GGUF K block size 256")
    if config.expert_feed_forward_length % _QK_K:
        raise ValueError("Laguna expert FFN size must be divisible by GGUF K block size 256")
    if config.expert_shared_feed_forward_length % _QK_K:
        raise ValueError("Laguna shared FFN size must be divisible by GGUF K block size 256")

    load_backend_kernel_package(backend)
    keys = {
        "router_logits": KernelKey(backend, "router_logits", "f32", _ROUTER_LOGITS_VARIANT),
        "router_select": KernelKey(
            backend,
            "laguna_sigmoid_router_topk",
            "f32",
            _ROUTER_SELECT_VARIANT,
        ),
        "selected_gate_up": KernelKey(
            backend,
            "moe_linear",
            "gguf_q4_k_t16_v1",
            _SELECTED_DUAL_VARIANT,
        ),
        "selected_silu": KernelKey(backend, "silu_mul_separate", "bf16", _SILU_VARIANT),
        "selected_down": KernelKey(
            backend,
            "moe_linear",
            "gguf_q6_k_t16_v1",
            _SELECTED_DOWN_VARIANT,
        ),
        "routed_sum": KernelKey(backend, "weighted_sum", "bf16", _WEIGHTED_SUM_VARIANT),
        "shared_silu": KernelKey(backend, "silu_mul_separate", "bf16", _SILU_VARIANT),
        "add": KernelKey(backend, "elementwise", "bf16", _ADD_VARIANT),
    }
    selected_down_keys = MappingProxyType(
        {
            quant: KernelKey(backend, "moe_linear", quant, _SELECTED_DOWN_VARIANT)
            for quant in ("gguf_q4_k_t16_v1", "gguf_q6_k_t16_v1")
        }
    )
    functions = {name: _resolve_exact(key) for name, key in keys.items()}
    selected_downs = MappingProxyType(
        {quant: _resolve_exact(key) for quant, key in selected_down_keys.items()}
    )
    return LagunaMoEKernelPlan(
        backend=backend,
        hidden_size=config.hidden_size,
        expert_count=config.expert_count,
        top_k=config.expert_used_count,
        expert_ffn_size=config.expert_feed_forward_length,
        shared_ffn_size=config.expert_shared_feed_forward_length,
        routed_scaling_factor=config.expert_weights_scale,
        router_logits_key=keys["router_logits"],
        router_select_key=keys["router_select"],
        selected_gate_up_key=keys["selected_gate_up"],
        selected_silu_key=keys["selected_silu"],
        selected_down_key=keys["selected_down"],
        selected_down_keys=selected_down_keys,
        routed_sum_key=keys["routed_sum"],
        shared_silu_key=keys["shared_silu"],
        add_key=keys["add"],
        router_logits=functions["router_logits"],
        router_select=functions["router_select"],
        selected_gate_up=functions["selected_gate_up"],
        selected_silu=functions["selected_silu"],
        selected_down=functions["selected_down"],
        selected_downs=selected_downs,
        routed_sum=functions["routed_sum"],
        shared_silu=functions["shared_silu"],
        add=functions["add"],
    )


def allocate_laguna_moe_scratch(
    plan: LagunaMoEKernelPlan,
    *,
    runtime: HipRuntime | None = None,
) -> LagunaMoEScratch:
    """Allocate all c=1 router/routed/shared intermediates with failure cleanup."""

    h = plan.hidden_size
    e = plan.expert_count
    k = plan.top_k
    f = plan.expert_ffn_size
    sf = plan.shared_ffn_size
    sizes = (
        e * _F32_NBYTES,
        e * _F32_NBYTES,
        e * _F32_NBYTES,
        k * _I64_NBYTES,
        k * _F32_NBYTES,
        k * _F32_NBYTES,
        k * f * _BF16_NBYTES,
        k * f * _BF16_NBYTES,
        k * f * _BF16_NBYTES,
        k * h * _BF16_NBYTES,
        h * _BF16_NBYTES,
        sf * _BF16_NBYTES,
        sf * _BF16_NBYTES,
        sf * _BF16_NBYTES,
        h * _BF16_NBYTES,
        h * _BF16_NBYTES,
    )
    buffers: list[DeviceBuffer] = []
    try:
        buffers.extend(malloc(nbytes, runtime=runtime) for nbytes in sizes)
    except Exception:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)
        raise
    return LagunaMoEScratch(plan, *buffers)


def validate_laguna_moe_layer(
    layer: LagunaGGUFResidentLayerWeights,
    plan: LagunaMoEKernelPlan,
) -> None:
    """Validate the complete production rank/layout/byte-stride contract."""

    if layer.mlp_type != SPARSE_MOE:
        raise ValueError(f"Laguna MoE requires layer.mlp_type={SPARSE_MOE!r}")
    resident_backends = {weight.backend for weight in layer.weights.values()}
    if resident_backends != {plan.backend}:
        raise ValueError(
            "Laguna MoE resident weight backends must match the plan: "
            f"plan={plan.backend!r} weights={tuple(sorted(resident_backends))}"
        )
    required = (
        "ffn_gate_inp",
        "exp_probs_b",
        "ffn_gate_exps",
        "ffn_up_exps",
        "ffn_down_exps",
        "ffn_gate_shexp",
        "ffn_up_shexp",
        "ffn_down_shexp",
    )
    missing = tuple(name for name in required if name not in layer.weights)
    if missing:
        raise ValueError(f"Laguna MoE resident layer is missing weights: {missing}")

    h, e, f, sf = (
        plan.hidden_size,
        plan.expert_count,
        plan.expert_ffn_size,
        plan.shared_ffn_size,
    )
    selected_down = layer.weight("ffn_down_exps")
    selected_down_contracts = {
        "gguf_q4_k_t16_v1": (
            LAYOUT_GGUF_Q4_K_T16,
            (e, h, (f // _QK_K) * 144),
        ),
        "gguf_q6_k_t16_v1": (
            LAYOUT_GGUF_Q6_K_T16,
            (e, h, (f // _QK_K) * 210),
        ),
    }
    try:
        selected_down_layout, selected_down_byte_shape = selected_down_contracts[
            selected_down.spec.quant_key
        ]
    except KeyError as exc:
        raise ValueError("ffn_down_exps must use a registered Q4_K or Q6_K T16 layout") from exc

    shared_down = layer.weight("ffn_down_shexp")
    shared_down_contracts = {
        "gguf_q4_k": (LAYOUT_Q4_K_PACK8, (h, (sf // _QK_K) * 144)),
        "gguf_q6_k": (LAYOUT_RAW_GGUF, (h, (sf // _QK_K) * 210)),
    }
    try:
        shared_down_layout, shared_down_byte_shape = shared_down_contracts[
            shared_down.spec.quant_key
        ]
    except KeyError as exc:
        raise ValueError(
            "ffn_down_shexp must use a registered Q4_K pack8 or raw Q6_K layout"
        ) from exc

    expected = {
        "ffn_gate_inp": ((e, h), LAYOUT_DENSE_F32, "f32", (e, h)),
        "exp_probs_b": ((e,), LAYOUT_DENSE_F32, "f32", (e,)),
        "ffn_gate_exps": (
            (e, f, h),
            LAYOUT_GGUF_Q4_K_T16,
            "gguf_q4_k_t16_v1",
            (e, f, (h // _QK_K) * 144),
        ),
        "ffn_up_exps": (
            (e, f, h),
            LAYOUT_GGUF_Q4_K_T16,
            "gguf_q4_k_t16_v1",
            (e, f, (h // _QK_K) * 144),
        ),
        "ffn_down_exps": (
            (e, h, f),
            selected_down_layout,
            selected_down.spec.quant_key,
            selected_down_byte_shape,
        ),
        "ffn_gate_shexp": (
            (sf, h),
            LAYOUT_Q4_K_PACK8,
            "gguf_q4_k",
            (sf, (h // _QK_K) * 144),
        ),
        "ffn_up_shexp": (
            (sf, h),
            LAYOUT_Q4_K_PACK8,
            "gguf_q4_k",
            (sf, (h // _QK_K) * 144),
        ),
        "ffn_down_shexp": (
            (h, sf),
            shared_down_layout,
            shared_down.spec.quant_key,
            shared_down_byte_shape,
        ),
    }
    for name, (shape, layout, quant, byte_shape) in expected.items():
        weight = layer.weight(name)
        source = weight.spec.source
        if source.shape != shape:
            raise ValueError(f"{name} shape must be {shape}, got {source.shape}")
        if source.byte_shape != byte_shape:
            raise ValueError(
                f"{name} raw byte shape/stride must be {byte_shape}, got {source.byte_shape}"
            )
        if weight.spec.layout != layout or weight.spec.quant_key != quant:
            raise ValueError(
                f"{name} must use layout/quant {layout}/{quant}, got "
                f"{weight.spec.layout}/{weight.spec.quant_key}"
            )

    q4_t16_nbytes = e * (f // _T16_COLUMNS) * (h // _QK_K) * GGUF_Q4_K_TILE16_BLOCK_BYTES
    for name in ("ffn_gate_exps", "ffn_up_exps"):
        if layer.weight(name).allocation("tiles").buffer.nbytes != q4_t16_nbytes:
            raise ValueError(f"{name} T16 allocation does not match rank-3 expert stride")
    selected_down_tile_bytes = (
        GGUF_Q4_K_TILE16_BLOCK_BYTES
        if selected_down.spec.quant_key == "gguf_q4_k_t16_v1"
        else GGUF_Q6_K_T16_BLOCK_BYTES
    )
    selected_down_nbytes = e * (h // _T16_COLUMNS) * (f // _QK_K) * selected_down_tile_bytes
    if selected_down.allocation("tiles").buffer.nbytes != selected_down_nbytes:
        raise ValueError("ffn_down_exps T16 allocation does not match rank-3 expert stride")


def run_laguna_moe_c1(
    hidden_bf16_ptr: int,
    layer: LagunaGGUFResidentLayerWeights,
    scratch: LagunaMoEScratch,
    *,
    stream: int = 0,
    runtime: HipRuntime | None = None,
    libraries: Mapping[str, object] | None = None,
) -> DeviceBuffer:
    """Run the exact staged Laguna routed plus always-on shared expert path."""

    plan = scratch.plan
    validate_laguna_moe_layer(layer, plan)
    h, e, k, f, sf = (
        plan.hidden_size,
        plan.expert_count,
        plan.top_k,
        plan.expert_ffn_size,
        plan.shared_ffn_size,
    )
    router = layer.weight("ffn_gate_inp").allocation("raw").tensor.ptr
    correction = layer.weight("exp_probs_b").allocation("raw").tensor.ptr
    gate_tiles = layer.weight("ffn_gate_exps").allocation("tiles").tensor.ptr
    up_tiles = layer.weight("ffn_up_exps").allocation("tiles").tensor.ptr
    down_weight = layer.weight("ffn_down_exps")
    down_tiles = down_weight.allocation("tiles").tensor.ptr
    try:
        selected_down_fn = plan.selected_downs[down_weight.spec.quant_key]
    except KeyError as exc:
        raise ValueError(
            f"no Laguna selected-down kernel for {down_weight.spec.quant_key!r}"
        ) from exc

    plan.router_logits(
        hidden_bf16_ptr,
        router,
        scratch.router_logits.ptr,
        1,
        h,
        e,
        **_stage_kwargs("router_logits", libraries, stream=stream, runtime=runtime),
    )
    plan.router_select(
        scratch.router_logits.ptr,
        correction,
        scratch.routing_scores.ptr,
        scratch.selection_scores.ptr,
        scratch.selected_experts.ptr,
        scratch.routing_weights.ptr,
        scratch.scaled_routing_weights.ptr,
        1,
        e,
        k,
        plan.routed_scaling_factor,
        **_stage_kwargs("router_select", libraries, stream=stream, runtime=runtime),
    )
    plan.selected_gate_up(
        hidden_bf16_ptr,
        scratch.selected_experts.ptr,
        gate_tiles,
        up_tiles,
        scratch.expert_gate.ptr,
        scratch.expert_up.ptr,
        1,
        k,
        e,
        h,
        f,
        **_stage_kwargs("selected_gate_up", libraries, stream=stream, runtime=runtime),
    )
    plan.selected_silu(
        scratch.expert_gate.ptr,
        scratch.expert_up.ptr,
        scratch.expert_intermediate.ptr,
        k,
        f,
        **_stage_kwargs("selected_silu", libraries, stream=stream, runtime=runtime),
    )
    selected_down_fn(
        scratch.expert_intermediate.ptr,
        scratch.selected_experts.ptr,
        down_tiles,
        scratch.expert_down.ptr,
        k,
        k,
        e,
        f,
        h,
        **_stage_kwargs("selected_down", libraries, stream=stream, runtime=runtime),
    )
    plan.routed_sum(
        scratch.expert_down.ptr,
        scratch.scaled_routing_weights.ptr,
        scratch.routed_output.ptr,
        k,
        h,
        **_stage_kwargs("routed_sum", libraries, stream=stream, runtime=runtime),
    )

    shared_gate = layer.weight("ffn_gate_shexp")
    shared_up = layer.weight("ffn_up_shexp")
    shared_down = layer.weight("ffn_down_shexp")
    launch_gguf_linear(
        shared_gate,
        hidden_bf16_ptr,
        scratch.shared_gate.ptr,
        1,
        h,
        sf,
        backend=plan.backend,
        stream=stream,
        runtime=runtime,
        libraries=libraries,
        use_wmma_prefill=False,
        use_gemv_decode=False,
    )
    launch_gguf_linear(
        shared_up,
        hidden_bf16_ptr,
        scratch.shared_up.ptr,
        1,
        h,
        sf,
        backend=plan.backend,
        stream=stream,
        runtime=runtime,
        libraries=libraries,
        use_wmma_prefill=False,
        use_gemv_decode=False,
    )
    plan.shared_silu(
        scratch.shared_gate.ptr,
        scratch.shared_up.ptr,
        scratch.shared_intermediate.ptr,
        1,
        sf,
        **_stage_kwargs("shared_silu", libraries, stream=stream, runtime=runtime),
    )
    launch_gguf_linear(
        shared_down,
        scratch.shared_intermediate.ptr,
        scratch.shared_output.ptr,
        1,
        sf,
        h,
        backend=plan.backend,
        stream=stream,
        runtime=runtime,
        libraries=libraries,
        use_wmma_prefill=False,
        use_gemv_decode=False,
    )
    plan.add(
        scratch.routed_output.ptr,
        scratch.shared_output.ptr,
        scratch.output.ptr,
        h,
        **_stage_kwargs("add", libraries, stream=stream, runtime=runtime),
    )
    return scratch.output


def _stage_kwargs(
    name: str,
    libraries: Mapping[str, object] | None,
    *,
    stream: int,
    runtime: HipRuntime | None,
) -> dict[str, object]:
    kwargs: dict[str, object] = {"stream": stream, "runtime": runtime}
    if libraries is not None and (library := libraries.get(name)) is not None:
        kwargs["library"] = library
    return kwargs


def _resolve_exact(key: KernelKey) -> Callable:
    if not is_registered(key):
        raise LookupError(f"required Laguna kernel is not registered: {key.display()}")
    function = resolve(
        backend=key.backend,
        layer=key.layer,
        quant=key.quant,
        variant=key.variant,
    )
    assert function is not None
    return function


__all__ = [
    "LagunaMoEKernelPlan",
    "LagunaMoEScratch",
    "allocate_laguna_moe_scratch",
    "resolve_laguna_moe_plan",
    "run_laguna_moe_c1",
    "validate_laguna_moe_layer",
]
