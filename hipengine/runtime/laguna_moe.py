"""Eager c=1 Laguna sigmoid-MoE execution over resident GGUF weights."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Callable

from hipengine.core.hip import HipRuntime, get_hip_runtime
from hipengine.core.memory import DeviceBuffer, free, malloc
from hipengine.kernels.backends import (
    backend_package_capability,
    load_backend_kernel_package,
)
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
from hipengine.runtime.gguf_linear import launch_gguf_linear, launch_gguf_linear_pair

_QK_K = 256
_Q8_1_MMQ_BLOCK = 128
_T16_COLUMNS = 16
_BF16_NBYTES = 2
_F32_NBYTES = 4
_I64_NBYTES = 8

_ROUTER_LOGITS_VARIANT = "bf16_hidden"
_ROUTER_SELECT_VARIANT = "correction_bias"
_SELECTED_DUAL_VARIANT = "selected_dual_t16_gemv_decode_bf16_bf16_out"
_SELECTED_DUAL_MMQ32_D4X3_VARIANT = (
    "selected_dual_q8_1_ds4x3_mmq32_prefill_compact32_bf16_bf16_out"
)
_SELECTED_DUAL_MMQ64X32_D4X3_F32_VARIANT = (
    "selected_dual_q8_1_ds4x3_f32_mmq64x32_"
    "prefill_compact32_bf16_bf16_out"
)
_SELECTED_DOWN_VARIANT = "selected_t16_gemv_decode_bf16_bf16_out"
_SELECTED_DOWN_MMQ64X32_D4X3_F32_VARIANT = (
    "selected_q8_1_ds4x3_f32_mmq64x32_"
    "prefill_compact32_bf16_bf16_out"
)
_SELECTED_WEIGHTED_DOWN_VARIANT = "selected_weighted_down_gemv_decode_bf16_bf16_out"
_SILU_VARIANT = "out"
_WEIGHTED_SUM_VARIANT = "out"
_ADD_VARIANT = "add"
_SELECTED_DOWN_MODES = frozenset(
    {
        "direct",
        "grouped_smallm",
        "adaptive_grouped_smallm",
        "grouped_smallm_fused",
        "adaptive_grouped_smallm_fused",
        "mmq64x32_d4_f32",
        "mmq64x32_d4_f32_rowvec",
        "mmq64x32_d4_f32_wavecols_q4",
        "mmq64x32_d4_f32_wavecols_direct_q4",
        "mmq64x64_d4_f32_q6_wavecols_direct_q4",
        "mmq64x32_d4x2_f32",
        "mmq64x32_d4x3_f32",
    }
)
_BASELINE_SELECTED_DOWN_MODE = "direct"
_GROUPED_SMALLM_MIN_ROWS = 32
_SELECTED_GATE_UP_MODES = frozenset(
    {
        "direct",
        "mmq32_d4x3",
        "mmq64x32_d4_f32",
        "mmq128x32_d8_f32",
        "mmq128x32_d8_f32_rowvec",
        "mmq128x32_d8_f32_wavecols",
        "mmq128x32_d8_f32_wavecols_direct",
        "mmq64x32_d4x2_f32",
        "mmq64x32_d4x3_f32",
    }
)
_BASELINE_SELECTED_GATE_UP_MODE = "direct"
_SELECTED_MMQ32_MIN_ROWS = 32
_DENSE_Q4_PREFILL_MODES = frozenset({"retained", "wmma_pack8"})
_BASELINE_DENSE_Q4_PREFILL_MODE = "retained"
_Q8_1_DS4_BLOCK_BYTES = 144
_Q8_1_DS4_F32_BLOCK_BYTES = 160
_Q8_1_DS4_RESIDUAL_PLANES = 3
_MMQ32_ROWS = 32
_MMQ64_ROWS = 64


def resolve_laguna_selected_gate_up_mode(
    backend: str,
    requested: str | None = None,
) -> str:
    """Resolve the exact default or explicit compact-MMQ prefill candidate."""

    selected = (
        backend_package_capability(
            backend,
            "LAGUNA_SELECTED_GATE_UP_MODE",
            _BASELINE_SELECTED_GATE_UP_MODE,
        )
        if requested is None
        else str(requested)
    )
    parsed = str(selected)
    if parsed not in _SELECTED_GATE_UP_MODES:
        raise ValueError("unsupported Laguna selected gate/up mode")
    return parsed


def resolve_laguna_selected_down_mode(
    backend: str,
    requested: str | None = None,
) -> str:
    """Resolve an explicit rollback or the architecture-qualified down default."""

    selected = (
        backend_package_capability(
            backend,
            "LAGUNA_SELECTED_DOWN_MODE",
            _BASELINE_SELECTED_DOWN_MODE,
        )
        if requested is None
        else str(requested)
    )
    parsed = str(selected)
    if parsed not in _SELECTED_DOWN_MODES:
        raise ValueError("unsupported Laguna selected-down mode")
    return parsed


def resolve_laguna_dense_q4_prefill_mode(
    backend: str,
    requested: str | None = None,
) -> str:
    """Resolve the retained dense/shared Q4 route or pack8 WMMA candidate."""

    selected = (
        backend_package_capability(
            backend,
            "LAGUNA_DENSE_Q4_PREFILL_MODE",
            _BASELINE_DENSE_Q4_PREFILL_MODE,
        )
        if requested is None
        else str(requested)
    )
    parsed = str(selected)
    if parsed not in _DENSE_Q4_PREFILL_MODES:
        raise ValueError("unsupported Laguna dense/shared Q4 prefill mode")
    return parsed


@dataclass(frozen=True)
class LagunaMoESelectedRoute:
    """One registry-resolved selected-expert ABI and resident allocation."""

    key: KernelKey
    function: Callable
    abi: str
    allocation_name: str
    library_key: str


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
    selected_gate_up_prefill_key: KernelKey
    activation_quant_key: KernelKey
    selected_gate_up_prefill_f32_key: KernelKey
    activation_quant_f32_key: KernelKey
    selected_down_prefill_key: KernelKey
    selected_down_prefill_q4_key: KernelKey
    down_activation_quant_key: KernelKey
    selected_gate_up_keys: Mapping[str, KernelKey]
    selected_gate_up_routes: Mapping[str, LagunaMoESelectedRoute]
    selected_silu_key: KernelKey
    selected_down_key: KernelKey
    selected_down_keys: Mapping[str, KernelKey]
    selected_down_routes: Mapping[str, LagunaMoESelectedRoute]
    selected_weighted_down_keys: Mapping[str, KernelKey]
    selected_weighted_down_routes: Mapping[str, LagunaMoESelectedRoute]
    routed_sum_key: KernelKey
    routed_sum_rows_key: KernelKey
    grouped_count_key: KernelKey
    grouped_prefix_active_key: KernelKey
    grouped_scatter_key: KernelKey
    grouped_compact_key: KernelKey
    grouped_compact_source_rows_key: KernelKey
    mmq_tile_map_key: KernelKey
    mmq64_tile_map_key: KernelKey
    grouped_gather_key: KernelKey
    grouped_smallm_down_keys: Mapping[str, KernelKey]
    grouped_weighted_sum_key: KernelKey
    grouped_weighted_sum_shared_add_key: KernelKey
    shared_silu_key: KernelKey
    add_key: KernelKey
    router_logits: Callable
    router_select: Callable
    selected_gate_up: Callable
    selected_silu: Callable
    selected_gate_up_prefill: Callable
    activation_quant: Callable
    selected_gate_up_prefill_f32: Callable
    activation_quant_f32: Callable
    selected_down_prefill: Callable
    selected_down_prefill_q4: Callable
    down_activation_quant: Callable
    selected_dual_silu_key: KernelKey
    selected_dual_silu: Callable
    selected_down: Callable
    selected_downs: Mapping[str, Callable]
    routed_sum: Callable
    routed_sum_rows: Callable
    grouped_count: Callable
    grouped_prefix_active: Callable
    grouped_scatter: Callable
    grouped_compact: Callable
    grouped_compact_source_rows: Callable
    mmq_tile_map: Callable
    mmq64_tile_map: Callable
    grouped_gather: Callable
    grouped_smallm_downs: Mapping[str, Callable]
    grouped_weighted_sum: Callable
    grouped_weighted_sum_shared_add: Callable
    shared_silu: Callable
    add: Callable

    @property
    def kernel_keys(self) -> tuple[KernelKey, ...]:
        return (
            self.router_logits_key,
            self.router_select_key,
            *tuple(self.selected_gate_up_keys.values()),
            self.selected_gate_up_prefill_key,
            self.activation_quant_key,
            self.selected_gate_up_prefill_f32_key,
            self.activation_quant_f32_key,
            self.selected_down_prefill_key,
            self.selected_down_prefill_q4_key,
            self.down_activation_quant_key,
            self.selected_silu_key,
            self.selected_dual_silu_key,
            *tuple(self.selected_down_keys.values()),
            *tuple(self.selected_weighted_down_keys.values()),
            self.routed_sum_key,
            self.routed_sum_rows_key,
            self.grouped_count_key,
            self.grouped_prefix_active_key,
            self.grouped_scatter_key,
            self.grouped_compact_key,
            self.grouped_compact_source_rows_key,
            self.mmq_tile_map_key,
            self.mmq64_tile_map_key,
            self.grouped_gather_key,
            *tuple(self.grouped_smallm_down_keys.values()),
            self.grouped_weighted_sum_key,
            self.grouped_weighted_sum_shared_add_key,
            self.shared_silu_key,
            self.add_key,
        )


@dataclass(frozen=True)
class LagunaMoEScratch:
    """Owned bounded buffers for the exact scalar or row-batched MoE chain."""

    plan: LagunaMoEKernelPlan
    max_rows: int
    router_logits: DeviceBuffer
    routing_scores: DeviceBuffer
    selection_scores: DeviceBuffer
    selected_experts: DeviceBuffer
    routing_weights: DeviceBuffer
    scaled_routing_weights: DeviceBuffer
    expert_gate: DeviceBuffer
    expert_up: DeviceBuffer
    expert_gate_up: DeviceBuffer
    activation_q8: DeviceBuffer
    expert_intermediate: DeviceBuffer
    expert_down: DeviceBuffer
    routed_output: DeviceBuffer
    grouped_counts: DeviceBuffer
    grouped_scatter_offsets: DeviceBuffer
    grouped_expert_start: DeviceBuffer
    grouped_expert_start_mmq32: DeviceBuffer
    grouped_mmq_total: DeviceBuffer
    grouped_active_experts: DeviceBuffer
    grouped_active_count: DeviceBuffer
    grouped_sorted_lanes: DeviceBuffer
    grouped_sorted_experts: DeviceBuffer
    grouped_sorted_weights: DeviceBuffer
    grouped_lane_to_row: DeviceBuffer
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
            self.expert_gate_up,
            self.activation_q8,
            self.expert_intermediate,
            self.expert_down,
            self.routed_output,
            self.grouped_counts,
            self.grouped_scatter_offsets,
            self.grouped_expert_start,
            self.grouped_expert_start_mmq32,
            self.grouped_mmq_total,
            self.grouped_active_experts,
            self.grouped_active_count,
            self.grouped_sorted_lanes,
            self.grouped_sorted_experts,
            self.grouped_sorted_weights,
            self.grouped_lane_to_row,
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
        "selected_gate_up_prefill": KernelKey(
            backend,
            "moe_linear",
            "gguf_q4_k_t16_v1",
            _SELECTED_DUAL_MMQ32_D4X3_VARIANT,
        ),
        "activation_quant": KernelKey(
            backend,
            "activation_quant",
            "q8_1_ds4x3",
            "bf16",
        ),
        "selected_gate_up_prefill_f32": KernelKey(
            backend,
            "moe_linear",
            "gguf_q4_k_t16_v1",
            _SELECTED_DUAL_MMQ64X32_D4X3_F32_VARIANT,
        ),
        "activation_quant_f32": KernelKey(
            backend,
            "activation_quant",
            "q8_1_ds4x3_f32",
            "bf16",
        ),
        "selected_down_prefill": KernelKey(
            backend,
            "moe_linear",
            "gguf_q6_k_t16_v1",
            _SELECTED_DOWN_MMQ64X32_D4X3_F32_VARIANT,
        ),
        "selected_down_prefill_q4": KernelKey(
            backend,
            "moe_linear",
            "gguf_q4_k_t16_v1",
            _SELECTED_DOWN_MMQ64X32_D4X3_F32_VARIANT,
        ),
        "down_activation_quant": KernelKey(
            backend,
            "activation_quant",
            "q8_1_ds4x3_f32",
            "bf16",
        ),
        "selected_silu": KernelKey(backend, "silu_mul_separate", "bf16", _SILU_VARIANT),
        "selected_dual_silu": KernelKey(
            backend, "silu_mul_dual", "bf16", _SILU_VARIANT
        ),
        "selected_down": KernelKey(
            backend,
            "moe_linear",
            "gguf_q6_k_t16_v1",
            _SELECTED_DOWN_VARIANT,
        ),
        "routed_sum": KernelKey(backend, "weighted_sum", "bf16", _WEIGHTED_SUM_VARIANT),
        "routed_sum_rows": KernelKey(backend, "weighted_sum", "bf16", "laguna_rows"),
        "grouped_count": KernelKey(
            backend, "moe_group_count", "generic", "selected_experts"
        ),
        "grouped_prefix_active": KernelKey(
            backend, "moe_group_prefix", "generic", "active_experts"
        ),
        "grouped_scatter": KernelKey(
            backend,
            "moe_group_scatter_gather",
            "generic",
            "bf16_lane_rows",
        ),
        "grouped_compact": KernelKey(
            backend, "moe_group_compact", "generic", "active_experts"
        ),
        "grouped_compact_source_rows": KernelKey(
            backend,
            "moe_group_compact",
            "generic",
            "active_experts_source_rows",
        ),
        "mmq_tile_map": KernelKey(
            backend,
            "moe_mmq_tile_map",
            "generic",
            "tile32",
        ),
        "mmq64_tile_map": KernelKey(
            backend,
            "moe_mmq_tile_map",
            "generic",
            "tile64",
        ),
        "grouped_gather": KernelKey(
            backend, "moe_gather_packed_hidden", "generic", "bf16_lanes"
        ),
        "grouped_weighted_sum": KernelKey(
            backend, "weighted_lanes_sum", "bf16", _WEIGHTED_SUM_VARIANT
        ),
        "grouped_weighted_sum_shared_add": KernelKey(
            backend,
            "weighted_lanes_sum+shared_add",
            "bf16",
            _WEIGHTED_SUM_VARIANT,
        ),
        "shared_silu": KernelKey(backend, "silu_mul_separate", "bf16", _SILU_VARIANT),
        "add": KernelKey(backend, "elementwise", "bf16", _ADD_VARIANT),
    }
    selected_gate_up_keys = MappingProxyType(
        {
            "gguf_q4_k_t16_v1": keys["selected_gate_up"],
            "gguf_iq2_xs": KernelKey(
                backend,
                "moe_linear",
                "gguf_iq2_xs",
                "selected_dual_silu_gemv_decode_bf16_bf16_out",
            ),
            "gguf_iq3_xxs": KernelKey(
                backend,
                "moe_linear",
                "gguf_iq3_xxs",
                "selected_dual_silu_gemv_decode_bf16_bf16_out",
            ),
        }
    )
    selected_down_keys = MappingProxyType(
        {
            "gguf_q4_k_t16_v1": KernelKey(
                backend, "moe_linear", "gguf_q4_k_t16_v1", _SELECTED_DOWN_VARIANT
            ),
            "gguf_q6_k_t16_v1": keys["selected_down"],
            "gguf_iq3_xxs": KernelKey(
                backend,
                "moe_linear",
                "gguf_iq3_xxs",
                "selected_gemv_decode_bf16_bf16_out",
            ),
            "gguf_iq4_xs": KernelKey(
                backend,
                "moe_linear",
                "gguf_iq4_xs",
                "selected_gemv_decode_bf16_bf16_out",
            ),
        }
    )
    functions = {name: _resolve_exact(key) for name, key in keys.items()}
    selected_gate_up_route_specs = {
        "gguf_q4_k_t16_v1": ("t16_dual", "tiles", "selected_gate_up"),
        "gguf_iq2_xs": ("raw_iq_dual_silu", "raw", "selected_gate_up_iq"),
        "gguf_iq3_xxs": ("raw_iq_dual_silu", "raw", "selected_gate_up_iq"),
    }
    selected_gate_up_routes = MappingProxyType(
        {
            quant: LagunaMoESelectedRoute(
                key=selected_gate_up_keys[quant],
                function=_resolve_exact(selected_gate_up_keys[quant]),
                abi=abi,
                allocation_name=allocation_name,
                library_key=library_key,
            )
            for quant, (abi, allocation_name, library_key) in selected_gate_up_route_specs.items()
        }
    )
    grouped_smallm_down_keys = MappingProxyType(
        {
            quant: KernelKey(
                backend,
                "moe_linear",
                quant,
                "selected_t16_grouped_smallm_bf16_bf16_out",
            )
            for quant in ("gguf_q4_k_t16_v1", "gguf_q6_k_t16_v1")
        }
    )
    grouped_smallm_downs = MappingProxyType(
        {quant: _resolve_exact(key) for quant, key in grouped_smallm_down_keys.items()}
    )
    selected_down_route_specs = {
        "gguf_q4_k_t16_v1": ("t16", "tiles", "selected_down"),
        "gguf_q6_k_t16_v1": ("t16", "tiles", "selected_down"),
        "gguf_iq3_xxs": ("raw_iq", "raw", "selected_down_iq"),
        "gguf_iq4_xs": ("raw_iq", "raw", "selected_down_iq"),
    }
    selected_down_routes = MappingProxyType(
        {
            quant: LagunaMoESelectedRoute(
                key=selected_down_keys[quant],
                function=_resolve_exact(selected_down_keys[quant]),
                abi=abi,
                allocation_name=allocation_name,
                library_key=library_key,
            )
            for quant, (abi, allocation_name, library_key) in selected_down_route_specs.items()
        }
    )
    selected_downs = MappingProxyType(
        {quant: route.function for quant, route in selected_down_routes.items()}
    )
    selected_weighted_down_keys = MappingProxyType(
        {
            "gguf_iq3_xxs": KernelKey(
                backend,
                "moe_linear",
                "gguf_iq3_xxs",
                _SELECTED_WEIGHTED_DOWN_VARIANT,
            )
        }
    )
    selected_weighted_down_routes = MappingProxyType(
        {
            quant: LagunaMoESelectedRoute(
                key=key,
                function=_resolve_exact(key),
                abi="raw_iq_weighted",
                allocation_name="raw",
                library_key="selected_down_iq",
            )
            for quant, key in selected_weighted_down_keys.items()
        }
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
        selected_gate_up_prefill_key=keys["selected_gate_up_prefill"],
        activation_quant_key=keys["activation_quant"],
        selected_gate_up_prefill_f32_key=keys[
            "selected_gate_up_prefill_f32"
        ],
        activation_quant_f32_key=keys["activation_quant_f32"],
        selected_down_prefill_key=keys["selected_down_prefill"],
        selected_down_prefill_q4_key=keys["selected_down_prefill_q4"],
        down_activation_quant_key=keys["down_activation_quant"],
        selected_gate_up_keys=selected_gate_up_keys,
        selected_gate_up_routes=selected_gate_up_routes,
        selected_silu_key=keys["selected_silu"],
        selected_dual_silu_key=keys["selected_dual_silu"],
        selected_down_key=keys["selected_down"],
        selected_down_keys=selected_down_keys,
        selected_down_routes=selected_down_routes,
        selected_weighted_down_keys=selected_weighted_down_keys,
        selected_weighted_down_routes=selected_weighted_down_routes,
        routed_sum_key=keys["routed_sum"],
        routed_sum_rows_key=keys["routed_sum_rows"],
        shared_silu_key=keys["shared_silu"],
        add_key=keys["add"],
        router_logits=functions["router_logits"],
        router_select=functions["router_select"],
        selected_gate_up=functions["selected_gate_up"],
        selected_gate_up_prefill=functions["selected_gate_up_prefill"],
        activation_quant=functions["activation_quant"],
        selected_gate_up_prefill_f32=functions[
            "selected_gate_up_prefill_f32"
        ],
        activation_quant_f32=functions["activation_quant_f32"],
        selected_down_prefill=functions["selected_down_prefill"],
        selected_down_prefill_q4=functions["selected_down_prefill_q4"],
        down_activation_quant=functions["down_activation_quant"],
        selected_silu=functions["selected_silu"],
        selected_dual_silu=functions["selected_dual_silu"],
        selected_down=functions["selected_down"],
        selected_downs=selected_downs,
        routed_sum=functions["routed_sum"],
        routed_sum_rows=functions["routed_sum_rows"],
        grouped_count_key=keys["grouped_count"],
        grouped_prefix_active_key=keys["grouped_prefix_active"],
        grouped_scatter_key=keys["grouped_scatter"],
        grouped_compact_key=keys["grouped_compact"],
        grouped_compact_source_rows_key=keys["grouped_compact_source_rows"],
        mmq_tile_map_key=keys["mmq_tile_map"],
        mmq64_tile_map_key=keys["mmq64_tile_map"],
        grouped_gather_key=keys["grouped_gather"],
        grouped_smallm_down_keys=grouped_smallm_down_keys,
        grouped_weighted_sum_key=keys["grouped_weighted_sum"],
        grouped_weighted_sum_shared_add_key=keys[
            "grouped_weighted_sum_shared_add"
        ],
        grouped_count=functions["grouped_count"],
        grouped_prefix_active=functions["grouped_prefix_active"],
        grouped_scatter=functions["grouped_scatter"],
        grouped_compact=functions["grouped_compact"],
        grouped_compact_source_rows=functions["grouped_compact_source_rows"],
        mmq_tile_map=functions["mmq_tile_map"],
        mmq64_tile_map=functions["mmq64_tile_map"],
        grouped_gather=functions["grouped_gather"],
        grouped_smallm_downs=grouped_smallm_downs,
        grouped_weighted_sum=functions["grouped_weighted_sum"],
        grouped_weighted_sum_shared_add=functions[
            "grouped_weighted_sum_shared_add"
        ],
        shared_silu=functions["shared_silu"],
        add=functions["add"],
    )


def _laguna_moe_scratch_sizes(
    plan: LagunaMoEKernelPlan,
    *,
    max_rows: int,
) -> tuple[int, ...]:
    rows = int(max_rows)
    if rows <= 0:
        raise ValueError("max_rows must be positive")
    h = plan.hidden_size
    e = plan.expert_count
    k = plan.top_k
    f = plan.expert_ffn_size
    sf = plan.shared_ffn_size
    return (
        rows * e * _F32_NBYTES,
        rows * e * _F32_NBYTES,
        rows * e * _F32_NBYTES,
        rows * k * _I64_NBYTES,
        rows * k * _F32_NBYTES,
        rows * k * _F32_NBYTES,
        rows * k * f * _BF16_NBYTES,
        rows * k * f * _BF16_NBYTES,
        rows * k * 2 * f * _BF16_NBYTES,
        (
            _Q8_1_DS4_RESIDUAL_PLANES
            * rows
            * (h // _Q8_1_MMQ_BLOCK)
            * _Q8_1_DS4_F32_BLOCK_BYTES
        ),
        rows * k * f * _BF16_NBYTES,
        rows * k * h * _BF16_NBYTES,
        rows * h * _BF16_NBYTES,
        e * 4,
        e * 4,
        (e + 1) * _I64_NBYTES,
        (e + 1) * _I64_NBYTES,
        _I64_NBYTES,
        e * _I64_NBYTES,
        _I64_NBYTES,
        rows * k * _I64_NBYTES,
        rows * k * _I64_NBYTES,
        rows * k * _F32_NBYTES,
        rows * k * _I64_NBYTES,
        rows * sf * _BF16_NBYTES,
        rows * sf * _BF16_NBYTES,
        rows * sf * _BF16_NBYTES,
        rows * h * _BF16_NBYTES,
        rows * h * _BF16_NBYTES,
    )


def laguna_moe_scratch_nbytes(
    plan: LagunaMoEKernelPlan,
    *,
    max_rows: int,
) -> int:
    """Return the exact bounded allocation bytes for one Laguna MoE scratch owner."""

    return sum(_laguna_moe_scratch_sizes(plan, max_rows=max_rows))


def allocate_laguna_moe_scratch(
    plan: LagunaMoEKernelPlan,
    *,
    max_rows: int = 1,
    runtime: HipRuntime | None = None,
) -> LagunaMoEScratch:
    """Allocate bounded router/routed/shared intermediates with failure cleanup."""

    rows = int(max_rows)
    sizes = _laguna_moe_scratch_sizes(plan, max_rows=rows)
    buffers: list[DeviceBuffer] = []
    try:
        buffers.extend(malloc(nbytes, runtime=runtime) for nbytes in sizes)
    except Exception:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)
        raise
    return LagunaMoEScratch(plan, rows, *buffers)


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
    dense_expected = {
        "ffn_gate_inp": ((e, h), LAYOUT_DENSE_F32, "f32", (e, h)),
        "exp_probs_b": ((e,), LAYOUT_DENSE_F32, "f32", (e,)),
    }
    for name, (shape, layout, quant, byte_shape) in dense_expected.items():
        weight = layer.weight(name)
        source = weight.spec.source
        if (
            source.shape != shape
            or source.byte_shape != byte_shape
            or weight.spec.layout != layout
            or weight.spec.quant_key != quant
        ):
            raise ValueError(f"{name} resident router contract mismatch")

    quant_blocks = {
        "gguf_q4_k": (_QK_K, 144),
        "gguf_q4_k_t16_v1": (_QK_K, 144),
        "gguf_q5_k": (_QK_K, 176),
        "gguf_q6_k_t16_v1": (_QK_K, 210),
        "gguf_q6_k": (_QK_K, 210),
        "gguf_q8_0": (32, 34),
        "gguf_iq2_xs": (_QK_K, 74),
        "gguf_iq3_xxs": (_QK_K, 98),
        "gguf_iq4_xs": (_QK_K, 136),
    }
    layouts_by_slot = {
        "ffn_gate_exps": {
            "gguf_q4_k_t16_v1": LAYOUT_GGUF_Q4_K_T16,
            "gguf_iq2_xs": LAYOUT_RAW_GGUF,
            "gguf_iq3_xxs": LAYOUT_RAW_GGUF,
        },
        "ffn_up_exps": {
            "gguf_q4_k_t16_v1": LAYOUT_GGUF_Q4_K_T16,
            "gguf_iq2_xs": LAYOUT_RAW_GGUF,
            "gguf_iq3_xxs": LAYOUT_RAW_GGUF,
        },
        "ffn_down_exps": {
            "gguf_q4_k_t16_v1": LAYOUT_GGUF_Q4_K_T16,
            "gguf_q6_k_t16_v1": LAYOUT_GGUF_Q6_K_T16,
            "gguf_iq3_xxs": LAYOUT_RAW_GGUF,
            "gguf_iq4_xs": LAYOUT_RAW_GGUF,
        },
        "ffn_gate_shexp": {
            "gguf_q4_k": LAYOUT_Q4_K_PACK8,
            "gguf_q5_k": LAYOUT_RAW_GGUF,
            "gguf_q6_k": LAYOUT_RAW_GGUF,
        },
        "ffn_up_shexp": {
            "gguf_q4_k": LAYOUT_Q4_K_PACK8,
            "gguf_q5_k": LAYOUT_RAW_GGUF,
            "gguf_q6_k": LAYOUT_RAW_GGUF,
        },
        "ffn_down_shexp": {
            "gguf_q4_k": LAYOUT_Q4_K_PACK8,
            "gguf_q6_k": LAYOUT_RAW_GGUF,
            "gguf_q8_0": LAYOUT_RAW_GGUF,
        },
    }
    shapes = {
        "ffn_gate_exps": (e, f, h),
        "ffn_up_exps": (e, f, h),
        "ffn_down_exps": (e, h, f),
        "ffn_gate_shexp": (sf, h),
        "ffn_up_shexp": (sf, h),
        "ffn_down_shexp": (h, sf),
    }
    for name, shape in shapes.items():
        weight = layer.weight(name)
        quant = weight.spec.quant_key
        expected_layout = layouts_by_slot[name].get(quant)
        block = quant_blocks.get(quant)
        if expected_layout is None or block is None:
            raise ValueError(f"{name} has no registered Laguna quant/layout route: {quant}")
        block_size, block_bytes = block
        byte_shape = (*shape[:-1], (shape[-1] // block_size) * block_bytes)
        if weight.spec.source.shape != shape or weight.spec.source.byte_shape != byte_shape:
            raise ValueError(
                f"{name} raw shape/stride must be {shape}/{byte_shape}, got "
                f"{weight.spec.source.shape}/{weight.spec.source.byte_shape}"
            )
        if weight.spec.layout != expected_layout:
            raise ValueError(
                f"{name} must use layout/quant {expected_layout}/{quant}, got "
                f"{weight.spec.layout}/{quant}"
            )

    gate_weight = layer.weight("ffn_gate_exps")
    up_weight = layer.weight("ffn_up_exps")
    if gate_weight.spec.quant_key != up_weight.spec.quant_key:
        raise ValueError("Laguna routed gate/up expert formats must match")
    try:
        gate_up_route = plan.selected_gate_up_routes[gate_weight.spec.quant_key]
    except KeyError as exc:
        raise ValueError(
            f"no Laguna selected gate/up route for {gate_weight.spec.quant_key!r}"
        ) from exc
    q4_t16_nbytes = e * (f // _T16_COLUMNS) * (h // _QK_K) * GGUF_Q4_K_TILE16_BLOCK_BYTES
    gate_up_nbytes = {
        "gguf_q4_k_t16_v1": q4_t16_nbytes,
        "gguf_iq2_xs": gate_weight.spec.source.nbytes,
        "gguf_iq3_xxs": gate_weight.spec.source.nbytes,
    }[gate_weight.spec.quant_key]
    for name in ("ffn_gate_exps", "ffn_up_exps"):
        allocation = layer.weight(name).allocation(gate_up_route.allocation_name)
        if allocation.buffer.nbytes != gate_up_nbytes:
            raise ValueError(f"{name} allocation does not match rank-3 expert stride")

    selected_down = layer.weight("ffn_down_exps")
    try:
        down_route = plan.selected_down_routes[selected_down.spec.quant_key]
    except KeyError as exc:
        raise ValueError(
            f"no Laguna selected down route for {selected_down.spec.quant_key!r}"
        ) from exc
    selected_down_nbytes = {
        "gguf_q4_k_t16_v1": (
            e * (h // _T16_COLUMNS) * (f // _QK_K) * GGUF_Q4_K_TILE16_BLOCK_BYTES
        ),
        "gguf_q6_k_t16_v1": (
            e * (h // _T16_COLUMNS) * (f // _QK_K) * GGUF_Q6_K_T16_BLOCK_BYTES
        ),
        "gguf_iq3_xxs": selected_down.spec.source.nbytes,
        "gguf_iq4_xs": selected_down.spec.source.nbytes,
    }[selected_down.spec.quant_key]
    if selected_down.allocation(down_route.allocation_name).buffer.nbytes != selected_down_nbytes:
        raise ValueError("ffn_down_exps allocation does not match rank-3 expert stride")


def _launch_selected_gate_up_t16(
    route: LagunaMoESelectedRoute,
    plan: LagunaMoEKernelPlan,
    gate_ptr: int,
    up_ptr: int,
    hidden_ptr: int,
    selected_ptr: int,
    scratch: LagunaMoEScratch,
    *,
    x_rows: int,
    lanes: int,
    stream: int,
    runtime: HipRuntime | None,
    libraries: Mapping[str, object] | None,
) -> None:
    route.function(
        hidden_ptr,
        selected_ptr,
        gate_ptr,
        up_ptr,
        scratch.expert_gate.ptr,
        scratch.expert_up.ptr,
        x_rows,
        lanes,
        plan.expert_count,
        plan.hidden_size,
        plan.expert_ffn_size,
        **_stage_kwargs(route.library_key, libraries, stream=stream, runtime=runtime),
    )
    plan.selected_silu(
        scratch.expert_gate.ptr,
        scratch.expert_up.ptr,
        scratch.expert_intermediate.ptr,
        lanes,
        plan.expert_ffn_size,
        **_stage_kwargs("selected_silu", libraries, stream=stream, runtime=runtime),
    )


def _launch_selected_gate_up_iq(
    route: LagunaMoESelectedRoute,
    plan: LagunaMoEKernelPlan,
    gate_ptr: int,
    up_ptr: int,
    hidden_ptr: int,
    selected_ptr: int,
    scratch: LagunaMoEScratch,
    *,
    x_rows: int,
    lanes: int,
    stream: int,
    runtime: HipRuntime | None,
    libraries: Mapping[str, object] | None,
) -> None:
    route.function(
        hidden_ptr,
        selected_ptr,
        gate_ptr,
        up_ptr,
        scratch.expert_intermediate.ptr,
        x_rows=x_rows,
        rows=lanes,
        num_experts=plan.expert_count,
        in_features=plan.hidden_size,
        out_features=plan.expert_ffn_size,
        **_stage_kwargs(route.library_key, libraries, stream=stream, runtime=runtime),
    )


_SELECTED_GATE_UP_ABIS = MappingProxyType(
    {
        "t16_dual": _launch_selected_gate_up_t16,
        "raw_iq_dual_silu": _launch_selected_gate_up_iq,
    }
)


def _launch_selected_gate_up(
    hidden_ptr: int,
    layer: LagunaGGUFResidentLayerWeights,
    scratch: LagunaMoEScratch,
    *,
    x_rows: int,
    lanes: int,
    stream: int,
    runtime: HipRuntime | None,
    libraries: Mapping[str, object] | None,
) -> None:
    plan = scratch.plan
    gate = layer.weight("ffn_gate_exps")
    up = layer.weight("ffn_up_exps")
    try:
        route = plan.selected_gate_up_routes[gate.spec.quant_key]
        launch = _SELECTED_GATE_UP_ABIS[route.abi]
    except KeyError as exc:
        raise ValueError(
            f"no Laguna selected gate/up route for {gate.spec.quant_key!r}"
        ) from exc
    launch(
        route,
        plan,
        gate.allocation(route.allocation_name).tensor.ptr,
        up.allocation(route.allocation_name).tensor.ptr,
        hidden_ptr,
        scratch.selected_experts.ptr,
        scratch,
        x_rows=x_rows,
        lanes=lanes,
        stream=stream,
        runtime=runtime,
        libraries=libraries,
    )


def _mmq32_static_upper_rows(lanes: int, expert_count: int) -> int:
    return _mmq_static_upper_rows(
        lanes,
        expert_count,
        tile_rows=_MMQ32_ROWS,
    )


def _mmq_static_upper_rows(
    lanes: int,
    expert_count: int,
    *,
    tile_rows: int,
) -> int:
    active_upper = min(int(lanes), int(expert_count))
    padded_upper = int(lanes) + (int(tile_rows) - 1) * active_upper
    return (
        (padded_upper + int(tile_rows) - 1) // int(tile_rows)
    ) * int(tile_rows)


def _launch_selected_gate_up_mmq32_d4x3(
    hidden_ptr: int,
    layer: LagunaGGUFResidentLayerWeights,
    scratch: LagunaMoEScratch,
    *,
    x_rows: int,
    lanes: int,
    stream: int,
    runtime: HipRuntime | None,
    libraries: Mapping[str, object] | None,
    residual_passes: int = 3,
    f32_wide: bool = False,
    split16: bool = False,
    rowvec: bool = False,
    wave_cols: bool = False,
    direct_wave_decode: bool = False,
) -> bool:
    """Run Q4T16 selected gate/up in compact order and retain its down metadata."""

    plan = scratch.plan
    gate = layer.weight("ffn_gate_exps")
    up = layer.weight("ffn_up_exps")
    down = layer.weight("ffn_down_exps")
    if (
        x_rows < _SELECTED_MMQ32_MIN_ROWS
        or gate.spec.quant_key != "gguf_q4_k_t16_v1"
        or up.spec.quant_key != "gguf_q4_k_t16_v1"
        or down.spec.quant_key not in plan.grouped_smallm_downs
    ):
        return False

    active_runtime = runtime or get_hip_runtime()
    mmq_total_rows = _mmq32_static_upper_rows(lanes, plan.expert_count)
    tile_capacity = mmq_total_rows // _MMQ32_ROWS
    if tile_capacity > lanes:
        raise ValueError("Laguna MMQ32 tile metadata exceeds bounded lane storage")

    plan.grouped_compact_source_rows(
        scratch.selected_experts.ptr,
        scratch.scaled_routing_weights.ptr,
        scratch.grouped_expert_start.ptr,
        scratch.grouped_active_experts.ptr,
        scratch.grouped_active_count.ptr,
        scratch.grouped_sorted_lanes.ptr,
        scratch.grouped_lane_to_row.ptr,
        scratch.grouped_sorted_weights.ptr,
        lanes,
        plan.expert_count,
        plan.top_k,
        **_stage_kwargs(
            "grouped_metadata", libraries, stream=stream, runtime=active_runtime
        ),
    )
    plan.mmq_tile_map(
        scratch.grouped_expert_start.ptr,
        scratch.grouped_expert_start_mmq32.ptr,
        scratch.grouped_sorted_experts.ptr,
        scratch.grouped_mmq_total.ptr,
        plan.expert_count,
        tile_capacity=tile_capacity,
        **_stage_kwargs(
            "grouped_metadata", libraries, stream=stream, runtime=active_runtime
        ),
    )
    activation_quant = (
        plan.activation_quant_f32 if f32_wide else plan.activation_quant
    )
    selected_prefill = (
        plan.selected_gate_up_prefill_f32
        if f32_wide
        else plan.selected_gate_up_prefill
    )
    activation_quant(
        hidden_ptr,
        scratch.activation_q8.ptr,
        x_rows,
        plan.hidden_size,
        **({"residual_passes": residual_passes} if f32_wide else {}),
        **({"split16": True} if split16 else {}),
        **_stage_kwargs(
            "selected_gate_up_prefill",
            libraries,
            stream=stream,
            runtime=active_runtime,
        ),
    )
    selected_prefill(
        scratch.activation_q8.ptr,
        scratch.grouped_lane_to_row.ptr,
        scratch.grouped_expert_start.ptr,
        scratch.grouped_expert_start_mmq32.ptr,
        scratch.grouped_sorted_experts.ptr,
        gate.allocation("tiles").tensor.ptr,
        up.allocation("tiles").tensor.ptr,
        scratch.expert_gate_up.ptr,
        lanes,
        x_rows,
        plan.hidden_size,
        plan.expert_ffn_size,
        plan.expert_ffn_size,
        plan.expert_count,
        mmq_total_rows,
        **({"residual_passes": residual_passes} if f32_wide else {}),
        **({"split16": True} if split16 else {}),
        **({"rowvec": True} if rowvec else {}),
        **({"wave_cols": True} if wave_cols else {}),
        **({"direct_wave_decode": True} if direct_wave_decode else {}),
        **_stage_kwargs(
            "selected_gate_up_prefill",
            libraries,
            stream=stream,
            runtime=active_runtime,
        ),
    )
    plan.selected_dual_silu(
        scratch.expert_gate_up.ptr,
        scratch.expert_intermediate.ptr,
        lanes,
        plan.expert_ffn_size,
        **_stage_kwargs(
            "selected_silu", libraries, stream=stream, runtime=active_runtime
        ),
    )
    return True


def _launch_selected_down_t16(
    route: LagunaMoESelectedRoute,
    plan: LagunaMoEKernelPlan,
    weight_ptr: int,
    scratch: LagunaMoEScratch,
    *,
    lanes: int,
    stream: int,
    runtime: HipRuntime | None,
    libraries: Mapping[str, object] | None,
) -> None:
    route.function(
        scratch.expert_intermediate.ptr,
        scratch.selected_experts.ptr,
        weight_ptr,
        scratch.expert_down.ptr,
        lanes,
        lanes,
        plan.expert_count,
        plan.expert_ffn_size,
        plan.hidden_size,
        **_stage_kwargs(route.library_key, libraries, stream=stream, runtime=runtime),
    )


def _launch_selected_down_iq(
    route: LagunaMoESelectedRoute,
    plan: LagunaMoEKernelPlan,
    weight_ptr: int,
    scratch: LagunaMoEScratch,
    *,
    lanes: int,
    stream: int,
    runtime: HipRuntime | None,
    libraries: Mapping[str, object] | None,
) -> None:
    route.function(
        scratch.expert_intermediate.ptr,
        scratch.selected_experts.ptr,
        weight_ptr,
        scratch.expert_down.ptr,
        x_rows=lanes,
        rows=lanes,
        num_experts=plan.expert_count,
        in_features=plan.expert_ffn_size,
        out_features=plan.hidden_size,
        **_stage_kwargs(route.library_key, libraries, stream=stream, runtime=runtime),
    )


_SELECTED_DOWN_ABIS = MappingProxyType(
    {"t16": _launch_selected_down_t16, "raw_iq": _launch_selected_down_iq}
)


def _launch_weighted_selected_down_iq(
    route: LagunaMoESelectedRoute,
    plan: LagunaMoEKernelPlan,
    weight_ptr: int,
    scratch: LagunaMoEScratch,
    *,
    tokens: int,
    stream: int,
    runtime: HipRuntime | None,
    libraries: Mapping[str, object] | None,
) -> None:
    route.function(
        scratch.expert_intermediate.ptr,
        scratch.selected_experts.ptr,
        scratch.scaled_routing_weights.ptr,
        weight_ptr,
        scratch.routed_output.ptr,
        tokens=tokens,
        top_k=plan.top_k,
        num_experts=plan.expert_count,
        in_features=plan.expert_ffn_size,
        out_features=plan.hidden_size,
        **_stage_kwargs(route.library_key, libraries, stream=stream, runtime=runtime),
    )


_SELECTED_WEIGHTED_DOWN_ABIS = MappingProxyType(
    {"raw_iq_weighted": _launch_weighted_selected_down_iq}
)


def _launch_weighted_selected_down(
    layer: LagunaGGUFResidentLayerWeights,
    scratch: LagunaMoEScratch,
    *,
    tokens: int,
    stream: int,
    runtime: HipRuntime | None,
    libraries: Mapping[str, object] | None,
) -> bool:
    # The composite is a c=1 decode schedule. Bulk rows retain the independent
    # selected projection plus row-wise weighted-sum fallback until measured.
    if tokens != 1:
        return False
    plan = scratch.plan
    weight = layer.weight("ffn_down_exps")
    try:
        route = plan.selected_weighted_down_routes[weight.spec.quant_key]
        launch = _SELECTED_WEIGHTED_DOWN_ABIS[route.abi]
    except KeyError:
        return False
    launch(
        route,
        plan,
        weight.allocation(route.allocation_name).tensor.ptr,
        scratch,
        tokens=tokens,
        stream=stream,
        runtime=runtime,
        libraries=libraries,
    )
    return True


def _launch_selected_down(
    layer: LagunaGGUFResidentLayerWeights,
    scratch: LagunaMoEScratch,
    *,
    lanes: int,
    stream: int,
    runtime: HipRuntime | None,
    libraries: Mapping[str, object] | None,
) -> None:
    plan = scratch.plan
    weight = layer.weight("ffn_down_exps")
    try:
        route = plan.selected_down_routes[weight.spec.quant_key]
        launch = _SELECTED_DOWN_ABIS[route.abi]
    except KeyError as exc:
        raise ValueError(
            f"no Laguna selected down route for {weight.spec.quant_key!r}"
        ) from exc
    launch(
        route,
        plan,
        weight.allocation(route.allocation_name).tensor.ptr,
        scratch,
        lanes=lanes,
        stream=stream,
        runtime=runtime,
        libraries=libraries,
    )


def _launch_selected_down_mmq64x32_d4x3_f32(
    layer: LagunaGGUFResidentLayerWeights,
    scratch: LagunaMoEScratch,
    *,
    lanes: int,
    stream: int,
    runtime: HipRuntime | None,
    libraries: Mapping[str, object] | None,
    residual_passes: int,
    rowvec: bool = False,
    wave_cols_q4: bool = False,
    direct_wave_decode_q4: bool = False,
    q6_tile_rows: int = _MMQ32_ROWS,
) -> bool:
    """Quantize compact post-SiLU rows and run range-safe Q6T16 MMQ down."""

    plan = scratch.plan
    weight = layer.weight("ffn_down_exps")
    if weight.spec.quant_key == "gguf_q6_k_t16_v1":
        selected_down_prefill = plan.selected_down_prefill
    elif (
        weight.spec.quant_key == "gguf_q4_k_t16_v1"
        and residual_passes == 1
    ):
        selected_down_prefill = plan.selected_down_prefill_q4
    else:
        return False
    wave_cols = (
        wave_cols_q4
        and weight.spec.quant_key == "gguf_q4_k_t16_v1"
    )
    active_runtime = runtime or get_hip_runtime()
    tile_rows = (
        q6_tile_rows
        if weight.spec.quant_key == "gguf_q6_k_t16_v1"
        else _MMQ32_ROWS
    )
    mmq_total_rows = _mmq_static_upper_rows(
        lanes,
        plan.expert_count,
        tile_rows=tile_rows,
    )
    if tile_rows == _MMQ64_ROWS:
        tile_capacity = mmq_total_rows // tile_rows
        if tile_capacity > lanes:
            raise ValueError("Laguna MMQ64 tile metadata exceeds bounded lane storage")
        plan.mmq64_tile_map(
            scratch.grouped_expert_start.ptr,
            scratch.grouped_expert_start_mmq32.ptr,
            scratch.grouped_sorted_experts.ptr,
            scratch.grouped_mmq_total.ptr,
            plan.expert_count,
            tile_capacity=tile_capacity,
            **_stage_kwargs(
                "grouped_metadata",
                libraries,
                stream=stream,
                runtime=active_runtime,
            ),
        )
    plan.down_activation_quant(
        scratch.expert_intermediate.ptr,
        scratch.expert_gate_up.ptr,
        lanes,
        plan.expert_ffn_size,
        residual_passes=residual_passes,
        **_stage_kwargs(
            "selected_gate_up_prefill",
            libraries,
            stream=stream,
            runtime=active_runtime,
        ),
    )
    selected_down_prefill(
        scratch.expert_gate_up.ptr,
        scratch.grouped_expert_start.ptr,
        scratch.grouped_expert_start_mmq32.ptr,
        scratch.grouped_sorted_experts.ptr,
        weight.allocation("tiles").tensor.ptr,
        scratch.expert_down.ptr,
        lanes,
        plan.expert_ffn_size,
        plan.hidden_size,
        plan.expert_count,
        mmq_total_rows,
        **(
            {
                "residual_passes": residual_passes,
                "tile_rows": tile_rows,
            }
            if weight.spec.quant_key == "gguf_q6_k_t16_v1"
            else {}
        ),
        rowvec=rowvec,
        **(
            {"wave_cols": wave_cols}
            if weight.spec.quant_key == "gguf_q4_k_t16_v1"
            else {}
        ),
        **(
            {"direct_wave_decode": True}
            if direct_wave_decode_q4
            and weight.spec.quant_key == "gguf_q4_k_t16_v1"
            else {}
        ),
        **_stage_kwargs(
            "selected_gate_up_prefill",
            libraries,
            stream=stream,
            runtime=active_runtime,
        ),
    )
    return True


def _launch_grouped_smallm_down(
    layer: LagunaGGUFResidentLayerWeights,
    scratch: LagunaMoEScratch,
    *,
    tokens: int,
    lanes: int,
    stream: int,
    runtime: HipRuntime | None,
    libraries: Mapping[str, object] | None,
    defer_weighted_sum: bool = False,
    compact_input_ready: bool = False,
) -> None:
    """Group lane-order intermediates on device and run exact small-M down."""

    plan = scratch.plan
    weight = layer.weight("ffn_down_exps")
    try:
        grouped_down = plan.grouped_smallm_downs[weight.spec.quant_key]
    except KeyError as exc:
        raise ValueError(
            f"no Laguna grouped small-M selected-down route for {weight.spec.quant_key!r}"
        ) from exc
    active_runtime = runtime or get_hip_runtime()
    grouped_input_ptr = scratch.expert_intermediate.ptr
    if not compact_input_ready:
        plan.grouped_compact(
            scratch.selected_experts.ptr,
            scratch.scaled_routing_weights.ptr,
            scratch.grouped_expert_start.ptr,
            scratch.grouped_active_experts.ptr,
            scratch.grouped_active_count.ptr,
            scratch.grouped_sorted_lanes.ptr,
            scratch.grouped_sorted_experts.ptr,
            scratch.grouped_sorted_weights.ptr,
            lanes,
            plan.expert_count,
            **_stage_kwargs(
                "grouped_metadata", libraries, stream=stream, runtime=active_runtime
            ),
        )
        plan.grouped_gather(
            scratch.expert_intermediate.ptr,
            scratch.grouped_sorted_lanes.ptr,
            scratch.expert_gate.ptr,
            lanes * plan.expert_ffn_size,
            lanes,
            1,
            plan.expert_ffn_size,
            **_stage_kwargs(
                "grouped_gather", libraries, stream=stream, runtime=active_runtime
            ),
        )
        grouped_input_ptr = scratch.expert_gate.ptr
    grouped_down(
        grouped_input_ptr,
        scratch.grouped_expert_start.ptr,
        scratch.grouped_active_experts.ptr,
        scratch.grouped_active_count.ptr,
        weight.allocation("tiles").tensor.ptr,
        scratch.expert_down.ptr,
        lanes,
        plan.expert_ffn_size,
        plan.hidden_size,
        plan.expert_count,
        **_stage_kwargs(
            "grouped_down", libraries, stream=stream, runtime=active_runtime
        ),
    )
    if not defer_weighted_sum:
        plan.grouped_weighted_sum(
            scratch.expert_down.ptr,
            scratch.grouped_sorted_weights.ptr,
            scratch.grouped_sorted_lanes.ptr,
            scratch.grouped_lane_to_row.ptr,
            scratch.routed_output.ptr,
            tokens,
            plan.top_k,
            plan.hidden_size,
            **_stage_kwargs(
                "grouped_weighted_sum",
                libraries,
                stream=stream,
                runtime=active_runtime,
            ),
        )


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
    if scratch.max_rows < 1:
        raise ValueError("Laguna MoE scratch cannot execute one row")
    validate_laguna_moe_layer(layer, plan)
    h, e, k, sf = (
        plan.hidden_size,
        plan.expert_count,
        plan.top_k,
        plan.shared_ffn_size,
    )
    router = layer.weight("ffn_gate_inp").allocation("raw").tensor.ptr
    correction = layer.weight("exp_probs_b").allocation("raw").tensor.ptr

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
    _launch_selected_gate_up(
        hidden_bf16_ptr,
        layer,
        scratch,
        x_rows=1,
        lanes=k,
        stream=stream,
        runtime=runtime,
        libraries=libraries,
    )
    routed_down_weighted = _launch_weighted_selected_down(
        layer,
        scratch,
        tokens=1,
        stream=stream,
        runtime=runtime,
        libraries=libraries,
    )
    if not routed_down_weighted:
        _launch_selected_down(
            layer,
            scratch,
            lanes=k,
            stream=stream,
            runtime=runtime,
            libraries=libraries,
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
    shared_pair = launch_gguf_linear_pair(
        shared_gate,
        shared_up,
        hidden_bf16_ptr,
        scratch.shared_gate.ptr,
        scratch.shared_up.ptr,
        1,
        h,
        sf,
        backend=plan.backend,
        stream=stream,
        libraries=libraries,
        runtime=runtime,
        use_wmma_prefill=False,
        use_gemv_decode=True,
        registered_decode_only=True,
    )
    if not shared_pair:
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
            use_gemv_decode=True,
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
            use_gemv_decode=True,
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
        use_gemv_decode=True,
    )
    plan.add(
        scratch.routed_output.ptr,
        scratch.shared_output.ptr,
        scratch.output.ptr,
        h,
        **_stage_kwargs("add", libraries, stream=stream, runtime=runtime),
    )
    return scratch.output


def run_laguna_moe_rows(
    hidden_bf16_ptr: int,
    layer: LagunaGGUFResidentLayerWeights,
    scratch: LagunaMoEScratch,
    *,
    rows: int,
    selected_down_mode: str = "direct",
    selected_gate_up_mode: str = "direct",
    dense_q4_prefill_mode: str = "retained",
    stream: int = 0,
    runtime: HipRuntime | None = None,
    libraries: Mapping[str, object] | None = None,
) -> DeviceBuffer:
    """Run exact row-batched sigmoid MoE with contiguous top-k expert lanes."""

    plan = scratch.plan
    tokens = int(rows)
    if selected_down_mode not in _SELECTED_DOWN_MODES:
        raise ValueError(
            "selected_down_mode must be one of "
            f"{tuple(sorted(_SELECTED_DOWN_MODES))}"
        )
    if selected_gate_up_mode not in _SELECTED_GATE_UP_MODES:
        raise ValueError(
            "selected_gate_up_mode must be one of "
            f"{tuple(sorted(_SELECTED_GATE_UP_MODES))}"
        )
    if dense_q4_prefill_mode not in _DENSE_Q4_PREFILL_MODES:
        raise ValueError("unsupported Laguna dense/shared Q4 prefill mode")
    use_q4_pack8_wmma = dense_q4_prefill_mode == "wmma_pack8"
    if tokens <= 0 or tokens > scratch.max_rows:
        raise ValueError(f"rows must be within [1, {scratch.max_rows}]")
    validate_laguna_moe_layer(layer, plan)
    h, e, k, sf = (
        plan.hidden_size,
        plan.expert_count,
        plan.top_k,
        plan.shared_ffn_size,
    )
    lanes = tokens * k
    router = layer.weight("ffn_gate_inp").allocation("raw").tensor.ptr
    correction = layer.weight("exp_probs_b").allocation("raw").tensor.ptr

    plan.router_logits(
        hidden_bf16_ptr,
        router,
        scratch.router_logits.ptr,
        tokens,
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
        tokens,
        e,
        k,
        plan.routed_scaling_factor,
        **_stage_kwargs("router_select", libraries, stream=stream, runtime=runtime),
    )
    compact_gate_up = False
    gate_up_f32_config = {
        "mmq64x32_d4_f32": (1, False, False, False, False),
        "mmq128x32_d8_f32": (1, True, False, False, False),
        "mmq128x32_d8_f32_rowvec": (1, True, True, False, False),
        "mmq128x32_d8_f32_wavecols": (1, True, True, True, False),
        "mmq128x32_d8_f32_wavecols_direct": (
            1,
            True,
            True,
            True,
            True,
        ),
        "mmq64x32_d4x2_f32": (2, False, False, False, False),
        "mmq64x32_d4x3_f32": (3, False, False, False, False),
    }.get(selected_gate_up_mode)
    if selected_gate_up_mode == "mmq32_d4x3" or gate_up_f32_config is not None:
        (
            gate_up_f32_passes,
            gate_up_split16,
            gate_up_rowvec,
            gate_up_wave_cols,
            gate_up_direct_wave_decode,
        ) = gate_up_f32_config or (3, False, False, False, False)
        compact_gate_up = _launch_selected_gate_up_mmq32_d4x3(
            hidden_bf16_ptr,
            layer,
            scratch,
            x_rows=tokens,
            lanes=lanes,
            stream=stream,
            runtime=runtime,
            libraries=libraries,
            residual_passes=gate_up_f32_passes,
            f32_wide=gate_up_f32_config is not None,
            split16=gate_up_split16,
            rowvec=gate_up_rowvec,
            wave_cols=gate_up_wave_cols,
            direct_wave_decode=gate_up_direct_wave_decode,
        )
    if not compact_gate_up:
        _launch_selected_gate_up(
            hidden_bf16_ptr,
            layer,
            scratch,
            x_rows=tokens,
            lanes=lanes,
            stream=stream,
            runtime=runtime,
            libraries=libraries,
        )
    mmq_down_config = {
        "mmq64x32_d4_f32": (1, False, False, False, _MMQ32_ROWS),
        "mmq64x32_d4_f32_rowvec": (1, True, False, False, _MMQ32_ROWS),
        "mmq64x32_d4_f32_wavecols_q4": (
            1,
            True,
            True,
            False,
            _MMQ32_ROWS,
        ),
        "mmq64x32_d4_f32_wavecols_direct_q4": (
            1,
            True,
            True,
            True,
            _MMQ32_ROWS,
        ),
        "mmq64x64_d4_f32_q6_wavecols_direct_q4": (
            1,
            True,
            True,
            True,
            _MMQ64_ROWS,
        ),
        "mmq64x32_d4x2_f32": (2, False, False, False, _MMQ32_ROWS),
        "mmq64x32_d4x3_f32": (3, False, False, False, _MMQ32_ROWS),
    }.get(selected_down_mode)
    (
        mmq_down_passes,
        mmq_down_rowvec,
        mmq_down_wave_cols_q4,
        mmq_down_direct_wave_decode_q4,
        mmq_down_q6_tile_rows,
    ) = mmq_down_config or (None, False, False, False, _MMQ32_ROWS)
    use_mmq_down = (
        compact_gate_up
        and mmq_down_passes is not None
        and _launch_selected_down_mmq64x32_d4x3_f32(
            layer,
            scratch,
            lanes=lanes,
            stream=stream,
            runtime=runtime,
            libraries=libraries,
            residual_passes=mmq_down_passes,
            rowvec=mmq_down_rowvec,
            wave_cols_q4=mmq_down_wave_cols_q4,
            direct_wave_decode_q4=mmq_down_direct_wave_decode_q4,
            q6_tile_rows=mmq_down_q6_tile_rows,
        )
    )
    use_grouped_fused_combine = selected_down_mode == "grouped_smallm_fused" or (
        selected_down_mode == "adaptive_grouped_smallm_fused"
        and tokens >= _GROUPED_SMALLM_MIN_ROWS
    )
    use_grouped_smallm = compact_gate_up or selected_down_mode == "grouped_smallm" or (
        selected_down_mode == "adaptive_grouped_smallm"
        and tokens >= _GROUPED_SMALLM_MIN_ROWS
    ) or use_grouped_fused_combine
    if use_mmq_down:
        plan.grouped_weighted_sum(
            scratch.expert_down.ptr,
            scratch.grouped_sorted_weights.ptr,
            scratch.grouped_sorted_lanes.ptr,
            scratch.grouped_lane_to_row.ptr,
            scratch.routed_output.ptr,
            tokens,
            plan.top_k,
            plan.hidden_size,
            **_stage_kwargs(
                "grouped_weighted_sum",
                libraries,
                stream=stream,
                runtime=runtime,
            ),
        )
    elif use_grouped_smallm:
        _launch_grouped_smallm_down(
            layer,
            scratch,
            tokens=tokens,
            lanes=lanes,
            stream=stream,
            runtime=runtime,
            libraries=libraries,
            defer_weighted_sum=use_grouped_fused_combine,
            compact_input_ready=compact_gate_up,
        )
    else:
        _launch_selected_down(
            layer,
            scratch,
            lanes=lanes,
            stream=stream,
            runtime=runtime,
            libraries=libraries,
        )
        plan.routed_sum_rows(
            scratch.expert_down.ptr,
            scratch.scaled_routing_weights.ptr,
            scratch.routed_output.ptr,
            tokens,
            k,
            h,
            **_stage_kwargs(
                "routed_sum_rows", libraries, stream=stream, runtime=runtime
            ),
        )

    shared_gate = layer.weight("ffn_gate_shexp")
    shared_up = layer.weight("ffn_up_shexp")
    shared_down = layer.weight("ffn_down_shexp")
    launch_gguf_linear(
        shared_gate,
        hidden_bf16_ptr,
        scratch.shared_gate.ptr,
        tokens,
        h,
        sf,
        backend=plan.backend,
        stream=stream,
        runtime=runtime,
        libraries=libraries,
        use_wmma_prefill=False,
        use_gemv_decode=tokens == 1,
        use_q4_pack8_wmma=use_q4_pack8_wmma,
    )
    launch_gguf_linear(
        shared_up,
        hidden_bf16_ptr,
        scratch.shared_up.ptr,
        tokens,
        h,
        sf,
        backend=plan.backend,
        stream=stream,
        runtime=runtime,
        libraries=libraries,
        use_wmma_prefill=False,
        use_gemv_decode=tokens == 1,
        use_q4_pack8_wmma=use_q4_pack8_wmma,
    )
    plan.shared_silu(
        scratch.shared_gate.ptr,
        scratch.shared_up.ptr,
        scratch.shared_intermediate.ptr,
        tokens,
        sf,
        **_stage_kwargs("shared_silu", libraries, stream=stream, runtime=runtime),
    )
    launch_gguf_linear(
        shared_down,
        scratch.shared_intermediate.ptr,
        scratch.shared_output.ptr,
        tokens,
        sf,
        h,
        backend=plan.backend,
        stream=stream,
        runtime=runtime,
        libraries=libraries,
        use_wmma_prefill=False,
        use_gemv_decode=tokens == 1,
        use_q4_pack8_wmma=use_q4_pack8_wmma,
    )
    if use_grouped_fused_combine:
        plan.grouped_weighted_sum_shared_add(
            scratch.expert_down.ptr,
            scratch.grouped_sorted_weights.ptr,
            scratch.grouped_sorted_lanes.ptr,
            scratch.grouped_lane_to_row.ptr,
            scratch.shared_output.ptr,
            scratch.output.ptr,
            tokens,
            k,
            h,
            **_stage_kwargs(
                "grouped_weighted_sum_shared_add",
                libraries,
                stream=stream,
                runtime=runtime,
            ),
        )
    else:
        plan.add(
            scratch.routed_output.ptr,
            scratch.shared_output.ptr,
            scratch.output.ptr,
            tokens * h,
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
    "laguna_moe_scratch_nbytes",
    "resolve_laguna_moe_plan",
    "resolve_laguna_selected_down_mode",
    "resolve_laguna_dense_q4_prefill_mode",
    "resolve_laguna_selected_gate_up_mode",
    "run_laguna_moe_c1",
    "run_laguna_moe_rows",
    "validate_laguna_moe_layer",
]
