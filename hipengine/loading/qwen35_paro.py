"""Qwen3.5/PARO checkpoint layout validation."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any

from safetensors import safe_open

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.hip import HipRuntime
from hipengine.core.tensor import Tensor
from hipengine.loading.materialize import (
    DeviceTensorAllocation,
    DeviceWeightMap,
    float_array_to_bf16_bits,
    load_host_array_to_device,
    load_host_array_to_device_as_dtype,
    load_tensor_info_to_device,
)
from hipengine.loading.safetensors import MissingTensorError, TensorInfo, WeightIndex

ROOT_PREFIXES = ("model.language_model.", "language_model.", "model.")


@dataclass(frozen=True)
class Qwen35ParoConfig:
    architecture: str
    num_hidden_layers: int
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    shared_expert_intermediate_size: int
    layer_types: tuple[str, ...]
    quant_method: str
    vocab_size: int = 0
    rms_norm_eps: float = 1.0e-6
    rope_theta: float = 1000000.0
    rotary_dim: int = 0
    linear_num_key_heads: int = 0
    linear_num_value_heads: int = 0
    linear_key_head_dim: int = 0
    linear_value_head_dim: int = 0
    linear_conv_kernel_dim: int = 0


@dataclass(frozen=True)
class Qwen35ParoLayoutValidation:
    config: Qwen35ParoConfig
    present: tuple[str, ...]
    missing: tuple[str, ...]
    shape_errors: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.missing and not self.shape_errors

    def raise_for_errors(self) -> None:
        if self.passed:
            return
        parts: list[str] = []
        if self.missing:
            preview = ", ".join(self.missing[:8])
            more = "" if len(self.missing) <= 8 else f" (+{len(self.missing) - 8} more)"
            parts.append(f"missing tensors: {preview}{more}")
        if self.shape_errors:
            preview = "; ".join(self.shape_errors[:4])
            more = "" if len(self.shape_errors) <= 4 else f" (+{len(self.shape_errors) - 4} more)"
            parts.append(f"shape errors: {preview}{more}")
        raise MissingTensorError("; ".join(parts))


@dataclass(frozen=True)
class Qwen35ParoLayerDeviceWeights:
    """Materialized normalized device weights for one Qwen3.5/PARO layer slice."""

    config: Qwen35ParoConfig
    layer_id: int
    weights: DeviceWeightMap

    def tensor(self, name: str) -> Tensor:
        return self.weights[normalize_qwen35_weight_name(name)]

    def allocation(self, name: str) -> DeviceTensorAllocation:
        return self.weights.allocation(normalize_qwen35_weight_name(name))

    def free(self, *, runtime: HipRuntime | None = None) -> None:
        self.weights.free(runtime=runtime)


def normalize_qwen35_weight_name(name: str) -> str:
    for prefix in ROOT_PREFIXES:
        if name.startswith(prefix):
            return name.removeprefix(prefix)
    return name


def qwen35_paro_config_from_hf(config: dict[str, Any]) -> Qwen35ParoConfig:
    text = config.get("text_config") if isinstance(config.get("text_config"), dict) else config
    model_type = str(text.get("model_type", config.get("model_type", "qwen3_5_moe")))
    if "qwen3_5_moe" not in model_type and "qwen3_5_text" not in model_type:
        raise ValueError(f"expected Qwen3.5 MoE/text model_type, got {model_type!r}")
    architectures = config.get("architectures") or text.get("architectures") or ()
    architecture = str(architectures[0]) if architectures else "Qwen3_5MoeForConditionalGeneration"
    num_layers = int(text["num_hidden_layers"])
    layer_types = tuple(text.get("layer_types") or ("full_attention",) * num_layers)
    if len(layer_types) != num_layers:
        raise ValueError(f"layer_types has {len(layer_types)} entries for {num_layers} layers")
    quant = config.get("quantization_config") or text.get("quantization_config") or {}
    quant_method = str(quant.get("quant_method", ""))
    hidden_size = int(text["hidden_size"])
    num_attention_heads = int(text.get("num_attention_heads", 0) or 0)
    num_key_value_heads = int(text.get("num_key_value_heads", num_attention_heads) or 0)
    head_dim = int(text.get("head_dim", (hidden_size // num_attention_heads) if num_attention_heads else 0) or 0)
    rope_parameters = text.get("rope_parameters") if isinstance(text.get("rope_parameters"), dict) else {}
    partial_rotary_factor = text.get("partial_rotary_factor", rope_parameters.get("partial_rotary_factor", 1.0))
    rotary_dim = int(head_dim * float(partial_rotary_factor)) if head_dim else 0
    return Qwen35ParoConfig(
        architecture=architecture,
        num_hidden_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        num_experts=int(text.get("num_experts", 0) or 0),
        num_experts_per_tok=int(text.get("num_experts_per_tok", 0) or 0),
        moe_intermediate_size=int(text.get("moe_intermediate_size", text.get("intermediate_size", 0)) or 0),
        shared_expert_intermediate_size=int(text.get("shared_expert_intermediate_size", 0) or 0),
        layer_types=layer_types,
        quant_method=quant_method,
        vocab_size=int(text.get("vocab_size", 0) or 0),
        rms_norm_eps=float(text.get("rms_norm_eps", 1.0e-6) or 1.0e-6),
        rope_theta=float(text.get("rope_theta", rope_parameters.get("rope_theta", 1000000.0)) or 1000000.0),
        rotary_dim=rotary_dim,
        linear_num_key_heads=int(text.get("linear_num_key_heads", 0) or 0),
        linear_num_value_heads=int(text.get("linear_num_value_heads", 0) or 0),
        linear_key_head_dim=int(text.get("linear_key_head_dim", 0) or 0),
        linear_value_head_dim=int(text.get("linear_value_head_dim", 0) or 0),
        linear_conv_kernel_dim=int(text.get("linear_conv_kernel_dim", 0) or 0),
    )


def required_full_attention_c1_tensor_names(*, layer_id: int) -> tuple[str, ...]:
    prefix = f"layers.{layer_id}.self_attn"
    names = [
        f"layers.{layer_id}.input_layernorm.weight",
        f"{prefix}.q_norm.weight",
        f"{prefix}.k_norm.weight",
    ]
    for proj in ("q_proj", "k_proj", "v_proj"):
        base = f"{prefix}.{proj}"
        names.extend(
            (
                f"{base}.qweight",
                f"{base}.qzeros",
                f"{base}.scales",
                f"{base}.theta",
                f"{base}.pairs",
                f"{base}.channel_scales",
            )
        )
    base = f"{prefix}.o_proj"
    names.extend(
        (
            f"{base}.qweight",
            f"{base}.qzeros",
            f"{base}.scales",
            f"{base}.theta",
            f"{base}.pairs",
            f"{base}.channel_scales",
        )
    )
    return tuple(names)


def required_full_attention_moe_c1_tensor_names(*, layer_id: int, num_experts: int) -> tuple[str, ...]:
    return required_full_attention_c1_tensor_names(layer_id=layer_id) + required_moe_c1_tensor_names(
        layer_id=layer_id,
        num_experts=num_experts,
    )


def required_linear_attention_c1_tensor_names(*, layer_id: int) -> tuple[str, ...]:
    prefix = f"layers.{layer_id}.linear_attn"
    names = [f"layers.{layer_id}.input_layernorm.weight"]
    for proj in ("in_proj_qkv", "in_proj_z", "out_proj"):
        base = f"{prefix}.{proj}"
        names.extend(
            (
                f"{base}.qweight",
                f"{base}.qzeros",
                f"{base}.scales",
                f"{base}.theta",
                f"{base}.pairs",
                f"{base}.channel_scales",
            )
        )
    names.extend(
        (
            f"{prefix}.in_proj_a.weight",
            f"{prefix}.in_proj_b.weight",
            f"{prefix}.conv1d.weight",
            f"{prefix}.A_log",
            f"{prefix}.dt_bias",
            f"{prefix}.norm.weight",
        )
    )
    return tuple(names)


def required_linear_attention_moe_c1_tensor_names(*, layer_id: int, num_experts: int) -> tuple[str, ...]:
    return required_linear_attention_c1_tensor_names(layer_id=layer_id) + required_moe_c1_tensor_names(
        layer_id=layer_id,
        num_experts=num_experts,
    )


def prepared_moe_c1_tensor_names(*, layer_id: int) -> tuple[str, ...]:
    prefix = f"layers.{layer_id}.mlp"
    experts = f"{prefix}.experts"
    names = [f"{prefix}.router_shared_gate.weight"]
    for proj in ("gate", "up", "down"):
        names.extend(
            (
                f"{experts}.stacked_{proj}_qweight",
                f"{experts}.stacked_{proj}_qweight_pack8_decode",
                f"{experts}.stacked_{proj}_qzeros",
                f"{experts}.stacked_{proj}_scales",
            )
        )
    shared = f"{prefix}.shared_expert"
    names.extend(
        (
            f"{shared}.gate_up_weight_w8a16",
            f"{shared}.gate_up_weight_w8a16_scale",
            f"{shared}.down_weight_w8a16",
            f"{shared}.down_weight_w8a16_scale",
        )
    )
    return tuple(names)


def runtime_prepared_moe_c1_tensor_names(*, layer_id: int) -> tuple[str, ...]:
    """Prepared MoE tensors actually consumed by the decode-state c=1 path."""

    prefix = f"layers.{layer_id}.mlp"
    experts = f"{prefix}.experts"
    names = [f"{prefix}.router_shared_gate.weight"]
    for proj in ("gate", "up", "down"):
        names.extend(
            (
                f"{experts}.stacked_{proj}_qweight_pack8_decode",
                f"{experts}.stacked_{proj}_qzeros",
                f"{experts}.stacked_{proj}_scales",
            )
        )
    shared = f"{prefix}.shared_expert"
    names.extend(
        (
            f"{shared}.gate_up_weight_w8a16",
            f"{shared}.gate_up_weight_w8a16_scale",
            f"{shared}.down_weight_w8a16",
            f"{shared}.down_weight_w8a16_scale",
        )
    )
    return tuple(names)


def runtime_full_attention_moe_c1_tensor_names(*, layer_id: int) -> tuple[str, ...]:
    """Normalized tensors needed by the current real full-attention runtime path."""

    attn = f"layers.{layer_id}.self_attn"
    mlp = f"layers.{layer_id}.mlp"
    experts = f"{mlp}.experts"
    names = [
        f"layers.{layer_id}.input_layernorm.weight",
        f"layers.{layer_id}.post_attention_layernorm.weight",
        f"{attn}.q_norm.weight",
        f"{attn}.k_norm.weight",
    ]
    for proj in ("q_proj", "k_proj", "v_proj"):
        base = f"{attn}.{proj}"
        names.extend(
            (
                f"{base}.qweight",
                f"{base}.qzeros",
                f"{base}.scales",
                f"{base}.theta",
                f"{base}.pairs",
                f"{base}.channel_scales",
            )
        )
    base = f"{attn}.o_proj"
    names.extend(
        (
            f"{base}.qweight",
            f"{base}.qzeros",
            f"{base}.scales",
            f"{base}.theta",
            f"{base}.pairs",
            f"{base}.channel_scales",
        )
    )
    names.extend(
        (
            f"{experts}.down_weight_pairs",
            f"{experts}.down_weight_theta",
            f"{experts}.down_weight_channel_scales",
        )
    )
    names.extend(runtime_prepared_moe_c1_tensor_names(layer_id=layer_id))
    return tuple(names)


def runtime_linear_attention_moe_c1_tensor_names(*, layer_id: int) -> tuple[str, ...]:
    """Normalized tensors needed by the current real linear-attention runtime path."""

    prefix = f"layers.{layer_id}.linear_attn"
    experts = f"layers.{layer_id}.mlp.experts"
    names = [
        f"layers.{layer_id}.input_layernorm.weight",
        f"layers.{layer_id}.post_attention_layernorm.weight",
    ]
    for proj in ("in_proj_qkv", "in_proj_z", "out_proj"):
        base = f"{prefix}.{proj}"
        names.extend(
            (
                f"{base}.qweight",
                f"{base}.qzeros",
                f"{base}.scales",
                f"{base}.theta",
                f"{base}.pairs",
                f"{base}.channel_scales",
            )
        )
    names.extend(
        (
            f"{prefix}.in_proj_a.weight",
            f"{prefix}.in_proj_b.weight",
            f"{prefix}.conv1d.weight",
            f"{prefix}.A_log",
            f"{prefix}.dt_bias",
            f"{prefix}.norm.weight",
            f"{experts}.down_weight_pairs",
            f"{experts}.down_weight_theta",
            f"{experts}.down_weight_channel_scales",
        )
    )
    names.extend(runtime_prepared_moe_c1_tensor_names(layer_id=layer_id))
    return tuple(names)


def required_moe_c1_tensor_names(*, layer_id: int, num_experts: int) -> tuple[str, ...]:
    prefix = f"layers.{layer_id}.mlp"
    names = [
        f"layers.{layer_id}.post_attention_layernorm.weight",
        f"{prefix}.gate.weight",
        f"{prefix}.shared_expert_gate.weight",
        f"{prefix}.shared_expert.gate_proj.weight",
        f"{prefix}.shared_expert.up_proj.weight",
        f"{prefix}.shared_expert.down_proj.weight",
        f"{prefix}.experts.gate_up_weight_theta",
        f"{prefix}.experts.gate_up_weight_pairs",
        f"{prefix}.experts.gate_up_weight_channel_scales",
        f"{prefix}.experts.down_weight_theta",
        f"{prefix}.experts.down_weight_pairs",
        f"{prefix}.experts.down_weight_channel_scales",
    ]
    for expert in range(num_experts):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            base = f"{prefix}.experts.{expert}.{proj}"
            names.extend((f"{base}.qweight", f"{base}.qzeros", f"{base}.scales"))
    return tuple(names)


def validate_qwen35_paro_moe_c1_layout(
    index: WeightIndex,
    *,
    layer_id: int = 0,
    raise_on_error: bool = False,
) -> Qwen35ParoLayoutValidation:
    config = qwen35_paro_config_from_hf(index.config)
    if layer_id < 0 or layer_id >= config.num_hidden_layers:
        raise ValueError(f"layer_id {layer_id} outside [0, {config.num_hidden_layers})")
    if config.quant_method and config.quant_method != "paroquant":
        raise ValueError(f"expected quant_method='paroquant', got {config.quant_method!r}")
    if config.num_experts <= 0:
        raise ValueError("Qwen3.5 PARO MoE layout requires num_experts > 0")

    normalized = _normalized_tensor_map(index)
    required = required_moe_c1_tensor_names(layer_id=layer_id, num_experts=config.num_experts)
    present = tuple(name for name in required if name in normalized)
    missing = tuple(name for name in required if name not in normalized)
    shape_errors = _validate_moe_c1_shapes(normalized, config, layer_id=layer_id)
    result = Qwen35ParoLayoutValidation(
        config=config,
        present=present,
        missing=missing,
        shape_errors=shape_errors,
    )
    if raise_on_error:
        result.raise_for_errors()
    return result


def materialize_qwen35_paro_moe_c1_layer(
    index: WeightIndex,
    *,
    layer_id: int = 0,
    device: Device | None = None,
    runtime: HipRuntime | None = None,
    validate: bool = True,
) -> Qwen35ParoLayerDeviceWeights:
    """Materialize the validated MoE c=1 layer slice using normalized names.

    The returned map is keyed by names without model-root prefixes, e.g.
    ``layers.0.mlp.experts.0.gate_proj.qweight``. This keeps runtime model code
    independent from Hugging Face checkpoint root conventions while preserving
    the original ``TensorInfo`` source on each allocation for diagnostics.
    """

    validation = validate_qwen35_paro_moe_c1_layout(index, layer_id=layer_id, raise_on_error=validate)
    if not validation.passed:
        validation.raise_for_errors()
    required = required_moe_c1_tensor_names(layer_id=layer_id, num_experts=validation.config.num_experts)
    return _materialize_normalized_layer(index, validation.config, layer_id, required, device=device, runtime=runtime)


def validate_qwen35_paro_full_attention_moe_c1_layout(
    index: WeightIndex,
    *,
    layer_id: int = 0,
    raise_on_error: bool = False,
) -> Qwen35ParoLayoutValidation:
    config = qwen35_paro_config_from_hf(index.config)
    if layer_id < 0 or layer_id >= config.num_hidden_layers:
        raise ValueError(f"layer_id {layer_id} outside [0, {config.num_hidden_layers})")
    if config.layer_types[layer_id] != "full_attention":
        raise ValueError(f"layer {layer_id} is {config.layer_types[layer_id]!r}, expected 'full_attention'")
    if config.num_attention_heads <= 0 or config.num_key_value_heads <= 0 or config.head_dim <= 0:
        raise ValueError("full-attention layout requires num_attention_heads, num_key_value_heads, and head_dim")
    if config.quant_method and config.quant_method != "paroquant":
        raise ValueError(f"expected quant_method='paroquant', got {config.quant_method!r}")

    normalized = _normalized_tensor_map(index)
    required = required_full_attention_moe_c1_tensor_names(layer_id=layer_id, num_experts=config.num_experts)
    present = tuple(name for name in required if name in normalized)
    missing = tuple(name for name in required if name not in normalized)
    shape_errors = _validate_full_attention_shapes(normalized, config, layer_id=layer_id) + _validate_moe_c1_shapes(
        normalized,
        config,
        layer_id=layer_id,
    )
    result = Qwen35ParoLayoutValidation(config=config, present=present, missing=missing, shape_errors=shape_errors)
    if raise_on_error:
        result.raise_for_errors()
    return result


def materialize_qwen35_paro_full_attention_moe_c1_layer(
    index: WeightIndex,
    *,
    layer_id: int = 0,
    device: Device | None = None,
    runtime: HipRuntime | None = None,
    validate: bool = True,
) -> Qwen35ParoLayerDeviceWeights:
    validation = validate_qwen35_paro_full_attention_moe_c1_layout(
        index,
        layer_id=layer_id,
        raise_on_error=validate,
    )
    if not validation.passed:
        validation.raise_for_errors()
    required = required_full_attention_moe_c1_tensor_names(
        layer_id=layer_id,
        num_experts=validation.config.num_experts,
    )
    return _materialize_normalized_layer(index, validation.config, layer_id, required, device=device, runtime=runtime)


def validate_qwen35_paro_linear_attention_moe_c1_layout(
    index: WeightIndex,
    *,
    layer_id: int = 0,
    raise_on_error: bool = False,
) -> Qwen35ParoLayoutValidation:
    config = qwen35_paro_config_from_hf(index.config)
    if layer_id < 0 or layer_id >= config.num_hidden_layers:
        raise ValueError(f"layer_id {layer_id} outside [0, {config.num_hidden_layers})")
    if config.layer_types[layer_id] != "linear_attention":
        raise ValueError(f"layer {layer_id} is {config.layer_types[layer_id]!r}, expected 'linear_attention'")
    if config.linear_num_key_heads <= 0 or config.linear_num_value_heads <= 0:
        raise ValueError("linear-attention layout requires linear_num_key_heads and linear_num_value_heads")
    if config.linear_key_head_dim <= 0 or config.linear_value_head_dim <= 0 or config.linear_conv_kernel_dim <= 0:
        raise ValueError("linear-attention layout requires key/value head dims and conv kernel dim")
    if config.quant_method and config.quant_method != "paroquant":
        raise ValueError(f"expected quant_method='paroquant', got {config.quant_method!r}")

    normalized = _normalized_tensor_map(index)
    required = required_linear_attention_moe_c1_tensor_names(layer_id=layer_id, num_experts=config.num_experts)
    present = tuple(name for name in required if name in normalized)
    missing = tuple(name for name in required if name not in normalized)
    shape_errors = _validate_linear_attention_shapes(normalized, config, layer_id=layer_id) + _validate_moe_c1_shapes(
        normalized,
        config,
        layer_id=layer_id,
    )
    result = Qwen35ParoLayoutValidation(config=config, present=present, missing=missing, shape_errors=shape_errors)
    if raise_on_error:
        result.raise_for_errors()
    return result


def prepare_qwen35_paro_moe_c1_host_tensors(
    index: WeightIndex,
    *,
    layer_id: int = 0,
    normalized: dict[str, TensorInfo] | None = None,
    reader: "_NormalizedTensorReader | None" = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, object]:
    """Prepare parent-compatible MoE c=1 host layouts without torch.

    This mirrors the optimized parent stack's load-time preparation: router and
    shared-gate rows are concatenated, per-expert gate/up/down tensors are
    stacked on expert dimension 0, and decode pack8 qweights are transposed on
    the last two dimensions.
    """

    config = qwen35_paro_config_from_hf(index.config)
    normalized = normalized or _normalized_tensor_map(index)
    owns_reader = reader is None
    reader = reader or _NormalizedTensorReader(normalized)
    try:
        prefix = f"layers.{layer_id}.mlp"
        experts = f"{prefix}.experts"
        prepared: dict[str, object] = {}
        _emit_progress(progress, "prepare_router_start", layer=layer_id)
        gate = _read_normalized_numpy_tensor(normalized, f"{prefix}.gate.weight", reader=reader)
        shared_gate = _read_normalized_numpy_tensor(normalized, f"{prefix}.shared_expert_gate.weight", reader=reader)
        prepared[f"{prefix}.router_shared_gate.weight"] = _concat_rows((gate, shared_gate))
        _emit_progress(progress, "prepare_router_done", layer=layer_id)
        for proj, hf_proj in (("gate", "gate_proj"), ("up", "up_proj"), ("down", "down_proj")):
            qweight = _stack_expert_refs(
                normalized,
                layer_id=layer_id,
                num_experts=config.num_experts,
                proj=hf_proj,
                suffix="qweight",
                reader=reader,
                progress=progress,
            )
            prepared[f"{experts}.stacked_{proj}_qweight"] = qweight
            prepared[f"{experts}.stacked_{proj}_qweight_pack8_decode"] = _transpose_decode_qweight(qweight)
            prepared[f"{experts}.stacked_{proj}_qzeros"] = _stack_expert_refs(
                normalized,
                layer_id=layer_id,
                num_experts=config.num_experts,
                proj=hf_proj,
                suffix="qzeros",
                reader=reader,
                progress=progress,
            )
            prepared[f"{experts}.stacked_{proj}_scales"] = _stack_expert_refs(
                normalized,
                layer_id=layer_id,
                num_experts=config.num_experts,
                proj=hf_proj,
                suffix="scales",
                reader=reader,
                progress=progress,
            )
        shared = f"{prefix}.shared_expert"
        _emit_progress(progress, "prepare_shared_expert_start", layer=layer_id)
        shared_gate = _read_normalized_numpy_tensor(normalized, f"{shared}.gate_proj.weight", reader=reader)
        shared_up = _read_normalized_numpy_tensor(normalized, f"{shared}.up_proj.weight", reader=reader)
        shared_down = _read_normalized_numpy_tensor(normalized, f"{shared}.down_proj.weight", reader=reader)
        gate_up_q, gate_up_scale = _quantize_w8a16_host(_concat_rows((shared_gate, shared_up)))
        down_q, down_scale = _quantize_w8a16_host(shared_down)
        prepared[f"{shared}.gate_up_weight_w8a16"] = gate_up_q
        prepared[f"{shared}.gate_up_weight_w8a16_scale"] = gate_up_scale
        prepared[f"{shared}.down_weight_w8a16"] = down_q
        prepared[f"{shared}.down_weight_w8a16_scale"] = down_scale
        _emit_progress(progress, "prepare_shared_expert_done", layer=layer_id)
        return prepared
    finally:
        if owns_reader:
            reader.close()


def materialize_qwen35_paro_full_attention_moe_c1_prepared_layer(
    index: WeightIndex,
    *,
    layer_id: int = 0,
    device: Device | None = None,
    runtime: HipRuntime | None = None,
    validate: bool = True,
) -> Qwen35ParoLayerDeviceWeights:
    validation = validate_qwen35_paro_full_attention_moe_c1_layout(
        index,
        layer_id=layer_id,
        raise_on_error=validate,
    )
    if not validation.passed:
        validation.raise_for_errors()
    required = required_full_attention_moe_c1_tensor_names(
        layer_id=layer_id,
        num_experts=validation.config.num_experts,
    )
    base = _materialize_normalized_layer(index, validation.config, layer_id, required, device=device, runtime=runtime)
    allocations = dict(base.weights.tensors)
    try:
        for name, array in prepare_qwen35_paro_moe_c1_host_tensors(index, layer_id=layer_id).items():
            allocations[name] = load_host_array_to_device(name, array, device=device, runtime=runtime)
    except Exception:
        DeviceWeightMap(allocations).free(runtime=runtime)
        raise
    return Qwen35ParoLayerDeviceWeights(
        config=validation.config,
        layer_id=layer_id,
        weights=DeviceWeightMap(allocations),
    )


def prepare_qwen35_paro_moe_c1_runtime_host_tensors(
    index: WeightIndex,
    *,
    layer_id: int = 0,
    normalized: dict[str, TensorInfo] | None = None,
    reader: "_NormalizedTensorReader | None" = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, object]:
    """Prepare decode-runtime MoE tensors with BF16 bit buffers where required."""

    _emit_progress(progress, "prepare_moe_start", layer=layer_id)
    prepared = prepare_qwen35_paro_moe_c1_host_tensors(
        index,
        layer_id=layer_id,
        normalized=normalized,
        reader=reader,
        progress=progress,
    )
    runtime_prepared: dict[str, object] = {}
    for name in runtime_prepared_moe_c1_tensor_names(layer_id=layer_id):
        array = prepared[name]
        _emit_progress(progress, "prepare_runtime_tensor_start", layer=layer_id, name=name)
        if _runtime_tensor_needs_bf16_bits(name):
            runtime_prepared[name] = float_array_to_bf16_bits(array)
        elif _runtime_tensor_needs_fp16(name):
            import numpy as np

            runtime_prepared[name] = np.ascontiguousarray(array, dtype=np.float16)
        else:
            runtime_prepared[name] = array
        _emit_progress(progress, "prepare_runtime_tensor_done", layer=layer_id, name=name)
    _emit_progress(progress, "prepare_moe_done", layer=layer_id)
    return runtime_prepared


def materialize_qwen35_paro_full_attention_moe_c1_runtime_layer(
    index: WeightIndex,
    *,
    layer_id: int = 0,
    device: Device | None = None,
    runtime: HipRuntime | None = None,
    validate: bool = True,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> Qwen35ParoLayerDeviceWeights:
    """Materialize the current real full-attention decode-state layer path."""

    validation = validate_qwen35_paro_full_attention_moe_c1_layout(
        index,
        layer_id=layer_id,
        raise_on_error=validate,
    )
    if not validation.passed:
        validation.raise_for_errors()
    return _materialize_runtime_layer(
        index,
        validation.config,
        layer_id,
        runtime_full_attention_moe_c1_tensor_names(layer_id=layer_id),
        device=device,
        runtime=runtime,
        progress=progress,
    )


def materialize_qwen35_paro_linear_attention_moe_c1_runtime_layer(
    index: WeightIndex,
    *,
    layer_id: int = 0,
    device: Device | None = None,
    runtime: HipRuntime | None = None,
    validate: bool = True,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> Qwen35ParoLayerDeviceWeights:
    """Materialize the current real linear-attention decode-state layer path."""

    validation = validate_qwen35_paro_linear_attention_moe_c1_layout(
        index,
        layer_id=layer_id,
        raise_on_error=validate,
    )
    if not validation.passed:
        validation.raise_for_errors()
    return _materialize_runtime_layer(
        index,
        validation.config,
        layer_id,
        runtime_linear_attention_moe_c1_tensor_names(layer_id=layer_id),
        device=device,
        runtime=runtime,
        progress=progress,
    )


class _NormalizedTensorReader:
    """Cached safetensors reader for one materialization pass."""

    def __init__(self, tensors: dict[str, TensorInfo]) -> None:
        self._tensors = tensors
        self._stack = ExitStack()
        self._handles: dict[str, Any] = {}

    def get(self, name: str):
        info = self._tensors[name]
        key = str(info.shard_path)
        handle = self._handles.get(key)
        if handle is None:
            handle = self._stack.enter_context(safe_open(key, framework="numpy"))
            self._handles[key] = handle
        return handle.get_tensor(info.name)

    def close(self) -> None:
        self._stack.close()

    def __enter__(self) -> "_NormalizedTensorReader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def _emit_progress(progress: Callable[[dict[str, Any]], None] | None, event: str, **fields: Any) -> None:
    if progress is not None:
        progress({"event": event, **fields})


def _read_normalized_numpy_tensor(
    tensors: dict[str, TensorInfo],
    name: str,
    *,
    reader: _NormalizedTensorReader | None = None,
):
    if reader is not None:
        return reader.get(name)
    with _NormalizedTensorReader(tensors) as as_reader:
        return as_reader.get(name)


def _stack_expert_refs(
    tensors: dict[str, TensorInfo],
    *,
    layer_id: int,
    num_experts: int,
    proj: str,
    suffix: str,
    reader: _NormalizedTensorReader | None = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
):
    import numpy as np

    arrays = []
    _emit_progress(progress, "expert_stack_start", layer=layer_id, proj=proj, suffix=suffix, total=num_experts)
    for expert in range(num_experts):
        if expert == 0 or (expert + 1) % 32 == 0 or expert + 1 == num_experts:
            _emit_progress(
                progress,
                "expert_stack_progress",
                layer=layer_id,
                proj=proj,
                suffix=suffix,
                expert=expert + 1,
                total=num_experts,
            )
        arrays.append(_read_normalized_numpy_tensor(tensors, f"layers.{layer_id}.mlp.experts.{expert}.{proj}.{suffix}", reader=reader))
    stacked = np.ascontiguousarray(np.stack(arrays, axis=0))
    _emit_progress(progress, "expert_stack_done", layer=layer_id, proj=proj, suffix=suffix, shape=tuple(stacked.shape))
    return stacked


def _concat_rows(arrays: tuple[object, ...]):
    import numpy as np

    return np.ascontiguousarray(np.concatenate(arrays, axis=0))


def _transpose_decode_qweight(array: object):
    import numpy as np

    if len(getattr(array, "shape")) < 3:
        raise ValueError("stacked qweight must have expert, input, and packed-output dimensions")
    return np.ascontiguousarray(np.swapaxes(array, 1, 2))


def _quantize_w8a16_host(weight: object):
    import numpy as np

    weight_f32 = np.asarray(weight, dtype=np.float32)
    scale = np.maximum(np.max(np.abs(weight_f32), axis=1), 1.0e-8).astype(np.float32) / np.float32(127.0)
    quantized = np.rint(weight_f32 / scale[:, None])
    quantized = np.clip(quantized, -127, 127).astype(np.int8)
    return np.ascontiguousarray(quantized), np.ascontiguousarray(scale)


def _runtime_tensor_needs_f32(name: str) -> bool:
    return (
        name.endswith(".conv1d.weight")
        or name.endswith(".A_log")
        or name.endswith(".dt_bias")
        or name.endswith(".linear_attn.norm.weight")
    )


def _runtime_tensor_needs_qwen_norm_offset(name: str) -> bool:
    if name.endswith(".linear_attn.norm.weight"):
        return False
    # Qwen3.5 stores normal RMSNorm scales as offsets and applies
    # ``norm(x) * (1 + weight)``.  Full-attention q/k head RMSNorm is the
    # exception in this runtime: the fused head-rmsnorm+rotary kernel itself
    # adds 1.0, so q_norm/k_norm stay as checkpoint-direct BF16 values.
    return (
        name.endswith(".input_layernorm.weight")
        or name.endswith(".post_attention_layernorm.weight")
        or name in {"norm.weight", "language_model.norm.weight", "model.norm.weight"}
    )


def _runtime_tensor_needs_bf16_bits(name: str) -> bool:
    # Parent fused full-attention head RMSNorm consumes checkpoint-direct q/k
    # offsets in BF16 and adds 1.0 inside the kernel.  Dense KV cache storage is
    # also BF16 but is allocated by runtime state, not checkpoint materialization.
    return name.endswith(".self_attn.q_norm.weight") or name.endswith(".self_attn.k_norm.weight")


def _runtime_tensor_needs_fp16(name: str) -> bool:
    return (
        name.endswith(".weight")
        or name.endswith(".scales")
        or name.endswith("_scales")
        or name.endswith(".theta")
        or name.endswith("_theta")
        or name.endswith(".channel_scales")
        or name.endswith("_channel_scales")
    ) and not name.endswith("_w8a16_scale")


def _prepare_linear_attention_qkv_z_pack8_runtime_tensors(
    normalized: dict[str, Any],
    *,
    names: tuple[str, ...],
    reader: _NormalizedTensorReader,
    layer_id: int,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, object]:
    """Prepare transposed generic qweights for fused linear-attention QKV/Z decode."""

    import numpy as np

    prefix = f"layers.{layer_id}.linear_attn"
    required = (f"{prefix}.in_proj_qkv.qweight", f"{prefix}.in_proj_z.qweight")
    if not all(name in names for name in required):
        return {}
    prepared: dict[str, object] = {}
    for source in required:
        target = source.removesuffix(".qweight") + ".qweight_pack8_decode"
        _emit_progress(progress, "prepare_runtime_tensor_start", layer=layer_id, name=target)
        qweight = np.asarray(_read_normalized_numpy_tensor(normalized, source, reader=reader), dtype=np.int32)
        prepared[target] = np.ascontiguousarray(qweight.T)
        _emit_progress(progress, "prepare_runtime_tensor_done", layer=layer_id, name=target, shape=tuple(prepared[target].shape))
    return prepared


def _prepare_full_attention_qk_pack8_runtime_tensors(
    normalized: dict[str, Any],
    *,
    names: tuple[str, ...],
    reader: _NormalizedTensorReader,
    layer_id: int,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, object]:
    """Prepare transposed generic qweights for fused full-attention Q/K decode."""

    import numpy as np

    prefix = f"layers.{layer_id}.self_attn"
    required = (f"{prefix}.q_proj.qweight", f"{prefix}.k_proj.qweight")
    if not all(name in names for name in required):
        return {}
    prepared: dict[str, object] = {}
    for source in required:
        target = source.removesuffix(".qweight") + ".qweight_pack8_decode"
        _emit_progress(progress, "prepare_runtime_tensor_start", layer=layer_id, name=target)
        qweight = np.asarray(_read_normalized_numpy_tensor(normalized, source, reader=reader), dtype=np.int32)
        prepared[target] = np.ascontiguousarray(qweight.T)
        _emit_progress(progress, "prepare_runtime_tensor_done", layer=layer_id, name=target, shape=tuple(prepared[target].shape))
    return prepared


def _materialize_runtime_layer(
    index: WeightIndex,
    config: Qwen35ParoConfig,
    layer_id: int,
    names: tuple[str, ...],
    *,
    device: Device | None,
    runtime: HipRuntime | None,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> Qwen35ParoLayerDeviceWeights:
    normalized = _normalized_tensor_map(index)
    prepared_names = set(runtime_prepared_moe_c1_tensor_names(layer_id=layer_id))
    allocations: dict[str, DeviceTensorAllocation] = {}
    reader = _NormalizedTensorReader(normalized)
    try:
        direct_names = tuple(name for name in names if name not in prepared_names)
        for idx, name in enumerate(direct_names, start=1):
            _emit_progress(
                progress,
                "materialize_tensor_start",
                layer=layer_id,
                name=name,
                index=idx,
                total=len(direct_names),
            )
            if _runtime_tensor_needs_f32(name):
                import numpy as np

                array = np.ascontiguousarray(_read_normalized_numpy_tensor(normalized, name, reader=reader), dtype=np.float32)
                allocations[name] = load_host_array_to_device_as_dtype(
                    name,
                    array,
                    DType.FP32,
                    device=device,
                    runtime=runtime,
                )
            elif _runtime_tensor_needs_qwen_norm_offset(name):
                import numpy as np

                direct = np.asarray(_read_normalized_numpy_tensor(normalized, name, reader=reader), dtype=np.float32)
                array = np.ascontiguousarray(direct + np.float32(1.0), dtype=np.float16)
                allocations[name] = load_host_array_to_device_as_dtype(
                    name,
                    array,
                    DType.FP16,
                    device=device,
                    runtime=runtime,
                )
            elif _runtime_tensor_needs_bf16_bits(name):
                array = float_array_to_bf16_bits(_read_normalized_numpy_tensor(normalized, name, reader=reader))
                allocations[name] = load_host_array_to_device_as_dtype(
                    name,
                    array,
                    DType.BF16,
                    device=device,
                    runtime=runtime,
                )
            elif _runtime_tensor_needs_fp16(name):
                import numpy as np

                array = np.ascontiguousarray(_read_normalized_numpy_tensor(normalized, name, reader=reader), dtype=np.float16)
                allocations[name] = load_host_array_to_device_as_dtype(
                    name,
                    array,
                    DType.FP16,
                    device=device,
                    runtime=runtime,
                )
            else:
                array = _read_normalized_numpy_tensor(normalized, name, reader=reader)
                allocations[name] = load_host_array_to_device(name, array, device=device, runtime=runtime)
            _emit_progress(
                progress,
                "materialize_tensor_done",
                layer=layer_id,
                name=name,
                index=idx,
                total=len(direct_names),
            )
        linear_pack8 = _prepare_linear_attention_qkv_z_pack8_runtime_tensors(
            normalized,
            names=names,
            reader=reader,
            layer_id=layer_id,
            progress=progress,
        )
        linear_pack8.update(
            _prepare_full_attention_qk_pack8_runtime_tensors(
                normalized,
                names=names,
                reader=reader,
                layer_id=layer_id,
                progress=progress,
            )
        )
        for idx, (name, array) in enumerate(linear_pack8.items(), start=1):
            _emit_progress(
                progress,
                "materialize_prepared_tensor_start",
                layer=layer_id,
                name=name,
                index=idx,
                total=len(linear_pack8),
            )
            allocations[name] = load_host_array_to_device(name, array, device=device, runtime=runtime)
            _emit_progress(
                progress,
                "materialize_prepared_tensor_done",
                layer=layer_id,
                name=name,
                index=idx,
                total=len(linear_pack8),
            )
        prepared = prepare_qwen35_paro_moe_c1_runtime_host_tensors(
            index,
            layer_id=layer_id,
            normalized=normalized,
            reader=reader,
            progress=progress,
        )
        for idx, (name, array) in enumerate(prepared.items(), start=1):
            _emit_progress(
                progress,
                "materialize_prepared_tensor_start",
                layer=layer_id,
                name=name,
                index=idx,
                total=len(prepared),
            )
            if _runtime_tensor_needs_bf16_bits(name):
                allocations[name] = load_host_array_to_device_as_dtype(
                    name,
                    array,
                    DType.BF16,
                    device=device,
                    runtime=runtime,
                )
            elif _runtime_tensor_needs_fp16(name):
                allocations[name] = load_host_array_to_device_as_dtype(
                    name,
                    array,
                    DType.FP16,
                    device=device,
                    runtime=runtime,
                )
            else:
                allocations[name] = load_host_array_to_device(name, array, device=device, runtime=runtime)
            _emit_progress(
                progress,
                "materialize_prepared_tensor_done",
                layer=layer_id,
                name=name,
                index=idx,
                total=len(prepared),
            )
    except Exception:
        DeviceWeightMap(allocations).free(runtime=runtime)
        raise
    finally:
        reader.close()
    return Qwen35ParoLayerDeviceWeights(
        config=config,
        layer_id=layer_id,
        weights=DeviceWeightMap(allocations),
    )


def _materialize_normalized_layer(
    index: WeightIndex,
    config: Qwen35ParoConfig,
    layer_id: int,
    required: tuple[str, ...],
    *,
    device: Device | None,
    runtime: HipRuntime | None,
) -> Qwen35ParoLayerDeviceWeights:
    normalized = _normalized_tensor_map(index)
    allocations: dict[str, DeviceTensorAllocation] = {}
    try:
        for normalized_name in required:
            allocation = load_tensor_info_to_device(normalized[normalized_name], device=device, runtime=runtime)
            allocations[normalized_name] = DeviceTensorAllocation(
                name=normalized_name,
                source=allocation.source,
                buffer=allocation.buffer,
                tensor=allocation.tensor,
            )
    except Exception:
        DeviceWeightMap(allocations).free(runtime=runtime)
        raise
    return Qwen35ParoLayerDeviceWeights(
        config=config,
        layer_id=layer_id,
        weights=DeviceWeightMap(allocations),
    )


def _normalized_tensor_map(index: WeightIndex) -> dict[str, TensorInfo]:
    out: dict[str, TensorInfo] = {}
    for name, info in index.tensors.items():
        normalized = normalize_qwen35_weight_name(name)
        if normalized in out:
            raise ValueError(f"duplicate normalized tensor name {normalized!r}")
        out[normalized] = info
    return out


def _validate_full_attention_shapes(
    tensors: dict[str, TensorInfo],
    config: Qwen35ParoConfig,
    *,
    layer_id: int,
) -> tuple[str, ...]:
    prefix = f"layers.{layer_id}.self_attn"
    expected: dict[str, tuple[int, ...]] = {
        f"layers.{layer_id}.input_layernorm.weight": (config.hidden_size,),
        f"{prefix}.q_norm.weight": (config.head_dim,),
        f"{prefix}.k_norm.weight": (config.head_dim,),
    }
    errors: list[str] = []
    for name, shape in expected.items():
        info = tensors.get(name)
        if info is not None and info.shape != shape:
            errors.append(f"{name}: expected {shape}, got {info.shape}")
    return tuple(errors)


def _validate_linear_attention_shapes(
    tensors: dict[str, TensorInfo],
    config: Qwen35ParoConfig,
    *,
    layer_id: int,
) -> tuple[str, ...]:
    prefix = f"layers.{layer_id}.linear_attn"
    qkv_width = 2 * config.linear_num_key_heads * config.linear_key_head_dim + config.linear_num_value_heads * config.linear_value_head_dim
    expected: dict[str, tuple[int, ...]] = {
        f"layers.{layer_id}.input_layernorm.weight": (config.hidden_size,),
        f"{prefix}.in_proj_a.weight": (config.linear_num_value_heads, config.hidden_size),
        f"{prefix}.in_proj_b.weight": (config.linear_num_value_heads, config.hidden_size),
        f"{prefix}.conv1d.weight": (qkv_width, 1, config.linear_conv_kernel_dim),
        f"{prefix}.A_log": (config.linear_num_value_heads,),
        f"{prefix}.dt_bias": (config.linear_num_value_heads,),
        f"{prefix}.norm.weight": (config.linear_value_head_dim,),
    }
    errors: list[str] = []
    for name, shape in expected.items():
        info = tensors.get(name)
        if info is not None and info.shape != shape:
            errors.append(f"{name}: expected {shape}, got {info.shape}")
    return tuple(errors)


def _validate_moe_c1_shapes(
    tensors: dict[str, TensorInfo],
    config: Qwen35ParoConfig,
    *,
    layer_id: int,
) -> tuple[str, ...]:
    prefix = f"layers.{layer_id}.mlp"
    expected: dict[str, tuple[int, ...]] = {
        f"layers.{layer_id}.post_attention_layernorm.weight": (config.hidden_size,),
        f"{prefix}.gate.weight": (config.num_experts, config.hidden_size),
        f"{prefix}.shared_expert_gate.weight": (1, config.hidden_size),
    }
    if config.shared_expert_intermediate_size > 0:
        shared = config.shared_expert_intermediate_size
        expected.update(
            {
                f"{prefix}.shared_expert.gate_proj.weight": (shared, config.hidden_size),
                f"{prefix}.shared_expert.up_proj.weight": (shared, config.hidden_size),
                f"{prefix}.shared_expert.down_proj.weight": (config.hidden_size, shared),
            }
        )

    errors: list[str] = []
    for name, shape in expected.items():
        info = tensors.get(name)
        if info is not None and info.shape != shape:
            errors.append(f"{name}: expected {shape}, got {info.shape}")
    return tuple(errors)
