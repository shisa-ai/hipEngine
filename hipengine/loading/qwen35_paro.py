"""Qwen3.5/PARO checkpoint layout validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hipengine.core.device import Device
from hipengine.core.hip import HipRuntime
from hipengine.core.tensor import Tensor
from hipengine.loading.materialize import (
    DeviceTensorAllocation,
    DeviceWeightMap,
    load_host_array_to_device,
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
    names.extend((f"{base}.qweight", f"{base}.qzeros", f"{base}.scales"))
    return tuple(names)


def required_full_attention_moe_c1_tensor_names(*, layer_id: int, num_experts: int) -> tuple[str, ...]:
    return required_full_attention_c1_tensor_names(layer_id=layer_id) + required_moe_c1_tensor_names(
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


def prepare_qwen35_paro_moe_c1_host_tensors(index: WeightIndex, *, layer_id: int = 0) -> dict[str, object]:
    """Prepare parent-compatible MoE c=1 host layouts without torch.

    This mirrors the optimized parent stack's load-time preparation: router and
    shared-gate rows are concatenated, per-expert gate/up/down tensors are
    stacked on expert dimension 0, and decode pack8 qweights are transposed on
    the last two dimensions.
    """

    config = qwen35_paro_config_from_hf(index.config)
    normalized = _normalized_tensor_map(index)
    prefix = f"layers.{layer_id}.mlp"
    experts = f"{prefix}.experts"
    prepared: dict[str, object] = {}
    gate = _read_normalized_numpy_tensor(normalized, f"{prefix}.gate.weight")
    shared_gate = _read_normalized_numpy_tensor(normalized, f"{prefix}.shared_expert_gate.weight")
    prepared[f"{prefix}.router_shared_gate.weight"] = _concat_rows((gate, shared_gate))
    for proj, hf_proj in (("gate", "gate_proj"), ("up", "up_proj"), ("down", "down_proj")):
        qweight = _stack_expert_refs(normalized, layer_id=layer_id, num_experts=config.num_experts, proj=hf_proj, suffix="qweight")
        prepared[f"{experts}.stacked_{proj}_qweight"] = qweight
        prepared[f"{experts}.stacked_{proj}_qweight_pack8_decode"] = _transpose_decode_qweight(qweight)
        prepared[f"{experts}.stacked_{proj}_qzeros"] = _stack_expert_refs(
            normalized,
            layer_id=layer_id,
            num_experts=config.num_experts,
            proj=hf_proj,
            suffix="qzeros",
        )
        prepared[f"{experts}.stacked_{proj}_scales"] = _stack_expert_refs(
            normalized,
            layer_id=layer_id,
            num_experts=config.num_experts,
            proj=hf_proj,
            suffix="scales",
        )
    return prepared


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


def _read_normalized_numpy_tensor(tensors: dict[str, TensorInfo], name: str):
    from safetensors import safe_open

    info = tensors[name]
    with safe_open(str(info.shard_path), framework="numpy") as handle:
        return handle.get_tensor(info.name)


def _stack_expert_refs(
    tensors: dict[str, TensorInfo],
    *,
    layer_id: int,
    num_experts: int,
    proj: str,
    suffix: str,
):
    import numpy as np

    arrays = [
        _read_normalized_numpy_tensor(tensors, f"layers.{layer_id}.mlp.experts.{expert}.{proj}.{suffix}")
        for expert in range(num_experts)
    ]
    return np.ascontiguousarray(np.stack(arrays, axis=0))


def _concat_rows(arrays: tuple[object, ...]):
    import numpy as np

    return np.ascontiguousarray(np.concatenate(arrays, axis=0))


def _transpose_decode_qweight(array: object):
    import numpy as np

    if len(getattr(array, "shape")) < 3:
        raise ValueError("stacked qweight must have expert, input, and packed-output dimensions")
    return np.ascontiguousarray(np.swapaxes(array, 1, 2))


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
