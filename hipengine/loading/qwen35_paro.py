"""Qwen3.5/PARO checkpoint layout validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hipengine.core.device import Device
from hipengine.core.hip import HipRuntime
from hipengine.core.tensor import Tensor
from hipengine.loading.materialize import DeviceTensorAllocation, DeviceWeightMap, load_tensor_info_to_device
from hipengine.loading.safetensors import MissingTensorError, TensorInfo, WeightIndex

ROOT_PREFIXES = ("model.language_model.", "language_model.", "model.")


@dataclass(frozen=True)
class Qwen35ParoConfig:
    architecture: str
    num_hidden_layers: int
    hidden_size: int
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
    return Qwen35ParoConfig(
        architecture=architecture,
        num_hidden_layers=num_layers,
        hidden_size=int(text["hidden_size"]),
        num_experts=int(text.get("num_experts", 0) or 0),
        num_experts_per_tok=int(text.get("num_experts_per_tok", 0) or 0),
        moe_intermediate_size=int(text.get("moe_intermediate_size", text.get("intermediate_size", 0)) or 0),
        shared_expert_intermediate_size=int(text.get("shared_expert_intermediate_size", 0) or 0),
        layer_types=layer_types,
        quant_method=quant_method,
    )


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
    normalized = _normalized_tensor_map(index)
    required = required_moe_c1_tensor_names(layer_id=layer_id, num_experts=validation.config.num_experts)
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
        config=validation.config,
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
