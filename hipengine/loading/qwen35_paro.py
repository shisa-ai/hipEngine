"""Qwen3.5/PARO checkpoint layout validation."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass
import os
from typing import Any

from safetensors import safe_open

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.hip import HipRuntime
from hipengine.core.tensor import Tensor
from hipengine.loading.materialize import (
    DeviceTensorAllocation,
    DeviceWeightMap,
    alias_device_allocation,
    float_array_to_bf16_bits,
    load_host_array_to_device,
    load_host_array_to_device_as_dtype,
    load_tensor_info_to_device,
)
from hipengine.loading.safetensors import MissingTensorError, TensorInfo, WeightIndex

ROOT_PREFIXES = ("model.language_model.", "language_model.", "model.")
SHARED_EXPERT_FORMAT_LEGACY_FP16 = "legacy_fp16"
SHARED_EXPERT_FORMAT_PACKED_PARO_W4 = "packed_paro_w4"
_SHARED_EXPERT_FORMATS = {SHARED_EXPERT_FORMAT_LEGACY_FP16, SHARED_EXPERT_FORMAT_PACKED_PARO_W4}


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
    max_position_embeddings: int = 0
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
    shared_expert_format: str = ""

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
        max_position_embeddings=int(text.get("max_position_embeddings", config.get("max_position_embeddings", 0)) or 0),
        rms_norm_eps=float(text.get("rms_norm_eps", 1.0e-6) or 1.0e-6),
        rope_theta=float(text.get("rope_theta", rope_parameters.get("rope_theta", 1000000.0)) or 1000000.0),
        rotary_dim=rotary_dim,
        linear_num_key_heads=int(text.get("linear_num_key_heads", 0) or 0),
        linear_num_value_heads=int(text.get("linear_num_value_heads", 0) or 0),
        linear_key_head_dim=int(text.get("linear_key_head_dim", 0) or 0),
        linear_value_head_dim=int(text.get("linear_value_head_dim", 0) or 0),
        linear_conv_kernel_dim=int(text.get("linear_conv_kernel_dim", 0) or 0),
    )


def _normalize_shared_expert_format(shared_expert_format: str) -> str:
    if shared_expert_format not in _SHARED_EXPERT_FORMATS:
        valid = ", ".join(sorted(_SHARED_EXPERT_FORMATS))
        raise ValueError(f"unknown shared_expert_format {shared_expert_format!r}; expected one of: {valid}")
    return shared_expert_format


def _legacy_shared_expert_tensor_names(*, layer_id: int) -> tuple[str, ...]:
    shared = f"layers.{layer_id}.mlp.shared_expert"
    return tuple(f"{shared}.{proj}.weight" for proj in ("gate_proj", "up_proj", "down_proj"))


def _packed_shared_expert_tensor_names(*, layer_id: int) -> tuple[str, ...]:
    shared = f"layers.{layer_id}.mlp.shared_expert"
    names: list[str] = []
    for proj in ("gate_proj", "up_proj", "down_proj"):
        base = f"{shared}.{proj}"
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


def _shared_expert_tensor_names(*, layer_id: int, shared_expert_format: str) -> tuple[str, ...]:
    shared_expert_format = _normalize_shared_expert_format(shared_expert_format)
    if shared_expert_format == SHARED_EXPERT_FORMAT_LEGACY_FP16:
        return _legacy_shared_expert_tensor_names(layer_id=layer_id)
    return _packed_shared_expert_tensor_names(layer_id=layer_id)


def _shared_expert_runtime_sidecar_tensor_names(*, layer_id: int, shared_expert_format: str) -> tuple[str, ...]:
    """Checkpoint tensors consumed directly by the runtime shared-expert path.

    Packed W4 materialization prepares ``qweight_pack8_decode`` from raw
    ``qweight`` at load time, so raw ``qweight`` is intentionally omitted from
    the runtime device map.  The other sidecars are read directly by the W4
    PARO kernels.
    """

    shared_expert_format = _normalize_shared_expert_format(shared_expert_format)
    if shared_expert_format == SHARED_EXPERT_FORMAT_LEGACY_FP16:
        return ()
    shared = f"layers.{layer_id}.mlp.shared_expert"
    names: list[str] = []
    for proj in ("gate_proj", "up_proj", "down_proj"):
        base = f"{shared}.{proj}"
        names.extend(
            (
                f"{base}.qzeros",
                f"{base}.scales",
                f"{base}.theta",
                f"{base}.pairs",
                f"{base}.channel_scales",
            )
        )
    return tuple(names)


def _detect_shared_expert_format(tensors: dict[str, Any], *, layer_id: int) -> str:
    """Detect the dense shared-expert representation for one layer.

    The public z-lab PARO checkpoint stores the dense shared expert as three
    fp16 ``*.weight`` matrices.  hipEngine's newer packed format stores the
    same projections as PARO W4 sidecars.  When both are present, prefer the
    packed sidecars so unstripped converter outputs exercise the new path.
    """

    packed = _packed_shared_expert_tensor_names(layer_id=layer_id)
    legacy = _legacy_shared_expert_tensor_names(layer_id=layer_id)
    if all(name in tensors for name in packed):
        return SHARED_EXPERT_FORMAT_PACKED_PARO_W4
    if all(name in tensors for name in legacy):
        return SHARED_EXPERT_FORMAT_LEGACY_FP16
    return ""


def _shared_expert_validation_choice(tensors: dict[str, Any], *, layer_id: int) -> tuple[str, tuple[str, ...]]:
    detected = _detect_shared_expert_format(tensors, layer_id=layer_id)
    if detected:
        names = _shared_expert_tensor_names(layer_id=layer_id, shared_expert_format=detected)
        return detected, tuple(name for name in names if name not in tensors)
    packed = _packed_shared_expert_tensor_names(layer_id=layer_id)
    legacy = _legacy_shared_expert_tensor_names(layer_id=layer_id)
    if any(name in tensors for name in packed):
        chosen = SHARED_EXPERT_FORMAT_PACKED_PARO_W4
        names = packed
    else:
        chosen = SHARED_EXPERT_FORMAT_LEGACY_FP16
        names = legacy
    return chosen, tuple(name for name in names if name not in tensors)


def _resolve_shared_expert_format(
    tensors: dict[str, Any],
    *,
    layer_id: int,
    shared_expert_format: str | None = None,
) -> str:
    if shared_expert_format is not None:
        shared_expert_format = _normalize_shared_expert_format(shared_expert_format)
        names = _shared_expert_tensor_names(layer_id=layer_id, shared_expert_format=shared_expert_format)
        missing = tuple(name for name in names if name not in tensors)
        if missing:
            preview = ", ".join(missing[:8])
            more = "" if len(missing) <= 8 else f" (+{len(missing) - 8} more)"
            raise MissingTensorError(f"missing tensors for {shared_expert_format} shared_expert: {preview}{more}")
        return shared_expert_format
    detected = _detect_shared_expert_format(tensors, layer_id=layer_id)
    if detected:
        return detected
    chosen, missing = _shared_expert_validation_choice(tensors, layer_id=layer_id)
    preview = ", ".join(missing[:8])
    more = "" if len(missing) <= 8 else f" (+{len(missing) - 8} more)"
    raise MissingTensorError(f"missing tensors for {chosen} shared_expert: {preview}{more}")


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


def required_full_attention_moe_c1_tensor_names(
    *,
    layer_id: int,
    num_experts: int,
    shared_expert_format: str = SHARED_EXPERT_FORMAT_PACKED_PARO_W4,
) -> tuple[str, ...]:
    return required_full_attention_c1_tensor_names(layer_id=layer_id) + required_moe_c1_tensor_names(
        layer_id=layer_id,
        num_experts=num_experts,
        shared_expert_format=shared_expert_format,
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


def required_linear_attention_moe_c1_tensor_names(
    *,
    layer_id: int,
    num_experts: int,
    shared_expert_format: str = SHARED_EXPERT_FORMAT_PACKED_PARO_W4,
) -> tuple[str, ...]:
    return required_linear_attention_c1_tensor_names(layer_id=layer_id) + required_moe_c1_tensor_names(
        layer_id=layer_id,
        num_experts=num_experts,
        shared_expert_format=shared_expert_format,
    )


def prepared_moe_c1_tensor_names(
    *,
    layer_id: int,
    shared_expert_format: str = SHARED_EXPERT_FORMAT_PACKED_PARO_W4,
) -> tuple[str, ...]:
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
    shared_expert_format = _normalize_shared_expert_format(shared_expert_format)
    if shared_expert_format == SHARED_EXPERT_FORMAT_LEGACY_FP16:
        names.extend(
            (
                f"{shared}.gate_up_weight_w8a16",
                f"{shared}.gate_up_weight_w8a16_scale",
                f"{shared}.down_weight_w8a16",
                f"{shared}.down_weight_w8a16_scale",
            )
        )
    else:
        for proj in ("gate_proj", "up_proj", "down_proj"):
            names.append(f"{shared}.{proj}.qweight_pack8_decode")
    return tuple(names)


def runtime_prepared_moe_c1_tensor_names(
    *,
    layer_id: int,
    shared_expert_format: str = SHARED_EXPERT_FORMAT_PACKED_PARO_W4,
) -> tuple[str, ...]:
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
    shared_expert_format = _normalize_shared_expert_format(shared_expert_format)
    if shared_expert_format == SHARED_EXPERT_FORMAT_LEGACY_FP16:
        names.extend(
            (
                f"{shared}.gate_up_weight_w8a16",
                f"{shared}.gate_up_weight_w8a16_scale",
                f"{shared}.down_weight_w8a16",
                f"{shared}.down_weight_w8a16_scale",
            )
        )
    else:
        for proj in ("gate_proj", "up_proj", "down_proj"):
            names.append(f"{shared}.{proj}.qweight_pack8_decode")
    return tuple(names)


def runtime_full_attention_moe_c1_tensor_names(
    *,
    layer_id: int,
    shared_expert_format: str = SHARED_EXPERT_FORMAT_PACKED_PARO_W4,
) -> tuple[str, ...]:
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
            f"{experts}.gate_up_weight_pairs",
            f"{experts}.gate_up_weight_theta",
            f"{experts}.gate_up_weight_channel_scales",
            f"{experts}.down_weight_pairs",
            f"{experts}.down_weight_theta",
            f"{experts}.down_weight_channel_scales",
        )
    )
    names.extend(_shared_expert_runtime_sidecar_tensor_names(layer_id=layer_id, shared_expert_format=shared_expert_format))
    names.extend(runtime_prepared_moe_c1_tensor_names(layer_id=layer_id, shared_expert_format=shared_expert_format))
    return tuple(names)


def runtime_linear_attention_moe_c1_tensor_names(
    *,
    layer_id: int,
    shared_expert_format: str = SHARED_EXPERT_FORMAT_PACKED_PARO_W4,
) -> tuple[str, ...]:
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
            f"{experts}.gate_up_weight_pairs",
            f"{experts}.gate_up_weight_theta",
            f"{experts}.gate_up_weight_channel_scales",
            f"{experts}.down_weight_pairs",
            f"{experts}.down_weight_theta",
            f"{experts}.down_weight_channel_scales",
        )
    )
    names.extend(_shared_expert_runtime_sidecar_tensor_names(layer_id=layer_id, shared_expert_format=shared_expert_format))
    names.extend(runtime_prepared_moe_c1_tensor_names(layer_id=layer_id, shared_expert_format=shared_expert_format))
    return tuple(names)


def _dense_mlp_paro_tensor_names(*, layer_id: int) -> tuple[str, ...]:
    prefix = f"layers.{layer_id}.mlp"
    names = [f"layers.{layer_id}.post_attention_layernorm.weight"]
    for proj in ("gate_proj", "up_proj", "down_proj"):
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
    return tuple(names)


def runtime_full_attention_dense_c1_tensor_names(*, layer_id: int) -> tuple[str, ...]:
    """Normalized tensors needed by the Qwen3.5 dense-text full-attention path."""

    attn = f"layers.{layer_id}.self_attn"
    names = [
        f"layers.{layer_id}.input_layernorm.weight",
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
    names.extend(_dense_mlp_paro_tensor_names(layer_id=layer_id))
    return tuple(names)


def runtime_linear_attention_dense_c1_tensor_names(*, layer_id: int) -> tuple[str, ...]:
    """Normalized tensors needed by the Qwen3.5 dense-text linear-attention path."""

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
    names.extend(_dense_mlp_paro_tensor_names(layer_id=layer_id))
    return tuple(names)


def required_moe_c1_tensor_names(
    *,
    layer_id: int,
    num_experts: int,
    shared_expert_format: str = SHARED_EXPERT_FORMAT_PACKED_PARO_W4,
) -> tuple[str, ...]:
    prefix = f"layers.{layer_id}.mlp"
    names = [
        f"layers.{layer_id}.post_attention_layernorm.weight",
        f"{prefix}.gate.weight",
        f"{prefix}.shared_expert_gate.weight",
        f"{prefix}.experts.gate_up_weight_theta",
        f"{prefix}.experts.gate_up_weight_pairs",
        f"{prefix}.experts.gate_up_weight_channel_scales",
        f"{prefix}.experts.down_weight_theta",
        f"{prefix}.experts.down_weight_pairs",
        f"{prefix}.experts.down_weight_channel_scales",
    ]
    names.extend(_shared_expert_tensor_names(layer_id=layer_id, shared_expert_format=shared_expert_format))
    for expert in range(num_experts):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            base = f"{prefix}.experts.{expert}.{proj}"
            names.extend((f"{base}.qweight", f"{base}.qzeros", f"{base}.scales"))
    return tuple(names)


def _required_moe_c1_tensor_names_for_layout(
    tensors: dict[str, Any],
    config: Qwen35ParoConfig,
    *,
    layer_id: int,
    shared_expert_format: str | None = None,
) -> tuple[tuple[str, ...], str]:
    if shared_expert_format is None:
        shared_expert_format, _missing_shared = _shared_expert_validation_choice(tensors, layer_id=layer_id)
    else:
        shared_expert_format = _normalize_shared_expert_format(shared_expert_format)
    return (
        required_moe_c1_tensor_names(
            layer_id=layer_id,
            num_experts=config.num_experts,
            shared_expert_format=shared_expert_format,
        ),
        shared_expert_format,
    )


def validate_qwen35_paro_moe_c1_layout(
    index: WeightIndex,
    *,
    layer_id: int = 0,
    raise_on_error: bool = False,
    shared_expert_format: str | None = None,
) -> Qwen35ParoLayoutValidation:
    config = qwen35_paro_config_from_hf(index.config)
    if layer_id < 0 or layer_id >= config.num_hidden_layers:
        raise ValueError(f"layer_id {layer_id} outside [0, {config.num_hidden_layers})")
    if config.quant_method and config.quant_method != "paroquant":
        raise ValueError(f"expected quant_method='paroquant', got {config.quant_method!r}")
    if config.num_experts <= 0:
        raise ValueError("Qwen3.5 PARO MoE layout requires num_experts > 0")

    normalized = _normalized_tensor_map(index)
    required, shared_expert_format = _required_moe_c1_tensor_names_for_layout(
        normalized,
        config,
        layer_id=layer_id,
        shared_expert_format=shared_expert_format,
    )
    present = tuple(name for name in required if name in normalized)
    missing = tuple(name for name in required if name not in normalized)
    shape_errors = _validate_moe_c1_shapes(
        normalized,
        config,
        layer_id=layer_id,
        shared_expert_format=shared_expert_format,
    )
    result = Qwen35ParoLayoutValidation(
        config=config,
        present=present,
        missing=missing,
        shape_errors=shape_errors,
        shared_expert_format=shared_expert_format,
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
    shared_expert_format: str | None = None,
) -> Qwen35ParoLayerDeviceWeights:
    """Materialize the validated MoE c=1 layer slice using normalized names.

    The returned map is keyed by names without model-root prefixes, e.g.
    ``layers.0.mlp.experts.0.gate_proj.qweight``. This keeps runtime model code
    independent from Hugging Face checkpoint root conventions while preserving
    the original ``TensorInfo`` source on each allocation for diagnostics.
    """

    validation = validate_qwen35_paro_moe_c1_layout(
        index,
        layer_id=layer_id,
        raise_on_error=validate,
        shared_expert_format=shared_expert_format,
    )
    if not validation.passed:
        validation.raise_for_errors()
    required = required_moe_c1_tensor_names(
        layer_id=layer_id,
        num_experts=validation.config.num_experts,
        shared_expert_format=validation.shared_expert_format,
    )
    return _materialize_normalized_layer(index, validation.config, layer_id, required, device=device, runtime=runtime)


def validate_qwen35_paro_full_attention_moe_c1_layout(
    index: WeightIndex,
    *,
    layer_id: int = 0,
    raise_on_error: bool = False,
    shared_expert_format: str | None = None,
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
    _moe_required, shared_expert_format = _required_moe_c1_tensor_names_for_layout(
        normalized,
        config,
        layer_id=layer_id,
        shared_expert_format=shared_expert_format,
    )
    required = required_full_attention_moe_c1_tensor_names(
        layer_id=layer_id,
        num_experts=config.num_experts,
        shared_expert_format=shared_expert_format,
    )
    present = tuple(name for name in required if name in normalized)
    missing = tuple(name for name in required if name not in normalized)
    shape_errors = _validate_full_attention_shapes(normalized, config, layer_id=layer_id) + _validate_moe_c1_shapes(
        normalized,
        config,
        layer_id=layer_id,
        shared_expert_format=shared_expert_format,
    )
    result = Qwen35ParoLayoutValidation(
        config=config,
        present=present,
        missing=missing,
        shape_errors=shape_errors,
        shared_expert_format=shared_expert_format,
    )
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
    shared_expert_format: str | None = None,
) -> Qwen35ParoLayerDeviceWeights:
    validation = validate_qwen35_paro_full_attention_moe_c1_layout(
        index,
        layer_id=layer_id,
        raise_on_error=validate,
        shared_expert_format=shared_expert_format,
    )
    if not validation.passed:
        validation.raise_for_errors()
    required = required_full_attention_moe_c1_tensor_names(
        layer_id=layer_id,
        num_experts=validation.config.num_experts,
        shared_expert_format=validation.shared_expert_format,
    )
    return _materialize_normalized_layer(index, validation.config, layer_id, required, device=device, runtime=runtime)


def validate_qwen35_paro_linear_attention_moe_c1_layout(
    index: WeightIndex,
    *,
    layer_id: int = 0,
    raise_on_error: bool = False,
    shared_expert_format: str | None = None,
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
    _moe_required, shared_expert_format = _required_moe_c1_tensor_names_for_layout(
        normalized,
        config,
        layer_id=layer_id,
        shared_expert_format=shared_expert_format,
    )
    required = required_linear_attention_moe_c1_tensor_names(
        layer_id=layer_id,
        num_experts=config.num_experts,
        shared_expert_format=shared_expert_format,
    )
    present = tuple(name for name in required if name in normalized)
    missing = tuple(name for name in required if name not in normalized)
    shape_errors = _validate_linear_attention_shapes(normalized, config, layer_id=layer_id) + _validate_moe_c1_shapes(
        normalized,
        config,
        layer_id=layer_id,
        shared_expert_format=shared_expert_format,
    )
    result = Qwen35ParoLayoutValidation(
        config=config,
        present=present,
        missing=missing,
        shape_errors=shape_errors,
        shared_expert_format=shared_expert_format,
    )
    if raise_on_error:
        result.raise_for_errors()
    return result


def _validate_dense_mlp_shapes(
    tensors: dict[str, TensorInfo],
    config: Qwen35ParoConfig,
    *,
    layer_id: int,
) -> tuple[str, ...]:
    prefix = f"layers.{layer_id}.mlp"
    expected: dict[str, tuple[int, ...]] = {
        f"layers.{layer_id}.post_attention_layernorm.weight": (config.hidden_size,),
    }
    errors: list[str] = []
    for name, shape in expected.items():
        info = tensors.get(name)
        if info is not None and info.shape != shape:
            errors.append(f"{name}: expected {shape}, got {info.shape}")
    # Dense PARO gate/up/down sidecars follow the same qweight/qzeros/scales
    # convention as attention projections.  Their exact packed width depends on
    # quantization metadata, so validate existence here and rely on kernel
    # wrappers for shape-specific launch checks.
    _ = prefix
    return tuple(errors)


def validate_qwen35_paro_full_attention_dense_c1_layout(
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
    required = runtime_full_attention_dense_c1_tensor_names(layer_id=layer_id)
    present = tuple(name for name in required if name in normalized)
    missing = tuple(name for name in required if name not in normalized)
    shape_errors = _validate_full_attention_shapes(normalized, config, layer_id=layer_id) + _validate_dense_mlp_shapes(
        normalized,
        config,
        layer_id=layer_id,
    )
    result = Qwen35ParoLayoutValidation(
        config=config,
        present=present,
        missing=missing,
        shape_errors=shape_errors,
        shared_expert_format="dense_paro_w4",
    )
    if raise_on_error:
        result.raise_for_errors()
    return result


def validate_qwen35_paro_linear_attention_dense_c1_layout(
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
    required = runtime_linear_attention_dense_c1_tensor_names(layer_id=layer_id)
    present = tuple(name for name in required if name in normalized)
    missing = tuple(name for name in required if name not in normalized)
    shape_errors = _validate_linear_attention_shapes(normalized, config, layer_id=layer_id) + _validate_dense_mlp_shapes(
        normalized,
        config,
        layer_id=layer_id,
    )
    result = Qwen35ParoLayoutValidation(
        config=config,
        present=present,
        missing=missing,
        shape_errors=shape_errors,
        shared_expert_format="dense_paro_w4",
    )
    if raise_on_error:
        result.raise_for_errors()
    return result


def materialize_qwen35_paro_full_attention_dense_c1_runtime_layer(
    index: WeightIndex,
    *,
    layer_id: int = 0,
    device: Device | None = None,
    runtime: HipRuntime | None = None,
    validate: bool = True,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> Qwen35ParoLayerDeviceWeights:
    validation = validate_qwen35_paro_full_attention_dense_c1_layout(
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
        runtime_full_attention_dense_c1_tensor_names(layer_id=layer_id),
        device=device,
        runtime=runtime,
        progress=progress,
        prepare_moe=False,
    )


def materialize_qwen35_paro_linear_attention_dense_c1_runtime_layer(
    index: WeightIndex,
    *,
    layer_id: int = 0,
    device: Device | None = None,
    runtime: HipRuntime | None = None,
    validate: bool = True,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> Qwen35ParoLayerDeviceWeights:
    validation = validate_qwen35_paro_linear_attention_dense_c1_layout(
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
        runtime_linear_attention_dense_c1_tensor_names(layer_id=layer_id),
        device=device,
        runtime=runtime,
        progress=progress,
        prepare_moe=False,
    )


def prepare_qwen35_paro_moe_c1_host_tensors(
    index: WeightIndex,
    *,
    layer_id: int = 0,
    normalized: dict[str, TensorInfo] | None = None,
    reader: "_NormalizedTensorReader | None" = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
    shared_expert_format: str | None = None,
) -> dict[str, object]:
    """Prepare parent-compatible MoE c=1 host layouts without torch.

    This mirrors the optimized parent stack's load-time preparation: router and
    shared-gate rows are concatenated, per-expert gate/up/down tensors are
    stacked on expert dimension 0, and decode pack8 qweights are transposed on
    the last two dimensions.
    """

    config = qwen35_paro_config_from_hf(index.config)
    normalized = normalized or _normalized_tensor_map(index)
    shared_expert_format = _resolve_shared_expert_format(
        normalized,
        layer_id=layer_id,
        shared_expert_format=shared_expert_format,
    )
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
        if shared_expert_format == SHARED_EXPERT_FORMAT_LEGACY_FP16:
            shared_gate = _read_normalized_numpy_tensor(normalized, f"{shared}.gate_proj.weight", reader=reader)
            shared_up = _read_normalized_numpy_tensor(normalized, f"{shared}.up_proj.weight", reader=reader)
            shared_down = _read_normalized_numpy_tensor(normalized, f"{shared}.down_proj.weight", reader=reader)
            gate_up_q, gate_up_scale = _quantize_w8a16_host(_concat_rows((shared_gate, shared_up)))
            down_q, down_scale = _quantize_w8a16_host(shared_down)
            prepared[f"{shared}.gate_up_weight_w8a16"] = gate_up_q
            prepared[f"{shared}.gate_up_weight_w8a16_scale"] = gate_up_scale
            prepared[f"{shared}.down_weight_w8a16"] = down_q
            prepared[f"{shared}.down_weight_w8a16_scale"] = down_scale
        else:
            for proj in ("gate_proj", "up_proj", "down_proj"):
                qweight = _read_normalized_numpy_tensor(normalized, f"{shared}.{proj}.qweight", reader=reader)
                prepared[f"{shared}.{proj}.qweight_pack8_decode"] = _transpose_generic_qweight(qweight)
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
        shared_expert_format=validation.shared_expert_format,
    )
    base = _materialize_normalized_layer(index, validation.config, layer_id, required, device=device, runtime=runtime)
    allocations = dict(base.weights.tensors)
    try:
        for name, array in prepare_qwen35_paro_moe_c1_host_tensors(
            index,
            layer_id=layer_id,
            shared_expert_format=validation.shared_expert_format,
        ).items():
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
    shared_expert_format: str | None = None,
) -> dict[str, object]:
    """Prepare decode-runtime MoE tensors with BF16 bit buffers where required."""

    _emit_progress(progress, "prepare_moe_start", layer=layer_id)
    prepared = prepare_qwen35_paro_moe_c1_host_tensors(
        index,
        layer_id=layer_id,
        normalized=normalized,
        reader=reader,
        progress=progress,
        shared_expert_format=shared_expert_format,
    )
    shared_expert_format = _resolve_shared_expert_format(
        normalized or _normalized_tensor_map(index),
        layer_id=layer_id,
        shared_expert_format=shared_expert_format,
    )
    runtime_prepared: dict[str, object] = {}
    for name in runtime_prepared_moe_c1_tensor_names(layer_id=layer_id, shared_expert_format=shared_expert_format):
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
    shared_expert_format: str | None = None,
) -> Qwen35ParoLayerDeviceWeights:
    """Materialize the current real full-attention decode-state layer path."""

    validation = validate_qwen35_paro_full_attention_moe_c1_layout(
        index,
        layer_id=layer_id,
        raise_on_error=validate,
        shared_expert_format=shared_expert_format,
    )
    if not validation.passed:
        validation.raise_for_errors()
    return _materialize_runtime_layer(
        index,
        validation.config,
        layer_id,
        runtime_full_attention_moe_c1_tensor_names(
            layer_id=layer_id,
            shared_expert_format=validation.shared_expert_format,
        ),
        device=device,
        runtime=runtime,
        progress=progress,
        shared_expert_format=validation.shared_expert_format,
    )


def materialize_qwen35_paro_linear_attention_moe_c1_runtime_layer(
    index: WeightIndex,
    *,
    layer_id: int = 0,
    device: Device | None = None,
    runtime: HipRuntime | None = None,
    validate: bool = True,
    progress: Callable[[dict[str, Any]], None] | None = None,
    shared_expert_format: str | None = None,
) -> Qwen35ParoLayerDeviceWeights:
    """Materialize the current real linear-attention decode-state layer path."""

    validation = validate_qwen35_paro_linear_attention_moe_c1_layout(
        index,
        layer_id=layer_id,
        raise_on_error=validate,
        shared_expert_format=shared_expert_format,
    )
    if not validation.passed:
        validation.raise_for_errors()
    return _materialize_runtime_layer(
        index,
        validation.config,
        layer_id,
        runtime_linear_attention_moe_c1_tensor_names(
            layer_id=layer_id,
            shared_expert_format=validation.shared_expert_format,
        ),
        device=device,
        runtime=runtime,
        progress=progress,
        shared_expert_format=validation.shared_expert_format,
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


def repack_paro_awq_to_marlin_k_host(
    qweight: object,
    qzeros: object,
    scales: object,
    *,
    bits: int = 4,
    group_size: int = 128,
):
    """Repack PARO/AWQ W4 tensors into the parent Marlin-K v0 host layout.

    Input layout mirrors the checkpoint/PARO pack8 layout used elsewhere in
    hipEngine: ``qweight [K, N/8]``, ``qzeros [K/group_size, N/8]``, and
    ``scales [K/group_size, N]``.  The returned layout matches the parent
    ``nano-vllm-amd`` qweight-neutral Marlin-K path documented in
    ``docs/MARLIN.md``:

    - ``qweight_mk [N/8, K/128, 128]``
    - ``qzeros_mk [N/8, K/128]``
    - ``scales_mk [N/8, K/128, 8]``
    """

    import numpy as np

    if bits != 4:
        raise ValueError(f"Marlin-K v0 supports bits=4 only, got {bits}")
    if group_size != 128:
        raise ValueError(f"Marlin-K v0 supports group_size=128 only, got {group_size}")
    qweight_arr = np.asarray(qweight)
    qzeros_arr = np.asarray(qzeros)
    scales_arr = np.asarray(scales)
    if qweight_arr.dtype != np.int32:
        raise ValueError(f"qweight dtype must be int32, got {qweight_arr.dtype}")
    if qzeros_arr.dtype != np.int32:
        raise ValueError(f"qzeros dtype must be int32, got {qzeros_arr.dtype}")
    if qweight_arr.ndim != 2:
        raise ValueError(f"qweight must be rank-2 [K, N/8], got shape {qweight_arr.shape}")
    if qzeros_arr.ndim != 2:
        raise ValueError(f"qzeros must be rank-2 [groups, N/8], got shape {qzeros_arr.shape}")
    if scales_arr.ndim != 2:
        raise ValueError(f"scales must be rank-2 [groups, N], got shape {scales_arr.shape}")
    in_features, out_packed = qweight_arr.shape
    if in_features % group_size != 0:
        raise ValueError(f"qweight K dimension {in_features} must be a multiple of group_size {group_size}")
    groups = in_features // group_size
    expected_qzeros = (groups, out_packed)
    if tuple(qzeros_arr.shape) != expected_qzeros:
        raise ValueError(f"qzeros shape must be {expected_qzeros}, got {qzeros_arr.shape}")
    expected_scales = (groups, out_packed * 8)
    if tuple(scales_arr.shape) != expected_scales:
        raise ValueError(f"scales shape must be {expected_scales}, got {scales_arr.shape}")
    qweight_contig = np.ascontiguousarray(qweight_arr, dtype=np.int32)
    qzeros_contig = np.ascontiguousarray(qzeros_arr, dtype=np.int32)
    scales_contig = np.ascontiguousarray(scales_arr)
    qweight_mk = np.ascontiguousarray(qweight_contig.reshape(groups, group_size, out_packed).transpose(2, 0, 1))
    qzeros_mk = np.ascontiguousarray(qzeros_contig.T)
    scales_mk = np.ascontiguousarray(scales_contig.reshape(groups, out_packed, 8).transpose(1, 0, 2))
    return qweight_mk, qzeros_mk, scales_mk


def paro_marlin_k_pack8_decode_view(qweight_mk: object):
    """Return the zero-copy pack8 decode view over ``qweight_mk``.

    The parent qweight-neutral path keeps one owning W4 buffer and exposes the
    existing pack8/fused paths through this view.  hipEngine runtime materialize
    code must preserve the same ownership property when this helper is used for
    device tensors.
    """

    import numpy as np

    qweight_mk_arr = np.asarray(qweight_mk)
    if qweight_mk_arr.dtype != np.int32:
        raise ValueError(f"qweight_mk dtype must be int32, got {qweight_mk_arr.dtype}")
    if qweight_mk_arr.ndim != 3 or qweight_mk_arr.shape[2] != 128:
        raise ValueError(f"qweight_mk shape must be [N/8, groups, 128], got {qweight_mk_arr.shape}")
    if not qweight_mk_arr.flags.c_contiguous:
        raise ValueError("qweight_mk must be C-contiguous to expose a zero-copy pack8 view")
    out_packed, groups, group_size = qweight_mk_arr.shape
    return qweight_mk_arr.reshape(out_packed, groups * group_size)


def _quantize_w8a16_host(weight: object):
    import numpy as np

    weight_f32 = np.asarray(weight, dtype=np.float32)
    scale = np.maximum(np.max(np.abs(weight_f32), axis=1), 1.0e-8).astype(np.float32) / np.float32(127.0)
    quantized = np.rint(weight_f32 / scale[:, None])
    quantized = np.clip(quantized, -127, 127).astype(np.int8)
    return np.ascontiguousarray(quantized), np.ascontiguousarray(scale)


def _transpose_generic_qweight(array: object):
    import numpy as np

    qweight = np.asarray(array)
    if qweight.ndim != 2:
        raise ValueError(
            f"generic qweight must be rank-2 [in_features, out_packed], got shape {qweight.shape}"
        )
    return np.ascontiguousarray(qweight.T)


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
    # exception in this runtime: parent first forms direct FP16 scales, then
    # passes BF16 offset weights to the fused head-rmsnorm+rotary kernel.
    return (
        name.endswith(".input_layernorm.weight")
        or name.endswith(".post_attention_layernorm.weight")
        or name in {"norm.weight", "language_model.norm.weight", "model.norm.weight"}
    )


def _runtime_tensor_needs_qwen_head_norm_offset_bits(name: str) -> bool:
    return name.endswith(".self_attn.q_norm.weight") or name.endswith(".self_attn.k_norm.weight")


def _runtime_tensor_needs_bf16_bits(name: str) -> bool:
    # Parent fused full-attention head RMSNorm receives q/k *offset* weights in
    # BF16, while the native router concatenates router/shared-gate weights and
    # casts that combined matrix to BF16 before calling the HIP top-k kernel.
    # Dense KV cache storage is BF16 too but is allocated by runtime state, not
    # checkpoint materialization.
    return _runtime_tensor_needs_qwen_head_norm_offset_bits(name) or name.endswith(".router_shared_gate.weight")


def _bf16_bits_to_float32(bits: object):
    import numpy as np

    return (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << np.uint32(16)).view(np.float32)


def _qwen_head_norm_offset_bf16_bits(array: object):
    """Match parent q/k head-RMSNorm offset preparation for fused rotary.

    Parent `load_paro_rmsnorm_module` first turns checkpoint offsets into direct
    scales in FP16 via `(weight + 1)`.  The fused head RMSNorm+RoPE path then
    passes `(scale.to(bfloat16) - 1)` to a kernel that adds 1.0 internally.
    """

    import numpy as np

    scale_fp16 = np.ascontiguousarray(np.asarray(array, dtype=np.float32) + np.float32(1.0), dtype=np.float16)
    scale_bf32 = _bf16_bits_to_float32(float_array_to_bf16_bits(scale_fp16))
    return float_array_to_bf16_bits(scale_bf32 - np.float32(1.0))


def _runtime_tensor_needs_fp16(name: str) -> bool:
    return (
        name.endswith(".weight")
        or name.endswith(".scales")
        or name.endswith(".scales_mk")
        or name.endswith("_scales")
        or name.endswith(".theta")
        or name.endswith("_theta")
        or name.endswith(".channel_scales")
        or name.endswith("_channel_scales")
    ) and not name.endswith("_w8a16_scale")


def _env_flag_enabled(name: str, *, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _use_paro_marlin_k_replacement() -> bool:
    return _env_flag_enabled("HIPENGINE_PARO_MARLIN_K_REPLACE", default=True)


def _non_expert_paro_linear_prefixes(config: Qwen35ParoConfig, *, layer_id: int) -> tuple[str, ...]:
    layer_type = config.layer_types[layer_id]
    if layer_type == "full_attention":
        attn = f"layers.{layer_id}.self_attn"
        return tuple(f"{attn}.{proj}" for proj in ("q_proj", "k_proj", "v_proj", "o_proj"))
    if layer_type == "linear_attention":
        linear = f"layers.{layer_id}.linear_attn"
        return tuple(f"{linear}.{proj}" for proj in ("in_proj_qkv", "in_proj_z", "out_proj"))
    return ()


def _prepare_linear_attention_qkv_z_pack8_runtime_tensors(
    normalized: dict[str, Any],
    *,
    names: tuple[str, ...],
    reader: _NormalizedTensorReader,
    layer_id: int,
    progress: Callable[[dict[str, Any]], None] | None = None,
    skip_sources: set[str] | None = None,
) -> dict[str, object]:
    """Prepare transposed generic qweights for fused linear-attention QKV/Z decode."""

    import numpy as np

    prefix = f"layers.{layer_id}.linear_attn"
    required = (f"{prefix}.in_proj_qkv.qweight", f"{prefix}.in_proj_z.qweight")
    skip_sources = skip_sources or set()
    if not all(name in names for name in required):
        return {}
    prepared: dict[str, object] = {}
    for source in required:
        if source in skip_sources:
            continue
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
    skip_sources: set[str] | None = None,
) -> dict[str, object]:
    """Prepare transposed generic qweights for fused full-attention Q/K decode."""

    import numpy as np

    prefix = f"layers.{layer_id}.self_attn"
    required = (f"{prefix}.q_proj.qweight", f"{prefix}.k_proj.qweight")
    skip_sources = skip_sources or set()
    if not all(name in names for name in required):
        return {}
    prepared: dict[str, object] = {}
    for source in required:
        if source in skip_sources:
            continue
        target = source.removesuffix(".qweight") + ".qweight_pack8_decode"
        _emit_progress(progress, "prepare_runtime_tensor_start", layer=layer_id, name=target)
        qweight = np.asarray(_read_normalized_numpy_tensor(normalized, source, reader=reader), dtype=np.int32)
        prepared[target] = np.ascontiguousarray(qweight.T)
        _emit_progress(progress, "prepare_runtime_tensor_done", layer=layer_id, name=target, shape=tuple(prepared[target].shape))
    return prepared


def _prepare_dense_mlp_pack8_runtime_tensors(
    normalized: dict[str, Any],
    *,
    names: tuple[str, ...],
    reader: _NormalizedTensorReader,
    layer_id: int,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, object]:
    """Prepare transposed generic qweights for dense PARO MLP decode/prefill."""

    import numpy as np

    prefix = f"layers.{layer_id}.mlp"
    prepared: dict[str, object] = {}
    for proj in ("gate_proj", "up_proj", "down_proj"):
        source = f"{prefix}.{proj}.qweight"
        if source not in names:
            continue
        target = source.removesuffix(".qweight") + ".qweight_pack8_decode"
        _emit_progress(progress, "prepare_runtime_tensor_start", layer=layer_id, name=target)
        qweight = np.asarray(_read_normalized_numpy_tensor(normalized, source, reader=reader), dtype=np.int32)
        prepared[target] = np.ascontiguousarray(qweight.T)
        _emit_progress(progress, "prepare_runtime_tensor_done", layer=layer_id, name=target, shape=tuple(prepared[target].shape))
    return prepared


def _prepare_non_expert_marlin_k_runtime_tensors(
    normalized: dict[str, Any],
    config: Qwen35ParoConfig,
    *,
    names: tuple[str, ...],
    reader: _NormalizedTensorReader,
    layer_id: int,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[dict[str, object], dict[str, tuple[str, tuple[int, ...]]], set[str]]:
    """Prepare qweight-neutral Marlin-K buffers and pack8 aliases.

    Returns ``(prepared_arrays, aliases, skipped_raw_qweights)``.  ``aliases``
    maps alias tensor name to ``(owner_name, alias_shape)`` and is materialized
    after the owner device allocation exists.
    """

    if not _use_paro_marlin_k_replacement():
        return {}, {}, set()

    import numpy as np

    prepared: dict[str, object] = {}
    aliases: dict[str, tuple[str, tuple[int, ...]]] = {}
    skipped_raw: set[str] = set()
    names_set = set(names)
    for prefix in _non_expert_paro_linear_prefixes(config, layer_id=layer_id):
        qweight_name = f"{prefix}.qweight"
        qzeros_name = f"{prefix}.qzeros"
        scales_name = f"{prefix}.scales"
        required = (qweight_name, qzeros_name, scales_name)
        if not all(name in names_set for name in required):
            continue
        _emit_progress(progress, "prepare_marlin_k_start", layer=layer_id, name=prefix)
        qweight = np.asarray(_read_normalized_numpy_tensor(normalized, qweight_name, reader=reader), dtype=np.int32)
        # Tiny unit fixtures use toy K values; Marlin-K v0 is only defined for
        # group_size=128 and exact K/group metadata.  Leave incompatible test or
        # fallback surfaces on the existing pack8 layout.
        if qweight.ndim != 2 or qweight.shape[0] % 128 != 0:
            _emit_progress(progress, "prepare_marlin_k_skip", layer=layer_id, name=prefix, reason="shape")
            continue
        qzeros = np.asarray(_read_normalized_numpy_tensor(normalized, qzeros_name, reader=reader), dtype=np.int32)
        scales = np.asarray(_read_normalized_numpy_tensor(normalized, scales_name, reader=reader), dtype=np.float16)
        qweight_mk, qzeros_mk, scales_mk = repack_paro_awq_to_marlin_k_host(qweight, qzeros, scales)
        owner_name = f"{prefix}.qweight_mk"
        prepared[owner_name] = qweight_mk
        prepared[f"{prefix}.qzeros_mk"] = qzeros_mk
        prepared[f"{prefix}.scales_mk"] = scales_mk
        aliases[f"{prefix}.qweight_pack8_decode"] = (owner_name, (qweight_mk.shape[0], qweight_mk.shape[1] * qweight_mk.shape[2]))
        skipped_raw.add(qweight_name)
        _emit_progress(
            progress,
            "prepare_marlin_k_done",
            layer=layer_id,
            name=prefix,
            qweight_shape=tuple(qweight_mk.shape),
        )
    return prepared, aliases, skipped_raw

def _materialize_runtime_layer(
    index: WeightIndex,
    config: Qwen35ParoConfig,
    layer_id: int,
    names: tuple[str, ...],
    *,
    device: Device | None,
    runtime: HipRuntime | None,
    progress: Callable[[dict[str, Any]], None] | None = None,
    shared_expert_format: str = SHARED_EXPERT_FORMAT_PACKED_PARO_W4,
    prepare_moe: bool = True,
) -> Qwen35ParoLayerDeviceWeights:
    normalized = _normalized_tensor_map(index)
    prepared_names = (
        set(runtime_prepared_moe_c1_tensor_names(layer_id=layer_id, shared_expert_format=shared_expert_format))
        if prepare_moe
        else set()
    )
    allocations: dict[str, DeviceTensorAllocation] = {}
    reader = _NormalizedTensorReader(normalized)
    try:
        marlin_prepared, marlin_aliases, marlin_raw_qweights = _prepare_non_expert_marlin_k_runtime_tensors(
            normalized,
            config,
            names=names,
            reader=reader,
            layer_id=layer_id,
            progress=progress,
        )
        direct_names = tuple(name for name in names if name not in prepared_names and name not in marlin_raw_qweights)
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
            elif _runtime_tensor_needs_qwen_head_norm_offset_bits(name):
                array = _qwen_head_norm_offset_bf16_bits(_read_normalized_numpy_tensor(normalized, name, reader=reader))
                allocations[name] = load_host_array_to_device_as_dtype(
                    name,
                    array,
                    DType.BF16,
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
        for idx, (name, array) in enumerate(marlin_prepared.items(), start=1):
            _emit_progress(
                progress,
                "materialize_marlin_k_tensor_start",
                layer=layer_id,
                name=name,
                index=idx,
                total=len(marlin_prepared),
            )
            if _runtime_tensor_needs_fp16(name):
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
                "materialize_marlin_k_tensor_done",
                layer=layer_id,
                name=name,
                index=idx,
                total=len(marlin_prepared),
            )
        for alias_name, (owner_name, alias_shape) in marlin_aliases.items():
            allocations[alias_name] = alias_device_allocation(
                alias_name,
                allocations[owner_name],
                alias_shape,
                DType.INT32,
                device=device,
            )
        linear_pack8 = _prepare_linear_attention_qkv_z_pack8_runtime_tensors(
            normalized,
            names=names,
            reader=reader,
            layer_id=layer_id,
            progress=progress,
            skip_sources=marlin_raw_qweights,
        )
        linear_pack8.update(
            _prepare_full_attention_qk_pack8_runtime_tensors(
                normalized,
                names=names,
                reader=reader,
                layer_id=layer_id,
                progress=progress,
                skip_sources=marlin_raw_qweights,
            )
        )
        linear_pack8.update(
            _prepare_dense_mlp_pack8_runtime_tensors(
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
        if prepare_moe:
            prepared = prepare_qwen35_paro_moe_c1_runtime_host_tensors(
                index,
                layer_id=layer_id,
                normalized=normalized,
                reader=reader,
                progress=progress,
                shared_expert_format=shared_expert_format,
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
    shared_expert_format: str = SHARED_EXPERT_FORMAT_PACKED_PARO_W4,
) -> tuple[str, ...]:
    prefix = f"layers.{layer_id}.mlp"
    expected: dict[str, tuple[int, ...]] = {
        f"layers.{layer_id}.post_attention_layernorm.weight": (config.hidden_size,),
        f"{prefix}.gate.weight": (config.num_experts, config.hidden_size),
        f"{prefix}.shared_expert_gate.weight": (1, config.hidden_size),
    }
    if shared_expert_format == SHARED_EXPERT_FORMAT_LEGACY_FP16:
        shared = f"{prefix}.shared_expert"
        expected.update(
            {
                f"{shared}.gate_proj.weight": (config.shared_expert_intermediate_size, config.hidden_size),
                f"{shared}.up_proj.weight": (config.shared_expert_intermediate_size, config.hidden_size),
                f"{shared}.down_proj.weight": (config.hidden_size, config.shared_expert_intermediate_size),
            }
        )
    # Packed shared-expert qweight/qzeros/scales/theta/pairs/channel_scales
    # mirror the attention dense-projection convention and are existence-only
    # validated (matching how routed-expert qweight/qzeros/scales are checked).

    errors: list[str] = []
    for name, shape in expected.items():
        info = tensors.get(name)
        if info is not None and info.shape != shape:
            errors.append(f"{name}: expected {shape}, got {info.shape}")
    return tuple(errors)
