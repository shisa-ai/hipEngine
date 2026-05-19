"""Torch-free Qwen3.5/Qwen3.6 target-attached MTP tensor metadata.

The current packed PARO target used by the DFlash lane does not contain
``mtp.*`` tensors, but the verifier/commit infrastructure is now shared enough
for MTP to plug in once a retained target-attached MTP artifact is available.
This module validates and materializes those tensors without importing torch.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from hipengine.core.device import Device
from hipengine.core.hip import HipRuntime
from hipengine.core.tensor import Tensor
from hipengine.loading.materialize import DeviceTensorAllocation, DeviceWeightMap, load_tensor_info_to_device
from hipengine.loading.qwen35_paro import Qwen35ParoConfig, normalize_qwen35_weight_name, qwen35_paro_config_from_hf
from hipengine.loading.safetensors import MissingTensorError, TensorInfo, WeightIndex, load_weight_index


@dataclass(frozen=True)
class MtpTensorRequirement:
    name: str
    dtype: str | tuple[str, ...]
    shape: tuple[int | None, ...]
    description: str = ""


@dataclass(frozen=True)
class Qwen35MtpConfig:
    architecture: str
    hidden_size: int
    vocab_size: int
    num_mtp_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rotary_dim: int
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    shared_expert_intermediate_size: int
    rms_norm_eps: float
    source_quant_method: str

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "architecture": self.architecture,
            "hidden_size": self.hidden_size,
            "vocab_size": self.vocab_size,
            "num_mtp_layers": self.num_mtp_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "rotary_dim": self.rotary_dim,
            "num_experts": self.num_experts,
            "num_experts_per_tok": self.num_experts_per_tok,
            "moe_intermediate_size": self.moe_intermediate_size,
            "shared_expert_intermediate_size": self.shared_expert_intermediate_size,
            "rms_norm_eps": self.rms_norm_eps,
            "source_quant_method": self.source_quant_method,
        }


@dataclass(frozen=True)
class Qwen35MtpValidation:
    model_path: str
    config: Qwen35MtpConfig
    present: tuple[str, ...]
    missing: tuple[str, ...]
    dtype_errors: tuple[str, ...]
    shape_errors: tuple[str, ...]
    unexpected_mtp: tuple[str, ...]
    config_errors: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return not (self.missing or self.dtype_errors or self.shape_errors or self.config_errors)

    def raise_for_errors(self) -> None:
        if self.passed:
            return
        parts: list[str] = []
        if self.config_errors:
            parts.append("config errors: " + _preview(self.config_errors))
        if self.missing:
            parts.append("missing tensors: " + _preview(self.missing))
        if self.dtype_errors:
            parts.append("dtype errors: " + _preview(self.dtype_errors, limit=4))
        if self.shape_errors:
            parts.append("shape errors: " + _preview(self.shape_errors, limit=4))
        raise MissingTensorError("; ".join(parts))

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": "qwen35_mtp_target_attached",
            "model_path": self.model_path,
            "passed": self.passed,
            "config": self.config.to_json_dict(),
            "present_count": len(self.present),
            "expected_count": len(qwen35_mtp_tensor_requirements(self.config)),
            "missing": list(self.missing),
            "dtype_errors": list(self.dtype_errors),
            "shape_errors": list(self.shape_errors),
            "unexpected_mtp": list(self.unexpected_mtp),
            "config_errors": list(self.config_errors),
        }


@dataclass(frozen=True)
class Qwen35MtpDeviceWeights:
    config: Qwen35MtpConfig
    weights: DeviceWeightMap

    def tensor(self, name: str) -> Tensor:
        return self.weights[normalize_qwen35_weight_name(name)]

    def allocation(self, name: str) -> DeviceTensorAllocation:
        return self.weights.allocation(normalize_qwen35_weight_name(name))

    def free(self, *, runtime: HipRuntime | None = None) -> None:
        self.weights.free(runtime=runtime)


def qwen35_mtp_config_from_target(config: Mapping[str, Any] | Qwen35ParoConfig) -> Qwen35MtpConfig:
    qwen = config if isinstance(config, Qwen35ParoConfig) else qwen35_paro_config_from_hf(dict(config))
    return Qwen35MtpConfig(
        architecture=f"{qwen.architecture}+MTP",
        hidden_size=qwen.hidden_size,
        vocab_size=qwen.vocab_size,
        num_mtp_layers=1,
        num_attention_heads=qwen.num_attention_heads,
        num_key_value_heads=qwen.num_key_value_heads,
        head_dim=qwen.head_dim,
        rotary_dim=qwen.rotary_dim,
        num_experts=qwen.num_experts,
        num_experts_per_tok=qwen.num_experts_per_tok,
        moe_intermediate_size=qwen.moe_intermediate_size,
        shared_expert_intermediate_size=qwen.shared_expert_intermediate_size,
        rms_norm_eps=qwen.rms_norm_eps,
        source_quant_method=qwen.quant_method,
    )


def qwen35_mtp_tensor_requirements(config: Qwen35MtpConfig) -> tuple[MtpTensorRequirement, ...]:
    hidden = int(config.hidden_size)
    attn_q = int(config.num_attention_heads) * int(config.head_dim)
    attn_kv = int(config.num_key_value_heads) * int(config.head_dim)
    moe = int(config.moe_intermediate_size)
    shared = int(config.shared_expert_intermediate_size)
    experts = int(config.num_experts)
    reqs: list[MtpTensorRequirement] = [
        MtpTensorRequirement("mtp.fc.weight", "BF16", (hidden, 2 * hidden), "embedding/hidden fusion projection"),
        MtpTensorRequirement("mtp.pre_fc_norm_embedding.weight", "BF16", (hidden,), "token embedding RMSNorm"),
        MtpTensorRequirement("mtp.pre_fc_norm_hidden.weight", "BF16", (hidden,), "target hidden RMSNorm"),
        MtpTensorRequirement("mtp.layers.0.input_layernorm.weight", "BF16", (hidden,), "MTP decoder input norm"),
        MtpTensorRequirement("mtp.layers.0.post_attention_layernorm.weight", "BF16", (hidden,), "MTP decoder post-attn norm"),
        MtpTensorRequirement("mtp.layers.0.self_attn.q_proj.weight", "BF16", (2 * attn_q, hidden), "Q+gate projection"),
        MtpTensorRequirement("mtp.layers.0.self_attn.k_proj.weight", "BF16", (attn_kv, hidden), "K projection"),
        MtpTensorRequirement("mtp.layers.0.self_attn.v_proj.weight", "BF16", (attn_kv, hidden), "V projection"),
        MtpTensorRequirement("mtp.layers.0.self_attn.o_proj.weight", "BF16", (hidden, attn_q), "O projection"),
        MtpTensorRequirement("mtp.layers.0.self_attn.q_norm.weight", "BF16", (config.head_dim,), "Q head norm"),
        MtpTensorRequirement("mtp.layers.0.self_attn.k_norm.weight", "BF16", (config.head_dim,), "K head norm"),
        MtpTensorRequirement("mtp.layers.0.mlp.gate.weight", "BF16", (experts, hidden), "router gate"),
        MtpTensorRequirement("mtp.layers.0.mlp.experts.gate_up_proj", "BF16", (experts, 2 * moe, hidden), "fused expert gate/up"),
        MtpTensorRequirement("mtp.layers.0.mlp.experts.down_proj", "BF16", (experts, hidden, moe), "expert down"),
        MtpTensorRequirement("mtp.layers.0.mlp.shared_expert_gate.weight", "BF16", (1, hidden), "shared expert gate"),
        MtpTensorRequirement("mtp.layers.0.mlp.shared_expert.gate_proj.weight", "BF16", (shared, hidden), "shared expert gate proj"),
        MtpTensorRequirement("mtp.layers.0.mlp.shared_expert.up_proj.weight", "BF16", (shared, hidden), "shared expert up proj"),
        MtpTensorRequirement("mtp.layers.0.mlp.shared_expert.down_proj.weight", "BF16", (hidden, shared), "shared expert down proj"),
        MtpTensorRequirement("mtp.norm.weight", "BF16", (hidden,), "MTP final norm"),
    ]
    return tuple(reqs)


def validate_qwen35_mtp_metadata(
    index: WeightIndex,
    *,
    raise_on_error: bool = False,
) -> Qwen35MtpValidation:
    config_errors: list[str] = []
    try:
        config = qwen35_mtp_config_from_target(index.config)
    except Exception as exc:
        fallback = Qwen35MtpConfig(
            architecture="invalid",
            hidden_size=0,
            vocab_size=0,
            num_mtp_layers=0,
            num_attention_heads=0,
            num_key_value_heads=0,
            head_dim=0,
            rotary_dim=0,
            num_experts=0,
            num_experts_per_tok=0,
            moe_intermediate_size=0,
            shared_expert_intermediate_size=0,
            rms_norm_eps=1.0e-6,
            source_quant_method="",
        )
        result = Qwen35MtpValidation(
            model_path=str(index.model_path),
            config=fallback,
            present=(),
            missing=(),
            dtype_errors=(),
            shape_errors=(),
            unexpected_mtp=(),
            config_errors=(f"invalid Qwen3.5/Qwen3.6 config: {exc}",),
        )
        if raise_on_error:
            result.raise_for_errors()
        return result

    if config.hidden_size <= 0 or config.vocab_size <= 0:
        config_errors.append("hidden_size and vocab_size must be positive")
    if config.num_mtp_layers != 1:
        config_errors.append("hipEngine MVP supports exactly one target-attached MTP layer")
    if config.num_attention_heads <= 0 or config.num_key_value_heads <= 0 or config.head_dim <= 0:
        config_errors.append("attention heads, kv heads, and head_dim must be positive")
    if config.num_attention_heads % config.num_key_value_heads != 0:
        config_errors.append("num_attention_heads must be divisible by num_key_value_heads")
    if config.num_experts <= 0 or config.num_experts_per_tok <= 0:
        config_errors.append("MTP MVP expects MoE metadata with positive experts/top-k")

    normalized = _normalized_tensor_map(index)
    requirements = qwen35_mtp_tensor_requirements(config)
    present, missing, dtype_errors, shape_errors = _validate_requirements(normalized, requirements)
    expected = {req.name for req in requirements}
    unexpected_mtp = tuple(sorted(name for name in normalized if name.startswith("mtp.") and name not in expected))
    result = Qwen35MtpValidation(
        model_path=str(index.model_path),
        config=config,
        present=present,
        missing=missing,
        dtype_errors=dtype_errors,
        shape_errors=shape_errors,
        unexpected_mtp=unexpected_mtp,
        config_errors=tuple(config_errors),
    )
    if raise_on_error:
        result.raise_for_errors()
    return result


def validate_qwen35_mtp_model(model: str | Path, *, raise_on_error: bool = False) -> Qwen35MtpValidation:
    return validate_qwen35_mtp_metadata(load_weight_index(model), raise_on_error=raise_on_error)


def qwen35_mtp_runtime_tensor_names(config: Qwen35MtpConfig) -> tuple[str, ...]:
    return tuple(req.name for req in qwen35_mtp_tensor_requirements(config))


def load_qwen35_mtp_bf16_weights(
    index: WeightIndex,
    *,
    device: Device | None = None,
    runtime: HipRuntime | None = None,
    validate: bool = True,
) -> Qwen35MtpDeviceWeights:
    """Materialize target-attached BF16 ``mtp.*`` tensors to device memory."""

    validation = validate_qwen35_mtp_metadata(index, raise_on_error=validate)
    config = validation.config
    names = qwen35_mtp_runtime_tensor_names(config)
    normalized = _normalized_tensor_map(index)
    allocations: dict[str, DeviceTensorAllocation] = {}
    try:
        for name in names:
            info = normalized[name]
            allocation = load_tensor_info_to_device(info, device=device, runtime=runtime)
            allocations[name] = DeviceTensorAllocation(
                name=name,
                source=allocation.source,
                buffer=allocation.buffer,
                tensor=allocation.tensor,
                owns_buffer=allocation.owns_buffer,
            )
    except Exception:
        DeviceWeightMap(allocations).free(runtime=runtime)
        raise
    return Qwen35MtpDeviceWeights(config=config, weights=DeviceWeightMap(allocations))


def _validate_requirements(
    tensors: Mapping[str, TensorInfo],
    requirements: Sequence[MtpTensorRequirement],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    present: list[str] = []
    missing: list[str] = []
    dtype_errors: list[str] = []
    shape_errors: list[str] = []
    for req in requirements:
        info = tensors.get(req.name)
        if info is None:
            missing.append(req.name)
            continue
        present.append(req.name)
        allowed = (req.dtype,) if isinstance(req.dtype, str) else req.dtype
        if info.dtype not in allowed:
            dtype_errors.append(f"{req.name}: expected dtype {_fmt_expected(allowed)}, got {info.dtype}")
        if not _shape_matches(info.shape, req.shape):
            shape_errors.append(f"{req.name}: expected shape {_fmt_shape(req.shape)}, got {tuple(info.shape)}")
    return tuple(present), tuple(missing), tuple(dtype_errors), tuple(shape_errors)


def _normalized_tensor_map(index: WeightIndex) -> dict[str, TensorInfo]:
    out: dict[str, TensorInfo] = {}
    for name, info in index.tensors.items():
        normalized = normalize_qwen35_weight_name(name)
        out[normalized] = TensorInfo(
            name=normalized,
            shard_path=info.shard_path,
            dtype=info.dtype,
            shape=info.shape,
        )
    return out


def _shape_matches(actual: tuple[int, ...], expected: tuple[int | None, ...]) -> bool:
    if len(actual) != len(expected):
        return False
    return all(exp is None or int(act) == int(exp) for act, exp in zip(actual, expected, strict=True))


def _fmt_expected(values: Sequence[str]) -> str:
    return values[0] if len(values) == 1 else "{" + ", ".join(values) + "}"


def _fmt_shape(shape: Sequence[int | None]) -> str:
    return "(" + ", ".join("*" if value is None else str(value) for value in shape) + ")"


def _preview(values: Sequence[str], *, limit: int = 8) -> str:
    preview = ", ".join(values[:limit])
    more = "" if len(values) <= limit else f" (+{len(values) - limit} more)"
    return preview + more


__all__ = [
    "MtpTensorRequirement",
    "Qwen35MtpConfig",
    "Qwen35MtpDeviceWeights",
    "Qwen35MtpValidation",
    "load_qwen35_mtp_bf16_weights",
    "qwen35_mtp_config_from_target",
    "qwen35_mtp_runtime_tensor_names",
    "qwen35_mtp_tensor_requirements",
    "validate_qwen35_mtp_metadata",
    "validate_qwen35_mtp_model",
]
