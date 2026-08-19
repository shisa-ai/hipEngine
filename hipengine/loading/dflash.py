"""Torch-free metadata validation for DFlash target/drafter artifacts."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from hipengine.core.hip import HipRuntime
from hipengine.core.device import Device
from hipengine.core.dtype import DType, dtype_itemsize
from hipengine.core.tensor import Tensor
from hipengine.loading.materialize import DeviceTensorAllocation, DeviceWeightMap, load_tensor_info_to_device
from hipengine.loading.qwen35_paro import (
    normalize_qwen35_weight_name,
    qwen35_paro_config_from_hf,
    runtime_full_attention_dense_c1_tensor_names,
    runtime_linear_attention_dense_c1_tensor_names,
)
from hipengine.loading.safetensors import MissingTensorError, TensorInfo, WeightIndex, load_weight_index

DFLASH_DRAFTER_MODEL = "z-lab/Qwen3.6-35B-A3B-DFlash"
DFLASH_PACKED_TARGET_MODEL = "shisa-ai/Qwen3.6-35B-A3B-PARO-full4096-e5-packed"


@dataclass(frozen=True)
class TensorRequirement:
    name: str
    dtype: str | tuple[str, ...]
    shape: tuple[int | None, ...] | None
    description: str = ""


@dataclass(frozen=True)
class DFlashDraftConfig:
    architecture: str
    block_size: int
    mask_token_id: int
    target_layer_ids: tuple[int, ...]
    num_target_layers: int
    hidden_size: int
    target_hidden_size: int
    target_hidden_concat_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rope_theta: float
    vocab_size: int
    dtype: str
    layer_types: tuple[str, ...]
    decoder_arch: str = "qwen"
    target_capture_depths: tuple[int, ...] = ()
    rms_norm_eps: float = 1.0e-6
    max_position_embeddings: int = 0
    sliding_windows: tuple[int, ...] = ()
    attention_gate_type: str = "none"
    qkv_layout: str = "separate"
    aux_hidden_norm_count: int = 0
    causal: bool = False
    draft_vocab_size: int = 0
    # DFlash2 grouped dynamic conv + candidate selector geometry (0 = not a DFlash2 drafter).
    conv_kernel_size: int = 0
    conv_group_size: int = 0
    selector_rank: int = 0
    selector_top_k: int = 0

    @property
    def q_features(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def kv_features(self) -> int:
        return self.num_key_value_heads * self.head_dim

    @property
    def qkv_features(self) -> int:
        return self.q_features + 2 * self.kv_features

    @property
    def is_dflash2(self) -> bool:
        return self.conv_group_size > 0 and self.conv_kernel_size > 0

    @property
    def conv_groups(self) -> int:
        """Number of dynamic-conv coefficient groups (hidden_size // group_size)."""

        if not self.is_dflash2 or self.hidden_size % self.conv_group_size != 0:
            return 0
        return self.hidden_size // self.conv_group_size

    @property
    def conv_projection_features(self) -> int:
        """Output width of kernel_projection: 2 * conv_kernel_size * conv_groups."""

        return 2 * self.conv_kernel_size * self.conv_groups

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "architecture": self.architecture,
            "decoder_arch": self.decoder_arch,
            "block_size": self.block_size,
            "mask_token_id": self.mask_token_id,
            "target_layer_ids": list(self.target_layer_ids),
            "target_capture_depths": list(self.target_capture_depths),
            "num_target_layers": self.num_target_layers,
            "hidden_size": self.hidden_size,
            "target_hidden_size": self.target_hidden_size,
            "target_hidden_concat_size": self.target_hidden_concat_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "rope_theta": self.rope_theta,
            "rms_norm_eps": self.rms_norm_eps,
            "max_position_embeddings": self.max_position_embeddings,
            "sliding_windows": list(self.sliding_windows),
            "attention_gate_type": self.attention_gate_type,
            "qkv_layout": self.qkv_layout,
            "aux_hidden_norm_count": self.aux_hidden_norm_count,
            "causal": self.causal,
            "vocab_size": self.vocab_size,
            "draft_vocab_size": self.draft_vocab_size,
            "dtype": self.dtype,
            "layer_types": list(self.layer_types),
            "conv_kernel_size": self.conv_kernel_size,
            "conv_group_size": self.conv_group_size,
            "selector_rank": self.selector_rank,
            "selector_top_k": self.selector_top_k,
        }


@dataclass(frozen=True)
class DFlashDrafterDeviceWeights:
    """Materialized BF16 DFlash drafter weights for the native root/query path."""

    config: DFlashDraftConfig
    weights: DeviceWeightMap
    layer_limit: int

    def tensor(self, name: str) -> Tensor:
        try:
            return self.weights[name]
        except KeyError:
            return self._fused_qkv_row_view(name)

    def qkv_row_views(self, layer: int) -> tuple[Tensor, Tensor, Tensor]:
        """Return zero-copy Q/K/V views over one fused Laguna QKV allocation."""

        prefix = f"layers.{int(layer)}.self_attn"
        return (
            self.tensor(f"{prefix}.q_proj.weight"),
            self.tensor(f"{prefix}.k_proj.weight"),
            self.tensor(f"{prefix}.v_proj.weight"),
        )

    def _fused_qkv_row_view(self, name: str) -> Tensor:
        if self.config.qkv_layout != "fused_qkv":
            raise KeyError(name)
        suffixes = {
            ".q_proj.weight": (0, self.config.q_features),
            ".k_proj.weight": (self.config.q_features, self.config.kv_features),
            ".v_proj.weight": (
                self.config.q_features + self.config.kv_features,
                self.config.kv_features,
            ),
        }
        for suffix, (row_start, row_count) in suffixes.items():
            if not name.endswith(suffix):
                continue
            prefix = name[: -len(suffix)]
            fused = self.weights[f"{prefix}.qkv_proj.weight"]
            if fused.dtype != DType.BF16 or fused.shape != (
                self.config.qkv_features,
                self.config.hidden_size,
            ):
                raise ValueError("fused QKV allocation does not match normalized DFlash config")
            row_bytes = self.config.hidden_size * dtype_itemsize(fused.dtype)
            return Tensor.from_handle(
                fused.ptr + row_start * row_bytes,
                (row_count, self.config.hidden_size),
                fused.dtype,
                fused.device,
            )
        raise KeyError(name)

    def allocation(self, name: str) -> DeviceTensorAllocation:
        return self.weights.allocation(name)

    def free(self, *, runtime: HipRuntime | None = None) -> None:
        self.weights.free(runtime=runtime)


@dataclass(frozen=True)
class DFlashTargetConfig:
    architecture: str
    num_hidden_layers: int
    hidden_size: int
    vocab_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    num_experts: int
    num_experts_per_tok: int
    quant_method: str
    shared_expert_format: str
    layer_types: tuple[str, ...]

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "architecture": self.architecture,
            "num_hidden_layers": self.num_hidden_layers,
            "hidden_size": self.hidden_size,
            "vocab_size": self.vocab_size,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "num_experts": self.num_experts,
            "num_experts_per_tok": self.num_experts_per_tok,
            "quant_method": self.quant_method,
            "shared_expert_format": self.shared_expert_format,
            "layer_types": list(self.layer_types),
        }


@dataclass(frozen=True)
class DFlashArtifactValidation:
    artifact_kind: str
    model_path: str
    config: DFlashDraftConfig | DFlashTargetConfig
    present: tuple[str, ...]
    missing: tuple[str, ...]
    dtype_errors: tuple[str, ...]
    shape_errors: tuple[str, ...]
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
            "artifact_kind": self.artifact_kind,
            "model_path": self.model_path,
            "passed": self.passed,
            "config": self.config.to_json_dict(),
            "present_count": len(self.present),
            "missing": list(self.missing),
            "dtype_errors": list(self.dtype_errors),
            "shape_errors": list(self.shape_errors),
            "config_errors": list(self.config_errors),
        }


DFlashDraftConfigParser = Callable[[Mapping[str, Any]], DFlashDraftConfig]
_DFLASH_DRAFT_CONFIG_PARSERS: dict[str, DFlashDraftConfigParser] = {}


def register_dflash_draft_config_parser(
    architecture: str,
    parser: DFlashDraftConfigParser,
    *,
    replace: bool = False,
) -> None:
    """Register an architecture-specific HF DFlash config normalizer."""

    name = str(architecture)
    if not name:
        raise ValueError("DFlash architecture must be non-empty")
    if name in _DFLASH_DRAFT_CONFIG_PARSERS and not replace:
        raise ValueError(f"DFlash architecture {name!r} is already registered")
    _DFLASH_DRAFT_CONFIG_PARSERS[name] = parser


def dflash_draft_config_from_hf(config: Mapping[str, Any]) -> DFlashDraftConfig:
    architectures = config.get("architectures") or ()
    architecture = str(architectures[0]) if architectures else "DFlashDraftModel"
    parser = _DFLASH_DRAFT_CONFIG_PARSERS.get(architecture)
    if parser is None:
        supported = ", ".join(sorted(_DFLASH_DRAFT_CONFIG_PARSERS))
        raise ValueError(
            f"unregistered DFlash architecture {architecture!r}; expected one of: {supported}"
        )
    return parser(config)


def _parse_qwen_dflash_config(config: Mapping[str, Any]) -> DFlashDraftConfig:
    dflash_config = _dflash_config_mapping(config)
    target_layer_ids = _int_tuple(dflash_config.get("target_layer_ids"))
    hidden_size = int(config["hidden_size"])
    num_layers = int(config["num_hidden_layers"])
    num_attention_heads = int(config["num_attention_heads"])
    capture_depths = _int_tuple(config.get("eagle_aux_hidden_state_layer_ids"))
    if not capture_depths:
        capture_depths = tuple(layer + 1 for layer in target_layer_ids)
    return DFlashDraftConfig(
        architecture="DFlashDraftModel",
        decoder_arch="qwen",
        block_size=_dflash_int(dflash_config, config, "block_size"),
        mask_token_id=int(dflash_config["mask_token_id"]),
        target_layer_ids=target_layer_ids,
        target_capture_depths=capture_depths,
        num_target_layers=int(config.get("num_target_layers", 0) or 0),
        hidden_size=hidden_size,
        target_hidden_size=int(config.get("target_hidden_size", hidden_size)),
        target_hidden_concat_size=len(target_layer_ids) * hidden_size,
        intermediate_size=int(config["intermediate_size"]),
        num_hidden_layers=num_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=int(config.get("num_key_value_heads", num_attention_heads)),
        head_dim=int(config.get("head_dim", hidden_size // num_attention_heads)),
        rope_theta=float(config.get("rope_theta", 10000.0)),
        rms_norm_eps=float(config.get("rms_norm_eps", 1.0e-6)),
        max_position_embeddings=int(config.get("max_position_embeddings", 0) or 0),
        sliding_windows=_normalized_sliding_windows(config, num_layers),
        attention_gate_type="none",
        qkv_layout="separate",
        aux_hidden_norm_count=0,
        causal=bool(dflash_config.get("causal", False)),
        vocab_size=int(config["vocab_size"]),
        draft_vocab_size=int(config.get("draft_vocab_size", config["vocab_size"])),
        dtype=str(config.get("dtype", config.get("torch_dtype", "bfloat16"))),
        layer_types=_layer_types(config, num_layers),
    )


def _parse_qwen_dflash2_config(config: Mapping[str, Any]) -> DFlashDraftConfig:
    """Normalize a DFlash2DraftModel HF config (Qwen backbone + grouped dynamic conv + selector)."""

    dflash_config = _dflash_config_mapping(config)
    base = _parse_qwen_dflash_config(config)
    return dataclasses.replace(
        base,
        architecture="DFlash2DraftModel",
        conv_kernel_size=int(dflash_config.get("conv_kernel_size", 0) or 0),
        conv_group_size=int(dflash_config.get("conv_group_size", 0) or 0),
        selector_rank=int(dflash_config.get("selector_rank", 0) or 0),
        selector_top_k=int(dflash_config.get("selector_top_k", 0) or 0),
    )


def _parse_laguna_dflash_config(config: Mapping[str, Any]) -> DFlashDraftConfig:
    dflash_config = _dflash_config_mapping(config)
    target_layer_ids = _int_tuple(dflash_config.get("target_layer_ids"))
    capture_depths = _int_tuple(config.get("eagle_aux_hidden_state_layer_ids"))
    if not capture_depths:
        capture_depths = tuple(layer + 1 for layer in target_layer_ids)
    hidden_size = int(config["hidden_size"])
    num_layers = int(config["num_hidden_layers"])
    num_attention_heads = int(config["num_attention_heads"])
    gating = str(config.get("gating", "per-head")).replace("-", "_")
    return DFlashDraftConfig(
        architecture="DFlashLagunaForCausalLM",
        decoder_arch="laguna",
        block_size=int(dflash_config["block_size"]),
        mask_token_id=int(dflash_config["mask_token_id"]),
        target_layer_ids=target_layer_ids,
        target_capture_depths=capture_depths,
        num_target_layers=int(dflash_config["num_target_layers"]),
        hidden_size=hidden_size,
        target_hidden_size=int(config.get("target_hidden_size", hidden_size)),
        target_hidden_concat_size=len(target_layer_ids) * hidden_size,
        intermediate_size=int(config["intermediate_size"]),
        num_hidden_layers=num_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=int(config["num_key_value_heads"]),
        head_dim=int(config["head_dim"]),
        rope_theta=float(config["rope_theta"]),
        rms_norm_eps=float(config.get("rms_norm_eps", 1.0e-6)),
        max_position_embeddings=int(config.get("max_position_embeddings", 0) or 0),
        sliding_windows=_normalized_sliding_windows(config, num_layers),
        attention_gate_type=gating,
        qkv_layout="fused_qkv",
        aux_hidden_norm_count=len(target_layer_ids),
        causal=bool(dflash_config.get("causal", False)),
        vocab_size=int(config["vocab_size"]),
        draft_vocab_size=int(config.get("draft_vocab_size", config["vocab_size"])),
        dtype=str(config.get("torch_dtype", config.get("dtype", "bfloat16"))),
        layer_types=_layer_types(config, num_layers),
    )


def _dflash_int(dflash_config: Mapping[str, Any], config: Mapping[str, Any], key: str) -> int:
    """Read an int from dflash_config first, then the top-level HF config."""

    for source in (dflash_config, config):
        value = source.get(key)
        if value is not None:
            return int(value)
    return 0


def _dflash_config_mapping(config: Mapping[str, Any]) -> Mapping[str, Any]:
    value = config.get("dflash_config")
    return value if isinstance(value, Mapping) else {}


def _int_tuple(value: object) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return ()
    return tuple(int(item) for item in value)


def _layer_types(config: Mapping[str, Any], num_layers: int) -> tuple[str, ...]:
    return tuple(str(item) for item in (config.get("layer_types") or ("full_attention",) * num_layers))


def _normalized_sliding_windows(config: Mapping[str, Any], num_layers: int) -> tuple[int, ...]:
    explicit = _int_tuple(config.get("sliding_windows"))
    if explicit:
        return explicit
    window = int(config.get("sliding_window", 0) or 0)
    return tuple(window if layer_type == "sliding_attention" else 0 for layer_type in _layer_types(config, num_layers))


register_dflash_draft_config_parser("DFlashDraftModel", _parse_qwen_dflash_config)
register_dflash_draft_config_parser("DFlash2DraftModel", _parse_qwen_dflash2_config)
register_dflash_draft_config_parser("DFlashLagunaForCausalLM", _parse_laguna_dflash_config)


def validate_dflash_drafter_metadata(
    index: WeightIndex,
    *,
    target_config: DFlashTargetConfig | None = None,
    raise_on_error: bool = False,
) -> DFlashArtifactValidation:
    config_errors: list[str] = []
    try:
        config = dflash_draft_config_from_hf(index.config)
    except Exception as exc:
        fallback = DFlashDraftConfig(
            architecture="invalid",
            block_size=0,
            mask_token_id=-1,
            target_layer_ids=(),
            num_target_layers=0,
            hidden_size=0,
            target_hidden_size=0,
            target_hidden_concat_size=0,
            intermediate_size=0,
            num_hidden_layers=0,
            num_attention_heads=0,
            num_key_value_heads=0,
            head_dim=0,
            rope_theta=10000.0,
            vocab_size=0,
            dtype="",
            layer_types=(),
        )
        result = DFlashArtifactValidation(
            artifact_kind="dflash_drafter",
            model_path=str(index.model_path),
            config=fallback,
            present=(),
            missing=(),
            dtype_errors=(),
            shape_errors=(),
            config_errors=(f"invalid DFlash config: {exc}",),
        )
        if raise_on_error:
            result.raise_for_errors()
        return result

    config_errors.extend(_validate_dflash_config(config, target_config=target_config))
    requirements = dflash_drafter_tensor_requirements(config)
    present, missing, dtype_errors, shape_errors = _validate_requirements(index.tensors, requirements)
    result = DFlashArtifactValidation(
        artifact_kind="dflash_drafter",
        model_path=str(index.model_path),
        config=config,
        present=present,
        missing=missing,
        dtype_errors=dtype_errors,
        shape_errors=shape_errors,
        config_errors=tuple(config_errors),
    )
    if raise_on_error:
        result.raise_for_errors()
    return result


def validate_dflash_target_metadata(
    index: WeightIndex,
    *,
    raise_on_error: bool = False,
) -> DFlashArtifactValidation:
    config_errors: list[str] = []
    try:
        qwen = qwen35_paro_config_from_hf(index.config)
        config = DFlashTargetConfig(
            architecture=qwen.architecture,
            num_hidden_layers=qwen.num_hidden_layers,
            hidden_size=qwen.hidden_size,
            vocab_size=qwen.vocab_size,
            num_attention_heads=qwen.num_attention_heads,
            num_key_value_heads=qwen.num_key_value_heads,
            head_dim=qwen.head_dim,
            num_experts=qwen.num_experts,
            num_experts_per_tok=qwen.num_experts_per_tok,
            quant_method=qwen.quant_method,
            shared_expert_format="dense_paro_w4" if qwen.num_experts <= 0 else "packed_paro_w4",
            layer_types=qwen.layer_types,
        )
    except Exception as exc:
        fallback = DFlashTargetConfig(
            architecture="invalid",
            num_hidden_layers=0,
            hidden_size=0,
            vocab_size=0,
            num_attention_heads=0,
            num_key_value_heads=0,
            head_dim=0,
            num_experts=0,
            num_experts_per_tok=0,
            quant_method="",
            shared_expert_format="",
            layer_types=(),
        )
        result = DFlashArtifactValidation(
            artifact_kind="dflash_target",
            model_path=str(index.model_path),
            config=fallback,
            present=(),
            missing=(),
            dtype_errors=(),
            shape_errors=(),
            config_errors=(f"invalid target config: {exc}",),
        )
        if raise_on_error:
            result.raise_for_errors()
        return result

    if config.quant_method != "paroquant":
        config_errors.append(f"expected quant_method='paroquant', got {config.quant_method!r}")
    if config.hidden_size <= 0 or config.vocab_size <= 0:
        config_errors.append("target hidden_size and vocab_size must be positive")
    if config.num_hidden_layers <= 0 or len(config.layer_types) != config.num_hidden_layers:
        config_errors.append(
            f"layer_types length {len(config.layer_types)} does not match num_hidden_layers {config.num_hidden_layers}"
        )

    normalized = _normalized_tensor_map(index)
    requirements = dflash_target_tensor_requirements(config)
    present, missing, dtype_errors, shape_errors = _validate_requirements(normalized, requirements)
    result = DFlashArtifactValidation(
        artifact_kind="dflash_target",
        model_path=str(index.model_path),
        config=config,
        present=present,
        missing=missing,
        dtype_errors=dtype_errors,
        shape_errors=shape_errors,
        config_errors=tuple(config_errors),
    )
    if raise_on_error:
        result.raise_for_errors()
    return result


def validate_dflash_artifact_pair(
    *,
    target_model: str | Path = DFLASH_PACKED_TARGET_MODEL,
    drafter_model: str | Path = DFLASH_DRAFTER_MODEL,
    raise_on_error: bool = False,
) -> dict[str, Any]:
    target_index = load_weight_index(target_model)
    drafter_index = load_weight_index(drafter_model)
    target = validate_dflash_target_metadata(target_index, raise_on_error=raise_on_error)
    drafter = validate_dflash_drafter_metadata(
        drafter_index,
        target_config=target.config if isinstance(target.config, DFlashTargetConfig) else None,
        raise_on_error=raise_on_error,
    )
    pair_errors: list[str] = []
    if isinstance(target.config, DFlashTargetConfig) and isinstance(drafter.config, DFlashDraftConfig):
        if target.config.hidden_size != drafter.config.target_hidden_size:
            pair_errors.append(
                f"target hidden_size {target.config.hidden_size} != drafter target_hidden_size {drafter.config.target_hidden_size}"
            )
        if target.config.vocab_size != drafter.config.vocab_size:
            pair_errors.append(f"target vocab_size {target.config.vocab_size} != drafter vocab_size {drafter.config.vocab_size}")
        if target.config.num_hidden_layers != drafter.config.num_target_layers:
            pair_errors.append(
                f"target layers {target.config.num_hidden_layers} != drafter num_target_layers {drafter.config.num_target_layers}"
            )
    passed = target.passed and drafter.passed and not pair_errors
    if raise_on_error and pair_errors:
        raise MissingTensorError("pair errors: " + _preview(pair_errors, limit=4))
    return {
        "passed": passed,
        "target": target.to_json_dict(),
        "drafter": drafter.to_json_dict(),
        "pair_errors": pair_errors,
    }


def dflash_drafter_runtime_tensor_names(
    config: DFlashDraftConfig,
    *,
    layer_limit: int | None = None,
    include_terminal: bool = True,
) -> tuple[str, ...]:
    """Return BF16 tensor names consumed by native DFlash drafter execution."""

    layers = config.num_hidden_layers if layer_limit is None else int(layer_limit)
    if layers < 0 or layers > config.num_hidden_layers:
        raise ValueError(f"layer_limit must be in [0, {config.num_hidden_layers}], got {layer_limit}")
    names = [
        *(f"aux_hidden_norms.{index}.weight" for index in range(config.aux_hidden_norm_count)),
        "fc.weight",
        "hidden_norm.weight",
    ]
    for layer in range(layers):
        prefix = f"layers.{layer}"
        attention = (
            (
                f"{prefix}.self_attn.qkv_proj.weight",
                f"{prefix}.self_attn.g_proj.weight",
            )
            if config.qkv_layout == "fused_qkv"
            else (
                f"{prefix}.self_attn.q_proj.weight",
                f"{prefix}.self_attn.k_proj.weight",
                f"{prefix}.self_attn.v_proj.weight",
            )
        )
        names.extend(
            (
                f"{prefix}.input_layernorm.weight",
                f"{prefix}.post_attention_layernorm.weight",
                *attention,
                f"{prefix}.self_attn.o_proj.weight",
                f"{prefix}.self_attn.q_norm.weight",
                f"{prefix}.self_attn.k_norm.weight",
                f"{prefix}.mlp.gate_proj.weight",
                f"{prefix}.mlp.up_proj.weight",
                f"{prefix}.mlp.down_proj.weight",
            )
        )
    if include_terminal:
        names.append("norm.weight")
    if config.is_dflash2:
        for layer in range(layers):
            prefix = f"layers.{layer}"
            names.extend(
                (
                    f"{prefix}.attention_conv.base_kernel",
                    f"{prefix}.attention_conv.kernel_projection.weight",
                    f"{prefix}.mlp_conv.base_kernel",
                    f"{prefix}.mlp_conv.kernel_projection.weight",
                )
            )
        names.extend(
            (
                "candidate_selector.hidden_projection.weight",
                "candidate_selector.predecessor_codebook",
                "candidate_selector.successor_codebook",
            )
        )
    return tuple(names)


def load_dflash_drafter_bf16_weights(
    index: WeightIndex,
    *,
    device: Device | None = None,
    runtime: HipRuntime | None = None,
    validate: bool = True,
    layer_limit: int | None = None,
) -> DFlashDrafterDeviceWeights:
    """Materialize registered DFlash drafter BF16 weights without PyTorch.

    BF16 payloads are copied byte-for-byte from safetensors storage via the raw
    header offsets, avoiding NumPy's missing bfloat16 dtype support and avoiding
    tensor materialization on the host beyond a transient uint16 view.
    """

    validation = validate_dflash_drafter_metadata(index, raise_on_error=validate)
    config = validation.config
    if not isinstance(config, DFlashDraftConfig):
        raise TypeError("DFlash drafter validation did not return a draft config")
    layers = config.num_hidden_layers if layer_limit is None else int(layer_limit)
    if layers < 0 or layers > config.num_hidden_layers:
        raise ValueError(f"layer_limit must be in [0, {config.num_hidden_layers}], got {layer_limit}")
    names = dflash_drafter_runtime_tensor_names(config, layer_limit=layers)
    allocations: dict[str, DeviceTensorAllocation] = {}
    try:
        for name in names:
            info = index.require((name,))[0]
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
    return DFlashDrafterDeviceWeights(
        config=config,
        weights=DeviceWeightMap(allocations),
        layer_limit=layers,
    )


def dflash_drafter_tensor_requirements(config: DFlashDraftConfig) -> tuple[TensorRequirement, ...]:
    reqs: list[TensorRequirement] = [
        *(
            TensorRequirement(
                f"aux_hidden_norms.{index}.weight",
                "BF16",
                (config.hidden_size,),
                "per-capture target hidden norm",
            )
            for index in range(config.aux_hidden_norm_count)
        ),
        TensorRequirement("fc.weight", "BF16", (config.hidden_size, config.target_hidden_concat_size), "target hidden projection"),
        TensorRequirement("hidden_norm.weight", "BF16", (config.hidden_size,), "post-fc target hidden norm"),
        TensorRequirement("norm.weight", "BF16", (config.hidden_size,), "final drafter norm"),
    ]
    for layer in range(config.num_hidden_layers):
        prefix = f"layers.{layer}"
        attention_requirements = (
            (
                TensorRequirement(
                    f"{prefix}.self_attn.qkv_proj.weight",
                    "BF16",
                    (config.qkv_features, config.hidden_size),
                    "fused Q/K/V rows",
                ),
                TensorRequirement(
                    f"{prefix}.self_attn.g_proj.weight",
                    "BF16",
                    (_attention_gate_width(config), config.hidden_size),
                    "Laguna softplus attention gate",
                ),
            )
            if config.qkv_layout == "fused_qkv"
            else (
                TensorRequirement(f"{prefix}.self_attn.q_proj.weight", "BF16", (config.q_features, config.hidden_size)),
                TensorRequirement(f"{prefix}.self_attn.k_proj.weight", "BF16", (config.kv_features, config.hidden_size)),
                TensorRequirement(f"{prefix}.self_attn.v_proj.weight", "BF16", (config.kv_features, config.hidden_size)),
            )
        )
        reqs.extend(
            (
                TensorRequirement(f"{prefix}.input_layernorm.weight", "BF16", (config.hidden_size,)),
                TensorRequirement(f"{prefix}.post_attention_layernorm.weight", "BF16", (config.hidden_size,)),
                *attention_requirements,
                TensorRequirement(f"{prefix}.self_attn.o_proj.weight", "BF16", (config.hidden_size, config.q_features)),
                TensorRequirement(f"{prefix}.self_attn.q_norm.weight", "BF16", (config.head_dim,)),
                TensorRequirement(f"{prefix}.self_attn.k_norm.weight", "BF16", (config.head_dim,)),
                TensorRequirement(f"{prefix}.mlp.gate_proj.weight", "BF16", (config.intermediate_size, config.hidden_size)),
                TensorRequirement(f"{prefix}.mlp.up_proj.weight", "BF16", (config.intermediate_size, config.hidden_size)),
                TensorRequirement(f"{prefix}.mlp.down_proj.weight", "BF16", (config.hidden_size, config.intermediate_size)),
            )
        )
    if config.is_dflash2:
        for layer in range(config.num_hidden_layers):
            prefix = f"layers.{layer}"
            reqs.extend(
                (
                    TensorRequirement(
                        f"{prefix}.attention_conv.base_kernel",
                        "BF16",
                        (2, config.conv_kernel_size, config.hidden_size),
                        "attention grouped dynamic conv base kernel",
                    ),
                    TensorRequirement(
                        f"{prefix}.attention_conv.kernel_projection.weight",
                        "BF16",
                        (config.conv_projection_features, config.hidden_size),
                        "attention grouped dynamic conv projection",
                    ),
                    TensorRequirement(
                        f"{prefix}.mlp_conv.base_kernel",
                        "BF16",
                        (2, config.conv_kernel_size, config.hidden_size),
                        "mlp grouped dynamic conv base kernel",
                    ),
                    TensorRequirement(
                        f"{prefix}.mlp_conv.kernel_projection.weight",
                        "BF16",
                        (config.conv_projection_features, config.hidden_size),
                        "mlp grouped dynamic conv projection",
                    ),
                )
            )
        reqs.extend(
            (
                TensorRequirement(
                    "candidate_selector.hidden_projection.weight",
                    "BF16",
                    (config.selector_rank, config.hidden_size),
                    "selector context gate projection",
                ),
                TensorRequirement(
                    "candidate_selector.predecessor_codebook",
                    "BF16",
                    (config.vocab_size, config.selector_rank),
                    "selector predecessor codebook",
                ),
                TensorRequirement(
                    "candidate_selector.successor_codebook",
                    "BF16",
                    (config.vocab_size, config.selector_rank),
                    "selector successor codebook",
                ),
            )
        )
    return tuple(reqs)


def _attention_gate_width(config: DFlashDraftConfig) -> int:
    if config.attention_gate_type == "per_head":
        return config.num_attention_heads
    if config.attention_gate_type == "per_element":
        return config.q_features
    raise ValueError(
        f"DFlash architecture {config.architecture!r} has no supported attention gate"
    )


def dflash_target_tensor_requirements(config: DFlashTargetConfig) -> tuple[TensorRequirement, ...]:
    reqs: list[TensorRequirement] = [
        TensorRequirement("embed_tokens.weight", ("F16", "BF16"), (config.vocab_size, config.hidden_size), "token embedding"),
        TensorRequirement("lm_head.weight", ("F16", "BF16"), (config.vocab_size, config.hidden_size), "LM head"),
    ]
    if config.num_experts <= 0:
        for layer, layer_type in enumerate(config.layer_types):
            if layer_type == "full_attention":
                names = runtime_full_attention_dense_c1_tensor_names(layer_id=layer)
            elif layer_type == "linear_attention":
                names = runtime_linear_attention_dense_c1_tensor_names(layer_id=layer)
            else:
                reqs.append(TensorRequirement(f"layers.{layer}", "F16", (), f"unsupported dense PARO layer_type {layer_type!r}"))
                continue
            reqs.extend(
                TensorRequirement(name, ("F16", "BF16", "I32", "I16"), None, "dense PARO runtime tensor")
                for name in names
            )
        return tuple(reqs)
    for layer in range(config.num_hidden_layers):
        shared = f"layers.{layer}.mlp.shared_expert"
        for proj in ("gate_proj", "up_proj", "down_proj"):
            base = f"{shared}.{proj}"
            reqs.extend(
                (
                    TensorRequirement(f"{base}.qweight", "I32", (None, None), "packed PARO qweight"),
                    TensorRequirement(f"{base}.qzeros", "I32", (None, None), "packed PARO qzeros"),
                    TensorRequirement(f"{base}.scales", ("F16", "BF16"), (None, None), "packed PARO scales"),
                    TensorRequirement(f"{base}.theta", ("F16", "BF16"), (None, None), "packed PARO theta"),
                    TensorRequirement(f"{base}.pairs", "I16", (None, config.hidden_size if proj != "down_proj" else None), "packed PARO pairs"),
                    TensorRequirement(f"{base}.channel_scales", ("F16", "BF16"), (1, None), "packed PARO channel scales"),
                )
            )
    return tuple(reqs)


def validate_dflash_drafter_against_gguf_target(
    drafter: DFlashDraftConfig,
    *,
    num_target_layers: int,
    hidden_size: int,
    vocab_size: int,
) -> tuple[str, ...]:
    """Pair-check a DFlash drafter config against GGUF target metadata.

    The GGUF target carries no PARO-specific geometry, so this validates only
    the shared contract: target layer count, hidden size, vocab size, and the
    in-range target-layer taps.
    """

    errors: list[str] = []
    if drafter.num_target_layers != num_target_layers:
        errors.append(
            f"num_target_layers {drafter.num_target_layers} does not match GGUF layers {num_target_layers}"
        )
    if drafter.target_hidden_size != hidden_size:
        errors.append(
            f"target_hidden_size {drafter.target_hidden_size} does not match GGUF hidden_size {hidden_size}"
        )
    if drafter.vocab_size != vocab_size:
        errors.append(f"vocab_size {drafter.vocab_size} does not match GGUF vocab_size {vocab_size}")
    for layer in drafter.target_layer_ids:
        if layer < 0 or layer >= num_target_layers:
            errors.append(f"target_layer_ids includes {layer} outside [0, {num_target_layers})")
    return tuple(errors)


def _validate_dflash_config(config: DFlashDraftConfig, *, target_config: DFlashTargetConfig | None) -> tuple[str, ...]:
    errors: list[str] = []
    if config.architecture not in _DFLASH_DRAFT_CONFIG_PARSERS:
        errors.append(f"unregistered DFlash architecture {config.architecture!r}")
    if config.block_size <= 0:
        errors.append(f"block_size must be positive, got {config.block_size}")
    if config.mask_token_id < 0:
        errors.append(f"mask_token_id must be non-negative, got {config.mask_token_id}")
    if not config.target_layer_ids:
        errors.append("dflash_config.target_layer_ids must be non-empty")
    if any(layer < 0 or layer >= config.num_target_layers for layer in config.target_layer_ids):
        errors.append(
            f"target_layer_ids {list(config.target_layer_ids)} must be within [0, {config.num_target_layers})"
        )
    if len(set(config.target_layer_ids)) != len(config.target_layer_ids):
        errors.append(f"target_layer_ids must be unique, got {list(config.target_layer_ids)}")
    expected_depths = tuple(layer + 1 for layer in config.target_layer_ids)
    if config.target_capture_depths != expected_depths:
        errors.append(
            "post-layer capture depths must equal zero-based target_layer_ids + 1: "
            f"expected {list(expected_depths)}, got {list(config.target_capture_depths)}"
        )
    if len(config.layer_types) != config.num_hidden_layers:
        errors.append(f"layer_types length {len(config.layer_types)} does not match num_hidden_layers {config.num_hidden_layers}")
    if config.sliding_windows and len(config.sliding_windows) != config.num_hidden_layers:
        errors.append(
            f"sliding_windows length {len(config.sliding_windows)} does not match num_hidden_layers {config.num_hidden_layers}"
        )
    if config.hidden_size <= 0 or config.intermediate_size <= 0:
        errors.append("hidden_size and intermediate_size must be positive")
    if config.num_attention_heads <= 0 or config.num_key_value_heads <= 0 or config.head_dim <= 0:
        errors.append("attention heads, kv heads, and head_dim must be positive")
    if config.num_attention_heads % config.num_key_value_heads != 0:
        errors.append(
            f"num_attention_heads {config.num_attention_heads} must be divisible by num_key_value_heads {config.num_key_value_heads}"
        )
    if config.rms_norm_eps <= 0.0:
        errors.append(f"rms_norm_eps must be positive, got {config.rms_norm_eps}")
    if config.draft_vocab_size != config.vocab_size:
        errors.append(
            f"draft_vocab_size {config.draft_vocab_size} does not match target vocab_size {config.vocab_size}"
        )
    if config.decoder_arch == "laguna":
        if config.qkv_layout != "fused_qkv":
            errors.append("Laguna DFlash requires fused_qkv projection layout")
        if config.aux_hidden_norm_count != len(config.target_layer_ids):
            errors.append(
                "Laguna DFlash auxiliary norm count must match target capture count"
            )
        if config.attention_gate_type not in {"per_head", "per_element"}:
            errors.append(
                f"Laguna DFlash attention gate must be per_head or per_element, got {config.attention_gate_type!r}"
            )
        if not config.causal:
            errors.append("Laguna DFlash requires causal=true")
        if any(layer_type != "sliding_attention" for layer_type in config.layer_types):
            errors.append("Laguna DFlash currently requires sliding_attention for every draft layer")
        if any(window <= 0 for window in config.sliding_windows):
            errors.append("Laguna DFlash sliding windows must all be positive")
    elif config.decoder_arch == "qwen":
        if config.qkv_layout != "separate":
            errors.append("Qwen DFlash requires separate Q/K/V projection layout")
        if config.aux_hidden_norm_count != 0:
            errors.append("Qwen DFlash does not support auxiliary hidden norms")
    else:
        errors.append(f"unsupported DFlash decoder architecture {config.decoder_arch!r}")
    if config.architecture == "DFlash2DraftModel":
        if config.decoder_arch != "qwen":
            errors.append("DFlash2DraftModel requires the qwen decoder backbone")
        if config.conv_kernel_size <= 0:
            errors.append(f"DFlash2 conv_kernel_size must be positive, got {config.conv_kernel_size}")
        if config.conv_group_size <= 0 or config.hidden_size % config.conv_group_size != 0:
            errors.append(
                f"DFlash2 conv_group_size {config.conv_group_size} must divide hidden_size {config.hidden_size}"
            )
        if config.conv_projection_features <= 0:
            errors.append(f"DFlash2 conv projection width must be positive, got {config.conv_projection_features}")
        if config.selector_rank <= 0:
            errors.append(f"DFlash2 selector_rank must be positive, got {config.selector_rank}")
        if config.selector_top_k <= 0 or config.selector_top_k > config.vocab_size:
            errors.append(
                f"DFlash2 selector_top_k {config.selector_top_k} must be in (0, vocab_size {config.vocab_size}]"
            )
        if config.causal:
            errors.append("DFlash2DraftModel requires non-causal (is_causal=false) drafting")
        if any(layer_type != "sliding_attention" for layer_type in config.layer_types):
            errors.append("DFlash2DraftModel currently requires sliding_attention for every draft layer")
        if any(window <= 0 for window in config.sliding_windows):
            errors.append("DFlash2 sliding windows must all be positive")
    if target_config is not None:
        if config.num_target_layers != target_config.num_hidden_layers:
            errors.append(
                f"num_target_layers {config.num_target_layers} does not match target layers {target_config.num_hidden_layers}"
            )
        if config.target_hidden_size != target_config.hidden_size:
            errors.append(
                f"target_hidden_size {config.target_hidden_size} does not match target hidden_size {target_config.hidden_size}"
            )
        if config.vocab_size != target_config.vocab_size:
            errors.append(f"vocab_size {config.vocab_size} does not match target vocab_size {target_config.vocab_size}")
    return tuple(errors)


def _validate_requirements(
    tensors: Mapping[str, TensorInfo],
    requirements: Sequence[TensorRequirement],
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


def _shape_matches(actual: tuple[int, ...], expected: tuple[int | None, ...] | None) -> bool:
    if expected is None:
        return True
    if len(actual) != len(expected):
        return False
    return all(exp is None or int(act) == int(exp) for act, exp in zip(actual, expected, strict=True))


def _fmt_expected(values: Sequence[str]) -> str:
    return values[0] if len(values) == 1 else "{" + ", ".join(values) + "}"


def _fmt_shape(shape: Sequence[int | None] | None) -> str:
    if shape is None:
        return "(*)"
    return "(" + ", ".join("*" if value is None else str(value) for value in shape) + ")"


def _preview(values: Sequence[str], *, limit: int = 8) -> str:
    preview = ", ".join(values[:limit])
    more = "" if len(values) <= limit else f" (+{len(values) - limit} more)"
    return preview + more
