"""Qwen3.5 GGUF tensor-name mapping and layout validation."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from hipengine.loading.gguf import GGUFModelInfo, GGUFTensorInfo, MissingGGUFTensorError
from hipengine.quant.gguf import GGMLQuantizationType

FULL_ATTENTION = "full_attention"
LINEAR_ATTENTION = "linear_attention"

_DENSE_ROOT_SLOTS = {
    "token_embedding": "token_embd.weight",
    "output_norm": "output_norm.weight",
    # Dense Qwen3.5 GGUF omits a separate output tensor for this local target.
    # The lm-head is tied to token_embd.weight and materialization should alias it.
    "lm_head": "token_embd.weight",
}

_MOE_ROOT_SLOTS = {
    "token_embedding": "token_embd.weight",
    "output_norm": "output_norm.weight",
    # Qwen3.6 35B-A3B GGUF carries an untied output projection.
    "lm_head": "output.weight",
}

_DENSE_MLP_LAYER_SLOTS = {
    "ffn_gate": "ffn_gate.weight",
    "ffn_up": "ffn_up.weight",
    "ffn_down": "ffn_down.weight",
}

_MOE_LAYER_SLOTS = {
    "ffn_gate_inp": "ffn_gate_inp.weight",
    "ffn_gate_inp_shexp": "ffn_gate_inp_shexp.weight",
    "ffn_gate_exps": "ffn_gate_exps.weight",
    "ffn_up_exps": "ffn_up_exps.weight",
    "ffn_down_exps": "ffn_down_exps.weight",
    "ffn_gate_shexp": "ffn_gate_shexp.weight",
    "ffn_up_shexp": "ffn_up_shexp.weight",
    "ffn_down_shexp": "ffn_down_shexp.weight",
}

_COMMON_LAYER_SLOTS = {
    "attn_norm": "attn_norm.weight",
    "post_attention_norm": "post_attention_norm.weight",
}

_LINEAR_LAYER_SLOTS = {
    "attn_gate": "attn_gate.weight",
    "attn_qkv": "attn_qkv.weight",
    "ssm_a": "ssm_a",
    "ssm_alpha": "ssm_alpha.weight",
    "ssm_beta": "ssm_beta.weight",
    "ssm_conv1d": "ssm_conv1d.weight",
    "ssm_dt_bias": "ssm_dt.bias",
    "ssm_norm": "ssm_norm.weight",
    "ssm_out": "ssm_out.weight",
}

_FULL_LAYER_SLOTS = {
    "attn_q": "attn_q.weight",
    "attn_k": "attn_k.weight",
    "attn_v": "attn_v.weight",
    "attn_output": "attn_output.weight",
    "attn_q_norm": "attn_q_norm.weight",
    "attn_k_norm": "attn_k_norm.weight",
}

_MTP_NEXTN_REQUIRED_SLOTS = {
    "nextn.eh_proj": "nextn.eh_proj.weight",
    "nextn.enorm": "nextn.enorm.weight",
    "nextn.hnorm": "nextn.hnorm.weight",
}

_MTP_NEXTN_OPTIONAL_FALLBACK_SLOTS = {
    "nextn.embed_tokens": ("nextn.embed_tokens.weight", "token_embedding"),
    "nextn.shared_head_head": ("nextn.shared_head_head.weight", "lm_head"),
    "nextn.shared_head_norm": ("nextn.shared_head_norm.weight", "output_norm"),
}


@dataclass(frozen=True)
class Qwen35GGUFConfig:
    """Qwen3.5/Qwen3.6 GGUF dimensions decoded from metadata."""

    architecture: str
    block_count: int
    hidden_size: int
    vocab_size: int
    feed_forward_length: int
    context_length: int
    head_count: int
    head_count_kv: int
    key_length: int
    value_length: int
    full_attention_interval: int
    layer_types: tuple[str, ...]
    rms_norm_eps: float
    rope_dimension_count: int
    rope_dimension_sections: tuple[int, ...]
    rope_freq_base: float
    ssm_inner_size: int
    ssm_group_count: int
    ssm_state_size: int
    ssm_conv_kernel: int
    ssm_time_step_rank: int
    expert_count: int = 0
    expert_used_count: int = 0
    expert_feed_forward_length: int = 0
    expert_shared_feed_forward_length: int = 0
    declared_block_count: int = 0
    ignored_block_ids: tuple[int, ...] = ()

    @property
    def is_moe(self) -> bool:
        return self.architecture == "qwen35moe"


@dataclass(frozen=True)
class Qwen35GGUFMappingValidation:
    config: Qwen35GGUFConfig
    present: tuple[str, ...]
    missing: tuple[str, ...]
    unexpected: tuple[str, ...]
    shape_errors: tuple[str, ...]
    ignored: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return not self.missing and not self.unexpected and not self.shape_errors

    def raise_for_errors(self) -> None:
        if self.passed:
            return
        parts: list[str] = []
        if self.missing:
            preview = ", ".join(self.missing[:8])
            more = "" if len(self.missing) <= 8 else f" (+{len(self.missing) - 8} more)"
            parts.append(f"missing tensors: {preview}{more}")
        if self.unexpected:
            preview = ", ".join(self.unexpected[:8])
            more = "" if len(self.unexpected) <= 8 else f" (+{len(self.unexpected) - 8} more)"
            parts.append(f"unexpected tensors: {preview}{more}")
        if self.shape_errors:
            preview = "; ".join(self.shape_errors[:4])
            more = "" if len(self.shape_errors) <= 4 else f" (+{len(self.shape_errors) - 4} more)"
            parts.append(f"shape errors: {preview}{more}")
        raise MissingGGUFTensorError("; ".join(parts))


@dataclass(frozen=True)
class Qwen35GGUFMTPBlockInventory:
    """Read-only inventory for a trailing GGUF MTP/NextN block.

    This descriptor is intentionally metadata-only.  It makes the currently
    AR-ignored trailing block visible for MTP-GGUF planning/tests without
    materializing weights or changing AR execution.
    """

    layer_id: int
    tensor_names: tuple[str, ...]
    nextn_tensor_names: tuple[str, ...]
    required_tensor_names: tuple[str, ...]
    missing_required_tensor_names: tuple[str, ...]
    optional_tensor_names: tuple[str, ...]
    missing_optional_tensor_names: tuple[str, ...]
    optional_fallback_tensor_names: Mapping[str, str]
    unexpected_tensor_names: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.missing_required_tensor_names


@dataclass(frozen=True)
class Qwen35GGUFMTPBlockMap:
    """Effective tensor slots for one trailing GGUF MTP/NextN block."""

    layer_id: int
    tensors: Mapping[str, GGUFTensorInfo]
    fallback_slots: Mapping[str, str]

    def tensor(self, slot: str) -> GGUFTensorInfo:
        try:
            return self.tensors[slot]
        except KeyError as exc:
            raise MissingGGUFTensorError(
                f"MTP block {self.layer_id} has no GGUF tensor slot {slot!r}"
            ) from exc

    @property
    def tensor_names(self) -> tuple[str, ...]:
        names: list[str] = []
        seen: set[str] = set()
        for tensor in self.tensors.values():
            if tensor.name not in seen:
                seen.add(tensor.name)
                names.append(tensor.name)
        return tuple(names)


@dataclass(frozen=True)
class Qwen35GGUFMTPDraftTensorSlot:
    """Resolved GGUF tensor slot for one CPU-reference MTP draft layer input."""

    slot: str
    tensor_name: str
    shape: tuple[int, ...]
    ggml_type_name: str
    fallback_slot: str | None = None

    def as_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "slot": self.slot,
            "tensor_name": self.tensor_name,
            "shape": list(self.shape),
            "ggml_type_name": self.ggml_type_name,
        }
        if self.fallback_slot is not None:
            result["fallback_slot"] = self.fallback_slot
        return result


@dataclass(frozen=True)
class Qwen35GGUFMTPDraftTensorBinding:
    """CPU-reference argument binding for one resolved GGUF tensor slot."""

    argument: str
    slot: str
    tensor_name: str
    shape: tuple[int, ...]
    ggml_type_name: str
    fallback_slot: str | None = None
    qtype_argument: str | None = None

    def as_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "argument": self.argument,
            "slot": self.slot,
            "tensor_name": self.tensor_name,
            "shape": list(self.shape),
            "ggml_type_name": self.ggml_type_name,
        }
        if self.fallback_slot is not None:
            result["fallback_slot"] = self.fallback_slot
        if self.qtype_argument is not None:
            result["qtype_argument"] = self.qtype_argument
        return result


@dataclass(frozen=True)
class Qwen35GGUFMTPDraftCPUCallSpec:
    """Metadata-only call spec for a Qwen35 GGUF MTP CPU-reference oracle."""

    layer_id: int
    cpu_reference_kernel: tuple[str, str, str, str]
    tensor_arguments: Mapping[str, str]
    qtype_arguments: Mapping[str, GGMLQuantizationType]
    keyword_arguments: Mapping[str, object]
    tensor_bindings: tuple[Qwen35GGUFMTPDraftTensorBinding, ...]
    fallback_slots: Mapping[str, str]

    def as_dict(self) -> dict[str, object]:
        return {
            "layer_id": self.layer_id,
            "cpu_reference_kernel": list(self.cpu_reference_kernel),
            "tensor_arguments": dict(self.tensor_arguments),
            "qtype_arguments": {
                argument: qtype.name for argument, qtype in self.qtype_arguments.items()
            },
            "keyword_arguments": dict(self.keyword_arguments),
            "tensor_bindings": [binding.as_dict() for binding in self.tensor_bindings],
            "fallback_slots": dict(self.fallback_slots),
        }


@dataclass(frozen=True)
class Qwen35GGUFMTPDraftTensorPlan:
    """Ordered tensor plan for feeding one GGUF MTP draft layer CPU oracle."""

    layer_id: int
    hidden_size: int
    vocab_size: int
    num_heads: int
    num_kv_heads: int
    qk_head_dim: int
    value_head_dim: int
    attention_width: int
    experts_used: int
    expert_weights_scale: float
    rms_norm_eps: float
    rotary_dim: int
    rope_freq_base: float
    rope_dimension_sections: tuple[int, ...]
    attention_scale: float
    cpu_reference_kernel: tuple[str, str, str, str]
    slots: tuple[Qwen35GGUFMTPDraftTensorSlot, ...]
    fallback_slots: Mapping[str, str]

    def slot(self, slot: str) -> Qwen35GGUFMTPDraftTensorSlot:
        for item in self.slots:
            if item.slot == slot:
                return item
        raise MissingGGUFTensorError(
            f"MTP draft tensor plan for block {self.layer_id} has no slot {slot!r}"
        )

    @property
    def tensor_names(self) -> tuple[str, ...]:
        return tuple(item.tensor_name for item in self.slots)

    @property
    def tensor_bindings(self) -> tuple[Qwen35GGUFMTPDraftTensorBinding, ...]:
        bindings: list[Qwen35GGUFMTPDraftTensorBinding] = []
        for argument, slot, qtype_argument in _MTP_NEXTN_LAYER_CPU_ORACLE_ARGUMENT_SLOTS:
            tensor = self.slot(slot)
            bindings.append(
                Qwen35GGUFMTPDraftTensorBinding(
                    argument=argument,
                    slot=slot,
                    tensor_name=tensor.tensor_name,
                    shape=tensor.shape,
                    ggml_type_name=tensor.ggml_type_name,
                    fallback_slot=tensor.fallback_slot,
                    qtype_argument=qtype_argument,
                )
            )
        return tuple(bindings)

    @property
    def tensor_argument_map(self) -> Mapping[str, str]:
        return MappingProxyType(
            {binding.argument: binding.tensor_name for binding in self.tensor_bindings}
        )

    @property
    def qtype_argument_map(self) -> Mapping[str, str]:
        result: dict[str, str] = {}
        for binding in self.tensor_bindings:
            if binding.qtype_argument is None:
                continue
            previous = result.get(binding.qtype_argument)
            if previous is not None and previous != binding.ggml_type_name:
                raise MissingGGUFTensorError(
                    f"MTP draft tensor plan for block {self.layer_id} maps "
                    f"{binding.qtype_argument} to both {previous} and "
                    f"{binding.ggml_type_name}"
                )
            result[binding.qtype_argument] = binding.ggml_type_name
        return MappingProxyType(result)

    @property
    def qtype_enum_argument_map(self) -> Mapping[str, GGMLQuantizationType]:
        result: dict[str, GGMLQuantizationType] = {}
        for argument, ggml_type_name in self.qtype_argument_map.items():
            try:
                result[argument] = GGMLQuantizationType[ggml_type_name]
            except KeyError as exc:
                raise MissingGGUFTensorError(
                    f"MTP draft tensor plan for block {self.layer_id} maps "
                    f"{argument} to unsupported GGML type {ggml_type_name!r}"
                ) from exc
        return MappingProxyType(result)

    @property
    def kernel_kwargs(self) -> Mapping[str, object]:
        """Scalar kwargs for ``qwen35_gguf_mtp_nextn_layer_logits``."""

        return MappingProxyType(
            {
                "num_heads": self.num_heads,
                "num_kv_heads": self.num_kv_heads,
                "experts_used": self.experts_used,
                "rotary_dim": self.rotary_dim,
                "scale": self.attention_scale,
                "expert_weights_scale": self.expert_weights_scale,
                "eps": self.rms_norm_eps,
            }
        )

    @property
    def cpu_reference_call_spec(self) -> Qwen35GGUFMTPDraftCPUCallSpec:
        """Return the metadata needed to invoke the CPU-reference NextN oracle."""

        return Qwen35GGUFMTPDraftCPUCallSpec(
            layer_id=self.layer_id,
            cpu_reference_kernel=self.cpu_reference_kernel,
            tensor_arguments=self.tensor_argument_map,
            qtype_arguments=self.qtype_enum_argument_map,
            keyword_arguments=self.kernel_kwargs,
            tensor_bindings=self.tensor_bindings,
            fallback_slots=self.fallback_slots,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "layer_id": self.layer_id,
            "hidden_size": self.hidden_size,
            "vocab_size": self.vocab_size,
            "num_heads": self.num_heads,
            "num_kv_heads": self.num_kv_heads,
            "qk_head_dim": self.qk_head_dim,
            "value_head_dim": self.value_head_dim,
            "attention_width": self.attention_width,
            "experts_used": self.experts_used,
            "expert_weights_scale": self.expert_weights_scale,
            "rms_norm_eps": self.rms_norm_eps,
            "rotary_dim": self.rotary_dim,
            "rope_freq_base": self.rope_freq_base,
            "rope_dimension_sections": list(self.rope_dimension_sections),
            "attention_scale": self.attention_scale,
            "kernel_kwargs": dict(self.kernel_kwargs),
            "cpu_reference_call_spec": self.cpu_reference_call_spec.as_dict(),
            "tensor_argument_map": dict(self.tensor_argument_map),
            "qtype_argument_map": dict(self.qtype_argument_map),
            "qtype_enum_argument_map": {
                argument: qtype.name
                for argument, qtype in self.qtype_enum_argument_map.items()
            },
            "cpu_reference_kernel": list(self.cpu_reference_kernel),
            "slots": [slot.as_dict() for slot in self.slots],
            "tensor_bindings": [binding.as_dict() for binding in self.tensor_bindings],
            "fallback_slots": dict(self.fallback_slots),
        }


@dataclass(frozen=True)
class Qwen35GGUFMTPDraftSpec:
    """Shape contract for draft-only GGUF NextN execution."""

    layer_id: int
    hidden_size: int
    vocab_size: int
    eh_proj_shape: tuple[int, ...]
    tensor_shapes: Mapping[str, tuple[int, ...]]
    embed_tokens_tensor: str
    shared_head_tensor: str
    shared_head_norm_tensor: str
    fallback_slots: Mapping[str, str]

    def as_dict(self) -> dict[str, object]:
        return {
            "layer_id": self.layer_id,
            "hidden_size": self.hidden_size,
            "vocab_size": self.vocab_size,
            "eh_proj_shape": list(self.eh_proj_shape),
            "tensor_shapes": {
                slot: list(shape) for slot, shape in self.tensor_shapes.items()
            },
            "embed_tokens_tensor": self.embed_tokens_tensor,
            "shared_head_tensor": self.shared_head_tensor,
            "shared_head_norm_tensor": self.shared_head_norm_tensor,
            "fallback_slots": dict(self.fallback_slots),
        }


@dataclass(frozen=True)
class Qwen35GGUFLayerMap:
    """Canonical tensor slots for one Qwen3.5 GGUF layer."""

    layer_id: int
    layer_type: str
    tensors: Mapping[str, GGUFTensorInfo]

    def tensor(self, slot: str) -> GGUFTensorInfo:
        try:
            return self.tensors[slot]
        except KeyError as exc:
            raise MissingGGUFTensorError(
                f"layer {self.layer_id} has no GGUF tensor slot {slot!r}"
            ) from exc

    @property
    def tensor_names(self) -> tuple[str, ...]:
        return tuple(tensor.name for tensor in self.tensors.values())


@dataclass(frozen=True)
class Qwen35GGUFModelMap:
    """Canonical root/layer tensor map for a Qwen3.5 GGUF file."""

    config: Qwen35GGUFConfig
    root_tensors: Mapping[str, GGUFTensorInfo]
    layers: tuple[Qwen35GGUFLayerMap, ...]
    validation: Qwen35GGUFMappingValidation
    mtp_blocks: tuple[Qwen35GGUFMTPBlockInventory, ...] = ()

    def root(self, slot: str) -> GGUFTensorInfo:
        try:
            return self.root_tensors[slot]
        except KeyError as exc:
            raise MissingGGUFTensorError(f"model has no GGUF root tensor slot {slot!r}") from exc

    def layer(self, layer_id: int) -> Qwen35GGUFLayerMap:
        return self.layers[layer_id]

    @property
    def tensor_names(self) -> tuple[str, ...]:
        names: list[str] = []
        seen: set[str] = set()
        for tensor in self.root_tensors.values():
            if tensor.name not in seen:
                seen.add(tensor.name)
                names.append(tensor.name)
        for layer in self.layers:
            for name in layer.tensor_names:
                if name not in seen:
                    seen.add(name)
                    names.append(name)
        return tuple(names)


def qwen35_gguf_config_from_metadata(info: GGUFModelInfo) -> Qwen35GGUFConfig:
    metadata = info.metadata
    architecture = str(metadata.get("general.architecture", ""))
    if architecture not in {"qwen35", "qwen35moe"}:
        raise ValueError(f"expected GGUF architecture 'qwen35' or 'qwen35moe', got {architecture!r}")
    prefix = architecture
    declared_block_count = _int_metadata(metadata, f"{prefix}.block_count")
    full_interval = int(metadata.get(f"{prefix}.full_attention_interval", 0) or 0)
    block_count, ignored_block_ids = _ar_block_count_from_tensor_inventory(
        info,
        declared_block_count=declared_block_count,
    )
    layer_types = tuple(
        FULL_ATTENTION if full_interval and (layer_id + 1) % full_interval == 0 else LINEAR_ATTENTION
        for layer_id in range(block_count)
    )
    token_embedding = info.tensor("token_embd.weight")
    expert_count = int(metadata.get(f"{prefix}.expert_count", 0) or 0)
    expert_used_count = int(metadata.get(f"{prefix}.expert_used_count", 0) or 0)
    expert_ffn = int(metadata.get(f"{prefix}.expert_feed_forward_length", 0) or 0)
    shared_ffn = int(metadata.get(f"{prefix}.expert_shared_feed_forward_length", 0) or 0)
    feed_forward_length = (
        expert_ffn if architecture == "qwen35moe" else _int_metadata(metadata, f"{prefix}.feed_forward_length")
    )
    return Qwen35GGUFConfig(
        architecture=architecture,
        block_count=block_count,
        hidden_size=_int_metadata(metadata, f"{prefix}.embedding_length"),
        vocab_size=int(token_embedding.shape[0]),
        feed_forward_length=feed_forward_length,
        context_length=_int_metadata(metadata, f"{prefix}.context_length"),
        head_count=_int_metadata(metadata, f"{prefix}.attention.head_count"),
        head_count_kv=_int_metadata(metadata, f"{prefix}.attention.head_count_kv"),
        key_length=_int_metadata(metadata, f"{prefix}.attention.key_length"),
        value_length=_int_metadata(metadata, f"{prefix}.attention.value_length"),
        full_attention_interval=full_interval,
        layer_types=layer_types,
        rms_norm_eps=float(metadata.get(f"{prefix}.attention.layer_norm_rms_epsilon", 1.0e-6)),
        rope_dimension_count=_int_metadata(metadata, f"{prefix}.rope.dimension_count"),
        rope_dimension_sections=tuple(int(item) for item in metadata.get(f"{prefix}.rope.dimension_sections", ())),
        rope_freq_base=float(metadata.get(f"{prefix}.rope.freq_base", 10000000.0)),
        ssm_inner_size=_int_metadata(metadata, f"{prefix}.ssm.inner_size"),
        ssm_group_count=_int_metadata(metadata, f"{prefix}.ssm.group_count"),
        ssm_state_size=_int_metadata(metadata, f"{prefix}.ssm.state_size"),
        ssm_conv_kernel=_int_metadata(metadata, f"{prefix}.ssm.conv_kernel"),
        ssm_time_step_rank=_int_metadata(metadata, f"{prefix}.ssm.time_step_rank"),
        expert_count=expert_count,
        expert_used_count=expert_used_count,
        expert_feed_forward_length=expert_ffn,
        expert_shared_feed_forward_length=shared_ffn,
        declared_block_count=declared_block_count,
        ignored_block_ids=ignored_block_ids,
    )


def required_qwen35_gguf_tensor_names(config: Qwen35GGUFConfig) -> tuple[str, ...]:
    names = list(_root_slots_for_config(config).values())
    for layer_id, layer_type in enumerate(config.layer_types):
        names.extend(_layer_required_tensor_names(config, layer_id, layer_type))
    return tuple(dict.fromkeys(names))


def validate_qwen35_gguf_tensor_map(info: GGUFModelInfo) -> Qwen35GGUFMappingValidation:
    config = qwen35_gguf_config_from_metadata(info)
    actual = {tensor.name: tensor for tensor in info.tensors}
    required = set(required_qwen35_gguf_tensor_names(config))
    actual_names = set(actual)
    ignored = set(_ignored_ar_tensor_names(config, actual_names))
    missing = tuple(sorted(required - actual_names))
    unexpected = tuple(sorted(actual_names - required - ignored))
    shape_errors = tuple(_shape_errors(config, actual))
    present = tuple(sorted(required & actual_names))
    return Qwen35GGUFMappingValidation(
        config=config,
        present=present,
        missing=missing,
        unexpected=unexpected,
        shape_errors=shape_errors,
        ignored=tuple(sorted(ignored)),
    )


def build_qwen35_gguf_tensor_map(info: GGUFModelInfo, *, strict: bool = True) -> Qwen35GGUFModelMap:
    validation = validate_qwen35_gguf_tensor_map(info)
    if strict:
        validation.raise_for_errors()
    actual = {tensor.name: tensor for tensor in info.tensors}
    root_tensors = MappingProxyType(
        {slot: actual[name] for slot, name in _root_slots_for_config(validation.config).items() if name in actual}
    )
    layers = tuple(
        _build_layer_map(validation.config, actual, layer_id)
        for layer_id in range(validation.config.block_count)
    )
    return Qwen35GGUFModelMap(
        config=validation.config,
        root_tensors=root_tensors,
        layers=layers,
        validation=validation,
        mtp_blocks=qwen35_gguf_mtp_block_inventories(info),
    )


def qwen35_gguf_mtp_block_inventories(info: GGUFModelInfo) -> tuple[Qwen35GGUFMTPBlockInventory, ...]:
    """Return metadata-only inventories for trailing GGUF MTP/NextN blocks."""

    config = qwen35_gguf_config_from_metadata(info)
    actual_names = {tensor.name for tensor in info.tensors}
    root_slots = _root_slots_for_config(config)
    inventories: list[Qwen35GGUFMTPBlockInventory] = []
    for layer_id in config.ignored_block_ids:
        prefix = f"blk.{layer_id}."
        block_names = tuple(sorted(name for name in actual_names if name.startswith(prefix)))
        required = set(_mtp_required_tensor_names(config, layer_id))
        optional = {
            f"{prefix}{suffix}": (slot, root_slots[fallback_slot])
            for slot, (suffix, fallback_slot) in _MTP_NEXTN_OPTIONAL_FALLBACK_SLOTS.items()
        }
        optional_names = set(optional)
        optional_present = tuple(sorted(optional_names & actual_names))
        optional_missing = tuple(sorted(optional_names - actual_names))
        optional_fallbacks = MappingProxyType(
            {
                slot: fallback_name
                for full_name in optional_missing
                for slot, fallback_name in (optional[full_name],)
            }
        )
        inventories.append(
            Qwen35GGUFMTPBlockInventory(
                layer_id=layer_id,
                tensor_names=block_names,
                nextn_tensor_names=tuple(
                    sorted(name for name in block_names if name.startswith(f"{prefix}nextn."))
                ),
                required_tensor_names=tuple(sorted(required)),
                missing_required_tensor_names=tuple(sorted(required - actual_names)),
                optional_tensor_names=optional_present,
                missing_optional_tensor_names=optional_missing,
                optional_fallback_tensor_names=optional_fallbacks,
                unexpected_tensor_names=tuple(sorted(set(block_names) - required - optional_names)),
            )
        )
    return tuple(inventories)


def build_qwen35_gguf_mtp_block_maps(
    info: GGUFModelInfo,
    *,
    strict: bool = True,
) -> tuple[Qwen35GGUFMTPBlockMap, ...]:
    """Return effective tensor maps for trailing MTP blocks.

    Optional NextN tensors are resolved to target-model fallback tensors when
    they are absent from the GGUF MTP block.
    """

    inventories = (
        validate_qwen35_gguf_mtp_blocks(info)
        if strict
        else qwen35_gguf_mtp_block_inventories(info)
    )
    config = qwen35_gguf_config_from_metadata(info)
    actual = {tensor.name: tensor for tensor in info.tensors}
    root_slots = _root_slots_for_config(config)
    maps: list[Qwen35GGUFMTPBlockMap] = []
    for block in inventories:
        prefix = f"blk.{block.layer_id}."
        tensors: dict[str, GGUFTensorInfo] = {}
        for slot, suffix in _layer_slot_suffixes(config, FULL_ATTENTION).items():
            full_name = f"{prefix}{suffix}"
            if full_name in actual:
                tensors[slot] = actual[full_name]
        for slot, suffix in _MTP_NEXTN_REQUIRED_SLOTS.items():
            full_name = f"{prefix}{suffix}"
            if full_name in actual:
                tensors[slot] = actual[full_name]
        fallback_slots: dict[str, str] = {}
        for slot, (suffix, fallback_slot) in _MTP_NEXTN_OPTIONAL_FALLBACK_SLOTS.items():
            full_name = f"{prefix}{suffix}"
            if full_name in actual:
                tensors[slot] = actual[full_name]
            else:
                tensors[slot] = actual[root_slots[fallback_slot]]
                fallback_slots[slot] = fallback_slot
        maps.append(
            Qwen35GGUFMTPBlockMap(
                layer_id=block.layer_id,
                tensors=MappingProxyType(tensors),
                fallback_slots=MappingProxyType(fallback_slots),
            )
        )
    return tuple(maps)


def build_qwen35_gguf_mtp_draft_tensor_plans(
    info: GGUFModelInfo,
    *,
    strict: bool = True,
) -> tuple[Qwen35GGUFMTPDraftTensorPlan, ...]:
    """Return ordered GGUF tensor plans for CPU-reference MTP draft oracles."""

    config = qwen35_gguf_config_from_metadata(info)
    specs = build_qwen35_gguf_mtp_draft_specs(info, strict=strict)
    block_maps = build_qwen35_gguf_mtp_block_maps(info, strict=strict)
    spec_by_layer = {spec.layer_id: spec for spec in specs}
    plans: list[Qwen35GGUFMTPDraftTensorPlan] = []
    for block_map in block_maps:
        spec = spec_by_layer[block_map.layer_id]
        slots: list[Qwen35GGUFMTPDraftTensorSlot] = []
        for slot in _MTP_NEXTN_LAYER_CPU_ORACLE_TENSOR_SLOTS:
            tensor = block_map.tensor(slot)
            slots.append(
                Qwen35GGUFMTPDraftTensorSlot(
                    slot=slot,
                    tensor_name=tensor.name,
                    shape=tensor.shape,
                    ggml_type_name=tensor.ggml_type_name,
                    fallback_slot=block_map.fallback_slots.get(slot),
                )
            )
        plans.append(
            Qwen35GGUFMTPDraftTensorPlan(
                layer_id=block_map.layer_id,
                hidden_size=spec.hidden_size,
                vocab_size=spec.vocab_size,
                num_heads=config.head_count,
                num_kv_heads=config.head_count_kv,
                qk_head_dim=config.key_length,
                value_head_dim=config.value_length,
                attention_width=_attention_output_width(config),
                experts_used=config.expert_used_count,
                expert_weights_scale=_expert_weights_scale(config),
                rms_norm_eps=config.rms_norm_eps,
                rotary_dim=config.rope_dimension_count,
                rope_freq_base=config.rope_freq_base,
                rope_dimension_sections=config.rope_dimension_sections,
                attention_scale=_effective_attention_scale(config),
                cpu_reference_kernel=_MTP_NEXTN_LAYER_CPU_REFERENCE_KERNEL,
                slots=tuple(slots),
                fallback_slots=block_map.fallback_slots,
            )
        )
    return tuple(plans)


def build_qwen35_gguf_mtp_draft_specs(
    info: GGUFModelInfo,
    *,
    strict: bool = True,
) -> tuple[Qwen35GGUFMTPDraftSpec, ...]:
    """Return NextN draft shape specs for trailing MTP blocks."""

    config = qwen35_gguf_config_from_metadata(info)
    block_maps = build_qwen35_gguf_mtp_block_maps(info, strict=strict)
    specs: list[Qwen35GGUFMTPDraftSpec] = []
    errors: list[str] = []
    hidden = config.hidden_size
    vocab = config.vocab_size
    for block_map in block_maps:
        errors.extend(_mtp_draft_shape_errors(block_map, config=config))
        specs.append(
            Qwen35GGUFMTPDraftSpec(
                layer_id=block_map.layer_id,
                hidden_size=hidden,
                vocab_size=vocab,
                eh_proj_shape=block_map.tensor("nextn.eh_proj").shape,
                tensor_shapes=MappingProxyType(
                    {
                        slot: block_map.tensor(slot).shape
                        for slot in _mtp_draft_expected_shapes(config)
                    }
                ),
                embed_tokens_tensor=block_map.tensor("nextn.embed_tokens").name,
                shared_head_tensor=block_map.tensor("nextn.shared_head_head").name,
                shared_head_norm_tensor=block_map.tensor("nextn.shared_head_norm").name,
                fallback_slots=block_map.fallback_slots,
            )
        )
    if errors:
        preview = "; ".join(errors[:6])
        more = "" if len(errors) <= 6 else f" (+{len(errors) - 6} more)"
        raise MissingGGUFTensorError(f"MTP draft shape errors: {preview}{more}")
    return tuple(specs)


def validate_qwen35_gguf_mtp_blocks(info: GGUFModelInfo) -> tuple[Qwen35GGUFMTPBlockInventory, ...]:
    """Validate required trailing GGUF MTP/NextN tensor metadata.

    AR mapping intentionally excludes ignored MTP blocks.  This separate gate is
    for the GGUF-MTP lane: optional tensors may be absent when a target-weight
    fallback is defined, but required tensors and unexpected trailing-block
    tensors are errors.
    """

    inventories = qwen35_gguf_mtp_block_inventories(info)
    errors: list[str] = []
    for block in inventories:
        if block.missing_required_tensor_names:
            preview = ", ".join(block.missing_required_tensor_names[:8])
            more = (
                ""
                if len(block.missing_required_tensor_names) <= 8
                else f" (+{len(block.missing_required_tensor_names) - 8} more)"
            )
            errors.append(f"MTP block {block.layer_id} missing required tensors: {preview}{more}")
        if block.unexpected_tensor_names:
            preview = ", ".join(block.unexpected_tensor_names[:8])
            more = (
                ""
                if len(block.unexpected_tensor_names) <= 8
                else f" (+{len(block.unexpected_tensor_names) - 8} more)"
            )
            errors.append(f"MTP block {block.layer_id} unexpected tensors: {preview}{more}")
    if errors:
        raise MissingGGUFTensorError("; ".join(errors))
    return inventories


_MTP_NEXTN_LAYER_CPU_REFERENCE_KERNEL = (
    "cpu_reference",
    "mtp_nextn_layer",
    "gguf_moe",
    "qwen35_dense_logits",
)

_MTP_NEXTN_LAYER_CPU_ORACLE_ARGUMENT_SLOTS = (
    ("token_embedding", "nextn.embed_tokens", None),
    ("eh_proj_weight", "nextn.eh_proj", None),
    ("hnorm_weight", "nextn.hnorm", None),
    ("enorm_weight", "nextn.enorm", None),
    ("attn_norm_weight", "attn_norm", None),
    ("wq_weight", "attn_q", None),
    ("wk_weight", "attn_k", None),
    ("wv_weight", "attn_v", None),
    ("wo_weight", "attn_output", None),
    ("q_norm_weight", "attn_q_norm", None),
    ("k_norm_weight", "attn_k_norm", None),
    ("attn_post_norm_weight", "post_attention_norm", None),
    ("router_weight", "ffn_gate_inp", None),
    ("gate_qweight", "ffn_gate_exps", "gate_qtype"),
    ("up_qweight", "ffn_up_exps", "up_qtype"),
    ("down_qweight", "ffn_down_exps", "down_qtype"),
    ("shared_gate_logit_weight", "ffn_gate_inp_shexp", None),
    ("shared_gate_qweight", "ffn_gate_shexp", "shared_qtype"),
    ("shared_up_qweight", "ffn_up_shexp", "shared_qtype"),
    ("shared_down_qweight", "ffn_down_shexp", "shared_qtype"),
    ("shared_head_norm_weight", "nextn.shared_head_norm", None),
    ("shared_head_weight", "nextn.shared_head_head", None),
)

_MTP_NEXTN_LAYER_CPU_ORACLE_TENSOR_SLOTS = (
    "nextn.embed_tokens",
    "nextn.eh_proj",
    "nextn.hnorm",
    "nextn.enorm",
    "attn_norm",
    "attn_q",
    "attn_k",
    "attn_v",
    "attn_output",
    "attn_q_norm",
    "attn_k_norm",
    "post_attention_norm",
    "ffn_gate_inp",
    "ffn_gate_exps",
    "ffn_up_exps",
    "ffn_down_exps",
    "ffn_gate_inp_shexp",
    "ffn_gate_shexp",
    "ffn_up_shexp",
    "ffn_down_shexp",
    "nextn.shared_head_norm",
    "nextn.shared_head_head",
)


def _mtp_draft_shape_errors(
    block_map: Qwen35GGUFMTPBlockMap,
    *,
    config: Qwen35GGUFConfig,
) -> list[str]:
    errors: list[str] = []
    for slot, shape in _mtp_draft_expected_shapes(config).items():
        actual = block_map.tensor(slot).shape
        if actual != shape:
            errors.append(
                f"MTP block {block_map.layer_id} slot {slot} "
                f"has shape {actual}, expected {shape}"
            )
    return errors


def _mtp_draft_expected_shapes(config: Qwen35GGUFConfig) -> dict[str, tuple[int, ...]]:
    hidden = config.hidden_size
    vocab = config.vocab_size
    attention_width = _attention_output_width(config)
    shapes: dict[str, tuple[int, ...]] = {
        "attn_norm": (hidden,),
        "post_attention_norm": (hidden,),
        "attn_q": (2 * config.head_count * config.key_length, hidden),
        "attn_k": (config.head_count_kv * config.key_length, hidden),
        "attn_v": (config.head_count_kv * config.value_length, hidden),
        "attn_output": (hidden, attention_width),
        "attn_q_norm": (config.key_length,),
        "attn_k_norm": (config.key_length,),
        "nextn.eh_proj": (hidden, hidden * 2),
        "nextn.enorm": (hidden,),
        "nextn.hnorm": (hidden,),
        "nextn.shared_head_norm": (hidden,),
        "nextn.embed_tokens": (vocab, hidden),
        "nextn.shared_head_head": (vocab, hidden),
    }
    if config.is_moe:
        shapes.update(
            {
                "ffn_gate_inp": (config.expert_count, hidden),
                "ffn_gate_inp_shexp": (hidden,),
                "ffn_gate_exps": (
                    config.expert_count,
                    config.expert_feed_forward_length,
                    hidden,
                ),
                "ffn_up_exps": (
                    config.expert_count,
                    config.expert_feed_forward_length,
                    hidden,
                ),
                "ffn_down_exps": (
                    config.expert_count,
                    hidden,
                    config.expert_feed_forward_length,
                ),
                "ffn_gate_shexp": (
                    config.expert_shared_feed_forward_length,
                    hidden,
                ),
                "ffn_up_shexp": (
                    config.expert_shared_feed_forward_length,
                    hidden,
                ),
                "ffn_down_shexp": (
                    hidden,
                    config.expert_shared_feed_forward_length,
                ),
            }
        )
    else:
        shapes.update(
            {
                "ffn_gate": (config.feed_forward_length, hidden),
                "ffn_up": (config.feed_forward_length, hidden),
                "ffn_down": (hidden, config.feed_forward_length),
            }
        )
    return shapes


def _build_layer_map(
    config: Qwen35GGUFConfig,
    actual: Mapping[str, GGUFTensorInfo],
    layer_id: int,
) -> Qwen35GGUFLayerMap:
    layer_type = config.layer_types[layer_id]
    slot_suffixes = _layer_slot_suffixes(config, layer_type)
    tensors = {
        slot: actual[f"blk.{layer_id}.{suffix}"]
        for slot, suffix in slot_suffixes.items()
        if f"blk.{layer_id}.{suffix}" in actual
    }
    return Qwen35GGUFLayerMap(
        layer_id=layer_id,
        layer_type=layer_type,
        tensors=MappingProxyType(tensors),
    )


def _root_slots_for_config(config: Qwen35GGUFConfig) -> Mapping[str, str]:
    return _MOE_ROOT_SLOTS if config.is_moe else _DENSE_ROOT_SLOTS


def _layer_slot_suffixes(config: Qwen35GGUFConfig, layer_type: str) -> dict[str, str]:
    slot_suffixes = dict(_COMMON_LAYER_SLOTS)
    slot_suffixes.update(_MOE_LAYER_SLOTS if config.is_moe else _DENSE_MLP_LAYER_SLOTS)
    if layer_type == FULL_ATTENTION:
        slot_suffixes.update(_FULL_LAYER_SLOTS)
    elif layer_type == LINEAR_ATTENTION:
        slot_suffixes.update(_LINEAR_LAYER_SLOTS)
    else:
        raise ValueError(f"unknown Qwen3.5 GGUF layer type {layer_type!r}")
    return slot_suffixes


def _layer_required_tensor_names(
    config: Qwen35GGUFConfig,
    layer_id: int,
    layer_type: str,
) -> tuple[str, ...]:
    return tuple(f"blk.{layer_id}.{suffix}" for suffix in _layer_slot_suffixes(config, layer_type).values())


def _ar_block_count_from_tensor_inventory(
    info: GGUFModelInfo,
    *,
    declared_block_count: int,
) -> tuple[int, tuple[int, ...]]:
    """Return executable AR layer count, excluding trailing MTP/nextn blocks.

    Recent Qwen3.6 GGUF exports may include the MTP predictor as an extra
    trailing ``blk.N`` with ``nextn`` tensors while keeping that block in the
    metadata ``block_count``.  The AR runtime should not materialize or execute
    that block, but all preceding model layers must remain strictly validated.
    """

    tensor_names = {tensor.name for tensor in info.tensors}
    block_count = int(declared_block_count)
    ignored: list[int] = []
    while block_count > 0 and _has_nextn_tensors(tensor_names, layer_id=block_count - 1):
        block_count -= 1
        ignored.append(block_count)
    ignored.reverse()
    return block_count, tuple(ignored)


def _ignored_ar_tensor_names(config: Qwen35GGUFConfig, actual_names: set[str]) -> tuple[str, ...]:
    ignored: list[str] = []
    for layer_id in config.ignored_block_ids:
        prefix = f"blk.{layer_id}."
        block_names = sorted(name for name in actual_names if name.startswith(prefix))
        if any(name.startswith(f"{prefix}nextn.") for name in block_names):
            ignored.extend(block_names)
    return tuple(ignored)


def _has_nextn_tensors(tensor_names: set[str], *, layer_id: int) -> bool:
    prefix = f"blk.{layer_id}.nextn."
    return any(name.startswith(prefix) for name in tensor_names)


def _shape_errors(config: Qwen35GGUFConfig, actual: Mapping[str, GGUFTensorInfo]) -> list[str]:
    expected = {
        "output_norm.weight": (config.hidden_size,),
        "token_embd.weight": (config.vocab_size, config.hidden_size),
    }
    if config.is_moe:
        expected["output.weight"] = (config.vocab_size, config.hidden_size)
    for layer_id, layer_type in enumerate(config.layer_types):
        prefix = f"blk.{layer_id}"
        expected.update(
            {
                f"{prefix}.attn_norm.weight": (config.hidden_size,),
                f"{prefix}.post_attention_norm.weight": (config.hidden_size,),
            }
        )
        if config.is_moe:
            expected.update(
                {
                    f"{prefix}.ffn_gate_inp.weight": (config.expert_count, config.hidden_size),
                    f"{prefix}.ffn_gate_inp_shexp.weight": (config.hidden_size,),
                    f"{prefix}.ffn_gate_exps.weight": (
                        config.expert_count,
                        config.expert_feed_forward_length,
                        config.hidden_size,
                    ),
                    f"{prefix}.ffn_up_exps.weight": (
                        config.expert_count,
                        config.expert_feed_forward_length,
                        config.hidden_size,
                    ),
                    f"{prefix}.ffn_down_exps.weight": (
                        config.expert_count,
                        config.hidden_size,
                        config.expert_feed_forward_length,
                    ),
                    f"{prefix}.ffn_gate_shexp.weight": (
                        config.expert_shared_feed_forward_length,
                        config.hidden_size,
                    ),
                    f"{prefix}.ffn_up_shexp.weight": (
                        config.expert_shared_feed_forward_length,
                        config.hidden_size,
                    ),
                    f"{prefix}.ffn_down_shexp.weight": (
                        config.hidden_size,
                        config.expert_shared_feed_forward_length,
                    ),
                }
            )
        else:
            expected.update(
                {
                    f"{prefix}.ffn_gate.weight": (config.feed_forward_length, config.hidden_size),
                    f"{prefix}.ffn_up.weight": (config.feed_forward_length, config.hidden_size),
                    f"{prefix}.ffn_down.weight": (config.hidden_size, config.feed_forward_length),
                }
            )
        if layer_type == LINEAR_ATTENTION:
            expected.update(
                {
                    f"{prefix}.attn_gate.weight": (config.ssm_inner_size, config.hidden_size),
                    f"{prefix}.attn_qkv.weight": (_linear_qkv_width(config), config.hidden_size),
                    f"{prefix}.ssm_a": (config.ssm_time_step_rank,),
                    f"{prefix}.ssm_alpha.weight": (config.ssm_time_step_rank, config.hidden_size),
                    f"{prefix}.ssm_beta.weight": (config.ssm_time_step_rank, config.hidden_size),
                    f"{prefix}.ssm_conv1d.weight": (_linear_qkv_width(config), config.ssm_conv_kernel),
                    f"{prefix}.ssm_dt.bias": (config.ssm_time_step_rank,),
                    f"{prefix}.ssm_norm.weight": (config.ssm_state_size,),
                    f"{prefix}.ssm_out.weight": (config.hidden_size, config.ssm_inner_size),
                }
            )
        else:
            expected.update(
                {
                    f"{prefix}.attn_q.weight": (2 * config.head_count * config.key_length, config.hidden_size),
                    f"{prefix}.attn_k.weight": (config.head_count_kv * config.key_length, config.hidden_size),
                    f"{prefix}.attn_v.weight": (config.head_count_kv * config.value_length, config.hidden_size),
                    f"{prefix}.attn_output.weight": (config.hidden_size, _attention_output_width(config)),
                    f"{prefix}.attn_q_norm.weight": (config.key_length,),
                    f"{prefix}.attn_k_norm.weight": (config.key_length,),
                }
            )
    errors: list[str] = []
    for name, shape in expected.items():
        tensor = actual.get(name)
        if tensor is not None and tensor.shape != shape:
            errors.append(f"{name}: expected shape {shape}, got {tensor.shape}")
    return errors


def _mtp_required_tensor_names(config: Qwen35GGUFConfig, layer_id: int) -> tuple[str, ...]:
    names = list(_layer_required_tensor_names(config, layer_id, FULL_ATTENTION))
    names.extend(
        f"blk.{layer_id}.{suffix}" for suffix in _MTP_NEXTN_REQUIRED_SLOTS.values()
    )
    return tuple(dict.fromkeys(names))


def _attention_output_width(config: Qwen35GGUFConfig) -> int:
    return config.head_count * config.value_length


def _effective_attention_scale(config: Qwen35GGUFConfig) -> float:
    return config.key_length ** -0.5


def _expert_weights_scale(config: Qwen35GGUFConfig) -> float:
    del config
    # Qwen35MoE does not currently load LLM_KV_EXPERT_WEIGHTS_SCALE in llama.cpp;
    # llama-hparams.h therefore leaves the scale at 0.0f. build_moe_ffn treats
    # both 0.0 and 1.0 as no-op scale values.
    return 0.0


def _linear_qkv_width(config: Qwen35GGUFConfig) -> int:
    return 2 * config.ssm_group_count * config.ssm_state_size + config.ssm_inner_size


def _int_metadata(metadata: Mapping[str, Any], key: str) -> int:
    try:
        return int(metadata[key])
    except KeyError as exc:
        raise KeyError(f"missing required Qwen3.5 GGUF metadata key {key!r}") from exc


__all__ = [
    "FULL_ATTENTION",
    "LINEAR_ATTENTION",
    "Qwen35GGUFConfig",
    "Qwen35GGUFLayerMap",
    "Qwen35GGUFMTPBlockInventory",
    "Qwen35GGUFMTPBlockMap",
    "Qwen35GGUFMTPDraftCPUCallSpec",
    "Qwen35GGUFMTPDraftTensorBinding",
    "Qwen35GGUFMTPDraftTensorPlan",
    "Qwen35GGUFMTPDraftTensorSlot",
    "Qwen35GGUFMappingValidation",
    "Qwen35GGUFModelMap",
    "build_qwen35_gguf_mtp_block_maps",
    "build_qwen35_gguf_mtp_draft_specs",
    "build_qwen35_gguf_mtp_draft_tensor_plans",
    "build_qwen35_gguf_tensor_map",
    "qwen35_gguf_config_from_metadata",
    "qwen35_gguf_mtp_block_inventories",
    "required_qwen35_gguf_tensor_names",
    "validate_qwen35_gguf_mtp_blocks",
    "validate_qwen35_gguf_tensor_map",
]
