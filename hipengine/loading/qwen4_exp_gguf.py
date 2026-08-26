"""Strict Qwen3.8-Flash-Next qwen4exp GGUF metadata contract."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import isclose
from pathlib import Path
from typing import Any, Mapping, Sequence

from hipengine.loading.gguf import GGUFModelInfo, GGUFTensorInfo

GDN = "gdn"
QSA = "qsa"

_PLE_HEAD_VOCAB_SIZES = (
    20_000_003,
    20_000_023,
    20_000_033,
    20_000_047,
    20_000_059,
    20_000_063,
    20_000_069,
    20_000_077,
    20_000_081,
    20_000_093,
    20_000_107,
    20_000_147,
    20_000_153,
    20_000_159,
    20_000_161,
    20_000_171,
)
_PLE_HEAD_OFFSETS = tuple(
    sum(_PLE_HEAD_VOCAB_SIZES[:index]) for index in range(len(_PLE_HEAD_VOCAB_SIZES))
)
_PLE_LAYER_MULTIPLIERS = (
    23_703_573_157_769,
    20_109_073_645_365,
    8_052_911_324_071,
)
_EXPECTED_COMPRESS_RATIOS = (0, 0, 0, 4) * 12


class Qwen4ExpGGUFConfigError(ValueError):
    """Raised when a qwen4exp artifact drifts from the frozen target geometry."""


@dataclass(frozen=True)
class Qwen4ExpGGUFConfig:
    architecture: str
    block_count: int
    context_length: int
    hidden_size: int
    vocab_size: int
    residual_branch_count: int
    residual_low_rank: int
    full_attention_interval: int
    attention_compress_ratios: tuple[int, ...]
    layer_types: tuple[str, ...]
    attention_head_count: int
    attention_kv_head_count: int
    attention_key_length: int
    attention_value_length: int
    attention_rms_epsilon: float
    indexer_head_count: int
    indexer_key_length: int
    qsa_token_budget: int
    rope_dimension_count: int
    rope_dimension_sections: tuple[int, ...]
    rope_freq_base: float
    gdn_conv_kernel: int
    gdn_group_count: int
    gdn_inner_size: int
    gdn_state_size: int
    gdn_time_step_rank: int
    expert_count: int
    expert_used_count: int
    expert_feed_forward_length: int
    shared_expert_feed_forward_length: int
    ple_layers: tuple[int, ...]
    ple_ngram_size: int
    ple_heads_per_ngram: int
    ple_conv_kernel: int
    ple_eos_token_id: int
    ple_image_token_id: int
    ple_row_width: int
    ple_layer_multipliers: tuple[int, ...]
    ple_head_offsets: tuple[int, ...]
    ple_head_vocab_sizes: tuple[int, ...]
    tokenizer_bos_token_id: int
    tokenizer_eos_token_id: int
    tokenizer_padding_token_id: int

    @property
    def residual_width(self) -> int:
        return self.hidden_size * self.residual_branch_count

    @property
    def qsa_compression_ratio(self) -> int:
        return max(self.attention_compress_ratios)

    @property
    def qsa_block_budget(self) -> int:
        return self.qsa_token_budget // self.qsa_compression_ratio

    @property
    def qsa_dense_equivalent_max_tokens(self) -> int:
        return self.qsa_token_budget + self.qsa_compression_ratio - 1

    @property
    def ple_row_count(self) -> int:
        return sum(self.ple_head_vocab_sizes)

    @property
    def qsa_layer_count(self) -> int:
        return self.layer_types.count(QSA)

    @property
    def bf16_kv_bytes_per_token(self) -> int:
        dtype_bytes = 2
        key_and_value = 2
        return (
            self.qsa_layer_count
            * self.attention_kv_head_count
            * self.attention_key_length
            * key_and_value
            * dtype_bytes
        )

    @property
    def bf16_compressed_index_bytes_per_token(self) -> int:
        dtype_bytes = 2
        return (
            self.qsa_layer_count
            * self.indexer_key_length
            * dtype_bytes
            // self.qsa_compression_ratio
        )

    @property
    def fp32_raw_index_bytes_per_token(self) -> int:
        """Current correctness-first raw index-key payload across QSA layers."""

        return self.qsa_layer_count * self.indexer_key_length * 4


def qwen4_exp_gguf_config_from_metadata(info: GGUFModelInfo) -> Qwen4ExpGGUFConfig:
    """Parse and strictly validate the frozen Qwen3.8-Flash-Next GGUF header."""

    metadata = info.metadata
    tokens = _array(metadata, "tokenizer.ggml.tokens")
    ratios = _int_array(metadata, "qwen4exp.attention.compress_ratios")
    config = Qwen4ExpGGUFConfig(
        architecture=str(_required(metadata, "general.architecture")),
        block_count=_integer(metadata, "qwen4exp.block_count"),
        context_length=_integer(metadata, "qwen4exp.context_length"),
        hidden_size=_integer(metadata, "qwen4exp.embedding_length"),
        vocab_size=len(tokens),
        residual_branch_count=_integer(metadata, "qwen4exp.hyper_connection.count"),
        residual_low_rank=_integer(metadata, "qwen4exp.hyper_connection.low_rank"),
        full_attention_interval=_integer(metadata, "qwen4exp.full_attention_interval"),
        attention_compress_ratios=ratios,
        layer_types=tuple(QSA if ratio else GDN for ratio in ratios),
        attention_head_count=_integer(metadata, "qwen4exp.attention.head_count"),
        attention_kv_head_count=_integer(metadata, "qwen4exp.attention.head_count_kv"),
        attention_key_length=_integer(metadata, "qwen4exp.attention.key_length"),
        attention_value_length=_integer(metadata, "qwen4exp.attention.value_length"),
        attention_rms_epsilon=_floating(
            metadata, "qwen4exp.attention.layer_norm_rms_epsilon"
        ),
        indexer_head_count=_integer(
            metadata, "qwen4exp.attention.indexer.head_count"
        ),
        indexer_key_length=_integer(
            metadata, "qwen4exp.attention.indexer.key_length"
        ),
        qsa_token_budget=_integer(metadata, "qwen4exp.attention.indexer.top_k"),
        rope_dimension_count=_integer(metadata, "qwen4exp.rope.dimension_count"),
        rope_dimension_sections=_int_array(
            metadata, "qwen4exp.rope.dimension_sections"
        ),
        rope_freq_base=_floating(metadata, "qwen4exp.rope.freq_base"),
        gdn_conv_kernel=_integer(metadata, "qwen4exp.ssm.conv_kernel"),
        gdn_group_count=_integer(metadata, "qwen4exp.ssm.group_count"),
        gdn_inner_size=_integer(metadata, "qwen4exp.ssm.inner_size"),
        gdn_state_size=_integer(metadata, "qwen4exp.ssm.state_size"),
        gdn_time_step_rank=_integer(metadata, "qwen4exp.ssm.time_step_rank"),
        expert_count=_integer(metadata, "qwen4exp.expert_count"),
        expert_used_count=_integer(metadata, "qwen4exp.expert_used_count"),
        expert_feed_forward_length=_integer(
            metadata, "qwen4exp.expert_feed_forward_length"
        ),
        shared_expert_feed_forward_length=_integer(
            metadata, "qwen4exp.expert_shared_feed_forward_length"
        ),
        ple_layers=_int_array(metadata, "qwen4exp.ple.layers"),
        ple_ngram_size=_integer(metadata, "qwen4exp.ple.ngram_size"),
        ple_heads_per_ngram=_integer(metadata, "qwen4exp.ple.heads_per_ngram"),
        ple_conv_kernel=_integer(metadata, "qwen4exp.ple.conv_kernel"),
        ple_eos_token_id=_integer(metadata, "qwen4exp.ple.eos_token_id"),
        ple_image_token_id=_integer(metadata, "qwen4exp.ple.image_token_id"),
        ple_row_width=_integer(
            metadata, "qwen4exp.embedding_length_per_layer_input"
        ),
        ple_layer_multipliers=_int_array(
            metadata, "qwen4exp.ple.layer_multipliers"
        ),
        ple_head_offsets=_int_array(metadata, "qwen4exp.ple.head_offsets"),
        ple_head_vocab_sizes=_int_array(
            metadata, "qwen4exp.ple.head_vocab_sizes"
        ),
        tokenizer_bos_token_id=_integer(metadata, "tokenizer.ggml.bos_token_id"),
        tokenizer_eos_token_id=_integer(metadata, "tokenizer.ggml.eos_token_id"),
        tokenizer_padding_token_id=_integer(
            metadata, "tokenizer.ggml.padding_token_id"
        ),
    )
    errors = _geometry_errors(config)
    if errors:
        raise Qwen4ExpGGUFConfigError("; ".join(errors))
    return config


def _geometry_errors(config: Qwen4ExpGGUFConfig) -> list[str]:
    errors: list[str] = []
    expected: tuple[tuple[str, Any, Any], ...] = (
        ("general.architecture", config.architecture, "qwen4exp"),
        ("qwen4exp.block_count", config.block_count, 48),
        ("qwen4exp.context_length", config.context_length, 262_144),
        ("qwen4exp.embedding_length", config.hidden_size, 2_560),
        ("tokenizer.ggml.tokens", config.vocab_size, 248_320),
        ("qwen4exp.hyper_connection.count", config.residual_branch_count, 4),
        ("qwen4exp.hyper_connection.low_rank", config.residual_low_rank, 320),
        ("qwen4exp.full_attention_interval", config.full_attention_interval, 4),
        (
            "qwen4exp.attention.compress_ratios",
            config.attention_compress_ratios,
            _EXPECTED_COMPRESS_RATIOS,
        ),
        ("qwen4exp.attention.head_count", config.attention_head_count, 24),
        ("qwen4exp.attention.head_count_kv", config.attention_kv_head_count, 2),
        ("qwen4exp.attention.key_length", config.attention_key_length, 256),
        ("qwen4exp.attention.value_length", config.attention_value_length, 256),
        ("qwen4exp.attention.indexer.head_count", config.indexer_head_count, 4),
        ("qwen4exp.attention.indexer.key_length", config.indexer_key_length, 128),
        ("qwen4exp.attention.indexer.top_k", config.qsa_token_budget, 2_048),
        ("qwen4exp.rope.dimension_count", config.rope_dimension_count, 64),
        ("qwen4exp.rope.dimension_sections", config.rope_dimension_sections, (11, 11, 10, 0)),
        ("qwen4exp.ssm.conv_kernel", config.gdn_conv_kernel, 4),
        ("qwen4exp.ssm.group_count", config.gdn_group_count, 16),
        ("qwen4exp.ssm.inner_size", config.gdn_inner_size, 6_144),
        ("qwen4exp.ssm.state_size", config.gdn_state_size, 128),
        ("qwen4exp.ssm.time_step_rank", config.gdn_time_step_rank, 48),
        ("qwen4exp.expert_count", config.expert_count, 512),
        ("qwen4exp.expert_used_count", config.expert_used_count, 10),
        (
            "qwen4exp.expert_feed_forward_length",
            config.expert_feed_forward_length,
            640,
        ),
        (
            "qwen4exp.expert_shared_feed_forward_length",
            config.shared_expert_feed_forward_length,
            640,
        ),
        ("qwen4exp.ple.layers", config.ple_layers, (1,)),
        ("qwen4exp.ple.ngram_size", config.ple_ngram_size, 3),
        ("qwen4exp.ple.heads_per_ngram", config.ple_heads_per_ngram, 8),
        ("qwen4exp.ple.conv_kernel", config.ple_conv_kernel, 4),
        ("qwen4exp.ple.eos_token_id", config.ple_eos_token_id, 248_044),
        ("qwen4exp.ple.image_token_id", config.ple_image_token_id, 248_056),
        ("qwen4exp.embedding_length_per_layer_input", config.ple_row_width, 160),
        (
            "qwen4exp.ple.layer_multipliers",
            config.ple_layer_multipliers,
            _PLE_LAYER_MULTIPLIERS,
        ),
        ("qwen4exp.ple.head_offsets", config.ple_head_offsets, _PLE_HEAD_OFFSETS),
        (
            "qwen4exp.ple.head_vocab_sizes",
            config.ple_head_vocab_sizes,
            _PLE_HEAD_VOCAB_SIZES,
        ),
        ("tokenizer.ggml.bos_token_id", config.tokenizer_bos_token_id, 248_044),
        ("tokenizer.ggml.eos_token_id", config.tokenizer_eos_token_id, 248_046),
        (
            "tokenizer.ggml.padding_token_id",
            config.tokenizer_padding_token_id,
            248_044,
        ),
    )
    for key, actual, wanted in expected:
        if actual != wanted:
            errors.append(f"{key} is {actual!r}, expected {wanted!r}")
    if not isclose(config.attention_rms_epsilon, 1e-6, rel_tol=1e-6, abs_tol=0.0):
        errors.append(
            "qwen4exp.attention.layer_norm_rms_epsilon is "
            f"{config.attention_rms_epsilon!r}, expected 1e-06"
        )
    if not isclose(config.rope_freq_base, 10_000_000.0, rel_tol=0.0, abs_tol=0.0):
        errors.append(
            f"qwen4exp.rope.freq_base is {config.rope_freq_base!r}, expected 10000000.0"
        )
    return errors


class Qwen4ExpGGUFTensorMapError(ValueError):
    """Raised when qwen4exp tensor names, owners, or shapes are invalid."""


@dataclass(frozen=True)
class Qwen4ExpGGUFTensorRef:
    part_index: int
    part_path: Path
    tensor: GGUFTensorInfo


@dataclass(frozen=True)
class Qwen4ExpGGUFLayerMap:
    layer_id: int
    layer_type: str
    slots: Mapping[str, Qwen4ExpGGUFTensorRef]

    def tensor(self, slot: str) -> Qwen4ExpGGUFTensorRef:
        try:
            return self.slots[slot]
        except KeyError as exc:
            raise KeyError(f"unknown qwen4exp layer slot {slot!r}") from exc


@dataclass(frozen=True)
class Qwen4ExpGGUFMappingValidation:
    config: Qwen4ExpGGUFConfig
    missing_tensor_names: tuple[str, ...]
    unexpected_tensor_names: tuple[str, ...]
    duplicate_tensor_names: tuple[str, ...]
    shape_errors: tuple[str, ...]
    split_errors: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not (
            self.missing_tensor_names
            or self.unexpected_tensor_names
            or self.duplicate_tensor_names
            or self.shape_errors
            or self.split_errors
        )


@dataclass(frozen=True)
class Qwen4ExpGGUFModelMap:
    config: Qwen4ExpGGUFConfig
    roots: Mapping[str, Qwen4ExpGGUFTensorRef]
    layers: tuple[Qwen4ExpGGUFLayerMap, ...]
    ple_table: Qwen4ExpGGUFTensorRef | None
    tensor_refs: tuple[Qwen4ExpGGUFTensorRef, ...]
    validation: Qwen4ExpGGUFMappingValidation
    part_paths: tuple[Path, ...]
    ple_padding_rows: int

    @property
    def tensor_names(self) -> tuple[str, ...]:
        return tuple(ref.tensor.name for ref in self.tensor_refs)

    def root(self, slot: str) -> Qwen4ExpGGUFTensorRef:
        try:
            return self.roots[slot]
        except KeyError as exc:
            raise KeyError(f"unknown qwen4exp root slot {slot!r}") from exc

    def layer(self, layer_id: int) -> Qwen4ExpGGUFLayerMap:
        if not 0 <= layer_id < len(self.layers):
            raise IndexError(f"qwen4exp layer {layer_id} is out of range")
        return self.layers[layer_id]


_ROOT_SLOTS = {
    "token_embedding": "token_embd.weight",
    "lm_head": "output.weight",
    "head_hc_norm": "output_hc_norm.weight",
    "head_hc_down": "output_hc_down.weight",
    "head_hc_up": "output_hc_up.weight",
}
_COMMON_LAYER_SLOTS = {
    "hc_attn_norm": "hc_attn_norm.weight",
    "hc_attn_down": "hc_attn_down.weight",
    "hc_attn_up": "hc_attn_up.weight",
    "hc_attn_inject": "hc_attn_inject.weight",
    "hc_ffn_norm": "hc_ffn_norm.weight",
    "hc_ffn_down": "hc_ffn_down.weight",
    "hc_ffn_up": "hc_ffn_up.weight",
    "hc_ffn_inject": "hc_ffn_inject.weight",
    "router": "ffn_gate_inp.weight",
    "shared_expert_gate": "ffn_gate_inp_shexp.weight",
    "expert_gate": "ffn_gate_exps.weight",
    "expert_up": "ffn_up_exps.weight",
    "expert_down": "ffn_down_exps.weight",
    "shared_gate": "ffn_gate_shexp.weight",
    "shared_up": "ffn_up_shexp.weight",
    "shared_down": "ffn_down_shexp.weight",
}
_GDN_LAYER_SLOTS = {
    "attn_qkv": "attn_qkv.weight",
    "attn_gate": "attn_gate.weight",
    "ssm_a": "ssm_a",
    "ssm_alpha": "ssm_alpha.weight",
    "ssm_beta": "ssm_beta.weight",
    "ssm_conv1d": "ssm_conv1d.weight",
    "ssm_dt_bias": "ssm_dt.bias",
    "ssm_norm": "ssm_norm.weight",
    "ssm_out": "ssm_out.weight",
}
_QSA_LAYER_SLOTS = {
    "attn_q": "attn_q.weight",
    "attn_q_norm": "attn_q_norm.weight",
    "attn_k": "attn_k.weight",
    "attn_k_norm": "attn_k_norm.weight",
    "attn_v": "attn_v.weight",
    "attn_output": "attn_output.weight",
    "index_q": "indexer.q_proj.weight",
    "index_k": "indexer.k_proj.weight",
    "index_q_norm": "indexer.q_norm.weight",
    "index_k_norm": "indexer.k_norm.weight",
}
_PLE_LAYER_SLOTS = {
    "ple_key": "ple_key.weight",
    "ple_value": "ple_value.weight",
    "ple_norm_key": "ple_norm_key.weight",
    "ple_norm_query": "ple_norm_query.weight",
    "ple_norm_conv": "ple_norm_conv.weight",
    "ple_conv1d": "ple_conv1d.weight",
}
_PLE_TABLE_NAME = "per_layer_token_embd.weight"


def required_qwen4_exp_gguf_tensor_names(
    config: Qwen4ExpGGUFConfig,
) -> tuple[str, ...]:
    names = [*_ROOT_SLOTS.values(), _PLE_TABLE_NAME]
    for layer_id, layer_type in enumerate(config.layer_types):
        suffixes = dict(_COMMON_LAYER_SLOTS)
        suffixes.update(_QSA_LAYER_SLOTS if layer_type == QSA else _GDN_LAYER_SLOTS)
        if layer_id in config.ple_layers:
            suffixes.update(_PLE_LAYER_SLOTS)
        names.extend(f"blk.{layer_id}.{suffix}" for suffix in suffixes.values())
    return tuple(names)


def validate_qwen4_exp_gguf_tensor_map(
    infos: Sequence[GGUFModelInfo],
) -> Qwen4ExpGGUFMappingValidation:
    parts = tuple(infos)
    if not parts:
        raise Qwen4ExpGGUFTensorMapError("at least one qwen4exp GGUF part is required")
    metadata_info = next(
        (
            info
            for info in parts
            if int(info.metadata.get("split.no", 0)) == 0
            and info.metadata.get("general.architecture") == "qwen4exp"
        ),
        parts[0],
    )
    config = qwen4_exp_gguf_config_from_metadata(metadata_info)
    names = Counter(tensor.name for info in parts for tensor in info.tensors)
    actual_names = set(names)
    required_names = set(required_qwen4_exp_gguf_tensor_names(config))
    by_name = {
        tensor.name: tensor
        for info in parts
        for tensor in info.tensors
        if names[tensor.name] == 1
    }
    split_errors = _qwen4_exp_split_errors(parts, expected_tensors=len(required_names))
    return Qwen4ExpGGUFMappingValidation(
        config=config,
        missing_tensor_names=tuple(sorted(required_names - actual_names)),
        unexpected_tensor_names=tuple(sorted(actual_names - required_names)),
        duplicate_tensor_names=tuple(sorted(name for name, count in names.items() if count > 1)),
        shape_errors=tuple(_qwen4_exp_shape_errors(config, by_name)),
        split_errors=tuple(split_errors),
    )


def build_qwen4_exp_gguf_tensor_map(
    infos: Sequence[GGUFModelInfo],
    *,
    strict: bool = True,
) -> Qwen4ExpGGUFModelMap:
    parts = tuple(infos)
    validation = validate_qwen4_exp_gguf_tensor_map(parts)
    if strict and not validation.passed:
        raise Qwen4ExpGGUFTensorMapError(_mapping_error_message(validation))

    owners: dict[str, Qwen4ExpGGUFTensorRef] = {}
    for part_index, info in enumerate(parts):
        for tensor in info.tensors:
            owners.setdefault(
                tensor.name,
                Qwen4ExpGGUFTensorRef(part_index, info.path, tensor),
            )
    required = required_qwen4_exp_gguf_tensor_names(validation.config)
    available = tuple(owners[name] for name in required if name in owners)
    roots = {
        slot: owners[name]
        for slot, name in _ROOT_SLOTS.items()
        if name in owners
    }
    layers: list[Qwen4ExpGGUFLayerMap] = []
    for layer_id, layer_type in enumerate(validation.config.layer_types):
        suffixes = dict(_COMMON_LAYER_SLOTS)
        suffixes.update(_QSA_LAYER_SLOTS if layer_type == QSA else _GDN_LAYER_SLOTS)
        if layer_id in validation.config.ple_layers:
            suffixes.update(_PLE_LAYER_SLOTS)
        slots = {
            slot: owners[f"blk.{layer_id}.{suffix}"]
            for slot, suffix in suffixes.items()
            if f"blk.{layer_id}.{suffix}" in owners
        }
        layers.append(Qwen4ExpGGUFLayerMap(layer_id, layer_type, slots))
    ple_ref = owners.get(_PLE_TABLE_NAME)
    if ple_ref is None:
        if strict:
            raise Qwen4ExpGGUFTensorMapError("missing PLE table")
        ple_padding = 0
        ple_ref = None
    else:
        ple_padding = ple_ref.tensor.shape[0] - validation.config.ple_row_count
    return Qwen4ExpGGUFModelMap(
        config=validation.config,
        roots=roots,
        layers=tuple(layers),
        ple_table=ple_ref,
        tensor_refs=available,
        validation=validation,
        part_paths=tuple(info.path for info in parts),
        ple_padding_rows=ple_padding,
    )


def _qwen4_exp_split_errors(
    infos: tuple[GGUFModelInfo, ...],
    *,
    expected_tensors: int,
) -> list[str]:
    counts = {int(info.metadata.get("split.count", 1)) for info in infos}
    declared = {
        int(info.metadata.get("split.tensors.count", expected_tensors)) for info in infos
    }
    numbers = sorted(
        int(info.metadata.get("split.no", index)) for index, info in enumerate(infos)
    )
    errors: list[str] = []
    if len(counts) != 1:
        errors.append(f"inconsistent split.count values: {sorted(counts)}")
    else:
        count = next(iter(counts))
        if numbers != list(range(count)):
            errors.append(f"split part numbers {numbers} do not cover {list(range(count))}")
    if len(declared) != 1:
        errors.append(f"inconsistent split.tensors.count values: {sorted(declared)}")
    elif next(iter(declared)) != expected_tensors:
        errors.append(
            f"split declares {next(iter(declared))} tensors, expected {expected_tensors}"
        )
    return errors


def _qwen4_exp_expected_shapes(
    config: Qwen4ExpGGUFConfig,
) -> dict[str, tuple[int, ...]]:
    hidden = config.hidden_size
    residual = config.residual_width
    low_rank = config.residual_low_rank
    shapes: dict[str, tuple[int, ...]] = {
        "token_embd.weight": (config.vocab_size, hidden),
        "output.weight": (config.vocab_size, hidden),
        "output_hc_norm.weight": (residual,),
        "output_hc_down.weight": (low_rank, residual),
        "output_hc_up.weight": (residual, low_rank),
    }
    for layer_id, layer_type in enumerate(config.layer_types):
        prefix = f"blk.{layer_id}."
        layer_shapes = {
            "hc_attn_norm.weight": (residual,),
            "hc_attn_down.weight": (low_rank, residual),
            "hc_attn_up.weight": (residual, low_rank),
            "hc_attn_inject.weight": (config.residual_branch_count, residual),
            "hc_ffn_norm.weight": (residual,),
            "hc_ffn_down.weight": (low_rank, residual),
            "hc_ffn_up.weight": (residual, low_rank),
            "hc_ffn_inject.weight": (config.residual_branch_count, residual),
            "ffn_gate_inp.weight": (config.expert_count, hidden),
            "ffn_gate_inp_shexp.weight": (hidden,),
            "ffn_gate_exps.weight": (
                config.expert_count,
                config.expert_feed_forward_length,
                hidden,
            ),
            "ffn_up_exps.weight": (
                config.expert_count,
                config.expert_feed_forward_length,
                hidden,
            ),
            "ffn_down_exps.weight": (
                config.expert_count,
                hidden,
                config.expert_feed_forward_length,
            ),
            "ffn_gate_shexp.weight": (config.shared_expert_feed_forward_length, hidden),
            "ffn_up_shexp.weight": (config.shared_expert_feed_forward_length, hidden),
            "ffn_down_shexp.weight": (hidden, config.shared_expert_feed_forward_length),
        }
        if layer_type == QSA:
            q_width = config.attention_head_count * config.attention_key_length * 2
            attention_width = config.attention_head_count * config.attention_value_length
            layer_shapes.update(
                {
                    "attn_q.weight": (q_width, hidden),
                    "attn_q_norm.weight": (config.attention_key_length,),
                    "attn_k.weight": (
                        config.attention_kv_head_count * config.attention_key_length,
                        hidden,
                    ),
                    "attn_k_norm.weight": (config.attention_key_length,),
                    "attn_v.weight": (
                        config.attention_kv_head_count * config.attention_value_length,
                        hidden,
                    ),
                    "attn_output.weight": (hidden, attention_width),
                    "indexer.q_proj.weight": (
                        config.indexer_head_count * config.indexer_key_length,
                        hidden,
                    ),
                    "indexer.k_proj.weight": (config.indexer_key_length, hidden),
                    "indexer.q_norm.weight": (config.indexer_key_length,),
                    "indexer.k_norm.weight": (config.indexer_key_length,),
                }
            )
        else:
            layer_shapes.update(
                {
                    "attn_qkv.weight": (10_240, hidden),
                    "attn_gate.weight": (config.gdn_inner_size, hidden),
                    "ssm_a": (config.gdn_time_step_rank,),
                    "ssm_alpha.weight": (config.gdn_time_step_rank, hidden),
                    "ssm_beta.weight": (config.gdn_time_step_rank, hidden),
                    "ssm_conv1d.weight": (10_240, config.gdn_conv_kernel),
                    "ssm_dt.bias": (config.gdn_time_step_rank,),
                    "ssm_norm.weight": (config.gdn_state_size,),
                    "ssm_out.weight": (hidden, config.gdn_inner_size),
                }
            )
        if layer_id in config.ple_layers:
            layer_shapes.update(
                {
                    "ple_key.weight": (residual, hidden),
                    "ple_value.weight": (hidden, hidden),
                    "ple_norm_key.weight": (residual,),
                    "ple_norm_query.weight": (residual,),
                    "ple_norm_conv.weight": (residual,),
                    "ple_conv1d.weight": (residual, config.ple_conv_kernel),
                }
            )
        shapes.update({prefix + suffix: shape for suffix, shape in layer_shapes.items()})
    return shapes


def _qwen4_exp_shape_errors(
    config: Qwen4ExpGGUFConfig,
    actual: Mapping[str, Any],
) -> list[str]:
    errors: list[str] = []
    expected = _qwen4_exp_expected_shapes(config)
    for name in sorted(expected.keys() & actual.keys()):
        if actual[name].shape != expected[name]:
            errors.append(
                f"{name} shape {actual[name].shape}, expected {expected[name]}"
            )
    ple = actual.get(_PLE_TABLE_NAME)
    if ple is not None:
        rows, width = ple.shape if len(ple.shape) == 2 else (-1, -1)
        max_rows = config.ple_row_count + 255
        if width != config.ple_row_width or not config.ple_row_count <= rows <= max_rows:
            errors.append(
                f"{_PLE_TABLE_NAME} shape {ple.shape}, expected "
                f"[{config.ple_row_count}..{max_rows}, {config.ple_row_width}]"
            )
    return errors


def _mapping_error_message(validation: Qwen4ExpGGUFMappingValidation) -> str:
    problems: list[str] = []
    if validation.missing_tensor_names:
        problems.append("missing: " + ", ".join(validation.missing_tensor_names[:8]))
    if validation.unexpected_tensor_names:
        problems.append("unexpected: " + ", ".join(validation.unexpected_tensor_names[:8]))
    if validation.duplicate_tensor_names:
        problems.append("duplicate: " + ", ".join(validation.duplicate_tensor_names[:8]))
    problems.extend(validation.shape_errors[:8])
    problems.extend(validation.split_errors[:8])
    return "invalid qwen4exp GGUF tensor map; " + "; ".join(problems)


def _required(metadata: Mapping[str, Any], key: str) -> Any:
    try:
        return metadata[key]
    except KeyError as exc:
        raise Qwen4ExpGGUFConfigError(f"missing required GGUF metadata: {key}") from exc


def _integer(metadata: Mapping[str, Any], key: str) -> int:
    value = _required(metadata, key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise Qwen4ExpGGUFConfigError(f"{key} must be an integer, got {type(value).__name__}")
    return int(value)


def _floating(metadata: Mapping[str, Any], key: str) -> float:
    value = _required(metadata, key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Qwen4ExpGGUFConfigError(f"{key} must be numeric, got {type(value).__name__}")
    return float(value)


def _array(metadata: Mapping[str, Any], key: str) -> list[Any]:
    value = _required(metadata, key)
    if not isinstance(value, list):
        raise Qwen4ExpGGUFConfigError(f"{key} must be an array, got {type(value).__name__}")
    return value


def _int_array(metadata: Mapping[str, Any], key: str) -> tuple[int, ...]:
    values = _array(metadata, key)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise Qwen4ExpGGUFConfigError(f"{key} must contain only integers")
    return tuple(int(value) for value in values)


__all__ = [
    "GDN",
    "QSA",
    "Qwen4ExpGGUFConfig",
    "Qwen4ExpGGUFConfigError",
    "Qwen4ExpGGUFLayerMap",
    "Qwen4ExpGGUFMappingValidation",
    "Qwen4ExpGGUFModelMap",
    "Qwen4ExpGGUFTensorMapError",
    "Qwen4ExpGGUFTensorRef",
    "build_qwen4_exp_gguf_tensor_map",
    "qwen4_exp_gguf_config_from_metadata",
    "required_qwen4_exp_gguf_tensor_names",
    "validate_qwen4_exp_gguf_tensor_map",
]
