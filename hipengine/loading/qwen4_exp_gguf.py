"""Strict Qwen3.8-Flash-Next qwen4exp GGUF metadata contract."""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose
from typing import Any, Mapping

from hipengine.loading.gguf import GGUFModelInfo

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
    "qwen4_exp_gguf_config_from_metadata",
]
