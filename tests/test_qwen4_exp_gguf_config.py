from __future__ import annotations

from pathlib import Path

import pytest

from hipengine.loading.gguf import GGUFModelInfo, scan_gguf
from hipengine.loading.qwen4_exp_gguf import (
    GDN,
    QSA,
    Qwen4ExpGGUFConfigError,
    qwen4_exp_gguf_config_from_metadata,
)

UNSLOTH_PART0 = Path(
    "/models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL/"
    "Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf"
)


def _metadata() -> dict[str, object]:
    head_sizes = (
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
    offsets: list[int] = []
    offset = 0
    for size in head_sizes:
        offsets.append(offset)
        offset += size
    return {
        "general.architecture": "qwen4exp",
        "qwen4exp.block_count": 48,
        "qwen4exp.context_length": 262_144,
        "qwen4exp.embedding_length": 2_560,
        "qwen4exp.hyper_connection.count": 4,
        "qwen4exp.hyper_connection.low_rank": 320,
        "qwen4exp.full_attention_interval": 4,
        "qwen4exp.attention.compress_ratios": [0, 0, 0, 4] * 12,
        "qwen4exp.attention.head_count": 24,
        "qwen4exp.attention.head_count_kv": 2,
        "qwen4exp.attention.key_length": 256,
        "qwen4exp.attention.value_length": 256,
        "qwen4exp.attention.layer_norm_rms_epsilon": 1e-6,
        "qwen4exp.attention.indexer.head_count": 4,
        "qwen4exp.attention.indexer.key_length": 128,
        "qwen4exp.attention.indexer.top_k": 2_048,
        "qwen4exp.rope.dimension_count": 64,
        "qwen4exp.rope.dimension_sections": [11, 11, 10, 0],
        "qwen4exp.rope.freq_base": 10_000_000.0,
        "qwen4exp.ssm.conv_kernel": 4,
        "qwen4exp.ssm.group_count": 16,
        "qwen4exp.ssm.inner_size": 6_144,
        "qwen4exp.ssm.state_size": 128,
        "qwen4exp.ssm.time_step_rank": 48,
        "qwen4exp.expert_count": 512,
        "qwen4exp.expert_used_count": 10,
        "qwen4exp.expert_feed_forward_length": 640,
        "qwen4exp.expert_shared_feed_forward_length": 640,
        "qwen4exp.ple.layers": [1],
        "qwen4exp.ple.ngram_size": 3,
        "qwen4exp.ple.heads_per_ngram": 8,
        "qwen4exp.ple.conv_kernel": 4,
        "qwen4exp.ple.eos_token_id": 248_044,
        "qwen4exp.ple.image_token_id": 248_056,
        "qwen4exp.embedding_length_per_layer_input": 160,
        "qwen4exp.ple.layer_multipliers": [
            23_703_573_157_769,
            20_109_073_645_365,
            8_052_911_324_071,
        ],
        "qwen4exp.ple.head_offsets": offsets,
        "qwen4exp.ple.head_vocab_sizes": list(head_sizes),
        "tokenizer.ggml.tokens": [None] * 248_320,
        "tokenizer.ggml.bos_token_id": 248_044,
        "tokenizer.ggml.eos_token_id": 248_046,
        "tokenizer.ggml.padding_token_id": 248_044,
    }


def _info(metadata: dict[str, object] | None = None) -> GGUFModelInfo:
    return GGUFModelInfo(
        path=Path("synthetic-qwen4exp.gguf"),
        version=3,
        alignment=32,
        metadata=_metadata() if metadata is None else metadata,
        tensors=(),
        tensor_data_offset=0,
    )


def test_qwen4_exp_gguf_config_parses_frozen_text_geometry() -> None:
    config = qwen4_exp_gguf_config_from_metadata(_info())

    assert config.architecture == "qwen4exp"
    assert config.block_count == 48
    assert config.layer_types == (GDN, GDN, GDN, QSA) * 12
    assert config.hidden_size == 2_560
    assert config.residual_branch_count == 4
    assert config.residual_width == 10_240
    assert config.qsa_token_budget == 2_048
    assert config.qsa_block_budget == 512
    assert config.qsa_dense_equivalent_max_tokens == 2_051
    assert config.ple_layers == (1,)
    assert config.ple_row_count == 320_001_446
    assert config.ple_row_width == 160
    assert config.vocab_size == 248_320
    assert config.bf16_kv_bytes_per_token == 24_576
    assert config.bf16_compressed_index_bytes_per_token == 768


@pytest.mark.parametrize(
    ("key", "value"),
    (
        ("general.architecture", "qwen35moe"),
        ("qwen4exp.embedding_length", 2_561),
        ("qwen4exp.hyper_connection.count", 3),
        ("qwen4exp.attention.compress_ratios", [0, 0, 0, 0] * 12),
        ("qwen4exp.attention.indexer.top_k", 512),
        ("qwen4exp.expert_used_count", 9),
        ("qwen4exp.ple.layers", [2]),
        (
            "qwen4exp.ple.layer_multipliers",
            [23_703_573_157_770, 20_109_073_645_365, 8_052_911_324_071],
        ),
        ("tokenizer.ggml.tokens", [None] * 248_319),
    ),
)
def test_qwen4_exp_gguf_config_rejects_geometry_drift(key: str, value: object) -> None:
    metadata = _metadata()
    metadata[key] = value

    with pytest.raises(Qwen4ExpGGUFConfigError, match=key):
        qwen4_exp_gguf_config_from_metadata(_info(metadata))


def test_qwen4_exp_gguf_config_rejects_missing_required_metadata() -> None:
    metadata = _metadata()
    metadata.pop("qwen4exp.ple.head_vocab_sizes")

    with pytest.raises(Qwen4ExpGGUFConfigError, match="ple.head_vocab_sizes"):
        qwen4_exp_gguf_config_from_metadata(_info(metadata))


def test_qwen4_exp_gguf_config_matches_real_unsloth_header_without_tensors() -> None:
    if not UNSLOTH_PART0.exists():
        pytest.skip(f"local Unsloth metadata shard not found: {UNSLOTH_PART0}")

    info = scan_gguf(UNSLOTH_PART0)
    config = qwen4_exp_gguf_config_from_metadata(info)

    assert info.tensor_count == 0
    assert config.block_count == 48
    assert config.ple_row_count == 320_001_446
    assert config.layer_types.count(QSA) == 12
