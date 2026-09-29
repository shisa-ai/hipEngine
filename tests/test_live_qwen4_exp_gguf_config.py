from __future__ import annotations

from pathlib import Path

import pytest

from hipengine.loading.gguf import scan_gguf
from hipengine.loading.qwen4_exp_gguf import (
    GDN,
    QSA,
    Qwen4ExpGGUFConfigError,
    qwen4_exp_gguf_config_from_metadata,
)
from tests._qwen4_exp_gguf_fixtures import _info, _metadata

UNSLOTH_PART0 = Path(
    "/models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL/"
    "Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf"
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
    assert config.fp32_raw_index_bytes_per_token == 6_144


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
