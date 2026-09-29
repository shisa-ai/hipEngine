"""Unit tests for Gemma 4 MTP assistant capability admission (punchlist M1).

The synthetic pair reproduces the measured geometry of gemma-4-26B-A4B and its
``mtp-gemma-4-26B-A4B-it-Q8_0`` sidecar: a 30-layer target whose sliding pattern
is ``[T,T,T,T,T,F] x 5`` and a 4-layer assistant ``[T,T,T,F]``, so the shared-KV
mapping lands on target layers 28 and 29. Admission is capability-based, so
every rejection below names the capability that failed rather than a file.
"""

from __future__ import annotations

from math import prod
from pathlib import Path

import pytest

from hipengine.loading.gemma4_mtp_gguf import (
    Gemma4MTPGGUFError,
    discover_gemma4_mtp_artifacts,
    parse_gemma4_mtp_config,
    require_gemma4_mtp_valid,
    shared_kv_target_layers,
    validate_gemma4_mtp_gguf,
)
from hipengine.loading.gguf import GGUFModelInfo, GGUFTensorInfo
from hipengine.quant.gguf import GGMLQuantizationType

TARGET_HIDDEN = 2816
ASSISTANT_HIDDEN = 1024
VOCAB = 262144
FEED_FORWARD = 8192
HEAD_COUNT = 16
TARGET_PATTERN = [True, True, True, True, True, False] * 5
ASSISTANT_PATTERN = [True, True, True, False]
ASSISTANT_KV_HEADS = [8, 8, 8, 2]


def _tensor(name: str, shape: tuple[int, ...], qtype: str) -> GGUFTensorInfo:
    quant = GGMLQuantizationType[qtype]
    return GGUFTensorInfo(
        name=name,
        shape=shape,
        ggml_shape=tuple(reversed(shape)),
        ggml_type=int(quant),
        ggml_type_name=qtype,
        n_elements=prod(shape),
        nbytes=prod(shape),
        offset=0,
        data_offset=0,
        byte_shape=shape,
    )


def _target_info(**overrides: object) -> GGUFModelInfo:
    metadata: dict[str, object] = {
        "general.architecture": "gemma4",
        "gemma4.block_count": len(TARGET_PATTERN),
        "gemma4.embedding_length": TARGET_HIDDEN,
        "gemma4.attention.head_count": HEAD_COUNT,
        "gemma4.attention.head_count_kv": [8, 8, 8, 8, 8, 2] * 5,
        "gemma4.attention.key_length": 512,
        "gemma4.attention.key_length_swa": 256,
        "gemma4.attention.value_length": 512,
        "gemma4.attention.value_length_swa": 256,
        "gemma4.attention.sliding_window_pattern": list(TARGET_PATTERN),
    }
    metadata.update(overrides)
    tensors = (
        _tensor("token_embd.weight", (VOCAB, TARGET_HIDDEN), "Q8_0"),
        _tensor("output_norm.weight", (TARGET_HIDDEN,), "F32"),
    )
    return GGUFModelInfo(
        path=Path("target.gguf"),
        version=3,
        alignment=32,
        metadata=metadata,
        tensors=tensors,
        tensor_data_offset=0,
    )


def _assistant_info(
    *, drop: set[str] | None = None, metadata_overrides: dict[str, object] | None = None
) -> GGUFModelInfo:
    metadata: dict[str, object] = {
        "general.architecture": "gemma4-assistant",
        "gemma4-assistant.block_count": 4,
        "gemma4-assistant.embedding_length": ASSISTANT_HIDDEN,
        "gemma4-assistant.embedding_length_out": TARGET_HIDDEN,
        "gemma4-assistant.feed_forward_length": FEED_FORWARD,
        "gemma4-assistant.nextn_predict_layers": 4,
        "gemma4-assistant.shared_kv_layers": 4,
        "gemma4-assistant.attention.head_count": HEAD_COUNT,
        "gemma4-assistant.attention.head_count_kv": list(ASSISTANT_KV_HEADS),
        "gemma4-assistant.attention.key_length": 512,
        "gemma4-assistant.attention.key_length_swa": 256,
        "gemma4-assistant.attention.value_length": 512,
        "gemma4-assistant.attention.value_length_swa": 256,
        "gemma4-assistant.attention.sliding_window_pattern": list(ASSISTANT_PATTERN),
    }
    if metadata_overrides:
        metadata.update(metadata_overrides)

    tensors = [
        _tensor("rope_freqs.weight", (256,), "F32"),
        _tensor("token_embd.weight", (VOCAB, ASSISTANT_HIDDEN), "Q8_0"),
        _tensor("output_norm.weight", (ASSISTANT_HIDDEN,), "F32"),
        _tensor("nextn.pre_projection.weight", (ASSISTANT_HIDDEN, 2 * TARGET_HIDDEN), "Q8_0"),
        _tensor("nextn.post_projection.weight", (TARGET_HIDDEN, ASSISTANT_HIDDEN), "Q8_0"),
    ]
    for layer, sliding in enumerate(ASSISTANT_PATTERN):
        key_len = 256 if sliding else 512
        value_len = 256 if sliding else 512
        tensors.extend(
            [
                _tensor(f"blk.{layer}.attn_norm.weight", (ASSISTANT_HIDDEN,), "F32"),
                _tensor(f"blk.{layer}.layer_output_scale.weight", (1,), "F32"),
                _tensor(f"blk.{layer}.ffn_down.weight", (ASSISTANT_HIDDEN, FEED_FORWARD), "Q8_0"),
                _tensor(f"blk.{layer}.ffn_gate.weight", (FEED_FORWARD, ASSISTANT_HIDDEN), "Q8_0"),
                _tensor(f"blk.{layer}.ffn_up.weight", (FEED_FORWARD, ASSISTANT_HIDDEN), "Q8_0"),
                _tensor(f"blk.{layer}.post_attention_norm.weight", (ASSISTANT_HIDDEN,), "F32"),
                _tensor(f"blk.{layer}.post_ffw_norm.weight", (ASSISTANT_HIDDEN,), "F32"),
                _tensor(f"blk.{layer}.ffn_norm.weight", (ASSISTANT_HIDDEN,), "F32"),
                _tensor(
                    f"blk.{layer}.attn_output.weight",
                    (ASSISTANT_HIDDEN, HEAD_COUNT * value_len),
                    "Q8_0",
                ),
                _tensor(f"blk.{layer}.attn_q_norm.weight", (256,), "F32"),
                _tensor(
                    f"blk.{layer}.attn_q.weight",
                    (HEAD_COUNT * key_len, ASSISTANT_HIDDEN),
                    "Q8_0",
                ),
            ]
        )
    if drop:
        tensors = [t for t in tensors if t.name not in drop]
    return GGUFModelInfo(
        path=Path("assistant.gguf"),
        version=3,
        alignment=32,
        metadata=metadata,
        tensors=tuple(tensors),
        tensor_data_offset=0,
    )


def test_admission_accepts_the_measured_geometry() -> None:
    validation = validate_gemma4_mtp_gguf(_target_info(), _assistant_info())
    assert validation.passed, validation
    assert validation.admission_errors == ()
    assert validation.missing_tensor_names == ()
    assert validation.shape_errors == ()


def test_shared_kv_mapping_lands_on_target_layers_28_and_29() -> None:
    validation = validate_gemma4_mtp_gguf(_target_info(), _assistant_info())
    assert validation.shared_kv_layers == (28, 28, 28, 29)


def test_config_decodes_without_reading_weight_payloads() -> None:
    config = parse_gemma4_mtp_config(_assistant_info())
    assert config.architecture == "gemma4-assistant"
    assert config.output_width == TARGET_HIDDEN
    assert config.hidden_size == ASSISTANT_HIDDEN
    assert config.sliding_window_pattern == (True, True, True, False)
    assert config.vocab_size == VOCAB


def test_rejects_wrong_architecture() -> None:
    validation = validate_gemma4_mtp_gguf(
        _target_info(),
        _assistant_info(metadata_overrides={"general.architecture": "gemma4"}),
    )
    assert not validation.passed
    assert any("architecture" in e for e in validation.admission_errors)


def test_rejects_output_width_that_is_not_the_target_hidden_size() -> None:
    validation = validate_gemma4_mtp_gguf(
        _target_info(),
        _assistant_info(
            metadata_overrides={"gemma4-assistant.embedding_length_out": ASSISTANT_HIDDEN}
        ),
    )
    assert not validation.passed
    assert any(
        "embedding_length_out" in e and str(TARGET_HIDDEN) in e
        for e in validation.admission_errors
    )


def test_rejects_vocabulary_that_the_target_cannot_embed() -> None:
    assistant = _assistant_info()
    tensors = tuple(
        t
        if t.name != "token_embd.weight"
        else _tensor("token_embd.weight", (1000, ASSISTANT_HIDDEN), "Q8_0")
        for t in assistant.tensors
    )
    validation = validate_gemma4_mtp_gguf(
        _target_info(), GGUFModelInfo(**{**assistant.__dict__, "tensors": tensors})
    )
    assert not validation.passed
    assert any("vocabulary" in e for e in validation.admission_errors)


def test_rejects_shared_kv_width_that_does_not_match_the_mapped_cache() -> None:
    # 4 KV heads against the target's 8 halves the sliding-layer cache width.
    validation = validate_gemma4_mtp_gguf(
        _target_info(),
        _assistant_info(
            metadata_overrides={
                "gemma4-assistant.attention.head_count_kv": [4, 4, 4, 2]
            }
        ),
    )
    assert not validation.passed
    assert any("shared-KV width" in e for e in validation.admission_errors)


def test_rejects_missing_projection_tensor() -> None:
    validation = validate_gemma4_mtp_gguf(
        _target_info(), _assistant_info(drop={"nextn.pre_projection.weight"})
    )
    assert not validation.passed
    assert "nextn.pre_projection.weight" in validation.missing_tensor_names


def test_rejects_shape_drift_on_a_shared_tensor() -> None:
    assistant = _assistant_info()
    tensors = tuple(
        t
        if t.name != "nextn.post_projection.weight"
        else _tensor("nextn.post_projection.weight", (1024, ASSISTANT_HIDDEN), "Q8_0")
        for t in assistant.tensors
    )
    validation = validate_gemma4_mtp_gguf(
        _target_info(), GGUFModelInfo(**{**assistant.__dict__, "tensors": tensors})
    )
    assert not validation.passed
    assert any("nextn.post_projection" in e for e in validation.shape_errors)


def test_rejects_layer_table_that_disagrees_with_block_count() -> None:
    validation = validate_gemma4_mtp_gguf(
        _target_info(),
        _assistant_info(
            metadata_overrides={
                "gemma4-assistant.attention.sliding_window_pattern": [True, True, False]
            }
        ),
    )
    assert not validation.passed
    assert any("sliding_window_pattern" in e for e in validation.admission_errors)


def test_mapping_requires_both_target_classes() -> None:
    with pytest.raises(Gemma4MTPGGUFError):
        shared_kv_target_layers([True, False], [True, True, True])
    with pytest.raises(Gemma4MTPGGUFError):
        shared_kv_target_layers([True], [])


def test_require_raises_naming_the_failed_capability() -> None:
    invalid = validate_gemma4_mtp_gguf(
        _target_info(),
        _assistant_info(
            metadata_overrides={"gemma4-assistant.embedding_length_out": ASSISTANT_HIDDEN}
        ),
    )
    with pytest.raises(Gemma4MTPGGUFError) as excinfo:
        require_gemma4_mtp_valid(invalid)
    message = str(excinfo.value)
    assert "capability admission" in message
    assert "embedding_length_out" in message
    assert str(TARGET_HIDDEN) in message

    valid = validate_gemma4_mtp_gguf(_target_info(), _assistant_info())
    assert require_gemma4_mtp_valid(valid) is valid


def test_discovery_finds_sidecars_and_absence_is_not_an_error(tmp_path: Path) -> None:
    target = tmp_path / "model.gguf"
    target.write_bytes(b"")
    assert discover_gemma4_mtp_artifacts(target) == ()

    mtp_dir = tmp_path / "MTP"
    mtp_dir.mkdir()
    sidecar = mtp_dir / "mtp-model.gguf"
    sidecar.write_bytes(b"")
    other = mtp_dir / "notes.txt"
    other.write_text("ignored")

    found = discover_gemma4_mtp_artifacts(target)
    assert found == (sidecar.resolve(),)
    # discovery is directory-relative, so a bare directory target behaves the same
    assert discover_gemma4_mtp_artifacts(tmp_path) == (sidecar.resolve(),)