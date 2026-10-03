"""Shape-contract tests for the Gemma 4 ``gemma4-assistant`` MTP draft head.

These exercise ``hipengine.loading.gemma4_assistant_gguf`` against the real
downloaded head when it is present, and against synthetic metadata otherwise.
No HIP is involved: the module reads GGUF headers only, so no ROCm guard is
needed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hipengine.loading.gguf import MissingGGUFTensorError, scan_gguf
from hipengine.loading.gemma4_assistant_gguf import (
    ARCHITECTURE,
    build_gemma4_assistant_tensor_map,
    expected_gemma4_assistant_shapes,
    gemma4_assistant_config_from_metadata,
    required_gemma4_assistant_tensor_names,
    validate_gemma4_assistant_tensor_map,
)

ARTIFACT = Path(
    "/models/gguf/gemma-4-26B-A4B-it-GGUF/mtp-gemma-4-26B-A4B-it-Q8_0.gguf"
)


def _synthetic_metadata() -> dict[str, object]:
    """The head's real metadata shape, at four blocks and two attention kinds."""

    return {
        "general.architecture": ARCHITECTURE,
        f"{ARCHITECTURE}.block_count": 4,
        f"{ARCHITECTURE}.embedding_length": 1024,
        f"{ARCHITECTURE}.embedding_length_out": 2816,
        f"{ARCHITECTURE}.feed_forward_length": 8192,
        f"{ARCHITECTURE}.attention.head_count": 16,
        f"{ARCHITECTURE}.attention.head_count_kv": [8, 8, 8, 8],
        f"{ARCHITECTURE}.attention.key_length": 512,
        f"{ARCHITECTURE}.attention.value_length": 512,
        f"{ARCHITECTURE}.attention.key_length_swa": 256,
        f"{ARCHITECTURE}.attention.value_length_swa": 256,
        f"{ARCHITECTURE}.attention.sliding_window": 1024,
        f"{ARCHITECTURE}.attention.sliding_window_pattern": [1, 1, 1, 0],
        f"{ARCHITECTURE}.rope.dimension_count": 512,
        f"{ARCHITECTURE}.rope.dimension_count_swa": 256,
        f"{ARCHITECTURE}.nextn_predict_layers": 4,
    }


def test_config_decodes_the_contract() -> None:
    config = gemma4_assistant_config_from_metadata(_synthetic_metadata())
    assert config.block_count == 4
    assert config.n_embd == 1024
    assert config.n_embd_backbone == 2816
    assert config.is_swa == (True, True, True, False)
    # concat(embedding, backbone_hidden) is the pre-projection input width.
    assert config.pre_projection_in == 5632


def test_config_fails_closed_on_missing_keys() -> None:
    metadata = _synthetic_metadata()
    del metadata[f"{ARCHITECTURE}.embedding_length_out"]
    with pytest.raises(MissingGGUFTensorError, match="embedding_length_out"):
        gemma4_assistant_config_from_metadata(metadata)


def test_config_rejects_a_short_sliding_window_pattern() -> None:
    metadata = _synthetic_metadata()
    metadata[f"{ARCHITECTURE}.attention.sliding_window_pattern"] = [1, 1, 1]
    with pytest.raises(MissingGGUFTensorError, match="sliding_window_pattern"):
        gemma4_assistant_config_from_metadata(metadata)


def test_required_names_are_four_blocks_of_eleven_plus_five() -> None:
    config = gemma4_assistant_config_from_metadata(_synthetic_metadata())
    names = required_gemma4_assistant_tensor_names(config)
    assert len(names) == 4 * 11 + 5 == 49
    # The head owns no cache, so it must not declare key or value tensors.
    assert not [n for n in names if n.endswith(("attn_k.weight", "attn_v.weight"))]


def test_shapes_follow_the_sliding_window_pattern() -> None:
    config = gemma4_assistant_config_from_metadata(_synthetic_metadata())
    shapes = expected_gemma4_assistant_shapes(config)
    # Sliding-window blocks use 16 heads x 256; the full block uses 16 x 512.
    assert shapes["blk.0.attn_q.weight"] == (1024, 4096)
    assert shapes["blk.0.attn_q_norm.weight"] == (256,)
    assert shapes["blk.3.attn_q.weight"] == (1024, 8192)
    assert shapes["blk.3.attn_q_norm.weight"] == (512,)
    # Globals.
    assert shapes["nextn.pre_projection.weight"] == (5632, 1024)
    assert shapes["nextn.post_projection.weight"] == (1024, 2816)
    assert len(shapes) == 49


@pytest.mark.skipif(not ARTIFACT.is_file(), reason=f"assistant head not present: {ARTIFACT}")
def test_real_artifact_satisfies_the_contract() -> None:
    info = scan_gguf(ARTIFACT)
    assert info.metadata["general.architecture"] == ARCHITECTURE

    validation = validate_gemma4_assistant_tensor_map(info)
    # No missing, no unexpected and no shape errors: the contract above is the
    # artifact's actual layout, not an approximation of it.
    validation.raise_for_errors()
    assert validation.passed
    assert len(validation.present) == 49

    tensor_map = build_gemma4_assistant_tensor_map(info)
    assert tensor_map.config.n_embd_backbone == 2816
    assert tensor_map.block_tensor(3, "attn_q.weight").ggml_shape == (1024, 8192)
    assert tensor_map.block_tensor(0, "attn_q.weight").ggml_shape == (1024, 4096)
    assert tensor_map.tensor("nextn.pre_projection.weight").ggml_shape == (5632, 1024)
