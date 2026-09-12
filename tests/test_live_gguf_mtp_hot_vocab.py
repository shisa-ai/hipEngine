from __future__ import annotations

import base64
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hipengine.loading.gguf_mtp_hot_vocab import (
    HOT_VOCAB_KIND,
    default_gguf_hot_vocab_path,
    gguf_tokenizer_tokens_sha256,
    load_gguf_hot_vocab_selection,
)
from scripts.build_gguf_mtp_hot_vocab import _matches_script, _script_token_ids


def _info(vocab_size: int = 64):
    return SimpleNamespace(
        file_type_name="MOSTLY_Q4_K_M",
        metadata={
            "general.architecture": "qwen35",
            "general.basename": "fixture",
            "qwen35.block_count": 65,
            "tokenizer.ggml.tokens": [f"token-{index}" for index in range(vocab_size)],
        }
    )


def _payload(info, token_ids):
    return {
        "schema_version": 1,
        "kind": HOT_VOCAB_KIND,
        "model": {
            "architecture": info.metadata["general.architecture"],
            "basename": info.metadata["general.basename"],
            "block_count": info.metadata["qwen35.block_count"],
            "file_type": info.file_type_name,
            "vocab_size": len(info.metadata["tokenizer.ggml.tokens"]),
            "tokenizer_tokens_sha256": gguf_tokenizer_tokens_sha256(info),
        },
        "selection": {"strategy": "fixture"},
        "token_ids": token_ids,
    }


def test_cjk_selection_covers_han_hiragana_and_katakana() -> None:
    tokenizer = SimpleNamespace(
        tokens=("plain", "日本", "かな", "カナ", "한글"),
        decode=lambda ids: ("plain", "日本", "かな", "カナ", "한글")[ids[0]],
    )

    assert _script_token_ids(tokenizer, ("cjk",)) == {1, 2, 3}
    assert _matches_script("日本語", "cjk")
    assert not _matches_script("한국어", "cjk")


def test_packaged_default_is_exact_qwen38_model_identity() -> None:
    model = Path("/models/gguf/Qwen3.8-27B-Q4_K_M.gguf")
    if not model.exists():
        pytest.skip(f"local GGUF fixture not found: {model}")
    from hipengine.loading.gguf import GGUFReader

    path = default_gguf_hot_vocab_path(GGUFReader(model).info)

    assert path is not None
    assert path.name == "qwen38-27b-hot131072-cjk-v1.json"

    q4ks = Path("/models/gguf/Qwen3.8-27B-Q4_K_S.gguf")
    if q4ks.exists():
        assert default_gguf_hot_vocab_path(GGUFReader(q4ks).info) is None


def test_hot_vocab_selection_accepts_compact_bitmap_encoding(tmp_path) -> None:
    info = _info()
    payload = _payload(info, list(range(16)))
    payload.pop("token_ids")
    bitmap = bytearray(8)
    for token_id in range(0, 64, 4):
        bitmap[token_id // 8] |= 1 << (token_id % 8)
    payload["token_bitmap_base64"] = base64.b64encode(bitmap).decode("ascii")
    payload["selection"]["selected_tokens"] = 16
    path = tmp_path / "hot.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    selection = load_gguf_hot_vocab_selection(path, info)

    assert selection.token_ids == tuple(range(0, 64, 4))


def test_hot_vocab_selection_is_model_bound_and_sorted(tmp_path) -> None:
    info = _info()
    path = tmp_path / "hot.json"
    path.write_text(json.dumps(_payload(info, list(range(0, 64, 2)))), encoding="utf-8")

    selection = load_gguf_hot_vocab_selection(path, info)

    assert selection.token_ids == tuple(range(0, 64, 2))
    assert selection.size == 32
    assert selection.metadata == {"strategy": "fixture"}


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda payload: payload.update(token_ids=list(range(15))), "divisible by 16"),
        (lambda payload: payload.update(token_ids=[*range(15), 14]), "sorted and unique"),
        (lambda payload: payload.update(token_ids=[*range(15), 64]), "outside"),
        (
            lambda payload: payload["model"].update(tokenizer_tokens_sha256="0" * 64),
            "tokenizer hash",
        ),
        (
            lambda payload: payload["model"].update(basename="foreign"),
            "basename",
        ),
        (
            lambda payload: payload["model"].update(file_type="MOSTLY_Q4_K_S"),
            "file_type",
        ),
    ],
)
def test_hot_vocab_selection_rejects_invalid_or_foreign_maps(
    tmp_path,
    mutate,
    match: str,
) -> None:
    info = _info()
    payload = _payload(info, list(range(16)))
    mutate(payload)
    path = tmp_path / "hot.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        load_gguf_hot_vocab_selection(path, info)
