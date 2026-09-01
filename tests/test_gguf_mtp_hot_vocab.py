from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from hipengine.loading.gguf_mtp_hot_vocab import (
    HOT_VOCAB_KIND,
    gguf_tokenizer_tokens_sha256,
    load_gguf_hot_vocab_selection,
)
from scripts.build_gguf_mtp_hot_vocab import _matches_script, _script_token_ids


def _info(vocab_size: int = 64):
    return SimpleNamespace(
        metadata={"tokenizer.ggml.tokens": [f"token-{index}" for index in range(vocab_size)]}
    )


def _payload(info, token_ids):
    return {
        "schema_version": 1,
        "kind": HOT_VOCAB_KIND,
        "model": {
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
