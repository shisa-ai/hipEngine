from __future__ import annotations

import pytest

from scripts.compare_prompt_token_inventories import compare_prompt_token_inventories
from scripts.llamacpp_gguf_prompt_token_inventory import (
    ServerTokenizeError,
    build_llamacpp_prompt_token_inventory,
    extract_token_ids_and_pieces,
)


def test_compare_prompt_token_inventories_reports_match_and_mismatch() -> None:
    left = {
        "kind": "hipengine_gguf_prompt_token_inventory",
        "prompts": [
            {"name": "a", "token_ids": [1, 2, 3], "token_ids_sha256": "left-a", "rendered_sha256": "text-a"},
            {"name": "b", "token_ids": [4, 5], "token_ids_sha256": "left-b", "rendered_sha256": "text-b"},
        ],
    }
    right = {
        "kind": "llamacpp_prompt_token_inventory",
        "prompts": [
            {"name": "a", "token_ids": [1, 2, 3], "token_ids_sha256": "right-a", "rendered_sha256": "text-a"},
            {"name": "b", "token_ids": [4, 7], "token_ids_sha256": "right-b", "rendered_sha256": "text-b"},
            {"name": "c", "token_ids": [8], "token_ids_sha256": "right-c", "rendered_sha256": "text-c"},
        ],
    }

    comparison = compare_prompt_token_inventories(
        left,
        right,
        left_label="hipengine",
        right_label="llamacpp",
        context_tokens=1,
    )

    assert not comparison["all_match"]
    assert comparison["compared_prompts"] == 2
    assert comparison["matched_prompts"] == ["a"]
    assert comparison["missing_in_left"] == ["c"]
    assert comparison["missing_in_right"] == []
    assert comparison["mismatches"] == [
        {
            "name": "b",
            "left_token_count": 2,
            "right_token_count": 2,
            "left_token_ids_sha256": "left-b",
            "right_token_ids_sha256": "right-b",
            "rendered_sha256_match": True,
            "first_mismatch_index": 1,
            "left_token_id": 5,
            "right_token_id": 7,
            "left_window": [4, 5],
            "right_window": [4, 7],
        }
    ]


def test_llamacpp_prompt_token_inventory_accepts_token_ids_and_pieces() -> None:
    responses = {
        "alpha": {
            "tokens": [
                {"id": 11, "piece": "a"},
                {"id": 12, "piece": [195]},
            ]
        }
    }

    inventory = build_llamacpp_prompt_token_inventory(
        prompts=[{"name": "p0", "prompt": "alpha"}],
        server_url="http://127.0.0.1:8080",
        tokenizer=responses.__getitem__,
        prompts_file="synthetic-prompts.json",
        model="model.gguf",
    )

    assert inventory["kind"] == "llamacpp_prompt_token_inventory"
    assert inventory["tokenization"] == "llamacpp.server.tokenize"
    assert inventory["model"] == "model.gguf"
    row = inventory["prompts"][0]
    assert row["name"] == "p0"
    assert row["token_ids"] == [11, 12]
    assert row["token_pieces"] == ["a", [195]]


def test_llamacpp_tokenize_response_accepts_plain_ids() -> None:
    token_ids, pieces = extract_token_ids_and_pieces({"tokens": [1, 2, 3]})

    assert token_ids == [1, 2, 3]
    assert pieces is None


def test_llamacpp_tokenize_response_rejects_invalid_tokens() -> None:
    with pytest.raises(ServerTokenizeError, match="invalid token entry"):
        extract_token_ids_and_pieces({"tokens": [{"piece": "missing id"}]})


def test_compare_prompt_token_inventories_passes_identical_ids() -> None:
    inventory = {
        "kind": "hipengine_gguf_prompt_token_inventory",
        "prompts": [
            {"name": "a", "token_ids": [1, 2, 3], "token_ids_sha256": "hash-a", "rendered_sha256": "text-a"},
        ],
    }

    comparison = compare_prompt_token_inventories(inventory, inventory)

    assert comparison["all_match"]
    assert comparison["matched_prompts"] == ["a"]
    assert comparison["mismatches"] == []
