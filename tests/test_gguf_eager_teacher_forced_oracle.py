from __future__ import annotations

from copy import deepcopy

from scripts.gguf_eager_teacher_forced_oracle import (
    OracleError,
    _compare_checkpoint,
    _parse_token_ids,
    _validate_external_trajectory,
)


def _buffer(digest: str) -> dict[str, object]:
    return {
        "nbytes": 16,
        "blake2b_128": digest,
        "finite": True,
        "rms": 1.0,
        "max_abs": 2.0,
    }


def _checkpoint() -> dict[str, object]:
    return {
        "position": 513,
        "current_token_id": 9707,
        "predicted_token_id": 9707,
        "hidden_seed": _buffer("hidden"),
        "layer_outputs": [
            {"layer": 0, "fingerprint": _buffer("layer-0")},
            {"layer": 1, "fingerprint": _buffer("layer-1")},
        ],
        "linear_states": [
            {
                "layer": 1,
                "conv": _buffer("conv-1"),
                "recurrent": _buffer("recurrent-1"),
            }
        ],
        "kv_states": [
            {
                "layer": 0,
                "live_positions": 513,
                "key": _buffer("key-0"),
                "value": _buffer("value-0"),
            }
        ],
    }


def test_compare_checkpoint_accepts_exact_teacher_forced_state() -> None:
    eager = _checkpoint()
    reference = deepcopy(eager)

    result = _compare_checkpoint(
        eager,
        reference,
        expected_predicted_token_id=9707,
    )

    assert result == {
        "passed": True,
        "token_match_external": True,
        "state_exact": True,
        "mismatches": [],
        "first_divergence": None,
    }


def test_compare_checkpoint_localizes_first_layer_and_state_divergence() -> None:
    eager = _checkpoint()
    reference = deepcopy(eager)
    eager["layer_outputs"][1]["fingerprint"]["blake2b_128"] = "bad-layer-1"
    eager["linear_states"][0]["recurrent"]["blake2b_128"] = "bad-recurrent-1"
    eager["kv_states"][0]["value"]["blake2b_128"] = "bad-kv-0"

    result = _compare_checkpoint(
        eager,
        reference,
        expected_predicted_token_id=9707,
    )

    assert result["passed"] is False
    assert result["state_exact"] is False
    assert result["first_divergence"] == {
        "component": "full_attention_kv",
        "layer": 0,
        "part": "value",
    }
    assert [row["component"] for row in result["mismatches"]] == [
        "full_attention_kv",
        "layer_output",
        "linear_state",
    ]


def test_compare_checkpoint_rejects_external_token_mismatch_without_state_drift() -> None:
    eager = _checkpoint()
    reference = deepcopy(eager)
    eager["predicted_token_id"] = 11
    reference["predicted_token_id"] = 11

    result = _compare_checkpoint(
        eager,
        reference,
        expected_predicted_token_id=9707,
    )

    assert result["passed"] is False
    assert result["token_match_external"] is False
    assert result["state_exact"] is True
    assert result["first_divergence"] == {
        "component": "external_token",
        "layer": None,
        "part": None,
    }


def test_external_trajectory_requires_exact_prompt_and_enough_generated_ids() -> None:
    result = _validate_external_trajectory(
        expected_prompt_ids=[9707] * 8,
        actual_prompt_ids=[9707] * 8,
        generated_token_ids=[9707] * 5,
        decode_steps=4,
    )

    assert result["passed"] is True
    assert result["required_generated_tokens"] == 5
    assert result["generated_token_ids"] == [9707] * 5

    try:
        _validate_external_trajectory(
            expected_prompt_ids=[9707] * 8,
            actual_prompt_ids=[9707] * 7,
            generated_token_ids=[9707] * 5,
            decode_steps=4,
        )
    except OracleError as exc:
        assert "prompt token IDs differ" in str(exc)
    else:
        raise AssertionError("prompt mismatch was accepted")


def test_parse_token_ids_is_strict() -> None:
    assert _parse_token_ids("[11, 271, 40, 1044]", label="test") == [
        11,
        271,
        40,
        1044,
    ]

    for value in ("not-json", "[1, true]", "[]", '{"token": 1}'):
        try:
            _parse_token_ids(value, label="test")
        except OracleError:
            pass
        else:
            raise AssertionError(f"invalid token payload was accepted: {value!r}")
