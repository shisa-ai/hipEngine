from __future__ import annotations

from scripts.paro_prefill_aotriton_stream_exactness import comparison_mismatches


def _leg(*, seed: int = 7, hidden: str = "hidden", conv: str = "conv", kv: str = "kv") -> dict:
    return {
        "seed_token_id": seed,
        "final_hidden_sha256": hidden,
        "state": {
            "live_count": 4,
            "linear": {"0": {"conv_sha256": conv, "recurrent_sha256": "recurrent"}},
            "full_kv": {"3": {"key_prefix_sha256": kv, "value_prefix_sha256": "value"}},
        },
    }


def test_comparison_mismatches_accepts_byte_identical_legs() -> None:
    assert comparison_mismatches(_leg(), _leg()) == []


def test_comparison_mismatches_names_output_and_state_differences() -> None:
    mismatches = comparison_mismatches(
        _leg(seed=8, hidden="changed", conv="changed", kv="changed"),
        _leg(),
    )

    assert mismatches == [
        "seed_token_id",
        "final_hidden_sha256",
        "state.linear.0.conv_sha256",
        "state.full_kv.3.key_prefix_sha256",
    ]
