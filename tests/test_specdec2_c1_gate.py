from __future__ import annotations

import pytest

from scripts.specdec2_c1_gate import _choice_ids, _choice_rows


def test_specdec2_c1_gate_reads_exact_choice_ids() -> None:
    payload = {
        "choices": [
            {"hipengine": {"generated_token_ids": [271, 9419, 0, 2500]}}
        ]
    }

    assert _choice_ids(payload) == (271, 9419, 0, 2500)


def test_specdec2_gate_reads_multiple_exact_choice_rows() -> None:
    payload = {
        "choices": [
            {"hipengine": {"generated_token_ids": [1, 2]}},
            {"hipengine": {"generated_token_ids": [3, 4]}},
        ]
    }

    assert _choice_rows(payload) == ((1, 2), (3, 4))


def test_specdec2_c1_gate_rejects_missing_exact_ids() -> None:
    with pytest.raises(KeyError):
        _choice_ids({"choices": [{}]})
