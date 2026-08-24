from __future__ import annotations

import pytest

from scripts.specdec2_c1_gate import _choice_ids


def test_specdec2_c1_gate_reads_exact_choice_ids() -> None:
    payload = {
        "choices": [
            {"hipengine": {"generated_token_ids": [271, 9419, 0, 2500]}}
        ]
    }

    assert _choice_ids(payload) == (271, 9419, 0, 2500)


def test_specdec2_c1_gate_rejects_missing_exact_ids() -> None:
    with pytest.raises(KeyError):
        _choice_ids({"choices": [{}]})
