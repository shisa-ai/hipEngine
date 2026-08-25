from __future__ import annotations

import pytest

from scripts.specdec2_c1_gate import (
    _choice_ids,
    _choice_rows,
    _physical_staged_row_passes,
    _provider_fingerprint_verdicts,
)


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


def test_specdec2_physical_gate_requires_device_resident_candidates() -> None:
    row = {
        "specdec2_mtp2_proposal_batch_calls": 3,
        "specdec2_mtp2_target_batch_calls": 3,
        "specdec2_mtp2_candidate_device_handoffs": 3,
        "specdec2_mtp2_candidate_d2h_after_target": 0,
        "specdec2_mtp2_device_accept_calls": 3,
        "specdec2_mtp2_selected_commit_batch_calls": 3,
        "specdec2_mtp2_execution_routes": ["eager"] * 3,
    }

    assert _physical_staged_row_passes(row)
    row["specdec2_mtp2_candidate_d2h_after_target"] = 3
    assert not _physical_staged_row_passes(row)


def test_provider_fingerprint_gate_compares_same_slot_repeats() -> None:
    slot0 = {"kv": "slot0"}
    slot1 = {"kv": "slot1"}
    direct = {"kv": "direct"}

    repeat_equal, direct_equal = _provider_fingerprint_verdicts(
        (slot0, slot1, slot0.copy(), slot1.copy(), direct, direct.copy()),
        concurrency=2,
    )

    assert repeat_equal
    assert not direct_equal
    assert _provider_fingerprint_verdicts(
        (slot0, slot1, slot0.copy(), slot1.copy(), slot0.copy(), slot1.copy()),
        concurrency=2,
    ) == (True, True)
