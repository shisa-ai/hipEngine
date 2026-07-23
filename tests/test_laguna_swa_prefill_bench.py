from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

from scripts.laguna_swa_prefill_bench import (
    LENGTHS,
    MODES,
    _aggregate,
    _correctness,
    _mode_order,
    _promotion_gate,
    _resident_weight_nbytes,
)


def _rows():
    rows = []
    for length in LENGTHS:
        for repetition in range(2):
            for mode in MODES:
                candidate = mode == "wave32_exact"
                digest = f"{length}"
                rows.append(
                    {
                        "length": length,
                        "mode": mode,
                        "repetition": repetition,
                        "prefill_seconds": 1.0 if candidate else 2.0,
                        "next_token_id": length,
                        "next_token_logit_hex": "0x1.0p+0",
                        "logits_sha256": digest,
                        "final_hidden_sha256": digest,
                        "post_layer_hidden_sha256": digest,
                        "final_position": length - 1,
                    }
                )
    return rows


def test_lpf5_swa_reads_the_resident_weight_contract() -> None:
    assert _resident_weight_nbytes(SimpleNamespace(resident_nbytes=123)) == 123
    with pytest.raises(AttributeError):
        _resident_weight_nbytes(SimpleNamespace(nbytes=123))


def test_lpf5_swa_mode_order_balances_lengths_and_repetitions() -> None:
    for index in range(len(LENGTHS)):
        assert _mode_order(index, 1) == tuple(reversed(_mode_order(index, 0)))


def test_lpf5_swa_gate_accepts_exact_faster_candidate() -> None:
    rows = _rows()
    correctness = _correctness(rows)
    aggregate = _aggregate(rows)
    gate = _promotion_gate(aggregate, correctness, recovered=True)

    assert correctness["pass"] is True
    assert all(
        aggregate["wave32_vs_baseline"][str(length)]["speedup"] == 2.0
        for length in LENGTHS
    )
    assert gate["pass"] is True
    assert gate["failed_checks"] == []


def test_lpf5_swa_gate_rejects_hash_mismatch_or_short_speedup() -> None:
    mismatch = deepcopy(_rows())
    candidate = next(row for row in mismatch if row["mode"] == "wave32_exact")
    candidate["final_hidden_sha256"] = "bad"
    correctness = _correctness(mismatch)
    gate = _promotion_gate(_aggregate(mismatch), correctness, recovered=True)
    assert correctness["pass"] is False
    assert "full_model_outputs_not_exact" in gate["failed_checks"]

    slow = deepcopy(_rows())
    for row in slow:
        if row["mode"] == "wave32_exact" and row["length"] == 4096:
            row["prefill_seconds"] = 1.95
    slow_gate = _promotion_gate(
        _aggregate(slow),
        _correctness(slow),
        recovered=True,
    )
    assert slow_gate["pass"] is False
    assert "length_4096_speedup_not_above_1.05" in slow_gate["failed_checks"]
