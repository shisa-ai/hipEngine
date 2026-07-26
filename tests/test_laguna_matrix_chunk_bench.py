from __future__ import annotations

from copy import deepcopy

import pytest

from scripts.laguna_matrix_chunk_bench import (
    LENGTHS,
    MATRIX_ROWS,
    _aggregate,
    _correctness,
    _decision,
    _mode_order,
)


def _rows() -> list[dict]:
    seconds = {
        128: {512: 10.0, 1024: 20.0, 4096: 80.0},
        256: {512: 9.0, 1024: 18.0, 4096: 72.0},
        512: {512: 8.0, 1024: 16.0, 4096: 64.0},
    }
    rows = []
    for repetition in range(2):
        for matrix_rows in MATRIX_ROWS:
            for length in LENGTHS:
                rows.append(
                    {
                        "matrix_rows": matrix_rows,
                        "length": length,
                        "repetition": repetition,
                        "prefill_seconds": seconds[matrix_rows][length],
                        "next_token_id": 17,
                        "next_token_logit_hex": "0x1.0p+0",
                        "logits_sha256": f"logits-{length}",
                        "final_hidden_sha256": f"hidden-{length}",
                        "post_layer_hidden_sha256": f"post-{length}",
                        "kv_sha256": f"kv-{length}",
                        "final_position": length - 1,
                    }
                )
    return rows


def test_laguna_matrix_chunk_order_rotates_all_policies() -> None:
    assert _mode_order(0, 0) == MATRIX_ROWS
    assert _mode_order(0, 1) == (256, 512, 128)
    assert _mode_order(1, 0) == (256, 512, 128)
    wide = (512, 1024, 2048)
    assert _mode_order(0, 0, matrix_rows=wide) == wide
    assert _mode_order(0, 1, matrix_rows=wide) == (1024, 2048, 512)


def test_laguna_matrix_chunk_gate_selects_fastest_exact_policy() -> None:
    rows = _rows()
    correctness = _correctness(rows)
    aggregate = _aggregate(rows)
    decision = _decision(aggregate, correctness, recovered=True)

    assert correctness["pass"] is True
    assert aggregate["512"]["speedup_vs_128"] == pytest.approx(1.25)
    assert decision["pass"] is True
    assert decision["selected_matrix_rows"] == 512
    assert decision["failed_checks"] == []


def test_laguna_matrix_chunk_gate_rejects_regressive_or_inexact_policy() -> None:
    rows = _rows()
    for row in rows:
        if row["matrix_rows"] == 512 and row["length"] == 4096:
            row["prefill_seconds"] = 88.0
    aggregate = _aggregate(rows)
    decision = _decision(aggregate, _correctness(rows), recovered=True)
    assert decision["selected_matrix_rows"] == 256

    mismatched = deepcopy(rows)
    next(
        row
        for row in mismatched
        if row["matrix_rows"] == 256 and row["length"] == 512
    )["kv_sha256"] = "changed"
    failed = _decision(_aggregate(mismatched), _correctness(mismatched), recovered=False)
    assert failed["pass"] is False
    assert "matrix_policy_outputs_or_state_not_exact" in failed["failed_checks"]
    assert "tracked_lifecycle_not_recovered" in failed["failed_checks"]
