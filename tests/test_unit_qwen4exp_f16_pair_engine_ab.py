from __future__ import annotations

import json

from scripts.qwen4exp_f16_pair_engine_ab import (
    ARM_SEQUENCE,
    CATEGORIES,
    HELDOUTS,
    _arm_order,
    _expected_pairs,
    _load_jsonl,
)


def test_dp2_arm_order_is_counterbalanced():
    assert _arm_order(0) == ARM_SEQUENCE
    assert _arm_order(1) == tuple(reversed(ARM_SEQUENCE))
    assert _arm_order(2) == ARM_SEQUENCE
    assert _arm_order(0).count("candidate") == 3
    assert _arm_order(1).count("parent") == 3


def test_dp2_pair_admission_only_qualified_rows():
    assert _expected_pairs(512) == 0
    assert _expected_pairs(1024) == 61
    assert _expected_pairs(4096) == 61
    assert _expected_pairs(1025) == 0


def test_dp2_heldout_suite_is_eight_rows_and_covers_categories():
    rows = _load_jsonl(HELDOUTS)
    assert len(rows) == 8
    assert {str(row["category"]) for row in rows} == set(CATEGORIES)
    assert len({str(row["id"]) for row in rows}) == 8
    for row in rows:
        assert isinstance(row["messages"], list)
        assert row["messages"]
