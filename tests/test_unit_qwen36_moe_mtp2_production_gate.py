from __future__ import annotations

from scripts.qwen36_moe_mtp2_production_gate import (
    numerical_verdict,
    paired_task_verdict,
    summarize_rows,
)


def _row(*, category: str = "code", shape: str = "k2", transition: str = "verify_to_verify"):
    return {
        "kl": 1.0e-5,
        "top1_equal": True,
        "top5_overlap": 1.0,
        "max_abs_logit_delta": 0.01,
        "category": category,
        "shape": shape,
        "transition": transition,
    }


def test_numerical_verdict_binds_global_and_each_scope() -> None:
    rows = [
        _row(category="code", shape="k2", transition="prefill_to_verify"),
        _row(category="general_en", shape="k1"),
        _row(category="general_ja", shape="k2"),
        _row(category="mixed_ja_en", shape="k2"),
    ]

    result = numerical_verdict(rows)

    assert result["passed"] is True
    assert result["aggregate"]["rows"] == 4
    assert set(result["scopes"]["category"]) == {
        "code",
        "general_en",
        "general_ja",
        "mixed_ja_en",
    }

    rows[0] = {**rows[0], "kl": 0.051}
    failed = numerical_verdict(rows)
    assert failed["passed"] is False
    assert "category:code" in failed["scope_failures"]


def test_summarize_rows_reports_tail_and_top1() -> None:
    rows = [_row(), {**_row(), "kl": 0.01, "top1_equal": False}]

    summary = summarize_rows(rows)

    assert summary["rows"] == 2
    assert summary["max_kl"] == 0.01
    assert summary["top1_agreement"] == 0.5
    assert summary["top1_mismatches"] == 1


def test_paired_task_gate_rejects_lost_strict_feature() -> None:
    strict = "- MTP improves 推論 throughput while KV cache remains visible."
    candidate = "- throughput improves."

    result = paired_task_verdict("mixed_ja_en_review", strict, candidate)

    assert result["passed"] is False
    assert "mtp_preserved" in result["regressions"]


def test_paired_task_gate_allows_different_noninferior_code_text() -> None:
    strict = "def merge_intervals(xs):\n    return xs\n"
    candidate = "def merge_intervals(intervals):\n    return sorted(intervals)\n"

    result = paired_task_verdict("code_merge_intervals", strict, candidate)

    assert result["passed"] is True
    assert result["regressions"] == []
