from __future__ import annotations

import pytest

from scripts.qwen38_dense_pair_perf import summarize_runs


def test_summarize_runs_uses_counterbalanced_pair_ratios() -> None:
    runs = [
        {"label": "strict", "pair": 1, "decode_tok_s": 10.0},
        {"label": "candidate", "pair": 1, "decode_tok_s": 11.0},
        {"label": "candidate", "pair": 2, "decode_tok_s": 9.0},
        {"label": "strict", "pair": 2, "decode_tok_s": 10.0},
        {"label": "strict", "pair": 3, "decode_tok_s": 8.0},
        {"label": "candidate", "pair": 3, "decode_tok_s": 8.0},
    ]

    summary = summarize_runs(runs)

    assert summary["candidate_over_strict"] == pytest.approx(0.9)
    assert summary["paired_ratios"] == pytest.approx([1.1, 0.9, 1.0])
    assert summary["paired_median"] == pytest.approx(1.0)
    assert summary["candidate_wins"] == 1
