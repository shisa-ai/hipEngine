from __future__ import annotations

import pytest

from scripts.q8t16_batch_route_perf import summarize_by_configuration


def test_summarize_by_configuration_keeps_matched_pair_ratios() -> None:
    runs = [
        {"configuration": "c4", "label": "strict", "pair": 1, "decode_tok_s": 100.0},
        {"configuration": "c4", "label": "candidate", "pair": 1, "decode_tok_s": 101.0},
        {"configuration": "c4", "label": "candidate", "pair": 2, "decode_tok_s": 98.0},
        {"configuration": "c4", "label": "strict", "pair": 2, "decode_tok_s": 100.0},
        {"configuration": "c8", "label": "strict", "pair": 1, "decode_tok_s": 150.0},
        {"configuration": "c8", "label": "candidate", "pair": 1, "decode_tok_s": 153.0},
        {"configuration": "c8", "label": "candidate", "pair": 2, "decode_tok_s": 147.0},
        {"configuration": "c8", "label": "strict", "pair": 2, "decode_tok_s": 150.0},
    ]

    summary = summarize_by_configuration(runs)

    assert summary["c4"]["candidate_over_strict"] == pytest.approx(0.995)
    assert summary["c4"]["paired_ratios"] == pytest.approx([1.01, 0.98])
    assert summary["c8"]["candidate_over_strict"] == pytest.approx(1.0)
    assert summary["c8"]["paired_ratios"] == pytest.approx([1.02, 0.98])
