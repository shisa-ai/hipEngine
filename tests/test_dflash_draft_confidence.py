from __future__ import annotations

import pytest

from scripts.dflash_chain_e2e_bench import (
    _confidence_limited_active_count,
    _top1_probabilities_from_topk,
    _top1_probability_from_topk,
)


def test_top1_probability_from_topk_is_stable_softmax() -> None:
    probability = _top1_probability_from_topk((1000.0, 999.0, 998.0))

    assert probability == pytest.approx(0.66524096)


def test_top1_probability_from_single_logit_defaults_to_one() -> None:
    assert _top1_probability_from_topk((42.0,)) == 1.0


def test_confidence_limited_active_count_stops_at_first_low_confidence() -> None:
    probabilities = _top1_probabilities_from_topk(
        (
            (4.0, 0.0),
            (3.0, 0.0),
            (0.1, 0.0),
            (5.0, 0.0),
        )
    )

    assert _confidence_limited_active_count(probabilities, max_active=4, p_min=0.70) == 2


def test_confidence_limited_active_count_disabled_keeps_budget() -> None:
    probabilities = (0.01, 0.02)

    assert _confidence_limited_active_count(probabilities, max_active=4, p_min=0.0) == 4
