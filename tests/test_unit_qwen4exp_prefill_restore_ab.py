from collections import Counter

import pytest

from scripts.qwen4exp_layer2_profile_gate import CANDIDATES
from scripts.qwen4exp_prefill_restore_ab import round_order, validate_shared_graphs


def test_shared_graphs_allow_only_reviewed_uncaptured_switches():
    validate_shared_graphs([
        CANDIDATES["q8_blockscale_guarded_quad"].environment,
        CANDIDATES["q8_blockscale_guarded_ordered"].environment,
    ])
    with pytest.raises(ValueError, match="captured decode"):
        validate_shared_graphs([{"HIPENGINE_QWEN4_EXP_Q4_DP4A64": "1"}])


def test_three_arm_order_balances_positions_and_neighbor_directions():
    arms = ("before", "quad", "ordered")
    permutations = []
    for case_index in (0, 1):
        orders = [round_order(case_index, arms, repetition) for repetition in range(3)]
        for position in range(3):
            assert Counter(order[position] for order in orders) == Counter(arms)
        permutations.extend(tuple(order) for order in orders)
    assert len(set(permutations)) == 6


def test_two_arm_order_preserves_counterbalance():
    arms = ("before", "after")
    assert [round_order(0, arms, i) for i in range(3)] == [
        ["before", "after"], ["after", "before"], ["before", "after"]]
