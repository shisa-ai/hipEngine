from __future__ import annotations

import pytest

from hipengine.speculative.mtp_budget import (
    MtpAdaptiveBudgetConfig,
    MtpAdaptiveBudgetPolicy,
    MtpBudgetCycleResult,
    MtpBudgetSequencePolicy,
)


def _record(
    policy: MtpAdaptiveBudgetPolicy,
    *,
    cycle: int,
    budget: int,
    accepted: int,
    wall_ms: float,
) -> None:
    policy.record_cycle(
        MtpBudgetCycleResult(
            cycle=cycle,
            budget=budget,
            accepted_count=accepted,
            visible_tokens=accepted + 1,
            cycle_wall_ms=wall_ms,
            full_accept=accepted == budget,
        )
    )


def test_sequence_policy_emits_transition_matrix_without_prompt_data() -> None:
    policy = MtpBudgetSequencePolicy((1, 1, 2, 1, 3, 2, 2, 3, 3, 1))
    policy.start_request(request_id=7, max_budget=3)

    observed = tuple(
        policy.choose_budget(cycle=index + 1, max_budget=3, remaining_decode=32)
        for index in range(10)
    )

    assert observed == (1, 1, 2, 1, 3, 2, 2, 3, 3, 1)
    assert set(zip(observed, observed[1:], strict=False)) == {
        (1, 1),
        (1, 2),
        (1, 3),
        (2, 1),
        (2, 2),
        (2, 3),
        (3, 1),
        (3, 2),
        (3, 3),
    }
    assert "prompt" not in policy.summary()


def test_budget_policy_is_request_owned_and_fails_closed() -> None:
    policy = MtpBudgetSequencePolicy((1, 2, 3))
    policy.start_request(request_id=1, max_budget=3)
    with pytest.raises(RuntimeError, match="already owns request"):
        policy.start_request(request_id=2, max_budget=3)
    assert policy.choose_budget(cycle=2, max_budget=1, remaining_decode=1) == 1


def test_adaptive_policy_explores_each_independent_bucket_then_scores() -> None:
    policy = MtpAdaptiveBudgetPolicy(
        MtpAdaptiveBudgetConfig(
            budgets=(1, 2, 3),
            ema_alpha=1.0,
            switch_margin=0.0,
            exploration_samples_per_budget=1,
        )
    )
    policy.start_request(request_id=4, max_budget=3)

    assert policy.choose_budget(cycle=1, max_budget=3, remaining_decode=32) == 1
    _record(policy, cycle=1, budget=1, accepted=1, wall_ms=2.0)
    assert policy.choose_budget(cycle=2, max_budget=3, remaining_decode=32) == 2
    _record(policy, cycle=2, budget=2, accepted=2, wall_ms=2.0)
    assert policy.choose_budget(cycle=3, max_budget=3, remaining_decode=32) == 3
    _record(policy, cycle=3, budget=3, accepted=3, wall_ms=2.0)

    assert policy.choose_budget(cycle=4, max_budget=3, remaining_decode=32) == 3
    summary = policy.summary()
    assert summary["decision_counts"] == {"B1": 1, "B2": 1, "B3": 2}
    assert summary["cycle_count"] == 3
    assert summary["conditional_acceptance"] == [1.0, 1.0, 1.0]


def test_adaptive_policy_uses_conditional_acceptance_and_hysteresis() -> None:
    policy = MtpAdaptiveBudgetPolicy(
        MtpAdaptiveBudgetConfig(
            budgets=(1, 2, 3),
            ema_alpha=1.0,
            switch_margin=0.10,
            exploration_samples_per_budget=1,
        )
    )
    policy.start_request(request_id=9, max_budget=3)
    for cycle, (budget, accepted, wall_ms) in enumerate(
        ((1, 1, 2.0), (2, 1, 2.0), (3, 1, 2.0)),
        start=1,
    ):
        assert policy.choose_budget(
            cycle=cycle,
            max_budget=3,
            remaining_decode=32,
        ) == budget
        _record(
            policy,
            cycle=cycle,
            budget=budget,
            accepted=accepted,
            wall_ms=wall_ms,
        )

    # p(depth1)=1, p(depth2)=0, so all budgets predict two visible tokens.
    # Stable tie order keeps the incumbent B3 rather than flapping.
    assert policy.choose_budget(cycle=4, max_budget=3, remaining_decode=32) == 3
    summary = policy.summary()
    assert summary["conditional_acceptance"] == [1.0, 0.0, None]
    assert summary["scores"]["B1"] == pytest.approx(1.0)
    assert summary["scores"]["B2"] == pytest.approx(1.0)
    assert summary["scores"]["B3"] == pytest.approx(1.0)


def test_adaptive_policy_tail_selects_only_a_qualified_budget() -> None:
    policy = MtpAdaptiveBudgetPolicy()
    policy.start_request(request_id=3, max_budget=3)

    assert policy.choose_budget(cycle=1, max_budget=1, remaining_decode=1) == 1
    _record(policy, cycle=1, budget=1, accepted=0, wall_ms=1.0)
    assert policy.choose_budget(cycle=2, max_budget=2, remaining_decode=2) == 2
