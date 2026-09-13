from __future__ import annotations

import pytest

from hipengine.speculative.adaptive_budget import AdaptiveBudgetConfig, AdaptiveBudgetController


def test_disabled_controller_always_chooses_dflash_and_logs_profit() -> None:
    config = AdaptiveBudgetConfig(min_remaining_tokens_for_dflash=128)
    controller = AdaptiveBudgetController(enabled=False, ar_decode_tok_s_estimate=100.0, config=config)

    decision = controller.begin_cycle(cycle=1, context_tokens=128, remaining_tokens=16, active_budget=4)
    assert decision.use_dflash is True
    assert decision.reason == "disabled"

    controller.record_dflash_cycle(
        decision,
        visible_tokens=2,
        cycle_wall_ms=15.0,
        accepted_tokens=1,
        active_budget=4,
        draft_ms=5.0,
        verify_ms=9.0,
        commit_ms=1.0,
        context_tokens=128,
    )

    summary = controller.summary()
    assert summary["mode"] == "off"
    assert summary["transitions"] == []
    assert summary["cycle_log"][0]["profit_ms"] == pytest.approx(5.0)
    assert summary["cycle_log"][0]["mode_used"] == "DFLASH"


def test_enabled_controller_uses_ar_when_remaining_horizon_is_too_short() -> None:
    config = AdaptiveBudgetConfig(min_remaining_tokens_for_dflash=64)
    controller = AdaptiveBudgetController(enabled=True, ar_decode_tok_s_estimate=100.0, config=config)

    decision = controller.begin_cycle(cycle=1, context_tokens=128, remaining_tokens=32, active_budget=4)

    assert decision.use_dflash is False
    assert decision.reason == "remaining_tokens_guard"

    controller.record_ar_cycle(decision, cycle=1, cycle_wall_ms=10.0, context_tokens=128)
    summary = controller.summary()
    assert summary["state"] == "AR_PROBE"
    assert summary["cycle_log"][0]["mode_used"] == "AR"
    assert summary["cycle_log"][0]["decision_reason"] == "remaining_tokens_guard"
    assert summary["cycle_log"][0]["probe_remaining"] == 1


def test_remaining_horizon_guard_does_not_consume_cooldown() -> None:
    config = AdaptiveBudgetConfig(
        demote_after_cycles=1,
        demote_mean_profit_ms=-1.0,
        initial_cooldown_cycles=2,
        min_remaining_tokens_for_dflash=64,
        initial_probe_cycles=0,
    )
    controller = AdaptiveBudgetController(enabled=True, ar_decode_tok_s_estimate=100.0, config=config)

    decision = controller.begin_cycle(cycle=1, context_tokens=100, remaining_tokens=128, active_budget=4)
    controller.record_dflash_cycle(decision, visible_tokens=1, cycle_wall_ms=30.0, accepted_tokens=0, active_budget=4)
    assert controller.summary()["state"] == "AR_LOCKED"

    guarded = controller.begin_cycle(cycle=2, context_tokens=101, remaining_tokens=32, active_budget=4)
    assert guarded.reason == "remaining_tokens_guard"
    controller.record_ar_cycle(guarded, cycle=2, cycle_wall_ms=10.0, context_tokens=101, update_state=False)

    summary = controller.summary()
    assert summary["state"] == "AR_LOCKED"
    assert summary["cycle_log"][-1]["cooldown_remaining"] == 2
    assert summary["transitions"][-1]["to"] == "AR_LOCKED"


def test_remaining_horizon_guard_allows_long_prompts() -> None:
    config = AdaptiveBudgetConfig(min_remaining_tokens_for_dflash=64)
    controller = AdaptiveBudgetController(enabled=True, ar_decode_tok_s_estimate=100.0, config=config)

    decision = controller.begin_cycle(cycle=1, context_tokens=128, remaining_tokens=128, active_budget=4)

    assert decision.use_dflash is True
    assert decision.reason == "probe"


def test_probe_amortization_guard_blocks_unamortizable_probe() -> None:
    config = AdaptiveBudgetConfig(
        min_remaining_tokens_for_dflash=128,
        probe_min_amortization_tokens=64,
    )
    controller = AdaptiveBudgetController(enabled=True, ar_decode_tok_s_estimate=100.0, config=config)

    decision = controller.begin_cycle(cycle=1, context_tokens=128, remaining_tokens=160, active_budget=4)

    assert decision.use_dflash is False
    assert decision.reason == "probe_amortization_guard"
    assert decision.state == "AR_PROBE"


def test_enabled_controller_starts_with_bounded_negative_probe() -> None:
    config = AdaptiveBudgetConfig(
        demote_after_cycles=4,
        retry_cooldown_cycles=7,
        probe_cycles=1,
        promote_mean_profit_ms=0.5,
    )
    controller = AdaptiveBudgetController(enabled=True, ar_decode_tok_s_estimate=100.0, config=config)

    decision = controller.begin_cycle(cycle=1, context_tokens=128, remaining_tokens=160, active_budget=4)
    assert decision.use_dflash is True
    assert decision.reason == "probe"

    # Profit = 1 visible token * 10 ms/token - 70 ms = -60 ms.  The
    # startup probe should relock immediately rather than paying a four-cycle
    # DFlash demotion window.
    controller.record_dflash_cycle(decision, visible_tokens=1, cycle_wall_ms=70.0, accepted_tokens=0, active_budget=4)

    summary = controller.summary()
    assert summary["state"] == "AR_LOCKED"
    assert summary["mode_counts"] == {"dflash": 1}
    assert summary["transitions"] == [
        {
            "cycle": 1,
            "from": "AR_PROBE",
            "to": "AR_LOCKED",
            "reason": "probe_mean_profit_ms=-60.000 <= 0.500",
            "cooldown_remaining": 7,
            "probe_remaining": 0,
        }
    ]


def test_enabled_controller_demotes_after_negative_profit_window() -> None:
    config = AdaptiveBudgetConfig(
        demote_after_cycles=4,
        demote_mean_profit_ms=-5.0,
        initial_cooldown_cycles=3,
        retry_cooldown_cycles=5,
        probe_cycles=2,
        initial_probe_cycles=0,
    )
    controller = AdaptiveBudgetController(enabled=True, ar_decode_tok_s_estimate=100.0, config=config)

    # Profit = 1 visible token * 10 ms/token - 30 ms = -20 ms.
    for cycle in range(1, 5):
        decision = controller.begin_cycle(cycle=cycle, context_tokens=100 + cycle, remaining_tokens=16, active_budget=4)
        assert decision.use_dflash is True
        controller.record_dflash_cycle(
            decision,
            visible_tokens=1,
            cycle_wall_ms=30.0,
            accepted_tokens=0,
            active_budget=4,
            context_tokens=100 + cycle,
        )

    summary = controller.summary()
    assert summary["state"] == "AR_LOCKED"
    assert summary["transitions"] == [
        {
            "cycle": 4,
            "from": "DFLASH",
            "to": "AR_LOCKED",
            "reason": "mean_profit_last_4=-20.000 < -5.000",
            "cooldown_remaining": 3,
            "probe_remaining": 0,
        }
    ]

    locked_decision = controller.begin_cycle(cycle=5, context_tokens=105, remaining_tokens=12, active_budget=4)
    assert locked_decision.use_dflash is False
    controller.record_ar_cycle(locked_decision, cycle=5, cycle_wall_ms=10.0, context_tokens=105)
    assert controller.summary()["state"] == "AR_LOCKED"
    assert controller.summary()["cycle_log"][-1]["mode_used"] == "AR"


def test_cooldown_expires_into_probe_and_positive_probe_promotes() -> None:
    config = AdaptiveBudgetConfig(
        demote_after_cycles=1,
        demote_mean_profit_ms=-1.0,
        initial_cooldown_cycles=2,
        retry_cooldown_cycles=5,
        probe_cycles=2,
        promote_mean_profit_ms=0.5,
        initial_probe_cycles=0,
        probe_min_amortization_tokens=0,
    )
    controller = AdaptiveBudgetController(enabled=True, ar_decode_tok_s_estimate=100.0, config=config)

    decision = controller.begin_cycle(cycle=1, context_tokens=100, remaining_tokens=16, active_budget=4)
    controller.record_dflash_cycle(decision, visible_tokens=1, cycle_wall_ms=30.0, accepted_tokens=0, active_budget=4)
    assert controller.summary()["state"] == "AR_LOCKED"

    for cycle in (2, 3):
        decision = controller.begin_cycle(cycle=cycle, context_tokens=100 + cycle, remaining_tokens=16, active_budget=4)
        assert decision.use_dflash is False
        controller.record_ar_cycle(decision, cycle=cycle, cycle_wall_ms=10.0, context_tokens=100 + cycle)

    assert controller.summary()["state"] == "AR_PROBE"

    for cycle in (4, 5):
        decision = controller.begin_cycle(cycle=cycle, context_tokens=100 + cycle, remaining_tokens=16, active_budget=4)
        assert decision.use_dflash is True
        # Profit = 3 * 10 - 20 = +10 ms.
        controller.record_dflash_cycle(decision, visible_tokens=3, cycle_wall_ms=20.0, accepted_tokens=2, active_budget=4)

    summary = controller.summary()
    assert summary["state"] == "DFLASH"
    assert summary["transitions"][-1]["from"] == "AR_PROBE"
    assert summary["transitions"][-1]["to"] == "DFLASH"
    assert summary["mode_counts"] == {"ar": 2, "dflash": 3}


def test_negative_probe_relocks_with_retry_cooldown() -> None:
    config = AdaptiveBudgetConfig(
        demote_after_cycles=1,
        demote_mean_profit_ms=-1.0,
        initial_cooldown_cycles=1,
        retry_cooldown_cycles=7,
        probe_cycles=1,
        promote_mean_profit_ms=0.5,
        initial_probe_cycles=0,
        probe_min_amortization_tokens=0,
    )
    controller = AdaptiveBudgetController(enabled=True, ar_decode_tok_s_estimate=100.0, config=config)

    decision = controller.begin_cycle(cycle=1, context_tokens=100, remaining_tokens=16, active_budget=4)
    controller.record_dflash_cycle(decision, visible_tokens=1, cycle_wall_ms=30.0, accepted_tokens=0, active_budget=4)
    decision = controller.begin_cycle(cycle=2, context_tokens=101, remaining_tokens=16, active_budget=4)
    controller.record_ar_cycle(decision, cycle=2, cycle_wall_ms=10.0)
    assert controller.summary()["state"] == "AR_PROBE"

    decision = controller.begin_cycle(cycle=3, context_tokens=102, remaining_tokens=16, active_budget=4)
    controller.record_dflash_cycle(decision, visible_tokens=1, cycle_wall_ms=20.0, accepted_tokens=0, active_budget=4)

    summary = controller.summary()
    assert summary["state"] == "AR_LOCKED"
    assert summary["transitions"][-1]["to"] == "AR_LOCKED"
    assert summary["transitions"][-1]["cooldown_remaining"] == 7


def test_enabled_controller_requires_positive_ar_rate() -> None:
    with pytest.raises(ValueError, match="positive AR tok/s"):
        AdaptiveBudgetController(enabled=True, ar_decode_tok_s_estimate=None)
    with pytest.raises(ValueError, match="positive AR tok/s"):
        AdaptiveBudgetController(enabled=True, ar_decode_tok_s_estimate=0.0)


def test_config_rejects_negative_remaining_horizon_guard() -> None:
    with pytest.raises(ValueError, match="min_remaining_tokens_for_dflash"):
        AdaptiveBudgetConfig(min_remaining_tokens_for_dflash=-1)


def test_config_rejects_negative_initial_probe_cycles() -> None:
    with pytest.raises(ValueError, match="initial_probe_cycles"):
        AdaptiveBudgetConfig(initial_probe_cycles=-1)


def test_config_rejects_negative_probe_amortization_tokens() -> None:
    with pytest.raises(ValueError, match="probe_min_amortization_tokens"):
        AdaptiveBudgetConfig(probe_min_amortization_tokens=-1)
