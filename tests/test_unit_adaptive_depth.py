"""Unit tests for acceptance-triggered depth adaptation (CPU only).

Synthetic cycle economies pin the policy: climb on sustained high acceptance
when deeper cycles yield more tokens/ms, descend on low acceptance when
shallower cycles are faster, stay put in the mid-band, respect bounds and
cooldowns, and remain static when disabled.
"""

from __future__ import annotations

import pytest

from hipengine.speculative.adaptive_depth import AdaptiveDepthConfig, AdaptiveDepthController


def _run(controller, economy, cycles):
    """economy(budget) -> (visible_tokens, cycle_ms, accepted, proposed)."""
    for _ in range(cycles):
        budget = controller.begin_cycle()
        visible, ms, accepted, proposed = economy(budget)
        controller.record_cycle(
            budget=budget, visible_tokens=visible, cycle_wall_ms=ms,
            accepted_tokens=accepted, proposed_tokens=proposed,
        )
    return controller


def test_disabled_is_static():
    c = AdaptiveDepthController(enabled=False, initial_budget=4)
    _run(c, lambda b: (b + 1, 10.0, b, b), 100)
    assert c.current_budget == 4
    assert c.summary()["decisions"] == []


def test_high_acceptance_climbs_when_deeper_is_faster():
    # Deeper budgets commit more tokens per cycle at sub-linear cost:
    # tokens/ms rises with budget; acceptance stays high.
    def economy(b):
        visible = b + 1
        ms = 8.0 + 1.0 * b
        return visible, ms, b, b

    c = AdaptiveDepthController(enabled=True, initial_budget=4,
                                config=AdaptiveDepthConfig(decision_interval=4, probe_cycles=4, probe_cooldown=4))
    _run(c, economy, 200)
    assert c.current_budget == 7, c.summary()


def test_low_acceptance_descends_when_shallower_is_faster():
    # Acceptance collapses with depth; shallow cycles emit tokens faster.
    def economy(b):
        accepted = max(0, 3 - b)
        visible = accepted + 1
        ms = 5.0 + 2.0 * b
        return visible, ms, accepted, b

    c = AdaptiveDepthController(enabled=True, initial_budget=6,
                                config=AdaptiveDepthConfig(decision_interval=4, probe_cycles=4, probe_cooldown=4))
    _run(c, economy, 200)
    assert c.current_budget == 2, c.summary()


def test_mid_band_acceptance_stays_put():
    # Acceptance ~0.43 (the natural-suite average): no probe direction fires.
    def economy(b):
        accepted = int(0.43 * b)
        return accepted + 1, 10.0 + b, accepted, b

    c = AdaptiveDepthController(enabled=True, initial_budget=4,
                                config=AdaptiveDepthConfig(decision_interval=4, probe_cycles=4))
    _run(c, economy, 200)
    assert c.current_budget == 4
    assert all(d["action"] != "probe" for d in c.summary()["decisions"])


def test_probe_reverts_when_neighbor_is_worse_and_cooldown_blocks_flapping():
    # High acceptance triggers up-probes, but deeper cycles are slower;
    # the controller must revert and not probe every decision.
    def economy(b):
        visible = b + 1
        ms = 5.0 + 5.0 * b
        return visible, ms, b, b

    c = AdaptiveDepthController(enabled=True, initial_budget=4,
                                config=AdaptiveDepthConfig(decision_interval=4, probe_cycles=4, probe_cooldown=32))
    _run(c, economy, 200)
    assert c.current_budget == 4
    probes = [d for d in c.summary()["decisions"] if d["action"] == "probe"]
    reverts = [d for d in c.summary()["decisions"] if d["action"] == "revert"]
    assert probes and len(reverts) == len(probes)
    # cooldown 32 >> interval 4 keeps probes sparse
    assert len(probes) <= 200 / 32 + 2


def test_bounds_respect_block_cap():
    c = AdaptiveDepthController(enabled=True, initial_budget=9, max_budget_cap=5)
    assert c.initial_budget == 5
    assert c.max_budget == 5
    _run(c, lambda b: (b + 1, 10.0, b, b), 100)
    assert c.current_budget == 5
    c2 = AdaptiveDepthController(enabled=True, initial_budget=1, max_budget_cap=7)
    assert c2.initial_budget == 2  # clamped to min_budget


def test_record_validation():
    c = AdaptiveDepthController(enabled=True, initial_budget=4)
    with pytest.raises(ValueError):
        c.record_cycle(budget=4, visible_tokens=0, cycle_wall_ms=1.0, accepted_tokens=0, proposed_tokens=4)
    with pytest.raises(ValueError):
        c.record_cycle(budget=4, visible_tokens=1, cycle_wall_ms=-1.0, accepted_tokens=0, proposed_tokens=4)
    with pytest.raises(ValueError):
        c.record_cycle(budget=4, visible_tokens=1, cycle_wall_ms=1.0, accepted_tokens=0, proposed_tokens=0)


def test_config_validation():
    with pytest.raises(ValueError):
        AdaptiveDepthConfig(min_budget=0)
    with pytest.raises(ValueError):
        AdaptiveDepthConfig(min_budget=5, max_budget=4)
    with pytest.raises(ValueError):
        AdaptiveDepthConfig(low_acceptance=0.9, high_acceptance=0.2)
    with pytest.raises(ValueError):
        AdaptiveDepthConfig(decision_interval=0)