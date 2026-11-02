"""Journey plain-Q4 choices follow snapshots, family overlays and phases."""
from types import SimpleNamespace
import inspect
import os
import pytest
from scripts import qwen4exp_journey_localize as journey

KEY = journey.PREFIX + 'Q4_IU8_PREFILL'


@pytest.mark.parametrize('production', [False, True])
def test_q4_snapshot_overlay_declaration_and_restore(monkeypatch, production):
    flags = journey.q4_prefill_snapshot({'OTHER': '1'}, published=False)
    arms = {'strict': dict(flags), 'bound': {}, 'on': {KEY: '1'}, 'declared': dict(flags)}
    values = journey.migrate_q4_prefill_arms(arms, production_value=production,
                                           declared={'declared': True})
    assert values == {'strict': False, 'bound': production, 'on': True, 'declared': True}
    assert arms == {'strict': {'OTHER': '1'}, 'bound': {}, 'on': {}, 'declared': {'OTHER': '1'}}
    monkeypatch.setattr(journey, '_Q4_PREFILL_ARM_VALUES', values)
    monkeypatch.setenv(KEY, 'ambient')
    owner = SimpleNamespace(q4_iu8_prefill_enabled=production,
                            runner=SimpleNamespace(q4_iu8_prefill_enabled=None))
    with pytest.raises(RuntimeError, match='restore'):
        with journey.q4_prefill_arm_scope(owner, 'on') as selected:
            assert selected is True
            assert owner.q4_iu8_prefill_enabled is owner.runner.q4_iu8_prefill_enabled is True
            raise RuntimeError('restore')
    assert owner.q4_iu8_prefill_enabled is production
    assert owner.runner.q4_iu8_prefill_enabled is None
    assert os.environ[KEY] == 'ambient'


@pytest.mark.parametrize('arm', ['strict_prefill_production_decode', 'production_prefill_strict_decode'])
@pytest.mark.parametrize('phase', ['prefill', 'decode'])
def test_q4_split_restores_base(monkeypatch, arm, phase):
    monkeypatch.setattr(journey, '_Q4_PREFILL_STRICT_VALUE', False)
    owner = SimpleNamespace(q4_iu8_prefill_enabled=True)
    with journey.q4_prefill_split_scope(owner, arm, phase):
        assert owner.q4_iu8_prefill_enabled is (not journey.split_phase_is_strict(arm, phase))
    assert owner.q4_iu8_prefill_enabled is True


def test_q4_family_overlays_keep_snapshot_meaning():
    strict = journey.q4_prefill_snapshot({}, published=False)
    production = journey.q4_prefill_snapshot({}, published=True)
    arms = {'q8_selected_down_strict': {}}
    journey.snapshot_arms(strict, production, arms=arms)
    values = journey.migrate_q4_prefill_arms(arms, production_value=True)
    assert values['strict_plus_matrix'] is True
    assert values['strict_plus_dense'] is False
    assert values['q8down_off_without_matrix'] is False
    assert values['q8down_off_without_qsa'] is True
    assert values['all_flags_strict'] is False
    assert all(KEY not in env for env in arms.values())


def test_split_trajectory_measures_phase_choice_and_restores_on_failure(monkeypatch):
    import numpy as np
    from contextlib import nullcontext
    monkeypatch.setattr(journey, '_Q4_PREFILL_STRICT_VALUE', False)
    monkeypatch.setattr(journey, 'clear_arm_graphs', lambda runner: None)
    monkeypatch.setattr(journey, 'set_flags', lambda flags: None)
    for name in ('gr_iu8_split_scope', 'selected_down_split_scope', 'flash_split_scope',
                 'generic_wmma_baseline_scope', 'phase_peer_scope', 'qsa_h256_prefill_mode',
                 'gdn_register_prefill', 'q4_pair_prefill', 'q8_wave_scale_prefill',
                 'gr_wave_scale', 'gr_dpp_split_scope', 'q4_dp4a_split_scope',
                 'q8_mmq_split_scope', 'production_moe_split_scope', 'gr_wide_split_scope',
                 'gdn_colwarps_split_scope', 'qsa_ordered_split_scope'):
        monkeypatch.setattr(journey, name, lambda *a, **k: nullcontext())
    observed = []
    owner = SimpleNamespace(q4_iu8_prefill_enabled=True, reset=lambda: None)
    def prefill(ids):
        observed.append(owner.q4_iu8_prefill_enabled)
        return SimpleNamespace(logits=np.zeros(2))
    def step(token):
        observed.append(owner.q4_iu8_prefill_enabled)
        raise RuntimeError('step failed')
    owner.prefill, owner.step = prefill, step
    for arm, expected in [('strict_prefill_production_decode', [False, True]),
                          ('production_prefill_strict_decode', [True, False])]:
        observed.clear()
        with pytest.raises(RuntimeError, match='step failed'):
            journey.split_trajectory(owner, [1], [2], {}, {}, arm=arm)
        assert observed == expected
        assert owner.q4_iu8_prefill_enabled is True


def test_journey_q4_wiring_and_phase_reports():
    source = inspect.getsource(journey.main)
    assert 'migrate_q4_prefill_arms(' in source and 'q4_prefill_arm_scope(' in source
    assert 'q4_iu8_prefill_enabled_phases' in source
    assert '"q4_iu8_prefill_enabled": effective_q4_prefill' in source
    assert 'and key != _Q4_PREFILL_ENV' in source
    assert inspect.getsource(journey.split_trajectory).count('q4_prefill_split_scope(') == 2
