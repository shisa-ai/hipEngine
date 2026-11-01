"""Journey selected-down snapshots and phases bind the measured owner."""
from types import SimpleNamespace
import inspect
import os
import pytest
from scripts import qwen4exp_journey_localize as journey

ENV = 'HIPENGINE_QWEN4_EXP_Q8_0_SELECTED_WMMA_DOWN'


@pytest.mark.parametrize('production', [False, True])
def test_selected_snapshot_families(monkeypatch, production):
    monkeypatch.setenv(ENV, 'stale')
    strict = journey.selected_down_snapshot({ENV: 'stale'}, published=False)
    current = journey.selected_down_snapshot({}, published=production)
    assert strict == {ENV: '0'}
    arms = {'bound': {}, 'q8_selected_down_strict': {ENV: '0'},
            'candidate': {ENV: '0'}}
    journey.snapshot_arms(strict, current, arms=arms)
    values = journey.migrate_selected_down_arms(
        arms, production_value=production, declared={'candidate': True})
    assert values['bound'] is production
    assert values['all_flags_strict'] is False
    assert values['strict_plus_matrix'] is production
    assert values['strict_plus_qsa'] is False
    assert values['q8down_off_without_qsa'] is False
    assert values['q8down_off_without_matrix'] is False
    assert values['candidate'] is True
    assert all(ENV not in env for env in arms.values())
    assert ENV not in journey.strict_environment({ENV: 'stale'})
    monkeypatch.setattr(journey, '_SELECTED_DOWN_ARM_VALUES', values)
    owner = SimpleNamespace(q8_selected_down_enabled=production,
        runner=SimpleNamespace(q8_selected_down_enabled=production))
    with pytest.raises(RuntimeError, match='restore'):
        with journey.selected_down_arm_scope(owner, 'candidate') as enabled:
            assert enabled is True
            assert owner.q8_selected_down_enabled is True
            assert owner.runner.q8_selected_down_enabled is True
            assert os.environ[ENV] == 'stale'
            raise RuntimeError('restore')
    assert owner.q8_selected_down_enabled is production
    assert owner.runner.q8_selected_down_enabled is production


@pytest.mark.parametrize('arm', list(journey.SPLIT_PHASE_ARMS))
@pytest.mark.parametrize('phase', ['prefill', 'decode'])
@pytest.mark.parametrize('strict', [False, True])
def test_selected_split_phases(monkeypatch, arm, phase, strict):
    monkeypatch.setattr(journey, '_SELECTED_DOWN_STRICT_VALUE', strict)
    owner = SimpleNamespace(q8_selected_down_enabled=not strict)
    with journey.selected_down_split_scope(owner, arm, phase):
        assert owner.q8_selected_down_enabled is (
            strict if journey.split_phase_is_strict(arm, phase) else not strict)
    assert owner.q8_selected_down_enabled is (not strict)


def test_selected_journey_wiring():
    source = inspect.getsource(journey.main)
    assert 'selected_down_arm_scope(generator, arm)' in source
    assert 'migrate_selected_down_arms(' in source
    assert 'selected_down_snapshot(' in source
    assert 'q8_selected_down_enabled_phases' in source
    source = inspect.getsource(journey.split_trajectory)
    for phase in ['prefill', 'decode']:
        assert f"selected_down_split_scope(runner, arm, '{phase}')" in source
