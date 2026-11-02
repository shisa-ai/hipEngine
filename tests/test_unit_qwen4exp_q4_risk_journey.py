"""Journey multipliers preserve snapshot overlays and split-phase values."""
from types import SimpleNamespace
import inspect
import os

import pytest
from scripts import qwen4exp_journey_localize as journey

KEY = journey.PREFIX + 'Q4_IU8_RISK_MULT'


@pytest.mark.parametrize('production', [0.5, 1.0, 4.0])
def test_multiplier_snapshot_overlay_and_restore(monkeypatch, production):
    flags = journey.q4_risk_snapshot({'OTHER': '1'}, published=4.0)
    arms = {'strict': dict(flags), 'bound': {}, 'explicit': {KEY: '0.5'}, 'declared': dict(flags)}
    values = journey.migrate_q4_risk_arms(arms, production_value=production,
                                         declared={'declared': 1.0})
    assert values == {'strict': 4.0, 'bound': production, 'explicit': 0.5, 'declared': 1.0}
    assert all(KEY not in env for env in arms.values())
    monkeypatch.setattr(journey, '_Q4_RISK_ARM_VALUES', values)
    monkeypatch.setenv(KEY, '8')
    owner = SimpleNamespace(q4_iu8_risk_mult=production,
                            runner=SimpleNamespace(q4_iu8_risk_mult=None))
    with pytest.raises(RuntimeError, match='restore'):
        with journey.q4_risk_arm_scope(owner, 'explicit') as selected:
            assert selected == owner.q4_iu8_risk_mult == owner.runner.q4_iu8_risk_mult == 0.5
            raise RuntimeError('restore')
    assert owner.q4_iu8_risk_mult == production and owner.runner.q4_iu8_risk_mult is None
    assert os.environ[KEY] == '8'


@pytest.mark.parametrize('arm', ['strict_prefill_production_decode', 'production_prefill_strict_decode', 'bound'])
@pytest.mark.parametrize('phase', ['prefill', 'decode'])
def test_multiplier_split_restores_base(monkeypatch, arm, phase):
    monkeypatch.setattr(journey, '_Q4_RISK_STRICT_VALUE', 4.0)
    owner = SimpleNamespace(q4_iu8_risk_mult=1.0)
    with journey.q4_risk_split_scope(owner, arm, phase):
        strict = arm in journey.SPLIT_PHASE_ARMS and journey.split_phase_is_strict(arm, phase)
        assert owner.q4_iu8_risk_mult == (4.0 if strict else 1.0)
    assert owner.q4_iu8_risk_mult == 1.0


def test_multiplier_family_overlays_keep_snapshot_meaning():
    strict = journey.q4_risk_snapshot({}, published=4.0)
    production = journey.q4_risk_snapshot({}, published=1.0)
    arms = {'q8_selected_down_strict': {}}
    journey.snapshot_arms(strict, production, arms=arms)
    values = journey.migrate_q4_risk_arms(arms, production_value=1.0)
    assert values['strict_plus_matrix'] == 1.0
    assert values['strict_plus_dense'] == 4.0
    assert values['q8down_off_without_matrix'] == 4.0
    assert values['all_flags_strict'] == 4.0


def test_journey_wiring_and_other_consumer_reports():
    from scripts import qwen4exp_prefill_restore_ab as restore
    from scripts import qwen4exp_conservative_cost as cost
    from scripts import qwen4exp_long_context_gate as long
    source = inspect.getsource(journey.main)
    assert 'migrate_q4_risk_arms(' in source and 'q4_risk_arm_scope(' in source
    assert 'q4_iu8_risk_mult_phases' in source
    assert '"q4_iu8_risk_mult": effective_q4_risk' in source
    assert 'key != _Q4_RISK_ENV' in source
    assert source.count('q4_risk_snapshot(') == 2
    assert inspect.getsource(journey.split_trajectory).count('q4_risk_split_scope(') == 2
    assert 'q4_iu8_risk_mult' in restore.SHARED_DECODE_SAFE_OWNER_FIELDS
    assert "row['q4_iu8_risk_mult'] = resolve_q4_iu8_risk_mult(runner)" in inspect.getsource(restore)
    assert "row['q4_iu8_risk_mult'] = resolve_q4_iu8_risk_mult(runner)" in inspect.getsource(cost)
    assert '"q4_iu8_risk_mult": candidate_owners["q4_iu8_risk_mult"]' in inspect.getsource(long)
