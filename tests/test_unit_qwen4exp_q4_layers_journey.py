"""Journey layer scopes follow snapshots, overlays and split-phase ownership."""
from types import SimpleNamespace
import inspect
import os

import pytest
from scripts import qwen4exp_journey_localize as journey

KEY = journey.PREFIX + 'Q4_IU8_LAYERS'


@pytest.mark.parametrize('production', ['', 'all', '35-47'])
def test_layer_snapshot_overlay_declaration_and_restore(monkeypatch, production):
    flags = journey.q4_layers_snapshot({'OTHER': '1'}, published='999')
    arms = {'strict': dict(flags), 'bound': {}, 'empty': {KEY: ''}, 'declared': dict(flags)}
    values = journey.migrate_q4_layers_arms(arms, production_value=production,
                                          declared={'declared': 'all'})
    assert values == {'strict': '999', 'bound': production, 'empty': '', 'declared': 'all'}
    assert arms == {'strict': {'OTHER': '1'}, 'bound': {}, 'empty': {}, 'declared': {'OTHER': '1'}}
    monkeypatch.setattr(journey, '_Q4_LAYERS_ARM_VALUES', values)
    monkeypatch.setenv(KEY, 'ambient')
    owner = SimpleNamespace(q4_iu8_layers=production,
                            runner=SimpleNamespace(q4_iu8_layers=None))
    with pytest.raises(RuntimeError, match='restore'):
        with journey.q4_layers_arm_scope(owner, 'empty') as selected:
            assert selected == owner.q4_iu8_layers == owner.runner.q4_iu8_layers == ''
            raise RuntimeError('restore')
    assert owner.q4_iu8_layers == production
    assert owner.runner.q4_iu8_layers is None
    assert os.environ[KEY] == 'ambient'


@pytest.mark.parametrize('arm', ['strict_prefill_production_decode', 'production_prefill_strict_decode', 'bound'])
@pytest.mark.parametrize('phase', ['prefill', 'decode'])
def test_layer_split_restores_base(monkeypatch, arm, phase):
    monkeypatch.setattr(journey, '_Q4_LAYERS_STRICT_VALUE', '999')
    owner = SimpleNamespace(q4_iu8_layers='all')
    with journey.q4_layers_split_scope(owner, arm, phase):
        strict = arm in journey.SPLIT_PHASE_ARMS and journey.split_phase_is_strict(arm, phase)
        assert owner.q4_iu8_layers == ('999' if strict else 'all')
    assert owner.q4_iu8_layers == 'all'


def test_layer_family_overlays_keep_snapshot_meaning():
    strict = journey.q4_layers_snapshot({}, published='999')
    production = journey.q4_layers_snapshot({}, published='35-47')
    arms = {'q8_selected_down_strict': {}}
    journey.snapshot_arms(strict, production, arms=arms)
    values = journey.migrate_q4_layers_arms(arms, production_value='35-47')
    assert values['strict_plus_matrix'] == '35-47'
    assert values['strict_plus_dense'] == '999'
    assert values['q8down_off_without_matrix'] == '999'
    assert values['q8down_off_without_qsa'] == '35-47'
    assert values['all_flags_strict'] == '999'
    assert all(KEY not in env for env in arms.values())


def test_journey_layer_wiring_and_phase_reports():
    source = inspect.getsource(journey.main)
    assert 'migrate_q4_layers_arms(' in source and 'q4_layers_arm_scope(' in source
    assert 'q4_iu8_layers_phases' in source
    assert '"q4_iu8_layers": effective_q4_layers' in source
    assert 'key != _Q4_LAYERS_ENV' in source
    assert source.count('q4_layers_snapshot(') == 2
    assert inspect.getsource(journey.split_trajectory).count('q4_layers_split_scope(') == 2
