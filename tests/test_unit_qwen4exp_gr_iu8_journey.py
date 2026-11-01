"""Journey GR pairs follow snapshot overlays and split-phase ownership."""
from contextlib import ExitStack
from types import SimpleNamespace
import inspect
import os
import pytest
from hipengine.runtime.qwen4_exp_runner import GRIU8Policy
from scripts import qwen4exp_journey_localize as journey


@pytest.mark.parametrize('up,down', [(False, False), (True, False), (False, True), (True, True)])
def test_journey_gr_snapshot_overlay_and_declaration(monkeypatch, up, down):
    strict = GRIU8Policy(False, False)
    production = GRIU8Policy(up, down)
    flags = journey.gr_iu8_snapshot({'OTHER': '1'}, published=strict)
    arms = {'strict': dict(flags), 'bound': {}, 'up': {journey._GR_IU8_KEYS[0]: '1'},
            'declared': dict(flags)}
    values = journey.migrate_gr_iu8_arms(arms, production_value=production,
                                        declared={'declared': (None, True)})
    assert values == {'strict': strict, 'bound': production,
                      'up': GRIU8Policy(True, down), 'declared': GRIU8Policy(False, True)}
    assert arms == {'strict': {'OTHER': '1'}, 'bound': {}, 'up': {}, 'declared': {'OTHER': '1'}}
    monkeypatch.setattr(journey, '_GR_IU8_ARM_VALUES', values)
    monkeypatch.setenv(journey._GR_IU8_KEYS[0], 'ambient-up')
    monkeypatch.setenv(journey._GR_IU8_KEYS[1], 'ambient-down')
    owner = SimpleNamespace(gr_iu8_policy=production,
                            runner=SimpleNamespace(gr_iu8_policy=None))
    with pytest.raises(RuntimeError, match='restore'):
        with journey.gr_iu8_arm_scope(owner, 'strict') as policy:
            assert policy == strict
            assert owner.gr_iu8_policy is owner.runner.gr_iu8_policy is policy
            raise RuntimeError('restore')
    assert owner.gr_iu8_policy is production and owner.runner.gr_iu8_policy is None
    assert os.environ[journey._GR_IU8_KEYS[0]] == 'ambient-up'
    assert os.environ[journey._GR_IU8_KEYS[1]] == 'ambient-down'


@pytest.mark.parametrize('arm', ['strict_prefill_production_decode', 'production_prefill_strict_decode'])
@pytest.mark.parametrize('phase', ['prefill', 'decode'])
def test_journey_gr_split_restores_published_pair(monkeypatch, arm, phase):
    strict, production = GRIU8Policy(False, False), GRIU8Policy(True, False)
    monkeypatch.setattr(journey, '_GR_IU8_STRICT_POLICY', strict)
    owner = SimpleNamespace(gr_iu8_policy=production)
    with journey.gr_iu8_split_scope(owner, arm, phase):
        assert owner.gr_iu8_policy == (strict if journey.split_phase_is_strict(arm, phase) else production)
    assert owner.gr_iu8_policy is production


def test_journey_gr_family_overlays_keep_both_axes():
    strict = journey.gr_iu8_snapshot({}, published=GRIU8Policy(False, False))
    production = journey.gr_iu8_snapshot({}, published=GRIU8Policy(True, True))
    arms = {'q8_selected_down_strict': {}}
    journey.snapshot_arms(strict, production, arms=arms)
    values = journey.migrate_gr_iu8_arms(arms, production_value=GRIU8Policy(True, True))
    assert values['strict_plus_dense'] == GRIU8Policy(True, True)
    assert values['strict_plus_matrix'] == GRIU8Policy(False, False)
    assert values['strict_plus_gr_iu8'] == GRIU8Policy(True, False)
    assert values['strict_plus_gr_iu8_down'] == GRIU8Policy(False, True)
    assert values['q8down_off_without_dense'] == GRIU8Policy(False, False)


def test_journey_gr_wiring():
    source = inspect.getsource(journey.main)
    assert 'migrate_gr_iu8_arms(' in source and 'gr_iu8_arm_scope(' in source
    assert 'gr_iu8_policy_phases' in source
    assert inspect.getsource(journey.split_trajectory).count('gr_iu8_split_scope(') == 2
