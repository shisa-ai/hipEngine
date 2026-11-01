"""Journey flash snapshots preserve both axes without process transport."""
from types import SimpleNamespace
import pytest
from scripts import qwen4exp_journey_localize as journey
from hipengine.runtime.qwen4_exp_runner import QSAFlashPrefillPolicy as Policy

E = 'HIPENGINE_QWEN4_EXP_QSA_FLASH_PREFILL'
L = 'HIPENGINE_QWEN4_EXP_QSA_FLASH_LAYERS'


def test_flash_snapshot_migration(monkeypatch):
    strict, production = Policy(False, ''), Policy(True, '35-47')
    monkeypatch.setenv(E, 'stale')
    monkeypatch.setenv(L, '2')
    snapshot = journey.flash_snapshot({E: 'stale', L: '2', 'OTHER': 'x'}, published=strict)
    assert snapshot == {E: '0', L: '', 'OTHER': 'x'}
    arms = {'strict': dict(snapshot), 'production': {}, 'layer_only': {L: '2'},
            'enable_only': {E: '0'}, 'declared': {E: '1'}}
    values = journey.migrate_flash_arms(arms, production_value=production,
                                      declared={'declared': False})
    assert values == {'strict': strict, 'production': production,
                      'layer_only': Policy(True, '2'), 'enable_only': Policy(False, '35-47'),
                      'declared': Policy(False, '35-47')}
    assert all(E not in env and L not in env for env in arms.values())
    monkeypatch.setattr(journey, '_FLASH_ARM_VALUES', values)
    owner = SimpleNamespace(qsa_flash_prefill_policy=production)
    with pytest.raises(RuntimeError, match='restore'):
        with journey.flash_arm_scope(owner, 'strict') as policy:
            assert policy == strict
            assert owner.qsa_flash_prefill_policy == strict
            raise RuntimeError('restore')
    assert owner.qsa_flash_prefill_policy is production


@pytest.mark.parametrize('production_enabled', [False, True])
def test_flash_family_snapshot_semantics(monkeypatch, production_enabled):
    strict, production = Policy(False, ''), Policy(production_enabled, '35-47')
    arms = {'q8_selected_down_strict': {}}
    journey.snapshot_arms(journey.flash_snapshot({}, published=strict),
                          journey.flash_snapshot({}, published=production), arms=arms)
    values = journey.migrate_flash_arms(arms, production_value=production)
    assert values['all_flags_strict'] == strict
    assert values['strict_plus_qsa'] == production
    assert values['strict_plus_matrix'] == strict
    assert values['q8down_off_without_qsa'] == strict
    assert values['q8down_off_without_dense'] == production
    assert journey.strict_environment({E: '1', L: '2'}) == {}


@pytest.mark.parametrize('arm', list(journey.SPLIT_PHASE_ARMS))
@pytest.mark.parametrize('phase', ['prefill', 'decode'])
def test_flash_phase(arm, phase, monkeypatch):
    prior = Policy(True, '35-47')
    owner = SimpleNamespace(qsa_flash_prefill_policy=prior)
    monkeypatch.setattr(journey, '_FLASH_STRICT_POLICY', Policy(False, ''))
    with journey.flash_split_scope(owner, arm, phase):
        assert owner.qsa_flash_prefill_policy == (
            Policy(False, '') if journey.split_phase_is_strict(arm, phase) else prior)
    assert owner.qsa_flash_prefill_policy is prior
