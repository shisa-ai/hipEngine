"""Journey plane choices preserve snapshot overlays and phase ownership."""
from types import SimpleNamespace
import inspect
import os

import pytest
from scripts import qwen4exp_journey_localize as journey

KEY = journey.PREFIX + 'Q4_IU8_PLANES'


@pytest.mark.parametrize('production', [2, 3])
def test_plane_snapshot_overlay_and_restore(monkeypatch, production):
    flags = journey.q4_planes_snapshot({'OTHER': '1'}, published=3)
    arms = {'strict': dict(flags), 'bound': {}, 'explicit': {KEY: '2'}, 'declared': dict(flags)}
    values = journey.migrate_q4_planes_arms(arms, production_value=production,
                                           declared={'declared': 2})
    assert values == {'strict': 3, 'bound': production, 'explicit': 2, 'declared': 2}
    assert all(KEY not in env for env in arms.values())
    monkeypatch.setattr(journey, '_Q4_PLANES_ARM_VALUES', values)
    monkeypatch.setenv(KEY, '3')
    owner = SimpleNamespace(q4_iu8_planes=production,
                            runner=SimpleNamespace(q4_iu8_planes=None))
    with pytest.raises(RuntimeError, match='restore'):
        with journey.q4_planes_arm_scope(owner, 'explicit') as selected:
            assert selected == owner.q4_iu8_planes == owner.runner.q4_iu8_planes == 2
            raise RuntimeError('restore')
    assert owner.q4_iu8_planes == production and owner.runner.q4_iu8_planes is None
    assert os.environ[KEY] == '3'


@pytest.mark.parametrize('arm', ['strict_prefill_production_decode', 'production_prefill_strict_decode', 'bound'])
@pytest.mark.parametrize('phase', ['prefill', 'decode'])
def test_plane_split_restores_base(monkeypatch, arm, phase):
    monkeypatch.setattr(journey, '_Q4_PLANES_STRICT_VALUE', 3)
    owner = SimpleNamespace(q4_iu8_planes=2)
    with journey.q4_planes_split_scope(owner, arm, phase):
        strict = arm in journey.SPLIT_PHASE_ARMS and journey.split_phase_is_strict(arm, phase)
        assert owner.q4_iu8_planes == (3 if strict else 2)
    assert owner.q4_iu8_planes == 2


def test_plane_family_overlays_keep_snapshot_meaning():
    strict = journey.q4_planes_snapshot({}, published=3)
    production = journey.q4_planes_snapshot({}, published=2)
    arms = {'q8_selected_down_strict': {}}
    journey.snapshot_arms(strict, production, arms=arms)
    values = journey.migrate_q4_planes_arms(arms, production_value=2)
    assert values['strict_plus_matrix'] == 2
    assert values['strict_plus_dense'] == 3
    assert values['q8down_off_without_matrix'] == 3
    assert values['all_flags_strict'] == 3


def test_journey_wiring_and_other_consumer_reports():
    from scripts import qwen4exp_prefill_restore_ab as restore
    from scripts import qwen4exp_conservative_cost as cost
    from scripts import qwen4exp_long_context_gate as long
    source = inspect.getsource(journey.main)
    assert 'migrate_q4_planes_arms(' in source and 'q4_planes_arm_scope(' in source
    assert 'q4_iu8_planes_phases' in source
    assert '"q4_iu8_planes": effective_q4_planes' in source
    assert 'key != _Q4_PLANES_ENV' in source
    assert source.count('q4_planes_snapshot(') == 2
    assert inspect.getsource(journey.split_trajectory).count('q4_planes_split_scope(') == 2
    assert 'q4_iu8_planes' in restore.SHARED_DECODE_SAFE_OWNER_FIELDS
    assert "row['q4_iu8_planes'] = resolve_q4_iu8_planes(runner)" in inspect.getsource(restore)
    assert "row['q4_iu8_planes'] = resolve_q4_iu8_planes(runner)" in inspect.getsource(cost)
    assert '"q4_iu8_planes": candidate_owners["q4_iu8_planes"]' in inspect.getsource(long)
