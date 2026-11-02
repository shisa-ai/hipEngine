"""Ignored ambient overrides are reported on the bound owner, never globally."""
from types import SimpleNamespace
import os
import warnings

import pytest

from hipengine.execution_profiles import ExecutionProfile
from hipengine.generation import qwen4_exp_profiles as profiles
from hipengine.generation.qwen4_exp_gguf import _inherit_lane_decisions
from tests.test_unit_qwen4exp_q4_dp4a_owner_transport import _resolve, _bind

ENV = 'HIPENGINE_QWEN4_EXP_Q8_MMQ_PREPACK'


@pytest.mark.parametrize('order', [('strict', 'production'), ('production', 'strict')])
@pytest.mark.parametrize('quant', profiles.QWEN4_EXP_QUANTS)
def test_ignored_override_reports_belong_to_each_owner(monkeypatch, order, quant):
    monkeypatch.setenv(ENV, 'caller')
    owners = {}
    for profile in order:
        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            owner, runner = _bind(_resolve(ExecutionProfile(profile), quant=quant))
        expected = ('caller', '1' if owner.q8_mmq_prepack_enabled else '0')
        assert profiles.profile_ignored_overrides(owner)[ENV] == expected
        assert profiles.profile_ignored_overrides(runner)[ENV] == expected
        owners[profile] = owner
    for profile, owner in owners.items():
        assert profiles.profile_ignored_overrides(owner)[ENV] == (
            'caller', '1' if owner.q8_mmq_prepack_enabled else '0')
    assert os.environ[ENV] == 'caller'


def test_rebind_replaces_report_and_returned_snapshot_is_detached(monkeypatch):
    monkeypatch.setenv(ENV, 'caller')
    production = _resolve(ExecutionProfile.PRODUCTION)
    owner, runner = _bind(production)
    snapshot = profiles.profile_ignored_overrides(owner)
    snapshot.clear()
    assert ENV in profiles.profile_ignored_overrides(owner)
    monkeypatch.setenv(ENV, '1')
    production.binder(owner, production)
    assert ENV not in profiles.profile_ignored_overrides(owner)
    assert ENV not in profiles.profile_ignored_overrides(runner)
    assert profiles.profile_ignored_overrides(SimpleNamespace()) == {}


def test_recovery_reports_only_final_selection_and_late_runner_inherits(monkeypatch):
    env = profiles.PRODUCTION_MOE_PREFILL_ENV
    monkeypatch.setenv(env, '0')
    owner, _ = _bind(_resolve(ExecutionProfile.PRODUCTION, quant='gguf_ud_q4_k_xl'))
    assert env not in profiles.profile_ignored_overrides(owner)
    monkeypatch.setenv(env, '1')
    owner, _ = _bind(_resolve(ExecutionProfile.PRODUCTION, quant='gguf_ud_q4_k_xl'))
    assert profiles.profile_ignored_overrides(owner)[env] == ('1', '0')
    late = _inherit_lane_decisions(owner, SimpleNamespace())
    assert profiles.profile_ignored_overrides(late) == profiles.profile_ignored_overrides(owner)
    late._profile_ignored_overrides.clear()
    assert profiles.profile_ignored_overrides(owner)


def test_legacy_global_conflict_control_is_removed():
    for name in ('PREBINDER_CONFLICT_ENV', '_last_prebinder_conflicts',
                 '_owner_binder_discarded', 'last_prebinder_conflicts',
                 '_report_prebinder_conflicts'):
        assert not hasattr(profiles, name)
