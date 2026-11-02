"""Profile bindings have no process-environment writer or write history."""
import os
import pytest
from hipengine.execution_profiles import ExecutionProfile
from hipengine.generation import qwen4_exp_profiles as profiles
from tests.test_unit_qwen4exp_q4_dp4a_owner_transport import _resolve, _bind


@pytest.mark.parametrize('quant', profiles.QWEN4_EXP_QUANTS)
@pytest.mark.parametrize('profile', list(ExecutionProfile))
def test_profiles_do_not_write_environment(monkeypatch, quant, profile):
    resolved = _resolve(profile, quant=quant)
    before = dict(os.environ)
    def refuse(*args, **kwargs):
        pytest.fail('profile attempted to change process environment')
    with monkeypatch.context() as patch:
        patch.setattr(type(os.environ), '__setitem__', refuse)
        patch.setattr(type(os.environ), '__delitem__', refuse)
        owner, runner = _bind(resolved)
        assert profiles.selected_prefill_routes(owner) == profiles.selected_prefill_routes(runner)
    assert dict(os.environ) == before


def test_obsolete_process_writer_and_history_are_removed():
    assert not hasattr(profiles, '_apply_profile_env')
    assert not hasattr(profiles, '_binder_written')
    assert not hasattr(profiles, '_PROCESS_RESTORED_Q8_QSA_FLAGS')
