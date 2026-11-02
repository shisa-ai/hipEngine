"""Profile-gap plain Q4 prefill overrides use owner state."""
from contextlib import ExitStack
import inspect
import os
from types import SimpleNamespace
import pytest
from scripts import qwen4exp_profile_gap as gap

ENV = 'HIPENGINE_QWEN4_EXP_Q4_IU8_PREFILL'


@pytest.mark.parametrize('raw,expected', [('0', False), ('1', True), ('FALSE', True)])
def test_plain_prefill_override_is_owner_local(monkeypatch, raw, expected):
    ambient = '0' if expected else '1'
    monkeypatch.setenv(ENV, ambient)
    runner = SimpleNamespace(q4_iu8_prefill_enabled=None)
    owner = SimpleNamespace(runner=runner, q4_iu8_prefill_enabled=False)
    peer = SimpleNamespace(q4_iu8_prefill_enabled=None)
    with pytest.raises(RuntimeError):
        with ExitStack() as stack:
            report = gap.bind_q4_iu8_prefill_override(stack, owner, {ENV: raw})
            assert report == {'q4_iu8_prefill_override': expected,
                              'q4_iu8_prefill_enabled': expected}
            assert owner.q4_iu8_prefill_enabled is runner.q4_iu8_prefill_enabled is expected
            gap._apply_post_binder_overrides({ENV: raw})
            assert os.environ[ENV] == ambient
            assert peer.q4_iu8_prefill_enabled is None
            raise RuntimeError('restore')
    assert owner.q4_iu8_prefill_enabled is False
    assert runner.q4_iu8_prefill_enabled is None


def test_plain_prefill_omission_keeps_bound_owner(monkeypatch):
    monkeypatch.setenv(ENV, '1')
    owner = SimpleNamespace(q4_iu8_prefill_enabled=False)
    with ExitStack() as stack:
        assert gap.bind_q4_iu8_prefill_override(stack, owner, {}) == {
            'q4_iu8_prefill_override': None, 'q4_iu8_prefill_enabled': False}
    assert 'report.update(bind_q4_iu8_prefill_override(' in inspect.getsource(gap.main)
