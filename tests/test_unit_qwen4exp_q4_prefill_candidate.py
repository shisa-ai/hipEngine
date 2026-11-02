"""Layer-profile candidates can bind plain Q4 prefill on their owner."""
from contextlib import ExitStack
from dataclasses import replace
import inspect
import os
from types import SimpleNamespace
import pytest
from scripts import qwen4exp_layer2_profile_gate as gate

KEY = 'HIPENGINE_QWEN4_EXP_Q4_IU8_PREFILL'


@pytest.mark.parametrize('enabled', [None, False, True])
def test_candidate_plain_prefill_preserves_omission_and_restores(monkeypatch, enabled):
    monkeypatch.setenv(KEY, '1')
    spec = replace(gate.CANDIDATES['production_baseline'], q4_iu8_prefill_enabled=enabled)
    runner = SimpleNamespace(q4_iu8_prefill_enabled=None)
    owner = SimpleNamespace(q4_iu8_prefill_enabled=False, runner=runner)
    with pytest.raises(RuntimeError):
        with ExitStack() as stack:
            assert gate.bind_candidate_q4_prefill(stack, owner, spec) is (False if enabled is None else enabled)
            assert owner.q4_iu8_prefill_enabled is (False if enabled is None else enabled)
            assert runner.q4_iu8_prefill_enabled is enabled
            assert os.environ[KEY] == '1'
            raise RuntimeError('restore')
    assert owner.q4_iu8_prefill_enabled is False
    assert runner.q4_iu8_prefill_enabled is None
    assert gate._candidate_named_profile_intact(spec) is (enabled is None)


def test_candidate_common_and_primary_paths_report_plain_prefill():
    source = inspect.getsource(gate.bind_candidate_owners)
    assert 'bind_candidate_q4_prefill(' in source
    assert '"q4_iu8_prefill_enabled"' in source
    source = inspect.getsource(gate._run)
    assert 'bind_candidate_q4_prefill(' in source
    assert '"q4_iu8_prefill_enabled": resolved_q4_prefill' in source
    assert '"q4_iu8_prefill_override": candidate_spec.q4_iu8_prefill_enabled' in source
