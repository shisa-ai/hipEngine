"""Candidate plane transport preserves profile state on omission."""
from contextlib import ExitStack
from dataclasses import replace
import inspect
import os
from types import SimpleNamespace

import pytest
from scripts import qwen4exp_layer2_profile_gate as gate

KEY = 'HIPENGINE_QWEN4_EXP_Q4_IU8_PLANES'


@pytest.mark.parametrize('value', [None, 2, 3])
def test_candidate_plane_binding_and_restore(monkeypatch, value):
    monkeypatch.setenv(KEY, '2')
    spec = replace(gate.CANDIDATES['production_baseline'], q4_iu8_planes=value)
    owner = SimpleNamespace(q4_iu8_planes=3, runner=SimpleNamespace(q4_iu8_planes=2))
    with pytest.raises(RuntimeError, match='restore'):
        with ExitStack() as stack:
            gate.bind_candidate_q4_prefill(stack, owner, spec)
            assert owner.q4_iu8_planes == (3 if value is None else value)
            assert owner.runner.q4_iu8_planes == (2 if value is None else value)
            assert os.environ[KEY] == '2'
            raise RuntimeError('restore')
    assert owner.q4_iu8_planes == 3 and owner.runner.q4_iu8_planes == 2
    assert gate._candidate_named_profile_intact(spec) is (value is None)
    assert ('q4_iu8_planes' in gate.candidate_owner_changes(spec)) is (value is not None)
    with gate.candidate_q4_prefill_scope(owner, spec):
        assert owner.q4_iu8_planes == (3 if value is None else value)
    assert owner.q4_iu8_planes == 3


def test_candidate_paths_report_requested_and_effective_planes():
    common = inspect.getsource(gate.bind_candidate_owners)
    primary = inspect.getsource(gate._run)
    assert '"q4_iu8_planes": resolve_q4_iu8_planes(candidate_generator)' in common
    assert 'resolved_q4_planes = resolve_q4_iu8_planes(candidate_generator)' in primary
    assert '"q4_iu8_planes": resolved_q4_planes' in primary
    assert '"q4_iu8_planes_override": candidate_spec.q4_iu8_planes' in primary
