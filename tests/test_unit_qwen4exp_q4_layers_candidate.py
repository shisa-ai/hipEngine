"""Candidate Q4 layer overrides preserve omission and explicit empty scopes."""
from contextlib import ExitStack
from dataclasses import replace
import inspect
import os
from types import SimpleNamespace
import pytest
from scripts import qwen4exp_layer2_profile_gate as gate

KEY = 'HIPENGINE_QWEN4_EXP_Q4_IU8_LAYERS'


@pytest.mark.parametrize('layers', [None, '', 'all', '2,17-19', '999'])
def test_candidate_layer_binding_and_restore(monkeypatch, layers):
    monkeypatch.setenv(KEY, 'ambient')
    spec = replace(gate.CANDIDATES['production_baseline'], q4_iu8_layers=layers)
    owner = SimpleNamespace(q4_iu8_layers='owner', runner=SimpleNamespace(q4_iu8_layers='runner'))
    with pytest.raises(RuntimeError, match='restore'):
        with ExitStack() as stack:
            gate.bind_candidate_q4_prefill(stack, owner, spec)
            assert owner.q4_iu8_layers == ('owner' if layers is None else layers)
            assert owner.runner.q4_iu8_layers == ('runner' if layers is None else layers)
            assert os.environ[KEY] == 'ambient'
            raise RuntimeError('restore')
    assert owner.q4_iu8_layers == 'owner'
    assert owner.runner.q4_iu8_layers == 'runner'
    assert gate._candidate_named_profile_intact(spec) is (layers is None)
    changes = gate.candidate_owner_changes(spec)
    assert ('q4_iu8_layers' in changes) is (layers is not None)


def test_candidate_paths_report_effective_layers_and_requested_scope():
    common = inspect.getsource(gate.bind_candidate_owners)
    primary = inspect.getsource(gate._run)
    assert '"q4_iu8_layers": resolve_q4_iu8_layers(candidate_generator)' in common
    assert 'resolved_q4_layers = resolve_q4_iu8_layers(candidate_generator)' in primary
    assert '"q4_iu8_layers": resolved_q4_layers' in primary
    assert '"q4_iu8_layers_override": candidate_spec.q4_iu8_layers' in primary
