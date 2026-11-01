"""Layer2 selected-down gates are candidate owner declarations."""
from contextlib import ExitStack
from dataclasses import replace
import inspect
import os
from types import SimpleNamespace
import pytest
from scripts import qwen4exp_layer2_profile_gate as gate

ENV = 'HIPENGINE_QWEN4_EXP_Q8_0_SELECTED_WMMA_DOWN'


def test_candidates_declare_selected_down_not_process_input():
    expected = {'q8_blockscale': True, 'production_conservative': False,
        'production_q8_fallback': False, 'production_q8down_gdn_fallback': False}
    for name, enabled in expected.items():
        assert gate.CANDIDATES[name].q8_selected_down_enabled is enabled
    assert all(ENV not in spec.environment for spec in gate.CANDIDATES.values())


@pytest.mark.parametrize('enabled', [None, False, True])
def test_layer2_selected_scope_binds_and_restores(monkeypatch, enabled):
    monkeypatch.setenv(ENV, 'ambient')
    owner = SimpleNamespace(q8_selected_down_enabled=True,
        runner=SimpleNamespace(q8_selected_down_enabled=False))
    spec = replace(gate.CANDIDATES['production_baseline'], q8_selected_down_enabled=enabled)
    with ExitStack() as stack:
        assert gate.bind_candidate_selected_q8_down(stack, owner, spec) is (
            True if enabled is None else enabled)
        assert owner.q8_selected_down_enabled is (True if enabled is None else enabled)
        assert os.environ[ENV] == 'ambient'
    assert owner.q8_selected_down_enabled is True
    assert owner.runner.q8_selected_down_enabled is False
    assert ('q8_selected_down_enabled' in gate.candidate_owner_changes(spec)) is (enabled is not None)


def test_layer2_selected_gate_wiring():
    assert 'bind_candidate_selected_q8_down(' in inspect.getsource(gate.bind_candidate_owners)
    assert 'bind_candidate_selected_q8_down(' in inspect.getsource(gate._run)
    for name in ['qwen4exp_prefill_restore_ab', 'qwen4exp_conservative_cost']:
        module = __import__('scripts.' + name, fromlist=[name])
        assert 'candidate_selected_q8_down_scope(' in inspect.getsource(module)
