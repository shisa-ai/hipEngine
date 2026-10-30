"""QSA WMMA is selected by the owning runner, not process state."""
import ast
import inspect
from types import SimpleNamespace

import pytest
from hipengine.runtime import qwen4_exp_runner as runtime
from hipengine.generation.qwen4_exp_gguf import _inherit_lane_decisions


def test_scope_is_local_nested_and_restores_on_error(monkeypatch):
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_QSA_WMMA", "1")
    a, b = SimpleNamespace(qsa_wmma_enabled=True), SimpleNamespace(qsa_wmma_enabled=False)
    with pytest.raises(RuntimeError), runtime.qsa_wmma_attention(a, False):
        assert not a.qsa_wmma_enabled and not b.qsa_wmma_enabled
        with runtime.qsa_wmma_attention(a, True):
            assert a.qsa_wmma_enabled
        assert not a.qsa_wmma_enabled
        raise RuntimeError("diagnostic failed")
    assert a.qsa_wmma_enabled and not b.qsa_wmma_enabled
    with runtime.qsa_wmma_attention(b):
        assert not b.qsa_wmma_enabled


@pytest.mark.parametrize("enabled", [False, True])
def test_late_runner_inherits_selection(enabled):
    runner = _inherit_lane_decisions(SimpleNamespace(qsa_wmma_enabled=enabled), SimpleNamespace())
    assert runner.qsa_wmma_enabled is enabled


def test_diagnostic_candidates_bind_wmma_explicitly():
    from scripts.qwen4exp_layer2_profile_gate import CANDIDATES
    for name, candidate in CANDIDATES.items():
        assert "HIPENGINE_QWEN4_EXP_QSA_WMMA" not in candidate.environment
        if name.startswith("recovery_"):
            assert candidate.qsa_wmma_enabled is (name == "recovery_plus_qsa_wmma")
    assert CANDIDATES["fast_composition"].qsa_wmma_enabled is True


def test_prefill_passes_explicit_selection():
    source = inspect.getsource(runtime)
    tree = ast.parse(source)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name)
             and node.func.id in {"run_qwen4_exp_qsa_prefill_layer", "run_qwen4_exp_qsa_prefill_token_mixer"}]
    assert calls
    assert all(any(kw.arg == "qsa_wmma_enabled" for kw in call.keywords) for call in calls)
    assert '_QSA_WMMA_ENV' not in source
