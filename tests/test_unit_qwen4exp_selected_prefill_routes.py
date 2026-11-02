"""The prefill gate reads selected owners rather than global bind history."""
import inspect
from types import SimpleNamespace
import pytest
from hipengine.runtime import gguf_linear as gl
from hipengine.generation import qwen4_exp_profiles as profiles
from scripts import execution_profile_q8_wmma_prefill_layers_gate as gate


def test_selected_prefill_route_report_is_owner_local(monkeypatch):
    first = SimpleNamespace(q8_wmma_layer_scope=gl.Q8WMMALayerScope.parse('0,7'),
                            q8_dense_wide_policy=gl.Q8DenseWidePolicy(True, frozenset({9})))
    second = SimpleNamespace(q8_wmma_layer_scope=gl.Q8WMMALayerScope.empty(),
                             q8_dense_wide_policy=gl.Q8DenseWidePolicy(False, None))
    before = profiles.selected_prefill_routes(first)
    profiles.selected_prefill_routes(second)
    monkeypatch.setenv('HIPENGINE_QWEN4_EXP_Q8_WMMA_LAYERS', '17')
    monkeypatch.setenv('HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE', '0')
    assert profiles.selected_prefill_routes(first) == before
    assert before == {'q8_wmma_layers': first.q8_wmma_layer_scope.to_report(),
                      'q8_dense_wide_policy': first.q8_dense_wide_policy.to_report()}
    # Reports are snapshots, not mutable aliases back into the owner.
    before.clear()
    assert profiles.selected_prefill_routes(first)


def test_gate_does_not_consult_global_conflicts():
    source = inspect.getsource(gate._run_gate)
    assert 'selected_prefill_routes(generator)' in source
    assert 'last_prebinder_conflicts' not in source
    assert "'bound_owner_routes': bound_owner_routes" in source
