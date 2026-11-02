"""Plain-Q4 layer scope follows the owner and preserves unbound parsing."""
import inspect
import os
from types import SimpleNamespace
import pytest
from hipengine.runtime import qwen4_exp_runner as runtime
from hipengine.runtime import qwen4_exp_batch_runner as batch
from tests.test_unit_qwen4exp_batch_runner import _fake_runner

ENV = 'HIPENGINE_QWEN4_EXP_Q4_IU8_LAYERS'


def weights(layer):
    return {'expert_gate': SimpleNamespace(spec=SimpleNamespace(slot_path=f'layers.{layer}.expert_gate'))}


@pytest.mark.parametrize('scope,selected', [('', True), ('all', True), ('2,17-19', False),
                                           ('35-47', True), ('0', False)])
def test_layer_owner_beats_ambient(monkeypatch, scope, selected):
    monkeypatch.setenv(ENV, 'all' if not selected else '0')
    assert runtime._qwen4_exp_q4_iu8_layer_allowed(weights(36), scope) is selected
    assert runtime.resolve_q4_iu8_layers(SimpleNamespace(q4_iu8_layers=scope)) == scope
    assert runtime._qwen4_exp_q4_iu8_layer_allowed(weights(36)) is (not selected)


def test_layer_scope_restores_missing_distinct_and_nested_state(monkeypatch):
    monkeypatch.setenv(ENV, '35-47')
    owner = SimpleNamespace(runner=SimpleNamespace(q4_iu8_layers='2'))
    with pytest.raises(RuntimeError, match='restore'):
        with runtime.q4_iu8_layers(owner, 'all'):
            assert owner.q4_iu8_layers == owner.runner.q4_iu8_layers == 'all'
            with runtime.q4_iu8_layers(owner, ''):
                assert owner.q4_iu8_layers == owner.runner.q4_iu8_layers == ''
            assert owner.q4_iu8_layers == 'all'
            raise RuntimeError('restore')
    assert not hasattr(owner, 'q4_iu8_layers')
    assert owner.runner.q4_iu8_layers == '2'
    assert os.environ[ENV] == '35-47'


def test_scope_carriers_and_inheritance():
    from hipengine.generation.qwen4_exp_gguf import _inherit_lane_decisions
    assert _inherit_lane_decisions(SimpleNamespace(q4_iu8_layers='all'), SimpleNamespace()).q4_iu8_layers == 'all'
    for fn in (runtime.run_qwen4_exp_moe, runtime.run_qwen4_exp_dense_qsa_layer,
               runtime.run_qwen4_exp_qsa_prefill_layer, runtime.run_qwen4_exp_gdn_layer,
               batch.run_qwen4_exp_batch_gdn_layer, batch.run_qwen4_exp_batch_qsa_layer,
               batch._run_layer_ffn):
        assert inspect.signature(fn).parameters['q4_iu8_layers'].default is None
        assert 'q4_iu8_layers=q4_iu8_layers' in inspect.getsource(fn)
    assert '_qwen4_exp_q4_iu8_layer_allowed(weights, q4_iu8_layers)' in inspect.getsource(runtime.run_qwen4_exp_moe)


def test_layer_scope_capture_identity(monkeypatch):
    owner = _fake_runner(slots=1)
    owner.q4_iu8_layers = 'all'
    before = owner._expert_down_policy()
    owner.q4_iu8_layers = '2'
    assert owner._expert_down_policy() != before
    owner.q4_iu8_layers = None
    monkeypatch.setenv(ENV, 'all')
    assert owner._expert_down_policy() == before
