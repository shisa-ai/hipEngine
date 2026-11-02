"""Q4 risk plane selection follows explicit owners, not ambient input."""
import inspect
import os
from types import SimpleNamespace

import pytest
from hipengine.runtime import qwen4_exp_runner as runtime
from hipengine.runtime import qwen4_exp_batch_runner as batch
from tests.test_unit_qwen4exp_batch_runner import _fake_runner

ENV = 'HIPENGINE_QWEN4_EXP_Q4_IU8_PLANES'


@pytest.mark.parametrize('ambient', ['2', '3', 'bad', ''])
def test_plane_selection_ignores_unbound_ambient(monkeypatch, ambient):
    monkeypatch.setenv(ENV, ambient)
    assert runtime.resolve_q4_iu8_planes(SimpleNamespace()) == 3
    assert runtime.resolve_q4_iu8_planes(SimpleNamespace(q4_iu8_planes=None)) == 3
    for value in (2, 3):
        assert runtime.resolve_q4_iu8_planes(SimpleNamespace(q4_iu8_planes=value)) == value


def test_plane_scope_restores_missing_distinct_and_nested_state(monkeypatch):
    monkeypatch.setenv(ENV, '3')
    owner = SimpleNamespace(runner=SimpleNamespace(q4_iu8_planes=3))
    with pytest.raises(RuntimeError, match='restore'):
        with runtime.q4_iu8_planes(owner, 2) as selected:
            assert selected == owner.q4_iu8_planes == owner.runner.q4_iu8_planes == 2
            with runtime.q4_iu8_planes(owner, 3):
                assert owner.q4_iu8_planes == 3
            assert owner.q4_iu8_planes == 2
            raise RuntimeError('restore')
    assert not hasattr(owner, 'q4_iu8_planes')
    assert owner.runner.q4_iu8_planes == 3
    assert os.environ[ENV] == '3'


def test_plane_carriers_and_inheritance():
    from hipengine.generation.qwen4_exp_gguf import _inherit_lane_decisions
    assert _inherit_lane_decisions(SimpleNamespace(q4_iu8_planes=2), SimpleNamespace()).q4_iu8_planes == 2
    for fn in (runtime.run_qwen4_exp_moe, runtime.run_qwen4_exp_dense_qsa_layer,
               runtime.run_qwen4_exp_qsa_prefill_layer, runtime.run_qwen4_exp_gdn_layer,
               batch.run_qwen4_exp_batch_gdn_layer, batch.run_qwen4_exp_batch_qsa_layer,
               batch._run_layer_ffn):
        assert inspect.signature(fn).parameters['q4_iu8_planes'].default is None
        assert 'q4_iu8_planes=q4_iu8_planes' in inspect.getsource(fn)
    assert '_qwen4_exp_q4_iu8_planes(q4_iu8_planes) == 2' in inspect.getsource(runtime.run_qwen4_exp_moe)


def test_plane_capture_identity(monkeypatch):
    owner = _fake_runner(slots=1)
    owner.q4_iu8_planes = 3
    before = owner._expert_down_policy()
    owner.q4_iu8_planes = 2
    assert owner._expert_down_policy() != before
    owner.q4_iu8_planes = None
    monkeypatch.setenv(ENV, '2')
    assert owner._expert_down_policy() == before
