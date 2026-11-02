"""Q4 risk multipliers follow explicit owners, without ambient input."""
import inspect
import os
from types import SimpleNamespace

import pytest
from hipengine.runtime import qwen4_exp_runner as runtime
from hipengine.runtime import qwen4_exp_batch_runner as batch
from tests.test_unit_qwen4exp_batch_runner import _fake_runner

ENV = 'HIPENGINE_QWEN4_EXP_Q4_IU8_RISK_MULT'


@pytest.mark.parametrize('value', [0.5, 1.0, 4.0, 16.0])
def test_owner_multiplier_beats_ambient(monkeypatch, value):
    monkeypatch.setenv(ENV, '8')
    assert runtime._qwen4_exp_q4_iu8_risk_multiplier(value) == value
    assert runtime.resolve_q4_iu8_risk_mult(SimpleNamespace(q4_iu8_risk_mult=value)) == value
    assert runtime._qwen4_exp_q4_iu8_risk_multiplier() == 4.0


@pytest.mark.parametrize('raw', ['0', '-1', 'nan', 'bad'])
def test_invalid_ambient_multiplier_is_ignored(monkeypatch, raw):
    monkeypatch.setenv(ENV, raw)
    assert runtime._qwen4_exp_q4_iu8_risk_multiplier() == 4.0
    assert runtime.resolve_q4_iu8_risk_mult(SimpleNamespace()) == 4.0
    assert runtime._qwen4_exp_q4_iu8_risk_multiplier(1.0) == 1.0


@pytest.mark.parametrize('raw', ['0', '-1', 'nan', 'bad'])
def test_invalid_explicit_multiplier_still_fails(raw):
    with pytest.raises(ValueError, match='must be a positive float'):
        runtime._qwen4_exp_q4_iu8_risk_multiplier(raw)


def test_multiplier_scope_restores_missing_distinct_and_nested_state(monkeypatch):
    monkeypatch.setenv(ENV, '8')
    owner = SimpleNamespace(runner=SimpleNamespace(q4_iu8_risk_mult=4.0))
    with pytest.raises(RuntimeError, match='restore'):
        with runtime.q4_iu8_risk_mult(owner, 1.0):
            assert owner.q4_iu8_risk_mult == owner.runner.q4_iu8_risk_mult == 1.0
            with runtime.q4_iu8_risk_mult(owner, 0.5):
                assert owner.q4_iu8_risk_mult == 0.5
            assert owner.q4_iu8_risk_mult == 1.0
            raise RuntimeError('restore')
    assert not hasattr(owner, 'q4_iu8_risk_mult')
    assert owner.runner.q4_iu8_risk_mult == 4.0
    assert os.environ[ENV] == '8'


def test_multiplier_carriers_and_inheritance():
    from hipengine.generation.qwen4_exp_gguf import _inherit_lane_decisions
    assert _inherit_lane_decisions(SimpleNamespace(q4_iu8_risk_mult=1.0), SimpleNamespace()).q4_iu8_risk_mult == 1.0
    for fn in (runtime.run_qwen4_exp_moe, runtime.run_qwen4_exp_dense_qsa_layer,
               runtime.run_qwen4_exp_qsa_prefill_layer, runtime.run_qwen4_exp_gdn_layer,
               batch.run_qwen4_exp_batch_gdn_layer, batch.run_qwen4_exp_batch_qsa_layer,
               batch._run_layer_ffn):
        assert inspect.signature(fn).parameters['q4_iu8_risk_mult'].default is None
        assert 'q4_iu8_risk_mult=q4_iu8_risk_mult' in inspect.getsource(fn)
    assert '_qwen4_exp_q4_iu8_risk_multiplier(q4_iu8_risk_mult)' in inspect.getsource(runtime.run_qwen4_exp_moe)


def test_multiplier_capture_identity(monkeypatch):
    owner = _fake_runner(slots=1)
    owner.q4_iu8_risk_mult = 4.0
    before = owner._expert_down_policy()
    owner.q4_iu8_risk_mult = 1.0
    assert owner._expert_down_policy() != before
    owner.q4_iu8_risk_mult = None
    monkeypatch.setenv(ENV, '1')
    assert owner._expert_down_policy() == before
