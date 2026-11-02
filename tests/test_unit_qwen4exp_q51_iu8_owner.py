"""Q5_1 exact IU8 selection follows an explicit owner or direct-call input."""
import inspect
from types import SimpleNamespace
import pytest
from hipengine.runtime import qwen4_exp_runner as runtime
from hipengine.runtime import qwen4_exp_batch_runner as batch
from tests.test_unit_qwen4exp_batch_runner import _fake_runner

ENV = 'HIPENGINE_QWEN4_EXP_Q51_IU8_EXACT'


@pytest.mark.parametrize('ambient', ['', '0', 'false', 'False', '1', 'FALSE'])
def test_q51_exact_owner_beats_ambient(monkeypatch, ambient):
    monkeypatch.setenv(ENV, ambient)
    for value in (None, False, True):
        owner = SimpleNamespace(q51_iu8_exact_enabled=value)
        expected = ambient not in {'', '0', 'false', 'False'} if value is None else value
        assert runtime.resolve_q51_iu8_exact(owner) is expected
        assert runtime._qwen4_exp_q51_iu8_exact_enabled(value) is expected


def test_q51_exact_scope_restores_missing_and_distinct_owners():
    runner = SimpleNamespace(q51_iu8_exact_enabled=False)
    owner = SimpleNamespace(runner=runner)
    with pytest.raises(RuntimeError):
        with runtime.q51_iu8_exact_enabled(owner, True):
            assert owner.q51_iu8_exact_enabled is runner.q51_iu8_exact_enabled is True
            raise RuntimeError('scope')
    assert not hasattr(owner, 'q51_iu8_exact_enabled')
    assert runner.q51_iu8_exact_enabled is False


def test_q51_exact_field_reaches_moe_carriers():
    for module, calls in ((runtime, 4), (batch, 2)):
        cls = (runtime.Qwen4ExpGGUFResidentModelRunner if module is runtime
               else batch.Qwen4ExpGGUFBatchRunner)
        assert cls.q51_iu8_exact_enabled is None
        assert inspect.getsource(cls).count(
            'q51_iu8_exact_enabled=self.q51_iu8_exact_enabled') == calls
    for function in (runtime.run_qwen4_exp_moe, runtime.run_qwen4_exp_dense_qsa_layer,
        runtime.run_qwen4_exp_qsa_prefill_layer, runtime.run_qwen4_exp_gdn_layer,
        batch.run_qwen4_exp_batch_gdn_layer, batch.run_qwen4_exp_batch_qsa_layer,
        batch._run_layer_ffn):
        assert inspect.signature(function).parameters['q51_iu8_exact_enabled'].default is None
        assert 'q51_iu8_exact_enabled=q51_iu8_exact_enabled' in inspect.getsource(function)


def test_q51_exact_inherits_on_new_runner():
    from hipengine.generation.qwen4_exp_gguf import _inherit_lane_decisions
    runner = _inherit_lane_decisions(SimpleNamespace(q51_iu8_exact_enabled=True), SimpleNamespace())
    assert runner.q51_iu8_exact_enabled is True


def test_q51_exact_capture_identity(monkeypatch):
    runner = _fake_runner(slots=1)
    runner.q51_iu8_exact_enabled = False
    first = runner._expert_down_policy()
    runner.q51_iu8_exact_enabled = True
    assert runner._expert_down_policy() != first
    runner.q51_iu8_exact_enabled = None
    monkeypatch.setenv(ENV, '0')
    assert runner._expert_down_policy() == first
