"""Plain Q4 IU8 prefill enablement follows owner state."""
import inspect
from types import SimpleNamespace
import pytest
from hipengine.runtime import qwen4_exp_runner as runtime
from hipengine.runtime import qwen4_exp_batch_runner as batch
from tests.test_unit_qwen4exp_batch_runner import _fake_runner

ENV = 'HIPENGINE_QWEN4_EXP_Q4_IU8_PREFILL'


@pytest.mark.parametrize('ambient', ['', '0', 'false', 'False', '1', 'FALSE'])
def test_prefill_owner_beats_ambient(monkeypatch, ambient):
    monkeypatch.setenv(ENV, ambient)
    for enabled in (None, False, True):
        owner = SimpleNamespace(q4_iu8_prefill_enabled=enabled)
        expected = ambient not in {'', '0', 'false', 'False'} if enabled is None else enabled
        assert runtime.resolve_q4_iu8_prefill(owner) is expected


def test_prefill_scope_restores_missing_and_distinct_values():
    runner = SimpleNamespace(q4_iu8_prefill_enabled=False)
    owner = SimpleNamespace(runner=runner)
    with pytest.raises(RuntimeError):
        with runtime.q4_iu8_prefill_enabled(owner, True):
            assert owner.q4_iu8_prefill_enabled is runner.q4_iu8_prefill_enabled is True
            raise RuntimeError('restore')
    assert not hasattr(owner, 'q4_iu8_prefill_enabled')
    assert runner.q4_iu8_prefill_enabled is False


def test_prefill_carriers_and_inheritance():
    from hipengine.generation.qwen4_exp_gguf import _inherit_lane_decisions
    assert _inherit_lane_decisions(SimpleNamespace(q4_iu8_prefill_enabled=True), SimpleNamespace()).q4_iu8_prefill_enabled is True
    for fn in (runtime.run_qwen4_exp_moe, runtime.run_qwen4_exp_dense_qsa_layer,
        runtime.run_qwen4_exp_qsa_prefill_layer, runtime.run_qwen4_exp_gdn_layer,
        batch.run_qwen4_exp_batch_gdn_layer, batch.run_qwen4_exp_batch_qsa_layer,
        batch._run_layer_ffn):
        assert inspect.signature(fn).parameters['q4_iu8_prefill_enabled'].default is None
        assert 'q4_iu8_prefill_enabled=q4_iu8_prefill_enabled' in inspect.getsource(fn)


def test_prefill_capture_identity(monkeypatch):
    owner = _fake_runner(slots=1)
    owner.q4_iu8_prefill_enabled = False
    before = owner._expert_down_policy()
    owner.q4_iu8_prefill_enabled = True
    assert owner._expert_down_policy() != before
    owner.q4_iu8_prefill_enabled = None
    monkeypatch.setenv(ENV, '0')
    assert owner._expert_down_policy() == before
