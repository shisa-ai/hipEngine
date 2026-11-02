"""Selected Q8 down enablement follows its owner, not its peer's process input."""
import inspect
from types import SimpleNamespace
import pytest
from hipengine.runtime import qwen4_exp_runner as runtime
from hipengine.runtime import qwen4_exp_batch_runner as batch
from tests.test_unit_qwen4exp_batch_runner import _fake_runner

ENV = 'HIPENGINE_QWEN4_EXP_Q8_0_SELECTED_WMMA_DOWN'


@pytest.mark.parametrize('ambient', ['', '0', 'false', 'False', '1', 'FALSE'])
def test_selected_down_resolves_explicit_owner_not_ambient(monkeypatch, ambient):
    monkeypatch.setenv(ENV, ambient)
    for value in (None, False, True):
        owner = SimpleNamespace(q8_selected_down_enabled=value)
        expected = bool(value)
        assert runtime.resolve_q8_selected_down(owner) is expected


def test_selected_down_scope_restores_missing_and_distinct_owners():
    runner = SimpleNamespace(q8_selected_down_enabled=False)
    owner = SimpleNamespace(runner=runner)
    with pytest.raises(RuntimeError):
        with runtime.q8_selected_down_enabled(owner, True):
            assert owner.q8_selected_down_enabled is runner.q8_selected_down_enabled is True
            raise RuntimeError('scope')
    assert not hasattr(owner, 'q8_selected_down_enabled')
    assert runner.q8_selected_down_enabled is False


def test_selected_down_field_reaches_all_moe_carriers():
    for module, calls in ((runtime, 4), (batch, 2)):
        cls = (runtime.Qwen4ExpGGUFResidentModelRunner if module is runtime
               else batch.Qwen4ExpGGUFBatchRunner)
        assert cls.q8_selected_down_enabled is None
        source = inspect.getsource(cls)
        assert source.count('q8_selected_down_enabled=self.q8_selected_down_enabled') == calls
    for function in (runtime.run_qwen4_exp_moe, runtime.run_qwen4_exp_dense_qsa_layer,
        runtime.run_qwen4_exp_qsa_prefill_layer, runtime.run_qwen4_exp_gdn_layer,
        batch.run_qwen4_exp_batch_gdn_layer, batch.run_qwen4_exp_batch_qsa_layer,
        batch._run_layer_ffn):
        assert inspect.signature(function).parameters['q8_selected_down_enabled'].default is None
        assert 'q8_selected_down_enabled=q8_selected_down_enabled' in inspect.getsource(function)


@pytest.mark.parametrize('enabled', [False, True])
def test_selected_down_real_dispatch_uses_owner(monkeypatch, enabled):
    from tests.test_unit_qwen4exp_q8_down_variant_owner import (
        _fake_moe_dispatch, GUARDED)
    real = runtime.run_qwen4_exp_moe
    def bound(*args, **kwargs):
        kwargs['q8_selected_down_enabled'] = enabled
        monkeypatch.setenv(ENV, '0' if enabled else '1')
        return real(*args, **kwargs)
    monkeypatch.setattr(runtime, 'run_qwen4_exp_moe', bound)
    record = _fake_moe_dispatch(monkeypatch, bound_variant=GUARDED)
    assert bool(record['launches']) is enabled


def test_selected_down_inherits_on_new_runner():
    from hipengine.generation.qwen4_exp_gguf import _inherit_lane_decisions
    owner = SimpleNamespace(q8_selected_down_enabled=True)
    runner = SimpleNamespace()
    _inherit_lane_decisions(owner, runner)
    assert runner.q8_selected_down_enabled is True


def test_selected_down_capture_key_tracks_effective_decision(monkeypatch):
    runner = _fake_runner(slots=1)
    runner.q8_selected_down_enabled = False
    first = runner._expert_down_policy()
    runner.q8_selected_down_enabled = True
    assert runner._expert_down_policy() != first
    runner.q8_selected_down_enabled = None
    monkeypatch.setenv(ENV, '1')
    assert runner._expert_down_policy() == first
