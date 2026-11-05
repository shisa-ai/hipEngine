"""Q5 risk controls belong to an invocation, never ambient process state."""
from types import SimpleNamespace
import inspect

import pytest

from hipengine.runtime import qwen4_exp_runner as runner


@pytest.mark.parametrize("family,key", [
    ("q5k", "HIPENGINE_QWEN4_EXP_Q5_K_IU8_RISK_MULT"),
    ("q51", "HIPENGINE_QWEN4_EXP_Q51_IU8_RISK_MULT"),
])
def test_multiplier_is_explicit_and_environment_is_inert(monkeypatch, family, key):
    resolve = getattr(runner, f"resolve_{family}_iu8_risk_mult")
    scope = getattr(runner, f"{family}_iu8_risk_mult")
    field = f"{family}_iu8_risk_mult"
    owner = SimpleNamespace(runner=SimpleNamespace(**{field: 32.0}))
    other = SimpleNamespace()
    monkeypatch.setenv(key, "not-a-number")
    assert resolve(owner) == 16.0
    with pytest.raises(RuntimeError, match="unwind"):
        with scope(owner, 24.0):
            assert resolve(owner) == resolve(owner.runner) == 24.0
            assert resolve(other) == 16.0
            with scope(owner, 48.0):
                assert resolve(owner) == resolve(owner.runner) == 48.0
            assert resolve(owner) == 24.0
            raise RuntimeError("unwind")
    assert not hasattr(owner, field)
    assert resolve(owner.runner) == 32.0
    for invalid in (0, -1, float("nan"), "bad"):
        with pytest.raises(ValueError, match="positive float"):
            with scope(owner, invalid):
                pass
        assert not hasattr(owner, field)
        assert resolve(owner.runner) == 32.0


def test_clones_and_batch_capture_keys_preserve_multiplier_ownership():
    from hipengine.generation.qwen4_exp_gguf import _inherit_lane_decisions
    from hipengine.runtime.qwen4_exp_batch_runner import Qwen4ExpGGUFBatchRunner

    source = SimpleNamespace(q5k_iu8_risk_mult=24.0, q51_iu8_risk_mult=32.0)
    clone = SimpleNamespace()
    _inherit_lane_decisions(source, clone)
    assert clone.q5k_iu8_risk_mult == 24.0
    assert clone.q51_iu8_risk_mult == 32.0
    fields = ("q51_down_m1", "q51_fold128_prefill", "q51_fold_pair_prefill",
              "q51_exact_grouped_down", "q51_wmma_down", "q51_out8",
              "q4_exact_grouped_gate", "q4_exact_grouped_all", "q4_out4",
              "forkb_grouped_down", "q8_down_row4_prefill", "q8_down_bundle_prefill",
              "q5k_row4_prefill")
    owner = SimpleNamespace(**dict.fromkeys(fields, False))
    baseline = Qwen4ExpGGUFBatchRunner._expert_down_policy(owner)
    owner.q5k_iu8_risk_mult = 24.0
    first = Qwen4ExpGGUFBatchRunner._expert_down_policy(owner)
    owner.q51_iu8_risk_mult = 32.0
    second = Qwen4ExpGGUFBatchRunner._expert_down_policy(owner)
    assert baseline != first != second
    assert baseline[-3:-1] == (16.0, 16.0)
    assert second[-3:-1] == (24.0, 32.0)


def test_all_layer_surfaces_forward_explicit_q5_controls():
    from hipengine.runtime import qwen4_exp_batch_runner as batch
    surfaces = [runner.run_qwen4_exp_moe, runner.run_qwen4_exp_dense_qsa_layer,
                runner.run_qwen4_exp_qsa_prefill_layer, runner.run_qwen4_exp_gdn_layer,
                batch.run_qwen4_exp_batch_gdn_layer, batch._run_layer_ffn,
                batch.run_qwen4_exp_batch_qsa_layer,
                runner.Qwen4ExpGGUFResidentModelRunner]
    for surface in surfaces:
        parameters = inspect.signature(surface).parameters
        for field in ("q5k_iu8_risk_mult", "q51_iu8_risk_mult"):
            assert parameters[field].default is None
