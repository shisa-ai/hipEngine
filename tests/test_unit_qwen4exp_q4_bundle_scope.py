"""Q4 bundle selection must be local to the owning runner."""

import inspect
from types import SimpleNamespace

import pytest

from hipengine.runtime import qwen4_exp_runner as runner
from hipengine.runtime import qwen4_exp_batch_runner as batch
from hipengine.generation.qwen4_exp_gguf import _inherit_lane_decisions

ENV = "HIPENGINE_QWEN4_EXP_Q4_BUNDLE_PREFILL"


@pytest.mark.parametrize("published", [False, True])
@pytest.mark.parametrize("stale", ["0", "1"])
def test_bundle_admission_ignores_ambient_values(monkeypatch, published, stale):
    monkeypatch.setenv(ENV, stale)
    monkeypatch.setattr(runner, "is_registered", lambda key: key.backend == "present")
    assert runner.qwen4_exp_q4_bundle_prefill_selected(
        "present", "gguf_q4_k", published=published) is published
    assert not runner.qwen4_exp_q4_bundle_prefill_selected(
        "missing", "gguf_q4_k", published=published)


def test_nested_scope_restores_without_changing_another_owner():
    owner = SimpleNamespace(q4_bundle_prefill=True)
    other = SimpleNamespace(q4_bundle_prefill=False)
    with pytest.raises(RuntimeError, match="failed"):
        with runner.q4_bundle_prefill(owner, False):
            assert owner.q4_bundle_prefill is False
            with runner.q4_bundle_prefill(owner, True):
                assert owner.q4_bundle_prefill is True
            assert owner.q4_bundle_prefill is False
            assert other.q4_bundle_prefill is False
            raise RuntimeError("failed")
    assert owner.q4_bundle_prefill is True


@pytest.mark.parametrize("published", [False, True])
def test_late_runner_inherits_bundle_choice(published):
    owner = _inherit_lane_decisions(
        SimpleNamespace(q4_bundle_prefill=published), SimpleNamespace())
    assert owner.q4_bundle_prefill is published


@pytest.mark.parametrize("function", [
    runner.run_qwen4_exp_moe,
    runner.run_qwen4_exp_dense_qsa_layer,
    runner.run_qwen4_exp_qsa_prefill_layer,
    runner.run_qwen4_exp_gdn_layer,
    batch.run_qwen4_exp_batch_gdn_layer,
    batch.run_qwen4_exp_batch_qsa_layer,
    batch._run_layer_ffn,
])
def test_moe_forwarders_have_explicit_bundle_choice(function):
    parameter = inspect.signature(function).parameters["q4_bundle_prefill"]
    assert parameter.default is False
    assert "q4_bundle_prefill=q4_bundle_prefill" in inspect.getsource(function)


def test_bundle_key_preserves_pair_precedence_and_grid_guard(monkeypatch):
    monkeypatch.setattr(runner, "is_registered", lambda key: True)
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_EXACT_EXPERT_GRID", "64")
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_Q4_PAIR_PREFILL", "0")
    kwargs = dict(backend="test", quant="gguf_q4_k", rows=131, in_features=256)
    assert "bundle" in runner._qwen4_exp_q4_gate_up_key(
        **kwargs, q4_bundle_prefill=True).variant
    assert "bundle" not in runner._qwen4_exp_q4_gate_up_key(
        **kwargs, q4_bundle_prefill=False).variant
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_Q4_PAIR_PREFILL", "1")
    assert "pair2" in runner._qwen4_exp_q4_gate_up_key(
        **kwargs, q4_bundle_prefill=True).variant
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_EXACT_EXPERT_GRID", "0")
    assert "bundle" not in runner._qwen4_exp_q4_gate_up_key(
        **kwargs, q4_bundle_prefill=True).variant


@pytest.mark.parametrize("package", ["q4-bundle", "prefill-bundle"])
@pytest.mark.parametrize("mode", ["before", "after"])
def test_campaign_scope_restores_both_controls_on_failure(package, mode):
    from scripts.qwen4exp_halo_box_campaign_ab import route_scope
    from scripts.qwen4exp_row4_state_gate import apply_state_gate_mode

    owner = SimpleNamespace(q4_bundle_prefill=True, qsa_h256_prefill_mode="wave")
    env = {ENV: "stale"}
    apply_state_gate_mode(owner, package, "1", "", environment=env)
    with pytest.raises(RuntimeError, match="failure"):
        with route_scope(package, mode, owner, environment=env):
            assert owner.q4_bundle_prefill is (mode == "after")
            expected = ("page256" if mode == "after" else "parent")
            assert owner.qsa_h256_prefill_mode == (
                expected if package == "prefill-bundle" else "wave")
            assert env == {ENV: "stale"}
            raise RuntimeError("failure")
    assert owner.q4_bundle_prefill is True
    assert owner.qsa_h256_prefill_mode == "wave"


@pytest.mark.parametrize("package", ["q4-bundle", "prefill-bundle"])
def test_campaign_scope_requires_owning_runner(package):
    from scripts.qwen4exp_halo_box_campaign_ab import route_scope
    with pytest.raises(ValueError, match="needs the owning runner"):
        route_scope(package, "after", None)
