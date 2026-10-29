"""Q4 pair2 admission belongs to its runner rather than the environment."""

import inspect
from types import SimpleNamespace

import pytest

from hipengine.runtime import qwen4_exp_runner as runner
from hipengine.runtime import qwen4_exp_batch_runner as batch
from hipengine.generation.qwen4_exp_gguf import _inherit_lane_decisions

ENV = "HIPENGINE_QWEN4_EXP_Q4_PAIR_PREFILL"


@pytest.mark.parametrize("rows", [17, 63, 64, 65, 131])
@pytest.mark.parametrize("published", [False, True])
@pytest.mark.parametrize("stale", ["0", "1"])
def test_pair_choice_preserves_shape_and_registry_guards(monkeypatch, rows, published, stale):
    monkeypatch.setenv(ENV, stale)
    monkeypatch.setattr(runner, "is_registered", lambda key: key.backend == "present")
    select = runner.qwen4_exp_q4_pair_prefill_selected
    args = dict(rows=rows, in_features=256, published=published)
    assert select("present", "gguf_q4_k", **args) is (published and rows >= 64)
    assert not select("missing", "gguf_q4_k", **args)
    for width in (0, 255, 4352):
        assert not select("present", "gguf_q4_k", **(args | {"in_features": width}))


def test_pair_scope_restores_nested_failure_and_keeps_other_owner():
    owner = SimpleNamespace(q4_pair_prefill=True)
    other = SimpleNamespace(q4_pair_prefill=True)
    with pytest.raises(RuntimeError, match="failed"):
        with runner.q4_pair_prefill(owner, False):
            with runner.q4_pair_prefill(owner, True):
                assert owner.q4_pair_prefill is True
            assert owner.q4_pair_prefill is False
            assert other.q4_pair_prefill is True
            raise RuntimeError("failed")
    assert owner.q4_pair_prefill is True


@pytest.mark.parametrize("published", [False, True])
def test_late_clone_inherits_pair_choice(published):
    child = _inherit_lane_decisions(SimpleNamespace(q4_pair_prefill=published),
                                   SimpleNamespace())
    assert child.q4_pair_prefill is published


@pytest.mark.parametrize("function", [
    runner.run_qwen4_exp_moe, runner.run_qwen4_exp_dense_qsa_layer,
    runner.run_qwen4_exp_qsa_prefill_layer, runner.run_qwen4_exp_gdn_layer,
    batch.run_qwen4_exp_batch_gdn_layer, batch.run_qwen4_exp_batch_qsa_layer,
    batch._run_layer_ffn,
])
def test_layer_forwarders_carry_explicit_pair_choice(function):
    assert inspect.signature(function).parameters["q4_pair_prefill"].default is False
    assert "q4_pair_prefill=q4_pair_prefill" in inspect.getsource(function)


def test_pair_precedes_bundle_only_with_expert_grid(monkeypatch):
    monkeypatch.setattr(runner, "is_registered", lambda key: True)
    monkeypatch.setenv(ENV, "1")
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_EXACT_EXPERT_GRID", "64")
    args = dict(backend="test", quant="gguf_q4_k", rows=131, in_features=256,
                q4_bundle_prefill=True)
    assert "bundle" in runner._qwen4_exp_q4_gate_up_key(
        **args, q4_pair_prefill=False).variant
    assert "pair2" in runner._qwen4_exp_q4_gate_up_key(
        **args, q4_pair_prefill=True).variant
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_EXACT_EXPERT_GRID", "0")
    key = runner._qwen4_exp_q4_gate_up_key(**args, q4_pair_prefill=True)
    assert "pair2" not in key.variant and "bundle" not in key.variant


def test_batch_graph_policy_discriminates_pair_selection():
    owner = object.__new__(batch.Qwen4ExpGGUFBatchRunner)
    before = owner._expert_down_policy()
    with runner.q4_pair_prefill(owner, True):
        after = owner._expert_down_policy()
    assert sum(a != b for a, b in zip(before, after, strict=True)) == 1
    assert owner._expert_down_policy() == before
