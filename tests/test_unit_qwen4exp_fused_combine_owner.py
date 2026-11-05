"""Grouped combine diagnostics select explicit owners without ambient reads."""
from types import SimpleNamespace
import inspect

import pytest
from hipengine.runtime import qwen4_exp_runner as runtime


def test_fused_combine_scope_restores_distinct_and_missing_state(monkeypatch):
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_FUSED_COMBINE", "1")
    owner = SimpleNamespace(runner=SimpleNamespace(fused_combine_enabled=True))
    other = SimpleNamespace()
    assert runtime.resolve_fused_combine(owner) is False
    with pytest.raises(RuntimeError, match="unwind"):
        with runtime.fused_combine_enabled(owner, True):
            assert runtime.resolve_fused_combine(owner) is True
            assert runtime.resolve_fused_combine(other) is False
            with runtime.fused_combine_enabled(owner, False):
                assert runtime.resolve_fused_combine(owner.runner) is False
            assert runtime.resolve_fused_combine(owner.runner) is True
            raise RuntimeError("unwind")
    assert not hasattr(owner, "fused_combine_enabled")
    assert owner.runner.fused_combine_enabled is True


def test_clones_and_capture_keys_keep_combine_choice():
    from hipengine.generation.qwen4_exp_gguf import _inherit_lane_decisions
    from hipengine.runtime.qwen4_exp_batch_runner import Qwen4ExpGGUFBatchRunner

    clone = SimpleNamespace()
    _inherit_lane_decisions(SimpleNamespace(fused_combine_enabled=True), clone)
    assert clone.fused_combine_enabled is True
    fields = ("q51_down_m1", "q51_fold128_prefill", "q51_fold_pair_prefill",
              "q51_exact_grouped_down", "q51_wmma_down", "q51_out8",
              "q4_exact_grouped_gate", "q4_exact_grouped_all", "q4_out4",
              "forkb_grouped_down", "q8_down_row4_prefill", "q8_down_bundle_prefill",
              "q5k_row4_prefill")
    owner = SimpleNamespace(**dict.fromkeys(fields, False))
    before = Qwen4ExpGGUFBatchRunner._expert_down_policy(owner)
    owner.fused_combine_enabled = True
    after = Qwen4ExpGGUFBatchRunner._expert_down_policy(owner)
    assert before != after
    assert before[:-1] == after[:-1]
    assert before[-1] is False and after[-1] is True


def test_fused_combine_is_an_explicit_layer_and_constructor_input():
    from hipengine.runtime import qwen4_exp_batch_runner as batch
    for surface in (runtime.run_qwen4_exp_moe, runtime.run_qwen4_exp_dense_qsa_layer,
                    runtime.run_qwen4_exp_qsa_prefill_layer, runtime.run_qwen4_exp_gdn_layer,
                    batch.run_qwen4_exp_batch_gdn_layer, batch._run_layer_ffn,
                    batch.run_qwen4_exp_batch_qsa_layer,
                    runtime.Qwen4ExpGGUFResidentModelRunner):
        assert inspect.signature(surface).parameters["fused_combine_enabled"].default is False
