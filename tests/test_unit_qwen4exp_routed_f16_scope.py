"""Routed F16 selection belongs to the bound runner, not the process."""

import inspect
import os
from types import SimpleNamespace

import pytest

from hipengine.execution_profiles import ExecutionProfile, resolve_runtime_profile
from hipengine.generation.qwen4_exp_gguf import _inherit_lane_decisions
from hipengine.generation.qwen4_exp_profiles import register_qwen4_exp_gfx1151_profiles
from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
from hipengine.runtime import qwen4_exp_runner as runtime


FIELD = "routed_f16_enabled"
RETIRED_ENV = "HIPENGINE_QWEN4_EXP_ROUTED_F16"


@pytest.mark.parametrize("quant", ["gguf_q4_k_m", "gguf_ud_q4_k_xl"])
@pytest.mark.parametrize("profile", [ExecutionProfile.STRICT, ExecutionProfile.PRODUCTION])
def test_manifest_publishes_runner_local_selection(monkeypatch, quant, profile):
    monkeypatch.setenv(RETIRED_ENV, "caller-sentinel")
    register_gfx1151_kernels(replace=True)
    register_qwen4_exp_gfx1151_profiles()
    resolved = resolve_runtime_profile(
        model="qwen4_exp_gguf", backend="hip_gfx1151", quant=quant, profile=profile)
    runner = SimpleNamespace()
    generator = SimpleNamespace(runner=runner)
    resolved.binder(generator, resolved)
    selected = any(
        row["selected_variant"] == runtime.ROUTED_F16_GATE_UP_VARIANT
        for row in resolved.manifest["selections"])
    assert getattr(generator, FIELD) is selected
    assert getattr(runner, FIELD) is selected
    assert os.environ[RETIRED_ENV] == "caller-sentinel"
    clone = SimpleNamespace()
    _inherit_lane_decisions(generator, clone)
    assert getattr(clone, FIELD) is selected


def test_runner_scope_restores_after_failure_and_isolates_peers():
    first = SimpleNamespace(routed_f16_enabled=False)
    second = SimpleNamespace(routed_f16_enabled=True)
    with pytest.raises(RuntimeError, match="diagnostic failed"):
        with runtime.routed_f16_enabled(first, True):
            assert first.routed_f16_enabled is True
            with runtime.routed_f16_enabled(second, False):
                assert first.routed_f16_enabled is True
                assert second.routed_f16_enabled is False
            assert second.routed_f16_enabled is True
            raise RuntimeError("diagnostic failed")
    assert first.routed_f16_enabled is False


def test_unbound_dispatch_defaults_to_exact_parent():
    parameters = inspect.signature(runtime.run_qwen4_exp_moe).parameters
    assert parameters[FIELD].default is False
    assert not hasattr(runtime, "_routed_f16_enabled")


@pytest.mark.parametrize("module_name,function_name", [
    ("qwen4_exp_runner", "run_qwen4_exp_moe"),
    ("qwen4_exp_runner", "run_qwen4_exp_dense_qsa_layer"),
    ("qwen4_exp_runner", "run_qwen4_exp_qsa_prefill_layer"),
    ("qwen4_exp_runner", "run_qwen4_exp_gdn_layer"),
    ("qwen4_exp_batch_runner", "run_qwen4_exp_batch_gdn_layer"),
    ("qwen4_exp_batch_runner", "run_qwen4_exp_batch_qsa_layer"),
    ("qwen4_exp_batch_runner", "_run_layer_ffn"),
])
def test_every_moe_forwarder_carries_selection(module_name, function_name):
    import ast
    import importlib
    import textwrap
    module = importlib.import_module("hipengine.runtime." + module_name)
    function = getattr(module, function_name)
    assert inspect.signature(function).parameters[FIELD].default is False
    calls = [node for node in ast.walk(ast.parse(textwrap.dedent(inspect.getsource(function))))
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
             and node.func.id in {"run_qwen4_exp_moe", "_run_layer_ffn"}]
    assert calls
    for call in calls:
        assert any(keyword.arg == FIELD and isinstance(keyword.value, ast.Name)
                   and keyword.value.id == FIELD for keyword in call.keywords)


def test_batch_capture_key_changes_with_selection():
    from hipengine.runtime.qwen4_exp_batch_runner import Qwen4ExpGGUFBatchRunner
    runner = object.__new__(Qwen4ExpGGUFBatchRunner)
    before = runner._expert_down_policy()
    with runtime.routed_f16_enabled(runner, True):
        assert runner._expert_down_policy() != before
    assert runner._expert_down_policy() == before


@pytest.mark.parametrize("name", ["q51_pair2_scope", "q8_down_scope", "q8_mapped_down_scope"])
def test_diagnostic_closes_field_and_restores_on_failure(name):
    from scripts import qwen4exp_halo_box_campaign_ab as campaign
    runner = SimpleNamespace(routed_f16_enabled=True, q51_exact_grouped_down=True,
                             forkb_grouped_down=True, q5k_row4_prefill=True)
    environment = {RETIRED_ENV: "stale-value"}
    with pytest.raises(RuntimeError, match="arm failed"):
        with getattr(campaign, name)(runner=runner, environment=environment):
            assert runner.routed_f16_enabled is False
            assert environment[RETIRED_ENV] == "stale-value"
            raise RuntimeError("arm failed")
    assert runner.routed_f16_enabled is True
    assert environment == {RETIRED_ENV: "stale-value"}
