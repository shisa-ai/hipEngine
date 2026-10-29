"""GDN decode selection follows the owning profile, never process state."""
import ast
import inspect
import textwrap
import os
from types import SimpleNamespace

import pytest

from hipengine.execution_profiles import ExecutionProfile, resolve_runtime_profile
from hipengine.generation.qwen4_exp_gguf import _inherit_lane_decisions
from hipengine.generation.qwen4_exp_profiles import register_qwen4_exp_gfx1151_profiles
from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
from hipengine.runtime import qwen4_exp_runner as runtime

ENV = "HIPENGINE_QWEN4_EXP_GDN_DECODE_ROWSPLIT"
SCOPE = "decode_c1_d128"
VARIANT = "qwen4exp_gdn_rowsplit_decode"


@pytest.mark.parametrize("quant", ["gguf_q4_k_m", "gguf_ud_q4_k_xl"])
@pytest.mark.parametrize("profile", [ExecutionProfile.STRICT, ExecutionProfile.PRODUCTION])
def test_profile_declares_and_publishes_decode_choice(monkeypatch, quant, profile):
    monkeypatch.setenv(ENV, "stale")
    register_gfx1151_kernels(replace=True)
    register_qwen4_exp_gfx1151_profiles()
    resolved = resolve_runtime_profile(model="qwen4_exp_gguf", backend="hip_gfx1151",
                                       quant=quant, profile=profile)
    row = next(row for row in resolved.manifest["selections"] if row["scope"] == SCOPE)
    selected = profile == ExecutionProfile.PRODUCTION and quant == "gguf_ud_q4_k_xl"
    assert row["selected_variant"] == (VARIANT if selected else "qwen4exp_sigmoid_strict")
    assert row["strict_fallback_variant"] == "qwen4exp_sigmoid_strict"
    generator = SimpleNamespace(runner=SimpleNamespace())
    resolved.binder(generator, resolved)
    clone = SimpleNamespace()
    _inherit_lane_decisions(generator, clone)
    assert generator.gdn_decode_rowsplit is selected
    assert generator.runner.gdn_decode_rowsplit is selected
    assert clone.gdn_decode_rowsplit is selected
    assert os.environ[ENV] == "stale"


def test_diagnostic_scope_restores_and_isolates():
    first = SimpleNamespace(gdn_decode_rowsplit=True)
    peer = SimpleNamespace(gdn_decode_rowsplit=True)
    with pytest.raises(RuntimeError):
        with runtime.gdn_decode_rowsplit(first, False):
            assert not first.gdn_decode_rowsplit
            assert peer.gdn_decode_rowsplit
            raise RuntimeError("arm failed")
    assert first.gdn_decode_rowsplit


@pytest.mark.parametrize("enabled,dim,registered", [(True, 128, True), (False, 128, True),
                                                    (True, 64, True), (True, 256, True),
                                                    (True, 128, False)])
def test_decode_dispatch_preserves_geometry_and_registration_fallback(monkeypatch, enabled, dim, registered):
    monkeypatch.setenv(ENV, "0" if enabled else "1")
    sentinel = object()
    monkeypatch.setattr(runtime, "is_registered", lambda key: registered)
    monkeypatch.setattr(runtime, "resolve", lambda **kw: sentinel)
    fn = runtime._qwen4_exp_gdn_decode_step("hip_gfx1151", dim, enabled)
    assert fn is (sentinel if enabled and dim == 128 and registered
                  else runtime.qwen4_exp_gdn_decode_f32)


@pytest.mark.parametrize("function", [runtime.run_qwen4_exp_gdn_layer,
                                     runtime.run_qwen4_exp_gdn_token_mixer])
def test_unbound_defaults_to_strict(function):
    assert inspect.signature(function).parameters["gdn_decode_rowsplit"].default is False


@pytest.mark.parametrize("function", [runtime.Qwen4ExpGGUFResidentModelRunner.step,
                                     runtime.Qwen4ExpGGUFResidentModelRunner._prefill_chunk_steps,
                                     runtime.run_qwen4_exp_gdn_layer])
def test_every_scalar_gdn_call_forwards_owner(function):
    calls = [node for node in ast.walk(ast.parse(textwrap.dedent(inspect.getsource(function))))
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
             and node.func.id in {"run_qwen4_exp_gdn_layer", "run_qwen4_exp_gdn_token_mixer"}]
    assert calls
    for call in calls:
        argument = next(k.value for k in call.keywords if k.arg == "gdn_decode_rowsplit")
        if function is runtime.run_qwen4_exp_gdn_layer:
            assert isinstance(argument, ast.Name) and argument.id == "gdn_decode_rowsplit"
        else:
            assert isinstance(argument, ast.Attribute) and argument.attr == "gdn_decode_rowsplit"


def test_recovery_arms_use_explicit_control():
    from scripts.qwen4exp_layer2_profile_gate import _f16_route_isolation_candidates
    for candidate in _f16_route_isolation_candidates().values():
        assert ENV not in candidate.environment
        assert candidate.gdn_decode_rowsplit is False
