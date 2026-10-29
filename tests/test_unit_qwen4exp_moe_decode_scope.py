"""MoE decode choices belong to profiles and runners, not process switches."""
import ast
import inspect
import os
import textwrap
from types import SimpleNamespace

import pytest

from hipengine.execution_profiles import ExecutionProfile, resolve_runtime_profile
from hipengine.generation.qwen4_exp_gguf import _inherit_lane_decisions
from hipengine.generation.qwen4_exp_profiles import register_qwen4_exp_gfx1151_profiles
from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
from hipengine.runtime import qwen4_exp_runner as runtime
from hipengine.runtime import qwen4_exp_batch_runner as batch


@pytest.mark.parametrize("quant", ["gguf_q4_k_m", "gguf_ud_q4_k_xl"])
@pytest.mark.parametrize("profile", [ExecutionProfile.STRICT, ExecutionProfile.PRODUCTION])
def test_manifest_and_runner_own_warp_choices(monkeypatch, quant, profile):
    for leg in ("GATE_UP", "DOWN"):
        monkeypatch.setenv(f"HIPENGINE_QWEN4_EXP_MOE_DECODE_WARP_{leg}", "stale")
    register_gfx1151_kernels(replace=True)
    register_qwen4_exp_gfx1151_profiles()
    result = resolve_runtime_profile(model="qwen4_exp_gguf", backend="hip_gfx1151",
                                     quant=quant, profile=profile)
    rows = {r["scope"]: r for r in result.manifest["selections"]}
    active = profile == ExecutionProfile.PRODUCTION and quant == "gguf_ud_q4_k_xl"
    for scope in ("decode_c1_compact_le12_q4_dp4a_enabled_gate_up",
                  "decode_c1_compact_le12_q51_down"):
        row = rows[scope]
        assert ("warp256" in row["selected_variant"]) is active
        assert "warp256" not in row["strict_fallback_variant"]
    generator = SimpleNamespace(runner=SimpleNamespace())
    result.binder(generator, result)
    clone = _inherit_lane_decisions(generator, SimpleNamespace())
    expected = frozenset({"GATE_UP", "DOWN"}) if active else frozenset()
    assert generator.moe_decode_warp == generator.runner.moe_decode_warp == clone.moe_decode_warp == expected
    for leg in ("GATE_UP", "DOWN"):
        assert os.environ[f"HIPENGINE_QWEN4_EXP_MOE_DECODE_WARP_{leg}"] == "stale"
    if active:
        assert os.environ["HIPENGINE_QWEN4_EXP_Q4_DP4A64"] == "0"


def test_scope_restores_after_failure_and_preserves_peer():
    owner = SimpleNamespace(moe_decode_warp=frozenset({"GATE_UP", "DOWN"}))
    peer = SimpleNamespace(moe_decode_warp=owner.moe_decode_warp)
    with pytest.raises(RuntimeError):
        with runtime.moe_decode_warp(owner, frozenset()):
            assert owner.moe_decode_warp == frozenset()
            assert peer.moe_decode_warp == frozenset({"GATE_UP", "DOWN"})
            raise RuntimeError("failed arm")
    assert owner.moe_decode_warp == peer.moe_decode_warp


@pytest.mark.parametrize("function", [
    runtime.run_qwen4_exp_moe, runtime.run_qwen4_exp_dense_qsa_layer,
    runtime.run_qwen4_exp_qsa_prefill_layer, runtime.run_qwen4_exp_gdn_layer,
    runtime.Qwen4ExpGGUFResidentModelRunner.step,
    runtime.Qwen4ExpGGUFResidentModelRunner._prefill_chunk_steps,
    batch.run_qwen4_exp_batch_gdn_layer, batch.run_qwen4_exp_batch_qsa_layer,
    batch._run_layer_ffn, batch.Qwen4ExpGGUFBatchRunner._execute_step,
])
def test_forwarded_with_existing_expert_policy(function):
    calls = [n for n in ast.walk(ast.parse(textwrap.dedent(inspect.getsource(function))))
             if isinstance(n, ast.Call)
             and any(k.arg == "routed_f16_enabled" for k in n.keywords)]
    assert calls
    assert all(any(k.arg == "moe_decode_warp" for k in call.keywords) for call in calls)


@pytest.mark.parametrize("leg", ["GATE_UP", "DOWN"])
@pytest.mark.parametrize("compact", [1, 7, 12, 13])
@pytest.mark.parametrize("enabled,registered", [(True, True), (False, True), (True, False)])
def test_warp_geometry_and_registration_fallback(monkeypatch, leg, compact, enabled, registered):
    key = runtime.KernelKey("backend", "linear", "quant", "parent")
    monkeypatch.setattr(runtime, "is_registered", lambda key: registered)
    monkeypatch.setenv(f"HIPENGINE_QWEN4_EXP_MOE_DECODE_WARP_{leg}", "0" if enabled else "1")
    chosen = runtime._qwen4_exp_moe_decode_warp_key(
        key, compact, frozenset({leg}) if enabled else frozenset(), leg)
    assert (chosen != key) is (enabled and registered and compact <= 12)
    assert chosen.backend == key.backend and chosen.quant == key.quant


def test_graph_identity_tracks_selection():
    owner = object.__new__(batch.Qwen4ExpGGUFBatchRunner)
    before = owner._expert_down_policy()
    with runtime.moe_decode_warp(owner, frozenset({"DOWN"})):
        assert owner._expert_down_policy() != before
    assert owner._expert_down_policy() == before
    keys = []
    scratch = SimpleNamespace(closed=False, runtime=object(),
                              output=SimpleNamespace(ptr=1), selected=object(), routing=object())
    cache = SimpleNamespace(run=lambda key, **kwargs: keys.append(key))
    for choice in (frozenset(), frozenset({"DOWN"}), frozenset()):
        runtime.run_qwen4_exp_moe(0, {}, scratch=scratch, rows=1, hidden=2,
                                 ffn=2, experts=2, top_k=1, graph_cache=cache,
                                 graph_key="layer", moe_decode_warp=choice)
    assert keys[0] == keys[2] and keys[0] != keys[1]


def test_diagnostic_candidates_use_owner_state():
    from scripts.qwen4exp_layer2_profile_gate import CANDIDATES, _f16_route_isolation_candidates
    for candidate in _f16_route_isolation_candidates().values():
        assert candidate.moe_decode_warp == frozenset()
        assert not any("MOE_DECODE_WARP" in k for k in candidate.environment)
    candidate = CANDIDATES["decode_dp4a_linear"]
    assert candidate.moe_decode_warp == frozenset({"DOWN"})
    assert not any("MOE_DECODE_WARP" in k for k in candidate.environment)
