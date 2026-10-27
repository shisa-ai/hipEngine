"""The Q5_1 exact grouped down chain gate is request-local runner state.

``HIPENGINE_QWEN4_EXP_EXACT_GROUPED_DOWN`` was the chain's own process-wide
gate: it decided whether the exact grouped Q5_1 down chain ran at all, so it
also decided whether every Q5_1 ladder rung inside that chain (down M1, fold128,
folded pair, register cache, row publish) was reachable. It is the owning
runner's ``q51_exact_grouped_down`` field now:

- ``True``  - the exact grouped Q5_1 down chain, production for every multi-row
  prefill chunk.
- ``False`` - skip the chain and fall through to the sorted/WMMA down owners
  below it. That is the diagnostic arm the F16 down candidates ride.

The Q4 gate's own switch and the arithmetic kernels are untouched: this unit
moves the chain gate's transport, not the chain's contents. These tests pin the
field, the scoped control, the whole forwarding chain (c1 decode, prefill
chunks, the batch/mixed layers, serving clones), the batch capture key, the
surviving direct fallback, and that no process key carries the gate.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
from pathlib import Path
import sys
import textwrap
from types import SimpleNamespace

import pytest

from hipengine.execution_profiles import ExecutionProfile, resolve_runtime_profile
from hipengine.generation import qwen4_exp_ar_batch, qwen4_exp_gguf
from hipengine.generation.qwen4_exp_profiles import (
    QWEN4_EXP_BACKEND,
    QWEN4_EXP_MODEL,
    register_qwen4_exp_gfx1151_profiles,
)
from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
from hipengine.runtime import qwen4_exp_batch_runner as batch_module
from hipengine.runtime import qwen4_exp_runner as runner_module

RETIRED_ENV = "HIPENGINE_QWEN4_EXP_EXACT_GROUPED_DOWN"
FIELD = "q51_exact_grouped_down"
ROOT = Path(__file__).resolve().parents[1]

#: Every module that used to read or write the retired key.
RETIRED_KEY_FREE_MODULES = (
    ROOT / "hipengine" / "runtime" / "qwen4_exp_runner.py",
    ROOT / "hipengine" / "runtime" / "qwen4_exp_batch_runner.py",
    ROOT / "hipengine" / "generation" / "qwen4_exp_profiles.py",
    ROOT / "hipengine" / "generation" / "qwen4_exp_gguf.py",
    ROOT / "scripts" / "qwen4exp_halo_box_campaign_ab.py",
    ROOT / "scripts" / "qwen4exp_layer2_profile_gate.py",
    ROOT / "scripts" / "qwen4exp_row4_state_gate.py",
)


@pytest.fixture(autouse=True)
def isolate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(RETIRED_ENV, raising=False)


# --- the decision and its scoped control ------------------------------------


def test_runner_defaults_to_the_exact_grouped_chain() -> None:
    runner = object.__new__(runner_module.Qwen4ExpGGUFResidentModelRunner)
    assert getattr(runner, FIELD) is True


def test_scope_binds_the_fallback_chain_and_restores() -> None:
    runner = SimpleNamespace(**{FIELD: True})

    with getattr(runner_module, FIELD)(runner, False) as prior:
        assert prior is True
        assert getattr(runner, FIELD) is False
        with getattr(runner_module, FIELD)(runner, True):
            assert getattr(runner, FIELD) is True
        assert getattr(runner, FIELD) is False
    assert getattr(runner, FIELD) is True


def test_scope_leaves_an_unset_binding_alone() -> None:
    runner = SimpleNamespace(**{FIELD: True})

    with getattr(runner_module, FIELD)(runner):
        assert getattr(runner, FIELD) is True
    assert getattr(runner, FIELD) is True


def test_scope_restores_after_an_exception() -> None:
    runner = SimpleNamespace(**{FIELD: True})

    with pytest.raises(RuntimeError, match="boom"):
        with getattr(runner_module, FIELD)(runner, False):
            raise RuntimeError("boom")
    assert getattr(runner, FIELD) is True


def test_scope_is_per_runner() -> None:
    first = SimpleNamespace(**{FIELD: True})
    second = SimpleNamespace(**{FIELD: True})

    with getattr(runner_module, FIELD)(first, False):
        assert getattr(second, FIELD) is True
        with getattr(runner_module, FIELD)(second, False):
            assert getattr(first, FIELD) is False
            assert getattr(second, FIELD) is False
        assert getattr(second, FIELD) is True
    assert getattr(first, FIELD) is True


# --- the dispatch -----------------------------------------------------------


def test_dispatch_reads_the_gate_from_its_parameter() -> None:
    parameters = inspect.signature(runner_module.run_qwen4_exp_moe).parameters
    assert FIELD in parameters
    assert parameters[FIELD].default is True

    source = inspect.getsource(runner_module.run_qwen4_exp_moe)
    assert RETIRED_ENV not in source
    # The chain predicate still names its own variable and now reads the
    # parameter; the Q5_1 quant and rows>=2 guards stay at the dispatch site.
    assert "and q51_exact_grouped_down" in source
    assert 'weights["expert_down"].spec.quant_key == "gguf_q5_1"' in source


def test_the_chain_and_its_ladder_are_unchanged() -> None:
    """Closing the gate is a transport change; the arithmetic is not touched."""

    source = inspect.getsource(runner_module.run_qwen4_exp_moe)

    # Both chain branches survive: gate/up and down.
    assert source.count("elif exact_grouped_down:") == 2
    # The Q5_1 ladder inside the chain is untouched, so every rung stays
    # reachable while the gate is open.
    for rung in ("q51_iu8_exact", "q51_fold128_prefill", "q51_fold_pair_prefill",
                 "q51_register_cache", "q51_row_publish"):
        assert rung in source, rung
    # The other EXACT_GROUPED_Q4 gates are out of this unit's scope.
    assert "HIPENGINE_QWEN4_EXP_EXACT_GROUPED_Q4" in source


#: The functions that eventually dispatch the MoE from a layer body. The batch
#: layers reach it through ``_run_layer_ffn``; the c1 layers call it directly.
MOE_FORWARDERS = ("run_qwen4_exp_moe", "_run_layer_ffn")


def _calls_named(source: str, names) -> list[ast.Call]:
    tree = ast.parse(textwrap.dedent(source))
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in names
    ]


def _matching_calls(source: str) -> list[ast.Call]:
    """Every call to one of the layer owners (the MoE forwarding sites)."""

    tree = ast.parse(textwrap.dedent(source))
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id.startswith("run_qwen4_exp_")
        and node.func.id.endswith("layer")
    ]


def _gate_keyword(call: ast.Call) -> ast.expr | None:
    for keyword in call.keywords:
        if keyword.arg == FIELD:
            return keyword.value
    return None


@pytest.mark.parametrize(
    "function",
    [
        runner_module.run_qwen4_exp_moe,
        runner_module.run_qwen4_exp_dense_qsa_layer,
        runner_module.run_qwen4_exp_qsa_prefill_layer,
        runner_module.run_qwen4_exp_gdn_layer,
        batch_module.run_qwen4_exp_batch_gdn_layer,
        batch_module.run_qwen4_exp_batch_qsa_layer,
        batch_module._run_layer_ffn,
    ],
)
def test_every_forwarder_carries_the_gate(function) -> None:
    """Each layer that forwards MoE decisions forwards this one too."""

    parameters = inspect.signature(function).parameters
    assert FIELD in parameters, function.__qualname__
    assert parameters[FIELD].default is True, function.__qualname__

    # Every one of these forwards the MoE call, including the graph capture's
    # ``eager`` closure: a captured graph must carry the owner's decision rather
    # than silently reopening the chain inside it.
    calls = _calls_named(inspect.getsource(function), MOE_FORWARDERS)
    assert calls, f"{function.__qualname__} forwards no MoE call"

    for call in calls:
        value = _gate_keyword(call)
        assert isinstance(value, ast.Name), (function.__qualname__, ast.dump(call))
        assert value.id == FIELD


def test_the_runner_call_sites_forward_the_field() -> None:
    """c1 decode and both prefill paths read the runner's own decision."""

    source = inspect.getsource(runner_module.Qwen4ExpGGUFResidentModelRunner)
    calls = _matching_calls(source)
    assert len(calls) == 4

    for call in calls:
        value = _gate_keyword(call)
        assert isinstance(value, ast.Attribute), ast.dump(call)
        assert value.attr == FIELD


def test_the_batch_call_sites_forward_the_field() -> None:
    source = inspect.getsource(batch_module.Qwen4ExpGGUFBatchRunner)
    calls = _matching_calls(source)
    assert len(calls) == 2

    for call in calls:
        value = _gate_keyword(call)
        assert isinstance(value, ast.Attribute), ast.dump(call)
        assert value.attr == FIELD


def test_the_batch_capture_key_carries_the_gate() -> None:
    """A captured multi-row segment bakes the chain in; its key must say so."""

    owner = SimpleNamespace(
        q51_down_m1=False,
        q51_fold128_prefill=False,
        q51_fold_pair_prefill=False,
        q51_exact_grouped_down=True,
        q51_out8=True,
        forkb_grouped_down=False,
        q8_down_row4_prefill=False,
        q8_down_bundle_prefill=False,
    )
    assert batch_module.Qwen4ExpGGUFBatchRunner._expert_down_policy(owner) == (
        False, False, False, True, True, False, False, False)

    owner.q51_exact_grouped_down = False
    assert batch_module.Qwen4ExpGGUFBatchRunner._expert_down_policy(owner) == (
        False, False, False, False, True, False, False, False)


def test_the_generator_default_keeps_the_chain_on() -> None:
    """An unbound generator publishes the chain, unlike the production-only rungs."""

    source = textwrap.dedent(
        inspect.getsource(qwen4_exp_gguf.Qwen4ExpGGUFTextGenerator.__init__))
    defaults = {}
    for node in ast.walk(ast.parse(source)):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Attribute)
                and isinstance(node.targets[0].value, ast.Name)
                and node.targets[0].value.id == "self"
                and isinstance(node.value, ast.Constant)):
            defaults[node.targets[0].attr] = node.value.value

    assert defaults[FIELD] is True
    # The rungs inside the chain keep their production-only defaults.
    assert defaults["q51_down_m1"] is False
    assert defaults["q51_fold128_prefill"] is False


def test_the_ar_batch_constructor_forwards_the_gate() -> None:
    """A batch owner built before the bind still runs the production chain."""

    source = textwrap.dedent(
        inspect.getsource(qwen4_exp_ar_batch.make_qwen4_exp_ar_batch_runner))
    assignments = {}
    for node in ast.walk(ast.parse(source)):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Attribute)
                and isinstance(node.targets[0].value, ast.Name)
                and node.targets[0].value.id == "batch"):
            assignments[node.targets[0].attr] = node.value

    value = assignments[FIELD]
    assert isinstance(value, ast.Call) and value.func.id == "bool"
    fallback = value.args[0]
    assert isinstance(fallback, ast.Call) and fallback.func.id == "getattr"
    assert fallback.args[0].id == "generator"
    assert fallback.args[1].value == FIELD
    assert fallback.args[2].value is True
    # The production-only rungs keep their False fallback.
    assert assignments["q51_down_m1"].args[0].args[2].value is False


# --- publication and inheritance -------------------------------------------


def _resolve(profile: ExecutionProfile):
    return resolve_runtime_profile(
        model=QWEN4_EXP_MODEL,
        backend=QWEN4_EXP_BACKEND,
        quant="gguf_ud_q4_k_xl",
        profile=profile,
    )


@pytest.fixture
def profiles():
    register_gfx1151_kernels(replace=True)
    register_qwen4_exp_gfx1151_profiles()
    return None


@pytest.mark.parametrize(
    "profile", [ExecutionProfile.STRICT, ExecutionProfile.PRODUCTION])
def test_the_binder_publishes_the_field_and_writes_no_process_key(
    profiles, profile
) -> None:
    resolved = _resolve(profile)
    assert resolved.binder is not None
    existing = SimpleNamespace()
    resolved.binder(existing, resolved)

    assert getattr(existing, FIELD) is True
    assert RETIRED_ENV not in __import__("os").environ


def test_the_binder_reaches_a_runner_it_finds(profiles) -> None:
    resolved = _resolve(ExecutionProfile.PRODUCTION)
    runner = SimpleNamespace()
    generator = SimpleNamespace(runner=runner)
    resolved.binder(generator, resolved)

    assert getattr(runner, FIELD) is True


def test_serving_clones_inherit_the_decision() -> None:
    generator = SimpleNamespace(**{FIELD: True})
    clone = SimpleNamespace()
    qwen4_exp_gguf._inherit_lane_decisions(generator, clone)
    assert getattr(clone, FIELD) is True

    pinned = SimpleNamespace(**{FIELD: False})
    pinned_clone = SimpleNamespace()
    qwen4_exp_gguf._inherit_lane_decisions(pinned, pinned_clone)
    assert getattr(pinned_clone, FIELD) is False

    # An unbound carrier keeps the production default, never the other lane's
    # value.
    default_clone = SimpleNamespace()
    qwen4_exp_gguf._inherit_lane_decisions(SimpleNamespace(), default_clone)
    assert getattr(default_clone, FIELD) is True


# --- the diagnostics --------------------------------------------------------


def _load_campaign():
    return __import__(
        "scripts.qwen4exp_halo_box_campaign_ab", fromlist=["route_scope"])


def test_the_pair2_scope_reads_the_gate_from_the_runner() -> None:
    campaign = _load_campaign()
    env = {key: "1" for key in campaign.Q51_PAIR2_PREEMPTING_ENV}

    with pytest.raises(ValueError, match=campaign.Q51_PAIR2_GATE_FIELD):
        with campaign.q51_pair2_scope(
                runner=SimpleNamespace(**{FIELD: False}), environment=env):
            pass
    # An arm that names no runner cannot prove the chain is open either.
    with pytest.raises(ValueError, match=campaign.Q51_PAIR2_GATE_FIELD):
        with campaign.q51_pair2_scope(environment=env):
            pass

    owner = SimpleNamespace(**{FIELD: True})
    with campaign.q51_pair2_scope(runner=owner, environment=env):
        assert all(env[key] == "0" for key in campaign.Q51_PAIR2_PREEMPTING_ENV)
        assert getattr(owner, FIELD) is True
    assert all(env[key] == "1" for key in campaign.Q51_PAIR2_PREEMPTING_ENV)


def test_the_pair2_control_record_names_the_field_and_its_value() -> None:
    campaign = _load_campaign()
    env = {key: "1" for key in campaign.Q51_PAIR2_PREEMPTING_ENV}
    owner = SimpleNamespace(**{FIELD: True})

    record = campaign.q51_pair2_control_record(owner, environment=env)
    assert record["pair2_gate"] == {
        "runner_field": campaign.Q51_PAIR2_GATE_FIELD,
        "published_value": True,
    }
    assert campaign.q51_pair2_control_record(
        environment=env)["pair2_gate"]["published_value"] is None


def test_the_pf13_gate_list_no_longer_names_the_retired_key() -> None:
    campaign = _load_campaign()
    assert RETIRED_ENV not in campaign.PF13_GATE_ENV
    assert campaign.PF13_GATE_FIELD == campaign.Q51_PAIR2_GATE_FIELD


def test_the_pf13_scope_refuses_a_closed_chain_on_the_runner() -> None:
    campaign = _load_campaign()
    runner = SimpleNamespace(
        q51_exact_grouped_down=False,
        q51_down_m1=True, forkb_grouped_down=True,
        q51_fold128_prefill=True, q51_fold_pair_prefill=True,
        q51_register_cache=True, q51_row_publish=True,
        q8_down_row4_prefill=True, q8_down_bundle_prefill=True,
        q8_down_register=True)
    environment = {"HIPENGINE_QWEN4_EXP_EXACT_GROUPED_Q4": "1"}

    with pytest.raises(ValueError, match=campaign.PF13_GATE_FIELD):
        with campaign.route_scope(
                "pf13", "after", runner, environment=environment):
            pass
    # The refusal happens before anything is bound.
    assert runner.q51_down_m1 is True
    assert runner.forkb_grouped_down is True


def test_the_layer2_down_arms_close_the_chain_on_the_runner() -> None:
    spec = importlib.util.spec_from_file_location(
        "qwen4exp_layer2_profile_gate",
        ROOT / "scripts" / "qwen4exp_layer2_profile_gate.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    for name in ("f16_compact_down", "mx1_full_route"):
        candidate = module.CANDIDATES[name]
        assert candidate.q51_exact_grouped_down is False, name
        assert RETIRED_ENV not in candidate.environment, name
        # The two owners that take the down leg ahead of the F16 owner. The
        # production base binds the routed F16 route on, so an arm that closes
        # only the exact chain still measures the routed route.
        assert candidate.environment["HIPENGINE_QWEN4_EXP_ROUTED_F16"] == "0"
        assert candidate.environment["HIPENGINE_QWEN4_EXP_T16_DOWN"] == "0"
        assert candidate.environment["HIPENGINE_QWEN4_EXP_Q5_1_WMMA"] == "1"

    # The gate/up half stays a process key: this unit did not move it.
    assert module.CANDIDATES["f16_compact_gate_up"].environment[
        "HIPENGINE_QWEN4_EXP_EXACT_GROUPED_Q4"] == "0"
    assert module.CANDIDATES["f16_compact_gate_up"].q51_exact_grouped_down is None


# --- the retired key --------------------------------------------------------


def test_no_process_key_carries_the_gate() -> None:
    for path in RETIRED_KEY_FREE_MODULES:
        assert RETIRED_ENV not in path.read_text(), path
