"""The Q4 exact grouped chain gate and its `_ALL` joint term are runner state.

``HIPENGINE_QWEN4_EXP_EXACT_GROUPED_Q4`` and
``HIPENGINE_QWEN4_EXP_EXACT_GROUPED_Q4_ALL`` were the process-wide transport for
the exact grouped Q4_K gate/up chain:

- ``q4_exact_grouped_gate`` (old ``..._Q4``) - the chain's own gate. ``True``
  runs the exact grouped Q4_K gate/up chain, ``False`` falls through to the
  plain-iu8 or F16 compact owner below it.
- ``q4_exact_grouped_all`` (old ``..._Q4_ALL``) - the joint term. The predicate
  is ``gate and (exact_grouped_down or all)``, so `_ALL` only matters where the
  exact grouped Q5_1 down chain is off: with it on the Q4 chain runs on every
  Q4_K multi-row chunk, with it off the chain is restricted to chunks the Q5_1
  down chain already owns.

Every plan publishes both on. The rest of the predicate is dispatch-time state
and is untouched: the ``production_grouped_moe`` preemptor, ``rows >= 2`` and the
Q4_K/Q4_K quant pair. No kernel arithmetic moves; the strict fallback (the F16
compact owner) stays registered and reachable.

The last block pins the diagnostics: the layer2 candidates close the gate
through the candidate spec instead of an environment key, the r2e ablation
records both as required runner fields (checked before any measured work), and
PF13 requires both chain gates on the runner it measures.
"""

from __future__ import annotations

import ast
import contextlib
import inspect
import os
from pathlib import Path
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

GATE_ENV = "HIPENGINE_QWEN4_EXP_EXACT_GROUPED_Q4"
ALL_ENV = "HIPENGINE_QWEN4_EXP_EXACT_GROUPED_Q4_ALL"
GATE_FIELD = "q4_exact_grouped_gate"
ALL_FIELD = "q4_exact_grouped_all"
ROOT = Path(__file__).resolve().parents[1]

RETIRED_KEY_FREE_MODULES = (
    ROOT / "hipengine" / "runtime" / "qwen4_exp_runner.py",
    ROOT / "hipengine" / "runtime" / "qwen4_exp_batch_runner.py",
    ROOT / "hipengine" / "runtime" / "qwen4_exp_mixed_target.py",
    ROOT / "hipengine" / "generation" / "qwen4_exp_profiles.py",
    ROOT / "hipengine" / "generation" / "qwen4_exp_gguf.py",
    ROOT / "hipengine" / "generation" / "qwen4_exp_ar_batch.py",
    ROOT / "scripts" / "qwen4exp_r2e_route_ablation.py",
    ROOT / "scripts" / "qwen4exp_layer2_profile_gate.py",
    ROOT / "scripts" / "qwen4exp_halo_box_campaign_ab.py",
    ROOT / "scripts" / "qwen4exp_disabled_selector_ranking.py",
)


@pytest.fixture(autouse=True)
def isolate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(GATE_ENV, raising=False)
    monkeypatch.delenv(ALL_ENV, raising=False)


# --- the decisions and their scoped controls --------------------------------


@pytest.mark.parametrize("field", [GATE_FIELD, ALL_FIELD])
def test_the_runner_defaults_both_on(field: str) -> None:
    runner = object.__new__(runner_module.Qwen4ExpGGUFResidentModelRunner)
    assert getattr(runner, field) is True


@pytest.mark.parametrize("field", [GATE_FIELD, ALL_FIELD])
def test_the_scope_binds_and_restores(field: str) -> None:
    runner = SimpleNamespace(**{field: True})

    with getattr(runner_module, field)(runner, False) as prior:
        assert prior is True
        assert getattr(runner, field) is False
        with getattr(runner_module, field)(runner, True):
            assert getattr(runner, field) is True
        assert getattr(runner, field) is False
    assert getattr(runner, field) is True


@pytest.mark.parametrize("field", [GATE_FIELD, ALL_FIELD])
def test_the_scope_leaves_an_unset_binding_alone(field: str) -> None:
    runner = SimpleNamespace(**{field: False})

    with getattr(runner_module, field)(runner):
        assert getattr(runner, field) is False
    assert getattr(runner, field) is False


@pytest.mark.parametrize("field", [GATE_FIELD, ALL_FIELD])
def test_the_scope_restores_after_an_exception(field: str) -> None:
    runner = SimpleNamespace(**{field: True})

    with pytest.raises(RuntimeError, match="boom"):
        with getattr(runner_module, field)(runner, False):
            raise RuntimeError("boom")
    assert getattr(runner, field) is True


def test_the_two_scopes_are_independent() -> None:
    runner = SimpleNamespace(q4_exact_grouped_gate=True, q4_exact_grouped_all=True)

    with runner_module.q4_exact_grouped_gate(runner, False):
        assert runner.q4_exact_grouped_all is True
        with runner_module.q4_exact_grouped_all(runner, False):
            assert runner.q4_exact_grouped_gate is False
            assert runner.q4_exact_grouped_all is False
        assert runner.q4_exact_grouped_all is True
    assert runner.q4_exact_grouped_gate is True


# --- the joint predicate ----------------------------------------------------


def _chain(**overrides) -> bool:
    values = dict(
        production_grouped_moe=False,
        rows=16,
        gate_quant="gguf_q4_k",
        up_quant="gguf_q4_k",
        exact_grouped_gate=True,
        exact_grouped_all=True,
        exact_grouped_down=False,
    )
    values.update(overrides)
    return runner_module._qwen4_exp_q4_exact_grouped_chain(**values)


def test_the_predicate_runs_the_chain_by_default() -> None:
    assert _chain() is True


def test_the_gate_alone_closes_the_chain() -> None:
    assert _chain(exact_grouped_gate=False) is False
    # A closed gate dominates every other term, including `_ALL` and the down
    # chain.
    assert _chain(
        exact_grouped_gate=False, exact_grouped_all=True,
        exact_grouped_down=True) is False


def test_the_all_term_only_matters_where_the_down_chain_is_off() -> None:
    # `_ALL` off with the down chain off: the chain is restricted away.
    assert _chain(exact_grouped_all=False, exact_grouped_down=False) is False
    # `_ALL` off but the down chain owns the chunk: the chain still runs.
    assert _chain(exact_grouped_all=False, exact_grouped_down=True) is True
    # `_ALL` on: the chain runs whether or not the down chain does.
    assert _chain(exact_grouped_all=True, exact_grouped_down=False) is True
    assert _chain(exact_grouped_all=True, exact_grouped_down=True) is True


def test_the_row_and_quant_guards_are_not_runner_decisions() -> None:
    assert _chain(rows=1) is False
    assert _chain(rows=2) is True
    assert _chain(gate_quant="gguf_q5_k") is False
    assert _chain(up_quant="gguf_q8_0") is False


def test_the_production_grouped_owner_preempts_the_chain() -> None:
    assert _chain(production_grouped_moe=True) is False


def test_the_dispatch_calls_the_predicate_with_the_real_guards() -> None:
    source = inspect.getsource(runner_module.run_qwen4_exp_moe)
    assert "exact_grouped_q4_gate = _qwen4_exp_q4_exact_grouped_chain(" in source
    for keyword in (
        "production_grouped_moe=production_grouped_moe",
        "rows=rows",
        'gate_quant=weights["expert_gate"].spec.quant_key',
        'up_quant=weights["expert_up"].spec.quant_key',
        "exact_grouped_gate=q4_exact_grouped_gate",
        "exact_grouped_all=q4_exact_grouped_all",
        "exact_grouped_down=exact_grouped_down",
    ):
        assert keyword in source, keyword
    # No process key reads survive in the predicate's caller.
    assert GATE_ENV not in source
    assert ALL_ENV not in source


# --- forwarding -------------------------------------------------------------


def _calls_named(source: str, names) -> list[ast.Call]:
    tree = ast.parse(textwrap.dedent(source))
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in names
    ]


def _keyword(call: ast.Call, name: str) -> ast.expr | None:
    for keyword in call.keywords:
        if keyword.arg == name:
            return keyword.value
    return None


MOE_FORWARDERS = ("run_qwen4_exp_moe", "_run_layer_ffn")


@pytest.mark.parametrize(
    "function",
    [
        runner_module.run_qwen4_exp_dense_qsa_layer,
        runner_module.run_qwen4_exp_qsa_prefill_layer,
        runner_module.run_qwen4_exp_gdn_layer,
        batch_module.run_qwen4_exp_batch_gdn_layer,
        batch_module.run_qwen4_exp_batch_qsa_layer,
        batch_module._run_layer_ffn,
    ],
)
@pytest.mark.parametrize("field", [GATE_FIELD, ALL_FIELD])
def test_every_forwarder_carries_both_fields(function, field: str) -> None:
    parameters = inspect.signature(function).parameters
    assert field in parameters, function.__qualname__
    assert parameters[field].default is True, function.__qualname__

    calls = _calls_named(inspect.getsource(function), MOE_FORWARDERS)
    assert calls, f"{function.__qualname__} forwards no MoE call"
    for call in calls:
        value = _keyword(call, field)
        assert isinstance(value, ast.Name), (function.__qualname__, ast.dump(call))
        assert value.id == field


#: The parameters the captured-segment closure re-passes; everything else is
#: launch plumbing that the recursive call rebuilds itself.
MOE_PLUMBING = frozenset({
    "mixed_ptr", "weights", "scratch", "rows", "hidden", "ffn", "experts",
    "top_k", "stream", "runtime", "graph_cache", "graph_key",
})


def test_the_captured_segment_closure_forwards_every_decision() -> None:
    parameters = inspect.signature(runner_module.run_qwen4_exp_moe).parameters
    decisions = set(parameters) - MOE_PLUMBING
    assert {GATE_FIELD, ALL_FIELD} <= decisions

    source = inspect.getsource(runner_module.run_qwen4_exp_moe)
    calls = [
        node for node in ast.walk(ast.parse(textwrap.dedent(source)))
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", None) == "run_qwen4_exp_moe"
    ]
    assert len(calls) == 1, "the captured-segment closure is gone"
    forwarded = {keyword.arg for keyword in calls[0].keywords}
    assert decisions - forwarded == set(), decisions - forwarded


def _layer_calls(source: str) -> list[ast.Call]:
    tree = ast.parse(textwrap.dedent(source))
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id.startswith("run_qwen4_exp_")
        and node.func.id.endswith("layer")
    ]


@pytest.mark.parametrize("field", [GATE_FIELD, ALL_FIELD])
def test_the_runner_call_sites_forward_the_field(field: str) -> None:
    source = inspect.getsource(runner_module.Qwen4ExpGGUFResidentModelRunner)
    calls = _layer_calls(source)
    assert len(calls) == 4
    for call in calls:
        value = _keyword(call, field)
        assert isinstance(value, ast.Attribute), ast.dump(call)
        assert value.attr == field


@pytest.mark.parametrize("field", [GATE_FIELD, ALL_FIELD])
def test_the_batch_call_sites_forward_the_field(field: str) -> None:
    source = inspect.getsource(batch_module.Qwen4ExpGGUFBatchRunner)
    calls = _layer_calls(source)
    assert len(calls) == 2
    for call in calls:
        value = _keyword(call, field)
        assert isinstance(value, ast.Attribute), ast.dump(call)
        assert value.attr == field


def test_the_batch_capture_key_carries_both_gates() -> None:
    owner = SimpleNamespace(
        q51_down_m1=False,
        q51_fold128_prefill=False,
        q51_fold_pair_prefill=False,
        q51_exact_grouped_down=True,
        q51_wmma_down=False,
        q51_out8=True,
        q4_exact_grouped_gate=True,
        q4_exact_grouped_all=True,
        q4_out4=True,
        forkb_grouped_down=False,
        q8_down_row4_prefill=False,
        q8_down_bundle_prefill=False,
        q5k_row4_prefill=False,
    )
    assert batch_module.Qwen4ExpGGUFBatchRunner._expert_down_policy(owner) == (
        False, False, False, False, True, False, True, True, True, True, False, False,
        False, False, False, False)

    owner.q4_exact_grouped_gate = False
    assert batch_module.Qwen4ExpGGUFBatchRunner._expert_down_policy(owner) == (
        False, False, False, False, True, False, True, False, True, True, False, False,
        False, False, False, False)

    owner.q4_exact_grouped_gate = True
    owner.q4_exact_grouped_all = False
    assert batch_module.Qwen4ExpGGUFBatchRunner._expert_down_policy(owner) == (
        False, False, False, False, True, False, True, True, False, True, False, False,
        False, False, False, False)


def test_the_mixed_pass_reads_both_gates_from_the_batch_owner() -> None:
    from hipengine.runtime.qwen4_exp_mixed_target import Qwen4ExpMixedTargetPass

    # ``verify`` is the execution entry and binds the batch's Q8_0 tile pair
    # around the row work; the forwarding itself lives in ``_verify_rows``.
    assert "self._verify_rows()" in inspect.getsource(
        Qwen4ExpMixedTargetPass.verify)
    tree = ast.parse(textwrap.dedent(
        inspect.getsource(Qwen4ExpMixedTargetPass._verify_rows)))
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", None) == "_run_layer_ffn"
    ]
    assert calls
    for field in (GATE_FIELD, ALL_FIELD):
        values = [
            _keyword(call, field) for call in calls
        ]
        assert values and all(value is not None for value in values), field
        for value in values:
            assert isinstance(value, ast.Call), ast.dump(value)
            assert value.func.id == "getattr"
            assert value.args[0].id == "batch"
            assert value.args[1].value == field
            assert value.args[2].value is True


# --- publication and inheritance -------------------------------------------


@pytest.mark.parametrize("field", [GATE_FIELD, ALL_FIELD])
def test_the_generator_defaults_keep_both_on(field: str) -> None:
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
    assert defaults[field] is True


@pytest.mark.parametrize("field", [GATE_FIELD, ALL_FIELD])
def test_serving_clones_inherit_the_decisions(field: str) -> None:
    generator = SimpleNamespace(**{field: False})
    clone = SimpleNamespace()
    qwen4_exp_gguf._inherit_lane_decisions(generator, clone)
    assert getattr(clone, field) is False

    default_clone = SimpleNamespace()
    qwen4_exp_gguf._inherit_lane_decisions(SimpleNamespace(), default_clone)
    assert getattr(default_clone, field) is True


@pytest.mark.parametrize("field", [GATE_FIELD, ALL_FIELD])
def test_the_ar_batch_constructor_forwards_the_field(field: str) -> None:
    source = textwrap.dedent(
        inspect.getsource(qwen4_exp_ar_batch.make_qwen4_exp_ar_batch_runner))
    assignments = {}
    for node in ast.walk(ast.parse(source)):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Attribute)
                and isinstance(node.targets[0].value, ast.Name)
                and node.targets[0].value.id == "batch"):
            assignments[node.targets[0].attr] = node.value

    value = assignments[field]
    assert isinstance(value, ast.Call) and value.func.id == "bool"
    fallback = value.args[0]
    assert isinstance(fallback, ast.Call) and fallback.func.id == "getattr"
    assert fallback.args[0].id == "generator"
    assert fallback.args[1].value == field
    assert fallback.args[2].value is True


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
def test_the_binder_publishes_both_fields_and_writes_no_process_key(
    profiles, profile, monkeypatch
) -> None:
    resolved = _resolve(profile)
    assert resolved.binder is not None
    generator = SimpleNamespace(runner=SimpleNamespace())
    resolved.binder(generator, resolved)

    assert generator.q4_exact_grouped_gate is True
    assert generator.q4_exact_grouped_all is True
    assert generator.runner.q4_exact_grouped_gate is True
    assert generator.runner.q4_exact_grouped_all is True
    # The retired transport is gone: the binder no longer writes the keys.
    assert GATE_ENV not in os.environ
    assert ALL_ENV not in os.environ


# --- the diagnostics --------------------------------------------------------


def _gate_module():
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(
        "qwen4exp_layer2_profile_gate",
        ROOT / "scripts" / "qwen4exp_layer2_profile_gate.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("field", [GATE_FIELD, ALL_FIELD])
def test_the_profile_gate_can_bind_both_decisions(field: str) -> None:
    import dataclasses

    module = _gate_module()
    spec = {entry.name: entry for entry in dataclasses.fields(module.CandidateSpec)}
    assert spec[field].default is None
    source = inspect.getsource(module)
    assert f"candidate_spec.{field} is not None" in source
    assert f"enter_context({field}(" in source


def test_the_gate_up_candidates_close_the_chain_through_the_spec() -> None:
    module = _gate_module()

    for name in ("repair_free_gate_up", "f16_compact_gate_up", "mx1_full_route"):
        candidate = module.CANDIDATES[name]
        assert candidate.q4_exact_grouped_gate is False, name
        assert GATE_ENV not in candidate.environment, name
        # `_ALL` is not part of any of these arms: the gate alone closes the
        # chain, and overriding the joint term would be a second change.
        assert candidate.q4_exact_grouped_all is None, name
        assert ALL_ENV not in candidate.environment, name


def test_the_down_only_candidate_leaves_the_gate_up_chain_alone() -> None:
    module = _gate_module()
    candidate = module.CANDIDATES["f16_compact_down"]

    assert candidate.q4_exact_grouped_gate is None
    assert candidate.q4_exact_grouped_all is None
    assert GATE_ENV not in candidate.environment
    assert ALL_ENV not in candidate.environment


def test_the_ablation_records_both_gates_as_required_runner_fields() -> None:
    from scripts import qwen4exp_r2e_route_ablation as module

    assert module.RUNNER_ROUTE_REQUIRED_FIELDS["q4_out4"] == {
        GATE_FIELD: True, ALL_FIELD: True}
    # The process-key prerequisite mechanism this replaced is gone rather than
    # left empty.
    assert not hasattr(module, "RUNNER_ROUTE_REQUIRED_ENV")
    assert not hasattr(module, "Q4_GATE_ENV")
    assert not hasattr(module, "Q4_GATE_ALL_ENV")


def test_the_ablation_checks_its_prerequisites_before_measured_work() -> None:
    from scripts import qwen4exp_r2e_route_ablation as module

    source = inspect.getsource(module.main)
    check = source.index("required runner field not published")
    teacher = source.index("teacher = np.load")
    assert check < teacher, "the gate check must precede the first measured read"
    assert "getattr(generator.runner, field, None) is value" in source


def test_pf13_requires_both_chain_gates_on_the_runner() -> None:
    from scripts.qwen4exp_halo_box_campaign_ab import (
        PF13_GATE_FIELD, PF13_GATE_FIELDS, pf13_scope,
    )

    assert PF13_GATE_FIELD == "q51_exact_grouped_down"
    assert PF13_GATE_FIELDS == ("q51_exact_grouped_down", GATE_FIELD)
    source = inspect.getsource(pf13_scope)
    assert "for field in PF13_GATE_FIELDS:" in source
    assert "PF13_GATE_ENV" not in source
    # The gates are read on the runner, not from the environment mapping.
    assert "getattr(runner, field, False)" in source


def test_pf13_refuses_a_closed_q4_gate() -> None:
    from scripts.qwen4exp_halo_box_campaign_ab import pf13_scope

    runner = SimpleNamespace(
        q51_exact_grouped_down=True,
        q4_exact_grouped_gate=False,
        q4_exact_grouped_all=True,
        q51_down_m1=True, forkb_grouped_down=True,
        q51_fold128_prefill=True, q51_fold_pair_prefill=True,
        q51_register_cache=True, q51_row_publish=True,
        q8_down_row4_prefill=True, q8_down_bundle_prefill=True,
        q8_down_register=True)
    with pytest.raises(ValueError, match=GATE_FIELD):
        with pf13_scope("after", runner, environment={}):
            pass
    # The refusal happens before anything is bound.
    assert runner.q51_down_m1 is True


def test_the_ranking_harness_carries_no_inert_gate_toggle() -> None:
    """The chain gate is runner state; an env-only arm cannot clear it."""

    source = (ROOT / "scripts" / "qwen4exp_disabled_selector_ranking.py").read_text()
    assert GATE_ENV not in source
    assert ALL_ENV not in source
    # The note points at the runner binding that actually reaches the owner.
    assert "q4_exact_grouped_gate" in source


# --- the retired keys -------------------------------------------------------


@pytest.mark.parametrize("key", [GATE_ENV, ALL_ENV])
def test_no_module_reads_the_retired_keys(key: str) -> None:
    for path in RETIRED_KEY_FREE_MODULES:
        assert key not in path.read_text(), (path, key)
