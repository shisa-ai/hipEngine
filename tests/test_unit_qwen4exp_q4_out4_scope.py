"""The Q4 out4 fallback selector is request-local runner state.

``HIPENGINE_QWEN4_EXP_Q4_OUT4`` was the process-wide transport for one rung of
the exact grouped Q4_K gate/up ladder: the fallback below the exact expert-grid
family, which chose the out4 rowbatch8 owner over the strict base. It is the
owning runner's ``q4_out4`` field now:

- ``True``  - the out4 rowbatch8 owner
  (``selected_dual_grouped_rowbatch8_out4_bf16_bf16_out``), the promoted
  production fallback and the old env default.
- ``False`` - the strict rowbatch8 base
  (``selected_dual_grouped_rowbatch8_bf16_bf16_out``).

Only the transport moves. The ladder around it is untouched: the routed-F16
gate/up owner above the chain, the chain's own ``EXACT_GROUPED_Q4`` /
``EXACT_GROUPED_Q4_ALL`` switches, the exact iu8-risk+repair chain and its
``Q4_IU8_EXACT`` switch, the ``EXACT_EXPERT_GRID`` family (whose
``out4_expertgrid64`` variant and the pair2/bundle preemptors inside it do not
consult this decision), and the owners below the chain. The rung lives in
``_qwen4_exp_q4_gate_up_key``, which the dispatch calls, so the selection is
exercisable on CPU without a device.

The final block is the engagement regression. It publishes the production plan
through the real binder and reads the competitors the shipped lane actually runs
(the routed-F16 gate/up owner, the exact iu8 chain), then applies the route
ablation's own closure and counts launches through the registry. The chain's
gate is an environment switch here, so it is required and recorded rather than
forced: the strict base is also the owner the dispatch falls through to below a
closed chain.
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

RETIRED_ENV = "HIPENGINE_QWEN4_EXP_Q4_OUT4"
FIELD = "q4_out4"
EXPERT_GRID_ENV = "HIPENGINE_QWEN4_EXP_EXACT_EXPERT_GRID"
IU8_EXACT_ENV = "HIPENGINE_QWEN4_EXP_Q4_IU8_EXACT"
T16_GATE_UP_ENV = "HIPENGINE_QWEN4_EXP_T16_GATE_UP"
PAIR_ENV = "HIPENGINE_QWEN4_EXP_Q4_PAIR_PREFILL"
BUNDLE_ENV = "HIPENGINE_QWEN4_EXP_Q4_BUNDLE_PREFILL"
ROUTED_F16_ENV = "HIPENGINE_QWEN4_EXP_ROUTED_F16"
PRODUCTION_MOE_ENV = "HIPENGINE_QWEN4_EXP_PRODUCTION_MOE_PREFILL"
GATE_ENV = "HIPENGINE_QWEN4_EXP_EXACT_GROUPED_Q4"
GATE_ALL_ENV = "HIPENGINE_QWEN4_EXP_EXACT_GROUPED_Q4_ALL"
GRID64_VARIANT = "selected_dual_grouped_rowbatch8_out4_expertgrid64_bf16_bf16_out"
OUT4_VARIANT = "selected_dual_grouped_rowbatch8_out4_bf16_bf16_out"
BASE_VARIANT = "selected_dual_grouped_rowbatch8_bf16_bf16_out"
PAIR2_VARIANT = "selected_dual_grouped_pair2_bf16_bf16_out"
BUNDLE_VARIANT = (
    "selected_dual_grouped_rowbatch8_out4_expertgrid64_bundle_bf16_bf16_out"
)
ROOT = Path(__file__).resolve().parents[1]

#: Every module that used to read the retired key, plus the ones that now carry
#: the field.
RETIRED_KEY_FREE_MODULES = (
    ROOT / "hipengine" / "runtime" / "qwen4_exp_runner.py",
    ROOT / "hipengine" / "runtime" / "qwen4_exp_batch_runner.py",
    ROOT / "hipengine" / "runtime" / "qwen4_exp_mixed_target.py",
    ROOT / "hipengine" / "generation" / "qwen4_exp_profiles.py",
    ROOT / "hipengine" / "generation" / "qwen4_exp_gguf.py",
    ROOT / "hipengine" / "generation" / "qwen4_exp_ar_batch.py",
    ROOT / "scripts" / "qwen4exp_r2e_route_ablation.py",
    ROOT / "scripts" / "qwen4exp_layer2_profile_gate.py",
)

#: Every process key that keeps the rung - or the chain it sits in - from
#: running, in dispatch order.
COMPETITOR_ENV = (
    ROUTED_F16_ENV, PRODUCTION_MOE_ENV, IU8_EXACT_ENV, T16_GATE_UP_ENV,
    EXPERT_GRID_ENV, PAIR_ENV, BUNDLE_ENV,
)

@pytest.fixture(autouse=True)
def isolate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(RETIRED_ENV, raising=False)


# --- the decision and its scoped control ------------------------------------


def test_runner_defaults_to_the_out4_fallback() -> None:
    runner = object.__new__(runner_module.Qwen4ExpGGUFResidentModelRunner)
    assert getattr(runner, FIELD) is True


def test_scope_binds_the_strict_base_and_restores() -> None:
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


# --- the dispatch and the selector ------------------------------------------


def test_the_dispatch_forwards_the_field_into_the_selector() -> None:
    parameters = inspect.signature(runner_module.run_qwen4_exp_moe).parameters
    assert FIELD in parameters
    assert parameters[FIELD].default is True

    source = inspect.getsource(runner_module.run_qwen4_exp_moe)
    assert RETIRED_ENV not in source
    calls = [
        node for node in ast.walk(ast.parse(textwrap.dedent(source)))
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", None) == "_qwen4_exp_q4_gate_up_key"
    ]
    assert len(calls) == 1, "the dispatch no longer builds the Q4 gate/up key"
    value = _keyword(calls[0], FIELD)
    assert isinstance(value, ast.Name) and value.id == FIELD


def test_the_selector_reads_the_field_and_no_process_key() -> None:
    parameters = inspect.signature(
        runner_module._qwen4_exp_q4_gate_up_key).parameters
    assert FIELD in parameters
    assert parameters[FIELD].default is True

    source = inspect.getsource(runner_module._qwen4_exp_q4_gate_up_key)
    assert RETIRED_ENV not in source
    assert "elif q4_out4:" in source
    assert source.index(EXPERT_GRID_ENV) < source.index("elif q4_out4:")


def _variant_constants(source: str, name: str) -> list[str]:
    found = [
        (node.lineno, node.value.value)
        for node in ast.walk(ast.parse(textwrap.dedent(source)))
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == name
            and isinstance(node.value, ast.Constant))
    ]
    return [value for _line, value in sorted(found)]


def test_the_ladder_around_the_rung_is_unchanged() -> None:
    source = inspect.getsource(runner_module._qwen4_exp_q4_gate_up_key)

    # The expert-grid family and the two preemptors inside it survive.
    assert EXPERT_GRID_ENV in source
    for rung in ("qwen4_exp_q4_pair_prefill_selected",
                 "qwen4_exp_q4_bundle_prefill_selected"):
        assert rung in source, rung
    # The rung's own arms are the out4 owner and, below it, the strict base.
    assert _variant_constants(source, "variant") == [
        GRID64_VARIANT, OUT4_VARIANT, BASE_VARIANT, PAIR2_VARIANT,
        BUNDLE_VARIANT]


def test_the_chain_switches_still_sit_above_the_selector() -> None:
    dispatch = inspect.getsource(runner_module.run_qwen4_exp_moe)
    assert "_qwen4_exp_q4_exact_grouped_chain(" in dispatch
    # The chain's own switches are the owning runner's fields now, not process
    # keys.
    assert "q4_exact_grouped_gate=q4_exact_grouped_gate" in dispatch
    assert "q4_exact_grouped_all=q4_exact_grouped_all" in dispatch
    assert GATE_ENV not in dispatch
    assert GATE_ALL_ENV not in dispatch
    assert IU8_EXACT_ENV in inspect.getsource(
        runner_module._qwen4_exp_q4_iu8_exact_enabled)
    assert dispatch.index("elif exact_grouped_q4_gate:") < dispatch.index(
        "_qwen4_exp_q4_gate_up_key(")


def test_the_q5_1_ladder_still_reads_its_own_expert_grid_rung() -> None:
    """``EXACT_EXPERT_GRID`` is shared; this unit moves only the Q4 read."""

    assert EXPERT_GRID_ENV in inspect.getsource(
        runner_module._qwen4_exp_q51_down_key)
    assert "HIPENGINE_QWEN4_EXP_Q51_IU8_EXACT" in inspect.getsource(
        runner_module._qwen4_exp_q51_iu8_exact_enabled)


def test_every_arm_resolves_through_the_registry() -> None:
    """A counted wrapper on any of these keys sees the dispatch's launch."""

    source = inspect.getsource(runner_module.run_qwen4_exp_moe)
    assert "grouped_q4_key = _qwen4_exp_q4_gate_up_key(" in source
    assert "variant=grouped_q4_key.variant)" in source
    assert not hasattr(runner_module, "_Q4_GATE_UP_LAUNCHERS")
    assert not hasattr(runner_module, "_qwen4_exp_q4_gate_up_launcher")


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


#: The functions that eventually dispatch the MoE from a layer body.
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
def test_every_forwarder_carries_the_field(function) -> None:
    parameters = inspect.signature(function).parameters
    assert FIELD in parameters, function.__qualname__
    assert parameters[FIELD].default is True, function.__qualname__

    calls = _calls_named(inspect.getsource(function), MOE_FORWARDERS)
    assert calls, f"{function.__qualname__} forwards no MoE call"
    for call in calls:
        value = _keyword(call, FIELD)
        assert isinstance(value, ast.Name), (function.__qualname__, ast.dump(call))
        assert value.id == FIELD


#: The parameters the captured-segment closure re-passes; everything else is
#: launch plumbing that the recursive call rebuilds itself.
MOE_PLUMBING = frozenset({
    "mixed_ptr", "weights", "scratch", "rows", "hidden", "ffn", "experts",
    "top_k", "stream", "runtime", "graph_cache", "graph_key",
})


def test_the_captured_segment_closure_forwards_every_decision() -> None:
    """The c1 graph closure re-enters the dispatch; it must not drop a field."""

    parameters = inspect.signature(runner_module.run_qwen4_exp_moe).parameters
    decisions = set(parameters) - MOE_PLUMBING
    assert FIELD in decisions

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


def test_the_runner_call_sites_forward_the_field() -> None:
    source = inspect.getsource(runner_module.Qwen4ExpGGUFResidentModelRunner)
    calls = _layer_calls(source)
    assert len(calls) == 4

    for call in calls:
        value = _keyword(call, FIELD)
        assert isinstance(value, ast.Attribute), ast.dump(call)
        assert value.attr == FIELD


def test_the_batch_call_sites_forward_the_field() -> None:
    source = inspect.getsource(batch_module.Qwen4ExpGGUFBatchRunner)
    calls = _layer_calls(source)
    assert len(calls) == 2

    for call in calls:
        value = _keyword(call, FIELD)
        assert isinstance(value, ast.Attribute), ast.dump(call)
        assert value.attr == FIELD


def test_the_batch_capture_key_carries_the_field() -> None:
    """A captured segment bakes the gate/up launch in; its key must say so."""

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
        False, False, False)

    owner.q4_out4 = False
    assert batch_module.Qwen4ExpGGUFBatchRunner._expert_down_policy(owner) == (
        False, False, False, False, True, False, True, True, True, False, False, False,
        False, False, False)


def test_the_mixed_pass_reads_the_field_from_the_batch_owner() -> None:
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
    for call in calls:
        value = _keyword(call, FIELD)
        assert isinstance(value, ast.Call), ast.dump(call)
        assert value.func.id == "getattr"
        assert value.args[0].id == "batch"
        assert value.args[1].value == FIELD
        assert value.args[2].value is True


# --- publication and inheritance -------------------------------------------


def test_the_generator_default_keeps_the_out4_fallback() -> None:
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


def test_serving_clones_inherit_the_decision() -> None:
    generator = SimpleNamespace(**{FIELD: True})
    clone = SimpleNamespace()
    qwen4_exp_gguf._inherit_lane_decisions(generator, clone)
    assert getattr(clone, FIELD) is True

    pinned = SimpleNamespace(**{FIELD: False})
    pinned_clone = SimpleNamespace()
    qwen4_exp_gguf._inherit_lane_decisions(pinned, pinned_clone)
    assert getattr(pinned_clone, FIELD) is False

    default_clone = SimpleNamespace()
    qwen4_exp_gguf._inherit_lane_decisions(SimpleNamespace(), default_clone)
    assert getattr(default_clone, FIELD) is True


def test_the_ar_batch_constructor_forwards_the_field() -> None:
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
    assert RETIRED_ENV not in os.environ


def test_the_binder_reaches_a_runner_it_finds(profiles) -> None:
    resolved = _resolve(ExecutionProfile.PRODUCTION)
    runner = SimpleNamespace()
    generator = SimpleNamespace(runner=runner)
    resolved.binder(generator, resolved)

    assert getattr(runner, FIELD) is True


# --- the diagnostics --------------------------------------------------------


def test_the_profile_gate_can_bind_the_rung() -> None:
    import dataclasses

    from scripts import qwen4exp_layer2_profile_gate as gate

    spec = {field.name: field for field in dataclasses.fields(gate.CandidateSpec)}
    assert spec[FIELD].default is None
    source = inspect.getsource(gate)
    assert f"candidate_spec.{FIELD} is not None" in source
    assert f"enter_context({FIELD}(" in source


# --- the production publication, the closure, and the launches --------------


@contextlib.contextmanager
def _published(monkeypatch):
    """Publish the production plan through the real binder and report its keys."""

    for name in COMPETITOR_ENV:
        monkeypatch.delenv(name, raising=False)
    resolved = _resolve(ExecutionProfile.PRODUCTION)
    generator = SimpleNamespace(runner=SimpleNamespace())
    resolved.binder(generator, resolved)
    yield {name: os.environ.get(name) for name in COMPETITOR_ENV}, generator


def _gate_up_key(**kwargs):
    base = dict(backend=QWEN4_EXP_BACKEND, quant="gguf_q4_k", rows=1024,
                in_features=512)
    base.update(kwargs)
    return runner_module._qwen4_exp_q4_gate_up_key(**base)


def _launch(key) -> None:
    """Resolve and call the owner exactly as the dispatch does."""

    runner_module.resolve(
        backend=key.backend, layer=key.layer, quant=key.quant,
        variant=key.variant)()


@pytest.fixture
def counting_stubs():
    """Count launches by replacing the registry entries, then put them back."""

    from hipengine.kernels.registry import (
        KernelKey, is_registered, register, resolve, unregister,
    )

    saved: dict = {}
    added: list = []

    def make(variants) -> dict[str, int]:
        counts = {variant: 0 for variant in variants}
        for variant in variants:
            key = KernelKey(
                QWEN4_EXP_BACKEND, "moe_linear", "gguf_q4_k", variant)
            if is_registered(key):
                saved[key] = resolve(
                    backend=key.backend, layer=key.layer, quant=key.quant,
                    variant=key.variant)
            else:
                added.append(key)

            def stub(*_args, _variant=variant, **_kwargs):
                counts[_variant] += 1

            register(key, stub, replace=True)
        return counts

    yield make
    for key, prior in saved.items():
        register(key, prior, replace=True)
    for key in added:
        unregister(key)


def test_the_route_closes_every_owner_the_dispatch_checks_first() -> None:
    from scripts import qwen4exp_r2e_route_ablation as module

    bindings, envs, off = module.RUNNER_ROUTE_OFF_WITH_ENV["q4_out4"]
    assert dict(bindings)[FIELD] is False
    assert off == "0"
    assert set(envs) == {
        ROUTED_F16_ENV, PRODUCTION_MOE_ENV, IU8_EXACT_ENV, T16_GATE_UP_ENV,
        EXPERT_GRID_ENV, PAIR_ENV}
    assert dict(bindings)["q4_bundle_prefill"] is False
    assert envs == module.Q4_OUT4_PREEMPTING_ENV
    # The chain's own switches are runner state, so the arm records and
    # requires them instead of forcing them.
    assert module.RUNNER_ROUTE_REQUIRED_FIELDS["q4_out4"] == {
        "q4_exact_grouped_gate": True, "q4_exact_grouped_all": True}
    assert not hasattr(module, "RUNNER_ROUTE_REQUIRED_ENV")
    assert "q4_out4" not in module.ROUTES
    assert "q4_out4" not in module.RUNNER_ROUTE_OFF
    for field, _value in bindings:
        assert callable(getattr(runner_module, field)), field


def test_the_production_publication_keeps_the_rung_from_deciding(
        profiles, monkeypatch) -> None:
    with _published(monkeypatch) as (published, generator):
        assert published[ROUTED_F16_ENV] == "1"
        assert published[IU8_EXACT_ENV] == "1"
        assert published[PAIR_ENV] == "1"
        assert published[BUNDLE_ENV] is None
        assert generator.runner.q4_bundle_prefill is True
        assert published[EXPERT_GRID_ENV] is None   # default 64
        # The chain's own switches are runner state, published on every plan.
        assert generator.runner.q4_exact_grouped_gate is True
        assert generator.runner.q4_exact_grouped_all is True

        # The real predicates, in dispatch order: the routed-F16 owner takes the
        # gate/up leg, and the iu8 chain sits above the ladder.
        assert runner_module._routed_f16_enabled() is True
        assert runner_module._qwen4_exp_q4_iu8_exact_enabled() is True
        # Even with those out of the way the expert-grid family replaces the
        # rung at its default mode, and the published pair2 preemptor inside it
        # takes the launch - so neither of this rung's owners can run.
        assert _gate_up_key(q4_out4=True).variant == PAIR2_VARIANT
        assert _gate_up_key(q4_out4=False).variant == PAIR2_VARIANT


def test_the_route_closure_makes_the_strict_base_the_owner(
        profiles, monkeypatch) -> None:
    from scripts import qwen4exp_r2e_route_ablation as module

    bindings, envs, off = module.RUNNER_ROUTE_OFF_WITH_ENV["q4_out4"]
    runner = SimpleNamespace()
    with _published(monkeypatch) as (_published_keys, generator):
        for name in envs:
            monkeypatch.setenv(name, off)
        with contextlib.ExitStack() as stack:
            for field, value in bindings:
                stack.enter_context(
                    getattr(runner_module, field)(runner, value))
            for rows in (32, 1024):
                assert _gate_up_key(rows=rows, q4_out4=False).variant == (
                    BASE_VARIANT), rows
                assert _gate_up_key(rows=rows, q4_out4=True).variant == (
                    OUT4_VARIANT), rows
            # The pair2/bundle preemptors are gated on the expert-grid family, so
            # publishing them cannot replace this rung's owner while the family
            # is closed.
            monkeypatch.setenv(PAIR_ENV, "1")
            monkeypatch.setenv(BUNDLE_ENV, "1")
            assert _gate_up_key(q4_out4=False).variant == BASE_VARIANT
            assert _gate_up_key(q4_out4=True).variant == OUT4_VARIANT
            # The gates the arm records are still published on the runner.
            assert generator.runner.q4_exact_grouped_gate is True
            assert generator.runner.q4_exact_grouped_all is True


def test_the_route_closure_launches_only_the_owner_it_names(
        profiles, monkeypatch, counting_stubs) -> None:
    """Launch counting through the real registry, not a field value."""

    from scripts import qwen4exp_r2e_route_ablation as module

    bindings, envs, off = module.RUNNER_ROUTE_OFF_WITH_ENV["q4_out4"]
    counts = counting_stubs(
        (GRID64_VARIANT, OUT4_VARIANT, BASE_VARIANT, PAIR2_VARIANT,
         BUNDLE_VARIANT))
    runner = SimpleNamespace()
    with _published(monkeypatch) as (_published_keys, _generator):
        # The lane's own arm first: the expert-grid family owns the launch (its
        # published pair2 preemptor at 1024 rows), so the rung's owners stay at
        # zero however the field is set.
        _launch(_gate_up_key(q4_out4=False))
        assert counts == {
            GRID64_VARIANT: 0, OUT4_VARIANT: 0, BASE_VARIANT: 0,
            PAIR2_VARIANT: 1, BUNDLE_VARIANT: 0}

        for name in envs:
            monkeypatch.setenv(name, off)
        # Default arm: the published rung.
        with contextlib.ExitStack() as stack:
            for field, value in bindings:
                if field == FIELD:
                    continue
                stack.enter_context(
                    getattr(runner_module, field)(runner, value))
            _launch(_gate_up_key(q4_out4=getattr(runner, FIELD, True)))
        assert counts == {
            GRID64_VARIANT: 0, OUT4_VARIANT: 1, BASE_VARIANT: 0,
            PAIR2_VARIANT: 1, BUNDLE_VARIANT: 0}

        # Ablated arm: the shipped binding, and only the strict base launches.
        with contextlib.ExitStack() as stack:
            for field, value in bindings:
                stack.enter_context(
                    getattr(runner_module, field)(runner, value))
            _launch(_gate_up_key(q4_out4=getattr(runner, FIELD, True)))
        assert counts == {
            GRID64_VARIANT: 0, OUT4_VARIANT: 1, BASE_VARIANT: 1,
            PAIR2_VARIANT: 1, BUNDLE_VARIANT: 0}


# --- the retired key --------------------------------------------------------


def test_no_process_key_carries_the_decision() -> None:
    for path in RETIRED_KEY_FREE_MODULES:
        assert RETIRED_ENV not in path.read_text(), path
