"""The Q5_1 out8 fallback selector is request-local runner state.

``HIPENGINE_QWEN4_EXP_Q5_1_OUT8`` was the process-wide transport for one rung of
the exact grouped Q5_1 down ladder: the fallback below the exact expert-grid
family, which chose the out8 rowbatch8 owner over the output1 one. It is the
owning runner's ``q51_out8`` field now:

- ``True``  - the out8 rowbatch8 owner (``..._rowbatch8_out8_bf16_bf16_out``),
  the promoted production fallback and the old env default.
- ``False`` - the output1 rowbatch8 owner (``..._rowbatch8_bf16_bf16_out``), the
  strict base of that rung.

Only the transport moves. The ladder around it is untouched: the exact
iu8-risk+repair chain and the ``Q51_IU8_EXACT`` switch above it, the
``EXACT_EXPERT_GRID`` family (whose ``out8_expertgrid64`` and
``out8_expertgrid64_m1`` variants do not consult this decision at all), the pair2
preemptor, the fold128/folded-pair/register-cache/row-publish siblings below it,
and the ``Q4_OUT4`` switch on the Q4 gate/up ladder. The rung itself lives in
``_qwen4_exp_q51_down_key``, which the dispatch calls and then resolves, so the
selection is exercisable on CPU without a device.

The final block is the engagement regression. It publishes the production plan
through the real binder and reads the competitors the shipped lane actually runs
(the routed F16 down owner, the exact iu8 chain, the pair2 parent), then applies
the route ablation's own closure - the campaign's ``Q51_PAIR2_PREEMPTING_ENV``
set plus the pair2 and expert-grid keys, and the rung plus its four fold siblings
on the runner - and counts launches through the registry. A three-key fixture
would have hidden the routed-F16 owner the lane takes first, and a field-value
assertion would have accepted the output1 owner that a closed chain falls
through to *below* the chain, so the chain gate is required and recorded rather
than bound.
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

RETIRED_ENV = "HIPENGINE_QWEN4_EXP_Q5_1_OUT8"
FIELD = "q51_out8"
EXPERT_GRID_ENV = "HIPENGINE_QWEN4_EXP_EXACT_EXPERT_GRID"
IU8_EXACT_ENV = "HIPENGINE_QWEN4_EXP_Q51_IU8_EXACT"
Q4_OUT4_ENV = "HIPENGINE_QWEN4_EXP_Q4_OUT4"
OUT8_VARIANT = "selected_grouped_prefill_compact_rowbatch8_out8_bf16_bf16_out"
OUT1_VARIANT = "selected_grouped_prefill_compact_rowbatch8_bf16_bf16_out"
M1_VARIANT = (
    "selected_grouped_prefill_compact_rowbatch8_out8_expertgrid64_m1_bf16_bf16_out"
)
GRID64_VARIANT = (
    "selected_grouped_prefill_compact_rowbatch8_out8_expertgrid64_bf16_bf16_out"
)
PAIR2_VARIANT = "selected_grouped_prefill_pair2_bf16_bf16_out"
ROOT = Path(__file__).resolve().parents[1]

#: Every module that used to read or write the retired key, plus the ones that
#: now carry the field.
RETIRED_KEY_FREE_MODULES = (
    ROOT / "hipengine" / "runtime" / "qwen4_exp_runner.py",
    ROOT / "hipengine" / "runtime" / "qwen4_exp_batch_runner.py",
    ROOT / "hipengine" / "runtime" / "qwen4_exp_mixed_target.py",
    ROOT / "hipengine" / "generation" / "qwen4_exp_profiles.py",
    ROOT / "hipengine" / "generation" / "qwen4_exp_gguf.py",
    ROOT / "hipengine" / "generation" / "qwen4_exp_ar_batch.py",
    ROOT / "scripts" / "qwen4exp_r2e_route_ablation.py",
)


@pytest.fixture(autouse=True)
def isolate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(RETIRED_ENV, raising=False)


# --- the decision and its scoped control ------------------------------------


def test_runner_defaults_to_the_out8_fallback() -> None:
    runner = object.__new__(runner_module.Qwen4ExpGGUFResidentModelRunner)
    assert getattr(runner, FIELD) is True


def test_scope_binds_the_output1_base_and_restores() -> None:
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


def test_the_dispatch_forwards_the_field_into_the_selector() -> None:
    parameters = inspect.signature(runner_module.run_qwen4_exp_moe).parameters
    assert FIELD in parameters
    assert parameters[FIELD].default is True

    source = inspect.getsource(runner_module.run_qwen4_exp_moe)
    assert RETIRED_ENV not in source
    calls = [
        node for node in ast.walk(ast.parse(textwrap.dedent(source)))
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", None) == "_qwen4_exp_q51_down_key"
    ]
    assert len(calls) == 1, "the dispatch no longer builds the Q5_1 down key"
    value = _keyword(calls[0], FIELD)
    assert isinstance(value, ast.Name) and value.id == FIELD
    # The owner it resolves is the selector's own answer, not a second ladder.
    assert "resolve(\n                        backend=grouped_q5_key.backend" in source


def test_the_selector_reads_the_field_and_no_process_key() -> None:
    parameters = inspect.signature(
        runner_module._qwen4_exp_q51_down_key).parameters
    assert FIELD in parameters
    assert parameters[FIELD].default is True

    source = inspect.getsource(runner_module._qwen4_exp_q51_down_key)
    assert RETIRED_ENV not in source
    # The rung still reads the parameter and still sits below the expert-grid
    # family and above the pair2 preemptor.
    assert "elif q51_out8:" in source
    assert source.index(EXPERT_GRID_ENV) < source.index("elif q51_out8:")
    assert source.index("elif q51_out8:") < source.index(
        "qwen4_exp_q51_pair_prefill_selected(")


def _variant_constants(source: str, name: str) -> list[str]:
    """The constant values assigned to ``name``, in source order."""

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
    """This unit moves one rung's transport; the ladder is not touched."""

    source = inspect.getsource(runner_module._qwen4_exp_q51_down_key)

    # The expert-grid family keeps both of its out8 variants and the M1 rung.
    literals = {
        node.value for node in ast.walk(ast.parse(textwrap.dedent(source)))
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert GRID64_VARIANT in literals
    assert M1_VARIANT in literals
    assert EXPERT_GRID_ENV in source
    # The pair2 preemptor and every fold sibling survive.
    for rung in ("qwen4_exp_q51_pair_prefill_selected", "q51_fold128_prefill",
                 "q51_fold_pair_prefill", "q51_register_cache",
                 "q51_row_publish"):
        assert rung in source, rung
    # The rung's constant arms are the out8 owner, below it the output1 base,
    # and the pair2 preemptor that replaces either of them.
    assert _variant_constants(source, "variant") == [
        OUT8_VARIANT, OUT1_VARIANT, PAIR2_VARIANT]


def test_the_exact_iu8_chain_still_sits_above_the_selector() -> None:
    """The chain that preempts the whole ladder is the caller's, not the rung's."""

    source = inspect.getsource(runner_module.run_qwen4_exp_moe)
    assert "q51_iu8_exact" in source
    assert IU8_EXACT_ENV in inspect.getsource(
        runner_module._qwen4_exp_q51_iu8_exact_enabled)
    # The selector is reached only in the chain's ``else`` arm.
    assert source.index("elif exact_grouped_down:") < source.index(
        "_qwen4_exp_q51_down_key(")


def test_the_q4_side_and_its_out4_switch_are_out_of_scope() -> None:
    """``EXACT_EXPERT_GRID`` is shared; the Q4 rung moved to its own selector."""

    dispatch = inspect.getsource(runner_module.run_qwen4_exp_moe)
    selector = inspect.getsource(runner_module._qwen4_exp_q51_down_key)
    q4_selector = inspect.getsource(runner_module._qwen4_exp_q4_gate_up_key)
    assert dispatch.count(EXPERT_GRID_ENV) == 0   # both ladders read it locally
    assert selector.count(EXPERT_GRID_ENV) == 1   # the Q5_1 down rung
    assert q4_selector.count(EXPERT_GRID_ENV) == 1  # the Q4 gate/up rung
    # The Q4 rung's own switch is the owning runner's field now, not a process
    # key: no module reads the retired name.
    assert Q4_OUT4_ENV not in dispatch


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
        runner_module.run_qwen4_exp_moe,
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
    """A captured multi-row segment bakes the rung in; its key must say so."""

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
    )
    assert batch_module.Qwen4ExpGGUFBatchRunner._expert_down_policy(owner) == (
        False, False, False, True, False, True, True, True, True, False, False,
        False)

    owner.q51_out8 = False
    assert batch_module.Qwen4ExpGGUFBatchRunner._expert_down_policy(owner) == (
        False, False, False, True, False, False, True, True, True, False, False,
        False)


def test_the_mixed_pass_reads_the_field_from_the_batch_owner() -> None:
    from hipengine.runtime.qwen4_exp_mixed_target import Qwen4ExpMixedTargetPass

    tree = ast.parse(textwrap.dedent(
        inspect.getsource(Qwen4ExpMixedTargetPass.verify)))
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


def test_the_generator_default_keeps_the_out8_fallback() -> None:
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

    # An unbound carrier keeps the production default, never the other lane's
    # value.
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
    assert RETIRED_ENV not in __import__("os").environ


def test_the_binder_reaches_a_runner_it_finds(profiles) -> None:
    resolved = _resolve(ExecutionProfile.PRODUCTION)
    runner = SimpleNamespace()
    generator = SimpleNamespace(runner=runner)
    resolved.binder(generator, resolved)

    assert getattr(runner, FIELD) is True


# --- the diagnostics --------------------------------------------------------


def test_the_profile_gate_can_bind_the_rung() -> None:
    """The gate's candidates name request-local leaves; this one is nameable."""

    import dataclasses

    from scripts import qwen4exp_layer2_profile_gate as gate

    spec = {field.name: field for field in dataclasses.fields(gate.CandidateSpec)}
    assert spec[FIELD].default is None
    source = inspect.getsource(gate)
    assert f"candidate_spec.{FIELD} is not None" in source
    assert f"enter_context({FIELD}(" in source


ROUTED_F16_ENV = "HIPENGINE_QWEN4_EXP_ROUTED_F16"
PRODUCTION_MOE_ENV = "HIPENGINE_QWEN4_EXP_PRODUCTION_MOE_PREFILL"
T16_DOWN_ENV = "HIPENGINE_QWEN4_EXP_T16_DOWN"
#: The retired process switch for the F16 WMMA down owner below the chain;
#: nothing reads it, and the tests keep the literal to prove it inert.
WMMA_ENV = "HIPENGINE_QWEN4_EXP_Q5_1_WMMA"
PAIR_ENV = "HIPENGINE_QWEN4_EXP_Q51_PAIR_PREFILL"
GATE_FIELD = "q51_exact_grouped_down"
FOLD_FIELDS = (
    "q51_fold128_prefill", "q51_fold_pair_prefill", "q51_register_cache",
    "q51_row_publish",
)
#: Every process key that keeps the rung - or the chain it sits in - from
#: running, as the dispatch orders them.
COMPETITOR_ENV = (
    ROUTED_F16_ENV, PRODUCTION_MOE_ENV, T16_DOWN_ENV, IU8_EXACT_ENV,
    WMMA_ENV, PAIR_ENV, EXPERT_GRID_ENV,
)


@contextlib.contextmanager
def _published(monkeypatch):
    """Publish the production plan through the real binder and report its keys.

    The shipped lane's competitors are not the code defaults: the production plan
    publishes the routed F16 down owner, the exact iu8 chain and the pair2 parent
    on, and pins the Q5_1 WMMA owner off. An arm that closed only the three keys
    a fixture happened to set would leave the lane's own owners in place.
    """

    for name in COMPETITOR_ENV:
        monkeypatch.delenv(name, raising=False)
    resolved = _resolve(ExecutionProfile.PRODUCTION)
    resolved.binder(SimpleNamespace(), resolved)
    yield {name: os.environ.get(name) for name in COMPETITOR_ENV}


def _down_key(**kwargs):
    base = dict(backend=QWEN4_EXP_BACKEND, quant="gguf_q5_1", rows=1024,
                in_features=512)
    base.update(kwargs)
    return runner_module._qwen4_exp_q51_down_key(**base)


def _launch(key) -> None:
    """Resolve and call the owner exactly as the dispatch does."""

    from hipengine.kernels.registry import resolve

    resolve(backend=key.backend, layer=key.layer, quant=key.quant,
            variant=key.variant)()


def _counting_stubs(variants) -> dict[str, int]:
    from hipengine.kernels.registry import KernelKey, register

    counts = {variant: 0 for variant in variants}
    for variant in variants:
        def stub(*_args, _variant=variant, **_kwargs):
            counts[_variant] += 1

        register(KernelKey(QWEN4_EXP_BACKEND, "moe_linear", "gguf_q5_1", variant),
                 stub, replace=True)
    return counts


def test_the_route_closes_every_owner_the_dispatch_checks_first() -> None:
    from scripts import qwen4exp_halo_box_campaign_ab as campaign
    from scripts import qwen4exp_r2e_route_ablation as module

    bindings, envs, off = module.RUNNER_ROUTE_OFF_WITH_ENV["q51_out8"]
    assert dict(bindings)[FIELD] is False
    assert off == "0"
    # The five owners the campaign already closes for the pair2 parent, plus the
    # pair2 selection itself and the expert-grid family.
    assert set(envs) == set(campaign.Q51_PAIR2_PREEMPTING_ENV) | {
        PAIR_ENV, EXPERT_GRID_ENV}
    assert envs == module.OUT8_PREEMPTING_ENV
    # The fold siblings are held off too, even though their guards require the
    # pair2 parent variant.
    assert set(FOLD_FIELDS) <= {field for field, _value in bindings}
    assert all(value is False for _field, value in bindings)
    # The chain's own gate is recorded and required, never bound: the output1
    # base is also the owner below a closed chain.
    assert GATE_FIELD not in dict(bindings)
    assert module.RUNNER_ROUTE_REQUIRED_FIELDS["q51_out8"] == {GATE_FIELD: True}
    assert "q51_out8" not in module.ROUTES
    assert "q51_out8" not in module.RUNNER_ROUTE_OFF
    for field, _value in bindings:
        assert callable(getattr(runner_module, field)), field


def test_the_production_publication_keeps_the_rung_from_deciding(
        profiles, monkeypatch) -> None:
    """The shipped lane's own owners, read from the binder, not from a fixture."""

    with _published(monkeypatch) as published:
        assert published[ROUTED_F16_ENV] == "1"
        assert published[IU8_EXACT_ENV] == "1"
        assert published[PAIR_ENV] == "1"
        # The F16 WMMA down owner below the chain is runner state now, so the
        # plan publishes it off through the field and names no process key.
        assert published[WMMA_ENV] is None
        assert published[T16_DOWN_ENV] is None      # default off
        assert published[EXPERT_GRID_ENV] is None   # default 64

        # The real predicates, in dispatch order. Routed F16 takes the whole
        # down leg, and the iu8 chain sits above the ladder inside the chain.
        assert runner_module._routed_f16_enabled() is True
        assert runner_module._qwen4_exp_q51_iu8_exact_enabled() is True
        # With those two out of the way the pair2 parent still replaces the rung
        # on a production chunk, so the rung decides nothing either way.
        assert runner_module.qwen4_exp_q51_pair_prefill_selected(
            QWEN4_EXP_BACKEND, "gguf_q5_1", rows=1024, in_features=512) is True
        assert _down_key(q51_out8=True).variant == PAIR2_VARIANT
        assert _down_key(q51_out8=False).variant == PAIR2_VARIANT


def test_the_route_closure_makes_the_output1_base_the_owner(
        profiles, monkeypatch) -> None:
    from scripts import qwen4exp_r2e_route_ablation as module

    bindings, envs, off = module.RUNNER_ROUTE_OFF_WITH_ENV["q51_out8"]
    runner = SimpleNamespace(**{GATE_FIELD: True})
    with _published(monkeypatch):
        for name in envs:
            monkeypatch.setenv(name, off)
        with contextlib.ExitStack() as stack:
            for field, value in bindings:
                stack.enter_context(
                    getattr(runner_module, field)(runner, value))
            # The rung itself, on the production chunk width and on a small one.
            for rows in (32, 1024):
                assert _down_key(rows=rows, q51_out8=False).variant == (
                    OUT1_VARIANT), rows
                assert _down_key(rows=rows, q51_out8=True).variant == (
                    OUT8_VARIANT), rows
            # The fold siblings are variant-gated on the pair2 parent, so a plan
            # that publishes them cannot replace this rung's owner - which is
            # why the route's own bindings are defensive rather than load
            # bearing.
            assert _down_key(q51_out8=False, **{f: True for f in FOLD_FIELDS}
                             ).variant == OUT1_VARIANT
            assert _down_key(q51_out8=True, **{f: True for f in FOLD_FIELDS}
                             ).variant == OUT8_VARIANT
            # And the chain gate the arm records is still published.
            assert getattr(runner, GATE_FIELD) is True


def test_the_route_closure_launches_only_the_owner_it_names(
        profiles, monkeypatch) -> None:
    """Launch counting through the real registry, not a field value."""

    from scripts import qwen4exp_r2e_route_ablation as module

    bindings, envs, off = module.RUNNER_ROUTE_OFF_WITH_ENV["q51_out8"]
    counts = _counting_stubs(
        (OUT8_VARIANT, OUT1_VARIANT, PAIR2_VARIANT, GRID64_VARIANT, M1_VARIANT))
    runner = SimpleNamespace(**{GATE_FIELD: True})
    with _published(monkeypatch):
        # The lane's own arm first: the launch reaches the pair2 parent, so the
        # rung's owners stay at zero however the field is set.
        _launch(_down_key(q51_out8=False))
        assert counts == {
            OUT8_VARIANT: 0, OUT1_VARIANT: 0, PAIR2_VARIANT: 1,
            GRID64_VARIANT: 0, M1_VARIANT: 0}

        for name in envs:
            monkeypatch.setenv(name, off)
        # Default arm: the published rung, with the route's other bindings in
        # force so the comparison isolates this decision.
        with contextlib.ExitStack() as stack:
            for field, value in bindings:
                if field == FIELD:
                    continue
                stack.enter_context(
                    getattr(runner_module, field)(runner, value))
            # The field the runner carries, as the dispatch reads it.
            _launch(_down_key(q51_out8=getattr(runner, FIELD, True)))
        assert counts == {
            OUT8_VARIANT: 1, OUT1_VARIANT: 0, PAIR2_VARIANT: 1,
            GRID64_VARIANT: 0, M1_VARIANT: 0}

        # Ablated arm: the shipped binding, and only the output1 owner launches.
        with contextlib.ExitStack() as stack:
            for field, value in bindings:
                stack.enter_context(
                    getattr(runner_module, field)(runner, value))
            _launch(_down_key(q51_out8=getattr(runner, FIELD, True)))
        assert counts == {
            OUT8_VARIANT: 1, OUT1_VARIANT: 1, PAIR2_VARIANT: 1,
            GRID64_VARIANT: 0, M1_VARIANT: 0}


# --- the retired key --------------------------------------------------------


def test_no_process_key_carries_the_decision() -> None:
    for path in RETIRED_KEY_FREE_MODULES:
        assert RETIRED_ENV not in path.read_text(), path
