"""R0443: the GR-up DPP comparison is owner state, not a process switch.

``HIPENGINE_QWEN4_EXP_GR_DPP_REDUCTION`` was the process-wide transport for one
dispatch decision: whether the fused exact GR-up read takes the DPP lane-zero
reduction sibling or keeps the wave-scale parent it was promoted over. The
migration target is an explicit owner boolean, ``gr_dpp_reduction``, published
by the profile binder from the plan's own GR-up row, bound by a scoped control,
and read by the GR-up selector.

The routes this file pins, in the order the dispatch evaluates them:

- ``gr_wide`` (rows > 256, ``low_rank % 64 == 0``) -> the wide F16 tile. It
  preempts the fused-up branch outright, so it is not part of this selector;
- ``HIPENGINE_QWEN4_EXP_GR_IU8`` (rows > 256)      -> the iu8 projection and its
  unfused epilogue. It preempts the fused-up branch the same way;
- fused up, owner off, wave-scale published        -> the wave-scale parent;
- fused up, owner on                               -> the registered DPP
  sibling, with the wave-scale parent as its first fallback;
- fused up, owner on, DPP unregistered             -> the wave-scale parent, or
  the fused exact parent when that is unregistered too.

Three contracts sit around that choice and are pinned separately:

- the DPP sibling is an exact replacement and the shuffle-tree wave-scale parent
  stays a live numerical reference, so this file never asks a template
  instantiation to disappear - only the process-key reader;
- ``gr_wave_scope`` must keep closing DPP **on the measured owner**, because a
  wave-scale comparison that leaves DPP published measures the sibling in both
  arms; the DPP package must close the two preemptors the same way;
- the plan's own manifest row is the publication source, and the UD recovery
  sweep cannot rewrite it, so no arm compares a value nothing published.
"""

from __future__ import annotations

import ast
import inspect
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from hipengine.execution_profiles import (
    ExecutionProfile,
    clear_runtime_profile_registry_for_tests,
    resolve_runtime_profile,
)
from hipengine.generation.qwen4_exp_gguf import _inherit_lane_decisions
from hipengine.generation.qwen4_exp_profiles import (
    PRODUCTION_ARITHMETIC_RECOVERY_FLAGS,
    QWEN4_EXP_BACKEND,
    QWEN4_EXP_MODEL,
    register_qwen4_exp_gfx1151_profiles,
)
from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
from hipengine.kernels.registry import KernelKey
from hipengine.runtime import qwen4_exp_runner as runner_module

#: The process key this migration retires. The literal stays so a leftover
#: value can be proven inert.
RETIRED_ENV = "HIPENGINE_QWEN4_EXP_GR_DPP_REDUCTION"
FIELD = "gr_dpp_reduction"
UD_QUANT = "gguf_ud_q4_k_xl"
GR_LAYER = "linear+gr_gated_mean"
GR_SCOPE = "qwen4exp_rows_gt256_gr_up"

BASE = "coltile2_branch4_rowbatch4_f32_exact"
WAVE = "coltile2_branch4_rowbatch4_wave_scale_f32_exact"
DPP = "coltile2_branch4_rowbatch4_wave_scale_dpp_f32_exact"
#: The specialization ``_qwen4_exp_gr_k320_full_rows_key`` binds for K=320,
#: H=2560, four branches, and complete ROW_BATCH=4 groups above 256 rows. It
#: replaces the promoted DPP sibling, so a harness counting only that sibling
#: records zero for a 512-row arm.
K320 = "coltile2_branch4_rowbatch4_wave_scale_dpp_k320_full_rows_f32_exact"

ROOT = Path(__file__).resolve().parents[1]
GR_READ = "run_qwen4_exp_gr_read"
LAYERS = (
    "run_qwen4_exp_dense_qsa_layer",
    "run_qwen4_exp_qsa_prefill_layer",
    "run_qwen4_exp_gdn_layer",
)
K320_SCRIPTS = (
    "qwen4exp_gr_k320_full_rows_engine_ab.py",
    "qwen4exp_gr_k320_full_rows_natural_owner.py",
    "qwen4exp_gr_k320_full_rows_state_gate.py",
)


@pytest.fixture(autouse=True)
def isolate(monkeypatch: pytest.MonkeyPatch) -> None:
    """No test starts with the retired switch in force."""

    monkeypatch.delenv(RETIRED_ENV, raising=False)
    clear_runtime_profile_registry_for_tests()
    yield
    clear_runtime_profile_registry_for_tests()


# --- the owner field, its resolver, and its scoped control ------------------


def test_the_runner_class_declares_the_unbound_default() -> None:
    """An unbound runner keeps the wave-scale fallback chain, not the sibling."""

    runner = runner_module.Qwen4ExpGGUFResidentModelRunner
    assert runner.__annotations__[FIELD] == "bool"
    assert getattr(runner, FIELD) is False


def test_the_resolver_reads_the_owner_field() -> None:
    resolve = runner_module.resolve_gr_dpp_reduction
    assert resolve(SimpleNamespace(**{FIELD: True})) is True
    assert resolve(SimpleNamespace(**{FIELD: False})) is False
    assert resolve(SimpleNamespace(**{FIELD: 1})) is True
    # An owner built before the field existed, or a bare diagnostic object,
    # keeps the parent route rather than raising.
    assert resolve(SimpleNamespace()) is False


def test_the_scope_binds_one_owner_and_restores_the_published_value() -> None:
    first = SimpleNamespace(**{FIELD: False})
    second = SimpleNamespace(**{FIELD: True})
    with runner_module.gr_dpp_reduction(first, True):
        assert getattr(first, FIELD) is True
        assert getattr(second, FIELD) is True
        with runner_module.gr_dpp_reduction(second, False):
            assert getattr(first, FIELD) is True
            assert getattr(second, FIELD) is False
        assert getattr(second, FIELD) is True
    assert getattr(first, FIELD) is False
    assert getattr(second, FIELD) is True


def test_the_scope_restores_the_published_value_after_an_exception() -> None:
    owner = SimpleNamespace(**{FIELD: True})
    with pytest.raises(RuntimeError, match="failed"):
        with runner_module.gr_dpp_reduction(owner, False):
            assert getattr(owner, FIELD) is False
            raise RuntimeError("failed")
    assert getattr(owner, FIELD) is True


def test_the_scope_binds_a_generator_and_its_current_runner() -> None:
    """A harness holding the generator selects the route its dispatches read."""

    generator = SimpleNamespace(
        **{FIELD: False}, runner=SimpleNamespace(**{FIELD: False}))
    with runner_module.gr_dpp_reduction(generator, True):
        assert getattr(generator, FIELD) is True
        assert getattr(generator.runner, FIELD) is True
    assert getattr(generator, FIELD) is False
    assert getattr(generator.runner, FIELD) is False


def test_the_scope_writes_no_process_key() -> None:
    owner = SimpleNamespace(**{FIELD: False})
    with runner_module.gr_dpp_reduction(owner, True):
        assert RETIRED_ENV not in os.environ
    assert RETIRED_ENV not in os.environ


# --- the scalar selector ---------------------------------------------------


def _key(variant: str) -> KernelKey:
    return KernelKey("hip_gfx1151", GR_LAYER, "gguf_q8_0", variant)


def _select(
    monkeypatch: pytest.MonkeyPatch,
    *,
    available: set[str],
    rows: int = 513,
    branches: int = 4,
    published: bool = False,
    dpp: bool = False,
) -> KernelKey:
    """Run the real selector with one registry and one owner decision.

    The parameter check is deliberate: without it a missing owner parameter
    would surface as a ``TypeError`` from this helper instead of naming the
    contract that is absent.
    """

    parameters = inspect.signature(
        runner_module._qwen4_exp_gr_wave_scale_key).parameters
    assert "dpp" in parameters, (
        "the GR-up selector still reads the retired process key instead of an "
        "owner decision")
    monkeypatch.setattr(
        runner_module, "is_registered", lambda key: key.variant in available)
    return runner_module._qwen4_exp_gr_wave_scale_key(
        _key(BASE), rows=rows, branches=branches, published=published, dpp=dpp)


@pytest.mark.parametrize("published", [False, True])
def test_an_unbound_owner_keeps_the_published_route(
    monkeypatch: pytest.MonkeyPatch, published: bool,
) -> None:
    selected = _select(
        monkeypatch, available={DPP, WAVE}, published=published)
    assert selected.variant == (WAVE if published else BASE)


@pytest.mark.parametrize("rows", [257, 513, 1024, 4096])
def test_the_owner_selects_the_dpp_sibling(
    monkeypatch: pytest.MonkeyPatch, rows: int,
) -> None:
    selected = _select(
        monkeypatch, available={DPP, WAVE}, rows=rows, dpp=True)
    assert selected.variant == DPP


def test_dpp_preempts_the_wave_scale_owner_when_both_are_published(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The production UD plan publishes both; DPP is the measured route."""

    assert _select(
        monkeypatch, available={DPP, WAVE}, published=True, dpp=True,
    ).variant == DPP


@pytest.mark.parametrize("rows,branches", [
    (64, 4),    # decode / batch width
    (256, 4),   # the row threshold itself
    (513, 2),   # branch guard
])
def test_the_shape_guards_stay_dispatch_time(
    monkeypatch: pytest.MonkeyPatch, rows: int, branches: int,
) -> None:
    selected = _select(
        monkeypatch, available={DPP, WAVE}, rows=rows, branches=branches,
        dpp=True)
    assert selected == _key(BASE)


@pytest.mark.parametrize("available,expected", [
    ({DPP, WAVE}, DPP),
    ({WAVE}, WAVE),      # DPP unregistered keeps the exact wave-scale fallback
    (set(), BASE),       # neither registered keeps the fused exact parent
])
def test_dpp_keeps_the_wave_scale_first_fallback(
    monkeypatch: pytest.MonkeyPatch, available: set[str], expected: str,
) -> None:
    assert _select(
        monkeypatch, available=available, dpp=True).variant == expected


def test_a_stale_process_key_cannot_move_the_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(RETIRED_ENV, "1")
    assert _select(
        monkeypatch, available={DPP, WAVE}, published=False).variant == BASE
    assert _select(
        monkeypatch, available={DPP, WAVE}, published=True).variant == WAVE


def test_the_selector_reads_no_environment() -> None:
    """The variable must be dead, not merely shadowed by an owner field."""

    source = inspect.getsource(runner_module._qwen4_exp_gr_wave_scale_key)
    assert "os.environ" not in source
    assert RETIRED_ENV not in source


def test_the_runner_module_has_no_reader_for_the_retired_key() -> None:
    source = Path(inspect.getfile(runner_module)).read_text()
    assert RETIRED_ENV not in source


# --- forwarding: the published field must reach the kernel choice -----------


def _calls_named(source: str, names) -> list[ast.Call]:
    tree = ast.parse(inspect.cleandoc(source))
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


def _env_writes(source: str) -> set[str]:
    """String keys this module assigns into ``os.environ``."""

    keys: set[str] = set()
    for node in ast.walk(ast.parse(inspect.cleandoc(source))):
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            targets = [node.target]
        for target in targets:
            if not isinstance(target, ast.Subscript):
                continue
            owner = target.value
            if not (isinstance(owner, ast.Attribute) and owner.attr == "environ"):
                continue
            if isinstance(target.slice, ast.Constant) and isinstance(
                    target.slice.value, str):
                keys.add(target.slice.value)
    return keys


def _attribute_writes(source: str, field: str) -> bool:
    """Whether this module assigns ``<something>.<field>``."""

    for node in ast.walk(ast.parse(inspect.cleandoc(source))):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Attribute) and target.attr == field:
                return True
    return False


def test_the_gr_read_takes_the_owner_and_passes_it_to_the_selector() -> None:
    parameters = inspect.signature(runner_module.run_qwen4_exp_gr_read).parameters
    assert FIELD in parameters, (
        f"{GR_READ} does not take the GR DPP owner")
    assert parameters[FIELD].default is False
    calls = _calls_named(
        inspect.getsource(runner_module.run_qwen4_exp_gr_read),
        ("_qwen4_exp_gr_wave_scale_key",))
    assert len(calls) == 1
    value = _keyword(calls[0], "dpp")
    assert isinstance(value, ast.Name), ast.dump(calls[0])
    assert value.id == FIELD
    # The wave-scale publication keeps its own argument beside it.
    assert _keyword(calls[0], "published") is not None


def test_every_gr_read_that_carries_the_wave_scale_owner_also_carries_the_dpp_owner() -> None:
    """One owner transport for one selector; no read keeps a partial handoff.

    The captured batch segments pass neither decision because a captured width
    tops out below the selector's row guard, so the rule is conditional on the
    wave-scale owner rather than on the call site.
    """

    runtime = ROOT / "hipengine" / "runtime"
    carriers, missing = 0, []
    for name in (
        "qwen4_exp_runner.py",
        "qwen4_exp_batch_runner.py",
        "qwen4_exp_mixed_target.py",
    ):
        for call in _calls_named(
            (runtime / name).read_text(), (GR_READ,)
        ):
            if _keyword(call, "gr_wave_scale") is None:
                continue
            carriers += 1
            if _keyword(call, FIELD) is None:
                missing.append(f"{name}:{call.lineno}")
    assert carriers >= 10, f"the GR read surface did not load ({carriers} sites)"
    assert missing == [], missing


@pytest.mark.parametrize("name", LAYERS)
def test_the_layers_take_the_owner_with_the_parent_default(
    monkeypatch: pytest.MonkeyPatch, name: str,
) -> None:
    layer = getattr(runner_module, name)
    parameters = inspect.signature(layer).parameters
    assert FIELD in parameters, f"{name} does not take the GR DPP owner"
    assert parameters[FIELD].default is False

    runtime = object()
    gr = SimpleNamespace(norm_weight_ptr=1, down=object(), up=object(), inject=None)
    weights = SimpleNamespace(attention_gr=gr, ffn_gr=gr)
    scratch = SimpleNamespace(closed=False, runtime=runtime, attention_gr=object())
    seen = []

    def read(*args, **kwargs):
        seen.append(kwargs[FIELD])
        raise RuntimeError("observed GR")

    monkeypatch.setattr(runner_module, GR_READ, read)
    kwargs = {key: 1 for key, parameter in inspect.signature(layer).parameters.items()
              if parameter.default is inspect.Parameter.empty}
    kwargs.update(weights=weights, scratch=scratch, runtime=runtime, rows=389)
    for published in (True, False):
        with pytest.raises(RuntimeError, match="observed GR"):
            layer(**kwargs, **{FIELD: published})
    assert seen == [True, False]


def test_the_runner_forwards_its_own_decision() -> None:
    for name in ("step", "_prefill_chunk_steps"):
        source = inspect.getsource(
            getattr(runner_module.Qwen4ExpGGUFResidentModelRunner, name))
        calls = _calls_named(source, LAYERS)
        assert calls, f"{name} launches no layer"
        for call in calls:
            value = _keyword(call, FIELD)
            assert isinstance(value, ast.Attribute), (name, ast.dump(call))
            assert value.attr == FIELD, (name, ast.dump(call))


def test_the_batch_owner_carries_the_decision_but_not_the_capture_key() -> None:
    from hipengine.generation.qwen4_exp_ar_batch import QWEN4_EXP_MAX_BATCH_ROWS
    from hipengine.runtime import qwen4_exp_batch_runner as batch

    assert FIELD in inspect.signature(batch._run_layer_ffn).parameters, (
        "_run_layer_ffn does not forward the GR DPP owner")
    assert FIELD in batch.Qwen4ExpGGUFBatchRunner.__annotations__, (
        "the batch runner does not carry the GR DPP owner")
    # A captured segment tops out below the selector's own row guard, so the
    # decision cannot reach a baked launch and does not belong in the key.
    assert QWEN4_EXP_MAX_BATCH_ROWS <= 256
    key = inspect.getsource(
        batch.Qwen4ExpGGUFBatchRunner._expert_down_policy)
    assert FIELD not in key and "gr_wave_scale" not in key


def test_the_batch_layer_ffn_read_forwards_its_parameter() -> None:
    """The captured FFN read forwards the argument, not a constant.

    The carrier rule pins that the read names the owner at all. This pins the
    value: a literal ``False`` there would pass that rule and silently keep the
    base parent on any future batch width above the selector's guard.
    """

    from hipengine.runtime import qwen4_exp_batch_runner as batch

    calls = _calls_named(inspect.getsource(batch._run_layer_ffn), (GR_READ,))
    assert len(calls) == 1, "the captured FFN body reads GR once"
    value = _keyword(calls[0], FIELD)
    assert isinstance(value, ast.Name), ast.dump(calls[0])
    assert value.id == FIELD, ast.dump(value)


def test_the_late_clone_inherits_the_published_owner() -> None:
    for published in (True, False):
        clone = _inherit_lane_decisions(
            SimpleNamespace(**{FIELD: published}), SimpleNamespace())
        assert getattr(clone, FIELD) is published


# --- the plan publishes the decision; no switch selects it ------------------


def _bind(
    monkeypatch: pytest.MonkeyPatch,
    profile_name,
    *,
    quant: str = UD_QUANT,
    runner_handle=None,
    manifest=None,
):
    monkeypatch.setattr(os, "environ", os.environ.copy())
    os.environ.pop(RETIRED_ENV, None)
    register_gfx1151_kernels(replace=True)
    register_qwen4_exp_gfx1151_profiles()
    profile = resolve_runtime_profile(
        model=QWEN4_EXP_MODEL, backend=QWEN4_EXP_BACKEND, quant=quant,
        profile=profile_name)
    if manifest is not None:
        profile = replace(profile, manifest=manifest)
    generator = SimpleNamespace()
    if runner_handle is not None:
        generator.runner = runner_handle
    profile.binder(generator, profile)
    return generator, profile


@pytest.mark.parametrize("profile_name,expected", [
    (ExecutionProfile.PRODUCTION, True),
    (ExecutionProfile.STRICT, False),
    (ExecutionProfile.BATCH_INVARIANT, False),
])
def test_the_binder_publishes_the_plan_decision_and_writes_no_switch(
    monkeypatch: pytest.MonkeyPatch, profile_name, expected: bool,
) -> None:
    generator, _ = _bind(monkeypatch, profile_name)
    assert getattr(generator, FIELD) is expected
    assert RETIRED_ENV not in os.environ


def test_the_binder_publishes_on_the_runner_it_finds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handle = SimpleNamespace()
    generator, _ = _bind(
        monkeypatch, ExecutionProfile.PRODUCTION, runner_handle=handle)
    assert getattr(generator, FIELD) is True
    assert getattr(handle, FIELD) is True
    assert RETIRED_ENV not in os.environ


def test_the_manifest_row_is_the_source_of_the_publication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forcing the plan's own GR-up row to the parent publishes the parent.

    The retired key was bound from a quant name; the owner is published from the
    row the resolved plan already declares, so a plan that keeps the wave-scale
    parent cannot publish the sibling.
    """

    _, resolved = _bind(monkeypatch, ExecutionProfile.PRODUCTION)
    rows = [
        dict(row, selected_variant=BASE)
        if row["layer"] == GR_LAYER and row["scope"] == GR_SCOPE
        else row
        for row in resolved.manifest["selections"]
    ]
    manifest = dict(resolved.manifest)
    manifest["selections"] = rows
    generator, _ = _bind(
        monkeypatch, ExecutionProfile.PRODUCTION, manifest=manifest)
    assert getattr(generator, FIELD) is False
    assert RETIRED_ENV not in os.environ


def test_a_row_on_an_unrelated_layer_cannot_publish_the_sibling() -> None:
    """The decision belongs to the GR-up row, not to any row with its scope.

    ``qwen4exp_rows_gt256_gr_up`` is a selection scope, so another layer can
    carry the same scope string. Only the GR-up layer's row names this
    dispatch decision; a matching variant on another layer must publish the
    parent, or an unrelated plan row would silently select the sibling.

    The positive control runs first so a failure here reports a missing layer
    predicate rather than a helper that never matches.
    """

    from hipengine.generation.qwen4_exp_profiles import (
        _gr_dpp_reduction_selected,
    )

    row = {"layer": GR_LAYER, "scope": GR_SCOPE, "selected_variant": DPP}
    assert _gr_dpp_reduction_selected({"selections": [row]}) is True
    unrelated = dict(row, layer="unrelated_linear")
    assert _gr_dpp_reduction_selected({"selections": [unrelated]}) is False


def test_a_production_request_on_a_non_ud_lane_keeps_the_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the UD plan carries the DPP row; every other lane runs strict."""

    generator, profile = _bind(
        monkeypatch, ExecutionProfile.PRODUCTION, quant="gguf_q4_k_m")
    assert profile.binder.__name__ == "_strict_binder"
    assert getattr(generator, FIELD) is False
    assert RETIRED_ENV not in os.environ


def test_a_strict_bind_after_a_production_bind_republishes_its_own_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    production, _ = _bind(monkeypatch, ExecutionProfile.PRODUCTION)
    assert getattr(production, FIELD) is True
    strict, _ = _bind(monkeypatch, ExecutionProfile.STRICT)
    assert getattr(strict, FIELD) is False
    assert RETIRED_ENV not in os.environ


def test_the_arithmetic_recovery_table_cannot_rewrite_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The recovery table never carried the retired key, and must not now.

    The key was published from the plan's quant name outside the recovery
    sweep, so the sweep had no say in it and cannot start having one. A
    request-local decision is not a process key: the plan's own row already
    carries the answer.
    """

    assert "GR_DPP_REDUCTION" not in PRODUCTION_ARITHMETIC_RECOVERY_FLAGS
    generator, profile = _bind(
        monkeypatch, ExecutionProfile.PRODUCTION, quant=UD_QUANT)
    assert profile.manifest["quant"] == UD_QUANT
    assert getattr(generator, FIELD) is True
    assert RETIRED_ENV not in os.environ


def test_the_binder_records_a_discarded_prebinder_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hipengine.generation.qwen4_exp_profiles import last_prebinder_conflicts

    monkeypatch.setattr(os, "environ", os.environ.copy())
    register_gfx1151_kernels(replace=True)
    register_qwen4_exp_gfx1151_profiles()
    resolved = resolve_runtime_profile(
        model=QWEN4_EXP_MODEL, backend=QWEN4_EXP_BACKEND, quant=UD_QUANT,
        profile=ExecutionProfile.STRICT)
    monkeypatch.setenv(RETIRED_ENV, "1")
    owner = SimpleNamespace(runner=SimpleNamespace())
    with pytest.warns(RuntimeWarning, match=RETIRED_ENV):
        resolved.binder(owner, resolved)
    assert last_prebinder_conflicts()[RETIRED_ENV] == ("1", "0")
    assert getattr(owner, FIELD) is False


# --- the live harnesses name the owner field, not a switch ------------------


def test_the_wave_scale_campaign_closes_dpp_on_the_measured_owner() -> None:
    """A wave-scale arm that leaves DPP published measures the sibling.

    ``gr_wave_scope`` closed the retired key in both arms; retiring the key must
    move that closure onto the owner the arm measures, not drop it.
    """

    from scripts.qwen4exp_halo_box_campaign_ab import gr_wave_scope

    owner = SimpleNamespace(gr_wave_scale=True, gr_wide=True, **{FIELD: True})
    environment = {"HIPENGINE_QWEN4_EXP_GR_IU8": "1"}
    with gr_wave_scope(owner, True, environment=environment):
        assert getattr(owner, FIELD) is False, (
            "a wave-scale arm must close DPP on the owner it measures")
        assert owner.gr_wave_scale is True
        # The wide tile preempts the wave-scale owner, so the package closes it
        # on the runner as well.
        assert owner.gr_wide is False
        assert RETIRED_ENV not in environment
        assert environment["HIPENGINE_QWEN4_EXP_GR_IU8"] == "0"
    assert getattr(owner, FIELD) is True
    assert owner.gr_wave_scale is True
    assert owner.gr_wide is True
    assert environment["HIPENGINE_QWEN4_EXP_GR_IU8"] == "1"


def test_the_wave_scale_campaign_leaves_a_stale_dpp_value_inert() -> None:
    from scripts.qwen4exp_halo_box_campaign_ab import gr_wave_scope

    owner = SimpleNamespace(gr_wave_scale=True, **{FIELD: True})
    environment = {RETIRED_ENV: "1"}
    with gr_wave_scope(owner, True, environment=environment):
        assert getattr(owner, FIELD) is False
    # Nothing reads the retired key, so the package neither rewrites nor
    # restores it.
    assert environment[RETIRED_ENV] == "1"


def test_the_dpp_campaign_binds_the_owner_and_refuses_an_arm_without_one() -> None:
    from scripts.qwen4exp_halo_box_campaign_ab import route_scope

    with pytest.raises(ValueError, match="owning runner"):
        route_scope("gr-dpp-reduction", "after")

    owner = SimpleNamespace(
        gr_wave_scale=True, gr_wide=True, **{FIELD: False})
    environment = {"HIPENGINE_QWEN4_EXP_GR_IU8": "1"}
    with route_scope("gr-dpp-reduction", "after", owner,
                     environment=environment):
        assert getattr(owner, FIELD) is True
        # Both preemptors are closed for the measured window, so the counted
        # leaf is what both arms actually run.
        assert owner.gr_wide is False, (
            "the wide tile preempts the fused-up branch")
        assert environment["HIPENGINE_QWEN4_EXP_GR_IU8"] == "0", (
            "the iu8 projection preempts the fused-up branch")
    assert getattr(owner, FIELD) is False
    assert owner.gr_wide is True
    assert environment["HIPENGINE_QWEN4_EXP_GR_IU8"] == "1"


def test_the_dpp_arm_writes_no_process_key() -> None:
    from scripts.qwen4exp_halo_box_campaign_ab import _apply_mode

    environment = {"other": "untouched"}
    for mode in ("before", "after"):
        _apply_mode(mode, route_package="gr-dpp-reduction",
                    environment=environment)
    assert environment == {"other": "untouched"}


def test_the_dpp_control_record_names_the_owner_field() -> None:
    from scripts.qwen4exp_halo_box_campaign_ab import gr_dpp_control_record

    owner = SimpleNamespace(gr_wave_scale=True, **{FIELD: True})
    record = gr_dpp_control_record(owner)
    assert record["runner_field"] == FIELD
    assert record["published_value"] is True
    assert record["counted_variant"] == DPP


def test_the_state_gate_arm_binds_the_owner_not_the_environment() -> None:
    from scripts.qwen4exp_row4_state_gate import apply_state_gate_mode

    environment = {"other": "untouched"}
    apply_state_gate_mode(
        None, "gr-dpp-reduction", "1", RETIRED_ENV, environment=environment)
    assert environment == {"other": "untouched"}


@pytest.mark.parametrize("name", K320_SCRIPTS)
def test_the_k320_diagnostics_bind_the_owner(name: str) -> None:
    """The K320 screens ride the DPP owner; they must name it, not a key.

    Each script pinned the retired key so its parent arm was the DPP sibling
    rather than the wave-scale owner. The owner field carries that premise.
    """

    source = (ROOT / "scripts" / name).read_text()
    assert RETIRED_ENV not in _env_writes(source), name
    assert _attribute_writes(source, FIELD), (
        f"{name} does not bind the GR DPP owner")


def test_the_journey_records_each_arms_former_route() -> None:
    """The retired key was a captured process key, so arms carried values.

    An arm built from the strict snapshot inherited the strict bind's off, a
    ``GR_`` family arm overlaid the production slice and followed it, and an arm
    that named it nowhere ran whatever the production bind published.
    """

    from scripts import qwen4exp_journey_localize as journey

    arms = {
        "strict_snapshot": {RETIRED_ENV: "0"},
        "dense_family": {RETIRED_ENV: "1"},
        "unnamed": {},
    }
    values = journey.migrate_gr_dpp_arms(arms, production_value=True)
    assert values == {
        "strict_snapshot": False,
        "dense_family": True,
        "unnamed": True,
    }
    assert all(RETIRED_ENV not in env for env in arms.values())


def test_the_journey_split_phase_closes_the_strict_half() -> None:
    from scripts import qwen4exp_journey_localize as journey

    owner = SimpleNamespace(**{FIELD: True})
    with journey.gr_dpp_split_scope(
            owner, "strict_prefill_production_decode", "prefill"):
        assert getattr(owner, FIELD) is False
    assert getattr(owner, FIELD) is True
    # The production half keeps whatever the arm published.
    with journey.gr_dpp_split_scope(
            owner, "strict_prefill_production_decode", "decode"):
        assert getattr(owner, FIELD) is True


# --- dynamic arm wiring ----------------------------------------------------


@pytest.mark.parametrize("mode,expected", [("before", False), ("after", True)])
def test_the_dpp_package_binds_each_arm_and_closes_both_preemptors(
    mode: str, expected: bool,
) -> None:
    """Both arms run the same closure; only the owner differs between them."""

    from scripts.qwen4exp_halo_box_campaign_ab import route_scope

    owner = SimpleNamespace(gr_wave_scale=True, gr_wide=True, **{FIELD: False})
    environment = {"HIPENGINE_QWEN4_EXP_GR_IU8": "1"}
    with route_scope("gr-dpp-reduction", mode, owner,
                     environment=environment):
        assert getattr(owner, FIELD) is expected
        assert owner.gr_wave_scale is True, (
            "the before arm must be the wave-scale parent, not the base")
        assert owner.gr_wide is False
        assert environment["HIPENGINE_QWEN4_EXP_GR_IU8"] == "0"
    assert getattr(owner, FIELD) is False
    assert owner.gr_wave_scale is True
    assert owner.gr_wide is True
    assert environment["HIPENGINE_QWEN4_EXP_GR_IU8"] == "1"


def test_the_dpp_package_restores_every_control_after_an_exception() -> None:
    from scripts.qwen4exp_halo_box_campaign_ab import gr_dpp_scope

    owner = SimpleNamespace(gr_wave_scale=True, gr_wide=True, **{FIELD: False})
    environment = {"HIPENGINE_QWEN4_EXP_GR_IU8": "1"}
    with pytest.raises(RuntimeError, match="aborted"):
        with gr_dpp_scope(owner, True, environment=environment):
            assert getattr(owner, FIELD) is True
            raise RuntimeError("aborted")
    assert getattr(owner, FIELD) is False
    assert owner.gr_wide is True
    assert owner.gr_wave_scale is True
    assert environment["HIPENGINE_QWEN4_EXP_GR_IU8"] == "1"


def test_the_arms_own_binding_decides_the_forced_wave_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the sibling unregistered, the forced arm runs the wave parent.

    The package holds the wave-scale owner on in both arms, so the fallback the
    forced arm takes is the exact parent it replaces rather than the fused base.
    """

    from scripts.qwen4exp_halo_box_campaign_ab import gr_dpp_scope

    monkeypatch.setattr(
        runner_module, "is_registered", lambda key: key.variant == WAVE)
    owner = SimpleNamespace(gr_wave_scale=False, gr_wide=False, **{FIELD: False})

    def selected() -> str:
        return runner_module._qwen4_exp_gr_wave_scale_key(
            _key(BASE), rows=389, branches=4,
            published=owner.gr_wave_scale, dpp=owner.gr_dpp_reduction,
        ).variant

    with gr_dpp_scope(owner, False):
        assert selected() == WAVE
    with gr_dpp_scope(owner, True):
        assert getattr(owner, FIELD) is True
        assert selected() == WAVE
    assert getattr(owner, FIELD) is False


def test_the_journey_snapshots_feed_the_arm_migration() -> None:
    """The arms are built from the two bindings' former snapshots, not names."""

    from scripts import qwen4exp_journey_localize as journey

    flags = {"HIPENGINE_QWEN4_EXP_GR_IU8": "1"}
    strict = journey.gr_dpp_snapshot(flags, published=False)
    production = journey.gr_dpp_snapshot(dict(flags), published=True)
    arms = {name: dict(env) for name, env in journey.ARMS.items()}
    journey.snapshot_arms(strict, production, arms=arms)

    values = journey.migrate_gr_dpp_arms(arms, production_value=True)
    # The strict snapshot and every arm built from it carried the parent; the
    # dense family overlaid the production ``GR_*`` slice and followed it.
    for name in ("all_flags_strict", "strict_plus_matrix", "strict_plus_qsa",
                 "strict_plus_gdn", "strict_plus_gr_iu8"):
        assert values[name] is False, name
    assert values["strict_plus_dense"] is True
    # An arm that named it nowhere ran whatever the production bind published.
    assert values["bound"] is True
    assert all(journey._GR_DPP_ENV not in env for env in arms.values())


def test_the_journey_strict_snapshot_drops_a_stale_retired_key() -> None:
    """A leftover process value cannot ride the strict snapshot into a phase.

    ``strict_flags`` feeds ``split_trajectory``'s prefill flags, so a stale
    caller value would be written into the strict phase's environment. The
    retired key is excluded at capture; the arms rebuild it explicitly from the
    binding that published the decision.
    """

    from scripts import qwen4exp_journey_localize as journey

    snapshot = journey.strict_environment({
        RETIRED_ENV: "1",
        "HIPENGINE_QWEN4_EXP_GR_WIDE": "0",
        "HIPENGINE_QWEN4_EXP_GR_IU8": "1",
        "HIPENGINE_QWEN4_EXP_Q4_DP4A64": "1",
        "UNRELATED": "1",
    })
    assert RETIRED_ENV not in snapshot
    # The live GR keys and the per-arm Q4 pair keep their own rules.
    assert snapshot["HIPENGINE_QWEN4_EXP_GR_WIDE"] == "0"
    assert snapshot["HIPENGINE_QWEN4_EXP_GR_IU8"] == "1"
    assert "HIPENGINE_QWEN4_EXP_Q4_DP4A64" not in snapshot
    assert "UNRELATED" not in snapshot
    # ``gr_dpp_snapshot`` re-adds the key from the binding's publication, so
    # the arm builders still see the value each former snapshot carried.
    assert journey.gr_dpp_snapshot(snapshot, published=False)[RETIRED_ENV] == "0"


class _SplitRunner:
    """The minimum a split trajectory touches, recording the DPP owner."""

    def __init__(self) -> None:
        self.runtime = SimpleNamespace(device_synchronize=lambda: None)
        self.gdn_colwarps_prefill = True
        self.gdn_register_prefill = False
        self.q4_pair_prefill = False
        self.q8_wave_scale_prefill = False
        self.gr_wave_scale = False
        self.gr_wide = False
        self.gr_dpp_reduction = True
        self.seen: list[tuple[str, bool]] = []

    def reset(self) -> None:
        self.seen.append(("prefill", self.gr_dpp_reduction))

    def prefill(self, prompt_ids):
        return SimpleNamespace(logits=[0.0, 1.0])

    def step(self, token):
        self.seen.append(("decode", self.gr_dpp_reduction))
        return SimpleNamespace(logits=[0.5])


def test_the_split_trajectory_binds_the_owner_per_phase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The split phases call the scope, and the strict phase really closes it."""

    from scripts import qwen4exp_journey_localize as journey

    runner = _SplitRunner()
    calls: list[tuple[str, str]] = []
    real = journey.gr_dpp_split_scope

    def recording(target, arm, phase):
        calls.append((arm, phase))
        return real(target, arm, phase)

    monkeypatch.setattr(journey, "gr_dpp_split_scope", recording)
    journey.split_trajectory(
        runner, [1, 2], [3], {}, {},
        arm="strict_prefill_production_decode")
    assert calls == [
        ("strict_prefill_production_decode", "prefill"),
        ("strict_prefill_production_decode", "decode"),
    ]
    assert runner.seen == [("prefill", False), ("decode", True)]
    assert runner.gr_dpp_reduction is True


# --- arm accounting: the sibling and the parent it replaces ----------------


@pytest.mark.parametrize(
    "enabled,tokens,chunk,expected",
    [
        (True, 512, 512, (96, 0)),
        (False, 512, 512, (0, 96)),
        # At the guard neither leaf runs: the parent is only selected above it.
        (True, 256, 512, (0, 0)),
        (False, 256, 512, (0, 0)),
        # One token past the guard is the first chunk that carries either.
        (True, 257, 512, (96, 0)),
        (False, 389, 512, (0, 96)),
        (True, 1024, 512, (192, 0)),
        (False, 4096, 1024, (0, 384)),
    ],
)
def test_one_arm_expects_the_sibling_or_the_parent_but_not_both(
    enabled: bool, tokens: int, chunk: int, expected: tuple[int, int],
) -> None:
    """The pair states which leaf one arm must observe, and only one runs."""

    from scripts import qwen4exp_halo_box_campaign_ab as campaign

    assert campaign.gr_dpp_arm_expectations(enabled, tokens, chunk) == expected
    dpp, wave = campaign.gr_dpp_arm_expectations(enabled, tokens, chunk)
    assert not (dpp and wave), "one arm cannot run both leaves"


@pytest.mark.parametrize(
    "enabled,tokens,chunk", [(1, 512, 512), (True, 0, 512), (True, 512, 0)],
)
def test_the_arm_accounting_rejects_a_non_bool_or_empty_shape(
    enabled: object, tokens: int, chunk: int,
) -> None:
    from scripts import qwen4exp_halo_box_campaign_ab as campaign

    with pytest.raises(ValueError):
        campaign.gr_dpp_arm_expectations(enabled, tokens, chunk)


def test_an_ineligible_state_case_cannot_be_recorded_as_engaged() -> None:
    """A selected case below the guard is refused, not recorded as 0 == 0."""

    from scripts import qwen4exp_halo_box_campaign_ab as campaign

    short = {"id": "short-p256", "prompt_tokens": 256}
    dpp, wave = campaign.gr_dpp_arm_expectations(True, 256, 512)
    assert (dpp, wave) == (0, 0)
    with pytest.raises(AssertionError, match="zero expectation"):
        campaign.require_engaged_expectation(
            "gr-dpp-reduction", short, dpp, 512, arm_enabled=True)
    # The control arm may expect nothing; the rule binds enabled arms only.
    campaign.require_engaged_expectation(
        "gr-dpp-reduction", short, dpp, 512, arm_enabled=False)
    # An eligible case still passes the same check.
    dpp, _ = campaign.gr_dpp_arm_expectations(True, 512, 512)
    campaign.require_engaged_expectation(
        "gr-dpp-reduction", {"id": "p512", "prompt_tokens": 512}, dpp, 512,
        arm_enabled=True)


def _package_branches(source: str, package: str) -> list[ast.If]:
    """Every ``if/elif args.route_package == <package>`` branch in a script."""

    found = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if not isinstance(test, ast.Compare) or len(test.ops) != 1:
            continue
        if not isinstance(test.ops[0], ast.Eq):
            continue
        left = test.left
        if not (isinstance(left, ast.Attribute) and left.attr == "route_package"):
            continue
        right = test.comparators[0]
        if isinstance(right, ast.Constant) and right.value == package:
            found.append(node)
    return found


def _called_names(node: ast.AST) -> set[str]:
    names = set()
    for call in ast.walk(node):
        if not isinstance(call, ast.Call):
            continue
        if isinstance(call.func, ast.Name):
            names.add(call.func.id)
        elif isinstance(call.func, ast.Attribute):
            names.add(call.func.attr)
    return names


def _variant_kernel_keys(node: ast.AST, variant: str) -> int:
    """``KernelKey(...)`` calls whose fourth positional argument is ``variant``."""

    count = 0
    for call in ast.walk(node):
        if not isinstance(call, ast.Call):
            continue
        if not (isinstance(call.func, ast.Name) and call.func.id == "KernelKey"):
            continue
        if len(call.args) >= 4 and isinstance(call.args[3], ast.Constant):
            if call.args[3].value == variant:
                count += 1
    return count


def _key_names_for(node: ast.AST, variant: str) -> set[str]:
    """Names bound to a ``KernelKey(..., variant)`` assignment inside ``node``."""

    names = set()
    for stmt in ast.walk(node):
        if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
            continue
        target = stmt.targets[0]
        value = stmt.value
        if not isinstance(target, ast.Name):
            continue
        if not (isinstance(value, ast.Call) and isinstance(value.func, ast.Name)
                and value.func.id == "KernelKey"):
            continue
        if len(value.args) >= 4 and isinstance(value.args[3], ast.Constant):
            if value.args[3].value == variant:
                names.add(target.id)
    return names


def _wrappers_for(node: ast.AST, key_names: set[str]) -> list[ast.FunctionDef]:
    """The functions ``register(<key name>, <wrapper>)`` installs in ``node``."""

    functions = {child.name: child for child in ast.walk(node)
                 if isinstance(child, ast.FunctionDef)}
    wrappers = []
    for call in ast.walk(node):
        if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                and call.func.id == "register" and len(call.args) >= 2):
            continue
        if not isinstance(call.args[0], ast.Name):
            continue
        if call.args[0].id not in key_names:
            continue
        if isinstance(call.args[1], ast.Name) and call.args[1].id in functions:
            wrappers.append(functions[call.args[1].id])
    return wrappers


def _incremented_counters(node: ast.AST) -> set[str]:
    """The ``calls[0] += 1`` style counters a wrapper increments."""

    names = set()
    for child in ast.walk(node):
        if not isinstance(child, ast.AugAssign):
            continue
        target = child.target
        if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
            names.add(target.value.id)
    return names


def _restores_in_finally(node: ast.AST, key: str, original: str) -> bool:
    """``node``'s ``finally`` blocks re-register ``original`` on ``key``."""

    finalizers = [child for child in ast.walk(node)
                  if isinstance(child, ast.Try) and child.finalbody]
    return any(
        isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name) and call.func.id == "register"
        and len(call.args) >= 2
        and isinstance(call.args[0], ast.Name) and call.args[0].id == key
        and isinstance(call.args[1], ast.Name) and call.args[1].id == original
        for block in finalizers for stmt in block.finalbody
        for call in ast.walk(stmt)
    )


def test_the_counted_leaves_and_their_parent_all_resolve() -> None:
    """Both harnesses resolve these keys before an arm runs.

    A renamed or unregistered variant raises inside the campaign or the state
    gate before the first measurement, so the counter keys - the promoted DPP
    sibling, the K320 specialization it hands off to, and the wave-scale parent
    - are pinned against the backend that registers them.
    """

    from hipengine.kernels.registry import resolve

    register_gfx1151_kernels(replace=True)
    for variant in (DPP, K320, WAVE):
        entry = resolve(
            backend="hip_gfx1151", layer=GR_LAYER, quant="gguf_q8_0",
            variant=variant)
        assert callable(entry), variant


@pytest.mark.parametrize(
    "name",
    ("qwen4exp_halo_box_campaign_ab.py", "qwen4exp_row4_state_gate.py"),
)
def test_each_gate_counts_the_wave_scale_parent_in_the_control_arm(name: str) -> None:
    """Both harnesses count the parent leaf, not only the sibling.

    A zero sibling expectation alone is satisfied by any route, so the control
    arm must observe the wave-scale parent it claims to compare against.
    """

    source = (ROOT / "scripts" / name).read_text()
    branches = _package_branches(source, "gr-dpp-reduction")
    assert branches, name
    assert any(_variant_kernel_keys(branch, WAVE) for branch in branches), (
        f"{name} registers no counter on the wave-scale parent")
    assert any("register" in _called_names(branch) for branch in branches), name
    assert any("gr_dpp_arm_expectations" in _called_names(branch)
               for branch in branches), (
        f"{name} does not state the arm's parent expectation")
    assert "wave_parent_calls" in source, name
    main = next(node for node in ast.parse(source).body
                if isinstance(node, ast.FunctionDef) and node.name == "main")
    assert _restores_in_finally(main, "gr_wave_key", "gr_wave_original"), (
        f"{name} leaks its wave-parent counter after the comparison")


@pytest.mark.parametrize(
    "name",
    ("qwen4exp_halo_box_campaign_ab.py", "qwen4exp_row4_state_gate.py"),
)
def test_each_gate_counts_the_k320_leaf_in_the_enabled_arm(name: str) -> None:
    """The enabled arm's 512-row chunk runs the K320 specialization.

    ``_qwen4_exp_gr_k320_full_rows_key`` hands the promoted DPP sibling to the
    K320 kernel for K=320, H=2560, four branches, and complete ROW_BATCH=4
    groups above 256 rows. A harness that counted only the sibling recorded 0
    for that arm, so the K320 counter must feed the same candidate count the
    arm asserts, and the wrapper must be restored after the comparison.
    """

    source = (ROOT / "scripts" / name).read_text()
    tree = ast.parse(source)
    branches = _package_branches(source, "gr-dpp-reduction")
    assert branches, name
    assert any(_variant_kernel_keys(branch, K320) for branch in branches), (
        f"{name} registers no counter on the K320 specialization")
    main = next(node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == "main")
    k320_keys = _key_names_for(main, K320)
    dpp_keys = _key_names_for(main, DPP)
    assert k320_keys and dpp_keys, name
    k320_wrappers = _wrappers_for(main, k320_keys)
    dpp_wrappers = _wrappers_for(main, dpp_keys)
    assert k320_wrappers, f"{name} leaves the K320 key uncounted"
    assert dpp_wrappers, f"{name} leaves the DPP key uncounted"
    # Both DPP leaves must feed the one counter the arm's candidate count is
    # derived from; a private K320 list would be a silent zero again.
    k320_counters = set().union(
        *(_incremented_counters(fn) for fn in k320_wrappers))
    dpp_counters = set().union(
        *(_incremented_counters(fn) for fn in dpp_wrappers))
    assert len(k320_counters) == 1, (
        f"{name} increments {sorted(k320_counters)} for the K320 leaf")
    counter = next(iter(k320_counters))
    assert k320_counters <= dpp_counters, (
        f"{name} counts the K320 leaf somewhere the sibling does not")
    assert f"{counter}[0] - start_calls" in source, (
        f"{name} records its candidate count from another counter")
    assert _restores_in_finally(main, "gr_k320_key", "gr_k320_original"), (
        f"{name} leaks its K320 counter after the comparison")
    if name == "qwen4exp_halo_box_campaign_ab.py":
        # The artifact must name every leaf the arm can run, so a reader can
        # tell the promoted sibling from the K320 handoff.
        segment = "\n".join(
            ast.get_source_segment(source, branch) or "" for branch in branches)
        assert "row4_key.variant" in segment and "gr_k320_key.variant" in segment, (
            f"{name} records one leaf for an arm that can run two")


def test_the_state_gate_refuses_an_ineligible_case_before_it_asserts() -> None:
    """The engagement guard is wired into the branch, not only available."""

    source = (ROOT / "scripts" / "qwen4exp_row4_state_gate.py").read_text()
    branches = _package_branches(source, "gr-dpp-reduction")
    assert any("require_engaged_expectation" in _called_names(branch)
               for branch in branches), (
        "the state gate can still record an ineligible case as engaged")
