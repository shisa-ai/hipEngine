"""Diagnostic callers of the tile-16 GDN prefill owner, not just its runtime.

``tests/test_unit_qwen4exp_gdn_tile16_variant_owner.py`` pins the runtime owner
(selection, scope, publication, dispatch, forwarding). This module pins the
migration of everything that used to *carry* the leaf as a process value:

* every candidate declaration that names a leaf reaches a real binding and is
  restored afterwards, on success and after an exception;
* the journey arm map, the layer2 candidate binding, the restore-arm scope and
  the A/B scopes all resolve through the same owner slot;
* the two probes report the owner in force at dispatch time rather than a value
  cached when the generator was built;
* a failure while preparing a measured arm cannot leave the owner switched;
* no script assigns the retired ``HIPENGINE_QWEN4_EXP_GDN_TILE16_VARIANT`` key.

The arm loops themselves need a model, so the consumer checks walk the real
call site: the helper must be *called* inside the function that sets up the arm,
not merely imported.
"""

from __future__ import annotations

import ast
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace

import pytest

from hipengine.runtime.qwen4_exp_runner import (
    gdn_tile16_variant,
    resolve_gdn_tile16_variant,
)
from scripts import qwen4exp_gdn_multi_engine_ab as multi_engine_ab
from scripts import qwen4exp_gdn_multicolumn_route_probe as route_probe
from scripts import qwen4exp_ple_gather_ab as ple_gather_ab
from scripts.qwen4exp_journey_localize import (
    ARMS,
    _GDN_TILE16_ARM_VARIANTS,
    declare_gdn_tile16_owner,
    declared_gdn_tile16_variants,
    gdn_tile16_arm_report,
    gdn_tile16_arm_scope,
    gdn_tile16_arm_variant,
)
from scripts.qwen4exp_layer2_profile_gate import (
    CANDIDATES,
    bind_candidate_gdn_tile16_variant,
    candidate_owner_changes,
    gdn_tile16_variant_arm_scope,
)

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
RETIRED_KEY = "HIPENGINE_QWEN4_EXP_GDN_TILE16_VARIANT"
PUBLISHED = "qwen4exp_gdn_rowsplit_prefill"
MULTI = "qwen4exp_gdn_tiled16_multi_prefill"


def _owner(published: str | None = PUBLISHED):
    """A generator/runner pair that carries a published tile-16 owner."""

    runner = SimpleNamespace(gdn_tile16_variant=published)
    generator = SimpleNamespace(gdn_tile16_variant=published, runner=runner)
    return generator, runner


# --------------------------------------------------------------------------
# Candidate declarations reach a real binding and are restored
# --------------------------------------------------------------------------


def test_declaring_candidates_exist_and_name_a_leaf():
    declaring = {
        name: spec.gdn_tile16_variant
        for name, spec in CANDIDATES.items()
        if spec.gdn_tile16_variant is not None
    }
    assert declaring
    assert set(declaring.values()) <= {
        MULTI, "qwen4exp_gdn_tiled16_dpp_prefill", "qwen4exp_gdn_tiled16_prefill"}


@pytest.mark.parametrize(
    "name", sorted(
        name for name, spec in CANDIDATES.items()
        if spec.gdn_tile16_variant is not None))
def test_candidate_binding_sets_the_declared_leaf_and_restores_it(name):
    spec = CANDIDATES[name]
    generator, runner = _owner()
    with ExitStack() as stack:
        in_force = bind_candidate_gdn_tile16_variant(stack, generator, spec)
        assert in_force == spec.gdn_tile16_variant
        assert generator.gdn_tile16_variant == spec.gdn_tile16_variant
        assert runner.gdn_tile16_variant == spec.gdn_tile16_variant
        assert resolve_gdn_tile16_variant(generator) == spec.gdn_tile16_variant
    assert generator.gdn_tile16_variant == PUBLISHED
    assert runner.gdn_tile16_variant == PUBLISHED


@pytest.mark.parametrize(
    "name", sorted(
        name for name, spec in CANDIDATES.items()
        if spec.gdn_tile16_variant is not None))
def test_candidate_binding_restores_after_an_exception(name):
    spec = CANDIDATES[name]
    generator, runner = _owner()

    class ArmFailed(RuntimeError):
        pass

    with pytest.raises(ArmFailed):
        with ExitStack() as stack:
            bind_candidate_gdn_tile16_variant(stack, generator, spec)
            assert generator.gdn_tile16_variant == spec.gdn_tile16_variant
            raise ArmFailed
    assert generator.gdn_tile16_variant == PUBLISHED
    assert runner.gdn_tile16_variant == PUBLISHED


def test_a_candidate_that_names_no_leaf_keeps_the_published_owner():
    spec = CANDIDATES["production_conservative"]
    assert spec.gdn_tile16_variant is None
    generator, runner = _owner()
    with ExitStack() as stack:
        in_force = bind_candidate_gdn_tile16_variant(stack, generator, spec)
        assert in_force == PUBLISHED
        assert generator.gdn_tile16_variant == PUBLISHED
        assert runner.gdn_tile16_variant == PUBLISHED
    assert runner.gdn_tile16_variant == PUBLISHED


# --------------------------------------------------------------------------
# The restore-arm scope the prefill A/B loop uses
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name", sorted(
        name for name, spec in CANDIDATES.items()
        if spec.gdn_tile16_variant is not None))
def test_restore_arm_scope_binds_the_declared_leaf(name):
    """The A/B loop's scope receives exactly what the declaration names."""

    declared = CANDIDATES[name].gdn_tile16_variant
    runner = SimpleNamespace(gdn_tile16_variant=PUBLISHED)
    with gdn_tile16_variant_arm_scope(runner, declared):
        assert runner.gdn_tile16_variant == declared
    assert runner.gdn_tile16_variant == PUBLISHED


def test_restore_arm_scope_without_a_declaration_keeps_the_published_owner():
    runner = SimpleNamespace(gdn_tile16_variant=PUBLISHED)
    with gdn_tile16_variant_arm_scope(runner, None):
        assert runner.gdn_tile16_variant == PUBLISHED
    assert runner.gdn_tile16_variant == PUBLISHED


def test_restore_arm_scope_restores_after_an_exception():
    runner = SimpleNamespace(gdn_tile16_variant=PUBLISHED)
    with pytest.raises(ValueError):
        with gdn_tile16_variant_arm_scope(runner, MULTI):
            assert runner.gdn_tile16_variant == MULTI
            raise ValueError("arm aborted")
    assert runner.gdn_tile16_variant == PUBLISHED


def test_restore_inventory_names_the_tile16_field():
    changes = candidate_owner_changes(CANDIDATES["gdn_multi_column"])
    assert changes["gdn_tile16_variant"] == MULTI
    # The conservative arm closes columnwarp, ordered decode and flash;
    # all owner changes must be inventoried.
    assert candidate_owner_changes(
        CANDIDATES["production_conservative"]) == {
        "gdn_colwarps_prefill": False, "qsa_ordered_decode": False,
        "qsa_flash_prefill_enabled": False, "q8_selected_down_enabled": False,
        "gr_iu8_up": False, "gr_iu8_down": False}
    assert candidate_owner_changes(CANDIDATES["production_baseline"]) == {}


# --------------------------------------------------------------------------
# The journey arm map
# --------------------------------------------------------------------------


def test_journey_arm_map_names_the_multi_leaf_and_leaves_other_arms_alone():
    assert gdn_tile16_arm_variant("strict_plus_gdn_multi") == MULTI
    assert gdn_tile16_arm_variant("bound") is None
    assert gdn_tile16_arm_variant("strict_plus_gdn") is None


@pytest.mark.parametrize(
    "name", sorted(
        name for name, spec in CANDIDATES.items()
        if spec.gdn_tile16_variant is not None))
def test_journey_arm_map_is_seeded_from_candidate_declarations(name):
    """The arm map is the candidates' declarations, not a second hand-kept list."""

    expected = {
        arm: spec.gdn_tile16_variant
        for arm, spec in ((arm, CANDIDATES.get(arm)) for arm in ARMS)
        if spec is not None and spec.gdn_tile16_variant is not None
    }
    assert declared_gdn_tile16_variants() == expected
    if name in ARMS:
        assert gdn_tile16_arm_variant(name) == CANDIDATES[name].gdn_tile16_variant
    else:
        # A candidate that is not a journey arm cannot leak into the map.
        assert name not in _GDN_TILE16_ARM_VARIANTS


def test_journey_arm_map_names_only_declared_leaves():
    declared = set(declared_gdn_tile16_variants().values()) | {MULTI}
    for arm in ARMS:
        variant = gdn_tile16_arm_variant(arm)
        assert variant is None or variant in declared, (arm, variant)


def test_journey_declaration_overrides_named_arms_only():
    declare_gdn_tile16_owner("qwen4exp_gdn_tiled16_dpp_prefill", arms=["bound"])
    try:
        assert gdn_tile16_arm_variant("bound") == "qwen4exp_gdn_tiled16_dpp_prefill"
        assert gdn_tile16_arm_variant("strict_plus_gdn_multi") == MULTI
    finally:
        _GDN_TILE16_ARM_VARIANTS.pop("bound", None)
    assert gdn_tile16_arm_variant("bound") is None


def test_journey_arm_scope_binds_the_declared_leaf_and_restores_it():
    generator, runner = _owner()
    with gdn_tile16_arm_scope(generator, "strict_plus_gdn_multi"):
        assert generator.gdn_tile16_variant == MULTI
        assert runner.gdn_tile16_variant == MULTI
        assert gdn_tile16_arm_report(generator, "strict_plus_gdn_multi") == MULTI
    assert generator.gdn_tile16_variant == PUBLISHED
    assert runner.gdn_tile16_variant == PUBLISHED


def test_journey_arm_scope_for_an_undeclared_arm_keeps_the_plan_owner():
    generator, runner = _owner()
    with gdn_tile16_arm_scope(generator, "bound"):
        assert generator.gdn_tile16_variant == PUBLISHED
        assert runner.gdn_tile16_variant == PUBLISHED
    assert gdn_tile16_arm_report(generator, "bound") == PUBLISHED


def test_journey_arm_scope_restores_after_an_exception():
    generator, runner = _owner()
    with pytest.raises(RuntimeError):
        with gdn_tile16_arm_scope(generator, "strict_plus_gdn_multi"):
            raise RuntimeError("arm aborted")
    assert generator.gdn_tile16_variant == PUBLISHED
    assert runner.gdn_tile16_variant == PUBLISHED


# --------------------------------------------------------------------------
# Probe reporting: the owner in force at dispatch time, not a cached value
# --------------------------------------------------------------------------


def test_route_probe_resolves_the_runner_field_live():
    generator, runner = _owner()
    assert route_probe.dispatch_tile16_owner(generator) == PUBLISHED
    runner.gdn_tile16_variant = MULTI
    assert route_probe.dispatch_tile16_owner(generator) == MULTI
    runner.gdn_tile16_variant = None
    assert route_probe.dispatch_tile16_owner(generator) is None


def test_route_probe_unwraps_a_wrapper_chain_to_the_runner():
    runner = SimpleNamespace(gdn_tile16_variant=MULTI)
    inner = SimpleNamespace(runner=runner)
    wrapper = SimpleNamespace(inner=inner)
    assert route_probe.dispatch_tile16_owner(wrapper) == MULTI
    assert route_probe.dispatch_tile16_owner(inner) == MULTI


def test_route_probe_reports_the_parent_for_an_unbound_owner():
    assert route_probe.dispatch_tile16_owner(None) is None
    assert route_probe.dispatch_tile16_owner(SimpleNamespace()) is None
    assert route_probe.dispatch_tile16_owner(
        SimpleNamespace(gdn_tile16_variant="")) is None


def test_route_probe_unwrap_stops_on_a_self_reference():
    looped = SimpleNamespace(gdn_tile16_variant=MULTI)
    looped.inner = looped
    assert route_probe.unwrap_owner(looped) is looped
    assert route_probe.dispatch_tile16_owner(looped) == MULTI


# --------------------------------------------------------------------------
# The PLE/DPP A/B setup: a failed arm preparation cannot leave the owner bound
# --------------------------------------------------------------------------


def _ab_generator(published: str | None = PUBLISHED):
    table = SimpleNamespace(gather_rows=lambda ids: ids)
    runner = SimpleNamespace(gdn_tile16_variant=published,
                             step=lambda token: token)
    generator = SimpleNamespace(
        gdn_tile16_variant=published, runner=runner,
        _resident=SimpleNamespace(ple_table=table))
    return generator, runner, table


def test_ab_measurement_scope_binds_the_parent_and_restores_every_mutation(
        monkeypatch):
    generator, runner, table = _ab_generator()
    original_step, original_gather = runner.step, table.gather_rows
    monkeypatch.setattr(
        ple_gather_ab, "dpp_library",
        lambda report: SimpleNamespace(_name=str(SCRIPTS / "no-such-library.so")))
    args = SimpleNamespace(method="gdn_dpp")
    with ple_gather_ab.measurement_scope(generator, args, {}, 4) as harness:
        assert generator.gdn_tile16_variant is None
        assert runner.gdn_tile16_variant is None
        assert harness.table is table
        assert runner.step is not original_step
    assert generator.gdn_tile16_variant == PUBLISHED
    assert runner.gdn_tile16_variant == PUBLISHED
    assert runner.step is original_step
    assert table.gather_rows is original_gather


def test_ab_measurement_scope_restores_the_owner_after_an_arm_exception(
        monkeypatch):
    generator, runner, _table = _ab_generator()
    monkeypatch.setattr(
        ple_gather_ab, "dpp_library",
        lambda report: SimpleNamespace(_name=str(SCRIPTS / "no-such-library.so")))
    args = SimpleNamespace(method="gdn_dpp")
    with pytest.raises(RuntimeError):
        with ple_gather_ab.measurement_scope(generator, args, {}, 4):
            assert runner.gdn_tile16_variant is None
            raise RuntimeError("arm aborted")
    assert generator.gdn_tile16_variant == PUBLISHED
    assert runner.gdn_tile16_variant == PUBLISHED


def test_ab_library_failure_leaves_the_published_owner(monkeypatch):
    """The DPP library build runs hipcc; a failure must not switch the owner."""

    generator, runner, table = _ab_generator()
    original_step, original_gather = runner.step, table.gather_rows
    seen = []

    class LibraryBuildFailed(RuntimeError):
        pass

    def fail(_report):
        # The binding is already in force when the risky build runs, so the
        # scope owns it and the unwind below restores the published value.
        seen.append(generator.gdn_tile16_variant)
        raise LibraryBuildFailed("hipcc unavailable")

    monkeypatch.setattr(ple_gather_ab, "dpp_library", fail)
    args = SimpleNamespace(method="gdn_dpp")
    with pytest.raises(LibraryBuildFailed):
        with ple_gather_ab.measurement_scope(generator, args, {}, 4):
            pytest.fail("the scope must not be entered")
    assert seen == [None]
    assert generator.gdn_tile16_variant == PUBLISHED
    assert runner.gdn_tile16_variant == PUBLISHED
    assert runner.step is original_step
    assert table.gather_rows is original_gather


def test_ab_measurement_scope_without_the_library_arm_binds_nothing():
    generator, runner, _table = _ab_generator()
    args = SimpleNamespace(method="mmap_random")
    with ple_gather_ab.measurement_scope(generator, args, {}, 4):
        assert runner.gdn_tile16_variant == PUBLISHED
    assert runner.gdn_tile16_variant == PUBLISHED


# --------------------------------------------------------------------------
# Reachability: the arm setup actually calls the helper
# --------------------------------------------------------------------------

#: (script module, function that sets up the arm, helper it must call).
CONSUMER_CALL_SITES = (
    ("qwen4exp_layer2_profile_gate.py", "_run",
     "bind_candidate_gdn_tile16_variant"),
    # These four bind every declared owner through one call, so the tile-16
    # helper is reached through it rather than by name.
    ("qwen4exp_active_task_checks.py", "main", "bind_candidate_owners"),
    ("qwen4exp_long_context_gate.py", "run", "bind_candidate_owners"),
    ("qwen4exp_q8_repair_depth_gate.py", "main", "bind_candidate_owners"),
    ("qwen4exp_q8_repair_tasks.py", "main", "bind_candidate_owners"),
    ("qwen4exp_prefill_restore_ab.py", "main", "gdn_tile16_variant_arm_scope"),
    ("qwen4exp_journey_localize.py", "main", "gdn_tile16_arm_scope"),
    ("qwen4exp_gdn_multi_engine_ab.py", "main", "gdn_tile16_variant"),
    ("qwen4exp_gdn_tiled16_multi_engagement_probe.py", "main",
     "gdn_tile16_variant"),
    ("qwen4exp_ple_gather_ab.py", "measurement_scope", "gdn_tile16_variant"),
)


def _module_tree(name: str) -> ast.Module:
    return ast.parse((SCRIPTS / name).read_text(), filename=name)


def _function_node(tree: ast.Module, name: str) -> ast.AST:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"no function {name!r}")


def _called_names(node: ast.AST) -> set[str]:
    names: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            func = child.func
            if isinstance(func, ast.Name):
                names.add(func.id)
            elif isinstance(func, ast.Attribute):
                names.add(func.attr)
    return names


@pytest.mark.parametrize("script, function, helper", CONSUMER_CALL_SITES)
def test_arm_setup_calls_the_owner_helper(script, function, helper):
    tree = _module_tree(script)
    assert helper in _called_names(_function_node(tree, function)), (
        f"{script}:{function} must call {helper}, not only import it")


def test_probe_predicate_has_no_cached_owner_cell():
    """The predicate must resolve the owner per call, not append a snapshot."""

    tree = _module_tree("qwen4exp_gdn_multicolumn_route_probe.py")
    predicate = _function_node(tree, "counting_predicate")
    assert "dispatch_tile16_owner" in _called_names(predicate)


def _assigned_environ_keys(tree: ast.Module) -> set[str]:
    keys: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if (isinstance(target, ast.Subscript)
                        and isinstance(target.value, ast.Attribute)
                        and target.value.attr == "environ"
                        and isinstance(target.slice, ast.Constant)
                        and isinstance(target.slice.value, str)):
                    keys.add(target.slice.value)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in {"update", "setdefault"}:
                for arg in node.args:
                    if isinstance(arg, ast.Dict):
                        for key in arg.keys:
                            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                                keys.add(key.value)
    return keys


@pytest.mark.parametrize("script", sorted(p.name for p in SCRIPTS.glob("qwen4exp_*.py")))
def test_no_script_assigns_the_retired_tile16_key(script):
    assert RETIRED_KEY not in _assigned_environ_keys(_module_tree(script)), (
        f"{script} still writes {RETIRED_KEY}; bind the owner instead")


def test_the_retired_key_is_not_an_arm_environment_value():
    """No candidate carries the retired key as an environment override."""

    offenders = {
        name: spec.environment for name, spec in CANDIDATES.items()
        if RETIRED_KEY in spec.environment
    }
    assert offenders == {}


# --------------------------------------------------------------------------
# The runtime scope the scripts share
# --------------------------------------------------------------------------


def test_shared_scope_binds_generator_and_runner_together():
    generator, runner = _owner()
    with gdn_tile16_variant(generator, MULTI):
        assert generator.gdn_tile16_variant == MULTI
        assert runner.gdn_tile16_variant == MULTI
    assert generator.gdn_tile16_variant == PUBLISHED
    assert runner.gdn_tile16_variant == PUBLISHED


def test_shared_scope_reports_the_prior_values():
    generator, runner = _owner()
    with gdn_tile16_variant(generator, MULTI) as prior:
        assert [value for _target, value in prior] == [PUBLISHED, PUBLISHED]


def test_multi_engine_ab_names_the_owner_not_a_process_key():
    assert not hasattr(multi_engine_ab, "TILE16_ENV")
    tree = _module_tree("qwen4exp_gdn_multi_engine_ab.py")
    assert RETIRED_KEY not in _assigned_environ_keys(tree)
