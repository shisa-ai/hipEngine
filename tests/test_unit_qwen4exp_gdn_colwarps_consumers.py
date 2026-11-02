"""Every harness that measures a candidate binds the gate it declares.

``HIPENGINE_QWEN4_EXP_GDN_COLWARPS_PREFILL`` used to be a process key, so a
harness could apply a candidate's environment and get the candidate's serial
prefill branch for free. The dispatch reads owner state now, which means each
harness has to bind ``CandidateSpec.gdn_colwarps_prefill`` itself, and a harness
that forgets silently measures the plan's published gate while reporting the
candidate's name.

What these tests pin:

* **Shared arm step.** ``bind_candidate_owners`` binds the grouped Q8_0 down
  owner, the tile-16 GDN prefill owner and the columnwarp gate from one
  declaration set, reports the values in force, keeps publication for an
  undeclared field, and restores everything when the arm raises.
* **Reachability.** The four candidate harnesses that measure through that
  helper call it on their measured arm.
* **Per-arm mappings.** ``qwen4exp_prefill_restore_ab``, ``qwen4exp_
  conservative_cost``, ``qwen4exp_gr_router_localize`` and ``qwen4exp_journey_
  localize`` each derive the gate from the arm's own declaration: a serial arm
  closes the branch, a candidate arm honors its declaration, and an arm that
  never named the gate keeps whatever the profile binder published.
* **Journey report and split phases.** The arm scope yields the value the
  measured rows ran under rather than the publication the unwind restored, and
  a split arm binds the gate and the peer scope from the phase that is
  running, not from the outer arm name.

CPU only: no GPU, no kernels, no performance claim.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
from contextlib import ExitStack
from types import SimpleNamespace

import pytest

import numpy as np

from scripts import qwen4exp_active_task_checks as active_task_checks
from scripts import qwen4exp_conservative_cost as conservative_cost
from scripts import qwen4exp_gr_router_localize as gr_router_localize
from scripts import qwen4exp_journey_localize as journey
from scripts import qwen4exp_layer2_profile_gate as layer2
from scripts import qwen4exp_long_context_gate as long_context_gate
from scripts import qwen4exp_prefill_restore_ab as prefill_restore_ab
from scripts import qwen4exp_q8_repair_depth_gate as q8_repair_depth_gate
from scripts import qwen4exp_q8_repair_tasks as q8_repair_tasks
from scripts.qwen4exp_layer2_profile_gate import CANDIDATES, candidate_owner_changes

SERIAL_CANDIDATE = CANDIDATES["production_conservative"]
DECLARES_NOTHING = dataclasses.replace(
    SERIAL_CANDIDATE,
    q8_down_variant=None,
    gdn_tile16_variant=None,
    gdn_colwarps_prefill=None,
)
OPEN_GATE = dataclasses.replace(DECLARES_NOTHING, gdn_colwarps_prefill=True)


def _owner(gate: bool = True):
    runner = SimpleNamespace(gdn_colwarps_prefill=gate)
    return SimpleNamespace(
        runner=runner,
        gdn_colwarps_prefill=gate,
        q8_down_variant=None,
        gdn_tile16_variant=None,
    )


# --------------------------------------------------------------------------
# The shared arm step
# --------------------------------------------------------------------------


def test_the_shared_arm_step_binds_and_restores_the_declared_gate():
    owner = _owner()
    stack = ExitStack()

    with stack:
        owners = layer2.bind_candidate_owners(stack, owner, SERIAL_CANDIDATE)

        assert owners["gdn_colwarps_prefill"] is False
        assert owner.gdn_colwarps_prefill is False
        assert owner.runner.gdn_colwarps_prefill is False

    assert owner.gdn_colwarps_prefill is True
    assert owner.runner.gdn_colwarps_prefill is True


def test_an_undeclared_gate_keeps_the_published_decision():
    owner = _owner()
    stack = ExitStack()

    with stack:
        owners = layer2.bind_candidate_owners(stack, owner, DECLARES_NOTHING)

        assert owners["gdn_colwarps_prefill"] is True
        assert owner.gdn_colwarps_prefill is True
        assert owner.runner.gdn_colwarps_prefill is True

    assert owner.gdn_colwarps_prefill is True


def test_a_declared_open_gate_reopens_a_closed_publication():
    owner = _owner(gate=False)
    stack = ExitStack()

    with stack:
        owners = layer2.bind_candidate_owners(stack, owner, OPEN_GATE)

        assert owners["gdn_colwarps_prefill"] is True
        assert owner.gdn_colwarps_prefill is True
        assert owner.runner.gdn_colwarps_prefill is True

    assert owner.gdn_colwarps_prefill is False
    assert owner.runner.gdn_colwarps_prefill is False


def test_the_shared_arm_step_restores_the_gate_after_a_raising_arm():
    owner = _owner()
    stack = ExitStack()

    with pytest.raises(RuntimeError, match="measurement"):
        with stack:
            layer2.bind_candidate_owners(stack, owner, SERIAL_CANDIDATE)
            assert owner.gdn_colwarps_prefill is False
            raise RuntimeError("measurement")

    assert owner.gdn_colwarps_prefill is True
    assert owner.runner.gdn_colwarps_prefill is True


@pytest.mark.parametrize(
    "module",
    (active_task_checks, long_context_gate, q8_repair_depth_gate, q8_repair_tasks),
    ids=lambda module: module.__name__.rsplit(".", 1)[-1],
)
def test_the_measured_arm_calls_the_shared_owner_step(module):
    """A harness that measures a candidate must bind its declared owners."""

    tree = ast.parse(inspect.getsource(module))
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "bind_candidate_owners"
    ]

    assert calls, f"{module.__name__} never binds the candidate's owners"


# --------------------------------------------------------------------------
# Per-arm mappings
# --------------------------------------------------------------------------


def test_prefill_restore_arms_declare_the_gate_only_when_they_named_it():
    assert prefill_restore_ab.arm_gdn_colwarps("before") is None
    assert prefill_restore_ab.arm_gdn_colwarps("production_baseline") is None
    assert prefill_restore_ab.arm_gdn_colwarps("production_conservative") is False


def test_prefill_restore_binds_the_declared_gate_and_restores_publication():
    runner = SimpleNamespace(gdn_colwarps_prefill=True)

    with prefill_restore_ab.gdn_colwarps_prefill_arm_scope(
            runner,
            prefill_restore_ab.arm_gdn_colwarps("production_conservative")):
        assert runner.gdn_colwarps_prefill is False

    assert runner.gdn_colwarps_prefill is True

    with prefill_restore_ab.gdn_colwarps_prefill_arm_scope(
            runner, prefill_restore_ab.arm_gdn_colwarps("before")):
        assert runner.gdn_colwarps_prefill is True

    assert runner.gdn_colwarps_prefill is True


def test_the_shared_graph_inventory_admits_the_gate_but_not_captured_units():
    """The conservative arm changes only prefill-or-uncaptured selectors."""

    assert candidate_owner_changes(SERIAL_CANDIDATE) == {
        "gdn_colwarps_prefill": False, "qsa_ordered_decode": False,
        "qsa_flash_prefill_enabled": False, "q8_selected_down_enabled": False,
        "gr_iu8_up": False, "gr_iu8_down": False,
        "q4_iu8_prefill_enabled": False}

    prefill_restore_ab.validate_shared_graphs(
        [], [candidate_owner_changes(SERIAL_CANDIDATE)])

    with pytest.raises(ValueError, match="captured decode units"):
        prefill_restore_ab.validate_shared_graphs(
            [], [{"gdn_peer_prefill_layers": "all"}])


def test_conservative_cost_modes_bind_the_serial_gate():
    assert conservative_cost.arm_gdn_colwarps("before") is None
    assert conservative_cost.arm_gdn_colwarps("after") is False

    runner = SimpleNamespace(gdn_colwarps_prefill=True)
    with conservative_cost.gdn_colwarps_prefill_arm_scope(
            runner, conservative_cost.arm_gdn_colwarps("after")):
        assert runner.gdn_colwarps_prefill is False
    assert runner.gdn_colwarps_prefill is True

    with conservative_cost.gdn_colwarps_prefill_arm_scope(
            runner, conservative_cost.arm_gdn_colwarps("before")):
        assert runner.gdn_colwarps_prefill is True


def test_gr_router_localize_honors_the_candidate_gate_declaration():
    spec = CANDIDATES["production_gr_up_restore"]
    assert gr_router_localize.arm_gdn_colwarps(spec) is None

    owner = _owner()
    with layer2.gdn_colwarps_prefill_arm_scope(
            owner, gr_router_localize.arm_gdn_colwarps(spec)):
        assert owner.gdn_colwarps_prefill is True

    closed = dataclasses.replace(spec, gdn_colwarps_prefill=False)
    assert gr_router_localize.arm_gdn_colwarps(closed) is False
    with layer2.gdn_colwarps_prefill_arm_scope(
            owner, gr_router_localize.arm_gdn_colwarps(closed)):
        assert owner.gdn_colwarps_prefill is False
        assert owner.runner.gdn_colwarps_prefill is False

    assert owner.gdn_colwarps_prefill is True
    assert owner.runner.gdn_colwarps_prefill is True


# --------------------------------------------------------------------------
# Journey arms
# --------------------------------------------------------------------------


def test_journey_arms_name_a_gate_only_when_they_declared_one():
    assert journey.arm_gdn_colwarps("gdn_strict") is False
    assert journey.arm_gdn_colwarps("strict_plus_gdn_serial") is False
    assert journey.arm_gdn_colwarps("production_conservative") is False
    assert journey.arm_gdn_colwarps("bound") is None
    assert journey.arm_gdn_colwarps("moe_strict") is None


#: Former gate of every arm the harness constructs, from the dicts that built
#: each arm's former environment. ``None`` means the arm never named and never
#: inherited the gate, so it ran the plan's published value.
EXPECTED_JOURNEY_GATES = {
    "all_flags_strict": False,
    "all_numerics_strict": False,
    "base_plus_gdn_flags": False,
    "base_plus_moe_flags": False,
    "base_plus_q8_gr_flags": False,
    "base_plus_q8_selected_down_strict": False,
    "bound": None,
    "decode_strict": None,
    "dense_q8_gr_strict": None,
    "gdn_flags_strict": False,
    "gdn_strict": False,
    "moe_strict": None,
    "production_conservative": False,
    "production_prefill_strict_decode": None,
    "q8_blockscale": None,
    "q8_blockscale_guarded": None,
    "q8_blockscale_guarded_mmq": None,
    "q8_blockscale_mmq": None,
    "q8_selected_down_strict": None,
    "q8down_off_without_dense": None,
    "q8down_off_without_gr_iu8": None,
    "q8down_off_without_gr_iu8_down": None,
    "q8down_off_without_matrix": None,
    "q8down_off_without_q8_mmq_prefill": None,
    "q8down_off_without_qsa": None,
    "strict_plus_dense": False,
    "strict_plus_gdn": True,
    "strict_plus_gdn_multi": True,
    "strict_plus_gdn_serial": False,
    "strict_plus_gr_iu8": False,
    "strict_plus_gr_iu8_down": False,
    "strict_plus_matrix": False,
    "strict_plus_q8_mmq_prefill": False,
    "strict_plus_qsa": False,
    "strict_prefill_production_decode": None,
}


@pytest.fixture(autouse=True)
def _isolate_journey_owner_state():
    """The migration records module state; keep it per-test."""

    saved = (
        dict(journey._GDN_GATE_ARM_VALUES),
        set(journey._STRICT_GATE_ARMS),
        set(journey._PRODUCTION_GATE_ARMS),
        dict(journey._Q8_MMQ_ARM_VALUES),
    )
    yield
    for store, snapshot in (
            (journey._GDN_GATE_ARM_VALUES, saved[0]),
            (journey._STRICT_GATE_ARMS, saved[1]),
            (journey._PRODUCTION_GATE_ARMS, saved[2]),
            (journey._Q8_MMQ_ARM_VALUES, saved[3])):
        store.clear()
        store.update(snapshot)


def _simulate_journey_arms(*, strict_gate, production_gate):
    """Build every arm the way ``main`` does, from simulated bindings."""

    arms = dict(journey.ARMS)
    strict_snapshot = {
        journey.PREFIX + "GR_WIDE": "1" if strict_gate else "0",
        journey.PREFIX + "GDN_REGISTER_PREFILL": "0",
    }
    production_snapshot = {
        journey.PREFIX + "GR_WIDE": "1" if production_gate else "0",
    }
    journey.snapshot_arms(strict_snapshot, production_snapshot, arms=arms)
    journey._GDN_GATE_ARM_VALUES.update(journey.migrate_gdn_colwarps_arms(
        strict_gate=strict_gate, production_gate=production_gate, arms=arms))
    return arms


def test_every_constructed_journey_arm_keeps_its_former_gate():
    """A row per arm, so a new builder arm cannot land without a decision."""

    arms = _simulate_journey_arms(strict_gate=False, production_gate=True)

    assert {name: journey.arm_gdn_colwarps(name) for name in arms} == (
        EXPECTED_JOURNEY_GATES)


def _gate_this_arm_runs(*, arm, strict_gate, production_gate):
    """The value the arm's construction inherited, from its provenance."""

    if arm in journey._PRODUCTION_GATE_ARMS:
        return bool(production_gate)
    if arm in journey._STRICT_GATE_ARMS:
        return bool(strict_gate)
    return journey.arm_gdn_colwarps(arm)


@pytest.mark.parametrize("strict_gate,production_gate", [
    (False, True), (False, False), (True, True), (True, False),
])
@pytest.mark.parametrize("published", [True, False])
@pytest.mark.parametrize("arm", sorted(EXPECTED_JOURNEY_GATES))
def test_a_simulated_snapshot_binds_and_restores_every_arm_gate(
        arm, published, strict_gate, production_gate):
    """Every arm's former gate is bound for the rows and restored after them."""

    _simulate_journey_arms(
        strict_gate=strict_gate, production_gate=production_gate)
    expected = _gate_this_arm_runs(
        arm=arm, strict_gate=strict_gate, production_gate=production_gate)
    if expected is None:
        expected = published
    owner = _owner(published)

    with journey.gdn_colwarps_arm_scope(owner, arm) as effective:
        assert effective is expected
        assert owner.gdn_colwarps_prefill is expected
        assert owner.runner.gdn_colwarps_prefill is expected

    assert owner.gdn_colwarps_prefill is published
    assert owner.runner.gdn_colwarps_prefill is published


@pytest.mark.parametrize("arm,published", [
    ("strict_plus_matrix", True),
    ("all_numerics_strict", True),
    ("strict_plus_gdn", False),
    ("production_conservative", True),
])
def test_a_simulated_snapshot_restores_the_gate_after_a_raising_arm(
        arm, published):
    """The arm's binding is a field, so a failed row must not leak it."""

    _simulate_journey_arms(strict_gate=False, production_gate=True)
    expected = _gate_this_arm_runs(
        arm=arm, strict_gate=False, production_gate=True)
    assert expected is not published
    owner = _owner(published)

    with pytest.raises(RuntimeError, match="row failed"):
        with journey.gdn_colwarps_arm_scope(owner, arm):
            assert owner.gdn_colwarps_prefill is expected
            raise RuntimeError("row failed")

    assert owner.gdn_colwarps_prefill is published
    assert owner.runner.gdn_colwarps_prefill is published


def test_a_non_serial_journey_arm_keeps_the_published_gate():
    """An arm that never named or inherited the gate moves nothing."""

    owner = _owner()

    with journey.gdn_colwarps_arm_scope(owner, "bound") as effective:
        assert effective is True
        assert owner.gdn_colwarps_prefill is True
        assert owner.runner.gdn_colwarps_prefill is True

    assert owner.gdn_colwarps_prefill is True


def test_a_serial_journey_arm_reports_the_value_its_rows_ran_under():
    """The report must carry the measured gate, not the restored publication."""

    owner = _owner()

    with journey.gdn_colwarps_arm_scope(owner, "gdn_strict") as effective:
        assert effective is False
        assert owner.gdn_colwarps_prefill is False
        assert owner.runner.gdn_colwarps_prefill is False

    assert effective is False
    assert owner.gdn_colwarps_prefill is True
    assert owner.runner.gdn_colwarps_prefill is True


@pytest.mark.parametrize("arm,phase,expected", [
    ("strict_prefill_production_decode", "prefill", False),
    ("strict_prefill_production_decode", "decode", True),
    ("production_prefill_strict_decode", "prefill", True),
    ("production_prefill_strict_decode", "decode", False),
    ("bound", "prefill", True),
    ("bound", "decode", True),
])
def test_a_split_phase_binds_the_gate_from_the_phase(arm, phase, expected):
    runner = SimpleNamespace(gdn_colwarps_prefill=True)

    with journey.gdn_colwarps_split_scope(runner, arm, phase):
        assert runner.gdn_colwarps_prefill is expected

    assert runner.gdn_colwarps_prefill is True


@pytest.mark.parametrize("arm,published,serial", [
    # Split arms keep their phase semantics.
    ("strict_prefill_production_decode", True, True),
    ("strict_prefill_production_decode", False, True),
    ("production_prefill_strict_decode", True, False),
    ("production_prefill_strict_decode", False, True),
    # The serial override closes.
    ("gdn_strict", True, True),
    ("strict_plus_gdn_serial", True, True),
    # A strict snapshot's inherited value closes.
    ("strict_plus_matrix", True, True),
    ("all_numerics_strict", True, True),
    ("base_plus_gdn_flags", True, True),
    # A candidate declaration closes.
    ("production_conservative", True, True),
    # A production-GDN isolation arm follows the plan.
    ("strict_plus_gdn", True, False),
    ("strict_plus_gdn", False, True),
    ("strict_plus_gdn_multi", True, False),
    # An arm that named nothing follows the plan, including a closed one.
    ("bound", True, False),
    ("bound", False, True),
    ("moe_strict", False, True),
    ("q8down_off_without_matrix", True, False),
])
def test_the_peer_scope_follows_the_effective_arm_gate(arm, published, serial):
    """The peer owner follows the gate the arm runs under, not its name."""

    _simulate_journey_arms(strict_gate=False, production_gate=published)
    runner = SimpleNamespace(
        gdn_colwarps_prefill=published, gdn_peer_prefill_layers="35-47")

    with journey.phase_peer_scope(runner, arm):
        assert runner.gdn_peer_prefill_layers == (
            None if serial else "35-47")

    assert runner.gdn_peer_prefill_layers == "35-47"


@pytest.mark.parametrize("arm,published,expected", [
    ("strict_plus_matrix", True, True),
    ("all_numerics_strict", False, True),
    ("production_conservative", True, True),
    ("strict_plus_gdn", True, False),
    ("strict_plus_gdn", False, True),
    ("bound", True, False),
    ("bound", False, True),
    ("gdn_strict", True, True),
    ("strict_prefill_production_decode", True, False),
    ("production_prefill_strict_decode", False, True),
])
def test_serial_gdn_follows_the_effective_gate(arm, published, expected):
    """The merged arm environment decided this before the migration."""

    _simulate_journey_arms(strict_gate=False, production_gate=published)

    assert journey.serial_gdn(arm, published) is expected


@pytest.mark.parametrize("arm,published", [
    ("strict_plus_matrix", True),
    ("production_conservative", True),
    ("strict_plus_gdn", False),
    ("bound", False),
])
def test_the_peer_scope_restores_the_published_scope_after_a_raise(
        arm, published):
    _simulate_journey_arms(strict_gate=False, production_gate=published)
    runner = SimpleNamespace(
        gdn_colwarps_prefill=published, gdn_peer_prefill_layers="35-47")

    with pytest.raises(RuntimeError, match="row failed"):
        with journey.phase_peer_scope(runner, arm):
            assert runner.gdn_peer_prefill_layers is None
            raise RuntimeError("row failed")

    assert runner.gdn_peer_prefill_layers == "35-47"


def _trajectory_runner(gate: bool = True):
    runner = SimpleNamespace(
        gdn_colwarps_prefill=gate,
        gdn_peer_prefill_layers="35-47",
        gdn_register_prefill=True,
        q4_pair_prefill=True,
        q8_wave_scale_prefill=True,
        gr_wave_scale=True,
        runtime=SimpleNamespace(device_synchronize=lambda: None),
    )
    observed: list[tuple[str, bool, object]] = []

    def prefill(ids):
        observed.append(
            ("prefill", runner.gdn_colwarps_prefill,
             runner.gdn_peer_prefill_layers))
        return SimpleNamespace(logits=np.zeros(4, dtype=np.float32))

    def step(token):
        observed.append(
            ("decode", runner.gdn_colwarps_prefill,
             runner.gdn_peer_prefill_layers))
        return SimpleNamespace(logits=np.zeros(4, dtype=np.float32))

    runner.reset = lambda: None
    runner.prefill = prefill
    runner.step = step
    return runner, observed


@pytest.mark.parametrize("arm,prefill_gate,decode_gate,prefill_peer", [
    ("strict_prefill_production_decode", False, True, None),
    ("production_prefill_strict_decode", True, False, "35-47"),
])
def test_a_split_trajectory_binds_the_gate_per_phase(
        monkeypatch, arm, prefill_gate, decode_gate, prefill_peer):
    """The measured prefill and decode rows see their own phase's gate."""

    monkeypatch.setattr(journey, "clear_arm_graphs", lambda runner: None)
    runner, observed = _trajectory_runner()

    journey.split_trajectory(runner, [1, 2], [3, 4], {}, {}, arm=arm)

    phases = [row for row in observed if row[0] == "prefill"]
    decodes = [row for row in observed if row[0] == "decode"]
    assert [row[1] for row in phases] == [prefill_gate]
    assert [row[2] for row in phases] == [prefill_peer]
    assert decodes and all(row[1] is decode_gate for row in decodes)
    assert runner.gdn_colwarps_prefill is True
    assert runner.gdn_peer_prefill_layers == "35-47"
