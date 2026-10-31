"""Every Q8_0 expert-down consumer binds the owner its arm means.

``HIPENGINE_QWEN4_EXP_Q8_DOWN_VARIANT`` retired into a ``CandidateSpec``
declaration and an owner field on the generator/runner. A harness that still
applied only ``spec.environment`` therefore measured the plan's published leaf
while naming a different arm: a ``q8_blockscale`` arm silently ran the guarded
block-scale default, and a probe that counts or instruments the unguarded WMMA
parent counted nothing.

What these tests pin:

* **Shared helper.** One owner scope binds a named leaf, keeps the published
  value when the arm names none, and restores the published value on exit,
  including after a raise.
* **Candidate consumers.** Every script that applies a candidate's environment
  binds the owner through the shared helper after the profile is constructed,
  and none writes the retired key. The two consumers whose candidates declare
  no owner are named explicitly so a new consumer cannot slip in.
* **Diagnostic arms.** ``journey_localize`` derives its arm map from the
  candidate declarations; the probes that count or instrument the WMMA parent
  declare that leaf for their own arms; and the ``q8-wmma-down`` route scope
  binds the counted leaf while the plan's owner comes back on exit.

CPU only: no GPU, no kernels, no performance claim.
"""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path

import pytest

from hipengine.runtime import qwen4_exp_runner as runtime
from scripts import qwen4exp_halo_box_campaign_ab as campaign
from scripts import qwen4exp_journey_localize as localize
from scripts import qwen4exp_layer2_profile_gate as gate

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
RETIRED_KEY = "HIPENGINE_QWEN4_EXP_Q8_DOWN_VARIANT"
WMMA = runtime.Q8_DOWN_WMMA_VARIANT
BLOCKSCALE = "selected_grouped_blockscale_prefill_bf16_bf16_out"
GUARDED = runtime.Q8_DOWN_GUARDED_VARIANT

#: Candidate consumers that bind the arm's owner, and the helper each uses.
OWNER_BINDING_CONSUMERS = {
    "qwen4exp_active_task_checks.py": "bind_candidate_q8_down_variant",
    "qwen4exp_layer2_profile_gate.py": "bind_candidate_q8_down_variant",
    "qwen4exp_long_context_gate.py": "bind_candidate_q8_down_variant",
    "qwen4exp_prefill_restore_ab.py": "q8_down_variant_arm_scope",
    "qwen4exp_q8_repair_depth_gate.py": "bind_candidate_q8_down_variant",
    "qwen4exp_q8_repair_tasks.py": "bind_candidate_q8_down_variant",
    "qwen4exp_journey_localize.py": "q8_down_arm_scope",
}
#: Candidate consumers whose only candidate declares no owner. Every other
#: consumer runs an arbitrary ``CANDIDATES`` entry, so it must bind.
OWNER_FREE_CONSUMERS = {
    "qwen4exp_conservative_cost.py": "production_conservative",
    "qwen4exp_gr_router_localize.py": "production_gr_up_restore",
}


class _Owner:
    """A generator-shaped owner: the scope binds it and its runner."""

    def __init__(self, published: str | None = None):
        self.q8_down_variant = published
        self.runner = type("Runner", (), {"q8_down_variant": published})()


@pytest.fixture
def arm_state():
    """Snapshot the module-level arm tables ``define_arms`` mutates."""

    arms, variants = dict(localize.ARMS), dict(localize._Q8_DOWN_ARM_VARIANTS)
    mmq = dict(localize._Q8_MMQ_ARM_VALUES)
    try:
        yield
    finally:
        localize.ARMS.clear()
        localize.ARMS.update(arms)
        localize._Q8_DOWN_ARM_VARIANTS.clear()
        localize._Q8_DOWN_ARM_VARIANTS.update(variants)
        localize._Q8_MMQ_ARM_VALUES.clear()
        localize._Q8_MMQ_ARM_VALUES.update(mmq)


# --------------------------------------------------------------------------
# The shared owner scope
# --------------------------------------------------------------------------


def test_the_arm_scope_binds_a_named_leaf_and_restores_the_published_one():
    owner = _Owner(GUARDED)

    with gate.q8_down_variant_arm_scope(owner, BLOCKSCALE):
        assert owner.q8_down_variant == BLOCKSCALE
        assert owner.runner.q8_down_variant == BLOCKSCALE

    assert owner.q8_down_variant == GUARDED
    assert owner.runner.q8_down_variant == GUARDED


def test_the_arm_scope_keeps_a_published_owner_when_the_arm_names_none():
    owner = _Owner(GUARDED)

    with gate.q8_down_variant_arm_scope(owner, None):
        assert owner.q8_down_variant == GUARDED

    assert owner.q8_down_variant == GUARDED


def test_the_arm_scope_restores_the_published_owner_after_a_raise():
    owner = _Owner(GUARDED)

    with pytest.raises(RuntimeError):
        with gate.q8_down_variant_arm_scope(owner, WMMA):
            assert owner.q8_down_variant == WMMA
            raise RuntimeError("arm failed mid-run")

    assert owner.q8_down_variant == GUARDED


def test_the_candidate_binder_applies_the_declaration_on_a_stack():
    from contextlib import ExitStack

    spec = gate.CANDIDATES["q8_blockscale"]
    owner = _Owner(GUARDED)
    stack = ExitStack()
    with stack:
        bound = gate.bind_candidate_q8_down_variant(stack, owner, spec)
        assert bound == BLOCKSCALE
        assert owner.q8_down_variant == BLOCKSCALE
    assert owner.q8_down_variant == GUARDED


def test_the_candidate_binder_keeps_the_plan_owner_for_an_undeclared_arm():
    from contextlib import ExitStack

    spec = gate.CANDIDATES["production_baseline"]
    owner = _Owner(GUARDED)
    stack = ExitStack()
    with stack:
        assert gate.bind_candidate_q8_down_variant(stack, owner, spec) == GUARDED
        assert owner.q8_down_variant == GUARDED


# --------------------------------------------------------------------------
# journey_localize's arm map
# --------------------------------------------------------------------------


def test_the_journey_arm_map_names_each_declared_candidate_leaf():
    declared = localize.declared_q8_down_variants()

    assert declared == {
        "q8_blockscale": BLOCKSCALE,
        "q8_blockscale_mmq": BLOCKSCALE,
        "q8_blockscale_guarded": GUARDED,
        "q8_blockscale_guarded_mmq": GUARDED,
    }
    for arm, leaf in declared.items():
        assert localize.q8_down_arm_variant(arm) == leaf


def test_a_journey_arm_without_a_declaration_keeps_the_published_owner():
    assert localize.q8_down_arm_variant("production_conservative") is None
    owner = _Owner(GUARDED)

    with localize.q8_down_arm_scope(owner, "production_conservative"):
        assert owner.q8_down_variant == GUARDED

    assert owner.q8_down_variant == GUARDED


def test_the_journey_arm_report_names_the_bound_leaf():
    owner = _Owner(GUARDED)

    with localize.q8_down_arm_scope(owner, "q8_blockscale"):
        assert localize.q8_down_arm_report(owner, "q8_blockscale") == BLOCKSCALE

    assert localize.q8_down_arm_report(owner, "q8_blockscale") == GUARDED


def test_declare_q8_down_owner_names_a_probe_arm(arm_state):
    localize.declare_q8_down_owner(WMMA, arms=["down_off_2"])

    assert localize.q8_down_arm_variant("down_off_2") == WMMA
    owner = _Owner(GUARDED)
    with localize.q8_down_arm_scope(owner, "down_off_2"):
        assert owner.q8_down_variant == WMMA
    assert owner.q8_down_variant == GUARDED


# --------------------------------------------------------------------------
# Probes that count or instrument the unguarded WMMA parent
# --------------------------------------------------------------------------


def test_the_q8_localize_probe_binds_the_wmma_parent_for_every_arm_it_defines(
    arm_state,
):
    q8_localize = importlib.import_module("scripts.qwen4exp_q8_localize")

    defined = q8_localize.define_arms()

    assert defined, "the probe must define arms"
    for arm in defined:
        assert localize.q8_down_arm_variant(arm) == WMMA


def test_the_q8_localize_arms_keep_their_layer_and_mmq_controls(arm_state):
    q8_localize = importlib.import_module("scripts.qwen4exp_q8_localize")

    defined = q8_localize.define_arms()

    for layer in (2, 4, 30, 46, 47):
        arm = f"down_off_{layer}"
        assert arm in defined
        assert localize.ARMS[arm][q8_localize.LAYER_FLAG] == str(layer)
        # The historical arm disabled Q8 MMQ as well; that control stays.
        assert localize._Q8_MMQ_ARM_VALUES[arm] is False


def test_the_boundary_replay_probe_binds_the_wmma_parent_and_skips_declared_arms(
    arm_state,
):
    replay = importlib.import_module("scripts.qwen4exp_q8_boundary_replay")

    arms = replay.declare_arm_owners()

    assert "moe_strict" in arms
    assert "q8_blockscale" not in arms
    assert localize.q8_down_arm_variant("moe_strict") == WMMA
    # An arm that declares another leaf keeps it: the probe cannot instrument it.
    assert localize.q8_down_arm_variant("q8_blockscale") == BLOCKSCALE


def test_the_wmma_logits_probe_binds_the_owner_for_its_candidate_rows():
    probe = importlib.import_module("scripts.qwen4exp_q8_0_wmma_down_logits_probe")
    source = inspect.getsource(probe.main)

    assert "q8_down_variant(generator, Q8_DOWN_WMMA_VARIANT)" in source
    assert 'if enabled == "1"' in source
    assert '"candidate_owner": Q8_DOWN_WMMA_VARIANT' in inspect.getsource(probe)


def test_the_q8_wmma_down_route_scope_binds_the_counted_leaf():
    runner = _Owner(GUARDED).runner

    with campaign.route_scope("q8-wmma-down", "after", runner, environment={}):
        assert runner.q8_down_variant == WMMA

    assert runner.q8_down_variant == GUARDED


def test_the_state_gate_records_the_counted_and_published_owner():
    source = (SCRIPTS / "qwen4exp_row4_state_gate.py").read_text()

    assert '"runner_field": "q8_down_variant"' in source
    assert '"published": resolve_q8_down_variant(generator.runner)' in source
    assert '"bound": Q8_DOWN_WMMA_VARIANT' in source


# --------------------------------------------------------------------------
# Every candidate-environment consumer binds the owner
# --------------------------------------------------------------------------


def _candidate_environment_consumers() -> set[str]:
    consumers = set()
    for path in sorted(SCRIPTS.glob("qwen4exp_*.py")):
        text = path.read_text()
        if "CANDIDATES[" in text and ".environment" in text:
            consumers.add(path.name)
    return consumers


def test_every_candidate_environment_consumer_binds_the_owner():
    expected = set(OWNER_BINDING_CONSUMERS) | set(OWNER_FREE_CONSUMERS)

    assert _candidate_environment_consumers() == expected
    for name, helper in OWNER_BINDING_CONSUMERS.items():
        text = (SCRIPTS / name).read_text()
        assert helper in text, f"{name} does not use {helper}"


def test_the_owner_free_consumers_run_candidates_that_declare_no_owner():
    for name, candidate in OWNER_FREE_CONSUMERS.items():
        assert gate.CANDIDATES[candidate].q8_down_variant is None, name


def test_each_candidate_consumer_uses_the_shared_helper():
    expected = {
        "qwen4exp_active_task_checks": gate.bind_candidate_q8_down_variant,
        "qwen4exp_long_context_gate": gate.bind_candidate_q8_down_variant,
        "qwen4exp_q8_repair_depth_gate": gate.bind_candidate_q8_down_variant,
        "qwen4exp_q8_repair_tasks": gate.bind_candidate_q8_down_variant,
        "qwen4exp_prefill_restore_ab": gate.q8_down_variant_arm_scope,
    }
    for name, helper in expected.items():
        module = importlib.import_module(f"scripts.{name}")
        assert getattr(module, helper.__name__) is helper, name


def test_no_harness_writes_the_retired_process_key():
    offenders = []
    for path in sorted(SCRIPTS.glob("qwen4exp_*.py")):
        text = path.read_text()
        if f'"{RETIRED_KEY}"' in text or f"'{RETIRED_KEY}'" in text:
            offenders.append(path.name)
    assert offenders == []


def test_no_candidate_environment_carries_the_retired_key():
    for name, spec in gate.CANDIDATES.items():
        assert "Q8_DOWN_VARIANT" not in spec.environment, name
