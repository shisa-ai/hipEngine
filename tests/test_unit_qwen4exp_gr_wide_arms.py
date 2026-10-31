"""GR-up wide owner bindings in every candidate-driven diagnostic harness.

``HIPENGINE_QWEN4_EXP_GR_WIDE`` used to be a process key that the profile binder
published and each harness overrode by copying a candidate's environment. The
route is the owning runner's field now, so every harness that applies a
candidate environment has to bind that field instead. These tests cover the part
that can be wrong without a GPU:

- the shared scope/binder reads ``CandidateSpec.gr_wide`` and restores the plan's
  route on unwind;
- each harness's per-arm helper reproduces the value its former process key
  carried, including the ``None`` case where the arm never named the key and ran
  whatever the bind had published;
- the journey localization reconstructs both bindings' former snapshots and
  records the same per-arm table the environment produced before retirement.
"""

from __future__ import annotations

import contextlib
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import qwen4exp_layer2_profile_gate as gate  # noqa: E402
from scripts.qwen4exp_layer2_profile_gate import CANDIDATES  # noqa: E402

#: Harnesses whose only GR-up wide input is ``CandidateSpec.environment`` plus
#: the per-arm owner fields they already bind.
CANDIDATE_HARNESSES = (
    "qwen4exp_prefill_restore_ab",
    "qwen4exp_active_task_checks",
    "qwen4exp_q8_repair_tasks",
    "qwen4exp_long_context_gate",
)


def _harness(name):
    return importlib.import_module("scripts." + name)


def test_shared_scope_binds_only_an_explicit_route():
    runner = SimpleNamespace(gr_wide=True)
    with gate.gr_wide_arm_scope(runner, None):
        assert runner.gr_wide is True
    with gate.gr_wide_arm_scope(runner, False):
        assert runner.gr_wide is False
    assert runner.gr_wide is True
    with gate.gr_wide_arm_scope(runner, True):
        assert runner.gr_wide is True


def test_shared_binder_reports_the_route_it_bound():
    generator = SimpleNamespace(gr_wide=True, runner=SimpleNamespace(gr_wide=True))
    with contextlib.ExitStack() as stack:
        assert gate.bind_candidate_gr_wide(
            stack, generator, SimpleNamespace(gr_wide=False)) is False
        assert generator.runner.gr_wide is False
    assert generator.runner.gr_wide is True
    with contextlib.ExitStack() as stack:
        assert gate.bind_candidate_gr_wide(
            stack, generator, SimpleNamespace(gr_wide=None)) is True
        assert generator.runner.gr_wide is True


@pytest.mark.parametrize(
    "name, expected",
    (
        ("production_conservative", False),
        ("recovery_plan", False),
        ("recovery_plus_routed_f16", False),
        ("recovery_plus_gr_wide", True),
        ("fast_composition", True),
        ("production_gr_up_wide", True),
        ("production_baseline", None),
        ("ar_batch", None),
    ),
)
def test_candidate_declares_its_former_wide_route(name, expected):
    """The candidate declaration is the arm's former process value."""

    assert CANDIDATES[name].gr_wide is expected


@pytest.mark.parametrize("module_name", CANDIDATE_HARNESSES)
def test_harness_arm_route_follows_the_candidate_declaration(module_name):
    module = _harness(module_name)
    # An arm that names the former key keeps its value in both directions.
    assert module.arm_gr_wide("production_conservative", True) is False
    assert module.arm_gr_wide("recovery_plan", True) is False
    assert module.arm_gr_wide("recovery_plus_gr_wide", False) is True
    # An arm that never named the key ran whatever the bind had published, and
    # the strict arm's bind publishes the fused exact parent.
    assert module.arm_gr_wide("production_baseline", True) is True
    assert module.arm_gr_wide("production_baseline", False) is False
    assert module.arm_gr_wide("strict", False) is False


def test_conservative_cost_keeps_the_former_before_after_pair():
    module = _harness("qwen4exp_conservative_cost")
    assert module.arm_gr_wide("before", True) is True
    assert module.arm_gr_wide("after", True) is False
    assert module.arm_gr_wide("after", False) is False


def test_journey_snapshot_carries_the_bindings_route():
    module = _harness("qwen4exp_journey_localize")
    flags = {"HIPENGINE_QWEN4_EXP_GR_IU8": "0"}
    strict = module.gr_wide_snapshot(flags, published=False)
    production = module.gr_wide_snapshot(flags, published=True)
    assert strict["HIPENGINE_QWEN4_EXP_GR_WIDE"] == "0"
    assert production["HIPENGINE_QWEN4_EXP_GR_WIDE"] == "1"
    assert strict["HIPENGINE_QWEN4_EXP_GR_IU8"] == "0"
    # The snapshot is a copy: the caller's flag dict is not mutated.
    assert "HIPENGINE_QWEN4_EXP_GR_WIDE" not in flags


def test_journey_arms_keep_the_route_their_former_environment_carried():
    """The whole arm table, built the way ``main`` builds it."""

    module = _harness("qwen4exp_journey_localize")
    strict = module.gr_wide_snapshot(
        {
            "HIPENGINE_QWEN4_EXP_GR_IU8": "0",
            "HIPENGINE_QWEN4_EXP_GR_IU8_DOWN": "0",
            "HIPENGINE_QWEN4_EXP_Q8_MMQ_PREFILL": "0",
            "HIPENGINE_QWEN4_EXP_Q4_IU8_PREFILL": "0",
            "HIPENGINE_QWEN4_EXP_GDN_COLWARPS_PREFILL": "0",
        },
        published=False,
    )
    production = module.gr_wide_snapshot(
        {
            "HIPENGINE_QWEN4_EXP_GR_IU8": "0",
            "HIPENGINE_QWEN4_EXP_GR_IU8_DOWN": "0",
            "HIPENGINE_QWEN4_EXP_Q8_MMQ_PREFILL": "1",
            "HIPENGINE_QWEN4_EXP_Q4_IU8_PREFILL": "1",
            "HIPENGINE_QWEN4_EXP_GDN_COLWARPS_PREFILL": "1",
        },
        published=True,
    )
    arms = {name: dict(env) for name, env in module.ARMS.items()}
    module.snapshot_arms(strict, production, arms=arms)
    values = module.migrate_gr_wide_arms(
        arms, production_value=True, declared=module.declared_gr_wide())

    # Strict snapshots: the arm ran the strict bind's route.
    for name in ("all_flags_strict", "strict_plus_matrix", "strict_plus_qsa",
                 "strict_plus_gdn", "strict_plus_gdn_serial",
                 "strict_plus_gdn_multi", "strict_plus_q8_mmq_prefill",
                 "strict_plus_gr_iu8", "strict_plus_gr_iu8_down",
                 "base_plus_q8_gr_flags", "q8down_off_without_dense"):
        assert values[name] is False, name
    # The dense family took the production values for its ``GR_*`` keys.
    assert values["strict_plus_dense"] is True
    # Arms that never named the key ran the production bind's route.
    for name in ("bound", "gdn_flags_strict", "all_numerics_strict",
                 "gdn_strict", "moe_strict", "dense_q8_gr_strict",
                 "decode_strict", "base_plus_moe_flags", "base_plus_gdn_flags",
                 "q8_selected_down_strict", "base_plus_q8_selected_down_strict",
                 "strict_prefill_production_decode",
                 "production_prefill_strict_decode",
                 "q8down_off_without_matrix", "q8down_off_without_qsa"):
        assert values[name] is True, name
    # Candidate arms keep the declaration their environment carried.
    assert values["production_conservative"] is False
    # No harness writes the retired key as a process value any more.
    assert all("HIPENGINE_QWEN4_EXP_GR_WIDE" not in env for env in arms.values())


def test_journey_arm_enabled_falls_back_to_the_published_route():
    module = _harness("qwen4exp_journey_localize")
    assert module.gr_wide_arm_enabled("production_conservative", True) is False
    assert module.gr_wide_arm_enabled("not-an-arm", True) is True
    assert module.gr_wide_arm_enabled("not-an-arm", False) is False


def test_journey_split_arm_closes_the_tile_only_in_its_strict_phase():
    module = _harness("qwen4exp_journey_localize")
    runner = SimpleNamespace(gr_wide=True)
    with module.gr_wide_split_scope(
            runner, "strict_prefill_production_decode", "prefill"):
        assert runner.gr_wide is False
    assert runner.gr_wide is True
    with module.gr_wide_split_scope(
            runner, "strict_prefill_production_decode", "decode"):
        assert runner.gr_wide is True
    with module.gr_wide_split_scope(
            runner, "production_prefill_strict_decode", "prefill"):
        assert runner.gr_wide is True
    with module.gr_wide_split_scope(
            runner, "production_prefill_strict_decode", "decode"):
        assert runner.gr_wide is False
    assert runner.gr_wide is True
    with module.gr_wide_split_scope(runner, "bound", "prefill"):
        assert runner.gr_wide is True


def test_wide_arm_declares_every_leaf_the_composition_can_take():
    """The F16-direct composition runs a different leaf than the F32 tile."""

    spec = CANDIDATES["production_gr_up_wide"]
    assert spec.direct_dispatch_target == (
        "hipengine.runtime.qwen4_exp_runner", "gguf_q8_0_dense_wide256_f32_f32_out")
    assert {target[1] for target in spec.extra_direct_dispatch_targets} == {
        "qwen4_exp_gr_up_mean_residual_f16in_f32_out",
        "qwen4_exp_gr_up_mean_f16in_f32_out",
    }


def test_counter_names_the_f16_direct_leaf_it_observed(monkeypatch):
    """Engagement counting follows the leaf that ran, not the declared tile."""

    from collections import Counter

    from scripts.qwen4exp_candidate_dispatch import count_candidate_dispatch

    runner_module = importlib.import_module("hipengine.runtime.qwen4_exp_runner")
    seen = []
    for name in ("qwen4_exp_gr_up_mean_residual_f16in_f32_out",
                 "qwen4_exp_gr_up_mean_f16in_f32_out"):
        monkeypatch.setattr(runner_module, name,
                            lambda *args, _name=name: seen.append(_name))
    spec = CANDIDATES["production_gr_up_wide"]
    with count_candidate_dispatch(
        direct_target=spec.direct_dispatch_target,
        direct_reference=spec.direct_dispatch_reference,
        shape_positions=spec.dispatch_shape_positions,
        extra_direct_targets=spec.extra_direct_dispatch_targets,
    ) as counter:
        runner_module.qwen4_exp_gr_up_mean_residual_f16in_f32_out(1, 2, 3)
    assert seen == ["qwen4_exp_gr_up_mean_residual_f16in_f32_out"]
    assert counter["calls"] == 1
    assert counter["leaves"] == {
        "hipengine.runtime.qwen4_exp_runner."
        "qwen4_exp_gr_up_mean_residual_f16in_f32_out": 1}
    # Shape records describe the declared primary target's argument positions.
    assert counter["shapes"] == Counter()
