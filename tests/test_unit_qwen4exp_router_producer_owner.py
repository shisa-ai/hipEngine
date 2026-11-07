"""The router producer is a plan-level choice on the runner, not a process key.

R0029: gate harnesses need a plan-level producer option before the launch-time
environment reader and the binder's environment publication are removed. These
tests pin that contract: the runner field is the single owner, the launch
selection ignores a stale environment value, harness arms bind the runner they
measure, and the binder publishes owner state instead of a process key.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from hipengine.execution_profiles import (
    ExecutionProfile,
    resolve_runtime_profile,
)
from hipengine.generation import qwen4_exp_profiles as profiles
from hipengine.generation.qwen4_exp_gguf import _inherit_lane_decisions
from hipengine.runtime import qwen4_exp_runner as runner

ENV = "HIPENGINE_QWEN4_EXP_ROUTER_VARIANT"
TILE4 = "f32_hidden_token_tile4_dense_exact"
WARP_REDUCE = "f32_hidden_warp_reduce_dense_exact"
DPP = "f32_hidden_token_tile4_dpp_exact"
F16_GEMM = "f32_hidden_f16_wmma_gemm"


@pytest.mark.parametrize("stale", ["", "1", "0", WARP_REDUCE, DPP, F16_GEMM, "bogus"])
@pytest.mark.parametrize(
    "requested,expected",
    [(None, TILE4), (TILE4, TILE4), (WARP_REDUCE, WARP_REDUCE), (DPP, DPP)],
)
def test_launch_selection_reads_the_owner_not_the_environment(
    monkeypatch, stale, requested, expected,
) -> None:
    """A stale environment value must not change the producer the plan named."""

    monkeypatch.setenv(ENV, stale)
    assert runner._qwen4_exp_router_producer_variant(
        1024, requested=requested,
    ) == expected


def test_launch_selection_keeps_the_linear_path_and_the_small_row_guard(
    monkeypatch,
) -> None:
    monkeypatch.setenv(ENV, DPP)
    assert runner._qwen4_exp_router_producer_variant(1, requested=DPP) is None
    # The F16 WMMA GEMM stages 128-row tiles; small widths keep the exact DPP
    # producer, and that guard belongs to the owner value, not the environment.
    assert runner._qwen4_exp_router_producer_variant(
        4, requested=F16_GEMM,
    ) == DPP
    assert runner._qwen4_exp_router_producer_variant(
        4096, requested=F16_GEMM,
    ) == F16_GEMM


def test_launch_selection_rejects_an_unrecognized_owner_value(monkeypatch) -> None:
    monkeypatch.delenv(ENV, raising=False)
    with pytest.raises(ValueError, match="router"):
        runner._qwen4_exp_router_producer_variant(1024, requested="f32_hidden_nope")


def test_moe_dispatch_takes_the_producer_from_its_caller() -> None:
    """The MoE entry point must accept the owner value and forward it."""

    import inspect

    signature = inspect.signature(runner.run_qwen4_exp_moe)
    assert signature.parameters["router_producer_variant"].default is None
    source = inspect.getsource(runner.run_qwen4_exp_moe)
    assert "router_producer_variant=router_producer_variant" in source
    # The retired launch-time reader is gone, not merely bypassed.
    assert not hasattr(runner, "_ROUTER_PRODUCER_VARIANT_ENV")
    selector = inspect.getsource(runner._qwen4_exp_router_producer_variant)
    assert "os.environ" not in selector
    assert "HIPENGINE_QWEN4_EXP_ROUTER_VARIANT" not in selector


def test_resident_runner_publishes_the_field_and_forwards_it() -> None:
    import inspect

    assert "router_producer_variant" in runner.Qwen4ExpGGUFResidentModelRunner.__annotations__
    source = inspect.getsource(runner.Qwen4ExpGGUFResidentModelRunner)
    assert "router_producer_variant=self.router_producer_variant" in source


@pytest.mark.parametrize("mode", [TILE4, WARP_REDUCE, DPP, F16_GEMM])
def test_scope_binds_and_restores_the_owning_runner(mode) -> None:
    owner = SimpleNamespace(router_producer_variant=F16_GEMM)
    neighbor = SimpleNamespace(router_producer_variant=None)
    with runner.router_producer_variant(owner, mode) as prior:
        assert prior == F16_GEMM
        assert owner.router_producer_variant == mode
        assert neighbor.router_producer_variant is None
    assert owner.router_producer_variant == F16_GEMM


def test_nested_scope_restores_the_owner_on_failure() -> None:
    owner = SimpleNamespace(router_producer_variant=F16_GEMM)
    with pytest.raises(RuntimeError, match="arm failed"):
        with runner.router_producer_variant(owner, DPP):
            with runner.router_producer_variant(owner, TILE4):
                assert owner.router_producer_variant == TILE4
            assert owner.router_producer_variant == DPP
            raise RuntimeError("arm failed")
    assert owner.router_producer_variant == F16_GEMM
    with pytest.raises(ValueError, match="router"):
        with runner.router_producer_variant(owner, "bogus"):
            pass
    assert owner.router_producer_variant == F16_GEMM


@pytest.mark.parametrize("mode", [None, TILE4, WARP_REDUCE, DPP, F16_GEMM])
def test_clones_inherit_the_profile_choice(mode) -> None:
    generator = SimpleNamespace(router_producer_variant=mode)
    assert _inherit_lane_decisions(
        generator, SimpleNamespace(),
    ).router_producer_variant == mode
    assert _inherit_lane_decisions(
        SimpleNamespace(), SimpleNamespace(),
    ).router_producer_variant is None


@pytest.mark.parametrize("stale", ["", WARP_REDUCE, DPP, F16_GEMM])
def test_profile_selects_the_plan_default_without_the_environment(
    monkeypatch, stale,
) -> None:
    monkeypatch.setenv(ENV, stale)
    assert profiles._selected_router_producer_variant() == TILE4
    assert profiles._selected_router_producer_variant(recovery=True) == DPP


def test_binder_publishes_owner_state_instead_of_the_process_key(monkeypatch) -> None:
    """The resolved selection reaches the generator and runner, not os.environ."""

    profiles.register_qwen4_exp_gfx1151_profiles()
    monkeypatch.delenv(ENV, raising=False)
    production = resolve_runtime_profile(
        model=profiles.QWEN4_EXP_MODEL, backend=profiles.QWEN4_EXP_BACKEND,
        quant="gguf_ud_q4_k_xl", profile=ExecutionProfile.PRODUCTION,
    )

    owner = SimpleNamespace(runner=SimpleNamespace())
    production.binder(owner, production)
    selected = next(
        row["selected_variant"] for row in production.manifest["selections"]
        if row["layer"] == "router_logits"
    )
    assert selected == F16_GEMM
    assert owner.router_producer_variant == selected
    assert owner.runner.router_producer_variant == selected
    assert ENV not in os.environ


def test_binder_reports_a_discarded_pre_bind_value(monkeypatch) -> None:
    profiles.register_qwen4_exp_gfx1151_profiles()
    monkeypatch.setenv(ENV, TILE4)
    production = resolve_runtime_profile(
        model=profiles.QWEN4_EXP_MODEL, backend=profiles.QWEN4_EXP_BACKEND,
        quant="gguf_ud_q4_k_xl", profile=ExecutionProfile.PRODUCTION,
    )
    owner = SimpleNamespace(runner=SimpleNamespace())
    production.binder(owner, production)
    # A process value the owner publication makes inert is still reported, so a
    # caller cannot believe its route ran.
    assert profiles.profile_ignored_overrides(owner)[ENV] == (TILE4, F16_GEMM)
    # The binder no longer writes the key, so the caller's inert value stays
    # where it was rather than being replaced by a value it never published.
    assert os.environ[ENV] == TILE4


def test_strict_plan_publishes_the_exact_parent(monkeypatch) -> None:
    profiles.register_qwen4_exp_gfx1151_profiles()
    monkeypatch.delenv(ENV, raising=False)
    strict = resolve_runtime_profile(
        model=profiles.QWEN4_EXP_MODEL, backend=profiles.QWEN4_EXP_BACKEND,
        quant="gguf_ud_q4_k_xl", profile=ExecutionProfile.STRICT,
    )
    owner = SimpleNamespace(runner=SimpleNamespace())
    strict.binder(owner, strict)
    assert owner.router_producer_variant == TILE4
    assert owner.runner.router_producer_variant == TILE4
    assert ENV not in os.environ


def test_campaign_arms_bind_the_runner_and_leave_the_environment_alone() -> None:
    from scripts.qwen4exp_halo_box_campaign_ab import _apply_mode, route_scope

    owner = SimpleNamespace(router_producer_variant=F16_GEMM)
    environment = {"unrelated": "unchanged"}
    for mode, expected in (("before", TILE4), ("after", DPP)):
        _apply_mode(mode, route_package="router-dpp", environment=environment)
        assert environment == {"unrelated": "unchanged"}
        with route_scope("router-dpp", mode, owner, environment=environment):
            assert owner.router_producer_variant == expected
            assert environment == {"unrelated": "unchanged"}
        assert owner.router_producer_variant == F16_GEMM


def test_state_gate_arm_binds_the_runner_and_leaves_the_environment_alone() -> None:
    from scripts.qwen4exp_row4_state_gate import apply_state_gate_mode

    environment = {"unrelated": "unchanged"}
    apply_state_gate_mode(None, "router-dpp", "1", "", environment=environment)
    assert environment == {"unrelated": "unchanged"}


def test_layer2_arms_declare_the_producer_on_the_candidate() -> None:
    from scripts.qwen4exp_layer2_profile_gate import (
        CANDIDATES, _F16_ROUTES_OFF, _candidate_named_profile_intact,
        _f16_route_isolation_candidates,
    )

    assert CANDIDATES["router_warp_reduce"].router_producer_variant == WARP_REDUCE
    assert CANDIDATES["router_warp_reduce"].environment == {}
    assert ENV not in _F16_ROUTES_OFF
    arms = _f16_route_isolation_candidates()
    assert arms["recovery_plan"].router_producer_variant == DPP
    assert arms["recovery_plus_indexer_router_f16"].router_producer_variant == F16_GEMM
    assert not _candidate_named_profile_intact(
        SimpleNamespace(
            environment={}, qsa_wmma_enabled=None,
            qsa_flash_prefill_enabled=None,
            q4_iu8_prefill_enabled=None,
            q4_iu8_layers=None, q4_iu8_planes=None, q4_iu8_risk_mult=None,
            q8_selected_down_enabled=None,
            gr_iu8_up=None, gr_iu8_down=None,
            production_moe_prefill_enabled=None,
            q8_mmq_prefill_enabled=None,
            router_producer_variant=DPP,
        )
    )


def test_router_arm_is_attributed_by_dispatch_not_by_environment() -> None:
    """A named producer must be provable from dispatch counts, not the shell."""

    from scripts.qwen4exp_layer2_profile_gate import CANDIDATES

    arm = CANDIDATES["router_warp_reduce"]
    assert arm.count_registered_dispatch is True
    assert arm.candidate_key == ("hip_gfx1151", "router_logits", "f32", WARP_REDUCE)
    assert arm.fallback_key == ("hip_gfx1151", "router_logits", "f32", TILE4)
    assert arm.dispatch_shape_positions == (3, 4, 5)


def test_profile_gap_names_the_producer_by_flag_not_override() -> None:
    from pathlib import Path

    from scripts.qwen4exp_profile_gap import (
        ROUTE_ENV_KEYS, _parse_overrides, build_parser,
    )

    assert ENV not in ROUTE_ENV_KEYS
    with pytest.raises(ValueError, match="--router-producer-variant"):
        _parse_overrides([f"{ENV}={DPP}"])
    args = build_parser().parse_args([
        "--model-root", "/unused", "--mode", "prefill",
        "--case-id", "case", "--output", "/unused.json",
        "--router-producer-variant", WARP_REDUCE,
    ])
    assert args.router_producer_variant == WARP_REDUCE


def _keyword(call, name):
    for keyword in call.keywords:
        if keyword.arg == name:
            return keyword.value
    return None


def _layer_calls(source):
    import ast
    import textwrap

    tree = ast.parse(textwrap.dedent(source))
    return [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id.startswith("run_qwen4_exp_")
        and (node.func.id.endswith("layer") or node.func.id.endswith("_moe"))
    ]


@pytest.mark.parametrize(
    "function",
    [
        runner.run_qwen4_exp_moe,
        runner.run_qwen4_exp_dense_qsa_layer,
        runner.run_qwen4_exp_qsa_prefill_layer,
        runner.run_qwen4_exp_gdn_layer,
    ],
)
def test_every_moe_forwarder_carries_the_producer(function) -> None:
    import inspect

    parameters = inspect.signature(function).parameters
    assert "router_producer_variant" in parameters, function.__qualname__
    assert parameters["router_producer_variant"].default is None

    calls = [
        call for call in _layer_calls(inspect.getsource(function))
        if call.func.id == "run_qwen4_exp_moe"
    ]
    assert calls, f"{function.__qualname__} forwards no MoE call"
    for call in calls:
        value = _keyword(call, "router_producer_variant")
        assert value is not None, (function.__qualname__, call.func.id)
        assert getattr(value, "id", None) == "router_producer_variant"


def test_the_resident_runner_forwards_its_own_field() -> None:
    import inspect

    source = inspect.getsource(runner.Qwen4ExpGGUFResidentModelRunner)
    calls = [
        call for call in _layer_calls(source)
        if call.func.id.startswith("run_qwen4_exp_")
        and call.func.id.endswith("layer")
    ]
    assert len(calls) == 4
    for call in calls:
        value = _keyword(call, "router_producer_variant")
        assert value is not None, call.func.id
        assert getattr(value, "attr", None) == "router_producer_variant"


def test_the_batch_owner_carries_the_plan_decision() -> None:
    """A c1 captured segment bakes the router kernel in, so the carrier and
    the batch owner must hand the plan's producer over rather than default it."""

    import inspect

    from hipengine.generation import qwen4_exp_ar_batch
    from hipengine.runtime import qwen4_exp_batch_runner as batch

    carrier = inspect.getsource(qwen4_exp_ar_batch)
    assert "batch.router_producer_variant = getattr(" in carrier

    owner = batch.Qwen4ExpGGUFBatchRunner
    assert "router_producer_variant" in owner.__annotations__
    source = inspect.getsource(owner)
    calls = [
        call for call in _layer_calls(source)
        if call.func.id.endswith("layer")
    ]
    assert len(calls) == 2
    for call in calls:
        value = _keyword(call, "router_producer_variant")
        assert getattr(value, "attr", None) == "router_producer_variant"

    # The captured-segment identity must change with the producer.
    def key(mode):
        owner = SimpleNamespace(
            q51_down_m1=False, q51_fold128_prefill=False,
            q51_fold_pair_prefill=False, q51_exact_grouped_down=True,
            q51_wmma_down=False, q51_out8=True, q4_exact_grouped_gate=True,
            q4_exact_grouped_all=True, q4_out4=True,
            forkb_grouped_down=False, q8_down_row4_prefill=False,
            q8_down_bundle_prefill=False, q5k_row4_prefill=False,
            router_producer_variant=mode,
        )
        return batch.Qwen4ExpGGUFBatchRunner._expert_down_policy(owner)

    assert key(None) != key(F16_GEMM)


@pytest.mark.parametrize(
    "function",
    ["run_qwen4_exp_batch_gdn_layer", "run_qwen4_exp_batch_qsa_layer", "_run_layer_ffn"],
)
def test_every_batch_forwarder_carries_the_producer(function) -> None:
    import inspect

    from hipengine.runtime import qwen4_exp_batch_runner as batch

    target = getattr(batch, function)
    parameters = inspect.signature(target).parameters
    assert "router_producer_variant" in parameters
    assert parameters["router_producer_variant"].default is None


def test_scope_with_none_binds_the_unbound_tile4_selection() -> None:
    """``None`` is a binding, not a restore: it selects the unbound tile4 parent.

    The dispatch maps a ``None`` owner value to the exact tile4 parent, so a
    ``None`` scope around a plan that publishes the F16-operand GEMM runs tile4
    for its duration and gets the published value back afterwards.
    """

    owner = SimpleNamespace(router_producer_variant=F16_GEMM)
    with runner.router_producer_variant(owner, None):
        assert owner.router_producer_variant is None
        assert runner._qwen4_exp_router_producer_variant(
            1024, owner.router_producer_variant,
        ) == TILE4
    assert owner.router_producer_variant == F16_GEMM
    assert runner._qwen4_exp_router_producer_variant(
        1024, owner.router_producer_variant,
    ) == F16_GEMM
