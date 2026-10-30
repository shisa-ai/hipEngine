"""QSA prefill head-mode harness controls.

The head leaf is request-local runner state for the scalar sparse-prefill rows
owner, so an A/B arm selects it through a scoped control rather than a
process-wide environment variable. These tests pin that the campaign/state-gate
helpers write no environment and that the scope restores whatever the profile
published.
"""

from types import SimpleNamespace

import pytest


def _runner(mode: str = "quad") -> SimpleNamespace:
    return SimpleNamespace(qsa_prefill_head_mode=mode)


@pytest.mark.parametrize("package", ["qsa-head-pair", "qsa-head-quad"])
@pytest.mark.parametrize("mode", ["before", "after"])
def test_apply_mode_leaves_the_environment_alone(package, mode):
    from scripts.qwen4exp_halo_box_campaign_ab import _apply_mode

    env = {"other": "keep"}
    _apply_mode(mode, route_package=package, environment=env)
    assert env == {"other": "keep"}


@pytest.mark.parametrize(
    "package,mode,expected",
    [
        ("qsa-head-pair", "after", "pair"),
        ("qsa-head-pair", "before", "parent"),
        ("qsa-head-quad", "after", "quad"),
        ("qsa-head-quad", "before", "parent"),
    ],
)
def test_route_scope_selects_and_restores_the_runner_mode(package, mode, expected):
    from scripts.qwen4exp_halo_box_campaign_ab import route_scope

    owned = _runner("quad")
    with route_scope(package, mode, owned):
        assert owned.qsa_prefill_head_mode == expected
    assert owned.qsa_prefill_head_mode == "quad"


def test_route_scope_leaves_non_head_packages_untouched():
    from scripts.qwen4exp_halo_box_campaign_ab import route_scope

    owned = _runner("quad")
    with route_scope("qsa-h256-page256", "after", owned):
        assert owned.qsa_prefill_head_mode == "quad"
    with pytest.raises(ValueError, match="needs the owning runner"):
        route_scope("qsa-h256-page256", "after", None)
    assert owned.qsa_prefill_head_mode == "quad"


@pytest.mark.parametrize("package", ["qsa-head-pair", "qsa-head-quad"])
def test_route_scope_rejects_a_head_arm_without_its_runner(package):
    """Binding nothing would measure the parent and report it as the head arm."""

    from scripts.qwen4exp_halo_box_campaign_ab import route_scope

    with pytest.raises(ValueError, match="needs the owning runner"):
        route_scope(package, "after", None)
    with pytest.raises(ValueError, match="needs the owning runner"):
        route_scope(package, "before", None)


def test_route_scope_rejects_an_invalid_mode():
    from scripts.qwen4exp_halo_box_campaign_ab import route_scope

    with pytest.raises(ValueError):
        with route_scope("qsa-head-quad", "sideways", _runner("quad")):
            pass


@pytest.mark.parametrize("package", ["qsa-head-pair", "qsa-head-quad"])
@pytest.mark.parametrize("enabled", ["0", "1"])
def test_state_gate_leaves_the_head_arm_to_the_scoped_control(package, enabled):
    """The gate binds no env key and no runner field of its own."""

    from scripts.qwen4exp_row4_state_gate import apply_state_gate_mode

    owned = _runner("quad")
    env: dict[str, str] = {}
    apply_state_gate_mode(owned, package, enabled, "unused", environment=env)
    assert env == {}
    assert owned.qsa_prefill_head_mode == "quad"


def test_state_gate_hands_the_runner_to_the_scoped_control():
    """A gate that forgot the runner would silently measure the parent.

    The gate enters the arm's control in ``main`` and the control builds the
    scoped selection, so both hops have to carry the arm's runner.
    """

    import ast
    import inspect

    from scripts import qwen4exp_row4_state_gate

    calls = [
        node
        for node in ast.walk(ast.parse(inspect.getsource(qwen4exp_row4_state_gate)))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "route_scope"
    ]
    assert calls, "the gate no longer uses the scoped control"
    for call in calls:
        assert len(call.args) == 3, "route_scope needs the arm's runner"
        runner_arg = call.args[2]
        assert isinstance(runner_arg, ast.Name) and runner_arg.id == "runner"

    entered = [
        node
        for node in ast.walk(ast.parse(inspect.getsource(qwen4exp_row4_state_gate.main)))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "arm_control"
    ]
    assert entered, "the gate no longer enters the arm's control"
    for call in entered:
        assert len(call.args) == 3, "arm_control needs the arm's runner"
        runner_arg = call.args[2]
        assert isinstance(runner_arg, ast.Attribute) and runner_arg.attr == "runner"


def test_r2e_ablation_binds_the_runner_mode():
    """The route ablation keeps a runner-scoped arm instead of an env clear."""

    from scripts.qwen4exp_r2e_route_ablation import ROUTES, RUNNER_ROUTE_MODES

    assert RUNNER_ROUTE_MODES == {"qsa_h256_quad": "parent"}
    assert "qsa_h256_quad" not in ROUTES
    for envs, _ in ROUTES.values():
        assert "HIPENGINE_QWEN4_EXP_QSA_HEAD_PAIR" not in envs


def test_restore_harness_keeps_the_arm_mode_out_of_the_environment():
    from scripts.qwen4exp_layer2_profile_gate import CANDIDATES
    from scripts.qwen4exp_prefill_restore_ab import SHARED_DECODE_SAFE_FLAGS

    assert "HIPENGINE_QWEN4_EXP_QSA_HEAD_PAIR" not in SHARED_DECODE_SAFE_FLAGS
    candidate = CANDIDATES["q8_blockscale_guarded_quad"]
    assert candidate.qsa_prefill_head_mode == "quad"
    assert "HIPENGINE_QWEN4_EXP_QSA_HEAD_PAIR" not in candidate.environment


def test_layer2_gate_binds_a_declared_arm_mode_on_the_candidate_runner():
    """A declared mode that the gate never binds would measure the base profile."""

    import ast
    import inspect

    from scripts import qwen4exp_layer2_profile_gate

    tree = ast.parse(inspect.getsource(qwen4exp_layer2_profile_gate.run))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "qsa_prefill_head_mode"
    ]
    assert calls, "the gate no longer binds the arm's head mode"
    assert any(
        isinstance(argument, ast.Attribute) and argument.attr == "runner"
        for call in calls for argument in call.args
    )
    # It must be entered on the stack that unwinds with the measured rows.
    assert any(
        isinstance(node, ast.Attribute) and node.attr == "enter_context"
        for node in ast.walk(tree)
    )


@pytest.mark.parametrize("previous", [False, True])
def test_probe_head_arm_disables_and_restores_wmma(previous):
    """A head arm must not measure the WMMA kernel it never launches."""
    from scripts.qwen4exp_qsa_decode_flag_probe import (
        restore_scalar_head_measurement, scalar_head_measurement)

    runner = SimpleNamespace(qsa_wmma_enabled=previous)
    record = scalar_head_measurement(True, runner=runner)
    assert runner.qsa_wmma_enabled is False
    assert record["previous"] is previous
    assert record["arms"] == "both" and record["measured"] is True
    restore_scalar_head_measurement(record, runner=runner)
    assert runner.qsa_wmma_enabled is previous
    assert scalar_head_measurement(False, runner=runner) is None
    assert runner.qsa_wmma_enabled is previous


@pytest.mark.parametrize("previous", [False, True])
def test_halo_head_route_disables_and_restores_wmma(previous):
    from scripts.qwen4exp_halo_box_campaign_ab import (
        head_route_configuration, restore_route_configuration)

    for package in ("qsa-head-pair", "qsa-head-quad"):
        runner = SimpleNamespace(qsa_wmma_enabled=previous)
        record = head_route_configuration(package, runner=runner)
        assert runner.qsa_wmma_enabled is False
        assert record["previous"] is previous and record["arms"] == "both"
        restore_route_configuration(record, runner=runner)
        assert runner.qsa_wmma_enabled is previous

    assert head_route_configuration("qsa-h256-page256", runner=runner) is None
    assert runner.qsa_wmma_enabled is previous


def test_r2e_counts_launches_instead_of_reading_the_field():
    from scripts.qwen4exp_r2e_route_ablation import (
        QSA_HEAD_VARIANTS, QSA_SCALAR_PARENT_VARIANT, count_scalar_launches,
        restore_scalar_launches)

    launched: list[str] = []

    def fake_resolve(**kwargs):
        return lambda *a, **kw: launched.append(kwargs["variant"])

    module = SimpleNamespace(resolve=fake_resolve)
    counts = {"head": 0, "parent": 0}
    original = count_scalar_launches(counts, module=module)

    module.resolve(backend="b", layer="qsa_sparse_attention", quant="bf16_kv",
                   variant=QSA_SCALAR_PARENT_VARIANT)()
    for variant in QSA_HEAD_VARIANTS:
        module.resolve(backend="b", layer="qsa_sparse_attention", quant="bf16_kv",
                       variant=variant)()
    # Other layers are not wrapped, and resolving without calling is not a launch.
    module.resolve(backend="b", layer="linear", quant="q", variant="v")()
    module.resolve(backend="b", layer="qsa_sparse_attention", quant="bf16_kv",
                   variant="strict_h256_head_quad_rows_spans")

    assert counts == {"head": 2, "parent": 1}
    assert launched.count(QSA_SCALAR_PARENT_VARIANT) == 1
    restore_scalar_launches(original, module=module)
    assert module.resolve is fake_resolve


def test_r2e_head_route_reports_launches_and_restores_wmma():
    import ast
    import inspect

    from scripts import qwen4exp_r2e_route_ablation as ablation

    source = inspect.getsource(ablation.main)
    tree = ast.parse(source)
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert "count_scalar_launches" in names
    assert "head_route_configuration" in names
    assert "head_engagement" in {
        node.value for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)}
    # The field check stays a configuration assert, not the engagement claim.
    assert "engagement lost: qsa_prefill_head_mode" not in source
    assert "configuration lost: qsa_prefill_head_mode" in source
    # Both the override and the counter are restored on any failure below.
    assert "restore_route_configuration" in names
    assert "restore_scalar_launches" in names


def _restored_from_finally(source, apply_name, restore_name):
    """Whether ``restore_name`` runs in a finally covering ``apply_name``."""

    import ast

    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Try) or not node.finalbody:
            continue
        final = {
            child.id for block in node.finalbody
            for child in ast.walk(block) if isinstance(child, ast.Name)}
        body = {
            child.id for block in node.body
            for child in ast.walk(block) if isinstance(child, ast.Name)}
        if restore_name in final and apply_name in body:
            return True
    return False


def test_halo_restores_the_wmma_override_in_its_finally():
    import inspect

    from scripts import qwen4exp_halo_box_campaign_ab as halo

    assert _restored_from_finally(
        inspect.getsource(halo.main),
        "head_route_configuration", "restore_route_configuration")


def test_probe_restores_the_wmma_override_in_its_finally():
    import inspect

    from scripts import qwen4exp_qsa_decode_flag_probe as probe

    assert _restored_from_finally(
        inspect.getsource(probe.main),
        "scalar_head_measurement", "restore_scalar_head_measurement")


def test_probe_and_halo_record_the_wmma_override():
    import ast
    import inspect

    from scripts import qwen4exp_halo_box_campaign_ab, qwen4exp_qsa_decode_flag_probe

    for module, attribute in (
        (qwen4exp_qsa_decode_flag_probe, "qsa_wmma"),
        (qwen4exp_halo_box_campaign_ab, "qsa_wmma"),
    ):
        constants = {
            node.value for node in ast.walk(ast.parse(inspect.getsource(module.main)))
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        assert attribute in constants, module.__name__


def test_state_gate_h256_binding_does_not_change_head_grouping():
    from scripts.qwen4exp_row4_state_gate import apply_state_gate_mode

    owned = _runner("quad")
    env: dict[str, str] = {}
    apply_state_gate_mode(owned, "qsa-h256-page256", "1", "WAVE", environment=env)
    assert env == {}
    assert not hasattr(owned, "qsa_h256_prefill_mode")
    from scripts.qwen4exp_halo_box_campaign_ab import route_scope
    with route_scope("qsa-h256-page256", "after", owned):
        assert owned.qsa_h256_prefill_mode == "page256"
        assert owned.qsa_prefill_head_mode == "quad"
