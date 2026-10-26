"""The Q5_1 fold and Q8_0 row4/bundle arms isolate the leaf they count.

Each package is one rung of a ladder: the fold-pair and bundle rungs replace
their parents on the same rows>=512 scope, and the published register/row-publish
siblings replace them in turn. An arm that leaves any of those in place counts
the other owner and reports it as its own, so the scope must pin the rung it does
not count off, require the parent it rides, bind the runner-local siblings off,
and restore every one of them on exit.
"""

from types import SimpleNamespace

import pytest

from hipengine.kernels.registry import KernelKey
from hipengine.runtime import qwen4_exp_runner as runner_module
from scripts.qwen4exp_halo_box_campaign_ab import (
    Q51_FOLD128_ENV,
    Q51_FOLD_PAIR_ENV,
    Q51_PAIR2_GATE_ENV,
    Q51_PAIR2_PREEMPTING_ENV,
    Q51_PAIR_PREFILL_ENV,
    Q8_DOWN_BUNDLE_ENV,
    Q8_DOWN_CHAIN_GATES,
    Q8_DOWN_FORKB_ENV,
    Q8_DOWN_PREEMPTING_ENV,
    Q8_DOWN_ROW4_ENV,
    QSA_ORDERED_DECODE_ENV,
    QSA_ORDERED_V2_FUSED_ENV,
    ROW4_ENGAGEMENT_EXEMPT,
    _apply_mode,
    q51_fold_control_record,
    q51_fold_scope,
    q8_down_control_record,
    q8_down_ladder_control_record,
    q8_down_ladder_scope,
    route_scope,
)

PAIR2 = "selected_grouped_prefill_pair2_bf16_bf16_out"
FOLD128 = "selected_grouped_prefill_pair2_fold128_bf16_bf16_out"
FOLD_PAIR = "selected_grouped_prefill_pair2_fold128_pair_bf16_bf16_out"
ROW4 = "selected_grouped_row4_gemv_bf16_bf16_out"
BUNDLE = "selected_grouped_row4_bundle_gemv_bf16_bf16_out"
REGISTER = "selected_grouped_row4_register_gemv_bf16_bf16_out"
GROUPED = "selected_grouped_gemv_bf16_bf16_out"


def _production_env():
    """The shipped lane's bindings for both ladders, as a mutable mapping."""

    env = {
        Q51_PAIR2_GATE_ENV: "1",
        Q51_PAIR_PREFILL_ENV: "1",
        Q51_FOLD128_ENV: "1",
        Q51_FOLD_PAIR_ENV: "1",
        Q8_DOWN_FORKB_ENV: "1",
        Q8_DOWN_ROW4_ENV: "1",
        Q8_DOWN_BUNDLE_ENV: "1",
    }
    env.update({key: "1" for key in Q51_PAIR2_PREEMPTING_ENV})
    env.update({key: "1" for key in Q8_DOWN_PREEMPTING_ENV})
    return env


def _published_runner():
    """A runner carrying the values the production binder publishes."""

    return SimpleNamespace(
        q51_register_cache=True, q51_row_publish=True, q8_down_register=True)


def _q51_leaf(owner):
    """The dispatch chain the grouped Q5_1 down runs, in source order."""

    key = KernelKey("hip_gfx1151", "moe_linear", "gguf_q5_1", PAIR2)
    key = runner_module._qwen4_exp_q51_fold128_key(key, rows=1024)
    key = runner_module._qwen4_exp_q51_fold_pair_key(key, rows=1024)
    key = runner_module._qwen4_exp_q51_register_cache_key(
        key, rows=1024, in_features=640,
        published=owner.q51_register_cache)
    key = runner_module._qwen4_exp_q51_row_publish_key(
        key, rows=1024, in_features=640, published=owner.q51_row_publish)
    return key.variant


def _q8_leaf(owner):
    """The dispatch chain the sorted grouped Q8_0 down runs, in source order."""

    key = KernelKey("hip_gfx1151", "linear", "gguf_q8_0", GROUPED)
    key = runner_module._qwen4_exp_q8_down_row4_key(key, rows=1024)
    key = runner_module._qwen4_exp_q8_down_bundle_key(key, rows=1024)
    key = runner_module._qwen4_exp_q8_down_register_key(
        key, rows=1024, in_features=640,
        published=owner.q8_down_register)
    return key.variant


@pytest.fixture
def registered(monkeypatch):
    monkeypatch.setattr(runner_module, "is_registered", lambda key: True)


@pytest.fixture
def shipped_lane(monkeypatch):
    """Publish the shipped bindings into the process env the selectors read."""

    for key, value in _production_env().items():
        monkeypatch.setenv(key, value)


# --- engagement: the arm's own leaf is the one the dispatch selects ----------


@pytest.mark.parametrize(
    "package,mode,expected",
    [
        ("q51-fold128", "before", PAIR2),
        ("q51-fold128", "after", FOLD128),
        ("q51-fold-pair", "before", FOLD128),
        ("q51-fold-pair", "after", FOLD_PAIR),
        ("q8-down-row4", "before", GROUPED),
        ("q8-down-row4", "after", ROW4),
        ("q8-down-bundle", "before", ROW4),
        ("q8-down-bundle", "after", BUNDLE),
    ],
)
def test_arm_selects_exactly_the_leaf_it_declares(
        registered, shipped_lane, monkeypatch, package, mode, expected):
    # ``_apply_mode`` is what the campaign calls before the measured run; the
    # scope is what has to make that binding reach the counted leaf.
    _apply_mode(mode, route_package=package)
    owner = _published_runner()
    leaf = _q51_leaf if package.startswith("q51-") else _q8_leaf
    with route_scope(package, mode, owner):
        assert leaf(owner) == expected


def test_row4_arm_counts_row4_not_bundle_or_register(registered, shipped_lane):
    # Production binds both downstream rungs on; without the closure the after
    # arm launches bundle (or register) and reports it as row4.
    _apply_mode("after", route_package="q8-down-row4")
    owner = _published_runner()
    with route_scope("q8-down-row4", "after", owner):
        assert _q8_leaf(owner) == ROW4


def test_fold128_arm_counts_fold128_not_fold_pair(registered, shipped_lane):
    _apply_mode("after", route_package="q51-fold128")
    owner = _published_runner()
    with route_scope("q51-fold128", "after", owner):
        assert _q51_leaf(owner) == FOLD128


# --- restoration: every temporary selection comes back ----------------------


@pytest.mark.parametrize("package", ["q51-fold128", "q51-fold-pair"])
def test_fold_arm_restores_env_and_runner_state(package):
    env = _production_env()
    owner = _published_runner()
    before = dict(env)
    with route_scope(package, "after", owner, environment=env):
        assert all(env[key] == "0" for key in Q51_PAIR2_PREEMPTING_ENV)
        assert env[Q51_PAIR2_GATE_ENV] == "1"
        assert env[Q51_PAIR_PREFILL_ENV] == "1"
        assert owner.q51_register_cache is False
        assert owner.q51_row_publish is False
    assert env == before
    assert owner.q51_register_cache is True
    assert owner.q51_row_publish is True


@pytest.mark.parametrize("package", ["q8-down-row4", "q8-down-bundle"])
def test_q8_ladder_arm_restores_env_and_runner_state(package):
    env = _production_env()
    owner = _published_runner()
    before = dict(env)
    with route_scope(package, "after", owner, environment=env):
        assert all(env[key] == "0" for key in Q8_DOWN_PREEMPTING_ENV)
        assert env[Q8_DOWN_FORKB_ENV] == "1"
        assert owner.q8_down_register is False
    assert env == before
    assert owner.q8_down_register is True


def test_absent_keys_come_back_absent():
    env = {Q51_PAIR2_GATE_ENV: "1", Q51_PAIR_PREFILL_ENV: "1"}
    env.update({key: "1" for key in Q51_PAIR2_PREEMPTING_ENV})
    env.pop(Q51_FOLD_PAIR_ENV, None)
    owner = _published_runner()
    with route_scope("q51-fold128", "after", owner, environment=env):
        assert env[Q51_FOLD_PAIR_ENV] == "0"
    assert Q51_FOLD_PAIR_ENV not in env

    env = {Q8_DOWN_FORKB_ENV: "1", Q8_DOWN_ROW4_ENV: "1"}
    env.update({key: "1" for key in Q8_DOWN_PREEMPTING_ENV})
    env.pop(Q8_DOWN_BUNDLE_ENV, None)
    with route_scope("q8-down-row4", "after", owner, environment=env):
        assert env[Q8_DOWN_BUNDLE_ENV] == "0"
    assert Q8_DOWN_BUNDLE_ENV not in env


def test_scope_restores_after_an_exception():
    env = _production_env()
    owner = _published_runner()
    with pytest.raises(RuntimeError, match="arm failed"):
        with route_scope("q8-down-row4", "after", owner, environment=env):
            raise RuntimeError("arm failed")
    assert env == _production_env()
    assert owner.q8_down_register is True


@pytest.mark.parametrize(
    "package,gate",
    [("q51-fold-pair", Q51_FOLD128_ENV),
     ("q8-down-bundle", Q8_DOWN_ROW4_ENV)])
def test_a_late_prerequisite_failure_unwinds_the_earlier_bindings(
        package, gate):
    # The preemptors close and the runner siblings bind before the rung check
    # runs; a missing parent must still leave the caller's state untouched.
    env = _production_env()
    env[gate] = "0"
    before = dict(env)
    owner = _published_runner()
    with pytest.raises(ValueError, match=gate):
        with route_scope(package, "after", owner, environment=env):
            pass
    assert env == before
    assert owner.q51_register_cache is True
    assert owner.q51_row_publish is True
    assert owner.q8_down_register is True


def test_building_the_scope_changes_nothing_until_entered():
    env = _production_env()
    owner = _published_runner()
    scope = route_scope("q51-fold128", "after", owner, environment=env)
    assert env[Q51_FOLD_PAIR_ENV] == "1"
    assert owner.q51_row_publish is True
    with scope:
        assert env[Q51_FOLD_PAIR_ENV] == "0"
    assert env[Q51_FOLD_PAIR_ENV] == "1"


# --- prerequisites fail loudly instead of measuring another owner -----------


def test_fold_pair_arm_requires_its_fold128_parent():
    for missing in (None, "", "0", "false", "False"):
        env = _production_env()
        if missing is None:
            env.pop(Q51_FOLD128_ENV)
        else:
            env[Q51_FOLD128_ENV] = missing
        before = dict(env)
        with pytest.raises(ValueError, match=Q51_FOLD128_ENV):
            with q51_fold_scope("q51-fold-pair", environment=env):
                pass
        assert env == before


def test_fold128_arm_does_not_require_its_own_switch():
    # The before arm is the parent control, so its own rung is legitimately off.
    env = _production_env()
    env[Q51_FOLD128_ENV] = "0"
    with q51_fold_scope("q51-fold128", environment=env):
        assert env[Q51_FOLD128_ENV] == "0"
        assert env[Q51_FOLD_PAIR_ENV] == "0"
    assert env[Q51_FOLD_PAIR_ENV] == "1"


@pytest.mark.parametrize("package", ["q51-fold128", "q51-fold-pair"])
@pytest.mark.parametrize("gate", [None, "", "0", "false", "False"])
def test_fold_arm_requires_the_pair2_selection_gate(package, gate):
    env = _production_env()
    if gate is None:
        env.pop(Q51_PAIR_PREFILL_ENV)
    else:
        env[Q51_PAIR_PREFILL_ENV] = gate
    with pytest.raises(ValueError, match=Q51_PAIR_PREFILL_ENV):
        with q51_fold_scope(package, environment=env):
            pass


@pytest.mark.parametrize("package", ["q51-fold128", "q51-fold-pair"])
def test_fold_arm_requires_the_pair2_chain_gate(package):
    env = _production_env()
    env[Q51_PAIR2_GATE_ENV] = "0"
    owner = _published_runner()
    with pytest.raises(ValueError, match=Q51_PAIR2_GATE_ENV):
        with route_scope(package, "after", owner, environment=env):
            pass
    assert owner.q51_row_publish is True


def test_bundle_arm_requires_its_row4_parent():
    for missing in (None, "", "0", "false", "False"):
        env = _production_env()
        if missing is None:
            env.pop(Q8_DOWN_ROW4_ENV)
        else:
            env[Q8_DOWN_ROW4_ENV] = missing
        before = dict(env)
        with pytest.raises(ValueError, match=Q8_DOWN_ROW4_ENV):
            with q8_down_ladder_scope("q8-down-bundle", environment=env):
                pass
        assert env == before


def test_row4_arm_does_not_require_bundle_or_its_own_switch():
    # The row4 arm pins bundle off; the generic chain-gate check would refuse it.
    env = _production_env()
    env[Q8_DOWN_ROW4_ENV] = "0"
    env[Q8_DOWN_BUNDLE_ENV] = "1"
    with q8_down_ladder_scope("q8-down-row4", environment=env):
        assert env[Q8_DOWN_BUNDLE_ENV] == "0"
        assert env[Q8_DOWN_ROW4_ENV] == "0"
    assert env[Q8_DOWN_BUNDLE_ENV] == "1"


@pytest.mark.parametrize("package", ["q8-down-row4", "q8-down-bundle"])
def test_q8_ladder_arm_requires_the_forkb_chain_gate(package):
    env = _production_env()
    env[Q8_DOWN_FORKB_ENV] = "0"
    owner = _published_runner()
    with pytest.raises(ValueError, match=Q8_DOWN_FORKB_ENV):
        with route_scope(package, "after", owner, environment=env):
            pass
    assert owner.q8_down_register is True


@pytest.mark.parametrize(
    "package", ["q51-fold128", "q51-fold-pair", "q8-down-row4", "q8-down-bundle"])
def test_arm_requires_the_owning_runner(package):
    # The downstream siblings are runner state; without the runner the arm
    # cannot hold them off and would measure whatever the plan published.
    with pytest.raises(ValueError, match="needs the owning runner"):
        route_scope(package, "after")


@pytest.mark.parametrize(
    "package", ["q51-fold128", "q51-fold-pair", "q8-down-row4", "q8-down-bundle"])
def test_arm_rejects_an_unknown_mode(package):
    with pytest.raises(ValueError, match="invalid campaign A/B mode"):
        route_scope(package, "during", _published_runner())


def test_ladder_scopes_reject_a_foreign_package():
    with pytest.raises(ValueError, match="not a Q5_1 fold route package"):
        with q51_fold_scope("q8-down-row4"):
            pass
    with pytest.raises(ValueError, match="not a sorted Q8_0 down ladder package"):
        with q8_down_ladder_scope("q51-fold128"):
            pass


# --- the artifact records what the arm forced and what it replaced ----------


def test_fold_control_record_separates_forced_and_previous_values():
    env = _production_env()
    record = q51_fold_control_record(
        "q51-fold128", _published_runner(), environment=env)
    assert record["preempting_owners_closed"] == {
        key: "0" for key in Q51_PAIR2_PREEMPTING_ENV}
    assert record["preempting_owners_previous"] == {
        key: "1" for key in Q51_PAIR2_PREEMPTING_ENV}
    assert record["rung"] == {
        "held_off": Q51_FOLD_PAIR_ENV,
        "forced": "0",
        "previous": "1",
    }
    assert record["prerequisite"][Q51_PAIR_PREFILL_ENV] == "1"
    assert record["pair2_gate"] == {Q51_PAIR2_GATE_ENV: "1"}
    assert record["downstream_siblings"]["q51_register_cache"] == {
        "bound_value": False, "published_value": True}
    assert record["downstream_siblings"]["q51_row_publish"] == {
        "bound_value": False, "published_value": True}
    # Recording changes nothing.
    assert env == _production_env()


def test_fold_pair_control_record_names_its_parent():
    env = _production_env()
    record = q51_fold_control_record(
        "q51-fold-pair", _published_runner(), environment=env)
    assert record["rung"] == {"prerequisite": Q51_FOLD128_ENV, "value": "1"}
    assert "held_off" not in record["rung"]


def test_q8_ladder_control_record_separates_forced_and_previous_values():
    env = _production_env()
    record = q8_down_ladder_control_record(
        "q8-down-row4", _published_runner(), environment=env)
    assert record["preempting_owners_closed"] == {
        key: "0" for key in Q8_DOWN_PREEMPTING_ENV}
    assert record["preempting_owners_previous"] == {
        key: "1" for key in Q8_DOWN_PREEMPTING_ENV}
    assert record["rung"] == {
        "held_off": Q8_DOWN_BUNDLE_ENV,
        "forced": "0",
        "previous": "1",
    }
    assert record["chain_gates"] == {key: "1" for key in Q8_DOWN_CHAIN_GATES}
    assert record["downstream_siblings"]["q8_down_register"] == {
        "bound_value": False, "published_value": True}
    assert env == _production_env()


def test_q8_bundle_control_record_names_its_parent():
    record = q8_down_ladder_control_record(
        "q8-down-bundle", _published_runner(), environment=_production_env())
    assert record["rung"] == {"prerequisite": Q8_DOWN_ROW4_ENV, "value": "1"}
    assert "held_off" not in record["rung"]


# --- the harnesses that own these arms --------------------------------------


@pytest.mark.parametrize(
    "package,flag",
    [
        ("q51-fold128", Q51_FOLD128_ENV),
        ("q51-fold-pair", Q51_FOLD_PAIR_ENV),
        ("q8-down-row4", Q8_DOWN_ROW4_ENV),
        ("q8-down-bundle", Q8_DOWN_BUNDLE_ENV),
    ],
)
def test_campaign_arm_writes_only_its_own_switch(package, flag):
    for mode, expected in (("before", "0"), ("after", "1")):
        env = {"other": "keep"}
        _apply_mode(mode, route_package=package, environment=env)
        assert env == {flag: expected, "other": "keep"}, (mode, env)


@pytest.mark.parametrize(
    "package", ["q51-fold128", "q51-fold-pair", "q8-down-row4", "q8-down-bundle"])
def test_campaign_validates_engagement_for_the_four(package):
    # Each arm counts one leaf of its own ladder, so before==0 / after>0 is a
    # real engagement assertion; exempting them is what let a neutral A/B pass.
    assert package not in ROW4_ENGAGEMENT_EXEMPT


@pytest.mark.parametrize(
    "package,flag",
    [
        ("q51-fold128", Q51_FOLD128_ENV),
        ("q51-fold-pair", Q51_FOLD_PAIR_ENV),
        ("q8-down-row4", Q8_DOWN_ROW4_ENV),
        ("q8-down-bundle", Q8_DOWN_BUNDLE_ENV),
    ],
)
def test_state_gate_leaves_the_isolation_to_the_shared_scope(package, flag):
    # The state gate binds the arm's own switch and enters the same scope; if it
    # also wrote the rung it does not count, the scope's restore would fight it.
    from scripts.qwen4exp_row4_state_gate import apply_state_gate_mode

    for enabled in ("0", "1"):
        env = {}
        apply_state_gate_mode(
            SimpleNamespace(), package, enabled, flag, environment=env)
        assert env == {flag: enabled}


# --- the state gate's arm lifecycle -----------------------------------------
# The gate keeps the arm's live control open across the measured prefill (and,
# for the decode-only ordered-v2 arm, across the decode steps). That block must
# release the control on every exit path: a failed run that left the preemptors
# closed, a rung pinned or a sibling bound off would silently change the next
# arm's measurement.


@pytest.mark.parametrize(
    "package", ["q51-fold128", "q51-fold-pair", "q8-down-row4", "q8-down-bundle"])
@pytest.mark.parametrize("enabled", ["0", "1"])
def test_a_failed_measured_run_releases_the_arm_control(package, enabled):
    from scripts.qwen4exp_row4_state_gate import arm_control

    env = _production_env()
    before = dict(env)
    owner = _published_runner()
    # ``ExitStack`` runs its callbacks only when it is closed, never when it is
    # collected, so the sentinel witnesses the close on the failure path
    # without depending on garbage collection.
    closed = []
    with pytest.raises(RuntimeError, match="prefill failed"):
        with arm_control(package, enabled, owner, environment=env) as scope:
            scope.callback(lambda: closed.append(True))
            raise RuntimeError("prefill failed")
    assert closed == [True], "the arm's stack must close on the failure path"
    assert env == before
    assert (owner.q51_register_cache, owner.q51_row_publish) == (True, True)
    assert owner.q8_down_register is True


def test_the_prefill_only_lifetime_still_releases_at_the_prefill():
    # The gate closes the control itself once the prefill returns; the decode
    # steps must not run under the arm's bindings.
    from scripts.qwen4exp_row4_state_gate import arm_control

    env = _production_env()
    owner = _published_runner()
    with arm_control("q8-down-row4", "1", owner, environment=env) as scope:
        assert env[Q8_DOWN_BUNDLE_ENV] == "0"
        assert owner.q8_down_register is False
        scope.close()
        assert env == _production_env()
        assert owner.q8_down_register is True


def test_the_decode_arm_keeps_its_binding_until_the_block_ends():
    # The ordered-v2 leaf is decode-only, so its binding has to survive the
    # prefill and the steps that count it: the gate closes this arm only after
    # those steps, which is the block's exit.
    from scripts.qwen4exp_row4_state_gate import arm_control

    env = {QSA_ORDERED_DECODE_ENV: "1", "other": "keep"}
    owner = SimpleNamespace(qsa_ordered_decode_v2=False)
    with arm_control("qsa-ordered-v2", "1", owner, environment=env):
        assert owner.qsa_ordered_decode_v2 is True
        assert env[QSA_ORDERED_V2_FUSED_ENV] == "0"
    assert owner.qsa_ordered_decode_v2 is False
    assert env == {QSA_ORDERED_DECODE_ENV: "1", "other": "keep"}


def test_a_failed_decode_releases_the_decode_arm():
    from scripts.qwen4exp_row4_state_gate import arm_control

    env = {QSA_ORDERED_DECODE_ENV: "1"}
    owner = SimpleNamespace(qsa_ordered_decode_v2=False)
    closed = []
    with pytest.raises(RuntimeError, match="decode failed"):
        with arm_control("qsa-ordered-v2", "1", owner, environment=env) as scope:
            scope.callback(lambda: closed.append(True))
            raise RuntimeError("decode failed")
    assert closed == [True]
    assert owner.qsa_ordered_decode_v2 is False
    assert env == {QSA_ORDERED_DECODE_ENV: "1"}


def test_the_gate_enters_the_arm_control_under_a_with_block():
    # The lifecycle is only exception-safe while the stack is entered under
    # ``with``; a bare enter followed by the measured work leaks on a raise.
    import ast
    import inspect

    from scripts import qwen4exp_row4_state_gate

    main = ast.parse(inspect.getsource(qwen4exp_row4_state_gate.main))
    enters = [
        node for node in ast.walk(main)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id == "arm_control"
    ]
    assert enters, "the gate no longer enters the arm control"
    guarded = set()
    for node in ast.walk(main):
        if isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                if (isinstance(item.context_expr, ast.Call)
                        and isinstance(item.context_expr.func, ast.Name)
                        and item.context_expr.func.id == "arm_control"):
                    guarded.add(id(item.context_expr))
    assert all(id(call) in guarded for call in enters), (
        "the arm control must be entered under ``with`` so a failed measured "
        "run cannot leak its bindings")


def test_the_prefill_only_close_stays_conditional():
    # The intended lifetime is unchanged: the prefill arms release the control
    # themselves, the ordered-v2 arm does not. The exit close backs both up.
    import ast
    import inspect

    from scripts import qwen4exp_row4_state_gate

    main = ast.parse(inspect.getsource(qwen4exp_row4_state_gate.main))

    def closes_in(node):
        return [
            call for call in ast.walk(node)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute) and call.func.attr == "close"
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "case_scope"
        ]

    assert len(closes_in(main)) == 2, "expected the early and the exit close"
    conditional = [
        node for node in ast.walk(main)
        if isinstance(node, ast.If)
        and any(isinstance(child, ast.Constant) and child.value == "qsa-ordered-v2"
                for child in ast.walk(node.test))
        and closes_in(node)
    ]
    assert conditional, (
        "the early close must stay conditional on the package, so the "
        "decode-only arm keeps its binding through the counted steps")
