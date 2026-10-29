"""The Q5_1 fold and Q8_0 row4/bundle arms isolate the leaf they count.

Each package is one rung of a ladder: the fold-pair and bundle rungs replace
their parents on the same rows>=512 scope, and the published register/row-publish
siblings replace them in turn. Every rung is request-local runner state now, so
an arm binds the rung it counts, holds the rung it does not count off, requires
the parent it rides, binds the runner-local siblings off, and restores every one
of them on exit.
"""

from types import SimpleNamespace

import pytest

from hipengine.kernels.registry import KernelKey
from hipengine.runtime import qwen4_exp_runner as runner_module
from scripts.qwen4exp_halo_box_campaign_ab import (
    Q51_FOLD128_FIELD,
    Q51_FOLD_PAIR_FIELD,
    Q51_PAIR2_GATE_FIELD,
    Q51_PAIR2_PREEMPTING_ENV,
    Q51_PAIR_FIELD,
    Q8_DOWN_BUNDLE_FIELD,
    Q8_DOWN_FORKB_FIELD,
    Q8_DOWN_PREEMPTING_ENV,
    Q8_DOWN_ROW4_FIELD,
    QSA_ORDERED_DECODE_ENV,
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
ROW_PUBLISH = "selected_grouped_prefill_pair2_row_publish_bf16_bf16_out"
GROUPED = "selected_grouped_gemv_bf16_bf16_out"

def _production_env():
    """The shipped lane's process bindings around both ladders."""

    env = {"HIPENGINE_QWEN4_EXP_Q51_PAIR_PREFILL": "0"}
    env.update({key: "1" for key in Q51_PAIR2_PREEMPTING_ENV})
    env.update({key: "1" for key in Q8_DOWN_PREEMPTING_ENV})
    return env


def _published_runner():
    """A runner carrying the values the production binder publishes."""

    return SimpleNamespace(
        q51_exact_grouped_down=True, q51_pair_prefill=True,
        q51_fold128_prefill=True, q51_fold_pair_prefill=True,
        q51_register_cache=True, q51_row_publish=True,
        forkb_grouped_down=True,
        q8_down_row4_prefill=True, q8_down_bundle_prefill=True,
        q8_down_register=True)


def _q51_leaf(owner):
    """The dispatch chain the grouped Q5_1 down runs, in source order."""

    key = KernelKey("hip_gfx1151", "moe_linear", "gguf_q5_1", PAIR2)
    key = runner_module._qwen4_exp_q51_fold128_key(
        key, rows=1024, published=owner.q51_fold128_prefill)
    key = runner_module._qwen4_exp_q51_fold_pair_key(
        key, rows=1024, published=owner.q51_fold_pair_prefill)
    key = runner_module._qwen4_exp_q51_register_cache_key(
        key, rows=1024, in_features=640,
        published=owner.q51_register_cache)
    key = runner_module._qwen4_exp_q51_row_publish_key(
        key, rows=1024, in_features=640, published=owner.q51_row_publish)
    return key.variant


def _q8_leaf(owner):
    """The dispatch chain the sorted grouped Q8_0 down runs, in source order."""

    key = KernelKey("hip_gfx1151", "linear", "gguf_q8_0", GROUPED)
    key = runner_module._qwen4_exp_q8_down_row4_key(
        key, rows=1024, published=owner.q8_down_row4_prefill)
    key = runner_module._qwen4_exp_q8_down_bundle_key(
        key, rows=1024, published=owner.q8_down_bundle_prefill)
    key = runner_module._qwen4_exp_q8_down_register_key(
        key, rows=1024, in_features=640,
        published=owner.q8_down_register)
    return key.variant


@pytest.fixture
def registered(monkeypatch):
    monkeypatch.setattr(runner_module, "is_registered", lambda key: True)


@pytest.fixture
def shipped_lane(monkeypatch):
    """Publish the process keys the two ladders still depend on.

    The rungs are runner state now; what remains a process key is each chain's
    own selection gate.
    """

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
        registered, shipped_lane, package, mode, expected):
    owner = _published_runner()
    leaf = _q51_leaf if package.startswith("q51-") else _q8_leaf
    with route_scope(package, mode, owner):
        assert leaf(owner) == expected
    # The plan's own values come back on exit: with every rung and sibling
    # published, each chain lands on its published leaf again.
    assert leaf(owner) == (ROW_PUBLISH if package.startswith("q51-")
                           else REGISTER)


def test_row4_arm_counts_row4_not_bundle_or_register(registered, shipped_lane):
    # Production publishes both downstream rungs on; without the closure the
    # after arm launches bundle (or register) and reports it as row4.
    owner = _published_runner()
    with route_scope("q8-down-row4", "after", owner):
        assert _q8_leaf(owner) == ROW4


def test_fold128_arm_counts_fold128_not_fold_pair(registered, shipped_lane):
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
        assert owner.q51_exact_grouped_down is True
        assert owner.q51_pair_prefill is True
        assert owner.q51_register_cache is False
        assert owner.q51_row_publish is False
    assert env == before
    assert owner.q51_fold128_prefill is True
    assert owner.q51_fold_pair_prefill is True
    assert owner.q51_register_cache is True
    assert owner.q51_row_publish is True


@pytest.mark.parametrize("package", ["q8-down-row4", "q8-down-bundle"])
def test_q8_ladder_arm_restores_env_and_runner_state(package):
    env = _production_env()
    owner = _published_runner()
    before = dict(env)
    with route_scope(package, "after", owner, environment=env):
        assert all(env[key] == "0" for key in Q8_DOWN_PREEMPTING_ENV)
        assert owner.forkb_grouped_down is True
        assert owner.q8_down_register is False
    assert env == before
    assert owner.q8_down_row4_prefill is True
    assert owner.q8_down_bundle_prefill is True
    assert owner.q8_down_register is True


def test_a_runner_without_the_field_comes_back_unpublished():
    # An unbound runner (no rung attribute yet) keeps the parent after the
    # scope. The pair2 chain gate is not one of the rungs: the arm requires it
    # published, because a closed chain would make the counted leaf unreachable.
    owner = SimpleNamespace(q51_exact_grouped_down=True, q51_pair_prefill=True)
    env = _production_env()
    with route_scope("q51-fold128", "after", owner, environment=env):
        assert owner.q51_fold128_prefill is True
    assert owner.q51_fold128_prefill is False
    assert owner.q51_fold_pair_prefill is False


def test_scope_restores_after_an_exception():
    env = _production_env()
    owner = _published_runner()
    with pytest.raises(RuntimeError, match="arm failed"):
        with route_scope("q8-down-row4", "after", owner, environment=env):
            raise RuntimeError("arm failed")
    assert env == _production_env()
    assert owner.q8_down_row4_prefill is True
    assert owner.q8_down_bundle_prefill is True
    assert owner.q8_down_register is True


@pytest.mark.parametrize(
    "package,field",
    [("q51-fold-pair", Q51_FOLD128_FIELD),
     ("q8-down-bundle", Q8_DOWN_ROW4_FIELD)])
def test_a_late_prerequisite_failure_unwinds_the_earlier_bindings(
        package, field):
    # The preemptors close and the runner siblings bind before the rung check
    # runs; a missing parent must still leave the caller's state untouched.
    env = _production_env()
    before = dict(env)
    owner = _published_runner()
    setattr(owner, field, False)
    with pytest.raises(ValueError, match=field):
        with route_scope(package, "after", owner, environment=env):
            pass
    assert env == before
    assert owner.q51_register_cache is True
    assert owner.q51_row_publish is True
    assert owner.q8_down_register is True
    assert getattr(owner, field) is False


def test_building_the_scope_changes_nothing_until_entered():
    env = _production_env()
    owner = _published_runner()
    scope = route_scope("q51-fold128", "after", owner, environment=env)
    assert owner.q51_fold_pair_prefill is True
    assert owner.q51_row_publish is True
    with scope:
        assert owner.q51_fold_pair_prefill is False
    assert owner.q51_fold_pair_prefill is True


# --- prerequisites fail loudly instead of measuring another owner -----------


def test_fold_pair_arm_requires_its_published_fold128_parent():
    for missing in (False, None):
        owner = _published_runner()
        if missing is None:
            del owner.q51_fold128_prefill
        else:
            owner.q51_fold128_prefill = missing
        with pytest.raises(ValueError, match=Q51_FOLD128_FIELD):
            with q51_fold_scope("q51-fold-pair", "after", owner,
                                environment=_production_env()):
                pass


def test_fold128_arm_does_not_require_its_own_rung():
    # The before arm is the parent control, so its own rung is legitimately off.
    owner = _published_runner()
    owner.q51_fold128_prefill = False
    with q51_fold_scope("q51-fold128", "after", owner,
                        environment=_production_env()):
        assert owner.q51_fold128_prefill is True
        assert owner.q51_fold_pair_prefill is False
    assert owner.q51_fold128_prefill is False
    assert owner.q51_fold_pair_prefill is True


@pytest.mark.parametrize("package", ["q51-fold128", "q51-fold-pair"])
@pytest.mark.parametrize("gate", [None, False])
def test_fold_arm_requires_the_pair2_selection_gate(package, gate):
    env = _production_env()
    owner = _published_runner()
    if gate is None:
        del owner.q51_pair_prefill
    else:
        owner.q51_pair_prefill = gate
    with pytest.raises(ValueError, match=Q51_PAIR_FIELD):
        with q51_fold_scope(package, "after", owner,
                            environment=env):
            pass


@pytest.mark.parametrize("package", ["q51-fold128", "q51-fold-pair"])
def test_fold_arm_requires_the_pair2_chain_gate(package):
    env = _production_env()
    owner = _published_runner()
    owner.q51_exact_grouped_down = False
    with pytest.raises(ValueError, match=Q51_PAIR2_GATE_FIELD):
        with route_scope(package, "after", owner, environment=env):
            pass
    assert owner.q51_row_publish is True


def test_bundle_arm_requires_its_published_row4_parent():
    for missing in (False, None):
        owner = _published_runner()
        if missing is None:
            del owner.q8_down_row4_prefill
        else:
            owner.q8_down_row4_prefill = missing
        with pytest.raises(ValueError, match=Q8_DOWN_ROW4_FIELD):
            with q8_down_ladder_scope("q8-down-bundle", "after", owner,
                                      environment=_production_env()):
                pass


def test_row4_arm_does_not_require_bundle_or_its_own_rung():
    # The row4 arm holds bundle off; the generic chain-gate check would refuse it.
    owner = _published_runner()
    owner.q8_down_row4_prefill = False
    with q8_down_ladder_scope("q8-down-row4", "after", owner,
                              environment=_production_env()):
        assert owner.q8_down_bundle_prefill is False
        assert owner.q8_down_row4_prefill is True
    assert owner.q8_down_bundle_prefill is True
    assert owner.q8_down_row4_prefill is False


@pytest.mark.parametrize("package", ["q8-down-row4", "q8-down-bundle"])
def test_q8_ladder_arm_requires_the_forkb_chain_gate(package):
    env = _production_env()
    owner = _published_runner()
    owner.forkb_grouped_down = False
    with pytest.raises(ValueError, match=Q8_DOWN_FORKB_FIELD):
        with route_scope(package, "after", owner, environment=env):
            pass
    assert owner.q8_down_register is True


@pytest.mark.parametrize(
    "package", ["q51-fold128", "q51-fold-pair", "q8-down-row4", "q8-down-bundle"])
def test_arm_requires_the_owning_runner(package):
    # The rungs are runner state; without the runner the arm cannot bind them
    # and would measure whatever the plan published.
    with pytest.raises(ValueError, match="needs the owning runner"):
        route_scope(package, "after")


@pytest.mark.parametrize(
    "package", ["q51-fold128", "q51-fold-pair", "q8-down-row4", "q8-down-bundle"])
def test_arm_rejects_an_unknown_mode(package):
    with pytest.raises(ValueError, match="invalid campaign A/B mode"):
        route_scope(package, "during", _published_runner())


def test_ladder_scopes_reject_a_foreign_package():
    owner = _published_runner()
    env = _production_env()
    with pytest.raises(ValueError, match="not a Q5_1 fold route package"):
        with q51_fold_scope("q8-down-row4", "after", owner, environment=env):
            pass
    with pytest.raises(ValueError, match="not a sorted Q8_0 down ladder package"):
        with q8_down_ladder_scope("q51-fold128", "after", owner,
                                  environment=env):
            pass


# --- the register arms require the rungs below them -------------------------


@pytest.mark.parametrize(
    "package,fields",
    [
        ("q51-register-cache", (Q51_FOLD128_FIELD, Q51_FOLD_PAIR_FIELD)),
        ("q51-row-publish", (Q51_FOLD128_FIELD, Q51_FOLD_PAIR_FIELD)),
        ("q8-down-register", (Q8_DOWN_ROW4_FIELD, Q8_DOWN_BUNDLE_FIELD)),
    ],
)
@pytest.mark.parametrize("missing", [0, 1])
def test_a_downstream_arm_requires_its_published_ladder(
        registered, package, fields, missing):
    env = _production_env()
    before = dict(env)
    owner = _published_runner()
    setattr(owner, fields[missing], False)
    with pytest.raises(ValueError, match=fields[missing]):
        with route_scope(package, "after", owner, environment=env):
            pass
    assert env == before
    assert owner.q51_register_cache is True
    assert owner.q51_row_publish is True
    assert owner.q8_down_register is True


# --- the artifact records what the arm bound and what it replaced -----------


def test_fold_control_record_separates_bound_and_published_values():
    env = _production_env()
    record = q51_fold_control_record(
        "q51-fold128", _published_runner(), environment=env)
    assert record["preempting_owners_closed"] == {
        key: "0" for key in Q51_PAIR2_PREEMPTING_ENV}
    assert record["preempting_owners_previous"] == {
        key: "1" for key in Q51_PAIR2_PREEMPTING_ENV}
    assert record["runner_field"] == Q51_FOLD128_FIELD
    assert record["rung"] == {
        "held_off": Q51_FOLD_PAIR_FIELD,
        "bound_value": False,
        "published_value": True,
    }
    assert record["prerequisite"][Q51_PAIR_FIELD] is True
    assert record["pair2_gate"] == {
        "runner_field": Q51_PAIR2_GATE_FIELD,
        "published_value": True,
    }
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
    assert record["runner_field"] == Q51_FOLD_PAIR_FIELD
    assert record["rung"] == {
        "prerequisite": Q51_FOLD128_FIELD, "published_value": True}
    assert "held_off" not in record["rung"]


def test_q8_ladder_control_record_separates_bound_and_published_values():
    env = _production_env()
    record = q8_down_ladder_control_record(
        "q8-down-row4", _published_runner(), environment=env)
    assert record["preempting_owners_closed"] == {
        key: "0" for key in Q8_DOWN_PREEMPTING_ENV}
    assert record["preempting_owners_previous"] == {
        key: "1" for key in Q8_DOWN_PREEMPTING_ENV}
    assert record["runner_field"] == Q8_DOWN_ROW4_FIELD
    assert record["rung"] == {
        "held_off": Q8_DOWN_BUNDLE_FIELD,
        "bound_value": False,
        "published_value": True,
    }
    assert record["chain_gate"] == {
        "runner_field": Q8_DOWN_FORKB_FIELD, "published_value": True}
    assert record["downstream_siblings"]["q8_down_register"] == {
        "bound_value": False, "published_value": True}
    assert env == _production_env()


def test_q8_bundle_control_record_names_its_parent():
    record = q8_down_ladder_control_record(
        "q8-down-bundle", _published_runner(), environment=_production_env())
    assert record["runner_field"] == Q8_DOWN_BUNDLE_FIELD
    assert record["rung"] == {
        "prerequisite": Q8_DOWN_ROW4_FIELD, "published_value": True}
    assert "held_off" not in record["rung"]


# --- the harnesses that own these arms --------------------------------------


@pytest.mark.parametrize(
    "package", ["q51-fold128", "q51-fold-pair", "q8-down-row4", "q8-down-bundle"])
@pytest.mark.parametrize("mode", ["before", "after"])
def test_campaign_arm_writes_no_process_key(package, mode):
    env = {"other": "keep"}
    _apply_mode(mode, route_package=package, environment=env)
    assert env == {"other": "keep"}, (mode, env)


@pytest.mark.parametrize(
    "package", ["q51-fold128", "q51-fold-pair", "q8-down-row4", "q8-down-bundle"])
def test_campaign_validates_engagement_for_the_four(package):
    # Each arm counts one leaf of its own ladder, so before==0 / after>0 is a
    # real engagement assertion; exempting them is what let a neutral A/B pass.
    assert package not in ROW4_ENGAGEMENT_EXEMPT


@pytest.mark.parametrize(
    "package", ["q51-fold128", "q51-fold-pair", "q8-down-row4", "q8-down-bundle"])
def test_state_gate_leaves_the_isolation_to_the_shared_scope(package):
    # The gate binds nothing itself: ``route_scope`` owns the rung binding and
    # the restore, so a gate that also wrote a key would fight it.
    from scripts.qwen4exp_row4_state_gate import apply_state_gate_mode

    for enabled in ("0", "1"):
        env = {}
        apply_state_gate_mode(
            SimpleNamespace(), package, enabled, "", environment=env)
        assert env == {}


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
    assert (owner.q51_fold128_prefill, owner.q51_fold_pair_prefill) == (
        True, True)
    assert (owner.q8_down_row4_prefill, owner.q8_down_bundle_prefill) == (
        True, True)


def test_the_prefill_only_lifetime_still_releases_at_the_prefill():
    # The gate closes the control itself once the prefill returns; the decode
    # steps must not run under the arm's bindings.
    from scripts.qwen4exp_row4_state_gate import arm_control

    env = _production_env()
    owner = _published_runner()
    with arm_control("q8-down-row4", "1", owner, environment=env) as scope:
        assert owner.q8_down_bundle_prefill is False
        assert owner.q8_down_register is False
        scope.close()
        assert env == _production_env()
        assert owner.q8_down_row4_prefill is True
        assert owner.q8_down_bundle_prefill is True
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
        assert env == {QSA_ORDERED_DECODE_ENV: "1", "other": "keep"}
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
