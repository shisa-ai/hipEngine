"""Q5_1 register-cache selection: plan-published runner state, not a switch."""

from types import SimpleNamespace

import pytest

from hipengine.kernels import registry
from hipengine.kernels.registry import KernelKey
from hipengine.runtime import qwen4_exp_runner as runner
from scripts.qwen4exp_halo_box_campaign_ab import (
    Q51_PAIR2_GATE_ENV,
    Q51_PAIR2_PREEMPTING_ENV,
    _apply_mode,
    q51_register_cache_expected_calls,
    q51_register_cache_control_record,
    route_scope,
)
from scripts.qwen4exp_row4_state_gate import apply_state_gate_mode
from tests.test_unit_qwen4exp_q51_row_publish_route import (
    _gr_read,
    _layer_scratch,
    _layer_weights,
)

#: The retired process switch. Nothing reads it any more; the tests keep the
#: literal so a leftover value can be proven inert.
RETIRED_REGISTER_CACHE_ENV = "HIPENGINE_QWEN4_EXP_Q51_REGISTER_CACHE"

FOLD_PAIR = "selected_grouped_prefill_pair2_fold128_pair_bf16_bf16_out"
REGISTER_CACHE = "selected_grouped_prefill_pair2_register_cache_bf16_bf16_out"
ROW_PUBLISH = "selected_grouped_prefill_pair2_row_publish_bf16_bf16_out"


def _key(variant: str = FOLD_PAIR) -> KernelKey:
    return KernelKey("hip_gfx1151", "moe_linear", "gguf_q5_1", variant)


def _env(**overrides):
    env = {key: "1" for key in Q51_PAIR2_PREEMPTING_ENV}
    env[Q51_PAIR2_GATE_ENV] = "1"
    env.update(overrides)
    return env


# --- the selector ------------------------------------------------------------


def test_route_selects_register_cache_only_when_the_plan_published_it(monkeypatch):
    monkeypatch.setattr(runner, "is_registered", lambda key: True)
    select = runner._qwen4_exp_q51_register_cache_key
    assert select(_key(), rows=1024, in_features=640, published=False) == _key()
    assert select(
        _key(), rows=1024, in_features=640,
        published=True).variant == REGISTER_CACHE


@pytest.mark.parametrize("rows,selected", [
    (1, False), (511, False), (512, True), (1024, True), (4096, True),
])
def test_route_keeps_the_rows_ge_512_guard(monkeypatch, rows, selected):
    monkeypatch.setattr(runner, "is_registered", lambda key: True)
    key = runner._qwen4_exp_q51_register_cache_key(
        _key(), rows=rows, in_features=640, published=True)
    assert (key.variant == REGISTER_CACHE) is selected


def test_route_keeps_the_k640_and_folded_pair_parent_guards(monkeypatch):
    monkeypatch.setattr(runner, "is_registered", lambda key: True)
    select = runner._qwen4_exp_q51_register_cache_key
    assert select(_key(), rows=1024, in_features=256, published=True) == _key()
    # The leaf is the child of the folded-pair parent, never of the fold128 or
    # the plain pair2 owner.
    for parent in ("selected_grouped_prefill_pair2_bf16_bf16_out",
                   "selected_grouped_prefill_pair2_fold128_bf16_bf16_out"):
        other = _key(parent)
        assert select(other, rows=1024, in_features=640,
                      published=True) == other


def test_route_falls_back_when_the_leaf_is_not_registered(monkeypatch):
    monkeypatch.setattr(runner, "is_registered", lambda key: False)
    assert runner._qwen4_exp_q51_register_cache_key(
        _key(), rows=1024, in_features=640, published=True) == _key()


def test_route_ignores_the_retired_process_switch(monkeypatch):
    # The decision is the plan's, published on the owning runner. A leftover env
    # value may not open the route on an arm that published the parent, and the
    # published value may not depend on the environment.
    monkeypatch.setattr(runner, "is_registered", lambda key: True)
    monkeypatch.setenv(RETIRED_REGISTER_CACHE_ENV, "1")
    assert runner._qwen4_exp_q51_register_cache_key(
        _key(), rows=1024, in_features=640, published=False) == _key()
    monkeypatch.setenv(RETIRED_REGISTER_CACHE_ENV, "0")
    assert runner._qwen4_exp_q51_register_cache_key(
        _key(), rows=1024, in_features=640,
        published=True).variant == REGISTER_CACHE


def test_two_runners_keep_independent_decisions():
    published = SimpleNamespace(q51_register_cache=True)
    parent = SimpleNamespace(q51_register_cache=False)
    assert runner._qwen4_exp_q51_register_cache_key(
        _key(), rows=1024, in_features=640,
        published=published.q51_register_cache).variant == REGISTER_CACHE
    assert runner._qwen4_exp_q51_register_cache_key(
        _key(), rows=1024, in_features=640,
        published=parent.q51_register_cache) == _key()


# --- the scoped diagnostic control ------------------------------------------


def test_binding_a_scope_changes_nothing_until_entered():
    owner = SimpleNamespace(q51_register_cache=False)
    scope = runner.q51_register_cache(owner, True)
    assert owner.q51_register_cache is False
    with scope:
        assert owner.q51_register_cache is True
    assert owner.q51_register_cache is False


def test_scope_binds_one_runner_and_restores_the_published_value():
    first = SimpleNamespace(q51_register_cache=False)
    second = SimpleNamespace(q51_register_cache=False)
    with runner.q51_register_cache(first, True):
        assert first.q51_register_cache is True
        assert second.q51_register_cache is False
    assert first.q51_register_cache is False


def test_scope_binds_the_parent_for_a_scalar_teacher_arm():
    # The explicit parent control the diagnostic keeps: an arm can force the
    # scalar owner inside its own window without touching the plan's value.
    owner = SimpleNamespace(q51_register_cache=True)
    with runner.q51_register_cache(owner, False):
        assert owner.q51_register_cache is False
        assert runner._qwen4_exp_q51_register_cache_key(
            _key(), rows=1024, in_features=640,
            published=owner.q51_register_cache) == _key()
    assert owner.q51_register_cache is True


def test_scope_restores_the_published_value_after_an_exception():
    owner = SimpleNamespace(q51_register_cache=True)
    with pytest.raises(RuntimeError, match="arm failed"):
        with runner.q51_register_cache(owner, False):
            raise RuntimeError("arm failed")
    assert owner.q51_register_cache is True


def test_unset_scope_keeps_the_published_value():
    owner = SimpleNamespace(q51_register_cache=True)
    with runner.q51_register_cache(owner):
        assert owner.q51_register_cache is True
    assert owner.q51_register_cache is True


def test_cloned_runner_inherits_the_published_decision():
    from hipengine.generation.qwen4_exp_gguf import _inherit_lane_decisions

    clone = SimpleNamespace()
    _inherit_lane_decisions(SimpleNamespace(q51_register_cache=True), clone)
    assert clone.q51_register_cache is True
    # An unbound carrier publishes the parent, never the other lane's value.
    _inherit_lane_decisions(SimpleNamespace(), clone)
    assert clone.q51_register_cache is False


# --- the harness arm --------------------------------------------------------


def test_campaign_arm_writes_no_process_switch():
    for mode in ("before", "after"):
        env = {"other": "keep"}
        _apply_mode(mode, route_package="q51-register-cache", environment=env)
        assert env == {"other": "keep"}, (mode, env)


def test_state_gate_arm_writes_no_process_switch():
    env = {"other": "keep"}
    apply_state_gate_mode(
        None, "q51-register-cache", "1", "", environment=env)
    assert env == {"other": "keep"}


@pytest.mark.parametrize("mode,expected", [("before", False), ("after", True)])
def test_route_scope_binds_the_runner_selection(mode, expected):
    owner = SimpleNamespace(q51_register_cache=None, q51_row_publish=True)
    env = _env()
    with route_scope("q51-register-cache", mode, owner, environment=env):
        assert owner.q51_register_cache is expected
        # Every preempting owner is closed so the counted pair2 leaf can run.
        assert all(env[key] == "0" for key in Q51_PAIR2_PREEMPTING_ENV)
        # The pair2 gate is a prerequisite, never forced.
        assert env[Q51_PAIR2_GATE_ENV] == "1"
    assert owner.q51_register_cache is None
    assert all(env[key] == "1" for key in Q51_PAIR2_PREEMPTING_ENV)


@pytest.mark.parametrize("mode", ["before", "after"])
def test_route_scope_holds_the_downstream_row_publish_off_in_both_arms(mode):
    # The row-publish sibling replaces the register-cache leaf the arm counts,
    # so an arm that leaves the plan's row-publish decision in force measures
    # the other kernel.
    owner = SimpleNamespace(q51_register_cache=False, q51_row_publish=True)
    env = _env()
    with route_scope("q51-register-cache", mode, owner, environment=env):
        assert owner.q51_row_publish is False
        assert runner._qwen4_exp_q51_row_publish_key(
            _key(REGISTER_CACHE), rows=1024, in_features=640,
            published=owner.q51_row_publish).variant == REGISTER_CACHE
    assert owner.q51_row_publish is True
    assert runner._qwen4_exp_q51_row_publish_key(
        _key(REGISTER_CACHE), rows=1024, in_features=640,
        published=owner.q51_row_publish).variant == ROW_PUBLISH


def test_route_scope_restores_every_decision_after_an_exception():
    owner = SimpleNamespace(q51_register_cache=True, q51_row_publish=True)
    env = _env()
    with pytest.raises(RuntimeError, match="measured run failed"):
        with route_scope("q51-register-cache", "before", owner, environment=env):
            raise RuntimeError("measured run failed")
    assert owner.q51_register_cache is True
    assert owner.q51_row_publish is True
    assert all(env[key] == "1" for key in Q51_PAIR2_PREEMPTING_ENV)


def test_route_scope_rejects_the_arm_without_its_runner():
    with pytest.raises(ValueError, match="needs the owning runner"):
        route_scope("q51-register-cache", "after")


def test_route_scope_fails_loudly_without_the_chain_gate():
    owner = SimpleNamespace(q51_register_cache=False, q51_row_publish=True)
    env = {key: "1" for key in Q51_PAIR2_PREEMPTING_ENV}
    with pytest.raises(ValueError, match=Q51_PAIR2_GATE_ENV):
        with route_scope("q51-register-cache", "after", owner, environment=env):
            pass
    # Neither the child nor the downstream pin is bound by a refused arm.
    assert owner.q51_register_cache is False
    assert owner.q51_row_publish is True
    assert all(env[key] == "1" for key in Q51_PAIR2_PREEMPTING_ENV)


def test_building_a_q51_register_cache_scope_binds_nothing():
    owner = SimpleNamespace(q51_register_cache=True, q51_row_publish=True)
    env = _env()
    scope = route_scope("q51-register-cache", "after", owner, environment=env)
    assert all(env[key] == "1" for key in Q51_PAIR2_PREEMPTING_ENV)
    assert owner.q51_register_cache is True
    assert owner.q51_row_publish is True
    with scope:
        assert all(env[key] == "0" for key in Q51_PAIR2_PREEMPTING_ENV)
        assert owner.q51_register_cache is True
        assert owner.q51_row_publish is False
    assert all(env[key] == "1" for key in Q51_PAIR2_PREEMPTING_ENV)
    assert owner.q51_row_publish is True


def test_control_record_names_the_field_the_preemptors_and_the_pin():
    env = _env()
    owner = SimpleNamespace(q51_register_cache=True, q51_row_publish=True)
    record = q51_register_cache_control_record(owner, environment=env)
    assert record["runner_field"] == "q51_register_cache"
    assert record["before"] is False and record["after"] is True
    assert record["published_value"] is True
    assert record["preempting_owners_closed"] == {
        key: "0" for key in Q51_PAIR2_PREEMPTING_ENV}
    assert record["preempting_owners_previous"] == {
        key: "1" for key in Q51_PAIR2_PREEMPTING_ENV}
    assert record["pair2_gate"] == {Q51_PAIR2_GATE_ENV: "1"}
    pin = record["downstream_row_publish"]
    assert pin["runner_field"] == "q51_row_publish"
    assert pin["bound_value"] is False
    assert pin["published_value"] is True
    assert "register-cache" in pin["reason"]
    # Recording changes nothing.
    assert env[Q51_PAIR2_GATE_ENV] == "1"
    assert owner.q51_register_cache is True


def test_route_scope_leaves_the_preemptors_alone_for_other_packages():
    env = _env()
    owner = SimpleNamespace(gdn_prefill_variant=None, q51_row_publish=True)
    with route_scope("gdn-wave-norm", "after", owner, environment=env):
        assert all(env[key] == "1" for key in Q51_PAIR2_PREEMPTING_ENV)
        assert owner.q51_row_publish is True
    assert all(env[key] == "1" for key in Q51_PAIR2_PREEMPTING_ENV)


@pytest.mark.parametrize("tokens,expected", [(512, 43), (1024, 86), (4096, 344), (511, 0)])
def test_current_chunk_counts(monkeypatch, tokens, expected):
    # The arm's engagement assertion uses this leaf's own count at the gate's
    # 512-row chunk: the loaded Q5_1 down layers times the eligible chunks.
    from tests._qwen4exp_expert_down_fixtures import expert_down_runner

    monkeypatch.setattr(registry, "is_registered", lambda key: True)
    assert q51_register_cache_expected_calls(
        expert_down_runner(), tokens, 512) == expected


# --- forwarding: the runner field must reach run_qwen4_exp_moe ---------------
# The selector tests above prove the decision; these prove the decision is
# actually carried there from the owning runner, which a selector-only test
# cannot (a layer that drops the keyword still passes them).


def _moe_seen(seen):
    def moe(*args, **kwargs):
        seen.append(kwargs.get("q51_register_cache", "absent"))
        return SimpleNamespace(output=SimpleNamespace(ptr=14))
    return moe


def _stub_common(monkeypatch, seen):
    monkeypatch.setattr(runner, "run_qwen4_exp_gr_read", _gr_read)
    monkeypatch.setattr(runner, "run_qwen4_exp_moe", _moe_seen(seen))
    monkeypatch.setattr(runner, "_qwen4_exp_gr_pending_write", lambda *a, **k: None)
    monkeypatch.setattr(runner, "qwen4_exp_gr_write_bf16_f32", lambda *a, **k: None)
    monkeypatch.setattr(runner, "qwen4_exp_gr_write_bf16_bf16", lambda *a, **k: None)


def test_gdn_layer_forwards_the_selection_to_run_moe(monkeypatch):
    seen = []
    runtime = object()
    _stub_common(monkeypatch, seen)
    monkeypatch.setattr(
        runner, "run_qwen4_exp_gdn_token_mixer",
        lambda *a, **k: SimpleNamespace(ptr=13))
    monkeypatch.setattr(runner, "_qwen4_exp_gr_defer_write", lambda *a, **k: False)
    weights = _layer_weights()
    weights.mixer = SimpleNamespace(projections={}, conv_weight_ptr=5,
                                    dt_bias_ptr=6, a_ptr=7, norm_weight_ptr=8)
    kwargs = dict(conv_state_ptr=0, recurrent_state_ptr=0,
                  scratch=_layer_scratch(runtime), rows=512, branches=1,
                  hidden=2, low_rank=2, num_k_heads=16, num_v_heads=48,
                  head_dim=128, conv_kernel=4, ffn=8, experts=1, top_k=1,
                  runtime=runtime)
    runner.run_qwen4_exp_gdn_layer(100, weights, **kwargs, q51_register_cache=True)
    runner.run_qwen4_exp_gdn_layer(100, weights, **kwargs)
    assert seen == [True, False]


def test_qsa_prefill_layer_forwards_the_selection_to_run_moe(monkeypatch):
    seen = []
    runtime = object()
    _stub_common(monkeypatch, seen)
    monkeypatch.setattr(
        runner, "run_qwen4_exp_qsa_prefill_token_mixer",
        lambda *a, **k: SimpleNamespace(ptr=13))
    kwargs = dict(attention_state=object(), index_state=object(),
                  scratch=_layer_scratch(runtime), metadata=object(),
                  start_position=0, rows=512, branches=1, hidden=2, low_rank=2,
                  query_heads=24, kv_heads=2, head_dim=128, rotary_dim=64,
                  theta=1e6, index_heads=1, index_dim=1, index_rotary_dim=64,
                  ffn=8, experts=1, top_k=1, runtime=runtime)
    runner.run_qwen4_exp_qsa_prefill_layer(
        100, _layer_weights(), **kwargs, q51_register_cache=True)
    runner.run_qwen4_exp_qsa_prefill_layer(100, _layer_weights(), **kwargs)
    assert seen == [True, False]


def test_dense_qsa_layer_forwards_the_selection_to_run_moe(monkeypatch):
    seen = []
    runtime = object()
    _stub_common(monkeypatch, seen)
    monkeypatch.setattr(
        runner, "run_qwen4_exp_dense_qsa_token_mixer",
        lambda *a, **k: SimpleNamespace(ptr=13))
    kwargs = dict(attention_state=object(), scratch=_layer_scratch(runtime),
                  position=0, rows=512, branches=1, hidden=2, low_rank=2,
                  query_heads=24, kv_heads=2, head_dim=128, rotary_dim=64,
                  theta=1e6, ffn=8, experts=1, top_k=1, runtime=runtime)
    runner.run_qwen4_exp_dense_qsa_layer(
        100, _layer_weights(), **kwargs, q51_register_cache=True)
    runner.run_qwen4_exp_dense_qsa_layer(100, _layer_weights(), **kwargs)
    assert seen == [True, False]


def test_moe_graph_capture_forwards_the_selection(monkeypatch):
    # The captured decode path recurses through ``eager``; rows=1 never selects
    # the leaf, but the closure still has to carry the owner's decision rather
    # than silently publishing the parent inside a graph.
    seen = []
    runtime = object()

    class Cache:
        def run(self, key, *, eager, out_ptr, out_nbytes, stream):
            eager(3)

    monkeypatch.setattr(runner, "run_qwen4_exp_moe", _moe_seen(seen))
    runner.run_qwen4_exp_moe(
        1, {}, scratch=_layer_scratch(runtime), rows=1, hidden=2, ffn=4,
        experts=1, top_k=1, graph_cache=Cache(), graph_key="k",
        q51_register_cache=True)
    assert seen == [True]
