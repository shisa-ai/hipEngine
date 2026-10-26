"""Q5_1 row-publish selection: plan-published runner state, not a process switch."""

from types import SimpleNamespace

import pytest

from hipengine.kernels.registry import KernelKey
from hipengine.runtime import qwen4_exp_runner as runner
from scripts.qwen4exp_halo_box_campaign_ab import (
    Q51_IU8_EXACT_ENV,
    Q51_PAIR2_GATE_ENV,
    Q51_PAIR2_PREEMPTING_ENV,
    _apply_mode,
    q51_fold_pair_expected_calls,
    q51_pair2_control_record,
    q51_pair2_scope,
    route_scope,
)

#: The retired process switch. Nothing reads it any more; the tests keep the
#: literal so a leftover value can be proven inert.
RETIRED_ROW_PUBLISH_ENV = "HIPENGINE_QWEN4_EXP_Q51_ROW_PUBLISH"

PARENT = "selected_grouped_prefill_pair2_register_cache_bf16_bf16_out"
ROW_PUBLISH = "selected_grouped_prefill_pair2_row_publish_bf16_bf16_out"


def _key(variant: str = PARENT) -> KernelKey:
    return KernelKey("hip_gfx1151", "moe_linear", "gguf_q5_1", variant)


def test_route_selects_row_publish_only_when_the_plan_published_it(monkeypatch):
    monkeypatch.setattr(runner, "is_registered", lambda key: True)
    select = runner._qwen4_exp_q51_row_publish_key
    assert select(_key(), rows=1024, in_features=640, published=False) == _key()
    assert select(
        _key(), rows=1024, in_features=640, published=True).variant == ROW_PUBLISH


def test_route_keeps_the_rows_k640_and_parent_guards(monkeypatch):
    monkeypatch.setattr(runner, "is_registered", lambda key: True)
    select = runner._qwen4_exp_q51_row_publish_key
    assert select(_key(), rows=511, in_features=640, published=True) == _key()
    assert select(_key(), rows=1024, in_features=256, published=True) == _key()
    other = _key("selected_grouped_prefill_pair2_fold128_pair_bf16_bf16_out")
    assert select(other, rows=1024, in_features=640, published=True) == other


def test_route_falls_back_when_the_leaf_is_not_registered(monkeypatch):
    monkeypatch.setattr(runner, "is_registered", lambda key: False)
    assert runner._qwen4_exp_q51_row_publish_key(
        _key(), rows=1024, in_features=640, published=True) == _key()


def test_route_ignores_the_retired_process_switch(monkeypatch):
    # The decision is the plan's, published on the owning runner. A leftover env
    # value may not re-open the route on an arm that published the parent, and
    # the published value may not depend on the environment.
    monkeypatch.setattr(runner, "is_registered", lambda key: True)
    monkeypatch.setenv(RETIRED_ROW_PUBLISH_ENV, "1")
    assert runner._qwen4_exp_q51_row_publish_key(
        _key(), rows=1024, in_features=640, published=False) == _key()
    monkeypatch.setenv(RETIRED_ROW_PUBLISH_ENV, "0")
    assert runner._qwen4_exp_q51_row_publish_key(
        _key(), rows=1024, in_features=640, published=True).variant == ROW_PUBLISH


def test_two_runners_keep_independent_decisions():
    # Request-local state, so one lane's arm cannot move another lane's route.
    published = SimpleNamespace(q51_row_publish=True)
    parent = SimpleNamespace(q51_row_publish=False)
    assert runner._qwen4_exp_q51_row_publish_key(
        _key(), rows=1024, in_features=640,
        published=published.q51_row_publish).variant == ROW_PUBLISH
    assert runner._qwen4_exp_q51_row_publish_key(
        _key(), rows=1024, in_features=640,
        published=parent.q51_row_publish) == _key()


def test_binding_a_scope_changes_nothing_until_entered():
    owner = SimpleNamespace(q51_row_publish=False)
    scope = runner.q51_row_publish(owner, True)
    assert owner.q51_row_publish is False
    with scope:
        assert owner.q51_row_publish is True
    assert owner.q51_row_publish is False


def test_scope_binds_one_runner_and_restores_the_published_value():
    first = SimpleNamespace(q51_row_publish=False)
    second = SimpleNamespace(q51_row_publish=False)
    with runner.q51_row_publish(first, True):
        assert first.q51_row_publish is True
        assert second.q51_row_publish is False
    assert first.q51_row_publish is False


def test_scope_restores_the_published_value_after_an_exception():
    owner = SimpleNamespace(q51_row_publish=True)
    with pytest.raises(RuntimeError, match="arm failed"):
        with runner.q51_row_publish(owner, False):
            raise RuntimeError("arm failed")
    assert owner.q51_row_publish is True


def test_unset_scope_keeps_the_published_value():
    owner = SimpleNamespace(q51_row_publish=True)
    with runner.q51_row_publish(owner):
        assert owner.q51_row_publish is True
    assert owner.q51_row_publish is True


def test_cloned_runner_inherits_the_published_decision():
    from hipengine.generation.qwen4_exp_gguf import _inherit_lane_decisions

    clone = SimpleNamespace()
    _inherit_lane_decisions(SimpleNamespace(q51_row_publish=True), clone)
    assert clone.q51_row_publish is True
    # An unbound carrier publishes the parent, never the other lane's value.
    _inherit_lane_decisions(SimpleNamespace(), clone)
    assert clone.q51_row_publish is False


def test_campaign_arm_writes_no_process_switch():
    for mode in ("before", "after"):
        env = {"other": "keep"}
        _apply_mode(mode, route_package="q51-row-publish", environment=env)
        assert env == {"other": "keep"}, (mode, env)


@pytest.mark.parametrize("mode,expected", [("before", False), ("after", True)])
def test_route_scope_binds_the_runner_selection(mode, expected):
    owner = SimpleNamespace(q51_row_publish=None)
    env = {
        RETIRED_ROW_PUBLISH_ENV: "1",
        Q51_PAIR2_GATE_ENV: "1",
        **{key: "1" for key in Q51_PAIR2_PREEMPTING_ENV},
    }
    with route_scope("q51-row-publish", mode, owner, environment=env):
        assert owner.q51_row_publish is expected
        # Every preempting owner is closed so the counted pair2 leaf can run.
        assert all(env[key] == "0" for key in Q51_PAIR2_PREEMPTING_ENV)
        # The pair2 gate and the retired switch are untouched.
        assert env[Q51_PAIR2_GATE_ENV] == "1"
        assert env[RETIRED_ROW_PUBLISH_ENV] == "1"
    assert owner.q51_row_publish is None
    assert env[RETIRED_ROW_PUBLISH_ENV] == "1"
    assert env[Q51_PAIR2_GATE_ENV] == "1"
    assert all(env[key] == "1" for key in Q51_PAIR2_PREEMPTING_ENV)


def test_route_scope_rejects_the_arm_without_its_runner():
    with pytest.raises(ValueError, match="needs the owning runner"):
        route_scope("q51-row-publish", "after")


@pytest.mark.parametrize("tokens,expected", [(512, 25), (1024, 25), (4096, 100), (511, 0)])
def test_current_chunk_counts(tokens, expected):
    assert q51_fold_pair_expected_calls(tokens, 1024) == expected


# --- forwarding: the runner field must reach run_qwen4_exp_moe ---------------
# The selector tests above prove the decision; these prove the decision is
# actually carried there from the owning runner, which a selector-only test
# cannot (a layer that drops the keyword still passes them).


def _layer_scratch(runtime):
    return SimpleNamespace(
        closed=False, runtime=runtime,
        attention_gr=SimpleNamespace(), qsa=SimpleNamespace(),
        gdn=SimpleNamespace(),
        after_attention=SimpleNamespace(ptr=16), ffn_gr=SimpleNamespace(),
        moe=SimpleNamespace(), output=SimpleNamespace(ptr=15))


def _layer_weights():
    return SimpleNamespace(
        attention_gr=SimpleNamespace(norm_weight_ptr=1, down=2, up=3, inject=4),
        mixer=SimpleNamespace(),
        ffn_gr=SimpleNamespace(norm_weight_ptr=9, down=10, up=11, inject=12),
        moe=SimpleNamespace())


def _gr_read(*args, **kwargs):
    return SimpleNamespace(normalized=None, gate=None,
                           mixed=SimpleNamespace(ptr=11),
                           inject_logits=SimpleNamespace(ptr=12))


def _moe_seen(seen):
    def moe(*args, **kwargs):
        seen.append(kwargs.get("q51_row_publish", "absent"))
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
    runner.run_qwen4_exp_gdn_layer(100, weights, **kwargs, q51_row_publish=True)
    runner.run_qwen4_exp_gdn_layer(100, weights, **kwargs)
    assert seen == [True, False]


def test_qsa_prefill_layer_forwards_the_selection_to_run_moe(monkeypatch):
    seen = []
    runtime = object()
    _stub_common(monkeypatch, seen)
    monkeypatch.setattr(
        runner, "run_qwen4_exp_qsa_prefill_token_mixer",
        lambda *a, **k: SimpleNamespace(ptr=13))
    runner.run_qwen4_exp_qsa_prefill_layer(
        100, _layer_weights(), attention_state=object(), index_state=object(),
        scratch=_layer_scratch(runtime), metadata=object(), start_position=0,
        rows=512, branches=1, hidden=2, low_rank=2, query_heads=24,
        kv_heads=2, head_dim=128, rotary_dim=64, theta=1e6,
        index_heads=1, index_dim=1, index_rotary_dim=64, ffn=8,
        experts=1, top_k=1, runtime=runtime, q51_row_publish=True)
    runner.run_qwen4_exp_qsa_prefill_layer(
        100, _layer_weights(), attention_state=object(), index_state=object(),
        scratch=_layer_scratch(runtime), metadata=object(), start_position=0,
        rows=512, branches=1, hidden=2, low_rank=2, query_heads=24,
        kv_heads=2, head_dim=128, rotary_dim=64, theta=1e6,
        index_heads=1, index_dim=1, index_rotary_dim=64, ffn=8,
        experts=1, top_k=1, runtime=runtime)
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
        100, _layer_weights(), **kwargs, q51_row_publish=True)
    runner.run_qwen4_exp_dense_qsa_layer(100, _layer_weights(), **kwargs)
    assert seen == [True, False]


def test_moe_graph_capture_forwards_the_selection(monkeypatch):
    # The captured decode path recurses through ``eager``; rows=1 never selects
    # the row-publish leaf, but the closure still has to carry the owner's
    # decision rather than silently publishing the parent inside a graph.
    seen = []
    runtime = object()

    class Cache:
        def run(self, key, *, eager, out_ptr, out_nbytes, stream):
            eager(3)

    monkeypatch.setattr(runner, "run_qwen4_exp_moe", _moe_seen(seen))
    runner.run_qwen4_exp_moe(
        1, {}, scratch=_layer_scratch(runtime), rows=1, hidden=2, ffn=4,
        experts=1, top_k=1, graph_cache=Cache(), graph_key="k",
        q51_row_publish=True)
    assert seen == [True]


def test_pair2_scope_restores_absent_and_present_values():
    env = {Q51_PAIR2_PREEMPTING_ENV[0]: "1", Q51_PAIR2_GATE_ENV: "1"}
    with q51_pair2_scope(environment=env):
        assert all(env[key] == "0" for key in Q51_PAIR2_PREEMPTING_ENV)
    assert env == {Q51_PAIR2_PREEMPTING_ENV[0]: "1", Q51_PAIR2_GATE_ENV: "1"}


def test_pair2_scope_restores_after_an_exception():
    env = {key: "1" for key in Q51_PAIR2_PREEMPTING_ENV}
    env[Q51_PAIR2_GATE_ENV] = "1"
    with pytest.raises(RuntimeError, match="arm failed"):
        with q51_pair2_scope(environment=env):
            raise RuntimeError("arm failed")
    assert all(env[key] == "1" for key in Q51_PAIR2_PREEMPTING_ENV)


def test_pair2_scope_changes_nothing_until_entered():
    env = {key: "1" for key in Q51_PAIR2_PREEMPTING_ENV}
    env[Q51_PAIR2_GATE_ENV] = "1"
    scope = q51_pair2_scope(environment=env)
    assert all(env[key] == "1" for key in Q51_PAIR2_PREEMPTING_ENV)
    with scope:
        assert all(env[key] == "0" for key in Q51_PAIR2_PREEMPTING_ENV)
    assert all(env[key] == "1" for key in Q51_PAIR2_PREEMPTING_ENV)


def test_pair2_control_names_every_preemptor():
    # The routed F16 down is the production owner; the rest sit behind it. The
    # pair2 gate is deliberately not in the forced list.
    assert Q51_PAIR2_PREEMPTING_ENV == (
        "HIPENGINE_QWEN4_EXP_ROUTED_F16",
        "HIPENGINE_QWEN4_EXP_Q51_IU8_EXACT",
        "HIPENGINE_QWEN4_EXP_Q5_1_WMMA",
        "HIPENGINE_QWEN4_EXP_PRODUCTION_MOE_PREFILL",
        "HIPENGINE_QWEN4_EXP_Q5_1_MMQ_PREFILL",
        "HIPENGINE_QWEN4_EXP_T16_DOWN",
    )
    assert Q51_PAIR2_GATE_ENV not in Q51_PAIR2_PREEMPTING_ENV
    assert Q51_IU8_EXACT_ENV == "HIPENGINE_QWEN4_EXP_Q51_IU8_EXACT"


def test_route_scope_leaves_the_preemptors_alone_for_other_packages():
    env = {key: "1" for key in Q51_PAIR2_PREEMPTING_ENV}
    owner = SimpleNamespace(gdn_prefill_variant=None)
    with route_scope("gdn-wave-norm", "after", owner, environment=env):
        assert all(env[key] == "1" for key in Q51_PAIR2_PREEMPTING_ENV)
    assert all(env[key] == "1" for key in Q51_PAIR2_PREEMPTING_ENV)


def test_building_a_q51_route_scope_binds_nothing():
    owner = SimpleNamespace(q51_row_publish=True)
    env = {key: "1" for key in Q51_PAIR2_PREEMPTING_ENV}
    env[Q51_PAIR2_GATE_ENV] = "1"
    scope = route_scope("q51-row-publish", "after", owner, environment=env)
    assert all(env[key] == "1" for key in Q51_PAIR2_PREEMPTING_ENV)
    assert owner.q51_row_publish is True
    with scope:
        assert all(env[key] == "0" for key in Q51_PAIR2_PREEMPTING_ENV)
        assert owner.q51_row_publish is True
    assert all(env[key] == "1" for key in Q51_PAIR2_PREEMPTING_ENV)
    assert owner.q51_row_publish is True


@pytest.mark.parametrize("gate", [None, "", "0", "false", "False"])
def test_pair2_scope_fails_loudly_without_the_chain_gate(gate):
    # Without the pair2 chain's own gate the chain is skipped entirely, so an
    # arm would report an engagement it cannot produce.
    env = {key: "1" for key in Q51_PAIR2_PREEMPTING_ENV}
    if gate is not None:
        env[Q51_PAIR2_GATE_ENV] = gate
    with pytest.raises(ValueError, match=Q51_PAIR2_GATE_ENV):
        with q51_pair2_scope(environment=env):
            pass
    assert all(env[key] == "1" for key in Q51_PAIR2_PREEMPTING_ENV)


@pytest.mark.parametrize("mode", ["before", "after"])
def test_route_scope_fails_loudly_without_the_chain_gate(mode):
    owner = SimpleNamespace(q51_row_publish=None)
    env = {key: "1" for key in Q51_PAIR2_PREEMPTING_ENV}
    with pytest.raises(ValueError, match=Q51_PAIR2_GATE_ENV):
        with route_scope("q51-row-publish", mode, owner, environment=env):
            pass
    assert owner.q51_row_publish is None
    assert all(env[key] == "1" for key in Q51_PAIR2_PREEMPTING_ENV)


def test_control_record_separates_the_forced_value_from_the_previous_one():
    # A reader must not see the production binding under a key that claims the
    # arm closed it.
    env = {
        "HIPENGINE_QWEN4_EXP_ROUTED_F16": "1",
        "HIPENGINE_QWEN4_EXP_Q51_IU8_EXACT": "1",
        Q51_PAIR2_GATE_ENV: "1",
    }
    record = q51_pair2_control_record(environment=env)
    assert record["preempting_owners_closed"] == {
        key: "0" for key in Q51_PAIR2_PREEMPTING_ENV}
    assert record["preempting_owners_previous"][
        "HIPENGINE_QWEN4_EXP_ROUTED_F16"] == "1"
    assert record["preempting_owners_previous"][
        "HIPENGINE_QWEN4_EXP_Q5_1_WMMA"] is None
    assert record["pair2_gate"] == {Q51_PAIR2_GATE_ENV: "1"}
    # Recording changes nothing.
    assert env["HIPENGINE_QWEN4_EXP_ROUTED_F16"] == "1"
