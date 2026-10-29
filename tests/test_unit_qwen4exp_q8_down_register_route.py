"""Q8_0 down register selection: plan-published runner state, not a process switch."""

from types import SimpleNamespace

import pytest

from hipengine.kernels import registry
from hipengine.kernels.registry import KernelKey
from hipengine.runtime import qwen4_exp_runner as runner
from scripts.qwen4exp_halo_box_campaign_ab import (
    Q8_DOWN_CHAIN_GATES,
    Q8_DOWN_FORKB_FIELD,
    Q8_DOWN_PREEMPTING_ENV,
    _apply_mode,
    q8_down_control_record,
    q8_down_register_expected_calls,
    q8_down_scope,
    q8_mapped_down_expected_calls,
    route_scope,
)

#: The retired process switch. Nothing reads it any more; the tests keep the
#: literal so a leftover value can be proven inert.
RETIRED_DOWN_REGISTER_ENV = "HIPENGINE_QWEN4_EXP_Q8_DOWN_REGISTER"

BUNDLE = "selected_grouped_row4_bundle_gemv_bf16_bf16_out"
REGISTER = "selected_grouped_row4_register_gemv_bf16_bf16_out"


def _key(variant: str = BUNDLE) -> KernelKey:
    return KernelKey("hip_gfx1151", "linear", "gguf_q8_0", variant)


def _owner(**fields):
    """A runner with the sorted chain's own gate published."""

    return SimpleNamespace(**{Q8_DOWN_FORKB_FIELD: True, **fields})


def test_route_selects_register_only_when_the_plan_published_it(monkeypatch):
    monkeypatch.setattr(runner, "is_registered", lambda key: True)
    select = runner._qwen4_exp_q8_down_register_key
    assert select(_key(), rows=1024, in_features=640, published=False) == _key()
    assert select(
        _key(), rows=1024, in_features=640, published=True).variant == REGISTER


def test_route_keeps_the_rows_k640_and_parent_guards(monkeypatch):
    monkeypatch.setattr(runner, "is_registered", lambda key: True)
    select = runner._qwen4_exp_q8_down_register_key
    assert select(_key(), rows=511, in_features=640, published=True) == _key()
    assert select(_key(), rows=1024, in_features=512, published=True) == _key()
    other = _key("selected_grouped_row4_gemv_bf16_bf16_out")
    assert select(other, rows=1024, in_features=640, published=True) == other


def test_route_falls_back_when_the_leaf_is_not_registered(monkeypatch):
    monkeypatch.setattr(runner, "is_registered", lambda key: False)
    assert runner._qwen4_exp_q8_down_register_key(
        _key(), rows=1024, in_features=640, published=True) == _key()


def test_route_ignores_the_retired_process_switch(monkeypatch):
    # The decision is the plan's, published on the owning runner. A leftover env
    # value may not re-open the route on an arm that published the parent, and
    # the published value may not depend on the environment.
    monkeypatch.setattr(runner, "is_registered", lambda key: True)
    monkeypatch.setenv(RETIRED_DOWN_REGISTER_ENV, "1")
    assert runner._qwen4_exp_q8_down_register_key(
        _key(), rows=1024, in_features=640, published=False) == _key()
    monkeypatch.setenv(RETIRED_DOWN_REGISTER_ENV, "0")
    assert runner._qwen4_exp_q8_down_register_key(
        _key(), rows=1024, in_features=640, published=True).variant == REGISTER


def test_sorted_and_mapped_paths_share_one_published_decision(monkeypatch):
    # The sorted path reaches the selector through the row4/bundle chain; the
    # mapped path builds the same bundle key from the mmap table. Both must
    # follow the one runner decision, in both directions.
    monkeypatch.setattr(runner, "is_registered", lambda key: True)
    sorted_key = runner._qwen4_exp_q8_down_bundle_key(
        runner._qwen4_exp_q8_down_row4_key(
            KernelKey("hip_gfx1151", "linear", "gguf_q8_0",
                      "selected_grouped_gemv_bf16_bf16_out"), rows=1024,
            published=True),
        rows=1024, published=True)
    assert sorted_key.variant == BUNDLE
    mapped_key = runner._qwen4_exp_mapped_down_key(
        backend="hip_gfx1151", quant="gguf_q8_0", rows=1024, map_ready=True,
        published=True)
    assert mapped_key is not None and mapped_key.variant == BUNDLE
    select = runner._qwen4_exp_q8_down_register_key
    for published, expected in ((True, REGISTER), (False, BUNDLE)):
        assert select(sorted_key, rows=1024, in_features=640,
                      published=published).variant == expected
        assert select(mapped_key, rows=1024, in_features=640,
                      published=published).variant == expected


def test_two_runners_keep_independent_decisions():
    # Request-local state, so one lane's arm cannot move another lane's route.
    published = SimpleNamespace(q8_down_register=True)
    parent = SimpleNamespace(q8_down_register=False)
    assert runner._qwen4_exp_q8_down_register_key(
        _key(), rows=1024, in_features=640,
        published=published.q8_down_register).variant == REGISTER
    assert runner._qwen4_exp_q8_down_register_key(
        _key(), rows=1024, in_features=640,
        published=parent.q8_down_register) == _key()


def test_binding_a_scope_changes_nothing_until_entered():
    owner = SimpleNamespace(q8_down_register=False)
    scope = runner.q8_down_register(owner, True)
    assert owner.q8_down_register is False
    with scope:
        assert owner.q8_down_register is True
    assert owner.q8_down_register is False


def test_scope_binds_one_runner_and_restores_the_published_value():
    first = SimpleNamespace(q8_down_register=False)
    second = SimpleNamespace(q8_down_register=False)
    with runner.q8_down_register(first, True):
        assert first.q8_down_register is True
        assert second.q8_down_register is False
    assert first.q8_down_register is False


def test_interleaved_scopes_restore_each_runner():
    # The two arms of a campaign alternate on one process; each binding must
    # unwind to the value that runner held, not to the other's.
    first = SimpleNamespace(q8_down_register=True)
    second = SimpleNamespace(q8_down_register=False)
    with runner.q8_down_register(first, False):
        with runner.q8_down_register(second, True):
            assert (first.q8_down_register, second.q8_down_register) == (False, True)
        assert (first.q8_down_register, second.q8_down_register) == (False, False)
    assert (first.q8_down_register, second.q8_down_register) == (True, False)


def test_scope_restores_the_published_value_after_an_exception():
    owner = SimpleNamespace(q8_down_register=True)
    with pytest.raises(RuntimeError, match="arm failed"):
        with runner.q8_down_register(owner, False):
            raise RuntimeError("arm failed")
    assert owner.q8_down_register is True


def test_unset_scope_keeps_the_published_value():
    owner = SimpleNamespace(q8_down_register=True)
    with runner.q8_down_register(owner):
        assert owner.q8_down_register is True
    assert owner.q8_down_register is True


def test_cloned_runner_inherits_the_published_decision():
    from hipengine.generation.qwen4_exp_gguf import _inherit_lane_decisions

    clone = SimpleNamespace()
    _inherit_lane_decisions(SimpleNamespace(q8_down_register=True), clone)
    assert clone.q8_down_register is True
    # An unbound carrier publishes the parent, never the other lane's value.
    _inherit_lane_decisions(SimpleNamespace(), clone)
    assert clone.q8_down_register is False


def test_campaign_arm_writes_no_process_switch():
    for mode in ("before", "after"):
        env = {"other": "keep"}
        _apply_mode(mode, route_package="q8-down-register", environment=env)
        assert env == {"other": "keep"}, (mode, env)


@pytest.mark.parametrize("mode,expected", [("before", False), ("after", True)])
def test_route_scope_binds_the_runner_selection(mode, expected):
    # The register leaf rides both Q8_0 ladder rungs, which are runner state.
    owner = SimpleNamespace(q8_down_register=None,
                            q8_down_row4_prefill=True,
                            q8_down_bundle_prefill=True,
                            forkb_grouped_down=True)
    env = {
        RETIRED_DOWN_REGISTER_ENV: "1",
        **{key: "1" for key in Q8_DOWN_PREEMPTING_ENV},
    }
    with route_scope("q8-down-register", mode, owner, environment=env):
        assert owner.q8_down_register is expected
        # Every preempting owner is closed so the counted chain can run.
        assert all(env[key] == "0" for key in Q8_DOWN_PREEMPTING_ENV)
        # The chain's own gate and the retired switch are untouched.
        assert owner.forkb_grouped_down is True
        assert env[RETIRED_DOWN_REGISTER_ENV] == "1"
    assert owner.q8_down_register is None
    assert env[RETIRED_DOWN_REGISTER_ENV] == "1"
    assert owner.forkb_grouped_down is True
    assert all(env[key] == "1" for key in Q8_DOWN_PREEMPTING_ENV)


def test_route_scope_rejects_the_arm_without_its_runner():
    with pytest.raises(ValueError, match="needs the owning runner"):
        route_scope("q8-down-register", "after")


@pytest.mark.parametrize("tokens,expected", [(512, 6), (1024, 6),
                                            (4096, 24), (511, 0), (1536, 12)])
def test_current_chunk_counts(monkeypatch, tokens, expected):
    # The arm counts the grouped register leaf (the artifact's Q8_0 down layers
    # per rows>=512 chunk) plus the token-major mapped path's own count.
    from tests._qwen4exp_expert_down_fixtures import expert_down_runner

    monkeypatch.setattr(registry, "is_registered", lambda key: True)
    assert q8_down_register_expected_calls(
        expert_down_runner(), tokens, 1024) == expected


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
        seen.append(kwargs.get("q8_down_register", "absent"))
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
    runner.run_qwen4_exp_gdn_layer(100, weights, **kwargs, q8_down_register=True)
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
        100, _layer_weights(), **kwargs, q8_down_register=True)
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
        100, _layer_weights(), **kwargs, q8_down_register=True)
    runner.run_qwen4_exp_dense_qsa_layer(100, _layer_weights(), **kwargs)
    assert seen == [True, False]


def test_moe_graph_capture_forwards_the_selection(monkeypatch):
    # The captured decode path recurses through ``eager``; rows=1 never selects
    # the register leaf, but the closure still has to carry the owner's
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
        q8_down_register=True)
    assert seen == [True]


def test_q8_down_scope_restores_absent_and_present_values():
    env = {Q8_DOWN_PREEMPTING_ENV[0]: "1"}
    with q8_down_scope(environment=env, runner=_owner()):
        assert all(env[key] == "0" for key in Q8_DOWN_PREEMPTING_ENV)
    assert env == {Q8_DOWN_PREEMPTING_ENV[0]: "1"}


def test_q8_down_scope_restores_after_an_exception():
    env = {key: "1" for key in Q8_DOWN_PREEMPTING_ENV}
    with pytest.raises(RuntimeError, match="arm failed"):
        with q8_down_scope(environment=env, runner=_owner()):
            raise RuntimeError("arm failed")
    assert all(env[key] == "1" for key in Q8_DOWN_PREEMPTING_ENV)


def test_q8_down_scope_changes_nothing_until_entered():
    env = {key: "1" for key in Q8_DOWN_PREEMPTING_ENV}
    scope = q8_down_scope(environment=env, runner=_owner())
    assert all(env[key] == "1" for key in Q8_DOWN_PREEMPTING_ENV)
    with scope:
        assert all(env[key] == "0" for key in Q8_DOWN_PREEMPTING_ENV)
    assert all(env[key] == "1" for key in Q8_DOWN_PREEMPTING_ENV)


def test_q8_down_control_names_every_preemptor():
    # The routed F16 down is the production owner of the grouped chain and the
    # WMMA down sits in front of the sorted chain; both are forced closed. The
    # chain's own gate is runner state, recorded never forced.
    assert Q8_DOWN_PREEMPTING_ENV == (
        "HIPENGINE_QWEN4_EXP_Q8_0_SELECTED_WMMA_DOWN",
    )
    # Both ladder rungs and the chain's own gate are request-local runner state,
    # so no process key gates either chain any more.
    assert Q8_DOWN_CHAIN_GATES == ()
    assert Q8_DOWN_FORKB_FIELD == "forkb_grouped_down"
    assert not set(Q8_DOWN_CHAIN_GATES) & set(Q8_DOWN_PREEMPTING_ENV)


def test_route_scope_leaves_the_preemptors_alone_for_other_packages():
    env = {key: "1" for key in Q8_DOWN_PREEMPTING_ENV}
    owner = SimpleNamespace(gdn_prefill_variant=None)
    with route_scope("gdn-wave-norm", "after", owner, environment=env):
        assert all(env[key] == "1" for key in Q8_DOWN_PREEMPTING_ENV)
    assert all(env[key] == "1" for key in Q8_DOWN_PREEMPTING_ENV)


def test_building_a_q8_down_route_scope_binds_nothing():
    owner = SimpleNamespace(q8_down_register=True, q8_down_row4_prefill=True,
                            q8_down_bundle_prefill=True,
                            forkb_grouped_down=True)
    env = {key: "1" for key in Q8_DOWN_PREEMPTING_ENV}
    scope = route_scope("q8-down-register", "after", owner, environment=env)
    assert all(env[key] == "1" for key in Q8_DOWN_PREEMPTING_ENV)
    assert owner.q8_down_register is True
    with scope:
        assert all(env[key] == "0" for key in Q8_DOWN_PREEMPTING_ENV)
        assert owner.q8_down_register is True
    assert all(env[key] == "1" for key in Q8_DOWN_PREEMPTING_ENV)
    assert owner.q8_down_register is True


@pytest.mark.parametrize("gate", [None, False])
def test_q8_down_scope_fails_loudly_without_a_chain_gate(gate):
    # Without the chain's own gate the counted leaf cannot launch, so an arm
    # would report an engagement it cannot produce. The gate is runner state,
    # so the scope refuses an unpublish or an explicit off value alike.
    env = {key: "1" for key in Q8_DOWN_PREEMPTING_ENV}
    owner = SimpleNamespace() if gate is None else SimpleNamespace(
        **{Q8_DOWN_FORKB_FIELD: gate})
    with pytest.raises(ValueError, match=Q8_DOWN_FORKB_FIELD):
        with q8_down_scope(environment=env, runner=owner):
            pass
    assert all(env[key] == "1" for key in Q8_DOWN_PREEMPTING_ENV)


@pytest.mark.parametrize("mode", ["before", "after"])
def test_route_scope_fails_loudly_without_a_chain_gate(mode):
    # The register leaf rides both Q8_0 ladder rungs, which are runner state.
    owner = SimpleNamespace(q8_down_register=None,
                            q8_down_row4_prefill=True,
                            q8_down_bundle_prefill=True)
    env = {key: "1" for key in Q8_DOWN_PREEMPTING_ENV}
    with pytest.raises(ValueError, match=Q8_DOWN_FORKB_FIELD):
        with route_scope("q8-down-register", mode, owner, environment=env):
            pass
    assert owner.q8_down_register is None
    assert all(env[key] == "1" for key in Q8_DOWN_PREEMPTING_ENV)


def test_control_record_separates_the_forced_value_from_the_previous_one():
    # A reader must not see the production binding under a key that claims the
    # arm closed it.
    env = {
        "HIPENGINE_QWEN4_EXP_ROUTED_F16": "1",
        "HIPENGINE_QWEN4_EXP_Q8_0_SELECTED_WMMA_DOWN": "1",
    }
    record = q8_down_control_record(runner=_owner(), environment=env)
    assert record["preempting_owners_closed"] == {
        key: "0" for key in Q8_DOWN_PREEMPTING_ENV}
    assert record["preempting_fields"]["routed_f16_enabled"] == {
        "bound_value": False, "published_value": None}
    assert record["chain_gate"] == {
        "runner_field": Q8_DOWN_FORKB_FIELD, "published_value": True}
    # Recording changes nothing.
    assert env["HIPENGINE_QWEN4_EXP_ROUTED_F16"] == "1"


def test_mapped_and_compact_split_matches_the_shared_counts(monkeypatch):
    # The arm counts both paths through one expectation; the mapped share is the
    # token-major path's own count.
    from tests._qwen4exp_expert_down_fixtures import expert_down_runner

    monkeypatch.setattr(registry, "is_registered", lambda key: True)
    owner = expert_down_runner()
    for tokens in (512, 1024, 1536, 4096):
        total = q8_down_register_expected_calls(owner, tokens, 1024)
        mapped = q8_mapped_down_expected_calls(tokens, 1024)
        assert 0 < mapped <= total
