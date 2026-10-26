"""Q8_0 token-major mapped down: plan-published runner state, not a process switch."""

from types import SimpleNamespace

import pytest

from hipengine.kernels.registry import KernelKey
from hipengine.runtime import qwen4_exp_runner as runner
from scripts.qwen4exp_halo_box_campaign_ab import (
    Q8_DOWN_CHAIN_GATES,
    Q8_DOWN_PREEMPTING_ENV,
    Q8_MAPPED_DOWN_MAP_GATE,
    Q8_MAPPED_DOWN_PREEMPTING_ENV,
    _apply_mode,
    q8_bundle_call_in_scope,
    q8_down_control_record,
    q8_down_scope,
    q8_mapped_down_control_record,
    q8_mapped_down_expected_calls,
    q8_mapped_down_scope,
    route_scope,
)

#: The retired process switch. Nothing reads it any more; the tests keep the
#: literal so a leftover value can be proven inert.
RETIRED_MAPPED_DOWN_ENV = "HIPENGINE_QWEN4_EXP_Q8_MAPPED_DOWN"

BUNDLE = "selected_grouped_row4_bundle_gemv_bf16_bf16_out"
REGISTER = "selected_grouped_row4_register_gemv_bf16_bf16_out"


def _select(**overrides):
    args = dict(backend="hip_gfx1151", quant="gguf_q8_0", rows=512, map_ready=True)
    args.update(overrides)
    return runner._qwen4_exp_mapped_down_key(**args)


def test_route_needs_the_plan_published_decision(monkeypatch):
    monkeypatch.setattr(runner, "is_registered", lambda key: True)
    assert _select(published=False) is None
    selected = _select(published=True)
    assert selected is not None and selected.variant == BUNDLE


@pytest.mark.parametrize("rows,expected", [
    (1, None), (63, None), (511, None),
    (512, BUNDLE), (513, BUNDLE), (700, BUNDLE), (4096, BUNDLE),
])
def test_route_keeps_the_rows_ge_512_guard(monkeypatch, rows, expected):
    monkeypatch.setattr(runner, "is_registered", lambda key: True)
    selected = _select(rows=rows, published=True)
    assert (None if selected is None else selected.variant) == expected


def test_map_ready_false_never_admits_the_key(monkeypatch):
    # Allocated scratch is not ownership: the map has to come from this
    # invocation, so a ready registry and large row count cannot substitute.
    monkeypatch.setattr(runner, "is_registered", lambda key: True)
    for rows in (512, 1024, 4096):
        assert _select(rows=rows, map_ready=False, published=True) is None


def test_route_falls_back_when_the_leaf_is_not_registered(monkeypatch):
    monkeypatch.setattr(runner, "is_registered", lambda key: False)
    assert _select(published=True) is None


def test_route_ignores_the_retired_process_switch(monkeypatch):
    # A leftover env value may not re-open the route on a runner that published
    # the parent, and the published value may not depend on the environment.
    monkeypatch.setattr(runner, "is_registered", lambda key: True)
    monkeypatch.setenv(RETIRED_MAPPED_DOWN_ENV, "1")
    assert _select(published=False) is None
    monkeypatch.setenv(RETIRED_MAPPED_DOWN_ENV, "0")
    assert _select(published=True).variant == BUNDLE


def test_the_mapped_key_is_the_register_selector_parent(monkeypatch):
    # The token-major mapped path resolves the bundle key; the published
    # register decision may move it to its sibling, exactly as the sorted chain
    # does, and the parent comes back when that decision is False.
    monkeypatch.setattr(runner, "is_registered", lambda key: True)
    mapped = _select(published=True)
    select = runner._qwen4_exp_q8_down_register_key
    for published, expected in ((True, REGISTER), (False, BUNDLE)):
        assert select(mapped, rows=512, in_features=640,
                      published=published).variant == expected


def test_two_runners_keep_independent_mapped_decisions():
    published = SimpleNamespace(q8_mapped_down=True)
    parent = SimpleNamespace(q8_mapped_down=False)
    assert _select(published=published.q8_mapped_down).variant == BUNDLE
    assert _select(published=parent.q8_mapped_down) is None


def test_binding_a_scope_changes_nothing_until_entered():
    owner = SimpleNamespace(q8_mapped_down=False)
    scope = runner.q8_mapped_down(owner, True)
    assert owner.q8_mapped_down is False
    with scope:
        assert owner.q8_mapped_down is True
    assert owner.q8_mapped_down is False


def test_scope_binds_one_runner_and_restores_the_published_value():
    first = SimpleNamespace(q8_mapped_down=False)
    second = SimpleNamespace(q8_mapped_down=False)
    with runner.q8_mapped_down(first, True):
        assert first.q8_mapped_down is True
        assert second.q8_mapped_down is False
    assert first.q8_mapped_down is False


def test_interleaved_scopes_restore_each_runner():
    # The two arms of a campaign alternate on one process; each binding must
    # unwind to the value that runner held, not to the other's.
    first = SimpleNamespace(q8_mapped_down=True)
    second = SimpleNamespace(q8_mapped_down=False)
    with runner.q8_mapped_down(first, False):
        with runner.q8_mapped_down(second, True):
            assert (first.q8_mapped_down, second.q8_mapped_down) == (False, True)
        assert (first.q8_mapped_down, second.q8_mapped_down) == (False, False)
    assert (first.q8_mapped_down, second.q8_mapped_down) == (True, False)


def test_scope_restores_the_published_value_after_an_exception():
    owner = SimpleNamespace(q8_mapped_down=True)
    with pytest.raises(RuntimeError, match="arm failed"):
        with runner.q8_mapped_down(owner, False):
            raise RuntimeError("arm failed")
    assert owner.q8_mapped_down is True


def test_unset_scope_keeps_the_published_value():
    owner = SimpleNamespace(q8_mapped_down=True)
    with runner.q8_mapped_down(owner):
        assert owner.q8_mapped_down is True
    assert owner.q8_mapped_down is True


def test_cloned_runner_inherits_the_published_decision():
    from hipengine.generation.qwen4_exp_gguf import _inherit_lane_decisions

    clone = SimpleNamespace()
    _inherit_lane_decisions(SimpleNamespace(q8_mapped_down=True), clone)
    assert clone.q8_mapped_down is True
    # An unbound carrier publishes the parent, never the other lane's value.
    _inherit_lane_decisions(SimpleNamespace(), clone)
    assert clone.q8_mapped_down is False


def test_campaign_arm_writes_no_process_switch():
    for mode in ("before", "after"):
        env = {"other": "keep"}
        _apply_mode(mode, route_package="q8-mapped-down", environment=env)
        assert env == {"other": "keep"}, (mode, env)


@pytest.mark.parametrize("mode,expected", [("before", False), ("after", True)])
def test_route_scope_binds_the_runner_selection(mode, expected):
    owner = SimpleNamespace(q8_mapped_down=None, q8_down_register=True)
    env = {
        RETIRED_MAPPED_DOWN_ENV: "1",
        Q8_MAPPED_DOWN_MAP_GATE: "1",
        **{key: "1" for key in Q8_DOWN_CHAIN_GATES},
        **{key: "1" for key in Q8_MAPPED_DOWN_PREEMPTING_ENV},
    }
    with route_scope("q8-mapped-down", mode, owner, environment=env):
        assert owner.q8_mapped_down is expected
        # The register selector is a separate decision; the mapped arm must
        # leave it exactly as the plan published it.
        assert owner.q8_down_register is True
        # Every preempting owner is closed so the counted path can run.
        assert all(env[key] == "0" for key in Q8_MAPPED_DOWN_PREEMPTING_ENV)
        # The chain's own gates, the map gate and the retired switch are
        # untouched.
        assert all(env[key] == "1" for key in Q8_DOWN_CHAIN_GATES)
        assert env[Q8_MAPPED_DOWN_MAP_GATE] == "1"
        assert env[RETIRED_MAPPED_DOWN_ENV] == "1"
    assert owner.q8_mapped_down is None
    assert owner.q8_down_register is True
    assert env[RETIRED_MAPPED_DOWN_ENV] == "1"
    assert all(env[key] == "1" for key in Q8_DOWN_CHAIN_GATES)
    assert all(env[key] == "1" for key in Q8_MAPPED_DOWN_PREEMPTING_ENV)


def test_route_scope_rejects_the_arm_without_its_runner():
    with pytest.raises(ValueError, match="needs the owning runner"):
        route_scope("q8-mapped-down", "after")


def test_building_the_route_scope_enters_nothing():
    # The campaign builds one scope per arm and enters it around the measured
    # run; an eager ExitStack would bind the arm before the run and leave it
    # bound when the caller never enters.
    owner = SimpleNamespace(q8_mapped_down=True)
    env = {Q8_MAPPED_DOWN_MAP_GATE: "1"}
    env.update({key: "1" for key in Q8_MAPPED_DOWN_PREEMPTING_ENV})
    env.update({key: "1" for key in Q8_DOWN_CHAIN_GATES})
    scope = route_scope("q8-mapped-down", "before", owner, environment=env)
    assert all(env[key] == "1" for key in Q8_MAPPED_DOWN_PREEMPTING_ENV)
    assert owner.q8_mapped_down is True
    with scope:
        assert all(env[key] == "0" for key in Q8_MAPPED_DOWN_PREEMPTING_ENV)
        assert owner.q8_mapped_down is False
    assert all(env[key] == "1" for key in Q8_MAPPED_DOWN_PREEMPTING_ENV)
    assert owner.q8_mapped_down is True


@pytest.mark.parametrize("gate", [None, "", "0", "false", "False"])
def test_route_scope_fails_loudly_without_the_row4_map_gate(gate):
    # Without the gate the invocation builds no row4 map, so the mapped owner
    # cannot engage and the arm would report a neutral A/B as the comparison.
    owner = SimpleNamespace(q8_mapped_down=True)
    env = {key: "1" for key in Q8_MAPPED_DOWN_PREEMPTING_ENV}
    env.update({key: "1" for key in Q8_DOWN_CHAIN_GATES})
    if gate is None:
        env.pop(Q8_MAPPED_DOWN_MAP_GATE, None)
    else:
        env[Q8_MAPPED_DOWN_MAP_GATE] = gate
    with pytest.raises(ValueError, match=Q8_MAPPED_DOWN_MAP_GATE):
        with route_scope("q8-mapped-down", "after", owner, environment=env):
            pass
    assert owner.q8_mapped_down is True
    assert all(env[key] == "1" for key in Q8_MAPPED_DOWN_PREEMPTING_ENV)


def test_mapped_scope_closes_the_grouped_iu8_preemptor():
    # The grouped Q5_K IU8 gate/up owner sets ``grouped_prefill`` and keeps the
    # token-major Q5_K map producer from running, so the mapped arm closes it.
    assert "HIPENGINE_QWEN4_EXP_Q5_K_IU8_EXACT" in Q8_MAPPED_DOWN_PREEMPTING_ENV
    env = {Q8_MAPPED_DOWN_MAP_GATE: "1"}
    env.update({key: "1" for key in Q8_DOWN_CHAIN_GATES})
    env.update({key: "1" for key in Q8_MAPPED_DOWN_PREEMPTING_ENV})
    with q8_mapped_down_scope(environment=env):
        assert env["HIPENGINE_QWEN4_EXP_Q5_K_IU8_EXACT"] == "0"
        assert all(env[key] == "0" for key in Q8_MAPPED_DOWN_PREEMPTING_ENV)
    assert env["HIPENGINE_QWEN4_EXP_Q5_K_IU8_EXACT"] == "1"


def test_register_scope_leaves_the_grouped_iu8_gate_alone():
    # The grouped IU8 owner enables the grouped chain the register/bundle arms
    # count, so closing it belongs to the mapped arm only.
    assert "HIPENGINE_QWEN4_EXP_Q5_K_IU8_EXACT" not in Q8_DOWN_PREEMPTING_ENV
    env = {"HIPENGINE_QWEN4_EXP_Q5_K_IU8_EXACT": "1"}
    env.update({key: "1" for key in Q8_DOWN_CHAIN_GATES})
    env.update({key: "1" for key in Q8_DOWN_PREEMPTING_ENV})
    with q8_down_scope(environment=env):
        assert env["HIPENGINE_QWEN4_EXP_Q5_K_IU8_EXACT"] == "1"
    assert env["HIPENGINE_QWEN4_EXP_Q5_K_IU8_EXACT"] == "1"


def test_route_scope_leaves_the_register_selector_alone():
    # The mapped admission is not the register rewrite; a mapped arm must not
    # move the register decision it inherits from the plan.
    owner = SimpleNamespace(q8_mapped_down=True, q8_down_register=True)
    env = {Q8_MAPPED_DOWN_MAP_GATE: "1"}
    env.update({key: "1" for key in Q8_DOWN_PREEMPTING_ENV})
    env.update({key: "1" for key in Q8_DOWN_CHAIN_GATES})
    with route_scope("q8-mapped-down", "before", owner, environment=env):
        assert owner.q8_down_register is True
    assert owner.q8_down_register is True


def test_row4_map_gate_names_the_grouped_gate():
    assert Q8_MAPPED_DOWN_MAP_GATE == "HIPENGINE_QWEN4_EXP_GROUPED_ROW4_PREFILL"


def test_state_gate_arm_writes_no_process_switch():
    from scripts.qwen4exp_row4_state_gate import apply_state_gate_mode

    for enabled in ("0", "1"):
        env = {"other": "keep"}
        owner = SimpleNamespace(q8_mapped_down=True)
        apply_state_gate_mode(
            owner, "q8-mapped-down", enabled, "", environment=env)
        assert env == {"other": "keep"}, (enabled, env)
        assert owner.q8_mapped_down is True


@pytest.mark.parametrize("prompt,chunk,expected", [
    (512, 512, 1), (1024, 512, 2), (4096, 512, 8),
    (576, 512, 1), (511, 512, 0), (2048, 1024, 2),
])
def test_current_chunk_counts(prompt, chunk, expected):
    assert q8_mapped_down_expected_calls(prompt, chunk) == expected


@pytest.mark.parametrize("route,pointer,expected", [
    ("q8-mapped-down", None, False), ("q8-mapped-down", 0, False),
    ("q8-mapped-down", 1024, True), ("q8-down-bundle", None, True),
    ("q8-down-bundle", 1024, False), ("q4-pair", 1024, True),
])
def test_shared_kernel_counter_scope(route, pointer, expected):
    assert q8_bundle_call_in_scope(route, (1, 2, pointer)) is expected


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
        seen.append(kwargs.get("q8_mapped_down", "absent"))
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
    runner.run_qwen4_exp_gdn_layer(100, weights, **kwargs, q8_mapped_down=True)
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
        100, _layer_weights(), **kwargs, q8_mapped_down=True)
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
        100, _layer_weights(), **kwargs, q8_mapped_down=True)
    runner.run_qwen4_exp_dense_qsa_layer(100, _layer_weights(), **kwargs)
    assert seen == [True, False]


def test_moe_graph_capture_forwards_the_selection(monkeypatch):
    # The captured decode path recurses through ``eager``; rows=1 never selects
    # the mapped leaf, but the closure still has to carry the owner's decision
    # rather than silently publishing the parent inside a graph.
    seen = []
    runtime = object()

    class Cache:
        def run(self, key, *, eager, out_ptr, out_nbytes, stream):
            eager(3)

    monkeypatch.setattr(runner, "run_qwen4_exp_moe", _moe_seen(seen))
    runner.run_qwen4_exp_moe(
        1, {}, scratch=_layer_scratch(runtime), rows=1, hidden=2, ffn=4,
        experts=1, top_k=1, graph_cache=Cache(), graph_key="k",
        q8_mapped_down=True)
    assert seen == [True]


def test_q8_down_scope_restores_absent_and_present_values():
    env = {Q8_DOWN_PREEMPTING_ENV[0]: "1", **{key: "1" for key in Q8_DOWN_CHAIN_GATES}}
    with q8_down_scope(environment=env):
        assert all(env[key] == "0" for key in Q8_DOWN_PREEMPTING_ENV)
    assert env == {Q8_DOWN_PREEMPTING_ENV[0]: "1",
                   **{key: "1" for key in Q8_DOWN_CHAIN_GATES}}


def test_control_record_separates_the_forced_value_from_the_previous_one():
    env = {
        "HIPENGINE_QWEN4_EXP_ROUTED_F16": "1",
        "HIPENGINE_QWEN4_EXP_Q8_0_SELECTED_WMMA_DOWN": "1",
        **{key: "1" for key in Q8_DOWN_CHAIN_GATES},
    }
    record = q8_down_control_record(environment=env)
    assert record["preempting_owners_closed"] == {
        key: "0" for key in Q8_DOWN_PREEMPTING_ENV}
    assert record["chain_gates"] == {key: "1" for key in Q8_DOWN_CHAIN_GATES}
    assert env["HIPENGINE_QWEN4_EXP_ROUTED_F16"] == "1"


def test_mapped_control_record_separates_prior_and_effective():
    env = {Q8_MAPPED_DOWN_MAP_GATE: "1"}
    env.update({key: "1" for key in Q8_DOWN_CHAIN_GATES})
    env.update({key: "1" for key in Q8_MAPPED_DOWN_PREEMPTING_ENV})
    record = q8_mapped_down_control_record(environment=env)
    assert record["preempting_owners_closed"] == {
        key: "0" for key in Q8_MAPPED_DOWN_PREEMPTING_ENV}
    assert record["preempting_owners_previous"] == {
        key: "1" for key in Q8_MAPPED_DOWN_PREEMPTING_ENV}
    assert record["chain_gates"] == {key: "1" for key in Q8_DOWN_CHAIN_GATES}
    assert record["mapped_gate_up_map"] == "1"
    assert env[Q8_MAPPED_DOWN_MAP_GATE] == "1"
