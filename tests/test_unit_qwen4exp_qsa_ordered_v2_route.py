"""QSA ordered-v2 sparse decode: plan-published runner state, not a process switch."""

import os
from types import SimpleNamespace

import pytest

from hipengine.execution_profiles import ExecutionProfile, resolve_runtime_profile
from hipengine.generation.qwen4_exp_profiles import (
    PRODUCTION_ARITHMETIC_RECOVERY_FLAGS,
    PRODUCTION_Q8_QSA_RESTORED_FLAGS,
    register_qwen4_exp_gfx1151_profiles,
)
from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
from hipengine.kernels.registry import KernelKey, is_registered
from hipengine.runtime import qwen4_exp_runner as runner
from scripts.qwen4exp_halo_box_campaign_ab import (
    QSA_ORDERED_DECODE_ENV,
    _apply_mode,
    qsa_ordered_decode_v2_control_record,
    route_scope,
)
from scripts.qwen4exp_row4_state_gate import apply_state_gate_mode

#: The retired process switch. Nothing reads it any more; the tests keep the
#: literal so a leftover value can be proven inert.
RETIRED_V2_ENV = "HIPENGINE_QWEN4_EXP_QSA_ORDERED_DECODE_V2"
#: The coefficients+values sibling keeps its own diagnostic switch.
FUSED_ENV = "HIPENGINE_QWEN4_EXP_QSA_ORDERED_DECODE_V2_FUSED"

V1 = runner.qwen4_exp_qsa_sparse_attention_paged_bf16_ordered_f32
V2 = runner.qwen4_exp_qsa_sparse_attention_paged_bf16_ordered_v2_f32
FUSED = runner.qwen4_exp_qsa_sparse_attention_paged_bf16_fused_v2_f32


def _bind(monkeypatch, profile_name, *, runner_handle=None):
    monkeypatch.setattr(os, "environ", os.environ.copy())
    register_gfx1151_kernels(replace=True)
    register_qwen4_exp_gfx1151_profiles()
    profile = resolve_runtime_profile(
        model="qwen4_exp_gguf", backend="hip_gfx1151",
        quant="gguf_ud_q4_k_xl", profile=profile_name)
    generator = SimpleNamespace()
    if runner_handle is not None:
        generator.runner = runner_handle
    profile.binder(generator, profile)
    return generator, profile


# --- the plan publishes the decision; no switch selects it ------------------


@pytest.mark.parametrize("profile_name,expected", [
    (ExecutionProfile.PRODUCTION, True),
    (ExecutionProfile.STRICT, False),
    (ExecutionProfile.BATCH_INVARIANT, False),
])
def test_the_binder_publishes_the_plan_decision_and_writes_no_switch(
        monkeypatch, profile_name, expected):
    generator, _ = _bind(monkeypatch, profile_name)
    assert generator.qsa_ordered_decode_v2 is expected
    assert RETIRED_V2_ENV not in os.environ


def test_the_binder_publishes_on_the_runner_it_finds(monkeypatch):
    handle = SimpleNamespace()
    generator, _ = _bind(monkeypatch, ExecutionProfile.PRODUCTION,
                         runner_handle=handle)
    assert generator.runner.qsa_ordered_decode_v2 is True
    assert RETIRED_V2_ENV not in os.environ


def test_the_recovery_and_restoration_tables_do_not_carry_the_leaf():
    # The leaf is request-local now, so neither table may rewrite it; the v1
    # route it rides stays an ordinary recovery switch.
    from scripts.qwen4exp_layer2_profile_gate import CONSERVATIVE_ARITHMETIC_FLAGS

    for table in (PRODUCTION_ARITHMETIC_RECOVERY_FLAGS,
                  tuple(PRODUCTION_Q8_QSA_RESTORED_FLAGS),
                  CONSERVATIVE_ARITHMETIC_FLAGS):
        assert "QSA_ORDERED_DECODE_V2" not in table
    assert "QSA_ORDERED_DECODE" in PRODUCTION_ARITHMETIC_RECOVERY_FLAGS
    assert PRODUCTION_Q8_QSA_RESTORED_FLAGS["QSA_ORDERED_DECODE"] == "1"


# --- dispatch-time guards ---------------------------------------------------


def test_the_leaf_keeps_its_v1_h256_and_count_guards():
    guard = runner._qwen4_exp_qsa_ordered_decode_v2
    assert guard(ordered=True, head_dim=256, selected_count=1, published=True)
    assert guard(ordered=True, head_dim=256, selected_count=4096, published=True)
    assert guard(ordered=True, head_dim=256, selected_count=2048, published=True)
    assert not guard(ordered=True, head_dim=256, selected_count=0, published=True)
    assert not guard(ordered=True, head_dim=256, selected_count=4097, published=True)
    # The v1 ordered route is the prerequisite, not a sibling selection.
    assert not guard(ordered=False, head_dim=256, selected_count=100,
                     published=True)
    # Unrelated shapes stay on the parent too.
    assert not guard(ordered=True, head_dim=128, selected_count=100,
                     published=True)
    assert not guard(ordered=True, head_dim=512, selected_count=100,
                     published=True)
    assert not guard(ordered=True, head_dim=256, selected_count=100,
                     published=False)


def test_the_retired_switch_cannot_move_the_selection(monkeypatch):
    monkeypatch.setenv(RETIRED_V2_ENV, "1")
    assert not runner._qwen4_exp_qsa_ordered_decode_v2(
        ordered=True, head_dim=256, selected_count=100, published=False)
    monkeypatch.setenv(RETIRED_V2_ENV, "0")
    assert runner._qwen4_exp_qsa_ordered_decode_v2(
        ordered=True, head_dim=256, selected_count=100, published=True)


def _leaf(**overrides):
    kwargs = dict(ordered=True, head_dim=256, selected_count=100,
                  published=True, fused=False, resolve_v2=lambda: V2)
    kwargs.update(overrides)
    return runner._qwen4_exp_qsa_ordered_decode_leaf(**kwargs)


def test_a_published_production_step_selects_the_v2_leaf():
    assert _leaf() is V2
    # The ordered route off selects nothing: the caller keeps the non-ordered
    # sparse owner.
    assert _leaf(ordered=False) is None


def test_an_unregistered_v2_kernel_falls_back_to_the_ordered_parent():
    # Not to the non-ordered owner: the step is still an ordered decode.
    assert _leaf(resolve_v2=lambda: None) is V1
    assert _leaf(resolve_v2=None) is V1


def test_the_fused_sibling_requires_an_eligible_v2_step(monkeypatch):
    # Its own switch is unchanged; it only stacks on top of the v2 leaf.
    assert _leaf(fused=True) is FUSED
    assert _leaf(fused=True, published=False) is V1
    assert _leaf(fused=True, head_dim=128) is V1
    assert _leaf(fused=True, selected_count=4097) is V1
    assert _leaf(fused=True, ordered=False) is None
    # The env name is untouched and still a separate diagnostic.
    assert FUSED_ENV == "HIPENGINE_QWEN4_EXP_QSA_ORDERED_DECODE_V2_FUSED"
    monkeypatch.setenv(FUSED_ENV, "1")
    assert not runner._qwen4_exp_qsa_ordered_decode_v2(
        ordered=True, head_dim=256, selected_count=100, published=False)


def test_the_gfx1151_backend_registers_the_v2_leaf():
    register_gfx1151_kernels(replace=True)
    assert is_registered(KernelKey("hip_gfx1151", "qsa_sparse_attention",
                                   "bf16_kv",
                                   "strict_ordered_three_pass_v2_spans"))


# --- request-local scope ----------------------------------------------------


def test_binding_a_scope_changes_nothing_until_entered():
    owner = SimpleNamespace(qsa_ordered_decode_v2=False)
    scope = runner.qsa_ordered_decode_v2(owner, True)
    assert owner.qsa_ordered_decode_v2 is False
    with scope:
        assert owner.qsa_ordered_decode_v2 is True
    assert owner.qsa_ordered_decode_v2 is False


def test_scope_binds_one_runner_and_restores_the_published_value():
    first = SimpleNamespace(qsa_ordered_decode_v2=False)
    second = SimpleNamespace(qsa_ordered_decode_v2=True)
    with runner.qsa_ordered_decode_v2(first, True):
        assert first.qsa_ordered_decode_v2 is True
        assert second.qsa_ordered_decode_v2 is True
    assert first.qsa_ordered_decode_v2 is False
    assert second.qsa_ordered_decode_v2 is True


def test_scope_restores_the_published_value_after_an_exception():
    owner = SimpleNamespace(qsa_ordered_decode_v2=True)
    with pytest.raises(RuntimeError, match="arm failed"):
        with runner.qsa_ordered_decode_v2(owner, False):
            raise RuntimeError("arm failed")
    assert owner.qsa_ordered_decode_v2 is True


def test_unset_scope_keeps_the_published_value():
    owner = SimpleNamespace(qsa_ordered_decode_v2=True)
    with runner.qsa_ordered_decode_v2(owner):
        assert owner.qsa_ordered_decode_v2 is True
    assert owner.qsa_ordered_decode_v2 is True


def test_a_runner_built_without_the_field_keeps_the_parent():
    owner = object.__new__(runner.Qwen4ExpGGUFResidentModelRunner)
    assert owner.qsa_ordered_decode_v2 is False
    with runner.qsa_ordered_decode_v2(owner, True):
        assert owner.qsa_ordered_decode_v2 is True
    assert owner.qsa_ordered_decode_v2 is False


def test_cloned_runner_inherits_the_published_decision():
    from hipengine.generation.qwen4_exp_gguf import _inherit_lane_decisions

    clone = SimpleNamespace()
    _inherit_lane_decisions(SimpleNamespace(qsa_ordered_decode_v2=True), clone)
    assert clone.qsa_ordered_decode_v2 is True
    # An unbound carrier publishes the parent, never the other lane's value.
    _inherit_lane_decisions(SimpleNamespace(), clone)
    assert clone.qsa_ordered_decode_v2 is False


def test_the_batch_route_carries_no_c1_ordered_v2_selection():
    # The batch owner has its own mixed sparse path; the c1 decision must not
    # leak into it.
    import inspect

    from hipengine.runtime import qwen4_exp_batch_qsa

    assert "qsa_ordered_decode_v2" not in inspect.getsource(qwen4_exp_batch_qsa)


def test_the_batch_decode_owner_keeps_its_rows_kernels():
    # Proof that the ordered-v2 work leaves the batch defaults alone: the batch
    # still selects the exact rows owners for both geometries, never the
    # ordered c1 decode kernels.
    from hipengine.runtime import qwen4_exp_batch_qsa as batch

    _, h256 = batch.batch_qsa_attention_implementation(
        head_dim=256, block_size=256, query_heads=24, kv_heads=2)
    assert h256 == "strict_h256_head_pair_rows_spans"
    _, rows = batch.batch_qsa_attention_implementation(
        head_dim=128, block_size=256, query_heads=24, kv_heads=2)
    assert rows == "strict_rows_spans"


def test_a_plain_ar_row_decodes_in_the_shared_batch_by_default(monkeypatch):
    # The public serving route for a plain AR request is the shared batch, so a
    # zero ordered-v2 count there is a routing fact. ``AR_BATCH=0`` is the
    # documented rollback lever that keeps a request on its c1 runner; it is
    # not a change this work makes.
    from hipengine.generation import qwen4_exp_ar_batch as ar

    monkeypatch.delenv(ar.AR_BATCH_ENV, raising=False)
    assert ar.ar_batch_enabled() is True
    monkeypatch.setenv(ar.AR_BATCH_ENV, "0")
    assert ar.ar_batch_enabled() is False


def test_a_plain_ar_row_is_batched_and_an_oversized_one_keeps_its_c1_runner():
    # Why a public 4171-token request shows no c1 ordered-v2 launches: a plain
    # AR row is a batch row, so it decodes through the shared owner's rows
    # kernels. Only a row the batch cannot hold, an MTP row, or the AR_BATCH=0
    # rollback keeps the c1 chain that selects the ordered leaf.
    from hipengine.generation.qwen4_exp_gguf import (
        Qwen4ExpResidentServingRunner as resident,
    )

    def row(mtp_budget=0):
        return SimpleNamespace(
            prompt_ids=[0] * 4171,
            request=SimpleNamespace(max_tokens=8,
                                    speculative_mtp_automatic=False),
            mtp_budget=mtp_budget)

    batched = SimpleNamespace(
        _ar_batch=SimpleNamespace(fits=lambda n: True,
                                  runner=SimpleNamespace(supports_mixed_target=False)))
    batched._row_extent = lambda item: resident._row_extent(batched, item)
    assert batched._row_extent(row()) == 4178
    assert resident._batch_row(batched, row()) is True
    # An MTP row keeps its c1 runner when the shared mixed target cannot take it.
    assert resident._batch_row(batched, row(mtp_budget=1)) is False

    oversized = SimpleNamespace(
        _ar_batch=SimpleNamespace(fits=lambda n: False,
                                  runner=SimpleNamespace(supports_mixed_target=False)))
    oversized._row_extent = batched._row_extent
    assert resident._batch_row(oversized, row()) is False


# --- forwarding: the published field must reach the kernel choice -----------


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


def _stub_common(monkeypatch):
    monkeypatch.setattr(runner, "run_qwen4_exp_gr_read", _gr_read)
    monkeypatch.setattr(
        runner, "run_qwen4_exp_moe",
        lambda *a, **k: SimpleNamespace(output=SimpleNamespace(ptr=14)))
    monkeypatch.setattr(runner, "_qwen4_exp_gr_pending_write", lambda *a, **k: None)
    monkeypatch.setattr(runner, "qwen4_exp_gr_write_bf16_f32", lambda *a, **k: None)
    monkeypatch.setattr(runner, "qwen4_exp_gr_write_bf16_bf16", lambda *a, **k: None)


def test_the_dense_qsa_layer_forwards_the_decision_to_the_mixer(monkeypatch):
    seen = []
    runtime = object()
    _stub_common(monkeypatch)
    monkeypatch.setattr(
        runner, "run_qwen4_exp_dense_qsa_token_mixer",
        lambda *a, **k: (seen.append(k.get("qsa_ordered_decode_v2", "absent")),
                         SimpleNamespace(ptr=13))[1])
    kwargs = dict(attention_state=object(), scratch=_layer_scratch(runtime),
                  position=0, rows=1, branches=1, hidden=2, low_rank=2,
                  query_heads=24, kv_heads=2, head_dim=256, rotary_dim=64,
                  theta=1e6, ffn=8, experts=1, top_k=1, runtime=runtime)
    runner.run_qwen4_exp_dense_qsa_layer(
        100, _layer_weights(), **kwargs, qsa_ordered_decode_v2=True)
    runner.run_qwen4_exp_dense_qsa_layer(100, _layer_weights(), **kwargs)
    assert seen == [True, False]


def _draft_stub():
    from hipengine.runtime import qwen4_exp_mtp as draft_module

    draft = object.__new__(draft_module.Qwen4ExpGGUFMTPDraftRunner)
    draft.position, draft.max_sequence_length, draft.closed = 0, 8, False
    draft.config = SimpleNamespace(**{
        name: 2 for name in (
            "residual_branch_count", "hidden_size", "residual_low_rank",
            "attention_head_count", "attention_kv_head_count",
            "attention_key_length", "rope_dimension_count", "rope_freq_base",
            "indexer_head_count", "indexer_key_length",
            "expert_feed_forward_length", "expert_count", "expert_used_count")})
    draft._buffers = [SimpleNamespace(ptr=i, nbytes=4) for i in range(12)]
    draft.runtime = SimpleNamespace(memcpy=lambda *a: None)
    draft.binding = draft.attention = draft.index = draft.layer_scratch = object()
    draft._fuse_inputs = lambda *a, **kw: None
    return draft


def test_the_mtp_draft_forwards_the_published_decision(monkeypatch):
    import numpy as np

    from hipengine.runtime import qwen4_exp_mtp as draft_module

    draft = _draft_stub()
    calls = []
    monkeypatch.setattr(draft_module, "run_qwen4_exp_dense_qsa_layer",
                        lambda *a, **kw: calls.append(kw))
    draft.qsa_ordered_decode_v2 = True
    draft.forward(1, np.zeros(2), run_head=False, cache_only=True)
    assert calls[-1]["qsa_ordered_decode_v2"] is True
    # The cache-only contract is unchanged by the new keyword.
    assert calls[-1]["cache_only"] is True
    draft.qsa_ordered_decode_v2 = False
    draft.forward(2, np.zeros(2), run_head=False, cache_only=True)
    assert calls[-1]["qsa_ordered_decode_v2"] is False
    assert calls[-1]["cache_only"] is True


def test_an_unpublished_draft_keeps_the_ordered_parent(monkeypatch):
    import numpy as np

    from hipengine.runtime import qwen4_exp_mtp as draft_module

    draft = _draft_stub()
    calls = []
    monkeypatch.setattr(draft_module, "run_qwen4_exp_dense_qsa_layer",
                        lambda *a, **kw: calls.append(kw))
    draft.forward(1, np.zeros(2), run_head=False, cache_only=True)
    assert calls[-1]["qsa_ordered_decode_v2"] is False


# --- the two draft construction sites publish it ----------------------------


def test_the_provider_publishes_the_decision_on_the_draft_it_builds(monkeypatch):
    from pathlib import Path

    from hipengine.generation import qwen4_exp_mtp as provider_module
    from hipengine.speculative.registry import SpeculativeProviderConfig

    target = SimpleNamespace(
        backend="hip_gfx1151",
        runner=SimpleNamespace(qsa_ordered_decode_v2=True, runtime=object(),
                               config=object(), max_sequence_length=16))
    built = []

    def factory(*args, **kwargs):
        built.append(SimpleNamespace(**kwargs))
        return built[-1]

    monkeypatch.setattr(provider_module, "discover_gguf_files",
                        lambda path: [Path("draft.gguf")])
    monkeypatch.setattr(provider_module, "GGUFReader",
                        lambda path: SimpleNamespace(info=None))
    monkeypatch.setattr(provider_module, "build_qwen4_exp_mtp_gguf_map",
                        lambda infos: {})
    monkeypatch.setattr(provider_module, "plan_qwen4_exp_mtp_residency",
                        lambda model_map: {})
    monkeypatch.setattr(
        provider_module, "materialize_qwen4_exp_mtp_weights",
        lambda readers, **kwargs: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(provider_module, "Qwen4ExpGGUFMTPDraftRunner", factory)

    provider = provider_module.Qwen4ExpMTPTextProvider(
        target_generator=target,
        config=SpeculativeProviderConfig(
            provider="qwen4_exp_mtp", draft_model="draft.gguf",
            candidate_budget=1))

    assert len(built) == 1
    assert built[0].qsa_ordered_decode_v2 is True
    assert provider.draft is built[0]


def test_the_provider_publishes_the_parent_for_an_unpublished_target(monkeypatch):
    from pathlib import Path

    from hipengine.generation import qwen4_exp_mtp as provider_module
    from hipengine.speculative.registry import SpeculativeProviderConfig

    target = SimpleNamespace(
        backend="hip_gfx1151",
        runner=SimpleNamespace(runtime=object(), config=object(),
                               max_sequence_length=16))
    built = []

    def factory(*args, **kwargs):
        built.append(SimpleNamespace(**kwargs))
        return built[-1]

    monkeypatch.setattr(provider_module, "discover_gguf_files",
                        lambda path: [Path("draft.gguf")])
    monkeypatch.setattr(provider_module, "GGUFReader",
                        lambda path: SimpleNamespace(info=None))
    monkeypatch.setattr(provider_module, "build_qwen4_exp_mtp_gguf_map",
                        lambda infos: {})
    monkeypatch.setattr(provider_module, "plan_qwen4_exp_mtp_residency",
                        lambda model_map: {})
    monkeypatch.setattr(
        provider_module, "materialize_qwen4_exp_mtp_weights",
        lambda readers, **kwargs: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(provider_module, "Qwen4ExpGGUFMTPDraftRunner", factory)

    provider_module.Qwen4ExpMTPTextProvider(
        target_generator=target,
        config=SpeculativeProviderConfig(
            provider="qwen4_exp_mtp", draft_model="draft.gguf",
            candidate_budget=1))

    assert built[0].qsa_ordered_decode_v2 is False


def test_the_serving_pool_publishes_the_decision_on_each_staged_draft(monkeypatch):
    from hipengine.generation import qwen4_exp_serving_mtp as serving_module

    target = SimpleNamespace(
        begin_device_transaction=lambda reuse_snapshot=False: object(),
        commit_device_transaction=lambda checkpoint: None)
    provider = SimpleNamespace(draft=SimpleNamespace(qsa_ordered_decode_v2=False),
                               _resident=object(),
                               iter_generate=lambda **kwargs: None)
    owner = object.__new__(serving_module.Qwen4ExpServingMTP)
    owner.generator = SimpleNamespace(
        _speculative_provider=provider,
        runner=SimpleNamespace(qsa_ordered_decode_v2=True, config=object(),
                               max_sequence_length=8, runtime=object()),
        backend="hip_gfx1151")
    owner.capacity = 2
    owner.staging_capacity = 2
    owner._all_runners = [target]
    owner._init_mtp()
    built = []

    def factory(*args, **kwargs):
        built.append(SimpleNamespace(**kwargs))
        return built[-1]

    monkeypatch.setattr(serving_module, "Qwen4ExpGGUFMTPDraftRunner", factory)
    owner._prepare_mtp()

    assert len(built) == 1
    assert built[0].qsa_ordered_decode_v2 is True
    # The provider's own draft is the first pool entry and keeps its value.
    assert owner._all_drafts[0] is provider.draft
    assert owner._all_drafts[0].qsa_ordered_decode_v2 is False
    assert len(owner._all_drafts) == 2


# --- the live harnesses name the runner field, not a switch -----------------


def test_campaign_arm_writes_no_process_switch():
    for mode in ("before", "after"):
        env = {"other": "keep"}
        _apply_mode(mode, route_package="qsa-ordered-v2", environment=env)
        assert env == {"other": "keep"}, (mode, env)


@pytest.mark.parametrize("mode,expected", [("before", False), ("after", True)])
def test_route_scope_binds_the_runner_selection(mode, expected):
    owner = SimpleNamespace(qsa_ordered_decode_v2=None)
    env = {QSA_ORDERED_DECODE_ENV: "1", RETIRED_V2_ENV: "1"}
    with route_scope("qsa-ordered-v2", mode, owner, environment=env):
        assert owner.qsa_ordered_decode_v2 is expected
        # The v1 route and the retired switch are recorded, never forced; the
        # fused sibling is a separate diagnostic and is held off in both arms.
        assert env == {QSA_ORDERED_DECODE_ENV: "1", RETIRED_V2_ENV: "1",
                       FUSED_ENV: "0"}
    assert owner.qsa_ordered_decode_v2 is None
    assert env == {QSA_ORDERED_DECODE_ENV: "1", RETIRED_V2_ENV: "1"}


@pytest.mark.parametrize("mode", ["before", "after"])
@pytest.mark.parametrize("prior", ["1", None])
def test_the_campaign_arm_holds_the_fused_sibling_off(mode, prior):
    owner = SimpleNamespace(qsa_ordered_decode_v2=None)
    env = {QSA_ORDERED_DECODE_ENV: "1"}
    if prior is not None:
        env[FUSED_ENV] = prior
    with route_scope("qsa-ordered-v2", mode, owner, environment=env):
        # A fused kernel would otherwise launch while the arm reports the
        # three-pass v2 owner.
        assert env[FUSED_ENV] == "0"
    # The caller's value, including an absent key, comes back.
    assert env.get(FUSED_ENV) == prior


def test_the_control_record_records_the_fused_prior_and_effective():
    record = qsa_ordered_decode_v2_control_record(
        environment={QSA_ORDERED_DECODE_ENV: "1", FUSED_ENV: "1"})
    assert record["fused_sibling"]["environment"] == FUSED_ENV
    assert record["fused_sibling"]["effective"] == "0"
    assert record["fused_sibling"]["previous"] == "1"
    assert qsa_ordered_decode_v2_control_record(
        environment={})["fused_sibling"]["previous"] is None


@pytest.mark.parametrize("arm,expected", [("off", False), ("on", True)])
def test_the_drift_probe_arm_closes_the_fused_sibling(arm, expected):
    from scripts import qwen4exp_qsa_v2_drift_probe as probe

    owner = SimpleNamespace(qsa_ordered_decode_v2=True)
    env = {FUSED_ENV: "1"}
    with probe.decode_arm_scope(owner, arm, environment=env):
        assert owner.qsa_ordered_decode_v2 is expected
        assert env[FUSED_ENV] == "0"
    assert owner.qsa_ordered_decode_v2 is True
    assert env[FUSED_ENV] == "1"


def test_the_drift_probe_arm_rejects_an_unknown_arm():
    from scripts import qwen4exp_qsa_v2_drift_probe as probe

    with pytest.raises(ValueError, match="invalid decode arm"):
        with probe.decode_arm_scope(SimpleNamespace(qsa_ordered_decode_v2=False),
                                    "maybe", environment={}):
            pass


def test_the_drift_probe_records_the_fused_control():
    from scripts import qwen4exp_qsa_v2_drift_probe as probe

    assert probe.QSA_ORDERED_V2_FUSED_ENV == FUSED_ENV
    assert probe.CONTROL_FIELD == "qsa_ordered_decode_v2"
    assert probe.PREREQUISITE_ENV == QSA_ORDERED_DECODE_ENV


def test_route_scope_rejects_the_arm_without_its_runner():
    with pytest.raises(ValueError, match="needs the owning runner"):
        route_scope("qsa-ordered-v2", "after",
                    environment={QSA_ORDERED_DECODE_ENV: "1"})


@pytest.mark.parametrize("env", [
    {QSA_ORDERED_DECODE_ENV: "0"},
    {QSA_ORDERED_DECODE_ENV: "false"},
    {},
])
def test_route_scope_rejects_the_arm_without_the_v1_route(env):
    with pytest.raises(ValueError, match="ordered v1 route"):
        route_scope("qsa-ordered-v2", "after",
                    SimpleNamespace(qsa_ordered_decode_v2=False),
                    environment=env)


def test_state_gate_arm_writes_no_process_switch():
    env = {"other": "keep"}
    apply_state_gate_mode(SimpleNamespace(), "qsa-ordered-v2", "1", "",
                          environment=env)
    assert env == {"other": "keep"}


def test_the_control_record_names_the_field_and_the_prerequisite():
    record = qsa_ordered_decode_v2_control_record(
        SimpleNamespace(qsa_ordered_decode_v2=True),
        environment={QSA_ORDERED_DECODE_ENV: "1"})
    assert record["runner_field"] == "qsa_ordered_decode_v2"
    assert record["before"] is False
    assert record["after"] is True
    assert record["prerequisite"] == {QSA_ORDERED_DECODE_ENV: "1"}
    assert record["published_value"] is True
    # An arm without a runner records no published value rather than guessing.
    assert qsa_ordered_decode_v2_control_record()["published_value"] is None
