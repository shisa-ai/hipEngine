"""The MoE C1 native cycle's sampled accept route.

The Qwen3.6 MoE target decides a drafted chain on its device accept/commit
graph. That graph has a sampled variant, so a temperature/top-p/min-p request
can stay on MTP instead of being decided by argmax or handed to the
autoregressive path. This module pins the three boundaries that keep the route
honest:

* the adapter advertises the sampled mode only when the loaded plugin declares
  it (a capability declaration, never an artifact hash or model name);
* a servable sampled row produces a device accept plan and the cycle is driven
  with it, and the request's native sampler stream advances by exactly the
  committed tokens;
* a sampled row the cycle cannot decide (a logits processor, a bounded top-k, a
  forced-token or text-keyed constraint, logprob metadata) raises a named
  capability error and is contained, rather than silently deciding by argmax or
  falling back to autoregressive decoding.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.generation.qwen35_gguf_moe_mtp2 import (
    MoeSampledRouteUnsupportedError,
    Qwen35GGUFMoEMTP2Adapter,
)
from hipengine.generation.sampling import RowSamplingState
from hipengine.speculative.frontier import (
    SpecK0Class,
    SpecPlanReason,
    SpecRequestPlan,
    SpecTransactionMode,
)
from hipengine.speculative.provider import SpeculativeRequestSemantics


def _params(**overrides):
    values = {
        "temperature": 0.7,
        "top_p": 1.0,
        "top_k": 0,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "logit_bias": (),
        "suppress_token_ids": (),
        "min_tokens": 0,
        "eos_token_id": None,
        "ignore_eos": False,
        "seed": 5,
        "row_seeds": (),
        "stop_token_ids": (),
        "stop_token_sequences": (),
        "forced_tokens_pending": (),
        "forced_token_reason": None,
        "post_thinking_forced_tokens_pending": (),
        "post_thinking_forced_token_reason": None,
        "force_sequence_completion_token_sequences": (),
        "force_sequence_completion_reason": None,
        "json_object_close_forcing": False,
        "tool_call_constraint": None,
        "thinking_close_token_ids": (),
        "thinking_hard_token_cap": None,
        "thinking_soft_close_window": 0,
        "logprobs": False,
        "top_logprobs": 0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _declaration(*, modes=("greedy_fast", "sampled")):
    return SimpleNamespace(
        sampling_modes=tuple(modes),
        backends=(("hip_gfx1100", "gfx1100"),),
    )


def _generator(*, declaration=None):
    plugin = SimpleNamespace(
        speculative_mtp_serving_implementations=(
            () if declaration is None else (declaration,)
        ),
        speculative_mtp_serving_evidence=(),
    )
    return SimpleNamespace(
        backend="hip_gfx1100",
        target_arch="gfx1100",
        execution_profile=SimpleNamespace(value="strict"),
        model_plugin=plugin,
    )


def _plan(*, rid=7, budget=2):
    return SpecRequestPlan(
        operation_id="op:moe-sampled-test",
        cycle_id=1,
        request_ids=(rid,),
        resident_slots=(0,),
        candidate_counts=(budget,),
        reasons=(SpecPlanReason.SPECULATIVE_QUALIFIED,),
        k0_classes=(SpecK0Class.NOT_K0,),
        mode="verify_chain",
        capability_key="gguf_moe_mtp2_c1",
        provider_key="qwen_nextn_moe",
        target_transaction_mode=SpecTransactionMode.REVERSIBLE_JOURNAL,
        provider_transaction_mode=SpecTransactionMode.REVERSIBLE_JOURNAL,
        proposal_widths=(1,),
        target_row_decomposition=(budget + 1,),
        context_bucket_size=1024,
        execution_route="graph",
    )


def _adapter(*, declaration=None, sampling_mode="sampled"):
    adapter = object.__new__(Qwen35GGUFMoEMTP2Adapter)
    adapter.enabled = True
    adapter.candidate_budget = 2
    adapter.quant = "gguf_q4_k_m"
    adapter.target_verify_mode = "native"
    adapter._disabled_requests = set()
    adapter._states = {7: object()}
    adapter._sampling_mode_by_request = {7: sampling_mode}
    adapter._last_sampled_decline_reason = None
    adapter._transaction_sequence = 0
    adapter._assets = SimpleNamespace(
        rope_cos=np.zeros((8, 4), dtype=np.float32),
        rope_sin=np.zeros((8, 4), dtype=np.float32),
    )
    adapter.generator = _generator(declaration=declaration)
    return adapter


def _row(*, params=None, sampling_state=None, native_sampler=True):
    return SimpleNamespace(
        request_id=7,
        native_greedy=False,
        native_sampler=native_sampler,
        first_token_emitted=True,
        sampling_request=params,
        request=SimpleNamespace(max_tokens=64),
        sampling_state=sampling_state,
        mtp2_recoverable_failures=0,
        mtp2_failure_reasons=[],
        mtp2_proposal_ms=0.0,
        mtp2_target_ms=0.0,
        mtp2_provider_update_ms=0.0,
        mtp2_execution_routes=[],
    )


def _target(*, position=3, sampled_graph=True):
    target = SimpleNamespace(position=position)
    if sampled_graph:
        # The C1 cycle selects the captured sampled accept variant; the adapter
        # refuses a sampled row whose graph was not warmed at prefill.
        target._native_spec_b2_target_graph_n2_sampled = object()

    def run_native_spec_mtp_cycle(draft, context, **kwargs):
        target.last_kwargs = kwargs
        target.run_calls = getattr(target, "run_calls", 0) + 1
        target.position = position + 2
        return SimpleNamespace(
            draft_cache_len_after=kwargs["draft_cache_len"] + 2,
            output_token_ids=(11, 12),
            accepted_draft_tokens=1,
            end_position=position + 2,
            proposal_wall_ms=0.5,
            target_wall_ms=0.5,
            mtp_kv_commit_wall_ms=0.1,
        )

    target.run_native_spec_mtp_cycle = run_native_spec_mtp_cycle
    return target


def _wire_execute(adapter, *, row, target, budget=2):
    adapter.owner = SimpleNamespace(
        capacity=1,
        _row=lambda request_id: row,
        _flush_row_owner=lambda owner_row: None,
    )
    adapter._states = {
        7: SimpleNamespace(
            draft=object(),
            context=object(),
            key_cache=SimpleNamespace(),
            value_cache=SimpleNamespace(),
            cache_len=3,
            # The primed allocation this row writes into. A state that publishes
            # none cannot prove a cycle stays inside one, so the write guard
            # refuses it rather than assuming it is safe.
            draft_kv_rows=8,
        )
    }
    row.lease = SimpleNamespace(session=target)
    row.slot = SimpleNamespace(
        generated_ids=[10],
        prev_token=10,
        seq_position=target.position,
        native_decode_steps=0,
        done=False,
    )


def test_moe_capability_advertises_sampled_only_when_declared() -> None:
    """The sampled mode is a plugin capability declaration, not a measurement."""

    row = _row(
        params=_params(),
        sampling_state=RowSamplingState(seed=5, request_id=7),
    )
    target = SimpleNamespace(
        position=6,
        target_layout=SimpleNamespace(max_sequence_length=1024),
        kv_storage_dtype="bf16",
        _native_spec_b2_target_graph_n2_sampled=object(),
    )
    row.lease = SimpleNamespace(session=target)
    row.slot = object()
    semantics = (SpeculativeRequestSemantics(7, "sampled", "verify_chain", 6, 8),)

    qualified = _adapter(declaration=_declaration())
    qualified.owner = SimpleNamespace(capacity=1, _row=lambda request_id: row)
    capability = qualified.capability(semantics)
    assert capability is not None
    assert capability.supported_sampling_modes == ("greedy", "sampled")

    unqualified = _adapter(declaration=None)
    unqualified.owner = SimpleNamespace(capacity=1, _row=lambda request_id: row)
    # No declaration: the row is not native-greedy and the sampled route is
    # closed, so the capability refuses it rather than claiming a mode it
    # cannot execute.
    assert unqualified.capability(semantics) is None


def test_moe_device_sampled_plan_serves_the_native_sampler_subset() -> None:
    adapter = _adapter(declaration=_declaration())
    target = SimpleNamespace(
        native_sampler_algorithm="sorted",
        _native_spec_b2_target_graph_n2_sampled=object(),
    )
    state = RowSamplingState(seed=5, request_id=7, step_index=3)
    row = _row(params=_params(), sampling_state=state)
    adapter.owner = SimpleNamespace(capacity=1, _row=lambda request_id: row)

    plan = adapter._device_sampled_accept_plan(7, target, 2)

    assert plan is not None
    assert plan.rows == 3
    assert plan.seed == 5
    assert plan.step_index == 3
    assert adapter._last_sampled_decline_reason is None


@pytest.mark.parametrize(
    "params,reason",
    [
        (_params(repetition_penalty=1.2), "logits processors"),
        (_params(top_k=4), "bounded top-k"),
        (_params(logprobs=True), "logprobs requested"),
    ],
)
def test_moe_device_sampled_plan_declines_with_a_named_reason(params, reason) -> None:
    adapter = _adapter(declaration=_declaration())
    target = SimpleNamespace(native_sampler_algorithm="sorted")
    state = RowSamplingState(seed=5, request_id=7)
    row = _row(params=params, sampling_state=state)
    adapter.owner = SimpleNamespace(capacity=1, _row=lambda request_id: row)

    assert adapter._device_sampled_accept_plan(7, target, 2) is None
    assert reason in str(adapter._last_sampled_decline_reason)


def test_moe_sampled_cycle_drives_the_device_plan_and_advances_the_stream() -> None:
    adapter = _adapter(declaration=_declaration())
    state = RowSamplingState(seed=5, request_id=7)
    row = _row(params=_params(), sampling_state=state)
    target = _target(position=3)
    _wire_execute(adapter, row=row, target=target)

    result = adapter.execute_cycle(_plan(), commit=True)

    assert target.last_kwargs["sampled_accept"] is True
    assert target.last_kwargs["sampled_accept_state"] is not None
    assert target.last_kwargs["sampled_accept_state"].rows == 3
    assert row.slot.generated_ids == [10, 11, 12]
    # Two committed tokens advance the native sampler stream by exactly two
    # draws, the same count the autoregressive route would consume.
    assert state.step_index == 2
    assert result.stage.name == "COMMITTED"


def test_moe_greedy_cycle_leaves_the_sampled_plan_off() -> None:
    adapter = _adapter(declaration=_declaration(), sampling_mode="greedy")
    row = _row(params=_params(temperature=0.0), sampling_state=None)
    row.native_greedy = True
    target = _target(position=3)
    _wire_execute(adapter, row=row, target=target)

    adapter.execute_cycle(_plan(), commit=True)

    assert target.last_kwargs["sampled_accept"] is False
    assert target.last_kwargs["sampled_accept_state"] is None


def test_moe_sampled_cycle_fails_loudly_for_an_unservable_request() -> None:
    adapter = _adapter(declaration=_declaration())
    state = RowSamplingState(seed=5, request_id=7)
    row = _row(params=_params(repetition_penalty=1.2), sampling_state=state)
    target = _target(position=3)
    _wire_execute(adapter, row=row, target=target)

    with pytest.raises(MoeSampledRouteUnsupportedError) as excinfo:
        adapter.execute_cycle(_plan(), commit=True)

    assert "logits processors" in str(excinfo.value)
    # The refusal is resolved before the cycle runs, so no device work happened.
    assert getattr(target, "run_calls", 0) == 0


def test_moe_sampled_unsupported_failure_is_contained_not_ar() -> None:
    """A sampled refusal is retired, not silently downgraded to AR."""

    adapter = _adapter(declaration=_declaration())
    plan = _plan()
    error = MoeSampledRouteUnsupportedError("no host logits rows")

    assert adapter.recover_cycle_failure(plan, error) is False
    assert adapter.containment_evidence(plan, error) == ("none", (7,))


def _sampled_capability_adapter(*, params, target, sampling_mode="sampled"):
    adapter = _adapter(declaration=_declaration(), sampling_mode=sampling_mode)
    state = RowSamplingState(seed=5, request_id=7)
    row = _row(params=params, sampling_state=state)
    row.lease = SimpleNamespace(session=target)
    row.slot = object()
    adapter.owner = SimpleNamespace(capacity=1, _row=lambda request_id: row)
    return adapter


def _sampled_target(*, sampled_graph=True):
    target = SimpleNamespace(
        position=6,
        native_sampler_algorithm="sorted",
        target_layout=SimpleNamespace(max_sequence_length=1024),
        kv_storage_dtype="bf16",
    )
    if sampled_graph:
        target._native_spec_b2_target_graph_n2_sampled = object()
    return target


def test_moe_capability_refuses_a_sampled_row_the_device_accept_cannot_decide() -> None:
    """A request the C1 device accept cannot decide is routed to AR, not admitted."""

    semantics = (SpeculativeRequestSemantics(7, "sampled", "verify_chain", 6, 8),)

    # A logits processor (repetition penalty) has no device decider, so the
    # capability declines the row and the planner routes it to autoregressive
    # decoding instead of admitting it and failing at execute time.
    assert (
        _sampled_capability_adapter(
            params=_params(repetition_penalty=1.2), target=_sampled_target()
        ).capability(semantics)
        is None
    )
    # A bounded top-k is the same class of refusal.
    assert (
        _sampled_capability_adapter(
            params=_params(top_k=4), target=_sampled_target()
        ).capability(semantics)
        is None
    )
    # The unconstrained sampled row is still admitted.
    capability = _sampled_capability_adapter(
        params=_params(), target=_sampled_target()
    ).capability(semantics)
    assert capability is not None
    assert capability.supported_sampling_modes == ("greedy", "sampled")

    # ``min_tokens`` is a sampler processor the device accept cannot apply, so
    # it is refused by name. The other finish-rule fields (EOS, stop ids,
    # ``ignore_eos``) are applied by the service boundary's finish rule and do
    # not block admission.
    assert (
        _sampled_capability_adapter(
            params=_params(min_tokens=4), target=_sampled_target()
        ).capability(semantics)
        is None
    )
    assert (
        _sampled_capability_adapter(
            params=_params(ignore_eos=True, stop_token_ids=(2, 3)),
            target=_sampled_target(),
        ).capability(semantics)
        is not None
    )


def test_moe_capability_refuses_a_sampled_row_without_a_warmed_graph() -> None:
    """A sampled row whose graph was not captured at prefill is not admitted."""

    semantics = (SpeculativeRequestSemantics(7, "sampled", "verify_chain", 6, 8),)
    adapter = _sampled_capability_adapter(
        params=_params(), target=_sampled_target(sampled_graph=False)
    )

    assert adapter.capability(semantics) is None


def test_moe_sampled_accept_candidate_matches_the_warmup_predicate() -> None:
    """The prefill warmup targets exactly the rows the device accept can serve."""

    adapter = _adapter(declaration=_declaration())
    row = _row(
        params=_params(), sampling_state=RowSamplingState(seed=5, request_id=7)
    )
    adapter.owner = SimpleNamespace(capacity=1, _row=lambda request_id: row)
    adapter.generator.native_sampler_algorithm = "sorted"

    assert adapter._sampled_accept_candidate(7) is True

    # A processor-bearing, bounded-top-k, or zero-temperature request never
    # needs the variant.
    row.sampling_request = _params(repetition_penalty=1.2)
    assert adapter._sampled_accept_candidate(7) is False
    row.sampling_request = _params(top_k=4)
    assert adapter._sampled_accept_candidate(7) is False
    row.sampling_request = _params(temperature=0.0)
    assert adapter._sampled_accept_candidate(7) is False

    # An undeclared plugin never warms the variant.
    row.sampling_request = _params()
    adapter.generator.model_plugin = SimpleNamespace(
        speculative_mtp_serving_implementations=(),
        speculative_mtp_serving_evidence=(),
    )
    assert adapter._sampled_accept_candidate(7) is False


def test_moe_sampled_graph_failure_is_a_named_contained_refusal() -> None:
    """An unavailable sampled graph is named and contained, not silently AR."""

    from hipengine.runtime.gguf_native_spec_cycle import (
        NativeSpecTargetGraphUnsupportedError,
    )

    adapter = _adapter(declaration=_declaration())
    state = RowSamplingState(seed=5, request_id=7)
    row = _row(params=_params(), sampling_state=state)
    target = _target(position=3)

    def fail_cycle(*args, **kwargs):
        raise NativeSpecTargetGraphUnsupportedError("sampled graph capture failed")

    target.run_native_spec_mtp_cycle = fail_cycle
    _wire_execute(adapter, row=row, target=target)

    with pytest.raises(MoeSampledRouteUnsupportedError) as excinfo:
        adapter.execute_cycle(_plan(), commit=True)

    assert "sampled target graph unavailable" in str(excinfo.value)
    assert adapter.recover_cycle_failure(_plan(), excinfo.value) is False
    assert adapter.containment_evidence(_plan(), excinfo.value) == ("none", (7,))


def test_moe_greedy_graph_failure_keeps_its_recovery_path() -> None:
    """A greedy graph miss still recovers onto the autoregressive path."""

    from hipengine.runtime.gguf_native_spec_cycle import (
        NativeSpecTargetGraphUnsupportedError,
    )

    adapter = _adapter(declaration=_declaration(), sampling_mode="greedy")
    row = _row(params=_params(temperature=0.0), sampling_state=None)
    row.native_greedy = True
    target = _target(position=3)

    def fail_cycle(*args, **kwargs):
        raise NativeSpecTargetGraphUnsupportedError("greedy graph miss")

    target.run_native_spec_mtp_cycle = fail_cycle
    _wire_execute(adapter, row=row, target=target)

    with pytest.raises(NativeSpecTargetGraphUnsupportedError):
        adapter.execute_cycle(_plan(), commit=True)

    # The greedy row is not a sampled refusal: its recovery is unchanged.
    assert (
        adapter.recover_cycle_failure(
            _plan(), NativeSpecTargetGraphUnsupportedError("greedy graph miss")
        )
        is True
    )


def test_moe_staged_c2_refuses_a_sampled_row_by_name() -> None:
    """A width-2 MoE target refuses sampled rows instead of argmax-deciding them."""

    adapter = _adapter(declaration=_declaration())
    adapter._active_claims = None
    adapter.owner = SimpleNamespace(capacity=2, _row=lambda request_id: None)
    plan = _plan(rid=7)
    plan = SpecRequestPlan(
        operation_id=plan.operation_id,
        cycle_id=plan.cycle_id,
        request_ids=(7, 8),
        resident_slots=(0, 1),
        candidate_counts=(2, 2),
        reasons=(SpecPlanReason.SPECULATIVE_QUALIFIED,) * 2,
        k0_classes=(SpecK0Class.NOT_K0,) * 2,
        mode="verify_chain",
        capability_key="gguf_moe_mtp2_c2",
        provider_key="qwen_nextn_moe",
        target_transaction_mode=SpecTransactionMode.REVERSIBLE_JOURNAL,
        provider_transaction_mode=SpecTransactionMode.REVERSIBLE_JOURNAL,
        proposal_widths=(2,),
        target_row_decomposition=(3, 3),
        context_bucket_size=1024,
        execution_route="graph",
    )
    adapter._sampling_modes()[8] = "sampled"
    frontier = SimpleNamespace(target_batch=object())

    with pytest.raises(NotImplementedError, match="uses execute_cycle"):
        adapter.execute_target_frontier(
            plan,
            frontier,
            None,
            commit=True,
            cancelled_request_ids=lambda: (),
        )


# --------------------------------------------- sampled availability at c=N


def test_moe_plugin_admits_explicit_sampled_through_c2_and_refuses_wider() -> None:
    """Sampled MTP is contractually available at the server's c1-c2 widths.

    The declaration bounds the offered width, so an explicit sampled request
    resolves at c1 and c2 and a wider one is refused by name instead of being
    silently clamped or downgraded to greedy.
    """

    from hipengine.models.qwen35 import Qwen35MoeGGUFModel
    from hipengine.speculative.serving import SpeculativeMTPServingKey

    plugin = Qwen35MoeGGUFModel()

    def key(rows: int) -> SpeculativeMTPServingKey:
        return SpeculativeMTPServingKey(
            artifact_sha256=None,
            artifact_size_bytes=None,
            content_verified=False,
            backend="hip_gfx1100",
            target_arch="gfx1100",
            weight_quant="gguf_q4_k_m",
            kv_storage="bf16",
            kv_layout="uniform",
            realized_group_rows=rows,
            resident_capacity=max(rows, 2),
            candidate_budget=2,
            sampling_mode="sampled",
            memory_fit=True,
        )

    for rows in (1, 2):
        decision = plugin.resolve_speculative_mtp_serving_plan(
            key=key(rows), request_mode="explicit"
        )
        assert decision.admitted, rows
        payload = decision.as_dict()
        assert payload["admission_basis"] == "implementation"
        assert payload["implementation_key"] == "gguf_moe_bf16_gfx1100_native_chain"

    wider = plugin.resolve_speculative_mtp_serving_plan(
        key=key(3), request_mode="explicit"
    )
    assert not wider.admitted
    assert wider.as_dict()["reason"] == "moe_group_above_offered_width"


def test_moe_c2_groups_route_to_the_dense_adapter_with_the_packed_sampler() -> None:
    """Production c=N MoE execution runs the dense adapter's sampled decider.

    ``create_qwen35_gguf_moe_mtp2_adapter`` resolves the C1 native-cycle adapter
    only at capacity 1; wider capacities get the dense adapter, whose packed
    target samples each row exactly before commit
    (``_sample_packed_target_rows``). This pins the routing so the c=N sampled
    story cannot regress into the C1-only adapter unnoticed.
    """

    from hipengine.generation.qwen35_gguf_moe_mtp2 import (
        create_qwen35_gguf_moe_mtp2_adapter,
    )
    from hipengine.generation.qwen35_gguf_mtp2 import Qwen35GGUFMTP2Adapter

    c1_owner = SimpleNamespace(capacity=1, generator=None)
    c1 = create_qwen35_gguf_moe_mtp2_adapter(
        c1_owner,
        enabled=True,
        target_verify_mode="native",
        candidate_budget=2,
    )
    assert type(c1) is Qwen35GGUFMoEMTP2Adapter

    c2_owner = SimpleNamespace(capacity=2, generator=None)
    c2 = create_qwen35_gguf_moe_mtp2_adapter(
        c2_owner,
        enabled=True,
        target_verify_mode="native",
        candidate_budget=2,
    )
    assert type(c2) is Qwen35GGUFMTP2Adapter


def test_dense_adapter_qualifies_the_moe_sampled_route_only_where_declared() -> None:
    """The packed lane's sampled qualification follows the plugin declaration.

    The dense adapter's sampled qualification is plugin- and backend-scoped:
    the MoE declaration covers sampled on both supported gfx backends.
    """

    from hipengine.generation.qwen35_gguf_mtp2 import Qwen35GGUFMTP2Adapter
    from hipengine.models.qwen35 import Qwen35MoeGGUFModel

    adapter = Qwen35GGUFMTP2Adapter.__new__(Qwen35GGUFMTP2Adapter)
    adapter.quant = "gguf_q4_k_m"

    adapter.generator = SimpleNamespace(
        backend="hip_gfx1100",
        target_arch="gfx1100",
        model_plugin=Qwen35MoeGGUFModel(),
    )
    assert adapter._sampled_route_qualified() is True

    adapter.generator = SimpleNamespace(
        backend="hip_gfx1151",
        target_arch="gfx1151",
        model_plugin=Qwen35MoeGGUFModel(),
    )
    assert adapter._sampled_route_qualified() is True


def test_packed_batch_refuses_sampled_rows_without_device_results_by_name() -> None:
    """A sampled row on a host-batch lane is a named refusal, never argmax.

    The packed target can only decide a sampled row from resident device
    results; the host-visible batch lane has no per-row logits span to draw
    from. The guard must stay a named RuntimeError (contained by the shared
    recovery boundary) rather than letting the greedy argmax decide the row.
    """

    import inspect

    from hipengine.generation.qwen35_gguf_mtp2 import Qwen35GGUFMTP2Adapter

    source = inspect.getsource(Qwen35GGUFMTP2Adapter._execute_target_frontier_batch)
    assert "any(self._sampled_route_request(rid) for rid in ids)" in source
    assert (
        "sampled physical verification requires resident device results" in source
    )
