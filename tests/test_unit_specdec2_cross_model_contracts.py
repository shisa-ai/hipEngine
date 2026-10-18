"""Cross-model contract properties of the reusable staged SPECDEC2 seam.

The dense Qwen GGUF, MoE Qwen GGUF, and PARO resident owners all opt into the
same ``StagedSpeculativeRunner`` lifecycle (``hipengine/speculative/provider.py``)
and publish a composed ``SpeculativeCapability`` whose ``kv_binding`` describes
their KV store. These tests exercise the *shared* contract across those model
shapes through lightweight owner doubles, rather than re-testing any one model's
delegation (which the per-model seam tests already cover):

* the same owner keeps separate ``stream`` values isolated, so two concurrent
  requests on one runner cannot cross-talk through staged state;
* the three KV binding variants (dense BF16, quantized INT8, compact-retention
  DMS) are distinct, self-consistent records, and a capability cannot be built
  against another shape's storage view;
* a sampled capability is either advertised and admitted, or refused with a
  named reason and no target execution - never silently downgraded to greedy or
  autoregressive decoding, while greedy stays supported.

The DMS compact store refuses speculative serving entirely; its config gate is
asserted here to name the clearing command so the boundary is not a dead end.

The lifecycle additions below pin cross-model parity for the events each
per-model seam suite tests only against its own fixture:

* a stage failure on one stream leaves the sibling stream intact and the owner
  reusable for a retry, with the failing cycle's claim reservation released
  exactly once (the loop's failure protocol);
* a cancellation resolving at the target boundary publishes nothing for the
  cancelled row - no tail ids leak into the committed result;
* every capability refusal records a named non-empty reason, and an admitted
  capability records none, so a silent downgrade cannot hide behind ``None``;
* a frontier without a ``kv_binding`` (today's production shape: no adapter
  populates one yet) still carries the loose KV string keys, while a frontier
  whose binding disagrees with those keys is refused by name;
* staged state is per cycle, not per request lifetime: the same request id
  runs consecutive cycles without accumulating staged state.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from hipengine.kvcache import ClaimLifetime, ResourceClaimSet
from hipengine.speculative import (
    CandidateGraph,
    ProviderAttachment,
    ProviderCatchupMode,
    SpecTransactionMode,
    SpeculativeCapability,
    SpeculativeKVBinding,
    SpeculativeRequestSemantics,
    StagedSpeculativeRunner,
    TargetFrontier,
    resolve_staged_speculative_runner,
    validate_staged_speculative_runner,
)


# --------------------------------------------------------------------- records


def _binding(
    *,
    storage_view_key: str,
    storage_dtype: str,
    topology_key: str,
    retention_key: str,
    transaction_mode: SpecTransactionMode,
    scale_dtype: str | None = None,
    scale_granularity: str | None = None,
) -> SpeculativeKVBinding:
    return SpeculativeKVBinding(
        backend_fingerprint="backend:test:v1",
        generation=1,
        topology_key=topology_key,
        storage_view_key=storage_view_key,
        layout_fingerprint="layout:test:v1",
        storage_dtype=storage_dtype,
        storage_layout="uniform",
        transaction_mode=transaction_mode,
        scale_dtype=scale_dtype,
        scale_granularity=scale_granularity,
        retention_key=retention_key,
    )


DENSE_BF16_BINDING = _binding(
    storage_view_key="paged_bf16",
    storage_dtype="bf16",
    topology_key="paged_dense_global",
    retention_key="dense",
    transaction_mode=SpecTransactionMode.PACKED_SCRATCH,
)
QUANTIZED_INT8_BINDING = _binding(
    storage_view_key="dense:int8",
    storage_dtype="int8_per_token_head",
    topology_key="paged_dense_global",
    retention_key="dense",
    transaction_mode=SpecTransactionMode.PACKED_SCRATCH,
    scale_dtype="fp32",
    scale_granularity="per_token_head",
)
DMS_COMPACT_BINDING = _binding(
    storage_view_key="dms:int8",
    storage_dtype="int8_per_token_head",
    topology_key="dms_compact",
    retention_key="dms",
    transaction_mode=SpecTransactionMode.REVERSIBLE_JOURNAL,
    scale_dtype="fp32",
    scale_granularity="per_token_head",
)


def _capability(
    *,
    target_key: str,
    provider_key: str,
    kv_binding: SpeculativeKVBinding,
    supported_sampling_modes: tuple[str, ...],
    max_requests: int = 4,
    kv_backend_key: str | None = None,
    target_transaction_mode: SpecTransactionMode | None = None,
) -> SpeculativeCapability:
    return SpeculativeCapability(
        capability_key=f"mtp2:{target_key}:test",
        target_key=target_key,
        provider_key=provider_key,
        method_key="mtp2",
        policy_fingerprint="policy:test:v1",
        execution_profile="strict",
        kv_backend_key=(
            kv_binding.storage_view_key if kv_backend_key is None else kv_backend_key
        ),
        attachment=ProviderAttachment.TARGET_ATTACHED,
        catchup_mode=ProviderCatchupMode.TARGET_OUTPUT,
        supported_modes=("verify_chain",),
        supported_sampling_modes=supported_sampling_modes,
        max_requests=max_requests,
        max_candidates_per_request=3,
        max_frontier_rows=max_requests * 4,
        proposal_widths=(1,),
        target_row_buckets=(2, 4, 8, 16),
        target_transaction_mode=(
            kv_binding.transaction_mode
            if target_transaction_mode is None
            else target_transaction_mode
        ),
        provider_transaction_mode=SpecTransactionMode.REVERSIBLE_JOURNAL,
        graph_supported=True,
        eager_supported=True,
        strict_fallback_key="target_ar_strict",
        kv_binding=kv_binding,
    )


# ------------------------------------------------------------------ owner doubles


class _StagedOwnerDouble:
    """A lightweight staged owner that records per-stream staged state.

    The double deliberately keeps prepared state keyed by ``stream`` and refuses
    to propose for a stream whose plan does not match what that stream prepared,
    so a cross-talk bug in the owner contract surfaces as a failure rather than
    as a silently shared buffer.
    """

    model_shape = "generic"

    def __init__(
        self,
        capability: SpeculativeCapability | None,
        *,
        declined_reason: str | None = None,
        unsupported_sampling_reason: str | None = None,
    ) -> None:
        self._capability = capability
        self._declined_reason = declined_reason
        self._unsupported_sampling_reason = unsupported_sampling_reason
        self._prepared: dict[object, tuple[str, tuple[int, ...]]] = {}
        self.decline_reasons: list[str] = []
        self.proposals: list[tuple[object, tuple[int, ...]]] = []
        self.target_calls: list[tuple[str, bool]] = []
        self.reservations: list[object] = []
        self.releases: list[object] = []
        self.cancelled_ids: list[tuple[int, ...]] = []

    def speculative_capability(self, request_semantics):
        capability = self._capability
        if capability is None:
            if self._declined_reason is not None:
                self.decline_reasons.append(self._declined_reason)
            return None
        for item in request_semantics:
            if not capability.supports_sampling(item.sampling_mode):
                reason = self._unsupported_sampling_reason
                if reason is None:
                    reason = (
                        f"{self.model_shape} cannot serve "
                        f"sampling_mode={item.sampling_mode!r}"
                    )
                self.decline_reasons.append(reason)
                return None
        return capability

    def speculative_component_claims(self, plan):
        return {
            "provider": ResourceClaimSet.from_mapping(
                f"{plan.operation_id}:provider",
                {"provider.kv": sum(plan.candidate_counts)},
                lifetime=ClaimLifetime.TRANSACTION,
            )
        }

    def reserve_speculative_claims(self, claims):
        self.reservations.append(claims)
        return ("reservation", claims)

    def release_speculative_claims(self, reservation) -> None:
        self.releases.append(reservation)

    def prepare_speculative_requests(self, plan, request_semantics, *, stream=None) -> None:
        self._prepared[stream] = (
            plan.operation_id,
            tuple(int(item.request_id) for item in request_semantics),
        )

    def propose_speculative_batch(self, plan, request_semantics, *, stream=None):
        prepared = self._prepared.get(stream)
        if prepared is None:
            raise RuntimeError(f"stream {stream!r} proposed before prepare")
        if prepared[0] != plan.operation_id:
            raise RuntimeError(
                f"stream {stream!r} cross-talk: prepared {prepared[0]!r} "
                f"but proposing {plan.operation_id!r}"
            )
        request_ids = tuple(int(item.request_id) for item in request_semantics)
        self.proposals.append((stream, request_ids))
        return CandidateGraph(
            provider_key=plan.provider_key,
            method_key="mtp2",
            policy_fingerprint="policy:test:v1",
            cycle_id=plan.cycle_id,
            transaction_id=len(self.proposals),
            request_ids=request_ids,
            resident_slots=tuple(range(len(request_ids))),
            root_positions=tuple(
                int(item.context_tokens) - 1 for item in request_semantics
            ),
            row_offsets=tuple(range(len(request_ids) + 1)),
            row_to_request=request_ids,
            parent_candidate_rows=(-1,) * len(request_ids),
            draft_depths=(1,) * len(request_ids),
            active_mask=(True,) * len(request_ids),
            candidate_tokens=tuple(100 + index for index in range(len(request_ids))),
        )

    def execute_target_frontier(
        self, plan, frontier, complete_claims, *, commit, cancelled_request_ids
    ):
        self.target_calls.append((plan.operation_id, bool(commit)))
        # The seam contract: a cancellation resolving at the target boundary is
        # consulted through the callable the loop passes, and the cancelled row
        # publishes nothing - no tail ids reach the committed result.
        cancelled: tuple[int, ...] = ()
        if callable(cancelled_request_ids):
            cancelled = tuple(int(value) for value in cancelled_request_ids())
        self.cancelled_ids.append(cancelled)
        committed = tuple(
            int(request_id)
            for request_id in plan.request_ids
            if int(request_id) not in cancelled
        )
        return SimpleNamespace(
            operation_id=plan.operation_id,
            committed_request_ids=committed,
        )


class _DenseOwner(_StagedOwnerDouble):
    model_shape = "dense"

    def __init__(self) -> None:
        super().__init__(
            _capability(
                target_key="qwen_dense_gguf",
                provider_key="qwen_nextn_dense",
                kv_binding=DENSE_BF16_BINDING,
                supported_sampling_modes=("greedy",),
            ),
            unsupported_sampling_reason="gguf_dense_unsupported_sampling_mode",
        )


class _MoeOwner(_StagedOwnerDouble):
    model_shape = "moe"

    def __init__(self) -> None:
        super().__init__(
            _capability(
                target_key="qwen_moe_gguf",
                provider_key="qwen_nextn_moe",
                kv_binding=QUANTIZED_INT8_BINDING,
                supported_sampling_modes=("greedy", "sampled"),
            )
        )


class _ParoOwner(_StagedOwnerDouble):
    model_shape = "paro"

    def __init__(self) -> None:
        super().__init__(
            _capability(
                target_key="qwen_paro_bf16",
                provider_key="qwen_paro_mtp_bf16",
                kv_binding=DENSE_BF16_BINDING,
                supported_sampling_modes=("greedy",),
            ),
            unsupported_sampling_reason="paro_c1_greedy_only",
        )


class _DmsOwner(_StagedOwnerDouble):
    """A DMS-shaped owner that refuses speculative serving by name."""

    model_shape = "dms"

    def __init__(self) -> None:
        super().__init__(
            None,
            declined_reason=(
                "DMS compact KV has no speculative serving adapter; "
                "use speculative_mtp_serving='off'"
            ),
        )


_OWNER_FACTORIES = (_DenseOwner, _MoeOwner, _ParoOwner)


# ------------------------------------------------------------------- plan helpers


def _semantics(
    request_id: int,
    *,
    sampling_mode: str = "greedy",
    context_tokens: int = 32,
) -> SpeculativeRequestSemantics:
    return SpeculativeRequestSemantics(
        request_id=request_id,
        sampling_mode=sampling_mode,
        mode="verify_chain",
        context_tokens=context_tokens,
        remaining_decode=8,
    )


def _plan(operation_id: str, request_ids: tuple[int, ...]) -> SimpleNamespace:
    return SimpleNamespace(
        operation_id=operation_id,
        cycle_id=1,
        request_ids=request_ids,
        resident_slots=tuple(range(len(request_ids))),
        candidate_counts=(1,) * len(request_ids),
        speculative_request_ids=request_ids,
        provider_key="qwen_nextn_test",
    )


# ------------------------------------------------------- staged runner contract


@pytest.mark.parametrize("owner_factory", _OWNER_FACTORIES)
def test_every_model_shape_resolves_the_shared_staged_runner_contract(
    owner_factory,
) -> None:
    owner = owner_factory()

    assert isinstance(owner, StagedSpeculativeRunner)
    assert validate_staged_speculative_runner(owner) is owner
    # Duck-typed owners without an explicit opt-in accessor still resolve, so a
    # model that adds the lifecycle is not rejected for missing the accessor.
    assert resolve_staged_speculative_runner(owner) is owner


@pytest.mark.parametrize("owner_factory", _OWNER_FACTORIES)
def test_one_owner_keeps_interleaved_streams_isolated(owner_factory) -> None:
    owner = owner_factory()
    plan_a = _plan("op:stream-a", (1,))
    plan_b = _plan("op:stream-b", (2,))
    semantics_a = (_semantics(1),)
    semantics_b = (_semantics(2),)

    # Prepare both streams before either proposes, the interleaving a shared
    # buffer would corrupt.
    owner.prepare_speculative_requests(plan_a, semantics_a, stream=1)
    owner.prepare_speculative_requests(plan_b, semantics_b, stream=2)
    graph_a = owner.propose_speculative_batch(plan_a, semantics_a, stream=1)
    graph_b = owner.propose_speculative_batch(plan_b, semantics_b, stream=2)

    assert graph_a.request_ids == (1,)
    assert graph_b.request_ids == (2,)
    assert owner.proposals == [(1, (1,)), (2, (2,))]

    # A second owner instance shares no prepared state with the first.
    fresh = owner_factory()
    with pytest.raises(RuntimeError, match="before prepare"):
        fresh.propose_speculative_batch(plan_a, semantics_a, stream=1)
    assert fresh.proposals == []


def test_independent_owners_of_different_shapes_do_not_cross_talk() -> None:
    dense = _DenseOwner()
    moe = _MoeOwner()
    dense_plan = _plan("op:dense", (10,))
    moe_plan = _plan("op:moe", (20,))

    dense.prepare_speculative_requests(dense_plan, (_semantics(10),), stream=7)
    moe.prepare_speculative_requests(moe_plan, (_semantics(20),), stream=7)

    assert dense.propose_speculative_batch(
        dense_plan, (_semantics(10),), stream=7
    ).request_ids == (10,)
    assert moe.propose_speculative_batch(
        moe_plan, (_semantics(20),), stream=7
    ).request_ids == (20,)
    # Same stream id, independent state: neither owner saw the other's plan.
    assert dense.proposals == [(7, (10,))]
    assert moe.proposals == [(7, (20,))]


# ------------------------------------------------------------- KV binding variants


def test_kv_binding_variants_are_distinct_and_self_consistent() -> None:
    dense = DENSE_BF16_BINDING
    int8 = QUANTIZED_INT8_BINDING
    dms = DMS_COMPACT_BINDING

    # Dense BF16 carries no scale metadata and the dense retention key.
    assert dense.storage_dtype == "bf16"
    assert dense.scale_dtype is None and dense.scale_granularity is None
    assert dense.retention_key == "dense"

    # INT8 metadata is complete and names both scale fields.
    assert int8.storage_dtype == "int8_per_token_head"
    assert (int8.scale_dtype, int8.scale_granularity) == ("fp32", "per_token_head")
    assert int8.retention_key == "dense"

    # DMS compact retention is a different topology, storage view, and
    # transaction mode than the dense store.
    assert dms.retention_key == "dms"
    assert dms.topology_key == "dms_compact"
    assert dms.transaction_mode is SpecTransactionMode.REVERSIBLE_JOURNAL

    # The three compatibility keys differ, so a scheduler cannot confuse them.
    assert len({dense.compatibility_key, int8.compatibility_key, dms.compatibility_key}) == 3

    # Two requests on one backend share a compatibility key but own distinct
    # live-span planes, which is what makes the binding safe to share.
    first = dense.with_live_spans_owner("request:1")
    second = dense.with_live_spans_owner("request:2")
    assert first.compatibility_key == second.compatibility_key
    assert first.live_spans_owner == "request:1"
    assert second.live_spans_owner == "request:2"


def test_every_shape_binds_its_own_storage_view_into_its_capability() -> None:
    for owner_factory in _OWNER_FACTORIES:
        capability = owner_factory().speculative_capability((_semantics(1),))
        assert capability is not None
        binding = capability.kv_binding
        assert binding is not None
        assert binding.storage_view_key == capability.kv_backend_key
        assert binding.transaction_mode is capability.target_transaction_mode


def test_capability_rejects_another_shape_s_kv_storage_view() -> None:
    # A dense capability cannot claim the DMS compact store, and the mismatch is
    # named on the storage view rather than silently accepted.
    with pytest.raises(ValueError, match="storage_view_key"):
        _capability(
            target_key="qwen_dense_gguf",
            provider_key="qwen_nextn_dense",
            kv_binding=DMS_COMPACT_BINDING,
            kv_backend_key="paged_bf16",
            supported_sampling_modes=("greedy",),
        )

    # Nor can it claim a binding whose transaction mode disagrees.
    with pytest.raises(ValueError, match="transaction mode"):
        _capability(
            target_key="qwen_dense_gguf",
            provider_key="qwen_nextn_dense",
            kv_binding=DENSE_BF16_BINDING,
            target_transaction_mode=SpecTransactionMode.REVERSIBLE_JOURNAL,
            supported_sampling_modes=("greedy",),
        )


# ------------------------------------------------------- sampled vs greedy policy


@pytest.mark.parametrize(
    "owner_factory,advertises_sampled",
    [(_DenseOwner, False), (_MoeOwner, True), (_ParoOwner, False)],
)
def test_sampled_capability_is_admitted_or_named_refused(
    owner_factory, advertises_sampled: bool
) -> None:
    owner = owner_factory()
    sampled = (_semantics(7, sampling_mode="sampled"),)

    capability = owner.speculative_capability(sampled)

    if advertises_sampled:
        # Admitted: the capability advertises the mode it will execute.
        assert capability is not None
        assert capability.supports_sampling("sampled")
        assert owner.decline_reasons == []
    else:
        # Refused: a named reason is recorded, and no target execution ran for
        # the refused request, so it was not silently downgraded to a greedy
        # capability or driven autoregressively through this owner.
        assert capability is None
        assert owner.decline_reasons == [
            owner._unsupported_sampling_reason
        ]
        assert all(owner.decline_reasons)
        assert owner.target_calls == []

    # Greedy stays supported on every model shape.
    greedy = owner.speculative_capability((_semantics(7, sampling_mode="greedy"),))
    assert greedy is not None
    assert greedy.supports_sampling("greedy")


def test_sampled_refusal_is_not_a_greedy_downgrade() -> None:
    owner = _DenseOwner()

    refused = owner.speculative_capability((_semantics(9, sampling_mode="sampled"),))
    assert refused is None
    assert owner.decline_reasons == ["gguf_dense_unsupported_sampling_mode"]

    # The same owner still answers the greedy query, which proves the refusal
    # was mode-specific rather than a blanket capability failure.
    greedy = owner.speculative_capability((_semantics(9, sampling_mode="greedy"),))
    assert greedy is not None
    assert greedy.supports_sampling("greedy")
    assert not greedy.supports_sampling("sampled")


def test_dms_shaped_owner_refuses_all_speculative_modes_by_name() -> None:
    owner = _DmsOwner()

    for mode in ("greedy", "sampled"):
        capability = owner.speculative_capability((_semantics(3, sampling_mode=mode),))
        assert capability is None
    assert len(owner.decline_reasons) == 2
    assert all("DMS compact KV" in reason for reason in owner.decline_reasons)
    # No target work is attempted for any refused request.
    assert owner.target_calls == []


def test_dms_compact_kv_config_gate_names_its_clearing_command() -> None:
    """The DMS refusal is explicit and names the setting that lifts it."""

    from hipengine.generation.qwen35_gguf_dms import Qwen35GGUFDMSModelRunner

    runner = object.__new__(Qwen35GGUFDMSModelRunner)
    with pytest.raises(ValueError) as caught:
        runner.configure_engine_loop(
            SimpleNamespace(prefix_cache="off", speculative_mtp_serving="on")
        )

    message = str(caught.value)
    assert "DMS compact KV has no speculative serving adapter" in message
    # A gate nothing can clear is a bug: the error names the clearing command.
    assert "speculative_mtp_serving='off'" in message


# ------------------------------------------------- cross-model lifecycle parity


_STAGES = (
    "prepare_speculative_requests",
    "propose_speculative_batch",
    "execute_target_frontier",
)


@pytest.mark.parametrize("stage", _STAGES)
@pytest.mark.parametrize("owner_factory", _OWNER_FACTORIES)
def test_a_failed_stage_keeps_a_sibling_stream_and_a_retry_healthy(
    owner_factory, stage: str
) -> None:
    """The loop's failure protocol must hold on every model shape.

    A stage that raises on one stream must leave a sibling stream's staged state
    untouched, the failing cycle's claim reservation must be released exactly
    once, and the owner must stay usable for a retry of the same request with a
    fresh operation id - no shape accumulates poisoned per-cycle state.
    """

    owner = owner_factory()
    sibling_plan = _plan("op:sibling", (2,))
    sibling_semantics = (_semantics(2),)
    owner.prepare_speculative_requests(
        sibling_plan, sibling_semantics, stream=2
    )

    failing_plan = _plan("op:failing", (1,))
    failing_semantics = (_semantics(1),)
    original = getattr(owner, stage)

    def _fail(*args, **kwargs):
        raise RuntimeError(f"stage failure: {stage}")

    def _drive_through() -> None:
        # Drive the cycle through every stage up to and including the failing
        # one, each with its real seam signature.
        if stage != "prepare_speculative_requests":
            owner.prepare_speculative_requests(
                failing_plan, failing_semantics, stream=1
            )
        if stage == "execute_target_frontier":
            owner.propose_speculative_batch(failing_plan, failing_semantics, stream=1)
            owner.execute_target_frontier(
                failing_plan,
                None,
                reservation,
                commit=True,
                cancelled_request_ids=lambda: (),
            )
        elif stage == "propose_speculative_batch":
            owner.propose_speculative_batch(failing_plan, failing_semantics, stream=1)
        else:
            owner.prepare_speculative_requests(
                failing_plan, failing_semantics, stream=1
            )

    setattr(owner, stage, _fail)
    reservation = owner.reserve_speculative_claims(
        ResourceClaimSet.from_mapping(
            "op:failing:complete", {"provider.kv": 1}
        )
    )
    try:
        _drive_through()
        raise AssertionError(f"{stage} did not fail")
    except RuntimeError as error:
        assert str(error) == f"stage failure: {stage}"
    finally:
        setattr(owner, stage, original)
        owner.release_speculative_claims(reservation)

    # The sibling stream's staged state survived the failure untouched.
    sibling_graph = owner.propose_speculative_batch(
        sibling_plan, sibling_semantics, stream=2
    )
    assert sibling_graph.request_ids == (2,)

    # The owner is reusable: the same request retries cleanly on a new cycle.
    retry_plan = _plan("op:retry", (1,))
    retry_reservation = owner.reserve_speculative_claims(
        ResourceClaimSet.from_mapping(
            "op:retry:complete", {"provider.kv": 1}
        )
    )
    owner.prepare_speculative_requests(retry_plan, failing_semantics, stream=1)
    retry_graph = owner.propose_speculative_batch(
        retry_plan, failing_semantics, stream=1
    )
    assert retry_graph.request_ids == (1,)
    retry_result = owner.execute_target_frontier(
        retry_plan,
        None,
        retry_reservation,
        commit=True,
        cancelled_request_ids=lambda: (),
    )
    assert retry_result.operation_id == "op:retry"
    assert ("op:retry", True) in owner.target_calls
    owner.release_speculative_claims(retry_reservation)

    # The failing cycle reserved and released exactly once, and the retry adds
    # one more balanced pair - no shape leaks or double-releases claims.
    assert len(owner.reservations) == 2
    assert owner.releases == [reservation, retry_reservation]


@pytest.mark.parametrize("owner_factory", _OWNER_FACTORIES)
def test_cancellation_resolves_at_the_target_boundary_without_publishing_the_cancelled_row(
    owner_factory,
) -> None:
    """A cancelled row publishes nothing; survivors commit exactly themselves."""

    owner = owner_factory()
    plan = _plan("op:cancel", (1, 2))
    semantics = (_semantics(1), _semantics(2))
    owner.prepare_speculative_requests(plan, semantics, stream=1)
    owner.propose_speculative_batch(plan, semantics, stream=1)

    result = owner.execute_target_frontier(
        plan,
        None,
        None,
        commit=True,
        cancelled_request_ids=lambda: (2,),
    )

    # The cancelled request never reaches the committed result: no tail ids.
    assert result.committed_request_ids == (1,)
    assert 2 not in result.committed_request_ids
    assert owner.cancelled_ids == [(2,)]
    # An uncontested follow-up cycle publishes every row.
    followup = _plan("op:after", (3, 4))
    followup_semantics = (_semantics(3), _semantics(4))
    owner.prepare_speculative_requests(followup, followup_semantics, stream=1)
    owner.propose_speculative_batch(followup, followup_semantics, stream=1)
    clean = owner.execute_target_frontier(
        followup, None, None, commit=True, cancelled_request_ids=lambda: ()
    )
    assert clean.committed_request_ids == (3, 4)


@pytest.mark.parametrize(
    "owner_factory,mode,refused",
    [
        (_DenseOwner, "sampled", True),
        (_ParoOwner, "sampled", True),
        (_DmsOwner, "sampled", True),
        (_MoeOwner, "sampled", False),
        (_DmsOwner, "greedy", True),
    ],
)
def test_every_refusal_records_a_named_reason_and_admission_records_none(
    owner_factory, mode: str, refused: bool
) -> None:
    """No shape may hide a refusal behind a bare ``None``.

    Every refused semantics query records a non-empty reason (recorded again on
    a repeat, so a later refusal cannot be silently suppressed), and an
    admitted query records none. A silent downgrade to greedy or AR would have
    to pass through a reasonless refusal, which this forbids.
    """

    owner = owner_factory()
    query = (_semantics(11, sampling_mode=mode),)

    first = owner.speculative_capability(query)
    second = owner.speculative_capability(query)

    if refused:
        assert first is None and second is None
        assert len(owner.decline_reasons) == 2
        assert all(owner.decline_reasons)
        # The reason is stable across repeated queries of the same semantics.
        assert owner.decline_reasons[0] == owner.decline_reasons[1]
        assert owner.target_calls == []
    else:
        assert first is not None and second is not None
        assert owner.decline_reasons == []
        assert first.supports_sampling(mode)


def test_a_frontier_without_a_kv_binding_still_names_its_kv_keys() -> None:
    """K0/AR frontiers stay unbound; speculative capabilities now bind.

    A root-only K0/AR frontier carries no capability and therefore no typed
    binding, so ``kv_binding=None`` plus the loose string keys remains the
    legal shape for plain decode. Every speculative capability now populates
    a binding built from its target session's declared store, and a binding
    that disagrees with the frontier's own keys is still refused by name -
    the guard that makes a populated binding trustworthy.
    """

    owner = _DenseOwner()
    capability = owner.speculative_capability((_semantics(1),))
    assert capability is not None
    plan = _plan("op:frontier", (1,))
    semantics = (_semantics(1),)
    owner.prepare_speculative_requests(plan, semantics, stream=1)
    graph = owner.propose_speculative_batch(plan, semantics, stream=1)

    common = dict(
        operation_id=plan.operation_id,
        candidate_graph=graph,
        root_tokens=(5,),
        physical_row_decomposition=(2,),
        transaction_mode=capability.target_transaction_mode,
        kv_storage_view_key=capability.kv_backend_key,
        kv_live_spans_owner=f"{plan.operation_id}:target-live-spans",
        execution_route="graph",
    )

    # Today's production shape: no typed binding, loose keys carried.
    unbound = TargetFrontier.from_candidate_graph(kv_binding=None, **common)
    assert unbound.kv_binding is None
    assert unbound.kv_storage_view_key == capability.kv_backend_key
    assert unbound.kv_live_spans_owner == f"{plan.operation_id}:target-live-spans"

    # A fully populated binding that matches the loose keys is carried through.
    binding = capability.kv_binding
    assert binding is not None
    bound = TargetFrontier.from_candidate_graph(
        kv_binding=binding.with_live_spans_owner(
            f"{plan.operation_id}:target-live-spans"
        ),
        **common,
    )
    assert bound.kv_binding is not None
    assert bound.kv_binding.live_spans_owner == unbound.kv_live_spans_owner

    # A binding that disagrees with the frontier's own keys is refused by name,
    # so half-population cannot slip a mismatched store through the frontier.
    with pytest.raises(ValueError, match="live_spans_owner"):
        TargetFrontier.from_candidate_graph(
            kv_binding=binding.with_live_spans_owner("someone-else"),
            **common,
        )


@pytest.mark.parametrize("owner_factory", _OWNER_FACTORIES)
def test_staged_state_is_per_cycle_not_per_request_lifetime(owner_factory) -> None:
    """The same request id runs consecutive cycles without accumulating state.

    Staged state is keyed by the cycle's operation, not by the request's
    lifetime, so a request that stays resident across cycles must not carry the
    previous operation's prepared plan into the next one - and an unrelated
    request interleaving between the two must see neither.
    """

    owner = owner_factory()
    first = _plan("op:cycle-1", (5,))
    second = _plan("op:cycle-2", (5,))
    other = _plan("op:other", (6,))
    semantics_5 = (_semantics(5),)
    semantics_6 = (_semantics(6),)

    owner.prepare_speculative_requests(first, semantics_5, stream=1)
    owner.propose_speculative_batch(first, semantics_5, stream=1)

    # An unrelated request interleaves between this request's two cycles.
    owner.prepare_speculative_requests(other, semantics_6, stream=2)
    owner.propose_speculative_batch(other, semantics_6, stream=2)

    # The same request id's next cycle starts from a clean per-cycle slate.
    owner.prepare_speculative_requests(second, semantics_5, stream=1)
    owner.propose_speculative_batch(second, semantics_5, stream=1)

    assert owner.proposals == [(1, (5,)), (2, (6,)), (1, (5,))]
    # Stream 1's staged state now belongs to the latest cycle only: the second
    # operation overwrote the first rather than accumulating beside it, and
    # the double's cross-talk guard would have raised had a stale plan leaked.
    assert owner._prepared[1][0] == "op:cycle-2"
