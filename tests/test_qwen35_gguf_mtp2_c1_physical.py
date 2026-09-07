"""Packet 2 CPU seam REDs: explicitly qualified physical C1 on the staged cycle.

A one-request group is a valid one-row provider group plus one R2/R3/R4 target
frontier under the same staged adapter/resource transaction as C>1. The legacy
AR-row singleton route (`Qwen35GGUFTransactionalVerifier` /
`_ensure_active_singleton_target_verifier`) stays authoritative for gfx1151,
Qwen3.6, and capacity-1 engines; these tests pin the gfx1100 physical-C1 route
without touching those uses.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

import hipengine.generation.qwen35_gguf_mtp2 as mtp2_module
from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.tensor import Tensor
from hipengine.generation.qwen35_gguf_mtp2 import (
    Qwen35GGUFMTP2Adapter,
    _MTP2RequestState,
)
from hipengine.speculative import (
    CandidateGraph,
    SpecK0Class,
    SpecPlanReason,
    SpecRequestPlan,
    SpecTransactionMode,
    SpeculativeMTPStaticEligibility,
    SpeculativeMTPStaticState,
    SpeculativeRequestSemantics,
    TargetFrontier,
)
from hipengine.kvcache.backend import ResourceClaimSet
from hipengine.speculative.transaction import SpecCycleStage
from hipengine.kernels.backends import backend_package_capability
from hipengine.runtime.qwen35_gguf_nextn import Qwen35GGUFNextNBatchDeviceProposal


def _owner(*, backend: str, capacity: int) -> SimpleNamespace:
    return SimpleNamespace(
        generator=SimpleNamespace(
            backend=backend,
            execution_profile="production",
        ),
        capacity=capacity,
        _shared_runner=None,
        _row=lambda rid: SimpleNamespace(
            native_greedy=True,
            first_token_emitted=True,
            lease=SimpleNamespace(
                session=SimpleNamespace(
                    runner=SimpleNamespace(fp16_recurrent_state=False),
                    _target_scratch_owner=SimpleNamespace(slot_count=capacity),
                    target_layout=SimpleNamespace(max_sequence_length=1024),
                    kv_storage_dtype="bf16",
                    position=31,
                )
            ),
            slot=SimpleNamespace(),
        ),
    )


def _c1_eligibility(rid: int) -> SpeculativeMTPStaticEligibility:
    return SpeculativeMTPStaticEligibility(
        state=SpeculativeMTPStaticState.SPECULATIVE_CAPABLE,
        reason="qualified_test_physical_c1",
        max_candidate_count=3,
        max_realized_group_rows=1,
        automatic_eligible=False,
        strict_fallback_key="gguf_target_ar",
        evidence_key=f"test-physical-c1-{rid}",
        evidence_fingerprint=f"sha256:test-physical-c1-{rid}",
    )


def _wide_eligibility(rid: int, *, rows: int) -> SpeculativeMTPStaticEligibility:
    return SpeculativeMTPStaticEligibility(
        state=SpeculativeMTPStaticState.SPECULATIVE_CAPABLE,
        reason="qualified_test_physical_wide",
        max_candidate_count=3,
        max_realized_group_rows=rows,
        automatic_eligible=False,
        strict_fallback_key="gguf_target_ar",
        evidence_key=f"test-physical-wide-{rid}",
        evidence_fingerprint=f"sha256:test-physical-wide-{rid}",
    )


def _adapter(*, backend: str, capacity: int, rid: int, eligibility) -> Qwen35GGUFMTP2Adapter:
    adapter = Qwen35GGUFMTP2Adapter(
        _owner(backend=backend, capacity=capacity),
        enabled=True,
        target_verify_mode="native",
        candidate_budget=3,
    )
    adapter._intents = {rid: 3}
    adapter._static_eligibility_by_request = {rid: eligibility}
    adapter._prompt_hidden_rows = {rid: object()}
    adapter._states = {}
    adapter._disabled_requests = set()
    adapter._active_claims = None
    return adapter


@pytest.mark.parametrize("screening", [False, True])
@pytest.mark.parametrize("invalid", ["missing", "singleton", "depth"])
def test_claims_recheck_each_requests_safety_envelope(monkeypatch, screening, invalid) -> None:
    if screening:
        monkeypatch.setenv("HIPENGINE_MTP2_SCREEN_UNQUALIFIED_CELLS", "1")
    else:
        monkeypatch.delenv("HIPENGINE_MTP2_SCREEN_UNQUALIFIED_CELLS", raising=False)
    adapter = _adapter(
        backend="hip_gfx1100", capacity=8, rid=7,
        eligibility=_wide_eligibility(7, rows=2),
    )
    adapter._static_eligibility_by_request[8] = _wide_eligibility(8, rows=2)
    plan = SimpleNamespace(
        request_ids=(7, 8), speculative_request_ids=(7, 8), candidate_counts=(3, 3),
    )
    assert adapter.claims_fit(plan) is True
    if invalid == "missing":
        del adapter._static_eligibility_by_request[8]
    elif invalid == "singleton":
        adapter._static_eligibility_by_request[8] = _c1_eligibility(8)
    else:
        adapter._static_eligibility_by_request[8] = replace(
            _wide_eligibility(8, rows=2), max_candidate_count=2,
        )
    assert adapter.claims_fit(plan) is False
    assert adapter._states == {}


def test_physical_c1_route_flag_is_package_owned() -> None:
    """gfx1100 owns the physical-C1 route; gfx1151 keeps the legacy singleton."""

    assert (
        backend_package_capability("hip_gfx1100", "GGUF_SPECDEC2_MTP2_PHYSICAL_C1", False)
        is True
    )
    assert (
        backend_package_capability("hip_gfx1151", "GGUF_SPECDEC2_MTP2_PHYSICAL_C1", False)
        is False
    )


def test_physical_c1_partition_admits_qualified_single_request() -> None:
    """A C1-qualified request resolves partition bound 1; others stay closed."""

    adapter = _adapter(
        backend="hip_gfx1100",
        capacity=8,
        rid=7,
        eligibility=_c1_eligibility(7),
    )
    assert adapter.partition_max_requests((7,)) == 1


def test_partition_still_rejects_unqualified_singletons() -> None:
    """Legacy singletons stay K0; wide evidence keeps its existing width bound."""

    wide = _adapter(
        backend="hip_gfx1100",
        capacity=8,
        rid=7,
        eligibility=_wide_eligibility(7, rows=8),
    )
    # Pre-existing qualified behavior: a one-request due group with wide
    # evidence partitions at its evidence width and rides the packed route.
    assert wide.partition_max_requests((7,)) == 8


def test_partition_never_decomposes_multi_request_batch_into_serial_c1() -> None:
    """A multi-request due batch of C1-qualified rows must not chain C1 cycles."""

    adapter = _adapter(
        backend="hip_gfx1100",
        capacity=8,
        rid=7,
        eligibility=_c1_eligibility(7),
    )
    adapter._intents[8] = 3
    adapter._static_eligibility_by_request[8] = _c1_eligibility(8)
    adapter._prompt_hidden_rows[8] = object()
    assert adapter.partition_max_requests((7, 8)) == 0

    legacy = _adapter(
        backend="hip_gfx1151",
        capacity=8,
        rid=7,
        eligibility=_c1_eligibility(7),
    )
    assert legacy.partition_max_requests((7,)) == 0


def test_physical_c1_claims_fit_admits_listed_one_row_cell() -> None:
    adapter = _adapter(
        backend="hip_gfx1100",
        capacity=8,
        rid=7,
        eligibility=_c1_eligibility(7),
    )
    assert adapter.claims_fit(
        SimpleNamespace(
            request_ids=(7,),
            speculative_request_ids=(7,),
            candidate_counts=(3,),
        )
    ) is True
    # K2 is the other listed C1 depth cell.
    assert adapter.claims_fit(
        SimpleNamespace(
            request_ids=(7,),
            speculative_request_ids=(7,),
            candidate_counts=(2,),
        )
    ) is True
    # A K1 claim at a depth the policy does not list for C1 stays closed.
    assert adapter.claims_fit(
        SimpleNamespace(
            request_ids=(7,),
            speculative_request_ids=(7,),
            candidate_counts=(1,),
        )
    ) is False


def test_physical_c1_capability_returns_packed_cell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The C1 capability resolves without touching legacy singleton owners."""

    def _forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("physical C1 must not construct the legacy verifier")

    monkeypatch.setattr(
        mtp2_module, "Qwen35GGUFTransactionalVerifier", _forbidden
    )
    adapter = _adapter(
        backend="hip_gfx1100",
        capacity=8,
        rid=7,
        eligibility=_c1_eligibility(7),
    )
    semantics = (
        SpeculativeRequestSemantics(7, "greedy", "verify_chain", 32, 25),
    )
    capability = adapter.capability(semantics)
    assert capability is not None
    assert capability.max_candidates_per_request == 3


def test_physical_c1_single_survivor_keeps_legacy_route_for_gfx1151(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """gfx1151 rows==1 requests keep the legacy singleton verifier decision."""

    constructed: list[object] = []

    class FakeVerifier:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            constructed.append(self)

    monkeypatch.setattr(
        mtp2_module, "Qwen35GGUFTransactionalVerifier", FakeVerifier
    )
    adapter = _adapter(
        backend="hip_gfx1151",
        capacity=8,
        rid=7,
        eligibility=_c1_eligibility(7),
    )
    assert adapter._singleton_only(7) is True
    assert adapter._physical_c1_request(7) is False
    assert constructed == []


def test_physical_c1_state_open_routes_to_batch_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing C1 state opens the physical batch group, not the legacy open."""

    calls: list[tuple[str, tuple[int, ...]]] = []
    adapter = _adapter(
        backend="hip_gfx1100",
        capacity=8,
        rid=7,
        eligibility=_c1_eligibility(7),
    )
    monkeypatch.setattr(
        adapter,
        "_open_batch_requests",
        lambda ids: calls.append(("batch", tuple(int(v) for v in ids))),
    )
    monkeypatch.setattr(
        adapter,
        "_open_request",
        lambda rid: calls.append(("legacy", (int(rid),))),
    )
    adapter._ensure_request_states((7,))
    assert calls == [("batch", (7,))]

    legacy = _adapter(
        backend="hip_gfx1151",
        capacity=8,
        rid=7,
        eligibility=_c1_eligibility(7),
    )
    calls.clear()
    monkeypatch.setattr(
        legacy,
        "_open_batch_requests",
        lambda ids: calls.append(("batch", tuple(int(v) for v in ids))),
    )
    monkeypatch.setattr(
        legacy,
        "_open_request",
        lambda rid: calls.append(("legacy", (int(rid),))),
    )
    legacy._ensure_request_states((7,))
    assert calls == [("legacy", (7,))]


def _c1_plan(rid: int = 7) -> SpecRequestPlan:
    return SpecRequestPlan(
        operation_id="specdec2-cycle:test",
        cycle_id=1,
        request_ids=(rid,),
        resident_slots=(0,),
        candidate_counts=(3,),
        reasons=(SpecPlanReason.SPECULATIVE_QUALIFIED,),
        k0_classes=(SpecK0Class.NOT_K0,),
        mode="verify_chain",
        capability_key="gguf_mtp2_c1:hip_gfx1100:gguf_q4_k_m:native:3",
        provider_key="qwen_nextn_dense",
        target_transaction_mode=SpecTransactionMode.RESERVED_APPEND,
        provider_transaction_mode=SpecTransactionMode.RESERVED_APPEND,
        proposal_widths=(1,),
        target_row_decomposition=(4,),
        context_bucket_size=64,
        execution_route="graph",
    )


def _c1_frontier(plan: SpecRequestPlan, graph_tokens: Tensor) -> TargetFrontier:
    graph = CandidateGraph(
        provider_key="qwen_nextn_dense",
        method_key="mtp2",
        policy_fingerprint="fake-policy:v1",
        cycle_id=plan.cycle_id,
        transaction_id=1,
        request_ids=plan.request_ids,
        resident_slots=plan.resident_slots,
        root_positions=(31,),
        row_offsets=(0, 3),
        row_to_request=(plan.request_ids[0],) * 3,
        parent_candidate_rows=(-1, 0, 1),
        draft_depths=(1, 2, 3),
        active_mask=(True, True, True),
        candidate_tokens=(101, 102, 103),
        token_ids=graph_tokens,
    )
    return TargetFrontier(
        operation_id=plan.operation_id,
        cycle_id=plan.cycle_id,
        request_ids=plan.request_ids,
        resident_slots=plan.resident_slots,
        root_tokens=(90,),
        root_positions=(31,),
        physical_row_decomposition=(4,),
        transaction_mode=plan.target_transaction_mode,
        kv_storage_view_key="bf16",
        kv_live_spans_owner="fake-live-spans",
        execution_route="graph",
        candidate_graph=graph,
        provider_transaction_id=1,
    )


class _CancelExecutor:
    def __init__(self, calls: list) -> None:
        self._calls = calls

    def restore_request_checkpoint(self, checkpoint):
        self._calls.append(("restore", checkpoint))

    def release_request_checkpoint(self, checkpoint):
        self._calls.append(("release", checkpoint))


def test_physical_c1_cancelled_cycle_restores_provider_and_fails_closed() -> None:
    """A cancelled C1 cycle never mutates target or provider state."""

    calls: list[tuple[object, ...]] = []
    adapter = _adapter(
        backend="hip_gfx1100",
        capacity=8,
        rid=7,
        eligibility=_c1_eligibility(7),
    )
    provider = SimpleNamespace(
        executor=_CancelExecutor(calls),
        release_request=lambda rid: None,
    )
    graph_tokens = Tensor.from_handle(0x5000, (3,), DType.INT32, Device("hip", 0))
    proposal = Qwen35GGUFNextNBatchDeviceProposal(
        request_ids=(7,),
        root_tokens=(90,),
        root_positions=(31,),
        candidate_counts=(3,),
        token_ids=graph_tokens,
        hidden_rows=(
            (
                Tensor.from_handle(0x6000, (1, 4), DType.BF16, Device("hip", 0)),
                Tensor.from_handle(0x6010, (1, 4), DType.BF16, Device("hip", 0)),
                Tensor.from_handle(0x6020, (1, 4), DType.BF16, Device("hip", 0)),
            ),
        ),
    )
    adapter._states[7] = _MTP2RequestState(
        request_id=7,
        provider=provider,
        provider_pool_key=None,
        provider_group_key=(7,),
        verifier=None,
        root_hidden_buffer=SimpleNamespace(ptr=1),
        proposal_checkpoint="checkpoint-1",
        proposal_device_batch=proposal,
    )
    adapter._provider_groups[(7,)] = SimpleNamespace(
        key=(7,), provider=provider, request_ids={7}
    )
    claims = ResourceClaimSet(claim_id="specdec2-cycle:test")
    adapter._active_claims = claims

    result = adapter.execute_target_frontier(
        _c1_plan(7),
        _c1_frontier(_c1_plan(7), graph_tokens),
        claims,
        commit=True,
        cancelled_request_ids=lambda: (7,),
    )

    assert result.stage is SpecCycleStage.CANCELLED
    assert result.cancelled_request_ids == (7,)
    assert result.transaction.rolled_back is True
    assert result.transaction.target_open is False
    assert result.transaction.provider_open is False
    assert calls == [("restore", "checkpoint-1"), ("release", "checkpoint-1")]
    assert adapter._states[7].proposal_checkpoint is None


def test_physical_c1_survivor_of_larger_group_claims_one_row_plan() -> None:
    """A lone C1-qualified survivor of a wider owner keeps the packed route."""

    adapter = _adapter(
        backend="hip_gfx1100",
        capacity=8,
        rid=3,
        eligibility=_c1_eligibility(3),
    )
    provider = SimpleNamespace(executor=SimpleNamespace(max_requests=8))
    adapter._states[3] = _MTP2RequestState(
        request_id=3,
        provider=provider,
        provider_pool_key=None,
        provider_group_key=(1, 2, 3, 4, 5, 6, 7, 8),
        verifier=None,
        root_hidden_buffer=SimpleNamespace(ptr=3),
    )
    adapter._provider_groups[(1, 2, 3, 4, 5, 6, 7, 8)] = SimpleNamespace(
        key=(1, 2, 3, 4, 5, 6, 7, 8),
        provider=provider,
        request_ids={3},
    )

    # The survivor keeps its physical group membership (verifier stays None)
    # and a one-row cycle claim fits the listed (1, K) cell.
    assert adapter._states[3].verifier is None
    assert adapter._physical_c1_request(3) is True
    assert adapter.claims_fit(
        SimpleNamespace(
            request_ids=(3,),
            speculative_request_ids=(3,),
            candidate_counts=(3,),
        )
    ) is True


def test_physical_c1_refill_keeps_pair_closed_until_wide_evidence() -> None:
    """A newcomer may join the group's provider, but the pair stays K0."""

    adapter = _adapter(
        backend="hip_gfx1100",
        capacity=8,
        rid=7,
        eligibility=_c1_eligibility(7),
    )
    adapter._intents[8] = 3
    adapter._static_eligibility_by_request[8] = _c1_eligibility(8)
    adapter._prompt_hidden_rows[8] = object()

    # Refill: the C1-qualified newcomer opens its own one-row physical group
    # (independent C1 route lifecycle); the resident survivor keeps its group.
    adapter._states[7] = _MTP2RequestState(
        request_id=7,
        provider=SimpleNamespace(executor=SimpleNamespace(max_requests=8)),
        provider_pool_key=None,
        provider_group_key=(7,),
        verifier=None,
        root_hidden_buffer=SimpleNamespace(ptr=7),
    )
    adapter._open_batch_requests = lambda ids: adapter._states.update(
        {
            rid: _MTP2RequestState(
                request_id=rid,
                provider=SimpleNamespace(executor=SimpleNamespace(max_requests=8)),
                provider_pool_key=None,
                provider_group_key=(rid,),
                verifier=None,
                root_hidden_buffer=SimpleNamespace(ptr=rid),
            )
            for rid in ids
        }
    )
    adapter._ensure_request_states((7, 8))

    assert adapter._states[8].verifier is None
    assert adapter._states[8].provider_group_key == (8,)
    assert adapter._states[7].verifier is None

    # Cycle level: the two rows==1 requests never compose an MTP batch and
    # never chain serial C1 cycles.
    assert adapter.partition_max_requests((7, 8)) == 0
    assert adapter.claims_fit(
        SimpleNamespace(
            request_ids=(7, 8),
            speculative_request_ids=(7, 8),
            candidate_counts=(3, 3),
        )
    ) is False

    # Once the newcomer drains, the survivor returns to its C1 route.
    adapter._states.pop(8)
    assert adapter.partition_max_requests((7,)) == 1
    assert adapter.claims_fit(
        SimpleNamespace(
            request_ids=(7,),
            speculative_request_ids=(7,),
            candidate_counts=(3,),
        )
    ) is True


def test_physical_c1_frontier_pads_into_the_shared_accept_bucket() -> None:
    """The C1 R4 frontier pads to the rows-6 shape inside the r36 bucket."""

    from hipengine.speculative.frontier import physical_group_pad_rows

    adapter = _adapter(
        backend="hip_gfx1100",
        capacity=8,
        rid=7,
        eligibility=_c1_eligibility(7),
    )
    # gfx1100 production verifies at the rows-6 tile: the one-row K3 frontier
    # (4 active rows) pads by 2 inactive rows, still inside the shared
    # max-shaped accept bucket (8 requests x 4 rows padded to 36).
    assert adapter.production_target_pad_row_counts == (6,)
    assert (
        physical_group_pad_rows(
            adapter.production_target_pad_row_counts,
            request_count=1,
            candidate_rows=3,
            max_rows=adapter.physical_accept_max_rows,
        )
        == 2
    )
    assert adapter.physical_accept_max_rows == 36
