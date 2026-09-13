from __future__ import annotations

import pytest

from hipengine.core.device import Device
from hipengine.core.tensor import Tensor
from hipengine.speculative import (
    CandidateGraph,
    ProviderAttachment,
    ProviderCatchupMode,
    SpecK0Class,
    SpecPlanReason,
    SpecRequestPlan,
    SpecTransactionMode,
    SpeculativeCapability,
    TargetFrontier,
)


def _capability(**overrides) -> SpeculativeCapability:
    values = {
        "capability_key": "mtp2:qwen38:gfx1151:strict",
        "target_key": "qwen38_27b_q4ks",
        "provider_key": "qwen38_nextn",
        "method_key": "mtp2",
        "policy_fingerprint": "sha256:policy",
        "execution_profile": "strict",
        "kv_backend_key": "paged_bf16",
        "attachment": ProviderAttachment.TARGET_ATTACHED,
        "catchup_mode": ProviderCatchupMode.TARGET_OUTPUT,
        "supported_modes": ("verify_chain", "verify_tree"),
        "supported_sampling_modes": ("greedy",),
        "max_requests": 4,
        "max_candidates_per_request": 3,
        "max_frontier_rows": 16,
        "proposal_widths": (1, 2, 4),
        "target_row_buckets": (2, 4, 8, 16),
        "target_transaction_mode": SpecTransactionMode.PACKED_SCRATCH,
        "provider_transaction_mode": SpecTransactionMode.REVERSIBLE_JOURNAL,
        "graph_supported": True,
        "eager_supported": True,
        "strict_fallback_key": "target_ar_strict",
    }
    values.update(overrides)
    return SpeculativeCapability(**values)


def test_capability_validates_shape_and_execution_contract() -> None:
    capability = _capability()

    assert capability.supports_shape(
        request_count=4,
        candidate_counts=(3, 0, 2, 1),
        mode="verify_chain",
    )
    assert not capability.supports_shape(
        request_count=5,
        candidate_counts=(1, 1, 1, 1, 1),
        mode="verify_chain",
    )
    assert not capability.supports_shape(
        request_count=2,
        candidate_counts=(4, 1),
        mode="verify_chain",
    )
    with pytest.raises(ValueError, match="at least one execution route"):
        _capability(graph_supported=False, eager_supported=False)
    with pytest.raises(ValueError, match="target row bucket"):
        _capability(target_row_buckets=(2, 32))


def test_request_plan_represents_ar_as_k0_without_provider_mutation() -> None:
    plan = SpecRequestPlan(
        operation_id="spec-plan:1",
        cycle_id=1,
        request_ids=(10, 20),
        resident_slots=(3, 1),
        candidate_counts=(0, 0),
        reasons=(SpecPlanReason.NO_PROVIDER, SpecPlanReason.POLICY_SELECTED_AR),
        k0_classes=(SpecK0Class.TRANSITIONAL, SpecK0Class.PURE),
        mode="decode",
        capability_key=None,
        provider_key=None,
        target_transaction_mode=SpecTransactionMode.RESERVED_APPEND,
        provider_transaction_mode=None,
        proposal_widths=(),
        target_row_decomposition=(2,),
        context_bucket_size=256,
        execution_route="ar",
    )

    assert plan.is_ar_only
    assert not plan.has_speculative_rows
    assert plan.logical_frontier_rows == 2
    assert plan.speculative_request_ids == ()


def test_request_plan_supports_mixed_k0_and_speculative_rows() -> None:
    plan = SpecRequestPlan(
        operation_id="spec-plan:2",
        cycle_id=7,
        request_ids=(11, 22, 33),
        resident_slots=(0, 2, 1),
        candidate_counts=(0, 2, 1),
        reasons=(
            SpecPlanReason.POLICY_SELECTED_AR,
            SpecPlanReason.SPECULATIVE_QUALIFIED,
            SpecPlanReason.SPECULATIVE_QUALIFIED,
        ),
        k0_classes=(
            SpecK0Class.PURE,
            SpecK0Class.NOT_K0,
            SpecK0Class.NOT_K0,
        ),
        mode="verify_chain",
        capability_key="mtp2:qwen38:gfx1151:strict",
        provider_key="qwen38_nextn",
        target_transaction_mode=SpecTransactionMode.PACKED_SCRATCH,
        provider_transaction_mode=SpecTransactionMode.REVERSIBLE_JOURNAL,
        proposal_widths=(2,),
        target_row_decomposition=(4, 2),
        context_bucket_size=1024,
        execution_route="graph",
    )

    assert plan.has_speculative_rows
    assert not plan.is_ar_only
    assert plan.logical_frontier_rows == 6
    assert plan.speculative_request_ids == (22, 33)
    assert plan.max_candidate_count == 2

    with pytest.raises(ValueError, match="speculative-qualified"):
        SpecRequestPlan(
            operation_id="bad",
            cycle_id=1,
            request_ids=(1,),
            resident_slots=(0,),
            candidate_counts=(0,),
            reasons=(SpecPlanReason.SPECULATIVE_QUALIFIED,),
            k0_classes=(SpecK0Class.PURE,),
            mode="decode",
            capability_key=None,
            provider_key=None,
            target_transaction_mode=SpecTransactionMode.RESERVED_APPEND,
            provider_transaction_mode=None,
            proposal_widths=(),
            target_row_decomposition=(1,),
            context_bucket_size=256,
            execution_route="ar",
        )


def test_candidate_graph_projects_mixed_chain_to_existing_draft_contract() -> None:
    graph = CandidateGraph(
        provider_key="qwen38_nextn",
        method_key="mtp2",
        policy_fingerprint="sha256:policy",
        cycle_id=4,
        transaction_id=9,
        request_ids=(10, 20),
        resident_slots=(3, 1),
        root_positions=(100, 200),
        row_offsets=(0, 0, 2),
        row_to_request=(20, 20),
        parent_candidate_rows=(-1, 0),
        draft_depths=(1, 2),
        active_mask=(True, True),
        candidate_tokens=(501, 502),
        mode="verify_chain",
    )

    assert graph.candidate_counts == (0, 2)
    draft = graph.to_draft_batch()
    assert draft.request_ids == (10, 20)
    assert draft.candidate_tokens == (501, 502)
    assert draft.parent_positions == (200, 201)
    assert draft.row_to_request == (20, 20)
    assert draft.tree_parents == (-1, 0)
    assert draft.resident_slots == (1, 1)

    frontier = TargetFrontier.from_candidate_graph(
        operation_id="spec-cycle:4:9",
        candidate_graph=graph,
        root_tokens=(1001, 2001),
        physical_row_decomposition=(4,),
        transaction_mode=SpecTransactionMode.PACKED_SCRATCH,
        kv_storage_view_key="paged_bf16",
        kv_live_spans_owner="target-session:7",
        execution_route="graph",
    )
    assert frontier.logical_rows == 4
    assert frontier.candidate_counts == (0, 2)
    assert frontier.target_batch is not None
    assert frontier.target_batch.tokens == (1001, 2001, 501, 502)
    assert frontier.target_batch.parent_rows == (-1, -1, 1, 2)


def test_candidate_graph_rejects_cross_request_parent_ownership() -> None:
    with pytest.raises(ValueError, match="same request"):
        CandidateGraph(
            provider_key="provider",
            method_key="mtp2",
            policy_fingerprint="policy",
            cycle_id=1,
            transaction_id=1,
            request_ids=(1, 2),
            resident_slots=(0, 1),
            root_positions=(5, 9),
            row_offsets=(0, 1, 2),
            row_to_request=(1, 2),
            parent_candidate_rows=(-1, 0),
            draft_depths=(1, 2),
            active_mask=(True, True),
            candidate_tokens=(10, 20),
        )


def test_device_candidate_graph_does_not_require_host_token_copy() -> None:
    token_ids = Tensor.from_handle(0x1000, (2,), "int32", Device("hip", 0))
    graph = CandidateGraph(
        provider_key="provider",
        method_key="mtp2",
        policy_fingerprint="policy",
        cycle_id=1,
        transaction_id=1,
        request_ids=(1,),
        resident_slots=(2,),
        root_positions=(8,),
        row_offsets=(0, 2),
        row_to_request=(1, 1),
        parent_candidate_rows=(-1, 0),
        draft_depths=(1, 2),
        active_mask=(True, True),
        token_ids=token_ids,
    )

    assert graph.device is token_ids.device
    with pytest.raises(RuntimeError, match="host candidate tokens"):
        graph.to_draft_batch()

    frontier = TargetFrontier.from_candidate_graph(
        operation_id="device-cycle:1",
        candidate_graph=graph,
        root_tokens=(99,),
        physical_row_decomposition=(3,),
        transaction_mode=SpecTransactionMode.PACKED_SCRATCH,
        kv_storage_view_key="paged_bf16",
        kv_live_spans_owner="target-session:device",
        execution_route="graph",
    )
    assert frontier.logical_rows == 3
    assert frontier.candidate_graph is graph
    assert frontier.target_batch is None


def test_target_frontier_represents_root_only_ar() -> None:
    frontier = TargetFrontier.from_ar_roots(
        operation_id="ar-cycle:3",
        cycle_id=3,
        request_ids=(4, 9),
        resident_slots=(7, 2),
        root_tokens=(44, 99),
        root_positions=(12, 30),
        physical_row_decomposition=(2,),
        transaction_mode=SpecTransactionMode.RESERVED_APPEND,
        kv_storage_view_key="paged_bf16",
        kv_live_spans_owner="target-session:2",
        execution_route="ar",
    )

    assert frontier.is_ar_only
    assert frontier.logical_rows == 2
    assert frontier.candidate_counts == (0, 0)
    assert frontier.target_batch is None
    assert frontier.root_tokens == (44, 99)
