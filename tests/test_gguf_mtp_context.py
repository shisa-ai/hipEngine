from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from hipengine.kernels.cpu_reference import register_cpu_reference_kernels
from hipengine.kernels.hip_gfx1100.sampling.sampler import register_sampler_kernels
from hipengine.kvcache.policy import KVTransaction
from hipengine.speculative.interfaces import DraftBatch, TargetAcceptSummary, TargetCommitPlan, TargetVerifyBatch
from hipengine.speculative.gguf_mtp import (
    GGUF_MTP_ACCEPTED_DRAFT_COMPARABLE,
    GGUF_MTP_ACCEPTED_DRAFT_NOT_COMPARABLE_DEBUG_TRACE,
    GGUF_MTP_ACCEPTED_OUTPUT_COMPARABLE,
    GGUF_MTP_ACCEPTED_OUTPUT_NOT_COMPARABLE_DEBUG_TRACE,
    GGUF_MTP_FULL_TRACE_BUDGET_COVERAGE,
    GGUF_MTP_METRICS_CONTRACT_READY,
    GGUF_MTP_PARTIAL_TRACE_BUDGET_COVERAGE,
    Qwen35GGUFMTPAcceptStep,
    Qwen35GGUFMTPAcceptStepMetrics,
    Qwen35GGUFMTPTop1AcceptSpec,
    Qwen35GGUFMTPContext,
    Qwen35GGUFMTPDraftBatch,
    Qwen35GGUFMTPDraftExecutionPlan,
    Qwen35GGUFMTPDraftProposal,
    Qwen35GGUFMTPDraftRow,
    Qwen35GGUFMTPKVLiveSpansPlan,
    Qwen35GGUFMTPPerformanceReadiness,
    Qwen35GGUFMTPRuntimeKernelPlan,
    Qwen35GGUFMTPSeedRow,
    Qwen35GGUFMTPVerificationMetrics,
    Qwen35GGUFMTPVerificationResult,
)


@dataclass(frozen=True)
class _Contract:
    ready_for_mtp: bool = True
    rows: int = 1
    hidden_size: int = 8


@dataclass(frozen=True)
class _Seed:
    token_id: int
    position: int
    hidden_ptr: int
    hidden_contract: _Contract = _Contract()


class _TargetSession:
    def __init__(self, seed: _Seed) -> None:
        self.seed = seed
        self.calls: list[tuple[int, int]] = []

    def mtp_draft_seed(self, *, token_id: int, position: int) -> _Seed:
        self.calls.append((token_id, position))
        return _Seed(
            token_id=token_id,
            position=position,
            hidden_ptr=self.seed.hidden_ptr,
            hidden_contract=self.seed.hidden_contract,
        )


def test_gguf_mtp_context_captures_target_seed_and_builds_b1_row() -> None:
    target = _TargetSession(_Seed(token_id=0, position=0, hidden_ptr=0x1000))

    context = Qwen35GGUFMTPContext.from_target_seed(
        target,
        token_id=42,
        position=17,
        mtp_block=object(),
    )
    batch = context.build_b1_draft_batch(request_id=3, token_id=99)

    assert target.calls == [(42, 17)]
    assert context.pending_seed == Qwen35GGUFMTPSeedRow(
        token_id=42,
        position=17,
        hidden_ptr=0x1000,
        hidden_size=8,
        source="target",
    )
    assert batch.request_ids == (3,)
    assert batch.token_ids == (99,)
    assert batch.embedding_seed_ptrs == (0x1000,)
    assert batch.as_dict()["rows"] == [
        {
            "request_id": 3,
            "token_id": 99,
            "position": 18,
            "draft_depth": 1,
            "embedding_seed_ptr": 0x1000,
            "embedding_hidden_size": 8,
            "parent_token_id": 42,
            "parent_position": 17,
        }
    ]


def test_gguf_mtp_context_builds_b1_proposal_from_registered_topk_logits() -> None:
    register_cpu_reference_kernels(replace=True)
    context = Qwen35GGUFMTPContext(target_session=object())
    context.capture_pending_seed(_Seed(token_id=10, position=5, hidden_ptr=0x1000))
    logits = np.asarray([[0.5, 3.0, 3.0, -1.0]], dtype=np.float32)

    proposal = context.build_draft_proposal_from_logits(request_id=4, logits=logits, top_k=2)

    assert isinstance(proposal, Qwen35GGUFMTPDraftProposal)
    assert proposal.proposed_token_ids == (1,)
    assert proposal.top_k_token_ids == ((1, 2),)
    assert proposal.top_k_logits == ((3.0, 3.0),)
    assert proposal.batch.token_ids == (1,)
    assert proposal.as_dict()["topk_kernel"] == ["cpu_reference", "mtp_draft_topk", "w4_gguf", "full_vocab_d2h"]
    assert proposal.as_dict()["proposed_token_ids"] == [1]


def test_gguf_mtp_context_builds_multi_depth_proposal_from_registered_topk_logits() -> None:
    register_cpu_reference_kernels(replace=True)
    context = Qwen35GGUFMTPContext(target_session=object())
    seeds = (
        Qwen35GGUFMTPSeedRow(token_id=10, position=5, hidden_ptr=0x1000, hidden_size=8, source="target"),
        Qwen35GGUFMTPSeedRow(token_id=11, position=6, hidden_ptr=0x2000, hidden_size=8, source="draft[0]"),
    )
    logits = np.asarray(
        [
            [0.0, 4.0, 1.0],
            [3.0, 2.0, 5.0],
        ],
        dtype=np.float32,
    )

    proposal = context.build_draft_proposal_from_logits(
        request_id=7,
        logits=logits,
        seed_rows=seeds,
        top_k=1,
    )

    assert proposal.proposed_token_ids == (1, 2)
    assert [row.draft_depth for row in proposal.batch.rows] == [1, 2]
    assert [row.position for row in proposal.batch.rows] == [6, 7]
    assert proposal.as_dict()["top_k_token_ids"] == [[1], [2]]


def test_gguf_mtp_context_rejects_invalid_topk_proposal_contract() -> None:
    register_cpu_reference_kernels(replace=True)
    context = Qwen35GGUFMTPContext(target_session=object())
    context.capture_pending_seed(_Seed(token_id=10, position=5, hidden_ptr=0x1000))
    logits = np.asarray([[0.0, 1.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="selected_index"):
        context.build_draft_proposal_from_logits(request_id=0, logits=logits, top_k=1, selected_index=1)
    with pytest.raises(ValueError, match="four-axis"):
        context.build_draft_proposal_from_logits(
            request_id=0,
            logits=logits,
            topk_kernel=("cpu_reference", "mtp_draft_topk", "w4_gguf"),  # type: ignore[arg-type]
        )
    row = context.build_b1_draft_batch(request_id=0, token_id=1)
    with pytest.raises(ValueError, match="match selected top-k"):
        Qwen35GGUFMTPDraftProposal(
            batch=row,
            top_k_token_ids=((0,),),
            top_k_logits=((1.0,),),
        )


def test_gguf_mtp_context_builds_multi_depth_draft_batch_from_seed_rows() -> None:
    context = Qwen35GGUFMTPContext(target_session=object())
    seeds = (
        Qwen35GGUFMTPSeedRow(token_id=10, position=5, hidden_ptr=0x1000, hidden_size=8, source="target"),
        Qwen35GGUFMTPSeedRow(token_id=11, position=6, hidden_ptr=0x2000, hidden_size=8, source="draft[0]"),
        Qwen35GGUFMTPSeedRow(token_id=12, position=7, hidden_ptr=0x3000, hidden_size=8, source="draft[1]"),
    )

    batch = context.build_draft_batch(request_id=4, token_ids=(101, 102, 103), seed_rows=seeds)

    assert batch.request_ids == (4,)
    assert batch.token_ids == (101, 102, 103)
    assert batch.embedding_seed_ptrs == (0x1000, 0x2000, 0x3000)
    assert [row.draft_depth for row in batch.rows] == [1, 2, 3]
    assert [row.position for row in batch.rows] == [6, 7, 8]
    assert [row.parent_token_id for row in batch.rows] == [10, 11, 12]
    assert [row.parent_position for row in batch.rows] == [5, 6, 7]


def test_gguf_mtp_draft_batch_projects_to_shared_verifier_batch() -> None:
    rows = (
        Qwen35GGUFMTPDraftRow(
            request_id=10,
            token_id=101,
            position=6,
            draft_depth=1,
            embedding_seed_ptr=0x1000,
            embedding_hidden_size=8,
            parent_token_id=100,
            parent_position=5,
        ),
        Qwen35GGUFMTPDraftRow(
            request_id=20,
            token_id=201,
            position=8,
            draft_depth=1,
            embedding_seed_ptr=0x2000,
            embedding_hidden_size=8,
            parent_token_id=200,
            parent_position=7,
        ),
        Qwen35GGUFMTPDraftRow(
            request_id=10,
            token_id=102,
            position=7,
            draft_depth=2,
            embedding_seed_ptr=0x3000,
            embedding_hidden_size=8,
            parent_token_id=101,
            parent_position=6,
        ),
        Qwen35GGUFMTPDraftRow(
            request_id=20,
            token_id=202,
            position=9,
            draft_depth=2,
            embedding_seed_ptr=0x4000,
            embedding_hidden_size=8,
            parent_token_id=201,
            parent_position=8,
        ),
    )
    batch = Qwen35GGUFMTPDraftBatch(rows=rows)

    shared = batch.to_shared_draft_batch()
    verify = batch.to_target_verify_batch()

    assert isinstance(shared, DraftBatch)
    assert isinstance(verify, TargetVerifyBatch)
    assert shared.request_ids == (10, 20)
    assert shared.candidate_tokens == (101, 201, 102, 202)
    assert shared.parent_positions == (5, 7, 6, 8)
    assert shared.draft_depths == (1, 1, 2, 2)
    assert shared.row_to_request == (10, 20, 10, 20)
    assert shared.tree_parents == (-1, -1, 0, 1)
    assert shared.active_mask == (True, True, True, True)
    assert verify.tokens == (100, 200, 101, 201, 102, 202)
    assert verify.positions == (5, 7, 6, 8, 7, 9)
    assert verify.parent_rows == (-1, -1, 0, 1, 2, 3)
    assert verify.tree_shape == (0, 0, 1, 2)


def test_gguf_mtp_draft_batch_rejects_missing_shared_parent_row() -> None:
    batch = Qwen35GGUFMTPDraftBatch(
        rows=(
            Qwen35GGUFMTPDraftRow(
                request_id=10,
                token_id=102,
                position=7,
                draft_depth=2,
                embedding_seed_ptr=0x3000,
                embedding_hidden_size=8,
                parent_token_id=101,
                parent_position=6,
            ),
        )
    )

    with pytest.raises(ValueError, match="parent before child"):
        batch.to_shared_draft_batch()
    with pytest.raises(ValueError, match="depth-1 root row"):
        batch.to_target_verify_batch()


def test_gguf_mtp_context_builds_draft_execution_plan_from_logits() -> None:
    register_cpu_reference_kernels(replace=True)
    context = Qwen35GGUFMTPContext(target_session=object())
    context.capture_pending_seed(_Seed(token_id=10, position=5, hidden_ptr=0x1000))
    logits = np.asarray([[0.2, 4.0, 4.0]], dtype=np.float32)

    plan = context.build_draft_execution_plan_from_logits(
        request_id=9,
        logits=logits,
        top_k=2,
        block_size=4,
    )

    assert isinstance(plan, Qwen35GGUFMTPDraftExecutionPlan)
    assert plan.proposed_token_ids == (1,)
    assert plan.proposal.top_k_token_ids == ((1, 2),)
    assert plan.kv_live_spans.token_positions == (6,)
    assert plan.cpu_reference_kwargs() == {
        "append": {
            "kv_base_offsets": [[0, 1]],
            "kv_live_counts": [6],
            "kv_token_positions": [6],
            "kv_evict_mask": None,
            "block_size": 4,
        },
        "decode": {
            "kv_base_offsets": [[0, 1]],
            "kv_live_counts": [7],
            "kv_token_positions": [6],
            "kv_evict_mask": None,
            "block_size": 4,
        },
    }
    verify = plan.to_target_verify_batch()
    transaction = plan.target_verify_transaction(12)
    assert verify.tokens == (10, 1)
    assert verify.positions == (5, 6)
    assert verify.parent_rows == (-1, 0)
    assert isinstance(transaction, KVTransaction)
    assert transaction.transaction_id == 12
    assert transaction.request_ids == (9,)
    assert transaction.draft_rows == 1
    assert transaction.role == "verify_chain"
    assert transaction.candidate_counts == (1,)
    assert not transaction.committed
    assert not transaction.rolled_back
    assert plan.as_dict()["proposed_token_ids"] == [1]
    assert plan.as_dict()["proposal"]["topk_kernel"] == [
        "cpu_reference",
        "mtp_draft_topk",
        "w4_gguf",
        "full_vocab_d2h",
    ]


def test_gguf_mtp_execution_plan_builds_target_accept_summary_from_top1() -> None:
    context = Qwen35GGUFMTPContext(target_session=object())
    seeds = (
        Qwen35GGUFMTPSeedRow(token_id=10, position=5, hidden_ptr=0x1000, hidden_size=8),
        Qwen35GGUFMTPSeedRow(token_id=1, position=6, hidden_ptr=0x2000, hidden_size=8),
    )
    batch = context.build_draft_batch(request_id=7, token_ids=(1, 2), seed_rows=seeds)
    proposal = Qwen35GGUFMTPDraftProposal(
        batch=batch,
        top_k_token_ids=((1,), (2,)),
        top_k_logits=((4.0,), (3.0,)),
    )
    plan = Qwen35GGUFMTPDraftExecutionPlan(
        proposal=proposal,
        kv_live_spans=context.build_kvlivespans_plan(batch, block_size=4),
    )

    partial = plan.target_accept_summary_from_top1((1, 9, 77), transaction_id=12)
    full_budgeted = plan.target_accept_summary_from_top1((1, 2, 77), remaining_decode=(2,))

    assert isinstance(partial, TargetAcceptSummary)
    assert partial.request_ids == (7,)
    assert partial.accepted_counts == (1,)
    assert partial.accepted_tokens == ((1,),)
    assert partial.commit_rows == (1,)
    assert partial.commit_tokens == (1,)
    assert partial.commit_positions == (6,)
    assert partial.full_accept == (False,)
    assert partial.next_tokens == (9,)
    assert partial.candidate_counts == (2,)
    assert partial.transaction_id == 12
    assert partial.draft_depth == 2
    assert partial.tree_shape == (0, 1)
    assert full_budgeted.accepted_counts == (2,)
    assert full_budgeted.commit_rows == (2,)
    assert full_budgeted.commit_tokens == (2,)
    assert full_budgeted.commit_positions == (7,)
    assert full_budgeted.full_accept == (True,)
    assert full_budgeted.next_tokens == (None,)


def test_gguf_mtp_execution_plan_builds_target_commit_plan_from_summary() -> None:
    context = Qwen35GGUFMTPContext(target_session=object())
    seeds = (
        Qwen35GGUFMTPSeedRow(token_id=10, position=5, hidden_ptr=0x1000, hidden_size=8),
        Qwen35GGUFMTPSeedRow(token_id=1, position=6, hidden_ptr=0x2000, hidden_size=8),
    )
    batch = context.build_draft_batch(request_id=7, token_ids=(1,), seed_rows=seeds[:1])
    proposal = Qwen35GGUFMTPDraftProposal(
        batch=batch,
        top_k_token_ids=((1,),),
        top_k_logits=((4.0,),),
    )
    plan = Qwen35GGUFMTPDraftExecutionPlan(
        proposal=proposal,
        kv_live_spans=context.build_kvlivespans_plan(batch, block_size=4),
    )
    summary = plan.target_accept_summary_from_top1((1, 99), transaction_id=12, remaining_decode=(1,))
    transaction = KVTransaction(
        transaction_id=12,
        request_ids=(7,),
        draft_rows=1,
        role="verify_chain",
        candidate_counts=(1,),
    )

    commit = plan.target_commit_plan_from_summary(summary, transaction)

    assert isinstance(commit, TargetCommitPlan)
    assert commit.transaction_id == 12
    assert commit.request_ids == (7,)
    assert commit.accepted_counts == (1,)
    assert commit.commit_rows == (1,)
    assert commit.commit_tokens == (1,)
    assert commit.commit_positions == (6,)
    assert commit.next_tokens == (None,)
    assert commit.candidate_counts == (1,)
    assert commit.draft_depth == 1
    assert commit.tree_shape == (0,)
    assert commit.mode == "verify_chain"

    wrong_token = TargetAcceptSummary(
        request_ids=(7,),
        accepted_counts=(1,),
        accepted_tokens=((1,),),
        commit_rows=(1,),
        commit_tokens=(2,),
        commit_positions=(6,),
        full_accept=(True,),
        candidate_counts=(1,),
        transaction_id=12,
        draft_depth=1,
        tree_shape=(0,),
    )
    with pytest.raises(ValueError, match="token/position"):
        plan.target_commit_plan_from_summary(wrong_token, transaction)
    wrong_txn = KVTransaction(
        transaction_id=13,
        request_ids=(7,),
        draft_rows=1,
        role="verify_chain",
        candidate_counts=(1,),
    )
    with pytest.raises(ValueError, match="transaction_id"):
        plan.target_commit_plan_from_summary(summary, wrong_txn)


def test_gguf_mtp_execution_plan_builds_target_commit_plan_from_top1() -> None:
    context = Qwen35GGUFMTPContext(target_session=object())
    seeds = (
        Qwen35GGUFMTPSeedRow(token_id=10, position=5, hidden_ptr=0x1000, hidden_size=8),
        Qwen35GGUFMTPSeedRow(token_id=1, position=6, hidden_ptr=0x2000, hidden_size=8),
    )
    batch = context.build_draft_batch(request_id=7, token_ids=(1, 2), seed_rows=seeds)
    proposal = Qwen35GGUFMTPDraftProposal(
        batch=batch,
        top_k_token_ids=((1,), (2,)),
        top_k_logits=((4.0,), (3.0,)),
    )
    plan = Qwen35GGUFMTPDraftExecutionPlan(
        proposal=proposal,
        kv_live_spans=context.build_kvlivespans_plan(batch, block_size=4),
    )

    partial = plan.target_commit_plan_from_top1((1, 9, 77), transaction_id=12)
    full_budgeted = plan.target_commit_plan_from_top1(
        (1, 2, 77),
        transaction_id=13,
        remaining_decode=(2,),
    )

    assert partial.transaction_id == 12
    assert partial.request_ids == (7,)
    assert partial.accepted_counts == (1,)
    assert partial.commit_rows == (1,)
    assert partial.commit_tokens == (1,)
    assert partial.commit_positions == (6,)
    assert partial.next_tokens == (9,)
    assert partial.candidate_counts == (2,)
    assert partial.draft_depth == 2
    assert partial.tree_shape == (0, 1)
    assert partial.mode == "verify_chain"
    assert full_budgeted.transaction_id == 13
    assert full_budgeted.accepted_counts == (2,)
    assert full_budgeted.commit_rows == (2,)
    assert full_budgeted.commit_tokens == (2,)
    assert full_budgeted.commit_positions == (7,)
    assert full_budgeted.next_tokens == (None,)
    assert full_budgeted.candidate_counts == (2,)


def test_gguf_mtp_context_accepts_target_top1_with_commit_plan_reseed() -> None:
    context = Qwen35GGUFMTPContext(target_session=object())
    seeds = (
        Qwen35GGUFMTPSeedRow(token_id=10, position=5, hidden_ptr=0x1000, hidden_size=8),
        Qwen35GGUFMTPSeedRow(token_id=1, position=6, hidden_ptr=0x2000, hidden_size=8),
        Qwen35GGUFMTPSeedRow(token_id=2, position=7, hidden_ptr=0x3000, hidden_size=8),
    )
    batch = context.build_draft_batch(request_id=7, token_ids=(1, 2), seed_rows=seeds[:2])
    proposal = Qwen35GGUFMTPDraftProposal(
        batch=batch,
        top_k_token_ids=((1,), (2,)),
        top_k_logits=((4.0,), (3.0,)),
    )
    plan = Qwen35GGUFMTPDraftExecutionPlan(
        proposal=proposal,
        kv_live_spans=context.build_kvlivespans_plan(batch, block_size=4),
    )
    context.record_verify_seeds(seeds)

    partial_step = context.accept_target_top1(
        plan,
        (1, 9, 77),
        transaction_id=12,
        request_id=7,
    )
    partial_commit, partial_seed = partial_step
    full_step = context.accept_target_top1(
        plan,
        (1, 2, 77),
        transaction_id=13,
        remaining_decode=(2,),
        request_id=7,
    )
    full_commit, full_seed = full_step

    assert isinstance(partial_step, Qwen35GGUFMTPAcceptStep)
    assert partial_step.as_dict()["transaction_id"] == 12
    assert partial_step.as_dict()["request_ids"] == [7]
    assert partial_step.as_dict()["accepted_counts"] == [1]
    assert partial_step.as_dict()["commit_rows"] == [1]
    assert partial_step.as_dict()["next_tokens"] == [9]
    assert partial_step.as_dict()["candidate_counts"] == [2]
    assert partial_step.as_dict()["tree_shape"] == [0, 1]
    assert partial_step.as_dict()["reseed"] == seeds[1].as_dict()
    assert partial_commit.accepted_counts == (1,)
    assert partial_commit.commit_rows == (1,)
    assert partial_commit.next_tokens == (9,)
    assert partial_seed == seeds[1]
    assert full_commit.accepted_counts == (2,)
    assert full_commit.commit_rows == (2,)
    assert full_commit.next_tokens == (None,)
    assert full_seed == seeds[2]
    assert context.pending_seed == seeds[2]

    with pytest.raises(ValueError, match="request_id"):
        context.accept_target_top1(plan, (1, 9, 77), transaction_id=14, request_id=8)


def test_gguf_mtp_accept_step_metrics_aggregate_denominators() -> None:
    context = Qwen35GGUFMTPContext(target_session=object())
    seeds = (
        Qwen35GGUFMTPSeedRow(token_id=10, position=5, hidden_ptr=0x1000, hidden_size=8),
        Qwen35GGUFMTPSeedRow(token_id=1, position=6, hidden_ptr=0x2000, hidden_size=8),
        Qwen35GGUFMTPSeedRow(token_id=2, position=7, hidden_ptr=0x3000, hidden_size=8),
    )
    batch = context.build_draft_batch(request_id=7, token_ids=(1, 2), seed_rows=seeds[:2])
    proposal = Qwen35GGUFMTPDraftProposal(
        batch=batch,
        top_k_token_ids=((1,), (2,)),
        top_k_logits=((4.0,), (3.0,)),
    )
    plan = Qwen35GGUFMTPDraftExecutionPlan(
        proposal=proposal,
        kv_live_spans=context.build_kvlivespans_plan(batch, block_size=4),
    )
    context.record_verify_seeds(seeds)
    partial = context.accept_target_top1(plan, (1, 9, 77), transaction_id=12, request_id=7)
    full_budgeted = context.accept_target_top1(
        plan,
        (1, 2, 77),
        transaction_id=13,
        remaining_decode=(2,),
        request_id=7,
    )

    metrics = Qwen35GGUFMTPAcceptStepMetrics.from_steps(
        (partial, full_budgeted),
        output_token_count=5,
    )

    assert metrics.cycle_count == 2
    assert metrics.candidate_budget == 2
    assert metrics.budget_label == "B2"
    assert metrics.draft_token_count == 4
    assert metrics.accepted_token_count == 3
    assert metrics.step_transaction_ids == (12, 13)
    assert metrics.step_candidate_token_counts == (2, 2)
    assert metrics.step_accepted_token_counts == (1, 2)
    assert metrics.accepted_per_draft == 0.75
    assert metrics.accepted_per_output == 0.6
    assert metrics.as_dict()["schema"] == 1
    assert metrics.as_dict()["kind"] == "hipengine_gguf_mtp_accept_step_metrics"
    assert metrics.as_dict()["source"] == "Qwen35GGUFMTPAcceptStepMetrics"
    assert metrics.as_dict()["result_source"] == "Qwen35GGUFMTPAcceptStep"
    assert metrics.as_dict()["candidate_budget"] == 2
    assert metrics.as_dict()["budget_label"] == "B2"
    assert metrics.as_dict()["step_transaction_ids"] == [12, 13]
    assert metrics.as_dict()["step_candidate_token_counts"] == [2, 2]
    assert metrics.as_dict()["step_accepted_token_counts"] == [1, 2]
    assert metrics.as_dict()["step_rows"] == [
        {
            "transaction_id": 12,
            "request_ids": [7],
            "candidate_token_count": 2,
            "accepted_token_count": 1,
            "candidate_counts": [2],
            "accepted_counts": [1],
        },
        {
            "transaction_id": 13,
            "request_ids": [7],
            "candidate_token_count": 2,
            "accepted_token_count": 2,
            "candidate_counts": [2],
            "accepted_counts": [2],
        },
    ]
    assert metrics.as_dict()["denominators"] == {
        "accepted_per_draft": "accepted_token_count / draft_token_count",
        "accepted_per_output": "accepted_token_count / output_token_count",
    }
    assert metrics.as_dict()["steps"][0]["accepted_counts"] == [1]
    assert metrics.as_dict()["steps"][1]["accepted_counts"] == [2]

    with pytest.raises(ValueError, match="at least one"):
        Qwen35GGUFMTPAcceptStepMetrics.from_steps((), output_token_count=1)
    with pytest.raises(ValueError, match="output_token_count"):
        Qwen35GGUFMTPAcceptStepMetrics.from_steps((partial,), output_token_count=0)
    missing_counts = TargetCommitPlan(
        transaction_id=99,
        request_ids=(7,),
        accepted_counts=(0,),
        commit_rows=(0,),
        commit_tokens=(10,),
        commit_positions=(5,),
    )
    with pytest.raises(ValueError, match="candidate_counts"):
        Qwen35GGUFMTPAcceptStepMetrics.from_steps(
            (Qwen35GGUFMTPAcceptStep(commit_plan=missing_counts, reseed=seeds[0]),),
            output_token_count=1,
        )


def test_gguf_mtp_context_accepts_top1_specs_into_metrics() -> None:
    context = Qwen35GGUFMTPContext(target_session=object())
    seeds = (
        Qwen35GGUFMTPSeedRow(token_id=10, position=5, hidden_ptr=0x1000, hidden_size=8),
        Qwen35GGUFMTPSeedRow(token_id=1, position=6, hidden_ptr=0x2000, hidden_size=8),
        Qwen35GGUFMTPSeedRow(token_id=2, position=7, hidden_ptr=0x3000, hidden_size=8),
    )
    batch = context.build_draft_batch(request_id=7, token_ids=(1, 2), seed_rows=seeds[:2])
    proposal = Qwen35GGUFMTPDraftProposal(
        batch=batch,
        top_k_token_ids=((1,), (2,)),
        top_k_logits=((4.0,), (3.0,)),
    )
    plan = Qwen35GGUFMTPDraftExecutionPlan(
        proposal=proposal,
        kv_live_spans=context.build_kvlivespans_plan(batch, block_size=4),
    )
    specs = (
        Qwen35GGUFMTPTop1AcceptSpec(
            plan=plan,
            target_top1=[1, 9, 77],
            transaction_id=12,
            verify_seeds=seeds,
            request_id=7,
        ),
        Qwen35GGUFMTPTop1AcceptSpec(
            plan=plan,
            target_top1=(1, 2, 77),
            transaction_id=13,
            verify_seeds=seeds,
            remaining_decode=[2],
            request_id=7,
        ),
    )

    metrics = context.accept_target_top1_metrics(specs, output_token_count=5)

    assert specs[0].target_top1 == (1, 9, 77)
    assert specs[1].remaining_decode == (2,)
    assert metrics.accepted_token_count == 3
    assert metrics.draft_token_count == 4
    assert metrics.candidate_budget == 2
    assert metrics.budget_label == "B2"
    assert metrics.as_dict()["candidate_budget"] == 2
    assert metrics.as_dict()["budget_label"] == "B2"
    assert metrics.as_dict()["step_transaction_ids"] == [12, 13]
    assert metrics.as_dict()["step_candidate_token_counts"] == [2, 2]
    assert metrics.as_dict()["step_accepted_token_counts"] == [1, 2]
    assert metrics.as_dict()["steps"][0]["transaction_id"] == 12
    assert metrics.as_dict()["steps"][1]["transaction_id"] == 13
    assert context.pending_seed == seeds[2]

    with pytest.raises(ValueError, match="target_top1"):
        Qwen35GGUFMTPTop1AcceptSpec(plan=plan, target_top1=(), transaction_id=0)
    with pytest.raises(ValueError, match="transaction_id"):
        Qwen35GGUFMTPTop1AcceptSpec(plan=plan, target_top1=(1,), transaction_id=-1)


def test_gguf_mtp_context_applies_target_commit_plan_reseed_rule() -> None:
    context = Qwen35GGUFMTPContext(target_session=object())
    seeds = (
        Qwen35GGUFMTPSeedRow(token_id=10, position=5, hidden_ptr=0x1000, hidden_size=8),
        Qwen35GGUFMTPSeedRow(token_id=1, position=6, hidden_ptr=0x2000, hidden_size=8),
    )
    batch = context.build_draft_batch(request_id=7, token_ids=(1,), seed_rows=seeds[:1])
    proposal = Qwen35GGUFMTPDraftProposal(
        batch=batch,
        top_k_token_ids=((1,),),
        top_k_logits=((4.0,),),
    )
    plan = Qwen35GGUFMTPDraftExecutionPlan(
        proposal=proposal,
        kv_live_spans=context.build_kvlivespans_plan(batch, block_size=4),
    )
    summary = plan.target_accept_summary_from_top1((1, 99), transaction_id=12, remaining_decode=(1,))
    transaction = KVTransaction(
        transaction_id=12,
        request_ids=(7,),
        draft_rows=1,
        role="verify_chain",
        candidate_counts=(1,),
    )
    commit = plan.target_commit_plan_from_summary(summary, transaction)
    context.record_verify_seeds(seeds)

    assert context.accept_target_commit_plan(commit, request_id=7) == seeds[1]
    assert context.pending_seed == seeds[1]

    with pytest.raises(ValueError, match="request_id"):
        context.accept_target_commit_plan(commit, request_id=8)
    with pytest.raises(RuntimeError, match="candidate rows plus"):
        Qwen35GGUFMTPContext(target_session=object()).accept_target_commit_plan(commit)
    multi = TargetCommitPlan(
        transaction_id=12,
        request_ids=(7, 8),
        accepted_counts=(0, 0),
        commit_rows=(0, 1),
        commit_tokens=(10, 20),
        commit_positions=(5, 6),
        candidate_counts=(0, 0),
    )
    with pytest.raises(ValueError, match="one request"):
        context.accept_target_commit_plan(multi)


def test_gguf_mtp_context_applies_target_accept_summary_reseed_rule() -> None:
    context = Qwen35GGUFMTPContext(target_session=object())
    seeds = (
        Qwen35GGUFMTPSeedRow(token_id=10, position=5, hidden_ptr=0x1000, hidden_size=8),
        Qwen35GGUFMTPSeedRow(token_id=1, position=6, hidden_ptr=0x2000, hidden_size=8),
        Qwen35GGUFMTPSeedRow(token_id=2, position=7, hidden_ptr=0x3000, hidden_size=8),
    )
    batch = context.build_draft_batch(request_id=7, token_ids=(1, 2), seed_rows=seeds[:2])
    proposal = Qwen35GGUFMTPDraftProposal(
        batch=batch,
        top_k_token_ids=((1,), (2,)),
        top_k_logits=((4.0,), (3.0,)),
    )
    plan = Qwen35GGUFMTPDraftExecutionPlan(
        proposal=proposal,
        kv_live_spans=context.build_kvlivespans_plan(batch, block_size=4),
    )
    context.record_verify_seeds(seeds)

    partial = plan.target_accept_summary_from_top1((1, 9, 77), transaction_id=12)
    full_budgeted = plan.target_accept_summary_from_top1((1, 2, 77), remaining_decode=(2,))

    assert context.accept_target_summary(partial, request_id=7) == seeds[1]
    assert context.pending_seed == seeds[1]
    assert context.accept_target_summary(full_budgeted, request_id=7) == seeds[2]
    assert context.pending_seed == seeds[2]

    with pytest.raises(ValueError, match="request_id"):
        context.accept_target_summary(partial, request_id=8)
    with pytest.raises(RuntimeError, match="candidate rows plus"):
        Qwen35GGUFMTPContext(target_session=object()).accept_target_summary(partial)
    multi = TargetAcceptSummary(
        request_ids=(7, 8),
        accepted_counts=(0, 0),
        accepted_tokens=((), ()),
        commit_rows=(0, 1),
        commit_tokens=(10, 20),
        commit_positions=(5, 6),
        full_accept=(False, False),
        candidate_counts=(0, 0),
    )
    with pytest.raises(ValueError, match="one request"):
        context.accept_target_summary(multi)


def test_gguf_mtp_execution_plan_validates_positions_match_spans() -> None:
    context = Qwen35GGUFMTPContext(target_session=object())
    context.capture_pending_seed(_Seed(token_id=10, position=5, hidden_ptr=0x1000))
    batch = context.build_b1_draft_batch(request_id=0, token_id=20)
    proposal = Qwen35GGUFMTPDraftProposal(
        batch=batch,
        top_k_token_ids=((20,),),
        top_k_logits=((1.0,),),
    )
    mismatched_spans = Qwen35GGUFMTPKVLiveSpansPlan(
        rows=1,
        block_size=4,
        logical_blocks=2,
        base_offsets=((0, 1),),
        append_live_counts=(99,),
        decode_live_counts=(100,),
        token_positions=(99,),
    )

    with pytest.raises(ValueError, match="token_positions"):
        Qwen35GGUFMTPDraftExecutionPlan(proposal=proposal, kv_live_spans=mismatched_spans)


def test_gguf_mtp_context_verifies_proposal_prefix_and_reseeds_from_mismatch_row() -> None:
    register_cpu_reference_kernels(replace=True)
    context = Qwen35GGUFMTPContext(target_session=object())
    seeds = (
        Qwen35GGUFMTPSeedRow(token_id=10, position=5, hidden_ptr=0x1000, hidden_size=8),
        Qwen35GGUFMTPSeedRow(token_id=11, position=6, hidden_ptr=0x2000, hidden_size=8),
        Qwen35GGUFMTPSeedRow(token_id=12, position=7, hidden_ptr=0x3000, hidden_size=8),
    )
    logits = np.asarray(
        [
            [0.0, 4.0, 1.0],
            [3.0, 2.0, 5.0],
        ],
        dtype=np.float32,
    )
    proposal = context.build_draft_proposal_from_logits(request_id=0, logits=logits, seed_rows=seeds[:2], top_k=1)

    result = context.verify_draft_proposal(
        proposal,
        target_token_ids=(1, 9),
        verify_seeds=seeds,
    )

    assert isinstance(result, Qwen35GGUFMTPVerificationResult)
    assert result.n_accepted == 1
    assert result.accepted_token_ids == (1,)
    assert result.first_mismatch_index == 1
    assert result.rejected_proposal_token_id == 2
    assert result.target_token_id_at_mismatch == 9
    assert result.reseed == seeds[1]
    assert context.pending_seed == seeds[1]
    assert result.as_dict()["accepted_per_draft"] == 0.5


def test_gguf_mtp_context_verifies_full_proposal_acceptance_from_execution_plan() -> None:
    register_cpu_reference_kernels(replace=True)
    context = Qwen35GGUFMTPContext(target_session=object())
    seeds = (
        Qwen35GGUFMTPSeedRow(token_id=10, position=5, hidden_ptr=0x1000, hidden_size=8),
        Qwen35GGUFMTPSeedRow(token_id=11, position=6, hidden_ptr=0x2000, hidden_size=8),
        Qwen35GGUFMTPSeedRow(token_id=12, position=7, hidden_ptr=0x3000, hidden_size=8),
    )
    logits = np.asarray(
        [
            [0.0, 4.0, 1.0],
            [3.0, 2.0, 5.0],
        ],
        dtype=np.float32,
    )
    plan = context.build_draft_execution_plan_from_logits(
        request_id=0,
        logits=logits,
        seed_rows=seeds[:2],
        top_k=1,
    )

    result = context.verify_draft_proposal(plan, target_token_ids=(1, 2), verify_seeds=seeds)

    assert result.n_accepted == 2
    assert result.accepted_token_ids == (1, 2)
    assert result.first_mismatch_index is None
    assert result.rejected_proposal_token_id is None
    assert result.target_token_id_at_mismatch is None
    assert result.reseed == seeds[2]
    assert result.as_dict()["accepted_per_draft"] == 1.0


def test_gguf_mtp_runtime_kernel_plan_reports_oracles_and_missing_native_keys() -> None:
    register_cpu_reference_kernels(replace=True)
    register_sampler_kernels(replace=True)

    plan = Qwen35GGUFMTPRuntimeKernelPlan.from_registry(backend="hip_gfx1100")
    payload = plan.as_dict()

    assert plan.exactness_oracles_ready is True
    assert plan.native_runtime_kernels_ready is False
    assert plan.optimization_kernels_ready is True
    assert payload["missing_exactness_oracle_keys"] == []
    assert payload["missing_native_runtime_keys"] == [
        ["hip_gfx1100", "mtp_nextn_layer", "w4_gguf", "qwen35_dense_logits"],
        ["hip_gfx1100", "paged_kv_write", "w4_gguf", "mixed_bf16_spans"],
        ["hip_gfx1100", "paged_attn_decode", "w4_gguf", "bf16_context_spans"],
    ]
    assert payload["missing_optimization_keys"] == []
    assert [item["name"] for item in payload["checks"]] == [
        "cpu_nextn_oracle",
        "draft_topk_fallback_oracle",
        "native_nextn_runtime",
        "native_nextn_paged_kv_write",
        "native_nextn_paged_attn_decode",
        "native_draft_topk_device",
    ]


def test_gguf_mtp_runtime_kernel_plan_validates_inputs() -> None:
    with pytest.raises(ValueError, match="backend"):
        Qwen35GGUFMTPRuntimeKernelPlan.from_registry(backend="")
    with pytest.raises(ValueError, match="four-axis"):
        Qwen35GGUFMTPRuntimeKernelPlan.from_registry(
            backend="hip_gfx1100",
            draft_topk_kernel=("cpu_reference", "mtp_draft_topk", "w4_gguf"),  # type: ignore[arg-type]
        )


def test_gguf_mtp_verification_metrics_aggregate_denominators() -> None:
    accepted_one = Qwen35GGUFMTPVerificationResult(
        proposed_token_ids=(1, 2),
        target_token_ids=(1, 9),
        n_accepted=1,
        first_mismatch_index=1,
        rejected_proposal_token_id=2,
        target_token_id_at_mismatch=9,
        verify_seed_count=3,
        reseed=Qwen35GGUFMTPSeedRow(token_id=11, position=6, hidden_ptr=0x2000, hidden_size=8),
    )
    accepted_two = Qwen35GGUFMTPVerificationResult(
        proposed_token_ids=(3, 4),
        target_token_ids=(3, 4),
        n_accepted=2,
        verify_seed_count=3,
        reseed=Qwen35GGUFMTPSeedRow(token_id=13, position=8, hidden_ptr=0x4000, hidden_size=8),
    )

    metrics = Qwen35GGUFMTPVerificationMetrics.from_results(
        (accepted_one, accepted_two),
        output_token_count=5,
    )

    assert metrics.cycle_count == 2
    assert metrics.draft_token_count == 4
    assert metrics.accepted_token_count == 3
    assert metrics.accepted_per_draft == 0.75
    assert metrics.accepted_per_output == 0.6
    assert metrics.as_dict()["accepted_per_output"] == 0.6
    assert metrics.as_dict()["denominators"] == {
        "accepted_per_draft": "accepted_token_count / draft_token_count",
        "accepted_per_output": "accepted_token_count / output_token_count",
    }
    assert metrics.as_dict()["results"][0]["first_mismatch_index"] == 1


def test_gguf_mtp_verification_metrics_validate_denominators() -> None:
    result = Qwen35GGUFMTPVerificationResult(
        proposed_token_ids=(1,),
        target_token_ids=(1,),
        n_accepted=1,
        verify_seed_count=2,
        reseed=Qwen35GGUFMTPSeedRow(token_id=2, position=1, hidden_ptr=0x2000, hidden_size=8),
    )

    with pytest.raises(ValueError, match="at least one"):
        Qwen35GGUFMTPVerificationMetrics.from_results((), output_token_count=1)
    with pytest.raises(ValueError, match="output_token_count"):
        Qwen35GGUFMTPVerificationMetrics.from_results((result,), output_token_count=0)
    with pytest.raises(ValueError, match="visible output"):
        Qwen35GGUFMTPVerificationMetrics.from_results((result, result), output_token_count=1)


def test_gguf_mtp_performance_readiness_accepts_fully_ready_inputs() -> None:
    readiness = Qwen35GGUFMTPPerformanceReadiness.from_gate_inputs(
        parity_precheck=True,
        draft_budget_precheck=True,
        draft_sampling_contract_precheck=True,
        hidden_seed_contract_precheck=True,
        exactness_gate="passed",
        kvlivespans_paged_cache_smoke=True,
        llamacpp_trace_budget_coverage=GGUF_MTP_FULL_TRACE_BUDGET_COVERAGE,
        accepted_per_draft_status=GGUF_MTP_ACCEPTED_DRAFT_COMPARABLE,
        accepted_per_output_status=GGUF_MTP_ACCEPTED_OUTPUT_COMPARABLE,
        native_runtime_kernels_ready=True,
        optimization_kernels_ready=True,
        metrics_contract_status=GGUF_MTP_METRICS_CONTRACT_READY,
    )

    assert readiness.ready is True
    assert readiness.blockers == ()
    assert readiness.as_dict() == {"ready": True, "blockers": []}


def test_gguf_mtp_performance_readiness_reports_ordered_blockers() -> None:
    readiness = Qwen35GGUFMTPPerformanceReadiness.from_gate_inputs(
        parity_precheck=False,
        draft_budget_precheck=False,
        draft_sampling_contract_precheck=False,
        hidden_seed_contract_precheck=False,
        exactness_gate="blocked",
        kvlivespans_paged_cache_smoke=False,
        llamacpp_trace_budget_coverage=GGUF_MTP_PARTIAL_TRACE_BUDGET_COVERAGE,
        accepted_per_draft_status=GGUF_MTP_ACCEPTED_DRAFT_NOT_COMPARABLE_DEBUG_TRACE,
        accepted_per_output_status=GGUF_MTP_ACCEPTED_OUTPUT_NOT_COMPARABLE_DEBUG_TRACE,
        native_runtime_kernels_ready=False,
        optimization_kernels_ready=False,
        metrics_contract_status="not_run",
    )

    assert readiness.ready is False
    assert readiness.blockers == (
        "parity_precheck_failed",
        "draft_budget_precheck_failed",
        "draft_sampling_contract_precheck_failed",
        "hidden_seed_contract_precheck_failed",
        "exactness_gate_failed",
        "kvlivespans_paged_cache_smoke_failed",
        "partial_llamacpp_trace_budget_coverage",
        "accepted_draft_denominator_not_comparable",
        "accepted_output_denominator_not_comparable",
        "native_runtime_kernels_missing",
        "optimization_kernels_missing",
        "hipengine_metrics_not_ready",
    )


def test_gguf_mtp_performance_readiness_blocks_missing_optimization_kernels() -> None:
    readiness = Qwen35GGUFMTPPerformanceReadiness.from_gate_inputs(
        parity_precheck=True,
        draft_budget_precheck=True,
        draft_sampling_contract_precheck=True,
        hidden_seed_contract_precheck=True,
        exactness_gate="passed",
        kvlivespans_paged_cache_smoke=True,
        llamacpp_trace_budget_coverage=GGUF_MTP_FULL_TRACE_BUDGET_COVERAGE,
        accepted_per_draft_status=GGUF_MTP_ACCEPTED_DRAFT_COMPARABLE,
        accepted_per_output_status=GGUF_MTP_ACCEPTED_OUTPUT_COMPARABLE,
        native_runtime_kernels_ready=True,
        optimization_kernels_ready=False,
        metrics_contract_status=GGUF_MTP_METRICS_CONTRACT_READY,
    )

    assert readiness.ready is False
    assert readiness.blockers == ("optimization_kernels_missing",)


def test_gguf_mtp_performance_readiness_blocks_failed_kvlivespans_smoke() -> None:
    readiness = Qwen35GGUFMTPPerformanceReadiness.from_gate_inputs(
        parity_precheck=True,
        draft_budget_precheck=True,
        draft_sampling_contract_precheck=True,
        hidden_seed_contract_precheck=True,
        exactness_gate="passed",
        kvlivespans_paged_cache_smoke=False,
        llamacpp_trace_budget_coverage=GGUF_MTP_FULL_TRACE_BUDGET_COVERAGE,
        accepted_per_draft_status=GGUF_MTP_ACCEPTED_DRAFT_COMPARABLE,
        accepted_per_output_status=GGUF_MTP_ACCEPTED_OUTPUT_COMPARABLE,
        native_runtime_kernels_ready=True,
        optimization_kernels_ready=True,
        metrics_contract_status=GGUF_MTP_METRICS_CONTRACT_READY,
    )

    assert readiness.ready is False
    assert readiness.blockers == ("kvlivespans_paged_cache_smoke_failed",)


def test_gguf_mtp_performance_readiness_blocks_noncomparable_accepted_draft() -> None:
    readiness = Qwen35GGUFMTPPerformanceReadiness.from_gate_inputs(
        parity_precheck=True,
        draft_budget_precheck=True,
        draft_sampling_contract_precheck=True,
        hidden_seed_contract_precheck=True,
        exactness_gate="passed",
        kvlivespans_paged_cache_smoke=True,
        llamacpp_trace_budget_coverage=GGUF_MTP_FULL_TRACE_BUDGET_COVERAGE,
        accepted_per_draft_status=GGUF_MTP_ACCEPTED_DRAFT_NOT_COMPARABLE_DEBUG_TRACE,
        accepted_per_output_status=GGUF_MTP_ACCEPTED_OUTPUT_COMPARABLE,
        native_runtime_kernels_ready=True,
        optimization_kernels_ready=True,
        metrics_contract_status=GGUF_MTP_METRICS_CONTRACT_READY,
    )

    assert readiness.ready is False
    assert readiness.blockers == ("accepted_draft_denominator_not_comparable",)


def test_gguf_mtp_context_rejects_incomplete_verification_inputs() -> None:
    register_cpu_reference_kernels(replace=True)
    context = Qwen35GGUFMTPContext(target_session=object())
    seeds = (
        Qwen35GGUFMTPSeedRow(token_id=10, position=5, hidden_ptr=0x1000, hidden_size=8),
        Qwen35GGUFMTPSeedRow(token_id=11, position=6, hidden_ptr=0x2000, hidden_size=8),
    )
    logits = np.asarray([[0.0, 4.0, 1.0], [3.0, 2.0, 5.0]], dtype=np.float32)
    proposal = context.build_draft_proposal_from_logits(request_id=0, logits=logits, seed_rows=seeds, top_k=1)

    with pytest.raises(ValueError, match="target_token_ids"):
        context.verify_draft_proposal(proposal, target_token_ids=(1,), verify_seeds=seeds)
    with pytest.raises(ValueError, match="verify_seeds"):
        context.verify_draft_proposal(proposal, target_token_ids=(1, 2), verify_seeds=seeds)


def test_gguf_mtp_context_builds_metadata_only_kvlivespans_plan_for_draft_batch() -> None:
    context = Qwen35GGUFMTPContext(target_session=object())
    seeds = (
        Qwen35GGUFMTPSeedRow(token_id=10, position=5, hidden_ptr=0x1000, hidden_size=8),
        Qwen35GGUFMTPSeedRow(token_id=11, position=6, hidden_ptr=0x2000, hidden_size=8),
    )
    batch = context.build_draft_batch(request_id=4, token_ids=(101, 102), seed_rows=seeds)

    plan = context.build_kvlivespans_plan(batch, block_size=4)

    assert isinstance(plan, Qwen35GGUFMTPKVLiveSpansPlan)
    assert plan.as_dict() == {
        "spans_mode": "uniform",
        "storage_dtype": "bf16",
        "rows": 2,
        "block_size": 4,
        "logical_blocks": 2,
        "base_offsets": [[0, 1], [0, 1]],
        "append_live_counts": [6, 7],
        "decode_live_counts": [7, 8],
        "token_positions": [6, 7],
        "evict_mask": None,
    }
    assert plan.cpu_reference_kwargs(role="append") == {
        "kv_base_offsets": [[0, 1], [0, 1]],
        "kv_live_counts": [6, 7],
        "kv_token_positions": [6, 7],
        "kv_evict_mask": None,
        "block_size": 4,
    }
    assert plan.cpu_reference_kwargs(role="decode") == {
        "kv_base_offsets": [[0, 1], [0, 1]],
        "kv_live_counts": [7, 8],
        "kv_token_positions": [6, 7],
        "kv_evict_mask": None,
        "block_size": 4,
    }


def test_gguf_mtp_kvlivespans_plan_validates_abi_fields() -> None:
    context = Qwen35GGUFMTPContext(target_session=object())
    context.capture_pending_seed(_Seed(token_id=10, position=5, hidden_ptr=0x1000))
    batch = context.build_b1_draft_batch(request_id=0, token_id=20)

    with pytest.raises(ValueError, match="block_size"):
        context.build_kvlivespans_plan(batch, block_size=0)
    with pytest.raises(ValueError, match="base_offsets"):
        Qwen35GGUFMTPKVLiveSpansPlan(
            rows=1,
            block_size=4,
            logical_blocks=2,
            base_offsets=((0,),),
            append_live_counts=(6,),
            decode_live_counts=(7,),
            token_positions=(6,),
        )
    with pytest.raises(ValueError, match="role"):
        context.build_kvlivespans_plan(batch).cpu_reference_kwargs(role="prefill")


def test_gguf_mtp_context_requires_explicit_seed_rows_for_multi_depth_batch() -> None:
    context = Qwen35GGUFMTPContext(target_session=object())
    context.capture_pending_seed(_Seed(token_id=10, position=5, hidden_ptr=0x1000))

    with pytest.raises(ValueError, match="multi-depth GGUF MTP draft batches require explicit seed_rows"):
        context.build_draft_batch(request_id=0, token_ids=(11, 12))
    with pytest.raises(ValueError, match="seed_rows length"):
        context.build_draft_batch(
            request_id=0,
            token_ids=(11, 12),
            seed_rows=(Qwen35GGUFMTPSeedRow(token_id=10, position=5, hidden_ptr=0x1000, hidden_size=8),),
        )
    with pytest.raises(ValueError, match="share hidden_size"):
        context.build_draft_batch(
            request_id=0,
            token_ids=(11, 12),
            seed_rows=(
                Qwen35GGUFMTPSeedRow(token_id=10, position=5, hidden_ptr=0x1000, hidden_size=8),
                Qwen35GGUFMTPSeedRow(token_id=11, position=6, hidden_ptr=0x2000, hidden_size=16),
            ),
        )


def test_gguf_mtp_context_accept_reseeds_from_verify_row_min_accepted() -> None:
    context = Qwen35GGUFMTPContext(target_session=object())
    context.capture_pending_seed(_Seed(token_id=10, position=5, hidden_ptr=0x1000))
    verify = context.record_verify_seeds(
        [
            _Seed(token_id=11, position=6, hidden_ptr=0x2000),
            _Seed(token_id=12, position=7, hidden_ptr=0x3000),
            _Seed(token_id=13, position=8, hidden_ptr=0x4000),
        ]
    )

    selected = context.accept(99)

    assert verify[2].source == "verify[2]"
    assert selected == verify[2]
    assert context.pending_seed == verify[2]
    assert context.build_b1_draft_batch(request_id=0, token_id=20).rows[0].position == 9


def test_gguf_mtp_context_rejects_unready_or_multirow_seed_contract() -> None:
    with pytest.raises(ValueError, match="ready fp32 hidden contract"):
        Qwen35GGUFMTPSeedRow.from_seed(
            _Seed(token_id=1, position=1, hidden_ptr=1, hidden_contract=_Contract(ready_for_mtp=False))
        )
    with pytest.raises(ValueError, match="one hidden seed row"):
        Qwen35GGUFMTPSeedRow.from_seed(
            _Seed(token_id=1, position=1, hidden_ptr=1, hidden_contract=_Contract(rows=2))
        )


def test_gguf_mtp_context_requires_seed_before_b1_batch_or_accept() -> None:
    context = Qwen35GGUFMTPContext(target_session=object())

    with pytest.raises(RuntimeError, match="pending GGUF MTP hidden seed"):
        context.build_b1_draft_batch(request_id=0, token_id=1)
    with pytest.raises(RuntimeError, match="record_verify_seeds"):
        context.accept(0)


def test_gguf_mtp_draft_batch_validates_embedding_seed_rows() -> None:
    with pytest.raises(ValueError, match="embedding_seed_ptr"):
        Qwen35GGUFMTPDraftRow(
            request_id=0,
            token_id=1,
            position=2,
            draft_depth=1,
            embedding_seed_ptr=0,
            embedding_hidden_size=8,
            parent_token_id=0,
            parent_position=1,
        )
    row = Qwen35GGUFMTPDraftRow(
        request_id=0,
        token_id=1,
        position=2,
        draft_depth=1,
        embedding_seed_ptr=1,
        embedding_hidden_size=8,
        parent_token_id=0,
        parent_position=1,
    )
    with pytest.raises(ValueError, match="duplicate request_id/draft_depth"):
        Qwen35GGUFMTPDraftBatch(rows=(row, row))
