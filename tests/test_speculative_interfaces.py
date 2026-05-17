from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.tensor import Tensor
from hipengine.dispatch import ActiveBatch, RequestState
from hipengine.kvcache import FixedPagedKVPolicy
from hipengine.speculative import (
    AcceptResult,
    DraftBatch,
    TargetAcceptSummary,
    TargetCommitPlan,
    TargetStateCommitBuffers,
    TargetVerifyBatch,
    TargetVerifyBuffers,
)
from scripts.qwen35_dflash_ddtree_blocker import build_payload


def _tensor(ptr: int, shape: tuple[int, ...], dtype: str) -> Tensor:
    return Tensor.from_handle(ptr, shape, dtype, Device("hip", 0))


def test_speculative_request_ids_must_be_unique() -> None:
    with pytest.raises(ValueError, match="unique"):
        DraftBatch(
            request_ids=(1, 1),
            candidate_tokens=(10,),
            parent_positions=(0,),
            draft_depths=(1,),
            row_to_request=(1,),
        )
    with pytest.raises(ValueError, match="unique"):
        AcceptResult(request_ids=(1, 1), accepted_counts=(1, 0), accepted_tokens=((10,), ()))
    with pytest.raises(ValueError, match="transaction_id"):
        AcceptResult(request_ids=(1,), accepted_counts=(1,), accepted_tokens=((10,),), transaction_id=-1)
    with pytest.raises(ValueError, match="selected_candidate_rows"):
        AcceptResult(request_ids=(1,), accepted_counts=(1,), accepted_tokens=((10,),), selected_candidate_rows=(-1,))
    with pytest.raises(ValueError, match="selected_candidate_rows"):
        AcceptResult(request_ids=(1,), accepted_counts=(1,), accepted_tokens=((10,),), selected_candidate_rows=(1, 2))
    with pytest.raises(ValueError, match="next_tokens"):
        AcceptResult(request_ids=(1,), accepted_counts=(1,), accepted_tokens=((10,),), next_tokens=(-1,))
    with pytest.raises(ValueError, match="next_tokens"):
        AcceptResult(request_ids=(1,), accepted_counts=(1,), accepted_tokens=((10,),), next_tokens=(10, 11))
    with pytest.raises(ValueError, match="unique"):
        TargetVerifyBatch(
            request_ids=(1, 1),
            tokens=(100, 200, 10),
            positions=(0, 0, 1),
            row_to_request=(1, 1, 1),
            parent_rows=(-1, -1, 0),
            root_rows=(0, 1),
            candidate_rows=(2,),
            draft_depths=(0, 0, 1),
            active_mask=(True, True, True),
        )
    with pytest.raises(ValueError, match="unique"):
        TargetCommitPlan(
            transaction_id=0,
            request_ids=(1, 1),
            accepted_counts=(1, 0),
            commit_rows=(2, 0),
            commit_tokens=(10, 100),
            commit_positions=(1, 0),
            candidate_counts=(1, 0),
        )


def test_draft_batch_and_accept_result_validate_row_metadata() -> None:
    draft = DraftBatch(
        request_ids=(1, 2),
        candidate_tokens=(10, 11, 20),
        parent_positions=(5, 6, 3),
        draft_depths=(1, 2, 1),
        row_to_request=(1, 1, 2),
        active_mask=(True, True, True),
    )
    assert draft.draft_rows == 3
    assert draft.kind == "verify_chain"

    result = AcceptResult(request_ids=(1, 2), accepted_counts=(2, 1), accepted_tokens=((10, 11), (20,)))
    assert result.accepted_tokens == ((10, 11), (20,))
    assert result.selected_candidate_rows is None
    assert result.next_tokens is None

    with pytest.raises(ValueError, match="align"):
        DraftBatch(
            request_ids=(1,),
            candidate_tokens=(10,),
            parent_positions=(),
            draft_depths=(1,),
            row_to_request=(1,),
        )
    with pytest.raises(ValueError, match="lengths"):
        AcceptResult(request_ids=(1,), accepted_counts=(2,), accepted_tokens=((10,),))


def test_target_verify_batch_materializes_root_and_candidate_rows() -> None:
    draft = DraftBatch(
        request_ids=(1, 2),
        candidate_tokens=(10, 11, 20),
        parent_positions=(5, 6, 3),
        draft_depths=(1, 2, 1),
        row_to_request=(1, 1, 2),
        mode="verify_tree",
        tree_parents=(-1, 0, -1),
        active_mask=(True, True, False),
    )

    target = TargetVerifyBatch.from_draft(draft, root_tokens=(100, 200), root_positions=(5, 3))

    assert target.rows == 5
    assert target.candidate_count == 3
    assert target.request_ids == (1, 2)
    assert target.tokens == (100, 200, 10, 11, 20)
    assert target.positions == (5, 3, 6, 7, 4)
    assert target.row_to_request == (1, 2, 1, 1, 2)
    assert target.root_rows == (0, 1)
    assert target.candidate_rows == (2, 3, 4)
    assert target.parent_rows == (-1, -1, 0, 2, 1)
    assert target.draft_depths == (0, 0, 1, 2, 1)
    assert target.active_mask == (True, True, True, True, False)
    assert target.candidate_counts == (2, 1)
    assert target.draft_depth == 2
    assert target.tree_shape == (0, 1, 0)
    assert target.mode == "verify_tree"

    chain = TargetVerifyBatch.from_draft(
        DraftBatch(
            request_ids=(7,),
            candidate_tokens=(31, 32, 33),
            parent_positions=(8, 9, 10),
            draft_depths=(1, 2, 3),
            row_to_request=(7, 7, 7),
        ),
        root_tokens=(30,),
        root_positions=(8,),
    )
    assert chain.parent_rows == (-1, 0, 1, 2)
    assert chain.positions == (8, 9, 10, 11)
    assert chain.tree_shape == (0, 1, 2)


def test_target_verify_batch_builds_graph_shape_key_from_active_batch() -> None:
    draft = DraftBatch(
        request_ids=(1, 2),
        candidate_tokens=(10, 11, 20),
        parent_positions=(5, 6, 3),
        draft_depths=(1, 2, 1),
        row_to_request=(1, 1, 2),
        mode="verify_tree",
        tree_parents=(-1, 0, -1),
    )
    target = TargetVerifyBatch.from_draft(draft, root_tokens=(100, 200), root_positions=(5, 3))
    active = ActiveBatch(2)
    active.admit(RequestState(request_id=1, prompt_tokens=(1, 2, 3, 4, 5), max_new_tokens=4, next_prompt_index=5))
    active.admit(RequestState(request_id=2, prompt_tokens=(6, 7, 8), max_new_tokens=4, next_prompt_index=3))

    key = target.shape_key(active, context_bucket_size=4, top_k=8, experts_per_token=8, replay_steps=2)

    assert key.mode.value == "verify_tree"
    assert key.active_c == 2
    assert key.context_bucket == 8
    assert key.active_mask == (True, True)
    assert key.top_k == 8
    assert key.experts_per_token == 8
    assert key.replay_steps == 2
    assert key.draft_depth == 2
    assert key.tree_shape == (0, 1, 0)


def test_target_verify_batch_projects_candidate_rows_to_work_item() -> None:
    draft = DraftBatch(
        request_ids=(1, 2),
        candidate_tokens=(10, 11, 20),
        parent_positions=(5, 6, 3),
        draft_depths=(1, 2, 1),
        row_to_request=(1, 1, 2),
        mode="verify_tree",
        tree_parents=(-1, 0, -1),
    )
    target = TargetVerifyBatch.from_draft(draft, root_tokens=(100, 200), root_positions=(5, 3))

    work = target.to_work_item()

    assert work.kind.value == "verify_tree"
    assert work.request_ids == (1, 2)
    assert work.row_to_request == (1, 1, 2)
    assert work.token_rows == ((10,), (11,), (20,))
    assert work.draft_depth == 2
    assert work.tree_parents == (0, 1, 0)


def test_target_verify_buffers_validate_device_abi() -> None:
    draft = DraftBatch(
        request_ids=(1, 2),
        candidate_tokens=(10, 11, 20),
        parent_positions=(5, 6, 3),
        draft_depths=(1, 2, 1),
        row_to_request=(1, 1, 2),
        mode="verify_tree",
        tree_parents=(-1, 0, -1),
    )
    target = TargetVerifyBatch.from_draft(draft, root_tokens=(100, 200), root_positions=(5, 3))

    buffers = TargetVerifyBuffers.for_batch(
        target,
        token_ids=_tensor(0x3000, (5,), "int32"),
        positions=_tensor(0x3100, (5,), "int32"),
        parent_rows=_tensor(0x3200, (5,), "int32"),
        draft_depths=_tensor(0x3300, (5,), "int32"),
        row_to_request=_tensor(0x3400, (5,), "int32"),
        active_mask=_tensor(0x3500, (5,), "bool"),
        target_top1=_tensor(0x3600, (5,), "int32"),
        accepted_counts=_tensor(0x3700, (2,), "int32"),
        commit_rows=_tensor(0x3800, (2,), "int32"),
        commit_tokens=_tensor(0x3900, (2,), "int32"),
        commit_positions=_tensor(0x3A00, (2,), "int32"),
        next_tokens=_tensor(0x3B00, (2,), "int32"),
        full_accept=_tensor(0x3C00, (2,), "bool"),
        committed_output_ids=_tensor(0x3D00, (2, 5), "int32"),
        committed_output_lengths=_tensor(0x3E00, (2,), "int32"),
        transaction_id=7,
    )

    assert buffers.transaction_id == 7
    assert buffers.rows == 5
    assert buffers.candidate_rows == 3
    assert buffers.candidate_counts == (2, 1)
    assert buffers.draft_depth == 2
    assert buffers.tree_shape == (0, 1, 0)
    assert buffers.request_ids == (1, 2)
    assert buffers.request_count == 2
    assert str(buffers.device) == "hip:0"
    assert buffers.next_tokens is not None
    assert buffers.next_tokens.shape == (2,)
    assert buffers.full_accept is not None
    assert buffers.full_accept.dtype == DType.BOOL
    assert buffers.committed_output_ids is not None
    assert buffers.committed_output_ids.shape == (2, 5)
    assert buffers.committed_output_lengths is not None
    assert buffers.committed_output_lengths.shape == (2,)
    assert buffers.mode == "verify_tree"

    with pytest.raises(ValueError, match="transaction_id"):
        replace(buffers, transaction_id=-1)
    with pytest.raises(ValueError, match="candidate_counts"):
        replace(buffers, candidate_counts=(3, 1))
    with pytest.raises(ValueError, match="draft_depth"):
        replace(buffers, draft_depth=-1)
    with pytest.raises(ValueError, match="tree_shape"):
        replace(buffers, tree_shape=(0, 1))
    with pytest.raises(ValueError, match="summary tensors"):
        replace(buffers, next_tokens=_tensor(0x3C00, (1,), "int32"))
    with pytest.raises(ValueError, match="integer buffers"):
        replace(buffers, next_tokens=_tensor(0x3F00, (2,), "fp16"))
    with pytest.raises(ValueError, match="full_accept"):
        replace(buffers, full_accept=_tensor(0x4000, (1,), "bool"))
    with pytest.raises(ValueError, match="full_accept"):
        replace(buffers, full_accept=_tensor(0x4100, (2,), "int32"))
    with pytest.raises(ValueError, match="committed_output_ids"):
        replace(buffers, committed_output_ids=_tensor(0x4200, (2,), "int32"))
    with pytest.raises(ValueError, match="committed_output_ids"):
        replace(buffers, committed_output_ids=_tensor(0x4300, (2, 5), "fp16"))
    with pytest.raises(ValueError, match="row tensors"):
        TargetVerifyBuffers.for_batch(
            target,
            token_ids=_tensor(0x3000, (4,), "int32"),
            positions=_tensor(0x3100, (5,), "int32"),
            parent_rows=_tensor(0x3200, (5,), "int32"),
            draft_depths=_tensor(0x3300, (5,), "int32"),
            row_to_request=_tensor(0x3400, (5,), "int32"),
            active_mask=_tensor(0x3500, (5,), "bool"),
            target_top1=_tensor(0x3600, (5,), "int32"),
            accepted_counts=_tensor(0x3700, (2,), "int32"),
            commit_rows=_tensor(0x3800, (2,), "int32"),
            commit_tokens=_tensor(0x3900, (2,), "int32"),
            commit_positions=_tensor(0x3A00, (2,), "int32"),
        )
    with pytest.raises(ValueError, match="active_mask"):
        TargetVerifyBuffers.for_batch(
            target,
            token_ids=_tensor(0x3000, (5,), "int32"),
            positions=_tensor(0x3100, (5,), "int32"),
            parent_rows=_tensor(0x3200, (5,), "int32"),
            draft_depths=_tensor(0x3300, (5,), "int32"),
            row_to_request=_tensor(0x3400, (5,), "int32"),
            active_mask=_tensor(0x3500, (5,), "int32"),
            target_top1=_tensor(0x3600, (5,), "int32"),
            accepted_counts=_tensor(0x3700, (2,), "int32"),
            commit_rows=_tensor(0x3800, (2,), "int32"),
            commit_tokens=_tensor(0x3900, (2,), "int32"),
            commit_positions=_tensor(0x3A00, (2,), "int32"),
        )


def test_target_verify_batch_selects_commit_rows_from_accept_counts() -> None:
    draft = DraftBatch(
        request_ids=(1, 2),
        candidate_tokens=(10, 11, 20),
        parent_positions=(5, 6, 3),
        draft_depths=(1, 2, 1),
        row_to_request=(1, 1, 2),
        mode="verify_tree",
        tree_parents=(-1, 0, -1),
    )
    target = TargetVerifyBatch.from_draft(draft, root_tokens=(100, 200), root_positions=(5, 3))

    assert target.candidate_counts == (2, 1)
    selected = target.select_commit_rows((2, 1))
    assert selected.request_ids == (1, 2)
    assert selected.accepted_counts == (2, 1)
    assert selected.selected_rows == (3, 4)
    assert selected.selected_tokens == (11, 20)
    assert selected.selected_positions == (7, 4)
    assert selected.mode == "verify_tree"

    zero = target.select_commit_rows((0, 0))
    assert zero.selected_rows == (0, 1)
    assert zero.selected_tokens == (100, 200)
    assert zero.selected_positions == (5, 3)


def test_target_verify_batch_accept_from_top1_oracles_device_accept_summary() -> None:
    draft = DraftBatch(
        request_ids=(1, 2),
        candidate_tokens=(10, 11, 20),
        parent_positions=(5, 6, 3),
        draft_depths=(1, 2, 1),
        row_to_request=(1, 1, 2),
        mode="verify_tree",
        tree_parents=(-1, 0, -1),
    )
    target = TargetVerifyBatch.from_draft(draft, root_tokens=(100, 200), root_positions=(5, 3))

    result = target.accept_from_top1((10, 21, 11, 12, 22), transaction_id=3)

    assert result.transaction_id == 3
    assert result.request_ids == (1, 2)
    assert result.accepted_counts == (2, 0)
    assert result.accepted_tokens == ((10, 11), ())
    assert result.selected_candidate_rows == (3, 1)
    assert result.next_tokens == (12, 21)
    summary = TargetAcceptSummary.from_accept_result(target, result)
    assert summary.commit_rows == (3, 1)
    assert summary.commit_tokens == (11, 200)
    assert summary.next_tokens == (12, 21)

    budgeted = target.accept_from_top1((10, 20, 11, 12, 21), remaining_decode=(1, 1))
    assert budgeted.accepted_counts == (1, 1)
    assert budgeted.accepted_tokens == ((10,), (20,))
    assert budgeted.selected_candidate_rows == (2, 4)
    assert budgeted.next_tokens == (None, None)

    with pytest.raises(ValueError, match="target_top1"):
        target.accept_from_top1((10, 20))
    with pytest.raises(ValueError, match="target_top1"):
        target.accept_from_top1((10, 20, 11, 12, -1))
    with pytest.raises(ValueError, match="remaining_decode"):
        target.accept_from_top1((10, 20, 11, 12, 21), remaining_decode=(1,))

    ambiguous = TargetVerifyBatch.from_draft(
        DraftBatch(
            request_ids=(1,),
            candidate_tokens=(10, 10),
            parent_positions=(5, 5),
            draft_depths=(1, 1),
            row_to_request=(1, 1),
            mode="verify_tree",
            tree_parents=(-1, -1),
        ),
        root_tokens=(100,),
        root_positions=(5,),
    )
    with pytest.raises(ValueError, match="multiple candidate"):
        ambiguous.accept_from_top1((10, 11, 12))


def test_target_accept_summary_validates_paths_and_commit_rows() -> None:
    draft = DraftBatch(
        request_ids=(1, 2),
        candidate_tokens=(10, 11, 20),
        parent_positions=(5, 6, 3),
        draft_depths=(1, 2, 1),
        row_to_request=(1, 1, 2),
        mode="verify_tree",
        tree_parents=(-1, 0, -1),
    )
    target = TargetVerifyBatch.from_draft(draft, root_tokens=(100, 200), root_positions=(5, 3))

    summary = TargetAcceptSummary.from_accept_result(
        target,
        AcceptResult(
            request_ids=(1, 2),
            accepted_counts=(2, 1),
            accepted_tokens=((10, 11), (20,)),
            transaction_id=7,
            next_tokens=(12, 21),
        ),
    )

    assert summary.transaction_id == 7
    assert summary.request_ids == (1, 2)
    assert summary.accepted_counts == (2, 1)
    assert summary.accepted_tokens == ((10, 11), (20,))
    assert summary.next_tokens == (12, 21)
    assert summary.commit_rows == (3, 4)
    assert summary.commit_tokens == (11, 20)
    assert summary.commit_positions == (7, 4)
    assert summary.full_accept == (True, True)
    assert summary.candidate_counts == (2, 1)
    assert summary.draft_depth == 2
    assert summary.tree_shape == (0, 1, 0)
    assert summary.mode == "verify_tree"

    zero = TargetAcceptSummary.from_accept_result(
        target,
        AcceptResult(request_ids=(1, 2), accepted_counts=(0, 0), accepted_tokens=((), ())),
    )
    assert zero.commit_rows == (0, 1)
    assert zero.commit_tokens == (100, 200)
    assert zero.commit_positions == (5, 3)
    assert zero.full_accept == (False, False)
    assert zero.candidate_counts == (2, 1)
    assert zero.draft_depth == 2
    assert zero.tree_shape == (0, 1, 0)

    with pytest.raises(ValueError, match="transaction_id"):
        replace(summary, transaction_id=-1)
    with pytest.raises(ValueError, match="next_tokens"):
        replace(summary, next_tokens=(12,))
    with pytest.raises(ValueError, match="next_tokens"):
        replace(summary, next_tokens=(12, -1))
    with pytest.raises(ValueError, match="draft_depth"):
        replace(summary, draft_depth=-1)
    with pytest.raises(ValueError, match="tree_shape"):
        replace(summary, tree_shape=(0, 1))
    with pytest.raises(ValueError, match="selected target verify paths"):
        TargetAcceptSummary.from_accept_result(
            target,
            AcceptResult(request_ids=(1, 2), accepted_counts=(2, 1), accepted_tokens=((10, 12), (20,))),
        )


def test_target_commit_plan_binds_accept_summary_to_kv_transaction() -> None:
    policy = FixedPagedKVPolicy()
    for request_id, ptr in [(1, 0x1000), (2, 0x2000)]:
        policy.register(
            request_id,
            block_table=_tensor(ptr, (4,), "int32"),
            live_counts=_tensor(ptr + 0x100, (1,), "int64"),
            max_live_count=4,
        )
    draft = DraftBatch(
        request_ids=(1, 2),
        candidate_tokens=(10, 11, 20),
        parent_positions=(5, 6, 3),
        draft_depths=(1, 2, 1),
        row_to_request=(1, 1, 2),
        mode="verify_tree",
        tree_parents=(-1, 0, -1),
    )
    target = TargetVerifyBatch.from_draft(draft, root_tokens=(100, 200), root_positions=(5, 3))
    txn = policy.begin_transaction((1, 2), target)
    summary = TargetAcceptSummary.from_accept_result(
        target,
        AcceptResult(
            request_ids=(1, 2),
            accepted_counts=(2, 1),
            accepted_tokens=((10, 11), (20,)),
            transaction_id=txn.transaction_id,
            next_tokens=(12, 21),
        ),
    )

    plan = TargetCommitPlan.from_summary(summary, txn)

    assert plan.transaction_id == txn.transaction_id
    assert plan.request_ids == (1, 2)
    assert plan.accepted_counts == (2, 1)
    assert plan.kv_accept_counts == (2, 1)
    assert plan.commit_rows == (3, 4)
    assert plan.commit_tokens == (11, 20)
    assert plan.commit_positions == (7, 4)
    assert plan.next_tokens == (12, 21)
    assert plan.candidate_counts == (2, 1)
    assert plan.draft_depth == 2
    assert plan.tree_shape == (0, 1, 0)
    assert plan.mode == "verify_tree"
    committed = policy.commit(txn, plan.kv_accept_counts)
    assert committed.committed
    assert committed.accepted_counts == plan.accepted_counts

    mismatch_txn = SimpleNamespace(
        transaction_id=8,
        request_ids=(1, 2),
        candidate_counts=(2, 1),
        committed=False,
        rolled_back=False,
        role="verify_chain",
    )
    with pytest.raises(ValueError, match="role must match"):
        TargetCommitPlan.from_summary(replace(summary, transaction_id=mismatch_txn.transaction_id), mismatch_txn)
    invalid_role_txn = SimpleNamespace(
        transaction_id=9,
        request_ids=(1, 2),
        candidate_counts=(2, 1),
        committed=False,
        rolled_back=False,
        role="decode",
    )
    with pytest.raises(ValueError, match="verify_chain or verify_tree"):
        TargetCommitPlan.from_summary(replace(summary, transaction_id=invalid_role_txn.transaction_id), invalid_role_txn)
    mismatched_summary_txn = replace(summary, transaction_id=txn.transaction_id + 1)
    with pytest.raises(ValueError, match="transaction_id"):
        TargetCommitPlan.from_summary(mismatched_summary_txn, txn)
    mismatched_candidate_txn = SimpleNamespace(
        transaction_id=10,
        request_ids=(1, 2),
        candidate_counts=(3, 1),
        committed=False,
        rolled_back=False,
        role="verify_tree",
    )
    with pytest.raises(ValueError, match="candidate_counts must match"):
        TargetCommitPlan.from_summary(replace(summary, transaction_id=mismatched_candidate_txn.transaction_id), mismatched_candidate_txn)

    with pytest.raises(ValueError, match="candidate_counts"):
        TargetCommitPlan(
            transaction_id=7,
            request_ids=(1,),
            accepted_counts=(2,),
            commit_rows=(3,),
            commit_tokens=(11,),
            commit_positions=(7,),
            candidate_counts=(1,),
        )
    with pytest.raises(ValueError, match="next_tokens"):
        replace(plan, next_tokens=(12,))
    with pytest.raises(ValueError, match="next_tokens"):
        replace(plan, next_tokens=(12, -1))


def test_target_state_commit_buffers_validate_copy_contract() -> None:
    plan = TargetCommitPlan(
        transaction_id=3,
        request_ids=(1, 2),
        accepted_counts=(2, 1),
        commit_rows=(3, 4),
        commit_tokens=(11, 20),
        commit_positions=(7, 4),
        candidate_counts=(2, 1),
        mode="verify_tree",
    )

    buffers = TargetStateCommitBuffers.for_plan(
        plan,
        accepted_counts=_tensor(0x4000, (2,), "int32"),
        commit_rows=_tensor(0x4100, (2,), "int32"),
        commit_positions=_tensor(0x4200, (2,), "int32"),
        parent_rows=_tensor(0x4250, (5,), "int32"),
        linear_state_src=_tensor(0x4300, (5, 40, 128), "bf16"),
        linear_state_dst=_tensor(0x4400, (2, 40, 128), "bf16"),
        kv_rows_src=_tensor(0x4500, (5, 8, 128), "bf16"),
        kv_rows_dst=_tensor(0x4600, (3, 8, 128), "bf16"),
        hidden_taps_src=_tensor(0x4700, (2, 5, 128), "bf16"),
        hidden_taps_dst=_tensor(0x4800, (2, 2, 128), "bf16"),
        next_tokens_src=_tensor(0x4880, (2,), "int32"),
        committed_output_ids_src=_tensor(0x4900, (2, 5), "int32"),
        committed_output_lengths_src=_tensor(0x4A00, (2,), "int32"),
        output_ids_dst=_tensor(0x4B00, (2, 5), "int32"),
        output_lengths_dst=_tensor(0x4C00, (2,), "int32"),
        last_positions_dst=_tensor(0x4D00, (2,), "int32"),
        context_lengths_dst=_tensor(0x4E00, (2,), "int32"),
    )

    assert buffers.request_ids == (1, 2)
    assert buffers.transaction_id == plan.transaction_id
    assert buffers.request_count == 2
    assert str(buffers.device) == "hip:0"
    assert buffers.has_linear_state
    assert buffers.has_kv_rows
    assert buffers.has_hidden_taps
    assert buffers.has_output_ring
    assert buffers.next_tokens_src is not None
    assert buffers.has_context_metadata
    assert buffers.mode == "verify_tree"

    with pytest.raises(ValueError, match="transaction_id"):
        replace(buffers, transaction_id=-1)
    with pytest.raises(ValueError, match="src/dst pair"):
        TargetStateCommitBuffers.for_plan(
            plan,
            accepted_counts=_tensor(0x4000, (2,), "int32"),
            commit_rows=_tensor(0x4100, (2,), "int32"),
            commit_positions=_tensor(0x4200, (2,), "int32"),
            linear_state_src=_tensor(0x4300, (5, 40, 128), "bf16"),
        )
    with pytest.raises(ValueError, match="tail shape"):
        TargetStateCommitBuffers.for_plan(
            plan,
            accepted_counts=_tensor(0x4000, (2,), "int32"),
            commit_rows=_tensor(0x4100, (2,), "int32"),
            commit_positions=_tensor(0x4200, (2,), "int32"),
            parent_rows=_tensor(0x4250, (5,), "int32"),
            kv_rows_src=_tensor(0x4500, (5, 8, 128), "bf16"),
            kv_rows_dst=_tensor(0x4600, (3, 4, 128), "bf16"),
        )
    with pytest.raises(ValueError, match="parent_rows"):
        TargetStateCommitBuffers.for_plan(
            plan,
            accepted_counts=_tensor(0x4000, (2,), "int32"),
            commit_rows=_tensor(0x4100, (2,), "int32"),
            commit_positions=_tensor(0x4200, (2,), "int32"),
            kv_rows_src=_tensor(0x4500, (5, 8, 128), "bf16"),
            kv_rows_dst=_tensor(0x4600, (3, 8, 128), "bf16"),
        )
    with pytest.raises(ValueError, match="hidden_taps"):
        TargetStateCommitBuffers.for_plan(
            plan,
            accepted_counts=_tensor(0x4000, (2,), "int32"),
            commit_rows=_tensor(0x4100, (2,), "int32"),
            commit_positions=_tensor(0x4200, (2,), "int32"),
            hidden_taps_src=_tensor(0x4700, (2, 5, 128), "bf16"),
            hidden_taps_dst=_tensor(0x4800, (1, 2, 128), "bf16"),
        )
    with pytest.raises(ValueError, match="output ring"):
        TargetStateCommitBuffers.for_plan(
            plan,
            accepted_counts=_tensor(0x4000, (2,), "int32"),
            commit_rows=_tensor(0x4100, (2,), "int32"),
            commit_positions=_tensor(0x4200, (2,), "int32"),
            committed_output_ids_src=_tensor(0x4900, (2, 5), "int32"),
            output_ids_dst=_tensor(0x4B00, (2, 5), "int32"),
        )


def test_target_verify_batch_requires_selected_rows_for_ambiguous_tree_depth() -> None:
    target = TargetVerifyBatch.from_draft(
        DraftBatch(
            request_ids=(1,),
            candidate_tokens=(10, 11),
            parent_positions=(5, 5),
            draft_depths=(1, 1),
            row_to_request=(1, 1),
            mode="verify_tree",
            tree_parents=(-1, -1),
        ),
        root_tokens=(100,),
        root_positions=(5,),
    )

    with pytest.raises(ValueError, match="ambiguous"):
        target.select_commit_rows((1,))
    selected = target.select_commit_rows((1,), selected_candidate_rows=(2,))
    assert selected.selected_rows == (2,)
    assert selected.selected_tokens == (11,)
    result = AcceptResult(
        request_ids=(1,),
        accepted_counts=(1,),
        accepted_tokens=((11,),),
        selected_candidate_rows=(2,),
    )
    summary = TargetAcceptSummary.from_accept_result(target, result)
    assert summary.commit_rows == (2,)
    assert summary.accepted_tokens == ((11,),)
    with pytest.raises(ValueError, match="selected_candidate_rows"):
        TargetAcceptSummary.from_accept_result(target, result, selected_candidate_rows=(1,))
    with pytest.raises(ValueError, match="candidate row"):
        target.select_commit_rows((1,), selected_candidate_rows=(0,))
    with pytest.raises(ValueError, match="candidate row"):
        TargetAcceptSummary.from_accept_result(target, replace(result, selected_candidate_rows=(0,)))


def test_target_verify_batch_validates_native_row_layout() -> None:
    with pytest.raises(ValueError, match="root tokens/positions"):
        TargetVerifyBatch.from_draft(
            DraftBatch(
                request_ids=(1, 2),
                candidate_tokens=(10,),
                parent_positions=(5,),
                draft_depths=(1,),
                row_to_request=(1,),
            ),
            root_tokens=(100,),
            root_positions=(5,),
        )
    with pytest.raises(ValueError, match="earlier candidate"):
        TargetVerifyBatch.from_draft(
            DraftBatch(
                request_ids=(1,),
                candidate_tokens=(10, 11),
                parent_positions=(5, 6),
                draft_depths=(1, 2),
                row_to_request=(1, 1),
                tree_parents=(1, 0),
            ),
            root_tokens=(100,),
            root_positions=(5,),
        )
    with pytest.raises(ValueError, match="root rows"):
        TargetVerifyBatch(
            request_ids=(1,),
            tokens=(100, 10),
            positions=(5, 6),
            row_to_request=(1, 1),
            parent_rows=(0, 0),
            root_rows=(0,),
            candidate_rows=(1,),
            draft_depths=(0, 1),
            active_mask=(True, True),
        )
    with pytest.raises(ValueError, match="same request"):
        TargetVerifyBatch(
            request_ids=(1, 2),
            tokens=(100, 200, 10),
            positions=(5, 3, 6),
            row_to_request=(1, 2, 1),
            parent_rows=(-1, -1, 1),
            root_rows=(0, 1),
            candidate_rows=(2,),
            draft_depths=(0, 0, 1),
            active_mask=(True, True, True),
        )


def test_speculative_draft_batch_drives_kv_transaction_commit_and_rollback() -> None:
    policy = FixedPagedKVPolicy()
    for request_id, ptr in [(1, 0x1000), (2, 0x2000)]:
        policy.register(
            request_id,
            block_table=_tensor(ptr, (4,), "int32"),
            live_counts=_tensor(ptr + 0x100, (1,), "int64"),
            max_live_count=4,
        )
    draft = DraftBatch(
        request_ids=(1, 2),
        candidate_tokens=(10, 11, 20),
        parent_positions=(5, 6, 3),
        draft_depths=(1, 2, 1),
        row_to_request=(1, 1, 2),
        mode="verify_tree",
        tree_parents=(-1, 0, -1),
    )

    txn = policy.begin_transaction([SimpleNamespace(request_id=1), SimpleNamespace(request_id=2)], draft)
    assert txn.request_ids == (1, 2)
    assert txn.draft_rows == 3
    assert txn.candidate_counts == (2, 1)
    assert txn.role == "verify_tree"

    accepted = AcceptResult(request_ids=(1, 2), accepted_counts=(2, 1), accepted_tokens=((10, 11), (20,)))
    committed = policy.commit(txn, accepted.accepted_counts)
    assert committed.committed
    assert committed.accepted_counts == (2, 1)

    target = TargetVerifyBatch.from_draft(draft, root_tokens=(100, 200), root_positions=(5, 3))
    txn_target = policy.begin_transaction([SimpleNamespace(request_id=1), SimpleNamespace(request_id=2)], target)
    assert txn_target.request_ids == (1, 2)
    assert txn_target.draft_rows == 3
    assert txn_target.candidate_counts == (2, 1)
    assert txn_target.role == "verify_tree"
    with pytest.raises(ValueError, match="candidate_counts"):
        policy.commit(txn_target, [3, 0])

    txn2 = policy.begin_transaction([1], DraftBatch(
        request_ids=(1,),
        candidate_tokens=(12,),
        parent_positions=(7,),
        draft_depths=(1,),
        row_to_request=(1,),
    ))
    rolled = policy.rollback(txn2)
    assert rolled.rolled_back


def test_qwen35_dflash_blocker_payload_records_missing_native_verifier(tmp_path) -> None:
    batch_artifact = tmp_path / "batch.json"
    batch_artifact.write_text(
        json.dumps(
            {
                "status": "blocked",
                "performance_claim": False,
                "workload": {"concurrency": 8},
                "execution": {
                    "batch_execution": {
                        "path": "scheduler_serial_slot_bridge",
                        "row_execution": "serial_c1_layer_path",
                        "throughput_claim_eligible": False,
                        "native_prefill_plan": {
                            "linear_prefix_layers": 3,
                            "full_layer_limit_native": False,
                            "first_unsupported_layer": 3,
                            "first_unsupported_type": "full_attention",
                        },
                    }
                },
            }
        )
    )
    prefill_artifact = tmp_path / "prefill.json"
    prefill_artifact.write_text(json.dumps({"native_prefill_plan": {"linear_prefix_layers": 3}}))

    payload = build_payload(batch_artifact=batch_artifact, prefill_artifact=prefill_artifact, argv=[])

    assert payload["status"] == "blocked"
    assert not payload["performance_claim"]
    assert not payload["implementation_status"]["native_target_verify_ready"]
    assert payload["implementation_status"]["interfaces_present"]["target_verify_batch"] == "TargetVerifyBatch"
    assert payload["implementation_status"]["interfaces_present"]["target_accept_summary"] == "TargetAcceptSummary"
    assert payload["implementation_status"]["interfaces_present"]["target_commit_plan"] == "TargetCommitPlan"
    assert payload["implementation_status"]["interfaces_present"]["target_state_commit_buffers"] == "TargetStateCommitBuffers"
    assert payload["implementation_status"]["interfaces_present"]["target_verify_buffers"] == "TargetVerifyBuffers"
    assert payload["implementation_status"]["interfaces_present"]["speculative_request_ids_unique_checked"]
    assert payload["implementation_status"]["interfaces_present"]["kv_transaction_request_ids_unique_checked"]
    assert payload["implementation_status"]["interfaces_present"]["kv_transaction_terminal_state_checked"]
    assert payload["implementation_status"]["interfaces_present"]["kv_transaction_role_checked"]
    assert payload["implementation_status"]["interfaces_present"]["target_commit_plan_transaction_role_checked"]
    assert payload["implementation_status"]["interfaces_present"]["target_commit_plan_candidate_budget_checked"]
    assert payload["implementation_status"]["interfaces_present"]["target_accept_summary_transaction_id_checked"]
    assert payload["implementation_status"]["interfaces_present"]["target_accept_oracle_checked"]
    assert payload["implementation_status"]["interfaces_present"]["accept_result_selected_rows_checked"]
    assert payload["implementation_status"]["interfaces_present"]["accept_result_next_tokens_checked"]
    assert payload["implementation_status"]["interfaces_present"]["target_accept_summary_topology_checked"]
    assert payload["implementation_status"]["interfaces_present"]["target_verify_buffers_transaction_id_checked"]
    assert payload["implementation_status"]["interfaces_present"]["target_verify_buffers_candidate_counts_checked"]
    assert payload["implementation_status"]["interfaces_present"]["target_verify_buffers_topology_checked"]
    assert payload["implementation_status"]["interfaces_present"]["target_verify_buffers_next_tokens_checked"]
    assert payload["implementation_status"]["interfaces_present"]["scheduler_speculative_verify_work"]
    assert payload["implementation_status"]["interfaces_present"]["scheduler_speculative_accept"]
    assert payload["implementation_status"]["interfaces_present"]["scheduler_speculative_next_tokens_checked"]
    assert payload["implementation_status"]["interfaces_present"]["scheduler_speculative_shape_key"]
    assert payload["implementation_status"]["interfaces_present"]["scheduler_speculative_graph_cache"]
    assert payload["implementation_status"]["interfaces_present"]["scheduler_speculative_kv_transaction"]
    assert payload["implementation_status"]["interfaces_present"]["scheduler_speculative_verify_plan"]
    assert payload["implementation_status"]["interfaces_present"]["scheduler_speculative_buffer_plan"]
    assert payload["implementation_status"]["interfaces_present"]["scheduler_speculative_commit_plan"]
    assert payload["implementation_status"]["interfaces_present"]["scheduler_speculative_commit_from_top1"]
    assert payload["implementation_status"]["interfaces_present"]["scheduler_speculative_state_commit_plan"]
    assert payload["implementation_status"]["interfaces_present"]["scheduler_speculative_kv_commit"]
    assert payload["implementation_status"]["interfaces_present"]["scheduler_speculative_kv_rollback"]
    assert payload["implementation_status"]["interfaces_present"]["scheduler_speculative_accept_finalize"]
    assert payload["implementation_status"]["kv_transaction_target_verify"]["target_verify_rows"] == 5
    assert payload["implementation_status"]["kv_transaction_target_verify"]["candidate_counts"] == [2, 1]
    assert payload["implementation_status"]["kv_transaction_target_verify"]["commit_selection_rows"] == [3, 4]
    assert payload["implementation_status"]["kv_transaction_target_verify"]["target_top1"] == [10, 20, 11, 12, 21]
    assert payload["implementation_status"]["kv_transaction_target_verify"]["accept_result"]["selected_candidate_rows"] == [3, 4]
    assert payload["implementation_status"]["kv_transaction_target_verify"]["accept_result"]["next_tokens"] == [12, 21]
    assert payload["implementation_status"]["kv_transaction_target_verify"]["accept_summary"]["commit_rows"] == [3, 4]
    assert payload["implementation_status"]["kv_transaction_target_verify"]["accept_summary"]["transaction_id"] == 0
    assert payload["implementation_status"]["kv_transaction_target_verify"]["accept_summary"]["accepted_tokens"] == [[10, 11], [20]]
    assert payload["implementation_status"]["kv_transaction_target_verify"]["accept_summary"]["next_tokens"] == [12, 21]
    assert payload["implementation_status"]["kv_transaction_target_verify"]["accept_summary"]["candidate_counts"] == [2, 1]
    assert payload["implementation_status"]["kv_transaction_target_verify"]["accept_summary"]["draft_depth"] == 2
    assert payload["implementation_status"]["kv_transaction_target_verify"]["accept_summary"]["tree_shape"] == [0, 1, 0]
    assert payload["implementation_status"]["kv_transaction_target_verify"]["commit_plan"]["accepted_counts"] == [2, 1]
    assert payload["implementation_status"]["kv_transaction_target_verify"]["commit_plan"]["next_tokens"] == [12, 21]
    assert payload["implementation_status"]["kv_transaction_target_verify"]["commit_plan"]["candidate_counts"] == [2, 1]
    assert payload["implementation_status"]["kv_transaction_target_verify"]["commit_plan"]["draft_depth"] == 2
    assert payload["implementation_status"]["kv_transaction_target_verify"]["commit_plan"]["tree_shape"] == [0, 1, 0]
    assert payload["implementation_status"]["kv_transaction_target_verify"]["state_commit_buffers"]["transaction_id"] == 0
    assert payload["implementation_status"]["kv_transaction_target_verify"]["device_buffers"]["rows"] == 5
    assert payload["implementation_status"]["kv_transaction_target_verify"]["device_buffers"]["candidate_counts"] == [2, 1]
    assert payload["implementation_status"]["kv_transaction_target_verify"]["device_buffers"]["draft_depth"] == 2
    assert payload["implementation_status"]["kv_transaction_target_verify"]["device_buffers"]["tree_shape"] == [0, 1, 0]
    assert payload["implementation_status"]["kv_transaction_target_verify"]["device_buffers"]["summary_rows"] == 2
    assert payload["implementation_status"]["kv_transaction_target_verify"]["device_buffers"]["next_tokens_shape"] == [2]
    assert payload["implementation_status"]["kv_transaction_target_verify"]["device_buffers"]["next_tokens_dtype"] == "int32"
    assert payload["implementation_status"]["kv_transaction_target_verify"]["state_commit_buffers"]["has_linear_state"]
    assert payload["implementation_status"]["kv_transaction_target_verify"]["state_commit_buffers"]["has_kv_rows"]
    assert payload["implementation_status"]["kv_transaction_target_verify"]["shape_key"]["tree_shape"] == [0, 1, 0]
    assert payload["implementation_status"]["kv_transaction_target_verify"]["work_item"]["tree_parents"] == [0, 1, 0]
    assert payload["implementation_status"]["kv_transaction_target_verify"]["transaction_draft_rows"] == 3
    assert payload["implementation_status"]["kv_transaction_target_verify"]["root_rows_excluded_from_journal"]
    plan = payload["implementation_status"]["kv_transaction_target_verify"]["scheduler_verify_plan"]
    assert plan["request_ids"] == [1, 2]
    assert plan["transaction_draft_rows"] == 3
    assert plan["candidate_counts"] == [2, 1]
    assert plan["shape_key_matches_target"]
    assert plan["graph_cache_entries"] == 1
    assert plan["graph_mode"] == "verify_tree"
    assert plan["graph_draft_depth"] == 2
    buffer_plan = payload["implementation_status"]["kv_transaction_target_verify"]["scheduler_buffer_plan"]
    assert buffer_plan["request_ids"] == [1, 2]
    assert buffer_plan["rows"] == 5
    assert buffer_plan["candidate_rows"] == 3
    assert buffer_plan["mode"] == "verify_tree"
    assert buffer_plan["target_batch_rows"] == 5
    assert buffer_plan["candidate_counts"] == [2, 1]
    assert buffer_plan["target_candidate_counts"] == [2, 1]
    assert buffer_plan["candidate_counts_match"]
    assert buffer_plan["draft_depth"] == 2
    assert buffer_plan["target_draft_depth"] == 2
    assert buffer_plan["draft_depth_matches"]
    assert buffer_plan["tree_shape"] == [0, 1, 0]
    assert buffer_plan["target_tree_shape"] == [0, 1, 0]
    assert buffer_plan["tree_shape_matches"]
    assert buffer_plan["transaction_id"] == plan["transaction_id"]
    assert buffer_plan["next_tokens_shape"] == [2]
    commit_plan = payload["implementation_status"]["kv_transaction_target_verify"]["scheduler_commit_plan"]
    assert commit_plan["transaction_id"] == plan["transaction_id"]
    assert commit_plan["summary_transaction_id"] == plan["transaction_id"]
    assert commit_plan["from_top1"]
    assert commit_plan["request_ids"] == [1, 2]
    assert commit_plan["accepted_counts"] == [2, 1]
    assert commit_plan["commit_rows"] == [3, 4]
    assert commit_plan["commit_positions"] == [7, 4]
    assert commit_plan["next_tokens"] == [12, 21]
    assert commit_plan["candidate_counts"] == [2, 1]
    assert commit_plan["mode"] == "verify_tree"
    state_plan = payload["implementation_status"]["kv_transaction_target_verify"]["scheduler_state_commit_plan"]
    assert state_plan["request_ids"] == [1, 2]
    assert state_plan["request_rows"] == 2
    assert state_plan["mode"] == "verify_tree"
    assert state_plan["device"] == "hip:0"
    assert state_plan["verify_device"] == "hip:0"
    assert state_plan["device_matches_verify"]
    assert state_plan["buffer_transaction_id"] == state_plan["transaction_id"]
    assert state_plan["transaction_id_matches"]
    assert state_plan["target_rows"] == 5
    assert state_plan["accepted_rows"] == 3
    assert state_plan["has_linear_state"]
    assert state_plan["linear_src_rows"] == 5
    assert state_plan["linear_src_covers_target"]
    assert state_plan["has_kv_rows"]
    assert state_plan["kv_src_rows"] == 5
    assert state_plan["kv_src_covers_target"]
    assert state_plan["kv_dst_rows"] == 3
    assert state_plan["kv_dst_covers_accepts"]
    assert state_plan["transaction_id"] == plan["transaction_id"]
    kv_commit = payload["implementation_status"]["kv_transaction_target_verify"]["scheduler_kv_commit"]
    assert kv_commit["transaction_id"] == plan["transaction_id"]
    assert kv_commit["request_ids"] == [1, 2]
    assert kv_commit["accepted_counts"] == [2, 1]
    assert kv_commit["committed"]
    assert not kv_commit["rolled_back"]
    kv_rollback = payload["implementation_status"]["kv_transaction_target_verify"]["scheduler_kv_rollback"]
    assert kv_rollback["request_ids"] == [1, 2]
    assert not kv_rollback["committed"]
    assert kv_rollback["rolled_back"]
    assert kv_rollback["transaction_id"] != kv_commit["transaction_id"]
    finalize = payload["implementation_status"]["kv_transaction_target_verify"]["scheduler_accept_finalize"]
    assert finalize["completed_request_ids"] == []
    assert finalize["active_generated_counts"] == {"1": 3, "2": 2}
    assert finalize["completed_generated_counts"] == {}
    assert payload["implementation_status"]["resident_api"]["step_batch_serial"]
    assert payload["implementation_status"]["resident_api"]["native_target_verify_batch"]
    assert payload["implementation_status"]["resident_api"]["speculative_verify_batch"]
    assert payload["implementation_status"]["resident_api"]["target_verify_buffers_transaction_id_checked"]
    assert payload["implementation_status"]["resident_api"]["target_verify_buffers_resident_device_checked"]
    assert payload["implementation_status"]["resident_api"]["commit_verified_state"]
    assert payload["implementation_status"]["resident_api"]["commit_verified_state_transaction_id_checked"]
    assert payload["implementation_status"]["resident_api"]["commit_verified_state_row_coverage_checked"]
    assert not payload["implementation_status"]["resident_api"]["native_target_verify_executes_kernels"]
    assert payload["implementation_status"]["resident_api"]["commit_verified_state_executes_copies"]
    assert not payload["implementation_status"]["native_target_verify_ready"]
    assert payload["evidence"]["batch_execution"]["path"] == "scheduler_serial_slot_bridge"
    assert any("native verifier loop" in blocker for blocker in payload["blockers"])
    assert any("throughput_claim_eligible=false" in blocker for blocker in payload["blockers"])
