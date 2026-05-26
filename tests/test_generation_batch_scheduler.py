from __future__ import annotations

import json
from dataclasses import replace

import pytest

from hipengine.core.device import Device
from hipengine.core.tensor import Tensor
from hipengine.dispatch import WorkKind
from hipengine.generation import (
    CompactPromptSlab,
    GeneratedToken,
    GraphBucketCache,
    ResidentBatchScheduler,
    ResidentEngineLoop,
    SpeculativeCommitPlan,
    SpeculativeStateCommitPlan,
    SpeculativeVerifyBufferPlan,
    SpeculativeVerifyPlan,
    SpeculativeVerifyWork,
)
from hipengine.kvcache import FixedPagedKVPolicy
from hipengine.speculative import AcceptResult, DraftBatch, TargetAcceptSummary, TargetStateCommitBuffers, TargetVerifyBuffers
from scripts.qwen35_batch_artifact_schema import validate_cn_diagnostic_artifact_payload
from scripts.qwen35_batch_serial_bench import _load_prompt_slices, _summarize_samples


def _tensor(ptr: int, shape: tuple[int, ...], dtype: str) -> Tensor:
    return Tensor.from_handle(ptr, shape, dtype, Device("hip", 0))


class _FakeSerialBridgeRunner:
    def __init__(self) -> None:
        self.prefills = []
        self.decodes = []
        self._counts: dict[int, int] = {}

    def prefill(self, work) -> None:
        self.prefills.append(work)

    def decode(self, work) -> tuple[GeneratedToken, ...]:
        self.decodes.append(work)
        tokens: list[GeneratedToken] = []
        for request_id in work.request_ids:
            count = self._counts.get(request_id, 0)
            self._counts[request_id] = count + 1
            tokens.append(GeneratedToken(request_id, 1000 + request_id * 10 + count))
        return tuple(tokens)


def test_resident_engine_loop_submit_poll_cancel_and_reclaim() -> None:
    runner = _FakeSerialBridgeRunner()
    loop = ResidentEngineLoop(runner, capacity=2, prefill_chunk_size=8, context_bucket_size=4)
    r0 = loop.submit([10, 11], max_new_tokens=2)
    r1 = loop.submit([20], max_new_tokens=1)
    r2 = loop.submit([30], max_new_tokens=1)
    r3 = loop.submit([40], max_new_tokens=4)

    assert loop.cancel(r3) is True
    assert loop.cancel(9999) is False
    assert loop.completed[r3].finished is True
    assert loop.pending_count == 3

    events = loop.poll(max_ticks=8)

    assert [(work.kind, work.request_ids) for work in runner.prefills] == [
        (WorkKind.PREFILL, (r0,)),
        (WorkKind.PREFILL, (r2,)),
        (WorkKind.PREFILL, (r1,)),
    ]
    assert [(work.kind, work.request_ids) for work in runner.decodes] == [
        (WorkKind.DECODE, (r0,)),
        (WorkKind.DECODE, (r0,)),
        (WorkKind.DECODE, (r2,)),
        (WorkKind.DECODE, (r1,)),
    ]
    assert [event.request_id for event in events if event.kind == "completed"] == [r0, r2, r1]
    assert loop.pending_count == 0
    assert loop.active_count == 0
    assert set(loop.completed) == {r0, r1, r2, r3}
    assert loop.completed[r0].generated_tokens == (1000, 1001)
    assert loop.completed[r1].generated_tokens == (1010,)
    assert loop.completed[r2].generated_tokens == (1020,)

    active_cancel_loop = ResidentEngineLoop(_FakeSerialBridgeRunner(), capacity=1, prefill_chunk_size=8)
    active_id = active_cancel_loop.submit([50], max_new_tokens=4)
    assert active_cancel_loop.poll(max_ticks=1)[0].kind == "admitted"
    assert active_cancel_loop.active_count == 1
    assert active_cancel_loop.cancel(active_id) is True
    assert active_cancel_loop.active_count == 0
    assert active_cancel_loop.completed[active_id].finished is True


def test_resident_engine_loop_prefill_decode_policies() -> None:
    protect_runner = _FakeSerialBridgeRunner()
    protect_loop = ResidentEngineLoop(
        protect_runner,
        capacity=2,
        prefill_chunk_size=8,
        prefill_decode_policy="protect_decode",
    )
    r0 = protect_loop.submit([10], max_new_tokens=2)
    r1 = protect_loop.submit([20], max_new_tokens=1)
    protect_loop.poll(max_ticks=1)
    protect_loop.poll(max_ticks=1)
    assert [(work.kind, work.request_ids) for work in protect_runner.decodes] == [(WorkKind.DECODE, (r0,))]
    assert [(work.kind, work.request_ids) for work in protect_runner.prefills] == [(WorkKind.PREFILL, (r0,))]

    ttft_runner = _FakeSerialBridgeRunner()
    ttft_loop = ResidentEngineLoop(
        ttft_runner,
        capacity=2,
        prefill_chunk_size=8,
        prefill_decode_policy="protect_ttft",
    )
    r0 = ttft_loop.submit([10], max_new_tokens=2)
    r1 = ttft_loop.submit([20], max_new_tokens=1)
    ttft_loop.poll(max_ticks=1)
    ttft_loop.poll(max_ticks=1)
    assert ttft_runner.decodes == []
    assert [(work.kind, work.request_ids) for work in ttft_runner.prefills] == [
        (WorkKind.PREFILL, (r0,)),
        (WorkKind.PREFILL, (r1,)),
    ]

    fair_runner = _FakeSerialBridgeRunner()
    fair_loop = ResidentEngineLoop(
        fair_runner,
        capacity=2,
        prefill_chunk_size=8,
        prefill_decode_policy="fair",
    )
    r0 = fair_loop.submit([10], max_new_tokens=2)
    r1 = fair_loop.submit([20], max_new_tokens=1)
    fair_loop.poll(max_ticks=1)
    fair_loop.poll(max_ticks=1)
    fair_loop.poll(max_ticks=1)
    assert [(work.kind, work.request_ids) for work in fair_runner.decodes] == [(WorkKind.DECODE, (r0,))]
    assert [(work.kind, work.request_ids) for work in fair_runner.prefills] == [
        (WorkKind.PREFILL, (r0,)),
        (WorkKind.PREFILL, (r1,)),
    ]

    with pytest.raises(ValueError, match="prefill_decode_policy"):
        ResidentEngineLoop(_FakeSerialBridgeRunner(), capacity=1, prefill_decode_policy="unknown")


def test_resident_batch_scheduler_admits_compacts_and_routes_decode() -> None:
    scheduler = ResidentBatchScheduler(capacity=2, context_bucket_size=4)
    r0 = scheduler.submit([10, 11], max_new_tokens=2)
    r1 = scheduler.submit([20], max_new_tokens=1)
    r2 = scheduler.submit([30], max_new_tokens=1)

    assert (r0, r1, r2) == (0, 1, 2)
    assert scheduler.admit_pending() == (0, 1)
    assert scheduler.pending_count == 1
    assert scheduler.active_batch.slot_to_request == (0, 1)

    work = scheduler.next_prefill_work(chunk_size=8)
    assert work is not None
    assert work.kind is WorkKind.PREFILL
    assert work.request_ids == (0,)
    assert work.token_rows == ((10, 11),)

    work = scheduler.next_prefill_work(chunk_size=8)
    assert work is not None
    assert work.request_ids == (1,)
    assert work.token_rows == ((20,),)

    decode = scheduler.next_decode_work()
    assert decode is not None
    assert decode.kind is WorkKind.DECODE
    assert decode.request_ids == (0, 1)
    assert decode.row_to_request == (0, 1)

    completed = scheduler.record_generated([(1, 101, True)])
    assert [item.request_id for item in completed] == [1]
    assert scheduler.active_batch.slot_to_request == (0, None)

    assert scheduler.admit_pending() == (2,)
    assert scheduler.active_batch.slot_to_request == (0, 2)

    moves = scheduler.compact(order=(2, 0))
    assert [(move.request_id, move.old_slot, move.new_slot) for move in moves] == [(2, 1, 0), (0, 0, 1)]
    assert scheduler.active_batch.slot_to_request == (2, 0)


def test_resident_batch_scheduler_bucketizes_and_builds_compact_prefill_slabs() -> None:
    scheduler = ResidentBatchScheduler(capacity=3, context_bucket_size=4)
    r0 = scheduler.submit([10, 11, 12], max_new_tokens=1)
    r1 = scheduler.submit([20, 21, 22, 23, 24], max_new_tokens=1)
    r2 = scheduler.submit([30, 31], max_new_tokens=1)
    scheduler.admit_pending()

    buckets = scheduler.bucketize_by_block_count(chunk_size=8, block_size=4)

    assert [(bucket.block_count, bucket.request_ids) for bucket in buckets] == [
        (1, (r0, r2)),
        (2, (r1,)),
    ]

    slabs = scheduler.next_compact_prefill_slabs(chunk_size=8, block_size=4)

    assert len(slabs) == 2
    first = slabs[0]
    assert first.request_ids == (r0, r2)
    assert first.slot_ids == (0, 2)
    assert first.physical_slot_ids == (0, 2)
    assert first.token_ids == (10, 11, 12, 30, 31)
    assert first.positions == (0, 1, 2, 0, 1)
    assert first.append_counts == first.positions
    assert first.context_counts == (1, 2, 3, 1, 2)
    assert first.cu_seqlens_q == (0, 3, 5)
    assert first.cu_seqlens_k == (0, 3, 5)
    assert first.row_to_request == (r0, r0, r0, r2, r2)
    assert first.block_count == 1
    assert first.block_tables == ((0,), (0,), (0,), (0,), (0,))
    assert first.to_work_item().token_rows == ((10, 11, 12), (30, 31))

    second = slabs[1]
    assert second.request_ids == (r1,)
    assert second.slot_ids == (1,)
    assert second.token_ids == (20, 21, 22, 23, 24)
    assert second.cu_seqlens_q == (0, 5)
    assert second.block_count == 2
    assert second.block_tables == ((0, 1),) * 5
    assert scheduler.active_batch.requests[r0].remaining_prefill == 0
    assert scheduler.active_batch.requests[r1].remaining_prefill == 0
    assert scheduler.active_batch.requests[r2].remaining_prefill == 0


def test_resident_batch_scheduler_emits_speculative_verify_work() -> None:
    scheduler = ResidentBatchScheduler(capacity=2, context_bucket_size=4)
    r0 = scheduler.submit([10, 11], max_new_tokens=3)
    r1 = scheduler.submit([20], max_new_tokens=1)
    scheduler.admit_pending()
    scheduler.next_prefill_work(chunk_size=8)
    scheduler.next_prefill_work(chunk_size=8)
    draft = DraftBatch(
        request_ids=(r0, r1),
        candidate_tokens=(101, 102, 201),
        parent_positions=(1, 2, 0),
        draft_depths=(1, 2, 1),
        row_to_request=(r0, r0, r1),
        mode="verify_tree",
        tree_parents=(-1, 0, -1),
    )

    work = scheduler.next_speculative_verify_work(
        draft,
        root_tokens=(11, 20),
        root_positions=(1, 0),
    )

    assert isinstance(work, SpeculativeVerifyWork)
    assert work.target_batch.rows == 5
    assert work.target_batch.tokens == (11, 20, 101, 102, 201)
    assert work.target_batch.parent_rows == (-1, -1, 0, 2, 1)
    assert work.work_item.kind is WorkKind.VERIFY_TREE
    assert work.work_item.request_ids == (r0, r1)
    assert work.work_item.row_to_request == (r0, r0, r1)
    assert work.work_item.token_rows == ((101,), (102,), (201,))
    assert work.work_item.tree_parents == (0, 1, 0)

    key = scheduler.speculative_verify_shape_key(work, top_k=8, experts_per_token=8, replay_steps=2)
    assert key.mode is WorkKind.VERIFY_TREE
    assert key.active_c == 2
    assert key.context_bucket == 4
    assert key.active_mask == (True, True)
    assert key.top_k == 8
    assert key.experts_per_token == 8
    assert key.replay_steps == 2
    assert key.draft_depth == 2
    assert key.tree_shape == (0, 1, 0)
    graph = scheduler.get_or_create_speculative_verify_graph(
        work,
        lambda bucket: {"bucket": bucket},
        top_k=8,
        experts_per_token=8,
        replay_steps=2,
    )
    assert graph == {"bucket": key}
    assert scheduler.graph_buckets.stats.entries == 1
    assert scheduler.get_or_create_speculative_verify_graph(
        work,
        lambda bucket: {"unexpected": bucket},
        top_k=8,
        experts_per_token=8,
        replay_steps=2,
    ) is graph
    assert scheduler.graph_buckets.stats.hits == 1

    policy = FixedPagedKVPolicy()
    for request_id, ptr in [(r0, 0x1000), (r1, 0x2000)]:
        policy.register(
            request_id,
            block_table=_tensor(ptr, (4,), "int32"),
            live_counts=_tensor(ptr + 0x100, (1,), "int64"),
            max_live_count=4,
        )
    txn = scheduler.begin_speculative_verify_transaction(policy, work)
    assert txn.request_ids == (r0, r1)
    assert txn.draft_rows == 3
    assert txn.candidate_counts == (2, 1)
    assert txn.role == "verify_tree"
    plan = scheduler.plan_speculative_verify(
        policy,
        work,
        lambda bucket: {"unexpected": bucket},
        top_k=8,
        experts_per_token=8,
        replay_steps=2,
    )
    assert isinstance(plan, SpeculativeVerifyPlan)
    assert plan.target_batch is work.target_batch
    assert plan.work_item is work.work_item
    assert plan.transaction.request_ids == (r0, r1)
    assert plan.transaction.draft_rows == 3
    assert plan.transaction.candidate_counts == (2, 1)
    assert plan.shape_key == key
    assert plan.graph is graph
    assert scheduler.graph_buckets.stats.hits == 2
    rollback_plan = scheduler.plan_speculative_verify(
        policy,
        work,
        lambda bucket: {"unexpected_rollback": bucket},
        top_k=8,
        experts_per_token=8,
        replay_steps=2,
    )
    rolled_txn = scheduler.rollback_speculative_kv_transaction(policy, rollback_plan)
    assert rolled_txn.transaction_id == rollback_plan.transaction.transaction_id
    assert rolled_txn.request_ids == (r0, r1)
    assert rolled_txn.rolled_back
    assert not rolled_txn.committed
    assert scheduler.graph_buckets.stats.hits == 3
    buffers = TargetVerifyBuffers.for_batch(
        work.target_batch,
        token_ids=_tensor(0x3000, (work.target_batch.rows,), "int32"),
        positions=_tensor(0x3100, (work.target_batch.rows,), "int32"),
        parent_rows=_tensor(0x3200, (work.target_batch.rows,), "int32"),
        draft_depths=_tensor(0x3300, (work.target_batch.rows,), "int32"),
        row_to_request=_tensor(0x3400, (work.target_batch.rows,), "int32"),
        active_mask=_tensor(0x3500, (work.target_batch.rows,), "bool"),
        target_top1=_tensor(0x3600, (work.target_batch.rows,), "int32"),
        accepted_counts=_tensor(0x3700, (len(work.target_batch.request_ids),), "int32"),
        commit_rows=_tensor(0x3800, (len(work.target_batch.request_ids),), "int32"),
        commit_tokens=_tensor(0x3900, (len(work.target_batch.request_ids),), "int32"),
        commit_positions=_tensor(0x3A00, (len(work.target_batch.request_ids),), "int32"),
        transaction_id=plan.transaction.transaction_id,
    )
    buffer_plan = scheduler.bind_speculative_verify_buffers(plan, buffers)
    assert buffers.transaction_id == plan.transaction.transaction_id
    assert buffers.candidate_counts == work.target_batch.candidate_counts
    assert buffers.draft_depth == work.target_batch.draft_depth
    assert buffers.tree_shape == work.target_batch.tree_shape
    assert isinstance(buffer_plan, SpeculativeVerifyBufferPlan)
    assert buffer_plan.plan is plan
    assert buffer_plan.buffers is buffers
    wrong_verify_buffers = replace(buffers, transaction_id=plan.transaction.transaction_id + 1)
    with pytest.raises(ValueError, match="transaction_id"):
        scheduler.bind_speculative_verify_buffers(plan, wrong_verify_buffers)
    wrong_candidate_buffers = replace(buffers, candidate_counts=(1, 2))
    with pytest.raises(ValueError, match="candidate_counts"):
        scheduler.bind_speculative_verify_buffers(plan, wrong_candidate_buffers)
    wrong_depth_buffers = replace(buffers, draft_depth=work.target_batch.draft_depth + 1)
    with pytest.raises(ValueError, match="draft_depth"):
        scheduler.bind_speculative_verify_buffers(plan, wrong_depth_buffers)
    wrong_tree_buffers = replace(buffers, tree_shape=(0, 0, 1))
    with pytest.raises(ValueError, match="tree_shape"):
        scheduler.bind_speculative_verify_buffers(plan, wrong_tree_buffers)

    commit = scheduler.plan_speculative_commit_from_top1(buffer_plan, (101, 201, 102, 103, 202))
    assert isinstance(commit, SpeculativeCommitPlan)
    assert commit.verify_plan is buffer_plan
    summary = commit.summary
    assert summary.transaction_id == plan.transaction.transaction_id
    assert summary.accepted_tokens == ((101, 102), (201,))
    assert summary.next_tokens == (103, None)
    assert commit.commit_plan.transaction_id == plan.transaction.transaction_id
    assert commit.commit_plan.request_ids == (r0, r1)
    assert commit.commit_plan.accepted_counts == (2, 1)
    assert commit.commit_plan.commit_rows == (3, 4)
    assert commit.commit_plan.next_tokens == (103, None)
    assert commit.commit_plan.candidate_counts == (2, 1)
    assert commit.commit_plan.draft_depth == work.target_batch.draft_depth
    assert commit.commit_plan.tree_shape == work.target_batch.tree_shape
    assert commit.commit_plan.mode == "verify_tree"
    with pytest.raises(ValueError, match="target_top1"):
        scheduler.plan_speculative_commit_from_top1(buffer_plan, (101, 201))
    wrong_summary_txn = replace(summary, transaction_id=plan.transaction.transaction_id + 1)
    with pytest.raises(ValueError, match="transaction_id"):
        scheduler.plan_speculative_commit(buffer_plan, wrong_summary_txn)
    wrong_summary_depth = replace(summary, draft_depth=work.target_batch.draft_depth + 1)
    with pytest.raises(ValueError, match="draft_depth"):
        scheduler.plan_speculative_commit(buffer_plan, wrong_summary_depth)
    wrong_summary_tree = replace(summary, tree_shape=(0, 0, 1))
    with pytest.raises(ValueError, match="tree_shape"):
        scheduler.plan_speculative_commit(buffer_plan, wrong_summary_tree)
    state_buffers = TargetStateCommitBuffers.for_plan(
        commit.commit_plan,
        accepted_counts=_tensor(0x3B00, (len(work.target_batch.request_ids),), "int32"),
        commit_rows=_tensor(0x3C00, (len(work.target_batch.request_ids),), "int32"),
        commit_positions=_tensor(0x3D00, (len(work.target_batch.request_ids),), "int32"),
        linear_state_src=_tensor(0x3E00, (work.target_batch.rows, 4), "bf16"),
        linear_state_dst=_tensor(0x3F00, (len(work.target_batch.request_ids), 4), "bf16"),
        kv_rows_src=_tensor(0x4000, (work.target_batch.rows, 2, 4), "bf16"),
        kv_rows_dst=_tensor(0x4100, (sum(summary.accepted_counts), 2, 4), "bf16"),
    )
    state_plan = scheduler.bind_speculative_commit_buffers(commit, state_buffers)
    assert state_buffers.transaction_id == commit.commit_plan.transaction_id
    assert isinstance(state_plan, SpeculativeStateCommitPlan)
    assert state_plan.commit_plan is commit
    assert state_plan.buffers is state_buffers
    assert state_plan.buffers.device == buffer_plan.buffers.device
    assert state_plan.buffers.linear_state_src is not None
    assert state_plan.buffers.linear_state_src.shape[0] == work.target_batch.rows
    assert state_plan.buffers.kv_rows_dst is not None
    assert state_plan.buffers.kv_rows_dst.shape[0] == sum(summary.accepted_counts)
    assert state_plan.buffers.has_linear_state
    assert state_plan.buffers.has_kv_rows
    short_kv_dst_buffers = TargetStateCommitBuffers.for_plan(
        commit.commit_plan,
        accepted_counts=_tensor(0x4200, (len(work.target_batch.request_ids),), "int32"),
        commit_rows=_tensor(0x4300, (len(work.target_batch.request_ids),), "int32"),
        commit_positions=_tensor(0x4400, (len(work.target_batch.request_ids),), "int32"),
        kv_rows_src=_tensor(0x4500, (work.target_batch.rows, 2, 4), "bf16"),
        kv_rows_dst=_tensor(0x4600, (len(work.target_batch.request_ids), 2, 4), "bf16"),
    )
    with pytest.raises(ValueError, match="accepted token rows"):
        scheduler.bind_speculative_commit_buffers(commit, short_kv_dst_buffers)
    wrong_transaction_buffers = replace(state_buffers, transaction_id=commit.commit_plan.transaction_id + 1)
    with pytest.raises(ValueError, match="transaction_id"):
        scheduler.bind_speculative_commit_buffers(commit, wrong_transaction_buffers)
    other_device = Device("hip", 1)
    other_state_buffers = TargetStateCommitBuffers.for_plan(
        commit.commit_plan,
        accepted_counts=Tensor.from_handle(0x4700, (len(work.target_batch.request_ids),), "int32", other_device),
        commit_rows=Tensor.from_handle(0x4800, (len(work.target_batch.request_ids),), "int32", other_device),
        commit_positions=Tensor.from_handle(0x4900, (len(work.target_batch.request_ids),), "int32", other_device),
        linear_state_src=Tensor.from_handle(0x4A00, (work.target_batch.rows, 4), "bf16", other_device),
        linear_state_dst=Tensor.from_handle(0x4B00, (len(work.target_batch.request_ids), 4), "bf16", other_device),
    )
    with pytest.raises(ValueError, match="target verify device"):
        scheduler.bind_speculative_commit_buffers(commit, other_state_buffers)
    committed_txn = scheduler.commit_speculative_kv_transaction(policy, state_plan)
    assert committed_txn.transaction_id == plan.transaction.transaction_id
    assert committed_txn.request_ids == (r0, r1)
    assert committed_txn.accepted_counts == (2, 1)
    assert committed_txn.committed
    completed = scheduler.finalize_speculative_accept(committed_txn, state_plan)

    assert [item.request_id for item in completed] == [r0, r1]
    assert scheduler.completed[r0].generated_tokens == (101, 102, 103)
    assert scheduler.completed[r1].generated_tokens == (201,)
    assert scheduler.active_batch.slot_to_request == (None, None)


def test_resident_batch_scheduler_rejects_speculative_accept_over_budget() -> None:
    scheduler = ResidentBatchScheduler(capacity=1)
    r0 = scheduler.submit([10], max_new_tokens=1)
    scheduler.admit_pending()
    scheduler.next_prefill_work(chunk_size=8)
    draft = DraftBatch(
        request_ids=(r0,),
        candidate_tokens=(101, 102),
        parent_positions=(0, 1),
        draft_depths=(1, 2),
        row_to_request=(r0, r0),
    )
    work = scheduler.next_speculative_verify_work(draft, root_tokens=(10,), root_positions=(0,))
    summary = TargetAcceptSummary.from_accept_result(
        work.target_batch,
        AcceptResult(request_ids=(r0,), accepted_counts=(2,), accepted_tokens=((101, 102),)),
    )

    with pytest.raises(ValueError, match="remaining decode"):
        scheduler.record_speculative_accept(summary)

    next_token_over_budget_summary = TargetAcceptSummary.from_accept_result(
        work.target_batch,
        AcceptResult(request_ids=(r0,), accepted_counts=(1,), accepted_tokens=((101,),), next_tokens=(102,)),
        selected_candidate_rows=(1,),
    )
    with pytest.raises(ValueError, match="remaining decode"):
        scheduler.record_speculative_accept(next_token_over_budget_summary)


def test_resident_batch_scheduler_rejects_speculative_verify_before_prefill() -> None:
    scheduler = ResidentBatchScheduler(capacity=1)
    r0 = scheduler.submit([10, 11], max_new_tokens=3)
    scheduler.admit_pending()
    draft = DraftBatch(
        request_ids=(r0,),
        candidate_tokens=(101,),
        parent_positions=(1,),
        draft_depths=(1,),
        row_to_request=(r0,),
    )

    with pytest.raises(ValueError, match="completed prefill"):
        scheduler.next_speculative_verify_work(draft, root_tokens=(11,), root_positions=(1,))


def test_resident_batch_scheduler_shape_key_graph_bucket_and_completion() -> None:
    scheduler = ResidentBatchScheduler(capacity=4, context_bucket_size=4)
    r0 = scheduler.submit([1], max_new_tokens=1)
    r1 = scheduler.submit([2, 3, 4, 5], max_new_tokens=2)
    scheduler.admit_pending()
    scheduler.next_prefill_work(chunk_size=1)
    scheduler.next_prefill_work(chunk_size=4)

    key = scheduler.shape_key(mode=WorkKind.DECODE, top_k=8, experts_per_token=8, replay_steps=2)
    assert key.mode is WorkKind.DECODE
    assert key.active_c == 2
    assert key.context_bucket == 4
    assert key.active_mask == (True, True, False, False)
    assert key.top_k == 8
    assert key.experts_per_token == 8
    assert key.replay_steps == 2

    graph = scheduler.graph_buckets.get_or_create(key, lambda bucket: {"bucket": bucket})
    assert graph == {"bucket": key}
    assert scheduler.graph_buckets.stats.entries == 1
    assert scheduler.graph_buckets.stats.hits == 0
    assert scheduler.graph_buckets.stats.misses == 1
    assert scheduler.graph_buckets.get(key) is graph
    assert scheduler.graph_buckets.stats.hits == 1

    done = scheduler.record_generated([GeneratedToken(r0, 99)])
    assert [item.request_id for item in done] == [r0]
    assert scheduler.completed[r0].generated_tokens == (99,)
    assert not scheduler.completed[r0].prompt_tokens == ()

    scheduler.record_generated([(r1, 100), (r1, 101)])
    assert scheduler.completed[r1].generated_tokens == (100, 101)
    assert scheduler.active_count == 0


def test_graph_bucket_cache_clear_resets_entries_and_counters() -> None:
    cache = GraphBucketCache()
    scheduler = ResidentBatchScheduler(capacity=1, context_bucket_size=4)
    scheduler.submit([1], max_new_tokens=1)
    scheduler.admit_pending()
    scheduler.next_prefill_work(chunk_size=1)
    key = scheduler.shape_key(mode="decode")

    assert cache.get(key) is None
    cache.put(key, object())
    assert cache.stats.entries == 1
    cache.clear()
    assert cache.stats.entries == 0
    assert cache.stats.hits == 0
    assert cache.stats.misses == 0


def test_compact_prompt_slab_tracks_optional_physical_slots() -> None:
    slab = CompactPromptSlab.from_token_rows(
        request_ids=(10, 11),
        token_rows=((1,), (2, 3)),
        start_positions=(0, 4),
        block_count=1,
        slot_ids=(1, 0),
    )

    assert slab.slot_ids == (1, 0)
    assert slab.physical_slot_ids == (1, 0)

    legacy = CompactPromptSlab.from_token_rows(
        request_ids=(3,), token_rows=((4,),), start_positions=(0,), block_count=1
    )
    assert legacy.physical_slot_ids == (3,)

    with pytest.raises(ValueError, match="slot_ids"):
        CompactPromptSlab.from_token_rows(
            request_ids=(10, 11),
            token_rows=((1,), (2,)),
            start_positions=(0, 0),
            block_count=1,
            slot_ids=(0,),
        )


def test_compact_prompt_slab_validates_cu_seqlens_and_row_shapes() -> None:
    with pytest.raises(ValueError, match="cu_seqlens must end"):
        CompactPromptSlab(
            request_ids=(1,),
            token_ids=(10, 11),
            positions=(0, 1),
            cu_seqlens_q=(0, 1),
            cu_seqlens_k=(0, 2),
            row_to_request=(1, 1),
            block_tables=((0,), (0,)),
            append_counts=(0, 1),
            context_counts=(1, 2),
            token_rows=((10, 11),),
            block_count=1,
        )
    with pytest.raises(ValueError, match="block_tables rows"):
        CompactPromptSlab.from_token_rows(
            request_ids=(1,),
            token_rows=((10,),),
            start_positions=(0,),
            block_count=2,
            block_tables_by_request=((0,),),
        )


def test_resident_batch_scheduler_rejects_duplicate_ids_and_invalid_chunks() -> None:
    scheduler = ResidentBatchScheduler(capacity=1)
    scheduler.submit([1], max_new_tokens=1, request_id=7)
    with pytest.raises(ValueError, match="already exists"):
        scheduler.submit([2], max_new_tokens=1, request_id=7)
    scheduler.admit_pending()
    with pytest.raises(ValueError, match="chunk_size"):
        scheduler.next_prefill_work(chunk_size=0)


def test_qwen35_batch_serial_bench_helpers_summarize_and_slice(tmp_path) -> None:
    fixture = tmp_path / "fixture.json"
    fixture.write_text(json.dumps({"prompt_ids": list(range(12))}))

    assert _load_prompt_slices(fixture, prompt_length=3, batch_size=4) == [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
        [9, 10, 11],
    ]
    with pytest.raises(ValueError, match="need at least"):
        _load_prompt_slices(fixture, prompt_length=4, batch_size=4)

    empty = _summarize_samples([])
    assert empty["median"] is None
    stats = _summarize_samples([3.0, 1.0, 2.0, 10.0])
    assert stats["samples"] == [3.0, 1.0, 2.0, 10.0]
    assert stats["median"] == 2.5
    assert stats["p95"] == 10.0
    assert stats["min"] == 1.0
    assert stats["max"] == 10.0
    assert stats["stdev"] > 0.0


def test_qwen35_batch_diagnostic_artifact_schema_requires_label_fields() -> None:
    payload = {
        "status": "blocked",
        "performance_claim": False,
        "workload": {
            "native_compact_prefill": True,
            "native_caware_decode": False,
        },
        "correctness": {"passed": True},
        "execution": {
            "batch_execution": {
                "native_compact_prefill": True,
                "native_caware_decode": False,
                "throughput_claim_eligible": False,
            }
        },
        "decision": {"accepted": False},
    }

    validate_cn_diagnostic_artifact_payload(payload)

    missing = dict(payload)
    missing["execution"] = {"batch_execution": {"native_compact_prefill": True}}

    with pytest.raises(ValueError, match="native_caware_decode"):
        validate_cn_diagnostic_artifact_payload(missing)


def test_qwen35_batch_diagnostic_artifact_schema_rejects_missing_correctness() -> None:
    payload = {
        "status": "blocked",
        "performance_claim": False,
        "workload": {
            "native_compact_prefill": False,
            "native_caware_decode": False,
        },
        "execution": {
            "batch_execution": {
                "native_compact_prefill": False,
                "native_caware_decode": False,
                "throughput_claim_eligible": False,
            }
        },
        "decision": {"accepted": False},
    }

    with pytest.raises(ValueError, match="correctness"):
        validate_cn_diagnostic_artifact_payload(payload)
