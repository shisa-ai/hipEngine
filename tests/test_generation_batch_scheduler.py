from __future__ import annotations

import json
from dataclasses import replace

import pytest

from hipengine.core.device import Device
from hipengine.core.tensor import Tensor
from hipengine.dispatch import WorkKind
from hipengine.generation import (
    GeneratedToken,
    GraphBucketCache,
    ResidentBatchScheduler,
    SpeculativeCommitPlan,
    SpeculativeStateCommitPlan,
    SpeculativeVerifyBufferPlan,
    SpeculativeVerifyPlan,
    SpeculativeVerifyWork,
)
from hipengine.kvcache import FixedPagedKVPolicy
from hipengine.speculative import AcceptResult, DraftBatch, TargetAcceptSummary, TargetStateCommitBuffers, TargetVerifyBuffers
from scripts.qwen35_batch_serial_bench import _load_prompt_slices, _summarize_samples


def _tensor(ptr: int, shape: tuple[int, ...], dtype: str) -> Tensor:
    return Tensor.from_handle(ptr, shape, dtype, Device("hip", 0))


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
    )
    buffer_plan = scheduler.bind_speculative_verify_buffers(plan, buffers)
    assert isinstance(buffer_plan, SpeculativeVerifyBufferPlan)
    assert buffer_plan.plan is plan
    assert buffer_plan.buffers is buffers

    summary = TargetAcceptSummary.from_accept_result(
        work.target_batch,
        AcceptResult(
            request_ids=(r0, r1),
            accepted_counts=(2, 1),
            accepted_tokens=((101, 102), (201,)),
        ),
    )
    commit = scheduler.plan_speculative_commit(buffer_plan, summary)
    assert isinstance(commit, SpeculativeCommitPlan)
    assert commit.verify_plan is buffer_plan
    assert commit.summary is summary
    assert commit.commit_plan.transaction_id == plan.transaction.transaction_id
    assert commit.commit_plan.request_ids == (r0, r1)
    assert commit.commit_plan.accepted_counts == (2, 1)
    assert commit.commit_plan.commit_rows == (3, 4)
    assert commit.commit_plan.candidate_counts == (2, 1)
    assert commit.commit_plan.mode == "verify_tree"
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

    assert [item.request_id for item in completed] == [r1]
    assert scheduler.active_batch.requests[r0].generated_tokens == (101, 102)
    assert scheduler.completed[r1].generated_tokens == (201,)
    assert scheduler.active_batch.slot_to_request == (r0, None)


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
