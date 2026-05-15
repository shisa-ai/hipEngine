from __future__ import annotations

from types import SimpleNamespace

import pytest

from hipengine.core.device import Device
from hipengine.core.tensor import Tensor
from hipengine.dispatch import WorkItem, WorkKind
from hipengine.kvcache import FixedPagedKVPolicy, KVLiveSpans, KVTransaction


def _tensor(ptr: int, shape: tuple[int, ...], dtype: str, device: Device | None = None) -> Tensor:
    return Tensor.from_handle(ptr, shape, dtype, device or Device("hip", 0))


def _register(policy: FixedPagedKVPolicy, request_id: int, *, ptr_base: int) -> None:
    policy.register(
        request_id,
        block_table=_tensor(ptr_base, (4,), "int32"),
        live_counts=_tensor(ptr_base + 0x100, (1,), "int64"),
        max_live_count=3,
    )


def test_kv_live_spans_accepts_batch_row_metadata() -> None:
    spans = KVLiveSpans.paged_uniform(
        block_table=_tensor(0x1000, (2, 4), "int32"),
        live_counts=_tensor(0x2000, (2,), "int64"),
        max_live_count=5,
        storage_dtype="bf16",
        request_ids=_tensor(0x3000, (2,), "int64"),
        row_positions=_tensor(0x4000, (2,), "int32"),
        span_role="verify_tree",
    )

    assert spans.request_ids is not None and spans.request_ids.ptr == 0x3000
    assert spans.row_positions is not None and spans.row_positions.ptr == 0x4000
    assert spans.span_role == "verify_tree"
    assert spans.live_counts.numel == 2


def test_kv_live_spans_validates_batch_row_metadata() -> None:
    with pytest.raises(ValueError, match="request_ids must be int64"):
        KVLiveSpans.paged_uniform(
            block_table=_tensor(1, (2, 4), "int32"),
            live_counts=_tensor(2, (2,), "int64"),
            max_live_count=1,
            storage_dtype="bf16",
            request_ids=_tensor(3, (2,), "int32"),
        )
    with pytest.raises(ValueError, match="one entry"):
        KVLiveSpans.paged_uniform(
            block_table=_tensor(1, (2, 4), "int32"),
            live_counts=_tensor(2, (2,), "int64"),
            max_live_count=1,
            storage_dtype="bf16",
            request_ids=_tensor(3, (1,), "int64"),
        )
    with pytest.raises(ValueError, match="span_role"):
        KVLiveSpans.paged_uniform(
            block_table=_tensor(1, (2, 4), "int32"),
            live_counts=_tensor(2, (2,), "int64"),
            max_live_count=1,
            storage_dtype="bf16",
            span_role="draft",
        )


def test_fixed_paged_policy_c1_spans_and_admission_cap() -> None:
    policy = FixedPagedKVPolicy(block_size=16, storage_dtype="bf16")
    _register(policy, 101, ptr_base=0x1000)

    req = SimpleNamespace(request_id=101)
    spans = policy.batch_spans([req])

    assert spans.base_offsets.ptr == 0x1000
    assert spans.live_counts.ptr == 0x1100
    assert spans.max_live_count == 3
    assert spans.span_role == "decode"
    assert policy.admission_cap(req) == 64 - 3


def test_fixed_paged_policy_requires_packed_metadata_for_c_gt_1() -> None:
    policy = FixedPagedKVPolicy(block_size=16)
    _register(policy, 1, ptr_base=0x1000)
    _register(policy, 2, ptr_base=0x2000)

    with pytest.raises(ValueError, match="packed block_table"):
        policy.batch_spans([1, 2])

    spans = policy.batch_spans(
        [1, 2],
        role="prefill",
        block_table=_tensor(0xA000, (2, 4), "int32"),
        live_counts=_tensor(0xB000, (2,), "int64"),
        request_ids=_tensor(0xC000, (2,), "int64"),
        row_positions=_tensor(0xD000, (2,), "int64"),
        max_live_count=7,
    )

    assert spans.base_offsets.shape == (2, 4)
    assert spans.live_counts.shape == (2,)
    assert spans.span_role == "prefill"
    assert spans.request_ids is not None and spans.request_ids.ptr == 0xC000
    assert spans.row_positions is not None and spans.row_positions.ptr == 0xD000


def test_fixed_paged_policy_rejects_duplicate_transaction_requests() -> None:
    with pytest.raises(ValueError, match="unique"):
        KVTransaction(transaction_id=0, request_ids=(1, 1), draft_rows=1, role="verify_chain")

    policy = FixedPagedKVPolicy()
    _register(policy, 1, ptr_base=0x1000)
    draft = WorkItem(kind=WorkKind.VERIFY_CHAIN, request_ids=(1,), row_to_request=(1,), draft_depth=1)

    with pytest.raises(ValueError, match="unique"):
        policy.begin_transaction([1, SimpleNamespace(request_id=1)], draft)


def test_fixed_paged_policy_transaction_commit_and_rollback() -> None:
    policy = FixedPagedKVPolicy()
    _register(policy, 1, ptr_base=0x1000)
    _register(policy, 2, ptr_base=0x2000)
    draft = WorkItem(kind=WorkKind.VERIFY_CHAIN, request_ids=(1, 2), row_to_request=(1, 1, 2), draft_depth=2)

    txn = policy.begin_transaction([1, 2], draft)
    assert txn == KVTransaction(transaction_id=0, request_ids=(1, 2), draft_rows=3, role="verify_chain", candidate_counts=(2, 1))

    committed = policy.commit(txn, [2, 1])
    assert committed.committed
    assert committed.accepted_counts == (2, 1)
    with pytest.raises(ValueError, match="committed"):
        policy.rollback(committed)
    with pytest.raises(ValueError, match="candidate_counts"):
        policy.commit(txn, [3, 0])

    txn2 = policy.begin_transaction([1], WorkItem(kind=WorkKind.VERIFY_TREE, request_ids=(1,), row_to_request=(1,), draft_depth=1))
    rolled_back = policy.rollback(txn2)
    assert rolled_back.rolled_back
    with pytest.raises(ValueError, match="rolled-back"):
        policy.commit(rolled_back, [0])


def test_fixed_paged_policy_reclaims_reservations() -> None:
    policy = FixedPagedKVPolicy()
    _register(policy, 77, ptr_base=0x7000)

    reservation = policy.reclaim(77)
    assert reservation.request_id == 77
    with pytest.raises(KeyError):
        policy.admission_cap(77)
