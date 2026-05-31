from __future__ import annotations

import pytest

from hipengine.dispatch import ActiveBatch, BatchShapeKey, RequestState, WorkItem, WorkKind


def test_request_state_tracks_prefill_and_decode_progress() -> None:
    req = RequestState.from_tokens(7, [10, 11, 12], max_new_tokens=2)

    req, chunk = req.take_prefill(2)
    assert chunk == (10, 11)
    assert req.context_len == 2
    assert req.remaining_prefill == 1

    req, chunk = req.take_prefill(8)
    assert chunk == (12,)
    assert req.context_len == 3
    assert req.remaining_prefill == 0

    req = req.append_generated(99)
    assert req.context_len == 4
    assert not req.finished
    req = req.append_generated(100)
    assert req.generated_tokens == (99, 100)
    assert req.finished


def test_active_batch_admits_finishes_and_compacts_stable_requests() -> None:
    batch = ActiveBatch(capacity=4)
    assert batch.active_mask == (False, False, False, False)

    assert batch.admit(RequestState.from_tokens(101, [1, 2], max_new_tokens=4)) == 0
    assert batch.admit(RequestState.from_tokens(202, [3], max_new_tokens=4)) == 1
    assert batch.admit(RequestState.from_tokens(303, [4, 5, 6], max_new_tokens=4)) == 2
    assert batch.request_to_slot == {101: 0, 202: 1, 303: 2}
    assert batch.slot_to_request == (101, 202, 303, None)
    assert batch.active_mask == (True, True, True, False)

    batch.finish(202)
    assert batch.active_mask == (True, False, True, False)
    assert batch.requests[202].finished

    assert batch.admit(RequestState.from_tokens(404, [7], max_new_tokens=1)) == 1
    assert batch.request_to_slot == {101: 0, 404: 1, 303: 2}

    moves = batch.compact(order=(303, 101, 404))
    assert [(move.request_id, move.old_slot, move.new_slot) for move in moves] == [
        (303, 2, 0),
        (101, 0, 1),
        (404, 1, 2),
    ]
    assert batch.slot_to_request == (303, 101, 404, None)
    assert batch.request_to_slot == {303: 0, 101: 1, 404: 2}

    reclaimed = batch.reclaim(202)
    assert reclaimed.request_id == 202
    assert 202 not in batch.requests


def test_active_batch_row_maps_are_slot_and_request_shaped() -> None:
    batch = ActiveBatch(capacity=4)
    batch.admit(RequestState.from_tokens(11, [1], max_new_tokens=1))
    batch.admit(RequestState.from_tokens(22, [2], max_new_tokens=1))
    batch.admit(RequestState.from_tokens(33, [3], max_new_tokens=1))
    batch.finish(22)

    assert batch.active_mask == (True, False, True, False)
    assert batch.row_map(rows_per_request=3) == (0, 0, 0, 2, 2, 2)
    assert batch.request_row_map(rows_per_request=3) == (11, 11, 11, 33, 33, 33)

    with pytest.raises(KeyError):
        batch.row_map(rows_per_request=1, request_ids=(22,))
    with pytest.raises(ValueError, match="rows_per_request"):
        batch.row_map(rows_per_request=0)


def test_batch_shape_key_includes_context_bucket_mask_and_mode() -> None:
    batch = ActiveBatch(capacity=4)
    first = RequestState.from_tokens(1, [10, 11, 12], max_new_tokens=3)
    first, _ = first.take_prefill(3)
    first = first.append_generated(50)
    second = RequestState.from_tokens(2, [20], max_new_tokens=3)
    second, _ = second.take_prefill(1)
    batch.admit(first)
    batch.admit(second)

    key = batch.shape_key(
        mode=WorkKind.DECODE,
        context_bucket_size=4,
        top_k=8,
        experts_per_token=8,
        replay_steps=2,
    )

    assert key == BatchShapeKey(
        mode=WorkKind.DECODE,
        active_c=2,
        context_bucket=4,
        active_mask=(True, True, False, False),
        top_k=8,
        experts_per_token=8,
        replay_steps=2,
    )

    int8_key = batch.shape_key(
        mode=WorkKind.DECODE,
        context_bucket_size=4,
        kv_storage_dtype="int8_per_token_head",
        layer_plan="max_layers=8",
    )
    assert int8_key.kv_storage_dtype == "int8_per_token_head"
    assert int8_key.layer_plan == "max_layers=8"
    assert int8_key != key

    with pytest.raises(ValueError, match="kv_storage_dtype"):
        BatchShapeKey(mode=WorkKind.DECODE, active_c=0, context_bucket=0, active_mask=(), kv_storage_dtype="")
    with pytest.raises(ValueError, match="layer_plan"):
        BatchShapeKey(mode=WorkKind.DECODE, active_c=0, context_bucket=0, active_mask=(), layer_plan="")

    verify_key = batch.shape_key(
        mode="verify_tree",
        context_bucket_size=4,
        draft_depth=3,
        tree_shape=(1, 2, 4),
    )
    assert verify_key.mode is WorkKind.VERIFY_TREE
    assert verify_key.draft_depth == 3
    assert verify_key.tree_shape == (1, 2, 4)


def test_work_item_validates_request_and_verify_metadata() -> None:
    item = WorkItem(
        kind=WorkKind.DECODE,
        request_ids=(1, 2),
        row_to_request=(1, 2),
    )
    assert item.kind is WorkKind.DECODE

    with pytest.raises(ValueError, match="row_to_request"):
        WorkItem(kind=WorkKind.DECODE, request_ids=(1,), row_to_request=(2,))
    with pytest.raises(ValueError, match="positive draft_depth"):
        WorkItem(kind=WorkKind.VERIFY_CHAIN, request_ids=(1,), row_to_request=(1,))
