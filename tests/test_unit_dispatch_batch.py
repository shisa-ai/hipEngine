from __future__ import annotations

import pytest

from hipengine.dispatch import (
    ActiveBatch,
    BatchShapeKey,
    PhysicalBatchGroup,
    RequestState,
    WorkItem,
    WorkKind,
    plan_physical_batch_groups,
)


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


def test_request_state_transitions_do_not_revalidate_immutable_token_history() -> None:
    int_calls: list[int] = []

    class CountedToken:
        def __init__(self, value: int) -> None:
            self.value = int(value)

        def __int__(self) -> int:
            int_calls.append(self.value)
            return self.value

    request = RequestState(
        request_id=7,
        prompt_tokens=(CountedToken(10), CountedToken(11)),
        max_new_tokens=4,
        generated_tokens=(CountedToken(12),),
    )
    assert int_calls == [10, 11, 12]

    int_calls.clear()
    advanced = request.append_generated(13)

    assert int_calls == []
    assert tuple(int(token) for token in advanced.generated_tokens) == (12, 13)


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
        slot_ids=(0, 2),
        active_mask=(True, False, True, False),
    )
    assert item.kind is WorkKind.DECODE
    assert item.slot_ids == (0, 2)
    assert item.active_mask == (True, False, True, False)

    with pytest.raises(ValueError, match="row_to_request"):
        WorkItem(kind=WorkKind.DECODE, request_ids=(1,), row_to_request=(2,))
    with pytest.raises(ValueError, match="slot_ids"):
        WorkItem(
            kind=WorkKind.DECODE,
            request_ids=(1, 2),
            row_to_request=(1, 2),
            slot_ids=(0,),
        )
    with pytest.raises(ValueError, match="active_mask"):
        WorkItem(
            kind=WorkKind.DECODE,
            request_ids=(1, 2),
            row_to_request=(1, 2),
            slot_ids=(0, 2),
            active_mask=(True, True, False),
        )
    with pytest.raises(ValueError, match="positive draft_depth"):
        WorkItem(kind=WorkKind.VERIFY_CHAIN, request_ids=(1,), row_to_request=(1,))


@pytest.mark.parametrize(
    ("logical_c", "expected_widths", "expected_masks"),
    [
        (3, (4,), ((True, True, True, False),)),
        (5, (8,), ((True, True, True, True, True, False, False, False),)),
        (6, (8,), ((True, True, True, True, True, True, False, False),)),
        (7, (8,), ((True, True, True, True, True, True, True, False),)),
        (9, (8, 1), ((True,) * 8, (True,))),
        (13, (8, 8), ((True,) * 8, (True, True, True, True, True, False, False, False))),
    ],
)
def test_physical_batch_group_plan_lowers_arbitrary_c_to_declared_buckets(
    logical_c: int,
    expected_widths: tuple[int, ...],
    expected_masks: tuple[tuple[bool, ...], ...],
) -> None:
    request_ids = tuple(range(100, 100 + logical_c))
    work = WorkItem(
        kind=WorkKind.DECODE,
        request_ids=request_ids,
        row_to_request=request_ids,
        slot_ids=tuple(range(logical_c)),
        active_mask=(True,) * logical_c,
    )

    groups = plan_physical_batch_groups(work, physical_bucket_widths=(1, 2, 4, 8))

    assert tuple(group.physical_rows for group in groups) == expected_widths
    assert tuple(group.active_mask for group in groups) == expected_masks
    assert tuple(group.logical_c for group in groups) == (logical_c,) * len(groups)
    assert tuple(group.group_index for group in groups) == tuple(range(len(groups)))
    assert tuple(group.group_count for group in groups) == (len(groups),) * len(groups)
    assert tuple(request_id for group in groups for request_id in group.request_ids) == request_ids
    assert set(group.physical_rows for group in groups) <= {1, 2, 4, 8}


def test_physical_batch_group_plan_preserves_sparse_global_slots_without_compaction() -> None:
    work = WorkItem(
        kind=WorkKind.DECODE,
        request_ids=(10, 12, 18, 22),
        row_to_request=(10, 12, 18, 22),
        slot_ids=(0, 2, 8, 12),
        active_mask=(
            True,
            False,
            True,
            False,
            False,
            False,
            False,
            False,
            True,
            False,
            False,
            False,
            True,
        ),
    )

    groups = plan_physical_batch_groups(work, physical_bucket_widths=(1, 2, 4, 8))

    assert groups == (
        PhysicalBatchGroup(
            logical_c=4,
            group_index=0,
            group_count=2,
            physical_slot_base=0,
            physical_slot_extent=8,
            physical_rows=8,
            request_ids=(10, 12),
            global_slot_indices=(0, 2),
            active_slot_indices=(0, 2),
            active_mask=(True, False, True, False, False, False, False, False),
        ),
        PhysicalBatchGroup(
            logical_c=4,
            group_index=1,
            group_count=2,
            physical_slot_base=8,
            physical_slot_extent=5,
            physical_rows=8,
            request_ids=(18, 22),
            global_slot_indices=(8, 12),
            active_slot_indices=(0, 4),
            active_mask=(True, False, False, False, True, False, False, False),
        ),
    )
    assert groups[1].to_json_dict() == {
        "logical_c": 4,
        "group_index": 1,
        "group_count": 2,
        "physical_slot_base": 8,
        "physical_slot_extent": 5,
        "physical_rows": 8,
        "active_rows": 2,
        "request_ids": [18, 22],
        "global_slot_indices": [8, 12],
        "active_slot_indices": [0, 4],
        "active_mask": [True, False, False, False, True, False, False, False],
    }


def test_physical_batch_group_plan_compacts_only_execution_rows_for_adaptive_widths() -> None:
    work = WorkItem(
        kind=WorkKind.DECODE,
        request_ids=(10, 12, 18),
        row_to_request=(10, 12, 18),
        slot_ids=(0, 2, 7),
        active_mask=(True, False, True, False, False, False, False, True),
    )

    groups = plan_physical_batch_groups(
        work,
        physical_bucket_widths=(1, 2, 4, 8),
        compact_active_rows=True,
    )

    assert groups == (
        PhysicalBatchGroup(
            logical_c=3,
            group_index=0,
            group_count=1,
            physical_slot_base=0,
            physical_slot_extent=3,
            physical_rows=4,
            request_ids=(10, 12, 18),
            global_slot_indices=(0, 2, 7),
            active_slot_indices=(0, 1, 2),
            active_mask=(True, True, True, False),
            dense_execution_rows=True,
        ),
    )
    assert groups[0].to_json_dict()["execution_row_mapping"] == "dense_active_rows"


def test_physical_batch_group_plan_adaptive_c9_uses_c8_plus_c1() -> None:
    request_ids = tuple(range(100, 109))
    work = WorkItem(
        kind=WorkKind.DECODE,
        request_ids=request_ids,
        row_to_request=request_ids,
        slot_ids=tuple(range(9)),
        active_mask=(True,) * 9,
    )

    groups = plan_physical_batch_groups(
        work,
        physical_bucket_widths=(1, 2, 4, 8),
        compact_active_rows=True,
    )

    assert tuple(group.physical_rows for group in groups) == (8, 1)
    assert tuple(group.request_ids for group in groups) == (request_ids[:8], request_ids[8:])
    assert tuple(group.active_mask for group in groups) == ((True,) * 8, (True,))
    assert all(group.dense_execution_rows for group in groups)


def test_physical_batch_group_plan_width_sequence_overrides_ceiling() -> None:
    request_ids = tuple(range(100, 113))
    work = WorkItem(
        kind=WorkKind.DECODE,
        request_ids=request_ids,
        row_to_request=request_ids,
        slot_ids=tuple(range(13)),
        active_mask=(True,) * 13,
    )

    # Ceiling with direct widths: 8 + 5.
    ceiling = plan_physical_batch_groups(
        work,
        physical_bucket_widths=(1, 2, 3, 4, 5, 6, 7, 8),
        compact_active_rows=True,
    )
    assert tuple(group.physical_rows for group in ceiling) == (8, 5)

    # D2 width_sequence: 7 + 6.
    d2 = plan_physical_batch_groups(
        work,
        physical_bucket_widths=(1, 2, 3, 4, 5, 6, 7, 8),
        compact_active_rows=True,
        width_sequence=(7, 6),
    )
    assert tuple(group.physical_rows for group in d2) == (7, 6)
    assert tuple(group.request_ids for group in d2) == (request_ids[:7], request_ids[7:])
    assert tuple(group.active_mask for group in d2) == ((True,) * 7, (True,) * 6)
    assert all(group.dense_execution_rows for group in d2)

    # Invalid width_sequence fails closed before model work.
    with pytest.raises(ValueError, match="cover the active row count exactly"):
        plan_physical_batch_groups(
            work,
            physical_bucket_widths=(1, 2, 3, 4, 5, 6, 7, 8),
            compact_active_rows=True,
            width_sequence=(7, 7),
        )
    with pytest.raises(ValueError, match="declared buckets"):
        plan_physical_batch_groups(
            work,
            physical_bucket_widths=(1, 2, 4, 8),
            compact_active_rows=True,
            width_sequence=(9, 4),
        )
    with pytest.raises(ValueError, match="declared buckets"):
        plan_physical_batch_groups(
            work,
            physical_bucket_widths=(1, 2, 4, 8),
            compact_active_rows=True,
            width_sequence=(8, 3, 2),
        )


def test_physical_batch_group_plan_validates_declared_widths_and_slot_metadata() -> None:
    work = WorkItem(
        kind=WorkKind.DECODE,
        request_ids=(1, 2),
        row_to_request=(1, 2),
        slot_ids=(0, 1),
        active_mask=(True, True),
    )

    with pytest.raises(ValueError, match="physical_bucket_widths"):
        plan_physical_batch_groups(work, physical_bucket_widths=())
    with pytest.raises(ValueError, match="strictly increasing"):
        plan_physical_batch_groups(work, physical_bucket_widths=(1, 4, 2))
    with pytest.raises(ValueError, match="positive"):
        plan_physical_batch_groups(work, physical_bucket_widths=(0, 1, 2))

    without_slots = WorkItem(
        kind=WorkKind.DECODE,
        request_ids=(3, 4, 5),
        row_to_request=(3, 4, 5),
    )
    assert plan_physical_batch_groups(
        without_slots,
        physical_bucket_widths=(1, 2, 4, 8),
    )[0].to_json_dict() == {
        "logical_c": 3,
        "group_index": 0,
        "group_count": 1,
        "physical_slot_base": 0,
        "physical_slot_extent": 3,
        "physical_rows": 4,
        "active_rows": 3,
        "request_ids": [3, 4, 5],
        "global_slot_indices": [0, 1, 2],
        "active_slot_indices": [0, 1, 2],
        "active_mask": [True, True, True, False],
    }
