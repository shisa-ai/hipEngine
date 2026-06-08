from __future__ import annotations

from types import SimpleNamespace

import pytest

from hipengine.core.device import Device
from hipengine.core.tensor import Tensor
from hipengine.dispatch import (
    PagedAttnDecodeKind,
    PagedKVWriteKind,
    WorkItem,
    WorkKind,
    plan_paged_attn_decode,
    plan_paged_kv_write,
)
from hipengine.kvcache import (
    ChunkedKVPool,
    FixedPagedKVPolicy,
    KVLiveSpans,
    KVScaleMetadata,
    KVTransaction,
    RadixCache,
    resolve_kv_policy,
    resolve_prefix_cache_mode,
)


def _tensor(ptr: int, shape: tuple[int, ...], dtype: str, device: Device | None = None) -> Tensor:
    return Tensor.from_handle(ptr, shape, dtype, device or Device("hip", 0))


def _scale_metadata(ptr_base: int, *, shape: tuple[int, ...] = (4, 16, 2)) -> KVScaleMetadata:
    return KVScaleMetadata(
        k_scale=_tensor(ptr_base, shape, "fp16"),
        v_scale=_tensor(ptr_base + 0x100, shape, "fp16"),
    )


def _register(policy: FixedPagedKVPolicy, request_id: int, *, ptr_base: int) -> None:
    scale_metadata = _scale_metadata(ptr_base + 0x200) if policy.storage_dtype.value == "int8_per_token_head" else None
    policy.register(
        request_id,
        block_table=_tensor(ptr_base, (4,), "int32"),
        live_counts=_tensor(ptr_base + 0x100, (1,), "int64"),
        max_live_count=3,
        scale_metadata=scale_metadata,
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


def test_resolve_kv_policy_records_explicit_and_admission_selection() -> None:
    default = resolve_kv_policy("auto", block_size=256)
    assert default.storage_dtype.value == "bf16"
    assert default.to_json_dict()["int8_explicit"] is False
    assert default.to_json_dict()["int8_admission_gated"] is False
    assert default.to_json_dict()["scale_metadata_format"]["present"] is False

    explicit = resolve_kv_policy("int8_per_token_head", scale_dtype="fp32")
    payload = explicit.to_json_dict()
    assert payload["resolved_storage_dtype"] == "int8_per_token_head"
    assert payload["int8_explicit"] is True
    assert payload["int8_admission_gated"] is False
    assert payload["scale_metadata_format"] == {
        "present": True,
        "scale_dtype": "fp32",
        "granularity": "per_token_head",
        "k_scale": "per_token_head",
        "v_scale": "per_token_head",
    }

    gated = resolve_kv_policy("auto", admission_gated_int8=True)
    assert gated.storage_dtype.value == "int8_per_token_head"
    assert gated.to_json_dict()["int8_explicit"] is False
    assert gated.to_json_dict()["int8_admission_gated"] is True


def test_radix_cache_hits_full_blocks_and_misses_partial_edges() -> None:
    assert resolve_prefix_cache_mode(None) == "off"
    assert resolve_prefix_cache_mode("RADIX") == "radix"
    with pytest.raises(ValueError, match="prefix cache"):
        resolve_prefix_cache_mode("tree")

    cache = RadixCache(block_size=4)
    inserted = cache.insert(1, [10, 11, 12, 13, 14, 15], [100, 101])

    assert inserted.cached_tokens == 4
    assert inserted.cached_blocks == 1
    assert cache.stats.entries == 1

    hit = cache.match([10, 11, 12, 13, 99])
    assert hit.hit is True
    assert hit.matched_tokens == (10, 11, 12, 13)
    assert hit.block_ids == (100,)
    assert hit.remaining_tokens == (99,)

    partial = cache.match([10, 11, 12, 99])
    assert partial.hit is False
    assert partial.matched_tokens == ()
    assert partial.remaining_tokens == (10, 11, 12, 99)
    assert cache.stats.partial_block_misses == 1


def test_radix_cache_cancellation_removes_live_prefix_ownership() -> None:
    cache = RadixCache(block_size=2)
    cache.insert(1, [1, 2, 3, 4], [10, 11])
    cache.insert(2, [1, 2], [10])

    assert cache.match([1, 2, 5]).block_ids == (10,)
    cancel_one = cache.cancel(1)
    assert cancel_one.removed_entries == 2
    assert cancel_one.removed_blocks == (10, 11)
    assert cache.match([1, 2, 5]).block_ids == (10,)
    assert cache.stats.entries == 1

    cancel_two = cache.cancel(2)
    assert cancel_two.removed_entries == 1
    assert cancel_two.removed_blocks == (10,)
    assert cache.match([1, 2, 5]).hit is False
    assert cache.stats.entries == 0


def test_radix_cache_entry_state_is_pointer_independent_kvtc_guardrail() -> None:
    cache = RadixCache(block_size=2)
    cache.insert(1, [1, 2, 3, 4], [10, 11])
    cache.insert(2, [1, 2], [10])

    state = cache.entry_state([1, 2])
    assert state.block_ids == (10,)
    assert state.owner_request_ids == (1, 2)
    assert state.refcount == 2
    assert state.eviction_state == "resident"
    assert "pointer" not in state.to_json_dict()

    tiered = cache.mark_entry_eviction_state([1, 2], "tiered:host")
    assert tiered.block_ids == (10,)
    assert tiered.refcount == 2
    assert tiered.eviction_state == "tiered:host"
    assert cache.match([1, 2, 99]).block_ids == (10,)

    states = {entry.matched_tokens: entry for entry in cache.entry_states()}
    assert states[(1, 2)].block_ids == (10,)
    assert states[(1, 2, 3, 4)].block_ids == (10, 11)
    assert states[(1, 2)].to_json_dict() == {
        "matched_tokens": [1, 2],
        "block_ids": [10],
        "owner_request_ids": [1, 2],
        "refcount": 2,
        "eviction_state": "tiered:host",
    }

    cache.cancel(1)
    shared = cache.entry_state([1, 2])
    assert shared.block_ids == (10,)
    assert shared.owner_request_ids == (2,)
    assert shared.refcount == 1
    assert shared.eviction_state == "tiered:host"

    cache.cancel(2)
    with pytest.raises(KeyError, match="no live prefix cache entry"):
        cache.entry_state([1, 2])


def test_chunked_kv_pool_grows_and_shrinks_on_burst_idle() -> None:
    pool = ChunkedKVPool(
        page_bytes=1024,
        initial_pages=2,
        low_water_pages=2,
        high_water_pages=6,
        chunk_pages=2,
        idle_grace_seconds=5.0,
    )

    burst = pool.allocate(5, now_seconds=1.0)

    assert burst.block_ids == (0, 1, 2, 3, 4)
    assert burst.pointers == tuple(pool.pointer_for(block_id) for block_id in burst.block_ids)
    assert pool.stats.grow_events == 2
    assert pool.stats.current_pages == 6
    assert pool.stats.free_pages == 1
    assert pool.stats.refcounted_pages == 5
    assert pool.stats.high_water_observed_pages == 6

    pool.release(list(burst.block_ids), now_seconds=2.0)
    assert pool.shrink_idle(now_seconds=4.0) == 0
    assert pool.shrink_idle(now_seconds=8.0) == 4

    stats = pool.stats
    assert stats.current_pages == 2
    assert stats.shrink_events == 2
    assert stats.free_pages == 2
    assert stats.refcounted_pages == 0

    again = pool.allocate(1, now_seconds=9.0)
    assert again.block_ids == (0,)
    assert again.pointers == (pool.pointer_for(0),)


def test_chunked_kv_pool_copy_on_write_fork_preserves_prefix_and_splits_suffix() -> None:
    pool = ChunkedKVPool(
        page_bytes=2048,
        initial_pages=4,
        low_water_pages=2,
        high_water_pages=8,
        chunk_pages=2,
        idle_grace_seconds=0.0,
    )
    prefix = pool.allocate(2, now_seconds=1.0)
    prefix_pointers = prefix.pointers

    left = pool.fork_copy_on_write(prefix.block_ids, suffix_pages=1, first_divergent_token=512, now_seconds=2.0)
    right = pool.fork_copy_on_write(prefix.block_ids, suffix_pages=1, first_divergent_token=512, now_seconds=3.0)

    assert left.shared_block_ids == prefix.block_ids
    assert right.shared_block_ids == prefix.block_ids
    assert left.shared_pointers == prefix_pointers
    assert right.shared_pointers == prefix_pointers
    assert left.forked_block_ids != right.forked_block_ids
    assert set(left.forked_block_ids).isdisjoint(right.forked_block_ids)
    assert set(left.forked_pointers).isdisjoint(right.forked_pointers)
    assert pool.refcount(prefix.block_ids[0]) == 3
    assert pool.refcount(prefix.block_ids[1]) == 3
    assert pool.refcount(left.forked_block_ids[0]) == 1
    assert pool.refcount(right.forked_block_ids[0]) == 1
    assert pool.stats.cow_fork_events == 2
    assert pool.stats.cow_forked_pages == 2

    pool.release(prefix.block_ids, now_seconds=4.0)
    assert pool.refcount(prefix.block_ids[0]) == 2
    assert pool.refcount(prefix.block_ids[1]) == 2
    assert pool.stats.free_pages == 0

    pool.release(left.block_ids, now_seconds=5.0)
    assert pool.refcount(prefix.block_ids[0]) == 1
    assert pool.refcount(prefix.block_ids[1]) == 1
    assert pool.refcount(left.forked_block_ids[0]) == 0
    probe = pool.allocate(1, now_seconds=6.0)
    assert probe.block_ids == left.forked_block_ids
    pool.release(probe.block_ids, now_seconds=6.5)

    pool.release(right.block_ids, now_seconds=7.0)
    assert pool.refcount(prefix.block_ids[0]) == 0
    assert pool.refcount(prefix.block_ids[1]) == 0
    assert pool.refcount(right.forked_block_ids[0]) == 0

    with pytest.raises(ValueError, match="copy-on-write"):
        pool.fork_copy_on_write(prefix.block_ids, suffix_pages=0, first_divergent_token=512)


def test_chunked_kv_pool_shared_prefix_refcounts_reclaim_zero_only() -> None:
    pool = ChunkedKVPool(
        page_bytes=2048,
        initial_pages=4,
        low_water_pages=2,
        high_water_pages=6,
        chunk_pages=2,
        idle_grace_seconds=0.0,
    )
    parent = pool.allocate(2, now_seconds=1.0)

    child = pool.admit_with_shared_prefix(parent.block_ids, suffix_pages=1, now_seconds=2.0)

    assert child.reused_block_ids == parent.block_ids
    assert child.allocated_block_ids == (2,)
    assert child.block_ids == (0, 1, 2)
    assert child.pointers == tuple(pool.pointer_for(block_id) for block_id in child.block_ids)
    assert pool.refcount(0) == 2
    assert pool.refcount(1) == 2
    assert pool.refcount(2) == 1
    assert pool.stats.prefix_reuse_events == 1
    assert pool.stats.prefix_reused_pages == 2

    pool.release(parent.block_ids, now_seconds=3.0)

    assert pool.refcount(0) == 1
    assert pool.refcount(1) == 1
    assert pool.refcount(2) == 1
    assert pool.stats.free_pages == 1
    probe = pool.allocate(1, now_seconds=4.0)
    assert probe.block_ids == (3,)
    pool.release(probe.block_ids, now_seconds=5.0)

    pool.release(child.block_ids, now_seconds=6.0)

    assert pool.refcount(0) == 0
    assert pool.refcount(1) == 0
    assert pool.refcount(2) == 0
    assert pool.stats.free_pages == 4
    assert pool.shrink_idle(now_seconds=6.0) == 0

    with pytest.raises(ValueError, match="free prefix"):
        pool.admit_with_shared_prefix((0,), suffix_pages=0, now_seconds=7.0)


def test_chunked_kv_pool_preserves_refcounted_tail_and_capacity_failures() -> None:
    pool = ChunkedKVPool(
        page_bytes=4096,
        initial_pages=1,
        low_water_pages=1,
        high_water_pages=3,
        chunk_pages=1,
        idle_grace_seconds=0.0,
    )
    allocation = pool.allocate(3, now_seconds=1.0)
    pool.incref([allocation.block_ids[-1]])
    pool.release([allocation.block_ids[0], allocation.block_ids[1], allocation.block_ids[-1]], now_seconds=2.0)

    assert pool.refcount(allocation.block_ids[-1]) == 1
    assert pool.shrink_idle(now_seconds=2.0) == 0
    assert pool.stats.current_pages == 3

    pool.release([allocation.block_ids[-1]], now_seconds=3.0)
    assert pool.shrink_idle(now_seconds=3.0) == 2
    assert pool.stats.current_pages == 1

    with pytest.raises(MemoryError, match="cannot grow"):
        pool.allocate(4, now_seconds=4.0)
    assert pool.stats.grow_failures == 1


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


def test_fixed_paged_policy_global_admission_cap_tracks_reclaim() -> None:
    policy = FixedPagedKVPolicy(block_size=16, storage_dtype="bf16", total_capacity_tokens=128)
    assert policy.admission_cap() == 128

    policy.register(
        101,
        block_table=_tensor(0x1000, (4,), "int32"),
        live_counts=_tensor(0x2000, (1,), "int64"),
        max_live_count=3,
        capacity_tokens=64,
    )
    policy.register(
        202,
        block_table=_tensor(0x3000, (2,), "int32"),
        live_counts=_tensor(0x4000, (1,), "int64"),
        max_live_count=2,
        capacity_tokens=32,
    )

    assert policy.admission_cap() == 32
    assert policy.admission_cap(SimpleNamespace(request_id=101)) == 32

    policy.reclaim(101)

    assert policy.admission_cap() == 96
    policy.register(
        303,
        block_table=_tensor(0x5000, (6,), "int32"),
        live_counts=_tensor(0x6000, (1,), "int64"),
        max_live_count=0,
        capacity_tokens=96,
    )
    assert policy.admission_cap() == 0

    with pytest.raises(ValueError, match="admission capacity"):
        policy.register(
            404,
            block_table=_tensor(0x7000, (1,), "int32"),
            live_counts=_tensor(0x8000, (1,), "int64"),
            max_live_count=0,
            capacity_tokens=16,
        )


def test_fixed_paged_policy_audits_append_only_block_pointers() -> None:
    policy = FixedPagedKVPolicy(block_size=16, storage_dtype="bf16")
    policy.register(
        101,
        block_table=_tensor(0x1000, (2,), "int32"),
        live_counts=_tensor(0x2000, (1,), "int64"),
        max_live_count=1,
        block_pointer_map={0: 0xA000, 1: 0xB000},
    )

    with pytest.raises(ValueError, match="already live"):
        policy.register(
            202,
            block_table=_tensor(0x3000, (2,), "int32"),
            live_counts=_tensor(0x4000, (1,), "int64"),
            max_live_count=1,
            block_pointer_map={1: 0xB000, 2: 0xC000},
        )

    policy.reclaim(101)
    policy.register(
        303,
        block_table=_tensor(0x5000, (1,), "int32"),
        live_counts=_tensor(0x6000, (1,), "int64"),
        max_live_count=0,
        block_pointer_map={0: 0xA000},
    )
    policy.reclaim(303)

    with pytest.raises(ValueError, match="backing pointer changed"):
        policy.register(
            404,
            block_table=_tensor(0x7000, (1,), "int32"),
            live_counts=_tensor(0x8000, (1,), "int64"),
            max_live_count=0,
            block_pointer_map={0: 0xD000},
        )


def test_fixed_paged_policy_accepts_int8_scale_metadata() -> None:
    policy = FixedPagedKVPolicy(block_size=16, storage_dtype="int8_per_token_head")
    metadata = _scale_metadata(0x3000)
    reservation = policy.register(
        202,
        block_table=_tensor(0x1000, (4,), "int32"),
        live_counts=_tensor(0x2000, (1,), "int64"),
        max_live_count=3,
        scale_metadata=metadata,
    )
    spans = policy.batch_spans([202])

    assert reservation.storage_dtype.value == "int8_per_token_head"
    assert reservation.scale_metadata is metadata
    assert spans.storage_dtype.value == "int8_per_token_head"
    assert spans.scale_metadata is metadata
    assert policy.admission_cap(202) == 64 - 3


def test_fixed_paged_policy_requires_int8_scale_metadata() -> None:
    policy = FixedPagedKVPolicy(block_size=16, storage_dtype="int8_per_token_head")
    with pytest.raises(ValueError, match="require scale metadata"):
        policy.register(
            303,
            block_table=_tensor(0x1000, (4,), "int32"),
            live_counts=_tensor(0x2000, (1,), "int64"),
            max_live_count=3,
        )


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


def test_fixed_paged_policy_per_row_spans_route_bf16_and_int8_dispatch() -> None:
    bf16 = FixedPagedKVPolicy(block_size=16, storage_dtype="bf16")
    _register(bf16, 11, ptr_base=0x11000)
    _register(bf16, 22, ptr_base=0x22000)
    bf16_spans = bf16.batch_spans(
        [11, 22],
        role="decode",
        block_table=_tensor(0xA000, (2, 4), "int32"),
        live_counts=_tensor(0xB000, (2,), "int64"),
        request_ids=_tensor(0xC000, (2,), "int64"),
        row_positions=_tensor(0xD000, (2,), "int32"),
        max_live_count=33,
    )

    assert bf16_spans.request_ids is not None and bf16_spans.request_ids.shape == (2,)
    assert bf16_spans.row_positions is not None and bf16_spans.row_positions.shape == (2,)
    bf16_write = plan_paged_kv_write(
        bf16_spans,
        kind=PagedKVWriteKind.BATCH,
        source_dtype="bf16",
        model_quant="w4_paro",
    )
    bf16_attn = plan_paged_attn_decode(
        bf16_spans,
        kind=PagedAttnDecodeKind.GQA_SPLITK,
        model_quant="w4_paro",
    )
    assert (bf16_write.layer, bf16_write.quant, bf16_write.variant) == (
        "paged_kv_write",
        "w4_paro",
        "mixed_bf16_batch_spans",
    )
    assert (bf16_attn.layer, bf16_attn.quant, bf16_attn.variant) == (
        "paged_attn_decode",
        "w4_paro",
        "bf16_split_k_gqa_spans",
    )

    int8 = FixedPagedKVPolicy(block_size=16, storage_dtype="int8_per_token_head")
    _register(int8, 33, ptr_base=0x33000)
    _register(int8, 44, ptr_base=0x44000)
    packed_scale = _scale_metadata(0x55000, shape=(2, 16, 2))
    int8_spans = int8.batch_spans(
        [33, 44],
        role="decode",
        block_table=_tensor(0xE000, (2, 4), "int32"),
        live_counts=_tensor(0xF000, (2,), "int64"),
        request_ids=_tensor(0xAB00, (2,), "int64"),
        row_positions=_tensor(0xBC00, (2,), "int32"),
        max_live_count=33,
        scale_metadata=packed_scale,
    )

    assert int8_spans.storage_dtype.value == "int8_per_token_head"
    assert int8_spans.scale_metadata is packed_scale
    assert int8_spans.request_ids is not None and int8_spans.request_ids.ptr == 0xAB00
    assert int8_spans.row_positions is not None and int8_spans.row_positions.ptr == 0xBC00
    int8_write = plan_paged_kv_write(
        int8_spans,
        kind=PagedKVWriteKind.BATCH,
        source_dtype="fp32",
        model_quant="w4_paro",
    )
    int8_attn = plan_paged_attn_decode(
        int8_spans,
        kind=PagedAttnDecodeKind.GQA_SPLITK,
        model_quant="w4_paro",
    )
    assert (int8_write.layer, int8_write.quant, int8_write.variant) == (
        "paged_kv_write",
        "int8_per_token_head",
        "per_token_head_batch_spans",
    )
    assert (int8_attn.layer, int8_attn.quant, int8_attn.variant) == (
        "paged_attn_decode",
        "int8_per_token_head",
        "per_token_head_gqa_splitk_spans",
    )


def test_fixed_paged_policy_rejects_duplicate_transaction_requests() -> None:
    with pytest.raises(ValueError, match="unique"):
        KVTransaction(transaction_id=0, request_ids=(1, 1), draft_rows=1, role="verify_chain")

    policy = FixedPagedKVPolicy()
    _register(policy, 1, ptr_base=0x1000)
    draft = WorkItem(kind=WorkKind.VERIFY_CHAIN, request_ids=(1,), row_to_request=(1,), draft_depth=1)

    with pytest.raises(ValueError, match="unique"):
        policy.begin_transaction([1, SimpleNamespace(request_id=1)], draft)


def test_kv_transaction_validates_role() -> None:
    with pytest.raises(ValueError, match="role"):
        KVTransaction(transaction_id=0, request_ids=(1,), draft_rows=1, role="decode")


def test_kv_transaction_validates_terminal_state() -> None:
    with pytest.raises(ValueError, match="requires accepted_counts"):
        KVTransaction(transaction_id=0, request_ids=(1,), draft_rows=1, role="verify_chain", committed=True)
    with pytest.raises(ValueError, match="require committed"):
        KVTransaction(transaction_id=0, request_ids=(1,), draft_rows=1, role="verify_chain", accepted_counts=(1,))
    with pytest.raises(ValueError, match="non-negative"):
        KVTransaction(
            transaction_id=0,
            request_ids=(1,),
            draft_rows=1,
            role="verify_chain",
            committed=True,
            accepted_counts=(-1,),
        )
    with pytest.raises(ValueError, match="candidate_counts"):
        KVTransaction(
            transaction_id=0,
            request_ids=(1,),
            draft_rows=1,
            role="verify_chain",
            candidate_counts=(0,),
            committed=True,
            accepted_counts=(1,),
        )
    with pytest.raises(ValueError, match="draft_rows"):
        KVTransaction(
            transaction_id=0,
            request_ids=(1, 2),
            draft_rows=1,
            role="verify_chain",
            committed=True,
            accepted_counts=(1, 1),
        )


def test_fixed_paged_policy_transaction_commit_and_rollback() -> None:
    policy = FixedPagedKVPolicy()
    _register(policy, 1, ptr_base=0x1000)
    _register(policy, 2, ptr_base=0x2000)
    draft = WorkItem(kind=WorkKind.VERIFY_CHAIN, request_ids=(1, 2), row_to_request=(1, 1, 2), draft_depth=2)

    txn = policy.begin_transaction([1, 2], draft)
    assert txn == KVTransaction(transaction_id=0, request_ids=(1, 2), draft_rows=3, role="verify_chain", candidate_counts=(2, 1))

    with pytest.raises(ValueError, match="candidate_counts"):
        policy.commit(txn, [3, 0])
    committed = policy.commit(txn, [2, 1])
    assert committed.committed
    assert committed.accepted_counts == (2, 1)
    with pytest.raises(ValueError, match="committed"):
        policy.rollback(committed)
    with pytest.raises(ValueError, match="already committed"):
        policy.commit(committed, [2, 1])

    txn2 = policy.begin_transaction([1], WorkItem(kind=WorkKind.VERIFY_TREE, request_ids=(1,), row_to_request=(1,), draft_depth=1))
    rolled_back = policy.rollback(txn2)
    assert rolled_back.rolled_back
    with pytest.raises(ValueError, match="rolled-back"):
        policy.commit(rolled_back, [0])
    with pytest.raises(ValueError, match="already rolled-back"):
        policy.rollback(rolled_back)


def test_fixed_paged_policy_reclaims_reservations() -> None:
    policy = FixedPagedKVPolicy()
    _register(policy, 77, ptr_base=0x7000)

    reservation = policy.reclaim(77)
    assert reservation.request_id == 77
    with pytest.raises(KeyError):
        policy.admission_cap(77)
