"""Independent KV page addressing oracle, without production copy helpers."""
from types import SimpleNamespace as NS
import pytest
from scripts import qwen38_packed_c1_kv as m


def test_paged_cross_boundary_and_chunk_origin():
    assert m.page_rows((12, 10), 255, 3, chunk_start=10) == (767, 0, 1)


def test_resident_nonzero_slot_and_paged_override():
    s = NS(_resident_slot_index=3, _resident_batch_owner=NS(_target_scratch_owner=NS(max_positions=1024)))
    assert m.resident_rows(s, 255, 3) == (3327, 3328, 3329)
    s._device_kv_allocation = NS(block_ids=(12, 10), chunk_start_block_id=10)
    assert m.resident_rows(s, 255, 3) == (767, 0, 1)


def test_packed_slot_uses_its_own_pages():
    p = NS(slot_count=2, blocks_per_slot=2, block_size=256, page_ids=(8, 6, 9, 4))
    assert m.packed_rows(p, 1, 255, 3) == (2559, 1024, 1025)


@pytest.mark.parametrize('pages,start,count,origin', [((1,), -1, 1, 0), ((1,), 0, -1, 0), ((1,), 255, 2, 0), ((1,), 0, 1, 2), ((1,1), 0, 1, 0)])
def test_invalid_page_mapping_fails_closed(pages, start, count, origin):
    with pytest.raises(ValueError):
        m.page_rows(pages, start, count, chunk_start=origin)


def test_slot_and_capacity_fail_closed():
    p = NS(slot_count=2, blocks_per_slot=1, block_size=256, page_ids=(8, 6))
    with pytest.raises(ValueError):
        m.packed_rows(p, 2, 0, 1)
    s = NS(_resident_slot_index=1)
    with pytest.raises(ValueError):
        m.resident_rows(s, 0, 1)


@pytest.mark.parametrize('fault', [None, 'old', 'new', 'page', 'pointer'])
def test_selected_kv_commit_preserves_old_and_copies_new(monkeypatch, fault):
    from hipengine.core import DType
    # Two-byte rows, logical 255/256 straddles arbitrary physical pages.
    src = NS(ptr=10000, nbytes=4096)
    dst = NS(ptr=20000, nbytes=4096)
    state = NS(kv_layout=NS(storage_dtype=DType.BF16), slot_count=1,
               blocks_per_slot=2, block_size=256, page_ids=(3, 1),
               full_key_caches=(src,), full_value_caches=(src,))
    s = NS(runtime=NS(device_synchronize=lambda: None), kv_storage_dtype=DType.BF16,
           runner=NS(weights=NS(config=NS(head_count_kv=1, key_length=1))),
           scratch=NS(full_key_caches=(dst,), full_value_caches=(dst,)),
           _device_kv_allocation=NS(block_ids=(12, 10), chunk_start_block_id=10))
    r = NS(start_position=255, row_start=4, row_end=6,
           deferred_packed_state=NS(packed_state=state, slot_index=0))
    memory = {10000 + 1023 * 2: 'a', 10000 + 256 * 2: 'b'}
    monkeypatch.setattr(m, '_device_hash', lambda s, b: memory.get(b.ptr, f'old:{b.ptr}:{b.nbytes}'))
    expected = m.selected_kv_sources(s, r, accepted=1)
    memory[20000 + 767 * 2] = 'a'
    memory[20000] = 'b'
    if fault == 'old':
        memory[20000 + 512 * 2] = 'corrupt'
    elif fault == 'new':
        memory[20000] = 'corrupt'
    elif fault == 'page':
        s._device_kv_allocation.block_ids = (10, 12)
    elif fault == 'pointer':
        dst.ptr += 2
    if fault:
        with pytest.raises(ValueError):
            m.assert_kv_commit(s, expected)
    else:
        m.assert_kv_commit(s, expected)


def test_buffer_rows_hash_logical_order_and_bounds(monkeypatch):
    monkeypatch.setattr(m, '_device_hash', lambda s, b: f'{b.ptr}:{b.nbytes}')
    b = NS(ptr=100, nbytes=16)
    assert m.hash_rows(None, b, (3, 0, 1), 4) == ('112:4', '100:8')
    with pytest.raises(ValueError):
        m.hash_rows(None, b, (4,), 4)
