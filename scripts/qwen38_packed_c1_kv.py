"""Independent diagnostic KV addressing; never calls runtime copy planners."""
from types import SimpleNamespace
from scripts.gguf_packed_ar_state_oracle import _device_hash


def page_rows(pages, start, count, *, chunk_start=0, block_size=256):
    pages = tuple(map(int, pages))
    if (start < 0 or count < 0 or block_size <= 0 or chunk_start < 0
            or len(set(pages)) != len(pages) or any(p < chunk_start for p in pages)
            or start + count > len(pages) * block_size):
        raise ValueError('invalid KV page range')
    return tuple((pages[i // block_size] - chunk_start) * block_size + i % block_size
                 for i in range(start, start + count))


def resident_rows(session, start, count):
    allocation = getattr(session, '_device_kv_allocation', None)
    if allocation is not None:
        return page_rows(allocation.block_ids, start, count,
                         chunk_start=int(allocation.chunk_start_block_id))
    slot = int(getattr(session, '_resident_slot_index', 0) or 0)
    owner = getattr(session, '_resident_batch_owner', None)
    scratch_owner = getattr(owner, '_target_scratch_owner', None)
    stride = int(scratch_owner.max_positions) if scratch_owner is not None else 0
    if (start < 0 or count < 0 or slot < 0 or (slot and not stride)
            or (stride and start + count > stride)):
        raise ValueError('invalid private KV slot range')
    return tuple(slot * stride + i for i in range(start, start + count))


def packed_rows(state, slot, start, count):
    if not 0 <= slot < int(state.slot_count):
        raise ValueError('invalid packed KV slot')
    width = int(state.blocks_per_slot)
    if width <= 0 or len(state.page_ids) != int(state.slot_count) * width:
        raise ValueError('invalid packed KV page reservation')
    return page_rows(state.page_ids[slot * width:(slot + 1) * width], start, count,
                     block_size=int(state.block_size))


def selected_kv_sources(session, result, *, accepted):
    """Capture old live rows and CPU-selected new rows before device commit."""
    from hipengine.core import DType
    state = result.deferred_packed_state.packed_state
    if state.kv_layout.storage_dtype != DType.BF16 or session.kv_storage_dtype != DType.BF16:
        raise ValueError('KV ownership diagnostic requires BF16 storage')
    start = int(result.start_position)
    if not 0 <= accepted < int(result.row_end) - int(result.row_start):
        raise ValueError('invalid selected KV prefix')
    cfg = session.runner.weights.config
    stride = int(cfg.head_count_kv) * int(cfg.key_length) * 2
    old_rows = resident_rows(session, 0, start)
    new_rows = resident_rows(session, start, accepted + 1)
    sources = packed_rows(state, int(result.deferred_packed_state.slot_index), start, accepted + 1)
    session.runtime.device_synchronize()
    expected = {}
    for plane in ('full_key_caches', 'full_value_caches'):
        for layer, (src, dst) in enumerate(zip(getattr(state, plane), getattr(session.scratch, plane), strict=True)):
            if src is None and dst is None:
                continue
            # One hash per new logical row: source/destination page breaks may differ.
            expected[(plane, layer)] = dict(ptr=int(dst.ptr), nbytes=int(dst.nbytes),
                old=hash_rows(session, dst, old_rows, stride),
                new=tuple(hash_rows(session, src, (row,), stride) for row in sources))
    if not expected:
        raise ValueError('no KV planes checked')
    return dict(start=start, count=accepted + 1, stride=stride, old_rows=old_rows,
                new_rows=new_rows, buffers=expected)


def assert_kv_commit(session, expected):
    session.runtime.device_synchronize()
    if (resident_rows(session, 0, expected['start']) != expected['old_rows']
            or resident_rows(session, expected['start'], expected['count']) != expected['new_rows']):
        raise ValueError('KV page ownership changed during commit')
    for (plane, layer), row in expected['buffers'].items():
        dst = getattr(session.scratch, plane)[layer]
        if (dst is None or int(dst.ptr) != row['ptr'] or int(dst.nbytes) != row['nbytes']
                or hash_rows(session, dst, expected['old_rows'], expected['stride']) != row['old']
                or tuple(hash_rows(session, dst, (i,), expected['stride']) for i in expected['new_rows']) != row['new']):
            raise ValueError(f'selected KV commit mismatch: {plane}:{layer}')


def hash_rows(session, buffer, rows, row_bytes):
    """Hash contiguous physical runs in logical order, validating every address."""
    if buffer is None or int(buffer.ptr) <= 0 or row_bytes <= 0:
        raise ValueError('invalid KV buffer')
    runs = []
    for row in rows:
        if row < 0 or (row + 1) * row_bytes > int(buffer.nbytes):
            raise ValueError('KV row outside allocation')
        if runs and runs[-1][0] + runs[-1][1] == row:
            runs[-1][1] += 1
        else:
            runs.append([row, 1])
    return tuple(_device_hash(session, SimpleNamespace(
        ptr=int(buffer.ptr) + row * row_bytes, nbytes=count * row_bytes))
        for row, count in runs)
