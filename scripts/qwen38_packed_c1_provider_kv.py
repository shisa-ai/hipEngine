"""Provider live-prefix hashes; rejected suffix bytes are intentionally excluded."""
from scripts.qwen38_packed_c1_kv import resident_rows, hash_rows


def snapshot_provider_kv(executor, checkpoint):
    from hipengine.core import DType
    slot, position = int(checkpoint.slot), int(checkpoint.position)
    if position < 0 or executor._request_slots.get(checkpoint.request_id) != slot:
        raise ValueError('provider KV checkpoint ownership is invalid')
    session = executor._batch_sessions[slot]
    if session.kv_storage_dtype != DType.BF16:
        raise ValueError('provider KV diagnostic requires BF16 storage')
    session.runtime.device_synchronize()
    rows = resident_rows(session, 0, position)
    cfg = session.runner.weights.config
    stride = int(cfg.head_count_kv) * int(cfg.key_length) * 2
    buffers = {}
    for layer, (key, value) in enumerate(zip(session.scratch.full_key_caches,
                                            session.scratch.full_value_caches, strict=True)):
        if key is None and value is None:
            continue
        if key is None or value is None:
            raise ValueError('provider KV plane is missing')
        for name, buffer in (('key', key), ('value', value)):
            buffers[f'{name}:{layer}'] = dict(ptr=int(buffer.ptr), nbytes=int(buffer.nbytes),
                hash=hash_rows(session, buffer, rows, stride))
    if not buffers:
        raise ValueError('no provider KV planes checked')
    return dict(request_id=int(checkpoint.request_id), slot=slot, position=position,
                physical_rows=rows, row_bytes=stride, buffers=buffers)
