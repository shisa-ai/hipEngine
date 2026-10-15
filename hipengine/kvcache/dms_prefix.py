"""Portable compact-prefix ownership, independent of the source allocator."""

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np

from hipengine.kvcache.dms_device import DMSDeviceExtentSnapshot, DMSDevicePayloadSnapshot

if TYPE_CHECKING:
    from hipengine.kvcache.dms import DMSCompactBackend


def _owned(array):
    result = np.array(array, copy=True, order="C")
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class DMSPrefixSnapshot:
    layout: str
    logical_tokens: int
    live_counts: np.ndarray
    extents: tuple[DMSDeviceExtentSnapshot, ...]

    @property
    def nbytes(self) -> int:
        return self.live_counts.nbytes + sum(
            array.nbytes
            for extent in self.extents
            for array in (extent.k_bits, extent.v_bits, extent.positions,
                          extent.evict, extent.k_scales, extent.v_scales)
            if array is not None
        )


def capture_prefix(backend: "DMSCompactBackend", request_id: int) -> DMSPrefixSnapshot:
    """Capture only live rows; never retain a lease, allocator, or spare capacity."""
    state = backend.state_for_request(request_id)
    store = backend._device_store
    device = None if store is None else store.snapshot(state.base_offsets, state.range_capacity)
    extents = []
    for layer in range(backend.retrofit.num_layers):
        for head in range(backend.retrofit.num_kv_heads):
            key = (layer, head)
            live = int(state.live_counts[key])
            if device is None:
                k, v = state.k_payload[key], state.v_payload[key]
                if backend.codec == "bf16":
                    k = (k.view(np.uint32) >> np.uint32(16)).astype(np.uint16)
                    v = (v.view(np.uint32) >> np.uint32(16)).astype(np.uint16)
                ks, vs = state.k_scales.get(key), state.v_scales.get(key)
                positions, evict = state.token_positions[key], state.evict_mask[key]
            else:
                item = device.extents[layer * backend.retrofit.num_kv_heads + head]
                if int(device.live_counts[key]) != live:
                    raise RuntimeError("DMS prefix host/device live counts disagree")
                k, v, ks, vs = item.k_bits, item.v_bits, item.k_scales, item.v_scales
                positions, evict = item.positions, item.evict
            extents.append(DMSDeviceExtentSnapshot(
                layer=layer, head=head, start=0, length=live,
                k_bits=_owned(k[:live]), v_bits=_owned(v[:live]),
                positions=_owned(positions[:live]),
                evict=_owned(np.asarray(evict[:live], dtype=np.uint8)),
                k_scales=None if ks is None else _owned(ks[:live]),
                v_scales=None if vs is None else _owned(vs[:live]),
            ))
    return DMSPrefixSnapshot(backend.spec.layout_fingerprint, int(state.logical_tokens),
                             _owned(state.live_counts), tuple(extents))


def restore_prefix(backend: "DMSCompactBackend", request_id: int,
                   snapshot: DMSPrefixSnapshot) -> None:
    """Restore into a fresh reservation; its ownership and capacity do not change."""
    state = backend.state_for_request(request_id)
    if snapshot.layout != backend.spec.layout_fingerprint:
        raise ValueError("DMS prefix layout or retention policy mismatch")
    if state.logical_tokens or np.any(state.live_counts):
        raise ValueError("DMS prefix restore requires an empty reservation")
    shape = state.live_counts.shape
    if snapshot.live_counts.shape != shape or len(snapshot.extents) != shape[0] * shape[1]:
        raise ValueError("DMS prefix geometry mismatch")
    if np.any(snapshot.live_counts < 0) or np.any(snapshot.live_counts > state.range_capacity):
        raise ValueError("DMS prefix exceeds reserved capacity")
    # Validate all planes before any host or device mutation.
    for index, item in enumerate(snapshot.extents):
        key = divmod(index, shape[1])
        live = int(snapshot.live_counts[key])
        if (item.layer, item.head) != key or item.length != live:
            raise ValueError("DMS prefix extent order or live count mismatch")
        dtype = np.uint16 if backend.codec == "bf16" else np.int8
        for payload in (item.k_bits, item.v_bits):
            if payload.shape != (live, backend.retrofit.head_dim) or payload.dtype != dtype:
                raise ValueError("DMS prefix payload shape or dtype mismatch")
        if item.positions.shape != (live,) or item.positions.dtype != np.int32:
            raise ValueError("DMS prefix position shape or dtype mismatch")
        if item.evict.shape != (live,) or item.evict.dtype != np.uint8:
            raise ValueError("DMS prefix eviction shape or dtype mismatch")
        for scales in (item.k_scales, item.v_scales):
            if backend.codec == "bf16":
                if scales is not None:
                    raise ValueError("BF16 DMS prefix must not contain scales")
            elif scales is None or scales.shape != (live,) or scales.dtype != np.float32:
                raise ValueError("INT8 DMS prefix scale shape or dtype mismatch")
    store = backend._device_store
    if store is not None:
        relocated = []
        for item in snapshot.extents:
            key = item.layer, item.head
            # Device journal restore requires nonempty extents. A zero-live
            # head gets a cleared spare slot, never a read of uninitialized KV.
            if item.length == 0:
                item = replace(item, length=1,
                    k_bits=np.zeros((1, backend.retrofit.head_dim), dtype=item.k_bits.dtype),
                    v_bits=np.zeros((1, backend.retrofit.head_dim), dtype=item.v_bits.dtype),
                    positions=np.full(1, -1, dtype=np.int32), evict=np.zeros(1, dtype=np.uint8),
                    k_scales=None if item.k_scales is None else np.ones(1, dtype=np.float32),
                    v_scales=None if item.v_scales is None else np.ones(1, dtype=np.float32))
            relocated.append(replace(item, start=int(state.base_offsets[key])))
        store.restore(DMSDevicePayloadSnapshot(tuple(relocated), snapshot.live_counts))
        for layer in range(shape[0]):
            store.configure_layer(layer, base=state.base_offsets[layer],
                                  capacity=state.range_capacity[layer], live=snapshot.live_counts[layer])
    for item in snapshot.extents:
        key = item.layer, item.head
        live = item.length
        state.token_positions[key][:] = -1
        state.evict_mask[key][:] = False
        state.token_positions[key][:live] = item.positions
        state.evict_mask[key][:live] = item.evict
        if store is None:
            if backend.codec == "bf16":
                state.k_payload[key] = (item.k_bits.astype(np.uint32) << np.uint32(16)).view(np.float32)
                state.v_payload[key] = (item.v_bits.astype(np.uint32) << np.uint32(16)).view(np.float32)
            else:
                state.k_payload[key] = item.k_bits.copy()
                state.v_payload[key] = item.v_bits.copy()
            if item.k_scales is not None:
                state.k_scales[key] = item.k_scales.copy()
                state.v_scales[key] = item.v_scales.copy()
    state.live_counts[...] = snapshot.live_counts
    state.logical_tokens = snapshot.logical_tokens
