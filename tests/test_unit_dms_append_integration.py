"""Exercise backend, verifier journal and real tail helpers in fake device memory."""

from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.runtime.qwen35_gguf_mtp import _DMSStoreJournal
from test_unit_dms_append_journal import prepared
from test_unit_dms_verifier_journal import (
    _RecordingResident,
    _store_state,
    _assert_store_state_equal,
)
from test_unit_dms_device_layer_staging import _store, _fill_planes


@pytest.mark.parametrize("codec", ["bf16", "int8_per_token_head"])
@pytest.mark.parametrize("selected", [None, 0, 3, "incomplete"])
def test_integrated_bounded_undo_preserves_all_planes_and_neighbor_slots(
    monkeypatch, codec, selected
):
    backend, _ = prepared()
    state = backend.state_for_request(0)
    state.live_counts[0, 1] = 2
    state.token_positions[0, 1, 2] = -1
    store, fake = _store(monkeypatch, codec=codec, layers=1, heads=2, dim=16, slots=128)
    store._ensure_layer(0)
    _fill_planes(store, fake, 0, seed=11)
    backend._device_store = store
    live = fake.backing[store._live_meta[0].ptr].view(np.int32)
    live[:] = state.live_counts[0]
    resident = _RecordingResident()
    journal = _DMSStoreJournal(
        target=SimpleNamespace(_dms_backend=backend, position=3), resident=resident, max_rows=4
    )
    journal.bind_request(0)
    initial = _store_state(backend, 0)
    initial_device = {ptr: data.copy() for ptr, data in fake.backing.items()}
    allocations = fake.malloc_count
    journal.capture_initial()
    expected_state = initial
    expected_device = initial_device
    for row in range(4):
        counts = state.live_counts.copy()
        for head, count in enumerate(counts[0]):
            slot = int(state.base_offsets[0, head] + count)
            for buffers, width in (
                (store._k_slot, 16 * store._payload_dtype.itemsize),
                (store._v_slot, 16 * store._payload_dtype.itemsize),
                (store._positions, 4),
                (store._slot_evict, 1),
                (store._k_scales, 4),
                (store._v_scales, 4),
            ):
                if buffers:
                    fake.backing[buffers[0].ptr][slot * width : (slot + 1) * width] = (
                        row + 20 + head
                    )
            live[head] += 1
            if selected == "incomplete":
                # Exception after device writes to one head, before host finalize.
                break
        if selected == "incomplete":
            break
        for head, count in enumerate(counts[0]):
            state.token_positions[0, head, count] = 3 + row
            state.evict_mask[0, head, count] = False
        state.live_counts += 1
        state.logical_tokens = 4 + row
        backend.decode_appends += 1
        (journal.capture_final_row if row == 3 else journal.capture_row)(row)
        if selected == row:
            expected_state = _store_state(backend, 0)
            expected_device = {ptr: data.copy() for ptr, data in fake.backing.items()}
    if selected in (None, "incomplete"):
        journal.close()
    else:
        journal.restore_row(selected)
        journal.finish_transaction()
        journal.close()
    _assert_store_state_equal(_store_state(backend, 0), expected_state)
    for ptr, expected in expected_device.items():
        np.testing.assert_array_equal(fake.backing[ptr], expected)
    assert fake.malloc_count == allocations
