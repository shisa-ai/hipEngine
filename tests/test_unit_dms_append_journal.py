"""Bounded undo is equivalent to full DMS rollback when no row can expire."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.runtime.qwen35_gguf_mtp import _DMSStoreJournal
from test_unit_dms_verifier_journal import (
    _RecordingResident,
    _admit,
    _backend,
    _pack_rows,
    _store_state,
    _assert_store_state_equal,
)


class TailStore:
    """Fake device: explicit tail writes, no host payload mirror."""

    def __init__(self, layers, heads, slots, dim):
        self.payload = np.arange(layers * heads * slots * dim, dtype=np.int32).reshape(
            layers, heads, slots, dim
        )
        self.live = np.zeros((layers, heads), dtype=np.int32)
        self.snapshots = 0
        self.restores = []

    def snapshot_append_tail(self, base_offsets, live_counts, rows):
        self.snapshots += 1
        return SimpleNamespace(
            live=live_counts.copy(),
            rows=rows,
            payload=np.stack(
                [
                    np.stack(
                        [
                            self.payload[layer, h, int(n) : int(n) + rows].copy()
                            for h, n in enumerate(counts)
                        ]
                    )
                    for layer, counts in enumerate(live_counts)
                ]
            ),
        )

    def restore_append_tail(self, snapshot, accepted_rows=0):
        self.restores.append(accepted_rows)
        for layer, counts in enumerate(snapshot.live):
            for h, n in enumerate(counts):
                self.payload[layer, h, int(n) + accepted_rows : int(n) + snapshot.rows] = (
                    snapshot.payload[layer, h, accepted_rows:]
                )
        self.live[...] = snapshot.live + accepted_rows

    def snapshot(self, *args):
        raise AssertionError("full payload snapshot on the bounded path")


def prepared(*, evict=False, tokens=3, window=4, capacity=12):
    backend = _backend()
    backend.retrofit = replace(backend.retrofit, window_size=window)
    request = SimpleNamespace(request_id=0, prompt_tokens=(), max_new_tokens=capacity - tokens)
    claims = backend.estimate(
        request, None, {"kind": "admission", "tokens": tokens, "max_new_tokens": capacity - tokens}
    )
    backend.reserve(claims)
    rng = np.random.default_rng(7)
    backend.streaming_pack(
        0,
        _pack_rows(rng, tokens),
        _pack_rows(rng, tokens),
        np.full((tokens, 1, 2), evict, dtype=bool),
    )
    state = backend.state_for_request(0)
    # The shipping device backend has no host payload maps.
    state.k_payload.clear()
    state.v_payload.clear()
    state.k_scales.clear()
    state.v_scales.clear()
    store = TailStore(1, 2, capacity + 8, 4)
    store.live[...] = state.live_counts
    backend._device_store = store
    return backend, store


def append(backend, store, position, *, eviction=False):
    state = backend.state_for_request(0)
    for layer, counts in enumerate(state.live_counts):
        for h, n in enumerate(counts):
            store.payload[layer, h, int(n)] = 9000 + position
            store.live[layer, h] += 1
    # Implement the successful no-expiry control update; do not invoke kernels.
    for layer, counts in enumerate(state.live_counts.copy()):
        for h, n in enumerate(counts):
            state.token_positions[layer, h, n] = position
            state.evict_mask[layer, h, n] = eviction
    state.live_counts += 1
    state.logical_tokens = position + 1
    backend.decode_appends += 1


@pytest.mark.parametrize("selected", [None, "close", 0, 1, 3])
def test_bounded_journal_restores_exact_selected_prefix_without_full_snapshots(selected):
    backend, store = prepared()
    target = SimpleNamespace(_dms_backend=backend, position=3)
    resident = _RecordingResident()
    journal = _DMSStoreJournal(target=target, resident=resident, max_rows=4)
    journal.bind_request(0)
    initial = _store_state(backend, 0)
    initial_payload = store.payload.copy()
    journal.capture_initial()
    states = []
    payloads = []
    for row in range(4):
        append(backend, store, 3 + row, eviction=True)
        (journal.capture_final_row if row == 3 else journal.capture_row)(row)
        states.append(_store_state(backend, 0))
        payloads.append(store.payload.copy())
    if selected == "close":
        journal.close()
    elif selected is None:
        journal.restore_initial()
    else:
        journal.restore_row(selected)
        journal.finish_transaction()
    index = None if selected in (None, "close") else selected
    _assert_store_state_equal(_store_state(backend, 0), initial if index is None else states[index])
    np.testing.assert_array_equal(
        store.payload, initial_payload if index is None else payloads[index]
    )
    assert store.snapshots == 1
    assert backend.decode_appends == (0 if index is None else index + 1)
    assert backend.state_for_request(0).logical_tokens == (3 if index is None else 4 + index)
    journal.close()


@pytest.mark.parametrize("case", ["expiry", "capacity", "new_expiry", "position", "host_payload"])
def test_bounded_admission_requires_proven_no_expiry_and_unchanged_layout(case):
    backend, store = prepared(evict=case == "expiry")
    state = backend.state_for_request(0)
    position = 3
    rows = 4
    if case == "capacity":
        rows = 20
    if case == "new_expiry":
        rows = 6
    if case == "position":
        position = 4
    if case == "host_payload":
        state.k_payload[(0, 0)] = np.zeros((3, 16), dtype=np.int8)
    operation = backend.begin_append_transaction(0, start_position=position, max_rows=rows)
    assert operation is None
    assert store.snapshots == 0


def test_no_expiry_inclusive_window_boundary_and_followup_cycle():
    backend, store = prepared(evict=True)
    # Oldest position zero remains protected at position four, but not five.
    operation = backend.begin_append_transaction(0, start_position=3, max_rows=2)
    assert operation is not None
    append(backend, store, 3)
    checkpoint = backend.checkpoint_append_transaction(operation)
    append(backend, store, 4)
    backend.rollback(checkpoint)
    assert backend.state_for_request(0).logical_tokens == 4
    follow = backend.begin_append_transaction(0, start_position=4, max_rows=1)
    assert follow is not None
    append(backend, store, 4)
    backend.rollback(follow)
    assert backend.state_for_request(0).logical_tokens == 4
    assert backend.begin_append_transaction(0, start_position=4, max_rows=2) is None


def test_bounded_checkpoint_rejects_unexpected_state_change_before_restore():
    backend, store = prepared()
    operation = backend.begin_append_transaction(0, start_position=3, max_rows=4)
    append(backend, store, 3)
    state = backend.state_for_request(0)
    state.base_offsets[0, 0] += 1
    with pytest.raises(RuntimeError, match="layout"):
        backend.checkpoint_append_transaction(operation)
    assert not store.restores


def test_full_snapshot_path_remains_available_for_host_backend():
    backend = _backend()
    _admit(backend, 0, 8)
    assert backend.begin_append_transaction(0, start_position=0, max_rows=4) is None
