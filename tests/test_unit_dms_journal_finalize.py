"""Undo survives row selection until the enclosing transaction is finalized."""

from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.runtime.qwen35_gguf_mtp import _DMSStoreJournal
from test_unit_dms_append_journal import prepared, append
from test_unit_dms_verifier_journal import (
    _RecordingResident,
    _store_state,
    _assert_store_state_equal,
    _prepared_backend,
    _run_cycle,
)


class MutableResident(_RecordingResident):
    def __init__(self):
        super().__init__()
        self.value = 0
        self.fail_restore = False

    def capture_initial(self, **kwargs):
        self.initial = self.value
        super().capture_initial(**kwargs)

    def restore_initial(self, **kwargs):
        self.value = self.initial
        super().restore_initial(**kwargs)

    def restore_row(self, row, **kwargs):
        if self.fail_restore:
            raise RuntimeError("resident restore failure")
        self.value = row + 1
        super().restore_row(row, **kwargs)


def cycle():
    backend, store = prepared()
    resident = MutableResident()
    journal = _DMSStoreJournal(
        target=SimpleNamespace(_dms_backend=backend, position=3), resident=resident, max_rows=4
    )
    journal.bind_request(0)
    initial = _store_state(backend, 0)
    payload = store.payload.copy()
    journal.capture_initial()
    for row in range(4):
        append(backend, store, 3 + row)
        resident.value = row + 1
        (journal.capture_final_row if row == 3 else journal.capture_row)(row)
    return backend, store, resident, journal, initial, payload


@pytest.mark.parametrize("selected", [0, 2, 3])
@pytest.mark.parametrize("failure", ["resident", "enclosing_commit"])
def test_selected_prefix_can_still_fully_roll_back_until_finish(selected, failure):
    backend, store, resident, journal, initial, payload = cycle()
    if failure == "resident":
        resident.fail_restore = True
        with pytest.raises(RuntimeError, match="resident restore"):
            journal.restore_row(selected)
    else:
        journal.restore_row(selected)
        # Position publication, synchronization or scheduler commit can now fail.
    assert journal._initial is not None
    journal.restore_initial()
    _assert_store_state_equal(_store_state(backend, 0), initial)
    np.testing.assert_array_equal(store.payload, payload)
    assert resident.value == 0
    journal.close()


def test_close_restores_resident_and_keeps_failed_undo_retryable(monkeypatch):
    backend, store, resident, journal, initial, payload = cycle()
    restore = store.restore_append_tail
    calls = []

    def failed(snapshot, accepted_rows=0):
        calls.append(1)
        store.payload[0, 0, 3] = 99999  # A copy may fail after a partial device write.
        raise RuntimeError("copy failed")

    monkeypatch.setattr(store, "restore_append_tail", failed)
    with pytest.raises(RuntimeError, match="copy failed"):
        journal.close()
    assert not journal.closed and not resident.closed
    assert journal._initial is not None
    monkeypatch.setattr(store, "restore_append_tail", restore)
    journal.close()
    assert journal.closed and resident.closed and resident.value == 0
    _assert_store_state_equal(_store_state(backend, 0), initial)
    np.testing.assert_array_equal(store.payload, payload)
    journal.close()


@pytest.mark.parametrize("selected", [0, 2])
def test_full_snapshot_selection_keeps_initial_undo_for_failed_enclosing_commit(selected):
    backend = _prepared_backend()
    resident = MutableResident()
    journal = _DMSStoreJournal(
        target=SimpleNamespace(_dms_backend=backend), resident=resident, max_rows=4
    )
    journal.bind_request(0)
    initial = _store_state(backend, 0)
    journal.capture_initial()
    _run_cycle(
        backend, journal, request_id=0, rows=3, start_position=9, rng=np.random.default_rng(17)
    )
    journal.restore_row(selected)
    # A later failure in verifier.commit or the scheduler must restore all KV.
    journal.restore_initial()
    _assert_store_state_equal(_store_state(backend, 0), initial)
    journal.close()


def test_verifier_finish_retires_dms_undo_only_after_success():
    from hipengine.runtime.qwen35_gguf_mtp import Qwen35GGUFTransactionalVerifier

    backend, store, resident, journal, initial, payload = cycle()
    journal.restore_row(0)
    verifier = object.__new__(Qwen35GGUFTransactionalVerifier)
    verifier.journal = journal
    prepared = object()
    verifier._prepared = prepared
    verifier.finish(prepared)
    assert verifier._prepared is None and journal._initial is None
    selected = _store_state(backend, 0)
    journal.close()
    _assert_store_state_equal(_store_state(backend, 0), selected)


@pytest.mark.parametrize("selected", [0, 3])
def test_finish_keeps_selected_prefix_and_next_cycle_can_rollback(selected):
    backend, store, resident, journal, initial, payload = cycle()
    journal.restore_row(selected)
    selected_state = _store_state(backend, 0)
    selected_payload = store.payload.copy()
    journal.finish_transaction()
    assert journal._initial is None
    journal.target.position = 4 + selected
    journal.capture_initial()
    append(backend, store, 4 + selected)
    journal.capture_row(0)
    journal.restore_initial()
    _assert_store_state_equal(_store_state(backend, 0), selected_state)
    np.testing.assert_array_equal(store.payload, selected_payload)
    journal.close()
