"""Shared QSA metadata uploads identical positions only once per chunk."""

from types import SimpleNamespace

import numpy as np
import pytest

import hipengine.runtime.qwen4_exp_runner as module


@pytest.fixture
def metadata(monkeypatch):
    owner = object.__new__(module.Qwen4ExpQSAPrefillMetadata)
    owner.closed = False
    owner.rows = 8
    owner.max_positions = 8192
    owner.runtime = object()
    owner.positions = SimpleNamespace(ptr=11)
    owner.context_counts = SimpleNamespace(ptr=22)
    owner.positions_host = np.zeros(8, dtype=np.int64)
    owner.context_counts_host = np.zeros(8, dtype=np.int64)
    owner.selected_positions_host = np.full((8, 16), -1, dtype=np.int64)
    owner.selected_counts_host = np.zeros(8, dtype=np.int32)
    calls = []

    def copy(buf, host, nbytes, **kwargs):
        calls.append((buf.ptr, nbytes))

    monkeypatch.setattr(module, "copy_host_array_to_device", copy)
    return owner, calls, copy


def test_same_chunk_reuses_upload_but_resets_layer_selection(metadata):
    owner, calls, _ = metadata
    owner.set_contiguous(1025, 4)
    owner.selected_positions_host[:4] = 42
    owner.selected_counts_host[:4] = 16
    owner.set_contiguous(1025, 4)
    assert calls == [(11, 32), (22, 32)]
    np.testing.assert_array_equal(owner.positions_host[:4], [1025, 1026, 1027, 1028])
    np.testing.assert_array_equal(owner.context_counts_host[:4], [1026, 1027, 1028, 1029])
    assert np.all(owner.selected_positions_host[:4] == -1)
    assert np.all(owner.selected_counts_host[:4] == 0)
    owner.set_contiguous(1025, 3)
    owner.set_contiguous(1026, 3)
    owner.set_contiguous(17, 3)
    assert calls[2:] == [(11, 24), (22, 24)] * 3


@pytest.mark.parametrize("failed_ptr", [11, 22])
def test_failed_upload_invalidates_previous_chunk(metadata, monkeypatch, failed_ptr):
    owner, calls, copy = metadata
    owner.set_contiguous(17, 4)

    def fail(buf, host, nbytes, **kwargs):
        copy(buf, host, nbytes, **kwargs)
        if buf.ptr == failed_ptr:
            raise RuntimeError("copy failed")

    monkeypatch.setattr(module, "copy_host_array_to_device", fail)
    with pytest.raises(RuntimeError, match="copy failed"):
        owner.set_contiguous(1025, 4)
    assert owner._uploaded_contiguous is None
    monkeypatch.setattr(module, "copy_host_array_to_device", copy)
    calls.clear()
    owner.set_contiguous(17, 4)
    assert calls == [(11, 32), (22, 32)]


def test_reused_chunk_still_checks_lifetime_and_bounds(metadata):
    owner, calls, _ = metadata
    owner.set_contiguous(17, 4)
    for start, rows in ((8191, 4), (-1, 4), (17, 0), (17, 9)):
        with pytest.raises(ValueError):
            owner.set_contiguous(start, rows)
    owner.closed = True
    with pytest.raises(RuntimeError, match="closed"):
        owner.set_contiguous(17, 4)
    assert len(calls) == 2
