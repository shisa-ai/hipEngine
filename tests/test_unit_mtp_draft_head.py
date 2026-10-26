"""CPU ownership and launch-shape contracts for the compact draft head."""
from dataclasses import dataclass
from types import SimpleNamespace
import ctypes

import numpy as np
import pytest

from hipengine.core.memory import DeviceBuffer
from hipengine.core.tensor import Tensor
from hipengine.core.dtype import DType
from hipengine.runtime import mtp_draft_head as module


class Runtime:
    def __init__(self):
        self.buffers = {}
        self.copies = []

    def malloc(self, size):
        data = ctypes.create_string_buffer(size)
        ptr = ctypes.addressof(data)
        self.buffers[ptr] = data
        return ptr

    def free(self, ptr):
        del self.buffers[ptr]

    def memset(self, ptr, value, size):
        ctypes.memset(ptr, value, size)

    def memcpy(self, dst, src, size, kind):
        self.copies.append((dst, src, size, kind))
        ctypes.memmove(dst, src, size)

    def mem_get_info(self):
        return 1 << 30, 1 << 30


@dataclass
class Allocation:
    buffer: DeviceBuffer
    tensor: Tensor
    owns_buffer: bool = True


@dataclass
class Weight:
    spec: object
    allocations: dict
    backend: str = 'hip_gfx1151'

    def allocation(self, key):
        return self.allocations[key]


@pytest.fixture
def owner(monkeypatch):
    runtime = Runtime()
    data = np.arange(64 * 4, dtype=np.uint8)
    ptr = runtime.malloc(data.nbytes)
    ctypes.memmove(ptr, data.ctypes.data, data.nbytes)
    raw = Allocation(DeviceBuffer(ptr, data.nbytes), Tensor.from_handle(ptr, (data.nbytes,), DType.INT8, 'hip'))
    weight = Weight(SimpleNamespace(layout='raw_gguf'), {'raw': raw})
    calls = []
    monkeypatch.setattr(module, 'launch_gguf_linear', lambda *a, **kw: calls.append((a, kw)))
    head = module.MTPSubsetHead(weight, vocab_size=64, hidden_size=4, special_ids=(63,),
                                runtime=runtime, prefix_size=16, extra_capacity=8)
    yield head, runtime, calls, raw
    head.close()
    assert list(runtime.buffers) == [ptr]


def test_committed_rows_are_copied_in_table_order_and_only_once(owner):
    head, rt, _, raw = owner
    head.prepare([0, 29, 1], anchor=1)
    assert head.policy.extra_ids == (63, 29)
    rows = np.frombuffer(ctypes.string_at(head.extra_buffer.ptr, 8 * 4), dtype=np.uint8).reshape(8, 4)
    source = np.arange(256, dtype=np.uint8).reshape(64, 4)
    np.testing.assert_array_equal(rows[:2], source[[63, 29]])
    before = len(rt.copies)
    head.prepare([0, 29, 1], anchor=1)
    assert len(rt.copies) == before
    head.project(101, 202)
    assert head.diagnostics()['subset_calls'] == 1


def test_compact_projection_pads_rows_but_argmax_excludes_padding(owner):
    head, _, calls, _ = owner
    head.prepare([0, 29, 1], anchor=1)
    assert head.project(101, 202) == 18
    assert [c[0][5] for c in calls] == [16, 8]
    assert calls[1][0][2] == 202 + 16 * 4
    assert calls[1][1]['threads'] == 128


def test_recent_misses_and_logit_capture_keep_the_full_head(owner):
    head, _, calls, _ = owner
    head.prepare([20, 21], anchor=21)
    assert head.project(101, 202) == 64
    assert len(calls) == 1 and calls[0][0][5] == 64
    assert head.diagnostics()['full_reasons'] == {'recent_vocabulary_misses': 1}
    head.prepare([20, 21] + [0] * 40, anchor=0)
    assert head.project(101, 202, capture_logits=True) == 64
    assert head.diagnostics()['full_reasons']['logits_requested'] == 1


def test_reset_removes_prior_request_rows_without_reallocating(owner):
    head, rt, _, _ = owner
    head.prepare([29, 0], anchor=0)
    ptrs = tuple(rt.buffers)
    head.project(101, 202)
    head.reset()
    assert tuple(rt.buffers) == ptrs
    assert head.policy.extra_ids == (63,)
    assert head.policy.scanned == 0
    assert head.diagnostics()['subset_calls'] == 0
    head.prepare([30, 0], anchor=0)
    assert head.policy.extra_ids == (63, 30)


def test_unprepared_head_stays_full(owner):
    head, _, _, _ = owner
    assert head.project(101, 202) == 64
    assert head.diagnostics()['full_reasons'] == {'request_not_prepared': 1}


@pytest.mark.parametrize('cause', ['layout', 'specials', 'memory'])
def test_unavailable_subset_preserves_full_head_with_named_reason(owner, cause, monkeypatch):
    _, rt, _, raw = owner
    spec = SimpleNamespace(layout='raw_gguf' if cause != 'layout' else 'packed_tiles')
    specials = (61, 62, 63) if cause == 'specials' else (63,)
    if cause == 'memory':
        monkeypatch.setattr(rt, 'mem_get_info', lambda: (0, 0))
    before = tuple(rt.buffers)
    head = module.MTPSubsetHead(Weight(spec, {'raw': raw}), vocab_size=64, hidden_size=4,
        special_ids=specials, runtime=rt, prefix_size=16, extra_capacity=2)
    try:
        head.prepare([1], anchor=1)
        assert head.project(101, 202) == 64
        reason = {'layout': 'head_layout_not_row_addressable',
                  'specials': 'special_table_exceeds_capacity',
                  'memory': 'insufficient_memory_for_extra_rows'}[cause]
        assert head.diagnostics()['full_reasons'] == {reason: 1}
        assert tuple(rt.buffers) == before
    finally:
        head.close()


def test_small_vocab_needs_no_extra_allocation(owner):
    _, rt, _, raw = owner
    before = tuple(rt.buffers)
    weight = Weight(SimpleNamespace(layout='raw_gguf'), {'raw': raw})
    other = module.MTPSubsetHead(weight, vocab_size=64, hidden_size=4, special_ids=(), runtime=rt)
    other.prepare([1], anchor=1)
    assert other.project(101, 202) == 64
    assert tuple(rt.buffers) == before
    other.close()


def test_padding_may_not_overrun_full_logits_buffer(owner):
    _, rt, _, raw = owner
    weight = Weight(SimpleNamespace(layout='raw_gguf'), {'raw': raw})
    head = module.MTPSubsetHead(weight, vocab_size=64, hidden_size=4, special_ids=(63,),
                                runtime=rt, prefix_size=60, extra_capacity=8)
    try:
        head.prepare([1], anchor=1)
        assert head.project(101, 202) == 64
        assert head.diagnostics()['full_reasons'] == {'subset_not_smaller': 1}
    finally:
        head.close()


def test_extra_id_upload_owns_the_named_row_array(owner, monkeypatch):
    """Row ids upload through the owning helper at the row's byte offset."""

    _, rt, _, raw = owner
    uploads = []
    legacy = []

    def upload(buffer, array, nbytes=None, *, runtime=None):
        uploads.append((buffer, array, nbytes, runtime))

    def old_upload(*args, **kwargs):
        legacy.append((args, kwargs))

    monkeypatch.setattr(module, 'copy_host_array_to_device', upload, raising=False)
    monkeypatch.setattr(module, 'copy_host_to_device', old_upload, raising=False)
    weight = Weight(SimpleNamespace(layout='raw_gguf'), {'raw': raw})
    head = module.MTPSubsetHead(weight, vocab_size=64, hidden_size=4, special_ids=(63,),
                                runtime=rt, prefix_size=16, extra_capacity=8)
    try:
        # The constructor publishes the initial special row at offset zero.
        assert legacy == []
        assert [(call[0].ptr, call[2], call[3] is rt) for call in uploads] == [
            (head.extra_ids_buffer.ptr, 8, True),
        ]
        head.prepare([0, 29, 1], anchor=1)
        # The appended row lands at its own 8-byte slot, with its own count.
        assert [(call[0].ptr, call[2], call[3] is rt) for call in uploads] == [
            (head.extra_ids_buffer.ptr, 8, True),
            (head.extra_ids_buffer.ptr + 8, 8, True),
        ]
        assert [call[0].nbytes for call in uploads] == [8, 8]
        assert [call[1].dtype for call in uploads] == [np.int64, np.int64]
        assert [call[1].tolist() for call in uploads] == [[63], [29]]
        assert head.policy.extra_ids == (63, 29)
    finally:
        head.close()
