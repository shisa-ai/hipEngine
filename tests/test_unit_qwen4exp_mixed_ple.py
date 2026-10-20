"""Mixed PLE history transactions over CPU-backed fake device memory."""
import ctypes
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.core.memory import DeviceBuffer
from hipengine.runtime import qwen4_exp_batch_ple as batch
from hipengine.runtime import qwen4_exp_mixed_ple as mixed
from hipengine.runtime.qwen4_exp_mixed_layout import Qwen4ExpMixedRequest, compile_mixed_layout


class Runtime:
    def __init__(self):
        self.allocations = {}
        self.freed = []
        self.copies = []
        self.fail_free = None

    def malloc(self, size):
        buffer = ctypes.create_string_buffer(max(1, size))
        ptr = ctypes.addressof(buffer)
        self.allocations[ptr] = buffer
        return ptr

    def free(self, ptr):
        if ptr == self.fail_free:
            self.fail_free = None
            raise RuntimeError('injected free failure')
        self.allocations.pop(ptr)
        self.freed.append(ptr)

    def memcpy(self, dst, src, size, kind):
        self.copies.append((dst, src, size))
        ctypes.memmove(dst, src, size)

    def memset(self, dst, value, size):
        ctypes.memset(dst, value, size)


def view(buffer, dtype=np.float32):
    return np.frombuffer((ctypes.c_ubyte * buffer.nbytes).from_address(buffer.ptr), dtype=dtype)


def fixture(monkeypatch, *, permuted=False, requests=None):
    runtime = Runtime()
    if requests is None:
        requests = [Qwen4ExpMixedRequest(1, 4, 11, 7, 2, (3, 4)),
                    Qwen4ExpMixedRequest(2, 1, 12, 13, 5, ()),
                    Qwen4ExpMixedRequest(3, 3, 13, 4, 6, (7,))]
    live = batch.Qwen4ExpBatchPleState.allocate(
        state_slots=max(5, max(request.slot for request in requests) + 1),
        branches=1, hidden=2, conv_kernel=2, dilation=1, runtime=runtime,
    )
    view(live.history)[:] = np.arange(live.state_slots * 2, dtype=np.float32) + 10
    owners = {request.slot: request.owner_token for request in requests}
    layout = compile_mixed_layout(requests, physical_row_order=(3, 0, 4, 1, 5, 2) if permuted else None)
    calls = []

    def conv(values, weights, history, output, indices, *, rows, **kwargs):
        calls.append(rows)
        indices = np.ctypeslib.as_array((ctypes.c_int64 * rows).from_address(indices))
        src = np.ctypeslib.as_array((ctypes.c_float * (rows * 2)).from_address(values)).reshape(rows, 2)
        dst = np.ctypeslib.as_array((ctypes.c_float * (rows * 2)).from_address(output)).reshape(rows, 2)
        hist = np.ctypeslib.as_array((ctypes.c_float * (layout.total_rows * 2)).from_address(history)).reshape(-1, 2)
        for row, slot in enumerate(indices):
            dst[row] = hist[slot] + src[row]
            hist[slot] = src[row]

    monkeypatch.setattr(batch, '_resolve_indexed_conv', lambda weights: conv)

    def prepare(plan, residual, embedding, weights, *, scratch, **kwargs):
        values = np.ctypeslib.as_array((ctypes.c_float * (plan.rows * 2)).from_address(embedding))
        view(scratch.normalized)[:plan.rows * 2] = values

    def finish(plan, residual, *, scratch):
        # Keep FP32 conv values for a hand-checkable oracle without BF16 kernels.
        view(scratch.output, np.uint16)[:plan.rows * 2] = view(scratch.conv)[:plan.rows * 2].astype(np.uint16)
        return scratch.output

    monkeypatch.setattr(mixed, 'run_qwen4_exp_batch_ple_prepare', prepare)
    monkeypatch.setattr(mixed, 'run_qwen4_exp_batch_ple_finish', finish)
    adapter = mixed.Qwen4ExpMixedPleAdapter(layout, live, owner_of=owners.get)
    residual = DeviceBuffer(runtime.malloc(layout.total_rows * 4), layout.total_rows * 4)
    embedding = DeviceBuffer(runtime.malloc(layout.total_rows * 8), layout.total_rows * 8)
    for row in layout.rows:
        view(embedding).reshape(-1, 2)[row.physical_row] = (row.token, row.token + 1)
    return SimpleNamespace(runtime=runtime, live=live, layout=layout, owners=owners,
                           adapter=adapter, residual=residual, embedding=embedding, calls=calls)


def run(f):
    return f.adapter.verify(f.residual, f.embedding, {'ple_key': 'key', 'ple_value': 'value'},
                            norm_key_ptr=8, norm_query_ptr=16, norm_conv_ptr=24, conv_weight_ptr=32)


def close(f):
    f.adapter.close()
    f.live.close()
    f.runtime.free(f.residual.ptr)
    f.runtime.free(f.embedding.ptr)
    assert not f.runtime.allocations


@pytest.mark.parametrize('permuted', [False, True])
def test_depth_waves_preserve_live_history_and_physical_outputs(monkeypatch, permuted):
    f = fixture(monkeypatch, permuted=permuted)
    before = view(f.live.history).copy()
    try:
        output = run(f)
        assert f.calls == [3, 2, 1]
        np.testing.assert_array_equal(view(f.live.history), before)
        actual = view(output, np.uint16).reshape(-1, 2)
        for row in f.layout.rows:
            parent = (before.reshape(-1, 2)[row.slot] if row.depth == 0 else
                      np.array([row.token - 1, row.token]))
            np.testing.assert_array_equal(actual[row.physical_row], parent + [row.token, row.token + 1])
        # Repeated verification starts from live roots again.
        expected = actual.copy()
        np.testing.assert_array_equal(view(run(f), np.uint16).reshape(-1, 2), expected)
    finally:
        close(f)


@pytest.mark.parametrize('accepted', [0, 1, 2])
def test_commit_selects_one_prefix_snapshot_and_advances_ar_once(monkeypatch, accepted):
    f = fixture(monkeypatch, permuted=True)
    before = view(f.live.history).reshape(-1, 2).copy()
    try:
        run(f)
        plan = f.adapter.commit((accepted, 0, 1))
        assert plan.commit_positions == (7 + accepted, 13, 5)
        after = view(f.live.history).reshape(-1, 2)
        np.testing.assert_array_equal(after[4], [2 + accepted, 3 + accepted])
        np.testing.assert_array_equal(after[1], [5, 6])
        np.testing.assert_array_equal(after[3], [7, 8])
        np.testing.assert_array_equal(after[[0, 2]], before[[0, 2]])
        with pytest.raises(RuntimeError, match='uncommitted'):
            f.adapter.commit((accepted, 0, 1))
    finally:
        close(f)


def test_stale_owner_and_invalid_commit_never_write_live_state(monkeypatch):
    f = fixture(monkeypatch)
    try:
        run(f)
        count = len(f.runtime.copies)
        with pytest.raises(ValueError):
            f.adapter.commit((3, 0, 1))
        assert len(f.runtime.copies) == count
        f.owners[4] = 99
        with pytest.raises(ValueError, match='stale'):
            f.adapter.commit((0, 0, 1))
        with pytest.raises(ValueError, match='stale'):
            run(f)
        assert len(f.runtime.copies) == count
    finally:
        close(f)


def test_short_input_fails_before_copy_or_publication(monkeypatch):
    f = fixture(monkeypatch)
    try:
        count = len(f.runtime.copies)
        f.embedding = DeviceBuffer(f.embedding.ptr, 1)
        with pytest.raises(ValueError, match='too small'):
            run(f)
        assert len(f.runtime.copies) == count
    finally:
        close(f)


def test_total_verification_rows_above_indexed_dispatch_width(monkeypatch):
    requests = [Qwen4ExpMixedRequest(i + 1, i, i + 11, 7, 2, (3, 4, 5, 6)) for i in range(16)]
    f = fixture(monkeypatch, requests=requests)
    before = view(f.live.history).copy()
    try:
        assert f.layout.total_rows == 80
        assert f.layout.physical_request_width == 16
        run(f)
        assert f.calls == [16] * 5
        np.testing.assert_array_equal(view(f.live.history), before)
        f.adapter.commit((4,) * 16)
        np.testing.assert_array_equal(view(f.live.history).reshape(-1, 2), [[6, 7]] * 16)
    finally:
        close(f)


def test_close_retries_only_unfreed_scratch(monkeypatch):
    f = fixture(monkeypatch)
    f.runtime.fail_free = f.adapter.wave.conv.ptr
    with pytest.raises(RuntimeError, match='injected'):
        f.adapter.close()
    freed = tuple(f.runtime.freed)
    with pytest.raises(RuntimeError, match='closing'):
        run(f)
    close(f)
    assert len(f.runtime.freed) == len(set(f.runtime.freed))
    assert tuple(f.runtime.freed[:len(freed)]) == freed


def test_failed_convolution_cannot_commit_and_live_history_is_unchanged(monkeypatch):
    f = fixture(monkeypatch)
    original = mixed.run_qwen4_exp_batch_ple_convolution
    before = view(f.live.history).copy()
    try:
        def fail(*args, **kwargs):
            original(*args, **kwargs)
            raise RuntimeError('injected convolution failure')
        monkeypatch.setattr(mixed, 'run_qwen4_exp_batch_ple_convolution', fail)
        with pytest.raises(RuntimeError, match='injected'):
            run(f)
        np.testing.assert_array_equal(view(f.live.history), before)
        with pytest.raises(RuntimeError, match='uncommitted'):
            f.adapter.commit((0, 0, 0))
        monkeypatch.setattr(mixed, 'run_qwen4_exp_batch_ple_convolution', original)
        run(f)
        f.adapter.commit((1, 0, 1))
    finally:
        close(f)
