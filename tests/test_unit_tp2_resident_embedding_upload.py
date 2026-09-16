"""Exact upload regression for resident embedding's unattributed buffer views."""
import ctypes
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.core import memory
from hipengine.core.device import Device
from hipengine.core.runtime import MemcpyKind


class CopyRuntime:
    device_kind = 'hip'
    def __init__(self):
        self.current = 1
        self.calls = []
    def get_device(self): return self.current
    def set_device(self, device): self.current = device
    def memcpy(self, dst, src, count, kind):
        self.calls.append((self.current, dst, src, count, kind))
        ctypes.memmove(dst, src, count)


def test_recorded_position_reset_overwrites_stale_device_bytes():
    fixture = json.loads((Path(__file__).parent / 'fixtures/tp2/resident_position_h2d_noop.json').read_text())
    source = np.array(fixture['host_expected_bytes'], dtype=np.uint8)
    target = np.array(fixture['device_observed_bytes'], dtype=np.uint8)
    runtime = CopyRuntime()
    memory.copy_host_to_device(memory.DeviceBuffer(target.ctypes.data, target.nbytes),
                               source.ctypes.data, runtime=runtime)
    np.testing.assert_array_equal(target, source)


@pytest.mark.parametrize('attributed', [False, True])
def test_embedding_upload_view_copies_exact_bytes_and_preserves_tail(attributed):
    # Tiny extracted shape of the resident token upload: int64 token rows.
    source = np.array([248045, 846, 198], dtype=np.int64)
    target = np.full(source.nbytes + 8, 0xA5, dtype=np.uint8)
    runtime = CopyRuntime()
    view = memory.DeviceBuffer(target.ctypes.data, source.nbytes,
                               Device('hip', 0) if attributed else None)
    memory.copy_host_to_device(view, source.ctypes.data, runtime=runtime)
    np.testing.assert_array_equal(target[:source.nbytes], source.view(np.uint8))
    assert (target[source.nbytes:] == 0xA5).all()
    assert len(runtime.calls) == 1
    assert runtime.calls[0][0] == (0 if attributed else 1)
    assert runtime.current == 1


@pytest.mark.parametrize('attributed', [False, True])
def test_bounded_chunks_cover_unattributed_and_owned_uploads(attributed):
    chunk = 1 << 28
    calls = []
    runtime = CopyRuntime()
    runtime.memcpy = lambda dst, src, count, kind: calls.append((runtime.current, dst, src, count, kind))
    view = memory.DeviceBuffer(4096, chunk + 13, Device('hip', 0) if attributed else None)
    memory.copy_host_to_device(view, 8192, runtime=runtime)
    device = 0 if attributed else 1
    assert calls == [(device, 4096, 8192, chunk, MemcpyKind.HOST_TO_DEVICE),
                     (device, 4096 + chunk, 8192 + chunk, 13, MemcpyKind.HOST_TO_DEVICE)]
    assert runtime.current == 1


def test_resident_device_embedding_uploads_tokens_before_launch(monkeypatch):
    from hipengine.runtime import qwen35_gguf_runner as runner
    source = np.array([248045, 846, 198], dtype=np.int64)
    target = np.full_like(source, -1)
    calls = []
    def launch(weight, token_ptr, out_ptr, **kwargs):
        # The independent host input, not the broken uploader, is the oracle.
        np.testing.assert_array_equal(target, source)
        calls.append('embedding')
    monkeypatch.setattr(runner, 'launch_gguf_embedding', launch)
    session = SimpleNamespace(
        runner=SimpleNamespace(weights=object(), hidden_size=8, vocab_size=248320),
        runtime=CopyRuntime(), host_token_embedding_enabled=False,
        _device_token_embedding_weight=lambda **kw: object())
    runner.Qwen35GGUFResidentSession._copy_token_embeddings_to_device(
        session, source, 1234, rows=3, token_ids_device_ptr=target.ctypes.data)
    assert calls == ['embedding']
