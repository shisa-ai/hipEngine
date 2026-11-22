import ctypes
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from hipengine.loading import inkling2_gguf_device as device


def test_file_backed_upload_copies_bounded_chunks(tmp_path, monkeypatch):
    payload = np.arange(97, dtype=np.uint8)
    path = tmp_path / 'packed.bin'
    path.write_bytes(payload.tobytes())
    mapped = np.memmap(path, mode='r', dtype=np.uint8)
    chunks = []
    runtime = SimpleNamespace(memcpy=lambda ptr, src, size, kind: chunks.append((ptr, ctypes.string_at(src, size))))
    owner = device.Inkling2GGUFDeviceProjections.__new__(device.Inkling2GGUFDeviceProjections)
    owner._runtime = runtime
    owner._upload_staging = None
    monkeypatch.setattr(device, '_UPLOAD_CHUNK_BYTES', 32)
    monkeypatch.setattr(device, 'malloc', lambda nbytes, runtime: SimpleNamespace(ptr=1000, nbytes=nbytes))
    out = owner._upload(mapped)
    assert out.nbytes == len(payload)
    assert [(p, len(c)) for p, c in chunks] == [(1000, 32), (1032, 32), (1064, 32), (1096, 1)]
    assert b''.join(c for _, c in chunks) == payload.tobytes()
    assert owner._upload_staging.nbytes == 32


def test_partial_upload_releases_the_unpublished_device_buffer(tmp_path, monkeypatch):
    path = tmp_path / 'packed.bin'
    path.write_bytes(b'abcdefghij')
    mapped = np.memmap(path, mode='r', dtype=np.uint8)
    owner = device.Inkling2GGUFDeviceProjections.__new__(device.Inkling2GGUFDeviceProjections)
    owner._runtime = SimpleNamespace(memcpy=Mock(side_effect=[None, RuntimeError('copy error')]))
    owner._upload_staging = None
    allocation = SimpleNamespace(ptr=1000, nbytes=len(mapped))
    release = Mock()
    monkeypatch.setattr(device, '_UPLOAD_CHUNK_BYTES', 5)
    monkeypatch.setattr(device, 'malloc', lambda nbytes, runtime: allocation)
    monkeypatch.setattr(device, 'free', release)
    with pytest.raises(RuntimeError, match='copy error'):
        owner._upload(mapped)
    release.assert_called_once_with(allocation, runtime=owner._runtime)
