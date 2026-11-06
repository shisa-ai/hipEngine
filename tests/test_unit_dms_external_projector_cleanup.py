"""Failure containment for the external DMS sidecar device owner."""
from types import SimpleNamespace

import pytest

from hipengine.kvcache import dms_device as module


def test_projector_close_drains_buffers_once_after_release_failure(monkeypatch):
    owner = object.__new__(module.DMSExternalLinearDeviceProjector)
    owner._closed = False
    owner._runtime = object()
    owner._buffers = [SimpleNamespace(ptr=1, nbytes=4), SimpleNamespace(ptr=2, nbytes=4)]
    calls = []

    def release(buffer, *, runtime):
        assert runtime is owner._runtime
        calls.append(buffer.ptr)
        raise RuntimeError(f"release {buffer.ptr}")

    monkeypatch.setattr(module, "free", release)
    with pytest.raises(RuntimeError, match="release 2") as caught:
        owner.close()
    assert calls == [2, 1]
    assert any("release 1" in note for note in caught.value.__notes__)
    assert owner.resident_bytes == 0
    owner.close()
    assert calls == [2, 1]
