"""Failure containment for the external DMS sidecar device owner."""
from types import SimpleNamespace

import numpy as np
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


@pytest.mark.parametrize("failure", ["allocation", "upload"])
def test_projector_constructor_preserves_error_and_drains_rollback(monkeypatch, failure):
    from hipengine.core import hip
    from hipengine.kernels.hip_gfx1100 import attention

    monkeypatch.setattr(hip, "get_hip_runtime", lambda: object())
    monkeypatch.setattr(attention, "build_dms_compact", lambda **kw: object())
    monkeypatch.setattr(module, "is_registered", lambda key: True)
    monkeypatch.setattr(module, "resolve", lambda **kw: object())
    acquired, released = [], []

    def allocate(size, **kw):
        if failure == "allocation" and acquired:
            raise ValueError("allocation root cause")
        buffer = SimpleNamespace(ptr=len(acquired) + 1, nbytes=size)
        acquired.append(buffer)
        return buffer

    def upload(*args, **kw):
        raise ValueError("upload root cause")

    def release(buffer, **kw):
        released.append(buffer.ptr)
        raise RuntimeError(f"cleanup {buffer.ptr}")

    monkeypatch.setattr(module, "malloc", allocate)
    monkeypatch.setattr(module, "copy_host_array_to_device", upload)
    monkeypatch.setattr(module, "free", release)
    source = SimpleNamespace(
        config=SimpleNamespace(schema_version=2, decision_source="external_linear_sidecar_v1",
                               num_layers=1, num_kv_heads=1, hidden_size=2,
                               alpha_scale=1.0, alpha_offset=0.0),
        weight=np.zeros((1, 1, 2), np.float32), bias=np.zeros((1, 1), np.float32),
    )
    with pytest.raises(ValueError, match=f"{failure} root cause") as caught:
        module.DMSExternalLinearDeviceProjector(source, backend="hip_gfx1151")
    assert released == [buffer.ptr for buffer in reversed(acquired)]
    assert any("cleanup" in note for note in caught.value.__notes__)
