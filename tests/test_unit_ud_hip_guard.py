"""CPU checks for the optional UD HIP fixture; never open the real runtime."""
import os
from types import SimpleNamespace

import pytest

from tests import _ud_hip


def fake_hip(monkeypatch, *, status=0, count=1, backend="hip_gfx1151"):
    def get_count(ptr):
        ptr._obj.value = count
        return status
    monkeypatch.setattr(_ud_hip.ctypes, "CDLL", lambda name: SimpleNamespace(hipGetDeviceCount=get_count))
    arch = {"hip_gfx1100": "gfx1100", "hip_gfx1151": "gfx1151"}.get(backend, "gfx9999")
    monkeypatch.setattr(_ud_hip, "detect_hip_target_arches", lambda: (arch,))


@pytest.mark.parametrize("status,count", ((3, 0), (35, 0), (100, 0), (0, 0), (0, 2)))
def test_unavailable_or_ambiguous_device_skips(monkeypatch, status, count):
    fake_hip(monkeypatch, status=status, count=count)
    with pytest.raises(pytest.skip.Exception):
        next(_ud_hip.ud_hip_backend.__wrapped__())


@pytest.mark.parametrize("arches", ((), ("gfx1100", "gfx1151")))
def test_ambiguous_physical_architectures_skip_before_resolution(monkeypatch, arches):
    fake_hip(monkeypatch, count=1)
    monkeypatch.setattr(_ud_hip, "detect_hip_target_arches", lambda: arches)
    monkeypatch.setattr(_ud_hip, "resolve_backend", lambda **kw: pytest.fail("resolved ambiguous architecture"))
    with pytest.raises(pytest.skip.Exception, match="one detected physical architecture"):
        next(_ud_hip.ud_hip_backend.__wrapped__())


def test_missing_runtime_skips(monkeypatch):
    def missing(name):
        raise OSError("runtime not installed")
    monkeypatch.setattr(_ud_hip.ctypes, "CDLL", missing)
    with pytest.raises(pytest.skip.Exception, match="HIP runtime unavailable"):
        next(_ud_hip.ud_hip_backend.__wrapped__())


def test_unexpected_runtime_error_is_not_hidden(monkeypatch):
    fake_hip(monkeypatch, status=999)
    with pytest.raises(RuntimeError, match="status 999"):
        next(_ud_hip.ud_hip_backend.__wrapped__())


def test_unsupported_backend_skips(monkeypatch):
    fake_hip(monkeypatch, backend="cpu_reference")
    with pytest.raises(pytest.skip.Exception, match="requires gfx1100 or gfx1151"):
        next(_ud_hip.ud_hip_backend.__wrapped__())


@pytest.mark.parametrize("backend,arch", (("hip_gfx1100", "gfx1100"), ("hip_gfx1151", "gfx1151")))
def test_target_override_covers_fixture_lifetime_and_restores(monkeypatch, backend, arch):
    fake_hip(monkeypatch, backend=backend)
    monkeypatch.setenv("HIPENGINE_HIP_ARCH", "stale-target")
    monkeypatch.setenv("HIPENGINE_BACKEND", "cpu_reference")
    fixture = _ud_hip.ud_hip_backend.__wrapped__()
    assert next(fixture) == backend
    try:
        assert os.environ["HIPENGINE_HIP_ARCH"] == arch
    finally:
        fixture.close()
    assert os.environ["HIPENGINE_HIP_ARCH"] == "stale-target"
