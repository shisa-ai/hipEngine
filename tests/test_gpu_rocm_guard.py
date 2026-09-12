"""CPU tests for the shared ROCm/torch availability guard.

The guard exists because ``pytest.importorskip("torch")`` only catches
``ImportError``, while the two-ROCm-stack failure in this environment is an
``OSError`` from ``dlopen`` that errored the whole collection. These tests pin
the skip-not-error behaviour without needing a broken runtime.
"""

from __future__ import annotations

import ctypes

import pytest

from tests import _rocm_guard


def test_hip_runtime_available_false_when_dlopen_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(name, *args, **kwargs):
        raise OSError(f"cannot load {name}")

    monkeypatch.setattr(ctypes, "CDLL", boom)
    assert _rocm_guard.hip_runtime_available() is False


def test_torch_or_skip_skips_when_hip_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_rocm_guard, "hip_runtime_available", lambda: False)
    with pytest.raises(pytest.skip.Exception, match="HIP runtime unavailable"):
        _rocm_guard.torch_or_skip("tests.example", module_level=True)


def test_torch_or_skip_skips_on_dlopen_oserror(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The exact regression: an OSError from the torch import must skip."""

    monkeypatch.setattr(_rocm_guard, "hip_runtime_available", lambda: True)

    def boom(name, *args, **kwargs):
        raise OSError("undefined symbol: hsa_amd_vmem_export_fabric_handle")

    monkeypatch.setattr(ctypes, "CDLL", boom)
    # ``import torch`` inside the helper will succeed (torch is importable
    # standalone here), so force the failure path directly.
    real_import = __import__

    def failing_import(name, *args, **kwargs):
        if name == "torch":
            raise OSError("undefined symbol: hsa_amd_vmem_export_fabric_handle")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", failing_import)
    with pytest.raises(pytest.skip.Exception, match="torch unavailable"):
        _rocm_guard.torch_or_skip("tests.example", module_level=True)


def test_torch_or_skip_returns_torch_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_rocm_guard, "hip_runtime_available", lambda: True)
    module = _rocm_guard.torch_or_skip("tests.example")
    assert module.__name__ == "torch"
