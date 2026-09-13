"""Shared ROCm / torch availability guard for tests that need the torch bridge.

``hipengine`` loads the process HIP runtime with
``ctypes.CDLL("libamdhip64.so")`` (``hipengine/core/hip.py``), which resolves to
the system ROCm install. When the optional ``torch`` extra ships its own ROCm
SDK, torch's ``rocm_sdk.initialize_process()`` then preloads a second
``libamdhip64.so.7`` from ``_rocm_sdk_core`` and fails with

    undefined symbol: hsa_amd_vmem_export_fabric_handle, version ROCR_1

because the already-loaded system ROCR lacks that symbol. Two ROCm stacks
cannot share one process, so a test module that needs torch must skip with a
diagnostic instead of raising at collection time.

``pytest.importorskip("torch")`` is not enough: the failure is an ``OSError``
from ``dlopen``, not an ``ImportError``, so it escapes the importorskip guard
and errors the whole session. These helpers catch both and, following the
repository's HIP-availability-guard rule, turn an unusable runtime into a skip.
"""

from __future__ import annotations

import ctypes

import pytest


def hip_runtime_available() -> bool:
    """True when the process HIP runtime loads; a broken stack must skip."""

    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def torch_or_skip(context: str, *, module_level: bool = False):
    """Return the ``torch`` module, or skip with a diagnostic.

    Pass ``module_level=True`` when calling during module import (collection):
    pytest requires the flag to skip a whole module rather than a test.
    """

    if not hip_runtime_available():
        pytest.skip(
            f"{context}: HIP runtime unavailable",
            allow_module_level=module_level,
        )
    try:
        import torch
    except (ImportError, OSError) as exc:
        pytest.skip(
            f"{context}: torch unavailable ({exc})",
            allow_module_level=module_level,
        )
    return torch
