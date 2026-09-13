"""Cross-caller parity between runtime and audit GGUF capability resolution.

The quant-route audit resolves backend capabilities by reading backend package
sources (no import, no HIP); the runtime resolves them through
``backend_package_capability`` (which imports the package). Both feed the same
pure ``resolve_gguf_dense_flags`` API. These tests prove the two readers produce
identical dense flags and FP16-recurrent-state defaults for the real backend
packages, so the audit report can never drift from runtime policy.

Requires the HIP runtime because importing a kernel backend package registers
kernels; skipped on machines without ROCm.
"""

from __future__ import annotations

import ctypes
import importlib.util
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "gguf_quant_route_audit.py"

spec = importlib.util.spec_from_file_location("gguf_quant_route_audit_parity", SCRIPT)
assert spec and spec.loader
audit = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = audit
spec.loader.exec_module(audit)

from hipengine.loading.qwen35_gguf_policy import (  # noqa: E402
    gguf_fp16_recurrent_state_default,
    resolve_gguf_dense_flags,
)


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


@pytest.mark.skipif(not _hip_available(), reason="ROCm/HIP runtime not available")
@pytest.mark.parametrize("backend", ["hip_gfx1100", "hip_gfx1151"])
@pytest.mark.parametrize("file_type_name", ["MOSTLY_Q4_K_M", "MOSTLY_Q4_K_S", None])
def test_source_reader_matches_runtime_capability_resolution(backend, file_type_name):
    from hipengine.kernels.backends import backend_package_capability

    runtime_flags = resolve_gguf_dense_flags(
        backend, file_type_name, capability_reader=backend_package_capability
    )
    source_flags = resolve_gguf_dense_flags(
        backend, file_type_name, capability_reader=audit.source_capability_reader()
    )
    assert source_flags == runtime_flags

    assert (
        gguf_fp16_recurrent_state_default(
            backend, file_type_name, capability_reader=backend_package_capability
        )
        == gguf_fp16_recurrent_state_default(
            backend, file_type_name, capability_reader=audit.source_capability_reader()
        )
    )
