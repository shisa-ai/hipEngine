"""RED/GREEN test for the hip_gfx1100/hip_gfx1151 ``mtp_nextn_eh_proj`` kernel.

M3 deliverable: a *real* GPU NextN eh_proj sub-kernel (RMSNorm x2 + concat +
F32 GEMV) registered under ``KernelKey(backend, "mtp_nextn_eh_proj", "gguf_f32",
"qwen35")``.  The registry otherwise auto-falls-back to the ``cpu_reference``
numpy oracle (``registry._candidate_keys`` appends ``cpu_reference`` last), so
this test asserts the *exact* hip key is registered -- otherwise it would
silently pass on the numpy fallback, which is exactly the drift that left M3
with no native runtime kernel.

Correctness gate: output matches ``cpu_reference.qwen35_gguf_mtp_eh_proj`` on
the committed F32 MTP fixture within a tight f32 tolerance.
"""

from __future__ import annotations

import ctypes
import json
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = REPO_ROOT / "benchmarks/fixtures/qwen35_gguf_mtp_nextn_cpu_reference_fixture.json"


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


pytestmark = pytest.mark.skipif(not _hip_available(), reason="ROCm/HIP not available")

# Import at module scope (collection time) so the hip mtp_nextn registrations
# are captured in tests/conftest.py's baseline snapshot.  Importing the package
# is pure-Python (no GPU touch); registration happens at import.
import hipengine.kernels.cpu_reference  # noqa: F401,E402 - registers cpu oracle
import hipengine.kernels.hip_gfx1151  # noqa: F401,E402 - registers hip aliases incl. mtp_nextn


@pytest.fixture(scope="module")
def eh_proj_inputs() -> dict:
    fixture = json.loads(FIXTURE.read_text())
    inputs = fixture["inputs"]
    return {
        "hidden_seed": np.ascontiguousarray(inputs["hidden_seed"], dtype=np.float32),
        "token_embedding": np.ascontiguousarray(inputs["token_embedding"], dtype=np.float32),
        "eh_proj_weight": np.ascontiguousarray(inputs["eh_proj_weight"], dtype=np.float32),
        "hnorm_weight": np.ascontiguousarray(inputs["hnorm_weight"], dtype=np.float32),
        "enorm_weight": np.ascontiguousarray(inputs["enorm_weight"], dtype=np.float32),
        "eps": float(fixture["kwargs"].get("eps", 1e-6)),
    }


def _exact_key_registered(backend: str) -> bool:
    from hipengine.kernels.registry import KernelKey, registered_keys

    import hipengine.kernels.cpu_reference  # noqa: F401 - registers cpu oracle
    import hipengine.kernels.hip_gfx1151  # noqa: F401 - registers hip aliases

    target = KernelKey(backend, "mtp_nextn_eh_proj", "gguf_f32", "qwen35")
    return target in registered_keys()


def test_hip_eh_proj_key_registered():
    """RED gate: a real hip mtp_nextn_eh_proj kernel must be registered, not the
    cpu_reference fallback."""
    assert _exact_key_registered("hip_gfx1100"), (
        "no hip_gfx1100 mtp_nextn_eh_proj kernel registered (M3 native kernel missing; "
        "registry is falling back to cpu_reference)"
    )
    assert _exact_key_registered("hip_gfx1151"), (
        "no hip_gfx1151 mtp_nextn_eh_proj alias registered"
    )


@pytest.mark.parametrize("backend", ["hip_gfx1100", "hip_gfx1151"])
def test_hip_eh_proj_matches_cpu_reference(backend, eh_proj_inputs):
    if not _exact_key_registered(backend):
        pytest.skip(f"{backend} mtp_nextn_eh_proj not registered yet (RED)")

    from hipengine.kernels.cpu_reference.ops import qwen35_gguf_mtp_eh_proj
    from hipengine.kernels.registry import resolve

    expected = qwen35_gguf_mtp_eh_proj(
        eh_proj_inputs["hidden_seed"],
        eh_proj_inputs["token_embedding"],
        eh_proj_inputs["eh_proj_weight"],
        eh_proj_inputs["hnorm_weight"],
        eh_proj_inputs["enorm_weight"],
        eps=eh_proj_inputs["eps"],
    )

    kernel = resolve(
        backend=backend, layer="mtp_nextn_eh_proj", quant="gguf_f32", variant="qwen35"
    )
    # Defense in depth: the resolved kernel must be the native hip kernel, not
    # the cpu_reference numpy oracle reached via fallback.
    assert kernel is not qwen35_gguf_mtp_eh_proj, (
        f"resolve({backend}) returned the cpu_reference oracle via fallback; "
        "native hip kernel not registered"
    )

    actual = kernel(
        eh_proj_inputs["hidden_seed"],
        eh_proj_inputs["token_embedding"],
        eh_proj_inputs["eh_proj_weight"],
        eh_proj_inputs["hnorm_weight"],
        eh_proj_inputs["enorm_weight"],
        eps=eh_proj_inputs["eps"],
    )
    actual = np.asarray(actual, dtype=np.float32)

    assert actual.shape == expected.shape, f"shape {actual.shape} != {expected.shape}"
    max_abs = float(np.max(np.abs(actual - expected)))
    assert max_abs < 1e-4, f"{backend} eh_proj max_abs={max_abs} exceeds 1e-4 vs cpu_reference"
