"""RED/GREEN test for the Q5_K gemv numpy adapter (M6).

Tests that the hip GPU ``gguf_q5_k_gemv_f32_f32_out`` kernel (via a numpy-in/out
adapter) matches the ``cpu_reference.gguf_q5_k_gemv`` oracle on synthetic Q5_K
data.  This is the first M6 step: wiring existing Q5_K GPU kernels into the
mtp_nextn FFN composite so the real Qwen3.6-35B-A3B GGUF model (which has Q5_K
'down' expert weights) can run on the GPU backend.
"""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tests"))


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


pytestmark = pytest.mark.skipif(not _hip_available(), reason="ROCm/HIP not available")

import hipengine.kernels.cpu_reference  # noqa: F401,E402
import hipengine.kernels.hip_gfx1151  # noqa: F401,E402

from _gguf_synthetic_weights import make_q5_k_weight  # noqa: E402
from hipengine.kernels.cpu_reference import gguf_q5_k_gemv  # noqa: E402
from hipengine.quant.gguf import GGMLQuantizationType, dequantize_gguf_data  # noqa: E402


def test_q5_k_adapter_registered():
    """RED gate: the Q5_K numpy adapter must be registered under the mtp key."""
    from hipengine.kernels.registry import KernelKey, registered_keys

    target = KernelKey("hip_gfx1100", "mtp_q5_k_gemv", "gguf_f32", "qwen35")
    assert target in registered_keys(), (
        "no hip_gfx1100 mtp_q5_k_gemv adapter registered (M6 wiring missing)"
    )
    target1151 = KernelKey("hip_gfx1151", "mtp_q5_k_gemv", "gguf_f32", "qwen35")
    assert target1151 in registered_keys(), (
        "no hip_gfx1151 mtp_q5_k_gemv adapter registered"
    )


@pytest.mark.parametrize("backend", ["hip_gfx1100", "hip_gfx1151"])
@pytest.mark.parametrize(
    ("rows", "in_features", "out_features"),
    [
        (1, 256, 5),
        (2, 512, 8),
        (1, 768, 3),
    ],
)
def test_hip_q5_k_gemv_matches_cpu_reference(backend, rows, in_features, out_features):
    from hipengine.kernels.registry import resolve

    try:
        kernel = resolve(
            backend=backend, layer="mtp_q5_k_gemv", quant="gguf_f32", variant="qwen35"
        )
    except Exception:
        pytest.skip(f"{backend} mtp_q5_k_gemv not registered yet (RED)")

    if kernel is gguf_q5_k_gemv:
        pytest.skip(f"{backend} returned cpu_reference via fallback (RED)")

    x = (np.arange(rows * in_features, dtype=np.float32).reshape(rows, in_features) % 13 - 6) / 8.0
    qweight = make_q5_k_weight(out_features=out_features, in_features=in_features)

    expected = gguf_q5_k_gemv(x, qweight)
    actual = kernel(x, qweight)
    actual = np.asarray(actual, dtype=np.float32)

    assert actual.shape == expected.shape, f"shape {actual.shape} != {expected.shape}"
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)