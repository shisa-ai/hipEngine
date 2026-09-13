from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
import pytest

from hipengine.loading.hf_cache import resolve_model_path

FIXTURE = Path(__file__).parent / "fixtures" / "cpu_reference" / "timesfm_2p5_200m_decode.npz"
PINNED_MODEL_ID = "google/timesfm-2.5-200m-pytorch"


def _gpu_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    snapshot = _snapshot()
    if snapshot is None:
        return False
    return True


def _snapshot() -> Path | None:
    try:
        path = resolve_model_path(PINNED_MODEL_ID)
    except Exception:
        return None
    return path if path.is_dir() else None


pytestmark = pytest.mark.skipif(
    not FIXTURE.is_file() or not _gpu_available(),
    reason="TimesFM fixture, checkpoint, or ROCm/HIP runtime not available",
)


def _run_decode(precision: str):
    from hipengine.loading.timesfm import load_timesfm_model
    from hipengine.runtime.timesfm_decode import TimesFMGPUDecoder

    fixture = np.load(FIXTURE)
    inputs, masks, horizon = fixture["inputs"], fixture["masks"], int(fixture["horizon"])
    local = load_timesfm_model(str(_snapshot()))
    try:
        decoder = TimesFMGPUDecoder(local, precision=precision)
        try:
            return decoder.decode(horizon, inputs, masks), fixture
        finally:
            decoder.close()
    finally:
        local.free()


def test_gpu_decode_fp32_strict_parity() -> None:
    (pf, qs, ar), fixture = _run_decode("fp32")
    np.testing.assert_allclose(pf, fixture["renormed_outputs"], atol=5.0e-4, rtol=1.0e-2)
    np.testing.assert_allclose(qs, fixture["quantile_spread"], atol=5.0e-4, rtol=1.0e-2)
    np.testing.assert_allclose(ar, fixture["ar_outputs"], atol=5.0e-4, rtol=1.0e-2)


def test_gpu_decode_fp16_production_gate() -> None:
    """FP16 production path: max <= 2%, mean <= 0.5% of per-series signal scale."""

    (pf, qs, ar), fixture = _run_decode("fp16")
    inputs, masks = fixture["inputs"], fixture["masks"]
    for name, actual, expected in (
        ("renormed_outputs", pf, fixture["renormed_outputs"]),
        ("quantile_spread", qs, fixture["quantile_spread"]),
        ("ar_outputs", ar, fixture["ar_outputs"]),
    ):
        error = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
        for b in range(expected.shape[0]):
            scale = float(np.std(inputs[b][~masks[b]]))
            assert error[b].max() / scale <= 0.02, f"{name} b{b} exceeds 2% of signal scale"
            assert error[b].mean() / scale <= 0.005, f"{name} b{b} exceeds 0.5% mean of signal scale"


def test_gpu_decode_rejects_unknown_precision() -> None:
    from hipengine.loading.timesfm import load_timesfm_model
    from hipengine.runtime.timesfm_decode import TimesFMGPUDecoder

    local = load_timesfm_model(str(_snapshot()))
    try:
        with pytest.raises(ValueError, match="precision"):
            TimesFMGPUDecoder(local, precision="bf16")
    finally:
        local.free()
