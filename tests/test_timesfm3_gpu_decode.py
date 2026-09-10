from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
import pytest

from hipengine.loading.hf_cache import resolve_model_path

FIXTURES = (
    Path(__file__).parent / "fixtures" / "cpu_reference" / "timesfm_3p0_decode.npz",
    Path(__file__).parent / "fixtures" / "cpu_reference" / "timesfm_3p0_decode_edge.npz",
    Path(__file__).parent / "fixtures" / "cpu_reference" / "timesfm_3p0_decode_covmask.npz",
)
PINNED_MODEL_ID = "google/timesfm-3.0-pytorch"


def _snapshot() -> Path | None:
    try:
        path = resolve_model_path(PINNED_MODEL_ID)
    except Exception:
        return None
    return path if path.is_dir() else None


def _gpu_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return _snapshot() is not None


pytestmark = pytest.mark.skipif(
    not all(f.is_file() for f in FIXTURES) or not _gpu_available(),
    reason="TimesFM 3.0 fixtures, checkpoint, or ROCm/HIP runtime not available",
)


def _fixture_kwargs(fixture) -> dict:
    kw: dict = {}
    if "past_only_covariates" in fixture.files:
        kw["past_only_covariates"] = fixture["past_only_covariates"]
    if "past_future_covariates" in fixture.files:
        kw["past_future_covariates"] = fixture["past_future_covariates"]
    if "target_mask" in fixture.files:
        kw["target_mask"] = fixture["target_mask"]
    if "global_mask" in fixture.files:
        kw["mask"] = fixture["global_mask"]
    if "past_only_mask" in fixture.files:
        kw["past_only_mask"] = fixture["past_only_mask"]
    if "past_future_mask" in fixture.files:
        kw["past_future_mask"] = fixture["past_future_mask"]
    return kw


def _run_decode(precision: str, fixture_path: Path):
    from hipengine.loading.timesfm3 import load_timesfm3_model
    from hipengine.runtime.timesfm3_decode import TimesFM3GPUDecoder

    fixture = np.load(fixture_path)
    local = load_timesfm3_model(str(_snapshot()))
    try:
        decoder = TimesFM3GPUDecoder(local, precision=precision)
        try:
            return (
                decoder.decode(fixture["target"], int(fixture["horizon"]), **_fixture_kwargs(fixture)),
                fixture,
            )
        finally:
            decoder.close()
    finally:
        local.free()


@pytest.mark.parametrize("fixture_path", FIXTURES, ids=["base", "edge", "covmask"])
def test_gpu_decode_fp32_strict_parity(fixture_path) -> None:
    out, fixture = _run_decode("fp32", fixture_path)
    assert bool(np.isfinite(out).all())
    np.testing.assert_allclose(
        out, fixture["decode_logits"], atol=1.0e-4, rtol=1.0e-2
    )


@pytest.mark.parametrize("fixture_path", FIXTURES, ids=["base", "edge", "covmask"])
def test_gpu_decode_fp16_production_gate(fixture_path) -> None:
    """FP16: max <= 2%, mean <= 0.5% of per-series signal scale vs the oracle."""

    out, fixture = _run_decode("fp16", fixture_path)
    error = np.abs(out.astype(np.float64) - fixture["decode_logits"].astype(np.float64))
    assert bool(np.isfinite(error).all())
    target = fixture["target"]
    mask = fixture["target_mask"] if "target_mask" in fixture.files else None
    num_targets = target.shape[1]
    for b in range(target.shape[0]):
        scale = float(
            np.std(
                target[b, 0][
                    ~mask[b, 0] if mask is not None else np.ones(target.shape[2], bool)
                ]
            )
        )
        for u in range(num_targets):
            m = mask[b, u] if mask is not None else None
            s = scale if m is None else float(np.std(target[b, u][~m]))
            assert error[b, u].max() / s <= 0.02, f"b{b} u{u} exceeds 2% of signal scale"
            assert error[b, u].mean() / s <= 0.005, f"b{b} u{u} exceeds 0.5% mean"


def test_gpu_decode_rejects_unknown_precision() -> None:
    from hipengine.loading.timesfm3 import load_timesfm3_model
    from hipengine.runtime.timesfm3_decode import TimesFM3GPUDecoder

    local = load_timesfm3_model(str(_snapshot()))
    try:
        with pytest.raises(ValueError, match="precision"):
            TimesFM3GPUDecoder(local, precision="bf16")
    finally:
        local.free()
