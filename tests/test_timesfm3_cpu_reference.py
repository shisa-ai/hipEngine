from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hipengine.loading.hf_cache import resolve_model_path

FIXTURE = Path(__file__).parent / "fixtures" / "cpu_reference" / "timesfm_3p0_decode.npz"
PINNED_MODEL_ID = "google/timesfm-3.0-pytorch"

if not FIXTURE.is_file():
    pytest.skip("TimesFM 3.0 decode fixture not present", allow_module_level=True)


def _cached_snapshot() -> Path | None:
    try:
        path = resolve_model_path(PINNED_MODEL_ID)
    except Exception:
        return None
    return path if path.is_dir() else None


@pytest.fixture(scope="module")
def host_weights():
    snapshot = _cached_snapshot()
    if snapshot is None:
        pytest.skip(f"{PINNED_MODEL_ID} not in local HF cache")
    from hipengine.kernels.cpu_reference.timesfm3 import TimesFM3HostWeights

    return TimesFM3HostWeights.load(str(snapshot))


def test_numpy_decode_matches_torch_oracle(host_weights) -> None:
    from hipengine.kernels.cpu_reference.timesfm3 import timesfm3_decode

    fixture = np.load(FIXTURE)
    target = fixture["target"]
    target_mask = fixture["target_mask"]
    past_only = fixture["past_only_covariates"]
    past_future = fixture["past_future_covariates"]
    horizon = int(fixture["horizon"])
    assert target.shape == (2, 2, 512)
    assert past_future.shape == (2, 1, 512 + horizon)
    assert target_mask[1, :, :64].all() and not target_mask[1, :, 64:].any()

    out = timesfm3_decode(
        host_weights,
        target,
        horizon,
        past_only_covariates=past_only,
        past_future_covariates=past_future,
        target_mask=target_mask,
    )
    ref = fixture["decode_logits"]
    assert out.shape == ref.shape == (2, 4, horizon, 9)
    np.testing.assert_allclose(out, ref, atol=1.0e-4, rtol=1.0e-3)
    assert np.abs(out - ref).max() < 1.0e-5  # measured 5.7e-6


def test_numpy_decode_deterministic(host_weights) -> None:
    from hipengine.kernels.cpu_reference.timesfm3 import timesfm3_decode

    fixture = np.load(FIXTURE)
    horizon = int(fixture["horizon"])
    kwargs = dict(
        past_only_covariates=fixture["past_only_covariates"],
        past_future_covariates=fixture["past_future_covariates"],
        target_mask=fixture["target_mask"],
    )
    out1 = timesfm3_decode(host_weights, fixture["target"], horizon, **kwargs)
    out2 = timesfm3_decode(host_weights, fixture["target"], horizon, **kwargs)
    np.testing.assert_array_equal(out1, out2)


def test_numpy_decode_covariate_invariance(host_weights) -> None:
    """Without covariates the model must not use them: 2-variate output."""

    from hipengine.kernels.cpu_reference.timesfm3 import timesfm3_decode

    fixture = np.load(FIXTURE)
    horizon = int(fixture["horizon"])
    out = timesfm3_decode(
        host_weights,
        fixture["target"][:1],
        horizon,
        target_mask=fixture["target_mask"][:1],
    )
    assert out.shape == (1, 2, horizon, 9)
    assert np.isfinite(out).all()
    # The univariate decode differs from the covariate-conditioned one.
    with_cov = timesfm3_decode(
        host_weights,
        fixture["target"][:1],
        horizon,
        past_only_covariates=fixture["past_only_covariates"][:1],
        past_future_covariates=fixture["past_future_covariates"][:1],
        target_mask=fixture["target_mask"][:1],
    )
    assert not np.allclose(out, with_cov[:, :2], atol=1e-3)

EDGE_FIXTURE = Path(__file__).parent / "fixtures" / "cpu_reference" / "timesfm_3p0_decode_edge.npz"


@pytest.mark.skipif(not EDGE_FIXTURE.is_file(), reason="edge fixture not present")
def test_numpy_decode_edge_cases_match_torch_oracle(host_weights) -> None:
    """Unaligned context/horizon, rank-2 global mask, true univariate, detrend."""

    from hipengine.kernels.cpu_reference.timesfm3 import timesfm3_decode

    fixture = np.load(EDGE_FIXTURE)
    horizon = int(fixture["horizon"])
    assert fixture["target"].shape == (2, 1, 500)  # true univariate, unaligned
    assert horizon == 100

    out = timesfm3_decode(
        host_weights,
        fixture["target"],
        horizon,
        target_mask=fixture["target_mask"],
        mask=fixture["global_mask"],
    )
    ref = fixture["decode_logits"]
    assert out.shape == ref.shape == (2, 1, horizon, 9)
    np.testing.assert_allclose(out, ref, atol=1.0e-4, rtol=1.0e-3)
    assert np.abs(out - ref).max() < 2.0e-5  # measured 1.1e-5


@pytest.mark.skipif(not EDGE_FIXTURE.is_file(), reason="edge fixture not present")
def test_numpy_forward_freeze_after_matches_torch_oracle(host_weights) -> None:
    """freeze_after freezes mean/std post-hoc while counts accumulate."""

    from hipengine.kernels.cpu_reference.timesfm3 import timesfm3_forward

    fixture = np.load(EDGE_FIXTURE)
    out = timesfm3_forward(
        host_weights,
        fixture["forward_values"],
        fixture["forward_masks"],
        fixture["forward_is_target"],
        None,
        freeze_after=int(fixture["forward_freeze_after"]),
    )
    ref = fixture["forward_logits"]
    assert out.shape == ref.shape
    np.testing.assert_allclose(out, ref, atol=1.0e-4, rtol=1.0e-3)
    assert np.abs(out - ref).max() < 1.0e-4  # measured 4.1e-5
