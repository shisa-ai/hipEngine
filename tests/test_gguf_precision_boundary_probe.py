from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.gguf_precision_boundary_probe import compute_precision_boundary_metrics

FIXTURE = Path("benchmarks/fixtures/qwen35_gguf_precision_boundary_fixture.json")


def test_precision_boundary_probe_matches_committed_synthetic_fixture() -> None:
    payload = json.loads(FIXTURE.read_text())

    actual = compute_precision_boundary_metrics(
        np.asarray(payload["token_embedding"], dtype=np.float32),
        np.asarray(payload["attn_norm_weight"], dtype=np.float32),
        np.asarray(payload["ssm_alpha"], dtype=np.float32),
        np.asarray(payload["ssm_beta"], dtype=np.float32),
        rms_norm_eps=float(payload["rms_norm_eps"]),
    )

    assert payload["kind"] == "qwen35_gguf_precision_boundary_fixture"
    _assert_metrics_close(actual["attn_norm_boundary"], payload["expected"]["attn_norm_boundary"])
    _assert_metrics_close(
        actual["projections"]["ssm_alpha"],
        payload["expected"]["projections"]["ssm_alpha"],
    )
    _assert_metrics_close(
        actual["projections"]["ssm_beta"],
        payload["expected"]["projections"]["ssm_beta"],
    )
    assert actual["attn_norm_boundary"]["max_abs_diff"] > 0.0
    assert actual["projections"]["ssm_alpha"]["rows"] == 3
    assert actual["projections"]["ssm_beta"]["rows"] == 2


def test_precision_boundary_probe_rejects_shape_mismatches() -> None:
    token = np.ones((1, 4), dtype=np.float32)
    norm = np.ones((4,), dtype=np.float32)
    proj = np.ones((2, 4), dtype=np.float32)

    with pytest.raises(ValueError, match="token_embedding must be a 1D hidden row"):
        compute_precision_boundary_metrics(token, norm, proj, proj, rms_norm_eps=1.0e-6)

    with pytest.raises(ValueError, match="attn_norm_weight shape"):
        compute_precision_boundary_metrics(token[0], norm[:3], proj, proj, rms_norm_eps=1.0e-6)

    with pytest.raises(ValueError, match="ssm_alpha must be a 2D projection"):
        compute_precision_boundary_metrics(token[0], norm, proj[:, :3], proj, rms_norm_eps=1.0e-6)


def _assert_metrics_close(actual: dict[str, object], expected: dict[str, object]) -> None:
    assert actual["shape"] == expected["shape"]
    for key, expected_value in expected.items():
        if key == "shape":
            continue
        if isinstance(expected_value, list):
            np.testing.assert_allclose(actual[key], expected_value, rtol=1.0e-7, atol=1.0e-7)
        elif isinstance(expected_value, (int, float)):
            assert actual[key] == pytest.approx(expected_value, rel=1.0e-7, abs=1.0e-7)
        else:
            assert actual[key] == expected_value
