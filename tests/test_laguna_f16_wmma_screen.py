from __future__ import annotations

import json

import pytest

from scripts.laguna_f16_library_ceiling import DEFAULT_ROWS, _FAMILIES
from scripts.laguna_f16_wmma_screen import _load_library_ceiling, _summarize


def _samples(*, exact: float = 10.0, wmma: float = 2.0):
    return {
        row: {
            family: {
                "exact": [exact, exact * 1.01, exact * 0.99],
                "wmma": [wmma, wmma * 1.01, wmma * 0.99],
            }
            for family in _FAMILIES
        }
        for row in DEFAULT_ROWS
    }


def _library(value: float = 1.0):
    return {
        row: {family: value for family in _FAMILIES} for row in DEFAULT_ROWS
    }


def test_laguna_f16_wmma_summary_admits_positive_matrix_leaf() -> None:
    samples = _samples()
    summary = _summarize(DEFAULT_ROWS, samples, samples, _library())

    assert summary["pass"] is True
    assert summary["failed_checks"] == []
    assert summary["m128_weighted_projection_sum"]["speedup"] == pytest.approx(5.0)
    assert summary["shapes"]["128"]["families"]["full"][
        "wmma_speedup_vs_exact"
    ] == pytest.approx(5.0)
    assert summary["shapes"]["128"]["families"]["full"][
        "wmma_speedup_vs_library_inclusive"
    ] == pytest.approx(0.5)


def test_laguna_f16_wmma_summary_fails_closed_on_any_slow_family() -> None:
    samples = _samples()
    samples[16]["swa"]["wmma"] = [11.0, 11.1, 10.9]

    summary = _summarize(DEFAULT_ROWS, samples, samples, _library())

    assert summary["pass"] is False
    assert "rows_16_swa_wmma_not_faster" in summary["failed_checks"]


def test_laguna_f16_wmma_summary_requires_material_m128_speedup() -> None:
    samples = _samples(exact=10.0, wmma=6.0)

    summary = _summarize(DEFAULT_ROWS, samples, samples, _library())

    assert summary["pass"] is False
    assert "m128_weighted_speedup_below_2x" in summary["failed_checks"]


def test_laguna_f16_wmma_loads_only_passing_library_ceiling(tmp_path) -> None:
    artifact = tmp_path / "ceiling.json"
    shapes = {
        str(row): {
            "families": {
                family: {"hipblaslt_inclusive": {"gpu_ms_median": row / 100.0}}
                for family in _FAMILIES
            }
        }
        for row in DEFAULT_ROWS
    }
    artifact.write_text(
        json.dumps(
            {
                "kind": "hipengine_laguna_f16_library_ceiling",
                "pass": True,
                "summary": {"shapes": shapes},
            }
        ),
        encoding="utf-8",
    )

    loaded = _load_library_ceiling(artifact, DEFAULT_ROWS)

    assert loaded[128]["full"] == pytest.approx(1.28)
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    payload["pass"] = False
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="passing Laguna F16 ceiling"):
        _load_library_ceiling(artifact, DEFAULT_ROWS)
