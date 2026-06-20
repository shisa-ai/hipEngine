from __future__ import annotations

import pytest

from scripts.gguf_layer_capture_diff import compare_capture_arrays


def test_compare_capture_arrays_reports_diff_metrics() -> None:
    reference = {
        "arrays": {
            "hidden_in_f32": [1.0, 2.0, -3.0],
            "attn_out_f32": [0.5, -0.5, 1.5],
        }
    }
    candidate = {
        "arrays": {
            "hidden_in_f32": [1.5, 1.0, -2.0],
            "attn_out_f32": [0.25, -0.25, 1.25],
        }
    }

    comparisons = compare_capture_arrays(
        reference,
        candidate,
        keys=("hidden_in_f32", "attn_out_f32"),
    )

    assert comparisons["hidden_in_f32"]["max_abs_diff"] == pytest.approx(1.0)
    assert comparisons["hidden_in_f32"]["diff_sample"] == pytest.approx([0.5, -1.0, 1.0])
    assert comparisons["attn_out_f32"]["mean_abs_diff"] == pytest.approx(0.25)


def test_compare_capture_arrays_rejects_missing_arrays_or_keys() -> None:
    with pytest.raises(ValueError, match="reference artifact must include arrays"):
        compare_capture_arrays({}, {"arrays": {}}, keys=("hidden_in_f32",))
    with pytest.raises(ValueError, match="candidate artifact missing arrays.attn_out_f32"):
        compare_capture_arrays(
            {"arrays": {"attn_out_f32": [1.0]}},
            {"arrays": {}},
            keys=("attn_out_f32",),
        )
    with pytest.raises(ValueError, match="at least one array key"):
        compare_capture_arrays({"arrays": {}}, {"arrays": {}}, keys=())
