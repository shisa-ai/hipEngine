from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.gguf_arbitrary_c_lifecycle import (
    _all_packed,
    _expected_dense_group_masks,
    _expected_hole_group_masks,
    _group_masks,
    _load_quality_gate,
    _resolve_widths,
    _state_kv_accepted,
    build_parser,
    run,
)


def test_gguf_arbitrary_c_lifecycle_defaults_cover_both_physical_windows() -> None:
    args = build_parser().parse_args(["--model", "/tmp/model.gguf"])

    assert args.rows == 13
    assert tuple(args.cancel_slots) == (2, 10)
    assert args.original_max_tokens == 5
    assert args.newcomer_max_tokens == 3
    assert args.prefill_chunk_size == 256
    assert args.compact_after_middle_hole is False
    assert args.backend == "hip_gfx1100"

    compact = build_parser().parse_args(
        ["--model", "/tmp/model.gguf", "--compact-after-middle-hole"]
    )
    assert compact.compact_after_middle_hole is True


def test_gguf_arbitrary_c_lifecycle_summarizes_declared_packed_masks() -> None:
    plan = {
        "group_count": 2,
        "groups": [
            {
                "physical_rows": 8,
                "active_mask": [True, True, False, True, True, True, True, True],
                "execution_path": "packed_native",
            },
            {
                "physical_rows": 8,
                "active_mask": [True, True, False, True, True, False, False, False],
                "execution_path": "packed_native",
            },
        ],
    }

    assert _all_packed(plan)
    assert _group_masks(plan) == ["11011111", "11011000"]
    plan["groups"][1]["execution_path"] = "serial_fallback"
    assert not _all_packed(plan)


def test_gguf_arbitrary_c_lifecycle_accepts_single_physical_group_shape() -> None:
    args = build_parser().parse_args(
        [
            "--model",
            "/missing/model.gguf",
            "--rows",
            "8",
            "--cancel-slots",
            "2",
            "6",
        ]
    )
    with pytest.raises(ValueError, match="model does not exist"):
        run(args)


def test_gguf_arbitrary_c_lifecycle_derives_single_and_cross_group_masks() -> None:
    assert _expected_dense_group_masks(8) == ["11111111"]
    assert _expected_dense_group_masks(13) == ["11111111", "11111000"]
    assert _expected_hole_group_masks(8, (2, 6), compact=True) == ["11111100"]
    assert _expected_hole_group_masks(13, (2, 10), compact=False) == [
        "11011111",
        "11011000",
    ]


def test_gguf_arbitrary_c_lifecycle_derives_direct_width_masks() -> None:
    widths = (1, 2, 3, 4, 5, 6, 7, 8)
    assert _expected_dense_group_masks(3, widths) == ["111"]
    assert _expected_dense_group_masks(5, widths) == ["11111"]
    assert _expected_dense_group_masks(6, widths) == ["111111"]
    assert _expected_dense_group_masks(7, widths) == ["1111111"]
    assert _expected_dense_group_masks(13, widths) == ["11111111", "11111"]
    assert _expected_hole_group_masks(5, (2,), compact=False, buckets=widths) == [
        "11011"
    ]
    assert _expected_hole_group_masks(13, (2, 10), compact=False, buckets=widths) == [
        "11011111",
        "11011",
    ]
    # The masked default remains unchanged from the production owner.
    assert _expected_dense_group_masks(5, (1, 2, 4, 8)) == ["11111000"]


def test_gguf_arbitrary_c_lifecycle_resolves_active_width_set(monkeypatch) -> None:
    monkeypatch.delenv("HIPENGINE_GGUF_SHARED_SLOT_AR_PHYSICAL_WIDTHS", raising=False)
    assert _resolve_widths() == (1, 2, 4, 8)
    monkeypatch.setenv(
        "HIPENGINE_GGUF_SHARED_SLOT_AR_PHYSICAL_WIDTHS", "1,2,3,4,5,6,7,8"
    )
    assert _resolve_widths() == (1, 2, 3, 4, 5, 6, 7, 8)
    monkeypatch.setenv("HIPENGINE_GGUF_SHARED_SLOT_AR_PHYSICAL_WIDTHS", "1 2 4 8")
    assert _resolve_widths() == (1, 2, 4, 8)
    monkeypatch.setenv("HIPENGINE_GGUF_SHARED_SLOT_AR_PHYSICAL_WIDTHS", "1 3 2")
    with pytest.raises(ValueError, match="sorted unique widths starting at c1"):
        _resolve_widths()


def test_gguf_arbitrary_c_lifecycle_requires_explicit_arithmetic_drift_policy(
    tmp_path: Path,
) -> None:
    assert _state_kv_accepted(bit_exact=True, allow_c1_arithmetic_drift=False)
    assert not _state_kv_accepted(bit_exact=False, allow_c1_arithmetic_drift=False)
    assert _state_kv_accepted(bit_exact=False, allow_c1_arithmetic_drift=True)

    model = tmp_path / "model.gguf"
    model.write_bytes(b"fixture")
    artifact = tmp_path / "quality.json"
    artifact.write_text(
        json.dumps(
            {
                "kind": "hipengine_execution_profile_gguf_batch_route_requalification_capture",
                "measurement_valid": True,
                "quality": {
                    "hard_gates_passed": True,
                    "summary": {
                        "kl_max": 0.0,
                        "top1_agreement": 1.0,
                        "rows": 1050,
                    },
                },
                "provenance": {
                    "dirty": False,
                    "resolved_backend": "hip_gfx1151",
                    "model_path": str(model),
                    "hipengine_commit": "abc123",
                    "host_name": "zbook",
                },
            }
        ),
        encoding="utf-8",
    )
    gate = _load_quality_gate(artifact, model=model, backend="hip_gfx1151")
    assert gate["rows"] == 1050
    assert gate["hard_gate_passed"] is True


def test_gguf_arbitrary_c_lifecycle_rejects_too_small_shape_before_model_io() -> None:
    args = build_parser().parse_args(["--model", "/missing/model.gguf", "--rows", "3"])
    with pytest.raises(ValueError, match="at least 4"):
        run(args)

    args = build_parser().parse_args(
        [
            "--model",
            "/missing/model.gguf",
            "--cancel-slots",
            "2",
            "2",
        ]
    )
    with pytest.raises(ValueError, match="unique"):
        run(args)
