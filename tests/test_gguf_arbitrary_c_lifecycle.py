from __future__ import annotations

import pytest

from scripts.gguf_arbitrary_c_lifecycle import (
    _all_packed,
    _group_masks,
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
    assert args.backend == "hip_gfx1100"


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


def test_gguf_arbitrary_c_lifecycle_rejects_non_arbitrary_shape_before_model_io() -> None:
    args = build_parser().parse_args(["--model", "/missing/model.gguf", "--rows", "8"])
    with pytest.raises(ValueError, match="greater than 8"):
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
