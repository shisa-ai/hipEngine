from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

import scripts.laguna_matrix_chunk_bench as matrix_bench
from scripts.laguna_matrix_chunk_bench import (
    LENGTHS,
    MATRIX_ROWS,
    _aggregate,
    _correctness,
    _decision,
    _mode_order,
    _relative_quality,
    _routing_occupancy_summary,
    _validate_protocol,
)


def _rows() -> list[dict]:
    seconds = {
        128: {512: 10.0, 1024: 20.0, 4096: 80.0},
        256: {512: 9.0, 1024: 18.0, 4096: 72.0},
        512: {512: 8.0, 1024: 16.0, 4096: 64.0},
    }
    rows = []
    for repetition in range(2):
        for matrix_rows in MATRIX_ROWS:
            for length in LENGTHS:
                rows.append(
                    {
                        "matrix_rows": matrix_rows,
                        "length": length,
                        "repetition": repetition,
                        "prefill_seconds": seconds[matrix_rows][length],
                        "next_token_id": 17,
                        "next_token_logit_hex": "0x1.0p+0",
                        "logits_sha256": f"logits-{length}",
                        "final_hidden_sha256": f"hidden-{length}",
                        "post_layer_hidden_sha256": f"post-{length}",
                        "kv_sha256": f"kv-{length}",
                        "final_position": length - 1,
                    }
                )
    return rows


def test_laguna_matrix_chunk_order_rotates_all_policies() -> None:
    assert _mode_order(0, 0) == MATRIX_ROWS
    assert _mode_order(0, 1) == (256, 512, 128)
    assert _mode_order(1, 0) == (256, 512, 128)
    wide = (512, 1024, 2048)
    assert _mode_order(0, 0, matrix_rows=wide) == wide
    assert _mode_order(0, 1, matrix_rows=wide) == (1024, 2048, 512)


def test_laguna_matrix_chunk_gate_selects_fastest_exact_policy() -> None:
    rows = _rows()
    correctness = _correctness(rows)
    aggregate = _aggregate(rows)
    decision = _decision(aggregate, correctness, recovered=True)

    assert correctness["pass"] is True
    assert aggregate["512"]["speedup_vs_128"] == pytest.approx(1.25)
    assert decision["pass"] is True
    assert decision["selected_matrix_rows"] == 512
    assert decision["failed_checks"] == []


def test_laguna_matrix_chunk_gate_rejects_regressive_or_inexact_policy() -> None:
    rows = _rows()
    for row in rows:
        if row["matrix_rows"] == 512 and row["length"] == 4096:
            row["prefill_seconds"] = 88.0
    aggregate = _aggregate(rows)
    decision = _decision(aggregate, _correctness(rows), recovered=True)
    assert decision["selected_matrix_rows"] == 256

    mismatched = deepcopy(rows)
    next(
        row
        for row in mismatched
        if row["matrix_rows"] == 256 and row["length"] == 512
    )["kv_sha256"] = "changed"
    failed = _decision(_aggregate(mismatched), _correctness(mismatched), recovered=False)
    assert failed["pass"] is False
    assert "matrix_policy_outputs_or_state_not_exact" in failed["failed_checks"]
    assert "tracked_lifecycle_not_recovered" in failed["failed_checks"]

    failed_probes = _decision(
        _aggregate(_rows()),
        _correctness(_rows()),
        recovered=True,
        full_state_pass=False,
        routing_prefix_pass=False,
    )
    assert failed_probes["pass"] is False
    assert "all_hidden_boundaries_not_exact" in failed_probes["failed_checks"]
    assert "shared_prefix_routing_not_exact" in failed_probes["failed_checks"]


def test_laguna_wide_matrix_quality_gates_full_logits() -> None:
    matrix_rows = (512, 1024, 2048)
    lengths = (512, 1024)
    rows = []
    logits = {
        512: {
            512: [4.0, 2.0, -1.0],
            1024: [5.0, 1.0, -2.0],
        },
        1024: {
            512: [4.0, 2.0, -1.0],
            1024: [4.99, 1.01, -2.0],
        },
        2048: {
            512: [4.0, 2.0, -1.0],
            1024: [-2.0, 1.0, 5.0],
        },
    }
    for repetition in range(2):
        for mode in matrix_rows:
            for length in lengths:
                rows.append(
                    {
                        "matrix_rows": mode,
                        "length": length,
                        "repetition": repetition,
                        "_logits": logits[mode][length],
                    }
                )

    quality = _relative_quality(
        rows,
        matrix_rows=matrix_rows,
        lengths=lengths,
    )

    assert quality["by_matrix_rows"]["512"]["pass"] is True
    assert quality["by_matrix_rows"]["1024"]["pass"] is True
    assert quality["by_matrix_rows"]["1024"]["top1_agreement"] == 1.0
    assert quality["by_matrix_rows"]["2048"]["pass"] is False
    assert quality["by_matrix_rows"]["2048"]["top1_agreement"] == 0.5


def test_laguna_matrix_chunk_protocol_accepts_short_only_gate() -> None:
    _validate_protocol(
        lengths=(512, 1024),
        matrix_rows=MATRIX_ROWS,
        attention_rows=128,
        context_length=1024,
        repetitions=3,
        warmup_rows=128,
    )

    with pytest.raises(ValueError, match="ascending"):
        _validate_protocol(
            lengths=(512, 1024),
            matrix_rows=(128, 512, 256),
            attention_rows=128,
            context_length=1024,
            repetitions=3,
            warmup_rows=128,
        )
    with pytest.raises(ValueError, match="attention rows 128"):
        _validate_protocol(
            lengths=(512, 1024),
            matrix_rows=MATRIX_ROWS,
            attention_rows=256,
            context_length=1024,
            repetitions=3,
            warmup_rows=128,
        )


def test_laguna_matrix_chunk_session_fixes_global_and_swa_attention(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict = {}

    selected_modes: list[tuple[str, str]] = []

    def fake_session(*args, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            set_selected_gate_up_mode=lambda mode: selected_modes.append(("gate", mode)),
            set_selected_down_mode=lambda mode: selected_modes.append(("down", mode)),
        )

    monkeypatch.setattr(matrix_bench, "LagunaGGUFResidentSession", fake_session)
    monkeypatch.setattr(matrix_bench, "_compiler_version", lambda _path: "hipcc")
    owner = SimpleNamespace(weights=object(), runtime=object())
    args = SimpleNamespace(
        context_length=1024,
        backend="hip_gfx1100",
        compiler_version_file=None,
        require_cached_build=True,
        attention_rows=128,
        grouped_exact_iq=True,
    )

    matrix_bench._session(owner, args, matrix_rows=512)

    assert captured["prefill_chunk_size"] == 512
    assert captured["prefill_attention_chunk_size"] == 128
    assert captured["prefill_global_attention_chunk_size"] == 128
    assert selected_modes == [
        ("gate", "grouped_exact"),
        ("down", "grouped_exact"),
    ]


def test_laguna_matrix_chunk_reports_routing_occupancy_and_tails() -> None:
    summary = _routing_occupancy_summary(
        {
            1: (0, 0, 1, 2, 0, 1, 2, 2),
            2: (3, 3, 3, 3, 0, 1, 2, 3),
        },
        rows=4,
        top_k=2,
        expert_count=4,
    )

    assert summary["layers"]["1"]["active_experts"] == 3
    assert summary["layers"]["1"]["mean_routes_per_all_expert"] == 2.0
    assert summary["layers"]["1"]["mean_routes_per_active_expert"] == pytest.approx(
        8 / 3
    )
    assert summary["layers"]["1"]["full_route_slots_by_rowbatch"]["2"] == 6
    assert summary["layers"]["1"]["tail_route_slots_by_rowbatch"]["2"] == 2
    assert summary["layers"]["1"]["experts_at_least_rows"]["4"] == 0
    assert summary["aggregate"]["layer_count"] == 2
    assert summary["aggregate"]["route_slots_per_layer"] == 8
