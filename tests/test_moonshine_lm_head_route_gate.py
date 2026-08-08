from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts.moonshine_lm_head_route_gate import (
    SIX_REAL_FIXTURES,
    _fixture_identity,
    build_gate_summary,
    smoke_report_failures,
)


def _good_report(route: str, token_route: str) -> dict[str, object]:
    partial_count = 4_608 if route == "wave8_top1" else 0
    report: dict[str, object] = {
        "all_passed": True,
        "boundary_capture": (
            "all_layer_boundaries" if token_route == "eager" else "final_hidden_only"
        ),
        "encoder_frames": 40,
        "source_frames": 24,
        "failures": [],
        "first_eos_position": 1,
        "generation_tokens_exact": True,
        "selected_tokens_exact": True,
        "post_eos_unselected_token_mismatches": [],
        "positions": [0, 193],
        "max_abs": 0.5,
        "max_relative_l2": 0.001,
        "logit_kl_mean": 0.0,
        "logit_kl_max": 0.0,
        "logit_top1_agreement": 1.0,
        "logit_gate_passed": True,
        "timed_step_allocations": 0,
        "token_route": token_route,
        "lm_head": {
            "route": route,
            "materializes_full_fp16_logits": True,
            "stable_lowest_id_top1": True,
            "partial_count": partial_count,
            "partial_value_dtype": "fp16" if partial_count else None,
            "partial_index_dtype": "int64" if partial_count else None,
            "fallback": "wave8_argmax",
        },
        "resident_nbytes": 1_000_000 + (46_080 if partial_count else 0),
        "teardown_current_bytes": 0,
        "teardown_active_allocations": 0,
        "token_graph": None,
    }
    if token_route == "graph":
        report["token_graph"] = {
            "captured": True,
            "graph_count": 4,
            "buckets": ["position_0", "position_1", "positions_2_3", "positions_4_193"],
            "capture_positions": [0, 1, 2, 4],
            "capture_wall_ms": 1.0,
            "instantiate_wall_ms": 0.1,
            "replay_count": 194,
        }
    return report


def _row(fixture: str, route: str, token_route: str) -> dict[str, object]:
    return {
        "fixture": fixture,
        "lm_head_route": route,
        "token_route": token_route,
        "returncode": 0,
        "report": _good_report(route, token_route),
    }


def test_smoke_report_gate_requires_transcript_state_lifecycle_and_four_graphs() -> None:
    assert smoke_report_failures(
        _good_report("wave8_top1", "graph"),
        expected_route="wave8_top1",
        expected_token_route="graph",
    ) == []

    broken = _good_report("wave8_top1", "graph")
    broken["generation_tokens_exact"] = False
    broken["timed_step_allocations"] = 4
    broken["token_graph"]["graph_count"] = 3  # type: ignore[index]
    failures = smoke_report_failures(
        broken,
        expected_route="wave8_top1",
        expected_token_route="graph",
    )
    assert any("generation_tokens_exact" in failure for failure in failures)
    assert any("timed_step_allocations" in failure for failure in failures)
    assert any("graph_count" in failure for failure in failures)


def test_gate_summary_requires_complete_paired_matrix_and_exact_route_outcomes() -> None:
    fixture = "audio-test-fp16"
    rows = [
        _row(fixture, route, token_route)
        for token_route in ("eager", "graph")
        for route in ("wave8_argmax", "wave8_top1")
    ]
    summary = build_gate_summary(rows, expected_fixtures=(fixture,))
    assert summary["passed"] is True
    assert summary["matrix_rows"] == 4
    assert summary["paired_route_outcomes_exact"] is True
    assert summary["candidate_extra_resident_bytes"] == [46_080]

    missing = rows[:-1]
    with pytest.raises(ValueError, match="matrix"):
        build_gate_summary(missing, expected_fixtures=(fixture,))

    divergent = deepcopy(rows)
    divergent[-1]["report"]["max_abs"] = 0.75  # type: ignore[index]
    summary = build_gate_summary(divergent, expected_fixtures=(fixture,))
    assert summary["passed"] is False
    assert summary["paired_route_outcomes_exact"] is False
    assert any("max_abs" in failure for failure in summary["failures"])


def test_fixture_identity_allows_per_file_eos_metadata_and_includes_eos(tmp_path) -> None:
    fixture_dir = tmp_path / "moonshine-fixtures-six"
    fixture_dir.mkdir()
    for fixture in SIX_REAL_FIXTURES:
        (fixture_dir / f"{fixture}.npz").write_bytes(b"fixture")
        (fixture_dir / f"{fixture}.json").write_text(
            json.dumps(
                {
                    "producer": {
                        "torch": "test",
                        "first_eos_position": "1",
                        "encoder_mask_source": "attention_mask",
                    },
                    "input": {"encoder_frames": 40},
                    "decoder": {"positions": [0, 193], "token_ids": [1, 42, 2, 2]},
                }
            )
        )

    identity = _fixture_identity(fixture_dir)
    assert identity["producer"] == {"torch": "test"}
    assert identity["fixtures"][SIX_REAL_FIXTURES[0]]["generated_ids_through_eos"] == [
        1,
        42,
        2,
    ]


def test_decoder_smoke_cli_uses_the_promoted_runtime_default(monkeypatch) -> None:
    from scripts import moonshine_decoder_smoke

    monkeypatch.setattr(
        "sys.argv",
        [
            "moonshine_decoder_smoke.py",
            "--compiler-version-file",
            "/tmp/hipcc.txt",
            "--prebuild-only",
        ],
    )
    assert moonshine_decoder_smoke.parse_args().lm_head_route == "wave8_top1"


def test_retained_gfx1151_route_admission_artifact_promotes_exact_top1() -> None:
    artifact = json.loads(
        Path(
            "benchmarks/results/2026-08-08-gfx1151-moonshine-lm-head-wave8-top1-promoted.json"
        ).read_text()
    )
    assert artifact["kind"] == "hipengine_moonshine_lm_head_route_admission"
    assert artifact["status"] == "accepted_runtime_default"
    assert artifact["performance_claim"] is False
    assert artifact["decision"] == {
        "promote_wave8_top1": True,
        "runtime_default": "wave8_top1",
        "fallback": "wave8_argmax",
        "reason": (
            "all 24 real-audio eager/graph route rows passed transcript, state, "
            "allocation, graph, paired-outcome, and teardown gates"
        ),
    }
    gate = artifact["gate"]
    assert gate["passed"] is True
    assert gate["matrix_rows"] == 24
    assert gate["fixture_count"] == 6
    assert gate["failures"] == []
    assert gate["candidate_extra_resident_bytes"] == [46_080]
    for field in (
        "all_transcripts_exact_through_eos",
        "all_selected_fixture_tokens_exact",
        "all_logit_gates_passed",
        "all_timed_allocations_zero",
        "all_teardowns_clean",
        "all_graph_runs_four_bucket_194_replay",
        "paired_route_outcomes_exact",
    ):
        assert gate[field] is True
    assert set(artifact["fixture_collection"]["fixtures"]) == set(SIX_REAL_FIXTURES)
    assert all(
        fixture["generated_ids_through_eos"][-1] == 2
        for fixture in artifact["fixture_collection"]["fixtures"].values()
    )
    assert len(artifact["rows"]) == 24
    assert all(row["returncode"] == 0 for row in artifact["rows"])
    assert artifact["provenance"]["dirty"] is False
    assert artifact["provenance"]["untracked_count"] == 0
