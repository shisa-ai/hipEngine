from __future__ import annotations

import json

import pytest

from scripts.gguf_gdn_prefill_ab import (
    BenchmarkError,
    _load_correctness_gate,
    _parse_contexts,
    _promotion_decision,
    _summarize_context,
)


def test_parse_contexts_requires_unique_positive_integers() -> None:
    assert _parse_contexts("512,4096") == (512, 4096)
    with pytest.raises(BenchmarkError, match="positive"):
        _parse_contexts("512,0")
    with pytest.raises(BenchmarkError, match="duplicates"):
        _parse_contexts("512,512")


def test_correctness_gate_requires_passing_context_coverage(tmp_path) -> None:
    path = tmp_path / "correctness.json"
    path.write_text(
        json.dumps(
            {
                "kind": "hipengine_gguf_gdn_prefill_exact_matrix",
                "source_revision": "abc123",
                "classification": {"passed": True, "status": "accepted_exact_matrix"},
                "cases": [
                    {"passed": True, "prompt": {"length": 512}},
                    {"passed": True, "prompt": {"length": 4096}},
                ],
            }
        ),
        encoding="utf-8",
    )
    gate = _load_correctness_gate(path, contexts=(512, 4096))
    assert gate["passed"] is True
    assert gate["covered_contexts"] == [512, 4096]
    assert len(gate["sha256"]) == 64
    with pytest.raises(BenchmarkError, match="does not cover"):
        _load_correctness_gate(path, contexts=(512, 1024, 4096))


def test_context_summary_uses_balanced_mode_samples_and_exact_tokens() -> None:
    rows = [
        {
            "repetition": 0,
            "order": ["fused", "chain"],
            "modes": {
                "fused": {"wall_ms": 100.0, "token_id": 9707},
                "chain": {"wall_ms": 90.0, "token_id": 9707},
            },
        },
        {
            "repetition": 1,
            "order": ["chain", "fused"],
            "modes": {
                "chain": {"wall_ms": 92.0, "token_id": 9707},
                "fused": {"wall_ms": 104.0, "token_id": 9707},
            },
        },
    ]
    summary = _summarize_context(rows, expected_token_id=9707)
    assert summary["statistics"]["fused"]["median_ms"] == 102.0
    assert summary["statistics"]["chain"]["median_ms"] == 91.0
    assert summary["paired_chain_minus_fused_ms"] == [-10.0, -12.0]
    assert summary["tokens_exact"] is True
    assert summary["chain_wins"] is True
    assert summary["chain_speedup_vs_fused"] == pytest.approx(102.0 / 91.0)


def test_promotion_requires_clean_provenance_and_wins_every_context() -> None:
    accepted = _promotion_decision(
        [
            {"tokens_exact": True, "chain_wins": True},
            {"tokens_exact": True, "chain_wins": True},
        ],
        provenance={"dirty": False},
        correctness_gate_passed=True,
    )
    assert accepted["status"] == "promote_chain"
    assert accepted["selected_default"] == "chain"

    rejected = _promotion_decision(
        [
            {"tokens_exact": True, "chain_wins": True},
            {"tokens_exact": True, "chain_wins": False},
        ],
        provenance={"dirty": False},
        correctness_gate_passed=True,
    )
    assert rejected["measurement_valid"] is True
    assert rejected["status"] == "retain_fused_reject_chain_promotion"
    assert rejected["selected_default"] == "fused"

    invalid = _promotion_decision(
        [{"tokens_exact": True, "chain_wins": True}],
        provenance={"dirty": True},
        correctness_gate_passed=True,
    )
    assert invalid["measurement_valid"] is False
    assert invalid["status"] == "invalid_measurement"
