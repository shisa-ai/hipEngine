from __future__ import annotations

import json

import pytest

from scripts.gguf_gdn_prefill_ab import (
    BenchmarkError,
    _load_correctness_gate,
    _parse_contexts,
    _promotion_decision,
    _summarize_context,
    build_parser,
)


def test_parse_contexts_requires_unique_positive_integers() -> None:
    assert _parse_contexts("512,4096") == (512, 4096)
    with pytest.raises(BenchmarkError, match="positive"):
        _parse_contexts("512,0")
    with pytest.raises(BenchmarkError, match="duplicates"):
        _parse_contexts("512,512")


@pytest.mark.parametrize(
    "mode", ("chain_lds32", "chain_lds64", "chain_lds32_direct")
)
def test_parser_accepts_exact_lds_candidate_modes(mode: str) -> None:
    args = build_parser().parse_args(
        ["--candidate-mode", mode, "--json", "/tmp/out.json"]
    )
    assert args.candidate_mode == mode


def test_correctness_gate_requires_passing_context_coverage(tmp_path) -> None:
    path = tmp_path / "correctness.json"
    path.write_text(
        json.dumps(
            {
                "kind": "hipengine_gguf_gdn_prefill_exact_matrix",
                "source_revision": "abc123",
                "classification": {"passed": True, "status": "accepted_exact_matrix"},
                "protocol": {"modes": ["fused", "chain"]},
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
    assert gate["contract"] == "byte_exact"
    assert gate["default_promotion_eligible"] is True
    assert gate["covered_contexts"] == [512, 4096]
    assert len(gate["sha256"]) == 64
    with pytest.raises(BenchmarkError, match="does not cover"):
        _load_correctness_gate(path, contexts=(512, 1024, 4096))


def test_correctness_gate_accepts_matching_non_exact_project_gate(tmp_path) -> None:
    path = tmp_path / "project-gate.json"
    path.write_text(
        json.dumps(
            {
                "kind": "hipengine_gguf_prefill_optimization_candidate",
                "status": "candidate_gate_passed_pending_promotion",
                "protocol": {
                    "selector": (
                        "HIPENGINE_GGUF_GDN_PREFILL_MODE=chain_wave32_tree"
                    )
                },
                "software": {"candidate_base_commit": "abc123"},
                "correctness": {
                    "project_gate": {
                        "kl_threshold": 0.05,
                        "top1_threshold": 0.9,
                        "cases_passed": 2,
                        "cases_total": 2,
                        "kl_mean_range": [1.0e-6, 2.0e-5],
                        "top1_agreement_min": 1.0,
                        "sampled_tokens_identical": True,
                    },
                    "cases": [
                        {"prompt": "repeated/512", "kl_mean": 2.0e-5},
                        {"prompt": "repeated/4096", "kl_mean": 1.0e-6},
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    gate = _load_correctness_gate(
        path,
        contexts=(512, 4096),
        candidate_mode="chain_wave32_tree",
    )
    assert gate["contract"] == "project_kl_top1_non_exact"
    assert gate["default_promotion_eligible"] is False
    assert gate["source_revision"] == "abc123"

    with pytest.raises(BenchmarkError, match="candidate does not match"):
        _load_correctness_gate(
            path,
            contexts=(512, 4096),
            candidate_mode="chain_tile64",
        )


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
    summary = _summarize_context(
        rows,
        expected_token_id=9707,
        candidate_mode="chain",
    )
    assert summary["statistics"]["fused"]["median_ms"] == 102.0
    assert summary["statistics"]["chain"]["median_ms"] == 91.0
    assert summary["paired_candidate_minus_baseline_ms"] == [-10.0, -12.0]
    assert summary["tokens_exact"] is True
    assert summary["candidate_wins"] is True
    assert summary["candidate_speedup_vs_baseline"] == pytest.approx(102.0 / 91.0)


def test_promotion_requires_clean_provenance_and_wins_every_context() -> None:
    accepted = _promotion_decision(
        [
            {"tokens_exact": True, "candidate_wins": True},
            {"tokens_exact": True, "candidate_wins": True},
        ],
        provenance={"dirty": False},
        correctness_gate={
            "passed": True,
            "contract": "byte_exact",
            "default_promotion_eligible": True,
        },
        candidate_mode="chain",
    )
    assert accepted["status"] == "promote_candidate"
    assert accepted["selected_default"] == "chain"

    rejected = _promotion_decision(
        [
            {"tokens_exact": True, "candidate_wins": True},
            {"tokens_exact": True, "candidate_wins": False},
        ],
        provenance={"dirty": False},
        correctness_gate={
            "passed": True,
            "contract": "byte_exact",
            "default_promotion_eligible": True,
        },
        candidate_mode="chain",
    )
    assert rejected["measurement_valid"] is True
    assert rejected["status"] == "retain_baseline_reject_candidate_performance"
    assert rejected["selected_default"] == "fused"

    invalid = _promotion_decision(
        [{"tokens_exact": True, "candidate_wins": True}],
        provenance={"dirty": True},
        correctness_gate={
            "passed": True,
            "contract": "byte_exact",
            "default_promotion_eligible": True,
        },
        candidate_mode="chain",
    )
    assert invalid["measurement_valid"] is False
    assert invalid["status"] == "invalid_measurement"


def test_project_gate_win_does_not_imply_default_promotion() -> None:
    pending = _promotion_decision(
        [{"tokens_exact": True, "candidate_wins": True}],
        provenance={"dirty": False},
        correctness_gate={
            "passed": True,
            "contract": "project_kl_top1_non_exact",
            "default_promotion_eligible": False,
        },
        candidate_mode="chain_wave32_tree",
    )
    assert pending["measurement_valid"] is True
    assert pending["status"] == "candidate_wins_pending_correctness_contract"
    assert pending["selected_default"] == "unchanged"
