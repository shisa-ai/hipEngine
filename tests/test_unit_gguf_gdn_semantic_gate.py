from __future__ import annotations

import os

import numpy as np
import pytest

from scripts.gguf_gdn_semantic_gate import (
    _aggregate_gate,
    _compare_teacher_forced,
    _configure_gate_environment,
    build_parser,
)


def _step(token_id: int, logits: list[float]) -> dict[str, object]:
    return {
        "token_id": token_id,
        "logits": np.asarray(logits, dtype=np.float32),
    }


def test_parser_accepts_full_suite_plus_heldout_and_registered_candidate() -> None:
    args = build_parser().parse_args(
        [
            "--prompts",
            "benchmarks/prompts/mtpbench-code-general-ja.jsonl",
            "--prompts",
            "benchmarks/prompts/gdn-prefill-category-heldouts.jsonl",
            "--candidate-mode",
            "chain_wave32_tree",
            "--json",
            "/tmp/out.json",
        ]
    )

    assert len(args.prompts) == 2
    assert args.baseline_mode == "chain_lds32_direct"
    assert args.candidate_mode == "chain_wave32_tree"


def test_parser_accepts_explicit_k2_candidate() -> None:
    args = build_parser().parse_args(
        ["--candidate-mode", "chain_k2", "--json", "/tmp/out.json"]
    )

    assert args.candidate_mode == "chain_k2"


def test_gate_environment_disables_compact_scratch_aliasing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HIPENGINE_GGUF_VERIFY_GDN_SEMANTIC_GATE", raising=False)
    monkeypatch.delenv("HIPENGINE_GGUF_DECODE_REPACK", raising=False)

    _configure_gate_environment(decode_repack=True)

    assert os.environ["HIPENGINE_GGUF_VERIFY_GDN_SEMANTIC_GATE"] == "1"
    assert os.environ["HIPENGINE_GGUF_DECODE_REPACK"] == "1"


def test_teacher_forced_comparison_uses_aggregate_top1_contract() -> None:
    baseline = [_step(1, [1.0, 0.0]) for _ in range(100)]
    candidate = [_step(1, [0.99, 0.0]) for _ in range(100)]
    candidate[-1] = _step(0, [0.0, 1.0])

    accepted = _compare_teacher_forced(
        baseline,
        candidate,
        kl_threshold=1.0,
        top1_threshold=0.99,
    )

    assert accepted["passed"] is True
    assert accepted["top1_matches"] == 99
    assert accepted["top1_agreement"] == 0.99

    candidate[-2] = _step(0, [0.0, 1.0])
    rejected = _compare_teacher_forced(
        baseline,
        candidate,
        kl_threshold=1.0,
        top1_threshold=0.99,
    )
    assert rejected["passed"] is False
    assert rejected["top1_agreement"] == 0.98


def test_aggregate_gate_requires_kl_top1_and_decode_nonregression() -> None:
    prompts = [
        {
            "correctness": {
                "kl_passed": True,
                "transitions_total": 100,
                "top1_matches": 99,
                "kl_max": 0.01,
            },
            "decode_performance": {
                "baseline_median_ms": 100.0,
                "candidate_median_ms": 99.0,
            },
        },
        {
            "correctness": {
                "kl_passed": True,
                "transitions_total": 100,
                "top1_matches": 99,
                "kl_max": 0.02,
            },
            "decode_performance": {
                "baseline_median_ms": 101.0,
                "candidate_median_ms": 101.0,
            },
        },
    ]

    summary = _aggregate_gate(
        prompts,
        kl_threshold=0.05,
        top1_threshold=0.99,
    )

    assert summary["passed"] is True
    assert summary["top1_agreement"] == 0.99
    assert summary["decode_non_regressive"] is True

    prompts[1]["decode_performance"]["candidate_median_ms"] = 103.0
    rejected = _aggregate_gate(
        prompts,
        kl_threshold=0.05,
        top1_threshold=0.99,
    )
    assert rejected["passed"] is False
    assert rejected["decode_non_regressive"] is False
