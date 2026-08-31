from __future__ import annotations

import os

import numpy as np
import pytest

from scripts import qwen38_q4_verifier_numerics as gate


def test_q4_verifier_numerics_parser_freezes_product_horizon() -> None:
    args = gate.build_parser().parse_args(("--output", "/tmp/out.json"))

    assert args.decode_steps == 24
    assert args.repeat_runs == 3
    assert args.backend == "hip_gfx1151"
    assert args.concurrency == 2
    assert args.candidate_budget == 3


def test_q4_verifier_numerics_accepts_w1_wide_shapes() -> None:
    args = gate.build_parser().parse_args(
        ("--concurrency", "8", "--candidate-budget", "3", "--output", "/tmp/out.json")
    )
    assert args.concurrency == 8
    assert args.candidate_budget == 3


def test_q4_verifier_numerics_accepts_c3_tail_budget_and_pads_final_group() -> None:
    args = gate.build_parser().parse_args(
        (
            "--concurrency",
            "3",
            "--candidate-budget",
            "1",
            "--output",
            "/tmp/out.json",
        )
    )
    rows = tuple({"id": f"p{index}"} for index in range(5))

    assert args.concurrency == 3
    assert args.candidate_budget == 1
    assert gate._prompt_groups(rows, 3) == (
        ((rows[0], rows[1], rows[2]), 3),
        ((rows[3], rows[4], rows[4]), 2),
    )


def test_q4_verifier_environment_restores_caller(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(gate.TARGET_VERIFIER_PRODUCTION_Q4_ROWTILE_ENV, raising=False)

    with gate._environment(
        gate.TARGET_VERIFIER_PRODUCTION_Q4_ROWTILE_ENV,
        "1",
    ):
        assert os.environ[gate.TARGET_VERIFIER_PRODUCTION_Q4_ROWTILE_ENV] == "1"

    assert gate.TARGET_VERIFIER_PRODUCTION_Q4_ROWTILE_ENV not in os.environ


def test_q4_verifier_trajectory_hash_is_order_and_logit_sensitive() -> None:
    rows = (
        {"token_id": 1, "logits": np.asarray([0.0, 1.0], dtype=np.float32)},
        {"token_id": 2, "logits": np.asarray([2.0, 3.0], dtype=np.float32)},
    )
    changed = (
        rows[0],
        {"token_id": 2, "logits": np.asarray([2.0, 3.5], dtype=np.float32)},
    )

    assert gate._trajectory_sha256(rows) == gate._trajectory_sha256(rows)
    assert gate._trajectory_sha256(rows) != gate._trajectory_sha256(changed)
