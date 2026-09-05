from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest

import hipengine.runtime.qwen35_gguf_runner as runner_module
from hipengine.runtime.qwen35_gguf_runner import (
    _gguf_c8_q5_raw_mmq_enabled,
    _gguf_c8_q5_source_mmq_enabled,
)
from scripts import qwen38_q4_verifier_numerics as gate


def test_q4_verifier_numerics_parser_freezes_product_horizon() -> None:
    args = gate.build_parser().parse_args(("--output", "/tmp/out.json"))

    assert args.decode_steps == 24
    assert args.repeat_runs == 3
    assert args.backend == "hip_gfx1151"
    assert args.concurrency == 2
    assert args.candidate_budget == 3
    assert args.candidate_q5_raw_mmq is False
    assert args.candidate_q5_source_mmq is False


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


def test_q4_verifier_numerics_accepts_c8_raw_q5_candidate() -> None:
    args = gate.build_parser().parse_args(
        (
            "--backend",
            "hip_gfx1100",
            "--concurrency",
            "8",
            "--candidate-q5-raw-mmq",
            "--output",
            "/tmp/out.json",
        )
    )

    assert args.backend == "hip_gfx1100"
    assert args.concurrency == 8
    assert args.candidate_q5_raw_mmq is True
    assert args.candidate_q5_source_mmq is False


def test_q4_verifier_numerics_accepts_c8_source_q5_candidate() -> None:
    args = gate.build_parser().parse_args(
        (
            "--backend",
            "hip_gfx1100",
            "--concurrency",
            "8",
            "--candidate-q5-source-mmq",
            "--output",
            "/tmp/out.json",
        )
    )

    assert args.candidate_q5_raw_mmq is False
    assert args.candidate_q5_source_mmq is True


def test_c8_raw_q5_mmq_is_gfx1100_default_with_opt_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(gate.Q5_RAW_MMQ_ENV, raising=False)
    assert _gguf_c8_q5_raw_mmq_enabled("hip_gfx1100", request_count=8) is True
    assert _gguf_c8_q5_raw_mmq_enabled("hip_gfx1100", request_count=7) is False
    assert _gguf_c8_q5_raw_mmq_enabled("hip_gfx1151", request_count=8) is False

    monkeypatch.setenv(gate.Q5_RAW_MMQ_ENV, "0")
    assert _gguf_c8_q5_raw_mmq_enabled("hip_gfx1100", request_count=8) is False


def test_c8_source_q5_mmq_is_exact_width_default_with_rollback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HIPENGINE_GGUF_C8_Q5_RAW_MMQ", raising=False)
    monkeypatch.delenv("HIPENGINE_GGUF_C8_Q5_SOURCE_MMQ", raising=False)
    assert _gguf_c8_q5_source_mmq_enabled("hip_gfx1100", request_count=8) is True
    assert _gguf_c8_q5_source_mmq_enabled("hip_gfx1100", request_count=7) is False
    assert _gguf_c8_q5_source_mmq_enabled("hip_gfx1151", request_count=8) is False

    monkeypatch.setenv("HIPENGINE_GGUF_C8_Q5_SOURCE_MMQ", "0")
    assert _gguf_c8_q5_source_mmq_enabled("hip_gfx1100", request_count=8) is False

    monkeypatch.setenv("HIPENGINE_GGUF_C8_Q5_SOURCE_MMQ", "1")
    monkeypatch.setenv("HIPENGINE_GGUF_C8_Q5_RAW_MMQ", "0")
    assert _gguf_c8_q5_source_mmq_enabled("hip_gfx1100", request_count=8) is False


def test_q4_verifier_session_environment_lives_until_stack_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeSession:
        def __init__(self, *args, **kwargs) -> None:
            shared_runner = kwargs.get("shared_runner")
            self.runner = shared_runner or SimpleNamespace(
                fp16_recurrent_state=os.environ[gate.FP16_STATE_ENV] == "1",
            )
            self.runtime = object()
            self.target_verifier_production_q4_rowtile = (
                os.environ[gate.TARGET_VERIFIER_PRODUCTION_Q4_ROWTILE_ENV] == "1"
            )

        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            return None

    monkeypatch.setattr(runner_module, "Qwen35GGUFResidentSession", FakeSession)
    for name in (
        gate.FP16_STATE_ENV,
        gate.TARGET_VERIFIER_PRODUCTION_Q4_ROWTILE_ENV,
        gate.Q5_RAW_MMQ_ENV,
        gate.Q5_SOURCE_MMQ_ENV,
    ):
        monkeypatch.delenv(name, raising=False)
    args = gate.build_parser().parse_args(
        ("--backend", "hip_gfx1100", "--concurrency", "2", "--output", "/tmp/out")
    )

    stack, sessions = gate._make_sessions(
        args,
        fp16_state=True,
        q4_rowtile=True,
        q5_raw_mmq=True,
        q5_source_mmq=True,
        max_sequence_length=128,
    )
    assert len(sessions) == 2
    assert os.environ[gate.FP16_STATE_ENV] == "1"
    assert os.environ[gate.TARGET_VERIFIER_PRODUCTION_Q4_ROWTILE_ENV] == "1"
    assert os.environ[gate.Q5_RAW_MMQ_ENV] == "1"
    assert os.environ[gate.Q5_SOURCE_MMQ_ENV] == "1"

    stack.close()
    assert gate.FP16_STATE_ENV not in os.environ
    assert gate.TARGET_VERIFIER_PRODUCTION_Q4_ROWTILE_ENV not in os.environ
    assert gate.Q5_RAW_MMQ_ENV not in os.environ
    assert gate.Q5_SOURCE_MMQ_ENV not in os.environ


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
