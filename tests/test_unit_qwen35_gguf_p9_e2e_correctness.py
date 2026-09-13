from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/qwen35_gguf_p9_e2e_correctness.py"
FIXTURE = Path(__file__).resolve().parent / "fixtures/gguf/qwen36_35b_a3b_q4km_p9_e2e.json"

_spec = importlib.util.spec_from_file_location("qwen35_gguf_p9_e2e_correctness", SCRIPT)
assert _spec is not None and _spec.loader is not None
p9_gate = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = p9_gate
_spec.loader.exec_module(p9_gate)


def _run(ids: list[int], logits: np.ndarray, *, finite: bool = True) -> p9_gate.SequenceRun:
    return p9_gate.SequenceRun(
        generated_token_ids=list(ids),
        logits=np.asarray(logits, dtype=np.float32),
        final_token_id=int(ids[-1]),
        final_logit=float(np.asarray(logits, dtype=np.float32)[-1, ids[-1] % logits.shape[-1]]),
        finite_logits=finite,
        memory={"peak_allocated_bytes": 0},
    )


def test_p9_e2e_fixture_declares_decode_repack_and_gemv_env_contract() -> None:
    fixture = p9_gate.load_fixture(FIXTURE)

    assert fixture["model"]["architecture"] == "qwen35moe"
    assert fixture["model"]["path"].endswith("Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
    assert fixture["generation"]["prompt_length"] == 512
    assert fixture["generation"]["decode_tokens"] == 128
    assert fixture["generation"]["repeats"] == 3
    assert p9_gate.prompt_tokens_from_fixture(fixture) == [9707] * 512
    assert p9_gate.expected_env_from_mode(fixture["reference"]) == {
        "HIPENGINE_GGUF_WMMA_PREFILL": "0",
        "HIPENGINE_GGUF_GEMV_DECODE": "0",
    }
    assert p9_gate.expected_env_from_mode(fixture["candidate"]) == {
        "HIPENGINE_GGUF_WMMA_PREFILL": "1",
        "HIPENGINE_GGUF_GEMV_DECODE": "1",
        "HIPENGINE_GGUF_DECODE_REPACK": "1",
    }


def test_p9_e2e_metric_gate_accepts_small_logit_drift_and_deterministic_tail() -> None:
    reference = _run(
        [10, 11, 12, 13],
        np.array(
            [
                [6.0, 1.0, 0.0, -1.0],
                [0.0, 6.0, 1.0, -1.0],
                [0.0, 1.0, 6.0, -1.0],
                [0.0, 1.0, -1.0, 6.0],
            ],
            dtype=np.float32,
        ),
    )
    candidate_logits = reference.logits + np.float32(1.0e-3)
    candidates = [_run([10, 11, 12, 13], candidate_logits) for _ in range(3)]

    result = p9_gate.evaluate_candidate_repeats(
        reference=reference,
        candidates=candidates,
        kl_threshold=0.05,
        top1_threshold=0.90,
        deterministic_tail_required=True,
        finite_final_logits_required=True,
    )

    assert result["passed"]
    assert result["candidate_deterministic_tail"]
    assert result["candidate_deterministic_full_sequence"]
    assert result["worst_kl_mean"] <= 0.05
    assert result["worst_top1_agreement"] >= 0.90


def test_p9_e2e_metric_gate_fails_loudly_on_reduction_order_drift() -> None:
    reference = _run(
        [0, 1, 2, 3],
        np.array(
            [
                [8.0, 0.0, 0.0, 0.0],
                [0.0, 8.0, 0.0, 0.0],
                [0.0, 0.0, 8.0, 0.0],
                [0.0, 0.0, 0.0, 8.0],
            ],
            dtype=np.float32,
        ),
    )
    # Every row has the wrong top-1 and a large KL; this models the P8-style
    # reduction-order drift growing beyond the accepted threshold.
    bad = _run(
        [1, 2, 3, 0],
        np.array(
            [
                [0.0, 8.0, 0.0, 0.0],
                [0.0, 0.0, 8.0, 0.0],
                [0.0, 0.0, 0.0, 8.0],
                [8.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )

    result = p9_gate.evaluate_candidate_repeats(
        reference=reference,
        candidates=[bad, bad, bad],
        kl_threshold=0.05,
        top1_threshold=0.90,
        deterministic_tail_required=True,
        finite_final_logits_required=True,
    )

    assert not result["passed"]
    assert result["worst_top1_agreement"] == pytest.approx(0.0)
    assert result["worst_kl_mean"] > 0.05
    assert any("drift exceeds threshold" in error for error in result["errors"])


def test_p9_e2e_metric_gate_fails_when_candidate_tail_is_not_deterministic() -> None:
    logits = np.tile(np.array([[4.0, 0.0]], dtype=np.float32), (4, 1))
    reference = _run([0, 0, 0, 0], logits)
    candidates = [
        _run([0, 0, 0, 0], logits),
        _run([0, 0, 0, 1], logits),
        _run([0, 0, 0, 0], logits),
    ]

    result = p9_gate.evaluate_candidate_repeats(
        reference=reference,
        candidates=candidates,
        kl_threshold=0.05,
        top1_threshold=0.90,
        deterministic_tail_required=True,
        finite_final_logits_required=True,
    )

    assert not result["passed"]
    assert any("not deterministic" in error for error in result["errors"])
