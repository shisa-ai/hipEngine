from __future__ import annotations

import json

import pytest

from scripts.gguf_q4_t16_prefill_ab import (
    GateError,
    _classify_gate,
    _load_kernel_correctness_gate,
    _parse_contexts,
    _summarize_context,
)


def test_parse_contexts_requires_unique_positive_values() -> None:
    assert _parse_contexts("512,1024,4096") == (512, 1024, 4096)
    with pytest.raises(GateError, match="positive"):
        _parse_contexts("512,0")
    with pytest.raises(GateError, match="duplicates"):
        _parse_contexts("512,512")


def test_kernel_correctness_gate_requires_both_output_dtypes_exact(tmp_path) -> None:
    path = tmp_path / "kernel.json"
    path.write_text(
        json.dumps(
            {
                "kind": "hipengine_gguf_q4t16_shared_activation_candidate_replay_compact",
                "correctness": {
                    "tests_passed": 18,
                    "bf16_raw_bytes_exact": True,
                    "fp16_raw_bytes_exact": True,
                },
            }
        ),
        encoding="utf-8",
    )
    gate = _load_kernel_correctness_gate(path)
    assert gate["passed"] is True
    assert gate["tests_passed"] == 18
    assert len(gate["sha256"]) == 64

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["correctness"]["fp16_raw_bytes_exact"] = False
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(GateError, match="byte-exact"):
        _load_kernel_correctness_gate(path)


def _leg(prefill_ms: float, decode_ms: float, token_ids: list[int]) -> dict[str, object]:
    return {
        "prefill_wall_ms": prefill_ms,
        "decode_wall_ms": decode_ms,
        "token_ids": token_ids,
    }


def test_context_summary_keeps_prefill_and_decode_separate() -> None:
    rows = [
        {
            "repetition": 0,
            "order": ["baseline", "shared_x"],
            "modes": {
                "baseline": _leg(100.0, 200.0, [7, 8, 9]),
                "shared_x": _leg(90.0, 199.0, [7, 8, 9]),
            },
        },
        {
            "repetition": 1,
            "order": ["shared_x", "baseline"],
            "modes": {
                "shared_x": _leg(92.0, 201.0, [7, 8, 9]),
                "baseline": _leg(104.0, 202.0, [7, 8, 9]),
            },
        },
    ]

    summary = _summarize_context(rows, context_tokens=512)

    assert summary["prefill"]["statistics"]["baseline"]["median_ms"] == 102.0
    assert summary["prefill"]["statistics"]["shared_x"]["median_ms"] == 91.0
    assert summary["prefill"]["paired_candidate_minus_baseline_ms"] == [-10.0, -12.0]
    assert summary["prefill"]["candidate_wins"] is True
    assert summary["decode"]["statistics"]["baseline"]["median_ms"] == 201.0
    assert summary["decode"]["statistics"]["shared_x"]["median_ms"] == 200.0
    assert summary["trajectories_exact"] is True
    assert summary["reference_token_ids"] == [7, 8, 9]


def test_classification_requires_clean_exact_wins_and_decode_nonregression() -> None:
    accepted = _classify_gate(
        [
            {
                "correctness": {"passed": True, "logits_byte_exact": True},
                "prefill": {"candidate_wins": True},
                "decode": {
                    "statistics": {
                        "baseline": {"median_ms": 200.0},
                        "shared_x": {"median_ms": 199.0},
                    }
                },
                "trajectories_exact": True,
            },
            {
                "correctness": {"passed": True, "logits_byte_exact": True},
                "prefill": {"candidate_wins": True},
                "decode": {
                    "statistics": {
                        "baseline": {"median_ms": 210.0},
                        "shared_x": {"median_ms": 211.0},
                    }
                },
                "trajectories_exact": True,
            },
        ],
        provenance={"dirty": False},
    )
    assert accepted["status"] == "promote_shared_x"
    assert accepted["decode_non_regressive"] is True
    assert accepted["selected_default"] == "shared_x"

    decode_rejected = _classify_gate(
        [
            {
                "correctness": {"passed": True, "logits_byte_exact": True},
                "prefill": {"candidate_wins": True},
                "decode": {
                    "statistics": {
                        "baseline": {"median_ms": 200.0},
                        "shared_x": {"median_ms": 200.01},
                    }
                },
                "trajectories_exact": True,
            }
        ],
        provenance={"dirty": False},
    )
    assert decode_rejected["status"] == "reject_decode_regression"
    assert decode_rejected["selected_default"] == "baseline"

    invalid = _classify_gate(
        [
            {
                "correctness": {"passed": True, "logits_byte_exact": True},
                "prefill": {"candidate_wins": True},
                "decode": {
                    "statistics": {
                        "baseline": {"median_ms": 200.0},
                        "shared_x": {"median_ms": 199.0},
                    }
                },
                "trajectories_exact": True,
            }
        ],
        provenance={"dirty": True},
    )
    assert invalid["status"] == "invalid_measurement"
    assert invalid["selected_default"] == "unchanged"
