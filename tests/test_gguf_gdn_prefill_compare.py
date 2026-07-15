from __future__ import annotations

import numpy as np
import pytest

from scripts.gguf_gdn_prefill_compare import (
    GREETING_PROMPT_IDS,
    CompareError,
    _array_comparison,
    _build_prompt_ids,
    build_parser,
    _classify,
    _first_layer_part_divergence,
    _logit_gate,
)


@pytest.mark.parametrize(
    "mode",
    ("chain_wave32_tree", "chain_lds32", "chain_lds64", "chain_lds32_direct"),
)
def test_parser_accepts_named_gdn_candidate_mode(mode: str) -> None:
    args = build_parser().parse_args(
        ["--candidate-mode", mode, "--json", "/tmp/out.json"]
    )
    assert args.candidate_mode == mode


def test_array_comparison_reports_exact_and_numeric_drift() -> None:
    left = np.array([1.0, -2.0, 3.0], dtype=np.float32)
    exact = _array_comparison(left, left.copy())
    assert exact["exact"] is True
    assert exact["mismatch_elements"] == 0
    assert exact["max_abs"] == 0.0

    right = left.copy()
    right[1] = np.nextafter(right[1], np.float32(0.0))
    drift = _array_comparison(left, right)
    assert drift["exact"] is False
    assert drift["mismatch_elements"] == 1
    assert drift["max_abs"] > 0.0
    assert drift["left"]["blake2b_128"] != drift["right"]["blake2b_128"]


def test_logit_gate_reports_project_thresholds() -> None:
    reference = np.asarray([[3.0, 1.0, -2.0]], dtype=np.float32)
    gate = _logit_gate(reference, reference.copy())
    assert gate == {
        "kl_mean": 0.0,
        "kl_max": 0.0,
        "top1_agreement": 1.0,
        "kl_threshold": 0.05,
        "top1_threshold": 0.9,
        "passed": True,
    }


def test_array_comparison_rejects_shape_mismatch() -> None:
    with pytest.raises(CompareError, match="shape mismatch"):
        _array_comparison(
            np.zeros((2,), dtype=np.float32),
            np.zeros((1, 2), dtype=np.float32),
        )


def test_first_layer_part_divergence_is_layer_major() -> None:
    comparisons = {
        2: {"conv": {"exact": False}, "recurrent": {"exact": False}},
        0: {"conv": {"exact": True}, "recurrent": {"exact": False}},
        1: {"conv": {"exact": False}, "recurrent": {"exact": True}},
    }
    assert _first_layer_part_divergence(comparisons) == {
        "layer": 0,
        "part": "recurrent",
    }


def test_build_prompt_ids_supports_greeting_and_repeated() -> None:
    assert _build_prompt_ids(kind="greeting", token_id=0, length=0) == list(
        GREETING_PROMPT_IDS
    )
    assert _build_prompt_ids(kind="repeated", token_id=9707, length=4) == [9707] * 4
    with pytest.raises(CompareError, match="positive"):
        _build_prompt_ids(kind="repeated", token_id=9707, length=0)


def test_classification_distinguishes_visible_token_from_state_parity() -> None:
    failed = _classify(
        fused_token=9419,
        chain_token=9419,
        actual_hidden_exact=False,
        actual_first_state={"layer": 0, "part": "recurrent"},
        bisect_first_hidden={"layer": 1, "part": "layer_output"},
        bisect_first_state={"layer": 0, "part": "recurrent"},
    )
    assert failed["passed"] is False
    assert failed["visible_token_exact"] is True
    assert failed["status"] == "visible_token_match_state_divergence"

    passed = _classify(
        fused_token=9707,
        chain_token=9707,
        actual_hidden_exact=True,
        actual_first_state=None,
        bisect_first_hidden=None,
        bisect_first_state=None,
    )
    assert passed["passed"] is True
    assert passed["status"] == "fused_chain_exact"
