"""CPU checks for public PARO depth-benchmark accounting and pooling."""
from copy import deepcopy

import pytest

from scripts.paro_mtp_public_bench import pad_prompt, pooled, strict_mismatches, validate_arm


def test_padding_keeps_task_suffix_and_exact_context():
    assert pad_prompt([7, 8], [1, 2], 7) == [1, 2, 1, 2, 1, 7, 8]
    assert pad_prompt([7, 8], [], 2) == [7, 8]
    with pytest.raises(ValueError):
        pad_prompt([7, 8], [1], 1)
    with pytest.raises(ValueError):
        pad_prompt([7], [], 2)


def _arm():
    return {"ids": [1, 2, 3, 4, 5], "spec": {"requested_budget": 3,
        "candidate_budget": 3, "mtp_output_tokens": 4, "ar_output_tokens_in_cycles": 1,
        "recoverable_failures": 0}, "route": {"specdec2_mtp2_used": True,
        "specdec2_mtp2_candidate_counts": [3]}}


def test_route_check_requires_real_depth_and_balanced_counts():
    validate_arm("b3", _arm(), 5)
    for key, value in [("requested_budget", 1), ("candidate_budget", 1),
                       ("mtp_output_tokens", 3), ("recoverable_failures", 1)]:
        arm = deepcopy(_arm())
        arm["spec"][key] = value
        with pytest.raises(AssertionError):
            validate_arm("b3", arm, 5)
    arm = _arm()
    arm["route"]["specdec2_mtp2_candidate_counts"] = [1, 1]
    with pytest.raises(AssertionError, match="did not execute"):
        validate_arm("b3", arm, 5)
    with pytest.raises(AssertionError, match="AR control"):
        validate_arm("ar", _arm(), 5)


def test_pool_uses_elapsed_weighting_not_mean_of_rates():
    rows = [{"arms": {"ar": {"ids": [1, 2], "wall_s": 1.}}},
            {"arms": {"ar": {"ids": [1, 2], "wall_s": 3.}}}]
    assert pooled(rows)["ar"]["whole_request_tok_s"] == 1.
    assert pooled(rows)["ar"]["ids_equal_ar"] == 2


def test_strict_diagnostic_preserves_each_first_divergence():
    row = {"id": "prompt", "max_tokens": 3, "run": 0, "arms": {
        "ar": {"ids": [1, 2, 3]}, "b1": {"ids": [1, 4, 3]},
        "b2": {"ids": [1, 2, 3]}, "b3": {"ids": [5, 2, 3]}}}
    assert strict_mismatches(row) == [
        {"id": "prompt", "max_tokens": 3, "run": 0, "arm": "b1", "first_mismatch": 1},
        {"id": "prompt", "max_tokens": 3, "run": 0, "arm": "b3", "first_mismatch": 0}]
