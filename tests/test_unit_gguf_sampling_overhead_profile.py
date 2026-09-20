from collections import defaultdict
from contextlib import ExitStack
from types import SimpleNamespace

import pytest

from scripts.gguf_sampling_overhead_profile import Trace, summarize, trajectory_checks


def test_profile_excludes_capture_transitions_but_retains_route_counts():
    transitions = [
        {"wall_s": 20, "seconds": {"graph_capture": 19},
         "calls": {"graph_capture": 1}, "d2h_bytes": {}},
        {"wall_s": 10, "seconds": {}, "calls": {}, "d2h_bytes": {}},
        {"wall_s": 0.1, "seconds": {"host_selection": 0.02},
         "calls": {"host_selection": 1}, "d2h_bytes": {"model_d2h": 1000}},
        {"wall_s": 0.3, "seconds": {"host_selection": 0.04},
         "calls": {"host_selection": 1}, "d2h_bytes": {"model_d2h": 1000}},
    ]
    result = summarize(transitions)
    assert result["steady_transitions"] == 2
    assert result["steady_wall_s"] == pytest.approx(0.4)
    assert result["median_transition_ms"] == pytest.approx(200)
    assert result["steady_totals"]["seconds"]["host_selection"] == pytest.approx(0.06)
    assert result["steady_totals"]["d2h_bytes"]["model_d2h"] == 2000
    assert result["all_calls"]["graph_capture"] == 1
    with pytest.raises(ValueError, match="three"):
        summarize(transitions[:2])


def test_profile_wrapper_preserves_failure_and_restores_original():
    def failing():
        raise LookupError("original failure")

    target = SimpleNamespace(call=failing)
    trace = Trace()
    # Use a real transition-shaped counter and ensure the wrapper observes it
    # even when the measured method fails.
    trace.current = {"seconds": defaultdict(float), "calls": defaultdict(int)}
    with ExitStack() as stack:
        trace.wrap(stack, target, "call", "test")
        with pytest.raises(LookupError, match="original failure"):
            target.call()
        assert trace.current["calls"]["test"] == 1
    assert target.call is failing


def test_trajectory_checks_detect_repeat_drift_without_equating_sampler_rngs():
    def row(arm, ids):
        return {"prompt_id": "test", "prompt_tokens": 4,
                "arm": arm, "generated_token_ids": ids}

    rows = [row("greedy_default", [1, 2]), row("greedy_eager", [1, 2]),
            row("host_sampled", [3, 4]), row("native_sampled", [5, 6]),
            row("native_sampled", [5, 6])]
    assert trajectory_checks(rows)["fixed_seed_repeats_exact"] is True
    rows[-1]["generated_token_ids"] = [5, 7]
    assert trajectory_checks(rows)["fixed_seed_repeats_exact"] is False
    assert trajectory_checks(rows)["greedy_default_eager_exact"] is True
    rows[1]["generated_token_ids"] = [1, 8]
    assert trajectory_checks(rows)["greedy_default_eager_exact"] is False
