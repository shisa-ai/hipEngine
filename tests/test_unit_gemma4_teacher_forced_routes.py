"""CPU regression coverage for observed evaluator route evidence."""
import pytest

from scripts.gemma4_teacher_forced_gate import (
    observe_decode_routes, require_candidate_route, route_summary,
)


@pytest.mark.parametrize("forced", [None, 2, 4])
def test_split_candidate_cannot_pass_without_a_split_launch(forced):
    verdict = {"passed": True, "failed": []}
    require_candidate_route(verdict, route_summary([
        {"keys": 777, "head_dim": 256, "selection": 1},
    ]), forced)
    assert not verdict["passed"]
    assert verdict["failed"] == ["split_not_exercised"]


def test_strict_self_gate_and_observed_split_are_distinguished():
    for selection, forced in [(1, 1), (2, None)]:
        verdict = {"passed": True, "failed": []}
        require_candidate_route(verdict, route_summary([
            {"keys": 1057, "head_dim": 256, "selection": selection},
        ]), forced)
        assert verdict["passed"]
    verdict = {"passed": True, "failed": []}
    require_candidate_route(verdict, route_summary([]), 1)
    assert not verdict["passed"]


def test_observer_records_actual_selection_and_restores_on_failure(monkeypatch):
    from hipengine.kernels.hip_gfx1100.gemma4 import gemma4_attention as attention

    def launch(*args, **kwargs):
        return None

    monkeypatch.setattr(attention, "_launch_prefill", launch)
    monkeypatch.setattr(attention, "decode_selection", lambda library: 1)
    routes = []
    with pytest.raises(RuntimeError):
        with observe_decode_routes(routes):
            attention._launch_prefill(tokens=1, keys=1057, head_dim=256)
            attention._launch_prefill(tokens=8, keys=1057, head_dim=256)
            raise RuntimeError("failed forward")
    assert attention._launch_prefill is launch
    assert routes == [{"keys": 1057, "head_dim": 256, "selection": 1}]
