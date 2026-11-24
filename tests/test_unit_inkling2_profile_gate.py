from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.benchmark.execution_profiles import compare_control_records
from scripts import inkling2_profile_gate as gate


def _requests():
    return [
        {"id": "short", "category": "a", "prompt_tokens": list(range(100, 131)), "decode_steps": 4,
         "block_width": 64, "first_step": 0},
        {"id": "long", "category": "b", "prompt_tokens": list(range(200, 365)), "decode_steps": 3,
         "block_width": 64, "first_step": 4},
    ]


def _trajectory(request, chain, *, fed=None, buckets=None, contexts=None, graph=False):
    """A synthetic session record: prompt blocks of 64, then the forced decode calls.

    ``graph`` makes the first one-token call a capture and the rest replays, the
    way a graph candidate that engaged would report itself.
    """

    prompt = request["prompt_tokens"]
    cap = min(len(prompt) + request["decode_steps"] + 1, 1024)
    calls = []
    captured = False
    for start in range(0, len(prompt), 64):
        block = prompt[start : start + 64]
        bucket = "c1_eager"
        if len(block) == 1 and graph:
            bucket, captured = "c1_graph_capture", True
        calls.append(gate.Call("PREFILL", tuple(block), ((0, start + len(block)),), True, bucket, "r", True))
    for index in range(request["decode_steps"] - 1):
        token = chain[index] if fed is None else fed[index]
        position = len(prompt) + index + 1 if contexts is None else contexts[index]
        bucket = "c1_eager"
        if graph:
            bucket = "c1_graph_replay" if captured else "c1_graph_capture"
            captured = True
        if buckets is not None:
            bucket = buckets[index]
        calls.append(gate.Call("DECODE", (token,), ((0, position),), True, bucket, "r", True))
    rows = request["decode_steps"]
    return gate.Trajectory(request["id"], np.zeros((rows, 4), np.float32), (0,) * rows, tuple(chain),
                           tuple(calls), {"finite": True, "min_bank_cap": cap, "state_sha256": "s"})


def _compare(requests, teacher, candidate, trajectories, **overrides):
    hashes = {
        request["id"]: ["r"] * (-(-len(request["prompt_tokens"]) // 64) + request["decode_steps"] - 1)
        for request in requests
    }
    hashes.update(overrides.pop("teacher_route_hashes", {}))
    expected = gate.expected_controls("sc", requests, teacher, candidate=candidate, window=1024,
                                      route_top_k=8, teacher_route_hashes=hashes, **overrides)
    actual = []
    for request, trajectory in zip(requests, trajectories, strict=True):
        actual.extend(gate.actual_controls("sc", request, trajectory, route_top_k=8))
    return expected, actual, compare_control_records(expected, actual, **overrides.get("compare", {}))


def test_section_211_verdict_passes_at_rounding_scale_and_bounds_each_bar():
    rows = 400
    kl = np.full(rows, 1e-5)
    kl[-12:] = 0.2  # 3% of rows over the damage line: allowed globally
    top1 = np.ones(rows, dtype=bool)
    top1[-12:] = False
    categories = ["x"] * 300 + ["y"] * 100
    verdict = gate.section_211_verdict(kl, top1, categories)
    assert verdict["global"]["passed"] and verdict["global"]["flip_damage_rate"] == pytest.approx(0.03)
    assert verdict["global"]["top1_agreement"] == pytest.approx(0.97)
    # The same damaged rows all sit in the 100-row category: its p90, its 12% damage rate and
    # its 88% top-1 are all over the category bars, so the packet still fails.
    assert not verdict["passed"] and verdict["failed_categories"] == ["y"]
    assert verdict["by_category"]["y"]["failed"] == ["kl_p90", "flip_damage_rate", "top1_rate"]
    assert verdict["by_category"]["x"]["passed"]
    # Spread over two 200-row categories the same rows stay inside every category bar.
    spread = gate.section_211_verdict(kl, top1, ["x"] * 200 + ["y"] * 200)
    assert spread["global"]["flip_damage_rate"] == pytest.approx(0.03) and spread["passed"]


@pytest.mark.parametrize(
    "kl, top1, failed",
    [
        (np.full(100, 2e-3), np.ones(100, dtype=bool), ["kl_median"]),
        (np.r_[np.full(89, 1e-6), np.full(11, 3e-2)], np.ones(100, dtype=bool), ["kl_p90"]),
        (np.r_[np.full(95, 1e-6), np.full(5, 0.2)], np.ones(100, dtype=bool), ["flip_damage_rate"]),
        (np.full(100, 1e-6), np.r_[np.ones(95, dtype=bool), np.zeros(5, dtype=bool)], ["top1_rate"]),
    ],
)
def test_section_211_verdict_fails_the_bar_it_exceeds(kl, top1, failed):
    verdict = gate.section_211_verdict(kl, top1, ["c"] * 100)
    assert verdict["global"]["failed"] == failed and not verdict["passed"]


def test_section_211_verdict_rejects_misaligned_or_nonfinite_input():
    with pytest.raises(gate.GateError, match="aligned"):
        gate.section_211_verdict([1e-6, 1e-6], [True], ["a", "b"])
    with pytest.raises(gate.GateError, match="finite"):
        gate.section_211_verdict([np.nan, 1e-6], [True, True], ["a", "b"])


def test_shipped_suite_is_well_formed_and_covers_both_sides_of_the_window():
    suite = gate.load_suite(gate.DEFAULT_SUITE)
    assert len(suite) == 25
    assert len({row["id"] for row in suite}) == len(suite)
    assert {row["template"] for row in suite} == {"chat", "raw"}
    assert sum(row["decode_steps"] for row in suite) == 672
    assert all(gate.render_prompt(row).strip() for row in suite)
    requests = [
        {**row, "prompt_tokens": [7] * length}
        for row, length in zip(suite, [40] * 22 + [1162, 2289, 1004], strict=True)
    ]
    coverage = gate.validate_coverage(requests, window=1024)
    assert coverage == {"window_tokens": 1024, "saturated_window_prompts": 2,
                        "window_crossing_prompts": 1, "prompts": 25}
    requests[-1]["prompt_tokens"] = [7] * 970  # 970 + 47 rows never reach the window
    with pytest.raises(gate.GateError, match="do not cross"):
        gate.validate_coverage(requests, window=1024)
    requests[-2]["prompt_tokens"] = [7] * 1000
    with pytest.raises(gate.GateError, match="needs at least 1024"):
        gate.validate_coverage(requests, window=1024)


def test_load_suite_rejects_duplicate_ids_and_short_row_counts(tmp_path):
    path = tmp_path / "suite.jsonl"
    path.write_text('{"id":"a","category":"c","prompt":"x","decode_steps":3}\n'
                    '{"id":"a","category":"c","prompt":"y","decode_steps":3}\n', encoding="utf-8")
    with pytest.raises(gate.GateError, match="duplicate prompt id"):
        gate.load_suite(path)
    path.write_text('{"id":"a","category":"c","prompt":"x","decode_steps":1}\n', encoding="utf-8")
    with pytest.raises(gate.GateError, match="at least 2"):
        gate.load_suite(path)
    path.write_text('{"id":"a","category":"c","prompt":"x","decode_steps":3}\n', encoding="utf-8")
    assert [row["id"] for row in gate.load_suite(path, limit=5)] == ["a"]
    with pytest.raises(gate.GateError, match="empty"):
        gate.load_suite(path, limit=0)


def test_controls_pass_for_a_faithful_eager_run_and_match_the_schedule():
    requests = _requests()
    teacher = {"short": (1, 2, 3), "long": (4, 5)}
    expected, actual, result = _compare(requests, teacher, gate.CANDIDATES["fast"],
                                        [_trajectory(r, teacher[r["id"]]) for r in requests])
    assert result["passed"] and not result["mismatches"]
    assert len(expected) == 9 and result["expected_scenario"]["request_count"] == 2
    compact = gate._capped(dict(result))
    assert compact["expected_scenario"] == {
        "records": 9, "steps": 9, "request_count": 2, "width_transition_count": 4,
        "compaction_count": 0, "page_boundary_count": 0, "work_classes": ["DECODE", "PREFILL"],
        "ragged_step_count": 0, "width_min_max": [1, 64]}
    assert compact["mismatch_count"] == 0 and compact["passed"]
    first_decode = actual[1]
    assert (first_decode.input_token_id, first_decode.position, first_decode.context_length,
            first_decode.kv_live_count, first_decode.graph_bucket) == (1, 31, 32, 32, "c1_eager")
    assert actual[6].kv_live_count == 165 and actual[6].physical_width == 37


def test_controls_catch_a_candidate_that_leaves_the_teachers_chain():
    requests = _requests()
    teacher = {"short": (1, 2, 3), "long": (4, 5)}
    drifted = _trajectory(requests[0], teacher["short"], fed=(1, 9, 3))
    _expected, _actual, result = _compare(requests, teacher, gate.CANDIDATES["fast"],
                                          [drifted, _trajectory(requests[1], teacher["long"])])
    fields = {(tuple(m["key"]), m["field"], m["expected"], m["actual"]) for m in result["mismatches"]}
    assert fields == {(("sc", 2, "DECODE", "short"), "input_token_id", 2, 9)}


def test_controls_catch_out_of_order_positions_and_wrong_graph_route():
    requests = _requests()
    teacher = {"short": (1, 2, 3), "long": (4, 5)}
    skipped = _trajectory(requests[0], teacher["short"], contexts=(40, 41, 42))
    _expected, _actual, result = _compare(requests, teacher, gate.CANDIDATES["fast"], [skipped, _trajectory(requests[1], teacher["long"])])
    assert {m["field"] for m in result["mismatches"]} == {
        "position", "context_length", "transaction_id", "kv_live_count", "kv_token_position",
        "kv_append_ordinal", "state_update_ordinal", "kv_evict",  # position 39 is past the 36-slot ring
    }
    # A graph candidate must capture on its first one-token call and replay afterwards.
    graph = [_trajectory(r, teacher[r["id"]], graph=True) for r in requests]
    _expected, _actual, result = _compare(requests, teacher, gate.CANDIDATES["fast_graph"], graph)
    assert result["passed"]
    graph[0] = _trajectory(requests[0], teacher["short"])
    _expected, _actual, result = _compare(requests, teacher, gate.CANDIDATES["fast_graph"], graph)
    assert {m["field"] for m in result["mismatches"]} == {"graph_bucket"}
    assert {m["actual"] for m in result["mismatches"]} == {"c1_eager"}


def test_controls_flag_a_single_token_prompt_block_that_runs_the_graph():
    requests = _requests()[:1]
    requests[0]["prompt_tokens"] = [5] * 65
    teacher = {"short": (1, 2, 3)}
    expected = gate.expected_controls("sc", requests, teacher, candidate=gate.CANDIDATES["fast_graph"],
                                      window=1024, route_top_k=8)
    assert [r.graph_bucket for r in expected] == ["prefill", "c1_graph_capture",
                                                   "c1_graph_replay", "c1_graph_replay",
                                                   "c1_graph_replay"]


def test_route_field_is_diagnostic_and_uses_the_teachers_routing():
    requests = _requests()
    teacher = {"short": (1, 2, 3), "long": (4, 5)}
    hashes = {"short": ["a"] * 4, "long": ["b"] * 5}
    expected, actual, _ = _compare(requests, teacher, gate.CANDIDATES["fast_resident"],
                                   [_trajectory(r, teacher[r["id"]]) for r in requests],
                                   teacher_route_hashes=hashes)
    assert all(record.route_decision_hash == "r" for record in actual)
    strict = compare_control_records(expected, actual)
    candidate = compare_control_records(expected, actual, diagnostic_fields=("route_decision_hash",))
    assert {m["field"] for m in strict["mismatches"]} == {"route_decision_hash"}
    assert candidate["passed"] and candidate["diagnostic_mismatches"]


def test_actual_controls_reject_a_session_that_made_the_wrong_number_of_calls():
    requests = _requests()[:1]
    teacher = {"short": (1, 2, 3)}
    trajectory = _trajectory(requests[0], teacher["short"])
    trimmed = gate.Trajectory(trajectory.prompt_id, trajectory.logits, trajectory.argmax, trajectory.fed,
                              trajectory.calls[:-1], trajectory.state)
    with pytest.raises(gate.GateError, match="calls for 4 scheduled"):
        gate.actual_controls("sc", requests[0], trimmed, route_top_k=8)
    split = gate.Call("DECODE", (1,), ((0, 32), (1, 31)), True, "c1_eager", "r", True)
    disagreeing = gate.Trajectory(trajectory.prompt_id, trajectory.logits, trajectory.argmax, trajectory.fed,
                                  trajectory.calls[:1] + (split,) + trajectory.calls[2:], trajectory.state)
    with pytest.raises(gate.GateError, match="attention visits disagree"):
        gate.actual_controls("sc", requests[0], disagreeing, route_top_k=8)


def test_task_verdict_is_paired_and_fails_on_any_regression():
    def row(correct, ids):
        return {"prompt_token_ids_sha256": "d", "generated_token_ids": ids, "correct": correct}

    strict = {"a": row(True, [1]), "b": row(False, [2])}
    candidate = {"a": row(True, [1]), "b": row(True, [3])}
    verdict = gate.task_verdict(strict, candidate)
    assert verdict["passed"] and verdict["improved_cases"] == ["b"]
    assert verdict["generated_ids_identical_cases"] == 1
    verdict = gate.task_verdict(strict, {"a": row(False, [9]), "b": row(False, [2])})
    assert not verdict["passed"] and verdict["regressed_cases"] == ["a"]
    with pytest.raises(gate.GateError, match="inputs differed"):
        gate.task_verdict(strict, {"a": row(True, [1]), "b": {**candidate["b"], "prompt_token_ids_sha256": "e"}})
    with pytest.raises(gate.GateError, match="case sets differ"):
        gate.task_verdict(strict, {"a": candidate["a"]})


def test_repeat_verdict_requires_bit_stable_logits_calls_and_state():
    requests = _requests()[:1]
    teacher = {"short": (1, 2, 3)}
    first = _trajectory(requests[0], teacher["short"])
    runs = {"short": [first, _trajectory(requests[0], teacher["short"])]}
    assert gate.repeat_verdict(runs)["passed"]
    moved = gate.Trajectory(first.prompt_id, first.logits + 1.0, first.argmax, first.fed, first.calls,
                            {**first.state, "state_sha256": "other"})
    runs["short"].append(moved)
    verdict = gate.repeat_verdict(runs)
    assert not verdict["passed"] and verdict["mismatches"] == [
        {"request": "short", "repeat_index": 2, "logits_exact": False, "fed_tokens_exact": True,
         "calls_exact": True, "state_sha256_exact": False}
    ]


def test_isolation_and_routing_diagnostics():
    primary = {"a": np.zeros((4, 3), np.float32), "b": np.ones((2, 3), np.float32)}
    assert gate.isolation_verdict(primary, dict(primary))["passed"]
    replay = {"a": np.zeros((4, 3), np.float32), "b": np.ones((2, 3), np.float32) + 1}
    verdict = gate.isolation_verdict(primary, replay)
    assert verdict["mismatched"] == ["b"] and not verdict["passed"]
    assert not gate.isolation_verdict(primary, {})["passed"]
    unobserved = gate.UNOBSERVED_ROUTES_HASH
    strict = [gate.Call("DECODE", (1,), (), True, "c1_eager", "s", True),
              gate.Call("DECODE", (2,), (), True, "c1_eager", unobserved, True)]
    candidate = [gate.Call("DECODE", (1,), (), True, "c1_eager", "s", True),
                 gate.Call("DECODE", (2,), (), True, "c1_eager", unobserved, True)]
    assert gate.routing_agreement(strict, candidate)["calls_routed_as_strict"] == 1
    assert gate.routing_agreement(strict, candidate)["calls_routed_on_host_in_both_arms"] == 1


def test_public_route_check_compares_generate_detailed_with_the_session_chain():
    free = gate.Trajectory("p", np.zeros((4, 5), np.float32), (3, 9, 1, 4), (3, 9, 1), (), {})
    calls = []

    def generate_detailed(ids, sampling):
        calls.append((list(ids), sampling.max_tokens, sampling.temperature))
        return [SimpleNamespace(generated_token_ids=(3, 9, 1))]

    llm = SimpleNamespace(generate_detailed=generate_detailed)
    assert gate.public_route_check(llm, [7, 8], free)["passed"]
    assert calls == [([7, 8], 3, 0.0)]
    llm = SimpleNamespace(generate_detailed=lambda ids, sampling: [SimpleNamespace(generated_token_ids=(3, 4))])
    result = gate.public_route_check(llm, [7, 8], free)
    assert not result["passed"] and result["first_differing_token"] == 1
    empty = SimpleNamespace(generate_detailed=lambda ids, sampling: [SimpleNamespace(generated_token_ids=())])
    assert not gate.public_route_check(empty, [7, 8], free)["passed"]


def test_environment_is_restored_around_construction_even_on_failure(monkeypatch):
    for key in gate.ROUTE_ENV_KEYS:
        monkeypatch.setenv(key, "caller")
    values = {"HIPENGINE_INKLING2_RESIDENT_DECODE": "1"}
    with pytest.raises(RuntimeError, match="construction failed"):
        with gate._environment(values):
            assert os.environ["HIPENGINE_INKLING2_RESIDENT_DECODE"] == "1"
            assert os.environ["HIPENGINE_INKLING2_DECODE_GRAPH"] == "caller"
            raise RuntimeError("construction failed")
    assert os.environ["HIPENGINE_INKLING2_RESIDENT_DECODE"] == "caller"
    monkeypatch.delenv("HIPENGINE_INKLING2_DECODE_GRAPH")
    with gate._environment({"HIPENGINE_INKLING2_DECODE_GRAPH": "0"}):
        assert os.environ["HIPENGINE_INKLING2_DECODE_GRAPH"] == "0"
    assert "HIPENGINE_INKLING2_DECODE_GRAPH" not in os.environ


def test_route_recorder_hashes_host_routes_and_restores_the_method():
    class Model:
        def _select_routes(self, visit, logits):
            return ([visit.weight_layer, 4], [0.5, 0.5])

    def visit(layer):
        return SimpleNamespace(weight_layer=layer)

    model = Model()
    with gate._RouteRecorder(model) as recorder:
        model._select_routes(visit(3), None)
        first = recorder.take()
        model._select_routes(visit(3), None)
        model._select_routes(visit(1), None)
        second = recorder.take()
    assert first != second and recorder.take() == gate.UNOBSERVED_ROUTES_HASH
    assert "_select_routes" not in vars(model)
    assert model._select_routes(visit(3), None) == ([3, 4], [0.5, 0.5])
    owned = SimpleNamespace(_select_routes=lambda visit, logits: ([1], [1.0]))
    with gate._RouteRecorder(owned) as recorder:
        owned._select_routes(visit(0), None)
        assert recorder.take() != gate.UNOBSERVED_ROUTES_HASH
    assert owned._select_routes(visit(0), None) == ([1], [1.0]) and "_select_routes" in vars(owned)


def test_route_metadata_requires_the_profile_and_the_routes_under_test():
    projection = SimpleNamespace(device_routes_enabled=lambda: True, _target_arch="gfx1100")
    model = SimpleNamespace(projection=projection, resident_decode=True, decode_graph=False)
    llm = SimpleNamespace(resolved_execution_profile="fast", execution_profile_fell_back_to_strict=False,
                          execution_profile_manifest_sha256="m", backend="hip_gfx1100")
    route = gate._route_metadata(llm, model, gate.CANDIDATES["fast_resident"])
    assert route["routes"] == {"device_routes": True, "resident_decode": True, "decode_graph": False}
    assert route["target_arch"] == "gfx1100" and route["profile"] == "fast"
    with pytest.raises(gate.GateError, match="engaged"):
        gate._route_metadata(llm, model, gate.CANDIDATES["fast_graph"])
    with pytest.raises(gate.GateError, match="engaged"):
        gate._route_metadata(llm, model, gate.CANDIDATES["fast"])
    llm.execution_profile_fell_back_to_strict = True
    with pytest.raises(gate.GateError, match="resolved profile"):
        gate._route_metadata(llm, model, gate.CANDIDATES["fast_resident"])
    assert gate.expected_flags(gate.STRICT_ARM) == {
        "device_routes": False, "resident_decode": False, "decode_graph": False
    }


def test_candidates_declare_their_mechanism_strict_fallback_and_flags():
    for name, candidate in gate.CANDIDATES.items():
        assert candidate.strict_fallback.startswith("strict") and candidate.mechanism
        assert candidate.profile == "fast" and set(candidate.environment) == set(gate.ROUTE_ENV_KEYS)
    assert gate.expected_flags(gate.CANDIDATES["fast_graph"]) == {
        "device_routes": True, "resident_decode": True, "decode_graph": True
    }
    assert gate.CANDIDATES["fast_graph"].graph and not gate.CANDIDATES["fast_resident"].graph
