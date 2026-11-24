"""CPU tests for scripts/inkling2_decode_pairs.py (no model, no GPU)."""

import importlib.util
import json
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parent.parent / "scripts" / "inkling2_decode_pairs.py"
spec = importlib.util.spec_from_file_location("inkling2_decode_pairs", MODULE_PATH)
pairs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pairs)


def _run(order, tokens, digest, tok=10.0, calls=None):
    return {"order": order, "pair": 0, "arm": "x", "resolved_profile": "fast", "fell_back_to_strict": False,
            "depths": [{"prompt_tokens": tokens, "reps": 2, "decode_tok_s": {"median": tok, "min": tok, "max": tok},
                        "decode_ms_per_token": 1000.0 / tok,
                        "first_token_s": {"median": 0.25, "min": 0.25, "max": 0.25},
                        "identical_output": True, "output_ids_sha256": [digest],
                        "calls_per_token": calls, "graph": None}]}


def test_variables_reject_anything_but_a_named_hipengine_value():
    assert pairs.variables("HIPENGINE_SUBMISSION_TRANSPORT=aql") == {"HIPENGINE_SUBMISSION_TRANSPORT": "aql"}
    for bad in ("HIPENGINE_", "OTHER_X=1", "HIPENGINE_lower=1", "=1"):
        with pytest.raises(ValueError):
            pairs.variables(bad)


def test_resolve_arms_lists_each_arm_with_only_the_variables_it_changes():
    arms = pairs.resolve_arms(["default", "graph"], ["graph:HIPENGINE_SUBMISSION_TRANSPORT=aql"])
    assert arms == {"default": {},
                    "graph": {"HIPENGINE_INKLING2_DEVICE_ROUTES": "1", "HIPENGINE_INKLING2_RESIDENT_DECODE": "1",
                              "HIPENGINE_INKLING2_DECODE_GRAPH": "1", "HIPENGINE_SUBMISSION_TRANSPORT": "aql"}}
    assert pairs.resolve_arms(["custom"], ["custom:HIPENGINE_INKLING2_DECODE_GRAPH=1"]) == {
        "custom": {"HIPENGINE_INKLING2_DECODE_GRAPH": "1"}}
    with pytest.raises(ValueError, match="arm that runs"):
        pairs.resolve_arms(["default"], ["graph:HIPENGINE_INKLING2_DECODE_GRAPH=1"])


def test_schedule_runs_every_pair_in_the_listed_order_and_again_reversed():
    assert pairs.schedule(["a", "b", "c"], 2) == [
        ("forward", 0, "a"), ("forward", 0, "b"), ("forward", 0, "c"),
        ("forward", 1, "a"), ("forward", 1, "b"), ("forward", 1, "c"),
        ("reverse", 0, "c"), ("reverse", 0, "b"), ("reverse", 0, "a"),
        ("reverse", 1, "c"), ("reverse", 1, "b"), ("reverse", 1, "a")]


def test_summary_and_decision_need_a_win_in_both_orders_and_the_reference_output():
    reference = [_run("forward", 47, "d"), _run("reverse", 47, "d")]
    faster = [_run("forward", 47, "d", 12.0), _run("reverse", 47, "d", 11.0)]
    other_output = [_run("forward", 47, "e", 12.0), _run("reverse", 47, "e", 11.0)]
    one_order = [_run("forward", 47, "d", 12.0), _run("reverse", 47, "d", 10.1)]
    depths = pairs.summarize_arm(faster, reference)
    assert depths[0]["decode_tok_s"] == {"median": 11.5, "min": 11.0, "max": 12.0}
    assert depths[0]["tok_s_ratio_to_reference"] == {"forward": pytest.approx(1.2), "reverse": pytest.approx(1.1)}
    assert depths[0]["output_identical_to_reference"] and depths[0]["identical_output"]
    (decision,) = pairs.decide(depths, pairs.summarize_arm(reference, reference))
    assert decision["faster_in_both_orders"] and decision["promote"]
    assert decision["reference_decode_tok_s"]["median"] == 10.0
    (decision,) = pairs.decide(pairs.summarize_arm(other_output, reference), pairs.summarize_arm(reference, reference))
    assert decision["faster_in_both_orders"] and not decision["output_identical_to_reference"]
    assert not decision["promote"]
    (decision,) = pairs.decide(pairs.summarize_arm(one_order, reference), pairs.summarize_arm(reference, reference))
    assert not decision["faster_in_both_orders"] and not decision["promote"]


def test_summary_averages_call_tallies_and_graph_accounting_over_runs():
    calls = {"stream_synchronize": {"calls": 4.0, "seconds": 0.4, "bytes": 0.0}}
    first = _run("forward", 47, "d", calls=calls)
    second = _run("reverse", 47, "d", calls={"stream_synchronize": {"calls": 2.0, "seconds": 0.2, "bytes": 0.0}})
    for run, graph in ((first, {"captures": 1, "capture_s": 0.3, "replays_per_token": 126.0}),
                       (second, {"captures": 1, "capture_s": 0.1, "replays_per_token": 130.0})):
        run["depths"][0]["graph"] = graph
    (depth,) = pairs.summarize_arm([first, second], [first, second])
    assert depth["calls_per_token"] == {"stream_synchronize": {"calls": 3.0, "seconds": pytest.approx(0.3), "bytes": 0.0}}
    assert depth["graph"] == {"captures_per_request": 1, "capture_s": pytest.approx(0.2), "replays_per_token": 128.0}


def test_trace_range_sums_kernels_inside_the_marker_range_per_step(tmp_path):
    kernel = tmp_path / "kernel.csv"
    kernel.write_text(
        'Kind,Kernel_Name,Start_Timestamp,End_Timestamp\n'
        '"KERNEL_DISPATCH","void (anonymous namespace)::thing<float>(float*)",90,190\n'
        '"KERNEL_DISPATCH","split_scores",100,160\n'
        '"KERNEL_DISPATCH","outside",1200,1300\n', encoding="utf-8")
    marker = tmp_path / "marker.csv"
    marker.write_text('Domain,Function,Start_Timestamp,End_Timestamp\n"M","inkling_ordinary_decode_p47",100,1100\n',
                      encoding="utf-8")
    trace = pairs.trace_range(kernel, marker, "inkling_ordinary_decode_p47", 10)
    # Only the 60 ns dispatch lies inside the range: one that starts before it and one after it do not.
    assert trace["kernel_ms_per_step"] == pytest.approx(6e-6)
    assert trace["wall_ms_per_step"] == pytest.approx(1e-4)
    assert trace["launches_per_step"] == pytest.approx(0.1)
    assert [item["family"] for item in trace["families"]] == ["split_scores"]
    with pytest.raises(ValueError, match="not found"):
        pairs.trace_range(kernel, marker, "missing", 10)
    assert json.loads(json.dumps(trace))["label"] == "inkling_ordinary_decode_p47"


def test_kept_depth_keys_cover_everything_the_summary_reads():
    """A run keeps only RUN_DEPTH_KEYS; the summary must not read anything else (2026-11-25)."""
    full = {**_run("forward", 47, "d")["depths"][0], "calls_per_token": {}, "discarded_diagnostic": 1}
    kept = [{key: depth[key] for key in pairs.RUN_DEPTH_KEYS} for depth in [full]]
    runs = [{**_run(order, 47, "d"), "depths": kept} for order in ("forward", "reverse")]
    (depth,) = pairs.summarize_arm(runs, runs)
    assert depth["calls_per_token"] == {} and depth["graph"] is None
    assert json.dumps({"arms": {"x": depth, "y": None}, "runs": runs}, allow_nan=False)
