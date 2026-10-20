"""CPU contracts for the bounded streamed mixed-economics capture.

The live half of ``scripts/qwen38fn_mixed_economics.py`` needs a resident
server; these tests cover the parts that decide what the capture means: arm
flags, streamed token accounting under coalescing and missing telemetry, wave
aggregation, and the decode telemetry delta that keeps physical rows distinct
from HTTP requests.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[1]
_spec = importlib.util.spec_from_file_location(
    "qwen38fn_mixed_economics",
    REPO_ROOT / "scripts/qwen38fn_mixed_economics.py",
)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


def delta_event(count: int | None, *, elapsed_ms: float | None = 12.5) -> dict:
    """A streamed delta carrying the server's cumulative token count."""

    metadata: dict = {}
    if count is not None:
        metadata["tokens"] = {"streamed_tokens": count}
    event: dict = {"choices": [{"finish_reason": None, "hipengine": metadata}]}
    if elapsed_ms is not None:
        event["hipengine"] = {"timing": {"elapsed_ms": elapsed_ms}}
    return event


def done_event(
    *,
    ids: list[int] | None = None,
    count: int | None = None,
    route: str | None = None,
    mtp: dict | None = None,
) -> dict:
    metadata: dict = {}
    if ids is not None:
        metadata["generated_token_ids"] = ids
    if count is not None:
        metadata["generated_tokens"] = count
    if route is not None:
        metadata["generation_shape"] = {"route": route}
    if mtp is not None:
        metadata["speculative_mtp"] = mtp
    return {
        "choices": [{"finish_reason": "length", "hipengine": metadata}],
    }


def capture_with(events: list[dict], *, started_at: float = 100.0) -> "_module.StreamCapture":
    capture = _module.StreamCapture(started_at=started_at)
    for index, event in enumerate(events):
        capture.observe(event, now=started_at + 0.1 * (index + 1))
    return capture


def test_ar_peer_keeps_shared_target_diagnostics_without_inventing_mtp():
    event = done_event(count=8)
    diagnostics = {"mixed_target": {"physical_request_width_histogram": {"2": 3},
                                    "target_rows": 12, "committed_tokens": 3}}
    event["choices"][0]["hipengine"]["diagnostics"] = diagnostics
    capture = capture_with([event])
    record = _module.request_record(
        capture, row={"id": "peer", "category": "code", "prompt": "x"},
        arm="mixed", requested_speculative_mtp=False, steps=8, wall_s=1.0)
    assert record["diagnostics"] == diagnostics
    assert record["speculative_mtp"] is None
    assert record["speculative_mtp_present"] is False


def test_missing_peer_diagnostics_stay_absent():
    capture = capture_with([done_event(count=8)])
    record = _module.request_record(
        capture, row={"id": "peer", "prompt": "x"}, arm="ar",
        requested_speculative_mtp=False, steps=8, wall_s=1.0)
    assert record["diagnostics"] is None


# --- arm flags ------------------------------------------------------------


def test_ar_arm_pins_speculative_mtp_false_instead_of_omitting_it():
    assert _module.arm_flags("ar", 4) == [False, False, False, False]
    payload = _module.request_payload(
        "m", "prompt", speculative_mtp=False, steps=64
    )
    # An omitted key lets the server's automatic MTP selection run, which is the
    # defect this arm exists to prevent.
    assert "speculative_mtp" in payload
    assert payload["speculative_mtp"] is False
    assert payload["stream_options"] == {
        "include_usage": True,
        "include_hipengine": True,
    }
    assert payload["stream"] is True
    assert payload["max_tokens"] == 64
    assert payload["ignore_eos"] is True


def test_mixed_arm_is_exactly_one_mtp_request_plus_ar_peers():
    assert _module.arm_flags("mixed", 8) == [True] + [False] * 7
    assert _module.arm_flags("mixed", 1) == [True]
    assert _module.request_payload(
        "m", "prompt", speculative_mtp=True, steps=8
    )["speculative_mtp"] is True


def test_odd_and_partial_groups_keep_one_mtp_request():
    for width in (3, 5, 7):
        flags = _module.arm_flags("mixed", width)
        assert len(flags) == width
        assert flags.count(True) == 1
    # A group narrower than the requested width is a real wave, not a rounding.
    rows = [{"id": f"p{index}", "category": "code", "prompt": "x"} for index in range(5)]
    prompts = _module.wave_prompts(rows, width=8, offset=0)
    assert len(prompts) == 8
    assert [row["id"] for row in prompts[:6]] == ["p0", "p1", "p2", "p3", "p4", "p0"]
    assert _module.arm_flags("mixed", len(prompts)).count(True) == 1


def test_unknown_arm_and_bad_width_are_rejected():
    with pytest.raises(ValueError):
        _module.arm_flags("mtp", 4)
    with pytest.raises(ValueError):
        _module.arm_flags("ar", 0)
    with pytest.raises(ValueError):
        _module.request_payload("m", "p", speculative_mtp=False, steps=0)


# --- final token accounting and coalescing --------------------------------


def test_coalesced_deltas_get_one_timestamp_and_their_true_width():
    capture = capture_with(
        [
            delta_event(1),
            delta_event(2),
            delta_event(4),
            delta_event(5),
            done_event(ids=[1, 2, 3, 4, 5], count=5),
        ]
    )
    assert len(capture.token_times) == 4
    assert capture.event_token_counts == [1, 1, 2, 1]
    assert capture.event_elapsed_ms == [12.5] * 4
    accounting = capture.accounting()
    assert accounting["coalesced_events"] == 1
    assert accounting["delivered_events"] == 4
    assert accounting["event_streamed_tokens"] == 5
    assert accounting["done_event_generated_tokens"] == 5
    assert accounting["consistent"] is True
    assert capture.final_generated_token_ids == [1, 2, 3, 4, 5]
    assert capture.final_generated_tokens() == 5


def test_missing_streamed_counts_keep_timestamps_and_report_absence():
    capture = capture_with(
        [delta_event(None), delta_event(None), delta_event(None)]
    )
    assert len(capture.token_times) == 3
    assert capture.event_token_counts == [None, None, None]
    accounting = capture.accounting()
    assert accounting["event_streamed_tokens"] is None
    assert accounting["coalesced_events"] == 0
    assert accounting["delivered_events"] == 3
    # The deltas were timed but never counted, so no total is claimed.
    assert capture.final_generated_tokens() is None


def test_done_event_ids_drive_the_final_count_when_deltas_were_uncounted():
    capture = capture_with(
        [delta_event(None), delta_event(None), done_event(ids=[7, 8, 9])]
    )
    assert capture.final_generated_tokens() == 3
    assert capture.final_generated_token_ids == [7, 8, 9]
    accounting = capture.accounting()
    assert accounting["event_streamed_tokens"] is None
    assert accounting["done_event_generated_tokens"] == 3
    # One published count cannot disagree with itself, and silence is not
    # agreement.
    assert accounting["consistent"] is None


def test_usage_is_the_fallback_count_when_the_done_event_omits_ids():
    capture = capture_with(
        [delta_event(None), {"usage": {"prompt_tokens": 11, "completion_tokens": 4}}]
    )
    assert capture.final_generated_tokens() == 4
    assert capture.usage == {
        "prompt_tokens": 11,
        "completion_tokens": 4,
        "total_tokens": None,
    }


def test_a_disagreeing_count_is_flagged_rather_than_averaged():
    capture = capture_with(
        [delta_event(1), delta_event(2), done_event(ids=[1, 2, 3], count=3)]
    )
    accounting = capture.accounting()
    assert accounting["event_streamed_tokens"] == 2
    assert accounting["done_event_generated_tokens"] == 3
    assert accounting["consistent"] is False


def test_zero_advance_delta_is_not_a_latency_sample():
    capture = capture_with([delta_event(1), delta_event(1), delta_event(2)])
    assert capture.events == 3
    assert len(capture.token_times) == 2
    assert capture.accounting()["delivered_events"] == 2


def test_backwards_token_count_is_a_protocol_failure():
    with pytest.raises(RuntimeError):
        capture_with([delta_event(3), delta_event(2)])


def test_stream_error_event_raises():
    with pytest.raises(RuntimeError):
        capture_with([{"error": {"message": "boom"}}])


def test_absent_speculative_block_is_recorded_as_absent():
    capture = capture_with([delta_event(1), done_event(ids=[5], count=1)])
    record = _module.request_record(
        capture,
        row={"id": "p", "category": "code", "prompt": "x"},
        arm="ar",
        requested_speculative_mtp=False,
        steps=1,
        wall_s=0.5,
    )
    assert record["speculative_mtp"] is None
    assert record["speculative_mtp_present"] is False
    assert record["route"] is None
    assert record["route_published"] is False
    assert record["expected_tokens_match"] is True


def test_per_request_route_and_acceptance_diagnostics_are_kept():
    capture = capture_with(
        [
            delta_event(1),
            done_event(
                ids=[5],
                count=1,
                route="speculative_mtp",
                mtp={
                    "used": True,
                    "acceptance_rate": 0.75,
                    "accepted_draft_tokens": 3,
                    "draft_tokens": 4,
                    "decision_reason": "requested",
                    "output_accounting": {
                        "reconciled": True,
                        "reconciled_reasons": [],
                        "mtp_output_tokens": 1,
                    },
                },
            ),
        ]
    )
    record = _module.request_record(
        capture,
        row={"id": "p", "category": "code", "prompt": "x"},
        arm="mixed",
        requested_speculative_mtp=True,
        steps=1,
        wall_s=0.5,
    )
    assert record["route"] == "speculative_mtp"
    assert record["route_published"] is True
    diagnostics = record["speculative_mtp"]
    assert diagnostics["used"] is True
    assert diagnostics["acceptance_rate"] == pytest.approx(0.75)
    assert diagnostics["accepted_draft_tokens"] == 3
    assert diagnostics["decision_reason"] == "requested"
    assert diagnostics["output_accounting"]["reconciled"] is True


# --- wave aggregation ----------------------------------------------------


def record(
    *,
    prompt_id: str,
    mtp: bool,
    times: list[float],
    tokens: int,
    status: int = 200,
    acceptance: float | None = None,
    used: bool | None = None,
) -> dict:
    diagnostics = None
    if acceptance is not None or used is not None:
        diagnostics = {"used": used, "acceptance_rate": acceptance}
    return {
        "prompt_id": prompt_id,
        "status": status,
        "requested_speculative_mtp": mtp,
        "final_generated_tokens": tokens,
        "token_times": times,
        "expected_tokens_match": True,
        "accounting": {"delivered_events": len(times), "coalesced_events": 0},
        "speculative_mtp": diagnostics,
        "speculative_mtp_present": diagnostics is not None,
    }


def test_wave_latency_uses_ar_peers_only_and_keeps_rows_apart():
    records = [
        record(prompt_id="mtp", mtp=True, times=[0.0, 0.5], tokens=8, used=True),
        record(prompt_id="a", mtp=False, times=[0.0, 0.2, 0.4], tokens=4),
        record(prompt_id="b", mtp=False, times=[0.1, 0.35], tokens=4),
    ]
    summary = _module.summarize_wave(records, 2.0)
    assert summary["requests"] == 3
    assert summary["mtp_requests"] == 1
    assert summary["ar_requests"] == 2
    # 200 ms, 200 ms and 250 ms from the two AR peers; the MTP request's 500 ms
    # gap is a different quantity and must not enter the AR-peer sample.
    assert summary["ar_peer_inter_token"]["intervals"] == 3
    assert summary["ar_peer_inter_token"]["p50_ms"] == pytest.approx(200.0)
    assert summary["ar_peer_inter_token"]["max_ms"] == pytest.approx(250.0)
    assert summary["generated_tokens"] == 16
    assert summary["wave_tok_s"] == pytest.approx(8.0)
    assert summary["speculative_used_requests"] == 1


def test_wave_reports_missing_telemetry_and_failures_separately():
    records = [
        record(prompt_id="a", mtp=False, times=[0.0, 0.1], tokens=2),
        record(prompt_id="b", mtp=False, times=[], tokens=0, status=500),
    ]
    summary = _module.summarize_wave(records, 1.0)
    assert summary["errors"] == 1
    assert summary["ar_requests"] == 1
    assert summary["speculative_absent_requests"] == 1
    assert summary["speculative_present_requests"] == 0
    assert summary["acceptance_rate_samples"] == 0
    assert summary["reused_prompts"] == 0


def test_reused_prompts_are_counted_when_the_suite_is_shorter_than_the_width():
    records = [
        record(prompt_id=prompt_id, mtp=False, times=[0.0, 0.1], tokens=2)
        for prompt_id in ("a", "b", "a")
    ]
    summary = _module.summarize_wave(records, 1.0)
    assert summary["distinct_prompts"] == 2
    assert summary["reused_prompts"] == 1


def test_aggregate_pools_intervals_and_does_not_average_rates():
    fast = {
        "summary": _module.summarize_wave(
            [record(prompt_id="a", mtp=False, times=[0.0, 0.1, 0.2], tokens=3)], 1.0
        ),
        "decode_delta": {"steps_delta": 4, "step_width_delta": {"1": 4},
                         "mixed_target_passes_delta": 0},
    }
    slow = {
        "summary": _module.summarize_wave(
            [record(prompt_id="b", mtp=False, times=[0.0, 1.0, 2.0], tokens=3)], 4.0
        ),
        "decode_delta": {"steps_delta": 2, "step_width_delta": {"2": 2},
                         "mixed_target_passes_delta": 1},
    }
    totals = _module.aggregate([fast, slow])
    assert totals["waves"] == 2
    assert totals["wall_s"] == pytest.approx(5.0)
    assert totals["generated_tokens"] == 6
    assert totals["wave_tok_s"] == pytest.approx(1.2)
    # Pooled: two 100 ms intervals and two 1000 ms intervals, not the mean of
    # two per-wave p50s.
    assert totals["ar_peer_inter_token"]["intervals"] == 4
    assert totals["ar_peer_inter_token"]["p50_ms"] == pytest.approx(550.0)
    assert totals["ar_peer_inter_token"]["max_ms"] == pytest.approx(1000.0)
    assert totals["physical_decode_steps"] == 6
    assert totals["physical_step_widths"] == {"1": 4, "2": 2}
    assert totals["mixed_target_passes"] == 1


def test_aggregate_reports_no_physical_widths_when_telemetry_was_absent():
    wave = {
        "summary": _module.summarize_wave(
            [record(prompt_id="a", mtp=False, times=[0.0, 0.1], tokens=2)], 1.0
        ),
        "decode_delta": {
            "available": False,
            "steps_delta": None,
            "step_width_delta": None,
            "mixed_target_passes_delta": None,
        },
    }
    totals = _module.aggregate([wave])
    assert totals["physical_decode_steps"] is None
    assert totals["physical_step_widths"] is None
    assert totals["mixed_target_passes"] is None


def test_arm_ratios_pair_the_same_cell_and_prompts():
    waves = []
    for arm, rate in (("ar", 2.0), ("mixed", 3.0)):
        summary = _module.summarize_wave(
            [record(prompt_id="a", mtp=False, times=[0.0, 0.1], tokens=int(rate))],
            1.0,
        )
        waves.append({"repeat": 0, "width": 2, "cell": 0, "arm": arm, "summary": summary})
    ratios = _module.arm_ratios(waves, ["ar", "mixed"])
    assert len(ratios) == 1
    assert ratios[0]["arm"] == "mixed"
    assert ratios[0]["ratio_vs_ar"] == pytest.approx(1.5)
    assert ratios[0]["ar_prompt_ids"] == ratios[0]["arm_prompt_ids"]
    # An AR-only capture has nothing to compare against.
    assert _module.arm_ratios(waves[:1], ["ar"]) == []


# --- wave plan -----------------------------------------------------------


def test_plan_alternates_arm_order_per_repeat():
    plan = _module.wave_plan(
        arms=("ar", "mixed"), widths=(1, 4), repeats=3, row_count=18
    )
    by_repeat: dict[int, list[str]] = {}
    for entry in plan:
        by_repeat.setdefault(entry["repeat"], []).append(entry["arm"])
    assert by_repeat[0] == ["ar", "mixed", "ar", "mixed"]
    assert by_repeat[1] == ["mixed", "ar", "mixed", "ar"]
    assert by_repeat[2] == by_repeat[0]
    assert [entry["index"] for entry in plan] == list(range(len(plan)))
    assert len(plan) == 3 * 2 * 2


def test_plan_keeps_prompt_selection_stable_across_arms_of_one_cell():
    plan = _module.wave_plan(
        arms=("ar", "mixed"), widths=(2, 8), repeats=2, row_count=18
    )
    offsets: dict[tuple[int, int, int], set[int]] = {}
    for entry in plan:
        key = (entry["repeat"], entry["width"], entry["cell"])
        offsets.setdefault(key, set()).add(entry["prompt_offset"])
    # The arm is not part of the key, so both arms of a cell see one offset.
    assert {key: next(iter(values)) for key, values in offsets.items()} == {
        (0, 2, 0): 0,
        (0, 8, 0): 1,
        (1, 2, 0): 2,
        (1, 8, 0): 3,
    }


def test_full_suite_cells_cover_every_prompt_in_the_mtp_role_at_each_width():
    rows = [{'id': str(index), 'prompt': 'x'} for index in range(18)]
    plan = _module.wave_plan(arms=('ar', 'mixed'), widths=(1, 2, 4, 8),
                            repeats=3, row_count=18, waves_per_cell=18)
    for repeat in range(3):
        for width in (1, 2, 4, 8):
            selected = [entry for entry in plan if entry['repeat'] == repeat
                        and entry['width'] == width and entry['arm'] == 'mixed']
            mtp_ids = {_module.wave_prompts(rows, width=width,
                       offset=entry['prompt_offset'])[0]['id'] for entry in selected}
            assert mtp_ids == {row['id'] for row in rows}


def test_plan_rejects_empty_or_negative_shape():
    with pytest.raises(ValueError):
        _module.wave_plan(arms=(), widths=(1,), repeats=1, row_count=4)
    with pytest.raises(ValueError):
        _module.wave_plan(arms=("ar",), widths=(1,), repeats=0, row_count=4)
    with pytest.raises(ValueError):
        _module.wave_plan(arms=("ar",), widths=(1,), repeats=1, row_count=0)


def test_arm_order_reverses_on_odd_repeats():
    assert _module.arm_order(("ar", "mixed"), 0) == ["ar", "mixed"]
    assert _module.arm_order(("ar", "mixed"), 1) == ["mixed", "ar"]


# --- decode telemetry ----------------------------------------------------


def decode_body(
    *,
    steps: int,
    widths: dict[str, int],
    passes: int = 0,
    last_pass: dict | None = None,
    live_slots: int = 0,
) -> dict:
    return {
        "object": "hipengine.decode_telemetry",
        "available": True,
        "decode": {
            "decode_path": "qwen4exp_full_model_ar_batch",
            "request_capacity": 8,
            "ar_batch": {
                "steps": steps,
                "live_slots": live_slots,
                "step_widths": widths,
                "failure": None,
            },
            "mixed_target_passes": passes,
            "last_mixed_target_pass": last_pass,
        },
    }


def test_decode_delta_keeps_physical_widths_and_target_rows_apart():
    before = _module.decode_snapshot(decode_body(steps=10, widths={"1": 10}, passes=0))
    after = _module.decode_snapshot(
        decode_body(
            steps=26,
            widths={"1": 10, "2": 12, "4": 4},
            passes=2,
            last_pass={"request_width": 4, "target_rows": 6},
            live_slots=4,
        )
    )
    delta = _module.decode_delta(before, after)
    assert delta["available"] is True
    assert delta["steps_delta"] == 16
    assert delta["step_width_delta"] == {"1": 0, "2": 12, "4": 4}
    assert delta["mixed_target_passes_delta"] == 2
    # Six physical rows served four requests: the two quantities stay named.
    assert delta["mixed_target_pass"]["target_rows"] == 6
    assert delta["mixed_target_pass"]["request_width"] == 4
    assert "not HTTP request counts" in delta["scope"]


def test_decode_delta_does_not_attribute_a_previous_pass_to_this_wave():
    # An AR wave runs no mixed pass, but the server still reports the most
    # recent pass it ever ran.
    stale = {"request_width": 4, "target_rows": 6}
    before = _module.decode_snapshot(
        decode_body(steps=10, widths={"1": 10}, passes=2, last_pass=stale)
    )
    after = _module.decode_snapshot(
        decode_body(steps=14, widths={"1": 14}, passes=2, last_pass=stale)
    )
    delta = _module.decode_delta(before, after)
    assert delta["mixed_target_passes_delta"] == 0
    assert delta["mixed_target_pass"] is None
    assert delta["last_mixed_target_pass"] == stale


def test_decode_delta_reports_absent_telemetry_without_zeroing_it():
    unavailable = _module.decode_snapshot(
        {"object": "hipengine.decode_telemetry", "available": False,
         "reason": "engine_has_no_decode_telemetry"}
    )
    assert unavailable == {
        "available": False,
        "reason": "engine_has_no_decode_telemetry",
    }
    delta = _module.decode_delta(unavailable, unavailable)
    assert delta["available"] is False
    assert delta["reason"] == "engine_has_no_decode_telemetry"
    assert delta["steps_delta"] is None
    assert delta["step_width_delta"] is None
    assert delta["step_widths"] is None
    assert delta["mixed_target_passes_delta"] is None


def test_decode_delta_leaves_widths_absent_when_no_batch_owner_exists():
    body = {
        "available": True,
        "decode": {"decode_path": "qwen4exp_c1_request_owned_step", "ar_batch": None},
    }
    snapshot = _module.decode_snapshot(body)
    assert snapshot["available"] is True
    assert snapshot["ar_batch_present"] is False
    assert snapshot["step_widths"] is None
    delta = _module.decode_delta(snapshot, snapshot)
    assert delta["step_width_delta"] is None
    assert delta["steps_delta"] is None


def test_decode_snapshot_rejects_a_malformed_body():
    assert _module.decode_snapshot(["not", "a", "mapping"]) == {
        "available": False,
        "reason": "decode_telemetry_not_an_object",
    }
    assert _module.decode_snapshot({"available": True}) == {
        "available": False,
        "reason": "decode_telemetry_missing_decode",
    }


# --- latency helpers -----------------------------------------------------


def test_empty_latency_is_absent_rather_than_zero():
    assert _module.latency_summary([]) == {
        "intervals": 0,
        "p50_ms": None,
        "p95_ms": None,
        "p99_ms": None,
        "max_ms": None,
    }


def test_percentiles_interpolate_without_rounding_away_tail_samples():
    assert _module.percentile([5.0, 1.0, 2.0, 100.0], 0.95) == pytest.approx(85.75)
    assert _module.percentile([], 0.5) is None


# --- suite loading -------------------------------------------------------


def test_suites_load_through_the_canonical_loader():
    rows = _module.load_suite(_module.DEFAULT_PROMPTS)
    assert len(rows) == 18
    assert len({row["id"] for row in rows}) == 18
    heldout = next(row for row in rows if row["id"] == "heldout_mixed_review")
    # load_prompt_rows joins the user content; that raw text is what the request
    # sends, so the recorded hash is over the normalized prompt.
    assert heldout["prompt"] == heldout["source"]["messages"][0]["content"]
    assert heldout["category"] == "mixed_ja_en"
    assert _module.prompt_sha256(heldout["prompt"]) == _module.prompt_sha256(
        heldout["source"]["messages"][0]["content"]
    )
    assert heldout["source_path"].endswith("gdn-prefill-category-heldouts.jsonl")


def test_missing_suite_and_duplicate_ids_are_refused(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        _module.load_suite([tmp_path / "absent.jsonl"])
    duplicate = tmp_path / "dup.jsonl"
    duplicate.write_text(
        json.dumps({"id": "a", "category": "code", "prompt": "one"}) + "\n"
        + json.dumps({"id": "a", "category": "code", "prompt": "two"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        _module.load_suite([duplicate])


def test_cross_file_duplicate_prompt_ids_are_refused(tmp_path: Path):
    first = tmp_path / "a.jsonl"
    second = tmp_path / "b.jsonl"
    for path in (first, second):
        path.write_text(
            json.dumps({"id": "same", "category": "code", "prompt": "x"}) + "\n",
            encoding="utf-8",
        )
    with pytest.raises(ValueError, match="duplicate prompt id"):
        _module.load_suite([first, second])


# --- verdict -------------------------------------------------------------


def _wave(index: int, *, arm: str, width: int, records: list[dict]) -> dict:
    return {
        "index": index,
        "repeat": 0,
        "width": width,
        "cell": 0,
        "arm": arm,
        "summary": _module.summarize_wave(records, 1.0),
        "requests": records,
        "decode_delta": {},
    }


def test_verdict_accepts_a_capture_with_absent_mixed_telemetry():
    artifact = {
        "waves": [
            _wave(0, arm="ar", width=2,
                  records=[record(prompt_id="a", mtp=False, times=[0.0, 0.1], tokens=2)]),
            _wave(1, arm="mixed", width=2,
                  records=[
                      record(prompt_id="a", mtp=True, times=[0.0, 0.1], tokens=2),
                      record(prompt_id="b", mtp=False, times=[0.0, 0.1], tokens=2),
                  ]),
        ],
        "source_unchanged": True,
    }
    assert _module.verdict(artifact) == []


def test_verdict_flags_a_requested_mtp_that_did_not_run():
    artifact = {
        "waves": [
            _wave(0, arm="mixed", width=1,
                  records=[record(prompt_id="a", mtp=True, times=[0.0, 0.1],
                                  tokens=2, used=False, acceptance=None)]),
        ],
        "source_unchanged": True,
    }
    problems = _module.verdict(artifact)
    assert len(problems) == 1
    assert "did not run" in problems[0]


def test_verdict_flags_failed_requests_and_source_drift():
    artifact = {
        "waves": [
            _wave(0, arm="ar", width=1,
                  records=[record(prompt_id="a", mtp=False, times=[], tokens=0,
                                  status=500)]),
        ],
        "source_unchanged": False,
    }
    problems = _module.verdict(artifact)
    assert any("failed request" in problem for problem in problems)
    assert any("source changed" in problem for problem in problems)


# --- command line --------------------------------------------------------


def test_cli_rejects_unknown_arms_and_repeated_widths():
    parser = _module.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--arms", "mtp", "--server-repo", ".", "--label", "x",
                           "--output", "out.json"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--widths", "1,1", "--server-repo", ".", "--label", "x",
                           "--output", "out.json"])
    args = parser.parse_args(["--server-repo", ".", "--label", "run-1",
                              "--output", "out.json"])
    assert args.widths == [1, 2, 4, 8]
    assert args.steps == 64
    assert args.repeats == 3
    assert args.warmups == 1
    assert args.arms == ["ar", "mixed"]
    assert list(args.prompts) == list(_module.DEFAULT_PROMPTS)
