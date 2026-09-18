"""Unit coverage for the ShareGPT MTP routing recorder's reporting contract.

The harness runs against a live server, so its row/summary logic is exercised
here against recorded response shapes instead. Every case below is a gap the
first two live passes exposed:

- a route-level refusal looked like a lying route because only the planned route
  was reported, not the realized one;
- a request rejected mid-stream recorded "stream ended without usage" and hid
  the error body that named the cause;
- ``first_fallback_position`` has three distinct states (never entered, left
  mid-stream, ended inside speculation) and two of them collapsed into one empty
  block;
- per-bucket refusals (``prompt_activation_in_flight``, ``no_provider``) were
  invisible in a route-level share.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _module():
    spec = importlib.util.spec_from_file_location(
        "sharegpt_mtp_routing_pass",
        REPO_ROOT / "scripts" / "sharegpt_mtp_routing_pass.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _response(
    *,
    route="ar",
    selected_route="speculative_mtp",
    reason="automatic_route",
    refusal="no_provider",
    prompt_tokens=900,
    timeline=None,
):
    mtp: dict = {
        "used": False,
        "fallback_reason": None,
        "fallback_event_counts": {},
        "fallback_event_total": 0,
        "first_fallback_position": 0,
    }
    if refusal is not None:
        mtp["fallback_reason"] = refusal
        mtp["fallback_event_counts"] = {refusal: 3}
        mtp["fallback_event_total"] = 3
    return {
        "ttft_ms": 100.0,
        "first_answer_ms": 120.0,
        "first_thinking_ms": 100.0,
        "e2e_ms": 900.0,
        "timeline": timeline or [[100.0, 1], [200.0, 17], [300.0, 33]],
        "usage": {"prompt_tokens": int(prompt_tokens), "completion_tokens": 64},
        "answer_characters": 40,
        "thinking_characters": 200,
        "speculative_mtp": mtp,
        "generation_shape": {
            "route": route,
            "route_decision": {
                "requested_route": "speculative_mtp",
                "selected_route": selected_route,
                "reason": reason,
                "k0_class": "transitional_k0",
                "policy_reason": reason,
                "realized_group_rows": 1,
                "output_horizon_tokens": 64,
                "selected_candidate_count": 0,
            },
        },
    }


def test_request_row_reports_realized_and_planned_route_separately() -> None:
    module = _module()
    row = module._request_row({"source_id": 1, "prompt_tokens": 900}, _response())

    assert row["effective_route"] == "ar"
    assert row["selected_route"] == "speculative_mtp"
    assert row["requested_route"] == "speculative_mtp"
    assert row["route_decision_reason"] == "automatic_route"
    assert row["k0_class"] == "transitional_k0"
    assert row["mtp_used"] is False


def test_error_row_records_the_in_stream_error_body() -> None:
    module = _module()
    result = _response()
    result["usage"] = {}
    result["error_body"] = {
        "message": "min_tokens requires eos_token_id",
        "type": "invalid_request_error",
        "code": "unsupported_parameter",
        "hipengine": {"code": "unsupported_parameter", "status_code": 400},
    }
    result["generation_shape"] = {
        "route": "ar",
        "route_decision": {"selected_route": "ar", "reason": "min_tokens"},
    }

    row = module._request_row({"source_id": 7, "prompt_tokens": 300}, result)

    assert row["error_code"] == "unsupported_parameter"
    assert row["error_status_code"] == 400
    assert "min_tokens requires eos_token_id" in row["error"]
    assert "stream ended without usage" not in row["error"]
    assert row["error_body"]["message"] == "min_tokens requires eos_token_id"
    # A rejected request still reports the route it was refused on.
    assert row["effective_route"] == "ar"
    assert row["selected_route"] == "ar"


def test_error_row_without_a_body_still_names_the_symptom() -> None:
    module = _module()
    result = _response()
    result["usage"] = {}

    row = module._request_row({"source_id": 8, "prompt_tokens": 300}, result)

    assert row["error"].startswith("stream ended without usage")
    assert row["error_code"] is None
    assert row["error_body"] is None


def test_cliff_separates_never_entered_from_ended_inside_speculation() -> None:
    module = _module()
    never = {
        "source_id": 1,
        "mtp_used": False,
        "first_fallback_position": 0,
        "completion_tokens": 32,
        "timeline": [[0.0, 0], [100.0, 32]],
    }
    ended_inside = {
        "source_id": 2,
        "mtp_used": True,
        "first_fallback_position": None,
        "completion_tokens": 64,
        "timeline": [[0.0, 0], [100.0, 64]],
    }

    cliff = module._cliff([never, ended_inside])

    assert cliff["case_counts"] == {
        "ended_inside_speculation": 1,
        "never_entered_speculation": 1,
    }
    assert cliff["never_entered_speculation_requests"] == 1
    assert cliff["ended_inside_speculation_requests"] == 1
    assert cliff["left_mid_stream_requests"] == 0
    assert cliff["requests"] == 0
    assert "no request both left speculation" in cliff["note"]


def test_cliff_measures_only_requests_that_left_speculation_mid_stream() -> None:
    module = _module()
    timeline = [[float(index * 10), index] for index in range(96)]
    left = {
        "source_id": 3,
        "mtp_used": True,
        "first_fallback_position": 48,
        "completion_tokens": 96,
        "timeline": timeline,
    }
    covered = {
        "source_id": 4,
        "mtp_used": True,
        "first_fallback_position": None,
        "completion_tokens": 96,
        "timeline": timeline,
    }

    cliff = module._cliff([left, covered])

    assert cliff["left_mid_stream_requests"] == 1
    assert cliff["ended_inside_speculation_requests"] == 1
    assert cliff["requests"] == 1
    assert cliff["detail"][0]["source_id"] == 3


def test_summary_reports_route_disagreements_and_bucket_refusals() -> None:
    module = _module()
    rows = [
        module._request_row({"source_id": 1, "prompt_tokens": 900}, _response()),
        module._request_row(
            {"source_id": 2, "prompt_tokens": 1200},
            _response(
                route="speculative_mtp",
                selected_route="speculative_mtp",
                refusal=None,
                prompt_tokens=1200,
            ),
        ),
    ]

    summary = module.summarize(rows)

    assert summary["effective_routes"] == {"ar": 1, "speculative_mtp": 1}
    assert summary["selected_routes"] == {"speculative_mtp": 2}
    assert len(summary["route_disagreements"]) == 1
    assert summary["route_disagreements"][0]["source_id"] == 1
    assert summary["thinking_character_share"] == pytest.approx(200 / 240)
    assert summary["ar_tokens_by_reason"] == {}
    assert summary["ar_tokens_attributed"] == 0
    bucket = summary["prompt_token_buckets"]["768-1022"]
    assert bucket["requests"] == 1
    assert bucket["no_provider_requests"] == 1
    assert bucket["no_provider_events"] == 3
    assert bucket["primary_no_provider_requests"] == 1
    assert summary["prompt_token_buckets"]["1023+"]["requests"] == 1


def test_summary_counts_prompt_activation_refusals_per_bucket() -> None:
    module = _module()
    result = _response(
        refusal="prompt_activation_in_flight",
        prompt_tokens=1500,
    )
    result["speculative_mtp"]["fallback_event_counts"] = {
        "prompt_activation_in_flight": 1
    }
    result["speculative_mtp"]["fallback_event_total"] = 1
    rows = [module._request_row({"source_id": 5, "prompt_tokens": 1500}, result)]

    summary = module.summarize(rows)

    bucket = summary["prompt_token_buckets"]["1023+"]
    assert bucket["prompt_activation_in_flight_requests"] == 1
    assert bucket["prompt_activation_in_flight_events"] == 1
    assert summary["non_mtp_reasons"] == {"prompt_activation_in_flight": 1}


def test_summary_is_json_serializable_with_a_failed_row() -> None:
    module = _module()
    result = _response()
    result["usage"] = {}
    result["error_body"] = {"message": "boom", "code": "execution_failed"}
    rows = [module._request_row({"source_id": 9, "prompt_tokens": 100}, result)]

    payload = json.dumps({"summary": module.summarize(rows), "rows": rows})

    assert "execution_failed" in payload


def test_summary_separates_event_counts_from_token_attribution() -> None:
    """A retried planning refusal costs a step, not a route."""

    module = _module()
    result = _response(refusal="no_provider")
    row = module._request_row({"source_id": 6, "prompt_tokens": 20}, result)
    row["mtp_used"] = True
    row["mtp_output_tokens"] = 46
    row["ar_output_tokens"] = 2
    row["first_fallback_position"] = None
    row["ar_output_tokens_by_reason"] = {"no_provider": 1}
    row["completion_tokens"] = 48
    row["span_accounting"] = {"reconciled": True, "ar_tokens_by_reason": {"no_provider": 1}}

    summary = module.summarize([row])

    # One refusal event, one autoregressive token, forty-six speculative ones.
    assert summary["fallback_event_total"] == 3
    assert summary["ar_tokens_by_reason"] == {"no_provider": 1}
    assert summary["ar_tokens_attributed"] == 1
    assert summary["ar_output_tokens"] == 2
    assert summary["mtp_output_share"] == 46 / 48
    assert summary["cliff"]["ended_inside_speculation_requests"] == 1


def test_request_payload_sends_the_true_ar_override_and_omits_it_for_auto() -> None:
    """The AR control is a request field, not a route label."""

    module = _module()
    captured: list[dict] = []

    class _Response:
        def __enter__(self):
            return iter([b'data: {"usage": {"prompt_tokens": 1, "completion_tokens": 1}}\n'])

        def __exit__(self, *exc):
            return False

    def _urlopen(request, timeout=None):
        captured.append(json.loads(request.data.decode()))
        return _Response()

    original = module.urllib.request.urlopen
    module.urllib.request.urlopen = _urlopen
    try:
        module._stream_request(
            "http://127.0.0.1:1",
            model="m",
            prompt="p",
            max_tokens=4,
            temperature=None,
            timeout=5.0,
            speculative_mtp=False,
        )
        module._stream_request(
            "http://127.0.0.1:1",
            model="m",
            prompt="p",
            max_tokens=4,
            temperature=None,
            timeout=5.0,
            speculative_mtp=True,
        )
        module._stream_request(
            "http://127.0.0.1:1",
            model="m",
            prompt="p",
            max_tokens=4,
            temperature=None,
            timeout=5.0,
            speculative_mtp=None,
        )
    finally:
        module.urllib.request.urlopen = original

    assert captured[0]["speculative_mtp"] is False
    assert captured[1]["speculative_mtp"] is True
    assert "speculative_mtp" not in captured[2]


def test_summary_reports_group_composition_and_per_request_decode_rate() -> None:
    module = _module()
    wide = module._request_row({"source_id": 1, "prompt_tokens": 900}, _response())
    wide["completion_tokens"] = 48
    wide["mtp_used"] = True
    wide["mtp_output_tokens"] = 30
    wide["realized_group_rows"] = 2
    wide["e2e_ms"] = 1000.0
    wide["ttft_ms"] = 200.0
    narrow = module._request_row({"source_id": 2, "prompt_tokens": 900}, _response())
    narrow["completion_tokens"] = 48
    narrow["mtp_used"] = False
    narrow["mtp_output_tokens"] = 0
    narrow["realized_group_rows"] = 2
    narrow["e2e_ms"] = 1000.0
    narrow["ttft_ms"] = 200.0

    summary = module.summarize([wide, narrow])

    groups = summary["groups_by_realized_rows"]
    assert groups["2"]["requests"] == 2
    assert groups["2"]["mtp_requests"] == 1
    assert groups["2"]["mtp_output_tokens"] == 30
    # 48 tokens in 800 ms after the first token, per request.
    assert summary["median_decode_tokens_per_second_per_request"] == pytest.approx(60.0)
    assert summary["decode_requests"] == 2


def test_min_prompt_len_selects_the_boundary_crossing_row(monkeypatch) -> None:
    module = _module()
    dataset = Path("/tmp/sharegpt-recorder-min-len.json")
    dataset.write_text(
        json.dumps(
            [
                {
                    "id": "short",
                    "conversations": [
                        {"from": "human", "value": "hello there my friend"},
                        {"from": "gpt", "value": "hi there my friend"},
                    ],
                },
                {
                    "id": "long",
                    "conversations": [
                        {"from": "human", "value": "word " * 900},
                        {"from": "gpt", "value": "answer " * 40},
                    ],
                },
            ]
        )
    )
    monkeypatch.setattr(
        module,
        "_tokenizer",
        lambda gguf: type(
            "Tokenizer",
            (),
            {"encode": staticmethod(lambda text: list(range(len(text.split()))))},
        )(),
    )
    tokenizer = module._tokenizer(dataset)

    all_samples = module.load_samples(
        dataset,
        tokenizer=tokenizer,
        count=2,
        seed=0,
        output_len=None,
        max_prompt_len=1024,
        max_total_len=2048,
        min_prompt_len=4,
    )
    long_only = module.load_samples(
        dataset,
        tokenizer=tokenizer,
        count=1,
        seed=0,
        output_len=None,
        max_prompt_len=1024,
        max_total_len=2048,
        min_prompt_len=512,
    )

    assert sorted(sample["source_id"] for sample in all_samples) == ["long", "short"]
    assert [sample["source_id"] for sample in long_only] == ["long"]


def test_cancel_after_tokens_marks_a_deliberate_abort() -> None:
    """A client cancel is a cancel, not a failure or an empty success row."""

    module = _module()
    chunks = [
        b'data: {"choices": [{"delta": {"content": "a"}, "hipengine": {"decode_state": {"generated_tokens": 1}}}]}\n',
        b'data: {"choices": [{"delta": {"content": "b"}, "hipengine": {"decode_state": {"generated_tokens": 5}}}]}\n',
        b'data: {"choices": [{"delta": {"content": "c"}, "hipengine": {"decode_state": {"generated_tokens": 9}}}]}\n',
        b"data: [DONE]\n",
    ]

    class _Response:
        def __enter__(self):
            return iter(chunks)

        def __exit__(self, *exc):
            return False

    original = module.urllib.request.urlopen
    module.urllib.request.urlopen = lambda request, timeout=None: _Response()
    try:
        result = module._stream_request(
            "http://127.0.0.1:1",
            model="m",
            prompt="p",
            max_tokens=64,
            temperature=None,
            timeout=5.0,
            cancel_after_tokens=4,
        )
    finally:
        module.urllib.request.urlopen = original

    assert result["cancelled"] is True
    # The stream was closed after the crossing chunk, so token 9 never arrived.
    assert [event[1] for event in result["timeline"]] == [1, 5]
    row = module._request_row({"source_id": 3, "prompt_tokens": 100}, result)
    assert row["cancelled"] is True
    assert row["error"].startswith("stream ended without usage")


def test_summary_totals_are_not_shadowed_by_the_per_request_decode_loop() -> None:
    """Three rows of 48 tokens are 144, not the last row's 48."""

    module = _module()
    rows = []
    for index, source_id in enumerate(("a", "b", "c")):
        row = module._request_row({"source_id": source_id, "prompt_tokens": 100}, _response())
        row["completion_tokens"] = 48
        row["mtp_output_tokens"] = 46
        row["ar_output_tokens"] = 2
        row["mtp_used"] = True
        row["e2e_ms"] = 1000.0 + index
        row["ttft_ms"] = 200.0
        rows.append(row)

    summary = module.summarize(rows)

    assert summary["completion_tokens"] == 144
    assert summary["mtp_output_tokens"] == 138
    assert summary["mtp_output_share"] == pytest.approx(138 / 144)
    assert summary["ar_output_tokens"] == 6


def test_inter_token_latency_survives_a_buffered_commit_burst() -> None:
    """A speculative cycle's tokens arrive buffered; rate must not be fooled.

    The live c=2 artifact reported a 0.29 ms median because consecutive decode
    reports inside a cycle are microseconds apart while the wait for the next
    cycle is reported nowhere. Windowed rate cannot be faked that way.
    """

    module = _module()
    # Four cycles of four tokens. Each cycle waits 200 ms and then delivers its
    # four reports within 0.4 ms of each other: 50 ms of real cost per token.
    timeline = [[0.0, 0]]
    now = 0.0
    tokens = 0
    for _ in range(4):
        now += 200.0
        for _ in range(4):
            tokens += 1
            timeline.append([now, tokens])
            now += 0.1

    intervals = module._inter_token_latencies_ms(timeline)

    assert intervals
    assert all(interval == pytest.approx(50.0, abs=5.0) for interval in intervals)
    assert module._decode_ms_per_token(timeline) == pytest.approx(50.0, abs=1.0)


def test_inter_token_latency_exposes_a_stalled_window() -> None:
    """A stall must move the tail, not vanish into one big interval."""

    module = _module()
    timeline = [[float(ms), tokens] for ms, tokens in [(0, 0), (100, 16), (200, 32), (5000, 48), (5100, 64)]]

    intervals = module._inter_token_latencies_ms(timeline)

    assert module._percentile(intervals, 100.0) == pytest.approx(4800.0 / 16, abs=1.0)
    assert max(intervals) > 3 * module._percentile(intervals, 50.0)


def test_decode_ms_per_token_uses_the_whole_timeline() -> None:
    """The aggregate is the honest cross-check on a burst-amortized median."""

    module = _module()
    timeline = [[100.0, 1], [300.0, 2], [300.4, 3], [300.8, 4], [301.2, 5]]

    assert module._decode_ms_per_token(timeline) == pytest.approx(201.2 / 4)
    assert module._decode_ms_per_token([]) is None
    assert module._decode_ms_per_token([[100.0, 1], [200.0, 1]]) is None


def test_request_row_reports_itl_and_output_digest() -> None:
    module = _module()
    result = _response()
    result["output_sha256"] = "deadbeef"
    result["timeline"] = [[100.0, 1], [300.0, 5]]

    row = module._request_row({"source_id": 4, "prompt_tokens": 100}, result)

    assert row["itl_median_ms"] == pytest.approx(50.0)
    assert row["itl_samples"] == 1
    assert row["output_sha256"] == "deadbeef"


def test_summary_reports_serving_metrics_and_output_identity() -> None:
    module = _module()
    rows = []
    for index, source_id in enumerate(("a", "b")):
        # 200 ms for 4 committed tokens is 50 ms per token, and it is the
        # timeline the row's own latency is derived from.
        row = module._request_row(
            {"source_id": source_id, "prompt_tokens": 100},
            _response(timeline=[[200.0, 1], [400.0, 5]]),
        )
        row["completion_tokens"] = 48
        row["mtp_output_tokens"] = 0
        row["e2e_ms"] = 1000.0 + index * 100.0
        row["ttft_ms"] = 200.0
        row["output_sha256"] = "same" if index == 0 else "different"
        rows.append(row)

    summary = module.summarize(rows)

    assert summary["median_e2e_ms"] == pytest.approx(1050.0)
    assert summary["p95_e2e_ms"] == pytest.approx(1095.0)
    assert summary["median_itl_ms"] == pytest.approx(50.0)
    assert summary["itl_samples"] == 2
    assert summary["output_hashes"] == {"different": 1, "same": 1}
    assert summary["duplicate_output_hashes"] == 0


def test_case_mix_selects_a_short_row_beside_a_boundary_crossing_row() -> None:
    """One load must be able to hold both sides of the provider's window."""

    module = _module()
    dataset = Path("/tmp/sharegpt-recorder-case-mix.json")
    dataset.write_text(
        json.dumps(
            [
                {
                    "id": "short",
                    "conversations": [
                        {"from": "human", "value": "a " * 10},
                        {"from": "gpt", "value": "b " * 10},
                    ],
                },
                {
                    "id": "mid",
                    "conversations": [
                        {"from": "human", "value": "a " * 200},
                        {"from": "gpt", "value": "b " * 10},
                    ],
                },
                {
                    "id": "long",
                    "conversations": [
                        {"from": "human", "value": "a " * 700},
                        {"from": "gpt", "value": "b " * 10},
                    ],
                },
            ]
        )
    )
    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(
            module,
            "_tokenizer",
            lambda gguf: type(
                "Tokenizer",
                (),
                {"encode": staticmethod(lambda text: list(range(len(text.split()))))},
            )(),
        )
        tokenizer = module._tokenizer(dataset)
        samples = module.load_samples(
            dataset,
            tokenizer=tokenizer,
            count=2,
            seed=0,
            output_len=None,
            max_prompt_len=1024,
            max_total_len=2048,
            cases=["short", "long"],
        )
    finally:
        monkeypatch.undo()

    assert [(sample["source_id"], sample["case"]) for sample in samples] == [
        ("short", "short"),
        ("long", "long"),
    ]


def test_summary_splits_by_arm_and_case_and_reports_identity() -> None:
    """The reviewer's matrix: two arms and two cases in one load, one report."""

    module = _module()

    def row(arm, case, ids, sha, *, mtp=False):
        return {
            "source_id": f"{case}-prompt",
            "case": case,
            "route_arm": arm,
            "repeat_index": 0,
            "prompt_tokens": 600 if case == "long" else 20,
            "completion_tokens": 32,
            "ttft_ms": 500.0,
            "e2e_ms": 4000.0,
            "itl_median_ms": 90.0,
            "output_sha256": sha,
            "generated_token_ids": ids,
            "mtp_used": mtp,
            "mtp_output_tokens": 20 if mtp else 0,
            "fallback_reason": None if mtp else "automatic_route_ar",
            "timeline": [[0.0, 1], [90.0, 33]],
        }

    rows = [
        row("off", "short", [1, 2, 3, 4], "sha-ar-short"),
        row("auto", "short", [1, 2, 3, 4], "sha-ar-short", mtp=True),
        row("off", "long", [1, 2, 3, 4], "sha-ar-long"),
        row("auto", "long", [1, 2, 9, 4], "sha-mtp-long", mtp=True),
    ]
    summary = module.summarize(rows)

    assert set(summary["by_route_arm"]) == {"off", "auto"}
    assert set(summary["by_case"]) == {"short", "long"}
    assert summary["by_route_arm"]["off"]["mtp_requests"] == 0
    assert summary["by_route_arm"]["auto"]["mtp_requests"] == 2
    assert summary["by_case"]["long"]["median_prompt_tokens"] == 600
    assert summary["by_case"]["short"]["median_prompt_tokens"] == 20

    identity = summary["arm_identity"]
    assert identity["compared_prompts"] == 2
    assert identity["identical_prompts"] == 1
    assert identity["differing_prompts"] == 1
    assert identity["min_shared_prefix_tokens"] == 2
    assert identity["differing"][0]["case"] == "long"
    assert identity["differing"][0]["left_arm"] == "off"
    assert identity["differing"][0]["right_arm"] == "auto"
    json.dumps(summary)


def test_arm_identity_falls_back_to_the_digest_without_token_ids() -> None:
    module = _module()
    rows = [
        {
            "source_id": "p",
            "case": None,
            "route_arm": "off",
            "repeat_index": 0,
            "completion_tokens": 8,
            "e2e_ms": 100.0,
            "output_sha256": "same",
            "generated_token_ids": None,
        },
        {
            "source_id": "p",
            "case": None,
            "route_arm": "auto",
            "repeat_index": 0,
            "completion_tokens": 8,
            "e2e_ms": 100.0,
            "output_sha256": "same",
            "generated_token_ids": None,
        },
    ]
    identity = module.summarize(rows)["arm_identity"]
    assert identity["compared_prompts"] == 1
    assert identity["compared_prompts_with_ids"] == 0
    assert identity["identical_prompts"] == 1
    assert identity["median_shared_prefix_tokens"] is None
