"""CPU checks for common public speculative-stream measurements."""
from types import SimpleNamespace as NS

import pytest

from scripts.qwen38_public_comparison import (
    method_order, summarize, consume_stream, validate_group,
)


def chunk(text="x", *, final=False, ids=None):
    return NS(text=text, finish_details=NS(reason="length") if final else None,
              generated_token_ids=ids, telemetry=None)


def test_order_rotates_every_method_through_each_position():
    orders = [method_order(i) for i in range(4)]
    for pos in range(4):
        assert {order[pos] for order in orders} == {"ar", "mtp", "eagle4", "dspine"}
    assert method_order(4) == orders[0]


@pytest.mark.parametrize("separate_terminal", [False, True])
def test_stream_timing_preserves_empty_token_chunks_and_final_ids(separate_terminal):
    events = [chunk(), chunk("")]
    if separate_terminal:
        events.extend([chunk(), chunk("", final=True, ids=(1, 2, 3))])
        ticks = iter([10, 12, 14, 16, 17, 18])
    else:
        events.append(chunk(final=True, ids=(1, 2, 3)))
        ticks = iter([10, 12, 14, 16, 18])
    row = consume_stream(iter(events), expected_tokens=3, clock=lambda: next(ticks))
    assert row["output_ids"] == [1, 2, 3]
    assert row["wall_s"] == 8
    assert row["first_token_s"] == row["first_text_s"] == 2
    assert row["token_arrival_s"] == [2, 4, 6]
    assert row["arrival_decode_s"] == 4
    assert row["transition_tokens"] == 2


@pytest.mark.parametrize("events", [
    [chunk(final=True, ids=(1, 2))],
    [chunk(), chunk(), chunk()],
    [chunk(), chunk(), chunk(final=True, ids=(1, 2))],
    [chunk(), chunk(), chunk(final=True, ids=(1, 2, 3)), chunk("", final=True, ids=(1, 2, 3))],
])
def test_incomplete_buffered_or_duplicate_terminal_streams_fail(events):
    with pytest.raises(ValueError):
        consume_stream(iter(events), expected_tokens=3)


def test_stream_is_closed_after_failure():
    closed = []
    def stream():
        try:
            yield chunk()
            raise RuntimeError("request failed")
        finally:
            closed.append(True)
    with pytest.raises(RuntimeError, match="request failed"):
        consume_stream(stream(), expected_tokens=3)
    assert closed == [True]


def test_summary_pools_counts_and_times_not_prompt_rates():
    rows = [dict(method="ar", category="code", output_ids=[1, 2, 3], wall_s=t,
                 transition_tokens=2, arrival_decode_s=t, first_text_s=1,
                 first_token_s=1, exact=True) for t in (1, 9)]
    result = summarize(rows)["ar"]["all"]
    assert result["request_tok_s"] == pytest.approx(6 / 10)
    assert result["arrival_transition_tok_s"] == pytest.approx(4 / 10)
    assert result["exact_all"]


def test_route_checks_use_terminal_request_cycles_not_stale_batch_snapshot(monkeypatch):
    from scripts import qwen38_public_comparison as bench
    telemetry = {"decode_state": {"execution_path": "gguf_specdec2_mtp2"},
                 "diagnostics": {"specdec2_mtp2": {
                     "cycles": 14, "generated_draft_tokens": 42, "candidate_budget": 3,
                     "selected_depth_histogram": {"3": 14}}}}
    owner = NS(last_batch_generation={"path": "gguf_packed_ar_server_decode"})
    llm = NS(_get_text_generator=lambda: owner,
             stream_speculative_mtp_detailed=lambda *a: None)
    monkeypatch.setattr(bench, "hardware_snapshot", lambda: {})
    monkeypatch.setattr(bench, "consume_stream", lambda *a, **kw: {"telemetry": telemetry})
    assert bench.run_one(llm, "mtp", (1,), 33)["intended_route_ran"]
    # The output-room tail can shorten the final draft without changing the
    # requested maximum budget or silently disabling the feature.
    telemetry["diagnostics"]["specdec2_mtp2"]["selected_depth_histogram"] = {"2": 1, "3": 13}
    assert bench.run_one(llm, "mtp", (1,), 33)["intended_route_ran"]
    telemetry["diagnostics"]["specdec2_mtp2"]["cycles"] = 0
    assert not bench.run_one(llm, "mtp", (1,), 33)["intended_route_ran"]


def test_owner_details_include_leased_resident_sessions():
    from scripts.qwen38_public_comparison import owner_details
    session = NS(target_verifier_production_q4_rowtile=True, runner=NS(fp16_recurrent_state=False),
                 use_wmma_prefill=True, use_gemv_decode=True, kv_storage_dtype="bf16")
    owner = NS(_shared_session_pool={}, _resident_model_runner=NS(_resident_sessions=lambda: (session,)),
               execution_profile="production")
    llm = NS(_get_text_generator=lambda: owner, execution_profile_manifest={},
             execution_profile_manifest_sha256="test", speculative_capabilities={}, speculative_candidate_budget=3)
    assert owner_details(llm)["sessions"][0]["target_verifier_production_q4_rowtile"] is True


def test_exactness_rejects_short_matching_prefix_and_repeat_drift():
    previous = {}
    good = [dict(method=m, output_ids=[1, 2, 3]) for m in method_order(0)]
    validate_group(good, previous, "p", expected_tokens=3)
    assert all(r["exact"] and r["repeat_exact"] for r in good)
    short = [dict(r) for r in good]
    short[-1]["output_ids"] = [1, 2]
    with pytest.raises(ValueError, match="exact"):
        validate_group(short, previous, "p", expected_tokens=3)
    changed = [dict(method=m, output_ids=[1, 2, 4]) for m in method_order(0)]
    with pytest.raises(ValueError, match="exact"):
        validate_group(changed, previous, "p", expected_tokens=3)


def test_production_records_cross_route_difference_without_hiding_repeat_failure():
    previous = {}
    rows = [dict(method=m, output_ids=[1, 2, 3 if m == "ar" else 4]) for m in method_order(0)]
    validate_group(rows, previous, "p", expected_tokens=3, execution_profile="production")
    assert all(r["repeat_exact"] for r in rows)
    assert rows[0]["exact"] and rows[0]["first_difference_vs_ar"] is None
    assert all(not r["exact"] and r["first_difference_vs_ar"] == 2 for r in rows[1:])
    validate_group(rows, previous, "p", expected_tokens=3, execution_profile="production")
    rows[-1]["output_ids"] = [1, 2, 5]
    with pytest.raises(ValueError, match="repeat exactness"):
        validate_group(rows, previous, "p", expected_tokens=3, execution_profile="production")


def test_strict_comparison_still_rejects_cross_route_difference():
    rows = [dict(method=m, output_ids=[1, 2, 3 if m == "ar" else 4]) for m in method_order(0)]
    with pytest.raises(ValueError, match="strict cross-method exactness"):
        validate_group(rows, {}, "p", expected_tokens=3, execution_profile="strict")
