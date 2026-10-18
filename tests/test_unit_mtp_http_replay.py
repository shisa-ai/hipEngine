"""HTTP replay checks actual speculative execution and keeps channels separate."""
import json

from scripts.mtp_http_replay import check_row, explicit_mtp, summarize


def test_stream_summary_preserves_reasoning_and_terminal_metadata():
    events = [
        {"choices": [{"index": 0, "delta": {"reasoning_content": "think"}}]},
        {"choices": [{"index": 0, "delta": {"content": "answer"}}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "length", "hipengine": {"speculative_mtp": {"used": True}}}]},
        {"choices": [], "usage": {"completion_tokens": 2}},
    ]
    response = summarize({"sse_raw": "\n\n".join("data: " + json.dumps(event) for event in events) + "\n\ndata: [DONE]\n"})
    assert response["content"] == "answer"
    assert response["reasoning_content"] == "think"
    assert response["hipengine"]["speculative_mtp"]["used"] is True
    assert response["usage"] == {"completion_tokens": 2}


def test_http_success_without_mtp_is_a_failed_explicit_check():
    row = {"status": 200, "request": {"speculative_mtp": True}, "summary": summarize({"hipengine": {"speculative_mtp": {"used": False}}})}
    assert check_row(row) == ["explicit_mtp_did_not_run"]
    row["request"]["speculative_mtp"] = False
    assert check_row(row) == []


def test_stream_error_cannot_pass_as_mtp_success():
    row = {"status": 200, "request": {"speculative_mtp": True}, "summary": summarize({"sse_raw": 'data: {"error":{"code":"unsupported_parameter"}}\n\ndata: [DONE]\n'})}
    assert check_row(row) == ["request_failed"]
    assert explicit_mtp({"speculative": True})
    assert explicit_mtp({"speculative_mtp": {"enabled": True}})
    assert not explicit_mtp({"speculative_mtp": {"enabled": False}})
