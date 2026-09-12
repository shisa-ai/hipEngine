import pytest

from scripts.qwen38_serving_closure import stream_summary, validate_response, payload


def test_stream_requires_done_and_terminal_without_error():
    rows = [
        {"choices": [{"text": "a", "finish_reason": None}]},
        {"choices": [{"text": "", "finish_reason": "length"}]},
        "[DONE]",
    ]
    assert stream_summary(rows)["text"] == "a"
    assert stream_summary(rows)["complete"]
    assert not stream_summary(rows[:-1])["complete"]
    assert not stream_summary(rows + [{"error": {"message": "failed"}}])["complete"]


def test_blocking_requires_exact_server_owned_ids_and_count():
    response = {
        "choices": [{"text": "ok", "finish_reason": "length"}],
        "usage": {"completion_tokens": 2},
        "hipengine": {"token_accounting": {
            "choice_generated_token_ids": [[1, 2]], "total_generated_tokens": 2,
        }},
    }
    assert validate_response(response)["ids"] == [1, 2]
    response["usage"]["completion_tokens"] = 3
    with pytest.raises(ValueError):
        validate_response(response)


def test_stream_uses_authoritative_terminal_ids_without_resident_request_id():
    events = [
        {"choices": [{"text": "ok", "finish_reason": None}]},
        {"choices": [{"text": "", "finish_reason": "length", "hipengine": {
            "generated_token_ids": [1, 2], "generated_tokens": 2,
        }}]},
        "[DONE]",
    ]
    assert stream_summary(events)["ids"] == [1, 2]


def test_automatic_mtp_payload_omits_request_override():
    assert "speculative_mtp" not in payload([1, 2], 25, mtp=None)
    assert payload([1, 2], 25, mtp=False)["speculative_mtp"] is False
