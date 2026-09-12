import pytest

from scripts.qwen38_serving_closure import stream_summary, validate_response


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
