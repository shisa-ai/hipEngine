"""The isolation and task-quality gates fail loudly instead of reporting 200s."""
import json

import httpx

from scripts.qwen4exp_http_isolation_gate import _chat, _check, _concurrent
from scripts.qwen4exp_http_task_quality_gate import _load


def test_check_records_failure_with_detail():
    checks: list[dict] = []
    _check(checks, "repeat.identical", False, texts=["a", "b"])
    assert checks == [
        {"name": "repeat.identical", "passed": False, "texts": ["a", "b"]}
    ]


def test_chat_reports_a_refusal_instead_of_a_missing_answer():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": {"code": "resident_prepare_failed"}})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = _chat(client, "http://fixture", "m", "hello")
    assert result["status"] == 503
    assert result["body"]["error"]["code"] == "resident_prepare_failed"
    assert "text" not in result


def test_chat_carries_ids_and_usage_from_a_served_answer():
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["temperature"] == 0.0
        assert payload["max_tokens"] == 24
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"content": "Paris"},
                        "finish_reason": "stop",
                    }
                ],
                "hipengine": {"generated_token_ids": [11, 12]},
                "usage": {"completion_tokens": 2},
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = _chat(client, "http://fixture", "m", "capital?", max_tokens=24)
    assert result["text"] == "Paris"
    assert result["ids"] == [11, 12]
    assert result["usage"]["completion_tokens"] == 2


def test_concurrent_keeps_every_result_in_request_order():
    import time

    def fake_call(client, url, model, content, **extra):
        # Later calls finish first, so a result list keyed by completion order
        # would come back reversed.
        time.sleep(0.05 * (3 - len(content)))
        return {"status": 200, "text": content.upper(), "ids": [len(content)]}

    calls = [{"content": "a"}, {"content": "bb"}, {"content": "ccc"}]
    results = _concurrent(calls, "http://fixture", "m", 5.0, call_fn=fake_call)
    assert [item["text"] for item in results] == ["A", "BB", "CCC"]


def test_concurrent_records_a_raised_client_error_as_a_failure():
    def boom(client, url, model, content, **extra):
        raise RuntimeError("connection reset")

    results = _concurrent([{"content": "a"}], "http://fixture", "m", 5.0, call_fn=boom)
    assert results[0]["status"] == 0
    assert "connection reset" in results[0]["error"]


def test_task_quality_loader_skips_blank_lines(tmp_path):
    path = tmp_path / "suite.jsonl"
    path.write_text(
        json.dumps({"id": "a", "category": "code", "messages": []}) + "\n\n"
        + json.dumps({"id": "b", "category": "general_ja", "messages": []}) + "\n",
        encoding="utf-8",
    )
    rows = _load(path)
    assert [row["id"] for row in rows] == ["a", "b"]
