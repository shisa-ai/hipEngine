"""The HTTP audit records route evidence without turning HTTP 200 into a pass."""
import asyncio
import json

import httpx

from scripts.qwen4exp_http_grid import compact, request


def test_compact_preserves_fallback_and_exact_ids_without_large_prompt():
    row = {
        "name": "requested-mtp", "status": 200, "wall_s": 1.0,
        "request": {"prompt": [123] * 30000, "speculative_mtp": True},
        "body": {"hipengine": {
            "speculative_mtp": {"used": False, "decision_reason": "unsupported_sampling_k0"},
            "token_accounting": {"choice_generated_token_ids": [[1, 2]]},
            "generation_shape": {"route": "default"}}, "choices": []},
    }
    result = compact(row)
    assert result["hipengine"] == row["body"]["hipengine"]
    assert result["request_options"] == {"speculative_mtp": True}
    assert len(result["request_sha256"]) == 64
    assert "request" not in result and "body" not in result
    assert "passed" not in result


def test_probe_records_rejection_body(tmp_path):
    async def run():
        transport = httpx.MockTransport(lambda req: httpx.Response(
            400, json={"error": {"code": "unsupported_parameter"}}))
        async with httpx.AsyncClient(transport=transport, base_url="http://fixture") as client:
            return await request(client, tmp_path, "blocked", {"speculative": True})
    result = asyncio.run(run())
    assert result["status"] == 400
    assert result["error"]["code"] == "unsupported_parameter"
    raw = json.loads((tmp_path / "blocked.json").read_text())
    assert raw["request"] == {"speculative": True}
    assert raw["body"]["error"] == result["error"]


def test_probe_keeps_non_json_error(tmp_path):
    async def run():
        transport = httpx.MockTransport(lambda req: httpx.Response(500, text="Internal Server Error"))
        async with httpx.AsyncClient(transport=transport, base_url="http://fixture") as client:
            return await request(client, tmp_path, "plain-error", {})
    result = asyncio.run(run())
    assert result["status"] == 500
    assert result["unparsed_body"] == "Internal Server Error"
    assert "transport_error" not in result


def test_probe_keeps_stream_events_and_errors(tmp_path):
    async def run():
        content = 'data: {"error":{"code":"generation_failed"}}\n\ndata: [DONE]\n\n'
        transport = httpx.MockTransport(lambda req: httpx.Response(200, text=content))
        async with httpx.AsyncClient(transport=transport, base_url="http://fixture") as client:
            return await request(client, tmp_path, "sse", {"stream": True})
    result = asyncio.run(run())
    assert result["status"] == 200
    assert result["events"][0]["data"]["error"]["code"] == "generation_failed"
    assert result["events"][1]["data"] == "[DONE]"
    assert result["event_count"] == 2
    assert result["events"][0]["elapsed_s"] <= result["events"][1]["elapsed_s"]
