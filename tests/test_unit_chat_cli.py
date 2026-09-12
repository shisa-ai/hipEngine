from __future__ import annotations

import io
import json
from types import SimpleNamespace

import pytest

from hipengine.chat_cli import build_parser, run


def test_chat_parser_defaults_to_local_server() -> None:
    args = build_parser().parse_args([])
    assert args.server == "http://127.0.0.1:8000"
    assert args.model is None
    assert args.max_tokens == 512


def test_chat_fails_clearly_without_a_running_server(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args, **kwargs):
        raise OSError("connection refused")

    monkeypatch.setattr("hipengine.chat_cli.urlopen", fail)
    error = io.StringIO()
    result = run(build_parser().parse_args([]), input_stream=io.StringIO(), output_stream=io.StringIO())
    assert result == 1


def test_chat_discovers_model_and_streams_completion(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = [
        {"data": [{"id": "local-model"}]},
        {
            "context": {"effective_max_context_tokens": 176128},
            "queue": {"max_active_requests": 4},
            "kv_capacity": {
                "storage": "int8_per_token_head",
                "scale_dtype": "fp32",
                "pool": {"current_bytes": 1024, "budget_bytes": 4096},
            },
        },
    ]

    class Response:
        def __init__(self, payload):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def __iter__(self):
            yield b'data: {"choices":[{"delta":{"content":"hello"}}]}\n'
            yield b'data: {"choices":[{"delta":{"content":" world"}}]}\n'
            yield b"data: [DONE]\n"

    class JsonResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(responses.pop(0)).encode()

    def fake_urlopen(request, timeout=None):
        if request.full_url.endswith("/v1/models") or request.full_url.endswith("/ready"):
            return JsonResponse()
        return Response({})

    monkeypatch.setattr("hipengine.chat_cli.urlopen", fake_urlopen)
    output = io.StringIO()
    result = run(
        build_parser().parse_args([]),
        input_stream=io.StringIO("hello\n/quit\n"),
        output_stream=output,
    )
    assert result == 0
    assert "context=176128" in output.getvalue()
    assert "assistant> hello world" in output.getvalue()
