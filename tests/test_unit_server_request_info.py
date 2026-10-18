from __future__ import annotations

import logging

import pytest

from hipengine.server.api import (
    ChatCompletionRequest,
    CompletionRequest,
    ServerConfig,
    _log_request_info,
)

_LOGGER_NAME = "uvicorn.error"


def _info_line(caplog, *, config: ServerConfig, request, **kwargs) -> str:
    with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
        _log_request_info(config=config, request=request, **kwargs)
    lines = [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("REQUEST_INFO:")
    ]
    assert len(lines) == 1, lines
    return lines[0]


def _config(**overrides) -> ServerConfig:
    values = {
        "model": "/models/fake",
        "served_model_name": "fake-model",
        "info": True,
    }
    values.update(overrides)
    return ServerConfig(**values)


def _streaming_chat_request(**overrides) -> ChatCompletionRequest:
    values = {
        "model": "fake-model",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    }
    values.update(overrides)
    return ChatCompletionRequest(**values)


def test_info_line_reports_a_stream_window_rate_not_engine_throughput(caplog) -> None:
    """A streamed rate is named for the server's window, never the engine's span.

    Engine telemetry inside a stream carries ``request_total_ms`` as a
    prefill-time snapshot, so reading it as a decode span would report a
    throughput the request never reached.
    """

    line = _info_line(
        caplog,
        config=_config(),
        request=_streaming_chat_request(),
        usage={"prompt_tokens": 2_000, "completion_tokens": 45},
        timing={
            "prefill_ms": 101_005.6,
            "ttft_ms": 106_190.0,
            "decode_elapsed_ms": 168.8,
            "decode_tokens_per_second": 266.62,
            # Prefill-time snapshot: using it as a decode span would be wrong.
            "request_total_ms": 106_010.0,
        },
    )

    assert "prefill_ms=101005.6" in line
    assert "prefill_tok_s=19.8" in line
    assert "ttft_ms=106190.0" in line
    assert "stream_decode_ms=168.8" in line
    assert "stream_tok_s=266.62" in line
    assert " decode_tok_s=" not in line
    assert " decode_ms=" not in line


def test_info_line_derives_a_blocking_decode_rate_from_engine_phases(caplog) -> None:
    """A blocking request has no first-token timestamp, so the engine reports it."""

    line = _info_line(
        caplog,
        config=_config(),
        request=CompletionRequest(model="fake-model", prompt="hi", max_tokens=51),
        usage={"prompt_tokens": 49, "completion_tokens": 51},
        timing={"prefill_ms": 345.2, "request_total_ms": 5_442.6},
        wall_ms=5_500.0,
    )

    assert "prefill_ms=345.2" in line
    assert "decode_ms=5097.4" in line
    assert "decode_tok_s=10.01" in line
    assert "wall_ms=5500.0" in line
    assert "stream_tok_s=" not in line
    assert "stream_decode_ms=" not in line
    assert "ttft_ms=" not in line


def test_info_line_omits_phases_the_backend_does_not_report(caplog) -> None:
    """Absent phases are omitted rather than logged as zero."""

    line = _info_line(
        caplog,
        config=_config(),
        request=CompletionRequest(model="fake-model", prompt="hi", max_tokens=4),
        usage={"prompt_tokens": 3, "completion_tokens": 2},
        timing={},
    )

    assert "tokens_in=3" in line
    assert "tokens_out=2" in line
    for absent in ("prefill_ms=", "prefill_tok_s=", "decode_ms=", "decode_tok_s=", "ttft_ms="):
        assert absent not in line


def test_info_line_stays_silent_without_the_flag(caplog) -> None:
    with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
        _log_request_info(
            config=_config(info=False),
            request=CompletionRequest(model="fake-model", prompt="hi", max_tokens=4),
            usage={"prompt_tokens": 3, "completion_tokens": 2},
            timing={"prefill_ms": 1.0, "request_total_ms": 2.0},
        )

    assert "REQUEST_INFO:" not in caplog.text


def test_info_line_never_logs_prompt_or_generated_text(caplog) -> None:
    line = _info_line(
        caplog,
        config=_config(),
        request=ChatCompletionRequest(
            model="fake-model",
            messages=[{"role": "user", "content": "SECRET-PROMPT-TEXT"}],
        ),
        usage={"prompt_tokens": 3, "completion_tokens": 2},
        timing={"prefill_ms": 1.0, "request_total_ms": 2.0},
    )

    assert "SECRET-PROMPT-TEXT" not in line


def test_info_line_reports_the_requests_own_kv_allocation(caplog) -> None:
    line = _info_line(
        caplog,
        config=_config(),
        request=CompletionRequest(model="fake-model", prompt="hi", max_tokens=4),
        usage={"prompt_tokens": 3, "completion_tokens": 2},
        timing={"prefill_ms": 1.0, "request_total_ms": 2.0},
        kv_request_bytes=16 * 2**20,
        kv_pool={
            "current_bytes": 66 * 2**30,
            "current_pages": 4224,
            "pinned_pages": 4096,
            "grow_events": 0,
        },
    )

    assert "kv_request_alloc_mib=16.0" in line
    assert "kv_pool_gib=66.00" in line
    assert "kv_pool_pages=4224" in line
    assert "kv_pool_pinned_pages=4096" in line
    assert "kv_pool_grows=0" in line
