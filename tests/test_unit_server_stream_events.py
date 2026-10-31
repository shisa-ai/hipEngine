"""SSE frame projection and stream accounting golden contract.

The fixture was captured from ``f2db17d3f`` before the stream projection owner
was extracted.  It pins the exact SSE bytes, tracker/accounting mutation state,
and the default finish/error wrappers that stay in the application module.
"""

from __future__ import annotations

import ast
from dataclasses import asdict
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from hipengine.generation import (
    DecodeState,
    FinishDetails,
    GenerationStreamChunk,
    GenerationTelemetry,
    TokenLogprob,
)
from hipengine.server import api, stream_events
from hipengine.server.chat_output import _ParsedToolCall

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "server_stream_events_golden.json"
FIXTURE = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
CLOCK = float(FIXTURE["clock"])
ROUTING = {"backend": "hip_gfx1100", "model": "fixture"}
KV_POOL = {"persistent_total_bytes": 1024, "persistent_used_bytes": 512}
FINISH_DETAILS = {"reason": "length", "length_limit": 2, "cache_action": "append_none"}
EXTRA_HIPENGINE = {
    "generation_shape": {"route": "speculative"},
    "speculative_mtp": {"used": True, "draft_tokens": 3},
}
COMPLETION_LOGPROBS = {
    "tokens": [{"token": "A", "logprob": -0.25}],
    "token_logprobs": [-0.25],
}
CHAT_LOGPROBS = {
    "content": [
        {"token": "A", "logprob": -0.25, "bytes": [65], "top_logprobs": []}
    ],
    "refusal": None,
}
REASONING_LOGPROBS = {
    "content": [
        {
            "token": "why",
            "logprob": -0.5,
            "bytes": [119, 104, 121],
            "top_logprobs": [],
        }
    ],
    "public_text": "why",
    "refusal": None,
}
TOOL_CALL = _ParsedToolCall(
    id="call_fixed",
    name="bash",
    arguments='{"command":"pwd"}',
)


def _telemetry() -> GenerationTelemetry:
    return GenerationTelemetry(
        decode_state=DecodeState(
            request_id="req-fixed",
            row_index=0,
            step_index=3,
            prompt_tokens=5,
            generated_tokens=3,
            phase="answer",
            reasoning_tokens=1,
            answer_tokens=2,
            execution_path="qwen4exp_mtp_batched_state_rows_verify",
            continuation_eligible=True,
        ),
        timing={"decode_ms": 1.25, "prefill_ms": 2.5, "backend_custom": 3.0},
        usage={"prompt_tokens": 5, "completion_tokens": 3},
    )


def _chunk() -> GenerationStreamChunk:
    return GenerationStreamChunk(
        text="A",
        token_logprobs=(TokenLogprob(token_id=7, token_text="A", logprob=-0.25),),
        telemetry=_telemetry(),
        generated_token_ids=(7,),
    )


def _finish_chunk() -> GenerationStreamChunk:
    return GenerationStreamChunk(
        text="",
        telemetry=_telemetry(),
        generated_token_ids=(7,),
    )


def _buffered_source_chunk() -> GenerationStreamChunk:
    return GenerationStreamChunk(
        text="A",
        telemetry=_telemetry(),
        finish_details=FinishDetails(reason="length", length_limit=2),
    )


def _tracker(
    *, streaming: bool = True, buffered: bool = False
) -> stream_events.StreamTimingTracker:
    tracker = stream_events.StreamTimingTracker(started_at=0.0)
    if streaming:
        tracker.first_token_at = 1.0
        tracker.first_chunk_tokens = 2
        tracker.last_token_at = 4.0
        tracker.last_chunk_tokens = 10
        tracker.observed_tokens = 10
    tracker.buffered = buffered
    return tracker


def _frame(call):
    with patch.object(stream_events.time, "perf_counter", lambda: CLOCK):
        return call()


def _invoke(name: str):
    """Return (frame, tracker, accounting) for one golden case."""
    tracker = None
    accounting = None
    if name == "completion_delta_plain":
        frame = stream_events.completion_delta(
            "cmpl-fixed", 1700000000, "fixture-model", "A"
        )
    elif name == "completion_delta_metadata":
        tracker = _tracker()
        frame = _frame(
            lambda: stream_events.completion_delta(
                "cmpl-fixed",
                1700000000,
                "fixture-model",
                "A",
                logprobs=COMPLETION_LOGPROBS,
                tokens={
                    "streamed_tokens": 3,
                    "delta_tokens": 1,
                    "answer_tokens": 3,
                },
                stream_chunk=_chunk(),
                include_hipengine=True,
                stream_started_at=tracker,
                routing=ROUTING,
                phase="answer",
            )
        )
    elif name == "completion_delta_buffered_source":
        tracker = _tracker(streaming=False)
        frame = _frame(
            lambda: stream_events.completion_delta(
                "cmpl-fixed",
                1700000000,
                "fixture-model",
                "A",
                include_hipengine=True,
                stream_started_at=tracker,
                stream_chunk=_buffered_source_chunk(),
            )
        )
    elif name == "completion_delta_accounting":
        accounting = stream_events.StreamTokenAccounting(lambda text: len(text))
        tokens = accounting.observe("answer", "A", cumulative_tokens=3)
        tracker = _tracker()
        frame = _frame(
            lambda: stream_events.completion_delta(
                "cmpl-fixed",
                1700000000,
                "fixture-model",
                "A",
                tokens=tokens,
                include_hipengine=True,
                stream_started_at=tracker,
            )
        )
    elif name == "completion_done_plain":
        frame = stream_events.completion_done(
            "cmpl-fixed",
            1700000000,
            "fixture-model",
            "length",
            finish_details=FINISH_DETAILS,
            stream_chunk=_finish_chunk(),
        )
    elif name == "completion_done_metadata":
        tracker = _tracker()
        frame = _frame(
            lambda: stream_events.completion_done(
                "cmpl-fixed",
                1700000000,
                "fixture-model",
                "length",
                finish_details=FINISH_DETAILS,
                tokens={"streamed_tokens": 10, "delta_tokens": 0},
                stream_chunk=_finish_chunk(),
                include_hipengine=True,
                extra_hipengine=EXTRA_HIPENGINE,
                stream_started_at=tracker,
                routing=ROUTING,
                kv_pool=KV_POOL,
                phase="structured",
            )
        )
    elif name == "completion_done_default":
        frame = api._completion_stream_done(
            "cmpl-fixed", 1700000000, "fixture-model", "stop"
        )
    elif name == "completion_usage_metadata":
        tracker = _tracker()
        frame = _frame(
            lambda: stream_events.completion_usage(
                "cmpl-fixed",
                1700000000,
                "fixture-model",
                {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15},
                include_hipengine=True,
                stream_started_at=tracker,
                routing=ROUTING,
                kv_pool=KV_POOL,
            )
        )
    elif name == "completion_error_plain":
        frame = api._completion_stream_error(
            "cmpl-fixed",
            1700000000,
            "fixture-model",
            "generated tool call failed validation",
            status_code=400,
            code="invalid_tool_call",
            param="tool_calls",
            error_type="invalid_request_error",
            finish_details=FINISH_DETAILS,
            extra={"request_id": "req-fixed"},
        )
    elif name == "completion_error_metadata":
        tracker = _tracker()
        frame = _frame(
            lambda: api._completion_stream_error(
                "cmpl-fixed",
                1700000000,
                "fixture-model",
                "generated tool call failed validation",
                status_code=400,
                code="invalid_tool_call",
                param="tool_calls",
                error_type="invalid_request_error",
                finish_details=FINISH_DETAILS,
                extra={"request_id": "req-fixed"},
                include_hipengine=True,
                stream_started_at=tracker,
                routing=ROUTING,
            )
        )
    elif name == "completion_error_default":
        frame = api._completion_stream_error(
            "cmpl-fixed", 1700000000, "fixture-model", "generation failed"
        )
    elif name == "chat_role_plain":
        frame = stream_events.chat_role(
            "chatcmpl-fixed", 1700000000, "fixture-model"
        )
    elif name == "chat_role_metadata":
        tracker = _tracker(streaming=False)
        frame = _frame(
            lambda: stream_events.chat_role(
                "chatcmpl-fixed",
                1700000000,
                "fixture-model",
                index=1,
                include_hipengine=True,
                stream_started_at=tracker,
                routing=ROUTING,
            )
        )
    elif name == "chat_delta_content_metadata":
        tracker = _tracker()
        frame = _frame(
            lambda: stream_events.chat_delta(
                "chatcmpl-fixed",
                1700000000,
                "fixture-model",
                "content",
                "A",
                index=1,
                logprobs=CHAT_LOGPROBS,
                tokens={"streamed_tokens": 3, "delta_tokens": 1},
                stream_chunk=_chunk(),
                include_hipengine=True,
                stream_started_at=tracker,
                routing=ROUTING,
                phase="answer",
            )
        )
    elif name == "chat_delta_reasoning_metadata":
        tracker = _tracker()
        frame = _frame(
            lambda: stream_events.chat_delta(
                "chatcmpl-fixed",
                1700000000,
                "fixture-model",
                "reasoning_content",
                "why",
                logprobs=None,
                reasoning_logprobs=REASONING_LOGPROBS,
                tokens={
                    "streamed_tokens": 3,
                    "delta_tokens": 1,
                    "reasoning_tokens": 1,
                },
                stream_chunk=_chunk(),
                include_hipengine=True,
                stream_started_at=tracker,
                routing=ROUTING,
                phase="think",
            )
        )
    elif name == "chat_tool_call_plain":
        frame = stream_events.chat_tool_call(
            "chatcmpl-fixed",
            1700000000,
            "fixture-model",
            TOOL_CALL,
            argument_chunk='{"command":',
            include_name=True,
        )
    elif name == "chat_tool_call_metadata":
        tracker = _tracker()
        frame = _frame(
            lambda: stream_events.chat_tool_call(
                "chatcmpl-fixed",
                1700000000,
                "fixture-model",
                TOOL_CALL,
                index=1,
                tool_index=2,
                argument_chunk='pwd"}',
                include_name=False,
                logprobs=CHAT_LOGPROBS,
                tokens={
                    "streamed_tokens": 3,
                    "delta_tokens": 1,
                    "tool_call_tokens": 1,
                },
                stream_chunk=_chunk(),
                include_hipengine=True,
                stream_started_at=tracker,
                routing=ROUTING,
            )
        )
    elif name == "chat_done_plain":
        frame = stream_events.chat_done(
            "chatcmpl-fixed",
            1700000000,
            "fixture-model",
            "length",
            finish_details=FINISH_DETAILS,
            stream_chunk=_finish_chunk(),
        )
    elif name == "chat_done_metadata":
        tracker = _tracker()
        frame = _frame(
            lambda: stream_events.chat_done(
                "chatcmpl-fixed",
                1700000000,
                "fixture-model",
                "tool_calls",
                index=1,
                finish_details=FINISH_DETAILS,
                tokens={"streamed_tokens": 10},
                stream_chunk=_finish_chunk(),
                include_hipengine=True,
                extra_hipengine=EXTRA_HIPENGINE,
                stream_started_at=tracker,
                routing=ROUTING,
                kv_pool=KV_POOL,
                phase="tool_call",
            )
        )
    elif name == "chat_done_default":
        frame = api._chat_stream_done(
            "chatcmpl-fixed", 1700000000, "fixture-model", "stop"
        )
    elif name == "chat_usage_metadata":
        tracker = _tracker()
        frame = _frame(
            lambda: stream_events.chat_usage(
                "chatcmpl-fixed",
                1700000000,
                "fixture-model",
                {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15},
                include_hipengine=True,
                stream_started_at=tracker,
                routing=ROUTING,
                kv_pool=KV_POOL,
            )
        )
    elif name == "chat_error_plain":
        frame = api._chat_stream_error(
            "chatcmpl-fixed",
            1700000000,
            "fixture-model",
            "generation failed",
            status_code=500,
            code="generation_failed",
            finish_details=FINISH_DETAILS,
            extra={"request_id": "req-fixed"},
        )
    elif name == "chat_error_metadata":
        tracker = _tracker()
        frame = _frame(
            lambda: api._chat_stream_error(
                "chatcmpl-fixed",
                1700000000,
                "fixture-model",
                "generation failed",
                status_code=500,
                code="generation_failed",
                finish_details=FINISH_DETAILS,
                extra={"request_id": "req-fixed"},
                include_hipengine=True,
                stream_started_at=tracker,
                routing=ROUTING,
            )
        )
    elif name == "chat_error_default":
        frame = api._chat_stream_error(
            "chatcmpl-fixed", 1700000000, "fixture-model", "generation failed"
        )
    else:  # pragma: no cover - keeps the fixture and builder in lockstep
        raise AssertionError(f"unhandled golden case: {name}")
    return frame, tracker, accounting


@pytest.mark.parametrize("name", sorted(FIXTURE["cases"]))
def test_stream_projection_matches_old_head_golden(name: str) -> None:
    expected = FIXTURE["cases"][name]
    frame, tracker, accounting = _invoke(name)
    assert frame == expected["frame"]
    state = expected.get("state", {})
    if "tracker" in state:
        assert tracker is not None
        assert asdict(tracker) == state["tracker"]
    if "accounting" in state:
        assert accounting is not None
        assert accounting.snapshot() == state["accounting"]


def test_error_serializer_frames_a_prebuilt_error_payload() -> None:
    frame = stream_events.chat_error(
        "chatcmpl-fixed",
        1700000000,
        "fixture-model",
        error={"message": "prebuilt", "type": "server_error", "code": "sentinel"},
        finish_details=None,
        include_hipengine=False,
    )
    payload = json.loads(frame.removeprefix("data: ").strip())
    assert payload["error"] == {
        "message": "prebuilt",
        "type": "server_error",
        "code": "sentinel",
    }
    assert "finish_details" not in payload["choices"][0]


def test_stream_usage_token_payload_merges_accounting_snapshot() -> None:
    accounting = stream_events.StreamTokenAccounting(lambda text: len(text))
    accounting.observe("think", "ab", cumulative_tokens=2)
    accounting.observe("answer", "cde")
    assert stream_events.stream_usage_token_payload(
        {"prompt_tokens": 1, "completion_tokens": 5, "total_tokens": 6},
        accounting,
    ) == {
        "prompt_tokens": 1,
        "completion_tokens": 5,
        "total_tokens": 6,
        "streamed_tokens": 5,
        "reasoning_tokens": 2,
        "answer_tokens": 3,
    }


def test_api_token_accounting_wraps_engine_counter_with_safe_count() -> None:
    class Engine:
        @staticmethod
        def count_tokens(_text: str) -> int:
            raise RuntimeError("counter failed")

    accounting = api._stream_token_accounting(Engine())
    assert accounting is not None
    assert accounting.observe("answer", "A") == {"streamed_tokens": 0, "delta_tokens": 0}


def test_tool_call_argument_chunks_split_at_owner_boundary() -> None:
    text = "x" * 129
    chunks = stream_events.tool_call_argument_chunks(text)
    assert [len(chunk) for chunk in chunks] == [128, 1]
    assert "".join(chunks) == text
    assert stream_events.tool_call_argument_chunks("short") == ("short",)


def test_stream_events_owner_has_no_application_or_live_stream_dependencies() -> None:
    tree = ast.parse(Path(stream_events.__file__).read_text(encoding="utf-8"))
    forbidden_imports = {
        "hipengine.server.api",
        "fastapi",
        "starlette",
        "hipengine.server.tool_stream",
    }
    forbidden_names = {
        "ServerConfig",
        "Request",
        "StreamingResponse",
        "XMLToolStream",
        "_parse_chat_tool_calls",
        "_ReasoningSplitter",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert node.module not in forbidden_imports, node.module
            for alias in node.names:
                assert alias.name not in forbidden_names, alias.name
        elif isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name not in forbidden_imports, alias.name
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            assert not isinstance(node, ast.AsyncFunctionDef), node.name
            arguments = [
                *node.args.posonlyargs,
                *node.args.args,
                *node.args.kwonlyargs,
            ]
            assert not ({argument.arg for argument in arguments} & {
                "app",
                "config",
                "request",
                "engine",
            }), node.name
