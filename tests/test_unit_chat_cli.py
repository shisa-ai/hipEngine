from __future__ import annotations

import io
import json
import os
from urllib.error import HTTPError

import pytest

from hipengine.chat_cli import build_parser, run


def test_chat_parser_defaults_to_local_server() -> None:
    args = build_parser().parse_args([])
    assert args.server == "http://127.0.0.1:8000"
    assert args.model is None
    assert args.max_tokens is None
    assert args.think == "default"


def test_chat_fails_clearly_without_a_running_server(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args, **kwargs):
        raise OSError("connection refused")

    monkeypatch.setattr("hipengine.chat_cli.urlopen", fail)
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
    assert "assistant> " in output.getvalue()
    assert "hello world" in output.getvalue()


def test_rich_chat_renders_status_reasoning_markdown_and_stats(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("rich")
    from rich.console import Console
    from rich.theme import Theme

    from hipengine.chat_cli import _THEME, _RichChat, _Settings

    ready = {
        "context": {"effective_max_context_tokens": 176128},
        "queue": {"max_active_requests": None},
        "kv_capacity": {
            "storage": "int8_per_token_head",
            "scale_dtype": "fp32",
            "pool": {"current_bytes": 6 * 1024**3, "budget_bytes": 21 * 1024**3},
        },
    }

    class JsonResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(ready).encode()

    class StreamResponse(JsonResponse):
        def __iter__(self):
            yield b'data: {"choices":[{"delta":{"reasoning_content":"hmm"}}]}\n'
            yield b'data: {"choices":[{"delta":{"content":"**hello**"}}]}\n'
            yield b'data: {"choices":[],"usage":{"completion_tokens":7}}\n'
            yield b"data: [DONE]\n"

    def fake_urlopen(request, timeout=None):
        return StreamResponse() if request.full_url.endswith("/chat/completions") else JsonResponse()

    monkeypatch.setattr("hipengine.chat_cli.urlopen", fake_urlopen)
    lines = iter(["hi", "/status", "/retry", "/think high", "/temp abc", "/bogus", "/quit"])
    console = Console(file=io.StringIO(), theme=Theme(_THEME), width=100, record=True)
    settings = _Settings(build_parser().parse_args([]))
    chat = _RichChat(console, "http://x", "local-model", settings, read_line=lambda: next(lines))
    assert chat.loop() == 0
    text = console.export_text()
    # the opening card and the /status redraw
    assert text.count("176,128 tokens") == 2
    assert "176,128 tokens" in text and "concurrency auto" in text
    assert "thought for" in text
    assert "hello" in text and "**" not in text
    assert "7 tokens" in text
    assert "unknown command /bogus" in text
    assert "reasoning: high" in text and "expects a float" in text
    assert [turn["role"] for turn in chat.convo.turns] == ["user", "assistant"]


@pytest.mark.parametrize("interrupt", [KeyboardInterrupt, EOFError])
def test_rich_chat_loop_exits_on_ctrl_c_or_ctrl_d(
    monkeypatch: pytest.MonkeyPatch, interrupt: type[BaseException]
) -> None:
    pytest.importorskip("rich")
    from rich.console import Console
    from rich.theme import Theme

    from hipengine.chat_cli import _THEME, _RichChat, _Settings

    class JsonResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({"context": {}, "kv_capacity": {}}).encode()

    monkeypatch.setattr("hipengine.chat_cli.urlopen", lambda request, timeout=None: JsonResponse())
    console = Console(file=io.StringIO(), theme=Theme(_THEME), width=100, record=True)

    def reader() -> str:
        raise interrupt

    chat = _RichChat(console, "http://x", "m", _Settings(build_parser().parse_args([])), read_line=reader)
    assert chat.loop() == 0
    assert "bye" in console.export_text()


def test_prompt_toolkit_ctrl_c_clears_typed_input_before_quitting() -> None:
    pytest.importorskip("prompt_toolkit")
    from hipengine.chat_cli import _prompt_bindings

    class Buffer:
        def __init__(self, text: str) -> None:
            self.text = text
            self.resets = 0

        def reset(self) -> None:
            self.resets += 1
            self.text = ""

    class App:
        def __init__(self) -> None:
            self.exited = None
            self.invalidated = 0

        def exit(self, exception=None, style=None) -> None:
            self.exited = exception

        def invalidate(self) -> None:
            self.invalidated += 1

    class Event:
        def __init__(self, text: str) -> None:
            self.current_buffer = Buffer(text)
            self.app = App()

    handler = next(binding.handler for binding in _prompt_bindings().bindings if "c-c" in binding.keys)

    typed = Event("a half-typed question")
    handler(typed)
    assert typed.app.exited is None
    assert typed.current_buffer.text == ""
    assert typed.current_buffer.resets == 1

    empty = Event("")
    handler(empty)
    assert empty.app.exited is EOFError


def test_prompt_toolkit_session_quits_on_ctrl_c() -> None:
    pytest.importorskip("prompt_toolkit")
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput
    from prompt_toolkit.shortcuts import PromptSession

    from hipengine.chat_cli import _prompt_bindings

    with create_pipe_input() as pipe:
        pipe.send_text("a half-typed question\x03")  # first Ctrl-C clears the line
        pipe.send_text("\x03")  # Ctrl-C at the empty prompt quits
        session = PromptSession(input=pipe, output=DummyOutput(), key_bindings=_prompt_bindings())
        with pytest.raises(EOFError):
            session.prompt()


def test_usage_rows_sum_server_reports_and_reset_with_the_conversation() -> None:
    from hipengine.chat_cli import _Conversation, _Settings, _usage_note, _usage_rows

    convo = _Conversation(_Settings(build_parser().parse_args([])))
    convo.turns = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "more"},
    ]
    convo.usage.add({"prompt_tokens": 100, "completion_tokens": 20, "reasoning_tokens": 5})
    convo.usage.add(
        {
            "prompt_tokens": 140,
            "completion_tokens": 30,
            "completion_tokens_details": {"reasoning_tokens": 7},
        }
    )
    convo.usage.add(None)  # stopped before the server reported usage

    rows = dict(_usage_rows(convo, 1000))
    assert rows["messages"] == "3 messages  ·  1 turn"
    assert rows["in"] == "240 tokens"
    assert rows["out"] == "50 tokens  ·  reasoning 12"
    assert rows["total"] == "290 tokens"
    assert rows["context"] == "140 / 1,000 tokens  ·  14.0% used"
    assert convo.usage.unreported == 1
    assert _usage_note(convo.usage) == "1 request reported no token usage (stopped or failed)"

    convo.clear()
    cleared = dict(_usage_rows(convo, 1000))
    assert cleared["messages"] == "0 messages  ·  0 turns"
    assert cleared["in"] == "0 tokens" and cleared["total"] == "0 tokens"
    assert "context" not in cleared
    assert _usage_note(convo.usage) is None


def test_rich_chat_usage_command_reports_tokens_and_context_share(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("rich")
    from rich.console import Console
    from rich.theme import Theme

    from hipengine.chat_cli import _THEME, _RichChat, _Settings

    ready = {"context": {"effective_max_context_tokens": 1000}, "kv_capacity": {}}

    class JsonResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(ready).encode()

    class StreamResponse(JsonResponse):
        def __iter__(self):
            yield b'data: {"choices":[{"delta":{"content":"ok"}}]}\n'
            yield b'data: {"choices":[],"usage":{"prompt_tokens":300,"completion_tokens":12}}\n'
            yield b"data: [DONE]\n"

    def fake_urlopen(request, timeout=None):
        return StreamResponse() if request.full_url.endswith("/chat/completions") else JsonResponse()

    monkeypatch.setattr("hipengine.chat_cli.urlopen", fake_urlopen)
    lines = iter(["hello", "/usage", "/quit"])
    console = Console(file=io.StringIO(), theme=Theme(_THEME), width=100, record=True)
    chat = _RichChat(
        console, "http://x", "m", _Settings(build_parser().parse_args([])), read_line=lambda: next(lines)
    )
    assert chat.loop() == 0
    text = console.export_text()
    assert "2 messages  ·  1 turn" in text
    assert "300 tokens" in text
    assert "312 tokens" in text
    assert "300 / 1,000 tokens  ·  30.0% used" in text


def test_plain_chat_usage_command_reports_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    ready = {"context": {"effective_max_context_tokens": 1000}, "kv_capacity": {}}

    class JsonResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(ready).encode()

    class StreamResponse(JsonResponse):
        def __iter__(self):
            yield b'data: {"choices":[{"delta":{"content":"ok"}}]}\n'
            yield b'data: {"choices":[],"usage":{"prompt_tokens":300,"completion_tokens":12}}\n'
            yield b"data: [DONE]\n"

    def fake_urlopen(request, timeout=None):
        return StreamResponse() if request.full_url.endswith("/chat/completions") else JsonResponse()

    monkeypatch.setattr("hipengine.chat_cli.urlopen", fake_urlopen)
    output = io.StringIO()
    result = run(
        build_parser().parse_args(["--model", "m"]),
        input_stream=io.StringIO("hello\n/usage\n/quit\n"),
        output_stream=output,
    )
    assert result == 0
    text = output.getvalue()
    assert "300 tokens" in text and "312 tokens" in text
    assert "300 / 1,000 tokens  ·  30.0% used" in text


def test_quiet_tty_degrades_when_the_stream_cannot_be_reconfigured() -> None:
    from hipengine.chat_cli import _quiet_tty

    with open(os.devnull, "rb") as stream:
        with _quiet_tty(stream):
            pass  # a non-tty fd (termios.error) must not escape

    class NoFileno:
        pass

    with _quiet_tty(NoFileno()):
        pass  # nor a stream without a file descriptor


def test_quiet_tty_disables_echo_on_a_tty_and_restores_it() -> None:
    termios = pytest.importorskip("termios")
    if not hasattr(os, "openpty"):
        pytest.skip("no pty support on this platform")

    from hipengine.chat_cli import _quiet_tty

    try:
        master, slave = os.openpty()
    except OSError:  # pragma: no cover - depends on the sandbox
        pytest.skip("no pty available")
    try:
        with os.fdopen(slave, "rb", buffering=0) as stream:
            initial = list(termios.tcgetattr(stream.fileno()))
            initial[3] |= termios.ECHO  # do not depend on the platform's pty defaults
            termios.tcsetattr(stream.fileno(), termios.TCSANOW, initial)
            before = termios.tcgetattr(stream.fileno())
            with _quiet_tty(stream):
                inside = termios.tcgetattr(stream.fileno())
            after = termios.tcgetattr(stream.fileno())
    finally:
        os.close(master)

    assert before[3] & termios.ECHO
    assert not inside[3] & termios.ECHO
    assert inside[3] & termios.NOFLSH
    assert after[3] == before[3]


def test_quiet_tty_reports_a_failed_restore_without_raising(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    termios = pytest.importorskip("termios")
    if not hasattr(os, "openpty"):
        pytest.skip("no pty support on this platform")

    from hipengine.chat_cli import _quiet_tty

    try:
        master, slave = os.openpty()
    except OSError:  # pragma: no cover - depends on the sandbox
        pytest.skip("no pty available")

    real_tcsetattr = termios.tcsetattr
    calls = {"count": 0}

    def flaky_tcsetattr(fd, when, attributes):
        calls["count"] += 1
        if calls["count"] > 1:
            raise termios.error(5, "Input/output error")
        return real_tcsetattr(fd, when, attributes)

    monkeypatch.setattr(termios, "tcsetattr", flaky_tcsetattr)
    try:
        with os.fdopen(slave, "rb", buffering=0) as stream:
            with _quiet_tty(stream):
                pass  # a failed restore must warn, not abort the chat
    finally:
        os.close(master)

    assert calls["count"] == 2
    assert "could not restore terminal settings" in capsys.readouterr().err


def test_chat_settings_map_reasoning_and_sampling_to_request_fields() -> None:
    from hipengine.chat_cli import _Settings, _think_fields

    settings = _Settings(build_parser().parse_args(["--think", "off", "--top-p", "0.9"]))
    assert settings.request_fields() == {"temperature": 0.0, "top_p": 0.9, "enable_thinking": False}
    assert _think_fields("medium") == {"enable_thinking": True, "reasoning_effort": "medium"}
    assert _think_fields("2048") == {"enable_thinking": True, "thinking_token_budget": 2048}
    assert _think_fields("default") == {}
    with pytest.raises(ValueError):
        _think_fields("sometimes")
    settings.command("/max", "1024")
    settings.command("/temp", "default")
    settings.command("/think", "low")
    assert settings.request_fields() == {
        "top_p": 0.9,
        "max_tokens": 1024,
        "enable_thinking": True,
        "reasoning_effort": "low",
    }
    assert settings.command("/nope", "") is None


def test_plain_chat_sends_settings_and_stashes_failed_prompt_for_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    bodies = []

    class JsonResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({"data": [{"id": "m"}]}).encode()

    class StreamResponse(JsonResponse):
        def __iter__(self):
            yield b'data: {"choices":[{"delta":{"content":"ok"}}]}\n'
            yield b"data: [DONE]\n"

    def fake_urlopen(request, timeout=None):
        if request.full_url.endswith("/chat/completions"):
            bodies.append(json.loads(request.data))
            if len(bodies) == 1:
                raise OSError("boom")
            return StreamResponse()
        return JsonResponse()

    monkeypatch.setattr("hipengine.chat_cli.urlopen", fake_urlopen)
    output = io.StringIO()
    result = run(
        build_parser().parse_args(["--model", "m"]),
        input_stream=io.StringIO("/think 512\nhello\n/retry\n/quit\n"),
        output_stream=output,
    )
    assert result == 0
    assert len(bodies) == 2
    assert bodies[1]["thinking_token_budget"] == 512
    assert bodies[1]["messages"] == [{"role": "user", "content": "hello"}]
    assert "ok" in output.getvalue()


def test_turn_stats_report_prefill_decode_and_cached_tokens() -> None:
    from hipengine.chat_cli import _RichChat, _Settings, _Turn

    turn = _Turn(_Settings(build_parser().parse_args([])))
    turn.add("meta", {"timing": {"backend_prefill_ms": 1626.638, "decode_tokens_per_second": 32.66}})
    turn.add(
        "meta",
        {
            "diagnostics": {
                "prefix_cache": {"hit": False, "reused_tokens": 0, "executed_prefill_tokens": 525}
            }
        },
    )
    turn.add("content", "hello")
    turn.add("usage", {"prompt_tokens": 525, "completion_tokens": 24})
    turn.first_token = turn.started + 1.98
    turn.finished = turn.started + 2.90

    line, rate = _RichChat._stats(turn)

    assert line == "24 tokens  ·  prefill 323 tok/s  ·  decode 32.7 tok/s  ·  ttft 1.98s  ·  2.90s"
    assert rate == "32.7 tok/s"


def test_turn_stats_report_a_cache_hit_as_reused_tokens_and_smaller_prefill() -> None:
    from hipengine.chat_cli import _RichChat, _Settings, _Turn

    turn = _Turn(_Settings(build_parser().parse_args([])))
    turn.add("meta", {"timing": {"backend_prefill_ms": 250.0, "decode_tokens_per_second": 32.5}})
    turn.add(
        "meta",
        {
            "diagnostics": {
                "prefix_cache": {
                    "hit": True,
                    "reused_tokens": 512,
                    "executed_prefill_tokens": 13,
                    "fallback_reason": None,
                }
            }
        },
    )
    turn.add("content", "hello")
    turn.add("usage", {"prompt_tokens": 525, "completion_tokens": 24})
    turn.first_token = turn.started + 0.54
    turn.finished = turn.started + 1.40

    line, _ = _RichChat._stats(turn)

    assert "prefill 52 tok/s" in line
    assert "cached 512" in line
    assert "decode 32.5 tok/s" in line


def test_turn_stats_prefer_the_usage_cache_field_over_diagnostics() -> None:
    """The vLLM-style usage field is authoritative when the server sends it."""

    from hipengine.chat_cli import _RichChat, _Settings, _Turn

    turn = _Turn(_Settings(build_parser().parse_args([])))
    turn.add("meta", {"diagnostics": {"prefix_cache": {"reused_tokens": 512}}})
    turn.add(
        "usage",
        {
            "prompt_tokens": 525,
            "completion_tokens": 24,
            "prompt_tokens_details": {"cached_tokens": 256},
        },
    )
    turn.add("content", "hello")
    turn.first_token = turn.started + 0.5
    turn.finished = turn.started + 1.0

    line, _ = _RichChat._stats(turn)

    assert "cached 256" in line
    assert "cached 512" not in line


def test_turn_stats_degrade_to_client_timings_without_server_metadata() -> None:
    """A server that reports no timing still yields a decode rate and prefill."""

    from hipengine.chat_cli import _RichChat, _Settings, _Turn

    turn = _Turn(_Settings(build_parser().parse_args([])))
    turn.add("content", "hello")
    turn.add("usage", {"prompt_tokens": 500, "completion_tokens": 24})
    turn.first_token = turn.started + 1.0
    turn.finished = turn.started + 2.0

    line, rate = _RichChat._stats(turn)

    assert line == "24 tokens  ·  prefill 500 tok/s  ·  decode 23.0 tok/s  ·  ttft 1.00s  ·  2.00s"
    assert rate == "23.0 tok/s"


def test_usage_rows_count_cached_tokens_from_usage_and_diagnostics() -> None:
    from hipengine.chat_cli import _Conversation, _Settings, _usage_rows

    convo = _Conversation(_Settings(build_parser().parse_args([])))
    convo.usage.add({"prompt_tokens": 1000, "completion_tokens": 20})
    # A server that reports reuse only in backend diagnostics.
    convo.usage.add({"prompt_tokens": 525, "completion_tokens": 24}, cached_tokens=512)
    # A server that reports it in usage wins, and must not be counted twice.
    convo.usage.add(
        {
            "prompt_tokens": 525,
            "completion_tokens": 24,
            "prompt_tokens_details": {"cached_tokens": 256},
        },
        cached_tokens=999,
    )

    rows = dict(_usage_rows(convo, 4096))

    assert rows["in"] == "2,050 tokens"
    assert rows["cached"] == "768 tokens  ·  37.5% of in"
    assert convo.usage.cached_tokens == 768

    convo.clear()

    assert "cached" not in dict(_usage_rows(convo, 4096))


def _sse(payload: dict) -> bytes:
    return b"data: " + json.dumps(payload).encode() + b"\n"


class _BenchStreamServer:
    """Fake server answering /bench with realistic streamed metadata.

    The first request for a prompt is cold; a repeat hits the 512-token boundary
    the fake pretends to have cached. MTP routes decode faster, as the engine does.
    """

    def __init__(
        self,
        *,
        usage_reports_cached: bool = False,
        refuse_mtp: bool = False,
        mtp_cycles: int = 7,
    ) -> None:
        self.bodies: list[dict] = []
        self.seen: dict[str, int] = {}
        self.usage_reports_cached = usage_reports_cached
        self.refuse_mtp = refuse_mtp
        self.mtp_cycles = mtp_cycles

    def __call__(self, request, timeout=None):
        if not request.full_url.endswith("/chat/completions"):
            return _ReadyResponse()
        body = json.loads(request.data)
        self.bodies.append(body)
        if body.get("speculative_mtp") and self.refuse_mtp:
            detail = json.dumps({"error": {"message": "speculative_mtp is not available"}}).encode()
            raise HTTPError(request.full_url, 400, "Bad Request", None, io.BytesIO(detail))
        return _BenchStreamResponse(self, body)

    def events(self, body: dict):
        speculative = bool(body.get("speculative_mtp"))
        prompt = body["messages"][0]["content"]
        seen = self.seen.get(prompt, 0) + 1
        self.seen[prompt] = seen
        prompt_tokens = len(prompt.split()) + 6
        cached = 512 if seen > 1 else 0
        prefill_ms = 250.0 if cached else 1600.0
        timing = {
            "elapsed_ms": prefill_ms + 700.0,
            "ttft_ms": prefill_ms + 40.0,
            "backend_prefill_ms": prefill_ms,
            "decode_tokens_per_second": 32.7 if speculative else 11.8,
        }
        diagnostics = {
            "prefix_cache": {
                "mode": "radix",
                "block_size_tokens": 256,
                "eligible": True,
                "lookup": True,
                "hit": bool(cached),
                "reused_tokens": cached,
                "executed_prefill_tokens": max(0, prompt_tokens - cached),
                "fallback_reason": None if cached else "miss",
            }
        }
        if speculative:
            diagnostics["specdec2_mtp2"] = {
                "requested_budget": 3,
                "cycles": self.mtp_cycles,
                "accepted_draft_tokens": 12,
                "provider_decline_reason": None if self.mtp_cycles else "provider_not_ready",
            }
        yield _sse(
            {
                "choices": [{"index": 0, "delta": {"content": "ok"}, "finish_reason": None}],
                "hipengine": {"event": "delta", "timing": timing},
            }
        )
        yield _sse(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "length",
                        "hipengine": {"phase": "done", "diagnostics": diagnostics},
                    }
                ],
                "hipengine": {"event": "done", "timing": timing},
            }
        )
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": 24,
            "total_tokens": prompt_tokens + 24,
        }
        if self.usage_reports_cached:
            usage["prompt_tokens_details"] = {"cached_tokens": cached}
        yield _sse({"choices": [], "usage": usage, "hipengine": {"event": "usage", "timing": timing}})
        yield b"data: [DONE]\n"


class _ReadyResponse:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps({"context": {"effective_max_context_tokens": 1000}, "kv_capacity": {}}).encode()


class _BenchStreamResponse:
    def __init__(self, server: _BenchStreamServer, body: dict) -> None:
        self.server = server
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def __iter__(self):
        yield from self.server.events(self.body)



def _rich_bench_text(server: _BenchStreamServer, monkeypatch: pytest.MonkeyPatch, command: str = "/bench") -> str:
    """Drive one /bench through the rich client and return the rendered text."""

    pytest.importorskip("rich")
    from rich.console import Console
    from rich.theme import Theme

    from hipengine.chat_cli import _THEME, _RichChat, _Settings

    monkeypatch.setattr("hipengine.chat_cli.urlopen", server)
    lines = iter([command, "/quit"])
    console = Console(file=io.StringIO(), theme=Theme(_THEME), width=160, record=True)
    chat = _RichChat(
        console, "http://x", "m", _Settings(build_parser().parse_args([])), read_line=lambda: next(lines)
    )

    assert chat.loop() == 0
    return console.export_text()


def test_bench_measures_cold_and_cached_runs_for_both_routes(monkeypatch: pytest.MonkeyPatch) -> None:
    server = _BenchStreamServer()

    text = _rich_bench_text(server, monkeypatch)

    assert len(server.bodies) == 4
    assert [body["speculative_mtp"] for body in server.bodies] == [True, True, False, False]
    assert server.bodies[0]["max_tokens"] == 32
    # Both requests of a route share one prompt; the routes differ so that the
    # second route's cold run cannot hit the first route's cache entry.
    assert server.bodies[0]["messages"] == server.bodies[1]["messages"]
    assert server.bodies[2]["messages"] != server.bodies[0]["messages"]
    assert text.count("hit 512") == 2
    assert text.count("miss") == 2
    assert "32.7 tok/s" in text and "11.8 tok/s" in text
    assert "cold then cached per route" in text


def test_bench_reads_cached_tokens_from_usage_when_the_server_reports_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = _BenchStreamServer(usage_reports_cached=True)

    text = _rich_bench_text(server, monkeypatch)

    assert text.count("hit 512") == 2


def test_bench_reports_a_route_the_server_refuses(monkeypatch: pytest.MonkeyPatch) -> None:
    server = _BenchStreamServer(refuse_mtp=True)

    text = _rich_bench_text(server, monkeypatch)

    # The refused request was sent, but that route never reached its cached repeat.
    assert [body["speculative_mtp"] for body in server.bodies] == [True, False, False]
    assert "mtp on: HTTP 400: speculative_mtp is not available" in text
    assert "mtp off" in text


def test_bench_notes_an_mtp_route_that_committed_no_speculative_cycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = _BenchStreamServer(mtp_cycles=0)

    text = _rich_bench_text(server, monkeypatch)

    assert "! mtp on: no speculative cycle ran (provider_decline_reason=provider_not_ready)" in text
    assert "mtp off" in text


def test_bench_rejects_a_non_numeric_token_count(monkeypatch: pytest.MonkeyPatch) -> None:
    server = _BenchStreamServer()

    text = _rich_bench_text(server, monkeypatch, command="/bench abc")

    assert "usage: /bench [prompt_tokens] [max_tokens]" in text
    assert server.bodies == []


def test_plain_chat_bench_reports_the_same_measurements(monkeypatch: pytest.MonkeyPatch) -> None:
    server = _BenchStreamServer()
    monkeypatch.setattr("hipengine.chat_cli.urlopen", server)
    output = io.StringIO()

    result = run(
        build_parser().parse_args(["--model", "m", "--plain"]),
        input_stream=io.StringIO("/bench\n/quit\n"),
        output_stream=output,
    )

    assert result == 0
    text = output.getvalue()
    assert "cold then cached per route" in text
    assert text.count("hit 512") == 2
    assert "mtp on" in text and "mtp off" in text
    assert len(server.bodies) == 4
    assert [body["speculative_mtp"] for body in server.bodies] == [True, True, False, False]


def test_plain_chat_bench_reports_a_route_the_server_refuses(monkeypatch: pytest.MonkeyPatch) -> None:
    server = _BenchStreamServer(refuse_mtp=True)
    monkeypatch.setattr("hipengine.chat_cli.urlopen", server)
    output = io.StringIO()

    result = run(
        build_parser().parse_args(["--model", "m", "--plain"]),
        input_stream=io.StringIO("/bench 256 8\n/quit\n"),
        output_stream=output,
    )

    assert result == 0
    text = output.getvalue()
    assert "! mtp on: HTTP 400: speculative_mtp is not available" in text
    assert "mtp off" in text
    assert [body["speculative_mtp"] for body in server.bodies] == [True, False, False]
    assert server.bodies[0]["max_tokens"] == 8
