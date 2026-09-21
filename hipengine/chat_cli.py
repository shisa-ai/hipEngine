"""Terminal client for a running hipEngine OpenAI-compatible server.

With ``rich`` installed and an interactive terminal, ``hipengine chat`` renders a
TUI (status card, live-streamed Markdown, collapsible reasoning, per-turn stats);
``prompt_toolkit`` additionally enables history, completion, and a status bar.
Otherwise it falls back to a dependency-free line-oriented client. Both modes
share the same slash commands for reasoning and sampling controls.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import re
import sys
import time
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

_REQUEST_ERRORS = (HTTPError, URLError, OSError, RuntimeError, ValueError)
_THINK_EFFORTS = ("minimal", "low", "medium", "high", "xhigh")

_COMMANDS = (
    ("/help", "show commands"),
    ("/status", "show model, context, and KV pool"),
    ("/usage", "show messages, turns, and token counts"),
    ("/bench [in] [out]", "measure prefill, decode, and cache reuse; default 512 in, 32 out"),
    ("/params", "show sampling and reasoning settings"),
    ("/think <mode>", "reasoning: default, off, on, " + ", ".join(_THINK_EFFORTS) + ", or a token budget"),
    ("/show", "toggle showing reasoning text"),
    ("/temp /top_p /top_k", "set sampling (no value shows it; 'default' resets)"),
    ("/min_p /rep /max /seed", "min-p, repetition penalty, max tokens, seed"),
    ("/system <text>", "set the system message (no text clears it)"),
    ("/retry", "regenerate the last reply"),
    ("/clear", "clear the conversation"),
    ("/quit", "exit (also Ctrl-C, Ctrl-D)"),
)

_THEME = {
    "hip.accent": "bold #d787ff",
    "hip.user": "bold #5fd7ff",
    "hip.label": "#8a8a8a",
    "hip.dim": "#6c6c6c",
    "hip.thinking": "italic #6c6c6c",
    "hip.ok": "#87d787",
    "hip.hit": "bold #7fe0b0",
    "hip.miss": "bold #ff8a75",
    "hip.warn": "#ffaf5f",
    "hip.err": "bold #ff5f5f",
    "hip.border": "#4e4e4e",
    # /bench metric columns: prompt-side, cache-served, output-side, latency.
    "hip.metric.prefill": "bold #5fd7ff",
    "hip.metric.prompt": "bold #7fe0b0",
    "hip.metric.decode": "bold #d787ff",
    "hip.metric.ttft": "bold #ffaf5f",
    "hip.metric.tpot": "bold #ffd7af",
}

# /usage row -> value style
_USAGE_STYLES = {
    "messages": "bold",
    "in": "hip.user",
    "cached": "hip.hit",
    "out": "hip.accent",
    "total": "bold",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hipengine chat",
        description="Chat with a running local hipEngine server.",
    )
    parser.add_argument(
        "--server",
        default="http://127.0.0.1:8000",
        help="Server URL (default: http://127.0.0.1:8000)",
    )
    parser.add_argument(
        "--model",
        help="Model id sent to the server; defaults to the first served model.",
    )
    parser.add_argument("--system", help="Optional system message.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--min-p", type=float)
    parser.add_argument("--repetition-penalty", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--max-tokens",
        type=int,
        help="Maximum generated tokens per reply (default: server's model-aware default).",
    )
    parser.add_argument(
        "--think",
        default="default",
        help="Reasoning mode: default, off, on, "
        + ", ".join(_THINK_EFFORTS)
        + ", or a thinking-token budget.",
    )
    parser.add_argument(
        "--show-thinking",
        action="store_true",
        help="Show model reasoning text instead of a one-line summary.",
    )
    parser.add_argument(
        "--plain",
        action="store_true",
        help="Use the plain line-oriented client even when rich is installed.",
    )
    return parser


# -- settings ------------------------------------------------------------------

# command -> (request field, parser)
_SAMPLING = {
    "/temp": ("temperature", float),
    "/top_p": ("top_p", float),
    "/top_k": ("top_k", int),
    "/min_p": ("min_p", float),
    "/rep": ("repetition_penalty", float),
    "/max": ("max_tokens", int),
    "/seed": ("seed", int),
}


def _think_fields(mode: str) -> dict:
    mode = mode.strip().lower()
    if mode in {"", "default", "auto"}:
        return {}
    if mode in {"off", "false", "no", "none"}:
        return {"enable_thinking": False}
    if mode in {"on", "true", "yes"}:
        return {"enable_thinking": True}
    if mode in _THINK_EFFORTS:
        return {"enable_thinking": True, "reasoning_effort": mode}
    if mode.isdigit():
        return {"enable_thinking": True, "thinking_token_budget": int(mode)}
    raise ValueError(
        f"unknown reasoning mode {mode!r}; use default, off, on, "
        + ", ".join(_THINK_EFFORTS)
        + ", or a token budget"
    )


class _Settings:
    def __init__(self, args: argparse.Namespace) -> None:
        self.sampling: dict[str, float | int | None] = {
            "temperature": args.temperature,
            "top_p": getattr(args, "top_p", None),
            "top_k": getattr(args, "top_k", None),
            "min_p": getattr(args, "min_p", None),
            "repetition_penalty": getattr(args, "repetition_penalty", None),
            "max_tokens": args.max_tokens,
            "seed": getattr(args, "seed", None),
        }
        self.think = "default"
        self.set_think(str(getattr(args, "think", "default") or "default"))
        self.show_thinking = bool(getattr(args, "show_thinking", False))
        self.system: str | None = args.system or None

    def set_think(self, mode: str) -> str:
        _think_fields(mode)  # validate
        self.think = mode.strip().lower() or "default"
        return f"reasoning: {self.think}"

    def request_fields(self) -> dict:
        fields = {key: value for key, value in self.sampling.items() if value is not None}
        fields.update(_think_fields(self.think))
        return fields

    def rows(self) -> list[tuple[str, str]]:
        rows = [(name, _fmt(self.sampling[field])) for name, (field, _) in _SAMPLING.items()]
        rows.append(("/think", self.think))
        rows.append(("/show", "on" if self.show_thinking else "off"))
        return rows

    def command(self, name: str, rest: str) -> str | None:
        """Apply a settings command; return a message, or None if not a settings command."""

        rest = rest.strip()
        if name in _SAMPLING:
            field, cast = _SAMPLING[name]
            if not rest:
                return f"{field} = {_fmt(self.sampling[field])}"
            if rest.lower() in {"default", "none", "reset"}:
                self.sampling[field] = None
            else:
                try:
                    self.sampling[field] = cast(rest)
                except ValueError as exc:
                    raise ValueError(f"{name} expects a {cast.__name__}, got {rest!r}") from exc
            return f"{field} = {_fmt(self.sampling[field])}"
        if name == "/think":
            return self.set_think(rest) if rest else f"reasoning: {self.think}"
        if name == "/show":
            self.show_thinking = not self.show_thinking
            return "reasoning text " + ("shown" if self.show_thinking else "hidden")
        if name == "/system":
            self.system = rest or None
            return "system message " + ("set" if self.system else "cleared")
        return None


def _fmt(value) -> str:
    return "default" if value is None else str(value)


# -- server I/O ----------------------------------------------------------------


def _get_json(url: str) -> dict:
    with urlopen(Request(url, headers={"Accept": "application/json"}), timeout=5) as response:
        payload = json.load(response)
    if not isinstance(payload, dict):
        raise ValueError(f"server returned non-object JSON from {url}")
    return payload


def _resolve_model(server: str, requested: str | None) -> str:
    if requested:
        return requested
    payload = _get_json(f"{server.rstrip('/')}/v1/models")
    models = payload.get("data")
    if not isinstance(models, list) or not models or not isinstance(models[0], dict):
        raise RuntimeError("server returned no served models; pass --model explicitly")
    model = models[0].get("id")
    if not isinstance(model, str) or not model:
        raise RuntimeError("server returned an invalid model id")
    return model


def _gib(value) -> str:
    try:
        return f"{float(value) / 1024**3:.2f} GiB"
    except (TypeError, ValueError):
        return "unknown"


def _status_fields(ready: dict) -> dict:
    context = ready.get("context", {})
    queue = ready.get("queue", {})
    kv = ready.get("kv_capacity", {})
    context = context if isinstance(context, dict) else {}
    queue = queue if isinstance(queue, dict) else {}
    kv = kv if isinstance(kv, dict) else {}
    pool = kv.get("pool") if isinstance(kv.get("pool"), dict) else {}
    concurrency = queue.get("max_active_requests")
    return {
        "context": context.get("effective_max_context_tokens", "auto"),
        "concurrency": "auto" if concurrency is None else concurrency,
        "kv_storage": kv.get("storage", "unknown"),
        "kv_scale": kv.get("scale_dtype", "unknown"),
        "pool_current": pool.get("current_bytes", 0),
        "pool_budget": pool.get("budget_bytes"),
    }


def _print_status(server: str, model: str, *, output_stream=None) -> None:
    output_stream = sys.stdout if output_stream is None else output_stream
    fields = _status_fields(_get_json(f"{server.rstrip('/')}/ready"))
    budget = fields["pool_budget"]
    print(
        f"[{model}] "
        f"context={fields['context']} "
        f"concurrency={fields['concurrency']} "
        f"kv={fields['kv_storage']}/{fields['kv_scale']} "
        f"pool={_gib(fields['pool_current'])}/"
        f"{_gib(budget) if budget is not None else 'automatic'}",
        file=output_stream,
    )


def _max_context(server: str) -> int | None:
    """The server's effective context limit in tokens, or None when unavailable."""

    try:
        context = _status_fields(_get_json(f"{server.rstrip('/')}/ready"))["context"]
    except _REQUEST_ERRORS:
        return None
    return context if isinstance(context, int) and not isinstance(context, bool) else None


def _stream_events(
    server: str,
    *,
    model: str,
    messages: list[dict[str, str]],
    fields: dict,
) -> Iterator[tuple[str, object]]:
    """Yield deltas, ``("usage", dict)``, and ``("meta", dict)`` server metadata.

    Metadata is requested with ``include_hipengine``, which is what carries the
    backend's own timing (prefill, decode, ttft) and per-request prefix-cache
    diagnostics. Servers predating that field ignore the option and simply send
    no metadata, so the caller degrades to client-measured numbers.
    """

    body = json.dumps(
        {
            "model": model,
            "messages": messages,
            **fields,
            "stream": True,
            "stream_options": {"include_usage": True, "include_hipengine": True},
        }
    ).encode()
    request = Request(
        f"{server.rstrip('/')}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    with urlopen(request, timeout=None) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                return
            payload = json.loads(data)
            usage = payload.get("usage")
            if isinstance(usage, dict):
                yield "usage", usage
            meta = payload.get("hipengine")
            if isinstance(meta, dict):
                yield "meta", meta
            choices = payload.get("choices", [])
            if not choices or not isinstance(choices[0], dict):
                continue
            choice_meta = choices[0].get("hipengine")
            if isinstance(choice_meta, dict):
                yield "meta", choice_meta
            delta = choices[0].get("delta", {})
            if not isinstance(delta, dict):
                continue
            reasoning = delta.get("reasoning_content")
            if isinstance(reasoning, str) and reasoning:
                yield "reasoning", reasoning
            text = delta.get("content")
            if isinstance(text, str) and text:
                yield "content", text


def _http_error_text(exc: Exception) -> str:
    if isinstance(exc, HTTPError):
        try:
            detail = json.loads(exc.read() or b"{}")
            message = detail.get("error", {}).get("message") or detail.get("detail")
            if message:
                return f"HTTP {exc.code}: {message}"
        except (ValueError, AttributeError, OSError):
            pass
    return str(exc)


def _meta_cached_tokens(meta: dict) -> int | None:
    """Prefix-cache reuse from one metadata event, when the backend reported it."""

    diagnostics = meta.get("diagnostics")
    block = diagnostics.get("prefix_cache") if isinstance(diagnostics, dict) else None
    return _as_int(block.get("reused_tokens")) if isinstance(block, dict) else None


def _as_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _as_float(value: object) -> float | None:
    """Return a finite float, so a missing or non-numeric metric stays absent."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


class _Usage:
    """Conversation token accounting, summed from server-reported usage payloads."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.requests = 0
        self.unreported = 0
        self.prompt_tokens = 0
        self.cached_tokens = 0
        self.completion_tokens = 0
        self.reasoning_tokens = 0
        self.last_prompt_tokens = 0

    def add(self, usage: object, *, cached_tokens: int | None = None) -> None:
        """Count one finished request; ``usage`` is the server payload when it arrived.

        ``cached_tokens`` is a fallback for servers that report prefix-cache reuse
        only in per-request backend diagnostics rather than in usage.
        """

        self.requests += 1
        if not isinstance(usage, dict):
            self.unreported += 1
            return
        prompt = _as_int(usage.get("prompt_tokens"))
        completion = _as_int(usage.get("completion_tokens"))
        if prompt is None and completion is None:
            self.unreported += 1
            return
        if prompt is not None:
            self.prompt_tokens += prompt
            self.last_prompt_tokens = prompt
        if completion is not None:
            self.completion_tokens += completion
        prompt_details = usage.get("prompt_tokens_details")
        cached = _as_int(prompt_details.get("cached_tokens")) if isinstance(prompt_details, dict) else None
        if cached is None:
            cached = cached_tokens
        if cached:
            self.cached_tokens += cached
        details = usage.get("completion_tokens_details")
        reasoning = _as_int(usage.get("reasoning_tokens"))
        if reasoning is None and isinstance(details, dict):
            reasoning = _as_int(details.get("reasoning_tokens"))
        if reasoning:
            self.reasoning_tokens += reasoning


class _Conversation:
    """Chat turns plus a stash of the last prompt whose reply was abandoned."""

    def __init__(self, settings: _Settings) -> None:
        self.settings = settings
        self.turns: list[dict[str, str]] = []
        self.pending: str | None = None
        self.usage = _Usage()

    def messages(self) -> list[dict[str, str]]:
        system = self.settings.system
        head = [{"role": "system", "content": system}] if system else []
        return head + self.turns

    def begin_retry(self) -> bool:
        if self.turns and self.turns[-1]["role"] == "assistant":
            self.turns.pop()
        elif self.pending is not None:
            self.turns.append({"role": "user", "content": self.pending})
        self.pending = None
        return bool(self.turns) and self.turns[-1]["role"] == "user"

    def abandon(self) -> None:
        """Drop an unanswered user turn, keeping it available for /retry."""

        if self.turns and self.turns[-1]["role"] == "user":
            self.pending = self.turns.pop()["content"]

    def clear(self) -> None:
        self.turns.clear()
        self.pending = None
        self.usage.reset()


def _plural(count: int, word: str) -> str:
    return f"{count:,} {word}" if count == 1 else f"{count:,} {word}s"


def _usage_rows(convo: _Conversation, max_context: int | None = None) -> list[tuple[str, str]]:
    """Return ``(label, value)`` rows for /usage; shared by the rich and plain clients."""

    usage = convo.usage
    turns = sum(1 for message in convo.turns if message["role"] == "assistant")
    rows = [
        ("messages", f"{_plural(len(convo.messages()), 'message')}  ·  {_plural(turns, 'turn')}"),
        ("in", f"{usage.prompt_tokens:,} tokens"),
    ]
    if usage.cached_tokens:
        share = 100.0 * usage.cached_tokens / usage.prompt_tokens if usage.prompt_tokens else 0.0
        rows.append(("cached", f"{usage.cached_tokens:,} tokens  ·  {share:.1f}% of in"))
    rows.append(("out", f"{usage.completion_tokens:,} tokens"))
    if usage.reasoning_tokens:
        rows[-1] = (
            "out",
            f"{usage.completion_tokens:,} tokens  ·  reasoning {usage.reasoning_tokens:,}",
        )
    rows.append(("total", f"{usage.prompt_tokens + usage.completion_tokens:,} tokens"))
    if max_context and usage.last_prompt_tokens:
        share = 100.0 * usage.last_prompt_tokens / max_context
        rows.append(
            (
                "context",
                f"{usage.last_prompt_tokens:,} / {max_context:,} tokens  ·  {share:.1f}% used",
            )
        )
    return rows


def _usage_note(usage: _Usage) -> str | None:
    if not usage.unreported:
        return None
    noun = "request" if usage.unreported == 1 else "requests"
    return f"{usage.unreported} {noun} reported no token usage (stopped or failed)"


def _print_usage(convo: _Conversation, server: str, *, output_stream=None) -> None:
    output_stream = sys.stdout if output_stream is None else output_stream
    rows = _usage_rows(convo, _max_context(server))
    width = max(len(label) for label, _ in rows)
    for label, value in rows:
        print(f"{label:<{width}}  {value}", file=output_stream)
    note = _usage_note(convo.usage)
    if note:
        print(note, file=output_stream)


# -- entry point ---------------------------------------------------------------


def run(args: argparse.Namespace, *, input_stream=None, output_stream=None) -> int:
    input_stream = sys.stdin if input_stream is None else input_stream
    output_stream = sys.stdout if output_stream is None else output_stream
    server = str(args.server).rstrip("/")
    try:
        settings = _Settings(args)
    except ValueError as exc:
        print(f"hipengine chat: {exc}", file=sys.stderr)
        return 2
    console = None
    if not getattr(args, "plain", False) and _is_tty(input_stream) and _is_tty(output_stream):
        console = _make_console(output_stream)
    try:
        model = _resolve_model(server, args.model)
    except _REQUEST_ERRORS as exc:
        _print_connect_error(server, exc, console)
        return 1
    if console is not None:
        return _RichChat(console, server, model, settings).loop()
    try:
        _print_status(server, model, output_stream=output_stream)
    except _REQUEST_ERRORS as exc:
        _print_connect_error(server, exc, None)
        return 1
    return _run_plain(settings, server, model, input_stream, output_stream)


def main(argv: Sequence[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


def _is_tty(stream) -> bool:
    try:
        return bool(stream.isatty())
    except (AttributeError, ValueError):
        return False


def _make_console(output_stream):
    try:
        from rich.console import Console
        from rich.theme import Theme
    except ImportError:
        return None
    return Console(file=output_stream, theme=Theme(_THEME), highlight=False)


def _print_connect_error(server: str, exc: Exception, console) -> None:
    hint = (
        "Start a server first with `hipengine serve --model <path>` "
        "or pass --server for another endpoint."
    )
    if console is None:
        print(f"hipengine chat: cannot connect to {server}: {exc}\n{hint}", file=sys.stderr)
        return
    from rich.panel import Panel
    from rich.text import Text

    console.print(
        Panel(
            Text.assemble((f"{exc}\n\n", "hip.dim"), hint),
            title=Text(f" ✗ cannot connect to {server} ", style="hip.err"),
            title_align="left",
            border_style="hip.err",
            expand=False,
        )
    )


# -- plain mode ----------------------------------------------------------------


def _render_markdown_plain(text: str, output_stream) -> None:
    in_code = False
    for line in str(text).splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            print(f"  {line}", file=output_stream)
            continue
        line = re.sub(r"^#{1,6}\s+", "", line)
        line = re.sub(r"^\s*[-*]\s+", "- ", line)
        line = re.sub(r"\*\*(.+?)\*\*", r"\1", line)
        line = re.sub(r"`(.+?)`", r"\1", line)
        print(line, file=output_stream)


def _run_plain(settings: _Settings, server: str, model: str, input_stream, output_stream) -> int:
    convo = _Conversation(settings)
    print("Type /help for commands. Ctrl-C, Ctrl-D, or /quit exits.", file=output_stream)
    if _is_tty(output_stream) and _make_console(output_stream) is None:
        print("Tip: pip install 'hipengine[chat]' for the rich terminal UI.", file=output_stream)
    while True:
        try:
            print("you> ", end="", file=output_stream, flush=True)
            line = input_stream.readline()
        except KeyboardInterrupt:
            print(file=output_stream)
            print("bye 👋", file=output_stream)
            return 0
        if not line:
            print(file=output_stream)
            print("bye 👋", file=output_stream)
            return 0
        prompt = line.rstrip("\n")
        if prompt.strip().startswith("/"):
            name, _, rest = prompt.strip().partition(" ")
            name = name.lower()
            if name in {"/quit", "/exit"}:
                return 0
            if name == "/retry":
                if not convo.begin_retry():
                    print("Nothing to retry.", file=output_stream)
                    continue
            else:
                _plain_command(name, rest, settings, convo, server, model, output_stream)
                continue
        elif not prompt.strip():
            continue
        else:
            convo.turns.append({"role": "user", "content": prompt})
        print("assistant> ", end="", file=output_stream, flush=True)
        answer_parts: list[str] = []
        usage: dict | None = None
        cached_tokens: int | None = None
        failure: Exception | None = None
        try:
            for kind, text in _stream_events(
                server, model=model, messages=convo.messages(), fields=settings.request_fields()
            ):
                if kind == "content":
                    answer_parts.append(str(text))
                elif kind == "usage":
                    usage = text if isinstance(text, dict) else None
                elif kind == "meta" and isinstance(text, dict):
                    cached = _meta_cached_tokens(text)
                    if cached is not None:
                        cached_tokens = cached
                elif kind == "reasoning" and settings.show_thinking:
                    print(text, end="", file=output_stream, flush=True)
        except KeyboardInterrupt:
            print("\n[stopped]", file=output_stream)
        except (HTTPError, URLError, OSError, ValueError) as exc:
            failure = exc
        convo.usage.add(usage, cached_tokens=cached_tokens)
        if failure is not None:
            print(f"\nrequest failed: {_http_error_text(failure)}", file=sys.stderr)
            convo.abandon()
            continue
        print(file=output_stream)
        answer = "".join(answer_parts)
        if not answer:
            convo.abandon()
            continue
        _render_markdown_plain(answer, output_stream)
        convo.turns.append({"role": "assistant", "content": answer})


def _plain_command(name, rest, settings, convo, server, model, output_stream) -> None:
    if name == "/help":
        width = max(len(command) for command, _ in _COMMANDS)
        for command, text in _COMMANDS:
            print(f"{command:<{width}}  {text}", file=output_stream)
    elif name == "/clear":
        convo.clear()
        print("Conversation cleared.", file=output_stream)
    elif name == "/params":
        for command, value in settings.rows():
            print(f"{command:<7} {value}", file=output_stream)
    elif name == "/status":
        try:
            _print_status(server, model, output_stream=output_stream)
        except _REQUEST_ERRORS as exc:
            print(f"status unavailable: {exc}", file=sys.stderr)
    elif name == "/usage":
        _print_usage(convo, server, output_stream=output_stream)
    elif name == "/bench":
        _plain_bench(rest, settings, server, model, output_stream)
    else:
        try:
            message = settings.command(name, rest)
        except ValueError as exc:
            message = str(exc)
        print(message or f"Unknown command {name}; try /help.", file=output_stream)


def _plain_bench(rest: str, settings: _Settings, server: str, model: str, output_stream) -> None:
    """Run /bench in the line-oriented client; same measurements, plain rows."""

    try:
        prompt_tokens, max_tokens = _bench_args(rest)
    except ValueError as exc:
        print(str(exc), file=output_stream)
        return
    print(
        f"bench: measuring cold and cached requests (~{prompt_tokens:,} prompt tokens)…",
        file=output_stream,
        flush=True,
    )
    try:
        bench = _run_bench(server, settings, model=model, prompt_tokens=prompt_tokens, max_tokens=max_tokens)
    except KeyboardInterrupt:
        print("bench cancelled", file=output_stream)
        return
    except _REQUEST_ERRORS as exc:
        print(f"bench failed: {_http_error_text(exc)}", file=sys.stderr)
        return
    print(bench.header(), file=output_stream)
    for line in _bench_plain_lines(bench):
        print(line, file=output_stream)
    for note in bench.notes:
        print(f"  ! {note}", file=output_stream)


# -- rich mode -----------------------------------------------------------------


@contextlib.contextmanager
def _quiet_tty(stream=None):
    """While streaming: no keystroke echo, and Ctrl-C must not flush queued output.

    Without NOFLSH the tty driver discards pending output on SIGINT, which can drop
    Rich Live's erase sequence and leave a ghost of the streaming view on screen.
    A stream that cannot be reconfigured (no termios, no file descriptor, not a tty)
    is used as-is rather than aborting the reply.
    """

    stream = sys.stdin if stream is None else stream
    try:
        import termios
    except ImportError:
        yield
        return
    try:
        fd = stream.fileno()
        saved = termios.tcgetattr(fd)
        quiet = list(saved)
        quiet[3] = (quiet[3] | termios.NOFLSH) & ~(termios.ECHO | getattr(termios, "ECHOCTL", 0))
        termios.tcsetattr(fd, termios.TCSANOW, quiet)
    except (AttributeError, OSError, ValueError, termios.error):
        yield
        return
    try:
        yield
    finally:
        try:
            termios.tcsetattr(fd, termios.TCSANOW, saved)
        except (OSError, ValueError, termios.error) as exc:
            # The tty went away mid-stream; warn instead of losing the reply.
            print(f"hipengine chat: could not restore terminal settings: {exc}", file=sys.stderr)


def _prompt_bindings():
    """Key bindings for the prompt_toolkit reader.

    ``alt-enter`` inserts a newline. Ctrl-C clears typed text first, and quits at
    an empty prompt (Ctrl-D's behaviour), so an accidental Ctrl-C mid-sentence
    does not discard the session.
    """

    from prompt_toolkit.key_binding import KeyBindings

    bindings = KeyBindings()

    @bindings.add("escape", "enter")
    def _newline(event) -> None:
        event.current_buffer.insert_text("\n")

    @bindings.add("c-c")
    @bindings.add("<sigint>")
    def _interrupt(event) -> None:
        if event.current_buffer.text:
            event.current_buffer.reset()
            event.app.invalidate()
        else:
            event.app.exit(exception=EOFError)

    return bindings


class _Tail:
    """Render a renderable but keep only the bottom lines that fit the terminal."""

    def __init__(self, renderable, reserve: int = 3) -> None:
        self.renderable = renderable
        self.reserve = reserve

    def __rich_console__(self, console, options):
        from rich.segment import Segment

        height = max(4, console.size.height - self.reserve)
        lines = console.render_lines(self.renderable, options, pad=False)
        for line in lines[-height:]:
            yield from line
            yield Segment.line()


class _Turn:
    """Mutable state of one streaming reply; also the Live renderable.

    Besides the streamed text it keeps the server's own metadata: engine timing
    (``backend_prefill_ms``, ``decode_tokens_per_second``, ...) and per-request
    backend diagnostics, which is where prefix-cache reuse is reported. Those
    numbers are measurements of the engine; the client-side clocks remain the
    fallback for a server that reports neither.
    """

    def __init__(self, settings: _Settings) -> None:
        self.settings = settings
        self.content: list[str] = []
        self.reasoning: list[str] = []
        self.usage: dict | None = None
        self.meta: dict = {}
        self.started = time.perf_counter()
        self.first_token: float | None = None
        self.first_content: float | None = None
        self.finished: float | None = None
        self.deltas = 0

    def add(self, kind: str, value: object) -> None:
        if kind == "usage":
            self.usage = value if isinstance(value, dict) else None
            return
        if kind == "meta":
            # Server metadata is cumulative, so the last event carrying a key owns
            # its final value; the done chunk carries the completed diagnostics.
            if isinstance(value, dict):
                self.meta.update(value)
            return
        now = time.perf_counter()
        if self.first_token is None:
            self.first_token = now
        self.deltas += 1
        if kind == "reasoning":
            self.reasoning.append(str(value))
        else:
            if self.first_content is None:
                self.first_content = now
            self.content.append(str(value))

    def completion_tokens(self) -> int:
        tokens = (self.usage or {}).get("completion_tokens")
        return tokens if isinstance(tokens, int) else self.deltas

    def prompt_tokens(self) -> int | None:
        """Prompt tokens the server billed for this request."""

        return _as_int((self.usage or {}).get("prompt_tokens"))

    def prompt_token_count(self) -> int | None:
        """Prompt length the request was served, from usage or from backend diagnostics."""

        tokens = self.prompt_tokens()
        if tokens is not None:
            return tokens
        executed = self.prefill_tokens()
        if executed is None:
            return None
        return executed + (self.cached_tokens() or 0)

    def timing(self) -> dict:
        timing = self.meta.get("timing")
        return timing if isinstance(timing, dict) else {}

    def prefix_cache(self) -> dict | None:
        """Per-request prefix-cache telemetry, when the backend published it."""

        diagnostics = self.meta.get("diagnostics")
        block = diagnostics.get("prefix_cache") if isinstance(diagnostics, dict) else None
        return block if isinstance(block, dict) else None

    def cached_tokens(self) -> int | None:
        """Prompt tokens the server served from its prefix cache.

        Prefers the OpenAI/vLLM usage field and falls back to the backend's own
        prefix-cache diagnostics, which older servers report without it.
        """

        details = (self.usage or {}).get("prompt_tokens_details")
        cached = _as_int(details.get("cached_tokens")) if isinstance(details, dict) else None
        return cached if cached is not None else _meta_cached_tokens(self.meta)

    def prefill_tokens(self) -> int | None:
        """Prompt tokens the server actually prefilled, excluding cache reuse."""

        block = self.prefix_cache()
        executed = _as_int(block.get("executed_prefill_tokens")) if block else None
        if executed is not None:
            return executed
        prompt = self.prompt_tokens()
        if prompt is None:
            return None
        return max(0, prompt - (self.cached_tokens() or 0))

    def prefill_seconds(self) -> float | None:
        """Seconds spent prefilling: the engine's own timer, else the client's ttft."""

        reported = _as_float(self.timing().get("backend_prefill_ms"))
        if reported is not None and reported > 0:
            return reported / 1000.0
        if self.first_token is None:
            return None
        return max(0.0, self.first_token - self.started)

    def prefill_rate(self) -> float | None:
        """Uncached prompt tokens per second over the prefill window."""

        seconds = self.prefill_seconds()
        tokens = self.prefill_tokens()
        if seconds is None or seconds <= 0 or not tokens:
            return None
        return tokens / seconds

    def prompt_rate(self) -> float | None:
        """Prompt tokens per second end to end, counting tokens the cache served.

        The engine prefills only the uncached remainder, so a cache hit raises this
        rate far above the engine's own prefill rate for the tokens it still ran.
        Both are reported: this one says what the request cost, the other says how
        fast the engine prefills.
        """

        seconds = self.prefill_seconds()
        tokens = self.prompt_token_count()
        if seconds is None or seconds <= 0 or not tokens:
            return None
        return tokens / seconds

    def decode_seconds(self) -> float | None:
        """Seconds between the first streamed token and the end of the reply."""

        if self.first_token is None:
            return None
        end = self.finished or time.perf_counter()
        return max(0.0, end - self.first_token)

    def decode_rate(self) -> float | None:
        """Output tokens per second, preferring the engine's own measurement."""

        reported = _as_float(self.timing().get("decode_tokens_per_second"))
        if reported is not None and reported > 0:
            return reported
        seconds = self.decode_seconds()
        tokens = self.completion_tokens()
        if seconds is None or seconds <= 0 or tokens <= 1:
            return None
        return (tokens - 1) / seconds

    def decode_ms_per_token(self) -> float | None:
        """Milliseconds per output token: the inverse of the decode rate."""

        rate = self.decode_rate()
        if rate is None or rate <= 0:
            return None
        return 1000.0 / rate

    def thinking_seconds(self) -> float:
        if self.first_token is None or not self.reasoning:
            return 0.0
        end = self.first_content or self.finished or time.perf_counter()
        return end - self.first_token

    def __rich_console__(self, console, options):
        from rich.markdown import Markdown
        from rich.padding import Padding
        from rich.spinner import Spinner
        from rich.text import Text

        if not self.content:
            if self.reasoning:
                label = f"thinking · {self.thinking_seconds():.1f}s · {self.deltas} tokens"
            else:
                label = f"waiting for first token · {time.perf_counter() - self.started:.1f}s"
            label = Text(label + "   ctrl-c to stop", style="hip.dim")
            yield Padding(Spinner("dots", text=label, style="hip.accent"), (0, 0, 0, 2))
            if self.reasoning and self.settings.show_thinking:
                thought = Text("".join(self.reasoning), style="hip.thinking")
                yield _Tail(Padding(thought, (0, 0, 0, 4)), reserve=4)
            return
        markdown = Markdown("".join(self.content) + " ▍", code_theme="monokai")
        yield _Tail(Padding(markdown, (0, 0, 0, 2)))


# -- /bench --------------------------------------------------------------------

_BENCH_SENTENCE = (
    "The prefix cache keeps block-aligned prompt state so a later request can skip "
    "prefilling tokens it has already processed. "
)
# Rough tokens per sentence; the server reports the exact prompt length in usage.
_BENCH_SENTENCE_TOKENS = 24
_BENCH_DEFAULT_PROMPT_TOKENS = 512
_BENCH_DEFAULT_MAX_TOKENS = 32
# Route label -> the request's speculative_mtp value.
_BENCH_ROUTES = (("mtp on", True), ("mtp off", False))


def _bench_prompt(tokens: int, route: str) -> str:
    """Build filler text of roughly ``tokens`` tokens for one bench route.

    The nonce makes each route's prompt unique, so its cold measurement cannot
    hit an entry an earlier run left behind. Both requests of a route share the
    same text, which is what the cached measurement matches against.
    """

    repeats = max(1, round(tokens / _BENCH_SENTENCE_TOKENS))
    nonce = f"{os.getpid():x}-{time.time_ns():x}"
    return f"[bench {route} {nonce}] " + _BENCH_SENTENCE * repeats + "\nReply with one short sentence."


def _bench_args(rest: str) -> tuple[int, int]:
    """Parse ``/bench [prompt_tokens] [max_tokens]``."""

    parts = rest.split()
    if len(parts) > 2:
        raise ValueError("usage: /bench [prompt_tokens] [max_tokens]")
    values: list[int] = []
    for part in parts:
        try:
            value = int(part)
        except ValueError:
            raise ValueError(
                f"{part!r} is not a token count; usage: /bench [prompt_tokens] [max_tokens]"
            ) from None
        if value <= 0:
            raise ValueError("bench token counts must be positive")
        values.append(value)
    prompt_tokens = values[0] if values else _BENCH_DEFAULT_PROMPT_TOKENS
    max_tokens = values[1] if len(values) > 1 else _BENCH_DEFAULT_MAX_TOKENS
    return prompt_tokens, max_tokens


def _measure(
    server: str,
    settings: _Settings,
    *,
    model: str,
    messages: list[dict[str, str]],
    fields: dict,
) -> _Turn:
    """Run one request to completion and return its measured turn state."""

    turn = _Turn(settings)
    for kind, value in _stream_events(server, model=model, messages=messages, fields=fields):
        turn.add(kind, value)
    turn.finished = time.perf_counter()
    return turn


def _mtp_block(turn: _Turn) -> dict | None:
    diagnostics = turn.meta.get("diagnostics")
    block = diagnostics.get("specdec2_mtp2") if isinstance(diagnostics, dict) else None
    return block if isinstance(block, dict) else None


def _mtp_cycles(turn: _Turn) -> int | None:
    """Speculative cycles the backend committed, when it reported MTP intent."""

    block = _mtp_block(turn)
    return None if block is None else _as_int(block.get("cycles"))


def _mtp_decline_reason(turn: _Turn) -> str | None:
    """Why a request that asked for MTP ran without a speculative cycle."""

    block = _mtp_block(turn)
    if block is None:
        return None
    for key in ("provider_decline_reason", "prompt_fallback_reason", "activation_reason", "plan_reason"):
        value = block.get(key)
        if isinstance(value, str) and value:
            return f"{key}={value}"
    return None


def _bench_prefill_cell(turn: _Turn) -> str:
    """Engine prefill throughput, over the tokens it actually ran."""

    rate = turn.prefill_rate()
    return "—" if rate is None else f"{rate:.0f}"


def _bench_prompt_cell(turn: _Turn) -> str:
    """Whole-prompt throughput, counting the tokens the cache served for free.

    On a hit this is the number the cache bought, against the engine's own prefill
    rate beside it; on a miss the two agree, because nothing was reused.
    """

    rate = turn.prompt_rate()
    return "—" if rate is None else f"{rate:,.0f}"


def _bench_decode_cell(turn: _Turn) -> str:
    rate = turn.decode_rate()
    return "—" if rate is None else f"{rate:.1f}"


def _bench_cache_cell(turn: _Turn) -> str:
    cached = turn.cached_tokens()
    if cached:
        return f"hit {cached:,}"
    block = turn.prefix_cache()
    reason = block.get("fallback_reason") if block else None
    if isinstance(reason, str) and reason and reason != "miss":
        return f"miss ({reason})"
    return "miss"


def _bench_cache_style(cell: str) -> str:
    """Colour a /bench cache outcome: mint for reuse, coral for none."""

    if cell.startswith("hit"):
        return "hip.hit"
    return "hip.miss" if cell.startswith("miss") else ""


# /bench metric columns, in row order: prefill work, whole prompt, decode, ttft, tpot.
_BENCH_COLUMN_STYLES = (
    "hip.metric.prefill",
    "hip.metric.prompt",
    "hip.metric.decode",
    "hip.metric.ttft",
    "hip.metric.tpot",
)

# The table header carries the units, so every row stays a set of bare numbers: the
# MTP switch, the cache outcome, then one column each for throughput and latency.
_BENCH_HEADERS = (
    "mtp",
    "cache",
    "prefill tok/s",
    "prompt tok/s",
    "decode tok/s",
    "ttft s",
    "tpot ms",
)


def _bench_ttft_cell(turn: _Turn) -> str:
    if turn.first_token is None:
        return "—"
    return f"{turn.first_token - turn.started:.2f}"


def _bench_tpot_cell(turn: _Turn) -> str:
    per_token = turn.decode_ms_per_token()
    return "—" if per_token is None else f"{per_token:.1f}"


def _bench_plain_lines(bench: _Bench) -> list[str]:
    """The /bench table for the line-oriented client: same cells, padded columns."""

    rows = [_BENCH_HEADERS, *bench.rows]
    widths = [max(len(row[index]) for row in rows) for index in range(len(_BENCH_HEADERS))]
    lines = []
    for row in rows:
        cells = [
            cell.ljust(width) if index < 2 else cell.rjust(width)
            for index, (cell, width) in enumerate(zip(row, widths))
        ]
        lines.append("  " + "  ".join(cells).rstrip())
    return lines


class _Bench:
    """Measurements from one /bench: a cold request and its cached repeat, per route.

    ``rows`` returns ``(mtp, cache, prefill, prompt, decode, ttft, tpot)`` cells so
    the rich and plain clients show the same numbers. ``prefill`` covers the tokens
    the engine actually prefilled; ``prompt`` covers the whole prompt, so on a cache
    hit it is the one that shows what the cache bought. These are single interactive
    requests, not benchmark-harness artifacts: they include queueing and client
    transport, so read them as a sanity check rather than as a result.
    """

    def __init__(self, max_tokens: int) -> None:
        self.max_tokens = max_tokens
        self.prompt_tokens: int | None = None
        self.rows: list[tuple[str, str, str, str, str, str, str]] = []
        self.notes: list[str] = []

    def header(self) -> str:
        prompt = "prompt length unknown" if self.prompt_tokens is None else f"{self.prompt_tokens:,} prompt tokens"
        return f"bench · {prompt} · {self.max_tokens} max output · cold then cached per route"

    def add_run(self, route: str, cold: _Turn, cached: _Turn, *, speculative: bool) -> None:
        if self.prompt_tokens is None:
            self.prompt_tokens = cold.prompt_token_count()
        # The routes differ only in the MTP switch, so the column is just on/off and
        # the full route name is left to the notes, where it reads as a sentence.
        switch = "on" if speculative else "off"
        for turn in (cold, cached):
            self.rows.append(
                (
                    switch,
                    _bench_cache_cell(turn),
                    _bench_prefill_cell(turn),
                    _bench_prompt_cell(turn),
                    _bench_decode_cell(turn),
                    _bench_ttft_cell(turn),
                    _bench_tpot_cell(turn),
                )
            )
        if speculative:
            cycles = _mtp_cycles(cold)
            # A backend that reports no MTP accounting at all is not evidence of a
            # declined request, so only a reported-but-empty cycle count is noted.
            if cycles == 0:
                reason = _mtp_decline_reason(cold)
                self.notes.append(f"{route}: no speculative cycle ran" + (f" ({reason})" if reason else ""))


def _run_bench(
    server: str,
    settings: _Settings,
    *,
    model: str,
    prompt_tokens: int,
    max_tokens: int,
) -> _Bench:
    """Measure a cold prefill/decode request and its cached repeat, per route.

    Each route sends the same prompt twice, so the second request should reuse
    what the first one cached. Routes get different prompts, which keeps one
    route's cold measurement from hitting the other route's entry. A route the
    server refuses is reported in ``notes`` instead of failing the whole bench.
    """

    fields = dict(settings.request_fields())
    fields["max_tokens"] = max_tokens
    bench = _Bench(max_tokens)
    for route, speculative in _BENCH_ROUTES:
        route_fields = {**fields, "speculative_mtp": speculative}
        messages = [{"role": "user", "content": _bench_prompt(prompt_tokens, route)}]
        try:
            cold = _measure(server, settings, model=model, messages=messages, fields=route_fields)
            cached = _measure(server, settings, model=model, messages=messages, fields=route_fields)
        except _REQUEST_ERRORS as exc:
            bench.notes.append(f"{route}: {_http_error_text(exc)}")
            continue
        bench.add_run(route, cold, cached, speculative=speculative)
    return bench


class _RichChat:
    def __init__(
        self,
        console,
        server: str,
        model: str,
        settings: _Settings,
        *,
        read_line: Callable[[], str] | None = None,
    ) -> None:
        self.console = console
        self.server = server
        self.model = model
        self.settings = settings
        self.convo = _Conversation(settings)
        self.last_stats = ""
        self.last_rate = ""
        self.read_line = read_line or self._make_reader()

    # -- input -----------------------------------------------------------------

    def _make_reader(self) -> Callable[[], str]:
        try:
            from prompt_toolkit import PromptSession
            from prompt_toolkit.completion import WordCompleter
            from prompt_toolkit.history import FileHistory, InMemoryHistory
            from prompt_toolkit.styles import Style
        except ImportError:
            try:
                import readline  # noqa: F401  (line editing for console.input)
            except ImportError:
                pass
            from rich.text import Text

            return lambda: self.console.input(Text("❯ ", style="hip.user"))

        state = Path(os.environ.get("XDG_STATE_HOME") or Path.home() / ".local" / "state")
        try:
            (state / "hipengine").mkdir(parents=True, exist_ok=True)
            history = FileHistory(str(state / "hipengine" / "chat_history"))
        except OSError:
            history = InMemoryHistory()
        style = Style.from_dict(
            {
                "prompt": "#5fd7ff bold",
                "continuation": "#4e4e4e",
                "bottom-toolbar": "noreverse bg:default #6c6c6c",
                "bottom-toolbar.model": "noreverse bg:default #d787ff",
            }
        )

        def toolbar():
            sampling = self.settings.sampling
            parts = [
                ("class:bottom-toolbar.model", f" ◆ {self.model}"),
                ("", f"  ·  think {self.settings.think}"),
                ("", f"  ·  temp {_fmt(sampling['temperature'])}"),
            ]
            if self.last_rate:
                parts.append(("", f"  ·  last {self.last_rate}"))
            parts.append(("", "  ·  alt-enter newline  ·  /help"))
            return parts

        commands = sorted({name for spec, _ in _COMMANDS for name in spec.split() if name.startswith("/")})
        session = PromptSession(
            history=history,
            completer=WordCompleter(commands, WORD=True),
            complete_while_typing=True,
            key_bindings=_prompt_bindings(),
            style=style,
            bottom_toolbar=toolbar,
            prompt_continuation=lambda width, line, wrap: [("class:continuation", "┆ ".rjust(width))],
        )
        return lambda: session.prompt([("class:prompt", "❯ ")])

    # -- rendering -------------------------------------------------------------

    def note(self, text: str, style: str = "hip.dim") -> None:
        from rich.text import Text

        self.console.print(Text(f"  {text}", style=style))

    def status_card(self):
        from rich.panel import Panel
        from rich.progress_bar import ProgressBar
        from rich.table import Table
        from rich.text import Text

        grid = Table.grid(padding=(0, 2))
        grid.add_column(style="hip.label", justify="right")
        grid.add_column()
        grid.add_row("model", Text(self.model, style="bold"))
        try:
            fields = _status_fields(_get_json(f"{self.server}/ready"))
        except _REQUEST_ERRORS as exc:
            grid.add_row(
                "server",
                Text.assemble(self.server, ("  ● unavailable  ", "hip.err"), (str(exc), "hip.dim")),
            )
            fields = None
        else:
            grid.add_row("server", Text.assemble(self.server, ("  ● ready", "hip.ok")))
        if fields is not None:
            context = fields["context"]
            context_text = f"{context:,} tokens" if isinstance(context, int) else str(context)
            grid.add_row(
                "context",
                Text.assemble(context_text, ("  ·  concurrency ", "hip.label"), str(fields["concurrency"])),
            )
            grid.add_row("kv cache", f"{fields['kv_storage']} / {fields['kv_scale']}")
            current, budget = fields["pool_current"], fields["pool_budget"]
            if isinstance(budget, (int, float)) and budget > 0 and isinstance(current, (int, float)):
                pool = Table.grid(padding=(0, 2))
                pool.add_row(
                    ProgressBar(
                        total=float(budget),
                        completed=float(current),
                        width=24,
                        complete_style="hip.accent",
                        style="hip.border",
                    ),
                    f"{_gib(current)} / {_gib(budget)}",
                )
                grid.add_row("kv pool", pool)
            else:
                grid.add_row("kv pool", f"{_gib(current)} / automatic")
        grid.add_row("reasoning", Text.assemble(self.settings.think, ("  ·  /think to change", "hip.dim")))
        system = self.settings.system
        if system:
            grid.add_row("system", Text(system if len(system) <= 60 else system[:57] + "…", style="hip.dim"))
        return Panel(
            grid,
            title=Text.assemble(" ◆ ", ("hipEngine", "hip.accent"), " chat "),
            title_align="left",
            subtitle=Text(" /help · ctrl-c / ctrl-d to exit ", style="hip.dim"),
            subtitle_align="right",
            border_style="hip.border",
            padding=(0, 1),
            expand=False,
        )

    def help_table(self):
        from rich.padding import Padding
        from rich.table import Table

        table = Table.grid(padding=(0, 3))
        table.add_column(style="hip.user", no_wrap=True)
        table.add_column(style="hip.dim")
        for name, text in _COMMANDS:
            table.add_row(name, text)
        table.add_row("alt-enter", "insert a newline")
        table.add_row("ctrl-c", "stop generation, clear input, or exit (at an empty prompt)")
        table.add_row("ctrl-d", "exit")
        return Padding(table, (0, 0, 0, 2))

    def params_table(self):
        from rich.padding import Padding
        from rich.table import Table

        table = Table.grid(padding=(0, 3))
        table.add_column(style="hip.user", no_wrap=True)
        table.add_column()
        for name, value in self.settings.rows():
            table.add_row(name, value)
        return Padding(table, (0, 0, 0, 2))

    def usage_table(self):
        from rich.padding import Padding
        from rich.table import Table
        from rich.text import Text

        table = Table.grid(padding=(0, 3))
        table.add_column(style="hip.label", justify="right")
        table.add_column()
        for label, value in _usage_rows(self.convo, _max_context(self.server)):
            table.add_row(label, Text(value, style=_USAGE_STYLES.get(label, "")))
        note = _usage_note(self.convo.usage)
        if note:
            table.add_row("", Text(note, style="hip.dim"))
        return Padding(table, (0, 0, 0, 2))

    # -- loop ------------------------------------------------------------------

    def loop(self) -> int:
        self.console.print(self.status_card())
        while True:
            self.console.print()
            try:
                line = self.read_line()
            except (EOFError, KeyboardInterrupt):
                # Ctrl-D, or Ctrl-C at an empty prompt (see _prompt_bindings).
                self.note("bye 👋")
                return 0
            prompt = line.strip()
            if not prompt:
                continue
            if prompt.startswith("/"):
                if self.command(prompt):
                    return 0
                continue
            self.convo.pending = None
            self.convo.turns.append({"role": "user", "content": line})
            self.respond()

    def command(self, prompt: str) -> bool:
        """Handle a slash command; return True to exit."""

        name, _, rest = prompt.partition(" ")
        name = name.lower()
        if name in {"/quit", "/exit"}:
            return True
        if name == "/help":
            self.console.print(self.help_table())
        elif name == "/status":
            self.console.print(self.status_card())
        elif name == "/usage":
            self.console.print(self.usage_table())
        elif name == "/params":
            self.console.print(self.params_table())
        elif name == "/clear":
            self.convo.clear()
            self.last_stats = self.last_rate = ""
            self.console.clear()
            self.console.print(self.status_card())
            self.note("conversation cleared")
        elif name == "/retry":
            if self.convo.begin_retry():
                self.respond()
            else:
                self.note("nothing to retry", "hip.warn")
        elif name == "/bench":
            self.run_bench(rest)
        else:
            try:
                message = self.settings.command(name, rest)
            except ValueError as exc:
                self.note(str(exc), "hip.warn")
                return False
            if message is None:
                self.note(f"unknown command {name} — try /help", "hip.warn")
            else:
                self.note(message)
        return False

    def bench_table(self, bench: _Bench):
        from rich.padding import Padding
        from rich.table import Table
        from rich.text import Text

        table = Table.grid(padding=(0, 2))
        table.add_column(style="hip.label", no_wrap=True)
        table.add_column(no_wrap=True)
        for _ in range(5):
            # Numbers right-align under their units, so a column reads down.
            table.add_column(no_wrap=True, justify="right")
        table.add_row(*[Text(name, style="hip.dim") for name in _BENCH_HEADERS])
        for row in bench.rows:
            cells = [Text(row[0], style="hip.label"), Text(row[1], style=_bench_cache_style(row[1]))]
            cells.extend(Text(cell, style=style) for cell, style in zip(row[2:], _BENCH_COLUMN_STYLES))
            table.add_row(*cells)
        return Padding(table, (0, 0, 0, 2))

    def run_bench(self, rest: str) -> None:
        """Run /bench: prefill, decode, and prefix-cache reuse for each route."""

        try:
            prompt_tokens, max_tokens = _bench_args(rest)
        except ValueError as exc:
            self.note(str(exc), "hip.warn")
            return
        self.note(f"bench · measuring cold and cached requests (~{prompt_tokens:,} prompt tokens)…")
        try:
            bench = _run_bench(
                self.server,
                self.settings,
                model=self.model,
                prompt_tokens=prompt_tokens,
                max_tokens=max_tokens,
            )
        except KeyboardInterrupt:
            self.note("bench cancelled", "hip.warn")
            return
        except _REQUEST_ERRORS as exc:
            self.note(f"✗ bench failed: {_http_error_text(exc)}", "hip.err")
            return
        self.note(bench.header())
        self.console.print(self.bench_table(bench))
        for note in bench.notes:
            self.note(f"! {note}", "hip.warn")

    def respond(self) -> None:
        from rich.live import Live
        from rich.markdown import Markdown
        from rich.padding import Padding
        from rich.text import Text

        console = self.console
        console.print(Text.assemble(("◆ ", "hip.accent"), (self.model, "hip.label")))
        turn = _Turn(self.settings)
        error: Exception | None = None
        interrupted = False
        with _quiet_tty(), Live(turn, console=console, transient=True, refresh_per_second=15):
            try:
                for kind, value in _stream_events(
                    self.server,
                    model=self.model,
                    messages=self.convo.messages(),
                    fields=self.settings.request_fields(),
                ):
                    turn.add(kind, value)
            except KeyboardInterrupt:
                interrupted = True
            except (HTTPError, URLError, OSError, ValueError) as exc:
                error = exc
        turn.finished = time.perf_counter()
        self.convo.usage.add(turn.usage, cached_tokens=turn.cached_tokens())

        if turn.reasoning:
            summary = f"∴ thought for {turn.thinking_seconds():.1f}s"
            if self.settings.show_thinking:
                self.note(summary)
                thought = Text("".join(turn.reasoning).strip(), style="hip.thinking")
                console.print(Padding(thought, (0, 0, 0, 4)))
            else:
                self.note(summary + "  ·  /show to expand")
        answer = "".join(turn.content)
        if answer:
            console.print(Padding(Markdown(answer, code_theme="monokai"), (0, 0, 0, 2)))
            self.convo.turns.append({"role": "assistant", "content": answer})
        else:
            self.convo.abandon()

        self.last_stats, self.last_rate = self._stats(turn)
        if error is not None:
            self.note(f"✗ request failed: {_http_error_text(error)}", "hip.err")
        elif interrupted:
            kept = "partial reply kept" if answer else "no reply kept · /retry to resend"
            self.note(f"⏹ stopped  ·  {kept}  ·  {self.last_stats}", "hip.warn")
        else:
            self.note(self.last_stats)

    @staticmethod
    def _stats(turn: _Turn) -> tuple[str, str]:
        """Return (full stats line, decode-rate summary).

        Prefill covers the tokens the server actually prefilled. When a cache hit
        made that smaller than the prompt, the line also reports the whole-prompt
        rate, which is the number the cache is buying.
        """

        end = turn.finished or time.perf_counter()
        parts = [f"{turn.completion_tokens()} tokens"]
        cached = turn.cached_tokens()
        prefill = turn.prefill_rate()
        if prefill is not None:
            parts.append(f"prefill {prefill:.0f} tok/s")
        prompt = turn.prompt_rate() if cached else None
        if prompt is not None:
            parts.append(f"prompt {prompt:,.0f} tok/s")
        decode = turn.decode_rate()
        rate = "" if decode is None else f"{decode:.1f} tok/s"
        if rate:
            parts.append(f"decode {rate}")
        if cached:
            parts.append(f"cached {cached:,}")
        if turn.first_token is not None:
            parts.append(f"ttft {turn.first_token - turn.started:.2f}s")
        parts.append(f"{end - turn.started:.2f}s")
        return "  ·  ".join(parts), rate
