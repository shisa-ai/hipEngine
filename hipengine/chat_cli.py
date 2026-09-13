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
    "hip.warn": "#ffaf5f",
    "hip.err": "bold #ff5f5f",
    "hip.border": "#4e4e4e",
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


def _stream_events(
    server: str,
    *,
    model: str,
    messages: list[dict[str, str]],
    fields: dict,
) -> Iterator[tuple[str, object]]:
    """Yield ``("content"|"reasoning", text)`` deltas and ``("usage", dict)``."""

    body = json.dumps(
        {
            "model": model,
            "messages": messages,
            **fields,
            "stream": True,
            "stream_options": {"include_usage": True},
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
            choices = payload.get("choices", [])
            if not choices or not isinstance(choices[0], dict):
                continue
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


class _Conversation:
    """Chat turns plus a stash of the last prompt whose reply was abandoned."""

    def __init__(self, settings: _Settings) -> None:
        self.settings = settings
        self.turns: list[dict[str, str]] = []
        self.pending: str | None = None

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
        try:
            for kind, text in _stream_events(
                server, model=model, messages=convo.messages(), fields=settings.request_fields()
            ):
                if kind == "content":
                    answer_parts.append(str(text))
                elif kind == "reasoning" and settings.show_thinking:
                    print(text, end="", file=output_stream, flush=True)
        except KeyboardInterrupt:
            print("\n[stopped]", file=output_stream)
        except (HTTPError, URLError, OSError, ValueError) as exc:
            print(f"\nrequest failed: {_http_error_text(exc)}", file=sys.stderr)
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
    else:
        try:
            message = settings.command(name, rest)
        except ValueError as exc:
            message = str(exc)
        print(message or f"Unknown command {name}; try /help.", file=output_stream)


# -- rich mode -----------------------------------------------------------------


@contextlib.contextmanager
def _quiet_tty(stream=None):
    """While streaming: no keystroke echo, and Ctrl-C must not flush queued output.

    Without NOFLSH the tty driver discards pending output on SIGINT, which can drop
    Rich Live's erase sequence and leave a ghost of the streaming view on screen.
    """

    stream = sys.stdin if stream is None else stream
    try:
        import termios

        fd = stream.fileno()
        saved = termios.tcgetattr(fd)
    except (ImportError, AttributeError, OSError, ValueError):
        yield
        return
    quiet = list(saved)
    quiet[3] = (quiet[3] | termios.NOFLSH) & ~(termios.ECHO | getattr(termios, "ECHOCTL", 0))
    try:
        termios.tcsetattr(fd, termios.TCSANOW, quiet)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSANOW, saved)


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
    """Mutable state of one streaming reply; also the Live renderable."""

    def __init__(self, settings: _Settings) -> None:
        self.settings = settings
        self.content: list[str] = []
        self.reasoning: list[str] = []
        self.usage: dict | None = None
        self.started = time.perf_counter()
        self.first_token: float | None = None
        self.first_content: float | None = None
        self.finished: float | None = None
        self.deltas = 0

    def add(self, kind: str, value: object) -> None:
        if kind == "usage":
            self.usage = value if isinstance(value, dict) else None
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
        """Return (full stats line, decode-rate summary)."""

        end = turn.finished or time.perf_counter()
        tokens = turn.completion_tokens()
        parts = [f"{tokens} tokens"]
        rate = ""
        if turn.first_token is not None:
            decode = end - turn.first_token
            if tokens > 1 and decode > 0:
                rate = f"{(tokens - 1) / decode:.1f} tok/s"
                parts.append(rate)
            parts.append(f"ttft {turn.first_token - turn.started:.2f}s")
        parts.append(f"{end - turn.started:.2f}s")
        return "  ·  ".join(parts), rate
