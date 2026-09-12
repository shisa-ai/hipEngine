"""Small terminal client for a running hipEngine OpenAI-compatible server."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Iterator, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


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
    parser.add_argument("--max-tokens", type=int, default=512)
    return parser


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


def _print_status(server: str, model: str, *, output_stream=None) -> None:
    output_stream = sys.stdout if output_stream is None else output_stream
    ready = _get_json(f"{server.rstrip('/')}/ready")
    context = ready.get("context", {})
    queue = ready.get("queue", {})
    kv = ready.get("kv_capacity", {})
    pool = kv.get("pool") if isinstance(kv, dict) else {}
    concurrency = queue.get("max_active_requests", "auto")
    current_bytes = pool.get("current_bytes", 0) if isinstance(pool, dict) else 0
    budget_bytes = pool.get("budget_bytes") if isinstance(pool, dict) else None

    def gib(value):
        try:
            return f"{float(value) / 1024**3:.2f} GiB"
        except (TypeError, ValueError):
            return "unknown"

    print(
        f"[{model}] "
        f"context={context.get('effective_max_context_tokens', 'auto')} "
        f"concurrency={concurrency} "
        f"kv={kv.get('storage', 'unknown')}/{kv.get('scale_dtype', 'unknown')} "
        f"pool={gib(current_bytes)}/"
        f"{gib(budget_bytes) if budget_bytes is not None else 'automatic'}",
        file=output_stream,
    )


def _stream_completion(
    server: str,
    *,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
) -> Iterator[str]:
    body = json.dumps(
        {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
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
            choices = payload.get("choices", [])
            if not choices or not isinstance(choices[0], dict):
                continue
            delta = choices[0].get("delta", {})
            text = delta.get("content") if isinstance(delta, dict) else None
            if isinstance(text, str):
                yield text


def _render_markdown(text: str, output_stream) -> None:
    """Render completed Markdown with optional Rich and a dependency-free fallback."""

    try:
        from rich.console import Console
        from rich.markdown import Markdown

        Console(file=output_stream, force_terminal=output_stream.isatty()).print(
            Markdown(text)
        )
        return
    except (ImportError, AttributeError):
        pass

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


def run(args: argparse.Namespace, *, input_stream=None, output_stream=None) -> int:
    input_stream = sys.stdin if input_stream is None else input_stream
    output_stream = sys.stdout if output_stream is None else output_stream
    server = str(args.server).rstrip("/")
    try:
        model = _resolve_model(server, args.model)
        _print_status(server, model, output_stream=output_stream)
    except (HTTPError, URLError, OSError, RuntimeError, ValueError) as exc:
        print(
            f"hipengine chat: cannot connect to {server}: {exc}\n"
            "Start a server first with `hipengine serve --model <path>` "
            "or pass --server for another endpoint.",
            file=sys.stderr,
        )
        return 1

    messages: list[dict[str, str]] = []
    if args.system:
        messages.append({"role": "system", "content": str(args.system)})
    print("Type /help for commands. Ctrl-D or /quit exits.", file=output_stream)
    while True:
        try:
            print("you> ", end="", file=output_stream, flush=True)
            line = input_stream.readline()
        except KeyboardInterrupt:
            print(file=output_stream)
            return 0
        if not line:
            print(file=output_stream)
            return 0
        prompt = line.rstrip("\n")
        command = prompt.strip().lower()
        if command in {"/quit", "/exit"}:
            return 0
        if command == "/help":
            print("/status  show server limits\n/clear   clear conversation\n/quit    exit", file=output_stream)
            continue
        if command == "/clear":
            messages = messages[:1] if args.system else []
            print("Conversation cleared.", file=output_stream)
            continue
        if command == "/status":
            try:
                _print_status(server, model, output_stream=output_stream)
            except (HTTPError, URLError, OSError, RuntimeError, ValueError) as exc:
                print(f"status unavailable: {exc}", file=sys.stderr)
            continue
        if not prompt.strip():
            continue
        messages.append({"role": "user", "content": prompt})
        print("assistant> ", end="", file=output_stream, flush=True)
        answer_parts: list[str] = []
        try:
            for text in _stream_completion(
                server,
                model=model,
                messages=messages,
                temperature=float(args.temperature),
                max_tokens=int(args.max_tokens),
            ):
                answer_parts.append(text)
        except (HTTPError, URLError, OSError, ValueError) as exc:
            print(f"\nrequest failed: {exc}", file=sys.stderr)
            messages.pop()
            continue
        print(file=output_stream)
        _render_markdown("".join(answer_parts), output_stream)
        messages.append({"role": "assistant", "content": "".join(answer_parts)})


def main(argv: Sequence[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))
