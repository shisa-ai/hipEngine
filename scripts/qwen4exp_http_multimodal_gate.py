#!/usr/bin/env python3
"""Multimodal serving gate: does the advertised image scope match behavior?

The resident route accepts one base64 PNG image in a blocking chat request. This
gate drives the live server and checks both halves of that claim: an image the
model can read is actually read, and every documented limit is refused with a
named reason rather than being silently ignored or crashing.

Run it against a server started with ``--vision-model <mmproj.gguf>``; the
text-only refusal is checked separately against a server started without it.
"""

from __future__ import annotations

import argparse
import base64
import json
import struct
import sys
import time
import zlib
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _png(width: int, height: int, rgb: tuple[int, int, int]) -> bytes:
    raw = b"".join(b"\x00" + bytes(rgb) * width for _ in range(height))

    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def _data_url(payload: bytes, mime: str = "image/png") -> str:
    return f"data:{mime};base64," + base64.b64encode(payload).decode()


def _check(checks: list[dict[str, Any]], name: str, passed: bool, **detail: Any) -> None:
    checks.append({"name": name, "passed": bool(passed), **detail})


def _chat(client: Any, url: str, model: str, content: Any, **extra: Any) -> tuple[int, dict[str, Any]]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.0,
        "max_tokens": extra.pop("max_tokens", 32),
        **extra,
    }
    response = client.post(f"{url}/v1/chat/completions", json=payload)
    return response.status_code, response.json()


def _error_code(body: dict[str, Any]) -> str | None:
    return ((body.get("error") or {}).get("code")) if isinstance(body, dict) else None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:18397")
    parser.add_argument("--model", default="qwen38fn-capon")
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument(
        "--expect-vision",
        action="store_true",
        help="the server was started with --vision-model, so an image must be read",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    import httpx

    url = args.url.rstrip("/")
    checks: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    client = httpx.Client(timeout=args.timeout)

    red = _data_url(_png(64, 64, (220, 20, 30)))
    image_content = [
        {"type": "text", "text": "What single color fills this image? Answer with one word."},
        {"type": "image_url", "image_url": {"url": red}},
    ]

    # Text still works with a vision sidecar attached.
    status, body = _chat(client, url, args.model, "Reply with exactly: TEXT-OK")
    text = ((body.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    _check(checks, "text.still_serves", status == 200 and "TEXT-OK" in text, status=status, text=text)

    status, body = _chat(client, url, args.model, image_content)
    details["image_status"] = status
    details["image_body"] = body if status != 200 else None
    if args.expect_vision:
        answer = ((body.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
        details["image_answer"] = answer
        _check(checks, "image.request_accepted", status == 200, status=status, body=body if status != 200 else None)
        _check(
            checks,
            "image.answer_names_the_color",
            status == 200 and "red" in answer.lower(),
            answer=answer,
        )
    else:
        _check(
            checks,
            "image.refused_without_vision",
            status == 400 and _error_code(body) in {"unsupported_content_type", "unsupported_parameter"},
            status=status,
            code=_error_code(body),
        )

    # Every documented limit must be refused with a named reason.
    limits: list[tuple[str, Any, dict[str, Any], str]] = [
        (
            "limit.jpeg_is_refused",
            [
                {"type": "text", "text": "What color?"},
                {
                    "type": "image_url",
                    "image_url": {"url": _data_url(b"\xff\xd8\xff\xe0" + b"0" * 64, "image/jpeg")},
                },
            ],
            {},
            "PNG",
        ),
        (
            "limit.remote_url_is_refused",
            [
                {"type": "text", "text": "What color?"},
                {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
            ],
            {},
            "data URL",
        ),
        (
            "limit.stream_is_refused",
            image_content,
            {"stream": True},
            "stream",
        ),
        (
            "limit.n_gt_1_is_refused",
            image_content,
            {"n": 2},
            "n=1",
        ),
        (
            "limit.tools_are_refused",
            image_content,
            {
                "tools": [
                    {
                        "type": "function",
                        "function": {"name": "noop", "parameters": {"type": "object", "properties": {}}},
                    }
                ]
            },
            "tools",
        ),
    ]
    for name, content, extra, marker in limits:
        if args.expect_vision:
            status, body = _chat(client, url, args.model, content, **extra)
            message = ((body.get("error") or {}).get("message") or "")
            _check(
                checks,
                name,
                status == 400 and marker.lower() in message.lower(),
                status=status,
                code=_error_code(body),
                message=message[:200],
            )
        else:
            status, body = _chat(client, url, args.model, content, **extra)
            _check(
                checks,
                name,
                status == 400,
                status=status,
                code=_error_code(body),
            )

    if args.expect_vision:
        # An oversized payload is a bounded refusal, not a memory event.
        oversized = _data_url(_png(1024, 1024, (10, 200, 30)) + b"\x00" * (9 * 1024 * 1024))
        status, body = _chat(client, url, args.model, [
            {"type": "text", "text": "What color?"},
            {"type": "image_url", "image_url": {"url": oversized}},
        ])
        message = ((body.get("error") or {}).get("message") or "")
        _check(
            checks,
            "limit.oversized_is_refused",
            status == 400 and "exceed" in message.lower(),
            status=status,
            code=_error_code(body),
            message=message[:200],
        )
        # Text must still serve after the refusals.
        status, body = _chat(client, url, args.model, "Reply with exactly: AFTER-OK")
        after = ((body.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
        _check(checks, "recovery.text_after_refusals", status == 200 and "AFTER-OK" in after,
               status=status, text=after)

    client.close()
    failures = [check for check in checks if not check["passed"]]
    summary = {
        "schema": "hipengine.qwen4exp.http_multimodal_gate.v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "url": url,
        "model": args.model,
        "expect_vision": bool(args.expect_vision),
        "checks": checks,
        "details": details,
        "check_count": len(checks),
        "failure_count": len(failures),
        "status": "passed" if not failures else "failed",
    }
    if args.output is not None:
        args.output.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    for check in checks:
        print(f"{'PASS' if check['passed'] else 'FAIL'} {check['name']}")
    print(json.dumps({"status": summary["status"], "checks": len(checks), "failures": len(failures)}))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
