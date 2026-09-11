#!/usr/bin/env python3
"""Capture and compare repeatable OpenAI-compatible chat responses.

Use ``capture`` to preserve canonical assistant messages, response timing, and
hashes for every prompt and repetition. Use ``compare`` to check repeatability
within each capture and exact message equality across modes such as AR and MTP.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path
from typing import Any, Sequence


class CaptureError(RuntimeError):
    """Raised when an input or server response violates the capture contract."""


def sha256_json(value: object) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def load_prompts(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CaptureError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise CaptureError(f"prompt row at {path}:{line_number} must be an object")
        prompt_id = row.get("id") or row.get("name")
        messages = row.get("messages")
        if not isinstance(prompt_id, str) or not prompt_id:
            raise CaptureError(f"prompt row at {path}:{line_number} requires id or name")
        if prompt_id in seen:
            raise CaptureError(f"duplicate prompt id in {path}: {prompt_id}")
        if not isinstance(messages, list) or not messages:
            raise CaptureError(f"prompt {prompt_id!r} requires a non-empty messages list")
        seen.add(prompt_id)
        rows.append({"id": prompt_id, "category": row.get("category"), "messages": messages})
    if not rows:
        raise CaptureError(f"no prompts found in {path}")
    return rows


def canonical_message(response: dict[str, Any]) -> dict[str, str]:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise CaptureError("response requires a non-empty choices list")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise CaptureError("response choice requires a message object")
    reasoning = message.get("reasoning_content")
    if reasoning is None:
        reasoning = message.get("reasoning")
    content = message.get("content")
    if reasoning is not None and not isinstance(reasoning, str):
        raise CaptureError("assistant reasoning must be text or null")
    if content is not None and not isinstance(content, str):
        raise CaptureError("assistant content must be text or null")
    return {"reasoning_content": reasoning or "", "content": content or ""}


def request_payload(args: argparse.Namespace, messages: list[dict[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": args.model,
        "messages": messages,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "temperature": args.temperature,
        "ignore_eos": args.ignore_eos,
        "cache_prompt": args.cache_prompt,
        "stream": False,
    }
    if args.top_k is not None:
        payload["top_k"] = args.top_k
    return payload


def post_chat(url: str, payload: dict[str, Any], *, timeout: float) -> tuple[dict[str, Any], float]:
    request = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.load(response)
    elapsed = time.perf_counter() - start
    if not isinstance(body, dict):
        raise CaptureError("server response must be a JSON object")
    return body, elapsed


def summarize_results(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_id: dict[str, list[dict[str, Any]]] = {}
    for row in results:
        by_id.setdefault(str(row["id"]), []).append(row)
    prompts: dict[str, Any] = {}
    for prompt_id, rows in by_id.items():
        ordered = sorted(rows, key=lambda row: int(row["repetition"]))
        hashes = [str(row["message_sha256"]) for row in ordered]
        prompts[prompt_id] = {
            "category": ordered[0].get("category"),
            "repetitions": len(ordered),
            "unique_message_hashes": list(dict.fromkeys(hashes)),
            "repeat_exact": len(set(hashes)) == 1,
        }
    return {
        "prompt_count": len(prompts),
        "sample_count": len(results),
        "repeat_exact_prompts": sum(row["repeat_exact"] for row in prompts.values()),
        "all_repeat_exact": all(row["repeat_exact"] for row in prompts.values()),
        "prompts": prompts,
    }


def capture(args: argparse.Namespace) -> dict[str, Any]:
    prompts = load_prompts(args.prompts)
    results: list[dict[str, Any]] = []
    for repetition in range(args.repetitions):
        for prompt in prompts:
            payload = request_payload(args, prompt["messages"])
            response, wall_s = post_chat(args.url, payload, timeout=args.timeout)
            message = canonical_message(response)
            results.append(
                {
                    "id": prompt["id"],
                    "category": prompt["category"],
                    "repetition": repetition,
                    "wall_s": wall_s,
                    "message": message,
                    "message_sha256": sha256_json(message),
                    "timings": response.get("timings"),
                    "usage": response.get("usage"),
                }
            )
    common_payload = request_payload(args, [])
    common_payload.pop("messages")
    return {
        "schema": 1,
        "kind": "openai_chat_message_capture",
        "url": args.url,
        "prompt_file": str(args.prompts),
        "prompt_sha256": hashlib.sha256(args.prompts.read_bytes()).hexdigest(),
        "repetitions": args.repetitions,
        "payload": common_payload,
        "results": results,
        "summary": summarize_results(results),
    }


def parse_named_capture(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not name or not raw_path:
        raise argparse.ArgumentTypeError("captures must use NAME=PATH")
    return name, Path(raw_path)


def compare_captures(named_paths: Sequence[tuple[str, Path]]) -> dict[str, Any]:
    if len(named_paths) < 2:
        raise CaptureError("compare requires at least two captures")
    captures: dict[str, dict[str, Any]] = {}
    prompt_sha256: str | None = None
    prompt_ids: set[str] | None = None
    for name, path in named_paths:
        if name in captures:
            raise CaptureError(f"duplicate capture name: {name}")
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("kind") != "openai_chat_message_capture":
            raise CaptureError(f"{path} is not an OpenAI chat message capture")
        digest = data.get("prompt_sha256")
        ids = {str(row["id"]) for row in data.get("results", [])}
        if prompt_sha256 is not None and digest != prompt_sha256:
            raise CaptureError("captures use different prompt files")
        if prompt_ids is not None and ids != prompt_ids:
            raise CaptureError("captures contain different prompt ids")
        prompt_sha256 = str(digest)
        prompt_ids = ids
        captures[name] = data

    rows: list[dict[str, Any]] = []
    for prompt_id in sorted(prompt_ids or set()):
        hashes: dict[str, list[str]] = {}
        categories: set[str] = set()
        for name, data in captures.items():
            selected = [row for row in data["results"] if str(row["id"]) == prompt_id]
            selected.sort(key=lambda row: int(row["repetition"]))
            hashes[name] = [str(row["message_sha256"]) for row in selected]
            category = selected[0].get("category")
            if category is not None:
                categories.add(str(category))
        if len(categories) > 1:
            raise CaptureError(f"capture category mismatch for prompt {prompt_id}")
        mode_self_exact = {name: len(set(values)) == 1 for name, values in hashes.items()}
        all_hashes = [value for values in hashes.values() for value in values]
        rows.append(
            {
                "id": prompt_id,
                "category": next(iter(categories), None),
                "hashes": hashes,
                "mode_self_exact": mode_self_exact,
                "cross_capture_exact": len(set(all_hashes)) == 1,
            }
        )
    return {
        "schema": 1,
        "kind": "openai_chat_message_capture_comparison",
        "prompt_sha256": prompt_sha256,
        "captures": {name: str(path) for name, path in named_paths},
        "rows": rows,
        "total_rows": len(rows),
        "cross_capture_exact_rows": sum(row["cross_capture_exact"] for row in rows),
        "all_cross_capture_exact": all(row["cross_capture_exact"] for row in rows),
        "self_exact_rows": {
            name: sum(row["mode_self_exact"][name] for row in rows) for name in captures
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    capture_parser = subparsers.add_parser("capture", help="capture repeated chat responses")
    capture_parser.add_argument("--url", required=True, help="server base URL")
    capture_parser.add_argument("--prompts", required=True, type=Path, help="JSONL prompt suite")
    capture_parser.add_argument("--output", required=True, type=Path, help="capture JSON path")
    capture_parser.add_argument("--repetitions", type=int, default=2)
    capture_parser.add_argument("--timeout", type=float, default=300.0)
    capture_parser.add_argument("--model", default="llama")
    capture_parser.add_argument("--max-tokens", type=int, default=16)
    capture_parser.add_argument("--seed", type=int, default=42)
    capture_parser.add_argument("--temperature", type=float, default=0.0)
    capture_parser.add_argument("--top-k", type=int, default=1)
    capture_parser.add_argument("--ignore-eos", action=argparse.BooleanOptionalAction, default=True)
    capture_parser.add_argument("--cache-prompt", action=argparse.BooleanOptionalAction, default=False)

    compare_parser = subparsers.add_parser("compare", help="compare captures across modes")
    compare_parser.add_argument(
        "--capture",
        action="append",
        required=True,
        type=parse_named_capture,
        metavar="NAME=PATH",
        help="named capture; specify at least twice",
    )
    compare_parser.add_argument("--output", required=True, type=Path, help="comparison JSON path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "capture":
        if args.repetitions < 1:
            raise CaptureError("--repetitions must be at least 1")
        result = capture(args)
    else:
        result = compare_captures(args.capture)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
