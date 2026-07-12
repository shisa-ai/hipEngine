#!/usr/bin/env python3
"""Record llama.cpp server prompt token IDs for MTP-GGUF parity checks.

The script expects an already-running llama.cpp `llama-server` and uses its
`POST /tokenize` endpoint.  It does not launch or configure the server; the
server command/model/build must be recorded by the caller's benchmark artifact.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Sequence
from urllib.parse import urljoin

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.gguf_prompt_token_inventory import (  # noqa: E402
    DEFAULT_PROMPTS,
    load_prompt_suite,
    select_prompts,
    sha256_text,
    sha256_token_ids,
)

TokenizeClient = Callable[[str], dict[str, Any]]


class ServerTokenizeError(RuntimeError):
    """Raised when llama.cpp /tokenize returns an invalid response."""


def normalize_server_url(server_url: str) -> str:
    if not server_url:
        raise ValueError("server_url must be non-empty")
    return server_url if server_url.endswith("/") else server_url + "/"


def make_llamacpp_tokenize_client(
    *,
    server_url: str,
    add_special: bool = False,
    parse_special: bool = True,
    with_pieces: bool = True,
    timeout: float = 30.0,
) -> TokenizeClient:
    """Return a callable that tokenizes one prompt with llama.cpp server."""

    endpoint = urljoin(normalize_server_url(server_url), "tokenize")

    def tokenize(content: str) -> dict[str, Any]:
        payload = json.dumps(
            {
                "content": content,
                "add_special": add_special,
                "parse_special": parse_special,
                "with_pieces": with_pieces,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            endpoint,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read().decode("utf-8")
        except urllib.error.URLError as exc:  # pragma: no cover - exercised in integration use
            raise ServerTokenizeError(f"failed to call llama.cpp /tokenize at {endpoint}: {exc}") from exc
        try:
            decoded = json.loads(body)
        except json.JSONDecodeError as exc:
            raise ServerTokenizeError(f"llama.cpp /tokenize returned invalid JSON: {body[:200]!r}") from exc
        if not isinstance(decoded, dict):
            raise ServerTokenizeError("llama.cpp /tokenize response was not a JSON object")
        return decoded

    return tokenize


def extract_token_ids_and_pieces(response: dict[str, Any]) -> tuple[list[int], list[Any] | None]:
    tokens = response.get("tokens")
    if not isinstance(tokens, list):
        raise ServerTokenizeError("llama.cpp /tokenize response missing list field 'tokens'")
    token_ids: list[int] = []
    pieces: list[Any] | None = []
    for token in tokens:
        if isinstance(token, int):
            token_ids.append(int(token))
            pieces = None
        elif isinstance(token, dict) and isinstance(token.get("id"), int):
            token_ids.append(int(token["id"]))
            if pieces is not None:
                pieces.append(token.get("piece"))
        else:
            raise ServerTokenizeError(f"invalid token entry from llama.cpp /tokenize: {token!r}")
    return token_ids, pieces


def build_llamacpp_prompt_token_inventory(
    *,
    prompts: Sequence[dict[str, str]],
    server_url: str,
    tokenizer: TokenizeClient,
    prompts_file: str | Path,
    model: str | Path | None = None,
    prompt_render: str = "raw",
    add_special: bool = False,
    parse_special: bool = True,
    with_pieces: bool = True,
) -> dict[str, Any]:
    if prompt_render != "raw":
        raise ValueError("only raw prompt rendering is currently supported")

    rows: list[dict[str, Any]] = []
    for prompt in prompts:
        source_text = str(prompt["prompt"])
        rendered_text = source_text
        token_ids, pieces = extract_token_ids_and_pieces(tokenizer(rendered_text))
        row: dict[str, Any] = {
            "name": str(prompt["name"]),
            "source_chars": len(source_text),
            "source_sha256": sha256_text(source_text),
            "rendered_chars": len(rendered_text),
            "rendered_sha256": sha256_text(rendered_text),
            "token_count": len(token_ids),
            "token_ids": token_ids,
            "token_ids_sha256": sha256_token_ids(token_ids),
        }
        if pieces is not None:
            row["token_pieces"] = pieces
        rows.append(row)

    return {
        "schema": 1,
        "kind": "llamacpp_prompt_token_inventory",
        "model": str(model) if model is not None else None,
        "server_url": server_url,
        "prompts_file": str(prompts_file),
        "prompt_render": prompt_render,
        "add_special": add_special,
        "parse_special": parse_special,
        "with_pieces": with_pieces,
        "tokenization": "llamacpp.server.tokenize",
        "warning": (
            "Compare this artifact to hipEngine with scripts/compare_prompt_token_inventories.py "
            "before comparing accepted/output metrics."
        ),
        "prompts": rows,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    suite = load_prompt_suite(args.prompts_file)
    prompts = select_prompts(suite, names_csv=args.prompt_names, limit=args.limit)
    client = make_llamacpp_tokenize_client(
        server_url=args.server_url,
        add_special=args.add_special,
        parse_special=args.parse_special,
        with_pieces=args.with_pieces,
        timeout=args.timeout,
    )
    return build_llamacpp_prompt_token_inventory(
        prompts=prompts,
        server_url=args.server_url,
        tokenizer=client,
        prompts_file=args.prompts_file,
        model=args.model,
        prompt_render=args.prompt_render,
        add_special=args.add_special,
        parse_special=args.parse_special,
        with_pieces=args.with_pieces,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-url", default="http://127.0.0.1:8080")
    parser.add_argument("--model", type=Path, help="model path used by the running server, for metadata only")
    parser.add_argument("--prompts-file", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--prompt-names", help="comma-separated prompt names to include")
    parser.add_argument("--limit", type=int, help="include only the first N selected prompts")
    parser.add_argument("--prompt-render", choices=("raw",), default="raw")
    parser.add_argument("--add-special", action="store_true")
    parser.add_argument("--parse-special", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--with-pieces", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--out", type=Path, help="write JSON to this path instead of stdout")
    args = parser.parse_args()

    payload = json.dumps(run(args), indent=2, ensure_ascii=False) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
