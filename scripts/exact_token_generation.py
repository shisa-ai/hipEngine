#!/usr/bin/env python3
"""Create and compare exact-token direct/HTTP generation oracles."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib import error, request

from hipengine import LLM, SamplingParams
from hipengine.benchmark.exact_tokens import (
    DEFAULT_EXACT_TOKEN_FIXTURE,
    ExactTokenFixture,
    ExactTokenOracle,
    load_exact_token_fixture,
    validate_exact_token_parity,
)
from hipengine.benchmark.prompts import text_sha256, token_ids_sha256
from hipengine.benchmark.provenance import collect_artifact_provenance


REPO_ROOT = Path(__file__).resolve().parents[1]


class ExactTokenBenchError(RuntimeError):
    pass


def _generated_rows(outputs: Sequence[Any], *, expected_rows: int) -> tuple[tuple[int, ...], ...]:
    if len(outputs) != int(expected_rows):
        raise ExactTokenBenchError(
            f"generator returned {len(outputs)} rows; expected {expected_rows}"
        )
    rows: list[tuple[int, ...]] = []
    for index, output in enumerate(outputs):
        token_ids = getattr(output, "generated_token_ids", None)
        if token_ids is None:
            raise ExactTokenBenchError(f"direct output {index} omitted generated_token_ids")
        rows.append(tuple(int(token) for token in token_ids))
    return tuple(rows)


def parse_http_response(
    payload: Mapping[str, Any],
    *,
    prompt_rows: Sequence[Sequence[int]],
    max_tokens: int,
) -> tuple[tuple[int, ...], ...]:
    hipengine = payload.get("hipengine")
    if not isinstance(hipengine, Mapping):
        raise ExactTokenBenchError("HTTP response omitted hipengine metadata")
    prompt_accounting = hipengine.get("prompt_token_accounting")
    if not isinstance(prompt_accounting, Mapping):
        raise ExactTokenBenchError("HTTP response omitted exact prompt-token accounting")
    expected_prompt_hashes = [token_ids_sha256(row) for row in prompt_rows]
    if prompt_accounting.get("input_type") != "token_ids":
        raise ExactTokenBenchError("HTTP response did not identify token-ID prompt input")
    if prompt_accounting.get("schema_version") != 1:
        raise ExactTokenBenchError("HTTP response prompt-token schema_version is unsupported")
    if prompt_accounting.get("prompt_token_ids_sha256") != expected_prompt_hashes:
        raise ExactTokenBenchError("HTTP response prompt-token hashes differ from request")
    expected_counts = [len(row) for row in prompt_rows]
    if prompt_accounting.get("prompt_tokens") != expected_counts:
        raise ExactTokenBenchError("HTTP response prompt-token counts differ from request")
    if prompt_accounting.get("total_prompt_tokens") != sum(expected_counts):
        raise ExactTokenBenchError("HTTP response total prompt-token count differs from request")

    token_accounting = hipengine.get("token_accounting")
    if not isinstance(token_accounting, Mapping):
        raise ExactTokenBenchError("HTTP response omitted exact generated-token accounting")
    raw_rows = token_accounting.get("choice_generated_token_ids")
    if not isinstance(raw_rows, list) or len(raw_rows) != len(prompt_rows):
        raise ExactTokenBenchError("HTTP response generated-token rows differ from request rows")
    try:
        oracle = ExactTokenOracle.from_rows(
            mode="http",
            prompt_rows=prompt_rows,
            generated_rows=raw_rows,
            max_tokens=max_tokens,
        )
    except ValueError as exc:
        raise ExactTokenBenchError(str(exc)) from exc
    usage = payload.get("usage")
    if not isinstance(usage, Mapping):
        raise ExactTokenBenchError("HTTP response omitted usage")
    if usage.get("prompt_tokens") != sum(expected_counts):
        raise ExactTokenBenchError("HTTP usage prompt_tokens differs from exact request count")
    expected_generated = len(prompt_rows) * int(max_tokens)
    if token_accounting.get("choice_generated_tokens") != [int(max_tokens)] * len(prompt_rows):
        raise ExactTokenBenchError("HTTP response generated-token counts differ from ID rows")
    if token_accounting.get("total_generated_tokens") != expected_generated:
        raise ExactTokenBenchError("HTTP response total generated-token count differs from ID rows")
    if usage.get("completion_tokens") != expected_generated:
        raise ExactTokenBenchError("HTTP usage completion_tokens differs from generated IDs")
    return oracle.generated_rows


def _post_json(url: str, payload: Mapping[str, Any], *, timeout: float) -> dict[str, Any]:
    body = json.dumps(dict(payload), separators=(",", ":")).encode("utf-8")
    req = request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=float(timeout)) as response:
            parsed = json.loads(response.read())
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise ExactTokenBenchError(f"HTTP {exc.code}: {detail}") from exc
    if not isinstance(parsed, dict):
        raise ExactTokenBenchError("HTTP response must be a JSON object")
    return parsed


def _base_artifact(
    *,
    oracle: ExactTokenOracle,
    fixture: ExactTokenFixture,
    args: argparse.Namespace,
    wall_s: float,
    output_texts: Sequence[str],
    resolved_backend: str | None,
    parity: Mapping[str, Any],
    response_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = oracle.to_json_dict()
    payload.update(
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "performance_claim": False,
            "fixture": fixture.to_json_dict(),
            "request": {
                "route": oracle.mode,
                "model": str(args.model),
                "temperature": 0.0,
                "top_p": 1.0,
                "ignore_eos": True,
            },
            "measurement": {
                "wall_s": float(wall_s),
                "timing_scope": "client_e2e" if oracle.mode == "http" else "direct_call",
                "eligible": False,
                "reason": "SOL-E5 correctness/identity gate; not a performance row",
            },
            "output_text_sha256": [text_sha256(str(text)) for text in output_texts],
            "exact_token_parity": dict(parity),
            "response_metadata": None if response_metadata is None else dict(response_metadata),
            "provenance": collect_artifact_provenance(
                repo_root=REPO_ROOT,
                configured_backend=str(args.backend),
                resolved_backend=resolved_backend,
                target_arch=args.target_arch,
                device_name=args.device_name,
                model_path=args.model_path,
                quant=args.quant,
                kv_dtype=args.kv_dtype,
                command=sys.argv,
                timing_protocol="client_e2e" if oracle.mode == "http" else "direct_call",
                warmups=0,
                repetitions=1,
                profiler={"enabled": False, "kind": None, "command": None},
            ),
        }
    )
    return payload


def run_direct(args: argparse.Namespace, fixture: ExactTokenFixture) -> dict[str, Any]:
    engine = LLM(str(args.model_path), backend=str(args.backend), quant=str(args.quant))
    sampling = SamplingParams(
        max_tokens=int(args.max_tokens),
        temperature=0.0,
        top_p=1.0,
        ignore_eos=True,
    )
    started = time.perf_counter()
    outputs = engine.generate_detailed(fixture.prompt_rows, sampling)
    wall_s = time.perf_counter() - started
    generated_rows = _generated_rows(outputs, expected_rows=fixture.prompt_count)
    oracle = ExactTokenOracle.from_rows(
        mode="direct",
        prompt_rows=fixture.prompt_rows,
        generated_rows=generated_rows,
        max_tokens=args.max_tokens,
    )
    args.quant = engine.resolved_quant
    return _base_artifact(
        oracle=oracle,
        fixture=fixture,
        args=args,
        wall_s=wall_s,
        output_texts=[str(output.text) for output in outputs],
        resolved_backend=engine.resolved_backend,
        parity={
            "passed": None,
            "status": "awaiting_http_oracle_comparison",
            "prompt_ids_preserved": True,
            "generated_ids_recorded": True,
        },
    )


def run_http(args: argparse.Namespace, fixture: ExactTokenFixture) -> dict[str, Any]:
    oracle = ExactTokenOracle.from_json_path(args.oracle)
    request_prompt: list[int] | list[list[int]]
    if fixture.prompt_count == 1:
        request_prompt = list(fixture.prompt_rows[0])
    else:
        request_prompt = [list(row) for row in fixture.prompt_rows]
    request_payload = {
        "model": str(args.model),
        "prompt": request_prompt,
        "max_tokens": int(args.max_tokens),
        "temperature": 0.0,
        "top_p": 1.0,
        "ignore_eos": True,
    }
    started = time.perf_counter()
    response = _post_json(
        str(args.url).rstrip("/") + "/v1/completions",
        request_payload,
        timeout=args.timeout,
    )
    wall_s = time.perf_counter() - started
    generated_rows = parse_http_response(
        response,
        prompt_rows=fixture.prompt_rows,
        max_tokens=args.max_tokens,
    )
    parity = validate_exact_token_parity(
        oracle,
        mode="http",
        prompt_rows=fixture.prompt_rows,
        generated_rows=generated_rows,
        max_tokens=args.max_tokens,
    )
    candidate = ExactTokenOracle.from_rows(
        mode="http",
        prompt_rows=fixture.prompt_rows,
        generated_rows=generated_rows,
        max_tokens=args.max_tokens,
    )
    choices = response.get("choices")
    output_texts = [str(choice.get("text", "")) for choice in choices] if isinstance(choices, list) else []
    return _base_artifact(
        oracle=candidate,
        fixture=fixture,
        args=args,
        wall_s=wall_s,
        output_texts=output_texts,
        resolved_backend=args.resolved_backend,
        parity=parity,
        response_metadata={
            "usage": response.get("usage"),
            "hipengine": response.get("hipengine"),
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("direct", "http"))
    parser.add_argument("--fixture", type=Path, default=DEFAULT_EXACT_TOKEN_FIXTURE)
    parser.add_argument("--prompt-length", type=int, default=512)
    parser.add_argument("--prompt-count", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model", help="HTTP served model name; defaults to --model-path")
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--resolved-backend")
    parser.add_argument("--target-arch")
    parser.add_argument("--device-name")
    parser.add_argument("--quant", default="auto")
    parser.add_argument("--kv-dtype", default="bf16")
    parser.add_argument("--url", default="http://127.0.0.1:8008")
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--oracle", type=Path)
    parser.add_argument("--json", type=Path, required=True)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    if int(args.prompt_length) <= 0 or int(args.prompt_count) <= 0:
        raise ExactTokenBenchError("prompt length/count must be positive")
    if int(args.max_tokens) <= 0:
        raise ExactTokenBenchError("--max-tokens must be positive")
    if args.mode == "http" and args.oracle is None:
        raise ExactTokenBenchError("http mode requires --oracle from direct mode")
    if args.model is None:
        args.model = str(args.model_path)
    fixture = load_exact_token_fixture(
        args.fixture,
        prompt_length=args.prompt_length,
        prompt_count=args.prompt_count,
    )
    return run_direct(args, fixture) if args.mode == "direct" else run_http(args, fixture)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = run(args)
    except (ExactTokenBenchError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
