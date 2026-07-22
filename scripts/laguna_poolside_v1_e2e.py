#!/usr/bin/env python3
"""Run live Laguna Poolside-v1 chat/reasoning/tool API gates on gfx1151."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from hipengine import LLM
from hipengine.chat.poolside_v1 import PoolsideV1ToolParser
from hipengine.core.memory import memory_stats
from hipengine.server import ServerConfig, create_app

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path("/home/lhl/models/gguf/laguna-s-2.1-Q4_K_M.gguf")
DEFAULT_TOOL_FIXTURE = ROOT / "tests/fixtures/laguna_poolside_v1_tools.json"
DEFAULT_MODEL_SHA256 = "7da520c5f44bc3c79d4eeebfd1151ba7114c5d7568e72a995638417093c5753f"
SERVED_MODEL = "laguna-poolside-v1-e2e"

_WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather for one city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "days": {"type": "integer"},
            },
            "required": ["city", "days"],
        },
    },
}
_WRITE_TOOL = {
    "type": "function",
    "function": {
        "name": "write_file",
        "description": "Write exact UTF-8 content to a file.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
                "mode": {"type": "integer"},
            },
            "required": ["path", "content", "mode"],
        },
    },
}
_UTF8_CONTENT = 'café 東京 "quoted" \\slash\nline2'


def _cases() -> tuple[dict[str, Any], ...]:
    return (
        {
            "name": "thinking_disabled_eot",
            "request": {
                "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
                "enable_thinking": False,
                "max_tokens": 16,
            },
            "expected": {
                "content": "OK",
                "reasoning": "",
                "tool_calls": [],
                "finish_reason": "stop",
            },
        },
        {
            "name": "thinking_enabled_open_length",
            "request": {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Compute 27 * 34. Reason step by step before giving only "
                            "the final number."
                        ),
                    }
                ],
                "enable_thinking": True,
                "max_tokens": 64,
            },
            "expected": {
                "content": "",
                "reasoning_nonempty": True,
                "tool_calls": [],
                "finish_reason": "length",
            },
        },
        {
            "name": "mixed_text_single_tool",
            "request": {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Say exactly Checking. and then call get_weather for Paris "
                            "for 2 days."
                        ),
                    }
                ],
                "tools": [_WEATHER_TOOL],
                "enable_thinking": True,
                "max_tokens": 64,
            },
            "expected": {
                "content": "Checking.",
                "reasoning": "",
                "tool_calls": [
                    {"name": "get_weather", "arguments": {"city": "Paris", "days": 2}}
                ],
                "finish_reason": "tool_calls",
            },
        },
        {
            "name": "multiple_tools",
            "request": {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Call get_weather separately for Paris for 2 days and Tokyo "
                            "for 3 days. Make exactly two function calls and no prose."
                        ),
                    }
                ],
                "tools": [_WEATHER_TOOL],
                "parallel_tool_calls": True,
                "enable_thinking": True,
                "max_tokens": 96,
            },
            "expected": {
                "content": "",
                "reasoning": "",
                "tool_calls": [
                    {"name": "get_weather", "arguments": {"city": "Paris", "days": 2}},
                    {"name": "get_weather", "arguments": {"city": "Tokyo", "days": 3}},
                ],
                "finish_reason": "tool_calls",
            },
        },
        {
            "name": "utf8_escaped_tool_arguments",
            "request": {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Return only a write_file call. Use path notes.txt, integer "
                            "mode 420, and content exactly equal to these two lines:\n"
                            f"{_UTF8_CONTENT}"
                        ),
                    }
                ],
                "tools": [_WRITE_TOOL],
                "enable_thinking": False,
                "max_tokens": 128,
            },
            "expected": {
                "content": "",
                "reasoning": "",
                "tool_calls": [
                    {
                        "name": "write_file",
                        "arguments": {
                            "path": "notes.txt",
                            "content": _UTF8_CONTENT,
                            "mode": 420,
                        },
                    }
                ],
                "finish_reason": "tool_calls",
            },
        },
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--backend", default="hip_gfx1151")
    parser.add_argument("--model-sha256", default=DEFAULT_MODEL_SHA256)
    parser.add_argument("--tool-fixture", type=Path, default=DEFAULT_TOOL_FIXTURE)
    parser.add_argument("--case", action="append", dest="case_names")
    parser.add_argument("--output", type=Path)
    return parser


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _git_revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _sse_payloads(text: str) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for line in text.splitlines():
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        payloads.append(json.loads(line.removeprefix("data: ")))
    return payloads


def _normalize_message(message: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(message)
    for index, call in enumerate(normalized.get("tool_calls", [])):
        call["id"] = f"call_{index}"
    return normalized


def _tool_calls_from_message(message: dict[str, Any]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for call in message.get("tool_calls", []):
        function = call["function"]
        calls.append(
            {
                "name": function["name"],
                "arguments": json.loads(function["arguments"]),
            }
        )
    return calls


def _stream_result(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    content: list[str] = []
    reasoning: list[str] = []
    tool_calls: dict[int, dict[str, Any]] = {}
    ids_by_index: dict[int, set[str]] = {}
    fragments_by_index: dict[int, int] = {}
    finish_reason: str | None = None
    finish_details: dict[str, Any] | None = None
    for payload in payloads:
        for choice in payload.get("choices", []):
            delta = choice.get("delta", {})
            if delta.get("content"):
                content.append(str(delta["content"]))
            if delta.get("reasoning_content"):
                reasoning.append(str(delta["reasoning_content"]))
            for raw_call in delta.get("tool_calls", []):
                index = int(raw_call["index"])
                state = tool_calls.setdefault(
                    index,
                    {
                        "id": raw_call.get("id", ""),
                        "type": raw_call.get("type", "function"),
                        "function": {"name": "", "arguments": ""},
                    },
                )
                call_id = str(raw_call.get("id", state["id"]))
                ids_by_index.setdefault(index, set()).add(call_id)
                function = raw_call.get("function", {})
                if function.get("name"):
                    state["function"]["name"] += str(function["name"])
                state["function"]["arguments"] += str(function.get("arguments", ""))
                fragments_by_index[index] = fragments_by_index.get(index, 0) + 1
            if choice.get("finish_reason") is not None:
                finish_reason = str(choice["finish_reason"])
                raw_details = choice.get("finish_details")
                finish_details = dict(raw_details) if isinstance(raw_details, dict) else None
    message: dict[str, Any] = {"role": "assistant", "content": "".join(content)}
    if reasoning:
        message["reasoning_content"] = "".join(reasoning)
    if tool_calls:
        message["tool_calls"] = [tool_calls[index] for index in sorted(tool_calls)]
    return {
        "message": message,
        "finish_reason": finish_reason,
        "finish_details": finish_details,
        "tool_ids_stable": all(len(ids) == 1 for ids in ids_by_index.values()),
        "tool_argument_fragments": [fragments_by_index[index] for index in sorted(tool_calls)],
    }


def _inner_generator(llm: LLM) -> Any:
    wrapper = getattr(llm, "_text_generator", None)
    return getattr(wrapper, "inner", wrapper)


def _last_generated_ids(llm: LLM) -> tuple[int, ...]:
    outputs = getattr(_inner_generator(llm), "last_generation_outputs", ())
    if len(outputs) != 1 or outputs[0].generated_token_ids is None:
        raise RuntimeError("Laguna generator did not expose one exact generated-token row")
    return tuple(int(token) for token in outputs[0].generated_token_ids)


def _expected_checks(message: dict[str, Any], finish_reason: str, expected: dict[str, Any]) -> dict[str, bool]:
    reasoning = str(message.get("reasoning_content", ""))
    checks = {
        "content": str(message.get("content", "")) == str(expected["content"]),
        "finish_reason": str(finish_reason) == str(expected["finish_reason"]),
        "tool_calls": _tool_calls_from_message(message) == expected["tool_calls"],
    }
    if expected.get("reasoning_nonempty"):
        checks["reasoning"] = bool(reasoning)
    else:
        checks["reasoning"] = reasoning == str(expected.get("reasoning", ""))
    return checks


def _run_case(client: TestClient, llm: LLM, case: dict[str, Any]) -> dict[str, Any]:
    target = _inner_generator(llm)
    request = {"model": SERVED_MODEL, **copy.deepcopy(case["request"])}
    rendered = target.render_chat_prompt(
        request["messages"],
        tools=request.get("tools"),
        enable_thinking=bool(request.get("enable_thinking", False)),
        add_generation_prompt=True,
    )
    prompt_ids = tuple(int(token) for token in llm.tokenize(rendered))

    blocking_started = time.perf_counter()
    blocking_response = client.post("/v1/chat/completions", json=request)
    blocking_seconds = time.perf_counter() - blocking_started
    blocking_payload = blocking_response.json()
    blocking_choice = blocking_payload.get("choices", [{}])[0]
    blocking_ids = _last_generated_ids(llm) if blocking_response.status_code == 200 else ()

    stream_request = {**request, "stream": True, "stream_options": {"include_hipengine": True}}
    stream_started = time.perf_counter()
    stream_response = client.post("/v1/chat/completions", json=stream_request)
    stream_seconds = time.perf_counter() - stream_started
    payloads = _sse_payloads(stream_response.text)
    streamed = _stream_result(payloads)
    stream_ids = _last_generated_ids(llm) if stream_response.status_code == 200 else ()

    blocking_message = dict(blocking_choice.get("message", {}))
    blocking_finish = str(blocking_choice.get("finish_reason"))
    blocking_checks = _expected_checks(blocking_message, blocking_finish, case["expected"])
    stream_checks = _expected_checks(
        streamed["message"],
        str(streamed["finish_reason"]),
        case["expected"],
    )
    public_serialized = json.dumps(
        {"blocking": blocking_message, "stream": streamed["message"]},
        ensure_ascii=False,
    )
    checks = {
        "blocking_http_200": blocking_response.status_code == 200,
        "stream_http_200": stream_response.status_code == 200,
        "blocking_expected": all(blocking_checks.values()),
        "stream_expected": all(stream_checks.values()),
        "blocking_stream_message_equal": _normalize_message(blocking_message)
        == _normalize_message(streamed["message"]),
        "blocking_stream_ids_equal": blocking_ids == stream_ids,
        "stream_tool_ids_stable": bool(streamed["tool_ids_stable"]),
        "control_markup_absent": all(
            marker not in public_serialized
            for marker in (
                "<think>",
                "</think>",
                "<tool_call>",
                "</tool_call>",
                "<arg_key>",
                "</arg_key>",
                "<arg_value>",
                "</arg_value>",
                "</assistant>",
            )
        ),
    }
    return {
        "name": case["name"],
        "request": case["request"],
        "rendered_prompt": rendered,
        "rendered_prompt_sha256": _sha256_text(rendered),
        "prompt_token_ids": list(prompt_ids),
        "blocking": {
            "seconds": blocking_seconds,
            "generated_token_ids": list(blocking_ids),
            "raw_decoded_output": llm.detokenize(blocking_ids, skip_special=False),
            "message": blocking_message,
            "finish_reason": blocking_choice.get("finish_reason"),
            "finish_details": blocking_choice.get("finish_details"),
            "checks": blocking_checks,
        },
        "stream": {
            "seconds": stream_seconds,
            "generated_token_ids": list(stream_ids),
            "raw_decoded_output": llm.detokenize(stream_ids, skip_special=False),
            **streamed,
            "checks": stream_checks,
            "sse_payload_count": len(payloads),
        },
        "checks": checks,
        "pass": all(checks.values()),
    }


def _validate_tool_fixture(path: Path) -> dict[str, Any]:
    fixture = json.loads(path.read_text(encoding="utf-8"))
    parser = PoolsideV1ToolParser()
    cases: list[dict[str, Any]] = []
    for case in fixture["cases"]:
        parsed = parser.parse(case["output"], tools=fixture["tools"])
        observed_calls = [
            {"name": call.name, "arguments": json.loads(call.arguments)}
            for call in parsed.tool_calls
        ]
        passed = (
            parsed.content == case["content"]
            and observed_calls == case["calls"]
            and bool(parsed.invalid_blocks) is bool(case["invalid"])
        )
        cases.append({"name": case["name"], "pass": passed})
    return {
        "fixture": str(path.resolve()),
        "cases": cases,
        "pass": all(case["pass"] for case in cases),
    }


def run_gate(
    model: Path,
    *,
    backend: str,
    model_sha256: str,
    tool_fixture: Path,
    case_names: set[str] | None,
) -> dict[str, Any]:
    selected = tuple(
        case for case in _cases() if case_names is None or case["name"] in case_names
    )
    if case_names is not None:
        missing = case_names - {case["name"] for case in selected}
        if missing:
            raise ValueError(f"unknown --case values: {', '.join(sorted(missing))}")
    tracked_before = memory_stats()
    llm = LLM(str(model), backend=backend)
    client = TestClient(
        create_app(
            ServerConfig(model=str(model), served_model_name=SERVED_MODEL),
            llm=llm,
        )
    )
    # Resolve the lazy public generator before querying model-owned chat
    # capabilities or rendering the first prompt. This does not load weights.
    llm.count_tokens("")
    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    capabilities: dict[str, Any] = {}
    try:
        capabilities_response = client.get("/v1/hipengine/capabilities")
        capabilities = capabilities_response.json()
        for case in selected:
            result = _run_case(client, llm, case)
            results.append(result)
            print(
                f"case={case['name']} pass={result['pass']} "
                f"blocking_s={result['blocking']['seconds']:.3f} "
                f"stream_s={result['stream']['seconds']:.3f}",
                file=sys.stderr,
                flush=True,
            )
    finally:
        llm.close()
    tracked_after = memory_stats()
    tool_fixture_result = _validate_tool_fixture(tool_fixture)
    parser_capability = capabilities.get("features", {}).get("tools", {})
    capability_checks = {
        "chat_family": capabilities.get("chat_template", {}).get("family")
        == "poolside_v1",
        "reasoning_parser": capabilities.get("chat_template", {}).get(
            "reasoning_parser"
        )
        == "poolside_v1",
        "tool_parser": parser_capability.get("parser") == "poolside_v1",
        "tool_format": parser_capability.get("format") == "poolside_v1_xml",
    }
    lifecycle_checks = {
        "tracked_bytes": tracked_after["current_allocated_bytes"]
        == tracked_before["current_allocated_bytes"],
        "tracked_allocations": tracked_after["active_allocations"]
        == tracked_before["active_allocations"],
    }
    passed = (
        bool(results)
        and all(result["pass"] for result in results)
        and tool_fixture_result["pass"]
        and all(capability_checks.values())
        and all(lifecycle_checks.values())
    )
    return {
        "schema": 1,
        "status": "accepted" if passed else "failed",
        "performance_claim": False,
        "scope": "Laguna Poolside-v1 live parser/API correctness",
        "date": "2026-07-22",
        "source_revision": _git_revision(),
        "command": [sys.executable, *sys.argv],
        "model": {
            "path": str(model.resolve()),
            "sha256": model_sha256,
            "backend": backend,
            "quant": "gguf_q4_k_m",
        },
        "platform": {
            "HIPENGINE_HIP_ARCH": "gfx1151",
            "GPU_MAX_HW_QUEUES": "1",
        },
        "cases": results,
        "deterministic_tool_fixture": tool_fixture_result,
        "capability_checks": capability_checks,
        "lifecycle": {
            "before": tracked_before,
            "after": tracked_after,
            "checks": lifecycle_checks,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "coverage": {
            "live": [
                "thinking disabled and EOT",
                "thinking enabled and prompt-open reasoning",
                "mixed text plus call",
                "multiple calls",
                "UTF-8, quote, backslash, and newline arguments",
                "blocking and fragmented model-token streaming",
            ],
            "deterministic_fixture": [
                "newline-less call",
                "typed values and verbatim whitespace",
                "ordinary content",
                "partial call",
                "malformed call",
                "empty name",
            ],
            "external_tests": [
                "tests/test_poolside_v1_reasoning.py",
                "tests/test_poolside_v1_tools.py",
                "tests/test_server_api.py -k 'tool or reasoning or thinking or capabilit'",
                "tests/test_agentic_server_conformance.py",
            ],
        },
        "pass": passed,
    }


def main() -> int:
    args = _parser().parse_args()
    result = run_gate(
        args.model,
        backend=str(args.backend),
        model_sha256=str(args.model_sha256),
        tool_fixture=args.tool_fixture,
        case_names=None if args.case_names is None else set(args.case_names),
    )
    payload = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    print(payload)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
