#!/usr/bin/env python3
"""Validate a local-agent config against a running hipEngine server."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class ConfigValidationError(ValueError):
    """Raised when a local-agent config does not match server capabilities."""


_EXPECTED_CHAT_SMOKE_RESULT = "ok"


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ConfigValidationError("config root must be a JSON object")
    return payload


def validate_config_against_capabilities(
    config: dict[str, Any], capabilities: dict[str, Any]
) -> dict[str, Any]:
    """Validate config fields against a capabilities manifest.

    The config format is intentionally adapter-neutral: only
    ``chat_completions.defaults`` is treated as fields that will be sent to the
    OpenAI-compatible API on every request.
    """

    if config.get("schema") != "hipengine.local_agent.v1":
        raise ConfigValidationError("schema must be hipengine.local_agent.v1")
    chat = _object(config, "chat_completions")
    defaults = _object(chat, "defaults")
    unsupported = {str(item) for item in capabilities.get("unsupported_fields", ())}
    default_fields = set(defaults)
    blocked_defaults = sorted(default_fields & unsupported)
    if blocked_defaults:
        raise ConfigValidationError(
            "chat_completions.defaults includes unsupported fields: "
            + ", ".join(blocked_defaults)
        )
    do_not_send = set(_string_list(chat.get("do_not_send", ()), "chat_completions.do_not_send"))
    missing_blocklist = sorted(unsupported - do_not_send)
    if missing_blocklist:
        raise ConfigValidationError(
            "chat_completions.do_not_send must list unsupported fields: "
            + ", ".join(missing_blocklist)
        )

    features = _object(capabilities, "features")
    if not features.get("chat_completions"):
        raise ConfigValidationError("server does not advertise chat_completions")
    if defaults.get("stream") and not features.get("streaming"):
        raise ConfigValidationError(
            "config enables streaming but server does not advertise streaming"
        )
    stream_options = defaults.get("stream_options")
    if stream_options is not None:
        supported_stream_options = _object(features, "stream_options")
        requested_stream_options = _object_value(
            stream_options, "chat_completions.defaults.stream_options"
        )
        for name, enabled in requested_stream_options.items():
            if bool(enabled) and not supported_stream_options.get(name):
                raise ConfigValidationError(f"stream option {name!r} is not advertised")
    if "timeout_ms" in defaults:
        timeouts = _object(features, "request_timeouts")
        if not timeouts.get("timeout_ms"):
            raise ConfigValidationError(
                "config uses timeout_ms but server does not advertise request timeouts"
            )
    if "reasoning_effort" in defaults:
        reasoning = _object(features, "reasoning_controls")
        if not reasoning.get("enabled") or "reasoning_effort" not in reasoning.get("fields", ()):
            raise ConfigValidationError(
                "config uses reasoning_effort but server does not advertise it"
            )
    tool_calling = _object(chat, "tool_calling")
    if tool_calling.get("enabled"):
        tools = _object(features, "tools")
        if not tools.get("enabled"):
            raise ConfigValidationError("config enables tools but server does not advertise tools")
        if tool_calling.get("strict_decoding_required") and not tools.get("strict_decoding"):
            raise ConfigValidationError(
                "config requires strict tool decoding but server advertises prompt/parse tools only"
            )

    token_diagnostics = _object(config, "token_diagnostics")
    advertised_diag = _object(features, "token_diagnostics")
    if token_diagnostics.get("count_before_send") and not advertised_diag.get("count_tokens"):
        raise ConfigValidationError("config requires count_tokens but server does not advertise it")
    if token_diagnostics.get("fit_context_before_large_requests") and not advertised_diag.get(
        "fit_context"
    ):
        raise ConfigValidationError("config requires fit_context but server does not advertise it")

    return {
        "model": _model_id(config, capabilities),
        "default_fields": sorted(default_fields),
        "blocked_fields": sorted(do_not_send),
        "streaming": bool(defaults.get("stream")),
        "tools": bool(tool_calling.get("enabled")),
    }


def build_chat_smoke_payload(
    config: dict[str, Any], capabilities: dict[str, Any]
) -> dict[str, Any]:
    """Build a small chat request that exercises the documented config shape."""

    chat = _object(config, "chat_completions")
    defaults = dict(_object(chat, "defaults"))
    defaults["stream"] = False
    defaults["max_tokens"] = min(int(defaults.get("max_tokens", 8)), 8)
    payload: dict[str, Any] = {
        "model": _model_id(config, capabilities),
        "messages": [
            {
                "role": "user",
                "content": f"Use the provided tool to record {_EXPECTED_CHAT_SMOKE_RESULT}.",
            }
        ],
        **defaults,
    }
    tool_calling = _object(chat, "tool_calling")
    if tool_calling.get("enabled"):
        payload["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": "record_result",
                    "description": "Record a short result string.",
                    "parameters": {
                        "type": "object",
                        "properties": {"result": {"type": "string"}},
                        "required": ["result"],
                        "additionalProperties": False,
                    },
                },
            }
        ]
        payload["tool_choice"] = {"type": "function", "function": {"name": "record_result"}}
    for blocked in _string_list(chat.get("do_not_send", ()), "chat_completions.do_not_send"):
        payload.pop(blocked, None)
    return payload


def fetch_capabilities(
    base_url: str,
    *,
    path: str = "/hipengine/capabilities",
    api_key: str | None = None,
    timeout: float = 10.0,
) -> dict[str, Any]:
    return _request_json("GET", _join_url(base_url, path), api_key=api_key, timeout=timeout)


def run_chat_smoke(
    base_url: str,
    config: dict[str, Any],
    capabilities: dict[str, Any],
    *,
    api_key: str | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    payload = build_chat_smoke_payload(config, capabilities)
    tool_calling = _object(_object(config, "chat_completions"), "tool_calling")
    response = _request_json(
        "POST",
        _join_url(
            base_url,
            _object(config, "chat_completions").get("endpoint", "/chat/completions"),
        ),
        api_key=api_key,
        payload=payload,
        timeout=timeout,
    )
    validate_chat_smoke_response(response, expect_tool_call=bool(tool_calling.get("enabled")))
    return response


def validate_chat_smoke_response(
    response: dict[str, Any], *, expect_tool_call: bool = False
) -> dict[str, Any]:
    choices = _list(response.get("choices"), "chat smoke response.choices")
    if not choices:
        raise ConfigValidationError("chat smoke response.choices must contain at least one choice")
    choice = _object_value(choices[0], "chat smoke response.choices[0]")
    message = _object(choice, "message")
    finish_reason = choice.get("finish_reason")
    if message.get("role") != "assistant":
        raise ConfigValidationError("chat smoke response message.role must be 'assistant'")
    raw_tool_field = _first_message_field_containing(message, "<tool_call>")
    if raw_tool_field is not None:
        raise ConfigValidationError(
            f"chat smoke returned raw <tool_call> text in message.{raw_tool_field} "
            "instead of parsed message.tool_calls; "
            "check that the client is using /v1/chat/completions with tools enabled"
        )
    if not expect_tool_call:
        return {"finish_reason": None if finish_reason is None else str(finish_reason)}
    if finish_reason != "tool_calls":
        raise ConfigValidationError(
            "chat smoke did not finish with tool_calls; "
            f"finish_reason={finish_reason!r}"
        )
    content = message.get("content")
    if content not in (None, ""):
        raise ConfigValidationError(
            "chat smoke tool-call response must not include assistant content; "
            "expected parsed message.tool_calls only"
        )
    tool_calls = _list(message.get("tool_calls"), "chat smoke response message.tool_calls")
    if len(tool_calls) != 1:
        raise ConfigValidationError(
            f"chat smoke expected exactly one tool call, got {len(tool_calls)}"
        )
    call = _openai_function_tool_call(
        tool_calls[0],
        "chat smoke response message.tool_calls[0]",
    )
    function = call["function"]
    name = function.get("name")
    if name != "record_result":
        raise ConfigValidationError(f"chat smoke selected unexpected tool {name!r}")
    arguments = function.get("arguments")
    if not isinstance(arguments, str):
        raise ConfigValidationError("chat smoke tool arguments must be a JSON string")
    try:
        decoded_args = json.loads(arguments)
    except json.JSONDecodeError as exc:
        raise ConfigValidationError(f"chat smoke tool arguments are not valid JSON: {exc}") from exc
    if not isinstance(decoded_args, dict):
        raise ConfigValidationError("chat smoke tool arguments must decode to a JSON object")
    result = decoded_args.get("result")
    if not isinstance(result, str):
        raise ConfigValidationError("chat smoke tool arguments must include string field 'result'")
    if result != _EXPECTED_CHAT_SMOKE_RESULT:
        raise ConfigValidationError(
            "chat smoke tool arguments must set "
            f"'result' to {_EXPECTED_CHAT_SMOKE_RESULT!r}, got {result!r}"
        )
    return {
        "finish_reason": "tool_calls",
        "tool_name": "record_result",
        "argument_keys": sorted(str(key) for key in decoded_args),
        "result": result,
    }


def _openai_function_tool_call(raw: Any, label: str) -> dict[str, Any]:
    call = _object_value(raw, label)
    _require_exact_keys(call, {"id", "type", "function"}, label)
    call_id = call.get("id")
    if not isinstance(call_id, str) or not call_id.strip():
        raise ConfigValidationError(f"{label}.id must be a non-empty string")
    if call.get("type") != "function":
        raise ConfigValidationError(f"{label}.type must be 'function'")
    function = _object_value(call.get("function"), f"{label}.function")
    _require_exact_keys(function, {"name", "arguments"}, f"{label}.function")
    name = function.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ConfigValidationError(f"{label}.function.name must be a non-empty string")
    arguments = function.get("arguments")
    if not isinstance(arguments, str):
        raise ConfigValidationError(f"{label}.function.arguments must be a JSON string")
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


def _require_exact_keys(mapping: dict[str, Any], expected: set[str], label: str) -> None:
    actual = set(mapping)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ConfigValidationError(
            f"{label} must have exactly keys {sorted(expected)}; "
            f"missing={missing}, extra={extra}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="docs/examples/local-agent/openai-compatible.json")
    parser.add_argument(
        "--base-url",
        help="Override config server.base_url, for example http://127.0.0.1:8000/v1",
    )
    parser.add_argument(
        "--api-key",
        help="Bearer token. Defaults to config server.api_key_env when set.",
    )
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument(
        "--chat-smoke",
        action="store_true",
        help="Also POST a small chat/tools request.",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    server = _object(config, "server")
    base_url = str(args.base_url or server.get("base_url") or "http://127.0.0.1:8000/v1")
    api_key = args.api_key
    if api_key is None:
        api_key_env = server.get("api_key_env")
        api_key = os.environ.get(str(api_key_env)) if api_key_env else None
    capabilities = fetch_capabilities(
        base_url,
        path=str(server.get("capabilities_path") or "/hipengine/capabilities"),
        api_key=api_key,
        timeout=args.timeout,
    )
    summary = validate_config_against_capabilities(config, capabilities)
    if args.chat_smoke:
        response = run_chat_smoke(
            base_url,
            config,
            capabilities,
            api_key=api_key,
            timeout=max(args.timeout, 30.0),
        )
        summary["chat_smoke_object"] = response.get("object")
        tool_calling = _object(_object(config, "chat_completions"), "tool_calling")
        summary["chat_smoke"] = validate_chat_smoke_response(
            response,
            expect_tool_call=bool(tool_calling.get("enabled")),
        )
    print(json.dumps({"ok": True, **summary}, indent=2, sort_keys=True))
    return 0


def _request_json(
    method: str,
    url: str,
    *,
    api_key: str | None = None,
    payload: dict[str, Any] | None = None,
    timeout: float,
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Accept": "application/json"}
    if payload is not None:
        headers["Content-Type"] = "application/json"
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = Request(url, data=data, headers=headers, method=method)
    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read()
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise ConfigValidationError(
            f"{method} {url} failed with HTTP {exc.code}: {detail}"
        ) from exc
    except URLError as exc:
        raise ConfigValidationError(f"{method} {url} failed: {exc}") from exc
    decoded = json.loads(body.decode("utf-8"))
    if not isinstance(decoded, dict):
        raise ConfigValidationError(f"{method} {url} did not return a JSON object")
    return decoded


def _join_url(base_url: str, path: Any) -> str:
    return str(base_url).rstrip("/") + "/" + str(path).lstrip("/")


def _model_id(config: dict[str, Any], capabilities: dict[str, Any]) -> str:
    model = _object(config, "model")
    if model.get("source") == "capabilities.model.id":
        return str(_object(capabilities, "model").get("id"))
    if model.get("id"):
        return str(model["id"])
    raise ConfigValidationError(
        "model.source must be capabilities.model.id or model.id must be set"
    )


def _object(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    return _object_value(value, key)


def _object_value(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigValidationError(f"{label} must be a JSON object")
    return value


def _first_message_field_containing(message: dict[str, Any], marker: str) -> str | None:
    for field in ("content", "reasoning_content"):
        value = message.get(field)
        if isinstance(value, str) and marker in value:
            return field
    return None


def _string_list(value: Any, label: str) -> list[str]:
    if not isinstance(value, (list, tuple)):
        raise ConfigValidationError(f"{label} must be a JSON array")
    return [str(item) for item in value]


def _list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ConfigValidationError(f"{label} must be a JSON array")
    return list(value)


if __name__ == "__main__":
    raise SystemExit(main())
