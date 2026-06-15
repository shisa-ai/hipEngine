#!/usr/bin/env python3
"""Validate a pi-agent models.json for the hipEngine Qwen endpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class PiConfigValidationError(ValueError):
    """Raised when a pi-agent models.json does not match hipEngine guidance."""


_EXPECTED_CHAT_SMOKE_RESULT = "ok"
_EXPECTED_REASONING_SMOKE_ANSWER = "OK"


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise PiConfigValidationError("config root must be a JSON object")
    return payload


def validate_pi_models_config(
    config: dict[str, Any], *, provider_name: str | None = None
) -> dict[str, Any]:
    providers = _object(config, "providers")
    if not providers:
        raise PiConfigValidationError("providers must contain at least one provider")

    selected: list[tuple[str, Any]]
    if provider_name is None:
        selected = list(providers.items())
    else:
        if provider_name not in providers:
            raise PiConfigValidationError(f"provider {provider_name!r} is not present")
        selected = [(provider_name, providers[provider_name])]

    summaries: list[dict[str, Any]] = []
    errors: list[str] = []
    for provider_id, raw_provider in selected:
        label = f"providers.{provider_id}"
        provider_error_start = len(errors)
        try:
            provider = _object_value(raw_provider, label)
        except PiConfigValidationError as exc:
            errors.append(str(exc))
            continue
        try:
            _validate_provider(provider, label)
        except PiConfigValidationError as exc:
            errors.append(str(exc))
        try:
            models = _list(provider.get("models"), f"{label}.models")
        except PiConfigValidationError as exc:
            errors.append(str(exc))
            continue
        if not models:
            errors.append(f"{label}.models must contain at least one model")
            continue
        model_summaries = []
        for index, raw_model in enumerate(models):
            model_label = f"{label}.models[{index}]"
            try:
                model = _object_value(raw_model, model_label)
            except PiConfigValidationError as exc:
                errors.append(str(exc))
                continue
            try:
                _validate_model(model, model_label)
            except PiConfigValidationError as exc:
                errors.append(str(exc))
                continue
            model_summaries.append(
                {
                    "id": str(model["id"]),
                    "reasoning": bool(model["reasoning"]),
                    "contextWindow": int(model["contextWindow"]),
                    "maxTokens": int(model["maxTokens"]),
                }
            )
        if len(errors) == provider_error_start:
            summaries.append(
                {
                    "provider": provider_id,
                    "baseUrl": str(provider["baseUrl"]),
                    "models": model_summaries,
                }
            )

    _raise_config_errors(errors)

    return {
        "providers": summaries,
        "provider_count": len(summaries),
        "model_count": sum(len(item["models"]) for item in summaries),
    }


def validate_pi_models_against_capabilities(
    config: dict[str, Any],
    capabilities: dict[str, Any],
    *,
    provider_name: str | None = None,
) -> dict[str, Any]:
    summary = validate_pi_models_config(config, provider_name=provider_name)
    selected_models = {
        model["id"]
        for provider in summary["providers"]
        for model in provider["models"]
    }
    capability_model_id = str(_object(capabilities, "model").get("id") or "")
    if capability_model_id not in selected_models:
        raise PiConfigValidationError(
            f"served model {capability_model_id!r} is not listed in the pi config"
        )

    matching_model = next(
        model
        for provider in summary["providers"]
        for model in provider["models"]
        if model["id"] == capability_model_id
    )
    context = _object(capabilities, "context")
    effective_context = context.get("effective_max_context_tokens")
    if effective_context is not None and int(matching_model["contextWindow"]) > int(effective_context):
        raise PiConfigValidationError(
            "model.contextWindow exceeds server effective_max_context_tokens"
        )

    features = _object(capabilities, "features")
    if not features.get("chat_completions"):
        raise PiConfigValidationError("server does not advertise chat_completions")
    stream_options = _object(features, "stream_options")
    if not features.get("streaming") or not stream_options.get("include_usage"):
        raise PiConfigValidationError("server does not advertise streaming usage support")
    tools = _object(features, "tools")
    if not tools.get("enabled"):
        raise PiConfigValidationError("server does not advertise tool support")
    reasoning = _object(features, "reasoning_controls")
    if not reasoning.get("enabled") or "enable_thinking" not in reasoning.get("fields", ()):
        raise PiConfigValidationError("server does not advertise Qwen enable_thinking support")

    return {
        **summary,
        "capability_model": capability_model_id,
        "effective_context": effective_context,
        "tools": True,
        "streaming_usage": True,
        "qwen_thinking": True,
    }


def build_pi_chat_smoke_payload(config: dict[str, Any], capabilities: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": str(_object(capabilities, "model").get("id")),
        "messages": [
            {
                "role": "user",
                "content": f"Use the provided tool to record {_EXPECTED_CHAT_SMOKE_RESULT}.",
            }
        ],
        "temperature": 0,
        "max_tokens": 64,
        "enable_thinking": False,
        "tools": [
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
        ],
        "tool_choice": {"type": "function", "function": {"name": "record_result"}},
        "session": {"commit": "append_none"},
    }


def build_pi_streaming_chat_smoke_payload(
    config: dict[str, Any], capabilities: dict[str, Any]
) -> dict[str, Any]:
    payload = build_pi_chat_smoke_payload(config, capabilities)
    payload["stream"] = True
    payload["stream_options"] = {"include_usage": True}
    return payload


def build_pi_reasoning_smoke_payload(
    config: dict[str, Any], capabilities: dict[str, Any]
) -> dict[str, Any]:
    return {
        "model": str(_object(capabilities, "model").get("id")),
        "messages": [
            {
                "role": "user",
                "content": (
                    "Think briefly, then answer with exactly "
                    f"{_EXPECTED_REASONING_SMOKE_ANSWER}."
                ),
            }
        ],
        "temperature": 0,
        "max_tokens": 96,
        "enable_thinking": True,
        "session": {"commit": "append_none"},
    }


def fetch_capabilities(base_url: str, *, api_key: str | None = None, timeout: float = 10.0) -> dict[str, Any]:
    return _request_json(
        "GET",
        _join_url(base_url, "/hipengine/capabilities"),
        api_key=api_key,
        timeout=timeout,
    )


def run_pi_chat_smoke(
    base_url: str,
    config: dict[str, Any],
    capabilities: dict[str, Any],
    *,
    api_key: str | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    response = _request_json(
        "POST",
        _join_url(base_url, "/chat/completions"),
        api_key=api_key,
        payload=build_pi_chat_smoke_payload(config, capabilities),
        timeout=timeout,
    )
    validate_pi_chat_smoke_response(response)
    return response


def run_pi_streaming_chat_smoke(
    base_url: str,
    config: dict[str, Any],
    capabilities: dict[str, Any],
    *,
    api_key: str | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    response_text = _request_text(
        "POST",
        _join_url(base_url, "/chat/completions"),
        api_key=api_key,
        payload=build_pi_streaming_chat_smoke_payload(config, capabilities),
        timeout=timeout,
        accept="text/event-stream",
    )
    return validate_pi_streaming_chat_smoke_response(response_text)


def run_pi_reasoning_smoke(
    base_url: str,
    config: dict[str, Any],
    capabilities: dict[str, Any],
    *,
    api_key: str | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    response = _request_json(
        "POST",
        _join_url(base_url, "/chat/completions"),
        api_key=api_key,
        payload=build_pi_reasoning_smoke_payload(config, capabilities),
        timeout=timeout,
    )
    validate_pi_reasoning_smoke_response(response)
    return response


def validate_pi_chat_smoke_response(response: dict[str, Any]) -> dict[str, Any]:
    choices = _list(response.get("choices"), "chat smoke response.choices")
    if not choices:
        raise PiConfigValidationError("chat smoke response.choices must contain at least one choice")
    choice = _object_value(choices[0], "chat smoke response.choices[0]")
    message = _object(choice, "message", label="chat smoke response.choices[0].message")
    if message.get("role") != "assistant":
        raise PiConfigValidationError("chat smoke response message.role must be 'assistant'")
    raw_tool_field = _first_message_field_containing(message, "<tool_call>")
    if raw_tool_field is not None:
        raise PiConfigValidationError(
            f"chat smoke returned raw <tool_call> text in message.{raw_tool_field} "
            "instead of parsed message.tool_calls; "
            "check that pi is using the OpenAI chat-completions adapter with tools enabled"
        )
    if choice.get("finish_reason") != "tool_calls":
        raise PiConfigValidationError(
            "chat smoke did not finish with tool_calls; "
            f"finish_reason={choice.get('finish_reason')!r}"
        )
    content = message.get("content")
    if content not in (None, ""):
        raise PiConfigValidationError(
            "chat smoke tool-call response must not include assistant content; "
            "expected parsed message.tool_calls only"
        )
    tool_calls = _list(message.get("tool_calls"), "chat smoke response message.tool_calls")
    if len(tool_calls) != 1:
        raise PiConfigValidationError(
            f"chat smoke expected exactly one tool call, got {len(tool_calls)}"
        )
    call = _openai_function_tool_call(
        tool_calls[0],
        "chat smoke response message.tool_calls[0]",
    )
    function = call["function"]
    name = function.get("name")
    if name != "record_result":
        raise PiConfigValidationError(f"chat smoke selected unexpected tool {name!r}")
    arguments = function.get("arguments")
    if not isinstance(arguments, str):
        raise PiConfigValidationError("chat smoke tool arguments must be a JSON string")
    try:
        decoded_args = json.loads(arguments)
    except json.JSONDecodeError as exc:
        raise PiConfigValidationError(f"chat smoke tool arguments are not valid JSON: {exc}") from exc
    if not isinstance(decoded_args, dict):
        raise PiConfigValidationError("chat smoke tool arguments must decode to a JSON object")
    result = decoded_args.get("result")
    if not isinstance(result, str):
        raise PiConfigValidationError("chat smoke tool arguments must include string field 'result'")
    if result != _EXPECTED_CHAT_SMOKE_RESULT:
        raise PiConfigValidationError(
            "chat smoke tool arguments must set "
            f"'result' to {_EXPECTED_CHAT_SMOKE_RESULT!r}, got {result!r}"
        )
    return {
        "finish_reason": "tool_calls",
        "tool_name": "record_result",
        "argument_keys": sorted(str(key) for key in decoded_args),
        "result": result,
    }


def validate_pi_streaming_chat_smoke_response(response_text: str) -> dict[str, Any]:
    payloads, done_seen = _parse_sse_payloads(response_text)
    if not done_seen:
        raise PiConfigValidationError("streaming chat smoke did not end with data: [DONE]")
    return validate_pi_streaming_chat_smoke_payloads(payloads)


def validate_pi_streaming_chat_smoke_payloads(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    if not payloads:
        raise PiConfigValidationError("streaming chat smoke response did not include SSE payloads")

    usage_seen = False
    finish_reason: str | None = None
    calls: dict[int, dict[str, Any]] = {}
    for payload_index, payload in enumerate(payloads):
        choices = payload.get("choices")
        if choices == [] and isinstance(payload.get("usage"), dict):
            usage_seen = True
            continue
        if not isinstance(choices, list):
            raise PiConfigValidationError(
                f"streaming chat smoke SSE payload {payload_index} choices must be an array"
            )
        for choice_index, raw_choice in enumerate(choices):
            choice = _object_value(
                raw_choice,
                f"streaming chat smoke SSE payload {payload_index} choices[{choice_index}]",
            )
            choice_finish = choice.get("finish_reason")
            if choice_finish is not None:
                finish_reason = str(choice_finish)
            delta = choice.get("delta")
            if delta is None:
                continue
            delta_obj = _object_value(
                delta,
                f"streaming chat smoke SSE payload {payload_index} choices[{choice_index}].delta",
            )
            raw_tool_field = _first_message_field_containing(delta_obj, "<tool_call>")
            if raw_tool_field is not None:
                raise PiConfigValidationError(
                    f"streaming chat smoke returned raw <tool_call> text in delta.{raw_tool_field} "
                    "instead of parsed delta.tool_calls"
                )
            tool_call_deltas = delta_obj.get("tool_calls")
            if tool_call_deltas is None:
                continue
            for raw_call in _list(tool_call_deltas, "streaming chat smoke delta.tool_calls"):
                call_delta = _object_value(raw_call, "streaming chat smoke delta.tool_calls[]")
                _require_subset_keys(
                    call_delta,
                    {"index", "id", "type", "function"},
                    "streaming chat smoke delta.tool_calls[]",
                )
                raw_index = call_delta.get("index", 0)
                if isinstance(raw_index, bool) or not isinstance(raw_index, int) or raw_index < 0:
                    raise PiConfigValidationError("streaming chat smoke tool_call index must be a non-negative integer")
                first_delta = int(raw_index) not in calls
                if first_delta:
                    _validate_stream_tool_call_start(
                        call_delta,
                        "streaming chat smoke delta.tool_calls[]",
                    )
                call = calls.setdefault(
                    int(raw_index),
                    {
                        "id": call_delta.get("id"),
                        "type": call_delta.get("type"),
                        "function": {"name": "", "arguments": ""},
                    },
                )
                if call_delta.get("id") is not None:
                    call["id"] = call_delta["id"]
                if call_delta.get("type") is not None:
                    call["type"] = call_delta["type"]
                function_delta = call_delta.get("function")
                if function_delta is not None:
                    function = _object_value(function_delta, "streaming chat smoke tool_call.function")
                    _require_subset_keys(
                        function,
                        {"name", "arguments"},
                        "streaming chat smoke tool_call.function",
                    )
                    if function.get("name") is not None:
                        name = function["name"]
                        if not isinstance(name, str) or not name.strip():
                            raise PiConfigValidationError(
                                "streaming chat smoke tool_call.function.name must be a non-empty string"
                            )
                        call["function"]["name"] = name
                    if function.get("arguments") is not None:
                        arguments = function["arguments"]
                        if not isinstance(arguments, str):
                            raise PiConfigValidationError(
                                "streaming chat smoke tool_call.function.arguments must be a string"
                            )
                        call["function"]["arguments"] += arguments

    if not usage_seen:
        raise PiConfigValidationError("streaming chat smoke did not include a usage SSE payload")
    if not calls:
        raise PiConfigValidationError("streaming chat smoke did not include parsed delta.tool_calls")
    reconstructed = {
        "object": "chat.completion",
        "choices": [
            {
                "finish_reason": finish_reason,
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [calls[index] for index in sorted(calls)],
                },
            }
        ],
    }
    summary = validate_pi_chat_smoke_response(reconstructed)
    return {
        **summary,
        "sse_payloads": len(payloads),
        "usage_chunk": True,
        "done": True,
    }


def validate_pi_reasoning_smoke_response(response: dict[str, Any]) -> dict[str, Any]:
    choices = _list(response.get("choices"), "reasoning smoke response.choices")
    if not choices:
        raise PiConfigValidationError("reasoning smoke response.choices must contain at least one choice")
    choice = _object_value(choices[0], "reasoning smoke response.choices[0]")
    finish_reason = choice.get("finish_reason")
    if finish_reason not in (None, "stop"):
        raise PiConfigValidationError(
            "reasoning smoke did not finish cleanly; "
            f"finish_reason={finish_reason!r}"
        )
    message = _object(choice, "message", label="reasoning smoke response.choices[0].message")
    content = message.get("content")
    reasoning = message.get("reasoning_content")
    if isinstance(content, str) and ("<think>" in content or "</think>" in content):
        raise PiConfigValidationError(
            "reasoning smoke returned raw <think> text instead of parsed message.reasoning_content; "
            "check that pi is using the OpenAI chat-completions adapter with Qwen thinking enabled"
        )
    if isinstance(reasoning, str) and ("<think>" in reasoning or "</think>" in reasoning):
        raise PiConfigValidationError("reasoning smoke reasoning_content still contains Qwen think tags")
    if not isinstance(reasoning, str) or not reasoning.strip():
        raise PiConfigValidationError(
            "reasoning smoke response did not include non-empty message.reasoning_content"
        )
    if not isinstance(content, str) or not content.strip():
        raise PiConfigValidationError("reasoning smoke response did not include visible assistant content")
    return {
        "finish_reason": None if finish_reason is None else str(finish_reason),
        "reasoning_chars": len(reasoning),
        "answer_chars": len(content),
    }


def _openai_function_tool_call(raw: Any, label: str) -> dict[str, Any]:
    call = _object_value(raw, label)
    _require_exact_keys(call, {"id", "type", "function"}, label)
    call_id = call.get("id")
    if not isinstance(call_id, str) or not call_id.strip():
        raise PiConfigValidationError(f"{label}.id must be a non-empty string")
    if call.get("type") != "function":
        raise PiConfigValidationError(f"{label}.type must be 'function'")
    function = _object_value(call.get("function"), f"{label}.function")
    _require_exact_keys(function, {"name", "arguments"}, f"{label}.function")
    name = function.get("name")
    if not isinstance(name, str) or not name.strip():
        raise PiConfigValidationError(f"{label}.function.name must be a non-empty string")
    arguments = function.get("arguments")
    if not isinstance(arguments, str):
        raise PiConfigValidationError(f"{label}.function.arguments must be a JSON string")
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


def _validate_stream_tool_call_start(call_delta: dict[str, Any], label: str) -> None:
    missing = sorted({"id", "type", "function"} - set(call_delta))
    if missing:
        raise PiConfigValidationError(
            f"{label} first fragment must include id, type, and function; missing={missing}"
        )
    call_id = call_delta.get("id")
    if not isinstance(call_id, str) or not call_id.strip():
        raise PiConfigValidationError(f"{label}.id must be a non-empty string")
    if call_delta.get("type") != "function":
        raise PiConfigValidationError(f"{label}.type must be 'function'")
    function = _object_value(call_delta.get("function"), f"{label}.function")
    if function.get("name") is None:
        raise PiConfigValidationError(f"{label}.function.name is required on the first fragment")
    if function.get("arguments") is not None and not isinstance(function.get("arguments"), str):
        raise PiConfigValidationError(f"{label}.function.arguments must be a string")


def _require_exact_keys(mapping: dict[str, Any], expected: set[str], label: str) -> None:
    actual = set(mapping)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise PiConfigValidationError(
            f"{label} must have exactly keys {sorted(expected)}; "
            f"missing={missing}, extra={extra}"
        )


def _require_subset_keys(mapping: dict[str, Any], allowed: set[str], label: str) -> None:
    extra = sorted(set(mapping) - allowed)
    if extra:
        raise PiConfigValidationError(
            f"{label} contains unsupported OpenAI tool-call fields: {extra}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="docs/examples/pi-agent/models.json")
    parser.add_argument("--provider", help="Validate only one provider entry.")
    parser.add_argument(
        "--base-url",
        help="Validate against a running server, for example http://127.0.0.1:8000/v1.",
    )
    parser.add_argument("--api-key", help="Bearer token for a running server.")
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument(
        "--chat-smoke",
        action="store_true",
        help="Also POST a small Qwen tool-call smoke request to the running server.",
    )
    parser.add_argument(
        "--streaming-smoke",
        action="store_true",
        help="Also POST the Qwen tool-call smoke request as stream=true SSE and require usage metadata.",
    )
    parser.add_argument(
        "--reasoning-smoke",
        action="store_true",
        help="Also POST a small Qwen thinking smoke request to the running server.",
    )
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
        if args.base_url or args.chat_smoke or args.streaming_smoke or args.reasoning_smoke:
            provider = _live_provider(config, provider_name=args.provider)
            base_url = str(args.base_url or provider["baseUrl"])
            api_key = args.api_key if args.api_key is not None else str(provider.get("apiKey") or "")
            capabilities = fetch_capabilities(base_url, api_key=api_key, timeout=args.timeout)
            summary = validate_pi_models_against_capabilities(
                config,
                capabilities,
                provider_name=args.provider,
            )
            if args.chat_smoke:
                response = run_pi_chat_smoke(
                    base_url,
                    config,
                    capabilities,
                    api_key=api_key,
                    timeout=max(args.timeout, 30.0),
                )
                summary["chat_smoke_object"] = response.get("object")
                summary["chat_smoke"] = validate_pi_chat_smoke_response(response)
            if args.streaming_smoke:
                summary["streaming_smoke"] = run_pi_streaming_chat_smoke(
                    base_url,
                    config,
                    capabilities,
                    api_key=api_key,
                    timeout=max(args.timeout, 30.0),
                )
            if args.reasoning_smoke:
                response = run_pi_reasoning_smoke(
                    base_url,
                    config,
                    capabilities,
                    api_key=api_key,
                    timeout=max(args.timeout, 30.0),
                )
                summary["reasoning_smoke_object"] = response.get("object")
                summary["reasoning_smoke"] = validate_pi_reasoning_smoke_response(response)
        else:
            summary = validate_pi_models_config(config, provider_name=args.provider)
    except PiConfigValidationError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2, sort_keys=True), file=sys.stderr)
        return 1

    print(json.dumps({"ok": True, **summary}, indent=2, sort_keys=True))
    return 0


def _validate_provider(provider: dict[str, Any], label: str) -> None:
    errors: list[str] = []
    base_url = provider.get("baseUrl")
    if not isinstance(base_url, str) or not base_url.rstrip("/").endswith("/v1"):
        errors.append(f"{label}.baseUrl must point at the OpenAI /v1 endpoint")
    if provider.get("api") != "openai-completions":
        errors.append(f"{label}.api must be 'openai-completions'")
    try:
        compat = _object(provider, "compat", label=f"{label}.compat")
    except PiConfigValidationError as exc:
        errors.append(str(exc))
    else:
        if compat.get("thinkingFormat") != "qwen":
            errors.append(f"{label}.compat.thinkingFormat must be 'qwen'")
        if compat.get("supportsReasoningEffort") is not False:
            errors.append(f"{label}.compat.supportsReasoningEffort must be false")
        if compat.get("supportsUsageInStreaming") is not True:
            errors.append(f"{label}.compat.supportsUsageInStreaming must be true")
        if compat.get("maxTokensField") != "max_tokens":
            errors.append(f"{label}.compat.maxTokensField must be 'max_tokens'")
    _raise_config_errors(errors)


def _validate_model(model: dict[str, Any], label: str) -> None:
    errors: list[str] = []
    model_id = model.get("id")
    if not isinstance(model_id, str) or not model_id.strip():
        errors.append(f"{label}.id must be a non-empty string")
    if model.get("reasoning") is not True:
        errors.append(f"{label}.reasoning must be true so pi enables thinking for this model")
    try:
        inputs = _list(model.get("input"), f"{label}.input")
    except PiConfigValidationError as exc:
        errors.append(str(exc))
    else:
        if "text" not in {str(item) for item in inputs}:
            errors.append(f"{label}.input must include 'text'")
    context_window: int | None = None
    max_tokens: int | None = None
    try:
        context_window = _positive_int(model.get("contextWindow"), f"{label}.contextWindow")
    except PiConfigValidationError as exc:
        errors.append(str(exc))
    try:
        max_tokens = _positive_int(model.get("maxTokens"), f"{label}.maxTokens")
    except PiConfigValidationError as exc:
        errors.append(str(exc))
    if context_window is not None and max_tokens is not None and max_tokens > context_window:
        errors.append(f"{label}.maxTokens must not exceed contextWindow")
    _raise_config_errors(errors)


def _raise_config_errors(errors: list[str]) -> None:
    if not errors:
        return
    if len(errors) == 1:
        raise PiConfigValidationError(errors[0])
    raise PiConfigValidationError("pi config validation failed: " + "; ".join(errors))


def _live_provider(config: dict[str, Any], *, provider_name: str | None = None) -> dict[str, Any]:
    providers = _object(config, "providers")
    if provider_name is not None:
        if provider_name not in providers:
            raise PiConfigValidationError(f"provider {provider_name!r} is not present")
        return _object_value(providers[provider_name], f"providers.{provider_name}")
    if len(providers) != 1:
        raise PiConfigValidationError("--provider is required when validating multiple providers live")
    provider_id, provider = next(iter(providers.items()))
    return _object_value(provider, f"providers.{provider_id}")


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
        raise PiConfigValidationError(f"{method} {url} failed with HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise PiConfigValidationError(f"{method} {url} failed: {exc}") from exc
    decoded = json.loads(body.decode("utf-8"))
    if not isinstance(decoded, dict):
        raise PiConfigValidationError(f"{method} {url} did not return a JSON object")
    return decoded


def _request_text(
    method: str,
    url: str,
    *,
    api_key: str | None = None,
    payload: dict[str, Any] | None = None,
    timeout: float,
    accept: str = "text/plain",
) -> str:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Accept": accept}
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
        raise PiConfigValidationError(f"{method} {url} failed with HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise PiConfigValidationError(f"{method} {url} failed: {exc}") from exc
    return body.decode("utf-8", errors="replace")


def _parse_sse_payloads(response_text: str) -> tuple[list[dict[str, Any]], bool]:
    payloads: list[dict[str, Any]] = []
    done_seen = False
    for line_number, raw_line in enumerate(response_text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith(":"):
            continue
        if not line.startswith("data: "):
            continue
        data = line[6:]
        if data == "[DONE]":
            done_seen = True
            continue
        try:
            decoded = json.loads(data)
        except json.JSONDecodeError as exc:
            raise PiConfigValidationError(f"SSE data line {line_number} is not valid JSON: {exc}") from exc
        if not isinstance(decoded, dict):
            raise PiConfigValidationError(f"SSE data line {line_number} must decode to a JSON object")
        payloads.append(decoded)
    return payloads, done_seen


def _join_url(base_url: str, path: Any) -> str:
    return str(base_url).rstrip("/") + "/" + str(path).lstrip("/")


def _object(payload: dict[str, Any], key: str, *, label: str | None = None) -> dict[str, Any]:
    return _object_value(payload.get(key), label or key)


def _object_value(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PiConfigValidationError(f"{label} must be a JSON object")
    return value


def _first_message_field_containing(message: dict[str, Any], marker: str) -> str | None:
    for field in ("content", "reasoning_content"):
        value = message.get(field)
        if isinstance(value, str) and marker in value:
            return field
    return None


def _list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise PiConfigValidationError(f"{label} must be a JSON array")
    return list(value)


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PiConfigValidationError(f"{label} must be a positive integer")
    return int(value)


if __name__ == "__main__":
    raise SystemExit(main())
