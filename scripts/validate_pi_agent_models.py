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
    for provider_id, raw_provider in selected:
        label = f"providers.{provider_id}"
        provider = _object_value(raw_provider, label)
        _validate_provider(provider, label)
        models = _list(provider.get("models"), f"{label}.models")
        if not models:
            raise PiConfigValidationError(f"{label}.models must contain at least one model")
        model_summaries = []
        for index, raw_model in enumerate(models):
            model = _object_value(raw_model, f"{label}.models[{index}]")
            _validate_model(model, f"{label}.models[{index}]")
            model_summaries.append(
                {
                    "id": str(model["id"]),
                    "reasoning": bool(model["reasoning"]),
                    "contextWindow": int(model["contextWindow"]),
                    "maxTokens": int(model["maxTokens"]),
                }
            )
        summaries.append(
            {
                "provider": provider_id,
                "baseUrl": str(provider["baseUrl"]),
                "models": model_summaries,
            }
        )

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
                "content": "Use the provided tool to record ok.",
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


def validate_pi_chat_smoke_response(response: dict[str, Any]) -> dict[str, Any]:
    choices = _list(response.get("choices"), "chat smoke response.choices")
    if not choices:
        raise PiConfigValidationError("chat smoke response.choices must contain at least one choice")
    choice = _object_value(choices[0], "chat smoke response.choices[0]")
    message = _object(choice, "message", label="chat smoke response.choices[0].message")
    if choice.get("finish_reason") != "tool_calls":
        content = message.get("content")
        if isinstance(content, str) and "<tool_call>" in content:
            raise PiConfigValidationError(
                "chat smoke returned raw <tool_call> text instead of parsed message.tool_calls; "
                "check that pi is using the OpenAI chat-completions adapter with tools enabled"
            )
        raise PiConfigValidationError(
            "chat smoke did not finish with tool_calls; "
            f"finish_reason={choice.get('finish_reason')!r}"
        )
    tool_calls = _list(message.get("tool_calls"), "chat smoke response message.tool_calls")
    if len(tool_calls) != 1:
        raise PiConfigValidationError(
            f"chat smoke expected exactly one tool call, got {len(tool_calls)}"
        )
    call = _object_value(tool_calls[0], "chat smoke response message.tool_calls[0]")
    function = _object(call, "function", label="chat smoke response tool call function")
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
    if "result" not in decoded_args or not isinstance(decoded_args["result"], str):
        raise PiConfigValidationError("chat smoke tool arguments must include string field 'result'")
    return {
        "finish_reason": "tool_calls",
        "tool_name": "record_result",
        "argument_keys": sorted(str(key) for key in decoded_args),
    }


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
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
        if args.base_url or args.chat_smoke:
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
        else:
            summary = validate_pi_models_config(config, provider_name=args.provider)
    except PiConfigValidationError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2, sort_keys=True), file=sys.stderr)
        return 1

    print(json.dumps({"ok": True, **summary}, indent=2, sort_keys=True))
    return 0


def _validate_provider(provider: dict[str, Any], label: str) -> None:
    base_url = provider.get("baseUrl")
    if not isinstance(base_url, str) or not base_url.rstrip("/").endswith("/v1"):
        raise PiConfigValidationError(f"{label}.baseUrl must point at the OpenAI /v1 endpoint")
    if provider.get("api") != "openai-completions":
        raise PiConfigValidationError(f"{label}.api must be 'openai-completions'")
    compat = _object(provider, "compat", label=f"{label}.compat")
    if compat.get("thinkingFormat") != "qwen":
        raise PiConfigValidationError(f"{label}.compat.thinkingFormat must be 'qwen'")
    if compat.get("supportsReasoningEffort") is not False:
        raise PiConfigValidationError(f"{label}.compat.supportsReasoningEffort must be false")
    if compat.get("supportsUsageInStreaming") is not True:
        raise PiConfigValidationError(f"{label}.compat.supportsUsageInStreaming must be true")
    if compat.get("maxTokensField") != "max_tokens":
        raise PiConfigValidationError(f"{label}.compat.maxTokensField must be 'max_tokens'")


def _validate_model(model: dict[str, Any], label: str) -> None:
    model_id = model.get("id")
    if not isinstance(model_id, str) or not model_id.strip():
        raise PiConfigValidationError(f"{label}.id must be a non-empty string")
    if model.get("reasoning") is not True:
        raise PiConfigValidationError(
            f"{label}.reasoning must be true so pi enables thinking for this model"
        )
    inputs = _list(model.get("input"), f"{label}.input")
    if "text" not in {str(item) for item in inputs}:
        raise PiConfigValidationError(f"{label}.input must include 'text'")
    context_window = _positive_int(model.get("contextWindow"), f"{label}.contextWindow")
    max_tokens = _positive_int(model.get("maxTokens"), f"{label}.maxTokens")
    if max_tokens > context_window:
        raise PiConfigValidationError(f"{label}.maxTokens must not exceed contextWindow")


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


def _join_url(base_url: str, path: Any) -> str:
    return str(base_url).rstrip("/") + "/" + str(path).lstrip("/")


def _object(payload: dict[str, Any], key: str, *, label: str | None = None) -> dict[str, Any]:
    return _object_value(payload.get(key), label or key)


def _object_value(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PiConfigValidationError(f"{label} must be a JSON object")
    return value


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
