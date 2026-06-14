#!/usr/bin/env python3
"""Validate a pi-agent models.json for the hipEngine Qwen endpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="docs/examples/pi-agent/models.json")
    parser.add_argument("--provider", help="Validate only one provider entry.")
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
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
