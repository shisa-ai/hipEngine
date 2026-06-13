"""Command-line entry point for the hipEngine OpenAI-compatible server."""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence

from hipengine.kvcache import PREFIX_CACHE_CHOICES
from hipengine.server.api import ServerConfig, create_app


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be >= 1")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0.0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return parsed


def _env_positive_int(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return None
    return _positive_int(raw)


def _env_nonnegative_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return float(default)
    return _nonnegative_float(raw)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the hipEngine OpenAI-compatible server")
    parser.add_argument("--model", required=True, help="Path or model id served by hipEngine")
    parser.add_argument(
        "--backend",
        default="auto",
        help=(
            "Kernel backend key (default: auto-detect gfx1100/gfx1151; "
            "use HIPENGINE_BACKEND or this flag to force)"
        ),
    )
    parser.add_argument("--quant", default="w4_paro", help="Quantization key")
    parser.add_argument("--served-model-name", help="Public model id exposed by /v1/models")
    parser.add_argument(
        "--api-key",
        default=os.environ.get("HIPENGINE_API_KEY"),
        help="Optional bearer token; defaults to HIPENGINE_API_KEY",
    )
    parser.add_argument(
        "--eager-load",
        action=argparse.BooleanOptionalAction,
        default=_env_bool("HIPENGINE_EAGER_LOAD", True),
        help="Warm the model/session during server startup (default: true)",
    )
    parser.add_argument(
        "--eager-load-prompt",
        default=os.environ.get("HIPENGINE_EAGER_LOAD_PROMPT", "one two three four"),
        help="Prompt used for startup warmup",
    )
    parser.add_argument(
        "--eager-load-max-tokens",
        type=_positive_int,
        default=int(os.environ.get("HIPENGINE_EAGER_LOAD_MAX_TOKENS", "1")),
        help="Generated tokens used for startup warmup (default: 1)",
    )
    parser.add_argument(
        "--max-context-tokens",
        type=_positive_int,
        default=_env_positive_int("HIPENGINE_MAX_CONTEXT_TOKENS"),
        help=(
            "Resident session/KV context tokens preallocated at startup "
            "(default: auto = min(model max context, estimated allocatable KV context))"
        ),
    )
    parser.add_argument(
        "--kv-storage",
        default=os.environ.get("HIPENGINE_KV_STORAGE", "auto"),
        help="Server-wide KV storage policy: auto, bf16, or int8_per_token_head",
    )
    parser.add_argument(
        "--kv-scale-dtype",
        default=os.environ.get("HIPENGINE_KV_SCALE_DTYPE", "fp16"),
        help="INT8 KV scale dtype: fp16 or fp32 (default: fp16)",
    )
    parser.add_argument(
        "--kv-scale-granularity",
        default=os.environ.get("HIPENGINE_KV_SCALE_GRANULARITY", "per_token_head"),
        help="INT8 KV scale granularity (default: per_token_head)",
    )
    parser.add_argument(
        "--generation-batch-window-ms",
        type=_nonnegative_float,
        default=_env_nonnegative_float("HIPENGINE_GENERATION_BATCH_WINDOW_MS", 0.0),
        help="Milliseconds to opt into cold-path coalescing for compatible requests (default: 0 = off)",
    )
    parser.add_argument(
        "--metrics",
        choices=("off", "prometheus"),
        default=os.environ.get("HIPENGINE_METRICS", "off"),
        help="Metrics endpoint mode (env HIPENGINE_METRICS; default: off)",
    )
    parser.add_argument(
        "--prefix-cache",
        choices=PREFIX_CACHE_CHOICES,
        default=os.environ.get("HIPENGINE_PREFIX_CACHE", "off"),
        help="Prefix-cache mode (env HIPENGINE_PREFIX_CACHE; default: off)",
    )
    parser.add_argument(
        "--debug",
        action=argparse.BooleanOptionalAction,
        default=_env_bool("HIPENGINE_DEBUG", False),
        help="Log full HTTP request/response payloads and extra server diagnostics (default: false)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    parser.add_argument("--log-level", default="info", help="uvicorn log level")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = ServerConfig(
        model=args.model,
        backend=args.backend,
        quant=args.quant,
        served_model_name=args.served_model_name,
        api_key=args.api_key,
        eager_load=args.eager_load,
        eager_load_prompt=args.eager_load_prompt,
        eager_load_max_tokens=args.eager_load_max_tokens,
        max_context_tokens=args.max_context_tokens,
        kv_storage=args.kv_storage,
        kv_scale_dtype=args.kv_scale_dtype,
        kv_scale_granularity=args.kv_scale_granularity,
        generation_batch_window_ms=args.generation_batch_window_ms,
        metrics=args.metrics,
        prefix_cache=args.prefix_cache,
        debug=args.debug,
    )
    app = create_app(config)
    try:
        import uvicorn
    except Exception as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("uvicorn is required; reinstall hipengine with its default dependencies") from exc
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
