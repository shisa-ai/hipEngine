"""Command-line entry point for the optional hipEngine server."""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence

from hipengine.server.api import ServerConfig, create_app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the hipEngine OpenAI-compatible server")
    parser.add_argument("--model", required=True, help="Path or model id served by hipEngine")
    parser.add_argument("--backend", default="hip_gfx1100", help="Kernel backend key")
    parser.add_argument("--quant", default="w4_paro", help="Quantization key")
    parser.add_argument("--served-model-name", help="Public model id exposed by /v1/models")
    parser.add_argument(
        "--api-key",
        default=os.environ.get("HIPENGINE_API_KEY"),
        help="Optional bearer token; defaults to HIPENGINE_API_KEY",
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
    )
    app = create_app(config)
    try:
        import uvicorn
    except Exception as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("uvicorn is required; install hipengine[server]") from exc
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
