#!/usr/bin/env python3
"""Emit the reproducible llama.cpp trace-capture plan for GGUF MTP greeting B2.

This script does not launch llama.cpp.  It writes a compact JSON plan containing
(1) the verbose llama-server command needed to expose draft-MTP candidates,
(2) the deterministic request payload for the greeting prompt, and
(3) the parser command that converts the captured log into a trace artifact.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_SERVER_BIN = "/home/lhl/llama.cpp/llama.cpp-hip/build/bin/llama-server"
DEFAULT_MODEL = "/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8093
DEFAULT_PROMPT = "Write a short greeting."
DEFAULT_TRACE_JSON = "benchmarks/results/llamacpp-mtp-greeting-b2-draft-trace.json"
DEFAULT_SERVER_LOG = "benchmarks/results/llamacpp-mtp-greeting-b2-server.log"
DEFAULT_METADATA_JSON = "benchmarks/results/llamacpp-mtp-greeting-b2-metadata.json"


def build_trace_plan(
    *,
    server_bin: str = DEFAULT_SERVER_BIN,
    model: str = DEFAULT_MODEL,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    prompt: str = DEFAULT_PROMPT,
    draft_n_max: int = 2,
    top_k: int = 10,
    max_tokens: int = 6,
    seed: int = 12345,
    server_log: str = DEFAULT_SERVER_LOG,
    metadata_json: str = DEFAULT_METADATA_JSON,
    trace_json: str = DEFAULT_TRACE_JSON,
) -> dict[str, Any]:
    """Build a deterministic command plan for the needed llama.cpp trace."""
    server_command = [
        server_bin,
        "-m",
        model,
        "--host",
        host,
        "--port",
        str(port),
        "-c",
        "4096",
        "-ngl",
        "999",
        "--no-webui",
        "--spec-type",
        "draft-mtp",
        "--spec-draft-n-max",
        str(draft_n_max),
        "--no-spec-draft-backend-sampling",
        "--log-verbosity",
        "5",
        "--no-log-prefix",
        "--no-log-timestamps",
    ]
    request_endpoint = f"http://{host}:{port}/v1/chat/completions"
    request_payload = {
        "model": "qwen36",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "top_k": 1,
        "top_p": 1.0,
        "min_p": 0.0,
        "max_tokens": max_tokens,
        "seed": seed,
        "stream": False,
        "cache_prompt": False,
    }
    metadata = {
        "request": {
            "prompt_name": "greeting",
            "prompt": prompt,
            "draft_n_max": draft_n_max,
            "top_k_candidates": top_k,
            "temperature": 0.0,
            "seed": seed,
            "max_tokens": max_tokens,
        },
        "native_reference_artifact": "benchmarks/results/mtp-bench-1781845600-b2-greeting-topk-diagnostic.json",
        "reason": "Compare llama.cpp draft-mtp top-k candidates to native MTP top-k where native target 220 is absent.",
    }
    parser_command = [
        "python3",
        "scripts/llamacpp_mtp_draft_trace.py",
        server_log,
        "--top-k",
        str(top_k),
        "--metadata",
        metadata_json,
        "--out",
        trace_json,
    ]
    return {
        "schema": 1,
        "kind": "llamacpp_mtp_trace_capture_plan",
        "server_command": server_command,
        "server_log": server_log,
        "request_endpoint": request_endpoint,
        "request_payload": request_payload,
        "metadata_path": metadata_json,
        "metadata": metadata,
        "parser_command": parser_command,
        "trace_json": trace_json,
        "instructions": [
            "Start server_command and redirect stdout/stderr to server_log.",
            "Wait for /health, then POST request_payload to request_endpoint.",
            "Stop llama-server after response, write metadata to metadata_path, then run parser_command.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-bin", default=DEFAULT_SERVER_BIN)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--draft-n-max", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=6)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--server-log", default=DEFAULT_SERVER_LOG)
    parser.add_argument("--metadata-json", default=DEFAULT_METADATA_JSON)
    parser.add_argument("--trace-json", default=DEFAULT_TRACE_JSON)
    parser.add_argument("--out", type=Path, help="Write the plan JSON to this file")
    args = parser.parse_args()

    plan = build_trace_plan(
        server_bin=args.server_bin,
        model=args.model,
        host=args.host,
        port=args.port,
        prompt=args.prompt,
        draft_n_max=args.draft_n_max,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        seed=args.seed,
        server_log=args.server_log,
        metadata_json=args.metadata_json,
        trace_json=args.trace_json,
    )
    payload = json.dumps(plan, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload)
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
