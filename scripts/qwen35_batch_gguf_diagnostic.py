#!/usr/bin/env python3
"""GGUF c>N generated-token equality diagnostic template.

This is the GGUF counterpart to the Qwen/PARO c>N correctness harnesses.  The
current GGUF runtime has public single-request E2E gates, but no native c>N
resident equality runner yet.  Rather than make a throughput claim, this script
emits a compact, reproducible diagnostic artifact with an explicit ``blocked``
status and the exact commands needed to reproduce the blocker / future gate.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from scripts.qwen35_batch_artifact_schema import _load_payload
from scripts.qwen35_batch_constants import (
    RETAINED_ARTIFACT_GGUF_DIAGNOSTIC_SCRIPT,
    RETAINED_ARTIFACT_GGUF_E2E_CORRECTNESS_SCRIPT,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE = REPO_ROOT / "tests/fixtures/gguf/qwen35_0_8b_q4_k_m_e2e.json"
GGUF_QUANTS = ("gguf_q4_k_m", "gguf_q5_k_m", "gguf_q6_k", "gguf_q8_0")
_GGUF_DIAGNOSTIC_SCRIPT = RETAINED_ARTIFACT_GGUF_DIAGNOSTIC_SCRIPT
_GGUF_E2E_CORRECTNESS_SCRIPT = RETAINED_ARTIFACT_GGUF_E2E_CORRECTNESS_SCRIPT
_COMMAND_ENV_KEYS = ("HIP_VISIBLE_DEVICES",)


def _command_env_prefix_parts() -> list[str]:
    assignments = [
        f"{key}={value}"
        for key in _COMMAND_ENV_KEYS
        if (value := os.environ.get(key))
    ]
    return ["env", *assignments] if assignments else []


def _payload_json(payload: Any) -> str:
    return json.dumps(payload, indent=2, allow_nan=False)


def _load_fixture(path: Path) -> dict[str, Any]:
    fixture = _load_payload(path)
    required = {"model", "prompt", "prompt_ids", "sampling", "acceptance"}
    missing = sorted(required - set(fixture))
    if missing:
        raise ValueError(f"fixture {path} missing required keys: {', '.join(missing)}")
    return fixture


def _canonical_command(args: argparse.Namespace) -> str:
    argv = [
        *_command_env_prefix_parts(),
        "python3",
        _GGUF_DIAGNOSTIC_SCRIPT,
        "--fixture",
        str(args.fixture),
        "--rows",
        str(args.rows),
        "--backend",
        str(args.backend),
        "--quant",
        str(args.quant),
        "--max-new-tokens",
        str(args.max_new_tokens),
    ]
    if args.model:
        argv.extend(["--model", str(args.model)])
    return shlex.join(argv)


def _single_row_command(args: argparse.Namespace, *, model: str, row: int) -> str:
    argv = [
        *_command_env_prefix_parts(),
        "python3",
        _GGUF_E2E_CORRECTNESS_SCRIPT,
        "--fixture",
        str(args.fixture),
        "--model",
        model,
        "--backend",
        str(args.backend),
        "--quant",
        str(args.quant),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--json",
        f"/tmp/hipengine-gguf-c1-row{row}.json",
    ]
    return shlex.join(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.rows <= 0:
        raise ValueError("rows must be positive")
    fixture = _load_fixture(Path(args.fixture))
    model = str(args.model or fixture["model"].get("path", ""))
    quant = str(args.quant or fixture["acceptance"].get("quant", ""))
    backend = str(args.backend or fixture["acceptance"].get("backend", ""))
    max_new_tokens = int(args.max_new_tokens or fixture["sampling"].get("max_new_tokens", 0))
    normalized = argparse.Namespace(
        fixture=Path(args.fixture),
        rows=int(args.rows),
        model=str(args.model or ""),
        backend=backend,
        quant=quant,
        max_new_tokens=max_new_tokens,
    )

    blockers = [
        "native GGUF c>N resident equality runner is not wired yet",
        "diagnostic records template commands only; no c>N performance claim is allowed",
    ]
    if int(args.rows) < 2:
        blockers.append("c>N diagnostic requires rows >= 2")
    if quant not in GGUF_QUANTS:
        blockers.append(f"quant {quant!r} is not in the supported GGUF template set {GGUF_QUANTS!r}")
    if model and not Path(model).exists():
        blockers.append(f"model path is not present on this host: {model}")

    independent_c1 = [
        _single_row_command(normalized, model=model, row=row)
        for row in range(int(args.rows))
    ]
    payload = {
        "schema": 1,
        "mode": "gguf_cN_equality_template",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "blocked",
        "rows": int(args.rows),
        "model": model,
        "backend": backend,
        "quant": quant,
        "fixture": str(Path(args.fixture)),
        "prompt_token_count": len(fixture["prompt_ids"]),
        "max_new_tokens": max_new_tokens,
        "command": _canonical_command(normalized),
        "independent_c1_commands": independent_c1,
        "native_cN_command": _canonical_command(normalized),
        "expected_terminal_statuses": ["eq_ok", "blocked", "rejected_correctness"],
        "blockers": blockers,
        "notes": [
            "C3.5 allows blocked/rejected GGUF c>N diagnostics while the native runner is being wired.",
            "A future eq_ok artifact must compare generated token ids from native c>N against independent c=1 rows.",
            "No benchmark rollup or retained performance claim should consume this blocked template artifact.",
        ],
    }
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--model", default="", help="Override fixture model path")
    parser.add_argument("--rows", type=int, default=2)
    parser.add_argument("--backend", default="hip_gfx1100")
    parser.add_argument("--quant", choices=GGUF_QUANTS, default="gguf_q4_k_m")
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--json", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = run(args)
    text = _payload_json(payload)
    print(text)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n")
    return 0 if payload["status"] in {"eq_ok", "blocked", "rejected_correctness"} else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
