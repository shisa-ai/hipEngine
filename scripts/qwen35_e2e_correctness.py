#!/usr/bin/env python3
"""Qwen3.5/PARO resident E2E correctness gate.

This is a correctness smoke, not a benchmark.  It runs real resident c=1
prefill/decode, checks finite logits, verifies deterministic repeated runs, and
optionally checks an expected generated-token list captured from a known-good
reference.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.runtime.qwen35_paro_runner import Qwen35ParoNextTokenRunner, Qwen35ParoResidentSession

DEFAULT_MODEL = (
    "/models/huggingface/hub/models--z-lab--Qwen3.5-35B-A3B-PARO/"
    "snapshots/dca2736e88e9f70855128fc81a8e918043a163cd"
)


def _run_once(
    runner: Qwen35ParoNextTokenRunner,
    prompt_tokens: list[int],
    *,
    max_new_tokens: int,
    max_layers: int,
) -> list[dict[str, Any]]:
    max_sequence = len(prompt_tokens) + max_new_tokens + 1
    out = []
    with Qwen35ParoResidentSession(runner, max_sequence_length=max_sequence, max_layers=max_layers) as session:
        next_result = None
        for pos, token_id in enumerate(prompt_tokens):
            next_result = session.step(token_id, position=pos, sample=(pos == len(prompt_tokens) - 1))
        if next_result is None:
            raise RuntimeError("prefill did not produce a sampled token")
        out.append(next_result.to_json_dict())
        current = next_result
        for offset in range(1, max_new_tokens):
            current = session.step(current.token_id, position=len(prompt_tokens) + offset - 1)
            if current is None:
                raise RuntimeError("decode did not produce a sampled token")
            out.append(current.to_json_dict())
    return out


def _prompt_tokens(token_id: int, prompt_length: int) -> list[int]:
    if prompt_length <= 0:
        raise ValueError("prompt_length must be positive")
    return [int(token_id)] * int(prompt_length)


def run(args: argparse.Namespace) -> dict[str, Any]:
    prompt_tokens = _prompt_tokens(args.token_id, args.prompt_length)
    runner = Qwen35ParoNextTokenRunner(args.model)
    runs = [
        _run_once(runner, prompt_tokens, max_new_tokens=args.max_new_tokens, max_layers=args.max_layers)
        for _ in range(args.repeat)
    ]
    token_ids = [[int(item["token_id"]) for item in run] for run in runs]
    logits = [[float(item["logit"]) for item in run] for run in runs]
    finite_logits = all(math.isfinite(logit) for run in logits for logit in run)
    deterministic = all(ids == token_ids[0] for ids in token_ids)
    expected = tuple(int(item) for item in args.expected_token_ids.split(",") if item.strip()) if args.expected_token_ids else ()
    expected_match = True if not expected else tuple(token_ids[0]) == expected
    passed = finite_logits and deterministic and expected_match
    return {
        "schema": 1,
        "model": str(args.model),
        "quant": "w4_paro",
        "backend": "hip_gfx1100",
        "mode": "resident_c1_e2e_correctness",
        "batch_size": 1,
        "specdec_enabled": False,
        "prompt_source": "repeated_token_id",
        "token_id": int(args.token_id),
        "prompt_length": int(args.prompt_length),
        "max_new_tokens": int(args.max_new_tokens),
        "max_layers": int(args.max_layers),
        "repeat": int(args.repeat),
        "token_ids": token_ids,
        "logits": logits,
        "finite_logits": finite_logits,
        "deterministic": deterministic,
        "expected_token_ids": list(expected),
        "expected_match": expected_match,
        "passed": passed,
        "notes": [
            "c=1 resident E2E gate; c>N parity hooks are separate until batched layer runner lands.",
            "Use --expected-token-ids with parent nano-vllm-amd output when available.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--token-id", type=int, default=9707)
    parser.add_argument("--prompt-length", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--max-layers", type=int, default=1, help="0 means all layers")
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--expected-token-ids", default="", help="Comma-separated expected generated token ids")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if args.max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if args.repeat <= 0:
        raise ValueError("repeat must be positive")
    result = run(args)
    payload = json.dumps(result, indent=2)
    print(payload)
    if args.json is not None:
        args.json.write_text(payload + "\n")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
