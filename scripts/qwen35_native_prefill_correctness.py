#!/usr/bin/env python3
"""Compare Qwen3.5/PARO native prefill against serial c=1.

This is a correctness/blocker helper, not a benchmark.  It runs the same fixed
prompt through (1) serial token-by-token resident prefill and (2)
``prefill_native(...)``, then compares the prefill seed token and one decode
token. Mismatches are emitted as ``rejected_correctness`` artifacts.
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

from hipengine.kvcache import ResolvedKVPolicy, resolve_kv_policy
from hipengine.runtime.qwen35_paro_runner import Qwen35ParoNextTokenRunner, Qwen35ParoResidentSession
from scripts.qwen35_kv_policy_args import add_kv_policy_args, append_kv_policy_flags, kv_policy_json, resolve_args_kv_policy

DEFAULT_MODEL = (
    "/models/huggingface/hub/models--z-lab--Qwen3.5-35B-A3B-PARO/"
    "snapshots/dca2736e88e9f70855128fc81a8e918043a163cd"
)


def _prompt_tokens(token_id: int, prompt_length: int) -> list[int]:
    if prompt_length <= 0:
        raise ValueError("prompt_length must be positive")
    return [int(token_id)] * int(prompt_length)


def _result_dict(result) -> dict[str, Any]:
    return {
        "token_id": int(result.token_id),
        "token_text": result.token_text,
        "logit": float(result.logit),
    }


def _run_serial(
    runner: Qwen35ParoNextTokenRunner,
    prompt_tokens: list[int],
    *,
    max_layers: int,
    kv_policy: ResolvedKVPolicy,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    with Qwen35ParoResidentSession(
        runner,
        max_sequence_length=len(prompt_tokens) + 2,
        max_layers=max_layers,
        kv_policy=kv_policy.create_policy(),
        kv_scale_dtype=kv_policy.scale_dtype,
        kv_scale_granularity=kv_policy.scale_granularity,
    ) as session:
        plan = session.native_prefill_plan().to_json_dict()
        seed = None
        for position, token_id in enumerate(prompt_tokens):
            seed = session.step(token_id, position=position, sample=(position == len(prompt_tokens) - 1))
        if seed is None:
            raise RuntimeError("serial prefill did not produce a seed token")
        decode = session.step(seed.token_id, position=len(prompt_tokens), sample=True)
        if decode is None:
            raise RuntimeError("serial decode did not produce a token")
    return _result_dict(seed), _result_dict(decode), plan


def _run_native(
    runner: Qwen35ParoNextTokenRunner,
    prompt_tokens: list[int],
    *,
    max_layers: int,
    kv_policy: ResolvedKVPolicy,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    with Qwen35ParoResidentSession(
        runner,
        max_sequence_length=len(prompt_tokens) + 2,
        max_layers=max_layers,
        kv_policy=kv_policy.create_policy(),
        kv_scale_dtype=kv_policy.scale_dtype,
        kv_scale_granularity=kv_policy.scale_granularity,
    ) as session:
        plan = session.native_prefill_plan().to_json_dict()
        seed = session.prefill_native(prompt_tokens, sample=True)
        if seed is None:
            raise RuntimeError("native prefill did not produce a seed token")
        decode = session.step(seed.token_id, position=len(prompt_tokens), sample=True)
        if decode is None:
            raise RuntimeError("native decode did not produce a token")
    return _result_dict(seed), _result_dict(decode), plan


def _finite(*rows: dict[str, Any]) -> bool:
    return all(math.isfinite(float(row["logit"])) for row in rows)


def _case_payload(
    runner: Qwen35ParoNextTokenRunner,
    prompt_tokens: list[int],
    *,
    max_layers: int,
    serial_kv_policy: ResolvedKVPolicy,
    native_kv_policy: ResolvedKVPolicy,
) -> dict[str, Any]:
    serial_seed, serial_decode, serial_plan = _run_serial(runner, prompt_tokens, max_layers=max_layers, kv_policy=serial_kv_policy)
    native_seed, native_decode, native_plan = _run_native(runner, prompt_tokens, max_layers=max_layers, kv_policy=native_kv_policy)
    seed_match = native_seed["token_id"] == serial_seed["token_id"]
    decode_match = native_decode["token_id"] == serial_decode["token_id"]
    finite_logits = _finite(serial_seed, serial_decode, native_seed, native_decode)
    passed = finite_logits and seed_match and decode_match
    return {
        "status": "accepted" if passed else "rejected_correctness",
        "max_layers": int(max_layers),
        "serial_native_prefill_plan": serial_plan,
        "native_prefill_plan": native_plan,
        "serial_reference_kv_policy": kv_policy_json(serial_kv_policy),
        "native_kv_policy": kv_policy_json(native_kv_policy),
        "serial": {"seed": serial_seed, "decode": serial_decode},
        "native": {"seed": native_seed, "decode": native_decode},
        "seed_match": seed_match,
        "decode_match": decode_match,
        "finite_logits": finite_logits,
        "logit_abs_delta": {
            "seed": abs(float(native_seed["logit"]) - float(serial_seed["logit"])),
            "decode": abs(float(native_decode["logit"]) - float(serial_decode["logit"])),
        },
        "passed": passed,
    }


def _command(args: argparse.Namespace) -> str:
    command = (
        "python3 scripts/qwen35_native_prefill_correctness.py "
        f"--model {args.model} --token-id {args.token_id} --prompt-length {args.prompt_length}"
    )
    if args.sweep_layer_prefixes is None:
        command += f" --max-layers {args.max_layers}"
    else:
        command += f" --sweep-layer-prefixes {args.sweep_layer_prefixes}"
    command = append_kv_policy_flags(command, args)
    if args.json is not None:
        command += f" --json {args.json}"
    return command


def _base_payload(args: argparse.Namespace, prompt_tokens: list[int]) -> dict[str, Any]:
    native_kv_policy = resolve_args_kv_policy(args, block_size=256)
    serial_kv_policy = resolve_kv_policy("bf16")
    return {
        "schema": 1,
        "model": str(Path(args.model)),
        "quant": "w4_paro",
        "backend": "hip_gfx1100",
        "command": _command(args),
        "performance_claim": False,
        "prompt_source": "repeated_token_id",
        "token_id": int(args.token_id),
        "prompt_length": len(prompt_tokens),
        "kv_storage_dtype": native_kv_policy.storage_dtype.value,
        "kv_policy": kv_policy_json(native_kv_policy),
        "serial_reference_kv_policy": kv_policy_json(serial_kv_policy),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--token-id", type=int, default=9707)
    parser.add_argument("--prompt-length", type=int, default=4)
    parser.add_argument("--max-layers", type=int, default=3)
    parser.add_argument(
        "--sweep-layer-prefixes",
        type=int,
        help="Run max_layers=1..N and report the first prefix that mismatches serial c=1.",
    )
    add_kv_policy_args(parser, help_prefix="KV storage for native prefill candidate; serial reference remains BF16")
    parser.add_argument("--json", type=Path, help="Optional path to write JSON output")
    args = parser.parse_args(argv)

    if args.max_layers <= 0:
        raise ValueError("max_layers must select a finite layer prefix for this helper")
    if args.sweep_layer_prefixes is not None and args.sweep_layer_prefixes <= 0:
        raise ValueError("sweep-layer-prefixes must be positive")
    prompt_tokens = _prompt_tokens(args.token_id, args.prompt_length)
    runner = Qwen35ParoNextTokenRunner(Path(args.model))
    serial_kv_policy = resolve_kv_policy("bf16")
    native_kv_policy = resolve_args_kv_policy(args, block_size=256)
    if args.sweep_layer_prefixes is None:
        case = _case_payload(
            runner,
            prompt_tokens,
            max_layers=args.max_layers,
            serial_kv_policy=serial_kv_policy,
            native_kv_policy=native_kv_policy,
        )
        passed = bool(case["passed"])
        payload = {
            **_base_payload(args, prompt_tokens),
            "status": "accepted" if passed else "rejected_correctness",
            "blocked_reason": None if passed else "native prefill does not match serial c=1 token-by-token prefill",
            "mode": "qwen35_paro_native_prefill_vs_serial_correctness",
            **case,
            "notes": [
                "Correctness helper only; timings are intentionally omitted and no throughput claim is made.",
                "The native prefill helper is validated against serial c=1 before retaining generation or throughput artifacts.",
            ],
        }
    else:
        cases = [
            _case_payload(
                runner,
                prompt_tokens,
                max_layers=layers,
                serial_kv_policy=serial_kv_policy,
                native_kv_policy=native_kv_policy,
            )
            for layers in range(1, int(args.sweep_layer_prefixes) + 1)
        ]
        first_mismatch = next((case["max_layers"] for case in cases if not case["passed"]), None)
        passed = first_mismatch is None
        payload = {
            **_base_payload(args, prompt_tokens),
            "status": "accepted" if passed else "rejected_correctness",
            "blocked_reason": None if passed else "native prefill mismatches serial c=1 at one or more layer prefixes",
            "mode": "qwen35_paro_native_prefill_prefix_sweep_correctness",
            "sweep_layer_prefixes": int(args.sweep_layer_prefixes),
            "first_mismatching_prefix": first_mismatch,
            "cases": cases,
            "passed": passed,
            "notes": [
                "Correctness helper only; timings are intentionally omitted and no throughput claim is made.",
                "Sweep mode validates native prefill and reports the first mismatching layer prefix, if any.",
            ],
        }
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    print(text)
    if args.json is not None:
        args.json.write_text(text + "\n")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
