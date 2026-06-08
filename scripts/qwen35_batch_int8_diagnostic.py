#!/usr/bin/env python3
"""Qwen3.5/PARO INT8-KV c>N generated-token equality diagnostic template.

C3.1 requires a generated-token equality gate for INT8 retained KV.  The
current runtime still blocks compact c>N INT8 native prefill/decode before that
gate can execute, so this script emits a schema-checked ``blocked`` artifact
with the exact future retained-bench command and no throughput claim.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from scripts.qwen35_batch_artifact_schema import validate_cn_diagnostic_artifact_payload
from scripts.qwen35_batch_constants import RETAINED_ARTIFACT_RETAINED_BENCH_SCRIPT
from scripts.qwen35_batch_retained_bench import DEFAULT_FIXTURE, DEFAULT_MODEL

_RETAINED_BENCH_SCRIPT = RETAINED_ARTIFACT_RETAINED_BENCH_SCRIPT
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


def _future_gate_command(args: argparse.Namespace) -> str:
    argv = [
        *_command_env_prefix_parts(),
        "python3",
        _RETAINED_BENCH_SCRIPT,
        "--model",
        str(args.model),
        "--fixture",
        str(args.fixture),
        "--prompt-length",
        str(args.prompt_length),
        "--batch-size",
        str(args.rows),
        "--decode-tokens",
        str(args.decode_tokens),
        "--warmup-decode-tokens",
        str(args.warmup_decode_tokens),
        "--max-layers",
        str(args.max_layers),
        "--kv-storage",
        "int8_per_token_head",
        "--kv-scale-dtype",
        str(args.kv_scale_dtype),
        "--kv-scale-granularity",
        "per_token_head",
        "--int8-kv-primitive-cpu-json",
        str(args.primitive_cpu_json),
        "--int8-kv-primitive-hip-json",
        str(args.primitive_hip_json),
        "--json",
        str(args.future_json),
    ]
    return shlex.join(argv)


def _primitive_layer_accuracy_command(args: argparse.Namespace, *, device: str, output: Path) -> str:
    argv = [
        *_command_env_prefix_parts(),
        "python3",
        "scripts/qwen35_kv_int8_accuracy.py",
        "--device",
        device,
        "--contexts",
        f"{args.prompt_length},{args.prompt_length + 1}",
        "--block-size",
        "256",
        "--num-q-heads",
        "16",
        "--num-kv-heads",
        "2",
        "--head-dim",
        "256",
        "--scale-dtype",
        str(args.kv_scale_dtype),
        "--seed",
        "1234",
        "--json",
        str(output),
    ]
    if device == "hip":
        argv.append("--require-int8-hip")
    return shlex.join(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.rows <= 1:
        raise ValueError("INT8 c>N diagnostic requires --rows > 1")
    if args.prompt_length <= 0 or args.decode_tokens <= 0:
        raise ValueError("prompt/decode token counts must be positive")
    if args.max_layers <= 0:
        raise ValueError("--max-layers must be positive")

    blockers = [
        "compact c>N native prefill is not wired for int8_per_token_head retained KV",
        "step_batch_native currently rejects non-BF16 KV before generated-token equality can run",
        "diagnostic is blocked before execution; no throughput or scaling claim is allowed",
    ]
    payload: dict[str, Any] = {
        "schema": 1,
        "mode": "qwen35_paro_int8_cN_equality_template",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "blocked",
        "performance_claim": False,
        "workload": {
            "shape": f"c={args.rows} prompt={args.prompt_length} decode={args.decode_tokens}",
            "model": "Qwen3.5/3.6-35B-A3B-PARO",
            "model_path": str(args.model),
            "quant": "w4_paro",
            "kv_storage_dtype": "int8_per_token_head",
            "kv_scale_dtype": str(args.kv_scale_dtype),
            "kv_scale_granularity": "per_token_head",
            "fixture": str(args.fixture),
            "prompt_tokens_per_request": int(args.prompt_length),
            "gen_tokens_per_request": int(args.decode_tokens),
            "warmup_decode_tokens": int(args.warmup_decode_tokens),
            "concurrency": int(args.rows),
            "max_layers": int(args.max_layers),
            "native_compact_prefill": False,
            "native_caware_decode": False,
        },
        "commands": {
            "future_generated_token_gate": _future_gate_command(args),
            "primitive_layer_accuracy_cpu_reference": _primitive_layer_accuracy_command(
                args,
                device="cpu",
                output=args.primitive_cpu_json,
            ),
            "primitive_layer_accuracy_hip_gate": _primitive_layer_accuracy_command(
                args,
                device="hip",
                output=args.primitive_hip_json,
            ),
            "correctness_reference": "inline independent c=1 resident rows in qwen35_batch_retained_bench.py once INT8 c>N prefill/decode is wired",
        },
        "correctness": {
            "passed": False,
            "oracle": "generated-token equality vs independent c=1 resident rows",
            "generated_token_equality": {
                "skipped": True,
                "reason": "; ".join(blockers[:2]),
                "batch_sequences": None,
                "c1_sequences": None,
                "mismatches": [],
            },
        },
        "execution": {
            "batch_execution": {
                "path": "int8_cN_blocked_before_execution",
                "scheduler_owned": True,
                "row_execution": "not_executed",
                "native_compact_prefill": False,
                "native_caware_decode": False,
                "throughput_claim_eligible": False,
                "blockers": blockers,
            }
        },
        "decision": {
            "accepted": False,
            "reason": "; ".join(blockers),
        },
        "blockers": blockers,
        "notes": [
            "C3.1 remains open: its acceptance requires eq_ok or rejected_correctness with the generated-token gate executed.",
            "This blocked artifact is for queue tracking and future command reproducibility only.",
        ],
    }
    validate_cn_diagnostic_artifact_payload(payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--fixture", type=Path, default=Path(DEFAULT_FIXTURE))
    parser.add_argument("--rows", type=int, default=2)
    parser.add_argument("--prompt-length", type=int, default=512)
    parser.add_argument("--decode-tokens", type=int, default=128)
    parser.add_argument("--warmup-decode-tokens", type=int, default=8)
    parser.add_argument("--max-layers", type=int, default=40)
    parser.add_argument("--kv-scale-dtype", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--future-json", type=Path, default=Path("/tmp/hipengine-int8-c2-retained-future.json"))
    parser.add_argument("--primitive-cpu-json", type=Path, default=Path("/tmp/hipengine-int8-c2-primitive-cpu.json"))
    parser.add_argument("--primitive-hip-json", type=Path, default=Path("/tmp/hipengine-int8-c2-primitive-hip.json"))
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
    return 0 if payload["status"] in {"blocked", "eq_ok", "rejected_correctness"} else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
