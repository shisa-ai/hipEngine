#!/usr/bin/env python3
"""Qwen3.5/PARO resident E2E correctness gate.

This is a correctness smoke, not a benchmark. It runs real resident
prefill/decode, checks finite logits, verifies deterministic repeated runs, and
optionally checks expected generated-token IDs captured from the parent
nano-vllm-amd implementation. Native single-request prefill is the default;
serial c=1 prefill remains an explicit diagnostic mode.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.kvcache import ResolvedKVPolicy
from hipengine.runtime.qwen35_paro_runner import Qwen35ParoNextTokenRunner, Qwen35ParoResidentSession
from scripts.qwen35_kv_policy_args import add_kv_policy_args, kv_policy_json, resolve_args_kv_policy

DEFAULT_MODEL = (
    "/models/huggingface/hub/models--z-lab--Qwen3.5-35B-A3B-PARO/"
    "snapshots/dca2736e88e9f70855128fc81a8e918043a163cd"
)


def _run_once(
    runner: Qwen35ParoNextTokenRunner,
    prompt_tokens: list[int],
    *,
    decode_tokens: int,
    max_layers: int,
    include_prefill_seed: bool,
    prefill_mode: str,
    kv_policy: ResolvedKVPolicy,
) -> dict[str, Any]:
    """Run one resident c=1 prompt+decode pass.

    ``include_prefill_seed=True`` matches public ``LLM.generate`` semantics: the
    argmax from prompt prefill is the first generated token. Parent benchmark
    fixtures use ``False`` because ``bench_paro_native_engine.py`` measures the
    decode loop after consuming that seed token.
    """

    max_sequence = len(prompt_tokens) + decode_tokens + 2
    out: list[dict[str, Any]] = []
    with Qwen35ParoResidentSession(
        runner,
        max_sequence_length=max_sequence,
        max_layers=max_layers,
        kv_policy=kv_policy.create_policy(),
        kv_scale_dtype=kv_policy.scale_dtype,
        kv_scale_granularity=kv_policy.scale_granularity,
    ) as session:
        owned_device_bytes = _owned_device_bytes(session)
        next_result = None
        prefill_start = time.perf_counter()
        if prefill_mode == "native":
            next_result = session.prefill_native(prompt_tokens, sample=True)
        elif prefill_mode == "serial-diagnostic":
            for pos, token_id in enumerate(prompt_tokens):
                next_result = session.step(token_id, position=pos, sample=(pos == len(prompt_tokens) - 1))
        else:
            raise ValueError(f"unsupported prefill_mode {prefill_mode!r}")
        prefill_seconds = time.perf_counter() - prefill_start
        if next_result is None:
            raise RuntimeError("prefill did not produce a sampled token")
        seed = next_result.to_json_dict()
        current = next_result
        if include_prefill_seed:
            out.append(seed)
            decode_iterations = max(0, decode_tokens - 1)
        else:
            decode_iterations = decode_tokens
        decode_start = time.perf_counter()
        for offset in range(decode_iterations):
            current = session.step(current.token_id, position=len(prompt_tokens) + offset)
            if current is None:
                raise RuntimeError("decode did not produce a sampled token")
            out.append(current.to_json_dict())
        decode_seconds = time.perf_counter() - decode_start
    return {
        "seed": seed,
        "generated": out,
        "prefill_seconds": prefill_seconds,
        "decode_seconds": decode_seconds,
        "owned_device_bytes": owned_device_bytes,
        "prefill_execution_detail": getattr(session, "last_prefill_execution", None),
    }


def _prompt_tokens(token_id: int, prompt_length: int) -> list[int]:
    if prompt_length <= 0:
        raise ValueError("prompt_length must be positive")
    return [int(token_id)] * int(prompt_length)


def _load_fixture(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    fixture = json.loads(path.read_text())
    if "prompt_ids" not in fixture or "expected_generated_token_ids" not in fixture:
        raise ValueError("fixture must contain prompt_ids and expected_generated_token_ids")
    return fixture


def _expected_tokens(expected_arg: str, fixture: dict[str, Any] | None) -> tuple[int, ...]:
    if expected_arg:
        return tuple(int(item) for item in expected_arg.split(",") if item.strip())
    if fixture is not None:
        return tuple(int(item) for item in fixture["expected_generated_token_ids"])
    return ()


def _owned_device_bytes(session: Qwen35ParoResidentSession) -> int:
    allocation_bytes = sum(int(allocation.buffer.nbytes) for allocation in session.allocations)
    buffer_bytes = sum(int(buffer.nbytes) for buffer in session.buffers)
    state_bytes = sum(
        int(state.workspace.allocation(name).buffer.nbytes)
        for state in session.states
        for name in state.workspace.names
    )
    return allocation_bytes + buffer_bytes + state_bytes


def run(args: argparse.Namespace) -> dict[str, Any]:
    fixture = _load_fixture(args.fixture)
    include_prefill_seed = fixture is None
    prompt_tokens = list(fixture["prompt_ids"]) if fixture is not None else _prompt_tokens(args.token_id, args.prompt_length)
    decode_tokens = int(fixture["decode_len"]) if fixture is not None and args.max_new_tokens is None else int(args.max_new_tokens or 1)
    expected = _expected_tokens(args.expected_token_ids, fixture)
    runner = Qwen35ParoNextTokenRunner(args.model)
    kv_policy = resolve_args_kv_policy(args, block_size=256)
    runs = [
        _run_once(
            runner,
            prompt_tokens,
            decode_tokens=decode_tokens,
            max_layers=args.max_layers,
            include_prefill_seed=include_prefill_seed,
            prefill_mode=args.prefill_mode,
            kv_policy=kv_policy,
        )
        for _ in range(args.repeat)
    ]
    generated_runs = [run["generated"] for run in runs]
    token_ids = [[int(item["token_id"]) for item in generated] for generated in generated_runs]
    logits = [[float(item["logit"]) for item in generated] for generated in generated_runs]
    seed_token_ids = [int(run["seed"]["token_id"]) for run in runs]
    finite_logits = all(math.isfinite(logit) for run in logits for logit in run)
    deterministic = all(ids == token_ids[0] for ids in token_ids) and all(seed == seed_token_ids[0] for seed in seed_token_ids)
    expected_match = True if not expected else tuple(token_ids[0]) == expected
    passed = finite_logits and deterministic and expected_match
    prefill_seconds = [float(run["prefill_seconds"]) for run in runs]
    decode_seconds = [float(run["decode_seconds"]) for run in runs]
    owned_device_bytes = [int(run["owned_device_bytes"]) for run in runs]
    parent_metrics = fixture.get("parent_metrics") if fixture is not None else None
    return {
        "schema": 2,
        "model": str(args.model),
        "quant": "w4_paro",
        "backend": "hip_gfx1100",
        "mode": "resident_c1_e2e_correctness",
        "prefill_mode": args.prefill_mode,
        "kv_storage_dtype": kv_policy.storage_dtype.value,
        "kv_policy": kv_policy_json(kv_policy),
        "batch_size": 1,
        "specdec_enabled": False,
        "prompt_source": "parent_fixture" if fixture is not None else "repeated_token_id",
        "fixture": None if fixture is None else args.fixture.as_posix(),
        "token_id": int(args.token_id) if fixture is None else None,
        "prompt_length": len(prompt_tokens),
        "max_new_tokens": decode_tokens,
        "include_prefill_seed_in_generated": include_prefill_seed,
        "max_layers": int(args.max_layers),
        "repeat": int(args.repeat),
        "seed_token_ids": seed_token_ids,
        "prefill_execution_details": [run["prefill_execution_detail"] for run in runs],
        "token_ids": token_ids,
        "logits": logits,
        "timings": {
            "prefill_seconds": prefill_seconds,
            "decode_seconds": decode_seconds,
            "prefill_tok_s": [len(prompt_tokens) / item if item > 0 else None for item in prefill_seconds],
            "decode_tok_s": [decode_tokens / item if item > 0 else None for item in decode_seconds],
        },
        "memory": {
            "owned_device_bytes": owned_device_bytes,
            "owned_device_gib": [item / (1024**3) for item in owned_device_bytes],
            "parent_metrics": parent_metrics,
        },
        "finite_logits": finite_logits,
        "deterministic": deterministic,
        "expected_token_ids": list(expected),
        "expected_match": expected_match,
        "passed": passed,
        "notes": [
            "c=1 resident E2E gate; c>N parity hooks are separate until batched layer runner lands.",
            "Native prefill mode uses prefill_native(...); serial-diagnostic mode is an explicit fallback only.",
            "Fixture mode compares hipEngine decode-loop outputs against parent nano-vllm-amd outputs after consuming the prefill seed token.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--token-id", type=int, default=9707)
    parser.add_argument("--prompt-length", type=int, default=4)
    parser.add_argument(
        "--prefill-mode",
        choices=("native", "serial-diagnostic"),
        default="native",
        help="Prompt prefill implementation to gate; native is the retained path.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--max-layers", type=int, default=1, help="0 means all layers")
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--expected-token-ids", default="", help="Comma-separated expected generated token ids")
    parser.add_argument("--fixture", type=Path, help="Parent fixture JSON containing prompt_ids and expected_generated_token_ids")
    add_kv_policy_args(parser, help_prefix="Resident KV storage for E2E correctness")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if args.max_new_tokens is not None and args.max_new_tokens <= 0:
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
