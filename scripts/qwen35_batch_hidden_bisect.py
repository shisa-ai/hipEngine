#!/usr/bin/env python3
"""Qwen3.5/PARO native c>N hidden-state bisection diagnostic.

This diagnostic compares compact native c=2 decode hidden tensors against
independent c=1 resident sessions at configurable layer limits.  It is a
correctness-only tool: it emits JSON with token and hidden mismatches, and it
never marks a throughput result accepted.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.core.dtype import DType
from hipengine.core.memory import DeviceBuffer, copy_device_to_host, host_array_ptr
from hipengine.generation import ResidentBatchScheduler
from hipengine.kvcache import FixedPagedKVPolicy
from hipengine.runtime.qwen35_paro_runner import Qwen35ParoNextTokenRunner, Qwen35ParoResidentSession
from scripts.qwen35_batch_retained_bench import DEFAULT_FIXTURE, DEFAULT_MODEL, _compiler_version, _load_prompt_slices


@dataclass(frozen=True)
class HiddenRun:
    seed_tokens: list[int]
    generated_tokens: list[list[int]]
    hidden_bits_by_step: list[np.ndarray]


def _command(argv: Sequence[str] | None) -> str:
    parts = ["python3", "scripts/qwen35_batch_hidden_bisect.py"]
    parts.extend(sys.argv[1:] if argv is None else list(argv))
    return " ".join(shlex.quote(part) for part in parts)


def _parse_layer_limits(value: str | None, *, max_layers: int) -> list[int]:
    if max_layers <= 0:
        raise ValueError("max_layers must be positive")
    if value is None or not value.strip() or value.strip().lower() == "all":
        return list(range(1, max_layers + 1))
    limits: list[int] = []
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            if end < start:
                raise ValueError("layer limit ranges must be ascending")
            limits.extend(range(start, end + 1))
        else:
            limits.append(int(part))
    if not limits:
        raise ValueError("at least one layer limit is required")
    deduped = sorted(set(limits))
    if deduped[0] <= 0 or deduped[-1] > max_layers:
        raise ValueError(f"layer limits must be within [1, {max_layers}]")
    return deduped


def _fp16_bits_to_f32(bits: np.ndarray) -> np.ndarray:
    return np.asarray(bits, dtype=np.uint16).view(np.float16).astype(np.float32)


def hidden_comparison(batch_bits: np.ndarray, c1_bits: np.ndarray, *, atol: float) -> dict[str, Any]:
    if batch_bits.shape != c1_bits.shape:
        raise ValueError(f"hidden shapes differ: batch={batch_bits.shape!r} c1={c1_bits.shape!r}")
    diff = np.abs(_fp16_bits_to_f32(batch_bits) - _fp16_bits_to_f32(c1_bits))
    bit_mismatch = int(np.count_nonzero(np.asarray(batch_bits, dtype=np.uint16) != np.asarray(c1_bits, dtype=np.uint16)))
    max_abs = float(diff.max(initial=0.0))
    return {
        "shape": list(batch_bits.shape),
        "max_abs": max_abs,
        "mean_abs": float(diff.mean()) if diff.size else 0.0,
        "bit_mismatch": bit_mismatch,
        "passed": bool(max_abs <= float(atol)),
    }


def _first_hidden_mismatch(layer_summaries: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    for summary in layer_summaries:
        for step in summary.get("steps", []):
            for row in step.get("rows", []):
                comparison = row.get("hidden_comparison", {})
                if not comparison.get("passed", False):
                    result: dict[str, Any] = {
                        "layer_limit": int(summary["layer_limit"]),
                        "decode_step": int(step["decode_step"]),
                        "generated_index": int(step["generated_index"]),
                        "row": int(row["row"]),
                        "max_abs": float(comparison.get("max_abs", 0.0)),
                        "bit_mismatch": int(comparison.get("bit_mismatch", 0)),
                    }
                    if "last_layer_index" in summary:
                        result["last_layer_index"] = int(summary["last_layer_index"])
                    if "last_layer_type" in summary:
                        result["last_layer_type"] = str(summary["last_layer_type"])
                    return result
    return None


def _first_token_mismatch(layer_summaries: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    for summary in layer_summaries:
        for row in summary.get("token_mismatches", []):
            return {"layer_limit": int(summary["layer_limit"]), **row}
    return None


def _copy_hidden_bits(session: Qwen35ParoResidentSession, hidden, *, rows: int) -> np.ndarray:
    bits = np.empty((rows, session.config.hidden_size), dtype=np.uint16)
    copy_device_to_host(
        host_array_ptr(bits),
        DeviceBuffer(hidden.ptr, bits.nbytes),
        runtime=session.runtime,
    )
    return bits


def _prefill_batch(
    session: Qwen35ParoResidentSession,
    prompts: list[list[int]],
    *,
    decode_tokens: int,
) -> list[int]:
    scheduler = ResidentBatchScheduler(capacity=len(prompts))
    request_ids = [scheduler.submit(prompt, max_new_tokens=decode_tokens) for prompt in prompts]
    admitted = scheduler.admit_pending()
    if tuple(request_ids) != tuple(admitted):
        raise RuntimeError(f"unexpected admitted request ids {admitted!r}")
    slabs = scheduler.next_compact_prefill_slabs(chunk_size=max(len(prompt) for prompt in prompts), block_size=session.block_size)
    if len(slabs) != 1:
        raise RuntimeError(f"expected one compact prefill slab, got {len(slabs)}")
    results = session.prefill_native_packed(slabs[0], sample=True)
    seed_tokens: list[int] = []
    for result in results:
        if result is None:
            raise RuntimeError("batch prefill did not produce a seed token")
        seed_tokens.append(int(result.token_id))
    return seed_tokens


def _run_batch_hidden(
    runner: Qwen35ParoNextTokenRunner,
    prompts: list[list[int]],
    *,
    layer_limit: int,
    decode_tokens: int,
    max_sequence_length: int,
    compiler_version: str | None,
    require_cached_build: bool,
) -> HiddenRun:
    rows = len(prompts)
    with Qwen35ParoResidentSession(
        runner,
        max_sequence_length=max_sequence_length,
        max_layers=layer_limit,
        max_batch_size=rows,
        compiler_version=compiler_version,
        require_cached_build=require_cached_build,
        kv_policy=FixedPagedKVPolicy(block_size=256, storage_dtype=DType.BF16),
    ) as session:
        seed_tokens = _prefill_batch(session, prompts, decode_tokens=decode_tokens)
        next_tokens = list(seed_tokens)
        generated_tokens = [[] for _ in prompts]
        hidden_bits_by_step: list[np.ndarray] = []
        for step in range(decode_tokens):
            positions = tuple(len(prompt) + step for prompt in prompts)
            session._set_batch_token_embeddings(next_tokens, stream=0)
            session._set_batch_positions(positions, stream=0)
            hidden = session._run_layers_batch_decode(
                rows=rows,
                positions=positions,
                slots=tuple(range(rows)),
                stream=0,
            )
            session.runtime.device_synchronize()
            hidden_bits_by_step.append(_copy_hidden_bits(session, hidden, rows=rows))
            results = session._sample_batch_from_hidden(hidden, rows=rows)
            next_tokens = []
            for row, result in enumerate(results):
                token_id = int(result.token_id)
                generated_tokens[row].append(token_id)
                next_tokens.append(token_id)
        return HiddenRun(seed_tokens=seed_tokens, generated_tokens=generated_tokens, hidden_bits_by_step=hidden_bits_by_step)


def _run_c1_hidden(
    runner: Qwen35ParoNextTokenRunner,
    prompts: list[list[int]],
    *,
    layer_limit: int,
    decode_tokens: int,
    max_sequence_length: int,
    compiler_version: str | None,
    require_cached_build: bool,
) -> HiddenRun:
    rows = len(prompts)
    seed_tokens: list[int] = []
    generated_tokens: list[list[int]] = []
    hidden_by_step = [np.empty((rows, runner.config.hidden_size), dtype=np.uint16) for _ in range(decode_tokens)]
    with Qwen35ParoResidentSession(
        runner,
        max_sequence_length=max_sequence_length,
        max_layers=layer_limit,
        max_batch_size=1,
        compiler_version=compiler_version,
        require_cached_build=require_cached_build,
        kv_policy=FixedPagedKVPolicy(block_size=256, storage_dtype=DType.BF16),
    ) as session:
        for row, prompt in enumerate(prompts):
            result = session.prefill_native(prompt, sample=True)
            if result is None:
                raise RuntimeError("c=1 prefill did not produce a seed token")
            next_token = int(result.token_id)
            seed_tokens.append(next_token)
            row_generated: list[int] = []
            for step in range(decode_tokens):
                position = len(prompt) + step
                session._set_token_embedding(next_token, stream=0)
                session._set_position(position, stream=0)
                hidden = session._run_layers(position=position, stream=0)
                session.runtime.device_synchronize()
                hidden_by_step[step][row : row + 1] = _copy_hidden_bits(session, hidden, rows=1)
                step_result = session._sample_from_hidden(hidden)
                next_token = int(step_result.token_id)
                row_generated.append(next_token)
            generated_tokens.append(row_generated)
            session.reset()
    return HiddenRun(seed_tokens=seed_tokens, generated_tokens=generated_tokens, hidden_bits_by_step=hidden_by_step)


def _token_mismatches(batch: HiddenRun, c1: HiddenRun) -> list[dict[str, Any]]:
    mismatches: list[dict[str, Any]] = []
    for row, (batch_seed, c1_seed) in enumerate(zip(batch.seed_tokens, c1.seed_tokens, strict=True)):
        batch_sequence = [int(batch_seed), *[int(token) for token in batch.generated_tokens[row]]]
        c1_sequence = [int(c1_seed), *[int(token) for token in c1.generated_tokens[row]]]
        if batch_sequence != c1_sequence:
            first_index = next(
                (idx for idx, (left, right) in enumerate(zip(batch_sequence, c1_sequence, strict=False)) if left != right),
                min(len(batch_sequence), len(c1_sequence)),
            )
            mismatches.append(
                {
                    "row": row,
                    "first_index": int(first_index),
                    "batch": batch_sequence,
                    "c1": c1_sequence,
                }
            )
    return mismatches


def _layer_limit_metadata(layer_limit: int, layer_types: Sequence[str] | None) -> dict[str, Any]:
    metadata: dict[str, Any] = {"last_layer_index": int(layer_limit) - 1}
    if layer_types is not None and 0 <= metadata["last_layer_index"] < len(layer_types):
        metadata["last_layer_type"] = str(layer_types[metadata["last_layer_index"]])
    return metadata


def _summarize_layer_limit(
    batch: HiddenRun,
    c1: HiddenRun,
    *,
    layer_limit: int,
    atol: float,
    layer_types: Sequence[str] | None = None,
) -> dict[str, Any]:
    steps: list[dict[str, Any]] = []
    for step, (batch_bits, c1_bits) in enumerate(zip(batch.hidden_bits_by_step, c1.hidden_bits_by_step, strict=True)):
        rows: list[dict[str, Any]] = []
        for row in range(batch_bits.shape[0]):
            rows.append(
                {
                    "row": row,
                    "hidden_comparison": hidden_comparison(
                        batch_bits[row : row + 1],
                        c1_bits[row : row + 1],
                        atol=atol,
                    ),
                }
            )
        steps.append({"decode_step": step, "generated_index": step + 1, "rows": rows})
    token_mismatches = _token_mismatches(batch, c1)
    return {
        "layer_limit": int(layer_limit),
        **_layer_limit_metadata(layer_limit, layer_types),
        "hidden_passed": all(row["hidden_comparison"]["passed"] for step in steps for row in step["rows"]),
        "token_passed": not token_mismatches,
        "seed_tokens": {"batch": batch.seed_tokens, "c1": c1.seed_tokens},
        "generated_tokens": {"batch": batch.generated_tokens, "c1": c1.generated_tokens},
        "token_mismatches": token_mismatches,
        "steps": steps,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--fixture", default=DEFAULT_FIXTURE)
    parser.add_argument("--prompt-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--decode-tokens", type=int, default=16)
    parser.add_argument("--max-layers", type=int, default=8)
    parser.add_argument("--layer-limits", default=None, help="Comma/range list such as '1,4,8' or '1-8'; default all")
    parser.add_argument("--max-sequence-length", type=int, default=1024)
    parser.add_argument("--hidden-atol", type=float, default=1.0e-3)
    parser.add_argument("--compiler-version-file", type=Path, default=None)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Emit planned layer limits and commands without touching HIP")
    return parser


def run(args: argparse.Namespace, argv: Sequence[str] | None = None) -> dict[str, Any]:
    layer_limits = _parse_layer_limits(args.layer_limits, max_layers=args.max_layers)
    prompt_lengths: list[int] = []
    if args.dry_run:
        prompts = []
    else:
        prompts = _load_prompt_slices(Path(args.fixture), prompt_length=args.prompt_length, batch_size=args.batch_size)
        prompt_lengths = [len(prompt) for prompt in prompts]
        if args.max_sequence_length < max(prompt_lengths) + args.decode_tokens + 1:
            raise ValueError("max_sequence_length must cover prompt_length + decode_tokens + 1")
    payload: dict[str, Any] = {
        "schema": 1,
        "status": "planned" if args.dry_run else "running",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": "qwen35_paro_native_hidden_bisect",
        "command": _command(argv),
        "performance_claim": False,
        "workload": {
            "model": str(args.model),
            "fixture": str(args.fixture),
            "prompt_length": int(args.prompt_length),
            "prompt_lengths": prompt_lengths,
            "batch_size": int(args.batch_size),
            "decode_tokens": int(args.decode_tokens),
            "max_layers": int(args.max_layers),
            "layer_limits": layer_limits,
            "max_sequence_length": int(args.max_sequence_length),
            "kv_storage_dtype": "bf16",
            "native_compact_prefill": True,
            "native_caware_decode": bool(args.prompt_length + args.decode_tokens < 1024),
            "full_attention_decode_path": "batch_context" if args.prompt_length + args.decode_tokens < 1024 else "per_row_splitk_fallback",
        },
        "correctness": {
            "oracle": "hidden tensors and generated-token IDs vs independent c=1 resident sessions",
            "hidden_atol": float(args.hidden_atol),
            "passed": False,
        },
        "layer_summaries": [],
        "blockers": [],
    }
    if args.dry_run:
        payload["commands"] = [
            _command([*sys.argv[1:], "--layer-limits", str(limit)] if argv is None else [*argv, "--layer-limits", str(limit)])
            for limit in layer_limits
        ]
        if args.json is not None:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(json.dumps(payload, indent=2) + "\n")
        return payload

    os.environ.setdefault("HIPENGINE_QWEN35_EXPERIMENTAL_NATIVE_BATCH_DECODE", "1")
    runner = Qwen35ParoNextTokenRunner(args.model)
    layer_types = tuple(str(layer_type) for layer_type in getattr(runner.config, "layer_types", ()))
    compiler_version = _compiler_version(args.compiler_version_file)
    layer_summaries: list[dict[str, Any]] = []
    for layer_limit in layer_limits:
        batch = _run_batch_hidden(
            runner,
            prompts,
            layer_limit=layer_limit,
            decode_tokens=args.decode_tokens,
            max_sequence_length=args.max_sequence_length,
            compiler_version=compiler_version,
            require_cached_build=args.require_cached_build,
        )
        c1 = _run_c1_hidden(
            runner,
            prompts,
            layer_limit=layer_limit,
            decode_tokens=args.decode_tokens,
            max_sequence_length=args.max_sequence_length,
            compiler_version=compiler_version,
            require_cached_build=args.require_cached_build,
        )
        layer_summaries.append(
            _summarize_layer_limit(batch, c1, layer_limit=layer_limit, atol=args.hidden_atol, layer_types=layer_types)
        )

    hidden_mismatch = _first_hidden_mismatch(layer_summaries)
    token_mismatch = _first_token_mismatch(layer_summaries)
    passed = hidden_mismatch is None and token_mismatch is None
    payload["status"] = "eq_ok" if passed else "mismatch_found"
    payload["correctness"].update(
        {
            "passed": passed,
            "first_hidden_mismatch": hidden_mismatch,
            "first_token_mismatch": token_mismatch,
        }
    )
    payload["layer_summaries"] = layer_summaries
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    payload = run(args, argv)
    print(json.dumps(payload, indent=2))
    return 0 if payload["status"] in {"eq_ok", "mismatch_found", "planned"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
