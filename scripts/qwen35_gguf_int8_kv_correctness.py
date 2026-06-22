#!/usr/bin/env python3
# ruff: noqa: E402
"""GGUF BF16-vs-INT8 KV correctness gate for Qwen3.6-35B-A3B.

This compares the resident GGUF BF16 full-attention KV path with the explicit
``int8_per_token_head`` path on the same prompt and teacher-forced token
trajectory. Short INT8 sessions may route decode through the BF16 mirror. Long
INT8 sessions use a hybrid BF16-prefix/INT8-suffix full-attention layout by
default; set ``HIPENGINE_GGUF_INT8_KV_ALLOW_UNVERIFIED_LONG=1`` together with a
large ``--max-sequence-length`` (for example ``131202``) only when reproducing
the blocked pure INT8-only capacity/quality diagnostic.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.benchmark.correctness import evaluate_logits
from hipengine.core.dtype import DType
from hipengine.core.memory import memory_stats, reset_memory_stats
from hipengine.kvcache import FixedPagedKVPolicy
from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession

DEFAULT_MODEL = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")


@dataclass
class SequenceRun:
    kv_storage: str
    generated_token_ids: list[int]
    logits: np.ndarray
    token_ids_for_next_step: list[int]
    finite_logits: bool
    elapsed_seconds: float
    memory: dict[str, Any]
    mirror_cache_count: int
    bf16_primary_layer_count: int
    int8_layer_count: int
    effective_kv_scale_dtype: str
    max_sequence_length: int


def _parse_count(text: str) -> int:
    value = text.strip().lower()
    if value.endswith("k"):
        return int(float(value[:-1]) * 1024)
    return int(value)


def _parse_prompt_lengths(value: str) -> list[int]:
    lengths = [_parse_count(item) for item in value.split(",") if item.strip()]
    if not lengths or any(length <= 0 for length in lengths):
        raise argparse.ArgumentTypeError("prompt lengths must be a comma-separated list of positive integers")
    return lengths


def _read_compiler_version(path: Path | None) -> str | None:
    if path is None:
        return None
    return path.read_text().strip()


def _command(args: argparse.Namespace) -> str:
    parts = [
        "python3 scripts/qwen35_gguf_int8_kv_correctness.py",
        f"--model {args.model}",
        f"--quant {args.quant}",
        f"--prompt-lengths {args.prompt_lengths_raw}",
        f"--decode-steps {args.decode_steps}",
        f"--token-id {args.token_id}",
        f"--max-sequence-length {args.max_sequence_length}",
        f"--kv-scale-dtype {args.kv_scale_dtype}",
        f"--kl-threshold {args.kl_threshold}",
        f"--top1-threshold {args.top1_threshold}",
    ]
    if args.compiler_version_file is not None:
        parts.append(f"--compiler-version-file {args.compiler_version_file}")
    if args.require_cached_build:
        parts.append("--require-cached-build")
    if args.require_no_bf16_mirror:
        parts.append("--require-no-bf16-mirror")
    if args.json is not None:
        parts.append(f"--json {args.json}")
    return " ".join(parts)


def _kv_layout_counts(session: Qwen35GGUFResidentSession) -> tuple[int, int, int]:
    scratch = session.scratch
    if scratch is None:
        return 0, 0, 0
    keys = getattr(scratch, "full_bf16_mirror_key_caches", ())
    vals = getattr(scratch, "full_bf16_mirror_value_caches", ())
    mirror_count = sum(1 for key, value in zip(keys, vals, strict=False) if key is not None and value is not None)
    primary_keys = getattr(scratch, "full_key_caches", ())
    metadata = getattr(scratch, "full_kv_scale_metadata", ())
    bf16_primary = 0
    int8_layers = 0
    for key_cache, scale_metadata in zip(primary_keys, metadata, strict=False):
        if key_cache is None:
            continue
        if scale_metadata is None:
            bf16_primary += 1
        else:
            int8_layers += 1
    return int(mirror_count), int(bf16_primary), int(int8_layers)


def _run_sequence(
    *,
    model: Path,
    compiler_version: str | None,
    require_cached_build: bool,
    prompt_tokens: list[int],
    decode_steps: int,
    max_sequence_length: int,
    kv_storage: str,
    kv_scale_dtype: str,
    teacher_tokens: list[int] | None = None,
) -> SequenceRun:
    reset_memory_stats()
    logits_rows: list[np.ndarray] = []
    generated_token_ids: list[int] = []
    token_ids_for_next_step: list[int] = []
    finite_logits = True
    policy = FixedPagedKVPolicy(block_size=256, storage_dtype=DType.parse(kv_storage))
    start = time.perf_counter()
    with Qwen35GGUFResidentSession(
        model,
        compiler_version=compiler_version,
        require_cached_build=require_cached_build,
        max_sequence_length=max_sequence_length,
        use_wmma_prefill=True,
        use_gemv_decode=True,
        kv_policy=policy,
        kv_scale_dtype=DType.parse(kv_scale_dtype),
        kv_scale_granularity="per_token_head",
    ) as session:
        first = session.prefill(
            prompt_tokens,
            use_bulk=True,
            bulk_attention_mode="bulk",
            return_logits=True,
        )
        generated_token_ids.append(int(first.token_id))
        token_ids_for_next_step.append(int(first.token_id))
        logits_rows.append(_checked_logits(first.logits, "prefill"))
        for step_index in range(int(decode_steps)):
            if teacher_tokens is None:
                next_input = int(token_ids_for_next_step[-1])
            else:
                if step_index >= len(teacher_tokens):
                    raise ValueError("not enough teacher tokens for requested decode steps")
                next_input = int(teacher_tokens[step_index])
            current = session.step(next_input, return_logits=True)
            generated_token_ids.append(int(current.token_id))
            token_ids_for_next_step.append(int(current.token_id))
            logits_rows.append(_checked_logits(current.logits, f"decode[{step_index}]"))
        mirror_count, bf16_primary_layer_count, int8_layer_count = _kv_layout_counts(session)
        effective_kv_scale_dtype = session.kv_scale_dtype.value
        stats = memory_stats()
    elapsed = time.perf_counter() - start
    logits = np.vstack(logits_rows).astype(np.float32, copy=False)
    finite_logits = bool(finite_logits and np.all(np.isfinite(logits)))
    gc.collect()
    return SequenceRun(
        kv_storage=kv_storage,
        generated_token_ids=generated_token_ids,
        logits=logits,
        token_ids_for_next_step=token_ids_for_next_step,
        finite_logits=finite_logits,
        elapsed_seconds=float(elapsed),
        memory=stats,
        mirror_cache_count=int(mirror_count),
        bf16_primary_layer_count=int(bf16_primary_layer_count),
        int8_layer_count=int(int8_layer_count),
        effective_kv_scale_dtype=str(effective_kv_scale_dtype),
        max_sequence_length=int(max_sequence_length),
    )


def _checked_logits(logits: np.ndarray, label: str) -> np.ndarray:
    arr = np.asarray(logits, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        raise FloatingPointError(f"{label} logits are empty")
    if not np.all(np.isfinite(arr)):
        raise FloatingPointError(f"{label} logits contain NaN/Inf")
    return arr.copy()


def _summarise_pair(
    *,
    prompt_length: int,
    reference: SequenceRun,
    candidate: SequenceRun,
    kl_threshold: float,
    top1_threshold: float,
    require_no_bf16_mirror: bool,
) -> dict[str, Any]:
    metrics = evaluate_logits(
        reference.logits,
        candidate.logits,
        kl_threshold=float(kl_threshold),
        top1_threshold=float(top1_threshold),
    )
    ref_top1 = np.argmax(reference.logits, axis=-1).astype(np.int64)
    cand_top1 = np.argmax(candidate.logits, axis=-1).astype(np.int64)
    mismatches = np.nonzero(ref_top1 != cand_top1)[0]
    errors: list[str] = []
    if not reference.finite_logits:
        errors.append("BF16 reference logits are not finite")
    if not candidate.finite_logits:
        errors.append("INT8 candidate logits are not finite")
    if require_no_bf16_mirror and candidate.mirror_cache_count != 0:
        errors.append(f"candidate allocated {candidate.mirror_cache_count} BF16 mirror caches")
    if not metrics.passed:
        errors.append(
            f"KL/top1 gate failed: kl_mean={metrics.kl_mean:.6g} <= {kl_threshold}, "
            f"top1={metrics.top1_agreement:.6g} >= {top1_threshold}"
        )
    return {
        "prompt_length": int(prompt_length),
        "passed": not errors,
        "errors": errors,
        "metrics": {
            "kl_mean": float(metrics.kl_mean),
            "kl_max": float(metrics.kl_max),
            "top1_agreement": float(metrics.top1_agreement),
            "passed": bool(metrics.passed),
            "first_top1_mismatch_index": None if mismatches.size == 0 else int(mismatches[0]),
            "top1_mismatch_count": int(mismatches.size),
        },
        "reference": _run_to_json(reference),
        "candidate": _run_to_json(candidate),
    }


def _run_to_json(run: SequenceRun) -> dict[str, Any]:
    return {
        "kv_storage": run.kv_storage,
        "generated_token_ids": run.generated_token_ids,
        "top1_token_ids": [int(x) for x in np.argmax(run.logits, axis=-1).tolist()],
        "finite_logits": bool(run.finite_logits),
        "elapsed_seconds": float(run.elapsed_seconds),
        "mirror_cache_count": int(run.mirror_cache_count),
        "bf16_primary_layer_count": int(run.bf16_primary_layer_count),
        "int8_layer_count": int(run.int8_layer_count),
        "effective_kv_scale_dtype": str(run.effective_kv_scale_dtype),
        "max_sequence_length": int(run.max_sequence_length),
        "tracked_peak_allocated_gib": float(run.memory.get("peak_allocated_bytes", 0)) / (1024**3),
        "tracked_current_allocated_gib": float(run.memory.get("current_allocated_bytes", 0)) / (1024**3),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    compiler_version = _read_compiler_version(args.compiler_version_file)
    prompt_lengths = _parse_prompt_lengths(args.prompt_lengths_raw)
    rows: list[dict[str, Any]] = []
    blocked: list[str] = []
    failures: list[str] = []
    for prompt_length in prompt_lengths:
        max_sequence_length = int(args.max_sequence_length or (prompt_length + int(args.decode_steps) + 2))
        prompt_tokens = [int(args.token_id)] * int(prompt_length)
        try:
            print(
                f"[gguf-int8-kv] prompt={prompt_length} max_sequence={max_sequence_length}: running BF16 reference",
                file=sys.stderr,
                flush=True,
            )
            reference = _run_sequence(
                model=args.model,
                compiler_version=compiler_version,
                require_cached_build=bool(args.require_cached_build),
                prompt_tokens=prompt_tokens,
                decode_steps=int(args.decode_steps),
                max_sequence_length=max_sequence_length,
                kv_storage="bf16",
                kv_scale_dtype=args.kv_scale_dtype,
            )
            teacher_tokens = reference.token_ids_for_next_step[: int(args.decode_steps)]
            print(
                f"[gguf-int8-kv] prompt={prompt_length}: running INT8 candidate",
                file=sys.stderr,
                flush=True,
            )
            candidate = _run_sequence(
                model=args.model,
                compiler_version=compiler_version,
                require_cached_build=bool(args.require_cached_build),
                prompt_tokens=prompt_tokens,
                decode_steps=int(args.decode_steps),
                max_sequence_length=max_sequence_length,
                kv_storage="int8_per_token_head",
                kv_scale_dtype=args.kv_scale_dtype,
                teacher_tokens=teacher_tokens,
            )
            row = _summarise_pair(
                prompt_length=prompt_length,
                reference=reference,
                candidate=candidate,
                kl_threshold=float(args.kl_threshold),
                top1_threshold=float(args.top1_threshold),
                require_no_bf16_mirror=bool(args.require_no_bf16_mirror),
            )
            rows.append(row)
            failures.extend(f"prompt={prompt_length}: {err}" for err in row["errors"])
            print(
                f"[gguf-int8-kv] prompt={prompt_length}: passed={row['passed']} "
                f"kl_mean={row['metrics']['kl_mean']:.6g} top1={row['metrics']['top1_agreement']:.6g}",
                file=sys.stderr,
                flush=True,
            )
        except Exception as exc:
            blocked.append(f"prompt={prompt_length}: {type(exc).__name__}: {exc}")
            rows.append({"prompt_length": int(prompt_length), "passed": False, "blocked_reason": blocked[-1]})
    passed = not blocked and not failures
    status = "accepted" if passed else ("blocked" if blocked else "rejected_correctness")
    return {
        "schema": 1,
        "mode": "qwen35_gguf_int8_kv_correctness",
        "status": status,
        "passed": bool(passed),
        "performance_claim": False,
        "command": _command(args),
        "model": str(args.model),
        "quant": args.quant,
        "thresholds": {
            "kl_mean_max": float(args.kl_threshold),
            "top1_agreement_min": float(args.top1_threshold),
            "require_no_bf16_mirror": bool(args.require_no_bf16_mirror),
        },
        "environment": {
            "hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES"),
            "hipengine_hip_arch": os.environ.get("HIPENGINE_HIP_ARCH"),
            "rocm_path": os.environ.get("ROCM_PATH"),
            "compiler_version_first_line": None if compiler_version is None else compiler_version.splitlines()[0],
        },
        "rows": rows,
        "blocked_reasons": blocked,
        "correctness_failures": failures,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--quant", default="gguf_q4_k_m")
    parser.add_argument("--prompt-lengths", dest="prompt_lengths_raw", default="512")
    parser.add_argument("--decode-steps", type=int, default=1)
    parser.add_argument("--token-id", type=int, default=9707)
    parser.add_argument("--max-sequence-length", type=int, default=0)
    parser.add_argument("--kv-scale-dtype", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--kl-threshold", type=float, default=0.05)
    parser.add_argument("--top1-threshold", type=float, default=0.90)
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--require-no-bf16-mirror", action="store_true")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    payload = run(args)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text)
    print(text)
    return 0 if payload.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
