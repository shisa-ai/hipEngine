#!/usr/bin/env python3
"""Multi-category dense-teacher suite for the integrated Qwen3.8 DMS owner."""

from __future__ import annotations

import argparse
import hashlib
import json
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from hipengine.core.memory import memory_stats
from hipengine.runtime.qwen35_gguf_runner import (
    Qwen35GGUFFullStackRunner,
    Qwen35GGUFResidentSession,
)
from scripts.qwen38_dms_integrated_quality import _compare, _prompt, _summary

_REQUIRED_CATEGORIES = ("code", "general_en", "general_ja", "mixed_ja_en")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True, check=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
    )
    return {"commit": commit, "working_tree_clean": not dirty}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--prompt-tokens", type=int, required=True)
    parser.add_argument("--prompt-split", default="validation")
    parser.add_argument("--categories", default=",".join(_REQUIRED_CATEGORIES))
    parser.add_argument("--decode-steps", type=int, default=8)
    parser.add_argument("--modes", default="no_evict,sidecar")
    parser.add_argument("--backend", default="hip_gfx1151")
    parser.add_argument("--max-kl", type=float, default=0.05)
    parser.add_argument("--min-top1", type=float, default=0.9)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fail-on-fail", action="store_true")
    return parser


def _parse_csv(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in str(value).split(",") if part.strip())


def run(args: argparse.Namespace) -> dict[str, Any]:
    prompt_tokens = int(args.prompt_tokens)
    decode_steps = int(args.decode_steps)
    if prompt_tokens <= 0 or decode_steps <= 0:
        raise ValueError("prompt-tokens and decode-steps must be positive")
    categories = _parse_csv(args.categories)
    if not categories or len(set(categories)) != len(categories):
        raise ValueError("categories must be a non-empty unique list")
    if any(category not in _REQUIRED_CATEGORIES for category in categories):
        raise ValueError("categories contains an unsupported DMS category")
    modes = _parse_csv(args.modes)
    if not modes or any(mode not in {"no_evict", "sidecar"} for mode in modes):
        raise ValueError("modes must be a subset of no_evict,sidecar")

    prompts: dict[str, dict[str, Any]] = {}
    for category in categories:
        tokens, digest, sequence_ids = _prompt(
            args.data_manifest,
            prompt_tokens,
            split=str(args.prompt_split),
            category=category,
        )
        prompts[category] = {
            "tokens": tokens,
            "token_ids_sha256": digest,
            "sequence_ids": sequence_ids,
        }

    started = time.perf_counter()
    runner = Qwen35GGUFFullStackRunner(args.model, backend=str(args.backend))
    loaded_at = time.perf_counter()
    category_rows: dict[str, Any] = {}
    try:
        for category in categories:
            prompt = prompts[category]["tokens"]
            max_positions = prompt_tokens + decode_steps
            teacher_logits: list[np.ndarray] = []
            teacher_inputs: list[int] = []
            teacher_rows: list[dict[str, Any]] = []
            dense_started = time.perf_counter()
            with Qwen35GGUFResidentSession(
                args.model,
                backend=str(args.backend),
                shared_runner=runner,
                max_sequence_length=max_positions,
                use_wmma_prefill=True,
                use_gemv_decode=True,
            ) as dense:
                seed = dense.prefill(
                    prompt,
                    use_bulk=True,
                    bulk_attention_mode="bulk",
                    return_logits=True,
                )
                teacher_prefill = seed.logits.copy()
                current = int(seed.token_id)
                for step in range(decode_steps):
                    teacher_inputs.append(current)
                    step_started = time.perf_counter()
                    result = dense.step(current, return_logits=True)
                    teacher_logits.append(result.logits.copy())
                    teacher_rows.append(
                        {
                            "step": step,
                            "input_token": current,
                            "output_token": int(result.token_id),
                            "seconds": time.perf_counter() - step_started,
                        }
                    )
                    current = int(result.token_id)
                dense_memory = memory_stats()
            teacher = {
                "decode_rows": teacher_rows,
                "timing_seconds": time.perf_counter() - dense_started,
                "memory_before_close": dense_memory,
            }

            candidates: dict[str, Any] = {}
            for mode in modes:
                mode_started = time.perf_counter()
                with Qwen35GGUFResidentSession(
                    args.model,
                    backend=str(args.backend),
                    shared_runner=runner,
                    max_sequence_length=max_positions,
                    dms_metadata_path=args.metadata,
                    dms_max_new_tokens=decode_steps,
                    dms_decision_mode=mode,
                    use_wmma_prefill=True,
                    use_gemv_decode=True,
                ) as candidate:
                    seed = candidate.prefill(
                        prompt,
                        use_bulk=True,
                        bulk_attention_mode="bulk",
                        return_logits=True,
                    )
                    prefill_comparison = _compare(teacher_prefill, seed.logits)
                    rows: list[dict[str, Any]] = []
                    for step, input_token in enumerate(teacher_inputs):
                        step_started = time.perf_counter()
                        result = candidate.step(input_token, return_logits=True)
                        comparison = _compare(teacher_logits[step], result.logits)
                        comparison.update(
                            {
                                "category": category,
                                "step": step,
                                "input_token": input_token,
                                "teacher_output_token": teacher_rows[step]["output_token"],
                                "candidate_output_token": int(result.token_id),
                                "seconds": time.perf_counter() - step_started,
                            }
                        )
                        rows.append(comparison)
                    snapshot = candidate._dms_backend.observability_snapshot()
                    candidate_memory = memory_stats()
                candidates[mode] = {
                    "prefill_comparison": prefill_comparison,
                    "decode_rows": rows,
                    "summary": _summary(
                        rows,
                        max_kl=float(args.max_kl),
                        min_top1=float(args.min_top1),
                    ),
                    "dms": snapshot,
                    "timing_seconds": time.perf_counter() - mode_started,
                    "memory_before_close": candidate_memory,
                }
            category_rows[category] = {
                "prompt": {
                    "tokens": prompt_tokens,
                    "token_ids_sha256": prompts[category]["token_ids_sha256"],
                    "sequence_ids": prompts[category]["sequence_ids"],
                },
                "teacher": teacher,
                "candidates": candidates,
            }
    finally:
        runner.close()

    aggregate: dict[str, Any] = {}
    for mode in modes:
        rows = [
            row
            for category in categories
            for row in category_rows[category]["candidates"][mode]["decode_rows"]
        ]
        aggregate[mode] = _summary(
            rows,
            max_kl=float(args.max_kl),
            min_top1=float(args.min_top1),
        )
        aggregate[mode]["all_categories_passed"] = all(
            bool(category_rows[category]["candidates"][mode]["summary"]["passed"])
            for category in categories
        )
        aggregate[mode]["passed"] = bool(
            aggregate[mode]["passed"] and aggregate[mode]["all_categories_passed"]
        )
    passed = all(bool(row["passed"]) for row in aggregate.values())
    ended = time.perf_counter()
    result = {
        "schema_version": 1,
        "kind": "hipengine_qwen38_integrated_dms_dense_teacher_quality_suite",
        "status": "passed" if passed else "rejected_quality",
        "performance_claim": False,
        "host": socket.gethostname(),
        "backend": str(args.backend),
        "model": {"path": str(args.model.resolve()), "sha256": _sha256(args.model)},
        "metadata": {"path": str(args.metadata.resolve()), "sha256": _sha256(args.metadata)},
        "data_manifest": {
            "path": str(args.data_manifest.resolve()),
            "sha256": _sha256(args.data_manifest),
            "split": str(args.prompt_split),
        },
        "protocol": {
            "teacher": "dense BF16 KV exact-Q4 resident session",
            "trajectory": "strict dense-teacher input tokens",
            "candidate_owner": "integrated compact device route; dense KV released after prefill",
            "prompt_tokens": prompt_tokens,
            "decode_steps": decode_steps,
            "categories": list(categories),
            "modes": list(modes),
            "thresholds": {
                "max_kl": float(args.max_kl),
                "min_top1_agreement": float(args.min_top1),
            },
        },
        "categories": category_rows,
        "aggregate": aggregate,
        "timing": {
            "load_seconds": loaded_at - started,
            "total_seconds": ended - started,
        },
        "memory_after_close": memory_stats(),
        "provenance": _git(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    args = build_parser().parse_args()
    result = run(args)
    print(
        json.dumps(
            {
                "status": result["status"],
                "protocol": result["protocol"],
                "aggregate": result["aggregate"],
                "category_summaries": {
                    category: {
                        mode: row["summary"]
                        for mode, row in category_row["candidates"].items()
                    }
                    for category, category_row in result["categories"].items()
                },
                "capacity": {
                    category: {
                        mode: row["dms"]["capacity"]
                        for mode, row in category_row["candidates"].items()
                    }
                    for category, category_row in result["categories"].items()
                },
                "timing": result["timing"],
                "memory_after_close": result["memory_after_close"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if args.fail_on_fail and result["status"] != "passed":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
