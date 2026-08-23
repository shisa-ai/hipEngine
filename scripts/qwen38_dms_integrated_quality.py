#!/usr/bin/env python3
"""Dense-teacher quality gate for the integrated Qwen3.8 compact DMS owner."""

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


def _prompt(
    path: Path,
    count: int,
    *,
    split: str | None = None,
    category: str | None = None,
) -> tuple[list[int], str, list[str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    sequences = sorted(raw["sequences"], key=lambda row: str(row["sequence_id"]))
    if split is not None:
        sequences = [row for row in sequences if str(row.get("split")) == str(split)]
    if category is not None:
        sequences = [
            row for row in sequences if str(row.get("category")) == str(category)
        ]
    if not sequences:
        raise ValueError("data manifest filters select no prompt sequences")
    stream = [int(token) for row in sequences for token in row["token_ids"]]
    if not stream:
        raise ValueError("data manifest contains no selected prompt tokens")
    repeats = (int(count) + len(stream) - 1) // len(stream)
    tokens = (stream * repeats)[: int(count)]
    digest = hashlib.sha256(np.asarray(tokens, dtype=np.int64).tobytes()).hexdigest()
    return tokens, digest, [str(row["sequence_id"]) for row in sequences]


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    shifted = values - float(np.max(values))
    return shifted - float(np.log(np.exp(shifted).sum()))


def _compare(teacher: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    teacher_logp = _log_softmax(teacher)
    candidate_logp = _log_softmax(candidate)
    kl = float(np.sum(np.exp(teacher_logp) * (teacher_logp - candidate_logp)))
    teacher_top1 = int(np.argmax(teacher))
    candidate_top1 = int(np.argmax(candidate))
    return {
        "kl": kl,
        "top1_agrees": teacher_top1 == candidate_top1,
        "teacher_top1": teacher_top1,
        "candidate_top1": candidate_top1,
        "finite_candidate_logits": bool(np.isfinite(candidate).all()),
    }


def _summary(rows: list[dict[str, Any]], *, max_kl: float, min_top1: float) -> dict[str, Any]:
    kls = np.asarray([float(row["kl"]) for row in rows], dtype=np.float64)
    top1 = float(np.mean([bool(row["top1_agrees"]) for row in rows]))
    finite = all(bool(row["finite_candidate_logits"]) for row in rows)
    result = {
        "rows": len(rows),
        "mean_kl": float(np.mean(kls)),
        "p95_kl": float(np.quantile(kls, 0.95)),
        "p99_kl": float(np.quantile(kls, 0.99)),
        "max_kl": float(np.max(kls)),
        "top1_agreement": top1,
        "finite_logits": finite,
        "thresholds": {"max_kl": float(max_kl), "min_top1_agreement": float(min_top1)},
    }
    result["passed"] = bool(
        finite and result["max_kl"] <= float(max_kl) and top1 >= float(min_top1)
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--prompt-tokens", type=int, required=True)
    parser.add_argument("--prompt-split")
    parser.add_argument("--prompt-category")
    parser.add_argument("--decode-steps", type=int, default=8)
    parser.add_argument("--modes", default="no_evict,sidecar")
    parser.add_argument("--backend", default="hip_gfx1151")
    parser.add_argument("--max-kl", type=float, default=0.05)
    parser.add_argument("--min-top1", type=float, default=0.9)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fail-on-fail", action="store_true")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    if int(args.prompt_tokens) <= 0 or int(args.decode_steps) <= 0:
        raise ValueError("prompt-tokens and decode-steps must be positive")
    modes = tuple(part.strip() for part in str(args.modes).split(",") if part.strip())
    if not modes or any(mode not in {"no_evict", "sidecar"} for mode in modes):
        raise ValueError("modes must be a comma-separated subset of no_evict,sidecar")
    prompt, prompt_sha, prompt_sequence_ids = _prompt(
        args.data_manifest,
        args.prompt_tokens,
        split=args.prompt_split,
        category=args.prompt_category,
    )
    max_positions = int(args.prompt_tokens) + int(args.decode_steps)
    started = time.perf_counter()
    runner = Qwen35GGUFFullStackRunner(args.model, backend=str(args.backend))
    loaded_at = time.perf_counter()
    teacher_logits: list[np.ndarray] = []
    teacher_rows: list[dict[str, Any]] = []
    teacher_inputs: list[int] = []
    candidates: dict[str, Any] = {}
    try:
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
            teacher_prefill_logits = seed.logits.copy()
            current = int(seed.token_id)
            for step in range(int(args.decode_steps)):
                teacher_inputs.append(current)
                step_started = time.perf_counter()
                row = dense.step(current, return_logits=True)
                teacher_logits.append(row.logits.copy())
                teacher_rows.append(
                    {
                        "step": step,
                        "input_token": current,
                        "output_token": int(row.token_id),
                        "seconds": time.perf_counter() - step_started,
                    }
                )
                current = int(row.token_id)
            dense_memory = memory_stats()
        dense_seconds = time.perf_counter() - dense_started

        for mode in modes:
            mode_started = time.perf_counter()
            with Qwen35GGUFResidentSession(
                args.model,
                backend=str(args.backend),
                shared_runner=runner,
                max_sequence_length=max_positions,
                dms_metadata_path=args.metadata,
                dms_max_new_tokens=int(args.decode_steps),
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
                prefill_comparison = _compare(teacher_prefill_logits, seed.logits)
                rows: list[dict[str, Any]] = []
                for step, input_token in enumerate(teacher_inputs):
                    step_started = time.perf_counter()
                    row = candidate.step(input_token, return_logits=True)
                    comparison = _compare(teacher_logits[step], row.logits)
                    comparison.update(
                        {
                            "step": step,
                            "input_token": input_token,
                            "teacher_output_token": teacher_rows[step]["output_token"],
                            "candidate_output_token": int(row.token_id),
                            "seconds": time.perf_counter() - step_started,
                        }
                    )
                    rows.append(comparison)
                snapshot = candidate._dms_backend.observability_snapshot()
                candidate_memory = memory_stats()
            candidates[mode] = {
                "decision_mode": mode,
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
    finally:
        runner.close()
    ended = time.perf_counter()
    all_passed = all(bool(row["summary"]["passed"]) for row in candidates.values())
    result = {
        "schema_version": 1,
        "kind": "hipengine_qwen38_integrated_dms_dense_teacher_quality",
        "status": "passed" if all_passed else "rejected_quality",
        "performance_claim": False,
        "host": socket.gethostname(),
        "backend": str(args.backend),
        "model": {"path": str(args.model.resolve()), "sha256": _sha256(args.model)},
        "metadata": {"path": str(args.metadata.resolve()), "sha256": _sha256(args.metadata)},
        "prompt": {
            "tokens": int(args.prompt_tokens),
            "token_ids_sha256": prompt_sha,
            "data_manifest_sha256": _sha256(args.data_manifest),
            "split_filter": args.prompt_split,
            "category_filter": args.prompt_category,
            "sequence_ids": prompt_sequence_ids,
            "source_note": "deterministic selected corpus stream; scope depends on recorded manifest split/category",
        },
        "protocol": {
            "teacher": "dense BF16 KV exact-Q4 resident session",
            "trajectory": "strict dense-teacher input tokens for every candidate step",
            "candidate_owner": "integrated compact device route with dense KV released after prefill",
            "modes": list(modes),
            "decode_steps": int(args.decode_steps),
        },
        "teacher": {
            "rows": teacher_rows,
            "timing_seconds": dense_seconds,
            "memory_before_close": dense_memory,
        },
        "candidates": candidates,
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
                "prompt_tokens": result["prompt"]["tokens"],
                "teacher_timing_seconds": result["teacher"]["timing_seconds"],
                "candidates": {
                    mode: {
                        "summary": row["summary"],
                        "capacity": row["dms"]["capacity"],
                        "timing_seconds": row["timing_seconds"],
                    }
                    for mode, row in result["candidates"].items()
                },
                "memory_after_close": result["memory_after_close"],
                "total_seconds": result["timing"]["total_seconds"],
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
