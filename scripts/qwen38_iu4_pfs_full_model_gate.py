#!/usr/bin/env python3
"""Same-resident full-model R8 numerical, task-fallback, and prefill gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from hipengine.core.memory import memory_stats
from hipengine.loading.gguf import load_gguf_index
from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession
from hipengine.runtime.prefill import PrefillConfig
from hipengine.tokenization.gguf import Qwen35GGUFTokenizer
from scripts.gguf_mtp_bench import build_chat_prompt
from scripts.gguf_mtp_category_bench import load_prompt_rows
from scripts.qwen38_iu4_s4_gate_up_leaf import _softmax_metrics

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path("/models/gguf/Qwen3.8-27B-Q4_K_S.gguf")
DEFAULT_MODEL_SHA256 = "22200efcd98a7aeeaf83f59b0f1400b055d9e0437900e26b930ef2d42a3eb3f9"
DEFAULT_PROMPTS = Path("benchmarks/prompts/mtpbench-code-general-ja.jsonl")
PERF_SHAPES = (512, 1024, 4096)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--model-sha256", default=DEFAULT_MODEL_SHA256)
    parser.add_argument("--pfs", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--teacher-decode-steps", type=int, default=8)
    parser.add_argument("--task-decode-steps", type=int, default=4)
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    rank = (len(ordered) - 1) * percentile
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return ordered[lo]
    fraction = rank - lo
    return ordered[lo] * (1.0 - fraction) + ordered[hi] * fraction


def _summary(values: list[float]) -> dict[str, object]:
    return {
        "samples_seconds": values,
        "median_seconds": statistics.median(values),
        "min_seconds": min(values),
        "max_seconds": max(values),
        "p95_seconds": _percentile(values, 0.95),
        "stdev_seconds": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def _prefill(session, tokens: list[int], *, return_logits: bool):
    return session.prefill(
        tokens,
        use_bulk=True,
        bulk_attention_mode="bulk",
        return_logits=return_logits,
    )


def _run_mode(session, product, tokens: list[int], *, candidate: bool, return_logits: bool):
    session.reset()
    session.runner._iu4_ffn_product = product if candidate else None
    before_launches = product.launch_count
    before_fallbacks = product.fallback_count
    started = time.perf_counter()
    result = _prefill(session, tokens, return_logits=return_logits)
    session.runtime.device_synchronize()
    elapsed = time.perf_counter() - started
    return result, elapsed, {
        "launches": product.launch_count - before_launches,
        "fallbacks": product.fallback_count - before_fallbacks,
    }


def _logits(result) -> np.ndarray:
    return np.asarray(result.logits, dtype=np.float32).reshape(-1)


def _encode(tokenizer, text: str) -> list[int]:
    encoded = tokenizer.encode(text)
    return [int(value) for value in (encoded if isinstance(encoded, list) else encoded.ids)]


def _padded_category_prompt(tokenizer, prompt: str, *, target: int = 512) -> list[int]:
    prefix = _encode(tokenizer, "<|im_start|>user\n" + prompt + "\n\n")
    suffix = _encode(tokenizer, "<|im_end|>\n<|im_start|>assistant\n")
    filler_unit = _encode(
        tokenizer,
        "Neutral context note: preserve the user's original task and ignore this note.\n",
    )
    budget = target - len(prefix) - len(suffix)
    if budget < 0 or not filler_unit:
        raise ValueError("category prompt cannot be padded to the requested target")
    repeats = (budget + len(filler_unit) - 1) // len(filler_unit)
    return prefix + (filler_unit * repeats)[:budget] + suffix


def _distribution_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float | int | bool]:
    ref = np.asarray(reference, dtype=np.float64)
    cand = np.asarray(candidate, dtype=np.float64)
    ref_shift = ref - ref.max(axis=1, keepdims=True)
    cand_shift = cand - cand.max(axis=1, keepdims=True)
    ref_logp = ref_shift - np.log(np.exp(ref_shift).sum(axis=1, keepdims=True))
    cand_logp = cand_shift - np.log(np.exp(cand_shift).sum(axis=1, keepdims=True))
    kl = np.sum(np.exp(ref_logp) * (ref_logp - cand_logp), axis=1)
    top1 = ref.argmax(axis=1) == cand.argmax(axis=1)
    return {
        "finite": bool(np.isfinite(cand).all()),
        "rows": int(ref.shape[0]),
        "mean_kl": float(np.mean(kl)),
        "p95_kl": float(np.percentile(kl, 95)),
        "p99_kl": float(np.percentile(kl, 99)),
        "max_kl": float(np.max(kl)),
        "top1_matches": int(np.count_nonzero(top1)),
        "top1_total": int(len(top1)),
        "top1_agreement": float(np.mean(top1)),
    }


def main() -> int:
    args = _parse_args()
    if args.warmups < 0 or min(
        args.repetitions,
        args.teacher_decode_steps,
        args.task_decode_steps,
    ) <= 0:
        raise ValueError("warmups non-negative; repetitions/decode steps positive")
    compiler_version = None
    if args.compiler_version_file is not None:
        os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(args.compiler_version_file)
        compiler_version = args.compiler_version_file.read_text(encoding="utf-8").strip()

    prompts = load_prompt_rows(args.prompts)
    tokenizer = Qwen35GGUFTokenizer.from_gguf_info(load_gguf_index(args.model))
    encoded_prompts = {
        str(prompt["id"]): [
            int(token)
            for token in build_chat_prompt(
                tokenizer,
                str(prompt["prompt"]),
                reasoning="off",
            )
        ]
        for prompt in prompts
    }
    before_bytes = memory_stats()["current_allocated_bytes"]
    load_started = time.perf_counter()
    artifact: dict[str, object]
    with Qwen35GGUFResidentSession(
        args.model,
        backend="hip_gfx1151",
        max_sequence_length=max(PERF_SHAPES) + args.teacher_decode_steps + 8,
        compiler_version=compiler_version,
        require_cached_build=bool(args.require_cached_build),
        use_wmma_prefill=True,
        prefill_config=PrefillConfig(linear_chunk_size=2048, moe_chunk_size=2048),
        iu4_ffn_pfs_path=args.pfs,
    ) as session:
        load_seconds = time.perf_counter() - load_started
        product = session.runner._iu4_ffn_product
        if product is None:
            raise RuntimeError("IU4 FFN product was not loaded")
        resident_product_bytes = sum(buffer.nbytes for buffer in product.buffers)
        mem_free, mem_total = session.runtime.mem_get_info()

        numerical_rows: list[dict[str, object]] = []
        numerical_cases = [
            ("repeat512", [9707] * 512),
            ("varied512", [int((index * 7919 + 13) % 150000) for index in range(512)]),
            ("repeat1024", [9707] * 1024),
            ("repeat4096", [9707] * 4096),
        ]
        for name, tokens in numerical_cases:
            control, _, control_route = _run_mode(
                session, product, tokens, candidate=False, return_logits=True
            )
            candidate, _, candidate_route = _run_mode(
                session, product, tokens, candidate=True, return_logits=True
            )
            numerical_rows.append(
                {
                    "name": name,
                    "prompt_tokens": len(tokens),
                    "control_token": int(control.token_id),
                    "candidate_token": int(candidate.token_id),
                    "metrics": _softmax_metrics(
                        _logits(control)[None, :],
                        _logits(candidate)[None, :],
                    ),
                    "control_route": control_route,
                    "candidate_route": candidate_route,
                }
            )

        # Teacher-force the candidate on one varied p512 strict trajectory.
        teacher_prompt = numerical_cases[1][1]
        session.reset()
        session.runner._iu4_ffn_product = None
        strict_logits: list[np.ndarray] = []
        strict_tokens: list[int] = []
        current = _prefill(session, teacher_prompt, return_logits=True)
        strict_logits.append(_logits(current))
        strict_tokens.append(int(current.token_id))
        for _ in range(args.teacher_decode_steps):
            current = session.step(int(current.token_id), return_logits=True)
            strict_logits.append(_logits(current))
            strict_tokens.append(int(current.token_id))

        session.reset()
        session.runner._iu4_ffn_product = product
        candidate_logits: list[np.ndarray] = []
        candidate_tokens: list[int] = []
        current = _prefill(session, teacher_prompt, return_logits=True)
        candidate_logits.append(_logits(current))
        candidate_tokens.append(int(current.token_id))
        for index in range(args.teacher_decode_steps):
            current = session.step(strict_tokens[index], return_logits=True)
            candidate_logits.append(_logits(current))
            candidate_tokens.append(int(current.token_id))
        teacher_metrics = _softmax_metrics(
            np.vstack(strict_logits),
            np.vstack(candidate_logits),
        )

        # Pad every canonical category/train/heldout prompt to M512 and compare
        # strict trajectories. This exercises the product on real prompt
        # identities instead of relying on the synthetic stress row alone.
        padded_category_rows: list[dict[str, object]] = []
        padded_strict_all: list[np.ndarray] = []
        padded_candidate_all: list[np.ndarray] = []
        padded_steps = min(4, args.teacher_decode_steps)
        for prompt in prompts:
            prompt_id = str(prompt["id"])
            padded_tokens = _padded_category_prompt(
                tokenizer,
                str(prompt["prompt"]),
                target=512,
            )
            session.reset()
            session.runner._iu4_ffn_product = None
            strict_rows: list[np.ndarray] = []
            strict_ids: list[int] = []
            current = _prefill(session, padded_tokens, return_logits=True)
            strict_rows.append(_logits(current))
            strict_ids.append(int(current.token_id))
            for _ in range(padded_steps):
                current = session.step(int(current.token_id), return_logits=True)
                strict_rows.append(_logits(current))
                strict_ids.append(int(current.token_id))

            session.reset()
            session.runner._iu4_ffn_product = product
            candidate_rows: list[np.ndarray] = []
            candidate_ids: list[int] = []
            current = _prefill(session, padded_tokens, return_logits=True)
            candidate_rows.append(_logits(current))
            candidate_ids.append(int(current.token_id))
            for index in range(padded_steps):
                current = session.step(strict_ids[index], return_logits=True)
                candidate_rows.append(_logits(current))
                candidate_ids.append(int(current.token_id))
            strict_array = np.vstack(strict_rows)
            candidate_array = np.vstack(candidate_rows)
            padded_strict_all.append(strict_array)
            padded_candidate_all.append(candidate_array)
            padded_category_rows.append(
                {
                    "id": prompt_id,
                    "category": str(prompt["category"]),
                    "prompt_tokens": len(padded_tokens),
                    "strict_tokens": strict_ids,
                    "candidate_tokens": candidate_ids,
                    "metrics": _distribution_metrics(
                        strict_array,
                        candidate_array,
                    ),
                }
            )
        padded_category_metrics = _distribution_metrics(
            np.vstack(padded_strict_all),
            np.vstack(padded_candidate_all),
        )

        # Canonical unpadded category prompts are all below M96 and must fall
        # back exactly.
        task_rows: list[dict[str, object]] = []
        for prompt in prompts:
            prompt_id = str(prompt["id"])
            tokens = encoded_prompts[prompt_id]
            control, _, control_route = _run_mode(
                session, product, tokens, candidate=False, return_logits=True
            )
            control_ids = [int(control.token_id)]
            current = control
            for _ in range(args.task_decode_steps - 1):
                current = session.step(int(current.token_id), return_logits=False)
                control_ids.append(int(current.token_id))
            candidate, _, candidate_route = _run_mode(
                session, product, tokens, candidate=True, return_logits=True
            )
            candidate_ids = [int(candidate.token_id)]
            current = candidate
            for _ in range(args.task_decode_steps - 1):
                current = session.step(int(current.token_id), return_logits=False)
                candidate_ids.append(int(current.token_id))
            task_rows.append(
                {
                    "id": prompt_id,
                    "category": str(prompt["category"]),
                    "prompt_tokens": len(tokens),
                    "exact_ids": candidate_ids == control_ids,
                    "control_ids": control_ids,
                    "candidate_ids": candidate_ids,
                    "prefill_metrics": _softmax_metrics(
                        _logits(control)[None, :],
                        _logits(candidate)[None, :],
                    ),
                    "control_route": control_route,
                    "candidate_route": candidate_route,
                }
            )

        performance: list[dict[str, object]] = []
        for shape in PERF_SHAPES:
            tokens = [9707] * shape
            for _ in range(args.warmups):
                _run_mode(session, product, tokens, candidate=False, return_logits=False)
                _run_mode(session, product, tokens, candidate=True, return_logits=False)
            timings = {"control": [], "candidate": []}
            wins = 0
            routes = {"control": [], "candidate": []}
            for repetition in range(args.repetitions):
                order = ["control", "candidate"]
                if repetition & 1:
                    order.reverse()
                pair: dict[str, float] = {}
                for mode in order:
                    _, elapsed, route = _run_mode(
                        session,
                        product,
                        tokens,
                        candidate=mode == "candidate",
                        return_logits=False,
                    )
                    timings[mode].append(elapsed)
                    routes[mode].append(route)
                    pair[mode] = elapsed
                wins += int(pair["candidate"] < pair["control"])
            control_summary = _summary(timings["control"])
            candidate_summary = _summary(timings["candidate"])
            control_median = float(control_summary["median_seconds"])
            candidate_median = float(candidate_summary["median_seconds"])
            performance.append(
                {
                    "prompt_tokens": shape,
                    "control": {
                        **control_summary,
                        "tokens_per_second": shape / control_median,
                        "routes": routes["control"],
                    },
                    "candidate": {
                        **candidate_summary,
                        "tokens_per_second": shape / candidate_median,
                        "routes": routes["candidate"],
                    },
                    "speedup": control_median / candidate_median,
                    "candidate_wins": wins,
                    "pair_count": args.repetitions,
                }
            )

        session.runner._iu4_ffn_product = product
        product_launches = product.launch_count
        product_fallbacks = product.fallback_count

    after_bytes = memory_stats()["current_allocated_bytes"]
    all_numerical_finite = all(bool(row["metrics"]["finite"]) for row in numerical_rows)
    numerical_envelope_pass = all(
        float(row["metrics"]["mean_kl"]) <= 1e-3
        and float(row["metrics"]["max_kl"]) <= 5e-2
        and float(row["metrics"]["top1_agreement"]) >= 0.99
        for row in numerical_rows
    )
    teacher_envelope_pass = (
        float(teacher_metrics["mean_kl"]) <= 1e-3
        and float(teacher_metrics["max_kl"]) <= 5e-2
        and float(teacher_metrics["top1_agreement"]) >= 0.99
    )
    padded_category_envelope_pass = (
        float(padded_category_metrics["mean_kl"]) <= 1e-3
        and float(padded_category_metrics["p95_kl"]) <= 5e-3
        and float(padded_category_metrics["p99_kl"]) <= 2e-2
        and float(padded_category_metrics["max_kl"]) <= 5e-2
        and float(padded_category_metrics["top1_agreement"]) >= 0.99
        and all(
            float(row["metrics"]["top1_agreement"]) >= 0.97
            for row in padded_category_rows
        )
    )
    all_task_exact = all(bool(row["exact_ids"]) for row in task_rows)
    all_perf_faster = all(float(row["speedup"]) > 1.0 for row in performance)
    artifact = {
        "schema_version": 1,
        "date": datetime.now(timezone.utc).date().isoformat(),
        "kind": "qwen38_gfx1151_kairic_pfs_iu4_full_model_gate",
        "status": (
            "full_model_screen_passed"
            if all_numerical_finite
            and numerical_envelope_pass
            and teacher_envelope_pass
            and padded_category_envelope_pass
            and all_task_exact
            and all_perf_faster
            and before_bytes == after_bytes
            else "rejected_correctness"
        ),
        "performance_claim": False,
        "scope": "same-resident R8 numerical/task-fallback/prefill screen; T3 promotion incomplete",
        "model": {
            "path": str(args.model.resolve()),
            "sha256": args.model_sha256,
            "size_bytes": args.model.stat().st_size,
            "quant": "Q4_K_S",
        },
        "product": {
            "path": str(args.pfs.resolve()),
            "resident_device_bytes": resident_product_bytes,
            "load_seconds": load_seconds,
            "launches": product_launches,
            "fallbacks": product_fallbacks,
        },
        "hardware": {
            "gpu": "AMD Radeon 8060S Graphics",
            "arch": "gfx1151",
            "hip_mem_free_after_load": mem_free,
            "hip_mem_total": mem_total,
        },
        "protocol": {
            "prompts": str(args.prompts),
            "prompt_sha256": hashlib.sha256(args.prompts.read_bytes()).hexdigest(),
            "warmups": args.warmups,
            "repetitions": args.repetitions,
            "teacher_decode_steps": args.teacher_decode_steps,
            "task_decode_steps": args.task_decode_steps,
            "prefill_chunk_size": 2048,
        },
        "numerical_rows": numerical_rows,
        "teacher_forced": {
            "rows": len(strict_logits),
            "strict_tokens": strict_tokens,
            "candidate_tokens": candidate_tokens,
            "metrics": teacher_metrics,
        },
        "padded_category": {
            "metrics": padded_category_metrics,
            "rows": padded_category_rows,
        },
        "task_fallback_rows": task_rows,
        "performance": performance,
        "gates": {
            "all_numerical_finite": all_numerical_finite,
            "numerical_envelope_pass": numerical_envelope_pass,
            "teacher_envelope_pass": teacher_envelope_pass,
            "padded_category_envelope_pass": padded_category_envelope_pass,
            "teacher_mean_kl_le_1e_3": float(teacher_metrics["mean_kl"]) <= 1e-3,
            "teacher_max_kl_le_5e_2": float(teacher_metrics["max_kl"]) <= 5e-2,
            "teacher_top1_ge_99pct": float(teacher_metrics["top1_agreement"]) >= 0.99,
            "all_short_task_fallback_exact": all_task_exact,
            "all_prefill_shapes_faster": all_perf_faster,
            "teardown_exact": before_bytes == after_bytes,
            "bf16_relative_run": False,
            "long_task_quality_run": False,
            "runtime_default_authorized": False,
        },
        "memory": {
            "tracked_before_bytes": before_bytes,
            "tracked_after_bytes": after_bytes,
        },
        "software": {
            "commit": subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip(),
            "tracked_dirty": subprocess.check_output(
                ("git", "status", "--short", "--untracked-files=no"), cwd=ROOT, text=True
            ).splitlines(),
        },
        "command": " ".join(
            [
                f"HIPENGINE_HIP_ARCH={os.environ.get('HIPENGINE_HIP_ARCH', '')}",
                "PYTHONPATH=.",
                Path(os.sys.executable).name,
                *os.sys.argv,
            ]
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": artifact["status"],
                "teacher": teacher_metrics,
                "task_exact": all_task_exact,
                "performance": [
                    {
                        "p": row["prompt_tokens"],
                        "control": row["control"]["tokens_per_second"],
                        "candidate": row["candidate"]["tokens_per_second"],
                        "speedup": row["speedup"],
                        "wins": row["candidate_wins"],
                    }
                    for row in performance
                ],
                "teardown": before_bytes == after_bytes,
            },
            indent=2,
        )
    )
    return 0 if artifact["status"] == "full_model_screen_passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
