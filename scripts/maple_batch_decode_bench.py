#!/usr/bin/env python3
"""Recertify Maple M6/D1 fixed-capacity batch-decode throughput.

This low-level benchmark measures ``MapleBatchRunner.batch_step`` directly at
c=2/4/8, gates every measured width against c1 serial trajectories, exercises a
sparse category-derived seed group, and records tracked lifecycle. It isolates
resident decode from the public scheduler and prompt-prefill wall measured by
``maple_public_batch_bench.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import statistics
import subprocess
import sys
import time
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.core.memory import memory_stats, reset_memory_stats  # noqa: E402
from hipengine.loading.maple import load_maple_checkpoint  # noqa: E402
from hipengine.runtime.maple import MapleBatchRunner, MapleRunner  # noqa: E402
from hipengine.tokenization.maple import MapleTokenizer  # noqa: E402

DEFAULT_MODEL = "deepgrove/maple-preview-2bit-mlx"
DEFAULT_SUITE = REPO_ROOT / "benchmarks/prompts/mtpbench-code-general-ja.jsonl"
DEFAULT_HELDOUT = REPO_ROOT / "benchmarks/prompts/gdn-prefill-category-heldouts.jsonl"
REQUIRED_CATEGORIES = ("code", "general_en", "general_ja", "mixed_ja_en")
PINNED_REVISION = "361db5da5e74ff6fcdd852d478e1f266ce11013a"


def _capture(command: list[str], *, timeout: float = 30.0) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    return {
        "command": shlex.join(command),
        "returncode": int(completed.returncode),
        "output": (completed.stdout + completed.stderr).strip(),
    }


def _load_prompts(path: Path, *, heldout: bool) -> list[dict[str, Any]]:
    prompts: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        users = [
            message["content"]
            for message in row.get("messages", [])
            if message.get("role") == "user"
        ]
        if len(users) != 1:
            raise ValueError(f"{path}: prompt {row.get('id')!r} must have one user message")
        category = str(row["category"])
        if category not in REQUIRED_CATEGORIES:
            raise ValueError(f"{path}: unsupported category {category!r}")
        prompts.append(
            {
                "id": str(row["id"]),
                "category": category,
                "text": str(users[0]),
                "heldout": bool(heldout),
            }
        )
    return prompts


def _tokenizer(checkpoint) -> MapleTokenizer:
    spec = checkpoint.spec
    return MapleTokenizer.from_model_path(
        checkpoint.index.model_path,
        model_vocab_size=spec.vocab_size,
        eos_token_id=spec.eos_token_id,
        bos_token_id=spec.bos_token_id,
    )


def _serial_trajectories(
    checkpoint,
    seeds: list[int],
    *,
    steps: int,
    backend: str,
) -> list[list[int]]:
    runner = MapleRunner.load(
        checkpoint,
        backend=backend,
        max_context=steps + 2,
    )
    trajectories: list[list[int]] = []
    try:
        for seed in seeds:
            runner.reset()
            output = [runner.step(seed).token_id]
            for _ in range(steps - 1):
                output.append(runner.step(output[-1]).token_id)
            trajectories.append(output)
    finally:
        runner.close()
    return trajectories


def _run_batch(
    runner: MapleBatchRunner,
    seeds: list[int],
    *,
    steps: int,
) -> tuple[list[list[int]], float]:
    if not seeds or len(seeds) > runner.batch_size:
        raise ValueError("seed count must be within the fixed batch capacity")
    active_mask = [request < len(seeds) for request in range(runner.batch_size)]
    inputs = [0] * runner.batch_size
    outputs: list[list[int]] = [[] for _ in seeds]
    for request, seed in enumerate(seeds):
        runner.reset_request(request)
        inputs[request] = int(seed)

    started = time.perf_counter()
    for _ in range(steps):
        next_tokens = runner.batch_step(inputs, active_mask=active_mask)
        for request in range(len(seeds)):
            token = int(next_tokens[request])
            outputs[request].append(token)
            inputs[request] = token
    elapsed = time.perf_counter() - started
    return outputs, elapsed


def _natural_seed_gate(
    checkpoint,
    tokenizer: MapleTokenizer,
    prompts: list[dict[str, Any]],
    *,
    backend: str,
    steps: int,
) -> dict[str, Any]:
    rows = []
    seeds = []
    for prompt in prompts:
        content_tokens = tokenizer.encode(prompt["text"])
        if not content_tokens:
            raise ValueError(f"prompt {prompt['id']} tokenized to empty content")
        seed = int(content_tokens[len(content_tokens) // 2])
        rows.append({**prompt, "seed": seed})
        seeds.append(seed)
    serial = _serial_trajectories(
        checkpoint,
        seeds,
        steps=steps,
        backend=backend,
    )
    batch_outputs: list[list[int]] = []
    lifecycle_rows = []
    for offset in range(0, len(seeds), 8):
        group = seeds[offset : offset + 8]
        reset_memory_stats()
        runner = MapleBatchRunner.load(
            checkpoint,
            backend=backend,
            batch_size=8,
            per_capacity=steps + 2,
        )
        try:
            output, _ = _run_batch(runner, group, steps=steps)
            batch_outputs.extend(output)
        finally:
            runner.close()
        after = memory_stats()
        lifecycle_rows.append(after)
        if after["current_allocated_bytes"] != 0 or after["active_allocations"] != 0:
            raise RuntimeError("natural-seed batch gate leaked tracked allocations")
    matches = [batch == reference for batch, reference in zip(batch_outputs, serial)]
    category_counts = Counter(row["category"] for row in rows)
    heldout_counts = Counter(row["category"] for row in rows if row["heldout"])
    passed = (
        len(rows) == 18
        and len(batch_outputs) == len(serial)
        and all(matches)
        and all(category_counts[category] > 0 for category in REQUIRED_CATEGORIES)
        and all(heldout_counts[category] > 0 for category in REQUIRED_CATEGORIES)
    )
    for row, match, batch, reference in zip(rows, matches, batch_outputs, serial):
        row["trajectory_equal"] = match
        row["batch_tokens"] = batch
        row["serial_tokens"] = reference
    return {
        "passed": passed,
        "prompt_count": len(rows),
        "decode_steps": steps,
        "trajectory_matches": sum(matches),
        "category_counts": dict(sorted(category_counts.items())),
        "heldout_category_counts": dict(sorted(heldout_counts.items())),
        "sparse_final_group_size": len(seeds) % 8 or 8,
        "rows": rows,
        "lifecycle_rows": lifecycle_rows,
        "scope": "natural-prompt-derived empty-context seeds; not batched prompt-prefill E2E",
    }


def _benchmark_widths(
    checkpoint,
    *,
    backend: str,
    steps: int,
    repetitions: int,
    warmup_steps: int,
) -> list[dict[str, Any]]:
    benchmark_seeds = [9_000 + index for index in range(8)]
    serial = _serial_trajectories(
        checkpoint,
        benchmark_seeds,
        steps=steps,
        backend=backend,
    )
    rows = []
    for concurrency in (2, 4, 8):
        reset_memory_stats()
        runner = MapleBatchRunner.load(
            checkpoint,
            backend=backend,
            batch_size=concurrency,
            per_capacity=steps + 2,
        )
        resident = memory_stats()
        samples = []
        try:
            runner.reset()
            _run_batch(
                runner,
                benchmark_seeds[:concurrency],
                steps=warmup_steps,
            )
            for repetition in range(repetitions):
                runner.reset()
                outputs, elapsed = _run_batch(
                    runner,
                    benchmark_seeds[:concurrency],
                    steps=steps,
                )
                matches = [
                    output == reference
                    for output, reference in zip(outputs, serial[:concurrency])
                ]
                samples.append(
                    {
                        "repetition": repetition,
                        "elapsed_seconds": elapsed,
                        "total_tokens": concurrency * steps,
                        "aggregate_tokens_per_second": concurrency * steps / elapsed,
                        "trajectory_matches": sum(matches),
                        "all_trajectories_equal": all(matches),
                    }
                )
        finally:
            runner.close()
        after_close = memory_stats()
        lifecycle_passed = (
            after_close["current_allocated_bytes"] == 0
            and after_close["active_allocations"] == 0
        )
        row_passed = all(sample["all_trajectories_equal"] for sample in samples)
        row_passed = row_passed and lifecycle_passed
        throughputs = [sample["aggregate_tokens_per_second"] for sample in samples]
        rows.append(
            {
                "concurrency": concurrency,
                "tokens_per_request": steps,
                "total_tokens_per_sample": concurrency * steps,
                "warmup_steps": warmup_steps,
                "repetitions": repetitions,
                "median_aggregate_tokens_per_second": statistics.median(throughputs),
                "min_aggregate_tokens_per_second": min(throughputs),
                "max_aggregate_tokens_per_second": max(throughputs),
                "samples": samples,
                "correctness_passed": row_passed,
                "resident_tracked": resident,
                "after_close": after_close,
                "lifecycle_passed": lifecycle_passed,
            }
        )
    return rows


def _git_context() -> dict[str, Any]:
    head = _capture(["git", "rev-parse", "HEAD"])
    status = _capture(["git", "status", "--short", "--untracked-files=no"])
    return {
        "head": head["output"],
        "tracked_status": status["output"],
        "tracked_clean": status["returncode"] == 0 and not status["output"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--backend", default="hip_gfx1151")
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--heldout", type=Path, default=DEFAULT_HELDOUT)
    parser.add_argument("--steps", type=int, default=64, help="tokens per request")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmup-steps", type=int, default=8)
    parser.add_argument("--natural-gate-steps", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if min(args.steps, args.repetitions, args.warmup_steps, args.natural_gate_steps) <= 0:
        raise ValueError("step/repetition counts must be positive")

    git = _git_context()
    checkpoint = load_maple_checkpoint(args.model)
    tokenizer = _tokenizer(checkpoint)
    prompts = _load_prompts(args.suite, heldout=False) + _load_prompts(
        args.heldout, heldout=True
    )
    natural_gate = _natural_seed_gate(
        checkpoint,
        tokenizer,
        prompts,
        backend=args.backend,
        steps=args.natural_gate_steps,
    )
    rows = _benchmark_widths(
        checkpoint,
        backend=args.backend,
        steps=args.steps,
        repetitions=args.repetitions,
        warmup_steps=args.warmup_steps,
    )
    width_gate = all(row["correctness_passed"] for row in rows)
    rocminfo = _capture(["bash", "-lc", "rocminfo | grep -E 'Name:|Marketing Name:|gfx' | head -8"])
    rocm_smi = _capture(["rocm-smi", "--showmeminfo", "vram", "--showuse", "--showtemp"])
    hipcc = _capture(["hipcc", "--version"])
    status = "accepted" if natural_gate["passed"] and width_gate and git["tracked_clean"] else "rejected"
    artifact = {
        "schema_version": 1,
        "date": date.today().isoformat(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_type": "maple_m6_batch_decode_recertification",
        "status": status,
        "performance_claim": status == "accepted",
        "claim_scope": "fixed-capacity MapleBatchRunner decode helper; excludes public scheduler and prompt-prefill E2E",
        "model": {
            "id": args.model,
            "revision": PINNED_REVISION,
            "resolved_path": str(Path(checkpoint.index.model_path).resolve()),
            "quant": "maple_ternary2",
            "exact_weight_bytes": checkpoint.validation.exact_weight_bytes,
        },
        "hardware": {
            "gpu": "AMD Radeon 8060S Graphics",
            "architecture": "gfx1151",
            "host": platform.node(),
            "rocminfo": rocminfo,
            "rocm_smi": rocm_smi,
            "hipcc": hipcc,
        },
        "software": {"python": platform.python_version(), "git": git},
        "protocol": {
            "command": shlex.join([sys.executable, *sys.argv]),
            "environment": {
                "GPU_MAX_HW_QUEUES": os.environ.get("GPU_MAX_HW_QUEUES"),
                "HIPENGINE_HIP_ARCH": os.environ.get("HIPENGINE_HIP_ARCH"),
            },
            "backend": args.backend,
            "suite": str(args.suite),
            "heldout": str(args.heldout),
            "tokens_per_request": args.steps,
            "repetitions": args.repetitions,
            "warmup_steps": args.warmup_steps,
            "natural_gate_steps": args.natural_gate_steps,
            "concurrencies": [2, 4, 8],
            "timing_scope": "direct resident MapleBatchRunner.batch_step loop; excludes model load, public scheduling, and prompt prefill",
        },
        "correctness": {
            "all_widths_passed": width_gate,
            "natural_seed_gate": natural_gate,
        },
        "performance": {"rows": rows},
        "notes": [
            "Every c=2/4/8 timing sample is compared with all corresponding serial c1 trajectories.",
            "The 18-prompt category/heldout gate derives diverse seed IDs from natural prompt content and exercises a sparse final c=8 group.",
            "This direct MapleBatchRunner gate isolates D1 decode; public scheduler throughput is measured separately by maple_public_batch_bench.py.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": status,
        "natural_seed_gate": {key: natural_gate[key] for key in ("prompt_count", "trajectory_matches", "sparse_final_group_size", "passed")},
        "rows": [{key: row[key] for key in ("concurrency", "median_aggregate_tokens_per_second", "min_aggregate_tokens_per_second", "max_aggregate_tokens_per_second", "correctness_passed")} | {"resident_tracked_bytes": row["resident_tracked"]["current_allocated_bytes"]} for row in rows],
        "artifact": str(args.out),
    }, indent=2, sort_keys=True))
    return 0 if status == "accepted" else 2


if __name__ == "__main__":
    raise SystemExit(main())
