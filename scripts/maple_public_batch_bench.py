#!/usr/bin/env python3
"""Qualify Maple public c1/c2/c4/c8 admission, decode, and reclaim.

The timed rows all use ``SubmitPollTextGenerator`` with the same warmup and
request protocol. Model load and lazy allocation are excluded; prompt admission,
native prefill, decode, sampling, output collection, and slot reclaim are timed.
An independent serial ``MapleRunner`` supplies trajectory oracles only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import statistics
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from hipengine.core.memory import memory_stats, reset_memory_stats
from hipengine.generation.engine_loop import EngineLoopConfig, SubmitPollTextGenerator
from hipengine.generation.maple import MapleGenerator
from hipengine.generation.registry import GenerationRequest
from hipengine.loading.maple import load_maple_checkpoint
from hipengine.runtime.maple import MapleRunner
from hipengine.tokenization.maple import MapleTokenizer

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUITE = ROOT / "benchmarks/prompts/mtpbench-code-general-ja.jsonl"
DEFAULT_HELDOUT = ROOT / "benchmarks/prompts/gdn-prefill-category-heldouts.jsonl"
QUALIFIED_WIDTHS = (1, 2, 4, 8)


def _capture(command: list[str]) -> dict[str, Any]:
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    return {
        "command": shlex.join(command),
        "returncode": completed.returncode,
        "output": (completed.stdout + completed.stderr).strip(),
    }


def _load_rows(path: Path, *, heldout: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        users = [
            message["content"]
            for message in record["messages"]
            if message["role"] == "user"
        ]
        if len(users) != 1:
            raise ValueError(f"prompt {record.get('id')} must contain one user row")
        rows.append(
            {
                "id": str(record["id"]),
                "category": str(record["category"]),
                "heldout": bool(heldout),
                "text": str(users[0]),
            }
        )
    return rows


def _trajectory_hash(rows: list[tuple[int, ...]]) -> str:
    payload = b"".join(
        int(token).to_bytes(4, "little", signed=True)
        for row in rows
        for token in row
    )
    return hashlib.sha256(payload).hexdigest()


def _generation_request(
    prompts: tuple[tuple[int, ...], ...], *, steps: int
) -> GenerationRequest:
    return GenerationRequest(
        prompts=prompts,
        max_tokens=steps,
        temperature=0.0,
        top_p=1.0,
        ignore_eos=True,
    )


def _serial_oracles(
    checkpoint,
    prompts: list[tuple[int, ...]],
    *,
    backend: str,
    max_context: int,
    steps: int,
) -> list[tuple[int, ...]]:
    runner = MapleRunner.load(
        checkpoint,
        backend=backend,
        max_context=max_context,
    )
    trajectories: list[tuple[int, ...]] = []
    try:
        for prompt in prompts:
            runner.reset()
            result = runner.prefill_native(prompt)
            generated: list[int] = []
            for _ in range(steps):
                generated.append(int(result.token_id))
                result = runner.step(int(result.token_id))
            trajectories.append(tuple(generated))
    finally:
        runner.close()
    return trajectories


def _measure_groups(
    adapter: SubmitPollTextGenerator,
    generator: MapleGenerator,
    prompts: list[tuple[int, ...]],
    expected: list[tuple[int, ...]],
    *,
    width: int,
    steps: int,
    repetitions: int,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for repetition in range(repetitions):
        got: list[tuple[int, ...]] = []
        groups: list[dict[str, Any]] = []
        elapsed = 0.0
        for offset in range(0, len(prompts), width):
            group = tuple(prompts[offset : offset + width])
            started = time.perf_counter()
            outputs = adapter.generate_detailed(
                _generation_request(group, steps=steps)
            )
            elapsed += time.perf_counter() - started
            got.extend(tuple(output.generated_token_ids or ()) for output in outputs)
            metadata = dict(generator.last_batch_generation or {})
            snapshot = adapter.live_loop_snapshot()["runner"]
            groups.append(
                {
                    "logical_rows": len(group),
                    "physical_rows": metadata.get("physical_batch_rows"),
                    "slots_reclaimed": all(
                        request_id is None
                        for request_id in snapshot["slot_to_request"]
                    ),
                }
            )
        matches = [
            actual == reference
            for actual, reference in zip(got, expected, strict=True)
        ]
        samples.append(
            {
                "repetition": repetition,
                "elapsed_seconds": elapsed,
                "aggregate_generated_tokens_per_second": (
                    len(prompts) * steps / elapsed
                ),
                "trajectory_matches": sum(matches),
                "all_trajectories_equal": all(matches),
                "groups": groups,
            }
        )
    return samples


def _summary(
    samples: list[dict[str, Any]],
    *,
    width: int,
    resident: dict[str, int],
    after_close: dict[str, int],
) -> dict[str, Any]:
    rates = [sample["aggregate_generated_tokens_per_second"] for sample in samples]
    return {
        "concurrency": width,
        "samples": samples,
        "median_aggregate_generated_tokens_per_second": statistics.median(rates),
        "min_aggregate_generated_tokens_per_second": min(rates),
        "max_aggregate_generated_tokens_per_second": max(rates),
        "all_trajectories_equal": all(
            sample["all_trajectories_equal"] for sample in samples
        ),
        "resident_tracked": resident,
        "after_close": after_close,
        "lifecycle_passed": (
            after_close["current_allocated_bytes"] == 0
            and after_close["active_allocations"] == 0
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="deepgrove/maple-preview-2bit-mlx")
    parser.add_argument("--backend", default="hip_gfx1151")
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--heldout", type=Path, default=DEFAULT_HELDOUT)
    parser.add_argument("--widths", default="1,2,4,8")
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--single-active-capacity", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    widths = tuple(int(part) for part in args.widths.split(",") if part)
    if (
        not widths
        or len(set(widths)) != len(widths)
        or any(width <= 0 or width > 8 for width in widths)
    ):
        raise ValueError("--widths must contain unique values in [1, 8]")
    if args.steps <= 0 or args.repetitions <= 0 or args.warmup_steps <= 0:
        raise ValueError("steps, repetitions, and warmup steps must be positive")
    if args.single_active_capacity not in widths:
        raise ValueError("--single-active-capacity must be one of --widths")

    prompt_rows = _load_rows(args.suite, heldout=False) + _load_rows(
        args.heldout, heldout=True
    )
    checkpoint = load_maple_checkpoint(args.model)
    spec = checkpoint.spec
    tokenizer = MapleTokenizer.from_model_path(
        checkpoint.index.model_path,
        model_vocab_size=spec.vocab_size,
        eos_token_id=spec.eos_token_id,
        bos_token_id=spec.bos_token_id,
    )
    prompts = [tokenizer.encode_chat(row["text"]) for row in prompt_rows]
    max_context = max(map(len, prompts)) + args.steps + 2
    if max_context > spec.max_position_embeddings:
        raise ValueError("prompt plus generated tokens exceed checkpoint context")

    expected = _serial_oracles(
        checkpoint,
        prompts,
        backend=args.backend,
        max_context=max_context,
        steps=args.steps,
    )
    if memory_stats()["current_allocated_bytes"] != 0:
        raise RuntimeError("serial oracle owner did not close to zero")

    width_rows: list[dict[str, Any]] = []
    single_active: dict[str, Any] | None = None
    for width in widths:
        reset_memory_stats()
        generator = MapleGenerator(
            model_path=checkpoint.index.model_path,
            weight_index=checkpoint.index,
            model_plugin=SimpleNamespace(),
            backend=args.backend,
        )
        adapter = SubmitPollTextGenerator(
            generator,
            capacity=width,
            config=EngineLoopConfig(
                max_active_requests=width,
                max_prefill_chunk_tokens=256,
                prefill_decode_policy="protect_ttft",
            ),
        )
        samples: list[dict[str, Any]]
        single_samples: list[dict[str, Any]] | None = None
        try:
            adapter.prepare(max_sequence_length=max_context)
            adapter.generate_detailed(
                _generation_request(
                    (prompts[0],),
                    steps=args.warmup_steps,
                )
            )
            resident = memory_stats().copy()
            samples = _measure_groups(
                adapter,
                generator,
                prompts,
                expected,
                width=width,
                steps=args.steps,
                repetitions=args.repetitions,
            )
            if width == args.single_active_capacity:
                single_samples = _measure_groups(
                    adapter,
                    generator,
                    prompts,
                    expected,
                    width=1,
                    steps=args.steps,
                    repetitions=args.repetitions,
                )
        finally:
            adapter.close()
        after_close = memory_stats().copy()
        width_rows.append(
            _summary(
                samples,
                width=width,
                resident=resident,
                after_close=after_close,
            )
        )
        if single_samples is not None:
            single_active = _summary(
                single_samples,
                width=1,
                resident=resident,
                after_close=after_close,
            )
            single_active["physical_capacity"] = width

    baseline = next((row for row in width_rows if row["concurrency"] == 1), None)
    if baseline is None:
        raise ValueError("qualified public throughput requires a c1 row")
    baseline_rate = baseline["median_aggregate_generated_tokens_per_second"]
    for row in width_rows:
        row["speedup_vs_public_c1"] = (
            row["median_aggregate_generated_tokens_per_second"] / baseline_rate
        )
    assert single_active is not None
    single_active["rate_ratio_vs_public_c1"] = (
        single_active["median_aggregate_generated_tokens_per_second"]
        / baseline_rate
    )

    tracked_status = _capture(
        ["git", "status", "--porcelain=v1", "--untracked-files=no"]
    )["output"]
    head = _capture(["git", "rev-parse", "HEAD"])["output"]
    qualified = (
        widths == QUALIFIED_WIDTHS
        and args.steps >= 64
        and args.repetitions >= 3
        and args.warmup_steps >= 2
        and len(prompt_rows) == 18
    )
    correctness_passed = (
        all(row["all_trajectories_equal"] for row in width_rows)
        and single_active["all_trajectories_equal"]
    )
    lifecycle_passed = (
        all(row["lifecycle_passed"] for row in width_rows)
        and single_active["lifecycle_passed"]
    )
    accepted = (
        qualified
        and correctness_passed
        and lifecycle_passed
        and not tracked_status
    )
    timestamp = datetime.now(timezone.utc)
    command = shlex.join([sys.executable, *sys.argv])
    artifact = {
        "schema_version": 1,
        "date": datetime.now().astimezone().date().isoformat(),
        "timestamp_utc": timestamp.isoformat(),
        "artifact_type": "maple_p4_public_batch_admission",
        "status": "accepted" if accepted else "rejected",
        "performance_claim": accepted,
        "claim_scope": (
            "public SubmitPollTextGenerator greedy generation; model load and one "
            "lazy-allocation warmup excluded; prompt admission, native prefill, "
            "decode, sampling, output collection, and reclaim included"
        ),
        "model": {
            "id": args.model,
            "revision": checkpoint.index.model_path.name,
            "quant": "maple_ternary2",
            "exact_weight_bytes": checkpoint.validation.exact_weight_bytes,
        },
        "hardware": {
            "gpu": "AMD Radeon 8060S Graphics",
            "architecture": "gfx1151",
            "host": "gfx1151",
            "gpu_max_hw_queues": os.environ.get("GPU_MAX_HW_QUEUES"),
            "rocminfo": _capture(
                [
                    "bash",
                    "-lc",
                    "rocminfo | grep -E 'Name:|Marketing Name:|gfx' | head -8",
                ]
            ),
            "rocm_smi": _capture(
                ["rocm-smi", "--showmeminfo", "vram", "--showuse", "--showtemp"]
            ),
            "hipcc": _capture(["hipcc", "--version"]),
        },
        "software": {
            "git": {
                "head": head,
                "tracked_clean": not tracked_status,
                "tracked_status": tracked_status,
            },
            "python": sys.version.split()[0],
            "harness_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
        },
        "protocol": {
            "command": command,
            "environment": {
                name: os.environ.get(name)
                for name in (
                    "GPU_MAX_HW_QUEUES",
                    "HIPENGINE_HIP_ARCH",
                    "HIPENGINE_COMPILER_VERSION_FILE",
                    "HIPENGINE_REQUIRE_CACHED_BUILD",
                    "HIPENGINE_MAPLE_PREFILL_GROUPED_MOE",
                    "HIPENGINE_MAPLE_PREFILL_GQA4",
                    "HIPENGINE_MAPLE_ROUTER_SINGLE_DISPATCH",
                    "HIPENGINE_MAPLE_AFFINE4_WAVE32_EXACT",
                    "HIPENGINE_MAPLE_BATCH_AFFINE4_ROWREUSE_EXACT",
                )
            },
            "suite": str(args.suite),
            "heldout": str(args.heldout),
            "widths": list(widths),
            "steps_per_request": args.steps,
            "repetitions": args.repetitions,
            "warmup_steps": args.warmup_steps,
            "single_active_physical_capacity": args.single_active_capacity,
            "prompt_count": len(prompt_rows),
            "prompt_tokens": sum(map(len, prompts)),
            "qualified": qualified,
        },
        "performance": {
            "public_rows": width_rows,
            "single_active": single_active,
        },
        "correctness": {
            "passed": correctness_passed,
            "serial_oracle_trajectory_sha256": _trajectory_hash(expected),
            "category_counts": dict(Counter(row["category"] for row in prompt_rows)),
            "heldout_category_counts": dict(
                Counter(
                    row["category"] for row in prompt_rows if row["heldout"]
                )
            ),
            "trajectory_sets_checked": (
                len(width_rows) * args.repetitions + args.repetitions
            ),
            "trajectory_rows_per_set": len(prompt_rows),
            "sparse_final_groups": True,
            "all_groups_reclaimed": all(
                group["slots_reclaimed"]
                for row in width_rows
                for sample in row["samples"]
                for group in sample["groups"]
            ),
        },
        "lifecycle": {
            "passed": lifecycle_passed,
            "all_owners_close_to_zero": lifecycle_passed,
        },
        "notes": [
            "The c1 denominator uses the same public scheduler protocol as c2/c4/c8; the independent serial runner is an untimed correctness oracle.",
            "The capacity-8 single-active row uses the retained request-local c1 path rather than evaluating seven empty dense rows.",
            "All rows use exact full-vocabulary affine4 logits; FlashHead is not enabled.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "artifact": str(args.out),
                "status": artifact["status"],
                "correctness_passed": correctness_passed,
                "lifecycle_passed": lifecycle_passed,
                "rows": [
                    {
                        "concurrency": row["concurrency"],
                        "median_aggregate_generated_tokens_per_second": row[
                            "median_aggregate_generated_tokens_per_second"
                        ],
                        "speedup_vs_public_c1": row["speedup_vs_public_c1"],
                    }
                    for row in width_rows
                ],
                "single_active": {
                    "physical_capacity": single_active["physical_capacity"],
                    "median_aggregate_generated_tokens_per_second": single_active[
                        "median_aggregate_generated_tokens_per_second"
                    ],
                    "rate_ratio_vs_public_c1": single_active[
                        "rate_ratio_vs_public_c1"
                    ],
                },
            },
            indent=2,
        )
    )
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
