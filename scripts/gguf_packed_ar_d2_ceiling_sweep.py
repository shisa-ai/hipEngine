#!/usr/bin/env python3
"""Diagnose eager resident-session composed wall for D2 vs ceiling.

This is a model-step diagnostic, not an actual-server or continuous-owner gate:
it creates resident sessions directly, chooses compositions with host helpers,
and invokes ``step_batch_native`` eagerly. It does not exercise EngineService,
HTTP, scheduler-owned lowering/telemetry, graph replay, TTFT/ITL, dynamic
membership, or server memory/drain. Production-default promotion requires a
separate real-server same-protocol gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:] = [str(REPO_ROOT), *(entry for entry in sys.path if entry != str(REPO_ROOT))]

import hipengine  # noqa: E402

if Path(hipengine.__file__).resolve().parents[1] != REPO_ROOT:
    raise RuntimeError("sweep imported hipengine from another checkout")
DEFAULT_MODEL = Path("/models/gguf/Qwen3.8-27B-Q4_K_M.gguf")
DEFAULT_COST_MAP = (
    REPO_ROOT / "benchmarks" / "results" / "2026-08-20-concurrency2-qwen38-d2-cost-map.json"
)

_ENV = {
    "HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN": "1",
    "HIPENGINE_GGUF_GDN_PREFILL_MODE": "exact",
    "HIPENGINE_GGUF_INT8_KV_ALLOW_UNVERIFIED": "1",
    "HIPENGINE_GGUF_AR_PACKED_DECODE": "1",
    "HIPENGINE_GGUF_AR_PACKED_PREFILL": "1",
}


def _session_build_policy(args: argparse.Namespace) -> dict[str, Any]:
    compiler_version = None
    if args.compiler_version_file is not None:
        compiler_version = args.compiler_version_file.expanduser().read_text(
            encoding="utf-8"
        ).strip()
        if not compiler_version:
            raise ValueError("compiler version file is empty")
    return {
        "compiler_version": compiler_version,
        "require_cached_build": bool(args.require_cached_build),
    }


def _prompt(token_id: int, prompt_length: int, row: int) -> list[int]:
    prompt = [int(token_id)] * int(prompt_length)
    prompt[-1] = int(token_id) + (row % 4)
    return prompt


def _aggregate_goodput_tokens_per_s(logical_c: int, wall_ms: float) -> float:
    if int(logical_c) <= 0 or float(wall_ms) <= 0.0:
        raise ValueError("goodput requires positive logical_c and wall_ms")
    return int(logical_c) * 1000.0 / float(wall_ms)


def _stats(values: Sequence[float]) -> dict[str, float]:
    values = [float(value) for value in values]
    return {
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "min": min(values),
        "max": max(values),
        "samples": len(values),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    from hipengine.dispatch.d2_resolver import (
        ceiling_partition,
        cost_table_from_artifact,
        d2_partition,
    )
    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession

    os.environ.update(_ENV)
    model = Path(args.model).expanduser().resolve()
    if not model.is_file():
        raise ValueError(f"model does not exist: {model}")
    cost_map = Path(args.cost_map).expanduser().resolve()
    if not cost_map.is_file():
        raise ValueError(f"D2 cost map does not exist: {cost_map}")

    max_rows = int(args.max_rows)
    min_rows = int(args.min_rows)
    prompt_length = int(args.prompt_length)
    token_id = int(args.prompt_token_id)
    warmup = int(args.warmup)
    measured_steps = int(args.measured)

    cost_table = cost_table_from_artifact(cost_map)
    widths = cost_table.widths
    if widths[-1] != 8:
        raise ValueError(f"cost map widths {widths} do not include max physical width 8")
    if min_rows < 1 or max_rows > 32 or min_rows > max_rows:
        raise ValueError("require 1 <= min_rows <= max_rows <= 32")

    build_policy = _session_build_policy(args)
    max_sequence_length = prompt_length + measured_steps + warmup + 2

    # One weight-loading owner is materialized once; every per-c sweep batch
    # shares its runner/runtime so per-c session creation is cheap and each c
    # starts from a fresh prefill scratch (no cross-c position accumulation).
    with ExitStack() as stack:
        weights_owner = stack.enter_context(
            Qwen35GGUFResidentSession(
                model,
                backend=str(args.backend),
                max_sequence_length=max_sequence_length,
                **build_policy,
            )
        )
        shared_runner = weights_owner.runner
        if shared_runner is None:
            raise RuntimeError("GGUF shared runner was not materialized")
        shared_runtime = weights_owner.runtime

        def make_sessions(c: int, cstack: ExitStack) -> list[Any]:
            batch: list[Any] = []
            for _ in range(c):
                batch.append(
                    cstack.enter_context(
                        Qwen35GGUFResidentSession(
                            model,
                            backend=str(args.backend),
                            runtime=shared_runtime,
                            shared_runner=shared_runner,
                            max_sequence_length=max_sequence_length,
                            **build_policy,
                        )
                    )
                )
            return batch

        def measure_composition(
            c: int,
            width_sequence: Sequence[int],
            sessions: list[Any],
        ) -> dict[str, Any]:
            """Prefill then time serial per-group eager decode for one
            composition on a fresh per-(c, composition) session batch."""
            offsets: list[int] = []
            cursor = 0
            for width in width_sequence:
                offsets.append(cursor)
                cursor += int(width)
            assert cursor == c
            tokens = [0] * c
            for width, base in zip(width_sequence, offsets, strict=True):
                indices = tuple(range(base, base + width))
                group_sessions = tuple(sessions[index] for index in indices)
                group_prompts = tuple(
                    _prompt(token_id, prompt_length, index) for index in indices
                )
                results = group_sessions[0].prefill_batch_native(
                    group_prompts,
                    sessions=group_sessions,
                    return_logits=False,
                )
                for index, result in zip(indices, results, strict=True):
                    tokens[index] = int(result.token_id)
            shared_runtime.device_synchronize()

            for _ in range(warmup):
                for width, base in zip(width_sequence, offsets, strict=True):
                    indices = tuple(range(base, base + width))
                    group_sessions = tuple(sessions[index] for index in indices)
                    results = group_sessions[0].step_batch_native(
                        [tokens[index] for index in indices],
                        sessions=group_sessions,
                        positions=[
                            int(session.position) for session in group_sessions
                        ],
                        return_logits=False,
                        scatter_state=False,
                        physical_rows=width,
                        active_slot_indices=tuple(range(width)),
                    )
                    for index, result in zip(indices, results, strict=True):
                        tokens[index] = int(result.token_id)
            shared_runtime.device_synchronize()

            step_walls: list[float] = []
            for _ in range(measured_steps):
                step_start = time.perf_counter()
                for width, base in zip(width_sequence, offsets, strict=True):
                    indices = tuple(range(base, base + width))
                    group_sessions = tuple(sessions[index] for index in indices)
                    results = group_sessions[0].step_batch_native(
                        [tokens[index] for index in indices],
                        sessions=group_sessions,
                        positions=[
                            int(session.position) for session in group_sessions
                        ],
                        return_logits=False,
                        scatter_state=False,
                        physical_rows=width,
                        active_slot_indices=tuple(range(width)),
                    )
                    for index, result in zip(indices, results, strict=True):
                        tokens[index] = int(result.token_id)
                shared_runtime.device_synchronize()
                step_walls.append(time.perf_counter() - step_start)
            return {"wall_ms": _stats([wall * 1e3 for wall in step_walls])}

        rows_evidence: list[dict[str, Any]] = []
        d2_wins = 0
        ceiling_wins = 0
        ties = 0
        differentiated = 0
        worse_beyond_noise: list[int] = []
        for c in range(min_rows, max_rows + 1):
            d2_comp = d2_partition(c, cost_table)
            ceil_comp = ceiling_partition(c, widths)
            d2_est_ms = sum(cost_table.cost_ms(width) for width in d2_comp)
            ceil_est_ms = sum(cost_table.cost_ms(width) for width in ceil_comp)
            with ExitStack() as cstack:
                sessions = make_sessions(c, cstack)
                d2_wall = measure_composition(c, d2_comp, sessions)
            with ExitStack() as cstack2:
                sessions2 = make_sessions(c, cstack2)
                ceil_wall = measure_composition(c, ceil_comp, sessions2)
            d2_med = d2_wall["wall_ms"]["median"]
            ceil_med = ceil_wall["wall_ms"]["median"]
            margin = d2_med - ceil_med
            # Noise-aware verdict. Rows whose D2/ceiling compositions are
            # identical (c<=8) are definitional ties; only differentiated rows
            # decide the gate, and a D2 deficit within a 2% or 0.5 ms noise
            # floor is treated as a tie, not a regression.
            is_differentiated = tuple(d2_comp) != tuple(ceil_comp)
            noise_tol_ms = max(0.5, 0.02 * ceil_med)
            verdict = "tie"
            if is_differentiated:
                differentiated += 1
                if margin < -noise_tol_ms:
                    d2_wins += 1
                    verdict = "d2_win"
                elif margin > noise_tol_ms:
                    ceiling_wins += 1
                    worse_beyond_noise.append(c)
                    verdict = "ceiling_win"
                else:
                    ties += 1
            else:
                ties += 1
            d2_goodput = _aggregate_goodput_tokens_per_s(c, d2_med)
            ceiling_goodput = _aggregate_goodput_tokens_per_s(c, ceil_med)
            rows_evidence.append(
                {
                    "logical_c": c,
                    "d2_composition": list(d2_comp),
                    "ceiling_composition": list(ceil_comp),
                    "differentiated": is_differentiated,
                    "d2_estimated_model_step_ms": round(d2_est_ms, 3),
                    "ceiling_estimated_model_step_ms": round(ceil_est_ms, 3),
                    "d2_measured_wall_ms": d2_wall["wall_ms"],
                    "ceiling_measured_wall_ms": ceil_wall["wall_ms"],
                    "d2_minus_ceiling_median_ms": round(margin, 3),
                    "noise_tol_ms": round(noise_tol_ms, 3),
                    "verdict": verdict,
                    "d2_no_slower_than_ceiling": bool(margin <= noise_tol_ms),
                    "d2_aggregate_decode_tokens_per_s": d2_goodput,
                    "ceiling_aggregate_decode_tokens_per_s": ceiling_goodput,
                    "d2_goodput_pct_vs_ceiling": (
                        (d2_goodput / ceiling_goodput - 1.0) * 100.0
                    ),
                }
            )
            print(
                f"c{c:2} d2={list(d2_comp)!s:18} ceil={list(ceil_comp)!s:18} "
                f"d2={d2_med:7.2f}ms ceil={ceil_med:7.2f}ms "
                f"margin={margin:+7.2f}ms {verdict}"
            )
            sys.stdout.flush()

        observation_passed = len(worse_beyond_noise) == 0
        return {
            "schema": 2,
            "kind": "concurrency2_qwen38_d2_vs_ceiling_composed_wall_sweep",
            "status": "diagnostic_complete",
            "passed": False,
            "measurement_valid": False,
            "diagnostic_observation_passed": observation_passed,
            "performance_claim": False,
            "scope": "direct_resident_eager_model_step",
            "objective": "serial_sum_model_step_wall",
            "noise_policy": {
                "noise_tol_ms": "max(0.5, 0.02 * ceiling_median)",
                "only_differentiated_rows_decide": True,
            },
            "identity": getattr(cost_table, "identity", None).to_json_dict()
            if getattr(cost_table, "identity", None) is not None
            else None,
            "workload": {
                "model": str(model),
                "backend": str(args.backend),
                "quant": str(args.quant),
                "kv_dtype": "bf16",
                "execution_profile": "strict",
                "graph_mode": "captured_replay",
                "physical_widths": list(widths),
                "prompt_length": prompt_length,
                "warmup_steps": warmup,
                "measured_steps": measured_steps,
                "max_rows": max_rows,
            },
            "limitations": [
                "not HTTP, EngineService, scheduler-owned lowering, or continuous-owner execution",
                "eager step_batch_native rather than captured graph replay",
                "no TTFT, ITL, dynamic membership, server memory, or final-drain evidence",
                "D2 is always measured before ceiling and the order is not counterbalanced",
                "2%/0.5 ms noise floor is diagnostic, not a production non-regression policy",
            ],
            "summary": {
                "rows": len(rows_evidence),
                "differentiated": differentiated,
                "d2_wins": d2_wins,
                "ceiling_wins": ceiling_wins,
                "ties": ties,
                "worse_beyond_noise": worse_beyond_noise,
            },
            "rows": rows_evidence,
            "cost_map_source": str(cost_map),
            "cost_map_sha256": hashlib.sha256(cost_map.read_bytes()).hexdigest(),
        }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--backend", default="hip_gfx1100")
    parser.add_argument("--quant", default="gguf_q4_k_m")
    parser.add_argument("--cost-map", type=Path, default=DEFAULT_COST_MAP)
    parser.add_argument("--prompt-token-id", type=int, default=9707)
    parser.add_argument("--prompt-length", type=int, default=512)
    parser.add_argument("--max-rows", type=int, default=32)
    parser.add_argument("--min-rows", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--measured", type=int, default=8)
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    result = run(args)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0 if result["diagnostic_observation_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
