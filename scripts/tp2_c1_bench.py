#!/usr/bin/env python3
"""Matched-protocol c=1 prefill/decode/memory cell for the TP1 and TP2 routes.

The TP2 documentation's headline numbers come from different harnesses: the
decode cell is a 4-prompt x 16-transition diagnostic and the prefill schedule
is token-serial, so there is no single c=1 cell that reports prefill throughput,
decode throughput and per-rank resident memory on one protocol. That is exactly
the shape an external engine comparison needs, so this script produces it:

* one fresh session at a declared context, on the requested devices and mode;
* one prompt of ``--prompt-length`` tokens, then ``--decode-tokens`` greedy
  transitions, all inside a single ``generate`` call whose per-step traces
  separate the prefill positions from the decode positions;
* per-rank resident bytes after load and the peak including the step
  workspace, read from the runtime's own counters;
* the resolved route, including per-rank shard widths and variants.

Prefill and decode rates are computed from the same traces the session already
records for attribution, so the numbers are end-to-end walls that include the
synchronized logits readback on every step - not device times.

Usage::

    python scripts/tp2_c1_bench.py --mode tp2 --devices 0,1 \
        --prompt-length 512 --decode-tokens 128 \
        --json benchmarks/results/2026-09-19-w7900-tp2-c1-cell.json
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_MODEL = Path("/models/gguf/Qwen3.8-27B-Q4_K_M.gguf")


def _parse_fractions(text: str | None) -> tuple[float, ...] | None:
    if text is None:
        return None
    parts = [part.strip() for part in str(text).split("/")]
    if len(parts) < 2 or any(not part for part in parts):
        raise SystemExit(f"--fractions must be a '/' separated share per rank, got {text!r}")
    try:
        return tuple(float(part) for part in parts)
    except ValueError as error:
        raise SystemExit(f"--fractions is not numeric: {text!r}") from error


def _rate(times: list[float]) -> dict[str, float] | None:
    if not times:
        return None
    total = float(sum(times))
    return {
        "tokens": len(times),
        "total_s": total,
        "mean_ms_per_token": 1000.0 * total / len(times),
        "p50_ms_per_token": 1000.0 * statistics.median(times),
        "tok_per_s": len(times) / total if total > 0 else 0.0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--mode", choices=("tp1", "tp2"), default="tp2")
    parser.add_argument("--devices", default=None, help="default: 0,1 for tp2, 0 for tp1")
    parser.add_argument("--prompt-length", type=int, default=512)
    parser.add_argument("--decode-tokens", type=int, default=128)
    parser.add_argument("--fractions", type=_parse_fractions, default=None)
    parser.add_argument("--max-sequence-length", type=int, default=None)
    parser.add_argument("--token-id", type=int, default=9707)
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="generations measured on one resident session; each call resets sequence state",
    )
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    devices = (
        tuple(int(part) for part in str(args.devices).split(","))
        if args.devices
        else ((0,) if args.mode == "tp1" else (0, 1))
    )
    if args.fractions is not None and args.mode != "tp2":
        raise SystemExit("--fractions applies to the tp2 route only")
    context = int(args.max_sequence_length or (int(args.prompt_length) + int(args.decode_tokens)))

    from hipengine.core.device import scoped_current_device
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import memory_stats, reset_memory_stats
    from hipengine.distributed.tp2_generate import MlpTP2GenerationSession

    runtime = get_hip_runtime()
    result: dict[str, Any] = {
        "schema": 1,
        "kind": "tp2-c1-bench",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": platform.node(),
        "model": str(args.model),
        "mode": args.mode,
        "devices": list(devices),
        "prompt_length": int(args.prompt_length),
        "decode_tokens": int(args.decode_tokens),
        "max_sequence_length": context,
        "fractions": None if args.fractions is None else list(args.fractions),
    }

    # The device's own counter, before this process allocates anything: a
    # per-rank VRAM delta is the number an external engine comparison can use,
    # because it does not depend on what else is resident on the card.
    baseline: dict[str, float] = {}
    for rank in devices:
        with scoped_current_device(runtime, rank):
            free_bytes, total_bytes = runtime.mem_get_info()
            baseline[str(rank)] = round((total_bytes - free_bytes) / 2**30, 6)
    result["vram_used_gib_baseline"] = baseline

    session: MlpTP2GenerationSession | None = None
    try:
        started = time.perf_counter()
        session = MlpTP2GenerationSession(
            args.model,
            devices=devices,
            mode=args.mode,
            max_sequence_length=context,
            uneven_split=(args.fractions if args.mode == "tp2" else None),
        )
        result["build_seconds"] = round(time.perf_counter() - started, 3)
        group = getattr(session, "_shard_group", None)
        result["resolved"] = {
            "schedule": getattr(session, "schedule", None),
            "prefill_schedule": getattr(session, "prefill_schedule", None),
            "driver": getattr(session, "driver", None),
            "reduce_mode": getattr(session, "reduce_mode", None),
            "decode_partial_dtype": getattr(session, "decode_partial_dtype", None),
            "head_shard": getattr(session, "head_shard", None),
            "per_rank_ffn": (
                {str(k): int(v) for k, v in group.per_rank_ffn.items()} if group else None
            ),
        }
        if group is not None:
            variant = group.mlp_decode_variant
            result["resolved"]["per_rank_variants"] = (
                {str(k): v for k, v in variant.items()}
                if isinstance(variant, dict)
                else {str(rank): variant for rank in sorted(group.per_rank_ffn)}
            )

        per_rank: dict[str, dict[str, float]] = {}
        for rank in devices:
            with scoped_current_device(runtime, rank):
                free_bytes, total_bytes = runtime.mem_get_info()
                per_rank[str(rank)] = {
                    "resident_gib": round(
                        int(memory_stats().get("current_allocated_bytes", 0)) / 2**30, 6
                    ),
                    "device": str(runtime.device_get_name(rank)),
                    "free_gib_after_load": round(free_bytes / 2**30, 6),
                    "total_gib": round(total_bytes / 2**30, 6),
                    "vram_used_gib_after_load": round(
                        (total_bytes - free_bytes) / 2**30, 6
                    ),
                    "session_vram_gib": round(
                        (total_bytes - free_bytes) / 2**30 - baseline[str(rank)], 6
                    ),
                }
                reset_memory_stats()
        result["memory_after_load"] = per_rank

        prompt_ids = [int(args.token_id)] * int(args.prompt_length)
        # One warmup generation, then the measured repeats, all on the resident
        # session: the first call pays graph capture, and a c=1 cell should not
        # report that as decode time.
        session.generate(prompt_ids, max_new_tokens=1)
        prefill_runs: list[dict[str, float]] = []
        decode_runs: list[dict[str, float]] = []
        started = time.perf_counter()
        for _ in range(max(1, int(args.repeats))):
            generation = session.generate(
                prompt_ids,
                max_new_tokens=int(args.decode_tokens),
            )
            prefill_runs.append(
                _rate([t.total_s for t in generation.step_traces if t.kind == "prefill"])
            )
            decode_runs.append(
                _rate([t.total_s for t in generation.step_traces if t.kind == "decode"])
            )
            result.setdefault("generated_tokens", [int(t) for t in generation.token_ids[:8]])
        result["generate_seconds"] = round(time.perf_counter() - started, 3)
        result["prefill"] = prefill_runs[-1]
        result["decode"] = decode_runs[-1]
        result["prefill_runs"] = prefill_runs
        result["decode_runs"] = decode_runs
        for section, runs in (("prefill", prefill_runs), ("decode", decode_runs)):
            rates = sorted(run["tok_per_s"] for run in runs)
            result[f"{section}_tok_per_s_median"] = rates[len(rates) // 2]
            result[f"{section}_tok_per_s_samples"] = rates

        for rank in devices:
            with scoped_current_device(runtime, rank):
                peak = int(memory_stats().get("peak_allocated_bytes", 0))
                resident = int(per_rank[str(rank)]["resident_gib"] * 2**30)
                free_bytes, total_bytes = runtime.mem_get_info()
                per_rank[str(rank)].update(
                    {
                        "transient_peak_gib": round((peak - resident) / 2**30, 6),
                        "resident_plus_transient_peak_gib": round(peak / 2**30, 6),
                        "vram_used_gib_after_decode": round(
                            (total_bytes - free_bytes) / 2**30, 6
                        ),
                    }
                )
        result["memory"] = per_rank
    finally:
        if session is not None:
            try:
                session.close()
            except Exception:  # noqa: BLE001 - teardown best effort
                pass

    for section in ("prefill", "decode"):
        rate = result.get(section)
        if rate:
            print(
                f"{section}: {rate['tokens']} tokens in {rate['total_s']:.3f}s "
                f"= {rate['tok_per_s']:.3f} tok/s (p50 {rate['p50_ms_per_token']:.3f} ms), "
                f"median over {len(result.get(section + '_tok_per_s_samples', []))} runs "
                f"{result.get(section + '_tok_per_s_median', 0):.3f} tok/s"
            )
    for rank, entry in (result.get("memory") or {}).items():
        print(
            f"rank {rank} ({entry['device']}): session VRAM "
            f"{entry.get('session_vram_gib', 0):.3f} GiB "
            f"(device used {entry.get('vram_used_gib_after_load', 0):.3f} of "
            f"{entry.get('total_gib', 0):.1f} GiB, baseline "
            f"{result['vram_used_gib_baseline'][rank]:.3f}), "
            f"tracked resident {entry['resident_gib']:.3f} GiB"
        )
    payload = json.dumps(result, indent=2)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(payload + "\n", encoding="utf-8")
        print("wrote", args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
