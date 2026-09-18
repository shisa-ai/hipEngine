#!/usr/bin/env python3
"""Tier-1 capacity probe for the TP2 route (see benchmarks/HARNESSES.md).

``scripts/gguf_capacity_probe.py`` applies the two-tier capacity protocol to the
TP1 resident route: the per-layer KV caches, scales and metadata tables are
sized by ``max_positions`` at session initialization, not by the prompt, so a
short prompt at a target ``--max-sequence-length`` proves the same memory
envelope a full-length ladder point proves. The TP2 route needs the same
treatment for a different reason: an uneven MLP split moves resident weight
bytes *between* the two cards (the faster card gains the slower card's shard),
so the question "does this context still fit?" is asked per rank and can have a
different answer on each.

This probe therefore:

* builds a real :class:`MlpTP2GenerationSession` at the target declared context
  - the point where every rank's KV pool, weight shard and scratch is acquired;
* runs a short prompt plus a few decode transitions and checks that every
  logits row is finite, which is what makes the envelope *usable* rather than
  merely allocatable;
* reports the per-rank resident and transient high-water marks and the resolved
  per-rank FFN widths, so a ladder point states which split it measured.

It is a fitting question only. Deep-context kernel stability and tok/s at depth
remain Tier 2 (a full-prompt harness point), which this probe must not be used
to search with.

Usage::

    python scripts/tp2_capacity_probe.py --max-sequence-length 8192 \
        --fractions 0.417145/0.582855 --json /tmp/tp2-cap-8192-uneven.json
"""

from __future__ import annotations

import argparse
import json
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--max-sequence-length", type=int, required=True)
    parser.add_argument("--fractions", type=_parse_fractions, default=None)
    parser.add_argument("--devices", default="0,1")
    parser.add_argument(
        "--prompt-length",
        type=int,
        default=2048,
        help="short prompt; memory validity does not need the full context",
    )
    parser.add_argument("--decode-tokens", type=int, default=4)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    import numpy as np

    from hipengine.core.device import scoped_current_device
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import memory_stats, reset_memory_stats
    from hipengine.distributed.tp2_generate import MlpTP2GenerationSession

    devices = tuple(int(part) for part in str(args.devices).split(","))
    if len(devices) != 2:
        raise SystemExit(f"--devices must name exactly two devices, got {args.devices!r}")
    if args.fractions is not None and len(args.fractions) != len(devices):
        raise SystemExit(
            f"--fractions names {len(args.fractions)} shares for {len(devices)} devices"
        )
    if int(args.prompt_length) >= int(args.max_sequence_length):
        raise SystemExit(
            f"--prompt-length ({args.prompt_length}) must be below the declared "
            f"context ({args.max_sequence_length})"
        )

    result: dict[str, Any] = {
        "kind": "tp2_capacity_probe",
        "protocol_tier": 1,
        "model": str(args.model),
        "max_sequence_length": int(args.max_sequence_length),
        "prompt_length": int(args.prompt_length),
        "decode_tokens": int(args.decode_tokens),
        "devices": list(devices),
        "fractions": None if args.fractions is None else list(args.fractions),
    }
    runtime = get_hip_runtime()

    def _device_used_gib(device: int) -> float:
        with scoped_current_device(runtime, device):
            free, total = runtime.mem_get_info()
        return (int(total) - int(free)) / 2**30

    baseline_vram = {int(rank): _device_used_gib(int(rank)) for rank in devices}
    result["vram_used_gib_baseline"] = {
        str(rank): round(value, 6) for rank, value in baseline_vram.items()
    }
    session: MlpTP2GenerationSession | None = None
    try:
        started = time.perf_counter()
        session = MlpTP2GenerationSession(
            args.model,
            devices=devices,
            mode="tp2",
            max_sequence_length=int(args.max_sequence_length),
            uneven_split=args.fractions,
        )
        result["build_seconds"] = round(time.perf_counter() - started, 3)
        result["per_rank_ffn"] = {
            str(rank): int(width) for rank, width in session._shard_group.per_rank_ffn.items()
        }
        variants = session._shard_group.mlp_decode_variant
        result["per_rank_variants"] = (
            {str(rank): value for rank, value in variants.items()}
            if isinstance(variants, dict)
            else {str(rank): variants for rank in sorted(session._shard_group.per_rank_ffn)}
        )

        # Two different quantities, deliberately kept apart:
        # ``current_allocated_bytes`` is hipEngine's own process-wide tracked
        # counter (it is *not* device-scoped, so every rank reports the same
        # value), while ``vram_used_gib`` is the device truth from
        # hipMemGetInfo minus this rank's pre-build baseline. The capacity
        # question is per rank, so the device number is the one to read; the
        # tracked total stays for regression comparisons against other probes.
        per_rank: dict[str, Any] = {}
        for rank in devices:
            with scoped_current_device(runtime, rank):
                per_rank[str(rank)] = {
                    "vram_used_gib": round(
                        _device_used_gib(int(rank)) - baseline_vram[int(rank)], 6
                    ),
                    "tracked_allocated_gib_process_wide": round(
                        int(memory_stats().get("current_allocated_bytes", 0)) / 2**30, 6
                    ),
                }
                reset_memory_stats()

        prompt_ids = [9707] * int(args.prompt_length)
        started = time.perf_counter()
        generation = session.generate(
            prompt_ids,
            max_new_tokens=int(args.decode_tokens),
            capture_logits=True,
        )
        result["generate_seconds"] = round(time.perf_counter() - started, 3)
        logits = np.asarray(generation.logits, dtype=np.float32)
        finite = bool(logits.size) and bool(np.all(np.isfinite(logits)))
        result["generated_tokens"] = [int(token) for token in generation.token_ids]
        for rank in devices:
            with scoped_current_device(runtime, rank):
                peak = int(memory_stats().get("peak_allocated_bytes", 0))
                per_rank[str(rank)].update(
                    {
                        "vram_used_gib_after_decode": round(
                            _device_used_gib(int(rank)) - baseline_vram[int(rank)], 6
                        ),
                        "tracked_peak_gib_process_wide": round(peak / 2**30, 6),
                    }
                )
        result["per_rank_memory"] = per_rank
        result.update(
            {
                "status": "pass" if finite else "nonfinite_logits",
                "finite_logits": finite,
                "decode_rows": int(logits.shape[0]) if logits.ndim else 0,
            }
        )
    except Exception as exc:  # noqa: BLE001 - hip OOM and host OOM both land here
        message = str(exc)
        if "out of memory" in message.lower() or "oom" in message.lower():
            result["status"] = "oom"
            result["error"] = message[:300]
        else:
            result["status"] = "error"
            result["error"] = f"{type(exc).__name__}: {message[:300]}"
    finally:
        if session is not None:
            try:
                session.close()
            except Exception:  # noqa: BLE001 - teardown best effort
                pass

    payload = json.dumps(result, indent=2)
    print(payload)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(payload + "\n", encoding="utf-8")
    return 0 if result.get("status") == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
