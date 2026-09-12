#!/usr/bin/env python3
"""Fast GGUF capacity probe: allocation validity without long prefills.

THE CAPACITY PROTOCOL (see benchmarks/HARNESSES.md "Capacity testing"):

  Tier 1 - THIS PROBE (~3 min/point). The resident scratch's per-layer KV
  caches, scales, and metadata tables are sized by ``max_positions`` at
  session initialization, NOT by the prompt, and the bulk prefill
  workspace acquires before any attention work. A short-prompt run at a
  target ``--max-sequence-length`` therefore proves the same memory
  envelope a full-length synthetic ladder point proves: every allocation
  succeeding plus one bulk prefill and decode transitions with finite
  logits. Use this for every capacity question about whether a context
  size FITS (bracketing bounds, OOM checks, regression checks).

  Tier 2 - full-prompt harness point (qwen35_gguf_bench.py, ~1 min per
  32K tokens on the 27B). Only for questions that need the prefill path
  to actually RUN at that depth: kernel stability across the full
  context, finite logits after deep attention, or tok/s-at-depth claims.
  Never use Tier 2 to search for a bound; use it to confirm the bound
  Tier 1 found.

The 2026-09-09 lesson this encodes: full-prompt ladder points cost
~90 minutes each on the 27B at 160K+ tokens while the allocation
decision they were run for is made in the first ~3 minutes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("/models/gguf/Qwen3.8-27B-Q4_K_M.gguf"))
    parser.add_argument("--max-sequence-length", type=int, required=True)
    parser.add_argument("--kv-storage", default="int8_per_token_head",
                        help="KV storage policy for the probe (default: the capacity route)")
    parser.add_argument("--kv-scale-dtype", default="fp32")
    parser.add_argument("--prompt-length", type=int, default=2048,
                        help="Short prompt (memory validity does not need the full context)")
    parser.add_argument("--decode-tokens", type=int, default=4)
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=1,
        help=(
            "Resident slot count to size for. The server reserves one full-context "
            "KV plane per slot, so a probe that must certify a serving envelope has "
            "to pass the same slot count the server uses (4 for auto-selected GGUF)."
        ),
    )
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    import numpy as np
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import memory_stats, reset_memory_stats
    from hipengine.kvcache import resolve_kv_policy
    from hipengine.runtime.prefill import PrefillConfig
    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession

    result = {
        "kind": "gguf_capacity_probe",
        "protocol_tier": 1,
        "model": str(args.model),
        "max_sequence_length": int(args.max_sequence_length),
        "kv_storage": str(args.kv_storage),
        "prompt_length": int(args.prompt_length),
        "decode_tokens": int(args.decode_tokens),
        "max_batch_size": int(args.max_batch_size),
    }
    prompt_ids = [9707] * int(args.prompt_length)
    runtime = get_hip_runtime()
    policy = resolve_kv_policy(
        args.kv_storage,
        scale_dtype=args.kv_scale_dtype,
    )
    try:
        with Qwen35GGUFResidentSession(
            args.model,
            runtime=runtime,
            max_sequence_length=int(args.max_sequence_length),
            prefill_config=PrefillConfig(),
            kv_policy=policy.create_policy(),
            kv_scale_dtype=str(args.kv_scale_dtype),
            kv_scale_granularity=str(policy.scale_granularity),
            max_batch_size=int(args.max_batch_size),
        ) as session:
            # Split the envelope the way the capacity model does: the resident
            # term is weights + KV + persistent scratch and scales with the
            # declared context, while the transient term is the prefill/decode
            # workspace. Resetting the high-water mark here (which preserves live
            # allocations) keeps the model-load peak from masking the transient
            # peak at small contexts.
            resident_bytes = int(memory_stats().get("current_allocated_bytes", 0))
            process_peak_bytes = int(memory_stats().get("peak_allocated_bytes", 0))
            reset_memory_stats()
            first = session.prefill(prompt_ids, use_bulk=True, return_logits=True)
            logits = np.asarray(first.logits, dtype=np.float32)
            finite = bool(np.all(np.isfinite(logits)))
            next_token = int(first.token_id)
            for _ in range(int(args.decode_tokens) - 1):
                step = session.step(next_token, return_logits=True)
                finite = finite and bool(np.all(np.isfinite(np.asarray(step.logits, dtype=np.float32))))
                next_token = int(step.token_id)
            peak_bytes = int(memory_stats().get("peak_allocated_bytes", 0))
            result.update(
                {
                    "status": "pass" if finite else "nonfinite_logits",
                    "finite_logits": finite,
                    "first_token": int(first.token_id),
                    "tracked_peak_gib": round(process_peak_bytes / 2**30, 6),
                    "resident_gib": round(resident_bytes / 2**30, 6),
                    "resident_plus_transient_peak_gib": round(peak_bytes / 2**30, 6),
                    "transient_peak_gib": round((peak_bytes - resident_bytes) / 2**30, 6),
                    "scratch_max_positions": int(session.scratch.max_positions) if session.scratch else None,
                }
            )
    except Exception as exc:  # hip OOM surfaces as HipError; MemoryError for host paths
        message = str(exc)
        if "out of memory" in message.lower() or "oom" in message.lower():
            result["status"] = "oom"
            result["error"] = message[:200]
        else:
            raise

    payload = json.dumps(result, indent=2)
    print(payload)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(payload + "\n", encoding="utf-8")
    return 0 if result.get("status") == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
