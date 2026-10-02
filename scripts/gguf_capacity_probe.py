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

The probe routes on the artifact's ``general.architecture``. The qwen35
branch drives ``Qwen35GGUFResidentSession`` directly, which is the
capacity route that family's ladders were measured on. The gemma4 branch
resolves the same generator ``hipengine.LLM`` would and drives its runner,
so the envelope it certifies is the one a user reaches. Adding a branch
here is how a new architecture gets a tier-1 answer; without one the
protocol's only option is the expensive tier-2 point it exists to avoid.
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
    from hipengine.loading.gguf import GGUFReader

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
    architecture = GGUFReader(args.model).info.architecture
    result["architecture"] = architecture
    prompt_ids = [9707] * int(args.prompt_length)

    if architecture == "gemma4":
        # The gemma4 branch certifies the envelope ``LLM`` creates, so the
        # generator's own KV route applies and ``--kv-storage`` is not used.
        # Recording that here keeps the artifact from claiming a policy the
        # probe did not select.
        result["kv_storage"] = None
        result["kv_storage_note"] = "the generator's own route; --kv-storage does not apply to gemma4"
        _probe_gemma4(args, result, prompt_ids, np=np)
    elif architecture in ("qwen35", "qwen35moe"):
        _probe_qwen35(args, result, prompt_ids, np=np)
    else:
        # Named, and named before any allocation: falling through to the
        # qwen35 branch would raise a confusing "expected architecture" error
        # from deep inside a loader that was never going to serve this file.
        raise SystemExit(
            f"unsupported GGUF architecture {architecture!r}: this probe has a "
            f"tier-1 capacity branch for gemma4, qwen35 and qwen35moe. Add a "
            f"branch for {architecture!r} rather than running the expensive "
            f"tier-2 point this probe exists to avoid."
        )

    payload = json.dumps(result, indent=2)
    print(payload)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(payload + "\n", encoding="utf-8")
    return 0 if result.get("status") == "pass" else 1


def _probe_gemma4(args, result: dict, prompt_ids: list[int], *, np) -> None:
    """Certify a Gemma 4 envelope by the same short-prompt route.

    Everything the declared context sizes -- weights, the per-layer KV
    planes, and the persistent scratch -- is acquired when the runner is
    built, not when the prompt arrives, so a short prompt at the target
    ``max_sequence_length`` exercises the same allocation the full-length
    point would. The generator is resolved through ``LLM`` rather than
    built by hand so the probe certifies the envelope the public surface
    actually creates, including the execution profile's variant selection.
    """
    import hipengine

    from hipengine.core.memory import memory_stats, reset_memory_stats

    llm = hipengine.LLM(
        model=str(args.model),
        max_sequence_length=int(args.max_sequence_length),
        max_active_requests=int(args.max_batch_size),
    )
    generator = llm._get_text_generator()
    # ``LLM`` wraps the model generator in a scheduler adapter; the runner
    # lives on the inner object.
    inner = getattr(generator, "_inner", None)
    if inner is not None:
        generator = inner
    result["generator"] = type(generator).__name__
    result["execution_profile"] = getattr(generator, "execution_profile", None)

    try:
        runner = generator._ensure_runner()
        resident_bytes = int(memory_stats().get("current_allocated_bytes", 0))
        process_peak_bytes = int(memory_stats().get("peak_allocated_bytes", 0))
        reset_memory_stats()

        logits = np.asarray(runner.forward(prompt_ids), dtype=np.float32)
        finite = bool(np.all(np.isfinite(logits)))
        next_token = int(np.argmax(logits))
        for _ in range(int(args.decode_tokens) - 1):
            logits = np.asarray(runner.forward([next_token]), dtype=np.float32)
            finite = finite and bool(np.all(np.isfinite(logits)))
            next_token = int(np.argmax(logits))

        peak_bytes = int(memory_stats().get("peak_allocated_bytes", 0))
        result.update(
            {
                "status": "pass" if finite else "nonfinite_logits",
                "finite_logits": finite,
                "first_token": next_token,
                "tracked_peak_gib": round(process_peak_bytes / 2**30, 6),
                "resident_gib": round(resident_bytes / 2**30, 6),
                "resident_plus_transient_peak_gib": round(peak_bytes / 2**30, 6),
                "transient_peak_gib": round((peak_bytes - resident_bytes) / 2**30, 6),
                "runner_capacity": int(runner.capacity),
                "layers": len(getattr(runner, "_scratches", ())),
            }
        )
    except Exception as exc:  # hip OOM surfaces as HipError; MemoryError for host paths
        message = str(exc)
        if "out of memory" in message.lower() or "oom" in message.lower():
            result["status"] = "oom"
            result["error"] = message[:200]
        else:
            raise


def _probe_qwen35(args, result: dict, prompt_ids: list[int], *, np) -> None:
    """Certify a qwen35 envelope through its resident session."""
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import memory_stats, reset_memory_stats
    from hipengine.kvcache import resolve_kv_policy
    from hipengine.runtime.prefill import PrefillConfig
    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession

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


if __name__ == "__main__":
    raise SystemExit(main())
