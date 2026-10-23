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
The qwen4exp branch prepares the public serving pool and runs a short request
cohort, then checks full-vocabulary logits on its c1 runner. It certifies fixed
resident allocations and the exercised transient shapes, not arbitrary prefix
cache growth or all deep-context workspaces.
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
    parser.add_argument("--model", type=Path, default=Path("/models/gguf/Qwen3.8-27B-Q4_K_M.gguf"),
                        help="GGUF file or model directory; use the complete directory for split qwen4exp weights")
    parser.add_argument("--max-sequence-length", type=int, required=True)
    parser.add_argument("--kv-storage", default=None,
                        help="KV policy: qwen35 defaults to int8_per_token_head; qwen4exp to bf16")
    parser.add_argument("--execution-profile", default="production", choices=("strict", "production"),
                        help="Execution profile for the qwen4exp public-route probe")
    parser.add_argument("--sidecar", type=Path, help="qwen4exp MTP sidecar; omit for true AR")
    parser.add_argument("--candidate-budget", type=int, default=3, help="qwen4exp MTP candidate budget")
    parser.add_argument("--prefix-cache", choices=("off", "radix"), default="off",
                        help="qwen4exp public prefix-cache mode; short probes do not certify maximum cache growth")
    parser.add_argument("--kv-scale-dtype", default="fp32")
    parser.add_argument("--prompt-length", type=int, default=2048,
                        help="Short prompt (memory validity does not need the full context)")
    parser.add_argument("--decode-tokens", type=int, default=4)
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=1,
        help=(
            "Resident request capacity to probe. Match the server's configured "
            "request count; each model chooses its own physical state layout."
        ),
    )
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument(
        "--prefill-chunk-size",
        type=int,
        default=0,
        help=(
            "Prefill chunk rows to certify; 0 keeps the model's defaults. "
            "qwen35 applies this size to its scratch settings. qwen4exp requires "
            "the public route's selected chunk to match; it does not override admission."
        ),
    )
    args = parser.parse_args()
    if (min(args.max_sequence_length, args.prompt_length, args.decode_tokens, args.max_batch_size) <= 0
            or args.prefill_chunk_size < 0
            or args.prompt_length + args.decode_tokens - 1 > args.max_sequence_length):
        parser.error("positive context/prompt/output/width and a prompt plus decode within capacity are required")

    import numpy as np
    from hipengine.loading.gguf import GGUFReader, discover_gguf_files

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
    result["prefill_chunk_size"] = int(args.prefill_chunk_size)
    metadata_path = discover_gguf_files(args.model)[0] if args.model.is_dir() else args.model
    architecture = GGUFReader(metadata_path).info.architecture
    result["architecture"] = architecture
    args.kv_storage = args.kv_storage or ("bf16" if architecture == "qwen4exp" else "int8_per_token_head")
    result["kv_storage"] = args.kv_storage
    if args.sidecar is not None and architecture != "qwen4exp":
        parser.error("--sidecar is implemented only for the qwen4exp capacity probe")
    prompt_ids = [9707] * int(args.prompt_length)

    if architecture == "qwen4exp":
        if args.kv_storage not in ("auto", "bf16"):
            parser.error("qwen4exp implements BF16 KV storage, not " + args.kv_storage)
        result.update(kv_storage="bf16", command=[sys.executable, *sys.argv],
                      probe_implementation="qwen4exp_public_pool_and_c1_logits")
        _probe_qwen4exp(args, result, prompt_ids, np=np)
    elif architecture == "gemma4":
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
            f"tier-1 capacity branch for qwen4exp, gemma4, qwen35 and qwen35moe. Add a "
            f"branch for {architecture!r} rather than running the expensive "
            f"tier-2 point this probe exists to avoid."
        )

    payload = json.dumps(result, indent=2)
    print(payload)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(payload + "\n", encoding="utf-8")
    return 0 if result.get("status") == "pass" else 1


def _request_cohort(llm, prompt_ids, params, width):
    """Submit independent public requests, not an unsupported multi-prompt MTP group."""
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    ready = Barrier(width)
    def request():
        ready.wait(timeout=30)
        outputs = llm.generate_detailed([list(prompt_ids)], params)
        if len(outputs) != 1:
            raise ValueError("public capacity probe returned the wrong number of responses for one request")
        return outputs[0]

    with ThreadPoolExecutor(max_workers=width) as pool:
        pending = [pool.submit(request) for _ in range(width)]
        return [future.result() for future in pending]


def _probe_qwen4exp(args, result: dict, prompt_ids: list[int], *, np) -> None:
    """Allocate the public pool, exercise it, then check a full-vocabulary c1 row."""
    from hipengine import LLM, SamplingParams
    from hipengine.core.memory import memory_stats, reset_memory_stats

    llm = None
    before = memory_stats()
    result.update(kv_storage="bf16", execution_profile=args.execution_profile,
                  finite_logits_scope="separate c1 prefill and decode; public cohort checked by exact counts",
                  capacity_scope="fixed resident pool and exercised short-request transients; not maximum prefix-cache growth")
    try:
        options = dict(model=str(args.model), max_sequence_length=int(args.max_sequence_length),
                       max_active_requests=int(args.max_batch_size), execution_profile=args.execution_profile,
                       kv_storage="bf16", prefix_cache=args.prefix_cache)
        if args.sidecar is not None:
            options.update(speculative_provider="qwen4_exp_mtp", draft_model=str(args.sidecar),
                           speculative_candidate_budget=int(args.candidate_budget))
        llm = LLM(**options)
        params = SamplingParams(max_tokens=int(args.decode_tokens), temperature=0.0,
                                top_k=1, ignore_eos=True, kv_storage="bf16",
                                speculative_mtp=args.sidecar is not None)
        prepared = llm.prepare(max_sequence_length=int(args.max_sequence_length), sampling_params=params)
        if prepared != int(args.max_sequence_length):
            raise ValueError(f"requested capacity {args.max_sequence_length}, prepared {prepared}")
        result["scratch"] = llm.prepare_request_scratch(
            max_prompt_tokens=len(prompt_ids), max_new_tokens=int(args.decode_tokens),
            sampling_params=params, max_batch_size=int(args.max_batch_size), release_after_probe=False)
        generator = llm._get_text_generator()
        generator = getattr(generator, "_inner", generator)
        runner = generator.runner
        result["execution_profile_manifest_sha256"] = getattr(generator, "execution_profile_manifest_sha256", None)
        result.update(generator=type(generator).__name__, runner_capacity=int(runner.max_sequence_length),
                      prefill_chunk_size=int(runner.prefill_chunk_size), context_admission=generator.context_admission)
        if runner.max_sequence_length != int(args.max_sequence_length):
            raise ValueError("resident runner capacity differs from requested context")
        if args.prefill_chunk_size and args.prefill_chunk_size != runner.prefill_chunk_size:
            raise ValueError(f"public route selected chunk {runner.prefill_chunk_size}; cannot certify requested chunk {args.prefill_chunk_size}")
        resident_bytes = int(memory_stats().get("current_allocated_bytes", 0))
        load_peak = int(memory_stats().get("peak_allocated_bytes", 0))
        reset_memory_stats()
        result["request_protocol"] = "concurrent independent one-prompt LLM.generate_detailed calls"
        outputs = _request_cohort(llm, prompt_ids, params, args.max_batch_size)
        if len(outputs) != args.max_batch_size:
            raise ValueError("public capacity probe returned the wrong number of responses")
        result["responses"] = []
        for output in outputs:
            ids = list(output.generated_token_ids or ())
            diagnostics = {} if output.telemetry is None else dict(output.telemetry.diagnostics or {})
            prompt_count = output.prompt_tokens
            if prompt_count is None and output.telemetry is not None:
                prompt_count = output.telemetry.decode_state.prompt_tokens
            if len(ids) != args.decode_tokens or prompt_count != len(prompt_ids):
                raise ValueError("public capacity probe returned incomplete native token counts")
            used = diagnostics.get("speculative_provider") == "qwen4_exp_mtp"
            if used != (args.sidecar is not None):
                raise ValueError("public capacity probe did not execute its requested AR/MTP route")
            if used and args.decode_tokens > 1 and not diagnostics.get("proposed_draft_tokens", 0):
                raise ValueError("MTP probe has no draft-work witness")
            result["responses"].append(dict(generated_token_ids=ids, prompt_tokens=prompt_count,
                                            speculative_used=used, diagnostics=diagnostics))
        result["decode"] = generator.decode_telemetry()
        # Public requests have completed and released their leases. The idle
        # primary runner can now supply the separate full-logit sanity check.
        runner.reset()
        row = runner.prefill(prompt_ids, capture_logits=True, capture_hidden_seeds=False,
                             capture_target_hidden=False)
        first_token = int(row.token_id)
        finite = True
        for step in range(args.decode_tokens):
            logits = np.asarray(row.logits, dtype=np.float32)
            finite = finite and logits.size == runner.config.vocab_size and bool(np.all(np.isfinite(logits)))
            if step + 1 < args.decode_tokens:
                row = runner.step(row.token_id, capture_logits=True, capture_hidden_seed=False,
                                  capture_target_hidden=False)
        runner.reset()
        peak = int(memory_stats().get("peak_allocated_bytes", 0))
        result.update(status="pass" if finite else "nonfinite_logits", finite_logits=bool(finite),
                      first_token=first_token, tracked_peak_gib=max(load_peak, peak) / 2**30,
                      resident_gib=resident_bytes / 2**30,
                      resident_plus_transient_peak_gib=peak / 2**30,
                      transient_peak_gib=(peak - resident_bytes) / 2**30)
    except Exception as exc:
        result.update(status="oom" if isinstance(exc, MemoryError) or "out of memory" in str(exc).lower()
                      else "error", error=f"{type(exc).__name__}: {exc}")
    finally:
        if llm is not None:
            llm.close()
        after = memory_stats()
        result["memory_after_close"] = after
        if after.get("current_allocated_bytes", 0) != before.get("current_allocated_bytes", 0):
            result.update(status="leaked_allocations", error=result.get("error", "probe did not release its device allocations"))


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
    from dataclasses import replace

    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import memory_stats, reset_memory_stats
    from hipengine.kvcache import resolve_kv_policy
    from hipengine.runtime.prefill import PrefillConfig
    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession

    prefill_config = PrefillConfig()
    if int(args.prefill_chunk_size) > 0:
        # The chunk lever's whole cost is activation scratch, so the probe has
        # to be able to certify a non-default chunk: every chunk-sized knob is
        # raised together, because the runner takes the smallest positive one.
        chunk = int(args.prefill_chunk_size)
        prefill_config = replace(
            prefill_config,
            linear_chunk_size=chunk,
            moe_chunk_size=chunk,
            full_attn_post_chunk_size=chunk,
            full_attn_rope_chunk_size=chunk,
        )

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
            prefill_config=prefill_config,
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
