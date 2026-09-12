#!/usr/bin/env python3
"""README timing protocol using a session configured by the public profile."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hipengine.benchmark.provenance import collect_artifact_provenance
from scripts.qwen35_gguf_bench import (
    _RoctxProfilerControl,
    _default_decode_graph_request,
    _run_existing_session_once,
    _summary,
)
from scripts.qwen38_production_ar_gate import profile_session


def run_workload(session, args, prompt_length):
    runs = []
    graph_requested = _default_decode_graph_request(session, args.decode_tokens)
    for index in range(args.warmups + args.repetitions):
        measured = index >= args.warmups
        graph = bool(graph_requested and measured)
        holder = {} if graph else None
        try:
            run = _run_existing_session_once(
                session=session, runtime=session.runtime, model=Path(args.model),
                quant="gguf_q4_k_m", prompt_tokens=[9707] * prompt_length,
                decode_tokens=args.decode_tokens, warmup_decode_tokens=1,
                graph_replay_decode=graph, graph_steps_per_replay=1,
                use_bulk_prefill=True, bulk_attention_mode="bulk",
                use_wmma_prefill=True, use_gemv_decode=True,
                prefill_chunk_size=0, measured=measured, run_index=index + 1,
                load_seconds=0.0, persistent_session=True, graph_holder=holder,
                roctx=_RoctxProfilerControl(enabled=False), rocprof_selected_region="none",
            )
            if measured:
                runs.append(run)
            print(f"p{prompt_length}/d{args.decode_tokens} run {index + 1}", flush=True)
        finally:
            if holder is not None and holder.get("graph") is not None:
                holder["graph"].close()
    return runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("/models/gguf/Qwen3.8-27B-Q4_K_M.gguf"))
    parser.add_argument("--prompt-lengths", type=int, nargs="+", default=[512, 1024, 4096])
    parser.add_argument("--decode-tokens", type=int, default=128)
    parser.add_argument("--max-sequence-length", type=int, default=8192)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (min(args.prompt_lengths) < 1 or args.decode_tokens < 1 or args.warmups < 1
            or args.repetitions < 3
            or max(args.prompt_lengths) + args.decode_tokens + 1 >= args.max_sequence_length):
        parser.error("requires valid context, one warmup and at least three measured repetitions")
    with profile_session(args, None) as (session, profile):
        runs = {str(length): run_workload(session, args, length)
                for length in args.prompt_lengths}
    summaries = {length: _summary(values) for length, values in runs.items()}
    provenance = collect_artifact_provenance(
        repo_root=ROOT, configured_backend="hip_gfx1151", resolved_backend="hip_gfx1151",
        target_arch="gfx1151", model_path=args.model, quant="gguf_q4_k_m", kv_dtype="bf16",
        command=[sys.executable, *sys.argv],
        environment={k: v for k, v in os.environ.items()
                     if k.startswith(("HIPENGINE_", "GPU_MAX_HW_QUEUES"))},
        build_profile="public_profile_readme_sweep",
        timing_protocol="one_resident_session_per_shape_warmup_then_median_three",
        warmups=args.warmups, repetitions=args.repetitions, profiler={"enabled": False},
    )
    payload = {
        "kind": "qwen38_gfx1151_public_profile_readme_sweep", "schema_version": 1,
        "profile": profile, "provenance": provenance, "runs": runs,
        "summaries": summaries, "decode_tokens": args.decode_tokens,
        "load_timing": "not measured; per-run load_seconds is zero by protocol",
        "performance_claim": False,
        "limitation": "Publication requires separately linked profile quality and serving gates",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summaries, indent=2))
    return 0 if all(row["finite_final_logits_all"] for row in summaries.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
