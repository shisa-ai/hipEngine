#!/usr/bin/env python3
"""Named-profile AR numerical gate; separate from MTP and HTTP qualification."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hipengine.benchmark.provenance import collect_artifact_provenance
from scripts.execution_profile_gdn_calibration import (
    PromptCalibrationCapture,
    build_candidate_quality,
)
from scripts.execution_profile_gguf_c1_route_gate import (
    _state_summary,
    build_state_repeat_gate,
)
from scripts.execution_profile_gguf_fp16_state_gate import (
    _run_logits_trajectory,
    _run_teacher_forced_candidate,
)
from scripts.gguf_gdn_semantic_gate import DEFAULT_PROMPTS, _load_suites
from scripts.gguf_mtp_bench import build_chat_prompt
from scripts.gguf_mtp_category_bench import prompt_sha256


def profile_identity(llm, generator, session, *, requested):
    expected = "production" if requested is None else requested
    if (str(generator.execution_profile) != expected
            or generator.execution_profile_fell_back_to_strict):
        raise ValueError("unexpected resolved profile or strict fallback")
    digest = llm.execution_profile_manifest_sha256
    if not digest or digest != generator.execution_profile_manifest_sha256:
        raise ValueError("LLM and generator manifest identities differ")
    if str(session.kv_storage_dtype) != "bf16":
        raise ValueError("gate requires actual BF16 KV storage")
    return {
        "requested": requested,
        "resolved": str(generator.execution_profile),
        "manifest_sha256": digest,
        "manifest": llm.execution_profile_manifest,
        "fp16_recurrent_state": bool(session.runner.fp16_recurrent_state),
        "kv_storage_dtype": str(session.kv_storage_dtype),
    }


@contextmanager
def profile_session(args, requested):
    from hipengine import LLM

    llm = LLM(str(args.model), backend="hip_gfx1151", quant="gguf_q4_k_m",
              execution_profile=requested, max_active_requests=1,
              max_sequence_length=args.max_sequence_length)
    try:
        llm.prepare(max_sequence_length=args.max_sequence_length)
        generator = llm._get_text_generator()
        with generator._resident_session_scope(
            shared_runner=generator._get_shared_runner(),
            pool_name="production_ar_qualification",
        ) as (session, _reused):
            yield session, profile_identity(
                llm, generator, session, requested=requested)
    finally:
        llm.close()


def run(args):
    from hipengine.loading.gguf import scan_gguf
    from hipengine.tokenization.gguf import Qwen35GGUFTokenizer

    rows = _load_suites(DEFAULT_PROMPTS)
    tokenizer = Qwen35GGUFTokenizer.from_gguf_info(scan_gguf(args.model))
    tokens = {row["id"]: build_chat_prompt(tokenizer, str(row["prompt"]))
              for row in rows}
    if max(map(len, tokens.values())) + args.decode_steps >= args.max_sequence_length:
        raise ValueError("declared context cannot hold prompt and decode horizon")
    strict, strict_states, candidates, candidate_states = {}, [], {}, []
    with profile_session(args, "strict") as (session, strict_profile):
        for row in rows:
            key = row["id"]
            trajectory = tuple(_run_logits_trajectory(
                session, prompt_ids=tokens[key], decode_steps=args.decode_steps,
                bulk_attention_mode="bulk"))
            strict[key] = trajectory
            forced = [step["token_id"] for step in trajectory[:-1]]
            state = _state_summary(session, trajectory, forced)
            strict_states.append(dict(state, prompt_id=key))
            print(f"strict {key}: {len(trajectory)} rows", flush=True)
    with profile_session(args, None) as (session, candidate_profile):
        for row in rows:
            key = row["id"]
            forced = [step["token_id"] for step in strict[key][:-1]]
            runs, states = [], []
            for repeat in range(args.repeat_runs):
                trajectory = tuple(_run_teacher_forced_candidate(
                    session, prompt_ids=tokens[key], forced_input_ids=forced,
                    bulk_attention_mode="bulk"))
                runs.append(trajectory)
                states.append(dict(_state_summary(session, trajectory, forced),
                                   prompt_id=key))
                print(f"default {key}: repeat {repeat + 1}", flush=True)
            candidates[key] = tuple(runs)
            candidate_states.append(states)
    captures = tuple(PromptCalibrationCapture(
        prompt_id=row["id"], category=row["category"], strict=strict[row["id"]],
        candidate_runs={"production": candidates[row["id"]]},
    ) for row in rows)
    quality = build_candidate_quality(
        captures, candidate_mode="production", scenario_id="qwen38-default-ar")
    state_gate = build_state_repeat_gate(strict_states, candidate_states)
    provenance = collect_artifact_provenance(
        repo_root=ROOT, configured_backend="hip_gfx1151",
        resolved_backend="hip_gfx1151", target_arch="gfx1151",
        model_path=args.model, quant="gguf_q4_k_m", kv_dtype="bf16",
        command=[sys.executable, *sys.argv], environment=dict(
            (key, value) for key, value in os.environ.items()
            if key.startswith(("HIPENGINE_", "GPU_MAX_HW_QUEUES"))),
        build_profile="named_profile_ar_gate", timing_protocol="quality_only",
        warmups=0, repetitions=args.repeat_runs,
        profiler={"enabled": False},
    )
    passed = bool(quality["quality"]["hard_gates_passed"]
                  and quality["repeat_determinism"]["passed"]
                  and state_gate["passed"])
    return {
        "kind": "qwen38_named_default_ar_numerics", "schema_version": 1,
        "passed": passed, "performance_claim": False,
        "full_production_qualification": False,
        "limitations": ["C1 eager AR only; HTTP, graphs, and MTP gated separately",
                        "No BF16-weight teacher available in this gate"],
        "strict_profile": strict_profile, "candidate_profile": candidate_profile,
        "decode_steps": args.decode_steps, "repeat_runs": args.repeat_runs,
        "teacher_rows": sum(len(value) for value in strict.values()),
        "prompts": [dict(id=row["id"], category=row["category"], suite=row["suite"],
                         prompt_sha256=prompt_sha256(str(row["prompt"])))
                    for row in rows],
        "quality": quality, "state_repeat_gate": state_gate,
        "provenance": provenance,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path,
                        default=Path("/models/gguf/Qwen3.8-27B-Q4_K_M.gguf"))
    parser.add_argument("--decode-steps", type=int, default=128)
    parser.add_argument("--repeat-runs", type=int, default=3)
    parser.add_argument("--max-sequence-length", type=int, default=2048)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.decode_steps < 32 or args.repeat_runs < 3:
        parser.error("requires at least 32 decode steps and three repeats")
    payload = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"passed": payload["passed"], "output": str(args.output)}))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
