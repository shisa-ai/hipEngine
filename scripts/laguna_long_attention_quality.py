#!/usr/bin/env python3
"""Teacher-force two Laguna long-attention routes from the same model state."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

from hipengine.benchmark.provenance import collect_artifact_provenance
from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import copy_device_to_host, host_array_ptr, memory_stats
from hipengine.loading.gguf import GGUFReader
from hipengine.runtime.laguna_gguf_runner import (
    LagunaEagerTokenResult,
    LagunaGGUFResidentSession,
)
from hipengine.tokenization.gguf import LagunaGGUFTokenizer
from scripts.laguna_long_context_profile import _profile_token_stream
from scripts.laguna_target_ar_bench import (
    DEFAULT_CACHE,
    DEFAULT_MODEL,
    DEFAULT_MODEL_SHA256,
    DEFAULT_PROMPTS,
    _compiler_version,
    _load_prompts,
    _progress,
    _repo_state,
    _sha256_json,
)

ROOT = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--backend", default="hip_gfx1151")
    parser.add_argument("--context-length", type=int, default=131_200)
    parser.add_argument("--length", type=int, default=16_384)
    parser.add_argument(
        "--prompt-id",
        help="canonical prompt to extend; defaults to the longest prompt",
    )
    parser.add_argument("--teacher-steps", type=int, default=127)
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--repacked-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--model-sha256", default=DEFAULT_MODEL_SHA256)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _copy_logits(
    session: LagunaGGUFResidentSession,
    result: LagunaEagerTokenResult,
) -> np.ndarray:
    logits = np.empty(session.config.vocab_size, dtype=np.float32)
    copy_device_to_host(
        host_array_ptr(logits),
        result.logits,
        runtime=session.runtime,
    )
    return logits


def _distribution_metrics(
    control: np.ndarray,
    candidate: np.ndarray,
) -> dict[str, float | int | bool]:
    finite = bool(np.isfinite(control).all() and np.isfinite(candidate).all())
    if not finite:
        return {
            "finite": False,
            "kl_divergence": math.inf,
            "control_top1": -1,
            "candidate_top1": -1,
            "top1_match": False,
            "max_abs_logit_delta": math.inf,
        }
    control64 = control.astype(np.float64)
    candidate64 = candidate.astype(np.float64)
    control_shift = control64 - float(np.max(control64))
    candidate_shift = candidate64 - float(np.max(candidate64))
    control_exp = np.exp(control_shift)
    control_log_z = math.log(float(np.sum(control_exp)))
    candidate_log_z = math.log(float(np.sum(np.exp(candidate_shift))))
    control_log_probs = control_shift - control_log_z
    candidate_log_probs = candidate_shift - candidate_log_z
    kl = float(
        np.sum(control_exp / float(np.sum(control_exp)) * (
            control_log_probs - candidate_log_probs
        ))
    )
    control_top1 = int(np.argmax(control))
    candidate_top1 = int(np.argmax(candidate))
    return {
        "finite": True,
        "kl_divergence": kl,
        "control_top1": control_top1,
        "candidate_top1": candidate_top1,
        "top1_match": control_top1 == candidate_top1,
        "max_abs_logit_delta": float(np.max(np.abs(control - candidate))),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.length <= 6000:
        raise ValueError("quality screen must exercise the long global-attention route")
    if args.teacher_steps <= 0:
        raise ValueError("teacher steps must be positive")
    if args.context_length < args.length + args.teacher_steps:
        raise ValueError("prompt plus teacher-forced horizon exceeds context length")
    repo = _repo_state()
    if not args.allow_dirty and not repo["tracked_clean"]:
        raise RuntimeError("tracked worktree is dirty; pass --allow-dirty for development")

    runtime = get_hip_runtime()
    provenance = collect_artifact_provenance(
        repo_root=ROOT,
        configured_backend=args.backend,
        resolved_backend=args.backend,
        target_arch=args.backend.removeprefix("hip_"),
        model_path=args.model,
        quant="gguf_q4_k_m",
        kv_dtype="bf16",
        command=(str(Path(sys.executable).resolve()), *sys.argv),
        build_profile="laguna_long_attention_teacher_forced_quality",
        timing_protocol=f"p{args.length}_teacher{args.teacher_steps}_c1",
        warmups=0,
        repetitions=1,
    )
    reader = GGUFReader(args.model)
    tokenizer = LagunaGGUFTokenizer.from_gguf_info(reader.info)
    prompts = _load_prompts(args.prompts, tokenizer)
    prompt_pool = prompts
    if args.prompt_id is not None:
        prompt_pool = tuple(
            prompt for prompt in prompts if str(prompt.get("id")) == args.prompt_id
        )
        if not prompt_pool:
            raise ValueError(f"unknown prompt id: {args.prompt_id}")
    token_stream, token_source = _profile_token_stream(prompt_pool, args.length)
    tracked_before = memory_stats()
    free_before, total_bytes = runtime.mem_get_info()
    control: LagunaGGUFResidentSession | None = None
    candidate: LagunaGGUFResidentSession | None = None
    started = time.perf_counter()
    steps: list[dict[str, float | int | bool]] = []
    try:
        control = LagunaGGUFResidentSession(
            args.model,
            context_length=args.context_length,
            backend=args.backend,
            runtime=runtime,
            compiler_version=_compiler_version(args.compiler_version_file),
            require_cached_build=args.require_cached_build,
            progress=_progress,
            repacked_cache=args.repacked_cache,
            model_sha256=args.model_sha256,
            prefill_chunk_size=args.chunk_size,
        )
        assert control.weights is not None
        candidate = LagunaGGUFResidentSession(
            resident_weights=control.weights,
            context_length=args.context_length,
            backend=args.backend,
            runtime=runtime,
            compiler_version=_compiler_version(args.compiler_version_file),
            require_cached_build=args.require_cached_build,
            prefill_chunk_size=args.chunk_size,
        )
        control.kv_cache.global_split_gqa6_ctx4096_min_layer = None
        control.kv_cache.global_split_gqa6_ctx4096_compensated_layer = None
        candidate.kv_cache.global_split_gqa6_ctx4096_min_layer = 32
        candidate.kv_cache.global_split_gqa6_ctx4096_compensated_layer = 28
        control_result = control.prefill(token_stream, use_bulk=True)
        candidate_result = candidate.prefill(token_stream, use_bulk=True)
        if int(control_result.next_token_id) != int(candidate_result.next_token_id):
            raise RuntimeError("decode-only attention routes changed prefill output")
        teacher_token = int(control_result.next_token_id)
        for step in range(args.teacher_steps):
            control_result = control.forward_token(teacher_token)
            candidate_result = candidate.forward_token(teacher_token)
            control_logits = _copy_logits(control, control_result)
            candidate_logits = _copy_logits(candidate, candidate_result)
            metrics = _distribution_metrics(control_logits, candidate_logits)
            metrics.update(
                {
                    "step": step,
                    "input_token_id": teacher_token,
                    "control_session_top1": int(control_result.next_token_id),
                    "candidate_session_top1": int(candidate_result.next_token_id),
                }
            )
            steps.append(metrics)
            teacher_token = int(control_result.next_token_id)
    finally:
        if candidate is not None:
            candidate.close()
        if control is not None:
            control.close()
    elapsed = time.perf_counter() - started
    tracked_after = memory_stats()
    free_after, total_after = runtime.mem_get_info()
    if total_after != total_bytes:
        raise RuntimeError("HIP total memory changed during quality screen")

    finite = all(bool(step["finite"]) for step in steps)
    max_kl = max(float(step["kl_divergence"]) for step in steps)
    mean_kl = sum(float(step["kl_divergence"]) for step in steps) / len(steps)
    top1_matches = sum(bool(step["top1_match"]) for step in steps)
    top1_agreement = top1_matches / len(steps)
    lifecycle_ok = (
        tracked_after["current_allocated_bytes"]
        == tracked_before["current_allocated_bytes"]
        and tracked_after["active_allocations"]
        == tracked_before["active_allocations"]
    )
    passed = finite and max_kl <= 0.05 and top1_agreement >= 0.90 and lifecycle_ok
    return {
        "schema": 1,
        "kind": "hipengine_laguna_long_attention_teacher_forced_quality",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": {
            "path": str(args.model),
            "sha256": args.model_sha256,
            "repacked_cache": str(args.repacked_cache),
        },
        "platform": {
            "backend": args.backend,
            "device_name": provenance["device_name"],
            "hip_total_bytes": total_bytes,
        },
        "provenance": provenance,
        "protocol": {
            "length": args.length,
            "teacher_steps": args.teacher_steps,
            "context_length": args.context_length,
            "chunk_size": args.chunk_size,
            "token_source": token_source,
            "control_route": "exact_gqa6_deferrednorm_dim32_vstage64",
            "candidate_route": (
                "ctx4096_compensated_layer28_plus_retained_uncompensated_"
                "layers32_36_40_44"
            ),
        },
        "quality": {
            "finite": finite,
            "max_kl_divergence": max_kl,
            "mean_kl_divergence": mean_kl,
            "top1_matches": top1_matches,
            "top1_agreement": top1_agreement,
            "max_abs_logit_delta": max(
                float(step["max_abs_logit_delta"]) for step in steps
            ),
            "control_top1_sha256": _sha256_json(
                [int(step["control_top1"]) for step in steps]
            ),
            "candidate_top1_sha256": _sha256_json(
                [int(step["candidate_top1"]) for step in steps]
            ),
            "divergent_steps": [
                int(step["step"]) for step in steps if not step["top1_match"]
            ],
            "thresholds": {
                "max_kl_divergence": 0.05,
                "minimum_top1_agreement": 0.90,
            },
        },
        "steps": steps,
        "memory": {
            "gpu_free_before": free_before,
            "gpu_free_after": free_after,
            "tracked_before": tracked_before,
            "tracked_after": tracked_after,
            "tracked_returned_to_baseline": lifecycle_ok,
        },
        "elapsed_seconds": elapsed,
        "repo": repo,
        "pass": passed,
    }


def main() -> None:
    args = _parse_args()
    artifact = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(artifact, indent=2))
    if not artifact["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
