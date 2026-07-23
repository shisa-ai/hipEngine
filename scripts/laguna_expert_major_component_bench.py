#!/usr/bin/env python3
"""Bisect Laguna expert-major Q4 gate/up versus Q4/Q6 down quality and wall."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np

from hipengine.benchmark.provenance import collect_artifact_provenance
from hipengine.core.hip import HipMemcpyKind, get_hip_runtime
from hipengine.core.memory import host_array_ptr, memory_stats
from hipengine.loading.gguf import GGUFReader
from hipengine.runtime.laguna_gguf_runner import LagunaGGUFResidentSession
from hipengine.tokenization.gguf import LagunaGGUFTokenizer
from scripts.laguna_grouped_down_category_bench import _teacher_forced_quality
from scripts.laguna_target_ar_bench import (
    DEFAULT_CACHE,
    DEFAULT_MODEL,
    DEFAULT_MODEL_SHA256,
    DEFAULT_ORACLE,
    DEFAULT_ORACLE_LOGPROBS,
    DEFAULT_PROMPTS,
    DEFAULT_TEMPLATE,
    _compiler_version,
    _load_prompts,
    _normalized_log_probs,
    _progress,
    _repo_state,
    _session,
    _sha256_bytes,
)

ROOT = Path(__file__).resolve().parents[1]
BASELINE_MODE = "adaptive_grouped_smallm_fused"
CANDIDATE_MODES = (
    "adaptive_expert_major_gate_up_comp",
    "adaptive_expert_major_down_comp",
    "adaptive_expert_major_wmma_comp",
)
MODES = (BASELINE_MODE, *CANDIDATE_MODES)
DEFAULT_SOURCE_REJECTION = (
    ROOT
    / "benchmarks/results/2026-07-24-gfx1151-laguna-expert-major-wmma-category-rejected.json"
)
DEFAULT_OUTPUT = Path("/tmp/laguna-expert-major-component-bisection.raw.json")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--oracle", type=Path, default=DEFAULT_ORACLE)
    parser.add_argument("--oracle-logprobs", type=Path, default=DEFAULT_ORACLE_LOGPROBS)
    parser.add_argument("--source-rejection", type=Path, default=DEFAULT_SOURCE_REJECTION)
    parser.add_argument("--backend", default="hip_gfx1151")
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--teacher-forced-tokens", type=int, default=32)
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--repacked-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--model-sha256", default=DEFAULT_MODEL_SHA256)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _mode_order(prompt_index: int, repetition: int) -> tuple[str, ...]:
    offset = (int(prompt_index) + int(repetition)) % len(MODES)
    return MODES[offset:] + MODES[:offset]


def _expanded_token_ids(prompt: Mapping[str, Any], target: int = 128) -> tuple[int, ...]:
    token_ids = tuple(int(value) for value in prompt["token_ids"])
    if len(token_ids) >= target:
        return token_ids[:target]
    repeat = token_ids[1:] if len(token_ids) > 1 else token_ids
    if not repeat:
        raise ValueError("cannot expand an empty category prompt")
    expanded = list(token_ids)
    while len(expanded) < target:
        expanded.extend(repeat[: target - len(expanded)])
    return tuple(expanded)


def _load_source_rejection(args: argparse.Namespace) -> dict[str, Any]:
    artifact = json.loads(args.source_rejection.read_text(encoding="utf-8"))
    passed = bool(
        artifact.get("kind") == "hipengine_laguna_prefill_expert_major_wmma_category"
        and artifact.get("status") == "rejected_category_gate"
        and artifact.get("pass") is False
        and artifact.get("model", {}).get("sha256") == args.model_sha256
        and artifact.get("quality", {})
        .get("teacher_forced", {})
        .get("max_kl_divergence", 0.0)
        > 0.05
    )
    if not passed:
        raise ValueError("source expert-major category rejection is not accepted")
    return {
        "pass": True,
        "path": str(args.source_rejection.resolve()),
        "sha256": _sha256_bytes(args.source_rejection.read_bytes()),
        "revision": artifact.get("repo", {}).get("revision"),
        "max_kl_divergence": artifact["quality"]["teacher_forced"][
            "max_kl_divergence"
        ],
        "prefill_speedup": artifact["performance"]["candidate_vs_retained"][
            "prefill_speedup"
        ],
    }


def _open_mode_session(
    owner: LagunaGGUFResidentSession,
    args: argparse.Namespace,
    mode: str,
):
    if mode not in MODES:
        raise ValueError(f"unknown expert-major component mode {mode!r}")
    session = _session(owner, args)
    session.set_selected_down_mode(mode)
    return session


def _copy_logits(session, result, destination: np.ndarray) -> None:
    session.runtime.memcpy(
        host_array_ptr(destination),
        result.logits.ptr,
        destination.nbytes,
        HipMemcpyKind.DEVICE_TO_HOST,
    )


def _poolside_oracle(
    owner: LagunaGGUFResidentSession,
    args: argparse.Namespace,
    mode: str,
) -> dict[str, Any]:
    template = json.loads(args.template.read_text(encoding="utf-8"))
    oracle = json.loads(args.oracle.read_text(encoding="utf-8"))
    prompt_case = next(
        case for case in template["cases"] if case["name"] == oracle["prompt"]["case"]
    )
    prompt_ids = tuple(int(value) for value in prompt_case["token_ids"])
    session = _open_mode_session(owner, args, mode)
    try:
        result = session.prefill(prompt_ids, use_bulk=True)
        logits = np.empty(session.config.vocab_size, dtype=np.float32)
        _copy_logits(session, result, logits)
    finally:
        session.close()
    reference = _normalized_log_probs(np.load(args.oracle_logprobs, allow_pickle=False))
    candidate = _normalized_log_probs(logits)
    probabilities = np.exp(reference)
    kl = float(np.sum(probabilities * (reference - candidate)))
    candidate_top1 = int(np.argmax(logits))
    reference_top1 = int(oracle["first_token"]["id"])
    finite = bool(np.isfinite(logits).all())
    return {
        "pass": bool(finite and kl <= 0.05 and candidate_top1 == reference_top1),
        "mode": mode,
        "prompt_tokens": len(prompt_ids),
        "kl_divergence": kl,
        "kl_threshold": 0.05,
        "candidate_top1": candidate_top1,
        "reference_top1": reference_top1,
        "top1_agreement": float(candidate_top1 == reference_top1),
        "top1_threshold": 0.9,
        "finite_logits": finite,
        "oracle_artifact": str(args.oracle.resolve()),
        "oracle_distribution": str(args.oracle_logprobs.resolve()),
    }


def _time_prefill(
    owner: LagunaGGUFResidentSession,
    prompt: Mapping[str, Any],
    args: argparse.Namespace,
    *,
    mode: str,
    repetition: int,
) -> dict[str, Any]:
    session = _open_mode_session(owner, args, mode)
    token_ids = _expanded_token_ids(prompt)
    try:
        started = time.perf_counter()
        session.prefill(token_ids, use_bulk=True)
        seconds = time.perf_counter() - started
    finally:
        session.close()
    return {
        "prompt_id": prompt["id"],
        "category": prompt["category"],
        "mode": mode,
        "repetition": int(repetition),
        "prompt_tokens": len(token_ids),
        "source_prompt_tokens": int(prompt["prompt_tokens"]),
        "prompt_token_ids_sha256": _sha256_bytes(
            json.dumps(token_ids, separators=(",", ":")).encode()
        ),
        "prefill_seconds": seconds,
        "prefill_tok_s": len(token_ids) / seconds,
    }


def _teacher_forced_prompt(
    owner: LagunaGGUFResidentSession,
    prompt: Mapping[str, Any],
    args: argparse.Namespace,
) -> dict[str, list[dict[str, Any]]]:
    sessions = {mode: _open_mode_session(owner, args, mode) for mode in MODES}
    logits = {
        mode: np.empty(sessions[mode].config.vocab_size, dtype=np.float32)
        for mode in MODES
    }
    token_ids = _expanded_token_ids(prompt)
    candidate_steps = {mode: [] for mode in CANDIDATE_MODES}
    try:
        results = {
            mode: sessions[mode].prefill(token_ids, use_bulk=True) for mode in MODES
        }
        for index in range(args.teacher_forced_tokens):
            for mode in MODES:
                _copy_logits(sessions[mode], results[mode], logits[mode])
            baseline_logp = _normalized_log_probs(logits[BASELINE_MODE])
            probabilities = np.exp(baseline_logp)
            baseline_top1 = int(np.argmax(logits[BASELINE_MODE]))
            for mode in CANDIDATE_MODES:
                candidate_logp = _normalized_log_probs(logits[mode])
                kl = float(
                    np.sum(probabilities * (baseline_logp - candidate_logp))
                )
                candidate_top1 = int(np.argmax(logits[mode]))
                candidate_steps[mode].append(
                    {
                        "index": index,
                        "teacher_token_id": baseline_top1,
                        "top1_agreement": baseline_top1 == candidate_top1,
                        "kl_divergence": kl,
                        "finite": bool(
                            np.isfinite(logits[BASELINE_MODE]).all()
                            and np.isfinite(logits[mode]).all()
                            and math.isfinite(kl)
                        ),
                    }
                )
            if index + 1 < args.teacher_forced_tokens:
                results = {
                    mode: sessions[mode].forward_token(baseline_top1) for mode in MODES
                }
    finally:
        for session in sessions.values():
            session.close()
    return {
        mode: [
            {
                "prompt_id": prompt["id"],
                "category": prompt["category"],
                "prompt_tokens": len(token_ids),
                "source_prompt_tokens": int(prompt["prompt_tokens"]),
                "prompt_token_ids_sha256": _sha256_bytes(
                    json.dumps(token_ids, separators=(",", ":")).encode()
                ),
                "steps": candidate_steps[mode],
            }
        ]
        for mode in CANDIDATE_MODES
    }


def _aggregate_performance(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    modes: dict[str, Any] = {}
    for mode in MODES:
        selected = [row for row in rows if row["mode"] == mode]
        total_tokens = sum(int(row["prompt_tokens"]) for row in selected)
        total_seconds = float(sum(float(row["prefill_seconds"]) for row in selected))
        categories = {}
        for category in ("code", "general_en", "general_ja", "mixed_ja_en"):
            category_rows = [row for row in selected if row["category"] == category]
            category_tokens = sum(int(row["prompt_tokens"]) for row in category_rows)
            category_seconds = float(
                sum(float(row["prefill_seconds"]) for row in category_rows)
            )
            categories[category] = {
                "prompt_tokens": category_tokens,
                "prefill_seconds": category_seconds,
                "prefill_tok_s": category_tokens / category_seconds,
            }
        modes[mode] = {
            "prompt_tokens": total_tokens,
            "prefill_seconds": total_seconds,
            "prefill_tok_s": total_tokens / total_seconds,
            "categories": categories,
        }
    baseline_seconds = float(modes[BASELINE_MODE]["prefill_seconds"])
    return {
        "modes": modes,
        "speedups_vs_retained": {
            mode: baseline_seconds / float(modes[mode]["prefill_seconds"])
            for mode in CANDIDATE_MODES
        },
    }


def _evaluate(
    performance: Mapping[str, Any],
    quality_by_mode: Mapping[str, Mapping[str, Any]],
    oracle_by_mode: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    mode_results = {}
    passing = []
    for mode, quality in quality_by_mode.items():
        failed = []
        if not quality.get("pass"):
            failed.append("teacher_forced_quality_failed")
        if float(performance["speedups_vs_retained"][mode]) <= 1.0:
            failed.append("prefill_not_faster")
        if oracle_by_mode is not None and not oracle_by_mode[mode].get("pass"):
            failed.append("poolside_oracle_failed")
        mode_results[mode] = {
            "pass": not failed,
            "failed_checks": failed,
            "prefill_speedup": performance["speedups_vs_retained"][mode],
            "max_kl_divergence": quality.get("max_kl_divergence"),
            "top1_agreement": quality.get("top1_agreement"),
        }
        if not failed:
            passing.append(mode)
    selected = (
        max(
            passing,
            key=lambda mode: float(performance["speedups_vs_retained"][mode]),
        )
        if passing
        else None
    )
    return {
        "pass": bool(passing),
        "passing_modes": passing,
        "selected_mode": selected,
        "modes": mode_results,
        "policy": {
            "suite_and_each_category": "all ten prompts/four categories",
            "quality": "finite logits, KL <= 0.05, top-1 >= 90% suite-wide and per category",
            "performance": "aggregate prefill speedup > 1.0x",
            "oracle": "frozen Poolside first-token gate",
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.repetitions < 3:
        raise ValueError("component bisection requires at least three repetitions")
    if args.teacher_forced_tokens != 32:
        raise ValueError("component bisection requires 32 teacher-forced steps")
    if args.chunk_size != 128:
        raise ValueError("component bisection requires chunk size 128")
    if not args.model.is_file():
        raise FileNotFoundError(f"Laguna model not found: {args.model}")
    repo = _repo_state()
    if not repo["tracked_clean"]:
        raise RuntimeError("component bisection requires a clean tracked worktree")
    source_rejection = _load_source_rejection(args)
    provenance = collect_artifact_provenance(
        repo_root=ROOT,
        configured_backend=args.backend,
        resolved_backend=args.backend,
        target_arch=args.backend.removeprefix("hip_"),
        model_path=args.model,
        quant="gguf_q4_k_m",
        kv_dtype="bf16",
        command=(str(Path(sys.executable).resolve()), *sys.argv),
        build_profile="laguna_expert_major_component_bisection",
        timing_protocol="one_owner_counterbalanced_m128_prefill_plus_multiway_teacher_force",
        warmups=len(MODES),
        repetitions=args.repetitions,
    )
    reader = GGUFReader(args.model)
    tokenizer = LagunaGGUFTokenizer.from_gguf_info(reader.info)
    prompts = _load_prompts(args.prompts, tokenizer)

    runtime = get_hip_runtime()
    gpu_free_before, gpu_total = runtime.mem_get_info()
    tracked_before = memory_stats()
    owner = None
    rows = []
    teacher_rows = {mode: [] for mode in CANDIDATE_MODES}
    oracle_by_mode = {}
    load_started = time.perf_counter()
    try:
        owner = LagunaGGUFResidentSession(
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
        load_seconds = time.perf_counter() - load_started
        for mode in MODES:
            warmup = _open_mode_session(owner, args, mode)
            try:
                warmup.prefill(_expanded_token_ids(prompts[0]), use_bulk=True)
            finally:
                warmup.close()
        for repetition in range(args.repetitions):
            for prompt_index, prompt in enumerate(prompts):
                for mode in _mode_order(prompt_index, repetition):
                    row = _time_prefill(
                        owner,
                        prompt,
                        args,
                        mode=mode,
                        repetition=repetition,
                    )
                    rows.append(row)
                    print(
                        f"rep={repetition} prompt={prompt['id']} mode={mode} "
                        f"prefill={row['prefill_tok_s']:.3f} tok/s",
                        file=sys.stderr,
                        flush=True,
                    )
        for prompt in prompts:
            prompt_rows = _teacher_forced_prompt(owner, prompt, args)
            for mode in CANDIDATE_MODES:
                teacher_rows[mode].extend(prompt_rows[mode])
                steps = prompt_rows[mode][0]["steps"]
                matches = sum(bool(step["top1_agreement"]) for step in steps)
                max_kl = max(float(step["kl_divergence"]) for step in steps)
                print(
                    f"teacher prompt={prompt['id']} mode={mode} "
                    f"top1={matches}/{len(steps)} max_kl={max_kl:.6g}",
                    file=sys.stderr,
                    flush=True,
                )
        for mode in CANDIDATE_MODES:
            oracle_by_mode[mode] = _poolside_oracle(owner, args, mode)
        resident_nbytes = owner.resident_nbytes
    finally:
        if owner is not None:
            owner.close()
    tracked_after = memory_stats()
    gpu_free_after, gpu_total_after = runtime.mem_get_info()
    if gpu_total_after != gpu_total:
        raise RuntimeError("HIP total memory changed during component bisection")

    performance = _aggregate_performance(rows)
    quality_by_mode = {
        mode: _teacher_forced_quality(teacher_rows[mode]) for mode in CANDIDATE_MODES
    }
    evaluation = _evaluate(performance, quality_by_mode, oracle_by_mode)
    recovered = bool(
        tracked_after["current_allocated_bytes"]
        == tracked_before["current_allocated_bytes"]
        and tracked_after["active_allocations"]
        == tracked_before["active_allocations"]
    )
    if not recovered:
        evaluation["pass"] = False
        evaluation["selected_mode"] = None
        evaluation["failed_check"] = "tracked_lifecycle_not_recovered"
    manifest_path = args.repacked_cache / "manifest.json"
    prompt_payload = args.prompts.read_bytes()
    return {
        "schema": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "kind": "hipengine_laguna_expert_major_component_bisection",
        "status": "component_quality_admitted" if evaluation["pass"] else "component_bisection_rejected",
        "pass": bool(evaluation["pass"]),
        "performance_claim": False,
        "provenance": provenance,
        "repo": repo,
        "model": {
            "path": str(args.model.resolve()),
            "sha256": args.model_sha256,
            "quant": "Q4_K_M mixed GGUF v3",
            "repacked_cache": str(args.repacked_cache.resolve()),
            "repacked_cache_manifest_sha256": (
                _sha256_bytes(manifest_path.read_bytes()) if manifest_path.is_file() else None
            ),
        },
        "platform": {
            "backend": args.backend,
            "target_arch": args.backend.removeprefix("hip_"),
            "device_name": provenance["device_name"],
            "machine": platform.machine(),
            "hip_total_bytes": gpu_total,
        },
        "protocol": {
            "modes": list(MODES),
            "chunk_size": args.chunk_size,
            "prompt_tokens": 128,
            "prompt_expansion": "repeat without leading BOS, then truncate; identical for every mode",
            "repetitions": args.repetitions,
            "teacher_forced_tokens_per_prompt": args.teacher_forced_tokens,
            "prompt_suite": str(args.prompts.resolve()),
            "prompt_suite_sha256": _sha256_bytes(prompt_payload),
            "timing_scope": "resident same-owner prefill only; model load and quality copies excluded",
            "teacher_forcing": "one retained top-1 ID feeds all four isolated sessions per step",
        },
        "source_rejection": source_rejection,
        "load": {"seconds_excluded": load_seconds, "resident_nbytes": resident_nbytes},
        "performance": performance,
        "quality": {
            "teacher_forced": quality_by_mode,
            "teacher_forced_prompts": teacher_rows,
            "poolside_oracle": oracle_by_mode,
            "tracked_returned_to_baseline": recovered,
        },
        "evaluation": evaluation,
        "rows": rows,
        "memory": {
            "tracked_before": tracked_before,
            "tracked_after": tracked_after,
            "gpu_free_before": gpu_free_before,
            "gpu_free_after": gpu_free_after,
            "hip_total_bytes": gpu_total,
        },
        "command": [str(Path(sys.executable).resolve()), *sys.argv],
        "notes": [
            "All ten prompts/four categories are uniformly expanded to M128; no prompt-conditioned dispatch is used.",
            "Gate/up-only keeps exact grouped Q4/Q6 down; down-only keeps exact selected Q4 gate/up.",
            "All adaptive component modes retain the exact grouped fallback below M128 and exact c=1 decode.",
            "This is a component admission diagnostic, not a retained default or topline claim.",
        ],
    }


def main() -> int:
    args = _parse_args()
    result = run(args)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")
    sys.stdout.write(payload)
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
