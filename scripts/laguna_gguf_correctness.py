#!/usr/bin/env python3
"""Validate eager Laguna S 2.1 logits and greedy state against Poolside."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import (
    copy_device_to_host,
    free,
    host_array_ptr,
    malloc,
    memory_stats,
)
from hipengine.quant.gguf import bf16_to_float32
from hipengine.runtime.laguna_gguf_runner import (
    LAGUNA_DFLASH_CAPTURE_DEPTHS,
    LagunaGGUFResidentSession,
    LagunaHiddenCaptureTargets,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path("/home/lhl/models/gguf/laguna-s-2.1-Q4_K_M.gguf")
DEFAULT_TEMPLATE = ROOT / "tests/fixtures/laguna_poolside_v1_template.json"
DEFAULT_ORACLE = ROOT / "tests/fixtures/laguna_poolside_v1_oracle.json"


def _compiler_version(path: Path | None) -> str | None:
    return None if path is None else path.read_text(encoding="utf-8")


def _progress(completed: int, total: int, spec) -> None:
    if completed == 1 or completed == total or completed % 25 == 0:
        print(
            f"load {completed}/{total}: {spec.source.name} ({spec.layout})",
            file=sys.stderr,
            flush=True,
        )


def _normalized_log_probs(values: np.ndarray) -> np.ndarray:
    logits = np.asarray(values, dtype=np.float64)
    maximum = float(np.max(logits))
    return logits - (maximum + math.log(float(np.exp(logits - maximum).sum())))


def _kl_from_reference_log_probs(
    reference_log_probs: np.ndarray,
    candidate_logits: np.ndarray,
) -> float:
    reference = _normalized_log_probs(reference_log_probs)
    candidate = _normalized_log_probs(candidate_logits)
    probabilities = np.exp(reference)
    return float(np.sum(probabilities * (reference - candidate)))


def _quality_gate_passes(
    result: dict[str, object],
    *,
    captures_pass: bool,
) -> bool:
    first = result["first_token"]
    repeat = result["repeat"]
    teacher_forced = result["teacher_forced"]
    candidate_vs_exact = teacher_forced.get("candidate_vs_exact")
    return (
        bool(first["finite_logits"])
        and float(first["kl_divergence"]) <= 0.05
        and float(first["top1_agreement"]) >= 0.9
        and float(teacher_forced["top1_agreement"]) >= 0.9
        and (
            candidate_vs_exact is None
            or float(candidate_vs_exact["max_kl"]) <= 0.05
        )
        and bool(repeat["exact"])
        and float(repeat["first_logits_max_abs"]) <= 1e-6
        and captures_pass
        and bool(result["tracked_returned_to_baseline"])
    )


def _greedy_step_metrics(
    logits: np.ndarray,
    *,
    expected_id: int,
    top_n: int = 5,
) -> dict[str, object]:
    values = np.asarray(logits, dtype=np.float32)
    top = np.argpartition(values, -top_n)[-top_n:]
    top = top[np.argsort(values[top])[::-1]]
    expected = int(expected_id)
    top1_id = int(top[0])
    return {
        "expected_id": expected,
        "expected_logit": float(values[expected]),
        "expected_is_top1": top1_id == expected,
        "expected_margin_to_top1": float(values[expected] - values[top1_id]),
        "top": [{"id": int(token_id), "logit": float(values[token_id])} for token_id in top],
        "top1_margin": float(values[top[0]] - values[top[1]]),
    }


def run_correctness(
    model: str | Path,
    *,
    template_path: Path,
    oracle_path: Path,
    backend: str,
    greedy_tokens: int,
    compiler_version: str | None,
    require_cached: bool,
    safety_reserve_nbytes: int,
    repacked_cache: Path | None = None,
    model_sha256: str | None = None,
    selected_halfdot: bool = False,
) -> dict[str, object]:
    template = json.loads(template_path.read_text(encoding="utf-8"))
    oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
    prompt_case = next(
        case for case in template["cases"] if case["name"] == oracle["prompt"]["case"]
    )
    prompt_ids = tuple(int(value) for value in prompt_case["token_ids"])
    expected_greedy = tuple(int(value) for value in oracle["greedy32"]["token_ids"])[:greedy_tokens]
    reference_log_probs = np.load(
        oracle_path.parent / oracle["first_token"]["full_distribution"]["path"],
        allow_pickle=False,
    )

    runtime = get_hip_runtime()
    tracked_before = memory_stats()
    free_before, total_before = runtime.mem_get_info()
    session = None
    capture_buffers = []
    load_started = time.perf_counter()
    try:
        session = LagunaGGUFResidentSession(
            model,
            context_length=int(oracle["server"]["context_length"]),
            backend=backend,
            runtime=runtime,
            compiler_version=compiler_version,
            require_cached_build=require_cached,
            safety_reserve_nbytes=safety_reserve_nbytes,
            progress=_progress,
            repacked_cache=repacked_cache,
            model_sha256=model_sha256,
            use_selected_halfdot_decode=selected_halfdot,
        )
        load_seconds = time.perf_counter() - load_started
        row_nbytes = session.config.hidden_size * 2
        capture_buffers = [
            malloc(row_nbytes, runtime=runtime) for _ in LAGUNA_DFLASH_CAPTURE_DEPTHS
        ]
        capture_targets = LagunaHiddenCaptureTargets(
            hidden_size=session.config.hidden_size,
            buffers=dict(zip(LAGUNA_DFLASH_CAPTURE_DEPTHS, capture_buffers, strict=True)),
        )

        inference_started = time.perf_counter()
        first = session.prefill(prompt_ids, capture_last=capture_targets)
        candidate_logits = np.empty(session.config.vocab_size, dtype=np.float32)
        copy_device_to_host(
            host_array_ptr(candidate_logits),
            first.logits,
            runtime=runtime,
        )
        kl = _kl_from_reference_log_probs(reference_log_probs, candidate_logits)
        candidate_top1 = int(np.argmax(candidate_logits))
        reference_top1 = int(oracle["first_token"]["id"])
        first_logits = candidate_logits.copy()
        first_logits_finite = bool(np.isfinite(first_logits).all())
        first_max_logit = float(np.max(first_logits))
        first_min_logit = float(np.min(first_logits))

        generated = [first.next_token_id]
        greedy_step_logits = [
            _greedy_step_metrics(candidate_logits, expected_id=expected_greedy[0])
        ]
        result = first
        while len(generated) < greedy_tokens:
            result = session.forward_token(result.next_token_id)
            generated.append(result.next_token_id)
            copy_device_to_host(
                host_array_ptr(candidate_logits),
                result.logits,
                runtime=runtime,
            )
            greedy_step_logits.append(
                _greedy_step_metrics(
                    candidate_logits,
                    expected_id=expected_greedy[len(generated) - 1],
                )
            )
        inference_seconds = time.perf_counter() - inference_started

        repeat_started = time.perf_counter()
        assert session.weights is not None
        repeat_session = LagunaGGUFResidentSession(
            resident_weights=session.weights,
            context_length=int(oracle["server"]["context_length"]),
            backend=backend,
            runtime=runtime,
            compiler_version=compiler_version,
            require_cached_build=require_cached,
            use_selected_halfdot_decode=selected_halfdot,
        )
        try:
            repeat_first = repeat_session.prefill(prompt_ids)
            copy_device_to_host(
                host_array_ptr(candidate_logits),
                repeat_first.logits,
                runtime=runtime,
            )
            repeat_first_logits_max_abs = float(np.max(np.abs(candidate_logits - first_logits)))
            repeat_generated = [repeat_first.next_token_id]
            repeat_result = repeat_first
            while len(repeat_generated) < greedy_tokens:
                repeat_result = repeat_session.forward_token(repeat_result.next_token_id)
                repeat_generated.append(repeat_result.next_token_id)
        finally:
            repeat_session.close()
        repeat_seconds = time.perf_counter() - repeat_started

        exact_teacher_logits: list[np.ndarray] | None = None
        if selected_halfdot:
            exact_teacher_logits = []
            exact_teacher_session = LagunaGGUFResidentSession(
                resident_weights=session.weights,
                context_length=int(oracle["server"]["context_length"]),
                backend=backend,
                runtime=runtime,
                compiler_version=compiler_version,
                require_cached_build=require_cached,
                use_selected_halfdot_decode=False,
            )
            try:
                exact_result = exact_teacher_session.prefill(prompt_ids)
                for index, expected_id in enumerate(expected_greedy):
                    copy_device_to_host(
                        host_array_ptr(candidate_logits),
                        exact_result.logits,
                        runtime=runtime,
                    )
                    exact_teacher_logits.append(candidate_logits.copy())
                    if index + 1 < greedy_tokens:
                        exact_result = exact_teacher_session.forward_token(
                            expected_id
                        )
            finally:
                exact_teacher_session.close()

        teacher_forced_started = time.perf_counter()
        teacher_forced_session = LagunaGGUFResidentSession(
            resident_weights=session.weights,
            context_length=int(oracle["server"]["context_length"]),
            backend=backend,
            runtime=runtime,
            compiler_version=compiler_version,
            require_cached_build=require_cached,
            use_selected_halfdot_decode=selected_halfdot,
        )
        teacher_forced_steps: list[dict[str, object]] = []
        teacher_forced_exact_kls: list[float] = []
        try:
            teacher_result = teacher_forced_session.prefill(prompt_ids)
            for index, expected_id in enumerate(expected_greedy):
                copy_device_to_host(
                    host_array_ptr(candidate_logits),
                    teacher_result.logits,
                    runtime=runtime,
                )
                teacher_forced_steps.append(
                    _greedy_step_metrics(candidate_logits, expected_id=expected_id)
                )
                if exact_teacher_logits is not None:
                    teacher_forced_exact_kls.append(
                        _kl_from_reference_log_probs(
                            exact_teacher_logits[index],
                            candidate_logits,
                        )
                    )
                if index + 1 < greedy_tokens:
                    teacher_result = teacher_forced_session.forward_token(expected_id)
        finally:
            teacher_forced_session.close()
        teacher_forced_seconds = time.perf_counter() - teacher_forced_started
        teacher_forced_matches = sum(
            bool(step["expected_is_top1"]) for step in teacher_forced_steps
        )

        capture_metrics: dict[str, dict[str, float | bool]] = {}
        for depth, buffer in capture_targets.buffers.items():
            bits = np.empty(session.config.hidden_size, dtype=np.uint16)
            copy_device_to_host(host_array_ptr(bits), buffer, runtime=runtime)
            values = bf16_to_float32(bits)
            capture_metrics[str(depth)] = {
                "finite": bool(np.isfinite(values).all()),
                "max_abs": float(np.max(np.abs(values))),
                "rms": float(np.sqrt(np.mean(values.astype(np.float64) ** 2))),
            }

        generated_tuple = tuple(int(value) for value in generated)
        prefix_matches = 0
        for actual, expected in zip(generated_tuple, expected_greedy, strict=True):
            if actual != expected:
                break
            prefix_matches += 1
        return {
            "schema": 1,
            "model": str(model),
            "model_sha256": model_sha256,
            "repacked_cache": None if repacked_cache is None else str(repacked_cache),
            "backend": backend,
            "context_length": int(oracle["server"]["context_length"]),
            "prompt_tokens": len(prompt_ids),
            "load_seconds": load_seconds,
            "inference_seconds": inference_seconds,
            "resident_nbytes": session.resident_nbytes,
            "first_token": {
                "reference": reference_top1,
                "candidate": candidate_top1,
                "session_argmax": first.next_token_id,
                "top1_agreement": float(candidate_top1 == reference_top1),
                "kl_divergence": kl,
                "finite_logits": first_logits_finite,
                "max_logit": first_max_logit,
                "min_logit": first_min_logit,
            },
            "greedy": {
                "expected": list(expected_greedy),
                "actual": list(generated_tuple),
                "exact": generated_tuple == expected_greedy,
                "matching_prefix_tokens": prefix_matches,
                "step_logits": greedy_step_logits,
            },
            "repeat": {
                "actual": [int(value) for value in repeat_generated],
                "exact": tuple(repeat_generated) == generated_tuple,
                "first_logits_max_abs": repeat_first_logits_max_abs,
                "seconds": repeat_seconds,
            },
            "teacher_forced": {
                "top1_agreement": teacher_forced_matches / greedy_tokens,
                "top1_matches": teacher_forced_matches,
                "steps": teacher_forced_steps,
                "seconds": teacher_forced_seconds,
                **(
                    {
                        "candidate_vs_exact": {
                            "max_kl": max(teacher_forced_exact_kls),
                            "mean_kl": (
                                sum(teacher_forced_exact_kls)
                                / len(teacher_forced_exact_kls)
                            ),
                            "step_kls": teacher_forced_exact_kls,
                        }
                    }
                    if teacher_forced_exact_kls
                    else {}
                ),
            },
            "hidden_captures": capture_metrics,
            "hip_total_bytes": total_before,
            "hip_free_before": free_before,
            "tracked_before": tracked_before,
        }
    finally:
        for buffer in reversed(capture_buffers):
            free(buffer, runtime=runtime)
        if session is not None:
            session.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--oracle", type=Path, default=DEFAULT_ORACLE)
    parser.add_argument("--backend", default="hip_gfx1151")
    parser.add_argument("--greedy-tokens", type=int, default=32)
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--safety-reserve-gib", type=float, default=8.0)
    parser.add_argument("--repacked-cache", type=Path)
    parser.add_argument("--model-sha256")
    parser.add_argument(
        "--selected-halfdot",
        action="store_true",
        help="Diagnostic: replace only gfx1151 c=1 Q4T16 selected gate/up.",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.greedy_tokens <= 0 or args.greedy_tokens > 32:
        raise ValueError("--greedy-tokens must be within [1, 32]")
    if args.safety_reserve_gib <= 0.0:
        raise ValueError("--safety-reserve-gib must be positive")
    if args.selected_halfdot:
        if args.backend != "hip_gfx1151":
            raise ValueError("--selected-halfdot is scoped to hip_gfx1151")
    result = run_correctness(
        args.model,
        template_path=args.template,
        oracle_path=args.oracle,
        backend=args.backend,
        greedy_tokens=args.greedy_tokens,
        compiler_version=_compiler_version(args.compiler_version_file),
        require_cached=args.require_cached_build,
        safety_reserve_nbytes=int(args.safety_reserve_gib * 2**30),
        repacked_cache=args.repacked_cache,
        model_sha256=args.model_sha256,
        selected_halfdot=args.selected_halfdot,
    )
    runtime = get_hip_runtime()
    runtime.device_synchronize()
    result["tracked_after"] = memory_stats()
    result["hip_free_after"] = runtime.mem_get_info()[0]
    before = result["tracked_before"]
    after = result["tracked_after"]
    result["tracked_returned_to_baseline"] = (
        after["current_allocated_bytes"] == before["current_allocated_bytes"]
        and after["active_allocations"] == before["active_allocations"]
    )
    captures_pass = all(bool(metrics["finite"]) for metrics in result["hidden_captures"].values())
    strict_greedy_exact = bool(result["greedy"]["exact"])
    result["strict_cross_runtime_greedy_exact"] = strict_greedy_exact
    result["quality_gate"] = {
        "max_kl": 0.05,
        "max_teacher_forced_candidate_vs_exact_kl": 0.05,
        "min_first_top1_agreement": 0.9,
        "min_teacher_forced_top1_agreement": 0.9,
        "requires_cross_runtime_greedy_exact": False,
        "requires_repeat_exact": True,
    }
    passed = _quality_gate_passes(result, captures_pass=captures_pass)
    result["pass"] = passed
    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
