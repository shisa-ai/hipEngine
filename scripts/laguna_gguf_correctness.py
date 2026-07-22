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


def run_correctness(
    model: str | Path,
    *,
    template_path: Path,
    oracle_path: Path,
    backend: str,
    greedy_tokens: int,
    compiler_version: str | None,
    require_cached: bool,
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
            progress=_progress,
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

        generated = [first.next_token_id]
        result = first
        while len(generated) < greedy_tokens:
            result = session.forward_token(result.next_token_id)
            generated.append(result.next_token_id)
        inference_seconds = time.perf_counter() - inference_started

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
                "finite_logits": bool(np.isfinite(candidate_logits).all()),
                "max_logit": float(np.max(candidate_logits)),
                "min_logit": float(np.min(candidate_logits)),
            },
            "greedy": {
                "expected": list(expected_greedy),
                "actual": list(generated_tuple),
                "exact": generated_tuple == expected_greedy,
                "matching_prefix_tokens": prefix_matches,
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
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.greedy_tokens <= 0 or args.greedy_tokens > 32:
        raise ValueError("--greedy-tokens must be within [1, 32]")
    result = run_correctness(
        args.model,
        template_path=args.template,
        oracle_path=args.oracle,
        backend=args.backend,
        greedy_tokens=args.greedy_tokens,
        compiler_version=_compiler_version(args.compiler_version_file),
        require_cached=args.require_cached_build,
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
    first = result["first_token"]
    passed = (
        bool(first["finite_logits"])
        and float(first["kl_divergence"]) <= 0.05
        and float(first["top1_agreement"]) >= 0.9
        and bool(result["greedy"]["exact"])
        and captures_pass
        and bool(result["tracked_returned_to_baseline"])
    )
    result["pass"] = passed
    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
