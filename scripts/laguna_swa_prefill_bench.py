#!/usr/bin/env python3
"""Benchmark exact wave32 SWA prefill against Laguna's 128-thread baseline."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
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
from scripts.laguna_prefill_profile import _profile_token_stream
from scripts.laguna_target_ar_bench import (
    DEFAULT_CACHE,
    DEFAULT_MODEL,
    DEFAULT_MODEL_SHA256,
    DEFAULT_PROMPTS,
    _compiler_version,
    _load_prompts,
    _progress,
    _repo_state,
    _sha256_bytes,
    _sha256_json,
)

ROOT = Path(__file__).resolve().parents[1]
LENGTHS = (512, 1024, 4096)
CHUNK_SIZE = 128
BASELINE_VARIANT = "swa_context_rows_spans"
CANDIDATE_VARIANT = "swa_context_rows_wave32_exact_spans"
MODES = ("baseline", "wave32_exact")
VARIANTS = {"baseline": BASELINE_VARIANT, "wave32_exact": CANDIDATE_VARIANT}
DEFAULT_OUTPUT = Path(
    "benchmarks/results/2026-07-23-gfx1151-laguna-prefill-lpf5-swa-wave32.json"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--backend", default="hip_gfx1151")
    parser.add_argument("--context-length", type=int, default=max(LENGTHS))
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--warmup-rows", type=int, default=CHUNK_SIZE)
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--repacked-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--model-sha256", default=DEFAULT_MODEL_SHA256)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _mode_order(length_index: int, repetition: int) -> tuple[str, str]:
    return MODES if (int(length_index) + int(repetition)) % 2 == 0 else tuple(reversed(MODES))


def _hash_array(value: np.ndarray) -> str:
    return hashlib.sha256(memoryview(np.ascontiguousarray(value)).cast("B")).hexdigest()


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {mode: {} for mode in MODES}
    for mode in MODES:
        for length in LENGTHS:
            selected = [
                row for row in rows if row["mode"] == mode and int(row["length"]) == length
            ]
            seconds = [float(row["prefill_seconds"]) for row in selected]
            median = statistics.median(seconds)
            result[mode][str(length)] = {
                "samples_seconds": seconds,
                "median_seconds": median,
                "median_tok_s": length / median,
            }
    result["wave32_vs_baseline"] = {
        str(length): {
            "speedup": result["baseline"][str(length)]["median_seconds"]
            / result["wave32_exact"][str(length)]["median_seconds"],
            "seconds_saved": result["baseline"][str(length)]["median_seconds"]
            - result["wave32_exact"][str(length)]["median_seconds"],
        }
        for length in LENGTHS
    }
    return result


def _correctness(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[int, int], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        grouped[(int(row["length"]), int(row["repetition"]))][str(row["mode"])] = row
    pairs = []
    for (length, repetition), modes in sorted(grouped.items()):
        if set(modes) != set(MODES):
            raise ValueError(f"missing LPF-5 mode pair at length {length} rep {repetition}")
        baseline = modes["baseline"]
        candidate = modes["wave32_exact"]
        checks = {
            "next_token": int(baseline["next_token_id"]) == int(candidate["next_token_id"]),
            "next_logit_bits": baseline["next_token_logit_hex"]
            == candidate["next_token_logit_hex"],
            "logits": baseline["logits_sha256"] == candidate["logits_sha256"],
            "final_hidden": baseline["final_hidden_sha256"]
            == candidate["final_hidden_sha256"],
            "post_layer_hidden": baseline["post_layer_hidden_sha256"]
            == candidate["post_layer_hidden_sha256"],
            "final_position": int(baseline["final_position"])
            == int(candidate["final_position"])
            == length - 1,
        }
        pairs.append(
            {
                "length": length,
                "repetition": repetition,
                "checks": checks,
                "pass": all(checks.values()),
            }
        )
    deterministic = True
    for mode in MODES:
        for length in LENGTHS:
            selected = [
                row for row in rows if row["mode"] == mode and int(row["length"]) == length
            ]
            deterministic = deterministic and len(
                {
                    (
                        row["logits_sha256"],
                        row["final_hidden_sha256"],
                        row["post_layer_hidden_sha256"],
                    )
                    for row in selected
                }
            ) == 1
    return {
        "pass": bool(all(pair["pass"] for pair in pairs) and deterministic),
        "pairs": pairs,
        "same_mode_repeat_deterministic": bool(deterministic),
    }


def _promotion_gate(
    aggregate: Mapping[str, Any],
    correctness: Mapping[str, Any],
    *,
    recovered: bool,
) -> dict[str, Any]:
    failed = []
    if not correctness["pass"]:
        failed.append("full_model_outputs_not_exact")
    if not recovered:
        failed.append("tracked_lifecycle_not_recovered")
    for length in LENGTHS:
        if float(aggregate["wave32_vs_baseline"][str(length)]["speedup"]) <= 1.05:
            failed.append(f"length_{length}_speedup_not_above_1.05")
    return {
        "pass": not failed,
        "failed_checks": failed,
        "policy": "full logits/hidden/cursor exact; every 512/1K/4K length >1.05x; lifecycle exact",
    }


def _run_once(
    owner: LagunaGGUFResidentSession,
    token_stream: Sequence[int],
    *,
    length: int,
    mode: str,
    repetition: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    assert owner.weights is not None
    session = LagunaGGUFResidentSession(
        resident_weights=owner.weights,
        context_length=args.context_length,
        backend=args.backend,
        runtime=owner.runtime,
        compiler_version=_compiler_version(args.compiler_version_file),
        require_cached_build=args.require_cached_build,
        prefill_chunk_size=args.chunk_size,
        swa_prefill_variant=VARIANTS[mode],
    )
    try:
        started = time.perf_counter()
        result = session.prefill(token_stream[:length], use_bulk=True)
        owner.runtime.device_synchronize()
        elapsed = time.perf_counter() - started
        logits = np.empty(session.config.vocab_size, dtype=np.float32)
        final_hidden = np.empty(session.config.hidden_size, dtype=np.uint16)
        post_layer_hidden = np.empty(session.config.hidden_size, dtype=np.uint16)
        for output, buffer in (
            (logits, result.logits),
            (final_hidden, result.final_hidden),
            (post_layer_hidden, result.post_layer_hidden),
        ):
            owner.runtime.memcpy(
                host_array_ptr(output),
                buffer.ptr,
                output.nbytes,
                HipMemcpyKind.DEVICE_TO_HOST,
            )
        return {
            "length": length,
            "chunks": math.ceil(length / args.chunk_size),
            "mode": mode,
            "variant": VARIANTS[mode],
            "repetition": repetition,
            "prefill_seconds": elapsed,
            "prefill_tok_s": length / elapsed,
            "next_token_id": int(result.next_token_id),
            "next_token_logit": float(result.next_token_logit),
            "next_token_logit_hex": float(result.next_token_logit).hex(),
            "final_position": int(session.position),
            "logits_sha256": _hash_array(logits),
            "final_hidden_sha256": _hash_array(final_hidden),
            "post_layer_hidden_sha256": _hash_array(post_layer_hidden),
        }
    finally:
        session.close()


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.context_length < max(LENGTHS):
        raise ValueError("LPF-5 gate requires an admitted 4K context")
    if args.chunk_size != CHUNK_SIZE:
        raise ValueError(f"LPF-5 gate requires chunk size {CHUNK_SIZE}")
    if args.repetitions <= 0:
        raise ValueError("LPF-5 repetitions must be positive")
    if args.warmup_rows <= 0 or args.warmup_rows > args.chunk_size:
        raise ValueError("LPF-5 warmup must fit one chunk")
    if not args.model.is_file():
        raise FileNotFoundError(f"Laguna model not found: {args.model}")
    if not args.model_sha256:
        raise ValueError("--model-sha256 is required")
    repo = _repo_state()
    if not repo["tracked_clean"]:
        raise RuntimeError("retained Laguna LPF-5 gate requires a clean tracked worktree")

    provenance = collect_artifact_provenance(
        repo_root=ROOT,
        configured_backend=args.backend,
        resolved_backend=args.backend,
        target_arch=args.backend.removeprefix("hip_"),
        model_path=args.model,
        quant="gguf_q4_k_m",
        kv_dtype="bf16",
        command=(str(Path(sys.executable).resolve()), *sys.argv),
        build_profile="laguna_prefill_lpf5_swa_wave32_ab",
        timing_protocol="same_resident_weights_512_1024_4096_balanced_variant_order",
        warmups=len(MODES),
        repetitions=args.repetitions,
    )
    reader = GGUFReader(args.model)
    tokenizer = LagunaGGUFTokenizer.from_gguf_info(reader.info)
    prompts = _load_prompts(args.prompts, tokenizer)
    token_stream, token_source = _profile_token_stream(prompts, max(LENGTHS))

    runtime = get_hip_runtime()
    gpu_free_before, gpu_total = runtime.mem_get_info()
    tracked_before = memory_stats()
    owner: LagunaGGUFResidentSession | None = None
    rows: list[dict[str, Any]] = []
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
            _run_once(
                owner,
                token_stream,
                length=args.warmup_rows,
                mode=mode,
                repetition=-1,
                args=args,
            )
        for repetition in range(args.repetitions):
            for length_index, length in enumerate(LENGTHS):
                for mode in _mode_order(length_index, repetition):
                    row = _run_once(
                        owner,
                        token_stream,
                        length=length,
                        mode=mode,
                        repetition=repetition,
                        args=args,
                    )
                    rows.append(row)
                    print(
                        f"rep={repetition} length={length} mode={mode} "
                        f"prefill={row['prefill_tok_s']:.3f} tok/s next={row['next_token_id']}",
                        file=sys.stderr,
                        flush=True,
                    )
        resident_weight_nbytes = owner.weights.nbytes
    finally:
        if owner is not None:
            owner.close()
    tracked_after = memory_stats()
    gpu_free_after, gpu_total_after = runtime.mem_get_info()
    if gpu_total_after != gpu_total:
        raise RuntimeError("HIP total memory changed during Laguna LPF-5 gate")

    aggregate = _aggregate(rows)
    correctness = _correctness(rows)
    recovered = bool(
        tracked_after["current_allocated_bytes"] == tracked_before["current_allocated_bytes"]
        and tracked_after["active_allocations"] == tracked_before["active_allocations"]
    )
    promotion = _promotion_gate(aggregate, correctness, recovered=recovered)
    passed = bool(promotion["pass"])
    prompt_payload = args.prompts.read_bytes()
    manifest_path = args.repacked_cache / "manifest.json"
    return {
        "schema": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "kind": "hipengine_laguna_prefill_lpf5_swa_wave32_ab",
        "status": "retained" if passed else "rejected",
        "pass": passed,
        "performance_claim": passed,
        "performance_claim_scope": (
            "same resident weights and deterministic 512/1K/4K prefill; model/session "
            "construction excluded; exact baseline/candidate device-state hashes"
        ),
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
            "lengths": list(LENGTHS),
            "chunk_size": args.chunk_size,
            "repetitions": args.repetitions,
            "warmup_rows_per_mode": args.warmup_rows,
            "mode_variants": VARIANTS,
            "timed_order": "alternating baseline/wave32 by length and reversed next repetition",
            "timing_scope": "prefill through device synchronize; shared model load and per-run session construction excluded",
            "prompt_suite": str(args.prompts.resolve()),
            "prompt_suite_sha256": _sha256_bytes(prompt_payload),
            "token_stream_sha256": _sha256_json(token_stream),
            "token_source": token_source,
        },
        "load": {
            "seconds_excluded": load_seconds,
            "resident_weight_nbytes": resident_weight_nbytes,
        },
        "rows": rows,
        "aggregate": aggregate,
        "correctness": {
            **correctness,
            "tracked_returned_to_baseline": recovered,
            "boundary_fixture_evidence": [
                "tests/test_laguna_cpu_reference.py::test_laguna_global_and_swa_masks_match_transformers_at_511_512_513",
                "tests/test_laguna_kv_attention.py::test_laguna_bulk_global_and_swa_prefill_match_serial_across_ring_wrap",
            ],
        },
        "promotion": promotion,
        "memory": {
            "tracked_before": tracked_before,
            "tracked_after": tracked_after,
            "gpu_free_before": gpu_free_before,
            "gpu_free_after": gpu_free_after,
            "hip_total_bytes": gpu_total,
        },
        "command": [str(Path(sys.executable).resolve()), *sys.argv],
        "notes": [
            "Both routes share immutable resident weights and allocate isolated request state.",
            "Host copies for exact hashes occur after synchronized timing.",
            "The candidate preserves the baseline 128-thread FP32 reduction tree exactly.",
        ],
    }


def main() -> int:
    args = _parse_args()
    result = run(args)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    sys.stdout.write(payload)
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
