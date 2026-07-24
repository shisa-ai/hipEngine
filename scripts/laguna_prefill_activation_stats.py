#!/usr/bin/env python3
"""Capture compact deterministic Laguna post-layer BF16 activation statistics."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time
from typing import Any, Mapping

import numpy as np

from hipengine.benchmark.provenance import collect_artifact_provenance
from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import (
    DeviceBuffer,
    copy_device_to_host,
    free,
    host_array_ptr,
    malloc,
    memory_stats,
)
from hipengine.loading.gguf import GGUFReader
from hipengine.runtime.laguna_gguf_runner import (
    LAGUNA_DFLASH_CAPTURE_DEPTHS,
    LagunaGGUFResidentSession,
    LagunaHiddenCaptureTargets,
)
from hipengine.tokenization.gguf import LagunaGGUFTokenizer
from scripts.laguna_prefill_profile import _profile_token_stream
from scripts.laguna_target_ar_bench import (
    DEFAULT_CACHE,
    DEFAULT_MODEL,
    DEFAULT_MODEL_SHA256,
    DEFAULT_PROMPTS,
    _compiler_version,
    _laguna_f16_prefill_configuration,
    _load_prompts,
    _progress,
    _repo_state,
    _sha256_bytes,
    _sha256_json,
)

ROOT = Path(__file__).resolve().parents[1]
ACTIVATION_ROWS = (32, 55, 64, 122, 128, 256, 512)
CAPTURE_DEPTHS = tuple(int(value) for value in LAGUNA_DFLASH_CAPTURE_DEPTHS)
DEFAULT_REPETITIONS = 2
DEFAULT_OUTPUT = (
    ROOT / "benchmarks/results/2026-07-24-gfx1151-laguna-prefill-activation-stats.json"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--backend", default="hip_gfx1151")
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--repacked-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--model-sha256", default=DEFAULT_MODEL_SHA256)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _bf16_bits(values: np.ndarray) -> np.ndarray:
    """Round float32 values to their BF16 bit patterns."""

    source = np.ascontiguousarray(values, dtype=np.float32)
    bits = source.view(np.uint32)
    rounding_bias = np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return np.ascontiguousarray(
        ((bits + rounding_bias) >> np.uint32(16)).astype(np.uint16)
    )


def _bf16_to_float32(values: np.ndarray) -> np.ndarray:
    bits = np.ascontiguousarray(values, dtype=np.uint16)
    expanded = bits.astype(np.uint32) << np.uint32(16)
    return expanded.view(np.float32)


def _distribution(values: np.ndarray) -> dict[str, float | None]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return {
            "minimum": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "p99_9": None,
            "maximum": None,
            "mean": None,
        }
    percentiles = np.percentile(finite, (50.0, 90.0, 95.0, 99.0, 99.9))
    return {
        "minimum": float(np.min(finite)),
        "p50": float(percentiles[0]),
        "p90": float(percentiles[1]),
        "p95": float(percentiles[2]),
        "p99": float(percentiles[3]),
        "p99_9": float(percentiles[4]),
        "maximum": float(np.max(finite)),
        "mean": float(np.mean(finite)),
    }


def _summarize_bf16_activation(values: np.ndarray) -> dict[str, Any]:
    source = np.asarray(values)
    if source.dtype != np.uint16:
        raise TypeError("BF16 activation storage must use uint16 bits")
    if source.ndim != 2:
        raise ValueError("BF16 activation storage must be two-dimensional")
    if not source.size or not source.shape[0] or not source.shape[1]:
        raise ValueError("BF16 activation storage must be non-empty")
    bits = np.ascontiguousarray(source)
    decoded = _bf16_to_float32(bits)
    finite_mask = np.isfinite(decoded)
    finite = decoded[finite_mask].astype(np.float64)
    finite_count = int(finite.size)
    count = int(decoded.size)
    zero_count = int(np.count_nonzero((bits & np.uint16(0x7FFF)) == 0))
    if finite_count:
        squared = np.square(finite, dtype=np.float64)
        mean = float(np.mean(finite))
        standard_deviation = float(np.std(finite))
        rms = float(math.sqrt(float(np.mean(squared))))
        minimum = float(np.min(finite))
        maximum = float(np.max(finite))
        absolute = _distribution(np.abs(finite))
    else:
        mean = None
        standard_deviation = None
        rms = None
        minimum = None
        maximum = None
        absolute = _distribution(np.empty(0, dtype=np.float64))

    decoded64 = decoded.astype(np.float64)
    with np.errstate(invalid="ignore", over="ignore"):
        row_rms = np.sqrt(np.mean(np.square(decoded64), axis=1))
    return {
        "shape": [int(value) for value in bits.shape],
        "count": count,
        "finite_count": finite_count,
        "nonfinite_count": count - finite_count,
        "zero_count": zero_count,
        "zero_fraction": zero_count / count,
        "minimum": minimum,
        "maximum": maximum,
        "mean": mean,
        "standard_deviation": standard_deviation,
        "rms": rms,
        "absolute": absolute,
        "row_rms": _distribution(row_rms),
        "bf16_sha256": hashlib.sha256(bits.tobytes(order="C")).hexdigest(),
    }


def _capture_once(
    owner: LagunaGGUFResidentSession,
    token_ids: tuple[int, ...],
    *,
    buffers: Mapping[int, DeviceBuffer],
) -> tuple[dict[str, Any], int]:
    rows = len(token_ids)
    targets = LagunaHiddenCaptureTargets(
        hidden_size=owner.config.hidden_size,
        rows=rows,
        buffers=buffers,
    )
    owner.reset_state()
    # This diagnostic intentionally uses the runtime's existing row-capture ABI
    # so production dispatch and arithmetic remain unchanged.
    owner._execute_rows(token_ids, capture_rows=targets, stream=0)
    owner.runtime.device_synchronize()
    by_depth: dict[str, Any] = {}
    for depth, buffer in buffers.items():
        host = np.empty((rows, owner.config.hidden_size), dtype=np.uint16)
        copy_device_to_host(host_array_ptr(host), buffer, runtime=owner.runtime)
        by_depth[str(depth)] = {
            str(prefix): _summarize_bf16_activation(host[:prefix])
            for prefix in ACTIVATION_ROWS
        }
    return by_depth, int(owner.position)


def _all_finite(capture: Mapping[str, Any]) -> bool:
    return all(
        int(summary["nonfinite_count"]) == 0
        for prefixes in capture.values()
        for summary in prefixes.values()
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    maximum_rows = max(ACTIVATION_ROWS)
    if args.context_length < maximum_rows:
        raise ValueError("Laguna activation capture context must admit 512 rows")
    if args.repetitions < 2:
        raise ValueError("Laguna activation capture requires at least two repetitions")
    if not args.model.is_file():
        raise FileNotFoundError(f"Laguna model not found: {args.model}")
    if not args.model_sha256:
        raise ValueError("--model-sha256 is required")
    repo = _repo_state()
    if not repo["tracked_clean"]:
        raise RuntimeError("retained Laguna activation capture requires a clean tracked worktree")
    f16_prefill = _laguna_f16_prefill_configuration(args.backend)
    if f16_prefill["requested"] != "auto":
        raise RuntimeError(
            "shipping-control activation capture requires HIPENGINE_LAGUNA_F16_PREFILL=auto"
        )

    provenance = collect_artifact_provenance(
        repo_root=ROOT,
        configured_backend=args.backend,
        resolved_backend=args.backend,
        target_arch=args.backend.removeprefix("hip_"),
        model_path=args.model,
        quant="gguf_q4_k_m",
        kv_dtype="bf16",
        command=(str(Path(sys.executable).resolve()), *sys.argv),
        build_profile="laguna_prefill_activation_stats",
        timing_protocol="untimed_repeat_exact_post_layer_bf16_capture",
        warmups=0,
        repetitions=args.repetitions,
    )
    reader = GGUFReader(args.model)
    tokenizer = LagunaGGUFTokenizer.from_gguf_info(reader.info)
    prompts = _load_prompts(args.prompts, tokenizer)
    token_stream, token_source = _profile_token_stream(prompts, maximum_rows)
    selected_tokens = tuple(int(value) for value in token_stream[:maximum_rows])

    runtime = get_hip_runtime()
    gpu_free_before, gpu_total = runtime.mem_get_info()
    tracked_before = memory_stats()
    owner: LagunaGGUFResidentSession | None = None
    buffers: dict[int, DeviceBuffer] = {}
    repetitions: list[dict[str, Any]] = []
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
            prefill_chunk_size=maximum_rows,
        )
        load_seconds = time.perf_counter() - load_started
        capture_nbytes = (
            maximum_rows
            * owner.config.hidden_size
            * np.dtype(np.uint16).itemsize
        )
        buffers = {
            depth: malloc(capture_nbytes, runtime=runtime) for depth in CAPTURE_DEPTHS
        }
        for repetition in range(args.repetitions):
            capture, final_position = _capture_once(
                owner,
                selected_tokens,
                buffers=buffers,
            )
            repetitions.append(
                {
                    "repetition": repetition,
                    "final_position": final_position,
                    "capture_sha256": _sha256_json(capture),
                    "capture": capture,
                }
            )
            print(
                f"rep={repetition} rows={maximum_rows} "
                f"capture={repetitions[-1]['capture_sha256'][:16]}",
                file=sys.stderr,
                flush=True,
            )
        resident_nbytes = owner.resident_nbytes
        resolved = {
            "f16_prefill": f16_prefill,
            "global_prefill_variant": owner.global_prefill_variant,
            "swa_prefill_variant": owner.swa_prefill_variant,
            "selected_down_mode": owner.selected_down_mode,
            "matrix_rows": owner.prefill_chunk_size,
            "attention_rows": owner.prefill_attention_chunk_size,
        }
    finally:
        for buffer in reversed(tuple(buffers.values())):
            free(buffer, runtime=runtime)
        if owner is not None:
            owner.close()
    tracked_after = memory_stats()
    gpu_free_after, gpu_total_after = runtime.mem_get_info()
    if gpu_total_after != gpu_total:
        raise RuntimeError("HIP total memory changed during Laguna activation capture")

    first_capture = repetitions[0]["capture"]
    deterministic = len(
        {str(repetition["capture_sha256"]) for repetition in repetitions}
    ) == 1
    positions_exact = all(
        int(repetition["final_position"]) == maximum_rows - 1
        for repetition in repetitions
    )
    finite = all(_all_finite(repetition["capture"]) for repetition in repetitions)
    recovered = bool(
        tracked_after["current_allocated_bytes"]
        == tracked_before["current_allocated_bytes"]
        and tracked_after["active_allocations"] == tracked_before["active_allocations"]
    )
    passed = bool(deterministic and positions_exact and finite and recovered)
    manifest_path = args.repacked_cache / "manifest.json"
    return {
        "schema": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "kind": "hipengine_laguna_prefill_activation_stats",
        "status": "accepted_activation_diagnostic" if passed else "rejected",
        "pass": passed,
        "performance_claim": False,
        "scope": "Laguna shipping-control post-layer BF16 activation summaries",
        "provenance": provenance,
        "repo": repo,
        "model": {
            "path": str(args.model.resolve()),
            "sha256": args.model_sha256,
            "quant": "Q4_K_M mixed GGUF v3",
            "repacked_cache": str(args.repacked_cache.resolve()),
            "repacked_cache_manifest_sha256": (
                _sha256_bytes(manifest_path.read_bytes())
                if manifest_path.is_file()
                else None
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
            "rows": list(ACTIVATION_ROWS),
            "capture_rows": maximum_rows,
            "capture_depths": list(CAPTURE_DEPTHS),
            "capture_point": "post-layer residual hidden BF16",
            "repetitions": args.repetitions,
            "context_length": args.context_length,
            "prompt_suite": str(args.prompts.resolve()),
            "prompt_suite_sha256": _sha256_bytes(args.prompts.read_bytes()),
            "token_stream_sha256": _sha256_json(selected_tokens),
            "token_source": token_source,
            "raw_activations_persisted": False,
            "timing_scope": "untimed diagnostic; model load excluded",
            "resolved_shipping_control": resolved,
        },
        "load": {
            "seconds_excluded": load_seconds,
            "resident_nbytes": resident_nbytes,
            "capture_buffers_nbytes": sum(buffer.nbytes for buffer in buffers.values()),
        },
        "activation_stats": first_capture,
        "repeat_capture_sha256": [
            str(repetition["capture_sha256"]) for repetition in repetitions
        ],
        "correctness": {
            "pass": passed,
            "repeat_bf16_bit_deterministic": deterministic,
            "final_positions_exact": positions_exact,
            "all_captured_values_finite": finite,
            "tracked_returned_to_baseline": recovered,
        },
        "memory": {
            "tracked_before": tracked_before,
            "tracked_after": tracked_after,
            "gpu_free_before": gpu_free_before,
            "gpu_free_after": gpu_free_after,
            "hip_total_bytes": gpu_total,
        },
        "command": [str(Path(sys.executable).resolve()), *sys.argv],
        "notes": [
            "Only compact statistics and BF16 hashes are persisted; raw prompt activations are discarded.",
            "These six post-layer hidden taps are deterministic scale/distribution proxies for LAP-1.",
            "LAP-2 calibration must separately capture exact projection inputs before freezing residual-pack policy.",
            "This artifact selects no arithmetic threshold and makes no throughput claim.",
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
