#!/usr/bin/env python3
"""Screen byte-neutral T16 Q6 shared-down decode on actual Laguna weights."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
from typing import Callable

import numpy as np

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q6_k_pack8_gemv import (
    build_gguf_q6_k_pack8_gemv,
    gguf_q6_k_pack8_gemv_decode_bf16_bf16_out,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q6_k_t16_gemv import (
    build_gguf_q6_k_t16_gemv,
    gguf_q6_k_t16_gemv_decode_bf16_bf16_out,
)
from hipengine.loading.gguf import GGUFReader
from hipengine.quant.gguf_t16 import repack_gguf_q6_k_tile16

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path("/home/lhl/models/gguf/laguna-s-2.1-Q4_K_M.gguf")
DEFAULT_OUTPUT = Path("/tmp/laguna-q6-t16-dense-single-decode-leaf.json")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--samples", type=int, default=21)
    parser.add_argument("--burst", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _tracked_status() -> list[str]:
    output = subprocess.check_output(
        ("git", "status", "--short", "--untracked-files=no"),
        cwd=ROOT,
        text=True,
    )
    return [line for line in output.splitlines() if line]


def _bf16_bits(values: np.ndarray) -> np.ndarray:
    f32 = np.ascontiguousarray(values, dtype=np.float32)
    bits = f32.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + (
        (bits >> 16) & np.uint32(1)
    )
    return np.ascontiguousarray((rounded >> 16).astype(np.uint16))


def _upload(runtime, values: np.ndarray):
    array = np.asarray(values)
    if not array.flags.c_contiguous:
        array = np.ascontiguousarray(array)
    buffer = malloc(array.nbytes, runtime=runtime)
    copy_host_to_device(
        buffer,
        host_array_ptr(array),
        runtime=runtime,
    )
    return buffer


def _read_bf16(runtime, buffer, shape: tuple[int, ...]) -> np.ndarray:
    result = np.empty(shape, dtype=np.uint16)
    copy_device_to_host(
        host_array_ptr(result),
        buffer,
        runtime=runtime,
    )
    return result


def _event_ms(runtime, fn: Callable[[], None], *, burst: int) -> float:
    start = runtime.event_create()
    end = runtime.event_create()
    try:
        runtime.event_record(start)
        for _ in range(burst):
            fn()
        runtime.event_record(end)
        runtime.event_synchronize(end)
        return float(runtime.event_elapsed_time_ms(start, end)) / burst
    finally:
        runtime.event_destroy(end)
        runtime.event_destroy(start)


def _screen_weight(
    *,
    runtime,
    raw_library,
    t16_library,
    x: np.ndarray,
    raw: np.ndarray,
    t16: np.ndarray,
    in_features: int,
    out_features: int,
    warmups: int,
    samples: int,
    burst: int,
) -> dict:
    buffers = []
    try:
        x_dev = _upload(runtime, x)
        raw_dev = _upload(runtime, raw)
        t16_dev = _upload(runtime, t16)
        control_out_dev = malloc(out_features * 2, runtime=runtime)
        candidate_out_dev = malloc(out_features * 2, runtime=runtime)
        buffers.extend(
            (
                x_dev,
                raw_dev,
                t16_dev,
                control_out_dev,
                candidate_out_dev,
            )
        )

        def control() -> None:
            gguf_q6_k_pack8_gemv_decode_bf16_bf16_out(
                x_dev.ptr,
                raw_dev.ptr,
                control_out_dev.ptr,
                1,
                in_features,
                out_features,
                library=raw_library,
                runtime=runtime,
            )

        def candidate() -> None:
            gguf_q6_k_t16_gemv_decode_bf16_bf16_out(
                x_dev.ptr,
                t16_dev.ptr,
                candidate_out_dev.ptr,
                1,
                in_features,
                out_features,
                library=t16_library,
                runtime=runtime,
            )

        launchers = {"control_raw": control, "candidate_t16": candidate}
        for _ in range(warmups):
            for launcher in launchers.values():
                launcher()
        runtime.device_synchronize()
        timings = {name: [] for name in launchers}
        candidate_wins = 0
        for sample in range(samples):
            order = (
                ("control_raw", "candidate_t16")
                if sample % 2 == 0
                else ("candidate_t16", "control_raw")
            )
            pair = {}
            for name in order:
                pair[name] = _event_ms(
                    runtime,
                    launchers[name],
                    burst=burst,
                )
                timings[name].append(pair[name])
            candidate_wins += int(
                pair["candidate_t16"] < pair["control_raw"]
            )

        control()
        candidate()
        runtime.device_synchronize()
        control_out = _read_bf16(
            runtime,
            control_out_dev,
            (1, out_features),
        )
        candidate_out = _read_bf16(
            runtime,
            candidate_out_dev,
            (1, out_features),
        )
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)

    medians = {
        name: statistics.median(values)
        for name, values in timings.items()
    }
    ratio = medians["candidate_t16"] / medians["control_raw"]
    mismatch = candidate_out != control_out
    return {
        "samples_ms": timings,
        "median_ms": medians,
        "candidate_over_control": ratio,
        "candidate_delta_percent": (ratio - 1.0) * 100.0,
        "candidate_wins": candidate_wins,
        "pair_count": samples,
        "bf16_mismatches": int(np.count_nonzero(mismatch)),
        "bf16_max_ulp": int(
            np.max(
                np.abs(
                    candidate_out.astype(np.int32)
                    - control_out.astype(np.int32)
                )
            )
        ),
        "candidate_sha256": hashlib.sha256(
            candidate_out.astype("<u2").tobytes()
        ).hexdigest(),
        "control_sha256": hashlib.sha256(
            control_out.astype("<u2").tobytes()
        ).hexdigest(),
        "resident_bytes": {
            "raw": int(raw.nbytes),
            "t16": int(t16.nbytes),
        },
    }


def main() -> int:
    args = _parse_args()
    if _tracked_status() and not args.allow_dirty:
        raise SystemExit(
            "tracked worktree must be clean; pass --allow-dirty for a screen"
        )
    if args.warmups < 0 or min(args.samples, args.burst) <= 0:
        raise ValueError("warmups must be non-negative; samples/burst positive")
    if args.compiler_version_file is not None:
        os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(
            args.compiler_version_file
        )

    reader = GGUFReader(args.model)
    runtime = get_hip_runtime()
    raw_library = build_gguf_q6_k_pack8_gemv(
        load=True,
        require_cached=args.require_cached_build,
    )
    t16_library = build_gguf_q6_k_t16_gemv(
        load=True,
        require_cached=args.require_cached_build,
    )
    rng = np.random.default_rng(args.seed)
    x = _bf16_bits(
        rng.normal(0.0, 0.55, size=(1, 1024)).astype(np.float32)
    )

    results = {}
    for tensor in reader.info.tensors:
        tensor_name = tensor.name
        if not tensor_name.endswith(".ffn_down_shexp.weight"):
            continue
        if tensor.ggml_type_name != "Q6_K":
            continue
        raw = np.ascontiguousarray(reader.tensor_data(tensor_name))
        t16 = repack_gguf_q6_k_tile16(raw[None, ...]).tiles
        results[tensor_name] = _screen_weight(
            runtime=runtime,
            raw_library=raw_library,
            t16_library=t16_library,
            x=x,
            raw=raw,
            t16=t16,
            in_features=1024,
            out_features=3072,
            warmups=args.warmups,
            samples=args.samples,
            burst=args.burst,
        )

    exact = all(row["bf16_mismatches"] == 0 for row in results.values())
    all_positive = all(
        row["candidate_over_control"] < 1.0 for row in results.values()
    )
    family_control_ms = sum(
        row["median_ms"]["control_raw"] for row in results.values()
    )
    family_candidate_ms = sum(
        row["median_ms"]["candidate_t16"] for row in results.values()
    )
    family_ratio = family_candidate_ms / family_control_ms
    artifact = {
        "schema": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "kind": "hipengine_laguna_q6_t16_dense_single_decode_leaf",
        "status": "candidate" if exact and all_positive else "rejected",
        "performance_claim": False,
        "repo_revision": subprocess.check_output(
            ("git", "rev-parse", "HEAD"),
            cwd=ROOT,
            text=True,
        ).strip(),
        "hardware": {
            "gpu": "AMD Radeon 8060S Graphics",
            "architecture": "gfx1151",
        },
        "protocol": {
            "warmups": args.warmups,
            "samples": args.samples,
            "burst": args.burst,
            "timing": "counterbalanced HIP-event elapsed time",
            "weights": "all actual rank-2 Q6_K shared-down tensors",
            "activation": "deterministic synthetic BF16 post-SiLU row",
            "control": "resident raw Q6_K BF16 owner",
            "candidate": "byte-neutral dense Q6T16 BF16 owner",
        },
        "results": results,
        "family": {
            "weights": len(results),
            "control_ms": family_control_ms,
            "candidate_ms": family_candidate_ms,
            "candidate_over_control": family_ratio,
            "candidate_delta_percent": (family_ratio - 1.0) * 100.0,
        },
        "correctness_pass": exact,
        "all_weights_positive": all_positive,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0 if exact else 1


if __name__ == "__main__":
    raise SystemExit(main())
