#!/usr/bin/env python3
"""Screen same-resident Q4T16 dense gate/up+SiLU c1 fusion on Qwen3.8."""

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

from hipengine.benchmark.provenance import detect_device_name
from hipengine.core.hip import get_hip_runtime
from hipengine.kernels.backends import detect_hip_target_arches
from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
    memory_stats,
)
from hipengine.kernels.hip_gfx1100.fused.paro_silu import (
    build_paro_silu,
    silu_mul_separate_out_bf16,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_t16_selected_gemv import (
    build_gguf_t16_selected_gemv,
    gguf_q4_k_t16_dense_dual_local32_silu_bf16_bf16_out,
    gguf_q4_k_t16_dense_single_local32_bf16_bf16_out,
)
from hipengine.loading.gguf import GGUFReader
from hipengine.quant.gguf_q4_k import repack_gguf_q4_k_tile16

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path("/models/gguf/Qwen3.8-27B-Q4_K_M.gguf")
DEFAULT_OUTPUT = Path("/tmp/qwen38-q4-t16-dense-dual-decode-leaf.json")
_MIB = 1024 * 1024


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--layers", default="0,8,63")
    parser.add_argument("--warmups", type=int, default=4)
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--burst", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0x38D5)
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
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & np.uint32(1))
    return np.ascontiguousarray((rounded >> 16).astype(np.uint16))


def _bf16_f32(values: np.ndarray) -> np.ndarray:
    return (np.asarray(values, dtype=np.uint16).astype(np.uint32) << 16).view(
        np.float32
    )


def _upload(runtime, values: np.ndarray):
    array = np.ascontiguousarray(values)
    buffer = malloc(array.nbytes, runtime=runtime)
    copy_host_to_device(buffer, host_array_ptr(array), runtime=runtime)
    return buffer


def _read_bf16(runtime, buffer, shape: tuple[int, ...]) -> np.ndarray:
    result = np.empty(shape, dtype=np.uint16)
    copy_device_to_host(host_array_ptr(result), buffer, runtime=runtime)
    return result


def _event_ms(runtime, launcher: Callable[[], None], *, burst: int) -> float:
    start = runtime.event_create()
    end = runtime.event_create()
    try:
        runtime.event_record(start)
        for _ in range(burst):
            launcher()
        runtime.event_record(end)
        runtime.event_synchronize(end)
        return float(runtime.event_elapsed_time_ms(start, end)) / burst
    finally:
        runtime.event_destroy(end)
        runtime.event_destroy(start)


def _screen_layer(
    *,
    reader: GGUFReader,
    layer: int,
    runtime,
    t16_library,
    silu_library,
    warmups: int,
    samples: int,
    burst: int,
    seed: int,
) -> dict[str, object]:
    gate_name = f"blk.{layer}.ffn_gate.weight"
    up_name = f"blk.{layer}.ffn_up.weight"
    gate_info = reader.tensor_info(gate_name)
    up_info = reader.tensor_info(up_name)
    if gate_info.ggml_type_name != "Q4_K" or up_info.ggml_type_name != "Q4_K":
        raise ValueError(f"layer {layer} gate/up must both be Q4_K")
    if gate_info.shape != up_info.shape or len(gate_info.shape) != 2:
        raise ValueError(f"layer {layer} gate/up geometry differs")
    out_features, in_features = (int(value) for value in gate_info.shape)
    source_pool_bytes = int(gate_info.nbytes) + int(up_info.nbytes)
    if source_pool_bytes <= 64 * _MIB:
        raise ValueError(
            f"layer {layer} source pair must exceed 64 MiB, got "
            f"{source_pool_bytes / _MIB:.3f} MiB"
        )

    gate_raw = np.asarray(reader.tensor_data(gate_name))
    up_raw = np.asarray(reader.tensor_data(up_name))
    gate_tiles = repack_gguf_q4_k_tile16(gate_raw[None, ...]).tiles
    up_tiles = repack_gguf_q4_k_tile16(up_raw[None, ...]).tiles
    del gate_raw, up_raw
    rng = np.random.default_rng(seed + layer)
    x = _bf16_bits(
        rng.normal(0.0, 0.2, size=(1, in_features)).astype(np.float32)
    )

    buffers = []
    try:
        x_dev = _upload(runtime, x)
        gate_tiles_dev = _upload(runtime, gate_tiles)
        up_tiles_dev = _upload(runtime, up_tiles)
        control_gate_dev = malloc(out_features * 2, runtime=runtime)
        control_up_dev = malloc(out_features * 2, runtime=runtime)
        control_out_dev = malloc(out_features * 2, runtime=runtime)
        candidate_out_dev = malloc(out_features * 2, runtime=runtime)
        buffers.extend(
            (
                x_dev,
                gate_tiles_dev,
                up_tiles_dev,
                control_gate_dev,
                control_up_dev,
                control_out_dev,
                candidate_out_dev,
            )
        )

        def control() -> None:
            gguf_q4_k_t16_dense_single_local32_bf16_bf16_out(
                x_dev.ptr,
                gate_tiles_dev.ptr,
                control_gate_dev.ptr,
                1,
                in_features,
                out_features,
                library=t16_library,
                runtime=runtime,
            )
            gguf_q4_k_t16_dense_single_local32_bf16_bf16_out(
                x_dev.ptr,
                up_tiles_dev.ptr,
                control_up_dev.ptr,
                1,
                in_features,
                out_features,
                library=t16_library,
                runtime=runtime,
            )
            silu_mul_separate_out_bf16(
                control_gate_dev.ptr,
                control_up_dev.ptr,
                control_out_dev.ptr,
                rows=1,
                features=out_features,
                library=silu_library,
                runtime=runtime,
            )

        def candidate() -> None:
            gguf_q4_k_t16_dense_dual_local32_silu_bf16_bf16_out(
                x_dev.ptr,
                gate_tiles_dev.ptr,
                up_tiles_dev.ptr,
                candidate_out_dev.ptr,
                1,
                in_features,
                out_features,
                library=t16_library,
                runtime=runtime,
            )

        launchers = {"control_unfused": control, "candidate_fused": candidate}
        for _ in range(warmups):
            for launcher in launchers.values():
                launcher()
        runtime.device_synchronize()
        timings = {name: [] for name in launchers}
        candidate_wins = 0
        for sample in range(samples):
            order = (
                ("control_unfused", "candidate_fused")
                if sample % 2 == 0
                else ("candidate_fused", "control_unfused")
            )
            pair: dict[str, float] = {}
            for name in order:
                pair[name] = _event_ms(
                    runtime,
                    launchers[name],
                    burst=burst,
                )
                timings[name].append(pair[name])
            candidate_wins += int(
                pair["candidate_fused"] < pair["control_unfused"]
            )

        control()
        candidate()
        runtime.device_synchronize()
        control_out = _read_bf16(runtime, control_out_dev, (1, out_features))
        candidate_out = _read_bf16(runtime, candidate_out_dev, (1, out_features))
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)

    medians = {
        name: statistics.median(values) for name, values in timings.items()
    }
    ratio = medians["candidate_fused"] / medians["control_unfused"]
    mismatches = candidate_out != control_out
    return {
        "layer": layer,
        "shape": [out_features, in_features],
        "source_pool_bytes": source_pool_bytes,
        "source_pool_mib": source_pool_bytes / _MIB,
        "resident_t16_bytes": int(gate_tiles.nbytes + up_tiles.nbytes),
        "samples_ms": timings,
        "median_ms": medians,
        "candidate_over_control": ratio,
        "candidate_speedup": 1.0 / ratio,
        "candidate_delta_percent": (ratio - 1.0) * 100.0,
        "candidate_wins": candidate_wins,
        "pair_count": samples,
        "bf16_mismatches": int(np.count_nonzero(mismatches)),
        "bf16_max_ulp": int(
            np.max(
                np.abs(
                    candidate_out.astype(np.int32)
                    - control_out.astype(np.int32)
                )
            )
        ),
        "finite": bool(np.isfinite(_bf16_f32(candidate_out)).all()),
        "control_sha256": hashlib.sha256(
            control_out.astype("<u2").tobytes()
        ).hexdigest(),
        "candidate_sha256": hashlib.sha256(
            candidate_out.astype("<u2").tobytes()
        ).hexdigest(),
    }


def main() -> int:
    args = _parse_args()
    dirty = _tracked_status()
    if dirty and not args.allow_dirty:
        raise SystemExit(
            "tracked worktree must be clean; pass --allow-dirty for a screen"
        )
    if args.warmups < 0 or min(args.samples, args.burst) <= 0:
        raise ValueError("warmups must be non-negative; samples/burst positive")
    layers = tuple(int(value) for value in args.layers.split(","))
    if not layers or len(set(layers)) != len(layers) or min(layers) < 0:
        raise ValueError("layers must be unique non-negative integers")
    if args.compiler_version_file is not None:
        os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(
            args.compiler_version_file
        )

    runtime = get_hip_runtime()
    initial_current = memory_stats()["current_allocated_bytes"]
    t16_library = build_gguf_t16_selected_gemv(
        load=True,
        require_cached=args.require_cached_build,
    )
    silu_library = build_paro_silu(
        load=True,
        require_cached=args.require_cached_build,
    )
    reader = GGUFReader(args.model)
    results = [
        _screen_layer(
            reader=reader,
            layer=layer,
            runtime=runtime,
            t16_library=t16_library,
            silu_library=silu_library,
            warmups=args.warmups,
            samples=args.samples,
            burst=args.burst,
            seed=args.seed,
        )
        for layer in layers
    ]
    runtime.device_synchronize()
    final_current = memory_stats()["current_allocated_bytes"]

    all_exact = all(row["bf16_mismatches"] == 0 for row in results)
    all_finite = all(bool(row["finite"]) for row in results)
    all_positive = all(row["candidate_over_control"] < 1.0 for row in results)
    control_ms = sum(row["median_ms"]["control_unfused"] for row in results)
    candidate_ms = sum(row["median_ms"]["candidate_fused"] for row in results)
    family_ratio = candidate_ms / control_ms
    # Preserved post-P4 eager 512/128 ledger: gate/up plus primitive SiLU own
    # 4,181.033072 ms of 11,087.430579 ms across 128 transitions.
    ledger_family_ms = 4143.079039 + 37.954033
    ledger_wall_ms = 11087.430579
    projected_saved_ms = ledger_family_ms * (1.0 - family_ratio)
    projected_request_saving = projected_saved_ms / ledger_wall_ms
    teardown = final_current == initial_current
    artifact = {
        "schema_version": 1,
        "kind": "qwen38_gfx1151_q4_t16_dense_dual_decode_leaf",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": (
            "candidate"
            if all_exact
            and all_finite
            and all_positive
            and teardown
            and projected_request_saving >= 0.01
            else "rejected"
        ),
        "performance_claim": False,
        "model": str(args.model.resolve()),
        "model_size_bytes": args.model.stat().st_size,
        "hardware": {
            "device": detect_device_name() or "unknown",
            "arch": ",".join(detect_hip_target_arches()) or "unknown",
        },
        "protocol": {
            "layers": list(layers),
            "warmups": args.warmups,
            "samples": args.samples,
            "burst": args.burst,
            "timing": "counterbalanced HIP-event elapsed time",
            "control": "two sole-Q4T16 local32 projections plus primitive BF16 SiLU",
            "candidate": "same-resident sole-Q4T16 dual local32 plus BF16 SiLU",
            "cache_contract": "each actual gate/up source pair exceeds 64 MiB and 2x 32-MiB MALL",
        },
        "results": results,
        "family": {
            "control_ms": control_ms,
            "candidate_ms": candidate_ms,
            "candidate_over_control": family_ratio,
            "candidate_speedup": 1.0 / family_ratio,
            "candidate_delta_percent": (family_ratio - 1.0) * 100.0,
            "all_layers_positive": all_positive,
        },
        "projection": {
            "source": "/tmp/qwen38-task21-p4-postprofile/profile-summary.json",
            "eager_512_128_wall_ms": ledger_wall_ms,
            "eager_gate_up_plus_silu_ms": ledger_family_ms,
            "projected_saved_ms": projected_saved_ms,
            "projected_request_saving": projected_request_saving,
            "projected_request_saving_percent": projected_request_saving * 100.0,
            "advance_threshold_percent": 1.0,
        },
        "correctness": {
            "all_bf16_exact": all_exact,
            "all_finite": all_finite,
        },
        "ownership": {
            "resident_layout": "gguf_q4_k_t16_v1",
            "new_resident_bytes": 0,
            "new_workspace_bytes": 0,
        },
        "memory": {
            "tracked_current_before_bytes": initial_current,
            "tracked_current_after_bytes": final_current,
            "teardown_exact": teardown,
        },
        "provenance": {
            "commit": subprocess.check_output(
                ("git", "rev-parse", "HEAD"), cwd=ROOT, text=True
            ).strip(),
            "tracked_dirty_paths": dirty,
            "target_arch": os.environ.get("HIPENGINE_HIP_ARCH"),
            "compiler_version_file": (
                str(args.compiler_version_file.resolve())
                if args.compiler_version_file is not None
                else None
            ),
            "require_cached_build": bool(args.require_cached_build),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0 if all_exact and all_finite and teardown else 1


if __name__ == "__main__":
    raise SystemExit(main())
