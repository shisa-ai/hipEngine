#!/usr/bin/env python3
"""Gate sole-Q4T16 operations on actual Qwen3.8-27B Q4_K weights.

This is a correctness and ownership screen, not a performance benchmark.  It
uses a role-diverse source-weight pool larger than 64 MiB and checks the exact
c1, verifier-row, tail, and bulk paths used by the dense package.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import (
    DeviceBuffer,
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
    memory_stats,
)
from hipengine.kernels.hip_gfx1100.fused.gguf_ops import (
    build_gguf_ops,
    gguf_bf16_add,
)
from hipengine.kernels.hip_gfx1100.fused.paro_silu import (
    build_paro_silu,
    silu_mul_separate_out_bf16,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_k_t16_selected_prefill import (
    build_gguf_k_t16_selected_prefill,
    gguf_q4_k_t16_dense_dual_wmma_prefill_silu_bf16_bf16_out,
    gguf_q4_k_t16_wmma_prefill_bf16_bf16_out,
    gguf_q4_k_t16_wmma_prefill_shared_b_bf16_bf16_out,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_gemv import (
    build_gguf_q4_k_gemv,
    gguf_q4_k_pack8_dual_rowtile_silu_bf16_bf16_out,
    gguf_q4_k_pack8_dual_silu_bf16_bf16_out,
    gguf_q4_k_pack8_gemv_bf16_bf16_out,
    gguf_q4_k_pack8_rowtile_bf16_bf16_out,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_t16_selected_gemv import (
    build_gguf_t16_selected_gemv,
    gguf_q4_k_t16_dense_dual_local32_silu_bf16_bf16_out,
    gguf_q4_k_t16_dense_dual_rowtile_silu_bf16_bf16_out,
    gguf_q4_k_t16_dense_rowtile_bf16_bf16_out,
    gguf_q4_k_t16_dense_rowtile_bf16_residual_bf16_out,
    gguf_q4_k_t16_dense_single_local32_bf16_bf16_out,
)
from hipengine.loading.gguf import GGUFReader
from hipengine.quant.gguf_q4_k import (
    repack_gguf_q4_k_pack8,
    repack_gguf_q4_k_tile16,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path("/models/gguf/Qwen3.8-27B-Q4_K_M.gguf")
DEFAULT_OUTPUT = Path("/tmp/qwen38-q4-t16-actual-operation-gate.json")
_REQUIRED_ROWS = (1, 2, 3, 4, 16, 33, 512, 1024, 4096)


@dataclass(frozen=True)
class UploadedQ4:
    name: str
    in_features: int
    out_features: int
    source_nbytes: int
    tiles: DeviceBuffer
    qweight: DeviceBuffer
    scales: DeviceBuffer
    mins: DeviceBuffer

    def free(self, runtime) -> None:
        for buffer in (self.mins, self.scales, self.qweight, self.tiles):
            free(buffer, runtime=runtime)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument(
        "--rows",
        default=",".join(str(value) for value in _REQUIRED_ROWS),
    )
    parser.add_argument("--seed", type=int, default=0x38D00)
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


def _upload(runtime, values: np.ndarray) -> DeviceBuffer:
    array = np.ascontiguousarray(values)
    buffer = malloc(array.nbytes, runtime=runtime)
    copy_host_to_device(buffer, host_array_ptr(array), runtime=runtime)
    return buffer


def _read(runtime, buffer: DeviceBuffer, shape: tuple[int, ...]) -> np.ndarray:
    output = np.empty(shape, dtype=np.uint16)
    copy_device_to_host(host_array_ptr(output), buffer, runtime=runtime)
    return output


def _sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<u2").tobytes()).hexdigest()


def _verdict(control: np.ndarray, candidate: np.ndarray) -> dict[str, object]:
    mismatches = int(np.count_nonzero(candidate != control))
    return {
        "bf16_mismatches": mismatches,
        "control_sha256": _sha256(control),
        "candidate_sha256": _sha256(candidate),
        "finite": bool(np.isfinite(_bf16_f32(candidate)).all()),
        "exact": mismatches == 0,
    }


def _upload_weight(reader: GGUFReader, name: str, runtime) -> UploadedQ4:
    info = reader.tensor_info(name)
    if info.ggml_type_name != "Q4_K" or len(info.shape) != 2:
        raise ValueError(f"{name} is not a rank-2 Q4_K tensor: {info}")
    raw = np.asarray(reader.tensor_data(name))
    packed = repack_gguf_q4_k_pack8(raw)
    tiles = repack_gguf_q4_k_tile16(raw[None, ...]).tiles
    return UploadedQ4(
        name=name,
        in_features=int(info.shape[1]),
        out_features=int(info.shape[0]),
        source_nbytes=int(info.nbytes),
        tiles=_upload(runtime, tiles),
        qweight=_upload(runtime, packed.qweight),
        scales=_upload(runtime, packed.scales),
        mins=_upload(runtime, packed.mins),
    )


def _single_gate(
    *,
    runtime,
    t16_library,
    q4_library,
    prefill_library,
    weight: UploadedQ4,
    rows: int,
    seed: int,
) -> dict[str, object]:
    rng = np.random.default_rng(seed + rows + weight.in_features)
    x = _bf16_bits(
        rng.normal(0.0, 0.2, size=(rows, weight.in_features)).astype(np.float32)
    )
    buffers: list[DeviceBuffer] = []
    try:
        x_dev = _upload(runtime, x)
        control_dev = malloc(rows * weight.out_features * 2, runtime=runtime)
        candidate_dev = malloc(rows * weight.out_features * 2, runtime=runtime)
        buffers.extend((x_dev, control_dev, candidate_dev))
        if rows == 1:
            gguf_q4_k_pack8_gemv_bf16_bf16_out(
                x_dev.ptr,
                weight.qweight.ptr,
                weight.scales.ptr,
                weight.mins.ptr,
                control_dev.ptr,
                rows,
                weight.in_features,
                weight.out_features,
                threads=32,
                library=q4_library,
                runtime=runtime,
            )
            gguf_q4_k_t16_dense_single_local32_bf16_bf16_out(
                x_dev.ptr,
                weight.tiles.ptr,
                candidate_dev.ptr,
                rows,
                weight.in_features,
                weight.out_features,
                library=t16_library,
                runtime=runtime,
            )
            reference = "opening_pack8_c1"
        elif rows in (2, 3, 4):
            gguf_q4_k_pack8_rowtile_bf16_bf16_out(
                x_dev.ptr,
                weight.qweight.ptr,
                weight.scales.ptr,
                weight.mins.ptr,
                control_dev.ptr,
                rows,
                weight.in_features,
                weight.out_features,
                library=q4_library,
                runtime=runtime,
            )
            gguf_q4_k_t16_dense_rowtile_bf16_bf16_out(
                x_dev.ptr,
                weight.tiles.ptr,
                candidate_dev.ptr,
                rows,
                weight.in_features,
                weight.out_features,
                library=t16_library,
                runtime=runtime,
            )
            reference = "opening_pack8_rowtile"
        else:
            gguf_q4_k_t16_wmma_prefill_bf16_bf16_out(
                x_dev.ptr,
                weight.tiles.ptr,
                control_dev.ptr,
                rows,
                weight.in_features,
                weight.out_features,
                library=prefill_library,
                runtime=runtime,
            )
            gguf_q4_k_t16_wmma_prefill_shared_b_bf16_bf16_out(
                x_dev.ptr,
                weight.tiles.ptr,
                candidate_dev.ptr,
                rows,
                weight.in_features,
                weight.out_features,
                library=prefill_library,
                runtime=runtime,
            )
            reference = "same_t16_singleton"
        runtime.device_synchronize()
        shape = (rows, weight.out_features)
        result = _verdict(
            _read(runtime, control_dev, shape),
            _read(runtime, candidate_dev, shape),
        )
        result["reference"] = reference
        return result
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)


def _dual_silu_gate(
    *,
    runtime,
    t16_library,
    q4_library,
    prefill_library,
    silu_library,
    gate: UploadedQ4,
    up: UploadedQ4,
    rows: int,
    seed: int,
) -> dict[str, object]:
    if (gate.in_features, gate.out_features) != (up.in_features, up.out_features):
        raise ValueError("gate/up geometry differs")
    rng = np.random.default_rng(seed + rows + 0xD00)
    x = _bf16_bits(
        rng.normal(0.0, 0.2, size=(rows, gate.in_features)).astype(np.float32)
    )
    nbytes = rows * gate.out_features * 2
    buffers: list[DeviceBuffer] = []
    try:
        x_dev = _upload(runtime, x)
        control_dev = malloc(nbytes, runtime=runtime)
        candidate_dev = malloc(nbytes, runtime=runtime)
        buffers.extend((x_dev, control_dev, candidate_dev))
        if rows == 1:
            gguf_q4_k_pack8_dual_silu_bf16_bf16_out(
                x_dev.ptr,
                gate.qweight.ptr,
                gate.scales.ptr,
                gate.mins.ptr,
                up.qweight.ptr,
                up.scales.ptr,
                up.mins.ptr,
                control_dev.ptr,
                rows,
                gate.in_features,
                gate.out_features,
                threads=32,
                library=q4_library,
                runtime=runtime,
            )
            gguf_q4_k_t16_dense_dual_local32_silu_bf16_bf16_out(
                x_dev.ptr,
                gate.tiles.ptr,
                up.tiles.ptr,
                candidate_dev.ptr,
                rows,
                gate.in_features,
                gate.out_features,
                library=t16_library,
                runtime=runtime,
            )
            reference = "opening_pack8_dual_c1"
        elif rows in (2, 3, 4):
            gguf_q4_k_pack8_dual_rowtile_silu_bf16_bf16_out(
                x_dev.ptr,
                gate.qweight.ptr,
                gate.scales.ptr,
                gate.mins.ptr,
                up.qweight.ptr,
                up.scales.ptr,
                up.mins.ptr,
                control_dev.ptr,
                rows,
                gate.in_features,
                gate.out_features,
                library=q4_library,
                runtime=runtime,
            )
            gguf_q4_k_t16_dense_dual_rowtile_silu_bf16_bf16_out(
                x_dev.ptr,
                gate.tiles.ptr,
                up.tiles.ptr,
                candidate_dev.ptr,
                rows,
                gate.in_features,
                gate.out_features,
                library=t16_library,
                runtime=runtime,
            )
            reference = "opening_pack8_dual_rowtile"
        else:
            gate_dev = malloc(nbytes, runtime=runtime)
            up_dev = malloc(nbytes, runtime=runtime)
            buffers.extend((gate_dev, up_dev))
            gguf_q4_k_t16_wmma_prefill_shared_b_bf16_bf16_out(
                x_dev.ptr,
                gate.tiles.ptr,
                gate_dev.ptr,
                rows,
                gate.in_features,
                gate.out_features,
                library=prefill_library,
                runtime=runtime,
            )
            gguf_q4_k_t16_wmma_prefill_shared_b_bf16_bf16_out(
                x_dev.ptr,
                up.tiles.ptr,
                up_dev.ptr,
                rows,
                up.in_features,
                up.out_features,
                library=prefill_library,
                runtime=runtime,
            )
            silu_mul_separate_out_bf16(
                gate_dev.ptr,
                up_dev.ptr,
                control_dev.ptr,
                rows,
                gate.out_features,
                library=silu_library,
                runtime=runtime,
            )
            gguf_q4_k_t16_dense_dual_wmma_prefill_silu_bf16_bf16_out(
                x_dev.ptr,
                gate.tiles.ptr,
                up.tiles.ptr,
                candidate_dev.ptr,
                rows,
                gate.in_features,
                gate.out_features,
                library=prefill_library,
                runtime=runtime,
            )
            reference = "same_t16_unfused_silu"
        runtime.device_synchronize()
        shape = (rows, gate.out_features)
        result = _verdict(
            _read(runtime, control_dev, shape),
            _read(runtime, candidate_dev, shape),
        )
        result["reference"] = reference
        return result
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)


def _down_residual_gate(
    *,
    runtime,
    t16_library,
    prefill_library,
    ops_library,
    weight: UploadedQ4,
    rows: int,
    seed: int,
) -> dict[str, object]:
    rng = np.random.default_rng(seed + rows + 0xADD)
    x = _bf16_bits(
        rng.normal(0.0, 0.2, size=(rows, weight.in_features)).astype(np.float32)
    )
    residual = _bf16_bits(
        rng.normal(0.0, 0.3, size=(rows, weight.out_features)).astype(np.float32)
    )
    nbytes = rows * weight.out_features * 2
    buffers: list[DeviceBuffer] = []
    try:
        x_dev = _upload(runtime, x)
        residual_dev = _upload(runtime, residual)
        projected_dev = malloc(nbytes, runtime=runtime)
        control_dev = malloc(nbytes, runtime=runtime)
        candidate_dev = malloc(nbytes, runtime=runtime)
        buffers.extend(
            (x_dev, residual_dev, projected_dev, control_dev, candidate_dev)
        )
        if rows == 1:
            gguf_q4_k_t16_dense_single_local32_bf16_bf16_out(
                x_dev.ptr,
                weight.tiles.ptr,
                projected_dev.ptr,
                rows,
                weight.in_features,
                weight.out_features,
                library=t16_library,
                runtime=runtime,
            )
        elif rows in (2, 3, 4):
            gguf_q4_k_t16_dense_rowtile_bf16_bf16_out(
                x_dev.ptr,
                weight.tiles.ptr,
                projected_dev.ptr,
                rows,
                weight.in_features,
                weight.out_features,
                library=t16_library,
                runtime=runtime,
            )
        else:
            gguf_q4_k_t16_wmma_prefill_shared_b_bf16_bf16_out(
                x_dev.ptr,
                weight.tiles.ptr,
                projected_dev.ptr,
                rows,
                weight.in_features,
                weight.out_features,
                library=prefill_library,
                runtime=runtime,
            )
        gguf_bf16_add(
            projected_dev.ptr,
            residual_dev.ptr,
            control_dev.ptr,
            rows * weight.out_features,
            library=ops_library,
            runtime=runtime,
        )
        if rows in (2, 3, 4):
            gguf_q4_k_t16_dense_rowtile_bf16_residual_bf16_out(
                x_dev.ptr,
                weight.tiles.ptr,
                residual_dev.ptr,
                candidate_dev.ptr,
                rows,
                weight.in_features,
                weight.out_features,
                library=t16_library,
                runtime=runtime,
            )
            reference = "same_t16_unfused_add"
        else:
            gguf_bf16_add(
                projected_dev.ptr,
                residual_dev.ptr,
                candidate_dev.ptr,
                rows * weight.out_features,
                library=ops_library,
                runtime=runtime,
            )
            reference = "production_same_t16_unfused_add"
        runtime.device_synchronize()
        shape = (rows, weight.out_features)
        result = _verdict(
            _read(runtime, control_dev, shape),
            _read(runtime, candidate_dev, shape),
        )
        result["reference"] = reference
        return result
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)


def main() -> int:
    args = _parse_args()
    dirty = _tracked_status()
    if dirty and not args.allow_dirty:
        raise SystemExit("tracked worktree must be clean; pass --allow-dirty for a gate")
    rows = tuple(int(value) for value in args.rows.split(","))
    if rows != _REQUIRED_ROWS:
        raise ValueError(f"rows must be exactly {_REQUIRED_ROWS}, got {rows}")
    if args.compiler_version_file is not None:
        os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(
            args.compiler_version_file
        )

    reader = GGUFReader(args.model)
    names = {
        "gate": f"blk.{args.layer}.ffn_gate.weight",
        "up": f"blk.{args.layer}.ffn_up.weight",
        "down": f"blk.{args.layer}.ffn_down.weight",
    }
    infos = {role: reader.tensor_info(name) for role, name in names.items()}
    source_pool_bytes = sum(int(info.nbytes) for info in infos.values())
    if source_pool_bytes <= 64 * 1024 * 1024:
        raise ValueError(
            f"actual source pool must exceed 64 MiB, got {source_pool_bytes} bytes"
        )

    runtime = get_hip_runtime()
    initial_current = memory_stats()["current_allocated_bytes"]
    require_cached = bool(args.require_cached_build)
    t16_library = build_gguf_t16_selected_gemv(
        load=True, require_cached=require_cached
    )
    q4_library = build_gguf_q4_k_gemv(load=True, require_cached=require_cached)
    prefill_library = build_gguf_k_t16_selected_prefill(
        load=True, require_cached=require_cached
    )
    silu_library = build_paro_silu(load=True, require_cached=require_cached)
    ops_library = build_gguf_ops(load=True, require_cached=require_cached)
    weights: dict[str, UploadedQ4] = {}
    try:
        for role, name in names.items():
            weights[role] = _upload_weight(reader, name, runtime)
        results: dict[str, dict[str, object]] = {}
        for rows_value in rows:
            results[str(rows_value)] = {
                "single_gate": _single_gate(
                    runtime=runtime,
                    t16_library=t16_library,
                    q4_library=q4_library,
                    prefill_library=prefill_library,
                    weight=weights["gate"],
                    rows=rows_value,
                    seed=args.seed,
                ),
                "single_down": _single_gate(
                    runtime=runtime,
                    t16_library=t16_library,
                    q4_library=q4_library,
                    prefill_library=prefill_library,
                    weight=weights["down"],
                    rows=rows_value,
                    seed=args.seed + 0x1000,
                ),
                "dual_silu": _dual_silu_gate(
                    runtime=runtime,
                    t16_library=t16_library,
                    q4_library=q4_library,
                    prefill_library=prefill_library,
                    silu_library=silu_library,
                    gate=weights["gate"],
                    up=weights["up"],
                    rows=rows_value,
                    seed=args.seed,
                ),
                "down_residual": _down_residual_gate(
                    runtime=runtime,
                    t16_library=t16_library,
                    prefill_library=prefill_library,
                    ops_library=ops_library,
                    weight=weights["down"],
                    rows=rows_value,
                    seed=args.seed,
                ),
            }
    finally:
        for weight in reversed(tuple(weights.values())):
            weight.free(runtime)
    runtime.device_synchronize()
    final_current = memory_stats()["current_allocated_bytes"]

    all_results = [
        operation
        for row in results.values()
        for operation in row.values()
    ]
    exact = all(bool(result["exact"]) for result in all_results)
    finite = all(bool(result["finite"]) for result in all_results)
    teardown = final_current == initial_current
    artifact = {
        "schema_version": 1,
        "kind": "qwen38_q4_t16_actual_operation_gate",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "exact" if exact and finite and teardown else "failed",
        "performance_claim": False,
        "model": str(args.model.resolve()),
        "model_size_bytes": args.model.stat().st_size,
        "layer": args.layer,
        "rows": list(rows),
        "source_pool_bytes": source_pool_bytes,
        "source_pool_mib": source_pool_bytes / (1024 * 1024),
        "weights": {
            role: {
                "name": weight.name,
                "shape": [weight.out_features, weight.in_features],
                "source_nbytes": weight.source_nbytes,
                "candidate_layout": "gguf_q4_k_t16_v1",
                "candidate_allocations": ["tiles"],
            }
            for role, weight in weights.items()
        },
        "results": results,
        "correctness": {
            "all_bf16_exact": exact,
            "all_finite": finite,
        },
        "ownership": {
            "candidate_layout": "gguf_q4_k_t16_v1",
            "candidate_allocations": ["tiles"],
            "pack8_candidate_bytes": 0,
            "control_pack8_lifetime": "gate-only; freed before teardown verdict",
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
            "require_cached_build": require_cached,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(artifact, indent=2))
    if artifact["status"] != "exact":
        raise SystemExit("actual-weight Q4T16 operation gate failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
