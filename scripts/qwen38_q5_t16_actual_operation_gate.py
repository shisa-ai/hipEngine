#!/usr/bin/env python3
"""Gate sole-Q5T16 SSM-output operations on actual Qwen3.8 weights.

This is a correctness/ownership gate, not a performance benchmark.  It rotates
four immutable K6144/N5120 Q5_K recurrent-output planes (more than 64 MiB of
source data) through the exact c1, verifier-row, tail, and bulk consumers.  The
one-expert selected producers are the bit-exact physical oracle; an actual-Q5
CPU oracle supplies the project KL/top-1 quality gate at M16.
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
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
from hipengine.kernels.hip_gfx1100.quant.gguf_k_t16_selected_prefill import (
    build_gguf_k_t16_selected_prefill,
    gguf_q5_k_t16_selected_wmma_prefill_compact_bf16_bf16_out,
    gguf_q5_k_t16_wmma_prefill_bf16_bf16_out,
)
from hipengine.kernels.cpu_reference import gguf_quant_gemv
from hipengine.kernels.hip_gfx1100.quant.gguf_t16_selected_gemv import (
    build_gguf_t16_selected_gemv,
    gguf_q5_k_t16_gemv_decode_bf16_bf16_out,
    gguf_q5_k_t16_gemv_rowtile_bf16_bf16_out,
    gguf_q5_k_t16_selected_gemv_bf16_bf16_out,
)
from hipengine.loading.gguf import GGUFReader
from hipengine.quant.gguf import GGMLQuantizationType
from hipengine.quant.gguf_t16 import repack_gguf_q5_k_tile16

DEFAULT_MODEL = Path("/models/gguf/Qwen3.8-27B-Q4_K_M.gguf")
DEFAULT_OUTPUT = Path("/tmp/qwen38-q5-t16-actual-operation-gate.json")
DEFAULT_SMOKE = Path("/tmp/qwen38-task20-q5-smoke/smoke.json")
_REQUIRED_ROWS = (1, 2, 3, 4, 5, 6, 7, 8, 16, 33, 512, 1024, 4096)
_REQUIRED_LAYERS = (0, 1, 2, 4)


@dataclass(frozen=True)
class UploadedQ5:
    name: str
    layer: int
    in_features: int
    out_features: int
    source_nbytes: int
    raw: np.ndarray
    tiles: DeviceBuffer
    tiles_nbytes: int

    def close(self, runtime) -> None:
        free(self.tiles, runtime=runtime)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--layers", default=",".join(str(value) for value in _REQUIRED_LAYERS)
    )
    parser.add_argument(
        "--rows", default=",".join(str(value) for value in _REQUIRED_ROWS)
    )
    parser.add_argument("--seed", type=int, default=0x38D05)
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--full-model-smoke", type=Path, default=DEFAULT_SMOKE)
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


def _upload_weight(reader: GGUFReader, layer: int, runtime) -> UploadedQ5:
    name = f"blk.{layer}.ssm_out.weight"
    info = reader.tensor_info(name)
    if info.ggml_type_name != "Q5_K" or tuple(info.shape) != (5_120, 6_144):
        raise ValueError(f"{name} is not the required K6144/N5120 Q5_K plane: {info}")
    raw = np.ascontiguousarray(reader.tensor_data(name))
    tiles = np.ascontiguousarray(repack_gguf_q5_k_tile16(raw[None, ...]).tiles)
    return UploadedQ5(
        name=name,
        layer=layer,
        in_features=int(info.shape[1]),
        out_features=int(info.shape[0]),
        source_nbytes=int(info.nbytes),
        raw=raw,
        tiles=_upload(runtime, tiles),
        tiles_nbytes=int(tiles.nbytes),
    )


def _run_gate(
    *,
    runtime,
    decode_library,
    prefill_library,
    weight: UploadedQ5,
    rows: int,
    seed: int,
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed + weight.layer * 0x1000 + rows)
    x = _bf16_bits(
        rng.normal(0.0, 0.2, size=(rows, weight.in_features)).astype(np.float32)
    )
    buffers: list[DeviceBuffer] = []
    try:
        x_dev = _upload(runtime, x)
        control_dev = malloc(rows * weight.out_features * 2, runtime=runtime)
        candidate_dev = malloc(rows * weight.out_features * 2, runtime=runtime)
        buffers.extend((x_dev, control_dev, candidate_dev))
        if rows <= 8:
            selected = np.zeros(rows, dtype=np.int64)
            selected_dev = _upload(runtime, selected)
            buffers.append(selected_dev)
            gguf_q5_k_t16_selected_gemv_bf16_bf16_out(
                x_dev.ptr,
                selected_dev.ptr,
                weight.tiles.ptr,
                control_dev.ptr,
                rows,
                rows,
                1,
                weight.in_features,
                weight.out_features,
                library=decode_library,
                runtime=runtime,
            )
            if rows == 1:
                gguf_q5_k_t16_gemv_decode_bf16_bf16_out(
                    x_dev.ptr,
                    weight.tiles.ptr,
                    candidate_dev.ptr,
                    rows,
                    weight.in_features,
                    weight.out_features,
                    library=decode_library,
                    runtime=runtime,
                )
                route = "dense_c1_direct_vs_selected_one_expert"
            else:
                gguf_q5_k_t16_gemv_rowtile_bf16_bf16_out(
                    x_dev.ptr,
                    weight.tiles.ptr,
                    candidate_dev.ptr,
                    rows,
                    weight.in_features,
                    weight.out_features,
                    library=decode_library,
                    runtime=runtime,
                )
                route = "dense_rows2_8_rowtile_vs_selected_one_expert"
        else:
            padded_rows = ((rows + 15) // 16) * 16
            start_compact = np.asarray((0, rows), dtype=np.int64)
            start_wmma = np.asarray((0, padded_rows), dtype=np.int64)
            tile_expert = np.zeros(padded_rows // 16, dtype=np.int64)
            for array in (start_compact, start_wmma, tile_expert):
                buffers.append(_upload(runtime, array))
            gguf_q5_k_t16_selected_wmma_prefill_compact_bf16_bf16_out(
                x_dev.ptr,
                buffers[-3].ptr,
                buffers[-2].ptr,
                buffers[-1].ptr,
                weight.tiles.ptr,
                control_dev.ptr,
                rows,
                weight.in_features,
                weight.out_features,
                1,
                padded_rows,
                library=prefill_library,
                runtime=runtime,
            )
            gguf_q5_k_t16_wmma_prefill_bf16_bf16_out(
                x_dev.ptr,
                weight.tiles.ptr,
                candidate_dev.ptr,
                rows,
                weight.in_features,
                weight.out_features,
                library=prefill_library,
                runtime=runtime,
            )
            route = "dense_wmma_vs_selected_compact_one_expert"
        runtime.device_synchronize()
        shape = (rows, weight.out_features)
        control = _read(runtime, control_dev, shape)
        candidate = _read(runtime, candidate_dev, shape)
        result = _verdict(control, candidate)
        result["route"] = route
        return result, x, candidate
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)


def _softmax_kl(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    ref = np.asarray(reference, dtype=np.float64)
    cand = np.asarray(candidate, dtype=np.float64)
    ref -= ref.max(axis=-1, keepdims=True)
    cand -= cand.max(axis=-1, keepdims=True)
    ref_logp = ref - np.log(np.exp(ref).sum(axis=-1, keepdims=True))
    cand_logp = cand - np.log(np.exp(cand).sum(axis=-1, keepdims=True))
    return np.sum(np.exp(ref_logp) * (ref_logp - cand_logp), axis=-1)


def _quality_gate(weight: UploadedQ5, x: np.ndarray, candidate: np.ndarray) -> dict[str, object]:
    reference = gguf_quant_gemv(
        _bf16_f32(x), weight.raw, GGMLQuantizationType.Q5_K
    )
    actual = _bf16_f32(candidate)
    kl = _softmax_kl(reference, actual)
    top1 = reference.argmax(axis=-1) == actual.argmax(axis=-1)
    return {
        "rows": int(x.shape[0]),
        "max_kl": float(kl.max()),
        "top1_matches": int(top1.sum()),
        "top1_total": int(top1.size),
        "max_abs": float(np.max(np.abs(reference - actual))),
        "finite": bool(np.isfinite(actual).all()),
    }


def _validate_full_model_smoke(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text())
    raw_runs = (
        payload.get("runs_by_workload", {}).get("512/8")
        if isinstance(payload.get("runs_by_workload"), dict)
        else None
    )
    if raw_runs is None:
        raw_runs = payload.get("runs", [])
    runs = [run for run in raw_runs if run["measured"]]
    if len(runs) != 1:
        raise ValueError(f"expected one measured smoke run, got {len(runs)}")
    run = runs[0]
    weights = run["memory_snapshots"]["after_load"]["owned_session_breakdown"][
        "families"
    ]["weights"]
    teardown = payload["persistent_session_memory"]["summary"][
        "tracked_current_allocated_bytes_after_close"
    ]
    result = {
        "path": str(path.resolve()),
        "finite_final_logits": bool(run["correctness_sanity"]["finite_final_logits"]),
        "final_token_id": int(run["correctness_sanity"]["final_token_id"]),
        "production_graph": bool(run["effective_graph_replay_decode"]),
        "q5_t16_bytes": int(weights["by_layout_bytes"].get("gguf_q5_k_t16_v1", 0)),
        "dense_bf16_bytes": int(weights["by_layout_bytes"].get("dense_bf16", 0)),
        "tracked_teardown_bytes": int(teardown),
    }
    result["passed"] = (
        result["finite_final_logits"]
        and result["production_graph"]
        and result["q5_t16_bytes"] == 1_061_683_200
        and result["dense_bf16_bytes"] == 83_886_080
        and result["tracked_teardown_bytes"] == 0
    )
    return result


def main() -> int:
    args = _parse_args()
    dirty = _tracked_status()
    if dirty and not args.allow_dirty:
        raise SystemExit("tracked worktree must be clean; pass --allow-dirty for development")
    rows = tuple(int(value) for value in args.rows.split(","))
    layers = tuple(int(value) for value in args.layers.split(","))
    if rows != _REQUIRED_ROWS:
        raise ValueError(f"rows must be exactly {_REQUIRED_ROWS}, got {rows}")
    if layers != _REQUIRED_LAYERS:
        raise ValueError(f"layers must be exactly {_REQUIRED_LAYERS}, got {layers}")
    if args.compiler_version_file is not None:
        os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(args.compiler_version_file)

    reader = GGUFReader(args.model)
    source_pool_bytes = sum(
        int(reader.tensor_info(f"blk.{layer}.ssm_out.weight").nbytes)
        for layer in layers
    )
    if source_pool_bytes <= 64 * 1024 * 1024:
        raise ValueError(
            f"actual source pool must exceed 64 MiB, got {source_pool_bytes} bytes"
        )

    runtime = get_hip_runtime()
    initial_current = int(memory_stats()["current_allocated_bytes"])
    require_cached = bool(args.require_cached_build)
    decode_library = build_gguf_t16_selected_gemv(
        load=True, require_cached=require_cached
    )
    prefill_library = build_gguf_k_t16_selected_prefill(
        load=True, require_cached=require_cached
    )
    weights: list[UploadedQ5] = []
    results: dict[str, dict[str, object]] = {}
    quality_rows: list[dict[str, object]] = []
    try:
        for layer in layers:
            weights.append(_upload_weight(reader, layer, runtime))
        for weight in weights:
            layer_results: dict[str, object] = {}
            for rows_value in rows:
                result, x, candidate = _run_gate(
                    runtime=runtime,
                    decode_library=decode_library,
                    prefill_library=prefill_library,
                    weight=weight,
                    rows=rows_value,
                    seed=args.seed,
                )
                layer_results[str(rows_value)] = result
                if rows_value == 16:
                    quality = _quality_gate(weight, x, candidate)
                    quality["layer"] = weight.layer
                    quality_rows.append(quality)
            results[str(weight.layer)] = layer_results
    finally:
        for weight in reversed(weights):
            weight.close(runtime)
    runtime.device_synchronize()
    final_current = int(memory_stats()["current_allocated_bytes"])

    operation_rows = [result for layer in results.values() for result in layer.values()]
    exact = all(bool(result["exact"]) for result in operation_rows)
    finite = all(bool(result["finite"]) for result in operation_rows)
    max_kl = max(float(result["max_kl"]) for result in quality_rows)
    top1_matches = sum(int(result["top1_matches"]) for result in quality_rows)
    top1_total = sum(int(result["top1_total"]) for result in quality_rows)
    top1 = top1_matches / top1_total
    quality_passed = max_kl <= 0.05 and top1 >= 0.9
    smoke = _validate_full_model_smoke(args.full_model_smoke)
    teardown = final_current == initial_current
    passed = exact and finite and quality_passed and smoke["passed"] and teardown

    artifact = {
        "schema_version": 1,
        "kind": "qwen38_q5_t16_actual_operation_gate",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "exact_quality_gated" if passed else "failed",
        "performance_claim": False,
        "model": str(args.model.resolve()),
        "model_size_bytes": args.model.stat().st_size,
        "layers": list(layers),
        "rows": list(rows),
        "source_pool_bytes": source_pool_bytes,
        "source_pool_mib": source_pool_bytes / (1024 * 1024),
        "candidate_pool_bytes": sum(weight.tiles_nbytes for weight in weights),
        "weights": [
            {
                "layer": weight.layer,
                "name": weight.name,
                "shape": [weight.out_features, weight.in_features],
                "source_nbytes": weight.source_nbytes,
                "candidate_nbytes": weight.tiles_nbytes,
                "candidate_layout": "gguf_q5_k_t16_v1",
                "candidate_allocations": ["tiles"],
            }
            for weight in weights
        ],
        "results": results,
        "correctness": {
            "operation_verdicts": len(operation_rows),
            "all_bf16_exact_to_selected_physical_oracle": exact,
            "all_finite": finite,
            "cpu_quality": {
                "rows": quality_rows,
                "max_kl": max_kl,
                "top1_matches": top1_matches,
                "top1_total": top1_total,
                "top1_agreement": top1,
                "project_max_kl": 0.05,
                "project_min_top1": 0.9,
                "passed": quality_passed,
            },
            "full_model_projection_handoff": smoke,
        },
        "ownership": {
            "candidate_layout": "gguf_q5_k_t16_v1",
            "candidate_allocations": ["tiles"],
            "dense_bf16_candidate_shadow_bytes": 0,
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
    if not passed:
        raise SystemExit("actual-weight Q5T16 operation gate failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
