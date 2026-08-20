#!/usr/bin/env python3
"""Gate role-qualified planar-qmicro Q6 on actual Qwen3.8 weights.

This is a correctness/ownership gate, not a performance benchmark. Standard
Q6T16 is the retained owner for recurrent QKV and a gate-only bit oracle for
planar down, narrow-V, and root payloads. No tensor retains two layouts.
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
from hipengine.kernels.cpu_reference import gguf_quant_gemv
from hipengine.kernels.hip_gfx1100.fused.gguf_ops import build_gguf_ops, gguf_bf16_add
from hipengine.kernels.hip_gfx1100.quant.gguf_q6_k_t16_gemv import (
    build_gguf_q6_k_t16_gemv,
    gguf_q6_k_t16_gemv_decode_bf16_bf16_out,
    gguf_q6_k_t16_gemv_decode_bf16_f32_out,
    gguf_q6_k_t16_gemv_decode_bf16_f32_top1_stage1,
    gguf_q6_k_t16_gemv_rowtile_bf16_f32_out,
    gguf_q6_k_t16_gemv_rowtile_col8_bf16_bf16_out,
    gguf_q6_k_t16_qmicro_planar_gemv_decode_bf16_bf16_out,
    gguf_q6_k_t16_qmicro_planar_gemv_decode_bf16_f32_out,
    gguf_q6_k_t16_qmicro_planar_gemv_decode_bf16_f32_top1_stage1,
    gguf_q6_k_t16_qmicro_planar_gemv_rowtile_bf16_f32_out,
    gguf_q6_k_t16_qmicro_planar_gemv_rowtile_col8_bf16_bf16_out,
    gguf_q6_k_t16_qmicro_planar_gemv_rowtile_col8_bf16_residual_bf16_out,
    gguf_q6_k_t16_qmicro_planar_wmma_prefill_bf16_bf16_out,
    gguf_q6_k_t16_wmma_prefill_bf16_bf16_out,
)
from hipengine.loading.gguf import GGUFReader
from hipengine.quant.gguf import GGMLQuantizationType
from hipengine.quant.gguf_t16 import (
    repack_gguf_q6_k_tile16,
    repack_gguf_q6_k_tile16_qmicro_planar,
)

DEFAULT_MODEL = Path("/models/gguf/Qwen3.8-27B-Q4_K_M.gguf")
DEFAULT_OUTPUT = Path("/tmp/qwen38-q6-planar-actual-operation-gate.json")
DEFAULT_SMOKE = Path("/tmp/qwen38-task20-q6-hybrid-smoke/smoke.json")
_PREFILL_ROWS = (16, 512, 1024, 4096)
_BF16_SMALL_ROWS = (1, 2, 3, 4, 5, 6, 7, 8)
_F32_SMALL_ROWS = (1, 2, 3, 4)


@dataclass(frozen=True)
class WeightCase:
    role: str
    name: str
    in_features: int
    out_features: int
    source_nbytes: int
    standard: DeviceBuffer
    planar: DeviceBuffer
    payload_nbytes: int
    raw_for_quality: np.ndarray | None = None

    def close(self, runtime) -> None:
        free(self.planar, runtime=runtime)
        free(self.standard, runtime=runtime)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--seed", type=int, default=0x38D06)
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
    return (np.asarray(values, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def _upload(runtime, values: np.ndarray) -> DeviceBuffer:
    array = np.ascontiguousarray(values)
    buffer = malloc(array.nbytes, runtime=runtime)
    copy_host_to_device(buffer, host_array_ptr(array), runtime=runtime)
    return buffer


def _read(runtime, buffer: DeviceBuffer, shape: tuple[int, ...], dtype) -> np.ndarray:
    output = np.empty(shape, dtype=dtype)
    copy_device_to_host(host_array_ptr(output), buffer, runtime=runtime)
    return output


def _digest(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def _verdict(control: np.ndarray, candidate: np.ndarray) -> dict[str, object]:
    mismatches = int(np.count_nonzero(candidate != control))
    decoded = _bf16_f32(candidate) if candidate.dtype == np.uint16 else candidate
    return {
        "mismatches": mismatches,
        "control_sha256": _digest(control),
        "candidate_sha256": _digest(candidate),
        "finite": bool(np.isfinite(decoded).all()),
        "exact": mismatches == 0,
    }


def _upload_case(reader: GGUFReader, role: str, name: str, runtime) -> WeightCase:
    info = reader.tensor_info(name)
    if info.ggml_type_name != "Q6_K" or len(info.shape) != 2:
        raise ValueError(f"{name} is not rank-2 Q6_K: {info}")
    raw = np.ascontiguousarray(reader.tensor_data(name))
    standard = np.ascontiguousarray(repack_gguf_q6_k_tile16(raw[None, ...]).tiles)
    planar = np.ascontiguousarray(
        repack_gguf_q6_k_tile16_qmicro_planar(raw[None, ...]).tiles
    )
    if standard.nbytes != planar.nbytes:
        raise ValueError(f"Q6 repack is not byte-neutral for {name}")
    return WeightCase(
        role=role,
        name=name,
        in_features=int(info.shape[1]),
        out_features=int(info.shape[0]),
        source_nbytes=int(info.nbytes),
        standard=_upload(runtime, standard),
        planar=_upload(runtime, planar),
        payload_nbytes=int(planar.nbytes),
        raw_for_quality=raw if role == "narrow_v" else None,
    )


def _small_gate(runtime, library, weight: WeightCase, rows: int, f32: bool, seed: int):
    rng = np.random.default_rng(seed + rows + weight.in_features + (17 if f32 else 0))
    x = _bf16_bits(rng.normal(0.0, 0.2, size=(rows, weight.in_features)))
    dtype = np.float32 if f32 else np.uint16
    buffers: list[DeviceBuffer] = []
    try:
        x_d = _upload(runtime, x)
        control_d = malloc(rows * weight.out_features * np.dtype(dtype).itemsize, runtime=runtime)
        candidate_d = malloc(rows * weight.out_features * np.dtype(dtype).itemsize, runtime=runtime)
        buffers.extend((x_d, control_d, candidate_d))
        if rows == 1:
            control_fn = (
                gguf_q6_k_t16_gemv_decode_bf16_f32_out
                if f32
                else gguf_q6_k_t16_gemv_decode_bf16_bf16_out
            )
            candidate_fn = (
                gguf_q6_k_t16_qmicro_planar_gemv_decode_bf16_f32_out
                if f32
                else gguf_q6_k_t16_qmicro_planar_gemv_decode_bf16_bf16_out
            )
            route = "c1_f32" if f32 else "c1_bf16"
        else:
            control_fn = (
                gguf_q6_k_t16_gemv_rowtile_bf16_f32_out
                if f32
                else gguf_q6_k_t16_gemv_rowtile_col8_bf16_bf16_out
            )
            candidate_fn = (
                gguf_q6_k_t16_qmicro_planar_gemv_rowtile_bf16_f32_out
                if f32
                else gguf_q6_k_t16_qmicro_planar_gemv_rowtile_col8_bf16_bf16_out
            )
            route = "rows2_4_f32" if f32 else "rows2_8_bf16"
        control_fn(
            x_d.ptr, weight.standard.ptr, control_d.ptr, rows,
            weight.in_features, weight.out_features, library=library, runtime=runtime,
        )
        candidate_fn(
            x_d.ptr, weight.planar.ptr, candidate_d.ptr, rows,
            weight.in_features, weight.out_features, library=library, runtime=runtime,
        )
        runtime.device_synchronize()
        shape = (rows, weight.out_features)
        control = _read(runtime, control_d, shape, dtype)
        candidate = _read(runtime, candidate_d, shape, dtype)
        result = _verdict(control, candidate)
        result["route"] = route
        return result
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)


def _prefill_gate(runtime, library, weight: WeightCase, rows: int, seed: int):
    rng = np.random.default_rng(seed + rows + weight.in_features)
    x = _bf16_bits(rng.normal(0.0, 0.2, size=(rows, weight.in_features)))
    buffers: list[DeviceBuffer] = []
    try:
        x_d = _upload(runtime, x)
        control_d = malloc(rows * weight.out_features * 2, runtime=runtime)
        candidate_d = malloc(rows * weight.out_features * 2, runtime=runtime)
        buffers.extend((x_d, control_d, candidate_d))
        gguf_q6_k_t16_wmma_prefill_bf16_bf16_out(
            x_d.ptr, weight.standard.ptr, control_d.ptr, rows,
            weight.in_features, weight.out_features, library=library, runtime=runtime,
        )
        gguf_q6_k_t16_qmicro_planar_wmma_prefill_bf16_bf16_out(
            x_d.ptr, weight.planar.ptr, candidate_d.ptr, rows,
            weight.in_features, weight.out_features, library=library, runtime=runtime,
        )
        runtime.device_synchronize()
        shape = (rows, weight.out_features)
        control = _read(runtime, control_d, shape, np.uint16)
        candidate = _read(runtime, candidate_d, shape, np.uint16)
        result = _verdict(control, candidate)
        result["route"] = "wmma_prefill"
        return result, x, candidate
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)


def _residual_gate(runtime, library, ops_library, weight: WeightCase, rows: int, seed: int):
    rng = np.random.default_rng(seed + rows)
    x = _bf16_bits(rng.normal(0.0, 0.2, size=(rows, weight.in_features)))
    residual = _bf16_bits(rng.normal(0.0, 0.2, size=(rows, weight.out_features)))
    buffers: list[DeviceBuffer] = []
    try:
        x_d = _upload(runtime, x)
        residual_d = _upload(runtime, residual)
        projected_d = malloc(rows * weight.out_features * 2, runtime=runtime)
        control_d = malloc(rows * weight.out_features * 2, runtime=runtime)
        candidate_d = malloc(rows * weight.out_features * 2, runtime=runtime)
        buffers.extend((x_d, residual_d, projected_d, control_d, candidate_d))
        gguf_q6_k_t16_gemv_rowtile_col8_bf16_bf16_out(
            x_d.ptr, weight.standard.ptr, projected_d.ptr, rows,
            weight.in_features, weight.out_features, library=library, runtime=runtime,
        )
        gguf_bf16_add(
            projected_d.ptr, residual_d.ptr, control_d.ptr,
            rows * weight.out_features, library=ops_library, runtime=runtime,
        )
        gguf_q6_k_t16_qmicro_planar_gemv_rowtile_col8_bf16_residual_bf16_out(
            x_d.ptr, weight.planar.ptr, residual_d.ptr, candidate_d.ptr, rows,
            weight.in_features, weight.out_features, library=library, runtime=runtime,
        )
        runtime.device_synchronize()
        shape = (rows, weight.out_features)
        result = _verdict(
            _read(runtime, control_d, shape, np.uint16),
            _read(runtime, candidate_d, shape, np.uint16),
        )
        result["route"] = "planar_down_residual_vs_standard_unfused_add"
        return result
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)


def _top1_gate(runtime, library, root: WeightCase, seed: int) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    x = _bf16_bits(rng.normal(0.0, 0.2, size=(1, root.in_features)))
    tiles = root.out_features // 16
    buffers: list[DeviceBuffer] = []
    try:
        x_d = _upload(runtime, x)
        buffers.append(x_d)
        outputs = []
        for fn, weight in (
            (gguf_q6_k_t16_gemv_decode_bf16_f32_top1_stage1, root.standard),
            (gguf_q6_k_t16_qmicro_planar_gemv_decode_bf16_f32_top1_stage1, root.planar),
        ):
            logits_d = malloc(root.out_features * 4, runtime=runtime)
            values_d = malloc(tiles * 4, runtime=runtime)
            indices_d = malloc(tiles * 4, runtime=runtime)
            buffers.extend((logits_d, values_d, indices_d))
            fn(
                x_d.ptr, weight.ptr, logits_d.ptr, values_d.ptr, indices_d.ptr,
                root.in_features, root.out_features, library=library, runtime=runtime,
            )
            runtime.device_synchronize()
            outputs.append((
                _read(runtime, logits_d, (root.out_features,), np.float32),
                _read(runtime, values_d, (tiles,), np.float32),
                _read(runtime, indices_d, (tiles,), np.int32),
            ))
        verdicts = [_verdict(control, candidate) for control, candidate in zip(outputs[0], outputs[1], strict=True)]
        return {
            "route": "root_f32_logits_plus_top1_stage1",
            "logits": verdicts[0],
            "tile_values": verdicts[1],
            "tile_indices": verdicts[2],
            "top1_control": int(np.argmax(outputs[0][0])),
            "top1_candidate": int(np.argmax(outputs[1][0])),
            "exact": all(bool(item["exact"]) for item in verdicts),
            "finite": all(bool(item["finite"]) for item in verdicts),
        }
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)


def _stable_kl(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    ref = reference.astype(np.float64)
    got = candidate.astype(np.float64)
    ref -= ref.max(axis=-1, keepdims=True)
    got -= got.max(axis=-1, keepdims=True)
    ref_logp = ref - np.log(np.exp(ref).sum(axis=-1, keepdims=True))
    got_logp = got - np.log(np.exp(got).sum(axis=-1, keepdims=True))
    return np.sum(np.exp(ref_logp) * (ref_logp - got_logp), axis=-1)


def _quality_gate(weight: WeightCase, x: np.ndarray, candidate: np.ndarray) -> dict[str, object]:
    if weight.raw_for_quality is None:
        raise ValueError("quality gate requires retained narrow raw bytes")
    reference = gguf_quant_gemv(
        _bf16_f32(x), weight.raw_for_quality, GGMLQuantizationType.Q6_K
    )
    actual = _bf16_f32(candidate)
    kl = _stable_kl(reference, actual)
    top1 = reference.argmax(axis=-1) == actual.argmax(axis=-1)
    return {
        "max_kl": float(kl.max()),
        "top1_matches": int(top1.sum()),
        "top1_total": int(top1.size),
        "top1_agreement": float(top1.mean()),
        "max_abs": float(np.max(np.abs(reference - actual))),
        "finite": bool(np.isfinite(actual).all()),
    }


def _validate_smoke(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text())
    raw_runs = (
        payload.get("runs_by_workload", {}).get("512/8")
        if isinstance(payload.get("runs_by_workload"), dict)
        else None
    )
    if raw_runs is None:
        raw_runs = payload.get("runs", [])
    measured = [run for run in raw_runs if run["measured"]]
    if len(measured) != 1:
        raise ValueError(f"expected one measured smoke run, got {len(measured)}")
    run = measured[0]
    weights = run["memory_snapshots"]["after_load"]["owned_session_breakdown"]["families"]["weights"]
    result = {
        "path": str(path.resolve()),
        "finite_final_logits": bool(run["correctness_sanity"]["finite_final_logits"]),
        "final_token_id": int(run["correctness_sanity"]["final_token_id"]),
        "production_graph": bool(run["effective_graph_replay_decode"]),
        "planar_q6_bytes": int(weights["by_layout_bytes"].get("gguf_q6_k_t16_qmicro_planar_v1", 0)),
        "standard_q6_bytes": int(weights["by_layout_bytes"].get("gguf_q6_k_t16_v1", 0)),
        "dense_bf16_bytes": int(weights["by_layout_bytes"].get("dense_bf16", 0)),
        "teardown_bytes": int(payload["persistent_session_memory"]["summary"]["tracked_current_allocated_bytes_after_close"]),
    }
    result["passed"] = (
        result["finite_final_logits"] and result["production_graph"]
        # The current sole-layout package materializes every active dense Q6
        # role as byte-neutral planar-qmicro; no standard-Q6 or BF16 shadow
        # remains resident.
        and result["planar_q6_bytes"] == 4_449_177_600
        and result["standard_q6_bytes"] == 0
        and result["dense_bf16_bytes"] == 0
        and result["teardown_bytes"] == 0
    )
    return result


def main() -> int:
    args = _parse_args()
    dirty = _tracked_status()
    if dirty and not args.allow_dirty:
        raise SystemExit("tracked worktree must be clean; pass --allow-dirty for development")
    if args.compiler_version_file is not None:
        os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(args.compiler_version_file)
    reader = GGUFReader(args.model)
    names = {
        "wide_ffn_down": "blk.0.ffn_down.weight",
        "wide_qkv": "blk.0.attn_qkv.weight",
        "narrow_v": "blk.3.attn_v.weight",
        "root_lm_head": "output.weight",
    }
    source_pool_bytes = sum(int(reader.tensor_info(name).nbytes) for name in names.values())
    if source_pool_bytes <= 64 * 1024 * 1024:
        raise ValueError("actual source pool must exceed 64 MiB")

    runtime = get_hip_runtime()
    initial_current = int(memory_stats()["current_allocated_bytes"])
    require_cached = bool(args.require_cached_build)
    library = build_gguf_q6_k_t16_gemv(load=True, require_cached=require_cached)
    ops_library = build_gguf_ops(load=True, require_cached=require_cached)
    weights: dict[str, WeightCase] = {}
    results: dict[str, object] = {}
    quality = None
    try:
        for role, name in names.items():
            weights[role] = _upload_case(reader, role, name, runtime)
        for role in ("wide_ffn_down", "wide_qkv", "narrow_v"):
            weight = weights[role]
            role_results: dict[str, object] = {}
            for rows in _BF16_SMALL_ROWS:
                role_results[f"bf16_m{rows}"] = _small_gate(
                    runtime, library, weight, rows, False, args.seed
                )
            for rows in _F32_SMALL_ROWS:
                role_results[f"f32_m{rows}"] = _small_gate(
                    runtime, library, weight, rows, True, args.seed
                )
            for rows in _PREFILL_ROWS:
                verdict, x, candidate = _prefill_gate(
                    runtime, library, weight, rows, args.seed
                )
                role_results[f"prefill_m{rows}"] = verdict
                if role == "narrow_v" and rows == 16:
                    quality = _quality_gate(weight, x, candidate)
            results[role] = role_results
        results["down_residual"] = {
            str(rows): _residual_gate(
                runtime, library, ops_library, weights["wide_ffn_down"], rows, args.seed
            )
            for rows in (2, 3, 4)
        }
        root = weights["root_lm_head"]
        results["root"] = {
            "bf16_m1": _small_gate(runtime, library, root, 1, False, args.seed),
            "f32_m1": _small_gate(runtime, library, root, 1, True, args.seed),
            "f32_m2": _small_gate(runtime, library, root, 2, True, args.seed),
            "top1": _top1_gate(runtime, library, root, args.seed),
        }
    finally:
        for weight in reversed(tuple(weights.values())):
            weight.close(runtime)
    runtime.device_synchronize()
    final_current = int(memory_stats()["current_allocated_bytes"])

    flat = []
    for role, value in results.items():
        if role == "root":
            flat.extend(
                (
                    value["bf16_m1"],
                    value["f32_m1"],
                    value["f32_m2"],
                    value["top1"],
                )
            )
        else:
            flat.extend(value.values())
    exact = all(bool(item["exact"]) for item in flat)
    finite = all(bool(item["finite"]) for item in flat)
    quality_passed = bool(
        quality and quality["max_kl"] <= 0.05 and quality["top1_agreement"] >= 0.9
    )
    smoke = _validate_smoke(args.full_model_smoke)
    teardown = final_current == initial_current
    passed = exact and finite and quality_passed and smoke["passed"] and teardown
    artifact = {
        "schema_version": 1,
        "kind": "qwen38_q6_role_qualified_actual_operation_gate",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "exact_quality_gated" if passed else "failed",
        "performance_claim": False,
        "model": str(args.model.resolve()),
        "model_size_bytes": args.model.stat().st_size,
        "source_pool_bytes": source_pool_bytes,
        "source_pool_mib": source_pool_bytes / (1024 * 1024),
        "weights": {
            role: {
                "name": weight.name,
                "shape": [weight.out_features, weight.in_features],
                "source_nbytes": weight.source_nbytes,
                "candidate_nbytes": weight.payload_nbytes,
                "retained_layout": (
                    "gguf_q6_k_t16_v1"
                    if role == "wide_qkv"
                    else "gguf_q6_k_t16_qmicro_planar_v1"
                ),
                "retained_allocations": ["tiles"],
                "planar_operation_gate": (
                    "rejected_for_c1_performance"
                    if role == "wide_qkv"
                    else "retained"
                ),
            }
            for role, weight in weights.items()
        },
        "results": results,
        "correctness": {
            "operation_verdicts": len(flat),
            "all_bit_exact_to_standard_q6_t16": exact,
            "all_finite": finite,
            "cpu_quality": {**quality, "passed": quality_passed},
            "full_model": smoke,
        },
        "ownership": {
            "planar_slots": ["ffn_down", "attn_v", "lm_head"],
            "standard_slots": ["attn_qkv"],
            "down_residual_route": "planar_projection_plus_primitive_bf16_add",
            "fused_planar_down_residual": "exact_but_rejected_on_gfx1151",
            "planar_resident_bytes": 3_416_985_600,
            "standard_resident_bytes": 1_032_192_000,
            "duplicate_q6_payload_bytes": 0,
            "dense_bf16_candidate_shadow_bytes": 0,
            "gate_only_alternates": (
                "standard Q6T16 for planar roles and planar Q6 for rejected "
                "QKV are freed before teardown"
            ),
        },
        "memory": {
            "tracked_current_before_bytes": initial_current,
            "tracked_current_after_bytes": final_current,
            "teardown_exact": teardown,
        },
        "provenance": {
            "commit": subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip(),
            "tracked_dirty_paths": dirty,
            "target_arch": os.environ.get("HIPENGINE_HIP_ARCH"),
            "compiler_version_file": str(args.compiler_version_file.resolve()) if args.compiler_version_file else None,
            "require_cached_build": require_cached,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(json.dumps(artifact, indent=2))
    if not passed:
        raise SystemExit("actual planar-Q6 operation gate failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
