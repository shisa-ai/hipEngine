#!/usr/bin/env python3
"""Audit the layer-0 attn_norm mismatch with CPU RMSNorm formula candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.loading.gguf import GGUFReader  # noqa: E402
from hipengine.loading.qwen35_gguf import build_qwen35_gguf_tensor_map  # noqa: E402
from hipengine.loading.qwen35_gguf_materialize import (  # noqa: E402
    plan_qwen35_gguf_materialization,
)
from scripts.llamacpp_mtp_compare_input_embed import (  # noqa: E402
    f32_to_bf16_roundtrip,
)
from scripts.llamacpp_mtp_compare_layer0_attn_norm import (  # noqa: E402
    capture_hipengine_layer0_attn_norm,
)
from scripts.llamacpp_mtp_compare_hidden_seed import redact_values  # noqa: E402

DEFAULT_INPUT_COMPARE = Path("benchmarks/results/mtp-gguf-iter318-input-embed-compare.json")
DEFAULT_ATTN_COMPARE = Path("benchmarks/results/mtp-gguf-iter320-layer0-attn-norm-compare.json")
DEFAULT_OUTPUT = Path("benchmarks/results/mtp-gguf-iter321-layer0-attn-norm-formula-audit.json")
DEFAULT_MODEL = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")

HipCaptureFn = Callable[[Path, tuple[int, ...], int, int, int | None], dict[str, Any]]
WeightLoaderFn = Callable[[Path, int], tuple[np.ndarray, float, dict[str, Any]]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-compare", type=Path, default=DEFAULT_INPUT_COMPARE)
    parser.add_argument("--attn-norm-compare", type=Path, default=DEFAULT_ATTN_COMPARE)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-sequence-length", type=int)
    parser.add_argument("--iteration", type=int, default=321)
    args = parser.parse_args()

    artifact = audit_layer0_attn_norm_formula(
        input_compare_path=args.input_compare,
        attn_norm_compare_path=args.attn_norm_compare,
        model_path=args.model,
        max_sequence_length=args.max_sequence_length,
        iteration=args.iteration,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    best_llama = artifact["best_candidates"]["vs_llamacpp_attn_norm"]
    best_hip = artifact["best_candidates"]["vs_hipengine_attn_norm"]
    print(
        json.dumps(
            {
                "status": artifact["status"],
                "conclusion": artifact["conclusion"],
                "best_vs_llama": best_llama["name"],
                "best_vs_llama_rmse": best_llama["delta"]["rmse"],
                "best_vs_hip": best_hip["name"],
                "best_vs_hip_rmse": best_hip["delta"]["rmse"],
                "next_action": artifact["next_action"],
            },
            indent=2,
        )
    )


def audit_layer0_attn_norm_formula(
    *,
    input_compare_path: Path,
    attn_norm_compare_path: Path,
    model_path: Path | None = None,
    max_sequence_length: int | None = None,
    iteration: int = 321,
    hip_capture_fn: HipCaptureFn = capture_hipengine_layer0_attn_norm,
    weight_loader: WeightLoaderFn | None = None,
) -> dict[str, Any]:
    input_compare = json.loads(input_compare_path.read_text())
    attn_compare = json.loads(attn_norm_compare_path.read_text())
    layer_id = int(attn_compare.get("layer_id", 0))
    if layer_id != 0:
        raise ValueError("formula audit currently requires layer_id=0")
    resolved_model = Path(model_path or attn_compare["model"])
    prompt_tokens = tuple(int(token) for token in attn_compare["prompt_tokens"])
    position = int(attn_compare["position"])
    input_f32 = load_capture_values(input_compare["llamacpp_capture"])
    llama_attn = load_capture_values(attn_compare["llamacpp_capture"])
    llama_attn_bf16 = bf16_roundtrip_array(llama_attn)
    selected_weight_loader = weight_loader or load_layer0_attn_norm_weight
    weight, eps, weight_metadata = selected_weight_loader(resolved_model, layer_id)
    hip_capture = hip_capture_fn(
        resolved_model,
        prompt_tokens,
        position,
        layer_id,
        max_sequence_length,
    )
    hip_values = np.asarray(hip_capture.get("values", ()), dtype=np.float32)
    candidates = build_formula_candidates(
        input_f32=input_f32,
        weight=weight,
        eps=float(eps),
        eps_alternates=(0.0, 1.0e-5),
    )
    candidate_records = [
        summarize_candidate(candidate, llama_attn, llama_attn_bf16, hip_values)
        for candidate in candidates
    ]
    best = best_candidates(candidate_records)
    conclusion = classify_formula_audit(best, hip_capture)
    return {
        "schema": 1,
        "kind": "layer0_attn_norm_formula_audit",
        "date": "2026-06-20",
        "loop": "mtp-gguf/run-20260615-103738",
        "iteration": int(iteration),
        "status": status_from_conclusion(conclusion, hip_capture),
        "input_compare_path": str(input_compare_path),
        "attn_norm_compare_path": str(attn_norm_compare_path),
        "model": str(resolved_model),
        "layer_id": layer_id,
        "position": position,
        "token_id": int(attn_compare["token_id"]),
        "prompt_tokens": list(prompt_tokens),
        "input_capture": summarize_array(input_f32),
        "input_bf16_roundtrip": summarize_array(bf16_roundtrip_array(input_f32)),
        "llamacpp_attn_norm": summarize_array(llama_attn),
        "llamacpp_attn_norm_bf16_roundtrip": summarize_array(llama_attn_bf16),
        "hipengine_attn_norm": redact_values(hip_capture),
        "weight": weight_metadata,
        "formula": {
            "rmsnorm": "x * rsqrt(mean(x^2) + eps) * weight",
            "model_eps": float(eps),
            "candidate_count": len(candidate_records),
        },
        "candidates": candidate_records,
        "best_candidates": best,
        "conclusion": conclusion,
        "external_checkout_modified": False,
        "next_action": next_action(conclusion),
    }


def build_formula_candidates(
    *,
    input_f32: np.ndarray,
    weight: np.ndarray,
    eps: float,
    eps_alternates: tuple[float, ...],
) -> list[dict[str, Any]]:
    input_sources = {
        "input_f32": np.asarray(input_f32, dtype=np.float32),
        "input_bf16": bf16_roundtrip_array(input_f32),
    }
    weight_sources = {
        "weight_f32": np.asarray(weight, dtype=np.float32),
        "weight_bf16": bf16_roundtrip_array(weight),
    }
    eps_sources = {"eps_model": float(eps)}
    for value in eps_alternates:
        eps_sources[f"eps_{value:g}"] = float(value)
    candidates: list[dict[str, Any]] = []
    for input_name, input_values in input_sources.items():
        for weight_name, weight_values in weight_sources.items():
            for eps_name, eps_value in eps_sources.items():
                f32 = rmsnorm_f32(input_values, weight_values, eps_value)
                candidates.append(
                    candidate_record(
                        input_name,
                        weight_name,
                        eps_name,
                        eps_value,
                        "f32_out",
                        f32,
                    )
                )
                candidates.append(
                    candidate_record(
                        input_name,
                        weight_name,
                        eps_name,
                        eps_value,
                        "bf16_out",
                        bf16_roundtrip_array(f32),
                    )
                )
    return candidates


def candidate_record(
    input_name: str,
    weight_name: str,
    eps_name: str,
    eps_value: float,
    output_name: str,
    values: np.ndarray,
) -> dict[str, Any]:
    return {
        "name": f"{input_name}_{weight_name}_{eps_name}_{output_name}",
        "input_source": input_name,
        "weight_source": weight_name,
        "eps_source": eps_name,
        "eps": float(eps_value),
        "output_dtype": output_name,
        "values": np.asarray(values, dtype=np.float32),
    }


def summarize_candidate(
    candidate: Mapping[str, Any],
    llama_attn: np.ndarray,
    llama_attn_bf16: np.ndarray,
    hip_values: np.ndarray,
) -> dict[str, Any]:
    values = np.asarray(candidate["values"], dtype=np.float32)
    record = {
        "name": candidate["name"],
        "input_source": candidate["input_source"],
        "weight_source": candidate["weight_source"],
        "eps_source": candidate["eps_source"],
        "eps": float(candidate["eps"]),
        "output_dtype": candidate["output_dtype"],
        "summary": summarize_array(values),
        "delta_vs_llamacpp_attn_norm": delta_summary(values, llama_attn),
        "delta_vs_llamacpp_attn_norm_bf16": delta_summary(values, llama_attn_bf16),
    }
    if hip_values.size:
        record["delta_vs_hipengine_attn_norm"] = delta_summary(values, hip_values)
    else:
        record["delta_vs_hipengine_attn_norm"] = {
            "available": False,
            "reason": "hipengine_capture_unavailable",
        }
    return record


def best_candidates(candidate_records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "vs_llamacpp_attn_norm": best_by_rmse(
            candidate_records,
            "delta_vs_llamacpp_attn_norm",
        ),
        "vs_llamacpp_attn_norm_bf16": best_by_rmse(
            candidate_records,
            "delta_vs_llamacpp_attn_norm_bf16",
        ),
        "vs_hipengine_attn_norm": best_by_rmse(
            candidate_records,
            "delta_vs_hipengine_attn_norm",
        ),
    }


def best_by_rmse(candidate_records: list[dict[str, Any]], delta_key: str) -> dict[str, Any]:
    available = [
        candidate
        for candidate in candidate_records
        if candidate.get(delta_key, {}).get("available")
        and candidate.get(delta_key, {}).get("shape_match")
    ]
    if not available:
        return {"available": False, "delta_key": delta_key}
    best = min(
        available,
        key=lambda item: (item[delta_key]["rmse"], item[delta_key]["max_abs_diff"]),
    )
    return {
        "available": True,
        "name": best["name"],
        "input_source": best["input_source"],
        "weight_source": best["weight_source"],
        "eps_source": best["eps_source"],
        "eps": best["eps"],
        "output_dtype": best["output_dtype"],
        "delta": best[delta_key],
    }


def classify_formula_audit(best: Mapping[str, Any], hip_capture: Mapping[str, Any]) -> str:
    if hip_capture.get("status") != "captured":
        return "attn_norm_formula_audit_unavailable"
    best_llama = best["vs_llamacpp_attn_norm"]
    best_hip = best["vs_hipengine_attn_norm"]
    if not best_llama.get("available") or not best_hip.get("available"):
        return "attn_norm_formula_audit_unavailable"
    if (
        best_llama["name"] == "input_f32_weight_f32_eps_model_f32_out"
        and best_llama["delta"].get("exact_match")
        and best_hip["name"] == "input_bf16_weight_f32_eps_model_bf16_out"
        and best_hip["delta"].get("exact_match")
    ):
        return "attn_norm_mismatch_explained_by_input_activation_bf16_contraction"
    if best_llama["eps_source"] != "eps_model" or best_hip["eps_source"] != "eps_model":
        return "attn_norm_mismatch_suggests_epsilon_mismatch"
    if best_hip["weight_source"] != "weight_f32":
        return "attn_norm_mismatch_suggests_weight_materialization_mismatch"
    return "attn_norm_formula_audit_needs_manual_review"


def status_from_conclusion(conclusion: str, hip_capture: Mapping[str, Any]) -> str:
    if hip_capture.get("status") != "captured":
        return str(hip_capture.get("status", "hipengine_capture_failed"))
    if conclusion.endswith("unavailable"):
        return "unavailable"
    return "ready"


def next_action(conclusion: str) -> str:
    if conclusion == "attn_norm_mismatch_explained_by_input_activation_bf16_contraction":
        return "decide_whether_to_add_f32_activation_path_or_adjust_llamacpp_oracle_dtype"
    if conclusion == "attn_norm_mismatch_suggests_epsilon_mismatch":
        return "audit_rmsnorm_epsilon_plumbing"
    if conclusion == "attn_norm_mismatch_suggests_weight_materialization_mismatch":
        return "audit_attn_norm_weight_materialization"
    if conclusion == "attn_norm_formula_audit_unavailable":
        return "rerun_layer0_attn_norm_formula_audit_on_rocm_host"
    return "inspect_layer0_attn_norm_formula_audit_candidates"


def load_layer0_attn_norm_weight(
    model_path: Path,
    layer_id: int,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    reader = GGUFReader(model_path)
    model_map = build_qwen35_gguf_tensor_map(reader.info)
    plan = plan_qwen35_gguf_materialization(model_map)
    tensor_name = f"blk.{layer_id}.attn_norm.weight"
    weight = np.asarray(reader.dequantize_tensor(tensor_name), dtype=np.float32)
    eps_key = "qwen35moe.attention.layer_norm_rms_epsilon"
    eps = float(reader.info.metadata.get(eps_key, model_map.config.rms_norm_eps))
    spec = plan.layer_specs[layer_id]["attn_norm"]
    return weight, eps, {
        "tensor_name": tensor_name,
        "ggml_type": reader.tensor_info(tensor_name).ggml_type_name,
        "shape": list(weight.shape),
        "summary": summarize_array(weight),
        "config_eps": float(model_map.config.rms_norm_eps),
        "metadata_eps_key": eps_key,
        "metadata_eps": eps,
        "materialization_slot": spec.slot_path,
        "materialization_layout": spec.layout,
        "materialization_quant_key": spec.quant_key,
    }


def load_capture_values(capture: Mapping[str, Any]) -> np.ndarray:
    binary = Path(str(capture.get("binary_path", "")))
    if not binary.exists():
        raise FileNotFoundError(f"capture binary does not exist: {binary}")
    return unpack_float32(binary.read_bytes())


def rmsnorm_f32(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    x32 = np.asarray(x, dtype=np.float32)
    w32 = np.asarray(weight, dtype=np.float32)
    if x32.shape != w32.shape:
        raise ValueError("RMSNorm input and weight must have the same shape")
    variance = float(np.mean(np.square(x32, dtype=np.float32), dtype=np.float32))
    inv_rms = np.float32(1.0 / math.sqrt(variance + float(eps)))
    return np.asarray(x32 * inv_rms * w32, dtype=np.float32)


def bf16_roundtrip_array(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            f32_to_bf16_roundtrip(float(value))
            for value in np.asarray(values, dtype=np.float32)
        ],
        dtype=np.float32,
    )


def delta_summary(
    actual: np.ndarray,
    reference: np.ndarray,
    *,
    exact_atol: float = 0.0,
) -> dict[str, Any]:
    actual32 = np.asarray(actual, dtype=np.float32)
    reference32 = np.asarray(reference, dtype=np.float32)
    if actual32.shape != reference32.shape:
        return {
            "available": True,
            "shape_match": False,
            "actual_shape": list(actual32.shape),
            "reference_shape": list(reference32.shape),
        }
    diff = actual32 - reference32
    abs_diff = np.abs(diff)
    max_abs = float(np.max(abs_diff)) if abs_diff.size else 0.0
    return {
        "available": True,
        "shape_match": True,
        "count": int(actual32.size),
        "actual_sha256": sha256_float32(actual32),
        "reference_sha256": sha256_float32(reference32),
        "max_abs_diff": max_abs,
        "mean_abs_diff": float(np.mean(abs_diff)) if abs_diff.size else 0.0,
        "rmse": float(np.sqrt(np.mean(diff * diff))) if diff.size else 0.0,
        "diff_samples": [round(float(value), 8) for value in diff[:8]],
        "top_abs_diff": top_abs_diff_entries(actual32, reference32),
        "exact_match": bool(max_abs <= exact_atol),
    }


def summarize_array(values: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(arr)
    if arr.size == 0:
        stats = {
            "count": 0,
            "finite_count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "l2": 0.0,
        }
    else:
        stats = {
            "count": int(arr.size),
            "finite_count": int(np.count_nonzero(finite)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "mean": float(np.mean(arr)),
            "l2": float(np.linalg.norm(arr.astype(np.float64))),
        }
    return {
        "shape": list(arr.shape),
        "count": int(arr.size),
        "sha256": sha256_float32(arr),
        "stats": stats,
        "samples": [round(float(value), 8) for value in arr[:8]],
        "top_abs": top_abs_entries(arr),
    }


def top_abs_entries(values: np.ndarray, *, limit: int = 8) -> list[dict[str, Any]]:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return []
    indices = np.argsort(-np.abs(arr))[:limit]
    return [
        {
            "index": int(index),
            "value": round(float(arr[index]), 8),
            "abs": round(float(abs(arr[index])), 8),
        }
        for index in indices
    ]


def top_abs_diff_entries(
    actual: np.ndarray,
    reference: np.ndarray,
    *,
    limit: int = 8,
) -> list[dict[str, Any]]:
    actual32 = np.asarray(actual, dtype=np.float32)
    reference32 = np.asarray(reference, dtype=np.float32)
    diff = actual32 - reference32
    if diff.size == 0:
        return []
    indices = np.argsort(-np.abs(diff))[:limit]
    return [
        {
            "index": int(index),
            "actual": round(float(actual32[index]), 8),
            "reference": round(float(reference32[index]), 8),
            "diff": round(float(diff[index]), 8),
            "abs_diff": round(float(abs(diff[index])), 8),
        }
        for index in indices
    ]


def pack_float32(values: np.ndarray) -> bytes:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    return struct.pack("<" + "f" * int(arr.size), *[float(value) for value in arr])


def unpack_float32(data: bytes) -> np.ndarray:
    if len(data) % 4:
        raise ValueError("float32 binary length must be divisible by 4")
    if not data:
        return np.empty((0,), dtype=np.float32)
    return np.asarray(struct.unpack("<" + "f" * (len(data) // 4), data), dtype=np.float32)


def sha256_float32(values: np.ndarray) -> str:
    return hashlib.sha256(pack_float32(values)).hexdigest()


if __name__ == "__main__":
    main()
