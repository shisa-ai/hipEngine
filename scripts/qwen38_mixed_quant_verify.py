#!/usr/bin/env python3
"""Verify one Qwen3.8 mixed-quant GGUF against its immutable plan."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from math import prod
from pathlib import Path
from typing import Any

from hipengine.loading import load_gguf_index
from hipengine.loading.qwen35_gguf import build_qwen35_gguf_tensor_map
from hipengine.loading.qwen35_gguf_materialize import plan_qwen35_gguf_materialization
from hipengine.loading.qwen35_gguf_nextn import (
    QWEN38_NATIVE_XL_OUTPUT_TYPE_MANIFEST_SHA256,
    QWEN38_NATIVE_XL_QUANT_VARIANT,
    build_qwen35_gguf_nextn_tensor_map,
)
from hipengine.loading.qwen35_gguf_nextn_materialize import (
    plan_qwen35_gguf_nextn_materialization,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _shape_manifest(tensors: dict[str, Any]) -> str:
    rows = [
        f"{name}\0{','.join(str(int(dim)) for dim in tensors[name].shape)}"
        for name in sorted(tensors)
    ]
    return hashlib.sha256("\n".join(rows).encode()).hexdigest()


def _type_manifest(tensors: dict[str, Any]) -> str:
    rows = [f"{name}={tensors[name].ggml_type_name}" for name in sorted(tensors)]
    return hashlib.sha256("\n".join(rows).encode()).hexdigest()


def verify(model_path: Path, plan_path: Path, *, hash_model: bool) -> dict[str, Any]:
    model = model_path.resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    info = load_gguf_index(model)
    actual = {tensor.name: tensor for tensor in info.tensors}
    expected_types = {
        str(name): str(qtype) for name, qtype in plan["output_types"].items()
    }
    failures: list[str] = []
    if set(actual) != set(expected_types):
        actual_only = sorted(set(actual) - set(expected_types))[:8]
        plan_only = sorted(set(expected_types) - set(actual))[:8]
        failures.append(
            f"tensor inventory mismatch: actual_only={actual_only}, plan_only={plan_only}"
        )
    mismatches = [
        f"{name}: expected {expected_types[name]}, got {actual[name].ggml_type_name}"
        for name in sorted(set(actual) & set(expected_types))
        if actual[name].ggml_type_name != expected_types[name]
    ]
    if mismatches:
        failures.append("type mismatches: " + "; ".join(mismatches[:8]))
    shape_sha256 = _shape_manifest(actual)
    if shape_sha256 != plan["tensor_shape_manifest_sha256"]:
        failures.append("tensor shape manifest SHA-256 differs from plan")
    type_sha256 = _type_manifest(actual)
    if type_sha256 != plan["output_type_manifest_sha256"]:
        failures.append("output type manifest SHA-256 differs from plan")
    metadata_contract = {
        "hipengine.quant.variant": QWEN38_NATIVE_XL_QUANT_VARIANT,
        "hipengine.quant.output_type_manifest_sha256": (
            QWEN38_NATIVE_XL_OUTPUT_TYPE_MANIFEST_SHA256
        ),
        "hipengine.quant.source_revision": str(plan["source"]["revision"]),
    }
    metadata_results = {}
    for key, expected in metadata_contract.items():
        actual_value = str(info.metadata.get(key, "") or "")
        metadata_results[key] = {
            "expected": expected,
            "actual": actual_value,
            "passed": actual_value == expected,
        }
        if actual_value != expected:
            failures.append(f"metadata {key}: expected {expected!r}, got {actual_value!r}")
    expected_payload = int(plan["budget"]["projected_tensor_payload_bytes"])
    if int(info.total_tensor_nbytes) != expected_payload:
        failures.append(
            f"tensor payload bytes: expected {expected_payload}, got {info.total_tensor_nbytes}"
        )

    loader = {"target": None, "nextn": None}
    try:
        target_map = build_qwen35_gguf_tensor_map(info)
        target_plan = plan_qwen35_gguf_materialization(target_map, decode_repack=True)
        loader["target"] = {
            "passed": True,
            "weights": len(target_plan.specs),
            "ignored_block_ids": list(target_map.config.ignored_block_ids),
        }
    except Exception as exc:
        loader["target"] = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}
        failures.append("target loader/materialization plan failed")
    try:
        nextn_map = build_qwen35_gguf_nextn_tensor_map(info)
        nextn_plan = plan_qwen35_gguf_nextn_materialization(
            nextn_map,
            decode_repack=True,
            dense_q4_t16=True,
            dense_q5_t16_ssm_out=True,
            dense_q5_t16_h5120=True,
            dense_q6_qmicro_planar=True,
        )
        loader["nextn"] = {
            "passed": True,
            "weights": len(nextn_plan.specs),
            "block_id": nextn_map.block_id,
        }
    except Exception as exc:
        loader["nextn"] = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}
        failures.append("NextN loader/materialization plan failed")

    total_params = sum(int(prod(tensor.shape)) for tensor in actual.values())
    effective_bpw = 8.0 * int(info.total_tensor_nbytes) / total_params
    return {
        "schema": 1,
        "kind": "qwen38_native_mixed_quant_verification",
        "status": "passed" if not failures else "failed",
        "model": {
            "path": str(model),
            "size_bytes": model.stat().st_size,
            "sha256": _sha256(model) if hash_model else None,
            "tensor_count": len(actual),
            "tensor_payload_bytes": int(info.total_tensor_nbytes),
            "total_parameters_including_untied_embedding_and_head": total_params,
            "effective_bpw": effective_bpw,
            "types": dict(sorted(Counter(t.ggml_type_name for t in actual.values()).items())),
            "shape_manifest_sha256": shape_sha256,
            "output_type_manifest_sha256": type_sha256,
        },
        "plan": {"path": str(plan_path.resolve()), "sha256": _sha256(plan_path)},
        "metadata_contract": metadata_results,
        "loader": loader,
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--hash-model", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = verify(args.model, args.plan, hash_model=bool(args.hash_model))
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "failures": payload["failures"]}, indent=2))
    return 0 if payload["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
