#!/usr/bin/env python3
"""Project a measured Qwen3.8 GGUF sensitivity map onto native codecs.

This tool writes two immutable planning artifacts:

* an exact, anchored llama.cpp ``--tensor-type-file``; and
* a JSON manifest binding source/template inventories, projection rules,
  per-tensor output types, byte budget, imatrix identity, and NextN floors.

It does not quantize or claim quality. Use a BF16/F16 source and the recorded
quantizer command, then gate the produced artifact separately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from math import prod
from pathlib import Path
from typing import Any, Iterable

from hipengine.loading import load_gguf_index
from hipengine.quant.gguf import GGMLQuantizationType, nbytes_for_shape

SCHEMA = 1
KIND = "qwen38_native_mixed_quant_plan"
NATIVE_TYPES = frozenset(("F32", "Q4_K", "Q5_K", "Q6_K", "Q8_0"))
PROJECTION = {
    "IQ4_XS": "Q4_K",
    "IQ4_NL": "Q4_K",
    "Q3_K": "Q4_K",
    "IQ3_S": "Q4_K",
}
NEXTN_MINIMUMS = {
    "blk.64.attn_k.weight": "Q8_0",
    "blk.64.attn_output.weight": "Q6_K",
    "blk.64.attn_q.weight": "Q6_K",
    "blk.64.attn_v.weight": "Q8_0",
    "blk.64.ffn_down.weight": "Q6_K",
    "blk.64.ffn_gate.weight": "Q6_K",
    "blk.64.ffn_up.weight": "Q6_K",
    "blk.64.nextn.eh_proj.weight": "Q6_K",
    "output.weight": "Q6_K",
}
_TYPE_RANK = {"Q4_K": 4, "Q5_K": 5, "Q6_K": 6, "Q8_0": 8, "F32": 32}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inventory(paths: Iterable[Path]) -> tuple[dict[str, Any], dict[str, Any]]:
    tensors: dict[str, Any] = {}
    metadata: dict[str, Any] = {}
    for path in paths:
        info = load_gguf_index(path)
        if not metadata:
            metadata = dict(info.metadata)
        for tensor in info.tensors:
            if tensor.name in tensors:
                raise ValueError(f"duplicate tensor across source shards: {tensor.name}")
            tensors[tensor.name] = tensor
    return tensors, metadata


def _project_type(type_name: str) -> str:
    projected = PROJECTION.get(str(type_name), str(type_name))
    if projected not in NATIVE_TYPES:
        raise ValueError(
            f"sensitivity template type {type_name!r} has no declared native projection"
        )
    return projected


def build_plan(
    *,
    source_paths: tuple[Path, ...],
    template_path: Path,
    imatrix_path: Path,
    source_revision: str,
    template_revision: str,
    source_sha256: tuple[str, ...],
    template_sha256: str,
    imatrix_sha256: str,
    target_bpw: float,
    hard_cap_bpw: float,
    hash_sources: bool,
) -> dict[str, Any]:
    source, source_metadata = _inventory(source_paths)
    template_info = load_gguf_index(template_path)
    template = {tensor.name: tensor for tensor in template_info.tensors}
    if set(source) != set(template):
        raise ValueError(
            "source/template tensor inventories differ: "
            f"source_only={sorted(set(source) - set(template))[:8]}, "
            f"template_only={sorted(set(template) - set(source))[:8]}"
        )
    if len(source) != 866 or "blk.64.nextn.eh_proj.weight" not in source:
        raise ValueError("Qwen3.8 plan requires the complete 866-tensor trailing-NextN inventory")
    if not imatrix_path.is_file():
        raise FileNotFoundError(f"importance matrix not found: {imatrix_path}")
    if len(source_sha256) != len(source_paths):
        raise ValueError("one --source-sha256 is required for every --source shard")
    for label, value in (
        ("template_sha256", template_sha256),
        ("imatrix_sha256", imatrix_sha256),
        *((f"source_sha256[{index}]", value) for index, value in enumerate(source_sha256)),
    ):
        if not re.fullmatch(r"[0-9a-f]{64}", str(value)):
            raise ValueError(f"{label} must be a lowercase SHA-256")
    actual_imatrix_sha256 = _sha256(imatrix_path)
    if actual_imatrix_sha256 != imatrix_sha256:
        raise ValueError("importance matrix SHA-256 does not match the declared artifact")
    local_source_sha256 = None
    if hash_sources:
        local_source_sha256 = {}
        for path, expected in zip(source_paths, source_sha256, strict=True):
            actual = _sha256(path)
            if actual != expected:
                raise ValueError(f"source shard SHA-256 mismatch: {path}")
            local_source_sha256[str(path.resolve())] = actual

    output_types: dict[str, str] = {}
    template_types: dict[str, str] = {}
    promotions: list[dict[str, str]] = []
    inventory_records: list[str] = []
    projected_counts: Counter[str] = Counter()
    template_counts: Counter[str] = Counter()
    total_params = 0
    projected_bytes = 0
    for name in sorted(source):
        src = source[name]
        donor = template[name]
        if tuple(src.shape) != tuple(donor.shape):
            raise ValueError(f"shape mismatch for {name}")
        output_type = _project_type(donor.ggml_type_name)
        nparams = int(prod(int(dim) for dim in src.shape))
        nbytes = int(nbytes_for_shape(src.shape, GGMLQuantizationType[output_type]))
        total_params += nparams
        projected_bytes += nbytes
        projected_counts[output_type] += 1
        template_counts[donor.ggml_type_name] += 1
        output_types[name] = output_type
        template_types[name] = donor.ggml_type_name
        inventory_records.append(
            f"{name}\0{','.join(str(int(dim)) for dim in src.shape)}\0{src.ggml_type_name}"
        )
        if output_type != donor.ggml_type_name:
            promotions.append(
                {"name": name, "template_type": donor.ggml_type_name, "output_type": output_type}
            )

    for name, minimum in NEXTN_MINIMUMS.items():
        actual = output_types[name]
        if _TYPE_RANK[actual] < _TYPE_RANK[minimum]:
            raise ValueError(f"precision floor failed for {name}: {actual} < {minimum}")

    effective_bpw = 8.0 * projected_bytes / total_params
    inventory_sha256 = hashlib.sha256("\n".join(inventory_records).encode()).hexdigest()
    type_manifest_sha256 = hashlib.sha256(
        "\n".join(f"{name}={output_types[name]}" for name in sorted(output_types)).encode()
    ).hexdigest()
    if effective_bpw > float(hard_cap_bpw):
        raise ValueError(
            f"projected effective bpw {effective_bpw:.6f} exceeds hard cap {hard_cap_bpw:.6f}"
        )
    return {
        "schema": SCHEMA,
        "kind": KIND,
        "status": "planned_not_quantized",
        "source": {
            "paths": [str(path.resolve()) for path in source_paths],
            "revision": str(source_revision),
            "tensor_count": len(source),
            "types": dict(sorted(Counter(t.ggml_type_name for t in source.values()).items())),
            "base_model_repo_url": source_metadata.get("general.base_model.0.repo_url"),
            "declared_sha256": {
                str(path.resolve()): digest
                for path, digest in zip(source_paths, source_sha256, strict=True)
            },
            "verified_local_sha256": local_source_sha256,
        },
        "sensitivity_template": {
            "path": str(template_path.resolve()),
            "revision": str(template_revision),
            "tensor_count": len(template),
            "types": dict(sorted(template_counts.items())),
            "full_artifact_sha256": str(template_sha256),
            "local_header_only": True,
        },
        "calibration": {
            "imatrix_path": str(imatrix_path.resolve()),
            "imatrix_sha256": actual_imatrix_sha256,
            "template_dataset": template_info.metadata.get("quantize.imatrix.dataset"),
            "template_chunks_count": template_info.metadata.get("quantize.imatrix.chunks_count"),
            "template_entries_count": template_info.metadata.get("quantize.imatrix.entries_count"),
            "benchmark_prompt_overlap_allowed": False,
        },
        "budget": {
            "target_bpw": float(target_bpw),
            "hard_cap_bpw": float(hard_cap_bpw),
            "projected_effective_bpw": effective_bpw,
            "projected_tensor_payload_bytes": projected_bytes,
            "total_parameters_including_untied_embedding_and_head": total_params,
            "projected_type_counts": dict(sorted(projected_counts.items())),
        },
        "projection": {
            "native_types": sorted(NATIVE_TYPES),
            "unsupported_type_map": dict(sorted(PROJECTION.items())),
            "policy": "promote unsupported sensitivity-template codecs; never requantize a quantized source",
            "nextn_minimums": dict(sorted(NEXTN_MINIMUMS.items())),
            "promotions": promotions,
        },
        "tensor_inventory_sha256": inventory_sha256,
        "output_type_manifest_sha256": type_manifest_sha256,
        "sensitivity_template_types": template_types,
        "output_types": output_types,
    }


def write_tensor_type_file(plan: dict[str, Any], path: Path) -> None:
    lines = [
        f"^{re.escape(str(name))}$={output_type}"
        for name, output_type in sorted(plan["output_types"].items())
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--sensitivity-template", type=Path, required=True)
    parser.add_argument("--imatrix", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--template-revision", required=True)
    parser.add_argument("--source-sha256", action="append", required=True)
    parser.add_argument("--template-sha256", required=True)
    parser.add_argument("--imatrix-sha256", required=True)
    parser.add_argument("--target-bpw", type=float, default=5.25)
    parser.add_argument("--hard-cap-bpw", type=float, default=5.30)
    parser.add_argument("--hash-sources", action="store_true")
    parser.add_argument("--tensor-type-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = build_plan(
        source_paths=tuple(args.source),
        template_path=args.sensitivity_template,
        imatrix_path=args.imatrix,
        source_revision=str(args.source_revision),
        template_revision=str(args.template_revision),
        source_sha256=tuple(str(value) for value in args.source_sha256),
        template_sha256=str(args.template_sha256),
        imatrix_sha256=str(args.imatrix_sha256),
        target_bpw=float(args.target_bpw),
        hard_cap_bpw=float(args.hard_cap_bpw),
        hash_sources=bool(args.hash_sources),
    )
    write_tensor_type_file(plan, args.tensor_type_file)
    plan["tensor_type_file"] = {
        "path": str(args.tensor_type_file.resolve()),
        "sha256": _sha256(args.tensor_type_file),
        "entries": len(plan["output_types"]),
        "matching": "anchored escaped ECMAScript regex",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "budget": plan["budget"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
