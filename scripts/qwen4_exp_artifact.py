#!/usr/bin/env python3
"""Verify Qwen3.8-Flash-Next source and summarize its qwen4exp GGUF artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.loading.gguf import GGUFReader  # noqa: E402

_SHARD_RE = re.compile(r"model-\d+-of-\d+\.safetensors\Z")


def _sha256(path: Path, *, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _source_identity(files: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in sorted(files):
        item = files[name]
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(int(item["size"])).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(item.get("blob_id", "")).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def verify_hf_snapshot(root: str | Path, revision: str) -> dict[str, Any]:
    """Verify one local-directory HF download against its cached frozen tree."""

    resolved = Path(root).expanduser().resolve()
    tree_path = resolved / ".cache" / "huggingface" / "trees" / f"{revision}.json"
    tree_raw = tree_path.read_bytes()
    tree = json.loads(tree_raw)
    files: dict[str, Any] = tree["files"]

    complete: list[str] = []
    missing: list[str] = []
    mismatches: list[dict[str, Any]] = []
    for name in sorted(files):
        expected = int(files[name]["size"])
        path = resolved / name
        if not path.is_file():
            missing.append(name)
            continue
        actual = path.stat().st_size
        if actual != expected:
            mismatches.append({"path": name, "actual": actual, "expected": expected})
            continue
        complete.append(name)

    actual_files = {
        path.relative_to(resolved).as_posix()
        for path in resolved.rglob("*")
        if path.is_file() and ".cache" not in path.relative_to(resolved).parts
    }
    unexpected = sorted(actual_files - set(files))
    expected_shards = sorted(name for name in files if _SHARD_RE.fullmatch(name))
    complete_shards = sorted(name for name in complete if _SHARD_RE.fullmatch(name))

    result: dict[str, Any] = {
        "schema": 1,
        "kind": "qwen4exp_hf_snapshot",
        "root": str(resolved),
        "revision": revision,
        "tree_manifest": str(tree_path),
        "tree_manifest_sha256": hashlib.sha256(tree_raw).hexdigest(),
        "source_identity_sha256": _source_identity(files),
        "expected_files": len(files),
        "complete_files": len(complete),
        "expected_bytes": sum(int(item["size"]) for item in files.values()),
        "complete_bytes": sum(int(files[name]["size"]) for name in complete),
        "expected_shards": len(expected_shards),
        "complete_shards": len(complete_shards),
        "missing": missing,
        "size_mismatches": mismatches,
        "unexpected": unexpected,
    }

    index_name = "model.safetensors.index.json"
    index_path = resolved / index_name
    index_error: str | None = None
    if index_name in complete:
        try:
            index = json.loads(index_path.read_bytes())
            weight_map = index["weight_map"]
            referenced = sorted(set(str(value) for value in weight_map.values()))
            result["weight_index"] = {
                "declared_tensor_bytes": int(index.get("metadata", {}).get("total_size", 0)),
                "tensor_count": len(weight_map),
                "referenced_shards": referenced,
                "missing_referenced_shards": sorted(
                    name for name in referenced if not (resolved / name).is_file()
                ),
                "unreferenced_snapshot_shards": sorted(set(expected_shards) - set(referenced)),
            }
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            index_error = f"{type(exc).__name__}: {exc}"
            result["weight_index_error"] = index_error

    weight_index = result.get("weight_index")
    index_contract_failed = bool(
        weight_index
        and (
            weight_index["missing_referenced_shards"]
            or weight_index["unreferenced_snapshot_shards"]
        )
    )
    result["passed"] = not (
        missing or mismatches or unexpected or index_error or index_contract_failed
    )
    return result


def tensor_role(name: str) -> str:
    """Classify frozen llama.cpp qwen4exp GGUF tensor names by runtime owner."""

    if name == "per_layer_token_embd.weight":
        return "ple_table"
    if name in {"token_embd.weight", "output.weight"}:
        return "root"
    if name.startswith("head.hc_") or ".hc_" in name:
        return "gated_residual"
    if ".ple_" in name:
        return "ple_compute"
    if ".indexer." in name:
        return "qsa_indexer"
    if "_shexp." in name:
        return "shared_expert"
    if "_exps." in name:
        return "routed_expert"
    if ".ffn_gate_inp." in name:
        return "router"
    if ".ssm_" in name or ".attn_qkv." in name or ".attn_gate." in name:
        return "gdn"
    if any(
        marker in name
        for marker in (
            ".attn_q.",
            ".attn_q_norm.",
            ".attn_k.",
            ".attn_k_norm.",
            ".attn_v.",
            ".attn_output.",
        )
    ):
        return "full_attention"
    return "unknown"


def _sorted_counter(counter: Counter[str]) -> dict[str, int]:
    return {key: int(counter[key]) for key in sorted(counter)}


def summarize_qwen4_exp_gguf(reader: GGUFReader) -> dict[str, Any]:
    """Summarize tensor bytes by qtype and qwen4exp runtime owner."""

    info = reader.info
    bytes_by_role: Counter[str] = Counter()
    count_by_role: Counter[str] = Counter()
    bytes_by_type: Counter[str] = Counter()
    count_by_type: Counter[str] = Counter()
    ple_tables: list[dict[str, Any]] = []
    for tensor in info.tensors:
        role = tensor_role(tensor.name)
        bytes_by_role[role] += int(tensor.nbytes)
        count_by_role[role] += 1
        bytes_by_type[tensor.ggml_type_name] += int(tensor.nbytes)
        count_by_type[tensor.ggml_type_name] += 1
        if role == "ple_table":
            ple_tables.append(
                {
                    "name": tensor.name,
                    "type": tensor.ggml_type_name,
                    "nbytes": int(tensor.nbytes),
                }
            )

    architecture_ok = info.architecture == "qwen4exp"
    ple_ok = len(ple_tables) == 1
    return {
        "schema": 1,
        "kind": "qwen4exp_gguf",
        "path": str(info.path),
        "architecture": info.architecture,
        "file_type": info.file_type,
        "file_type_name": info.file_type_name,
        "quantization_version": info.metadata.get("general.quantization_version"),
        "tensor_count": info.tensor_count,
        "total_tensor_nbytes": info.total_tensor_nbytes,
        "tensor_count_by_role": _sorted_counter(count_by_role),
        "tensor_bytes_by_role": _sorted_counter(bytes_by_role),
        "tensor_count_by_type": _sorted_counter(count_by_type),
        "tensor_bytes_by_type": _sorted_counter(bytes_by_type),
        "ple_table": ple_tables[0] if ple_ok else None,
        "unknown_tensors": sorted(
            tensor.name for tensor in info.tensors if tensor_role(tensor.name) == "unknown"
        ),
        "errors": [
            *(
                []
                if architecture_ok
                else [f"architecture is {info.architecture!r}, not 'qwen4exp'"]
            ),
            *([] if ple_ok else [f"expected exactly one PLE table, found {len(ple_tables)}"]),
        ],
        "passed": architecture_ok and ple_ok,
    }


def _emit(result: dict[str, Any], json_out: Path | None) -> None:
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(text + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    hf_parser = subparsers.add_parser("verify-hf", help="verify a frozen HF local directory")
    hf_parser.add_argument("root", type=Path)
    hf_parser.add_argument("--revision", required=True)
    hf_parser.add_argument("--json-out", type=Path)

    gguf_parser = subparsers.add_parser("inspect-gguf", help="summarize a qwen4exp GGUF")
    gguf_parser.add_argument("path", type=Path)
    gguf_parser.add_argument("--sha256", action="store_true")
    gguf_parser.add_argument("--json-out", type=Path)

    args = parser.parse_args(argv)
    if args.command == "verify-hf":
        result = verify_hf_snapshot(args.root, args.revision)
    else:
        result = summarize_qwen4_exp_gguf(GGUFReader(args.path))
        result["file_bytes"] = args.path.stat().st_size
        if args.sha256:
            result["sha256"] = _sha256(args.path)
    _emit(result, args.json_out)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
