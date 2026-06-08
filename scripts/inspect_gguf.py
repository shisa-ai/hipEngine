#!/usr/bin/env python3
"""Inspect GGUF metadata/tensor coverage without loading full weights."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.loading.gguf import GGUFReader, discover_gguf_files
from hipengine.quant.gguf import dequantization_supported, dequantize_gguf_data


def _counter_bytes(counter: Counter[str]) -> dict[str, int]:
    return {key: int(counter[key]) for key in sorted(counter)}


def summarize(reader: GGUFReader, *, check_dequant: bool, smoke_rows: int) -> dict[str, Any]:
    info = reader.info
    tensor_count_by_type: Counter[str] = Counter()
    tensor_bytes_by_type: Counter[str] = Counter()
    unsupported_dequant: set[str] = set()
    for tensor in info.tensors:
        tensor_count_by_type[tensor.ggml_type_name] += 1
        tensor_bytes_by_type[tensor.ggml_type_name] += int(tensor.nbytes)
        if not dequantization_supported(tensor.ggml_type):
            unsupported_dequant.add(tensor.ggml_type_name)

    summary: dict[str, Any] = {
        "path": str(info.path),
        "version": info.version,
        "alignment": info.alignment,
        "tensor_data_offset": info.tensor_data_offset,
        "architecture": info.architecture,
        "file_type": info.file_type,
        "file_type_name": info.file_type_name,
        "quantization_version": info.metadata.get("general.quantization_version"),
        "tokenizer_model": info.metadata.get("tokenizer.ggml.model"),
        "tensor_count": info.tensor_count,
        "total_tensor_nbytes": info.total_tensor_nbytes,
        "tensor_count_by_type": _counter_bytes(tensor_count_by_type),
        "tensor_bytes_by_type": _counter_bytes(tensor_bytes_by_type),
        "unsupported_dequant_types": sorted(unsupported_dequant),
        "first_tensors": [
            {
                "name": tensor.name,
                "type": tensor.ggml_type_name,
                "shape": list(tensor.shape),
                "byte_shape": list(tensor.byte_shape),
                "nbytes": tensor.nbytes,
            }
            for tensor in info.tensors[:12]
        ],
    }
    if check_dequant:
        summary["dequant_smoke"] = _dequant_smoke(reader, smoke_rows=smoke_rows)
    return summary


def _dequant_smoke(reader: GGUFReader, *, smoke_rows: int) -> list[dict[str, Any]]:
    seen: set[str] = set()
    smokes: list[dict[str, Any]] = []
    for tensor in reader.info.tensors:
        if tensor.ggml_type_name in seen or not dequantization_supported(tensor.ggml_type):
            continue
        seen.add(tensor.ggml_type_name)
        data = reader.tensor_data(tensor.name)
        sample = _sample_tensor_rows(data, rows=smoke_rows)
        dequant = dequantize_gguf_data(sample, tensor.ggml_type)
        smokes.append(
            {
                "type": tensor.ggml_type_name,
                "tensor": tensor.name,
                "sample_shape": list(sample.shape),
                "dequant_shape": list(dequant.shape),
                "finite": bool(np.isfinite(dequant).all()),
                "min": float(np.min(dequant)) if dequant.size else 0.0,
                "max": float(np.max(dequant)) if dequant.size else 0.0,
            }
        )
    return smokes


def _sample_tensor_rows(data: np.ndarray, *, rows: int) -> np.ndarray:
    rows = max(1, int(rows))
    if data.ndim == 0:
        return np.asarray(data)
    if data.ndim == 1:
        return np.asarray(data[: min(data.shape[0], rows)])
    flat = data.reshape((-1, data.shape[-1]))
    return np.asarray(flat[: min(flat.shape[0], rows)])


def _print_text(summary: dict[str, Any]) -> None:
    print(f"{summary['path']}")
    print(
        f"  version={summary['version']} alignment={summary['alignment']} "
        f"arch={summary['architecture']} "
        f"file_type={summary['file_type_name'] or summary['file_type']} "
        f"tensors={summary['tensor_count']} total_bytes={summary['total_tensor_nbytes']}"
    )
    print(
        f"  tokenizer={summary['tokenizer_model']} "
        f"quant_version={summary['quantization_version']}"
    )
    print(f"  tensor_count_by_type={summary['tensor_count_by_type']}")
    print(f"  tensor_bytes_by_type={summary['tensor_bytes_by_type']}")
    if summary["unsupported_dequant_types"]:
        print(f"  unsupported_dequant_types={summary['unsupported_dequant_types']}")
    else:
        print("  unsupported_dequant_types=[]")
    for tensor in summary["first_tensors"]:
        print(
            f"    {tensor['name']}: {tensor['type']} "
            f"shape={tensor['shape']} byte_shape={tensor['byte_shape']} nbytes={tensor['nbytes']}"
        )
    if "dequant_smoke" in summary:
        for smoke in summary["dequant_smoke"]:
            print(
                f"  dequant {smoke['type']} {smoke['tensor']} "
                f"sample={smoke['sample_shape']} -> {smoke['dequant_shape']} "
                f"finite={smoke['finite']} range=[{smoke['min']:.6g}, {smoke['max']:.6g}]"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths", nargs="+", help="GGUF file(s) or directories containing .gguf files"
    )
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text")
    parser.add_argument(
        "--check-dequant",
        action="store_true",
        help="dequantize a tiny sample for each supported tensor type",
    )
    parser.add_argument(
        "--smoke-rows", type=int, default=1, help="rows per tensor type for --check-dequant"
    )
    parser.add_argument(
        "--fail-on-unsupported-dequant",
        action="store_true",
        help="exit non-zero if any tensor type lacks a CPU fallback dequantizer",
    )
    args = parser.parse_args(argv)

    summaries: list[dict[str, Any]] = []
    for raw_path in args.paths:
        for path in discover_gguf_files(raw_path):
            summaries.append(
                summarize(
                    GGUFReader(path), check_dequant=args.check_dequant, smoke_rows=args.smoke_rows
                )
            )

    if args.json:
        print(
            json.dumps(
                summaries if len(summaries) != 1 else summaries[0], indent=2, sort_keys=True
            )
        )
    else:
        for idx, summary in enumerate(summaries):
            if idx:
                print()
            _print_text(summary)

    if args.fail_on_unsupported_dequant:
        unsupported = {
            item for summary in summaries for item in summary["unsupported_dequant_types"]
        }
        if unsupported:
            print(f"unsupported GGUF dequant types: {sorted(unsupported)}", file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
