#!/usr/bin/env python3
"""Create a packed ParoQuant safetensors export.

ParoQuant HF-bridge exports currently contain both quantized tensors
(``.qweight/.qzeros/.scales/...``) and duplicate fp16 ``.weight`` fallback
tensors for the same modules.  This script removes every ``.weight`` whose
module also has a ``.qweight``, producing the canonical packed PARO format.

This is the only output mode.  hipEngine itself materializes the packed shared
expert directly via ``hipengine.loading.qwen35_paro``; there is no separate
"keep fp16 fallback" path.

The script writes a new export directory; it never edits the input in-place.
"""
from __future__ import annotations

import argparse
import json
import shutil
import struct
from pathlib import Path
from typing import Any

DEFAULT_DENOMINATOR = 35_000_000_000


def read_safetensors_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(header_len))


def tensor_sizes_from_header(path: Path) -> dict[str, int]:
    header = read_safetensors_header(path)
    out: dict[str, int] = {}
    for key, meta in header.items():
        if key == "__metadata__":
            continue
        start, end = meta["data_offsets"]
        out[key] = int(end) - int(start)
    return out


def quant_prefixes(keys: set[str]) -> set[str]:
    return {k[: -len(".qweight")] for k in keys if k.endswith(".qweight")}


def is_duplicate_fallback(key: str, prefixes: set[str]) -> bool:
    return key.endswith(".weight") and key[: -len(".weight")] in prefixes


def plan_removal(keys: set[str]) -> set[str]:
    """Return the set of duplicate fp16 ``.weight`` fallback tensors to drop."""
    prefixes = quant_prefixes(keys)
    return {key for key in keys if is_duplicate_fallback(key, prefixes)}


def copy_sidecars(src: Path, dst: Path) -> None:
    skip_names = {"model.safetensors", "model.orig.safetensors"}
    for child in src.iterdir():
        if child.name in skip_names:
            continue
        target = dst / child.name
        if child.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(child, target)
        elif child.is_file():
            shutil.copy2(child, target)


def bpw(bytes_: int, denominator: int) -> float:
    return bytes_ * 8.0 / denominator


def main() -> int:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=__doc__)
    ap.add_argument("--input-dir", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--denominator-params", type=int, default=DEFAULT_DENOMINATOR)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    src_model = args.input_dir / "model.safetensors"
    dst_model = args.output_dir / "model.safetensors"
    if not src_model.exists():
        raise FileNotFoundError(src_model)
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite and not args.dry_run:
        raise FileExistsError(f"output directory is non-empty: {args.output_dir} (use --overwrite)")

    sizes = tensor_sizes_from_header(src_model)
    remove = plan_removal(set(sizes))
    kept = set(sizes) - remove
    src_bytes = src_model.stat().st_size
    removed_tensor_bytes = sum(sizes[k] for k in remove)
    kept_tensor_bytes = sum(sizes[k] for k in kept)
    payload = {
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "mode": "packed",
        "source_file_bytes": src_bytes,
        "source_bpw": bpw(src_bytes, args.denominator_params),
        "source_tensor_bytes": sum(sizes.values()),
        "removed_tensors": len(remove),
        "removed_tensor_bytes": removed_tensor_bytes,
        "removed_bpw": bpw(removed_tensor_bytes, args.denominator_params),
        "estimated_output_tensor_bytes": kept_tensor_bytes,
        "estimated_output_bpw_from_tensor_bytes": bpw(kept_tensor_bytes, args.denominator_params),
        "removed_preview": sorted(remove)[:50],
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if args.dry_run:
        return 0

    if args.output_dir.exists() and args.overwrite:
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    copy_sidecars(args.input_dir, args.output_dir)

    from safetensors.torch import load_file, save_file

    print(f"Loading tensors from {src_model} ...", flush=True)
    state = load_file(str(src_model), device="cpu")
    stripped = {k: v for k, v in state.items() if k not in remove}
    metadata = {
        "format": "pt",
        "paro_stripped_mode": "packed",
        "paro_removed_tensors": str(len(remove)),
        "paro_removed_tensor_bytes": str(removed_tensor_bytes),
    }
    tmp = dst_model.with_suffix(".safetensors.tmp")
    print(f"Writing {dst_model} with {len(stripped)} tensors ...", flush=True)
    save_file(stripped, str(tmp), metadata=metadata)
    tmp.replace(dst_model)
    payload["actual_output_file_bytes"] = dst_model.stat().st_size
    payload["actual_output_bpw"] = bpw(dst_model.stat().st_size, args.denominator_params)
    (args.output_dir / "strip_paro_safetensors_report.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(dst_model), "actual_bpw": payload["actual_output_bpw"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
