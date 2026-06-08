#!/usr/bin/env python3
"""Build explicit qwen35moe GGUF expert pack8 sidecar cache files.

This script is intentionally opt-in.  It writes generated ``.npz`` sidecars to a
cache directory (default: ``~/.cache/hipengine/gguf_sidecars`` or
``HIPENGINE_GGUF_SIDECAR_CACHE``) and never modifies the source GGUF file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hipengine.loading.gguf import GGUFReader
from hipengine.loading.qwen35_gguf import build_qwen35_gguf_tensor_map
from hipengine.loading.qwen35_gguf_expert_sidecar import (
    DEFAULT_EXPERT_SIDECAR_SLOTS,
    build_or_load_qwen35moe_expert_sidecar,
    default_expert_sidecar_cache_dir,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Path to qwen35moe GGUF file")
    parser.add_argument(
        "--layers",
        default="0",
        help="Comma-separated layer ids, ranges like 0-3, or 'all' (default: 0)",
    )
    parser.add_argument(
        "--slots",
        default=",".join(DEFAULT_EXPERT_SIDECAR_SLOTS),
        help="Comma-separated expert slots to build (default: ffn_gate_exps,ffn_up_exps,ffn_down_exps)",
    )
    parser.add_argument("--cache-dir", type=Path, default=None, help="Sidecar cache directory")
    parser.add_argument("--overwrite", action="store_true", help="Rebuild even if cached sidecars exist")
    parser.add_argument("--require-cached", action="store_true", help="Fail instead of building missing sidecars")
    parser.add_argument("--json", type=Path, default=None, help="Write summary JSON")
    args = parser.parse_args()

    reader = GGUFReader(args.model)
    model_map = build_qwen35_gguf_tensor_map(reader.info)
    layers = _parse_layers(args.layers, model_map.config.block_count)
    slots = tuple(item.strip() for item in args.slots.split(",") if item.strip())
    cache_dir = args.cache_dir or default_expert_sidecar_cache_dir()

    summaries = []
    for layer_id in layers:
        sidecar = build_or_load_qwen35moe_expert_sidecar(
            reader,
            layer_id=layer_id,
            slots=slots,
            cache_dir=cache_dir,
            overwrite=args.overwrite,
            require_cached=args.require_cached,
        )
        entry = sidecar.metadata()
        summaries.append(entry)
        print(
            f"layer {layer_id}: {len(sidecar.slots)} sidecars, "
            f"{sidecar.nbytes / (1024 ** 2):.1f} MiB packed arrays"
        )
        for slot in sidecar.slots:
            tensor = sidecar.tensor(slot)
            print(
                f"  {slot}: {tensor.quant_key} shape={tensor.shape} "
                f"out_packed={tensor.out_packed} cache={sidecar.cache_paths[slot]}"
            )

    summary = {
        "model": str(reader.info.path),
        "cache_dir": str(cache_dir),
        "layers": layers,
        "slots": slots,
        "sidecars": summaries,
    }
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    return 0


def _parse_layers(text: str, block_count: int) -> list[int]:
    text = text.strip()
    if text == "all":
        return list(range(block_count))
    layers: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            if end < start:
                raise ValueError(f"invalid descending layer range {part!r}")
            layers.extend(range(start, end + 1))
        else:
            layers.append(int(part))
    unique = sorted(dict.fromkeys(layers))
    for layer_id in unique:
        if layer_id < 0 or layer_id >= block_count:
            raise ValueError(f"layer {layer_id} outside [0, {block_count})")
    return unique


if __name__ == "__main__":
    raise SystemExit(main())
