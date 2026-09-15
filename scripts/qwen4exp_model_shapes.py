#!/usr/bin/env python3
"""Dump tensor geometry and routing config for a Qwen4Exp GGUF model.

Writes the shape map that ``scripts/qwen4exp_operation_cost.py`` uses to give
operations outside the GGUF quant launch path (MoE, QSA, GR, GDN, indexer) a
real shape. Geometry is read from the GGUF tensor headers, so the map is a
property of the model file rather than of any one run.

Layer quantizations differ within a single model: the expert down-projection is
mostly Q5_1 but Q8_0 in a few layers, and one layer's gate is Q5_K. Each tensor
therefore records the byte-weighted ``quant`` of the majority along with the
full ``quant_mix``, so a per-tensor FLOP count stays correct while a traffic
estimate does not silently assume one quantization for all layers.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from hipengine.loading.gguf import discover_gguf_files, scan_gguf

BYTES_PER_ELEMENT = {
    "Q8_0": 34 / 32, "Q4_K": 144 / 256, "Q5_K": 176 / 256,
    "Q6_K": 210 / 256, "Q5_1": 24 / 32, "Q4_0": 18 / 32,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    paths = discover_gguf_files(args.model_root)
    if not paths:
        raise SystemExit(f"no GGUF shards under {args.model_root}")

    # suffix -> geometry -> quant -> shard count
    seen: dict[str, dict[tuple[int, int, int], collections.Counter[str]]] = {}
    for path in paths:
        for tensor in scan_gguf(path).tensors:
            suffix = tensor.name.split(".", 2)[-1] if tensor.name.count(".") >= 2 else tensor.name
            shape = list(tensor.ggml_shape) + [1] * (4 - len(tensor.ggml_shape))
            geometry = (int(shape[0]), int(shape[1]), int(shape[2]) * int(shape[3]))
            seen.setdefault(suffix, {}).setdefault(geometry, collections.Counter())[
                tensor.ggml_type_name
            ] += 1

    tensors: dict[str, dict[str, object]] = {}
    for suffix, geometries in sorted(seen.items()):
        if len(geometries) > 1:
            raise SystemExit(
                f"{suffix} has {len(geometries)} distinct geometries; the shape map "
                "cannot represent it without a per-layer lookup"
            )
        (k, m, experts), quant_counts = next(iter(geometries.items()))
        total = sum(quant_counts.values())
        # Majority by shard count, tie-broken by name for determinism.
        majority = sorted(quant_counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        tensors[suffix] = {
            "quant": majority,
            "quant_mix": {q: c for q, c in sorted(quant_counts.items())},
            "quant_mix_bytes_weighted": round(
                sum(BYTES_PER_ELEMENT.get(q, 1.0) * c for q, c in quant_counts.items())
                / total, 4
            ),
            "in_features": k,
            "out_features": m,
            "experts": experts,
        }

    metadata = scan_gguf(paths[0]).metadata
    config = {
        key: value
        for key, value in metadata.items()
        if key.startswith("qwen4exp.") and not isinstance(value, list)
    }
    # List-valued entries still matter (PLE layer multipliers), so keep them as
    # lists rather than dropping them.
    config.update({
        key: value
        for key, value in metadata.items()
        if key.startswith("qwen4exp.") and isinstance(value, list)
    })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "schema": 1,
        "kind": "qwen4exp_model_shape_map",
        "model_root": args.model_root,
        "shard_count": len(paths),
        "config": dict(sorted(config.items())),
        "tensors": tensors,
    }, indent=1) + "\n")

    print(f"{len(tensors)} tensors from {len(paths)} shard(s) -> {args.output}")
    mixed = {k: v["quant_mix"] for k, v in tensors.items() if len(v["quant_mix"]) > 1}
    print(f"mixed-quant tensors: {mixed or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
