#!/usr/bin/env python3
"""Per-shape tile sweep for the GGUF Q8_0 WMMA prefill owner.

`gguf_q8_0_prefill.py::_default_tiles` picks (tile_m, tile_n) from the shape by a
heuristic tuned in 2026 on other models' shapes. Gemma 4's dense Q8_0
projections are 159 ms of a 554 ms per-prefill kernel budget, so the question is
whether that heuristic is right for *these* shapes.

The `HIPENGINE_GGUF_Q8_0_WMMA_TILE_M/N` overrides apply to every launch, so a
whole-prefill sweep conflates the six shapes with each other. This probe times
one shape at a time at rows=512 -- Gemma's block size -- over synthetic Q8_0
weights of the real geometry, and reports the best tile per shape against the
heuristic's current choice.

Usage:
    env -u HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES=0 PYTHONPATH=. \
        .venv/bin/python scripts/gemma4_dense_q8_tile_sweep.py \
        --json-out /tmp/gemma4_dense_q8_tile_sweep.json
"""

from __future__ import annotations

import argparse
import ctypes
import json
import statistics
import sys

# Gemma 4's dense projections, then the shapes `_default_tiles`' own comment
# names as the basis for the current heuristic, so a change here can be checked
# against the tuning it would replace rather than only against Gemma.
GEMMA4_DENSE_SHAPES = (
    (2048, 8192),  # Qwen35 linear-attention qkv / full-attn q+gate
    (2048, 4096),  # Qwen35 linear-attention gate
    (4096, 2048),  # Qwen35 ssm / shared down
    (2816, 512),   # a small-out shape (the `out <= 512` rule)
    (2816, 256),
    (2816, 4096),  # attn_q, sliding
    (2816, 8192),  # attn_q, global
    (2816, 2048),  # attn_k / attn_v, sliding
    (2816, 2112),  # dense ffn gate / up
    (4096, 2816),  # attn_output, sliding
    (8192, 2816),  # attn_output, global
    (2112, 2816),  # dense ffn down
    (2816, 1024),  # attn_k / attn_v, global
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=512)
    ap.add_argument("--rows-list", default="")
    ap.add_argument("--repeats", type=int, default=9)
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import (
        copy_host_to_device,
        free,
        host_buffer_ptr,
        malloc,
    )
    from hipengine.kernels.hip_gfx1100.quant.gguf_q8_0_prefill import (
        _ALLOWED_TILES,
        _default_tiles,
        build_gguf_q8_0_prefill,
        gguf_q8_0_wmma_prefill_bf16_bf16_out,
    )

    runtime = get_hip_runtime()
    library = build_gguf_q8_0_prefill(load=True)
    sync = runtime.device_synchronize
    row_counts = (
        [int(v) for v in args.rows_list.split(",") if v.strip()]
        if args.rows_list
        else [int(args.rows)]
    )
    report: dict[str, dict] = {}

    for rows, (in_features, out_features) in (
        (r, shape) for r in row_counts for shape in GEMMA4_DENSE_SHAPES
    ):
        weight_bytes = out_features * (in_features // 32) * 34
        weight = malloc(weight_bytes, runtime=runtime)
        # A scale of 1.0 with a fixed int8 pattern: the kernel reads every block
        # header and every quant byte, so the traffic is the real one; only the
        # values are synthetic, and no tile's cost depends on them.
        host_weight = (ctypes.c_uint8 * weight_bytes)()
        for block in range(weight_bytes // 34):
            base = block * 34
            host_weight[base] = 0x00
            host_weight[base + 1] = 0x3C  # fp16 1.0
        copy_host_to_device(
            weight, host_buffer_ptr(host_weight), weight_bytes, runtime=runtime
        )
        x = malloc(rows * in_features * 2, runtime=runtime)
        out = malloc(rows * out_features * 2, runtime=runtime)
        try:
            samples: dict[tuple[int, int], list[float]] = {}
            for _ in range(args.passes):
                for tile_m, tile_n in sorted(_ALLOWED_TILES):
                    for _ in range(args.repeats):
                        sync()
                        import time

                        begin = time.perf_counter()
                        gguf_q8_0_wmma_prefill_bf16_bf16_out(
                            x.ptr,
                            weight.ptr,
                            out.ptr,
                            rows,
                            in_features,
                            out_features,
                            tile_m=tile_m,
                            tile_n=tile_n,
                            stream=0,
                            library=library,
                            runtime=runtime,
                        )
                        sync()
                        samples.setdefault((tile_m, tile_n), []).append(
                            (time.perf_counter() - begin) * 1e3
                        )
            medians = {tile: statistics.median(v) for tile, v in samples.items()}
            best = min(medians, key=lambda t: medians[t])
            current = _default_tiles(rows, in_features, out_features)
            report[f"r{rows}:{in_features}->{out_features}"] = {
                "in_features": in_features,
                "out_features": out_features,
                "rows": rows,
                "current_tile": list(current),
                "current_ms": round(medians[current], 4),
                "best_tile": list(best),
                "best_ms": round(medians[best], 4),
                "speedup": round(medians[current] / medians[best], 4),
                "tiles": {
                    f"{m}x{n}": round(medians[(m, n)], 4)
                    for (m, n) in sorted(medians)
                },
            }
            entry = report[f"r{rows}:{in_features}->{out_features}"]
            print(
                f"r{rows:<5}{in_features:>5}->{out_features:<5} current {current} "
                f"{entry['current_ms']:8.3f} ms   best {best} "
                f"{entry['best_ms']:8.3f} ms   {entry['speedup']:.3f}x"
            )
        finally:
            free(out, runtime=runtime)
            free(x, runtime=runtime)
            free(weight, runtime=runtime)

    if args.json_out:
        with open(args.json_out, "w") as handle:
            json.dump({"rows": rows, "shapes": report}, handle, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
