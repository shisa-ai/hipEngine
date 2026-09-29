"""Per-shape breakdown of the prefill projection families, with achieved GB/s.

The prefill census aggregates by family, which hides the shape distribution: the
dense family averages 961 us over 410 calls of very different shapes, and the MoE
families average over 58 layers. This reuses that census's HIP-event timing and
labels each call by shape instead, then divides moved bytes (weights plus
activations) by elapsed time to show which shapes are actually inefficient.

Usage: PYTHONPATH=. .venv/bin/python scripts/gemma4_prefill_shape_census.py
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import hipengine
from hipengine.core.hip import get_hip_runtime

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gemma4_prefill_census import Census  # noqa: E402

ARTIFACT = "/mnt/nvme1/models/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf"
BW_GBPS = 864.0


def quant_of(weight) -> str:
    spec = getattr(weight, "spec", None)
    return str(getattr(spec, "quant_key", "dense_bf16"))


def weight_bytes(weight) -> int:
    try:
        return int(weight.allocation("raw").buffer.nbytes)
    except Exception:  # noqa: BLE001
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tokens",
        type=int,
        default=1024,
        help="prefill token count to census (default: %(default)s)",
    )
    parser.add_argument(
        "--prefill-block",
        type=int,
        default=None,
        help=(
            "override the runner's max_block, the prefill rows per forward block "
            "(default: min(capacity, 512)). This is the knob that sets the "
            "grouped MoE's compact-row count: compact_rows = block * top_k, so "
            "rows per expert = block * top_k / num_experts."
        ),
    )
    args = parser.parse_args()
    runtime = get_hip_runtime()
    llm = hipengine.LLM(model=ARTIFACT)
    generator = llm._get_text_generator()
    generator.context_length = 8192
    runner = generator._ensure_runner()
    if args.prefill_block is not None:
        if args.prefill_block <= 0:
            parser.error("--prefill-block must be positive")
        if args.prefill_block > runner.capacity:
            parser.error(
                f"--prefill-block {args.prefill_block} exceeds capacity {runner.capacity}"
            )
        runner.max_block = int(args.prefill_block)
        # __post_init__ sizes each layer's Gemma4LayerScratch from max_block
        # (runtime/gemma4.py:499-507) but _scratches is a dataclass field, so a
        # re-run appends rather than replaces and the per-layer call keeps using
        # self._scratches[0]. Free and clear first, or the override is inert.
        for _scratch in runner._scratches:
            _scratch.free()
        runner._scratches.clear()
        runner.__post_init__()
        print(
            f"### prefill_block={runner.max_block} capacity={runner.capacity} "
            f"scratches={len(runner._scratches)} "
            f"scratch_tokens={getattr(runner._scratches[0], 'tokens', None)}",
            flush=True,
        )

    from hipengine.kernels.hip_gfx1100.gemma4 import gemma4_experts, gemma4_layer

    census = Census(runtime)
    # weight bytes per shape, filled by the label functions as a side effect.
    shape_bytes: dict[str, int] = {}
    shape_flops: dict[str, float] = {}

    def dense_label(x_ptr, weight, out_ptr, rows, in_features, out_features, **kw):
        key = f"dense:{quant_of(weight)} r={rows} k={in_features} n={out_features}"
        if isinstance(weight, int):
            return key
        wb = weight_bytes(weight)
        shape_bytes[key] = wb + rows * (in_features + out_features) * 2
        shape_flops[key] = 2.0 * rows * in_features * out_features
        return key

    def grouped_label(
        weight, x_ptr, expert_start_ptr, out_ptr, compact_rows, num_experts,
        in_features, out_features, **kw,
    ):
        key = (
            f"moe_grouped:{quant_of(weight)} rows={compact_rows} "
            f"k={in_features} n={out_features} e={num_experts}"
        )
        if not isinstance(weight, int):
            wb = weight_bytes(weight)
            shape_bytes[key] = wb + compact_rows * (in_features + out_features) * 2
            shape_flops[key] = 2.0 * compact_rows * in_features * out_features
        return key

    def grouped_dual_label(
        weight, x_ptr, expert_start_ptr, out_ptr, compact_rows, num_experts,
        in_features, out_features, fused_width, **kw,
    ):
        key = (
            f"moe_grouped_dual:{quant_of(weight)} rows={compact_rows} "
            f"k={in_features} n={fused_width} e={num_experts}"
        )
        if not isinstance(weight, int):
            wb = weight_bytes(weight)
            shape_bytes[key] = wb + compact_rows * (in_features + fused_width) * 2
            shape_flops[key] = 2.0 * compact_rows * in_features * fused_width
        return key

    def attention_label(
        query_ptr: int,
        key_ptr: int,
        value_ptr: int,
        keep_mask_ptr: int,
        out_ptr: int,
        **kw: Any,
    ) -> str:
        tokens = int(kw["tokens"])
        num_heads = int(kw["num_heads"])
        num_kv_heads = int(kw["num_kv_heads"])
        head_dim = int(kw["head_dim"])
        keys = tokens if kw.get("keys") is None else int(kw["keys"])
        key = (
            f"attention_prefill t={tokens} keys={keys} h={num_heads} "
            f"kv={num_kv_heads} d={head_dim}"
        )
        # Irreducible bytes: Q read, K+V read once, the uint8 keep-mask, and the
        # output write. The kernel launches one CTA per (query head, query row),
        # so it re-reads K/V once per head and once per row and moves strictly
        # more than this. The GB/s derived from these bytes is therefore an
        # upper bound on achieved efficiency -- if it is already low, the kernel
        # is leaving bandwidth on the table even under the most favourable
        # accounting.
        shape_bytes[key] = (
            tokens * num_heads * head_dim * 2
            + 2 * keys * num_kv_heads * head_dim * 2
            + tokens * keys
            + tokens * num_heads * head_dim * 2
        )
        # QK^T and P.V, each 2 * tokens * heads * keys * head_dim.
        shape_flops[key] = 4.0 * tokens * num_heads * keys * head_dim
        return key

    census.wrap(gemma4_layer, "gemma4_project", dense_label)
    census.wrap(gemma4_layer, "gemma4_attention_prefill_bf16", attention_label)
    census.wrap(gemma4_experts, "gemma4_project_experts_grouped", grouped_label)
    census.wrap(
        gemma4_experts, "gemma4_project_experts_grouped_dual", grouped_dual_label
    )

    runner.reset()
    runner.forward([9707] * 64)
    census.records.clear()
    runner.reset()
    runner.forward([9707] * args.tokens)
    summary = census.summarize()

    print(f"### tokens={args.tokens}")
    print(
        f"{'shape':62s} {'calls':>6s} {'tot ms':>8s} {'mean us':>8s} "
        f"{'MB/call':>8s} {'GB/s':>7s} {'TF/s':>6s}"
    )
    for key in sorted(summary, key=lambda k: -summary[k]["total_ms"]):
        row = summary[key]
        mb = shape_bytes.get(key, 0) / 1e6
        gbs = mb / 1e3 / (row["mean_us"] / 1e6) if row["mean_us"] else 0.0
        tfs = (
            shape_flops.get(key, 0.0) / (row["mean_us"] * 1e-6) / 1e12
            if row["mean_us"]
            else 0.0
        )
        print(
            f"{key:62s} {row['calls']:6d} {row['total_ms']:8.1f} {row['mean_us']:8.1f} "
            f"{mb:8.3f} {gbs:7.1f} {tfs:6.1f}"
        )
    print(f"(device peak {BW_GBPS:.0f} GB/s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
