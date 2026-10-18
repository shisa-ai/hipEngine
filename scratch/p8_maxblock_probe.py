"""Probe the fitted prefill block under one of two _sizes() variants.

usage: p8_maxblock_probe.py with_f32|without_f32 [context]

with_f32    = the tree as it stands (prescaled_f32 counted)
without_f32 = simulate the pre-P8 scratch table (prescaled_f32 removed)

Prints the bench load line, which reports the fitted max_block. Same box,
same occupancy, seconds apart -- the only variable is the table entry.
"""

import sys

import hipengine.kernels.hip_gfx1100.gemma4.gemma4_router as router

variant, context = sys.argv[1], sys.argv[2]
if variant == "without_f32":
    _orig = router.Gemma4RouterScratch._sizes

    def _old_sizes(self):
        sizes = _orig(self)
        sizes.pop("prescaled_f32", None)
        return sizes

    router.Gemma4RouterScratch._sizes = _old_sizes
else:
    assert variant == "with_f32", variant

from scripts.gemma4_campaign_bench import main as _bench_main

raise SystemExit(
    _bench_main(
        [
            "--prompt", "8",
            "--output", "1",
            "--samples", "1",
            "--warmup", "0",
            "--context", context,
            "--label", f"p8-maxblock-{variant}-{context}",
        ]
    )
)
