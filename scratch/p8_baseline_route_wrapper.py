"""Run any gemma4 gate/bench CLI with the pre-change router projection tier.

Raises ``_ROUTER_SGEMM_MIN_TOKENS`` above any real block width so the chain
selects the token-tile route exactly as the tree did before the P8 sgemm
tier landed (tokens >= 32 -> tile, below -> untiled).  Used to capture the
baseline arm of the P8 teacher-forced gate and the tile-route bench arm.
"""

import sys

import hipengine.kernels.hip_gfx1100.gemma4.gemma4_router as _router

_router._ROUTER_SGEMM_MIN_TOKENS = 10**9

from scripts.gemma4_campaign_bench import main as _bench_main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(_bench_main(sys.argv[1:]))
