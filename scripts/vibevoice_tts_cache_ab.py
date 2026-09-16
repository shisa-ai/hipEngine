"""Interleaved cached/uncached diffusion A/B on the frozen TTS session request.

Promoted from the measurement's /tmp/ab_cache.py. The uncached arm disables
the timestep and condition caches only; both arms keep the SiLU hoist.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="microsoft/VibeVoice-1.5B")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--fixtures", type=Path,
                        default=Path(__file__).resolve().parents[1] / "tests/fixtures/vibevoice_tts")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from hipengine.runtime.vibevoice_tts_diffusion import VibevoiceTTSDiffusionHeadGPU
    from scripts.vibevoice_tts_session_bench import bench

    original = VibevoiceTTSDiffusionHeadGPU.forward_into

    def uncached(self, t_value, cond):
        self._t_mlp_cache.clear()
        self._cond_proj_key = None
        return original(self, t_value, cond)

    results = {"cached": [], "uncached": []}
    try:
        for arm in ("cached", "uncached", "cached", "uncached"):
            VibevoiceTTSDiffusionHeadGPU.forward_into = original if arm == "cached" else uncached
            result = bench(args.model, args.fixtures, args.repeats)
            results[arm].append({
                "rtf": result["pooled_rtf"],
                "warm": result["warm_synthesis_seconds"],
                "diffusion": result["stages"]["diffusion_seconds"],
                "chain_exact": result.get("chain_exact"),
            })
            print(arm, results[arm][-1], flush=True)
    finally:
        VibevoiceTTSDiffusionHeadGPU.forward_into = original
    print("RESULT " + json.dumps(results))


if __name__ == "__main__":
    main()
