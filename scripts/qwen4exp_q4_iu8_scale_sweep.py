#!/usr/bin/env python3
"""R4a calibration probe: iu8 risk-criterion escape onset vs activation scale.

For each activation scale 2^-k, runs the iu8 risk+repair chain at the
production multiplier 4.0 against the pair2 parent on a synthetic fixture
and counts mismatched BF16 outputs. With the row at-risk guard (amax floor
2^-80 plus nonfinite flag) the sweep must be clean at every scale; without
it, escapes begin between 2^-100 and 2^-110 and collapse to ~100% by 2^-115.
"""

import argparse
import ctypes
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import free
from hipengine.kernels.hip_gfx1100.quant import gguf_q4_k_selected_prefill as q4
from tests.test_gguf_q4_k_selected_wmma_prefill import _build_compact_fixture
from tests.test_qwen4exp_q4_iu8_exact import (
    _run_chain,
    _run_parent,
)


def sweep(multiplier: float, scales, seeds) -> list:
    runtime = get_hip_runtime()
    library = q4.build_gguf_q4_k_selected_prefill(load=True)
    fixture = _build_compact_fixture(
        counts=[9, 12, 3], in_features=256, out_features_a=128,
        out_features_b=128, dtype="bf16", seed=101,
    )
    rows = []
    for k in scales:
        for seed in seeds:
            rng = np.random.default_rng(1000 + seed)
            x = (rng.normal(0.0, 1.0, size=(fixture.compact_rows,
                                            fixture.in_features))
                 * np.float32(2.0) ** -k).astype(np.float32)
            bits = x.view(np.uint32)
            rounded = bits + 0x7FFF + ((bits >> 16) & 1)
            x_bits = (rounded >> 16).astype(np.uint16)
            fx = dataclasses.replace(fixture, x_host=x_bits)
            allocations: list = []
            try:
                parent = _run_parent(fx, runtime, library,
                                     allocations=allocations)
                chain, risks = _run_chain(fx, runtime, library,
                                           risk_multiplier=multiplier,
                                           allocations=allocations)
                bad = int(np.count_nonzero(chain != parent))
                total = chain.size
                # max ulp distance among mismatches
                max_ulp = 0
                if bad:
                    diff_idx = np.nonzero(chain != parent)
                    d = np.abs(
                        chain[diff_idx].astype(np.int64)
                        - parent[diff_idx].astype(np.int64)
                    )
                    max_ulp = int(d.max())
                print(f"k={k:4d} seed={seed} mismatch={bad:6d}/{total} "
                      f"risks={risks:6d} max_ulp={max_ulp}", flush=True)
                rows.append({
                    "scale_exponent": -k,
                    "seed": seed,
                    "mismatched_outputs": bad,
                    "total_outputs": total,
                    "queued_risks": risks,
                    "max_ulp_distance": max_ulp,
                })
            finally:
                for ptr in reversed(allocations):
                    free(ptr, runtime=runtime)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--multiplier", type=float, default=4.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    scales = [0, 30, 60, 80, 100, 110, 115, 118, 120, 122, 124, 126, 128, 130]
    rows = sweep(args.multiplier, scales, seeds=[1, 2, 3])
    payload = {
        "probe": "qwen4exp_q4_iu8_scale_sweep",
        "multiplier": args.multiplier,
        "fixture": {"counts": [9, 12, 3], "in_features": 256,
                    "out_features": [128, 128], "dtype": "bf16"},
        "rows": rows,
    }
    args.output.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"wrote {args.output}")
