#!/usr/bin/env python3
"""Report a GGUF artifact's rank-2 matmul mix by small-row verifier route.

At the MTP target-verifier shapes (rows 2-7) the two weight-amortized owners
are the raw rowtile family (Q4_K/Q5_K/Q6_K/Q8_0) and, from 8 rows up, the
dense-IQ W4A16 prefill owner. Every other rank-2 quant has no owner below 8
rows and keeps the strict per-row GEMV, so its MACs are the "dead zone" the
Phase 1 attribution blames.

Diagnostic only: MAC shares are static weights, not measured time.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.loading.gguf import scan_gguf  # noqa: E402
from hipengine.runtime.gguf_linear import _ROWTILE_QUANT_BLOCKS  # noqa: E402

# Source types whose rank-2 resident layout is the weight-amortized rowtile
# family at rows 2-8. Q4_K is included because the decode-repack resident
# layout (HIPENGINE_GGUF_DECODE_REPACK=1, the production default) is the
# gguf_q4_k_t16_v1 rowtile family; the non-repack pack8 plan is not the
# verifier's owner.
ROWTILE_SOURCE_TYPES = frozenset({"Q4_K", "Q5_K", "Q6_K", "Q8_0"})
# Source types the dense-IQ prefill policy admits, with its rows floor. Kept in
# sync with hipengine.kernels.hip_gfx1100.GGUF_IQ_DENSE_PREFILL_POLICY
# (min_rows 8), so these keep the strict per-row GEMV at rows 2-7.
IQ_PREFILL_SOURCE_TYPES = frozenset(
    {"IQ4_XS", "IQ4_NL", "IQ3_S", "IQ3_XXS", "IQ2_S", "IQ2_XS", "Q3_K"}
)
IQ_PREFILL_MIN_ROWS = 8

_ROWTILE_QUANT_KEYS = {"gguf_q4_k", "gguf_q5_k", "gguf_q6_k", "gguf_q8_0"}
assert set(_ROWTILE_QUANT_BLOCKS) == _ROWTILE_QUANT_KEYS, (
    "the rowtile family's quants drifted; update ROWTILE_SOURCE_TYPES"
)
assert not (ROWTILE_SOURCE_TYPES & IQ_PREFILL_SOURCE_TYPES)


def classify(ggml_type_name: str) -> str:
    name = str(ggml_type_name)
    if name in ROWTILE_SOURCE_TYPES:
        return "rowtile_rows2_8"
    if name in IQ_PREFILL_SOURCE_TYPES:
        return "iq_dead_zone_rows2_7"
    return "other"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.models:
        info = scan_gguf(path)
        totals: dict[str, int] = {}
        by_type: dict[str, int] = {}
        grand = 0
        for tensor in info.tensors:
            if len(tensor.shape) != 2:
                continue
            macs = 2 * int(tensor.shape[0]) * int(tensor.shape[1])
            grand += macs
            klass = classify(tensor.ggml_type_name)
            totals[klass] = totals.get(klass, 0) + macs
            if klass == "iq_dead_zone_rows2_7":
                by_type[tensor.ggml_type_name] = by_type.get(tensor.ggml_type_name, 0) + macs
        print(f"=== {path.name} (file_type={info.metadata.get('general.file_type', '?')})")
        for klass in ("rowtile_rows2_8", "iq_dead_zone_rows2_7", "other"):
            share = totals.get(klass, 0) / grand * 100.0 if grand else 0.0
            print(f"    {klass:24s} {share:6.2f}%")
        for name, macs in sorted(by_type.items(), key=lambda kv: -kv[1]):
            print(f"        {name:10s} {macs / grand * 100.0:6.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
