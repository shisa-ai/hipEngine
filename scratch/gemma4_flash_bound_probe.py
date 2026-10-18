"""Measure the worst per-element deviation over the battery's own case matrix.

The battery asserts ``assert_allclose(rtol=1e-4, atol=1e-4)`` for f32, which is
~100x looser than the agreement the route actually delivers. Picking a tighter
bound needs the worst case over the battery's *own* parametrization (both
packings x both dtypes x three mask modes x keys 1024/2055/4096, plus the
poisoned-workspace cases), not over a hand-picked subset.

This runs those cases and, instead of asserting, reports for each case the
ratio ``max(|cand - ref| / (atol + rtol * |ref|))`` -- the number that must be
<= 1 for a given (rtol, atol) pair to hold. It calls the battery's own helpers,
so nothing here can drift from what the battery runs.

Usage:
    PYTHONPATH=. python scratch/gemma4_flash_bound_probe.py [out.json]
"""

import json
import sys

import numpy as np

from tests import test_gpu_gemma4_flash_decode as battery
from tests.test_gpu_gemma4_flash_decode import _GLOBAL, _SLIDING, _as_float, _flash_ab

# Bounds under consideration for the f32 assertion.
_F32_CANDIDATES = (1e-6, 3e-6, 1e-5, 1e-4)
# bf16 storage ULP: both arms round to bf16, so the bound is one quantum of head.
_BF16_CANDIDATES = (6e-3, 2e-3, 1e-3)


def ratio(candidate, reference, rtol, atol):
    """max over elements of |a-b| / (atol + rtol*|b|); <= 1 means the bound holds."""
    candidate = np.asarray(candidate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    finite = np.isfinite(candidate) & np.isfinite(reference)
    if not finite.any():
        return 0.0
    diff = np.abs(candidate[finite] - reference[finite])
    denom = atol + rtol * np.abs(reference[finite])
    return float(np.max(diff / denom))


def main() -> None:
    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention import (
        build_gemma4_attention,
    )

    library = build_gemma4_attention(load=True)
    records = []
    for dtype in ("f32", "bf16"):
        candidates = _F32_CANDIDATES if dtype == "f32" else _BF16_CANDIDATES
        for mask_mode in ("keep", "holes", "window"):
            for keys in (1024, 2055, 4096):
                for name, geom in (("sliding", _SLIDING), ("global", _GLOBAL)):
                    single, flash, _ = _flash_ab(
                        library, dtype=dtype, mask_mode=mask_mode, keys=keys, **geom
                    )
                    ref = _as_float(single, dtype)
                    cand = _as_float(flash, dtype)
                    entry = {"case": f"{name} {dtype} {mask_mode} k{keys}", "dtype": dtype}
                    for bound in candidates:
                        entry[f"ratio_at_{bound:g}"] = ratio(cand, ref, bound, bound)
                    records.append(entry)
        # The poisoned-workspace case asserts the same bound; include it.
        for keys in (1024, 2055):
            for name, geom in (("sliding", _SLIDING), ("global", _GLOBAL)):
                single, flash, _ = _flash_ab(
                    library,
                    dtype=dtype,
                    mask_mode="keep",
                    keys=keys,
                    poison=True,
                    **geom,
                )
                ref = _as_float(single, dtype)
                cand = _as_float(flash, dtype)
                entry = {
                    "case": f"{name} {dtype} poison k{keys}",
                    "dtype": dtype,
                }
                for bound in candidates:
                    entry[f"ratio_at_{bound:g}"] = ratio(cand, ref, bound, bound)
                records.append(entry)

    for dtype, candidates in (("f32", _F32_CANDIDATES), ("bf16", _BF16_CANDIDATES)):
        subset = [r for r in records if r["dtype"] == dtype]
        print(f"== {dtype}: {len(subset)} cases")
        for bound in candidates:
            key = f"ratio_at_{bound:g}"
            worst = max(subset, key=lambda r: r[key])
            verdict = "holds" if worst[key] <= 1.0 else "VIOLATED"
            print(
                f"   rtol=atol={bound:g}: worst ratio {worst[key]:.3f} ({verdict})"
                f"  [{worst['case']}]"
            )
    if len(sys.argv) > 1:
        with open(sys.argv[1], "w") as handle:
            json.dump({"cases": records}, handle, indent=1)
        print(f"wrote {sys.argv[1]}")


if __name__ == "__main__":
    main()
