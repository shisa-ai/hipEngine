"""Report the flash route's observed deviation from the single kernel.

A diagnostic, not a gate. The battery (`tests/test_gpu_gemma4_flash_decode.py`)
*asserts* a tolerance; this prints the deviation itself so two trees can be
compared at the same shapes with the same inputs. It answers "did the kernel's
arithmetic move?", which the model-level teacher-forced gate cannot: that
instrument also moves when the model's sensitivity to the same arithmetic
changes.

Both arms run through the raw ABI on one input, alternating, so admission rules
do not enter. The shapes are the production ones at the short-context decode
band: sliding (head_dim 256, ratio 2) and global (head_dim 512, ratio 8).

Usage:
    PYTHONPATH=. python scratch/gemma4_flash_deviation.py [out.json]
"""

import json
import sys

import numpy as np

from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention import (
    build_gemma4_attention,
)
from tests.test_gpu_gemma4_flash_decode import _GLOBAL, _SLIDING, _as_float, _flash_ab


def deviation(library, *, label, dtype, mask_mode, keys, geom):
    out_single, out_flash, slices = _flash_ab(
        library, dtype=dtype, mask_mode=mask_mode, keys=keys, **geom
    )
    ref = _as_float(out_single, dtype).astype(np.float64)
    cand = _as_float(out_flash, dtype).astype(np.float64)
    diff = np.abs(cand - ref)
    scale = float(np.max(np.abs(ref))) or 1.0
    rms_ref = float(np.sqrt((ref**2).mean())) or 1.0
    row = {
        "case": label,
        "slices": int(slices),
        "max_abs": float(diff.max()),
        "max_abs_over_scale": float(diff.max() / scale),
        "rms_rel": float(np.sqrt((diff**2).mean())) / rms_ref,
        "p99_abs": float(np.percentile(diff, 99)),
    }
    print(
        f"{label:26s} slices={row['slices']:3d} "
        f"max_abs={row['max_abs']:.3e} max_abs/scale={row['max_abs_over_scale']:.3e} "
        f"rms_rel={row['rms_rel']:.3e} p99_abs={row['p99_abs']:.3e}"
    )
    return row


def main() -> None:
    library = build_gemma4_attention(load=True)
    rows = []
    for dtype in ("f32", "bf16"):
        for mask_mode in ("keep", "window", "holes"):
            for keys in (2047, 4096):
                for name, geom in (("sliding hd256", _SLIDING), ("global hd512", _GLOBAL)):
                    rows.append(
                        deviation(
                            library,
                            label=f"{name} {dtype} {mask_mode} k{keys}",
                            dtype=dtype,
                            mask_mode=mask_mode,
                            keys=keys,
                            geom=geom,
                        )
                    )
    worst = max(row["max_abs_over_scale"] for row in rows)
    print(f"worst max_abs/scale = {worst:.3e}")
    if len(sys.argv) > 1:
        with open(sys.argv[1], "w") as handle:
            json.dump({"cases": rows, "worst_max_abs_over_scale": worst}, handle, indent=1)


if __name__ == "__main__":
    main()
