"""Which arm is wrong: the flash route, or the battery's reference?

The flash battery's two arms disagree by far more than f32 precision, at every
scale. That says one of them is wrong, but not which. This adds a numpy oracle
as the third opinion, on the same inputs, and reports each arm's error against
it. Whichever arm tracks the oracle is the correct one; the other is the defect.
"""

from __future__ import annotations

import numpy as np

from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention import (
    build_gemma4_attention,
)
from scratch.flash_scale_probe import GEOMS, ab


def oracle(query, key, value, mask, scale):
    """Plain f64 softmax attention, the arithmetic both arms intend."""
    q = query[0].astype(np.float64)                       # (H, D)
    k = key.astype(np.float64)                            # (T, KV, D)
    v = value.astype(np.float64)                          # (T, KV, D)
    ratio = q.shape[0] // k.shape[1]
    out = np.zeros_like(q)
    keep = mask[0].astype(bool)                           # (T,)
    for h in range(q.shape[0]):
        kv = h // ratio
        logits = (k[:, kv, :] @ q[h]) * scale             # (T,)
        logits = np.where(keep, logits, -np.inf)
        top = np.max(logits)
        if not np.isfinite(top):
            out[h] = np.nan
            continue
        w = np.exp(logits - top)
        w /= w.sum()
        out[h] = w @ v[:, kv, :]
    return out[None].astype(np.float32)


def main() -> None:
    library = build_gemma4_attention(load=True)
    rng = np.random.default_rng(20260930)
    print(f"{'geom':>8} {'keys':>5} {'scale':>9} {'single_err':>12} {'flash_err':>12} {'verdict':>22}")
    for name, geom in GEOMS.items():
        for keys in (1024, 4096):
            head_dim = geom["head_dim"]
            for label, scale in (("1.0", 1.0), ("1/sqrt(d)", 1.0 / np.sqrt(head_dim))):
                # Rebuild the identical fixture: ab() re-seeds from the same
                # constant, so the arrays match by construction.
                num_heads, num_kv_heads = geom["num_heads"], geom["num_kv_heads"]
                r2 = np.random.default_rng(20260930)
                query = np.ascontiguousarray(
                    r2.standard_normal((1, num_heads, head_dim)) * 0.7, dtype=np.float32
                )
                key = np.ascontiguousarray(
                    r2.standard_normal((keys, num_kv_heads, head_dim)) * 0.7, dtype=np.float32
                )
                value = np.ascontiguousarray(
                    r2.standard_normal((keys, num_kv_heads, head_dim)) * 0.7, dtype=np.float32
                )
                mask = np.ones((1, keys), dtype=np.uint8)
                ref = oracle(query, key, value, mask, scale)

                single, flash, _ = ab(
                    library, keys=keys, scale=scale, mask_mode="keep", **geom
                )
                single_err = float(np.max(np.abs(single - ref)))
                flash_err = float(np.max(np.abs(flash - ref)))
                verdict = (
                    "flash WRONG" if flash_err > 10 * max(single_err, 1e-6)
                    else "single WRONG" if single_err > 10 * max(flash_err, 1e-6)
                    else "both within 10x"
                )
                print(
                    f"{name:>8} {keys:>5} {label:>9} {single_err:12.3e} "
                    f"{flash_err:12.3e} {verdict:>22}"
                )


if __name__ == "__main__":
    main()
