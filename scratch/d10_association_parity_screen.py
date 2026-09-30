"""D9 stage 2: association-parity rate of the fast q8_0 decode candidates vs legacy.

The pack8 decode rewrite was rejected at the Gemma production gate because its
8-consecutive-k walk reassociates the f32 sum against legacy's unit-stride walk,
flipping a ~1e-4 fraction of bf16 outputs (one ULP) and failing the frozen
teacher-forced chain (kl_max 2.447 vs 0.05).  The D9 row's screen
(``d10_q8_dense_gemv_screen.py``) timed every registered candidate and showed
the t16 family and rowvec8 are also 15-50% faster than legacy, but a
single-sample per-shape comparison cannot resolve a 1e-4 element rate -- it
observed ``0`` diffs even for the known-rejected kernel.

This script measures the rate properly over many independent trials at the
real production shapes:

* ``pack8_gemv_decode`` serves as the **instrument control**: the pinned parity
  test (``tests/test_gpu_gguf_q8_0_pack8_gemv_decode_parity.py``) measured
  ~1e-4 there, so a working instrument MUST report nonzero for it.
* ``t16_gemv_decode`` (and rowvec8 on the pair shape) are the candidates whose
  rate decides whether the D9 tuning win is takeable without a gate rerun.

Verdicts: rate == 0 over N trials bounds the candidate's divergence below
``-ln(0.05) / total_elements`` (95%); a nonzero rate means the candidate needs
the frozen teacher-forced gate exactly like the rejected rewrite did.

Usage:
  env -u HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES=1 PYTHONPATH=. \
      .venv/bin/python scratch/d10_association_parity_screen.py [trials]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import d10_q8_dense_gemv_screen as scr  # noqa: E402

from hipengine.core.hip import get_hip_runtime  # noqa: E402

# The production shape set lives with the timing screen so both instruments
# measure the same matrix.
SINGLE_SHAPES = scr.SINGLE_SHAPES


def bits(x_device: int, n: int) -> np.ndarray:
    return np.frombuffer(scr._download(x_device, n * 2), dtype=np.uint16)


def rate_vs_legacy_single(name, fn, kind, k, out, trials, seed):
    """Element-level bf16 bit divergence rate of candidate vs legacy."""
    legacy = scr.candidates_single()[0][1]
    total = 0
    diffs = 0
    t0 = time.time()
    for t in range(trials):
        rng = np.random.default_rng(seed + t)
        f32 = rng.standard_normal(k, dtype=np.float32)
        xbits = (f32.view(np.uint32) >> 16).astype(np.uint16)
        x = scr._alloc(k * 2)
        scr._upload(x, xbits.tobytes())
        w = scr.make_w(out, k)
        w_arg = w if kind == "raw" else scr.make_t16_tiles(w, out, k)
        o_legacy = scr.make_out(out * 2)
        o_cand = scr.make_out(out * 2)
        scr.call_single(legacy, x, w, o_legacy, 1, k, out)
        scr.call_single(fn, x, w_arg, o_cand, 1, k, out)
        get_hip_runtime().device_synchronize()
        a = bits(o_legacy, out)
        b = bits(o_cand, out)
        diffs += int(np.count_nonzero(a != b))
        total += out
        scr._free_all((x, w, w_arg, o_legacy, o_cand))
    rate = diffs / total if total else 0.0
    bound = -np.log(0.05) / total if diffs == 0 else float("nan")
    return dict(
        candidate=name,
        trials=trials,
        elements=total,
        diffs=diffs,
        rate=rate,
        upper95=bound if diffs == 0 else None,
        seconds=time.time() - t0,
    )


def main() -> int:
    trials = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    results = []
    controls = set()
    for shape_i, (label, k, out) in enumerate(SINGLE_SHAPES):
        for name, fn, kind in scr.candidates_single()[1:]:  # skip legacy itself
            seed = 9000 + shape_i * 100
            r = rate_vs_legacy_single(name, fn, kind, k, out, trials, seed)
            r["shape"] = label
            results.append(r)
            if "pack8_gemv_decode" in name:
                controls.add(name)
            print(
                f"{label:<16}{name:<30}diffs={r['diffs']:<8}of {r['elements']:<11}"
                f"rate={r['rate']:.3e}  ({r['seconds']:.0f}s)",
                flush=True,
            )

    print("\n== verdict ==")
    control_rates = [r["rate"] for r in results if "pack8_gemv_decode" in r["candidate"]]
    control_nonzero = any(r["diffs"] for r in results if "pack8_gemv_decode" in r["candidate"])
    print(
        f"instrument control (pack8_gemv_decode, the rejected kernel): "
        f"nonzero observed = {control_nonzero}, rates = "
        f"{[f'{v:.2e}' for v in control_rates]}"
    )
    if not control_nonzero:
        print(
            "WARNING: control shows 0 -- instrument underpowered or kernels "
            "changed; do not trust candidate zeros below."
        )
    for r in results:
        if "pack8_gemv_decode" in r["candidate"]:
            continue
        if r["diffs"]:
            print(
                f"{r['shape']}: {r['candidate']} DIVERGES at rate {r['rate']:.3e} "
                f"-- needs the frozen teacher-forced gate before any wiring"
            )
        else:
            print(
                f"{r['shape']}: {r['candidate']} 0/{r['elements']} bits differ; "
                f"95% upper bound on rate < {r['upper95']:.3e}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())