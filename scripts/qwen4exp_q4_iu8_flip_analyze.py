#!/usr/bin/env python3
"""Offline analysis of iu8-vs-pair2 BF16 flips against a float64 truth.

Replicates the 512-row uniform-routing probe case (same seeds) for one
tensor, computes a float64 reference dot product, and characterizes each
flipped output: distance of the true value to the nearest BF16 rounding
boundary, cancellation ratio, and magnitude. Diagnostic only.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from hipengine.loading.gguf import GGUFReader, discover_gguf_files
from hipengine.loading.materialize import float_array_to_bf16_bits
from hipengine.quant.gguf import bf16_to_float32
from tests.test_qwen4_exp_pf3_moe_schedules import _make_activation

ROWS = 512
EXPERTS = 512
K = 2560
N = 640
LAYER = 0


def dequant_q4_k_expert(blocks: np.ndarray) -> np.ndarray:
    """Vectorized dequant of one expert's Q4_K blocks (N, 10, 144) -> (N, 2560)."""

    n = blocks.shape[0]
    out = np.empty((n, K), dtype=np.float64)
    scales = blocks[:, :, 4:16].astype(np.int64)  # (n, 10, 12)
    qs = blocks[:, :, 16:]  # (n, 10, 128)
    d = blocks[:, :, 0:2].copy().view(np.float16).astype(np.float64).reshape(n, 10)
    dmin = blocks[:, :, 2:4].copy().view(np.float16).astype(np.float64).reshape(n, 10)

    sc_bits = np.empty((n, 10, 8), dtype=np.int64)
    mn_bits = np.empty((n, 10, 8), dtype=np.int64)
    sc_bits[:, :, :4] = scales[:, :, :4] & 0x3F
    mn_bits[:, :, :4] = scales[:, :, 4:8] & 0x3F

    sc_bits[:, :, 4:] = (scales[:, :, 8:12] & 0x0F) | ((scales[:, :, 0:4] >> 2) & 0x30)
    mn_bits[:, :, 4:] = (scales[:, :, 8:12] >> 4) | ((scales[:, :, 4:8] >> 2) & 0x30)
    s = d[:, :, None] * sc_bits  # (n, 10, 8)
    m = dmin[:, :, None] * mn_bits
    packed = qs.reshape(n, 10, 4, 32).astype(np.int64)  # 4 pairs of 32
    q0 = (packed & 0x0F).astype(np.float64)   # even subblocks
    q1 = ((packed >> 4) & 0x0F).astype(np.float64)  # odd subblocks
    w = np.empty((n, 10, 8, 32), dtype=np.float64)
    for p in range(4):
        w[:, :, 2 * p, :] = s[:, :, 2 * p, None] * q0[:, :, p, :] - m[:, :, 2 * p, None]
        w[:, :, 2 * p + 1, :] = s[:, :, 2 * p + 1, None] * q1[:, :, p, :] - m[:, :, 2 * p + 1, None]
    out = w.reshape(n, K)
    return out


def main() -> None:
    probe = np.load if False else None
    model_root = Path(sys.argv[1] if len(sys.argv) > 1 else
                      "/models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL")
    name = f"blk.{LAYER}.ffn_gate_exps.weight"
    readers = [GGUFReader(p) for p in discover_gguf_files(model_root)]
    reader = next(r for r in readers if any(t.name == name for t in r.info.tensors))
    raw = reader.tensor_data(name)  # (512, 640, 1440) uint8 view

    rng = np.random.default_rng(1788 + ROWS)
    scores = rng.random((ROWS, 512))
    selected = np.argsort(scores, axis=1)[:, :10]
    counts = np.bincount(selected.reshape(-1), minlength=512)
    starts = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
    compact = ROWS * 10
    x_bits, _ = _make_activation(compact, K, 3456 + ROWS)
    x64 = bf16_to_float32(x_bits).astype(np.float64)

    # Load the probe's parent (pair2) and candidate (iu8) gate outputs.
    parent = np.load("/tmp/qwen4exp-iu8-flip-probe-parent-gate.npy")
    cand = np.load("/tmp/qwen4exp-iu8-flip-probe-cand-gate.npy")

    truth = np.zeros((compact, N), dtype=np.float64)
    abs_terms = np.zeros((compact, N), dtype=np.float64)
    raw_f = raw.reshape(EXPERTS, N, K // 256, 144)
    for e in range(EXPERTS):
        lo, hi = starts[e], starts[e + 1]
        if hi <= lo:
            continue
        blocks = raw_f[e]  # (640, 10, 144)
        w = dequant_q4_k_expert(blocks)
        xe = x64[lo:hi]  # (rows_e, 2560)
        truth[lo:hi] = xe @ w.T
        abs_terms[lo:hi] = (np.abs(xe) @ np.abs(w).T)

    def bf16_round(v: np.ndarray) -> np.ndarray:
        return (float_array_to_bf16_bits(v.astype(np.float32))
                .view(np.uint16).reshape(v.shape))

    def bf16_center(bits: np.ndarray) -> np.ndarray:
        return bf16_to_float32(bits).astype(np.float64)

    true_bf16 = bf16_round(truth)
    center = bf16_to_float32(true_bf16).astype(np.float64)
    ulp = np.abs(center) * (2.0 ** -8)
    ulp[center == 0] = 2.0 ** -133
    # distance of truth to the rounding boundary it sits against
    diff = truth - center
    boundary_dist = np.abs(np.abs(diff) - ulp / 2)  # 0 => exactly at boundary
    rel_boundary = boundary_dist / np.maximum(ulp, 1e-300)  # in ulps of center

    p16 = parent.view(np.uint16) if parent.dtype == np.uint16 else parent
    c16 = cand.view(np.uint16) if cand.dtype == np.uint16 else cand
    flips = p16 != c16
    ulp_diff = (c16.astype(np.int32) - p16.astype(np.int32))

    print(f"compact={compact} outputs={compact*N}")
    print(f"flips vs truth: parent={np.count_nonzero(p16 != true_bf16)} "
          f"candidate={np.count_nonzero(c16 != true_bf16)} "
          f"both-same-but-wrong={np.count_nonzero((p16 == c16) & (p16 != true_bf16))}")
    print(f"pair2-vs-iu8 flips: {np.count_nonzero(flips)}")
    print("\nflips with |ulp_diff|==1:")
    m1 = flips & (np.abs(ulp_diff) == 1)
    print(f"  count={np.count_nonzero(m1)}  "
          f"rel_boundary max={rel_boundary[m1].max():.4g} ulps, "
          f"p99={np.percentile(rel_boundary[m1], 99):.4g}, "
          f"median={np.median(rel_boundary[m1]):.4g}")
    print(f"  truth rounds to parent: {np.count_nonzero(p16[m1] == true_bf16[m1])}, "
          f"to candidate: {np.count_nonzero(c16[m1] == true_bf16[m1])}")
    print("\nflips with |ulp_diff|>1:")
    mbig = flips & (np.abs(ulp_diff) > 1)
    print(f"  count={np.count_nonzero(mbig)}")
    if np.count_nonzero(mbig):
        rows_, cols_ = np.nonzero(mbig)
        for r, c in zip(rows_[:20], cols_[:20]):
            print(f"   row={r} col={c} ulp_diff={ulp_diff[r, c]} "
                  f"|truth|={abs(truth[r, c]):.3e} "
                  f"abs_terms={abs_terms[r, c]:.3e} "
                  f"cancel_ratio={abs_terms[r, c] / max(abs(truth[r, c]), 1e-300):.3g} "
                  f"rel_boundary={rel_boundary[r, c]:.4g}")
    # magnitude coverage: what fraction of outputs would a |truth|<t rule repair
    print("\nmagnitude quantiles of |truth| (all outputs):",
          np.percentile(np.abs(truth), [1, 5, 10, 25, 50]))
    if np.any(flips):
        print("magnitude quantiles of |truth| (flipped):",
              np.percentile(np.abs(truth[flips]), [50, 75, 90, 100]))
        print("cancel-ratio quantiles (flipped, |ulp|==1):",
              np.percentile((abs_terms / np.maximum(np.abs(truth), 1e-300))[m1],
                            [50, 90, 99, 100]) if np.any(m1) else "none")


if __name__ == "__main__":
    main()
