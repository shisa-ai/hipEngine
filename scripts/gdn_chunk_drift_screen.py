"""Screen a chunked GDN scan for state drift before any kernel is built.

Ledger row ``GD-4`` proposes replacing the sequential GDN prefill recurrence with a
chunked scan on fp16 WMMA (halo-box PR #91: fused conv/SiLU, q/k norm, gated RMS norm).
Its own action says the first thing to do is a **numerical/state screen**, because the
recurrence carries state across the whole prompt and a chunked schedule changes where
that state is carried and in what order the per-token terms are summed.

This script answers that question without building the kernel.

The recurrence is affine in the state. With ``D_t = prod_{i<=t} d_i``::

    d_t     = exp(-exp(a_log) * softplus(alpha_t + dt_bias))
    delta_t = beta_t * (v_t - k_t . (S_0 D_t) - sum_{j<t} (k_t.k_j)(D_t/D_j) delta_j)
    S_t     = S_0 D_t + sum_{j<=t} (k_j (x) delta_j)(D_t/D_j)
    o_t     = q_t . S_t,  then RMS-norm and gate

The first two lines are the UT/WY transform: a lower-triangular solve, exactly equivalent
to the sequential recurrence and exact in exact arithmetic. So the chunked form is *not*
an approximation, and the only question is what floating-point reordering costs.

Three effects are screened separately:

1. **Summation order inside a chunk** - the chunked form accumulates the ``k (x) delta``
   terms as a matrix product and applies prefix decay ratios, where the sequential form
   interleaves one FMA chain per token.
2. **State-carry precision at chunk boundaries** - the chunked form materialises the
   carried state once per chunk instead of once per token, which is where an fp16 WMMA
   path would round.
3. **Tail and chunk-boundary behaviour** - prompt lengths that are not multiples of the
   chunk size.

The sequential reference is ``replay_gdn`` from ``scripts/gguf_gdn_replay_compare.py``, so
this screen and the existing GPU-verification harness agree on the arithmetic by
construction rather than by transcription.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from gguf_gdn_replay_compare import GDNShape, replay_gdn  # noqa: E402

# Qwen4Exp's GDN shape, as used by the four registered routes measured in GD-3's screen.
DEFAULT_SHAPE = GDNShape(num_k_heads=16, num_v_heads=48, head_k_dim=128, head_v_dim=128)
DEFAULT_EPS = 1.0e-6
_TINY = 1.0e-30


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float32)


def _softplus(x: np.ndarray) -> np.ndarray:
    return np.where(x > 20.0, x, np.log1p(np.exp(np.minimum(x, 20.0)))).astype(np.float32)


def _silu(x: np.ndarray) -> np.ndarray:
    return (x / (1.0 + np.exp(-x))).astype(np.float32)


def make_inputs(tokens: int, shape: GDNShape, *, seed: int) -> dict[str, np.ndarray]:
    """Synthetic inputs at the scale the real path produces.

    The q/k halves of ``conv_out`` are normalised inside the recurrence, so unit-scale
    draws are faithful there. ``alpha`` is a small positive activation and ``beta`` a
    pre-sigmoid logit, matching the kernel's call sites.
    """
    rng = np.random.default_rng(seed)
    return {
        "conv_out": rng.standard_normal((tokens, shape.qkv_width), dtype=np.float32),
        "gate": rng.standard_normal((tokens, shape.value_dim), dtype=np.float32),
        "alpha": np.abs(rng.standard_normal((tokens, shape.num_v_heads), dtype=np.float32)),
        "beta": rng.standard_normal((tokens, shape.num_v_heads), dtype=np.float32),
        "norm_weight": np.ones((shape.head_v_dim,), dtype=np.float32),
        "dt_bias": np.zeros((shape.num_v_heads,), dtype=np.float32),
        "a_log": np.log(
            np.abs(rng.standard_normal((shape.num_v_heads,), dtype=np.float32)) + 0.5
        ).astype(np.float32),
    }


def _ratio_matrix(decay: np.ndarray, chunk: int) -> np.ndarray:
    """``R[i, j] = D_i / D_j`` for ``j <= i``, zero above the diagonal.

    ``D_i / D_j`` is the decay accumulated strictly between j and i, so it is computed
    here as a **direct suffix product** rather than as a ratio of prefix products.

    That is not a stylistic choice. The prefix form is numerically unsafe at long
    chunks: ``cum`` decays geometrically and eventually underflows to zero, but the
    underflow is not simultaneous across tokens. When ``cum[j]`` has underflowed while
    ``cum[i]`` has not, the ratio ``cum[i] / max(cum[j], tiny)`` **explodes**, the
    triangular matrix becomes enormous, and the solve diverges. Measured on the fp32
    reference this drove max abs error to 1.0e1 at chunk 256 while chunk 64 was two
    orders of magnitude better - larger chunks getting *worse*, which is the signature.
    The suffix product never divides, and each factor is a decay in (0, 1], so every
    entry stays bounded by 1 and underflows monotonically to zero.
    """
    ratio = np.zeros((chunk, chunk), dtype=np.float32)
    for i in range(chunk):
        acc = np.ones((i + 1,), dtype=np.float32)
        running = np.float32(1.0)
        # walk backwards from i: R[i, j] = prod_{k=j+1..i} decay[k]
        for j in range(i - 1, -1, -1):
            running = np.float32(running * decay[j + 1])
            acc[j] = running
        ratio[i, : i + 1] = acc
    return ratio


def _chunk_local(
    conv_out: np.ndarray,
    gate: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    norm_weight: np.ndarray,
    dt_bias: np.ndarray,
    a_log: np.ndarray,
    shape: GDNShape,
    eps: float,
    state: np.ndarray,
    *,
    state_dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray]:
    """One chunk via the UT transform; returns ``(outputs, carried_state)``."""
    chunk = conv_out.shape[0]
    num_v, dk, dv = shape.num_v_heads, shape.head_k_dim, shape.head_v_dim
    carried = np.array(state, dtype=state_dtype)
    outputs = np.empty((chunk, shape.value_dim), dtype=np.float32)
    sqrt_dk = np.sqrt(np.float32(dk))
    eye = np.eye(chunk, dtype=np.float32)

    for v_head in range(num_v):
        k_head = v_head % shape.num_k_heads
        q_base = k_head * dk
        k_base = shape.key_dim + k_head * dk
        v_base = 2 * shape.key_dim + v_head * dv

        q = conv_out[:, q_base : q_base + dk]
        k = conv_out[:, k_base : k_base + dk]
        value = conv_out[:, v_base : v_base + dv]

        q_scale = (np.float32(1.0) / np.maximum(np.sqrt(np.sum(q * q, axis=1)), 1.0e-6)) / sqrt_dk
        k_scale = np.float32(1.0) / np.maximum(np.sqrt(np.sum(k * k, axis=1)), 1.0e-6)
        q_norm = (q * q_scale[:, None]).astype(np.float32)
        k_norm = (k * k_scale[:, None]).astype(np.float32)

        beta_v = _sigmoid(beta[:, v_head])
        decay = np.exp(
            -np.exp(a_log[v_head]) * _softplus(alpha[:, v_head] + dt_bias[v_head])
        ).astype(np.float32)
        cum = np.cumprod(decay).astype(np.float32)
        ratio = _ratio_matrix(decay, chunk)

        start = np.array(carried[v_head], dtype=np.float32)
        # k_t . (S_0 D_t) is a dv-vector per token; the recurrence is independent across
        # the value dimension except through this term, so L below is shared by all dv
        # columns and the solve is one triangular system with dv right-hand sides.
        start_term = (cum[:, None] * (k_norm @ start).astype(np.float32)).astype(np.float32)
        # L[t, j] = beta_t (k_t . k_j) (D_t / D_j), strictly lower triangular
        kk = (k_norm @ k_norm.T).astype(np.float32)
        lower = np.tril((kk * ratio) * beta_v[:, None], -1).astype(np.float32)
        rhs = ((value - start_term) * beta_v[:, None]).astype(np.float32)
        # forward substitution: (I + L) delta = rhs, one dv-wide solve
        delta = np.empty_like(rhs)
        for i in range(chunk):
            acc = rhs[i].copy()
            if i:
                acc -= lower[i, :i] @ delta[:i]
            delta[i] = acc.astype(np.float32)

        # o_t = q_t . (S_0 D_t) + sum_{j<=t} (q_t.k_j)(D_t/D_j) delta_j
        qk = (q_norm @ k_norm.T).astype(np.float32)
        attn = np.tril((qk * ratio)).astype(np.float32)
        out_acc = (
            (attn @ delta).astype(np.float32)
            + (cum[:, None] * (q_norm @ start).astype(np.float32)).astype(np.float32)
        ).astype(np.float32)
        inv_rms = (np.float32(1.0) / np.sqrt(np.mean(out_acc * out_acc, axis=1) + np.float32(eps))).astype(np.float32)
        gate_v = gate[:, v_head * dv : (v_head + 1) * dv]
        outputs[:, v_head * dv : (v_head + 1) * dv] = (
            out_acc * inv_rms[:, None] * norm_weight * _silu(gate_v)
        ).astype(np.float32)

        # S_end = S_0 D_T + sum_j (k_j (x) delta_j)(D_T/D_j)
        w_end = ratio[-1, :][:, None].astype(np.float32)
        end = (start * cum[-1]).astype(np.float32) + ((k_norm * w_end).T @ delta).astype(np.float32)
        carried[v_head] = end.astype(state_dtype)

    del eye
    return outputs, carried


def chunked_reference(
    *,
    conv_out: np.ndarray,
    gate: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    norm_weight: np.ndarray,
    dt_bias: np.ndarray,
    a_log: np.ndarray,
    shape: GDNShape,
    eps: float,
    chunk_size: int,
    state_dtype: np.dtype = np.float32,
) -> np.ndarray:
    tokens = conv_out.shape[0]
    state = np.zeros((shape.num_v_heads, shape.head_k_dim, shape.head_v_dim), dtype=state_dtype)
    out = np.empty((tokens, shape.value_dim), dtype=np.float32)
    for start in range(0, tokens, chunk_size):
        stop = min(start + chunk_size, tokens)
        piece, state = _chunk_local(
            conv_out[start:stop], gate[start:stop], alpha[start:stop], beta[start:stop],
            norm_weight, dt_bias, a_log, shape, eps, state, state_dtype=state_dtype,
        )
        out[start:stop] = piece
    return out


def _metrics(ref: np.ndarray, cand: np.ndarray) -> dict[str, float]:
    diff = cand.astype(np.float64) - ref.astype(np.float64)
    rel = np.abs(diff) / np.maximum(np.abs(ref.astype(np.float64)), 1.0e-12)
    return {
        "max_abs": float(np.max(np.abs(diff))),
        "rms_abs": float(np.sqrt(np.mean(diff * diff))),
        "nmse": float(np.sum(diff * diff) / max(np.sum(ref.astype(np.float64) ** 2), 1e-30)),
        "max_rel": float(np.max(rel)),
        "p99_rel": float(np.percentile(rel, 99.0)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tokens", type=int, default=4096)
    ap.add_argument("--chunks", type=int, nargs="+", default=[32, 64, 128, 256])
    ap.add_argument("--tails", type=int, nargs="+", default=[4095, 4097, 4100])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", type=pathlib.Path)
    args = ap.parse_args()

    shape, eps = DEFAULT_SHAPE, DEFAULT_EPS
    report: dict[str, object] = {
        "shape": {"num_k_heads": shape.num_k_heads, "num_v_heads": shape.num_v_heads,
                  "head_k_dim": shape.head_k_dim, "head_v_dim": shape.head_v_dim},
        "tokens": args.tokens,
        "note": ("Numerical screen only; no kernel was built. The chunked form is "
                 "algebraically identical to the sequential one, so every figure below "
                 "is floating-point reordering cost, not approximation error."),
        "results": [],
    }

    inputs = make_inputs(args.tokens, shape, seed=args.seed)
    ref = replay_gdn(shape=shape, eps=eps, **inputs)
    print(f"sequential reference (fp32): {args.tokens} tokens, out {ref.shape}, "
          f"rms {float(np.sqrt(np.mean(ref.astype(np.float64) ** 2))):.4f}")
    print()
    print(f"{'chunk':>6} {'state':>6} {'max_abs':>11} {'rms_abs':>11} {'nmse':>11} {'p99_rel':>10}")
    for chunk in args.chunks:
        for state_dtype, label in ((np.float32, "fp32"), (np.float16, "fp16")):
            cand = chunked_reference(shape=shape, eps=eps, chunk_size=chunk,
                                     state_dtype=state_dtype, **inputs)
            m = _metrics(ref, cand)
            report["results"].append({"kind": "chunk", "chunk_size": chunk,
                                      "state_dtype": label, **m})
            print(f"{chunk:>6} {label:>6} {m['max_abs']:>11.3e} {m['rms_abs']:>11.3e} "
                  f"{m['nmse']:>11.3e} {m['p99_rel']:>10.3e}")

    print()
    print("chunk/tail boundary check (chunk 128, fp32 state):")
    print(f"{'tokens':>7} {'max_abs':>11} {'nmse':>11} {'p99_rel':>10}")
    for tokens in args.tails:
        ti = make_inputs(tokens, shape, seed=args.seed)
        m = _metrics(replay_gdn(shape=shape, eps=eps, **ti),
                     chunked_reference(shape=shape, eps=eps, chunk_size=128, **ti))
        report["results"].append({"kind": "tail", "tokens": tokens, "chunk_size": 128,
                                  "state_dtype": "fp32", **m})
        print(f"{tokens:>7} {m['max_abs']:>11.3e} {m['nmse']:>11.3e} {m['p99_rel']:>10.3e}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
