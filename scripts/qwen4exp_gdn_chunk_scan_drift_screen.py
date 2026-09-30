"""GD-4 screen: does a chunked fp16 reassociation of the GDN recurrence drift?

Diagnostic only; writes nothing. Models the arithmetic GD-4 proposes - the same
gated recurrence computed in chunks with fp16 operands and fp16 intra-chunk
accumulation, plus an f32 chunk carry - against the sequential f32 chain the
production kernel runs, at the model's real GDN shape (Hk16/Hv48/Dk=Dv=128),
over a 4K prompt.

The row asks first for state drift over 4K prompts and chunk/tail boundaries,
before any kernel build. This answers the arithmetic half of that: whether a
chunked fp16 reassociation is admissible in principle, and how the drift scales
with chunk size and with a non-divisible tail.

Chunked form, per chunk (A = exp(cumulative log-gate), shifted to start at 1):
    intra[t] = sum_{s<=t} A[t]/A[s] * (q_t . k_s) * v_s
             = (mask_tri * (A[t]/A[s]) * (Q K^T)) @ V
    cross[t] = A[t] * (q_t . S_carry)
    S_next   = A[-1] * S_carry + sum_s (A[-1]/A[s]) * k_s (x) v_s
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

HV, HK, DK, DV = 48, 16, 128, 128
SEQ = 4096
CHUNKS = (64, 128, 256, 512)


def build_inputs(seq, seed=0):
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((seq, HK, DK)) / np.sqrt(DK)
    k = rng.standard_normal((seq, HK, DK)) / np.sqrt(DK)
    v = rng.standard_normal((seq, HV, DV)) / np.sqrt(DV)
    # decay gate in (0,1); log-uniform so both short and long memory are hit
    gate = np.exp(-np.exp(rng.uniform(np.log(1e-3), np.log(1.0), size=(seq, HV))))
    return q, k, v, gate.astype(np.float32)


def repeat_kv(x, hv, hk):
    return np.repeat(x, hv // hk, axis=1)


def sequential(q, k, v, gate):
    """Production arithmetic: f32 chain, one token at a time."""
    seq = q.shape[0]
    kk, qq = repeat_kv(k, HV, HK), repeat_kv(q, HV, HK)
    state = np.zeros((HV, DK, DV))
    out = np.zeros((seq, HV, DV))
    for t in range(seq):
        state = gate[t][:, None, None] * state + kk[t][:, :, None] * v[t][:, None, :]
        out[t] = np.einsum("hk,hkv->hv", qq[t], state)
    return out, state


def chunked(q, k, v, gate, chunk, fp16_ops=True, fp16_intra=True):
    seq = q.shape[0]
    kk, qq = repeat_kv(k, HV, HK), repeat_kv(q, HV, HK)
    state = np.zeros((HV, DK, DV))
    out = np.zeros((seq, HV, DV))
    h = (lambda x: x.astype(np.float16).astype(np.float32)) if fp16_ops else (lambda x: x)
    acc = np.float16 if fp16_intra else np.float32

    for start in range(0, seq, chunk):
        stop = min(start + chunk, seq)
        n = stop - start
        Q, K, V = h(qq[start:stop]), h(kk[start:stop]), h(v[start:stop])
        cum = np.cumsum(np.log(np.maximum(gate[start:stop], 1e-30)), axis=0)
        # Unshifted cumulative log-gate: the decay from the chunk's first
        # position to t is exactly exp(cum[t]). The pairwise ratio is computed
        # in log space so it stays <= 1 and cannot hit 0/0 when exp(cum)
        # underflows on a long chunk with strong decay.
        A = np.exp(cum)
        # intra-chunk: (tril-mask * A[t]/A[s] * (q_t . k_s)) @ V
        G = np.einsum("nhd,shd->hns", Q, K)      # G[h,n,s]
        ratio = np.exp(cum.T[:, :, None] - cum.T[:, None, :])      # [h,t,s]
        mask = np.tril(np.ones((n, n)))
        M = (ratio * mask[None, :, :]).astype(acc).astype(np.float32)
        intra = np.einsum("hns,shd->nhd", M * G, V)
        # cross-chunk
        S16 = state.astype(np.float16).astype(np.float32) if fp16_ops else state
        cross = np.einsum("nhk,hkd->nhd", Q, S16)
        out[start:stop] = intra + cross * A[:, :, None]
        # carry
        w = np.exp(cum[-1][None, :] - cum).astype(acc).astype(np.float32)  # w[s,h]
        carry = np.einsum("sh,shd,she->hde", w, K, V)
        state = A[-1][:, None, None] * state + carry
    return out, state


def nmse(a, b):
    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    return float(np.sum((a64 - b64) ** 2) / np.sum(b64 ** 2))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seq", type=int, default=SEQ)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunks", default=",".join(str(c) for c in CHUNKS))
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()
    seq = args.seq
    chunks = tuple(int(c) for c in args.chunks.split(","))

    q, k, v, gate = build_inputs(seq, seed=args.seed)
    print(f"shape Hk{HK}/Hv{HV}/Dk{DK}/Dv{DV}, seq {seq}, {HV//HK} kv-repeat", flush=True)
    ref_out, ref_state = sequential(q, k, v, gate)
    print("sequential reference done", flush=True)

    print(f"\n{'variant':40s} {'state NMSE':>12s} {'out NMSE':>12s} {'max|dS|':>12s}")
    rows = {}
    for chunk in chunks:
        for label, f16ops, f16intra in (
            ("fp16 operands + fp16 intra", True, True),
            ("fp16 operands, f32 intra", True, False),
            ("f32 operands (reassoc only)", False, False),
        ):
            o, st = chunked(q, k, v, gate, chunk, f16ops, f16intra)
            key = f"chunk{chunk}:{label}"
            rows[key] = {
                "state_nmse": nmse(st, ref_state),
                "out_nmse": nmse(o, ref_out),
                "max_abs_dstate": float(np.max(np.abs(st.astype(np.float64)
                                                      - ref_state.astype(np.float64)))),
            }
            r = rows[key]
            print(f"{key:40s} {r['state_nmse']:12.3e} {r['out_nmse']:12.3e} "
                  f"{r['max_abs_dstate']:12.3e}", flush=True)

    tail_chunk = 384
    print(f"\nnon-divisible tail (chunk {tail_chunk} -> {seq} % {tail_chunk} "
          f"= {seq % tail_chunk})")
    o, st = chunked(q, k, v, gate, tail_chunk)
    print(f"  {'chunk384 (tail 256)':38s} {nmse(st, ref_state):12.3e} "
          f"{nmse(o, ref_out):12.3e}")

    print("\ndrift growth with prompt length (chunk 64, fp16 + fp16 intra)")
    growth = {}
    for prefix in (512, 1024, 2048, seq):
        if prefix > seq:
            continue
        rp_o, rp_s = sequential(q[:prefix], k[:prefix], v[:prefix], gate[:prefix])
        co, cs = chunked(q[:prefix], k[:prefix], v[:prefix], gate[:prefix], 64)
        growth[str(prefix)] = {"state_nmse": nmse(cs, rp_s), "out_nmse": nmse(co, rp_o)}
        print(f"  seq {prefix:5d}  state NMSE {nmse(cs, rp_s):.3e}  "
              f"out NMSE {nmse(co, rp_o):.3e}")

    payload = {
        "kind": "qwen4exp-gdn-chunk-scan-drift-screen",
        "shape": {"Hk": HK, "Hv": HV, "Dk": DK, "Dv": DV, "seq": seq,
                  "kv_repeat": HV // HK, "seed": args.seed},
        "host": {"platform": platform.platform(), "python": sys.version.split()[0],
                 "numpy": np.__version__},
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "variants": {k: {kk: float(vv) for kk, vv in val.items()}
                     for k, val in rows.items()},
        "tail": {"chunk": tail_chunk, "tail_len": seq % tail_chunk,
                 "state_nmse": nmse(st, ref_state), "out_nmse": nmse(o, ref_out)},
        "growth": growth,
        "caveat": ("synthetic unit-scale inputs and a NumPy model of the fp16 "
                   "arithmetic, not the real kernel; state/output NMSE is not a "
                   "KL measurement and does not establish production "
                   "admissibility"),
    }
    print("\nJSON:", json.dumps(payload["variants"]))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
