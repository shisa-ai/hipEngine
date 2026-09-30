"""GD-4 screen (correct form): how much does the chunkwise/WY reassociation add
over the serial GDN recurrence's own fp32 rounding?

Replaces `qwen4exp_gdn_chunk_scan_drift_screen.py`, which modelled gated LINEAR
attention and was retracted - GDN is the gated DELTA rule. See
worklog/entries/20260930T155808.304732Z-lhl-qwen4exp-gd4-screen-retraction-23cc13.md.

The question this answers: GPF-8 built a correct chunkwise/WY GDN kernel
(byte-exact at one token) and still failed the model gate at KL 0.056522, at
FP32. Either the WY reassociation itself is numerically dangerous, or GPF-8's KL
came from something else. This screen separates those by measuring the WY form's
fp32 drift against the serial form's fp32 drift on identical inputs, both scored
against the float64 oracle.

Reference (float64, from the validated CPU oracle):
    P[t] = cumprod(decay)[t]                     # prefix decay within the chunk
    R[t,s] = prod(decay[s+1..t])                 # transition, R[t,t] = 1
    delta[t] = beta[t]*(v[t] - P[t]*(k[t]@H0)
                        - sum_{s<t} R[t,s]*(k[t].k[s])*delta[s])
    out[t]   = P[t]*(q[t]@H0) + sum_{s<=t} R[t,s]*(q[t].k[s])*delta[s]
    H1       = P[-1]*H0 + sum_s R[-1,s]*outer(k[s], delta[s])

Serial form (what the production kernel runs, per token):
    memory = k . (decay * S)
    delta  = (v - memory) * beta
    S      = decay * S + outer(k, delta)
    out    = q . S

Both are evaluated in float32; the oracle in float64. Diagnostic only.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

HV, HK, DK, DV = 48, 16, 128, 128
SEQ = 1024
CHUNKS = (8, 16, 32, 64)


def build_inputs(seq: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    q = (rng.standard_normal((seq, HK, DK)) / np.sqrt(DK)).astype(np.float32)
    k = (rng.standard_normal((seq, HK, DK)) / np.sqrt(DK)).astype(np.float32)
    v = (rng.standard_normal((seq, HV, DV)) / np.sqrt(DV)).astype(np.float32)
    beta = rng.uniform(0.1, 0.9, size=(seq, HV)).astype(np.float32)
    decay = rng.uniform(0.7, 0.999, size=(seq, HV)).astype(np.float32)
    return q, k, v, beta, decay


def repeat_kv(x: np.ndarray) -> np.ndarray:
    return np.repeat(x, HV // HK, axis=1)


def serial(q, k, v, beta, decay, dtype=np.float32, fp16_ops=False):
    """Production arithmetic, one token at a time.

    fp16_ops rounds the operands (Q, K, V and the carried state) to fp16 and
    accumulates in fp32 - the arithmetic class GD-4 proposes via fp16 WMMA.
    """
    h = (lambda x: x.astype(np.float16).astype(dtype)) if fp16_ops else (lambda x: x)
    seq = q.shape[0]
    kk, qq = repeat_kv(k), repeat_kv(q)
    S = np.zeros((HV, DK, DV), dtype=dtype)
    out = np.zeros((seq, HV, DV), dtype=dtype)
    for t in range(seq):
        d = decay[t].astype(dtype)[:, None, None]
        decayed = (d * S).astype(dtype)
        memory = np.einsum("hk,hkv->hv", h(kk[t]), h(decayed)).astype(dtype)
        delta = ((v[t] - memory) * beta[t][:, None]).astype(dtype)
        S = (decayed + h(kk[t])[:, :, None] * delta[:, None, :]).astype(dtype)
        out[t] = np.einsum("hk,hkv->hv", h(qq[t]), h(S)).astype(dtype)
    return out, S


def chunked_wy(q, k, v, beta, decay, chunk: int, dtype=np.float32, fp16_ops=False):
    """The triangular/WY form the row's reference uses, evaluated in `dtype`."""
    seq = q.shape[0]
    kk, qq = repeat_kv(k), repeat_kv(q)
    S = np.zeros((HV, DK, DV), dtype=dtype)
    out = np.zeros((seq, HV, DV), dtype=dtype)
    for start in range(0, seq, chunk):
        stop = min(start + chunk, seq)
        n = stop - start
        Q = qq[start:stop].astype(dtype)
        K = kk[start:stop].astype(dtype)
        V = v[start:stop].astype(dtype)
        B = beta[start:stop].astype(dtype)
        D = decay[start:stop].astype(dtype)
        # P[t] = prefix decay; R[t,s] = decay after s through t. Both indexed [t,s,h].
        P = np.cumprod(D, axis=0).astype(dtype)                       # [n,HV]
        logR = np.cumsum(np.log(D.astype(np.float64)), axis=0)        # [n,HV]
        R = np.exp(logR[:, None, :] - logR[None, :, :]).astype(dtype)  # [t,s,h]
        # np.tril on a 3-D array masks the LAST two axes, not the first two, so
        # build the [t,s] causal masks explicitly.
        causal = np.tril(np.ones((n, n), dtype=bool), -1)             # strict s < t
        causal_incl = np.tril(np.ones((n, n), dtype=bool), 0)         # s <= t
        Rstrict = np.where(causal[:, :, None], R, 0.0).astype(dtype)
        Rincl = np.where(causal_incl[:, :, None], R, 0.0).astype(dtype)
        h = (lambda x: x.astype(np.float16).astype(dtype)) if fp16_ops else (lambda x: x)
        key_state = np.einsum("nhk,hkv->nhv", h(K), h(S)).astype(dtype)    # k @ H0
        query_state = np.einsum("nhk,hkv->nhv", h(Q), h(S)).astype(dtype)  # q @ H0
        key_key = np.einsum("nhk,shk->nsh", h(K), h(K)).astype(dtype)      # k_t . k_s
        query_key = np.einsum("nhk,shk->nsh", h(Q), h(K)).astype(dtype)    # q_t . k_s
        delta = np.zeros((n, HV, DV), dtype=dtype)
        for t in range(n):
            correction = np.einsum(
                "sh,sh,shv->hv", Rstrict[t], key_key[t], delta).astype(dtype)
            delta[t] = (B[t][:, None] * (V[t] - P[t][:, None] * key_state[t]
                                         - correction)).astype(dtype)
        coeff = (Rincl * query_key).astype(dtype)                     # [t,s,h]
        out[start:stop] = (P[:, :, None] * query_state
                           + np.einsum("tsh,shv->thv", coeff, delta)).astype(dtype)
        w = R[n - 1].astype(dtype)                                    # [s,h] = R[-1,s]
        S = (P[-1][:, None, None] * S
             + np.einsum("sh,shk,shv->hkv", w, K, delta).astype(dtype)).astype(dtype)
    return out, S


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
    chunks = tuple(int(c) for c in args.chunks.split(","))

    q, k, v, beta, decay = build_inputs(args.seq, seed=args.seed)
    print(f"shape Hk{HK}/Hv{HV}/Dk{DK}/Dv{DV}, seq {args.seq}, "
          f"{HV // HK} kv-repeat, beta[0.1,0.9], decay[0.7,0.999]", flush=True)

    # float64 oracle, from the committed CPU reference implementation.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from hipengine.kernels.cpu_reference import (
        gdn_prefill_chunkwise_wy_segments,
        gdn_prefill_recurrent_segments,
    )
    cu = np.asarray([0, args.seq], dtype=np.int32)
    si = np.asarray([0], dtype=np.int64)
    zero = np.zeros((1, HV, DK, DV), dtype=np.float32)
    ref_out, ref_state = gdn_prefill_recurrent_segments(
        repeat_kv(q), repeat_kv(k), v, beta, decay, zero, cu, si)
    # Cross-check the committed chunkwise oracle against the serial oracle.
    wy_out, wy_state = gdn_prefill_chunkwise_wy_segments(
        repeat_kv(q), repeat_kv(k), v, beta, decay, zero, cu, si, chunk_size=8)
    print(f"\ncommitted-oracle cross-check (both f64): "
          f"NMSE(wy8, serial) out {nmse(wy_out, ref_out):.3e} "
          f"state {nmse(wy_state, ref_state):.3e}", flush=True)

    rows = {}
    print(f"\n{'variant':38s} {'out NMSE':>12s} {'state NMSE':>12s} {'max|dout|':>12s}")
    o, s = serial(q, k, v, beta, decay)
    rows["serial_fp32"] = {"out_nmse": nmse(o, ref_out),
                           "state_nmse": nmse(s, ref_state),
                           "max_abs_dout": float(np.max(np.abs(
                               o.astype(np.float64) - ref_out.astype(np.float64))))}
    print(f"{'serial fp32 (production form)':38s} "
          f"{rows['serial_fp32']['out_nmse']:12.3e} {rows['serial_fp32']['state_nmse']:12.3e} "
          f"{rows['serial_fp32']['max_abs_dout']:12.3e}", flush=True)

    for chunk in chunks:
        t0 = time.time()
        o, s = chunked_wy(q, k, v, beta, decay, chunk)
        key = f"wy_fp32_chunk{chunk}"
        rows[key] = {"out_nmse": nmse(o, ref_out),
                     "state_nmse": nmse(s, ref_state),
                     "max_abs_dout": float(np.max(np.abs(
                         o.astype(np.float64) - ref_out.astype(np.float64))))}
        r = rows[key]
        ratio = r["out_nmse"] / max(rows["serial_fp32"]["out_nmse"], 1e-300)
        print(f"{'wy fp32 chunk ' + str(chunk):38s} "
              f"{r['out_nmse']:12.3e} {r['state_nmse']:12.3e} {r['max_abs_dout']:12.3e}"
              f"   ({ratio:6.2f}x serial)", flush=True)
        del o, s

    print(f"\n{'variant':38s} {'out NMSE':>12s} {'state NMSE':>12s} {'max|dout|':>12s}")
    for label, chunk, f16 in (("serial fp16 operands", None, True),
                              ("wy fp16 operands chunk 64", 64, True),
                              ("wy fp16 operands chunk 128", 128, True)):
        if chunk is None:
            o, s = serial(q, k, v, beta, decay, fp16_ops=True)
        else:
            o, s = chunked_wy(q, k, v, beta, decay, chunk, fp16_ops=True)
        key = label.replace(" ", "_")
        rows[key] = {"out_nmse": nmse(o, ref_out),
                     "state_nmse": nmse(s, ref_state),
                     "max_abs_dout": float(np.max(np.abs(
                         o.astype(np.float64) - ref_out.astype(np.float64)))),
                     "chunk": chunk, "fp16_operands": True}
        r = rows[key]
        ratio = r["out_nmse"] / max(rows["serial_fp32"]["out_nmse"], 1e-300)
        print(f"{label:38s} {r['out_nmse']:12.3e} {r['state_nmse']:12.3e} "
              f"{r['max_abs_dout']:12.3e}   ({ratio:8.1f}x serial)", flush=True)
        del o, s

    payload = {
        "kind": "qwen4exp-gdn-wy-chunk-drift-screen",
        "replaces": "qwen4exp_gdn_chunk_scan_drift_screen.py (retracted: modelled gated linear attention)",
        "shape": {"Hk": HK, "Hv": HV, "Dk": DK, "Dv": DV, "seq": args.seq,
                  "kv_repeat": HV // HK, "seed": args.seed},
        "host": {"platform": platform.platform(), "python": sys.version.split()[0],
                 "numpy": np.__version__},
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "variants": rows,
        "committed_oracle_cross_check": {"out_nmse": nmse(wy_out, ref_out),
                                         "state_nmse": nmse(wy_state, ref_state)},
        "caveat": ("synthetic inputs and a NumPy model of the fp32 arithmetic, not the "
                   "real kernel; this measures arithmetic drift only and does not "
                   "establish production admissibility"),
    }
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
