#!/usr/bin/env python3
"""Headroom of the rows==1 local32 IQ decode owner at verifier rows.

The strict per-row GEMV owns dense raw-IQ rows 2-7 (Phase 1 attribution) and
is weight-decode-bound: its time is nearly flat from rows 1 to 4. The local32
decode owner is 2.3-2.6x faster than that GEMV at rows==1 but is hard-coded to
rows=1. This measures the strict GEMV at rows 1/2/4 against local32 at rows=1
on the same real tensors, so the rows 2-4 sibling's headroom is bounded by
measured numbers rather than extrapolated.

CPU/GPU: needs a GPU. Reads real tensors from a published UD file.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import (copy_device_to_host, copy_host_to_device,
                                   free, host_array_ptr, malloc)
from hipengine.kernels.hip_gfx1100.quant.gguf_iq_dense import (
    build_gguf_iq_dense, launch, launch_local32)
from hipengine.loading.gguf import GGUFReader

ROWS = (1, 2, 4)
ROUNDS = 5


def bf16(x):
    bits = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    return ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


def bf16_to_f32(u):
    return (np.asarray(u, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def sweep(tensor, raw, cv, reps, quant):
    n, k = (int(d) for d in tensor.shape)
    hip = get_hip_runtime()
    lib = build_gguf_iq_dense(compiler_version=cv)
    qkey = 'gguf_' + quant.lower()
    results = []
    bufs = []
    try:
        w_b = malloc(raw.nbytes); bufs.append(w_b)
        copy_host_to_device(w_b, host_array_ptr(raw), raw.nbytes)
        for rows in ROWS:
            x = bf16(np.random.default_rng(11).normal(0, 0.1, (rows, k)))
            ref = np.zeros((rows, n), dtype=np.uint16)
            cand = np.zeros((rows, n), dtype=np.uint16)
            x_b = malloc(x.nbytes); bufs.append(x_b)
            copy_host_to_device(x_b, host_array_ptr(x), x.nbytes)
            ref_b = malloc(ref.nbytes); bufs.append(ref_b)
            cand_b = malloc(cand.nbytes); bufs.append(cand_b)

            def run_gemv():
                launch(x_b.ptr, w_b.ptr, ref_b.ptr, rows, k, n,
                       quant=qkey, output='bf16', library=lib)

            def run_local32():
                launch_local32(x_b.ptr, w_b.ptr, cand_b.ptr, 1, k, n,
                               quant=qkey, output='bf16', library=lib)

            best = {}
            for name, fn in (('gemv', run_gemv), ('local32', run_local32)):
                fn()
                hip.device_synchronize()
                best[name] = float('inf')
                for _ in range(ROUNDS):
                    t0 = time.perf_counter()
                    for _ in range(reps):
                        fn()
                    hip.device_synchronize()
                    best[name] = min(best[name], (time.perf_counter() - t0) / reps)

            ref_host = np.zeros((rows, n), dtype=np.uint16)
            cand_host = np.zeros((1, n), dtype=np.uint16)
            copy_device_to_host(host_array_ptr(ref_host), ref_b, ref_host.nbytes)
            copy_device_to_host(host_array_ptr(cand_host), cand_b, cand_host.nbytes)
            a = bf16_to_f32(ref_host).astype(np.float64)
            c = bf16_to_f32(cand_host).astype(np.float64)
            # local32 computes one row; compare against the same row of gemv.
            rel = float(np.abs(a[:1] - c).max() / max(np.abs(a[:1]).max(), 1e-9))
            corr = float(np.corrcoef(a[:1].ravel(), c.ravel())[0, 1])
            results.append(dict(
                rows=rows,
                gemv_ms=best['gemv'] * 1e3,
                local32_ms=best['local32'] * 1e3,
                headroom=best['gemv'] / best['local32'],
                max_rel=rel, corr=corr))
            for b in (cand_b, ref_b, x_b):
                free(b); bufs.remove(b)
    finally:
        for b in reversed(bufs):
            free(b)
    return n, k, results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', type=Path,
                    default=Path(os.environ.get(
                        'HEADROOM_MODEL',
                        '/models/gguf/Qwen3.8-27B-UD-Q4_K_M.gguf')))
    ap.add_argument('--quant', default=os.environ.get('HEADROOM_QUANT', 'IQ4_XS'))
    ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--json', type=Path)
    args = ap.parse_args()
    cv = Path('/tmp/ud-hipcc-version.txt').read_text()
    reader = GGUFReader(args.model)
    seen, picks = set(), []
    for t in reader.info.tensors:
        if (t.ggml_type_name != args.quant or not t.name.startswith('blk.')
                or len(t.shape) != 2):
            continue
        shape = (int(t.shape[0]), int(t.shape[1]))
        if shape in seen or shape[0] % 8 or shape[1] % 256:
            continue
        seen.add(shape); picks.append(t)
    picks.sort(key=lambda t: -t.nbytes)
    picks = picks[:4]
    out = []
    for tensor in picks:
        raw = np.frombuffer(reader.tensor_data(tensor.name), dtype=np.uint8)
        n, k, rows = sweep(tensor, raw, cv, args.reps, args.quant)
        print(f'\n{tensor.name}  N={n} K={k}  {raw.nbytes/1e6:.1f} MB')
        print(f'{"rows":>6s}{"gemv ms":>11s}{"local32 ms":>12s}{"headroom":>10s}'
              f'{"max_rel":>10s}{"corr":>10s}')
        for r in rows:
            print(f'{r["rows"]:6d}{r["gemv_ms"]:11.3f}{r["local32_ms"]:12.3f}'
                  f'{r["headroom"]:10.2f}{r["max_rel"]:10.4f}{r["corr"]:10.6f}')
        out.append(dict(tensor=tensor.name, n=n, k=k, rows=rows))
    if args.json:
        args.json.write_text(json.dumps(out, indent=2) + '\n')


if __name__ == '__main__':
    main()
