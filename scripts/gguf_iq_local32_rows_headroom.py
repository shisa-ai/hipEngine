#!/usr/bin/env python3
"""Headroom of the local32 IQ decode family at verifier rows.

The strict per-row GEMV owns dense raw-IQ rows 2-7 (Phase 1 attribution) and
is weight-decode-bound: its time is nearly flat from rows 1 to 4. The local32
decode owner is 2.0-7.4x faster than that GEMV at rows==1 but is hard-coded to
rows=1. This measures, on the same real tensors and in interleaved min-of-N
rounds:

  gemv     the strict per-row GEMV at rows 1/2/4
  local32  the rows==1 local32 owner launched once per row
  sibling  the rows 2-4 local32 verifier sibling (one block, ROWS rows)

The sibling shares the rows==1 kernel body and split-K rule, so each row is
bit-identical to the rows==1 owner; the third arm shows what that costs.

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
    build_gguf_iq_dense, launch, launch_local32, launch_local32_rows)
from hipengine.loading.gguf import GGUFReader

ROWS = (1, 2, 4)
ROUNDS = 5
# The quants with a rows==1 local32 owner. Q3_K has none, so it has no
# rows 2-4 sibling either; it stays on the strict per-row GEMV.
LOCAL32_QUANTS = ('gguf_iq4_xs', 'gguf_iq4_nl', 'gguf_iq3_s', 'gguf_iq3_xxs',
                  'gguf_iq2_s', 'gguf_iq2_xs')


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
            sib = np.zeros((rows, n), dtype=np.uint16)
            sib_b = malloc(sib.nbytes); bufs.append(sib_b)

            def run_gemv():
                launch(x_b.ptr, w_b.ptr, ref_b.ptr, rows, k, n,
                       quant=qkey, output='bf16', library=lib)

            def run_local32():
                for r in range(rows):
                    launch_local32(x_b.ptr + r * k * 2, w_b.ptr,
                                   cand_b.ptr + r * n * 2, 1, k, n,
                                   quant=qkey, output='bf16', library=lib)

            def run_sibling():
                if rows == 1:  # the sibling is the rows 2-4 geometry
                    return
                launch_local32_rows(x_b.ptr, w_b.ptr, sib_b.ptr, rows, k, n,
                                    quant=qkey, output='bf16', library=lib)

            best = {}
            for name, fn in (('gemv', run_gemv), ('local32', run_local32),
                             ('sibling', run_sibling)):
                if name == 'sibling' and rows == 1:
                    continue
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
            cand_host = np.zeros((rows, n), dtype=np.uint16)
            sib_host = np.zeros((rows, n), dtype=np.uint16)
            copy_device_to_host(host_array_ptr(ref_host), ref_b, ref_host.nbytes)
            copy_device_to_host(host_array_ptr(cand_host), cand_b, cand_host.nbytes)
            copy_device_to_host(host_array_ptr(sib_host), sib_b, sib_host.nbytes)
            a = bf16_to_f32(ref_host).astype(np.float64)
            c = bf16_to_f32(cand_host).astype(np.float64)
            s = (bf16_to_f32(sib_host).astype(np.float64) if rows > 1 else None)
            # local32 runs once per row; compare row-for-row against gemv.
            rel = float(np.abs(a - c).max() / max(np.abs(a).max(), 1e-9))
            corr = float(np.corrcoef(a.ravel(), c.ravel())[0, 1])
            # The sibling must be bit-exact with the per-row rows==1 owner.
            bit_exact = (bool((sib_host == cand_host).all())
                         if rows > 1 else None)
            sib_ms = best.get('sibling')
            sib_rel = (float(np.abs(a - s).max() / max(np.abs(a).max(), 1e-9))
                       if rows > 1 else None)
            results.append(dict(
                rows=rows,
                gemv_ms=best['gemv'] * 1e3,
                local32_ms=best['local32'] * 1e3,
                sibling_ms=None if sib_ms is None else sib_ms * 1e3,
                headroom=best['gemv'] / best['local32'],
                sibling_headroom=None if sib_ms is None else best['gemv'] / sib_ms,
                sibling_vs_local32=None if sib_ms is None else best['local32'] / sib_ms,
                max_rel=rel, corr=corr,
                sibling_bit_exact=bit_exact, sibling_max_rel=sib_rel))
            for b in (sib_b, cand_b, ref_b, x_b):
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
    qkey = 'gguf_' + args.quant.lower()
    if qkey not in LOCAL32_QUANTS:
        # Q3_K has no rows==1 local32 owner, so there is no sibling to bound.
        print(f'{args.quant} has no rows==1 local32 owner; nothing to measure')
        return
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
        print(f'{"rows":>6s}{"gemv ms":>11s}{"local32 ms":>12s}'
              f'{"sibling ms":>12s}{"headroom":>10s}{"sib hdrm":>10s}'
              f'{"sib/l32":>9s}{"bit-ex":>8s}')
        for r in rows:
            sib = ('%10.3f' % r['sibling_ms']) if r['sibling_ms'] else '         -'
            hdrm = ('%10.2f' % r['sibling_headroom']) if r['sibling_headroom'] else '         -'
            ratio = ('%9.2f' % r['sibling_vs_local32']) if r['sibling_vs_local32'] else '        -'
            exact = str(r['sibling_bit_exact'])
            print(f'{r["rows"]:6d}{r["gemv_ms"]:11.3f}{r["local32_ms"]:12.3f}'
                  f'{sib}{hdrm}{ratio}{exact:>8s}')
        out.append(dict(tensor=tensor.name, n=n, k=k, rows=rows))
    if not out:
        print(f'no {args.quant} tensor has a local32 sibling (Q3_K has none)')
    if args.json:
        args.json.write_text(json.dumps(out, indent=2) + '\n')


if __name__ == '__main__':
    main()
