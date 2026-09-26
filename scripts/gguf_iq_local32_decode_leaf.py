#!/usr/bin/env python3
"""E2b′ leaf timing: local32 owners against the strict GEMV, real UD shapes.

Campaign UD-GFX1151-OPTIMIZE2 E2b′ measures the per-launch factor the
pre-registered 0.98-1.02x paired prediction depends on: every local32 owner
is timed against the strict per-row GEMV on every real dense-IQ shape the
UD-Q4_K_M artifact carries, on this host's GPU, in interleaved min-of-N
rounds over the same device buffers.

Arms (singles, rows 1/2/4):
  gemv      the strict per-row GEMV. Q3_K has no local32 owner, so its
            shapes are timed gemv-only for the family-share table.
  local32   the rows==1 local32 decode owner, launched once per row at
            rows>1.
  sibling   the rows 2-4 local32 verifier sibling (one launch, ROWS rows).

Arms (real IQ4_XS/IQ4_XS gate/up pair, rows==1):
  strict    2x strict GEMV + silu_mul (the plain-control style chain)
  chain     2x local32 + silu_mul (what the pair runs without the dual)
  dual      the fused local32 dual SiLU owner; checked bit-exact vs chain.

CPU/GPU: needs a GPU. Reads real tensors from the published UD file.
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
from hipengine.kernels.hip_gfx1100.fused.paro_silu import (
    silu_mul_separate_out_bf16)
from hipengine.kernels.hip_gfx1100.quant.gguf_iq_dense import (
    build_gguf_iq_dense, launch, launch_local32, launch_local32_dual_silu,
    launch_local32_rows)
from hipengine.loading.gguf import GGUFReader

ROWS = (1, 2, 4)
ROUNDS = 7
LOCAL32_QUANTS = ('gguf_iq4_xs', 'gguf_iq4_nl', 'gguf_iq3_s', 'gguf_iq3_xxs',
                  'gguf_iq2_s', 'gguf_iq2_xs')
# Dense-IQ types the artifact may carry; Q3_K stays gemv-only.
DENSE_IQ_TYPE_NAMES = ('IQ4_XS', 'IQ4_NL', 'IQ3_S', 'IQ3_XXS', 'IQ2_S',
                       'IQ2_XS', 'Q3_K')


def bf16(x):
    bits = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    return ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


def bf16_to_f32(u):
    return (np.asarray(u, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def timed(arms, reps, rounds=ROUNDS):
    """Interleaved min-of-N: round-robin over arms, reps launches per block."""
    hip = get_hip_runtime()
    for _, fn in arms:
        fn()
        hip.device_synchronize()
    best = {name: float('inf') for name, _ in arms}
    for _ in range(rounds):
        for name, fn in arms:
            t0 = time.perf_counter()
            for _ in range(reps):
                fn()
            hip.device_synchronize()
            best[name] = min(best[name], (time.perf_counter() - t0) / reps)
    return best


def sweep_singles(tensor, raw, cv, reps, quant):
    n, k = (int(d) for d in tensor.shape)
    qkey = 'gguf_' + quant.lower()
    lib = build_gguf_iq_dense(compiler_version=cv)
    can_local32 = qkey in LOCAL32_QUANTS and not (k % 256 or n % 8)
    bufs, results = [], []

    def alloc(arr):
        b = malloc(arr.nbytes)
        bufs.append(b)
        return b

    try:
        w_b = alloc(raw)
        copy_host_to_device(w_b, host_array_ptr(raw), raw.nbytes)
        for rows in ROWS:
            x = bf16(np.random.default_rng(11).normal(0, 0.1, (rows, k)))
            x_b = alloc(x)
            copy_host_to_device(x_b, host_array_ptr(x), x.nbytes)
            ptrs, hosts = {}, {}
            for name in ('ref', 'cand', 'sib'):
                arr = np.zeros((rows, n), dtype=np.uint16)
                hosts[name] = arr
                ptrs[name] = alloc(arr)

            def run_gemv(b=x_b, w=w_b, o=ptrs['ref'], r=rows):
                launch(b.ptr, w.ptr, o.ptr, r, k, n,
                       quant=qkey, output='bf16', library=lib)

            def run_local32(b=x_b, w=w_b, o=ptrs['cand'], r=rows):
                for i in range(r):
                    launch_local32(b.ptr + i * k * 2, w.ptr,
                                   o.ptr + i * n * 2, 1, k, n,
                                   quant=qkey, output='bf16', library=lib)

            def run_sibling(b=x_b, w=w_b, o=ptrs['sib'], r=rows):
                if r == 1:
                    return
                launch_local32_rows(b.ptr, w.ptr, o.ptr, r, k, n,
                                    quant=qkey, output='bf16', library=lib)

            arms = [('gemv', run_gemv)]
            if can_local32:
                arms.append(('local32', run_local32))
                if rows > 1:
                    arms.append(('sibling', run_sibling))
            best = timed(arms, reps)

            for name in ('ref', 'cand', 'sib'):
                copy_device_to_host(host_array_ptr(hosts[name]), ptrs[name],
                                    hosts[name].nbytes)
            a = bf16_to_f32(hosts['ref']).astype(np.float64)
            c = bf16_to_f32(hosts['cand']).astype(np.float64)
            rel = float(np.abs(a - c).max() / max(np.abs(a).max(), 1e-9))
            corr = float(np.corrcoef(a.ravel(), c.ravel())[0, 1]) \
                if can_local32 else None
            bit_exact = bool((hosts['sib'] == hosts['cand']).all()) \
                if can_local32 and rows > 1 else None
            sib_ms = best.get('sibling')
            results.append(dict(
                rows=rows,
                gemv_ms=best['gemv'] * 1e3,
                local32_ms=None if 'local32' not in best
                else best['local32'] * 1e3,
                sibling_ms=None if sib_ms is None else sib_ms * 1e3,
                factor=None if 'local32' not in best
                else best['gemv'] / best['local32'],
                sibling_factor=None if sib_ms is None
                else best['gemv'] / sib_ms,
                max_rel=rel if can_local32 else None,
                corr=corr, sibling_bit_exact=bit_exact))
    finally:
        for b in reversed(bufs):
            free(b)
    return n, k, results, can_local32


def sweep_pair(gate_raw, up_raw, in_features, out_features, cv, reps):
    """rows==1 gate/up: strict chain vs local32 chain vs fused dual."""
    lib = build_gguf_iq_dense(compiler_version=cv)
    quant = 'gguf_iq4_xs'
    bufs = []

    def alloc(nbytes):
        b = malloc(nbytes)
        bufs.append(b)
        return b

    try:
        w_g = alloc(gate_raw.nbytes)
        w_u = alloc(up_raw.nbytes)
        copy_host_to_device(w_g, host_array_ptr(gate_raw), gate_raw.nbytes)
        copy_host_to_device(w_u, host_array_ptr(up_raw), up_raw.nbytes)
        x = bf16(np.random.default_rng(11).normal(0, 0.1,
                                                  (1, in_features)))
        x_b = alloc(x.nbytes)
        copy_host_to_device(x_b, host_array_ptr(x), x.nbytes)
        gate_buf = alloc(out_features * 2)
        up_buf = alloc(out_features * 2)
        outs = {name: alloc(out_features * 2)
                for name in ('strict', 'chain', 'dual')}

        def run_strict():
            launch(x_b.ptr, w_g.ptr, gate_buf.ptr, 1, in_features,
                   out_features, quant=quant, output='bf16', library=lib)
            launch(x_b.ptr, w_u.ptr, up_buf.ptr, 1, in_features,
                   out_features, quant=quant, output='bf16', library=lib)
            silu_mul_separate_out_bf16(gate_buf.ptr, up_buf.ptr,
                                       outs['strict'].ptr, 1, out_features)

        def run_chain():
            launch_local32(x_b.ptr, w_g.ptr, gate_buf.ptr, 1, in_features,
                           out_features, quant=quant, output='bf16',
                           library=lib)
            launch_local32(x_b.ptr, w_u.ptr, up_buf.ptr, 1, in_features,
                           out_features, quant=quant, output='bf16',
                           library=lib)
            silu_mul_separate_out_bf16(gate_buf.ptr, up_buf.ptr,
                                       outs['chain'].ptr, 1, out_features)

        def run_dual():
            launch_local32_dual_silu(x_b.ptr, w_g.ptr, w_u.ptr,
                                     outs['dual'].ptr, 1, in_features,
                                     out_features, quant=quant,
                                     output='bf16', library=lib)

        best = timed([('strict', run_strict), ('chain', run_chain),
                      ('dual', run_dual)], reps)

        hosts = {name: np.zeros((1, out_features), dtype=np.uint16)
                 for name in ('strict', 'chain', 'dual')}
        for name, host in hosts.items():
            copy_device_to_host(host_array_ptr(host), outs[name],
                                host.nbytes)
        checks = {
            'dual_chain_bit_exact':
                bool((hosts['dual'] == hosts['chain']).all()),
            'chain_vs_strict_max_rel': float(
                np.abs(bf16_to_f32(hosts['strict']).astype(np.float64)
                       - bf16_to_f32(hosts['chain']).astype(np.float64)
                       ).max()
                / max(np.abs(bf16_to_f32(hosts['strict'])
                             .astype(np.float64)).max(), 1e-9)),
        }
        return dict(
            in_features=in_features, out_features=out_features,
            strict_ms=best['strict'] * 1e3, chain_ms=best['chain'] * 1e3,
            dual_ms=best['dual'] * 1e3,
            strict_over_dual=best['strict'] / best['dual'],
            chain_over_dual=best['chain'] / best['dual'], **checks)
    finally:
        for b in reversed(bufs):
            free(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', type=Path,
                    default=Path(os.environ.get(
                        'HEADROOM_MODEL',
                        '/models/gguf/Qwen3.8-27B-UD-Q4_K_M.gguf')))
    ap.add_argument('--reps', type=int, default=5)
    ap.add_argument('--json', type=Path)
    args = ap.parse_args()
    cv = Path(os.environ.get('UD_HIPCC_VERSION_FILE',
                             '/tmp/ud-hipcc-version.txt')).read_text()
    reader = GGUFReader(args.model)

    # One representative tensor per (quant, shape), keeping the first found.
    per_quants = {q: {} for q in DENSE_IQ_TYPE_NAMES}
    up_by_layer = {}
    for t in reader.info.tensors:
        if not t.name.startswith('blk.') or len(t.shape) != 2:
            continue
        if t.ggml_type_name in per_quants:
            shape = (int(t.shape[0]), int(t.shape[1]))
            per_quants[t.ggml_type_name].setdefault(shape, t)
        if t.name.endswith('.ffn_up.weight') and t.ggml_type_name == 'IQ4_XS':
            up_by_layer[t.name.removesuffix('.ffn_up.weight')] = t

    singles = []
    for q in DENSE_IQ_TYPE_NAMES:
        for shape, t in sorted(per_quants[q].items(),
                               key=lambda kv: (-kv[1].nbytes, kv[0])):
            raw = np.frombuffer(reader.tensor_data(t.name), dtype=np.uint8)
            n, k, rows, has_l32 = sweep_singles(t, raw, cv, args.reps, q)
            print(f'\n== {q:7s} {t.name:26s} N={n:6d} K={k:6d} '
                  f'local32={has_l32}')
            print(f'{"rows":>5s}{"gemv ms":>10s}{"local32 ms":>12s}'
                  f'{"sibling ms":>12s}{"factor":>8s}{"sib fctr":>9s}'
                  f'{"max_rel":>10s}{"bit-ex":>8s}')
            for r in rows:
                l32 = (f'{r["local32_ms"]:12.4f}' if r['local32_ms']
                       else '           -')
                fac = (f'{r["factor"]:8.2f}' if r['factor'] else '       -')
                sib = (f'{r["sibling_ms"]:12.4f}' if r['sibling_ms']
                       else '           -')
                sf = (f'{r["sibling_factor"]:9.2f}' if r['sibling_factor']
                      else '        -')
                rel = (f'{r["max_rel"]:10.2e}' if r['max_rel'] is not None
                       else '         -')
                be = (str(r['sibling_bit_exact'])
                      if r['sibling_bit_exact'] is not None else '-')
                print(f'{r["rows"]:5d}{r["gemv_ms"]:10.4f}{l32}{sib}{fac}{sf}'
                      f'{rel}{be:>8s}')
            singles.append(dict(quant=q, tensor=t.name, n=n, k=k,
                                local32=has_l32, rows=rows))

    # Pair arm: one real IQ4_XS/IQ4_XS gate/up pair at rows==1.
    pair = None
    pair_layer = None
    for layer, up_t in sorted(up_by_layer.items()):
        up_shape = (int(up_t.shape[0]), int(up_t.shape[1]))
        gate_t = next((t for t in reader.info.tensors
                       if t.name == f'{layer}.ffn_gate.weight'), None)
        if (gate_t is not None and gate_t.ggml_type_name == 'IQ4_XS'
                and (int(gate_t.shape[0]), int(gate_t.shape[1])) == up_shape):
            pair_layer = layer
            break
    if pair_layer is not None:
        up_t = up_by_layer[pair_layer]
        gate_t = next(t for t in reader.info.tensors
                      if t.name == f'{pair_layer}.ffn_gate.weight')
        g = np.frombuffer(reader.tensor_data(gate_t.name), dtype=np.uint8)
        u = np.frombuffer(reader.tensor_data(up_t.name), dtype=np.uint8)
        out_features, in_features = int(up_t.shape[0]), int(up_t.shape[1])
        pair = sweep_pair(g, u, in_features, out_features, cv, args.reps)
        pair['gate_tensor'] = gate_t.name
        pair['up_tensor'] = up_t.name
        print(f'\n== pair  {gate_t.name} + {up_t.name}  '
              f'K={in_features} N={out_features}')
        print(f'{"strict ms":>11s}{"chain ms":>10s}{"dual ms":>9s}'
              f'{"str/dual":>9s}{"chn/dual":>9s}{"dual==chain":>12s}'
              f'{"chn~str":>10s}')
        print(f'{pair["strict_ms"]:11.4f}{pair["chain_ms"]:10.4f}'
              f'{pair["dual_ms"]:9.4f}{pair["strict_over_dual"]:9.2f}'
              f'{pair["chain_over_dual"]:9.2f}'
              f'{str(pair["dual_chain_bit_exact"]):>12s}'
              f'{pair["chain_vs_strict_max_rel"]:10.2e}')
    else:
        print('no IQ4_XS/IQ4_XS gate/up pair found; pair arm skipped')

    factors = [r['factor'] for s in singles for r in s['rows']
               if r['factor'] and r['rows'] == 1]
    if factors:
        print(f'\nrows==1 gemv/local32 factor over {len(factors)} shapes: '
              f'min={min(factors):.2f} max={max(factors):.2f} '
              f'mean={sum(factors) / len(factors):.2f}')
    out = dict(singles=singles, pair=pair)
    if args.json:
        args.json.write_text(json.dumps(out, indent=2) + '\n')
        print(f'wrote {args.json}')


if __name__ == '__main__':
    main()