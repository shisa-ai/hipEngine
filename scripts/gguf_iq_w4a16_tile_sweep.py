"""Sweep the W4A16 dense-IQ prefill tile config on this gfx1100 host.

The retained 16x128 tile was swept on gfx1151 (IQ4_XS, 512 rows) only; this
re-sweeps the grid on the current card via the HIPENGINE_IQ_WMMA_TILE_M/N
build-cache knobs, timing the kernel at the published-protocol 512-row
shapes on real tensors from the UD artifact.

CPU/GPU: needs a GPU. Reads real tensors from the published UD file.
"""
from __future__ import annotations

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
from hipengine.kernels.hip_gfx1100.quant import gguf_iq_wmma_prefill as w4a16
from hipengine.loading.gguf import GGUFReader

MODEL = os.environ.get('SWEEP_MODEL', '/models/gguf/Qwen3.8-27B-UD-Q4_K_M.gguf')
QUANT = 'gguf_iq4_xs'
ROUNDS = 5
REPS = 5
# (TILE_M, TILE_N): both multiples of 16; the kernel maps them to TM/TN
# 16x16 WMMA fragments. The gfx1151 sweep's retained point is 16x128.
TILES = [tuple(int(v) for v in t.split('x')) for t in
         os.environ.get('HIPENGINE_TILE_SWEEP', '16x64,32x64,16x128').split(',')]


def bf16(x):
    bits = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    return ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rows', type=int, default=512)
    ap.add_argument('--json', type=Path)
    args = ap.parse_args()
    cv = Path('/tmp/ud-hipcc-version.txt').read_text()
    reader = GGUFReader(MODEL)
    tensors = [t for t in reader.info.tensors
               if t.ggml_type_name == 'IQ4_XS' and t.name.startswith('blk.')
               and len(t.shape) == 2]
    seen, picks = set(), []
    for t in sorted(tensors, key=lambda t: -t.nbytes):
        s = (int(t.shape[0]), int(t.shape[1]))
        if s in seen:
            continue
        seen.add(s); picks.append(t)
    picks = picks[:3]  # the wide/mid/narrow real shapes
    raws = {t.name: np.frombuffer(reader.tensor_data(t.name), dtype=np.uint8)
            for t in picks}

    hip = get_hip_runtime()
    out = []
    for tile_m, tile_n in TILES:
        os.environ['HIPENGINE_IQ_WMMA_TILE_M'] = str(tile_m)
        os.environ['HIPENGINE_IQ_WMMA_TILE_N'] = str(tile_n)
        try:
            lib = w4a16.build_gguf_iq_wmma_prefill(load=True, compiler_version=cv)
        except Exception as e:  # build failure (e.g. register pressure)
            print(f'{tile_m}x{tile_n}: build failed: {str(e)[:80]}')
            out.append(dict(tile=[tile_m, tile_n], failed=True))
            continue
        row = dict(tile=[tile_m, tile_n], shapes={})
        for t in picks:
            n, k = int(t.shape[0]), int(t.shape[1])
            raw = raws[t.name]
            x = bf16(np.random.default_rng(11).normal(0, 0.1, (args.rows, k)))
            o = np.zeros((args.rows, n), dtype=np.uint16)
            bufs = []
            try:
                w_b = malloc(raw.nbytes); bufs.append(w_b)
                copy_host_to_device(w_b, host_array_ptr(raw), raw.nbytes)
                x_b = malloc(x.nbytes); bufs.append(x_b)
                copy_host_to_device(x_b, host_array_ptr(x), x.nbytes)
                o_b = malloc(o.nbytes); bufs.append(o_b)

                def run():
                    w4a16.launch(x_b.ptr, w_b.ptr, o_b.ptr, args.rows, k, n,
                                 quant=QUANT, library=lib)
                run(); hip.device_synchronize()
                best = float('inf')
                for _ in range(ROUNDS):
                    t0 = time.perf_counter()
                    for _ in range(REPS):
                        run()
                    hip.device_synchronize()
                    best = min(best, (time.perf_counter() - t0) / REPS)
                row['shapes'][f'{n}x{k}'] = round(best * 1e3, 3)
            finally:
                for b in reversed(bufs):
                    free(b)
        total = sum(row['shapes'].values())
        row['total_ms'] = round(total, 3)
        out.append(row)
        print(f'{tile_m:>3d}x{tile_n:<4d} total {total:8.2f} ms  ' +
              '  '.join(f'{s}={v:7.2f}' for s, v in row['shapes'].items()))
    ok = [r for r in out if not r.get('failed')]
    if ok:
        best = min(ok, key=lambda r: r['total_ms'])
        print(f"\nbest: {best['tile'][0]}x{best['tile'][1]} "
              f"(total {best['total_ms']:.2f} ms; retained 16x128 = "
              f"{next((r['total_ms'] for r in ok if r['tile'] == [16, 128]), float('nan')):.2f} ms)")
    if args.json:
        args.json.write_text(json.dumps(
            dict(rows=args.rows, model=MODEL, tiles=out), indent=2) + '\n')


if __name__ == '__main__':
    main()
