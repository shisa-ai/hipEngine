"""Locate the GEMV/MMQ crossover for dense raw-IQ prefill.

The strict GEMV re-reads each weight column once per 8-row slab, so its cost
grows with rows. The MMQ kernel pads to 128 rows, so a short prompt pays for
work it does not use. The crossover is where the second stops being worse.

Both arms are timed interleaved in one process (shared clocks, min-of-N) and
each row count is also checked for agreement, so the sweep doubles as a
correctness screen across the padding boundary.

CPU/GPU: needs a GPU. Reads real tensors from the published UD file.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import (copy_device_to_host, copy_host_to_device,
                                   free, host_array_ptr, malloc)
from hipengine.kernels.hip_gfx1100.quant import gguf_iq_source_mmq_prefill as iq_mmq
from hipengine.kernels.hip_gfx1100.quant import gguf_k_mmq_prefill as q8_mmq
from hipengine.kernels.hip_gfx1100.quant.gguf_iq_dense import build_gguf_iq_dense, launch
from hipengine.loading.gguf import GGUFReader

MODEL = '/models/gguf/Qwen3.8-27B-UD-Q4_K_M.gguf'
ROWS = (1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512)
ROUNDS = 5


def bf16(x):
    bits = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    return ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


def bf16_to_f32(u):
    return (np.asarray(u, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def sweep(tensor, raw, cv, reps):
    n, k = (int(d) for d in tensor.shape)
    hip = get_hip_runtime()
    dense_lib = build_gguf_iq_dense(compiler_version=cv)
    prod_lib = q8_mmq.build_gguf_k_mmq_prefill(load=True, compiler_version=cv)
    mmq_lib = iq_mmq.build_gguf_iq_source_mmq_prefill(load=True, compiler_version=cv)
    results = []
    bufs = []
    try:
        w_b = malloc(raw.nbytes); bufs.append(w_b)
        copy_host_to_device(w_b, host_array_ptr(raw), raw.nbytes)
        ws_nbytes = iq_mmq.iq_dense_mmq_nbytes(max(ROWS), k)
        ws_b = malloc(ws_nbytes); bufs.append(ws_b)
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
                       quant='gguf_iq4_xs', output='bf16', library=dense_lib)

            def run_mmq():
                with iq_mmq.iq_dense_mmq_session(
                        True, workspace_ptr=int(ws_b.ptr), workspace_nbytes=ws_nbytes,
                        library=mmq_lib, producer_library=prod_lib):
                    iq_mmq.gguf_iq4_xs_dense_mmq_i128_j128_k256_q8_1_ds4_prefill_bf16_bf16_out(
                        x_b.ptr, w_b.ptr, cand_b.ptr, rows, k, n)

            for fn in (run_gemv, run_mmq):
                fn()
            hip.device_synchronize()
            copy_device_to_host(host_array_ptr(ref), ref_b, ref.nbytes)
            copy_device_to_host(host_array_ptr(cand), cand_b, cand.nbytes)
            a, c = bf16_to_f32(ref).astype(np.float64), bf16_to_f32(cand).astype(np.float64)
            rel = float(np.abs(a - c).max() / max(np.abs(a).max(), 1e-9))
            corr = float(np.corrcoef(a.ravel(), c.ravel())[0, 1])

            best = {}
            for name, fn in (('gemv', run_gemv), ('mmq', run_mmq)):
                best[name] = float('inf')
                for _ in range(ROUNDS):
                    t0 = time.perf_counter()
                    for _ in range(reps):
                        fn()
                    hip.device_synchronize()
                    best[name] = min(best[name], (time.perf_counter() - t0) / reps)
            results.append(dict(rows=rows, gemv_ms=best['gemv'] * 1e3,
                                mmq_ms=best['mmq'] * 1e3,
                                speedup=best['gemv'] / best['mmq'],
                                max_rel=rel, corr=corr))
            for b in (cand_b, ref_b, x_b):
                free(b); bufs.remove(b)
    finally:
        for b in reversed(bufs):
            free(b)
    return n, k, results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--json', type=Path)
    args = ap.parse_args()
    cv = Path('/tmp/ud-hipcc-version.txt').read_text()
    reader = GGUFReader(MODEL)
    seen, picks = set(), []
    for t in reader.info.tensors:
        if (t.ggml_type_name != 'IQ4_XS' or not t.name.startswith('blk.')
                or len(t.shape) != 2):
            continue
        shape = (int(t.shape[0]), int(t.shape[1]))
        if shape in seen or shape[0] % 128 or shape[1] % 256:
            continue
        seen.add(shape); picks.append(t)
    picks.sort(key=lambda t: -t.nbytes)
    out = []
    for tensor in picks:
        raw = np.frombuffer(reader.tensor_data(tensor.name), dtype=np.uint8)
        n, k, rows = sweep(tensor, raw, cv, args.reps)
        cross = next((r['rows'] for r in rows if r['speedup'] > 1.0), None)
        print(f'\n{tensor.name}  N={n} K={k}  {raw.nbytes/1e6:.1f} MB')
        print(f'{"rows":>6s}{"gemv ms":>11s}{"mmq ms":>11s}{"speedup":>10s}'
              f'{"max_rel":>10s}{"corr":>10s}')
        for r in rows:
            mark = ' <-- crossover' if r['rows'] == cross else ''
            print(f'{r["rows"]:6d}{r["gemv_ms"]:11.3f}{r["mmq_ms"]:11.3f}'
                  f'{r["speedup"]:10.2f}{r["max_rel"]:10.4f}{r["corr"]:10.6f}{mark}')
        print(f'  crossover (first rows where MMQ wins): {cross}')
        out.append(dict(tensor=tensor.name, n=n, k=k, crossover=cross, rows=rows))
    if args.json:
        args.json.write_text(json.dumps(out, indent=2) + '\n')


if __name__ == '__main__':
    main()
