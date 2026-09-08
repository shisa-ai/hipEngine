"""De-risk item 3: run the existing IQ4_XS integer-MMQ prefill kernel on a real
DENSE tensor, as the degenerate single-expert case.

The kernel's weight addressing is
    qweight + expert*expert_bytes + out_row*weight_row_bytes + block*block_bytes
with expert_bytes = out_features*weight_row_bytes, so at expert=0 it is exactly
the dense raw GGUF layout - no repack, no sidecar. This probe checks that claim
against the trusted strict dense GEMV and times both.
"""
import time
from pathlib import Path

import numpy as np

from hipengine.core.memory import (copy_device_to_host, copy_host_to_device,
                                   free, host_array_ptr, malloc)
from hipengine.kernels.hip_gfx1100.quant import gguf_iq_source_mmq_prefill as iq_mmq
from hipengine.kernels.hip_gfx1100.quant import gguf_k_mmq_prefill as q8_mmq
from hipengine.kernels.hip_gfx1100.quant.gguf_iq_dense import build_gguf_iq_dense, launch
from hipengine.loading.gguf import GGUFReader

MODEL = '/models/gguf/Qwen3.8-27B-UD-Q4_K_M.gguf'
ROWS = 512
CV = Path('/tmp/ud-hipcc-version.txt').read_text()


def bf16(x):
    bits = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    return ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


def bf16_to_f32(u):
    return (np.asarray(u, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def main():
    reader = GGUFReader(MODEL)
    tensor = max((t for t in reader.info.tensors
                  if t.ggml_type_name == 'IQ4_XS' and t.name.startswith('blk.')
                  and len(t.shape) == 2 and int(t.shape[0]) % 128 == 0
                  and int(t.shape[1]) % 256 == 0),
                 key=lambda t: t.nbytes)
    n, k = (int(d) for d in tensor.shape)
    raw = np.frombuffer(reader.tensor_data(tensor.name), dtype=np.uint8)
    print(f'{tensor.name}  N={n} K={k}  {raw.nbytes/1e6:.1f} MB  rows={ROWS}')

    x = bf16(np.random.default_rng(11).normal(0, 0.1, (ROWS, k)))
    meta = iq_mmq.build_iq_source_mmq128_metadata([ROWS])
    compact_start = np.array([0, ROWS], dtype=np.int64)
    print(f'single-expert metadata: mmq_total_rows={meta.mmq_total_rows} '
          f'tiles={len(meta.tile_expert)} expert_start_mmq={meta.expert_start_mmq.tolist()}')

    dense_lib = build_gguf_iq_dense(compiler_version=CV)
    prod_lib = q8_mmq.build_gguf_k_mmq_prefill(load=True, compiler_version=CV)
    mmq_lib = iq_mmq.build_gguf_iq_source_mmq_prefill(load=True, compiler_version=CV)

    packed_nbytes = q8_mmq.q8_1_ds4_kmajor_nbytes(ROWS, k)
    ref = np.zeros((ROWS, n), dtype=np.uint16)
    cand = np.zeros((ROWS, n), dtype=np.uint16)
    bufs = []
    try:
        def dev(a):
            b = malloc(a.nbytes); bufs.append(b)
            copy_host_to_device(b, host_array_ptr(a), a.nbytes); return b
        x_b, w_b = dev(x), dev(raw)
        cs_b, ms_b, te_b = dev(compact_start), dev(meta.expert_start_mmq), dev(meta.tile_expert)
        packed_b = malloc(packed_nbytes); bufs.append(packed_b)
        ref_b = malloc(ref.nbytes); bufs.append(ref_b)
        cand_b = malloc(cand.nbytes); bufs.append(cand_b)

        from hipengine.core.hip import get_hip_runtime
        hip = get_hip_runtime()

        # --- reference: trusted strict dense GEMV (row tile) ---
        def run_ref():
            launch(x_b.ptr, w_b.ptr, ref_b.ptr, ROWS, k, n,
                   quant='gguf_iq4_xs', output='bf16', library=dense_lib)
        run_ref(); hip.device_synchronize()
        t0 = time.perf_counter(); run_ref(); hip.device_synchronize()
        ref_ms = (time.perf_counter() - t0) * 1e3
        copy_device_to_host(host_array_ptr(ref), ref_b, ref.nbytes)

        # --- candidate: existing MMQ kernel, single expert ---
        def run_cand():
            q8_mmq.gguf_q8_1_ds4_quantize_bf16_kmajor(
                x_b.ptr, packed_b.ptr, ROWS, k, library=prod_lib)
            iq_mmq.gguf_iq4_xs_selected_mmq_i128_j128_k256_q8_1_ds4_prefill_compact_bf16_bf16_out(
                packed_b.ptr, cs_b.ptr, ms_b.ptr, te_b.ptr, w_b.ptr, cand_b.ptr,
                compact_rows=ROWS, in_features=k, out_features=n,
                num_experts=1, mmq_total_rows=meta.mmq_total_rows,
                library=mmq_lib)
        run_cand(); hip.device_synchronize()
        t0 = time.perf_counter(); run_cand(); hip.device_synchronize()
        cand_ms = (time.perf_counter() - t0) * 1e3
        copy_device_to_host(host_array_ptr(cand), cand_b, cand.nbytes)
    finally:
        for b in reversed(bufs):
            free(b)

    a, c = bf16_to_f32(ref).astype(np.float64), bf16_to_f32(cand).astype(np.float64)
    finite = np.isfinite(c).all()
    denom = np.maximum(np.abs(a).max(), 1e-9)
    rel = np.abs(a - c).max() / denom
    corr = float(np.corrcoef(a.ravel(), c.ravel())[0, 1]) if finite else float('nan')
    print(f'\nfinite={finite}  max|ref-cand|/max|ref| = {rel:.4f}  corr = {corr:.6f}')
    print(f'strict GEMV (R=8):  {ref_ms:8.2f} ms')
    print(f'MMQ single-expert:  {cand_ms:8.2f} ms   -> {ref_ms/cand_ms:5.2f}x')
    # The GEMV re-reads the weight column once per row slab, so its *traffic*
    # is rows/R times the resident bytes; the MMQ reads them once.
    slabs = (ROWS + 7) // 8
    print(f'weight traffic: GEMV ~{slabs}x resident '
          f'({raw.nbytes*slabs/(ref_ms/1e3)/1e9:.0f} GB/s effective), '
          f'MMQ 1x ({raw.nbytes/(cand_ms/1e3)/1e9:.0f} GB/s)')


if __name__ == '__main__':
    main()
