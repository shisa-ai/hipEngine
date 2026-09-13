import sys, time
sys.path.insert(0, '/home/lhl/hipEngine-ud')
from pathlib import Path
import numpy as np
from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc
from hipengine.kernels.hip_gfx1100.quant.gguf_k_t16_selected_prefill import (
    build_gguf_k_t16_selected_prefill, gguf_q4_k_t16_wmma_prefill_shared_b_w64_bf16_bf16_out,
    gguf_q4_k_t16_dense_dual_wmma_prefill_silu_bf16_bf16_out)
from hipengine.loading.gguf import GGUFReader
from hipengine.quant.gguf_q4_k import repack_gguf_q4_k_tile16

reader = GGUFReader('/models/gguf/Qwen3.8-27B-Q4_K_M.gguf')
runtime = get_hip_runtime()
lib = build_gguf_k_t16_selected_prefill(load=True, compiler_version=Path('/tmp/ud-hipcc-version.txt').read_text())
rng = np.random.default_rng(3)
for gate in ('blk.1.ffn_gate.weight', 'blk.10.ffn_gate.weight'):
    up = gate.replace('gate', 'up')
    info = reader.tensor_info(gate)
    n, k = int(info.shape[0]), int(info.shape[1])
    ta = repack_gguf_q4_k_tile16(np.asarray(reader.tensor_data(gate))[None, ...]).tiles
    tb = repack_gguf_q4_k_tile16(np.asarray(reader.tensor_data(up))[None, ...]).tiles
    bufs = []
    try:
        def dev(arr):
            d = malloc(arr.nbytes, runtime=runtime); bufs.append(d)
            copy_host_to_device(d, host_array_ptr(arr), runtime=runtime); return d
        tda, tdb = dev(ta), dev(tb)
        rows = 512
        x_bits = ((rng.normal(0, 0.2, (rows, k)).astype(np.float32).view(np.uint32) + 0x7FFF) >> 16).astype(np.uint16)
        x_dev = dev(x_bits)
        o1 = dev(np.zeros((rows, n), dtype=np.uint16)); o2 = dev(np.zeros((rows, n), dtype=np.uint16)); o3 = dev(np.zeros((rows, n), dtype=np.uint16))
        ts = {}
        for name, fn, args in (("2x w64 single", None, None), ("q4 dual", gguf_q4_k_t16_dense_dual_wmma_prefill_silu_bf16_bf16_out, None)):
            if name.startswith("2x"):
                def run():
                    gguf_q4_k_t16_wmma_prefill_shared_b_w64_bf16_bf16_out(x_dev.ptr, tda.ptr, o1.ptr, rows, k, n, library=lib, runtime=runtime)
                    gguf_q4_k_t16_wmma_prefill_shared_b_w64_bf16_bf16_out(x_dev.ptr, tdb.ptr, o2.ptr, rows, k, n, library=lib, runtime=runtime)
            else:
                def run():
                    fn(x_dev.ptr, tda.ptr, tdb.ptr, o3.ptr, rows, k, n, library=lib, runtime=runtime)
            run(); runtime.device_synchronize()
            best = float('inf')
            for _ in range(3):
                t0 = time.perf_counter()
                for _ in range(5): run()
                runtime.device_synchronize()
                best = min(best, (time.perf_counter() - t0) / 5)
            ts[name] = best
        print(f"{gate} ({n}x{k}) rows={rows}: 2x-single={ts['2x w64 single']*1e3:.2f}ms  dual={ts['q4 dual']*1e3:.2f}ms ({ts['2x w64 single']/ts['q4 dual']:.2f}x)")
    finally:
        for b in reversed(bufs): free(b, runtime=runtime)
