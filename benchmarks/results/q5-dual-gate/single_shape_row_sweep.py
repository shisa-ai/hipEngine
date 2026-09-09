import sys, time
sys.path.insert(0, '/home/lhl/hipEngine-ud')
from pathlib import Path
import numpy as np
from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import copy_host_to_device, free, host_array_ptr, malloc
from hipengine.kernels.hip_gfx1100.quant.gguf_k_t16_selected_prefill import (
    build_gguf_k_t16_selected_prefill, gguf_q5_k_t16_wmma_prefill_bf16_bf16_out,
    gguf_q5_k_t16_wmma_prefill_shared8r2_bf16_bf16_out,
    gguf_q5_k_t16_wmma_prefill_shared8r3_bf16_bf16_out,
    gguf_q5_k_t16_wmma_prefill_shared8r4_bf16_bf16_out)
from hipengine.loading.gguf import GGUFReader
from hipengine.quant.gguf_t16 import repack_gguf_q5_k_tile16

reader = GGUFReader('/models/gguf/Qwen3.8-27B-UD-Q4_K_M.gguf')
FNS = {
    'plain': gguf_q5_k_t16_wmma_prefill_bf16_bf16_out,
    '8r2': gguf_q5_k_t16_wmma_prefill_shared8r2_bf16_bf16_out,
    '8r3': gguf_q5_k_t16_wmma_prefill_shared8r3_bf16_bf16_out,
    '8r4': gguf_q5_k_t16_wmma_prefill_shared8r4_bf16_bf16_out,
}
runtime = get_hip_runtime()
lib = build_gguf_k_t16_selected_prefill(load=True, compiler_version=Path('/tmp/ud-hipcc-version.txt').read_text())
rng = np.random.default_rng(1)
for tname in ('blk.25.ffn_gate.weight', 'blk.30.ssm_out.weight', 'blk.0.attn_qkv.weight'):
    info = reader.tensor_info(tname)
    n, k = int(info.shape[0]), int(info.shape[1])
    tiles = repack_gguf_q5_k_tile16(np.asarray(reader.tensor_data(tname))[None, ...]).tiles
    bufs = []
    try:
        t_dev = malloc(tiles.nbytes, runtime=runtime); bufs.append(t_dev)
        copy_host_to_device(t_dev, host_array_ptr(tiles), runtime=runtime)
        for rows in (256, 384, 512):
            x_bits = ((rng.normal(0, 0.2, (rows, k)).astype(np.float32).view(np.uint32) + 0x7FFF) >> 16).astype(np.uint16)
            x_dev = malloc(x_bits.nbytes, runtime=runtime); bufs.append(x_dev)
            copy_host_to_device(x_dev, host_array_ptr(x_bits), runtime=runtime)
            out = malloc(rows*n*2, runtime=runtime); bufs.append(out)
            times = {}
            for name, fn in FNS.items():
                def run():
                    fn(x_dev.ptr, t_dev.ptr, out.ptr, rows, k, n, library=lib, runtime=runtime)
                run(); runtime.device_synchronize()
                best = float('inf')
                for _ in range(3):
                    t0 = time.perf_counter()
                    for _ in range(5): run()
                    runtime.device_synchronize()
                    best = min(best, (time.perf_counter() - t0) / 5)
                times[name] = best
            base = times['plain']
            print(f"{tname} ({n}x{k}) rows={rows}: " + "  ".join(
                f"{nm}={t*1e3:5.2f}ms({t/base:.2f}x)" for nm, t in times.items()))
    finally:
        for b in reversed(bufs): free(b, runtime=runtime)
