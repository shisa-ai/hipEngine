import sys, time
sys.path.insert(0, '/home/lhl/hipEngine-ud')
from pathlib import Path
import numpy as np
from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import copy_host_to_device, copy_device_to_host, free, host_array_ptr, malloc
from hipengine.kernels.hip_gfx1100.quant.gguf_k_t16_selected_prefill import (
    build_gguf_k_t16_selected_prefill, gguf_q5_k_t16_wmma_prefill_shared8r2_bf16_bf16_out,
    gguf_q5_k_t16_wmma_prefill_shared8r2w64_bf16_bf16_out)
from hipengine.quant.gguf_t16 import repack_gguf_q5_k_tile16
from hipengine.loading.gguf import GGUFReader

reader = GGUFReader('/models/gguf/Qwen3.8-27B-UD-Q4_K_M.gguf')
runtime = get_hip_runtime()
lib = build_gguf_k_t16_selected_prefill(load=True, compiler_version=Path('/tmp/ud-hipcc-version.txt').read_text())
rng = np.random.default_rng(1)
q5_names = [t.name for t in reader.info.tensors if t.ggml_type_name == 'Q5_K']
seen = set()
for tname in q5_names:
    info = reader.tensor_info(tname)
    n, k = int(info.shape[0]), int(info.shape[1])
    if (n, k) in seen: continue
    seen.add((n, k))
    tiles = repack_gguf_q5_k_tile16(np.asarray(reader.tensor_data(tname))[None, ...]).tiles
    bufs = []
    try:
        t_dev = malloc(tiles.nbytes, runtime=runtime); bufs.append(t_dev)
        copy_host_to_device(t_dev, host_array_ptr(tiles), runtime=runtime)
        rows = 512
        x_bits = ((rng.normal(0, 0.2, (rows, k)).astype(np.float32).view(np.uint32) + 0x7FFF) >> 16).astype(np.uint16)
        x_dev = malloc(x_bits.nbytes, runtime=runtime); bufs.append(x_dev)
        copy_host_to_device(x_dev, host_array_ptr(x_bits), runtime=runtime)
        outa = malloc(rows*n*2, runtime=runtime); bufs.append(outa)
        outb = malloc(rows*n*2, runtime=runtime); bufs.append(outb)
        a = np.zeros((rows, n), dtype=np.uint16); b = np.zeros_like(a)
        gguf_q5_k_t16_wmma_prefill_shared8r2_bf16_bf16_out(x_dev.ptr, t_dev.ptr, outa.ptr, rows, k, n, library=lib, runtime=runtime)
        gguf_q5_k_t16_wmma_prefill_shared8r2w64_bf16_bf16_out(x_dev.ptr, t_dev.ptr, outb.ptr, rows, k, n, library=lib, runtime=runtime)
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(a), outa, a.nbytes, runtime=runtime)
        copy_device_to_host(host_array_ptr(b), outb, b.nbytes, runtime=runtime)
        ts = {}
        for name, fn in (("8r2w32", gguf_q5_k_t16_wmma_prefill_shared8r2_bf16_bf16_out),
                         ("8r2w64", gguf_q5_k_t16_wmma_prefill_shared8r2w64_bf16_bf16_out)):
            def run():
                fn(x_dev.ptr, t_dev.ptr, outa.ptr, rows, k, n, library=lib, runtime=runtime)
            run(); runtime.device_synchronize()
            best = float('inf')
            for _ in range(3):
                t0 = time.perf_counter()
                for _ in range(5): run()
                runtime.device_synchronize()
                best = min(best, (time.perf_counter() - t0) / 5)
            ts[name] = best
        print(f"Q5 {tname} ({n}x{k}) rows={rows}: mism={int(np.count_nonzero(a!=b))}  "
              f"8r2w32={ts['8r2w32']*1e3:.2f}ms 8r2w64={ts['8r2w64']*1e3:.2f}ms ({ts['8r2w32']/ts['8r2w64']:.2f}x) [{2*rows*n*k/ts['8r2w64']/1e12:.1f} TF]")
    finally:
        for b2 in reversed(bufs): free(b2, runtime=runtime)
