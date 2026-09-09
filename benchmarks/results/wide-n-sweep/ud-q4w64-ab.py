import sys, time
sys.path.insert(0, '/home/lhl/hipEngine-ud')
from pathlib import Path
import numpy as np
from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import copy_host_to_device, copy_device_to_host, free, host_array_ptr, malloc
from hipengine.kernels.hip_gfx1100.quant.gguf_k_t16_selected_prefill import (
    build_gguf_k_t16_selected_prefill, gguf_q4_k_t16_wmma_prefill_shared_b_bf16_bf16_out,
    gguf_q4_k_t16_wmma_prefill_shared_b_w64_bf16_bf16_out)
from hipengine.loading.gguf import GGUFReader
from hipengine.quant import gguf_t16

reader = GGUFReader('/models/gguf/Qwen3.8-27B-UD-Q4_K_M.gguf')
runtime = get_hip_runtime()
lib = build_gguf_k_t16_selected_prefill(load=True, compiler_version=Path('/tmp/ud-hipcc-version.txt').read_text())
rng = np.random.default_rng(1)
q4_names = [t.name for t in reader.info.tensors if t.ggml_type_name == 'Q4_K']
print(len(q4_names), "Q4_K tensors; repack fn:", [a for a in dir(gguf_t16) if 'q4' in a.lower()][:6])
from hipengine.quant.gguf_q4_k import repack_gguf_q4_k_tile16 as repack
seen = set()
for tname in q4_names:
    info = reader.tensor_info(tname)
    n, k = int(info.shape[0]), int(info.shape[1])
    if (n, k) in seen or repack is None: continue
    seen.add((n, k))
    tiles = repack(np.asarray(reader.tensor_data(tname))[None, ...]).tiles
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
        gguf_q4_k_t16_wmma_prefill_shared_b_bf16_bf16_out(x_dev.ptr, t_dev.ptr, outa.ptr, rows, k, n, library=lib, runtime=runtime)
        gguf_q4_k_t16_wmma_prefill_shared_b_w64_bf16_bf16_out(x_dev.ptr, t_dev.ptr, outb.ptr, rows, k, n, library=lib, runtime=runtime)
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(a), outa, a.nbytes, runtime=runtime)
        copy_device_to_host(host_array_ptr(b), outb, b.nbytes, runtime=runtime)
        ts = {}
        for name, fn in (("shb48", gguf_q4_k_t16_wmma_prefill_shared_b_bf16_bf16_out),
                         ("shb64", gguf_q4_k_t16_wmma_prefill_shared_b_w64_bf16_bf16_out)):
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
        print(f"Q4 {tname} ({n}x{k}) rows={rows}: mism={int(np.count_nonzero(a!=b))}  "
              f"shb48={ts['shb48']*1e3:.2f}ms shb64={ts['shb64']*1e3:.2f}ms ({ts['shb48']/ts['shb64']:.2f}x) [{2*rows*n*k/ts['shb64']/1e12:.1f} TF]")
    finally:
        for b2 in reversed(bufs): free(b2, runtime=runtime)
