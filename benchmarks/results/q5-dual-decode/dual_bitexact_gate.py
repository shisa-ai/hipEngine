import sys, time
sys.path.insert(0, '/home/lhl/hipEngine-ud')
from pathlib import Path
import numpy as np
from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc
from hipengine.kernels.hip_gfx1100.fused.paro_silu import build_paro_silu, silu_mul_separate_out_bf16
from hipengine.kernels.hip_gfx1100.quant import gguf_t16_selected_gemv as t16g
from hipengine.loading.gguf import GGUFReader
from hipengine.quant.gguf_t16 import repack_gguf_q5_k_tile16

reader = GGUFReader('/models/gguf/Qwen3.8-27B-UD-Q4_K_M.gguf')
tiles_a = repack_gguf_q5_k_tile16(np.asarray(reader.tensor_data('blk.25.ffn_gate.weight'))[None, ...]).tiles
tiles_b = repack_gguf_q5_k_tile16(np.asarray(reader.tensor_data('blk.25.ffn_up.weight'))[None, ...]).tiles
n, k = 17408, 5120
runtime = get_hip_runtime()
lib = t16g.build_gguf_t16_selected_gemv(load=True, compiler_version=Path('/tmp/ud-hipcc-version.txt').read_text())
silu_lib = build_paro_silu(load=True)
rng = np.random.default_rng(11)
x_bits = ((rng.normal(0, 0.2, (1, k)).astype(np.float32).view(np.uint32) + 0x7FFF) >> 16).astype(np.uint16)
exp = np.zeros((1, n), dtype=np.uint16); act = np.zeros_like(exp)
bufs = []
try:
    def dev(a):
        b = malloc(a.nbytes, runtime=runtime); bufs.append(b)
        copy_host_to_device(b, host_array_ptr(a), runtime=runtime); return b
    x_dev, a_dev, b_dev = dev(x_bits), dev(tiles_a), dev(tiles_b)
    o_g, o_u, o_c = dev(exp), dev(exp), dev(act)
    t16g.gguf_q5_k_t16_gemv_decode_bf16_bf16_out(x_dev.ptr, a_dev.ptr, o_g.ptr, 1, k, n, library=lib, runtime=runtime)
    t16g.gguf_q5_k_t16_gemv_decode_bf16_bf16_out(x_dev.ptr, b_dev.ptr, o_u.ptr, 1, k, n, library=lib, runtime=runtime)
    silu_mul_separate_out_bf16(o_g.ptr, o_u.ptr, o_c.ptr, 1, n, library=silu_lib, runtime=runtime)
    t16g.gguf_q5_k_t16_dense_dual_silu_gemv_bf16_bf16_out(x_dev.ptr, a_dev.ptr, b_dev.ptr, o_c.ptr if False else o_g.ptr, 1, k, n, library=lib, runtime=runtime)
    runtime.device_synchronize()
    copy_device_to_host(host_array_ptr(exp), o_c, exp.nbytes, runtime=runtime)
    # redo dual into separate buffer
    t16g.gguf_q5_k_t16_dense_dual_silu_gemv_bf16_bf16_out(x_dev.ptr, a_dev.ptr, b_dev.ptr, dev(act).ptr, 1, k, n, library=lib, runtime=runtime)
    runtime.device_synchronize()
    copy_device_to_host(host_array_ptr(act), bufs[-1], act.nbytes, runtime=runtime)
    print("bit-exact:", int(np.count_nonzero(act != exp)) == 0, " mismatches:", int(np.count_nonzero(act != exp)))
    # timing
    def singles():
        t16g.gguf_q5_k_t16_gemv_decode_bf16_bf16_out(x_dev.ptr, a_dev.ptr, o_g.ptr, 1, k, n, library=lib, runtime=runtime)
        t16g.gguf_q5_k_t16_gemv_decode_bf16_bf16_out(x_dev.ptr, b_dev.ptr, o_u.ptr, 1, k, n, library=lib, runtime=runtime)
        silu_mul_separate_out_bf16(o_g.ptr, o_u.ptr, o_c.ptr, 1, n, library=silu_lib, runtime=runtime)
    def dual():
        t16g.gguf_q5_k_t16_dense_dual_silu_gemv_bf16_bf16_out(x_dev.ptr, a_dev.ptr, b_dev.ptr, o_g.ptr, 1, k, n, library=lib, runtime=runtime)
    for name, fn in (("2x single + silu", singles), ("dual fused-SiLU", dual)):
        fn(); runtime.device_synchronize()
        best = float('inf')
        for _ in range(3):
            t0 = time.perf_counter()
            for _ in range(20): fn()
            runtime.device_synchronize()
            best = min(best, (time.perf_counter() - t0) / 20)
        print(f"{name:<18} {best*1e6:7.1f} us")
finally:
    for b in reversed(bufs): free(b, runtime=runtime)
