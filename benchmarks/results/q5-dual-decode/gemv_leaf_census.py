import sys, time
sys.path.insert(0, '/home/lhl/hipEngine-ud')
from pathlib import Path
import numpy as np
from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import copy_host_to_device, free, host_array_ptr, malloc
from hipengine.kernels.hip_gfx1100.quant import gguf_t16_selected_gemv as t16g
from hipengine.loading.gguf import GGUFReader
from hipengine.quant.gguf_t16 import repack_gguf_q5_k_tile16

reader = GGUFReader('/models/gguf/Qwen3.8-27B-UD-Q4_K_M.gguf')
runtime = get_hip_runtime()
lib = t16g.build_gguf_t16_selected_gemv(load=True, compiler_version=Path('/tmp/ud-hipcc-version.txt').read_text())
rng = np.random.default_rng(3)
total = 0.0
counts = {'ffn_gate/up (17408x5120)': 28, 'ffn_down (5120x17408)': 22, 'ssm_out (5120x6144)': 33,
          'attn_qkv (10240x5120)': 12, 'attn_gate (6144x5120)': 12, 'attn_output (5120x6144)': 11,
          'attn_q (12288x5120)': 2, 'attn_k/v (1024x5120)': 11}
shapes = {'ffn_gate/up (17408x5120)': 'blk.22.ffn_up.weight',
          'ffn_down (5120x17408)': 'blk.24.ffn_down.weight', 'ssm_out (5120x6144)': 'blk.0.ssm_out.weight',
          'attn_qkv (10240x5120)': 'blk.0.attn_qkv.weight', 'attn_gate (6144x5120)': 'blk.0.attn_gate.weight',
          'attn_output (5120x6144)': 'blk.2.ssm_out.weight', 'attn_q (12288x5120)': 'blk.3.attn_q.weight',
          'attn_k/v (1024x5120)': 'blk.3.attn_v.weight'}
for label, tname in shapes.items():
    n, k = (int(v) for v in label.split('(')[1].strip(')').split('x'))
    tiles = repack_gguf_q5_k_tile16(np.asarray(reader.tensor_data(tname))[None, ...]).tiles
    x_bits = ((rng.normal(0, 0.2, (1, k)).astype(np.float32).view(np.uint32) + 0x7FFF) >> 16).astype(np.uint16)
    bufs = []
    try:
        def dev(a):
            b = malloc(a.nbytes, runtime=runtime); bufs.append(b)
            copy_host_to_device(b, host_array_ptr(a), runtime=runtime); return b
        x_dev, t_dev = dev(x_bits), dev(tiles)
        out = malloc(n*2, runtime=runtime); bufs.append(out)
        def run():
            t16g.gguf_q5_k_t16_gemv_decode_bf16_bf16_out(x_dev.ptr, t_dev.ptr, out.ptr, 1, k, n, library=lib, runtime=runtime)
        run(); runtime.device_synchronize()
        best = float('inf')
        for _ in range(3):
            t0 = time.perf_counter()
            for _ in range(20): run()
            runtime.device_synchronize()
            best = min(best, (time.perf_counter() - t0) / 20)
        cnt = counts[label]
        total += best * cnt
        gbs = tiles.nbytes / best / 1e9
        print(f"{label:<28} {best*1e6:7.1f} us  x{cnt:3d} = {best*cnt*1e3:6.2f} ms/token  ({gbs:5.0f} GB/s)")
    finally:
        for b in reversed(bufs): free(b, runtime=runtime)
print(f"Q5 T16 GEMV total: {total*1e3:.2f} ms/token")
