import sys, time
sys.path.insert(0, '/home/lhl/hipEngine-ud')
import numpy as np
import hipengine.kernels.hip_gfx1100.quant.gguf_t16_selected_gemv as t16
from hipengine.core.memory import (copy_device_to_host, copy_host_to_device,
                                   free, host_array_ptr, malloc)
from hipengine.core.hip import get_hip_runtime

def bf16(x):
    x = np.asarray(x, np.float32); return (x.view(np.uint32) >> 16).astype(np.uint16)
def bf16_f32(u): return (u.astype(np.uint32) << 16).view(np.float32)

QK_K, T16_COLS = 256, 16
Q5_T16_BLOCK_BYTES = 2880
# offsets
D, DMIN = 0, 32
SCALE, MIN = 64, 64 + 128
QL, QH = 64 + 256, 64 + 256 + 2048

SHAPES = [(5120, 6144), (5120, 17408), (17408, 5120), (1024, 5120),
          (6144, 5120), (5120, 10240), (5120, 12288)]

rng = np.random.default_rng(7)
rt = get_hip_runtime()
stream = 0

def make_tiles(N, K):
    nb = K // 256
    tiles = np.zeros((nb * Q5_T16_BLOCK_BYTES,), np.uint8)  # one out_tile col-group? tiles indexed per (out_tile, blk)
    # Actually layout: tile = tiles + (out_tile * blocks_per_row + blk) * BYTES
    # For N outputs: out_tiles = N/16 groups... total bytes = (N/16)*nb*BYTES? NO:
    # out_tile indexes 16-col groups; each out_tile has blocks_per_row tiles.
    total = (N // 16) * nb * Q5_T16_BLOCK_BYTES
    t = np.zeros(total, np.uint8)
    # d/dmin: small f16
    d16 = np.full(total // 2, 0.01, np.float16).view(np.uint8)  # not per-offset; set per tile below cheaply
    # set d/dmin regions per tile: offsets 0..32 = d, 32..64 = dmin
    for i in range(total // Q5_T16_BLOCK_BYTES):
        base = i * Q5_T16_BLOCK_BYTES
        t[base+D:base+DMIN] = np.full(32, 0.01, np.float16).view(np.uint8) if False else np.concatenate([np.full(16, 0.01, np.float16).view(np.uint8)] )
        t[base+DMIN:base+SCALE] = t[base+D:base+DMIN]
        t[base+SCALE:base+MIN] = rng.integers(0, 8, 128, dtype=np.uint8)   # scale bytes (sb,col)
        t[base+MIN:base+QL] = rng.integers(0, 8, 128, dtype=np.uint8)     # min bytes
        t[base+QL:base+QH] = rng.integers(0, 256, 2048, dtype=np.uint8)   # nibbles
        t[base+QH:base+QH+512] = rng.integers(0, 256, 512, dtype=np.uint8)  # high bits
    return t

print(f"{'shape':>16s} {'direct us':>10s} {'local32 us':>11s} {'speedup':>8s} {'maxrel':>10s}")
for (K, N) in SHAPES:
    tiles = make_tiles(N, K)
    x = bf16(rng.normal(0, 0.1, (1, K)))
    got = np.zeros((1, N), np.uint16); ref = np.zeros((1, N), np.uint16)
    bufs = []
    try:
        def dev(a):
            b = malloc(a.nbytes); bufs.append(b)
            copy_host_to_device(b, host_array_ptr(a), a.nbytes); return b
        x_b, t_b = dev(x), dev(tiles)
        o_b = malloc(got.nbytes); bufs.append(o_b)
        r_b = malloc(ref.nbytes); bufs.append(r_b)
        t16.gguf_q5_k_t16_gemv_decode_bf16_bf16_out(x_b.ptr, t_b.ptr, r_b.ptr, 1, K, N)
        t16.gguf_q5_k_t16_dense_single_local32_bf16_bf16_out(x_b.ptr, t_b.ptr, o_b.ptr, 1, K, N)
        copy_device_to_host(host_array_ptr(ref), r_b, ref.nbytes)
        copy_device_to_host(host_array_ptr(got), o_b, got.nbytes)
        a = bf16_f32(ref).astype(np.float64); c = bf16_f32(got).astype(np.float64)
        rel = float(np.abs(a-c).max()) / max(float(np.abs(a).max()), 1e-30)
        # timing: 50 iterations each
        rt.stream_synchronize(0)
        for name, fn, ptr in (("d", t16.gguf_q5_k_t16_gemv_decode_bf16_bf16_out, r_b),
                              ("l", t16.gguf_q5_k_t16_dense_single_local32_bf16_bf16_out, o_b)):
            for _ in range(5): fn(x_b.ptr, t_b.ptr, ptr.ptr, 1, K, N)
            rt.stream_synchronize(0)
            t0 = time.perf_counter()
            for _ in range(50): fn(x_b.ptr, t_b.ptr, ptr.ptr, 1, K, N)
            rt.stream_synchronize(0)
            if name == "d": td = (time.perf_counter()-t0)/50*1e6
            else: tl = (time.perf_counter()-t0)/50*1e6
        print(f"{K:6d}x{N:<8d} {td:10.1f} {tl:11.1f} {td/tl:7.2f}x {rel:10.2e}")
    finally:
        for b in reversed(bufs): free(b)
