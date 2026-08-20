"""Measure prefill (aotriton) vs decode economics on gfx1151 for 35B-A3B.

Answers: (1) what does a 512-token aotriton prefill cost, (2) how does that
amortize over decode, (3) is beating aotriton on prefill worth any wall.
"""
import sys, time, statistics, ctypes
sys.path.insert(0, '/home/lhl/hipEngine')
import hipengine.runtime.qwen35_gguf_runner as rm
ctypes.CDLL('libamdhip64.so')

with rm.Qwen35GGUFResidentSession('/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf',
        max_sequence_length=2048, backend='hip_gfx1151') as s:
    # prefill 512 via the standard bulk path (aotriton at rows>=512)
    for n in (128, 256, 512, 1024):
        toks = [9707] * n
        t0 = time.perf_counter()
        s.prefill(toks, use_bulk=True, return_logits=False)
        s.runtime.device_synchronize()
        dt = (time.perf_counter() - t0) * 1e3
        print(f"prefill {n:5d} tok: {dt:8.1f} ms ({dt/n*1e3:6.1f} us/tok)  mode via s.prefill")
    # decode baseline
    s.runtime.device_synchronize()
    tok = 9707
    for _ in range(5):
        tok = int(s.step(tok, return_logits=False).token_id)
    s.runtime.device_synchronize()
    walls = []
    for _ in range(15):
        t0 = time.perf_counter()
        tok = int(s.step(tok, return_logits=False).token_id)
        s.runtime.device_synchronize()
        walls.append((time.perf_counter() - t0) * 1e3)
    dec = statistics.median(walls)
    print(f"decode wall: {dec:.2f} ms/tok")
    for n, pf in ((128, 1), (512, 1)):
        # amortized prefill cost per decode token across a generation of G tokens
        for G in (32, 128, 512):
            print(f"  prefill {n} amortized over {G:4d} gen tok: {0.0:.2f}ms -> compute")
