"""PN3-FULLATTN thread-geometry whole-decode A/B: 256 vs 1024 threads.

The kernel-level sweep showed 1024 threads beats the retained 256-thread c1
leaf by 6-26% at contexts 256-1024 on gfx1151. This measures the real 35B-A3B
c1 decode wall impact by swapping the short-batch leaf between the resolved
fixed256 (256-thread) fn and the parameterized threads wrapper (threads=1024).
The 1024-thread body is a production probe (not byte-exact: last-ulp drift via
different warp-reduction/value-group order), so tokens are diffed (KL/top-1)
across legs. Counter-rotated A/B/A/B, median of 30 steps.
"""
import os, sys, time, statistics, ctypes, functools, types
sys.path.insert(0, '/home/lhl/hipEngine')
import hipengine.runtime.qwen35_gguf_runner as rm
ctypes.CDLL('libamdhip64.so')
from hipengine.kernels.hip_gfx1100.attention import (
    qwen35_paged_full_attn_decode_context_bf16_batch_fixed256_threads_spans,
)

ORIG = rm.Qwen35GGUFFullStackRunner._full_attn_decode_short_batch_fn

def _leaf_threads(spans, threads):
    return functools.partial(
        qwen35_paged_full_attn_decode_context_bf16_batch_fixed256_threads_spans,
        threads=threads,
    )

def install(r, mode):
    if mode == '256':
        r._full_attn_decode_short_batch_fn = types.MethodType(ORIG, r)
    else:
        r._full_attn_decode_short_batch_fn = types.MethodType(
            lambda self, spans: _leaf_threads(spans, 1024), r)

def run(s, n=30, warmup=6):
    tok = 9707
    for _ in range(warmup):
        tok = int(s.step(tok, return_logits=False).token_id)
    walls = []
    toks = []
    for _ in range(n):
        t0 = time.perf_counter()
        tok = int(s.step(tok, return_logits=False).token_id)
        s.runtime.device_synchronize()
        walls.append((time.perf_counter() - t0) * 1e3)
        toks.append(tok)
    return statistics.median(walls), toks

def kl_top1(a, b):
    import numpy as np
    n = len(a)
    same = sum(1 for x, y in zip(a, b) if x == y)
    return same / n, same, n

with rm.Qwen35GGUFResidentSession('/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf',
        max_sequence_length=900, backend='hip_gfx1151') as s:
    s.prefill([9707]*512, use_bulk=True, return_logits=False)
    s.runtime.device_synchronize()
    install(s.runner, '256');  w1, t1 = run(s)
    install(s.runner, '1024'); w2, t2 = run(s)
    install(s.runner, '256');  w3, t3 = run(s)
    install(s.runner, '1024'); w4, t4 = run(s)
    print(f"256-thread   #1: {w1:.2f} ms/tok")
    print(f"1024-thread    : {w2:.2f} ms/tok")
    print(f"256-thread   #2: {w3:.2f} ms/tok")
    print(f"1024-thread  #2: {w4:.2f} ms/tok")
    mean256 = (w1 + w3) / 2
    mean1024 = (w2 + w4) / 2
    print(f"mean 256={mean256:.2f} mean 1024={mean1024:.2f} "
          f"delta={(mean256-mean1024)*1000:+.0f} us/tok ({(mean256/mean1024-1)*100:+.1f}%)")
    # exactness: 1024 legs vs the 256 legs
    ag1, same1, n1 = kl_top1(t2, t1)
    ag2, same2, n2 = kl_top1(t4, t3)
    print(f"top-1 agreement 1024#1 vs 256#1: {ag1*100:.1f}% ({same1}/{n1})")
    print(f"top-1 agreement 1024#2 vs 256#2: {ag2*100:.1f}% ({same2}/{n2})")
    # determinism within mode
    agA, sameA, _ = kl_top1(t1, t3)
    agB, sameB, _ = kl_top1(t2, t4)
    print(f"determinism 256#1==256#2: {agA*100:.1f}%  |  1024#1==1024#2: {agB*100:.1f}%")
    print(f"256 toks: {t1[:8]}... {t3[:8]}...")
    print(f"1024 toks: {t2[:8]}... {t4[:8]}...")
