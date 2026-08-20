"""Isolate aotriton full-attention prefill cost vs total prefill (35B-A3B).

Model = 40 layers (30 linear_attention/GDN + 10 full_attention). The 10
full-attention layers route to aotriton for batched prefill (rows>=512).
Time one full-attention layer through run_full_attention_prefill_layer
(aotriton path) at 512 rows and compare against whole-prefill per-layer cost.
"""
import sys, time, statistics, ctypes
sys.path.insert(0, '/home/lhl/hipEngine-main')
import hipengine.runtime.qwen35_gguf_runner as rm
import numpy as np
ctypes.CDLL('libamdhip64.so')

with rm.Qwen35GGUFResidentSession('/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf',
        max_sequence_length=1024, backend='hip_gfx1151') as s:
    cfg = s.runner.weights.config
    full_ids = [i for i, t in enumerate(cfg.layer_types) if t == 'full_attention']
    print("full_attention layers:", full_ids)
    # whole prefill 512 timing (aotriton active for full-attn layers)
    toks = [9707] * 512
    t0 = time.perf_counter()
    s.prefill(toks, use_bulk=True, return_logits=False)
    s.runtime.device_synchronize()
    total_ms = (time.perf_counter() - t0) * 1e3
    print(f"whole prefill 512: {total_ms:.0f} ms ({total_ms/40:.1f} ms/layer avg)")

    # per-layer aotriton full-attention cost at 512 rows
    hidden = (np.random.default_rng(0).standard_normal((512, s.runner.weights.config.hidden_size)) * 0.02).astype(np.float16).view(np.uint16)
    lid = full_ids[0]
    # warmup
    r = s.runner.run_full_attention_prefill_layer(lid, hidden)
    s.runtime.device_synchronize()
    print("mode:", r.mode)
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        r = s.runner.run_full_attention_prefill_layer(lid, hidden)
        s.runtime.device_synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    med = statistics.median(times)
    print(f"one full-attention layer (aotriton, 512 rows): {med:.1f} ms")
    print(f"x10 full-attention layers: {med*10:.0f} ms = {med*10/total_ms*100:.0f}% of prefill wall")
