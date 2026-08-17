"""Decisive recurrence-path host-vs-GPU A/B for the Qwen3.6-35B eager decode.

Short-circuits the per-layer conv + recurrent-rmsnorm-gate launches to measure
how much complete-wall the gdn_attention_core path really contributes. If the
wall drops by ~the 5.19 ms/token marker wall, the recurrence kernels are a real
GPU cost; if it drops by only ~the 1.2 ms/token leaf, the marker overstates and
the recurrence path has no exposed host mechanism (the GPU is busy with other
stages' kernels during the marker window).

NOTE: results are WRONG (recurrence outputs uncomputed) -- wall-timing only.
"""
import os, sys, time, statistics, ctypes
sys.path.insert(0, '/home/lhl/hipEngine')
import hipengine.runtime.qwen35_gguf_runner as rm
ctypes.CDLL('libamdhip64.so')

def noop(*a, **k):
    return None

def run_steps(s, n=32, warmup=8, patch=True):
    if patch:
        _orig_rec = rm.qwen35_gdn_recurrent_rmsnorm_gate_lowp_bf16
        _orig_conv = rm.qwen35_linear_attn_chain_conv_decode_bf16_tloop
        _orig_conv2 = rm.qwen35_linear_attn_conv_decode_bf16
        rm.qwen35_gdn_recurrent_rmsnorm_gate_lowp_bf16 = noop
        rm.qwen35_linear_attn_chain_conv_decode_bf16_tloop = noop
        rm.qwen35_linear_attn_conv_decode_bf16 = noop
    tok = 9707
    for _ in range(warmup):
        tok = int(s.step(tok, return_logits=False).token_id)
    walls = []
    for _ in range(n):
        t0 = time.perf_counter()
        tok = int(s.step(tok, return_logits=False).token_id)
        s.runtime.device_synchronize()
        walls.append((time.perf_counter() - t0) * 1e3)
    if patch:
        rm.qwen35_gdn_recurrent_rmsnorm_gate_lowp_bf16 = _orig_rec
        rm.qwen35_linear_attn_chain_conv_decode_bf16_tloop = _orig_conv
        rm.qwen35_linear_attn_conv_decode_bf16 = _orig_conv2
    return statistics.median(walls)

compiler = open('/tmp/hipengine-zbook-production-numerics/20260817T024001Z-94c6d457f9e6/hipcc-version.txt').read().strip() or None
with rm.Qwen35GGUFResidentSession('/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf',
        max_sequence_length=700, compiler_version=compiler,
        require_cached_build=False, backend='hip_gfx1151') as s:
    s.prefill([9707]*512, use_bulk=True, return_logits=False)
    s.runtime.device_synchronize()
    # baseline (no patch)
    rm.launch_gguf_linear = rm.launch_gguf_linear  # keep
    base = run_steps(s, patch=False)
    # conv+recurrence no-op
    noop_wall = run_steps(s, patch=True)
    print(f"wall baseline={base:.2f} ms/tok  conv+recurrence noop={noop_wall:.2f} ms/tok  "
          f"drop={((base-noop_wall))*1000:+.1f} us/token")
