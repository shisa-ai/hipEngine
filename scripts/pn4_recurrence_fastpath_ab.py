"""Decisive recurrence-path fast-path A/B for the Qwen3.6-35B eager decode.

Extends the PN4 host-bound probe to the per-layer conv + recurrent-rmsnorm-gate
launchers (which the launch_gguf_linear fast path did NOT cover). Caches the
resolved ctypes fn + argtypes for `qwen35_linear_attn_chain_conv_decode_bf16_tloop`
and `qwen35_gdn_recurrent_rmsnorm_gate_lowp_bf16` and calls them directly,
skipping the per-call `build(...)`, `getattr`, `fn.argtypes = [...]` reassignment,
and shape checks. Complete-wall A/B in the same session.

Results are CORRECT (same kernels, same args; only the Python dispatch
machinery is bypassed).
"""
import os, sys, time, statistics, ctypes
sys.path.insert(0, '/home/lhl/hipEngine')
import hipengine.runtime.qwen35_gguf_runner as rm
import hipengine.kernels.hip_gfx1100.linear_attn.gdn as gdn
import hipengine.kernels.hip_gfx1100.linear_attn.conv as conv
ctypes.CDLL('libamdhip64.so')

# --- recurrent fast path ---
_rec_fn = {"fn": None, "lib": None}
def _fast_rec(conv_out_ptr, gate_ptr, a_ptr, b_ptr, dt_bias_ptr, a_log_ptr,
              norm_weight_ptr, recurrent_state_ptr, out_ptr, eps,
              num_k_heads, num_v_heads, head_k_dim, head_v_dim, **kw):
    library = kw.get('library') or gdn.build_qwen35_linear_attn_gdn(load=True)
    if _rec_fn["fn"] is None or _rec_fn["lib"] is not library:
        fn = getattr(library, gdn._SYMBOL_LOWP)
        fn.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_float, ctypes.c_int64, ctypes.c_int64,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p,
        ]
        fn.restype = ctypes.c_int
        _rec_fn["fn"] = fn
        _rec_fn["lib"] = library
    runtime = kw.get('runtime') or gdn.get_hip_runtime()
    err = _rec_fn["fn"](
        ctypes.c_void_p(conv_out_ptr), ctypes.c_void_p(gate_ptr),
        ctypes.c_void_p(a_ptr), ctypes.c_void_p(b_ptr),
        ctypes.c_void_p(dt_bias_ptr), ctypes.c_void_p(a_log_ptr),
        ctypes.c_void_p(norm_weight_ptr), ctypes.c_void_p(recurrent_state_ptr),
        ctypes.c_void_p(out_ptr), ctypes.c_float(eps),
        ctypes.c_int64(num_k_heads), ctypes.c_int64(num_v_heads),
        ctypes.c_int64(head_k_dim), ctypes.c_int64(head_v_dim),
        ctypes.c_void_p(kw.get('stream', 0)),
    )
    gdn._check_launch(runtime, err)

# --- conv fast path ---
_cv_fn = {"fn": None, "lib": None}
def _fast_conv(hidden_states_ptr, base_conv_state_ptr, chain_conv_state_ptr,
               conv_weight_ptr, out_ptr, max_nodes, channels, kernel_size, **kw):
    library = kw.get('library') or conv.build_qwen35_linear_attn_conv(load=True)
    if _cv_fn["fn"] is None or _cv_fn["lib"] is not library:
        fn = getattr(library, conv._SYMBOL_CHAIN_BF16_TLOOP)
        fn.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
            ctypes.c_void_p,
        ]
        fn.restype = ctypes.c_int
        _cv_fn["fn"] = fn
        _cv_fn["lib"] = library
    runtime = kw.get('runtime') or conv.get_hip_runtime()
    err = _cv_fn["fn"](
        ctypes.c_void_p(hidden_states_ptr), ctypes.c_void_p(base_conv_state_ptr),
        ctypes.c_void_p(chain_conv_state_ptr), ctypes.c_void_p(conv_weight_ptr),
        ctypes.c_void_p(out_ptr), ctypes.c_int64(max_nodes),
        ctypes.c_int64(channels), ctypes.c_int64(kernel_size),
        ctypes.c_void_p(kw.get('stream', 0)),
    )
    conv._check_launch(runtime, err)

# save originals
_orig_rec = rm.qwen35_gdn_recurrent_rmsnorm_gate_lowp_bf16
_orig_conv = rm.qwen35_linear_attn_chain_conv_decode_bf16_tloop

def run_steps(s, n=32, warmup=8, mode="baseline"):
    if mode == "fast":
        rm.qwen35_gdn_recurrent_rmsnorm_gate_lowp_bf16 = _fast_rec
        rm.qwen35_linear_attn_chain_conv_decode_bf16_tloop = _fast_conv
    else:
        rm.qwen35_gdn_recurrent_rmsnorm_gate_lowp_bf16 = _orig_rec
        rm.qwen35_linear_attn_chain_conv_decode_bf16_tloop = _orig_conv
    tok = 9707
    for _ in range(warmup):
        tok = int(s.step(tok, return_logits=False).token_id)
    walls = []
    for _ in range(n):
        t0 = time.perf_counter()
        tok = int(s.step(tok, return_logits=False).token_id)
        s.runtime.device_synchronize()
        walls.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(walls)

compiler = open('/tmp/hipengine-zbook-production-numerics/20260817T024001Z-94c6d457f9e6/hipcc-version.txt').read().strip() or None
with rm.Qwen35GGUFResidentSession('/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf',
        max_sequence_length=900, compiler_version=compiler,
        require_cached_build=False, backend='hip_gfx1151') as s:
    s.prefill([9707]*512, use_bulk=True, return_logits=False)
    s.runtime.device_synchronize()
    # A/B/A/B interleave to control clock drift
    baselines = []
    fasts = []
    for i in range(4):
        b = run_steps(s, mode="baseline")
        f = run_steps(s, mode="fast")
        baselines.append(b)
        fasts.append(f)
    print(f"baseline med={statistics.median(baselines):.2f} ({[f'{x:.2f}' for x in baselines]})  "
          f"fast med={statistics.median(fasts):.2f} ({[f'{x:.2f}' for x in fasts]})  "
          f"delta={(statistics.median(fasts)-statistics.median(baselines))*1000:+.1f} us/token")
