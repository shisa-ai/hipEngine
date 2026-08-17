"""Decisive host-vs-GPU bound A/B for the Qwen3.6-35B eager decode.

Probe (NOT production code): monkeypatches launch_gguf_linear with a
memoized fast path that skips the ~20-line cache-key construction, the
`_resolve_use_*` env reads, the session gets, and `_DISPATCH_RESOLVE_CACHE`
lookup on cache hit, jumping straight to the launch tail with a cached
(abi, fn, quant, variant) tuple keyed by (id(weight), rows, inf, outf,
backend). Measures the steady-state per-step complete wall (sync'd) with the
fast path ON vs OFF in the same session. If the wall drops, the model is
host-bound; if flat, GPU-bound.
"""
import os, sys, time, statistics, ctypes
sys.path.insert(0, '/home/lhl/hipEngine')
import hipengine.runtime.gguf_linear as gl
from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession
ctypes.CDLL('libamdhip64.so')

_orig = gl.launch_gguf_linear
_FAST = {}
_fast_hits = 0
_fast_calls = 0

def _fast_launch(weight, x_ptr, out_ptr, rows, in_features, out_features, **kw):
    global _fast_hits, _fast_calls
    _fast_calls += 1
    activation_dtype = kw.get('activation_dtype', gl.GGUF_ACTIVATION_BF16)
    output_dtype = kw.get('output_dtype', gl.GGUF_OUTPUT_BF16)
    threads = kw.get('threads', 0)
    use_q4_pack8_wmma = kw.get('use_q4_pack8_wmma', False)
    registered_variant = kw.get('registered_variant')
    backend = kw.get('backend')
    hot = (rows == 1 and activation_dtype == gl.GGUF_ACTIVATION_BF16
           and output_dtype == gl.GGUF_OUTPUT_BF16 and threads == 0
           and not use_q4_pack8_wmma and registered_variant is None)
    if not hot:
        return _orig(weight, x_ptr, out_ptr, rows, in_features, out_features, **kw)
    resolved_backend = gl._weight_backend(weight, backend=backend)
    key = (id(weight), rows, in_features, out_features, resolved_backend)
    cached = _FAST.get(key)
    if cached is None:
        # First call: run the original (full correct resolution), then capture.
        _orig(weight, x_ptr, out_ptr, rows, in_features, out_features, **kw)
        # Re-derive the long cache key once to grab the resolved dispatch.
        f_gemv = gl._resolve_use_gemv_decode(kw.get('use_gemv_decode'))
        use_wmma = gl._resolve_use_wmma_prefill(kw.get('use_wmma_prefill'))
        f_rowtile = (not use_wmma) and gl._resolve_use_q4k_rowtile(None)
        raw_k_rowbatch = gl.raw_k_prefill_rowbatch()
        raw_k_variant = gl.raw_k_prefill_variant()
        raw_weight_ptr = (int(weight.allocation("raw").tensor.ptr)
                          if weight.spec.layout == gl.LAYOUT_RAW_GGUF else None)
        long_key = (
            gl.generation(), weight.spec.layout, weight.spec.quant_key,
            rows, in_features, out_features, activation_dtype, output_dtype,
            resolved_backend, f_gemv, use_wmma, f_rowtile,
            raw_k_rowbatch, raw_k_variant, bool(use_q4_pack8_wmma),
            os.environ.get("HIPENGINE_GGUF_Q4_PACK8_WMMA_BULK", "1") != "0",
            registered_variant, bool(gl._native_batch_decode_session_enabled),
            None, None, None, raw_weight_ptr,
        )
        cached = gl._DISPATCH_RESOLVE_CACHE.get(long_key)
        if cached is None:
            return _orig(weight, x_ptr, out_ptr, rows, in_features, out_features, **kw)
        _FAST[key] = cached
    abi, fn, quant, variant = cached
    _fast_hits += 1
    libraries = kw.get('libraries')
    runtime = kw.get('runtime')
    stream = kw.get('stream', 0)
    library = None
    if libraries is not None:
        library = libraries.get(f"{quant}:{variant}", libraries.get(quant))
    kwargs = {"stream": stream, "runtime": runtime}
    if abi == "t16" and quant == "gguf_q8_0_t16_v1":
        q8_t16_threads = gl._resolve_q8_t16_threads(threads)
        if q8_t16_threads:
            kwargs["threads"] = q8_t16_threads
    elif threads:
        kwargs["threads"] = threads
    if library is not None:
        kwargs["library"] = library
    if (abi == "t16" and quant == "gguf_q8_0_t16_v1"
            and activation_dtype == gl.GGUF_ACTIVATION_BF16
            and output_dtype == gl.GGUF_OUTPUT_BF16
            and gl._use_q8_t16_all_rowtile(rows=rows, in_features=in_features,
                                           threads=threads)):
        gl.gguf_q8_0_t16_gemv_decode_rowtile4_bf16_bf16_out(
            x_ptr, weight.allocation("tiles").tensor.ptr, out_ptr, rows,
            in_features, out_features, threads=gl._Q8_T16_ROWTILE_THREADS, **kwargs)
        return
    gl._LAUNCH_ABI[abi](fn, weight, x_ptr, out_ptr, rows, in_features,
                        out_features, kwargs)

gl.launch_gguf_linear = _fast_launch
# The runner imports launch_gguf_linear by name into its own module namespace;
# patch that reference too so the hot per-layer/per-token calls hit the fast path.
import hipengine.runtime.qwen35_gguf_runner as _rm
_orig_rm_launch = _rm.launch_gguf_linear
_rm.launch_gguf_linear = _fast_launch

def run_steps(s, n=32, warmup=8):
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
with Qwen35GGUFResidentSession('/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf',
        max_sequence_length=700, compiler_version=compiler,
        require_cached_build=False, backend='hip_gfx1151') as s:
    s.prefill([9707]*512, use_bulk=True, return_logits=False)
    s.runtime.device_synchronize()
    # Mode OFF: bypass fast path (restore original) for a true baseline.
    gl.launch_gguf_linear = _orig
    _rm.launch_gguf_linear = _orig_rm_launch
    off = run_steps(s)
    # Mode ON: fast path.
    gl.launch_gguf_linear = _fast_launch
    _rm.launch_gguf_linear = _fast_launch
    _FAST.clear()
    on = run_steps(s)
    print(f"wall OFF={off:.2f} ms/tok  ON(fastpath)={on:.2f} ms/tok  "
          f"delta={(on-off)*1000:+.1f} us  hits={_fast_hits} calls={_fast_calls}")
