"""PN5: isolate the raw coop router launch cost from runner per-call overhead.

Captures exact coop args via the runner wrapper during one live decode step,
then times raw `fn(...)` launches in a tight loop with real pointers.
Distinguishes ctypes/hipModuleLaunchKernel + build_hip(load=True) cost from
the runner's per-call Python overhead.
"""
import os, sys, time, statistics, ctypes
sys.path.insert(0, '/home/lhl/hipEngine')
import hipengine.runtime.qwen35_gguf_runner as rm
ctypes.CDLL('libamdhip64.so')

MODEL = '/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf'
COMPILER = '/tmp/hipengine-zbook-production-numerics/20260817T024001Z-94c6d457f9e6/hipcc-version.txt'
compiler = open(COMPILER).read().strip() or None

with rm.Qwen35GGUFResidentSession(MODEL,
        max_sequence_length=900, compiler_version=compiler,
        require_cached_build=True, backend='hip_gfx1151') as s:
    s.prefill([9707]*512, use_bulk=True, return_logits=False)
    s.runtime.device_synchronize()

    captured = {}
    orig = rm._try_launch_qwen35_router_topk_split_shared_bf16_f32w
    def capture(*a, **k):
        if not captured:
            captured["a"] = a
            captured["k"] = k
        return orig(*a, **k)
    rm._try_launch_qwen35_router_topk_split_shared_bf16_f32w = capture
    tok = 9707
    tok = int(s.step(tok, return_logits=False).token_id)
    s.runtime.device_synchronize()
    rm._try_launch_qwen35_router_topk_split_shared_bf16_f32w = orig
    a, k = captured["a"], captured["k"]
    # a = (hidden_ptr, expert_weight, shared_weight, logits, selected, routing, counter)
    hptr = int(a[0])
    wptr = int(a[1].allocation().tensor.ptr)
    wsptr = int(a[2].allocation().tensor.ptr)
    logits = int(a[3]); selected = int(a[4]); routing = int(a[5]); counter = int(a[6])
    hs = k["hidden_size"]; ne = k["num_experts"]; tk = k["top_k"]
    print(f"captured: hidden={hptr & 0xffffffff} w={wptr & 0xffffffff} shared={wsptr & 0xffffffff} hs={hs} ne={ne} topk={tk}")

    from hipengine.kernels.registry import resolve
    fn = resolve(backend="hip_gfx1151", layer="router_topk_split_shared",
                 quant="f32", variant="coop_out_bf16_hidden_persistent", missing="none")
    print(f"resolved fn: {fn.__name__}")
    args = (hptr, wptr, wsptr, logits, selected, routing, counter)

    for _ in range(5):
        fn(*args, 1, hs, ne, tk, threads=256, stream=0, runtime=s.runtime)
    s.runtime.device_synchronize()
    ts = []
    for _ in range(100):
        t0 = time.perf_counter()
        fn(*args, 1, hs, ne, tk, threads=256, stream=0, runtime=s.runtime)
        ts.append((time.perf_counter() - t0) * 1e6)
    s.runtime.device_synchronize()
    print(f"raw coop launch (default):       median={statistics.median(ts):.1f}us  p90={sorted(ts)[90]:.1f}us")

    # preloaded library (skip build_hip per call)
    from hipengine.kernels.hip_gfx1100.moe import router as router_mod
    lib = router_mod.build_qwen35_router(load=True)
    ts2 = []
    for _ in range(100):
        t0 = time.perf_counter()
        fn(*args, 1, hs, ne, tk, threads=256, stream=0, runtime=s.runtime, library=lib)
        ts2.append((time.perf_counter() - t0) * 1e6)
    s.runtime.device_synchronize()
    print(f"raw coop launch (preloaded lib):  median={statistics.median(ts2):.1f}us  p90={sorted(ts2)[90]:.1f}us")
