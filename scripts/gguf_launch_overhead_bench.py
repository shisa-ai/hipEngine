#!/usr/bin/env python3
"""GGUF kernel host-launch overhead microbenchmark.

Decomposes the per-call host (CPU) cost of issuing a single GGUF linear kernel
launch, to quantify how much is Python/ctypes wrapper tax vs the actual async
kernel launch. This is the diagnostic behind the 2026-06-28 finding that the
MTP verifier is ~50/50 host-dispatch-bound (~875 launches/verify) and that
~98% of the per-launch cost is reducible in pure Python (cached extern-C handle
+ argtypes set once + raw int args), not in the GPU kernel.

Measures, for ``launch_gguf_linear`` (rows=4, Q8_0 attn_qkv), best-of-N us/call:
  - full launch_gguf_linear (registry resolve + dispatch chain + ABI + wrapper)
  - dispatch-resolve only (registry/dispatch resolution, no launch)
  - precomputed registry fn() (skips resolve; still per-call _validate + lib lookup)
  - LEAN ctypes (cached extern-C handle, argtypes set once, raw ints) -> the floor

No correctness claim is retained from this run; it is a host-overhead probe.
Requires a local GGUF model and a live ROCm/HIP runtime.

Example:
  PYTHONPATH=. python3 scripts/gguf_launch_overhead_bench.py \
    --model /models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf --iters 2000
"""
from __future__ import annotations

import argparse
import ctypes
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_MODEL = "/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
DEFAULT_PROMPT = "Write a Python function that implements merge sort:"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--iters", type=int, default=2000)
    parser.add_argument("--rows", type=int, default=4)
    args = parser.parse_args(argv)

    from scripts.gguf_mtp_bench import build_chat_prompt
    from hipengine.loading.gguf import GGUFReader
    from hipengine.tokenization.gguf import Qwen35GGUFTokenizer
    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession
    from hipengine.core.memory import malloc, copy_host_to_device, host_array_ptr
    from hipengine.kernels.registry import resolve as registry_resolve
    from hipengine.kernels.hip_gfx1100.quant.gguf_k_gemv import build_gguf_k_gemv, _symbol
    import hipengine.runtime.gguf_linear as GL

    tok = Qwen35GGUFTokenizer.from_gguf_info(GGUFReader(args.model).info)
    prompt = build_chat_prompt(tok, DEFAULT_PROMPT, reasoning="off")
    s = Qwen35GGUFResidentSession(model_path=args.model, use_wmma_prefill=True, use_gemv_decode=True)
    rt = s.runtime
    s.prefill(prompt, return_logits=False)

    weight = s.runner.weights.layer(0).weight("attn_qkv")
    IN = s.runner.hidden_size
    OUT = s.runner.linear_qkv_width
    rows = int(args.rows)
    xbuf = malloc(rows * IN * 2, runtime=rt)
    obuf = malloc(rows * OUT * 2, runtime=rt)
    copy_host_to_device(xbuf, host_array_ptr(np.zeros(rows * IN, np.uint16)), runtime=rt)
    xptr, outptr = xbuf.ptr, obuf.ptr
    qwptr = weight.allocation("raw").tensor.ptr
    N = int(args.iters)

    def bench(fn) -> float:
        fn()
        rt.device_synchronize()
        t0 = time.perf_counter()
        for _ in range(N):
            fn()
        return (time.perf_counter() - t0) / N * 1e6

    def full() -> None:
        GL.launch_gguf_linear(weight, xptr, outptr, rows, IN, OUT, use_wmma_prefill=False, runtime=rt)

    def resolve_only():
        d = GL.resolve_gguf_linear_dispatch(weight, rows=rows)
        d = GL._pack8_decode_dispatch(d, rows=rows, out_features=OUT)
        d = GL._gemv_decode_dispatch(d, rows=rows, use_gemv_decode=GL._resolve_use_gemv_decode(None))
        d = GL._wmma_prefill_dispatch(d, rows=rows, in_features=IN, use_wmma=False)
        d = GL._rowtile_dispatch(d, rows=rows, in_features=IN, use_rowtile=True)
        GL._ensure_linear_kernel_registered(d.key)
        return registry_resolve(backend=d.key.backend, layer=d.key.layer, quant=d.key.quant, variant=d.key.variant)

    reg_fn = resolve_only()

    def precomputed() -> None:
        reg_fn(xptr, qwptr, outptr, rows, IN, OUT, runtime=rt)

    lib = build_gguf_k_gemv(load=True)
    cfn = getattr(lib, _symbol("gguf_q8_0", "gemv_rowtile_bf16_bf16_out"))
    cfn.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int64] * 4 + [ctypes.c_void_p]
    cfn.restype = ctypes.c_int

    def lean() -> None:
        cfn(xptr, qwptr, outptr, rows, IN, OUT, 128, 0)

    t_full = bench(full)
    t_resolve = bench(lambda: resolve_only())
    t_pre = bench(precomputed)
    t_lean = bench(lean)
    s.close()

    print(f"launch_gguf_linear host overhead (rows={rows}, us/call, best of {N}):")
    print(f"  full launch_gguf_linear            : {t_full:8.2f} us")
    print(f"  dispatch-resolve only              : {t_resolve:8.2f} us  ({100*t_resolve/t_full:3.0f}%)")
    print(f"  precomputed registry fn() (raw)    : {t_pre:8.2f} us  ({100*t_pre/t_full:3.0f}%)")
    print(f"  LEAN ctypes (cached handle + ints) : {t_lean:8.2f} us  ({100*t_lean/t_full:3.0f}%)")
    print(f"  -> pure-Python launch floor ~= {t_lean:.2f} us (vs {t_full:.2f} us current)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
