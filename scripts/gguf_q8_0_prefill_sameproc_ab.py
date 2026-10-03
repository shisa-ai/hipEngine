"""Same-process A/B of the two q8_0 prefill routes, comparing every call's output.

Cross-process comparison cannot separate an argument difference from allocator
layout, and comparing only the final logits says nothing about which of the 205
dispatches diverged. Running both arms in one process with the session toggle
removes the layout variable, and capturing each call's output buffer finds the
first dispatch whose result differs by more than bf16 output rounding.
"""
from __future__ import annotations
import json, sys
sys.path.insert(0, "."); sys.path.insert(0, "scripts")

import numpy as np
from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import DeviceBuffer, copy_device_to_host, host_array_ptr
from hipengine.quant.gguf import bf16_to_float32
from hipengine.kernels.hip_gfx1100.quant import gguf_k_gemv  # noqa: F401
from hipengine.kernels.hip_gfx1100.quant import gguf_q8_0_prefill  # noqa: F401
from hipengine.kernels.registry import KernelKey, register, _KERNELS
from hipengine.runtime.gguf_linear import wmma_prefill_session

TARGETS = (
    "exact_prefill_tile4x16_bf16_bf16_out",
    "exact_prefill_tile8x4_bf16_bf16_out",
    "exact_prefill_tile8x2_bf16_bf16_out",
    "wmma_prefill_bf16_bf16_out",
)
CAPTURE: list[dict] = []

runtime = get_hip_runtime()
for variant in TARGETS:
    key = KernelKey("hip_gfx1100", "linear", "gguf_q8_0", variant)
    if key not in _KERNELS:
        continue
    inner = _KERNELS[key]

    def make(inner=inner, variant=variant):
        def wrapper(*args, **kwargs):
            result = inner(*args, **kwargs)
            try:
                out_ptr, rows, outf = args[2], int(args[3]), int(args[5])
                host = np.empty(rows * outf, dtype=np.uint16)
                buf = DeviceBuffer(ptr=out_ptr, nbytes=host.nbytes)
                copy_device_to_host(host_array_ptr(host), buf, runtime=runtime)
                vals = bf16_to_float32(host).astype(np.float64)
                CAPTURE.append({
                    "variant": variant, "rows": rows, "out": outf,
                    "max_abs": float(np.abs(vals).max()),
                    "sum_abs": float(np.abs(vals).sum()),
                    "finite": bool(np.isfinite(vals).all()),
                })
            except Exception as exc:  # noqa: BLE001
                CAPTURE.append({"variant": variant, "error": repr(exc)})
            return result
        return wrapper

    register(key, make(), replace=True)


from scripts.gemma4_campaign_bench import (  # noqa: E402
    DEFAULT_ARTIFACT, _resolve_generator, exact_prompt_ids,
)

n = int(sys.argv[1])
llm, runner, meta = _resolve_generator(DEFAULT_ARTIFACT, 8192)
ids = exact_prompt_ids(llm.tokenize, n)

runner.reset()
with wmma_prefill_session(False):
    runner.forward(ids)
off = list(CAPTURE)
CAPTURE.clear()

runner.reset()
with wmma_prefill_session(True):
    runner.forward(ids)
on = list(CAPTURE)

json.dump({"off": off, "on": on}, open(sys.argv[2], "w"), indent=1)
print(f"off={len(off)} on={len(on)}", file=sys.stderr)
