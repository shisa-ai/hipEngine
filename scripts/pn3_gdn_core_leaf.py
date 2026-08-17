#!/usr/bin/env python3
"""PN3 leaf baseline: c1 linear-attention SSM/GDN core at Qwen3.6-35B dims.

Measures the two kernels that make up the ``gdn_attention_core`` decode-stage
exclusive window (fresh GPU-exclusive #2, 5.19 ms/token, ~173 us/layer from the
PN2 24-token marker trace):

* ``qwen35_linear_attn_conv_decode_lowp_kernel`` (bf16 input, conv kernel 4)
* ``qwen35_gdn_recurrent_rmsnorm_gate_lowp_kernel`` (16 k-heads, 32 v-heads,
  head_k_dim = head_v_dim = 128)

Dims mirror Qwen3.6-35B-A3B-UD: ssm_group_count=16, ssm_time_step_rank=32,
ssm_state_size=128, ssm_inner_size=4096 -> num_k_heads=16, num_v_heads=32,
head_k_dim=head_v_dim=128, channels=linear_qkv_width=8192. Timing is wall-clock
burst with device sync (HIP event elapsed reports 0 on this gfx1151/ROCm combo).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Callable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import (
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.hip_gfx1100.linear_attn.conv import (
    build_qwen35_linear_attn_conv,
    qwen35_linear_attn_conv_decode_bf16,
)
from hipengine.kernels.hip_gfx1100.linear_attn.gdn import (
    build_qwen35_linear_attn_gdn,
    qwen35_gdn_recurrent_rmsnorm_gate_lowp_bf16,
)

NUM_K_HEADS = 16
NUM_V_HEADS = 32
HEAD_K_DIM = 128
HEAD_V_DIM = 128
CHANNELS = 8192
SSM_INNER = 4096
CONV_KERNEL = 4


def _bf16_bits(values: np.ndarray) -> np.ndarray:
    f32 = np.ascontiguousarray(values, dtype=np.float32)
    bits = f32.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & np.uint32(1))
    return np.ascontiguousarray((rounded >> 16).astype(np.uint16))


def _upload(runtime, values: np.ndarray):
    array = np.ascontiguousarray(values)
    buffer = malloc(array.nbytes, runtime=runtime)
    copy_host_to_device(buffer, host_array_ptr(array), runtime=runtime)
    return buffer


def _ms(runtime, fn: Callable[[], None], *, burst: int) -> float:
    import time

    fn()
    runtime.device_synchronize()
    start = time.perf_counter()
    for _ in range(burst):
        fn()
    runtime.device_synchronize()
    return (time.perf_counter() - start) / burst * 1e3


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--samples", type=int, default=21)
    parser.add_argument("--burst", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("/tmp/pn3-gdn-core-leaf.json"))
    args = parser.parse_args()

    runtime = get_hip_runtime()
    conv_lib = build_qwen35_linear_attn_conv(require_cached=args.require_cached_build)
    gdn_lib = build_qwen35_linear_attn_gdn(require_cached=args.require_cached_build)
    rng = np.random.default_rng(args.seed)

    buffers = []
    try:
        hidden = _upload(runtime, _bf16_bits(rng.standard_normal(CHANNELS)))
        conv_state = _upload(runtime, rng.standard_normal((CHANNELS, CONV_KERNEL)).astype(np.float32))
        conv_weight = _upload(runtime, rng.standard_normal((CHANNELS, CONV_KERNEL)).astype(np.float32))
        conv_out = _upload(runtime, np.zeros(CHANNELS, np.float32))
        gate = _upload(runtime, _bf16_bits(rng.standard_normal(SSM_INNER)))
        a = _upload(runtime, _bf16_bits(rng.standard_normal(NUM_V_HEADS)))
        b = _upload(runtime, _bf16_bits(rng.standard_normal(NUM_V_HEADS)))
        dt_bias = _upload(runtime, rng.standard_normal(NUM_V_HEADS).astype(np.float32))
        a_log = _upload(runtime, rng.standard_normal(NUM_V_HEADS).astype(np.float32))
        norm_weight = _upload(runtime, rng.standard_normal(NUM_V_HEADS * HEAD_V_DIM).astype(np.float32))
        recurrent_state = _upload(
            runtime,
            rng.standard_normal((NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM)).astype(np.float32),
        )
        out = _upload(runtime, np.zeros(NUM_V_HEADS * HEAD_V_DIM, np.float32))
        out_bf16 = _upload(runtime, np.zeros(NUM_V_HEADS * HEAD_V_DIM, np.uint16))
        buffers.extend(
            (hidden, conv_state, conv_weight, conv_out, gate, a, b, dt_bias, a_log,
             norm_weight, recurrent_state, out, out_bf16)
        )

        def conv() -> None:
            qwen35_linear_attn_conv_decode_bf16(
                hidden.ptr, conv_state.ptr, conv_weight.ptr, conv_out.ptr,
                CHANNELS, CONV_KERNEL, library=conv_lib, runtime=runtime,
            )

        def recurrence() -> None:
            qwen35_gdn_recurrent_rmsnorm_gate_lowp_bf16(
                conv_out.ptr, gate.ptr, a.ptr, b.ptr, dt_bias.ptr, a_log.ptr,
                norm_weight.ptr, recurrent_state.ptr, out.ptr,
                1e-5, NUM_K_HEADS, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM,
                library=gdn_lib, runtime=runtime,
            )

        launchers = {"conv": conv, "recurrence": recurrence, "core_pair": lambda: (conv(), recurrence())}
        for _ in range(args.warmups):
            conv()
            recurrence()
        runtime.device_synchronize()
        results = {}
        for name, fn in launchers.items():
            timings = [_ms(runtime, fn, burst=args.burst) for _ in range(args.samples)]
            results[name] = {
                "median_ms": statistics.median(timings),
                "mean_ms": statistics.mean(timings),
                "samples_ms": timings,
            }
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)

    pair = results["core_pair"]["median_ms"]
    results["_summary"] = {
        "shape": [NUM_K_HEADS, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM],
        "conv_kernel": CONV_KERNEL,
        "channels": CHANNELS,
        "core_pair_median_ms_per_layer": pair,
        "projected_ms_per_token_30_layers": pair * 30,
        "trace_gdn_attention_core_ms_per_token": 5.190,
        "projected_vs_trace_ratio": pair * 30 / 5.190,
    }
    print(json.dumps(results, indent=2))
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
