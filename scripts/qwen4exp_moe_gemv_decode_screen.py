#!/usr/bin/env python3
"""#22 R11 screen: MoE expert GEMV decode pair, incumbent vs candidates.

Measures the two decode owners at the exact production geometry on real
weights — gate/up (Q4_K dual, q8_1-quantized activation, 8 selected
experts, 2560->512 silu'd) and down/combine (Q5_1 weighted sum, 512->2560)
— reporting per-call time, achieved weight bandwidth, and drift versus
the incumbent kernels. Diagnostic only; no runtime default changes.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import free
from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_gemv import (
    build_gguf_q4_k_gemv,
    gguf_q4_k_quantize_bf16_q8_1,
    gguf_q4_k_selected_dual_q8_1_dp4a_silu_logical128_t64_gemv_bf16_bf16_out,
    gguf_q4_k_selected_dual_q8_1_dp4a_silu_warp256_gemv_bf16_bf16_out,
)
from hipengine.kernels.hip_gfx1100.quant.qwen4_exp_q5_1 import (
    build_qwen4_exp_q5_1,
    qwen4_exp_q5_1_selected_weighted_sum_logical256_t64_bf16_bf16_out,
    qwen4_exp_q5_1_selected_weighted_sum_warp256_bf16_bf16_out,
)
from hipengine.loading.gguf import GGUFReader, discover_gguf_files
from scripts.qwen4exp_canonical_ar_bench import _git_metadata, _host_metadata
from tests.test_qwen4_exp_pf3_moe_schedules import _upload, _alloc, _download


def hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
        return True
    except OSError:
        return False


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-root", type=Path, required=True)
    p.add_argument("--compiler-version-file", type=Path, required=True)
    p.add_argument("--pairs", type=int, default=12)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if not hip_available():
        p.error("HIP runtime unavailable")
    os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(a.compiler_version_file)

    register_gfx1151_kernels(replace=True)
    runtime = get_hip_runtime()
    lib4 = build_gguf_q4_k_gemv(load=True)
    lib51 = build_qwen4_exp_q5_1(load=True)

    readers = [GGUFReader(path) for path in discover_gguf_files(a.model_root)]

    def tensor(name):
        reader = next(
            r for r in readers if any(t.name == name for t in r.info.tensors))
        info = reader.tensor_info(name)
        return reader.tensor_data(name), info

    gate_raw, _ = tensor("blk.4.ffn_gate_exps.weight")
    up_raw, _ = tensor("blk.4.ffn_up_exps.weight")
    # blk.4's down is Q8_0; the Q5_1 down incumbent/kernels target the
    # 43 Q5_1-down layers, so use blk.0's Q5_1 slab for the down leg.
    down_raw, _ = tensor("blk.0.ffn_down_exps.weight")

    hidden, ffn, experts = 2560, 640, 512
    top_k = 10
    # qwen4exp ffn_*_exps layout: [experts, out, in] raw Q4_K/Q5_1 slabs.
    report = {
        "schema": 1,
        "kind": "qwen4exp_moe_gemv_decode_screen",
        "source": _git_metadata(ROOT),
        "host": _host_metadata(),
        "command": sys.argv,
        "geometry": {
            "hidden": hidden, "ffn": ffn, "experts": experts, "top_k": top_k,
        },
        "runtime_default_changed": False,
        "cases": [],
    }
    allocations = []
    try:
        rng = np.random.default_rng(2211)
        # ---- gate/up incumbent ------------------------------------------
        dw_gate = _upload(gate_raw, runtime, allocations)
        dw_up = _upload(up_raw, runtime, allocations)
        x_bf16 = rng.normal(0, 1.0, size=(1, hidden)).astype(np.float16)
        x_bf16 = (x_bf16.astype(np.float32)).astype(
            np.dtype("float32")).view(np.uint16) if False else _bf16_bits(x_bf16)
        dx = _upload(x_bf16.reshape(hidden), runtime, allocations)
        selected = np.sort(rng.choice(experts, size=top_k, replace=False))
        dsel = _upload(selected.astype(np.int64), runtime, allocations)
        dq8 = _alloc(4096 * 64, np.uint8, runtime, allocations)  # q8_1 blocks
        out_gu = _alloc(top_k * ffn, np.uint16, runtime, allocations)

        out_gu_v2 = _alloc(top_k * ffn, np.uint16, runtime, allocations)

        def run_gate_up() -> None:
            gguf_q4_k_quantize_bf16_q8_1(
                dx.ptr, dq8.ptr, 1, hidden, library=lib4, runtime=runtime)
            gguf_q4_k_selected_dual_q8_1_dp4a_silu_logical128_t64_gemv_bf16_bf16_out(
                dq8.ptr, dsel.ptr, dw_gate.ptr, dw_up.ptr, out_gu.ptr,
                1, top_k, experts, hidden, ffn,
                library=lib4, runtime=runtime)
            runtime.device_synchronize()

        def run_gate_up_v2() -> None:
            gguf_q4_k_quantize_bf16_q8_1(
                dx.ptr, dq8.ptr, 1, hidden, library=lib4, runtime=runtime)
            gguf_q4_k_selected_dual_q8_1_dp4a_silu_warp256_gemv_bf16_bf16_out(
                dq8.ptr, dsel.ptr, dw_gate.ptr, dw_up.ptr, out_gu_v2.ptr,
                1, top_k, experts, hidden, ffn,
                library=lib4, runtime=runtime)
            runtime.device_synchronize()

        # ---- down incumbent ---------------------------------------------
        dw_down = _upload(down_raw, runtime, allocations)
        inter = rng.normal(0, 1.0, size=(top_k, ffn)).astype(np.float32)
        inter_bf16 = _bf16_bits(inter)
        dinter = _upload(inter_bf16.reshape(top_k * ffn), runtime, allocations)
        routing = np.abs(rng.normal(0, 0.1, size=top_k)).astype(np.float32)
        routing /= routing.sum()
        drouting = _upload(routing, runtime, allocations)
        out_down = _alloc(hidden, np.uint16, runtime, allocations)

        out_down_v2 = _alloc(hidden, np.uint16, runtime, allocations)

        def run_down() -> None:
            qwen4_exp_q5_1_selected_weighted_sum_logical256_t64_bf16_bf16_out(
                dinter.ptr, dsel.ptr, dw_down.ptr, drouting.ptr, out_down.ptr,
                top_k, experts, ffn, hidden,
                library=lib51, runtime=runtime)
            runtime.device_synchronize()

        def run_down_v2() -> None:
            qwen4_exp_q5_1_selected_weighted_sum_warp256_bf16_bf16_out(
                dinter.ptr, dsel.ptr, dw_down.ptr, drouting.ptr, out_down_v2.ptr,
                top_k, experts, ffn, hidden,
                library=lib51, runtime=runtime)
            runtime.device_synchronize()

        # reference outputs + timing
        for name, fn, bytes_moved, out_ptr, out_shape in (
            ("gate_up_dp4a", run_gate_up,
             top_k * 2 * ffn * hidden * 144 / 256.0, out_gu.ptr,
             (top_k, ffn)),
            ("down_wsum", run_down,
             top_k * hidden * ffn * 24 / 32.0, out_down.ptr, (hidden,)),
            ("down_warp256", run_down_v2,
             top_k * hidden * ffn * 24 / 32.0, out_down_v2.ptr, (hidden,)),
            ("gate_up_warp256", run_gate_up_v2,
             top_k * 2 * ffn * hidden * 144 / 256.0, out_gu_v2.ptr,
             (top_k, ffn)),
        ):
            fn()
            ref = _download(_Buf(out_ptr, int(np.prod(out_shape)) * 2), out_shape, np.uint16, runtime)
            times = []
            for pair in range(a.pairs):
                t0 = time.perf_counter()
                fn()
                times.append(time.perf_counter() - t0)
            med = statistics.median(times) * 1e6
            case = {
                "kernel": name,
                "median_us": med,
                "weight_bytes": bytes_moved,
                "achieved_bandwidth_gbs": bytes_moved / med / 1e3,
                "ref_sha256": hashlib.sha256(
                    ref.tobytes()).hexdigest()[:16],
            }
            if name == "down_warp256":
                inc = _bf16_to_f32(_download(_Buf(out_down.ptr, hidden * 2), (hidden,), np.uint16, runtime))
                cand = _bf16_to_f32(_download(_Buf(out_down_v2.ptr, hidden * 2), (hidden,), np.uint16, runtime))
                diff = np.abs(cand - inc)
                case["drift_abs_max"] = float(diff.max())
                case["drift_abs_p999"] = float(np.quantile(diff, 0.999))
            if name == "gate_up_warp256":
                inc = _bf16_to_f32(_download(_Buf(out_gu.ptr, top_k * ffn * 2), (top_k * ffn,), np.uint16, runtime))
                cand = _bf16_to_f32(_download(_Buf(out_gu_v2.ptr, top_k * ffn * 2), (top_k * ffn,), np.uint16, runtime))
                diff = np.abs(cand - inc)
                case["drift_abs_max"] = float(diff.max())
                case["drift_abs_p999"] = float(np.quantile(diff, 0.999))
            report["cases"].append(case)
            print(
                f"{name:14s} {med:8.1f} us  {bytes_moved/1e6:5.2f} MB  "
                f"{case['achieved_bandwidth_gbs']:6.1f} GB/s  "
                f"ref {case['ref_sha256']}"
            )
        report["status"] = "passed"
    finally:
        while allocations:
            free(allocations.pop())

    a.output.write_text(json.dumps(report, indent=1) + "\n")
    print(f"wrote {a.output}")


def _bf16_to_f32(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << 16).view(np.float32)


class _Buf:
    """Minimal ptr/nbytes adapter for _download."""

    def __init__(self, ptr: int, nbytes: int = 0) -> None:
        self.ptr = ptr
        self.nbytes = nbytes


def _bf16_bits(x: np.ndarray) -> np.ndarray:
    u32 = x.astype(np.float32).view(np.uint32)
    lsb = (u32 >> 16) & 1
    u32 = u32 + 0x7FFF + lsb
    return (u32 >> 16).astype(np.uint16)


if __name__ == "__main__":
    main()
