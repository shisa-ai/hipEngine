#!/usr/bin/env python3
"""Ordered scratch-trigger screen for representative Qwen4Exp decode victims.

Each invocation is a fresh process with a sham or scratch arm.  The victims use
real Qwen3.8-Flash-Next UD-Q4_K_XL expert weights and deterministic synthetic
activations/routes:

* ``q4-gate-up`` is the Q4_K dual gate/up + SiLU DP4A decode owner;
* ``q8-down`` is the Q8_0 selected expert-down GEMV owner.

The trigger is either the first scratch-using GDN dispatch (36-byte private
segment) or the dense-wide dispatch (92-byte private segment).  This is a
causality screen, not a serving benchmark or a promotion gate.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from hipengine.core.memory import free
from hipengine.kernels.hip_gfx1100.linear_attn import qwen4_exp_gdn as gdn
from hipengine.kernels.hip_gfx1100.quant import gguf_q8_0_dense_wide as dense
from hipengine.kernels.hip_gfx1100.quant.gguf_k_gemv import (
    build_gguf_k_gemv,
    gguf_q8_0_selected_gemv_bf16_bf16_out,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_gemv import (
    build_gguf_q4_k_gemv,
    gguf_q4_k_quantize_bf16_q8_1,
    gguf_q4_k_selected_dual_q8_1_dp4a_silu_logical128_t64_gemv_bf16_bf16_out,
)
from hipengine.loading.gguf import GGUFReader, discover_gguf_files
from scripts.qwen4exp_moe_gemv_decode_screen import _bf16_bits
from tests.test_gpu_gguf_q8_0_dense_wide import (
    _dequantize_weight,
    _f16_reference,
    make_q8_0_weight,
)
from tests.test_gpu_qwen4_exp_pf3_moe_schedules import _alloc, _download, _upload
from tests.test_gpu_qwen4exp_gdn_register import Fixture


Q4_GATE_NAME = "blk.4.ffn_gate_exps.weight"
Q4_UP_NAME = "blk.4.ffn_up_exps.weight"
Q8_DOWN_NAME = "blk.4.ffn_down_exps.weight"


def hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def clock_snapshot() -> dict[str, str]:
    values: dict[str, str] = {}
    for path in Path("/sys/class/drm").glob("card[0-9]*/device/pp_dpm_*clk"):
        try:
            values[str(path)] = path.read_text().strip()
        except OSError:
            pass
    return values


def measure_window(runtime, start, stop, launch, repeats) -> list[float]:
    """Time queued launches without putting host synchronization in the interval."""
    samples = []
    for _ in range(repeats):
        runtime.device_synchronize()
        runtime.event_record(start)
        launch()
        runtime.event_record(stop)
        runtime.event_synchronize(stop)
        samples.append(runtime.event_elapsed_time_ms(start, stop))
    return samples


def launch_dense_trigger(x_ptr, weight_ptr, out_ptr, *, library, runtime) -> None:
    """Launch the original scratch-using dense-wide tail, not the padded leaf."""
    dense.gguf_q8_0_dense_wide256_f32_f32_out(
        x_ptr, weight_ptr, out_ptr, 257, 256, 256, library=library, runtime=runtime,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--arm", choices=("sham", "scratch"), required=True)
    parser.add_argument("--victim", choices=("q4-gate-up", "q8-down"), required=True)
    parser.add_argument("--trigger", choices=("gdn", "dense-wide"), required=True)
    parser.add_argument("--repeats", type=int, default=31)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 3:
        parser.error("at least three repeats required")
    if not hip_available():
        parser.error("HIP runtime unavailable")

    fixture = Fixture(64)
    runtime = fixture.runtime
    q4_library = build_gguf_q4_k_gemv(load=True)
    q8_library = build_gguf_k_gemv(load=True)
    readers = [GGUFReader(path) for path in discover_gguf_files(args.model_root)]

    def tensor(name: str):
        reader = next(r for r in readers if any(t.name == name for t in r.info.tensors))
        return reader.tensor_data(name), reader.tensor_info(name)

    gate_raw, gate_info = tensor(Q4_GATE_NAME)
    up_raw, up_info = tensor(Q4_UP_NAME)
    down_raw, down_info = tensor(Q8_DOWN_NAME)
    expected_q4 = ((512, 640, 2560), "Q4_K")
    expected_q8 = ((512, 2560, 640), "Q8_0")
    if tuple(gate_info.shape) != expected_q4[0] or gate_info.ggml_type_name != expected_q4[1]:
        raise ValueError(f"unexpected gate tensor: {gate_info.shape}, {gate_info.ggml_type_name}")
    if tuple(up_info.shape) != expected_q4[0] or up_info.ggml_type_name != expected_q4[1]:
        raise ValueError(f"unexpected up tensor: {up_info.shape}, {up_info.ggml_type_name}")
    if tuple(down_info.shape) != expected_q8[0] or down_info.ggml_type_name != expected_q8[1]:
        raise ValueError(f"unexpected down tensor: {down_info.shape}, {down_info.ggml_type_name}")

    allocations = []
    start = stop = None
    dense_library = None
    try:
        rng = np.random.default_rng(20261007)
        experts, top_k, hidden, ffn = 512, 10, 2560, 640
        selected = np.sort(rng.choice(experts, size=top_k, replace=False)).astype(np.int64)
        dselected = _upload(selected, runtime, allocations)
        # Decode gate/up consumes one hidden row and emits one row per selected expert.
        hidden_bf16 = _bf16_bits(rng.normal(0, 1.0, size=(1, hidden)).astype(np.float32))
        dhidden = _upload(hidden_bf16.reshape(-1), runtime, allocations)
        dq8 = _alloc(4096 * 64, np.uint8, runtime, allocations)
        dgate = _upload(gate_raw, runtime, allocations)
        dup = _upload(up_raw, runtime, allocations)
        q4_out = _alloc(top_k * ffn, np.uint16, runtime, allocations)
        # Decode down consumes the ten post-SiLU rows and emits ten selected outputs.
        down_input = _bf16_bits(rng.normal(0, 1.0, size=(top_k, ffn)).astype(np.float32))
        ddown_input = _upload(down_input.reshape(-1), runtime, allocations)
        ddown = _upload(down_raw, runtime, allocations)
        q8_out = _alloc(top_k * hidden, np.uint16, runtime, allocations)
        start, stop = runtime.event_create(), runtime.event_create()
        # Prepare both arms identically; never execute the trigger before baseline.
        dense_ref = None
        if args.trigger == "dense-wide":
            dense_library = dense.build_gguf_q8_0_dense_wide(load=True)
            dense_x = rng.normal(0, 0.1, (257, 256)).astype(np.float32)
            dense_w = make_q8_0_weight(256, 256)
            dense_ref = _f16_reference(dense_x, _dequantize_weight(dense_w, 256))
            dense_dx = _upload(dense_x, runtime, allocations)
            dense_dw = _upload(dense_w, runtime, allocations)
            dense_do = _alloc(257 * 256, np.float32, runtime, allocations)

        def run_q4() -> None:
            gguf_q4_k_quantize_bf16_q8_1(
                dhidden.ptr, dq8.ptr, 1, hidden, library=q4_library, runtime=runtime
            )
            gguf_q4_k_selected_dual_q8_1_dp4a_silu_logical128_t64_gemv_bf16_bf16_out(
                dq8.ptr, dselected.ptr, dgate.ptr, dup.ptr, q4_out.ptr,
                1, top_k, experts, hidden, ffn, library=q4_library, runtime=runtime,
            )

        def run_q8() -> None:
            gguf_q8_0_selected_gemv_bf16_bf16_out(
                ddown_input.ptr, dselected.ptr, ddown.ptr, q8_out.ptr,
                top_k, top_k, experts, ffn, hidden, library=q8_library, runtime=runtime,
            )

        victim_fn = run_q4 if args.victim == "q4-gate-up" else run_q8
        victim_out = q4_out if args.victim == "q4-gate-up" else q8_out
        victim_shape = (top_k, ffn) if args.victim == "q4-gate-up" else (top_k, hidden)
        for _ in range(3):
            victim_fn()
        runtime.device_synchronize()
        expected = _download(victim_out, victim_shape, np.uint16, runtime)
        if not np.isfinite((expected.astype(np.uint32) << 16).view(np.float32)).all():
            raise AssertionError("nonfinite victim output")
        clocks = {"before": clock_snapshot()}
        before = measure_window(runtime, start, stop, victim_fn, args.repeats)
        clocks["pre-trigger"] = clock_snapshot()

        if args.arm == "scratch":
            if args.trigger == "gdn":
                gdn.qwen4_exp_gdn_wave_norm_prefill_f32(
                    *[p.ptr for p in fixture.inputs], fixture.states[1].ptr,
                    fixture.outputs[1].ptr, 64, 16, 48, 128, 128,
                    library=fixture.library, runtime=runtime,
                )
                runtime.device_synchronize()
            else:
                launch_dense_trigger(
                    dense_dx.ptr, dense_dw.ptr, dense_do.ptr,
                    library=dense_library, runtime=runtime,
                )
        else:
            victim_fn()
        runtime.device_synchronize()
        clocks["post-trigger"] = clock_snapshot()
        after = measure_window(runtime, start, stop, victim_fn, args.repeats)
        clocks["after"] = clock_snapshot()
        actual = _download(victim_out, victim_shape, np.uint16, runtime)
        np.testing.assert_array_equal(actual, expected)

        dense_check = None
        if args.arm == "scratch" and args.trigger == "dense-wide":
            dense_out = _download(dense_do, (257, 256), np.float32, runtime)
            error = float(np.max(np.abs(dense_out - dense_ref)))
            scale = float(np.max(np.abs(dense_ref)))
            if not np.isfinite(error) or error > 1e-3 * scale:
                raise AssertionError(f"dense trigger reference failed: {error} vs {scale}")
            dense_check = {
                "f16_reference_pass": True, "max_abs_error": error,
                "reference_scale": scale, "shape": [257, 256, 256],
            }
        trigger_exact = None
        if args.arm == "scratch" and args.trigger == "gdn":
            fixture.run(False)
            for got, ref in zip(fixture.result(True), fixture.result(False)):
                np.testing.assert_array_equal(got.view(np.uint32), ref.view(np.uint32))
            trigger_exact = True
        report = {
            "schema": 2,
            "kind": "qwen4exp_decode_victim_scratch_screen",
            "timing": "HIP events enclosing async victim launches; sync only outside interval",
            "protocol": "3 warmups, before window, one trigger/sham, after window; validation after windows",
            "environment": {key: os.environ.get(key) for key in (
                "HIPENGINE_HIP_ARCH", "HIPENGINE_REQUIRE_CACHED_BUILD",
                "HIPENGINE_COMPILER_VERSION_FILE", "HIP_PATH", "ROCM_PATH",
                "LD_LIBRARY_PATH", "PATH", "OPENBLAS_NUM_THREADS",
            )},
            "command": sys.argv,
            "arm": args.arm,
            "victim": args.victim,
            "trigger": args.trigger if args.arm == "scratch" else "victim",
            "requested_trigger": args.trigger,
            "model_root": str(args.model_root),
            "model": "unsloth Qwen3.8-Flash-Next UD-Q4_K_XL",
            "geometry": {"experts": experts, "top_k": top_k, "hidden": hidden, "ffn": ffn},
            "tensors": {
                "gate": {"name": Q4_GATE_NAME, "quant": gate_info.ggml_type_name,
                         "shape": list(gate_info.shape), "sha256": hashlib.sha256(gate_raw).hexdigest()},
                "up": {"name": Q4_UP_NAME, "quant": up_info.ggml_type_name,
                       "shape": list(up_info.shape), "sha256": hashlib.sha256(up_raw).hexdigest()},
                "down": {"name": Q8_DOWN_NAME, "quant": down_info.ggml_type_name,
                         "shape": list(down_info.shape), "sha256": hashlib.sha256(down_raw).hexdigest()},
            },
            "before_ms": before,
            "after_ms": after,
            "after_over_before": statistics.median(after) / statistics.median(before),
            "victim_output_exact": True,
            "trigger_parent_exact": trigger_exact,
            "output_sha256": hashlib.sha256(actual.tobytes()).hexdigest(),
            "dense_reference_check": dense_check,
            "clocks": clocks,
            "libraries": {
                lib._name: hashlib.sha256(Path(lib._name).read_bytes()).hexdigest()
                for lib in (q4_library, q8_library, fixture.library, dense_library)
                if lib is not None
            },
            "host": {"machine_id": Path("/etc/machine-id").read_text().strip(), "uname": list(os.uname())},
            "performance_claim": False,
            "limits": [
                "single real-weight expert bank per victim and synthetic decode activations/routes",
                "boundary clock snapshots only",
                "does not measure serving wall or establish universal scratch-throttle absence",
            ],
        }
    finally:
        if start is not None:
            runtime.event_destroy(start)
        if stop is not None:
            runtime.event_destroy(stop)
        for buffer in reversed(allocations):
            free(buffer, runtime=runtime)
        fixture.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"victim": args.victim, "arm": args.arm,
                      "trigger": args.trigger, "ratio": report["after_over_before"],
                      "exact": True}), flush=True)


if __name__ == "__main__":
    main()
