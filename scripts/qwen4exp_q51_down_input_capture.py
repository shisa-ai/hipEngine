#!/usr/bin/env python3
"""Capture actual Q5_1-down input activations from a live prefill.

Runs one canonical case prefill with the production grouped MoE route and
monkeypatches the Q5_1 down parent kernel entry to snapshot the
expert-sorted intermediate (SiLU(gate)*up) activations for the first
Q5_1-down layer encountered, together with the compact expert starts.
Output: .npz with x bits, expert_start, and metadata. Diagnostic only; no
runtime default changes.
"""

import argparse
import ctypes
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import copy_device_to_host, host_array_ptr
from hipengine.execution_profiles import ExecutionProfile, resolve_runtime_profile
from hipengine.generation.qwen4_exp_gguf import Qwen4ExpGGUFTextGenerator
from hipengine.generation.qwen4_exp_profiles import (
    register_qwen4_exp_gfx1151_profiles, QWEN4_EXP_MODEL,
    QWEN4_EXP_BACKEND, QWEN4_EXP_QUANTS,
)
from hipengine.kernels.hip_gfx1100.quant import qwen4_exp_q5_1 as q51
from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
from hipengine.loading.gguf import discover_gguf_files, load_gguf_index
from hipengine.models import resolve_model
from scripts.qwen4exp_canonical_ar_bench import DEFAULT_FIXTURE, load_fixture


def hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
        return True
    except OSError:
        return False


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-root", type=Path, required=True)
    p.add_argument("--case-id", default="general_en-p4096")
    p.add_argument("--call-index", type=int, default=0,
                    help="which Q5_1 down call to capture (0-based)")
    p.add_argument("--compiler-version-file", type=Path, required=True)
    p.add_argument("--require-cached-build", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(a.compiler_version_file)
    if a.require_cached_build:
        os.environ["HIPENGINE_REQUIRE_CACHED_BUILD"] = "1"

    runtime = get_hip_runtime()
    captured = {}
    seen = [0]

    original = q51.qwen4_exp_q5_1_selected_grouped_prefill_pair2_row_publish_bf16_bf16_out

    from hipengine.kernels.registry import KernelKey, register
    hook_key = KernelKey(
        "hip_gfx1151", "moe_linear", "gguf_q5_1",
        "selected_grouped_prefill_pair2_row_publish_bf16_bf16_out")

    def hooked(input_ptr, expert_start_ptr, weights_ptr, output_ptr,
               compact_rows, num_experts, in_features, out_features, **kw):
        if "x" not in captured and in_features == 640:
            if seen[0] < a.call_index:
                seen[0] += 1
                return original(input_ptr, expert_start_ptr, weights_ptr,
                                output_ptr, compact_rows, num_experts,
                                in_features, out_features, **kw)
            from hipengine.core.memory import DeviceBuffer
            n = int(compact_rows) * int(in_features)
            x = np.empty(n, dtype=np.uint16)
            copy_device_to_host(
                host_array_ptr(x), DeviceBuffer(int(input_ptr), n * 2),
                runtime=runtime)
            starts = np.empty(int(num_experts) + 1, dtype=np.int64)
            copy_device_to_host(
                host_array_ptr(starts),
                DeviceBuffer(int(expert_start_ptr), (int(num_experts) + 1) * 8),
                runtime=runtime)
            captured["x"] = x.reshape(int(compact_rows), int(in_features))
            nbytes = int(num_experts) * int(out_features) * (int(in_features) // 32) * 24
            w = np.empty(nbytes, dtype=np.uint8)
            copy_device_to_host(
                host_array_ptr(w), DeviceBuffer(int(weights_ptr), nbytes),
                runtime=runtime)
            captured["weights"] = w
            captured["expert_start"] = starts
            captured["out_features"] = int(out_features)
        return original(input_ptr, expert_start_ptr, weights_ptr, output_ptr,
                        compact_rows, num_experts, in_features, out_features,
                        **kw)

    q51.qwen4_exp_q5_1_selected_grouped_prefill_pair2_row_publish_bf16_bf16_out = hooked

    register_gfx1151_kernels(replace=True)
    register(hook_key, hooked, replace=True)
    register_qwen4_exp_gfx1151_profiles()
    resolved = resolve_runtime_profile(
        model=QWEN4_EXP_MODEL, backend=QWEN4_EXP_BACKEND,
        quant=QWEN4_EXP_QUANTS[1], profile=ExecutionProfile.PRODUCTION)
    fixture, _ = load_fixture(DEFAULT_FIXTURE)
    case = next(c for c in fixture["cases"] if c["id"] == a.case_id)
    index = load_gguf_index(discover_gguf_files(a.model_root)[0])
    generator = resolved.construct_generator(lambda: Qwen4ExpGGUFTextGenerator(
        model_path=a.model_root, weight_index=index,
        model_plugin=resolve_model(index.architecture or ""),
        backend="hip_gfx1151", max_sequence_length=4352,
        prefill_chunk_size=1024))
    try:
        generator.runner.prefill(case["prompt_token_ids"])
        assert captured, "no Q5_1 down call captured"
        a.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            a.output,
            x_bits=captured["x"],
            expert_start=captured["expert_start"],
            weights=captured["weights"],
            out_features=captured["out_features"],
            case_id=case["id"],
        )
        print(f"captured {captured['x'].shape} activations -> {a.output}")
    finally:
        generator.close()


if __name__ == "__main__":
    main()
