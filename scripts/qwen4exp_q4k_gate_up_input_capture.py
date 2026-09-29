#!/usr/bin/env python3
"""Capture actual Q4_K gate/up input activations from a live prefill.

MX-1's operand-admissibility packet covers ``expert_down`` on a real
natural-input capture, but the gate/up pair (Q4_K, 94 of the 144 expert tensors)
has weight-decode coverage only: no real gate/up input activation was captured,
so its activation-cast term is unmeasured. This closes that gap.

Runs one canonical case prefill with the production grouped MoE route and
monkeypatches the Q4_K gate/up parent kernel entry to snapshot the MoE block's
input hidden state (the tensor both gate and up read) for the first Q4_K gate/up
layer encountered, together with the compact expert starts. Output: .npz with
x bits, expert_start, and metadata. Diagnostic only; no runtime default changes.
"""

import argparse
import ctypes
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import DeviceBuffer, copy_device_to_host, host_array_ptr
from hipengine.execution_profiles import ExecutionProfile, resolve_runtime_profile
from hipengine.generation.qwen4_exp_gguf import Qwen4ExpGGUFTextGenerator
from hipengine.generation.qwen4_exp_profiles import (
    register_qwen4_exp_gfx1151_profiles,
    QWEN4_EXP_BACKEND,
    QWEN4_EXP_MODEL,
    QWEN4_EXP_QUANTS,
)
from hipengine.kernels.hip_gfx1100.quant import gguf_q4_k_selected_prefill as q4k
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--case-id", default="general_en-p4096")
    parser.add_argument("--call-index", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not hip_available():
        raise SystemExit("HIP unavailable")

    runtime = get_hip_runtime()
    captured: dict[str, object] = {}
    seen = [0]

    parent = q4k.gguf_q4_k_selected_dual_wmma_iu8_risk_prefill_bf16_bf16_out

    def hooked(x_ptr, expert_start_compact_ptr, expert_start_wmma_ptr,
               tile_expert_ptr, qweight_a_ptr, qweight_b_ptr, out_ptr,
               risk_count_ptr, risk_indices_ptr, max_risks, risk_multiplier,
               compact_rows, in_features, out_features_a, out_features_b,
               num_experts, wmma_total_rows, **kw):
        if "x" not in captured and int(in_features) == 2560:
            if seen[0] < args.call_index:
                seen[0] += 1
                return parent(
                    x_ptr, expert_start_compact_ptr, expert_start_wmma_ptr,
                    tile_expert_ptr, qweight_a_ptr, qweight_b_ptr, out_ptr,
                    risk_count_ptr, risk_indices_ptr, max_risks, risk_multiplier,
                    compact_rows, in_features, out_features_a, out_features_b,
                    num_experts, wmma_total_rows, **kw)
            n = int(compact_rows) * int(in_features)
            x = np.empty(n, dtype=np.uint16)
            copy_device_to_host(
                host_array_ptr(x), DeviceBuffer(int(x_ptr), n * 2), runtime=runtime)
            starts = np.empty(int(num_experts) + 1, dtype=np.int64)
            copy_device_to_host(
                host_array_ptr(starts),
                DeviceBuffer(int(expert_start_compact_ptr), (int(num_experts) + 1) * 8),
                runtime=runtime)
            captured["x"] = x.reshape(int(compact_rows), int(in_features))
            captured["expert_start"] = starts
            captured["in_features"] = int(in_features)
            captured["out_features"] = int(out_features_a)
            captured["num_experts"] = int(num_experts)
        return parent(
            x_ptr, expert_start_compact_ptr, expert_start_wmma_ptr,
            tile_expert_ptr, qweight_a_ptr, qweight_b_ptr, out_ptr,
            risk_count_ptr, risk_indices_ptr, max_risks, risk_multiplier,
            compact_rows, in_features, out_features_a, out_features_b,
            num_experts, wmma_total_rows, **kw)

    q4k.gguf_q4_k_selected_dual_wmma_iu8_risk_prefill_bf16_bf16_out = hooked
    register_gfx1151_kernels(replace=True)
    from hipengine.kernels.registry import KernelKey, register

    register(
        KernelKey("hip_gfx1151", "moe_linear", "gguf_q4_k",
                  "selected_dual_wmma_iu8_risk_prefill_bf16_bf16_out"),
        hooked, replace=True)
    register_qwen4_exp_gfx1151_profiles()
    resolved = resolve_runtime_profile(
        model=QWEN4_EXP_MODEL, backend=QWEN4_EXP_BACKEND,
        quant=QWEN4_EXP_QUANTS[1], profile=ExecutionProfile.PRODUCTION)
    fixture, _ = load_fixture(DEFAULT_FIXTURE)
    case = next(c for c in fixture["cases"] if c["id"] == args.case_id)
    index = load_gguf_index(discover_gguf_files(args.model_root)[0])
    generator = resolved.construct_generator(lambda: Qwen4ExpGGUFTextGenerator(
        model_path=args.model_root, weight_index=index,
        model_plugin=resolve_model(index.architecture or ""),
        backend="hip_gfx1151", max_sequence_length=4352, prefill_chunk_size=1024))
    try:
        generator.runner.prefill(case["prompt_token_ids"])
        assert captured, "no Q4_K gate/up call captured"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.output,
            x_bits=captured["x"],
            expert_start=captured["expert_start"],
            in_features=captured["in_features"],
            out_features=captured["out_features"],
            num_experts=captured["num_experts"],
            case_id=case["id"],
        )
        print(f"captured {captured['x'].shape} gate/up activations -> {args.output}")
    finally:
        generator.close()


if __name__ == "__main__":
    main()
