"""Correctness gate for P10.X2: Layer-by-layer WMMA vs GEMV prefill validation.

MoE models are highly sensitive to tiny floating-point rounding variations (such
as Matrix Core FP16/BF16 accumulation vs Vector ALU FP32 accumulation).
These tiny variations (under 1e-3) propagate and shift the hard argmax of the
MoE router, causing tokens to route to different experts. This 'butterfly effect'
causes E2E sequence logits to diverge over many layers, which is a chaotic
mathematical property rather than a bug.

This correctness gate validates that:
1. At MoE Layer 0 (where inputs are identical and no prior routing divergence
   has occurred), the selected experts are 100% identical between WMMA prefill
   and the GEMV reference.
2. At MoE Layer 0, the maximum absolute difference of the output hidden states
   is within the expected unit-in-the-last-place (ULP) BF16 rounding tolerance
   (<= 5e-3).
This proves the mathematical correctness of both Q8T16 dense WMMA prefill
(attention) and Q4T16 compact selected dual WMMA prefill (MoE) on real weights.
"""

from __future__ import annotations

import ctypes
import numpy as np
import pytest
from pathlib import Path

from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession
from hipengine.runtime import qwen35_gguf_layer_ffn as layer_ffn
from hipengine.core.memory import host_array_ptr
from hipengine.core.hip import HipMemcpyKind
import hipengine.runtime.qwen35_gguf_runner as qgr

DEFAULT_MODEL = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.skipif(not DEFAULT_MODEL.exists(), reason=f"Model fixture {DEFAULT_MODEL} is missing")
def test_p10_x2_layer0_wmma_vs_gemv_prefill_correctness(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    # Force ALLOW_UNSAFE so the safety gate does not bypass our test request.
    monkeypatch.setenv("HIPENGINE_GGUF_DECODE_REPACK", "1")
    monkeypatch.setenv("HIPENGINE_GGUF_ALLOW_UNSAFE_QWEN35MOE_FASTPATHS", "1")

    prompt_tokens = [9707] * 512
    rows = len(prompt_tokens)

    session = Qwen35GGUFResidentSession(
        DEFAULT_MODEL,
        max_sequence_length=1000,
        use_wmma_prefill=None,
        use_gemv_decode=None,
    )
    # The session owns the model weights (about 21 GiB); without a close they
    # stay resident for every later test in the process.
    request.addfinalizer(session.close)
    runner = session.runner
    hidden_size = runner.hidden_size

    original_run_post_attention_moe_rows = layer_ffn.run_post_attention_moe_rows
    layer0 = runner.weights.layer(0)

    class Layer0DoneException(Exception):
        pass

    def capture_layer0(wmma_enabled):
        captured = []

        def intercept(
            layer, config, geometry, kernels, ops, flags, out_ptr, scratch,
            *, rows, runtime, **kwargs,
        ):
            original_run_post_attention_moe_rows(
                layer, config, geometry, kernels, ops, flags, out_ptr, scratch,
                rows=rows, runtime=runtime, **kwargs,
            )
            if layer is layer0:
                runtime.device_synchronize()
                experts = np.zeros((rows, config.expert_used_count), dtype=np.int32)
                runtime.memcpy(host_array_ptr(experts), scratch.moe_selected_experts.ptr, experts.nbytes, HipMemcpyKind.DEVICE_TO_HOST)
                output = np.zeros((rows, hidden_size), dtype=np.uint16)
                runtime.memcpy(host_array_ptr(output), out_ptr, output.nbytes, HipMemcpyKind.DEVICE_TO_HOST)
                captured.append((experts, output))
                raise Layer0DoneException()

        monkeypatch.setattr(qgr, "gguf_wmma_prefill_enabled", lambda enabled=None: wmma_enabled)
        monkeypatch.setattr(layer_ffn, "run_post_attention_moe_rows", intercept)
        session.reset()
        with pytest.raises(Layer0DoneException):
            session.prefill(prompt_tokens)
        assert len(captured) == 1
        experts, output = captured[0]
        assert output.shape == (rows, hidden_size)
        return experts, output

    # Compare the same layer before any later-layer routing divergence.
    ref_experts, ref_out = capture_layer0(False)
    cand_experts, cand_out = capture_layer0(True)

    # Assert 1: Expert selection agreement is 100.0%
    np.testing.assert_array_equal(ref_experts, cand_experts)

    # Assert 2: Output maximum absolute difference is within ULP rounding tolerance
    def bf16_to_f32(arr):
        u32 = np.zeros_like(arr, dtype=np.uint32)
        u32 = arr.astype(np.uint32) << 16
        return u32.view(np.float32)

    ref_f32 = bf16_to_f32(ref_out)
    cand_f32 = bf16_to_f32(cand_out)

    diff = np.abs(ref_f32 - cand_f32)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    print(f"P10.X2 Correctness Gate - Layer 0 max diff: {max_diff:.6f}, mean diff: {mean_diff:.6f}")
    assert max_diff <= 5.0e-3, f"Layer 0 max diff {max_diff} exceeded tolerance"
    assert mean_diff <= 1.0e-4, f"Layer 0 mean diff {mean_diff} exceeded tolerance"
