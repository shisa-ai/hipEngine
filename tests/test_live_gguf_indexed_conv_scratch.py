"""Public packed decode keeps the indexed Conv input live during its write."""
from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
import pytest

MODEL = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")


def _hip_available():
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not (_hip_available() and MODEL.exists()),
    reason="requires ROCm and the MoE UD-Q4_K_M artifact",
)


@pytest.mark.parametrize("width", [2, 3, 4])
def test_public_indexed_conv_uses_disjoint_planes_and_finite_output(monkeypatch, width):
    from hipengine import LLM, SamplingParams
    from hipengine.core.hip import HipMemcpyKind, get_hip_runtime
    from hipengine.core.memory import memory_stats
    from hipengine.runtime import qwen35_gguf_runner as module

    runtime = get_hip_runtime()
    runtime.device_synchronize()
    before = memory_stats()
    original = module.Qwen35GGUFFullStackRunner._run_linear_attention_attn_rows_indexed_exact
    observed = []

    def checked(self, layer_id, hidden_ptr, out_ptr, scratch, **kwargs):
        rows = kwargs["rows"]
        qkv_start = int(scratch.linear_qkv.ptr)
        conv_start = int(scratch.conv_out.ptr)
        qkv_end = qkv_start + rows * self.linear_qkv_width * 2
        conv_end = conv_start + rows * self.linear_qkv_width * 4
        assert qkv_end <= conv_start or conv_end <= qkv_start
        result = original(self, layer_id, hidden_ptr, out_ptr, scratch, **kwargs)
        runtime.device_synchronize()
        bits = np.empty(rows * self.hidden_size, dtype=np.uint16)
        runtime.memcpy(bits.ctypes.data, int(out_ptr), bits.nbytes,
                       HipMemcpyKind.DEVICE_TO_HOST)
        values = (bits.astype(np.uint32) << 16).view(np.float32)
        assert np.isfinite(values).all()
        observed.append((layer_id, rows, scratch.allocation_mode))
        return result

    monkeypatch.setattr(module.Qwen35GGUFFullStackRunner,
                        "_run_linear_attention_attn_rows_indexed_exact", checked)
    llm = LLM(str(MODEL), backend="hip_gfx1151", execution_profile="production",
              max_sequence_length=1024, max_active_requests=width, prefix_cache="off")
    try:
        prompts = [[1] * length for length in (17, 23, 37, 53)[:width]]
        outputs = llm.generate_detailed(
            prompts, SamplingParams(max_tokens=4, temperature=0, ignore_eos=True))
        assert len(outputs) == width
        assert all(len(output.generated_token_ids) == 4 for output in outputs)
        assert observed, "public generation did not reach indexed decode"
        assert any(rows == width and mode == "liveness_aliased"
                   for _, rows, mode in observed)
    finally:
        llm.close()
    runtime.device_synchronize()
    after = memory_stats()
    for field in ("current_allocated_bytes", "active_allocations"):
        assert after[field] == before[field]
