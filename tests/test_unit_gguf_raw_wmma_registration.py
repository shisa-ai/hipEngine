"""Optional raw WMMA rewrites require the exact device launch registration."""
import pytest

from hipengine.kernels.registry import KernelKey
from hipengine.runtime import gguf_linear as gl


@pytest.mark.parametrize("present", [False, True])
def test_raw_wmma_rewrite_requires_exact_gpu_key(monkeypatch, present):
    parent = gl.GGUFLinearDispatch(
        KernelKey("hip_gfx1151", "linear", "gguf_q6_k", "prefill_bf16_f32_out"), "raw",
    )
    candidate = KernelKey("hip_gfx1151", "linear", "gguf_q6_k", "wmma_prefill_bf16_f32_out")
    monkeypatch.setattr(gl, "_ensure_linear_kernel_registered", lambda key: None)
    monkeypatch.setattr(gl, "is_registered", lambda key: key == candidate and present)
    result = gl._wmma_prefill_dispatch(parent, rows=3, in_features=256, use_wmma=True)
    assert result == (gl.GGUFLinearDispatch(candidate, "wmma_raw") if present else parent)
