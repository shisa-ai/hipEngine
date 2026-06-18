"""RED test for Q6_K gemv adapter and shared_head_head dispatch.

M6: Validates Q6_K gemv adapter on gfx1151, matching cpu_reference.
The real model's output.weight (tied to shared_head_head) is Q6_K [248320, 2048].
"""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tests"))


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


pytestmark = pytest.mark.skipif(not _hip_available(), reason="ROCm/HIP not available")

import hipengine.kernels.cpu_reference  # noqa: F401,E402
import hipengine.kernels.hip_gfx1151  # noqa: F401,E402

from hipengine.quant.gguf import GGMLQuantizationType  # noqa: E402

# Import weight factory helpers from the existing K-quant gemv test
import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location(
    "_gguf_k_gemv_helpers", REPO_ROOT / "tests" / "test_gguf_k_gemv.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
make_q6_k_weight = _mod.make_q6_k_weight


# ---------------------------------------------------------------------------
# Q6_K gemv adapter: direct hip kernel vs cpu_reference
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("in_features,out_features", [(256, 128), (2048, 512)])
def test_q6_k_gemv_adapter_matches_cpu_reference(in_features, out_features):
    from hipengine.kernels.cpu_reference import gguf_q6_k_gemv
    from hipengine.kernels.hip_gfx1100.quant.gguf_q6_k_pack8_gemv import (
        build_gguf_q6_k_pack8_gemv,
    )
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import (
        malloc, copy_host_to_device, copy_device_to_host,
        host_array_ptr, free,
    )

    rng = np.random.default_rng(42)
    x = rng.standard_normal((1, in_features)).astype(np.float32)
    x /= np.sqrt(in_features)

    weight = make_q6_k_weight(out_features, in_features)
    assert weight.dtype == np.uint8

    # cpu_reference
    cpu_out = gguf_q6_k_gemv(x, weight)
    assert cpu_out.shape == (1, out_features)

    # GPU: Q6_K pack8 only supports bf16/fp16 input. Use fp16 variant.
    x_fp16 = x.astype(np.float16)
    from hipengine.kernels.hip_gfx1100.quant.gguf_q6_k_pack8_gemv import (
        gguf_q6_k_pack8_gemv_decode_fp16_f32_out as _q6_k_launch,
    )

    runtime = get_hip_runtime()
    lib = build_gguf_q6_k_pack8_gemv(load=True)

    x_dev = malloc(x_fp16.nbytes, runtime=runtime)
    w_dev = malloc(weight.nbytes, runtime=runtime)
    out_dev = malloc(out_features * 4, runtime=runtime)
    copy_host_to_device(x_dev, host_array_ptr(x_fp16), runtime=runtime)
    copy_host_to_device(w_dev, host_array_ptr(weight), runtime=runtime)

    _q6_k_launch(x_dev.ptr, w_dev.ptr, out_dev.ptr, 1, in_features, out_features,
                 runtime=runtime, library=lib)
    runtime.device_synchronize()

    gpu_out = np.empty((1, out_features), dtype=np.float32)
    copy_device_to_host(host_array_ptr(gpu_out), out_dev, runtime=runtime)

    free(x_dev, runtime=runtime)
    free(w_dev, runtime=runtime)
    free(out_dev, runtime=runtime)

    assert gpu_out.shape == (1, out_features)
    max_abs = float(np.max(np.abs(cpu_out - gpu_out)))
    assert max_abs < 5e-3, f"Q6_K gemv max_abs={max_abs} exceeds 5e-3 (fp16 input tolerance)"


# ---------------------------------------------------------------------------
# Q6_K in _dispatch_gemv (FFN) and _attn_dispatch_gemv (attention)
# ---------------------------------------------------------------------------

def test_ffn_dispatch_q6_k():
    """FFN _dispatch_gemv should support Q6_K (currently raises NotImplementedError)."""
    from hipengine.kernels.hip_gfx1100.speculative.mtp_nextn import (
        qwen35_gguf_mtp_ffn_sublayer_f32 as gpu_ffn,
    )
    from hipengine.kernels.cpu_reference.ops import qwen35_gguf_mtp_ffn_sublayer as cpu_ffn

    hidden = 256
    inter = 512
    experts = 2
    top_k = 1

    rng = np.random.default_rng(123)
    x = rng.standard_normal((1, hidden)).astype(np.float32) * 0.1

    # Q6_K weights (rank-3 [E, out, block_bytes])
    gate_w = np.stack([make_q6_k_weight(inter, hidden) for _ in range(experts)])
    up_w = np.stack([make_q6_k_weight(inter, hidden) for _ in range(experts)])
    down_w = np.stack([make_q6_k_weight(hidden, inter) for _ in range(experts)])

    router = rng.standard_normal((experts, hidden)).astype(np.float32) * 0.1
    post_norm = np.ones(hidden, dtype=np.float32)  # simple post-attention norm
    shared_gate_logit = rng.standard_normal(hidden).astype(np.float32) * 0.1
    shared_gate_w = rng.standard_normal((inter, hidden)).astype(np.float32) * 0.01
    shared_up_w = rng.standard_normal((inter, hidden)).astype(np.float32) * 0.01
    shared_down_w = rng.standard_normal((hidden, inter)).astype(np.float32) * 0.01

    kwargs = dict(
        experts_used=top_k,
        gate_qtype=GGMLQuantizationType.Q6_K, up_qtype=GGMLQuantizationType.Q6_K,
        down_qtype=GGMLQuantizationType.Q6_K,
        shared_qtype=GGMLQuantizationType.F32,
        eps=1e-6,
    )

    # cpu_reference (raw Q6_K bytes → gguf_quant_gemv)
    cpu_out = cpu_ffn(x, post_norm, router, gate_w, up_w, down_w,
                      GGMLQuantizationType.Q6_K, GGMLQuantizationType.Q6_K,
                      GGMLQuantizationType.Q6_K,
                      shared_gate_logit, shared_gate_w, shared_up_w, shared_down_w,
                      GGMLQuantizationType.F32,
                      experts_used=top_k, eps=1e-6)

    # GPU (should dispatch Q6_K via _dispatch_gemv)
    gpu_out = gpu_ffn(x, post_norm, router, gate_w, up_w, down_w,
                      GGMLQuantizationType.Q6_K, GGMLQuantizationType.Q6_K,
                      GGMLQuantizationType.Q6_K,
                      shared_gate_logit, shared_gate_w, shared_up_w, shared_down_w,
                      GGMLQuantizationType.F32,
                      experts_used=top_k, eps=1e-6)
    gpu_out = np.asarray(gpu_out, dtype=np.float32)

    assert gpu_out.shape == cpu_out.shape
    max_abs = float(np.max(np.abs(cpu_out - gpu_out)))
    # Q6_K uses fp16 input which can overflow for large synthetic weights.
    # Use relative error for the Q6_K FFN test.
    max_val = float(np.max(np.abs(cpu_out))) + 1e-6
    max_rel = max_abs / max_val
    assert max_rel < 0.1, f"Q6_K FFN max_rel={max_rel} exceeds 0.1 (max_abs={max_abs})"