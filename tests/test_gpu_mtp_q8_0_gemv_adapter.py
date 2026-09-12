"""RED/GREEN test for Q8_0 gemv adapter (M6).

Tests that the hip GPU Q8_0 gemv adapter matches cpu_reference on synthetic
Q8_0 data, and that the FFN composite dispatches to Q8_0 instead of raising
NotImplementedError.
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

from _gguf_synthetic_weights import make_q8_0_weight  # noqa: E402
from hipengine.kernels.cpu_reference import gguf_q8_0_gemv  # noqa: E402
from hipengine.quant.gguf import GGMLQuantizationType  # noqa: E402


def _f32(v):
    return np.ascontiguousarray(v, dtype=np.float32)


def test_q8_0_adapter_registered():
    from hipengine.kernels.registry import KernelKey, registered_keys

    assert KernelKey("hip_gfx1100", "mtp_q8_0_gemv", "gguf_f32", "qwen35") in registered_keys()
    assert KernelKey("hip_gfx1151", "mtp_q8_0_gemv", "gguf_f32", "qwen35") in registered_keys()


@pytest.mark.parametrize("backend", ["hip_gfx1100", "hip_gfx1151"])
@pytest.mark.parametrize(
    ("rows", "in_features", "out_features"),
    [(1, 32, 4), (2, 64, 8), (1, 128, 3)],
)
def test_hip_q8_0_gemv_matches_cpu_reference(backend, rows, in_features, out_features):
    from hipengine.kernels.registry import resolve

    try:
        kernel = resolve(backend=backend, layer="mtp_q8_0_gemv", quant="gguf_f32", variant="qwen35")
    except Exception:
        pytest.skip(f"{backend} mtp_q8_0_gemv not registered yet (RED)")
    if kernel is gguf_q8_0_gemv:
        pytest.skip(f"{backend} returned cpu_reference via fallback (RED)")

    x = (np.arange(rows * in_features, dtype=np.float32).reshape(rows, in_features) % 13 - 6) / 8.0
    qweight = make_q8_0_weight(out_features=out_features, in_features=in_features)

    expected = gguf_q8_0_gemv(x, qweight)
    actual = kernel(x, qweight)
    actual = np.asarray(actual, dtype=np.float32)

    assert actual.shape == expected.shape, f"shape {actual.shape} != {expected.shape}"
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)


def test_ffn_no_longer_raises_notimpl_for_q8_0():
    from hipengine.kernels.hip_gfx1100.speculative.mtp_nextn import (
        qwen35_gguf_mtp_ffn_sublayer_f32,
    )

    tokens, hidden_size, inter_dim, num_experts, top_k = 1, 256, 256, 2, 1
    x = _f32(np.random.randn(tokens, hidden_size) * 0.1)
    norm_w = _f32(np.ones(hidden_size))
    router_w = _f32(np.random.randn(num_experts, hidden_size) * 0.1)
    gate_qw = np.stack([make_q8_0_weight(out_features=inter_dim, in_features=hidden_size)
                        for _ in range(num_experts)])
    up_qw = np.stack([make_q8_0_weight(out_features=inter_dim, in_features=hidden_size)
                      for _ in range(num_experts)])
    down_qw = np.stack([make_q8_0_weight(out_features=hidden_size, in_features=inter_dim)
                        for _ in range(num_experts)])
    shared_gate_qw = make_q8_0_weight(out_features=inter_dim, in_features=hidden_size)
    shared_up_qw = make_q8_0_weight(out_features=inter_dim, in_features=hidden_size)
    shared_down_qw = make_q8_0_weight(out_features=hidden_size, in_features=inter_dim)
    shared_gate_logit_w = _f32(np.random.randn(hidden_size) * 0.1)

    try:
        out = qwen35_gguf_mtp_ffn_sublayer_f32(
            x, norm_w, router_w,
            gate_qw, up_qw, down_qw,
            GGMLQuantizationType.Q8_0, GGMLQuantizationType.Q8_0, GGMLQuantizationType.Q8_0,
            shared_gate_logit_w, shared_gate_qw, shared_up_qw, shared_down_qw,
            GGMLQuantizationType.Q8_0,
            experts_used=top_k, eps=1e-6,
        )
        assert out.shape == (tokens, hidden_size), f"unexpected shape {out.shape}"
    except NotImplementedError as e:
        pytest.fail(f"FFN raised NotImplementedError for Q8_0: {e}")