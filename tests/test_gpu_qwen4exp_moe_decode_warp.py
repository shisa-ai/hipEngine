"""Warp-per-expert decode owners for the routed experts against the owners they replace.

Both keep the per-element arithmetic of their incumbents and change the
reduction order, so the contract is agreement within BF16 round-off of the
BF16 outputs, plus run-to-run determinism.
"""
import ctypes

import numpy as np
import pytest

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import (
    copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_gemv import (
    gguf_q4_k_quantize_bf16_q8_1,
    gguf_q4_k_selected_dual_q8_1_dp4a_silu_logical128_t64_gemv_bf16_bf16_out,
    gguf_q4_k_selected_dual_q8_1_dp4a_silu_warp256_gemv_bf16_bf16_out,
)
from hipengine.kernels.hip_gfx1100.quant.qwen4_exp_q5_1 import (
    qwen4_exp_q5_1_selected_weighted_sum_logical256_t64_bf16_bf16_out,
    qwen4_exp_q5_1_selected_weighted_sum_warp256_fast_bf16_bf16_out,
)
from tests._gguf_synthetic_weights import make_q4_k_weight
from tests.test_gpu_gguf_q5_1_t16_expert_major_wmma import make_q5_1_weight


def _hip_available():
    try:
        ctypes.CDLL("libamdhip64.so")
        return True
    except OSError:
        return False


pytestmark = pytest.mark.skipif(not _hip_available(), reason="HIP runtime unavailable")


def _to_bf16(values):
    bits = np.ascontiguousarray(values, np.float32).view(np.uint32)
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


def _from_bf16(bits):
    return (bits.astype(np.uint32) << 16).view(np.float32)


def _experts(make, experts, out_features, in_features):
    base = make(out_features, in_features)
    return np.ascontiguousarray(
        np.stack([np.roll(base, shift=3 * e + 1, axis=0) for e in range(experts)]))


class _Device:
    def __init__(self):
        self.runtime = get_hip_runtime()
        self.buffers = []

    def upload(self, array):
        buf = malloc(array.nbytes, runtime=self.runtime)
        copy_host_to_device(buf, host_array_ptr(array), runtime=self.runtime)
        self.buffers.append(buf)
        return buf

    def alloc(self, nbytes):
        buf = malloc(nbytes, runtime=self.runtime)
        self.buffers.append(buf)
        return buf

    def download(self, buf, shape, dtype):
        out = np.empty(shape, dtype)
        copy_device_to_host(host_array_ptr(out), buf, runtime=self.runtime)
        return out

    def close(self):
        for buf in reversed(self.buffers):
            free(buf, runtime=self.runtime)


@pytest.fixture
def device():
    dev = _Device()
    yield dev
    dev.close()


def _close_in_bf16(candidate_bits, incumbent_bits, *, ulps):
    candidate, incumbent = _from_bf16(candidate_bits), _from_bf16(incumbent_bits)
    assert np.isfinite(candidate).all()
    # One BF16 ulp is 2**-7 relative; allow `ulps` of them at the value's scale.
    tolerance = ulps * np.maximum(np.abs(incumbent), 1e-3) * 2.0 ** -7
    assert (np.abs(candidate - incumbent) <= tolerance).all()


@pytest.mark.parametrize("selected_rows", [10, 4])
def test_warp_gate_up_matches_the_logical128_owner(device, selected_rows):
    rng = np.random.default_rng(700 + selected_rows)
    experts, in_features, out_features = 12, 512, 96
    x = _to_bf16(rng.standard_normal((1, in_features)).astype(np.float32) * 0.5)
    gate = _experts(make_q4_k_weight, experts, out_features, in_features)
    up = np.ascontiguousarray(gate[::-1])
    selected = rng.permutation(experts)[:selected_rows].astype(np.int64)
    x_dev, gate_dev, up_dev, sel_dev = (
        device.upload(a) for a in (x, gate, up, selected))
    xq = device.alloc(in_features * 4 + 4096)
    gguf_q4_k_quantize_bf16_q8_1(x_dev.ptr, xq.ptr, 1, in_features, runtime=device.runtime)
    outputs = []
    for fn in (
        gguf_q4_k_selected_dual_q8_1_dp4a_silu_logical128_t64_gemv_bf16_bf16_out,
        gguf_q4_k_selected_dual_q8_1_dp4a_silu_warp256_gemv_bf16_bf16_out,
        gguf_q4_k_selected_dual_q8_1_dp4a_silu_warp256_gemv_bf16_bf16_out,
    ):
        out = device.upload(np.full((selected_rows, out_features), 0x7FC1, np.uint16))
        fn(xq.ptr, sel_dev.ptr, gate_dev.ptr, up_dev.ptr, out.ptr, 1, selected_rows,
           experts, in_features, out_features, runtime=device.runtime)
        device.runtime.device_synchronize()
        outputs.append(device.download(out, (selected_rows, out_features), np.uint16))
    incumbent, warp, repeat = outputs
    np.testing.assert_array_equal(repeat, warp)
    _close_in_bf16(warp, incumbent, ulps=2)


@pytest.mark.parametrize("selected_rows", [10, 4])
def test_warp_down_matches_the_logical256_owner(device, selected_rows):
    rng = np.random.default_rng(900 + selected_rows)
    experts, in_features, out_features = 12, 640, 160
    x = _to_bf16(rng.standard_normal((selected_rows, in_features)).astype(np.float32) * 0.5)
    weights = _experts(make_q5_1_weight, experts, out_features, in_features)
    selected = rng.permutation(experts)[:selected_rows].astype(np.int64)
    routing = rng.dirichlet(np.ones(selected_rows)).astype(np.float32)
    x_dev, w_dev, sel_dev, routing_dev = (
        device.upload(a) for a in (x, weights, selected, routing))
    outputs = []
    for fn in (
        qwen4_exp_q5_1_selected_weighted_sum_logical256_t64_bf16_bf16_out,
        qwen4_exp_q5_1_selected_weighted_sum_warp256_fast_bf16_bf16_out,
        qwen4_exp_q5_1_selected_weighted_sum_warp256_fast_bf16_bf16_out,
    ):
        out = device.upload(np.full(out_features, 0x7FC1, np.uint16))
        fn(x_dev.ptr, sel_dev.ptr, w_dev.ptr, routing_dev.ptr, out.ptr, selected_rows,
           experts, in_features, out_features, runtime=device.runtime)
        device.runtime.device_synchronize()
        outputs.append(device.download(out, (out_features,), np.uint16))
    incumbent, warp, repeat = outputs
    np.testing.assert_array_equal(repeat, warp)
    _close_in_bf16(warp, incumbent, ulps=3)
