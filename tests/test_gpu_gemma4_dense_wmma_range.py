"""T16 Q8 WMMA keeps its fast path and repairs FP16 operand overflow.

The oracle consumes the rounded BF16 activation, not the original FP32 input.
Oversized and mixed rows must remain finite and agree with the FP32 reference;
ordinary rows must also work. Tolerances cover BF16 output rounding and the
existing WMMA FP16 weight dequantization, not non-finite results.
"""
from __future__ import annotations

import numpy as np
import pytest

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import (
    copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc,
)
from hipengine.kernels.cpu_reference import gguf_q8_0_gemv
from hipengine.kernels.hip_gfx1100.quant.gguf_q8_0_t16_prefill import (
    gguf_q8_0_t16_dual_wmma_prefill_bf16_bf16_out,
    gguf_q8_0_t16_wmma_prefill_bf16_bf16_out,
    gguf_q8_0_t16_wmma_prefill_2wave_bf16_bf16_out,
    gguf_q8_0_t16_wmma_prefill_4wave_bf16_bf16_out,
)
from hipengine.quant.gguf_t16 import repack_gguf_q8_0_tile16
from tests._rocm_guard import hip_runtime_available
from tests.test_gpu_gguf_k_gemv import make_q8_0_weight

pytestmark = pytest.mark.skipif(not hip_runtime_available(), reason="HIP runtime unavailable")


def _bf16_bits(values: np.ndarray) -> np.ndarray:
    bits = values.astype(np.float32, copy=False).view(np.uint32)
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


def _bf16_bits_to_f32(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << 16).view(np.float32)


def _run(launcher, weight_tiles, activation, **kwargs):
    rows, in_features = activation.shape
    out_features = weight_tiles.shape[0] * 16
    host_in = _bf16_bits(activation)
    host_out = np.zeros((rows, out_features), dtype=np.uint16)
    runtime = get_hip_runtime()
    buffers = []
    try:
        for host in (host_in, weight_tiles, host_out):
            device = malloc(host.nbytes, runtime=runtime)
            buffers.append(device)
            copy_host_to_device(device, host_array_ptr(np.ascontiguousarray(host)), runtime=runtime)
        launcher(
            *(buffer.ptr for buffer in buffers), rows, in_features, out_features,
            runtime=runtime, **kwargs,
        )
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(host_out), buffers[2], runtime=runtime)
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)
    return _bf16_bits_to_f32(host_out)


@pytest.mark.parametrize("rows", [4, 17, 33])
@pytest.mark.parametrize("kind", ["normal", "boundary", "overflow", "mixed"])
@pytest.mark.parametrize("waves", [1, 2, 4])
def test_dense_wmma_repairs_overflow(rows, kind, waves):
    in_features, out_features = 256, 256
    activation = np.random.default_rng(42).uniform(-32, 32, (rows, in_features)).astype(np.float32)
    if kind == "boundary":
        activation[:, :4] = [65280, -65280, 65536, -65536]
    elif kind == "overflow":
        activation *= np.float32(1e6)
    elif kind == "mixed":
        activation[0, -1] = 1e7
        activation[-1, 0] = -1e7
    activation = _bf16_bits_to_f32(_bf16_bits(activation))
    weight = make_q8_0_weight(out_features, in_features)
    tiles = repack_gguf_q8_0_tile16(weight).tiles
    launcher = {
        1: gguf_q8_0_t16_wmma_prefill_bf16_bf16_out,
        2: gguf_q8_0_t16_wmma_prefill_2wave_bf16_bf16_out,
        4: gguf_q8_0_t16_wmma_prefill_4wave_bf16_bf16_out,
    }[waves]
    kwargs = {} if waves == 1 else {"tile_m": waves * 32, "tile_n": 32 if rows >= 32 else 16}
    actual = _run(launcher, tiles, activation, **kwargs)
    reference = gguf_q8_0_gemv(activation, weight)
    assert np.isfinite(actual).all()
    # Use each row's scale: a large neighbor must not hide error in ordinary rows.
    for got, expected in zip(actual, reference):
        assert np.allclose(got, expected, rtol=1e-2, atol=1e-2 * np.abs(expected).max())
    assert np.array_equal(actual, _run(launcher, tiles, activation, **kwargs))


def test_dual_dense_wmma_repairs_both_outputs():
    activation = np.full((17, 256), 1e7, dtype=np.float32)
    activation[0] = 1.0
    activation = _bf16_bits_to_f32(_bf16_bits(activation))
    weight = make_q8_0_weight(16, 256)
    tiles = repack_gguf_q8_0_tile16(weight).tiles
    expected = gguf_q8_0_gemv(activation, weight)
    other_host = np.zeros((17, 16), dtype=np.uint16)
    runtime = get_hip_runtime()
    other = malloc(other_host.nbytes, runtime=runtime)

    def launch(x, w, out, rows, ins, outs, **kwargs):
        gguf_q8_0_t16_dual_wmma_prefill_bf16_bf16_out(
            x, w, w, out, other.ptr, rows, ins, outs, **kwargs,
        )

    try:
        actual = _run(launch, tiles, activation)
        copy_device_to_host(host_array_ptr(other_host), other, runtime=runtime)
        assert np.array_equal(actual, _bf16_bits_to_f32(other_host))
        assert np.isfinite(actual).all()
        for got, reference in zip(actual, expected):
            assert np.allclose(got, reference, rtol=1e-2, atol=1e-2 * np.abs(reference).max())
    finally:
        free(other, runtime=runtime)


def test_dense_down_keeps_wmma_with_overflow_repair(tmp_path, monkeypatch):
    """Keep the oversized fixture and default expert route, not a blanket opt-out."""
    from hipengine.core.memory import DeviceBuffer
    from hipengine.loading.gguf import GGUFReader
    from hipengine.runtime.gemma4 import Gemma4Runner, load_gemma4_device_weights
    from tests._gemma4_gguf_fixture import default_fixture_tensors, fixture_metadata, write_fixture_gguf
    import hipengine.runtime.gguf_linear as linear

    artifact = GGUFReader(write_fixture_gguf(
        tmp_path / "gemma4.gguf", default_fixture_tensors(), fixture_metadata(),
    ))
    runtime = get_hip_runtime()
    launches = []
    original = linear.launch_gguf_linear

    def recording_launch(weight, x_ptr, out_ptr, rows, in_features, out_features, **kwargs):
        host = np.empty(int(rows) * int(in_features), dtype=np.uint16)
        copy_device_to_host(host_array_ptr(host), DeviceBuffer(ptr=int(x_ptr), nbytes=host.nbytes), runtime=runtime)
        launches.append((bool(kwargs.get("use_wmma_prefill")), float(np.abs(_bf16_bits_to_f32(host)).max())))
        return original(weight, x_ptr, out_ptr, rows, in_features, out_features, **kwargs)

    monkeypatch.setattr(linear, "launch_gguf_linear", recording_launch)
    weights = load_gemma4_device_weights(artifact)
    runner = Gemma4Runner(weights=weights, capacity=32)
    try:
        logits = runner.forward([1, 5, 9, 13], apply_softcap=False)
        assert np.isfinite(logits).all()
    finally:
        runner.close()
        weights.free()
    assert len(launches) == 13, launches
    assert any(magnitude > 65504 for _, magnitude in launches), launches
    assert all(enabled for enabled, _ in launches), launches
