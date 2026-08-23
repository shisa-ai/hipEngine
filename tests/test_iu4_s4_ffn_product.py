from __future__ import annotations

import ctypes
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.kernels.cpu_reference.iu4_s4 import (
    block_hadamard_f32,
    iu4_s4_corrected_i32,
    quantize_u4_hadamard_bf16,
    quantize_u4_swiglu_hadamard_bf16,
)
from hipengine.quant.iu4_ffn_pfs import pfs_s4_to_n16_k32_tiles
from hipengine.quant.iu4_s4 import bf16_bits_to_f32, f32_to_bf16_bits

GATE_SEED = 0xA511E9B3
DOWN_SEED = 0x63D83595


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def _pfs_weight_layout(values: np.ndarray) -> np.ndarray:
    q = np.asarray(values, dtype=np.int8)
    unsigned = (q.astype(np.int16) & 0xF).astype(np.uint8)
    packed = np.ascontiguousarray(
        unsigned[:, 0::2] | (unsigned[:, 1::2] << np.uint8(4))
    )
    return pfs_s4_to_n16_k32_tiles(packed)


def test_session_product_telemetry_is_explicit() -> None:
    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession

    session = object.__new__(Qwen35GGUFResidentSession)
    session.runner = SimpleNamespace(
        _iu4_ffn_product=SimpleNamespace(
            path="/tmp/product.pfs",
            sha256="a" * 64,
            minimum_rows=96,
            maximum_rows=2048,
            launch_count=128,
            fallback_count=64,
        )
    )

    assert session.iu4_ffn_telemetry() == {
        "enabled": True,
        "path": "/tmp/product.pfs",
        "sha256": "a" * 64,
        "minimum_rows": 96,
        "maximum_rows": 2048,
        "launches": 128,
        "fallbacks": 64,
    }


def test_product_runtime_admission_is_bounded_and_fail_closed() -> None:
    from hipengine.runtime.iu4_ffn_product import IU4FFNProductRuntime

    owner = object.__new__(IU4FFNProductRuntime)
    owner.layers = tuple([object()] * 64)
    owner.launch_count = 0
    owner.fallback_count = 0

    assert owner.supports(layer_id=0, rows=96)
    assert owner.supports(layer_id=63, rows=2048)
    assert not owner.supports(layer_id=0, rows=95)
    assert not owner.supports(layer_id=64, rows=512)
    assert owner.workspace_nbytes(512) == 512 * 17408 // 2 + 512 * 8


def test_gfx1151_product_registry_keys_resolve() -> None:
    from hipengine.kernels.backends import load_backend_kernel_package
    from hipengine.kernels.hip_gfx1151.quant.iu4_s4_ffn_product import (
        IU4_PFS_DOWN_PACK_KEY,
        IU4_PFS_GATE_PACK_KEY,
        IU4_PFS_LINEAR_KEY,
    )
    from hipengine.kernels.registry import is_registered

    load_backend_kernel_package("hip_gfx1151")
    assert is_registered(IU4_PFS_GATE_PACK_KEY)
    assert is_registered(IU4_PFS_DOWN_PACK_KEY)
    assert is_registered(IU4_PFS_LINEAR_KEY)


def test_block_hadamard_is_norm_preserving_and_seeded() -> None:
    values = np.arange(2048, dtype=np.float32).reshape(2, 1024) / np.float32(1024.0)

    transformed = block_hadamard_f32(values, seed=GATE_SEED)
    other = block_hadamard_f32(values, seed=DOWN_SEED)

    assert transformed.shape == values.shape
    assert np.allclose(
        np.square(transformed, dtype=np.float64).sum(axis=1),
        np.square(values, dtype=np.float64).sum(axis=1),
        rtol=1e-6,
        atol=1e-6,
    )
    assert not np.array_equal(transformed, other)


@pytest.mark.skipif(not _hip_available(), reason="ROCm/HIP runtime unavailable")
def test_gfx1151_hadamard_pack_and_pfs_linear_match_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HIPENGINE_HIP_ARCH", "gfx1151")

    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )
    from hipengine.kernels.hip_gfx1151.quant.iu4_s4_ffn_product import (
        build_iu4_s4_ffn_product,
        iu4_pfs_linear_bf16_out,
        iu4_pfs_pack_gate_bf16,
        iu4_pfs_packed_nbytes,
    )

    rng = np.random.default_rng(0xF5F4)
    rows, hidden, output = 37, 1024, 64
    x = f32_to_bf16_bits(rng.normal(0.0, 0.2, size=(rows, hidden)).astype(np.float32))
    q_weight = rng.integers(-8, 8, size=(output, hidden), dtype=np.int8)
    weight_scale = rng.uniform(0.002, 0.02, size=output).astype(np.float32)
    weight_sum = q_weight.sum(axis=1, dtype=np.int32)
    pfs_weight = _pfs_weight_layout(q_weight)
    cpu_u4 = quantize_u4_hadamard_bf16(x, seed=GATE_SEED)
    corrected = iu4_s4_corrected_i32(
        cpu_u4.quantized,
        cpu_u4.zero_points,
        q_weight,
        weight_sum,
    )
    expected = f32_to_bf16_bits(
        corrected.astype(np.float32)
        * cpu_u4.scales[:, None]
        * weight_scale[None, :]
    )

    runtime = get_hip_runtime()
    library = build_iu4_s4_ffn_product(load=True)
    buffers = []

    def upload(array: np.ndarray):
        contiguous = np.ascontiguousarray(array)
        buffer = malloc(contiguous.nbytes, runtime=runtime)
        copy_host_to_device(buffer, host_array_ptr(contiguous), runtime=runtime)
        buffers.append(buffer)
        return buffer

    try:
        x_dev = upload(x)
        weight_dev = upload(pfs_weight)
        weight_scale_dev = upload(weight_scale)
        weight_sum_dev = upload(weight_sum)
        packed_dev = malloc(iu4_pfs_packed_nbytes(rows, hidden), runtime=runtime)
        scale_dev = malloc(rows * 4, runtime=runtime)
        zero_dev = malloc(rows * 4, runtime=runtime)
        output_dev = malloc(expected.nbytes, runtime=runtime)
        buffers.extend((packed_dev, scale_dev, zero_dev, output_dev))

        iu4_pfs_pack_gate_bf16(
            x_dev.ptr,
            packed_dev.ptr,
            scale_dev.ptr,
            zero_dev.ptr,
            rows,
            hidden,
            library=library,
            runtime=runtime,
        )
        iu4_pfs_linear_bf16_out(
            packed_dev.ptr,
            scale_dev.ptr,
            zero_dev.ptr,
            weight_dev.ptr,
            weight_scale_dev.ptr,
            weight_sum_dev.ptr,
            output_dev.ptr,
            rows,
            hidden,
            output,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()

        packed = np.empty((rows, hidden // 8), dtype=np.uint32)
        scales = np.empty(rows, dtype=np.float32)
        zeros = np.empty(rows, dtype=np.int32)
        actual = np.empty_like(expected)
        copy_device_to_host(host_array_ptr(packed), packed_dev, runtime=runtime)
        copy_device_to_host(host_array_ptr(scales), scale_dev, runtime=runtime)
        copy_device_to_host(host_array_ptr(zeros), zero_dev, runtime=runtime)
        copy_device_to_host(host_array_ptr(actual), output_dev, runtime=runtime)
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)

    assert np.array_equal(packed, cpu_u4.packed_words)
    assert np.array_equal(zeros, cpu_u4.zero_points)
    assert np.allclose(scales, cpu_u4.scales, rtol=0.0, atol=1e-7)
    assert np.array_equal(actual, expected)


@pytest.mark.skipif(not _hip_available(), reason="ROCm/HIP runtime unavailable")
def test_gfx1151_swiglu_hadamard_pack_matches_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HIPENGINE_HIP_ARCH", "gfx1151")

    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )
    from hipengine.kernels.hip_gfx1151.quant.iu4_s4_ffn_product import (
        build_iu4_s4_ffn_product,
        iu4_pfs_pack_swiglu_down_bf16,
        iu4_pfs_packed_nbytes,
    )

    rng = np.random.default_rng(0xD04)
    rows, width = 2, 17408
    gate_up = f32_to_bf16_bits(
        rng.normal(0.0, 0.1, size=(rows, 2 * width)).astype(np.float32)
    )
    expected = quantize_u4_swiglu_hadamard_bf16(
        gate_up,
        width=width,
        seed=DOWN_SEED,
    )
    runtime = get_hip_runtime()
    library = build_iu4_s4_ffn_product(load=True)
    buffers = []
    try:
        source_dev = malloc(gate_up.nbytes, runtime=runtime)
        packed_dev = malloc(iu4_pfs_packed_nbytes(rows, width), runtime=runtime)
        scale_dev = malloc(rows * 4, runtime=runtime)
        zero_dev = malloc(rows * 4, runtime=runtime)
        buffers.extend((source_dev, packed_dev, scale_dev, zero_dev))
        copy_host_to_device(source_dev, host_array_ptr(gate_up), runtime=runtime)
        iu4_pfs_pack_swiglu_down_bf16(
            source_dev.ptr,
            packed_dev.ptr,
            scale_dev.ptr,
            zero_dev.ptr,
            rows,
            width,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        packed = np.empty((rows, width // 8), dtype=np.uint32)
        scales = np.empty(rows, dtype=np.float32)
        zeros = np.empty(rows, dtype=np.int32)
        copy_device_to_host(host_array_ptr(packed), packed_dev, runtime=runtime)
        copy_device_to_host(host_array_ptr(scales), scale_dev, runtime=runtime)
        copy_device_to_host(host_array_ptr(zeros), zero_dev, runtime=runtime)
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)

    assert np.array_equal(packed, expected.packed_words)
    assert np.array_equal(zeros, expected.zero_points)
    assert np.allclose(scales, expected.scales, rtol=0.0, atol=1e-7)
