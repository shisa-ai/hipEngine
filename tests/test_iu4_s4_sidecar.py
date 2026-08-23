from __future__ import annotations

import ctypes

import numpy as np
import pytest

from hipengine.kernels.cpu_reference.iu4_s4 import (
    iu4_s4_corrected_i32,
    iu4_s4_gate_up_silu_bf16,
)
from hipengine.quant.iu4_s4 import (
    S4Sidecar,
    U4Rows,
    bf16_bits_to_f32,
    f32_to_bf16_bits,
    iu4_s4_gate_up_silu_reference,
    iu4_s4_i32_reference,
    pack_s4_wmma_tiles,
    quantize_s4_per_output,
    quantize_u4_per_row,
    unpack_s4,
    unpack_u4,
    unpack_u4_wmma_tiles,
)


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def test_signed_nibble_round_trip_covers_every_s4_value() -> None:
    weights = np.tile(np.arange(-8, 8, dtype=np.float32), (3, 2))
    sidecar = quantize_s4_per_output(weights)

    assert np.array_equal(unpack_s4(sidecar.packed), weights.astype(np.int8))
    assert np.array_equal(sidecar.sums, weights.sum(axis=1, dtype=np.int32))
    assert np.array_equal(sidecar.scales, np.ones(3, dtype=np.float32))


def test_u4_asymmetric_quant_handles_zero_and_constant_rows() -> None:
    rows = np.asarray(
        [
            np.linspace(-7.0, 8.0, 32, dtype=np.float32),
            np.zeros(32, dtype=np.float32),
            np.full(32, 3.0, dtype=np.float32),
            np.full(32, -2.0, dtype=np.float32),
        ]
    )
    packed = quantize_u4_per_row(f32_to_bf16_bits(rows))
    q = unpack_u4(packed.packed)
    reconstructed = packed.scales[:, None] * (
        q.astype(np.float32) - packed.zero_points[:, None].astype(np.float32)
    )

    assert packed.zero_points.tolist() == [7, 0, 0, 15]
    assert np.array_equal(q[1], np.zeros(32, dtype=np.uint8))
    assert np.allclose(reconstructed[1:], rows[1:], rtol=0.0, atol=1e-6)


def test_u4_s4_correction_identity_is_exact_in_i32() -> None:
    q_a = np.tile(np.arange(16, dtype=np.uint8), (4, 2))
    q_w = np.tile(np.arange(-8, 8, dtype=np.int8), (5, 2))
    u4 = U4Rows.from_quantized(q_a, scales=np.ones(4), zero_points=np.asarray([0, 1, 7, 15]))
    s4 = S4Sidecar.from_quantized(q_w, scales=np.ones(5))

    corrected = iu4_s4_i32_reference(u4, s4)
    expected = iu4_s4_corrected_i32(q_a, u4.zero_points, q_w, s4.sums)

    assert np.array_equal(corrected, expected)


def test_gfx1151_registry_declares_candidate_and_exact_qmicro_fallback() -> None:
    from hipengine.kernels.backends import load_backend_kernel_package
    from hipengine.kernels.hip_gfx1151.quant.iu4_s4_sidecar import (
        IU4_S4_DUAL_SILU_KEY,
        IU4_S4_STRICT_FALLBACK_KEY,
        IU4_U4_QUANT_KEY,
        iu4_s4_dual_silu_bf16_out,
        iu4_u4_wmma_nbytes,
    )
    from hipengine.kernels.registry import is_registered
    from hipengine.quant.registry import resolve_quant

    load_backend_kernel_package("hip_gfx1151")
    assert resolve_quant("iu4_s4_sidecar_v1").kernel_family == "gfx1151_iu4_s4_sidecar"
    assert is_registered(IU4_U4_QUANT_KEY)
    assert is_registered(IU4_S4_DUAL_SILU_KEY)
    assert is_registered(IU4_S4_STRICT_FALLBACK_KEY)
    assert iu4_u4_wmma_nbytes(1024, 32) == 16_384
    with pytest.raises(ValueError, match=r"\[1, 1024\]"):
        iu4_u4_wmma_nbytes(1025, 32)
    with pytest.raises(ValueError, match="excludes M=1"):
        iu4_s4_dual_silu_bf16_out(*(0,) * 10, 1, 32, 16)


def test_gate_up_reference_publishes_bf16_before_silu() -> None:
    rng = np.random.default_rng(0x1A4)
    x = f32_to_bf16_bits(rng.normal(0.0, 0.3, size=(3, 32)).astype(np.float32))
    gate = quantize_s4_per_output(rng.normal(0.0, 0.2, size=(16, 32)).astype(np.float32))
    up = quantize_s4_per_output(rng.normal(0.0, 0.2, size=(16, 32)).astype(np.float32))

    output, gate_bits, up_bits = iu4_s4_gate_up_silu_reference(x, gate, up, return_projections=True)
    gate_f32 = bf16_bits_to_f32(gate_bits)
    up_f32 = bf16_bits_to_f32(up_bits)
    expected = f32_to_bf16_bits((gate_f32 / (1.0 + np.exp(-gate_f32))) * up_f32)

    assert np.array_equal(output, expected)


@pytest.mark.skipif(not _hip_available(), reason="ROCm/HIP runtime unavailable")
@pytest.mark.parametrize("rows", [5, 37])
def test_gfx1151_iu4_probe_and_operation_match_cpu_reference(
    monkeypatch: pytest.MonkeyPatch,
    rows: int,
) -> None:
    monkeypatch.setenv("HIPENGINE_HIP_ARCH", "gfx1151")

    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )
    from hipengine.kernels.hip_gfx1151.quant.iu4_s4_sidecar import (
        build_iu4_s4_sidecar,
        iu4_s4_dual_silu_bf16_out,
        iu4_s4_matmul_i32_probe,
        iu4_u4_quantize_bf16,
        iu4_u4_wmma_nbytes,
    )

    rng = np.random.default_rng(0x1151 + rows)
    hidden, output = 32, 32
    x = f32_to_bf16_bits(rng.normal(0.0, 0.4, size=(rows, hidden)).astype(np.float32))
    gate = quantize_s4_per_output(rng.normal(0.0, 0.25, size=(output, hidden)).astype(np.float32))
    up = quantize_s4_per_output(rng.normal(0.0, 0.25, size=(output, hidden)).astype(np.float32))
    u4 = quantize_u4_per_row(x)
    expected_i32 = (
        unpack_u4(u4.packed).astype(np.int32)
        @ unpack_s4(gate.packed).astype(np.int32).T
    )
    expected_output = iu4_s4_gate_up_silu_bf16(
        unpack_u4(u4.packed),
        u4.scales,
        u4.zero_points,
        unpack_s4(gate.packed),
        gate.scales,
        gate.sums,
        unpack_s4(up.packed),
        up.scales,
        up.sums,
    )

    runtime = get_hip_runtime()
    library = build_iu4_s4_sidecar(load=True)
    buffers = []

    def upload(array: np.ndarray):
        contiguous = np.ascontiguousarray(array)
        buffer = malloc(contiguous.nbytes, runtime=runtime)
        copy_host_to_device(buffer, host_array_ptr(contiguous), runtime=runtime)
        buffers.append(buffer)
        return buffer

    try:
        x_dev = upload(x)
        gate_packed_dev = upload(pack_s4_wmma_tiles(gate))
        gate_scale_dev = upload(gate.scales)
        gate_sum_dev = upload(gate.sums)
        up_packed_dev = upload(pack_s4_wmma_tiles(up))
        up_scale_dev = upload(up.scales)
        up_sum_dev = upload(up.sums)
        packed_dev = malloc(iu4_u4_wmma_nbytes(rows, hidden), runtime=runtime)
        scale_dev = malloc(rows * 4, runtime=runtime)
        zero_dev = malloc(rows * 4, runtime=runtime)
        probe_dev = malloc(rows * output * 4, runtime=runtime)
        out_dev = malloc(rows * output * 2, runtime=runtime)
        buffers.extend((packed_dev, scale_dev, zero_dev, probe_dev, out_dev))

        iu4_u4_quantize_bf16(
            x_dev.ptr,
            packed_dev.ptr,
            scale_dev.ptr,
            zero_dev.ptr,
            rows,
            hidden,
            library=library,
            runtime=runtime,
        )
        iu4_s4_matmul_i32_probe(
            packed_dev.ptr,
            gate_packed_dev.ptr,
            probe_dev.ptr,
            rows,
            hidden,
            output,
            library=library,
            runtime=runtime,
        )
        iu4_s4_dual_silu_bf16_out(
            packed_dev.ptr,
            scale_dev.ptr,
            zero_dev.ptr,
            gate_packed_dev.ptr,
            gate_scale_dev.ptr,
            gate_sum_dev.ptr,
            up_packed_dev.ptr,
            up_scale_dev.ptr,
            up_sum_dev.ptr,
            out_dev.ptr,
            rows,
            hidden,
            output,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()

        packed_host = np.empty(
            (((rows + 15) // 16), hidden // 32, 16, 16), dtype=np.uint8
        )
        scales_host = np.empty_like(u4.scales)
        zeros_host = np.empty_like(u4.zero_points)
        probe_host = np.empty((rows, output), dtype=np.int32)
        output_host = np.empty((rows, output), dtype=np.uint16)
        copy_device_to_host(host_array_ptr(packed_host), packed_dev, runtime=runtime)
        copy_device_to_host(host_array_ptr(scales_host), scale_dev, runtime=runtime)
        copy_device_to_host(host_array_ptr(zeros_host), zero_dev, runtime=runtime)
        copy_device_to_host(host_array_ptr(probe_host), probe_dev, runtime=runtime)
        copy_device_to_host(host_array_ptr(output_host), out_dev, runtime=runtime)
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)

    assert np.array_equal(
        unpack_u4_wmma_tiles(packed_host, rows=rows, hidden=hidden),
        unpack_u4(u4.packed),
    )
    assert np.array_equal(zeros_host, u4.zero_points)
    assert np.allclose(scales_host, u4.scales, rtol=0.0, atol=1e-7)
    assert np.array_equal(probe_host, expected_i32)
    actual = bf16_bits_to_f32(output_host)
    expected = bf16_bits_to_f32(expected_output)
    assert np.max(np.abs(actual - expected)) <= 2.0 * np.spacing(np.abs(expected).max()) + 1e-4
