from __future__ import annotations

import ctypes

import numpy as np
import pytest

from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.cpu_reference import gguf_q4_k_gemv, gguf_q4_k_pack8_gemv
from hipengine.kernels.hip_gfx1100.fused.paro_silu import (
    build_paro_silu,
    silu_mul_separate_out_bf16,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_gemv import (
    build_gguf_q4_k_gemv,
    gguf_q4_k_gemv_bf16_bf16_out,
    gguf_q4_k_gemv_bf16_f32_out,
    gguf_q4_k_gemv_f32_f32_out,
    gguf_q4_k_gemv_fp16_f32_out,
    gguf_q4_k_pack8_dual_prefill_bf16_bf16_out,
    gguf_q4_k_pack8_dual_silu_bf16_bf16_out,
    gguf_q4_k_pack8_gemv_bf16_bf16_out,
    gguf_q4_k_pack8_gemv_bf16_f32_out,
    gguf_q4_k_pack8_gemv_f32_f32_out,
    gguf_q4_k_pack8_gemv_fp16_f32_out,
    plan_gguf_q4_k_gemv_build,
)
from hipengine.kernels.registry import resolve
from hipengine.quant.gguf import GGMLQuantizationType, dequantize_gguf_data
from hipengine.quant.gguf_q4_k import repack_gguf_q4_k_pack8

QK_K = 256
Q4_K_BLOCK_BYTES = 144


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


HIP_AVAILABLE = _hip_available()


def make_q4_k_weight(out_features: int, in_features: int) -> np.ndarray:
    if in_features % QK_K:
        raise ValueError("in_features must be a multiple of 256")
    blocks_per_row = in_features // QK_K
    data = np.empty((out_features, blocks_per_row * Q4_K_BLOCK_BYTES), dtype=np.uint8)
    for out_idx in range(out_features):
        for block_idx in range(blocks_per_row):
            block = _make_q4_k_block(out_idx, block_idx)
            start = block_idx * Q4_K_BLOCK_BYTES
            data[out_idx, start : start + Q4_K_BLOCK_BYTES] = block
    return data


def _make_q4_k_block(out_idx: int, block_idx: int) -> np.ndarray:
    d = np.float16(0.015625 * (1 + (out_idx % 5)))
    dmin = np.float16(0.0078125 * (1 + (block_idx % 3)))
    scales = ((np.arange(8, dtype=np.uint8) * 3 + out_idx + block_idx) % 63 + 1).astype(np.uint8)
    mins = ((np.arange(8, dtype=np.uint8) * 5 + 2 * out_idx + block_idx) % 17).astype(np.uint8)
    q = ((np.arange(QK_K, dtype=np.uint16) + out_idx * 7 + block_idx * 11) % 16).astype(np.uint8)

    packed_scales = _pack_q4_k_scales(scales, mins)
    q_groups = q.reshape(8, 32)
    packed_q = np.empty(128, dtype=np.uint8)
    for pair in range(4):
        packed_q[pair * 32 : (pair + 1) * 32] = q_groups[2 * pair] | (q_groups[2 * pair + 1] << 4)
    return np.concatenate(
        [
            np.asarray([d], dtype=np.float16).view(np.uint8),
            np.asarray([dmin], dtype=np.float16).view(np.uint8),
            packed_scales,
            packed_q,
        ]
    )


def _pack_q4_k_scales(scales: np.ndarray, mins: np.ndarray) -> np.ndarray:
    scales = np.asarray(scales, dtype=np.uint8)
    mins = np.asarray(mins, dtype=np.uint8)
    out = np.zeros(12, dtype=np.uint8)
    out[:4] = (scales[:4] & 0x3F) | ((scales[4:] & 0x30) << 2)
    out[4:8] = (mins[:4] & 0x3F) | ((mins[4:] & 0x30) << 2)
    out[8:12] = (scales[4:] & 0x0F) | ((mins[4:] & 0x0F) << 4)
    return out


def test_cpu_reference_gguf_q4_k_gemv_matches_dequantized_matmul() -> None:
    x = (np.arange(2 * 512, dtype=np.float32).reshape(2, 512) % 13 - 6) / 8.0
    qweight = make_q4_k_weight(out_features=5, in_features=512)

    out = gguf_q4_k_gemv(x, qweight)
    weight = dequantize_gguf_data(qweight, GGMLQuantizationType.Q4_K)
    expected = np.matmul(x, weight.T).astype(np.float32)

    assert out.dtype == np.float32
    assert out.shape == (2, 5)
    np.testing.assert_allclose(out, expected, rtol=0.0, atol=1e-6)


def test_repacked_q4_k_pack8_matches_raw_reference() -> None:
    x = (np.arange(2 * 512, dtype=np.float32).reshape(2, 512) % 13 - 6) / 8.0
    qweight = make_q4_k_weight(out_features=16, in_features=512)

    packed = repack_gguf_q4_k_pack8(qweight)
    out = gguf_q4_k_pack8_gemv(x, packed.qweight, packed.scales, packed.mins)
    expected = gguf_q4_k_gemv(x, qweight)

    assert packed.qweight.shape == (2, 512)
    assert packed.scales.shape == (16, 16)
    assert packed.mins.shape == (16, 16)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, expected, rtol=0.0, atol=1e-6)


def test_repacked_q4_k_pack8_validates_shape() -> None:
    with pytest.raises(ValueError, match="divisible by 8"):
        repack_gguf_q4_k_pack8(make_q4_k_weight(out_features=5, in_features=256))


def test_gguf_q4_k_gemv_registry_and_build_plan() -> None:
    assert resolve(
        backend="hip_gfx1100",
        layer="linear",
        quant="gguf_q4_k",
        variant="gemv_f32_f32_out",
    ) is gguf_q4_k_gemv_f32_f32_out
    assert resolve(
        backend="hip_gfx1100",
        layer="linear",
        quant="gguf_q4_k",
        variant="gemv_fp16_f32_out",
    ) is gguf_q4_k_gemv_fp16_f32_out
    assert resolve(
        backend="hip_gfx1100",
        layer="linear",
        quant="gguf_q4_k",
        variant="gemv_bf16_f32_out",
    ) is gguf_q4_k_gemv_bf16_f32_out
    assert resolve(
        backend="hip_gfx1100",
        layer="linear",
        quant="gguf_q4_k",
        variant="gemv_bf16_bf16_out",
    ) is gguf_q4_k_gemv_bf16_bf16_out
    assert resolve(
        backend="hip_gfx1100",
        layer="linear",
        quant="gguf_q4_k",
        variant="pack8_f32_f32_out",
    ) is gguf_q4_k_pack8_gemv_f32_f32_out
    assert resolve(
        backend="hip_gfx1100",
        layer="linear",
        quant="gguf_q4_k",
        variant="pack8_fp16_f32_out",
    ) is gguf_q4_k_pack8_gemv_fp16_f32_out
    assert resolve(
        backend="hip_gfx1100",
        layer="linear",
        quant="gguf_q4_k",
        variant="pack8_bf16_f32_out",
    ) is gguf_q4_k_pack8_gemv_bf16_f32_out
    assert resolve(
        backend="hip_gfx1100",
        layer="linear",
        quant="gguf_q4_k",
        variant="pack8_bf16_bf16_out",
    ) is gguf_q4_k_pack8_gemv_bf16_bf16_out
    assert resolve(
        backend="cpu_reference",
        layer="linear",
        quant="gguf_q4_k",
        variant="gemv_f32_f32_out",
    ) is gguf_q4_k_gemv
    assert resolve(
        backend="cpu_reference",
        layer="linear",
        quant="gguf_q4_k",
        variant="pack8_f32_f32_out",
    ) is gguf_q4_k_pack8_gemv

    artifact = plan_gguf_q4_k_gemv_build(compiler_version="test-compiler")
    assert artifact.output_path.name == "gguf_q4_k_gemv.so"
    assert "gguf_q4_k_gemv" in str(artifact.output_path)
    assert any(path.name == "gguf_q4_k_gemv.hip" for path in artifact.sources)

    dry_run = build_gguf_q4_k_gemv(dry_run=True, compiler_version="test-compiler")
    assert dry_run.output_path == artifact.output_path


def test_gguf_q4_k_wrapper_validates_kernel_contract() -> None:
    with pytest.raises(ValueError, match="divisible"):
        gguf_q4_k_gemv_f32_f32_out(1, 2, 3, rows=1, in_features=255, out_features=1)

    with pytest.raises(ValueError, match="threads"):
        gguf_q4_k_gemv_f32_f32_out(
            1, 2, 3, rows=1, in_features=256, out_features=1, threads=96
        )

    with pytest.raises(ValueError, match="divisible by 8"):
        gguf_q4_k_pack8_gemv_f32_f32_out(
            1, 2, 3, 4, 5, rows=1, in_features=256, out_features=7
        )

    with pytest.raises(ValueError, match="threads"):
        gguf_q4_k_pack8_gemv_f32_f32_out(
            1, 2, 3, 4, 5, rows=1, in_features=256, out_features=8, threads=256
        )


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_laguna_pack8_dual_decode_is_bf16_exact_to_two_local32_owners() -> None:
    """The c=1 dual owner must preserve both retained projection boundaries."""

    rows, in_features, out_features = 1, 1024, 64
    raw_a = make_q4_k_weight(out_features, in_features)
    raw_b = np.roll(raw_a, 17, axis=0).copy()
    packed_a = repack_gguf_q4_k_pack8(raw_a)
    packed_b = repack_gguf_q4_k_pack8(raw_b)
    x_f32 = (
        (np.arange(in_features, dtype=np.float32) % 29.0) - 14.0
    ).reshape(1, in_features) / 32.0
    x_u32 = x_f32.view(np.uint32).copy()
    x_u32 += 0x7FFF + ((x_u32 >> 16) & 1)
    x_bf16 = (x_u32 >> 16).astype(np.uint16)
    control_a = np.empty((rows, out_features), dtype=np.uint16)
    control_b = np.empty_like(control_a)
    candidate_a = np.empty_like(control_a)
    candidate_b = np.empty_like(control_a)
    library = build_gguf_q4_k_gemv(load=True)

    host_arrays = (
        x_bf16,
        packed_a.qweight,
        packed_a.scales,
        packed_a.mins,
        packed_b.qweight,
        packed_b.scales,
        packed_b.mins,
    )
    device_inputs = [malloc(array.nbytes) for array in host_arrays]
    device_outputs = [
        malloc(control_a.nbytes),
        malloc(control_b.nbytes),
        malloc(candidate_a.nbytes),
        malloc(candidate_b.nbytes),
    ]
    try:
        for array, allocation in zip(host_arrays, device_inputs, strict=True):
            copy_host_to_device(allocation, host_array_ptr(array), array.nbytes)
        x_buf, qwa, sa, ma, qwb, sb, mb = device_inputs
        ca, cb, da, db = device_outputs
        gguf_q4_k_pack8_gemv_bf16_bf16_out(
            x_buf.ptr,
            qwa.ptr,
            sa.ptr,
            ma.ptr,
            ca.ptr,
            rows,
            in_features,
            out_features,
            threads=32,
            library=library,
        )
        gguf_q4_k_pack8_gemv_bf16_bf16_out(
            x_buf.ptr,
            qwb.ptr,
            sb.ptr,
            mb.ptr,
            cb.ptr,
            rows,
            in_features,
            out_features,
            threads=32,
            library=library,
        )
        gguf_q4_k_pack8_dual_prefill_bf16_bf16_out(
            x_buf.ptr,
            qwa.ptr,
            sa.ptr,
            ma.ptr,
            qwb.ptr,
            sb.ptr,
            mb.ptr,
            da.ptr,
            db.ptr,
            rows,
            in_features,
            out_features,
            threads=32,
            library=library,
        )
        for host, allocation in zip(
            (control_a, control_b, candidate_a, candidate_b),
            device_outputs,
            strict=True,
        ):
            copy_device_to_host(host_array_ptr(host), allocation, host.nbytes)
    finally:
        for allocation in (*device_outputs, *device_inputs):
            free(allocation)

    np.testing.assert_array_equal(candidate_a, control_a)
    np.testing.assert_array_equal(candidate_b, control_b)


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_laguna_pack8_dual_silu_preserves_the_unfused_bf16_boundaries() -> None:
    """The fused owner must equal dual BF16 outputs followed by BF16 SiLU."""

    rows, in_features, out_features = 1, 1024, 64
    raw_a = make_q4_k_weight(out_features, in_features)
    raw_b = np.roll(raw_a, 17, axis=0).copy()
    packed_a = repack_gguf_q4_k_pack8(raw_a)
    packed_b = repack_gguf_q4_k_pack8(raw_b)
    x_f32 = (
        (np.arange(in_features, dtype=np.float32) % 29.0) - 14.0
    ).reshape(1, in_features) / 32.0
    x_u32 = x_f32.view(np.uint32).copy()
    x_u32 += 0x7FFF + ((x_u32 >> 16) & 1)
    x_bf16 = (x_u32 >> 16).astype(np.uint16)
    control_gate = np.empty((rows, out_features), dtype=np.uint16)
    control_up = np.empty_like(control_gate)
    control = np.empty_like(control_gate)
    candidate = np.empty_like(control_gate)
    q4_library = build_gguf_q4_k_gemv(load=True)
    silu_library = build_paro_silu(load=True)

    host_arrays = (
        x_bf16,
        packed_a.qweight,
        packed_a.scales,
        packed_a.mins,
        packed_b.qweight,
        packed_b.scales,
        packed_b.mins,
    )
    device_inputs = [malloc(array.nbytes) for array in host_arrays]
    device_outputs = [
        malloc(control_gate.nbytes),
        malloc(control_up.nbytes),
        malloc(control.nbytes),
        malloc(candidate.nbytes),
    ]
    try:
        for array, allocation in zip(host_arrays, device_inputs, strict=True):
            copy_host_to_device(allocation, host_array_ptr(array), array.nbytes)
        x_buf, qwa, sa, ma, qwb, sb, mb = device_inputs
        gate_buf, up_buf, control_buf, candidate_buf = device_outputs
        gguf_q4_k_pack8_dual_prefill_bf16_bf16_out(
            x_buf.ptr,
            qwa.ptr,
            sa.ptr,
            ma.ptr,
            qwb.ptr,
            sb.ptr,
            mb.ptr,
            gate_buf.ptr,
            up_buf.ptr,
            rows,
            in_features,
            out_features,
            threads=32,
            library=q4_library,
        )
        silu_mul_separate_out_bf16(
            gate_buf.ptr,
            up_buf.ptr,
            control_buf.ptr,
            rows,
            out_features,
            library=silu_library,
        )
        gguf_q4_k_pack8_dual_silu_bf16_bf16_out(
            x_buf.ptr,
            qwa.ptr,
            sa.ptr,
            ma.ptr,
            qwb.ptr,
            sb.ptr,
            mb.ptr,
            candidate_buf.ptr,
            rows,
            in_features,
            out_features,
            threads=32,
            library=q4_library,
        )
        for host, allocation in zip(
            (control_gate, control_up, control, candidate),
            device_outputs,
            strict=True,
        ):
            copy_device_to_host(host_array_ptr(host), allocation, host.nbytes)
    finally:
        for allocation in (*device_outputs, *device_inputs):
            free(allocation)

    np.testing.assert_array_equal(candidate, control)
