from __future__ import annotations

import numpy as np
import pytest

from hipengine.kernels.cpu_reference import gguf_q4_k_gemv, gguf_q4_k_pack8_gemv
from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_gemv import (
    build_gguf_q4_k_gemv,
    gguf_q4_k_gemv_bf16_bf16_out,
    gguf_q4_k_gemv_bf16_f32_out,
    gguf_q4_k_gemv_f32_f32_out,
    gguf_q4_k_gemv_fp16_f32_out,
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
