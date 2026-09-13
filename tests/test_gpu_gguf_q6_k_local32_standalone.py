"""Exact standalone local32 Q6_K decode sibling selected for Laguna."""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

from hipengine.benchmark.correctness import evaluate_logits
from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.cpu_reference import gguf_quant_gemv
from hipengine.kernels.hip_gfx1100.quant.gguf_k_gemv import (
    build_gguf_k_gemv,
    gguf_q6_k_gemv_bf16_bf16_out,
    gguf_q6_k_wave32x2_fixed_meta_gemv_decode_bf16_bf16_out,
    register_gguf_k_gemv_kernels,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q6_k_pack8_gemv import (
    build_gguf_q6_k_pack8_gemv,
    gguf_q6_k_pack8_gemv_decode_bf16_bf16_out,
)
from hipengine.kernels.registry import (
    KernelKey,
    is_registered,
    registered_keys,
    resolve,
    unregister,
)
from hipengine.quant.gguf import GGMLQuantizationType
from tests._gguf_synthetic_weights import make_q6_k_weight

_VARIANT = "standalone_wave32x2_fixed_meta_gemv_decode_bf16_bf16_out"
_Q5_FIXED_META_VARIANT = "wave32x2_fixed_meta_gemv_decode_bf16_bf16_out"
_Q6_K_BLOCK_BYTES = 210


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


HIP_AVAILABLE = _hip_available()


@pytest.fixture(scope="module")
def q6_libraries():
    if not HIP_AVAILABLE:
        pytest.skip("HIP runtime is not available")
    return build_gguf_k_gemv(load=True), build_gguf_q6_k_pack8_gemv(load=True)


def _run_dense(fn, x: np.ndarray, qweight: np.ndarray, *, library) -> np.ndarray:
    rows, in_features = x.shape
    out = np.empty((rows, qweight.shape[0]), dtype=np.uint16)
    buffers = [malloc(x.nbytes), malloc(qweight.nbytes), malloc(out.nbytes)]
    x_buf, weight_buf, out_buf = buffers
    try:
        copy_host_to_device(x_buf, host_array_ptr(x), x.nbytes)
        copy_host_to_device(weight_buf, host_array_ptr(qweight), qweight.nbytes)
        fn(
            x_buf.ptr,
            weight_buf.ptr,
            out_buf.ptr,
            rows,
            in_features,
            qweight.shape[0],
            library=library,
        )
        copy_device_to_host(host_array_ptr(out), out_buf, out.nbytes)
        return out
    finally:
        for buffer in reversed(buffers):
            free(buffer)


def _edge_q6_weight(out_features: int, in_features: int) -> np.ndarray:
    qweight = make_q6_k_weight(out_features, in_features)
    ql_patterns = (0x00, 0xFF, 0x5A, 0xA5)
    qh_patterns = (0x00, 0xFF, 0x33, 0xCC)
    scale_patterns = np.asarray(
        [
            [-128, 127, -1, 0, 1, -127, 126, -64, 63, -32, 31, -16, 15, -8, 7, -2],
            [127, -128, 0, -1, -127, 1, -64, 126, -32, 63, -16, 31, -8, 15, -2, 7],
        ],
        dtype=np.int8,
    )
    blocks_per_row = in_features // 256
    for out_idx in range(out_features):
        for block_idx in range(blocks_per_row):
            start = block_idx * _Q6_K_BLOCK_BYTES
            pattern = out_idx + block_idx
            qweight[out_idx, start : start + 128] = ql_patterns[pattern % 4]
            qweight[out_idx, start + 128 : start + 192] = qh_patterns[(pattern * 3) % 4]
            qweight[out_idx, start + 192 : start + 208] = scale_patterns[pattern % 2].view(
                np.uint8
            )
    return qweight


def _edge_bf16(in_features: int) -> np.ndarray:
    values = np.asarray(
        [
            0x0000,
            0x8000,
            0x0001,
            0x8001,
            0x007F,
            0x807F,
            0x0080,
            0x8080,
            0x3F80,
            0xBF80,
            0x4700,
            0xC700,
            0x7F7F,
            0xFF7F,
        ],
        dtype=np.uint16,
    )
    return np.resize(values, (1, in_features)).copy()


def _bf16_to_f32(values: np.ndarray) -> np.ndarray:
    bits = np.ascontiguousarray(values, dtype=np.uint16)
    return (bits.astype(np.uint32) << 16).view(np.float32).reshape(bits.shape).copy()


def _f32_to_bf16(values: np.ndarray) -> np.ndarray:
    f32 = np.ascontiguousarray(values, dtype=np.float32)
    bits = f32.view(np.uint32).copy()
    rounded = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16)
    rounded[np.isnan(f32)] = 0x7FC0
    return rounded.reshape(f32.shape)


def test_q6_local32_standalone_registry_is_gfx1100_only() -> None:
    register_gguf_k_gemv_kernels()
    assert resolve(
        backend="hip_gfx1100",
        layer="linear",
        quant="gguf_q6_k",
        variant=_VARIANT,
    ) is gguf_q6_k_wave32x2_fixed_meta_gemv_decode_bf16_bf16_out
    assert not is_registered(
        KernelKey(
            "hip_gfx1100",
            "linear",
            "gguf_q6_k",
            _Q5_FIXED_META_VARIANT,
        )
    )

    keys_before = set(registered_keys())
    try:
        from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels

        register_gfx1151_kernels()
        assert not is_registered(
            KernelKey("hip_gfx1151", "linear", "gguf_q6_k", _VARIANT)
        )
    finally:
        for key in set(registered_keys()) - keys_before:
            unregister(key)


def test_q6_local32_standalone_rejects_out_of_scope_shapes() -> None:
    with pytest.raises(ValueError, match="rows must be exactly 1"):
        gguf_q6_k_wave32x2_fixed_meta_gemv_decode_bf16_bf16_out(
            1, 2, 3, rows=2, in_features=256, out_features=8
        )
    with pytest.raises(ValueError, match="divisible by 2"):
        gguf_q6_k_wave32x2_fixed_meta_gemv_decode_bf16_bf16_out(
            1, 2, 3, rows=1, in_features=256, out_features=7
        )
    with pytest.raises(ValueError, match="threads must be 32"):
        gguf_q6_k_wave32x2_fixed_meta_gemv_decode_bf16_bf16_out(
            1, 2, 3, rows=1, in_features=256, out_features=8, threads=128
        )


def test_rows_greater_than_one_stays_on_registered_pack8_body() -> None:
    assert resolve(
        backend="hip_gfx1100",
        layer="linear",
        quant="gguf_q6_k",
        variant="pack8_gemv_decode_bf16_bf16_out",
    ) is gguf_q6_k_pack8_gemv_decode_bf16_bf16_out


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
@pytest.mark.parametrize(
    "in_features,out_features",
    [
        (256, 2),
        (1024, 8),
        (3072, 8),
        (9216, 8),
        (12288, 8),
        (256, 1024),
        (256, 3072),
    ],
)
def test_q6_local32_standalone_is_bf16_bit_exact_at_boundaries(
    in_features: int,
    out_features: int,
    q6_libraries,
) -> None:
    candidate_library, retained_pack8_library = q6_libraries
    x = _edge_bf16(in_features)
    qweight = _edge_q6_weight(out_features, in_features)
    if out_features == 2:
        retained = _run_dense(
            gguf_q6_k_gemv_bf16_bf16_out,
            x,
            qweight,
            library=candidate_library,
        )
    else:
        retained = _run_dense(
            gguf_q6_k_pack8_gemv_decode_bf16_bf16_out,
            x,
            qweight,
            library=retained_pack8_library,
        )
    candidate = _run_dense(
        gguf_q6_k_wave32x2_fixed_meta_gemv_decode_bf16_bf16_out,
        x,
        qweight,
        library=candidate_library,
    )
    np.testing.assert_array_equal(candidate, retained)


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_q6_local32_standalone_passes_cpu_kl_top1_gate(q6_libraries) -> None:
    candidate_library, _ = q6_libraries
    rng = np.random.default_rng(20260730)
    in_features, out_features = 1024, 64
    qweight = make_q6_k_weight(out_features, in_features)
    x_bits = _f32_to_bf16(rng.normal(0.0, 0.2, size=(10, in_features)))
    candidate_bits = np.concatenate(
        [
            _run_dense(
                gguf_q6_k_wave32x2_fixed_meta_gemv_decode_bf16_bf16_out,
                x_bits[row : row + 1],
                qweight,
                library=candidate_library,
            )
            for row in range(x_bits.shape[0])
        ],
        axis=0,
    )
    reference = gguf_quant_gemv(
        _bf16_to_f32(x_bits),
        qweight,
        GGMLQuantizationType.Q6_K,
    )
    result = evaluate_logits(reference, _bf16_to_f32(candidate_bits))
    assert result.kl_mean <= 0.05
    assert result.top1_agreement >= 0.90
