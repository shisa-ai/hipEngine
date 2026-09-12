"""Exact heterogeneous Q5/Q6 activation-register pair reuse for Laguna."""

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
from hipengine.kernels.hip_gfx1100.quant import gguf_k_gemv as kernel_mod
from hipengine.kernels.registry import (
    KernelKey,
    is_registered,
    registered_keys,
    resolve,
    unregister,
)
from hipengine.quant.gguf import GGMLQuantizationType
from tests._gguf_synthetic_weights import (
    Q5_K_BLOCK_BYTES,
    Q6_K_BLOCK_BYTES,
    make_q5_k_weight,
    make_q6_k_weight,
)

_VARIANT = "mixed_pair_reuse_local32_fixed_meta_pack8_gemv_decode_bf16_f32_out"
_QUANT = "gguf_q5_k+gguf_q6_k+gguf_q6_k+gguf_q5_k"
_WRAPPER = (
    "gguf_q5_q6_attention_q5_qg_mixed_pair_reuse_local32_fixed_meta_"
    "gemv_decode_bf16_f32_out"
)
_RETAINED = (
    kernel_mod.gguf_q5_q6_attention_q5_qg_mixed_local32_fixed_meta_gemv_decode_bf16_f32_out
)


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


HIP_AVAILABLE = _hip_available()


def _candidate():
    candidate = getattr(kernel_mod, _WRAPPER, None)
    assert callable(candidate), f"missing selected wrapper {_WRAPPER}"
    return candidate


def _bf16_to_f32(values: np.ndarray) -> np.ndarray:
    bits = np.ascontiguousarray(values, dtype=np.uint16)
    return (bits.astype(np.uint32) << 16).view(np.float32).reshape(bits.shape).copy()


def _f32_to_bf16(values: np.ndarray) -> np.ndarray:
    f32 = np.ascontiguousarray(values, dtype=np.float32)
    bits = f32.view(np.uint32).copy()
    rounded = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16)
    rounded[np.isnan(f32)] = 0x7FC0
    return rounded.reshape(f32.shape)


def _edge_bf16(in_features: int) -> np.ndarray:
    bits = np.asarray(
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
            0x7F00,
            0xFF00,
        ],
        dtype=np.uint16,
    )
    return np.resize(bits, (1, in_features)).copy()


def _edge_q5_weight(out_features: int, in_features: int) -> np.ndarray:
    weight = make_q5_k_weight(out_features, in_features)
    blocks = weight.reshape(out_features, in_features // 256, Q5_K_BLOCK_BYTES)
    signs = np.indices(blocks.shape[:2]).sum(axis=0) & 1
    d = blocks[..., 0:2].copy().view(np.uint16).reshape(blocks.shape[:2])
    dmin = blocks[..., 2:4].copy().view(np.uint16).reshape(blocks.shape[:2])
    d |= (signs.astype(np.uint16) << 15)
    dmin |= ((1 - signs).astype(np.uint16) << 15)
    blocks[..., 0:2] = d[..., None].view(np.uint8).reshape(*d.shape, 2)
    blocks[..., 2:4] = dmin[..., None].view(np.uint8).reshape(*dmin.shape, 2)
    blocks[..., 4:16] ^= np.resize(
        np.asarray([0x00, 0xFF, 0x55, 0xAA], dtype=np.uint8),
        blocks[..., 4:16].shape,
    )
    return weight


def _edge_q6_weight(out_features: int, in_features: int) -> np.ndarray:
    weight = make_q6_k_weight(out_features, in_features)
    blocks = weight.reshape(out_features, in_features // 256, Q6_K_BLOCK_BYTES)
    blocks[..., 192:208] = np.resize(
        np.asarray(
            [-128, 127, -1, 0, 1, -127, 126, -64, 63, -32, 31, -16, 15, -8, 7, -2],
            dtype=np.int8,
        ).view(np.uint8),
        blocks[..., 192:208].shape,
    )
    return weight


def _run_mixed(
    fn,
    x: np.ndarray,
    weights: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    *,
    library,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows, in_features = x.shape
    outputs = tuple(
        np.empty((rows, weight.shape[0]), dtype=np.float32) for weight in weights
    )
    x_buf = malloc(x.nbytes)
    weight_buffers = [malloc(weight.nbytes) for weight in weights]
    output_buffers = [malloc(output.nbytes) for output in outputs]
    try:
        copy_host_to_device(x_buf, host_array_ptr(x), x.nbytes)
        for weight, buffer in zip(weights, weight_buffers, strict=True):
            copy_host_to_device(buffer, host_array_ptr(weight), weight.nbytes)
        fn(
            x_buf.ptr,
            *(buffer.ptr for buffer in weight_buffers),
            *(buffer.ptr for buffer in output_buffers),
            rows,
            in_features,
            *(weight.shape[0] for weight in weights),
            library=library,
        )
        for output, buffer in zip(outputs, output_buffers, strict=True):
            copy_device_to_host(host_array_ptr(output), buffer, output.nbytes)
        return outputs
    finally:
        for buffer in reversed(output_buffers):
            free(buffer)
        for buffer in reversed(weight_buffers):
            free(buffer)
        free(x_buf)


def test_pair_reuse_registry_is_exact_and_gfx1100_only() -> None:
    candidate = _candidate()
    kernel_mod.register_gguf_k_gemv_kernels()
    assert resolve(
        backend="hip_gfx1100",
        layer="attention_projection_quad",
        quant=_QUANT,
        variant=_VARIANT,
    ) is candidate

    keys_before = set(registered_keys())
    try:
        from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels

        register_gfx1151_kernels()
        assert not is_registered(
            KernelKey("hip_gfx1151", "attention_projection_quad", _QUANT, _VARIANT)
        )
        assert not is_registered(
            KernelKey("cuda_sm86", "attention_projection_quad", _QUANT, _VARIANT)
        )
    finally:
        for key in set(registered_keys()) - keys_before:
            unregister(key)


def test_pair_reuse_wrapper_rejects_before_library_load(monkeypatch) -> None:
    candidate = _candidate()

    def fail_build(*args, **kwargs):
        raise AssertionError("invalid input reached library load")

    monkeypatch.setattr(kernel_mod, "build_gguf_k_gemv", fail_build)
    valid = dict(
        rows=1,
        in_features=256,
        q_features=16,
        k_features=8,
        v_features=8,
        gate_features=8,
    )
    with pytest.raises(ValueError, match="pointers must be non-zero"):
        candidate(0, *range(2, 10), **valid)
    with pytest.raises(ValueError, match="rows must be exactly 1"):
        candidate(*range(1, 10), **(valid | {"rows": 2}))
    with pytest.raises(ValueError, match="divisible by GGUF K block size 256"):
        candidate(*range(1, 10), **(valid | {"in_features": 257}))
    with pytest.raises(ValueError, match="divisible by 8"):
        candidate(*range(1, 10), **(valid | {"gate_features": 7}))
    with pytest.raises(ValueError, match="Q5 total output features"):
        candidate(
            *range(1, 10),
            **(
                valid
                | {
                    "q_features": 8,
                    "gate_features": 8,
                    "k_features": 16,
                    "v_features": 16,
                }
            ),
        )


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
@pytest.mark.parametrize(
    "in_features,dimensions",
    [
        (256, (8, 8, 8, 8)),
        (1024, (16, 8, 8, 8)),
        (3072, (6144, 1024, 1024, 48)),
        (9216, (8, 8, 8, 8)),
    ],
)
def test_pair_reuse_matches_retained_f32_bits_at_boundaries(
    in_features: int,
    dimensions: tuple[int, int, int, int],
) -> None:
    candidate = _candidate()
    x = _edge_bf16(in_features)
    weights = (
        _edge_q5_weight(dimensions[0], in_features),
        _edge_q6_weight(dimensions[1], in_features),
        _edge_q6_weight(dimensions[2], in_features),
        _edge_q5_weight(dimensions[3], in_features),
    )
    library = kernel_mod.build_gguf_k_gemv(load=True)
    retained = _run_mixed(_RETAINED, x, weights, library=library)
    actual = _run_mixed(candidate, x, weights, library=library)
    for observed, expected in zip(actual, retained, strict=True):
        np.testing.assert_array_equal(observed.view(np.uint32), expected.view(np.uint32))


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_pair_reuse_passes_cpu_kl_top1_gate() -> None:
    candidate = _candidate()
    rng = np.random.default_rng(20260731)
    rows, in_features = 10, 1024
    dimensions = (64, 16, 16, 8)
    x_bits = _f32_to_bf16(rng.normal(0.0, 0.2, size=(rows, in_features)))
    weights = (
        make_q5_k_weight(dimensions[0], in_features),
        make_q6_k_weight(dimensions[1], in_features),
        make_q6_k_weight(dimensions[2], in_features),
        make_q5_k_weight(dimensions[3], in_features),
    )
    library = kernel_mod.build_gguf_k_gemv(load=True)
    actual = tuple(
        np.concatenate(parts, axis=0)
        for parts in zip(
            *(
                _run_mixed(
                    candidate,
                    x_bits[row : row + 1],
                    weights,
                    library=library,
                )
                for row in range(rows)
            ),
            strict=True,
        )
    )
    x_f32 = _bf16_to_f32(x_bits)
    reference = (
        gguf_quant_gemv(x_f32, weights[0], GGMLQuantizationType.Q5_K),
        gguf_quant_gemv(x_f32, weights[1], GGMLQuantizationType.Q6_K),
        gguf_quant_gemv(x_f32, weights[2], GGMLQuantizationType.Q6_K),
        gguf_quant_gemv(x_f32, weights[3], GGMLQuantizationType.Q5_K),
    )
    result = evaluate_logits(
        np.concatenate(reference, axis=1),
        np.concatenate(actual, axis=1),
    )
    assert result.kl_mean <= 0.05
    assert result.top1_agreement >= 0.90
