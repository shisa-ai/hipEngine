"""Q5_K/Q6_K raw-resident MMQ32 prefill correctness and backend scope."""

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
from hipengine.kernels.cpu_reference import gguf_q5_k_gemv, gguf_q6_k_gemv
from hipengine.kernels.hip_gfx1100.quant.gguf_k_mmq_prefill import (
    build_gguf_k_mmq_prefill,
    gguf_q5_k_mmq32_q8_1_d4s4_f32_bf16_bf16_out,
    gguf_q5_k_mmq32_q8_1_d4s4_f32_bf16_f32_out,
    gguf_q5_k_mmq32_q8_1_d8s8_f32_bf16_bf16_out,
    gguf_q5_k_mmq32_q8_1_d8s8_f32_bf16_f32_out,
    gguf_q6_k_mmq32_q8_1_d4s4_f32_bf16_bf16_out,
    gguf_q6_k_mmq32_q8_1_d4s4_f32_bf16_f32_out,
    gguf_q6_k_mmq32_q8_1_d8s8_f32_bf16_bf16_out,
    gguf_q6_k_mmq32_q8_1_d8s8_f32_bf16_f32_out,
    gguf_q8_1_d4s4_f32_quantize_bf16,
    gguf_q8_1_d8s8_f32_quantize_bf16,
    plan_gguf_k_mmq_prefill_build,
    q8_1_d4s4_f32_nbytes,
    q8_1_d8s8_f32_nbytes,
    register_gguf_k_mmq_prefill_kernels,
)
from hipengine.kernels.registry import KernelKey, is_registered, resolve
from tests.test_gguf_k_gemv import make_q5_k_weight, make_q6_k_weight


_D4S4_BLOCK = 128
_D4S4_BYTES = 160


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def _bf16_bits(values: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(values, dtype=np.float32)
    bits = contiguous.view(np.uint32)
    lsb = (bits >> 16) & 1
    return ((bits + 0x7FFF + lsb) >> 16).astype(np.uint16)


def _bf16_to_f32(values: np.ndarray) -> np.ndarray:
    return (np.asarray(values, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def _round_away(values: np.ndarray) -> np.ndarray:
    return np.sign(values) * np.floor(np.abs(values) + np.float32(0.5))


def _pack_cpu(x_bf16: np.ndarray) -> np.ndarray:
    x = _bf16_to_f32(np.ascontiguousarray(x_bf16, dtype=np.uint16))
    rows, hidden = x.shape
    blocks = hidden // _D4S4_BLOCK
    packed = np.zeros((rows, blocks, _D4S4_BYTES), dtype=np.uint8)
    for row in range(rows):
        for block in range(blocks):
            values = x[row, block * 128 : (block + 1) * 128].reshape(4, 32)
            scales = np.zeros((4,), dtype=np.float32)
            sums = np.zeros((4,), dtype=np.float32)
            quants = np.zeros((4, 32), dtype=np.int8)
            for group in range(4):
                group_values = values[group]
                scales[group] = np.max(np.abs(group_values)) / np.float32(127.0)
                partial = np.asarray(
                    [
                        ((group_values[i] + group_values[i + 1]) + group_values[i + 2])
                        + group_values[i + 3]
                        for i in range(0, 32, 4)
                    ],
                    dtype=np.float32,
                )
                partial[:4] = partial[:4] + partial[4:]
                partial[:2] = partial[:2] + partial[2:4]
                sums[group] = partial[0] + partial[1]
                if scales[group] != 0.0:
                    quants[group] = np.clip(
                        _round_away(group_values / scales[group]), -128, 127
                    ).astype(np.int8)
            packed[row, block, :16] = scales.view(np.uint8)
            packed[row, block, 16:32] = sums.view(np.uint8)
            packed[row, block, 32:] = quants.reshape(128).view(np.uint8)
    return packed


def _unpack_cpu(packed: np.ndarray) -> np.ndarray:
    rows, blocks, _ = packed.shape
    scales = packed[..., :16].copy().view(np.float32).reshape(rows, blocks, 4)
    quants = (
        packed[..., 32:]
        .copy()
        .view(np.int8)
        .astype(np.float32)
        .reshape(rows, blocks, 4, 32)
    )
    return (quants * scales[..., None]).reshape(rows, blocks * 128)


def _quality(candidate: np.ndarray, reference: np.ndarray) -> tuple[float, float]:
    candidate = np.asarray(candidate, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)
    candidate_shifted = candidate - np.max(candidate, axis=1, keepdims=True)
    reference_shifted = reference - np.max(reference, axis=1, keepdims=True)
    candidate_prob = np.exp(candidate_shifted)
    reference_prob = np.exp(reference_shifted)
    candidate_prob /= np.sum(candidate_prob, axis=1, keepdims=True)
    reference_prob /= np.sum(reference_prob, axis=1, keepdims=True)
    kl = np.sum(
        reference_prob
        * (
            np.log(np.maximum(reference_prob, np.float32(1e-20)))
            - np.log(np.maximum(candidate_prob, np.float32(1e-20)))
        ),
        axis=1,
    )
    top1 = np.mean(np.argmax(candidate, axis=1) == np.argmax(reference, axis=1))
    return float(np.max(kl)), float(top1)


def _pack_d8_cpu(x_bf16: np.ndarray) -> np.ndarray:
    x = _bf16_to_f32(np.ascontiguousarray(x_bf16, dtype=np.uint16))
    rows, hidden = x.shape
    packed = np.zeros((rows, hidden // 128, 192), dtype=np.uint8)
    for row in range(rows):
        for block in range(hidden // 128):
            values = x[row, block * 128 : (block + 1) * 128].reshape(8, 16)
            scales = np.zeros((8,), dtype=np.float32)
            sums = np.zeros((8,), dtype=np.float32)
            quants = np.zeros((8, 16), dtype=np.int8)
            for group in range(8):
                group_values = values[group]
                scales[group] = np.max(np.abs(group_values)) / np.float32(127.0)
                partial = np.asarray(
                    [
                        ((group_values[i] + group_values[i + 1]) + group_values[i + 2])
                        + group_values[i + 3]
                        for i in range(0, 16, 4)
                    ],
                    dtype=np.float32,
                )
                partial[:2] = partial[:2] + partial[2:]
                sums[group] = partial[0] + partial[1]
                if scales[group] != 0.0:
                    quants[group] = np.clip(
                        _round_away(group_values / scales[group]), -128, 127
                    ).astype(np.int8)
            packed[row, block, :32] = scales.view(np.uint8)
            packed[row, block, 32:64] = sums.view(np.uint8)
            packed[row, block, 64:] = quants.reshape(128).view(np.uint8)
    return packed


def _unpack_d8_cpu(packed: np.ndarray) -> np.ndarray:
    rows, blocks, _ = packed.shape
    scales = packed[..., :32].copy().view(np.float32).reshape(rows, blocks, 8)
    quants = (
        packed[..., 64:]
        .copy()
        .view(np.int8)
        .astype(np.float32)
        .reshape(rows, blocks, 8, 16)
    )
    return (quants * scales[..., None]).reshape(rows, blocks * 128)


def test_q5_q6_mmq_contract_registry_and_backend_scope() -> None:
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels

    register_gguf_k_mmq_prefill_kernels()
    register_gfx1151_kernels(replace=True)
    assert q8_1_d4s4_f32_nbytes(33, 512) == 33 * 4 * 160
    assert q8_1_d8s8_f32_nbytes(33, 512) == 33 * 4 * 192
    with pytest.raises(ValueError, match="multiple of 128"):
        q8_1_d4s4_f32_nbytes(33, 384 + 64)
    with pytest.raises(ValueError, match="multiple of 256"):
        gguf_q5_k_mmq32_q8_1_d4s4_f32_bf16_bf16_out(1, 2, 3, 17, 128, 32)

    artifact = plan_gguf_k_mmq_prefill_build(compiler_version="test")
    assert artifact.output_path.name == "gguf_k_mmq_prefill.so"
    assert any(path.name == "gguf_k_mmq_prefill.hip" for path in artifact.sources)

    assert (
        resolve(
            backend="hip_gfx1100",
            layer="activation_quant",
            quant="q8_1_d4s4_f32",
            variant="bf16",
        )
        is gguf_q8_1_d4s4_f32_quantize_bf16
    )
    assert not is_registered(
        KernelKey(
            "hip_gfx1151",
            "activation_quant",
            "q8_1_d4s4_f32",
            "bf16",
        )
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="activation_quant",
            quant="q8_1_d8s8_f32",
            variant="bf16",
        )
        is gguf_q8_1_d8s8_f32_quantize_bf16
    )
    assert not is_registered(
        KernelKey(
            "hip_gfx1151",
            "activation_quant",
            "q8_1_d8s8_f32",
            "bf16",
        )
    )
    for quant, bf16_fn, f32_fn in (
        (
            "gguf_q5_k",
            gguf_q5_k_mmq32_q8_1_d4s4_f32_bf16_bf16_out,
            gguf_q5_k_mmq32_q8_1_d4s4_f32_bf16_f32_out,
        ),
        (
            "gguf_q6_k",
            gguf_q6_k_mmq32_q8_1_d4s4_f32_bf16_bf16_out,
            gguf_q6_k_mmq32_q8_1_d4s4_f32_bf16_f32_out,
        ),
    ):
        policy_key = KernelKey(
            "hip_gfx1100",
            "linear_prefill_policy",
            quant,
            "raw_k_q8_1_mmq32",
        )
        assert is_registered(policy_key)
        assert not is_registered(
            KernelKey(
                "hip_gfx1151",
                policy_key.layer,
                policy_key.quant,
                policy_key.variant,
            )
        )
        assert (
            resolve(
                backend="hip_gfx1100",
                layer="linear",
                quant=quant,
                variant="mmq32_q8_1_d4s4_f32_bf16_bf16_out",
            )
            is bf16_fn
        )
        assert (
            resolve(
                backend="hip_gfx1100",
                layer="linear",
                quant=quant,
                variant="mmq32_q8_1_d4s4_f32_bf16_f32_out",
            )
            is f32_fn
        )
        assert not is_registered(
            KernelKey(
                "hip_gfx1151",
                "linear",
                quant,
                "mmq32_q8_1_d4s4_f32_bf16_bf16_out",
            )
        )
    for quant, bf16_fn, f32_fn in (
        (
            "gguf_q5_k",
            gguf_q5_k_mmq32_q8_1_d8s8_f32_bf16_bf16_out,
            gguf_q5_k_mmq32_q8_1_d8s8_f32_bf16_f32_out,
        ),
        (
            "gguf_q6_k",
            gguf_q6_k_mmq32_q8_1_d8s8_f32_bf16_bf16_out,
            gguf_q6_k_mmq32_q8_1_d8s8_f32_bf16_f32_out,
        ),
    ):
        for output_dtype, fn in (("bf16", bf16_fn), ("f32", f32_fn)):
            key = KernelKey(
                "hip_gfx1100",
                "linear",
                quant,
                f"mmq32_q8_1_d8s8_f32_bf16_{output_dtype}_out",
            )
            assert resolve(
                backend=key.backend,
                layer=key.layer,
                quant=key.quant,
                variant=key.variant,
            ) is fn
            assert not is_registered(
                KernelKey("hip_gfx1151", key.layer, key.quant, key.variant)
            )


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("rows", [17, 33])
@pytest.mark.parametrize("quant", ["q5", "q6"])
@pytest.mark.parametrize("layout", ["d4", "d8"])
def test_q5_q6_mmq_matches_q8_oracle_and_quality_gate(
    rows: int,
    quant: str,
    layout: str,
) -> None:
    from hipengine.core.hip import get_hip_runtime

    hidden, out_features = 512, 48
    rng = np.random.default_rng(
        20260729 + rows + (quant == "q6") + (layout == "d8")
    )
    x_bf16 = _bf16_bits(rng.normal(0.0, 0.125, size=(rows, hidden)).astype(np.float32))
    packed = np.zeros(
        (rows, hidden // 128, 160 if layout == "d4" else 192),
        dtype=np.uint8,
    )
    out_bf16 = np.zeros((rows, out_features), dtype=np.uint16)
    out_f32 = np.zeros((rows, out_features), dtype=np.float32)
    if quant == "q5":
        qweight = make_q5_k_weight(out_features, hidden)
        reference_fn = gguf_q5_k_gemv
        bf16_wrapper = gguf_q5_k_mmq32_q8_1_d4s4_f32_bf16_bf16_out
        f32_wrapper = gguf_q5_k_mmq32_q8_1_d4s4_f32_bf16_f32_out
        if layout == "d8":
            bf16_wrapper = gguf_q5_k_mmq32_q8_1_d8s8_f32_bf16_bf16_out
            f32_wrapper = gguf_q5_k_mmq32_q8_1_d8s8_f32_bf16_f32_out
    else:
        qweight = make_q6_k_weight(out_features, hidden)
        reference_fn = gguf_q6_k_gemv
        bf16_wrapper = gguf_q6_k_mmq32_q8_1_d4s4_f32_bf16_bf16_out
        f32_wrapper = gguf_q6_k_mmq32_q8_1_d4s4_f32_bf16_f32_out
        if layout == "d8":
            bf16_wrapper = gguf_q6_k_mmq32_q8_1_d8s8_f32_bf16_bf16_out
            f32_wrapper = gguf_q6_k_mmq32_q8_1_d8s8_f32_bf16_f32_out
    quantize = (
        gguf_q8_1_d4s4_f32_quantize_bf16
        if layout == "d4"
        else gguf_q8_1_d8s8_f32_quantize_bf16
    )

    runtime = get_hip_runtime()
    library = build_gguf_k_mmq_prefill(load=True)
    buffers = []
    try:
        x_dev = malloc(x_bf16.nbytes, runtime=runtime)
        packed_dev = malloc(packed.nbytes, runtime=runtime)
        weight_dev = malloc(qweight.nbytes, runtime=runtime)
        bf16_dev = malloc(out_bf16.nbytes, runtime=runtime)
        f32_dev = malloc(out_f32.nbytes, runtime=runtime)
        buffers.extend((x_dev, packed_dev, weight_dev, bf16_dev, f32_dev))
        copy_host_to_device(x_dev, host_array_ptr(x_bf16), runtime=runtime)
        copy_host_to_device(weight_dev, host_array_ptr(qweight), runtime=runtime)
        quantize(
            x_dev.ptr,
            packed_dev.ptr,
            rows,
            hidden,
            library=library,
            runtime=runtime,
        )
        bf16_wrapper(
            packed_dev.ptr,
            weight_dev.ptr,
            bf16_dev.ptr,
            rows,
            hidden,
            out_features,
            library=library,
            runtime=runtime,
        )
        f32_wrapper(
            packed_dev.ptr,
            weight_dev.ptr,
            f32_dev.ptr,
            rows,
            hidden,
            out_features,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(packed), packed_dev, runtime=runtime)
        copy_device_to_host(host_array_ptr(out_bf16), bf16_dev, runtime=runtime)
        copy_device_to_host(host_array_ptr(out_f32), f32_dev, runtime=runtime)
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)

    expected_packed = _pack_cpu(x_bf16) if layout == "d4" else _pack_d8_cpu(x_bf16)
    np.testing.assert_array_equal(packed, expected_packed)
    reconstructed = _unpack_cpu(packed) if layout == "d4" else _unpack_d8_cpu(packed)
    q8_reference = reference_fn(reconstructed, qweight)
    np.testing.assert_allclose(out_f32, q8_reference, rtol=2e-2, atol=1e-2)
    np.testing.assert_array_equal(out_bf16, _bf16_bits(out_f32))

    exact_reference = reference_fn(_bf16_to_f32(x_bf16), qweight)
    max_kl, top1 = _quality(_bf16_to_f32(out_bf16), exact_reference)
    assert max_kl <= 0.05
    assert top1 >= 0.9
