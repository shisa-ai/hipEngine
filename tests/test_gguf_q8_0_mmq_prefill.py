"""Correctness and policy gates for raw Q8_0 MMQ128 prefill."""

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
from hipengine.kernels.cpu_reference import gguf_q8_0_gemv
from hipengine.kernels.hip_gfx1100.quant.gguf_k_gemv import (
    build_gguf_k_gemv,
    gguf_q8_0_exact_prefill_tile16x4_bf16_bf16_out,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q8_0_mmq_prefill import (
    Q8_MMQ_PREFILL_POLICY_KEY,
    QWEN4EXP_Q8_MMQ_PREFILL_POLICY,
    UD_Q3_K_M_Q8_MMQ_PREFILL_POLICY,
    build_gguf_q8_0_mmq_prefill,
    gguf_q8_0_mmq128_prefill_q8_1_d4_bf16_bf16_out,
    gguf_q8_0_mmq128_prefill_q8_1_d4x2_bf16_bf16_out,
    gguf_q8_0_mmq128_prefill_q8_1_d4x3_bf16_bf16_out,
    gguf_q8_0_mmq128_prefill_q8_1_d4x3_guarded_bf16_bf16_out,
    gguf_q8_0_mmq128_prefill_q8_1_d4x3_guarded_f32_f32_out,
    gguf_q8_0_mmq128_quantize_bf16_d4,
    gguf_q8_0_mmq128_quantize_bf16_d4x2,
    gguf_q8_0_mmq128_quantize_bf16_d4x3,
    gguf_q8_0_mmq128_quantize_f32_d4x3,
    plan_gguf_q8_0_mmq_prefill_build,
    q8_mmq_d4_nbytes,
    q8_mmq_d4x2_nbytes,
    q8_mmq_d4x3_nbytes,
    gguf_q8_0_mmq128_sparse_exact_correct_bf16,
    gguf_q8_0_mmq128_sparse_exact_correct_f32,
    ud_q3_k_m_q8_mmq_prefill_policy,
)
from hipengine.kernels.registry import resolve
from tests.test_gguf_k_gemv import make_q8_0_weight


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def _bf16_bits(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32, copy=False)
    bits = values.view(np.uint32)
    lsb = (bits >> 16) & 1
    return ((bits + 0x7FFF + lsb) >> 16).astype(np.uint16)


def _bf16_to_f32(values: np.ndarray) -> np.ndarray:
    return (values.astype(np.uint32) << 16).view(np.float32)


def _round_away(values: np.ndarray) -> np.ndarray:
    return np.sign(values) * np.floor(np.abs(values) + 0.5)


def _pack_d4_cpu(x_bf16: np.ndarray) -> np.ndarray:
    x = _bf16_to_f32(np.ascontiguousarray(x_bf16, dtype=np.uint16))
    rows, hidden = x.shape
    blocks = x.reshape(rows, hidden // 128, 4, 32)
    amax = np.max(np.abs(blocks), axis=-1)
    d = (amax / np.float32(127.0)).astype(np.float32)
    q = np.where(d[..., None] > 0.0, _round_away(blocks / d[..., None]), 0.0)
    q = np.clip(q, -128, 127).astype(np.int8)
    nblocks = hidden // 128
    packed = np.empty((nblocks, rows, 144), dtype=np.uint8)
    packed[..., :16] = d.transpose(1, 0, 2).reshape(nblocks, rows, 4).view(np.uint8)
    packed[..., 16:] = q.transpose(1, 0, 2, 3).reshape(nblocks, rows, 128).view(np.uint8)
    return packed


def _decode_d4_cpu(packed: np.ndarray) -> np.ndarray:
    nblocks, rows, _ = packed.shape
    d = packed[..., :16].copy().view(np.float32).reshape(nblocks, rows, 4).transpose(1, 0, 2)
    q = (
        packed[..., 16:]
        .view(np.int8)
        .astype(np.float32)
        .reshape(nblocks, rows, 4, 32)
        .transpose(1, 0, 2, 3)
    )
    return (q * d[..., None]).reshape(rows, nblocks * 128)


def _activation(rows: int, hidden: int, seed: int) -> np.ndarray:
    values = np.arange(rows * hidden, dtype=np.float32).reshape(rows, hidden) + seed
    return ((values % 17) - 8) / 11.0


def test_raw_q8_mmq_registry_build_and_contract() -> None:
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="activation_quant",
            quant="q8_1_d4x3",
            variant="bf16",
        )
        is gguf_q8_0_mmq128_quantize_bf16_d4x3
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="linear",
            quant="gguf_q8_0",
            variant="mmq128_prefill_q8_1_d4x3_guarded_bf16_bf16_out",
        )
        is gguf_q8_0_mmq128_prefill_q8_1_d4x3_guarded_bf16_bf16_out
    )
    assert (
        resolve(
            backend=Q8_MMQ_PREFILL_POLICY_KEY.backend,
            layer=Q8_MMQ_PREFILL_POLICY_KEY.layer,
            quant=Q8_MMQ_PREFILL_POLICY_KEY.quant,
            variant=Q8_MMQ_PREFILL_POLICY_KEY.variant,
        )
        is UD_Q3_K_M_Q8_MMQ_PREFILL_POLICY
    )
    artifact = plan_gguf_q8_0_mmq_prefill_build(compiler_version="test")
    assert artifact.profile.name == "baseline"
    assert "-mcumode" not in artifact.command
    assert artifact.output_path.name == "gguf_q8_0_mmq_prefill.so"
    assert any(path.name == "gguf_q8_0_mmq_prefill.hip" for path in artifact.sources)

    assert q8_mmq_d4_nbytes(512, 2048) == 1_179_648
    assert q8_mmq_d4x2_nbytes(512, 2048) == 2_359_296
    assert q8_mmq_d4x3_nbytes(512, 2048) == 3_538_944
    assert UD_Q3_K_M_Q8_MMQ_PREFILL_POLICY.risk_threshold == 1.0e-5
    assert UD_Q3_K_M_Q8_MMQ_PREFILL_POLICY.risk_indices_nbytes(512) == 16_777_216
    with pytest.raises(ValueError, match="multiple of 256"):
        gguf_q8_0_mmq128_prefill_q8_1_d4_bf16_bf16_out(1, 2, 3, 17, 128, 80)
    with pytest.raises(ValueError, match="multiple of 128"):
        gguf_q8_0_mmq128_quantize_bf16_d4(1, 2, 17, 96)


@pytest.mark.parametrize(
    ("rows", "hidden", "out_features", "expected"),
    [
        (31, 2048, 8192, False),
        (32, 2048, 8192, True),
        (47, 2048, 4096, False),
        (48, 2048, 4096, True),
        (95, 4096, 2048, False),
        (96, 4096, 2048, False),
        (512, 512, 2048, False),
        (512, 2048, 512, False),
        (512, 1024, 2048, False),
        (4096, 2048, 8192, True),
        (4097, 2048, 8192, False),
    ],
)
def test_ud_q3_k_m_policy_uses_measured_shape_crossovers(
    rows: int,
    hidden: int,
    out_features: int,
    expected: bool,
) -> None:
    assert ud_q3_k_m_q8_mmq_prefill_policy(rows, hidden, out_features) is expected


def _run_gpu(
    rows: int,
    hidden: int,
    out_features: int,
    *,
    residual_passes: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_bf16 = _bf16_bits(_activation(rows, hidden, seed=47))
    qweight = np.ascontiguousarray(make_q8_0_weight(out_features, hidden), dtype=np.uint8)
    d4 = np.zeros((residual_passes * hidden // 128, rows, 144), dtype=np.uint8)
    out = np.zeros((rows, out_features), dtype=np.uint16)

    library = build_gguf_q8_0_mmq_prefill(load=True)
    from hipengine.core.hip import get_hip_runtime

    runtime = get_hip_runtime()
    bufs = []
    try:
        x_dev = malloc(x_bf16.nbytes, runtime=runtime)
        d4_dev = malloc(d4.nbytes, runtime=runtime)
        weight_dev = malloc(qweight.nbytes, runtime=runtime)
        out_dev = malloc(out.nbytes, runtime=runtime)
        bufs.extend((x_dev, d4_dev, weight_dev, out_dev))
        copy_host_to_device(x_dev, host_array_ptr(x_bf16), runtime=runtime)
        copy_host_to_device(weight_dev, host_array_ptr(qweight), runtime=runtime)
        if residual_passes == 1:
            gguf_q8_0_mmq128_quantize_bf16_d4(
                x_dev.ptr, d4_dev.ptr, rows, hidden, library=library, runtime=runtime
            )
            gguf_q8_0_mmq128_prefill_q8_1_d4_bf16_bf16_out(
                d4_dev.ptr,
                weight_dev.ptr,
                out_dev.ptr,
                rows,
                hidden,
                out_features,
                library=library,
                runtime=runtime,
            )
        elif residual_passes == 2:
            gguf_q8_0_mmq128_quantize_bf16_d4x2(
                x_dev.ptr, d4_dev.ptr, rows, hidden, library=library, runtime=runtime
            )
            gguf_q8_0_mmq128_prefill_q8_1_d4x2_bf16_bf16_out(
                d4_dev.ptr,
                weight_dev.ptr,
                out_dev.ptr,
                rows,
                hidden,
                out_features,
                library=library,
                runtime=runtime,
            )
        elif residual_passes == 3:
            gguf_q8_0_mmq128_quantize_bf16_d4x3(
                x_dev.ptr, d4_dev.ptr, rows, hidden, library=library, runtime=runtime
            )
            gguf_q8_0_mmq128_prefill_q8_1_d4x3_bf16_bf16_out(
                d4_dev.ptr,
                weight_dev.ptr,
                out_dev.ptr,
                rows,
                hidden,
                out_features,
                library=library,
                runtime=runtime,
            )
        else:
            raise ValueError(f"unsupported residual_passes={residual_passes}")
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(d4), d4_dev, runtime=runtime)
        copy_device_to_host(host_array_ptr(out), out_dev, runtime=runtime)
    finally:
        for buf in reversed(bufs):
            free(buf, runtime=runtime)

    nblocks = hidden // 128
    reconstructed = sum(
        (_decode_d4_cpu(d4[plane * nblocks : (plane + 1) * nblocks]) for plane in range(residual_passes)),
        start=np.zeros((rows, hidden), dtype=np.float32),
    )
    reference = gguf_q8_0_gemv(reconstructed, qweight)
    return _bf16_to_f32(out), reference, d4


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize(("rows", "hidden", "out_features"), [(32, 256, 128), (17, 256, 80)])
def test_raw_q8_mmq_matches_d4_cpu_oracle_and_quality_gate(
    rows: int,
    hidden: int,
    out_features: int,
) -> None:
    actual, reference, packed = _run_gpu(rows, hidden, out_features)
    expected_packed = _pack_d4_cpu(_bf16_bits(_activation(rows, hidden, seed=47)))
    np.testing.assert_array_equal(packed, expected_packed)

    relative_l2 = float(np.linalg.norm(actual - reference) / max(np.linalg.norm(reference), 1e-12))
    assert relative_l2 <= 0.02

    actual_shifted = actual - np.max(actual, axis=1, keepdims=True)
    reference_shifted = reference - np.max(reference, axis=1, keepdims=True)
    actual_prob = np.exp(actual_shifted)
    reference_prob = np.exp(reference_shifted)
    actual_prob /= np.sum(actual_prob, axis=1, keepdims=True)
    reference_prob /= np.sum(reference_prob, axis=1, keepdims=True)
    kl = np.sum(
        reference_prob
        * (np.log(np.maximum(reference_prob, 1e-20)) - np.log(np.maximum(actual_prob, 1e-20))),
        axis=1,
    )
    assert float(np.mean(kl)) <= 0.05
    assert float(np.mean(np.argmax(actual, axis=1) == np.argmax(reference, axis=1))) >= 0.9


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize(
    ("rows", "hidden", "out_features"),
    [(32, 256, 128), (17, 256, 80), (17, 2048, 80)],
)
def test_raw_q8_mmq_residual_d4x2_reconstructs_bf16_and_matches_cpu_oracle(
    rows: int,
    hidden: int,
    out_features: int,
) -> None:
    actual, packed_reference, packed = _run_gpu(
        rows,
        hidden,
        out_features,
        residual_passes=2,
    )
    nblocks = hidden // 128
    x_bf16 = _bf16_bits(_activation(rows, hidden, seed=47))
    np.testing.assert_array_equal(packed[:nblocks], _pack_d4_cpu(x_bf16))

    reconstructed = _decode_d4_cpu(packed[:nblocks]) + _decode_d4_cpu(packed[nblocks:])
    x_f32 = _bf16_to_f32(x_bf16)
    reconstruction_l2 = float(
        np.linalg.norm(reconstructed - x_f32) / max(np.linalg.norm(x_f32), 1e-12)
    )
    assert reconstruction_l2 <= 2e-4

    np.testing.assert_allclose(actual, packed_reference, rtol=8e-3, atol=0.2)
    qweight = np.ascontiguousarray(make_q8_0_weight(out_features, hidden), dtype=np.uint8)
    exact_reference = gguf_q8_0_gemv(x_f32, qweight)
    relative_l2 = float(
        np.linalg.norm(actual - exact_reference) / max(np.linalg.norm(exact_reference), 1e-12)
    )
    assert relative_l2 <= 2e-3


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_raw_q8_mmq_residual_d4x3_reduces_reconstruction_error() -> None:
    rows, hidden, out_features = 17, 2048, 80
    actual, packed_reference, packed = _run_gpu(
        rows,
        hidden,
        out_features,
        residual_passes=3,
    )
    nblocks = hidden // 128
    x_bf16 = _bf16_bits(_activation(rows, hidden, seed=47))
    np.testing.assert_array_equal(packed[:nblocks], _pack_d4_cpu(x_bf16))

    reconstructed = sum(
        (_decode_d4_cpu(packed[plane * nblocks : (plane + 1) * nblocks]) for plane in range(3)),
        start=np.zeros((rows, hidden), dtype=np.float32),
    )
    x_f32 = _bf16_to_f32(x_bf16)
    reconstruction_l2 = float(
        np.linalg.norm(reconstructed - x_f32) / max(np.linalg.norm(x_f32), 1e-12)
    )
    assert reconstruction_l2 <= 2e-6
    np.testing.assert_allclose(actual, packed_reference, rtol=8e-3, atol=0.2)


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_raw_q8_mmq_guarded_sparse_correction_matches_exact_tile() -> None:
    rows, hidden, out_features = 32, 256, 128
    x_bf16 = _bf16_bits(_activation(rows, hidden, seed=91))
    qweight = np.ascontiguousarray(make_q8_0_weight(out_features, hidden), dtype=np.uint8)
    out = np.zeros((rows, out_features), dtype=np.uint16)
    exact = np.zeros_like(out)
    risk_count = np.zeros((1,), dtype=np.int32)
    risk_indices = np.zeros((rows * out_features,), dtype=np.int32)

    library = build_gguf_q8_0_mmq_prefill(load=True)
    exact_library = build_gguf_k_gemv(load=True)
    from hipengine.core.hip import get_hip_runtime

    runtime = get_hip_runtime()
    bufs = []
    try:
        x_dev = malloc(x_bf16.nbytes, runtime=runtime)
        d4_dev = malloc(q8_mmq_d4x3_nbytes(rows, hidden), runtime=runtime)
        weight_dev = malloc(qweight.nbytes, runtime=runtime)
        out_dev = malloc(out.nbytes, runtime=runtime)
        exact_dev = malloc(exact.nbytes, runtime=runtime)
        count_dev = malloc(risk_count.nbytes, runtime=runtime)
        indices_dev = malloc(risk_indices.nbytes, runtime=runtime)
        bufs.extend((x_dev, d4_dev, weight_dev, out_dev, exact_dev, count_dev, indices_dev))
        copy_host_to_device(x_dev, host_array_ptr(x_bf16), runtime=runtime)
        copy_host_to_device(weight_dev, host_array_ptr(qweight), runtime=runtime)
        runtime.memset(count_dev.ptr, 0, count_dev.nbytes)
        gguf_q8_0_mmq128_quantize_bf16_d4x3(
            x_dev.ptr, d4_dev.ptr, rows, hidden, library=library, runtime=runtime
        )
        gguf_q8_0_mmq128_prefill_q8_1_d4x3_guarded_bf16_bf16_out(
            d4_dev.ptr,
            weight_dev.ptr,
            out_dev.ptr,
            count_dev.ptr,
            indices_dev.ptr,
            rows * out_features,
            float("inf"),
            rows,
            hidden,
            out_features,
            library=library,
            runtime=runtime,
        )
        gguf_q8_0_mmq128_sparse_exact_correct_bf16(
            x_dev.ptr,
            weight_dev.ptr,
            out_dev.ptr,
            count_dev.ptr,
            indices_dev.ptr,
            rows * out_features,
            rows,
            hidden,
            out_features,
            library=library,
            runtime=runtime,
        )
        gguf_q8_0_exact_prefill_tile16x4_bf16_bf16_out(
            x_dev.ptr,
            weight_dev.ptr,
            exact_dev.ptr,
            rows,
            hidden,
            out_features,
            library=exact_library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(out), out_dev, runtime=runtime)
        copy_device_to_host(host_array_ptr(exact), exact_dev, runtime=runtime)
        copy_device_to_host(host_array_ptr(risk_count), count_dev, runtime=runtime)
    finally:
        for buf in reversed(bufs):
            free(buf, runtime=runtime)

    assert int(risk_count[0]) == rows * out_features
    np.testing.assert_array_equal(out, exact)


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_qwen4exp_f32_mmq_chain_bounded_against_exact_f32_owner() -> None:
    """The F32 in/out MMQ chain stays near the exact coltile owner.

    Qwen4Exp prefill launches Q8_0 projections with F32 activations and
    outputs; the guarded chain quantizes to D4x3, multiplies via MMQ128, and
    (at the Qwen4Exp risk threshold of zero) repairs nothing. The result must
    be deterministic, top-1 equal to the exact owner, and within a small
    relative envelope of it.
    """

    from hipengine.core.hip import get_hip_runtime
    from hipengine.kernels.hip_gfx1100.quant.gguf_k_gemv import (
        gguf_q8_0_gemv_coltile8_rowbatch4_f32_f32_out,
    )

    rows, hidden, out_features = 128, 2560, 1024
    rng = np.random.default_rng(21)
    qweight = np.ascontiguousarray(
        make_q8_0_weight(out_features, hidden), dtype=np.uint8
    )
    x = (rng.standard_normal((rows, hidden)) * 0.5).astype(np.float32)
    runtime = get_hip_runtime()
    library = build_gguf_q8_0_mmq_prefill(load=True)
    bufs = []

    def alloc(nbytes: int):
        buf = malloc(nbytes, runtime=runtime)
        bufs.append(buf)
        return buf

    try:
        weight_dev = alloc(qweight.nbytes)
        x_dev = alloc(x.nbytes)
        exact_dev = alloc(rows * out_features * 4)
        mmq_dev = alloc(rows * out_features * 4)
        d4_dev = alloc(q8_mmq_d4x3_nbytes(rows, hidden))
        count_dev = alloc(4)
        indices_dev = alloc(rows * out_features * 4)
        copy_host_to_device(
            weight_dev, host_array_ptr(qweight), runtime=runtime
        )
        copy_host_to_device(x_dev, host_array_ptr(x), runtime=runtime)

        def run_mmq() -> np.ndarray:
            gguf_q8_0_mmq128_quantize_f32_d4x3(
                x_dev.ptr, d4_dev.ptr, rows, hidden,
                library=library, runtime=runtime,
            )
            gguf_q8_0_mmq128_prefill_q8_1_d4x3_guarded_f32_f32_out(
                d4_dev.ptr, weight_dev.ptr, mmq_dev.ptr,
                count_dev.ptr, indices_dev.ptr,
                rows * out_features,
                QWEN4EXP_Q8_MMQ_PREFILL_POLICY.risk_threshold,
                rows, hidden, out_features,
                library=library, runtime=runtime,
            )
            gguf_q8_0_mmq128_sparse_exact_correct_f32(
                x_dev.ptr, weight_dev.ptr, mmq_dev.ptr,
                count_dev.ptr, indices_dev.ptr,
                rows * out_features, rows, hidden, out_features,
                library=library, runtime=runtime,
            )
            runtime.device_synchronize()
            out = np.empty((rows, out_features), dtype=np.float32)
            copy_device_to_host(host_array_ptr(out), mmq_dev, runtime=runtime)
            return out

        gguf_q8_0_gemv_coltile8_rowbatch4_f32_f32_out(
            x_dev.ptr, weight_dev.ptr, exact_dev.ptr,
            rows, hidden, out_features, runtime=runtime,
        )
        runtime.device_synchronize()
        exact = np.empty((rows, out_features), dtype=np.float32)
        copy_device_to_host(host_array_ptr(exact), exact_dev, runtime=runtime)
        first = run_mmq()
        second = run_mmq()
    finally:
        for buf in reversed(bufs):
            free(buf, runtime=runtime)

    np.testing.assert_array_equal(first, second)
    diff = np.abs(first - exact)
    scale = np.maximum(np.abs(exact), 1e-6)
    assert float(diff.max()) < 2e-3
    assert float((diff / scale).mean()) < 1e-4
    np.testing.assert_array_equal(first.argmax(1), exact.argmax(1))
