"""PF-1b production-shape strict-parity gates for the Qwen4Exp dense owners.

Covers the three PF-1 kernel families from the halo-box campaign gap ledger
(docs/QWEN3.8-FLASH-NEXT-HALO-BOX-CAMPAIGN.md section 6.1, PF-1a mapping):

1. The production dense MMQ chain
   (``q8_0_mmq128_quantize_f32_residual_d4`` -> ``q8_0_raw_mmq128_q8_1_d4``
   guarded -> sparse exact repair) at every ``QWEN4EXP_Q8_MMQ_PREFILL_POLICY``
   production shape, deterministic and bounded against the exact F32 coltile
   owner that the execution profile registers as its strict fallback.
2. The strict F32 coltile chain (``gguf_k_prefill_out_coltile_rowbatch``)
   bit-identical across its registered tile variants at Qwen4Exp attention
   projection shapes.
3. The selected Q8_0 down path (``gguf_k_selected_prefill_out_kernel<quant8>``)
   against the CPU reference GEMV oracle at its ABI shape.

PF-1c/PF-1d variants must be added to ``MMQ_CHAIN_VARIANTS`` /
``COLTILE_VARIANTS`` / ``SELECTED_VARIANTS`` and keep these contracts green
before any whole-model A/B is admitted.
"""

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
    gguf_q8_0_gemv_coltile16_rowbatch4_f32_f32_out,
    gguf_q8_0_gemv_coltile4_rowbatch8_f32_f32_out,
    gguf_q8_0_gemv_coltile8_rowbatch4_f32_f32_out,
    gguf_q8_0_selected_gemv_bf16_bf16_out,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q8_0_mmq_prefill import (
    QWEN4EXP_Q8_MMQ_PREFILL_POLICY,
    build_gguf_q8_0_mmq_prefill,
    gguf_q8_0_mmq128_prefill_q8_1_d4x3_guarded_f32_f32_out,
    gguf_q8_0_mmq128_quantize_f32_d4x3,
    gguf_q8_0_mmq128_sparse_exact_correct_f32,
    q8_mmq_d4x3_nbytes,
)
from tests.test_gguf_k_gemv import Q8_0_BLOCK_BYTES, make_q8_0_weight


def make_q8_0_weight_large(out_features: int, in_features: int) -> np.ndarray:
    """Vectorized Q8_0 weight fixture for large ``out_features``.

    ``make_q8_0_weight`` overflows its int16 pattern past ~4.6k rows; this
    keeps the same 34-byte block layout (fp16 scale + 32 int8) with a bounded
    deterministic pattern.
    """

    if in_features % 32:
        raise ValueError("in_features must be a multiple of 32")
    blocks = in_features // 32
    out_idx = np.arange(out_features, dtype=np.int64)[:, None, None]
    block_idx = np.arange(blocks, dtype=np.int64)[None, :, None]
    lanes = np.arange(32, dtype=np.int64)[None, None, :]
    d = np.float16(0.03125 * (1 + (out_idx % 5)))
    q = ((lanes + out_idx * 7 + block_idx * 3) % 61 - 30).astype(np.int8)
    data = np.empty((out_features, blocks, Q8_0_BLOCK_BYTES), dtype=np.uint8)
    data[:, :, :2] = d.view(np.uint8)
    data[:, :, 2:] = q.view(np.uint8)
    return data.reshape(out_features, blocks * Q8_0_BLOCK_BYTES)


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def _bf16_bits_to_f32(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << 16).view(np.float32)


def _f32_to_bf16_bits(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32, copy=False)
    bits = values.view(np.uint32)
    lsb = (bits >> 16) & 1
    return ((bits + 0x7FFF + lsb) >> 16).astype(np.uint16)


# (hidden, out_features) pairs admitted by QWEN4EXP_Q8_MMQ_PREFILL_POLICY at
# prefill chunk rows; the gate ``rows >= 64`` and cap ``rows <= 2048`` are part
# of the contract under test.
PF1_MMQ_SHAPES = tuple(sorted(QWEN4EXP_Q8_MMQ_PREFILL_POLICY.min_rows))

# Qwen4Exp attention projection shapes that stay on the strict coltile chain
# (K % 256 == 0, not admitted to MMQ).
PF1_COLTILE_SHAPES = ((2560, 96), (2560, 512), (2560, 2048), (2560, 2560))

MMQ_CHAIN_VARIANTS = {
    "mmq128_prefill_q8_1_d4x3_guarded_f32_f32_out": (
        gguf_q8_0_mmq128_prefill_q8_1_d4x3_guarded_f32_f32_out
    ),
}

COLTILE_VARIANTS = {
    "coltile8_rowbatch4_f32_f32_out": gguf_q8_0_gemv_coltile8_rowbatch4_f32_f32_out,
    "coltile4_rowbatch8_f32_f32_out": gguf_q8_0_gemv_coltile4_rowbatch8_f32_f32_out,
    "coltile16_rowbatch4_f32_f32_out": gguf_q8_0_gemv_coltile16_rowbatch4_f32_f32_out,
}

SELECTED_VARIANTS = {
    "selected_gemv_bf16_bf16_out": gguf_q8_0_selected_gemv_bf16_bf16_out,
}


def _policy_gate(hidden: int, out_features: int, rows: int) -> bool:
    return QWEN4EXP_Q8_MMQ_PREFILL_POLICY(rows, hidden, out_features)


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("hidden,out_features", PF1_MMQ_SHAPES)
def test_pf1_mmq_policy_gates_production_shapes(hidden: int, out_features: int) -> None:
    """The policy admits prefill chunk rows and rejects sub-threshold rows."""

    assert _policy_gate(hidden, out_features, 512)
    assert _policy_gate(hidden, out_features, 64)
    assert not _policy_gate(hidden, out_features, 32)
    assert not _policy_gate(hidden, out_features, 4096)


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("hidden,out_features", PF1_MMQ_SHAPES)
@pytest.mark.parametrize("variant_name", sorted(MMQ_CHAIN_VARIANTS))
def test_pf1_mmq_chain_production_shapes(
    hidden: int, out_features: int, variant_name: str
) -> None:
    """Every MMQ chain variant is deterministic and bounded vs the exact owner.

    The exact owner is the registered strict fallback
    ``coltile8_rowbatch4_f32_f32_out``; the guarded chain carries the
    production-profile arithmetic, so the binding contract is determinism plus
    the bounded envelope and top-1 agreement, matching
    ``test_qwen4exp_f32_mmq_chain_bounded_against_exact_f32_owner`` at
    production shapes.
    """

    from hipengine.core.hip import get_hip_runtime

    rows = 512
    rng = np.random.default_rng(2026_09_02)
    qweight = np.ascontiguousarray(
        make_q8_0_weight_large(out_features, hidden), dtype=np.uint8
    )
    x = (rng.standard_normal((rows, hidden)) * 0.5).astype(np.float32)
    chain = MMQ_CHAIN_VARIANTS[variant_name]

    runtime = get_hip_runtime()
    mmq_library = build_gguf_q8_0_mmq_prefill(load=True)
    bufs: list = []

    def alloc(nbytes: int):
        buf = malloc(nbytes, runtime=runtime)
        bufs.append(buf)
        return buf

    try:
        weight_dev = alloc(qweight.nbytes)
        x_dev = alloc(x.nbytes)
        exact_dev = alloc(rows * out_features * 4)
        chain_dev = alloc(rows * out_features * 4)
        d4_dev = alloc(q8_mmq_d4x3_nbytes(rows, hidden))
        count_dev = alloc(4)
        indices_dev = alloc(rows * out_features * 4)
        copy_host_to_device(weight_dev, host_array_ptr(qweight), runtime=runtime)
        copy_host_to_device(x_dev, host_array_ptr(x), runtime=runtime)

        gguf_q8_0_gemv_coltile8_rowbatch4_f32_f32_out(
            x_dev.ptr,
            weight_dev.ptr,
            exact_dev.ptr,
            rows,
            hidden,
            out_features,
            library=build_gguf_k_gemv(load=True),
            runtime=runtime,
        )

        def run_chain() -> np.ndarray:
            gguf_q8_0_mmq128_quantize_f32_d4x3(
                x_dev.ptr,
                d4_dev.ptr,
                rows,
                hidden,
                library=mmq_library,
                runtime=runtime,
            )
            runtime.memset(count_dev.ptr, 0, count_dev.nbytes)
            chain(
                d4_dev.ptr,
                weight_dev.ptr,
                chain_dev.ptr,
                count_dev.ptr,
                indices_dev.ptr,
                rows * out_features,
                QWEN4EXP_Q8_MMQ_PREFILL_POLICY.risk_threshold,
                rows,
                hidden,
                out_features,
                library=mmq_library,
                runtime=runtime,
            )
            gguf_q8_0_mmq128_sparse_exact_correct_f32(
                x_dev.ptr,
                weight_dev.ptr,
                chain_dev.ptr,
                count_dev.ptr,
                indices_dev.ptr,
                rows * out_features,
                rows,
                hidden,
                out_features,
                library=mmq_library,
                runtime=runtime,
            )
            runtime.device_synchronize()
            out = np.empty((rows, out_features), dtype=np.float32)
            copy_device_to_host(host_array_ptr(out), chain_dev, runtime=runtime)
            return out

        exact = np.empty((rows, out_features), dtype=np.float32)
        copy_device_to_host(host_array_ptr(exact), exact_dev, runtime=runtime)
        first = run_chain()
        second = run_chain()
    finally:
        for buf in reversed(bufs):
            free(buf, runtime=runtime)

    np.testing.assert_array_equal(first, second)
    diff = np.abs(first - exact)
    scale = np.maximum(np.abs(exact), 1e-6)
    assert float(diff.max()) < 2e-3, (hidden, out_features, float(diff.max()))
    assert float((diff / scale).mean()) < 1e-4
    np.testing.assert_array_equal(first.argmax(1), exact.argmax(1))


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("hidden,out_features", PF1_COLTILE_SHAPES)
def test_pf1_coltile_strict_chain_bit_parity_at_production_shapes(
    hidden: int, out_features: int
) -> None:
    """All registered coltile variants publish identical F32 bits."""

    from hipengine.core.hip import get_hip_runtime

    rows = 512
    rng = np.random.default_rng(2026_09_03)
    qweight = np.ascontiguousarray(
        make_q8_0_weight_large(out_features, hidden), dtype=np.uint8
    )
    x = (rng.standard_normal((rows, hidden)) * 0.5).astype(np.float32)

    runtime = get_hip_runtime()
    library = build_gguf_k_gemv(load=True)
    bufs: list = []

    def alloc(nbytes: int):
        buf = malloc(nbytes, runtime=runtime)
        bufs.append(buf)
        return buf

    try:
        weight_dev = alloc(qweight.nbytes)
        x_dev = alloc(x.nbytes)
        copy_host_to_device(weight_dev, host_array_ptr(qweight), runtime=runtime)
        copy_host_to_device(x_dev, host_array_ptr(x), runtime=runtime)

        outputs: dict[str, np.ndarray] = {}
        for name, variant in COLTILE_VARIANTS.items():
            out_dev = alloc(rows * out_features * 4)
            variant(
                x_dev.ptr,
                weight_dev.ptr,
                out_dev.ptr,
                rows,
                hidden,
                out_features,
                library=library,
                runtime=runtime,
            )
            runtime.device_synchronize()
            out = np.empty((rows, out_features), dtype=np.float32)
            copy_device_to_host(host_array_ptr(out), out_dev, runtime=runtime)
            outputs[name] = out
    finally:
        for buf in reversed(bufs):
            free(buf, runtime=runtime)

    reference_name = "coltile8_rowbatch4_f32_f32_out"
    reference = outputs[reference_name]
    for name, out in outputs.items():
        np.testing.assert_array_equal(
            out, reference, err_msg=f"{name} vs {reference_name}"
        )


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("variant_name", sorted(SELECTED_VARIANTS))
def test_pf1_selected_q8_0_down_matches_cpu_reference(variant_name: str) -> None:
    """The selected Q8_0 down path matches the per-row CPU GEMV oracle.

    Shape mirrors the production ABI: each of ``x_rows`` source rows expands to
    ``k`` selected-expert rows (``rows % x_rows == 0``), matching the observed
    top-10 gather (512 tokens -> 5120 selected rows).
    """

    from hipengine.core.hip import get_hip_runtime

    x_rows, k_experts, num_experts, in_features, out_features = 8, 10, 16, 512, 1024
    rows = x_rows * k_experts
    rng = np.random.default_rng(2026_09_04)
    qweight = np.ascontiguousarray(
        make_q8_0_weight_large(num_experts * out_features, in_features),
        dtype=np.uint8,
    )
    weight_rows = qweight.reshape(num_experts, out_features, -1)
    x_float = (rng.standard_normal((x_rows, in_features)) * 0.5).astype(np.float32)
    x_bf16 = _f32_to_bf16_bits(x_float)
    x_source = _bf16_bits_to_f32(x_bf16)
    selected = rng.integers(0, num_experts, size=rows).astype(np.int64)

    runtime = get_hip_runtime()
    library = build_gguf_k_gemv(load=True)
    variant = SELECTED_VARIANTS[variant_name]
    bufs: list = []

    def alloc(nbytes: int):
        buf = malloc(nbytes, runtime=runtime)
        bufs.append(buf)
        return buf

    try:
        x_dev = alloc(x_bf16.nbytes)
        selected_dev = alloc(selected.nbytes)
        weight_dev = alloc(qweight.nbytes)
        out_dev = alloc(rows * out_features * 2)
        copy_host_to_device(x_dev, host_array_ptr(x_bf16), runtime=runtime)
        copy_host_to_device(
            selected_dev, host_array_ptr(selected), runtime=runtime
        )
        copy_host_to_device(weight_dev, host_array_ptr(qweight), runtime=runtime)

        def run() -> np.ndarray:
            variant(
                x_dev.ptr,
                selected_dev.ptr,
                weight_dev.ptr,
                out_dev.ptr,
                x_rows,
                rows,
                num_experts,
                in_features,
                out_features,
                library=library,
                runtime=runtime,
            )
            runtime.device_synchronize()
            out = np.empty((rows, out_features), dtype=np.uint16)
            copy_device_to_host(host_array_ptr(out), out_dev, runtime=runtime)
            return out

        first = run()
        second = run()
    finally:
        for buf in reversed(bufs):
            free(buf, runtime=runtime)

    np.testing.assert_array_equal(first, second)
    got = _bf16_bits_to_f32(first)
    for row_idx in range(rows):
        expert = int(selected[row_idx])
        source = x_source[row_idx // k_experts]
        expected = np.asarray(
            gguf_q8_0_gemv(source[None, :], np.ascontiguousarray(weight_rows[expert]))
        ).astype(np.float32)[0]
        actual = got[row_idx]
        scale = np.maximum(np.abs(expected), 1e-6)
        assert float(np.abs(actual - expected).max()) < 2e-2 * float(
            np.abs(expected).max()
        ), row_idx
        assert float((np.abs(actual - expected) / scale).mean()) < 2e-2, row_idx
