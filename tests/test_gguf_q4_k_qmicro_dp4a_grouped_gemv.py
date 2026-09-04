"""Grouped q8_1 DP4A qmicro-Q4 sibling: contract, numerics, determinism.

C8-P2 Q4 pool candidate: a grouped-row dp4a kernel for the Q4_K qmicro T16
layout (rows 8-64), consuming the qmicro tiles DIRECTLY (no planar
conversion). Arithmetic class is integer dp4a decode (changed vs the
retained grouped rowtile16-w2 BF16 owner, which remains the registered
strict fallback until a production-profile L4 campaign). The numerical
contract mirrors the Q6 grouped dp4a test: plain-q8_1 CPU oracle agreement
(rtol 2e-2), outer KL/top-1 floor vs the exact reference, and run-to-run
determinism. The activation producer (gguf_q4_k_quantize_bf16_q8_1,
36-byte blocks) is the SAME one the retained owner consumes, so a
production decode pass needs no second quantize launch. The Q4_K minimum
correction is distributed per 4-k pack via dp4a(0x01010101,...) — integer
exact, matching the registered single-row Q4 dp4a GEMV convention.
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
from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_gemv import (
    build_gguf_q4_k_gemv,
    gguf_q4_k_quantize_bf16_q8_1,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_qmicro_dp4a_grouped import (
    build_gguf_q4_k_qmicro_dp4a_grouped,
    gguf_q4_k_qmicro_q8_1_dp4a_grouped_bf16_bf16_out,
    gguf_q4_k_qmicro_q8_1_dp4a_grouped_bf16_f32_out,
)
from hipengine.kernels.cpu_reference import gguf_q4_k_gemv
from hipengine.quant.gguf_q4_k import repack_gguf_q4_k_tile16_qmicro
from tests.test_gguf_q6_k_t16_planar_q8_1_gemv import HIP_AVAILABLE
from tests.test_gguf_x8_selected_gemv import (
    _bf16_bits,
    _bf16_to_f32,
    _exact_oracle,
    _q8_oracle,
    _softmax_kl,
    _top1,
    _weights,
)

Q8_1_BLOCK_BYTES = 36


@pytest.fixture(scope="module")
def q4_library():
    if not HIP_AVAILABLE:
        pytest.skip("HIP runtime is not available")
    return build_gguf_q4_k_gemv(load=True)


@pytest.fixture(scope="module")
def dp4a_library():
    if not HIP_AVAILABLE:
        pytest.skip("HIP runtime is not available")
    return build_gguf_q4_k_qmicro_dp4a_grouped(load=True)


def _qmicro_tiles(qweight: np.ndarray) -> np.ndarray:
    qmicro = repack_gguf_q4_k_tile16_qmicro(qweight)
    return np.ascontiguousarray(qmicro.tiles)


def _run_grouped(
    x_bits: np.ndarray,
    qweight: np.ndarray,
    dp4a_library,
    q4_library,
    f32_out: bool = False,
):
    from hipengine.core.hip import get_hip_runtime

    runtime = get_hip_runtime()
    rows, in_features = x_bits.shape
    out_features = int(qweight.shape[1])
    tiles = _qmicro_tiles(qweight)
    buffers = []

    try:
        def upload(value: np.ndarray):
            value = np.ascontiguousarray(value)
            buffer = malloc(value.nbytes, runtime=runtime)
            buffers.append(buffer)
            copy_host_to_device(buffer, host_array_ptr(value), runtime=runtime)
            return buffer

        x_buf = upload(x_bits)
        tiles_buf = upload(tiles)
        xq_buf = malloc(
            rows * (in_features // 32) * Q8_1_BLOCK_BYTES, runtime=runtime
        )
        out_buf = malloc(rows * out_features * 4, runtime=runtime)
        buffers.append(xq_buf)
        buffers.append(out_buf)

        gguf_q4_k_quantize_bf16_q8_1(
            x_buf.ptr,
            xq_buf.ptr,
            rows,
            in_features,
            library=q4_library,
            runtime=runtime,
        )
        wrapper = (
            gguf_q4_k_qmicro_q8_1_dp4a_grouped_bf16_f32_out
            if f32_out
            else gguf_q4_k_qmicro_q8_1_dp4a_grouped_bf16_bf16_out
        )
        wrapper(
            xq_buf.ptr,
            tiles_buf.ptr,
            out_buf.ptr,
            rows,
            in_features,
            out_features,
            library=dp4a_library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        dtype = np.float32 if f32_out else np.uint16
        first = np.empty((rows, out_features), dtype=dtype)
        copy_device_to_host(
            host_array_ptr(first), out_buf, first.nbytes, runtime=runtime
        )
        wrapper(
            xq_buf.ptr,
            tiles_buf.ptr,
            out_buf.ptr,
            rows,
            in_features,
            out_features,
            library=dp4a_library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        second = np.empty_like(first)
        copy_device_to_host(
            host_array_ptr(second), out_buf, second.nbytes, runtime=runtime
        )
        return first, second
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
class TestQ4QMicroDp4aRegistry:
    def test_symbol_registered(self) -> None:
        from hipengine.kernels.registry import resolve

        entry = resolve(
            backend="hip_gfx1100",
            layer="linear",
            quant="gguf_q4_k",
            variant="q8_1_dp4a_grouped_bf16_bf16_out",
        )
        assert entry is not None

    def test_wrapper_rejects_bad_rows(self) -> None:
        for rows in (0, 4, 7, 12, 72):
            with pytest.raises(ValueError):
                gguf_q4_k_qmicro_q8_1_dp4a_grouped_bf16_bf16_out(
                    1,
                    1,
                    1,
                    rows,
                    512,
                    16,
                    library=ctypes.CDLL("libc.so.6"),
                )


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
@pytest.mark.parametrize("rows", [8, 16, 24, 32])
def test_grouped_dp4a_matches_q8_oracle_with_outer_floor(
    rows: int, dp4a_library, q4_library
) -> None:
    """Plain-q8_1 CPU oracle + outer KL/top-1 floor vs the exact reference.

    Oracle convention matches the Q6 grouped dp4a test: the candidate's
    exact math (plain q8_1 activations, integer dp4a weight decode, f32
    scale application) is compared to the CPU plain-q8_1 oracle at rtol
    2e-2, and both are floored against the exact-weight CPU GEMV over the
    BF16-reconstructed activations.
    """
    rng = np.random.default_rng(0x6A340100 + rows)
    in_features, out_features = 512, 256
    qweight = _weights(
        "q4",
        out_features=out_features,
        in_features=in_features,
        experts=1,
    )
    x_bits = _bf16_bits(
        rng.normal(0.0, 0.1, size=(rows, in_features)).astype(np.float32) + 0.002
    )
    first, second = _run_grouped(
        x_bits, qweight, dp4a_library, q4_library, f32_out=True
    )
    np.testing.assert_array_equal(first, second)  # deterministic

    x_f32 = _bf16_to_f32(x_bits)
    x_rows = np.arange(rows, dtype=np.int64)
    selected = np.zeros(rows, dtype=np.int64)
    q8_reference = _q8_oracle("q4", x_f32, x_rows, selected, qweight)
    candidate_f32 = np.asarray(first, dtype=np.float32)
    np.testing.assert_allclose(candidate_f32, q8_reference, rtol=2.0e-2, atol=2.0e-2)

    exact_reference = _exact_oracle("q4", x_f32, x_rows, selected, qweight)
    _, kl_max = _softmax_kl(exact_reference, candidate_f32)
    assert kl_max <= 0.05
    assert _top1(exact_reference, candidate_f32) >= 0.90


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_grouped_dp4a_bf16_out_deterministic(dp4a_library, q4_library) -> None:
    rng = np.random.default_rng(0x6A340577)
    in_features, out_features = 512, 256
    qweight = _weights(
        "q4",
        out_features=out_features,
        in_features=in_features,
        experts=1,
    )
    x_bits = _bf16_bits(
        rng.normal(0.0, 0.1, size=(16, in_features)).astype(np.float32) + 0.002
    )
    first, second = _run_grouped(
        x_bits, qweight, dp4a_library, q4_library, f32_out=False
    )
    np.testing.assert_array_equal(first, second)
    assert int(np.isfinite(_bf16_to_f32(first)).all())


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_grouped_dp4a_matches_cpu_reference_owner_shapes(
    dp4a_library, q4_library
) -> None:
    """Actual decode shapes: q8-oracle agreement at production geometry.

    Uses the production single-projection geometry (in 5120, out 10240
    qkv-like and 6144 ffn_down-like) at the grouped rows. The candidate
    must track the plain-q8_1 CPU oracle (its own activation-quantization
    class) — at this depth the q8 class's own KL vs the exact reference
    dominates (the oracle and candidate sit equally far from exact), so
    the KL/top-1 floor vs exact is only asserted at the 512-deep test
    above, matching the Q6 grouped test's convention.
    """
    for index, (in_features, out_features) in enumerate(
        ((5120, 10240), (5120, 6144))
    ):
        rng = np.random.default_rng(0x6A340900 + index)
        qweight = _weights(
            "q4",
            out_features=out_features,
            in_features=in_features,
            experts=1,
        )
        x_bits = _bf16_bits(
            rng.normal(0.0, 0.05, size=(24, in_features)).astype(np.float32)
            + 0.001
        )
        first, second = _run_grouped(
            x_bits, qweight, dp4a_library, q4_library, f32_out=False
        )
        np.testing.assert_array_equal(first, second)
        x_f32 = _bf16_to_f32(x_bits)
        x_rows = np.arange(24, dtype=np.int64)
        selected = np.zeros(24, dtype=np.int64)
        # Columns are independent 144-byte Q4_K blocks, so the first 256
        # columns form a valid standalone weight for the (slow) CPU oracle.
        q8_reference = _q8_oracle(
            "q4", x_f32, x_rows, selected, qweight[:, :256]
        )
        candidate_f32 = _bf16_to_f32(first)[:, :256]
        np.testing.assert_allclose(
            candidate_f32, q8_reference, rtol=2.0e-2, atol=2.0e-2
        )
        assert int(np.isfinite(_bf16_to_f32(first)).all())
