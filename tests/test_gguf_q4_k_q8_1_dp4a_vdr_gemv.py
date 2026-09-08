"""Dense single-row q8_1 DP4A Q4_K load-reuse A/B: control vs VDR kernels.

Leaf screen for the nasone32 "k-quant-boosts" adjacent-chunk load
amortization idea (llama.cpp-RDNA3 efa4e86410c07723deaa458bdadd8c08f1029928,
ggml/src/ggml-cuda/vecdotq.cuh VDR_Q4_K_Q8_1_MMVQ 2 -> 4) ported into the
hipEngine raw-GGUF q8_1-dp4a decode idiom. Two kernels share the SAME
subblock-strided thread mapping and compute the SAME per-pack f32
expression in the SAME order: the control kernel re-decodes the block
header (d/dmin), the 6-bit scale/min pair, and the q8_1 block scale for
every 4-k pack; the VDR kernel amortizes those loads once per 32-element
subblock and reuses them for all 8 packs and 8 output columns.

The RED contract is therefore BIT-EXACT equality between the two kernels:
identical integer dp4a terms and identical f32 evaluation order mean the
only admissible difference is load scheduling, not values. Both arms are
additionally checked against the plain-q8_1 CPU oracle (rtol 2e-2), the
outer KL <= 0.05 / top-1 >= 0.90 floor vs the exact-weight reference, and
run-to-run determinism.
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
from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_q8_1_dp4a_vdr_gemv import (
    build_gguf_q4_k_q8_1_dp4a_vdr_gemv,
    gguf_q4_k_q8_1_dp4a_ctl_bf16_f32_out,
    gguf_q4_k_q8_1_dp4a_vdr_bf16_f32_out,
)
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
def vdr_library():
    if not HIP_AVAILABLE:
        pytest.skip("HIP runtime is not available")
    return build_gguf_q4_k_q8_1_dp4a_vdr_gemv(load=True)


def _run_pair(
    x_bits: np.ndarray,
    qweight0: np.ndarray,
    vdr_library,
    q4_library,
):
    """Launch control and VDR kernels on identical inputs; return (ctl, vdr)."""
    from hipengine.core.hip import get_hip_runtime

    runtime = get_hip_runtime()
    rows, in_features = x_bits.shape
    out_features = int(qweight0.shape[0])

    buffers = []

    try:
        def upload(value: np.ndarray):
            value = np.ascontiguousarray(value)
            buffer = malloc(value.nbytes, runtime=runtime)
            buffers.append(buffer)
            copy_host_to_device(buffer, host_array_ptr(value), runtime=runtime)
            return buffer

        x_buf = upload(x_bits)
        w_buf = upload(qweight0)
        xq_buf = malloc(rows * (in_features // 32) * Q8_1_BLOCK_BYTES, runtime=runtime)
        ctl_buf = malloc(rows * out_features * 4, runtime=runtime)
        vdr_buf = malloc(rows * out_features * 4, runtime=runtime)
        buffers.extend((xq_buf, ctl_buf, vdr_buf))

        gguf_q4_k_quantize_bf16_q8_1(
            x_buf.ptr,
            xq_buf.ptr,
            rows,
            in_features,
            library=q4_library,
            runtime=runtime,
        )
        common = dict(
            rows=rows,
            in_features=in_features,
            out_features=out_features,
            library=vdr_library,
            runtime=runtime,
        )
        gguf_q4_k_q8_1_dp4a_ctl_bf16_f32_out(
            xq_buf.ptr, w_buf.ptr, ctl_buf.ptr, **common
        )
        gguf_q4_k_q8_1_dp4a_vdr_bf16_f32_out(
            xq_buf.ptr, w_buf.ptr, vdr_buf.ptr, **common
        )
        runtime.device_synchronize()

        def download(buf):
            host = np.empty((rows, out_features), dtype=np.float32)
            copy_device_to_host(host_array_ptr(host), buf, host.nbytes, runtime=runtime)
            return host

        return download(ctl_buf), download(vdr_buf)
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
@pytest.mark.parametrize(
    "in_features,out_features,rows",
    [
        (512, 256, 1),
        (512, 256, 2),
        (1024, 512, 4),
        (5120, 10240, 1),
    ],
)
def test_vdr_bit_identical_to_control(
    in_features: int, out_features: int, rows: int, vdr_library, q4_library
) -> None:
    """RED contract: amortizing loads must not change any output bit."""
    rng = np.random.default_rng(0x6A341000 + in_features + rows)
    qweight = _weights(
        "q4", out_features=out_features, in_features=in_features, experts=1
    )
    x_bits = _bf16_bits(
        rng.normal(0.0, 0.1, size=(rows, in_features)).astype(np.float32) + 0.002
    )
    ctl, vdr = _run_pair(x_bits, qweight[0], vdr_library, q4_library)
    np.testing.assert_array_equal(ctl, vdr)
    assert bool(np.isfinite(ctl).all())


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_dp4a_arms_match_q8_oracle_with_outer_floor(vdr_library, q4_library) -> None:
    """Both arms track the plain-q8_1 CPU oracle and pass the outer floor."""
    rng = np.random.default_rng(0x6A341577)
    in_features, out_features, rows = 512, 256, 1
    qweight = _weights(
        "q4", out_features=out_features, in_features=in_features, experts=1
    )
    x_bits = _bf16_bits(
        rng.normal(0.0, 0.1, size=(rows, in_features)).astype(np.float32) + 0.002
    )
    ctl, vdr = _run_pair(x_bits, qweight[0], vdr_library, q4_library)

    x_f32 = _bf16_to_f32(x_bits)
    x_rows = np.arange(rows, dtype=np.int64)
    selected = np.zeros(rows, dtype=np.int64)
    q8_reference = _q8_oracle("q4", x_f32, x_rows, selected, qweight)
    for name, candidate in (("control", ctl), ("vdr", vdr)):
        np.testing.assert_allclose(
            candidate, q8_reference, rtol=2.0e-2, atol=2.0e-2
        )
    exact_reference = _exact_oracle("q4", x_f32, x_rows, selected, qweight)
    for name, candidate in (("control", ctl), ("vdr", vdr)):
        _, kl_max = _softmax_kl(exact_reference, candidate)
        assert kl_max <= 0.05, name
        assert _top1(exact_reference, candidate) >= 0.90, name


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_vdr_deterministic_across_runs(vdr_library, q4_library) -> None:
    rng = np.random.default_rng(0x6A3419A1)
    in_features, out_features, rows = 1024, 512, 2
    qweight = _weights(
        "q4", out_features=out_features, in_features=in_features, experts=1
    )
    x_bits = _bf16_bits(
        rng.normal(0.0, 0.1, size=(rows, in_features)).astype(np.float32)
    )
    first_ctl, first_vdr = _run_pair(x_bits, qweight[0], vdr_library, q4_library)
    second_ctl, second_vdr = _run_pair(x_bits, qweight[0], vdr_library, q4_library)
    np.testing.assert_array_equal(first_ctl, second_ctl)
    np.testing.assert_array_equal(first_vdr, second_vdr)


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
class TestQ4Q81Dp4aVdrRegistry:
    def test_vdr_symbol_registered(self) -> None:
        from hipengine.kernels.registry import resolve

        entry = resolve(
            backend="hip_gfx1100",
            layer="linear",
            quant="gguf_q4_k",
            variant="q8_1_dp4a_vdr_bf16_f32_out",
        )
        assert entry is not None

    def test_ctl_symbol_registered(self) -> None:
        from hipengine.kernels.registry import resolve

        entry = resolve(
            backend="hip_gfx1100",
            layer="linear",
            quant="gguf_q4_k",
            variant="q8_1_dp4a_ctl_bf16_f32_out",
        )
        assert entry is not None

    def test_wrapper_rejects_bad_shapes(self) -> None:
        for rows, in_features, out_features in (
            (0, 512, 256),
            (1, 511, 256),
            (1, 512, 255),
            (1, 0, 256),
        ):
            with pytest.raises(ValueError):
                gguf_q4_k_q8_1_dp4a_vdr_bf16_f32_out(
                    1,
                    1,
                    1,
                    rows,
                    in_features,
                    out_features,
                    library=ctypes.CDLL("libc.so.6"),
                )
