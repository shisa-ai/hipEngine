"""gfx11 correctness gates for Maple router and MoE elementwise kernels."""

from __future__ import annotations

from typing import Self

import numpy as np
import pytest

from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.backends import (
    hip_target_arch_environment,
    load_backend_kernel_package,
)
from hipengine.kernels.cpu_reference.maple import (
    bf16_round,
    clamped_swiglu,
    f32_to_bf16_bits,
    router_topk,
    weighted_residual,
)
from hipengine.kernels.hip_gfx1100.moe.maple_moe import (
    build_maple_moe,
    maple_clamped_swiglu_bf16,
    maple_router_topk_bf16,
    maple_router_topk_parallel_bf16,
    maple_weighted_residual_bf16,
    plan_maple_moe_build,
)
from hipengine.kernels.registry import resolve


class DeviceArrays:
    def __init__(self) -> None:
        self.buffers = []

    def put(self, array: np.ndarray):
        host = np.ascontiguousarray(array)
        buffer = malloc(host.nbytes)
        self.buffers.append(buffer)
        copy_host_to_device(buffer, host_array_ptr(host))
        return buffer

    def empty(self, shape: tuple[int, ...], dtype: np.dtype):
        host = np.empty(shape, dtype=dtype)
        buffer = malloc(host.nbytes)
        self.buffers.append(buffer)
        return host, buffer

    def get(self, host: np.ndarray, buffer) -> np.ndarray:
        copy_device_to_host(host_array_ptr(host), buffer)
        return host

    def close(self) -> None:
        for buffer in reversed(self.buffers):
            free(buffer)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args) -> None:
        self.close()


def test_maple_moe_build_plan_and_gfx1151_registry_alias() -> None:
    plan = plan_maple_moe_build(compiler_version="hipcc-test")
    assert plan.family == "maple_moe"
    assert plan.output_path.name == "maple_moe.so"
    assert plan.sources[0].name == "maple_moe.hip"

    load_backend_kernel_package("hip_gfx1151")
    assert resolve(
        backend="hip_gfx1151",
        layer="maple_router_topk",
        quant="maple_ternary2",
        variant="bf16_fp32_softmax_renorm",
    ) is maple_router_topk_bf16


@pytest.fixture(scope="module")
def maple_moe_lib(hip_test_target_arch):
    with hip_target_arch_environment(hip_test_target_arch):
        return build_maple_moe(load=True)


def test_maple_router_topk_matches_fp32_softmax_renorm_oracle(maple_moe_lib) -> None:
    rng = np.random.default_rng(66)
    experts, hidden, top_k = 8, 32, 3
    x_f32 = rng.normal(size=hidden).astype(np.float32)
    weight_f32 = rng.normal(size=(experts, hidden)).astype(np.float32)
    x = f32_to_bf16_bits(x_f32)
    weight = f32_to_bf16_bits(weight_f32)
    expected_ids, expected_weights = router_topk(
        bf16_round(x_f32), bf16_round(weight_f32), top_k=top_k
    )

    with DeviceArrays() as dev:
        x_d, weight_d = dev.put(x), dev.put(weight)
        ids, ids_d = dev.empty((top_k,), np.dtype(np.int32))
        weights, weights_d = dev.empty((top_k,), np.dtype(np.float32))
        maple_router_topk_bf16(
            x_d.ptr,
            weight_d.ptr,
            ids_d.ptr,
            weights_d.ptr,
            hidden,
            experts,
            top_k,
            library=maple_moe_lib,
        )
        dev.get(ids, ids_d)
        dev.get(weights, weights_d)

    assert np.array_equal(ids, expected_ids.astype(np.int32))
    np.testing.assert_array_max_ulp(weights, expected_weights, maxulp=1)
    assert float(weights.sum()) == pytest.approx(1.0, abs=2e-7)


def test_maple_router_topk_parallel_matches_ids_and_renorm(
    maple_moe_lib, hip_test_target_arch
) -> None:
    """Parallel grid-over-experts router: IDs exact, weights close, sum ~= 1.

    The parallel variant computes each expert logit with a coalesced block tree
    reduce, so a few near-zero weights may differ by several ULP from the numpy
    BLAS reference (same as the model-level gate); the top-k IDs and the
    renormalized sum must still hold.
    """

    rng = np.random.default_rng(66)
    experts, hidden, top_k = 8, 32, 3
    x_f32 = rng.normal(size=hidden).astype(np.float32)
    weight_f32 = rng.normal(size=(experts, hidden)).astype(np.float32)
    x = f32_to_bf16_bits(x_f32)
    weight = f32_to_bf16_bits(weight_f32)
    expected_ids, expected_weights = router_topk(
        bf16_round(x_f32), bf16_round(weight_f32), top_k=top_k
    )

    with DeviceArrays() as dev:
        x_d, weight_d = dev.put(x), dev.put(weight)
        scratch, scratch_d = dev.empty((experts,), np.dtype(np.float32))
        ids, ids_d = dev.empty((top_k,), np.dtype(np.int32))
        weights, weights_d = dev.empty((top_k,), np.dtype(np.float32))
        maple_router_topk_parallel_bf16(
            x_d.ptr,
            weight_d.ptr,
            ids_d.ptr,
            weights_d.ptr,
            scratch_d.ptr,
            hidden,
            experts,
            top_k,
            library=maple_moe_lib,
        )
        dev.get(ids, ids_d)
        dev.get(weights, weights_d)

    assert np.array_equal(ids, expected_ids.astype(np.int32))
    np.testing.assert_allclose(weights, expected_weights, rtol=1e-2, atol=1e-4)
    assert float(weights.sum()) == pytest.approx(1.0, abs=2e-6)


def test_maple_clamped_swiglu_matches_trained_clamp_oracle(maple_moe_lib) -> None:
    gate_f32 = np.asarray(
        [[-10.0, -1.0, 0.0, 1.0, 10.0], [3.0, 7.0, 8.0, -8.0, 0.5]],
        dtype=np.float32,
    )
    up_f32 = np.asarray(
        [[-10.0, -2.0, 0.0, 2.0, 10.0], [8.0, -8.0, 6.0, -6.0, 0.25]],
        dtype=np.float32,
    )
    gate, up = f32_to_bf16_bits(gate_f32), f32_to_bf16_bits(up_f32)
    expected = f32_to_bf16_bits(
        clamped_swiglu(bf16_round(gate_f32), bf16_round(up_f32))
    )

    with DeviceArrays() as dev:
        gate_d, up_d = dev.put(gate), dev.put(up)
        out, out_d = dev.empty(gate.shape, np.dtype(np.uint16))
        maple_clamped_swiglu_bf16(
            gate_d.ptr,
            up_d.ptr,
            out_d.ptr,
            rows=gate.shape[0],
            features=gate.shape[1],
            library=maple_moe_lib,
        )
        dev.get(out, out_d)

    assert np.array_equal(out, expected)


def test_maple_weighted_residual_matches_two_bf16_boundaries(maple_moe_lib) -> None:
    rng = np.random.default_rng(77)
    top_k, hidden = 3, 17
    residual_f32 = rng.normal(size=hidden).astype(np.float32)
    experts_f32 = rng.normal(size=(top_k, hidden)).astype(np.float32)
    weights = np.asarray([0.125, 0.375, 0.5], dtype=np.float32)
    residual = f32_to_bf16_bits(residual_f32)
    experts = f32_to_bf16_bits(experts_f32)
    expected = f32_to_bf16_bits(
        weighted_residual(bf16_round(residual_f32), bf16_round(experts_f32), weights)
    )

    with DeviceArrays() as dev:
        residual_d, experts_d, weights_d = (
            dev.put(residual),
            dev.put(experts),
            dev.put(weights),
        )
        out, out_d = dev.empty((hidden,), np.dtype(np.uint16))
        maple_weighted_residual_bf16(
            residual_d.ptr,
            experts_d.ptr,
            weights_d.ptr,
            out_d.ptr,
            top_k,
            hidden,
            library=maple_moe_lib,
        )
        dev.get(out, out_d)

    assert np.array_equal(out, expected)
