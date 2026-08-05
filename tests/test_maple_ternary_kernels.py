"""gfx11 correctness gates for Maple packed ternary/affine4 kernels."""

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
    affine4_gemv_f32,
    bf16_round,
    dequantize_affine4,
    f32_to_bf16_bits,
    ternary_gemv,
)
from hipengine.kernels.hip_gfx1100.quant.maple_ternary import (
    build_maple_ternary,
    maple_affine4_embed_bf16,
    maple_affine4_gemv_f32,
    maple_selected_ternary_dual_gemv_bf16,
    maple_selected_ternary_gemv_bf16,
    maple_ternary_gemv_bf16,
    maple_ternary_qkv_gemv_bf16,
    plan_maple_ternary_build,
)
from hipengine.kernels.registry import resolve


def pack2(values: np.ndarray) -> np.ndarray:
    codes = np.asarray(values, dtype=np.int32) + 1
    return np.sum(
        codes.astype(np.uint32).reshape(*codes.shape[:-1], -1, 16)
        << np.arange(0, 32, 2, dtype=np.uint32),
        axis=-1,
    ).astype(np.uint32)


def pack4(codes: np.ndarray) -> np.ndarray:
    values = np.asarray(codes, dtype=np.uint32)
    return np.sum(
        values.reshape(*values.shape[:-1], -1, 8)
        << np.arange(0, 32, 4, dtype=np.uint32),
        axis=-1,
    ).astype(np.uint32)


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


def test_maple_ternary_build_plan_and_gfx1151_registry_alias() -> None:
    plan = plan_maple_ternary_build(compiler_version="hipcc-test")
    assert plan.family == "maple_ternary"
    assert plan.output_path.name == "maple_ternary.so"
    assert plan.sources[0].name == "maple_ternary.hip"

    load_backend_kernel_package("hip_gfx1151")
    assert resolve(
        backend="hip_gfx1151",
        layer="maple_ternary_gemv",
        quant="maple_ternary2",
        variant="row_alpha",
    ) is maple_ternary_gemv_bf16


@pytest.fixture(scope="module")
def maple_ternary_lib(hip_test_target_arch):
    with hip_target_arch_environment(hip_test_target_arch):
        return build_maple_ternary(load=True)


def test_maple_affine4_embedding_and_head_match_cpu_oracle(maple_ternary_lib) -> None:
    rng = np.random.default_rng(11)
    rows, hidden = 5, 64
    codes = rng.integers(0, 16, size=(rows, hidden), dtype=np.uint8)
    packed = pack4(codes)
    scales_f32 = rng.uniform(0.01, 0.2, size=(rows, 1)).astype(np.float32)
    biases_f32 = rng.uniform(-0.5, 0.5, size=(rows, 1)).astype(np.float32)
    scales = f32_to_bf16_bits(scales_f32)
    biases = f32_to_bf16_bits(biases_f32)
    x_f32 = rng.normal(size=hidden).astype(np.float32)
    x = f32_to_bf16_bits(x_f32)
    token_id = 3

    expected_embedding = f32_to_bf16_bits(
        dequantize_affine4(packed, scales, biases)[token_id]
    )
    expected_logits = affine4_gemv_f32(
        bf16_round(x_f32), packed, scales, biases
    )

    with DeviceArrays() as dev:
        packed_d = dev.put(packed)
        scales_d = dev.put(scales)
        biases_d = dev.put(biases)
        x_d = dev.put(x)
        embedding, embedding_d = dev.empty((hidden,), np.dtype(np.uint16))
        logits, logits_d = dev.empty((rows,), np.dtype(np.float32))
        maple_affine4_embed_bf16(
            packed_d.ptr,
            scales_d.ptr,
            biases_d.ptr,
            embedding_d.ptr,
            token_id,
            hidden,
            library=maple_ternary_lib,
        )
        maple_affine4_gemv_f32(
            x_d.ptr,
            packed_d.ptr,
            scales_d.ptr,
            biases_d.ptr,
            logits_d.ptr,
            hidden,
            rows,
            library=maple_ternary_lib,
        )
        dev.get(embedding, embedding_d)
        dev.get(logits, logits_d)

    assert np.array_equal(embedding, expected_embedding)
    assert np.allclose(logits, expected_logits, atol=2e-4, rtol=2e-4)


def test_maple_generic_and_fused_qkv_ternary_gemv_match_oracle(maple_ternary_lib) -> None:
    rng = np.random.default_rng(22)
    hidden, q_rows, kv_rows = 32, 5, 3
    x_f32 = rng.normal(size=hidden).astype(np.float32)
    x = f32_to_bf16_bits(x_f32)
    matrices = [
        rng.integers(-1, 2, size=(rows, hidden), dtype=np.int8)
        for rows in (q_rows, kv_rows, kv_rows)
    ]
    packed = [pack2(matrix) for matrix in matrices]
    alpha = [
        f32_to_bf16_bits(rng.uniform(0.01, 0.5, size=matrix.shape[0]).astype(np.float32))
        for matrix in matrices
    ]
    expected_parts = [
        f32_to_bf16_bits(ternary_gemv(bf16_round(x_f32), weight, scale))
        for weight, scale in zip(packed, alpha)
    ]
    expected_qkv = np.concatenate(expected_parts)

    with DeviceArrays() as dev:
        x_d = dev.put(x)
        packed_d = [dev.put(value) for value in packed]
        alpha_d = [dev.put(value) for value in alpha]
        generic, generic_d = dev.empty((q_rows,), np.dtype(np.uint16))
        qkv, qkv_d = dev.empty(expected_qkv.shape, np.dtype(np.uint16))
        maple_ternary_gemv_bf16(
            x_d.ptr,
            packed_d[0].ptr,
            alpha_d[0].ptr,
            generic_d.ptr,
            hidden,
            q_rows,
            library=maple_ternary_lib,
        )
        maple_ternary_qkv_gemv_bf16(
            x_d.ptr,
            packed_d[0].ptr,
            alpha_d[0].ptr,
            packed_d[1].ptr,
            alpha_d[1].ptr,
            packed_d[2].ptr,
            alpha_d[2].ptr,
            qkv_d.ptr,
            hidden,
            q_rows,
            kv_rows,
            library=maple_ternary_lib,
        )
        dev.get(generic, generic_d)
        dev.get(qkv, qkv_d)

    assert np.array_equal(generic, expected_parts[0])
    assert np.array_equal(qkv, expected_qkv)


def test_maple_selected_dual_and_down_ternary_gemv_match_oracle(maple_ternary_lib) -> None:
    rng = np.random.default_rng(33)
    experts, top_k, hidden, intermediate, out_rows = 4, 2, 32, 16, 7
    selected = np.asarray([3, 1], dtype=np.int32)
    x_f32 = rng.normal(size=hidden).astype(np.float32)
    x = f32_to_bf16_bits(x_f32)
    gate_values = rng.integers(-1, 2, size=(experts, intermediate, hidden), dtype=np.int8)
    up_values = rng.integers(-1, 2, size=(experts, intermediate, hidden), dtype=np.int8)
    gate = pack2(gate_values)
    up = pack2(up_values)
    gate_alpha = f32_to_bf16_bits(
        rng.uniform(0.01, 0.5, size=(experts, intermediate)).astype(np.float32)
    )
    up_alpha = f32_to_bf16_bits(
        rng.uniform(0.01, 0.5, size=(experts, intermediate)).astype(np.float32)
    )
    gate_expected = np.stack(
        [
            f32_to_bf16_bits(ternary_gemv(bf16_round(x_f32), gate[e], gate_alpha[e]))
            for e in selected
        ]
    )
    up_expected = np.stack(
        [
            f32_to_bf16_bits(ternary_gemv(bf16_round(x_f32), up[e], up_alpha[e]))
            for e in selected
        ]
    )

    down_x_f32 = rng.normal(size=(top_k, intermediate)).astype(np.float32)
    down_x = f32_to_bf16_bits(down_x_f32)
    down_values = rng.integers(
        -1, 2, size=(experts, out_rows, intermediate), dtype=np.int8
    )
    down = pack2(down_values)
    down_alpha = f32_to_bf16_bits(
        rng.uniform(0.01, 0.5, size=(experts, out_rows)).astype(np.float32)
    )
    down_expected = np.stack(
        [
            f32_to_bf16_bits(
                ternary_gemv(bf16_round(down_x_f32[route]), down[e], down_alpha[e])
            )
            for route, e in enumerate(selected)
        ]
    )

    with DeviceArrays() as dev:
        x_d = dev.put(x)
        selected_d = dev.put(selected)
        gate_d, up_d = dev.put(gate), dev.put(up)
        gate_alpha_d, up_alpha_d = dev.put(gate_alpha), dev.put(up_alpha)
        gate_out, gate_out_d = dev.empty(gate_expected.shape, np.dtype(np.uint16))
        up_out, up_out_d = dev.empty(up_expected.shape, np.dtype(np.uint16))
        maple_selected_ternary_dual_gemv_bf16(
            x_d.ptr,
            gate_d.ptr,
            gate_alpha_d.ptr,
            up_d.ptr,
            up_alpha_d.ptr,
            selected_d.ptr,
            gate_out_d.ptr,
            up_out_d.ptr,
            experts,
            top_k,
            hidden,
            intermediate,
            library=maple_ternary_lib,
        )

        down_x_d = dev.put(down_x)
        down_d = dev.put(down)
        down_alpha_d = dev.put(down_alpha)
        down_out, down_out_d = dev.empty(down_expected.shape, np.dtype(np.uint16))
        maple_selected_ternary_gemv_bf16(
            down_x_d.ptr,
            down_d.ptr,
            down_alpha_d.ptr,
            selected_d.ptr,
            down_out_d.ptr,
            experts,
            top_k,
            intermediate,
            out_rows,
            library=maple_ternary_lib,
        )
        dev.get(gate_out, gate_out_d)
        dev.get(up_out, up_out_d)
        dev.get(down_out, down_out_d)

    assert np.array_equal(gate_out, gate_expected)
    assert np.array_equal(up_out, up_expected)
    assert np.array_equal(down_out, down_expected)
