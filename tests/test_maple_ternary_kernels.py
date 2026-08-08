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
from hipengine.kernels.hip_gfx1100.moe.maple_moe import (
    build_maple_moe,
    maple_clamped_swiglu_bf16,
)
from hipengine.kernels.hip_gfx1100.quant.maple_ternary import (
    build_maple_ternary,
    maple_affine4_embed_bf16,
    maple_affine4_gemv_f32,
    maple_affine4_gemv_wave32_exact_f32,
    maple_moe_dual_swiglu_bf16,
    maple_selected_ternary_dual_gemv_bf16,
    maple_selected_ternary_dual_grouped_bf16,
    maple_selected_ternary_gemv_bf16,
    maple_selected_ternary_grouped_bf16,
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
    assert resolve(
        backend="hip_gfx1151",
        layer="maple_selected_ternary_dual",
        quant="maple_ternary2",
        variant="row_alpha_grouped",
    ) is maple_selected_ternary_dual_grouped_bf16
    assert resolve(
        backend="hip_gfx1151",
        layer="maple_selected_ternary",
        quant="maple_ternary2",
        variant="row_alpha_grouped",
    ) is maple_selected_ternary_grouped_bf16
    assert resolve(
        backend="hip_gfx1151",
        layer="maple_affine4_gemv",
        quant="maple_ternary2",
        variant="group64_wave32_exact",
    ) is maple_affine4_gemv_wave32_exact_f32


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


def test_maple_affine4_wave32_emulates_production_reduction_exactly(
    maple_ternary_lib,
) -> None:
    """D0 candidate must preserve every 128-thread FP32 logit bit at K=2048."""

    rng = np.random.default_rng(20260808)
    hidden, out = 2048, 257
    x = f32_to_bf16_bits(rng.normal(size=hidden).astype(np.float32))
    packed = pack4(rng.integers(0, 16, size=(out, hidden), dtype=np.uint8))
    scales = f32_to_bf16_bits(
        rng.uniform(0.001, 0.2, size=(out, hidden // 64)).astype(np.float32)
    )
    biases = f32_to_bf16_bits(
        rng.uniform(-0.5, 0.5, size=(out, hidden // 64)).astype(np.float32)
    )

    with DeviceArrays() as dev:
        x_d = dev.put(x)
        packed_d = dev.put(packed)
        scales_d, biases_d = dev.put(scales), dev.put(biases)
        baseline, baseline_d = dev.empty((out,), np.dtype(np.float32))
        candidate, candidate_d = dev.empty((out,), np.dtype(np.float32))
        maple_affine4_gemv_f32(
            x_d.ptr,
            packed_d.ptr,
            scales_d.ptr,
            biases_d.ptr,
            baseline_d.ptr,
            hidden,
            out,
            library=maple_ternary_lib,
        )
        maple_affine4_gemv_wave32_exact_f32(
            x_d.ptr,
            packed_d.ptr,
            scales_d.ptr,
            biases_d.ptr,
            candidate_d.ptr,
            hidden,
            out,
            library=maple_ternary_lib,
        )
        dev.get(baseline, baseline_d)
        dev.get(candidate, candidate_d)

    assert np.array_equal(candidate.view(np.uint32), baseline.view(np.uint32))


def test_maple_affine4_embed_batched_matches_oracle(maple_ternary_lib) -> None:
    """Batched affine4 embed of T token IDs matches dequantize_affine4 (P4)."""

    from hipengine.kernels.hip_gfx1100.quant.maple_ternary import (
        maple_affine4_embed_batched_bf16,
    )

    rng = np.random.default_rng(55)
    rows, hidden = 6, 64
    codes = rng.integers(0, 16, size=(rows, hidden), dtype=np.uint8)
    packed = pack4(codes)
    scales = f32_to_bf16_bits(
        rng.uniform(0.01, 0.2, size=(rows, 1)).astype(np.float32)
    )
    biases = f32_to_bf16_bits(
        rng.uniform(-0.5, 0.5, size=(rows, 1)).astype(np.float32)
    )
    token_ids = np.asarray([0, 4, 1, 5, 3, 2], dtype=np.int64)
    expected = f32_to_bf16_bits(
        np.stack([dequantize_affine4(packed, scales, biases)[t] for t in token_ids])
    )

    with DeviceArrays() as dev:
        packed_d = dev.put(packed)
        scales_d = dev.put(scales)
        biases_d = dev.put(biases)
        ids_d = dev.put(token_ids)
        out, out_d = dev.empty((rows, hidden), np.dtype(np.uint16))
        maple_affine4_embed_batched_bf16(
            packed_d.ptr,
            scales_d.ptr,
            biases_d.ptr,
            ids_d.ptr,
            out_d.ptr,
            rows,
            hidden,
            library=maple_ternary_lib,
        )
        dev.get(out, out_d)

    assert np.array_equal(out, expected)


@pytest.mark.parametrize("rows", [2, 4, 8])
def test_maple_affine4_gemv_batched_rowreuse_matches_production_bits(
    maple_ternary_lib,
    rows,
) -> None:
    """D1 row reuse must preserve the K=2048 group64 reduction bit-for-bit."""

    from hipengine.kernels.hip_gfx1100.quant.maple_ternary import (
        maple_affine4_gemv_batched_f32,
        maple_affine4_gemv_batched_rowreuse_exact_f32,
    )

    rng = np.random.default_rng(109 + rows)
    hidden, out = 2_048, 37
    x = f32_to_bf16_bits(rng.normal(size=(rows, hidden)).astype(np.float32))
    codes = rng.integers(0, 16, size=(out, hidden), dtype=np.uint8)
    packed = pack4(codes)
    scales = f32_to_bf16_bits(
        rng.uniform(0.01, 0.2, size=(out, hidden // 64)).astype(np.float32)
    )
    biases = f32_to_bf16_bits(
        rng.uniform(-0.5, 0.5, size=(out, hidden // 64)).astype(np.float32)
    )

    with DeviceArrays() as dev:
        x_d = dev.put(x)
        packed_d = dev.put(packed)
        scales_d, biases_d = dev.put(scales), dev.put(biases)
        baseline, baseline_d = dev.empty((rows, out), np.dtype(np.float32))
        candidate, candidate_d = dev.empty((rows, out), np.dtype(np.float32))
        maple_affine4_gemv_batched_f32(
            x_d.ptr,
            packed_d.ptr,
            scales_d.ptr,
            biases_d.ptr,
            baseline_d.ptr,
            rows,
            hidden,
            out,
            library=maple_ternary_lib,
        )
        maple_affine4_gemv_batched_rowreuse_exact_f32(
            x_d.ptr,
            packed_d.ptr,
            scales_d.ptr,
            biases_d.ptr,
            candidate_d.ptr,
            rows,
            hidden,
            out,
            library=maple_ternary_lib,
        )
        dev.get(baseline, baseline_d)
        dev.get(candidate, candidate_d)

    assert np.array_equal(candidate.view(np.uint32), baseline.view(np.uint32))


def test_maple_affine4_gemv_batched_matches_oracle(maple_ternary_lib) -> None:
    """Batched affine4 lm_head GEMM over T rows matches per-row oracle (P4)."""

    from hipengine.kernels.hip_gfx1100.quant.maple_ternary import (
        maple_affine4_gemv_batched_f32,
    )

    rng = np.random.default_rng(99)
    rows, hidden, out = 5, 64, 11
    x_f32 = rng.normal(size=(rows, hidden)).astype(np.float32)
    x = f32_to_bf16_bits(x_f32)
    codes = rng.integers(0, 16, size=(out, hidden), dtype=np.uint8)
    packed = pack4(codes)
    scales = f32_to_bf16_bits(
        rng.uniform(0.01, 0.2, size=(out, 1)).astype(np.float32)
    )
    biases = f32_to_bf16_bits(
        rng.uniform(-0.5, 0.5, size=(out, 1)).astype(np.float32)
    )
    expected = np.stack(
        [affine4_gemv_f32(bf16_round(row), packed, scales, biases) for row in x_f32]
    )

    with DeviceArrays() as dev:
        x_d = dev.put(x)
        packed_d = dev.put(packed)
        scales_d, biases_d = dev.put(scales), dev.put(biases)
        out_arr, out_d = dev.empty((rows, out), np.dtype(np.float32))
        maple_affine4_gemv_batched_f32(
            x_d.ptr,
            packed_d.ptr,
            scales_d.ptr,
            biases_d.ptr,
            out_d.ptr,
            rows,
            hidden,
            out,
            library=maple_ternary_lib,
        )
        dev.get(out_arr, out_d)

    np.testing.assert_allclose(out_arr, expected, atol=2e-4, rtol=2e-4)




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


def test_maple_ternary_gemm_batched_matches_cpu_oracle(maple_ternary_lib) -> None:
    """Batched [rows, out] ternary GEMM is bit-exact vs the CPU oracle (P1)."""

    from hipengine.kernels.hip_gfx1100.quant.maple_ternary import (
        maple_ternary_gemm_bf16,
    )

    rng = np.random.default_rng(44)
    rows, hidden, out_features = 13, 64, 11  # rows spans > one 8-token tile
    x_f32 = rng.normal(size=(rows, hidden)).astype(np.float32)
    x = f32_to_bf16_bits(x_f32)
    values = rng.integers(-1, 2, size=(out_features, hidden), dtype=np.int8)
    packed = pack2(values)
    alpha = f32_to_bf16_bits(
        rng.uniform(0.01, 0.5, size=(out_features,)).astype(np.float32)
    )
    expected = f32_to_bf16_bits(
        np.stack(
            [ternary_gemv(bf16_round(row), packed, alpha) for row in x_f32]
        )
    )

    with DeviceArrays() as dev:
        x_d = dev.put(x)
        packed_d = dev.put(packed)
        alpha_d = dev.put(alpha)
        out, out_d = dev.empty((rows, out_features), np.dtype(np.uint16))
        maple_ternary_gemm_bf16(
            x_d.ptr,
            packed_d.ptr,
            alpha_d.ptr,
            out_d.ptr,
            rows,
            hidden,
            out_features,
            library=maple_ternary_lib,
        )
        dev.get(out, out_d)

    assert np.array_equal(out, expected)


def test_maple_ternary_qkv_gemm_batched_matches_oracle(maple_ternary_lib) -> None:
    """Batched QKV ternary GEMM is bit-exact vs 3x CPU ternary_gemv per row (P1)."""

    from hipengine.kernels.hip_gfx1100.quant.maple_ternary import (
        maple_ternary_qkv_gemm_bf16,
    )

    rng = np.random.default_rng(77)
    rows, hidden, q_rows, kv_rows = 13, 64, 5, 3
    x_f32 = rng.normal(size=(rows, hidden)).astype(np.float32)
    x = f32_to_bf16_bits(x_f32)
    matrices = [
        rng.integers(-1, 2, size=(n, hidden), dtype=np.int8)
        for n in (q_rows, kv_rows, kv_rows)
    ]
    packed = [pack2(m) for m in matrices]
    alpha = [
        f32_to_bf16_bits(rng.uniform(0.01, 0.5, size=(m.shape[0],)).astype(np.float32))
        for m in matrices
    ]
    parts = [
        f32_to_bf16_bits(ternary_gemv(bf16_round(row), packed[i], alpha[i]))
        for row in x_f32
        for i in range(3)
    ]
    expected = np.concatenate(parts).reshape(rows, q_rows + 2 * kv_rows)
    total = q_rows + 2 * kv_rows

    with DeviceArrays() as dev:
        x_d = dev.put(x)
        packed_d = [dev.put(p) for p in packed]
        alpha_d = [dev.put(a) for a in alpha]
        qkv, qkv_d = dev.empty((rows, total), np.dtype(np.uint16))
        maple_ternary_qkv_gemm_bf16(
            x_d.ptr,
            packed_d[0].ptr,
            alpha_d[0].ptr,
            packed_d[1].ptr,
            alpha_d[1].ptr,
            packed_d[2].ptr,
            alpha_d[2].ptr,
            qkv_d.ptr,
            rows,
            hidden,
            q_rows,
            kv_rows,
            library=maple_ternary_lib,
        )
        dev.get(qkv, qkv_d)

    assert np.array_equal(qkv, expected)


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


def test_maple_selected_ternary_dual_gemv_batched_matches_oracle(maple_ternary_lib) -> None:
    """Batched selected-expert dual GEMV over (T, top_k) entries is bit-exact (P3)."""

    from hipengine.kernels.hip_gfx1100.quant.maple_ternary import (
        maple_selected_ternary_dual_gemv_batched_bf16,
    )

    rng = np.random.default_rng(44)
    rows, experts, top_k, hidden, intermediate = 5, 4, 2, 32, 16
    x_f32 = rng.normal(size=(rows, hidden)).astype(np.float32)
    x = f32_to_bf16_bits(x_f32)
    selected = rng.integers(0, experts, size=(rows, top_k)).astype(np.int32)
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
    gate_expected = np.zeros((rows, top_k, intermediate), dtype=np.uint16)
    up_expected = np.zeros_like(gate_expected)
    for r in range(rows):
        for s in range(top_k):
            e = int(selected[r, s])
            gate_expected[r, s] = f32_to_bf16_bits(
                ternary_gemv(bf16_round(x_f32[r]), gate[e], gate_alpha[e])
            )
            up_expected[r, s] = f32_to_bf16_bits(
                ternary_gemv(bf16_round(x_f32[r]), up[e], up_alpha[e])
            )

    with DeviceArrays() as dev:
        x_d = dev.put(x)
        selected_d = dev.put(selected)
        gate_d, up_d = dev.put(gate), dev.put(up)
        gate_alpha_d, up_alpha_d = dev.put(gate_alpha), dev.put(up_alpha)
        gate_out, gate_out_d = dev.empty(gate_expected.shape, np.dtype(np.uint16))
        up_out, up_out_d = dev.empty(up_expected.shape, np.dtype(np.uint16))
        maple_selected_ternary_dual_gemv_batched_bf16(
            x_d.ptr,
            gate_d.ptr,
            gate_alpha_d.ptr,
            up_d.ptr,
            up_alpha_d.ptr,
            selected_d.ptr,
            gate_out_d.ptr,
            up_out_d.ptr,
            rows,
            experts,
            top_k,
            hidden,
            intermediate,
            library=maple_ternary_lib,
        )
        dev.get(gate_out, gate_out_d)
        dev.get(up_out, up_out_d)

    assert np.array_equal(gate_out, gate_expected)
    assert np.array_equal(up_out, up_expected)


def test_maple_selected_ternary_gemv_batched_matches_oracle(maple_ternary_lib) -> None:
    """Batched selected-expert down GEMV over (T, top_k) entries is bit-exact (P3)."""

    from hipengine.kernels.hip_gfx1100.quant.maple_ternary import (
        maple_selected_ternary_gemv_batched_bf16,
    )

    rng = np.random.default_rng(66)
    rows, experts, top_k, in_f, out_f = 5, 4, 2, 16, 7
    x_f32 = rng.normal(size=(rows, top_k, in_f)).astype(np.float32)
    x = f32_to_bf16_bits(x_f32)
    selected = rng.integers(0, experts, size=(rows, top_k)).astype(np.int32)
    values = rng.integers(-1, 2, size=(experts, out_f, in_f), dtype=np.int8)
    packed = pack2(values)
    alpha = f32_to_bf16_bits(
        rng.uniform(0.01, 0.5, size=(experts, out_f)).astype(np.float32)
    )
    expected = np.zeros((rows, top_k, out_f), dtype=np.uint16)
    for r in range(rows):
        for s in range(top_k):
            e = int(selected[r, s])
            expected[r, s] = f32_to_bf16_bits(
                ternary_gemv(bf16_round(x_f32[r, s]), packed[e], alpha[e])
            )

    with DeviceArrays() as dev:
        x_d = dev.put(x)
        selected_d = dev.put(selected)
        packed_d, alpha_d = dev.put(packed), dev.put(alpha)
        out, out_d = dev.empty(expected.shape, np.dtype(np.uint16))
        maple_selected_ternary_gemv_batched_bf16(
            x_d.ptr,
            packed_d.ptr,
            alpha_d.ptr,
            selected_d.ptr,
            out_d.ptr,
            rows,
            experts,
            top_k,
            in_f,
            out_f,
            library=maple_ternary_lib,
        )
        dev.get(out, out_d)

    assert np.array_equal(out, expected)


def test_maple_i32_stable_compaction_matches_cpu_expert_order(
    maple_ternary_lib, hip_test_target_arch
) -> None:
    """Maple's int32 routes reuse the generic stable expert compactor."""
    del maple_ternary_lib
    from hipengine.kernels.hip_gfx1100.moe.group_scatter import (
        build_qwen35_moe_group_scatter,
        qwen35_moe_group_compact_active_i32_parallel,
    )

    selected = np.asarray(
        [[3, 1], [0, 3], [1, 1], [3, 0], [2, 3]], dtype=np.int32
    )
    routing = np.arange(selected.size, dtype=np.float32).reshape(selected.shape) / 17.0
    flat = selected.reshape(-1)
    expected_lanes = np.argsort(flat, kind="stable").astype(np.int64)
    expected_experts = flat[expected_lanes].astype(np.int64)
    counts = np.bincount(flat, minlength=4).astype(np.int64)
    expected_starts = np.zeros(5, dtype=np.int64)
    expected_starts[1:] = np.cumsum(counts, dtype=np.int64)
    expected_active = np.flatnonzero(counts).astype(np.int64)

    assert resolve(
        backend="hip_gfx1100",
        layer="moe_group_compact",
        quant="generic",
        variant="active_experts_i32_parallel",
    ) is qwen35_moe_group_compact_active_i32_parallel
    with hip_target_arch_environment(hip_test_target_arch):
        group_lib = build_qwen35_moe_group_scatter(load=True)
    with DeviceArrays() as dev:
        selected_d = dev.put(selected)
        routing_d = dev.put(routing)
        starts, starts_d = dev.empty((5,), np.dtype(np.int64))
        active, active_d = dev.empty((4,), np.dtype(np.int64))
        active_count, active_count_d = dev.empty((1,), np.dtype(np.int64))
        lanes, lanes_d = dev.empty(flat.shape, np.dtype(np.int64))
        experts, experts_d = dev.empty(flat.shape, np.dtype(np.int64))
        weights, weights_d = dev.empty(flat.shape, np.dtype(np.float32))
        qwen35_moe_group_compact_active_i32_parallel(
            selected_d.ptr,
            routing_d.ptr,
            starts_d.ptr,
            active_d.ptr,
            active_count_d.ptr,
            lanes_d.ptr,
            experts_d.ptr,
            weights_d.ptr,
            flat.size,
            4,
            library=group_lib,
        )
        for host, device in (
            (starts, starts_d),
            (active, active_d),
            (active_count, active_count_d),
            (lanes, lanes_d),
            (experts, experts_d),
            (weights, weights_d),
        ):
            dev.get(host, device)

    assert int(active_count[0]) == expected_active.size
    np.testing.assert_array_equal(starts, expected_starts)
    np.testing.assert_array_equal(active[: expected_active.size], expected_active)
    np.testing.assert_array_equal(lanes, expected_lanes)
    np.testing.assert_array_equal(experts, expected_experts)
    np.testing.assert_array_equal(weights, routing.reshape(-1)[expected_lanes])


def test_maple_selected_ternary_grouped_matches_row_route_oracle(
    maple_ternary_lib,
) -> None:
    """Expert-major gate/up/down preserves every row/route BF16 boundary."""
    rng = np.random.default_rng(407)
    rows, experts, top_k, hidden, intermediate = 7, 5, 3, 32, 16
    selected = np.asarray(
        [[4, 1, 4], [0, 4, 1], [3, 1, 0], [4, 2, 1], [0, 3, 4], [2, 1, 2], [4, 0, 3]],
        dtype=np.int32,
    )
    flat = selected.reshape(-1)
    sorted_lanes = np.argsort(flat, kind="stable").astype(np.int64)
    counts = np.bincount(flat, minlength=experts).astype(np.int64)
    starts = np.zeros(experts + 1, dtype=np.int64)
    starts[1:] = np.cumsum(counts, dtype=np.int64)

    x_f32 = rng.normal(size=(rows, hidden)).astype(np.float32)
    x = f32_to_bf16_bits(x_f32)
    gate = pack2(rng.integers(-1, 2, size=(experts, intermediate, hidden), dtype=np.int8))
    up = pack2(rng.integers(-1, 2, size=(experts, intermediate, hidden), dtype=np.int8))
    gate_alpha = f32_to_bf16_bits(
        rng.uniform(0.01, 0.5, size=(experts, intermediate)).astype(np.float32)
    )
    up_alpha = f32_to_bf16_bits(
        rng.uniform(0.01, 0.5, size=(experts, intermediate)).astype(np.float32)
    )
    gate_expected = np.empty((rows, top_k, intermediate), dtype=np.uint16)
    up_expected = np.empty_like(gate_expected)
    for row in range(rows):
        for route in range(top_k):
            expert = int(selected[row, route])
            gate_expected[row, route] = f32_to_bf16_bits(
                ternary_gemv(bf16_round(x_f32[row]), gate[expert], gate_alpha[expert])
            )
            up_expected[row, route] = f32_to_bf16_bits(
                ternary_gemv(bf16_round(x_f32[row]), up[expert], up_alpha[expert])
            )

    down_x_f32 = rng.normal(size=(rows, top_k, intermediate)).astype(np.float32)
    down_x = f32_to_bf16_bits(down_x_f32)
    down = pack2(rng.integers(-1, 2, size=(experts, hidden, intermediate), dtype=np.int8))
    down_alpha = f32_to_bf16_bits(
        rng.uniform(0.01, 0.5, size=(experts, hidden)).astype(np.float32)
    )
    down_expected = np.empty((rows, top_k, hidden), dtype=np.uint16)
    for row in range(rows):
        for route in range(top_k):
            expert = int(selected[row, route])
            down_expected[row, route] = f32_to_bf16_bits(
                ternary_gemv(
                    bf16_round(down_x_f32[row, route]),
                    down[expert],
                    down_alpha[expert],
                )
            )

    with DeviceArrays() as dev:
        starts_d = dev.put(starts)
        lanes_d = dev.put(sorted_lanes)
        x_d, gate_d, gate_alpha_d = dev.put(x), dev.put(gate), dev.put(gate_alpha)
        up_d, up_alpha_d = dev.put(up), dev.put(up_alpha)
        gate_out, gate_out_d = dev.empty(gate_expected.shape, np.dtype(np.uint16))
        up_out, up_out_d = dev.empty(up_expected.shape, np.dtype(np.uint16))
        maple_selected_ternary_dual_grouped_bf16(
            x_d.ptr,
            gate_d.ptr,
            gate_alpha_d.ptr,
            up_d.ptr,
            up_alpha_d.ptr,
            starts_d.ptr,
            lanes_d.ptr,
            gate_out_d.ptr,
            up_out_d.ptr,
            rows,
            experts,
            top_k,
            hidden,
            intermediate,
            library=maple_ternary_lib,
        )
        down_x_d, down_d, down_alpha_d = (
            dev.put(down_x),
            dev.put(down),
            dev.put(down_alpha),
        )
        down_out, down_out_d = dev.empty(down_expected.shape, np.dtype(np.uint16))
        maple_selected_ternary_grouped_bf16(
            down_x_d.ptr,
            down_d.ptr,
            down_alpha_d.ptr,
            starts_d.ptr,
            lanes_d.ptr,
            down_out_d.ptr,
            rows,
            experts,
            top_k,
            intermediate,
            hidden,
            library=maple_ternary_lib,
        )
        dev.get(gate_out, gate_out_d)
        dev.get(up_out, up_out_d)
        dev.get(down_out, down_out_d)

    np.testing.assert_array_equal(gate_out, gate_expected)
    np.testing.assert_array_equal(up_out, up_expected)
    np.testing.assert_array_equal(down_out, down_expected)


def test_maple_moe_fused_dual_swiglu_matches_unfused_chain(
    maple_ternary_lib, hip_test_target_arch
) -> None:
    """M2 fused dual gate/up + clamped SiLU is bit-exact with the unfused chain.

    Compares maple_moe_dual_swiglu against the standalone dual gemv + clamped
    swiglu chain. The fused intermediate output must match bit-for-bit.
    """
    with hip_target_arch_environment(hip_test_target_arch):
        moe_lib = build_maple_moe(load=True)

    rng = np.random.default_rng(44)
    experts, top_k, hidden, intermediate = 4, 2, 32, 64
    selected = np.asarray([3, 1], dtype=np.int32)

    # gate/up dual gemv inputs (x -> [top_k, intermediate] gate and up).
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

    with DeviceArrays() as dev:
        x_d = dev.put(x)
        selected_d = dev.put(selected)
        gate_d, up_d = dev.put(gate), dev.put(up)
        gate_alpha_d, up_alpha_d = dev.put(gate_alpha), dev.put(up_alpha)

        # Unfused chain: dual gemv -> gate/up, then clamped swiglu -> intermediate.
        _, gate_out_d = dev.empty((top_k, intermediate), np.dtype(np.uint16))
        _, up_out_d = dev.empty((top_k, intermediate), np.dtype(np.uint16))
        maple_selected_ternary_dual_gemv_bf16(
            x_d.ptr, gate_d.ptr, gate_alpha_d.ptr, up_d.ptr, up_alpha_d.ptr,
            selected_d.ptr, gate_out_d.ptr, up_out_d.ptr,
            experts, top_k, hidden, intermediate, library=maple_ternary_lib,
        )
        inter, inter_d = dev.empty((top_k, intermediate), np.dtype(np.uint16))
        maple_clamped_swiglu_bf16(
            gate_out_d.ptr, up_out_d.ptr, inter_d.ptr,
            top_k, intermediate, library=moe_lib,
        )

        # Fused path: single kernel -> intermediate.
        fused_inter, fused_inter_d = dev.empty((top_k, intermediate), np.dtype(np.uint16))
        maple_moe_dual_swiglu_bf16(
            x_d.ptr, gate_d.ptr, gate_alpha_d.ptr, up_d.ptr, up_alpha_d.ptr,
            selected_d.ptr, fused_inter_d.ptr,
            experts, top_k, hidden, intermediate, library=maple_ternary_lib,
        )

        dev.get(inter, inter_d)
        dev.get(fused_inter, fused_inter_d)

    assert np.array_equal(
        fused_inter, inter
    ), "fused dual+swiglu intermediate mismatch"
