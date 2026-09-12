from __future__ import annotations

import ctypes

import numpy as np
import pytest

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc
from hipengine.kernels.hip_gfx1100.quant.gguf_expert_pack8_gemv import (
    build_gguf_expert_pack8_gemv,
    gguf_q4_k_expert_pack8_dual_selected_bf16_bf16_out,
    gguf_q4_k_expert_pack8_selected_bf16_bf16_out,
    gguf_q5_k_expert_pack8_selected_bf16_bf16_out,
    gguf_q6_k_expert_pack8_selected_bf16_bf16_out,
    plan_gguf_expert_pack8_gemv_build,
)
from hipengine.kernels.registry import resolve
from hipengine.loading.materialize import float_array_to_bf16_bits
from hipengine.loading.qwen35_gguf_expert_sidecar import (
    GGUFExpertPackedTensor,
    dequantize_packed_expert_tensor,
    pack_gguf_expert_tensor,
)
from hipengine.quant.gguf import GGMLQuantizationType, bf16_to_float32, quant_layout


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


HIP_AVAILABLE = _hip_available()


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
@pytest.mark.parametrize(
    ("qtype", "kernel", "atol"),
    (
        (GGMLQuantizationType.Q4_K, gguf_q4_k_expert_pack8_selected_bf16_bf16_out, 2.5e-2),
        (GGMLQuantizationType.Q5_K, gguf_q5_k_expert_pack8_selected_bf16_bf16_out, 2.5e-2),
        (GGMLQuantizationType.Q6_K, gguf_q6_k_expert_pack8_selected_bf16_bf16_out, 2.5e-2),
    ),
)
def test_selected_expert_pack8_kernel_matches_cpu_reference(
    qtype: GGMLQuantizationType,
    kernel,
    atol: float,
) -> None:
    runtime = get_hip_runtime()
    library = build_gguf_expert_pack8_gemv(load=True)
    raw = _synthetic_expert_blocks(qtype, experts=3, out_features=16, blocks_per_row=2)
    packed = pack_gguf_expert_tensor(raw, qtype, tensor_name=f"synthetic.{qtype.name}", slot="ffn_gate_exps")
    x = ((np.arange(2 * packed.in_features, dtype=np.float32).reshape(2, packed.in_features) % 17) - 8) / 64.0
    x_bits = float_array_to_bf16_bits(x)
    selected = np.asarray([0, 2, 1, 2], dtype=np.int64)
    expected = _selected_reference(x, selected, packed)
    actual = np.empty_like(float_array_to_bf16_bits(expected))

    bufs = []
    try:
        x_dev = _dev(x_bits, runtime, bufs)
        selected_dev = _dev(selected, runtime, bufs)
        qlow_dev = _dev(packed.qweight_low, runtime, bufs)
        qhigh_dev = _dev(packed.qweight_high, runtime, bufs) if packed.qweight_high is not None else None
        scales_dev = _dev(packed.scales, runtime, bufs)
        mins_dev = _dev(packed.mins, runtime, bufs) if packed.mins is not None else None
        out_dev = malloc(actual.nbytes, runtime=runtime)
        bufs.append(out_dev)
        kernel(
            x_dev.ptr,
            selected_dev.ptr,
            qlow_dev.ptr,
            0 if qhigh_dev is None else qhigh_dev.ptr,
            scales_dev.ptr,
            0 if mins_dev is None else mins_dev.ptr,
            out_dev.ptr,
            x_rows=2,
            rows=4,
            num_experts=packed.num_experts,
            in_features=packed.in_features,
            out_features=packed.out_features,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(actual), out_dev, runtime=runtime)
    finally:
        for buf in reversed(bufs):
            free(buf, runtime=runtime)

    np.testing.assert_allclose(bf16_to_float32(actual), bf16_to_float32(float_array_to_bf16_bits(expected)), rtol=0.0, atol=atol)


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_q4_dual_selected_expert_pack8_matches_single_kernels() -> None:
    runtime = get_hip_runtime()
    library = build_gguf_expert_pack8_gemv(load=True)
    raw_a = _synthetic_expert_blocks(GGMLQuantizationType.Q4_K, experts=3, out_features=16, blocks_per_row=2, seed=11)
    raw_b = _synthetic_expert_blocks(GGMLQuantizationType.Q4_K, experts=3, out_features=16, blocks_per_row=2, seed=29)
    pack_a = pack_gguf_expert_tensor(raw_a, GGMLQuantizationType.Q4_K, tensor_name="a", slot="ffn_gate_exps")
    pack_b = pack_gguf_expert_tensor(raw_b, GGMLQuantizationType.Q4_K, tensor_name="b", slot="ffn_up_exps")
    x = ((np.arange(2 * pack_a.in_features, dtype=np.float32).reshape(2, pack_a.in_features) % 13) - 6) / 64.0
    x_bits = float_array_to_bf16_bits(x)
    selected = np.asarray([0, 2, 1, 2], dtype=np.int64)
    expected_a = np.empty((4, pack_a.out_features), dtype=np.uint16)
    expected_b = np.empty((4, pack_b.out_features), dtype=np.uint16)
    actual_a = np.empty_like(expected_a)
    actual_b = np.empty_like(expected_b)

    bufs = []
    try:
        x_dev = _dev(x_bits, runtime, bufs)
        selected_dev = _dev(selected, runtime, bufs)
        a_low = _dev(pack_a.qweight_low, runtime, bufs)
        a_scales = _dev(pack_a.scales, runtime, bufs)
        a_mins = _dev(pack_a.mins, runtime, bufs)
        b_low = _dev(pack_b.qweight_low, runtime, bufs)
        b_scales = _dev(pack_b.scales, runtime, bufs)
        b_mins = _dev(pack_b.mins, runtime, bufs)
        exp_a_dev = malloc(expected_a.nbytes, runtime=runtime)
        exp_b_dev = malloc(expected_b.nbytes, runtime=runtime)
        act_a_dev = malloc(actual_a.nbytes, runtime=runtime)
        act_b_dev = malloc(actual_b.nbytes, runtime=runtime)
        bufs.extend((exp_a_dev, exp_b_dev, act_a_dev, act_b_dev))
        gguf_q4_k_expert_pack8_selected_bf16_bf16_out(
            x_dev.ptr,
            selected_dev.ptr,
            a_low.ptr,
            0,
            a_scales.ptr,
            a_mins.ptr,
            exp_a_dev.ptr,
            x_rows=2,
            rows=4,
            num_experts=pack_a.num_experts,
            in_features=pack_a.in_features,
            out_features=pack_a.out_features,
            library=library,
            runtime=runtime,
        )
        gguf_q4_k_expert_pack8_selected_bf16_bf16_out(
            x_dev.ptr,
            selected_dev.ptr,
            b_low.ptr,
            0,
            b_scales.ptr,
            b_mins.ptr,
            exp_b_dev.ptr,
            x_rows=2,
            rows=4,
            num_experts=pack_b.num_experts,
            in_features=pack_b.in_features,
            out_features=pack_b.out_features,
            library=library,
            runtime=runtime,
        )
        gguf_q4_k_expert_pack8_dual_selected_bf16_bf16_out(
            x_dev.ptr,
            selected_dev.ptr,
            a_low.ptr,
            a_scales.ptr,
            a_mins.ptr,
            b_low.ptr,
            b_scales.ptr,
            b_mins.ptr,
            act_a_dev.ptr,
            act_b_dev.ptr,
            x_rows=2,
            rows=4,
            num_experts=pack_a.num_experts,
            in_features=pack_a.in_features,
            out_features=pack_a.out_features,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(expected_a), exp_a_dev, runtime=runtime)
        copy_device_to_host(host_array_ptr(expected_b), exp_b_dev, runtime=runtime)
        copy_device_to_host(host_array_ptr(actual_a), act_a_dev, runtime=runtime)
        copy_device_to_host(host_array_ptr(actual_b), act_b_dev, runtime=runtime)
    finally:
        for buf in reversed(bufs):
            free(buf, runtime=runtime)

    np.testing.assert_array_equal(actual_a, expected_a)
    np.testing.assert_array_equal(actual_b, expected_b)


def test_expert_pack8_registry_and_build_plan() -> None:
    assert resolve(
        backend="hip_gfx1100",
        layer="moe_linear",
        quant="gguf_q4_k",
        variant="expert_pack8_selected_bf16_bf16_out",
    ) is gguf_q4_k_expert_pack8_selected_bf16_bf16_out
    assert resolve(
        backend="hip_gfx1100",
        layer="moe_linear",
        quant="gguf_q5_k",
        variant="expert_pack8_selected_bf16_bf16_out",
    ) is gguf_q5_k_expert_pack8_selected_bf16_bf16_out
    artifact = plan_gguf_expert_pack8_gemv_build(compiler_version="test-compiler")
    assert artifact.output_path.name == "gguf_expert_pack8_gemv.so"
    assert any(path.name == "gguf_expert_pack8_gemv.hip" for path in artifact.sources)


def _selected_reference(x: np.ndarray, selected: np.ndarray, packed: GGUFExpertPackedTensor) -> np.ndarray:
    weights = dequantize_packed_expert_tensor(packed)
    rows = selected.size
    lanes_per_x_row = rows // x.shape[0]
    out = np.empty((rows, packed.out_features), dtype=np.float32)
    for row, expert in enumerate(selected.tolist()):
        x_row = 0 if x.shape[0] == 1 else row // lanes_per_x_row
        out[row] = x[x_row].astype(np.float32) @ weights[int(expert)].T.astype(np.float32)
    return out.astype(np.float32)


def _dev(array: np.ndarray, runtime, bufs: list):
    contiguous = np.ascontiguousarray(array)
    buf = malloc(contiguous.nbytes, runtime=runtime)
    bufs.append(buf)
    copy_host_to_device(buf, host_array_ptr(contiguous), runtime=runtime)
    return buf


def _synthetic_expert_blocks(
    qtype: GGMLQuantizationType,
    *,
    experts: int,
    out_features: int,
    blocks_per_row: int,
    seed: int = 1234,
) -> np.ndarray:
    layout = quant_layout(qtype)
    rng = np.random.default_rng(seed + int(qtype))
    blocks = rng.integers(
        0,
        256,
        size=(experts, out_features, blocks_per_row, layout.type_size),
        dtype=np.uint8,
    )
    if qtype == GGMLQuantizationType.Q4_K:
        _store_f16(blocks, 0, 0.015625)
        _store_f16(blocks, 2, 0.0078125)
    elif qtype == GGMLQuantizationType.Q5_K:
        _store_f16(blocks, 0, 0.0078125)
        _store_f16(blocks, 2, 0.00390625)
    elif qtype == GGMLQuantizationType.Q6_K:
        _store_f16(blocks, 208, 0.00390625)
    else:  # pragma: no cover
        raise AssertionError(qtype)
    return blocks.reshape(experts, out_features, blocks_per_row * layout.type_size)


def _store_f16(blocks: np.ndarray, offset: int, value: float) -> None:
    bits = np.asarray([value], dtype=np.float16).view(np.uint8)
    blocks[..., offset : offset + 2] = bits
