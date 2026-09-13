from __future__ import annotations

import ctypes

import numpy as np
import pytest

from hipengine.kernels.cpu_reference import gguf_q5_k_gemv, gguf_q6_k_gemv, gguf_q8_0_gemv
from hipengine.kernels.hip_gfx1100.quant.gguf_k_gemv import (
    build_gguf_k_gemv,
    gguf_q5_k_gemv_bf16_bf16_out,
    gguf_q5_k_gemv_bf16_f32_out,
    gguf_q5_k_gemv_f32_f32_out,
    gguf_q5_k_gemv_fp16_f32_out,
    gguf_q6_k_gemv_bf16_bf16_out,
    gguf_q6_k_gemv_bf16_f32_out,
    gguf_q6_k_gemv_f32_f32_out,
    gguf_q6_k_gemv_fp16_f32_out,
    gguf_q8_0_gemv_bf16_bf16_out,
    gguf_q8_0_gemv_bf16_f32_out,
    gguf_q8_0_dual_gemv_f32_f32_out,
    gguf_q8_0_pack8_gemv_f32_f32_out,
    gguf_q8_0_gemv_f32_f32_out,
    gguf_q8_0_gemv_fp16_f32_out,
    gguf_q8_0_gemv_rowbatch32_f32_f32_out,
    gguf_q8_0_gemv_coltile4_rowbatch8_f32_f32_out,
    gguf_q8_0_gemv_coltile8_rowbatch4_f32_f32_out,
    gguf_q8_0_gr_up_sigmoid_mean_coltile2_branch4_rowbatch4_f32,
    gguf_q8_0_gemv_coltile8_rowbatch8_f32_f32_out,
    gguf_q8_0_gemv_coltile16_rowbatch2_f32_f32_out,
    gguf_q8_0_gemv_coltile16_rowbatch4_f32_f32_out,
    gguf_q8_0_gemv_coltile32_rowbatch1_f32_f32_out,
    plan_gguf_k_gemv_build,
)
from hipengine.kernels.registry import resolve
from hipengine.quant.gguf import GGMLQuantizationType, dequantize_gguf_data

QK_K = 256
Q8_0_BLOCK_BYTES = 34
Q5_K_BLOCK_BYTES = 176
Q6_K_BLOCK_BYTES = 210


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def make_q8_0_weight(out_features: int, in_features: int) -> np.ndarray:
    if in_features % 32:
        raise ValueError("in_features must be a multiple of 32")
    blocks_per_row = in_features // 32
    data = np.empty((out_features, blocks_per_row * Q8_0_BLOCK_BYTES), dtype=np.uint8)
    for out_idx in range(out_features):
        for block_idx in range(blocks_per_row):
            start = block_idx * Q8_0_BLOCK_BYTES
            data[out_idx, start : start + Q8_0_BLOCK_BYTES] = _make_q8_0_block(
                out_idx, block_idx
            )
    return data


def make_q5_k_weight(out_features: int, in_features: int) -> np.ndarray:
    if in_features % QK_K:
        raise ValueError("in_features must be a multiple of 256")
    blocks_per_row = in_features // QK_K
    data = np.empty((out_features, blocks_per_row * Q5_K_BLOCK_BYTES), dtype=np.uint8)
    for out_idx in range(out_features):
        for block_idx in range(blocks_per_row):
            start = block_idx * Q5_K_BLOCK_BYTES
            data[out_idx, start : start + Q5_K_BLOCK_BYTES] = _make_q5_k_block(
                out_idx, block_idx
            )
    return data


def make_q6_k_weight(out_features: int, in_features: int) -> np.ndarray:
    if in_features % QK_K:
        raise ValueError("in_features must be a multiple of 256")
    blocks_per_row = in_features // QK_K
    data = np.empty((out_features, blocks_per_row * Q6_K_BLOCK_BYTES), dtype=np.uint8)
    for out_idx in range(out_features):
        for block_idx in range(blocks_per_row):
            start = block_idx * Q6_K_BLOCK_BYTES
            data[out_idx, start : start + Q6_K_BLOCK_BYTES] = _make_q6_k_block(
                out_idx, block_idx
            )
    return data


def _make_q8_0_block(out_idx: int, block_idx: int) -> np.ndarray:
    d = np.float16(0.03125 * (1 + (out_idx % 5)))
    q = ((np.arange(32, dtype=np.int16) + out_idx * 7 + block_idx * 3) % 31 - 15).astype(
        np.int8
    )
    return np.concatenate([np.asarray([d], dtype=np.float16).view(np.uint8), q.view(np.uint8)])


def _make_q5_k_block(out_idx: int, block_idx: int) -> np.ndarray:
    d = np.float16(0.015625 * (1 + (out_idx % 5)))
    dmin = np.float16(0.0078125 * (1 + (block_idx % 3)))
    scales = ((np.arange(8, dtype=np.uint8) * 3 + out_idx + block_idx) % 63 + 1).astype(
        np.uint8
    )
    mins = ((np.arange(8, dtype=np.uint8) * 5 + 2 * out_idx + block_idx) % 17).astype(
        np.uint8
    )
    q = ((np.arange(QK_K, dtype=np.uint16) + out_idx * 7 + block_idx * 11) % 32).astype(
        np.uint8
    )
    q_groups = q.reshape(8, 32)
    qh = np.zeros(32, dtype=np.uint8)
    for subblock in range(8):
        qh |= ((q_groups[subblock] >> np.uint8(4)) & np.uint8(1)) << np.uint8(subblock)
    low = q_groups & np.uint8(0x0F)
    qs = np.empty(128, dtype=np.uint8)
    for pair in range(4):
        qs[pair * 32 : (pair + 1) * 32] = low[2 * pair] | (low[2 * pair + 1] << 4)
    return np.concatenate(
        [
            np.asarray([d], dtype=np.float16).view(np.uint8),
            np.asarray([dmin], dtype=np.float16).view(np.uint8),
            _pack_q4_k_scales(scales, mins),
            qh,
            qs,
        ]
    )


def _make_q6_k_block(out_idx: int, block_idx: int) -> np.ndarray:
    d = np.float16(0.0107421875 * (1 + (out_idx % 3)))
    scales = ((np.arange(16, dtype=np.int16) * 3 + out_idx - block_idx) % 31 - 15).astype(
        np.int8
    )
    q_signed = ((np.arange(QK_K, dtype=np.int16) + out_idx * 5 + block_idx * 9) % 64 - 32)
    q = (q_signed + 32).astype(np.uint8)
    ql = np.zeros(128, dtype=np.uint8)
    qh = np.zeros(64, dtype=np.uint8)
    for k, value in enumerate(q):
        group32 = k >> 5
        lane = k & 31
        base64 = 64 if group32 >= 4 else 0
        ql_group = group32 & 1
        ql_idx = base64 + ql_group * 32 + lane
        if (group32 & 2) == 0:
            ql[ql_idx] |= value & np.uint8(0x0F)
        else:
            ql[ql_idx] |= (value & np.uint8(0x0F)) << np.uint8(4)
        qh_idx = (32 if group32 >= 4 else 0) + lane
        qh[qh_idx] |= ((value >> np.uint8(4)) & np.uint8(0x03)) << np.uint8(
            2 * (group32 & 3)
        )
    return np.concatenate(
        [ql, qh, scales.view(np.uint8), np.asarray([d], dtype=np.float16).view(np.uint8)]
    )


def _pack_q4_k_scales(scales: np.ndarray, mins: np.ndarray) -> np.ndarray:
    scales = np.asarray(scales, dtype=np.uint8)
    mins = np.asarray(mins, dtype=np.uint8)
    out = np.zeros(12, dtype=np.uint8)
    out[:4] = (scales[:4] & 0x3F) | ((scales[4:] & 0x30) << 2)
    out[4:8] = (mins[:4] & 0x3F) | ((mins[4:] & 0x30) << 2)
    out[8:12] = (scales[4:] & 0x0F) | ((mins[4:] & 0x0F) << 4)
    return out


@pytest.mark.parametrize(
    ("name", "qtype", "make_weight", "reference", "in_features"),
    [
        ("Q8_0", GGMLQuantizationType.Q8_0, make_q8_0_weight, gguf_q8_0_gemv, 64),
        ("Q5_K", GGMLQuantizationType.Q5_K, make_q5_k_weight, gguf_q5_k_gemv, 512),
        ("Q6_K", GGMLQuantizationType.Q6_K, make_q6_k_weight, gguf_q6_k_gemv, 512),
    ],
)
def test_cpu_reference_gguf_k_gemv_matches_dequantized_matmul(
    name: str,
    qtype: GGMLQuantizationType,
    make_weight,
    reference,
    in_features: int,
) -> None:
    x = (np.arange(2 * in_features, dtype=np.float32).reshape(2, in_features) % 13 - 6) / 8.0
    qweight = make_weight(out_features=5, in_features=in_features)

    out = reference(x, qweight)
    weight = dequantize_gguf_data(qweight, qtype)
    expected = np.matmul(x, weight.T).astype(np.float32)

    assert out.dtype == np.float32, name
    assert out.shape == (2, 5), name
    np.testing.assert_allclose(out, expected, rtol=0.0, atol=1e-5)


def test_gguf_k_gemv_registry_and_build_plan() -> None:
    expected = {
        "gguf_q8_0": (
            gguf_q8_0_gemv_f32_f32_out,
            gguf_q8_0_gemv_fp16_f32_out,
            gguf_q8_0_gemv_bf16_f32_out,
            gguf_q8_0_gemv_bf16_bf16_out,
            gguf_q8_0_gemv,
        ),
        "gguf_q5_k": (
            gguf_q5_k_gemv_f32_f32_out,
            gguf_q5_k_gemv_fp16_f32_out,
            gguf_q5_k_gemv_bf16_f32_out,
            gguf_q5_k_gemv_bf16_bf16_out,
            gguf_q5_k_gemv,
        ),
        "gguf_q6_k": (
            gguf_q6_k_gemv_f32_f32_out,
            gguf_q6_k_gemv_fp16_f32_out,
            gguf_q6_k_gemv_bf16_f32_out,
            gguf_q6_k_gemv_bf16_bf16_out,
            gguf_q6_k_gemv,
        ),
    }
    for quant, (f32_fn, fp16_fn, bf16_fn, bf16_out_fn, cpu_fn) in expected.items():
        assert resolve(
            backend="hip_gfx1100", layer="linear", quant=quant, variant="gemv_f32_f32_out"
        ) is f32_fn
        assert resolve(
            backend="hip_gfx1100", layer="linear", quant=quant, variant="gemv_fp16_f32_out"
        ) is fp16_fn
        assert resolve(
            backend="hip_gfx1100", layer="linear", quant=quant, variant="gemv_bf16_f32_out"
        ) is bf16_fn
        assert resolve(
            backend="hip_gfx1100", layer="linear", quant=quant, variant="gemv_bf16_bf16_out"
        ) is bf16_out_fn
        assert resolve(
            backend="cpu_reference", layer="linear", quant=quant, variant="gemv_f32_f32_out"
        ) is cpu_fn

    artifact = plan_gguf_k_gemv_build(compiler_version="test-compiler")
    assert artifact.output_path.name == "gguf_k_gemv.so"
    assert "gguf_k_gemv" in str(artifact.output_path)
    assert any(path.name == "gguf_k_gemv.hip" for path in artifact.sources)

    dry_run = build_gguf_k_gemv(dry_run=True, compiler_version="test-compiler")
    assert dry_run.output_path == artifact.output_path


def test_gguf_k_wrapper_validates_kernel_contract() -> None:
    with pytest.raises(ValueError, match="divisible"):
        gguf_q8_0_gemv_f32_f32_out(1, 2, 3, rows=1, in_features=31, out_features=1)

    with pytest.raises(ValueError, match="divisible"):
        gguf_q5_k_gemv_f32_f32_out(1, 2, 3, rows=1, in_features=255, out_features=1)

    with pytest.raises(ValueError, match="threads"):
        gguf_q6_k_gemv_f32_f32_out(
            1, 2, 3, rows=1, in_features=256, out_features=1, threads=96
        )


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_q8_0_f32_rowbatch32_matches_scalar_prefill_bits() -> None:
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc

    rows, in_features, out_features = 37, 256, 33
    rng = np.random.default_rng(3808)
    x = np.ascontiguousarray(rng.normal(0.0, 0.2, size=(rows, in_features)).astype(np.float32))
    weight = np.ascontiguousarray(make_q8_0_weight(out_features, in_features))
    runtime = get_hip_runtime(); library = build_gguf_k_gemv(load=True); bufs = []
    try:
        for host in (x, weight):
            device = malloc(host.nbytes, runtime=runtime); copy_host_to_device(device, host_array_ptr(host), runtime=runtime); bufs.append(device)
        baseline = malloc(rows * out_features * 4, runtime=runtime); candidate = malloc(rows * out_features * 4, runtime=runtime); coltiles = [malloc(rows * 32 * 4, runtime=runtime) for _ in range(6)]; bufs.extend((baseline, candidate, *coltiles))
        gguf_q8_0_gemv_f32_f32_out(bufs[0].ptr, bufs[1].ptr, baseline.ptr, rows, in_features, out_features, library=library, runtime=runtime)
        gguf_q8_0_gemv_rowbatch32_f32_f32_out(bufs[0].ptr, bufs[1].ptr, candidate.ptr, rows, in_features, out_features, library=library, runtime=runtime)
        # Constant-32 accumulator tiles cover every factorization of 32.
        for wrapper, output in zip((gguf_q8_0_gemv_coltile4_rowbatch8_f32_f32_out, gguf_q8_0_gemv_coltile8_rowbatch4_f32_f32_out, gguf_q8_0_gemv_coltile8_rowbatch8_f32_f32_out, gguf_q8_0_gemv_coltile16_rowbatch2_f32_f32_out, gguf_q8_0_gemv_coltile16_rowbatch4_f32_f32_out, gguf_q8_0_gemv_coltile32_rowbatch1_f32_f32_out), coltiles, strict=True):
            wrapper(bufs[0].ptr, bufs[1].ptr, output.ptr, rows, in_features, 32, library=library, runtime=runtime)
        runtime.device_synchronize(); got0=np.empty((rows,out_features),np.float32);got1=np.empty_like(got0);got_tiles=[np.empty((rows,32),np.float32) for _ in coltiles];copy_device_to_host(host_array_ptr(got0),baseline,runtime=runtime);copy_device_to_host(host_array_ptr(got1),candidate,runtime=runtime)
        for got, output in zip(got_tiles, coltiles, strict=True): copy_device_to_host(host_array_ptr(got),output,got.nbytes,runtime=runtime)
    finally:
        for buf in reversed(bufs): free(buf,runtime=runtime)
    np.testing.assert_array_equal(got1, got0)
    for got in got_tiles:
        np.testing.assert_array_equal(got, got0[:, :32])


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_q8_0_gr_up_sigmoid_mean_is_bit_exact_to_primitive_chain() -> None:
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )
    from hipengine.kernels.hip_gfx1100.fused.qwen4_exp_gr import (
        qwen4_exp_gated_mean_f32,
        qwen4_exp_sigmoid_f32,
    )

    rows, in_features, branches, hidden = 37, 320, 4, 2560
    out_features = branches * hidden
    rng = np.random.default_rng(3822)
    x = np.ascontiguousarray(
        rng.normal(0.0, 0.2, size=(rows, in_features)).astype(np.float32)
    )
    normalized = np.ascontiguousarray(
        rng.normal(0.0, 0.2, size=(rows, branches, hidden)).astype(np.float32)
    )
    weight = np.ascontiguousarray(
        np.tile(make_q8_0_weight(hidden, in_features), (branches, 1))
    )
    runtime = get_hip_runtime()
    library = build_gguf_k_gemv(load=True)
    buffers = []
    try:
        for host in (x, normalized, weight):
            device = malloc(host.nbytes, runtime=runtime)
            copy_host_to_device(device, host_array_ptr(host), runtime=runtime)
            buffers.append(device)
        for nbytes in (
            rows * out_features * 4,
            rows * hidden * 4,
            rows * out_features * 4,
            rows * hidden * 4,
        ):
            buffers.append(malloc(nbytes, runtime=runtime))
        gguf_q8_0_gemv_coltile8_rowbatch4_f32_f32_out(
            buffers[0].ptr, buffers[2].ptr, buffers[3].ptr,
            rows, in_features, out_features, library=library, runtime=runtime,
        )
        qwen4_exp_sigmoid_f32(
            buffers[3].ptr,
            buffers[3].ptr,
            rows * out_features,
            runtime=runtime,
        )
        qwen4_exp_gated_mean_f32(
            buffers[1].ptr,
            buffers[3].ptr,
            buffers[4].ptr,
            rows,
            branches,
            hidden,
            runtime=runtime,
        )
        gguf_q8_0_gr_up_sigmoid_mean_coltile2_branch4_rowbatch4_f32(
            buffers[0].ptr,
            buffers[2].ptr,
            buffers[1].ptr,
            buffers[5].ptr,
            buffers[6].ptr,
            rows,
            in_features,
            branches,
            hidden,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        outputs = []
        for device, shape in (
            (buffers[3], (rows, out_features)),
            (buffers[4], (rows, hidden)),
            (buffers[5], (rows, out_features)),
            (buffers[6], (rows, hidden)),
        ):
            host = np.empty(shape, dtype=np.float32)
            copy_device_to_host(host_array_ptr(host), device, runtime=runtime)
            outputs.append(host)
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)

    np.testing.assert_array_equal(outputs[2].view(np.uint32), outputs[0].view(np.uint32))
    np.testing.assert_array_equal(outputs[3].view(np.uint32), outputs[1].view(np.uint32))


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_q8_0_pack8_f32_matches_scalar_gemv_bits() -> None:
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )

    rows, in_features, out_features = 1, 256, 32
    rng = np.random.default_rng(3816)
    x = np.ascontiguousarray(
        rng.normal(0.0, 0.2, size=(rows, in_features)).astype(np.float32)
    )
    weight = np.ascontiguousarray(make_q8_0_weight(out_features, in_features))
    runtime = get_hip_runtime()
    library = build_gguf_k_gemv(load=True)
    buffers = []
    try:
        for host in (x, weight):
            device = malloc(host.nbytes, runtime=runtime)
            copy_host_to_device(device, host_array_ptr(host), runtime=runtime)
            buffers.append(device)
        baseline = malloc(rows * out_features * 4, runtime=runtime)
        candidate = malloc(rows * out_features * 4, runtime=runtime)
        buffers.extend((baseline, candidate))
        gguf_q8_0_gemv_f32_f32_out(
            buffers[0].ptr, buffers[1].ptr, baseline.ptr,
            rows, in_features, out_features, library=library, runtime=runtime,
        )
        gguf_q8_0_pack8_gemv_f32_f32_out(
            buffers[0].ptr, buffers[1].ptr, candidate.ptr,
            rows, in_features, out_features, library=library, runtime=runtime,
        )
        runtime.device_synchronize()
        got = np.empty((rows, out_features), dtype=np.float32)
        expected = np.empty_like(got)
        copy_device_to_host(host_array_ptr(got), candidate, runtime=runtime)
        copy_device_to_host(host_array_ptr(expected), baseline, runtime=runtime)
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)
    np.testing.assert_array_equal(got, expected)


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_q8_0_dual_f32_matches_two_single_gemvs() -> None:
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )

    rows, in_features, out_features = 2, 64, 7
    rng = np.random.default_rng(20260701)
    x = np.ascontiguousarray(rng.standard_normal((rows, in_features)).astype(np.float32) * 0.125)
    qweight_a = np.ascontiguousarray(make_q8_0_weight(out_features, in_features))
    qweight_b = np.ascontiguousarray(qweight_a[::-1].copy())
    runtime = get_hip_runtime()
    library = build_gguf_k_gemv(load=True)
    bufs = []
    try:
        xd = malloc(x.nbytes, runtime=runtime)
        wa = malloc(qweight_a.nbytes, runtime=runtime)
        wb = malloc(qweight_b.nbytes, runtime=runtime)
        single_a = malloc(rows * out_features * 4, runtime=runtime)
        single_b = malloc(rows * out_features * 4, runtime=runtime)
        dual_a = malloc(rows * out_features * 4, runtime=runtime)
        dual_b = malloc(rows * out_features * 4, runtime=runtime)
        bufs.extend((xd, wa, wb, single_a, single_b, dual_a, dual_b))
        copy_host_to_device(xd, host_array_ptr(x), runtime=runtime)
        copy_host_to_device(wa, host_array_ptr(qweight_a), runtime=runtime)
        copy_host_to_device(wb, host_array_ptr(qweight_b), runtime=runtime)

        gguf_q8_0_gemv_f32_f32_out(
            xd.ptr,
            wa.ptr,
            single_a.ptr,
            rows,
            in_features,
            out_features,
            library=library,
            runtime=runtime,
        )
        gguf_q8_0_gemv_f32_f32_out(
            xd.ptr,
            wb.ptr,
            single_b.ptr,
            rows,
            in_features,
            out_features,
            library=library,
            runtime=runtime,
        )
        gguf_q8_0_dual_gemv_f32_f32_out(
            xd.ptr,
            wa.ptr,
            wb.ptr,
            dual_a.ptr,
            dual_b.ptr,
            rows,
            in_features,
            out_features,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()

        got_single_a = np.empty((rows, out_features), dtype=np.float32)
        got_single_b = np.empty((rows, out_features), dtype=np.float32)
        got_dual_a = np.empty((rows, out_features), dtype=np.float32)
        got_dual_b = np.empty((rows, out_features), dtype=np.float32)
        copy_device_to_host(host_array_ptr(got_single_a), single_a, runtime=runtime)
        copy_device_to_host(host_array_ptr(got_single_b), single_b, runtime=runtime)
        copy_device_to_host(host_array_ptr(got_dual_a), dual_a, runtime=runtime)
        copy_device_to_host(host_array_ptr(got_dual_b), dual_b, runtime=runtime)
    finally:
        for buf in reversed(bufs):
            free(buf, runtime=runtime)

    np.testing.assert_array_equal(got_dual_a, got_single_a)
    np.testing.assert_array_equal(got_dual_b, got_single_b)
