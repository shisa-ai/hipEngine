"""Correctness fixtures for GGUF Q8_0 T16 GEMV decode (P9.H3c)."""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

from hipengine.core.memory import copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc
from hipengine.kernels.cpu_reference import gguf_quant_gemv
from hipengine.kernels.hip_gfx1100.quant.gguf_q8_0_t16_gemv import (
    build_gguf_q8_0_t16_gemv,
    gguf_q8_0_t16_dual_gate_up_gemv_decode_bf16_bf16_out,
    gguf_q8_0_t16_dual_gate_up_gemv_decode_fp16_fp16_out,
    gguf_q8_0_t16_dual_gemv_decode_bf16_bf16_out,
    gguf_q8_0_t16_dual_gemv_decode_fp16_fp16_out,
    gguf_q8_0_t16_dual_gemv_decode_q8_1_dp4a_bf16_bf16_out,
    gguf_q8_0_t16_dual_gemv_decode_rowtile2_bf16_bf16_out,
    gguf_q8_0_t16_dual_gemv_decode_rowtile4_bf16_bf16_out,
    gguf_q8_0_t16_gemv_decode_bf16_bf16_out,
    gguf_q8_0_t16_gemv_decode_f32_bf16_out,
    gguf_q8_0_t16_gemv_decode_fp16_fp16_out,
    gguf_q8_0_t16_triple_gemv_decode_bf16_bf16_out,
    gguf_q8_0_t16_triple_gemv_decode_fp16_fp16_out,
    gguf_q8_0_t16_gemv_decode_rowtile4_bf16_bf16_out,
    gguf_q8_0_t16_triple_gemv_decode_rowtile4_bf16_bf16_out,
    plan_gguf_q8_0_t16_gemv_build,
    register_gguf_q8_0_t16_gemv_kernels,
)
from hipengine.kernels.registry import resolve
from hipengine.quant.gguf import GGMLQuantizationType
from hipengine.quant.gguf_t16 import repack_gguf_q8_0_tile16
from tests._gguf_synthetic_weights import make_q8_0_weight

_Q8_1_BLOCK = 32
_Q8_0_RAW_BYTES = 34
_Q8_1_BLOCK_BYTES = 36


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


HIP_AVAILABLE = _hip_available()


@pytest.fixture(scope="module")
def q8_t16_library():
    if not HIP_AVAILABLE:
        pytest.skip("HIP runtime is not available")
    return build_gguf_q8_0_t16_gemv(load=True)


def _f32_to_bf16_u16(arr: np.ndarray) -> np.ndarray:
    f32 = np.ascontiguousarray(arr, dtype=np.float32)
    u32 = f32.view(np.uint32).copy()
    nan_mask = np.isnan(f32)
    lsb = (u32 >> 16) & 1
    rounded = ((u32 + 0x7FFF + lsb) >> 16).astype(np.uint16)
    rounded[nan_mask] = 0x7FC0
    return rounded.reshape(f32.shape)


def _bf16_u16_to_f32(arr: np.ndarray) -> np.ndarray:
    u16 = np.ascontiguousarray(arr, dtype=np.uint16)
    return (u16.astype(np.uint32) << 16).view(np.float32).reshape(u16.shape).copy()


def _quantize_q8_1_cpu(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rows, in_features = x.shape
    blocks = in_features // _Q8_1_BLOCK
    q = np.empty((rows, blocks, _Q8_1_BLOCK), dtype=np.int8)
    d = np.empty((rows, blocks), dtype=np.float32)
    for row in range(rows):
        for block in range(blocks):
            chunk = x[row, block * _Q8_1_BLOCK : (block + 1) * _Q8_1_BLOCK].astype(np.float32)
            amax = float(np.max(np.abs(chunk)))
            scale = 0.0 if amax == 0.0 else amax / 127.0
            d[row, block] = np.float16(scale).astype(np.float32)
            q[row, block] = 0 if amax == 0.0 else np.clip(np.round(chunk / scale), -128, 127).astype(np.int8)
    return q, d


def _q8_0_q8_1_oracle(x_bf16: np.ndarray, weight: np.ndarray) -> np.ndarray:
    rows, in_features = x_bf16.shape
    out_features = weight.shape[0]
    blocks = in_features // _Q8_1_BLOCK
    q8, d8 = _quantize_q8_1_cpu(x_bf16)
    out = np.zeros((rows, out_features), dtype=np.float32)
    for col in range(out_features):
        row_bytes = weight[col]
        for block in range(blocks):
            blk = row_bytes[block * _Q8_0_RAW_BYTES : (block + 1) * _Q8_0_RAW_BYTES]
            wd = blk[0:2].view(np.float16).astype(np.float32)[0]
            wq = blk[2:34].view(np.int8).astype(np.int32)
            dots = (q8[:, block, :].astype(np.int32) * wq[None, :]).sum(axis=1)
            out[:, col] += d8[:, block] * float(wd) * dots.astype(np.float32)
    return out


def _softmax_kl(ref: np.ndarray, cand: np.ndarray) -> np.ndarray:
    def logsm(z):
        z = z.astype(np.float64)
        s = z - z.max(axis=-1, keepdims=True)
        return s - np.log(np.exp(s).sum(axis=-1, keepdims=True))

    lr, lc = logsm(ref), logsm(cand)
    return np.sum(np.exp(lr) * (lr - lc), axis=-1)


def _run_single(fn, x, tiles, rows, in_features, out_features, out_dtype, library, *, threads: int = 0):
    x_buf = malloc(x.nbytes)
    copy_host_to_device(x_buf, host_array_ptr(x), x.nbytes)
    w_buf = malloc(tiles.nbytes)
    copy_host_to_device(w_buf, host_array_ptr(tiles), tiles.nbytes)
    out_arr = np.zeros((rows, out_features), dtype=out_dtype)
    out_buf = malloc(out_arr.nbytes)
    try:
        fn(x_buf.ptr, w_buf.ptr, out_buf.ptr, rows, in_features, out_features, library=library, threads=threads)
        copy_device_to_host(host_array_ptr(out_arr), out_buf, out_arr.nbytes)
        return out_arr
    finally:
        for b in (x_buf, w_buf, out_buf):
            free(b)


def _run_dual(fn, x, tiles_a, tiles_b, rows, in_features, oa, ob, out_dtype, library, *, threads: int = 0):
    x_buf = malloc(x.nbytes)
    copy_host_to_device(x_buf, host_array_ptr(x), x.nbytes)
    a_buf = malloc(tiles_a.nbytes)
    copy_host_to_device(a_buf, host_array_ptr(tiles_a), tiles_a.nbytes)
    b_buf = malloc(tiles_b.nbytes)
    copy_host_to_device(b_buf, host_array_ptr(tiles_b), tiles_b.nbytes)
    out_arr = np.zeros((rows, oa + ob), dtype=out_dtype)
    out_buf = malloc(out_arr.nbytes)
    try:
        fn(x_buf.ptr, a_buf.ptr, b_buf.ptr, out_buf.ptr, rows, in_features, oa, ob, library=library, threads=threads)
        copy_device_to_host(host_array_ptr(out_arr), out_buf, out_arr.nbytes)
        return out_arr
    finally:
        for b in (x_buf, a_buf, b_buf, out_buf):
            free(b)


def _run_dual_split(fn, x, tiles_a, tiles_b, rows, in_features, oa, ob, out_dtype, library, *, threads: int = 0):
    x_buf = malloc(x.nbytes)
    copy_host_to_device(x_buf, host_array_ptr(x), x.nbytes)
    a_buf = malloc(tiles_a.nbytes)
    copy_host_to_device(a_buf, host_array_ptr(tiles_a), tiles_a.nbytes)
    b_buf = malloc(tiles_b.nbytes)
    copy_host_to_device(b_buf, host_array_ptr(tiles_b), tiles_b.nbytes)
    out_a = np.zeros((rows, oa), dtype=out_dtype)
    out_b = np.zeros((rows, ob), dtype=out_dtype)
    a_out_buf = malloc(out_a.nbytes)
    b_out_buf = malloc(out_b.nbytes)
    try:
        fn(
            x_buf.ptr,
            a_buf.ptr,
            b_buf.ptr,
            a_out_buf.ptr,
            b_out_buf.ptr,
            rows,
            in_features,
            oa,
            ob,
            library=library,
            threads=threads,
        )
        copy_device_to_host(host_array_ptr(out_a), a_out_buf, out_a.nbytes)
        copy_device_to_host(host_array_ptr(out_b), b_out_buf, out_b.nbytes)
        return out_a, out_b
    finally:
        for b in (x_buf, a_buf, b_buf, a_out_buf, b_out_buf):
            free(b)


def _run_dual_split_q8_1(fn, x_bits, tiles_a, tiles_b, rows, in_features, oa, ob, out_dtype, library, q4_library, *, threads: int = 0):
    from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_gemv import gguf_q4_k_quantize_bf16_q8_1

    blocks = in_features // _Q8_1_BLOCK
    x_buf = malloc(x_bits.nbytes)
    copy_host_to_device(x_buf, host_array_ptr(x_bits), x_bits.nbytes)
    xq_buf = malloc(rows * blocks * _Q8_1_BLOCK_BYTES)
    a_buf = malloc(tiles_a.nbytes)
    copy_host_to_device(a_buf, host_array_ptr(tiles_a), tiles_a.nbytes)
    b_buf = malloc(tiles_b.nbytes)
    copy_host_to_device(b_buf, host_array_ptr(tiles_b), tiles_b.nbytes)
    out_a = np.zeros((rows, oa), dtype=out_dtype)
    out_b = np.zeros((rows, ob), dtype=out_dtype)
    a_out_buf = malloc(out_a.nbytes)
    b_out_buf = malloc(out_b.nbytes)
    try:
        gguf_q4_k_quantize_bf16_q8_1(x_buf.ptr, xq_buf.ptr, rows, in_features, library=q4_library)
        fn(
            xq_buf.ptr,
            a_buf.ptr,
            b_buf.ptr,
            a_out_buf.ptr,
            b_out_buf.ptr,
            rows,
            in_features,
            oa,
            ob,
            library=library,
            threads=threads,
        )
        copy_device_to_host(host_array_ptr(out_a), a_out_buf, out_a.nbytes)
        copy_device_to_host(host_array_ptr(out_b), b_out_buf, out_b.nbytes)
        return out_a, out_b
    finally:
        for b in (x_buf, xq_buf, a_buf, b_buf, a_out_buf, b_out_buf):
            free(b)


def _run_triple_split(fn, x, tiles_a, tiles_b, tiles_c, rows, in_features, oa, ob, oc, out_dtype, library, *, threads: int = 0):
    x_buf = malloc(x.nbytes)
    copy_host_to_device(x_buf, host_array_ptr(x), x.nbytes)
    a_buf = malloc(tiles_a.nbytes)
    copy_host_to_device(a_buf, host_array_ptr(tiles_a), tiles_a.nbytes)
    b_buf = malloc(tiles_b.nbytes)
    copy_host_to_device(b_buf, host_array_ptr(tiles_b), tiles_b.nbytes)
    c_buf = malloc(tiles_c.nbytes)
    copy_host_to_device(c_buf, host_array_ptr(tiles_c), tiles_c.nbytes)
    out_a = np.zeros((rows, oa), dtype=out_dtype)
    out_b = np.zeros((rows, ob), dtype=out_dtype)
    out_c = np.zeros((rows, oc), dtype=out_dtype)
    a_out_buf = malloc(out_a.nbytes)
    b_out_buf = malloc(out_b.nbytes)
    c_out_buf = malloc(out_c.nbytes)
    try:
        fn(
            x_buf.ptr,
            a_buf.ptr,
            b_buf.ptr,
            c_buf.ptr,
            a_out_buf.ptr,
            b_out_buf.ptr,
            c_out_buf.ptr,
            rows,
            in_features,
            oa,
            ob,
            oc,
            library=library,
            threads=threads,
        )
        copy_device_to_host(host_array_ptr(out_a), a_out_buf, out_a.nbytes)
        copy_device_to_host(host_array_ptr(out_b), b_out_buf, out_b.nbytes)
        copy_device_to_host(host_array_ptr(out_c), c_out_buf, out_c.nbytes)
        return out_a, out_b, out_c
    finally:
        for b in (x_buf, a_buf, b_buf, c_buf, a_out_buf, b_out_buf, c_out_buf):
            free(b)


_TOL = dict(atol=5.0e-4, rtol=5.0e-3)


def test_p9_h3c_registry_keys_resolve() -> None:
    register_gguf_q8_0_t16_gemv_kernels()
    for variant in (
        "t16_gemv_decode_bf16_bf16_out",
        "t16_gemv_decode_rowtile4_bf16_bf16_out",
        "t16_gemv_decode_f32_bf16_out",
        "t16_gemv_decode_fp16_fp16_out",
        "t16_dual_gate_up_gemv_decode_bf16_bf16_out",
        "t16_dual_gate_up_gemv_decode_fp16_fp16_out",
        "t16_dual_gemv_decode_bf16_bf16_out",
        "t16_dual_gemv_decode_fp16_fp16_out",
        "t16_dual_gemv_decode_rowtile2_bf16_bf16_out",
        "t16_dual_gemv_decode_rowtile4_bf16_bf16_out",
        "t16_dual_gemv_decode_q8_1_dp4a_bf16_bf16_out",
        "t16_triple_gemv_decode_bf16_bf16_out",
        "t16_triple_gemv_decode_rowtile4_bf16_bf16_out",
        "t16_triple_gemv_decode_fp16_fp16_out",
    ):
        assert resolve(
            backend="hip_gfx1100",
            layer="linear",
            quant="gguf_q8_0_t16_v1",
            variant=variant,
        ) is not None


def test_p9_h3c_build_plan_is_dry_run_safe() -> None:
    plan = plan_gguf_q8_0_t16_gemv_build()
    assert plan.output_path.name == "gguf_q8_0_t16_gemv.so"
    assert plan.sources[0].name == "gguf_q8_0_t16_gemv.hip"


def test_p9_h3c_wrappers_validate_args() -> None:
    with pytest.raises(ValueError, match="block size 32"):
        gguf_q8_0_t16_gemv_decode_bf16_bf16_out(0, 0, 0, 1, 31, 16)
    with pytest.raises(ValueError, match="multiple of 16"):
        gguf_q8_0_t16_gemv_decode_bf16_bf16_out(0, 0, 0, 1, 32, 8)
    with pytest.raises(ValueError, match="block size 32"):
        gguf_q8_0_t16_gemv_decode_f32_bf16_out(0, 0, 0, 1, 31, 16)
    with pytest.raises(ValueError, match="rows must be positive"):
        gguf_q8_0_t16_gemv_decode_bf16_bf16_out(0, 0, 0, 0, 32, 16)
    with pytest.raises(ValueError, match="multiples of 16"):
        gguf_q8_0_t16_dual_gate_up_gemv_decode_bf16_bf16_out(0, 0, 0, 0, 1, 32, 16, 8)
    with pytest.raises(ValueError, match="multiples of 16"):
        gguf_q8_0_t16_dual_gemv_decode_bf16_bf16_out(0, 0, 0, 0, 0, 1, 32, 16, 8)
    with pytest.raises(ValueError, match="multiples of 16"):
        gguf_q8_0_t16_triple_gemv_decode_bf16_bf16_out(0, 0, 0, 0, 0, 0, 0, 1, 32, 16, 16, 8)
    with pytest.raises(ValueError, match="threads must be one of 64, 128"):
        gguf_q8_0_t16_gemv_decode_bf16_bf16_out(0, 0, 0, 1, 32, 16, threads=256)


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
@pytest.mark.parametrize(
    "rows,in_features,out_features",
    [
        (1, 32, 16),
        (1, 256, 256),
        (1, 2048, 512),
        (4, 512, 256),
    ],
)
def test_p9_h3c_single_bf16_bf16_matches_cpu_oracle(rows, in_features, out_features, q8_t16_library) -> None:
    rng = np.random.default_rng(rows * 11 + in_features + out_features)
    qweight = make_q8_0_weight(out_features, in_features)
    tiles = repack_gguf_q8_0_tile16(qweight).tiles
    x = rng.normal(0.0, 0.3, size=(rows, in_features)).astype(np.float32)
    x_bf16 = _f32_to_bf16_u16(x)
    x_ref = _bf16_u16_to_f32(x_bf16)

    actual = _run_single(
        gguf_q8_0_t16_gemv_decode_bf16_bf16_out,
        x_bf16,
        tiles,
        rows,
        in_features,
        out_features,
        np.uint16,
        q8_t16_library,
    )

    expected = gguf_quant_gemv(x_ref, qweight, GGMLQuantizationType.Q8_0)
    expected_bf16 = _bf16_u16_to_f32(_f32_to_bf16_u16(expected))
    np.testing.assert_allclose(_bf16_u16_to_f32(actual), expected_bf16, **_TOL)


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
@pytest.mark.parametrize(
    "rows,in_features,out_features",
    [
        (3, 512, 128),
        (5, 512, 256),
        (3, 2048, 4096),
    ],
)
def test_q8_t16_single_rowtile4_matches_exact_single(rows, in_features, out_features, q8_t16_library) -> None:
    rng = np.random.default_rng(19100 + rows + in_features + out_features)
    qweight = make_q8_0_weight(out_features, in_features)
    tiles = repack_gguf_q8_0_tile16(qweight).tiles
    x = rng.normal(0.0, 0.3, size=(rows, in_features)).astype(np.float32)
    x_bf16 = _f32_to_bf16_u16(x)

    actual = _run_single(
        gguf_q8_0_t16_gemv_decode_rowtile4_bf16_bf16_out,
        x_bf16,
        tiles,
        rows,
        in_features,
        out_features,
        np.uint16,
        q8_t16_library,
        threads=64,
    )
    expected = _run_single(
        gguf_q8_0_t16_gemv_decode_bf16_bf16_out,
        x_bf16,
        tiles,
        rows,
        in_features,
        out_features,
        np.uint16,
        q8_t16_library,
        threads=128,
    )

    np.testing.assert_array_equal(actual, expected)


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
@pytest.mark.parametrize(
    "rows,in_features,out_features",
    [
        (1, 256, 256),
        (1, 2048, 512),
    ],
)
def test_p9_h3c_single_f32_bf16_matches_cpu_oracle(rows, in_features, out_features, q8_t16_library) -> None:
    rng = np.random.default_rng(rows * 19 + in_features + out_features)
    qweight = make_q8_0_weight(out_features, in_features)
    tiles = repack_gguf_q8_0_tile16(qweight).tiles
    x = rng.normal(0.0, 0.3, size=(rows, in_features)).astype(np.float32)

    actual = _run_single(
        gguf_q8_0_t16_gemv_decode_f32_bf16_out,
        x,
        tiles,
        rows,
        in_features,
        out_features,
        np.uint16,
        q8_t16_library,
    )

    expected = gguf_quant_gemv(x, qweight, GGMLQuantizationType.Q8_0)
    expected_bf16 = _bf16_u16_to_f32(_f32_to_bf16_u16(expected))
    np.testing.assert_allclose(_bf16_u16_to_f32(actual), expected_bf16, **_TOL)


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
@pytest.mark.parametrize(
    "rows,in_features,out_features",
    [
        (1, 256, 256),
        (4, 1024, 512),
    ],
)
def test_p9_h3c_single_fp16_fp16_matches_cpu_oracle(rows, in_features, out_features, q8_t16_library) -> None:
    rng = np.random.default_rng(rows * 17 + in_features + out_features)
    qweight = make_q8_0_weight(out_features, in_features)
    tiles = repack_gguf_q8_0_tile16(qweight).tiles
    x_f16 = rng.normal(0.0, 0.3, size=(rows, in_features)).astype(np.float16)

    actual = _run_single(
        gguf_q8_0_t16_gemv_decode_fp16_fp16_out,
        x_f16,
        tiles,
        rows,
        in_features,
        out_features,
        np.float16,
        q8_t16_library,
    )

    expected = gguf_quant_gemv(x_f16.astype(np.float32), qweight, GGMLQuantizationType.Q8_0)
    np.testing.assert_allclose(actual.astype(np.float32), expected.astype(np.float16).astype(np.float32), **_TOL)


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
@pytest.mark.parametrize(
    "rows,in_features,out_features_a,out_features_b",
    [
        (1, 256, 16, 32),
        (1, 2048, 512, 512),
        (4, 512, 256, 256),
    ],
)
def test_p9_h3c_dual_bf16_bf16_matches_cpu_oracle(
    rows, in_features, out_features_a, out_features_b, q8_t16_library,
) -> None:
    rng = np.random.default_rng(rows * 19 + in_features + out_features_a + out_features_b)
    qa = make_q8_0_weight(out_features_a, in_features)
    qb = make_q8_0_weight(out_features_b, in_features)
    ta = repack_gguf_q8_0_tile16(qa).tiles
    tb = repack_gguf_q8_0_tile16(qb).tiles
    x = rng.normal(0.0, 0.3, size=(rows, in_features)).astype(np.float32)
    x_bf16 = _f32_to_bf16_u16(x)
    x_ref = _bf16_u16_to_f32(x_bf16)

    actual = _run_dual(
        gguf_q8_0_t16_dual_gate_up_gemv_decode_bf16_bf16_out,
        x_bf16,
        ta,
        tb,
        rows,
        in_features,
        out_features_a,
        out_features_b,
        np.uint16,
        q8_t16_library,
    )

    expected = np.zeros((rows, out_features_a + out_features_b), dtype=np.float32)
    expected[:, :out_features_a] = gguf_quant_gemv(x_ref, qa, GGMLQuantizationType.Q8_0)
    expected[:, out_features_a:] = gguf_quant_gemv(x_ref, qb, GGMLQuantizationType.Q8_0)
    expected_bf16 = _bf16_u16_to_f32(_f32_to_bf16_u16(expected))
    np.testing.assert_allclose(_bf16_u16_to_f32(actual), expected_bf16, **_TOL)


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
@pytest.mark.parametrize(
    "rows,in_features,out_features_a,out_features_b",
    [
        (1, 256, 16, 32),
        (1, 2048, 512, 512),
        (3, 512, 64, 128),
    ],
)
def test_p9_d6_dual_split_bf16_bf16_matches_cpu_oracle(
    rows, in_features, out_features_a, out_features_b, q8_t16_library,
) -> None:
    rng = np.random.default_rng(rows * 29 + in_features + out_features_a + out_features_b)
    qa = make_q8_0_weight(out_features_a, in_features)
    qb = make_q8_0_weight(out_features_b, in_features)
    ta = repack_gguf_q8_0_tile16(qa).tiles
    tb = repack_gguf_q8_0_tile16(qb).tiles
    x = rng.normal(0.0, 0.3, size=(rows, in_features)).astype(np.float32)
    x_bf16 = _f32_to_bf16_u16(x)
    x_ref = _bf16_u16_to_f32(x_bf16)

    actual_a, actual_b = _run_dual_split(
        gguf_q8_0_t16_dual_gemv_decode_bf16_bf16_out,
        x_bf16,
        ta,
        tb,
        rows,
        in_features,
        out_features_a,
        out_features_b,
        np.uint16,
        q8_t16_library,
    )

    expected_a = gguf_quant_gemv(x_ref, qa, GGMLQuantizationType.Q8_0)
    expected_b = gguf_quant_gemv(x_ref, qb, GGMLQuantizationType.Q8_0)
    expected_a_bf16 = _bf16_u16_to_f32(_f32_to_bf16_u16(expected_a))
    expected_b_bf16 = _bf16_u16_to_f32(_f32_to_bf16_u16(expected_b))
    np.testing.assert_allclose(_bf16_u16_to_f32(actual_a), expected_a_bf16, **_TOL)
    np.testing.assert_allclose(_bf16_u16_to_f32(actual_b), expected_b_bf16, **_TOL)


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
@pytest.mark.parametrize("threads", [64, 128])
def test_q8_t16_dual_split_thread_override_matches_cpu_oracle(threads: int, q8_t16_library) -> None:
    rows, in_features, out_features_a, out_features_b = 3, 512, 64, 128
    rng = np.random.default_rng(5600 + threads)
    qa = make_q8_0_weight(out_features_a, in_features)
    qb = make_q8_0_weight(out_features_b, in_features)
    ta = repack_gguf_q8_0_tile16(qa).tiles
    tb = repack_gguf_q8_0_tile16(qb).tiles
    x = rng.normal(0.0, 0.3, size=(rows, in_features)).astype(np.float32)
    x_bf16 = _f32_to_bf16_u16(x)
    x_ref = _bf16_u16_to_f32(x_bf16)

    actual_a, actual_b = _run_dual_split(
        gguf_q8_0_t16_dual_gemv_decode_bf16_bf16_out,
        x_bf16,
        ta,
        tb,
        rows,
        in_features,
        out_features_a,
        out_features_b,
        np.uint16,
        q8_t16_library,
        threads=threads,
    )

    expected_a = gguf_quant_gemv(x_ref, qa, GGMLQuantizationType.Q8_0)
    expected_b = gguf_quant_gemv(x_ref, qb, GGMLQuantizationType.Q8_0)
    expected_a_bf16 = _bf16_u16_to_f32(_f32_to_bf16_u16(expected_a))
    expected_b_bf16 = _bf16_u16_to_f32(_f32_to_bf16_u16(expected_b))
    np.testing.assert_allclose(_bf16_u16_to_f32(actual_a), expected_a_bf16, **_TOL)
    np.testing.assert_allclose(_bf16_u16_to_f32(actual_b), expected_b_bf16, **_TOL)


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_q8_t16_dual_split_q8_1_dp4a_matches_oracle_and_quality_gate(q8_t16_library) -> None:
    from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_gemv import build_gguf_q4_k_gemv

    rows, in_features, out_features_a, out_features_b = 8, 512, 64, 128
    rng = np.random.default_rng(8817)
    qa = make_q8_0_weight(out_features_a, in_features)
    qb = make_q8_0_weight(out_features_b, in_features)
    ta = repack_gguf_q8_0_tile16(qa).tiles
    tb = repack_gguf_q8_0_tile16(qb).tiles
    x = rng.normal(0.0, 0.3, size=(rows, in_features)).astype(np.float32)
    x_bits = _f32_to_bf16_u16(x)
    x_ref = _bf16_u16_to_f32(x_bits)
    q4_library = build_gguf_q4_k_gemv(load=True)

    actual_a, actual_b = _run_dual_split_q8_1(
        gguf_q8_0_t16_dual_gemv_decode_q8_1_dp4a_bf16_bf16_out,
        x_bits,
        ta,
        tb,
        rows,
        in_features,
        out_features_a,
        out_features_b,
        np.uint16,
        q8_t16_library,
        q4_library,
    )
    actual = np.concatenate([_bf16_u16_to_f32(actual_a), _bf16_u16_to_f32(actual_b)], axis=1)

    q8_ref_a = _q8_0_q8_1_oracle(x_ref, qa)
    q8_ref_b = _q8_0_q8_1_oracle(x_ref, qb)
    q8_ref = np.concatenate([q8_ref_a, q8_ref_b], axis=1)
    rel_l2 = float(np.linalg.norm(actual - q8_ref) / (np.linalg.norm(q8_ref) + 1e-8))
    assert rel_l2 <= 2e-2, f"kernel vs q8_1 oracle rel_l2={rel_l2}"

    full_ref_a = gguf_quant_gemv(x_ref, qa, GGMLQuantizationType.Q8_0)
    full_ref_b = gguf_quant_gemv(x_ref, qb, GGMLQuantizationType.Q8_0)
    full_ref = np.concatenate([full_ref_a, full_ref_b], axis=1)
    kl = _softmax_kl(full_ref, actual)
    assert float(np.mean(kl)) <= 0.05, f"KL={float(np.mean(kl))}"
    top1 = float(np.mean(np.argmax(full_ref, axis=-1) == np.argmax(actual, axis=-1)))
    assert top1 >= 0.90, f"top1={top1}"


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
@pytest.mark.parametrize(
    "fn,rows",
    [
        (gguf_q8_0_t16_dual_gemv_decode_rowtile2_bf16_bf16_out, 3),
        (gguf_q8_0_t16_dual_gemv_decode_rowtile4_bf16_bf16_out, 5),
    ],
)
def test_q8_t16_dual_split_rowtile_matches_exact_pair(fn, rows: int, q8_t16_library) -> None:
    rows, in_features, out_features_a, out_features_b = rows, 512, 64, 128
    rng = np.random.default_rng(9100 + rows)
    qa = make_q8_0_weight(out_features_a, in_features)
    qb = make_q8_0_weight(out_features_b, in_features)
    ta = repack_gguf_q8_0_tile16(qa).tiles
    tb = repack_gguf_q8_0_tile16(qb).tiles
    x = rng.normal(0.0, 0.3, size=(rows, in_features)).astype(np.float32)
    x_bf16 = _f32_to_bf16_u16(x)

    actual_a, actual_b = _run_dual_split(
        fn,
        x_bf16,
        ta,
        tb,
        rows,
        in_features,
        out_features_a,
        out_features_b,
        np.uint16,
        q8_t16_library,
    )
    expected_a, expected_b = _run_dual_split(
        gguf_q8_0_t16_dual_gemv_decode_bf16_bf16_out,
        x_bf16,
        ta,
        tb,
        rows,
        in_features,
        out_features_a,
        out_features_b,
        np.uint16,
        q8_t16_library,
    )

    np.testing.assert_array_equal(actual_a, expected_a)
    np.testing.assert_array_equal(actual_b, expected_b)


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_q8_t16_dual_split_rowtile4_matches_exact_qwen35_attention_pair_shape(q8_t16_library) -> None:
    rows, in_features, out_features_a, out_features_b = 3, 2048, 8192, 4096
    rng = np.random.default_rng(9173)
    qa = make_q8_0_weight(out_features_a, in_features)
    qb = make_q8_0_weight(out_features_b, in_features)
    ta = repack_gguf_q8_0_tile16(qa).tiles
    tb = repack_gguf_q8_0_tile16(qb).tiles
    x = rng.normal(0.0, 0.3, size=(rows, in_features)).astype(np.float32)
    x_bf16 = _f32_to_bf16_u16(x)

    actual_a, actual_b = _run_dual_split(
        gguf_q8_0_t16_dual_gemv_decode_rowtile4_bf16_bf16_out,
        x_bf16,
        ta,
        tb,
        rows,
        in_features,
        out_features_a,
        out_features_b,
        np.uint16,
        q8_t16_library,
        threads=64,
    )
    expected_a, expected_b = _run_dual_split(
        gguf_q8_0_t16_dual_gemv_decode_bf16_bf16_out,
        x_bf16,
        ta,
        tb,
        rows,
        in_features,
        out_features_a,
        out_features_b,
        np.uint16,
        q8_t16_library,
        threads=128,
    )

    np.testing.assert_array_equal(actual_a, expected_a)
    np.testing.assert_array_equal(actual_b, expected_b)


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
@pytest.mark.parametrize(
    "rows,in_features,out_features_a,out_features_b,out_features_c",
    [
        (3, 512, 64, 128, 32),
        (5, 512, 64, 128, 32),
        (3, 2048, 4096, 1024, 1024),
    ],
)
def test_q8_t16_triple_rowtile4_matches_exact_triple(
    rows, in_features, out_features_a, out_features_b, out_features_c, q8_t16_library,
) -> None:
    rng = np.random.default_rng(19600 + rows + in_features + out_features_a + out_features_b + out_features_c)
    qa = make_q8_0_weight(out_features_a, in_features)
    qb = make_q8_0_weight(out_features_b, in_features)
    qc = make_q8_0_weight(out_features_c, in_features)
    ta = repack_gguf_q8_0_tile16(qa).tiles
    tb = repack_gguf_q8_0_tile16(qb).tiles
    tc = repack_gguf_q8_0_tile16(qc).tiles
    x = rng.normal(0.0, 0.3, size=(rows, in_features)).astype(np.float32)
    x_bf16 = _f32_to_bf16_u16(x)

    actual_a, actual_b, actual_c = _run_triple_split(
        gguf_q8_0_t16_triple_gemv_decode_rowtile4_bf16_bf16_out,
        x_bf16,
        ta,
        tb,
        tc,
        rows,
        in_features,
        out_features_a,
        out_features_b,
        out_features_c,
        np.uint16,
        q8_t16_library,
        threads=64,
    )
    expected_a, expected_b, expected_c = _run_triple_split(
        gguf_q8_0_t16_triple_gemv_decode_bf16_bf16_out,
        x_bf16,
        ta,
        tb,
        tc,
        rows,
        in_features,
        out_features_a,
        out_features_b,
        out_features_c,
        np.uint16,
        q8_t16_library,
        threads=128,
    )

    np.testing.assert_array_equal(actual_a, expected_a)
    np.testing.assert_array_equal(actual_b, expected_b)
    np.testing.assert_array_equal(actual_c, expected_c)


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_p10_x1_dual_split_matches_single_kernels_for_qwen35_linear_attention_shape(q8_t16_library) -> None:
    """Regression for P10.X1: fused Q8T16 attn_qkv+attn_gate must be bit-stable.

    Qwen3.6-35B-A3B linear-attention decode fuses Q8_0 T16 projections with
    asymmetric outputs (8192 qkv + 4096 gate).  The fused split kernel must use
    the same reduction geometry as the single-output kernels; otherwise tiny
    BF16 differences can flip later MoE routing and fail the public smoke.
    """

    rows, in_features, out_features_a, out_features_b = 1, 2048, 8192, 4096
    rng = np.random.default_rng(10001)
    qa = make_q8_0_weight(out_features_a, in_features)
    qb = make_q8_0_weight(out_features_b, in_features)
    ta = repack_gguf_q8_0_tile16(qa).tiles
    tb = repack_gguf_q8_0_tile16(qb).tiles
    x = rng.normal(0.0, 0.3, size=(rows, in_features)).astype(np.float32)
    x_bf16 = _f32_to_bf16_u16(x)

    actual_a, actual_b = _run_dual_split(
        gguf_q8_0_t16_dual_gemv_decode_bf16_bf16_out,
        x_bf16,
        ta,
        tb,
        rows,
        in_features,
        out_features_a,
        out_features_b,
        np.uint16,
        q8_t16_library,
    )
    expected_a = _run_single(
        gguf_q8_0_t16_gemv_decode_bf16_bf16_out,
        x_bf16,
        ta,
        rows,
        in_features,
        out_features_a,
        np.uint16,
        q8_t16_library,
    )
    expected_b = _run_single(
        gguf_q8_0_t16_gemv_decode_bf16_bf16_out,
        x_bf16,
        tb,
        rows,
        in_features,
        out_features_b,
        np.uint16,
        q8_t16_library,
    )

    np.testing.assert_array_equal(actual_a, expected_a)
    np.testing.assert_array_equal(actual_b, expected_b)


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
@pytest.mark.parametrize(
    "rows,in_features,out_features_a,out_features_b,out_features_c",
    [
        (1, 256, 32, 16, 48),
        (1, 2048, 1024, 512, 512),
        (3, 512, 64, 128, 32),
    ],
)
def test_p9_d9_triple_split_bf16_bf16_matches_cpu_oracle(
    rows, in_features, out_features_a, out_features_b, out_features_c, q8_t16_library,
) -> None:
    rng = np.random.default_rng(rows * 31 + in_features + out_features_a + out_features_b + out_features_c)
    qa = make_q8_0_weight(out_features_a, in_features)
    qb = make_q8_0_weight(out_features_b, in_features)
    qc = make_q8_0_weight(out_features_c, in_features)
    ta = repack_gguf_q8_0_tile16(qa).tiles
    tb = repack_gguf_q8_0_tile16(qb).tiles
    tc = repack_gguf_q8_0_tile16(qc).tiles
    x = rng.normal(0.0, 0.3, size=(rows, in_features)).astype(np.float32)
    x_bf16 = _f32_to_bf16_u16(x)
    x_ref = _bf16_u16_to_f32(x_bf16)

    actual_a, actual_b, actual_c = _run_triple_split(
        gguf_q8_0_t16_triple_gemv_decode_bf16_bf16_out,
        x_bf16,
        ta,
        tb,
        tc,
        rows,
        in_features,
        out_features_a,
        out_features_b,
        out_features_c,
        np.uint16,
        q8_t16_library,
    )

    expected_a = gguf_quant_gemv(x_ref, qa, GGMLQuantizationType.Q8_0)
    expected_b = gguf_quant_gemv(x_ref, qb, GGMLQuantizationType.Q8_0)
    expected_c = gguf_quant_gemv(x_ref, qc, GGMLQuantizationType.Q8_0)
    expected_a_bf16 = _bf16_u16_to_f32(_f32_to_bf16_u16(expected_a))
    expected_b_bf16 = _bf16_u16_to_f32(_f32_to_bf16_u16(expected_b))
    expected_c_bf16 = _bf16_u16_to_f32(_f32_to_bf16_u16(expected_c))
    np.testing.assert_allclose(_bf16_u16_to_f32(actual_a), expected_a_bf16, **_TOL)
    np.testing.assert_allclose(_bf16_u16_to_f32(actual_b), expected_b_bf16, **_TOL)
    np.testing.assert_allclose(_bf16_u16_to_f32(actual_c), expected_c_bf16, **_TOL)


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_p9_d9_triple_split_fp16_fp16_matches_cpu_oracle(q8_t16_library) -> None:
    rows, in_features, out_features_a, out_features_b, out_features_c = 2, 512, 64, 128, 32
    rng = np.random.default_rng(1229)
    qa = make_q8_0_weight(out_features_a, in_features)
    qb = make_q8_0_weight(out_features_b, in_features)
    qc = make_q8_0_weight(out_features_c, in_features)
    ta = repack_gguf_q8_0_tile16(qa).tiles
    tb = repack_gguf_q8_0_tile16(qb).tiles
    tc = repack_gguf_q8_0_tile16(qc).tiles
    x_f16 = rng.normal(0.0, 0.3, size=(rows, in_features)).astype(np.float16)

    actual_a, actual_b, actual_c = _run_triple_split(
        gguf_q8_0_t16_triple_gemv_decode_fp16_fp16_out,
        x_f16,
        ta,
        tb,
        tc,
        rows,
        in_features,
        out_features_a,
        out_features_b,
        out_features_c,
        np.float16,
        q8_t16_library,
    )

    expected_a = gguf_quant_gemv(x_f16.astype(np.float32), qa, GGMLQuantizationType.Q8_0)
    expected_b = gguf_quant_gemv(x_f16.astype(np.float32), qb, GGMLQuantizationType.Q8_0)
    expected_c = gguf_quant_gemv(x_f16.astype(np.float32), qc, GGMLQuantizationType.Q8_0)
    np.testing.assert_allclose(actual_a.astype(np.float32), expected_a.astype(np.float16).astype(np.float32), **_TOL)
    np.testing.assert_allclose(actual_b.astype(np.float32), expected_b.astype(np.float16).astype(np.float32), **_TOL)
    np.testing.assert_allclose(actual_c.astype(np.float32), expected_c.astype(np.float16).astype(np.float32), **_TOL)


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_p9_d6_dual_split_fp16_fp16_matches_cpu_oracle(q8_t16_library) -> None:
    rows, in_features, out_features_a, out_features_b = 2, 512, 64, 128
    rng = np.random.default_rng(1029)
    qa = make_q8_0_weight(out_features_a, in_features)
    qb = make_q8_0_weight(out_features_b, in_features)
    ta = repack_gguf_q8_0_tile16(qa).tiles
    tb = repack_gguf_q8_0_tile16(qb).tiles
    x_f16 = rng.normal(0.0, 0.3, size=(rows, in_features)).astype(np.float16)

    actual_a, actual_b = _run_dual_split(
        gguf_q8_0_t16_dual_gemv_decode_fp16_fp16_out,
        x_f16,
        ta,
        tb,
        rows,
        in_features,
        out_features_a,
        out_features_b,
        np.float16,
        q8_t16_library,
    )

    expected_a = gguf_quant_gemv(x_f16.astype(np.float32), qa, GGMLQuantizationType.Q8_0)
    expected_b = gguf_quant_gemv(x_f16.astype(np.float32), qb, GGMLQuantizationType.Q8_0)
    np.testing.assert_allclose(actual_a.astype(np.float32), expected_a.astype(np.float16).astype(np.float32), **_TOL)
    np.testing.assert_allclose(actual_b.astype(np.float32), expected_b.astype(np.float16).astype(np.float32), **_TOL)


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_p9_h3c_dual_fp16_fp16_matches_cpu_oracle(q8_t16_library) -> None:
    rows, in_features, out_features_a, out_features_b = 2, 512, 256, 256
    rng = np.random.default_rng(923)
    qa = make_q8_0_weight(out_features_a, in_features)
    qb = make_q8_0_weight(out_features_b, in_features)
    ta = repack_gguf_q8_0_tile16(qa).tiles
    tb = repack_gguf_q8_0_tile16(qb).tiles
    x_f16 = rng.normal(0.0, 0.3, size=(rows, in_features)).astype(np.float16)

    actual = _run_dual(
        gguf_q8_0_t16_dual_gate_up_gemv_decode_fp16_fp16_out,
        x_f16,
        ta,
        tb,
        rows,
        in_features,
        out_features_a,
        out_features_b,
        np.float16,
        q8_t16_library,
    )

    expected = np.zeros((rows, out_features_a + out_features_b), dtype=np.float32)
    expected[:, :out_features_a] = gguf_quant_gemv(x_f16.astype(np.float32), qa, GGMLQuantizationType.Q8_0)
    expected[:, out_features_a:] = gguf_quant_gemv(x_f16.astype(np.float32), qb, GGMLQuantizationType.Q8_0)
    expected_f16 = expected.astype(np.float16).astype(np.float32)
    np.testing.assert_allclose(actual.astype(np.float32), expected_f16, **_TOL)
