"""Dense bulk-prefill raw-Q4_K x DS4-Q8_1 integer MMQ leaf screen.

Screen for the 2026-09-08 engine-comparison PP8192 attribution follow-up
(worklog pp8192-gap-attribution-8bb338): the projection prefill owners
(float Q4T16 WMMA) hold ~63% of timed prefill, and the candidate arithmetic
class is nasone32-style raw-Q4_K x Q8_1-activation integer MMQ with
efa4e8641-family load reuse. The 2026-08-12 rejections bound the screen:
the T16-payload integer consumers lost (Q4 ~2.1x slower, Q5 +3.0-4.4%), so
these kernels consume RAW GGUF Q4_K bytes — a materially different
weight-consumer dataflow — in two classes:

  * mmq32: local128 workgroup, 32-column x 32-row output tile, per-256-block
    LDS staging (port of the retained raw Q5 MMQ32 C8 owner idiom).
  * wmma32: direct-global iu8 WMMA 16x16x16, 32-column x 16-row tiles (dense
    port of the 2026-06-16 selected ds4-wmma32 family winner).

Each class has ctl/vdr siblings with the SAME thread mapping, SAME integer
dots and SAME per-subblock f32 evaluation order; vdr only hoists loads (the
Q4_K block header d/dmin once per 256-block instead of once per subblock,
and reused activation fragments). The RED contract is therefore BIT-EXACT
ctl/vdr equality per class.

Activations are packed by the existing gguf_q8_1_mmq_ds4_pack_bf16 (one
residual pass: per-32 fp16 scale = maxabs/127 and fp16 sum = raw f32 sum;
the -dmin*min correction consumes the raw sum because x = scale*q8 exactly
up to quantization error). Both classes are additionally checked against a
CPU DS4+MMQ oracle and the outer KL <= 0.05 / top-1 >= 0.90 floor versus
the exact-weight reference.
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
from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_q8_1_selected_prefill import (
    build_gguf_q4_k_q8_1_selected_prefill,
    gguf_q8_1_mmq_ds4_pack_bf16,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_q8_1_mmq_prefill import (
    build_gguf_q4_k_q8_1_mmq_prefill,
    gguf_q4_k_q8_1_mmq32_ctl_dense_bf16_f32_out,
    gguf_q4_k_q8_1_mmq32_vdr_dense_bf16_f32_out,
    gguf_q4_k_q8_1_wmma32_ctl_dense_bf16_f32_out,
    gguf_q4_k_q8_1_wmma32_vdr_dense_bf16_f32_out,
)
from tests.test_gpu_gguf_q6_k_t16_planar_q8_1_gemv import HIP_AVAILABLE
from tests.test_gpu_gguf_x8_selected_gemv import (
    _bf16_bits,
    _bf16_to_f32,
    _exact_oracle,
    _softmax_kl,
    _top1,
    _weights,
)

QK_K = 256
Q4_K_BLOCK_BYTES = 144
DS4_BLOCK_BYTES = 144  # int8 qs[128] + 4 x (fp16 scale, fp16 sum)
FP16_INFO = np.finfo(np.float16)


@pytest.fixture(scope="module")
def pack_library():
    if not HIP_AVAILABLE:
        pytest.skip("HIP runtime is not available")
    return build_gguf_q4_k_q8_1_selected_prefill(load=True)


@pytest.fixture(scope="module")
def mmq_library():
    if not HIP_AVAILABLE:
        pytest.skip("HIP runtime is not available")
    return build_gguf_q4_k_q8_1_mmq_prefill(load=True)


def _f16(value: np.ndarray) -> np.ndarray:
    return np.asarray(value, dtype=np.float32).astype(np.float16).astype(
        np.float32
    )


def _ds4_pack_reference(x_f32: np.ndarray):
    """CPU port of gguf_q8_1_mmq_ds4_pack_bf16 (one residual pass).

    Returns int8 qs [rows, hidden] and per-32 fp16 (scale, sum) pairs as a
    [rows, hidden // 32, 2] f32 array (already fp16-rounded).
    """
    rows, hidden = x_f32.shape
    x = x_f32.astype(np.float32)
    groups = x.reshape(rows, hidden // 32, 32)
    max_abs = np.abs(groups).max(axis=2)
    scale = max_abs / np.float32(127.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.rint(groups / scale[:, :, None])
    q = np.clip(q, -127, 127)
    q[~np.isfinite(q)] = 0
    q = q.astype(np.int8)
    sums = groups.sum(axis=2)
    ds = np.stack(
        [_f16(scale), _f16(sums)],
        axis=2,
    )
    return q, ds


def _mmq_reference(
    qweight: np.ndarray,
    q8: np.ndarray,
    ds: np.ndarray,
    rows: int,
    out_features: int,
    in_features: int,
) -> np.ndarray:
    """CPU reference for the raw-Q4_K x DS4-Q8_1 MMQ arithmetic."""
    blocks_per_row = in_features // QK_K
    raw = qweight.reshape(out_features, blocks_per_row, Q4_K_BLOCK_BYTES)
    d = (
        raw[:, :, 0:2]
        .copy()
        .view(np.float16)
        .astype(np.float32)[:, :, 0]
    )
    dmin = (
        raw[:, :, 2:4]
        .copy()
        .view(np.float16)
        .astype(np.float32)[:, :, 0]
    )
    s = raw[:, :, 4:16].astype(np.uint8)
    scale6 = np.empty((out_features, blocks_per_row, 8), dtype=np.int64)
    min6 = np.empty((out_features, blocks_per_row, 8), dtype=np.int64)
    scale6[:, :, 0:4] = s[:, :, 0:4] & 0x3F
    min6[:, :, 0:4] = s[:, :, 4:8] & 0x3F
    scale6[:, :, 4:8] = (s[:, :, 8:12] & 0x0F) | ((s[:, :, 0:4] >> 2) & 0x30)
    min6[:, :, 4:8] = (s[:, :, 8:12] >> 4) | ((s[:, :, 4:8] >> 2) & 0x30)
    qs_pairs = raw[:, :, 16:].astype(np.uint8).reshape(
        out_features, blocks_per_row, 4, 32
    )
    q4 = np.empty((out_features, blocks_per_row, 8, 32), dtype=np.int64)
    q4[:, :, 0::2] = qs_pairs & 0x0F
    q4[:, :, 1::2] = qs_pairs >> 4
    a = q8.reshape(rows, blocks_per_row, 8, 32).astype(np.int64)
    out = np.zeros((rows, out_features), dtype=np.float64)
    for blk in range(blocks_per_row):
        dot = np.einsum("rkj,okj->rok", a[:, blk], q4[:, blk])
        ds_blk = ds[:, blk * 8 : (blk + 1) * 8, :]
        xd = ds_blk[:, :, 0][:, None, :]
        xsum = ds_blk[:, :, 1][:, None, :]
        out += (
            d[None, :, blk][:, :, None]
            * scale6[:, blk][None]
            * xd
            * dot
            - dmin[None, :, blk][:, :, None] * min6[:, blk][None] * xsum
        ).sum(axis=2)
    return out


def _run_arms(
    x_bits: np.ndarray,
    qweight: np.ndarray,
    mmq_library,
    pack_library,
) -> dict[str, np.ndarray]:
    from hipengine.core.hip import get_hip_runtime

    runtime = get_hip_runtime()
    rows, in_features = x_bits.shape
    out_features = int(qweight.shape[0])

    buffers = []

    def upload(value: np.ndarray):
        value = np.ascontiguousarray(value)
        buffer = malloc(value.nbytes, runtime=runtime)
        buffers.append(buffer)
        copy_host_to_device(buffer, host_array_ptr(value), runtime=runtime)
        return buffer

    x_buf = upload(x_bits)
    w_buf = upload(qweight)
    xq_buf = malloc(
        rows * (in_features // 128) * DS4_BLOCK_BYTES, runtime=runtime
    )
    arms = {
        "mmq32_ctl": malloc(rows * out_features * 4, runtime=runtime),
        "mmq32_vdr": malloc(rows * out_features * 4, runtime=runtime),
        "wmma32_ctl": malloc(rows * out_features * 4, runtime=runtime),
        "wmma32_vdr": malloc(rows * out_features * 4, runtime=runtime),
    }
    buffers.extend([xq_buf, *arms.values()])

    try:
        gguf_q8_1_mmq_ds4_pack_bf16(
            x_buf.ptr,
            xq_buf.ptr,
            rows,
            in_features,
            library=pack_library,
            runtime=runtime,
        )
        common = dict(
            rows=rows,
            in_features=in_features,
            out_features=out_features,
            library=mmq_library,
            runtime=runtime,
        )
        gguf_q4_k_q8_1_mmq32_ctl_dense_bf16_f32_out(
            xq_buf.ptr, w_buf.ptr, arms["mmq32_ctl"].ptr, **common
        )
        gguf_q4_k_q8_1_mmq32_vdr_dense_bf16_f32_out(
            xq_buf.ptr, w_buf.ptr, arms["mmq32_vdr"].ptr, **common
        )
        gguf_q4_k_q8_1_wmma32_ctl_dense_bf16_f32_out(
            xq_buf.ptr, w_buf.ptr, arms["wmma32_ctl"].ptr, **common
        )
        gguf_q4_k_q8_1_wmma32_vdr_dense_bf16_f32_out(
            xq_buf.ptr, w_buf.ptr, arms["wmma32_vdr"].ptr, **common
        )
        runtime.device_synchronize()

        def download(buf):
            host = np.empty((rows, out_features), dtype=np.float32)
            copy_device_to_host(
                host_array_ptr(host), buf, host.nbytes, runtime=runtime
            )
            return host

        return {name: download(buf) for name, buf in arms.items()}
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
@pytest.mark.parametrize(
    "in_features,out_features,rows",
    [
        (512, 256, 16),
        (512, 256, 32),
        (1024, 512, 64),
        (2560, 512, 17),
        (5120, 17408, 32),
    ],
)
def test_ctl_vdr_bit_identical_per_class(
    in_features: int,
    out_features: int,
    rows: int,
    mmq_library,
    pack_library,
) -> None:
    """RED contract: amortizing loads must not change any output bit."""
    rng = np.random.default_rng(0x6A342000 + in_features + rows)
    qweight = _weights(
        "q4", out_features=out_features, in_features=in_features, experts=1
    )
    x_bits = _bf16_bits(
        rng.normal(0.0, 0.1, size=(rows, in_features)).astype(np.float32)
        + 0.002
    )
    arms = _run_arms(x_bits, qweight[0], mmq_library, pack_library)
    np.testing.assert_array_equal(arms["mmq32_ctl"], arms["mmq32_vdr"])
    np.testing.assert_array_equal(arms["wmma32_ctl"], arms["wmma32_vdr"])
    assert bool(np.isfinite(arms["mmq32_ctl"]).all())
    assert bool(np.isfinite(arms["wmma32_ctl"]).all())


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
@pytest.mark.parametrize(
    "in_features,out_features,rows",
    [
        (512, 256, 16),
        (1024, 512, 33),
    ],
)
def test_arms_match_ds4_cpu_oracle(
    in_features: int,
    out_features: int,
    rows: int,
    mmq_library,
    pack_library,
) -> None:
    """All four arms track the CPU DS4-pack + MMQ reference."""
    rng = np.random.default_rng(0x6A342100 + in_features + rows)
    qweight = _weights(
        "q4", out_features=out_features, in_features=in_features, experts=1
    )
    x_bits = _bf16_bits(
        rng.normal(0.0, 0.1, size=(rows, in_features)).astype(np.float32)
        + 0.002
    )
    arms = _run_arms(x_bits, qweight[0], mmq_library, pack_library)

    x_f32 = _bf16_to_f32(x_bits)
    q8, ds = _ds4_pack_reference(x_f32)
    reference = _mmq_reference(
        qweight[0], q8, ds, rows, out_features, in_features
    )
    atol = 1.0e-3 * (float(np.abs(reference).max()) + 1.0)
    for name, candidate in arms.items():
        np.testing.assert_allclose(
            candidate, reference, rtol=2.0e-3, atol=atol, err_msg=name
        )


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_arms_pass_outer_floor_versus_exact_reference(
    mmq_library, pack_library
) -> None:
    rng = np.random.default_rng(0x6A342577)
    in_features, out_features, rows = 512, 256, 16
    qweight = _weights(
        "q4", out_features=out_features, in_features=in_features, experts=1
    )
    x_bits = _bf16_bits(
        rng.normal(0.0, 0.1, size=(rows, in_features)).astype(np.float32)
        + 0.002
    )
    arms = _run_arms(x_bits, qweight[0], mmq_library, pack_library)
    x_f32 = _bf16_to_f32(x_bits)
    exact_reference = _exact_oracle(
        "q4",
        x_f32,
        np.arange(rows, dtype=np.int64),
        np.zeros(rows, dtype=np.int64),
        qweight,
    )
    for name, candidate in arms.items():
        _, kl_max = _softmax_kl(exact_reference, candidate)
        assert kl_max <= 0.05, name
        assert _top1(exact_reference, candidate) >= 0.90, name


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_arms_deterministic_across_runs(mmq_library, pack_library) -> None:
    rng = np.random.default_rng(0x6A3429A1)
    in_features, out_features, rows = 512, 256, 48
    qweight = _weights(
        "q4", out_features=out_features, in_features=in_features, experts=1
    )
    x_bits = _bf16_bits(
        rng.normal(0.0, 0.1, size=(rows, in_features)).astype(np.float32)
    )
    first = _run_arms(x_bits, qweight[0], mmq_library, pack_library)
    second = _run_arms(x_bits, qweight[0], mmq_library, pack_library)
    for name in first:
        np.testing.assert_array_equal(first[name], second[name], err_msg=name)


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
class TestQ4Q81MmqPrefillRegistry:
    def test_all_variants_registered(self) -> None:
        from hipengine.kernels.registry import resolve

        for variant in (
            "mmq32_ctl_dense_bf16_f32_out",
            "mmq32_vdr_dense_bf16_f32_out",
            "wmma32_ctl_dense_bf16_f32_out",
            "wmma32_vdr_dense_bf16_f32_out",
            "mmq32_ctl_dense_bf16_bf16_out",
            "mmq32_vdr_dense_bf16_bf16_out",
            "wmma32_ctl_dense_bf16_bf16_out",
            "wmma32_vdr_dense_bf16_bf16_out",
        ):
            entry = resolve(
                backend="hip_gfx1100",
                layer="linear",
                quant="gguf_q4_k",
                variant=variant,
            )
            assert entry is not None, variant

    def test_wrapper_rejects_bad_shapes(self) -> None:
        for rows, in_features, out_features in (
            (0, 512, 256),
            (16, 511, 256),
            (16, 512, 255),
            (16, 512, 30),
            (16, 0, 256),
        ):
            with pytest.raises(ValueError):
                gguf_q4_k_q8_1_mmq32_vdr_dense_bf16_f32_out(
                    1,
                    1,
                    1,
                    rows,
                    in_features,
                    out_features,
                    library=ctypes.CDLL("libc.so.6"),
                )
