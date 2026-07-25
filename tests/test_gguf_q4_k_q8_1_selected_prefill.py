"""Correctness tests for the diagnostic Q4_K x Q8_1 selected prefill prototype."""

from __future__ import annotations

import numpy as np
import pytest

from hipengine.core.memory import copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc
from hipengine.kernels.cpu_reference import gguf_quant_gemv
from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_q8_1_selected_prefill import (
    build_gguf_q4_k_q8_1_selected_prefill,
    gguf_q4_k_q8_1_wmma_i8_probe_16x16,
    gguf_q8_1_mmq_ds4_pack_bf16,
    gguf_q8_1_mmq_ds4_pack_bf16_d4x3,
    gguf_q8_1_mmq_ds4_f32_pack_bf16_d4x3,
    gguf_q6_k_t16_selected_q8_1_ds4x3_f32_mmq64x32_prefill_compact32_bf16_bf16_out,
    gguf_q4_k_t16_selected_dual_q8_1_ds4_mmq32_prefill_compact32_bf16_bf16_out,
    gguf_q4_k_t16_selected_dual_q8_1_ds4x3_f32_mmq64x32_prefill_compact32_bf16_bf16_out,
    gguf_q4_k_t16_selected_dual_q8_1_ds4x3_guarded_mmq32_prefill_compact32_bf16_bf16_out,
    gguf_q4_k_t16_selected_dual_q8_1_ds4x3_mmq32_prefill_compact32_bf16_bf16_out,
    gguf_q4_k_t16_selected_q8_1_ds4_f32_mmq64x32_prefill_compact32_bf16_bf16_out,
    gguf_q4_k_t16_selected_dual_sparse_exact_correct_bf16,
    gguf_q4_k_x8_selected_dual_q8_1_ds4_mmq32_prefill_compact32_bf16_bf16_out,
    gguf_q4_k_selected_dual_q8_1_ds4_mmq32_prefill_compact32_bf16_bf16_out,
    gguf_q4_k_selected_dual_q8_1_ds4_prefill_compact32_bf16_bf16_out,
    gguf_q4_k_selected_dual_q8_1_ds4_preview_wmma32_prefill_compact32_bf16_bf16_out,
    gguf_q4_k_selected_dual_q8_1_ds4_wmma_prefill_compact32_bf16_bf16_out,
    gguf_q4_k_selected_dual_q8_1_ds4_wmma32_lds_prefill_compact32_bf16_bf16_out,
    gguf_q4_k_selected_dual_q8_1_ds4_wmma32_ldspack_prefill_compact32_bf16_bf16_out,
    gguf_q4_k_selected_dual_q8_1_ds4_wmma32_prefill_compact32_bf16_bf16_out,
    gguf_q4_k_selected_dual_q8_1_ds4_wmma64_prefill_compact32_bf16_bf16_out,
    gguf_q4_k_selected_dual_q8_1_prefill_compact32_bf16_bf16_out,
    plan_gguf_q4_k_q8_1_selected_prefill_build,
)
from hipengine.kernels.registry import resolve
from hipengine.quant.gguf import GGMLQuantizationType, dequantize_gguf_data
from hipengine.quant.gguf_q4_k import (
    GGUF_Q4_K_BLOCK_BYTES,
    GGUF_Q4_K_SUBBLOCK,
    GGUF_Q4_K_SUBBLOCKS,
    gguf_q4_k_mmq_tile16_preview_matmul,
    pack_gguf_q4_k_mmq_tile16_preview,
    pack_q8_1_mmq_ds4_from_bf16,
    repack_gguf_q4_k_tile16,
)
from hipengine.quant.gguf_x8 import repack_gguf_q4_k_x8
from tests.test_gguf_q4_k_selected_wmma_prefill import (
    _TOLERANCE_BF16,
    _bf16_bits_to_float32,
    _build_compact_fixture,
    _decode_output,
    _hip_available,
)

_Q8_1_BLOCK = 32


def _quantize_q8_1_blocks(x_bf16: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = _bf16_bits_to_float32(x_bf16).astype(np.float32, copy=False)
    blocks = x.reshape(x.shape[0], x.shape[1] // _Q8_1_BLOCK, _Q8_1_BLOCK)
    max_abs = np.max(np.abs(blocks), axis=-1)
    d = (max_abs / 127.0).astype(np.float32)
    safe_d = np.where(d > 0.0, d, 1.0).astype(np.float32)
    qs = np.rint(blocks / safe_d[..., None]).clip(-127, 127).astype(np.int8)
    qs = np.where(d[..., None] > 0.0, qs, np.zeros_like(qs)).astype(np.int8, copy=False)
    sums = (qs.astype(np.float32).sum(axis=-1) * d).astype(np.float32)
    return np.ascontiguousarray(qs), np.ascontiguousarray(d), np.ascontiguousarray(sums)


def _dequant_q8_1(qs: np.ndarray, d: np.ndarray) -> np.ndarray:
    return (qs.astype(np.float32) * d[..., None]).reshape(qs.shape[0], qs.shape[1] * qs.shape[2])


def _q8_1_selected_reference(fixture) -> np.ndarray:
    qs, d, _ = _quantize_q8_1_blocks(fixture.x_host)
    x_ref = _dequant_q8_1(qs, d)
    ref = np.zeros((fixture.compact_rows, fixture.out_features_a + fixture.out_features_b), dtype=np.float32)
    for expert in range(fixture.num_experts):
        start = int(fixture.expert_start_compact[expert])
        stop = int(fixture.expert_start_compact[expert + 1])
        if stop == start:
            continue
        ref[start:stop, : fixture.out_features_a] = gguf_quant_gemv(
            x_ref[start:stop], fixture.qweight_a[expert], GGMLQuantizationType.Q4_K
        )
        ref[start:stop, fixture.out_features_a :] = gguf_quant_gemv(
            x_ref[start:stop], fixture.qweight_b[expert], GGMLQuantizationType.Q4_K
        )
    return ref


def _q8_1_ds4_selected_reference(fixture) -> np.ndarray:
    q8_ds4 = pack_q8_1_mmq_ds4_from_bf16(fixture.x_host)
    ref = np.zeros((fixture.compact_rows, fixture.out_features_a + fixture.out_features_b), dtype=np.float32)
    for expert in range(fixture.num_experts):
        start = int(fixture.expert_start_compact[expert])
        stop = int(fixture.expert_start_compact[expert + 1])
        if stop == start:
            continue
        ref[start:stop, : fixture.out_features_a] = gguf_q4_k_mmq_tile16_preview_matmul(
            q8_ds4[start:stop], pack_gguf_q4_k_mmq_tile16_preview(fixture.qweight_a[expert])
        )
        ref[start:stop, fixture.out_features_a :] = gguf_q4_k_mmq_tile16_preview_matmul(
            q8_ds4[start:stop], pack_gguf_q4_k_mmq_tile16_preview(fixture.qweight_b[expert])
        )
    return ref


def _max_softmax_kl(reference: np.ndarray, candidate: np.ndarray) -> float:
    ref = reference.astype(np.float64)
    cand = candidate.astype(np.float64)
    ref -= ref.max(axis=-1, keepdims=True)
    cand -= cand.max(axis=-1, keepdims=True)
    ref_logp = ref - np.log(np.exp(ref).sum(axis=-1, keepdims=True))
    cand_logp = cand - np.log(np.exp(cand).sum(axis=-1, keepdims=True))
    return float(
        np.max(np.sum(np.exp(ref_logp) * (ref_logp - cand_logp), axis=-1))
    )


# ---------------------------------------------------------------------------
# No-GPU surface checks.
# ---------------------------------------------------------------------------


def test_gguf_q4_k_q8_1_selected_prefill_registry_and_build_plan() -> None:
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="activation_quant",
            quant="q8_1_ds4x3",
            variant="bf16",
        )
        is gguf_q8_1_mmq_ds4_pack_bf16_d4x3
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="activation_quant",
            quant="q8_1_ds4x3_f32",
            variant="bf16",
        )
        is gguf_q8_1_mmq_ds4_f32_pack_bf16_d4x3
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="moe_linear",
            quant="gguf_q6_k_t16_v1",
            variant=(
                "selected_q8_1_ds4x3_f32_mmq64x32_"
                "prefill_compact32_bf16_bf16_out"
            ),
        )
        is gguf_q6_k_t16_selected_q8_1_ds4x3_f32_mmq64x32_prefill_compact32_bf16_bf16_out
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="moe_linear",
            quant="gguf_q4_k",
            variant="selected_dual_q8_1_prefill_compact32_bf16_bf16_out",
        )
        is gguf_q4_k_selected_dual_q8_1_prefill_compact32_bf16_bf16_out
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="moe_linear",
            quant="gguf_q4_k",
            variant="selected_dual_q8_1_ds4_mmq32_prefill_compact32_bf16_bf16_out",
        )
        is gguf_q4_k_selected_dual_q8_1_ds4_mmq32_prefill_compact32_bf16_bf16_out
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="moe_linear",
            quant="gguf_q4_k_x8_v1",
            variant="selected_dual_q8_1_ds4_mmq32_prefill_compact32_bf16_bf16_out",
        )
        is gguf_q4_k_x8_selected_dual_q8_1_ds4_mmq32_prefill_compact32_bf16_bf16_out
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="moe_linear",
            quant="gguf_q4_k_t16_v1",
            variant="selected_dual_q8_1_ds4_mmq32_prefill_compact32_bf16_bf16_out",
        )
        is gguf_q4_k_t16_selected_dual_q8_1_ds4_mmq32_prefill_compact32_bf16_bf16_out
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="moe_linear",
            quant="gguf_q4_k_t16_v1",
            variant="selected_dual_q8_1_ds4x3_mmq32_prefill_compact32_bf16_bf16_out",
        )
        is gguf_q4_k_t16_selected_dual_q8_1_ds4x3_mmq32_prefill_compact32_bf16_bf16_out
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="moe_linear",
            quant="gguf_q4_k_t16_v1",
            variant="selected_dual_q8_1_ds4x3_guarded_mmq32_prefill_compact32_bf16_bf16_out",
        )
        is gguf_q4_k_t16_selected_dual_q8_1_ds4x3_guarded_mmq32_prefill_compact32_bf16_bf16_out
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="moe_linear_repair",
            quant="gguf_q4_k_t16_v1",
            variant="selected_dual_sparse_exact_bf16",
        )
        is gguf_q4_k_t16_selected_dual_sparse_exact_correct_bf16
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="moe_linear",
            quant="gguf_q4_k",
            variant="selected_dual_q8_1_ds4_prefill_compact32_bf16_bf16_out",
        )
        is gguf_q4_k_selected_dual_q8_1_ds4_prefill_compact32_bf16_bf16_out
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="moe_linear",
            quant="gguf_q4_k",
            variant="selected_dual_q8_1_ds4_wmma_prefill_compact32_bf16_bf16_out",
        )
        is gguf_q4_k_selected_dual_q8_1_ds4_wmma_prefill_compact32_bf16_bf16_out
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="moe_linear",
            quant="gguf_q4_k",
            variant="selected_dual_q8_1_ds4_wmma32_prefill_compact32_bf16_bf16_out",
        )
        is gguf_q4_k_selected_dual_q8_1_ds4_wmma32_prefill_compact32_bf16_bf16_out
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="moe_linear",
            quant="gguf_q4_k",
            variant="selected_dual_q8_1_ds4_wmma64_prefill_compact32_bf16_bf16_out",
        )
        is gguf_q4_k_selected_dual_q8_1_ds4_wmma64_prefill_compact32_bf16_bf16_out
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="moe_linear",
            quant="gguf_q4_k",
            variant="selected_dual_q8_1_ds4_preview_wmma32_prefill_compact32_bf16_bf16_out",
        )
        is gguf_q4_k_selected_dual_q8_1_ds4_preview_wmma32_prefill_compact32_bf16_bf16_out
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="moe_linear",
            quant="gguf_q4_k",
            variant="selected_dual_q8_1_ds4_wmma32_ldspack_prefill_compact32_bf16_bf16_out",
        )
        is gguf_q4_k_selected_dual_q8_1_ds4_wmma32_ldspack_prefill_compact32_bf16_bf16_out
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="moe_linear",
            quant="gguf_q4_k",
            variant="selected_dual_q8_1_ds4_wmma32_lds_prefill_compact32_bf16_bf16_out",
        )
        is gguf_q4_k_selected_dual_q8_1_ds4_wmma32_lds_prefill_compact32_bf16_bf16_out
    )
    artifact = plan_gguf_q4_k_q8_1_selected_prefill_build(compiler_version="test-compiler")
    assert artifact.output_path.name == "gguf_q4_k_q8_1_selected_prefill.so"
    assert any(path.name == "gguf_q4_k_q8_1_selected_prefill.hip" for path in artifact.sources)
    assert "-mcumode" in artifact.flags

    dry_run = build_gguf_q4_k_q8_1_selected_prefill(dry_run=True, compiler_version="test-compiler")
    assert dry_run.output_path == artifact.output_path


def test_q4_k_mmq_tile16_preview_reconstructs_raw_q4_k_values() -> None:
    fixture = _build_compact_fixture(
        counts=[3],
        in_features=512,
        out_features_a=32,
        out_features_b=16,
        dtype="bf16",
        seed=5,
    )
    raw = fixture.qweight_a[0]
    preview = pack_gguf_q4_k_mmq_tile16_preview(raw)

    reconstructed = np.empty((fixture.out_features_a, fixture.in_features), dtype=np.float32)
    for out_tile in range(preview.out_tiles):
        for col in range(16):
            out_col = out_tile * 16 + col
            for blk in range(preview.blocks_per_row):
                for sb in range(GGUF_Q4_K_SUBBLOCKS):
                    start = blk * 256 + sb * GGUF_Q4_K_SUBBLOCK
                    reconstructed[out_col, start : start + GGUF_Q4_K_SUBBLOCK] = (
                        preview.q4[out_tile, col, blk, sb].astype(np.float32)
                        * preview.scales[out_tile, col, blk, sb]
                        - preview.mins[out_tile, col, blk, sb]
                    )

    expected = dequantize_gguf_data(raw, GGMLQuantizationType.Q4_K)
    assert preview.q4.shape == (2, 16, 2, 8, 32)
    assert preview.scales.shape == (2, 16, 2, 8)
    assert preview.mins.shape == (2, 16, 2, 8)
    assert raw.shape[1] == preview.blocks_per_row * GGUF_Q4_K_BLOCK_BYTES
    np.testing.assert_allclose(reconstructed, expected, rtol=0.0, atol=1e-6)


def test_gguf_q4_k_q8_1_selected_prefill_wrapper_validates_common_contract() -> None:
    kwargs = dict(
        x_qs_ptr=1,
        x_d_ptr=2,
        x_sum_ptr=3,
        expert_start_compact_ptr=4,
        expert_start_wmma_ptr=5,
        tile_expert_ptr=6,
        qweight_a_ptr=7,
        qweight_b_ptr=8,
        out_ptr=9,
        compact_rows=17,
        in_features=256,
        out_features_a=32,
        out_features_b=32,
        num_experts=2,
        wmma_total_rows=32,
    )

    with pytest.raises(ValueError, match="compact_rows"):
        gguf_q4_k_selected_dual_q8_1_prefill_compact32_bf16_bf16_out(
            **{**kwargs, "compact_rows": 0}
        )
    with pytest.raises(ValueError, match="Q4_K block size 256"):
        gguf_q4_k_selected_dual_q8_1_prefill_compact32_bf16_bf16_out(
            **{**kwargs, "in_features": 128}
        )
    with pytest.raises(ValueError, match="out_features_a.*multiple of 16"):
        gguf_q4_k_selected_dual_q8_1_prefill_compact32_bf16_bf16_out(
            **{**kwargs, "out_features_a": 24}
        )
    with pytest.raises(ValueError, match="out_features_b.*multiple of 16"):
        gguf_q4_k_selected_dual_q8_1_prefill_compact32_bf16_bf16_out(
            **{**kwargs, "out_features_b": 24}
        )
    with pytest.raises(ValueError, match="wmma_total_rows.*multiple of 16"):
        gguf_q4_k_selected_dual_q8_1_prefill_compact32_bf16_bf16_out(
            **{**kwargs, "wmma_total_rows": 31}
        )

    ds4_kwargs = {
        k: v
        for k, v in kwargs.items()
        if k not in {"x_qs_ptr", "x_d_ptr", "x_sum_ptr"}
    }
    ds4_kwargs["x_q8_ptr"] = 1
    with pytest.raises(ValueError, match="compact_rows"):
        gguf_q4_k_selected_dual_q8_1_ds4_prefill_compact32_bf16_bf16_out(
            **{**ds4_kwargs, "compact_rows": 0}
        )

    mmq32_kwargs = {
        key: value
        for key, value in ds4_kwargs.items()
        if key not in {"expert_start_wmma_ptr", "tile_expert_ptr", "wmma_total_rows"}
    }
    mmq32_kwargs.update(
        compact_to_source_ptr=9,
        expert_start_mmq32_ptr=ds4_kwargs["expert_start_wmma_ptr"],
        mmq_tile_expert_ptr=ds4_kwargs["tile_expert_ptr"],
        mmq_total_rows=ds4_kwargs["wmma_total_rows"],
    )
    with pytest.raises(ValueError, match="out_features_a.*multiple of 32"):
        gguf_q4_k_selected_dual_q8_1_ds4_mmq32_prefill_compact32_bf16_bf16_out(
            **{**mmq32_kwargs, "out_features_a": 16}
        )
    with pytest.raises(ValueError, match="mmq_total_rows.*multiple of 32"):
        gguf_q4_k_selected_dual_q8_1_ds4_mmq32_prefill_compact32_bf16_bf16_out(
            **{**mmq32_kwargs, "mmq_total_rows": 31}
        )


# ---------------------------------------------------------------------------
# HIP correctness fixtures.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_wmma_i8_probe_16x16_matches_cpu_matmul() -> None:
    from hipengine.core.hip import get_hip_runtime

    runtime = get_hip_runtime()
    library = build_gguf_q4_k_q8_1_selected_prefill(load=True)
    a_rows = ((np.arange(16 * 16, dtype=np.int16).reshape(16, 16) % 17) - 8).astype(np.int8)
    b_cols = ((np.arange(16 * 16, dtype=np.uint16).reshape(16, 16) * 3 + 5) % 16).astype(np.uint8)
    actual = np.zeros((16, 16), dtype=np.int32)
    expected = a_rows.astype(np.int32) @ b_cols.astype(np.int32).T

    bufs = []
    try:
        a_dev = malloc(a_rows.nbytes, runtime=runtime)
        b_dev = malloc(b_cols.nbytes, runtime=runtime)
        out_dev = malloc(actual.nbytes, runtime=runtime)
        bufs.extend((a_dev, b_dev, out_dev))
        copy_host_to_device(a_dev, host_array_ptr(np.ascontiguousarray(a_rows)), runtime=runtime)
        copy_host_to_device(b_dev, host_array_ptr(np.ascontiguousarray(b_cols)), runtime=runtime)
        gguf_q4_k_q8_1_wmma_i8_probe_16x16(
            a_dev.ptr,
            b_dev.ptr,
            out_dev.ptr,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(actual), out_dev, runtime=runtime)
    finally:
        for buf in reversed(bufs):
            free(buf, runtime=runtime)

    np.testing.assert_array_equal(actual, expected)


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize(("counts", "in_features"), [([1], 256), ([3, 2], 512)])
def test_q8_1_mmq_ds4_pack_bf16_matches_cpu(counts: list[int], in_features: int) -> None:
    from hipengine.core.hip import get_hip_runtime

    fixture = _build_compact_fixture(
        counts=counts,
        in_features=in_features,
        out_features_a=32,
        out_features_b=32,
        dtype="bf16",
        seed=19,
    )
    expected = pack_q8_1_mmq_ds4_from_bf16(fixture.x_host)
    actual = np.zeros_like(expected)
    runtime = get_hip_runtime()
    library = build_gguf_q4_k_q8_1_selected_prefill(load=True)

    bufs = []
    try:
        x_dev = malloc(fixture.x_host.nbytes, runtime=runtime)
        out_dev = malloc(actual.nbytes, runtime=runtime)
        bufs.extend((x_dev, out_dev))
        copy_host_to_device(x_dev, host_array_ptr(np.ascontiguousarray(fixture.x_host)), runtime=runtime)
        gguf_q8_1_mmq_ds4_pack_bf16(
            x_dev.ptr,
            out_dev.ptr,
            fixture.compact_rows,
            fixture.in_features,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(actual), out_dev, runtime=runtime)
    finally:
        for buf in reversed(bufs):
            free(buf, runtime=runtime)

    np.testing.assert_array_equal(actual, expected)


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_q8_1_mmq_ds4_residual_x3_preserves_primary_and_reconstructs_bf16() -> None:
    from hipengine.core.hip import get_hip_runtime

    rows, hidden = 3, 512
    fixture = _build_compact_fixture(
        counts=[rows],
        in_features=hidden,
        out_features_a=32,
        out_features_b=32,
        dtype="bf16",
        seed=29,
    )
    primary = pack_q8_1_mmq_ds4_from_bf16(fixture.x_host)
    packed = np.zeros((3, *primary.shape), dtype=np.uint8)
    runtime = get_hip_runtime()
    library = build_gguf_q4_k_q8_1_selected_prefill(load=True)

    bufs = []
    try:
        x_dev = malloc(fixture.x_host.nbytes, runtime=runtime)
        out_dev = malloc(packed.nbytes, runtime=runtime)
        bufs.extend((x_dev, out_dev))
        copy_host_to_device(
            x_dev,
            host_array_ptr(np.ascontiguousarray(fixture.x_host)),
            runtime=runtime,
        )
        gguf_q8_1_mmq_ds4_pack_bf16_d4x3(
            x_dev.ptr,
            out_dev.ptr,
            rows,
            hidden,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(packed), out_dev, runtime=runtime)
    finally:
        for buf in reversed(bufs):
            free(buf, runtime=runtime)

    np.testing.assert_array_equal(packed[0], primary)
    reconstructed = np.zeros((rows, hidden), dtype=np.float32)
    blocks = packed.reshape(3, rows, hidden // 128, 144)
    for plane in range(3):
        scales = blocks[plane, :, :, :16].view(np.float16).astype(np.float32)
        quants = blocks[plane, :, :, 16:].view(np.int8).astype(np.float32)
        reconstructed += (
            quants.reshape(rows, hidden // 128, 4, 32)
            * scales[..., 0::2, None]
        ).reshape(rows, hidden)
    source = _bf16_bits_to_float32(fixture.x_host)
    relative_l2 = float(
        np.linalg.norm(reconstructed - source)
        / max(np.linalg.norm(source), 1e-12)
    )
    assert relative_l2 <= 5e-5


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_q8_1_mmq_ds8_f32_halves_quant_groups_without_more_bytes() -> None:
    from hipengine.core.hip import get_hip_runtime

    rows, hidden = 2, 512
    source_f32 = np.empty((rows, hidden), dtype=np.float32)
    source_halves = source_f32.reshape(rows, hidden // 16, 16)
    for half in range(source_halves.shape[1]):
        amplitude = 0.25 if half % 2 == 0 else 4.0
        source_halves[:, half] = np.linspace(
            -amplitude,
            amplitude,
            16,
            dtype=np.float32,
        )
    source_bits = source_f32.view(np.uint32)
    host_x = (
        source_bits + np.uint32(0x7FFF) + ((source_bits >> 16) & 1)
    ).astype(np.uint32)
    host_x = np.ascontiguousarray((host_x >> 16).astype(np.uint16))
    packed = np.zeros((rows, hidden // 128, 160), dtype=np.uint8)
    runtime = get_hip_runtime()
    library = build_gguf_q4_k_q8_1_selected_prefill(load=True)

    bufs = []
    try:
        x_dev = malloc(host_x.nbytes, runtime=runtime)
        out_dev = malloc(packed.nbytes, runtime=runtime)
        bufs.extend((x_dev, out_dev))
        copy_host_to_device(
            x_dev,
            host_array_ptr(host_x),
            runtime=runtime,
        )
        gguf_q8_1_mmq_ds4_f32_pack_bf16_d4x3(
            x_dev.ptr,
            out_dev.ptr,
            rows,
            hidden,
            residual_passes=1,
            split16=True,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(
            host_array_ptr(packed),
            out_dev,
            runtime=runtime,
        )
    finally:
        for buf in reversed(bufs):
            free(buf, runtime=runtime)

    scales = packed[..., :32].view(np.float32)
    quants = packed[..., 32:].view(np.int8).reshape(
        rows,
        hidden // 128,
        8,
        16,
    )
    reconstructed = (
        quants.astype(np.float32) * scales[..., None]
    ).reshape(rows, hidden)
    source = _bf16_bits_to_float32(host_x)
    relative_l2 = float(
        np.linalg.norm(reconstructed - source)
        / max(np.linalg.norm(source), 1e-12)
    )
    source_d4 = source.reshape(rows, hidden // 128, 4, 32)
    d4_scales = np.max(np.abs(source_d4), axis=-1) / 127.0
    safe_d4_scales = np.where(d4_scales > 0.0, d4_scales, 1.0)
    d4_quants = np.rint(source_d4 / safe_d4_scales[..., None]).clip(
        -127,
        127,
    )
    d4_reconstructed = (d4_quants * d4_scales[..., None]).reshape(
        rows,
        hidden,
    )
    d4_relative_l2 = float(
        np.linalg.norm(d4_reconstructed - source)
        / max(np.linalg.norm(source), 1e-12)
    )
    assert packed.nbytes == rows * (hidden // 128) * 160
    assert relative_l2 < d4_relative_l2


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("capacity_divisor", [1, 2])
def test_q4_k_t16_ds4x3_all_queued_repair_matches_production_exact_bits(
    capacity_divisor: int,
) -> None:
    from hipengine.core.hip import get_hip_runtime
    from hipengine.kernels.hip_gfx1100.quant.gguf_t16_selected_gemv import (
        build_gguf_t16_selected_gemv,
        gguf_q4_k_t16_selected_dual_gemv_bf16_bf16_out,
    )
    from tests.test_gguf_t16_selected_gemv_decode import _run_direct_dual

    counts = [4, 0, 5]
    fixture = _build_compact_fixture(
        counts=counts,
        in_features=512,
        out_features_a=32,
        out_features_b=32,
        dtype="bf16",
        seed=37,
    )
    selected = np.concatenate(
        [
            np.full(count, expert, dtype=np.int64)
            for expert, count in enumerate(counts)
        ]
    )
    tiles_a = repack_gguf_q4_k_tile16(fixture.qweight_a).tiles
    tiles_b = repack_gguf_q4_k_tile16(fixture.qweight_b).tiles
    exact_library = build_gguf_t16_selected_gemv(load=True)
    exact_a, exact_b = _run_direct_dual(
        gguf_q4_k_t16_selected_dual_gemv_bf16_bf16_out,
        fixture.x_host,
        selected,
        tiles_a,
        tiles_b,
        fixture.out_features_a,
        np.uint16,
        exact_library,
    )
    exact = np.concatenate((exact_a, exact_b), axis=1)

    source_x = np.ascontiguousarray(fixture.x_host[::-1])
    compact_to_source = np.arange(
        fixture.compact_rows - 1, -1, -1, dtype=np.int64
    )
    expert_start_mmq32, tile_expert, mmq_total_rows = _mmq32_metadata(
        fixture
    )
    total_risk_tiles = (
        fixture.compact_rows
        * (fixture.out_features_a + fixture.out_features_b)
        // 16
    )
    risk_capacity = total_risk_tiles // capacity_divisor
    actual = np.zeros_like(exact)
    risk_count = np.zeros((1,), dtype=np.int32)
    risk_tiles = np.zeros((risk_capacity,), dtype=np.int32)
    runtime = get_hip_runtime()
    library = build_gguf_q4_k_q8_1_selected_prefill(load=True)

    bufs = []
    try:
        source_dev = malloc(source_x.nbytes, runtime=runtime)
        q8_dev = malloc(
            3 * source_x.shape[0] * (fixture.in_features // 128) * 144,
            runtime=runtime,
        )
        compact_to_source_dev = malloc(
            compact_to_source.nbytes, runtime=runtime
        )
        starts_dev = malloc(
            fixture.expert_start_compact.nbytes, runtime=runtime
        )
        starts32_dev = malloc(
            expert_start_mmq32.nbytes, runtime=runtime
        )
        tile_expert_dev = malloc(tile_expert.nbytes, runtime=runtime)
        tiles_a_dev = malloc(tiles_a.nbytes, runtime=runtime)
        tiles_b_dev = malloc(tiles_b.nbytes, runtime=runtime)
        out_dev = malloc(actual.nbytes, runtime=runtime)
        count_dev = malloc(risk_count.nbytes, runtime=runtime)
        risks_dev = malloc(risk_tiles.nbytes, runtime=runtime)
        bufs.extend(
            (
                source_dev,
                q8_dev,
                compact_to_source_dev,
                starts_dev,
                starts32_dev,
                tile_expert_dev,
                tiles_a_dev,
                tiles_b_dev,
                out_dev,
                count_dev,
                risks_dev,
            )
        )
        for dev, arr in (
            (source_dev, source_x),
            (compact_to_source_dev, compact_to_source),
            (starts_dev, fixture.expert_start_compact),
            (starts32_dev, expert_start_mmq32),
            (tile_expert_dev, tile_expert),
            (tiles_a_dev, tiles_a),
            (tiles_b_dev, tiles_b),
        ):
            copy_host_to_device(
                dev,
                host_array_ptr(np.ascontiguousarray(arr)),
                runtime=runtime,
            )
        runtime.memset(count_dev.ptr, 0, count_dev.nbytes)
        gguf_q8_1_mmq_ds4_pack_bf16_d4x3(
            source_dev.ptr,
            q8_dev.ptr,
            source_x.shape[0],
            fixture.in_features,
            library=library,
            runtime=runtime,
        )
        gguf_q4_k_t16_selected_dual_q8_1_ds4x3_guarded_mmq32_prefill_compact32_bf16_bf16_out(
            q8_dev.ptr,
            compact_to_source_dev.ptr,
            starts_dev.ptr,
            starts32_dev.ptr,
            tile_expert_dev.ptr,
            tiles_a_dev.ptr,
            tiles_b_dev.ptr,
            out_dev.ptr,
            count_dev.ptr,
            risks_dev.ptr,
            risk_capacity,
            float("inf"),
            fixture.compact_rows,
            source_x.shape[0],
            fixture.in_features,
            fixture.out_features_a,
            fixture.out_features_b,
            fixture.num_experts,
            mmq_total_rows,
            library=library,
            runtime=runtime,
        )
        gguf_q4_k_t16_selected_dual_sparse_exact_correct_bf16(
            source_dev.ptr,
            compact_to_source_dev.ptr,
            starts_dev.ptr,
            tiles_a_dev.ptr,
            tiles_b_dev.ptr,
            out_dev.ptr,
            count_dev.ptr,
            risks_dev.ptr,
            risk_capacity,
            fixture.compact_rows,
            source_x.shape[0],
            fixture.in_features,
            fixture.out_features_a,
            fixture.out_features_b,
            fixture.num_experts,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(
            host_array_ptr(actual), out_dev, runtime=runtime
        )
        copy_device_to_host(
            host_array_ptr(risk_count), count_dev, runtime=runtime
        )
    finally:
        for buf in reversed(bufs):
            free(buf, runtime=runtime)

    assert int(risk_count[0]) == total_risk_tiles
    np.testing.assert_array_equal(actual, exact)


def _run_q8_1_selected_dual_gpu(fixture) -> np.ndarray:
    from hipengine.core.hip import get_hip_runtime

    runtime = get_hip_runtime()
    library = build_gguf_q4_k_q8_1_selected_prefill(load=True)
    host_out = np.zeros(
        (fixture.compact_rows, fixture.out_features_a + fixture.out_features_b),
        dtype=np.uint16,
    )
    q8_qs, q8_d, q8_sum = _quantize_q8_1_blocks(fixture.x_host)

    bufs = []
    try:
        q8_qs_dev = malloc(q8_qs.nbytes, runtime=runtime)
        q8_d_dev = malloc(q8_d.nbytes, runtime=runtime)
        q8_sum_dev = malloc(q8_sum.nbytes, runtime=runtime)
        start_compact_dev = malloc(fixture.expert_start_compact.nbytes, runtime=runtime)
        start_wmma_dev = malloc(fixture.expert_start_wmma.nbytes, runtime=runtime)
        tile_expert_dev = malloc(fixture.tile_expert.nbytes, runtime=runtime)
        qweight_a_dev = malloc(fixture.qweight_a.nbytes, runtime=runtime)
        qweight_b_dev = malloc(fixture.qweight_b.nbytes, runtime=runtime)
        out_dev = malloc(host_out.nbytes, runtime=runtime)
        bufs.extend(
            (
                q8_qs_dev,
                q8_d_dev,
                q8_sum_dev,
                start_compact_dev,
                start_wmma_dev,
                tile_expert_dev,
                qweight_a_dev,
                qweight_b_dev,
                out_dev,
            )
        )
        for dev, arr in (
            (q8_qs_dev, q8_qs),
            (q8_d_dev, q8_d),
            (q8_sum_dev, q8_sum),
            (start_compact_dev, fixture.expert_start_compact),
            (start_wmma_dev, fixture.expert_start_wmma),
            (tile_expert_dev, fixture.tile_expert),
            (qweight_a_dev, fixture.qweight_a),
            (qweight_b_dev, fixture.qweight_b),
        ):
            copy_host_to_device(dev, host_array_ptr(np.ascontiguousarray(arr)), runtime=runtime)

        gguf_q4_k_selected_dual_q8_1_prefill_compact32_bf16_bf16_out(
            q8_qs_dev.ptr,
            q8_d_dev.ptr,
            q8_sum_dev.ptr,
            start_compact_dev.ptr,
            start_wmma_dev.ptr,
            tile_expert_dev.ptr,
            qweight_a_dev.ptr,
            qweight_b_dev.ptr,
            out_dev.ptr,
            fixture.compact_rows,
            fixture.in_features,
            fixture.out_features_a,
            fixture.out_features_b,
            fixture.num_experts,
            fixture.wmma_total_rows,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(host_out), out_dev, runtime=runtime)
    finally:
        for buf in reversed(bufs):
            free(buf, runtime=runtime)

    return _decode_output(host_out, "bf16")


def _run_q8_1_ds4_variant_gpu(fixture, launcher) -> np.ndarray:
    from hipengine.core.hip import get_hip_runtime

    runtime = get_hip_runtime()
    library = build_gguf_q4_k_q8_1_selected_prefill(load=True)
    host_out = np.zeros(
        (fixture.compact_rows, fixture.out_features_a + fixture.out_features_b),
        dtype=np.uint16,
    )
    q8_ds4 = pack_q8_1_mmq_ds4_from_bf16(fixture.x_host)

    bufs = []
    try:
        q8_ds4_dev = malloc(q8_ds4.nbytes, runtime=runtime)
        start_compact_dev = malloc(fixture.expert_start_compact.nbytes, runtime=runtime)
        start_wmma_dev = malloc(fixture.expert_start_wmma.nbytes, runtime=runtime)
        tile_expert_dev = malloc(fixture.tile_expert.nbytes, runtime=runtime)
        qweight_a_dev = malloc(fixture.qweight_a.nbytes, runtime=runtime)
        qweight_b_dev = malloc(fixture.qweight_b.nbytes, runtime=runtime)
        out_dev = malloc(host_out.nbytes, runtime=runtime)
        bufs.extend(
            (
                q8_ds4_dev,
                start_compact_dev,
                start_wmma_dev,
                tile_expert_dev,
                qweight_a_dev,
                qweight_b_dev,
                out_dev,
            )
        )
        for dev, arr in (
            (q8_ds4_dev, q8_ds4),
            (start_compact_dev, fixture.expert_start_compact),
            (start_wmma_dev, fixture.expert_start_wmma),
            (tile_expert_dev, fixture.tile_expert),
            (qweight_a_dev, fixture.qweight_a),
            (qweight_b_dev, fixture.qweight_b),
        ):
            copy_host_to_device(dev, host_array_ptr(np.ascontiguousarray(arr)), runtime=runtime)

        launcher(
            q8_ds4_dev.ptr,
            start_compact_dev.ptr,
            start_wmma_dev.ptr,
            tile_expert_dev.ptr,
            qweight_a_dev.ptr,
            qweight_b_dev.ptr,
            out_dev.ptr,
            fixture.compact_rows,
            fixture.in_features,
            fixture.out_features_a,
            fixture.out_features_b,
            fixture.num_experts,
            fixture.wmma_total_rows,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(host_out), out_dev, runtime=runtime)
    finally:
        for buf in reversed(bufs):
            free(buf, runtime=runtime)

    return _decode_output(host_out, "bf16")


def _run_q8_1_ds4_preview_wmma32_selected_dual_gpu(fixture) -> np.ndarray:
    from hipengine.core.hip import get_hip_runtime

    runtime = get_hip_runtime()
    library = build_gguf_q4_k_q8_1_selected_prefill(load=True)
    host_out = np.zeros(
        (fixture.compact_rows, fixture.out_features_a + fixture.out_features_b),
        dtype=np.uint16,
    )
    q8_ds4 = pack_q8_1_mmq_ds4_from_bf16(fixture.x_host)
    previews_a = [pack_gguf_q4_k_mmq_tile16_preview(fixture.qweight_a[expert]) for expert in range(fixture.num_experts)]
    previews_b = [pack_gguf_q4_k_mmq_tile16_preview(fixture.qweight_b[expert]) for expert in range(fixture.num_experts)]
    q4_a = np.ascontiguousarray(np.stack([preview.q4 for preview in previews_a], axis=0))
    scale_a = np.ascontiguousarray(np.stack([preview.scales for preview in previews_a], axis=0), dtype=np.float32)
    min_a = np.ascontiguousarray(np.stack([preview.mins for preview in previews_a], axis=0), dtype=np.float32)
    q4_b = np.ascontiguousarray(np.stack([preview.q4 for preview in previews_b], axis=0))
    scale_b = np.ascontiguousarray(np.stack([preview.scales for preview in previews_b], axis=0), dtype=np.float32)
    min_b = np.ascontiguousarray(np.stack([preview.mins for preview in previews_b], axis=0), dtype=np.float32)

    bufs = []
    try:
        q8_ds4_dev = malloc(q8_ds4.nbytes, runtime=runtime)
        start_compact_dev = malloc(fixture.expert_start_compact.nbytes, runtime=runtime)
        start_wmma_dev = malloc(fixture.expert_start_wmma.nbytes, runtime=runtime)
        tile_expert_dev = malloc(fixture.tile_expert.nbytes, runtime=runtime)
        q4_a_dev = malloc(q4_a.nbytes, runtime=runtime)
        scale_a_dev = malloc(scale_a.nbytes, runtime=runtime)
        min_a_dev = malloc(min_a.nbytes, runtime=runtime)
        q4_b_dev = malloc(q4_b.nbytes, runtime=runtime)
        scale_b_dev = malloc(scale_b.nbytes, runtime=runtime)
        min_b_dev = malloc(min_b.nbytes, runtime=runtime)
        out_dev = malloc(host_out.nbytes, runtime=runtime)
        bufs.extend(
            (
                q8_ds4_dev,
                start_compact_dev,
                start_wmma_dev,
                tile_expert_dev,
                q4_a_dev,
                scale_a_dev,
                min_a_dev,
                q4_b_dev,
                scale_b_dev,
                min_b_dev,
                out_dev,
            )
        )
        for dev, arr in (
            (q8_ds4_dev, q8_ds4),
            (start_compact_dev, fixture.expert_start_compact),
            (start_wmma_dev, fixture.expert_start_wmma),
            (tile_expert_dev, fixture.tile_expert),
            (q4_a_dev, q4_a),
            (scale_a_dev, scale_a),
            (min_a_dev, min_a),
            (q4_b_dev, q4_b),
            (scale_b_dev, scale_b),
            (min_b_dev, min_b),
        ):
            copy_host_to_device(dev, host_array_ptr(np.ascontiguousarray(arr)), runtime=runtime)

        gguf_q4_k_selected_dual_q8_1_ds4_preview_wmma32_prefill_compact32_bf16_bf16_out(
            q8_ds4_dev.ptr,
            start_compact_dev.ptr,
            start_wmma_dev.ptr,
            tile_expert_dev.ptr,
            q4_a_dev.ptr,
            scale_a_dev.ptr,
            min_a_dev.ptr,
            q4_b_dev.ptr,
            scale_b_dev.ptr,
            min_b_dev.ptr,
            out_dev.ptr,
            fixture.compact_rows,
            fixture.in_features,
            fixture.out_features_a,
            fixture.out_features_b,
            fixture.num_experts,
            fixture.wmma_total_rows,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(host_out), out_dev, runtime=runtime)
    finally:
        for buf in reversed(bufs):
            free(buf, runtime=runtime)

    return _decode_output(host_out, "bf16")


def _run_q8_1_ds4_selected_dual_gpu(fixture) -> np.ndarray:
    return _run_q8_1_ds4_variant_gpu(
        fixture,
        gguf_q4_k_selected_dual_q8_1_ds4_prefill_compact32_bf16_bf16_out,
    )


def _mmq32_metadata(fixture) -> tuple[np.ndarray, np.ndarray, int]:
    counts = np.diff(fixture.expert_start_compact)
    padded_counts = ((counts + 31) // 32) * 32
    expert_start_mmq32 = np.zeros(fixture.num_experts + 1, dtype=np.int64)
    expert_start_mmq32[1:] = np.cumsum(padded_counts)
    tile_expert = np.asarray(
        [
            expert
            for expert, padded in enumerate(padded_counts)
            for _ in range(int(padded) // 32)
        ],
        dtype=np.int64,
    )
    mmq_total_rows = int(expert_start_mmq32[-1])
    assert tile_expert.size == mmq_total_rows // 32
    return expert_start_mmq32, tile_expert, mmq_total_rows


def _run_q8_1_ds4_mmq32_selected_dual_gpu(
    fixture,
    *,
    source_remap: bool = False,
    layout: str = "raw",
    activation_passes: int = 1,
) -> np.ndarray:
    from hipengine.core.hip import get_hip_runtime

    runtime = get_hip_runtime()
    library = build_gguf_q4_k_q8_1_selected_prefill(load=True)
    host_out = np.zeros(
        (fixture.compact_rows, fixture.out_features_a + fixture.out_features_b),
        dtype=np.uint16,
    )
    compact_to_source = np.arange(fixture.compact_rows, dtype=np.int64)
    source_x = fixture.x_host
    if source_remap:
        compact_to_source = compact_to_source[::-1].copy()
        source_x = fixture.x_host[::-1].copy()
    if activation_passes not in (1, 3):
        raise ValueError("activation_passes must be 1 or 3")
    q8_ds4 = pack_q8_1_mmq_ds4_from_bf16(source_x)
    if layout == "raw":
        qweight_a = fixture.qweight_a
        qweight_b = fixture.qweight_b
        launcher = gguf_q4_k_selected_dual_q8_1_ds4_mmq32_prefill_compact32_bf16_bf16_out
    elif layout == "x8":
        qweight_a = repack_gguf_q4_k_x8(fixture.qweight_a).tiles
        qweight_b = repack_gguf_q4_k_x8(fixture.qweight_b).tiles
        launcher = gguf_q4_k_x8_selected_dual_q8_1_ds4_mmq32_prefill_compact32_bf16_bf16_out
    elif layout == "t16":
        qweight_a = repack_gguf_q4_k_tile16(fixture.qweight_a).tiles
        qweight_b = repack_gguf_q4_k_tile16(fixture.qweight_b).tiles
        launcher = (
            gguf_q4_k_t16_selected_dual_q8_1_ds4_mmq32_prefill_compact32_bf16_bf16_out
            if activation_passes == 1
            else gguf_q4_k_t16_selected_dual_q8_1_ds4x3_mmq32_prefill_compact32_bf16_bf16_out
        )
    else:
        raise ValueError(f"unsupported MMQ32 weight layout: {layout}")
    if activation_passes == 3 and layout != "t16":
        raise ValueError("residual DS4 MMQ32 is currently implemented for T16")
    expert_start_mmq32, tile_expert, mmq_total_rows = _mmq32_metadata(fixture)

    bufs = []
    try:
        source_x_dev = malloc(source_x.nbytes, runtime=runtime)
        q8_ds4_dev = malloc(q8_ds4.nbytes * activation_passes, runtime=runtime)
        compact_to_source_dev = malloc(compact_to_source.nbytes, runtime=runtime)
        start_compact_dev = malloc(fixture.expert_start_compact.nbytes, runtime=runtime)
        start_mmq32_dev = malloc(expert_start_mmq32.nbytes, runtime=runtime)
        tile_expert_dev = malloc(tile_expert.nbytes, runtime=runtime)
        qweight_a_dev = malloc(qweight_a.nbytes, runtime=runtime)
        qweight_b_dev = malloc(qweight_b.nbytes, runtime=runtime)
        out_dev = malloc(host_out.nbytes, runtime=runtime)
        bufs.extend(
            (
                source_x_dev,
                q8_ds4_dev,
                compact_to_source_dev,
                start_compact_dev,
                start_mmq32_dev,
                tile_expert_dev,
                qweight_a_dev,
                qweight_b_dev,
                out_dev,
            )
        )
        for dev, arr in (
            (source_x_dev, source_x),
            (compact_to_source_dev, compact_to_source),
            (start_compact_dev, fixture.expert_start_compact),
            (start_mmq32_dev, expert_start_mmq32),
            (tile_expert_dev, tile_expert),
            (qweight_a_dev, qweight_a),
            (qweight_b_dev, qweight_b),
        ):
            copy_host_to_device(
                dev,
                host_array_ptr(np.ascontiguousarray(arr)),
                runtime=runtime,
            )
        if activation_passes == 1:
            copy_host_to_device(
                q8_ds4_dev,
                host_array_ptr(np.ascontiguousarray(q8_ds4)),
                runtime=runtime,
            )
        else:
            gguf_q8_1_mmq_ds4_pack_bf16_d4x3(
                source_x_dev.ptr,
                q8_ds4_dev.ptr,
                source_x.shape[0],
                fixture.in_features,
                library=library,
                runtime=runtime,
            )

        launch_args = [
            q8_ds4_dev.ptr,
            compact_to_source_dev.ptr,
            start_compact_dev.ptr,
            start_mmq32_dev.ptr,
            tile_expert_dev.ptr,
            qweight_a_dev.ptr,
            qweight_b_dev.ptr,
            out_dev.ptr,
            fixture.compact_rows,
        ]
        if activation_passes == 3:
            launch_args.append(source_x.shape[0])
        launch_args.extend(
            [
            fixture.in_features,
            fixture.out_features_a,
            fixture.out_features_b,
            fixture.num_experts,
            mmq_total_rows,
            ]
        )
        launcher(
            *launch_args,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(host_out), out_dev, runtime=runtime)
    finally:
        for buf in reversed(bufs):
            free(buf, runtime=runtime)

    return _decode_output(host_out, "bf16")


def _run_q8_1_ds4_wmma_selected_dual_gpu(fixture) -> np.ndarray:
    return _run_q8_1_ds4_variant_gpu(
        fixture,
        gguf_q4_k_selected_dual_q8_1_ds4_wmma_prefill_compact32_bf16_bf16_out,
    )


def _run_q8_1_ds4_wmma32_selected_dual_gpu(fixture) -> np.ndarray:
    return _run_q8_1_ds4_variant_gpu(
        fixture,
        gguf_q4_k_selected_dual_q8_1_ds4_wmma32_prefill_compact32_bf16_bf16_out,
    )


def _run_q8_1_ds4_wmma64_selected_dual_gpu(fixture) -> np.ndarray:
    return _run_q8_1_ds4_variant_gpu(
        fixture,
        gguf_q4_k_selected_dual_q8_1_ds4_wmma64_prefill_compact32_bf16_bf16_out,
    )


def _run_q8_1_ds4_wmma32_ldspack_selected_dual_gpu(fixture) -> np.ndarray:
    return _run_q8_1_ds4_variant_gpu(
        fixture,
        gguf_q4_k_selected_dual_q8_1_ds4_wmma32_ldspack_prefill_compact32_bf16_bf16_out,
    )


def _run_q8_1_ds4_wmma32_lds_selected_dual_gpu(fixture) -> np.ndarray:
    return _run_q8_1_ds4_variant_gpu(
        fixture,
        gguf_q4_k_selected_dual_q8_1_ds4_wmma32_lds_prefill_compact32_bf16_bf16_out,
    )


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize(
    ("counts", "in_features", "out_features_a", "out_features_b"),
    [
        pytest.param([4, 0, 5], 256, 16, 16, id="empty-middle-small-boundary"),
        pytest.param([0, 17, 31], 512, 32, 48, id="empty-first-multi-block"),
    ],
)
def test_q4_k_q8_1_selected_prefill_bf16_matches_quantized_cpu_reference(
    counts: list[int], in_features: int, out_features_a: int, out_features_b: int
) -> None:
    fixture = _build_compact_fixture(
        counts=counts,
        in_features=in_features,
        out_features_a=out_features_a,
        out_features_b=out_features_b,
        dtype="bf16",
        seed=7,
    )
    actual = _run_q8_1_selected_dual_gpu(fixture)
    expected = _q8_1_selected_reference(fixture)
    np.testing.assert_allclose(actual, expected, **_TOLERANCE_BF16)


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize(
    ("counts", "in_features", "out_features_a", "out_features_b"),
    [
        pytest.param([4, 0, 5], 256, 16, 16, id="empty-middle-small-boundary"),
        pytest.param([0, 17, 31], 512, 32, 48, id="empty-first-multi-block"),
    ],
)
def test_q4_k_q8_1_ds4_selected_prefill_bf16_matches_ds4_cpu_reference(
    counts: list[int], in_features: int, out_features_a: int, out_features_b: int
) -> None:
    fixture = _build_compact_fixture(
        counts=counts,
        in_features=in_features,
        out_features_a=out_features_a,
        out_features_b=out_features_b,
        dtype="bf16",
        seed=7,
    )
    actual = _run_q8_1_ds4_selected_dual_gpu(fixture)
    expected = _q8_1_ds4_selected_reference(fixture)
    np.testing.assert_allclose(actual, expected, **_TOLERANCE_BF16)


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize(
    ("counts", "in_features", "out_features_a", "out_features_b", "source_remap"),
    [
        pytest.param([4, 0, 5], 256, 32, 32, False, id="empty-middle-tail"),
        pytest.param(
            [0, 17, 31],
            512,
            32,
            64,
            True,
            id="empty-first-multi-block-source-remap",
        ),
    ],
)
def test_q4_k_q8_1_ds4_mmq32_selected_prefill_bf16_matches_ds4_cpu_reference(
    counts: list[int],
    in_features: int,
    out_features_a: int,
    out_features_b: int,
    source_remap: bool,
) -> None:
    fixture = _build_compact_fixture(
        counts=counts,
        in_features=in_features,
        out_features_a=out_features_a,
        out_features_b=out_features_b,
        dtype="bf16",
        seed=23,
    )
    actual = _run_q8_1_ds4_mmq32_selected_dual_gpu(
        fixture, source_remap=source_remap
    )
    expected = _q8_1_ds4_selected_reference(fixture)
    np.testing.assert_allclose(actual, expected, **_TOLERANCE_BF16)
    assert _max_softmax_kl(expected, actual) <= 0.05
    assert float(
        np.mean(np.argmax(expected, axis=-1) == np.argmax(actual, axis=-1))
    ) >= 0.9


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize(
    ("counts", "in_features", "out_features_a", "out_features_b", "source_remap"),
    [
        pytest.param([4, 0, 5], 256, 32, 32, False, id="empty-middle-tail"),
        pytest.param(
            [0, 17, 31],
            512,
            32,
            64,
            True,
            id="empty-first-multi-block-source-remap",
        ),
    ],
)
def test_q4_k_x8_q8_1_ds4_mmq32_selected_prefill_matches_raw_mmq32(
    counts: list[int],
    in_features: int,
    out_features_a: int,
    out_features_b: int,
    source_remap: bool,
) -> None:
    fixture = _build_compact_fixture(
        counts=counts,
        in_features=in_features,
        out_features_a=out_features_a,
        out_features_b=out_features_b,
        dtype="bf16",
        seed=23,
    )
    raw = _run_q8_1_ds4_mmq32_selected_dual_gpu(
        fixture, source_remap=source_remap
    )
    actual = _run_q8_1_ds4_mmq32_selected_dual_gpu(
        fixture, source_remap=source_remap, layout="x8"
    )
    expected = _q8_1_ds4_selected_reference(fixture)
    np.testing.assert_array_equal(actual, raw)
    np.testing.assert_allclose(actual, expected, **_TOLERANCE_BF16)
    assert _max_softmax_kl(expected, actual) <= 0.05
    assert float(
        np.mean(np.argmax(expected, axis=-1) == np.argmax(actual, axis=-1))
    ) >= 0.9


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize(
    ("counts", "in_features", "out_features_a", "out_features_b", "source_remap"),
    [
        pytest.param([4, 0, 5], 256, 32, 32, False, id="empty-middle-tail"),
        pytest.param(
            [0, 17, 31],
            512,
            32,
            64,
            True,
            id="empty-first-multi-block-source-remap",
        ),
    ],
)
def test_q4_k_t16_q8_1_ds4_mmq32_selected_prefill_matches_raw_mmq32(
    counts: list[int],
    in_features: int,
    out_features_a: int,
    out_features_b: int,
    source_remap: bool,
) -> None:
    fixture = _build_compact_fixture(
        counts=counts,
        in_features=in_features,
        out_features_a=out_features_a,
        out_features_b=out_features_b,
        dtype="bf16",
        seed=23,
    )
    raw = _run_q8_1_ds4_mmq32_selected_dual_gpu(
        fixture, source_remap=source_remap
    )
    actual = _run_q8_1_ds4_mmq32_selected_dual_gpu(
        fixture, source_remap=source_remap, layout="t16"
    )
    expected = _q8_1_ds4_selected_reference(fixture)
    np.testing.assert_array_equal(actual, raw)
    np.testing.assert_allclose(actual, expected, **_TOLERANCE_BF16)
    assert _max_softmax_kl(expected, actual) <= 0.05
    assert float(
        np.mean(np.argmax(expected, axis=-1) == np.argmax(actual, axis=-1))
    ) >= 0.9


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_q4_k_t16_q8_1_ds4x3_mmq32_reduces_exact_projection_error() -> None:
    fixture = _build_compact_fixture(
        counts=[4, 0, 5],
        in_features=256,
        out_features_a=32,
        out_features_b=32,
        dtype="bf16",
        seed=23,
    )
    primary = _run_q8_1_ds4_mmq32_selected_dual_gpu(
        fixture, layout="t16"
    )
    residual = _run_q8_1_ds4_mmq32_selected_dual_gpu(
        fixture, layout="t16", activation_passes=3
    )
    expected = fixture.reference
    expected_f64 = expected.astype(np.float64)
    primary_f64 = primary.astype(np.float64)
    residual_f64 = residual.astype(np.float64)
    expected_norm = max(float(np.linalg.norm(expected_f64)), 1e-12)
    primary_relative_l2 = float(
        np.linalg.norm(primary_f64 - expected_f64) / expected_norm
    )
    residual_relative_l2 = float(
        np.linalg.norm(residual_f64 - expected_f64) / expected_norm
    )

    assert residual_relative_l2 <= 0.7 * primary_relative_l2
    assert _max_softmax_kl(expected, residual) <= 0.05
    assert float(
        np.mean(np.argmax(expected, axis=-1) == np.argmax(residual, axis=-1))
    ) >= 0.9


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize(
    ("counts", "in_features", "out_features_a", "out_features_b"),
    [
        pytest.param([4, 0, 5], 256, 16, 16, id="empty-middle-small-boundary"),
        pytest.param([0, 17, 31], 512, 32, 48, id="empty-first-multi-block"),
    ],
)
def test_q4_k_q8_1_ds4_wmma_selected_prefill_bf16_matches_ds4_cpu_reference(
    counts: list[int], in_features: int, out_features_a: int, out_features_b: int
) -> None:
    fixture = _build_compact_fixture(
        counts=counts,
        in_features=in_features,
        out_features_a=out_features_a,
        out_features_b=out_features_b,
        dtype="bf16",
        seed=7,
    )
    actual = _run_q8_1_ds4_wmma_selected_dual_gpu(fixture)
    expected = _q8_1_ds4_selected_reference(fixture)
    np.testing.assert_allclose(actual, expected, **_TOLERANCE_BF16)


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize(
    ("counts", "in_features", "out_features_a", "out_features_b"),
    [
        pytest.param([4, 0, 5], 256, 16, 16, id="empty-middle-small-boundary"),
        pytest.param([0, 17, 31], 512, 32, 48, id="empty-first-multi-block"),
    ],
)
def test_q4_k_q8_1_ds4_wmma32_selected_prefill_bf16_matches_ds4_cpu_reference(
    counts: list[int], in_features: int, out_features_a: int, out_features_b: int
) -> None:
    fixture = _build_compact_fixture(
        counts=counts,
        in_features=in_features,
        out_features_a=out_features_a,
        out_features_b=out_features_b,
        dtype="bf16",
        seed=7,
    )
    actual = _run_q8_1_ds4_wmma32_selected_dual_gpu(fixture)
    expected = _q8_1_ds4_selected_reference(fixture)
    np.testing.assert_allclose(actual, expected, **_TOLERANCE_BF16)


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize(
    ("counts", "in_features", "out_features_a", "out_features_b"),
    [
        pytest.param([4, 0, 5], 256, 16, 16, id="empty-middle-small-boundary"),
        pytest.param([0, 17, 31], 512, 32, 48, id="empty-first-multi-block"),
    ],
)
def test_q4_k_q8_1_ds4_wmma64_selected_prefill_bf16_matches_ds4_cpu_reference(
    counts: list[int], in_features: int, out_features_a: int, out_features_b: int
) -> None:
    fixture = _build_compact_fixture(
        counts=counts,
        in_features=in_features,
        out_features_a=out_features_a,
        out_features_b=out_features_b,
        dtype="bf16",
        seed=7,
    )
    actual = _run_q8_1_ds4_wmma64_selected_dual_gpu(fixture)
    expected = _q8_1_ds4_selected_reference(fixture)
    np.testing.assert_allclose(actual, expected, **_TOLERANCE_BF16)


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize(
    ("counts", "in_features", "out_features_a", "out_features_b"),
    [
        pytest.param([4, 0, 5], 256, 16, 16, id="empty-middle-small-boundary"),
        pytest.param([0, 17, 31], 512, 32, 48, id="empty-first-multi-block"),
    ],
)
def test_q4_k_q8_1_ds4_preview_wmma32_selected_prefill_bf16_matches_ds4_cpu_reference(
    counts: list[int], in_features: int, out_features_a: int, out_features_b: int
) -> None:
    fixture = _build_compact_fixture(
        counts=counts,
        in_features=in_features,
        out_features_a=out_features_a,
        out_features_b=out_features_b,
        dtype="bf16",
        seed=7,
    )
    actual = _run_q8_1_ds4_preview_wmma32_selected_dual_gpu(fixture)
    expected = _q8_1_ds4_selected_reference(fixture)
    np.testing.assert_allclose(actual, expected, **_TOLERANCE_BF16)


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize(
    ("counts", "in_features", "out_features_a", "out_features_b"),
    [
        pytest.param([4, 0, 5], 256, 16, 16, id="empty-middle-small-boundary"),
        pytest.param([0, 17, 31], 512, 32, 48, id="empty-first-multi-block"),
    ],
)
def test_q4_k_q8_1_ds4_wmma32_ldspack_selected_prefill_bf16_matches_ds4_cpu_reference(
    counts: list[int], in_features: int, out_features_a: int, out_features_b: int
) -> None:
    fixture = _build_compact_fixture(
        counts=counts,
        in_features=in_features,
        out_features_a=out_features_a,
        out_features_b=out_features_b,
        dtype="bf16",
        seed=7,
    )
    actual = _run_q8_1_ds4_wmma32_ldspack_selected_dual_gpu(fixture)
    expected = _q8_1_ds4_selected_reference(fixture)
    np.testing.assert_allclose(actual, expected, **_TOLERANCE_BF16)


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize(
    ("counts", "in_features", "out_features_a", "out_features_b"),
    [
        pytest.param([4, 0, 5], 256, 16, 16, id="empty-middle-small-boundary"),
        pytest.param([0, 17, 31], 512, 32, 48, id="empty-first-multi-block"),
    ],
)
def test_q4_k_q8_1_ds4_wmma32_lds_selected_prefill_bf16_matches_ds4_cpu_reference(
    counts: list[int], in_features: int, out_features_a: int, out_features_b: int
) -> None:
    fixture = _build_compact_fixture(
        counts=counts,
        in_features=in_features,
        out_features_a=out_features_a,
        out_features_b=out_features_b,
        dtype="bf16",
        seed=7,
    )
    actual = _run_q8_1_ds4_wmma32_lds_selected_dual_gpu(fixture)
    expected = _q8_1_ds4_selected_reference(fixture)
    np.testing.assert_allclose(actual, expected, **_TOLERANCE_BF16)


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize(
    ("residual_passes", "rowvec"),
    [
        (1, False),
        (1, True),
        (2, False),
        (3, False),
    ],
)
def test_q6_k_t16_ds4x3_f32_mmq64x32_matches_cpu_quality_gate(
    residual_passes: int,
    rowvec: bool,
) -> None:
    from hipengine.core.hip import get_hip_runtime
    from tests.test_gguf_k_t16_selected_wmma_prefill import (
        _build_compact_t16_fixture,
    )

    fixture = _build_compact_t16_fixture(
        quant="gguf_q6_k_t16_v1",
        counts=[0, 7, 18, 33],
        in_features=512,
        out_features=64,
        dtype="bf16",
        seed=23,
    )
    counts = np.diff(fixture.expert_start_compact)
    padded = ((counts + 31) // 32) * 32
    expert_start_mmq32 = np.zeros(
        fixture.num_experts + 1, dtype=np.int64
    )
    expert_start_mmq32[1:] = np.cumsum(padded, dtype=np.int64)
    mmq_total_rows = int(expert_start_mmq32[-1])
    tile_expert = np.asarray(
        [
            expert
            for expert, padded_rows in enumerate(padded)
            for _ in range(int(padded_rows) // 32)
        ],
        dtype=np.int64,
    )

    runtime = get_hip_runtime()
    library = build_gguf_q4_k_q8_1_selected_prefill(load=True)
    q8_bytes = (
        fixture.compact_rows
        * (fixture.in_features // 128)
        * 160
        * 3
    )
    host_out = np.zeros(
        (fixture.compact_rows, fixture.out_features), dtype=np.uint16
    )
    host_baseline = np.zeros_like(host_out) if rowvec else None
    bufs = []
    try:
        arrays = (
            fixture.x_host,
            expert_start_mmq32,
            fixture.expert_start_compact,
            tile_expert,
            fixture.tiles,
        )
        for arr in arrays:
            dev = malloc(arr.nbytes, runtime=runtime)
            copy_host_to_device(
                dev,
                host_array_ptr(np.ascontiguousarray(arr)),
                runtime=runtime,
            )
            bufs.append(dev)
        q8_dev = malloc(q8_bytes, runtime=runtime)
        out_dev = malloc(host_out.nbytes, runtime=runtime)
        bufs.extend((q8_dev, out_dev))

        gguf_q8_1_mmq_ds4_f32_pack_bf16_d4x3(
            bufs[0].ptr,
            q8_dev.ptr,
            fixture.compact_rows,
            fixture.in_features,
            residual_passes=residual_passes,
            library=library,
            runtime=runtime,
        )
        gguf_q6_k_t16_selected_q8_1_ds4x3_f32_mmq64x32_prefill_compact32_bf16_bf16_out(
            q8_dev.ptr,
            bufs[2].ptr,
            bufs[1].ptr,
            bufs[3].ptr,
            bufs[4].ptr,
            out_dev.ptr,
            fixture.compact_rows,
            fixture.in_features,
            fixture.out_features,
            fixture.num_experts,
            mmq_total_rows,
            residual_passes=residual_passes,
            rowvec=rowvec,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(host_out), out_dev, runtime=runtime)
        if rowvec:
            baseline_dev = malloc(host_out.nbytes, runtime=runtime)
            bufs.append(baseline_dev)
            gguf_q6_k_t16_selected_q8_1_ds4x3_f32_mmq64x32_prefill_compact32_bf16_bf16_out(
                q8_dev.ptr,
                bufs[2].ptr,
                bufs[1].ptr,
                bufs[3].ptr,
                bufs[4].ptr,
                baseline_dev.ptr,
                fixture.compact_rows,
                fixture.in_features,
                fixture.out_features,
                fixture.num_experts,
                mmq_total_rows,
                residual_passes=residual_passes,
                library=library,
                runtime=runtime,
            )
            runtime.device_synchronize()
            copy_device_to_host(
                host_array_ptr(host_baseline),
                baseline_dev,
                runtime=runtime,
            )
    finally:
        for buf in reversed(bufs):
            free(buf, runtime=runtime)

    actual = _bf16_bits_to_float32(host_out)
    if host_baseline is not None:
        assert np.array_equal(host_out, host_baseline)
    assert np.isfinite(actual).all()
    assert _max_softmax_kl(fixture.reference, actual) <= 0.05
    assert np.mean(
        fixture.reference.argmax(axis=-1) == actual.argmax(axis=-1)
    ) >= 0.9
    relative_l2 = float(
        np.linalg.norm(actual - fixture.reference)
        / np.linalg.norm(fixture.reference)
    )
    assert relative_l2 <= 5.0e-3


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize(
    (
        "residual_passes",
        "split16",
        "rowvec",
        "wave_cols",
        "single_wave_cols",
        "direct_wave_decode",
        "single_direct_wave_decode",
    ),
    [
        (1, False, False, False, False, False, False),
        (2, False, False, False, False, False, False),
        (3, False, False, False, False, False, False),
        (1, False, True, False, False, False, False),
        (1, False, True, False, True, False, False),
        (1, False, True, False, True, False, True),
        (1, True, False, False, False, False, False),
        (1, True, True, False, False, False, False),
        (1, True, True, True, False, False, False),
        (1, True, True, True, False, True, False),
    ],
)
def test_q4_k_t16_ds4_f32_mmq64x32_matches_cpu_quality_gate(
    residual_passes: int,
    split16: bool,
    rowvec: bool,
    wave_cols: bool,
    single_wave_cols: bool,
    direct_wave_decode: bool,
    single_direct_wave_decode: bool,
) -> None:
    from hipengine.core.hip import get_hip_runtime

    fixture = _build_compact_fixture(
        counts=[0, 7, 18, 33],
        in_features=512,
        out_features_a=128 if split16 else 64,
        out_features_b=128 if split16 else 64,
        dtype="bf16",
        seed=29,
    )
    counts = np.diff(fixture.expert_start_compact)
    padded = ((counts + 31) // 32) * 32
    expert_start_mmq32 = np.zeros(
        fixture.num_experts + 1, dtype=np.int64
    )
    expert_start_mmq32[1:] = np.cumsum(padded, dtype=np.int64)
    mmq_total_rows = int(expert_start_mmq32[-1])
    tile_expert = np.asarray(
        [
            expert
            for expert, padded_rows in enumerate(padded)
            for _ in range(int(padded_rows) // 32)
        ],
        dtype=np.int64,
    )
    compact_to_source = np.arange(fixture.compact_rows, dtype=np.int64)
    tiles_a = np.ascontiguousarray(
        repack_gguf_q4_k_tile16(fixture.qweight_a).tiles
    )
    tiles_b = np.ascontiguousarray(
        repack_gguf_q4_k_tile16(fixture.qweight_b).tiles
    )

    runtime = get_hip_runtime()
    library = build_gguf_q4_k_q8_1_selected_prefill(load=True)
    q8_bytes = (
        fixture.compact_rows
        * (fixture.in_features // 128)
        * 160
        * 3
    )
    host_out = np.zeros(
        (
            fixture.compact_rows,
            fixture.out_features_a + fixture.out_features_b,
        ),
        dtype=np.uint16,
    )
    host_baseline = np.zeros_like(host_out) if rowvec else None
    host_single = (
        np.zeros(
            (fixture.compact_rows, fixture.out_features_a),
            dtype=np.uint16,
        )
        if rowvec and not split16
        else None
    )
    bufs = []
    try:
        arrays = (
            fixture.x_host,
            compact_to_source,
            fixture.expert_start_compact,
            expert_start_mmq32,
            tile_expert,
            tiles_a,
            tiles_b,
        )
        for arr in arrays:
            dev = malloc(arr.nbytes, runtime=runtime)
            copy_host_to_device(
                dev,
                host_array_ptr(np.ascontiguousarray(arr)),
                runtime=runtime,
            )
            bufs.append(dev)
        q8_dev = malloc(q8_bytes, runtime=runtime)
        out_dev = malloc(host_out.nbytes, runtime=runtime)
        bufs.extend((q8_dev, out_dev))

        gguf_q8_1_mmq_ds4_f32_pack_bf16_d4x3(
            bufs[0].ptr,
            q8_dev.ptr,
            fixture.compact_rows,
            fixture.in_features,
            residual_passes=residual_passes,
            split16=split16,
            library=library,
            runtime=runtime,
        )
        gguf_q4_k_t16_selected_dual_q8_1_ds4x3_f32_mmq64x32_prefill_compact32_bf16_bf16_out(
            q8_dev.ptr,
            bufs[1].ptr,
            bufs[2].ptr,
            bufs[3].ptr,
            bufs[4].ptr,
            bufs[5].ptr,
            bufs[6].ptr,
            out_dev.ptr,
            fixture.compact_rows,
            fixture.compact_rows,
            fixture.in_features,
            fixture.out_features_a,
            fixture.out_features_b,
            fixture.num_experts,
            mmq_total_rows,
            residual_passes=residual_passes,
            split16=split16,
            rowvec=rowvec,
            wave_cols=wave_cols,
            direct_wave_decode=direct_wave_decode,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(host_out), out_dev, runtime=runtime)
        if rowvec:
            baseline_dev = malloc(host_out.nbytes, runtime=runtime)
            bufs.append(baseline_dev)
            gguf_q4_k_t16_selected_dual_q8_1_ds4x3_f32_mmq64x32_prefill_compact32_bf16_bf16_out(
                q8_dev.ptr,
                bufs[1].ptr,
                bufs[2].ptr,
                bufs[3].ptr,
                bufs[4].ptr,
                bufs[5].ptr,
                bufs[6].ptr,
                baseline_dev.ptr,
                fixture.compact_rows,
                fixture.compact_rows,
                fixture.in_features,
                fixture.out_features_a,
                fixture.out_features_b,
                fixture.num_experts,
                mmq_total_rows,
                residual_passes=residual_passes,
                split16=split16,
                library=library,
                runtime=runtime,
            )
            runtime.device_synchronize()
            copy_device_to_host(
                host_array_ptr(host_baseline),
                baseline_dev,
                runtime=runtime,
            )
        if host_single is not None:
            single_dev = malloc(host_single.nbytes, runtime=runtime)
            bufs.append(single_dev)
            gguf_q4_k_t16_selected_q8_1_ds4_f32_mmq64x32_prefill_compact32_bf16_bf16_out(
                q8_dev.ptr,
                bufs[2].ptr,
                bufs[3].ptr,
                bufs[4].ptr,
                bufs[5].ptr,
                single_dev.ptr,
                fixture.compact_rows,
                fixture.in_features,
                fixture.out_features_a,
                fixture.num_experts,
                mmq_total_rows,
                rowvec=True,
                wave_cols=single_wave_cols,
                **(
                    {"direct_wave_decode": True}
                    if single_direct_wave_decode
                    else {}
                ),
                library=library,
                runtime=runtime,
            )
            runtime.device_synchronize()
            copy_device_to_host(
                host_array_ptr(host_single),
                single_dev,
                runtime=runtime,
            )
    finally:
        for buf in reversed(bufs):
            free(buf, runtime=runtime)

    actual = _bf16_bits_to_float32(host_out)
    if host_baseline is not None:
        assert np.array_equal(host_out, host_baseline)
    if host_single is not None:
        assert np.array_equal(
            host_single,
            host_out[:, : fixture.out_features_a],
        )
    assert np.isfinite(actual).all()
    assert _max_softmax_kl(fixture.reference, actual) <= 0.05
    assert np.mean(
        fixture.reference.argmax(axis=-1) == actual.argmax(axis=-1)
    ) >= 0.9
