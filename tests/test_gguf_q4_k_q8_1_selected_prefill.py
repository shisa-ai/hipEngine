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
)
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


# ---------------------------------------------------------------------------
# No-GPU surface checks.
# ---------------------------------------------------------------------------


def test_gguf_q4_k_q8_1_selected_prefill_registry_and_build_plan() -> None:
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
