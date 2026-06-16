"""Correctness tests for the P9.C14 Q4T16 selected-dual WMMA prototype."""

from __future__ import annotations

import numpy as np
import pytest

from hipengine.core.memory import copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc
from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_t16_selected_prefill import (
    build_gguf_q4_k_t16_selected_prefill,
    gguf_q4_k_t16_selected_dual_q8_1_ds4_wmma32_prefill_compact32_bf16_bf16_out,
    gguf_q4_k_t16_selected_dual_wmma_prefill_compact32_bf16_bf16_out,
    gguf_q4_k_t16_selected_dual_wmma_prefill_compact32_fp16_fp16_out,
    plan_gguf_q4_k_t16_selected_prefill_build,
)
from hipengine.kernels.registry import resolve
from hipengine.quant.gguf_q4_k import (
    GGUF_Q4_K_TILE16_BLOCK_BYTES,
    gguf_q4_k_mmq_tile16_preview_matmul,
    pack_gguf_q4_k_mmq_tile16_preview,
    pack_q8_1_mmq_ds4_from_bf16,
    repack_gguf_q4_k_tile16,
)
from tests.test_gguf_q4_k_selected_wmma_prefill import (
    _TOLERANCE_BF16,
    _TOLERANCE_FP16,
    _build_compact_fixture,
    _decode_output,
    _hip_available,
)


# ---------------------------------------------------------------------------
# No-GPU surface checks.
# ---------------------------------------------------------------------------


def test_gguf_q4_k_t16_selected_wmma_registry_and_build_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HIPENGINE_GGUF_SELECTED_WMMA_LAUNCH_BOUNDS", raising=False)
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="moe_linear",
            quant="gguf_q4_k_t16_v1",
            variant="selected_dual_wmma_prefill_compact32_bf16_bf16_out",
        )
        is gguf_q4_k_t16_selected_dual_wmma_prefill_compact32_bf16_bf16_out
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="moe_linear",
            quant="gguf_q4_k_t16_v1",
            variant="selected_dual_wmma_prefill_compact_bf16_bf16_out",
        )
        is gguf_q4_k_t16_selected_dual_wmma_prefill_compact32_bf16_bf16_out
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="moe_linear",
            quant="gguf_q4_k_t16_v1",
            variant="selected_dual_q8_1_ds4_wmma32_prefill_compact32_bf16_bf16_out",
        )
        is gguf_q4_k_t16_selected_dual_q8_1_ds4_wmma32_prefill_compact32_bf16_bf16_out
    )

    artifact = plan_gguf_q4_k_t16_selected_prefill_build(compiler_version="test-compiler")
    assert artifact.output_path.name == "gguf_q4_k_t16_selected_prefill.so"
    assert "gguf_q4_k_t16_selected_prefill" in str(artifact.output_path)
    assert any(path.name == "gguf_q4_k_t16_selected_prefill.hip" for path in artifact.sources)
    assert "-DHIPENGINE_SELECTED_WMMA_LAUNCH_BOUNDS=4" not in artifact.flags

    dry_run = build_gguf_q4_k_t16_selected_prefill(dry_run=True, compiler_version="test-compiler")
    assert dry_run.output_path == artifact.output_path

    monkeypatch.setenv("HIPENGINE_GGUF_SELECTED_WMMA_LAUNCH_BOUNDS", "4")
    lb4 = plan_gguf_q4_k_t16_selected_prefill_build(compiler_version="test-compiler")
    assert "-DHIPENGINE_SELECTED_WMMA_LAUNCH_BOUNDS=4" in lb4.flags
    assert lb4.cache_key != artifact.cache_key


def test_gguf_q4_k_t16_selected_wmma_wrapper_validates_common_contract() -> None:
    kwargs = dict(
        x_ptr=1,
        expert_start_compact_ptr=2,
        expert_start_wmma_ptr=3,
        tile_expert_ptr=4,
        tiles_a_ptr=5,
        tiles_b_ptr=6,
        out_ptr=7,
        compact_rows=17,
        in_features=256,
        out_features_a=32,
        out_features_b=32,
        num_experts=2,
        wmma_total_rows=32,
    )

    with pytest.raises(ValueError, match="compact_rows"):
        gguf_q4_k_t16_selected_dual_wmma_prefill_compact32_bf16_bf16_out(
            **{**kwargs, "compact_rows": 0}
        )
    with pytest.raises(ValueError, match="Q4_K block size 256"):
        gguf_q4_k_t16_selected_dual_wmma_prefill_compact32_bf16_bf16_out(
            **{**kwargs, "in_features": 128}
        )
    with pytest.raises(ValueError, match="out_features_a.*multiple of 16"):
        gguf_q4_k_t16_selected_dual_wmma_prefill_compact32_bf16_bf16_out(
            **{**kwargs, "out_features_a": 24}
        )
    with pytest.raises(ValueError, match="out_features_b.*multiple of 16"):
        gguf_q4_k_t16_selected_dual_wmma_prefill_compact32_bf16_bf16_out(
            **{**kwargs, "out_features_b": 24}
        )
    with pytest.raises(ValueError, match="wmma_total_rows.*multiple of 16"):
        gguf_q4_k_t16_selected_dual_wmma_prefill_compact32_bf16_bf16_out(
            **{**kwargs, "wmma_total_rows": 31}
        )


# ---------------------------------------------------------------------------
# HIP correctness fixtures.
# ---------------------------------------------------------------------------


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


def _run_t16_selected_dual_gpu(fixture, dtype: str) -> np.ndarray:
    from hipengine.core.hip import get_hip_runtime

    runtime = get_hip_runtime()
    library = build_gguf_q4_k_t16_selected_prefill(load=True)
    wrapper = (
        gguf_q4_k_t16_selected_dual_wmma_prefill_compact32_bf16_bf16_out
        if dtype == "bf16"
        else gguf_q4_k_t16_selected_dual_wmma_prefill_compact32_fp16_fp16_out
    )
    out_dtype = np.uint16 if dtype == "bf16" else np.float16
    host_out = np.zeros(
        (fixture.compact_rows, fixture.out_features_a + fixture.out_features_b),
        dtype=out_dtype,
    )
    tiles_a = repack_gguf_q4_k_tile16(fixture.qweight_a).tiles
    tiles_b = repack_gguf_q4_k_tile16(fixture.qweight_b).tiles
    assert tiles_a.shape[-1] == GGUF_Q4_K_TILE16_BLOCK_BYTES
    assert tiles_b.shape[-1] == GGUF_Q4_K_TILE16_BLOCK_BYTES

    bufs = []
    try:
        x_dev = malloc(fixture.x_host.nbytes, runtime=runtime)
        start_compact_dev = malloc(fixture.expert_start_compact.nbytes, runtime=runtime)
        start_wmma_dev = malloc(fixture.expert_start_wmma.nbytes, runtime=runtime)
        tile_expert_dev = malloc(fixture.tile_expert.nbytes, runtime=runtime)
        tiles_a_dev = malloc(tiles_a.nbytes, runtime=runtime)
        tiles_b_dev = malloc(tiles_b.nbytes, runtime=runtime)
        out_dev = malloc(host_out.nbytes, runtime=runtime)
        bufs.extend((x_dev, start_compact_dev, start_wmma_dev, tile_expert_dev, tiles_a_dev, tiles_b_dev, out_dev))
        for dev, arr in (
            (x_dev, fixture.x_host),
            (start_compact_dev, fixture.expert_start_compact),
            (start_wmma_dev, fixture.expert_start_wmma),
            (tile_expert_dev, fixture.tile_expert),
            (tiles_a_dev, tiles_a),
            (tiles_b_dev, tiles_b),
        ):
            copy_host_to_device(dev, host_array_ptr(np.ascontiguousarray(arr)), runtime=runtime)

        wrapper(
            x_dev.ptr,
            start_compact_dev.ptr,
            start_wmma_dev.ptr,
            tile_expert_dev.ptr,
            tiles_a_dev.ptr,
            tiles_b_dev.ptr,
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

    return _decode_output(host_out, dtype)


def _run_t16_q8_1_ds4_wmma32_selected_dual_gpu(fixture) -> np.ndarray:
    from hipengine.core.hip import get_hip_runtime

    runtime = get_hip_runtime()
    library = build_gguf_q4_k_t16_selected_prefill(load=True)
    host_out = np.zeros(
        (fixture.compact_rows, fixture.out_features_a + fixture.out_features_b),
        dtype=np.uint16,
    )
    q8_ds4 = pack_q8_1_mmq_ds4_from_bf16(fixture.x_host)
    tiles_a = repack_gguf_q4_k_tile16(fixture.qweight_a).tiles
    tiles_b = repack_gguf_q4_k_tile16(fixture.qweight_b).tiles

    bufs = []
    try:
        q8_dev = malloc(q8_ds4.nbytes, runtime=runtime)
        start_compact_dev = malloc(fixture.expert_start_compact.nbytes, runtime=runtime)
        start_wmma_dev = malloc(fixture.expert_start_wmma.nbytes, runtime=runtime)
        tile_expert_dev = malloc(fixture.tile_expert.nbytes, runtime=runtime)
        tiles_a_dev = malloc(tiles_a.nbytes, runtime=runtime)
        tiles_b_dev = malloc(tiles_b.nbytes, runtime=runtime)
        out_dev = malloc(host_out.nbytes, runtime=runtime)
        bufs.extend((q8_dev, start_compact_dev, start_wmma_dev, tile_expert_dev, tiles_a_dev, tiles_b_dev, out_dev))
        for dev, arr in (
            (q8_dev, q8_ds4),
            (start_compact_dev, fixture.expert_start_compact),
            (start_wmma_dev, fixture.expert_start_wmma),
            (tile_expert_dev, fixture.tile_expert),
            (tiles_a_dev, tiles_a),
            (tiles_b_dev, tiles_b),
        ):
            copy_host_to_device(dev, host_array_ptr(np.ascontiguousarray(arr)), runtime=runtime)

        gguf_q4_k_t16_selected_dual_q8_1_ds4_wmma32_prefill_compact32_bf16_bf16_out(
            q8_dev.ptr,
            start_compact_dev.ptr,
            start_wmma_dev.ptr,
            tile_expert_dev.ptr,
            tiles_a_dev.ptr,
            tiles_b_dev.ptr,
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


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize(
    ("counts", "in_features", "out_features_a", "out_features_b"),
    [
        pytest.param([4, 0, 5], 256, 16, 16, id="empty-middle-small-boundary"),
        pytest.param([16, 17, 31], 256, 32, 32, id="exact-plus-padding"),
        pytest.param([0, 33, 1, 16], 512, 48, 32, id="empty-first-multi-block"),
        pytest.param([7, 18, 0, 33], 512, 64, 16, id="empty-third-wide-gate"),
    ],
)
def test_p9_c14_q4_k_t16_selected_wmma_bf16_matches_cpu_selected_reference(
    counts: list[int], in_features: int, out_features_a: int, out_features_b: int
) -> None:
    fixture = _build_compact_fixture(
        counts=counts,
        in_features=in_features,
        out_features_a=out_features_a,
        out_features_b=out_features_b,
        dtype="bf16",
    )
    actual = _run_t16_selected_dual_gpu(fixture, "bf16")
    np.testing.assert_allclose(actual, fixture.reference, **_TOLERANCE_BF16)


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize(
    ("counts", "in_features", "out_features_a", "out_features_b"),
    [
        pytest.param([4, 0, 5], 256, 16, 16, id="empty-middle-small-boundary"),
        pytest.param([16, 17, 31], 512, 32, 32, id="exact-plus-padding-multiblock"),
    ],
)
def test_q4_k_t16_q8_1_ds4_wmma32_selected_prefill_bf16_matches_ds4_cpu_reference(
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
    actual = _run_t16_q8_1_ds4_wmma32_selected_dual_gpu(fixture)
    expected = _q8_1_ds4_selected_reference(fixture)
    np.testing.assert_allclose(actual, expected, **_TOLERANCE_BF16)


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize(
    ("counts", "in_features", "out_features_a", "out_features_b"),
    [
        pytest.param([5, 11, 0, 23], 256, 32, 16, id="fp16-uneven-empty"),
        pytest.param([0, 16, 17], 512, 48, 48, id="fp16-empty-first-wide"),
    ],
)
def test_p9_c14_q4_k_t16_selected_wmma_fp16_matches_cpu_selected_reference(
    counts: list[int], in_features: int, out_features_a: int, out_features_b: int
) -> None:
    fixture = _build_compact_fixture(
        counts=counts,
        in_features=in_features,
        out_features_a=out_features_a,
        out_features_b=out_features_b,
        dtype="fp16",
        seed=5,
    )
    actual = _run_t16_selected_dual_gpu(fixture, "fp16")
    np.testing.assert_allclose(actual, fixture.reference, **_TOLERANCE_FP16)


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_p9_c14_q4_k_t16_selected_wmma_runs_exported_bf16_symbol_on_w7900() -> None:
    fixture = _build_compact_fixture(
        counts=[1, 0],
        in_features=256,
        out_features_a=16,
        out_features_b=16,
        dtype="bf16",
        seed=9,
    )
    actual = _run_t16_selected_dual_gpu(fixture, "bf16")
    np.testing.assert_allclose(actual, fixture.reference, **_TOLERANCE_BF16)
