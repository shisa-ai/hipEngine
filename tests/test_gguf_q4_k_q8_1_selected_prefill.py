"""Correctness tests for the diagnostic Q4_K x Q8_1 selected prefill prototype."""

from __future__ import annotations

import numpy as np
import pytest

from hipengine.core.memory import copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc
from hipengine.kernels.cpu_reference import gguf_quant_gemv
from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_q8_1_selected_prefill import (
    build_gguf_q4_k_q8_1_selected_prefill,
    gguf_q4_k_selected_dual_q8_1_prefill_compact32_bf16_bf16_out,
    plan_gguf_q4_k_q8_1_selected_prefill_build,
)
from hipengine.kernels.registry import resolve
from hipengine.quant.gguf import GGMLQuantizationType
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
    artifact = plan_gguf_q4_k_q8_1_selected_prefill_build(compiler_version="test-compiler")
    assert artifact.output_path.name == "gguf_q4_k_q8_1_selected_prefill.so"
    assert any(path.name == "gguf_q4_k_q8_1_selected_prefill.hip" for path in artifact.sources)
    assert "-mcumode" in artifact.flags

    dry_run = build_gguf_q4_k_q8_1_selected_prefill(dry_run=True, compiler_version="test-compiler")
    assert dry_run.output_path == artifact.output_path


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


# ---------------------------------------------------------------------------
# HIP correctness fixtures.
# ---------------------------------------------------------------------------


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
