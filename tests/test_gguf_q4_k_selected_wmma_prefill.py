"""Correctness tests for P8.4 selected GGUF Q4_K MoE WMMA prefill.

The selected kernel consumes the existing compact-MoE ABI emitted by the
qwen35_moe_group_* scheduler stack:

* x[compact_rows, in_features]
* expert_start_compact[num_experts + 1]
* expert_start_wmma[num_experts + 1]
* tile_expert[wmma_total_rows / 16]
* raw rank-3 Q4_K expert weights [E, out_features, row_bytes]
* concatenated gate+up output [compact_rows, out_features_a + out_features_b]

These tests build that compact ABI synthetically, compare against a CPU
selected/MoE reference assembled from ``gguf_quant_gemv(..., Q4_K)`` per
expert, and exercise uneven row counts, padding, empty experts, and output
feature tile boundaries.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.cpu_reference import gguf_quant_gemv
from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_selected_prefill import (
    build_gguf_q4_k_selected_prefill,
    gguf_q4_k_selected_dual_wmma_prefill_compact_bf16_bf16_out,
    gguf_q4_k_selected_dual_wmma_prefill_compact_fp16_fp16_out,
    gguf_q4_k_selected_dual_wmma_prefill_compact_hot_fulltile_bf16_bf16_out,
    gguf_q4_k_selected_dual_wmma_prefill_compact_hot_fulltile_fp16_fp16_out,
    gguf_q4_k_selected_dual_wmma_prefill_compact_sidemeta_bf16_bf16_out,
    gguf_q4_k_selected_dual_wmma_prefill_compact_sidemeta_fp16_fp16_out,
    q4_k_predecode_scale_min_sidemeta,
    plan_gguf_q4_k_selected_prefill_build,
    selected_dual_wmma_prefill_compact_default_tiles,
)
from hipengine.kernels.registry import resolve
from hipengine.quant.gguf import GGMLQuantizationType
from tests.test_gguf_q4_k_gemv import make_q4_k_weight


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


# ---------------------------------------------------------------------------
# No-GPU surface checks.
# ---------------------------------------------------------------------------


def test_gguf_q4_k_selected_wmma_registry_and_build_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HIPENGINE_GGUF_SELECTED_WMMA_LAUNCH_BOUNDS", raising=False)
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="moe_linear",
            quant="gguf_q4_k",
            variant="selected_dual_wmma_prefill_compact_bf16_bf16_out",
        )
        is gguf_q4_k_selected_dual_wmma_prefill_compact_bf16_bf16_out
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="moe_linear",
            quant="gguf_q4_k",
            variant="selected_dual_wmma_prefill_bf16_bf16_out",
        )
        is gguf_q4_k_selected_dual_wmma_prefill_compact_bf16_bf16_out
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="moe_linear",
            quant="gguf_q4_k",
            variant="selected_dual_wmma_prefill_compact_fp16_fp16_out",
        )
        is gguf_q4_k_selected_dual_wmma_prefill_compact_fp16_fp16_out
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="moe_linear",
            quant="gguf_q4_k",
            variant="selected_dual_wmma_prefill_compact_hot_fulltile_bf16_bf16_out",
        )
        is gguf_q4_k_selected_dual_wmma_prefill_compact_hot_fulltile_bf16_bf16_out
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="moe_linear",
            quant="gguf_q4_k",
            variant="selected_dual_wmma_prefill_compact_sidemeta_bf16_bf16_out",
        )
        is gguf_q4_k_selected_dual_wmma_prefill_compact_sidemeta_bf16_bf16_out
    )

    artifact = plan_gguf_q4_k_selected_prefill_build(compiler_version="test-compiler")
    assert artifact.output_path.name == "gguf_q4_k_selected_prefill.so"
    assert "gguf_q4_k_selected_prefill" in str(artifact.output_path)
    assert any(path.name == "gguf_q4_k_selected_prefill.hip" for path in artifact.sources)
    assert "-DHIPENGINE_SELECTED_WMMA_LAUNCH_BOUNDS=4" not in artifact.flags

    dry_run = build_gguf_q4_k_selected_prefill(
        dry_run=True, compiler_version="test-compiler"
    )
    assert dry_run.output_path == artifact.output_path

    monkeypatch.setenv("HIPENGINE_GGUF_SELECTED_WMMA_LAUNCH_BOUNDS", "4")
    lb4 = plan_gguf_q4_k_selected_prefill_build(compiler_version="test-compiler")
    assert "-DHIPENGINE_SELECTED_WMMA_LAUNCH_BOUNDS=4" in lb4.flags
    assert lb4.cache_key != artifact.cache_key


def test_p9_c1_q4_k_selected_default_tile_decision(monkeypatch: pytest.MonkeyPatch) -> None:
    """P9.C1 dispatch pin: Q4_K dual selected prefill defaults to 32x16."""

    monkeypatch.delenv("HIPENGINE_GGUF_Q4_K_SELECTED_WMMA_TILE_M", raising=False)
    monkeypatch.delenv("HIPENGINE_GGUF_Q4_K_SELECTED_WMMA_TILE_N", raising=False)
    assert selected_dual_wmma_prefill_compact_default_tiles() == (32, 16)
    monkeypatch.setenv("HIPENGINE_GGUF_Q4_K_SELECTED_WMMA_TILE_M", "16")
    monkeypatch.setenv("HIPENGINE_GGUF_Q4_K_SELECTED_WMMA_TILE_N", "16")
    assert selected_dual_wmma_prefill_compact_default_tiles() == (16, 16)
    monkeypatch.setenv("HIPENGINE_GGUF_Q4_K_SELECTED_WMMA_TILE_M", "64")
    monkeypatch.setenv("HIPENGINE_GGUF_Q4_K_SELECTED_WMMA_TILE_N", "32")
    assert selected_dual_wmma_prefill_compact_default_tiles() == (64, 32)


def test_gguf_q4_k_selected_wmma_hot_wrapper_validates_threshold() -> None:
    kwargs = dict(
        x_ptr=1,
        expert_start_compact_ptr=2,
        expert_start_wmma_ptr=3,
        tile_expert_ptr=4,
        qweight_a_ptr=5,
        qweight_b_ptr=6,
        out_ptr=7,
        compact_rows=17,
        in_features=256,
        out_features_a=32,
        out_features_b=32,
        num_experts=2,
        wmma_total_rows=32,
    )
    with pytest.raises(ValueError, match="hot_threshold"):
        gguf_q4_k_selected_dual_wmma_prefill_compact_hot_fulltile_bf16_bf16_out(
            **kwargs, hot_threshold=0
        )


def test_gguf_q4_k_selected_wmma_wrapper_validates_common_contract() -> None:
    kwargs = dict(
        x_ptr=1,
        expert_start_compact_ptr=2,
        expert_start_wmma_ptr=3,
        tile_expert_ptr=4,
        qweight_a_ptr=5,
        qweight_b_ptr=6,
        out_ptr=7,
        compact_rows=17,
        in_features=256,
        out_features_a=32,
        out_features_b=32,
        num_experts=2,
        wmma_total_rows=32,
    )

    with pytest.raises(ValueError, match="compact_rows"):
        gguf_q4_k_selected_dual_wmma_prefill_compact_bf16_bf16_out(
            **{**kwargs, "compact_rows": 0}
        )
    with pytest.raises(ValueError, match="Q4_K block size 256"):
        gguf_q4_k_selected_dual_wmma_prefill_compact_bf16_bf16_out(
            **{**kwargs, "in_features": 128}
        )
    with pytest.raises(ValueError, match="out_features_a.*multiple of 16"):
        gguf_q4_k_selected_dual_wmma_prefill_compact_bf16_bf16_out(
            **{**kwargs, "out_features_a": 24}
        )
    with pytest.raises(ValueError, match="out_features_b.*multiple of 16"):
        gguf_q4_k_selected_dual_wmma_prefill_compact_bf16_bf16_out(
            **{**kwargs, "out_features_b": 24}
        )
    with pytest.raises(ValueError, match="wmma_total_rows.*multiple of 16"):
        gguf_q4_k_selected_dual_wmma_prefill_compact_bf16_bf16_out(
            **{**kwargs, "wmma_total_rows": 31}
        )


# ---------------------------------------------------------------------------
# Compact selected-MoE fixture helpers.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CompactFixture:
    x_host: np.ndarray
    expert_start_compact: np.ndarray
    expert_start_wmma: np.ndarray
    tile_expert: np.ndarray
    qweight_a: np.ndarray
    qweight_b: np.ndarray
    reference: np.ndarray
    compact_rows: int
    wmma_total_rows: int
    in_features: int
    out_features_a: int
    out_features_b: int
    num_experts: int


def _float_array_to_bf16_bits(arr: np.ndarray) -> np.ndarray:
    f32 = arr.astype(np.float32, copy=False)
    bits = f32.view(np.uint32)
    rounded = bits + 0x7FFF + ((bits >> 16) & 1)
    return (rounded >> 16).astype(np.uint16)


def _bf16_bits_to_float32(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << 16).view(np.float32)


def _make_activation(rows: int, in_features: int, seed: int = 0) -> np.ndarray:
    values = (
        np.arange(rows * in_features, dtype=np.float32).reshape(rows, in_features)
        + seed
    )
    # Smaller than the dense-Q4 test fixture so BF16/FP16 selected tests stay
    # focused on compact addressing/dequant and not fp16 operand stress.
    return ((values % 13) - 6) / 64.0


def _prepare_input(x_f32: np.ndarray, dtype: str) -> np.ndarray:
    if dtype == "bf16":
        return _float_array_to_bf16_bits(x_f32)
    if dtype == "fp16":
        return x_f32.astype(np.float16)
    raise ValueError(dtype)


def _decode_input_for_reference(host: np.ndarray, dtype: str) -> np.ndarray:
    if dtype == "bf16":
        decoded = _bf16_bits_to_float32(host)
    elif dtype == "fp16":
        decoded = host.astype(np.float32)
    else:
        raise ValueError(dtype)
    # The kernel feeds half WMMA operands; match that before invoking the CPU
    # selected/MoE oracle assembled from exact GGUF Q4_K GEMVs.
    return decoded.astype(np.float16).astype(np.float32)


def _decode_output(host: np.ndarray, dtype: str) -> np.ndarray:
    if dtype == "bf16":
        return _bf16_bits_to_float32(host)
    if dtype == "fp16":
        return host.astype(np.float32)
    raise ValueError(dtype)


def _make_expert_q4_k_weights(
    *, num_experts: int, out_features: int, in_features: int, offset: int
) -> np.ndarray:
    base = make_q4_k_weight(out_features, in_features)
    return np.ascontiguousarray(
        np.stack([np.roll(base, shift=offset + expert, axis=0) for expert in range(num_experts)], axis=0)
    )


def _build_compact_fixture(
    *,
    counts: list[int],
    in_features: int,
    out_features_a: int,
    out_features_b: int,
    dtype: str,
    seed: int = 0,
) -> CompactFixture:
    num_experts = len(counts)
    compact_rows = int(sum(counts))
    assert compact_rows > 0
    expert_start_compact = np.zeros(num_experts + 1, dtype=np.int64)
    expert_start_compact[1:] = np.cumsum(np.asarray(counts, dtype=np.int64))

    padded_counts = [((count + 15) // 16) * 16 for count in counts]
    expert_start_wmma = np.zeros(num_experts + 1, dtype=np.int64)
    expert_start_wmma[1:] = np.cumsum(np.asarray(padded_counts, dtype=np.int64))
    wmma_total_rows = int(expert_start_wmma[-1])
    tile_expert = np.asarray(
        [expert for expert, padded in enumerate(padded_counts) for _ in range(padded // 16)],
        dtype=np.int64,
    )
    assert tile_expert.size == wmma_total_rows // 16

    x_f32 = _make_activation(compact_rows, in_features, seed=seed)
    x_host = _prepare_input(x_f32, dtype)
    x_ref = _decode_input_for_reference(x_host, dtype)

    qweight_a = _make_expert_q4_k_weights(
        num_experts=num_experts,
        out_features=out_features_a,
        in_features=in_features,
        offset=0,
    )
    qweight_b = _make_expert_q4_k_weights(
        num_experts=num_experts,
        out_features=out_features_b,
        in_features=in_features,
        offset=3,
    )

    reference = np.zeros(
        (compact_rows, out_features_a + out_features_b), dtype=np.float32
    )
    for expert, count in enumerate(counts):
        if count == 0:
            continue
        start = int(expert_start_compact[expert])
        stop = start + count
        reference[start:stop, :out_features_a] = gguf_quant_gemv(
            x_ref[start:stop], qweight_a[expert], GGMLQuantizationType.Q4_K
        )
        reference[start:stop, out_features_a:] = gguf_quant_gemv(
            x_ref[start:stop], qweight_b[expert], GGMLQuantizationType.Q4_K
        )

    return CompactFixture(
        x_host=np.ascontiguousarray(x_host),
        expert_start_compact=expert_start_compact,
        expert_start_wmma=expert_start_wmma,
        tile_expert=tile_expert,
        qweight_a=qweight_a,
        qweight_b=qweight_b,
        reference=reference,
        compact_rows=compact_rows,
        wmma_total_rows=wmma_total_rows,
        in_features=in_features,
        out_features_a=out_features_a,
        out_features_b=out_features_b,
        num_experts=num_experts,
    )


def _run_selected_dual_gpu(
    fixture: CompactFixture,
    dtype: str,
    *,
    hot_fulltile: bool = False,
    hot_threshold: int = 64,
    sidemeta: bool = False,
) -> np.ndarray:
    from hipengine.core.hip import get_hip_runtime

    runtime = get_hip_runtime()
    library = build_gguf_q4_k_selected_prefill(load=True)
    out_dtype = np.uint16 if dtype == "bf16" else np.float16
    host_out = np.zeros(
        (fixture.compact_rows, fixture.out_features_a + fixture.out_features_b),
        dtype=out_dtype,
    )
    if sidemeta:
        wrapper = (
            gguf_q4_k_selected_dual_wmma_prefill_compact_sidemeta_bf16_bf16_out
            if dtype == "bf16"
            else gguf_q4_k_selected_dual_wmma_prefill_compact_sidemeta_fp16_fp16_out
        )
    elif hot_fulltile:
        wrapper = (
            gguf_q4_k_selected_dual_wmma_prefill_compact_hot_fulltile_bf16_bf16_out
            if dtype == "bf16"
            else gguf_q4_k_selected_dual_wmma_prefill_compact_hot_fulltile_fp16_fp16_out
        )
    else:
        wrapper = (
            gguf_q4_k_selected_dual_wmma_prefill_compact_bf16_bf16_out
            if dtype == "bf16"
            else gguf_q4_k_selected_dual_wmma_prefill_compact_fp16_fp16_out
        )

    bufs = []
    try:
        x_dev = malloc(fixture.x_host.nbytes, runtime=runtime)
        start_compact_dev = malloc(fixture.expert_start_compact.nbytes, runtime=runtime)
        start_wmma_dev = malloc(fixture.expert_start_wmma.nbytes, runtime=runtime)
        tile_expert_dev = malloc(fixture.tile_expert.nbytes, runtime=runtime)
        qweight_a_dev = malloc(fixture.qweight_a.nbytes, runtime=runtime)
        qweight_b_dev = malloc(fixture.qweight_b.nbytes, runtime=runtime)
        sidemeta_a = q4_k_predecode_scale_min_sidemeta(fixture.qweight_a) if sidemeta else None
        sidemeta_b = q4_k_predecode_scale_min_sidemeta(fixture.qweight_b) if sidemeta else None
        sidemeta_a_dev = malloc(sidemeta_a.nbytes, runtime=runtime) if sidemeta_a is not None else None
        sidemeta_b_dev = malloc(sidemeta_b.nbytes, runtime=runtime) if sidemeta_b is not None else None
        out_dev = malloc(host_out.nbytes, runtime=runtime)
        bufs.extend(
            (
                x_dev,
                start_compact_dev,
                start_wmma_dev,
                tile_expert_dev,
                qweight_a_dev,
                qweight_b_dev,
                *(buf for buf in (sidemeta_a_dev, sidemeta_b_dev) if buf is not None),
                out_dev,
            )
        )
        copy_pairs = [
            (x_dev, fixture.x_host),
            (start_compact_dev, fixture.expert_start_compact),
            (start_wmma_dev, fixture.expert_start_wmma),
            (tile_expert_dev, fixture.tile_expert),
            (qweight_a_dev, fixture.qweight_a),
            (qweight_b_dev, fixture.qweight_b),
        ]
        if sidemeta:
            copy_pairs.extend(((sidemeta_a_dev, sidemeta_a), (sidemeta_b_dev, sidemeta_b)))
        for dev, arr in copy_pairs:
            copy_host_to_device(
                dev, host_array_ptr(np.ascontiguousarray(arr)), runtime=runtime
            )

        kwargs = {"library": library, "runtime": runtime}
        if hot_fulltile:
            kwargs["hot_threshold"] = hot_threshold
        if sidemeta:
            wrapper(
                x_dev.ptr,
                start_compact_dev.ptr,
                start_wmma_dev.ptr,
                tile_expert_dev.ptr,
                qweight_a_dev.ptr,
                qweight_b_dev.ptr,
                sidemeta_a_dev.ptr,
                sidemeta_b_dev.ptr,
                out_dev.ptr,
                fixture.compact_rows,
                fixture.in_features,
                fixture.out_features_a,
                fixture.out_features_b,
                fixture.num_experts,
                fixture.wmma_total_rows,
                **kwargs,
            )
        else:
            wrapper(
                x_dev.ptr,
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
                **kwargs,
            )
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(host_out), out_dev, runtime=runtime)
    finally:
        for buf in reversed(bufs):
            free(buf, runtime=runtime)

    return _decode_output(host_out, dtype)


_SELECTED_CASES = [
    pytest.param([4, 0, 5], 256, 16, 16, id="empty-middle-small-boundary"),
    pytest.param([16, 17, 31], 256, 32, 32, id="exact-plus-padding"),
    pytest.param([0, 33, 1, 16], 512, 48, 32, id="empty-first-multi-block"),
    pytest.param([7, 18, 0, 33], 512, 64, 16, id="empty-third-wide-gate"),
    pytest.param([32, 0, 0, 17], 768, 64, 48, id="multi-block-empty-tail"),
]

_TOLERANCE_BF16 = {"rtol": 1.2e-2, "atol": 5.0e-1}
_TOLERANCE_FP16 = {"rtol": 6.0e-3, "atol": 1.5e-1}


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize(
    ("counts", "in_features", "out_features_a", "out_features_b"), _SELECTED_CASES
)
def test_gguf_q4_k_selected_wmma_bf16_matches_cpu_selected_reference(
    counts: list[int], in_features: int, out_features_a: int, out_features_b: int
) -> None:
    fixture = _build_compact_fixture(
        counts=counts,
        in_features=in_features,
        out_features_a=out_features_a,
        out_features_b=out_features_b,
        dtype="bf16",
    )
    actual = _run_selected_dual_gpu(fixture, "bf16")
    np.testing.assert_allclose(actual, fixture.reference, **_TOLERANCE_BF16)


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize(
    ("counts", "in_features", "out_features_a", "out_features_b"),
    [
        pytest.param([5, 11, 0, 23], 256, 32, 16, id="fp16-uneven-empty"),
        pytest.param([0, 16, 17], 512, 48, 48, id="fp16-empty-first-wide"),
    ],
)
def test_gguf_q4_k_selected_wmma_fp16_matches_cpu_selected_reference(
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
    actual = _run_selected_dual_gpu(fixture, "fp16")
    np.testing.assert_allclose(actual, fixture.reference, **_TOLERANCE_FP16)


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_p9_c5_q4_k_predecode_scale_min_sidemeta_shape_and_values() -> None:
    raw = _make_expert_q4_k_weights(num_experts=2, out_features=4, in_features=256, offset=1)
    sidemeta = q4_k_predecode_scale_min_sidemeta(raw)
    assert sidemeta.shape == (2, 4, 1, 8, 2)
    assert sidemeta.dtype == np.float16
    # Spot check subblock 0 against the same decode formula used by the HIP kernel.
    block = raw[0, 0]
    d_bits = np.uint16(int(block[0]) | (int(block[1]) << 8))
    dmin_bits = np.uint16(int(block[2]) | (int(block[3]) << 8))
    d = np.array(d_bits, dtype=np.uint16).view(np.float16).astype(np.float32)
    dmin = np.array(dmin_bits, dtype=np.uint16).view(np.float16).astype(np.float32)
    scales = block[4:16]
    assert sidemeta[0, 0, 0, 0, 0] == pytest.approx(float(d) * float(scales[0] & 0x3F))
    assert sidemeta[0, 0, 0, 0, 1] == pytest.approx(float(dmin) * float(scales[4] & 0x3F))


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_p9_c5_q4_k_sidemeta_matches_cpu_selected_reference() -> None:
    fixture = _build_compact_fixture(
        counts=[0, 7, 16, 63, 64, 79, 128],
        in_features=512,
        out_features_a=64,
        out_features_b=64,
        dtype="bf16",
        seed=15,
    )
    actual = _run_selected_dual_gpu(fixture, "bf16", sidemeta=True)
    np.testing.assert_allclose(actual, fixture.reference, **_TOLERANCE_BF16)


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_p9_c4_q4_k_hot_fulltile_hybrid_matches_cpu_selected_reference() -> None:
    """P9.C4 hot/full-tile prototype preserves the compact selected ABI.

    Counts include hot full tiles, hot tails, cold full tiles, cold tails, and
    empty experts so both the full-tile and tail/cold kernels must participate.
    """

    fixture = _build_compact_fixture(
        counts=[0, 7, 16, 63, 64, 79, 128],
        in_features=512,
        out_features_a=64,
        out_features_b=64,
        dtype="bf16",
        seed=13,
    )
    actual = _run_selected_dual_gpu(
        fixture, "bf16", hot_fulltile=True, hot_threshold=64
    )
    np.testing.assert_allclose(actual, fixture.reference, **_TOLERANCE_BF16)


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_gguf_q4_k_selected_wmma_runs_exported_bf16_symbol_on_w7900() -> None:
    """A tiny launch confirms the selected WMMA kernel surface runs on GPU."""

    fixture = _build_compact_fixture(
        counts=[1, 0],
        in_features=256,
        out_features_a=16,
        out_features_b=16,
        dtype="bf16",
        seed=9,
    )
    actual = _run_selected_dual_gpu(fixture, "bf16")
    np.testing.assert_allclose(actual, fixture.reference, **_TOLERANCE_BF16)
