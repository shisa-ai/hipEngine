"""Correctness tests for the dense GGUF Q4_K batched WMMA prefill kernels.

The P8.2 Q4_K kernels mirror the PARO/Q8_0 wave32 WMMA prefill shape and
replace only the inner K-loop with raw GGUF ``block_q4_K`` decoding. These
tests intentionally exercise the raw-block surface directly; runtime wiring
is task #9 because dense 2D Q4_K materialization currently uses the pack8
fallback layout and drops raw bytes.

Coverage:

1. No-GPU surface: registry, build-plan, default tile heuristic, wrapper
   contract validation for single and dual kernels.
2. GPU correctness: synthetic raw Q4_K weights from ``make_q4_k_weight`` and
   CPU oracle ``gguf_quant_gemv(..., GGMLQuantizationType.Q4_K)`` across rows,
   tile-boundary shapes, all single-output dtype combos, explicit tiles, and
   the dense BF16 gate+up dual path.

Numerics note: the WMMA kernel dequantizes Q4_K weights to fp32 and then casts
both activations and weights to ``half_t`` WMMA operands, accumulating in f32.
The CPU oracle dequantizes raw Q4_K weights in fp32. Tolerances are therefore
small but non-zero for f32 outputs, and output-ULP-sized for bf16/fp16 outputs.
"""

from __future__ import annotations

import ctypes
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
from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_prefill import (
    _ALLOWED_TILES,
    _default_tiles,
    build_gguf_q4_k_prefill,
    gguf_q4_k_wmma_prefill_bf16_bf16_out,
    gguf_q4_k_wmma_prefill_bf16_f32_out,
    gguf_q4_k_wmma_prefill_bf16_fp16_out,
    gguf_q4_k_wmma_prefill_dual_bf16_bf16_out,
    gguf_q4_k_wmma_prefill_f32_bf16_out,
    gguf_q4_k_wmma_prefill_f32_f32_out,
    gguf_q4_k_wmma_prefill_f32_fp16_out,
    gguf_q4_k_wmma_prefill_fp16_bf16_out,
    gguf_q4_k_wmma_prefill_fp16_f32_out,
    gguf_q4_k_wmma_prefill_fp16_fp16_out,
    plan_gguf_q4_k_prefill_build,
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
# 1. No-GPU surface: build plan, registry, contracts.
# ---------------------------------------------------------------------------


_ALL_DTYPE_WRAPPERS: dict[tuple[str, str], Any] = {
    ("bf16", "bf16"): gguf_q4_k_wmma_prefill_bf16_bf16_out,
    ("bf16", "fp16"): gguf_q4_k_wmma_prefill_bf16_fp16_out,
    ("bf16", "f32"): gguf_q4_k_wmma_prefill_bf16_f32_out,
    ("fp16", "bf16"): gguf_q4_k_wmma_prefill_fp16_bf16_out,
    ("fp16", "fp16"): gguf_q4_k_wmma_prefill_fp16_fp16_out,
    ("fp16", "f32"): gguf_q4_k_wmma_prefill_fp16_f32_out,
    ("f32", "bf16"): gguf_q4_k_wmma_prefill_f32_bf16_out,
    ("f32", "fp16"): gguf_q4_k_wmma_prefill_f32_fp16_out,
    ("f32", "f32"): gguf_q4_k_wmma_prefill_f32_f32_out,
}


def test_gguf_q4_k_wmma_prefill_registry_and_build_plan() -> None:
    """Every single-output dtype combo plus the dual key binds cleanly."""

    for (in_dtype, out_dtype), wrapper in _ALL_DTYPE_WRAPPERS.items():
        variant = f"wmma_prefill_{in_dtype}_{out_dtype}_out"
        assert (
            resolve(
                backend="hip_gfx1100",
                layer="linear",
                quant="gguf_q4_k",
                variant=variant,
            )
            is wrapper
        ), variant

    assert (
        resolve(
            backend="hip_gfx1100",
            layer="linear",
            quant="gguf_q4_k",
            variant="wmma_prefill_dual_bf16_bf16_out",
        )
        is gguf_q4_k_wmma_prefill_dual_bf16_bf16_out
    )

    artifact = plan_gguf_q4_k_prefill_build(compiler_version="test-compiler")
    assert artifact.output_path.name == "gguf_q4_k_prefill.so"
    assert "gguf_q4_k_prefill" in str(artifact.output_path)
    assert any(path.name == "gguf_q4_k_prefill.hip" for path in artifact.sources)

    dry_run = build_gguf_q4_k_prefill(dry_run=True, compiler_version="test-compiler")
    assert dry_run.output_path == artifact.output_path


def test_gguf_q4_k_wmma_prefill_default_tiles_match_paro_heuristic() -> None:
    assert _default_tiles(rows=512, out_features=2048) == (32, 32)
    assert _default_tiles(rows=32, out_features=2048) == (32, 32)
    assert _default_tiles(rows=31, out_features=2048) == (32, 16)
    assert _default_tiles(rows=8, out_features=2048) == (32, 16)
    assert _default_tiles(rows=512, out_features=16) == (16, 32)

    for tm, tn in _ALLOWED_TILES:
        assert tm in {16, 32, 64}
        assert tn in {16, 32}


def test_gguf_q4_k_wmma_prefill_wrapper_validates_contract() -> None:
    with pytest.raises(ValueError, match="Q4_K block size 256"):
        gguf_q4_k_wmma_prefill_bf16_f32_out(
            1, 2, 3, rows=4, in_features=255, out_features=16
        )

    with pytest.raises(ValueError, match="rows"):
        gguf_q4_k_wmma_prefill_bf16_f32_out(
            1, 2, 3, rows=0, in_features=256, out_features=16
        )

    with pytest.raises(ValueError, match="out_features"):
        gguf_q4_k_wmma_prefill_bf16_f32_out(
            1, 2, 3, rows=4, in_features=256, out_features=0
        )

    with pytest.raises(ValueError, match="tile"):
        gguf_q4_k_wmma_prefill_bf16_f32_out(
            1,
            2,
            3,
            rows=4,
            in_features=256,
            out_features=16,
            tile_m=24,
            tile_n=32,
        )

    with pytest.raises(ValueError, match="tile"):
        gguf_q4_k_wmma_prefill_bf16_f32_out(
            1,
            2,
            3,
            rows=4,
            in_features=256,
            out_features=16,
            tile_m=64,
            tile_n=64,
        )


def test_gguf_q4_k_wmma_prefill_dual_wrapper_validates_contract() -> None:
    with pytest.raises(ValueError, match="Q4_K block size 256"):
        gguf_q4_k_wmma_prefill_dual_bf16_bf16_out(
            1, 2, 3, 4, 5, rows=4, in_features=128, out_features=16
        )

    with pytest.raises(ValueError, match="tile"):
        gguf_q4_k_wmma_prefill_dual_bf16_bf16_out(
            1,
            2,
            3,
            4,
            5,
            rows=4,
            in_features=256,
            out_features=16,
            tile_m=16,
            tile_n=64,
        )


# ---------------------------------------------------------------------------
# 2. GPU correctness.
# ---------------------------------------------------------------------------


_OUT_DTYPE_TO_NUMPY = {
    "bf16": np.uint16,
    "fp16": np.float16,
    "f32": np.float32,
}


_TOLERANCES: dict[tuple[str, str], dict[str, float]] = {
    # f32 output preserves the f32 accumulator; tolerance mostly covers the
    # intentional fp16 WMMA operand casts vs the exact fp32 CPU oracle.
    ("bf16", "f32"): {"rtol": 5e-3, "atol": 1.5e-1},
    ("fp16", "f32"): {"rtol": 5e-3, "atol": 1.5e-1},
    ("f32", "f32"): {"rtol": 5e-3, "atol": 1.5e-1},
    # fp16 output adds one fp16 output rounding step.
    ("bf16", "fp16"): {"rtol": 6e-3, "atol": 2.0e-1},
    ("fp16", "fp16"): {"rtol": 6e-3, "atol": 2.0e-1},
    ("f32", "fp16"): {"rtol": 6e-3, "atol": 2.0e-1},
    # bf16 output is intentionally lossy. A bf16 ULP is 0.25 at 32 and 0.5 at
    # 64; synthetic magnitudes stay below that range for this fixture.
    ("bf16", "bf16"): {"rtol": 8e-3, "atol": 7.5e-1},
    ("fp16", "bf16"): {"rtol": 8e-3, "atol": 7.5e-1},
    ("f32", "bf16"): {"rtol": 8e-3, "atol": 7.5e-1},
}


def _float_array_to_bf16_bits(arr: np.ndarray) -> np.ndarray:
    f32 = arr.astype(np.float32, copy=False)
    bits = f32.view(np.uint32)
    lsb = (bits >> 16) & 1
    rounded = bits + 0x7FFF + lsb
    return (rounded >> 16).astype(np.uint16)


def _bf16_bits_to_float32(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << 16).view(np.float32)


def _make_activation(rows: int, in_features: int, seed: int = 0) -> np.ndarray:
    # Keep magnitudes modest and exactly representable in fp16. That isolates
    # Q4_K addressing/dequant correctness from large fp16 operand roundoff.
    rng = (
        np.arange(rows * in_features, dtype=np.float32).reshape(rows, in_features)
        + seed
    )
    return ((rng % 13) - 6) / 32.0


def _prepare_input(activation_f32: np.ndarray, in_dtype: str) -> np.ndarray:
    if in_dtype == "bf16":
        return _float_array_to_bf16_bits(activation_f32)
    if in_dtype == "fp16":
        return activation_f32.astype(np.float16)
    if in_dtype == "f32":
        return activation_f32.astype(np.float32)
    raise ValueError(f"unsupported in_dtype: {in_dtype}")


def _decode_input_for_cpu_reference(host_in: np.ndarray, in_dtype: str) -> np.ndarray:
    """Decode host input and apply the kernel's half WMMA operand cast."""

    if in_dtype == "bf16":
        decoded = _bf16_bits_to_float32(host_in)
    elif in_dtype == "fp16":
        decoded = host_in.astype(np.float32)
    elif in_dtype == "f32":
        decoded = host_in.astype(np.float32)
    else:
        raise ValueError(f"unsupported in_dtype: {in_dtype}")
    return decoded.astype(np.float16).astype(np.float32)


def _decode_output(out_arr: np.ndarray, out_dtype: str) -> np.ndarray:
    if out_dtype == "bf16":
        return _bf16_bits_to_float32(out_arr)
    if out_dtype == "fp16":
        return out_arr.astype(np.float32)
    if out_dtype == "f32":
        return out_arr
    raise ValueError(f"unsupported out_dtype: {out_dtype}")


def _cpu_q4_k_reference(x_f32: np.ndarray, qweight: np.ndarray) -> np.ndarray:
    return gguf_quant_gemv(x_f32, qweight, GGMLQuantizationType.Q4_K)


def _run_q4_k_wmma_prefill_gpu(
    *,
    rows: int,
    in_features: int,
    out_features: int,
    in_dtype: str,
    out_dtype: str,
    tile: tuple[int, int] | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    activation = _make_activation(rows, in_features, seed=seed)
    qweight = make_q4_k_weight(out_features, in_features)
    host_in = _prepare_input(activation, in_dtype)
    reference_input = _decode_input_for_cpu_reference(host_in, in_dtype)
    reference_f32 = _cpu_q4_k_reference(reference_input, qweight)
    host_out = np.zeros((rows, out_features), dtype=_OUT_DTYPE_TO_NUMPY[out_dtype])

    from hipengine.core.hip import get_hip_runtime

    runtime = get_hip_runtime()
    library = build_gguf_q4_k_prefill(load=True)
    wrapper = _ALL_DTYPE_WRAPPERS[(in_dtype, out_dtype)]

    bufs = []
    try:
        x_dev = malloc(host_in.nbytes, runtime=runtime)
        qw_dev = malloc(qweight.nbytes, runtime=runtime)
        out_dev = malloc(host_out.nbytes, runtime=runtime)
        bufs.extend((x_dev, qw_dev, out_dev))
        copy_host_to_device(
            x_dev, host_array_ptr(np.ascontiguousarray(host_in)), runtime=runtime
        )
        copy_host_to_device(
            qw_dev, host_array_ptr(np.ascontiguousarray(qweight)), runtime=runtime
        )
        kwargs: dict[str, Any] = {"library": library, "runtime": runtime}
        if tile is not None:
            kwargs["tile_m"], kwargs["tile_n"] = tile
        wrapper(
            x_dev.ptr,
            qw_dev.ptr,
            out_dev.ptr,
            rows,
            in_features,
            out_features,
            **kwargs,
        )
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(host_out), out_dev, runtime=runtime)
    finally:
        for buf in reversed(bufs):
            free(buf, runtime=runtime)

    return _decode_output(host_out, out_dtype), reference_f32


_ROWS_GRID = [4, 16, 17, 32, 64]

_SHAPE_GRID = [
    (256, 16),   # smallest raw Q4_K K block, aligned output tile
    (256, 24),   # output boundary tile
    (512, 48),
    (512, 80),   # output not multiple of 32
    (768, 128),  # multiple Q4_K superblocks
]


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("rows", _ROWS_GRID)
@pytest.mark.parametrize(("in_features", "out_features"), _SHAPE_GRID)
def test_gguf_q4_k_wmma_prefill_bf16_to_f32_matches_cpu_reference(
    rows: int, in_features: int, out_features: int
) -> None:
    actual, reference = _run_q4_k_wmma_prefill_gpu(
        rows=rows,
        in_features=in_features,
        out_features=out_features,
        in_dtype="bf16",
        out_dtype="f32",
    )
    np.testing.assert_allclose(actual, reference, **_TOLERANCES[("bf16", "f32")])


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize(
    ("in_dtype", "out_dtype"),
    [
        ("bf16", "bf16"),
        ("bf16", "fp16"),
        ("bf16", "f32"),
        ("fp16", "bf16"),
        ("fp16", "fp16"),
        ("fp16", "f32"),
        ("f32", "bf16"),
        ("f32", "fp16"),
        ("f32", "f32"),
    ],
)
def test_gguf_q4_k_wmma_prefill_full_dtype_matrix_matches_cpu_reference(
    in_dtype: str, out_dtype: str
) -> None:
    actual, reference = _run_q4_k_wmma_prefill_gpu(
        rows=32,
        in_features=512,
        out_features=80,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
    )
    np.testing.assert_allclose(actual, reference, **_TOLERANCES[(in_dtype, out_dtype)])


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("tile", sorted(_ALLOWED_TILES))
def test_gguf_q4_k_wmma_prefill_explicit_tile_selection_matches_cpu_reference(
    tile: tuple[int, int],
) -> None:
    actual, reference = _run_q4_k_wmma_prefill_gpu(
        rows=64,
        in_features=512,
        out_features=64,
        in_dtype="bf16",
        out_dtype="f32",
        tile=tile,
    )
    np.testing.assert_allclose(actual, reference, **_TOLERANCES[("bf16", "f32")])


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_gguf_q4_k_wmma_prefill_handles_unaligned_rows_and_out_features() -> None:
    actual, reference = _run_q4_k_wmma_prefill_gpu(
        rows=17,
        in_features=256,
        out_features=24,
        in_dtype="bf16",
        out_dtype="f32",
    )
    np.testing.assert_allclose(actual, reference, **_TOLERANCES[("bf16", "f32")])


def _run_q4_k_wmma_dual_prefill_gpu(
    *,
    rows: int,
    in_features: int,
    out_features: int,
    tile: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    activation = _make_activation(rows, in_features, seed=7)
    host_in = _prepare_input(activation, "bf16")
    reference_input = _decode_input_for_cpu_reference(host_in, "bf16")
    qweight_a = make_q4_k_weight(out_features, in_features)
    # Make B deterministic but different from A without requiring a second
    # fixture generator parameter.
    qweight_b = np.roll(qweight_a, shift=1, axis=0).copy()
    reference_a = _cpu_q4_k_reference(reference_input, qweight_a)
    reference_b = _cpu_q4_k_reference(reference_input, qweight_b)
    host_out_a = np.zeros((rows, out_features), dtype=np.uint16)
    host_out_b = np.zeros_like(host_out_a)

    from hipengine.core.hip import get_hip_runtime

    runtime = get_hip_runtime()
    library = build_gguf_q4_k_prefill(load=True)

    bufs = []
    try:
        x_dev = malloc(host_in.nbytes, runtime=runtime)
        qw_a_dev = malloc(qweight_a.nbytes, runtime=runtime)
        qw_b_dev = malloc(qweight_b.nbytes, runtime=runtime)
        out_a_dev = malloc(host_out_a.nbytes, runtime=runtime)
        out_b_dev = malloc(host_out_b.nbytes, runtime=runtime)
        bufs.extend((x_dev, qw_a_dev, qw_b_dev, out_a_dev, out_b_dev))
        copy_host_to_device(
            x_dev, host_array_ptr(np.ascontiguousarray(host_in)), runtime=runtime
        )
        copy_host_to_device(
            qw_a_dev, host_array_ptr(np.ascontiguousarray(qweight_a)), runtime=runtime
        )
        copy_host_to_device(
            qw_b_dev, host_array_ptr(np.ascontiguousarray(qweight_b)), runtime=runtime
        )
        kwargs: dict[str, Any] = {"library": library, "runtime": runtime}
        if tile is not None:
            kwargs["tile_m"], kwargs["tile_n"] = tile
        gguf_q4_k_wmma_prefill_dual_bf16_bf16_out(
            x_dev.ptr,
            qw_a_dev.ptr,
            qw_b_dev.ptr,
            out_a_dev.ptr,
            out_b_dev.ptr,
            rows,
            in_features,
            out_features,
            **kwargs,
        )
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(host_out_a), out_a_dev, runtime=runtime)
        copy_device_to_host(host_array_ptr(host_out_b), out_b_dev, runtime=runtime)
    finally:
        for buf in reversed(bufs):
            free(buf, runtime=runtime)

    return (
        _bf16_bits_to_float32(host_out_a),
        _bf16_bits_to_float32(host_out_b),
        reference_a,
        reference_b,
    )


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize(
    ("rows", "in_features", "out_features"),
    [
        (4, 256, 16),
        (17, 256, 24),
        (32, 512, 48),
        (64, 512, 64),
        (64, 768, 80),
    ],
)
def test_gguf_q4_k_wmma_prefill_dual_bf16_gate_up_matches_cpu_reference(
    rows: int, in_features: int, out_features: int
) -> None:
    actual_a, actual_b, reference_a, reference_b = _run_q4_k_wmma_dual_prefill_gpu(
        rows=rows,
        in_features=in_features,
        out_features=out_features,
    )
    tol = _TOLERANCES[("bf16", "bf16")]
    np.testing.assert_allclose(actual_a, reference_a, **tol)
    np.testing.assert_allclose(actual_b, reference_b, **tol)


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_gguf_q4_k_wmma_prefill_dual_explicit_tile_matches_cpu_reference() -> None:
    actual_a, actual_b, reference_a, reference_b = _run_q4_k_wmma_dual_prefill_gpu(
        rows=64,
        in_features=512,
        out_features=64,
        tile=(64, 32),
    )
    tol = _TOLERANCES[("bf16", "bf16")]
    np.testing.assert_allclose(actual_a, reference_a, **tol)
    np.testing.assert_allclose(actual_b, reference_b, **tol)
