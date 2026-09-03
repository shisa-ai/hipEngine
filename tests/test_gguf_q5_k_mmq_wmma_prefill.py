"""Source-shaped Q5_K MMQ prefill contracts.

The broad tile keeps llama.cpp's K-major DS4 records.  The C8 tile instead
uses K-major D4S4-FP32 records to preserve the retained raw-Q5 metadata while
changing only MMQ dataflow.  The retained raw-Q5 path remains the oracle.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np
import pytest

from hipengine.benchmark.correctness import evaluate_logits
from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
    memory_stats,
)
from hipengine.kernels.cpu_reference import gguf_q5_k_gemv
from hipengine.kernels.hip_gfx1100.quant import gguf_k_mmq_prefill as mmq
from hipengine.kernels.registry import KernelKey, is_registered, resolve
from tests.test_gguf_k_gemv import make_q5_k_weight

_Q8_BLOCK = 128
_DS4_BYTES = 144
_D4S4_F32_BYTES = 160
_SOURCE = (
    Path(__file__).parents[1]
    / "hipengine"
    / "kernels"
    / "hip_gfx1100"
    / "quant"
    / "gguf_k_mmq_prefill.hip"
)


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def _bf16_bits(values: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(values, dtype=np.float32)
    bits = contiguous.view(np.uint32)
    lsb = (bits >> 16) & 1
    return ((bits + 0x7FFF + lsb) >> 16).astype(np.uint16)


def _bf16_to_f32(values: np.ndarray) -> np.ndarray:
    return (np.asarray(values, dtype=np.uint16).astype(np.uint32) << 16).view(
        np.float32
    )


def _round_away(values: np.ndarray) -> np.ndarray:
    return np.sign(values) * np.floor(np.abs(values) + np.float32(0.5))


def _wave32_sum(values: np.ndarray) -> np.float32:
    """Reproduce the producer's eight-lane, four-values-per-lane reduction."""

    values = np.asarray(values, dtype=np.float32).reshape(8, 4)
    partial = np.asarray(
        [((lane[0] + lane[1]) + lane[2]) + lane[3] for lane in values],
        dtype=np.float32,
    )
    partial[:4] = partial[:4] + partial[4:]
    partial[:2] = partial[:2] + partial[2:4]
    return np.float32(partial[0] + partial[1])


def _pack_ds4_kmajor_cpu(x_bf16: np.ndarray) -> np.ndarray:
    """Pack llama.cpp DS4 records as [K128 block, row, 144 bytes]."""

    x = _bf16_to_f32(np.ascontiguousarray(x_bf16, dtype=np.uint16))
    rows, hidden = x.shape
    if hidden % _Q8_BLOCK:
        raise ValueError("hidden must be a multiple of 128")
    packed = np.zeros((hidden // _Q8_BLOCK, rows, _DS4_BYTES), dtype=np.uint8)
    for block in range(hidden // _Q8_BLOCK):
        for row in range(rows):
            values = x[row, block * 128 : (block + 1) * 128].reshape(4, 32)
            metadata = np.zeros((8,), dtype=np.float16)
            quants = np.zeros((4, 32), dtype=np.int8)
            for group, group_values in enumerate(values):
                amax = np.max(np.abs(group_values)).astype(np.float32)
                scale = np.float32(0.0) if amax == 0 else np.float32(1.0) / np.float32(127.0 / amax)
                metadata[2 * group] = np.float16(scale)
                metadata[2 * group + 1] = np.float16(_wave32_sum(group_values))
                if scale != 0.0:
                    quants[group] = np.clip(
                        _round_away(group_values * np.float32(1.0 / scale)),
                        -128,
                        127,
                    ).astype(np.int8)
            packed[block, row, :16] = metadata.view(np.uint8)
            packed[block, row, 16:] = quants.reshape(128).view(np.uint8)
    return packed


def _pack_d4s4_f32_kmajor_cpu(x_bf16: np.ndarray) -> np.ndarray:
    """Pack precision-preserving records as [K128 block, row, 160 bytes]."""

    x = _bf16_to_f32(np.ascontiguousarray(x_bf16, dtype=np.uint16))
    rows, hidden = x.shape
    if hidden % _Q8_BLOCK:
        raise ValueError("hidden must be a multiple of 128")
    packed = np.zeros(
        (hidden // _Q8_BLOCK, rows, _D4S4_F32_BYTES), dtype=np.uint8
    )
    for block in range(hidden // _Q8_BLOCK):
        for row in range(rows):
            values = x[row, block * 128 : (block + 1) * 128].reshape(4, 32)
            scales = np.zeros((4,), dtype=np.float32)
            sums = np.zeros((4,), dtype=np.float32)
            quants = np.zeros((4, 32), dtype=np.int8)
            for group, group_values in enumerate(values):
                amax = np.max(np.abs(group_values)).astype(np.float32)
                scale = (
                    np.float32(0.0)
                    if amax == 0
                    else np.float32(amax / np.float32(127.0))
                )
                scales[group] = scale
                sums[group] = _wave32_sum(group_values)
                if scale != 0.0:
                    quants[group] = np.clip(
                        _round_away(group_values / scale), -128, 127
                    ).astype(np.int8)
            packed[block, row, :16] = scales.view(np.uint8)
            packed[block, row, 16:32] = sums.view(np.uint8)
            packed[block, row, 32:] = quants.reshape(128).view(np.uint8)
    return packed


def _activation(rows: int, hidden: int) -> np.ndarray:
    values = np.arange(rows * hidden, dtype=np.float32).reshape(rows, hidden)
    values = ((values * np.float32(1.6180339)) % np.float32(23.0) - 11.0) / 32.0
    # Exercise the source-defined all-zero group without making every record zero.
    values[:, 32:64] = 0.0
    return values


def test_q5_source_mmq_registry_build_scope_and_workspace_contract() -> None:
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels

    mmq.register_gguf_k_mmq_prefill_kernels(replace=True)
    register_gfx1151_kernels(replace=True)

    assert mmq.q8_1_ds4_kmajor_nbytes(512, 9216) == 5_308_416
    assert mmq.q8_1_ds4_kmajor_nbytes(129, 256) == 2 * 129 * 144
    assert mmq.q8_1_d4s4_f32_kmajor_nbytes(32, 256) == 2 * 32 * 160
    with pytest.raises(ValueError, match="multiple of 128"):
        mmq.q8_1_ds4_kmajor_nbytes(17, 192)

    producer_key = KernelKey(
        "hip_gfx1100", "activation_quant", "q8_1_ds4", "bf16_kmajor"
    )
    assert resolve(
        backend=producer_key.backend,
        layer=producer_key.layer,
        quant=producer_key.quant,
        variant=producer_key.variant,
    ) is mmq.gguf_q8_1_ds4_quantize_bf16_kmajor
    assert not is_registered(
        KernelKey("hip_gfx1151", producer_key.layer, producer_key.quant, producer_key.variant)
    )
    c8_producer_key = KernelKey(
        "hip_gfx1100", "activation_quant", "q8_1_d4s4_f32", "bf16_kmajor"
    )
    assert resolve(
        backend=c8_producer_key.backend,
        layer=c8_producer_key.layer,
        quant=c8_producer_key.quant,
        variant=c8_producer_key.variant,
    ) is mmq.gguf_q8_1_d4s4_f32_quantize_bf16_kmajor
    assert not is_registered(
        KernelKey(
            "hip_gfx1151",
            c8_producer_key.layer,
            c8_producer_key.quant,
            c8_producer_key.variant,
        )
    )

    for tile, producer_layout, output_dtype, fn in (
        (
            "i128_j128",
            "ds4",
            "bf16",
            mmq.gguf_q5_k_mmq_i128_j128_k256_q8_1_ds4_bf16_bf16_out,
        ),
        (
            "i128_j128",
            "ds4",
            "f32",
            mmq.gguf_q5_k_mmq_i128_j128_k256_q8_1_ds4_bf16_f32_out,
        ),
        (
            "i64_j16_j32",
            "d4s4_f32_kmajor",
            "bf16",
            mmq.gguf_q5_k_mmq_i64_j16_j32_k256_q8_1_d4s4_f32_kmajor_bf16_bf16_out,
        ),
        (
            "i64_j16_j32",
            "d4s4_f32_kmajor",
            "f32",
            mmq.gguf_q5_k_mmq_i64_j16_j32_k256_q8_1_d4s4_f32_kmajor_bf16_f32_out,
        ),
    ):
        key = KernelKey(
            "hip_gfx1100",
            "linear",
            "gguf_q5_k",
            f"mmq_{tile}_k256_q8_1_{producer_layout}_bf16_{output_dtype}_out",
        )
        assert resolve(
            backend=key.backend,
            layer=key.layer,
            quant=key.quant,
            variant=key.variant,
        ) is fn
        assert not is_registered(KernelKey("hip_gfx1151", key.layer, key.quant, key.variant))

    artifact = mmq.plan_gguf_k_mmq_prefill_build(compiler_version="test")
    assert artifact.output_path.name == "gguf_k_mmq_prefill.so"
    assert any(path.name == "gguf_k_mmq_prefill.hip" for path in artifact.sources)
    source_artifact = mmq.plan_gguf_q5_k_source_mmq_prefill_build(
        compiler_version="test"
    )
    assert source_artifact.output_path.name == "gguf_q5_k_source_mmq_prefill.so"
    assert "-ffast-math" in source_artifact.flags
    assert "-ffast-math" not in artifact.flags
    source = _SOURCE.read_text()
    assert "__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32" in source
    assert "torch::Tensor" not in source


def test_q5_source_mmq_wrappers_reject_unsupported_k_before_loading_gpu() -> None:
    with pytest.raises(ValueError, match="multiple of 128"):
        mmq.gguf_q8_1_ds4_quantize_bf16_kmajor(1, 2, 17, 192)
    with pytest.raises(ValueError, match="multiple of 128"):
        mmq.gguf_q8_1_d4s4_f32_quantize_bf16_kmajor(1, 2, 17, 192)
    for fn in (
        mmq.gguf_q5_k_mmq_i128_j128_k256_q8_1_ds4_bf16_bf16_out,
        mmq.gguf_q5_k_mmq_i128_j128_k256_q8_1_ds4_bf16_f32_out,
    ):
        with pytest.raises(ValueError, match="multiple of 256"):
            fn(1, 2, 3, 17, 128, 72)


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("rows", [127, 128, 129])
def test_q5_source_ds4_producer_matches_kmajor_cpu_bytes(rows: int) -> None:
    hidden = 256
    x_bf16 = _bf16_bits(_activation(rows, hidden))
    expected = _pack_ds4_kmajor_cpu(x_bf16)
    actual = np.empty_like(expected)
    library = mmq.build_gguf_k_mmq_prefill(load=True)
    from hipengine.core.hip import get_hip_runtime

    runtime = get_hip_runtime()
    buffers = []
    try:
        x_dev = malloc(x_bf16.nbytes, runtime=runtime)
        packed_dev = malloc(actual.nbytes, runtime=runtime)
        buffers.extend((x_dev, packed_dev))
        copy_host_to_device(x_dev, host_array_ptr(x_bf16), runtime=runtime)
        mmq.gguf_q8_1_ds4_quantize_bf16_kmajor(
            x_dev.ptr,
            packed_dev.ptr,
            rows,
            hidden,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(actual), packed_dev, runtime=runtime)
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("rows", [24, 32])
def test_c8_q5_d4s4_f32_producer_matches_kmajor_cpu_bytes(rows: int) -> None:
    hidden = 256
    x_bf16 = _bf16_bits(_activation(rows, hidden))
    expected = _pack_d4s4_f32_kmajor_cpu(x_bf16)
    actual = np.empty_like(expected)
    library = mmq.build_gguf_k_mmq_prefill(load=True)
    from hipengine.core.hip import get_hip_runtime

    runtime = get_hip_runtime()
    buffers = []
    try:
        x_dev = malloc(x_bf16.nbytes, runtime=runtime)
        packed_dev = malloc(actual.nbytes, runtime=runtime)
        buffers.extend((x_dev, packed_dev))
        copy_host_to_device(x_dev, host_array_ptr(x_bf16), runtime=runtime)
        mmq.gguf_q8_1_d4s4_f32_quantize_bf16_kmajor(
            x_dev.ptr,
            packed_dev.ptr,
            rows,
            hidden,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(actual), packed_dev, runtime=runtime)
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)
    np.testing.assert_array_equal(actual, expected)


def _run_candidate(
    *,
    rows: int,
    hidden: int,
    out_features: int,
    output_dtype: str,
    consumer: str = "i128_j128",
) -> tuple[np.ndarray, np.ndarray]:
    x_bf16 = _bf16_bits(_activation(rows, hidden))
    qweight = np.ascontiguousarray(
        make_q5_k_weight(out_features, hidden), dtype=np.uint8
    )
    packed_nbytes = (
        mmq.q8_1_d4s4_f32_kmajor_nbytes(rows, hidden)
        if consumer in {"i64_j32", "i64_j16"}
        else mmq.q8_1_d4s4_f32_nbytes(rows, hidden)
        if consumer in {"mmq32", "mmq32_pipe"}
        else mmq.q8_1_ds4_kmajor_nbytes(rows, hidden)
    )
    out = np.empty(
        (rows, out_features), dtype=np.uint16 if output_dtype == "bf16" else np.float32
    )
    reference = gguf_q5_k_gemv(_bf16_to_f32(x_bf16), qweight)
    producer_library = mmq.build_gguf_k_mmq_prefill(load=True)
    consumer_library = (
        producer_library
        if consumer in {"mmq32", "mmq32_pipe"}
        else mmq.build_gguf_q5_k_source_mmq_prefill(load=True)
    )
    from hipengine.core.hip import get_hip_runtime

    runtime = get_hip_runtime()
    before = memory_stats()
    buffers = []
    try:
        x_dev = malloc(x_bf16.nbytes, runtime=runtime)
        packed_dev = malloc(packed_nbytes, runtime=runtime)
        weight_dev = malloc(qweight.nbytes, runtime=runtime)
        out_dev = malloc(out.nbytes, runtime=runtime)
        buffers.extend((x_dev, packed_dev, weight_dev, out_dev))
        copy_host_to_device(x_dev, host_array_ptr(x_bf16), runtime=runtime)
        copy_host_to_device(weight_dev, host_array_ptr(qweight), runtime=runtime)
        producer = (
            mmq.gguf_q8_1_d4s4_f32_quantize_bf16_kmajor
            if consumer in {"i64_j32", "i64_j16"}
            else mmq.gguf_q8_1_d4s4_f32_quantize_bf16
            if consumer in {"mmq32", "mmq32_pipe"}
            else mmq.gguf_q8_1_ds4_quantize_bf16_kmajor
        )
        producer(
            x_dev.ptr,
            packed_dev.ptr,
            rows,
            hidden,
            library=producer_library,
            runtime=runtime,
        )
        fn = {
            ("i128_j128", "bf16"): (
                mmq.gguf_q5_k_mmq_i128_j128_k256_q8_1_ds4_bf16_bf16_out
            ),
            ("i128_j128", "f32"): (
                mmq.gguf_q5_k_mmq_i128_j128_k256_q8_1_ds4_bf16_f32_out
            ),
            ("i64_j32", "bf16"): (
                mmq.gguf_q5_k_mmq_i64_j16_j32_k256_q8_1_d4s4_f32_kmajor_bf16_bf16_out
            ),
            ("i64_j32", "f32"): (
                mmq.gguf_q5_k_mmq_i64_j16_j32_k256_q8_1_d4s4_f32_kmajor_bf16_f32_out
            ),
            ("i64_j16", "bf16"): (
                mmq.gguf_q5_k_mmq_i64_j16_forced_k256_q8_1_d4s4_f32_kmajor_bf16_bf16_out
            ),
            ("i64_j16", "f32"): (
                mmq.gguf_q5_k_mmq_i64_j16_forced_k256_q8_1_d4s4_f32_kmajor_bf16_f32_out
            ),
            ("mmq32", "bf16"): (
                mmq.gguf_q5_k_mmq32_q8_1_d4s4_f32_bf16_bf16_out
            ),
            ("mmq32_pipe", "bf16"): (
                mmq.gguf_q5_k_mmq32_pipe_q8_1_d4s4_f32_bf16_bf16_out
            ),
        }[(consumer, output_dtype)]
        fn(
            packed_dev.ptr,
            weight_dev.ptr,
            out_dev.ptr,
            rows,
            hidden,
            out_features,
            library=consumer_library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(out), out_dev, runtime=runtime)
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)
    after = memory_stats()
    assert after["current_allocated_bytes"] == before["current_allocated_bytes"]
    assert after["active_allocations"] == before["active_allocations"]
    actual = _bf16_to_f32(out) if output_dtype == "bf16" else out
    return actual, reference


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize(
    ("rows", "out_features", "output_dtype"),
    [(17, 72, "f32"), (127, 128, "bf16"), (129, 128, "f32")],
)
def test_q5_source_mmq_tails_are_finite_and_pass_exact_path_quality(
    rows: int, out_features: int, output_dtype: str
) -> None:
    actual, reference = _run_candidate(
        rows=rows,
        hidden=256,
        out_features=out_features,
        output_dtype=output_dtype,
    )
    assert np.all(np.isfinite(actual))
    result = evaluate_logits(reference, actual)
    assert result.kl_mean <= 0.05, result
    assert result.top1_agreement >= 0.90, result
    assert result.passed, result


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("rows", [24, 32])
def test_c8_q5_i64_j32_source_mmq_passes_production_outer_floor(rows: int) -> None:
    actual, reference = _run_candidate(
        rows=rows,
        hidden=256,
        out_features=128,
        output_dtype="bf16",
        consumer="i64_j32",
    )
    assert np.all(np.isfinite(actual))
    result = evaluate_logits(reference, actual)
    assert result.kl_mean <= 0.05, result
    assert result.top1_agreement >= 0.90, result
    assert result.passed, result


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_c8_q5_j16_minitile_outputs_match_j32_owner_bit_exact() -> None:
    """Tile-policy parity contract for the retained J16 diagnostic path.

    Both launch geometries keep the same per-(row, output) accumulation
    order, so the bf16 outputs must be identical even though the J16
    minitile doubles the R32 workgroup count. The J32 policy remains the
    registered default after the 8-run e2e gate could not confirm the leaf
    win end to end.
    """
    rows, hidden, out_features = 32, 256, 128
    default_out, reference = _run_candidate(
        rows=rows,
        hidden=hidden,
        out_features=out_features,
        output_dtype="bf16",
        consumer="i64_j32",
    )
    forced_out, _ = _run_candidate(
        rows=rows,
        hidden=hidden,
        out_features=out_features,
        output_dtype="bf16",
        consumer="i64_j16",
    )
    assert np.array_equal(default_out, forced_out), (
        "J16 minitile output differs from the J32 owner"
    )
    assert np.all(np.isfinite(default_out))
    result = evaluate_logits(reference, default_out)
    assert result.kl_mean <= 0.05, result
    assert result.top1_agreement >= 0.90, result
    assert result.passed, result


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_c8_q5_mmq32_pipe_outputs_match_serial_owner_bit_exact() -> None:
    """Tile-schedule parity contract for the retained pipelined diagnostic.

    The pipelined variant prefetches the next K32 tile while computing the
    current one; the per-(row, output) accumulation order is unchanged, so
    the bf16 outputs must be identical to the serial mb2 owner. The serial
    owner remains the registered default after the correctly fed (non-kmajor
    d4s4) leaf measured the pipelined variant neutral (1.0015x, 22/48).
    """
    rows, hidden, out_features = 24, 256, 128
    default_out, reference = _run_candidate(
        rows=rows,
        hidden=hidden,
        out_features=out_features,
        output_dtype="bf16",
        consumer="mmq32",
    )
    forced_out, _ = _run_candidate(
        rows=rows,
        hidden=hidden,
        out_features=out_features,
        output_dtype="bf16",
        consumer="mmq32_pipe",
    )
    os.environ["HIPENGINE_GGUF_Q5_MMQ32_PIPE"] = "0"
    try:
        rollback_out, _ = _run_candidate(
            rows=rows,
            hidden=hidden,
            out_features=out_features,
            output_dtype="bf16",
            consumer="mmq32",
        )
    finally:
        os.environ.pop("HIPENGINE_GGUF_Q5_MMQ32_PIPE", None)
    assert np.array_equal(default_out, forced_out), (
        "pipelined output differs from the serial owner"
    )
    assert np.array_equal(default_out, rollback_out), (
        "serial owner output changed under the rollback env"
    )
    assert np.all(np.isfinite(default_out))
    result = evaluate_logits(reference, default_out)
    assert result.kl_mean <= 0.05, result
    assert result.top1_agreement >= 0.90, result
    assert result.passed, result
