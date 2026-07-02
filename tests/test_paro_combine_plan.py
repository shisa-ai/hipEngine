from __future__ import annotations

import ctypes

import numpy as np
import pytest

from hipengine.core.memory import copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc
from hipengine.kernels.hip_gfx1100.fused import (
    build_paro_combine,
    plan_paro_combine_build,
    register_paro_combine_kernels,
    shared_gate_combine_out_bf16,
    shared_gate_combine_out_fp16,
    shared_gate_combine_residual_batch_out_bf16,
    shared_gate_combine_residual_batch_out_fp16,
    shared_gate_combine_residual_out_bf16,
    shared_gate_combine_residual_out_fp16,
    weighted_lanes_sum_out_bf16_f32w,
    weighted_lanes_sum_out_fp16_f32w,
    weighted_sum_out_bf16_f32w,
    weighted_sum_out_fp16_f32w,
    weighted_sum_shared_gate_combine_residual_batch_out_bf16_f32w,
    weighted_sum_shared_gate_combine_residual_batch_out_f32_accum_f32w,
    weighted_sum_shared_gate_combine_residual_batch_out_f32_f32w,
    weighted_sum_shared_gate_combine_residual_batch_out_fp16_f32w,
    weighted_sum_shared_gate_combine_residual_out_bf16_f32w,
    weighted_sum_shared_gate_combine_residual_out_f32_accum_f32w,
    weighted_sum_shared_gate_combine_residual_out_f32_f32w,
    weighted_sum_shared_gate_combine_residual_out_fp16_f32w,
)
from hipengine.kernels.registry import clear_registry_for_tests, resolve


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def _bf16_bits(arr: np.ndarray) -> np.ndarray:
    u32 = np.ascontiguousarray(arr, dtype=np.float32).view(np.uint32)
    lsb = (u32 >> 16) & np.uint32(1)
    return ((u32 + np.uint32(0x7FFF) + lsb) >> 16).astype(np.uint16)


def _bf16_to_f32(bits: np.ndarray) -> np.ndarray:
    return (np.ascontiguousarray(bits, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def _dev(arr: np.ndarray):
    buf = malloc(arr.nbytes)
    copy_host_to_device(buf, host_array_ptr(arr), arr.nbytes)
    return buf


def setup_function() -> None:
    clear_registry_for_tests()


def test_paro_combine_registers_bf16_fp16_and_w4_paro_variants() -> None:
    register_paro_combine_kernels()

    for quant in ("bf16", "w4_paro"):
        assert (
            resolve(backend="hip_gfx1100", layer="weighted_lanes_sum", quant=quant, variant="out")
            is weighted_lanes_sum_out_bf16_f32w
        )
        assert (
            resolve(backend="hip_gfx1100", layer="weighted_lanes_sum", quant=quant, variant="out_fp16")
            is weighted_lanes_sum_out_fp16_f32w
        )
        assert (
            resolve(backend="hip_gfx1100", layer="weighted_sum", quant=quant, variant="out")
            is weighted_sum_out_bf16_f32w
        )
        assert (
            resolve(backend="hip_gfx1100", layer="weighted_sum", quant=quant, variant="out_fp16")
            is weighted_sum_out_fp16_f32w
        )
        assert (
            resolve(
                backend="hip_gfx1100",
                layer="weighted_sum+shared_gate+residual",
                quant=quant,
                variant="out",
            )
            is weighted_sum_shared_gate_combine_residual_out_bf16_f32w
        )
        assert (
            resolve(
                backend="hip_gfx1100",
                layer="weighted_sum+shared_gate+residual",
                quant=quant,
                variant="out_fp16",
            )
            is weighted_sum_shared_gate_combine_residual_out_fp16_f32w
        )
        assert (
            resolve(
                backend="hip_gfx1100",
                layer="weighted_sum+shared_gate+residual",
                quant=quant,
                variant="out_f32",
            )
            is weighted_sum_shared_gate_combine_residual_out_f32_f32w
        )
        assert (
            resolve(
                backend="hip_gfx1100",
                layer="weighted_sum+shared_gate+residual",
                quant=quant,
                variant="out_f32_accum",
            )
            is weighted_sum_shared_gate_combine_residual_out_f32_accum_f32w
        )
        assert (
            resolve(
                backend="hip_gfx1100",
                layer="weighted_sum+shared_gate+residual",
                quant=quant,
                variant="batch_out",
            )
            is weighted_sum_shared_gate_combine_residual_batch_out_bf16_f32w
        )
        assert (
            resolve(
                backend="hip_gfx1100",
                layer="weighted_sum+shared_gate+residual",
                quant=quant,
                variant="batch_out_fp16",
            )
            is weighted_sum_shared_gate_combine_residual_batch_out_fp16_f32w
        )
        assert (
            resolve(
                backend="hip_gfx1100",
                layer="weighted_sum+shared_gate+residual",
                quant=quant,
                variant="batch_out_f32",
            )
            is weighted_sum_shared_gate_combine_residual_batch_out_f32_f32w
        )
        assert (
            resolve(
                backend="hip_gfx1100",
                layer="weighted_sum+shared_gate+residual",
                quant=quant,
                variant="batch_out_f32_accum",
            )
            is weighted_sum_shared_gate_combine_residual_batch_out_f32_accum_f32w
        )
        assert (
            resolve(
                backend="hip_gfx1100",
                layer="shared_gate_combine",
                quant=quant,
                variant="out",
            )
            is shared_gate_combine_out_bf16
        )
        assert (
            resolve(
                backend="hip_gfx1100",
                layer="shared_gate_combine",
                quant=quant,
                variant="out_fp16",
            )
            is shared_gate_combine_out_fp16
        )
        assert (
            resolve(
                backend="hip_gfx1100",
                layer="shared_gate_combine+residual",
                quant=quant,
                variant="out",
            )
            is shared_gate_combine_residual_out_bf16
        )
        assert (
            resolve(
                backend="hip_gfx1100",
                layer="shared_gate_combine+residual",
                quant=quant,
                variant="out_fp16",
            )
            is shared_gate_combine_residual_out_fp16
        )
        assert (
            resolve(
                backend="hip_gfx1100",
                layer="shared_gate_combine+residual",
                quant=quant,
                variant="batch_out",
            )
            is shared_gate_combine_residual_batch_out_bf16
        )
        assert (
            resolve(
                backend="hip_gfx1100",
                layer="shared_gate_combine+residual",
                quant=quant,
                variant="batch_out_fp16",
            )
            is shared_gate_combine_residual_batch_out_fp16
        )
    assert resolve(backend="hip_gfx1100", layer="weighted_sum", quant="fp16", variant="out") is weighted_sum_out_fp16_f32w
    assert (
        resolve(backend="hip_gfx1100", layer="weighted_sum+shared_gate+residual", quant="f32", variant="out")
        is weighted_sum_shared_gate_combine_residual_out_f32_f32w
    )
    assert (
        resolve(backend="hip_gfx1100", layer="weighted_sum+shared_gate+residual", quant="f32", variant="out_accum")
        is weighted_sum_shared_gate_combine_residual_out_f32_accum_f32w
    )
    assert (
        resolve(backend="hip_gfx1100", layer="weighted_sum+shared_gate+residual", quant="f32", variant="batch_out")
        is weighted_sum_shared_gate_combine_residual_batch_out_f32_f32w
    )
    assert (
        resolve(backend="hip_gfx1100", layer="weighted_sum+shared_gate+residual", quant="f32", variant="batch_out_accum")
        is weighted_sum_shared_gate_combine_residual_batch_out_f32_accum_f32w
    )


def test_paro_combine_build_plan_is_dry_run_safe(tmp_path) -> None:
    artifact = plan_paro_combine_build(
        cache_root=tmp_path / "cache",
        compiler_version="hipcc paro combine test version",
    )

    assert artifact.family == "paro_combine"
    assert artifact.profile.name == "decode"
    assert artifact.profile.wavefront == 32
    assert artifact.flags[:2] == ("-mllvm", "-amdgpu-unroll-threshold-local=600")
    assert "-mcumode" in artifact.flags
    assert artifact.output_path.name == "paro_combine.so"
    assert artifact.compiler_version == "hipcc paro combine test version"
    assert any(str(path).endswith("paro_combine.hip") for path in artifact.sources)
    assert not artifact.cache_dir.exists()


def test_paro_combine_wrappers_validate_before_gpu_load() -> None:
    with pytest.raises(ValueError, match="tokens must be positive"):
        weighted_lanes_sum_out_bf16_f32w(0, 0, 0, 0, 0, 0, 8, 16)
    with pytest.raises(ValueError, match="top_k must be positive"):
        weighted_lanes_sum_out_fp16_f32w(0, 0, 0, 0, 0, 2, 0, 16)
    with pytest.raises(ValueError, match="rows must be positive"):
        weighted_sum_out_bf16_f32w(0, 0, 0, 0, 8)
    with pytest.raises(ValueError, match="features must be positive"):
        weighted_sum_shared_gate_combine_residual_out_bf16_f32w(0, 0, 0, 0, 0, 0, 2, 0)
    with pytest.raises(ValueError, match="threads must be one of"):
        shared_gate_combine_out_bf16(0, 0, 0, 0, 8, threads=32)
    with pytest.raises(ValueError, match="gate_stride must be positive"):
        weighted_sum_shared_gate_combine_residual_batch_out_bf16_f32w(0, 0, 0, 0, 0, 0, 2, 8, 16, 0)
    with pytest.raises(ValueError, match="features must be positive"):
        shared_gate_combine_residual_out_bf16(0, 0, 0, 0, 0, 0)
    with pytest.raises(ValueError, match="gate_stride must be positive"):
        shared_gate_combine_residual_batch_out_bf16(0, 0, 0, 0, 0, 2, 16, 0)
    with pytest.raises(ValueError, match="tokens must be positive"):
        shared_gate_combine_residual_batch_out_fp16(0, 0, 0, 0, 0, 0, 16, 129)
    with pytest.raises(ValueError, match="rows must be positive"):
        weighted_sum_out_fp16_f32w(0, 0, 0, 0, 8)
    with pytest.raises(ValueError, match="gate_stride must be positive"):
        weighted_sum_shared_gate_combine_residual_batch_out_fp16_f32w(0, 0, 0, 0, 0, 0, 2, 8, 16, 0)
    with pytest.raises(ValueError, match="features must be positive"):
        weighted_sum_shared_gate_combine_residual_out_f32_f32w(0, 0, 0, 0, 0, 0, 2, 0)
    with pytest.raises(ValueError, match="gate_stride must be positive"):
        weighted_sum_shared_gate_combine_residual_batch_out_f32_f32w(0, 0, 0, 0, 0, 0, 2, 8, 16, 0)
    with pytest.raises(ValueError, match="features must be positive"):
        weighted_sum_shared_gate_combine_residual_out_f32_accum_f32w(0, 0, 0, 0, 0, 0, 2, 0)
    with pytest.raises(ValueError, match="gate_stride must be positive"):
        weighted_sum_shared_gate_combine_residual_batch_out_f32_accum_f32w(0, 0, 0, 0, 0, 0, 2, 8, 16, 0)


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_f32_residual_combine_kernels_match_cpu_reference() -> None:
    rng = np.random.default_rng(20260702)
    rows = 3
    top_k = 2
    features = 17
    gate_stride = 5
    values = _bf16_bits(rng.normal(size=(rows * top_k, features)).astype(np.float32))
    weights = rng.normal(size=(rows * top_k,)).astype(np.float32)
    shared = _bf16_bits(rng.normal(size=(rows, features)).astype(np.float32))
    gate_logits = rng.normal(size=(rows, gate_stride)).astype(np.float32)
    residual = rng.normal(size=(rows, features)).astype(np.float32)
    out_single = np.empty((features,), dtype=np.float32)
    out_single_accum = np.empty((features,), dtype=np.float32)
    out_batch = np.empty((rows, features), dtype=np.float32)
    out_batch_accum = np.empty((rows, features), dtype=np.float32)

    values_f32 = _bf16_to_f32(values).reshape(rows, top_k, features)
    shared_f32 = _bf16_to_f32(shared)
    selected_accum = np.sum(values_f32 * weights.reshape(rows, top_k, 1), axis=1, dtype=np.float32)
    selected_rounded = _bf16_to_f32(_bf16_bits(selected_accum))
    gate = 1.0 / (1.0 + np.exp(-gate_logits[:, 0:1]))
    expected = residual + selected_rounded + gate * shared_f32
    expected_accum = residual + selected_accum + gate * shared_f32

    library = build_paro_combine(load=True)
    bufs = []
    try:
        values_d = _dev(np.ascontiguousarray(values))
        weights_d = _dev(np.ascontiguousarray(weights))
        shared_d = _dev(np.ascontiguousarray(shared))
        gate_d = _dev(np.ascontiguousarray(gate_logits))
        residual_d = _dev(np.ascontiguousarray(residual))
        out_single_d = _dev(out_single)
        out_single_accum_d = _dev(out_single_accum)
        out_batch_d = _dev(out_batch)
        out_batch_accum_d = _dev(out_batch_accum)
        bufs.extend(
            (
                values_d,
                weights_d,
                shared_d,
                gate_d,
                residual_d,
                out_single_d,
                out_single_accum_d,
                out_batch_d,
                out_batch_accum_d,
            )
        )

        weighted_sum_shared_gate_combine_residual_out_f32_f32w(
            values_d.ptr,
            weights_d.ptr,
            shared_d.ptr,
            gate_d.ptr,
            residual_d.ptr,
            out_single_d.ptr,
            top_k,
            features,
            library=library,
        )
        weighted_sum_shared_gate_combine_residual_out_f32_accum_f32w(
            values_d.ptr,
            weights_d.ptr,
            shared_d.ptr,
            gate_d.ptr,
            residual_d.ptr,
            out_single_accum_d.ptr,
            top_k,
            features,
            library=library,
        )
        weighted_sum_shared_gate_combine_residual_batch_out_f32_f32w(
            values_d.ptr,
            weights_d.ptr,
            shared_d.ptr,
            gate_d.ptr,
            residual_d.ptr,
            out_batch_d.ptr,
            rows,
            top_k,
            features,
            gate_stride,
            library=library,
        )
        weighted_sum_shared_gate_combine_residual_batch_out_f32_accum_f32w(
            values_d.ptr,
            weights_d.ptr,
            shared_d.ptr,
            gate_d.ptr,
            residual_d.ptr,
            out_batch_accum_d.ptr,
            rows,
            top_k,
            features,
            gate_stride,
            library=library,
        )
        copy_device_to_host(host_array_ptr(out_single), out_single_d, out_single.nbytes)
        copy_device_to_host(host_array_ptr(out_single_accum), out_single_accum_d, out_single_accum.nbytes)
        copy_device_to_host(host_array_ptr(out_batch), out_batch_d, out_batch.nbytes)
        copy_device_to_host(host_array_ptr(out_batch_accum), out_batch_accum_d, out_batch_accum.nbytes)
    finally:
        for buf in reversed(bufs):
            free(buf)

    np.testing.assert_allclose(out_single, expected[0], rtol=1.0e-6, atol=1.0e-6)
    np.testing.assert_allclose(out_single_accum, expected_accum[0], rtol=1.0e-6, atol=1.0e-6)
    np.testing.assert_allclose(out_batch, expected, rtol=1.0e-6, atol=1.0e-6)
    np.testing.assert_allclose(out_batch_accum, expected_accum, rtol=1.0e-6, atol=1.0e-6)
