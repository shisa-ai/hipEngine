from __future__ import annotations

import ctypes

import numpy as np
import pytest

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc
from hipengine.kernels.hip_gfx1100.fused import (
    build_paro_silu,
    plan_paro_silu_build,
    register_paro_silu_kernels,
    silu_mul_dual_out_bf16,
    silu_mul_dual_out_fp16,
    silu_mul_dual_rotate_out_bf16,
    silu_mul_dual_rotate_out_fp16,
    silu_mul_pair_rotate_out_bf16,
    silu_mul_pair_rotate_out_fp16,
    silu_mul_separate_out_bf16,
    silu_mul_separate_out_f32,
    silu_mul_separate_out_fp16,
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


def _dev(arr: np.ndarray, runtime, bufs: list) -> object:
    buf = malloc(arr.nbytes, runtime=runtime)
    copy_host_to_device(buf, host_array_ptr(arr), arr.nbytes, runtime=runtime)
    bufs.append(buf)
    return buf


def setup_function() -> None:
    clear_registry_for_tests()


def test_paro_silu_registers_bf16_fp16_and_w4_paro_variants() -> None:
    register_paro_silu_kernels()

    for quant in ("bf16", "w4_paro"):
        assert (
            resolve(backend="hip_gfx1100", layer="silu_mul_dual", quant=quant, variant="out")
            is silu_mul_dual_out_bf16
        )
        assert (
            resolve(backend="hip_gfx1100", layer="silu_mul_dual", quant=quant, variant="out_fp16")
            is silu_mul_dual_out_fp16
        )
        assert (
            resolve(
                backend="hip_gfx1100",
                layer="silu_mul_dual_rotate",
                quant=quant,
                variant="out",
            )
            is silu_mul_dual_rotate_out_bf16
        )
        assert (
            resolve(
                backend="hip_gfx1100",
                layer="silu_mul_dual_rotate",
                quant=quant,
                variant="out_fp16",
            )
            is silu_mul_dual_rotate_out_fp16
        )
        assert (
            resolve(
                backend="hip_gfx1100",
                layer="silu_mul_pair_rotate",
                quant=quant,
                variant="out",
            )
            is silu_mul_pair_rotate_out_bf16
        )
        assert (
            resolve(
                backend="hip_gfx1100",
                layer="silu_mul_pair_rotate",
                quant=quant,
                variant="out_fp16",
            )
            is silu_mul_pair_rotate_out_fp16
        )
        assert (
            resolve(
                backend="hip_gfx1100",
                layer="silu_mul_separate",
                quant=quant,
                variant="out",
            )
            is silu_mul_separate_out_bf16
        )
        assert (
            resolve(
                backend="hip_gfx1100",
                layer="silu_mul_separate",
                quant=quant,
                variant="out_fp16",
            )
            is silu_mul_separate_out_fp16
        )
        assert (
            resolve(
                backend="hip_gfx1100",
                layer="silu_mul_separate",
                quant=quant,
                variant="out_f32",
            )
            is silu_mul_separate_out_f32
        )
    assert resolve(backend="hip_gfx1100", layer="silu_mul_dual", quant="fp16", variant="out") is silu_mul_dual_out_fp16
    assert (
        resolve(backend="hip_gfx1100", layer="silu_mul_separate", quant="fp16", variant="out")
        is silu_mul_separate_out_fp16
    )
    assert (
        resolve(backend="hip_gfx1100", layer="silu_mul_separate", quant="bf16", variant="out_f32")
        is silu_mul_separate_out_f32
    )


def test_paro_silu_build_plan_is_dry_run_safe(tmp_path) -> None:
    artifact = plan_paro_silu_build(
        cache_root=tmp_path / "cache",
        compiler_version="hipcc paro silu test version",
    )

    assert artifact.family == "paro_silu"
    assert artifact.profile.name == "decode"
    assert artifact.profile.wavefront == 32
    assert artifact.flags[:2] == ("-mllvm", "-amdgpu-unroll-threshold-local=600")
    assert "-mcumode" in artifact.flags
    assert artifact.output_path.name == "paro_silu.so"
    assert artifact.compiler_version == "hipcc paro silu test version"
    assert any(str(path).endswith("paro_silu.hip") for path in artifact.sources)
    assert not artifact.cache_dir.exists()


def test_paro_silu_wrappers_validate_before_gpu_load() -> None:
    with pytest.raises(ValueError, match="rows must be positive"):
        silu_mul_dual_out_bf16(0, 0, 0, 8)
    with pytest.raises(ValueError, match="threads must be one of"):
        silu_mul_dual_out_bf16(0, 0, 1, 8, threads=32)
    with pytest.raises(ValueError, match="rows must be positive"):
        silu_mul_dual_out_fp16(0, 0, 0, 8)
    with pytest.raises(ValueError, match="group_size must be even"):
        silu_mul_dual_rotate_out_bf16(0, 0, 0, 0, 0, 1, 8, 3, 1)
    with pytest.raises(ValueError, match="features must be divisible"):
        silu_mul_dual_rotate_out_bf16(0, 0, 0, 0, 0, 1, 10, 8, 1)
    with pytest.raises(ValueError, match="krot must be positive"):
        silu_mul_pair_rotate_out_bf16(0, 0, 0, 0, 0, 0, 1, 8, 8, 0)
    with pytest.raises(ValueError, match="krot must be positive"):
        silu_mul_pair_rotate_out_fp16(0, 0, 0, 0, 0, 0, 1, 8, 8, 0)
    with pytest.raises(ValueError, match="rows must be positive"):
        silu_mul_separate_out_bf16(0, 0, 0, 0, 8)
    with pytest.raises(ValueError, match="threads must be one of"):
        silu_mul_separate_out_bf16(0, 0, 0, 1, 8, threads=32)
    with pytest.raises(ValueError, match="features must be positive"):
        silu_mul_separate_out_fp16(0, 0, 0, 1, 0)
    with pytest.raises(ValueError, match="rows must be positive"):
        silu_mul_separate_out_f32(0, 0, 0, 0, 8)
    with pytest.raises(ValueError, match="threads must be one of"):
        silu_mul_separate_out_f32(0, 0, 0, 1, 8, threads=32)


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_silu_mul_separate_out_f32_matches_cpu_reference() -> None:
    rng = np.random.default_rng(20260702)
    rows = 3
    features = 19
    gate = _bf16_bits(rng.normal(size=(rows, features)).astype(np.float32))
    up = _bf16_bits(rng.normal(size=(rows, features)).astype(np.float32))
    out = np.empty((rows, features), dtype=np.float32)

    runtime = get_hip_runtime()
    library = build_paro_silu(load=True)
    bufs: list = []
    try:
        gate_d = _dev(np.ascontiguousarray(gate), runtime, bufs)
        up_d = _dev(np.ascontiguousarray(up), runtime, bufs)
        out_d = malloc(out.nbytes, runtime=runtime)
        bufs.append(out_d)
        silu_mul_separate_out_f32(
            gate_d.ptr,
            up_d.ptr,
            out_d.ptr,
            rows,
            features,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(out), out_d, runtime=runtime)
    finally:
        for buf in reversed(bufs):
            free(buf, runtime=runtime)

    gate_f32 = _bf16_to_f32(gate)
    up_f32 = _bf16_to_f32(up)
    expected = gate_f32 * (1.0 / (1.0 + np.exp(-gate_f32))) * up_f32
    np.testing.assert_allclose(out, expected, rtol=1.0e-6, atol=1.0e-6)


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_silu_mul_separate_bf16_may_replace_gate_in_place() -> None:
    rng = np.random.default_rng(20260814)
    rows = 7
    features = 259
    gate = _bf16_bits(rng.normal(size=(rows, features)).astype(np.float32))
    up = _bf16_bits(rng.normal(size=(rows, features)).astype(np.float32))
    actual = np.empty_like(gate)

    runtime = get_hip_runtime()
    library = build_paro_silu(load=True)
    bufs: list = []
    try:
        gate_d = _dev(np.ascontiguousarray(gate), runtime, bufs)
        up_d = _dev(np.ascontiguousarray(up), runtime, bufs)
        silu_mul_separate_out_bf16(
            gate_d.ptr,
            up_d.ptr,
            gate_d.ptr,
            rows,
            features,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(actual), gate_d, runtime=runtime)
    finally:
        for buf in reversed(bufs):
            free(buf, runtime=runtime)

    gate_f32 = _bf16_to_f32(gate)
    up_f32 = _bf16_to_f32(up)
    expected = _bf16_bits(
        gate_f32 * (1.0 / (1.0 + np.exp(-gate_f32))) * up_f32
    )
    np.testing.assert_array_equal(actual, expected)
