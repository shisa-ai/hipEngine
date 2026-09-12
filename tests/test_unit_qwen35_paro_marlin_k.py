from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.memory import DeviceBuffer
from hipengine.core.tensor import Tensor
from hipengine.kernels.hip_gfx1100.quant import (
    gemv_paro_marlin_k_fma_fp16,
    marlin_k_default_threads,
    plan_paro_marlin_k_build,
    register_paro_marlin_k_kernels,
)
from hipengine.kernels.registry import clear_registry_for_tests, resolve
from hipengine.loading.materialize import DeviceTensorAllocation, DeviceWeightMap, alias_device_allocation
from hipengine.loading.safetensors import TensorInfo
from hipengine.loading import (
    paro_marlin_k_pack8_decode_view,
    repack_paro_awq_to_marlin_k_host,
)


def setup_function() -> None:
    clear_registry_for_tests()


def test_repack_paro_awq_to_marlin_k_host_matches_v0_layout() -> None:
    group_size = 128
    groups = 2
    out_packed = 3
    qweight = np.arange(groups * group_size * out_packed, dtype=np.int32).reshape(groups * group_size, out_packed)
    qzeros = (1000 + np.arange(groups * out_packed, dtype=np.int32)).reshape(groups, out_packed)
    scales = (np.arange(groups * out_packed * 8, dtype=np.float16).reshape(groups, out_packed * 8) / np.float16(16.0))

    qweight_mk, qzeros_mk, scales_mk = repack_paro_awq_to_marlin_k_host(
        qweight,
        qzeros,
        scales,
        bits=4,
        group_size=group_size,
    )

    assert qweight_mk.shape == (out_packed, groups, group_size)
    assert qzeros_mk.shape == (out_packed, groups)
    assert scales_mk.shape == (out_packed, groups, 8)
    assert qweight_mk.dtype == np.int32
    assert qzeros_mk.dtype == np.int32
    assert scales_mk.dtype == np.float16
    assert qweight_mk.flags.c_contiguous
    assert qzeros_mk.flags.c_contiguous
    assert scales_mk.flags.c_contiguous
    np.testing.assert_array_equal(qweight_mk, qweight.reshape(groups, group_size, out_packed).transpose(2, 0, 1))
    np.testing.assert_array_equal(qzeros_mk, qzeros.T)
    np.testing.assert_array_equal(scales_mk, scales.reshape(groups, out_packed, 8).transpose(1, 0, 2))


def test_paro_marlin_k_pack8_decode_view_aliases_qweight_mk() -> None:
    qweight = np.arange(2 * 128 * 3, dtype=np.int32).reshape(2 * 128, 3)
    qzeros = np.zeros((2, 3), dtype=np.int32)
    scales = np.zeros((2, 24), dtype=np.float16)
    qweight_mk, _, _ = repack_paro_awq_to_marlin_k_host(qweight, qzeros, scales)

    pack8_view = paro_marlin_k_pack8_decode_view(qweight_mk)

    assert pack8_view.shape == (3, 256)
    assert pack8_view.dtype == np.int32
    assert pack8_view.flags.c_contiguous
    assert np.shares_memory(pack8_view, qweight_mk)
    np.testing.assert_array_equal(pack8_view, qweight_mk.reshape(3, 256))


def test_device_weight_map_frees_only_marlin_k_owner_alias_once() -> None:
    class FakeRuntime:
        def __init__(self) -> None:
            self.freed: list[int] = []

        def free(self, ptr: int) -> None:
            self.freed.append(ptr)

    source = TensorInfo(name="owner", shard_path=Path("/tmp/fake.safetensors"), dtype="I32", shape=(3, 2, 128))
    owner = DeviceTensorAllocation(
        name="owner",
        source=source,
        buffer=DeviceBuffer(ptr=0xABC000, nbytes=3 * 2 * 128 * 4),
        tensor=Tensor.from_handle(0xABC000, (3, 2, 128), DType.INT32, Device("hip", 0)),
    )
    alias = alias_device_allocation("pack8", owner, (3, 256), DType.INT32)
    runtime = FakeRuntime()

    assert alias.tensor.ptr == owner.tensor.ptr
    assert not alias.owns_buffer
    DeviceWeightMap({"owner": owner, "pack8": alias}).free(runtime=runtime)  # type: ignore[arg-type]

    assert runtime.freed == [owner.buffer.ptr]


def test_repack_paro_awq_to_marlin_k_host_rejects_shape_mismatches() -> None:
    qweight = np.zeros((127, 1), dtype=np.int32)
    qzeros = np.zeros((1, 1), dtype=np.int32)
    scales = np.zeros((1, 8), dtype=np.float16)

    with pytest.raises(ValueError, match="multiple of group_size"):
        repack_paro_awq_to_marlin_k_host(qweight, qzeros, scales)

    qweight = np.zeros((128, 2), dtype=np.int32)
    with pytest.raises(ValueError, match="qzeros shape"):
        repack_paro_awq_to_marlin_k_host(qweight, qzeros, np.zeros((1, 16), dtype=np.float16))

    with pytest.raises(ValueError, match="scales shape"):
        repack_paro_awq_to_marlin_k_host(qweight, np.zeros((1, 2), dtype=np.int32), scales)

    with pytest.raises(ValueError, match="bits=4"):
        repack_paro_awq_to_marlin_k_host(qweight, np.zeros((1, 2), dtype=np.int32), np.zeros((1, 16), dtype=np.float16), bits=8)


def test_paro_marlin_k_build_plan_is_dry_run_safe(tmp_path) -> None:
    artifact = plan_paro_marlin_k_build(
        cache_root=tmp_path / "cache",
        compiler_version="hipcc paro marlin k test version",
    )

    assert artifact.family == "paro_marlin_k"
    assert artifact.profile.name == "decode"
    assert artifact.profile.wavefront == 32
    assert artifact.flags[:2] == ("-mllvm", "-amdgpu-unroll-threshold-local=600")
    assert "-mcumode" in artifact.flags
    assert artifact.output_path.name == "paro_marlin_k.so"
    assert artifact.compiler_version == "hipcc paro marlin k test version"
    assert any(str(path).endswith("paro_marlin_k.hip") for path in artifact.sources)
    assert not artifact.cache_dir.exists()


def test_paro_marlin_k_registers_kernel_variant() -> None:
    register_paro_marlin_k_kernels()

    assert (
        resolve(
            backend="hip_gfx1100",
            layer="marlin_k_gemv",
            quant="w4_paro",
            variant="fma_fp16",
        )
        is gemv_paro_marlin_k_fma_fp16
    )


def test_paro_marlin_k_thread_policy_matches_parent_shape_rules() -> None:
    assert marlin_k_default_threads(4096, 4096) == 128
    assert marlin_k_default_threads(2048, 2048) == 128
    assert marlin_k_default_threads(1024, 512) == 128
    assert marlin_k_default_threads(1024, 4096) == 64
    with pytest.raises(ValueError, match="positive"):
        marlin_k_default_threads(0, 4096)


def test_paro_marlin_k_wrapper_validates_before_gpu_load() -> None:
    with pytest.raises(ValueError, match="rows must be positive"):
        gemv_paro_marlin_k_fma_fp16(0, 0, 0, 0, 0, 0, 128, 1)
    with pytest.raises(ValueError, match="group_size=128"):
        gemv_paro_marlin_k_fma_fp16(0, 0, 0, 0, 0, 1, 128, 1, group_size=64)
    with pytest.raises(ValueError, match="in_features must be divisible"):
        gemv_paro_marlin_k_fma_fp16(0, 0, 0, 0, 0, 1, 129, 1)
    with pytest.raises(ValueError, match="threads must be one of 32, 64, or 128"):
        gemv_paro_marlin_k_fma_fp16(0, 0, 0, 0, 0, 1, 128, 1, threads=256)
