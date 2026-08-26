from __future__ import annotations

import ctypes
from math import prod
from pathlib import Path

import numpy as np
import pytest

from hipengine.core.dtype import DType
from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.cpu_reference.qwen4_exp import gr_read
from hipengine.loading.gguf import GGUFTensorInfo
from hipengine.loading.materialize import (
    float_array_to_bf16_bits,
    load_host_array_to_device_as_dtype,
)
from hipengine.loading.qwen4_exp_gguf import Qwen4ExpGGUFTensorRef
from hipengine.loading.qwen4_exp_materialize import (
    Qwen4ExpDeviceWeight,
    Qwen4ExpGGUFWeightSpec,
)
from hipengine.quant.gguf import GGMLQuantizationType, bf16_to_float32
from hipengine.runtime.qwen4_exp_runner import (
    Qwen4ExpGRScratch,
    run_qwen4_exp_gr_read,
)


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_qwen4_exp_runner_gr_read_composes_dispatch_and_native_primitives() -> None:
    from hipengine.core.hip import get_hip_runtime

    runtime = get_hip_runtime()
    rng = np.random.default_rng(4038)
    rows, branches, hidden, low_rank = 3, 2, 4, 3
    residual_bits = float_array_to_bf16_bits(
        rng.normal(0.0, 0.4, size=(rows, branches, hidden)).astype(np.float32)
    )
    residual = bf16_to_float32(residual_bits)
    norm = rng.normal(1.0, 0.05, size=(branches, hidden)).astype(np.float32)
    down = rng.normal(0.0, 0.2, size=(low_rank, branches * hidden)).astype(np.float32)
    up = rng.normal(0.0, 0.2, size=(branches * hidden, low_rank)).astype(np.float32)
    inject = rng.normal(0.0, 0.2, size=(branches, branches * hidden)).astype(np.float32)
    expected = gr_read(residual, norm, down, up, inject)

    allocations = []
    scratch = None
    try:
        d_residual = _upload(residual_bits, runtime, allocations)
        d_norm = _upload(norm, runtime, allocations)
        down_weight = _dense_f32_weight("down", down, runtime, allocations)
        up_weight = _dense_f32_weight("up", up, runtime, allocations)
        inject_weight = _dense_f32_weight("inject", inject, runtime, allocations)
        scratch = Qwen4ExpGRScratch.allocate(
            rows=rows,
            branches=branches,
            hidden=hidden,
            low_rank=low_rank,
            runtime=runtime,
        )
        result = run_qwen4_exp_gr_read(
            d_residual.ptr,
            d_norm.ptr,
            down_weight,
            up_weight,
            inject_weight,
            scratch,
            rows=rows,
            branches=branches,
            hidden=hidden,
            low_rank=low_rank,
            runtime=runtime,
        )
        runtime.device_synchronize()
        actual_norm = _download(
            result.normalized,
            expected.normalized.shape,
            np.float32,
            runtime,
        )
        actual_gate = _download(result.gate, expected.gate.shape, np.float32, runtime)
        actual_mixed = _download(result.mixed, expected.mixed.shape, np.float32, runtime)
        actual_inject = _download(
            result.inject_logits,
            expected.inject_logits.shape,
            np.float32,
            runtime,
        )
    finally:
        if scratch is not None:
            scratch.close()
            scratch.close()
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)

    np.testing.assert_allclose(actual_norm, expected.normalized, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(actual_gate, expected.gate, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(actual_mixed, expected.mixed, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(actual_inject, expected.inject_logits, rtol=2e-5, atol=2e-5)
    assert scratch.closed


def _dense_f32_weight(name, array, runtime, allocations):
    allocation = load_host_array_to_device_as_dtype(
        name,
        array,
        DType.FP32,
        source_dtype="F32",
        runtime=runtime,
    )
    allocations.append(allocation.buffer)
    shape = tuple(int(value) for value in array.shape)
    tensor = GGUFTensorInfo(
        name=f"{name}.weight",
        shape=shape,
        ggml_shape=tuple(reversed(shape)),
        ggml_type=int(GGMLQuantizationType.F32),
        ggml_type_name="F32",
        n_elements=prod(shape),
        nbytes=int(array.nbytes),
        offset=0,
        data_offset=0,
        byte_shape=shape,
    )
    spec = Qwen4ExpGGUFWeightSpec(
        slot_path=name,
        source_ref=Qwen4ExpGGUFTensorRef(0, Path("synthetic.gguf"), tensor),
        quant_key="f32",
        layout="dense_f32",
        allocation_names=("raw",),
        device_resident=True,
        device_nbytes=int(array.nbytes),
    )
    return Qwen4ExpDeviceWeight(spec, "hip_gfx1151", {"raw": allocation})


def _upload(array: np.ndarray, runtime, allocations):
    host = np.ascontiguousarray(array)
    device = malloc(host.nbytes, runtime=runtime)
    allocations.append(device)
    copy_host_to_device(device, host_array_ptr(host), runtime=runtime)
    return device


def _download(device, shape, dtype, runtime):
    host = np.empty(shape, dtype=dtype)
    copy_device_to_host(host_array_ptr(host), device, runtime=runtime)
    return host
