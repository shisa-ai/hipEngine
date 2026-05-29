from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np
import pytest

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc
from hipengine.kernels.cpu_reference import gguf_q3_k_gemv
from hipengine.kernels.hip_gfx1100.quant.gguf_k_gemv import (
    build_gguf_k_gemv,
    gguf_q3_k_gemv_bf16_f32_out,
    gguf_q3_k_selected_gemv_bf16_bf16_out,
    plan_gguf_k_gemv_build,
)
from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
from hipengine.kernels.registry import KernelKey, resolve
from hipengine.loading.gguf import GGUFReader, scan_gguf_splits
from hipengine.loading.materialize import float_array_to_bf16_bits
from hipengine.loading.stepfun_gguf import build_stepfun_gguf_tensor_map
from hipengine.quant.gguf import bf16_to_float32
from hipengine.runtime.gguf_linear import resolve_gguf_linear_dispatch
from tests.test_gguf_k_gemv import make_q3_k_weight

DEFAULT_STEPFUN_GGUF_DIR = Path("/data/models/gguf")


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


HIP_AVAILABLE = _hip_available()


def _dev(array: np.ndarray, runtime, bufs: list) -> object:
    contiguous = np.ascontiguousarray(array)
    buf = malloc(contiguous.nbytes, runtime=runtime)
    copy_host_to_device(buf, host_array_ptr(contiguous), runtime=runtime)
    bufs.append(buf)
    return buf


def _stepfun_gguf_paths() -> tuple[Path, ...]:
    root = Path(os.environ.get("HIPENGINE_STEPFUN_GGUF_DIR", DEFAULT_STEPFUN_GGUF_DIR))
    paths = tuple(sorted(root.glob("Step-3.7-flash-Q3_K_L-*.gguf")))
    if len(paths) != 3:
        pytest.skip(
            "StepFun GGUF Q3_K_L shards not found; set HIPENGINE_STEPFUN_GGUF_DIR "
            "to a directory containing Step-3.7-flash-Q3_K_L-00001..00003.gguf"
        )
    return paths


def _bf16_reference(x: np.ndarray, qweight: np.ndarray) -> np.ndarray:
    rounded_x = bf16_to_float32(float_array_to_bf16_bits(x))
    return gguf_q3_k_gemv(rounded_x, qweight)


def test_stepfun_q3k_kernel_plan_and_gfx1151_registration() -> None:
    artifact = plan_gguf_k_gemv_build(compiler_version="test-compiler")
    assert artifact.output_path.name == "gguf_k_gemv.so"
    assert any(path.name == "gguf_k_gemv.hip" for path in artifact.sources)

    register_gfx1151_kernels()
    assert resolve(
        backend="hip_gfx1151",
        layer="linear",
        quant="gguf_q3_k",
        variant="gemv_bf16_f32_out",
    ) is gguf_q3_k_gemv_bf16_f32_out
    assert resolve(
        backend="hip_gfx1151",
        layer="linear",
        quant="gguf_q3_k",
        variant="selected_gemv_bf16_bf16_out",
    ) is gguf_q3_k_selected_gemv_bf16_bf16_out

    fake_weight = type(
        "Weight",
        (),
        {"spec": type("Spec", (), {"layout": "raw_gguf", "quant_key": "gguf_q3_k"})()},
    )()
    dispatch = resolve_gguf_linear_dispatch(fake_weight, backend="hip_gfx1151")
    assert dispatch.key == KernelKey("hip_gfx1151", "linear", "gguf_q3_k", "gemv_bf16_bf16_out")
    assert dispatch.abi == "raw"


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_stepfun_q3k_synthetic_hip_gemv_matches_cpu_reference() -> None:
    runtime = get_hip_runtime()
    library = build_gguf_k_gemv(load=True)
    rows, in_features, out_features = 2, 512, 7
    x = ((np.arange(rows * in_features, dtype=np.float32).reshape(rows, in_features) % 19) - 9) / 32.0
    qweight = make_q3_k_weight(out_features=out_features, in_features=in_features)
    expected = _bf16_reference(x, qweight)
    actual = np.zeros((rows, out_features), dtype=np.float32)
    bufs: list = []
    try:
        x_dev = _dev(float_array_to_bf16_bits(x), runtime, bufs)
        q_dev = _dev(qweight, runtime, bufs)
        out_dev = malloc(actual.nbytes, runtime=runtime)
        bufs.append(out_dev)
        gguf_q3_k_gemv_bf16_f32_out(
            x_dev.ptr,
            q_dev.ptr,
            out_dev.ptr,
            rows=rows,
            in_features=in_features,
            out_features=out_features,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(actual), out_dev, runtime=runtime)
    finally:
        for buf in reversed(bufs):
            free(buf, runtime=runtime)

    np.testing.assert_allclose(actual, expected, rtol=1.5e-4, atol=1.5e-4)


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_stepfun_q3k_selected_expert_hip_matches_cpu_reference() -> None:
    runtime = get_hip_runtime()
    library = build_gguf_k_gemv(load=True)
    x_rows, rows, num_experts = 2, 4, 3
    in_features, out_features = 512, 8
    experts = np.stack(
        [make_q3_k_weight(out_features=out_features, in_features=in_features) for _ in range(num_experts)],
        axis=0,
    )
    # Make experts distinct without violating block layout by regenerating with
    # different output offsets through a row roll.
    experts[1] = np.roll(experts[1], shift=1, axis=0)
    experts[2] = np.roll(experts[2], shift=2, axis=0)
    x = ((np.arange(x_rows * in_features, dtype=np.float32).reshape(x_rows, in_features) % 23) - 11) / 64.0
    selected = np.asarray([0, 2, 1, 2], dtype=np.int64)
    rounded_x = bf16_to_float32(float_array_to_bf16_bits(x))
    expected_f32 = np.zeros((rows, out_features), dtype=np.float32)
    for row, expert in enumerate(selected):
        x_row = row // (rows // x_rows)
        expected_f32[row] = gguf_q3_k_gemv(rounded_x[x_row : x_row + 1], experts[int(expert)])[0]
    expected_bits = float_array_to_bf16_bits(expected_f32)
    actual_bits = np.zeros_like(expected_bits)
    bufs: list = []
    try:
        x_dev = _dev(float_array_to_bf16_bits(x), runtime, bufs)
        selected_dev = _dev(selected, runtime, bufs)
        q_dev = _dev(experts, runtime, bufs)
        out_dev = malloc(actual_bits.nbytes, runtime=runtime)
        bufs.append(out_dev)
        gguf_q3_k_selected_gemv_bf16_bf16_out(
            x_dev.ptr,
            selected_dev.ptr,
            q_dev.ptr,
            out_dev.ptr,
            x_rows=x_rows,
            rows=rows,
            num_experts=num_experts,
            in_features=in_features,
            out_features=out_features,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(actual_bits), out_dev, runtime=runtime)
    finally:
        for buf in reversed(bufs):
            free(buf, runtime=runtime)

    np.testing.assert_allclose(bf16_to_float32(actual_bits), bf16_to_float32(expected_bits), rtol=0.0, atol=0.0)


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_stepfun_real_q3k_tensor_slice_hip_matches_cpu_reference() -> None:
    info = scan_gguf_splits(_stepfun_gguf_paths())
    model_map = build_stepfun_gguf_tensor_map(info)
    tensor = model_map.layer(0).tensor("attn_q")
    raw = np.ascontiguousarray(GGUFReader(tensor.source_path).tensor_data(tensor.name)[:4])
    in_features = 4096
    out_features = raw.shape[0]
    x = ((np.arange(in_features, dtype=np.float32).reshape(1, in_features) % 29) - 14) / 128.0
    expected = _bf16_reference(x, raw)
    actual = np.zeros((1, out_features), dtype=np.float32)
    runtime = get_hip_runtime()
    library = build_gguf_k_gemv(load=True)
    bufs: list = []
    try:
        x_dev = _dev(float_array_to_bf16_bits(x), runtime, bufs)
        q_dev = _dev(raw, runtime, bufs)
        out_dev = malloc(actual.nbytes, runtime=runtime)
        bufs.append(out_dev)
        gguf_q3_k_gemv_bf16_f32_out(
            x_dev.ptr,
            q_dev.ptr,
            out_dev.ptr,
            rows=1,
            in_features=in_features,
            out_features=out_features,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(actual), out_dev, runtime=runtime)
    finally:
        for buf in reversed(bufs):
            free(buf, runtime=runtime)

    np.testing.assert_allclose(actual, expected, rtol=2.0e-3, atol=2.0e-3)
