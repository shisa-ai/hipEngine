from __future__ import annotations

import ctypes
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.core.memory import copy_host_to_device, free, host_array_ptr, malloc
from hipengine.runtime.qwen4_exp_runner import (
    Qwen4ExpGGUFResidentModelRunner,
    Qwen4ExpTokenResult,
)


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def test_qwen4_exp_token_result_exposes_last_widened_hidden_row() -> None:
    rows = np.arange(24, dtype=np.float32).reshape(3, 8)
    result = Qwen4ExpTokenResult(7, np.asarray([1.0], dtype=np.float32), rows)

    np.testing.assert_array_equal(result.hidden_seeds, rows)
    np.testing.assert_array_equal(result.hidden_seed, rows[-1])
    assert Qwen4ExpTokenResult(7, np.asarray([1.0], dtype=np.float32)).hidden_seed is None


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_qwen4_exp_hidden_capture_preserves_authoritative_bf16_rows() -> None:
    from hipengine.core.hip import get_hip_runtime
    from hipengine.loading.materialize import float_array_to_bf16_bits

    runtime = get_hip_runtime()
    values = np.asarray(
        [[-1.25, -0.5, 0.0, 0.75, 1.5, 2.0, 3.25, 4.5],
         [5.0, -6.0, 7.5, -8.5, 9.0, 10.0, -11.0, 12.0]],
        dtype=np.float32,
    )
    bits = np.ascontiguousarray(float_array_to_bf16_bits(values), dtype=np.uint16)
    device = malloc(bits.nbytes, runtime=runtime)
    try:
        copy_host_to_device(device, host_array_ptr(bits), runtime=runtime)
        runner = object.__new__(Qwen4ExpGGUFResidentModelRunner)
        runner.config = SimpleNamespace(residual_width=values.shape[1])
        runner.runtime = runtime

        captured = runner._read_hidden_seed_rows(device.ptr, values.shape[0])
    finally:
        free(device, runtime=runtime)

    expected = (bits.astype(np.uint32) << 16).view(np.float32)
    np.testing.assert_array_equal(captured, expected)
