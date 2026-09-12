from __future__ import annotations

import numpy as np
import pytest

from hipengine.loading.materialize import float_array_to_bf16_bits
from hipengine.quant.gguf import bf16_to_float32
from scripts.gguf_ssm_out_compare import compare_ssm_out


def test_compare_ssm_out_matches_bf16_rounded_cpu_matmul() -> None:
    weight = np.asarray(
        [
            [1.0, -2.0, 0.5],
            [0.25, 0.75, -1.5],
        ],
        dtype=np.float32,
    )
    recurrent = np.asarray([0.25, -0.5, 2.0], dtype=np.float32)
    cpu_f32 = np.matmul(weight, recurrent).astype(np.float32)
    device = bf16_to_float32(float_array_to_bf16_bits(cpu_f32)).astype(np.float32)

    comparison = compare_ssm_out(weight, recurrent, device)

    assert comparison["device_vs_cpu_bf16"]["max_abs_diff"] == 0.0
    assert comparison["cpu_f32_vs_cpu_bf16"]["count"] == 2
    assert comparison["samples"]["device_attn_out"] == pytest.approx(device.tolist())


def test_compare_ssm_out_rejects_shape_mismatches() -> None:
    weight = np.ones((2, 3), dtype=np.float32)
    recurrent = np.ones((3,), dtype=np.float32)
    output = np.ones((2,), dtype=np.float32)

    with pytest.raises(ValueError, match="2D matrix"):
        compare_ssm_out(weight.reshape(1, 2, 3), recurrent, output)
    with pytest.raises(ValueError, match="weight columns"):
        compare_ssm_out(weight, recurrent[:2], output)
    with pytest.raises(ValueError, match="weight rows"):
        compare_ssm_out(weight, recurrent, output[:1])
