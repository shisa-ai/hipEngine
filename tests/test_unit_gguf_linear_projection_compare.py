from __future__ import annotations

import numpy as np
import pytest

from hipengine.loading.materialize import float_array_to_bf16_bits
from hipengine.quant.gguf import bf16_to_float32
from scripts.gguf_linear_projection_compare import compare_projection


def test_compare_projection_matches_bf16_rounded_cpu_matmul() -> None:
    weight = np.asarray(
        [
            [1.0, -2.0, 0.5],
            [0.25, 0.75, -1.5],
        ],
        dtype=np.float32,
    )
    hidden = np.asarray([0.25, -0.5, 2.0], dtype=np.float32)
    cpu_f32 = np.matmul(weight, hidden).astype(np.float32)
    device = bf16_to_float32(float_array_to_bf16_bits(cpu_f32)).astype(np.float32)

    comparison = compare_projection(weight, hidden, device)

    assert comparison["device_vs_cpu_bf16"]["max_abs_diff"] == 0.0
    assert comparison["cpu_f32_vs_cpu_bf16"]["count"] == 2
    assert comparison["samples"]["device_output"] == pytest.approx(device.tolist())


def test_compare_projection_rejects_shape_mismatches() -> None:
    weight = np.ones((2, 3), dtype=np.float32)
    hidden = np.ones((3,), dtype=np.float32)
    output = np.ones((2,), dtype=np.float32)

    with pytest.raises(ValueError, match="2D matrix"):
        compare_projection(weight.reshape(1, 2, 3), hidden, output)
    with pytest.raises(ValueError, match="weight columns"):
        compare_projection(weight, hidden[:2], output)
    with pytest.raises(ValueError, match="weight rows"):
        compare_projection(weight, hidden, output[:1])
