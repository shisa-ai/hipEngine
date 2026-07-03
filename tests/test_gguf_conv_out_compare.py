from __future__ import annotations

import numpy as np
import pytest

from scripts.gguf_conv_out_compare import compare_conv_out


def test_compare_conv_out_matches_silu_convolution_window() -> None:
    weight = np.asarray(
        [
            [1.0, -2.0, 0.5],
            [0.25, 0.75, -1.5],
        ],
        dtype=np.float32,
    )
    window = np.asarray(
        [
            [0.25, -0.5],
            [2.0, 0.125],
            [-1.0, 1.5],
        ],
        dtype=np.float32,
    )
    acc = np.sum(window * weight.T, axis=0, dtype=np.float32)
    device = acc / (np.float32(1.0) + np.exp(-acc, dtype=np.float32))

    comparison = compare_conv_out(weight, window, device)

    assert comparison["device_vs_cpu"]["max_abs_diff"] == 0.0
    assert comparison["device_vs_cpu"]["count"] == 2
    assert comparison["samples"]["device_conv_out"] == pytest.approx(device.tolist())


def test_compare_conv_out_rejects_shape_mismatches() -> None:
    weight = np.ones((2, 3), dtype=np.float32)
    window = np.ones((3, 2), dtype=np.float32)
    output = np.ones((2,), dtype=np.float32)

    with pytest.raises(ValueError, match="2D"):
        compare_conv_out(weight.reshape(1, 2, 3), window, output)
    with pytest.raises(ValueError, match=r"\[kernel, channels\]"):
        compare_conv_out(weight, window[:2], output)
    with pytest.raises(ValueError, match="channels"):
        compare_conv_out(weight, window, output[:1])
