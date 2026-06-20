from __future__ import annotations

import numpy as np
import pytest

from hipengine.loading.materialize import float_array_to_bf16_bits
from hipengine.quant.gguf import bf16_to_float32
from scripts.gguf_layer_moe_combine_compare import compare_moe_combine


def test_compare_moe_combine_matches_weighted_shared_residual_formula() -> None:
    residual = np.asarray([0.25, -0.5, 1.0], dtype=np.float32)
    selected = np.asarray([[1.0, -2.0, 0.5], [0.25, 0.75, -1.5]], dtype=np.float32)
    weights = np.asarray([0.6, 0.4], dtype=np.float32)
    shared = np.asarray([0.125, -0.25, 0.5], dtype=np.float32)
    gate = np.asarray([-0.75], dtype=np.float32)
    weighted = _round_to_bf16(np.sum(selected * weights[:, None], axis=0, dtype=np.float32))
    layer_out = _round_to_bf16(residual + weighted + _sigmoid(float(gate[0])) * shared)

    comparison = compare_moe_combine(
        residual=residual,
        selected_down=selected,
        routing_weights=weights,
        shared_out=shared,
        shared_gate=gate,
        layer_out=layer_out,
    )

    assert comparison["layer_out_vs_cpu"]["max_abs_diff"] == 0.0
    assert comparison["weighted_selected_vs_bf16"]["count"] == 3
    assert comparison["samples"]["device_layer_out"] == pytest.approx(layer_out.tolist())


def test_compare_moe_combine_rejects_shape_mismatches() -> None:
    residual = np.ones((3,), dtype=np.float32)
    selected = np.ones((2, 3), dtype=np.float32)
    weights = np.ones((2,), dtype=np.float32)
    shared = np.ones((3,), dtype=np.float32)
    gate = np.ones((1,), dtype=np.float32)

    with pytest.raises(ValueError, match="selected_down"):
        compare_moe_combine(
            residual=residual,
            selected_down=selected.reshape(1, 2, 3),
            routing_weights=weights,
            shared_out=shared,
            shared_gate=gate,
            layer_out=residual,
        )
    with pytest.raises(ValueError, match="routing_weights"):
        compare_moe_combine(
            residual=residual,
            selected_down=selected,
            routing_weights=weights[:1],
            shared_out=shared,
            shared_gate=gate,
            layer_out=residual,
        )
    with pytest.raises(ValueError, match="shared_gate"):
        compare_moe_combine(
            residual=residual,
            selected_down=selected,
            routing_weights=weights,
            shared_out=shared,
            shared_gate=np.ones((2,), dtype=np.float32),
            layer_out=residual,
        )


def _round_to_bf16(array: np.ndarray) -> np.ndarray:
    return bf16_to_float32(float_array_to_bf16_bits(array.astype(np.float32))).astype(np.float32)


def _sigmoid(value: float) -> np.float32:
    return np.float32(1.0) / (np.float32(1.0) + np.exp(np.float32(-value)))
