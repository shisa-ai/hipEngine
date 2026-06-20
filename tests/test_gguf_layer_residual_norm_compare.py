from __future__ import annotations

import numpy as np
import pytest

from hipengine.loading.materialize import float_array_to_bf16_bits
from hipengine.quant.gguf import bf16_to_float32
from scripts.gguf_layer_residual_norm_compare import compare_residual_norm


def test_compare_residual_norm_matches_bf16_residual_and_rmsnorm() -> None:
    hidden = np.asarray([0.25, -0.5, 1.0, -2.0], dtype=np.float32)
    attn = np.asarray([0.125, 0.25, -0.5, 0.75], dtype=np.float32)
    norm_weight = np.asarray([1.0, -0.5, 0.75, 1.25], dtype=np.float32)
    residual = _round_to_bf16(hidden + attn)
    inv = np.float32(1.0) / np.sqrt(np.mean(residual * residual) + np.float32(1.0e-6))
    post_norm = _round_to_bf16(residual * inv * norm_weight)

    comparison = compare_residual_norm(
        hidden_in=hidden,
        attn_out=attn,
        residual=residual,
        post_norm=post_norm,
        norm_weight=norm_weight,
        eps=1.0e-6,
    )

    assert comparison["residual_vs_cpu_bf16"]["max_abs_diff"] == 0.0
    assert comparison["post_norm_vs_cpu_bf16"]["max_abs_diff"] == 0.0
    assert comparison["post_norm_cpu_f32_vs_bf16"]["count"] == 4


def test_compare_residual_norm_rejects_shape_mismatches() -> None:
    vec = np.ones((4,), dtype=np.float32)
    with pytest.raises(ValueError, match="same shape"):
        compare_residual_norm(
            hidden_in=vec,
            attn_out=vec[:3],
            residual=vec,
            post_norm=vec,
            norm_weight=vec,
            eps=1.0e-6,
        )
    with pytest.raises(ValueError, match="norm_weight"):
        compare_residual_norm(
            hidden_in=vec,
            attn_out=vec,
            residual=vec,
            post_norm=vec,
            norm_weight=vec[:3],
            eps=1.0e-6,
        )


def _round_to_bf16(array: np.ndarray) -> np.ndarray:
    return bf16_to_float32(float_array_to_bf16_bits(array.astype(np.float32))).astype(np.float32)
