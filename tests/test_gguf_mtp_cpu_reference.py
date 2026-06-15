from __future__ import annotations

import numpy as np
import pytest

from hipengine.kernels.cpu_reference import qwen35_gguf_mtp_eh_proj
from hipengine.kernels.registry import resolve


def _rmsnorm(x: np.ndarray, weight: np.ndarray, eps: float = 1.0e-6) -> np.ndarray:
    variance = np.mean(x * x, axis=-1, keepdims=True)
    return (x * np.reciprocal(np.sqrt(variance + eps))) * weight


def test_qwen35_gguf_mtp_eh_proj_normalizes_and_concatenates_embedding_then_hidden() -> None:
    hidden = np.asarray([[3.0, 4.0]], dtype=np.float32)
    embedding = np.asarray([[1.0, 2.0]], dtype=np.float32)
    hnorm = np.asarray([10.0, 20.0], dtype=np.float32)
    enorm = np.asarray([30.0, 40.0], dtype=np.float32)
    # Select [e_norm[0], h_norm[1]] to pin llama.cpp concat order:
    # concat = [e_norm, h_norm], not [h_norm, e_norm].
    weight = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    out = qwen35_gguf_mtp_eh_proj(hidden, embedding, weight, hnorm, enorm)

    h_norm = _rmsnorm(hidden, hnorm)
    e_norm = _rmsnorm(embedding, enorm)
    expected = np.asarray([[e_norm[0, 0], h_norm[0, 1]]], dtype=np.float32)
    np.testing.assert_allclose(out, expected, rtol=1.0e-6, atol=1.0e-6)


def test_qwen35_gguf_mtp_eh_proj_supports_multiple_rows() -> None:
    hidden = np.asarray([[1.0, 2.0], [5.0, 6.0]], dtype=np.float32)
    embedding = np.asarray([[3.0, 4.0], [7.0, 8.0]], dtype=np.float32)
    hnorm = np.ones((2,), dtype=np.float32)
    enorm = np.asarray([2.0, 3.0], dtype=np.float32)
    weight = np.asarray(
        [
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    out = qwen35_gguf_mtp_eh_proj(hidden, embedding, weight, hnorm, enorm)

    fused = np.concatenate([_rmsnorm(embedding, enorm), _rmsnorm(hidden, hnorm)], axis=-1)
    expected = np.matmul(fused, weight.T).astype(np.float32)
    np.testing.assert_allclose(out, expected, rtol=1.0e-6, atol=1.0e-6)


def test_qwen35_gguf_mtp_eh_proj_validates_shapes() -> None:
    hidden = np.asarray([[1.0, 2.0]], dtype=np.float32)
    embedding = np.asarray([[3.0, 4.0]], dtype=np.float32)
    weight = np.zeros((2, 4), dtype=np.float32)
    norm = np.ones((2,), dtype=np.float32)

    with pytest.raises(ValueError, match="hidden_seed must have shape"):
        qwen35_gguf_mtp_eh_proj(hidden[0], embedding, weight, norm, norm)
    with pytest.raises(ValueError, match="token_embedding must match"):
        qwen35_gguf_mtp_eh_proj(hidden, np.zeros((2, 2), dtype=np.float32), weight, norm, norm)
    with pytest.raises(ValueError, match="eh_proj_weight must have shape"):
        qwen35_gguf_mtp_eh_proj(hidden, embedding, np.zeros((2, 3), dtype=np.float32), norm, norm)
    with pytest.raises(ValueError, match="hnorm_weight must have shape"):
        qwen35_gguf_mtp_eh_proj(hidden, embedding, weight, np.ones((3,), dtype=np.float32), norm)
    with pytest.raises(ValueError, match="enorm_weight must have shape"):
        qwen35_gguf_mtp_eh_proj(hidden, embedding, weight, norm, np.ones((3,), dtype=np.float32))


def test_qwen35_gguf_mtp_eh_proj_is_registered() -> None:
    fn = resolve(
        backend="cpu_reference",
        layer="mtp_nextn_eh_proj",
        quant="gguf_f32",
        variant="qwen35",
    )

    assert fn is qwen35_gguf_mtp_eh_proj
