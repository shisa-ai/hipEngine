from __future__ import annotations

import numpy as np
import pytest

from hipengine.kernels.cpu_reference import qwen35_gguf_mtp_eh_proj
from hipengine.kernels.registry import resolve


def test_qwen35_gguf_mtp_eh_proj_concatenates_hidden_then_embedding() -> None:
    hidden = np.asarray([[1.0, 2.0]], dtype=np.float32)
    embedding = np.asarray([[3.0, 4.0]], dtype=np.float32)
    weight = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    out = qwen35_gguf_mtp_eh_proj(hidden, embedding, weight)

    np.testing.assert_array_equal(out, np.asarray([[1.0, 3.0]], dtype=np.float32))


def test_qwen35_gguf_mtp_eh_proj_supports_multiple_rows() -> None:
    hidden = np.asarray([[1.0, 2.0], [5.0, 6.0]], dtype=np.float32)
    embedding = np.asarray([[3.0, 4.0], [7.0, 8.0]], dtype=np.float32)
    weight = np.asarray(
        [
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    out = qwen35_gguf_mtp_eh_proj(hidden, embedding, weight)

    expected = np.asarray([[4.0, 6.0], [12.0, 14.0]], dtype=np.float32)
    np.testing.assert_array_equal(out, expected)


def test_qwen35_gguf_mtp_eh_proj_validates_shapes() -> None:
    hidden = np.asarray([[1.0, 2.0]], dtype=np.float32)
    embedding = np.asarray([[3.0, 4.0]], dtype=np.float32)
    weight = np.zeros((2, 4), dtype=np.float32)

    with pytest.raises(ValueError, match="hidden_seed must have shape"):
        qwen35_gguf_mtp_eh_proj(hidden[0], embedding, weight)
    with pytest.raises(ValueError, match="token_embedding must match"):
        qwen35_gguf_mtp_eh_proj(hidden, np.zeros((2, 2), dtype=np.float32), weight)
    with pytest.raises(ValueError, match="eh_proj_weight must have shape"):
        qwen35_gguf_mtp_eh_proj(hidden, embedding, np.zeros((2, 3), dtype=np.float32))


def test_qwen35_gguf_mtp_eh_proj_is_registered() -> None:
    fn = resolve(
        backend="cpu_reference",
        layer="mtp_nextn_eh_proj",
        quant="gguf_f32",
        variant="qwen35",
    )

    assert fn is qwen35_gguf_mtp_eh_proj
