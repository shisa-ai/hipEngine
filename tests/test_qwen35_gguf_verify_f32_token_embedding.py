from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.quant.gguf import GGMLQuantizationType
from hipengine.runtime.qwen35_gguf_runner import (
    _gguf_token_embedding_rows_f32,
    _gguf_verify_f32_token_embedding_enabled,
)


class _FakeReader:
    def __init__(self, data: np.ndarray, *, ggml_type: GGMLQuantizationType = GGMLQuantizationType.F32):
        self.data = data
        self.ggml_type = ggml_type

    def tensor_info(self, name: str):
        assert name == "token_embd.weight"
        return SimpleNamespace(
            name=name,
            shape=tuple(self.data.shape),
            ggml_type=int(self.ggml_type),
        )

    def tensor_data(self, name: str):
        assert name == "token_embd.weight"
        return self.data


def test_gguf_token_embedding_rows_f32_reads_requested_rows() -> None:
    data = np.arange(20, dtype=np.float32).reshape(5, 4)

    rows = _gguf_token_embedding_rows_f32(
        _FakeReader(data),
        np.asarray([2, 4], dtype=np.int64),
        hidden_size=4,
        vocab_size=5,
    )

    assert rows.dtype == np.float32
    assert rows.flags.c_contiguous
    np.testing.assert_array_equal(rows, data[[2, 4]])


def test_gguf_token_embedding_rows_f32_validates_token_bounds() -> None:
    data = np.zeros((5, 4), dtype=np.float32)

    with pytest.raises(ValueError, match=r"token_id 5 outside \[0, 5\)"):
        _gguf_token_embedding_rows_f32(
            _FakeReader(data),
            [5],
            hidden_size=4,
            vocab_size=5,
        )


def test_gguf_token_embedding_rows_f32_validates_tensor_shape() -> None:
    data = np.zeros((6, 4), dtype=np.float32)

    with pytest.raises(ValueError, match=r"shape \(6, 4\) does not match expected \(5, 4\)"):
        _gguf_token_embedding_rows_f32(
            _FakeReader(data),
            [1],
            hidden_size=4,
            vocab_size=5,
        )


def test_gguf_verify_f32_token_embedding_env_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HIPENGINE_GGUF_VERIFY_F32_TOKEN_EMBEDDING", raising=False)
    assert _gguf_verify_f32_token_embedding_enabled() is False

    monkeypatch.setenv("HIPENGINE_GGUF_VERIFY_F32_TOKEN_EMBEDDING", "1")
    assert _gguf_verify_f32_token_embedding_enabled() is True
