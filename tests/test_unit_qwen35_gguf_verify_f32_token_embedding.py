from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import hipengine.runtime.qwen35_gguf_runner as runner_mod
from hipengine.quant.gguf import GGMLQuantizationType, dequantize_gguf_data
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


@pytest.mark.parametrize("tokens", [[], [3, 1, 3], [0, 4]])
def test_quantized_embedding_rows_keep_f32_precision_and_requested_order(tokens):
    from tests._gguf_synthetic_weights import make_q6_k_weight

    raw = make_q6_k_weight(5, 256)
    reader = _FakeReader(raw, ggml_type=GGMLQuantizationType.Q6_K)
    tensor_info = reader.tensor_info
    reader.tensor_info = lambda name: SimpleNamespace(
        **{**vars(tensor_info(name)), "shape": (5, 256)},
    )
    expected = np.stack([dequantize_gguf_data(raw[token], GGMLQuantizationType.Q6_K) for token in tokens]) if tokens else np.empty((0, 256), dtype=np.float32)
    result = _gguf_token_embedding_rows_f32(reader, tokens, hidden_size=256, vocab_size=5)
    np.testing.assert_array_equal(result, expected)
    assert result.dtype == np.float32
    assert result.flags.c_contiguous
    if tokens:
        # Values below the BF16 boundary must survive the diagnostic gather.
        rounded = ((result.view(np.uint32) + 0x7FFF + ((result.view(np.uint32) >> 16) & 1)) >> 16).astype(np.uint16)
        bf16_f32 = (rounded.astype(np.uint32) << 16).view(np.float32)
        assert np.any(result != bf16_f32)


@pytest.mark.parametrize("tokens", [[-1], [5], [0, -2]])
def test_embedding_rows_reject_all_out_of_range_tokens(tokens):
    with pytest.raises(ValueError, match="outside"):
        _gguf_token_embedding_rows_f32(_FakeReader(np.zeros((5, 4), dtype=np.float32)), tokens, hidden_size=4, vocab_size=5)


def test_embedding_rows_reject_dequantized_width_mismatch(monkeypatch):
    monkeypatch.setattr(runner_mod, "dequantize_gguf_data", lambda *args: np.zeros(3, dtype=np.float32))
    with pytest.raises(ValueError, match="dequantized to 3 values, expected 4"):
        _gguf_token_embedding_rows_f32(_FakeReader(np.zeros((5, 4), dtype=np.float32)), [1], hidden_size=4, vocab_size=5)


def test_session_embedding_seed_reuses_reader_and_uploads_exact_rows(monkeypatch):
    session = object.__new__(runner_mod.Qwen35GGUFResidentSession)
    session.runner = SimpleNamespace(hidden_size=4, vocab_size=5)
    session.model_path = "synthetic.gguf"
    session._verify_token_embedding_reader = None
    session._verify_hidden_f32_a = SimpleNamespace(ptr=0x1000)
    data = np.arange(20, dtype=np.float32).reshape(5, 4) / np.float32(13)
    reader = _FakeReader(data)
    opened = []
    uploaded = []
    runtime = object()
    monkeypatch.setattr(runner_mod, "GGUFReader", lambda path: opened.append(path) or reader)
    monkeypatch.setattr(runner_mod, "copy_host_array_to_device", lambda buf, rows, nbytes, **kwargs: uploaded.append((buf, rows.copy(), nbytes, kwargs)))
    for tokens in ([3, 1, 3], [4]):
        session._seed_verify_hidden_f32_from_token_embedding(np.asarray(tokens, dtype=np.int64), runtime=runtime)
    assert opened == ["synthetic.gguf"]
    assert session._verify_token_embedding_reader is reader
    for (buf, actual, nbytes, kwargs), tokens in zip(uploaded, ([3, 1, 3], [4]), strict=True):
        assert buf is session._verify_hidden_f32_a
        np.testing.assert_array_equal(actual, data[tokens])
        assert actual.dtype == np.float32 and actual.flags.c_contiguous
        assert nbytes == actual.nbytes
        assert kwargs == {"runtime": runtime}


@pytest.mark.parametrize("closed", ["session", "residual-buffer"])
def test_session_embedding_seed_refuses_closed_owners_before_readback(monkeypatch, closed):
    session = object.__new__(runner_mod.Qwen35GGUFResidentSession)
    session.runner = None if closed == "session" else SimpleNamespace(hidden_size=4, vocab_size=5)
    session._verify_hidden_f32_a = None
    session._verify_token_embedding_reader_cached = lambda: pytest.fail("closed owner opened reader")
    with pytest.raises(RuntimeError, match="closed"):
        session._seed_verify_hidden_f32_from_token_embedding(np.asarray([1]), runtime=object())


def test_gguf_verify_f32_token_embedding_env_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HIPENGINE_GGUF_VERIFY_F32_TOKEN_EMBEDDING", raising=False)
    assert _gguf_verify_f32_token_embedding_enabled() is False

    monkeypatch.setenv("HIPENGINE_GGUF_VERIFY_F32_TOKEN_EMBEDDING", "1")
    assert _gguf_verify_f32_token_embedding_enabled() is True
