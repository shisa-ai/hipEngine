"""Cross-history contracts that must survive the Gemma merge."""

from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.generation.gemma4_mtp import Gemma4MTPTextProvider
from hipengine.generation.registry import GenerationRequest
from hipengine.kernels.hip_gfx1100.gemma4.gemma4_layer import _layer_backend
from hipengine.runtime.gemma4 import Gemma4Runner


def test_fused_layer_backend_does_not_require_the_removed_split_gate() -> None:
    layer = SimpleNamespace(mlp_gate_up_proj=SimpleNamespace(backend="hip_gfx1100"))
    assert _layer_backend(layer) == "hip_gfx1100"


def test_mtp_refuses_int8_storage_instead_of_silently_using_bf16() -> None:
    provider = Gemma4MTPTextProvider(
        target_generator=SimpleNamespace(),
        config=SimpleNamespace(candidate_budget=1),
    )
    provider._generate_one = lambda *args: pytest.fail("unsupported KV reached generation")
    request = GenerationRequest(
        prompts=("hello",), max_tokens=1, temperature=0.0, top_p=1.0,
        ignore_eos=False, kv_storage="int8_per_token_head"
    )
    with pytest.raises(NotImplementedError, match="BF16.*shared.*KV"):
        provider.generate_detailed(request)


def test_bf16_shared_view_refuses_an_int8_cache() -> None:
    runner = object.__new__(Gemma4Runner)
    runner._kv_storage_resolved = "int8_per_token_head"
    runner._int8_kv = object()
    with pytest.raises(NotImplementedError, match="requires BF16 KV storage"):
        runner.shared_kv(0)


def test_trailing_logit_rows_can_span_the_original_chunk_boundary(monkeypatch) -> None:
    from hipengine.runtime import gemma4 as module

    runner = object.__new__(Gemma4Runner)
    runner.weights = SimpleNamespace(config=SimpleNamespace(vocab_size=100))
    runner.capacity, runner.max_block, runner.max_logits_rows = 32, 8, 4
    runner._position, runner._closed = 0, False
    calls = []

    def block(tokens, **kwargs):
        calls.append((tuple(tokens), kwargs))
        assert kwargs["logits_rows"] <= len(tokens)
        runner._position += len(tokens)
        return np.zeros((kwargs["logits_rows"], 100))

    runner._forward_block = block
    monkeypatch.setattr(module, "_gemma4_block_wmma_session", lambda _: nullcontext())
    result = runner.forward(range(10), logits_rows=4)
    assert result.shape == (4, 100)
    assert tuple(t for tokens, _ in calls for t in tokens) == tuple(range(10))
    assert all(len(tokens) <= 8 for tokens, _ in calls)
    assert calls[-1][1]["needs_logits"]
    assert not any(kwargs["needs_logits"] for _, kwargs in calls[:-1])


def test_hidden_capture_preserves_requested_multirow_logits(monkeypatch) -> None:
    from hipengine.runtime import gemma4 as module

    runner = object.__new__(Gemma4Runner)
    runner.weights = SimpleNamespace(
        config=SimpleNamespace(hidden_size=4, vocab_size=100, embed_scale=1.0, rms_norm_eps=1e-6),
        layers=[], embed_tokens=object(), lm_head=object(),
        final_norm=SimpleNamespace(buffer=SimpleNamespace(ptr=4000)),
    )
    runner._token_ids = SimpleNamespace(ptr=3000)
    runner._hidden = SimpleNamespace(ptr=1000)
    runner._normalized = SimpleNamespace(ptr=2000)
    runner._logits = SimpleNamespace(ptr=5000)
    runner._int8_kv = None
    heads = []
    monkeypatch.setattr(module, "launch_gguf_embedding", lambda *a, **k: None)
    monkeypatch.setattr(module, "gemma4_scale_bf16", lambda *a, **k: None)
    monkeypatch.setattr(module, "gemma4_rmsnorm_f32w_bf16", lambda *a, **k: None)
    monkeypatch.setattr(module, "launch_gguf_linear", lambda *a, **k: heads.append(a))
    runner._launch_block(range(4), tables={}, masks={}, kv_write_offset=0,
                         key_begin_at=None, return_hidden=True, logits_rows=2)
    assert runner.normalized_hidden_rows == 4
    assert runner._last_logits_rows == 2
    assert heads[0][1] == 2016
    assert heads[0][3] == 2
