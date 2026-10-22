"""The NextN hot-vocabulary head yields to context on a memory-bound GPU.

The head is a decode-speed cache (about 0.5 GiB on Qwen3.8-27B). When the
automatic context cannot reach its target, the server frees the head and the
draft scores the full output head instead; an explicit env choice always wins.
"""

import threading
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from hipengine.generation import qwen35_gguf


def _generator(monkeypatch, *, profile="production"):
    generator = qwen35_gguf.Qwen35GGUFBringupGenerator.__new__(
        qwen35_gguf.Qwen35GGUFBringupGenerator
    )
    generator.execution_profile = profile
    generator._mtp_hot_vocab_choice = None
    generator._shared_mtp_nextn_weights = {}
    generator._shared_mtp_draft_pool = {}
    generator._shared_mtp_draft_pool_lock = threading.Lock()
    generator._shared_session_pool = {}
    generator._shared_session_pool_lock = threading.Lock()
    monkeypatch.delenv("HIPENGINE_GGUF_MTP_HOT_VOCAB", raising=False)
    return generator


@dataclass(frozen=True)
class _Weights:
    hot_vocab: object | None


def _hot_vocab(freed):
    # Same shape as ``DeviceTensorAllocation``: the size lives on its buffer.
    tiles = SimpleNamespace(buffer=SimpleNamespace(nbytes=512 * 1024**2), tensor=SimpleNamespace())
    return SimpleNamespace(
        lm_head=SimpleNamespace(allocation=lambda name: tiles),
        free=lambda runtime=None: freed.append(runtime),
    )


def test_the_packaged_default_is_decided_by_sizing(monkeypatch) -> None:
    generator = _generator(monkeypatch)
    assert generator._mtp_hot_vocab_path() == "auto"
    assert generator._mtp_hot_vocab_choice_pending()


@pytest.mark.parametrize("env", ["auto", "0", "4096"])
def test_an_explicit_env_choice_is_never_overridden(monkeypatch, env) -> None:
    generator = _generator(monkeypatch)
    monkeypatch.setenv("HIPENGINE_GGUF_MTP_HOT_VOCAB", env)
    assert not generator._mtp_hot_vocab_choice_pending()


def test_strict_profile_has_no_head_to_decide(monkeypatch) -> None:
    generator = _generator(monkeypatch, profile="strict")
    assert generator._mtp_hot_vocab_path() is None
    assert not generator._mtp_hot_vocab_choice_pending()


def test_dropping_the_head_frees_it_and_keeps_the_draft_block(monkeypatch) -> None:
    generator = _generator(monkeypatch)
    runtime = object()
    freed = []
    generator._shared_mtp_nextn_weights[(id(runtime), "auto")] = _Weights(_hot_vocab(freed))
    assert generator._drop_mtp_hot_vocab(
        object(), runtime=runtime, context_tokens=53_760, target_tokens=262_144,
    )
    assert freed == [runtime]
    assert generator._mtp_hot_vocab_path() is None
    assert not generator._mtp_hot_vocab_choice_pending()
    remaining = generator._shared_mtp_nextn_weights
    assert (id(runtime), "auto") not in remaining
    assert remaining[(id(runtime), None)].hot_vocab is None


def test_a_head_already_lent_to_a_provider_is_kept(monkeypatch) -> None:
    generator = _generator(monkeypatch)
    runtime = object()
    freed = []
    generator._shared_mtp_nextn_weights[(id(runtime), "auto")] = _Weights(_hot_vocab(freed))
    generator._shared_mtp_draft_pool[(id(runtime), "dense_nextn", 4096, 1, "auto")] = []
    assert not generator._drop_mtp_hot_vocab(
        object(), runtime=runtime, context_tokens=1, target_tokens=2,
    )
    assert freed == []
