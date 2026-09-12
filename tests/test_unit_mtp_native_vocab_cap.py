import pytest

from hipengine.speculative.mtp_native import _draft_vocab_from_env


def test_mtp_draft_vocab_default_is_retained_cap(monkeypatch):
    monkeypatch.delenv("HIPENGINE_MTP_DRAFT_VOCAB_CAP", raising=False)

    assert _draft_vocab_from_env(151936) == 65536


def test_mtp_draft_vocab_default_clamps_to_vocab(monkeypatch):
    monkeypatch.delenv("HIPENGINE_MTP_DRAFT_VOCAB_CAP", raising=False)

    assert _draft_vocab_from_env(32768) == 32768


def test_mtp_draft_vocab_explicit_zero_uses_full_vocab(monkeypatch):
    monkeypatch.setenv("HIPENGINE_MTP_DRAFT_VOCAB_CAP", "0")

    assert _draft_vocab_from_env(151936) == 151936


def test_mtp_draft_vocab_explicit_cap(monkeypatch):
    monkeypatch.setenv("HIPENGINE_MTP_DRAFT_VOCAB_CAP", "32768")

    assert _draft_vocab_from_env(151936) == 32768


def test_mtp_draft_vocab_requires_positive_vocab(monkeypatch):
    monkeypatch.delenv("HIPENGINE_MTP_DRAFT_VOCAB_CAP", raising=False)

    with pytest.raises(ValueError, match="vocab must be positive"):
        _draft_vocab_from_env(0)
