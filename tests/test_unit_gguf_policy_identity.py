"""Admission policy identity must not depend on a measured-manifest list."""
from types import SimpleNamespace

import pytest

from hipengine.loading import gguf_mtp_hot_vocab as hot_vocab
from hipengine.loading import qwen35_gguf_admission as admission


@pytest.mark.parametrize("fingerprint", ["new-layout", "0" * 64, "f" * 64])
def test_unlisted_report_keeps_generic_policy_identity(fingerprint):
    report = SimpleNamespace(preset=None, manifest_fingerprint=fingerprint)
    assert admission.qwen35_gguf_artifact_preset_key_for_report(report) is None


def test_unlisted_map_keeps_generic_policy_identity(monkeypatch):
    monkeypatch.setattr(admission, "resolve_qwen35_gguf_artifact_preset", lambda *a, **k: None)
    monkeypatch.setattr(admission, "build_qwen35_gguf_role_manifest",
                        lambda *a, **k: SimpleNamespace(fingerprint="unmeasured-layout"))
    assert admission.qwen35_gguf_artifact_preset_key(object()) is None


def test_specialized_preset_identity_is_preserved():
    preset = SimpleNamespace(preset_key="gguf_ud_q4_k_m")
    report = SimpleNamespace(preset=preset, manifest_fingerprint="layout")
    assert admission.qwen35_gguf_artifact_preset_key_for_report(report) == preset.preset_key


@pytest.mark.parametrize("preset", [None, "gguf_ud_q4_k_m", "unfamiliar-preset"])
def test_hot_vocab_default_depends_only_on_tokenizer(monkeypatch, preset):
    monkeypatch.setattr(hot_vocab, "gguf_tokenizer_tokens_sha256", lambda info: "compatible-tokenizer")
    monkeypatch.setattr(hot_vocab, "_DEFAULT_HOT_VOCAB_IDENTITIES",
                        {"compatible-tokenizer": "qwen38-27b-hot131072-cjk-v1.json"})
    info = SimpleNamespace(file_type_name="NEW_QUANT", metadata={
        "general.architecture": "compatible-plugin", "general.basename": "new-name",
        "qwen35.block_count": 1,
    })
    assert hot_vocab.default_gguf_hot_vocab_path(info, artifact_preset_key=preset) is not None


def test_hot_vocab_accepts_compatible_tokenizer_with_different_model_metadata(tmp_path):
    import json

    info = SimpleNamespace(file_type_name="NEW_QUANT", metadata={
        "tokenizer.ggml.tokens": [f"t{i}" for i in range(32)],
        "general.architecture": "new-plugin", "general.basename": "new-model",
        "qwen35.block_count": 2,
    })
    payload = {"schema_version": 1, "kind": hot_vocab.HOT_VOCAB_KIND,
               "model": {"vocab_size": 32,
                         "tokenizer_tokens_sha256": hot_vocab.gguf_tokenizer_tokens_sha256(info),
                         "architecture": "old-plugin", "basename": "old-model",
                         "block_count": 1, "file_type": "OLD_QUANT"},
               "token_ids": list(range(16)), "selection": {}}
    path = tmp_path / "hot.json"
    path.write_text(json.dumps(payload))
    assert hot_vocab.load_gguf_hot_vocab_selection(path, info).token_ids == tuple(range(16))
