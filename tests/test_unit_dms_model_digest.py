"""Model verification reuse must follow file identity, not its pathname alone."""

import os

import pytest

from hipengine.kvcache import dms


def test_model_digest_reuses_unchanged_file(tmp_path, monkeypatch):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"weights")
    original = dms._sha256_file
    calls = []

    def counted(path):
        calls.append(path)
        return original(path)

    monkeypatch.setattr(dms, "_sha256_file", counted)
    first = dms._model_sha256_file(model)
    assert dms._model_sha256_file(model) == first
    alias = tmp_path / "alias.gguf"
    alias.symlink_to(model)
    assert dms._model_sha256_file(alias) == first
    assert len(calls) == 1


def test_model_digest_invalidates_same_size_rewrite_and_replacement(tmp_path):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"before")
    first = dms._model_sha256_file(model)
    old_stat = model.stat()
    model.write_bytes(b"after!")
    os.utime(model, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns))
    second = dms._model_sha256_file(model)
    assert second != first
    replacement = tmp_path / "replacement"
    replacement.write_bytes(b"third!")
    os.utime(replacement, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns))
    replacement.replace(model)
    assert dms._model_sha256_file(model) not in {first, second}
    model.unlink()
    with pytest.raises(FileNotFoundError):
        dms._model_sha256_file(model)


def test_model_digest_rejects_mutation_during_hash_without_caching(tmp_path, monkeypatch):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"before")
    original = dms._sha256_file

    def mutate(path):
        result = original(path)
        path.write_bytes(b"after!")
        return result

    monkeypatch.setattr(dms, "_sha256_file", mutate)
    with pytest.raises(ValueError, match="changed during"):
        dms._model_sha256_file(model)
    monkeypatch.setattr(dms, "_sha256_file", original)
    assert dms._model_sha256_file(model) == original(model)
