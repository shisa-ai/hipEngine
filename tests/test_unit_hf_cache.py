from __future__ import annotations

import json

import numpy as np
import pytest
from safetensors.numpy import save_file

from hipengine.loading import MissingConfigError, load_weight_index, resolve_model_path
from hipengine.loading.gguf import discover_gguf_files


def test_resolve_model_path_prefers_existing_filesystem_path(tmp_path) -> None:
    model_dir = tmp_path / "local-model"
    model_dir.mkdir()

    assert resolve_model_path(model_dir) == model_dir


def test_resolve_model_path_finds_hf_cache_snapshot(monkeypatch, tmp_path) -> None:
    hub = tmp_path / "hub"
    snapshot = hub / "models--org--model" / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    refs = hub / "models--org--model" / "refs"
    refs.mkdir()
    (refs / "main").write_text("abc123", encoding="utf-8")
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(hub))

    assert resolve_model_path("org/model") == snapshot


def test_load_weight_index_reports_missing_hf_model_id_cache(monkeypatch, tmp_path) -> None:
    hub = tmp_path / "hub"
    hub.mkdir()
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(hub))
    repo_id = "missing-org/missing-paro"

    with pytest.raises(MissingConfigError) as exc_info:
        load_weight_index(repo_id)

    message = str(exc_info.value)
    assert repo_id in message
    assert "local cache" in message
    assert "config.json not found under missing-org" not in message


def test_load_weight_index_accepts_hf_model_id_from_cache(monkeypatch, tmp_path) -> None:
    hub = tmp_path / "hub"
    snapshot = hub / "models--org--paro" / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    refs = hub / "models--org--paro" / "refs"
    refs.mkdir()
    (refs / "main").write_text("abc123", encoding="utf-8")
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(hub))

    (snapshot / "config.json").write_text(
        json.dumps({"architectures": ["Qwen3_5MoeForConditionalGeneration"]}),
        encoding="utf-8",
    )
    save_file({"x": np.zeros((1,), dtype=np.float32)}, snapshot / "model.safetensors")

    index = load_weight_index("org/paro")

    assert index.model_path == snapshot.resolve()
    assert index.tensors["x"].shape == (1,)


def test_discover_gguf_files_accepts_hf_model_id_from_cache(monkeypatch, tmp_path) -> None:
    hub = tmp_path / "hub"
    snapshot = hub / "models--org--gguf" / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    refs = hub / "models--org--gguf" / "refs"
    refs.mkdir()
    (refs / "main").write_text("abc123", encoding="utf-8")
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(hub))
    model_file = snapshot / "model.gguf"
    model_file.write_bytes(b"GGUF")

    assert discover_gguf_files("org/gguf") == (model_file.resolve(),)
