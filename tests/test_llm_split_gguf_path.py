from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import hipengine.llm as llm_module


def test_llm_preserves_split_gguf_directory_for_generator_factory(monkeypatch, tmp_path: Path) -> None:
    files = (tmp_path / "model-00001-of-00002.gguf", tmp_path / "model-00002-of-00002.gguf")
    for path in files:
        path.write_bytes(b"GGUF")
    info = SimpleNamespace(path=files[0], architecture="qwen4exp")
    plugin = SimpleNamespace(name="qwen4_exp_gguf")

    monkeypatch.setattr("hipengine.loading.resolve_model_path", lambda value: tmp_path)
    monkeypatch.setattr("hipengine.loading.discover_gguf_files", lambda value: files)
    monkeypatch.setattr("hipengine.loading.load_gguf_index", lambda value: info)
    monkeypatch.setattr("hipengine.models.resolve_model", lambda value: plugin)

    llm = object.__new__(llm_module.LLM)
    llm.model = str(tmp_path)
    llm._weight_index = None
    llm._model_plugin = None

    loaded_info, loaded_plugin = llm._load_model_metadata()

    assert loaded_info is info
    assert loaded_plugin is plugin
    assert llm.model == str(tmp_path)
