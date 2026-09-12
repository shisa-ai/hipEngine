from __future__ import annotations

import json
import sys

import numpy as np
import pytest
from safetensors.numpy import save_file

from hipengine.loading import (
    MissingConfigError,
    MissingTensorError,
    TensorInfo,
    discover_safetensor_shards,
    load_weight_index,
    read_config,
)


def test_read_config_and_discover_indexed_safetensor_shards(tmp_path) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps({"architectures": ["Qwen3_5MoeForConditionalGeneration"]}),
        encoding="utf-8",
    )
    save_file({"model.embed_tokens.weight": np.zeros((2, 3), dtype=np.float32)}, tmp_path / "a.safetensors")
    save_file({"model.norm.weight": np.zeros((3,), dtype=np.float16)}, tmp_path / "b.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 18},
                "weight_map": {
                    "model.embed_tokens.weight": "a.safetensors",
                    "model.norm.weight": "b.safetensors",
                },
            }
        ),
        encoding="utf-8",
    )

    config = read_config(tmp_path)
    shards = discover_safetensor_shards(tmp_path)
    index = load_weight_index(tmp_path)

    assert config["architectures"] == ["Qwen3_5MoeForConditionalGeneration"]
    assert tuple(path.name for path in shards) == ("a.safetensors", "b.safetensors")
    assert index.model_path == tmp_path.resolve()
    assert index.shards == shards
    assert index.tensors["model.embed_tokens.weight"] == TensorInfo(
        name="model.embed_tokens.weight",
        shard_path=tmp_path.joinpath("a.safetensors").resolve(),
        dtype="F32",
        shape=(2, 3),
    )
    assert index.tensors["model.embed_tokens.weight"].nbytes == 24
    assert index.tensors["model.norm.weight"].dtype == "F16"
    assert index.names_with_prefix("model.") == (
        "model.embed_tokens.weight",
        "model.norm.weight",
    )
    assert index.require(["model.norm.weight"])[0].shape == (3,)


def test_discover_single_safetensors_file(tmp_path) -> None:
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    shard = tmp_path / "model.safetensors"
    save_file({"x": np.zeros((1,), dtype=np.int8)}, shard)

    index = load_weight_index(shard)

    assert index.shards == (shard,)
    assert index.tensors["x"].dtype == "I8"
    assert index.tensors["x"].nbytes == 1


def test_missing_config_and_required_tensors_are_clean_errors(tmp_path) -> None:
    with pytest.raises(MissingConfigError):
        read_config(tmp_path)

    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    save_file({"present": np.zeros((1,), dtype=np.float32)}, tmp_path / "model.safetensors")
    index = load_weight_index(tmp_path)

    with pytest.raises(MissingTensorError, match="missing required tensors: absent"):
        index.require(["absent"])


def test_loading_helpers_do_not_import_torch(tmp_path) -> None:
    had_torch = "torch" in sys.modules
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    save_file({"x": np.zeros((1,), dtype=np.float32)}, tmp_path / "model.safetensors")

    load_weight_index(tmp_path)

    if not had_torch:
        assert "torch" not in sys.modules
