from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import pytest

import hipengine.loading.gguf as gguf_module
from hipengine.loading.gguf import GGUFFormatError, GGUFModelInfo, GGUFTensorInfo, scan_gguf_splits
from hipengine.loading.stepfun_gguf import (
    DENSE_MLP,
    FULL_ATTENTION,
    MOE,
    SLIDING_ATTENTION,
    StepFunUnsupportedFeatureError,
    build_stepfun_gguf_tensor_map,
    required_stepfun_gguf_tensor_names,
    stepfun_gguf_config_from_metadata,
    validate_stepfun_gguf_tensor_map,
    validate_stepfun_multimodal_projector_assets,
)


DEFAULT_STEPFUN_GGUF_DIR = Path("/data/models/gguf")


def _stepfun_gguf_dir() -> Path:
    return Path(os.environ.get("HIPENGINE_STEPFUN_GGUF_DIR", DEFAULT_STEPFUN_GGUF_DIR))


def _stepfun_gguf_paths() -> tuple[Path, ...]:
    root = _stepfun_gguf_dir()
    paths = tuple(sorted(root.glob("Step-3.7-flash-Q3_K_L-*.gguf")))
    if len(paths) != 3:
        pytest.skip(
            "StepFun GGUF Q3_K_L shards not found; set HIPENGINE_STEPFUN_GGUF_DIR "
            "to a directory containing Step-3.7-flash-Q3_K_L-00001..00003.gguf"
        )
    return paths


def test_scan_gguf_splits_merges_stepfun_shards_and_retains_sources() -> None:
    paths = _stepfun_gguf_paths()
    info = scan_gguf_splits(paths)

    assert info.paths == tuple(path.resolve() for path in paths)
    assert info.split_count == 3
    assert info.tensor_count == 754
    assert info.architecture == "step35"
    assert info.file_type_name == "MOSTLY_Q3_K_L"
    assert [shard.metadata["split.no"] for shard in info.shards] == [0, 1, 2]

    output = info.tensor("output.weight")
    final_expert = info.tensor("blk.44.ffn_gate_exps.weight")
    assert output.source_path == paths[0].resolve()
    assert output.split_no == 0
    assert final_expert.source_path == paths[2].resolve()
    assert final_expert.split_no == 2
    assert final_expert.shape == (288, 1280, 4096)
    assert final_expert.ggml_type_name == "Q3_K"


def test_stepfun_tensor_map_validates_all_required_text_tensors() -> None:
    info = scan_gguf_splits(_stepfun_gguf_paths())
    config = stepfun_gguf_config_from_metadata(info)
    validation = validate_stepfun_gguf_tensor_map(info)
    model_map = build_stepfun_gguf_tensor_map(info)

    assert config.block_count == 45
    assert config.layer_attention_types[:5] == (
        FULL_ATTENTION,
        SLIDING_ATTENTION,
        SLIDING_ATTENTION,
        SLIDING_ATTENTION,
        FULL_ATTENTION,
    )
    assert config.layer_mlp_types[:5] == (DENSE_MLP, DENSE_MLP, DENSE_MLP, MOE, MOE)
    assert validation.passed
    assert validation.missing == ()
    assert validation.shape_errors == ()
    assert validation.type_errors == ()
    assert len(required_stepfun_gguf_tensor_names(config)) == 754
    assert len(model_map.tensor_names) == 754

    assert model_map.root("token_embedding").shape == (128_896, 4096)
    assert model_map.layer(0).attention_type == FULL_ATTENTION
    assert model_map.layer(0).mlp_type == DENSE_MLP
    assert model_map.layer(0).tensor("ffn_gate").shape == (11_264, 4096)
    assert model_map.layer(1).attention_type == SLIDING_ATTENTION
    assert model_map.layer(3).mlp_type == MOE
    assert model_map.layer(3).tensor("exp_probs_bias").shape == (288,)
    assert model_map.layer(44).attention_type == FULL_ATTENTION
    assert model_map.layer(44).tensor("ffn_down_shexp").shape == (4096, 1280)


def test_stepfun_tensor_map_reports_missing_and_type_errors() -> None:
    info = scan_gguf_splits(_stepfun_gguf_paths())
    missing_info = replace(
        info,
        tensors=tuple(tensor for tensor in info.tensors if tensor.name != "blk.3.exp_probs_b.bias"),
    )
    missing_validation = validate_stepfun_gguf_tensor_map(missing_info)
    assert not missing_validation.passed
    assert missing_validation.missing == ("blk.3.exp_probs_b.bias",)
    with pytest.raises(KeyError, match="missing tensors: blk.3.exp_probs_b.bias"):
        build_stepfun_gguf_tensor_map(missing_info)

    wrong_type_info = replace(
        info,
        tensors=tuple(
            replace(tensor, tensor=replace(tensor.tensor, ggml_type_name="Q8_0"))
            if tensor.name == "blk.0.attn_q.weight"
            else tensor
            for tensor in info.tensors
        ),
    )
    wrong_type_validation = validate_stepfun_gguf_tensor_map(wrong_type_info)
    assert not wrong_type_validation.passed
    assert wrong_type_validation.type_errors == (
        "blk.0.attn_q.weight: expected type Q3_K, got Q8_0",
    )


def test_stepfun_missing_multimodal_projector_has_clear_error() -> None:
    info = scan_gguf_splits(_stepfun_gguf_paths())

    with pytest.raises(StepFunUnsupportedFeatureError, match="text-only mode"):
        validate_stepfun_multimodal_projector_assets(info)


def test_scan_gguf_splits_rejects_duplicate_tensor_names(monkeypatch: pytest.MonkeyPatch) -> None:
    tensor = GGUFTensorInfo(
        name="duplicate.weight",
        shape=(1,),
        ggml_shape=(1,),
        ggml_type=0,
        ggml_type_name="F32",
        n_elements=1,
        nbytes=4,
        offset=0,
        data_offset=0,
        byte_shape=(1,),
    )
    shards = {
        "a.gguf": GGUFModelInfo(
            path=Path("a.gguf"),
            version=3,
            alignment=32,
            metadata={"split.no": 0, "split.count": 2, "split.tensors.count": 2},
            tensors=(tensor,),
            tensor_data_offset=0,
        ),
        "b.gguf": GGUFModelInfo(
            path=Path("b.gguf"),
            version=3,
            alignment=32,
            metadata={"split.no": 1, "split.count": 2, "split.tensors.count": 2},
            tensors=(tensor,),
            tensor_data_offset=0,
        ),
    }

    def fake_scan_gguf(path: str | Path) -> GGUFModelInfo:
        return shards[Path(path).name]

    monkeypatch.setattr(gguf_module, "scan_gguf", fake_scan_gguf)
    with pytest.raises(GGUFFormatError, match="duplicate GGUF tensor 'duplicate.weight'"):
        scan_gguf_splits([Path("a.gguf"), Path("b.gguf")])
