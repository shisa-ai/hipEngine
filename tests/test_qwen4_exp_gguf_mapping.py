from __future__ import annotations

from math import prod
from pathlib import Path

import pytest

from hipengine.loading.gguf import GGUFModelInfo, GGUFTensorInfo
from hipengine.loading.qwen4_exp_gguf import (
    GDN,
    QSA,
    Qwen4ExpGGUFTensorMapError,
    build_qwen4_exp_gguf_tensor_map,
    required_qwen4_exp_gguf_tensor_names,
    validate_qwen4_exp_gguf_tensor_map,
)
from hipengine.quant.gguf import GGMLQuantizationType
from tests.test_qwen4_exp_gguf_config import _metadata


def _tensor(name: str, shape: tuple[int, ...], *, offset: int = 0) -> GGUFTensorInfo:
    nbytes = prod(shape) * 4
    return GGUFTensorInfo(
        name=name,
        shape=shape,
        ggml_shape=tuple(reversed(shape)),
        ggml_type=int(GGMLQuantizationType.F32),
        ggml_type_name="F32",
        n_elements=prod(shape),
        nbytes=nbytes,
        offset=offset,
        data_offset=offset,
        byte_shape=shape,
    )


def _layer_tensors(layer: int) -> list[GGUFTensorInfo]:
    prefix = f"blk.{layer}."
    residual = 10_240
    hidden = 2_560
    low_rank = 320
    tensors = [
        _tensor(prefix + "hc_attn_norm.weight", (residual,)),
        _tensor(prefix + "hc_attn_down.weight", (low_rank, residual)),
        _tensor(prefix + "hc_attn_up.weight", (residual, low_rank)),
        _tensor(prefix + "hc_attn_inject.weight", (4, residual)),
        _tensor(prefix + "hc_ffn_norm.weight", (residual,)),
        _tensor(prefix + "hc_ffn_down.weight", (low_rank, residual)),
        _tensor(prefix + "hc_ffn_up.weight", (residual, low_rank)),
        _tensor(prefix + "hc_ffn_inject.weight", (4, residual)),
        _tensor(prefix + "ffn_gate_inp.weight", (512, hidden)),
        _tensor(prefix + "ffn_gate_inp_shexp.weight", (hidden,)),
        _tensor(prefix + "ffn_gate_exps.weight", (512, 640, hidden)),
        _tensor(prefix + "ffn_up_exps.weight", (512, 640, hidden)),
        _tensor(prefix + "ffn_down_exps.weight", (512, hidden, 640)),
        _tensor(prefix + "ffn_gate_shexp.weight", (640, hidden)),
        _tensor(prefix + "ffn_up_shexp.weight", (640, hidden)),
        _tensor(prefix + "ffn_down_shexp.weight", (hidden, 640)),
    ]
    if layer % 4 == 3:
        tensors.extend(
            (
                _tensor(prefix + "attn_q.weight", (12_288, hidden)),
                _tensor(prefix + "attn_q_norm.weight", (256,)),
                _tensor(prefix + "attn_k.weight", (512, hidden)),
                _tensor(prefix + "attn_k_norm.weight", (256,)),
                _tensor(prefix + "attn_v.weight", (512, hidden)),
                _tensor(prefix + "attn_output.weight", (hidden, 6_144)),
                _tensor(prefix + "indexer.q_proj.weight", (512, hidden)),
                _tensor(prefix + "indexer.k_proj.weight", (128, hidden)),
                _tensor(prefix + "indexer.q_norm.weight", (128,)),
                _tensor(prefix + "indexer.k_norm.weight", (128,)),
            )
        )
    else:
        tensors.extend(
            (
                _tensor(prefix + "attn_qkv.weight", (10_240, hidden)),
                _tensor(prefix + "attn_gate.weight", (6_144, hidden)),
                _tensor(prefix + "ssm_a", (48,)),
                _tensor(prefix + "ssm_alpha.weight", (48, hidden)),
                _tensor(prefix + "ssm_beta.weight", (48, hidden)),
                _tensor(prefix + "ssm_conv1d.weight", (10_240, 4)),
                _tensor(prefix + "ssm_dt.bias", (48,)),
                _tensor(prefix + "ssm_norm.weight", (128,)),
                _tensor(prefix + "ssm_out.weight", (hidden, 6_144)),
            )
        )
    if layer == 1:
        tensors.extend(
            (
                _tensor(prefix + "ple_key.weight", (10_240, hidden)),
                _tensor(prefix + "ple_value.weight", (hidden, hidden)),
                _tensor(prefix + "ple_norm_key.weight", (10_240,)),
                _tensor(prefix + "ple_norm_query.weight", (10_240,)),
                _tensor(prefix + "ple_norm_conv.weight", (10_240,)),
                _tensor(prefix + "ple_conv1d.weight", (10_240, 4)),
            )
        )
    return tensors


def _infos(
    *,
    drop: set[str] | None = None,
    extra: list[GGUFTensorInfo] | None = None,
    replacements: dict[str, GGUFTensorInfo] | None = None,
    ple_rows: int = 320_001_446,
    duplicate: str | None = None,
) -> tuple[GGUFModelInfo, GGUFModelInfo]:
    tensors = [
        _tensor("token_embd.weight", (248_320, 2_560)),
        _tensor("output.weight", (248_320, 2_560)),
        _tensor("output_hc_norm.weight", (10_240,)),
        _tensor("output_hc_down.weight", (320, 10_240)),
        _tensor("output_hc_up.weight", (10_240, 320)),
        _tensor("per_layer_token_embd.weight", (ple_rows, 160)),
    ]
    for layer in range(48):
        tensors.extend(_layer_tensors(layer))
    if drop:
        tensors = [tensor for tensor in tensors if tensor.name not in drop]
    if replacements:
        tensors = [replacements.get(tensor.name, tensor) for tensor in tensors]
    if extra:
        tensors.extend(extra)
    duplicate_tensor = next((tensor for tensor in tensors if tensor.name == duplicate), None)
    declared = len(tensors) + int(duplicate_tensor is not None)
    head_metadata = {
        **_metadata(),
        "split.count": 2,
        "split.no": 0,
        "split.tensors.count": declared,
    }
    data_metadata = {
        "split.count": 2,
        "split.no": 1,
        "split.tensors.count": declared,
    }
    head = GGUFModelInfo(
        path=Path("qwen4exp-00001-of-00002.gguf"),
        version=3,
        alignment=32,
        metadata=head_metadata,
        tensors=(() if duplicate_tensor is None else (duplicate_tensor,)),
        tensor_data_offset=0,
    )
    data = GGUFModelInfo(
        path=Path("qwen4exp-00002-of-00002.gguf"),
        version=3,
        alignment=32,
        metadata=data_metadata,
        tensors=tuple(tensors),
        tensor_data_offset=0,
    )
    return head, data


def test_qwen4_exp_gguf_tensor_map_covers_all_frozen_roles_and_parts() -> None:
    model_map = build_qwen4_exp_gguf_tensor_map(_infos())

    assert model_map.validation.passed
    assert len(model_map.tensor_names) == 1_224
    assert len(required_qwen4_exp_gguf_tensor_names(model_map.config)) == 1_224
    assert model_map.root("token_embedding").tensor.name == "token_embd.weight"
    assert model_map.root("lm_head").tensor.name == "output.weight"
    assert model_map.ple_table.tensor.shape == (320_001_446, 160)
    assert model_map.ple_padding_rows == 0
    assert model_map.layer(0).layer_type == GDN
    assert model_map.layer(0).tensor("ssm_a").tensor.name == "blk.0.ssm_a"
    assert model_map.layer(1).tensor("ple_key").tensor.name == "blk.1.ple_key.weight"
    assert model_map.layer(3).layer_type == QSA
    assert model_map.layer(3).tensor("index_q").tensor.shape == (512, 2_560)
    assert all(ref.part_index == 1 for ref in model_map.tensor_refs)


def test_qwen4_exp_gguf_tensor_map_reports_missing_and_unexpected_names() -> None:
    infos = _infos(
        drop={"blk.3.indexer.q_proj.weight"},
        extra=[_tensor("blk.3.unexpected.weight", (1,))],
    )

    validation = validate_qwen4_exp_gguf_tensor_map(infos)

    assert validation.passed is False
    assert validation.missing_tensor_names == ("blk.3.indexer.q_proj.weight",)
    assert validation.unexpected_tensor_names == ("blk.3.unexpected.weight",)
    with pytest.raises(Qwen4ExpGGUFTensorMapError, match="missing"):
        build_qwen4_exp_gguf_tensor_map(infos)


def test_qwen4_exp_gguf_tensor_map_rejects_shape_drift_and_duplicates() -> None:
    wrong = _tensor("blk.3.attn_q.weight", (6_144, 2_560))
    validation = validate_qwen4_exp_gguf_tensor_map(
        _infos(replacements={wrong.name: wrong}, duplicate="output.weight")
    )

    assert validation.passed is False
    assert "blk.3.attn_q.weight" in validation.shape_errors[0]
    assert validation.duplicate_tensor_names == ("output.weight",)


def test_qwen4_exp_gguf_tensor_map_accepts_only_bounded_unreachable_ple_padding() -> None:
    padded = build_qwen4_exp_gguf_tensor_map(_infos(ple_rows=320_001_536))
    assert padded.ple_padding_rows == 90

    invalid = validate_qwen4_exp_gguf_tensor_map(_infos(ple_rows=320_001_702))
    assert invalid.passed is False
    assert "per_layer_token_embd.weight" in invalid.shape_errors[0]
