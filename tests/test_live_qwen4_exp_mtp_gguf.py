from __future__ import annotations

from math import prod
from pathlib import Path

import pytest

from hipengine.loading.gguf import GGUFModelInfo, GGUFReader, GGUFTensorInfo
from hipengine.loading.qwen4_exp_mtp_gguf import (
    Qwen4ExpMTPGGUFError,
    _expected_qtypes,
    _expected_shapes,
    build_qwen4_exp_mtp_gguf_map,
    required_qwen4_exp_mtp_tensor_names,
    validate_qwen4_exp_mtp_gguf,
)
from hipengine.loading.qwen4_exp_mtp_materialize import (
    plan_qwen4_exp_mtp_residency,
)
from hipengine.quant.gguf import GGMLQuantizationType

_REAL_SIDECAR = Path(
    "/models/gguf/Qwen3.8-Flash-Next-MTP-Q8_0/"
    "mtp-Qwen3.8-Flash-Next-Q8_0.gguf"
)


def _metadata() -> dict[str, object]:
    return {
        "general.architecture": "qwen4exp",
        "qwen4exp.block_count": 49,
        "qwen4exp.nextn_predict_layers": 1,
        "qwen4exp.context_length": 262_144,
        "qwen4exp.embedding_length": 2_560,
        "qwen4exp.hyper_connection.count": 4,
        "qwen4exp.hyper_connection.low_rank": 320,
        "qwen4exp.attention.head_count": 24,
        "qwen4exp.attention.head_count_kv": 2,
        "qwen4exp.attention.key_length": 256,
        "qwen4exp.attention.indexer.head_count": 4,
        "qwen4exp.attention.indexer.key_length": 128,
        "qwen4exp.expert_count": 512,
        "qwen4exp.expert_used_count": 10,
        "qwen4exp.expert_feed_forward_length": 640,
        "qwen4exp.expert_shared_feed_forward_length": 640,
        "qwen4exp.attention.layer_norm_rms_epsilon": 1.0e-6,
        "qwen4exp.attention.compress_ratios": [0, 0, 0, 4] * 12 + [0],
    }


def _tensor(name: str, shape: tuple[int, ...], qtype: str) -> GGUFTensorInfo:
    quant = GGMLQuantizationType[qtype]
    return GGUFTensorInfo(
        name=name,
        shape=shape,
        ggml_shape=tuple(reversed(shape)),
        ggml_type=int(quant),
        ggml_type_name=qtype,
        n_elements=prod(shape),
        nbytes=prod(shape),
        offset=0,
        data_offset=0,
        byte_shape=shape,
    )


def _info(
    *,
    drop: set[str] | None = None,
    extra: GGUFTensorInfo | None = None,
    replace: GGUFTensorInfo | None = None,
) -> GGUFModelInfo:
    qtypes = _expected_qtypes()
    tensors = [
        _tensor(name, shape, qtypes[name])
        for name, shape in _expected_shapes().items()
        if not drop or name not in drop
    ]
    if replace is not None:
        tensors = [replace if tensor.name == replace.name else tensor for tensor in tensors]
    if extra is not None:
        tensors.append(extra)
    return GGUFModelInfo(
        path=Path("mtp-Qwen3.8-Flash-Next-Q8_0.gguf"),
        version=3,
        alignment=32,
        metadata=_metadata(),
        tensors=tuple(tensors),
        tensor_data_offset=0,
    )


def test_qwen4_exp_mtp_map_pins_complete_sidecar_inventory() -> None:
    model_map = build_qwen4_exp_mtp_gguf_map((_info(),))

    assert model_map.validation.passed
    assert len(required_qwen4_exp_mtp_tensor_names()) == 34
    assert len(model_map.tensor_refs) == 34
    assert model_map.config.block_count == 49
    assert model_map.config.nextn_predict_layers == 1
    assert model_map.config.residual_width == 10_240
    assert model_map.weight("nextn.eh_proj").tensor.shape == (2_560, 5_120)
    assert model_map.weight("layers.0.expert_down").tensor.shape == (
        512,
        2_560,
        640,
    )
    plan = plan_qwen4_exp_mtp_residency(model_map)
    assert len(plan.specs) == 34
    assert plan.alternate_layout_bytes == 0
    assert plan.replacement_payload_bytes == 0


def test_qwen4_exp_mtp_map_rejects_missing_extra_shape_and_qtype_drift() -> None:
    wrong = _tensor("blk.48.nextn.eh_proj.weight", (2_560, 2_560), "BF16")
    validation = validate_qwen4_exp_mtp_gguf(
        (
            _info(
                drop={"blk.48.attn_v.weight"},
                extra=_tensor("blk.48.unexpected.weight", (1,), "F32"),
                replace=wrong,
            ),
        )
    )

    assert validation.passed is False
    assert validation.missing_tensor_names == ("blk.48.attn_v.weight",)
    assert validation.unexpected_tensor_names == ("blk.48.unexpected.weight",)
    assert validation.shape_errors == (
        "blk.48.nextn.eh_proj.weight shape=(2560, 2560), expected (2560, 5120)",
    )
    assert validation.qtype_errors == (
        "blk.48.nextn.eh_proj.weight qtype=BF16, expected Q8_0",
    )
    with pytest.raises(Qwen4ExpMTPGGUFError, match="invalid Qwen4Exp MTP GGUF"):
        build_qwen4_exp_mtp_gguf_map((
            _info(drop={"blk.48.attn_v.weight"}),
        ))


@pytest.mark.skipif(not _REAL_SIDECAR.exists(), reason="pinned Qwen4Exp MTP sidecar is local-only")
def test_pinned_qwen4_exp_mtp_sidecar_header_matches_contract() -> None:
    reader = GGUFReader(_REAL_SIDECAR)
    model_map = build_qwen4_exp_mtp_gguf_map((reader.info,))
    plan = plan_qwen4_exp_mtp_residency(model_map)

    assert model_map.validation.passed
    assert len(model_map.tensor_refs) == 34
    assert plan.raw_payload_bytes == 4_126_482_432
    assert plan.device_weight_bytes == 4_126_482_432
