from __future__ import annotations

from collections import Counter
from math import prod
from pathlib import Path

import pytest

from hipengine.loading.gguf import GGUFReader, GGUFTensorInfo
from hipengine.loading.qwen35_gguf import build_qwen35_gguf_tensor_map
from hipengine.loading.qwen35_gguf_materialize import (
    LAYOUT_DENSE_BF16,
    LAYOUT_RAW_GGUF,
    _spec_for_tensor,
    plan_qwen35_gguf_materialization,
)
from hipengine.quant.gguf import (
    GGMLQuantizationType,
    nbytes_for_shape,
    quant_shape_to_byte_shape,
)

MODEL = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q3_K_M.gguf")


def _tensor_info(
    qtype: GGMLQuantizationType,
    shape: tuple[int, ...],
    *,
    name: str = "blk.0.ffn_gate_exps.weight",
) -> GGUFTensorInfo:
    return GGUFTensorInfo(
        name=name,
        shape=shape,
        ggml_shape=tuple(reversed(shape)),
        ggml_type=int(qtype),
        ggml_type_name=qtype.name,
        n_elements=prod(shape),
        nbytes=nbytes_for_shape(shape, qtype),
        offset=0,
        data_offset=0,
        byte_shape=quant_shape_to_byte_shape(shape, qtype),
    )


@pytest.mark.parametrize(
    ("qtype", "quant_key"),
    [
        (GGMLQuantizationType.IQ2_XS, "gguf_iq2_xs"),
        (GGMLQuantizationType.IQ3_XXS, "gguf_iq3_xxs"),
        (GGMLQuantizationType.IQ4_XS, "gguf_iq4_xs"),
        (GGMLQuantizationType.Q3_K, "gguf_q3_k"),
    ],
)
def test_rank3_iq_and_q3_experts_stay_raw_without_model_fixture(
    qtype: GGMLQuantizationType,
    quant_key: str,
) -> None:
    tensor = _tensor_info(qtype, (4, 8, 256))

    for decode_repack in (False, True):
        spec = _spec_for_tensor(
            "layers.0.ffn_gate_exps",
            tensor,
            decode_repack=decode_repack,
        )
        assert spec.layout == LAYOUT_RAW_GGUF
        assert spec.quant_key == quant_key
        assert spec.allocation_names == ("raw",)
        assert spec.sidecar_layouts == ()


@pytest.mark.parametrize("qtype", [GGMLQuantizationType.IQ2_XS, GGMLQuantizationType.IQ4_XS])
def test_rank2_iq2_iq4_xs_keep_dense_bf16_fallback_without_model_fixture(
    qtype: GGMLQuantizationType,
) -> None:
    tensor = _tensor_info(
        qtype,
        (8, 256),
        name="blk.0.ffn_gate.weight",
    )

    spec = _spec_for_tensor("layers.0.ffn_gate", tensor, decode_repack=False)

    assert spec.layout == LAYOUT_DENSE_BF16
    assert spec.quant_key == f"gguf_{qtype.name.lower()}"
    assert spec.allocation_names == ("raw",)


@pytest.mark.parametrize("qtype", [GGMLQuantizationType.IQ3_XXS, GGMLQuantizationType.Q3_K])
def test_rank2_iq3_and_q3_are_rejected_without_model_fixture(
    qtype: GGMLQuantizationType,
) -> None:
    tensor = _tensor_info(qtype, (8, 256), name="blk.0.ffn_gate.weight")

    with pytest.raises(ValueError, match="outside rank-3 expert slots"):
        _spec_for_tensor("layers.0.ffn_gate", tensor, decode_repack=False)


@pytest.mark.skipif(not MODEL.exists(), reason=f"local GGUF fixture not found: {MODEL}")
@pytest.mark.parametrize(
    (
        "decode_repack",
        "q6_quant_key",
        "q6_layout",
        "q6_allocation",
        "q8_quant_key",
        "q8_layout",
        "q8_allocation",
    ),
    [
        (
            False,
            "gguf_q6_k",
            LAYOUT_RAW_GGUF,
            "raw",
            "gguf_q8_0",
            LAYOUT_RAW_GGUF,
            "raw",
        ),
        (
            True,
            "gguf_q6_k",
            LAYOUT_RAW_GGUF,
            "raw",
            "gguf_q8_0",
            LAYOUT_RAW_GGUF,
            "raw",
        ),
    ],
)
def test_qwen35moe_ud_q3_k_m_plan_keeps_iq_experts_raw(
    decode_repack: bool,
    q6_quant_key: str,
    q6_layout: str,
    q6_allocation: str,
    q8_quant_key: str,
    q8_layout: str,
    q8_allocation: str,
) -> None:
    reader = GGUFReader(MODEL)
    model_map = build_qwen35_gguf_tensor_map(reader.info)
    plan = plan_qwen35_gguf_materialization(model_map, decode_repack=decode_repack)

    # The AR map covers blk.0-39. blk.40 is the nextn/MTP block and is
    # intentionally ignored by the base generate path.
    missing = {tensor.name for tensor in reader.info.tensors} - set(plan.tensor_names)
    assert len(missing) == 20
    assert all(name.startswith("blk.40.") for name in missing)
    assert len(plan.layer_specs) == 40

    gate_up_quants = Counter(
        layer[slot].quant_key
        for layer in plan.layer_specs
        for slot in ("ffn_gate_exps", "ffn_up_exps")
    )
    down_quants = Counter(layer["ffn_down_exps"].quant_key for layer in plan.layer_specs)
    assert gate_up_quants == Counter({"gguf_iq3_xxs": 78, "gguf_iq4_xs": 2})
    assert down_quants == Counter({"gguf_iq4_xs": 37, q6_quant_key: 3})
    nonexpert_quants = Counter(
        spec.quant_key
        for spec in plan.specs
        if not any(
            slot in spec.slot_path
            for slot in ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps")
        )
    )
    assert nonexpert_quants == Counter(
        {
            q8_quant_key: 250,
            "f32": 221,
            "bf16": 140,
            "gguf_q6_k": 2,
        }
    )
    assert all(spec.quant_key != "gguf_q3_k" for spec in plan.specs)

    # Main body: IQ3_XXS gate/up + IQ4_XS down experts stay compressed.
    layer0 = plan.layer_specs[0]
    assert layer0["ffn_gate_exps"].layout == LAYOUT_RAW_GGUF
    assert layer0["ffn_gate_exps"].quant_key == "gguf_iq3_xxs"
    assert layer0["ffn_gate_exps"].allocation_names == ("raw",)
    assert layer0["ffn_up_exps"].layout == LAYOUT_RAW_GGUF
    assert layer0["ffn_up_exps"].quant_key == "gguf_iq3_xxs"
    assert layer0["ffn_down_exps"].layout == LAYOUT_RAW_GGUF
    assert layer0["ffn_down_exps"].quant_key == "gguf_iq4_xs"

    # Router/dense/shared/GDN/full-attention roles keep their existing layouts.
    assert layer0["ffn_gate_inp"].layout == LAYOUT_DENSE_BF16
    assert layer0["ffn_gate_inp"].quant_key == "bf16"
    for slot in ("ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp", "attn_gate", "attn_qkv", "ssm_out"):
        assert layer0[slot].layout == q8_layout
        assert layer0[slot].quant_key == q8_quant_key
        assert layer0[slot].allocation_names == (q8_allocation,)
    full_attention_layer = plan.layer_specs[3]
    for slot in ("attn_q", "attn_k", "attn_v", "attn_output"):
        assert full_attention_layer[slot].layout == q8_layout
        assert full_attention_layer[slot].quant_key == q8_quant_key
    assert plan.root_specs["lm_head"].quant_key == q6_quant_key
    assert plan.root_specs["lm_head"].layout == q6_layout

    # Deep-layer outlier: IQ4_XS gate/up + already-supported Q6_K down.
    layer39 = plan.layer_specs[39]
    assert layer39["ffn_gate_exps"].layout == LAYOUT_RAW_GGUF
    assert layer39["ffn_gate_exps"].quant_key == "gguf_iq4_xs"
    assert layer39["ffn_up_exps"].layout == LAYOUT_RAW_GGUF
    assert layer39["ffn_up_exps"].quant_key == "gguf_iq4_xs"
    assert layer39["ffn_down_exps"].layout == q6_layout
    assert layer39["ffn_down_exps"].quant_key == q6_quant_key
    assert layer39["ffn_down_exps"].allocation_names == (q6_allocation,)
