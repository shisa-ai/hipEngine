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


def test_rank2_iq4_xs_keeps_dense_bf16_fallback_without_model_fixture() -> None:
    tensor = _tensor_info(
        GGMLQuantizationType.IQ4_XS,
        (8, 256),
        name="blk.0.ffn_gate.weight",
    )

    spec = _spec_for_tensor("layers.0.ffn_gate", tensor, decode_repack=False)

    assert spec.layout == LAYOUT_DENSE_BF16
    assert spec.quant_key == "gguf_iq4_xs"
    assert spec.allocation_names == ("raw",)


@pytest.mark.parametrize("qtype", [GGMLQuantizationType.IQ3_XXS, GGMLQuantizationType.Q3_K])
def test_rank2_iq3_and_q3_are_rejected_without_model_fixture(
    qtype: GGMLQuantizationType,
) -> None:
    tensor = _tensor_info(qtype, (8, 256), name="blk.0.ffn_gate.weight")

    with pytest.raises(ValueError, match="outside rank-3 expert slots"):
        _spec_for_tensor("layers.0.ffn_gate", tensor, decode_repack=False)


@pytest.mark.skipif(not MODEL.exists(), reason=f"local GGUF fixture not found: {MODEL}")
def test_qwen35moe_ud_q3_k_m_plan_keeps_iq_experts_raw() -> None:
    reader = GGUFReader(MODEL)
    model_map = build_qwen35_gguf_tensor_map(reader.info)
    plan = plan_qwen35_gguf_materialization(model_map, decode_repack=False)

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
    assert down_quants == Counter({"gguf_iq4_xs": 37, "gguf_q6_k": 3})

    # Main body: IQ3_XXS gate/up + IQ4_XS down experts stay compressed.
    layer0 = plan.layer_specs[0]
    assert layer0["ffn_gate_exps"].layout == LAYOUT_RAW_GGUF
    assert layer0["ffn_gate_exps"].quant_key == "gguf_iq3_xxs"
    assert layer0["ffn_gate_exps"].allocation_names == ("raw",)
    assert layer0["ffn_up_exps"].layout == LAYOUT_RAW_GGUF
    assert layer0["ffn_up_exps"].quant_key == "gguf_iq3_xxs"
    assert layer0["ffn_down_exps"].layout == LAYOUT_RAW_GGUF
    assert layer0["ffn_down_exps"].quant_key == "gguf_iq4_xs"

    # Deep-layer outlier: IQ4_XS gate/up + already-supported Q6_K down.
    layer39 = plan.layer_specs[39]
    assert layer39["ffn_gate_exps"].layout == LAYOUT_RAW_GGUF
    assert layer39["ffn_gate_exps"].quant_key == "gguf_iq4_xs"
    assert layer39["ffn_up_exps"].layout == LAYOUT_RAW_GGUF
    assert layer39["ffn_up_exps"].quant_key == "gguf_iq4_xs"
    assert layer39["ffn_down_exps"].layout == LAYOUT_RAW_GGUF
    assert layer39["ffn_down_exps"].quant_key == "gguf_q6_k"
