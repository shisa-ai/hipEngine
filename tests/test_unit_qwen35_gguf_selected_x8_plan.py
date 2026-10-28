from __future__ import annotations

import pytest

from hipengine.loading import qwen35_gguf_materialize as materialize
from hipengine.loading.gguf import GGUFTensorInfo
from hipengine.quant.gguf import GGMLQuantizationType, nbytes_for_shape, quant_shape_to_byte_shape


def _expert_tensor(slot: str, qtype: GGMLQuantizationType) -> GGUFTensorInfo:
    shape = (2, 256, 256)
    return GGUFTensorInfo(
        name=f"blk.0.{slot}.weight", shape=shape, ggml_shape=tuple(reversed(shape)),
        ggml_type=int(qtype), ggml_type_name=qtype.name, n_elements=2 * 256 * 256,
        nbytes=nbytes_for_shape(shape, qtype), offset=0, data_offset=0,
        byte_shape=quant_shape_to_byte_shape(shape, qtype),
    )


@pytest.mark.parametrize("mode", ["off", "q5", "q6", "both"])
@pytest.mark.parametrize("qtype", [GGMLQuantizationType.Q5_K, GGMLQuantizationType.Q6_K])
@pytest.mark.parametrize("decode_repack", [False, True])
@pytest.mark.parametrize("slot", ["ffn_down_exps", "ffn_gate_exps"])
def test_selected_down_x8_is_a_quant_scoped_replacement_layout(
    monkeypatch, mode: str, qtype: GGMLQuantizationType, decode_repack: bool, slot: str,
) -> None:
    monkeypatch.setenv("HIPENGINE_GGUF_SELECTED_X8_REPACK", mode)
    monkeypatch.setenv("HIPENGINE_GGUF_SELECTED_DOWN_RAW", "off")
    tensor = _expert_tensor(slot, qtype)
    spec = materialize.plan_qwen35_gguf_weight_spec(
        f"layers.0.{slot}", tensor, decode_repack=decode_repack,
    )
    quant = "q5" if qtype == GGMLQuantizationType.Q5_K else "q6"
    x8 = decode_repack and slot == "ffn_down_exps" and mode in {quant, "both"}
    if not decode_repack:
        assert spec.quant_key == f"gguf_{quant}_k"
        assert spec.allocation_names == ("raw",)
    elif x8:
        assert spec.quant_key == f"gguf_{quant}_k_x8_v1"
        assert spec.allocation_names == ("tiles",)
        assert dict(materialize.planned_qwen35_gguf_weight_allocation_nbytes(spec)) == {
            "tiles": tensor.nbytes,
        }
    else:
        expected = "gguf_q5_k_qmicro_t16_v1" if quant == "q5" else "gguf_q6_k_t16_v1"
        assert spec.quant_key == expected
        assert spec.allocation_names == ("tiles",)


@pytest.mark.parametrize("qtype", [GGMLQuantizationType.Q5_K, GGMLQuantizationType.Q6_K])
def test_explicit_raw_selected_down_precedes_x8_replacement(monkeypatch, qtype) -> None:
    monkeypatch.setenv("HIPENGINE_GGUF_SELECTED_X8_REPACK", "both")
    monkeypatch.setenv("HIPENGINE_GGUF_SELECTED_DOWN_RAW", "both")
    spec = materialize.plan_qwen35_gguf_weight_spec(
        "layers.0.ffn_down_exps", _expert_tensor("ffn_down_exps", qtype), decode_repack=True,
    )
    assert spec.layout == materialize.LAYOUT_RAW_GGUF
    assert spec.allocation_names == ("raw",)


@pytest.mark.parametrize("value,expected", [
    (False, "off"), (True, "both"), (" q5_k ", "q5"), ("gguf_q6_k", "q6"),
    ("all", "both"), ("off", "off"),
])
def test_selected_down_x8_explicit_argument_overrides_ambient_mode(monkeypatch, value, expected) -> None:
    monkeypatch.setenv("HIPENGINE_GGUF_SELECTED_X8_REPACK", "invalid")
    assert materialize.gguf_selected_x8_repack_mode(value) == expected
