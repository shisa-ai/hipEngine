from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hipengine.loading.gguf import GGUFReader
from hipengine.loading.qwen35_gguf import build_qwen35_gguf_tensor_map
from hipengine.loading.qwen35_gguf_expert_sidecar import (
    EXPERT_SIDECAR_FORMAT,
    EXPERT_SIDECAR_LAYOUT,
    GGUFExpertPackedTensor,
    dequantize_packed_expert_tensor,
    load_packed_expert_tensor,
    pack_gguf_expert_tensor,
    reference_dequantize_expert_tensor,
    save_packed_expert_tensor,
)
from hipengine.loading.qwen35_gguf_materialize import (
    LAYOUT_GGUF_EXPERT_PACK8_SIDECAR,
    plan_qwen35_gguf_materialization,
)
from hipengine.quant.gguf import GGMLQuantizationType, quant_layout

MOE_MODEL = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")


@pytest.mark.parametrize(
    "qtype",
    (GGMLQuantizationType.Q4_K, GGMLQuantizationType.Q5_K, GGMLQuantizationType.Q6_K),
)
def test_expert_sidecar_pack8_dequant_matches_raw_gguf(qtype: GGMLQuantizationType) -> None:
    raw = _synthetic_expert_blocks(qtype, experts=2, out_features=16, blocks_per_row=2)

    packed = pack_gguf_expert_tensor(raw, qtype, tensor_name=f"synthetic.{qtype.name}", slot="ffn_gate_exps")
    actual = dequantize_packed_expert_tensor(packed)
    expected = reference_dequantize_expert_tensor(raw, qtype)

    assert packed.format == EXPERT_SIDECAR_FORMAT
    assert packed.metadata()["layout"] == EXPERT_SIDECAR_LAYOUT
    assert packed.qweight_low.shape == (2, 2, 512)
    assert packed.scales.shape[0] == 2
    if qtype == GGMLQuantizationType.Q4_K:
        assert packed.qweight_high is None
        assert packed.mins is not None
        assert packed.scales.shape == (2, 16, 16)
    elif qtype == GGMLQuantizationType.Q5_K:
        assert packed.qweight_high is not None
        assert packed.qweight_high.dtype == np.uint8
        assert packed.mins is not None
        assert packed.scales.shape == (2, 16, 16)
    else:
        assert packed.qweight_high is not None
        assert packed.qweight_high.dtype == np.uint16
        assert packed.mins is None
        assert packed.scales.shape == (2, 32, 16)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)


def test_expert_sidecar_save_load_roundtrip(tmp_path: Path) -> None:
    raw = _synthetic_expert_blocks(GGMLQuantizationType.Q5_K, experts=1, out_features=8, blocks_per_row=1)
    packed = pack_gguf_expert_tensor(raw, GGMLQuantizationType.Q5_K, tensor_name="blk.0.ffn_down_exps.weight", slot="ffn_down_exps")

    path = save_packed_expert_tensor(tmp_path / "sidecar.npz", packed)
    loaded = load_packed_expert_tensor(path)

    assert isinstance(loaded, GGUFExpertPackedTensor)
    assert loaded.metadata() == packed.metadata()
    np.testing.assert_array_equal(loaded.qweight_low, packed.qweight_low)
    assert loaded.qweight_high is not None and packed.qweight_high is not None
    np.testing.assert_array_equal(loaded.qweight_high, packed.qweight_high)
    assert loaded.mins is not None and packed.mins is not None
    np.testing.assert_array_equal(loaded.mins, packed.mins)
    np.testing.assert_array_equal(loaded.scales, packed.scales)


def test_qwen35moe_plan_marks_expert_sidecar_eligible() -> None:
    if not MOE_MODEL.exists():
        pytest.skip(f"local GGUF fixture not found: {MOE_MODEL}")
    reader = GGUFReader(MOE_MODEL)
    model_map = build_qwen35_gguf_tensor_map(reader.info)
    plan = plan_qwen35_gguf_materialization(model_map)

    layer0 = plan.layer_specs[0]
    for slot in ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"):
        assert layer0[slot].layout == "raw_gguf"
        assert layer0[slot].sidecar_layouts == (LAYOUT_GGUF_EXPERT_PACK8_SIDECAR,)


def _synthetic_expert_blocks(
    qtype: GGMLQuantizationType,
    *,
    experts: int,
    out_features: int,
    blocks_per_row: int,
) -> np.ndarray:
    layout = quant_layout(qtype)
    rng = np.random.default_rng(1234 + int(qtype))
    blocks = rng.integers(
        0,
        256,
        size=(experts, out_features, blocks_per_row, layout.type_size),
        dtype=np.uint8,
    )
    if qtype == GGMLQuantizationType.Q4_K:
        _store_f16(blocks, 0, 0.25)
        _store_f16(blocks, 2, 0.125)
    elif qtype == GGMLQuantizationType.Q5_K:
        _store_f16(blocks, 0, 0.1875)
        _store_f16(blocks, 2, 0.0625)
    elif qtype == GGMLQuantizationType.Q6_K:
        _store_f16(blocks, 208, 0.03125)
    else:  # pragma: no cover - parametrization guards this.
        raise AssertionError(qtype)
    return blocks.reshape(experts, out_features, blocks_per_row * layout.type_size)


def _store_f16(blocks: np.ndarray, offset: int, value: float) -> None:
    bits = np.asarray([value], dtype=np.float16).view(np.uint8)
    blocks[..., offset : offset + 2] = bits
