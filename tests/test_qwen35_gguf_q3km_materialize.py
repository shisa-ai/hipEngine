from __future__ import annotations

from pathlib import Path

import pytest

from hipengine.loading.gguf import GGUFReader
from hipengine.loading.qwen35_gguf import build_qwen35_gguf_tensor_map
from hipengine.loading.qwen35_gguf_materialize import LAYOUT_RAW_GGUF, plan_qwen35_gguf_materialization

MODEL = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q3_K_M.gguf")
pytestmark = pytest.mark.skipif(not MODEL.exists(), reason=f"local GGUF fixture not found: {MODEL}")


def test_qwen35moe_ud_q3_k_m_plan_keeps_iq_experts_raw() -> None:
    reader = GGUFReader(MODEL)
    model_map = build_qwen35_gguf_tensor_map(reader.info)
    plan = plan_qwen35_gguf_materialization(model_map, decode_repack=False)

    # The AR map covers blk.0-39. blk.40 is the nextn/MTP block and is
    # intentionally ignored by the base generate path.
    missing = {tensor.name for tensor in reader.info.tensors} - set(plan.tensor_names)
    assert missing
    assert all(name.startswith("blk.40.") for name in missing)
    assert len(plan.layer_specs) == 40

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
