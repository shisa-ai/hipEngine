from __future__ import annotations

import ctypes
from pathlib import Path

import pytest

from hipengine.core.dtype import DType
from hipengine.core.hip import get_hip_runtime
from hipengine.loading.gguf import GGUFReader
from hipengine.loading.qwen35_gguf import build_qwen35_gguf_tensor_map
from hipengine.loading.qwen35_gguf_materialize import (
    LAYOUT_DENSE_F32,
    LAYOUT_Q4_K_PACK8,
    LAYOUT_RAW_GGUF,
    materialize_qwen35_gguf_weights,
    plan_qwen35_gguf_materialization,
)

MODEL = Path("/models/gguf/Qwen3.5-0.8B-Q4_K_M.gguf")
pytestmark = pytest.mark.skipif(not MODEL.exists(), reason=f"local GGUF fixture not found: {MODEL}")


def test_qwen35_gguf_materialization_plan_covers_every_tensor() -> None:
    reader = GGUFReader(MODEL)
    model_map = build_qwen35_gguf_tensor_map(reader.info)
    plan = plan_qwen35_gguf_materialization(model_map)

    assert set(plan.tensor_names) == {tensor.name for tensor in reader.info.tensors}
    assert len(plan.tensor_names) == len(reader.info.tensors)
    assert plan.root_specs["token_embedding"].source.name == "token_embd.weight"
    assert plan.root_specs["token_embedding"].layout == LAYOUT_RAW_GGUF
    assert plan.root_specs["token_embedding"].quant_key == "gguf_q6_k"
    assert plan.root_specs["lm_head"].source.name == "token_embd.weight"
    assert plan.root_specs["lm_head"].layout == LAYOUT_RAW_GGUF
    assert plan.root_specs["output_norm"].layout == LAYOUT_DENSE_F32

    layer0 = plan.layer_specs[0]
    assert layer0["attn_gate"].layout == LAYOUT_Q4_K_PACK8
    assert layer0["attn_gate"].allocation_names == ("qweight", "scales", "mins")
    assert layer0["attn_qkv"].layout == LAYOUT_RAW_GGUF
    assert layer0["attn_qkv"].quant_key == "gguf_q5_k"
    assert layer0["ssm_alpha"].quant_key == "gguf_q8_0"

    layer3 = plan.layer_specs[3]
    assert layer3["attn_q"].layout == LAYOUT_Q4_K_PACK8
    assert layer3["attn_v"].layout == LAYOUT_RAW_GGUF
    assert layer3["attn_v"].quant_key == "gguf_q6_k"


def test_qwen35_gguf_materializes_selected_resident_records() -> None:
    if not _hip_available():
        pytest.skip("HIP runtime is not available")
    runtime = get_hip_runtime()
    resident = materialize_qwen35_gguf_weights(
        MODEL,
        selected_slots=(
            "root.output_norm",
            "layers.0.attn_gate",
            "layers.0.attn_qkv",
            "layers.0.ssm_alpha",
            "layers.3.attn_v",
        ),
        runtime=runtime,
    )
    try:
        output_norm = resident.root("output_norm").allocation()
        assert output_norm.tensor.shape == (1024,)
        assert output_norm.tensor.dtype == DType.FP32

        attn_gate = resident.layer(0).weight("attn_gate")
        assert attn_gate.spec.layout == LAYOUT_Q4_K_PACK8
        assert set(attn_gate.allocations) == {"qweight", "scales", "mins"}
        assert attn_gate.allocation("qweight").tensor.dtype == DType.INT32
        assert attn_gate.allocation("qweight").tensor.shape == (256, 1024)
        assert attn_gate.allocation("scales").tensor.dtype == DType.FP32
        assert attn_gate.allocation("scales").tensor.shape == (32, 2048)
        assert attn_gate.allocation("mins").tensor.shape == (32, 2048)

        attn_qkv = resident.layer(0).weight("attn_qkv")
        assert attn_qkv.spec.layout == LAYOUT_RAW_GGUF
        assert attn_qkv.spec.quant_key == "gguf_q5_k"
        assert attn_qkv.allocation().tensor.dtype == DType.INT8
        assert attn_qkv.allocation().tensor.shape == (6144, 704)

        ssm_alpha = resident.layer(0).weight("ssm_alpha")
        assert ssm_alpha.spec.quant_key == "gguf_q8_0"
        assert ssm_alpha.allocation().tensor.shape == (16, 1088)

        attn_v = resident.layer(3).weight("attn_v")
        assert attn_v.spec.quant_key == "gguf_q6_k"
        assert attn_v.allocation().tensor.shape == (512, 840)
    finally:
        resident.free(runtime=runtime)


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True
