from __future__ import annotations

import ctypes
from pathlib import Path

import pytest

from hipengine.core.dtype import DType
from hipengine.core.hip import get_hip_runtime
from hipengine.loading.gguf import GGUFReader
from hipengine.loading.qwen35_gguf import build_qwen35_gguf_tensor_map
from hipengine.loading.qwen35_gguf_materialize import (
    HIPENGINE_GGUF_DECODE_REPACK_ENV,
    LAYOUT_DENSE_BF16,
    LAYOUT_DENSE_F32,
    LAYOUT_GGUF_EXPERT_PACK8_SIDECAR,
    LAYOUT_GGUF_Q4_K_T16,
    LAYOUT_GGUF_Q5_K_T16,
    LAYOUT_GGUF_Q6_K_T16,
    LAYOUT_GGUF_Q8_0_T16,
    LAYOUT_Q4_K_PACK8,
    LAYOUT_RAW_GGUF,
    gguf_decode_repack_enabled,
    materialize_qwen35_gguf_weights,
    plan_qwen35_gguf_materialization,
)

MODEL = Path("/models/gguf/Qwen3.5-0.8B-Q4_K_M.gguf")
MOE_MODEL = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
Q4_1_MODEL = Path("/models/gguf/Qwen3.5-0.8B-Q4_1.gguf")
Q8_0_MODEL = Path("/models/gguf/Qwen3.5-0.8B-Q8_0.gguf")
UD_Q4_K_XL_MODEL = Path("/models/gguf/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
pytestmark = pytest.mark.skipif(not MODEL.exists(), reason=f"local GGUF fixture not found: {MODEL}")


def test_qwen35_gguf_materialization_plan_covers_every_tensor() -> None:
    reader = GGUFReader(MODEL)
    model_map = build_qwen35_gguf_tensor_map(reader.info)
    plan = plan_qwen35_gguf_materialization(model_map, decode_repack=False)

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
    assert layer0["attn_qkv"].layout == LAYOUT_DENSE_BF16
    assert layer0["attn_qkv"].quant_key == "gguf_q5_k"
    assert layer0["ssm_alpha"].quant_key == "gguf_q8_0"

    layer3 = plan.layer_specs[3]
    assert layer3["attn_q"].layout == LAYOUT_Q4_K_PACK8
    assert layer3["attn_v"].layout == LAYOUT_DENSE_BF16
    assert layer3["attn_v"].quant_key == "gguf_q6_k"


def test_qwen35moe_gguf_materialization_plan_keeps_experts_raw() -> None:
    if not MOE_MODEL.exists():
        pytest.skip(f"local GGUF fixture not found: {MOE_MODEL}")
    reader = GGUFReader(MOE_MODEL)
    model_map = build_qwen35_gguf_tensor_map(reader.info)
    plan = plan_qwen35_gguf_materialization(model_map, decode_repack=False)

    assert set(plan.tensor_names) == {tensor.name for tensor in reader.info.tensors}
    assert plan.root_specs["lm_head"].source.name == "output.weight"
    assert plan.root_specs["lm_head"].layout == LAYOUT_RAW_GGUF
    assert plan.root_specs["lm_head"].quant_key == "gguf_q6_k"

    layer0 = plan.layer_specs[0]
    assert layer0["ffn_gate_inp"].layout == LAYOUT_DENSE_BF16
    assert layer0["ffn_gate_inp"].quant_key == "bf16"
    assert layer0["ffn_gate_inp_shexp"].layout == LAYOUT_DENSE_BF16
    assert layer0["ffn_gate_exps"].layout == LAYOUT_RAW_GGUF
    assert layer0["ffn_gate_exps"].quant_key == "gguf_q4_k"
    assert layer0["ffn_gate_exps"].sidecar_layouts == (LAYOUT_GGUF_EXPERT_PACK8_SIDECAR,)
    assert layer0["ffn_up_exps"].layout == LAYOUT_RAW_GGUF
    assert layer0["ffn_up_exps"].sidecar_layouts == (LAYOUT_GGUF_EXPERT_PACK8_SIDECAR,)
    assert layer0["ffn_down_exps"].layout == LAYOUT_RAW_GGUF
    assert layer0["ffn_down_exps"].quant_key == "gguf_q5_k"
    assert layer0["ffn_down_exps"].sidecar_layouts == (LAYOUT_GGUF_EXPERT_PACK8_SIDECAR,)
    assert layer0["ffn_gate_shexp"].layout == LAYOUT_RAW_GGUF
    assert layer0["ffn_down_shexp"].layout == LAYOUT_RAW_GGUF


def test_qwen35moe_decode_repack_plan_replaces_covered_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    if not MOE_MODEL.exists():
        pytest.skip(f"local GGUF fixture not found: {MOE_MODEL}")
    monkeypatch.setenv(HIPENGINE_GGUF_DECODE_REPACK_ENV, "1")
    reader = GGUFReader(MOE_MODEL)
    model_map = build_qwen35_gguf_tensor_map(reader.info)

    assert gguf_decode_repack_enabled() is True
    plan = plan_qwen35_gguf_materialization(model_map)

    assert set(plan.tensor_names) == {tensor.name for tensor in reader.info.tensors}
    assert plan.root_specs["lm_head"].layout == LAYOUT_GGUF_Q6_K_T16
    assert plan.root_specs["lm_head"].quant_key == "gguf_q6_k_t16_v1"
    assert plan.root_specs["lm_head"].allocation_names == ("tiles",)

    layer0 = plan.layer_specs[0]
    assert layer0["ffn_gate_exps"].layout == LAYOUT_GGUF_Q4_K_T16
    assert layer0["ffn_gate_exps"].quant_key == "gguf_q4_k_t16_v1"
    assert layer0["ffn_gate_exps"].allocation_names == ("tiles",)
    assert layer0["ffn_gate_exps"].sidecar_layouts == ()
    assert layer0["ffn_up_exps"].layout == LAYOUT_GGUF_Q4_K_T16
    assert layer0["ffn_down_exps"].layout == LAYOUT_GGUF_Q5_K_T16
    assert layer0["ffn_down_exps"].quant_key == "gguf_q5_k_t16_v1"
    assert layer0["ffn_gate_shexp"].layout == LAYOUT_GGUF_Q8_0_T16
    assert layer0["ffn_gate_shexp"].quant_key == "gguf_q8_0_t16_v1"
    assert layer0["ffn_gate_inp"].layout == LAYOUT_DENSE_BF16


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        (
            Q4_1_MODEL,
            {
                "root.token_embedding": (LAYOUT_RAW_GGUF, "gguf_q6_k"),
                "layers.0.attn_gate": (LAYOUT_DENSE_BF16, "gguf_q4_1"),
                "layers.0.attn_qkv": (LAYOUT_DENSE_BF16, "gguf_q4_1"),
                "layers.0.ssm_alpha": (LAYOUT_RAW_GGUF, "gguf_q8_0"),
            },
        ),
        (
            Q8_0_MODEL,
            {
                "root.token_embedding": (LAYOUT_RAW_GGUF, "gguf_q8_0"),
                "layers.0.attn_gate": (LAYOUT_RAW_GGUF, "gguf_q8_0"),
                "layers.3.attn_q": (LAYOUT_RAW_GGUF, "gguf_q8_0"),
            },
        ),
        (
            UD_Q4_K_XL_MODEL,
            {
                "root.token_embedding": (LAYOUT_RAW_GGUF, "gguf_q6_k"),
                "layers.0.ssm_alpha": (LAYOUT_DENSE_BF16, "fp16"),
                "layers.8.ffn_gate": (LAYOUT_DENSE_BF16, "gguf_iq4_xs"),
                "layers.3.attn_q": (LAYOUT_DENSE_BF16, "gguf_q5_k"),
            },
        ),
    ],
)
def test_qwen35_gguf_materialization_plan_covers_local_quant_variants(
    path: Path, expected: dict[str, tuple[str, str]]
) -> None:
    if not path.exists():
        pytest.skip(f"local GGUF fixture not found: {path}")
    reader = GGUFReader(path)
    model_map = build_qwen35_gguf_tensor_map(reader.info)
    plan = plan_qwen35_gguf_materialization(model_map, decode_repack=False)

    assert set(plan.tensor_names) == {tensor.name for tensor in reader.info.tensors}
    for slot_path, (layout, quant_key) in expected.items():
        spec = _spec_by_path(plan, slot_path)
        assert spec.layout == layout
        assert spec.quant_key == quant_key


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
        decode_repack=False,
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
        assert attn_qkv.spec.layout == LAYOUT_DENSE_BF16
        assert attn_qkv.spec.quant_key == "gguf_q5_k"
        assert attn_qkv.allocation().tensor.dtype == DType.BF16
        assert attn_qkv.allocation().tensor.shape == (6144, 1024)

        ssm_alpha = resident.layer(0).weight("ssm_alpha")
        assert ssm_alpha.spec.quant_key == "gguf_q8_0"
        assert ssm_alpha.allocation().tensor.shape == (16, 1088)

        attn_v = resident.layer(3).weight("attn_v")
        assert attn_v.spec.layout == LAYOUT_DENSE_BF16
        assert attn_v.spec.quant_key == "gguf_q6_k"
        assert attn_v.allocation().tensor.dtype == DType.BF16
        assert attn_v.allocation().tensor.shape == (512, 1024)
    finally:
        resident.free(runtime=runtime)


def test_qwen35_gguf_materializes_dense_bf16_fallback_records() -> None:
    if not _hip_available():
        pytest.skip("HIP runtime is not available")
    if not (Q4_1_MODEL.exists() and UD_Q4_K_XL_MODEL.exists()):
        pytest.skip("local Q4_1/UD-Q4_K_XL GGUF fixtures are not available")
    runtime = get_hip_runtime()

    q4_1 = materialize_qwen35_gguf_weights(
        Q4_1_MODEL,
        selected_slots=("layers.0.attn_gate",),
        decode_repack=False,
        runtime=runtime,
    )
    try:
        attn_gate = q4_1.layer(0).weight("attn_gate")
        assert attn_gate.spec.layout == LAYOUT_DENSE_BF16
        assert attn_gate.spec.quant_key == "gguf_q4_1"
        assert attn_gate.allocation().tensor.dtype == DType.BF16
        assert attn_gate.allocation().tensor.shape == (2048, 1024)
    finally:
        q4_1.free(runtime=runtime)

    ud = materialize_qwen35_gguf_weights(
        UD_Q4_K_XL_MODEL,
        selected_slots=("layers.0.ssm_alpha", "layers.8.ffn_gate"),
        decode_repack=False,
        runtime=runtime,
    )
    try:
        ssm_alpha = ud.layer(0).weight("ssm_alpha")
        assert ssm_alpha.spec.layout == LAYOUT_DENSE_BF16
        assert ssm_alpha.spec.quant_key == "fp16"
        assert ssm_alpha.allocation().tensor.dtype == DType.BF16
        assert ssm_alpha.allocation().tensor.shape == (16, 1024)

        ffn_gate = ud.layer(8).weight("ffn_gate")
        assert ffn_gate.spec.layout == LAYOUT_DENSE_BF16
        assert ffn_gate.spec.quant_key == "gguf_iq4_xs"
        assert ffn_gate.allocation().tensor.dtype == DType.BF16
        assert ffn_gate.allocation().tensor.shape == (3584, 1024)
    finally:
        ud.free(runtime=runtime)


def _spec_by_path(plan, slot_path: str):
    root, *rest = slot_path.split(".")
    if root == "root" and len(rest) == 1:
        return plan.root_specs[rest[0]]
    if root == "layers" and len(rest) == 2:
        return plan.layer_specs[int(rest[0])][rest[1]]
    raise AssertionError(f"unsupported slot path {slot_path!r}")


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True
