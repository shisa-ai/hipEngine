from __future__ import annotations

from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest

from hipengine.loading.gguf import GGUFReader, GGUFTensorInfo
from hipengine.loading.qwen35_gguf import build_qwen35_gguf_tensor_map
from hipengine.loading.qwen35_gguf_materialize import (
    LAYOUT_DENSE_BF16,
    LAYOUT_DENSE_F32,
    HIPENGINE_GGUF_DECODE_REPACK_ENV,
    LAYOUT_GGUF_Q5_K_QMICRO_T16,
    LAYOUT_Q4_K_PACK8,
    Qwen35GGUFMaterializationPlan,
    Qwen35GGUFWeightSpec,
    _gguf_ssm_a_to_kernel_a_log,
    audit_qwen35_gguf_precision_contractions,
    plan_qwen35_gguf_materialization,
    plan_qwen35_gguf_selective_weight_arena,
    plan_qwen35_gguf_weight_spec,
)
from hipengine.quant.gguf import GGMLQuantizationType


MOE_MODEL = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
DENSE_MODEL = Path("/models/gguf/Qwen3.6-27B-Q4_K_M.gguf")


def test_qwen36_dense_wide_weight_arena_plan_matches_exact_inventory() -> None:
    if not DENSE_MODEL.exists():
        pytest.skip(f"local GGUF fixture not found: {DENSE_MODEL}")
    reader = GGUFReader(DENSE_MODEL)
    plan = plan_qwen35_gguf_materialization(
        build_qwen35_gguf_tensor_map(reader.info),
        decode_repack=True,
        dense_q4_t16=True,
        dense_q5_t16_ssm_out=True,
        dense_q6_qmicro_planar=True,
    )

    arena = plan_qwen35_gguf_selective_weight_arena(
        plan,
        deferred_device_slots=("root.token_embedding",),
        max_allocation_bytes=80 * 1024 * 1024,
    )

    assert arena.supported is True
    assert arena.reason is None
    assert arena.alignment == 4096
    assert arena.max_allocation_bytes == 80 * 1024 * 1024
    assert arena.allocation_count == 849
    assert arena.requested_bytes == 15_363_373_056
    assert arena.capacity_bytes == 15_364_018_176
    assert arena.dedicated_allocation_count == 1
    assert arena.dedicated_requested_bytes == 1_042_944_000


def test_qwen35moe_selective_weight_arena_plan_matches_exact_inventory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not MOE_MODEL.exists():
        pytest.skip(f"local GGUF fixture not found: {MOE_MODEL}")
    monkeypatch.setenv(HIPENGINE_GGUF_DECODE_REPACK_ENV, "1")
    reader = GGUFReader(MOE_MODEL)
    plan = plan_qwen35_gguf_materialization(build_qwen35_gguf_tensor_map(reader.info))

    arena = plan_qwen35_gguf_selective_weight_arena(
        plan,
        deferred_device_slots=("root.token_embedding",),
    )

    assert arena.supported is True
    assert arena.reason is None
    assert arena.alignment == 4096
    assert arena.max_allocation_bytes == 16 * 1024 * 1024
    assert arena.allocation_count == 571
    assert arena.requested_bytes == 884_460_032
    assert arena.capacity_bytes == 884_867_072
    assert arena.dedicated_allocation_count == 161
    assert arena.dedicated_requested_bytes == 21_034_278_912


def test_selective_weight_arena_plan_fails_closed_for_unplanned_pack8_layout() -> None:
    spec = _spec(
        "layers.0.attn_gate",
        "blk.0.attn_gate.weight",
        GGMLQuantizationType.Q4_K,
        LAYOUT_Q4_K_PACK8,
        "gguf_q4_k_pack8_v1",
    )
    plan = Qwen35GGUFMaterializationPlan(
        config=None,
        root_specs=MappingProxyType({}),
        layer_specs=(MappingProxyType({"attn_gate": spec}),),
    )

    arena = plan_qwen35_gguf_selective_weight_arena(plan)

    assert arena.supported is False
    assert arena.capacity_bytes == 0
    assert "q4_k_pack8" in str(arena.reason)


def test_selected_q5_decode_repack_uses_qmicro_t16_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HIPENGINE_GGUF_SELECTED_DOWN_RAW", raising=False)
    monkeypatch.delenv("HIPENGINE_GGUF_SELECTED_X8_REPACK", raising=False)
    tensor = GGUFTensorInfo(
        name="blk.0.ffn_down_exps.weight",
        shape=(256, 2048, 512),
        ggml_shape=(512, 2048, 256),
        ggml_type=int(GGMLQuantizationType.Q5_K),
        ggml_type_name="Q5_K",
        n_elements=256 * 2048 * 512,
        nbytes=184_549_376,
        offset=0,
        data_offset=0,
        byte_shape=(256, 2048, 352),
    )

    spec = plan_qwen35_gguf_weight_spec(
        "layers.0.ffn_down_exps", tensor, decode_repack=True
    )

    assert spec.layout == LAYOUT_GGUF_Q5_K_QMICRO_T16
    assert spec.quant_key == "gguf_q5_k_qmicro_t16_v1"
    assert spec.allocation_names == ("tiles",)


def test_gguf_ssm_a_materialization_converts_decay_coefficients_to_kernel_log() -> None:
    coeff = np.asarray([-1.0, -0.25, -72.0], dtype=np.float32)
    converted = _gguf_ssm_a_to_kernel_a_log(coeff)

    assert converted.dtype == np.float32
    np.testing.assert_allclose(-np.exp(converted), coeff, rtol=1.0e-6, atol=1.0e-6)

    with pytest.raises(ValueError, match="negative decay coefficients"):
        _gguf_ssm_a_to_kernel_a_log(np.asarray([-1.0, 0.0], dtype=np.float32))
    with pytest.raises(ValueError, match="non-finite"):
        _gguf_ssm_a_to_kernel_a_log(np.asarray([-1.0, np.nan], dtype=np.float32))


def test_precision_contraction_audit_ignores_source_f32_tensors_retained_as_f32() -> None:
    plan = Qwen35GGUFMaterializationPlan(
        config=None,  # helper audit does not inspect model config
        root_specs=MappingProxyType(
            {
                "output_norm": _spec(
                    "root.output_norm",
                    "output_norm.weight",
                    GGMLQuantizationType.F32,
                    LAYOUT_DENSE_F32,
                    "f32",
                )
            }
        ),
        layer_specs=(
            MappingProxyType(
                {
                    "ffn_gate_inp": _spec(
                        "layers.0.ffn_gate_inp",
                        "blk.0.ffn_gate_inp.weight",
                        GGMLQuantizationType.F32,
                        LAYOUT_DENSE_F32,
                        "f32",
                    ),
                    "ssm_alpha": _spec(
                        "layers.0.ssm_alpha",
                        "blk.0.ssm_alpha.weight",
                        GGMLQuantizationType.F32,
                        LAYOUT_DENSE_F32,
                        "f32",
                    ),
                    "attn_qkv": _spec(
                        "layers.0.attn_qkv",
                        "blk.0.attn_qkv.weight",
                        GGMLQuantizationType.F16,
                        LAYOUT_DENSE_BF16,
                        "fp16",
                    ),
                }
            ),
        ),
    )

    findings = audit_qwen35_gguf_precision_contractions(plan)

    assert findings == ()


def _spec(
    slot_path: str,
    source_name: str,
    qtype: GGMLQuantizationType,
    layout: str,
    quant_key: str,
) -> Qwen35GGUFWeightSpec:
    return Qwen35GGUFWeightSpec(
        slot_path=slot_path,
        source=_tensor(source_name, qtype),
        quant_key=quant_key,
        layout=layout,
        allocation_names=("raw",),
    )


def _tensor(name: str, qtype: GGMLQuantizationType) -> GGUFTensorInfo:
    return GGUFTensorInfo(
        name=name,
        shape=(2, 3),
        ggml_shape=(3, 2),
        ggml_type=int(qtype),
        ggml_type_name=qtype.name,
        n_elements=6,
        nbytes=24,
        offset=0,
        data_offset=0,
        byte_shape=(2, 3),
    )
