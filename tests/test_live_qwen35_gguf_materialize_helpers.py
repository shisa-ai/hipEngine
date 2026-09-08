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
    LAYOUT_GGUF_Q4_K_T16,
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


# ---------------------------------------------------------------------------
# UD-U1: the AR raw-IQ decode-repack veto and the model-wide F32 linear
# contraction are separate policy knobs on the AR planner. Defaults are
# unchanged (both derive from the shared raw-IQ predicate), but enabling one
# must not silently move the other, and unchanged manifests keep identical
# plans (MoE semantics retained).
# ---------------------------------------------------------------------------


def _raw_iq_map() -> Qwen35GGUFMaterializationPlan:
    from hipengine.loading.qwen35_gguf import Qwen35GGUFConfig, Qwen35GGUFLayerMap, Qwen35GGUFModelMap
    from hipengine.loading.qwen35_gguf import LINEAR_ATTENTION
    from types import MappingProxyType as _MPP

    def tensor(name, shape, qtype):
        n = int(np.prod(shape)) if shape else 1
        return GGUFTensorInfo(
            name=name,
            shape=shape,
            ggml_shape=tuple(reversed(shape)),
            ggml_type=int(qtype),
            ggml_type_name=qtype.name,
            n_elements=n,
            nbytes=n * (4 if qtype == GGMLQuantizationType.F32 else 2),
            offset=0,
            data_offset=0,
            byte_shape=shape,
        )

    config = Qwen35GGUFConfig(
        architecture="qwen35",
        block_count=1,
        hidden_size=8,
        vocab_size=11,
        feed_forward_length=5,
        context_length=64,
        head_count=2,
        head_count_kv=1,
        key_length=4,
        value_length=4,
        full_attention_interval=4,
        layer_types=(LINEAR_ATTENTION,),
        rms_norm_eps=1e-6,
        rope_dimension_count=4,
        rope_dimension_sections=(),
        rope_freq_base=10000.0,
        ssm_inner_size=16,
        ssm_group_count=2,
        ssm_state_size=4,
        ssm_conv_kernel=2,
        ssm_time_step_rank=2,
        lm_head_tensor_name="token_embd.weight",
    )
    layer = {
        "attn_norm": tensor("blk.0.attn_norm.weight", (8,), GGMLQuantizationType.F32),
        "attn_qkv": tensor("blk.0.attn_qkv.weight", (28, 8), GGMLQuantizationType.IQ4_XS),
        "ssm_alpha": tensor("blk.0.ssm_alpha.weight", (2, 8), GGMLQuantizationType.F32),
        "ssm_beta": tensor("blk.0.ssm_beta.weight", (2, 8), GGMLQuantizationType.F32),
        "ssm_a": tensor("blk.0.ssm_a", (2,), GGMLQuantizationType.F32),
        "ssm_dt_bias": tensor("blk.0.ssm_dt.bias", (2,), GGMLQuantizationType.F32),
        "ssm_conv1d": tensor("blk.0.ssm_conv1d.weight", (28, 2), GGMLQuantizationType.F32),
        "ssm_norm": tensor("blk.0.ssm_norm.weight", (3,), GGMLQuantizationType.F32),
        "ssm_out": tensor("blk.0.ssm_out.weight", (8, 16), GGMLQuantizationType.Q4_K),
        "ffn_gate": tensor("blk.0.ffn_gate.weight", (17_408, 5_120), GGMLQuantizationType.Q4_K),
        "ffn_up": tensor("blk.0.ffn_up.weight", (17_408, 5_120), GGMLQuantizationType.Q4_K),
        "ffn_down": tensor("blk.0.ffn_down.weight", (5_120, 17_408), GGMLQuantizationType.Q4_K),
    }
    root = {
        "token_embedding": tensor("token_embd.weight", (11, 8), GGMLQuantizationType.Q4_K),
        "output_norm": tensor("output_norm.weight", (8,), GGMLQuantizationType.F32),
    }
    return Qwen35GGUFModelMap(
        config=config,
        root_tensors=_MPP(root),
        layers=(Qwen35GGUFLayerMap(layer_id=0, layer_type=LINEAR_ATTENTION, tensors=_MPP(layer)),),
        validation=None,
    )


def test_raw_iq_planner_defaults_per_tensor_repack_and_contract_f32():
    plan = plan_qwen35_gguf_materialization(
        _raw_iq_map(), decode_repack=True, dense_q4_t16=True
    )

    # UD-U3 per-tensor eligibility is the default: the Q4_K FFN slot repacks
    # to T16 while the model-wide F32 alpha/beta contraction stays bound to
    # its own raw-IQ knob.
    assert plan.layer_specs[0]["ffn_up"].layout == LAYOUT_GGUF_Q4_K_T16
    assert plan.layer_specs[0]["ssm_alpha"].layout == LAYOUT_DENSE_BF16
    assert plan.layer_specs[0]["ssm_alpha"].quant_key == "bf16"
    # Unrelated F32 slots keep their F32 residents.
    assert plan.layer_specs[0]["ssm_a"].layout == LAYOUT_DENSE_F32
    assert plan.layer_specs[0]["attn_norm"].layout == LAYOUT_DENSE_F32


def test_raw_iq_modelwide_repack_rollback_strips_repack():
    plan = plan_qwen35_gguf_materialization(
        _raw_iq_map(), decode_repack=True, dense_q4_t16=True, repack_veto=True
    )

    # Explicit model-wide veto restores the historical behaviour: the
    # policy-shaped Q4_K FFN slots stay on pack8.
    assert plan.layer_specs[0]["ffn_up"].layout == LAYOUT_Q4_K_PACK8
    assert plan.layer_specs[0]["ssm_alpha"].layout == LAYOUT_DENSE_BF16


def test_repack_eligibility_can_proceed_without_moving_f32_contraction():
    plan = plan_qwen35_gguf_materialization(
        _raw_iq_map(),
        decode_repack=True,
        dense_q4_t16=True,
        repack_veto=False,
    )

    # Per-tensor repack eligibility is granted (T16 selection for the Q4_K
    # shapes the sidecar policy covers) while the F32 alpha/beta contraction
    # stays bound to its own knob (contract_f32_linear defaults to the raw-IQ
    # predicate -> still contracted).
    assert plan.layer_specs[0]["ffn_up"].layout == LAYOUT_GGUF_Q4_K_T16
    assert plan.layer_specs[0]["ssm_alpha"].layout == LAYOUT_DENSE_BF16


def test_f32_contraction_can_be_disabled_without_granting_repack():
    plan = plan_qwen35_gguf_materialization(
        _raw_iq_map(),
        decode_repack=True,
        dense_q4_t16=True,
        contract_f32_linear=False,
        repack_veto=True,
    )

    # Under the explicit model-wide rollback, disabling the contraction does
    # not grant repack: the raw-IQ predicate still vetoes it...
    assert plan.layer_specs[0]["ffn_up"].layout == LAYOUT_Q4_K_PACK8
    # ...while the F32 alpha/beta slots keep their F32 residents.
    assert plan.layer_specs[0]["ssm_alpha"].layout == LAYOUT_DENSE_F32
    assert plan.layer_specs[0]["ssm_alpha"].quant_key == "f32"
