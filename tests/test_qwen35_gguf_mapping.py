from __future__ import annotations

import ctypes
from dataclasses import replace
from pathlib import Path

import pytest

from hipengine.core.dtype import DType
from hipengine.core.hip import get_hip_runtime
from hipengine.quant.gguf import GGMLQuantizationType

from hipengine.loading.gguf import (
    GGUFModelInfo,
    GGUFReader,
    GGUFTensorInfo,
    MissingGGUFTensorError,
)
from hipengine.loading.qwen35_gguf import (
    FULL_ATTENTION,
    LINEAR_ATTENTION,
    build_qwen35_gguf_tensor_map,
    required_qwen35_gguf_tensor_names,
    validate_qwen35_gguf_tensor_map,
)
from hipengine.loading.qwen35_gguf_materialize import (
    LAYOUT_DENSE_BF16,
    LAYOUT_GGUF_Q4_K_QMICRO_T16,
    LAYOUT_GGUF_Q4_K_T16,
    LAYOUT_GGUF_Q5_K_T16,
    LAYOUT_GGUF_Q6_K_T16,
    LAYOUT_GGUF_Q6_K_T16_QMICRO_PLANAR,
    LAYOUT_Q4_K_PACK8,
    materialize_qwen35_gguf_weights,
    plan_qwen35_gguf_materialization,
    plan_qwen35_gguf_weight_spec,
)

MODEL = Path("/models/gguf/Qwen3.5-0.8B-Q4_K_M.gguf")
MOE_MODEL = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
DENSE_UNTIED_MODEL = Path("/models/gguf/Qwen3.6-27B-Q4_K_M.gguf")
QWEN38_DENSE_MODEL = Path("/models/gguf/Qwen3.8-27B-Q4_K_M.gguf")
QWEN38_DENSE_Q4KS_MODEL = Path("/models/gguf/Qwen3.8-27B-Q4_K_S.gguf")
QWEN38_DENSE_Q4KS_SIZE = 16_121_359_328


def _info() -> GGUFModelInfo:
    if not MODEL.exists():
        pytest.skip(f"local GGUF fixture not found: {MODEL}")
    return GGUFReader(MODEL).info


def test_qwen35_gguf_tensor_map_covers_local_inventory() -> None:
    info = _info()
    model_map = build_qwen35_gguf_tensor_map(info)

    assert model_map.validation.passed
    assert model_map.config.architecture == "qwen35"
    assert model_map.config.block_count == 24
    assert model_map.config.hidden_size == 1024
    assert model_map.config.vocab_size == 248320
    assert model_map.config.layer_types.count(FULL_ATTENTION) == 6
    assert model_map.config.layer_types.count(LINEAR_ATTENTION) == 18
    assert model_map.config.lm_head_tensor_name == "token_embd.weight"
    assert len(model_map.layers) == 24
    assert set(model_map.tensor_names) == {tensor.name for tensor in info.tensors}
    assert len(model_map.tensor_names) == len(info.tensors)
    assert set(required_qwen35_gguf_tensor_names(model_map.config)) == set(model_map.tensor_names)

    assert model_map.root("token_embedding").name == "token_embd.weight"
    assert model_map.root("token_embedding").ggml_type_name == "Q6_K"
    assert model_map.root("lm_head").name == "token_embd.weight"
    assert model_map.root("output_norm").shape == (1024,)

    layer0 = model_map.layer(0)
    assert layer0.layer_type == LINEAR_ATTENTION
    assert layer0.tensor("attn_qkv").name == "blk.0.attn_qkv.weight"
    assert layer0.tensor("attn_qkv").ggml_type_name == "Q5_K"
    assert layer0.tensor("attn_gate").ggml_type_name == "Q4_K"
    assert layer0.tensor("ssm_out").ggml_type_name == "Q5_K"
    assert layer0.tensor("ssm_alpha").ggml_type_name == "Q8_0"

    layer3 = model_map.layer(3)
    assert layer3.layer_type == FULL_ATTENTION
    assert layer3.tensor("attn_q").name == "blk.3.attn_q.weight"
    assert layer3.tensor("attn_q").shape == (4096, 1024)
    assert layer3.tensor("attn_k").shape == (512, 1024)
    assert layer3.tensor("attn_v").ggml_type_name == "Q6_K"
    assert layer3.tensor("attn_output").ggml_type_name == "Q4_K"


def test_qwen35_08b_dense_q4_attn_q_policy_uses_single_t16_resident() -> None:
    reader = GGUFReader(MODEL) if MODEL.exists() else None
    if reader is None:
        pytest.skip(f"local GGUF fixture not found: {MODEL}")
    model_map = build_qwen35_gguf_tensor_map(reader.info)

    baseline = plan_qwen35_gguf_materialization(model_map, decode_repack=True)
    candidate = plan_qwen35_gguf_materialization(
        model_map,
        decode_repack=True,
        dense_q4_t16_attn_q_08b=True,
    )
    baseline_by_slot = {spec.slot_path: spec for spec in baseline.specs}
    candidate_attn_q = tuple(
        spec
        for spec in candidate.specs
        if spec.slot_path.endswith(".attn_q")
        and spec.source.ggml_type_name == "Q4_K"
    )

    assert len(candidate_attn_q) == 6
    assert all(spec.source.shape == (4_096, 1_024) for spec in candidate_attn_q)
    assert all(spec.layout == LAYOUT_GGUF_Q4_K_T16 for spec in candidate_attn_q)
    assert all(spec.quant_key == "gguf_q4_k_t16_v1" for spec in candidate_attn_q)
    assert all(spec.allocation_names == ("tiles",) for spec in candidate_attn_q)
    assert all(
        baseline_by_slot[spec.slot_path].layout == LAYOUT_Q4_K_PACK8
        for spec in candidate_attn_q
    )
    assert all(
        spec.layout == LAYOUT_Q4_K_PACK8
        for spec in candidate.specs
        if spec.source.ggml_type_name == "Q4_K"
        and not spec.slot_path.endswith(".attn_q")
        and spec.slot_path != "root.token_embedding"
    )


def test_qwen35_08b_ignores_dense_h5120_compressed_policies() -> None:
    reader = GGUFReader(MODEL) if MODEL.exists() else None
    if reader is None:
        pytest.skip(f"local GGUF fixture not found: {MODEL}")
    model_map = build_qwen35_gguf_tensor_map(reader.info)

    expected = plan_qwen35_gguf_materialization(
        model_map,
        decode_repack=True,
        dense_q4_t16_attn_q_08b=True,
    )
    candidate = plan_qwen35_gguf_materialization(
        model_map,
        decode_repack=True,
        dense_q4_t16=True,
        dense_q4_t16_attn_q_08b=True,
        dense_q5_t16_ssm_out=True,
    )

    assert tuple(
        (spec.slot_path, spec.layout, spec.quant_key, spec.allocation_names)
        for spec in candidate.specs
    ) == tuple(
        (spec.slot_path, spec.layout, spec.quant_key, spec.allocation_names)
        for spec in expected.specs
    )


def test_qwen35_08b_dense_q5_qkv_policy_uses_single_t16_resident() -> None:
    reader = GGUFReader(MODEL) if MODEL.exists() else None
    if reader is None:
        pytest.skip(f"local GGUF fixture not found: {MODEL}")
    model_map = build_qwen35_gguf_tensor_map(reader.info)

    baseline = plan_qwen35_gguf_materialization(model_map, decode_repack=True)
    candidate = plan_qwen35_gguf_materialization(
        model_map,
        decode_repack=True,
        dense_q5_t16_qkv=True,
    )
    baseline_by_slot = {spec.slot_path: spec for spec in baseline.specs}
    candidate_qkv = tuple(
        spec
        for spec in candidate.specs
        if spec.slot_path.endswith(".attn_qkv")
        and spec.source.ggml_type_name == "Q5_K"
    )

    assert len(candidate_qkv) == 18
    assert all(spec.source.shape == (6_144, 1_024) for spec in candidate_qkv)
    assert all(spec.layout == LAYOUT_GGUF_Q5_K_T16 for spec in candidate_qkv)
    assert all(spec.quant_key == "gguf_q5_k_t16_v1" for spec in candidate_qkv)
    assert all(spec.allocation_names == ("tiles",) for spec in candidate_qkv)
    assert all(
        baseline_by_slot[spec.slot_path].layout == LAYOUT_DENSE_BF16
        for spec in candidate_qkv
    )
    assert all(
        spec.layout == LAYOUT_DENSE_BF16
        for spec in candidate.specs
        if spec.slot_path.endswith(".ssm_out")
    )


def test_qwen35_08b_dense_q5_ssm_out_policy_uses_single_t16_resident() -> None:
    reader = GGUFReader(MODEL) if MODEL.exists() else None
    if reader is None:
        pytest.skip(f"local GGUF fixture not found: {MODEL}")
    model_map = build_qwen35_gguf_tensor_map(reader.info)

    baseline = plan_qwen35_gguf_materialization(model_map, decode_repack=True)
    candidate = plan_qwen35_gguf_materialization(
        model_map,
        decode_repack=True,
        dense_q5_t16_ssm_out_08b=True,
    )
    baseline_by_slot = {spec.slot_path: spec for spec in baseline.specs}
    candidate_ssm_out = tuple(
        spec
        for spec in candidate.specs
        if spec.slot_path.endswith(".ssm_out")
        and spec.source.ggml_type_name == "Q5_K"
    )

    assert len(candidate_ssm_out) == 18
    assert all(spec.source.shape == (1_024, 2_048) for spec in candidate_ssm_out)
    assert all(spec.layout == LAYOUT_GGUF_Q5_K_T16 for spec in candidate_ssm_out)
    assert all(spec.quant_key == "gguf_q5_k_t16_v1" for spec in candidate_ssm_out)
    assert all(spec.allocation_names == ("tiles",) for spec in candidate_ssm_out)
    assert all(
        baseline_by_slot[spec.slot_path].layout == LAYOUT_DENSE_BF16
        for spec in candidate_ssm_out
    )
    assert all(
        spec.layout == LAYOUT_DENSE_BF16
        for spec in candidate.specs
        if spec.slot_path.endswith(".attn_qkv")
        and spec.source.ggml_type_name == "Q5_K"
    )


def test_qwen35moe_gguf_tensor_map_covers_local_inventory() -> None:
    if not MOE_MODEL.exists():
        pytest.skip(f"local GGUF fixture not found: {MOE_MODEL}")
    info = GGUFReader(MOE_MODEL).info
    model_map = build_qwen35_gguf_tensor_map(info)

    assert model_map.validation.passed
    assert model_map.config.architecture == "qwen35moe"
    assert model_map.config.block_count == 40
    assert model_map.config.hidden_size == 2048
    assert model_map.config.vocab_size == 248320
    assert model_map.config.expert_count == 256
    assert model_map.config.expert_used_count == 8
    assert model_map.config.expert_feed_forward_length == 512
    assert model_map.config.expert_shared_feed_forward_length == 512
    assert model_map.config.layer_types.count(FULL_ATTENTION) == 10
    assert model_map.config.layer_types.count(LINEAR_ATTENTION) == 30
    assert model_map.config.lm_head_tensor_name == "output.weight"
    assert len(model_map.layers) == 40
    extra_block_ids = tuple(
        sorted(
            {
                int(parts[1])
                for tensor in info.tensors
                if (parts := tensor.name.split(".", 2))[0] == "blk"
                and parts[1].isdigit()
                and int(parts[1]) >= model_map.config.block_count
            }
        )
    )
    assert model_map.config.ignored_block_ids == extra_block_ids
    ar_tensor_names = {
        tensor.name
        for tensor in info.tensors
        if not any(
            tensor.name.startswith(f"blk.{block_id}.") for block_id in extra_block_ids
        )
    }
    assert set(model_map.tensor_names) == ar_tensor_names
    assert len(model_map.tensor_names) == len(ar_tensor_names)
    assert set(required_qwen35_gguf_tensor_names(model_map.config)) == set(model_map.tensor_names)

    assert model_map.root("token_embedding").name == "token_embd.weight"
    assert model_map.root("lm_head").name == "output.weight"
    assert model_map.root("lm_head").ggml_type_name == "Q6_K"

    layer0 = model_map.layer(0)
    assert layer0.layer_type == LINEAR_ATTENTION
    assert layer0.tensor("ffn_gate_inp").shape == (256, 2048)
    assert layer0.tensor("ffn_gate_exps").shape == (256, 512, 2048)
    assert layer0.tensor("ffn_down_exps").shape == (256, 2048, 512)
    assert layer0.tensor("ffn_gate_shexp").shape == (512, 2048)


def test_qwen36_moe_qmicro_plan_keeps_unmeasured_root_head_legacy() -> None:
    if not MOE_MODEL.exists():
        pytest.skip(f"local GGUF fixture not found: {MOE_MODEL}")
    model_map = build_qwen35_gguf_tensor_map(GGUFReader(MOE_MODEL).info)

    plan = plan_qwen35_gguf_materialization(
        model_map,
        decode_repack=True,
        dense_q6_qmicro_planar=True,
    )

    assert plan.root_specs["lm_head"].source.shape == (248_320, 2_048)
    assert plan.root_specs["lm_head"].layout == LAYOUT_GGUF_Q6_K_T16
    assert plan.root_specs["lm_head"].quant_key == "gguf_q6_k_t16_v1"


def test_qwen36_dense_untied_gguf_tensor_map_uses_output_weight() -> None:
    if not DENSE_UNTIED_MODEL.exists():
        pytest.skip(f"local GGUF fixture not found: {DENSE_UNTIED_MODEL}")
    info = GGUFReader(DENSE_UNTIED_MODEL).info

    model_map = build_qwen35_gguf_tensor_map(info)

    assert model_map.validation.passed
    assert model_map.config.architecture == "qwen35"
    assert not model_map.config.is_moe
    assert model_map.config.block_count == 64
    has_trailing_nextn = any(
        tensor.name.startswith("blk.64.nextn.") for tensor in info.tensors
    )
    assert model_map.config.ignored_block_ids == ((64,) if has_trailing_nextn else ())
    assert model_map.config.lm_head_tensor_name == "output.weight"
    assert model_map.root("token_embedding").name == "token_embd.weight"
    assert model_map.root("lm_head").name == "output.weight"
    assert model_map.root("lm_head").ggml_type_name == "Q6_K"
    assert set(model_map.tensor_names) == {
        tensor.name for tensor in info.tensors if not tensor.name.startswith("blk.64.")
    }


def test_qwen36_dense_decode_repack_replaces_wide_rank2_q6_and_root_head() -> None:
    if not DENSE_UNTIED_MODEL.exists():
        pytest.skip(f"local GGUF fixture not found: {DENSE_UNTIED_MODEL}")
    reader = GGUFReader(DENSE_UNTIED_MODEL)
    model_map = build_qwen35_gguf_tensor_map(reader.info)

    legacy = plan_qwen35_gguf_materialization(model_map, decode_repack=False)
    plan = plan_qwen35_gguf_materialization(model_map, decode_repack=True)
    qmicro_plan = plan_qwen35_gguf_materialization(
        model_map,
        decode_repack=True,
        dense_q6_qmicro_planar=True,
    )
    optimized_plan = plan_qwen35_gguf_materialization(
        model_map,
        decode_repack=True,
        dense_q4_t16=True,
        dense_q5_t16_ssm_out=True,
        dense_q6_qmicro_planar=True,
    )
    legacy_by_slot = {spec.slot_path: spec for spec in legacy.specs}
    plan_by_slot = {spec.slot_path: spec for spec in plan.specs}
    qmicro_by_slot = {spec.slot_path: spec for spec in qmicro_plan.specs}
    optimized_by_slot = {spec.slot_path: spec for spec in optimized_plan.specs}

    q5_ssm_out_slots = {
        spec.slot_path
        for spec in optimized_plan.specs
        if spec.layout == LAYOUT_GGUF_Q5_K_T16
    }
    assert len(q5_ssm_out_slots) == 48
    assert all(slot.endswith(".ssm_out") for slot in q5_ssm_out_slots)
    for slot in q5_ssm_out_slots:
        optimized = optimized_by_slot[slot]
        assert optimized.source.shape == (5_120, 6_144)
        assert optimized.quant_key == "gguf_q5_k_t16_v1"
        assert optimized.allocation_names == ("tiles",)
        assert plan_by_slot[slot].layout == LAYOUT_DENSE_BF16
        assert plan_by_slot[slot].allocation_names == ("raw",)

    wide_slots = {
        spec.slot_path
        for spec in plan.specs
        if spec.slot_path.startswith("layers.")
        and spec.source.ggml_type_name == "Q6_K"
        and spec.layout == LAYOUT_GGUF_Q6_K_T16
    }
    assert sum(slot.endswith(".ffn_down") for slot in wide_slots) == 32
    assert sum(slot.endswith(".attn_qkv") for slot in wide_slots) == 24
    assert len(wide_slots) == 56
    for slot in wide_slots:
        spec = plan_by_slot[slot]
        assert spec.quant_key == "gguf_q6_k_t16_v1"
        assert spec.allocation_names == ("tiles",)
        assert legacy_by_slot[slot].layout == LAYOUT_DENSE_BF16
        assert legacy_by_slot[slot].allocation_names == ("raw",)
        assert qmicro_by_slot[slot].layout == LAYOUT_GGUF_Q6_K_T16_QMICRO_PLANAR
        assert qmicro_by_slot[slot].quant_key == "gguf_q6_k_t16_qmicro_planar_v1"
        assert qmicro_by_slot[slot].allocation_names == ("tiles",)

    assert sum(
        spec.layout == LAYOUT_GGUF_Q6_K_T16_QMICRO_PLANAR
        for spec in qmicro_plan.specs
    ) == 65
    assert plan.root_specs["lm_head"].layout == LAYOUT_GGUF_Q6_K_T16
    assert qmicro_plan.root_specs["lm_head"].layout == LAYOUT_GGUF_Q6_K_T16_QMICRO_PLANAR
    assert (
        qmicro_plan.root_specs["lm_head"].quant_key
        == "gguf_q6_k_t16_qmicro_planar_v1"
    )
    assert qmicro_plan.root_specs["lm_head"].allocation_names == ("tiles",)

    narrow_v = [
        spec
        for spec in plan.specs
        if spec.slot_path.endswith(".attn_v")
        and spec.source.ggml_type_name == "Q6_K"
    ]
    qmicro_narrow_v = [qmicro_by_slot[spec.slot_path] for spec in narrow_v]
    assert len(narrow_v) == 8
    assert all(spec.layout == LAYOUT_DENSE_BF16 for spec in narrow_v)
    assert all(spec.quant_key == "gguf_q6_k" for spec in narrow_v)
    assert all(spec.allocation_names == ("raw",) for spec in narrow_v)
    assert all(
        spec.layout == LAYOUT_GGUF_Q6_K_T16_QMICRO_PLANAR
        for spec in qmicro_narrow_v
    )
    assert all(
        spec.quant_key == "gguf_q6_k_t16_qmicro_planar_v1"
        for spec in qmicro_narrow_v
    )
    assert all(spec.allocation_names == ("tiles",) for spec in qmicro_narrow_v)

    q4_t16_owners = [
        spec
        for spec in optimized_plan.specs
        if spec.source.ggml_type_name == "Q4_K"
        and spec.layout == LAYOUT_GGUF_Q4_K_T16
    ]
    expected_q4_owners = {
        "attn_gate": 48,
        "attn_k": 16,
        "attn_output": 16,
        "attn_q": 16,
        "attn_qkv": 24,
        "attn_v": 8,
        "ffn_down": 32,
        "ffn_gate": 64,
        "ffn_up": 64,
    }
    assert len(q4_t16_owners) == sum(expected_q4_owners.values()) == 288
    for role, count in expected_q4_owners.items():
        role_specs = [
            spec
            for spec in q4_t16_owners
            if spec.slot_path.endswith(f".{role}")
        ]
        assert len(role_specs) == count
        for spec in role_specs:
            assert spec.quant_key == "gguf_q4_k_t16_v1"
            assert spec.allocation_names == ("tiles",)
            current = plan_by_slot[spec.slot_path]
            assert current.layout == LAYOUT_Q4_K_PACK8
            assert any(
                name.startswith("decode_tiles")
                for name in current.allocation_names
            )
            assert all(
                not name.startswith("decode_tiles")
                for name in legacy_by_slot[spec.slot_path].allocation_names
            )
    assert all(
        not any(name.startswith("decode_tiles") for name in spec.allocation_names)
        for spec in plan.specs
        if spec.source.ggml_type_name == "Q4_K"
        and spec.slot_path.endswith(".token_embedding")
    )


def test_qwen36_dense_q4_materializes_sole_t16_owner() -> None:
    if not DENSE_UNTIED_MODEL.exists():
        pytest.skip(f"local GGUF fixture not found: {DENSE_UNTIED_MODEL}")
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        pytest.skip("HIP runtime is not available")
    runtime = get_hip_runtime()
    resident = materialize_qwen35_gguf_weights(
        DENSE_UNTIED_MODEL,
        selected_slots=("layers.0.attn_gate",),
        decode_repack=True,
        backend="hip_gfx1100",
        runtime=runtime,
    )
    try:
        weight = resident.layer(0).weight("attn_gate")
        assert weight.spec.layout == LAYOUT_GGUF_Q4_K_T16
        assert weight.spec.quant_key == "gguf_q4_k_t16_v1"
        assert tuple(weight.allocations) == ("tiles",)
        assert weight.allocation("tiles").tensor.dtype == DType.INT8
        assert weight.allocation("tiles").buffer.nbytes == 18_186_240
    finally:
        resident.free(runtime=runtime)


def test_qwen38_dense_q4_plan_uses_one_t16_payload_for_every_rank2_owner() -> None:
    if not QWEN38_DENSE_MODEL.exists():
        pytest.skip(f"local GGUF fixture not found: {QWEN38_DENSE_MODEL}")
    model_map = build_qwen35_gguf_tensor_map(GGUFReader(QWEN38_DENSE_MODEL).info)
    plan = plan_qwen35_gguf_materialization(
        model_map,
        decode_repack=True,
        dense_q4_t16=True,
    )
    q4_specs = tuple(
        spec
        for spec in plan.specs
        if spec.slot_path.startswith("layers.")
        and spec.source.ggml_type_name == "Q4_K"
        and len(spec.source.shape) == 2
    )

    assert len(q4_specs) == 288
    assert all(spec.layout == LAYOUT_GGUF_Q4_K_T16 for spec in q4_specs)
    assert all(spec.quant_key == "gguf_q4_k_t16_v1" for spec in q4_specs)
    assert all(spec.allocation_names == ("tiles",) for spec in q4_specs)
    assert all(
        "pack8" not in name and not name.startswith("decode_")
        for spec in q4_specs
        for name in spec.allocation_names
    )
    token_embedding = plan.root_specs["token_embedding"]
    assert token_embedding.layout == "raw_gguf"
    assert token_embedding.allocation_names == ("raw",)


def test_qwen38_dense_gate_up_plan_uses_sole_qmicro_payload() -> None:
    if not QWEN38_DENSE_MODEL.exists():
        pytest.skip(f"local GGUF fixture not found: {QWEN38_DENSE_MODEL}")
    model_map = build_qwen35_gguf_tensor_map(GGUFReader(QWEN38_DENSE_MODEL).info)
    plan = plan_qwen35_gguf_materialization(
        model_map,
        decode_repack=True,
        dense_q4_t16=True,
        dense_q4_qmicro_t16_gate_up=True,
    )
    qmicro_specs = tuple(
        spec
        for spec in plan.specs
        if spec.layout == LAYOUT_GGUF_Q4_K_QMICRO_T16
    )
    assert len(qmicro_specs) == 128
    assert all(
        spec.slot_path.endswith((".ffn_gate", ".ffn_up"))
        for spec in qmicro_specs
    )
    assert all(
        spec.quant_key == "gguf_q4_k_qmicro_t16_v1"
        and spec.allocation_names == ("tiles",)
        for spec in qmicro_specs
    )
    assert all(
        spec.layout == LAYOUT_GGUF_Q4_K_T16
        for spec in plan.specs
        if spec.slot_path.startswith("layers.")
        and spec.source.ggml_type_name == "Q4_K"
        and len(spec.source.shape) == 2
        and not spec.slot_path.endswith((".ffn_gate", ".ffn_up"))
    )


def test_qwen38_dense_q4km_materializes_standard_t16_owner_on_gfx1151() -> None:
    if not QWEN38_DENSE_MODEL.exists():
        pytest.skip(f"local GGUF fixture not found: {QWEN38_DENSE_MODEL}")
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        pytest.skip("HIP runtime is not available")
    runtime = get_hip_runtime()
    resident = materialize_qwen35_gguf_weights(
        QWEN38_DENSE_MODEL,
        selected_slots=("layers.0.attn_gate", "layers.0.ffn_gate"),
        decode_repack=True,
        backend="hip_gfx1151",
        runtime=runtime,
    )
    try:
        weight = resident.layer(0).weight("attn_gate")
        assert weight.spec.layout == LAYOUT_GGUF_Q4_K_T16
        assert weight.spec.quant_key == "gguf_q4_k_t16_v1"
        assert tuple(weight.allocations) == ("tiles",)
        assert weight.allocation("tiles").tensor.dtype == DType.INT8
        assert weight.allocation("tiles").buffer.nbytes == 18_186_240

        assert resident.file_type_name == "MOSTLY_Q4_K_M"
        gate = resident.layer(0).weight("ffn_gate")
        assert gate.spec.layout == LAYOUT_GGUF_Q4_K_T16
        assert gate.spec.quant_key == "gguf_q4_k_t16_v1"
        assert tuple(gate.allocations) == ("tiles",)
        assert gate.allocation("tiles").tensor.dtype == DType.INT8
        assert gate.allocation("tiles").buffer.nbytes == 51_527_680
    finally:
        resident.free(runtime=runtime)


def test_qwen38_dense_q4ks_plan_uses_compact_q5_for_every_rank2_owner() -> None:
    if (
        not QWEN38_DENSE_Q4KS_MODEL.exists()
        or QWEN38_DENSE_Q4KS_MODEL.stat().st_size != QWEN38_DENSE_Q4KS_SIZE
    ):
        pytest.skip(f"complete local GGUF fixture not found: {QWEN38_DENSE_Q4KS_MODEL}")
    info = GGUFReader(QWEN38_DENSE_Q4KS_MODEL).info
    model_map = build_qwen35_gguf_tensor_map(info)
    plan = plan_qwen35_gguf_materialization(
        model_map,
        decode_repack=True,
        dense_q4_t16=True,
        dense_q4_qmicro_t16_gate_up=True,
        dense_q5_t16_ssm_out=True,
        dense_q5_t16_h5120=True,
        dense_q6_qmicro_planar=True,
    )
    q5_specs = tuple(
        spec
        for spec in plan.specs
        if spec.source.ggml_type_name == "Q5_K" and len(spec.source.shape) == 2
    )
    h5120_specs = tuple(
        spec for spec in q5_specs if not spec.slot_path.endswith(".ssm_out")
    )

    assert info.file_type_name == "MOSTLY_Q4_K_S"
    assert len(q5_specs) == 60
    assert len(h5120_specs) == 12
    assert sum(spec.slot_path.endswith(".ffn_down") for spec in h5120_specs) == 8
    assert sum(spec.slot_path.endswith(".attn_qkv") for spec in h5120_specs) == 3
    assert sum(spec.slot_path.endswith(".attn_v") for spec in h5120_specs) == 1
    assert all(spec.layout == LAYOUT_GGUF_Q5_K_T16 for spec in q5_specs)
    assert all(spec.quant_key == "gguf_q5_k_t16_v1" for spec in q5_specs)
    assert all(spec.allocation_names == ("tiles",) for spec in q5_specs)


def test_qwen38_dense_q4ks_materializes_h5120_q5_t16_on_gfx1151() -> None:
    if (
        not QWEN38_DENSE_Q4KS_MODEL.exists()
        or QWEN38_DENSE_Q4KS_MODEL.stat().st_size != QWEN38_DENSE_Q4KS_SIZE
    ):
        pytest.skip(f"complete local GGUF fixture not found: {QWEN38_DENSE_Q4KS_MODEL}")
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        pytest.skip("HIP runtime is not available")
    runtime = get_hip_runtime()
    resident = materialize_qwen35_gguf_weights(
        QWEN38_DENSE_Q4KS_MODEL,
        selected_slots=(
            "layers.0.ffn_gate",
            "layers.0.ffn_down",
            "layers.0.attn_qkv",
            "layers.3.attn_v",
        ),
        decode_repack=True,
        backend="hip_gfx1151",
        runtime=runtime,
    )
    try:
        expected_bytes = (62_668_800, 36_864_000, 3_686_400)
        weights = (
            resident.layer(0).weight("ffn_down"),
            resident.layer(0).weight("attn_qkv"),
            resident.layer(3).weight("attn_v"),
        )
        assert all(weight.spec.layout == LAYOUT_GGUF_Q5_K_T16 for weight in weights)
        assert all(weight.spec.quant_key == "gguf_q5_k_t16_v1" for weight in weights)
        assert all(tuple(weight.allocations) == ("tiles",) for weight in weights)
        assert all(
            weight.allocation("tiles").tensor.dtype == DType.INT8
            for weight in weights
        )
        assert tuple(
            weight.allocation("tiles").buffer.nbytes for weight in weights
        ) == expected_bytes

        assert resident.file_type_name == "MOSTLY_Q4_K_S"
        gate = resident.layer(0).weight("ffn_gate")
        assert gate.spec.layout == LAYOUT_GGUF_Q4_K_QMICRO_T16
        assert gate.spec.quant_key == "gguf_q4_k_qmicro_t16_v1"
        assert tuple(gate.allocations) == ("tiles",)
        assert gate.allocation("tiles").tensor.dtype == DType.INT8
        assert gate.allocation("tiles").buffer.nbytes == 50_135_040
    finally:
        resident.free(runtime=runtime)


def test_qwen38_dense_q4ks_q5_t16_planning_is_role_and_shape_bounded() -> None:
    def source(out_features: int, in_features: int) -> GGUFTensorInfo:
        block_bytes = 176
        return GGUFTensorInfo(
            name=f"weight.{out_features}.{in_features}",
            shape=(out_features, in_features),
            ggml_shape=(in_features, out_features),
            ggml_type=int(GGMLQuantizationType.Q5_K),
            ggml_type_name="Q5_K",
            n_elements=out_features * in_features,
            nbytes=out_features * (in_features // 256) * block_bytes,
            offset=0,
            data_offset=0,
            byte_shape=(out_features, (in_features // 256) * block_bytes),
        )

    admitted = (
        ("layers.0.ffn_down", (5_120, 17_408)),
        ("layers.0.attn_qkv", (10_240, 5_120)),
        ("layers.3.attn_v", (1_024, 5_120)),
    )
    for slot_path, shape in admitted:
        baseline = plan_qwen35_gguf_weight_spec(
            slot_path,
            source(*shape),
            decode_repack=True,
        )
        candidate = plan_qwen35_gguf_weight_spec(
            slot_path,
            source(*shape),
            decode_repack=True,
            dense_q5_t16_h5120=True,
        )
        assert baseline.layout == LAYOUT_DENSE_BF16
        assert candidate.layout == LAYOUT_GGUF_Q5_K_T16
        assert candidate.quant_key == "gguf_q5_k_t16_v1"
        assert candidate.allocation_names == ("tiles",)

    for slot_path, shape in (
        ("layers.0.ffn_gate", (17_408, 5_120)),
        ("layers.0.ffn_down", (5_120, 3_584)),
        ("root.lm_head", (248_320, 5_120)),
    ):
        spec = plan_qwen35_gguf_weight_spec(
            slot_path,
            source(*shape),
            decode_repack=True,
            dense_q5_t16_h5120=True,
        )
        assert spec.layout == LAYOUT_DENSE_BF16


def test_qwen38_dense_q5_ssm_out_plan_uses_one_t16_payload_per_owner() -> None:
    if not QWEN38_DENSE_MODEL.exists():
        pytest.skip(f"local GGUF fixture not found: {QWEN38_DENSE_MODEL}")
    model_map = build_qwen35_gguf_tensor_map(GGUFReader(QWEN38_DENSE_MODEL).info)
    plan = plan_qwen35_gguf_materialization(
        model_map,
        decode_repack=True,
        dense_q5_t16_ssm_out=True,
    )
    q5_specs = tuple(
        spec
        for spec in plan.specs
        if spec.source.ggml_type_name == "Q5_K"
        and spec.slot_path.endswith(".ssm_out")
    )

    assert len(q5_specs) == 48
    assert all(spec.source.shape == (5_120, 6_144) for spec in q5_specs)
    assert all(spec.layout == LAYOUT_GGUF_Q5_K_T16 for spec in q5_specs)
    assert all(spec.quant_key == "gguf_q5_k_t16_v1" for spec in q5_specs)
    assert all(spec.allocation_names == ("tiles",) for spec in q5_specs)


def test_qwen38_dense_q5_ssm_out_plan_can_add_raw_mmq_sidecars() -> None:
    if not QWEN38_DENSE_MODEL.exists():
        pytest.skip(f"local GGUF fixture not found: {QWEN38_DENSE_MODEL}")
    model_map = build_qwen35_gguf_tensor_map(GGUFReader(QWEN38_DENSE_MODEL).info)
    plan = plan_qwen35_gguf_materialization(
        model_map,
        decode_repack=True,
        dense_q5_t16_ssm_out=True,
        dense_q5_raw_mmq_ssm_out=True,
    )
    q5_specs = tuple(
        spec
        for spec in plan.specs
        if spec.source.ggml_type_name == "Q5_K"
        and spec.slot_path.endswith(".ssm_out")
    )

    assert len(q5_specs) == 48
    assert all(spec.layout == LAYOUT_GGUF_Q5_K_T16 for spec in q5_specs)
    assert all(spec.allocation_names == ("tiles", "raw") for spec in q5_specs)


def test_qwen38_dense_q5_ssm_out_materializes_opt_in_raw_mmq_sidecar(
    monkeypatch,
) -> None:
    if not QWEN38_DENSE_MODEL.exists():
        pytest.skip(f"local GGUF fixture not found: {QWEN38_DENSE_MODEL}")
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        pytest.skip("HIP runtime is not available")
    monkeypatch.setenv("HIPENGINE_GGUF_C8_Q5_RAW_MMQ", "1")
    runtime = get_hip_runtime()
    resident = materialize_qwen35_gguf_weights(
        QWEN38_DENSE_MODEL,
        selected_slots=("layers.0.ssm_out",),
        decode_repack=True,
        backend="hip_gfx1100",
        runtime=runtime,
    )
    try:
        weight = resident.layer(0).weight("ssm_out")
        assert weight.spec.layout == LAYOUT_GGUF_Q5_K_T16
        assert weight.spec.allocation_names == ("tiles", "raw")
        assert tuple(weight.allocations) == ("tiles", "raw")
        assert weight.allocation("raw").buffer.nbytes == 21_626_880
    finally:
        resident.free(runtime=runtime)


def test_qwen38_dense_q5_ssm_out_materializes_sole_t16_on_gfx1151() -> None:
    if not QWEN38_DENSE_MODEL.exists():
        pytest.skip(f"local GGUF fixture not found: {QWEN38_DENSE_MODEL}")
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        pytest.skip("HIP runtime is not available")
    runtime = get_hip_runtime()
    resident = materialize_qwen35_gguf_weights(
        QWEN38_DENSE_MODEL,
        selected_slots=("layers.0.ssm_out",),
        decode_repack=True,
        backend="hip_gfx1151",
        runtime=runtime,
    )
    try:
        weight = resident.layer(0).weight("ssm_out")
        assert weight.spec.layout == LAYOUT_GGUF_Q5_K_T16
        assert weight.spec.quant_key == "gguf_q5_k_t16_v1"
        assert tuple(weight.allocations) == ("tiles",)
        assert weight.allocation("tiles").tensor.dtype == DType.INT8
        assert weight.allocation("tiles").buffer.nbytes == 22_118_400
    finally:
        resident.free(runtime=runtime)


def test_qwen38_dense_q6_plan_keeps_decode_blocked_qkv_standard() -> None:
    if not QWEN38_DENSE_MODEL.exists():
        pytest.skip(f"local GGUF fixture not found: {QWEN38_DENSE_MODEL}")
    model_map = build_qwen35_gguf_tensor_map(GGUFReader(QWEN38_DENSE_MODEL).info)
    plan = plan_qwen35_gguf_materialization(
        model_map,
        decode_repack=True,
        dense_q6_qmicro_planar=True,
        dense_q6_qmicro_planar_excluded_slots=("attn_qkv",),
    )
    q6_specs = tuple(
        spec
        for spec in plan.specs
        if spec.slot_path.startswith("layers.")
        and spec.source.ggml_type_name == "Q6_K"
        and len(spec.source.shape) == 2
    )

    assert len(q6_specs) == 64
    assert sum(spec.slot_path.endswith(".ffn_down") for spec in q6_specs) == 32
    assert sum(spec.slot_path.endswith(".attn_qkv") for spec in q6_specs) == 24
    assert sum(spec.slot_path.endswith(".attn_v") for spec in q6_specs) == 8
    qkv_specs = tuple(
        spec for spec in q6_specs if spec.slot_path.endswith(".attn_qkv")
    )
    planar_specs = tuple(
        spec for spec in q6_specs if not spec.slot_path.endswith(".attn_qkv")
    )
    assert len(qkv_specs) == 24
    assert all(spec.layout == LAYOUT_GGUF_Q6_K_T16 for spec in qkv_specs)
    assert all(spec.quant_key == "gguf_q6_k_t16_v1" for spec in qkv_specs)
    assert len(planar_specs) == 40
    assert all(
        spec.layout == LAYOUT_GGUF_Q6_K_T16_QMICRO_PLANAR
        for spec in planar_specs
    )
    assert all(
        spec.quant_key == "gguf_q6_k_t16_qmicro_planar_v1"
        for spec in planar_specs
    )
    assert all(spec.allocation_names == ("tiles",) for spec in q6_specs)
    lm_head = plan.root_specs["lm_head"]
    assert lm_head.layout == LAYOUT_GGUF_Q6_K_T16_QMICRO_PLANAR
    assert lm_head.quant_key == "gguf_q6_k_t16_qmicro_planar_v1"
    assert lm_head.allocation_names == ("tiles",)


def test_qwen38_dense_q6_materializes_role_qualified_layouts_on_gfx1151() -> None:
    if not QWEN38_DENSE_MODEL.exists():
        pytest.skip(f"local GGUF fixture not found: {QWEN38_DENSE_MODEL}")
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        pytest.skip("HIP runtime is not available")
    runtime = get_hip_runtime()
    resident = materialize_qwen35_gguf_weights(
        QWEN38_DENSE_MODEL,
        selected_slots=(
            "layers.0.ffn_down",
            "layers.0.attn_qkv",
            "layers.3.attn_v",
        ),
        decode_repack=True,
        backend="hip_gfx1151",
        runtime=runtime,
    )
    try:
        weights = (
            resident.layer(0).weight("ffn_down"),
            resident.layer(0).weight("attn_qkv"),
            resident.layer(3).weight("attn_v"),
        )
        assert weights[0].spec.layout == LAYOUT_GGUF_Q6_K_T16_QMICRO_PLANAR
        assert weights[0].spec.quant_key == "gguf_q6_k_t16_qmicro_planar_v1"
        assert weights[1].spec.layout == LAYOUT_GGUF_Q6_K_T16
        assert weights[1].spec.quant_key == "gguf_q6_k_t16_v1"
        assert weights[2].spec.layout == LAYOUT_GGUF_Q6_K_T16_QMICRO_PLANAR
        assert weights[2].spec.quant_key == "gguf_q6_k_t16_qmicro_planar_v1"
        assert all(tuple(weight.allocations) == ("tiles",) for weight in weights)
        assert all(
            weight.allocation("tiles").tensor.dtype == DType.INT8
            for weight in weights
        )
    finally:
        resident.free(runtime=runtime)


def test_qwen36_dense_attn_v_materializes_sole_qmicro_owner() -> None:
    if not DENSE_UNTIED_MODEL.exists():
        pytest.skip(f"local GGUF fixture not found: {DENSE_UNTIED_MODEL}")
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        pytest.skip("HIP runtime is not available")
    runtime = get_hip_runtime()
    resident = materialize_qwen35_gguf_weights(
        DENSE_UNTIED_MODEL,
        selected_slots=("layers.3.attn_v",),
        decode_repack=True,
        backend="hip_gfx1100",
        runtime=runtime,
    )
    try:
        attn_v = resident.layer(3).weight("attn_v")
        assert attn_v.spec.layout == LAYOUT_GGUF_Q6_K_T16_QMICRO_PLANAR
        assert attn_v.spec.quant_key == "gguf_q6_k_t16_qmicro_planar_v1"
        assert tuple(attn_v.allocations) == ("tiles",)
        assert attn_v.allocation("tiles").tensor.dtype == DType.INT8
        assert attn_v.allocation("tiles").buffer.nbytes == 4_300_800
    finally:
        resident.free(runtime=runtime)


def test_qwen35_gguf_tensor_map_reports_missing_tensor() -> None:
    info = _info()
    broken = _without_tensor(info, "blk.0.attn_qkv.weight")

    validation = validate_qwen35_gguf_tensor_map(broken)
    assert "blk.0.attn_qkv.weight" in validation.missing
    assert not validation.passed
    with pytest.raises(MissingGGUFTensorError, match="blk.0.attn_qkv.weight"):
        build_qwen35_gguf_tensor_map(broken)


def test_qwen35_gguf_tensor_map_reports_unexpected_tensor() -> None:
    info = _info()
    extra = replace(info.tensors[0], name="blk.0.unexpected.weight")
    broken = replace(info, tensors=info.tensors + (extra,))

    validation = validate_qwen35_gguf_tensor_map(broken)
    assert "blk.0.unexpected.weight" in validation.unexpected
    assert not validation.passed
    with pytest.raises(MissingGGUFTensorError, match="unexpected tensors"):
        build_qwen35_gguf_tensor_map(broken)


def test_qwen35_gguf_tensor_map_reports_shape_error() -> None:
    info = _info()
    broken = _replace_tensor(info, "blk.0.ssm_norm.weight", shape=(129,))

    validation = validate_qwen35_gguf_tensor_map(broken)
    assert any("blk.0.ssm_norm.weight" in item for item in validation.shape_errors)
    assert not validation.passed
    with pytest.raises(MissingGGUFTensorError, match="shape errors"):
        build_qwen35_gguf_tensor_map(broken)


def _without_tensor(info: GGUFModelInfo, name: str) -> GGUFModelInfo:
    return replace(info, tensors=tuple(tensor for tensor in info.tensors if tensor.name != name))


def _replace_tensor(info: GGUFModelInfo, name: str, **updates) -> GGUFModelInfo:
    tensors = tuple(replace(tensor, **updates) if tensor.name == name else tensor for tensor in info.tensors)
    return replace(info, tensors=tensors)
