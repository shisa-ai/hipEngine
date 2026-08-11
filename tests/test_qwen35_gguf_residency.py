from __future__ import annotations

from pathlib import Path
from types import MappingProxyType

import pytest

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.memory import DeviceBuffer
from hipengine.core.tensor import Tensor
from hipengine.loading.gguf import GGUFReader, GGUFTensorInfo
from hipengine.loading.materialize import DeviceTensorAllocation
from hipengine.loading.qwen35_gguf_materialize import (
    LAYOUT_GGUF_Q4_K_T16,
    LAYOUT_Q4_K_PACK8,
    Q4_T16_DECODE_TILES,
    Qwen35GGUFDeviceWeight,
    Qwen35GGUFWeightSpec,
    plan_qwen35_gguf_materialization,
)
from hipengine.generation.qwen35_gguf import _LLAMA_COMPAT_MTP_ENV
from hipengine.loading.qwen35_gguf import build_qwen35_gguf_tensor_map
from hipengine.loading.qwen35_gguf_nextn import build_qwen35_gguf_nextn_tensor_map
from hipengine.loading.qwen35_gguf_nextn_materialize import (
    plan_qwen35_gguf_nextn_materialization,
)
from hipengine.loading.qwen35_gguf_residency import (
    Qwen35GGUFResidentWeightRef,
    census_qwen35_gguf_resident_weight_refs,
    census_qwen35_gguf_weight_specs,
)


def _q4_info(name: str = "blk.0.ffn_gate.weight") -> GGUFTensorInfo:
    return GGUFTensorInfo(
        name=name,
        shape=(16, 256),
        ggml_shape=(256, 16),
        ggml_type=12,
        ggml_type_name="Q4_K",
        n_elements=16 * 256,
        nbytes=16 * 144,
        offset=0,
        data_offset=0,
        byte_shape=(16, 144),
    )


def _spec(
    slot_path: str,
    source: GGUFTensorInfo,
    *,
    layout: str,
    allocation_names: tuple[str, ...],
) -> Qwen35GGUFWeightSpec:
    return Qwen35GGUFWeightSpec(
        slot_path=slot_path,
        source=source,
        quant_key=(
            "gguf_q4_k_t16_v1"
            if layout == LAYOUT_GGUF_Q4_K_T16
            else "gguf_q4_k"
        ),
        layout=layout,
        allocation_names=allocation_names,
    )


def _allocation(name: str, ptr: int, nbytes: int) -> DeviceTensorAllocation:
    buffer = DeviceBuffer(ptr=ptr, nbytes=nbytes)
    tensor = Tensor.from_handle(ptr, (nbytes,), DType.INT8, Device("hip", 0))
    return DeviceTensorAllocation(
        name=name,
        source=_q4_info(name),
        buffer=buffer,
        tensor=tensor,
    )


def _weight(
    spec: Qwen35GGUFWeightSpec,
    allocations: dict[str, tuple[int, int]],
) -> Qwen35GGUFDeviceWeight:
    return Qwen35GGUFDeviceWeight(
        spec=spec,
        allocations=MappingProxyType(
            {
                name: _allocation(f"{spec.source.name}.{name}", ptr, nbytes)
                for name, (ptr, nbytes) in allocations.items()
            }
        ),
        backend="hip_gfx1100",
    )


def test_plan_census_exposes_pack8_t16_duplication_and_aliases() -> None:
    source = _q4_info()
    owner = _spec(
        "layers.0.ffn_gate",
        source,
        layout=LAYOUT_Q4_K_PACK8,
        allocation_names=("qweight", "scales", "mins", Q4_T16_DECODE_TILES),
    )
    alias = _spec(
        "aliases.layers.0.ffn_gate",
        source,
        layout=LAYOUT_Q4_K_PACK8,
        allocation_names=owner.allocation_names,
    )

    census = census_qwen35_gguf_weight_specs((owner, alias))

    assert census.logical_tensor_count == 1
    assert census.alias_count == 1
    assert census.source_nbytes == 2_304
    assert census.resident_nbytes == 5_440
    assert census.alternate_layout_nbytes == 2_368
    assert census.logical_weights[0].slot_paths == (
        "layers.0.ffn_gate",
        "aliases.layers.0.ffn_gate",
    )
    assert tuple(
        (allocation.name, allocation.layout, allocation.nbytes, allocation.is_alternate)
        for allocation in census.logical_weights[0].allocations
    ) == (
        ("qweight", LAYOUT_Q4_K_PACK8, 2_048, False),
        ("scales", LAYOUT_Q4_K_PACK8, 512, False),
        ("mins", LAYOUT_Q4_K_PACK8, 512, False),
        (Q4_T16_DECODE_TILES, LAYOUT_GGUF_Q4_K_T16, 2_368, True),
    )


def test_qwen36_27b_plan_census_freezes_current_288_q4_dual_layout_manifest() -> None:
    model = Path("/models/gguf/Qwen3.6-27B-Q4_K_M.gguf")
    if not model.exists():
        pytest.skip(f"local GGUF fixture not found: {model}")
    reader = GGUFReader(model)
    model_map = build_qwen35_gguf_tensor_map(reader.info)
    plan = plan_qwen35_gguf_materialization(
        model_map,
        decode_repack=True,
        dense_q5_t16_ssm_out=True,
        dense_q6_qmicro_planar=True,
    )
    specs = (
        *tuple(plan.root_specs.values()),
        *(spec for layer in plan.layer_specs for spec in layer.values()),
    )

    census = census_qwen35_gguf_weight_specs(specs)
    duplicated_q4 = tuple(
        weight
        for weight in census.logical_weights
        if any(
            allocation.layout == LAYOUT_GGUF_Q4_K_T16
            and allocation.is_alternate
            for allocation in weight.allocations
        )
    )

    assert len(duplicated_q4) == 288
    assert sum(
        allocation.nbytes
        for weight in duplicated_q4
        for allocation in weight.allocations
        if not allocation.is_alternate
    ) == 13_998_489_600
    assert sum(weight.alternate_layout_nbytes for weight in duplicated_q4) == 10_790_502_400
    assert census.alternate_layout_nbytes == 10_790_502_400


def test_qwen36_27b_candidate_census_predicts_zero_q4_alternate_layout_bytes() -> None:
    model = Path("/models/gguf/Qwen3.6-27B-Q4_K_M.gguf")
    if not model.exists():
        pytest.skip(f"local GGUF fixture not found: {model}")
    reader = GGUFReader(model)
    plan = plan_qwen35_gguf_materialization(
        build_qwen35_gguf_tensor_map(reader.info),
        decode_repack=True,
        dense_q4_t16=True,
        dense_q5_t16_ssm_out=True,
        dense_q6_qmicro_planar=True,
    )
    specs = (
        *tuple(plan.root_specs.values()),
        *(spec for layer in plan.layer_specs for spec in layer.values()),
    )
    q4_specs = tuple(
        spec
        for spec in specs
        if spec.source.ggml_type_name == "Q4_K"
        and spec.slot_path != "root.token_embedding"
    )

    census = census_qwen35_gguf_weight_specs(q4_specs)

    assert census.logical_tensor_count == 288
    assert census.resident_nbytes == 10_790_502_400
    assert census.alternate_layout_nbytes == 0
    assert all(
        weight.canonical_layout == LAYOUT_GGUF_Q4_K_T16
        and tuple(allocation.name for allocation in weight.allocations) == ("tiles",)
        for weight in census.logical_weights
    )
    census.assert_single_layout()


def test_qwen36_llama_compat_plan_has_no_q8_raw_or_other_alternate_layouts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = Path("/models/gguf/Qwen3.6-27B-Q4_K_M.gguf")
    if not model.exists():
        pytest.skip(f"local GGUF fixture not found: {model}")
    forbidden = {
        "HIPENGINE_GGUF_Q8_0_RAW_SIDECAR",
        "HIPENGINE_GGUF_DENSE_Q8_DP4A_ALL",
        "HIPENGINE_GGUF_DENSE_Q8_DP4A_F32",
        "HIPENGINE_RESIDENT_MTP_DRAFT_DENSE_Q8_DP4A",
        "HIPENGINE_RESIDENT_MTP_DRAFT_DENSE_Q8_DP4A_STAGES",
    }
    assert forbidden.isdisjoint(_LLAMA_COMPAT_MTP_ENV)
    for name, value in _LLAMA_COMPAT_MTP_ENV.items():
        monkeypatch.setenv(name, value)

    reader = GGUFReader(model)
    target = plan_qwen35_gguf_materialization(
        build_qwen35_gguf_tensor_map(reader.info),
        decode_repack=True,
        dense_q4_t16=True,
        dense_q5_t16_ssm_out=True,
        dense_q6_qmicro_planar=True,
    )
    draft = plan_qwen35_gguf_nextn_materialization(
        build_qwen35_gguf_nextn_tensor_map(reader.info),
        decode_repack=True,
        dense_q4_t16=True,
        dense_q5_t16_ssm_out=True,
        dense_q6_qmicro_planar=True,
    )
    target_specs = (
        *tuple(target.root_specs.values()),
        *(spec for layer in target.layer_specs for spec in layer.values()),
    )

    target_census = census_qwen35_gguf_weight_specs(target_specs)
    draft_census = census_qwen35_gguf_weight_specs(draft.specs)

    assert target_census.alternate_layout_nbytes == 0
    assert draft_census.alternate_layout_nbytes == 0
    target_census.assert_single_layout()
    draft_census.assert_single_layout()


def test_plan_census_accepts_rank2_sole_t16_as_one_allocation_family() -> None:
    spec = _spec(
        "layers.0.ffn_gate",
        _q4_info(),
        layout=LAYOUT_GGUF_Q4_K_T16,
        allocation_names=("tiles",),
    )

    census = census_qwen35_gguf_weight_specs((spec,))

    assert census.resident_nbytes == 2_368
    assert census.alternate_layout_nbytes == 0
    census.assert_single_layout()


def test_runtime_census_deduplicates_aliases_by_exact_physical_range() -> None:
    source = _q4_info()
    spec = _spec(
        "layers.0.ffn_gate",
        source,
        layout=LAYOUT_GGUF_Q4_K_T16,
        allocation_names=("tiles",),
    )
    weight = _weight(spec, {"tiles": (0x1000, 2_368)})

    census = census_qwen35_gguf_resident_weight_refs(
        (
            Qwen35GGUFResidentWeightRef("target", "layers.0.ffn_gate", weight),
            Qwen35GGUFResidentWeightRef("root_shared", "draft.ffn_gate", weight),
        )
    )

    assert census.physical_nbytes == 2_368
    assert len(census.physical_ranges) == 1
    assert census.physical_ranges[0].owners == (
        "draft.ffn_gate",
        "layers.0.ffn_gate",
    )
    assert census.physical_ranges[0].memory_classes == ("root_shared", "target")
    assert census.duplicate_payload_nbytes == 0
    census.assert_single_layout()


def test_runtime_census_rejects_duplicate_copy_and_alternate_layout() -> None:
    source = _q4_info()
    t16_spec = _spec(
        "layers.0.ffn_gate",
        source,
        layout=LAYOUT_GGUF_Q4_K_T16,
        allocation_names=("tiles",),
    )
    copied_spec = _spec(
        "draft.layer.ffn_gate",
        source,
        layout=LAYOUT_GGUF_Q4_K_T16,
        allocation_names=("tiles",),
    )
    dual_spec = _spec(
        "layers.0.ffn_up",
        _q4_info("blk.0.ffn_up.weight"),
        layout=LAYOUT_Q4_K_PACK8,
        allocation_names=("qweight", "scales", "mins", Q4_T16_DECODE_TILES),
    )
    target = _weight(t16_spec, {"tiles": (0x1000, 2_368)})
    copied = _weight(copied_spec, {"tiles": (0x3000, 2_368)})
    dual = _weight(
        dual_spec,
        {
            "qweight": (0x5000, 2_048),
            "scales": (0x6000, 512),
            "mins": (0x7000, 512),
            Q4_T16_DECODE_TILES: (0x8000, 2_368),
        },
    )

    census = census_qwen35_gguf_resident_weight_refs(
        (
            Qwen35GGUFResidentWeightRef("target", "layers.0.ffn_gate", target),
            Qwen35GGUFResidentWeightRef("nextn", "draft.layer.ffn_gate", copied),
            Qwen35GGUFResidentWeightRef("target", "layers.0.ffn_up", dual),
        )
    )

    assert census.duplicate_payload_nbytes == 2_368
    assert census.alternate_layout_nbytes == 2_368
    assert census.duplicate_allocation_roles == (
        ("blk.0.ffn_gate.weight", "tiles", LAYOUT_GGUF_Q4_K_T16),
    )
    with pytest.raises(ValueError, match="single-layout GGUF residency invariant"):
        census.assert_single_layout()


def test_runtime_census_keeps_disjoint_arena_views_distinct() -> None:
    source_a = _q4_info("blk.0.ffn_gate.weight")
    source_b = _q4_info("blk.0.ffn_up.weight")
    spec_a = _spec(
        "layers.0.ffn_gate",
        source_a,
        layout=LAYOUT_GGUF_Q4_K_T16,
        allocation_names=("tiles",),
    )
    spec_b = _spec(
        "layers.0.ffn_up",
        source_b,
        layout=LAYOUT_GGUF_Q4_K_T16,
        allocation_names=("tiles",),
    )

    census = census_qwen35_gguf_resident_weight_refs(
        (
            Qwen35GGUFResidentWeightRef(
                "target", "layers.0.ffn_gate", _weight(spec_a, {"tiles": (0x1000, 2_368)})
            ),
            Qwen35GGUFResidentWeightRef(
                "target", "layers.0.ffn_up", _weight(spec_b, {"tiles": (0x1940, 2_368)})
            ),
        )
    )

    assert len(census.physical_ranges) == 2
    assert census.physical_nbytes == 4_736
    census.assert_single_layout()
