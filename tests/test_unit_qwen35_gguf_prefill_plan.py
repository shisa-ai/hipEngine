"""Pure prefill scratch layout plan: formulas, layout invariants, purity.

The plan is the single source of truth for prefill scratch geometry, owner
order, views and liveness aliasing. These tests pin the arithmetic with explicit
numbers (not a second copy of the expressions), pin the placement algorithms on
synthetic lifetime sets, and guard the module's dependency boundary.
"""

from __future__ import annotations

import dataclasses
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from hipengine.core.dtype import DType
from hipengine.core.memory import DeviceBuffer
from hipengine.runtime import qwen35_gguf_runner as gguf_runner
from hipengine.runtime import qwen35_gguf_prefill_plan as prefill_plan
from hipengine.runtime.qwen35_gguf_prefill_plan import (
    _GGUF_PREFILL_SCRATCH_COLOR_BYTES,
    _GGUF_PREFILL_SCRATCH_COMPACT_DISABLED_FIELDS,
    _GGUF_PREFILL_SCRATCH_DENSE_DISABLED_FIELDS,
    _GGUF_PREFILL_SCRATCH_DENSE_LIFETIMES,
    _GGUF_PREFILL_SCRATCH_PEER_DISABLED_FIELDS,
    _plan_prefill_scratch_liveness_arena,
    _plan_prefill_scratch_liveness_owner_slots,
    _prefill_scratch_field_subranges,
    build_prefill_scratch_plan,
)
from tests.test_unit_qwen35_gguf_prefill_scratch_liveness import (
    _clear_diagnostic_environment,
    _fake_runner,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DENSE_LIVENESS_DISABLED = frozenset(
    (
        *_GGUF_PREFILL_SCRATCH_COMPACT_DISABLED_FIELDS,
        *_GGUF_PREFILL_SCRATCH_DENSE_DISABLED_FIELDS,
    )
)


def _dense_cfg() -> SimpleNamespace:
    return SimpleNamespace(
        is_moe=False,
        expert_used_count=0,
        expert_count=0,
        expert_shared_feed_forward_length=0,
        ssm_inner_size=48,
        ssm_conv_kernel=4,
        ssm_time_step_rank=6,
        ssm_state_size=8,
        head_count_kv=2,
        key_length=32,
        head_count=4,
        rope_dimension_count=8,
    )


def _moe_cfg() -> SimpleNamespace:
    return SimpleNamespace(
        is_moe=True,
        expert_used_count=2,
        expert_count=4,
        expert_shared_feed_forward_length=8,
        ssm_inner_size=48,
        ssm_conv_kernel=4,
        ssm_time_step_rank=6,
        ssm_state_size=8,
        head_count_kv=2,
        key_length=32,
        head_count=4,
        rope_dimension_count=8,
    )


def _build(cfg, **overrides):
    kwargs = dict(
        hidden_size=16,
        ffn_size=8,
        q_width=32,
        kv_width=16,
        linear_qkv_width=64,
        ssm_value_dim=4,
        fp16_recurrent_state=False,
        rows=3,
        capacity=3,
        segments=2,
        allocate_kv_cache=True,
        normalized_qk_heads=6,
        f16_policy=None,
        disabled_fields=None,
        arena_grouping="single",
        priority_min_live_stages=None,
    )
    kwargs.update(overrides)
    return build_prefill_scratch_plan(cfg, **kwargs)


# Explicit expected bytes for the 3-row dense fixture above. Every entry is the
# hand-evaluated formula from the allocator, so a drift in the plan shows up as
# a wrong number rather than as a matching second expression.
_EXPECTED_DENSE_FIELD_SIZES = {
    "norm": 96,
    "full_q": 384,
    "full_k": 96,
    "full_v": 96,
    "linear_qkv": 384,
    "linear_qkv_f32": 768,
    "linear_z": 288,
    "linear_z_f32": 576,
    "linear_alpha": 36,
    "linear_alpha_f32": 72,
    "linear_beta": 36,
    "linear_beta_f32": 72,
    "conv_out": 768,
    "prefill_query": 576,
    "prefill_key": 576,
    "prefill_value": 576,
    "q6_f16_x": 0,
    "q6_f16_weight": 0,
    "q6_f16_out": 0,
    "prefill_beta": 72,
    "prefill_decay": 72,
    "prefill_query_scale": 72,
    "prefill_key_scale": 72,
    "recurrent_out": 576,
    "recurrent_bf16": 288,
    "linear_conv_state_tmp": 1024,
    "linear_recurrent_state_tmp": 768,
    "full_query_raw": 384,
    "full_key_raw": 192,
    "full_query": 384,
    "full_key": 192,
    "full_query_bf16": 192,
    "full_gate": 192,
    "full_attn_bf16": 192,
    "full_gated": 192,
    "attn_out": 96,
    "post_norm": 96,
    "post_norm_f32": 192,
    "residual": 96,
    "ffn_gate_up": 96,
    "ffn_intermediate": 48,
    "ffn_intermediate_f32": 96,
    "ffn_down": 96,
    "moe_q8_1": 108,
    "moe_router_logits": 12,
    "moe_shared_gate_logits": 12,
    "moe_selected_experts": 24,
    "moe_routing_weights": 12,
    "moe_down_out": 96,
    "moe_down_out_f32": 192,
    "moe_group_counts": 4,
    "moe_padded_counts": 4,
    "moe_scatter_offsets": 4,
    "moe_expert_start_compact": 16,
    "moe_expert_start_wmma": 16,
    "moe_total_compact": 8,
    "moe_wmma_total": 8,
    "moe_tile_expert": 16,
    "moe_sorted_lanes": 24,
    "moe_sorted_experts": 24,
    "moe_sorted_weights": 12,
    "moe_lane_to_row": 24,
    "moe_shared_gate": 48,
    "moe_shared_up": 48,
    "moe_shared_intermediate": 48,
    "moe_shared_out": 96,
    "moe_shared_out_f32": 192,
}

_EXPECTED_DEDICATED_SIZES = {
    "gdn_cu_seqlens": 12,
    "gdn_state_indices": 16,
    "block_table": 12,
    "positions": 24,
    "context_counts": 24,
    "cos_table": 96,
    "sin_table": 96,
    "cu_q": 12,
    "cu_k": 12,
    "softmax_lse": 48,
    "full_attn_split_partial": 128,
    "full_attn_split_m": 16,
    "full_attn_split_l": 16,
    "atomic": 4,
    "moe_router_counter": 4,
    "key_cache": 32768,
    "value_cache": 32768,
}


def test_field_sizes_match_explicit_formulas_in_allocator_order() -> None:
    plan = _build(_dense_cfg())
    assert dict(plan.field_sizes) == _EXPECTED_DENSE_FIELD_SIZES
    assert tuple(plan.field_sizes) == tuple(_EXPECTED_DENSE_FIELD_SIZES)
    assert dict(plan.dedicated_sizes) == _EXPECTED_DEDICATED_SIZES
    assert tuple(plan.dedicated_sizes) == tuple(_EXPECTED_DEDICATED_SIZES)


def test_dedicated_layout_owner_order_is_bulk_then_metadata_then_kv() -> None:
    plan = _build(_dense_cfg())

    expected_owners = tuple(
        name
        for name in _EXPECTED_DENSE_FIELD_SIZES
        if name not in {"q6_f16_x", "q6_f16_weight", "q6_f16_out"}
    ) + tuple(_EXPECTED_DEDICATED_SIZES)
    assert plan.allocation_mode == "dedicated"
    assert tuple(plan.owner_sizes) == expected_owners
    assert tuple(plan.owner_sizes.values()) == tuple(
        _EXPECTED_DENSE_FIELD_SIZES[name] for name in expected_owners[: len(_EXPECTED_DENSE_FIELD_SIZES) - 3]
    ) + tuple(_EXPECTED_DEDICATED_SIZES.values())
    assert plan.total_nbytes == sum(plan.owner_sizes.values())
    assert plan.allocation_offsets == {}
    assert plan.allocation_lifetimes == {}
    assert plan.allocation_groups == {}
    assert plan.allocation_inplace_aliases == {}
    assert plan.allocation_subranges == {}
    for name in ("q6_f16_x", "q6_f16_weight", "q6_f16_out"):
        assert plan.views[name] is None
    for name in expected_owners:
        assert plan.views[name] == (name, 0, plan.owner_sizes[name])


def test_optional_kv_cache_views_and_block_rounded_bytes() -> None:
    without = _build(_dense_cfg(), allocate_kv_cache=False)
    assert "key_cache" not in without.owner_sizes
    assert "value_cache" not in without.owner_sizes
    assert without.views["key_cache"] is None
    assert without.views["value_cache"] is None

    with_cache = _build(_dense_cfg(), capacity=257, rows=3)
    assert with_cache.max_positions == 257
    assert with_cache.blocks == 2
    assert with_cache.block_rounded_positions == 512
    # 512 block-rounded positions x 2 kv heads x 32 key dims x 2 bytes.
    assert with_cache.owner_sizes["key_cache"] == 512 * 2 * 32 * 2
    assert with_cache.owner_sizes["value_cache"] == 512 * 2 * 32 * 2


def test_moe_and_split_geometry_fields() -> None:
    plan = _build(_moe_cfg(), rows=8, capacity=257)
    assert plan.full_attn_split_count == 2
    assert plan.full_attn_split_batch_rows == 8
    assert plan.full_attn_split_capacity_rows == 1
    assert plan.full_attn_split_partial_row_bytes == 32 * 2 * 4
    assert plan.full_attn_split_stat_row_bytes == 4 * 2 * 4
    assert plan.moe_experts == 4
    assert plan.moe_top_k == 2
    assert plan.moe_selected_rows_capacity == 16
    assert plan.moe_wmma_rows_capacity == 16 + 16 * 4
    assert plan.field_sizes["moe_group_counts"] == 4 * 4
    assert plan.field_sizes["moe_tile_expert"] == 5 * 8


def test_source_f16_policy_sizes_three_planes() -> None:
    policy = {
        "gguf_q4_k_t16_v1": {(32, 16): {8: 16, 4: 32}},
        "linear_variant_intervals_by_quant": {(32, 16): {(1, 8): "x"}},
        "max_rows_by_quant_shape": {"gguf_q4_k_t16_v1": {(32, 16): 64}},
        "pair_only_second_operand_policies": {},
    }
    plan = _build(_dense_cfg(), f16_policy=policy)
    # x: rows * hidden * 2; weight: max(tile * shape[0]) * 2; out: rows * max(tile) * 2
    assert plan.field_sizes["q6_f16_x"] == 3 * 16 * 2
    assert plan.field_sizes["q6_f16_weight"] == max(16 * 32, 32 * 32) * 2
    assert plan.field_sizes["q6_f16_out"] == 3 * 32 * 2
    assert plan.f16_policy_applied is True
    assert plan.f16_shape_tiles == (((32, 16), ((4, 32), (8, 16))),)
    assert "q6_f16_x" in plan.owner_sizes


def test_source_f16_policy_with_only_bookkeeping_keys_leaves_weight_planes_empty() -> None:
    policy = {
        "linear_variant_intervals_by_quant": {},
        "max_rows_by_quant_shape": {},
        "pair_only_second_operand_policies": {},
    }
    plan = _build(_dense_cfg(), f16_policy=policy)
    assert plan.field_sizes["q6_f16_x"] == 3 * 16 * 2
    assert plan.field_sizes["q6_f16_weight"] == 0
    assert plan.field_sizes["q6_f16_out"] == 0
    assert "q6_f16_x" in plan.owner_sizes
    assert plan.views["q6_f16_weight"] is None
    assert plan.views["q6_f16_out"] is None


def test_single_arena_layout_is_aligned_and_reports_its_groups() -> None:
    plan = _build(
        _moe_cfg(),
        rows=8,
        capacity=8,
        allocate_kv_cache=False,
        disabled_fields=_GGUF_PREFILL_SCRATCH_PEER_DISABLED_FIELDS,
    )
    assert plan.allocation_mode == "liveness_aliased"
    bulk_owners = [
        name for name in plan.owner_sizes if name not in plan.dedicated_sizes
    ]
    assert bulk_owners == ["arena_bulk"]
    arena_bytes = plan.owner_sizes["arena_bulk"]
    assert arena_bytes % 256 == 0
    assert plan.allocation_groups == {
        name: "single" for name in plan.allocation_offsets
    }
    assert plan.allocation_inplace_aliases == {}
    assert plan.allocation_subranges == {}
    for name, (offset, nbytes) in plan.allocation_offsets.items():
        assert offset % 256 == 0
        assert nbytes == plan.field_sizes[name]
        assert plan.views[name] == ("arena_bulk", offset, nbytes)
        assert offset + nbytes <= arena_bytes
    for name in _GGUF_PREFILL_SCRATCH_PEER_DISABLED_FIELDS:
        assert plan.views[name] is None
        assert name not in plan.allocation_offsets
    assert plan.total_nbytes == arena_bytes + sum(plan.dedicated_sizes.values())


def test_owner_slot_layout_uses_slot_pointers_and_virtual_offsets() -> None:
    plan = _build(
        _dense_cfg(),
        rows=8,
        capacity=8,
        allocate_kv_cache=False,
        disabled_fields=_DENSE_LIVENESS_DISABLED,
        arena_grouping="owner_slots",
    )
    assert plan.allocation_mode == "liveness_aliased"
    bulk_owners = [
        name for name in plan.owner_sizes if name not in plan.dedicated_sizes
    ]
    assert len(bulk_owners) > 1
    assert plan.allocation_inplace_aliases == {}
    assert plan.allocation_subranges == {}
    for owner_name in bulk_owners:
        assert owner_name.startswith("owner_slot_")
        assert plan.owner_sizes[owner_name] % 256 == 0
    for name, view in plan.views.items():
        if view is None or name in plan.dedicated_sizes:
            continue
        owner_name, physical_offset, nbytes = view
        assert physical_offset == 0
        assert nbytes == plan.field_sizes[name]
        assert plan.allocation_groups[name] == owner_name
    # Virtual offsets are cumulative across slots, not the physical slot base.
    assert any(
        offset != 0 for offset, _nbytes in plan.allocation_offsets.values()
    )


def test_dense_single_arena_aliases_intermediate_onto_gate_up() -> None:
    plan = _build(
        _dense_cfg(),
        rows=8,
        capacity=8,
        allocate_kv_cache=False,
        disabled_fields=_DENSE_LIVENESS_DISABLED,
        arena_grouping="single",
    )
    alias_bytes = plan.field_sizes["ffn_intermediate"]
    source_bytes = plan.field_sizes["ffn_gate_up"]
    assert alias_bytes == 128
    assert source_bytes == 256
    assert dict(plan.allocation_inplace_aliases) == {
        "ffn_intermediate": "ffn_gate_up"
    }
    assert "ffn_intermediate" not in plan.owner_sizes
    source_lifetimes = _GGUF_PREFILL_SCRATCH_DENSE_LIFETIMES["ffn_gate_up"]
    alias_lifetimes = _GGUF_PREFILL_SCRATCH_DENSE_LIFETIMES["ffn_intermediate"]
    combined = tuple(dict.fromkeys((*source_lifetimes, *alias_lifetimes)))
    assert plan.allocation_lifetimes["ffn_gate_up"] == combined
    assert plan.allocation_lifetimes["ffn_intermediate"] == alias_lifetimes
    assert plan.allocation_subranges["ffn_gate_up"] == (
        (0, alias_bytes, combined),
        (alias_bytes, source_bytes - alias_bytes, source_lifetimes),
    )
    owner, offset, nbytes = plan.views["ffn_intermediate"]
    assert (owner, offset, nbytes) == (
        plan.views["ffn_gate_up"][0],
        plan.views["ffn_gate_up"][1],
        alias_bytes,
    )
    assert plan.allocation_offsets["ffn_intermediate"] == (offset, alias_bytes)


def test_owner_slot_grouping_never_uses_the_dense_alias() -> None:
    plan = _build(
        _dense_cfg(),
        rows=8,
        capacity=8,
        allocate_kv_cache=False,
        disabled_fields=_DENSE_LIVENESS_DISABLED,
        arena_grouping="owner_slots",
    )
    assert dict(plan.allocation_inplace_aliases) == {}
    assert plan.views["ffn_intermediate"] is not None
    assert "ffn_intermediate" in plan.allocation_offsets
    assert plan.allocation_subranges == {}


def _synthetic_lifetimes(**entries):
    return {name: value for name, value in entries.items()}


def test_single_arena_places_live_fields_one_color_apart() -> None:
    lifetimes = _synthetic_lifetimes(
        a=(("x", 0, 1),),
        b=(("x", 0, 1),),
        c=(("x", 1, 2),),
    )
    arena, offsets = _plan_prefill_scratch_liveness_arena(
        {"a": 1000, "b": 1000, "c": 1000},
        lifetimes=lifetimes,
    )
    # Overlapping fields separate by one 64-KiB color; the disjoint field reuses
    # the base offset.
    assert offsets["a"] == (0, 1000)
    assert offsets["b"] == (66560, 1000)
    assert offsets["b"][0] - offsets["a"][0] == 66560
    assert offsets["b"][0] >= 1000 + _GGUF_PREFILL_SCRATCH_COLOR_BYTES
    assert offsets["c"] == (0, 1000)
    assert arena == 67584


def test_single_arena_priority_reorder_is_adopted_only_when_smaller() -> None:
    # Durations a=1, b=2, c=1, so priority order is b, c, a. Size order is
    # c, a, b. The priority plan is strictly smaller, so it is adopted.
    adopt_lifetimes = _synthetic_lifetimes(
        a=(("x", 0, 1),),
        b=(("x", 1, 3),),
        c=(("x", 0, 1),),
    )
    sizes = {"a": 500, "b": 500, "c": 800}
    base_arena, base_offsets = _plan_prefill_scratch_liveness_arena(
        sizes, lifetimes=adopt_lifetimes
    )
    priority_arena, priority_offsets = _plan_prefill_scratch_liveness_arena(
        sizes, lifetimes=adopt_lifetimes, priority_min_live_stages=1
    )
    assert base_arena == 67072
    assert dict(base_offsets) == {"c": (0, 800), "a": (66560, 500), "b": (0, 500)}
    assert priority_arena == 66560
    assert dict(priority_offsets) == {
        "b": (0, 500),
        "c": (0, 800),
        "a": (66048, 500),
    }

    # Durations a=1, b=2, c=1; priority order b, c, a is larger here, so the
    # established size-order addresses stay.
    reject_lifetimes = _synthetic_lifetimes(
        a=(("x", 0, 1),),
        b=(("x", 0, 2),),
        c=(("x", 1, 2),),
    )
    base_arena, base_offsets = _plan_prefill_scratch_liveness_arena(
        sizes, lifetimes=reject_lifetimes
    )
    priority_arena, priority_offsets = _plan_prefill_scratch_liveness_arena(
        sizes, lifetimes=reject_lifetimes, priority_min_live_stages=1
    )
    assert base_arena == 66560
    assert dict(base_offsets) == {"c": (0, 800), "a": (0, 500), "b": (66048, 500)}
    assert priority_arena == base_arena
    assert priority_offsets == base_offsets


def test_single_arena_priority_keeps_addresses_on_a_tie() -> None:
    lifetimes = _synthetic_lifetimes(
        a=(("x", 0, 1),),
        b=(("x", 0, 2),),
    )
    base_arena, base_offsets = _plan_prefill_scratch_liveness_arena(
        {"a": 1000, "b": 1000}, lifetimes=lifetimes
    )
    priority_arena, priority_offsets = _plan_prefill_scratch_liveness_arena(
        {"a": 1000, "b": 1000}, lifetimes=lifetimes, priority_min_live_stages=1
    )
    assert priority_arena == base_arena
    assert priority_offsets == base_offsets
    assert base_arena == 67584
    assert dict(base_offsets) == {"a": (0, 1000), "b": (66560, 1000)}


def test_owner_slots_group_by_disjoint_lifetime_and_place_virtually() -> None:
    lifetimes = _synthetic_lifetimes(
        a=(("x", 0, 1),),
        b=(("x", 0, 1),),
        c=(("x", 1, 2),),
    )
    members, owner_sizes, virtual_offsets, groups = (
        _plan_prefill_scratch_liveness_owner_slots(
            {"a": 100, "b": 100, "c": 50}, lifetimes=lifetimes
        )
    )
    assert members == (("a", "c"), ("b",))
    assert owner_sizes == (256, 256)
    assert virtual_offsets["a"] == (0, 100)
    assert virtual_offsets["c"] == (0, 50)
    assert virtual_offsets["b"] == (256, 100)
    assert groups == {"a": "owner_slot_00", "c": "owner_slot_00", "b": "owner_slot_01"}


def test_subrange_validation_rejects_out_of_bounds_planes() -> None:
    lifetimes = {"a": (("x", 0, 1),)}
    with pytest.raises(ValueError, match="invalid bulk-prefill allocation subrange"):
        _prefill_scratch_field_subranges(
            "a",
            64,
            lifetimes=lifetimes,
            allocation_subranges={"a": ((32, 64, lifetimes["a"]),)},
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"rows": 0}, "rows must be positive"),
        ({"segments": 0}, "segments must be positive"),
        ({"rows": 8, "capacity": 7}, "capacity 7 must be >= rows 8"),
        ({"arena_grouping": "slots"}, "arena grouping must be one of"),
        ({"disabled_fields": frozenset({"not_a_field"})}, "unknown bulk-prefill"),
    ),
)
def test_invalid_inputs_fail_loudly(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        _build(_dense_cfg(), **kwargs)


def test_plan_is_immutable() -> None:
    plan = _build(_dense_cfg())
    with pytest.raises(dataclasses.FrozenInstanceError):
        plan.rows = 4
    with pytest.raises(TypeError):
        plan.field_sizes["norm"] = 1
    with pytest.raises(TypeError):
        plan.owner_sizes["norm"] = 1
    with pytest.raises(KeyError, match="has no storage"):
        plan.storage("q6_f16_x")
    assert plan.storage("norm") == ("norm", 0, 96)


def test_planner_source_has_no_device_or_array_dependencies() -> None:
    source = Path(prefill_plan.__file__).read_text()
    for forbidden in (
        "import os",
        "os.environ",
        "numpy",
        "np.",
        "malloc(",
        "DeviceBuffer",
        "copy_host_to_device",
        "copy_host_array_to_device",
        "torch",
        "backend_package_capability",
    ):
        assert forbidden not in source, forbidden


def test_planner_imports_without_the_runtime_package() -> None:
    code = (
        "import importlib.util, sys\n"
        "import hipengine.core.dtype\n"
        "spec = importlib.util.spec_from_file_location('prefill_plan_probe', sys.argv[1])\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "sys.modules[spec.name] = module\n"
        "spec.loader.exec_module(module)\n"
        "assert 'hipengine.runtime' not in sys.modules\n"
        "assert 'hipengine.kernels' not in sys.modules\n"
        "assert 'torch' not in sys.modules\n"
        "assert hasattr(module, 'build_prefill_scratch_plan')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(Path(prefill_plan.__file__).resolve())],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


_LAYOUT_CASES = (
    # backend, rows, capacity, segments, expected allocation mode
    ("cpu_reference", 8, 8, 1, "dedicated"),
    ("cpu_reference", 256, 1024, 3, "dedicated"),
    ("hip_gfx1100", 256, 256, 1, "liveness_aliased"),
    ("hip_gfx1100", 768, 1024, 3, "liveness_aliased"),
    ("hip_gfx1151", 768, 768, 1, "liveness_aliased"),
    ("hip_gfx1151", 768, 1024, 3, "liveness_aliased"),
)


@pytest.mark.parametrize(
    ("backend", "rows", "capacity", "segments", "expected_mode"), _LAYOUT_CASES
)
def test_plan_owner_order_and_views_match_the_allocator(
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
    rows: int,
    capacity: int,
    segments: int,
    expected_mode: str,
) -> None:
    _clear_diagnostic_environment(monkeypatch)
    runner = _fake_runner(backend)
    allocations: list[DeviceBuffer] = []

    def fake_malloc(nbytes: int, *, runtime):
        buffer = DeviceBuffer(ptr=0x10000000 + 0x10000 * len(allocations), nbytes=int(nbytes))
        allocations.append(buffer)
        return buffer

    monkeypatch.setattr(gguf_runner, "malloc", fake_malloc)
    monkeypatch.setattr(gguf_runner, "copy_host_array_to_device", lambda *args, **kwargs: None)
    scratch = gguf_runner._GGUFFullAttentionPrefillScratch.allocate(
        runner,
        rows=rows,
        capacity=capacity,
        allocate_kv_cache=False,
        segments=segments,
        runtime=SimpleNamespace(),
    )
    plan = gguf_runner._gguf_prefill_scratch_plan(
        runner,
        rows=rows,
        capacity=capacity,
        segments=segments,
        allocate_kv_cache=False,
    )

    assert plan.allocation_mode == expected_mode
    assert scratch.allocation_mode == plan.allocation_mode
    assert [int(buffer.nbytes) for buffer in allocations] == list(plan.owner_sizes.values())
    owner_index = {name: index for index, name in enumerate(plan.owner_sizes)}
    for name, view in plan.views.items():
        field = getattr(scratch, name)
        if view is None:
            if name in gguf_runner._GGUF_PREFILL_SCRATCH_OPTIONAL_FIELDS:
                assert field is None
            else:
                assert int(field.ptr) == 0
                assert int(field.nbytes) == 0
            continue
        owner_name, offset, nbytes = view
        owner = allocations[owner_index[owner_name]]
        assert int(field.ptr) == int(owner.ptr) + offset
        assert int(field.nbytes) == nbytes
    assert sum(buffer.nbytes for buffer in allocations) == plan.total_nbytes
    assert dict(scratch.allocation_offsets) == dict(plan.allocation_offsets)
    assert dict(scratch.allocation_lifetimes) == dict(plan.allocation_lifetimes)
    assert dict(scratch.allocation_groups) == dict(plan.allocation_groups)


def test_odd_rotary_width_rounds_rope_tables_down_to_the_materialized_shape() -> None:
    cfg = _dense_cfg()
    cfg.rope_dimension_count = 65
    plan = _build(cfg, allocate_kv_cache=False)
    # ``_rope_tables`` materializes 2 * (65 // 2) = 64 columns per row.
    assert plan.owner_sizes["cos_table"] == 3 * 64 * 4
    assert plan.owner_sizes["sin_table"] == 3 * 64 * 4
    assert plan.owner_sizes["cos_table"] != 3 * 65 * 4


def test_compact_and_peer_sets_differ_only_in_normalized_qkv_planes() -> None:
    compact = _build(
        _moe_cfg(),
        rows=8,
        capacity=8,
        allocate_kv_cache=False,
        disabled_fields=_GGUF_PREFILL_SCRATCH_COMPACT_DISABLED_FIELDS,
    )
    peer = _build(
        _moe_cfg(),
        rows=8,
        capacity=8,
        allocate_kv_cache=False,
        disabled_fields=_GGUF_PREFILL_SCRATCH_PEER_DISABLED_FIELDS,
    )
    for name in ("prefill_query", "prefill_key", "prefill_value"):
        assert compact.views[name] is None
        assert peer.views[name] is not None
    assert compact.total_nbytes != peer.total_nbytes
    assert DType.BF16.itemsize == 2
