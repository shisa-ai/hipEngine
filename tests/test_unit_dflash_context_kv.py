from __future__ import annotations

import numpy as np
import pytest

from hipengine.core.device import Device
from hipengine.core.tensor import Tensor
from hipengine.speculative import (
    DFlashDraftKVCacheOwner,
    DFlashDraftKVCacheSpec,
    DFlashDraftKVLayerWeights,
    DFlashDraftKVMaterializerScratch,
    append_materialized_kv_reference,
    full_context_kv_reference,
    materialize_dflash_draft_kv_append_from_projected,
    plan_dflash_draft_kv_append,
)


class FakeWorkspace:
    def __init__(self, device: Device) -> None:
        self.device = device
        self.next_ptr = 0x100000
        self.tensors: dict[str, Tensor] = {}

    def reserve_tensor(self, name: str, shape, dtype) -> Tensor:
        tensor = Tensor.from_handle(self.next_ptr, tuple(int(dim) for dim in shape), dtype, self.device)
        self.next_ptr += 0x1000
        self.tensors[name] = tensor
        return tensor


def test_dflash_draft_kv_cache_owner_allocates_stable_metadata() -> None:
    spec = DFlashDraftKVCacheSpec(
        backend="hip_gfx1151",
        bucket="chain-n4",
        device=Device("hip", 0),
        layer_count=2,
        capacity_tokens=8,
        num_kv_heads=2,
        head_dim=4,
    )
    workspace = FakeWorkspace(spec.device)

    owner = DFlashDraftKVCacheOwner.allocate(spec, workspace=workspace)

    assert owner.keys.shape == (2, 8, 2, 4)
    assert owner.keys.dtype.value == "fp32"
    assert owner.values.dtype.value == "bf16"
    assert owner.positions.shape == (8,)
    assert owner.live_count.shape == (1,)
    assert owner.metadata["total_bytes"] == spec.total_bytes
    assert owner.metadata["phases"] == ("full_context_rebuild", "append_materialize", "query_only_drafter")
    assert len({tensor.ptr for tensor in workspace.tensors.values()}) == len(workspace.tensors)


def test_dflash_draft_kv_append_plan_validates_capacity() -> None:
    plan = plan_dflash_draft_kv_append(live_count=3, new_positions=(13, 14), capacity_tokens=5)

    assert plan.start == 3
    assert plan.end == 5
    assert plan.positions == (13, 14)
    with pytest.raises(ValueError, match="capacity"):
        plan_dflash_draft_kv_append(live_count=4, new_positions=(1, 2), capacity_tokens=5)
    with pytest.raises(ValueError, match="positions length"):
        type(plan)(start=0, count=2, positions=(1,))


def test_append_materialized_kv_matches_full_context_reference_and_preserves_suffix() -> None:
    rng = np.random.default_rng(0)
    layer_count = 2
    capacity = 6
    heads = 2
    head_dim = 3
    prefix_rows = 3
    append_rows = 2
    full_rows = prefix_rows + append_rows
    materialized_keys = rng.normal(size=(layer_count, full_rows, heads, head_dim)).astype(np.float32)
    materialized_values = rng.integers(0, 65535, size=(layer_count, full_rows, heads, head_dim), dtype=np.uint16)
    full_keys, full_values = full_context_kv_reference(materialized_keys, materialized_values, capacity_tokens=capacity)
    existing_keys, existing_values = full_context_kv_reference(
        materialized_keys[:, :prefix_rows],
        materialized_values[:, :prefix_rows],
        capacity_tokens=capacity,
    )
    sentinel_key = np.full((layer_count, capacity - prefix_rows, heads, head_dim), -999.0, dtype=np.float32)
    sentinel_value = np.full((layer_count, capacity - prefix_rows, heads, head_dim), 65535, dtype=np.uint16)
    existing_keys[:, prefix_rows:] = sentinel_key
    existing_values[:, prefix_rows:] = sentinel_value

    appended_keys, appended_values = append_materialized_kv_reference(
        existing_keys,
        existing_values,
        materialized_keys[:, prefix_rows:full_rows],
        materialized_values[:, prefix_rows:full_rows],
        start=prefix_rows,
    )

    np.testing.assert_array_equal(appended_keys[:, :full_rows], full_keys[:, :full_rows])
    np.testing.assert_array_equal(appended_values[:, :full_rows], full_values[:, :full_rows])
    np.testing.assert_array_equal(appended_keys[:, full_rows:], sentinel_key[:, append_rows:])
    np.testing.assert_array_equal(appended_values[:, full_rows:], sentinel_value[:, append_rows:])


def test_materialize_append_validates_scratch_and_reports_metadata(monkeypatch) -> None:
    device = Device("hip", 0)
    spec = DFlashDraftKVCacheSpec(
        backend="hip_gfx1151",
        bucket="chain-n2",
        device=device,
        layer_count=1,
        capacity_tokens=4,
        num_kv_heads=1,
        head_dim=4,
    )
    owner = DFlashDraftKVCacheOwner.allocate(spec, workspace=FakeWorkspace(device))
    plan = plan_dflash_draft_kv_append(live_count=1, new_positions=(6, 7), capacity_tokens=4)
    projected = Tensor.from_handle(0x2000, (2, 8), "bf16", device)
    positions = Tensor.from_handle(0x2100, (2,), "int32", device)
    weights = [
        DFlashDraftKVLayerWeights(
            k_proj=Tensor.from_handle(0x2200, (4, 8), "bf16", device),
            v_proj=Tensor.from_handle(0x2300, (4, 8), "bf16", device),
            k_norm=Tensor.from_handle(0x2400, (4,), "bf16", device),
        )
    ]
    scratch = DFlashDraftKVMaterializerScratch(
        projected_hidden=Tensor.from_handle(0x2500, (2, 8), "bf16", device),
        key_raw=Tensor.from_handle(0x2600, (2, 4), "fp32", device),
    )
    cos = Tensor.from_handle(0x2700, (16, 4), "fp32", device)
    sin = Tensor.from_handle(0x2800, (16, 4), "fp32", device)
    calls = []

    def fake_dense_f32(*args, **kwargs):
        calls.append(("k", args, kwargs))

    def fake_key_rotary(*args, **kwargs):
        calls.append(("rotary", args, kwargs))

    def fake_dense_bf16(*args, **kwargs):
        calls.append(("v", args, kwargs))

    def fake_metadata(*args, **kwargs):
        calls.append(("metadata", args, kwargs))

    import hipengine.kernels.hip_gfx1100.speculative.dflash_drafter as kernels

    monkeypatch.setattr(kernels, "dflash_dense_bf16_to_f32", fake_dense_f32)
    monkeypatch.setattr(kernels, "dflash_key_rmsnorm_rotary_f32", fake_key_rotary)
    monkeypatch.setattr(kernels, "dflash_dense_bf16_to_bf16", fake_dense_bf16)
    monkeypatch.setattr(kernels, "dflash_update_kv_metadata_i32", fake_metadata)

    result = materialize_dflash_draft_kv_append_from_projected(
        owner=owner,
        plan=plan,
        projected_hidden=projected,
        positions=positions,
        layer_weights=weights,
        scratch=scratch,
        cos_table=cos,
        sin_table=sin,
        library="lib",
        runtime="runtime",
    )

    assert [call[0] for call in calls] == ["k", "rotary", "v", "metadata"]
    assert result.as_metadata()["append_count"] == 2
    assert result.as_metadata()["live_count"] == 3
    assert result.as_metadata()["draft_kv_bytes"] == spec.total_bytes
    assert calls[-1][2]["start"] == 1
    assert calls[-1][2]["end"] == 3


def test_materialize_append_rejects_bad_positions_shape() -> None:
    device = Device("hip", 0)
    spec = DFlashDraftKVCacheSpec("hip_gfx1151", "bad", device, 1, 4, 1, 4)
    owner = DFlashDraftKVCacheOwner.allocate(spec, workspace=FakeWorkspace(device))
    plan = plan_dflash_draft_kv_append(live_count=0, new_positions=(1,), capacity_tokens=4)
    with pytest.raises(ValueError, match="positions tensor"):
        materialize_dflash_draft_kv_append_from_projected(
            owner=owner,
            plan=plan,
            projected_hidden=Tensor.from_handle(0x2000, (1, 8), "bf16", device),
            positions=Tensor.from_handle(0x2100, (2,), "int32", device),
            layer_weights=[
                DFlashDraftKVLayerWeights(
                    k_proj=Tensor.from_handle(0x2200, (4, 8), "bf16", device),
                    v_proj=Tensor.from_handle(0x2300, (4, 8), "bf16", device),
                    k_norm=Tensor.from_handle(0x2400, (4,), "bf16", device),
                )
            ],
            scratch=DFlashDraftKVMaterializerScratch(
                projected_hidden=Tensor.from_handle(0x2500, (1, 8), "bf16", device),
                key_raw=Tensor.from_handle(0x2600, (1, 4), "fp32", device),
            ),
            cos_table=Tensor.from_handle(0x2700, (16, 4), "fp32", device),
            sin_table=Tensor.from_handle(0x2800, (16, 4), "fp32", device),
        )


def test_append_materialized_kv_rejects_shape_mismatch() -> None:
    existing = np.zeros((2, 4, 1, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="layer/head"):
        append_materialized_kv_reference(existing, existing, np.zeros((3, 1, 1, 3)), np.zeros((3, 1, 1, 3)), start=0)
