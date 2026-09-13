"""Unit tests for byte-preserving GGUF tensor-parallel shard planning.

Three layers of coverage:

1. Synthetic byte semantics: for every split kind and every GGML block layout in
   this model family, the per-rank payloads must tile the source payload exactly
   once and reconstruct bit-identically.
2. Real-model structure: the Qwen3.8-27B shard plan must put whole attention
   heads, GDN head groups, and block-aligned column ranges on each rank, and
   must refuse degrees the geometry cannot support.
3. Layer fixtures: numpy MLP, full-attention, and GDN layer models computed from
   the planned shards must reproduce the unsharded layer result.

The layer fixtures are the semantic gate: byte preservation alone cannot detect
a plan that cuts the right bytes but the wrong heads.
"""

from __future__ import annotations

import ctypes
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.loading.qwen35_gguf_shards import (
    COLUMN,
    GROUP,
    OWNER,
    REPLICATED,
    ROW,
    AxisSegment,
    ShardManifest,
    RowSplitLayout,
    ShardPlanError,
    ShardSegment,
    TensorShardRule,
    TensorShardPlan,
    TensorShardSlice,
    build_shard_manifest,
    build_tensor_shard_plan,
    gdn_head_map,
    iter_rank_payloads,
    materialize_manifest,
    materialize_slice,
    streaming_memory_report,
    partition_groups,
    reconstruct_tensor,
    shard_rule_for_tensor,
    validate_plan_coverage,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
GGUF_PATH = "/models/gguf/Qwen3.8-27B-Q4_K_M.gguf"

# (ggml type id, name, block size, type size)
QUANT_LAYOUTS = (
    (0, "F32", 1, 4),
    (1, "F16", 1, 2),
    (8, "Q8_0", 32, 34),
    (12, "Q4_K", 256, 144),
    (13, "Q5_K", 256, 176),
    (14, "Q6_K", 256, 210),
)


def _payload(nbytes: int, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=int(nbytes), dtype=np.uint8)


def _round_trip(plan: TensorShardPlan, source: np.ndarray) -> None:
    payloads = {
        shard_slice.rank: materialize_slice(source, shard_slice) for shard_slice in plan.slices
    }
    rebuilt = reconstruct_tensor(
        _manifest_for(plan), plan.name, {rank: payload for rank, payload in payloads.items()}
    )
    assert np.array_equal(rebuilt, np.asarray(source)), f"{plan.name}: shard set is not bit-exact"


def _manifest_for(plan: TensorShardPlan):
    from hipengine.loading.qwen35_gguf_shards import ShardManifest

    return ShardManifest(
        model_hash="test",
        world_size=len(plan.slices),
        hidden_size=1,
        tensors=(plan,),
    )


def _make_plan(
    *,
    name: str,
    shape: tuple[int, ...],
    quant: tuple[int, str, int, int],
    rule: TensorShardRule,
    world_size: int,
) -> tuple[TensorShardPlan, np.ndarray]:
    type_id, type_name, block_size, type_size = quant
    rows = int(shape[0])
    cols = int(shape[1]) if len(shape) > 1 else 1
    nbytes = rows * (cols // block_size) * type_size
    plan = build_tensor_shard_plan(
        name=name,
        shape=shape,
        nbytes=nbytes,
        quant_type_id=type_id,
        quant_type_name=type_name,
        rule=rule,
        world_size=world_size,
    )
    return plan, _payload(nbytes)


# ---------------------------------------------------------------------------
# 1. Synthetic byte semantics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("quant", QUANT_LAYOUTS, ids=[entry[1] for entry in QUANT_LAYOUTS])
def test_unit_shards_column_split_round_trips(quant):
    """Column splits cut whole rows, so any row count is safe."""

    plan, source = _make_plan(
        name="w",
        shape=(256, 512),
        quant=quant,
        rule=TensorShardRule(kind=COLUMN, axis=0, segments=(AxisSegment(0, 256, 1),)),
        world_size=4,
    )
    block_size, type_size = quant[2], quant[3]
    local_rows = 64
    for shard_slice in plan.slices:
        assert shard_slice.local_shape == (local_rows, 512)
        assert shard_slice.local_nbytes == local_rows * (512 // block_size) * type_size
    assert [shard_slice.axis_ranges for shard_slice in plan.slices] == [
        ((0, 64),),
        ((64, 128),),
        ((128, 192),),
        ((192, 256),),
    ]
    _round_trip(plan, source)


@pytest.mark.parametrize("quant", QUANT_LAYOUTS, ids=[entry[1] for entry in QUANT_LAYOUTS])
def test_unit_shards_row_split_round_trips_and_shrinks_row_stride(quant):
    """Row splits keep each row's block-aligned column range contiguous."""

    block_size, type_size = quant[2], quant[3]
    plan, source = _make_plan(
        name="w",
        shape=(64, 4 * block_size),
        quant=quant,
        rule=TensorShardRule(kind=ROW, axis=1),
        world_size=4,
    )
    local_cols = block_size
    expected_row_bytes = local_cols // block_size * type_size
    for shard_slice in plan.slices:
        assert shard_slice.local_shape == (64, local_cols)
        assert shard_slice.local_nbytes == 64 * expected_row_bytes
        assert shard_slice.row_split.local_row_bytes == expected_row_bytes
        assert shard_slice.row_split.source_row_bytes == (4 * block_size) // block_size * type_size
    _round_trip(plan, source)


@pytest.mark.parametrize("quant", QUANT_LAYOUTS, ids=[entry[1] for entry in QUANT_LAYOUTS])
def test_unit_shards_group_split_round_trips(quant):
    plan, source = _make_plan(
        name="w",
        shape=(16 * 128, 512),
        quant=quant,
        rule=TensorShardRule(kind=GROUP, axis=0, segments=(AxisSegment(0, 16 * 128, 128),)),
        world_size=4,
    )
    for shard_slice in plan.slices:
        assert shard_slice.local_shape == (4 * 128, 512)
        assert shard_slice.axis_ranges == ((shard_slice.rank * 512, (shard_slice.rank + 1) * 512),)
    _round_trip(plan, source)


def test_unit_shards_multi_segment_group_split_round_trips():
    """A fused q/k/v tensor splits each segment family independently."""

    shape = (2048 + 2048 + 3 * 2048, 512)
    rule = TensorShardRule(
        kind=GROUP,
        axis=0,
        segments=(
            AxisSegment(0, 2048, 128),
            AxisSegment(2048, 4096, 128),
            AxisSegment(4096, 6144, 128),
            AxisSegment(6144, 8192, 128),
            AxisSegment(8192, 10240, 128),
        ),
    )
    plan, source = _make_plan(
        name="attn_qkv.weight", shape=shape, quant=(14, "Q6_K", 256, 210), rule=rule, world_size=2
    )
    assert plan.slices[0].axis_ranges == (
        (0, 1024),
        (2048, 3072),
        (4096, 5120),
        (6144, 7168),
        (8192, 9216),
    )
    assert plan.slices[1].axis_ranges == (
        (1024, 2048),
        (3072, 4096),
        (5120, 6144),
        (7168, 8192),
        (9216, 10240),
    )
    for shard_slice in plan.slices:
        assert shard_slice.local_shape == (5 * 1024, 512)
    _round_trip(plan, source)


def test_unit_shards_replicated_and_owner_kinds():
    replicated, source = _make_plan(
        name="norm",
        shape=(5120,),
        quant=(0, "F32", 1, 4),
        rule=TensorShardRule(kind=REPLICATED, replicated=True),
        world_size=2,
    )
    for shard_slice in replicated.slices:
        assert shard_slice.replicated is True
        assert shard_slice.local_nbytes == source.size
    _round_trip(replicated, source)

    owner, source = _make_plan(
        name="token_embd.weight",
        shape=(4096, 512),
        quant=(14, "Q6_K", 256, 210),
        rule=TensorShardRule(kind=OWNER, owner_rank=1),
        world_size=2,
    )
    assert owner.slices[0].local_nbytes == 0
    assert owner.slices[1].local_nbytes == source.size
    assert owner.slices[0].local_shape == (0, 0)
    _round_trip(owner, source)


def test_unit_shards_row_split_rejects_block_unaligned_cut():
    with pytest.raises(ShardPlanError, match="not a multiple of the 256-element quant block"):
        _make_plan(
            name="w",
            shape=(32, 256 + 128),
            quant=(12, "Q4_K", 256, 144),
            rule=TensorShardRule(kind=ROW, axis=1),
            world_size=2,
        )


def test_unit_shards_row_split_rejects_indivisible_axis():
    with pytest.raises(ShardPlanError, match="not divisible by world size 3"):
        _make_plan(
            name="w",
            shape=(32, 1024),
            quant=(12, "Q4_K", 256, 144),
            rule=TensorShardRule(kind=ROW, axis=1),
            world_size=3,
        )


def test_unit_shards_group_split_rejects_ragged_groups():
    """Uneven head counts would give ranks different local geometry."""

    with pytest.raises(ShardPlanError, match="does not divide evenly across 2 ranks"):
        _make_plan(
            name="w",
            shape=(3 * 128, 512),
            quant=(0, "F32", 1, 4),
            rule=TensorShardRule(kind=GROUP, axis=0, segments=(AxisSegment(0, 3 * 128, 128),)),
            world_size=2,
        )


def test_unit_shards_partition_groups_replicates_when_ranks_exceed_groups():
    segment = AxisSegment(0, 4 * 256, 256)
    ranges = partition_groups(segment, 8)
    assert len(ranges) == 8
    assert ranges[:4] == [(0, 256), (256, 512), (512, 768), (768, 1024)]
    assert ranges[4:] == ranges[:4]
    plan, source = _make_plan(
        name="attn_k.weight",
        shape=(4 * 256, 512),
        quant=(12, "Q4_K", 256, 144),
        rule=TensorShardRule(kind=GROUP, axis=0, segments=(segment,)),
        world_size=8,
    )
    for shard_slice in plan.slices:
        assert shard_slice.local_shape == (256, 512)
    # Uniform replication is legal; the source payload still round trips.
    _round_trip(plan, source)


def test_unit_shards_coverage_rejects_gap_overlap_and_ragged_replication():
    def plan_with(ranges_by_rank, *, nbytes=1024, source_bytes=1024):
        slices = []
        for rank, ranges in enumerate(ranges_by_rank):
            segments = tuple(
                ShardSegment(source_offset=start, nbytes=stop - start) for start, stop in ranges
            )
            declared = sum(seg.nbytes for seg in segments)
            slices.append(
                TensorShardSlice(
                    rank=rank,
                    axis_ranges=tuple(ranges),
                    local_shape=(declared // 4,),
                    local_nbytes=declared,
                    segments=segments,
                )
            )
        return TensorShardPlan(
            name="w",
            kind=COLUMN,
            axis=0,
            source_shape=(nbytes,),
            source_nbytes=source_bytes,
            quant_type="F32",
            block_size=1,
            type_size=4,
            slices=tuple(slices),
        )

    validate_plan_coverage(plan_with([[(0, 512)], [(512, 1024)]]))
    with pytest.raises(ShardPlanError, match="gap"):
        validate_plan_coverage(plan_with([[(0, 512)], [(600, 1024)]]))
    with pytest.raises(ShardPlanError, match="non-uniform shard coverage"):
        validate_plan_coverage(plan_with([[(0, 512)], [(512, 1024)], [(0, 512)]]))
    with pytest.raises(ShardPlanError, match="expected 1024"):
        validate_plan_coverage(plan_with([[(0, 512)], [(512, 900)]]))
    # A ragged third rank (partial replication) is not uniform.
    with pytest.raises(ShardPlanError, match="non-uniform shard coverage"):
        validate_plan_coverage(plan_with([[(0, 512)], [(512, 1024)], [(0, 256)]]))
    # Whole-tensor replication on two ranks is uniform and therefore legal.
    validate_plan_coverage(plan_with([[(0, 512)], [(512, 1024)], [(0, 256)], [(256, 1024)]]))


def test_unit_shards_local_shape_and_bytes_must_agree():
    with pytest.raises(ShardPlanError, match="implies .* bytes, descriptor says"):
        plan = TensorShardPlan(
            name="w",
            kind=COLUMN,
            axis=0,
            source_shape=(8,),
            source_nbytes=32,
            quant_type="F32",
            block_size=1,
            type_size=4,
            slices=(
                TensorShardSlice(
                    rank=0,
                    axis_ranges=((0, 8),),
                    local_shape=(8,),
                    local_nbytes=24,
                    segments=(ShardSegment(source_offset=0, nbytes=24),),
                ),
            ),
        )
        validate_plan_coverage(plan)


def test_unit_shards_materialize_rejects_out_of_range_segment():
    source = _payload(64)
    shard_slice = TensorShardSlice(
        rank=0,
        axis_ranges=((0, 8),),
        local_shape=(8,),
        local_nbytes=32,
        segments=(ShardSegment(source_offset=48, nbytes=32),),
    )
    with pytest.raises(ShardPlanError, match="exceeds source payload size"):
        materialize_slice(source, shard_slice)


def test_unit_shards_reconstruct_detects_missing_bytes():
    plan, source = _make_plan(
        name="w",
        shape=(64, 64),
        quant=(0, "F32", 1, 4),
        rule=TensorShardRule(kind=COLUMN, axis=0, segments=(AxisSegment(0, 64, 1),)),
        world_size=2,
    )
    payloads = {shard_slice.rank: materialize_slice(source, shard_slice) for shard_slice in plan.slices}
    payloads[1] = payloads[1][:-4]
    with pytest.raises(ShardPlanError, match="expected"):
        reconstruct_tensor(_manifest_for(plan), "w", payloads)


def test_unit_shards_row_split_layout_rejects_bad_widths():
    with pytest.raises(ShardPlanError, match="byte widths must be positive"):
        RowSplitLayout(
            source_row_bytes=0,
            local_row_bytes=8,
            ranges=((0, 8),),
            byte_ranges=((0, 8),),
        )
    with pytest.raises(ShardPlanError, match="must pair up"):
        RowSplitLayout(
            source_row_bytes=16,
            local_row_bytes=8,
            ranges=((0, 8),),
            byte_ranges=(),
        )


def test_unit_shards_rule_validation():
    with pytest.raises(ShardPlanError, match="unknown shard kind"):
        TensorShardRule(kind="diagonal")
    with pytest.raises(ShardPlanError, match="requires axis-0 segments"):
        TensorShardRule(kind=COLUMN)
    with pytest.raises(ShardPlanError, match="row rule must split logical axis 1"):
        TensorShardRule(kind=ROW, axis=0)
    with pytest.raises(ShardPlanError, match="owner_rank must be non-negative"):
        TensorShardRule(kind=OWNER, owner_rank=-1)


def test_unit_shards_axis_segment_validation():
    with pytest.raises(ShardPlanError, match="invalid axis segment"):
        AxisSegment(5, 1)
    with pytest.raises(ShardPlanError, match="group must be positive"):
        AxisSegment(0, 8, 0)


# ---------------------------------------------------------------------------
# 2. Real-model structure
# ---------------------------------------------------------------------------

_gguf_available = Path(GGUF_PATH).exists()
requires_model = pytest.mark.skipif(
    not _gguf_available, reason=f"GGUF model file not available: {GGUF_PATH}"
)


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


@pytest.fixture(scope="module")
def model_info():
    from hipengine.loading.gguf import scan_gguf

    return scan_gguf(GGUF_PATH)


@pytest.fixture(scope="module")
def model_config(model_info):
    from hipengine.loading.qwen35_gguf import qwen35_gguf_config_from_metadata

    return qwen35_gguf_config_from_metadata(model_info)


@pytest.fixture(scope="module")
def manifest_n2(model_info):
    return build_shard_manifest(model_info, world_size=2)


@requires_model
def test_unit_shards_manifest_is_deterministic_and_hashed(model_info):
    first = build_shard_manifest(model_info, world_size=2)
    second = build_shard_manifest(model_info, world_size=2)
    assert first.manifest_hash() == second.manifest_hash()
    assert first.to_dict() == second.to_dict()
    # A different degree is a different shard identity.
    assert build_shard_manifest(model_info, world_size=4).manifest_hash() != first.manifest_hash()


@requires_model
def test_unit_shards_manifest_covers_every_ar_tensor_once(manifest_n2):
    world_size = manifest_n2.world_size
    sharded = [plan for plan in manifest_n2.tensors if plan.kind not in {REPLICATED, OWNER}]
    replicated = [plan for plan in manifest_n2.tensors if plan.kind == REPLICATED]
    owned = [plan for plan in manifest_n2.tensors if plan.kind == OWNER]
    # Sharded tensors are split exactly once across ranks...
    assert sum(plan.source_nbytes for plan in sharded) == sum(
        shard_slice.local_nbytes for plan in sharded for shard_slice in plan.slices
    )
    # ...replicated tensors appear whole on every rank...
    for plan in replicated:
        for shard_slice in plan.slices:
            assert shard_slice.local_nbytes == plan.source_nbytes
    # ...single-owner tensors appear whole on exactly one rank...
    for plan in owned:
        assert sum(1 for shard_slice in plan.slices if shard_slice.local_nbytes) == 1
    # ...and the per-rank totals are exactly those three contributions.
    expected = (
        sum(plan.source_nbytes for plan in sharded)
        + sum(plan.source_nbytes for plan in replicated) * world_size
        + sum(plan.source_nbytes for plan in owned)
    )
    assert sum(manifest_n2.rank_bytes(rank) for rank in range(world_size)) == expected
    assert manifest_n2.tensor_names == tuple(plan.name for plan in manifest_n2.tensors)
    assert "output.weight" in manifest_n2.tensor_names
    assert not any(name.startswith("blk.64.") for name in manifest_n2.tensor_names)


@requires_model
def test_unit_shards_qwen38_head_geometry_per_rank(manifest_n2, model_config):
    """Each rank gets whole heads, whole GDN groups, and aligned columns."""

    assert manifest_n2.plan_for("blk.3.attn_q.weight").slices[0].local_shape == (
        2 * model_config.head_count * model_config.key_length // 2,
        model_config.hidden_size,
    )
    attn_k = manifest_n2.plan_for("blk.3.attn_k.weight")
    assert attn_k.slices[0].local_shape[0] == model_config.head_count_kv * model_config.key_length // 2
    assert attn_k.slices[0].axis_ranges == ((0, 512),)
    assert attn_k.slices[1].axis_ranges == ((512, 1024),)

    attn_output = manifest_n2.plan_for("blk.3.attn_output.weight")
    assert attn_output.slices[0].axis_ranges == ((0, 3072),)
    assert attn_output.slices[1].axis_ranges == ((3072, 6144),)

    # GDN: q and k blocks are 16 heads wide, the value block is 48 heads in
    # llama.cpp tiled order, so the split is per tile, not one contiguous run.
    qkv = manifest_n2.plan_for("blk.0.attn_qkv.weight")
    assert qkv.slices[0].axis_ranges == (
        (0, 1024),
        (2048, 3072),
        (4096, 5120),
        (6144, 7168),
        (8192, 9216),
    )
    assert qkv.slices[1].axis_ranges == (
        (1024, 2048),
        (3072, 4096),
        (5120, 6144),
        (7168, 8192),
        (9216, 10240),
    )
    assert qkv.slices[0].local_shape[0] == 5120
    assert manifest_n2.plan_for("blk.0.ssm_conv1d.weight").slices[0].axis_ranges == qkv.slices[0].axis_ranges
    assert manifest_n2.plan_for("blk.0.attn_gate.weight").slices[0].axis_ranges == (
        (0, 1024),
        (2048, 3072),
        (4096, 5120),
    )
    assert manifest_n2.plan_for("blk.0.ssm_a").slices[0].axis_ranges == (
        (0, 8),
        (16, 24),
        (32, 40),
    )
    # ssm_out is a row split on the value-head axis; every local range must be
    # a whole number of 256-element quant blocks.
    ssm_out = manifest_n2.plan_for("blk.0.ssm_out.weight")
    assert ssm_out.slices[0].axis_ranges == ((0, 1024), (2048, 3072), (4096, 5120))
    for shard_slice in ssm_out.slices:
        for start, stop in shard_slice.axis_ranges:
            assert (stop - start) % 256 == 0


@requires_model
def test_unit_shards_gdn_head_map_matches_kernel_pairing(model_config):
    """The planner's head map must agree with the kernel's v_head % k_heads."""

    for rank in range(2):
        head_map = gdn_head_map(model_config, rank, 2)
        assert len(head_map.v_heads_for()) == head_map.local_v_heads
        assert head_map.local_v_heads == 24
        assert head_map.local_k_heads == 8
        for local_v in range(head_map.local_v_heads):
            global_v = head_map.v_heads_for()[local_v]
            assert head_map.local_k_head(local_v) == global_v % head_map.k_heads
    with pytest.raises(ShardPlanError, match="outside world size"):
        gdn_head_map(model_config, 2, 2)


@requires_model
def test_unit_shards_degree_refusals_name_the_offending_tensor(model_info):
    names = {tensor.name for tensor in model_info.tensors}
    with pytest.raises(ShardPlanError) as error:
        build_shard_manifest(model_info, world_size=3)
    message = str(error.value)
    assert message.split(":")[0] in names, message
    assert "divide evenly" in message or "divisible" in message
    # N=8 divides the head geometry but not the 17408-column MLP input axis.
    with pytest.raises(ShardPlanError) as error:
        build_shard_manifest(model_info, world_size=8)
    message = str(error.value)
    assert message.startswith("blk.0.ffn_down.weight:"), message
    assert "quant block" in message


@requires_model
def test_unit_shards_supported_degrees_and_rank_bytes(model_info):
    totals = {}
    for degree in (1, 2, 4):
        manifest = build_shard_manifest(model_info, world_size=degree)
        totals[degree] = [manifest.rank_bytes(rank) for rank in range(degree)]
    # Every non-owner tensor halves exactly; the owner tensors stay whole.
    manifest2 = build_shard_manifest(model_info, world_size=2)
    owned = sum(plan.source_nbytes for plan in manifest2.tensors if plan.kind == OWNER)
    assert totals[2][0] - totals[2][1] == owned
    # A higher degree strictly reduces the largest rank's resident bytes.
    assert max(totals[2]) < totals[1][0]
    assert max(totals[4]) < max(totals[2])
    # Choosing the other owner rank moves exactly that payload.
    flipped = build_shard_manifest(model_info, world_size=2, owner_rank=1)
    assert flipped.rank_bytes(0) == totals[2][1]
    assert flipped.rank_bytes(1) == totals[2][0]
    assert any("rank 1" in note for note in flipped.notes)


@requires_model
def test_unit_shards_model_bytes_are_bit_preserved_for_every_kind(model_info, manifest_n2):
    from hipengine.loading.gguf import GGUFReader

    reader = GGUFReader(GGUF_PATH)
    sample = [
        "token_embd.weight",
        "output.weight",
        "output_norm.weight",
        "blk.0.attn_qkv.weight",
        "blk.0.attn_gate.weight",
        "blk.0.ssm_conv1d.weight",
        "blk.0.ssm_a",
        "blk.0.ssm_dt.bias",
        "blk.0.ssm_alpha.weight",
        "blk.0.ssm_norm.weight",
        "blk.0.ssm_out.weight",
        "blk.3.attn_q.weight",
        "blk.3.attn_k.weight",
        "blk.3.attn_output.weight",
        "blk.3.ffn_down.weight",
        "blk.3.ffn_gate.weight",
    ]
    for name in sample:
        plan = manifest_n2.plan_for(name)
        tensor = reader.tensor_info(name)
        source = np.memmap(
            reader.path, mode="r", dtype=np.uint8, offset=tensor.data_offset, shape=(tensor.nbytes,)
        )
        payloads = {
            shard_slice.rank: materialize_slice(source, shard_slice) for shard_slice in plan.slices
        }
        rebuilt = reconstruct_tensor(manifest_n2, name, payloads)
        assert np.array_equal(rebuilt, np.asarray(source)), name


@requires_model
def test_unit_shards_quant_types_are_never_repacked(manifest_n2):
    """Local byte widths must be a whole number of source quant blocks."""

    for plan in manifest_n2.tensors:
        if plan.kind in {REPLICATED, OWNER}:
            continue
        for shard_slice in plan.slices:
            if shard_slice.row_split is not None:
                assert shard_slice.row_split.local_row_bytes % plan.type_size == 0
            for segment in shard_slice.iter_segments():
                assert segment.nbytes % plan.type_size == 0, (plan.name, plan.quant_type)


# ---------------------------------------------------------------------------
# 3. Layer fixtures: planned shards must reproduce the unsharded layer
# ---------------------------------------------------------------------------


def _fixture_config(**overrides) -> SimpleNamespace:
    config = dict(
        architecture="qwen35",
        block_count=2,
        hidden_size=16,
        feed_forward_length=32,
        head_count=4,
        head_count_kv=2,
        key_length=8,
        value_length=8,
        ssm_inner_size=24,
        ssm_group_count=2,
        ssm_state_size=4,
        ssm_time_step_rank=6,
        ssm_conv_kernel=3,
        is_moe=False,
    )
    config.update(overrides)
    return SimpleNamespace(**config)


def _fixture_plan(config: SimpleNamespace, name: str, shape: tuple[int, ...], world_size: int = 2):
    rule = shard_rule_for_tensor(name, config=config)
    nbytes = int(np.prod(shape)) * 4
    return build_tensor_shard_plan(
        name=name,
        shape=shape,
        nbytes=nbytes,
        quant_type_id=0,
        quant_type_name="F32",
        rule=rule,
        world_size=world_size,
    )


def _fixture_values(shape: tuple[int, ...], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(shape).astype(np.float32)


def _rows_for(plan: TensorShardPlan, rank: int, values: np.ndarray) -> np.ndarray:
    """Gather a rank's local rows in the order the plan materializes them."""

    return np.concatenate([values[start:stop] for start, stop in plan.slice_for(rank).axis_ranges])


def _columns_for(plan: TensorShardPlan, rank: int, values: np.ndarray) -> np.ndarray:
    return np.concatenate([values[:, start:stop] for start, stop in plan.slice_for(rank).axis_ranges], axis=1)


def test_unit_fixture_mlp_column_and_row_parallel_matches_full():
    """Two-stage MLP: column split, local activation, row split, all-reduce."""

    config = _fixture_config()
    hidden, ffn = config.hidden_size, config.feed_forward_length
    tokens = 3
    x = _fixture_values((tokens, hidden), seed=1)
    gate = _fixture_values((ffn, hidden), seed=2)
    up = _fixture_values((ffn, hidden), seed=3)
    down = _fixture_values((hidden, ffn), seed=4)

    def silu(t):
        return t / (1.0 + np.exp(-t))

    full = down @ (silu(gate @ x.T) * (up @ x.T))

    gate_plan = _fixture_plan(config, "blk.0.ffn_gate.weight", (ffn, hidden))
    up_plan = _fixture_plan(config, "blk.0.ffn_up.weight", (ffn, hidden))
    down_plan = _fixture_plan(config, "blk.0.ffn_down.weight", (hidden, ffn))

    partials = []
    for rank in range(2):
        local_gate = _rows_for(gate_plan, rank, gate)
        local_up = _rows_for(up_plan, rank, up)
        local_down = _columns_for(down_plan, rank, down)
        assert local_gate.shape == (ffn // 2, hidden)
        assert local_down.shape == (hidden, ffn // 2)
        partials.append(local_down @ (silu(local_gate @ x.T) * (local_up @ x.T)))
    assert np.allclose(sum(partials), full, atol=1e-4)
    # Each rank's partial contribution must be non-trivial, otherwise the test
    # would pass on a plan that silently replicates everything.
    assert not np.allclose(partials[0], partials[1])
    assert np.allclose(partials[0] + partials[1], full, atol=1e-4)


def test_unit_fixture_full_attention_head_split_matches_full():
    """Interleaved q/gate rows, GQA pairing, and o_proj row split."""

    config = _fixture_config()
    heads, kv_heads, head_dim, hidden = (
        config.head_count,
        config.head_count_kv,
        config.key_length,
        config.hidden_size,
    )
    tokens = 4
    x = _fixture_values((tokens, hidden), seed=11)
    q_proj = _fixture_values((2 * heads * head_dim, hidden), seed=12)
    k_proj = _fixture_values((kv_heads * head_dim, hidden), seed=13)
    v_proj = _fixture_values((kv_heads * head_dim, hidden), seed=14)
    o_proj = _fixture_values((hidden, heads * head_dim), seed=15)

    def attention(q_rows, k_rows, v_rows, gate_rows):
        """q_rows/gate_rows are per-head [q, gate]; returns [tokens, heads*head_dim]."""

        q = q_rows[:, :, 0, :]
        gate = gate_rows
        k = k_rows
        v = v_rows
        repeat = q.shape[1] // k.shape[1]
        k = np.repeat(k, repeat, axis=1)
        v = np.repeat(v, repeat, axis=1)
        scores = np.einsum("thd,shd->ths", q, k) / np.sqrt(head_dim)
        causal = np.tril(np.ones((tokens, tokens), dtype=bool))
        scores = np.where(causal[:, None, :], scores, -np.inf)
        weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
        weights /= weights.sum(axis=-1, keepdims=True)
        out = np.einsum("ths,shd->thd", weights, v)
        return (out * (1.0 / (1.0 + np.exp(-gate)))).reshape(tokens, -1)

    q_heads = q_proj.reshape(heads, 2, head_dim, hidden)
    k_heads = k_proj.reshape(kv_heads, head_dim, hidden)
    v_heads = v_proj.reshape(kv_heads, head_dim, hidden)

    def project(weights):
        return np.einsum("oh,th->to", weights, x)

    full_q = project(q_heads.reshape(heads * 2 * head_dim, hidden)).reshape(
        tokens, heads, 2, head_dim
    )
    full_k = project(k_heads.reshape(kv_heads * head_dim, hidden)).reshape(
        tokens, kv_heads, head_dim
    )
    full_v = project(v_heads.reshape(kv_heads * head_dim, hidden)).reshape(
        tokens, kv_heads, head_dim
    )
    full = attention(full_q, full_k, full_v, full_q[:, :, 1, :]) @ o_proj.T

    q_plan = _fixture_plan(config, "blk.0.attn_q.weight", (2 * heads * head_dim, hidden))
    k_plan = _fixture_plan(config, "blk.0.attn_k.weight", (kv_heads * head_dim, hidden))
    v_plan = _fixture_plan(config, "blk.0.attn_v.weight", (kv_heads * head_dim, hidden))
    o_plan = _fixture_plan(config, "blk.0.attn_output.weight", (hidden, heads * head_dim))

    partials = []
    for rank in range(2):
        local_heads = heads // 2
        local_q = project(_rows_for(q_plan, rank, q_proj.reshape(heads * 2 * head_dim, hidden)))
        local_q = local_q.reshape(tokens, local_heads, 2, head_dim)
        local_k = project(_rows_for(k_plan, rank, k_proj.reshape(kv_heads * head_dim, hidden)))
        local_k = local_k.reshape(tokens, kv_heads // 2, head_dim)
        local_v = project(_rows_for(v_plan, rank, v_proj.reshape(kv_heads * head_dim, hidden)))
        local_v = local_v.reshape(tokens, kv_heads // 2, head_dim)
        local_o = _columns_for(o_plan, rank, o_proj)
        local_out = attention(local_q, local_k, local_v, local_q[:, :, 1, :])
        assert local_out.shape == (tokens, local_heads * head_dim)
        assert local_o.shape == (hidden, local_heads * head_dim)
        partials.append(local_out @ local_o.T)
    assert np.allclose(sum(partials), full, atol=1e-4)


def test_unit_fixture_gdn_tiled_head_split_matches_full():
    """GDN shards must reproduce the kernel's tiled value-head recurrence."""

    config = _fixture_config()
    hidden = config.hidden_size
    head_k = config.ssm_state_size
    head_v = config.ssm_inner_size // config.ssm_time_step_rank
    k_heads = config.ssm_group_count
    v_heads = config.ssm_time_step_rank
    tiles = v_heads // k_heads
    key_width = k_heads * head_k
    tokens = 5

    qkv_plan = _fixture_plan(
        config, "blk.0.attn_qkv.weight", (2 * key_width + v_heads * head_v, hidden)
    )
    gate_plan = _fixture_plan(config, "blk.0.attn_gate.weight", (v_heads * head_v, hidden))
    alpha_plan = _fixture_plan(config, "blk.0.ssm_alpha.weight", (v_heads, hidden))
    beta_plan = _fixture_plan(config, "blk.0.ssm_beta.weight", (v_heads, hidden))
    a_plan = _fixture_plan(config, "blk.0.ssm_a", (v_heads,))
    dt_plan = _fixture_plan(config, "blk.0.ssm_dt.bias", (v_heads,))
    out_plan = _fixture_plan(config, "blk.0.ssm_out.weight", (hidden, v_heads * head_v))

    x = _fixture_values((tokens, hidden), seed=21)
    qkv_weight = _fixture_values((2 * key_width + v_heads * head_v, hidden), seed=22)
    gate_weight = _fixture_values((v_heads * head_v, hidden), seed=23)
    alpha_weight = _fixture_values((v_heads, hidden), seed=24)
    beta_weight = _fixture_values((v_heads, hidden), seed=25)
    a_log = _fixture_values((v_heads,), seed=26)
    dt_bias = _fixture_values((v_heads,), seed=27)
    out_weight = _fixture_values((hidden, v_heads * head_v), seed=28)

    def softplus(t):
        return np.log1p(np.exp(-np.abs(t))) + np.maximum(t, 0.0)

    def l2norm(t):
        norm = np.sqrt(np.maximum(np.sum(t * t, axis=-1, keepdims=True), 0.0))
        return t / np.maximum(norm, 1e-6)

    def recurrence(
        conv_out, gate, alpha, beta, a_log_local, dt_bias_local, *, n_k_heads, n_v_heads
    ):
        """Replicate the kernel's tiled value-head recurrence for one rank's heads.

        ``conv_out`` is ``[tokens, channels]`` in local q/k/v order, where the q
        and k blocks are ``n_k_heads * head_k`` wide and the value block is
        ``n_v_heads * head_v`` wide. The kernel pairs value head ``v`` with key
        head ``v % n_k_heads``.
        """

        local_key_width = n_k_heads * head_k
        assert conv_out.shape[1] == 2 * local_key_width + n_v_heads * head_v
        q_block = conv_out[:, :local_key_width]
        k_block = conv_out[:, local_key_width : 2 * local_key_width]
        v_block = conv_out[:, 2 * local_key_width :]
        state = np.zeros((n_v_heads, head_k, head_v), dtype=np.float32)
        outputs = np.empty((tokens, n_v_heads, head_v), dtype=np.float32)
        for v_head in range(n_v_heads):
            k_head = v_head % n_k_heads
            q_vec = q_block[:, k_head * head_k : (k_head + 1) * head_k]
            k_vec = k_block[:, k_head * head_k : (k_head + 1) * head_k]
            q_norm = l2norm(q_vec) * (1.0 / np.sqrt(head_k))
            k_norm = l2norm(k_vec)
            decay = np.exp(
                -np.exp(a_log_local[v_head]) * softplus(alpha[:, v_head] + dt_bias_local[v_head])
            )
            beta_t = 1.0 / (1.0 + np.exp(-beta[:, v_head]))
            for t in range(tokens):
                state[v_head] *= decay[t]
                kv_mem = np.einsum("d,df->f", k_norm[t], state[v_head])
                delta = (v_block[t, v_head * head_v : (v_head + 1) * head_v] - kv_mem) * beta_t[t]
                state[v_head] += np.outer(k_norm[t], delta)
                outputs[t, v_head] = np.einsum("d,df->f", q_norm[t], state[v_head])
        variance = np.mean(outputs * outputs, axis=-1, keepdims=True)
        normalized = outputs / np.sqrt(variance + 1e-6)
        gate_heads = gate.reshape(tokens, n_v_heads, head_v)
        gated = (normalized * gate_heads) * (1.0 / (1.0 + np.exp(-gate_heads)))
        return gated.reshape(tokens, n_v_heads * head_v)

    # Full layer.
    full_qkv = qkv_weight @ x.T
    full_gate = gate_weight @ x.T
    full_alpha = alpha_weight @ x.T
    full_beta = beta_weight @ x.T
    full = recurrence(
        full_qkv.T,
        full_gate.T,
        full_alpha.T,
        full_beta.T,
        a_log,
        dt_bias,
        n_k_heads=k_heads,
        n_v_heads=v_heads,
    )
    full_out = full @ out_weight.T

    partials = []
    for rank in range(2):
        head_map = gdn_head_map(config, rank, 2)
        local_qkv = _rows_for(qkv_plan, rank, qkv_weight) @ x.T
        local_gate = _rows_for(gate_plan, rank, gate_weight) @ x.T
        local_alpha = _rows_for(alpha_plan, rank, alpha_weight) @ x.T
        local_beta = _rows_for(beta_plan, rank, beta_weight) @ x.T
        local_a = _rows_for(a_plan, rank, a_log)
        local_dt = _rows_for(dt_plan, rank, dt_bias)
        assert local_a.shape == (head_map.local_v_heads,)
        assert local_dt.shape == (head_map.local_v_heads,)
        local_out = recurrence(
            local_qkv.T,
            local_gate.T,
            local_alpha.T,
            local_beta.T,
            local_a,
            local_dt,
            n_k_heads=head_map.local_k_heads,
            n_v_heads=head_map.local_v_heads,
        )
        assert local_out.shape == (tokens, head_map.local_v_heads * head_v)

        # The value-head axis of the output must line up with the local value
        # heads in the order the planner materializes ssm_out columns.
        columns = np.concatenate([head_v * v + np.arange(head_v) for v in head_map.v_heads_for()])
        local_out_weight = _columns_for(out_plan, rank, out_weight)
        assert np.array_equal(columns, np.concatenate(
            [np.arange(start, stop) for start, stop in out_plan.slice_for(rank).axis_ranges]
        ))
        # Local heads must reproduce the full result for exactly those heads.
        assert np.allclose(local_out, full[:, columns], atol=1e-5)
        partials.append(local_out @ local_out_weight.T)

    assert np.allclose(sum(partials), full_out, atol=1e-4)


def test_unit_fixture_gdn_local_ranges_cover_every_head_exactly_once():
    """Sanity: the fixture's head map tiles the value heads."""

    config = _fixture_config()
    seen: list[int] = []
    for rank in range(2):
        seen.extend(gdn_head_map(config, rank, 2).v_heads_for())
    assert sorted(seen) == list(range(config.ssm_time_step_rank))
    assert len(seen) == len(set(seen))


def test_unit_fixture_attention_output_columns_match_local_heads():
    """o_proj row ranges must be exactly the local query-head blocks."""

    config = _fixture_config()
    plan = _fixture_plan(
        config, "blk.0.attn_output.weight", (config.hidden_size, config.head_count * config.key_length)
    )
    per_rank_heads = config.head_count // 2
    for rank in range(2):
        ranges = plan.slice_for(rank).axis_ranges
        # One contiguous range that is a whole number of query heads wide.
        assert ranges == ((rank * per_rank_heads * config.key_length,
                           (rank + 1) * per_rank_heads * config.key_length),)
        for start, stop in ranges:
            assert (stop - start) % config.key_length == 0


# ---------------------------------------------------------------------------
# 4. Non-power-of-two degrees on geometry that admits them
# ---------------------------------------------------------------------------


def _divisible_config(**overrides) -> SimpleNamespace:
    """Geometry divisible by 2, 3 and 4 so N=1..4 are all admissible."""

    return _fixture_config(
        hidden_size=8,
        feed_forward_length=24,
        head_count=12,
        head_count_kv=12,
        key_length=4,
        value_length=4,
        ssm_inner_size=48,
        ssm_group_count=12,
        ssm_state_size=2,
        ssm_time_step_rank=24,
        **overrides,
    )


def _quant_aligned_config(**overrides) -> SimpleNamespace:
    """Geometry where every split axis is also a multiple of a 256-block."""

    return _fixture_config(
        hidden_size=256,
        feed_forward_length=3072,
        head_count=12,
        head_count_kv=12,
        key_length=256,
        value_length=256,
        ssm_inner_size=6144,
        ssm_group_count=12,
        ssm_state_size=256,
        ssm_time_step_rank=24,
        ssm_conv_kernel=4,
        **overrides,
    )


def _synthetic_inventory(config: SimpleNamespace) -> tuple[tuple[str, tuple[int, ...]], ...]:
    """Every tensor family the planner knows, at fixture scale."""

    hidden = config.hidden_size
    ffn = config.feed_forward_length
    head, kv_head, head_dim = config.head_count, config.head_count_kv, config.key_length
    key_width = config.ssm_group_count * config.ssm_state_size
    inner = config.ssm_inner_size
    return (
        ("blk.0.attn_norm.weight", (hidden,)),
        ("blk.0.post_attention_norm.weight", (hidden,)),
        ("blk.0.attn_qkv.weight", (2 * key_width + inner, hidden)),
        ("blk.0.attn_gate.weight", (inner, hidden)),
        ("blk.0.ssm_conv1d.weight", (2 * key_width + inner, config.ssm_conv_kernel)),
        ("blk.0.ssm_alpha.weight", (config.ssm_time_step_rank, hidden)),
        ("blk.0.ssm_beta.weight", (config.ssm_time_step_rank, hidden)),
        ("blk.0.ssm_a", (config.ssm_time_step_rank,)),
        ("blk.0.ssm_dt.bias", (config.ssm_time_step_rank,)),
        ("blk.0.ssm_norm.weight", (config.ssm_state_size,)),
        ("blk.0.ssm_out.weight", (hidden, inner)),
        ("blk.0.ffn_gate.weight", (ffn, hidden)),
        ("blk.0.ffn_up.weight", (ffn, hidden)),
        ("blk.0.ffn_down.weight", (hidden, ffn)),
        ("blk.3.attn_q.weight", (2 * head * head_dim, hidden)),
        ("blk.3.attn_k.weight", (kv_head * head_dim, hidden)),
        ("blk.3.attn_v.weight", (kv_head * head_dim, hidden)),
        ("blk.3.attn_q_norm.weight", (head_dim,)),
        ("blk.3.attn_k_norm.weight", (head_dim,)),
        ("blk.3.attn_output.weight", (hidden, head * head_dim)),
        ("output_norm.weight", (hidden,)),
    )


@pytest.mark.parametrize("world_size", [1, 2, 3, 4])
@pytest.mark.parametrize(
    "quant",
    [(0, "F32", 1, 4), (12, "Q4_K", 256, 144), (14, "Q6_K", 256, 210)],
    ids=["F32", "Q4_K", "Q6_K"],
)
def test_unit_shards_all_degrees_round_trip_on_divisible_geometry(world_size, quant):
    """N=1,2,3,4 must plan and reconstruct every tensor family bit-exactly.

    N=3 is the non-power-of-two degree the plan explicitly requires. Tensors
    whose split axis is narrower than one 256-element quant block (the F32
    scalar/norm tensors in the real model) are skipped for quantized layouts.
    """

    config = _quant_aligned_config()
    type_id, type_name, block_size, type_size = quant
    checked = 0
    for name, shape in _synthetic_inventory(config):
        rows = int(shape[0])
        cols = int(shape[1]) if len(shape) > 1 else 1
        if cols % block_size:
            continue
        nbytes = rows * (cols // block_size) * type_size
        rule = shard_rule_for_tensor(name, config=config)
        plan = build_tensor_shard_plan(
            name=name,
            shape=shape,
            nbytes=nbytes,
            quant_type_id=type_id,
            quant_type_name=type_name,
            rule=rule,
            world_size=world_size,
        )
        source = _payload(nbytes, seed=abs(hash(name)) % 1000)
        _round_trip(plan, source)
        checked += 1
        if plan.kind in {REPLICATED, OWNER}:
            continue
        shapes = {tuple(shard_slice.local_shape) for shard_slice in plan.slices}
        assert len(shapes) == 1, (name, world_size, shapes)
        assert plan.slices[0].local_nbytes * world_size == plan.source_nbytes, (name, world_size)
    assert checked >= 12


def test_unit_shards_gdn_head_map_tiles_value_heads_at_n3():
    """Non-power-of-two degree keeps whole key heads and their tiled values."""

    config = _divisible_config()
    seen: list[int] = []
    for rank in range(3):
        head_map = gdn_head_map(config, rank, 3)
        assert head_map.local_k_heads == 4
        assert head_map.local_v_heads == 8
        seen.extend(head_map.v_heads_for())
        for local_v in range(head_map.local_v_heads):
            global_v = head_map.v_heads_for()[local_v]
            assert head_map.local_k_head(local_v) == global_v % head_map.k_heads
    assert sorted(seen) == list(range(config.ssm_time_step_rank))
    assert len(seen) == len(set(seen))


def test_unit_fixture_gdn_n3_matches_full():
    """The GDN fixture at N=3 must reproduce the unsharded layer."""

    config = _divisible_config()
    hidden = config.hidden_size
    head_k = config.ssm_state_size
    head_v = config.ssm_inner_size // config.ssm_time_step_rank
    k_heads = config.ssm_group_count
    v_heads = config.ssm_time_step_rank
    key_width = k_heads * head_k
    tokens = 3
    world_size = 3

    qkv_weight = _fixture_values((2 * key_width + v_heads * head_v, hidden), seed=31)
    alpha_weight = _fixture_values((v_heads, hidden), seed=32)
    beta_weight = _fixture_values((v_heads, hidden), seed=33)
    a_log = _fixture_values((v_heads,), seed=34)
    dt_bias = _fixture_values((v_heads,), seed=35)
    x = _fixture_values((tokens, hidden), seed=36)

    def softplus(t):
        return np.log1p(np.exp(-np.abs(t))) + np.maximum(t, 0.0)

    def l2norm(t):
        norm = np.sqrt(np.maximum(np.sum(t * t, axis=-1, keepdims=True), 0.0))
        return t / np.maximum(norm, 1e-6)

    def recurrence(conv_out, alpha, beta, a_local, dt_local, n_k_heads, n_v_heads):
        local_key = n_k_heads * head_k
        q_block = conv_out[:, :local_key]
        k_block = conv_out[:, local_key : 2 * local_key]
        v_block = conv_out[:, 2 * local_key :]
        state = np.zeros((n_v_heads, head_k, head_v), dtype=np.float32)
        outputs = np.empty((tokens, n_v_heads, head_v), dtype=np.float32)
        for v_head in range(n_v_heads):
            k_head = v_head % n_k_heads
            q_norm = l2norm(q_block[:, k_head * head_k : (k_head + 1) * head_k]) / np.sqrt(head_k)
            k_norm = l2norm(k_block[:, k_head * head_k : (k_head + 1) * head_k])
            decay = np.exp(-np.exp(a_local[v_head]) * softplus(alpha[:, v_head] + dt_local[v_head]))
            beta_t = 1.0 / (1.0 + np.exp(-beta[:, v_head]))
            for t in range(tokens):
                state[v_head] *= decay[t]
                kv_mem = np.einsum("d,df->f", k_norm[t], state[v_head])
                delta = (v_block[t, v_head * head_v : (v_head + 1) * head_v] - kv_mem) * beta_t[t]
                state[v_head] += np.outer(k_norm[t], delta)
                outputs[t, v_head] = np.einsum("d,df->f", q_norm[t], state[v_head])
        variance = np.mean(outputs * outputs, axis=-1, keepdims=True)
        return (outputs / np.sqrt(variance + 1e-6)).reshape(tokens, n_v_heads * head_v)

    full = recurrence(
        (qkv_weight @ x.T).T,
        (alpha_weight @ x.T).T,
        (beta_weight @ x.T).T,
        a_log,
        dt_bias,
        n_k_heads=k_heads,
        n_v_heads=v_heads,
    )
    qkv_plan = _fixture_plan(config, "blk.0.attn_qkv.weight", qkv_weight.shape, world_size)
    alpha_plan = _fixture_plan(config, "blk.0.ssm_alpha.weight", alpha_weight.shape, world_size)
    beta_plan = _fixture_plan(config, "blk.0.ssm_beta.weight", beta_weight.shape, world_size)
    a_plan = _fixture_plan(config, "blk.0.ssm_a", a_log.shape, world_size)
    dt_plan = _fixture_plan(config, "blk.0.ssm_dt.bias", dt_bias.shape, world_size)

    collected: dict[int, np.ndarray] = {}
    for rank in range(world_size):
        head_map = gdn_head_map(config, rank, world_size)
        local = recurrence(
            (_rows_for(qkv_plan, rank, qkv_weight) @ x.T).T,
            (_rows_for(alpha_plan, rank, alpha_weight) @ x.T).T,
            (_rows_for(beta_plan, rank, beta_weight) @ x.T).T,
            _rows_for(a_plan, rank, a_log),
            _rows_for(dt_plan, rank, dt_bias),
            n_k_heads=head_map.local_k_heads,
            n_v_heads=head_map.local_v_heads,
        )
        for local_v, global_v in enumerate(head_map.v_heads_for()):
            collected[global_v] = local[:, local_v * head_v : (local_v + 1) * head_v]
    reassembled = np.concatenate([collected[head] for head in range(v_heads)], axis=1)
    assert np.allclose(reassembled, full, atol=1e-5)


# ---------------------------------------------------------------------------
# 6. Streaming loader path: peak memory is one local tensor, not a full model
# ---------------------------------------------------------------------------


class _FakeReader:
    """Minimal GGUFReader stand-in: a real file plus per-tensor byte ranges."""

    def __init__(self, path: Path, spans: dict[str, tuple[int, int]]) -> None:
        self.path = path
        self._spans = spans
        self.tensor_info = self._tensor_info  # match the reader surface used by the module

    def _tensor_info(self, name: str):
        offset, nbytes = self._spans[name]
        return SimpleNamespace(data_offset=offset, nbytes=nbytes)


def _streaming_fixture(tmp_path: Path, *, tensors: int = 32, rows: int = 512, world_size: int = 2):
    """One backing file holding ``tensors`` independent Q8_0 tensors."""

    quant = (8, "Q8_0", 32, 34)
    type_size = quant[3]
    block_size = quant[2]
    per_tensor = rows * (256 // block_size) * type_size
    rng = np.random.default_rng(11)
    blob = rng.integers(0, 256, size=per_tensor * tensors, dtype=np.uint8)
    path = tmp_path / "stream.gguf"
    path.write_bytes(blob.tobytes())
    plans = []
    spans: dict[str, tuple[int, int]] = {}
    for index in range(tensors):
        name = f"blk.{index}.ffn_down.weight"
        plan, _ = _make_plan(
            name=name,
            shape=(rows, 256),
            quant=quant,
            rule=TensorShardRule(kind=COLUMN, axis=0, segments=(AxisSegment(start=0, stop=rows, group=1),)),
            world_size=world_size,
        )
        spans[name] = (index * per_tensor, per_tensor)
        plans.append(plan)
    manifest = ShardManifest(model_hash="stream", world_size=world_size, hidden_size=256, tensors=tuple(plans))
    return _FakeReader(path, spans), manifest, per_tensor


def test_unit_shards_streaming_iterator_holds_one_tensor_at_a_time(tmp_path: Path) -> None:
    """A rank materializes tensor by tensor, so no full-model copy is needed."""

    reader, manifest, per_tensor = _streaming_fixture(tmp_path)
    local_per_tensor = per_tensor // manifest.world_size
    live = 0
    peak_live = 0
    allocations: list[int] = []

    def tracking_allocator(nbytes: int, dtype=None):
        nonlocal live, peak_live
        live += int(nbytes)
        peak_live = max(peak_live, live)
        allocations.append(int(nbytes))
        return np.zeros(int(nbytes), dtype=np.uint8)

    seen: list[str] = []
    total = 0
    for plan, payload in iter_rank_payloads(reader, manifest, rank=1):
        seen.append(plan.name)
        total += int(payload.size)
        # The payload is the only live buffer while a loader would upload it.
        live = int(payload.size)
        peak_live = max(peak_live, live)
    assert seen == [plan.name for plan in manifest.tensors]
    assert total == manifest.rank_bytes(1)
    assert total > 2 * per_tensor
    assert peak_live == local_per_tensor, "streaming path held more than one local tensor"
    assert allocations == []  # the module used its own allocator, not ours
    # The oracle path is the one that holds everything; the loader path must not.
    payloads = materialize_manifest(reader, manifest)
    assert sum(int(payload.size) for payload in payloads[1].values()) == total
    del payloads


def test_unit_shards_streaming_memory_report_uses_one_tensor_of_anonymous_memory(tmp_path: Path) -> None:
    """Anonymous growth stays at one local tensor; resident bytes may include page cache."""

    reader, manifest, per_tensor = _streaming_fixture(tmp_path)
    local_per_tensor = per_tensor // manifest.world_size
    samples = iter(
        [(0, 0)] + [(local_per_tensor, local_per_tensor * (index + 1)) for index in range(len(manifest.tensors))]
    )
    report = streaming_memory_report(reader, manifest, rank=0, sampler=lambda: next(samples))
    assert report["tensors"] == len(manifest.tensors)
    assert report["local_bytes"] == manifest.rank_bytes(0)
    assert report["largest_tensor_bytes"] == local_per_tensor
    assert report["peak_anonymous_bytes"] == local_per_tensor
    assert report["anonymous_growth_bytes"] == local_per_tensor
    assert report["full_model_copy_bytes"] == manifest.rank_bytes(0)
    assert report["anonymous_growth_bytes"] < report["full_model_copy_bytes"]
    assert report["peak_rss_bytes"] > report["peak_anonymous_bytes"]


def test_unit_shards_streaming_report_reads_real_process_memory(tmp_path: Path) -> None:
    """The default sampler reports plausible sizes for the real process."""

    reader, manifest, _ = _streaming_fixture(tmp_path, tensors=4, rows=128)
    report = streaming_memory_report(reader, manifest, rank=0)
    assert report["tensors"] == 4
    assert report["peak_rss_bytes"] > 0
    assert report["peak_anonymous_bytes"] > 0
    assert report["anonymous_growth_bytes"] <= report["full_model_copy_bytes"]


def test_unit_fixture_gdn_snapshot_restore_replays_exactly():
    """A saved GDN state replays the remaining tokens bit-identically."""

    config = _divisible_config()
    head_k = config.ssm_state_size
    head_v = config.ssm_inner_size // config.ssm_time_step_rank
    k_heads = config.ssm_group_count
    v_heads = config.ssm_time_step_rank
    tokens = 4
    world_size = 2

    def softplus(t):
        return np.log1p(np.exp(-np.abs(t))) + np.maximum(t, 0.0)

    def l2norm(t):
        norm = np.sqrt(np.maximum(np.sum(t * t, axis=-1, keepdims=True), 0.0))
        return t / np.maximum(norm, 1e-6)

    def step(state, q_norm, k_norm, v_rows, decay, beta_t):
        """One token of the tiled recurrence for every local value head."""

        for v_head in range(state.shape[0]):
            state[v_head] *= decay[v_head]
            kv_mem = np.einsum("d,df->f", k_norm[v_head], state[v_head])
            delta = (v_rows[v_head] - kv_mem) * beta_t[v_head]
            state[v_head] += np.outer(k_norm[v_head], delta)
        return state

    def run(start_token: int, state, gates):
        """Run tokens from ``start_token`` on, returning outputs and step states."""

        local_v = state.shape[0]
        outputs = np.empty((tokens - start_token, local_v, head_v), dtype=np.float32)
        snapshots = [state.copy()]
        for index, token in enumerate(range(start_token, tokens)):
            q_norm, k_norm, v_rows, decay, beta_t = gates(token)
            step(state, q_norm, k_norm, v_rows, decay, beta_t)
            for v_head in range(local_v):
                outputs[index, v_head] = np.einsum("d,df->f", q_norm[v_head], state[v_head])
            snapshots.append(state.copy())
        return outputs, snapshots

    def make_gates(seed: int):
        rng = np.random.default_rng(seed)
        q = rng.normal(size=(tokens, v_heads, head_k)).astype(np.float32)
        k = rng.normal(size=(tokens, v_heads, head_k)).astype(np.float32)
        v = rng.normal(size=(tokens, v_heads, head_v)).astype(np.float32)
        alpha = rng.normal(size=(tokens, v_heads)).astype(np.float32)
        beta = rng.normal(size=(tokens, v_heads)).astype(np.float32)
        a_log = rng.normal(size=(v_heads,)).astype(np.float32)
        dt_bias = rng.normal(size=(v_heads,)).astype(np.float32)

        def gates(token: int):
            q_norm = l2norm(q[token]) / np.sqrt(head_k)
            k_norm = l2norm(k[token])
            decay = np.exp(-np.exp(a_log) * softplus(alpha[token] + dt_bias))
            beta_t = 1.0 / (1.0 + np.exp(-beta[token]))
            return q_norm, k_norm, v[token], decay, beta_t

        return gates

    full_gates = make_gates(51)
    full_state = np.zeros((v_heads, head_k, head_v), dtype=np.float32)
    full_outputs, full_snapshots = run(0, full_state, full_gates)

    # Restore the snapshot taken after token 1 and replay tokens 2-3.
    replayed = full_snapshots[2].copy()
    replayed_outputs, replayed_snapshots = run(2, replayed, full_gates)
    assert np.array_equal(replayed_outputs, full_outputs[2:]), "replay after restore diverged"
    assert np.array_equal(replayed_snapshots[-1], full_snapshots[-1])

    # Restoring an older snapshot and replaying from it is idempotent.
    again = full_snapshots[1].copy()
    again_outputs, _ = run(1, again, full_gates)
    assert np.array_equal(again_outputs, full_outputs[1:])
    assert np.array_equal(again, full_snapshots[-1])

    # A rank-local snapshot covers only its own value heads and their key heads.
    for rank in range(world_size):
        head_map = gdn_head_map(config, rank, world_size)
        local = np.zeros((head_map.local_v_heads, head_k, head_v), dtype=np.float32)
        assert local.nbytes == head_map.local_v_heads * head_k * head_v * 4
        assert local.nbytes * world_size == full_snapshots[-1].nbytes
        local_k = {head_map.local_k_head(v) for v in range(head_map.local_v_heads)}
        assert len(local_k) <= head_map.local_k_heads
        for v in range(head_map.local_v_heads):
            assert 0 <= head_map.local_k_head(v) < k_heads
