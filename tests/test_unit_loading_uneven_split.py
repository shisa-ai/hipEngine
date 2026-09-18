"""Unit tests for the uneven (rate-matched) TP2 shard split.

The uneven split exists because the two cards in the TP2 host are not equal:
on the measured pair the RX 7900 XTX runs every streaming kernel 15-21% faster
than the W7900, so an even split makes the slower card the pacer. Moving the
boundary is only safe for the MLP projections, and only if all three of them
move together - the rank that owns intermediate rows of ``ffn_gate``/``ffn_up``
must reduce over exactly those columns of ``ffn_down``. These tests pin the
safety properties, not just the arithmetic.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hipengine.loading.qwen35_gguf_shards import (
    COUPLED_MLP_LEAVES,
    SAFE_UNEVEN_LEAVES,
    AxisSegment,
    ShardPlanError,
    TensorShardRule,
    UnevenSplitPolicy,
    build_shard_manifest,
    build_tensor_shard_plan,
    partition_groups,
    validate_plan_coverage,
)

GGUF_PATH = Path("/models/gguf/Qwen3.8-27B-Q4_K_M.gguf")


def _model_available() -> bool:
    return GGUF_PATH.is_file()


requires_model = pytest.mark.skipif(
    not _model_available(), reason=f"model fixture {GGUF_PATH} is not present"
)

Q4_K = (12, "Q4_K", 256, 144)


def _plan(*, name, shape, rule, world_size=2, element_ranges=None):
    rows = int(shape[0])
    cols = int(shape[1]) if len(shape) > 1 else 1
    nbytes = rows * (cols // Q4_K[2]) * Q4_K[3]
    return build_tensor_shard_plan(
        name=name,
        shape=shape,
        nbytes=nbytes,
        quant_type_id=Q4_K[0],
        quant_type_name=Q4_K[1],
        rule=rule,
        world_size=world_size,
        element_ranges=element_ranges,
    )


class TestPolicyValidation:
    def test_accepts_a_two_rank_split_of_the_mlp(self) -> None:
        policy = UnevenSplitPolicy(fractions=(0.417145, 0.582855))
        assert policy.applies_to("blk.7.ffn_gate.weight")
        assert policy.applies_to("blk.63.ffn_down.weight")
        assert not policy.applies_to("blk.0.attn_q.weight")

    def test_rejects_shares_that_do_not_sum_to_one(self) -> None:
        with pytest.raises(ShardPlanError, match="sum to 1.0"):
            UnevenSplitPolicy(fractions=(0.4, 0.5))

    def test_rejects_a_zero_share(self) -> None:
        with pytest.raises(ShardPlanError, match="must be positive"):
            UnevenSplitPolicy(fractions=(0.0, 1.0))

    def test_rejects_a_single_rank(self) -> None:
        with pytest.raises(ShardPlanError, match="at least two ranks"):
            UnevenSplitPolicy(fractions=(1.0,))

    def test_rejects_a_head_structured_tensor(self) -> None:
        # The load-bearing refusal: an uneven attention split would hand a rank
        # query heads whose KV heads it does not own, which is silently wrong
        # attention rather than an error.
        with pytest.raises(ShardPlanError, match="without head semantics"):
            UnevenSplitPolicy(fractions=(0.4, 0.6), leaves=("ffn_gate", "attn_q"))
        for leaf in ("attn_k", "attn_v", "attn_output", "ssm_out", "attn_qkv"):
            with pytest.raises(ShardPlanError, match="without head semantics"):
                UnevenSplitPolicy(fractions=(0.4, 0.6), leaves=(leaf,))

    def test_rejects_a_partially_named_coupled_set(self) -> None:
        with pytest.raises(ShardPlanError, match="must be named together"):
            UnevenSplitPolicy(fractions=(0.4, 0.6), leaves=("ffn_gate", "ffn_up"))
        with pytest.raises(ShardPlanError, match="must be named together"):
            UnevenSplitPolicy(fractions=(0.4, 0.6), leaves=("ffn_down",))

    def test_default_leaves_are_exactly_the_safe_coupled_set(self) -> None:
        policy = UnevenSplitPolicy(fractions=(0.5, 0.5))
        assert set(policy.leaves) == set(COUPLED_MLP_LEAVES)
        assert set(SAFE_UNEVEN_LEAVES) == set(COUPLED_MLP_LEAVES)


class TestBoundaryPlacement:
    def test_tiles_the_axis_exactly(self) -> None:
        policy = UnevenSplitPolicy(fractions=(0.417145, 0.582855))
        ranges = policy.ranges(17408)
        assert ranges[0][0] == 0
        assert ranges[-1][1] == 17408
        for (_, stop), (start, _) in zip(ranges, ranges[1:]):
            assert stop == start
        assert [stop - start for start, stop in ranges] == [7168, 10240]

    def test_lands_on_the_alignment(self) -> None:
        policy = UnevenSplitPolicy(fractions=(0.417145, 0.582855), alignment=256)
        for start, stop in policy.ranges(17408):
            assert start % 256 == 0 and stop % 256 == 0

    def test_rounds_by_largest_remainder(self) -> None:
        # 68 units at 0.5/0.5 is even; 69 cannot be, and the extra unit goes to
        # the larger fractional remainder deterministically.
        assert UnevenSplitPolicy(fractions=(0.5, 0.5)).ranges(68 * 256) == (
            (0, 34 * 256),
            (34 * 256, 68 * 256),
        )

    def test_leaves_every_rank_at_least_one_unit(self) -> None:
        policy = UnevenSplitPolicy(fractions=(0.999, 0.001))
        ranges = policy.ranges(4 * 256)
        assert all(stop - start >= 256 for start, stop in ranges)

    def test_rejects_an_axis_that_is_not_a_whole_number_of_units(self) -> None:
        with pytest.raises(ShardPlanError, match="whole number of"):
            UnevenSplitPolicy(fractions=(0.5, 0.5)).ranges(17408 + 7)

    def test_rejects_an_axis_with_fewer_units_than_ranks(self) -> None:
        with pytest.raises(ShardPlanError, match="fewer than the 2 ranks"):
            UnevenSplitPolicy(fractions=(0.5, 0.5)).ranges(256)


class TestExplicitPartition:
    def test_rank_groups_stay_contiguous_and_cover_the_segment(self) -> None:
        segment = AxisSegment(0, 48, 1)
        ranges = partition_groups(segment, 2, rank_groups=(20, 28))
        assert ranges == [(0, 20), (20, 48)]

    def test_rank_groups_reject_a_short_vector(self) -> None:
        with pytest.raises(ShardPlanError, match="group counts for world size"):
            partition_groups(AxisSegment(0, 48, 1), 2, rank_groups=(48,))

    def test_rank_groups_reject_an_empty_rank(self) -> None:
        with pytest.raises(ShardPlanError, match="at least one whole group"):
            partition_groups(AxisSegment(0, 48, 1), 2, rank_groups=(0, 48))

    def test_rank_groups_reject_a_bad_total(self) -> None:
        with pytest.raises(ShardPlanError, match="do not sum to"):
            partition_groups(AxisSegment(0, 48, 1), 2, rank_groups=(20, 20))

    def test_even_split_is_unchanged_without_rank_groups(self) -> None:
        assert partition_groups(AxisSegment(0, 48, 1), 2) == [(0, 24), (24, 48)]


class TestExplicitTensorPlan:
    def test_column_split_places_the_boundary(self) -> None:
        plan = _plan(
            name="blk.0.ffn_gate.weight",
            shape=(17408, 5120),
            rule=TensorShardRule(kind="column", axis=0, segments=(AxisSegment(0, 17408, 1),)),
            element_ranges=((0, 7168), (7168, 17408)),
        )
        assert [slice_.local_shape for slice_ in plan.slices] == [
            (7168, 5120),
            (10240, 5120),
        ]
        assert plan.slices[0].local_nbytes + plan.slices[1].local_nbytes == plan.source_nbytes
        validate_plan_coverage(plan)

    def test_row_split_places_the_boundary(self) -> None:
        plan = _plan(
            name="blk.0.ffn_down.weight",
            shape=(5120, 17408),
            rule=TensorShardRule(kind="row", axis=1),
            element_ranges=((0, 7168), (7168, 17408)),
        )
        assert [slice_.local_shape for slice_ in plan.slices] == [
            (5120, 7168),
            (5120, 10240),
        ]
        validate_plan_coverage(plan)

    def test_rejects_ranges_that_leave_a_gap(self) -> None:
        with pytest.raises(ShardPlanError, match="must tile the axis"):
            _plan(
                name="blk.0.ffn_gate.weight",
                shape=(17408, 5120),
                rule=TensorShardRule(kind="column", axis=0, segments=(AxisSegment(0, 17408, 1),)),
                element_ranges=((0, 7000), (7168, 17408)),
            )

    def test_rejects_ranges_that_stop_short(self) -> None:
        with pytest.raises(ShardPlanError, match="short of"):
            _plan(
                name="blk.0.ffn_gate.weight",
                shape=(17408, 5120),
                rule=TensorShardRule(kind="column", axis=0, segments=(AxisSegment(0, 17408, 1),)),
                element_ranges=((0, 8704), (8704, 17000)),
            )

    def test_rejects_a_row_boundary_off_the_quant_block(self) -> None:
        # ffn_down reduces over quant blocks; a boundary inside one would read
        # a partial block.
        with pytest.raises(ShardPlanError, match="quant blocks"):
            _plan(
                name="blk.0.ffn_down.weight",
                shape=(5120, 17408),
                rule=TensorShardRule(kind="row", axis=1),
                element_ranges=((0, 7000), (7000, 17408)),
            )

    def test_rejects_an_explicit_split_of_a_multi_segment_rule(self) -> None:
        with pytest.raises(ShardPlanError, match="single-segment rules"):
            _plan(
                name="blk.0.attn_qkv.weight",
                shape=(10240, 5120),
                rule=TensorShardRule(
                    kind="group",
                    axis=0,
                    segments=(AxisSegment(0, 5120, 256), AxisSegment(5120, 10240, 256)),
                ),
                element_ranges=((0, 5000), (5000, 10240)),
            )

    def test_rejects_a_group_unaligned_boundary(self) -> None:
        with pytest.raises(ShardPlanError, match="whole number of"):
            _plan(
                name="blk.0.attn_k.weight",
                shape=(1024, 5120),
                rule=TensorShardRule(kind="group", axis=0, segments=(AxisSegment(0, 1024, 256),)),
                element_ranges=((0, 300), (300, 1024)),
            )


@requires_model
class TestManifestCoupling:
    @pytest.fixture(scope="class")
    def info(self):
        from hipengine.loading.gguf import scan_gguf

        return scan_gguf(str(GGUF_PATH))

    @pytest.fixture(scope="class")
    def policy(self):
        return UnevenSplitPolicy(fractions=(0.417145, 0.582855))

    @pytest.fixture(scope="class")
    def even(self, info):
        return build_shard_manifest(info, world_size=2)

    @pytest.fixture(scope="class")
    def uneven(self, info, policy):
        return build_shard_manifest(info, world_size=2, uneven_split=policy)

    @staticmethod
    def _ranges(manifest, name):
        return tuple(
            tuple(int(value) for value in shard.axis_ranges[0])
            for shard in manifest.plan_for(name).slices
        )

    def test_no_policy_leaves_the_even_manifest_untouched(self, even, info) -> None:
        assert even.rank_bytes(0) == build_shard_manifest(info, world_size=2).rank_bytes(0)
        assert self._ranges(even, "blk.0.ffn_gate.weight") == ((0, 8704), (8704, 17408))

    def test_coupled_roles_share_one_boundary_in_every_layer(self, uneven) -> None:
        for layer in range(64):
            ranges = {
                leaf: self._ranges(uneven, f"blk.{layer}.{leaf}.weight")
                for leaf in COUPLED_MLP_LEAVES
            }
            assert len(set(ranges.values())) == 1, (layer, ranges)
            assert ranges["ffn_gate"] == ((0, 7168), (7168, 17408))

    def test_boundary_lands_on_the_quant_block(self, uneven) -> None:
        plan = uneven.plan_for("blk.0.ffn_down.weight")
        for shard in plan.slices:
            for start, stop in shard.axis_ranges:
                assert start % 256 == 0 and stop % 256 == 0

    def test_only_the_named_tensors_move(self, even, uneven, policy) -> None:
        for plan in even.tensors:
            if policy.applies_to(plan.name):
                continue
            assert self._ranges(even, plan.name) == self._ranges(uneven, plan.name)
            assert (
                even.plan_for(plan.name).slices[0].local_nbytes
                == uneven.plan_for(plan.name).slices[0].local_nbytes
            )

    def test_bytes_move_from_the_slower_rank_to_the_faster_one(
        self, even, uneven
    ) -> None:
        assert uneven.rank_bytes(0) < even.rank_bytes(0)
        assert uneven.rank_bytes(1) > even.rank_bytes(1)
        # The pair stays balanced in resident bytes.
        assert abs(uneven.rank_bytes(0) - uneven.rank_bytes(1)) < abs(
            even.rank_bytes(0) - even.rank_bytes(1)
        )

    def test_coverage_still_tiles_every_tensor_once(self, uneven) -> None:
        for plan in uneven.tensors:
            validate_plan_coverage(plan)

    def test_manifest_records_the_split(self, uneven) -> None:
        notes = " ".join(uneven.notes)
        assert "uneven split shares" in notes
        assert "0.417145" in notes and "0.582855" in notes

    def test_refuses_shares_that_do_not_match_the_world_size(self, info, policy) -> None:
        with pytest.raises(ShardPlanError, match="for world size"):
            build_shard_manifest(info, world_size=3, uneven_split=policy)
