"""Unit tests for the TP2 split balance planner.

The planner turns measured per-rank service rates into a block-aligned,
uneven shard split. These tests cover the three things that can silently
produce a wrong plan: tensor eligibility (a head-structured tensor must never
be selected), block arithmetic (the split must preserve the tensor's block
count), and the balance solve plus its inventory accounting (a spin kernel's
duration is waiting, not cost).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

from tp2_split_balance_plan import (  # noqa: E402
    balance_fractions,
    derive_rank_costs,
    eligible,
    split_blocks,
)


class TestEligibility:
    def test_selects_the_mlp_projections(self) -> None:
        for name in (
            "blk.0.ffn_gate.weight",
            "blk.63.ffn_up.weight",
            "blk.7.ffn_down.weight",
        ):
            assert eligible(name), name

    def test_rejects_head_structured_and_non_mlp_tensors(self) -> None:
        # These split attention or GDN heads. partition_groups refuses uneven
        # splits for them on purpose: query-head ownership and KV-head loading
        # have to correspond, and an uneven split would be silently wrong
        # attention rather than an error.
        for name in (
            "blk.0.attn_q.weight",
            "blk.0.attn_k.weight",
            "blk.0.attn_v.weight",
            "blk.0.attn_output.weight",
            "blk.0.attn_gate.weight",
            "blk.0.attn_qkv.weight",
            "blk.0.ssm_out.weight",
            "blk.0.ssm_alpha.weight",
            "blk.0.ssm_conv1d.weight",
            "output.weight",
            "token_embd.weight",
            "blk.0.attn_norm.weight",
        ):
            assert not eligible(name), name

    def test_ignores_non_weight_entries(self) -> None:
        assert not eligible("blk.0.ffn_gate")
        assert not eligible("blk.0.ffn_gate.bias")


class TestSplitBlocks:
    def test_preserves_the_block_total(self) -> None:
        for blocks in (4, 24, 40, 68, 970):
            counts = split_blocks(blocks, [0.417, 0.583], 256)
            assert sum(counts) == blocks, blocks

    def test_tracks_the_requested_ratio(self) -> None:
        counts = split_blocks(68, [0.417145, 0.582855], 256)
        assert counts == [28, 40]
        assert abs(counts[0] / 68 - 0.417145) < 1.0 / 68

    def test_breaks_ties_by_largest_remainder(self) -> None:
        # 5 blocks at 50/50 cannot be even; one rank must take the extra block
        # and the assignment must be deterministic.
        assert split_blocks(5, [0.5, 0.5], 256) == [3, 2]

    def test_gives_every_rank_at_least_one_block(self) -> None:
        counts = split_blocks(4, [0.99, 0.01], 256)
        assert min(counts) >= 1
        assert sum(counts) == 4

    def test_rejects_fewer_blocks_than_ranks(self) -> None:
        with pytest.raises(ValueError, match="cannot cover"):
            split_blocks(1, [0.5, 0.5], 256)

    def test_rejects_a_non_positive_fraction_sum(self) -> None:
        with pytest.raises(ValueError, match="fractions must be positive"):
            split_blocks(8, [0.0, 0.0], 256)


class TestBalanceSolve:
    def test_splits_evenly_when_the_ranks_are_equal(self) -> None:
        fractions = balance_fractions([2.0, 2.0], [5.0, 5.0], [20.0, 20.0])
        assert fractions == pytest.approx([0.5, 0.5])

    def test_equalizes_the_predicted_times(self) -> None:
        fixed = [2.3473, 2.0923]
        other = [6.173, 5.065]
        per = [22.3355, 18.3207]
        fractions = balance_fractions(fixed, other, per)
        times = [
            fixed[i] + other[i] + fractions[i] * per[i] for i in range(2)
        ]
        assert times[0] == pytest.approx(times[1])

    def test_gives_the_faster_rank_more_of_the_pool(self) -> None:
        # Rank 1 has the smaller per-pool time, so it must receive the larger
        # fraction.
        fractions = balance_fractions([2.0, 2.0], [5.0, 5.0], [22.0, 18.0])
        assert fractions[1] > fractions[0]

    def test_rejects_a_hopeless_imbalance(self) -> None:
        # Rank 0 is so far ahead that even an empty pool leaves it slower.
        with pytest.raises(ValueError, match="do not describe an imbalance"):
            balance_fractions([50.0, 2.0], [5.0, 5.0], [20.0, 20.0])

    def test_requires_two_ranks(self) -> None:
        with pytest.raises(ValueError, match="two ranks"):
            balance_fractions([1.0, 1.0, 1.0], [1.0], [1.0])


class TestInventoryAccounting:
    @staticmethod
    def _entry(rows: list[tuple[str, float]]) -> dict:
        return {
            "kernels": [
                {"kernel": name, "ms_per_step": ms} for name, ms in rows
            ],
            "kernel_ms_per_step": sum(ms for _, ms in rows),
        }

    def test_separates_streamed_fixed_and_spin(self) -> None:
        inventory = {
            "0": self._entry(
                [
                    ("q4_k_t16_dense_single_local32_gemv<unsigned short>", 10.0),
                    ("gguf_norm_fixed5120_wave256<true, false>", 2.0),
                    ("tp2_dev_spin_add_bf16", 0.5),
                ]
            ),
            "1": self._entry(
                [
                    ("q4_k_t16_dense_single_local32_gemv<unsigned short>", 8.0),
                    ("gguf_norm_fixed5120_wave256<true, false>", 2.0),
                    ("tp2_dev_spin_add_bf16", 4.0),
                ]
            ),
        }
        rates, fixed, spins = derive_rank_costs(inventory, [7.0, 7.0])
        assert rates[0] == pytest.approx(700.0)
        assert rates[1] == pytest.approx(875.0)
        # Fixed work excludes both the streamed kernels and the spin wait.
        assert fixed == pytest.approx([2.0, 2.0])
        assert spins == pytest.approx([0.5, 4.0])

    def test_counts_narrow_pair_routes_as_streamed(self) -> None:
        inventory = {
            "0": self._entry(
                [("q4_q6_t16_narrow_col4_planar_pair_bf16", 3.0)]
            ),
            "1": self._entry(
                [("q4_q6_t16_narrow_col4_planar_pair_bf16", 3.0)]
            ),
        }
        rates, fixed, _ = derive_rank_costs(inventory, [3.0, 3.0])
        assert rates == pytest.approx([1000.0, 1000.0])
        assert fixed == pytest.approx([0.0, 0.0])

    def test_rejects_a_rank_without_streamed_kernels(self) -> None:
        inventory = {
            "0": self._entry([("gguf_norm_fixed5120_wave256<true, false>", 2.0)]),
            "1": self._entry([("gguf_norm_fixed5120_wave256<true, false>", 2.0)]),
        }
        with pytest.raises(ValueError, match="no bandwidth-bound kernel time"):
            derive_rank_costs(inventory, [7.0, 7.0])


class TestRecordedPlan:
    """The committed plan must stay consistent with the inventory it cites."""

    PLAN = REPO / "benchmarks/results/2026-09-18-w7900-tp2-split-balance-plan.json"
    INVENTORY = (
        REPO / "benchmarks/results/2026-09-18-w7900-tp2-decode-kernel-inventory.json"
    )

    def test_plan_matches_the_inventory_it_was_derived_from(self) -> None:
        plan = json.loads(self.PLAN.read_text())
        inventory = json.loads(self.INVENTORY.read_text())
        rates, fixed, spins = derive_rank_costs(
            inventory["per_rank"], plan["rank_streamed_gib"]
        )
        assert rates == pytest.approx(plan["rank_rates_gib_s"], rel=1e-4)
        assert fixed == pytest.approx(plan["rank_fixed_ms"], rel=1e-4)
        assert spins == pytest.approx(plan["rank_spin_ms"], rel=1e-4)

    def test_plan_reproduces_the_measured_even_split_pacer(self) -> None:
        plan = json.loads(self.PLAN.read_text())
        inventory = json.loads(self.INVENTORY.read_text())
        # The pacer's measured work is its kernel time without the spin wait.
        measured = max(
            inventory["per_rank"][str(rank)]["kernel_ms_per_step"]
            - plan["rank_spin_ms"][rank]
            for rank in range(2)
        )
        assert plan["predicted_pacer_ms_even"] == pytest.approx(measured, abs=0.01)

    def test_plan_predicts_a_saving_and_moves_bytes_to_the_faster_rank(self) -> None:
        plan = json.loads(self.PLAN.read_text())
        assert plan["predicted_saving_ms_per_step"] > 0
        # Rank 1 is the faster card, so it must end up holding more bytes.
        assert plan["rank_bytes_uneven_gib"][1] > plan["rank_bytes_even_gib"][1]
        assert plan["byte_fractions"][1] > plan["byte_fractions"][0]
        assert sum(plan["byte_fractions"]) == pytest.approx(1.0)

    def test_plan_splits_every_tensor_to_its_own_block_count(self) -> None:
        plan = json.loads(self.PLAN.read_text())
        for tensor in plan["tensors"]:
            assert sum(tensor["blocks_uneven"]) == tensor["blocks"]
            assert all(count >= 1 for count in tensor["blocks_uneven"])
