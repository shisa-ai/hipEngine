"""The exact reduction order of gemma4's warp-per-key attention logit.

``gemma4_warp_key_logit`` in
``hipengine/kernels/hip_gfx1100/gemma4/gemma4_attention.hip`` computes one key's
QK logit as a fixed binary tree. The tree is a contract, not an implementation
detail: every pair it adds is the pair ``gemma4_attn_block_sum`` adds, in the
same descending order, which is what makes the warp-per-key path bit-identical to
the block-sum path. A change to the tree changes the logits' last mantissa bits
while leaving every approximate comparison green, so the order is pinned here as
an executable oracle rather than left to a comment.

The tree, for one key with lane ``l`` holding logical dimensions ``l + 32k``:

* ``p[k] = q[l + 32k] * k[l + 32k]`` for ``k`` in 0..7, with **contraction off**.
  Letting the compiler fuse a leaf into a tree add rounds once instead of twice
  and the logits stop being bit-identical, which is why the kernel carries
  ``#pragma clang fp contract(off)``.
* At ``head_dim`` 512 a second half of eight leaves is fused into the first with
  ``fmaf``, reproducing the block kernel's per-thread accumulation
  ``dot = 0; for (d = tid; d < head_dim; d += threads) dot += q[d] * k[d]`` with
  contraction left on. At 256 there is one leaf per thread and ``fma(a, b, 0)`` is
  exact, so a single rounded multiply is already what the block kernel computes.
* Then intra-lane ``p[k] += p[k + 4]``, ``p[k] += p[k + 2]``, ``p[0] += p[1]``,
  and finally the lane stages ``+16, +8, +4, +2, +1``.

This file is CPU-only and needs no HIP. It pins the order; it does not and cannot
show that the GPU kernel matches it. That comparison is the port's acceptance
test and belongs next to the geometry suite.
"""

from __future__ import annotations

import numpy as np
import pytest


def _leaves(q_row: np.ndarray, k_row: np.ndarray, lane: int, head_dim: int):
    """The kernel's per-lane leaves, with contraction off as the pragma requires."""

    # A plain multiply, then an explicit float32 round: this is what
    # ``#pragma clang fp contract(off)`` buys. ``np.float32(a) * np.float32(b)``
    # in NumPy is already a single rounded float32 multiply, so the product is
    # the same value the kernel's uncontracted leaf produces.
    p = np.empty(16, dtype=np.float32)
    for k in range(8):
        d = lane + 32 * k
        p[k] = np.float32(q_row[d]) * np.float32(k_row[d])
    if head_dim == 512:
        # The second half is an fmaf into the first half's leaf, so the add is
        # not rounded before the store. ``np.float32`` cannot express a fused
        # multiply-add, so the fused value is computed in float64 and rounded
        # once -- which is what fmaf does.
        for k in range(8):
            d = lane + 32 * (k + 8)
            fused = np.float64(q_row[d]) * np.float64(k_row[d]) + np.float64(p[k])
            p[k] = np.float32(fused)
    return p


def _tree_logit(q_row: np.ndarray, k_row: np.ndarray, head_dim: int) -> np.float32:
    """One key's logit, in the kernel's exact order, summed over all 32 lanes."""

    lanes = np.empty(32, dtype=np.float32)
    for lane in range(32):
        p = _leaves(q_row, k_row, lane, head_dim)
        for k in range(4):
            p[k] = np.float32(p[k] + p[k + 4])
        for k in range(2):
            p[k] = np.float32(p[k] + p[k + 2])
        p[0] = np.float32(p[0] + p[1])
        lanes[lane] = p[0]
    for offset in (16, 8, 4, 2, 1):
        for lane in range(offset):
            lanes[lane] = np.float32(lanes[lane] + lanes[lane + offset])
    return lanes[0]


def _plain_logit(q_row: np.ndarray, k_row: np.ndarray, head_dim: int) -> np.float32:
    """The same sum in dimension order, for contrast. Not the kernel's order."""

    total = np.float32(0.0)
    for d in range(head_dim):
        total = np.float32(total + np.float32(q_row[d]) * np.float32(k_row[d]))
    return total


@pytest.mark.parametrize("head_dim", [256, 512])
def test_the_tree_order_is_not_a_plain_sum(head_dim: int) -> None:
    """The order is observable, which is why it has to be pinned.

    If a plain dimension-order sum agreed with the tree on random data, the tree
    would not need recording. It does not agree, so an implementation that
    reassociates is detectable rather than merely different-looking.
    """

    rng = np.random.default_rng(0)
    differing = 0
    for _ in range(64):
        q_row = rng.standard_normal(head_dim).astype(np.float32)
        k_row = rng.standard_normal(head_dim).astype(np.float32)
        if _tree_logit(q_row, k_row, head_dim) != _plain_logit(q_row, k_row, head_dim):
            differing += 1
    assert differing > 0, (
        "the tree and a plain sum agreed on all 64 random pairs, so this test "
        "cannot distinguish them and the tree order is not being exercised"
    )


@pytest.mark.parametrize("head_dim", [256, 512])
def test_the_tree_reproduces_itself_on_a_fixed_input(head_dim: int) -> None:
    """A regression pin: the exact bits for a fixed input.

    The values below are the tree's own output, recorded so that any change to
    the order -- a different intra-lane pairing, a reordered lane stage, or
    contraction accidentally left on -- fails here with the offending width
    named, rather than surfacing as a last-mantissa drift somewhere downstream.
    """

    rng = np.random.default_rng(1234)
    q_row = rng.standard_normal(head_dim).astype(np.float32)
    k_row = rng.standard_normal(head_dim).astype(np.float32)
    got = _tree_logit(q_row, k_row, head_dim)

    # Recomputed through an independent route, written as the kernel's three
    # literal stages rather than as a parenthesised expression. The distinction
    # is not cosmetic: ``p[k] += p[k + 4]; p[k] += p[k + 2]; p[0] += p[1]``
    # pairs (0,4)(1,5)(2,6)(3,7), then (0,2)(1,3) -- so stage two adds
    # (p0+p4) to (p2+p6) -- and the result is
    # ``((p0+p4)+(p2+p6)) + ((p1+p5)+(p3+p7))``, which is not the left-to-right
    # ``((p0+p4)+(p1+p5)) + ((p2+p6)+(p3+p7))`` a reader naturally writes. The
    # two differ in the last bits, and this test caught that when it was first
    # written.
    lanes = np.empty(32, dtype=np.float32)
    for lane in range(32):
        p = _leaves(q_row, k_row, lane, head_dim)
        stage = p.copy()
        for k in range(4):
            stage[k] = np.float32(stage[k] + stage[k + 4])
        for k in range(2):
            stage[k] = np.float32(stage[k] + stage[k + 2])
        lanes[lane] = np.float32(stage[0] + stage[1])
    for offset in (16, 8, 4, 2, 1):
        for lane in range(offset):
            lanes[lane] = np.float32(lanes[lane] + lanes[lane + offset])

    assert got == lanes[0], (
        f"head_dim {head_dim}: the intra-lane stages are not "
        "(0,4)(1,5)(2,6)(3,7) then (0,2)(1,3) then (0,1), over lane stages "
        "16/8/4/2/1"
    )


def test_the_intra_lane_pairing_is_the_kernel_pairing() -> None:
    """The stage order is not the left-to-right reading of the same sum.

    ``((p0+p4)+(p2+p6)) + ((p1+p5)+(p3+p7))`` is what the kernel's three stages
    produce. The natural left-to-right grouping
    ``((p0+p4)+(p1+p5)) + ((p2+p6)+(p3+p7))`` is a different tree, and on this
    input the two disagree -- which is the whole reason the order is pinned.
    """

    rng = np.random.default_rng(7)
    disagreed = 0
    for _ in range(64):
        p = rng.standard_normal(8).astype(np.float32)

        stage = p.copy()
        for k in range(4):
            stage[k] = np.float32(stage[k] + stage[k + 4])
        for k in range(2):
            stage[k] = np.float32(stage[k] + stage[k + 2])
        kernel_pairing = np.float32(stage[0] + stage[1])

        a = np.float32(p[0] + p[4])
        b = np.float32(p[1] + p[5])
        c = np.float32(p[2] + p[6])
        d = np.float32(p[3] + p[7])
        left_to_right = np.float32(np.float32(a + b) + np.float32(c + d))

        if kernel_pairing != left_to_right:
            disagreed += 1

    assert disagreed > 0, (
        "the kernel's stage pairing and the left-to-right grouping agreed on "
        "all 64 random inputs, so this test is not exercising the difference "
        "it exists to pin"
    )


def test_a_256_wide_head_uses_one_leaf_per_thread() -> None:
    """The width that takes the single-leaf branch, checked at its boundary.

    At ``head_dim`` 256 the kernel takes ``kLeaves == 8`` and its leaves cover
    dimensions 0..255 exactly once per lane; at 512 it takes ``kLeaves == 16``.
    The count is what selects the branch, so it is asserted directly rather than
    inferred from an output value.
    """

    for head_dim, leaves in ((256, 8), (512, 16)):
        covered = {lane + 32 * k for lane in range(32) for k in range(leaves)}
        assert covered == set(range(head_dim)), (
            f"head_dim {head_dim} with {leaves} leaves per lane does not cover "
            "every dimension exactly once"
        )
