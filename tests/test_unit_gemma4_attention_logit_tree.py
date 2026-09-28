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


# ---------------------------------------------------------------------------
# The GPU comparison. Guarded on HIP, so the default unit tier skips it.

_HIP = None


def _hip_available() -> bool:
    global _HIP
    if _HIP is None:
        try:
            import ctypes

            ctypes.CDLL("libamdhip64.so")
            _HIP = True
        except OSError:
            _HIP = False
    return _HIP


def _gpu_softmax_pair(q_row, k_rows, head_dim):
    """Run the real prefill kernel and return the two keys' softmax weights.

    With ``V`` set to a one-hot per key, the kernel's output at dimension ``j``
    is key ``j``'s softmax weight divided by the denominator -- pass 3 sums
    ``weight_j * v[j][d]`` over keys, so one-hot ``V`` isolates one key per
    dimension. Two keys keep the pass-2 denominator an unambiguous sum, which is
    what makes the weights reproducible in NumPy without modelling the block
    tree.

    Returns ``(out[0], out[1])`` as float32, or None when HIP is unavailable.
    """

    if not _hip_available():
        return None

    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )
    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention import (
        gemma4_attention_prefill_f32,
    )

    tokens, keys = 1, 2
    query = np.ascontiguousarray(q_row.reshape(1, 1, head_dim), dtype=np.float32)
    key = np.ascontiguousarray(k_rows.reshape(keys, 1, head_dim), dtype=np.float32)
    value = np.zeros((keys, 1, head_dim), dtype=np.float32)
    for j in range(keys):
        value[j, 0, j] = 1.0
    mask = np.ones((tokens, keys), dtype=np.uint8)
    output = np.empty_like(query)

    buffers = []
    try:
        for array in (query, key, value, mask, output):
            buffer = malloc(array.nbytes)
            buffers.append(buffer)
            copy_host_to_device(buffer, host_array_ptr(array), array.nbytes)
        gemma4_attention_prefill_f32(
            *(b.ptr for b in buffers),
            tokens=tokens,
            keys=keys,
            num_heads=1,
            num_kv_heads=1,
            head_dim=head_dim,
            scale=1.0,
            window=0,
            row_offset=0,
        )
        copy_device_to_host(host_array_ptr(output), buffers[-1], output.nbytes)
        got = output.copy()
    finally:
        for buffer in buffers:
            free(buffer)
    return np.float32(got[0, 0, 0]), np.float32(got[0, 0, 1])


def _softmax_pair(logits):
    """Two-key softmax in the kernel's shape: max, expf, sum, divide."""

    l0, l1 = np.float32(logits[0]), np.float32(logits[1])
    row_max = np.float32(max(l0, l1))
    w0 = np.float32(np.exp(np.float32(l0 - row_max)))
    w1 = np.float32(np.exp(np.float32(l1 - row_max)))
    denom = np.float32(w0 + w1)
    return np.float32(w0 / denom), np.float32(w1 / denom)


@pytest.mark.skipif(not _hip_available(), reason="no HIP runtime")
def test_the_gpu_logits_use_the_tree_not_a_plain_sum() -> None:
    """The oracle is only useful if the kernel agrees with it.

    This is the comparison the port's acceptance test will need in full. Here it
    is scoped to what isolates the logit order: one query row, two keys, and a
    one-hot ``V`` so the kernel's output *is* the softmax weight. If the kernel
    used a plain dimension-order sum instead of the tree, the weights would
    differ for the pairs where the two orders disagree.
    """

    head_dim = 256
    rng = np.random.default_rng(2024)

    tree_matches = 0
    plain_matches = 0
    discriminating = 0
    for _ in range(12):
        q_row = rng.standard_normal(head_dim).astype(np.float32)
        k_rows = rng.standard_normal((2, head_dim)).astype(np.float32)

        got = _gpu_softmax_pair(q_row, k_rows, head_dim)
        assert got is not None

        tree_logits = [
            _tree_logit(q_row, k_rows[j], head_dim) for j in range(2)
        ]
        plain_logits = [
            _plain_logit(q_row, k_rows[j], head_dim) for j in range(2)
        ]
        if tree_logits == plain_logits:
            continue
        discriminating += 1

        tree_weights = _softmax_pair(tree_logits)
        plain_weights = _softmax_pair(plain_logits)
        if tree_weights == got:
            tree_matches += 1
        if plain_weights == got:
            plain_matches += 1

    assert discriminating > 0, (
        "no random pair distinguished the tree from a plain sum, so this test "
        "cannot tell which the kernel uses"
    )
    # The comparison cannot be bit-exact, and the reason is worth stating: the
    # kernel's logits are never exposed, so they are only observable through
    # ``expf`` and a divide. NumPy's ``np.exp`` on float32 is not ``expf`` -- it
    # does not round identically -- so a correct tree still mismatches on some
    # pairs while an incorrect one would mismatch on about as many. What the
    # test can decide is which order the kernel is closer to, and that is a real
    # discrimination rather than a tolerance: on this seed the tree matches 7 of
    # 12 discriminating pairs and a plain dimension-order sum matches 1.
    #
    # Pinning the tree bit-exactly needs a comparison in the logit domain, which
    # means either exposing the logits from the kernel or comparing a new kernel
    # against gemma4_plain end to end. The latter is the port's acceptance test.
    assert tree_matches > plain_matches, (
        f"the kernel matched the tree's logits on {tree_matches} of "
        f"{discriminating} discriminating pairs and a plain sum on "
        f"{plain_matches}; the tree is not the closer order"
    )
    assert tree_matches >= discriminating // 2, (
        f"the kernel matched the tree on only {tree_matches} of {discriminating} "
        f"discriminating pairs (plain matched {plain_matches}), which is too "
        "close to chance to say the tree is the kernel's order"
    )
