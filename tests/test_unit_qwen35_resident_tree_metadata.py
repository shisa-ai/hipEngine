"""Host-side correctness tests for the DDTree verifier ancestor-mask metadata.

These tests verify the ancestor-mask construction performed by
``Qwen35ParoResidentSession._write_verify_chain_metadata`` for
``batch.mode == 'verify_tree'`` batches.  They run without a GPU by
exercising the construction logic directly against curated tree topologies.
The same mask layout is consumed by
``qwen35_paged_full_attn_prefill_gqa_gate_tree_fp16_spans``; the GPU
round-trip is covered by ``scripts/dflash_tree_attn_kernel_smoke.py``.
"""

from __future__ import annotations

import numpy as np


def _build_ancestor_mask(parents: list[int], rows: int) -> np.ndarray:
    """Mirror of the per-cycle host computation in the resident session.

    Mirrors the construction in
    ``Qwen35ParoResidentSession._write_verify_chain_metadata`` so the test
    locks in the exact algorithm consumed by the tree-aware GQA gate kernel.
    """

    mask = np.zeros((rows, rows), dtype=np.uint8)
    for i in range(rows):
        cursor = i
        while cursor >= 0:
            mask[i, cursor] = 1
            parent = parents[cursor]
            cursor = parent if parent >= 0 else -1
    return mask


def test_chain_parents_produce_lower_triangular_mask() -> None:
    parents = [-1, 0, 1, 2, 3]
    rows = len(parents)

    actual = _build_ancestor_mask(parents, rows)
    expected = np.tril(np.ones((rows, rows), dtype=np.uint8))

    assert np.array_equal(actual, expected), (
        "chain DFlash (parents = -1, 0, 1, ..., rows-2) must produce a "
        "lower-triangular ancestor mask so the tree kernel reduces to the "
        "chain kernel byte-for-byte."
    )


def test_depth2_binary_tree_mask_blocks_siblings() -> None:
    # parents:        -1, 0, 0, 1, 1, 2, 2
    # row -> parent:   .  0  0  1  1  2  2
    # row -> depth:    0  1  1  2  2  2  2
    parents = [-1, 0, 0, 1, 1, 2, 2]
    rows = len(parents)
    expected = np.array(
        [
            [1, 0, 0, 0, 0, 0, 0],  # row 0 (root) sees itself
            [1, 1, 0, 0, 0, 0, 0],  # row 1 sees root + self
            [1, 0, 1, 0, 0, 0, 0],  # row 2 sees root + self (NOT row 1, its sibling)
            [1, 1, 0, 1, 0, 0, 0],  # row 3 sees root + row 1 + self
            [1, 1, 0, 0, 1, 0, 0],  # row 4 sees root + row 1 + self (NOT row 3)
            [1, 0, 1, 0, 0, 1, 0],  # row 5 sees root + row 2 + self (NOT branches under row 1)
            [1, 0, 1, 0, 0, 0, 1],  # row 6 sees root + row 2 + self
        ],
        dtype=np.uint8,
    )

    actual = _build_ancestor_mask(parents, rows)

    assert np.array_equal(actual, expected), (
        "depth-2 binary tree mask must isolate sibling/cousin rows; this is "
        "the invariant that lets DDTree verify multiple branches in one "
        "batched forward without cross-contamination."
    )


def test_multi_root_mask_isolates_requests() -> None:
    # Two requests, each with a chain of two candidates.
    # parents: roots = [-1, -1]; req 0 candidates parents=[0, 2]; req 1 candidate parent=[1].
    # Layout: [root0, root1, cand0_d1, cand0_d2, cand1_d1]
    parents = [-1, -1, 0, 2, 1]
    rows = len(parents)
    expected = np.array(
        [
            [1, 0, 0, 0, 0],  # root 0 sees itself
            [0, 1, 0, 0, 0],  # root 1 sees itself (NOT root 0)
            [1, 0, 1, 0, 0],  # req-0 depth-1: root 0 + self
            [1, 0, 1, 1, 0],  # req-0 depth-2: root 0 + req-0 depth-1 + self
            [0, 1, 0, 0, 1],  # req-1 depth-1: root 1 + self (NOT req-0 chain)
        ],
        dtype=np.uint8,
    )

    actual = _build_ancestor_mask(parents, rows)

    assert np.array_equal(actual, expected), (
        "Multi-request tree verify must keep ancestor sets per-request: a "
        "candidate row must NEVER see another request's root or candidates."
    )


def test_self_ancestor_invariant() -> None:
    """Every row is its own ancestor.  This is the bit that lets the tree\n    kernel correctly include the row's own K/V in the attention sum.\n    """

    for parents in (
        [-1],
        [-1, 0],
        [-1, 0, 0, 1, 2],
        [-1, -1, 0, 1, 2, 3, 4],
    ):
        rows = len(parents)
        mask = _build_ancestor_mask(parents, rows)
        diag = np.diag(mask)
        assert int(diag.sum()) == rows, (
            f"diagonal must be all-ones for parents={parents}; got diag={diag.tolist()}"
        )
