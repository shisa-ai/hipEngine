from __future__ import annotations

from scripts.dflash_chain_e2e_bench import _build_branching_topk_tree_target_batch


def test_balanced_topk_tree_budget4_k2_shape_and_positions() -> None:
    compiled = _build_branching_topk_tree_target_batch(
        root_token=10,
        root_position=100,
        topk_tokens=((11, 12), (21, 22), (31, 32), (41, 42)),
        topk_values=((1.0, 0.9), (0.8, 0.7), (0.6, 0.5), (0.4, 0.3)),
        candidate_budget=4,
        tree_top_k=2,
        max_depth=3,
    )

    assert compiled.active_count == 4
    assert compiled.active_candidate_tokens == (11, 12, 21, 21)
    assert compiled.tree_parents == (-1, -1, 0, 1)
    assert compiled.draft_depths == (1, 1, 2, 2)
    assert compiled.child_ranks == (0, 1, 0, 0)

    batch = compiled.target_batch
    assert batch.mode == "verify_tree"
    assert batch.tokens == (10, 11, 12, 21, 21)
    assert batch.positions == (100, 101, 101, 102, 102)
    assert batch.parent_rows == (-1, 0, 0, 1, 2)
    assert batch.tree_shape == (0, 0, 1, 2)
    assert batch.active_mask == (True, True, True, True, True)


def test_topk_tree_accept_oracle_follows_non_first_sibling_path() -> None:
    compiled = _build_branching_topk_tree_target_batch(
        root_token=10,
        root_position=100,
        topk_tokens=((11, 12), (21, 22), (31, 32), (41, 42)),
        topk_values=((1.0, 0.9), (0.8, 0.7), (0.6, 0.5), (0.4, 0.3)),
        candidate_budget=4,
        tree_top_k=2,
        max_depth=3,
    )

    # Root predicts the second root child (12), then that child predicts its
    # depth-2 continuation (21).  The accepted rows are [root, row2, row4],
    # which is non-contiguous and therefore exercises the DDTree path semantics.
    result = compiled.target_batch.accept_from_top1((12, 999, 21, 888, 777))

    assert result.accepted_counts == (2,)
    assert result.accepted_tokens == ((12, 21),)
    assert result.selected_candidate_rows == (4,)
    assert result.next_tokens == (777,)


def test_topk_tree_respects_remaining_decode_depth_with_inactive_padding() -> None:
    compiled = _build_branching_topk_tree_target_batch(
        root_token=10,
        root_position=100,
        topk_tokens=((11, 12), (21, 22), (31, 32), (41, 42)),
        topk_values=((1.0, 0.9), (0.8, 0.7), (0.6, 0.5), (0.4, 0.3)),
        candidate_budget=4,
        tree_top_k=2,
        max_depth=1,
    )

    assert compiled.active_count == 2
    assert compiled.active_candidate_tokens == (11, 12)
    assert compiled.tree_parents == (-1, -1)
    batch = compiled.target_batch
    assert batch.tokens == (10, 11, 12, 0, 0)
    assert batch.parent_rows == (-1, 0, 0, 0, 0)
    assert batch.active_mask == (True, True, True, False, False)
