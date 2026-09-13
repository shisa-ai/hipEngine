from __future__ import annotations

from dataclasses import replace

import pytest

from hipengine.speculative import (
    DraftBatch,
    TargetVerifyBatch,
    compare_target_verify_ladder,
    synthetic_chain_target_verify_ladder,
    synthetic_chain_target_verify_snapshots,
)


def _chain_batch(candidate_count: int = 4) -> TargetVerifyBatch:
    if candidate_count <= 0:
        raise ValueError("candidate_count must be positive")
    draft = DraftBatch(
        request_ids=(7,),
        candidate_tokens=tuple(100 + index for index in range(candidate_count)),
        parent_positions=tuple(9 + index for index in range(candidate_count)),
        draft_depths=tuple(index + 1 for index in range(candidate_count)),
        row_to_request=(7,) * candidate_count,
        active_mask=(True,) * candidate_count,
        mode="verify_chain",
    )
    return TargetVerifyBatch.from_draft(draft, root_tokens=(99,), root_positions=(9,))


def test_synthetic_chain_target_verify_ladder_matches_serial_prefixes() -> None:
    batch = _chain_batch(4)

    result = synthetic_chain_target_verify_ladder(batch, hidden_size=6, vocab_size=11)

    assert result.passed
    assert result.first_failure is None
    assert [comparison.stage for comparison in result.comparisons] == [
        "embedding_position",
        "input_rmsnorm",
        "linear_attn_conv_gdn",
        "full_attn_kv",
        "moe",
        "final_norm_lm_head",
    ]
    assert result.terminal_logits()[0] == result.terminal_logits(bulk=False)[0]
    assert len(result.terminal_logits()) == batch.rows
    assert len(result.terminal_logits()[0]) == 11
    assert result.selectable_state(row=4, stage="linear_attn_conv_gdn", state="linear_recurrent") == result.selectable_state(
        row=4,
        stage="linear_attn_conv_gdn",
        state="linear_recurrent",
        bulk=False,
    )
    assert result.selectable_state(row=3, stage="full_attn_kv", state="kv_key") == result.selectable_state(
        row=3,
        stage="full_attn_kv",
        state="kv_key",
        bulk=False,
    )

    for prefix in range(1, 7):
        prefix_result = synthetic_chain_target_verify_ladder(batch, hidden_size=6, vocab_size=11, layer_limit=prefix)
        assert prefix_result.passed
        assert len(prefix_result.comparisons) == prefix


def test_target_verify_ladder_reports_first_failing_stage_row_and_tensor() -> None:
    batch = _chain_batch(4)
    serial = synthetic_chain_target_verify_snapshots(batch, hidden_size=6, vocab_size=11, execution="serial")
    bulk = list(synthetic_chain_target_verify_snapshots(batch, hidden_size=6, vocab_size=11, execution="bulk"))
    stage = bulk[2]
    hidden = [list(row) for row in stage.hidden_rows]
    hidden[3][1] += 0.01
    bulk[2] = replace(stage, hidden_rows=tuple(tuple(row) for row in hidden))

    result = compare_target_verify_ladder(serial, bulk, atol=1.0e-6, rtol=0.0)

    assert not result.passed
    assert [comparison.passed for comparison in result.comparisons[:3]] == [True, True, False]
    failure = result.first_failure
    assert failure is not None
    assert failure.stage == "linear_attn_conv_gdn"
    assert failure.family == "linear_attention"
    assert failure.layer_index == 2
    assert failure.tensor == "hidden"
    assert failure.row == 3
    assert failure.column == 1
    assert failure.abs_diff > failure.tolerance
    payload = result.to_json_dict()
    assert payload["passed"] is False
    assert payload["first_failure"]["stage"] == "linear_attn_conv_gdn"  # type: ignore[index]


def test_target_verify_ladder_validates_chain_mode_and_stage_shapes() -> None:
    tree_draft = DraftBatch(
        request_ids=(7,),
        candidate_tokens=(100, 101),
        parent_positions=(9, 9),
        draft_depths=(1, 1),
        row_to_request=(7, 7),
        mode="verify_tree",
        tree_parents=(-1, -1),
    )
    tree_batch = TargetVerifyBatch.from_draft(tree_draft, root_tokens=(99,), root_positions=(9,))

    with pytest.raises(ValueError, match="mode='verify_chain'"):
        synthetic_chain_target_verify_ladder(tree_batch)

    batch = _chain_batch(2)
    serial = synthetic_chain_target_verify_snapshots(batch, execution="serial")
    bulk = list(synthetic_chain_target_verify_snapshots(batch, execution="bulk"))
    bulk[0] = replace(bulk[0], stage="wrong_stage")
    with pytest.raises(ValueError, match="matching stage"):
        compare_target_verify_ladder(serial, bulk)

    with pytest.raises(ValueError, match="layer_limit"):
        synthetic_chain_target_verify_snapshots(batch, layer_limit=0)
