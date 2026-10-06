from __future__ import annotations

import pytest

from scripts.qwen4exp_f16_pair_engine_ab import (
    ARM_SEQUENCE,
    CATEGORIES,
    HELDOUTS,
    _arm_order,
    _expected_pairs,
    _load_jsonl,
    _select_cases,
    summarize_engine_ab,
)


def test_dp2_arm_order_is_counterbalanced():
    assert _arm_order(0) == ARM_SEQUENCE
    assert _arm_order(1) == tuple(reversed(ARM_SEQUENCE))
    assert _arm_order(2) == ARM_SEQUENCE
    assert _arm_order(0).count("candidate") == 3
    assert _arm_order(1).count("parent") == 3


@pytest.mark.parametrize("tokens, chunk, expected", [
    (512, 1024, 0), (1024, 1024, 61), (4096, 1024, 244),
    (1025, 1024, 61), (4096, 4096, 61), (1024, 4096, 61),
    (2048, 4096, 0), (8192, 4096, 122), (4096, 512, 0),
])
def test_dp2_pair_count_follows_executed_chunks(tokens, chunk, expected):
    assert _expected_pairs(tokens, chunk) == expected


@pytest.mark.parametrize("tokens, chunk", [(0, 1024), (4096, 0)])
def test_dp2_pair_count_rejects_invalid_dimensions(tokens, chunk):
    with pytest.raises(ValueError, match="positive"):
        _expected_pairs(tokens, chunk)


def test_dp2_subset_preserves_full_suite_counterbalance():
    cases = [dict(id=f"case-{i}", prompt_tokens=512 if i < 4 else 4096)
             for i in range(7)]
    assert _select_cases(cases, [4096]) == list(enumerate(cases))[4:]
    assert _select_cases(cases, []) == list(enumerate(cases))
    with pytest.raises(ValueError, match="no cases"):
        _select_cases(cases, [1024])


def sample_rows(tokens=4096, chunk=1024):
    from scripts.qwen4exp_canonical_ar_bench import token_ids_sha256

    expected = 244 if chunk == 1024 else 61
    return [dict(case_id="case", category="code", prompt_tokens=tokens,
                 mode=mode, source="canonical", repetition=slot // 2,
                 sequence_slot=slot, case_sequence=list(ARM_SEQUENCE),
                 output_token_ids=[1, 2], output_token_ids_sha256=token_ids_sha256([1, 2]),
                 prompt_token_ids_sha256="prompt", decode_transitions=1,
                 prefill_ms=100.0 if mode == "parent" else 99.0,
                 decode_ms=10.0, prefill_tok_s=1000 * tokens / 100.0,
                 decode_tok_s=100.0, client_wall_s=0.11,
                 engagement=dict(pairs=expected if mode == "candidate" else 0,
                                 accepted=2 * expected if mode == "candidate" else 0,
                                 casts=876 - (expected if mode == "candidate" else 0)))
            for slot, mode in enumerate(ARM_SEQUENCE)]


def test_dp2_summary_validates_all_samples_and_chunk_engagement():
    rows = sample_rows()
    summary = summarize_engine_ab(rows, chunk_size=1024)
    assert summary["correctness"]["all_cross_arm_output_exact"]
    assert summary["by_case"]["case"]["candidate_engagement"]["pairs"] == 244
    assert summary["by_shape"]["4096"]["candidate_over_parent_prefill"] == pytest.approx(100 / 99)


@pytest.mark.parametrize("damage", ["missing", "ids", "hash", "casts", "pairs", "accepted", "order"])
def test_dp2_summary_refuses_incomplete_or_corrupt_evidence(damage):
    rows = sample_rows()
    if damage == "missing":
        rows.pop()
    elif damage == "ids":
        rows[0]["output_token_ids"] = [9, 9]
    elif damage == "hash":
        rows[0]["output_token_ids_sha256"] = "forged"
    elif damage == "order":
        rows[0]["sequence_slot"] = 1
    else:
        rows[0]["engagement"][damage] += 1
    with pytest.raises(ValueError):
        summarize_engine_ab(rows, chunk_size=1024)


def test_dp2_heldout_suite_is_eight_rows_and_covers_categories():
    rows = _load_jsonl(HELDOUTS)
    assert len(rows) == 8
    assert {str(row["category"]) for row in rows} == set(CATEGORIES)
    assert len({str(row["id"]) for row in rows}) == 8
    for row in rows:
        assert isinstance(row["messages"], list)
        assert row["messages"]
