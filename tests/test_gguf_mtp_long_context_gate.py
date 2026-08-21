from __future__ import annotations

import pytest

from scripts.gguf_mtp_long_context_gate import (
    EagerMTPCase,
    _candidate_tokens,
    build_acceptance_cases,
    build_cycle_cases,
    gate_passed,
    parse_token_spec,
)


def test_parse_token_spec_supports_ranges_and_k_suffixes() -> None:
    assert parse_token_spec("1016-1018,2K,4k") == (1016, 1017, 1018, 2048, 4096)
    assert parse_token_spec("4,4,3") == (4, 3)

    with pytest.raises(ValueError, match="positive"):
        parse_token_spec("0")
    with pytest.raises(ValueError, match="ascending"):
        parse_token_spec("12-10")


def test_build_cycle_cases_keys_context_by_cycle_end() -> None:
    cases = build_cycle_cases(cycle_ends=(1016, 1024), candidate_budgets=(1, 2, 3))

    assert cases[0] == EagerMTPCase(cycle_end=1016, candidate_budget=1)
    assert cases[0].rows == 2
    assert cases[0].start_position == 1014
    assert cases[-1] == EagerMTPCase(cycle_end=1024, candidate_budget=3)
    assert cases[-1].rows == 4
    assert cases[-1].start_position == 1020
    assert len({case.case_id for case in cases}) == 6

    with pytest.raises(ValueError, match="cycle end"):
        build_cycle_cases(cycle_ends=(3,), candidate_budgets=(3,))


def test_build_acceptance_cases_covers_reject_every_partial_and_full() -> None:
    cases = build_acceptance_cases(cycle_ends=(1024,), candidate_budget=3)

    assert [case.expected_accepted_count for case in cases] == [0, 1, 2, 3]
    assert len({case.case_id for case in cases}) == 4
    assert all(case.start_position == 1020 for case in cases)


def test_acceptance_cases_corrupt_exactly_the_first_rejected_candidate() -> None:
    teacher = (10, 20, 30)

    assert _candidate_tokens(
        EagerMTPCase(1024, 3, expected_accepted_count=0),
        teacher_candidates=teacher,
        vocab_size=100,
    ) == (11, 20, 30)
    assert _candidate_tokens(
        EagerMTPCase(1024, 3, expected_accepted_count=2),
        teacher_candidates=teacher,
        vocab_size=100,
    ) == (10, 20, 31)
    assert _candidate_tokens(
        EagerMTPCase(1024, 3, expected_accepted_count=3),
        teacher_candidates=teacher,
        vocab_size=100,
    ) == teacher


def test_gate_passed_requires_eager_split_k_and_all_state_surfaces() -> None:
    short = {
        "passed": True,
        "cycle_end": 1023,
        "target_native_graph_submitted": False,
        "host_target_batch_materialized": True,
        "target_verify_route": "eager_native",
        "split_k_calls": 0,
        "target_logits_exact": True,
        "linear_state_exact": True,
        "kv_rows_exact": True,
        "hidden_exact": True,
        "cursor_exact": True,
        "commit_exact": True,
        "rollback_exact": True,
    }
    long = dict(short, cycle_end=1024, split_k_calls=8)
    assert gate_passed((short, long)) is True

    assert gate_passed((short, dict(long, split_k_calls=0))) is False
    assert gate_passed((short, dict(long, target_native_graph_submitted=True))) is False
    assert gate_passed((short, dict(long, kv_rows_exact=False))) is False
    assert gate_passed((short, dict(long, commit_exact=False))) is False
    assert gate_passed(
        (
            short,
            dict(
                long,
                expected_accepted_count=2,
                accepted_count=1,
            ),
        )
    ) is False
    assert gate_passed(()) is False
