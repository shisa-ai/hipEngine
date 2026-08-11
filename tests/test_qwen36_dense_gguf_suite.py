from __future__ import annotations

import math
from pathlib import Path

import pytest

from scripts.qwen36_dense_gguf_suite import (
    HELDOUT_PROMPT_IDS,
    _CallLedger,
    _TimedDraftProvider,
    _TimedVerifier,
    _borrowed_nextn_fallback_weights,
    aggregate_scopes,
    build_parser,
    parse_candidate_budgets,
    suite_speed_claim_eligible,
    timed_transition_count,
)


def _row(
    prompt_id: str,
    category: str,
    *,
    transitions: int,
    decode_seconds: float,
    accepted: int,
    proposed: int,
    cycles: int,
) -> dict[str, object]:
    return {
        "id": prompt_id,
        "category": category,
        "visible_outputs": transitions + 1,
        "timed_transitions": transitions,
        "decode_seconds": decode_seconds,
        "request_wall_seconds": decode_seconds * 2.0,
        "accepted_draft_tokens": accepted,
        "proposed_draft_tokens": proposed,
        "cycles": cycles,
        "target_forward_rows": cycles + proposed,
        "stage_seconds": {
            "proposal": decode_seconds * 0.25,
            "target_verify": decode_seconds * 0.50,
            "target_commit_finish": decode_seconds * 0.10,
            "scheduler_accept_replay_host_residual": decode_seconds * 0.15,
        },
    }


def test_candidate_budget_parser_accepts_unique_b1_b2_b3_subset() -> None:
    assert parse_candidate_budgets("3,1") == (3, 1)
    with pytest.raises(ValueError, match="subset of 1,2,3"):
        parse_candidate_budgets("1,4")
    with pytest.raises(ValueError, match="duplicates"):
        parse_candidate_budgets("2,2")


def test_dense_suite_borrows_effective_mapped_embedding_without_rehydration() -> None:
    placeholder_embedding = object()
    mapped_embedding = object()
    lm_head = object()
    calls: list[object] = []

    class Runner:
        weights = type(
            "Weights",
            (),
            {
                "root": staticmethod(
                    lambda slot: {
                        "token_embedding": placeholder_embedding,
                        "lm_head": lm_head,
                    }[slot]
                )
            },
        )()

        def ensure_device_token_embedding(self, *, runtime):
            calls.append(runtime)
            return mapped_embedding

    runtime = object()
    target = type("Target", (), {"runner": Runner(), "runtime": runtime})()

    borrowed = _borrowed_nextn_fallback_weights(target)

    assert borrowed == {
        "token_embedding": mapped_embedding,
        "lm_head": lm_head,
    }
    assert borrowed["token_embedding"] is not placeholder_embedding
    assert calls == [runtime]


def test_dense_suite_defaults_to_native_target_verify_with_serial_rollback() -> None:
    parser = build_parser()
    assert parser.parse_args(["--output", "/tmp/out.json"]).target_verify_mode == "native"
    assert (
        parser.parse_args(
            ["--target-verify-mode", "serial-exact", "--output", "/tmp/out.json"]
        ).target_verify_mode
        == "serial-exact"
    )


def test_speed_claim_eligibility_requires_the_exact_committed_full_suite(tmp_path) -> None:
    prompt_ids = [
        "code_merge_intervals",
        "code_topological_sort",
        "code_lru_cache",
        "code_markdown_table",
        "general_en_plan",
        "general_en_explain",
        "general_ja_plan",
        "general_ja_explain",
        "mixed_ja_en_translate",
        "mixed_ja_en_review",
    ]
    canonical = Path(__file__).resolve().parents[1] / "benchmarks/prompts/mtpbench-code-general-ja.jsonl"

    assert suite_speed_claim_eligible(
        prompts_path=canonical,
        prompt_ids=prompt_ids,
        max_new_tokens=25,
        all_exact=True,
    )
    assert not suite_speed_claim_eligible(
        prompts_path=tmp_path / "copied.jsonl",
        prompt_ids=prompt_ids,
        max_new_tokens=25,
        all_exact=True,
    )
    assert not suite_speed_claim_eligible(
        prompts_path=canonical,
        prompt_ids=prompt_ids[:-1],
        max_new_tokens=25,
        all_exact=True,
    )
    assert not suite_speed_claim_eligible(
        prompts_path=canonical,
        prompt_ids=prompt_ids,
        max_new_tokens=24,
        all_exact=True,
    )


def test_transition_count_excludes_prefill_sample() -> None:
    assert timed_transition_count(25) == 24
    assert timed_transition_count(1) == 0
    with pytest.raises(ValueError, match="positive"):
        timed_transition_count(0)


def test_timed_wrappers_publish_profile_markers_for_each_mtp_phase() -> None:
    marker_calls: list[str] = []

    class Markers:
        def push(self, name: str) -> None:
            marker_calls.append(f"push:{name}")

        def pop(self) -> None:
            marker_calls.append("pop")

    class Provider:
        def propose(self, value: int) -> int:
            return value + 1

        def advance_full_accept_tail(self, value: int) -> int:
            return value + 2

        def launch_device_proposal(self, value: int) -> int:
            return value + 6

    class Verifier:
        def prepare(self, value: int) -> int:
            return value + 3

        def commit(self, value: int) -> int:
            return value + 4

        def finish(self, value: int) -> int:
            return value + 5

    ledger = _CallLedger(Markers())  # type: ignore[arg-type]
    provider = _TimedDraftProvider(Provider(), ledger)
    verifier = _TimedVerifier(Verifier(), ledger)

    assert provider.propose(1) == 2
    assert provider.advance_full_accept_tail(1) == 3
    assert provider.launch_device_proposal(1) == 7
    assert verifier.prepare(1) == 4
    assert verifier.commit(1) == 5
    assert verifier.finish(1) == 6

    snapshot = ledger.snapshot()
    assert list(snapshot["totals_seconds"]) == [
        "proposal",
        "proposal_update",
        "target_commit",
        "target_finish",
        "target_verify",
    ]
    assert all(value >= 0.0 for value in snapshot["totals_seconds"].values())
    assert len(snapshot["samples_seconds"]["proposal"]) == 2
    assert len([call for call in marker_calls if call.startswith("push:")]) == 6
    assert marker_calls.count("pop") == 6


def test_scope_aggregation_uses_transition_normalized_wall_and_fixed_heldouts() -> None:
    rows = [
        _row(
            "code_merge_intervals",
            "code",
            transitions=24,
            decode_seconds=2.0,
            accepted=6,
            proposed=12,
            cycles=18,
        ),
        _row(
            "code_markdown_table",
            "code",
            transitions=24,
            decode_seconds=4.0,
            accepted=4,
            proposed=16,
            cycles=20,
        ),
        _row(
            "general_en_explain",
            "general_en",
            transitions=24,
            decode_seconds=6.0,
            accepted=0,
            proposed=12,
            cycles=24,
        ),
    ]

    scopes = aggregate_scopes(rows)

    assert HELDOUT_PROMPT_IDS >= {"code_markdown_table", "general_en_explain"}
    assert scopes["full"]["prompt_ids"] == [
        "code_merge_intervals",
        "code_markdown_table",
        "general_en_explain",
    ]
    assert scopes["full"]["timed_transitions"] == 72
    assert scopes["full"]["decode_seconds"] == 12.0
    assert scopes["full"]["decode_tok_s_weighted"] == 6.0
    assert scopes["full"]["client_transition_tok_s"] == 3.0
    assert scopes["full"]["draft_acceptance"] == 10 / 40
    assert scopes["full"]["accepted_per_output"] == 10 / 75
    assert scopes["full"]["accepted_per_transition"] == 10 / 72
    assert scopes["full"]["target_passes"] == 62
    assert scopes["train"]["prompt_ids"] == ["code_merge_intervals"]
    assert scopes["heldout"]["prompt_ids"] == [
        "code_markdown_table",
        "general_en_explain",
    ]
    assert scopes["categories"]["code"]["decode_tok_s_weighted"] == 8.0
    assert math.isclose(
        sum(scopes["full"]["stage_seconds"].values()),
        scopes["full"]["decode_seconds"],
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    assert scopes["full"]["stage_reconciled"] is True
