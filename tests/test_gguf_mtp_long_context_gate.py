from __future__ import annotations

import json

import pytest

import scripts.gguf_mtp_long_context_gate as gate_module
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


def test_main_supports_generation_only_and_checkpoints_each_event(
    monkeypatch, tmp_path, capsys
) -> None:
    model = tmp_path / "model.gguf"
    model.write_bytes(b"fixture")
    output = tmp_path / "result.json"
    writes: list[dict[str, object]] = []
    atomic_write = gate_module._atomic_write_json

    def record_write(path, payload) -> None:
        writes.append(json.loads(json.dumps(payload)))
        atomic_write(path, payload)

    def reject_direct(*_args, **_kwargs):
        raise AssertionError("generation-only mode must not run direct cases")

    def fake_generation(
        _model,
        contexts,
        *,
        candidate_budget,
        max_new_tokens,
        max_sequence_length,
        require_cached_build,
        progress,
        on_result,
    ):
        assert contexts == (32,)
        assert candidate_budget == 3
        assert max_new_tokens == 2
        assert max_sequence_length >= 34
        assert require_cached_build is False
        result = {"context_tokens": 32, "passed": True}
        progress("generation_case_start", {"context_tokens": 32})
        on_result("generation", result)
        progress("generation_case_complete", {"context_tokens": 32, "passed": True})
        return [result]

    monkeypatch.setattr(gate_module, "_atomic_write_json", record_write)
    monkeypatch.setattr(gate_module, "_run_direct_cases", reject_direct)
    monkeypatch.setattr(gate_module, "_run_generation_contexts", fake_generation)
    monkeypatch.setattr(
        gate_module,
        "_provenance",
        lambda *_args, **_kwargs: {"host": "test"},
    )

    rc = gate_module.main(
        (
            "--model",
            str(model),
            "--cycle-ends",
            "",
            "--acceptance-cycle-ends",
            "",
            "--generation-contexts",
            "32",
            "--max-new-tokens",
            "2",
            "--out",
            str(output),
            "--fail-on-fail",
        )
    )

    assert rc == 0
    payload = json.loads(output.read_text())
    assert payload["status"] == "passed"
    assert payload["summary"]["direct_cases"] == 0
    assert payload["summary"]["generation_cases"] == 1
    assert any(row["status"] == "running" for row in writes[:-1])
    assert any(
        row.get("active_event") == "generation_case_start" for row in writes
    )
    assert writes[-1]["status"] == "passed"
    assert "generation_case_start" in capsys.readouterr().err


def test_main_rejects_an_empty_scenario_set(tmp_path) -> None:
    model = tmp_path / "model.gguf"
    model.write_bytes(b"fixture")

    with pytest.raises(SystemExit, match="at least one direct or generation"):
        gate_module.main(
            (
                "--model",
                str(model),
                "--cycle-ends",
                "",
                "--acceptance-cycle-ends",
                "",
                "--generation-contexts",
                "",
            )
        )
