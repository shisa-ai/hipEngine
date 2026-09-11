"""Task #22: the paired UD/plain measurement is protocol-identical and gated.

``scripts/ud_mtp_paired.py`` exists so the paired comparison cannot drift: one
protocol mapping supplies every flag, so the resolved commands for the four
pairs differ only in the artifact.  It also refuses to run while the U6
certification unit is incomplete, which is the task's "only after U6
certification passes" precondition.

These tests cover the two structural guarantees.  The measurement itself is a
multi-hour GPU run and is not exercised here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import ud_mtp_paired as paired

# Arguments that are expected to differ between pairs; everything else must be
# byte-identical or the comparison is not paired.
_PER_PAIR_ARGUMENTS = {"--model", "--quant", "--output"}


def _flag_pairs(argv: list[str]) -> dict[str, str]:
    # Drop the leading program name; the rest is flag,value,flag,value,...
    # (this protocol uses no boolean flags).
    body = [part for part in argv[1:]]
    flags = [part for part in body if part.startswith("--")]
    values = [part for part in body if not part.startswith("--")]
    assert len(flags) == len(values), argv
    return dict(zip(flags, values))


def test_paired_commands_differ_only_in_the_artifact(tmp_path: Path):
    commands = paired.resolve_commands(runs=2, raw_dir=tmp_path, limit=None)
    assert {command["label"] for command in commands} == set(paired.PAIRS)
    assert len(commands) == 4

    parsed = [_flag_pairs([str(part) for part in command["argv"]]) for command in commands]
    shared_flags = set(parsed[0]) - _PER_PAIR_ARGUMENTS
    # No pair may introduce a flag the others do not carry.
    for flags in parsed[1:]:
        assert set(flags) - _PER_PAIR_ARGUMENTS == shared_flags

    for flag in shared_flags:
        values = {flags[flag] for flags in parsed}
        assert len(values) == 1, f"{flag} differs across pairs: {values}"

    # The task's named dimensions are all pinned by the protocol mapping.
    for key in (
        "prompts",
        "max_new_tokens",
        "candidate_budgets",
        "runs",
        "target_verify_mode",
        "draft_hidden_variant",
        "kv_layout",
        "ar_baseline",
        "correctness_gates",
    ):
        assert key in paired.PAIRED_PROTOCOL, key

    # Both families are present: a UD-only or plain-only run is not a pairing.
    assert {command["family"] for command in commands} == {"ud", "plain"}
    # Both quant tiers are present so the comparison is not confounded by tier.
    quants = {str(command["quant"]) for command in commands}
    assert any("q4_k_m" in quant for quant in quants)
    assert any("q4_k_s" in quant for quant in quants)


def test_paired_gate_mirrors_the_certification_record():
    from hipengine.loading.qwen35_gguf_admission import _UD_MTP_CERTIFICATIONS

    by_preset = {cert.preset_key: cert for cert in _UD_MTP_CERTIFICATIONS.values()}
    state = paired.u6_gate_state()
    assert set(state["gated"]) == set(paired.UD_GATED_LABELS)
    for label, entry in state["gated"].items():
        cert = by_preset[entry["preset_key"]]
        assert entry["measurement_ready"] is cert.measurement_ready()
        assert entry["pin_complete"] is cert.is_complete()
        assert entry["measurement_blockers"] == list(cert.measurement_blockers)
        # The run may start while the pin is still withheld: the paired_run
        # items are established by this very measurement.
        assert entry["measurement_ready"] or entry["measurement_blockers"]
    assert state["gate_passed"] == all(
        entry["measurement_ready"] for entry in state["gated"].values()
    )


def test_measurement_gate_is_weaker_than_the_pin_but_still_binding():
    """A closed pre-measurement item blocks the run; a closed paired_run item does not."""

    from hipengine.loading.qwen35_gguf_admission import (
        Qwen35GGUFUDMTPCertification,
        Qwen35GGUFUDMTPCertificationItem,
    )

    def record(*, structural_open: bool, run_open: bool) -> Qwen35GGUFUDMTPCertification:
        return Qwen35GGUFUDMTPCertification(
            fingerprint="x" * 64,
            preset_key="gguf_ud_test",
            backend="hip_gfx1100",
            execution_profile="production",
            context_max=4096,
            widths=(1,),
            items=(
                Qwen35GGUFUDMTPCertificationItem(
                    item="structural", contract="c", evidence="e",
                    qualified=not structural_open, blocker="b" if structural_open else "",
                ),
                Qwen35GGUFUDMTPCertificationItem(
                    item="run_item", contract="c", evidence="e", qualified=not run_open,
                    blocker="b" if run_open else "", phase="paired_run",
                ),
            ),
        )

    ready = record(structural_open=False, run_open=True)
    assert ready.measurement_ready() is True
    assert ready.is_complete() is False
    assert ready.measurement_blockers == ()

    blocked = record(structural_open=True, run_open=False)
    assert blocked.measurement_ready() is False
    assert blocked.is_complete() is False
    assert blocked.measurement_blockers == ("structural",)

    complete = record(structural_open=False, run_open=False)
    assert complete.measurement_ready() is True
    assert complete.is_complete() is True


def test_paired_run_refuses_when_a_pre_measurement_item_is_open(tmp_path: Path, capsys, monkeypatch):
    """The gate is not bypassable: a closed structural item stops the run."""

    monkeypatch.setattr(
        paired,
        "u6_gate_state",
        lambda: {
            "gate_passed": False,
            "pin_fingerprints": [],
            "gated": {
                "ud-q4-k-m": {
                    "preset_key": "gguf_ud_q4_k_m",
                    "measurement_ready": False,
                    "pin_complete": False,
                    "measurement_blockers": ["draft_and_verifier_state_ownership"],
                    "open_items": ["draft_and_verifier_state_ownership"],
                    "blockers": {"draft_and_verifier_state_ownership": "shared state"},
                }
            },
        },
    )
    output = tmp_path / "paired.json"
    import sys

    argv = sys.argv
    try:
        sys.argv = ["ud_mtp_paired.py", "--output", str(output), "--raw-dir", str(tmp_path)]
        code = paired.main()
    finally:
        sys.argv = argv
    assert code == 1
    assert not output.exists(), "the gated run must not write an artifact"
    captured = capsys.readouterr()
    assert "refusing to run" in captured.err
    assert "draft_and_verifier_state_ownership" in captured.err
    assert "shared state" in captured.err


def test_paired_dry_run_emits_the_protocol_without_running(tmp_path: Path, capsys):
    import sys

    argv = sys.argv
    try:
        sys.argv = ["ud_mtp_paired.py", "--dry-run", "--raw-dir", str(tmp_path)]
        code = paired.main()
    finally:
        sys.argv = argv
    payload = json.loads(capsys.readouterr().out)
    # JSON round-trips tuples to lists; compare on the normalized form.
    assert payload["protocol"] == json.loads(json.dumps(paired.PAIRED_PROTOCOL))
    assert len(payload["commands"]) == 4
    # Exit code reports the gate, and the dry run never writes a raw payload.
    assert code == (0 if payload["u6_gate"]["gate_passed"] else 1)
    assert not list(tmp_path.glob("paired-*.json"))


# --- evidence completeness -------------------------------------------------
#
# A payload that is not the complete evidence the resolved command asked for
# must fail, not pass vacuously.  The failure this guards is concrete: a
# truncated one-run payload gives each prompt a single token hash, so a
# hash-disagreement check alone finds nothing and reports deterministic
# repeats.


def _command(label: str = "ud-q4-k-m") -> dict:
    commands = paired.resolve_commands(runs=2, raw_dir=Path("/tmp"), limit=None)
    return next(c for c in commands if c["label"] == label)


def _prompt_ids() -> tuple[str, ...]:
    return paired.expected_prompt_ids(paired.REPO_ROOT / str(paired.PAIRED_PROTOCOL["prompts"]))


def _payload(
    *,
    runs: int = 2,
    prompt_ids: tuple[str, ...] | None = None,
    budget: int = 3,
    model: str | None = None,
    duplicate_first: bool = False,
    drop_last_prompt: bool = False,
    schema: int = 1,
) -> dict:
    ids = list(prompt_ids if prompt_ids is not None else _prompt_ids())
    if drop_last_prompt:
        ids = ids[:-1]
    ar_rows, mtp_rows = [], []
    for run in range(runs):
        for index, prompt in enumerate(ids):
            # Hashes are identical across runs by construction, so a complete
            # payload is deterministic and the completeness gate is what the
            # truncated-payload test exercises.
            ar_rows.append({"id": prompt, "run": run, "token_sha256_i64": f"ar{index}"})
            mtp_rows.append(
                {"id": prompt, "run": run, "token_sha256_i64": f"mtp{index}",
                 "exact_greedy_match": True}
            )
    if duplicate_first and len(ids) <= len(ar_rows):
        # Collapse the first row of run 1 onto run 0, keeping the row count at
        # the expected value so the duplicate check (not the count check) fires.
        ar_rows[len(ids)] = {**ar_rows[len(ids)], "run": 0}
    return {
        "schema": schema,
        "kind": "qwen36_dense_gguf_ar_mtp_suite",
        "status": "complete_exact",
        "correctness": {"all_exact_greedy": True, "all_gpu_accept_match_cpu": True},
        "model": {"path": model if model is not None else _command()["model"]},
        "workload": {
            "prompt_file": str(paired.REPO_ROOT / str(paired.PAIRED_PROTOCOL["prompts"])),
            "prompt_ids": list(ids),
            "prompt_count": len(ids),
            "max_new_tokens_visible": int(paired.PAIRED_PROTOCOL["max_new_tokens"]),
            "candidate_budgets": [budget],
        },
        "provenance": {
            "host_name": "epyc",
            "device_name": "AMD Radeon Pro W7900",
            "resolved_backend": "hip_gfx1100",
            "target_arch": "gfx1100",
            "hipengine_commit": "0" * 40,
        },
        "summary": {
            "true_ar": {"full": {"decode_tok_s_weighted": 10.0}},
            "mtp": {str(budget): {"full": {"decode_tok_s_weighted": 20.0, "mtp_vs_true_ar": 2.0}}},
        },
        "rows": {"true_ar": ar_rows, "mtp": {str(budget): mtp_rows}},
    }


def test_complete_payload_passes_evidence_completeness():
    command = _command()
    evidence = paired._verdict(_payload(), 2, command=command, prompt_ids=_prompt_ids())
    assert evidence["evidence_completeness"]["complete"], evidence["evidence_completeness"]
    assert evidence["binding_gates"]["evidence_complete"] is True
    assert evidence["binding_gates"]["deterministic_repeats"] is True
    assert evidence["binding_passed"] is True


def test_truncated_one_run_payload_is_rejected():
    """The exact hole: one run per prompt must not pass as deterministic."""

    command = _command()
    payload = _payload(runs=1)
    evidence = paired._verdict(payload, 2, command=command, prompt_ids=_prompt_ids())
    assert evidence["binding_gates"]["evidence_complete"] is False
    assert evidence["binding_gates"]["deterministic_repeats"] is False
    assert evidence["binding_passed"] is False
    problems = evidence["evidence_completeness"]["problems"]
    assert any("rows !=" in problem for problem in problems), problems
    # The naive hash-disagreement view is what used to pass.
    assert evidence["determinism"]["deterministic"] is True


def test_missing_prompt_is_rejected():
    command = _command()
    payload = _payload(drop_last_prompt=True)
    evidence = paired._verdict(payload, 2, command=command, prompt_ids=_prompt_ids())
    assert evidence["binding_gates"]["evidence_complete"] is False
    problems = evidence["evidence_completeness"]["problems"]
    assert any("missing prompts" in problem or "rows !=" in problem for problem in problems), problems


def test_duplicate_row_is_rejected():
    command = _command()
    payload = _payload(duplicate_first=True)
    evidence = paired._verdict(payload, 2, command=command, prompt_ids=_prompt_ids())
    assert evidence["binding_gates"]["evidence_complete"] is False
    assert any("duplicate" in p for p in evidence["evidence_completeness"]["problems"])


def test_wrong_model_or_budget_or_schema_is_rejected():
    command = _command()
    for payload in (
        _payload(model="/models/gguf/Qwen3.8-27B-Q4_K_S.gguf"),
        _payload(budget=1),
        _payload(schema=2),
    ):
        evidence = paired._verdict(payload, 2, command=command, prompt_ids=_prompt_ids())
        assert evidence["binding_gates"]["evidence_complete"] is False
        assert evidence["evidence_completeness"]["problems"]


def test_missing_provenance_is_rejected():
    command = _command()
    payload = _payload()
    payload["provenance"].pop("device_name")
    evidence = paired._verdict(payload, 2, command=command, prompt_ids=_prompt_ids())
    assert evidence["binding_gates"]["evidence_complete"] is False
    assert any("device_name" in p for p in evidence["evidence_completeness"]["problems"])


def test_from_raw_refuses_incomplete_evidence(tmp_path: Path, capsys):
    """--from-raw validates the payload against the resolved command."""

    import sys

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "paired-ud-q4-k-m.json").write_text(json.dumps(_payload(runs=1)))
    argv = sys.argv
    try:
        sys.argv = [
            "ud_mtp_paired.py", "--from-raw", "--runs", "2",
            "--raw-dir", str(raw_dir), "--output", str(tmp_path / "out.json"),
        ]
        code = paired.main()
    finally:
        sys.argv = argv
    assert code == 1
    err = capsys.readouterr().err
    assert "incomplete evidence" in err


def test_committed_artifact_is_complete_evidence():
    """The retained paired artifact must be complete evidence for its command.

    Guards the artifact itself, not just the validator: if someone regenerates
    it from a truncated or mismatched raw payload, this fails.
    """

    artifact = paired.REPO_ROOT / "benchmarks/results/paired-ud-plain-mtp-c1-natural25-b3.json"
    if not artifact.exists():
        pytest.skip("paired artifact not present")
    report = json.loads(artifact.read_text())
    commands = {
        command["label"]: command
        for command in paired.resolve_commands(
            runs=int(paired.PAIRED_PROTOCOL["runs"]), raw_dir=Path("/tmp"), limit=None
        )
    }
    prompt_ids = _prompt_ids()
    assert {pair["label"] for pair in report["pairs"]} == set(paired.PAIRS)
    for pair in report["pairs"]:
        command = commands[pair["label"]]
        evidence = pair["evidence"]
        assert evidence["evidence_completeness"]["complete"], (
            pair["label"],
            evidence["evidence_completeness"]["problems"],
        )
        assert evidence["binding_gates"]["evidence_complete"] is True
        assert evidence["binding_gates"]["deterministic_repeats"] is True
        assert evidence["binding_passed"] is True
        # The command recorded in the artifact is the command the protocol
        # resolves today, so the artifact cannot silently drift from the script.
        assert pair["command"] == " ".join(str(part) for part in command["argv"])
        assert pair["provenance"]["device_name"]
        assert pair["provenance"]["hipengine_commit"]
