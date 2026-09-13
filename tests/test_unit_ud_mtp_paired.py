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
import os
import sys
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
        "ar_decode_mode",
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
    command: dict | None = None,
) -> dict:
    resolved = _command() if command is None else command
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
        "model": {"path": model if model is not None else resolved["model"]},
        "workload": {
            "prompt_file": str(paired.REPO_ROOT / str(paired.PAIRED_PROTOCOL["prompts"])),
            "prompt_ids": list(ids),
            "prompt_count": len(ids),
            "max_new_tokens_visible": int(paired.PAIRED_PROTOCOL["max_new_tokens"]),
            "candidate_budgets": [budget],
            # Protocol identity: what the payload says it ran.  Shape checks
            # cannot see any of these.
            "target_verify_mode": paired.PAIRED_PROTOCOL["target_verify_mode"],
            "ar_decode_mode": paired.PAIRED_PROTOCOL["ar_decode_mode"],
            "draft_hidden_variant": paired.PAIRED_PROTOCOL["draft_hidden_variant"],
            "runs": runs,
            "warmup": paired.PAIRED_PROTOCOL["warmup"],
        },
        "provenance": {
            "host_name": "epyc",
            "device_name": "AMD Radeon Pro W7900",
            "resolved_backend": "hip_gfx1100",
            "target_arch": "gfx1100",
            "hipengine_commit": "0" * 40,
            "quant": resolved["quant"],
            "repetitions": runs,
            # The real payload records the absolute interpreter that ran the
            # suite; the checker must tolerate that prefix.
            "command": ["/usr/bin/python3.12", *[str(part) for part in resolved["argv"]]],
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


@pytest.mark.parametrize(
    "name",
    [
        "paired-ud-plain-mtp-c1-natural25-b3.json",
        "paired-ud-plain-mtp-c1-natural25-b3-xtx.json",
        "paired-ud-plain-mtp-c1-natural25-b3-graph-xtx.json",
    ],
)
def test_committed_artifact_is_complete_evidence(name: str):
    """Every retained paired artifact must be complete evidence for its command.

    Guards the artifact itself, not just the validator: if someone regenerates
    it from a truncated or mismatched raw payload, this fails.
    """

    artifact = paired.REPO_ROOT / "benchmarks/results" / name
    if not artifact.exists():
        pytest.skip("paired artifact not present")
    report = json.loads(artifact.read_text())
    raw_dirs = {
        Path(str(pair["raw_payload"])).parent
        for pair in report["pairs"]
    }
    assert len(raw_dirs) == 1, raw_dirs
    commands = {
        command["label"]: command
        for command in paired.resolve_commands(
            runs=int(paired.PAIRED_PROTOCOL["runs"]),
            raw_dir=next(iter(raw_dirs)),
            limit=None,
        )
    }
    prompt_ids = _prompt_ids()
    invalidated = bool((report.get("invalidation") or {}).get("invalid"))
    assert {pair["label"] for pair in report["pairs"]} == set(paired.PAIRS)
    # Every artifact states its timing verdict explicitly.
    assert report.get("timing_evidence_valid") is not invalidated
    for pair in report["pairs"]:
        command = commands[pair["label"]]
        evidence = pair["evidence"]
        assert evidence["evidence_completeness"]["complete"], (
            pair["label"],
            evidence["evidence_completeness"]["problems"],
        )
        assert evidence["binding_gates"]["evidence_complete"] is True
        assert evidence["binding_gates"]["deterministic_repeats"] is True
        # Every pair's evidence records which protocol produced it.
        assert pair["provenance"]["device_name"]
        if invalidated:
            # An invalidated artifact keeps its identity evidence (contention
            # changes timing, not token identity) but may not claim a rate:
            # every timing-derived gate must be failed by construction.
            assert evidence["timing_evidence_valid"] is False
            assert evidence["binding_gates"]["timing_evidence_valid"] is False
            assert evidence["binding_gates"]["faster_than_true_ar"] is False
            assert evidence["binding_gates"]["true_ar_denominator_present"] is False
            assert evidence["binding_passed"] is False
        else:
            assert evidence["timing_evidence_valid"] is True
            assert evidence["binding_gates"]["timing_evidence_valid"] is True
            assert evidence["binding_passed"] is True
            # A valid current artifact must record the command the protocol
            # resolves today. Invalidated attempts remain immutable historical
            # evidence and may predate a protocol correction.
            assert pair["command"] == " ".join(str(part) for part in command["argv"])
        assert pair["provenance"]["device_name"]
        assert pair["provenance"]["hipengine_commit"]
    if not invalidated and all(
        pair["evidence"]["binding_passed"] for pair in report["pairs"]
    ):
        assert report["comparisons"] == paired._paired_comparisons(report["pairs"])


# --- protocol identity ------------------------------------------------------
#
# Row coverage says how many rows arrived, not which protocol produced them.  A
# serial-verifier payload has exactly the same shape as a native-verifier one, so
# each dimension the resolved command pinned is checked against the payload's own
# declaration.  Every mutation below independently returned complete=True before
# this check existed.


def _recorded_command(payload: dict) -> list:
    return payload["provenance"]["command"]


def _retarget_recorded_flag(payload: dict, flag: str, value: str) -> dict:
    command = _recorded_command(payload)
    command[command.index(flag) + 1] = value
    return payload


@pytest.mark.parametrize(
    "mutate, needle",
    [
        pytest.param(
            lambda p: p["provenance"].__setitem__("quant", "gguf_q4_k_s"),
            "session quant identity",
            id="session-quant",
        ),
        pytest.param(
            lambda p: p["workload"].__setitem__("target_verify_mode", "serial"),
            "serial-verifier",
            id="target-verify-mode",
        ),
        pytest.param(
            lambda p: p["workload"].__setitem__("ar_decode_mode", "eager"),
            "ar_decode_mode",
            id="ar-decode-mode",
        ),
        pytest.param(
            lambda p: p["workload"].__setitem__("draft_hidden_variant", "post_output_norm"),
            "draft_hidden_variant",
            id="draft-hidden-variant",
        ),
        pytest.param(
            lambda p: p["workload"].__setitem__("runs", 3),
            "workload run count",
            id="workload-runs",
        ),
        pytest.param(
            lambda p: p["workload"].__setitem__("warmup", False),
            "warmup",
            id="warmup",
        ),
        pytest.param(
            lambda p: p["provenance"].__setitem__("repetitions", 1),
            "repetitions",
            id="provenance-repetitions",
        ),
        pytest.param(
            lambda p: _retarget_recorded_flag(p, "--target-verify-mode", "serial"),
            "provenance.command",
            id="recorded-command-flag",
        ),
        pytest.param(
            lambda p: p["provenance"].__setitem__("command", []),
            "provenance.command",
            id="recorded-command-missing",
        ),
    ],
)
def test_protocol_identity_mutations_are_rejected(mutate, needle):
    command = _command()
    payload = _payload(command=command)
    mutate(payload)
    evidence = paired._verdict(payload, 2, command=command, prompt_ids=_prompt_ids())
    assert evidence["binding_gates"]["evidence_complete"] is False
    problems = " | ".join(evidence["evidence_completeness"]["problems"])
    assert needle in problems, problems
    assert evidence["binding_passed"] is False


def test_recorded_command_tolerates_interpreter_and_output_path():
    """The two legitimate normalizations must not fail a real payload."""

    command = _command()
    payload = _payload(command=command)

    # No interpreter prefix at all.
    payload["provenance"]["command"] = [str(part) for part in command["argv"]]
    evidence = paired._verdict(payload, 2, command=command, prompt_ids=_prompt_ids())
    assert evidence["evidence_completeness"]["complete"], evidence["evidence_completeness"]

    # A relocated --output (the case --from-raw exists for).
    _retarget_recorded_flag(payload, "--output", "/somewhere/else/paired-ud-q4-k-m.json")
    evidence = paired._verdict(payload, 2, command=command, prompt_ids=_prompt_ids())
    assert evidence["evidence_completeness"]["complete"], evidence["evidence_completeness"]


# --- mechanical invalidation ------------------------------------------------


def test_invalidation_fails_every_timing_gate():
    """A descriptive note is not enough: invalidated rates must fail the gates."""

    command = _command()
    payload = _payload()

    valid = paired._verdict(payload, 2, command=command, prompt_ids=_prompt_ids())
    assert valid["timing_evidence_valid"] is True
    assert valid["binding_gates"]["faster_than_true_ar"] is True
    assert valid["binding_passed"] is True

    invalid = paired._verdict(
        payload, 2, command=command, prompt_ids=_prompt_ids(), timing_evidence_valid=False
    )
    assert invalid["timing_evidence_valid"] is False
    assert invalid["binding_gates"]["timing_evidence_valid"] is False
    assert invalid["binding_gates"]["faster_than_true_ar"] is False
    assert invalid["binding_gates"]["true_ar_denominator_present"] is False
    assert invalid["binding_passed"] is False
    # What the clock said is still recorded, just not binding.
    assert invalid["timing_as_measured"]["faster_than_true_ar"] is True
    # Identity evidence survives invalidation.
    assert invalid["evidence_completeness"]["complete"] is True
    assert invalid["determinism"]["deterministic"] is True
    assert invalid["binding_gates"]["all_gpu_accept_match_cpu"] is True


def test_invalidate_alters_the_exit_verdict(tmp_path: Path, capsys):
    """--invalidate must change the process exit code, not just the artifact."""

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    commands = paired.resolve_commands(runs=2, raw_dir=raw_dir, limit=None)
    for command in commands:
        label = str(command["label"])
        (raw_dir / f"paired-{label}.json").write_text(
            json.dumps(_payload(command=command))
        )

    def run(name: str, *extra: str) -> tuple[int, dict]:
        output = tmp_path / f"out-{name}.json"
        argv = sys.argv
        try:
            sys.argv = [
                "ud_mtp_paired.py", "--from-raw", "--runs", "2",
                "--raw-dir", str(raw_dir), "--output", str(output), *extra,
            ]
            code = paired.main()
        finally:
            sys.argv = argv
        capsys.readouterr()
        return code, json.loads(output.read_text())

    plain_code, plain = run("plain")
    invalid_code, invalid = run("invalid", "--invalidate", "device was contended")

    # Complete evidence for every pair, so the only difference is invalidation.
    assert plain_code == 0
    assert plain["timing_evidence_valid"] is True
    assert all(pair["evidence"]["binding_passed"] for pair in plain["pairs"])

    assert invalid_code == 1
    assert invalid["timing_evidence_valid"] is False
    assert invalid["invalidation"]["invalid"] is True
    assert invalid["invalidation"]["reason"] == "device was contended"
    assert "comparisons" not in invalid
    for pair in invalid["pairs"]:
        evidence = pair["evidence"]
        assert evidence["timing_evidence_valid"] is False
        assert evidence["binding_gates"]["timing_evidence_valid"] is False
        assert evidence["binding_gates"]["faster_than_true_ar"] is False
        assert evidence["binding_passed"] is False
        # Identity evidence is untouched by invalidation.
        assert evidence["evidence_completeness"]["complete"] is True
        assert evidence["determinism"]["deterministic"] is True
        assert evidence["timing_as_measured"]["faster_than_true_ar"] is True

    first = commands[0]
    failed_payload = _payload(command=first)
    failed_payload["correctness"]["all_gpu_accept_match_cpu"] = False
    (raw_dir / f"paired-{first['label']}.json").write_text(
        json.dumps(failed_payload)
    )
    failed_code, failed = run("failed")
    assert failed_code == 1
    assert "comparisons" not in failed


# --- scoped device selection ------------------------------------------------


def test_device_selected_pins_and_restores(monkeypatch):
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "7")
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "3")

    with paired.device_selected(1) as prior:
        # HIP_VISIBLE_DEVICES indexes into what ROCR_VISIBLE_DEVICES left
        # visible, so leaving a stale ROCR value would retarget the run.
        assert os.environ["HIP_VISIBLE_DEVICES"] == "1"
        assert "ROCR_VISIBLE_DEVICES" not in os.environ
        assert prior == {"HIP_VISIBLE_DEVICES": "7", "ROCR_VISIBLE_DEVICES": "3"}

    assert os.environ["HIP_VISIBLE_DEVICES"] == "7"
    assert os.environ["ROCR_VISIBLE_DEVICES"] == "3"


def test_device_selected_restores_on_failure(monkeypatch):
    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising=False)

    with pytest.raises(RuntimeError):
        with paired.device_selected(1):
            raise RuntimeError("boom")

    assert "HIP_VISIBLE_DEVICES" not in os.environ
    assert "ROCR_VISIBLE_DEVICES" not in os.environ


def test_main_scopes_and_restores_device_selection(tmp_path: Path, capsys, monkeypatch):
    """Ordered: a real run must not leak its device selection into later work."""

    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0")
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising=False)
    commands = {
        command["quant"]: command
        for command in paired.resolve_commands(runs=2, raw_dir=tmp_path, limit=None)
    }
    seen: list[str | None] = []

    def fake_run_pair(argv, output):
        seen.append(os.environ.get("HIP_VISIBLE_DEVICES"))
        quant = argv[argv.index("--quant") + 1]
        payload = _payload(command=commands[quant])
        Path(output).write_text(json.dumps(payload))
        return payload

    monkeypatch.setattr(paired, "_run_pair", fake_run_pair)
    output = tmp_path / "paired.json"
    argv = sys.argv
    try:
        sys.argv = [
            "ud_mtp_paired.py", "--runs", "2", "--raw-dir", str(tmp_path),
            "--output", str(output), "--device-index", "1",
        ]
        code = paired.main()
    finally:
        sys.argv = argv

    assert code == 0, capsys.readouterr()
    assert seen == ["1"] * len(paired.PAIRS), seen
    # Restored for whatever runs next in this process.
    assert os.environ["HIP_VISIBLE_DEVICES"] == "0"
    assert "ROCR_VISIBLE_DEVICES" not in os.environ

    report = json.loads(output.read_text())
    assert report["device"]["device_index"] == 1
    assert report["device"]["selected_environment"] == {
        "HIP_VISIBLE_DEVICES": "1",
        "ROCR_VISIBLE_DEVICES": None,
    }
    assert "prior_environment" not in report["device"]
    assert report["comparisons"] == {
        "q4_k_m": {
            "gap_to_plain_pct": {"mtp_b3": 0.0, "true_ar": 0.0},
            "plain_label": "plain-q4-k-m",
            "ud_label": "ud-q4-k-m",
            "ud_over_plain": {"mtp_b3": 1.0, "true_ar": 1.0},
        },
        "q4_k_s": {
            "gap_to_plain_pct": {"mtp_b3": 0.0, "true_ar": 0.0},
            "plain_label": "plain-q4-k-s",
            "ud_label": "ud-q4-k-s",
            "ud_over_plain": {"mtp_b3": 1.0, "true_ar": 1.0},
        },
    }


def test_real_run_requires_an_explicit_device(tmp_path: Path, capsys):
    """The device is never inferred: no --device-index means no run."""

    argv = sys.argv
    try:
        sys.argv = [
            "ud_mtp_paired.py", "--runs", "2", "--raw-dir", str(tmp_path),
            "--output", str(tmp_path / "out.json"),
        ]
        with pytest.raises(SystemExit) as excinfo:
            paired.main()
    finally:
        sys.argv = argv
    assert excinfo.value.code == 2
    assert "--device-index is required" in capsys.readouterr().err
    assert not (tmp_path / "out.json").exists()
