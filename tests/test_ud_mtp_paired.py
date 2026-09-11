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
