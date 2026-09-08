"""dense_vs_dms_speed_probe must fail closed and record full trajectories.

Review feedback: the probe recorded nonfinite logits without failing,
continued after failed arms before returning success, saved only the first
four output tokens, and never stated its decode seeding. These tests pin
the fail-closed contract and the trajectory/seed record.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "campaign-artifacts" / "c1-k-sweep"))

from dense_vs_dms_speed_probe import (  # noqa: E402
    _arm_row,
    _check_finite,
    main,
)


def test_check_finite_passes_on_all_finite() -> None:
    _check_finite("dms-int8", [True] * 8)


def test_check_finite_fails_closed_with_step_indices() -> None:
    with pytest.raises(RuntimeError, match="nonfinite logits at decode steps \\[1, 3\\]"):
        _check_finite("dms-bf16", [True, False, True, False])


def test_arm_row_records_full_trajectory_and_seed() -> None:
    prompt = [11, 22, 33, 44, 55]
    tokens = [101, 102, 103, 104]
    row = _arm_row(
        "dms-int8",
        loaded_at=1.25,
        prompt=prompt,
        prefill_seconds=2.0,
        step_walls=[0.04, 0.05, 0.04, 0.05],
        tokens=tokens,
        observability=None,
    )
    assert row["output_tokens"] == tokens
    assert row["first_output_tokens"] == tokens[:4]
    expected = hashlib.sha256(
        np.asarray(tokens, dtype=np.int64).tobytes()
    ).hexdigest()
    assert row["output_tokens_sha256"] == expected
    assert row["finite_logits_all_steps"] is True
    assert row["decode_seed"]["kind"] == "last_prompt_token"
    assert row["decode_seed"]["token"] == 55
    assert "prefill prediction" in row["decode_seed"]["note"]
    assert row["decode_steps"] == len(tokens)


def _fake_completed(rc: int):
    return SimpleNamespace(returncode=rc)


def test_orchestrator_exits_nonzero_on_failed_arm(monkeypatch, tmp_path) -> None:
    model = tmp_path / "model.gguf"
    metadata = tmp_path / "dms_metadata.json"
    manifest = tmp_path / "manifest.json"
    for path in (model, metadata, manifest):
        path.write_bytes(b"x")

    def fake_stream(path):
        return list(range(64))

    monkeypatch.setattr(
        "dense_vs_dms_speed_probe._validation_stream", fake_stream
    )

    def fake_run(command, cwd=None):
        part = Path(command[command.index("--output") + 1])
        arm = command[command.index("--single-arm") + 1]
        if arm == "dms-int8":
            return _fake_completed(1)
        part.write_text(json.dumps({"arm": arm, "status": "ok"}), encoding="utf-8")
        return _fake_completed(0)

    monkeypatch.setattr("dense_vs_dms_speed_probe.subprocess.run", fake_run)
    output = tmp_path / "ab.json"
    argv = [
        "probe",
        "--model", str(model),
        "--metadata", str(metadata),
        "--data-manifest", str(manifest),
        "--prompt-tokens", "32",
        "--decode-steps", "4",
        "--arms", "dense,dms-int8",
        "--output", str(output),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    rc = main()
    assert rc == 1
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["status"] == "arm_failed"
    arms = {row["arm"]: row for row in result["arms"]}
    assert arms["dense"]["status"] == "ok"
    assert arms["dms-int8"]["status"] == "failed"
    assert arms["dms-int8"]["rc"] == 1


def test_orchestrator_returns_zero_when_all_arms_pass(monkeypatch, tmp_path) -> None:
    model = tmp_path / "model.gguf"
    metadata = tmp_path / "dms_metadata.json"
    manifest = tmp_path / "manifest.json"
    for path in (model, metadata, manifest):
        path.write_bytes(b"x")

    monkeypatch.setattr(
        "dense_vs_dms_speed_probe._validation_stream",
        lambda path: list(range(64)),
    )

    def fake_run(command, cwd=None):
        part = Path(command[command.index("--output") + 1])
        arm = command[command.index("--single-arm") + 1]
        part.write_text(json.dumps({"arm": arm, "status": "ok"}), encoding="utf-8")
        return _fake_completed(0)

    monkeypatch.setattr("dense_vs_dms_speed_probe.subprocess.run", fake_run)
    output = tmp_path / "ab.json"
    argv = [
        "probe",
        "--model", str(model),
        "--metadata", str(metadata),
        "--data-manifest", str(manifest),
        "--prompt-tokens", "32",
        "--decode-steps", "4",
        "--arms", "dense,dms-bf16",
        "--output", str(output),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    rc = main()
    assert rc == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["status"] == "diagnostic_not_a_benchmark"
