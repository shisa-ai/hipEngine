"""The comparative WER driver must fail qualification on malformed transcripts.

Malformed generations are excluded from WER per lane independently, so two
lanes can end up scored over different clip sets. The per-lane WERs stay in the
artifact as diagnostics, but the run must not pass qualification: a lane that
emits garbage must not look better by having those clips quietly dropped.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import vibevoice_asr_wer_full as driver  # noqa: E402

GOOD = '[{"Start": 0.0, "End": 1.0, "Speaker": 0, "Content": "hello world"}]'
MALFORMED = '[{"Start": 0.0, "End": 1.0, "Speaker": 0}]'  # missing Content


def _lane_record(hypothesis: str, clips: int) -> dict:
    scored = 1 if driver._load_wer().parse_transcript(hypothesis)[1] == "ok" else 0
    return {
        "hypotheses": [hypothesis],
        "timings": [{"frontend_s": 0.1, "seconds": 1.0, "tokens": 3,
                     "finish_reason": "eos"}],
        "warmup_clips": 0,
        "malformed_transcripts": [] if scored else ["clip-0"],
        "truncated_transcripts": [],
        "clips_scored_for_wer": scored,
        "wer_excludes_malformed": not scored,
        "wer_fraction": 0.0 if scored else None,
        "wer_pct": 0.0 if scored else None,
        "mean_seconds_excl_warmup": 1.0,
        "clips_scored_for_timing": 1,
    }


# --- the verdict itself -------------------------------------------------

def test_qualification_passes_when_no_lane_is_malformed():
    systems = {"torch": _lane_record(GOOD, 1), "hipq4": _lane_record(GOOD, 1)}
    verdict = driver.qualify_lanes(systems)
    assert verdict["passed"] is True
    assert verdict["malformed_transcripts_total"] == 0
    assert verdict["same_clip_set"] is True


def test_qualification_fails_when_one_lane_is_malformed():
    systems = {"torch": _lane_record(MALFORMED, 1), "hipq4": _lane_record(GOOD, 1)}
    verdict = driver.qualify_lanes(systems)
    assert verdict["passed"] is False
    assert verdict["malformed_transcripts_total"] == 1
    # The lanes no longer cover the same clips, which is the actual hazard.
    assert verdict["same_clip_set"] is False
    assert verdict["scored_clip_counts"] == {"torch": 0, "hipq4": 1}


def test_diagnostic_scores_survive_a_failed_qualification():
    """Failing qualification must not discard the measured WERs."""
    systems = {"torch": _lane_record(MALFORMED, 1), "hipq4": _lane_record(GOOD, 1)}
    verdict = driver.qualify_lanes(systems)
    assert verdict["passed"] is False
    # hipq4 was scored and its number is still available for diagnosis.
    assert systems["hipq4"]["wer_pct"] == 0.0
    assert systems["hipq4"]["clips_scored_for_wer"] == 1
    assert systems["torch"]["wer_excludes_malformed"] is True


# --- the driver's exit code ---------------------------------------------

def _run_driver(monkeypatch, tmp_path, torch_hypothesis: str) -> tuple[int, dict]:
    """Run main() with the lane subprocesses stubbed to fixed transcripts."""
    clips = [{"clip_id": "clip-0", "text": "hello world", "seconds": 1.0,
              "wav": str(tmp_path / "x.npy")}]
    out = tmp_path / "out.json"

    wer = driver._load_wer()
    monkeypatch.setattr(wer, "_load_clips", lambda num, cache: clips)
    monkeypatch.setattr(driver, "prepare_all", lambda *a, **k: None)
    # Imported lazily inside main(), so it has to be patched at its source.
    import hipengine.loading.hf_cache as hf_cache

    monkeypatch.setattr(hf_cache, "resolve_model_path", lambda model: tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "vibevoice_asr_wer_full.py", "--num-clips", "1",
        "--lanes", "torch", "hipq4", "--out", str(out),
        "--request-cache", str(tmp_path), "--warmup", "0",
    ])

    def fake_run(cmd, **kwargs):
        lane = cmd[cmd.index("--lane-worker") + 1]
        hypothesis = torch_hypothesis if lane == "torch" else GOOD
        Path(cmd[cmd.index("--lane-out") + 1]).write_text(
            json.dumps(_lane_record(hypothesis, 1)))
        return None

    monkeypatch.setattr(driver.subprocess, "run", fake_run)
    code = driver.main()
    return code, json.loads(out.read_text())


def test_driver_exits_nonzero_when_a_lane_is_malformed(monkeypatch, tmp_path, capsys):
    code, artifact = _run_driver(monkeypatch, tmp_path, MALFORMED)
    assert code == 1, "a malformed lane must fail the run"
    assert artifact["qualification"]["passed"] is False
    assert artifact["malformed_transcripts_total"] == 1
    # The artifact is still written so the failure is diagnosable.
    assert artifact["systems"]["hipq4"]["wer_pct"] == 0.0
    assert "QUALIFICATION FAILED" in capsys.readouterr().err


def test_driver_exits_zero_when_all_lanes_are_clean(monkeypatch, tmp_path, capsys):
    code, artifact = _run_driver(monkeypatch, tmp_path, GOOD)
    assert code == 0
    assert artifact["qualification"]["passed"] is True
    assert "QUALIFICATION FAILED" not in capsys.readouterr().err
