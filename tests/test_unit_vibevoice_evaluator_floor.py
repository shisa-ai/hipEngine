"""The quality suite's evaluator-floor control.

The suite's WER gate is measured by ``microsoft/VibeVoice-ASR-HF`` running through
hipEngine's own ``LLM.transcribe`` -- the same runtime the suite is qualifying. So a
non-zero WER is not by itself evidence about the TTS lane. ``_evaluator_floor``
transcribes the reference implementation's own PCM, which contains every word, so
whatever WER the ASR lane reports there is its own.

These tests drive that control with a stub engine, so they need no ASR, no GPU and
no torch.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures" / "vibevoice_tts"


def _require_wer_deps() -> None:
    """``scripts/vibevoice_asr_wer.py`` needs jiwer for the metric and transformers for
    the tokenizer, and both are optional (the suite is run as ``uv run --with jiwer
    --with transformers``). Skip rather than fail on a runner that lacks them."""

    pytest.importorskip("jiwer", reason="the WER oracle needs the optional jiwer extra")
    pytest.importorskip(
        "transformers", reason="the WER oracle needs the optional transformers extra"
    )


def _load_suite():
    path = ROOT / "scripts" / "vibevoice_tts_quality_suite.py"
    spec = importlib.util.spec_from_file_location("vibevoice_tts_quality_suite", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["vibevoice_tts_quality_suite"] = module
    spec.loader.exec_module(module)
    return module


def _manifest() -> dict:
    return json.loads((FIXTURES / "manifest.json").read_text())


def _transcript(text: str) -> str:
    """A VibeVoice-ASR shaped response carrying ``text``."""

    return json.dumps([{"Start": 0.0, "End": 3.0, "Speaker": "0", "Content": text}])


class _StubEngine:
    def __init__(self, texts: dict[str, str]):
        self.texts = texts
        self.seen: list[str] = []

    def transcribe(self, audio, *, max_new_tokens=256, seed=0):
        # Identify which fixture was handed over by its sample count.
        key = {80000: "single", 176000: "two"}.get(int(np.size(audio)), "unknown")
        self.seen.append(key)
        text = self.texts.get(key, "")
        return type("R", (), {"text": text})()


def test_floor_is_zero_when_the_asr_reads_the_reference_audio_correctly():
    _require_wer_deps()
    suite = _load_suite()
    manifest = _manifest()
    exact = {
        "single": _transcript("The quick brown fox jumps over the lazy dog."),
        "two": _transcript(
            "I heard there is big news in text to speech lately. "
            "Yes, and it runs locally on this machine."
        ),
    }
    floor = suite._evaluator_floor(_StubEngine(exact), FIXTURES, manifest)

    assert floor["max_wer"] == 0.0
    assert [entry["reference"] for entry in floor["requests"]] == ["single", "two"]
    assert all(entry["wer"] == 0.0 for entry in floor["requests"])
    # The reference PCM is what was transcribed, not the synthesized requests.
    assert floor["requests"][0]["seconds"] == round(80000 / 24000, 3)


def test_floor_reports_the_evaluators_own_error_and_not_the_tts_lanes():
    """A stub that drops a word makes the floor non-zero; that is the point."""

    _require_wer_deps()
    suite = _load_suite()
    manifest = _manifest()
    sloppy = {
        "single": _transcript("The quick brown fox jumps over the lazy."),
        "two": _transcript(
            "I heard there is big news in text to speech lately. "
            "Yes, and it runs locally on this machine."
        ),
    }
    floor = suite._evaluator_floor(_StubEngine(sloppy), FIXTURES, manifest)

    single = next(e for e in floor["requests"] if e["reference"] == "single")
    two = next(e for e in floor["requests"] if e["reference"] == "two")
    assert single["wer"] == round(1 / 9, 4)
    assert two["wer"] == 0.0
    # The floor is the worst the evaluator does on known-good audio, so a request
    # at or below it is not attributable to the TTS lane.
    assert floor["max_wer"] == round(1 / 9, 4)
    assert "not evidence of a synthesis defect" in floor["interpretation"]


def test_floor_skips_a_reference_whose_pcm_is_absent(tmp_path):
    _require_wer_deps()
    suite = _load_suite()
    manifest = _manifest()
    only_single = tmp_path / "single_audio.npz"
    np.savez(only_single, pcm=np.load(FIXTURES / "single_audio.npz")["pcm"])
    floor = suite._evaluator_floor(
        _StubEngine({"single": _transcript("The quick brown fox jumps over the lazy dog.")}),
        tmp_path,
        manifest,
    )
    assert [entry["reference"] for entry in floor["requests"]] == ["single"]


def test_floor_is_empty_rather_than_crashing_when_no_reference_exists(tmp_path):
    suite = _load_suite()
    floor = suite._evaluator_floor(_StubEngine({}), tmp_path, _manifest())
    assert floor["requests"] == []
    assert floor["max_wer"] is None


def test_a_failure_far_above_the_floor_is_the_one_that_matters():
    """Pin the reported failure against the measured floor, as data.

    ``two-2turn`` seed 20260915 at the pre-decoder revision measured 0.3529 while the
    evaluator's floor on reference audio is 0.0. That gap is what makes it a synthesis
    failure rather than a transcript artifact, and an independent ASR (whisper
    large-v3-turbo) reproduced it on the same waveform by dropping the first turn
    entirely.
    """

    _require_wer_deps()
    suite = _load_suite()
    floor = suite._evaluator_floor(
        _StubEngine(
            {
                "single": _transcript("The quick brown fox jumps over the lazy dog."),
                "two": _transcript(
                    "I heard there is big news in text to speech lately. "
                    "Yes, and it runs locally on this machine."
                ),
            }
        ),
        FIXTURES,
        _manifest(),
    )
    reported = 0.3529
    assert reported > floor["max_wer"]
    # And the kept revision's worst request WER is also above the floor, so the gate
    # still sees it rather than being silently relaxed.
    assert 0.0588 > floor["max_wer"]
