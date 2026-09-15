"""The quality suite's codec-join check.

The model contract requires "sample-count continuity plus a click/duplication check at
every chunk boundary, because correct chunks can still click at joins". These tests
drive that check with synthetic waveforms, so they need no weights and no GPU.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def _load_suite():
    path = ROOT / "scripts" / "vibevoice_tts_quality_suite.py"
    spec = importlib.util.spec_from_file_location("vibevoice_tts_quality_suite_join", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _smooth(n: int, seed: int = 0) -> np.ndarray:
    """A band-limited signal, so its interior steps have a real distribution."""

    rng = np.random.default_rng(seed)
    raw = rng.standard_normal(n).astype(np.float32)
    # One-pole low pass: neighbouring samples become correlated, which is what makes a
    # join discontinuity detectable against the interior.
    out = np.empty_like(raw)
    acc = np.float32(0.0)
    for i, value in enumerate(raw):
        acc = np.float32(0.98) * acc + np.float32(0.02) * value
        out[i] = acc
    return out * np.float32(0.5)


def test_a_smooth_signal_with_continuous_chunks_passes():
    suite = _load_suite()
    chunk = suite.FRAME_SAMPLES
    audio = _smooth(chunk * 4)
    checks = suite._join_checks(audio, [chunk] * 4)

    assert checks["sample_count_continuous"] is True
    assert checks["whole_frames"] is True
    assert checks["chunk_count"] == 4
    assert checks["no_join_click"] is True


def test_a_sample_count_that_does_not_add_up_is_reported():
    suite = _load_suite()
    audio = _smooth(suite.FRAME_SAMPLES * 3)
    checks = suite._join_checks(audio, [suite.FRAME_SAMPLES] * 2)

    assert checks["sample_count_continuous"] is False


def test_a_partial_frame_is_not_a_whole_frame():
    suite = _load_suite()
    audio = _smooth(suite.FRAME_SAMPLES + 10)
    checks = suite._join_checks(audio, [suite.FRAME_SAMPLES, 10])

    assert checks["whole_frames"] is False


def test_a_click_at_a_boundary_fails_the_check():
    """The case the contract names: correct chunks that still click at the join."""

    suite = _load_suite()
    chunk = suite.FRAME_SAMPLES
    audio = _smooth(chunk * 4)
    audio[2 * chunk] = np.float32(0.9)  # a step the interior never produces
    checks = suite._join_checks(audio, [chunk] * 4)

    assert checks["no_join_click"] is False
    assert checks["boundary_max_step_ratio"] > suite.JOIN_CLICK_STEP_RATIO


def test_a_click_inside_a_chunk_is_not_a_join_artifact():
    """The check must be about boundaries, not about loud audio anywhere."""

    suite = _load_suite()
    chunk = suite.FRAME_SAMPLES
    audio = _smooth(chunk * 4)
    audio[2 * chunk + 7] = np.float32(0.9)
    checks = suite._join_checks(audio, [chunk] * 4)

    assert checks["no_join_click"] is True


def test_a_duplicated_sample_is_reported_as_a_diagnostic():
    """A repeated sample is the opposite signature: a boundary step below the signal's
    own typical step. It is reported, not gated, because a genuine silence at a speech
    boundary looks the same."""

    suite = _load_suite()
    chunk = suite.FRAME_SAMPLES
    audio = _smooth(chunk * 4)
    audio[2 * chunk] = audio[2 * chunk - 1]  # duplicate the previous sample
    checks = suite._join_checks(audio, [chunk] * 4)

    assert checks["boundary_min_step_ratio"] < 1.0
    assert "no_join_click" in checks


def test_fewer_than_two_chunks_has_no_boundary_to_check():
    suite = _load_suite()
    audio = _smooth(suite.FRAME_SAMPLES)
    checks = suite._join_checks(audio, [suite.FRAME_SAMPLES])

    assert checks["chunk_count"] == 1
    assert "no_join_click" not in checks
    assert checks["sample_count_continuous"] is True
