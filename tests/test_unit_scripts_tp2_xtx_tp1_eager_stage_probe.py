"""CPU tests for the XTX eager stage probe's logits classification (no HIP)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(REPO_ROOT))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def probe():
    return _load("tp2_xtx_tp1_eager_stage_probe", SCRIPTS / "tp2_xtx_tp1_eager_stage_probe.py")


def test_summarize_logits_detects_nan_and_inf(probe):
    summary = probe._summarize_logits(np.array([1.0, np.nan, 3.0]), 3)
    assert summary["present"] is True and summary["finite"] is False
    summary = probe._summarize_logits(np.array([1.0, np.inf]), 2)
    assert summary["finite"] is False


def test_summarize_logits_detects_unwritten_all_zero_and_sentinel(probe):
    summary = probe._summarize_logits(np.zeros(8), 8)
    assert summary["finite"] is True and summary["all_zero"] is True
    assert summary["nonzero"] == 0
    # 0x7BFF is the bf16 bit pattern used as the probe sentinel.
    summary = probe._summarize_logits(np.full(4, 0x7BFF), 4)
    assert summary["sentinel_fraction"] == 1.0


def test_summarize_logits_reports_argmax_and_range(probe):
    summary = probe._summarize_logits(np.array([0.5, 2.0, -1.0]), 3)
    assert summary["finite"] is True
    assert summary["argmax"] == 1
    assert summary["min"] == -1.0 and summary["max"] == 2.0
    assert summary["all_zero"] is False


def test_summarize_logits_handles_missing(probe):
    assert probe._summarize_logits(None, 3) == {"present": False}
