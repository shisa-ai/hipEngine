"""Unit tier: the TP2 c=1 bench's rate accounting.

The bench scores a generation from ``step_traces``. A token-serial prefill
emits one step per prompt position, so counting steps counts tokens; the bulk
prefill candidate consumes the whole prompt in a single step, and scoring that
step as one token reports the prompt-length-th of the true rate. That mistake
was made once and produced a "0.38 tok/s" bulk prefill row that was really
210 tok/s, so the token count is pinned here.

No device contact: ``_rate`` is pure arithmetic and the refusals are argument
validation.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCH_PATH = REPO_ROOT / "scripts" / "tp2_c1_bench.py"


def _load_bench():
    spec = importlib.util.spec_from_file_location("tp2_c1_bench", BENCH_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bench = _load_bench()


def test_rate_counts_steps_when_no_token_count_is_given() -> None:
    """Token-serial: one step per prompt position, so steps are tokens."""

    rate = bench._rate([0.01, 0.02, 0.03])
    assert rate is not None
    assert rate["tokens"] == 3
    assert rate["tok_per_s"] == pytest.approx(3 / 0.06)


def test_rate_scores_one_bulk_step_by_the_prompt_length() -> None:
    """Bulk: a single 2.44 s step for a 512-token prompt is 210 tok/s."""

    rate = bench._rate([2.44], tokens=512)
    assert rate is not None
    assert rate["tokens"] == 512
    assert rate["tok_per_s"] == pytest.approx(512 / 2.44)
    assert rate["mean_ms_per_token"] == pytest.approx(1000 * 2.44 / 512)
    assert rate["total_s"] == pytest.approx(2.44)


def test_rate_is_none_without_steps() -> None:
    assert bench._rate([]) is None


def test_rate_does_not_divide_by_zero_on_an_empty_wall() -> None:
    rate = bench._rate([0.0], tokens=512)
    assert rate is not None
    assert rate["tok_per_s"] == 0.0


def test_fractions_parse_two_shares() -> None:
    assert bench._parse_fractions("0.44/0.56") == pytest.approx((0.44, 0.56))


def test_fractions_reject_a_non_numeric_share() -> None:
    with pytest.raises(SystemExit):
        bench._parse_fractions("0.44/sixty")
