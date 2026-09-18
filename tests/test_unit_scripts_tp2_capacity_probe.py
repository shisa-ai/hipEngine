"""Unit tier: the TP2 capacity probe's argument contract.

The probe itself needs two devices, but its refusals must not: a ladder point
that was asked for a nonsensical envelope has to fail before any HIP contact,
and a fraction list that does not match the device count has to fail rather
than silently sizing one rank.

The probe is the Tier-1 tool for "does this context fit?" on the TP2 route, so
these tests also pin the two rules that make a ladder point meaningful: the
prompt stays below the declared context, and exactly two devices are named.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PROBE_PATH = REPO_ROOT / "scripts" / "tp2_capacity_probe.py"


def _load_probe():
    spec = importlib.util.spec_from_file_location("tp2_capacity_probe", PROBE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


probe = _load_probe()


def test_fractions_parse_two_shares() -> None:
    assert probe._parse_fractions("0.417145/0.582855") == pytest.approx(
        (0.417145, 0.582855)
    )


def test_fractions_none_is_the_even_split() -> None:
    assert probe._parse_fractions(None) is None


def test_fractions_reject_a_single_share() -> None:
    with pytest.raises(SystemExit):
        probe._parse_fractions("1.0")


def test_fractions_reject_non_numeric() -> None:
    with pytest.raises(SystemExit):
        probe._parse_fractions("half/rest")


def test_one_device_is_refused_before_any_device_work() -> None:
    with pytest.raises(SystemExit):
        probe.main(["--max-sequence-length", "8192", "--devices", "0"])


def test_three_devices_are_refused() -> None:
    with pytest.raises(SystemExit):
        probe.main(["--max-sequence-length", "8192", "--devices", "0,1,2"])


def test_a_fraction_list_must_match_the_device_count() -> None:
    with pytest.raises(SystemExit):
        probe.main(
            [
                "--max-sequence-length",
                "8192",
                "--devices",
                "0,1",
                "--fractions",
                "0.25/0.25/0.5",
            ]
        )


def test_a_prompt_at_or_above_the_declared_context_is_refused() -> None:
    """A ladder point must exercise the envelope, not redefine it."""

    with pytest.raises(SystemExit):
        probe.main(
            ["--max-sequence-length", "8192", "--devices", "0,1", "--prompt-length", "8192"]
        )
