from __future__ import annotations

import argparse

import pytest

from scripts.gguf_mtp_prompt_priming_bench import _parse_lengths, _prompt


def test_prompt_priming_bench_parses_unique_positive_lengths() -> None:
    assert _parse_lengths("512,4096,16384") == (512, 4096, 16384)


@pytest.mark.parametrize("value", ["", "0", "512,0", "512,512", "512,nope"])
def test_prompt_priming_bench_rejects_invalid_lengths(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_lengths(value)


def test_prompt_priming_bench_repeats_the_fixed_nonbenchmark_seed() -> None:
    assert _prompt(5, seed=(7, 8)) == (7, 8, 7, 8, 7)
