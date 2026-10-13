"""CPU contracts for the row-exact speculative verify block."""
from types import SimpleNamespace

import pytest

from hipengine.generation.gemma4_gguf import Gemma4GGUFGenerator
from hipengine.runtime import gemma4 as module


@pytest.mark.parametrize(
    ("profile", "exact"),
    [("production", False), ("strict", True), ("batch_invariant", True), (None, True)],
)
def test_only_the_production_profile_verifies_on_the_wmma_dense_kernels(profile, exact):
    generator = SimpleNamespace()
    if profile is not None:
        generator.execution_profile = profile
    assert Gemma4GGUFGenerator.verify_dense_exact.fget(generator) is exact


def _runner(*, decode_variants, prefill_variants):
    runner = module.Gemma4Runner.__new__(module.Gemma4Runner)
    runner.decode_attention_variants = decode_variants
    runner.prefill_attention_variants = prefill_variants
    return runner


@pytest.mark.parametrize("offset", [0, 5, 1023, 1024, 1100, 5000])
@pytest.mark.parametrize("base", [0, 64])
def test_each_row_gets_the_range_a_one_token_step_at_its_position_gets(offset, base):
    """Row t's first key and window are the one-row block's at ``offset + t``."""

    sliding = SimpleNamespace(sliding_window=1024)
    full = SimpleNamespace(sliding_window=None)
    runner = _runner(decode_variants=None, prefill_variants=None)
    rows = 5
    if module._sliding_read_range(sliding, offset, 1) < base:
        pytest.skip("a retained band never starts above the block's first live key")
    got = runner._row_decode_attention(sliding, offset, rows, base)
    for row in range(rows):
        position = offset + row
        assert got.key_begins[row] == max(0, position - 1024 + 1) - base
        assert got.windows[row] == 1024
        # The row's own key is the last one it reads and never more than a window.
        assert 1 <= position - base + 1 - got.key_begins[row] <= 1024
    got = runner._row_decode_attention(full, offset, rows, 0)
    assert got.key_begins == (0,) * rows
    assert got.windows == tuple(offset + row + 1 for row in range(rows))


def test_rows_carry_the_decode_step_variant_requests():
    """A plan with a decode scope owns its rows; one without keeps the prefill request."""

    geometry = SimpleNamespace(sliding_window=None)
    with_scope = _runner(decode_variants=("gemma4_flash",), prefill_variants=("gemma4_staged",))
    got = with_scope._row_decode_attention(geometry, 10, 2, 0)
    assert got.prefill_attention_variants is None
    assert got.decode_attention_variants == ("gemma4_flash",)
    without = _runner(decode_variants=None, prefill_variants=("gemma4_staged",))
    got = without._row_decode_attention(geometry, 10, 2, 0)
    assert got.prefill_attention_variants == ("gemma4_staged",)
    assert got.decode_attention_variants is None
