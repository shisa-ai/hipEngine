from __future__ import annotations

import pytest

from scripts.gguf_packed_ar_d2_ceiling_sweep import (
    _aggregate_goodput_tokens_per_s,
)


def test_d2_sweep_aggregate_goodput_has_tokens_per_second_units() -> None:
    assert _aggregate_goodput_tokens_per_s(1, 40.0) == 25.0
    assert _aggregate_goodput_tokens_per_s(10, 100.0) == 100.0
    assert _aggregate_goodput_tokens_per_s(32, 250.0) == 128.0


@pytest.mark.parametrize(("logical_c", "wall_ms"), ((0, 10.0), (1, 0.0)))
def test_d2_sweep_goodput_rejects_non_positive_inputs(
    logical_c: int,
    wall_ms: float,
) -> None:
    with pytest.raises(ValueError, match="positive"):
        _aggregate_goodput_tokens_per_s(logical_c, wall_ms)
