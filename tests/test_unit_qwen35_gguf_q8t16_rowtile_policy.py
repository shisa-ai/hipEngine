from __future__ import annotations

from hipengine.kernels.backends import backend_package_capability
from hipengine.runtime.qwen35_gguf_runner import (
    _gguf_q8_t16_decode_rowtile_all_for_rows,
)


def _gfx1100_pair_rowtile_admits(rows: int) -> bool:
    """Mirror the packed-decode enqueue pair-rowtile admission exactly."""

    min_rows = int(
        backend_package_capability(
            "hip_gfx1100",
            "GGUF_Q8_T16_DECODE_PAIR_ROWTILE_MIN_ROWS",
            0,
        )
    )
    return min_rows > 0 and rows >= min_rows


def test_gfx1151_q8t16_all_rowtile_floor_excludes_c2() -> None:
    assert _gguf_q8_t16_decode_rowtile_all_for_rows("hip_gfx1151", rows=2) is False
    assert _gguf_q8_t16_decode_rowtile_all_for_rows("hip_gfx1151", rows=4) is True
    assert _gguf_q8_t16_decode_rowtile_all_for_rows("hip_gfx1151", rows=8) is True


def test_gfx1100_q8t16_all_rowtile_floor_remains_disabled() -> None:
    assert _gguf_q8_t16_decode_rowtile_all_for_rows("hip_gfx1100", rows=2) is False
    assert _gguf_q8_t16_decode_rowtile_all_for_rows("hip_gfx1100", rows=4) is False
    assert _gguf_q8_t16_decode_rowtile_all_for_rows("hip_gfx1100", rows=8) is False


def test_gfx1100_q8t16_pair_rowtile_floor_admits_c8_only() -> None:
    """W7900 pair rowtile retention (audit packet C2): floor 8 admits the
    exact qkv+gate dual rowtile only at physical c8; c2/c4 keep per-row
    fallbacks and the broad all-projection route stays rejected."""

    assert _gfx1100_pair_rowtile_admits(2) is False
    assert _gfx1100_pair_rowtile_admits(4) is False
    assert _gfx1100_pair_rowtile_admits(8) is True
