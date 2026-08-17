from __future__ import annotations

from hipengine.runtime.qwen35_gguf_runner import (
    _gguf_q8_t16_decode_rowtile_all_for_rows,
)


def test_gfx1151_q8t16_all_rowtile_floor_excludes_c2() -> None:
    assert _gguf_q8_t16_decode_rowtile_all_for_rows("hip_gfx1151", rows=2) is False
    assert _gguf_q8_t16_decode_rowtile_all_for_rows("hip_gfx1151", rows=4) is True
    assert _gguf_q8_t16_decode_rowtile_all_for_rows("hip_gfx1151", rows=8) is True


def test_gfx1100_q8t16_all_rowtile_floor_remains_disabled() -> None:
    assert _gguf_q8_t16_decode_rowtile_all_for_rows("hip_gfx1100", rows=2) is False
    assert _gguf_q8_t16_decode_rowtile_all_for_rows("hip_gfx1100", rows=4) is False
    assert _gguf_q8_t16_decode_rowtile_all_for_rows("hip_gfx1100", rows=8) is False
