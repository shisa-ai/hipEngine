from __future__ import annotations

from pathlib import Path

import pytest


_SOURCE = (
    Path(__file__).parents[1]
    / "hipengine/kernels/hip_gfx1100/attention/aotriton_wrap.cc"
)


def _extern_body(source: str, symbol: str) -> str:
    body = source.split(f'extern "C" int {symbol}', 1)[1]
    return body.split('\nextern "C" int ', 1)[0]


@pytest.mark.parametrize(
    "symbol",
    (
        "hipengine_aotriton_check_gpu",
        "hipengine_aotriton_attn_fwd_compact_varlen",
        "hipengine_aotriton_attn_fwd_v3_compact_varlen",
    ),
)
def test_aotriton_vendor_calls_cannot_throw_across_c_abi(symbol: str) -> None:
    body = _extern_body(_SOURCE.read_text(encoding="utf-8"), symbol)

    assert "noexcept" in body.split("{", 1)[0]
    assert "try {" in body
    assert "catch (...)" in body
    assert "hipErrorUnknown" in body
