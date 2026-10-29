"""Fused GR decode admission depends on geometry, not a retired opt-out."""

import pytest

from hipengine.runtime.qwen4_exp_runner import _qwen4_exp_gr_decode_fused_enabled


@pytest.mark.parametrize("stale", ["0", "false", "1"])
@pytest.mark.parametrize("shape,expected", [
    ((1, 4, 2560, 320), True),
    ((1, 4, 512, 64), True),
    ((2, 4, 512, 64), False),
    ((1, 2, 512, 64), False),
    ((1, 4, 513, 64), False),
    ((1, 4, 512, 0), False),
    ((1, 4, 512, 65), False),
])
def test_gr_decode_fusion_ignores_retired_environment(monkeypatch, stale, shape, expected):
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_GR_DECODE_FUSED", stale)
    assert _qwen4_exp_gr_decode_fused_enabled(*shape) is expected
