from __future__ import annotations

import pytest

from hipengine.runtime.qwen4_exp_runner import _qwen4_exp_q8_mmq_policy


def test_qwen4_exp_q8_mmq_attn_gate_scope_is_explicit_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hipengine.kernels.hip_gfx1100.quant.gguf_q8_0_mmq_prefill import (
        Q8MMQPrefillPolicy,
    )

    base = Q8MMQPrefillPolicy(
        min_rows={(2560, 10240): 64},
        max_rows=2048,
        risk_threshold=0.0,
        max_out_features=12288,
    )
    monkeypatch.delenv("HIPENGINE_QWEN4_EXP_Q8_MMQ_ATTN_GATE", raising=False)
    assert _qwen4_exp_q8_mmq_policy(base) is base
    assert base(508, 2560, 6144) is False

    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_Q8_MMQ_ATTN_GATE", "1")
    candidate = _qwen4_exp_q8_mmq_policy(base)

    assert candidate is not base
    assert candidate(508, 2560, 6144) is True
    assert candidate(508, 2560, 10240) is True
    assert base(508, 2560, 6144) is False
