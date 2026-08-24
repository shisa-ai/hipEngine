from __future__ import annotations

import numpy as np
import pytest

from scripts.mtp_paro_fast_state_review import (
    _audit_cycle_ids,
    _bf16_finite,
    _fp16_finite,
    _tensor_host_finite,
)


def test_bf16_and_fp16_finite_checks_reject_nan_and_inf() -> None:
    bf16_finite = np.asarray([0x0000, 0x3F80, 0xFF7F], dtype=np.uint16)
    bf16_bad = np.asarray([0x7F80, 0x7FC0], dtype=np.uint16)
    fp16_finite = np.asarray([0x0000, 0x3C00, 0xFBFF], dtype=np.uint16)
    fp16_bad = np.asarray([0x7C00, 0x7E00], dtype=np.uint16)

    assert _bf16_finite(bf16_finite) is True
    assert _bf16_finite(bf16_bad) is False
    assert _fp16_finite(fp16_finite) is True
    assert _fp16_finite(fp16_bad) is False
    assert _tensor_host_finite(np.asarray([0.0, 1.0], dtype=np.float32), "fp32")


def test_audit_cycles_cover_first_last_reject_accept_and_mismatch() -> None:
    cycles = [
        {"cycle": 1, "strict_accepted": 1, "task_decision_mismatch": False},
        {"cycle": 2, "strict_accepted": 0, "task_decision_mismatch": False},
        {"cycle": 3, "strict_accepted": 1, "task_decision_mismatch": True},
        {"cycle": 4, "strict_accepted": 1, "task_decision_mismatch": False},
    ]
    assert _audit_cycle_ids(cycles) == {1, 2, 3, 4}


def test_audit_cycles_reject_empty_schedule() -> None:
    with pytest.raises(ValueError, match="at least one cycle"):
        _audit_cycle_ids([])
