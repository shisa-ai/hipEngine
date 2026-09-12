from __future__ import annotations

import ctypes

import pytest

from hipengine.runtime.qwen4_exp_runner import Qwen4ExpTargetVerifyOutput


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


@pytest.mark.parametrize("rows", (0, 9))
def test_qwen4_exp_target_verify_output_rejects_unbounded_rows(rows: int) -> None:
    with pytest.raises(ValueError, match="rows must be in 1..8"):
        Qwen4ExpTargetVerifyOutput.allocate(
            rows=rows,
            branches=4,
            hidden=16,
            low_rank=4,
            vocab=32,
            runtime=object(),
        )


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_qwen4_exp_target_verify_output_has_bounded_rows_and_clean_close() -> None:
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import memory_stats

    runtime = get_hip_runtime()
    before = memory_stats()["current_allocated_bytes"]
    output = Qwen4ExpTargetVerifyOutput.allocate(
        rows=8,
        branches=4,
        hidden=16,
        low_rank=4,
        vocab=32,
        runtime=runtime,
    )
    assert output.rows_capacity == 8
    assert output.nbytes_by_owner == {
        "residual_rows": 8 * 4 * 16 * 2,
        "logits_rows": 8 * 32 * 4,
        "token_ids": 8 * 8,
    }
    output.require_rows(1)
    output.require_rows(8)
    with pytest.raises(ValueError, match="rows must be in 1..8"):
        output.require_rows(9)

    output.close()
    output.close()
    assert output.closed
    assert memory_stats()["current_allocated_bytes"] == before
    with pytest.raises(RuntimeError, match="closed"):
        output.require_rows(1)
