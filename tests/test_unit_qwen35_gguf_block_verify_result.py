from __future__ import annotations

import numpy as np
import pytest

from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFBlockVerifyResult


def test_block_verify_result_accepts_optional_lm_head_logits() -> None:
    result = Qwen35GGUFBlockVerifyResult(
        input_token_ids=[10, 11],
        token_ids=[20, 21],
        hidden_seeds=np.zeros((2, 4), dtype=np.float32),
        start_position=7,
        lm_head_logits_f32=np.zeros((2, 8), dtype=np.float32),
    )

    assert result.lm_head_logits_f32 is not None
    assert result.lm_head_logits_f32.shape == (2, 8)


def test_block_verify_result_rejects_bad_lm_head_logits_shape() -> None:
    with pytest.raises(ValueError, match="rows must match"):
        Qwen35GGUFBlockVerifyResult(
            input_token_ids=[10, 11],
            token_ids=[20, 21],
            hidden_seeds=np.zeros((2, 4), dtype=np.float32),
            start_position=7,
            lm_head_logits_f32=np.zeros((1, 8), dtype=np.float32),
        )
