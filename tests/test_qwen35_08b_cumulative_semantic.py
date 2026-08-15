from __future__ import annotations

import os

import numpy as np
import pytest

from scripts import qwen35_08b_cumulative_semantic as cumulative


def test_expand_to_512_repeats_complete_prompt_tokens() -> None:
    expanded = cumulative.expand_to_512([11, 22, 33])

    assert len(expanded) == 512
    assert expanded[:9] == [11, 22, 33] * 3
    assert expanded[-2:] == [11, 22]
    with pytest.raises(ValueError, match="empty prompt"):
        cumulative.expand_to_512([])


def test_role_environment_restores_every_selector(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HIPENGINE_GGUF_Q4_PACK8_WMMA_BULK", "before")
    monkeypatch.delenv("HIPENGINE_GGUF_GDN_PREFILL_MODE", raising=False)

    with cumulative.role_environment("q4", "strict_x2"):
        assert os.environ["HIPENGINE_GGUF_Q4_PACK8_WMMA_BULK"] == "0"
        assert os.environ["HIPENGINE_GGUF_DENSE_WMMA_BULK"] == "0"
        assert os.environ["HIPENGINE_GGUF_GDN_PREFILL_MODE"] == "exact"
        assert os.environ["HIPENGINE_GGUF_HOST_TOKEN_EMBEDDING"] == "0"

    assert os.environ["HIPENGINE_GGUF_Q4_PACK8_WMMA_BULK"] == "before"
    assert "HIPENGINE_GGUF_GDN_PREFILL_MODE" not in os.environ


def test_trajectory_digest_covers_tokens_and_logits() -> None:
    baseline = [
        {"token_id": 7, "logits": np.asarray([1.0, 2.0], dtype=np.float32)},
        {"token_id": 8, "logits": np.asarray([3.0, 4.0], dtype=np.float32)},
    ]
    changed = [dict(row) for row in baseline]
    changed[1] = {"token_id": 9, "logits": changed[1]["logits"]}

    assert cumulative.trajectory_digest(baseline) == cumulative.trajectory_digest(baseline)
    assert cumulative.trajectory_digest(baseline) != cumulative.trajectory_digest(changed)
