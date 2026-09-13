from __future__ import annotations

import numpy as np
import pytest

from scripts.gguf_output_head_compare import (
    compare_output_head_logits,
    stream_lm_head_logits_from_chunks,
)


def test_stream_lm_head_logits_from_chunks_matches_dense_matmul() -> None:
    hidden = np.asarray([0.5, -1.0, 2.0], dtype=np.float32)
    weight = np.asarray(
        [
            [1.0, 0.0, 0.25],
            [-0.5, 0.25, 1.0],
            [0.0, -2.0, 0.5],
            [1.5, 1.0, -0.25],
        ],
        dtype=np.float32,
    )

    observed = stream_lm_head_logits_from_chunks(
        [(0, weight[:2]), (2, weight[2:])],
        hidden,
        vocab_size=4,
    )

    np.testing.assert_allclose(observed, weight @ hidden, rtol=0.0, atol=0.0)


def test_stream_lm_head_logits_from_chunks_rejects_bad_coverage() -> None:
    hidden = np.ones((2,), dtype=np.float32)
    with pytest.raises(ValueError, match="cover every vocab row"):
        stream_lm_head_logits_from_chunks(
            [(0, np.ones((1, 2), dtype=np.float32))],
            hidden,
            vocab_size=2,
        )
    with pytest.raises(ValueError, match="outside vocab_size"):
        stream_lm_head_logits_from_chunks(
            [(1, np.ones((2, 2), dtype=np.float32))],
            hidden,
            vocab_size=2,
        )


def test_compare_output_head_logits_reports_topk_and_diff_metrics() -> None:
    cpu = np.asarray([1.0, 4.0, 3.5, -1.0, 2.0], dtype=np.float32)
    device = np.asarray([1.25, 3.75, 3.75, -1.0, 2.5], dtype=np.float32)

    comparison = compare_output_head_logits(device, cpu, top_k=3)

    assert comparison["device_top1_token_id"] == 1
    assert comparison["cpu_top1_token_id"] == 1
    assert comparison["top1_match"] is True
    assert comparison["topk_overlap_count"] == 3
    assert comparison["diff"]["max_abs_diff"] == pytest.approx(0.5)
    selected = {item["token_id"]: item for item in comparison["selected_logits"]}
    assert selected[4]["device_logit"] == pytest.approx(2.5)


def test_compare_output_head_logits_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="same shape"):
        compare_output_head_logits(
            np.ones((2,), dtype=np.float32),
            np.ones((3,), dtype=np.float32),
        )
