from __future__ import annotations

import numpy as np

from scripts.mtp_paro_proposal_parity import _cpu_w8_logits, _stable_topk


def test_stable_topk_breaks_equal_scores_by_lower_token_id() -> None:
    values = np.asarray([1.0, 3.0, 3.0, -1.0], dtype=np.float32)
    assert _stable_topk(values, 3) == [1, 2, 0]


def test_cpu_w8_logits_matches_explicit_quantize_and_dot() -> None:
    weight = np.asarray(
        [
            [1.0, -2.0, 0.5, 0.25],
            [-0.5, 0.75, 1.5, -1.25],
            [0.125, 0.0, -0.25, 0.5],
        ],
        dtype=np.float16,
    )
    hidden = np.asarray([[0.5, -1.0, 0.25, 2.0]], dtype=np.float32)
    actual = _cpu_w8_logits(weight, hidden, chunk_rows=2)

    wf = weight.astype(np.float32)
    scale = np.maximum(np.max(np.abs(wf), axis=1), np.float32(1.0e-8)) / np.float32(127.0)
    q = np.clip(np.rint(wf / scale[:, None]), -127, 127).astype(np.int8)
    expected = (q.astype(np.float32) @ hidden.reshape(-1)) * scale
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)
