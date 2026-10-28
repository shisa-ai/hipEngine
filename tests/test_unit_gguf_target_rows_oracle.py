"""CPU contracts for the retained scalar target-chain oracle."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession
from hipengine.speculative import TargetVerifyBatch


def _batch(*, branched=False):
    return TargetVerifyBatch(
        request_ids=(17,), tokens=(1, 2, 3), positions=(0, 1, 2),
        row_to_request=(17, 17, 17), parent_rows=(-1, 0, 0 if branched else 1),
        root_rows=(0,), candidate_rows=(1, 2), draft_depths=(0, 1, 2),
        active_mask=(True, True, True), mode="verify_chain",
    )


def _session():
    session = object.__new__(Qwen35GGUFResidentSession)
    session.max_batch_size = 1
    calls = []

    def step(tokens, *, positions, slot_indices, **kwargs):
        calls.append((tokens, positions, slot_indices, kwargs))
        return SimpleNamespace(token_ids=(tokens[0] + 10,),
                               logits=np.array([[float(tokens[0])]], dtype=np.float32),
                               layer_hidden_bits={})

    session.step_rows = step
    return session, calls


@pytest.mark.parametrize("failure", ["branch", "inactive", "tree"])
def test_scalar_verifier_refuses_unsupported_topology_before_any_row_mutates(failure):
    session, calls = _session()
    batch = _batch(branched=failure == "branch")
    if failure == "inactive":
        batch = replace(batch, active_mask=(True, True, False))
    elif failure == "tree":
        batch = replace(batch, mode="verify_tree")

    with pytest.raises(NotImplementedError):
        session.verify_rows(batch, return_logits=True)

    assert calls == []


def test_scalar_verifier_preserves_row_order_positions_and_logits():
    session, calls = _session()
    batch = _batch()

    result = session.verify_rows(batch, return_logits=True, stream=23)

    assert [call[:3] for call in calls] == [
        ((token,), (position,), (0,)) for token, position in zip(batch.tokens, batch.positions)
    ]
    assert all(call[3]["span_role"] == "verify_chain" and call[3]["stream"] == 23 for call in calls)
    assert result.token_ids == (11, 12, 13)
    assert result.positions == batch.positions
    assert result.slot_indices == (0, 0, 0)
    np.testing.assert_array_equal(result.logits, np.array([[1], [2], [3]], dtype=np.float32))
