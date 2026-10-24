"""Restored hidden/cursors must not expose a discarded head result."""
from types import SimpleNamespace

import pytest

from hipengine.speculative.mtp_native import NativeMtpChainProposer, NativeMtpStepResult


@pytest.mark.parametrize("hidden_valid", [False, True])
def test_restore_invalidates_unsnapshotted_device_scores(hidden_valid):
    proposer = NativeMtpChainProposer.__new__(NativeMtpChainProposer)
    copies = []
    proposer.runtime = SimpleNamespace(memcpy_async=lambda *args: copies.append(args))
    proposer.max_mtp_tokens = 4
    proposer.hidden = 8
    proposer.snapshot_hidden_buf = SimpleNamespace(ptr=1024)
    proposer.final_hidden_buf = SimpleNamespace(ptr=2048)
    proposer.cache_len = 2
    proposer.position = 5
    proposer.current = NativeMtpStepResult(11, 1.0, (), ())
    proposer.final_hidden_valid = hidden_valid
    proposer.result_valid = hidden_valid
    snapshot = proposer.save_state(1)
    proposer.cache_len = 4
    proposer.position = 7
    proposer.current = NativeMtpStepResult(99, 3.0, (), ())
    proposer.final_hidden_valid = True
    proposer.result_valid = True
    proposer.restore_state(snapshot)
    assert (proposer.cache_len, proposer.position, proposer.current.token) == (2, 5, 11)
    assert proposer.final_hidden_valid == hidden_valid
    assert proposer.result_valid is False
    assert len(copies) == 2
    assert copies[0][:3] == (1040, 2048, 16)
    assert copies[1][:3] == (2048, 1040, 16)
    for query in (proposer.device_candidate_token_ids, proposer.vocab_topk, proposer.top1_prob_proxy):
        with pytest.raises(RuntimeError, match="result-producing advance"):
            query()
