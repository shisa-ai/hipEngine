"""Restored hidden/cursors must not expose a discarded head result.

A prefix snapshot restores two different things with two different lifetimes:

* the **scores** (top-1 index/value, block maxima, logits) are not snapshotted,
  so every score query stays invalid until a new result-producing advance;
* the **candidate identity** (the saved int32 winner, its row, the int64 chain
  token) is snapshotted, so a restored prefix can still hand the target its
  candidate or publish the chain token without scoring again.
"""
import ctypes
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.speculative import mtp_native
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
    # A prefix also owns the device candidate scalar and the int64 chain token,
    # so a restore puts the saved winner back instead of the discarded
    # advance's.
    proposer.scoring_head = SimpleNamespace(threads=256)
    proposer.candidate_ids_buf = SimpleNamespace(ptr=4096)
    proposer.max_candidate_rows = 4
    proposer._candidate_row = 0
    proposer._candidate_index_nbytes = 8
    proposer.snapshot_index_buf = SimpleNamespace(ptr=5120)
    proposer.snapshot_token_buf = SimpleNamespace(ptr=6144)
    proposer.token_buf = SimpleNamespace(ptr=7168)
    proposer._candidate_ready = True
    proposer._chain_token_current = True
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
    # Hidden, candidate scalar, chain token: saved then restored, per prefix.
    assert [copy[:3] for copy in copies] == [
        (1040, 2048, 16),
        (5128, 4096, 8),
        (6152, 7168, 8),
        (2048, 1040, 16),
        (4096, 5128, 8),
        (7168, 6152, 8),
    ]
    # Score queries read device head buffers the restore did not touch.
    for query in (proposer.vocab_topk, proposer.top1_prob_proxy):
        with pytest.raises(RuntimeError, match="result-producing advance"):
            query()
    # The candidate descriptor is snapshotted identity, not a score, so the
    # restored prefix keeps it available.
    descriptor = proposer.device_candidate_token_ids(1)
    assert descriptor.ptr == 4096
    assert descriptor.shape == (1,)


def test_rollback_keeps_the_snapshotted_candidate_proposable(monkeypatch) -> None:
    """A cancelled cycle's checkpoint still serves a count=1 candidate hand-off.

    The rollback restores the checkpoint taken before the cycle's chain, so the
    candidate the target must be handed is the saved row-0 winner. Refusing it
    because the *scores* are stale would make the recovered request unproposable
    until it scored again.
    """

    ids = np.zeros(3, dtype=np.int32)
    hidden = np.zeros(2, dtype=np.uint16)
    saved_hidden = np.zeros(6, dtype=np.uint16)
    saved_index = np.zeros(12, dtype=np.uint8)
    saved_token = np.zeros(3, dtype=np.int64)
    token = np.zeros(1, dtype=np.int64)

    proposer = NativeMtpChainProposer.__new__(NativeMtpChainProposer)
    proposer.runtime = SimpleNamespace(
        memcpy_async=lambda dst, src, count, *_: ctypes.memmove(dst, src, count)
    )
    proposer.hidden = 2
    proposer.max_mtp_tokens = 3
    proposer.max_candidate_rows = 3
    proposer.final_hidden_buf = SimpleNamespace(ptr=hidden.ctypes.data)
    proposer.snapshot_hidden_buf = SimpleNamespace(ptr=saved_hidden.ctypes.data)
    proposer.candidate_ids_buf = SimpleNamespace(ptr=ids.ctypes.data)
    proposer.snapshot_index_buf = SimpleNamespace(ptr=saved_index.ctypes.data)
    proposer.snapshot_token_buf = SimpleNamespace(ptr=saved_token.ctypes.data)
    proposer.token_buf = SimpleNamespace(ptr=token.ctypes.data)
    proposer.scoring_head = SimpleNamespace(threads=256)
    proposer.runtime_state_lib = object()
    proposer._candidate_index_nbytes = 4
    proposer.position = 127
    proposer.cache_len = 128
    proposer.current = NativeMtpStepResult(11, 1.0, (), ())
    proposer.final_hidden_valid = True
    proposer.result_valid = True
    proposer._candidate_ready = True
    proposer._chain_token_current = True
    proposer._candidate_row = 0
    ids[:] = [11, 29, 47]
    token[0] = 11

    checkpoint = proposer.save_state(0)

    # The cancelled cycle's later proposal overwrote every candidate row and
    # left a newer score and chain token behind.
    ids[:] = [101, 102, 103]
    token[0] = 103
    proposer._candidate_row = 2
    proposer._chain_token_current = False
    proposer.position = 129
    proposer.cache_len = 130

    proposer.restore_state(checkpoint)

    assert proposer.result_valid is False
    assert proposer._candidate_ready is True
    assert proposer._chain_token_current is True
    assert proposer._candidate_row == 0
    assert int(ids[0]) == 11

    # A count=1 cycle reads the restored descriptor without scoring again.
    descriptor = proposer.device_candidate_token_ids(1)
    assert descriptor.ptr == ids.ctypes.data
    assert descriptor.shape == (1,)
    restored = np.ctypeslib.as_array((ctypes.c_int32 * 1).from_address(descriptor.ptr))
    assert int(restored[0]) == 11

    # The snapshotted identity also publishes, widening the restored row 0.
    def cast(src, dst, count, **kwargs):
        assert count == 1
        ctypes.c_int64.from_address(dst).value = ctypes.c_int32.from_address(src).value

    monkeypatch.setattr(mtp_native, "copy_i32_to_i64", cast, raising=False)
    proposer.publish_chain_token()
    assert int(token[0]) == 11

    # The scores are still stale: only a new advance may answer them.
    with pytest.raises(RuntimeError, match="result-producing advance"):
        proposer.vocab_topk()
