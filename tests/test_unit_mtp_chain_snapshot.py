"""Real proposer snapshot/publication methods with byte-backed fake device memory."""
import ctypes
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.speculative import mtp_native


@pytest.mark.parametrize("row", [0, 1, 2])
@pytest.mark.parametrize("published", [False, True])
def test_restore_publishes_saved_winner_not_overwritten_scoring_scratch(monkeypatch, row, published):
    arrays = []

    def buffer(shape, dtype):
        data = np.zeros(shape, dtype=dtype)
        arrays.append(data)
        return SimpleNamespace(ptr=data.ctypes.data, nbytes=data.nbytes)

    proposer = object.__new__(mtp_native.NativeMtpChainProposer)
    proposer.runtime = SimpleNamespace(memcpy_async=lambda dst, src, n, *_: ctypes.memmove(dst, src, n))
    proposer.hidden = 2
    proposer.max_mtp_tokens = 3
    proposer.max_candidate_rows = 3
    proposer.final_hidden_buf = buffer(2, np.uint16)
    proposer.snapshot_hidden_buf = buffer(6, np.uint16)
    proposer.candidate_ids_buf = buffer(3, np.int32)
    proposer.w8_out_index_buf = SimpleNamespace(ptr=proposer.candidate_ids_buf.ptr, nbytes=4)
    proposer._candidate_index_buf = proposer.w8_out_index_buf
    proposer._candidate_index_nbytes = 4
    proposer.snapshot_index_buf = buffer(12, np.uint8)
    proposer.snapshot_token_buf = buffer(3, np.int64)
    proposer.token_buf = buffer(1, np.int64)
    proposer.scoring_head = SimpleNamespace(threads=256)
    proposer.runtime_state_lib = object()
    proposer.position = 7
    proposer.cache_len = 6
    proposer.current = mtp_native.NativeMtpStepResult(29, 1., (), ())
    proposer.final_hidden_valid = True
    proposer.result_valid = False
    proposer._candidate_ready = True
    proposer._candidate_row = row
    proposer._chain_token_current = published
    ids = np.ctypeslib.as_array((ctypes.c_int32 * 3).from_address(proposer.candidate_ids_buf.ptr))
    token = np.ctypeslib.as_array((ctypes.c_int64 * 1).from_address(proposer.token_buf.ptr))
    ids[:] = [11, 29, 47]
    saved_winner = int(ids[row])
    token[0] = saved_winner if published else -1

    def cast(src, dst, count, **kwargs):
        assert count == 1
        ctypes.c_int64.from_address(dst).value = ctypes.c_int32.from_address(src).value

    monkeypatch.setattr(mtp_native, "copy_i32_to_i64", cast, raising=False)
    snapshot = proposer.save_state(0)
    ids[:] = [101, 102, 103]  # A later score replaced all device candidate rows.
    token[0] = 103
    proposer._candidate_row = 2
    proposer.restore_state(snapshot)
    assert proposer._candidate_row == row
    assert ids[row] == saved_winner
    proposer.ensure_chain_token()
    assert token[0] == saved_winner
