"""The resident chain always uses device-written top-1 IDs at width one."""
from __future__ import annotations

import ctypes
import inspect
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.core.memory import DeviceBuffer
from hipengine.speculative import mtp_resident_draft as draft


@pytest.mark.parametrize("top_k", [1, 2, 8])
@pytest.mark.parametrize("depths", [1, 2, 5])
def test_chain_width_one_uses_top1_gather_and_other_widths_keep_general_topk(monkeypatch, top_k, depths):
    runner = object.__new__(draft.Qwen35GGUFResidentMTPDraftRunner)
    runner.runtime = SimpleNamespace(device_synchronize=lambda: None)
    runner.hidden_size, runner.qk_head_dim, runner.vocab = 4, 2, 8
    runner.token_embd_f32 = np.arange(32, dtype=np.float32).reshape(8, 4)
    # Even a stale in-memory rollback cannot force the deleted width-one path.
    runner._q6_top1_gather_enabled = False
    monkeypatch.setenv("HIPENGINE_RESIDENT_MTP_DRAFT_Q6_TOP1_GATHER", "0")
    for index, name in enumerate(("token_embed", "cos_all", "sin_all", "pos_all", "ctx_all", "topk_all", "_embed_table_f32")):
        setattr(runner, name, DeviceBuffer(0x1000 + index * 0x100, 1024))
    runner._gather_lib = object()
    runner._ensure_embed_table = lambda: None
    forward, general, gathers = [], [], []
    runner._run_one = lambda *args, **kwargs: forward.append(kwargs)
    runner._topk_indices_into = lambda *args, **kwargs: general.append((args, kwargs))
    monkeypatch.setattr(draft, "gather_f32_rows_by_i32id", lambda *args, **kwargs: gathers.append((args, kwargs)))
    monkeypatch.setattr(draft, "copy_host_array_to_device", lambda *args, **kwargs: None)

    def readback(dst, src, size, **kwargs):
        out = (ctypes.c_int32 * (size // 4)).from_address(dst)
        for i in range(len(out)):
            out[i] = i % 8

    monkeypatch.setattr(draft, "copy_device_to_host", readback)
    tokens, rows, cache_len = runner._propose_chain_device(
        current_seed=DeviceBuffer(0x8000, 16), next_seed=DeviceBuffer(0x9000, 16),
        start_token=1, start_position=0, draft_n_max=depths, top_k=top_k,
        rope_cos=np.ones((8, 2), dtype=np.float32), rope_sin=np.zeros((8, 2), dtype=np.float32),
        dense_key_cache=None, dense_value_cache=None, dense_cache_len=5,
    )
    assert len(tokens) == len(rows) == len(forward) == depths
    assert cache_len == 5
    if top_k == 1:
        assert general == gathers == []
        for depth, call in enumerate(forward):
            assert call["top1_out_ptr"] == runner.topk_all.ptr + depth * 4
            assert call["top1_next_embed_ptr"] == (runner.token_embed.ptr if depth + 1 < depths else None)
    else:
        assert len(general) == depths
        assert len(gathers) == depths - 1
        assert all(c["top1_out_ptr"] is None and c["top1_next_embed_ptr"] is None for c in forward)


def test_constructor_and_native_capture_have_no_top1_rollback():
    cls = draft.Qwen35GGUFResidentMTPDraftRunner
    assert "HIPENGINE_RESIDENT_MTP_DRAFT_Q6_TOP1_GATHER" not in inspect.getsource(cls.__post_init__)
    assert "_q6_top1_gather_enabled" not in inspect.getsource(cls._capture_native_proposal_graph)
