"""CPU contracts for chain packing and independent tree metadata uploads."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.memory import DeviceBuffer
from hipengine.core.tensor import Tensor
from hipengine.runtime import qwen35_paro_runner as runner
from hipengine.speculative import DraftBatch, TargetVerifyBatch


def _batch(rows, mode):
    draft = DraftBatch(
        request_ids=(7,), candidate_tokens=tuple(range(31, 30 + rows)),
        parent_positions=tuple(range(8, 7 + rows)),
        draft_depths=tuple(range(1, rows)), row_to_request=(7,) * (rows - 1),
        tree_parents=tuple(range(-1, rows - 2)),
    )
    return replace(TargetVerifyBatch.from_draft(draft, root_tokens=(30,), root_positions=(8,)), mode=mode)


def _session(monkeypatch):
    session = runner.Qwen35ParoResidentSession.__new__(runner.Qwen35ParoResidentSession)
    names = (
        "verify_token_ids_i64", "verify_token_ids_i32", "prefill_position_buf",
        "verify_positions_i32", "prefill_context_count_buf", "verify_parent_rows_i32",
        "verify_parent_rows_i64", "verify_draft_depths_i32", "verify_row_to_request_i32",
        "verify_active_mask_u8", "verify_candidate_counts_i32", "prefill_block_table_buf",
        "verify_dynamic_metadata_i64", "verify_ancestor_mask_u8", "verify_cache_slot_buf",
        "verify_tree_committed_buf",
    )
    for index, name in enumerate(names):
        setattr(session, name, DeviceBuffer(0x1000 + index * 0x1000, 4096))
    session.blocks = 3
    session.runtime = SimpleNamespace()
    session.libraries = {"runtime_state": object()}
    session._ensure_prefill_block_table_capacity = lambda rows, uniform: None
    session._ensure_verify_tree_uniform_buf = lambda **kwargs: None
    copies = []
    events = []
    by_ptr = {getattr(session, name).ptr: name for name in names}

    def copy(buffer, array, nbytes, *, runtime):
        assert runtime is session.runtime and nbytes == array.nbytes
        copies.append((by_ptr[buffer.ptr], array.copy()))
        events.append(("h2d", by_ptr[buffer.ptr]))

    def unpack(*args, **kwargs):
        assert args[:6] == tuple(getattr(session, name).ptr for name in (
            "verify_dynamic_metadata_i64", "verify_token_ids_i64", "verify_token_ids_i32",
            "prefill_position_buf", "verify_positions_i32", "prefill_context_count_buf",
        ))
        assert kwargs == {"stream": 13, "library": session.libraries["runtime_state"], "runtime": session.runtime}
        events.append(("unpack", args[6]))

    session.runtime.memcpy_async = lambda *args: events.append(("d2d", args))
    monkeypatch.setattr(runner, "copy_host_array_to_device", copy)
    monkeypatch.setattr(runner, "unpack_verify_chain_dynamic_metadata_i64", unpack)
    monkeypatch.setattr(runner, "copy_i32_to_i64", lambda *args, **kwargs: events.append(("widen", args, kwargs)))
    return session, copies, events


@pytest.mark.parametrize("retired", [None, "0", "false", "1"])
@pytest.mark.parametrize("rows", [2, 4, 7])
@pytest.mark.parametrize("device_candidates", [False, True])
def test_chain_always_packs_dynamic_values_and_preserves_topology_owner(
    monkeypatch, retired, rows, device_candidates,
):
    if retired is None:
        monkeypatch.delenv("HIPENGINE_VERIFY_PACK_DYNAMIC_METADATA", raising=False)
    else:
        monkeypatch.setenv("HIPENGINE_VERIFY_PACK_DYNAMIC_METADATA", retired)
    session, copies, events = _session(monkeypatch)
    batch = _batch(rows, "verify_chain")
    candidate = Tensor.from_handle(0xA0000, (rows - 1,), DType.INT32, Device("hip", 0)) if device_candidates else None
    for step, owner_change in enumerate((False, False, True)):
        copies.clear()
        events.clear()
        if owner_change:
            session._prefill_block_table_owner = ("prefill",)
        current = replace(batch, tokens=tuple(token + step * 101 for token in batch.tokens), positions=tuple(pos + step * 17 for pos in batch.positions))
        session._write_verify_chain_metadata(current, base_slot=2, stream=13, candidate_token_ids_i32=candidate)
        uploads = dict(copies)
        expected = np.asarray([
            (token, token, pos, pos, pos + 1)
            for token, pos in zip(current.tokens, current.positions, strict=True)
        ], dtype=np.int64)
        np.testing.assert_array_equal(uploads["verify_dynamic_metadata_i64"], expected)
        assert len(copies) == (1 if step == 1 else 8)
        if step != 1:
            np.testing.assert_array_equal(uploads["prefill_block_table_buf"], np.tile(np.arange(6, 9, dtype=np.int32), (rows, 1)))
            np.testing.assert_array_equal(uploads["verify_parent_rows_i32"], batch.parent_rows)
        assert events[-(3 if device_candidates else 1)] == ("unpack", rows)
        if device_candidates:
            assert events[-2][0] == "d2d"
            assert events[-2][1][:3] == (session.verify_token_ids_i32.ptr + 4, candidate.ptr, (rows - 1) * 4)
            assert events[-2][1][-1] == 13
            assert events[-1] == ("widen", (candidate.ptr, session.verify_token_ids_i64.ptr + 8, rows - 1), {"stream": 13, "library": session.libraries["runtime_state"], "runtime": session.runtime})
        assert session.verify_tree_committed_count == 0


@pytest.mark.parametrize("retired", [None, "0", "false", "1"])
def test_tree_keeps_separate_metadata_and_ancestor_uploads(monkeypatch, retired):
    if retired is None:
        monkeypatch.delenv("HIPENGINE_VERIFY_PACK_DYNAMIC_METADATA", raising=False)
    else:
        monkeypatch.setenv("HIPENGINE_VERIFY_PACK_DYNAMIC_METADATA", retired)
    session, copies, events = _session(monkeypatch)
    batch = _batch(4, "verify_tree")
    session._write_verify_chain_metadata(batch, base_slot=1, stream=13)
    uploads = dict(copies)
    assert "verify_dynamic_metadata_i64" not in uploads
    np.testing.assert_array_equal(uploads["verify_token_ids_i64"], batch.tokens)
    np.testing.assert_array_equal(uploads["verify_token_ids_i32"], batch.tokens)
    np.testing.assert_array_equal(uploads["prefill_position_buf"], batch.positions)
    np.testing.assert_array_equal(uploads["prefill_context_count_buf"], np.asarray(batch.positions) + 1)
    np.testing.assert_array_equal(uploads["verify_ancestor_mask_u8"], np.tri(4, dtype=np.uint8))
    np.testing.assert_array_equal(uploads["verify_cache_slot_buf"], np.arange(8, 12))
    assert session.verify_tree_committed_count == 8
    assert not any(event[0] == "unpack" for event in events)
