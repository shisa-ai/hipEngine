"""Native runtime uses load-time qualification and fixed session allocations.

Private field mutation is not a supported reconfiguration API. These tests
replace the experimental per-call pointer-inventory/authorization contracts.
"""
import pytest

from tests.test_gguf_execution_authorization import native_resident
from tests.test_gguf_ud_admission import _native_entry_session, _PositionOwnerSentinel


@pytest.mark.parametrize("entry", ["eager", "capture"])
def test_native_entry_does_not_requalify_resident(monkeypatch, tmp_path, entry):
    from hipengine.loading import qwen35_gguf_execution as execution
    resident = native_resident(monkeypatch, tmp_path)
    owner = _PositionOwnerSentinel()
    session = _native_entry_session(resident, scratch_owner=owner)

    def forbidden(*args, **kwargs):
        pytest.fail("execution repeated load-time qualification")

    monkeypatch.setattr(execution, "authorize_native_execution", forbidden)
    monkeypatch.setattr(execution, "resident_snapshot", forbidden)

    def reached_scratch(*args, **kwargs):
        raise RuntimeError("reached scratch")

    session._native_compact_scratch = reached_scratch
    with pytest.raises(RuntimeError, match="reached scratch"):
        if entry == "eager":
            session.step_rows_native((1, 2))
        else:
            session.capture_native_rows_graph(rows=2, max_context_len=64)
    assert owner.calls == [(0, 0)]


@pytest.mark.parametrize("ready,rows,capacity", [(False, 2, 8), (True, 1, 8), (True, 9, 16), (True, 4, 2)])
def test_native_route_and_row_bounds_remain_explicit(ready, rows, capacity):
    from types import SimpleNamespace
    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession
    session = SimpleNamespace(_native_rows_ready=ready, max_batch_size=capacity,
                              use_expert_sidecar=False, host_token_embedding_enabled=False)
    with pytest.raises(ValueError, match="ar_decode_native_rows"):
        Qwen35GGUFResidentSession._require_native_rows(session, rows)


def test_native_graph_replays_dynamic_tokens_without_requalification(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from hipengine.loading import qwen35_gguf_execution as execution
    from hipengine.runtime import qwen35_gguf_runner as runtime
    resident = native_resident(monkeypatch, tmp_path)
    owner = _PositionOwnerSentinel()
    session = _native_entry_session(resident, scratch_owner=owner)
    session._native_compact_scratch = lambda *a, **kw: owner
    session.runtime = SimpleNamespace(
        stream_create=lambda: 1, stream_begin_capture=lambda s: None,
        stream_end_capture=lambda s: 2, graph_instantiate=lambda g: 3,
        graph_launch=lambda *a: None, stream_synchronize=lambda *a: None)
    session._enqueue_native_rows_model = lambda *a, **kw: ({}, {})
    graph = session.capture_native_rows_graph(rows=2, max_context_len=64)
    monkeypatch.setattr(execution, "authorize_native_execution",
                        lambda *a, **kw: pytest.fail("graph requalified resident"))
    monkeypatch.setattr(execution, "resident_snapshot",
                        lambda *a, **kw: pytest.fail("graph scanned resident"))
    ptrs = []
    monkeypatch.setattr(runtime, "copy_host_to_device",
                        lambda buf, *a, **kw: ptrs.append(("h2d", buf.ptr)))
    def readback(host_ptr, buffer, nbytes, **kwargs):
        assert host_ptr == session._native_token_ids_host.ctypes.data
        assert nbytes == 8
        ptrs.append(("d2h", buffer.ptr))
        session._native_token_ids_host[:2] = (7, 8)
    monkeypatch.setattr(runtime, "copy_device_to_host", readback)
    owner.position_host[:] = 3
    session._position = 3
    result = graph.step((1, 2))
    assert result.token_ids == (7, 8)
    assert result.positions == (3, 3)
    assert ptrs == [("h2d", session._token_buf.ptr), ("d2h", session._lm_out_index.ptr)]
    assert owner.calls[-1] == (4, 4)


def test_native_graph_rejects_closed_session_before_device_work():
    from types import SimpleNamespace
    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFNativeRowsGraph
    session = SimpleNamespace(_token_buf=None, _native_token_ids_host=None)
    graph = Qwen35GGUFNativeRowsGraph(session, 1, 2, 3, 2, 64, "decode", {})
    with pytest.raises(RuntimeError, match="buffers are closed"):
        graph.step((1, 2))
