from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.core.dtype import DType
from hipengine.core.memory import DeviceBuffer
from hipengine.runtime import qwen35_gguf_runner as gguf_runner
from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession


class _FakeRuntime:
    def __init__(self) -> None:
        self.memcpy_async_calls: list[tuple[int, int, int, object, int]] = []

    def memcpy_async(self, dst: int, src: int, nbytes: int, kind: object, stream: int) -> None:
        self.memcpy_async_calls.append((int(dst), int(src), int(nbytes), kind, int(stream)))


def _fake_gguf_commit_session() -> Qwen35GGUFResidentSession:
    session = Qwen35GGUFResidentSession.__new__(Qwen35GGUFResidentSession)
    session.runtime = _FakeRuntime()
    session.runner = SimpleNamespace(hidden_size=4)
    session.scratch = SimpleNamespace(
        layer_conv_states=(
            DeviceBuffer(0x1000, 64),
            None,
            DeviceBuffer(0x2000, 64),
        ),
        layer_recurrent_states=(
            DeviceBuffer(0x3000, 128),
            None,
            DeviceBuffer(0x4000, 128),
        ),
        hidden_seed_fp32=DeviceBuffer(0x5000, 16),
        position_host=np.zeros((1,), dtype=np.int64),
        context_host=np.zeros((1,), dtype=np.int64),
        position_buf=DeviceBuffer(0x6000, 8),
        context_buf=DeviceBuffer(0x7000, 8),
    )
    session._verify_hidden_seed_buf = DeviceBuffer(0x8000, 2 * 4 * DType.FP32.itemsize)
    session._verify_linear_conv_state_rows = (
        DeviceBuffer(0x9000, 2 * 64),
        None,
        DeviceBuffer(0xA000, 2 * 64),
    )
    session._verify_linear_recurrent_state_rows = (
        DeviceBuffer(0xB000, 2 * 128),
        None,
        DeviceBuffer(0xC000, 2 * 128),
    )
    session._verify_linear_state_rows_capacity = 2
    session._verify_linear_state_src_conv_table_buf = None
    session._verify_linear_state_src_recurrent_table_buf = None
    session._verify_linear_state_dst_conv_table_buf = None
    session._verify_linear_state_dst_recurrent_table_buf = None
    session._verify_linear_state_commit_row_i32_buf = None
    session._verify_linear_state_src_conv_host = None
    session._verify_linear_state_src_recurrent_host = None
    session._verify_linear_state_src_conv_cached = None
    session._verify_linear_state_src_recurrent_cached = None
    session._verify_linear_state_dst_conv_host = None
    session._verify_linear_state_dst_recurrent_host = None
    session._verify_linear_state_conv_row_nbytes = 0
    session._verify_linear_state_recurrent_row_nbytes = 0
    session._verify_linear_state_layer_count = 0
    session._dflash_commit_library = object()
    session._runtime_state_library = object()
    session._buffers = ()
    session._position = 0
    session._hidden_seed_fp32_populated = False
    return session


def test_gguf_verify_linear_state_commit_uses_fused_chunked_kernel(monkeypatch) -> None:
    allocated: list[DeviceBuffer] = []
    h2d_calls: list[tuple[int, int]] = []
    commit_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    set_position_calls: list[tuple[int, int, int]] = []

    def fake_malloc(nbytes, *, runtime):
        _ = runtime
        buf = DeviceBuffer(0xD000 + len(allocated) * 0x100, int(nbytes))
        allocated.append(buf)
        return buf

    def fake_copy_host_to_device(buffer, host_ptr, nbytes, *, runtime):
        _ = host_ptr, runtime
        h2d_calls.append((int(buffer.ptr), int(nbytes)))

    def fake_chunked(*args, **kwargs):
        commit_calls.append((args, kwargs))

    monkeypatch.delenv("HIPENGINE_FUSED_LINEAR_STATE_COMMIT", raising=False)
    monkeypatch.delenv("HIPENGINE_LINEAR_STATE_COMMIT_CHUNKED", raising=False)
    monkeypatch.setattr(gguf_runner, "malloc", fake_malloc)
    monkeypatch.setattr(gguf_runner, "copy_host_to_device", fake_copy_host_to_device)
    monkeypatch.setattr(gguf_runner, "linear_state_pair_commit_chunked_i32", fake_chunked)
    monkeypatch.setattr(gguf_runner, "linear_state_pair_commit_i32", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        gguf_runner,
        "set_decode_position_i64",
        lambda position_ptr, context_ptr, position, **kwargs: set_position_calls.append(
            (int(position_ptr), int(context_ptr), int(position))
        ),
    )

    session = _fake_gguf_commit_session()
    session._commit_verify_linear_state_row(1, position=17, stream=5)

    assert len(commit_calls) == 1
    commit_args, commit_kwargs = commit_calls[0]
    assert commit_args[2] == 64
    assert commit_args[5] == 128
    assert commit_args[7] == 2
    assert commit_kwargs["stream"] == 5
    assert len(allocated) == 5
    assert len(h2d_calls) == 5
    assert session.runtime.memcpy_async_calls == [
        (
            0x5000,
            0x8000 + 1 * 4 * DType.FP32.itemsize,
            4 * DType.FP32.itemsize,
            gguf_runner.HipMemcpyKind.DEVICE_TO_DEVICE,
            5,
        )
    ]
    assert set_position_calls == [(0x6000, 0x7000, 17)]
    assert session.position == 17
    assert bool(session._hidden_seed_fp32_populated)


def test_gguf_verify_linear_state_commit_rejects_partial_captured_rows(monkeypatch) -> None:
    chunked_calls: list[object] = []

    monkeypatch.delenv("HIPENGINE_FUSED_LINEAR_STATE_COMMIT", raising=False)
    monkeypatch.setattr(
        gguf_runner,
        "linear_state_pair_commit_chunked_i32",
        lambda *args, **kwargs: chunked_calls.append(args),
    )

    session = _fake_gguf_commit_session()
    session._verify_linear_conv_state_rows = (
        session._verify_linear_conv_state_rows[0],
        None,
        None,
    )

    with pytest.raises(RuntimeError, match="linear-state rows for layer 2 were not captured"):
        session._commit_verify_linear_state_row(1, position=17, stream=5)

    assert chunked_calls == []
