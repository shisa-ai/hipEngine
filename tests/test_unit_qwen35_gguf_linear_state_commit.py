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


@pytest.mark.parametrize("fused_flag", [None, "0", "false", "1"])
@pytest.mark.parametrize("retired", [None, "0", "false", "1"])
def test_gguf_verify_linear_state_commit_uses_fused_chunked_kernel(monkeypatch, retired, fused_flag) -> None:
    allocated: list[DeviceBuffer] = []
    h2d_calls: list[tuple[int, int]] = []
    commit_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    set_position_calls: list[tuple[int, int, int]] = []

    def fake_malloc(nbytes, *, runtime):
        _ = runtime
        buf = DeviceBuffer(0xD000 + len(allocated) * 0x100, int(nbytes))
        allocated.append(buf)
        return buf

    def fake_copy_host_array_to_device(buffer, _array, nbytes=None, *, runtime):
        _ = runtime
        count = int(_array.nbytes if nbytes is None else nbytes)
        h2d_calls.append((int(buffer.ptr), count))

    def fake_chunked(*args, **kwargs):
        commit_calls.append((args, kwargs))

    if fused_flag is None:
        monkeypatch.delenv("HIPENGINE_FUSED_LINEAR_STATE_COMMIT", raising=False)
    else:
        monkeypatch.setenv("HIPENGINE_FUSED_LINEAR_STATE_COMMIT", fused_flag)
    if retired is None:
        monkeypatch.delenv("HIPENGINE_LINEAR_STATE_COMMIT_CHUNKED", raising=False)
    else:
        monkeypatch.setenv("HIPENGINE_LINEAR_STATE_COMMIT_CHUNKED", retired)
    monkeypatch.setattr(gguf_runner, "malloc", fake_malloc)
    monkeypatch.setattr(gguf_runner, "copy_host_array_to_device", fake_copy_host_array_to_device)
    monkeypatch.setattr(gguf_runner, "linear_state_pair_commit_chunked_i32", fake_chunked)
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


@pytest.mark.parametrize("retired", [None, "0", "false", "1"])
def test_gguf_packed_commit_selects_each_slots_final_row(monkeypatch, retired) -> None:
    session = _fake_gguf_commit_session()
    session._verify_linear_conv_state_rows = tuple(
        None if buffer is None else DeviceBuffer(buffer.ptr, 4 * 64)
        for buffer in session._verify_linear_conv_state_rows
    )
    session._verify_linear_recurrent_state_rows = tuple(
        None if buffer is None else DeviceBuffer(buffer.ptr, 4 * 128)
        for buffer in session._verify_linear_recurrent_state_rows
    )
    session._verify_linear_state_rows_capacity = 4
    session.runner.weights = SimpleNamespace(config=SimpleNamespace(
        layer_types=(gguf_runner.LINEAR_ATTENTION, gguf_runner.FULL_ATTENTION, gguf_runner.LINEAR_ATTENTION),
    ))
    allocations = []
    uploads = {}
    launches = []

    def allocate(nbytes, **kwargs):
        buffer = DeviceBuffer(0xD000 + 0x100 * len(allocations), nbytes)
        allocations.append(buffer)
        return buffer

    if retired is None:
        monkeypatch.delenv("HIPENGINE_FUSED_LINEAR_STATE_COMMIT", raising=False)
    else:
        monkeypatch.setenv("HIPENGINE_FUSED_LINEAR_STATE_COMMIT", retired)
    monkeypatch.setattr(gguf_runner, "malloc", allocate)
    monkeypatch.setattr(gguf_runner, "copy_host_array_to_device",
                        lambda buffer, array, *a, **k: uploads.update({buffer.ptr: array.copy()}))
    monkeypatch.setattr(gguf_runner, "linear_state_pair_commit_chunked_i32",
                        lambda *a, **k: launches.append((a, k)))
    destinations = {
        0: (DeviceBuffer(0x100000, 128), DeviceBuffer(0x200000, 256)),
        2: (DeviceBuffer(0x300000, 128), DeviceBuffer(0x400000, 256)),
    }
    session._commit_packed_decode_linear_state_rows(
        SimpleNamespace(slot_count=2, cu_seqlens=(0, 1, 4)),
        SimpleNamespace(linear_state_pair=destinations.__getitem__),
        runtime=session.runtime, stream=7,
    )
    assert len(launches) == 1
    args, kwargs = launches[0]
    assert args[2] == 64 and args[5] == 128 and args[7] == 4
    assert kwargs["stream"] == 7
    np.testing.assert_array_equal(uploads[args[0]], [0x9000, 0x9000 + 3 * 64, 0xA000, 0xA000 + 3 * 64])
    np.testing.assert_array_equal(uploads[args[3]], [0xB000, 0xB000 + 3 * 128, 0xC000, 0xC000 + 3 * 128])
    np.testing.assert_array_equal(uploads[args[1]], [0x100000, 0x100040, 0x300000, 0x300040])
    np.testing.assert_array_equal(uploads[args[4]], [0x200000, 0x200080, 0x400000, 0x400080])
    np.testing.assert_array_equal(uploads[args[6]], [0])
    assert session.runtime.memcpy_async_calls == []


def test_gguf_commit_preserves_nonuniform_per_layer_copy_fallback(monkeypatch) -> None:
    session = _fake_gguf_commit_session()
    session.scratch.layer_conv_states = (DeviceBuffer(0x1000, 64), None, DeviceBuffer(0x2000, 96))
    monkeypatch.setattr(gguf_runner, "linear_state_pair_commit_chunked_i32",
                        lambda *a, **k: pytest.fail("nonuniform state launched fused commit"))
    monkeypatch.setattr(gguf_runner, "set_decode_position_i64", lambda *a, **k: None)
    session._commit_verify_linear_state_row(1, position=17, stream=5)
    assert [(dst, src, size) for dst, src, size, _, _ in session.runtime.memcpy_async_calls] == [
        (0x1000, 0x9040, 64), (0x3000, 0xB080, 128),
        (0x2000, 0xA060, 96), (0x4000, 0xC080, 128),
        (0x5000, 0x8010, 16),
    ]
    assert session.position == 17


def test_gguf_packed_commit_preserves_nonuniform_per_layer_copy_fallback(monkeypatch) -> None:
    session = _fake_gguf_commit_session()
    session.runner.weights = SimpleNamespace(config=SimpleNamespace(
        layer_types=(gguf_runner.LINEAR_ATTENTION, gguf_runner.FULL_ATTENTION, gguf_runner.LINEAR_ATTENTION),
    ))
    session.scratch.layer_conv_states = (DeviceBuffer(0x1000, 64), None, DeviceBuffer(0x2000, 96))
    session._verify_linear_conv_state_rows = (DeviceBuffer(0x9000, 4 * 64), None, DeviceBuffer(0xA000, 4 * 96))
    session._verify_linear_recurrent_state_rows = (DeviceBuffer(0xB000, 4 * 128), None, DeviceBuffer(0xC000, 4 * 128))
    session._verify_linear_state_rows_capacity = 4
    monkeypatch.setattr(gguf_runner, "linear_state_pair_commit_chunked_i32",
                        lambda *a, **k: pytest.fail("nonuniform packed state launched fused commit"))
    destinations = {
        0: (DeviceBuffer(0x100000, 128), DeviceBuffer(0x200000, 256)),
        2: (DeviceBuffer(0x300000, 192), DeviceBuffer(0x400000, 256)),
    }
    session._commit_packed_decode_linear_state_rows(
        SimpleNamespace(slot_count=2, cu_seqlens=(0, 1, 4)),
        SimpleNamespace(linear_state_pair=destinations.__getitem__),
        runtime=session.runtime, stream=7,
    )
    assert [(dst, src, size) for dst, src, size, _, _ in session.runtime.memcpy_async_calls] == [
        (0x100000, 0x9000, 64), (0x200000, 0xB000, 128),
        (0x100040, 0x90C0, 64), (0x200080, 0xB180, 128),
        (0x300000, 0xA000, 96), (0x400000, 0xC000, 128),
        (0x300060, 0xA120, 96), (0x400080, 0xC180, 128),
    ]


def test_gguf_fused_commit_failure_does_not_retry_or_publish_hidden_cursor(monkeypatch) -> None:
    session = _fake_gguf_commit_session()
    for index, name in enumerate(("src_conv", "dst_conv", "src_recurrent", "dst_recurrent")):
        setattr(session, f"_verify_linear_state_{name}_table_buf", DeviceBuffer(0xD000 + index * 0x100, 16))
    session._verify_linear_state_commit_row_i32_buf = DeviceBuffer(0xD400, 4)
    session._ensure_verify_linear_state_commit_tables = lambda **kwargs: True
    monkeypatch.setattr(gguf_runner, "copy_host_array_to_device", lambda *a, **k: None)

    def fail(*args, **kwargs):
        raise RuntimeError("commit launch failed")

    monkeypatch.setattr(gguf_runner, "linear_state_pair_commit_chunked_i32", fail)
    with pytest.raises(RuntimeError, match="commit launch failed"):
        session._commit_verify_linear_state_row(1, position=17, stream=5)
    assert session.runtime.memcpy_async_calls == []
    assert session.position == 0
    assert not session._hidden_seed_fp32_populated
