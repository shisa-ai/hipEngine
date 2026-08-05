from __future__ import annotations

import ctypes
from types import SimpleNamespace

import numpy as np

from hipengine.core.memory import DeviceBuffer
from hipengine.kernels.hip_gfx1100.speculative.dflash_commit import (
    linear_state_pair_commit_chunked_i32,
    register_dflash_commit_kernels,
)
from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
from hipengine.kernels.registry import KernelKey, register, resolve, unregister
from hipengine.runtime import qwen35_gguf_mtp as mtp_module
from hipengine.runtime.qwen35_gguf_mtp import _StateJournal


class _FakeRuntime:
    def __init__(self) -> None:
        self.memcpy_async_calls: list[tuple[int, int, int, object, int]] = []

    def memcpy_async(self, dst: int, src: int, nbytes: int, kind: object, stream: int) -> None:
        self.memcpy_async_calls.append((int(dst), int(src), int(nbytes), kind, int(stream)))


def _fake_target(*, backend: str):
    runtime = _FakeRuntime()
    return SimpleNamespace(
        backend=backend,
        runtime=runtime,
        runner=SimpleNamespace(hidden_size=4),
        _target_scratch_owner=SimpleNamespace(
            slot_count=1,
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
        ),
        last_target_hidden=DeviceBuffer(0x5000, 8),
        _hidden_a=DeviceBuffer(0x6000, 8),
        _last_target_hidden_ptr=0,
        _dflash_commit_library=object(),
        compiler_version="hipcc:test",
        require_cached_build=True,
    )


def test_linear_state_snapshot_copy_registers_only_for_gfx1100() -> None:
    register_dflash_commit_kernels(replace=True)
    register_gfx1151_kernels(replace=True)

    assert (
        resolve(
            backend="hip_gfx1100",
            layer="linear_state_pair_copy",
            quant="f32",
            variant="chunked_i32",
        )
        is linear_state_pair_commit_chunked_i32
    )
    assert (
        resolve(
            backend="hip_gfx1151",
            layer="linear_state_pair_copy",
            quant="f32",
            variant="chunked_i32",
            missing="none",
        )
        is None
    )


def test_state_journal_initial_snapshot_and_rollback_use_one_pointer_table_launch(monkeypatch) -> None:
    backend = "test_state_journal_copy"
    key = KernelKey(backend, "linear_state_pair_copy", "f32", "chunked_i32")
    allocated: list[DeviceBuffer] = []
    freed: list[int] = []
    h2d: list[tuple[int, np.ndarray]] = []
    launches: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake_malloc(nbytes: int, *, runtime) -> DeviceBuffer:
        _ = runtime
        buffer = DeviceBuffer(0xA000 + len(allocated) * 0x1000, int(nbytes))
        allocated.append(buffer)
        return buffer

    def fake_free(buffer: DeviceBuffer, *, runtime) -> None:
        _ = runtime
        freed.append(int(buffer.ptr))

    def fake_copy_host_to_device(buffer, host_ptr: int, nbytes: int, *, runtime) -> None:
        _ = runtime
        if int(nbytes) == np.dtype(np.int32).itemsize:
            c_array = (ctypes.c_int32 * (int(nbytes) // np.dtype(np.int32).itemsize)).from_address(
                int(host_ptr)
            )
        else:
            c_array = (ctypes.c_uint64 * (int(nbytes) // np.dtype(np.uint64).itemsize)).from_address(
                int(host_ptr)
            )
        h2d.append((int(buffer.ptr), np.ctypeslib.as_array(c_array).copy()))

    def fake_copy(*args, **kwargs) -> None:
        launches.append((args, kwargs))

    monkeypatch.setattr(mtp_module, "malloc", fake_malloc)
    monkeypatch.setattr(mtp_module, "free", fake_free)
    monkeypatch.setattr(mtp_module, "copy_host_to_device", fake_copy_host_to_device)
    register(key, fake_copy, replace=True)
    try:
        target = _fake_target(backend=backend)
        journal = _StateJournal.allocate(target, max_rows=4)

        assert len(allocated) == 9  # Four journals, two hidden buffers, two tables, row zero.
        assert journal.initial_state_copy is not None
        assert [(ptr, values.tolist()) for ptr, values in h2d] == [
            (0x10000, [0x1000, 0x2000, 0x3000, 0x4000]),
            (0x11000, [0xA000, 0xB000, 0xC000, 0xD000]),
            (0x12000, [0]),
        ]

        journal.capture_initial(stream=7)

        assert target.runtime.memcpy_async_calls == [
            (0xE000, 0x5000, 8, mtp_module.HipMemcpyKind.DEVICE_TO_DEVICE, 7),
        ]
        assert len(launches) == 1
        capture_args, capture_kwargs = launches[0]
        assert capture_args == (
            0x10000,
            0x11000,
            64,
            0x10000 + 2 * np.dtype(np.uint64).itemsize,
            0x11000 + 2 * np.dtype(np.uint64).itemsize,
            128,
            0x12000,
            2,
        )
        assert capture_kwargs == {
            "stream": 7,
            "library": target._dflash_commit_library,
            "runtime": target.runtime,
        }

        journal.restore_initial(stream=9)

        assert len(launches) == 2
        restore_args, restore_kwargs = launches[1]
        assert restore_args == (
            0x11000,
            0x10000,
            64,
            0x11000 + 2 * np.dtype(np.uint64).itemsize,
            0x10000 + 2 * np.dtype(np.uint64).itemsize,
            128,
            0x12000,
            2,
        )
        assert restore_kwargs == {
            "stream": 9,
            "library": target._dflash_commit_library,
            "runtime": target.runtime,
        }
        assert target.runtime.memcpy_async_calls[-1] == (
            0x6000,
            0xE000,
            8,
            mtp_module.HipMemcpyKind.DEVICE_TO_DEVICE,
            9,
        )
        assert target._last_target_hidden_ptr == 0x6000

        # Per-row journal capture stays on the original exact memcpy path; only
        # the once-per-cycle rollback boundary is collapsed.
        target.runtime.memcpy_async_calls.clear()
        journal.capture_row(2, stream=11)
        assert len(launches) == 2
        assert target.runtime.memcpy_async_calls == [
            (0xF000 + 2 * 8, 0x5000, 8, mtp_module.HipMemcpyKind.DEVICE_TO_DEVICE, 11),
            (0xA000 + 3 * 64, 0x1000, 64, mtp_module.HipMemcpyKind.DEVICE_TO_DEVICE, 11),
            (0xB000 + 3 * 64, 0x2000, 64, mtp_module.HipMemcpyKind.DEVICE_TO_DEVICE, 11),
            (0xC000 + 3 * 128, 0x3000, 128, mtp_module.HipMemcpyKind.DEVICE_TO_DEVICE, 11),
            (0xD000 + 3 * 128, 0x4000, 128, mtp_module.HipMemcpyKind.DEVICE_TO_DEVICE, 11),
        ]

        journal.close()
        assert freed == [
            0x12000,
            0x11000,
            0x10000,
            0xF000,
            0xE000,
            0xD000,
            0xC000,
            0xB000,
            0xA000,
        ]
    finally:
        unregister(key)


def test_state_journal_producer_capture_preserves_post_commit_rollback(monkeypatch) -> None:
    backend = "test_state_journal_producer_capture"
    key = KernelKey(backend, "linear_state_pair_copy", "f32", "chunked_i32")
    allocated: list[DeviceBuffer] = []
    freed: list[int] = []
    launches: list[tuple[tuple[object, ...], dict[str, object]]] = []
    lifecycle: list[str] = []

    def fake_malloc(nbytes: int, *, runtime) -> DeviceBuffer:
        _ = runtime
        buffer = DeviceBuffer(0xA000 + len(allocated) * 0x1000, int(nbytes))
        allocated.append(buffer)
        return buffer

    def fake_free(buffer: DeviceBuffer, *, runtime) -> None:
        _ = runtime
        freed.append(int(buffer.ptr))

    def fake_copy_host_to_device(_buffer, _host_ptr: int, _nbytes: int, *, runtime) -> None:
        _ = runtime

    def fake_copy(*args, **kwargs) -> None:
        launches.append((args, kwargs))

    target = _fake_target(backend=backend)
    conv_states = target._target_scratch_owner.layer_conv_states
    recurrent_states = target._target_scratch_owner.layer_recurrent_states
    conv_capture = (
        (0, conv_states[0], DeviceBuffer(0x21000, 64)),
        (2, conv_states[2], DeviceBuffer(0x22000, 64)),
    )
    recurrent_capture = (
        (0, recurrent_states[0], DeviceBuffer(0x23000, 128)),
        (2, recurrent_states[2], DeviceBuffer(0x24000, 128)),
    )
    target._acquire_verify_initial_state_capture = lambda: (
        lifecycle.append("acquire") or (conv_capture, recurrent_capture)
    )
    target._release_verify_initial_state_capture = lambda: lifecycle.append("release")

    monkeypatch.setattr(mtp_module, "malloc", fake_malloc)
    monkeypatch.setattr(mtp_module, "free", fake_free)
    monkeypatch.setattr(mtp_module, "copy_host_to_device", fake_copy_host_to_device)
    register(key, fake_copy, replace=True)
    try:
        journal = _StateJournal.allocate(
            target,
            max_rows=4,
            producer_capture_initial_state=True,
        )
        assert journal.producer_capture_initial_state
        assert lifecycle == ["acquire"]
        assert [buffer.nbytes for buffer in allocated] == [8, 32, 32, 32, 4]
        assert [(state.ptr, snapshot.ptr) for state, snapshot in journal.state_rows] == [
            (0x1000, 0x21000),
            (0x2000, 0x22000),
            (0x3000, 0x23000),
            (0x4000, 0x24000),
        ]

        journal.capture_initial(stream=7)
        assert target.runtime.memcpy_async_calls == [
            (0xA000, 0x5000, 8, mtp_module.HipMemcpyKind.DEVICE_TO_DEVICE, 7),
        ]
        assert launches == []

        # A prepare failure before every producer retires must not restore a
        # partial snapshot over the still-immutable resident state.
        journal.restore_initial(stream=8)
        assert launches == []
        assert target.runtime.memcpy_async_calls[-1] == (
            0x6000,
            0xA000,
            8,
            mtp_module.HipMemcpyKind.DEVICE_TO_DEVICE,
            8,
        )

        target.runtime.memcpy_async_calls.clear()
        journal.capture_initial(stream=9)
        journal.mark_initial_state_captured()
        journal.restore_initial(stream=10)
        assert len(launches) == 1
        restore_args, restore_kwargs = launches[0]
        assert restore_args == (
            0xD000,
            0xC000,
            64,
            0xD000 + 2 * np.dtype(np.uint64).itemsize,
            0xC000 + 2 * np.dtype(np.uint64).itemsize,
            128,
            0xE000,
            2,
        )
        assert restore_kwargs["stream"] == 10
        assert target.runtime.memcpy_async_calls[-1] == (
            0x6000,
            0xA000,
            8,
            mtp_module.HipMemcpyKind.DEVICE_TO_DEVICE,
            10,
        )

        journal.close()
        assert lifecycle == ["acquire", "release"]
        assert freed == [0xE000, 0xD000, 0xC000, 0xB000, 0xA000]
        assert not ({0x21000, 0x22000, 0x23000, 0x24000} & set(freed))
    finally:
        unregister(key)


def test_state_journal_snapshot_copy_registry_miss_preserves_memcpy_fallback(monkeypatch) -> None:
    allocated: list[DeviceBuffer] = []

    def fake_malloc(nbytes: int, *, runtime) -> DeviceBuffer:
        _ = runtime
        buffer = DeviceBuffer(0xA000 + len(allocated) * 0x1000, int(nbytes))
        allocated.append(buffer)
        return buffer

    monkeypatch.setattr(mtp_module, "malloc", fake_malloc)
    target = _fake_target(backend="missing_state_journal_copy")
    journal = _StateJournal.allocate(target, max_rows=4)

    journal.capture_initial(stream=7)
    journal.restore_initial(stream=9)

    assert len(allocated) == 6
    assert journal.initial_state_copy is None
    assert target.runtime.memcpy_async_calls == [
        (0xE000, 0x5000, 8, mtp_module.HipMemcpyKind.DEVICE_TO_DEVICE, 7),
        (0xA000, 0x1000, 64, mtp_module.HipMemcpyKind.DEVICE_TO_DEVICE, 7),
        (0xB000, 0x2000, 64, mtp_module.HipMemcpyKind.DEVICE_TO_DEVICE, 7),
        (0xC000, 0x3000, 128, mtp_module.HipMemcpyKind.DEVICE_TO_DEVICE, 7),
        (0xD000, 0x4000, 128, mtp_module.HipMemcpyKind.DEVICE_TO_DEVICE, 7),
        (0x1000, 0xA000, 64, mtp_module.HipMemcpyKind.DEVICE_TO_DEVICE, 9),
        (0x2000, 0xB000, 64, mtp_module.HipMemcpyKind.DEVICE_TO_DEVICE, 9),
        (0x3000, 0xC000, 128, mtp_module.HipMemcpyKind.DEVICE_TO_DEVICE, 9),
        (0x4000, 0xD000, 128, mtp_module.HipMemcpyKind.DEVICE_TO_DEVICE, 9),
        (0x6000, 0xE000, 8, mtp_module.HipMemcpyKind.DEVICE_TO_DEVICE, 9),
    ]
