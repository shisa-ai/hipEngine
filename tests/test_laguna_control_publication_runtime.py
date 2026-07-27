"""Runtime-rejection contract for shared Laguna c=1 control publication."""

from __future__ import annotations

import ctypes
import inspect
from types import SimpleNamespace

import pytest

from hipengine.core.hip import HipMemcpyKind
from hipengine.core.memory import DeviceBuffer
from hipengine.runtime import laguna_gguf_runner as runner
from hipengine.runtime.laguna_gguf_runner import (
    LagunaEagerScratch,
    LagunaGGUFResidentSession,
)
from hipengine.runtime.laguna_kv import LagunaKVCache, allocate_laguna_kv_cache


def test_shared_control_runtime_owner_is_removed() -> None:
    import hipengine.kernels.hip_gfx1100 as gfx1100
    import hipengine.kernels.hip_gfx1151 as gfx1151

    assert not hasattr(gfx1100, "LAGUNA_SHARED_CONTROL_PUBLICATION")
    assert not hasattr(gfx1151, "LAGUNA_SHARED_CONTROL_PUBLICATION")
    assert not hasattr(runner, "resolve_laguna_shared_control_publication")
    assert "shared_control" not in inspect.signature(
        LagunaEagerScratch.allocate
    ).parameters
    assert "control" not in inspect.signature(LagunaEagerScratch).parameters
    assert "use_shared_control_publication" not in inspect.signature(
        LagunaGGUFResidentSession
    ).parameters
    assert "prepublished_row_position" not in inspect.signature(
        allocate_laguna_kv_cache
    ).parameters
    assert "row_position_prepublished" not in inspect.signature(
        LagunaKVCache
    ).parameters

    forward_source = inspect.getsource(LagunaGGUFResidentSession.forward_token)
    close_source = inspect.getsource(LagunaGGUFResidentSession.close)
    prepare_source = inspect.getsource(LagunaKVCache.prepare_position)
    assert "_copy_i64_pair" not in forward_source
    assert "use_shared_control_publication" not in forward_source
    assert "use_shared_control_publication" not in close_source
    assert "_copy_i64(self._row_position, parsed, self.runtime)" in prepare_source


def test_shared_control_benchmark_opt_in_is_removed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import laguna_target_ar_bench as benchmark

    monkeypatch.setattr(
        benchmark.sys,
        "argv",
        ["laguna_target_ar_bench.py", "--enable-shared-control-publication"],
    )
    with pytest.raises(SystemExit):
        benchmark._parse_args()


def test_reset_keeps_unfused_scratch_position_visible() -> None:
    class FakeRuntime:
        def __init__(self) -> None:
            self.copies: list[tuple[int, int, HipMemcpyKind, bytes]] = []

        def memcpy(
            self,
            dst: int,
            src: int,
            count: int,
            kind: HipMemcpyKind,
        ) -> None:
            self.copies.append(
                (
                    int(dst),
                    int(count),
                    kind,
                    ctypes.string_at(int(src), int(count)),
                )
            )

    class FakeKV:
        def __init__(self) -> None:
            self.calls = 0

        def reset(self) -> None:
            self.calls += 1

    runtime = FakeRuntime()
    session = object.__new__(LagunaGGUFResidentSession)
    session._closed = False
    session._staged_verifier_tokens = None
    session.runtime = runtime
    session.scratch = SimpleNamespace(position=DeviceBuffer(0x78000008, 8))
    session.kv_cache = FakeKV()
    session.position = 17
    session.last_result = SimpleNamespace()

    session.reset_state()

    assert session.kv_cache.calls == 1
    assert session.position == -1
    assert session.last_result is None
    assert len(runtime.copies) == 1
    dst, count, kind, payload = runtime.copies[0]
    assert (dst, count, kind) == (
        session.scratch.position.ptr,
        8,
        HipMemcpyKind.HOST_TO_DEVICE,
    )
    assert ctypes.c_int64.from_buffer_copy(payload).value == -1
