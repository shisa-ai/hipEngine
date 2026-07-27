"""Default-off runtime contract for Laguna argmax pair readback."""

from __future__ import annotations

import ctypes
import inspect
import struct
from types import SimpleNamespace

import pytest

from hipengine.core.hip import HipMemcpyKind
from hipengine.core.memory import DeviceBuffer
from hipengine.loading.laguna_gguf import laguna_gguf_config_from_metadata
from hipengine.runtime import laguna_gguf_runner as runner
from hipengine.runtime.laguna_gguf_runner import (
    LagunaEagerScratch,
    LagunaGGUFResidentSession,
)
from tests._laguna_synthetic import make_laguna_info


class _FakeRuntime:
    def __init__(
        self,
        *,
        fail_malloc_at: int | None = None,
        device_payload: bytes | None = None,
    ) -> None:
        self.next_ptr = 0x7A000000
        self.allocations: dict[int, int] = {}
        self.freed: list[int] = []
        self.copies: list[tuple[int, int, HipMemcpyKind]] = []
        self.fail_malloc_at = fail_malloc_at
        self.malloc_calls = 0
        self.device_payload = device_payload
        self.syncs = 0

    def malloc(self, nbytes: int) -> int:
        self.malloc_calls += 1
        if self.fail_malloc_at == self.malloc_calls:
            raise MemoryError("synthetic argmax-pair allocation failure")
        ptr = self.next_ptr
        self.next_ptr += max(0x1000, int(nbytes) + 0x100)
        self.allocations[ptr] = int(nbytes)
        return ptr

    def free(self, ptr: int) -> None:
        self.freed.append(int(ptr))
        self.allocations.pop(int(ptr), None)

    def memcpy(
        self,
        dst: int,
        src: int,
        count: int,
        kind: HipMemcpyKind,
    ) -> None:
        self.copies.append((int(src), int(count), kind))
        if kind == HipMemcpyKind.DEVICE_TO_HOST:
            assert self.device_payload is not None
            assert int(count) == len(self.device_payload)
            ctypes.memmove(int(dst), self.device_payload, int(count))

    def device_synchronize(self) -> None:
        self.syncs += 1

    def stream_synchronize(self, stream: int) -> None:
        assert int(stream) != 0
        self.syncs += 1


def _config():
    return laguna_gguf_config_from_metadata(make_laguna_info())


def test_argmax_pair_capability_is_explicit_default_off_and_fail_closed() -> None:
    import hipengine.kernels.hip_gfx1100 as gfx1100
    import hipengine.kernels.hip_gfx1151 as gfx1151

    assert gfx1100.LAGUNA_ARGMAX_PAIR_READBACK is False
    assert not hasattr(gfx1151, "LAGUNA_ARGMAX_PAIR_READBACK")
    assert not runner.resolve_laguna_argmax_pair_readback("hip_gfx1100")
    assert runner.resolve_laguna_argmax_pair_readback("hip_gfx1100", True)
    assert not runner.resolve_laguna_argmax_pair_readback("hip_gfx1100", False)
    assert not runner.resolve_laguna_argmax_pair_readback("hip_gfx1151", True)


def test_argmax_pair_scratch_owns_exact_12_bytes_and_frees_only_owner() -> None:
    runtime = _FakeRuntime()
    scratch = LagunaEagerScratch.allocate(
        _config(),
        argmax_pair_readback=True,
        runtime=runtime,
    )

    assert scratch.argmax_result is not None
    assert scratch.argmax_result.nbytes == 12
    assert scratch.argmax_id == DeviceBuffer(scratch.argmax_result.ptr, 8)
    assert scratch.argmax_value == DeviceBuffer(scratch.argmax_result.ptr + 8, 4)
    assert scratch.argmax_result in scratch.buffers
    assert scratch.argmax_id not in scratch.buffers
    assert scratch.argmax_value not in scratch.buffers
    assert len(runtime.allocations) == len(scratch.buffers) == 23
    assert scratch.nbytes == sum(buffer.nbytes for buffer in scratch.buffers)

    owned = tuple(buffer.ptr for buffer in scratch.buffers)
    scratch.free(runtime=runtime)
    assert runtime.freed == list(reversed(owned))
    assert runtime.allocations == {}
    scratch.free(runtime=runtime)
    assert runtime.freed == list(reversed(owned))

    fallback_runtime = _FakeRuntime()
    fallback = LagunaEagerScratch.allocate(
        _config(),
        argmax_pair_readback=False,
        runtime=fallback_runtime,
    )
    assert fallback.argmax_result is None
    assert fallback.argmax_id in fallback.buffers
    assert fallback.argmax_value in fallback.buffers
    assert fallback.argmax_id.ptr != fallback.argmax_value.ptr
    assert fallback.nbytes == scratch.nbytes
    fallback.free(runtime=fallback_runtime)
    assert fallback_runtime.allocations == {}

    failing = _FakeRuntime(fail_malloc_at=17)
    with pytest.raises(MemoryError, match="argmax-pair"):
        LagunaEagerScratch.allocate(
            _config(),
            argmax_pair_readback=True,
            runtime=failing,
        )
    assert failing.allocations == {}


def test_argmax_pair_read_is_one_exact_12_byte_copy() -> None:
    token_id = 69452
    value_bits = 0x80000000
    payload = struct.pack("<qI", token_id, value_bits)
    runtime = _FakeRuntime(device_payload=payload)
    owner = DeviceBuffer(0x7B000000, 12)

    actual_id, actual_value = runner._read_argmax_pair(owner, runtime)

    assert actual_id == token_id
    assert struct.pack("<f", actual_value) == struct.pack("<I", value_bits)
    assert runtime.copies == [
        (owner.ptr, 12, HipMemcpyKind.DEVICE_TO_HOST),
    ]
    with pytest.raises(ValueError, match="12 bytes"):
        runner._read_argmax_pair(DeviceBuffer(owner.ptr, 16), runtime)


def _sampling_session(*, pair: bool) -> LagunaGGUFResidentSession:
    class _Root:
        def allocation(self, slot: str):
            assert slot == "raw"
            return SimpleNamespace(tensor=SimpleNamespace(ptr=0x7C000000))

    class _Plan:
        def rmsnorm(self, *args, **kwargs) -> None:
            del args, kwargs

        def argmax(self, *args, **kwargs) -> None:
            del args, kwargs

    config = SimpleNamespace(hidden_size=4, vocab_size=8, rms_norm_eps=1e-6)
    session = object.__new__(LagunaGGUFResidentSession)
    session.runtime = _FakeRuntime()
    session.backend = "hip_gfx1100"
    session.weights = SimpleNamespace(config=config, root=lambda _slot: _Root())
    session.scratch = SimpleNamespace(
        hidden=DeviceBuffer(0x7D000000, 8),
        final_norm=DeviceBuffer(0x7D001000, 8),
        logits=DeviceBuffer(0x7D002000, 32),
        argmax_block_values=DeviceBuffer(0x7D003000, 4),
        argmax_block_indices=DeviceBuffer(0x7D004000, 8),
        argmax_id=DeviceBuffer(0x7D005000, 8),
        argmax_value=DeviceBuffer(0x7D006000, 4),
        argmax_result=DeviceBuffer(0x7D007000, 12) if pair else None,
    )
    session.rows_scratch = SimpleNamespace(
        hidden=DeviceBuffer(0x7E000000, 8),
        final_norm=DeviceBuffer(0x7E001000, 8),
        logits=DeviceBuffer(0x7E002000, 32),
    )
    session.kernel_plan = _Plan()
    session.libraries = SimpleNamespace(
        argmax=SimpleNamespace(),
        gguf_ops=SimpleNamespace(),
        linear={},
    )
    session._q4_lm_head_variant = "fixed"
    session.use_argmax_pair_readback = pair
    session.last_result = None
    return session


def test_both_scalar_sampling_sites_use_pair_or_separate_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pair_calls: list[int] = []
    scalar_calls: list[str] = []

    monkeypatch.setattr(runner, "launch_gguf_linear", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        runner,
        "_read_argmax_pair",
        lambda buffer, runtime: (pair_calls.append(buffer.ptr) or (7, 1.25)),
    )
    monkeypatch.setattr(
        runner,
        "_read_i64",
        lambda buffer, runtime: (scalar_calls.append("id") or 7),
    )
    monkeypatch.setattr(
        runner,
        "_read_f32",
        lambda buffer, runtime: (scalar_calls.append("value") or 1.25),
    )

    candidate = _sampling_session(pair=True)
    first = candidate._project_and_sample(input_token_id=605, position=0, stream=0)
    second = candidate._project_rows_last(
        input_token_id=2825,
        position=1,
        row_index=0,
        stream=0,
    )
    assert (first.next_token_id, first.next_token_logit) == (7, 1.25)
    assert (second.next_token_id, second.next_token_logit) == (7, 1.25)
    assert pair_calls == [
        candidate.scratch.argmax_result.ptr,
        candidate.scratch.argmax_result.ptr,
    ]
    assert scalar_calls == []

    pair_calls.clear()
    fallback = _sampling_session(pair=False)
    fallback._project_and_sample(input_token_id=605, position=0, stream=0)
    fallback._project_rows_last(
        input_token_id=2825,
        position=1,
        row_index=0,
        stream=0,
    )
    assert pair_calls == []
    assert scalar_calls == ["id", "value", "id", "value"]


def test_argmax_pair_session_and_benchmark_opt_in_are_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import laguna_target_ar_bench as benchmark

    option = "use_argmax_pair_readback"
    assert option in inspect.signature(LagunaGGUFResidentSession).parameters
    constructor_source = inspect.getsource(LagunaGGUFResidentSession.__init__)
    assert "argmax_pair_readback=self.use_argmax_pair_readback" in constructor_source
    monkeypatch.setattr(benchmark.sys, "argv", ["laguna_target_ar_bench.py"])
    assert benchmark._parse_args().enable_argmax_pair_readback is False
    monkeypatch.setattr(
        benchmark.sys,
        "argv",
        ["laguna_target_ar_bench.py", "--enable-argmax-pair-readback"],
    )
    assert benchmark._parse_args().enable_argmax_pair_readback is True
    assert option in inspect.getsource(benchmark._session)
    assert option in inspect.getsource(benchmark.run)
