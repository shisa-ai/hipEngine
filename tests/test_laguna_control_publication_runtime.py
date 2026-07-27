"""Default-off ownership contract for shared Laguna c=1 control publication."""

from __future__ import annotations

import ctypes
import inspect
from types import MethodType, SimpleNamespace

import pytest

from hipengine.core.hip import HipMemcpyKind
from hipengine.core.memory import DeviceBuffer
from hipengine.loading.laguna_gguf import (
    FULL_ATTENTION,
    SLIDING_ATTENTION,
    laguna_gguf_config_from_metadata,
)
from hipengine.runtime import laguna_gguf_runner as runner
from hipengine.runtime.laguna_gguf_runner import LagunaEagerScratch
from hipengine.runtime.laguna_kv import allocate_laguna_kv_cache
from tests._laguna_synthetic import make_laguna_info


class _FakeRuntime:
    def __init__(self, *, fail_malloc_at: int | None = None) -> None:
        self.next_ptr = 0x60000000
        self.allocations: dict[int, int] = {}
        self.freed: list[int] = []
        self.copies: list[tuple[int, int, HipMemcpyKind, bytes]] = []
        self.async_copies: list[tuple[int, int, int, HipMemcpyKind, int]] = []
        self.memsets: list[tuple[int, int, int]] = []
        self.fail_malloc_at = fail_malloc_at
        self.malloc_calls = 0

    def malloc(self, nbytes: int) -> int:
        self.malloc_calls += 1
        if self.fail_malloc_at == self.malloc_calls:
            raise MemoryError("synthetic shared-control allocation failure")
        ptr = self.next_ptr
        self.next_ptr += max(0x1000, int(nbytes) + 0x100)
        self.allocations[ptr] = int(nbytes)
        return ptr

    def free(self, ptr: int) -> None:
        self.freed.append(int(ptr))
        self.allocations.pop(int(ptr), None)

    def memcpy(self, dst: int, src: int, count: int, kind: HipMemcpyKind) -> None:
        if kind == HipMemcpyKind.HOST_TO_DEVICE:
            payload = ctypes.string_at(int(src), int(count))
        else:
            payload = b""
        self.copies.append((int(dst), int(count), kind, payload))

    def memcpy_async(
        self,
        dst: int,
        src: int,
        count: int,
        kind: HipMemcpyKind,
        stream: int,
    ) -> None:
        self.async_copies.append(
            (int(dst), int(src), int(count), kind, int(stream))
        )

    def memset(self, dst: int, value: int, nbytes: int) -> None:
        self.memsets.append((int(dst), int(value), int(nbytes)))


def _config():
    return laguna_gguf_config_from_metadata(make_laguna_info())


def _kv_config() -> SimpleNamespace:
    layer_types = tuple(
        FULL_ATTENTION if layer_id % 4 == 0 else SLIDING_ATTENTION
        for layer_id in range(48)
    )
    return SimpleNamespace(
        block_count=48,
        layer_types=layer_types,
        head_counts=tuple(
            48 if attention_type == FULL_ATTENTION else 72
            for attention_type in layer_types
        ),
        head_count_kv=8,
        key_length=128,
        value_length=128,
        sliding_window=512,
    )


def test_shared_control_capability_is_explicit_default_off_and_fail_closed() -> None:
    import hipengine.kernels.hip_gfx1100 as gfx1100
    import hipengine.kernels.hip_gfx1151 as gfx1151

    assert gfx1100.LAGUNA_SHARED_CONTROL_PUBLICATION is False
    assert not hasattr(gfx1151, "LAGUNA_SHARED_CONTROL_PUBLICATION")
    assert not runner.resolve_laguna_shared_control_publication("hip_gfx1100")
    assert runner.resolve_laguna_shared_control_publication("hip_gfx1100", True)
    assert not runner.resolve_laguna_shared_control_publication("hip_gfx1100", False)
    assert not runner.resolve_laguna_shared_control_publication("hip_gfx1151", True)


def test_shared_control_scratch_owns_one_exact_pair_and_frees_only_owners() -> None:
    runtime = _FakeRuntime()
    scratch = LagunaEagerScratch.allocate(
        _config(),
        shared_control=True,
        runtime=runtime,
    )

    assert scratch.control is not None
    assert scratch.control.nbytes == 16
    assert scratch.token_id == DeviceBuffer(scratch.control.ptr, 8)
    assert scratch.position == DeviceBuffer(scratch.control.ptr + 8, 8)
    assert scratch.buffers[0] is scratch.control
    assert scratch.token_id not in scratch.buffers
    assert scratch.position not in scratch.buffers
    assert len(runtime.allocations) == len(scratch.buffers)
    assert scratch.nbytes == sum(buffer.nbytes for buffer in scratch.buffers)

    owned = tuple(buffer.ptr for buffer in scratch.buffers)
    scratch.free(runtime=runtime)
    assert runtime.freed == list(reversed(owned))
    assert runtime.allocations == {}
    scratch.free(runtime=runtime)
    assert runtime.freed == list(reversed(owned))

    control_runtime = _FakeRuntime()
    control = LagunaEagerScratch.allocate(
        _config(),
        shared_control=False,
        runtime=control_runtime,
    )
    assert control.control is None
    assert control.token_id in control.buffers
    assert control.position in control.buffers
    assert control.token_id.ptr != control.position.ptr
    assert control.nbytes == scratch.nbytes
    control.free(runtime=control_runtime)
    assert control_runtime.allocations == {}

    failing = _FakeRuntime(fail_malloc_at=9)
    with pytest.raises(MemoryError, match="shared-control"):
        LagunaEagerScratch.allocate(
            _config(),
            shared_control=True,
            runtime=failing,
        )
    assert failing.allocations == {}


def test_borrowed_kv_position_skips_only_serial_republication_and_never_frees_owner() -> None:
    runtime = _FakeRuntime()
    owner = DeviceBuffer(runtime.malloc(16), 16)
    position = DeviceBuffer(owner.ptr + 8, 8)
    cache = allocate_laguna_kv_cache(
        _kv_config(),
        context_length=4096,
        backend="hip_gfx1151",
        runtime=runtime,
        prepublished_row_position=position,
    )

    assert cache.allocation_count == 242
    assert cache.resident_nbytes == sum(
        nbytes for ptr, nbytes in runtime.allocations.items() if ptr != owner.ptr
    )
    assert all(
        state.spans.row_positions is not None
        and state.spans.row_positions.ptr == position.ptr
        and state.append_spans.row_positions is not None
        and state.append_spans.row_positions.ptr == position.ptr
        for state in cache.layers
    )

    copies_after_allocate = len(runtime.copies)
    cache.prepare_position(0)
    assert cache.position == 0
    assert len(runtime.copies) == copies_after_allocate

    cache.prepare_rows((1, 2))
    assert runtime.copies[-1][0:3] == (
        position.ptr,
        8,
        HipMemcpyKind.HOST_TO_DEVICE,
    )
    cache.discard_rows()
    copies_before_reset = len(runtime.copies)
    cache.reset()
    assert len(runtime.copies) == copies_before_reset + 1
    assert runtime.copies[-1][0:3] == (
        position.ptr,
        8,
        HipMemcpyKind.HOST_TO_DEVICE,
    )

    cache.free()
    assert owner.ptr in runtime.allocations
    assert owner.ptr not in runtime.freed
    runtime.free(owner.ptr)
    assert runtime.allocations == {}

    standalone_runtime = _FakeRuntime()
    standalone = allocate_laguna_kv_cache(
        _kv_config(),
        context_length=4096,
        backend="hip_gfx1151",
        runtime=standalone_runtime,
    )
    assert standalone.allocation_count == 243
    copies_before_prepare = len(standalone_runtime.copies)
    standalone.prepare_position(0)
    assert len(standalone_runtime.copies) == copies_before_prepare + 1
    standalone.free()
    assert standalone_runtime.allocations == {}

    with pytest.raises(ValueError, match="row position"):
        allocate_laguna_kv_cache(
            _kv_config(),
            context_length=4096,
            backend="hip_gfx1151",
            runtime=_FakeRuntime(),
            prepublished_row_position=DeviceBuffer(0x1234, 7),
        )


def test_candidate_forward_publishes_one_exact_pair_before_kv_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _FakeRuntime()
    scratch = LagunaEagerScratch.allocate(
        _config(),
        shared_control=True,
        runtime=runtime,
    )
    runtime.copies.clear()
    events: list[tuple[str, int]] = []

    class FakeKV:
        def prepare_position(self, position: int) -> None:
            events.append(("prepare", int(position)))

    session = object.__new__(runner.LagunaGGUFResidentSession)
    session.runtime = runtime
    session.backend = "hip_gfx1100"
    session.context_length = 4096
    session.position = -1
    session.last_result = None
    session._closed = False
    session._staged_verifier_tokens = None
    session.use_shared_control_publication = True
    session.scratch = scratch
    session.kv_cache = FakeKV()
    session.kernel_plan = SimpleNamespace()
    session.libraries = SimpleNamespace(embedding_libraries={})
    session.weights = SimpleNamespace(
        config=_config(),
        root=lambda _slot: SimpleNamespace(),
    )
    session._run_layer = MethodType(
        lambda _self, layer_id, *, stream: events.append(("layer", int(layer_id))),
        session,
    )
    expected_result = SimpleNamespace(next_token_id=7)
    session._project_and_sample = MethodType(
        lambda _self, *, input_token_id, position, stream: expected_result,
        session,
    )
    monkeypatch.setattr(
        runner,
        "launch_gguf_embedding",
        lambda _weight, token_ptr, *_args, **_kwargs: events.append(
            ("embedding", int(token_ptr))
        ),
    )

    result = session.forward_token(605)

    assert result is expected_result
    assert session.position == 0
    assert len(runtime.copies) == 1
    dst, count, kind, payload = runtime.copies[0]
    assert (dst, count, kind) == (
        scratch.control.ptr,
        16,
        HipMemcpyKind.HOST_TO_DEVICE,
    )
    pair = (ctypes.c_int64 * 2).from_buffer_copy(payload)
    assert tuple(pair) == (605, 0)
    assert events[0] == ("prepare", 0)
    assert events[1] == ("embedding", scratch.token_id.ptr)
    scratch.free(runtime=runtime)


def test_reset_publishes_unfused_position_only_for_separate_control_owner() -> None:
    class FakeKV:
        def __init__(self) -> None:
            self.calls = 0

        def reset(self) -> None:
            self.calls += 1

    for shared, expected_copies in ((False, 1), (True, 0)):
        runtime = _FakeRuntime()
        session = object.__new__(runner.LagunaGGUFResidentSession)
        session._closed = False
        session._staged_verifier_tokens = None
        session.runtime = runtime
        session.use_shared_control_publication = shared
        session.scratch = SimpleNamespace(position=DeviceBuffer(0x78000008, 8))
        session.kv_cache = FakeKV()
        session.position = 17
        session.last_result = SimpleNamespace()

        session.reset_state()

        assert session.kv_cache.calls == 1
        assert session.position == -1
        assert session.last_result is None
        assert len(runtime.copies) == expected_copies
        if not shared:
            dst, count, kind, payload = runtime.copies[0]
            assert (dst, count, kind) == (
                session.scratch.position.ptr,
                8,
                HipMemcpyKind.HOST_TO_DEVICE,
            )
            assert ctypes.c_int64.from_buffer_copy(payload).value == -1


def test_candidate_session_wires_and_closes_borrowed_kv_before_scratch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    captured: dict[str, object] = {}
    config = _config()

    class Resource:
        def __init__(self, name: str) -> None:
            self.name = name
            self.split_gate_fusion = False
            self.swa_split_wave_local = False
            self.position = DeviceBuffer(0x77000008, 8)

        def free(self, **kwargs) -> None:
            del kwargs
            events.append(self.name)

    shared_weights = SimpleNamespace(config=config, backend="hip_gfx1100")
    monkeypatch.setattr(
        runner.LagunaGGUFResidentSession,
        "_validate_resident_weights",
        lambda self: None,
    )
    monkeypatch.setattr(
        runner,
        "load_laguna_eager_libraries",
        lambda **kwargs: Resource("libraries"),
    )
    ropes = iter((Resource("full_rope"), Resource("swa_rope")))
    monkeypatch.setattr(
        runner,
        "materialize_laguna_rope_tables",
        lambda *args, **kwargs: next(ropes),
    )

    def allocate_scratch(*args, **kwargs):
        captured["shared_control"] = kwargs.get("shared_control")
        return Resource("scratch")

    def allocate_kv(*args, **kwargs):
        captured["prepublished_row_position"] = kwargs.get(
            "prepublished_row_position"
        )
        return Resource("kv")

    monkeypatch.setattr(runner.LagunaEagerScratch, "allocate", allocate_scratch)
    monkeypatch.setattr(runner, "allocate_laguna_kv_cache", allocate_kv)
    monkeypatch.setattr(
        runner,
        "resolve_laguna_moe_plan",
        lambda *args, **kwargs: Resource("moe_plan"),
    )
    monkeypatch.setattr(
        runner,
        "allocate_laguna_moe_scratch",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            MemoryError("synthetic candidate session failure")
        ),
    )

    with pytest.raises(MemoryError, match="candidate session"):
        runner.LagunaGGUFResidentSession(
            resident_weights=shared_weights,
            backend="hip_gfx1100",
            runtime=SimpleNamespace(),
            use_shared_control_publication=True,
        )

    assert captured["shared_control"] is True
    assert captured["prepublished_row_position"] == DeviceBuffer(0x77000008, 8)
    assert events == ["kv", "scratch", "swa_rope", "full_rope"]


def test_shared_control_session_and_benchmark_opt_in_are_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import laguna_target_ar_bench as benchmark

    option = "use_shared_control_publication"
    assert option in inspect.signature(runner.LagunaGGUFResidentSession).parameters
    monkeypatch.setattr(benchmark.sys, "argv", ["laguna_target_ar_bench.py"])
    assert benchmark._parse_args().enable_shared_control_publication is False
    monkeypatch.setattr(
        benchmark.sys,
        "argv",
        ["laguna_target_ar_bench.py", "--enable-shared-control-publication"],
    )
    assert benchmark._parse_args().enable_shared_control_publication is True
