"""CPU tests for the compiled staged-exchange transport and the driver knob.

No ROCm is required: a fake driver library (the same ctypes surface the
compiled ``.so`` exposes) stands in for the hipcc-built host driver, so the
tests pin the wrapper's structure - validation at construction, per-call
pointer passing, the payload pointer every rank consumes, poison semantics,
exactly-once teardown, and the shard group's driver selection - without
touching hardware or a compiler.
"""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

from hipengine.distributed.shard_group import MlpShardGroup, ShardGroupError
from hipengine.distributed.staged import StagedExchangeTransport
from hipengine.distributed.staged_compiled import CompiledStagedExchangeTransport
from hipengine.distributed.transport import TransportError, TransportStateError

_CREATE_HANDLE = 0xBEEF
_PAYLOAD_PTR = 0x7000_0000


class FakeDriver:
    """The compiled driver's C ABI, as plain functions with shared state.

    The wrapper's ``_bind`` pins ``argtypes`` onto the library's symbols, so
    the four entry points are plain function attributes (Python functions
    accept attribute assignment, like ``ctypes._FuncPtr`` does). ``reduce``
    records the per-rank pointers it was handed and publishes a fixed payload
    address, so the wrapper's return value and pointer passing are assertable
    without HIP.
    """

    def __init__(self, *, fail_reduce_code: int = 0, fail_create: bool = False) -> None:
        self.create_calls: list[tuple[tuple[int, ...], int, int, int, int]] = []
        self.reduce_calls: list[list[int]] = []
        self.destroyed: list[int] = []
        self.fail_reduce_code = int(fail_reduce_code)
        self.fail_create = fail_create
        self._message = b""

        driver = self

        def tp2_staged_create(devices, world, streams, hidden, dtype, slots, err):
            code_ptr = ctypes.cast(err, ctypes.POINTER(ctypes.c_int32))
            if driver.fail_create:
                code_ptr[0] = -1
                driver._message = b"simulated create failure"
                return 0
            code_ptr[0] = 0
            driver.create_calls.append(
                (tuple(devices), int(world), int(hidden), int(dtype), int(slots))
            )
            return _CREATE_HANDLE

        def tp2_staged_reduce(handle, partials, out_payload):
            array = ctypes.cast(partials, ctypes.POINTER(ctypes.c_void_p))
            driver.reduce_calls.append([int(array[i]) for i in range(2)])
            out = ctypes.cast(out_payload, ctypes.POINTER(ctypes.c_uint64))
            if driver.fail_reduce_code:
                driver._message = b"simulated reduce failure"
                return driver.fail_reduce_code
            out[0] = _PAYLOAD_PTR
            return 0

        def tp2_staged_last_error(handle):
            return driver._message

        def tp2_staged_destroy(handle) -> None:
            value = handle.value if isinstance(handle, ctypes.c_void_p) else handle
            driver.destroyed.append(int(value))

        self.tp2_staged_create = tp2_staged_create
        self.tp2_staged_reduce = tp2_staged_reduce
        self.tp2_staged_last_error = tp2_staged_last_error
        self.tp2_staged_destroy = tp2_staged_destroy


def _transport(library: FakeDriver, **kwargs: object) -> CompiledStagedExchangeTransport:
    options: dict[str, object] = {
        "devices": (0, 1),
        "streams": {0: 100, 1: 101},
        "hidden": 8,
        "staging_dtype": "f32",
    }
    options.update(kwargs)
    return CompiledStagedExchangeTransport(
        object(),
        library=library,
        **options,  # type: ignore[arg-type]
    )


# -- construction -------------------------------------------------------------


def test_create_receives_the_validated_configuration() -> None:
    library = FakeDriver()
    transport = _transport(
        library, devices=(3, 5), streams={3: 9, 5: 7}, hidden=12, staging_dtype="bf16"
    )
    assert library.create_calls == [((3, 5), 2, 12, 1, 2)], (
        "the driver gets the device list, world, hidden, the bf16 code, and two slot sets"
    )
    assert transport.staging_nbytes == 12 * 2
    transport.close()


def test_invalid_construction_never_touches_the_driver() -> None:
    library = FakeDriver()
    with pytest.raises(TransportStateError, match="distinct devices"):
        _transport(library, devices=(0, 0), streams={0: 1})
    with pytest.raises(TransportStateError, match="no stream given"):
        _transport(library, streams={0: 1})
    with pytest.raises(TransportStateError, match="staging dtype"):
        _transport(library, staging_dtype="fp64")
    with pytest.raises(TransportStateError, match="at least one rank"):
        _transport(library, devices=())
    assert library.create_calls == [], "validation happens before any driver call"
    assert library.destroyed == []


def test_a_create_failure_refuses_with_the_driver_message() -> None:
    library = FakeDriver(fail_create=True)
    with pytest.raises(TransportStateError, match="simulated create failure"):
        _transport(library)
    assert library.destroyed == []


# -- the reduction ------------------------------------------------------------


def test_reduce_publishes_one_payload_pointer_for_every_rank() -> None:
    library = FakeDriver()
    transport = _transport(library)
    reduced = transport.reduce({0: 0x1000, 1: 0x2000})
    assert reduced == {0: _PAYLOAD_PTR, 1: _PAYLOAD_PTR}, (
        "both ranks consume the same mapped payload row"
    )
    assert library.reduce_calls == [[0x1000, 0x2000]], (
        "the per-rank partial pointers reach the driver in rank order"
    )
    assert transport.reductions == 1
    transport.close()


def test_reduce_passes_pointers_in_rank_order_across_calls() -> None:
    library = FakeDriver()
    transport = _transport(library)
    transport.reduce({0: 0x1000, 1: 0x2000})
    transport.reduce({0: 0x1000, 1: 0x2000})
    assert library.reduce_calls == [[0x1000, 0x2000], [0x1000, 0x2000]]
    assert transport.reductions == 2
    transport.close()


def test_a_missing_partial_poisons_and_refuses() -> None:
    library = FakeDriver()
    transport = _transport(library)
    with pytest.raises(TransportStateError, match="no partial given"):
        transport.reduce({0: 0x1000})
    assert transport.poisoned is True
    with pytest.raises(TransportError, match="poisoned"):
        transport.reduce({0: 0x1000, 1: 0x2000})
    assert library.reduce_calls == [], "the poisoned transport never calls the driver"
    transport.close()


def test_a_driver_failure_poisons_with_the_driver_message() -> None:
    library = FakeDriver(fail_reduce_code=-2)
    transport = _transport(library)
    with pytest.raises(TransportError, match="simulated reduce failure"):
        transport.reduce({0: 0x1000, 1: 0x2000})
    assert transport.poisoned is True
    with pytest.raises(TransportError, match="poisoned"):
        transport.reduce({0: 0x1000, 1: 0x2000})
    transport.close()


# -- teardown -----------------------------------------------------------------


def test_close_destroys_the_handle_exactly_once() -> None:
    library = FakeDriver()
    transport = _transport(library)
    transport.close()
    transport.close()
    assert library.destroyed == [_CREATE_HANDLE]
    with pytest.raises(TransportError, match="closed"):
        transport.reduce({0: 0x1000, 1: 0x2000})


# -- the shard group's driver knob ---------------------------------------------


class _GroupKernelSpy:
    def __init__(self) -> None:
        self.casts: list[tuple[int, int, int]] = []

    def linear(self, weight, x_ptr, out_ptr, rows, hidden, out_features, **kwargs):
        return True

    def silu(self, gate_ptr, up_ptr, act_ptr, rows, ffn, **kwargs):
        return None

    def cast(self, x_ptr, out_ptr, count, **kwargs):
        self.casts.append((int(x_ptr), int(out_ptr), int(count)))


@pytest.fixture()
def group_env(monkeypatch):
    from tests.test_unit_distributed_staged_and_shard import FakeHipRuntime

    rt = FakeHipRuntime()
    spy = _GroupKernelSpy()
    import hipengine.kernels.hip_gfx1100.convert as convert
    import hipengine.kernels.hip_gfx1100.fused.paro_silu as paro_silu
    import hipengine.runtime.gguf_linear as gguf_linear

    monkeypatch.setattr(gguf_linear, "launch_gguf_linear", spy.linear)
    monkeypatch.setattr(paro_silu, "silu_mul_separate_out_bf16", spy.silu)
    # shard_exec binds the silu launcher at module import; both bindings must
    # point at the spy or the real kernel launches against fake pointers.
    import hipengine.distributed.shard_exec as shard_exec_module

    monkeypatch.setattr(shard_exec_module, "silu_mul_separate_out_bf16", spy.silu)
    monkeypatch.setattr(convert, "f32_to_bf16", spy.cast)
    return rt, spy


def _weights(rt, device: int, hidden: int, per_rank_ffn: int):
    from hipengine.distributed.shard_exec import upload_shard_weight

    weights = {}
    for role in ("ffn_gate", "ffn_up"):
        weights[role] = upload_shard_weight(
            rt,
            device=device,
            name=f"blk.0.{role}.rank{device}",
            layout="dense_bf16",
            quant_key="dense_bf16",
            payload=np.zeros((per_rank_ffn, hidden), dtype=np.uint8),
        )
    weights["ffn_down"] = upload_shard_weight(
        rt,
        device=device,
        name=f"blk.0.ffn_down.rank{device}",
        layout="dense_bf16",
        quant_key="dense_bf16",
        payload=np.zeros((hidden, per_rank_ffn), dtype=np.uint8),
    )
    return weights


def _group(rt, **kwargs: object) -> MlpShardGroup:
    options: dict[str, object] = {
        "devices": (0, 1),
        "streams": {0: 0, 1: 0},
        "hidden": 8,
        "per_rank_ffn": 4,
        "weights": {0: {d: _weights(rt, d, 8, 4) for d in (0, 1)}},
    }
    options.update(kwargs)
    return MlpShardGroup(rt, **options)  # type: ignore[arg-type]


def test_the_compiled_driver_is_the_default_transport(group_env) -> None:
    rt, _ = group_env
    group = _group(rt)
    assert isinstance(group._transport, CompiledStagedExchangeTransport)
    group.close()


def test_the_python_driver_is_the_registered_fallback(group_env, monkeypatch) -> None:
    rt, _ = group_env
    monkeypatch.setattr(
        staged_compiled_module(), "build_tp2_staged_exchange", lambda **_kw: FakeDriver()
    )
    group = _group(rt, driver="python")
    assert isinstance(group._transport, StagedExchangeTransport)
    group.close()


def test_the_compiled_driver_uses_the_compiled_transport(group_env, monkeypatch) -> None:
    rt, spy = group_env
    library = FakeDriver()
    monkeypatch.setattr(
        staged_compiled_module(), "build_tp2_staged_exchange", lambda **_kw: library
    )
    group = _group(rt, driver="compiled")
    assert isinstance(group._transport, CompiledStagedExchangeTransport)
    reduced = group.forward(0, {0: rt.malloc(16), 1: rt.malloc(16)})
    assert library.reduce_calls, "the compiled driver performed the reduction"
    # The boundary cast consumes the driver's mapped payload pointer on both
    # ranks - the H2D return path is gone by construction here.
    assert [c[0] for c in spy.casts] == [_PAYLOAD_PTR] * 2
    assert {c[1] for c in spy.casts} == {int(reduced[0]), int(reduced[1])}
    assert group.reductions == 1
    group.close()
    assert library.destroyed == [_CREATE_HANDLE], "group close destroys the driver"


def test_an_unknown_driver_is_rejected(group_env) -> None:
    rt, _ = group_env
    with pytest.raises(ShardGroupError, match="driver"):
        _group(rt, driver="rccl")


def staged_compiled_module():
    from hipengine.distributed import staged_compiled

    return staged_compiled
