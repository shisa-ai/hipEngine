"""CPU tests for distributed context lifecycle and rank-bound ownership.

No ROCm is required: the hardware probe is faked and the mock transport supplies
collective semantics.
"""

from __future__ import annotations

import pytest

from hipengine.core.device import Device
from hipengine.distributed.context import DistributedContext, RankRuntime
from hipengine.distributed.mock import MockTransport
from hipengine.distributed.plan import DistributedPlan, PlanError
from hipengine.distributed.transport import TransportStateError


class FakeHipRuntime:
    """Records device selection and stream calls without loading libamdhip64."""

    def __init__(self, *, device_count: int = 2, current: int = 0) -> None:
        self._count = int(device_count)
        self._current = int(current)
        self.selection: list[int] = []
        self.calls: list[tuple[str, int]] = []
        self._next = 0x100

    def device_count(self) -> int:
        return self._count

    def get_device(self) -> int:
        return self._current

    def set_device(self, device: int) -> None:
        self._current = int(device)
        self.selection.append(int(device))

    def stream_create(self, *, nonblocking: bool = True) -> int:
        self.calls.append(("stream_create", self._current))
        self._next += 8
        return self._next

    def stream_destroy(self, stream: int) -> None:
        self.calls.append(("stream_destroy", self._current))

    def stream_synchronize(self, stream: int) -> None:
        self.calls.append(("stream_synchronize", self._current))

    def event_create(self, *, flags: int = 0) -> int:
        self.calls.append(("event_create", self._current))
        self._next += 8
        return self._next

    def event_record(self, event: int, stream: int = 0) -> None:
        self.calls.append(("event_record", self._current))

    def event_synchronize(self, event: int) -> None:
        self.calls.append(("event_synchronize", self._current))

    def event_destroy(self, event: int) -> None:
        self.calls.append(("event_destroy", self._current))

    def mem_get_info(self) -> tuple[int, int]:
        self.calls.append(("mem_get_info", self._current))
        return (1 << 30, 2 << 30)


def test_single_rank_context_has_no_transport() -> None:
    plan = DistributedPlan.resolve([0], hidden_size=5120)
    context = DistributedContext.create(plan, runtime=FakeHipRuntime(), require_hardware=True)
    assert context.world_size == 1
    assert context.transport is None
    assert context.collectives_available is False
    context.sync()  # no-op
    context.close()


def test_single_rank_context_rejects_transport() -> None:
    plan = DistributedPlan.resolve([0], hidden_size=5120)
    with pytest.raises(TransportStateError):
        DistributedContext.create(plan, transport=MockTransport(world_size=1), require_hardware=False)


def test_multi_rank_context_uses_supplied_transport() -> None:
    plan = DistributedPlan.resolve([0, 1], hidden_size=5120)
    transport = MockTransport(world_size=2)
    context = DistributedContext.create(
        plan, transport=transport, runtime=FakeHipRuntime(), require_hardware=False
    )
    assert context.collectives_available
    assert context.rank(1).device == Device("hip", 1)
    context.close()
    assert context.transport is None


def test_transport_world_size_mismatch_is_rejected_and_closed() -> None:
    plan = DistributedPlan.resolve([0, 1], hidden_size=5120)
    transport = MockTransport(world_size=3)
    with pytest.raises(TransportStateError):
        DistributedContext.create(plan, transport=transport, runtime=FakeHipRuntime(), require_hardware=False)
    with pytest.raises(TransportStateError):
        transport.group_start()  # closed by the failed context construction


def test_plan_device_validation_rejects_missing_device() -> None:
    plan = DistributedPlan.resolve([0, 1], hidden_size=5120)
    with pytest.raises(PlanError):
        DistributedContext.create(plan, transport=MockTransport(world_size=2), runtime=FakeHipRuntime(device_count=1))


def test_rank_runtime_scopes_device_for_stream_and_memory_calls() -> None:
    runtime = FakeHipRuntime(current=0)
    rank = RankRuntime(rank=1, device=Device("hip", 1), runtime=runtime)
    stream = rank.stream_create()
    rank.stream_synchronize(stream)
    rank.event_create()
    assert runtime.get_device() == 0
    assert runtime.selection == [1, 0, 1, 0, 1, 0]
    assert all(device == 1 for _, device in runtime.calls)


def test_rank_accessor_rejects_out_of_range() -> None:
    plan = DistributedPlan.resolve([0], hidden_size=5120)
    context = DistributedContext.create(plan, runtime=FakeHipRuntime(), require_hardware=False)
    with pytest.raises(PlanError):
        context.rank(3)
