"""Unit tests for the compiled device exchange's host-side lifecycle.

The GPU tier drives the real exchange against real kernels. These cover the part
that is pure host state and has to hold on every machine:

* a clear (``step_begin``/``reset_timeouts``) refuses while an earlier step's
  outcome is unobserved. The timeout flags are sticky on purpose, and the spin
  kernel writes nothing when it times out, so a clear that erases an unobserved
  failure would leave a stale output row passing as a result;
* ``wait`` is what observes a step, and therefore what makes the next clear
  legal;
* both real submission shapes still run: the graphed schedule's per-step
  ``step_begin`` plus one ``wait``, and the batched group's single
  ``reset_timeouts``, per-layer ``bump``, single ``wait``.

The driver is faked, so nothing here needs a GPU or a compiled library.
"""

from __future__ import annotations

import ctypes

import pytest

from hipengine.distributed.device_exchange_compiled import CompiledDeviceExchange
from hipengine.distributed.transport import TransportStateError


class FakeDeviceExchangeDriver:
    """Stand-in for the hipcc-built ``tp2_dev_exchange_*`` entry points.

    The functions are assigned as instance attributes rather than defined as
    methods, because the wrapper sets ``argtypes``/``restype`` on each of them
    and a bound method does not accept attribute assignment.
    """

    def __init__(self, *, fail_wait_at: int | None = None) -> None:
        self.calls: list[str] = []
        self.creates: list[tuple] = []
        self.destroyed: list[int] = []
        self.waits = 0
        self.fail_wait_at = fail_wait_at
        # What the driver would report as owned; the real one counts its device
        # and mapped allocations, and the lifecycle tests assert it returns to
        # its resting value.
        self.live_allocations = 0
        self._message = b""

        def create(devices, world, streams, num_layers, hidden, max_spins, error_code):
            self.calls.append("create")
            self.creates.append(
                (
                    tuple(int(devices[i]) for i in range(world)),
                    int(world),
                    tuple(int(streams[i]) for i in range(world)),
                    int(num_layers),
                    int(hidden),
                    int(max_spins),
                )
            )
            return 0x1234

        def step_begin(handle):
            self.calls.append("step_begin")
            return 0

        def reset_timeouts(handle):
            self.calls.append("reset_timeouts")
            return 0

        def bump(handle):
            self.calls.append("bump")
            return 0

        def enqueue_rank(handle, rank, own_partial, slot, out_payload):
            self.calls.append(f"enqueue:{int(rank)}:{int(slot)}")
            return 0

        def wait(handle):
            self.calls.append("wait")
            self.waits += 1
            if self.fail_wait_at is not None and self.waits >= self.fail_wait_at:
                self._message = b"spin timeout at slot 0"
                return 1
            return 0

        def last_error(handle):
            return self._message

        def live_allocations():
            return int(self.live_allocations)

        def destroy(handle) -> None:
            value = handle.value if isinstance(handle, ctypes.c_void_p) else handle
            self.destroyed.append(int(value))

        self.tp2_dev_exchange_create = create
        self.tp2_dev_exchange_step_begin = step_begin
        self.tp2_dev_exchange_reset_timeouts = reset_timeouts
        self.tp2_dev_exchange_bump = bump
        self.tp2_dev_exchange_enqueue_rank = enqueue_rank
        self.tp2_dev_exchange_wait = wait
        self.tp2_dev_exchange_last_error = last_error
        self.tp2_dev_exchange_live_allocations = live_allocations
        self.tp2_dev_exchange_destroy = destroy


def _exchange(library: FakeDeviceExchangeDriver, **kwargs: object) -> CompiledDeviceExchange:
    options: dict[str, object] = {
        "devices": (0, 1),
        "streams": {0: 100, 1: 101},
        "num_layers": 2,
        "hidden": 8,
        "max_spins": 1000,
    }
    options.update(kwargs)
    return CompiledDeviceExchange(object(), library=library, **options)  # type: ignore[arg-type]


def test_create_receives_the_validated_configuration() -> None:
    library = FakeDeviceExchangeDriver()
    exchange = _exchange(
        library, devices=(3, 5), streams={3: 9, 5: 7}, num_layers=4, hidden=16
    )
    assert library.creates == [((3, 5), 2, (9, 7), 4, 16, 1000)]
    exchange.close()
    assert library.destroyed == [0x1234]


def test_ownership_accounting_is_bound_to_the_driver() -> None:
    """The count the lifecycle tests read comes from the driver, not a local."""

    from hipengine.distributed.device_exchange_compiled import (
        device_exchange_live_allocations,
    )

    library = FakeDeviceExchangeDriver()
    assert device_exchange_live_allocations(library) == 0
    library.live_allocations = 8
    assert device_exchange_live_allocations(library) == 8


def test_invalid_construction_never_touches_the_driver() -> None:
    library = FakeDeviceExchangeDriver()
    with pytest.raises(TransportStateError, match="exactly two ranks"):
        _exchange(library, devices=(0,))
    with pytest.raises(TransportStateError, match="distinct devices"):
        _exchange(library, devices=(0, 0))
    with pytest.raises(TransportStateError, match="no stream given"):
        _exchange(library, streams={0: 1})
    assert library.calls == []


# -- the unobserved-clear rule ------------------------------------------------


def test_reset_refuses_while_a_step_is_unobserved() -> None:
    """The reported hole: enqueue, reset before wait, wait sees no failure.

    The reset is refused instead of enqueued, so the flag the timed-out spin set
    is still there when the wait looks for it.
    """

    library = FakeDeviceExchangeDriver()
    exchange = _exchange(library)
    exchange.step_begin()
    with pytest.raises(TransportStateError, match="not waited"):
        exchange.reset_timeouts()
    assert "reset_timeouts" not in library.calls, (
        "the clear must not reach the driver: enqueueing it is what erases the "
        "failure"
    )
    # The step is still observable, so the wait still reports it.
    exchange.wait()
    exchange.reset_timeouts()
    assert library.calls == ["create", "step_begin", "wait", "reset_timeouts"]
    exchange.close()


def test_step_begin_refuses_while_a_step_is_unobserved() -> None:
    """``step_begin`` clears the flags too, so it starts a step the same way."""

    library = FakeDeviceExchangeDriver()
    exchange = _exchange(library)
    exchange.bump()
    with pytest.raises(TransportStateError, match="not waited"):
        exchange.step_begin()
    assert library.calls == ["create", "bump"]
    exchange.close()


def test_a_batched_group_resets_once_bumps_per_layer_and_waits_once() -> None:
    """The bulk prefill's shape, and that the next group's reset is legal."""

    library = FakeDeviceExchangeDriver()
    exchange = _exchange(library)
    for _group in range(2):
        exchange.reset_timeouts()
        for layer in range(3):
            exchange.bump()
            exchange.enqueue_rank(0, 0x1000 + layer, layer, 0x2000 + layer)
            exchange.enqueue_rank(1, 0x3000 + layer, layer, 0x4000 + layer)
        exchange.wait()
    assert [c for c in library.calls if c == "reset_timeouts"] == [
        "reset_timeouts",
        "reset_timeouts",
    ]
    assert exchange.step_begins == 6
    assert exchange.waits == 2
    exchange.close()


def test_capture_time_enqueues_do_not_block_a_later_clear() -> None:
    """A capture records enqueues into a graph that has not run.

    Nothing has executed, so no flag can have been set, and the counter that
    tracks unobserved work must not treat them as a submitted step - otherwise
    the first token after a capture could not begin.
    """

    library = FakeDeviceExchangeDriver()
    exchange = _exchange(library)
    for layer in range(2):
        exchange.enqueue_rank(0, 0x1000 + layer, layer, 0x2000 + layer)
        exchange.enqueue_rank(1, 0x3000 + layer, layer, 0x4000 + layer)
    exchange.step_begin()
    exchange.wait()
    exchange.close()


def test_the_graphed_shape_begins_waits_and_then_begins_again() -> None:
    library = FakeDeviceExchangeDriver()
    exchange = _exchange(library)
    for _step in range(2):
        exchange.step_begin()
        exchange.enqueue_rank(0, 0x1000, 1, 0x2000)
        exchange.enqueue_rank(1, 0x3000, 1, 0x4000)
        exchange.wait()
    assert exchange.step_begins == 2 and exchange.waits == 2
    exchange.close()


def test_a_failed_wait_poisons_without_observing_the_step() -> None:
    library = FakeDeviceExchangeDriver(fail_wait_at=1)
    exchange = _exchange(library)
    exchange.step_begin()
    with pytest.raises(TransportStateError, match="spin timeout"):
        exchange.wait()
    assert exchange.poisoned
    with pytest.raises(TransportStateError, match="poisoned"):
        exchange.step_begin()
    exchange.close()
