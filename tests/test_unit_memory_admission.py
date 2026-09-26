"""Pin the unified admission budget: named refusals and both sides of each edge.

The acceptance for this budget is not "it refuses when full". Each priced
consumer has to be the named reason when *it* is the one that does not fit, the
exact boundary has to admit, and a value unrelated to any threshold has to behave
the same way on both sides. Everything here is pure arithmetic, so it runs
without a GPU; the live pre-OOM arm is separate.
"""

import pytest

from hipengine.runtime.memory_admission import (
    ADMISSION_REFUSAL_CODE,
    DEFAULT_ADMISSION_RESERVE_BYTES,
    MemoryAdmissionRefused,
    MemoryConsumer,
    price_memory_admission,
    require_memory_admission,
)

GIB = 1024**3

#: The five resident consumers the task names, in allocation order.
CONSUMERS = (
    MemoryConsumer("target_kv_pool", 8 * GIB),
    MemoryConsumer("verifier_scratch", 2 * GIB),
    MemoryConsumer("captured_graphs", GIB),
    MemoryConsumer("retained_prefix", GIB),
    MemoryConsumer("provider_kv", GIB),
)
TOTAL = sum(item.bytes for item in CONSUMERS)


def price(free_bytes, consumers):
    """Consumer arithmetic with no reserve; the reserve has its own test."""

    return price_memory_admission(free_bytes=free_bytes, consumers=consumers, reserve_bytes=0)


def _prefix(count):
    return CONSUMERS[:count]


def test_exact_boundary_admits_with_zero_headroom():
    """The request that fits exactly at the boundary succeeds."""

    decision = price_memory_admission(
        free_bytes=TOTAL + DEFAULT_ADMISSION_RESERVE_BYTES, consumers=CONSUMERS
    )
    assert decision.admitted, decision.reason
    assert decision.refused_consumer is None
    assert decision.required_bytes == TOTAL
    assert decision.available_bytes == TOTAL
    assert decision.headroom_bytes == 0
    assert decision.reason == "fits_admission_budget"


@pytest.mark.parametrize("count", range(1, len(CONSUMERS) + 1))
def test_each_consumer_is_the_named_refusal_when_it_does_not_fit(count):
    """Push every consumer to its own edge and name the one that crosses it.

    The headroom is set from the consumers before the one under test, so the
    refusal can only be attributed to that consumer.
    """

    prior = sum(item.bytes for item in _prefix(count - 1))
    subject = CONSUMERS[count - 1]

    # One byte short of the subject's own edge: the subject is the first
    # consumer that no longer fits the headroom its predecessors left.
    short = price(prior + subject.bytes - 1, CONSUMERS)
    assert not short.admitted
    assert short.refused_consumer == subject.name, (subject.name, short.refused_consumer)
    assert short.required_bytes == TOTAL
    assert short.available_bytes == prior + subject.bytes - 1
    assert short.headroom_bytes == (prior + subject.bytes - 1) - TOTAL
    assert subject.name in short.reason
    # Exactly at that edge the subject fits the headroom its predecessors left;
    # whether the whole decision admits then depends on the consumers after it,
    # which is the total only when the subject is the last one.
    exact = price(prior + subject.bytes, CONSUMERS)
    assert exact.admitted == (prior + subject.bytes >= TOTAL)


@pytest.mark.parametrize("count", range(1, len(CONSUMERS) + 1))
def test_each_consumer_admits_alone_with_one_byte_to_spare(count):
    """The other side of the same edge: one byte more than the edge admits."""

    prior = sum(item.bytes for item in _prefix(count - 1))
    subject = CONSUMERS[count - 1]
    decision = price(prior + subject.bytes + 1, _prefix(count))
    assert decision.admitted, decision.reason
    assert decision.headroom_bytes == 1


def test_unrelated_value_is_priced_like_any_other():
    """A consumer size unrelated to any threshold behaves identically on both sides."""

    odd = MemoryConsumer("unrelated_consumer", 777)
    others = (MemoryConsumer("target_kv_pool", 4096), MemoryConsumer("verifier_scratch", 512))
    fits = price(4096 + 512 + 777, others + (odd,))
    assert fits.admitted and fits.headroom_bytes == 0
    refuses = price(4096 + 512 + 776, others + (odd,))
    assert refuses.refused_consumer == "unrelated_consumer"
    # And a free-memory value unrelated to the totals is still just arithmetic.
    assert price(987654, others).admitted


def test_reserve_is_deducted_before_pricing():
    consumers = (MemoryConsumer("target_kv_pool", 1024),)
    assert price(1024, consumers).admitted
    reserved = price_memory_admission(free_bytes=1024, consumers=consumers, reserve_bytes=1)
    assert not reserved.admitted
    assert reserved.available_bytes == 1023
    assert reserved.refused_consumer == "target_kv_pool"


def test_consumer_order_decides_the_named_refusal():
    """Declaration order is allocation order, so it decides who is blamed."""

    big_first = (MemoryConsumer("first", 100), MemoryConsumer("second", 100))
    decision = price(150, big_first)
    assert decision.refused_consumer == "second"
    decision = price(50, big_first)
    assert decision.refused_consumer == "first"


def test_require_raises_a_named_memory_error_carrying_its_code():
    with pytest.raises(MemoryAdmissionRefused) as caught:
        require_memory_admission(free_bytes=GIB, consumers=CONSUMERS, reserve_bytes=0)
    refusal = caught.value
    assert isinstance(refusal, MemoryError)
    assert refusal.code == ADMISSION_REFUSAL_CODE
    assert refusal.refused_consumer == "target_kv_pool"
    assert refusal.required_bytes == TOTAL
    assert refusal.available_bytes == GIB
    assert refusal.decision.admitted is False
    assert refusal.decision.as_dict()["refused_consumer"] == "target_kv_pool"
    # The message names the consumer rather than only a total.
    assert "target_kv_pool" in str(refusal)


def test_require_returns_the_decision_when_it_fits():
    decision = require_memory_admission(free_bytes=TOTAL, consumers=CONSUMERS, reserve_bytes=0)
    assert decision.admitted and decision.refused_consumer is None


def test_mapping_input_and_input_validation():
    decision = price_memory_admission(
        free_bytes=300, consumers={"kv_pool": 100, "scratch": 200}, reserve_bytes=0
    )
    assert decision.admitted
    assert [item.name for item in decision.consumers] == ["kv_pool", "scratch"]
    with pytest.raises(ValueError):
        MemoryConsumer("  ", 1)
    with pytest.raises(ValueError):
        MemoryConsumer("kv_pool", -1)
    with pytest.raises(ValueError):
        price_memory_admission(free_bytes=1, consumers=CONSUMERS, reserve_bytes=-1)


def test_zero_byte_consumer_never_becomes_the_refusal():
    consumers = (MemoryConsumer("empty", 0), MemoryConsumer("kv_pool", 10))
    decision = price(5, consumers)
    assert decision.refused_consumer == "kv_pool"
