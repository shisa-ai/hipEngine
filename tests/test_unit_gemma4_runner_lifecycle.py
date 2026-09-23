"""CPU lifecycle regressions for :class:`Gemma4Runner`.

Two ownership defects are pinned here without any HIP execution:

* the reusable staging buffer used to allocate exactly the requested size and
  keep every replaced allocation, so a per-decode mask that grows with the
  sequence retained O(n^2) bytes over a session; and
* ``__post_init__`` allocated the runner buffers with no exception cleanup, so
  a failure partway through construction leaked every buffer already taken.

Both are checked through a fake allocator: ``malloc``/``free`` in the runner
module are replaced with counters, and the config/weights are a tiny in-memory
stand-in. No device, no kernel, no ROCm import.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from hipengine.core.memory import DeviceBuffer
from hipengine.kernels.cpu_reference.gemma4 import (
    Gemma4AttentionGeometry,
    Gemma4RopeConfig,
    Gemma4TextConfig,
)
from hipengine.runtime import gemma4 as gemma4_module
from hipengine.runtime.gemma4 import Gemma4Runner


class _FakeAllocator:
    """Deterministic malloc/free pair that records every call."""

    def __init__(self, *, fail_at: int | None = None) -> None:
        self.next_ptr = 0x1000
        self.allocated: list[DeviceBuffer] = []
        self.freed: list[DeviceBuffer] = []
        self.calls = 0
        self.fail_at = fail_at

    def malloc(self, nbytes: int) -> DeviceBuffer:
        self.calls += 1
        if self.fail_at is not None and self.calls == self.fail_at:
            raise MemoryError(f"fake allocator exhausted at call {self.calls}")
        buffer = DeviceBuffer(ptr=self.next_ptr, nbytes=nbytes)
        self.next_ptr += nbytes + 0x1000
        self.allocated.append(buffer)
        return buffer

    def free(self, buffer: DeviceBuffer) -> None:
        self.freed.append(buffer)


def _config(*, num_layers: int = 2, hidden_size: int = 16) -> Gemma4TextConfig:
    attention = tuple(
        Gemma4AttentionGeometry(
            layer_type="sliding_attention",
            num_heads=4,
            num_kv_heads=2,
            head_dim=8,
            rope=Gemma4RopeConfig(rope_theta=10_000.0, head_dim=8, rope_angles=4),
            sliding_window=16,
            k_eq_v=False,
        )
        for _ in range(num_layers)
    )
    return Gemma4TextConfig(
        hidden_size=hidden_size,
        intermediate_size=32,
        moe_intermediate_size=8,
        num_experts=4,
        top_k_experts=2,
        rms_norm_eps=1e-6,
        attention=attention,
        vocab_size=64,
    )


def _weights(config: Gemma4TextConfig) -> SimpleNamespace:
    # ``__post_init__`` reads only ``weights.config``; the projections are only
    # touched by a forward pass, which these lifecycle tests never run.
    return SimpleNamespace(config=config)


def _make_runner(
    monkeypatch: pytest.MonkeyPatch,
    *,
    allocator: _FakeAllocator | None = None,
    capacity: int = 64,
    max_block: int = 8,
    config: Gemma4TextConfig | None = None,
) -> tuple[Gemma4Runner, _FakeAllocator]:
    allocator = allocator or _FakeAllocator()
    monkeypatch.setattr(gemma4_module, "malloc", allocator.malloc)
    monkeypatch.setattr(gemma4_module, "free", allocator.free)
    runner = Gemma4Runner(
        weights=_weights(config or _config()), capacity=capacity, max_block=max_block
    )
    return runner, allocator


def test_staging_growth_retains_linear_not_quadratic_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mask that grows every decode must not retain O(n^2) bytes.

    The first allocation is sized 1 and each later request is one byte larger,
    which is the shape of a keep-mask whose column count follows the sequence.
    Exact-size growth retains the sum of every request (N(N+1)/2 for the last N
    decodes); geometric growth retains less than twice the largest request.
    """

    runner, allocator = _make_runner(monkeypatch, capacity=1024, max_block=8)
    try:
        before = len(allocator.allocated)
        largest_request = 0
        for nbytes in range(1, 513):
            buffer = runner._staging_buffer("mask", nbytes)
            assert buffer.nbytes >= nbytes
            largest_request = nbytes
        growth = allocator.allocated[before:]

        assert largest_request == 512
        assert growth, "the staging buffer never allocated; the test is vacuous"
        # Every retained buffer is at least the request it served.
        assert max(buffer.nbytes for buffer in growth) >= largest_request
        total_retained = sum(buffer.nbytes for buffer in growth)
        assert total_retained <= 2 * max(buffer.nbytes for buffer in growth), (
            f"staging growth retained {total_retained} bytes for a peak request "
            f"of {largest_request}; growth is not bounded by the largest request"
        )
        # Old allocations are deliberately kept: freeing a buffer an in-flight
        # kernel still reads is a use-after-free. None may be released here.
        assert allocator.freed == []
    finally:
        runner.close()


def test_staging_buffer_reuses_a_large_enough_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A smaller request must reuse the current buffer, not allocate again."""

    runner, allocator = _make_runner(monkeypatch, capacity=64, max_block=8)
    try:
        first = runner._staging_buffer("mask", 256)
        assert runner._staging_buffer("mask", 16) is first
        assert runner._staging_buffer("mask", 256) is first
        # Growth still happens, and only once, when the request exceeds it.
        grown = runner._staging_buffer("mask", 257)
        assert grown is not first
        assert grown.nbytes >= 257
        assert runner._staging_buffer("mask", 1) is grown
    finally:
        runner.close()


def test_constructor_failure_frees_every_allocation_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mid-constructor allocation failure must not leak earlier buffers."""

    allocator = _FakeAllocator(fail_at=3)
    monkeypatch.setattr(gemma4_module, "malloc", allocator.malloc)
    monkeypatch.setattr(gemma4_module, "free", allocator.free)

    with pytest.raises(MemoryError):
        Gemma4Runner(weights=_weights(_config()), capacity=64, max_block=8)

    # The failing call did not allocate, so the two successful ones did.
    assert len(allocator.allocated) == 2
    assert len(allocator.freed) == 2
    assert {buffer.ptr for buffer in allocator.freed} == {
        buffer.ptr for buffer in allocator.allocated
    }
    assert len({buffer.ptr for buffer in allocator.freed}) == len(allocator.freed)


def test_constructor_failure_after_layer_scratch_frees_the_scratch_too(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The failure can land after a layer scratch exists; it must be released.

    The scratch owns no buffers until its first forward use, so the observable
    contract is that cleanup runs over the scratch list rather than skipping it.
    """

    config = _config(num_layers=2)
    # token_ids, hidden, normalized, logits, then key/value for the first
    # layer. Fail while allocating the first layer's key cache.
    allocator = _FakeAllocator(fail_at=5)
    monkeypatch.setattr(gemma4_module, "malloc", allocator.malloc)
    monkeypatch.setattr(gemma4_module, "free", allocator.free)
    scratches_freed = []
    original_free = gemma4_module.Gemma4LayerScratch.free

    def record_free(scratch):
        scratches_freed.append(scratch)
        original_free(scratch)

    monkeypatch.setattr(gemma4_module.Gemma4LayerScratch, "free", record_free)
    with pytest.raises(MemoryError):
        Gemma4Runner(weights=_weights(config), capacity=64, max_block=8)

    assert len(scratches_freed) == 1
    assert len(allocator.allocated) == 4
    assert sorted(buffer.ptr for buffer in allocator.freed) == sorted(
        buffer.ptr for buffer in allocator.allocated
    )


def test_close_is_idempotent_and_frees_growth_allocations_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Normal shutdown releases every owned buffer, including grown staging."""

    runner, allocator = _make_runner(monkeypatch, capacity=64, max_block=8)
    runner._staging_buffer("mask", 1)
    runner._staging_buffer("mask", 100)
    runner._staging_buffer("cos", 16)

    runner.close()
    runner.close()

    assert len(allocator.freed) == len(allocator.allocated)
    assert sorted(buffer.ptr for buffer in allocator.freed) == sorted(
        buffer.ptr for buffer in allocator.allocated
    )
    assert len({buffer.ptr for buffer in allocator.freed}) == len(allocator.freed)
    assert runner._buffers == []
    assert runner._staging == {}
    assert runner._scratches == []
    assert runner._kv == []
    assert runner._caches == []


def test_close_after_constructor_failure_does_not_double_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The constructor's own cleanup must be the only release on failure."""

    allocator = _FakeAllocator(fail_at=3)
    monkeypatch.setattr(gemma4_module, "malloc", allocator.malloc)
    monkeypatch.setattr(gemma4_module, "free", allocator.free)

    runner = Gemma4Runner.__new__(Gemma4Runner)
    with pytest.raises(MemoryError):
        runner.__init__(weights=_weights(_config()), capacity=64, max_block=8)
    runner.close()
    runner.close()

    assert len(allocator.freed) == len(allocator.allocated)
    assert len({buffer.ptr for buffer in allocator.freed}) == len(allocator.freed)
