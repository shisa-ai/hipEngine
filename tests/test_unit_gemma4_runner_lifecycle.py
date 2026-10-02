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
from hipengine.runtime import gemma4_int8_kv as int8_kv_module
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


def _wide_config(*, head_dim: int = 512, num_layers: int = 1) -> Gemma4TextConfig:
    """A config whose attention layers have a real Gemma 4 global geometry.

    head_dim 512 is where the BF16 prefill bound and the direct INT8 consumer
    bound actually diverge, so it is the only width that can show which one a
    runner applies.
    """

    attention = tuple(
        Gemma4AttentionGeometry(
            layer_type="full_attention",
            num_heads=16,
            num_kv_heads=2,
            head_dim=head_dim,
            rope=Gemma4RopeConfig(
                rope_theta=10_000.0, head_dim=head_dim, rope_angles=head_dim // 64
            ),
            sliding_window=None,
            k_eq_v=False,
        )
        for _ in range(num_layers)
    )
    return Gemma4TextConfig(
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=8,
        num_experts=4,
        top_k_experts=2,
        rms_norm_eps=1e-6,
        attention=attention,
        vocab_size=64,
    )


def _fake_allocated_runner(
    monkeypatch: pytest.MonkeyPatch,
    *,
    config: Gemma4TextConfig,
    capacity: int,
    max_block: int = 8,
    kv_storage: str = "bf16",
) -> tuple[Gemma4Runner, _FakeAllocator, _FakeAllocator]:
    """Build a runner over fake allocators for both the runner and the INT8 owner.

    No device is touched, so the admission checks can be exercised at contexts a
    real card could not hold.
    """

    runner_allocator = _FakeAllocator()
    monkeypatch.setattr(gemma4_module, "malloc", runner_allocator.malloc)
    monkeypatch.setattr(gemma4_module, "free", runner_allocator.free)
    owner_allocator = _FakeAllocator()
    monkeypatch.setattr(int8_kv_module, "malloc", owner_allocator.malloc)
    monkeypatch.setattr(int8_kv_module, "free", owner_allocator.free)
    monkeypatch.setattr(
        int8_kv_module, "copy_host_to_device", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        int8_kv_module, "enqueue_host_to_device", lambda *args, **kwargs: None
    )
    runner = Gemma4Runner(
        weights=_weights(config),
        capacity=capacity,
        max_block=max_block,
        kv_storage=kv_storage,
    )
    return runner, runner_allocator, owner_allocator


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
    # token_ids, hidden, normalized, logits, argmax out, argmax scratch, then
    # key/value for the first layer. Fail while allocating the first layer's
    # key cache.
    allocator = _FakeAllocator(fail_at=7)
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
    assert len(allocator.allocated) == 6
    assert sorted(buffer.ptr for buffer in allocator.freed) == sorted(
        buffer.ptr for buffer in allocator.allocated
    )


def test_bf16_is_the_default_storage_and_allocates_no_int8_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The default request must leave the comparison path byte-for-byte as it was."""

    runner, allocator = _make_runner(monkeypatch, capacity=64, max_block=8)
    try:
        assert runner.uses_int8_kv is False
        assert runner.kv_cache is None
        assert runner.kv_storage_resolved == "bf16"
        # Two layers, key/value each: the BF16 path still owns its caches.
        assert len(runner._caches) == 4
        assert all(buffer.nbytes > 0 for buffer in runner._caches)
        assert allocator.allocated  # sanity: the fake allocator was used
    finally:
        runner.close()


def test_int8_storage_allocates_the_owner_instead_of_bf16_caches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """INT8 storage replaces the BF16 K/V caches with one owned INT8 cache.

    The runner must hold no BF16 K/V cache and no BF16 shadow on this path, and
    closing must release the owner's buffers exactly once.
    """

    allocator = _FakeAllocator()
    monkeypatch.setattr(gemma4_module, "malloc", allocator.malloc)
    monkeypatch.setattr(gemma4_module, "free", allocator.free)
    owner_allocator = _FakeAllocator()
    monkeypatch.setattr(int8_kv_module, "malloc", owner_allocator.malloc)
    monkeypatch.setattr(int8_kv_module, "free", owner_allocator.free)
    monkeypatch.setattr(
        int8_kv_module, "copy_host_to_device", lambda *args, **kwargs: None
    )

    runner = Gemma4Runner(
        weights=_weights(_config()),
        capacity=64,
        max_block=8,
        kv_storage="int8_per_token_head",
        kv_scale_dtype="fp16",
        kv_scale_granularity="per_token_head",
    )
    try:
        assert runner.uses_int8_kv is True
        assert runner.kv_cache is not None
        assert runner.kv_storage_resolved == "int8_per_token_head"
        assert runner._kv == []
        assert runner._caches == []
        assert owner_allocator.allocated, "the owner allocated nothing"
        assert runner.kv_cache.allocated_bytes == sum(
            buffer.nbytes for buffer in owner_allocator.allocated
        )
    finally:
        runner.close()

    assert runner.kv_cache is None
    assert len(owner_allocator.freed) == len(owner_allocator.allocated)
    assert sorted(b.ptr for b in owner_allocator.freed) == sorted(
        b.ptr for b in owner_allocator.allocated
    )


def test_int8_storage_rejects_unsupported_scale_granularity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unsupported layout is a named capability miss, not a silent fallback."""

    allocator = _FakeAllocator()
    monkeypatch.setattr(gemma4_module, "malloc", allocator.malloc)
    monkeypatch.setattr(gemma4_module, "free", allocator.free)
    with pytest.raises(ValueError, match="per_token_head"):
        Gemma4Runner(
            weights=_weights(_config()),
            capacity=64,
            max_block=8,
            kv_storage="int8_per_token_head",
            kv_scale_granularity="per_channel",
        )


def test_unsupported_kv_storage_is_refused_by_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    allocator = _FakeAllocator()
    monkeypatch.setattr(gemma4_module, "malloc", allocator.malloc)
    monkeypatch.setattr(gemma4_module, "free", allocator.free)
    with pytest.raises(ValueError, match="unsupported KV storage"):
        Gemma4Runner(
            weights=_weights(_config()), capacity=64, max_block=8, kv_storage="fp8"
        )


def test_int8_reset_rewinds_the_owner(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset must rewind the INT8 owner too, so a reused runner starts empty."""

    monkeypatch.setattr(gemma4_module, "malloc", _FakeAllocator().malloc)
    monkeypatch.setattr(gemma4_module, "free", _FakeAllocator().free)
    monkeypatch.setattr(int8_kv_module, "malloc", _FakeAllocator().malloc)
    monkeypatch.setattr(int8_kv_module, "free", _FakeAllocator().free)
    monkeypatch.setattr(
        int8_kv_module, "copy_host_to_device", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        int8_kv_module, "enqueue_host_to_device", lambda *args, **kwargs: None
    )
    runner = Gemma4Runner(
        weights=_weights(_config()),
        capacity=64,
        max_block=8,
        kv_storage="int8_per_token_head",
    )
    try:
        owner = runner.kv_cache
        assert owner is not None
        owner.begin_block(write_offset=4, rows=3, stream=0)
        assert owner._positions_host[:3].tolist() == [4, 5, 6]
        runner.reset()
        assert owner._positions_host[:3].tolist() == [0, 0, 0]
        assert runner.position == 0
    finally:
        runner.close()


@pytest.mark.parametrize(
    "capacity",
    [8192, 15700, 15856],
    ids=("unrelated-safe-length", "between-the-two-bounds", "int8-edge"),
)
def test_int8_storage_admits_a_context_only_the_selected_consumer_bounds(
    monkeypatch: pytest.MonkeyPatch, capacity: int
) -> None:
    """The selected consumer's bound decides, never the unselected one.

    At head_dim 512 the direct INT8 consumer needs ``(capacity + 512 + 16) * 4``
    bytes of LDS; the BF16 prefill kernel needs ``(capacity + 512 + 256) * 4``.
    Between the two bounds -- and at the INT8 edge, exactly 65536 bytes -- the
    INT8 path is runnable and the BF16 path is not. A runner that validated the
    BF16 bound regardless of storage refused these contexts.

    The lengths are deliberately not the configured block or window sizes: 8192
    is unrelated to both bounds, 15700 sits between them, and 15856 is the
    largest capacity the INT8 consumer fits.
    """

    runner, _, owner_allocator = _fake_allocated_runner(
        monkeypatch,
        config=_wide_config(),
        capacity=capacity,
        kv_storage="int8_per_token_head",
    )
    try:
        assert runner.uses_int8_kv is True
        assert runner.kv_storage_resolved == "int8_per_token_head"
        assert owner_allocator.allocated, "the INT8 owner allocated nothing"
    finally:
        runner.close()


def test_int8_storage_refuses_a_context_past_its_own_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One token past the INT8 edge is still a named capability miss."""

    with pytest.raises(ValueError, match="shared memory"):
        _fake_allocated_runner(
            monkeypatch,
            config=_wide_config(),
            capacity=15857,
            kv_storage="int8_per_token_head",
        )


def test_bf16_storage_still_refuses_past_its_own_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fix picks the right bound; it does not loosen either consumer's limit."""

    with pytest.raises(NotImplementedError, match="shared memory"):
        _fake_allocated_runner(
            monkeypatch,
            config=_wide_config(),
            capacity=15700,
            kv_storage="bf16",
        )


def test_close_releases_an_unused_runner_without_loading_the_hip_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Releasing a runner that never touched a device must not load libamdhip64.

    ``Gemma4LayerScratch.free`` used to acquire the HIP runtime unconditionally,
    so ``close()`` on a fake-allocated runner failed wherever libamdhip64.so is
    absent -- a no-ROCm CI or publish runner. An unused scratch owns no events
    and no buffers, so its release is a pure host operation.
    """

    import hipengine.core.hip as hip_module

    def _no_runtime(*args: object, **kwargs: object) -> object:
        raise OSError("libamdhip64.so: cannot open shared object file")

    monkeypatch.setattr(hip_module, "get_hip_runtime", _no_runtime)

    runner, allocator = _make_runner(monkeypatch, capacity=64, max_block=8)
    runner.close()
    assert len(allocator.freed) == len(allocator.allocated)

    int8_runner, runner_allocator, owner_allocator = _fake_allocated_runner(
        monkeypatch,
        config=_config(),
        capacity=64,
        kv_storage="int8_per_token_head",
    )
    int8_runner.close()
    assert len(runner_allocator.freed) == len(runner_allocator.allocated)
    assert len(owner_allocator.freed) == len(owner_allocator.allocated)


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


def test_multi_block_forward_takes_the_head_only_on_the_final_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the last block's logits survive ``forward``, so only that block needs a head.

    ``forward`` forwards a prompt as consecutive blocks and returns the logits of
    the final one. Every earlier block still ran the final norm, a vocab-wide
    projection and a device-to-host copy whose result the loop overwrites on the
    next iteration -- a head launch per block that nobody reads. On the real
    artifact that is 1 wasted launch at 1024 tokens and 7 at 4096.
    """

    from contextlib import contextmanager

    base = _config(num_layers=2, hidden_size=16)
    # A namespace rather than the dataclass: the forward path reads ``embed_scale``
    # and the CPU reference config does not declare it.
    config = SimpleNamespace(
        attention=base.attention,
        hidden_size=base.hidden_size,
        intermediate_size=base.intermediate_size,
        moe_intermediate_size=base.moe_intermediate_size,
        num_experts=base.num_experts,
        top_k_experts=base.top_k_experts,
        rms_norm_eps=base.rms_norm_eps,
        vocab_size=base.vocab_size,
        embed_scale=1.0,
        final_logit_softcapping=0.0,
    )
    runner, _ = _make_runner(monkeypatch, capacity=64, max_block=8, config=config)

    # Empty so the layer loop costs nothing here; this pins the head, not the stack.
    runner.weights.layers = []
    embedding = SimpleNamespace(buffer=DeviceBuffer(ptr=0x9000, nbytes=16))
    runner.weights.embed_tokens = embedding
    runner.weights.lm_head = embedding
    runner.weights.final_norm = SimpleNamespace(
        buffer=DeviceBuffer(ptr=0x9100, nbytes=16)
    )

    @contextmanager
    def _no_session(self: object) -> object:
        yield

    monkeypatch.setattr(
        gemma4_module.Gemma4Runner, "_q8_mmq_prefill_session", _no_session
    )
    for name in (
        "enqueue_host_to_device",
        "copy_device_to_host",
        "host_array_ptr",
        "launch_gguf_embedding",
        "gemma4_scale_bf16",
    ):
        monkeypatch.setattr(gemma4_module, name, lambda *args, **kwargs: None)

    norms: list[tuple] = []
    heads: list[tuple] = []
    monkeypatch.setattr(
        gemma4_module,
        "gemma4_rmsnorm_f32w_bf16",
        lambda *args, **kwargs: norms.append(args),
    )
    monkeypatch.setattr(
        gemma4_module, "launch_gguf_linear", lambda *args, **kwargs: heads.append(args)
    )

    logits = runner.forward(list(range(1, 17)))  # 16 tokens over two 8-token blocks

    assert logits.shape == (config.vocab_size,)
    assert len(norms) == 1, (
        f"the final norm ran {len(norms)} times across 2 blocks; only the block "
        "whose logits are returned can need it"
    )
    assert len(heads) == 1, (
        f"lm_head ran {len(heads)} times across 2 blocks; the loop overwrites "
        "every block but the last, so the earlier projections are unread work"
    )
