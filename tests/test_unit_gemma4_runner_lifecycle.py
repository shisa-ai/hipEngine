"""CPU lifecycle regressions for :class:`Gemma4Runner`.

Ownership defects pinned here without any HIP execution:

* the reusable staging buffer used to allocate exactly the requested size and
  keep every replaced allocation, so a per-decode mask that grows with the
  sequence retained O(n^2) bytes over a session;
* ``__post_init__`` allocated the runner buffers with no exception cleanup, so
  a failure partway through construction leaked every buffer already taken; and
* every layer carried its own transient attention workspace, so a deep prefill
  retained one growth chain per layer instead of one for the block.

The runner tests are checked through a fake allocator: ``malloc``/``free`` in
the runner module are replaced with counters, and the config/weights are a tiny
in-memory stand-in. The shared-workspace tests drive ``Gemma4AttentionScratch``
with a fake runtime of its own, which is the only thing it asks of a device:
``current_device``, an allocator, and ``stream_synchronize``. No device, no
kernel, no ROCm import.
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
    # ``__post_init__`` reads only ``weights.config`` and the per-layer dense
    # widths. ``layers``, ``final_norm`` and ``embed_tokens`` are only touched by
    # a forward pass, and the one test that runs the block body replaces every
    # kernel it reaches, so placeholders are enough.
    layers = tuple(SimpleNamespace() for _ in range(len(config.attention)))
    return SimpleNamespace(
        config=config,
        layers=layers,
        dense_intermediate=(),
        final_norm=SimpleNamespace(buffer=SimpleNamespace(ptr=0)),
        embed_tokens=SimpleNamespace(),
        lm_head=None,
    )


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


@pytest.mark.parametrize("capacity", [15700, 15856, 16640, 32771])
def test_bf16_storage_admits_global_logit_scratch_past_resident_bound(
    monkeypatch: pytest.MonkeyPatch, capacity: int,
) -> None:
    """BF16 can move logits to global scratch; INT8 still has its own bound."""
    runner, runner_allocator, _ = _fake_allocated_runner(
        monkeypatch,
        config=_wide_config(),
        capacity=capacity,
        kv_storage="bf16",
    )
    try:
        assert runner.kv_storage_resolved == "bf16"
        assert runner_allocator.allocated
    finally:
        runner.close()


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
class _FakeAttentionRuntime:
    """Stand-in for the HIP runtime that :class:`Gemma4AttentionScratch` asks for.

    The scratch is handed a runtime on every ``buffer`` call and asks it for
    three things: the current device, an allocation, and a stream
    synchronization. The buffers are keyed by stream, so the events recorded
    here are what show whether two layers shared one allocation and whether
    cleanup synchronized before freeing.
    """

    def __init__(self, device: int = 0) -> None:
        self.device = device
        self.next_ptr = 0x90000
        self.live: dict[int, int] = {}
        self.events: list[tuple[str, int]] = []

    def current_device(self) -> int:
        return self.device

    def malloc(self, nbytes: int) -> int:
        self.next_ptr += 0x1000
        self.live[self.next_ptr] = nbytes
        self.events.append(("malloc", self.next_ptr))
        return self.next_ptr

    def free(self, ptr: int) -> None:
        self.events.append(("free", ptr))
        del self.live[ptr]

    def stream_synchronize(self, stream: int) -> None:
        self.events.append(("sync", stream))


def test_runner_layers_share_one_attention_workspace_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One transient workspace serves every layer of a runner.

    A layer's attention scratch is temporary: it is live only while that
    layer's launch chain runs. Giving each layer its own owner retains one
    growth chain per layer, which is what makes a deep prefill's workspace
    scale with the layer count rather than with the widest request.
    """

    runner, _ = _make_runner(monkeypatch, config=_config(num_layers=4))
    try:
        owners = [scratch.attention for scratch in runner._scratches]
        assert len(owners) == 4
        assert all(owner is owners[0] for owner in owners)
    finally:
        runner.close()


def test_separate_runners_keep_separate_attention_workspace_owners(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sharing is per runner; a second runner is not handed the first's owner."""

    first, _ = _make_runner(monkeypatch, config=_config(num_layers=2))
    second, _ = _make_runner(monkeypatch, config=_config(num_layers=2))
    try:
        assert first._scratches[0].attention is first._scratches[1].attention
        assert second._scratches[0].attention is second._scratches[1].attention
        assert first._scratches[0].attention is not second._scratches[0].attention
    finally:
        first.close()
        second.close()


def test_shared_attention_owner_reuses_one_buffer_per_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every layer's same-stream request lands on one allocation.

    The layers are submitted serially on one stream, so a request made through
    any layer's scratch must find the allocation the previous layer's kernels
    are queued behind. A request on another stream must not: the scratch keys
    its buffers by stream, because a kernel queued on stream 0 may still be
    reading the workspace.
    """

    runner, _ = _make_runner(monkeypatch, config=_config(num_layers=4))
    runtime = _FakeAttentionRuntime()
    try:
        owners = [scratch.attention for scratch in runner._scratches]
        first = owners[0].buffer(4096, stream=0, runtime=runtime)
        for owner in owners[1:]:
            assert owner.buffer(4096, stream=0, runtime=runtime) is first
        # Per-layer owners would have left four live allocations here.
        assert runtime.live == {first.ptr: first.nbytes}

        # A wider request grows the one shared allocation, and every layer sees
        # the growth because they share the owner.
        grown = owners[-1].buffer(8192, stream=0, runtime=runtime)
        assert grown is not first
        assert grown.nbytes >= 8192
        assert owners[0].buffer(1, stream=0, runtime=runtime) is grown

        other = owners[0].buffer(4096, stream=1, runtime=runtime)
        assert other is not grown
        assert other.ptr != grown.ptr
        # Two streams hold two live workspaces; the replaced same-stream
        # allocation is retained rather than freed under a queued kernel.
        assert len(runtime.live) == 3
    finally:
        runner.close()


def test_close_synchronizes_then_frees_the_shared_workspace_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cleanup closes the shared owner once, after the stream is quiescent.

    Each layer scratch still closes the workspace it was handed, so a shared
    owner is closed once per layer; the close has to be idempotent and it has
    to synchronize the used streams before freeing, or a queued kernel would
    read a released allocation. The runner's own buffers must still be released
    exactly once as well.
    """

    runner, allocator = _make_runner(monkeypatch, config=_config(num_layers=4))
    runtime = _FakeAttentionRuntime()
    try:
        shared = runner._scratches[0].attention
        assert all(scratch.attention is shared for scratch in runner._scratches)
        buffer = shared.buffer(256, stream=0, runtime=runtime)
        runner.close()
        runner.close()

        assert runtime.live == {}
        frees = [event for event in runtime.events if event[0] == "free"]
        assert frees == [("free", buffer.ptr)]
        first_free = runtime.events.index(frees[0])
        assert ("sync", 0) in runtime.events[:first_free]
        assert [event for event in runtime.events if event[0] == "sync"] == [("sync", 0)]
        assert len(allocator.freed) == len(allocator.allocated)
    finally:
        runner.close()


def test_constructor_failure_closes_the_shared_workspace_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A live shared workspace is released once when construction fails.

    The failure lands while the second layer's KV is allocated, so two layer
    scratches already exist and both point at the same owner. Cleanup runs
    ``free`` over both, and the workspace must be synchronized and released
    exactly once across them.
    """

    runtime = _FakeAttentionRuntime()
    primed: list[int] = []
    owners: list[object] = []
    original_init = gemma4_module.Gemma4LayerScratch.__init__

    allocator = _FakeAllocator()

    def priming_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        owners.append(self.attention)
        if not primed:
            # Give the shared owner a live allocation before the failure, so
            # the cleanup path has something to release.
            primed.append(self.attention.buffer(256, stream=0, runtime=runtime).ptr)
        elif allocator.fail_at is None:
            # Fail on the next allocation after the second layer's scratch
            # exists, which is that layer's own KV. The runner's pre-layer
            # allocation count is not part of this test's contract, so the
            # failure point is derived rather than hardcoded.
            allocator.fail_at = allocator.calls + 1

    monkeypatch.setattr(gemma4_module.Gemma4LayerScratch, "__init__", priming_init)
    monkeypatch.setattr(gemma4_module, "malloc", allocator.malloc)
    monkeypatch.setattr(gemma4_module, "free", allocator.free)

    with pytest.raises(MemoryError):
        Gemma4Runner(weights=_weights(_config(num_layers=4)), capacity=64, max_block=8)

    assert primed, "the shared workspace was never exercised"
    # Two layers were constructed before the failure, on one owner.
    assert len(owners) == 2
    assert owners[0] is owners[1]
    assert runtime.live == {}
    frees = [event for event in runtime.events if event[0] == "free"]
    assert frees == [("free", primed[0])]
    first_free = runtime.events.index(frees[0])
    assert ("sync", 0) in runtime.events[:first_free]
    assert len(allocator.freed) == len(allocator.allocated)


def test_block_forward_hands_every_layer_the_shared_attention_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The block body runs every layer on the one owner, on one stream.

    The constructor-level identity check says the scratches share an owner; this
    says the block body actually hands that owner to each layer's launch chain,
    with the default stream the sharing argument depends on. Every kernel the
    block body reaches is replaced, so the layers' own arithmetic never runs.
    """

    runner, _ = _make_runner(monkeypatch, config=_config(num_layers=3), max_block=8)
    seen: list[tuple[object, int]] = []

    def fake_layer(*args, scratch, **kwargs):
        seen.append((scratch.attention, int(kwargs.get("stream", 0))))

    monkeypatch.setattr(gemma4_module, "gemma4_layer_forward_bf16", fake_layer)
    monkeypatch.setattr(gemma4_module, "enqueue_host_to_device", lambda *a, **k: None)
    monkeypatch.setattr(gemma4_module, "copy_device_to_host", lambda *a, **k: None)
    monkeypatch.setattr(gemma4_module, "launch_gguf_embedding", lambda *a, **k: None)
    monkeypatch.setattr(gemma4_module, "gemma4_scale_bf16", lambda *a, **k: None)
    monkeypatch.setattr(gemma4_module, "gemma4_rmsnorm_f32w_bf16", lambda *a, **k: None)
    monkeypatch.setattr(gemma4_module, "launch_gguf_linear", lambda *a, **k: None)

    try:
        runner._forward_block_inner([1, 2, 3], apply_softcap=False)
    finally:
        runner.close()

    assert len(seen) == 3
    assert [stream for _, stream in seen] == [0, 0, 0]
    assert len({id(owner) for owner, _ in seen}) == 1


def test_shared_kv_view_reports_geometry_and_live_positions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The assistant head's read view, without a device.

    The head allocates no KV and attends against the backbone's last two layers,
    so what it needs from the runner is the two buffer addresses, the live
    position count, and the geometry to index them. This checks the view is built
    from the config and the caches rather than from constants, and that a layer
    index outside the model is refused instead of silently wrapping.
    """

    config = _config(num_layers=4)
    runner, _ = _make_runner(monkeypatch, config=config, capacity=64, max_block=8)
    try:
        assert runner.layer_count == 4
        # Nothing has run, so a reader sees zero live positions.
        assert runner.shared_kv(0).live == 0

        for index in range(4):
            view = runner.shared_kv(index)
            attention = config.attention[index]
            assert view.layer_index == index
            assert view.key_cache == runner._kv[index].key_cache
            assert view.value_cache == runner._kv[index].value_cache
            assert view.capacity == 64
            assert view.num_kv_heads == attention.num_kv_heads
            assert view.head_dim == attention.head_dim
            assert view.kv_width == attention.num_kv_heads * attention.head_dim
            assert view.live_bytes == 0

        # Each layer's view names its own buffers.
        assert len({runner.shared_kv(i).key_cache for i in range(4)}) == 4

        for bad in (4, -1, 99):
            with pytest.raises(IndexError):
                runner.shared_kv(bad)

        # A write offset is not reachable through the view: the head reads.
        assert not hasattr(runner.shared_kv(0), "write_offset")
    finally:
        runner.close()


def test_hidden_state_row_indexing_uses_the_last_forward_not_the_position(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Row indexing must not accumulate across forwards.

    A prefill of many rows followed by a decode of one leaves exactly one valid
    hidden row. Indexing against the accumulated position count would hand back
    stale hidden state from before the decode, which for the assistant head means
    a draft step conditioned on the wrong token's activations.

    ``_last_rows`` is set directly here because a real forward needs a device;
    the GPU-guarded runner test asserts a real prefill-then-decode through this
    same accessor.
    """

    runner, _ = _make_runner(monkeypatch, config=_config(num_layers=2), max_block=8)
    try:
        with pytest.raises(ValueError):
            runner.hidden_state()

        runner._last_rows = 3
        runner._position = 513  # accumulated, and deliberately not the row count
        rows = [runner.hidden_state(i) for i in range(3)]
        assert len({row.ptr for row in rows}) == 3
        # Rows are consecutive and one hidden width apart.
        width = int(runner.weights.config.hidden_size) * 2
        assert [row.ptr for row in rows] == [rows[0].ptr + i * width for i in range(3)]
        assert all(row.nbytes == width for row in rows)

        # -1 is the last row of the last forward, not the 513th position.
        assert runner.hidden_state(-1).ptr == rows[-1].ptr
        assert runner.hidden_state(-3).ptr == rows[0].ptr

        for bad in (3, -4, 512):
            with pytest.raises(IndexError):
                runner.hidden_state(bad)
    finally:
        runner.close()
