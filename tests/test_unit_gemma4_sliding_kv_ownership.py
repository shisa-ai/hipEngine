"""Bounded sliding-window KV ownership for Gemma 4.

Gemma 4 has 5 global and 25 sliding layers on the 26B-A4B artifact, and the
runner allocated ``capacity`` KV rows for **every** layer. A sliding layer only
ever reads ``window + rows - 1`` keys, so at long context the extra 25 bands are
dead weight: measured tier-1 probes put the KV slope at 225,281 B/token, which
predicts ~86 GB at the 262144-token top of the ladder and is why the 131200 and
262272 capacity probes fail with ``HIP error 2``.

These tests pin the ownership model that bounds a sliding layer's cache to the
band it can actually read -- plus one decode-graph bucket of history -- while
leaving a global layer the full context. They are CPU-only: the allocator is
faked, the device copy is a real array-to-array move, and no HIP kernel is
launched.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.core.memory import DeviceBuffer
from hipengine.kernels.cpu_reference.gemma4 import (
    Gemma4AttentionGeometry,
    Gemma4RopeConfig,
    Gemma4TextConfig,
)
from hipengine.runtime import gemma4 as gemma4_module
from hipengine.runtime.gemma4 import (
    GEMMA4_SLIDING_BUCKET_WIDTH,
    Gemma4Runner,
    _resident_bytes,
    gemma4_layer_key_count,
    gemma4_sliding_cache_base,
    gemma4_sliding_cache_rows,
)

# The 26B-A4B geometry the capacity probes measured: 25 sliding layers of
# 8 KV heads x 256 and 5 global layers of 2 KV heads x 512, window 1024.
REAL_SLIDING_KV_BYTES = 8 * 256 * 2 * 2  # K+V, BF16
REAL_GLOBAL_KV_BYTES = 2 * 512 * 2 * 2
REAL_WINDOW = 1024
BUCKET = GEMMA4_SLIDING_BUCKET_WIDTH  # 64


class _FakeAllocator:
    """Deterministic malloc/free pair that records every call."""

    def __init__(self) -> None:
        self.next_ptr = 0x1000
        self.allocated: list[DeviceBuffer] = []
        self.freed: list[DeviceBuffer] = []

    def malloc(self, nbytes: int) -> DeviceBuffer:
        buffer = DeviceBuffer(ptr=self.next_ptr, nbytes=nbytes)
        self.next_ptr += nbytes + 0x1000
        self.allocated.append(buffer)
        return buffer

    def free(self, buffer: DeviceBuffer) -> None:
        self.freed.append(buffer)


class _RecordingRuntime:
    """A HIP runtime stand-in that records async D2D copies."""

    def __init__(self) -> None:
        self.copies: list[tuple[int, int, int, int, int]] = []

    def memcpy_async(self, dst, src, nbytes, kind, stream) -> None:
        self.copies.append((int(dst), int(src), int(nbytes), int(kind), int(stream)))


class _ArrayAllocator:
    """A malloc/free pair that backs every buffer with a numpy byte array.

    This is what makes a rebase observable: the compaction copies real bytes, so
    the slot a launch reads can be checked against the absolute position it must
    hold.
    """

    def __init__(self) -> None:
        self.next_ptr = 0x100000
        self.buffers: dict[int, np.ndarray] = {}

    def malloc(self, nbytes: int) -> DeviceBuffer:
        buffer = DeviceBuffer(ptr=self.next_ptr, nbytes=nbytes)
        self.buffers[self.next_ptr] = np.zeros(nbytes, dtype=np.uint8)
        self.next_ptr += nbytes + 0x1000
        return buffer

    def free(self, buffer: DeviceBuffer) -> None:
        self.buffers.pop(int(buffer.ptr), None)

    def array_at(self, ptr: int) -> tuple[np.ndarray, int]:
        for base, array in self.buffers.items():
            if base <= ptr < base + array.nbytes:
                return array, ptr - base
        raise KeyError(f"no allocation contains {ptr:#x}")


class _ArrayRuntime:
    """A HIP runtime stand-in whose D2D copies move real bytes."""

    def __init__(self, allocator: _ArrayAllocator) -> None:
        self.allocator = allocator
        self.copies: list[tuple[int, int, int]] = []

    def memcpy_async(self, dst, src, nbytes, kind, stream) -> None:
        self.copies.append((int(dst), int(src), int(nbytes)))
        d, d_off = self.allocator.array_at(int(dst))
        s, s_off = self.allocator.array_at(int(src))
        d[d_off : d_off + nbytes] = s[s_off : s_off + nbytes]


def _sliding_geometry(*, window: int = REAL_WINDOW) -> Gemma4AttentionGeometry:
    return Gemma4AttentionGeometry(
        layer_type="sliding_attention",
        num_heads=16,
        num_kv_heads=8,
        head_dim=256,
        rope=Gemma4RopeConfig(rope_theta=10_000.0, head_dim=256, rope_angles=4),
        sliding_window=window,
        k_eq_v=True,
    )


def _global_geometry() -> Gemma4AttentionGeometry:
    return Gemma4AttentionGeometry(
        layer_type="full_attention",
        num_heads=16,
        num_kv_heads=2,
        head_dim=512,
        rope=Gemma4RopeConfig(rope_theta=1_000_000.0, head_dim=512, rope_angles=8),
        sliding_window=None,
        k_eq_v=True,
    )


def _real_config(*, layers: int = 30) -> Gemma4TextConfig:
    """The probe's layer split: every 6th layer global, the last one forced global."""

    attention = tuple(
        _global_geometry() if (index + 1) % 6 == 0 else _sliding_geometry()
        for index in range(layers)
    )
    return Gemma4TextConfig(
        hidden_size=2816,
        intermediate_size=2112,
        moe_intermediate_size=704,
        num_experts=128,
        top_k_experts=8,
        rms_norm_eps=1e-6,
        attention=attention,
        vocab_size=262144,
    )


def _tiny_config(*, window: int = 64, num_layers: int = 2) -> Gemma4TextConfig:
    """A cheap config with a bounded sliding layer and a global one.

    Layer 0 is sliding (window ``window``) and layer 1 is global, so one runner
    carries both ownership shapes without touching a device.
    """

    attention = (
        Gemma4AttentionGeometry(
            layer_type="sliding_attention",
            num_heads=4,
            num_kv_heads=2,
            head_dim=8,
            rope=Gemma4RopeConfig(rope_theta=10_000.0, head_dim=8, rope_angles=4),
            sliding_window=window,
            k_eq_v=False,
        ),
        Gemma4AttentionGeometry(
            layer_type="full_attention",
            num_heads=4,
            num_kv_heads=2,
            head_dim=8,
            rope=Gemma4RopeConfig(rope_theta=10_000.0, head_dim=8, rope_angles=4),
            sliding_window=None,
            k_eq_v=False,
        ),
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


def _global_only_config(*, num_layers: int = 2) -> Gemma4TextConfig:
    """A cheap all-global config, the unbounded comparison for a mixed one."""

    return Gemma4TextConfig(
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=8,
        num_experts=4,
        top_k_experts=2,
        rms_norm_eps=1e-6,
        attention=tuple(
            Gemma4AttentionGeometry(
                layer_type="full_attention",
                num_heads=4,
                num_kv_heads=2,
                head_dim=8,
                rope=Gemma4RopeConfig(rope_theta=10_000.0, head_dim=8, rope_angles=4),
                sliding_window=None,
                k_eq_v=False,
            )
            for _ in range(num_layers)
        ),
        vocab_size=64,
    )


def _weights(config: Gemma4TextConfig) -> SimpleNamespace:
    return SimpleNamespace(
        config=config,
        layers=tuple(SimpleNamespace() for _ in range(len(config.attention))),
        dense_intermediate=(),
        final_norm=SimpleNamespace(buffer=SimpleNamespace(ptr=0)),
        embed_tokens=SimpleNamespace(),
        lm_head=None,
    )


def _make_runner(
    monkeypatch: pytest.MonkeyPatch,
    *,
    config: Gemma4TextConfig,
    capacity: int,
    max_block: int,
    allocator=None,
    reserve: int = BUCKET,
) -> tuple[Gemma4Runner, object]:
    allocator = allocator if allocator is not None else _FakeAllocator()
    monkeypatch.setattr(gemma4_module, "malloc", allocator.malloc)
    monkeypatch.setattr(gemma4_module, "free", allocator.free)
    runner = Gemma4Runner(
        weights=_weights(config),
        capacity=capacity,
        max_block=max_block,
        sliding_reserve=reserve,
    )
    return runner, allocator


def _install_array_runtime(
    monkeypatch: pytest.MonkeyPatch, allocator: _ArrayAllocator
) -> _ArrayRuntime:
    runtime = _ArrayRuntime(allocator)
    import hipengine.core.hip as hip

    monkeypatch.setattr(hip, "get_hip_runtime", lambda: runtime)
    monkeypatch.setattr(gemma4_module, "get_hip_runtime", lambda: runtime, raising=False)
    return runtime


# --- the read band and the cache band -------------------------------------


def test_a_sliding_layer_reads_only_its_window_band() -> None:
    """The span attention presents is ``window + rows - 1``, not ``capacity``."""

    attention = _sliding_geometry()
    assert gemma4_layer_key_count(attention, 262_272, rows=1024) == REAL_WINDOW + 1024 - 1


def test_a_sliding_layer_owns_the_read_band_plus_the_graph_reserve() -> None:
    """The cache is the read band plus one bucket below and one above."""

    attention = _sliding_geometry()
    rows = gemma4_sliding_cache_rows(attention, 262_272, rows=1024, reserve=BUCKET)
    assert rows == REAL_WINDOW - 1 + BUCKET + max(1024, BUCKET)
    # The reserve is small next to the context and does not push past it.
    assert rows == 2111
    assert gemma4_sliding_cache_rows(attention, 64, rows=1024, reserve=BUCKET) == 64


def test_a_global_layer_owns_the_whole_context() -> None:
    """A windowless layer reads every position, so nothing is bounded away."""

    assert gemma4_sliding_cache_rows(_global_geometry(), 262_272, rows=1024) == 262_272


def test_the_cache_band_grows_with_the_block_not_just_the_window() -> None:
    """A wider block keeps a wider band, and the reserve only pads it."""

    attention = _sliding_geometry(window=64)
    small = gemma4_sliding_cache_rows(attention, 4096, rows=8, reserve=BUCKET)
    large = gemma4_sliding_cache_rows(attention, 4096, rows=512, reserve=BUCKET)
    assert small == 64 - 1 + BUCKET + BUCKET
    assert large == 64 - 1 + BUCKET + 512


# --- the resident formula -------------------------------------------------


def test_resident_bytes_charges_sliding_layers_their_band_and_staging() -> None:
    """The KV term follows each layer's band; the staging plane is counted once."""

    config = _real_config()
    capacity, block = 262_272, 1024
    total = _resident_bytes(
        block=block,
        capacity=capacity,
        hidden=config.hidden_size,
        config=config,
        dense_widths=[2112] * len(config.attention),
    )
    sliding_rows = gemma4_sliding_cache_rows(_sliding_geometry(), capacity, rows=block)
    kv_term = 25 * sliding_rows * REAL_SLIDING_KV_BYTES + 5 * capacity * REAL_GLOBAL_KV_BYTES
    # One staging plane for the widest bounded sliding layer.
    staging = (REAL_WINDOW - 1 + BUCKET) * (8 * 256) * 2
    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_layer import Gemma4LayerScratch

    scratch = 0
    for index, attention in enumerate(config.attention):
        scratch += Gemma4LayerScratch(
            tokens=block,
            hidden_size=config.hidden_size,
            dense_intermediate=2112,
            geometry=gemma4_module._layer_geometry(attention),
            num_experts=config.num_experts,
            top_k=config.top_k_experts,
            expert_intermediate=config.moe_intermediate_size,
        ).resident_bytes()
    runner_buffers = (
        block * 8 + block * config.hidden_size * 2 + config.hidden_size * 2
        + int(config.vocab_size) * 4
    )
    assert total == runner_buffers + scratch + kv_term + staging


def test_the_bounded_kv_term_is_far_below_the_full_context_term() -> None:
    """The whole point: at 262144 the KV term drops from ~59 GB to ~5.4 GB."""

    capacity, block = 262_272, 1024
    sliding_rows = gemma4_sliding_cache_rows(_sliding_geometry(), capacity, rows=block)
    bounded = (
        25 * sliding_rows * REAL_SLIDING_KV_BYTES + 5 * capacity * REAL_GLOBAL_KV_BYTES
    )
    unbounded = capacity * (25 * REAL_SLIDING_KV_BYTES + 5 * REAL_GLOBAL_KV_BYTES)
    assert bounded < unbounded / 8
    # The bounded total matches llama.cpp's iswa split (5.37 GB + 0.42 GB),
    # rounded up by the one-bucket history reserve.
    assert bounded == 25 * 2111 * 8192 + 5 * 262_272 * 4096


# --- the rebase planner ---------------------------------------------------


def test_the_planner_leaves_a_block_that_fits_alone() -> None:
    """No rebase while the block and the graph bucket land inside the cache."""

    assert (
        gemma4_sliding_cache_base(
            base=0, capacity=2111, window=1024, start=0, rows=1024, reserve=BUCKET
        )
        == 0
    )
    assert (
        gemma4_sliding_cache_base(
            base=1, capacity=2111, window=1024, start=1024, rows=1024, reserve=BUCKET
        )
        == 1
    )


def test_the_planner_rebases_one_bucket_below_the_first_live_key() -> None:
    """A block that overflows keeps a bucket of history below its first live key."""

    # start=1024, rows=1024, base=0, capacity=2111 -> 2048 > 2111 is false, so
    # this fits; force an overflow at start=2048.
    assert (
        gemma4_sliding_cache_base(
            base=0, capacity=2111, window=1024, start=2048, rows=1024, reserve=BUCKET
        )
        == 2048 - 1024 + 1 - BUCKET
    )
    assert (
        gemma4_sliding_cache_base(
            base=1, capacity=2111, window=1024, start=4096, rows=1024, reserve=BUCKET
        )
        == 4096 - 1024 + 1 - BUCKET
    )


def test_the_planner_never_moves_the_base_backwards() -> None:
    """Rebasing is monotone, so a retained position is never overwritten."""

    capacity, window, reserve = 2111, 1024, BUCKET
    base = 0
    previous = 0
    for start in range(0, 12_000):
        base = gemma4_sliding_cache_base(
            base=base, capacity=capacity, window=window, start=start, rows=1,
            reserve=reserve,
        )
        assert base >= previous
        previous = base


@pytest.mark.parametrize(
    "window,max_block,reserve",
    [
        (1024, 1024, 64),  # the production ratio
        (64, 64, 64),  # 16x-scaled toy: the 2048 boundary lands at 128
        (16, 8, 64),  # reserve wider than the block
        (1024, 64, 64),
        (256, 1024, 64),
        (1024, 1024, 128),
        (8, 8, 64),
    ],
)
def test_the_reserve_keeps_the_frozen_bucket_inside_the_band(
    window: int, max_block: int, reserve: int
) -> None:
    """At every rebase the graph's frozen origin and end stay in the cache.

    This is the regression the reviewer traced: a rebased base above the frozen
    bucket origin forced the capture to clamp its first key, which changed the
    presented key count and the flash/split partition across the rebase. The
    reserve must keep ``base <= bucket.start - window`` and
    ``bucket.end - base <= band`` for every position and bucket shape.
    """

    from hipengine.runtime.gemma4_decode_graph import _Bucket

    capacity = window + max_block + 4 * reserve + 8
    band = gemma4_sliding_cache_rows(
        _sliding_geometry(window=window), capacity, rows=max_block, reserve=reserve
    )
    assert band < capacity  # the layer is actually bounded
    base = 0
    for position in range(0, 6 * max_block + 6 * window + 8 * reserve):
        base = gemma4_sliding_cache_base(
            base=base, capacity=band, window=window, start=position, rows=1,
            reserve=reserve,
        )
        bucket = _Bucket.for_position(position, width=reserve)
        origin = max(0, bucket.start - window)
        assert base <= origin, (position, base, origin)
        assert bucket.end - base <= band, (position, bucket.end - base, band)


# --- runner allocation ----------------------------------------------------


def test_a_sliding_layer_cache_is_bounded_to_its_band(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """capacity 256, window 64, reserve 64 -> a 191-row sliding cache, 256 global."""

    config = _tiny_config(window=64)
    runner, allocator = _make_runner(
        monkeypatch, config=config, capacity=256, max_block=8
    )
    try:
        assert runner._kv_capacity == [191, 256]
        assert runner._kv_base == [0, 0]
        assert runner._bounded_sliding is True
        kv_buffers = [
            buffer
            for buffer in allocator.allocated
            if buffer.nbytes in (191 * 16 * 2, 256 * 16 * 2)
        ]
        assert sorted(buffer.nbytes for buffer in kv_buffers) == sorted(
            [191 * 16 * 2, 191 * 16 * 2, 256 * 16 * 2, 256 * 16 * 2]
        )
    finally:
        runner.close()


def test_the_staging_plane_is_allocated_once_for_the_widest_layer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One plane, sized for the widest bounded layer, not one per rebase."""

    config = _tiny_config(window=64)
    runner, allocator = _make_runner(
        monkeypatch, config=config, capacity=256, max_block=8
    )
    runtime = _RecordingRuntime()
    import hipengine.core.hip as hip

    monkeypatch.setattr(hip, "get_hip_runtime", lambda: runtime)
    try:
        # window 64, reserve 64 -> retained tail 127 rows of kv_width 16 * 2 B.
        expected = (64 - 1 + BUCKET) * 16 * 2
        assert runner._compact_staging is not None
        assert runner._compact_staging.nbytes == expected
        staging_ptrs = {
            buffer.ptr
            for buffer in allocator.allocated
            if buffer.nbytes == expected
        }
        assert len(staging_ptrs) == 1
        # Rebase twice; the plane is reused, never reallocated or retained.
        runner._prepare_sliding_caches(128, 8, stream=0)
        runner._prepare_sliding_caches(256, 8, stream=0)
        assert runner._compact_staging.ptr in staging_ptrs
        after = [
            buffer for buffer in allocator.allocated if buffer.nbytes == expected
        ]
        assert len(after) == 1
    finally:
        runner.close()


def test_a_runner_without_sliding_layers_is_unbounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A windowless config allocates the full context everywhere and is untouched."""

    runner, _ = _make_runner(
        monkeypatch, config=_global_only_config(), capacity=64, max_block=8
    )
    try:
        assert runner._kv_capacity == [64, 64]
        assert runner._bounded_sliding is False
        assert runner._compact_staging is None
    finally:
        runner.close()


def test_the_fit_check_is_charged_the_runners_own_reserve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The block fit must see the same reserve the allocation below uses.

    A non-default reserve changes the sliding band and the staging plane, so a
    fit that charged the default would pick a block the allocation then
    overruns. The chosen block is the one the fit returns, so this pins the
    forwarded value rather than only the allocation.
    """

    seen: dict[str, object] = {}

    def fake_fit(block: int, **kwargs: object) -> int:
        seen.update(kwargs)
        return block

    monkeypatch.setattr(gemma4_module, "_fit_prefill_block", fake_fit)
    runner, _ = _make_runner(
        monkeypatch, config=_tiny_config(window=16), capacity=2000, max_block=0,
        reserve=7,
    )
    try:
        assert seen["sliding_reserve"] == 7
        assert runner.sliding_reserve == 7
        # The auto block is min(capacity, DEFAULT_PREFILL_BLOCK).
        assert runner.max_block == 1024
        # Allocation used the same reserve: window - 1 + reserve + max(block, reserve).
        assert runner._kv_capacity[0] == 16 - 1 + 7 + 1024
        assert runner._kv_capacity[1] == 2000
        assert runner._compact_staging.nbytes == (16 - 1 + 7) * 16 * 2
    finally:
        runner.close()


def test_the_resident_formula_follows_a_nondefault_reserve() -> None:
    """A larger reserve charges more KV rows and a larger staging plane."""

    config = _tiny_config(window=16)
    dense = [config.intermediate_size] * len(config.attention)
    kwargs = dict(block=8, capacity=2000, hidden=16, config=config, dense_widths=dense)
    small = _resident_bytes(**kwargs, sliding_reserve=7)
    default = _resident_bytes(**kwargs, sliding_reserve=BUCKET)
    assert small < default
    # Layer 0 sliding: 16 - 1 + reserve + max(8, reserve) rows of kv_width 16.
    small_rows = 16 - 1 + 7 + max(8, 7)
    default_rows = 16 - 1 + BUCKET + max(8, BUCKET)
    assert small_rows == 30 and default_rows == 143
    # KV is charged K and V (kv_width 16 x 2 bytes x 2 planes per row); the
    # staging plane is one plane (16 x 2 bytes per row).
    assert default - small == (
        (default_rows - small_rows) * 16 * 2 * 2
        + ((16 - 1 + BUCKET) - (16 - 1 + 7)) * 16 * 2
    )


def test_a_negative_sliding_reserve_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A negative reserve is a caller error, named rather than silently used."""

    allocator = _FakeAllocator()
    monkeypatch.setattr(gemma4_module, "malloc", allocator.malloc)
    monkeypatch.setattr(gemma4_module, "free", allocator.free)
    with pytest.raises(ValueError, match="sliding_reserve must be non-negative"):
        Gemma4Runner(
            weights=_weights(_tiny_config(window=16)),
            capacity=256,
            max_block=8,
            sliding_reserve=-1,
        )


# --- shared-KV exports ----------------------------------------------------


def test_shared_kv_reports_the_bounded_capacity_and_base(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A read view names the rows the buffer actually owns and where they start."""

    config = _tiny_config(window=64)
    runner, _ = _make_runner(monkeypatch, config=config, capacity=256, max_block=8)
    try:
        view = runner.shared_kv(0)
        assert view.capacity == 191
        assert view.base == 0
        assert view.live == 0
        assert view.key_cache == runner._kv[0].key_cache
        assert view.value_cache == runner._kv[0].value_cache
        global_view = runner.shared_kv(1)
        assert global_view.capacity == 256
        assert global_view.base == 0
    finally:
        runner.close()


def test_shared_kv_reports_a_rebased_sliding_view(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rebased view names its base and counts live rows in the cache's frame.

    Slot ``s`` holds absolute position ``base + s``, so ``live`` is the row
    count above the base and the assistant head offsets its mask and
    ``row_offset`` by the same base. The live count must stay inside the band.
    """

    config = _tiny_config(window=64)
    runner, _ = _make_runner(monkeypatch, config=config, capacity=256, max_block=8)
    try:
        runner._position = 150
        runner._kv_base[0] = 7
        view = runner.shared_kv(0)
        assert view.base == 7
        assert view.live == 150 - 7
        assert view.live <= view.capacity
        # The global layer has base 0 and is unaffected.
        assert runner.shared_kv(1).base == 0
        assert runner.shared_kv(1).live == 150
    finally:
        runner.close()


# --- lifecycle ------------------------------------------------------------


def test_reset_clears_the_base_and_rewind_keeps_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rewind drops positions but not the base a retained position needs."""

    config = _tiny_config(window=64)
    runner, _ = _make_runner(monkeypatch, config=config, capacity=256, max_block=8)
    try:
        runner._kv_base = [5, 0]
        runner._position = 100
        runner.rewind(90)
        assert runner._kv_base == [5, 0]
        assert runner._kv[0].write_offset == 90
        runner.reset()
        assert runner._kv_base == [0, 0]
        assert runner._kv[0].write_offset == 0
        assert runner._kv[0].capacity == 191
    finally:
        runner.close()


def test_reset_and_close_leave_the_staging_plane_consistent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """reset keeps the plane; close frees it exactly once."""

    config = _tiny_config(window=64)
    runner, allocator = _make_runner(
        monkeypatch, config=config, capacity=256, max_block=8
    )
    staging = runner._compact_staging
    assert staging is not None
    runner._kv_base = [9, 0]
    runner.reset()
    assert runner._compact_staging is staging
    assert allocator.freed == []
    runner.close()
    assert [buffer.ptr for buffer in allocator.freed].count(staging.ptr) == 1
    # A second close is idempotent.
    runner.close()
    assert [buffer.ptr for buffer in allocator.freed].count(staging.ptr) == 1


# --- compaction and launch translation ------------------------------------


def _dummy_buffer(ptr: int) -> SimpleNamespace:
    return SimpleNamespace(ptr=ptr)


def test_compaction_shifts_the_retained_band_to_slot_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real bytes move: after a rebase, slot ``s`` holds absolute ``base + s``."""

    config = _tiny_config(window=64)
    allocator = _ArrayAllocator()
    runner, _ = _make_runner(
        monkeypatch, config=config, capacity=256, max_block=8, allocator=allocator
    )
    _install_array_runtime(monkeypatch, allocator)
    kv_width = 2 * 8
    try:
        key = allocator.buffers[int(runner._kv[0].key_cache)].view(np.uint16).reshape(
            -1, kv_width
        )
        value = allocator.buffers[int(runner._kv[0].value_cache)].view(np.uint16).reshape(
            -1, kv_width
        )
        for position in range(128):
            key[position] = position
            value[position] = position
        # window 64, reserve 64, band 191: start 127 fits (127 + 64 <= 191) but
        # start 128 overflows, rebasing to max(0, 128 - 64 + 1 - 64) = 1 and
        # moving retained [1, 128) down to [0, 127).
        runner._prepare_sliding_caches(127, 1, stream=0)
        assert runner._kv_base == [0, 0]
        runner._prepare_sliding_caches(128, 1, stream=0)
        assert runner._kv_base == [1, 0]
        for slot in range(127):
            assert key[slot, 0] == 1 + slot
            assert value[slot, 0] == 1 + slot
        assert runner._kv_base[1] == 0
    finally:
        runner.close()


def test_a_block_that_fits_copies_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No device work when the block and bucket land inside the cache."""

    config = _tiny_config(window=64)
    runner, _ = _make_runner(monkeypatch, config=config, capacity=256, max_block=8)
    runtime = _RecordingRuntime()
    import hipengine.core.hip as hip

    monkeypatch.setattr(hip, "get_hip_runtime", lambda: runtime)
    try:
        runner._prepare_sliding_caches(0, 8, stream=0)
        assert runtime.copies == []
        assert runner._kv_base == [0, 0]
    finally:
        runner.close()


def test_launch_translates_offsets_for_a_bounded_layer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The layer sees cache-relative offsets; the global layer is unchanged."""

    config = _tiny_config(window=64)
    runner, _ = _make_runner(monkeypatch, config=config, capacity=256, max_block=8)
    captured: list[dict] = []

    monkeypatch.setattr(gemma4_module, "launch_gguf_embedding", lambda *a, **k: None)
    monkeypatch.setattr(gemma4_module, "gemma4_scale_bf16", lambda *a, **k: None)
    monkeypatch.setattr(
        gemma4_module, "gemma4_layer_forward_bf16", lambda *a, **k: captured.append(k)
    )
    try:
        runner._kv_base = [5, 0]
        runner._position = 100
        rope = config.attention[0].rope
        tables = {rope: (_dummy_buffer(1), _dummy_buffer(2))}
        masks = {64: _dummy_buffer(3), None: _dummy_buffer(4)}
        runner._launch_block(
            [1, 2, 3],
            tables=tables,
            masks=masks,
            kv_write_offset=100,
            key_begin_at=None,
            needs_logits=False,
        )
        assert len(captured) == 2
        sliding, global_layer = captured
        assert sliding["kv"].write_offset == 100 - 5
        # first_live = 100 - 64 + 1 = 37; translated = 37 - 5 = 32.
        assert sliding["key_begin"] == 32
        assert global_layer["kv"].write_offset == 100
        assert global_layer["key_begin"] == 0
    finally:
        runner.close()


def test_launch_translates_a_frozen_offset_for_a_bounded_layer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A decode-graph frozen offset is cache-relative after the base translation.

    The frozen offset is the bucket end, which the history reserve keeps inside
    the band, so this is a reachable launch and not an impossible fixture.
    """

    config = _tiny_config(window=64)
    runner, _ = _make_runner(monkeypatch, config=config, capacity=256, max_block=8)
    captured: list[dict] = []

    monkeypatch.setattr(gemma4_module, "launch_gguf_embedding", lambda *a, **k: None)
    monkeypatch.setattr(gemma4_module, "gemma4_scale_bf16", lambda *a, **k: None)
    monkeypatch.setattr(
        gemma4_module, "gemma4_layer_forward_bf16", lambda *a, **k: captured.append(k)
    )
    try:
        runner._kv_base = [7, 0]
        runner._position = 120
        # A capture's frozen shape: bucket [64, 128), so write bucket end - 1 and
        # the frozen origin 64 - 64 = 0.
        runner._launch_block(
            [1],
            tables={config.attention[0].rope: (_dummy_buffer(1), _dummy_buffer(2))},
            masks={64: _dummy_buffer(3), None: _dummy_buffer(4)},
            kv_write_offset=127,
            key_begin_at=lambda attention: 0,
            needs_logits=False,
        )
        assert len(captured) == 2
        sliding, global_layer = captured
        # The frozen slot 127 is cache-relative slot 127 - 7 for the rebased layer.
        assert sliding["kv"].write_offset == 127 - 7
        assert sliding["kv"].write_offset < runner._kv_capacity[0]
        # The frozen origin 0 is above the base, so the layer reads from slot 0.
        assert sliding["key_begin"] == 0
        assert global_layer["kv"].write_offset == 127
        assert global_layer["key_begin"] == 0
    finally:
        runner.close()


@pytest.mark.parametrize(
    "window,reserve",
    [(64, 64), (128, 64), (16, 64), (64, 128)],
)
def test_the_frozen_origin_survives_rebases_with_real_copies(
    monkeypatch: pytest.MonkeyPatch, window: int, reserve: int
) -> None:
    """Array-backed: the frozen origin's slot always holds its absolute position.

    This drives the real rebase planner and compaction for positions across
    several bucket boundaries and asserts, from the moved bytes, that the slot
    the decode graph reads for its frozen first key holds that key.
    """

    config = _tiny_config(window=window)
    allocator = _ArrayAllocator()
    capacity = window + 4 * reserve + 8 * 64
    runner, _ = _make_runner(
        monkeypatch,
        config=config,
        capacity=capacity,
        max_block=8,
        allocator=allocator,
        reserve=reserve,
    )
    _install_array_runtime(monkeypatch, allocator)
    kv_width = 2 * 8
    band = runner._kv_capacity[0]
    try:
        key = allocator.buffers[int(runner._kv[0].key_cache)].view(np.uint16).reshape(
            -1, kv_width
        )
        for position in range(capacity):
            runner._prepare_sliding_caches(position, 1, stream=0)
            base = runner._kv_base[0]
            # Simulate the decode append, then read the graph's frozen origin.
            key[position - base, 0] = position
            bucket_start = (position // reserve) * reserve
            origin = max(0, bucket_start - window)
            assert base <= origin, (position, base, origin)
            assert key[origin - base, 0] == origin, (position, base, origin)
            assert bucket_start + reserve - base <= band
    finally:
        runner.close()


# --- speculative rejection and rewind -------------------------------------


def test_speculative_partial_rejection_keeps_the_accepted_slots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Array-backed: rewinding to a prefix of the block keeps its KV rows."""

    config = _tiny_config(window=64)
    allocator = _ArrayAllocator()
    runner, _ = _make_runner(
        monkeypatch, config=config, capacity=256, max_block=8, allocator=allocator
    )
    _install_array_runtime(monkeypatch, allocator)
    kv_width = 2 * 8
    try:
        key = allocator.buffers[int(runner._kv[0].key_cache)].view(np.uint16).reshape(
            -1, kv_width
        )
        # Decode to position 100, then a speculative block of 4 rows.
        for position in range(104):
            runner._prepare_sliding_caches(position, 1, stream=0)
            base = runner._kv_base[0]
            key[position - base, 0] = position
        runner._position = 104
        base = runner._kv_base[0]
        # Partial rejection: accept the first two draft rows, rewind to 102.
        runner.rewind(102)
        assert runner._position == 102
        assert runner._kv_base[0] == base
        assert key[101 - base, 0] == 101
        assert runner._kv[0].write_offset == 102
        # The next append lands at slot 102 - base, overwriting the rejected row.
        key[102 - base, 0] = 102
        assert key[102 - base, 0] == 102
    finally:
        runner.close()


def test_speculative_full_rejection_keeps_the_base_and_slots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Array-backed: rewinding to the block start keeps the accepted history."""

    config = _tiny_config(window=64)
    allocator = _ArrayAllocator()
    runner, _ = _make_runner(
        monkeypatch, config=config, capacity=256, max_block=8, allocator=allocator
    )
    _install_array_runtime(monkeypatch, allocator)
    kv_width = 2 * 8
    try:
        key = allocator.buffers[int(runner._kv[0].key_cache)].view(np.uint16).reshape(
            -1, kv_width
        )
        for position in range(104):
            runner._prepare_sliding_caches(position, 1, stream=0)
            base = runner._kv_base[0]
            key[position - base, 0] = position
        runner._position = 104
        base = runner._kv_base[0]
        runner.rewind(100)
        assert runner._position == 100
        assert runner._kv_base[0] == base
        assert key[99 - base, 0] == 99
        assert runner._kv[0].write_offset == 100
    finally:
        runner.close()


def test_rewind_within_the_retained_band_keeps_the_base(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A speculative rewind to a retained position leaves the base alone."""

    config = _tiny_config(window=64)
    runner, _ = _make_runner(monkeypatch, config=config, capacity=256, max_block=8)
    try:
        runner._kv_base = [5, 0]
        runner._position = 100
        # first_live = max(0, 90 - 64 + 1) = 27 >= base 5.
        runner.rewind(90)
        assert runner._kv_base == [5, 0]
        assert runner._kv[0].write_offset == 90
        assert runner._position == 90
    finally:
        runner.close()


def test_rewind_below_the_retained_band_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An older rewind names the compacted history instead of shifting the window."""

    config = _tiny_config(window=64)
    runner, _ = _make_runner(monkeypatch, config=config, capacity=256, max_block=8)
    try:
        runner._kv_base = [5, 0]
        runner._position = 100
        # first_live = max(0, 67 - 64 + 1) = 4 < base 5: key 4 was compacted away.
        with pytest.raises(ValueError, match="compacted"):
            runner.rewind(67)
        assert runner._kv_base == [5, 0]
        assert runner._position == 100
    finally:
        runner.close()


def test_repeated_rebase_sequences_are_deterministic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reset and repeat a decode walk reproduces the base sequence exactly."""

    config = _tiny_config(window=64)
    runner, _ = _make_runner(monkeypatch, config=config, capacity=256, max_block=8)
    runtime = _RecordingRuntime()
    import hipengine.core.hip as hip

    monkeypatch.setattr(hip, "get_hip_runtime", lambda: runtime)

    def drive() -> list[int]:
        runner.reset()
        bases = []
        for start in range(0, 400, 7):
            runner._prepare_sliding_caches(start, 1, stream=0)
            bases.append(runner._kv_base[0])
        return bases

    try:
        first = drive()
        assert first == drive()
        assert first[-1] > first[0]
    finally:
        runner.close()


# --- assistant shared-KV mask translation ---------------------------------


def test_assistant_mask_indexes_absolute_positions_through_the_base() -> None:
    """The assistant mask keeps the same absolute keys whether or not it rebased."""

    from hipengine.runtime.gemma4_assistant import gemma4_assistant_keep_mask

    # 8 slots holding absolute positions 1000..1007, query at 1007, window 4:
    # kept when absolute <= 1007 and 1007 - absolute < 4, i.e. slots 4..7.
    rebased = gemma4_assistant_keep_mask(1007, 8, sliding_window=4, base=1000)
    assert rebased.tolist() == [[0, 0, 0, 0, 1, 1, 1, 1]]
    assert gemma4_assistant_keep_mask(7, 8, sliding_window=4).tolist() == [
        [0, 0, 0, 0, 1, 1, 1, 1]
    ]
    assert gemma4_assistant_keep_mask(1007, 8, sliding_window=None, base=1000).tolist() == [
        [1] * 8
    ]


def test_assistant_view_points_at_the_rebased_cache_and_mask_matches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """At a rebase the assistant reads the real cache pointers through the base."""

    from hipengine.runtime.gemma4_assistant import gemma4_assistant_keep_mask

    config = _tiny_config(window=64)
    runner, _ = _make_runner(monkeypatch, config=config, capacity=256, max_block=8)
    try:
        runner._position = 150
        runner._kv_base[0] = 80
        view = runner.shared_kv(0)
        assert view.key_cache == runner._kv[0].key_cache
        assert view.value_cache == runner._kv[0].value_cache
        assert view.base == 80
        assert view.live == 70
        # The head's slot-frame query position is ``position - base``.
        slot_position = 150 - view.base
        mask = gemma4_assistant_keep_mask(
            150, view.live, sliding_window=64, base=view.base
        )
        # Query at absolute 150, window 64: absolute 87..150 are kept, i.e.
        # slots 7..69 (0-based) of the 70 live rows.
        kept = [index for index, value in enumerate(mask[0]) if value]
        assert kept[0] == 7
        assert kept[-1] == 69
        assert slot_position == 70
    finally:
        runner.close()


# --- decode-graph bucket and retarget -------------------------------------


def test_a_context_end_bucket_stays_inside_every_cache_at_a_non_bucket_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The last bucket is cut at the context for a bounded runner too.

    ``_Bucket.for_position`` rounds the end up, so at capacity 200 (not a
    multiple of 64) position 199's bucket would end at 255. Every global layer
    keeps a full-context cache of exactly 200 rows indexed by absolute
    position, and a bounded runner keeps those global caches at the full
    context, so an uncut bucket would run 55 rows past the global K/V
    allocation -- and the zero-tail memset would write those 55 rows. The cut
    is at the physical context, never at the sliding band.
    """

    import hipengine.core.hip as hip

    from hipengine.runtime.gemma4_decode_graph import _Bucket, _context_end_bucket

    capacity = 200  # not a multiple of the bucket width
    position = capacity - 1
    raw = _Bucket.for_position(position)
    assert raw.end > capacity  # the uncut bucket overruns the context
    monkeypatch.setattr(hip, "get_hip_runtime", lambda: _RecordingRuntime())
    bounded, _ = _make_runner(
        monkeypatch,
        config=_tiny_config(window=16),
        capacity=capacity,
        max_block=8,
        reserve=BUCKET,
    )
    unbounded, _ = _make_runner(
        monkeypatch,
        config=_global_only_config(),
        capacity=capacity,
        max_block=8,
    )
    try:
        assert bounded._bounded_sliding is True
        assert unbounded._bounded_sliding is False
        # Layer 0 is sliding (bounded band), layer 1 is global (full context).
        assert bounded._kv_capacity[0] < capacity
        assert bounded._kv_capacity[1] == capacity
        assert unbounded._kv_capacity == [capacity, capacity]

        for runner in (bounded, unbounded):
            runner._prepare_sliding_caches(position, 1, stream=0)
            bucket = _context_end_bucket(position, runner.capacity)
            # The same cut for both runners, and it lands exactly at the context.
            assert bucket == _Bucket(raw.start, capacity)
            assert bucket.end == capacity
            frozen_write = bucket.end - 1
            for index in range(len(runner._kv)):
                base = int(runner._kv_base[index])
                rows = int(runner._kv_capacity[index])
                # The captured append slot and the frozen key extent are both
                # inside this layer's cache. Without the cut, the global layer's
                # frozen_write slot is 254 against a 200-row cache.
                assert 0 <= frozen_write - base < rows
                # The zero-tail range [position + 1, bucket.end) is inside too.
                first_pending = position + 1 - base
                end_slot = bucket.end - base
                assert 0 <= first_pending <= end_slot <= rows
    finally:
        bounded.close()
        unbounded.close()


def test_a_bounded_runner_cuts_the_bucket_at_the_context_not_the_band(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mid-context bucket is not cut at all, so the sliding shape is stable."""

    from hipengine.runtime.gemma4_decode_graph import _Bucket, _context_end_bucket

    capacity = 200
    bounded, _ = _make_runner(
        monkeypatch,
        config=_tiny_config(window=16),
        capacity=capacity,
        max_block=8,
        reserve=BUCKET,
    )
    try:
        # A position whose bucket already ends inside the context is untouched:
        # cutting it at the 143-row band would change the frozen attention shape
        # at every rebase.
        for position in (100, 150, 190):
            bucket = _context_end_bucket(position, capacity)
            assert bucket == _Bucket.for_position(position)
            assert bucket.end <= capacity
        # At position 190 the bucket ends at 192, past the 143-row sliding band,
        # and it is not cut: the cut is at the context, not the band.
        assert bounded._kv_capacity[0] == 143
        assert _context_end_bucket(190, capacity).end == 192
    finally:
        bounded.close()


def test_bucket_covers_the_position_without_a_cache_clamp() -> None:
    """The bucket is the plain grid window plus the tiled-route singleton cut."""

    from hipengine.runtime.gemma4_decode_graph import BUCKET_WIDTH, _Bucket

    assert BUCKET_WIDTH == BUCKET
    for position in range(0, 600):
        bucket = _Bucket.for_position(position)
        assert bucket.start <= position < bucket.end
        assert bucket.end - bucket.start <= BUCKET_WIDTH


def test_capture_refuses_a_runner_with_a_small_reserve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reserve below the bucket width would change the frozen shape; refuse."""

    from hipengine.runtime import gemma4_decode_graph as graph_module
    from hipengine.runtime.gemma4_decode_graph import _Bucket

    config = _tiny_config(window=64)
    runner, _ = _make_runner(
        monkeypatch, config=config, capacity=256, max_block=8, reserve=8
    )
    try:
        runner._position = 120
        runner._kv_base[0] = 49
        session = object.__new__(graph_module.Gemma4DecodeGraphSession)
        session._runner_ref = lambda: runner
        session._stream = 7
        session._retire = lambda: None
        with pytest.raises(RuntimeError, match="history reserve"):
            session._capture(
                _Bucket.for_position(120), {}, {}, ("k",), token=1
            )
    finally:
        runner.close()


class _CaptureRuntime:
    """A HIP runtime stand-in for the post-end capture steps."""

    def __init__(self, *, fail_instantiate: bool = False) -> None:
        self.fail_instantiate = fail_instantiate
        self.destroyed: list[int] = []
        self.instantiated = 0

    def graph_instantiate(self, graph):
        if self.fail_instantiate:
            raise RuntimeError("instantiate failed")
        self.instantiated += 1
        return 0xE

    def graph_destroy(self, graph):
        self.destroyed.append(int(graph))

    def graph_exec_destroy(self, exec_handle):
        self.destroyed.append(-int(exec_handle))


def _finalize_session(
    monkeypatch: pytest.MonkeyPatch,
    runtime: _CaptureRuntime,
    *,
    identify=None,
    zero=None,
):
    from hipengine.runtime import gemma4_decode_graph as graph_module
    from hipengine.runtime.gemma4_decode_graph import Gemma4DecodeGraphSession

    monkeypatch.setattr(graph_module, "get_hip_runtime", lambda: runtime)
    session = object.__new__(Gemma4DecodeGraphSession)
    session._graph = 0
    session._exec = 0
    session._key = None
    session._bucket = None
    session._appends = ()
    session._captures = 0
    session._identify_appends = identify or (lambda *a, **k: ())
    session._zero_pending_tail = zero or (lambda _: None)
    return session


def test_finalize_releases_the_graph_when_instantiate_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A post-end failure must not leak the captured graph."""

    from hipengine.runtime.gemma4_decode_graph import _Bucket

    runtime = _CaptureRuntime(fail_instantiate=True)
    session = _finalize_session(monkeypatch, runtime)
    with pytest.raises(RuntimeError, match="instantiate"):
        session._finalize_capture(
            0xBEEF, frozen_write=63, bucket=_Bucket(0, 64), key=("k",)
        )
    assert runtime.destroyed == [0xBEEF]
    assert session._exec == 0
    assert session._captures == 0


def test_finalize_releases_graph_and_exec_when_identify_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """instantiate succeeds, identify raises: both handles are released."""

    from hipengine.runtime.gemma4_decode_graph import _Bucket

    def identify(*args, **kwargs):
        raise RuntimeError("identify failed")

    runtime = _CaptureRuntime()
    session = _finalize_session(monkeypatch, runtime, identify=identify)
    with pytest.raises(RuntimeError, match="identify"):
        session._finalize_capture(
            0xBEEF, frozen_write=63, bucket=_Bucket(0, 64), key=("k",)
        )
    assert runtime.destroyed == [-0xE, 0xBEEF]
    assert session._exec == 0


def test_finalize_releases_graph_and_exec_when_zero_tail_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """instantiate and identify succeed, zero-tail raises: both are released."""

    from hipengine.runtime.gemma4_decode_graph import _Bucket

    def zero(_bucket):
        raise RuntimeError("zero tail failed")

    runtime = _CaptureRuntime()
    session = _finalize_session(monkeypatch, runtime, zero=zero)
    with pytest.raises(RuntimeError, match="zero tail"):
        session._finalize_capture(
            0xBEEF, frozen_write=63, bucket=_Bucket(0, 64), key=("k",)
        )
    assert runtime.destroyed == [-0xE, 0xBEEF]
    assert session._exec == 0


def test_finalize_adopts_the_graph_on_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No failure: the session adopts the graph and executable."""

    from hipengine.runtime.gemma4_decode_graph import _Bucket

    runtime = _CaptureRuntime()
    session = _finalize_session(monkeypatch, runtime)
    bucket = _Bucket(0, 64)
    session._finalize_capture(0xBEEF, frozen_write=63, bucket=bucket, key=("k",))
    assert runtime.destroyed == []
    assert session._graph == 0xBEEF
    assert session._exec == 0xE
    assert session._bucket == bucket
    assert session._captures == 1


def test_retarget_subtracts_the_layer_base(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A captured append's replay destination is ``position - base`` in slots."""

    from hipengine.runtime import gemma4_decode_graph as graph_module
    from hipengine.runtime.gemma4_decode_graph import (
        Gemma4DecodeGraphSession,
        _CaptureAppend,
    )

    calls: list[tuple[int, int, int]] = []

    class _RT:
        def graph_exec_memcpy_node_set_1d(
            self, exec_handle, node, *, dst, src, count, kind
        ) -> None:
            calls.append((int(node), int(dst), int(count)))

    monkeypatch.setattr(graph_module, "get_hip_runtime", lambda: _RT())
    session = object.__new__(Gemma4DecodeGraphSession)
    session._exec = 0x1
    session._appends = (
        _CaptureAppend(
            node=1, src=0x10, count=32, cache_base=0x1000, kv_width=16, base=5
        ),
        _CaptureAppend(
            node=2, src=0x20, count=32, cache_base=0x2000, kv_width=16, base=0
        ),
    )
    session._retarget_appends(20)
    assert calls[0][1] == 0x1000 + (20 - 5) * 16 * 2
    assert calls[1][1] == 0x2000 + 20 * 16 * 2
