"""Unit tests for the GGUF resident scratch sizing plan.

The plan is the single source of truth shared by ``_FullStackScratch.allocate``
and the resident capacity estimator, so the important property to pin down is
that the plan's arithmetic matches what the allocator actually allocates. These
tests use a fake device (no HIP, no model load) and compare the plan against the
real allocation path.
"""

from __future__ import annotations

from types import SimpleNamespace

from hipengine.core.memory import DeviceBuffer
from hipengine.generation import qwen35_gguf
from hipengine.kernels.policy import QWEN35_DENSE_H5120_GEOMETRY
from hipengine.loading.qwen35_gguf import FULL_ATTENTION, LINEAR_ATTENTION
from hipengine.runtime import qwen35_gguf_runner as gguf_runner


def _fake_dense_qwen36_runner() -> SimpleNamespace:
    cfg = SimpleNamespace(
        context_length=32_768,
        layer_types=tuple([LINEAR_ATTENTION] * 48 + [FULL_ATTENTION] * 16),
        expert_used_count=0,
        is_moe=False,
        expert_count=0,
        expert_shared_feed_forward_length=0,
        ssm_inner_size=6_144,
        ssm_conv_kernel=4,
        ssm_time_step_rank=48,
        ssm_state_size=128,
        ssm_group_count=16,
        head_count_kv=4,
        key_length=256,
        rope_dimension_count=64,
        rope_freq_base=10_000_000.0,
        head_count=24,
    )
    return SimpleNamespace(
        hidden_size=5_120,
        q_width=6_144,
        kv_width=1_024,
        ffn_size=17_408,
        linear_qkv_width=10_240,
        ssm_value_dim=128,
        fp16_recurrent_state=False,
        weights=SimpleNamespace(
            config=cfg,
            geometry=QWEN35_DENSE_H5120_GEOMETRY,
            model_name="arbitrary-finetune-name",
            file_type_name="MOSTLY_Q4_K_M",
        ),
    )


def _fake_moe_qwen36_runner() -> SimpleNamespace:
    cfg = SimpleNamespace(
        context_length=262_144,
        layer_types=tuple(
            [LINEAR_ATTENTION] * 3 + [FULL_ATTENTION]
        ) * 16,
        expert_used_count=8,
        is_moe=True,
        expert_count=256,
        expert_shared_feed_forward_length=512,
        ssm_inner_size=6_144,
        ssm_conv_kernel=4,
        ssm_time_step_rank=48,
        ssm_state_size=128,
        ssm_group_count=16,
        head_count_kv=4,
        key_length=256,
        rope_dimension_count=64,
        rope_freq_base=10_000_000.0,
        head_count=24,
    )
    return SimpleNamespace(
        hidden_size=5_120,
        q_width=6_144,
        kv_width=1_024,
        ffn_size=17_408,
        linear_qkv_width=10_240,
        ssm_value_dim=128,
        fp16_recurrent_state=False,
        weights=SimpleNamespace(
            config=cfg,
            geometry=QWEN35_DENSE_H5120_GEOMETRY,
            model_name="arbitrary-finetune-name",
            file_type_name="MOSTLY_Q4_K_M",
        ),
    )


class _FakeRuntime:
    def memset(self, *args, **kwargs) -> None:
        return None


def _install_fake_device(monkeypatch):
    next_ptr = 0x10000000
    allocations: list[DeviceBuffer] = []

    def fake_malloc(nbytes: int, *, runtime):
        nonlocal next_ptr
        size = int(nbytes)
        buffer = DeviceBuffer(ptr=next_ptr, nbytes=size)
        next_ptr += max(256, ((size + 255) // 256) * 256 + 256)
        allocations.append(buffer)
        return buffer

    monkeypatch.setattr(gguf_runner, "malloc", fake_malloc)
    monkeypatch.setattr(gguf_runner, "copy_host_to_device", lambda *args, **kwargs: None)
    return allocations


def _plan_for(runner, **kwargs):
    cfg = runner.weights.config
    return gguf_runner._full_stack_scratch_plan(
        cfg,
        hidden_size=runner.hidden_size,
        ffn_size=runner.ffn_size,
        q_width=runner.q_width,
        kv_width=runner.kv_width,
        linear_qkv_width=runner.linear_qkv_width,
        fp16_recurrent_state=runner.fp16_recurrent_state,
        **kwargs,
    )


def test_scratch_plan_matches_dedicated_allocation_byte_for_byte(monkeypatch) -> None:
    """The plan's owner_sizes must equal what the allocator actually asks for."""

    _install_fake_device(monkeypatch)
    runner = _fake_dense_qwen36_runner()
    plan = _plan_for(runner, max_sequence_length=640, max_batch_size=1)

    scratch = gguf_runner._FullStackScratch.allocate(
        runner,
        runtime=_FakeRuntime(),
        max_sequence_length=640,
        max_batch_size=1,
        use_single_arena=False,
    )

    assert scratch.allocation_mode == "dedicated"
    assert len(scratch.buffers) == len(plan.owner_sizes)
    # ``allocate`` reorders owners into fields/state/cache/metadata groups before
    # handing them to ``malloc``, so the guarantee is over the size multiset.
    # Allocation *order* is pinned by the single-arena test below, which derives
    # its offsets from ``owner_sizes`` directly.
    assert sorted(int(buffer.nbytes) for buffer in scratch.buffers) == sorted(plan.owner_sizes)
    assert sum(int(buffer.nbytes) for buffer in scratch.buffers) == plan.dedicated_bytes


_SCRATCH_PLAN_ALLOCATOR_MATRIX = (
    (
        "dense_bf16_640",
        _fake_dense_qwen36_runner,
        {"max_sequence_length": 640},
    ),
    (
        "dense_bf16_32768",
        _fake_dense_qwen36_runner,
        {"max_sequence_length": 32_768},
    ),
    (
        "dense_int8_fp32_8192_mirror_on",
        _fake_dense_qwen36_runner,
        {
            "max_sequence_length": 8_192,
            "kv_storage_dtype": "int8_per_token_head",
            "kv_scale_dtype": "fp32",
        },
    ),
    (
        "dense_int8_fp16_16384_mirror_off",
        _fake_dense_qwen36_runner,
        {
            "max_sequence_length": 16_384,
            "kv_storage_dtype": "int8_per_token_head",
            "kv_scale_dtype": "fp16",
        },
    ),
    (
        "dense_int8_fp32_deferred_kv_4slots",
        _fake_dense_qwen36_runner,
        {
            "max_sequence_length": 32_768,
            "max_batch_size": 4,
            "kv_storage_dtype": "int8_per_token_head",
            "kv_scale_dtype": "fp32",
            "allocate_kv_cache": False,
        },
    ),
    (
        "moe_int8_fp32_16384",
        _fake_moe_qwen36_runner,
        {
            "max_sequence_length": 16_384,
            "kv_storage_dtype": "int8_per_token_head",
            "kv_scale_dtype": "fp32",
        },
    ),
    (
        "moe_bf16_prefix_bf16_layers_2slots",
        _fake_moe_qwen36_runner,
        {
            "max_sequence_length": 16_384,
            "max_batch_size": 2,
            "int8_bf16_prefix_full_attention_layers": 3,
        },
    ),
)


def test_scratch_plan_matches_allocator_across_config_matrix(monkeypatch) -> None:
    """Pin the plan against the allocator for every KV policy the server can pick."""

    for name, factory, kwargs in _SCRATCH_PLAN_ALLOCATOR_MATRIX:
        _install_fake_device(monkeypatch)
        runner = factory()
        plan = _plan_for(runner, **kwargs)
        scratch = gguf_runner._FullStackScratch.allocate(
            runner,
            runtime=_FakeRuntime(),
            use_single_arena=False,
            **kwargs,
        )
        allocated = sorted(int(buffer.nbytes) for buffer in scratch.buffers)
        assert allocated == sorted(plan.owner_sizes), name
        assert sum(allocated) == plan.dedicated_bytes, name


def test_scratch_plan_matches_single_arena_allocation(monkeypatch) -> None:
    allocations = _install_fake_device(monkeypatch)
    runner = _fake_dense_qwen36_runner()
    plan = _plan_for(runner, max_sequence_length=640, max_batch_size=1)

    scratch = gguf_runner._FullStackScratch.allocate(
        runner,
        runtime=_FakeRuntime(),
        max_sequence_length=640,
        max_batch_size=1,
        use_single_arena=True,
    )

    assert scratch.allocation_mode == "single_arena"
    assert len(allocations) == 1
    assert int(allocations[0].nbytes) == plan.arena_bytes()


def test_scratch_plan_decomposition_is_exhaustive() -> None:
    """Every planned byte belongs to exactly one named accounting bucket."""

    for runner, kwargs in (
        (_fake_dense_qwen36_runner(), {"max_sequence_length": 640}),
        (
            _fake_dense_qwen36_runner(),
            {
                "max_sequence_length": 32_768,
                "kv_storage_dtype": "int8_per_token_head",
                "kv_scale_dtype": "fp32",
            },
        ),
        (_fake_moe_qwen36_runner(), {"max_sequence_length": 16_384}),
        (
            _fake_moe_qwen36_runner(),
            {
                "max_sequence_length": 16_384,
                "max_batch_size": 4,
                "kv_storage_dtype": "int8_per_token_head",
                "kv_scale_dtype": "fp32",
            },
        ),
    ):
        plan = _plan_for(runner, **kwargs)
        assert plan.dedicated_bytes == (
            plan.kv_payload_bytes
            + plan.kv_mirror_bytes
            + plan.kv_scale_bytes
            + plan.linear_state_bytes
            + plan.metadata_bytes
            + plan.workspace_bytes
        )
        assert plan.fixed_bytes == plan.dedicated_bytes - plan.context_scaled_bytes
        assert plan.context_scaled_bytes > 0
        assert plan.fixed_bytes > 0


def test_scratch_plan_context_slope_is_exactly_linear_above_the_mirror_threshold() -> None:
    """Above the BF16-mirror cutoff the per-token cost must be constant.

    Below the cutoff the INT8 route retains BF16 mirrors, which makes the
    footprint step rather than line up. That step is exactly why the capacity
    model has to come from the allocator instead of a single fitted slope.
    """

    runner = _fake_dense_qwen36_runner()
    common = {
        "kv_storage_dtype": "int8_per_token_head",
        "kv_scale_dtype": "fp32",
    }
    low = _plan_for(runner, max_sequence_length=8_192, **common)
    high = _plan_for(runner, max_sequence_length=16_384, **common)
    higher = _plan_for(runner, max_sequence_length=32_768, **common)

    assert low.kv_mirror_bytes > 0
    assert high.kv_mirror_bytes == 0
    assert higher.kv_mirror_bytes == 0

    per_token = high.context_scaled_bytes / high.max_positions
    assert higher.context_scaled_bytes / higher.max_positions == per_token
    assert (
        higher.context_scaled_bytes - high.context_scaled_bytes
    ) == per_token * (higher.max_positions - high.max_positions)


def test_scratch_plan_rejects_out_of_range_context() -> None:
    runner = _fake_dense_qwen36_runner()
    try:
        _plan_for(runner, max_sequence_length=32_769)
    except ValueError as exc:
        assert "exceeds GGUF context length" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected ValueError for context beyond the model maximum")


def test_scratch_plan_defaults_to_one_block_when_context_is_unspecified() -> None:
    runner = _fake_dense_qwen36_runner()
    plan = _plan_for(runner, max_sequence_length=None)
    assert plan.max_positions == 256
    assert plan.block_count == 1


# ---------------------------------------------------------------------------
# Resident capacity estimator
# ---------------------------------------------------------------------------


def _real_qwen38_27b_cfg() -> SimpleNamespace:
    """Qwen3.8-27B geometry as decoded from the shipped GGUF header."""

    return SimpleNamespace(
        context_length=262_144,
        layer_types=tuple(
            FULL_ATTENTION if (index + 1) % 4 == 0 else LINEAR_ATTENTION
            for index in range(65)
        ),
        expert_used_count=0,
        is_moe=False,
        expert_count=0,
        expert_shared_feed_forward_length=0,
        ssm_inner_size=6_144,
        ssm_conv_kernel=4,
        ssm_time_step_rank=48,
        ssm_state_size=128,
        ssm_group_count=16,
        head_count_kv=4,
        key_length=256,
        rope_dimension_count=64,
        rope_freq_base=10_000_000.0,
        head_count=24,
    )


_QWEN38_27B_GEOMETRY = {
    "hidden_size": 5_120,
    "ffn_size": 17_408,
    "q_width": 6_144,
    "kv_width": 1_024,
    "linear_qkv_width": 10_240,
}


def _estimate_27b(available_gib: float, **kwargs):
    return gguf_runner.estimate_qwen35_gguf_kv_capacity(
        _real_qwen38_27b_cfg(),
        available_bytes=int(available_gib * 2**30),
        requested_context_tokens=262_144,
        **_QWEN38_27B_GEOMETRY,
        kv_storage_dtype="int8_per_token_head",
        kv_scale_dtype="fp32",
        **kwargs,
    )


def test_gguf_capacity_marginal_slope_is_kv_pages_plus_transient() -> None:
    """The marginal cost must be the page slope plus the transient slope."""

    estimate = _estimate_27b(12.0)
    breakdown = gguf_runner.qwen35_gguf_resident_breakdown(
        _real_qwen38_27b_cfg(),
        context_tokens=estimate.allocatable_context_tokens,
        **_QWEN38_27B_GEOMETRY,
        kv_storage_dtype="int8_per_token_head",
        kv_scale_dtype="fp32",
    )
    expected = (
        breakdown.page_bytes / 256
        + 74 * 1024
        + breakdown.scratch_context_bytes / breakdown.max_positions
    )
    assert abs(estimate.marginal_bytes_per_token - expected) <= 64
    # 32.5 KiB of INT8 payload plus 64 KiB of oracle plus 10 KiB of hidden planes.
    assert 105 * 1024 <= estimate.marginal_bytes_per_token <= 110 * 1024


def test_gguf_capacity_scales_down_with_available_memory() -> None:
    contexts = [_estimate_27b(gib).allocatable_context_tokens for gib in (23.5, 12.0, 7.0, 4.0, 2.0)]
    assert contexts == sorted(contexts, reverse=True)
    assert contexts[0] > contexts[-1]
    assert all(context % 256 == 0 for context in contexts)


def test_gguf_capacity_matches_measured_24gb_server_ceiling() -> None:
    """The 24 GB server route measured a 54,272-token ceiling (2026-09-09 census).

    23.98 GiB card minus 16.45 GiB of resident weights and scratch leaves about
    7.5 GiB; the estimate must land in the same band rather than promising the
    232K the oracle-free direct route reaches.
    """

    estimate = _estimate_27b(7.53)
    assert 48_000 <= estimate.allocatable_context_tokens <= 64_000


def test_gguf_capacity_handles_the_int8_mirror_step() -> None:
    """Below the mirror threshold the INT8 route is *more* expensive per token.

    A naive bisection that assumed a monotonic curve would either refuse a
    context that fits or promise one that does not.
    """

    below = gguf_runner.qwen35_gguf_resident_breakdown(
        _real_qwen38_27b_cfg(),
        context_tokens=8_192,
        **_QWEN38_27B_GEOMETRY,
        kv_storage_dtype="int8_per_token_head",
        kv_scale_dtype="fp32",
    )
    above = gguf_runner.qwen35_gguf_resident_breakdown(
        _real_qwen38_27b_cfg(),
        context_tokens=8_448,
        **_QWEN38_27B_GEOMETRY,
        kv_storage_dtype="int8_per_token_head",
        kv_scale_dtype="fp32",
    )
    assert below.total_bytes > above.total_bytes

    # Pick a budget that only fits once the mirrors drop away.
    usable = above.total_bytes + 2**20
    estimate = gguf_runner.estimate_qwen35_gguf_kv_capacity(
        _real_qwen38_27b_cfg(),
        available_bytes=usable + 512 * 1024**2,
        requested_context_tokens=262_144,
        **_QWEN38_27B_GEOMETRY,
        kv_storage_dtype="int8_per_token_head",
        kv_scale_dtype="fp32",
    )
    assert estimate.allocatable_context_tokens >= 8_448


def test_gguf_capacity_excludes_deferred_kv_from_scratch() -> None:
    """The server defers KV to the page pool, so scratch must not double-charge it."""

    deferred = gguf_runner.qwen35_gguf_resident_breakdown(
        _real_qwen38_27b_cfg(),
        context_tokens=32_768,
        **_QWEN38_27B_GEOMETRY,
        kv_storage_dtype="int8_per_token_head",
        kv_scale_dtype="fp32",
        allocate_kv_cache=False,
    )
    resident = gguf_runner.qwen35_gguf_resident_breakdown(
        _real_qwen38_27b_cfg(),
        context_tokens=32_768,
        **_QWEN38_27B_GEOMETRY,
        kv_storage_dtype="int8_per_token_head",
        kv_scale_dtype="fp32",
        allocate_kv_cache=True,
    )
    assert deferred.kv_pool_bytes > 0
    assert resident.scratch_bytes > deferred.scratch_bytes
    assert resident.scratch_bytes - deferred.scratch_bytes == resident.kv_pool_bytes


def test_gguf_capacity_workspace_lease_mirrors_the_declared_context() -> None:
    plain = gguf_runner.qwen35_gguf_resident_breakdown(
        _real_qwen38_27b_cfg(),
        context_tokens=16_384,
        **_QWEN38_27B_GEOMETRY,
        kv_storage_dtype="int8_per_token_head",
        kv_scale_dtype="fp32",
        workspace_lease_needed=False,
    )
    leased = gguf_runner.qwen35_gguf_resident_breakdown(
        _real_qwen38_27b_cfg(),
        context_tokens=16_384,
        **_QWEN38_27B_GEOMETRY,
        kv_storage_dtype="int8_per_token_head",
        kv_scale_dtype="fp32",
        workspace_lease_needed=True,
    )
    # 16,384 tokens is 64 pages, which is above the 1,024-token packed floor.
    assert plain.workspace_lease_pages == 0
    assert leased.workspace_lease_pages == 64
    assert leased.workspace_lease_bytes == 64 * leased.page_bytes
    assert leased.total_bytes - plain.total_bytes == leased.workspace_lease_bytes


def test_gguf_capacity_reserve_and_transient_overrides_are_respected() -> None:
    generous = _estimate_27b(8.0, transient_bytes_per_token=0, transient_fixed_bytes=0)
    conservative = _estimate_27b(8.0, transient_bytes_per_token=128 * 1024, transient_fixed_bytes=2**31)
    assert generous.allocatable_context_tokens > conservative.allocatable_context_tokens

    small_reserve = _estimate_27b(8.0, reserve_bytes=0)
    large_reserve = _estimate_27b(8.0, reserve_bytes=2**31)
    assert small_reserve.allocatable_context_tokens > large_reserve.allocatable_context_tokens
    assert large_reserve.usable_bytes == max(0, large_reserve.available_bytes - 2**31)


def test_gguf_capacity_reports_requested_context_fit() -> None:
    fits = _estimate_27b(30.0)
    assert fits.fits_requested is True
    assert fits.fits_model_max is True
    assert fits.allocatable_context_tokens == fits.model_max_context_tokens == 262_144

    tight = _estimate_27b(4.0)
    assert tight.fits_requested is False
    assert tight.fits_model_max is False
    assert tight.requested_total_bytes > tight.usable_bytes


def test_gguf_capacity_estimate_is_json_serialisable() -> None:
    import json

    payload = _estimate_27b(12.0).to_json_dict()
    json.dumps(payload)
    assert payload["kv_storage_dtype"] == "int8_per_token_head"
    assert payload["allocatable_context_tokens"] > 0


# ---------------------------------------------------------------------------
# Generator-level auto-context resolution and fallback
# ---------------------------------------------------------------------------


def _auto_context_generator(**overrides):
    """Minimal generator instance for exercising the auto-context methods."""

    generator = qwen35_gguf.Qwen35GGUFBringupGenerator.__new__(
        qwen35_gguf.Qwen35GGUFBringupGenerator
    )
    generator.model_path = "/tmp/fake.gguf"
    generator.backend = "hip_gfx1100"
    generator._prepared_kv_signature = (
        "int8_per_token_head",
        "uniform",
        "fp32",
        "per_token_head",
    )
    generator._auto_resolved_max_sequence_length = None
    generator._auto_resolved_max_sequence_lengths = {}
    generator._auto_context_estimate = None
    for name, value in overrides.items():
        setattr(generator, name, value)
    return generator


def _resident_runner_stub(generator, *, capacity: int = 4):
    """Minimal resident model runner for exercising ``prepare`` retry logic."""

    runner = qwen35_gguf.Qwen35GGUFResidentModelRunner.__new__(
        qwen35_gguf.Qwen35GGUFResidentModelRunner
    )
    runner.generator = generator
    runner.capacity = int(capacity)
    runner._shared_runner = object()
    runner._engine_loop_config = None
    runner._kv_pool = None
    runner._available = []
    runner._rows = {}
    runner._resident_batch_owner = None
    runner._resident_batch_owner_pool_key = None
    runner._max_sequence_length = None
    runner._prefix_state_snapshots = {}
    return runner


def _auto_context_runner(*, free_gib: float = 8.0, with_geometry: bool = True):
    weights = SimpleNamespace(config=_real_qwen38_27b_cfg())
    runtime = SimpleNamespace(mem_get_info=lambda: (int(free_gib * 2**30), 24 * 2**30))
    runner = SimpleNamespace(
        weights=weights,
        runtime=runtime,
        fp16_recurrent_state=False,
    )
    if with_geometry:
        for name, value in _QWEN38_27B_GEOMETRY.items():
            setattr(runner, name, value)
    return runner


def test_auto_context_reserve_default_covers_measured_untracked_overhead(monkeypatch) -> None:
    """The reserve has to cover device memory the model never prices.

    Measured on the W7900 at the auto-selected 27B context: 42.91 GiB of
    whole-card use against 40.25 GiB of hipEngine-tracked allocations. The
    default is pinned here so lowering it back toward the old 512 MiB is a
    deliberate act with a failing test attached.
    """

    monkeypatch.delenv("HIPENGINE_GGUF_KV_CAPACITY_RESERVE_MIB", raising=False)
    reserve_mib = qwen35_gguf._gguf_auto_context_reserve_bytes() // 1024**2
    assert reserve_mib >= 2560  # measured 2.66 GiB of untracked device memory

    monkeypatch.setenv("HIPENGINE_GGUF_KV_CAPACITY_RESERVE_MIB", "256")
    assert qwen35_gguf._gguf_auto_context_reserve_bytes() == 256 * 1024**2


def test_auto_context_resolves_and_caches_a_context_that_fits(monkeypatch) -> None:
    monkeypatch.delenv("HIPENGINE_GGUF_AUTO_CONTEXT", raising=False)
    monkeypatch.delenv("HIPENGINE_GGUF_KV_CAPACITY_RESERVE_MIB", raising=False)
    generator = _auto_context_generator()
    runner = _auto_context_runner(free_gib=8.0)

    selected = generator._resolve_auto_context(
        runner, max_batch_size=1, defer_kv_allocation=True
    )
    assert selected is not None and selected > 0
    assert selected % 256 == 0
    assert selected < 262_144
    # Cached: a second call must not re-price against a changed free-memory view.
    runner.runtime = SimpleNamespace(mem_get_info=lambda: (1, 24 * 2**30))
    assert generator._resolve_auto_context(
        runner, max_batch_size=1, defer_kv_allocation=True
    ) == selected
    assert generator._auto_context_estimate.allocatable_context_tokens == selected


def test_auto_context_cache_is_keyed_by_batch_size(monkeypatch) -> None:
    """A larger batch prices tighter and must not inherit the batch-1 context."""

    monkeypatch.delenv("HIPENGINE_GGUF_AUTO_CONTEXT", raising=False)
    monkeypatch.delenv("HIPENGINE_GGUF_KV_CAPACITY_RESERVE_MIB", raising=False)
    generator = _auto_context_generator()
    runner = _auto_context_runner(free_gib=8.0)

    single = generator._resolve_auto_context(
        runner, max_batch_size=1, defer_kv_allocation=True
    )
    batched = generator._resolve_auto_context(
        runner, max_batch_size=8, defer_kv_allocation=True
    )

    assert single is not None and batched is not None
    assert batched <= single
    assert generator._auto_resolved_max_sequence_length == min(single, batched)


def test_auto_context_honours_the_disable_flag(monkeypatch) -> None:
    monkeypatch.setenv("HIPENGINE_GGUF_AUTO_CONTEXT", "0")
    generator = _auto_context_generator()
    assert (
        generator._resolve_auto_context(
            _auto_context_runner(), max_batch_size=1, defer_kv_allocation=True
        )
        is None
    )


def test_auto_context_degrades_when_geometry_is_unavailable(monkeypatch) -> None:
    """Fake runners in unit tests must keep the historical fixed path."""

    monkeypatch.delenv("HIPENGINE_GGUF_AUTO_CONTEXT", raising=False)
    generator = _auto_context_generator()
    assert (
        generator._resolve_auto_context(
            _auto_context_runner(with_geometry=False),
            max_batch_size=1,
            defer_kv_allocation=True,
        )
        is None
    )


def test_auto_context_reserve_shrinks_the_selected_context(monkeypatch) -> None:
    monkeypatch.delenv("HIPENGINE_GGUF_AUTO_CONTEXT", raising=False)
    runner = _auto_context_runner(free_gib=8.0)

    monkeypatch.delenv("HIPENGINE_GGUF_KV_CAPACITY_RESERVE_MIB", raising=False)
    baseline = _auto_context_generator()._resolve_auto_context(
        runner, max_batch_size=1, defer_kv_allocation=True
    )
    monkeypatch.setenv("HIPENGINE_GGUF_KV_CAPACITY_RESERVE_MIB", "4096")
    reserved = _auto_context_generator()._resolve_auto_context(
        runner, max_batch_size=1, defer_kv_allocation=True
    )

    assert baseline is not None and reserved is not None
    assert reserved < baseline
    assert reserved % 256 == 0


def test_construct_shared_session_retries_smaller_after_oom(monkeypatch) -> None:
    """A failed allocation must fall back to a smaller context, not surface."""

    monkeypatch.delenv("HIPENGINE_GGUF_AUTO_CONTEXT", raising=False)
    monkeypatch.setenv("HIPENGINE_GGUF_AUTO_CONTEXT_ATTEMPTS", "3")
    generator = _auto_context_generator()
    generator._prepared_session_kv_kwargs = lambda: {}
    generator._configure_session = lambda session: None
    generator._auto_context_estimate = None
    runner = _auto_context_runner(free_gib=8.0)
    attempts: list[int | None] = []
    ceiling = 16_384

    class _Session:
        def __init__(self, model_path, **kwargs):
            context = kwargs.get("max_sequence_length")
            attempts.append(context)
            if context is not None and int(context) > ceiling:
                raise MemoryError("hip out of memory")
            self.max_sequence_length = context

    monkeypatch.setattr(qwen35_gguf, "Qwen35GGUFResidentSession", _Session)

    session = generator._construct_shared_session(
        runner,
        max_sequence_length=131_072,
        max_batch_size=1,
        defer_kv_allocation=True,
        use_wmma_prefill=None,
        use_gemv_decode=None,
    )

    assert session.max_sequence_length is not None
    assert session.max_sequence_length <= ceiling
    assert len(attempts) >= 2
    assert attempts[0] == 131_072
    assert attempts == sorted(attempts, reverse=True)


def test_construct_shared_session_degrades_a_pinned_context_with_a_warning(monkeypatch) -> None:
    """An explicit context may degrade, but the failed request is observable."""

    generator = _auto_context_generator()
    generator._prepared_session_kv_kwargs = lambda: {}
    generator._configure_session = lambda session: None
    attempts: list[int | None] = []
    warnings: list[str] = []
    monkeypatch.setattr(
        qwen35_gguf._LOGGER,
        "warning",
        lambda message, *args: warnings.append(message % args),
    )

    class _Session:
        def __init__(self, model_path, **kwargs):
            attempts.append(kwargs.get("max_sequence_length"))
            context = kwargs.get("max_sequence_length")
            if context is not None and int(context) > 16_384:
                raise MemoryError("hip out of memory")
            self.max_sequence_length = context

    monkeypatch.setattr(qwen35_gguf, "Qwen35GGUFResidentSession", _Session)
    session = generator._construct_shared_session(
        _auto_context_runner(),
        max_sequence_length=131_072,
        max_batch_size=1,
        defer_kv_allocation=True,
        use_wmma_prefill=None,
        use_gemv_decode=None,
    )

    assert session.max_sequence_length < 131_072
    assert attempts[0] == 131_072
    assert attempts[-1] == session.max_sequence_length
    assert any("requested 131072 tokens failed" in warning for warning in warnings)


def test_packed_workspace_lease_mirrors_the_engine_loop_decision(monkeypatch) -> None:
    """The lease term has to match what the pool will actually reserve."""

    monkeypatch.delenv("HIPENGINE_GGUF_PACKED_KV_LEASE", raising=False)
    generator = _auto_context_generator()

    # A multi-slot owner always leases, whatever the policy says.
    assert generator._packed_workspace_lease_needed(max_batch_size=4) is True

    # One slot with no resolved policy yet: assume the lease.
    generator._resident_model_runner = None
    assert generator._packed_workspace_lease_needed(max_batch_size=1) is True

    owner = SimpleNamespace(
        _engine_loop_config=SimpleNamespace(speculative_mtp_serving="off"),
        _prefix_cache_mode="off",
    )
    generator._resident_model_runner = owner
    assert generator._packed_workspace_lease_needed(max_batch_size=1) is False

    owner._engine_loop_config = SimpleNamespace(speculative_mtp_serving="auto")
    assert generator._packed_workspace_lease_needed(max_batch_size=1) is True

    owner._engine_loop_config = SimpleNamespace(speculative_mtp_serving="off")
    owner._prefix_cache_mode = "radix"
    assert generator._packed_workspace_lease_needed(max_batch_size=1) is True

    owner._prefix_cache_mode = "off"
    monkeypatch.setenv("HIPENGINE_GGUF_PACKED_KV_LEASE", "1")
    assert generator._packed_workspace_lease_needed(max_batch_size=1) is True


def test_prepare_retries_when_the_kv_pool_allocation_fails(monkeypatch) -> None:
    """The pool is the dominant allocation; the retry has to cover it too."""

    monkeypatch.delenv("HIPENGINE_GGUF_AUTO_CONTEXT", raising=False)
    monkeypatch.setenv("HIPENGINE_GGUF_AUTO_CONTEXT_ATTEMPTS", "4")
    generator = _auto_context_generator()
    runner = _resident_runner_stub(generator, capacity=4)
    # Resolve once so the generator has a context to recalibrate from.
    generator._resolve_auto_context(
        _auto_context_runner(free_gib=8.0), max_batch_size=4, defer_kv_allocation=True
    )
    selected = generator._auto_resolved_max_sequence_length
    assert selected is not None

    reserved: list[int | None] = []

    def _reserve_sessions() -> None:
        context = generator._auto_resolved_max_sequence_length
        reserved.append(context)
        # Fail while the pool would need more than half the original budget.
        if context is not None and int(context) > selected // 2:
            raise MemoryError("hip out of memory")
        runner._available = [object()]

    runner._reserve_sessions = _reserve_sessions
    runner._clear_prefix_snapshots = lambda: None
    runner._release_available_sessions = lambda: None

    runner.prepare()

    assert len(reserved) >= 2
    assert reserved[0] == selected
    assert reserved == sorted(reserved, reverse=True)
    assert generator._auto_resolved_max_sequence_length < selected


def test_prepare_degrades_a_pinned_context_with_a_warning(monkeypatch) -> None:
    """A pinned context backs off too, and the warning names the asked-for size."""

    monkeypatch.delenv("HIPENGINE_GGUF_AUTO_CONTEXT", raising=False)
    monkeypatch.setenv("HIPENGINE_GGUF_AUTO_CONTEXT_ATTEMPTS", "4")
    generator = _auto_context_generator(_prepared_max_sequence_length=131_072)
    runner = _resident_runner_stub(generator, capacity=4)
    attempted: list[int | None] = []
    warnings: list[str] = []
    monkeypatch.setattr(
        qwen35_gguf._LOGGER,
        "warning",
        lambda message, *args: warnings.append(message % args),
    )

    def _reserve_sessions() -> None:
        attempted.append(runner._max_sequence_length)
        if runner._max_sequence_length is None or int(runner._max_sequence_length) > 32_768:
            raise MemoryError("hip out of memory")
        runner._available = [object()]

    runner._reserve_sessions = _reserve_sessions
    runner._clear_prefix_snapshots = lambda: None
    runner._release_available_sessions = lambda: None

    runner.prepare()

    assert attempted[0] == 131_072
    assert attempted[-1] is not None and attempted[-1] <= 32_768
    assert attempted == sorted(attempted, reverse=True)
    assert any("requested 131072 tokens failed" in warning for warning in warnings)


def test_auto_context_flag_disables_the_fallback_on_both_paths(monkeypatch) -> None:
    """``HIPENGINE_GGUF_AUTO_CONTEXT=0`` is the full rollback, not just sizing."""

    monkeypatch.setenv("HIPENGINE_GGUF_AUTO_CONTEXT", "0")
    monkeypatch.setenv("HIPENGINE_GGUF_AUTO_CONTEXT_ATTEMPTS", "4")
    generator = _auto_context_generator(_prepared_max_sequence_length=131_072)
    generator._prepared_session_kv_kwargs = lambda: {}
    generator._configure_session = lambda session: None
    attempts: list[int | None] = []

    class _Session:
        def __init__(self, model_path, **kwargs):
            attempts.append(kwargs.get("max_sequence_length"))
            raise MemoryError("hip out of memory")

    monkeypatch.setattr(qwen35_gguf, "Qwen35GGUFResidentSession", _Session)
    try:
        generator._construct_shared_session(
            _auto_context_runner(),
            max_sequence_length=131_072,
            max_batch_size=1,
            defer_kv_allocation=True,
            use_wmma_prefill=None,
            use_gemv_decode=None,
        )
    except MemoryError:
        pass
    else:  # pragma: no cover - defensive
        raise AssertionError("expected the failure to propagate with the flag off")
    assert attempts == [131_072]

    runner = _resident_runner_stub(generator, capacity=4)
    reserved: list[int | None] = []

    def _reserve_sessions() -> None:
        reserved.append(runner._max_sequence_length)
        raise MemoryError("hip out of memory")

    runner._reserve_sessions = _reserve_sessions
    runner._clear_prefix_snapshots = lambda: None
    runner._release_available_sessions = lambda: None
    try:
        runner.prepare()
    except MemoryError:
        pass
    else:  # pragma: no cover - defensive
        raise AssertionError("expected the failure to propagate with the flag off")
    assert reserved == [131_072]


def test_construct_shared_session_propagates_without_a_context(monkeypatch) -> None:
    """With no context there is nothing to back off to, so OOM must propagate."""

    monkeypatch.delenv("HIPENGINE_GGUF_AUTO_CONTEXT", raising=False)
    generator = _auto_context_generator()
    generator._prepared_session_kv_kwargs = lambda: {}
    generator._configure_session = lambda session: None
    attempts: list[int | None] = []

    class _Session:
        def __init__(self, model_path, **kwargs):
            attempts.append(kwargs.get("max_sequence_length"))
            raise MemoryError("hip out of memory")

    monkeypatch.setattr(qwen35_gguf, "Qwen35GGUFResidentSession", _Session)
    try:
        generator._construct_shared_session(
            _auto_context_runner(),
            max_sequence_length=None,
            max_batch_size=1,
            defer_kv_allocation=True,
            use_wmma_prefill=None,
            use_gemv_decode=None,
        )
    except MemoryError:
        pass
    else:  # pragma: no cover - defensive
        raise AssertionError("expected the contextless allocation failure to propagate")
    assert attempts == [None]


def test_recalibrated_auto_context_is_strictly_smaller_and_aligned(monkeypatch) -> None:
    monkeypatch.delenv("HIPENGINE_GGUF_AUTO_CONTEXT", raising=False)
    generator = _auto_context_generator()
    runner = _auto_context_runner(free_gib=8.0)
    for failed in (4_096, 65_536, 131_072):
        smaller = generator._recalibrated_auto_context(
            runner,
            failed_context=failed,
            max_batch_size=1,
            defer_kv_allocation=True,
        )
        assert smaller < failed
        assert smaller % 256 == 0
        assert smaller >= 256
