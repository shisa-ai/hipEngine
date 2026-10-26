"""Route-aware GGUF server capacity terms (24 GB serving fixes).

The flat model priced a 74 KiB/token transient for every KV storage, private
packed planes for every serving slot, a 1 GiB fixed transient, and nothing for
the MTP draft or the prefill scratch allocated after the free-memory reading.
On a 24 GB card that both over-priced BF16 (5.6K auto context where 32K serves)
and under-priced MTP (the warmup OOMed and closed the engine). These tests pin
the replacement terms against the allocators' own arithmetic.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from hipengine.core.memory import DeviceBuffer
from hipengine.generation import qwen35_gguf
from hipengine.loading.qwen35_gguf import FULL_ATTENTION, LINEAR_ATTENTION
from hipengine.runtime import qwen35_gguf_runner as gguf_runner

from tests.test_unit_qwen35_gguf_prefill_scratch_liveness import (
    _fake_dense_qwen36_runner,
)


def _dense_27b_runner():
    runner = _fake_dense_qwen36_runner()
    cfg = runner.weights.config
    cfg.context_length = 262_144
    cfg.block_count = 64
    cfg.declared_block_count = 64
    cfg.ignored_block_ids = ()
    cfg.layer_types = tuple(
        FULL_ATTENTION if (index + 1) % 4 == 0 else LINEAR_ATTENTION
        for index in range(64)
    )
    runner.fp16_recurrent_state = False
    return runner


def _real_allocation_bytes(monkeypatch, runner, *, rows, capacity, segments=1):
    allocated: list[int] = []
    next_ptr = [0x2000_0000]

    def fake_malloc(nbytes, *, runtime):
        allocated.append(int(nbytes))
        ptr = next_ptr[0]
        next_ptr[0] += ((int(nbytes) + 255) // 256) * 256 + 256
        return DeviceBuffer(ptr=ptr, nbytes=int(nbytes))

    with monkeypatch.context() as patch:
        patch.setattr(gguf_runner, "malloc", fake_malloc)
        patch.setattr(gguf_runner, "copy_host_array_to_device", lambda *a, **k: None)
        gguf_runner._GGUFFullAttentionPrefillScratch.allocate(
            runner,
            rows=rows,
            capacity=capacity,
            allocate_kv_cache=False,
            segments=segments,
            runtime=SimpleNamespace(memset=lambda *a, **k: None),
        )
    return sum(allocated)


@pytest.mark.parametrize(
    ("rows", "capacity", "segments"),
    [(1024, 1024, 1), (1024, 3584, 1), (1024, 32_768, 4), (128, 32_768, 1), (128, 1024, 1)],
)
def test_prefill_scratch_plan_matches_the_allocator(monkeypatch, rows, capacity, segments):
    """Allocation and estimation share the same exact block-rounded layout."""

    runner = _dense_27b_runner()
    expected = _real_allocation_bytes(
        monkeypatch, runner, rows=rows, capacity=capacity, segments=segments
    )
    measured = gguf_runner._gguf_prefill_scratch_nbytes(
        runner, rows=rows, capacity=capacity, segments=segments
    )
    assert measured == expected


def test_packed_state_linear_bytes_match_the_state_allocator() -> None:
    runner = _dense_27b_runner()
    per_slot = gguf_runner._gguf_packed_state_linear_bytes(runner, slot_count=1)
    # 48 linear layers x (conv 10240x4 fp32 + recurrent 48x128x128 fp32).
    assert per_slot == 48 * (10_240 * 4 * 4 + 48 * 128 * 128 * 4)
    assert gguf_runner._gguf_packed_state_linear_bytes(runner, slot_count=4) == 4 * per_slot


def _breakdown(**kwargs):
    runner = _dense_27b_runner()
    return gguf_runner.qwen35_gguf_resident_breakdown(
        runner.weights.config,
        hidden_size=5_120,
        ffn_size=17_408,
        q_width=6_144,
        kv_width=1_024,
        linear_qkv_width=10_240,
        kv_storage_dtype="bf16",
        **kwargs,
    )


def test_private_growth_is_priced_for_the_requested_slots_only() -> None:
    every_slot = _breakdown(
        context_tokens=32_768, workspace_lease_needed=True, workspace_lease_slots=4
    )
    one_request = _breakdown(
        context_tokens=32_768,
        workspace_lease_needed=True,
        workspace_lease_slots=4,
        private_workspace_slots=1,
    )
    none = _breakdown(
        context_tokens=32_768,
        workspace_lease_needed=True,
        workspace_lease_slots=4,
        private_workspace_slots=0,
    )
    assert every_slot.private_workspace_bytes == 4 * one_request.private_workspace_bytes
    assert one_request.private_workspace_bytes == one_request.kv_pool_bytes
    assert none.private_workspace_bytes == 0
    assert none.workspace_lease_bytes == every_slot.workspace_lease_bytes


def test_resolvers_replace_the_flat_transient_and_add_retained_terms() -> None:
    seen: list[int] = []

    def transient(positions):
        seen.append(positions)
        return 7 * positions

    breakdown = _breakdown(
        context_tokens=4_000,
        transient_bytes_resolver=transient,
        prefill_workspace_bytes_resolver=lambda positions: 11,
        speculative_bytes_resolver=lambda positions: 13 * positions,
        kv_pool_floor_pages=64,
    )
    assert seen == [breakdown.max_positions]
    assert breakdown.transient_context_bytes == 7 * breakdown.max_positions
    assert breakdown.prefill_workspace_bytes == 11
    assert breakdown.speculative_bytes == 13 * breakdown.max_positions
    assert breakdown.kv_pool_pages == 64
    assert breakdown.retained_bytes == (
        breakdown.scratch_bytes
        + breakdown.kv_pool_bytes
        + breakdown.workspace_lease_bytes
        + breakdown.private_workspace_bytes
        + 11
        + 13 * breakdown.max_positions
    )


def _route_generator(monkeypatch, *, mtp_policy: str = "auto"):
    generator = qwen35_gguf.Qwen35GGUFBringupGenerator.__new__(
        qwen35_gguf.Qwen35GGUFBringupGenerator
    )
    generator.model_path = "/tmp/fake.gguf"
    generator.backend = "hip_gfx1100"
    generator.dms_config = None
    generator.weight_index = object()
    generator.speculative_candidate_budget = 3
    generator._resident_model_runner = SimpleNamespace(
        _engine_loop_config=SimpleNamespace(speculative_mtp_serving=mtp_policy)
    )
    monkeypatch.setattr(qwen35_gguf, "_gguf_info_has_mtp_tensors", lambda index: True)
    return generator


def _route_terms(monkeypatch, *, storage, capacity=4, mtp=True, server_pool=True):
    generator = _route_generator(monkeypatch, mtp_policy="auto" if mtp else "off")
    return generator._capacity_route_terms(
        _dense_27b_runner(),
        storage=storage,
        qualified=True,
        serving_capacity=capacity,
        server_pool=server_pool,
        mtp_sized=generator._mtp_serving_enabled_for_sizing(),
    )


def test_bf16_prefill_has_no_per_token_transient(monkeypatch) -> None:
    terms = _route_terms(monkeypatch, storage="bf16", mtp=False)
    assert terms["transient_bytes_resolver"](32_768) == 0
    assert terms["transient_fixed_bytes"] == 0
    assert terms["private_workspace_slots"] == 0


def test_int8_transient_follows_the_layer_outer_executor(monkeypatch) -> None:
    pair = 2 * 32_768 * 4 * 256 * 2
    streamed = _route_terms(monkeypatch, storage="int8_per_token_head", mtp=True)
    # Streamed MTP prompts run layer-outer as well, publishing final rows to
    # the draft sink, so they keep the shared pair rather than one per layer.
    assert streamed["transient_bytes_resolver"](32_768) == 3 * pair
    plain = _route_terms(monkeypatch, storage="int8_per_token_head", mtp=False)
    # Layer-outer keeps the shared pair (plus the packed slab's) transiently;
    # its full-capacity hidden owner is allocated with the bulk workspace.
    assert plain["transient_bytes_resolver"](32_768) == 3 * pair
    bf16 = _route_terms(monkeypatch, storage="bf16", mtp=False)
    hidden_plane = 32_768 * 5_120 * 2
    assert (
        plain["prefill_workspace_bytes_resolver"](32_768)
        - bf16["prefill_workspace_bytes_resolver"](32_768)
        >= hidden_plane
    )
    assert plain["speculative_bytes_resolver"] is None
    # INT8 verification never grows packed planes.
    assert streamed["private_workspace_slots"] == 0


@pytest.mark.parametrize("positions", [4096, 32768, 55808])
@pytest.mark.parametrize("mtp", [False, True])
def test_int8_capacity_prices_the_allocated_hidden_plane_count(monkeypatch, positions, mtp):
    """The rollback adds one physical plane, not a different execution plan."""
    monkeypatch.setenv("HIPENGINE_INT8_LAYER_OUTER_HIDDEN_ALIAS", "0")
    two = _route_terms(monkeypatch, storage="int8_per_token_head", mtp=mtp)
    two_bytes = two["prefill_workspace_bytes_resolver"](positions)
    two_transient = two["transient_bytes_resolver"](positions)
    monkeypatch.setenv("HIPENGINE_INT8_LAYER_OUTER_HIDDEN_ALIAS", "1")
    one = _route_terms(monkeypatch, storage="int8_per_token_head", mtp=mtp)
    assert two_bytes - one["prefill_workspace_bytes_resolver"](positions) == positions * 5120 * 2
    assert one["transient_bytes_resolver"](positions) == two_transient


@pytest.mark.parametrize("positions", [4096, 32768, 65792])
@pytest.mark.parametrize("capacity", [1, 4])
def test_capacity_prices_resumable_checkpoint_allocator(monkeypatch, positions, capacity):
    runner = _dense_27b_runner()
    state_bytes = gguf_runner._gguf_packed_state_linear_bytes(runner, slot_count=1)
    allocated = []

    def fake_malloc(nbytes, *, runtime):
        allocated.append(nbytes)
        return DeviceBuffer(ptr=len(allocated) * 4096, nbytes=nbytes)

    with monkeypatch.context() as patch:
        patch.setattr(gguf_runner, "malloc", fake_malloc)
        gguf_runner._GGUFResumablePrefillScratch.allocate(
            runtime=object(),
            packed_state=SimpleNamespace(
                layer_conv_states=(DeviceBuffer(ptr=1, nbytes=state_bytes),),
                layer_recurrent_states=(),
            ),
            hidden_bytes=positions * runner.hidden_size * 2,
        )
    plain = _route_terms(monkeypatch, storage="int8_per_token_head", capacity=capacity, mtp=False)
    streamed = _route_terms(monkeypatch, storage="int8_per_token_head", capacity=capacity, mtp=True)
    bf16 = _route_terms(monkeypatch, storage="bf16", capacity=capacity, mtp=False)
    # A single-slot server suspends in place and allocates no checkpoint.
    checkpoints = 0 if capacity == 1 else capacity * sum(allocated)
    for terms in (plain, streamed):
        # BF16 never suspends a layer-outer prefill; the rest of the post-pool
        # workspace is storage-independent.
        assert (
            terms["_packed_scratch_bytes_resolver"](positions)
            - bf16["_packed_scratch_bytes_resolver"](positions)
        ) == checkpoints
    generator = _route_generator(monkeypatch, mtp_policy="off")
    generator._prepared_kv_signature = ("int8_per_token_head",)
    monkeypatch.setattr(qwen35_gguf, "_admitted_no_mirror_int8_capability", lambda _: True)
    reserve = generator.kv_pool_execution_reserve_bytes(
        runner, max_positions=positions, serving_capacity=capacity
    )
    safety = max(0, qwen35_gguf._gguf_auto_context_reserve_bytes()
                 - qwen35_gguf._GGUF_PRE_POOL_UNTRACKED_BYTES)
    assert reserve == (
        safety + plain["_packed_scratch_bytes_resolver"](positions)
        + plain["transient_bytes_resolver"](positions)
    )
    assert plain["_packed_scratch_bytes_resolver"](positions) >= (
        checkpoints
        + gguf_runner._gguf_packed_state_linear_bytes(runner, slot_count=capacity)
    )


def test_bf16_mtp_prices_verify_rows_and_scales_with_width(monkeypatch) -> None:
    c1 = _route_terms(monkeypatch, storage="bf16", capacity=1)
    c4 = _route_terms(monkeypatch, storage="bf16", capacity=4)
    # Single-request routes never grow full-context packed planes.
    assert c1["private_workspace_slots"] == 0
    single = c1["speculative_bytes_resolver"](16_384)
    wide = c4["speculative_bytes_resolver"](16_384)
    runner = _dense_27b_runner()
    per_row = gguf_runner._gguf_packed_state_linear_bytes(runner, slot_count=1)
    # c1 verifies 1 x (budget 3 + 1) rows; c4 groups two requests and adds a
    # batch provider shape next to the singleton one.
    assert single >= 4 * per_row
    assert wide - single >= 4 * per_row
    assert c4["speculative_bytes_resolver"](32_768) > wide


def test_prefill_workspace_counts_bulk_and_packed_scratch(monkeypatch) -> None:
    server = _route_terms(monkeypatch, storage="bf16", capacity=4, mtp=False)
    direct = _route_terms(monkeypatch, storage="bf16", capacity=4, mtp=False, server_pool=False)
    runner = _dense_27b_runner()
    state = gguf_runner._gguf_packed_state_linear_bytes(runner, slot_count=4)
    assert server["prefill_workspace_bytes_resolver"](8_192) > (
        direct["prefill_workspace_bytes_resolver"](8_192) + state
    )


def test_mtp_policy_off_disables_speculative_pricing(monkeypatch) -> None:
    generator = _route_generator(monkeypatch, mtp_policy="off")
    assert generator._mtp_serving_enabled_for_sizing() is False
    generator = _route_generator(monkeypatch, mtp_policy="auto")
    assert generator._mtp_serving_enabled_for_sizing() is True


def test_pinned_context_warning_does_not_claim_an_unfitting_context_fits(caplog) -> None:
    """A pinned context the model cannot fit is reported as such, with the fit."""

    from hipengine.server import api

    estimate = SimpleNamespace(
        kv_storage_dtype="bf16",
        kv_scale_dtype="fp32",
        max_batch_size=1,
        requested_context_tokens=32_768,
        model_max_context_tokens=262_144,
        allocatable_context_tokens=1_792,
        requested_kv_bytes=2 * 1024**3,
        requested_context_overhead_bytes=8 * 1024**3,
        requested_total_bytes=13 * 1024**3,
        bytes_per_token=404_064,
        usable_bytes=2 * 1024**3,
        reserve_bytes=3 * 1024**3,
        fits_requested=False,
        fits_model_max=False,
    )
    engine = SimpleNamespace(kv_capacity_estimate=estimate)
    with caplog.at_level("WARNING", logger=api._LOGGER.name):
        api._log_kv_capacity_summary(engine)
    warnings = [record.getMessage() for record in caplog.records if record.levelname == "WARNING"]
    assert any("requested context (32768 tokens)" in message and "fits 1792 tokens" in message for message in warnings)
    assert not any("is the largest that fits" in message for message in warnings)


def test_resident_scratch_is_priced_at_serving_capacity_but_kv_for_one_request() -> None:
    """The batch owner keeps recurrent state for every slot (about 1.2 GiB at C4)."""

    one = _breakdown(context_tokens=16_384, max_batch_size=1, allocate_kv_cache=False)
    four = _breakdown(
        context_tokens=16_384, max_batch_size=4, kv_request_slots=1, allocate_kv_cache=False
    )
    assert four.kv_pool_bytes == one.kv_pool_bytes
    runner = _dense_27b_runner()
    state = gguf_runner._gguf_packed_state_linear_bytes(runner, slot_count=1)
    assert four.scratch_bytes - one.scratch_bytes >= 3 * state


def _capacity_generator(monkeypatch, free_gib: float):
    from tests.test_unit_qwen35_gguf_capacity import (
        _auto_context_generator,
        _auto_context_runner,
    )

    generator = _auto_context_generator()
    runner = _auto_context_runner(free_gib=free_gib)
    monkeypatch.setattr(generator, "_get_shared_runner", lambda: runner, raising=False)
    return generator


def test_automatic_capacity_keeps_the_default_when_memory_allows(monkeypatch) -> None:
    generator = _capacity_generator(monkeypatch, free_gib=30.0)
    config = SimpleNamespace(speculative_mtp_serving="off", prefix_cache="off")
    assert generator.automatic_serving_capacity(config) == 4


def test_automatic_capacity_trades_slots_for_a_usable_context(monkeypatch) -> None:
    """On a tight card, fewer slots beats an unusably short context."""

    generator = _capacity_generator(monkeypatch, free_gib=4.0)
    config = SimpleNamespace(speculative_mtp_serving="off", prefix_cache="off")
    chosen = generator.automatic_serving_capacity(config)
    assert chosen in (1, 2)
    wide = generator._resident_capacity_estimate(
        generator._get_shared_runner(),
        max_batch_size=4,
        defer_kv_allocation=True,
        requested_context_tokens=None,
    )
    narrow = generator._resident_capacity_estimate(
        generator._get_shared_runner(),
        max_batch_size=chosen,
        defer_kv_allocation=True,
        requested_context_tokens=None,
    )
    assert narrow.allocatable_context_tokens > wide.allocatable_context_tokens
    # Sizing state does not leak into later estimates.
    assert getattr(generator, "_pending_sizing_loop_config", None) is None


def test_automatic_capacity_honours_a_pinned_context(monkeypatch) -> None:
    generator = _capacity_generator(monkeypatch, free_gib=30.0)
    config = SimpleNamespace(speculative_mtp_serving="off", prefix_cache="off")
    assert generator.automatic_serving_capacity(config, max_sequence_length=8_192) == 4
    tight = _capacity_generator(monkeypatch, free_gib=3.0)
    assert tight.automatic_serving_capacity(config, max_sequence_length=200_000) == 1


@pytest.mark.parametrize(
    "hint,expected",
    [
        (
            ("tail4_hadamard_group32", "fp32", "per_token_head"),
            ("int8_per_token_head", "tail4_hadamard_group32", "fp32", "hadamard_group32"),
        ),
        (
            ("int8_per_token_head", "fp32", "per_token_head"),
            ("int8_per_token_head", "uniform", "fp32", "per_token_head"),
        ),
        (("auto", "fp16", "per_token_head"), ("bf16", "uniform", "fp16", "per_token_head")),
    ],
    ids=["tail4-layout", "int8-uniform", "auto-bf16"],
)
def test_automatic_capacity_resolves_the_published_kv_hint(
    monkeypatch: pytest.MonkeyPatch,
    hint: tuple[str, str, str],
    expected: tuple[str, str, str, str],
) -> None:
    """Sizing accepts every storage name the server publishes.

    ``kv_storage`` names a layout as well as a dtype. ``tail4_hadamard_group32``
    resolves fine at load time, but sizing parsed the raw hint with
    ``DType.parse`` and failed the whole public load before any weights were
    read. Sizing now resolves the hint through the same policy resolver that
    produces the locked signature, and must choose the same capacity the locked
    signature would.
    """

    config = SimpleNamespace(speculative_mtp_serving="off", prefix_cache="off")
    generator = _capacity_generator(monkeypatch, free_gib=30.0)
    generator._prepared_kv_signature = expected
    locked_capacity = generator.automatic_serving_capacity(config)

    generator._prepared_kv_signature = None
    generator.request_kv_policy_hint = hint
    observed: list[tuple[str, str, str, str] | None] = []
    estimate = generator._resident_capacity_estimate

    def capture(*args, **kwargs):
        observed.append(generator._pending_sizing_kv_signature)
        return estimate(*args, **kwargs)

    monkeypatch.setattr(generator, "_resident_capacity_estimate", capture)

    assert generator.automatic_serving_capacity(config) == locked_capacity
    assert observed and observed[0] == expected
    assert getattr(generator, "_pending_sizing_kv_signature", None) is None


def test_refused_packed_decode_workspace_falls_back_to_row_decode() -> None:
    """A pool refusal for the packed decode workspace must not fail requests.

    On a 24 GB card the pages a prefix eviction frees only retire at the next
    loop barrier, so the packed workspace can be refused for one step; the rows
    then decode individually instead of ending their requests.
    """

    from collections import Counter

    from hipengine.runtime.memory_admission import (
        MemoryAdmissionRefused,
        price_memory_admission,
    )

    refusal = MemoryAdmissionRefused(
        price_memory_admission(free_bytes=1, consumers={"private_workspace": 2}, reserve_bytes=0)
    )

    class Owner:
        def step_batch_native(self, token_ids, **kwargs):
            raise refusal

    owner = Owner()
    sessions = (object(), object())
    slots = [
        SimpleNamespace(session=session, prev_token=1, seq_position=10, generated_ids=[2])
        for session in sessions
    ]
    rows = [
        SimpleNamespace(
            request_id=index,
            slot=slot,
            native_greedy=True,
            native_sampled=False,
            native_sampler=False,
            request=SimpleNamespace(max_tokens=8),
        )
        for index, slot in enumerate(slots)
    ]
    runner = qwen35_gguf.Qwen35GGUFResidentModelRunner.__new__(
        qwen35_gguf.Qwen35GGUFResidentModelRunner
    )
    runner.generator = SimpleNamespace(
        _flush_ar_packed_decode_owners_if_chunk_changed=lambda chunk: None,
    )
    runner._fallback_reasons = Counter()
    runner._route_counts = Counter()
    runner._close_c1_decode_graph = lambda row: None
    runner._close_packed_decode_graphs = lambda rows: None
    runner._packed_execution_owner = lambda session: owner

    assert runner._step_native_chunk(rows, allow_graph=False) is False
    assert runner._fallback_reasons["packed_decode_workspace_refused"] == 1


def test_dms_dense_prefill_kv_is_not_priced_twice() -> None:
    """DMS owns its dense prefill KV in the session scratch; no pool KV on top."""

    dms = _breakdown(context_tokens=32_768, allocate_kv_cache=True, kv_request_slots=0)
    pooled = _breakdown(context_tokens=32_768, allocate_kv_cache=True, kv_request_slots=1)
    assert dms.kv_pool_bytes == 0
    assert pooled.total_bytes - dms.total_bytes == pooled.kv_pool_bytes


@pytest.mark.parametrize("positions,bulk_bytes,active_layers", [
    (256, 1 << 40, 1), (8192, 1 << 40, 1), (32768, 1 << 40, 1),
    (131072, 0, 16),
])
def test_dms_compact_sizing_prices_codec_and_bounded_normalization(
    monkeypatch, positions, bulk_bytes, active_layers
):
    # Force either the prefill phase (one device layer) or the restored-store
    # phase (all layers) to dominate. These phases must not be summed.
    monkeypatch.setattr(gguf_runner, "_gguf_prefill_scratch_nbytes", lambda *a, **kw: bulk_bytes)
    monkeypatch.setattr(gguf_runner, "_gguf_aotriton_head_major_kv_scratch_nbytes", lambda **kw: 0)
    generator = _route_generator(monkeypatch)
    runner = _dense_27b_runner()
    speculative = lambda n: 123
    seed = {"speculative_bytes_resolver": speculative}
    costs = {}
    for storage in ("bf16", "int8_per_token_head"):
        generator._prepared_kv_signature = (storage, "uniform", "fp32", "per_token_head")
        terms = generator._dms_layer_outer_capacity_terms(runner, seed, mtp_sized=True)
        assert terms["speculative_bytes_resolver"] is speculative
        assert terms["transient_bytes_resolver"](positions) == 0
        costs[storage] = terms["prefill_workspace_bytes_resolver"](positions)
    cfg = runner.weights.config
    assert costs["bf16"] - costs["int8_per_token_head"] == (
        active_layers * cfg.head_count_kv * (positions + 4) * (2 * cfg.key_length - 8)
    )
    without_mtp = generator._dms_layer_outer_capacity_terms(runner, seed, mtp_sized=False)
    assert costs["int8_per_token_head"] - without_mtp["prefill_workspace_bytes_resolver"](positions) == (
        2 * min(positions, 4096) * runner.hidden_size
        + 2 * active_layers * cfg.head_count_kv * (2 * cfg.key_length + 13)
    )


def test_dms_compact_estimate_has_no_dense_kv_allocation(monkeypatch):
    generator = _route_generator(monkeypatch)
    generator.dms_metadata_path = "/tmp/dms_metadata.json"
    generator._prepared_kv_signature = ("int8_per_token_head", "uniform", "fp32", "per_token_head")
    runner = _dense_27b_runner()
    runner.runtime = SimpleNamespace(mem_get_info=lambda: (12 * 1024**3, 24 * 1024**3))
    monkeypatch.setattr(generator, "_shared_dense_mtp_nextn_weights", lambda *a, **kw: None)
    captured = {}
    monkeypatch.setattr(qwen35_gguf, "estimate_qwen35_gguf_kv_capacity",
                        lambda cfg, **kw: captured.update(kw) or object())
    assert generator._resident_capacity_estimate(
        runner, max_batch_size=1, defer_kv_allocation=True,
        requested_context_tokens=32768,
    ) is not None
    assert captured["allocate_kv_cache"] is False
    assert captured["kv_request_slots"] == 0
    assert captured["workspace_lease_needed"] is False
    assert captured["kv_storage_dtype"] == "int8_per_token_head"
    assert captured["transient_bytes_resolver"](32768) == 0
    assert captured["prefill_workspace_bytes_resolver"](32768) > 0


@pytest.mark.parametrize("dms", [True, False])
def test_dms_sizing_loads_mtp_draft_before_reading_free_memory(monkeypatch, dms) -> None:
    # DMS serves singleton NextN MTP; an unpriced draft OOMed its MTP warmup.
    generator = _route_generator(monkeypatch)
    generator.dms_metadata_path = "/tmp/dms_metadata.json" if dms else None
    calls = []
    monkeypatch.setattr(
        generator, "_mtp_serving_enabled_for_sizing", lambda: True, raising=False
    )
    monkeypatch.setattr(
        generator,
        "_shared_dense_mtp_nextn_weights",
        lambda *args, **kwargs: calls.append("draft"),
        raising=False,
    )

    def mem_get_info():
        calls.append("free")
        raise RuntimeError("stop after the free-memory reading")

    runner = SimpleNamespace(
        weights=SimpleNamespace(config=object()),
        runtime=SimpleNamespace(mem_get_info=mem_get_info),
    )
    assert (
        generator._resident_capacity_estimate(
            runner,
            max_batch_size=1,
            defer_kv_allocation=False,
            requested_context_tokens=None,
        )
        is None
    )
    assert calls == (["draft", "free"] if dms else ["free"])


@pytest.mark.parametrize("positions", [4096, 131072, 229376])
def test_dms_exact_budget_store_is_priced_by_retained_rows(monkeypatch, positions):
    """Outputs are clamped to the store, so it holds the context's retained rows."""
    monkeypatch.setattr(gguf_runner, "_gguf_prefill_scratch_nbytes", lambda *a, **kw: 0)
    monkeypatch.setattr(gguf_runner, "_gguf_aotriton_head_major_kv_scratch_nbytes", lambda **kw: 0)
    generator = _route_generator(monkeypatch)
    generator._prepared_kv_signature = ("int8_per_token_head", "uniform", "fp32", "per_token_head")
    runner = _dense_27b_runner()
    cfg = runner.weights.config
    seed = {"speculative_bytes_resolver": lambda n: 0}
    generator.__dict__["_dms_retention_cache"] = (None,)
    full = generator._dms_layer_outer_capacity_terms(runner, seed, mtp_sized=True)
    generator.__dict__["_dms_retention_cache"] = ((8192, 2),)
    bounded = generator._dms_layer_outer_capacity_terms(runner, seed, mtp_sized=True)
    retained = qwen35_gguf._dms_exact_budget_retained_rows(positions, window=8192, ratio=2)
    layers = sum(kind == "full_attention" for kind in cfg.layer_types)
    saved = (full["prefill_workspace_bytes_resolver"](positions)
             - bounded["prefill_workspace_bytes_resolver"](positions))
    # The decode split workspace also scales with the per-head slot count.
    split_rows = lambda per_head: (per_head + 255) // 256
    split = 4 * cfg.head_count * (split_rows(positions + 4) - split_rows(retained + 4)) * (cfg.key_length + 2)
    assert saved == layers * cfg.head_count_kv * (positions - retained) * (2 * cfg.key_length + 13) + split


@pytest.mark.parametrize("capacity", [2, 4])
def test_packed_linear_state_is_not_reserved_twice(monkeypatch, capacity):
    """The pool charges the packed verify linear state to its own budget.

    It belongs in the context sizing, but the pool's execution reserve (the
    memory kept outside the pool) must not hold it a second time: on a 24 GB
    card the double charge left a two-slot pool unable to admit one request
    at its advertised context.
    """
    sentinel = 10**12
    real = gguf_runner._gguf_packed_state_linear_bytes

    def priced(runner, *, slot_count):
        return sentinel if int(slot_count) == capacity else real(runner, slot_count=slot_count)

    # Capacity 1 shares ``slot_count=1`` with the prefix-snapshot price.
    monkeypatch.setattr(gguf_runner, "_gguf_packed_state_linear_bytes", priced)
    terms = _route_terms(monkeypatch, storage="bf16", capacity=capacity, mtp=True)
    assert terms["prefill_workspace_bytes_resolver"](32_768) >= sentinel
    assert terms["_packed_scratch_bytes_resolver"](32_768) < sentinel
