from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from hipengine.core.dtype import DType
from hipengine.core.memory import DeviceBuffer
from hipengine.core.specdec2_scope import moe_physical_c2_numerics_session
from hipengine.generation.qwen35_gguf_moe_mtp2 import (
    Qwen35GGUFMoEMTP2Adapter,
    _MoeTargetHiddenSink,
    create_qwen35_gguf_moe_mtp2_adapter,
)
from hipengine.generation.qwen35_gguf_mtp2 import Qwen35GGUFMTP2Adapter
from hipengine.generation.qwen35_gguf_mtp2_registry import (
    register_builtin_gguf_mtp2_adapters,
    resolve_gguf_mtp2_adapter,
)
from hipengine.speculative.provider import SpeculativeRequestSemantics
from hipengine.runtime.qwen35_gguf_runner import _gguf_verify_f32_residual_enabled


class _Runtime:
    def __init__(self) -> None:
        self.copies = []
        self.syncs = []

    def memcpy_async(self, dst, src, nbytes, kind, stream):
        self.copies.append((int(dst), int(src), int(nbytes), kind, int(stream)))

    def stream_synchronize(self, stream):
        self.syncs.append(int(stream))


def test_moe_physical_c2_numerics_scope_is_request_local(monkeypatch) -> None:
    monkeypatch.setenv("HIPENGINE_GGUF_VERIFY_F32_RESIDUAL", "1")

    assert _gguf_verify_f32_residual_enabled() is True
    with moe_physical_c2_numerics_session(True):
        assert _gguf_verify_f32_residual_enabled() is False
    assert _gguf_verify_f32_residual_enabled() is True


def test_moe_target_hidden_sink_preserves_rows_and_publishes_final_target() -> None:
    runtime = _Runtime()
    target = SimpleNamespace(
        runtime=runtime,
        scratch=SimpleNamespace(hidden_seed_fp32=DeviceBuffer(0x7000, 16)),
        _hidden_seed_fp32_populated=False,
        _last_target_hidden_ptr=0,
    )
    norm_calls = []

    def run_norm(src, dst, *, stream, capture_hidden_seed_fp32):
        norm_calls.append((src, dst, stream, capture_hidden_seed_fp32))

    target._run_output_norm_hidden = run_norm
    sink = _MoeTargetHiddenSink(
        request_id=7,
        hidden_size=4,
        total_rows=3,
        target=target,
        buffer=DeviceBuffer(0x1000, 3 * 4 * DType.BF16.itemsize),
        normalized=DeviceBuffer(0x2000, 3 * 4 * DType.FP32.itemsize),
        normalized_bf16=DeviceBuffer(0x3000, 4 * DType.BF16.itemsize),
    )

    sink.consume(request_id=7, chunk_start=1, hidden_ptr=0x5000, rows=2, stream=9)
    sink.finish(request_id=7, total_rows=3, stream=9)

    assert runtime.copies[0][:3] == (0x1008, 0x5000, 16)
    assert [call[0] for call in norm_calls] == [0x1000, 0x1008, 0x1010]
    assert target._hidden_seed_fp32_populated is True
    assert target._last_target_hidden_ptr == 0x1010
    assert sink.finished is True


@pytest.mark.parametrize("fail_index", (0, 1))
def test_shared_dense_kv_allocation_rolls_back_partial_failure(monkeypatch, fail_index):
    from hipengine.core import memory
    from hipengine.generation.qwen35_gguf import _allocate_mtp_dense_kv

    allocated, freed = [], []
    runtime = object()

    def allocate(nbytes, *, runtime):
        if len(allocated) == fail_index:
            raise MemoryError("injected draft allocation failure")
        buffer = SimpleNamespace(ptr=17, nbytes=nbytes)
        allocated.append(buffer)
        return buffer

    monkeypatch.setattr(memory, "malloc", allocate)
    monkeypatch.setattr(memory, "free", lambda buffer, **kwargs: freed.append(buffer))
    with pytest.raises(MemoryError, match="injected draft allocation failure"):
        _allocate_mtp_dense_kv(runtime=runtime, capacity=37, qk_head_dim=128)
    assert freed == list(reversed(allocated))


def test_moe_close_detaches_slab_without_a_synthetic_request(monkeypatch) -> None:
    from hipengine.generation import qwen35_gguf_moe_mtp2 as module

    adapter = object.__new__(Qwen35GGUFMoEMTP2Adapter)
    adapter.owner = SimpleNamespace()
    adapter._states = {}
    adapter._prompt_sinks = {}
    adapter._intents = {}
    adapter._disabled_requests = set()
    adapter._batch_accept_workspace = None
    adapter._resident_charge_tokens = {}
    adapter._draft_kv_slabs = {}
    target = SimpleNamespace(runtime=object(), runner=SimpleNamespace(hidden_size=4),
                             _last_target_hidden_ptr=0x1008)
    buffers = tuple(DeviceBuffer(ptr, 32) for ptr in (0x1000, 0x2000, 0x3000))
    adapter._target_hidden_slabs = {id(target): (target, *buffers)}
    freed = []
    monkeypatch.setattr(module, 'free', lambda buffer, **kwargs: freed.append(buffer.ptr))

    adapter.close()
    adapter.close()

    assert target._last_target_hidden_ptr == 0
    assert freed == [0x3000, 0x2000, 0x1000]
    assert adapter._target_hidden_slabs == {}


def _capability_for(
    adapter: Qwen35GGUFMoEMTP2Adapter,
    context_tokens: int,
) -> SpeculativeCapability | None:
    return adapter.capability(
        (
            SpeculativeRequestSemantics(
                7, "greedy", "verify_chain", int(context_tokens), 8
            ),
        )
    )


def _context_only_adapter(
    *,
    backend: str,
    session_rows: int,
    candidate_budget: int = 2,
    position: int = 6,
    scratch_positions: int | None = None,
) -> Qwen35GGUFMoEMTP2Adapter:
    """A C1 adapter whose only moving parts are position and capability.

    ``scratch_positions`` gives the target a resident session, which is what the
    reusable target graph's context-bucket selector reads; a target without one
    models the retained short graph and admits every cycle.
    """

    adapter = object.__new__(Qwen35GGUFMoEMTP2Adapter)
    adapter.enabled = True
    adapter.candidate_budget = candidate_budget
    adapter.quant = "gguf_q4_k_m"
    adapter.target_verify_mode = "native"
    adapter._disabled_requests = set()
    adapter._states = {7: object()}
    target = SimpleNamespace(
        position=int(position),
        backend=str(backend),
        target_layout=SimpleNamespace(max_sequence_length=int(session_rows)),
        kv_storage_dtype="bf16",
        runner=SimpleNamespace(hidden_size=4, weights=None),
    )
    if scratch_positions is not None:
        target.scratch = SimpleNamespace(
            max_positions=int(scratch_positions), block_size=256
        )
    row = SimpleNamespace(
        native_greedy=True,
        first_token_emitted=True,
        lease=SimpleNamespace(session=target),
        slot=object(),
    )
    adapter.owner = SimpleNamespace(capacity=1, _row=lambda request_id: row)
    adapter.generator = SimpleNamespace(
        backend=str(backend),
        execution_profile=SimpleNamespace(value="strict"),
    )
    return adapter


def _move_to(adapter: Qwen35GGUFMoEMTP2Adapter, position: int) -> None:
    """Advance the fake target and the context the planner would report."""

    adapter.owner._row(7).lease.session.position = int(position)



def test_moe_adapter_capability_is_c1_k2_at_the_declared_ceiling() -> None:
    adapter = _context_only_adapter(backend="hip_gfx1100", session_rows=1024)

    capability = _capability_for(adapter, 6)

    assert capability is not None
    assert capability.target_key == "qwen_moe_gguf"
    assert capability.provider_key == "qwen_nextn_moe"
    assert capability.max_requests == 1
    assert capability.max_candidates_per_request == 2
    assert capability.proposal_widths == (1,)
    assert capability.graph_supported is True
    # gfx1100 declares a 95-position native complete-cycle ceiling, so that is
    # the advertised bound. Admission no longer subtracts the candidate budget
    # from it: depth selection belongs to the shared planner.
    assert capability.max_context_tokens == 95
    # A longer context is admitted by capability: the planner, not admission,
    # decides whether a cycle still fits at that position.
    for context_tokens in (37, 92, 93, 94, 95, 96, 1025, 1500):
        assert _capability_for(adapter, context_tokens) is not None


def test_moe_capability_ceiling_is_declared_capability_and_session_capacity() -> None:
    """The advertised ceiling is the smaller of the declared bound and the session."""

    # gfx1151 declares 65544, so a 2048-position session is the binding bound.
    wide = _context_only_adapter(backend="hip_gfx1151", session_rows=2048)
    assert _capability_for(wide, 6).max_context_tokens == 2048
    # A short session binds even where the backend declares a long ceiling.
    short = _context_only_adapter(backend="hip_gfx1151", session_rows=512)
    assert _capability_for(short, 6).max_context_tokens == 512
    # gfx1100 declares 95, so the session does not raise it.
    narrow = _context_only_adapter(backend="hip_gfx1100", session_rows=2048)
    assert _capability_for(narrow, 6).max_context_tokens == 95


def test_moe_capability_ceiling_does_not_subtract_the_candidate_budget() -> None:
    """Admission and the shared planner must agree on the boundary."""

    adapter = _context_only_adapter(backend="hip_gfx1151", session_rows=1024)
    adapter.candidate_budget = 4

    capability = _capability_for(adapter, 6)

    assert capability is not None
    assert capability.max_candidates_per_request == 4
    assert capability.max_context_tokens == 1024


@pytest.mark.parametrize(
    "backend, session_rows, prompt_len, fits",
    (
        # gfx1100 declares 95 verifiable positions: the first plan needs
        # ``95 - (prompt_len + 1) - 1 > 0``, so the boundary is prompt_len 92.
        ("hip_gfx1100", 2048, 37, True),
        ("hip_gfx1100", 2048, 92, True),
        ("hip_gfx1100", 2048, 93, False),
        ("hip_gfx1100", 2048, 94, False),
        ("hip_gfx1100", 2048, 95, False),
        ("hip_gfx1100", 2048, 96, False),
        ("hip_gfx1100", 2048, 1025, False),
        ("hip_gfx1100", 2048, 1500, False),
        # gfx1151 declares 65544, so a 2048-position session admits all of them.
        ("hip_gfx1151", 2048, 37, True),
        ("hip_gfx1151", 2048, 92, True),
        ("hip_gfx1151", 2048, 93, True),
        ("hip_gfx1151", 2048, 94, True),
        ("hip_gfx1151", 2048, 95, True),
        ("hip_gfx1151", 2048, 96, True),
        ("hip_gfx1151", 2048, 1025, True),
        ("hip_gfx1151", 2048, 1500, True),
        ("hip_gfx1151", 2048, 2045, True),
        ("hip_gfx1151", 2048, 2046, False),
    ),
)
def test_moe_admission_boundary_is_the_planners_k0_boundary(
    backend: str,
    session_rows: int,
    prompt_len: int,
    fits: bool,
) -> None:
    """The adapter's prefill refusal agrees with ``plan_speculative_requests``.

    The planner decides whether a cycle runs, so the adapter must refuse exactly
    where the planner reports ``TARGET_GRAPH_CONTEXT_BUCKET_MISS``. This drives
    the real planner against the capability this adapter advertises at that
    prompt length and compares its verdict with the adapter's own room
    expression, which is what the prefill refusal reads.
    """

    from hipengine.speculative.policy import plan_speculative_requests

    adapter = _context_only_adapter(backend=backend, session_rows=session_rows)
    ceiling = adapter._target_context_ceiling(
        adapter.owner._row(7).lease.session
    )
    context_tokens = int(prompt_len) + 1
    adapter._states = {7: object()}
    capability = adapter.capability(
        (
            SpeculativeRequestSemantics(
                7, "greedy", "verify_chain", context_tokens, 8
            ),
        )
    )
    assert capability is not None

    plan = plan_speculative_requests(
        capability,
        (
            SpeculativeRequestSemantics(
                7, "greedy", "verify_chain", context_tokens, 8
            ),
        ),
        resident_slots=(0,),
        desired_candidate_counts=(capability.max_candidates_per_request,),
        operation_id="op",
        cycle_id=1,
        context_bucket_size=64,
    )

    adapter_room = adapter._planned_context_room(ceiling, context_tokens)
    assert (adapter_room > 0) is fits
    assert (plan.candidate_counts[0] > 0) is fits
    if fits:
        assert plan.candidate_counts[0] == min(
            capability.max_candidates_per_request,
            adapter_room,
            # output room: remaining_decode - 1
            7,
        )


def test_moe_admission_reports_the_deepest_depth_with_a_graph_bucket() -> None:
    """A schedule boundary narrows depth; it does not end the MTP route.

    The reusable target graph runs one attention schedule for every row, so a
    cycle whose rows span the short/split-K transition (1024) or two split
    workspaces (every 256 positions above it) has no legal bucket. Admission
    reports the deepest depth that does fit and the shared planner selects from
    it, so a shallower legal cycle still runs. Only a step where even one
    candidate has no bucket declines, and that step is planned as a K0
    transition that advances the provider seed.
    """

    adapter = _context_only_adapter(
        backend="hip_gfx1151", session_rows=2048, scratch_positions=2048
    )
    target = adapter.owner._row(7).lease.session

    cases = (
        # position, deepest candidate depth with a legal bucket (0 = K0)
        (37, 2),
        (1020, 2),   # rows end at 1023, inside the short bucket
        (1021, 1),   # depth 2 straddles 1024; depth 1 still fits
        (1022, 0),   # even one candidate straddles the transition
        (1023, 0),
        (1024, 2),   # first long-context bucket, 1280
        (1100, 2),
        (1277, 2),   # rows end at 1280, one split workspace
        (1278, 1),   # depth 2 straddles the 1280 boundary
        (1279, 0),
        (1280, 2),
    )
    for position, expected_depth in cases:
        _move_to(adapter, position)
        assert (
            adapter._bucket_candidate_ceiling(target, limit=2) == expected_depth
        ), position
        capability = _capability_for(adapter, position + 1)
        if expected_depth == 0:
            assert capability is None, position
        else:
            assert capability is not None, position
            # Depth selection stays with the planner, which takes the minimum
            # of this bound, the desired depth, output room and context room.
            assert capability.max_candidates_per_request == expected_depth, position
            assert capability.max_frontier_rows == expected_depth + 1, position
        # A declined step never disables the row or drops its provider.
        assert adapter._disabled_requests == set(), position
        assert 7 in adapter._states, position

    # The step after a boundary is admitted again, so MTP resumes by itself.
    _move_to(adapter, 1279)
    assert _capability_for(adapter, 1280) is None
    _move_to(adapter, 1280)
    assert _capability_for(adapter, 1281) is not None


def test_moe_cycle_asks_for_the_attention_schedule_its_bucket_declares() -> None:
    """Long-context cycles must request split-K native attention, short ones keep bulk."""

    from hipengine.runtime.gguf_native_spec_cycle import (
        native_target_graph_context_bucket,
    )

    short = _context_only_adapter(
        backend="hip_gfx1151", session_rows=2048, scratch_positions=2048, position=37
    )
    long = _context_only_adapter(
        backend="hip_gfx1151", session_rows=2048, scratch_positions=2048, position=1100
    )
    target_short = short.owner._row(7).lease.session
    target_long = long.owner._row(7).lease.session
    rows = int(short.candidate_budget) + 1

    assert native_target_graph_context_bucket(target_short, rows=rows) == 1023
    assert short._target_attention_mode(target_short, rows=rows) == "bulk"
    assert native_target_graph_context_bucket(target_long, rows=rows) == 1280
    assert long._target_attention_mode(target_long, rows=rows) == "native"

    # A straddling cycle has no bucket, so it is never launched with either
    # schedule; the bucket selector is the single source for both answers.
    _move_to(long, 1022)
    assert native_target_graph_context_bucket(target_long, rows=rows) is None
    assert long._bucket_candidate_ceiling(target_long, limit=2) == 0


def test_moe_adapter_capability_exposes_exact_physical_c2() -> None:
    adapter = object.__new__(Qwen35GGUFMoEMTP2Adapter)
    adapter.enabled = True
    adapter.candidate_budget = 2
    adapter.quant = "gguf_q4_k_m"
    adapter.target_verify_mode = "native"
    adapter._disabled_requests = set()
    adapter._states = {7: object(), 8: object()}
    targets = {
        rid: SimpleNamespace(
            position=6,
            target_layout=SimpleNamespace(max_sequence_length=1024),
            kv_storage_dtype="bf16",
        )
        for rid in (7, 8)
    }
    rows = {
        rid: SimpleNamespace(
            native_greedy=True,
            first_token_emitted=True,
            lease=SimpleNamespace(session=targets[rid]),
            slot=object(),
        )
        for rid in targets
    }
    adapter.owner = SimpleNamespace(
        capacity=2,
        _row=lambda request_id: rows[int(request_id)],
    )
    adapter.generator = SimpleNamespace(
        backend="hip_gfx1100",
        execution_profile=SimpleNamespace(value="production"),
    )
    one = SpeculativeRequestSemantics(7, "greedy", "verify_chain", 6, 8)
    two = SpeculativeRequestSemantics(8, "greedy", "verify_chain", 6, 8)

    assert adapter.capability((one,)) is None
    capability = adapter.capability((one, two))
    assert capability is not None
    assert adapter.staged_frontier is True
    assert capability.max_requests == 2
    assert capability.max_frontier_rows == 6
    assert capability.proposal_widths == (1, 2)
    adapter.generator.execution_profile = SimpleNamespace(value="strict")
    assert adapter.capability((one, two)) is None


def test_moe_adapter_accepts_generic_static_eligibility_registration() -> None:
    adapter = object.__new__(Qwen35GGUFMoEMTP2Adapter)
    adapter.candidate_budget = 2
    adapter._intents = {}
    adapter._disabled_requests = {7}

    adapter.register_request(7, 3, static_eligibility=object())

    assert adapter._intents == {7: 2}
    assert adapter._disabled_requests == set()


class _PrefillHarness:
    """A C1 MoE prefill driven through the adapter's real begin/observe pair.

    The sink comes from ``begin_prompt_streaming`` rather than being built by
    hand, so every test here exercises the same slab-growth path a prompt takes.
    Device work is recorded instead of executed: mallocs are counted, frees are
    recorded by pointer, and the dense draft-KV allocator returns buffers whose
    real byte size is what the write guard reads.
    """

    def __init__(
        self,
        monkeypatch,
        *,
        backend: str = "hip_gfx1151",
        session_rows: int = 2048,
        hidden_size: int = 4,
        refuse_charge: str | None = None,
        fail_malloc_at: int | None = None,
        short_kv_rows: int | None = None,
    ) -> None:
        from hipengine.generation import qwen35_gguf as gguf

        self.module = gguf
        self.mallocs: list[int] = []
        self.allocations: list[tuple[int, int]] = []
        self.freed: list[int] = []
        self.next_ptr = 0x100000
        self.fail_malloc_at = fail_malloc_at
        self.kv_allocations: list[int] = []
        self.hidden_allocations: list[int] = []
        self.released_runners: list[object] = []
        self.short_kv_rows = short_kv_rows

        def fake_malloc(nbytes, *, runtime=None):
            if self.fail_malloc_at is not None and len(self.mallocs) == self.fail_malloc_at:
                raise MemoryError("synthetic allocation failure")
            self.mallocs.append(int(nbytes))
            buffer = DeviceBuffer(self.next_ptr, int(nbytes))
            self.allocations.append((int(self.next_ptr), int(nbytes)))
            self.next_ptr += int(nbytes)
            return buffer

        def fake_free(buffer, *, runtime=None):
            self.freed.append(int(buffer.ptr))

        def fake_allocate_mtp_dense_kv(*, runtime, capacity, qk_head_dim, kv_heads):
            rows = int(capacity) if self.short_kv_rows is None else int(self.short_kv_rows)
            self.kv_allocations.append(rows)
            row_nbytes = int(kv_heads) * int(qk_head_dim) * DType.FP32.itemsize
            key_cache = DeviceBuffer(self.next_ptr, rows * row_nbytes)
            value_cache = DeviceBuffer(
                self.next_ptr + rows * row_nbytes, rows * row_nbytes
            )
            self.next_ptr += 2 * rows * row_nbytes
            return key_cache, value_cache, ()

        monkeypatch.setattr(
            "hipengine.generation.qwen35_gguf_moe_mtp2.malloc", fake_malloc
        )
        monkeypatch.setattr(
            "hipengine.generation.qwen35_gguf_moe_mtp2.free", fake_free
        )
        if short_kv_rows is not None:
            # An allocator that returns less than it was asked for: the slab is
            # smaller than the request, which the write guard must refuse.
            real_nbytes = gguf._mtp_dense_kv_nbytes
            short_row_bytes = 2 * 128 * DType.FP32.itemsize
            monkeypatch.setattr(
                gguf,
                "_mtp_dense_kv_nbytes",
                lambda *, capacity, qk_head_dim, kv_heads=2: min(
                    real_nbytes(
                        capacity=int(capacity),
                        qk_head_dim=int(qk_head_dim),
                        kv_heads=int(kv_heads),
                    ),
                    int(short_kv_rows) * short_row_bytes,
                ),
            )
        monkeypatch.setattr(gguf, "_new_mtp_context", lambda *a, **k: object())
        # Record the row count of every real slab allocation; the adapter owns
        # both allocations itself.
        original_ensure_hidden = Qwen35GGUFMoEMTP2Adapter._ensure_target_hidden_slab
        original_ensure_kv = Qwen35GGUFMoEMTP2Adapter._ensure_draft_kv_slab
        harness = self

        def recording_ensure_hidden(adapter, target, target_key, rows):
            before = len(harness.mallocs)
            result = original_ensure_hidden(adapter, target, target_key, rows)
            if len(harness.mallocs) > before:
                harness.hidden_allocations.append(int(rows))
            return result

        def recording_ensure_kv(adapter, target, target_key, *, draft, rows):
            before = len(harness.mallocs)
            result = original_ensure_kv(
                adapter, target, target_key, draft=draft, rows=rows
            )
            if len(harness.mallocs) > before:
                harness.kv_allocations.append(int(rows))
            return result

        monkeypatch.setattr(
            Qwen35GGUFMoEMTP2Adapter,
            "_ensure_target_hidden_slab",
            recording_ensure_hidden,
        )
        monkeypatch.setattr(
            Qwen35GGUFMoEMTP2Adapter, "_ensure_draft_kv_slab", recording_ensure_kv
        )

        self.pool = _RecordingPool(refuse=refuse_charge)
        self.runtime = _Runtime()
        self.target = SimpleNamespace(
            runtime=self.runtime,
            backend=str(backend),
            runner=SimpleNamespace(hidden_size=hidden_size),
            target_layout=SimpleNamespace(max_sequence_length=int(session_rows)),
            position=0,
            _last_target_hidden_ptr=0,
            kv_storage_dtype="bf16",
            run_native_spec_mtp_cycle=self._run_cycle,
        )
        self.draft = SimpleNamespace(
            num_kv_heads=2,
            qk_head_dim=128,
            write_calls=[],
            write_kv_rows_from_device_seed_base=self._write_rows,
        )
        self.rows: dict[int, Any] = {}
        self.cycle_calls: list[tuple[Any, Any, dict[str, Any]]] = []
        self.adapter = object.__new__(Qwen35GGUFMoEMTP2Adapter)
        self.adapter.enabled = True
        self.adapter.candidate_budget = 2
        self.adapter.quant = "gguf_q4_k_m"
        self.adapter.target_verify_mode = "native"
        self.adapter._intents = {}
        self.adapter._disabled_requests = set()
        self.adapter._states = {}
        self.adapter._prompt_sinks = {}
        self.adapter._assets = None
        self.adapter._draft_kv_slabs = {}
        self.adapter._target_hidden_slabs = {}
        self.adapter._resident_charge_tokens = {}
        self.adapter._transaction_sequence = 0
        self.adapter._sampled_accept_candidate = lambda request_id: False
        self.adapter.owner = SimpleNamespace(
            capacity=1,
            _row=self._row,
            _shared_runner=None,
            _kv_pool=self.pool,
            _flush_row_owner=lambda row: None,
        )
        self.adapter.generator = SimpleNamespace(
            backend=str(backend),
            execution_profile=SimpleNamespace(value="strict"),
            native_sampler_algorithm="sorted",
            _load_mtp_serving_assets=lambda: SimpleNamespace(
                rope_cos=DeviceBuffer(0x900000, 8),
                rope_sin=DeviceBuffer(0x900008, 8),
            ),
            _acquire_mtp_draft_runner=lambda assets, *, runtime, pool_enabled: (
                self.draft,
                "pool-key",
                False,
            ),
            _release_mtp_draft_runner=lambda pool_key, draft: (
                self.released_runners.append(pool_key)
            ),
        )

    def _write_rows(self, *args, **kwargs):
        self.draft.write_calls.append((args, kwargs))
        return len(args[1])

    def _run_cycle(self, draft, context, **kwargs):
        """Stand in for the native target cycle, recording what it was asked.

        The device work is not the subject here: the cycle's *contract* is, so
        this returns a self-consistent result (one accepted draft, its
        correction) and advances the fake target exactly as the real cycle does.
        """

        self.cycle_calls.append((draft, context, dict(kwargs)))
        start = int(kwargs["root_position"])
        self.target.position = start + 2
        return SimpleNamespace(
            draft_cache_len_after=int(kwargs["draft_cache_len"]) + 1,
            output_token_ids=(11, 12),
            accepted_draft_tokens=1,
            end_position=start + 2,
            proposal_wall_ms=0.1,
            target_wall_ms=0.1,
            mtp_kv_commit_wall_ms=0.0,
        )

    def cycle(self, request_id: int = 7, *, budget: int = 2) -> dict[str, Any]:
        """Drive one complete cycle and return the kwargs it was launched with."""

        from hipengine.speculative.frontier import (
            SpecK0Class,
            SpecPlanReason,
            SpecRequestPlan,
            SpecTransactionMode,
        )

        rid = int(request_id)
        state = self.adapter._states[rid]
        row = self.rows[rid]
        row.slot = SimpleNamespace(
            generated_ids=[1],
            prev_token=1,
            seq_position=int(self.target.position),
            native_decode_steps=0,
            done=False,
        )
        self.adapter._intents[rid] = int(budget)
        plan = SpecRequestPlan(
            operation_id="op",
            cycle_id=1,
            request_ids=(rid,),
            resident_slots=(0,),
            candidate_counts=(int(budget),),
            reasons=(SpecPlanReason.SPECULATIVE_QUALIFIED,),
            k0_classes=(SpecK0Class.NOT_K0,),
            mode="verify_chain",
            capability_key="cap",
            provider_key="prov",
            target_transaction_mode=SpecTransactionMode.REVERSIBLE_JOURNAL,
            provider_transaction_mode=SpecTransactionMode.REVERSIBLE_JOURNAL,
            proposal_widths=(1,),
            target_row_decomposition=(int(budget) + 1,),
            context_bucket_size=1024,
            execution_route="graph",
            declared_logical_c=1,
        )
        state.cache_len = int(self.target.position)
        self.adapter.execute_cycle(plan, commit=True)
        return self.cycle_calls[-1][2]

    def freed_of(self, *buffers) -> list[int]:
        """Frees of exactly these buffers, in order, ignoring priming scratch."""

        wanted = {int(buffer.ptr) for buffer in buffers}
        return [ptr for ptr in self.freed if ptr in wanted]

    def _row(self, request_id: int):
        return self.rows.get(int(request_id))

    def add_row(
        self,
        request_id: int,
        prompt_len: int,
        *,
        max_tokens: int = 16,
        automatic: bool = False,
    ) -> Any:
        row = SimpleNamespace(
            request_id=int(request_id),
            prompt_ids=tuple(range(int(prompt_len))),
            request=SimpleNamespace(
                max_tokens=int(max_tokens),
                speculative_mtp_automatic=bool(automatic),
            ),
            lease=SimpleNamespace(session=self.target),
            native_greedy=True,
            first_token_emitted=False,
            slot=None,
            mtp2_prompt_fallback_reason=None,
            mtp2_prompt_streaming=False,
            mtp2_cycles=0,
            mtp2_proposal_ms=0.0,
            mtp2_target_ms=0.0,
            mtp2_provider_update_ms=0.0,
            mtp2_execution_routes=[],
        )
        self.rows[int(request_id)] = row
        self.adapter._intents[int(request_id)] = 2
        return row

    def prefill(
        self,
        request_id: int,
        prompt_len: int,
        *,
        finish_previous: bool = True,
        **kwargs,
    ) -> None:
        """Prime one request. One request owns a target session at a time."""

        if finish_previous:
            for live in tuple(self.adapter._states):
                self.adapter.release_request(live)
        row = self.add_row(request_id, prompt_len, **kwargs)
        sinks = self.adapter.begin_prompt_streaming((int(request_id),))
        assert sinks is not None and len(sinks) == 1
        for sink in sinks:
            sink.finished = True
        self.adapter.observe_prefill_result(
            int(request_id),
            row.prompt_ids,
            SimpleNamespace(token_id=1),
        )
        self.target.position = int(prompt_len)
        row.first_token_emitted = True


def test_moe_prefill_missing_sink_is_a_control_error_for_every_prompt_length() -> None:
    """A lost sink is an ownership bug at any length, not a length gate."""

    for prompt_len in (37, 94, 95, 96, 1025):
        adapter = object.__new__(Qwen35GGUFMoEMTP2Adapter)
        adapter._intents = {7: 2}
        adapter._disabled_requests = set()
        adapter._states = {}
        adapter._prompt_sinks = {}
        target = SimpleNamespace()
        row = SimpleNamespace(lease=SimpleNamespace(session=target))
        adapter.owner = SimpleNamespace(_row=lambda request_id: row)

        with pytest.raises(RuntimeError, match="lost its target-hidden sink"):
            adapter.observe_prefill_result(
                7, tuple(range(int(prompt_len))), SimpleNamespace(token_id=1)
            )

        assert adapter._disabled_requests == set()


def test_moe_hidden_slab_is_sized_from_prompt_rows_and_grows(monkeypatch) -> None:
    """One BF16 plus one FP32 row per prompt token, grown on demand."""

    harness = _PrefillHarness(monkeypatch, hidden_size=4)
    hidden = 4
    row_bf16 = hidden * DType.BF16.itemsize
    row_f32 = hidden * DType.FP32.itemsize

    harness.prefill(7, 37)
    key = id(harness.target)
    first = harness.adapter._target_hidden_slabs[key]
    assert harness.adapter._hidden_slab_rows(first, hidden) == 37
    assert harness.hidden_allocations == [37]
    assert harness.pool.charges == [
        ("target_hidden_slab", 37 * row_bf16 + 37 * row_f32 + row_bf16),
        ("draft_kv", 55 * 2 * 2 * 128 * DType.FP32.itemsize),
    ]

    # A shorter prompt reuses the slab without charging or allocating again.
    harness.prefill(8, 12)
    assert harness.adapter._target_hidden_slabs[key] is first
    assert harness.hidden_allocations == [37]
    assert len(harness.pool.charges) == 2

    # 96 and 1025 both outgrow it; each growth frees the slab it replaces once
    # and releases exactly that slab's charge.
    harness.prefill(9, 96)
    second = harness.adapter._target_hidden_slabs[key]
    assert harness.adapter._hidden_slab_rows(second, hidden) == 96
    assert harness.hidden_allocations == [37, 96]
    assert harness.freed_of(*first[1:]) == [first[3].ptr, first[2].ptr, first[1].ptr]
    assert harness.pool.released == [
        ("target_hidden_slab", 37 * row_bf16 + 37 * row_f32 + row_bf16),
        ("draft_kv", 55 * 2 * 2 * 128 * DType.FP32.itemsize),
    ]
    roles = harness.adapter._slab_charge_tokens()
    assert (key, "target_hidden_slab") in roles
    assert len(harness.adapter._resident_charge_tokens[key]) == 2  # hidden + draft

    harness.prefill(10, 1025)
    third = harness.adapter._target_hidden_slabs[key]
    assert harness.adapter._hidden_slab_rows(third, hidden) == 1025
    assert harness.hidden_allocations == [37, 96, 1025]
    assert harness.freed_of(*second[1:]) == [
        second[3].ptr,
        second[2].ptr,
        second[1].ptr,
    ]
    assert len(harness.adapter._resident_charge_tokens[key]) == 2


@pytest.mark.parametrize("fail_index", (0, 1, 2))
def test_moe_hidden_slab_growth_failure_keeps_the_live_slab_and_charge(
    monkeypatch,
    fail_index: int,
) -> None:
    """A failed growth at any allocation leaves no leak and no lost charge."""

    harness = _PrefillHarness(monkeypatch, hidden_size=4)
    harness.prefill(7, 37)
    key = id(harness.target)
    installed = harness.adapter._target_hidden_slabs[key]
    roles = harness.adapter._slab_charge_tokens()
    hidden_token = roles[(key, "target_hidden_slab")]
    charges_before = list(harness.pool.charges)
    released_before = list(harness.pool.released)

    harness.fail_malloc_at = len(harness.mallocs) + fail_index
    allocations_before = len(harness.allocations)
    freed_before = len(harness.freed)
    with pytest.raises(MemoryError, match="synthetic allocation failure"):
        harness.prefill(8, 96)

    assert harness.adapter._target_hidden_slabs[key] is installed
    # Every buffer the failed attempt did allocate is freed exactly once, in
    # reverse order, so the released charge cannot outlive its memory.
    attempt = harness.allocations[allocations_before:]
    assert len(attempt) == fail_index
    assert harness.freed[freed_before:] == [
        ptr for ptr, _nbytes in reversed(attempt)
    ]
    assert harness.freed_of(*installed[1:]) == []
    # The replacement's charge is released again; the live slab stays charged.
    assert harness.pool.released[len(released_before) :] == [
        (
            "target_hidden_slab",
            96 * 4 * DType.BF16.itemsize
            + 96 * 4 * DType.FP32.itemsize
            + 4 * DType.BF16.itemsize,
        )
    ]
    assert harness.adapter._slab_charge_tokens()[(key, "target_hidden_slab")] is (
        hidden_token
    )
    assert len(harness.pool.charges) == len(charges_before) + 1
    assert harness.adapter._prompt_sinks == {}


@pytest.mark.parametrize("fail_index", (0, 1))
def test_moe_draft_kv_growth_failure_keeps_the_live_slab_and_charge(
    monkeypatch,
    fail_index: int,
) -> None:
    """The key/value pair is rolled back before its charge is released."""

    harness = _PrefillHarness(monkeypatch, hidden_size=4)
    harness.prefill(7, 37, max_tokens=16)
    harness.adapter.release_request(7)
    key = id(harness.target)
    installed = harness.adapter._draft_kv_slabs[key]
    roles = harness.adapter._slab_charge_tokens()
    kv_token = roles[(key, "draft_kv")]
    charges_before = list(harness.pool.charges)
    released_before = list(harness.pool.released)

    harness.fail_malloc_at = len(harness.mallocs) + fail_index
    allocations_before = len(harness.allocations)
    freed_before = len(harness.freed)
    with pytest.raises(MemoryError, match="synthetic allocation failure"):
        harness.adapter._ensure_draft_kv_slab(
            harness.target, key, draft=harness.draft, rows=1043
        )

    assert harness.adapter._draft_kv_slabs[key] is installed
    attempt = harness.allocations[allocations_before:]
    assert len(attempt) == fail_index
    assert harness.freed[freed_before:] == [
        ptr for ptr, _nbytes in reversed(attempt)
    ]
    assert harness.freed_of(*installed[1:]) == []
    assert harness.pool.released[len(released_before) :] == [
        ("draft_kv", 1043 * 2 * 2 * 128 * DType.FP32.itemsize)
    ]
    assert harness.adapter._slab_charge_tokens()[(key, "draft_kv")] is kv_token
    assert len(harness.pool.charges) == len(charges_before) + 1


def test_moe_draft_kv_growth_refuses_a_live_owner(monkeypatch) -> None:
    """A slab a live request state still points at is never freed."""

    harness = _PrefillHarness(monkeypatch, session_rows=2048, hidden_size=4)
    harness.prefill(7, 37, max_tokens=16)
    key = id(harness.target)
    installed = harness.adapter._draft_kv_slabs[key]
    kv_token = harness.adapter._slab_charge_tokens()[(key, "draft_kv")]

    # A second request needs a bigger slab while the first state still owns the
    # installed one: growing here would free rows that state still writes.
    with pytest.raises(RuntimeError, match="live request state"):
        harness.prefill(8, 1025, max_tokens=16, finish_previous=False)

    assert harness.adapter._draft_kv_slabs[key] is installed
    assert harness.freed_of(*installed[1:]) == []
    assert harness.adapter._slab_charge_tokens()[(key, "draft_kv")] is kv_token
    assert harness.adapter._states[7].key_cache is installed[1]


def test_moe_hidden_slab_growth_refuses_a_live_sink(monkeypatch) -> None:
    """A slab a live prompt sink still writes is never freed."""

    harness = _PrefillHarness(monkeypatch, hidden_size=4)
    harness.prefill(7, 37)
    key = id(harness.target)
    installed = harness.adapter._target_hidden_slabs[key]
    harness.add_row(8, 20)
    sinks = harness.adapter.begin_prompt_streaming((8,))
    assert sinks is not None and len(sinks) == 1

    with pytest.raises(RuntimeError, match="live prompt sink"):
        harness.adapter._ensure_target_hidden_slab(harness.target, key, 96)

    assert harness.adapter._target_hidden_slabs[key] is installed
    assert harness.freed_of(*installed[1:]) == []


def test_moe_draft_kv_reuse_does_not_need_the_owner_guard(monkeypatch) -> None:
    """A slab that already fits is shared, so a live owner never blocks it."""

    harness = _PrefillHarness(monkeypatch, session_rows=2048, hidden_size=4)
    harness.prefill(7, 37, max_tokens=16)
    key = id(harness.target)
    installed = harness.adapter._draft_kv_slabs[key]

    harness.prefill(8, 12, max_tokens=16, finish_previous=False)

    assert harness.adapter._draft_kv_slabs[key] is installed
    assert harness.kv_allocations == [55]


def test_moe_hidden_slab_charge_refusal_happens_before_allocating(monkeypatch) -> None:
    """A pool refusal names the slab and never reaches the malloc."""

    from hipengine.runtime.memory_admission import MemoryAdmissionRefused

    harness = _PrefillHarness(monkeypatch, hidden_size=4)
    harness.pool.refuse = "target_hidden_slab"

    with pytest.raises(MemoryAdmissionRefused) as caught:
        harness.prefill(7, 37)

    assert caught.value.refused_consumer == "target_hidden_slab"
    assert harness.mallocs == []
    assert harness.adapter._target_hidden_slabs == {}
    assert harness.adapter._resident_charge_tokens == {}


def test_moe_draft_kv_covers_the_request_and_grows_with_it(monkeypatch) -> None:
    """Draft-KV rows follow the request's own extent, clamped to the session."""

    harness = _PrefillHarness(monkeypatch, session_rows=2048, hidden_size=4)
    row_bytes = 2 * 128 * DType.FP32.itemsize

    harness.prefill(7, 37, max_tokens=16)
    # prompt 37 + up to 16 generated + one candidate depth of 2.
    assert harness.kv_allocations == [55]
    assert harness.pool.charges[-1] == ("draft_kv", 55 * 2 * row_bytes)
    assert harness.adapter._states[7].draft_kv_rows == 55

    # A longer request grows the same target's slab and frees the old one.
    first = harness.adapter._draft_kv_slabs[id(harness.target)]
    harness.prefill(8, 1025, max_tokens=16)
    assert harness.kv_allocations == [55, 1043]
    assert harness.freed_of(*first[1:]) == [first[2].ptr, first[1].ptr]
    assert harness.adapter._states[8].draft_kv_rows == 1043

    # The session's own positions clamp the slab, and a short prompt reuses it.
    harness.prefill(9, 2000, max_tokens=4096)
    assert harness.kv_allocations[-1] == 2048
    harness.prefill(10, 37, max_tokens=16)
    assert harness.kv_allocations[-1] == 2048
    assert harness.adapter._states[10].draft_kv_rows == 2048


def test_moe_draft_kv_is_clamped_to_the_target_verify_ceiling(monkeypatch) -> None:
    """No cycle can reach past the ceiling, so the slab is not sized past it."""

    harness = _PrefillHarness(
        monkeypatch, backend="hip_gfx1100", session_rows=2048, hidden_size=4
    )

    # 37 prompt + 100 requested + 2 candidate depth is 139 rows, but gfx1100
    # verifies at most 95 positions, so 95 is what the request can reach.
    harness.prefill(7, 37, max_tokens=100)

    assert harness.kv_allocations == [95]
    assert harness.adapter._states[7].draft_kv_rows == 95
    # The session, not the declared ceiling, binds on a wide backend.
    wide = _PrefillHarness(
        monkeypatch, backend="hip_gfx1151", session_rows=512, hidden_size=4
    )
    wide.prefill(7, 37, max_tokens=600)
    assert wide.kv_allocations == [512]


def test_moe_short_draft_kv_refuses_before_any_device_write(monkeypatch) -> None:
    """No row write may land outside the allocation that was primed."""

    from hipengine.generation.qwen35_gguf_moe_mtp2 import MoeDraftKVCapacityError

    harness = _PrefillHarness(monkeypatch, hidden_size=4, short_kv_rows=4)

    with pytest.raises(MoeDraftKVCapacityError, match="needs one row per token"):
        harness.prefill(7, 37)

    assert harness.draft.write_calls == []
    assert harness.adapter._states == {}
    assert harness.released_runners == ["pool-key"]


def test_moe_draft_kv_room_guard_bounds_every_write_site() -> None:
    """The cycle and proposal guards read the primed allocation, not an estimate."""

    from hipengine.generation.qwen35_gguf_moe_mtp2 import MoeDraftKVCapacityError

    state = SimpleNamespace(draft_kv_rows=8, cache_len=6)

    # Fits exactly: rows 6..7 of an 8-row allocation.
    Qwen35GGUFMoEMTP2Adapter._check_draft_kv_room(
        state, cursor=6, rows=2, context="cycle"
    )
    # A zero-row write is legal; a negative cursor or count is not.
    Qwen35GGUFMoEMTP2Adapter._check_draft_kv_room(
        state, cursor=8, rows=0, context="cycle"
    )
    with pytest.raises(MoeDraftKVCapacityError, match="invalid draft-KV write"):
        Qwen35GGUFMoEMTP2Adapter._check_draft_kv_room(
            state, cursor=-1, rows=2, context="cycle"
        )
    with pytest.raises(MoeDraftKVCapacityError, match="invalid draft-KV write"):
        Qwen35GGUFMoEMTP2Adapter._check_draft_kv_room(
            state, cursor=4, rows=-2, context="cycle"
        )
    with pytest.raises(MoeDraftKVCapacityError, match=r"rows 7\.\.8 outside the 8-row"):
        Qwen35GGUFMoEMTP2Adapter._check_draft_kv_room(
            state, cursor=7, rows=2, context="cycle"
        )
    # A state that never published an allocation cannot prove its write is
    # inside one, so the guard refuses rather than assuming it is safe.
    with pytest.raises(MoeDraftKVCapacityError, match="no published draft-KV"):
        Qwen35GGUFMoEMTP2Adapter._check_draft_kv_room(
            SimpleNamespace(cache_len=99), cursor=0, rows=1, context="cycle"
        )


def test_moe_cycle_refuses_an_out_of_capacity_write_before_device_work(
    monkeypatch,
) -> None:
    """``execute_cycle`` checks the allocation before it enqueues anything."""

    from hipengine.generation.qwen35_gguf_moe_mtp2 import MoeDraftKVCapacityError

    harness = _PrefillHarness(monkeypatch, hidden_size=4)
    harness.prefill(7, 37)
    state = harness.adapter._states[7]
    # Pretend the primed allocation is one row short of what this cycle writes.
    state.draft_kv_rows = int(state.cache_len) + 1
    target = harness.target
    target.position = int(state.cache_len) - 1
    row = harness.rows[7]
    row.slot = SimpleNamespace(
        seq_position=int(state.cache_len) - 1, generated_ids=[1]
    )
    device_work: list[object] = []
    target.run_native_spec_mtp_cycle = lambda *a, **k: device_work.append(a)

    plan = SimpleNamespace(
        operation_id="op",
        request_ids=(7,),
        speculative_request_ids=(7,),
        candidate_counts=(2,),
        logical_frontier_rows=3,
        resident_slots=(0,),
    )

    with pytest.raises(MoeDraftKVCapacityError, match="MoE MTP cycle for request 7"):
        harness.adapter.execute_cycle(plan, commit=True)

    # Nothing was enqueued: the guard runs before the target cycle opens.
    assert device_work == []
    assert state.cache_len == 37


def test_moe_explicit_prompt_beyond_the_ceiling_raises_by_name(monkeypatch) -> None:
    """An explicit MTP request over the ceiling fails loudly, never silently."""

    from hipengine.generation.qwen35_gguf_moe_mtp2 import (
        MoePromptContextUnsupportedError,
    )

    harness = _PrefillHarness(monkeypatch, backend="hip_gfx1100", session_rows=2048)

    # 92 is the last prompt length whose first plan still has candidate room.
    harness.prefill(7, 92)
    assert harness.adapter._states[7] is not None
    writes_before = len(harness.draft.write_calls)

    for prompt_len in (93, 94, 95, 96, 1025, 1500):
        request_id = 100 + int(prompt_len)
        row = harness.add_row(request_id, prompt_len)
        sinks = harness.adapter.begin_prompt_streaming((request_id,))
        for sink in sinks:
            sink.finished = True
        with pytest.raises(
            MoePromptContextUnsupportedError, match="no candidate room"
        ):
            harness.adapter.observe_prefill_result(
                request_id,
                row.prompt_ids,
                SimpleNamespace(token_id=1),
            )
        assert request_id not in harness.adapter._states
        assert request_id not in harness.adapter._disabled_requests
        assert row.mtp2_prompt_fallback_reason is None
        # Refused before any draft work, so nothing was primed.
        assert len(harness.draft.write_calls) == writes_before


def test_moe_automatic_prompt_beyond_the_ceiling_records_its_reason(
    monkeypatch,
) -> None:
    """An automatic selection may fall back, but only with a stated reason."""

    harness = _PrefillHarness(monkeypatch, backend="hip_gfx1100", session_rows=2048)
    row = harness.add_row(7, 96, automatic=True)
    sinks = harness.adapter.begin_prompt_streaming((7,))
    for sink in sinks:
        sink.finished = True

    harness.adapter.observe_prefill_result(
        7, row.prompt_ids, SimpleNamespace(token_id=1)
    )

    assert harness.adapter._disabled_requests == {7}
    assert row.mtp2_prompt_fallback_reason == "target_context_k0"
    assert 7 not in harness.adapter._states
    assert harness.draft.write_calls == []


def test_moe_prefill_is_unchanged_at_the_original_working_length(monkeypatch) -> None:
    """37 tokens still primes five draft-KV rows worth of cycles as before."""

    harness = _PrefillHarness(monkeypatch, backend="hip_gfx1100", session_rows=2048)

    harness.prefill(7, 37)

    state = harness.adapter._states[7]
    assert state.cache_len == 37
    assert harness.draft.write_calls and len(harness.draft.write_calls) == 1
    assert harness.rows[7].mtp2_prompt_prime_rows == 37
    assert harness.rows[7].mtp2_prompt_fallback_reason is None
    assert harness.runtime.syncs == [0]
    # The priming scratch is allocated and freed exactly once, so nothing leaks.
    assert len(harness.freed) == 1


def test_moe_adapter_factory_keeps_c1_and_selects_batched_c2() -> None:
    owner_c1 = SimpleNamespace(capacity=1, generator=SimpleNamespace())
    owner_c2 = SimpleNamespace(
        capacity=2,
        generator=SimpleNamespace(
            execution_profile=SimpleNamespace(value="production"),
            backend="hip_gfx1100",
        ),
    )
    kwargs = {
        "enabled": True,
        "target_verify_mode": "native",
        "candidate_budget": 2,
        "quant": "gguf_q4_k_m",
    }

    c1 = create_qwen35_gguf_moe_mtp2_adapter(owner_c1, **kwargs)
    c2 = create_qwen35_gguf_moe_mtp2_adapter(owner_c2, **kwargs)

    assert isinstance(c1, Qwen35GGUFMoEMTP2Adapter)
    assert isinstance(c2, Qwen35GGUFMTP2Adapter)
    assert c2.production_physical_extra_rowtiles is False
    assert c2.production_physical_q5_rowtile is False
    assert c2.production_physical_q6_rowtile is False
    assert c2.moe_physical_c2_numerics is True
    assert c2.moe_physical_c2_pairreuse is True
    assert c2.moe_physical_c2_exact_linear is True
    assert c2.target_key == "qwen_moe_gguf"
    assert c2.provider_key == "qwen_nextn_moe"
    assert c2.policy_prefix == "moe-nextn"


def test_builtin_registry_exposes_distinct_dense_and_moe_factories() -> None:
    register_builtin_gguf_mtp2_adapters()

    assert resolve_gguf_mtp2_adapter("dense_nextn").__name__ == "Qwen35GGUFMTP2Adapter"
    assert (
        resolve_gguf_mtp2_adapter("moe_nextn")
        is create_qwen35_gguf_moe_mtp2_adapter
    )


class _RecordingPool:
    """Minimal stand-in for the KV pool's named resident-consumer ledger."""

    def __init__(self, *, refuse: str | None = None) -> None:
        self.charges: list[tuple[str, int]] = []
        self.released: list[tuple[str, int]] = []
        self.refuse = refuse
        self._tokens: dict[object, tuple[str, int]] = {}

    def reserve_resident_bytes(self, name, nbytes):
        if self.refuse == str(name):
            from hipengine.runtime.memory_admission import (
                MemoryAdmissionRefused,
                MemoryConsumer,
                price_memory_admission,
            )

            raise MemoryAdmissionRefused(
                price_memory_admission(
                    free_bytes=1,
                    consumers=(MemoryConsumer(str(name), int(nbytes)),),
                    reserve_bytes=0,
                )
            )
        token = object()
        self.charges.append((str(name), int(nbytes)))
        self._tokens[token] = (str(name), int(nbytes))
        return token

    def release_private_workspace(self, token):
        self.released.append(self._tokens.pop(token))


def _charge_test_adapter(pool) -> Qwen35GGUFMoEMTP2Adapter:
    adapter = object.__new__(Qwen35GGUFMoEMTP2Adapter)
    adapter.owner = SimpleNamespace(_kv_pool=pool)
    adapter._resident_charge_tokens = {}
    return adapter


def test_moe_adapter_prices_its_slabs_through_the_pool_budget() -> None:
    """The draft-KV and target-hidden slabs are charged, per target, by name."""

    pool = _RecordingPool()
    adapter = _charge_test_adapter(pool)

    adapter._charge_resident_slab(7, "draft_kv", 2048)
    adapter._charge_resident_slab(7, "target_hidden_slab", 1024)
    adapter._charge_resident_slab(9, "draft_kv", 512)
    assert pool.charges == [
        ("draft_kv", 2048),
        ("target_hidden_slab", 1024),
        ("draft_kv", 512),
    ]

    # One target's slabs can be freed on their own; the rest stay charged.
    adapter._release_resident_slab_charges(7)
    assert pool.released == [("draft_kv", 2048), ("target_hidden_slab", 1024)]
    assert set(adapter._resident_charge_tokens) == {9}

    adapter._release_resident_slab_charges()
    assert pool.released[-1] == ("draft_kv", 512)
    assert adapter._resident_charge_tokens == {}
    assert pool._tokens == {}


def test_moe_adapter_slab_charge_is_refused_before_allocating() -> None:
    """A refusal propagates out of the charge, so no malloc is reached."""

    from hipengine.runtime.memory_admission import MemoryAdmissionRefused

    pool = _RecordingPool(refuse="draft_kv")
    adapter = _charge_test_adapter(pool)

    with pytest.raises(MemoryAdmissionRefused) as caught:
        adapter._charge_resident_slab(7, "draft_kv", 4096)

    assert caught.value.refused_consumer == "draft_kv"
    assert "draft_kv" in str(caught.value)
    assert pool.charges == []
    assert adapter._resident_charge_tokens == {}


def test_moe_adapter_skips_the_charge_when_the_pool_has_no_ledger() -> None:
    adapter = _charge_test_adapter(SimpleNamespace())
    adapter._charge_resident_slab(7, "draft_kv", 4096)
    adapter._charge_resident_slab(7, "target_hidden_slab", 0)
    assert adapter._resident_charge_tokens == {}
    # Releasing with nothing charged is a no-op rather than an error.
    adapter._release_resident_slab_charges()
    adapter._release_resident_slab_charges(7)


def test_moe_dense_kv_size_helper_matches_what_the_allocator_allocates(monkeypatch) -> None:
    """The priced bytes are the allocated bytes, not a second estimate."""

    from hipengine.generation import qwen35_gguf as gguf

    allocated: list[int] = []

    def fake_malloc(nbytes, *, runtime):
        allocated.append(int(nbytes))
        return DeviceBuffer(ptr=0x1000, nbytes=int(nbytes))

    monkeypatch.setattr("hipengine.core.memory.malloc", fake_malloc)
    key_cache, value_cache, buffers = gguf._allocate_mtp_dense_kv(
        runtime=SimpleNamespace(), capacity=64, qk_head_dim=128, kv_heads=2
    )

    priced = gguf._mtp_dense_kv_nbytes(capacity=64, qk_head_dim=128, kv_heads=2)
    assert priced == sum(allocated) == int(key_cache.nbytes) + int(value_cache.nbytes)
    assert len(buffers) == 2


def test_moe_cycle_forwards_the_attention_schedule_its_bucket_declares(
    monkeypatch,
) -> None:
    """The cycle asks the target for the schedule the bucket requires.

    A short-context cycle keeps the bulk schedule the measured short route runs.
    Once the cycle's rows land in a long-context bucket the graph verifies with
    split-K native attention and refuses bulk, so the same row asks for
    ``native``; the bucket selector is the only place that threshold lives.
    """

    harness = _PrefillHarness(monkeypatch, session_rows=2048, hidden_size=4)
    harness.add_row(7, 37)
    harness.prefill(7, 37)
    assert harness.cycle()["target_bulk_attention_mode"] == "bulk"

    # Same row, long-context bucket. The state is arranged directly because the
    # subject is the schedule the cycle requests, not how it reached 1100.
    harness.target.scratch = SimpleNamespace(max_positions=2048, block_size=256)
    harness.target.position = 1100
    harness.adapter._states[7].cache_len = 1100
    harness.adapter._states[7].draft_kv_rows = 1200
    kwargs = harness.cycle()
    assert kwargs["target_bulk_attention_mode"] == "native"
    assert kwargs["draft_cache_len"] == 1100
    assert kwargs["candidate_budget"] == 2
    # The cycle is still a complete native cycle: nothing here downgrades it to
    # an eager or autoregressive route at long context.
    assert kwargs["sampled_accept"] is False
    assert kwargs["k1_disable_f32_verifier"] is True
