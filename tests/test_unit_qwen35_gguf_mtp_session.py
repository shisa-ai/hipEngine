from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.hip import HipMemcpyKind
from hipengine.core.memory import DeviceBuffer
from hipengine.core.tensor import Tensor
from hipengine.generation.deadline import GenerationCancelled
from hipengine.kvcache import KVScaleMetadata
from hipengine.runtime import qwen35_gguf_mtp as mtp_module
from hipengine.runtime.qwen35_gguf_mtp import Qwen35GGUFMTPDecodeSession


def test_mtp_int8_target_policy_registration_keeps_scale_metadata() -> None:
    device = Device("hip", 0)
    block_table = Tensor.from_handle(0x1000, (4,), DType.INT32, device)
    live_counts = Tensor.from_handle(0x2000, (1,), DType.INT64, device)
    metadata = KVScaleMetadata(
        k_scale=Tensor.from_handle(0x3000, (4, 256, 2), DType.FP32, device),
        v_scale=Tensor.from_handle(0x4000, (4, 256, 2), DType.FP32, device),
        scale_dtype=DType.FP32,
    )
    owner = SimpleNamespace(
        block_size=256,
        kv_storage_dtype=DType.INT8_PER_TOKEN_HEAD,
        block_table_tensor=block_table,
        context_tensor=live_counts,
        max_positions=1024,
        full_kv_scale_metadata=(None, metadata, None),
    )
    decoder = Qwen35GGUFMTPDecodeSession.__new__(Qwen35GGUFMTPDecodeSession)
    decoder.target = SimpleNamespace(_target_scratch_owner=owner, position=9)

    policy = decoder._register_kv_policy(7)

    reservation = policy.reservations[7]
    assert reservation.storage_dtype is DType.INT8_PER_TOKEN_HEAD
    assert reservation.scale_metadata is metadata


@pytest.mark.parametrize(
    ("budget", "position", "remaining_decode", "expected", "expected_reason"),
    [
        (1, 1020, 2, True, None),
        (1, 1021, 2, False, "target_graph_proposal_handoff_boundary_miss"),
        (1, 1022, 2, False, "target_graph_context_bucket_miss"),
        (2, 1019, 3, True, None),
        (2, 1020, 3, False, "target_graph_proposal_handoff_boundary_miss"),
        (2, 1021, 3, False, "target_graph_context_bucket_miss"),
        (3, 1018, 4, True, None),
        (3, 1019, 4, False, "target_graph_proposal_handoff_boundary_miss"),
        (3, 1020, 4, False, "target_graph_context_bucket_miss"),
        (3, 1019, 3, False, "target_graph_output_room_miss"),
    ],
)
def test_device_proposal_ready_checks_live_cycle_end_and_output_room(
    budget: int,
    position: int,
    remaining_decode: int,
    expected: bool,
    expected_reason: str | None,
) -> None:
    class Graph:
        closed = False
        context_limit = 1023

        def compatible_with(self, _target, **_kwargs) -> bool:
            return True

        def launch_ineligibility_reason(self, _target, **kwargs) -> str | None:
            rows = int(kwargs["rows"])
            if int(kwargs["position"]) + rows > 1023:
                return "target_graph_context_bucket_miss"
            if int(kwargs["remaining_decode"]) < rows:
                return "target_graph_output_room_miss"
            return None

    target = SimpleNamespace(position=position)
    setattr(target, f"_native_spec_b{budget}_target_graph_n2", Graph())
    verifier = mtp_module.Qwen35GGUFTransactionalVerifier.__new__(
        mtp_module.Qwen35GGUFTransactionalVerifier
    )
    verifier.closed = False
    verifier.backend = "hip_gfx1151"
    verifier.target_verify_mode = "native"
    verifier.target = target
    verifier.last_device_proposal_fallback_reason = "stale"

    assert verifier.device_proposal_ready(
        budget,
        remaining_decode=remaining_decode,
    ) is expected
    assert verifier.last_device_proposal_fallback_reason == expected_reason


@pytest.mark.parametrize(
    "budget,position,expected",
    [(1, 93, True), (1, 94, False), (2, 92, True), (2, 93, False),
     (3, 91, True), (3, 92, False)],
)
def test_device_proposal_admission_respects_native_policy_before_launch(
    monkeypatch, budget, position, expected,
) -> None:
    monkeypatch.setattr(mtp_module, "backend_package_capability", lambda *_a: 95)
    calls = []

    class Graph:
        context_limit = 1023

        def launch_ineligibility_reason(self, _target, **kwargs):
            calls.append(kwargs)
            return None

    verifier = mtp_module.Qwen35GGUFTransactionalVerifier.__new__(
        mtp_module.Qwen35GGUFTransactionalVerifier
    )
    verifier.closed = False
    verifier.backend = "hip_gfx1100"
    verifier.target_verify_mode = "native"
    verifier.target = SimpleNamespace(position=position)
    setattr(verifier.target, f"_native_spec_b{budget}_target_graph_n2", Graph())
    assert verifier.device_proposal_ready(budget, remaining_decode=4) is expected
    assert bool(calls) is expected
    if not expected:
        assert verifier.last_device_proposal_fallback_reason == "target_graph_native_policy_miss"


def test_ineligible_cached_target_graph_never_launches_device_proposal() -> None:
    calls: list[tuple[object, ...]] = []

    class Verifier:
        last_device_proposal_fallback_reason = None

        def device_proposal_ready(self, budget, *, remaining_decode):
            calls.append(("ready", int(budget), int(remaining_decode)))
            self.last_device_proposal_fallback_reason = (
                "target_graph_context_bucket_miss"
            )
            return False

    class Provider:
        def launch_device_proposal(self, *_args, **_kwargs):
            calls.append(("launch",))
            raise AssertionError("ineligible target graph must prevent proposal launch")

    proposal = mtp_module._maybe_launch_device_proposal(
        Verifier(),
        Provider(),
        SimpleNamespace(root_positions=(1020,)),
        candidate_budget=3,
        remaining_decode=4,
        return_cycle_logits=False,
    )

    assert proposal is None
    assert calls == [("ready", 3, 4)]


@pytest.mark.parametrize("budget", range(1, 8))
@pytest.mark.parametrize("context_limit", [95, 1023, 1280, 1793])
@pytest.mark.parametrize("end_delta", [-1, 0, 1])
def test_device_handoff_boundary_prevents_proposal_submission(
    budget: int, context_limit: int, end_delta: int,
) -> None:
    """The real readiness check runs before the provider can mutate its state."""

    class Graph:
        def launch_ineligibility_reason(self, _target, **kwargs):
            if int(kwargs["position"]) + int(kwargs["rows"]) > self.context_limit:
                return "target_graph_context_bucket_miss"
            return None

    graph = Graph()
    graph.context_limit = context_limit
    position = context_limit - budget - 1 + end_delta
    target = SimpleNamespace(position=position)
    setattr(target, f"_native_spec_b{budget}_target_graph_n2", graph)
    verifier = mtp_module.Qwen35GGUFTransactionalVerifier.__new__(
        mtp_module.Qwen35GGUFTransactionalVerifier
    )
    verifier.closed = False
    verifier.backend = "hip_gfx1151"
    verifier.target_verify_mode = "native"
    verifier.target = target
    verifier.last_device_proposal_fallback_reason = "stale"
    context = SimpleNamespace(root_positions=(position,))
    launched = []
    result = object()

    class Provider:
        def launch_device_proposal(self, proposal_context, *, candidate_budget):
            launched.append((proposal_context, candidate_budget))
            return result

    proposal = mtp_module._maybe_launch_device_proposal(
        verifier, Provider(), context,
        candidate_budget=budget, remaining_decode=budget + 1,
        return_cycle_logits=False,
    )

    if end_delta < 0:
        assert proposal is result
        assert launched == [(context, budget)]
        assert verifier.last_device_proposal_fallback_reason is None
    else:
        assert proposal is None
        assert launched == []
        assert verifier.last_device_proposal_fallback_reason == (
            "target_graph_proposal_handoff_boundary_miss"
            if end_delta == 0 else "target_graph_context_bucket_miss"
        )
    assert target.position == position


def test_native_target_rows_follow_implementation_ladder_and_context_limit(monkeypatch) -> None:
    monkeypatch.setattr(mtp_module, "backend_package_capability", lambda *_a: 95)
    assert mtp_module._effective_target_verify_mode("native", rows=4) == "native"
    assert (
        mtp_module._effective_target_verify_mode(
            "native", rows=3, backend="hip_gfx1100", end_position=95
        )
        == "native"
    )
    assert (
        mtp_module._effective_target_verify_mode(
            "native", rows=4, backend="hip_gfx1100", end_position=96
        )
        == "serial_exact"
    )
    for rows in range(2, 9):
        assert mtp_module._effective_target_verify_mode("native", rows=rows) == "native"
        assert mtp_module._effective_target_verify_mode("serial_exact", rows=rows) == "serial_exact"
    assert mtp_module._effective_target_verify_mode("native", rows=9) == "serial_exact"
    assert mtp_module._initial_state_only_journal_applies(
        "native",
        max_candidate_budget=3,
        backend="hip_gfx1151",
        max_end_position=95,
    )
    for budget in range(4, 8):
        assert mtp_module._initial_state_only_journal_applies(
            "native", max_candidate_budget=budget,
            backend="hip_gfx1151", max_end_position=95,
        )
    assert not mtp_module._initial_state_only_journal_applies(
        "native", max_candidate_budget=8,
        backend="hip_gfx1151",
        max_end_position=4096,
    )
    assert not mtp_module._initial_state_only_journal_applies(
        "serial_exact",
        max_candidate_budget=3,
        backend="hip_gfx1151",
        max_end_position=4096,
    )


def test_initial_state_only_journal_bounds_the_whole_reachable_position_range() -> None:
    """A serial row route can appear at any decode position, not only by row count.

    gfx1100's native target context limit is 95, so a session that can reach
    position 96 uses ``serial_exact`` rows and needs row snapshots. The
    natural25 suite reproduces it: the tenth prompt (71 prompt tokens plus 25
    generated tokens) crosses 95 and previously failed at ``capture_row``.
    """

    assert (
        mtp_module._effective_target_verify_mode(
            "native", rows=4, backend="hip_gfx1100", end_position=96
        )
        == "serial_exact"
    )
    assert not mtp_module._initial_state_only_journal_applies(
        "native",
        max_candidate_budget=3,
        backend="hip_gfx1100",
        max_end_position=96,
    )
    assert mtp_module._initial_state_only_journal_applies(
        "native",
        max_candidate_budget=3,
        backend="hip_gfx1100",
        max_end_position=95,
    )
    # An unbounded position range can always reach the serial fallback.
    assert not mtp_module._initial_state_only_journal_applies(
        "native", max_candidate_budget=3, backend="hip_gfx1100"
    )
    assert not mtp_module._initial_state_only_journal_applies(
        "native", max_candidate_budget=3, max_end_position=4096
    )


def test_serial_row_route_requires_a_serial_capable_journal_before_mutation() -> None:
    """The journal route contract fails closed before any state mutation."""

    mtp_module._require_serial_capable_journal(
        SimpleNamespace(initial_state_only=True), "native"
    )
    with pytest.raises(RuntimeError, match="row-capable journal"):
        mtp_module._require_serial_capable_journal(
            SimpleNamespace(initial_state_only=True), "serial_exact"
        )
    mtp_module._require_serial_capable_journal(
        SimpleNamespace(initial_state_only=False), "serial_exact"
    )


def test_journal_plan_pairs_producer_capture_with_initial_state_only() -> None:
    """Producer capture lends one rollback row, so a row-capable journal owns its own.

    A row-capable journal that kept borrowing the single producer row would write
    past it at ``_capture_state_index(row + 1)``.
    """

    assert mtp_module._verify_journal_plan(
        "native",
        max_candidate_budget=3,
        backend="hip_gfx1151",
        max_end_position=4096,
    ) == (True, True)
    assert mtp_module._verify_journal_plan(
        "native",
        max_candidate_budget=3,
        backend="hip_gfx1100",
        max_end_position=260,
    ) == (False, False)
    assert mtp_module._verify_journal_plan(
        "serial_exact",
        max_candidate_budget=3,
        backend="hip_gfx1151",
        max_end_position=4096,
    ) == (False, False)


def test_state_row_capacity_matches_the_allocated_rollback_rows() -> None:
    journal = mtp_module._StateJournal.__new__(mtp_module._StateJournal)
    journal.max_rows = 4
    journal.initial_state_only = True
    assert journal.state_row_capacity == 1
    journal.initial_state_only = False
    assert journal.state_row_capacity == 5


def test_serial_fallback_forces_consumer_owned_initial_state_snapshot() -> None:
    journal = mtp_module._StateJournal.__new__(mtp_module._StateJournal)
    journal.target = SimpleNamespace(last_target_hidden=SimpleNamespace(ptr=0x2000))
    journal.initial_hidden = DeviceBuffer(0x1000, 8)
    journal.producer_capture_initial_state = True
    journal.initial_state_captured = False
    copies: list[tuple[int, int, int, int]] = []
    state_copies: list[tuple[bool, int]] = []
    journal._copy_d2d = lambda dst, src, nbytes, *, stream: copies.append(
        (int(dst), int(src), int(nbytes), int(stream))
    )
    journal._copy_initial_state = lambda *, restore, stream: (
        state_copies.append((bool(restore), int(stream))) or True
    )
    journal._capture_state_index = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("pair-copy snapshot should own this fixture")
    )

    journal.capture_initial(stream=9, force_consumer_state=True)

    assert copies == [(0x1000, 0x2000, 8, 9)]
    assert state_copies == [(False, 9)]
    assert journal.initial_state_captured


def test_native_verifier_lazily_owns_a_separate_serial_fallback_journal(monkeypatch) -> None:
    primary = SimpleNamespace(initial_state_only=True)
    serial = SimpleNamespace(initial_state_only=False)
    allocations = []
    verifier = mtp_module.Qwen35GGUFTransactionalVerifier.__new__(
        mtp_module.Qwen35GGUFTransactionalVerifier
    )
    verifier.target = object()
    verifier.max_candidate_budget = 3
    verifier.journal = primary
    verifier._primary_journal = primary
    verifier._serial_journal = None

    def allocate(target, **kwargs):
        allocations.append((target, kwargs))
        return serial

    monkeypatch.setattr(mtp_module._StateJournal, "allocate", allocate)
    verifier._select_journal("native")
    assert verifier.journal is primary
    assert not allocations
    verifier._select_journal("serial_exact")
    assert verifier.journal is serial
    assert allocations == [
        (verifier.target, {
            "max_rows": 4,
            "producer_capture_initial_state": False,
            "initial_state_only": False,
        }),
    ]
    verifier._select_journal("native")
    assert verifier.journal is primary
    verifier._select_journal("serial_exact")
    assert verifier.journal is serial
    assert len(allocations) == 1


def test_serial_only_verifier_reuses_its_primary_journal(monkeypatch) -> None:
    primary = SimpleNamespace(initial_state_only=False)
    verifier = mtp_module.Qwen35GGUFTransactionalVerifier.__new__(
        mtp_module.Qwen35GGUFTransactionalVerifier
    )
    verifier.journal = verifier._primary_journal = primary
    verifier._serial_journal = None
    monkeypatch.setattr(
        mtp_module._StateJournal, "allocate",
        lambda *_a, **_kw: pytest.fail("serial primary already has row storage"),
    )
    verifier._select_journal("serial_exact")
    assert verifier.journal is primary


def test_failed_serial_journal_allocation_keeps_native_owner(monkeypatch) -> None:
    primary = SimpleNamespace(initial_state_only=True)
    verifier = mtp_module.Qwen35GGUFTransactionalVerifier.__new__(
        mtp_module.Qwen35GGUFTransactionalVerifier
    )
    verifier.target = object()
    verifier.max_candidate_budget = 3
    verifier.journal = verifier._primary_journal = primary
    verifier._serial_journal = None

    def fail(*_args, **_kwargs):
        raise MemoryError("allocation failed")

    monkeypatch.setattr(mtp_module._StateJournal, "allocate", fail)
    with pytest.raises(MemoryError, match="allocation failed"):
        verifier._select_journal("serial_exact")
    assert verifier.journal is primary
    assert verifier._primary_journal is primary
    assert verifier._serial_journal is None


def test_prepare_selects_serial_journal_before_context_fallback_mutation(monkeypatch) -> None:
    monkeypatch.setattr(mtp_module, "backend_package_capability", lambda *_a: 95)
    captures = []

    def capture_serial(**kwargs):
        captures.append(kwargs)
        raise RuntimeError("stop after serial snapshot")

    primary = SimpleNamespace(initial_state_only=True)
    serial = SimpleNamespace(capture_initial=capture_serial, initial_state_only=False)
    verifier = mtp_module.Qwen35GGUFTransactionalVerifier.__new__(
        mtp_module.Qwen35GGUFTransactionalVerifier
    )
    verifier.target = SimpleNamespace(position=93)
    verifier.backend = "hip_gfx1100"
    verifier.target_verify_mode = "native"
    verifier.max_candidate_budget = 3
    verifier.closed = False
    verifier._prepared = None
    verifier.journal = verifier._primary_journal = primary
    verifier._serial_journal = None
    verifier._validate_chain = lambda _batch: None
    monkeypatch.setattr(mtp_module._StateJournal, "allocate", lambda *_a, **_kw: serial)
    batch = SimpleNamespace(
        mode="verify_chain", rows=3, request_ids=(0,), positions=(93, 94, 95),
        root_rows=(0,),
    )
    bucket = SimpleNamespace(owner=SimpleNamespace(
        spec=SimpleNamespace(mode="verify_chain", max_rows=3),
    ))
    with pytest.raises(RuntimeError, match="stop after serial snapshot"):
        verifier.prepare(batch, transaction_id=1, graph_bucket=bucket, remaining_decode=(2,))
    assert captures == [{"stream": 0, "force_consumer_state": True}]
    assert verifier.journal is serial
    assert verifier.target.position == 93
    assert verifier._prepared is None


@pytest.mark.parametrize("budget,remaining", [(1, 1), (2, 1), (2, 2), (3, 1), (3, 2), (3, 3)])
def test_device_proposal_tail_stays_on_device_accept_commit(budget, remaining) -> None:
    calls = []

    def device_target(proposal, **kwargs):
        calls.append((proposal, kwargs["remaining_decode"]))
        raise RuntimeError("device handoff reached")

    primary = SimpleNamespace(
        initial_state_only=True, capture_initial=lambda **_kw: None,
        restore_initial=lambda **_kw: None,
    )
    verifier = mtp_module.Qwen35GGUFTransactionalVerifier.__new__(
        mtp_module.Qwen35GGUFTransactionalVerifier
    )
    verifier.target = SimpleNamespace(
        position=25,
        verify_target_from_device_proposal=device_target,
        verify_target_block_native_cycle=lambda *_a, **_kw: pytest.fail("host fallback consumed placeholders"),
    )
    verifier.backend = "hip_gfx1100"
    verifier.target_verify_mode = "native"
    verifier.max_candidate_budget = 3
    verifier.closed = False
    verifier._prepared = None
    verifier.journal = verifier._primary_journal = primary
    verifier._serial_journal = None
    verifier._validate_chain = lambda _batch: None
    verifier._publish_position = lambda *_a, **_kw: None
    verifier._synchronize = lambda _stream: None
    proposal = SimpleNamespace(budget=budget, request_id=0)
    batch = SimpleNamespace(
        mode="verify_chain", rows=budget + 1, request_ids=(0,),
        positions=tuple(range(25, 26 + budget)), root_rows=(0,),
        tokens=(7, *(2147483647 for _ in range(budget))),
    )
    bucket = SimpleNamespace(
        replay_count=0,
        owner=SimpleNamespace(spec=SimpleNamespace(mode="verify_chain", max_rows=budget + 1)),
    )
    with pytest.raises(RuntimeError, match="device handoff reached"):
        verifier.prepare(
            batch, transaction_id=1, graph_bucket=bucket,
            remaining_decode=(remaining,), device_proposal=proposal,
        )
    assert calls == [(proposal, remaining)]


def test_verifier_closes_both_journals_once() -> None:
    closed = []
    verifier = mtp_module.Qwen35GGUFTransactionalVerifier.__new__(
        mtp_module.Qwen35GGUFTransactionalVerifier
    )
    verifier.closed = False
    verifier._prepared = None
    verifier._primary_journal = SimpleNamespace(close=lambda: closed.append("native"))
    verifier._serial_journal = SimpleNamespace(close=lambda: closed.append("serial"))
    verifier.journal = verifier._serial_journal
    verifier.workspace = SimpleNamespace(free=lambda: closed.append("workspace"))
    verifier._buckets = {}
    verifier.close()
    verifier.close()
    assert closed == ["serial", "native", "workspace"]


def test_eager_verify_does_not_inherit_previous_graph_telemetry(monkeypatch):
    from hipengine.runtime.gguf_native_spec_cycle import build_native_b2_target_batch
    from hipengine.speculative import TargetAcceptSummary

    batch = build_native_b2_target_batch([7, 8], start_position=25, request_id=0)
    summary = TargetAcceptSummary.from_accept_result(
        batch, batch.accept_from_top1([9, 10], transaction_id=1, remaining_decode=(2,)),
    )
    monkeypatch.setattr(TargetAcceptSummary, "from_gpu_payload", lambda *_a: summary)
    ptr = SimpleNamespace(ptr=1)
    names = ("token_ids", "positions", "parent_rows", "draft_depths", "active_mask",
             "target_top1", "accepted_counts", "commit_rows", "commit_tokens",
             "commit_positions", "next_tokens", "full_accept")
    buffers = SimpleNamespace(**dict.fromkeys(names, ptr),
                              committed_output_ids=None, committed_output_lengths=None)
    bucket = SimpleNamespace(
        owner=SimpleNamespace(spec=SimpleNamespace(mode="verify_chain", max_rows=2),
                              bind=lambda *_a, **_kw: buffers),
        remaining_decode=ptr, replay_count=0,
    )
    journal = SimpleNamespace(
        initial_state_only=True, producer_capture_initial_state=False,
        capture_initial=lambda **_kw: None, capture_hidden_rows=lambda *_a, **_kw: None,
    )
    verifier = mtp_module.Qwen35GGUFTransactionalVerifier.__new__(mtp_module.Qwen35GGUFTransactionalVerifier)
    verifier.target = SimpleNamespace(
        position=25, runtime=object(),
        verify_target_block=lambda *_a, **_kw: SimpleNamespace(
            start_position=25, token_ids=[9, 10], pre_output_norm_hidden=object(),
        ),
        last_native_spec_target_submitted=True, last_native_spec_target_capture_ms=12,
        last_native_spec_target_submit_ms=3, last_native_spec_target_readback_ms=4,
        last_native_spec_target_fallback_reason="stale",
    )
    verifier.backend = "hip_gfx1100"
    verifier.target_verify_mode = "native"
    verifier.max_candidate_budget = 3
    verifier.closed = False
    verifier._prepared = None
    verifier.journal = verifier._primary_journal = journal
    verifier._serial_journal = None
    verifier._accept_kernel = lambda *_a, **_kw: None
    verifier._accept_library = object()
    verifier._write_verify_inputs = lambda *_a: None
    verifier._read_accept_payload = lambda *_a, **_kw: {}
    prepared = verifier.prepare(
        batch, transaction_id=1, graph_bucket=bucket, remaining_decode=(2,), allow_graph=False,
    )
    assert not prepared.native_graph_submitted
    assert prepared.native_graph_capture_ms == prepared.native_graph_submit_ms == prepared.native_graph_readback_ms == 0
    assert prepared.native_graph_fallback_reason == "native target graph disabled by caller"


def test_mtp_prompt_admission_streams_shifted_draft_without_full_hidden_slab(
    monkeypatch,
) -> None:
    pending = DeviceBuffer(0x1000, 8)
    freed: list[int] = []
    monkeypatch.setattr(mtp_module, "malloc", lambda _nbytes, *, runtime: pending)
    monkeypatch.setattr(
        mtp_module,
        "free",
        lambda buffer, *, runtime: freed.append(int(buffer.ptr)),
    )

    target_calls: list[tuple[tuple[int, ...], dict[str, object]]] = []

    class Runtime:
        def memset(self, *_args) -> None:
            pass

        def memcpy_async(self, *_args) -> None:
            pass

    class Target:
        runner = SimpleNamespace(hidden_size=4)
        runtime = Runtime()

        def prefill(self, prompt, **kwargs):
            target_calls.append((tuple(prompt), kwargs))
            sink = kwargs["target_hidden_chunk_sink"]
            sink.consume(
                request_id=kwargs["target_hidden_request_id"],
                chunk_start=0,
                hidden_ptr=0x2000,
                rows=len(prompt),
                stream=0,
            )
            sink.finish(
                request_id=kwargs["target_hidden_request_id"],
                total_rows=len(prompt),
                stream=0,
            )
            return SimpleNamespace(token_id=91)

        def step(self, *_args, **_kwargs):
            raise AssertionError("bulk MTP admission must not serial-prefill the target")

    draft_calls: list[tuple[int, int, int, int]] = []
    finish_calls: list[tuple[int, int, bool]] = []

    class Executor:
        def enqueue_prompt_rows(self, request_id, token_ids, **kwargs):
            for index, token in enumerate(token_ids):
                draft_calls.append(
                    (
                        int(request_id),
                        int(token),
                        int(kwargs["position_start"]) + index,
                        int(kwargs["target_hidden_base_ptr"])
                        + index * int(kwargs["hidden_stride_bytes"]),
                    )
                )

        def finish_prompt_priming(self, request_id, *, stream, synchronize):
            finish_calls.append((int(request_id), int(stream), bool(synchronize)))

    decoder = Qwen35GGUFMTPDecodeSession.__new__(Qwen35GGUFMTPDecodeSession)
    decoder.target = Target()
    decoder.draft_provider = SimpleNamespace(executor=Executor())

    result = decoder._prefill_target_and_draft(
        (11, 22, 33),
        request_id=7,
        use_bulk=True,
    )

    assert result.token_id == 91
    assert len(target_calls) == 1
    prompt, kwargs = target_calls[0]
    assert prompt == (11, 22, 33)
    assert kwargs["use_bulk"] is True
    assert kwargs["return_logits"] is False
    assert kwargs["target_hidden_request_id"] == 7
    assert kwargs["target_hidden_chunk_sink"].total_rows == 3
    assert "capture_target_hidden_rows" not in kwargs
    assert draft_calls == [
        (7, 11, 0, 0x1000),
        (7, 22, 1, 0x2000),
        (7, 33, 2, 0x2008),
    ]
    assert finish_calls == [(7, 0, False)]
    assert freed == [0x1000]


def test_mtp_streaming_prompt_failure_drains_staging_and_frees_pending_row(
    monkeypatch,
) -> None:
    pending = DeviceBuffer(0x1000, 8)
    freed: list[int] = []
    monkeypatch.setattr(mtp_module, "malloc", lambda _nbytes, *, runtime: pending)
    monkeypatch.setattr(
        mtp_module,
        "free",
        lambda buffer, *, runtime: freed.append(int(buffer.ptr)),
    )

    class Runtime:
        def memset(self, *_args) -> None:
            pass

        def memcpy_async(self, *_args) -> None:
            pass

    class Target:
        runner = SimpleNamespace(hidden_size=4)
        runtime = Runtime()

        def prefill(self, prompt, **kwargs):
            kwargs["target_hidden_chunk_sink"].consume(
                request_id=kwargs["target_hidden_request_id"],
                chunk_start=0,
                hidden_ptr=0x2000,
                rows=1,
                stream=0,
            )
            raise RuntimeError("injected target failure")

    finish_calls: list[tuple[int, int, bool]] = []

    class Executor:
        def enqueue_prompt_rows(self, *_args, **_kwargs) -> None:
            pass

        def finish_prompt_priming(self, request_id, *, stream, synchronize):
            finish_calls.append((int(request_id), int(stream), bool(synchronize)))

    decoder = Qwen35GGUFMTPDecodeSession.__new__(Qwen35GGUFMTPDecodeSession)
    decoder.target = Target()
    decoder.draft_provider = SimpleNamespace(executor=Executor())

    with pytest.raises(RuntimeError, match="injected target failure"):
        decoder._prefill_target_and_draft(
            (11, 22),
            request_id=7,
            use_bulk=True,
        )

    assert finish_calls == [(7, 0, True)]
    assert freed == [pending.ptr]


@pytest.mark.parametrize("chunk_size", [1, 2, 7, 8, 9])
def test_streaming_prompt_sink_preserves_shifted_rows_across_chunk_partitions(
    monkeypatch,
    chunk_size: int,
) -> None:
    hidden_size = 4
    hidden_nbytes = hidden_size * DType.BF16.itemsize
    pending = DeviceBuffer(0x1000, hidden_nbytes)
    freed: list[int] = []
    monkeypatch.setattr(mtp_module, "malloc", lambda _nbytes, *, runtime: pending)
    monkeypatch.setattr(
        mtp_module,
        "free",
        lambda buffer, *, runtime: freed.append(int(buffer.ptr)),
    )

    copies: list[tuple[int, int, int, int, int]] = []

    class Runtime:
        def memset(self, ptr, value, nbytes) -> None:
            copies.append((int(ptr), int(value), int(nbytes), -1, -1))

        def memcpy_async(self, dst, src, nbytes, kind, stream) -> None:
            copies.append((int(dst), int(src), int(nbytes), int(kind), int(stream)))

    rows: list[tuple[int, int, int, int]] = []

    class Executor:
        def enqueue_prompt_rows(
            self,
            request_id,
            token_ids,
            *,
            position_start,
            target_hidden_base_ptr,
            hidden_stride_bytes,
            stream,
        ) -> None:
            for index, token in enumerate(token_ids):
                rows.append(
                    (
                        int(request_id),
                        int(token),
                        int(position_start) + index,
                        int(target_hidden_base_ptr) + index * int(hidden_stride_bytes),
                    )
                )

    prompt = tuple(range(101, 124))
    sink = mtp_module._StreamingNextNPromptSink(
        request_id=7,
        prompt_tokens=prompt,
        hidden_size=hidden_size,
        executor=Executor(),
        runtime=Runtime(),
        checkpoint=None,
    )
    source_base = 0x4000
    chunk_starts: set[int] = set()
    for start in range(0, len(prompt), chunk_size):
        count = min(chunk_size, len(prompt) - start)
        chunk_starts.add(start)
        sink.consume(
            request_id=7,
            chunk_start=start,
            hidden_ptr=source_base + start * hidden_nbytes,
            rows=count,
            stream=3,
        )
    sink.finish(request_id=7, total_rows=len(prompt), stream=3)

    assert [(token, position) for _, token, position, _ in rows] == [
        (token, position) for position, token in enumerate(prompt)
    ]
    for _, _, position, hidden_ptr in rows:
        if position in chunk_starts:
            assert hidden_ptr == pending.ptr
        else:
            assert hidden_ptr == source_base + (position - 1) * hidden_nbytes
    pending_copies = [copy for copy in copies if copy[3] == int(HipMemcpyKind.DEVICE_TO_DEVICE)]
    assert pending_copies == [
        (
            pending.ptr,
            source_base + (min(start + chunk_size, len(prompt)) - 1) * hidden_nbytes,
            hidden_nbytes,
            int(HipMemcpyKind.DEVICE_TO_DEVICE),
            3,
        )
        for start in range(0, len(prompt), chunk_size)
    ]
    assert sink.final_pending_hidden.ptr == pending.ptr
    sink.close()
    assert freed == [pending.ptr]


def test_streaming_prompt_sink_supports_warm_offset_and_fails_closed_on_owner_mismatch(
    monkeypatch,
) -> None:
    allocations = iter((DeviceBuffer(0x1000, 8),))
    monkeypatch.setattr(mtp_module, "malloc", lambda _nbytes, *, runtime: next(allocations))
    monkeypatch.setattr(mtp_module, "free", lambda _buffer, *, runtime: None)

    copies: list[tuple[int, int, int, int]] = []

    class Runtime:
        def memcpy(self, dst, src, nbytes, kind) -> None:
            copies.append((int(dst), int(src), int(nbytes), int(kind)))

        def memcpy_async(self, dst, src, nbytes, kind, stream) -> None:
            copies.append((int(dst), int(src), int(nbytes), int(kind)))

    rows: list[tuple[int, int, int, int]] = []

    class Executor:
        def enqueue_prompt_rows(self, request_id, token_ids, **kwargs) -> None:
            for index, token in enumerate(token_ids):
                rows.append(
                    (
                        int(request_id),
                        int(token),
                        int(kwargs["position_start"]) + index,
                        int(kwargs["target_hidden_base_ptr"])
                        + index * int(kwargs["hidden_stride_bytes"]),
                    )
                )

    initial = Tensor.from_handle(0x9000, (1, 4), DType.BF16, Device("hip", 0))
    sink = mtp_module._StreamingNextNPromptSink(
        request_id=11,
        prompt_tokens=(50, 51),
        hidden_size=4,
        executor=Executor(),
        runtime=Runtime(),
        checkpoint=None,
        start_position=100,
        initial_hidden=initial,
    )
    assert copies[0] == (0x1000, 0x9000, 8, int(HipMemcpyKind.DEVICE_TO_DEVICE))
    with pytest.raises(RuntimeError, match="owner"):
        sink.consume(
            request_id=12,
            chunk_start=0,
            hidden_ptr=0x5000,
            rows=1,
            stream=0,
        )
    sink.consume(
        request_id=11,
        chunk_start=0,
        hidden_ptr=0x5000,
        rows=1,
        stream=0,
    )
    with pytest.raises(RuntimeError, match="contiguous"):
        sink.consume(
            request_id=11,
            chunk_start=0,
            hidden_ptr=0x6000,
            rows=1,
            stream=0,
        )
    sink.consume(
        request_id=11,
        chunk_start=1,
        hidden_ptr=0x6000,
        rows=1,
        stream=0,
    )
    sink.finish(request_id=11, total_rows=2, stream=0)
    assert [(token, position) for _, token, position, _ in rows] == [(50, 100), (51, 101)]
    assert rows[0][3] == rows[1][3] == 0x1000
    sink.close()


def test_draft_hidden_policy_manifest_is_immutable_and_explicit() -> None:
    strict = mtp_module.Qwen35GGUFDraftHiddenPolicy()
    candidate = mtp_module.Qwen35GGUFDraftHiddenPolicy(
        target_hidden_variant="post_output_norm"
    )

    assert strict.target_hidden_variant == "pre_output_norm"
    assert strict.manifest()["prompt_target_hidden"] == "pre_output_norm"
    assert candidate.manifest() == {
        "target_hidden_variant": "post_output_norm",
        "prompt_target_hidden": "post_output_norm",
        "proposal_target_hidden": "post_output_norm",
        "target_commit_hidden": "pre_output_norm",
        "draft_chain_hidden": "nextn_post_output_norm",
    }
    with pytest.raises(ValueError, match="target_hidden_variant"):
        mtp_module.Qwen35GGUFDraftHiddenPolicy(target_hidden_variant="adaptive")
    with pytest.raises(Exception):
        candidate.target_hidden_variant = "pre_output_norm"


def test_post_output_norm_policy_normalizes_each_target_proposal_once(
    monkeypatch,
) -> None:
    calls: list[tuple[int, int, int, int, int]] = []
    syncs: list[str] = []
    target_hidden = Tensor.from_handle(0x1000, (1, 4), DType.BF16, Device("hip", 0))
    output_norm = SimpleNamespace(
        allocation=lambda: SimpleNamespace(tensor=SimpleNamespace(ptr=0x2000))
    )
    decoder = Qwen35GGUFMTPDecodeSession.__new__(Qwen35GGUFMTPDecodeSession)
    decoder.target = SimpleNamespace(
        last_target_hidden=target_hidden,
        runner=SimpleNamespace(
            hidden_size=4,
            weights=SimpleNamespace(
                root=lambda name: output_norm,
                config=SimpleNamespace(rms_norm_eps=1.0e-6),
            ),
        ),
        runtime=SimpleNamespace(device_synchronize=lambda: syncs.append("sync")),
    )
    decoder.draft_hidden_policy = mtp_module.Qwen35GGUFDraftHiddenPolicy(
        target_hidden_variant="post_output_norm"
    )
    decoder._draft_post_norm_hidden = DeviceBuffer(0x3000, 8)

    monkeypatch.setattr(
        mtp_module,
        "gguf_rmsnorm_bf16_f32_weight",
        lambda src, weight, dst, **kwargs: calls.append(
            (
                int(src),
                int(weight),
                int(dst),
                int(kwargs["rows"]),
                int(kwargs["hidden_size"]),
            )
        ),
    )

    normalized = decoder._proposal_target_hidden()

    assert normalized.ptr == 0x3000
    assert calls == [(0x1000, 0x2000, 0x3000, 1, 4)]
    assert syncs == ["sync"]

    decoder.draft_hidden_policy = mtp_module.Qwen35GGUFDraftHiddenPolicy()
    strict = decoder._proposal_target_hidden()
    assert strict.ptr == 0x1000
    assert calls == [(0x1000, 0x2000, 0x3000, 1, 4)]


def test_post_output_norm_decoder_close_frees_owned_policy_row(
    monkeypatch,
) -> None:
    freed: list[int] = []
    monkeypatch.setattr(
        mtp_module,
        "free",
        lambda buffer, *, runtime: freed.append(int(buffer.ptr)),
    )
    verifier_closes: list[str] = []
    decoder = Qwen35GGUFMTPDecodeSession.__new__(Qwen35GGUFMTPDecodeSession)
    decoder.target = SimpleNamespace(runtime=object())
    decoder._draft_post_norm_hidden = DeviceBuffer(0x3000, 8)
    decoder.owns_verifier = False
    decoder.verifier = SimpleNamespace(close=lambda: verifier_closes.append("close"))

    decoder.close()
    decoder.close()

    assert freed == [0x3000]
    assert verifier_closes == []


@pytest.mark.parametrize("chunk_limit", [0, 1, 2, 3])
def test_streaming_prompt_sink_applies_one_consistent_target_hidden_transform(
    monkeypatch, chunk_limit,
) -> None:
    pending = DeviceBuffer(0x1000, 8)
    monkeypatch.setattr(mtp_module, "malloc", lambda _nbytes, *, runtime: pending)
    monkeypatch.setattr(mtp_module, "free", lambda _buffer, *, runtime: None)
    copies: list[tuple[int, int, int]] = []

    class Runtime:
        def memset(self, *_args) -> None:
            pass

        def memcpy_async(self, dst, src, nbytes, _kind, _stream) -> None:
            copies.append((int(dst), int(src), int(nbytes)))

    rows: list[tuple[int, int, int]] = []

    class Executor:
        def enqueue_prompt_rows(self, _request_id, token_ids, **kwargs) -> None:
            for index, token in enumerate(token_ids):
                rows.append(
                    (
                        int(token),
                        int(kwargs["position_start"]) + index,
                        int(kwargs["target_hidden_base_ptr"])
                        + index * int(kwargs["hidden_stride_bytes"]),
                    )
                )

    transforms: list[tuple[int, int, int]] = []

    def transform(hidden_ptr: int, count: int, stream: int) -> int:
        transforms.append((int(hidden_ptr), int(count), int(stream)))
        return 0x7000

    sink = mtp_module._StreamingNextNPromptSink(
        request_id=7,
        prompt_tokens=(11, 22, 33),
        hidden_size=4,
        executor=Executor(),
        runtime=Runtime(),
        checkpoint=None,
        transform_hidden_rows=transform,
        max_chunk_rows=chunk_limit,
    )
    sink.consume(
        request_id=7,
        chunk_start=0,
        hidden_ptr=0x5000,
        rows=3,
        stream=5,
    )
    sink.finish(request_id=7, total_rows=3, stream=5)

    limit = chunk_limit or 3
    chunks = [(start, min(limit, 3 - start)) for start in range(0, 3, limit)]
    assert transforms == [(0x5000 + start * 8, count, 5) for start, count in chunks]
    expected_rows = []
    for start, count in chunks:
        expected_rows.append(((11, 22, 33)[start], start, 0x1000))
        expected_rows.extend(
            ((11, 22, 33)[start + i], start + i, 0x7000 + (i - 1) * 8)
            for i in range(1, count)
        )
    assert rows == expected_rows
    assert copies == [(0x1000, 0x7000 + (count - 1) * 8, 8) for _, count in chunks]
    sink.close()


@pytest.mark.parametrize("cancel", [False, True])
def test_bounded_streaming_transform_reuses_buffer_in_stream_order(monkeypatch, cancel):
    # Device reads execute later, not at enqueue time: overwriting the shared
    # normalization buffer early would corrupt the values this test observes.
    memory = {0x5000 + i * 8: i for i in range(7)}
    memory[0x2000] = 999
    queue, consumed, captures, transformed, freed = [], [], [], [], []
    monkeypatch.setattr(mtp_module, "malloc", lambda nbytes, **kw: DeviceBuffer(0x1000, nbytes))
    monkeypatch.setattr(mtp_module, "free", lambda buffer, **kw: freed.append(buffer.ptr))

    class Runtime:
        def memcpy(self, dst, src, nbytes, kind):
            memory[dst] = memory[src]

        def memcpy_async(self, dst, src, nbytes, kind, stream):
            assert stream == 3
            queue.append(lambda: memory.__setitem__(dst, memory[src]))

    class Executor:
        def enqueue_prompt_rows(self, request_id, token_ids, **kwargs):
            assert kwargs["stream"] == 3
            ptr = kwargs["target_hidden_base_ptr"]
            position = kwargs["position_start"]
            tokens = tuple(token_ids)
            queue.append(lambda: consumed.extend(
                (position + i, token, memory[ptr + i * 8]) for i, token in enumerate(tokens)
            ))

    def transform(src, count, stream):
        assert count <= 4 and stream == 3
        transformed.append(count)
        def normalize():
            for i in range(count):
                memory[0x7000 + i * 8] = memory[src + i * 8] + 1000
        queue.append(normalize)
        return 0x7000

    def capture(boundary, ptr):
        queue.append(lambda: captures.append((boundary, memory[ptr])))

    checkpoints = 0
    def checkpoint():
        nonlocal checkpoints
        checkpoints += 1
        if cancel and checkpoints == 2:
            raise RuntimeError("cancelled between chunks")

    sink = mtp_module._StreamingNextNPromptSink(
        request_id=7, prompt_tokens=tuple(range(11, 18)), hidden_size=4,
        executor=Executor(), runtime=Runtime(), checkpoint=checkpoint,
        start_position=10,
        initial_hidden=Tensor.from_handle(0x2000, (1, 4), DType.BF16, Device("hip", 0)),
        transform_hidden_rows=transform, prefix_capture=capture, max_chunk_rows=4,
    )
    sink.capture_boundaries = (12, 14, 17)
    if cancel:
        with pytest.raises(RuntimeError, match="cancelled between chunks"):
            sink.consume(request_id=7, chunk_start=0, hidden_ptr=0x5000, rows=7, stream=3)
    else:
        sink.consume(request_id=7, chunk_start=0, hidden_ptr=0x5000, rows=7, stream=3)
        sink.finish(request_id=7, total_rows=7, stream=3)
    for operation in queue:
        operation()
    expected_rows = 2 if cancel else 7
    assert consumed == [
        (10 + i, 11 + i, 999 if i == 0 else 999 + i)
        for i in range(expected_rows)
    ]
    assert captures == ([(12, 1001)] if cancel else [(12, 1001), (14, 1003), (17, 1006)])
    assert transformed == ([2] if cancel else [2, 2, 3])
    sink.close()
    sink.close()
    assert freed == [0x1000]


def test_streaming_prompt_sink_cancellation_between_chunks_releases_owned_row(
    monkeypatch,
) -> None:
    pending = DeviceBuffer(0x1000, 8)
    freed: list[int] = []
    monkeypatch.setattr(mtp_module, "malloc", lambda _nbytes, *, runtime: pending)
    monkeypatch.setattr(
        mtp_module,
        "free",
        lambda buffer, *, runtime: freed.append(int(buffer.ptr)),
    )

    class Runtime:
        def memset(self, *_args) -> None:
            pass

        def memcpy_async(self, *_args) -> None:
            pass

    calls: list[tuple[int, ...]] = []

    class Executor:
        def enqueue_prompt_rows(self, _request_id, token_ids, **_kwargs) -> None:
            calls.append(tuple(int(token) for token in token_ids))

    checkpoints = 0

    def checkpoint() -> None:
        nonlocal checkpoints
        checkpoints += 1
        if checkpoints == 2:
            raise GenerationCancelled()

    sink = mtp_module._StreamingNextNPromptSink(
        request_id=7,
        prompt_tokens=(1, 2, 3, 4),
        hidden_size=4,
        executor=Executor(),
        runtime=Runtime(),
        checkpoint=checkpoint,
    )
    try:
        sink.consume(
            request_id=7,
            chunk_start=0,
            hidden_ptr=0x5000,
            rows=2,
            stream=0,
        )
        with pytest.raises(GenerationCancelled):
            sink.consume(
                request_id=7,
                chunk_start=2,
                hidden_ptr=0x6000,
                rows=2,
                stream=0,
            )
        assert sink.consumed_rows == 2
        assert calls == [(1,), (2,)]
    finally:
        sink.close()
    assert freed == [pending.ptr]


def test_streaming_prompt_sink_transfers_final_carried_row_ownership(
    monkeypatch,
) -> None:
    pending = DeviceBuffer(0x1000, 8)
    freed: list[int] = []
    monkeypatch.setattr(mtp_module, "malloc", lambda _nbytes, *, runtime: pending)
    monkeypatch.setattr(
        mtp_module,
        "free",
        lambda buffer, *, runtime: freed.append(int(buffer.ptr)),
    )

    class Runtime:
        def memset(self, *_args) -> None:
            pass

        def memcpy_async(self, *_args) -> None:
            pass

    class Executor:
        def enqueue_prompt_rows(self, *_args, **_kwargs) -> None:
            pass

    sink = mtp_module._StreamingNextNPromptSink(
        request_id=7,
        prompt_tokens=(11, 22),
        hidden_size=4,
        executor=Executor(),
        runtime=Runtime(),
        checkpoint=None,
    )
    sink.consume(
        request_id=7,
        chunk_start=0,
        hidden_ptr=0x5000,
        rows=2,
        stream=0,
    )
    sink.finish(request_id=7, total_rows=2, stream=0)

    transferred = sink.take_final_pending_buffer()
    sink.close()

    assert transferred is pending
    assert freed == []
    with pytest.raises(RuntimeError, match="already transferred"):
        sink.take_final_pending_buffer()


def test_streaming_prompt_sink_splits_multiple_cache_boundaries(monkeypatch):
    monkeypatch.setattr(mtp_module, "malloc", lambda n, **kw: DeviceBuffer(0x1000, n))
    monkeypatch.setattr(mtp_module, "free", lambda *args, **kw: None)
    runtime = SimpleNamespace(memset=lambda *a: None, memcpy_async=lambda *a: None)
    calls, captured = [], []
    executor = SimpleNamespace(enqueue_prompt_rows=lambda *a, **kw: calls.append((a, kw)))
    sink = mtp_module._StreamingNextNPromptSink(
        request_id=7, prompt_tokens=(10, 20, 30, 40, 50), hidden_size=4,
        executor=executor, runtime=runtime, checkpoint=None,
        prefix_capture=lambda boundary, ptr: captured.append((boundary, ptr)), capture_boundary=5,
    )
    sink.capture_boundaries = (2, 4, 5)
    sink.consume(request_id=7, chunk_start=0, hidden_ptr=0x5000, rows=5, stream=0)
    sink.finish(request_id=7, total_rows=5, stream=0)
    assert captured == [(2, 0x5008), (4, 0x5018), (5, 0x5020)]
    assert [kw["position_start"] for _, kw in calls] == [0, 1, 2, 3, 4]
    sink.close()


def test_mtp_prompt_admission_preserves_target_default_bulk_selector(monkeypatch) -> None:
    allocations = iter((DeviceBuffer(0x1000, 8), DeviceBuffer(0x2000, 8)))
    monkeypatch.setattr(mtp_module, "malloc", lambda _nbytes, *, runtime: next(allocations))
    monkeypatch.setattr(mtp_module, "free", lambda _buffer, *, runtime: None)
    target_calls: list[object] = []

    class Target:
        runner = SimpleNamespace(hidden_size=4)
        runtime = SimpleNamespace(memset=lambda *_args: None)

        def prefill(self, _prompt, **kwargs):
            target_calls.append(kwargs["use_bulk"])
            return SimpleNamespace(token_id=91)

    decoder = Qwen35GGUFMTPDecodeSession.__new__(Qwen35GGUFMTPDecodeSession)
    decoder.target = Target()
    decoder.draft_provider = SimpleNamespace(
        executor=SimpleNamespace(run_step=lambda *_args, **_kwargs: None)
    )

    decoder._prefill_target_and_draft((11,), request_id=7, use_bulk=None)

    assert target_calls == [None]


def test_mtp_deadline_checkpoint_raises_on_expiry() -> None:
    assert mtp_module._deadline_checkpoint(None) is None
    future = mtp_module._deadline_checkpoint(time.monotonic() + 60.0)
    assert future is not None
    future()
    past = mtp_module._deadline_checkpoint(time.monotonic() - 1.0)
    with pytest.raises(mtp_module.GenerationDeadlineExceeded) as excinfo:
        past()
    assert excinfo.value.deadline_at is not None


def test_mtp_custom_checkpoint_propagates_error() -> None:
    def boom() -> None:
        raise RuntimeError("injected fault")

    with pytest.raises(RuntimeError, match="injected fault"):
        mtp_module._mtp_cycle_checkpoint(boom)


def test_mtp_rollback_decision_never_reopens_committed_transaction() -> None:
    open_txn = SimpleNamespace(committed=False, rolled_back=False)
    assert mtp_module._mtp_should_rollback_transaction(
        target_committed=False,
        transaction=open_txn,
    ) is True
    assert mtp_module._mtp_should_rollback_transaction(
        target_committed=True,
        transaction=open_txn,
    ) is False
    assert mtp_module._mtp_should_rollback_transaction(
        target_committed=False,
        transaction=SimpleNamespace(committed=True, rolled_back=False),
    ) is False


def test_mtp_lifecycle_phase_is_stable_and_propagates_fault() -> None:
    seen: list[str] = []
    mtp_module._mtp_lifecycle_phase(seen.append, "after_target_prepare")
    assert seen == ["after_target_prepare"]

    def fault(phase: str) -> None:
        raise RuntimeError(f"injected:{phase}")

    with pytest.raises(RuntimeError, match="injected:after_target_commit"):
        mtp_module._mtp_lifecycle_phase(fault, "after_target_commit")


def test_mtp_generate_cancellation_precedes_proposal_mutation(monkeypatch) -> None:
    """A checkpoint that raises at the first cycle boundary must stop before
    any proposal/target work: no draft propose, no device proposal, no
    verifier prepare."""

    proposed: list[tuple[object, ...]] = []
    device_proposals: list[tuple[object, ...]] = []
    prepared: list[tuple[object, ...]] = []
    budget_events: list[tuple[object, ...]] = []

    class BudgetPolicy:
        def start_request(self, *, request_id, max_budget):
            budget_events.append(("start", int(request_id), int(max_budget)))

        def choose_budget(self, **kwargs):
            budget_events.append(("choose", kwargs))
            raise AssertionError("cancellation must prevent budget choice")

        def record_cycle(self, result):
            budget_events.append(("record", result))

        def summary(self):
            return {"kind": "test"}

    class FakeScheduler:
        def __init__(self, capacity: int = 1) -> None:
            self.completed: set[int] = set()
            self._rid = 7

        def submit(self, _prompt, *, max_new_tokens, request_id):
            self._rid = int(request_id)
            return self._rid

        def admit_pending(self) -> None:
            pass

        def next_prefill_work(self, chunk_size) -> None:
            pass

        def record_generated(self, _rows) -> None:
            pass

        def finish_request_at_stop(self, rid, *, eos_token_id, stop_token_ids) -> None:
            pass

        @property
        def active_batch(self):
            return SimpleNamespace(
                requests={self._rid: SimpleNamespace(remaining_decode=4)}
            )

    monkeypatch.setattr(mtp_module, "ResidentBatchScheduler", FakeScheduler)

    decoder = Qwen35GGUFMTPDecodeSession.__new__(Qwen35GGUFMTPDecodeSession)
    decoder.candidate_budget = 2
    decoder.target = SimpleNamespace(reset=lambda: None)
    decoder.draft_provider = SimpleNamespace(reset_request=lambda _rid: None)
    decoder.verifier = SimpleNamespace()

    def fake_prefill(_prompt, *, request_id, use_bulk, checkpoint=None):
        return SimpleNamespace(token_id=11)

    decoder._prefill_target_and_draft = fake_prefill

    def fake_register_policy(_rid):
        return None

    decoder._register_kv_policy = fake_register_policy

    def fake_propose(_context, *, candidate_budget, return_logits):
        proposed.append((int(candidate_budget), bool(return_logits)))
        raise AssertionError("cancellation must prevent draft proposal")

    decoder.draft_provider.propose = fake_propose

    def fake_device_proposal(*_args, **_kwargs):
        device_proposals.append((_args, _kwargs))
        raise AssertionError("cancellation must prevent device proposal launch")

    monkeypatch.setattr(mtp_module, "_maybe_launch_device_proposal", fake_device_proposal)

    def fake_prepare(*_args, **_kwargs):
        prepared.append((_args, _kwargs))
        raise AssertionError("cancellation must prevent target verify")

    decoder.verifier.prepare = fake_prepare

    def cancel_before_first_cycle() -> None:
        raise GenerationCancelled()

    with pytest.raises(GenerationCancelled):
        decoder.generate(
            (1, 2, 3),
            max_new_tokens=4,
            request_id=7,
            checkpoint=cancel_before_first_cycle,
            budget_policy=BudgetPolicy(),
        )

    assert budget_events == [("start", 7, 2)]
    assert proposed == []
    assert device_proposals == []
    assert prepared == []


def test_adopt_accept_summary_keeps_the_open_transaction_identity() -> None:
    """A recomputed accept summary must be installed on the open transaction.

    The prepared object *is* the transaction's identity: commit and rollback
    accept only the instance prepare returned, and commit validates its plan
    against ``prepared.summary``. The sampled route recomputes the accept
    decision, and copying the prepared object instead of installing the summary
    on it made the verifier refuse the copy as "not the open transaction", which
    disabled the request and silently fell back to autoregressive decoding.
    """

    import dataclasses

    import numpy as np

    verifier = mtp_module.Qwen35GGUFTransactionalVerifier.__new__(
        mtp_module.Qwen35GGUFTransactionalVerifier
    )
    prepared = mtp_module.Qwen35GGUFPreparedVerify(
        batch=SimpleNamespace(),
        buffers=SimpleNamespace(),
        summary="argmax-summary",
        target_top1=(1,),
        target_logits=np.zeros((1, 1), dtype=np.float32),
        graph_bucket=SimpleNamespace(),
        initial_position=0,
        kv_journal_positions=(0,),
        gpu_accept_match_cpu=True,
        target_verify_mode="native",
    )
    verifier._prepared = prepared

    adopted = verifier.adopt_accept_summary(prepared, "sampled-summary")
    assert adopted is prepared
    assert prepared.summary == "sampled-summary"

    # A copy is still refused: the identity check is what commit relies on.
    copy = dataclasses.replace(prepared, summary="copied-summary")
    assert copy is not prepared
    with pytest.raises(ValueError, match="not the open transaction"):
        verifier.adopt_accept_summary(copy, "other-summary")
    with pytest.raises(ValueError, match="not the open transaction"):
        verifier._require_open(copy)


def test_sink_captures_once_at_the_boundary_it_was_configured_for(monkeypatch) -> None:
    """The capture fires on the chunk whose cumulative rows reach the boundary.

    The boundary row is the chunk's last row -- the row the sink carries
    forward and the next draft step consumes -- so a capture that used the
    chunk's first row, or fired on the chunk before the boundary, would
    snapshot a state that cannot be resumed from.
    """

    hidden_size = 4
    hidden_nbytes = hidden_size * DType.BF16.itemsize
    pending = DeviceBuffer(0x1000, hidden_nbytes)
    monkeypatch.setattr(mtp_module, "malloc", lambda _nbytes, *, runtime: pending)
    monkeypatch.setattr(mtp_module, "free", lambda buffer, *, runtime: None)

    class Runtime:
        def memset(self, ptr, value, nbytes) -> None:
            return None

        def memcpy_async(self, dst, src, nbytes, kind, stream) -> None:
            return None

    class Executor:
        def enqueue_prompt_rows(self, *args, **kwargs) -> None:
            return None

    prompt = tuple(range(101, 111))
    captures: list[tuple[int, int]] = []
    sink = mtp_module._StreamingNextNPromptSink(
        request_id=7,
        prompt_tokens=prompt,
        hidden_size=hidden_size,
        executor=Executor(),
        runtime=Runtime(),
        checkpoint=None,
        prefix_capture=lambda boundary, ptr: captures.append((int(boundary), int(ptr))),
        capture_boundary=8,
    )
    source_base = 0x4000
    for start in range(0, len(prompt), 4):
        count = min(4, len(prompt) - start)
        sink.consume(
            request_id=7,
            chunk_start=start,
            hidden_ptr=source_base + start * hidden_nbytes,
            rows=count,
            stream=3,
        )

    # Boundary 8 lands on the second chunk's end, and the captured row is that
    # chunk's last row rather than the carried row or the first row.
    assert captures == [(8, source_base + 7 * hidden_nbytes)]
    assert sink._capture_missed is False


def test_sink_capture_is_optional_and_contained(monkeypatch) -> None:
    hidden_size = 4
    hidden_nbytes = hidden_size * DType.BF16.itemsize
    pending = DeviceBuffer(0x1000, hidden_nbytes)
    monkeypatch.setattr(mtp_module, "malloc", lambda _nbytes, *, runtime: pending)
    monkeypatch.setattr(mtp_module, "free", lambda buffer, *, runtime: None)

    class Runtime:
        def memset(self, ptr, value, nbytes) -> None:
            return None

        def memcpy_async(self, dst, src, nbytes, kind, stream) -> None:
            return None

    class Executor:
        def enqueue_prompt_rows(self, *args, **kwargs) -> None:
            return None

    def boom(boundary: int, ptr: int) -> None:
        raise RuntimeError("snapshot refused")

    prompt = tuple(range(101, 105))
    sink = mtp_module._StreamingNextNPromptSink(
        request_id=7,
        prompt_tokens=prompt,
        hidden_size=hidden_size,
        executor=Executor(),
        runtime=Runtime(),
        checkpoint=None,
        prefix_capture=boom,
        capture_boundary=4,
    )
    # A refused capture must not fail the request's prefill.
    sink.consume(request_id=7, chunk_start=0, hidden_ptr=0x4000, rows=4, stream=3)
    sink.finish(request_id=7, total_rows=4, stream=3)

    assert sink.consumed_rows == 4
    assert sink._capture_failed is True


def test_sink_records_a_boundary_that_no_chunk_ends_on(monkeypatch) -> None:
    """An unreachable boundary is recorded, not silently skipped.

    One chunk's enqueues are batched, so the provider cursor only sits at a
    chunk end. A boundary inside a chunk can never be captured by this route,
    and a future change to the chunk width would otherwise disable capture with
    nothing to see.
    """

    hidden_size = 4
    hidden_nbytes = hidden_size * DType.BF16.itemsize
    pending = DeviceBuffer(0x1000, hidden_nbytes)
    monkeypatch.setattr(mtp_module, "malloc", lambda _nbytes, *, runtime: pending)
    monkeypatch.setattr(mtp_module, "free", lambda buffer, *, runtime: None)

    class Runtime:
        def memset(self, ptr, value, nbytes) -> None:
            return None

        def memcpy_async(self, dst, src, nbytes, kind, stream) -> None:
            return None

    class Executor:
        def enqueue_prompt_rows(self, *args, **kwargs) -> None:
            return None

    captures: list[tuple[int, int]] = []
    sink = mtp_module._StreamingNextNPromptSink(
        request_id=7,
        prompt_tokens=tuple(range(101, 109)),
        hidden_size=hidden_size,
        executor=Executor(),
        runtime=Runtime(),
        checkpoint=None,
        prefix_capture=lambda boundary, ptr: captures.append((int(boundary), int(ptr))),
        capture_boundary=6,
    )
    sink.consume(request_id=7, chunk_start=0, hidden_ptr=0x4000, rows=4, stream=3)
    assert sink._capture_missed is False
    sink.consume(request_id=7, chunk_start=4, hidden_ptr=0x4010, rows=4, stream=3)

    assert captures == []
    assert sink._capture_missed is True


class _PrimingOracle:
    """Byte-backed target/draft stream oracle for decoder priming tests."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.hidden_size = 4
        self.hidden_nbytes = self.hidden_size * DType.BF16.itemsize
        self.source_base = 0x5000
        self.output_norm_ptr = 0x3000
        self.memory: dict[int, bytes] = {}
        self.allocations: list[tuple[int, int]] = []
        self.freed: list[int] = []
        self.queue: list = []
        self.normalize_calls: list[tuple[int, int, int, int, int, int]] = []
        self.consumed: list[tuple[int, int, bytes]] = []
        self.finish_calls: list[tuple[int, int, bool]] = []
        self.prefill_calls: list[tuple[tuple[int, ...], object]] = []
        self.syncs: list[str] = []
        self._next_ptr = 0x1000
        self.runtime = _PrimingRuntime(self)
        monkeypatch.setattr(mtp_module, "malloc", self._malloc)
        monkeypatch.setattr(mtp_module, "free", self._free)
        monkeypatch.setattr(
            mtp_module,
            "gguf_rmsnorm_bf16_f32_weight",
            self._normalize,
        )

    def _malloc(self, nbytes: int, *, runtime) -> DeviceBuffer:
        ptr = self._next_ptr
        self._next_ptr += 0x100
        self.allocations.append((ptr, int(nbytes)))
        return DeviceBuffer(ptr, int(nbytes))

    def _free(self, buffer: DeviceBuffer, *, runtime) -> None:
        # Model a synchronous device free: all queued stream work retires first.
        self.drain()
        self.freed.append(int(buffer.ptr))

    def write(self, ptr: int, data: bytes) -> None:
        self.memory[int(ptr)] = bytes(data)

    def read(self, ptr: int, nbytes: int) -> bytes:
        return self.memory.get(int(ptr), b"\x00" * int(nbytes))[: int(nbytes)]

    def _normalize(self, src, weight, dst, **kwargs) -> None:
        rows = int(kwargs["rows"])
        hidden_size = int(kwargs["hidden_size"])
        stream = int(kwargs["stream"])
        self.normalize_calls.append(
            (
                int(src),
                int(weight),
                int(dst),
                rows,
                hidden_size,
                stream,
            )
        )

        def _run() -> None:
            for row in range(rows):
                source = self.read(
                    int(src) + row * self.hidden_nbytes,
                    self.hidden_nbytes,
                )
                self.write(
                    int(dst) + row * self.hidden_nbytes,
                    bytes((value + 100) % 256 for value in source),
                )

        self.queue.append(_run)

    def drain(self) -> None:
        while self.queue:
            self.queue.pop(0)()

    def source_row(self, row: int) -> bytes:
        return bytes((row * 17 + index) % 251 for index in range(self.hidden_nbytes))

    def seed_source(self, prompt_length: int) -> None:
        for row in range(prompt_length):
            self.write(
                self.source_base + row * self.hidden_nbytes,
                self.source_row(row),
            )


def _source_row_bytes(row: int, nbytes: int) -> bytes:
    return bytes((row * 17 + index) % 251 for index in range(nbytes))


class _PrimingRuntime:
    def __init__(self, oracle: _PrimingOracle) -> None:
        self.oracle = oracle

    def memset(self, ptr, value, nbytes) -> None:
        self.oracle.write(ptr, bytes([int(value)]) * int(nbytes))

    def memcpy(self, dst, src, nbytes, kind) -> None:
        self.oracle.write(dst, self.oracle.read(src, int(nbytes)))

    def memcpy_async(self, dst, src, nbytes, kind, stream) -> None:
        def _run() -> None:
            self.oracle.write(dst, self.oracle.read(src, int(nbytes)))

        self.oracle.queue.append(_run)

    def device_synchronize(self) -> None:
        self.oracle.syncs.append("device")

    def stream_synchronize(self, stream) -> None:
        self.oracle.syncs.append(f"stream:{int(stream)}")


class _PrimingExecutor:
    def __init__(self, oracle: _PrimingOracle) -> None:
        self.oracle = oracle

    def enqueue_prompt_rows(self, request_id, token_ids, **kwargs) -> None:
        ptr = int(kwargs["target_hidden_base_ptr"])
        stride = int(kwargs["hidden_stride_bytes"])
        position = int(kwargs["position_start"])
        tokens = tuple(int(token) for token in token_ids)

        def _run() -> None:
            for index, token in enumerate(tokens):
                self.oracle.consumed.append(
                    (
                        position + index,
                        token,
                        self.oracle.read(
                            ptr + index * stride,
                            self.oracle.hidden_nbytes,
                        ),
                    )
                )

        self.oracle.queue.append(_run)

    def finish_prompt_priming(self, request_id, *, stream, synchronize) -> None:
        self.oracle.finish_calls.append(
            (int(request_id), int(stream), bool(synchronize))
        )


class _PrimingTarget:
    def __init__(self, oracle: _PrimingOracle, *, fail_after_chunks=None) -> None:
        self.oracle = oracle
        self.fail_after_chunks = fail_after_chunks
        self.runner = SimpleNamespace(
            hidden_size=oracle.hidden_size,
            weights=SimpleNamespace(
                root=lambda name: SimpleNamespace(
                    allocation=lambda: SimpleNamespace(
                        tensor=SimpleNamespace(ptr=oracle.output_norm_ptr)
                    )
                ),
                config=SimpleNamespace(
                    ssm_conv_kernel=4,
                    rms_norm_eps=1.0e-6,
                ),
            ),
        )
        self.runtime = oracle.runtime
        self._prefill_hidden_a = DeviceBuffer(
            0x8000,
            4 * oracle.hidden_nbytes,
        )

    def prefill(self, prompt, **kwargs):
        self.oracle.prefill_calls.append((tuple(prompt), kwargs["use_bulk"]))
        sink = kwargs["target_hidden_chunk_sink"]
        self.oracle.seed_source(len(prompt))
        bulk = kwargs["use_bulk"] is True or (
            kwargs["use_bulk"] is None and len(prompt) >= 4
        )
        chunk_rows = 4 if bulk else 1
        for chunk_index, start in enumerate(range(0, len(prompt), chunk_rows)):
            if (
                self.fail_after_chunks is not None
                and chunk_index >= self.fail_after_chunks
            ):
                raise RuntimeError("injected target failure")
            count = min(chunk_rows, len(prompt) - start)
            sink.consume(
                request_id=kwargs["target_hidden_request_id"],
                chunk_start=start,
                hidden_ptr=self.oracle.source_base
                + start * self.oracle.hidden_nbytes,
                rows=count,
                stream=3,
            )
        sink.finish(
            request_id=kwargs["target_hidden_request_id"],
            total_rows=len(prompt),
            stream=3,
        )
        return SimpleNamespace(token_id=91)


def _new_priming_decoder(
    oracle: _PrimingOracle,
    *,
    variant: str,
    target: _PrimingTarget | None = None,
) -> Qwen35GGUFMTPDecodeSession:
    decoder = Qwen35GGUFMTPDecodeSession.__new__(Qwen35GGUFMTPDecodeSession)
    decoder.target = target or _PrimingTarget(oracle)
    decoder.draft_provider = SimpleNamespace(executor=_PrimingExecutor(oracle))
    decoder.draft_hidden_policy = mtp_module.Qwen35GGUFDraftHiddenPolicy(
        target_hidden_variant=variant
    )
    return decoder


@pytest.mark.parametrize("variant", ["pre_output_norm", "post_output_norm"])
@pytest.mark.parametrize("prompt_length", [3, 5, 7, 8, 9])
def test_decoder_prefill_hidden_convention_across_chunk_boundaries(
    monkeypatch: pytest.MonkeyPatch,
    variant: str,
    prompt_length: int,
) -> None:
    oracle = _PrimingOracle(monkeypatch)
    prompt = tuple(range(11, 11 + prompt_length))
    decoder = _new_priming_decoder(oracle, variant=variant)

    result = decoder._prefill_target_and_draft(
        prompt,
        request_id=7,
        use_bulk=None,
    )

    assert result.token_id == 91
    assert oracle.prefill_calls == [(prompt, None)]
    assert oracle.finish_calls == [(7, 0, False)]
    chunk_rows = 4 if prompt_length >= 4 else 1
    chunks = [
        (start, min(chunk_rows, prompt_length - start))
        for start in range(0, prompt_length, chunk_rows)
    ]

    pending_ptr = oracle.allocations[-1][0]
    if variant == "post_output_norm":
        transform_ptr = oracle.allocations[0][0]
        assert oracle.normalize_calls == [
            (
                oracle.source_base + start * oracle.hidden_nbytes,
                oracle.output_norm_ptr,
                transform_ptr,
                count,
                oracle.hidden_size,
                3,
            )
            for start, count in chunks
        ]
        assert oracle.freed == [pending_ptr, transform_ptr]
    else:
        assert oracle.normalize_calls == []
        assert oracle.freed == [pending_ptr]

    zero = b"\x00" * oracle.hidden_nbytes
    expected_rows = []
    for index, token in enumerate(prompt):
        if index == 0:
            hidden = zero
        else:
            source = _source_row_bytes(index - 1, oracle.hidden_nbytes)
            hidden = (
                bytes((value + 100) % 256 for value in source)
                if variant == "post_output_norm"
                else source
            )
        expected_rows.append((index, token, hidden))
    assert oracle.consumed == expected_rows


def test_post_output_norm_prefill_failure_releases_transform_and_pending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    oracle = _PrimingOracle(monkeypatch)
    target = _PrimingTarget(oracle, fail_after_chunks=1)
    decoder = _new_priming_decoder(
        oracle,
        variant="post_output_norm",
        target=target,
    )

    with pytest.raises(RuntimeError, match="injected target failure"):
        decoder._prefill_target_and_draft(
            tuple(range(11, 19)),
            request_id=7,
            use_bulk=True,
        )

    transform_ptr, pending_ptr = (ptr for ptr, _ in oracle.allocations)
    assert oracle.finish_calls == [(7, 0, True)]
    assert oracle.normalize_calls == [
        (
            oracle.source_base,
            oracle.output_norm_ptr,
            transform_ptr,
            4,
            oracle.hidden_size,
            3,
        )
    ]
    assert oracle.freed == [pending_ptr, transform_ptr]


def test_post_output_norm_prefill_cancellation_retires_stream_and_buffers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    oracle = _PrimingOracle(monkeypatch)
    decoder = _new_priming_decoder(oracle, variant="post_output_norm")
    checkpoints = 0

    def checkpoint() -> None:
        nonlocal checkpoints
        checkpoints += 1
        if checkpoints == 2:
            raise GenerationCancelled()

    with pytest.raises(GenerationCancelled):
        decoder._prefill_target_and_draft(
            tuple(range(11, 19)),
            request_id=7,
            use_bulk=True,
            checkpoint=checkpoint,
        )

    transform_ptr, pending_ptr = (ptr for ptr, _ in oracle.allocations)
    assert oracle.finish_calls == [(7, 0, True)]
    assert oracle.normalize_calls == [
        (
            oracle.source_base,
            oracle.output_norm_ptr,
            transform_ptr,
            4,
            oracle.hidden_size,
            3,
        )
    ]
    assert oracle.freed == [pending_ptr, transform_ptr]


def test_post_output_norm_prefill_finish_failure_on_success_cleans_and_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    oracle = _PrimingOracle(monkeypatch)
    decoder = _new_priming_decoder(oracle, variant="post_output_norm")
    finish_calls: list[tuple[int, int, bool]] = []

    def fail_finish(request_id, *, stream, synchronize) -> None:
        finish_calls.append((int(request_id), int(stream), bool(synchronize)))
        raise RuntimeError("injected finish failure")

    decoder.draft_provider.executor.finish_prompt_priming = fail_finish

    with pytest.raises(RuntimeError, match="injected finish failure"):
        decoder._prefill_target_and_draft(
            tuple(range(11, 19)),
            request_id=7,
            use_bulk=True,
        )

    transform_ptr, pending_ptr = (ptr for ptr, _ in oracle.allocations)
    assert finish_calls == [(7, 0, False)]
    assert oracle.normalize_calls == [
        (
            oracle.source_base + start * oracle.hidden_nbytes,
            oracle.output_norm_ptr,
            transform_ptr,
            4,
            oracle.hidden_size,
            3,
        )
        for start in (0, 4)
    ]
    assert oracle.freed == [pending_ptr, transform_ptr]
    assert len(oracle.consumed) == 8


def test_post_output_norm_prefill_finish_failure_does_not_mask_operational_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    oracle = _PrimingOracle(monkeypatch)
    target = _PrimingTarget(oracle, fail_after_chunks=1)
    decoder = _new_priming_decoder(
        oracle,
        variant="post_output_norm",
        target=target,
    )
    finish_calls: list[tuple[int, int, bool]] = []

    def fail_finish(request_id, *, stream, synchronize) -> None:
        finish_calls.append((int(request_id), int(stream), bool(synchronize)))
        raise RuntimeError("injected finish failure")

    decoder.draft_provider.executor.finish_prompt_priming = fail_finish

    with pytest.raises(RuntimeError) as excinfo:
        decoder._prefill_target_and_draft(
            tuple(range(11, 19)),
            request_id=7,
            use_bulk=True,
        )

    transform_ptr, pending_ptr = (ptr for ptr, _ in oracle.allocations)
    assert "injected target failure" in str(excinfo.value)
    notes = "\n".join(getattr(excinfo.value, "__notes__", ()))
    assert "injected finish failure" in notes
    assert finish_calls == [(7, 0, True)]
    assert oracle.freed == [pending_ptr, transform_ptr]


def test_post_output_norm_prefill_sink_close_failure_still_frees_transform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    oracle = _PrimingOracle(monkeypatch)
    decoder = _new_priming_decoder(oracle, variant="post_output_norm")
    close_attempts: list[int] = []

    def fail_close(sink) -> None:
        close_attempts.append(int(sink._pending_hidden.ptr))
        raise RuntimeError("injected sink close failure")

    monkeypatch.setattr(
        mtp_module._StreamingNextNPromptSink,
        "close",
        fail_close,
    )

    with pytest.raises(RuntimeError, match="injected sink close failure"):
        decoder._prefill_target_and_draft(
            tuple(range(11, 19)),
            request_id=7,
            use_bulk=True,
        )

    transform_ptr, pending_ptr = (ptr for ptr, _ in oracle.allocations)
    assert close_attempts == [pending_ptr]
    assert oracle.freed == [transform_ptr]


def test_post_output_norm_prefill_sink_close_failure_does_not_mask_operational_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    oracle = _PrimingOracle(monkeypatch)
    target = _PrimingTarget(oracle, fail_after_chunks=1)
    decoder = _new_priming_decoder(
        oracle,
        variant="post_output_norm",
        target=target,
    )
    close_attempts: list[int] = []

    def fail_close(sink) -> None:
        close_attempts.append(int(sink._pending_hidden.ptr))
        raise RuntimeError("injected sink close failure")

    monkeypatch.setattr(
        mtp_module._StreamingNextNPromptSink,
        "close",
        fail_close,
    )

    with pytest.raises(RuntimeError) as excinfo:
        decoder._prefill_target_and_draft(
            tuple(range(11, 19)),
            request_id=7,
            use_bulk=True,
        )

    transform_ptr, pending_ptr = (ptr for ptr, _ in oracle.allocations)
    assert "injected target failure" in str(excinfo.value)
    notes = "\n".join(getattr(excinfo.value, "__notes__", ()))
    assert "injected sink close failure" in notes
    assert close_attempts == [pending_ptr]
    assert oracle.freed == [transform_ptr]


def test_post_output_norm_proposal_repeat_and_commit_tap(monkeypatch) -> None:
    calls: list[tuple[int, int, int, int, int]] = []
    syncs: list[str] = []
    first_hidden = Tensor.from_handle(0x1000, (1, 4), DType.BF16, Device("hip", 0))
    second_hidden = Tensor.from_handle(0x1100, (1, 4), DType.BF16, Device("hip", 0))
    output_norm = SimpleNamespace(
        allocation=lambda: SimpleNamespace(tensor=SimpleNamespace(ptr=0x2000))
    )
    target = SimpleNamespace(
        last_target_hidden=first_hidden,
        runner=SimpleNamespace(
            hidden_size=4,
            weights=SimpleNamespace(
                root=lambda name: output_norm,
                config=SimpleNamespace(rms_norm_eps=1.0e-6),
            ),
        ),
        runtime=SimpleNamespace(device_synchronize=lambda: syncs.append("sync")),
    )
    decoder = Qwen35GGUFMTPDecodeSession.__new__(Qwen35GGUFMTPDecodeSession)
    decoder.target = target
    decoder.draft_hidden_policy = mtp_module.Qwen35GGUFDraftHiddenPolicy(
        target_hidden_variant="post_output_norm"
    )
    decoder._draft_post_norm_hidden = DeviceBuffer(0x3000, 8)
    monkeypatch.setattr(
        mtp_module,
        "gguf_rmsnorm_bf16_f32_weight",
        lambda src, weight, dst, **kwargs: calls.append(
            (
                int(src),
                int(weight),
                int(dst),
                int(kwargs["rows"]),
                int(kwargs["hidden_size"]),
            )
        ),
    )

    first = decoder._proposal_target_hidden()
    target.last_target_hidden = second_hidden
    second = decoder._proposal_target_hidden()

    assert first.ptr == second.ptr == 0x3000
    assert target.last_target_hidden is second_hidden
    assert calls == [
        (0x1000, 0x2000, 0x3000, 1, 4),
        (0x1100, 0x2000, 0x3000, 1, 4),
    ]
    assert syncs == ["sync", "sync"]
    assert decoder.draft_hidden_manifest["target_commit_hidden"] == "pre_output_norm"

    decoder.draft_hidden_policy = mtp_module.Qwen35GGUFDraftHiddenPolicy()
    strict = decoder._proposal_target_hidden()

    assert strict.ptr == 0x1100
    assert target.last_target_hidden is second_hidden
    assert calls == [
        (0x1000, 0x2000, 0x3000, 1, 4),
        (0x1100, 0x2000, 0x3000, 1, 4),
    ]
    assert decoder.draft_hidden_manifest["target_commit_hidden"] == "pre_output_norm"


def test_post_output_norm_prefill_repeat_keeps_owned_row_and_frees_temporaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    oracle = _PrimingOracle(monkeypatch)
    decoder = _new_priming_decoder(oracle, variant="post_output_norm")
    decoder._draft_post_norm_hidden = DeviceBuffer(0x9000, 8)
    decoder.owns_verifier = False
    decoder.verifier = SimpleNamespace(close=lambda: None)

    assert decoder._prefill_target_and_draft(
        tuple(range(11, 19)),
        request_id=7,
        use_bulk=True,
    ).token_id == 91
    assert decoder._prefill_target_and_draft(
        tuple(range(21, 24)),
        request_id=8,
        use_bulk=None,
    ).token_id == 91

    assert decoder._draft_post_norm_hidden.ptr == 0x9000
    assert 0x9000 not in oracle.freed
    assert len(oracle.allocations) == 4
    assert sorted(oracle.freed) == sorted(ptr for ptr, _ in oracle.allocations)

    decoder.close()
    assert oracle.freed[-1] == 0x9000


def test_post_output_norm_constructor_allocates_and_releases_owned_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    allocations: list[tuple[int, object]] = []
    freed: list[int] = []

    def _malloc(nbytes: int, *, runtime) -> DeviceBuffer:
        allocations.append((int(nbytes), runtime))
        return DeviceBuffer(0x3000, int(nbytes))

    monkeypatch.setattr(mtp_module, "malloc", _malloc)
    monkeypatch.setattr(
        mtp_module,
        "free",
        lambda buffer, *, runtime: freed.append(int(buffer.ptr)),
    )
    target = SimpleNamespace(
        runner=SimpleNamespace(hidden_size=4),
        runtime=object(),
    )
    verifier = SimpleNamespace(close=lambda: None)

    decoder = Qwen35GGUFMTPDecodeSession(
        target,
        object(),
        verifier=verifier,
        owns_verifier=False,
        draft_hidden_variant="post_output_norm",
    )

    assert allocations == [(8, target.runtime)]
    assert decoder._draft_post_norm_hidden is not None
    assert decoder._draft_post_norm_hidden.ptr == 0x3000
    assert decoder.draft_hidden_manifest["prompt_target_hidden"] == "post_output_norm"
    assert decoder.draft_hidden_manifest["target_commit_hidden"] == "pre_output_norm"

    decoder.close()
    assert freed == [0x3000]

    strict = Qwen35GGUFMTPDecodeSession(
        target,
        object(),
        verifier=verifier,
        owns_verifier=False,
    )

    assert strict._draft_post_norm_hidden is None
    assert strict.draft_hidden_manifest["prompt_target_hidden"] == "pre_output_norm"
    strict.close()
    assert freed == [0x3000]
