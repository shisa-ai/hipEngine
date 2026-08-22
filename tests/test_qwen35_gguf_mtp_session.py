from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from hipengine.core.device import Device
from hipengine.core.dtype import DType
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
        (1, 1021, 2, True, None),
        (1, 1022, 2, False, "target_graph_context_bucket_miss"),
        (2, 1020, 3, True, None),
        (2, 1021, 3, False, "target_graph_context_bucket_miss"),
        (3, 1019, 4, True, None),
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
    verifier.target_verify_mode = "native"
    verifier.target = target
    verifier.last_device_proposal_fallback_reason = "stale"

    assert verifier.device_proposal_ready(
        budget,
        remaining_decode=remaining_decode,
    ) is expected
    assert verifier.last_device_proposal_fallback_reason == expected_reason


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


def test_b4_native_request_falls_back_to_serial_exact_target_rows() -> None:
    assert mtp_module._effective_target_verify_mode("native", rows=4) == "native"
    assert mtp_module._effective_target_verify_mode("native", rows=5) == "serial_exact"
    assert mtp_module._effective_target_verify_mode("serial_exact", rows=5) == "serial_exact"
    assert mtp_module._initial_state_only_journal_applies(
        "native",
        max_candidate_budget=3,
    )
    assert not mtp_module._initial_state_only_journal_applies(
        "native",
        max_candidate_budget=4,
    )
    assert not mtp_module._initial_state_only_journal_applies(
        "serial_exact",
        max_candidate_budget=3,
    )


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


def test_mtp_prompt_admission_bulk_prefills_target_then_catches_up_shifted_draft(
    monkeypatch,
) -> None:
    allocations = iter((DeviceBuffer(0x1000, 8), DeviceBuffer(0x2000, 24)))
    freed: list[int] = []
    monkeypatch.setattr(mtp_module, "malloc", lambda _nbytes, *, runtime: next(allocations))
    monkeypatch.setattr(
        mtp_module,
        "free",
        lambda buffer, *, runtime: freed.append(int(buffer.ptr)),
    )

    target_calls: list[tuple[tuple[int, ...], dict[str, object]]] = []

    class Target:
        runner = SimpleNamespace(hidden_size=4)
        runtime = SimpleNamespace(memset=lambda *_args: None)

        def prefill(self, prompt, **kwargs):
            target_calls.append((tuple(prompt), kwargs))
            return SimpleNamespace(token_id=91)

        def step(self, *_args, **_kwargs):
            raise AssertionError("bulk MTP admission must not serial-prefill the target")

    draft_calls: list[tuple[int, int, int, int]] = []

    class Executor:
        def run_step(self, request_id, token_id, position, target_hidden, **_kwargs):
            draft_calls.append(
                (
                    int(request_id),
                    int(token_id),
                    int(position),
                    int(target_hidden.ptr),
                )
            )

    decoder = Qwen35GGUFMTPDecodeSession.__new__(Qwen35GGUFMTPDecodeSession)
    decoder.target = Target()
    decoder.draft_provider = SimpleNamespace(executor=Executor())

    result = decoder._prefill_target_and_draft(
        (11, 22, 33),
        request_id=7,
        use_bulk=True,
    )

    assert result.token_id == 91
    assert target_calls == [
        (
            (11, 22, 33),
            {
                "use_bulk": True,
                "return_logits": False,
                "capture_target_hidden_rows": DeviceBuffer(0x2000, 24),
            },
        )
    ]
    assert draft_calls == [
        (7, 11, 0, 0x1000),
        (7, 22, 1, 0x2000),
        (7, 33, 2, 0x2008),
    ]
    assert freed == [0x2000, 0x1000]


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

    def fake_prefill(_prompt, *, request_id, use_bulk):
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
        )

    assert proposed == []
    assert device_proposals == []
    assert prepared == []
