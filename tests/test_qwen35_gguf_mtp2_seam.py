from __future__ import annotations

import ctypes
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.memory import DeviceBuffer
from hipengine.core.tensor import Tensor
from hipengine.generation.qwen35_gguf import Qwen35GGUFResidentModelRunner
import hipengine.generation.qwen35_gguf_mtp2 as mtp2_module
from hipengine.generation.qwen35_gguf_mtp2 import (
    Qwen35GGUFMTP2Adapter,
    _target_verify_mode_for_context,
    _MTP2RequestState,
)
from hipengine.kernels.backends import backend_package_capability
from hipengine.runtime import qwen35_gguf_runner as runner_mod
from hipengine.runtime.qwen35_gguf_nextn import (
    Qwen35GGUFNextNBatchDeviceProposal,
)
from hipengine.speculative import (
    DraftBatch,
    MtpProposalContext,
    SpeculativeRequestSemantics,
    TargetVerifyBatch,
    TargetVerifyBuffers,
)


class _AdapterDouble:
    def __init__(self) -> None:
        self.calls = []

    def register_request(self, request_id, candidate_budget):
        self.calls.append(("register", request_id, candidate_budget))

    def capability(self, semantics):
        self.calls.append(("capability", tuple(semantics)))
        return "capability"

    def claims_fit(self, plan):
        return plan == "plan"

    def component_claims(self, plan):
        return {"plan": plan}

    def reserve_claims(self, claims):
        return ("reservation", claims)

    def release_claims(self, reservation):
        self.calls.append(("release", reservation))

    def prepare_requests(self, plan, semantics, *, stream=None):
        self.calls.append(("prepare", plan, tuple(semantics), stream))

    def propose_batch(self, plan, semantics, *, stream=None):
        return ("proposal", plan, tuple(semantics), stream)

    def execute_target_frontier(self, *args, **kwargs):
        return (args, kwargs)

    def rollback_cycle(self, *args):
        self.calls.append(("rollback", args))


def test_gfx1100_target_mode_resolves_before_verifier_construction() -> None:
    assert _target_verify_mode_for_context(
        "native", backend="hip_gfx1100", end_position=127
    ) == "native"
    assert _target_verify_mode_for_context(
        "native", backend="hip_gfx1100", end_position=128
    ) == "serial_exact"
    assert _target_verify_mode_for_context(
        "native", backend="hip_gfx1151", end_position=128
    ) == "native"


def test_backend_packages_expose_independently_qualified_adapter_scopes() -> None:
    assert backend_package_capability(
        "hip_gfx1151", "GGUF_SPECDEC2_MTP2_C1", False
    ) is True
    assert backend_package_capability(
        "hip_gfx1151", "GGUF_SPECDEC2_MTP2_C4", False
    ) is True
    assert backend_package_capability(
        "hip_gfx1100", "GGUF_SPECDEC2_MTP2_C1", False
    ) is True
    assert backend_package_capability(
        "hip_gfx1100", "GGUF_SPECDEC2_MTP2_C4", False
    ) is False


def test_resident_runner_delegates_staged_methods_without_backend_branches() -> None:
    runner = object.__new__(Qwen35GGUFResidentModelRunner)
    adapter = _AdapterDouble()
    runner._mtp2_adapter = adapter
    runner._mtp2_adapter_resolved = True
    runner.generator = SimpleNamespace(target_arch="gfx1151")
    runner._rows = {7: SimpleNamespace(mtp2_candidate_budget=0)}

    runner.register_speculative_request(7, 3)
    assert runner._rows[7].mtp2_candidate_budget == 3
    assert adapter.calls == [("register", 7, 3)]
    assert runner.speculative_capability(("semantics",)) == "capability"
    assert runner.speculative_claims_fit("plan") is True
    assert runner.speculative_component_claims("plan") == {"plan": "plan"}
    reservation = runner.reserve_speculative_claims("claims")
    assert reservation == ("reservation", "claims")
    runner.release_speculative_claims(reservation)
    runner.prepare_speculative_requests("plan", ("s",), stream=9)
    assert runner.propose_speculative_batch("plan", ("s",), stream=4) == (
        "proposal",
        "plan",
        ("s",),
        4,
    )
    assert runner.speculative_kv_live_spans_owner(SimpleNamespace(operation_id="op"))


def test_physical_specdec2_uses_qualified_eager_when_graph_is_uncached() -> None:
    runner = object.__new__(Qwen35GGUFResidentModelRunner)
    runner._mtp2_adapter = SimpleNamespace()
    runner._mtp2_adapter_resolved = True

    assert runner.speculative_graph_available(object()) is False


def test_real_adapter_requires_ar_root_and_exact_prefill_hidden_rows() -> None:
    target = SimpleNamespace(
        target_layout=SimpleNamespace(max_sequence_length=4096),
        kv_storage_dtype="bf16",
    )
    row = SimpleNamespace(
        native_greedy=True,
        first_token_emitted=False,
        lease=SimpleNamespace(session=target),
        slot=SimpleNamespace(generated_ids=[99]),
        prompt_ids=(10, 11, 12),
    )
    owner = SimpleNamespace(
        generator=SimpleNamespace(
            backend="hip_gfx1151",
            execution_profile="strict",
        ),
        _shared_runner=SimpleNamespace(hidden_size=4),
        _row=lambda request_id: row,
    )
    adapter = Qwen35GGUFMTP2Adapter(
        owner,
        enabled=True,
        target_verify_mode="native",
        candidate_budget=3,
    )
    semantics = (
        SpeculativeRequestSemantics(
            request_id=7,
            sampling_mode="greedy",
            mode="verify_chain",
            context_tokens=4,
            remaining_decode=8,
        ),
    )
    adapter.register_request(7, 3)
    adapter.observe_prefill_result(
        7,
        row.prompt_ids,
        SimpleNamespace(hidden_seeds=np.zeros((3, 4), dtype=np.float32)),
    )

    assert adapter.capability(semantics) is None
    row.first_token_emitted = True
    capability = adapter.capability(semantics)
    assert capability is not None
    assert capability.max_requests == 4
    assert capability.max_candidates_per_request == 3
    assert capability.max_frontier_rows == 16
    assert capability.max_context_tokens == 1023

    target.runner = SimpleNamespace(fp16_recurrent_state=True)
    assert adapter.capability(semantics) is None


def test_physical_adapter_returns_device_candidate_graph_before_target(
    monkeypatch,
) -> None:
    runtime = SimpleNamespace(memcpy=lambda *args: None)
    targets = (
        SimpleNamespace(
            position=5,
            last_target_hidden=Tensor.from_handle(
                0x1100, (1, 8), DType.BF16, Device("hip", 0)
            ),
            runtime=runtime,
        ),
        SimpleNamespace(
            position=8,
            last_target_hidden=Tensor.from_handle(
                0x1200, (1, 8), DType.BF16, Device("hip", 0)
            ),
            runtime=runtime,
        ),
    )
    device_draft = Qwen35GGUFNextNBatchDeviceProposal(
        request_ids=(10, 20),
        root_tokens=(100, 200),
        root_positions=(5, 8),
        candidate_counts=(1, 2),
        token_ids=Tensor.from_handle(
            0x5000, (3,), DType.INT32, Device("hip", 0)
        ),
        hidden_rows=(
            (Tensor.from_handle(0x6000, (1, 8), DType.BF16, Device("hip", 0)),),
            (
                Tensor.from_handle(0x7000, (1, 8), DType.BF16, Device("hip", 0)),
                Tensor.from_handle(0x8000, (1, 8), DType.BF16, Device("hip", 0)),
            ),
        ),
    )
    calls = []
    executor = SimpleNamespace(
        hidden_size=8,
        capture_request_checkpoint=lambda request_id: f"checkpoint-{request_id}",
    )
    provider = SimpleNamespace(
        executor=executor,
        propose_batch_device=lambda context, candidate_counts: (
            calls.append(("propose", tuple(candidate_counts))) or device_draft
        ),
    )
    rows = tuple(
        SimpleNamespace(
            lease=SimpleNamespace(session=target),
            slot=SimpleNamespace(
                generated_ids=[token],
                seq_position=int(target.position),
            ),
            mtp2_proposal_batch_calls=0,
            mtp2_proposal_physical_rows=[],
            mtp2_candidate_device_handoffs=0,
        )
        for target, token in zip(targets, (100, 200), strict=True)
    )
    owner = SimpleNamespace(
        _row=lambda request_id: rows[(10, 20).index(request_id)],
        _flush_row_owner=lambda row: None,
    )
    adapter = object.__new__(Qwen35GGUFMTP2Adapter)
    adapter.owner = owner
    adapter._states = {
        request_id: _MTP2RequestState(
            request_id=request_id,
            provider=provider,
            provider_pool_key=None,
            provider_group_key=(10, 20),
            verifier=None,
            root_hidden_buffer=SimpleNamespace(ptr=1),
        )
        for request_id in (10, 20)
    }
    monkeypatch.setattr(
        mtp2_module,
        "malloc",
        lambda nbytes, runtime: SimpleNamespace(ptr=0x9000, nbytes=nbytes),
    )
    monkeypatch.setattr(mtp2_module, "free", lambda buffer, runtime: None)
    plan = SimpleNamespace(
        speculative_request_ids=(10, 20),
        request_ids=(10, 20),
        candidate_counts=(1, 2),
        provider_key="nextn",
        cycle_id=3,
        resident_slots=(0, 1),
    )
    semantics = (
        SpeculativeRequestSemantics(10, "greedy", "verify_chain", 6, 8),
        SpeculativeRequestSemantics(20, "greedy", "verify_chain", 9, 8),
    )

    graph = adapter.propose_batch(plan, semantics)

    assert graph.candidate_tokens == ()
    assert graph.token_ids is device_draft.token_ids
    assert graph.candidate_counts == (1, 2)
    assert graph.provider_metadata[0] == ("candidate_handoff", "device_i32")
    assert calls == [("propose", (1, 2))]
    assert all(row.mtp2_candidate_device_handoffs == 1 for row in rows)
    assert all(
        state.proposal_device_batch is device_draft
        for state in adapter._states.values()
    )


def test_physical_adapter_emits_one_gpu_accept_payload_for_the_group(
    monkeypatch,
) -> None:
    draft = DraftBatch(
        request_ids=(10, 20),
        candidate_tokens=(101, 201),
        parent_positions=(5, 8),
        draft_depths=(1, 1),
        row_to_request=(10, 20),
        tree_parents=(-1, -1),
        active_mask=(True, True),
    )
    batch = TargetVerifyBatch.from_draft(
        draft,
        root_tokens=(100, 200),
        root_positions=(5, 8),
    )
    target_top1 = (101, 999, 303, 404)
    remaining = (3, 3)
    expected_accept = batch.accept_from_top1(
        target_top1,
        transaction_id=7,
        remaining_decode=remaining,
    )
    expected = mtp2_module.TargetAcceptSummary.from_accept_result(
        batch,
        expected_accept,
    )
    payload_host = np.asarray(
        [
            [
                expected.accepted_counts[index],
                expected.commit_rows[index],
                expected.commit_tokens[index],
                expected.commit_positions[index],
                -1 if expected.next_tokens[index] is None else expected.next_tokens[index],
                int(expected.full_accept[index]),
                expected.accepted_counts[index] + 1,
            ]
            for index in range(2)
        ],
        dtype=np.int32,
    )
    pointer = iter(range(0x1000, 0x3000, 0x100))

    def tensor(shape, dtype=DType.INT32):
        return Tensor.from_handle(next(pointer), shape, dtype, Device("hip", 0))

    buffers = TargetVerifyBuffers.for_batch(
        batch,
        token_ids=tensor((4,)),
        positions=tensor((4,)),
        parent_rows=tensor((4,)),
        draft_depths=tensor((4,)),
        row_to_request=tensor((4,)),
        active_mask=tensor((4,), DType.BOOL),
        target_top1=tensor((4,)),
        accepted_counts=tensor((2,)),
        commit_rows=tensor((2,)),
        commit_tokens=tensor((2,)),
        commit_positions=tensor((2,)),
        next_tokens=tensor((2,)),
        full_accept=tensor((2,), DType.BOOL),
        committed_output_ids=tensor((2, 4)),
        committed_output_lengths=tensor((2,)),
        transaction_id=7,
    )
    owner = SimpleNamespace(bind=lambda bound, transaction_id: buffers)
    remaining_tensor = tensor((4,))
    payload_tensor = tensor((4, 7))
    adapter = object.__new__(Qwen35GGUFMTP2Adapter)
    adapter.generator = SimpleNamespace(
        backend="hip_gfx1151",
        compiler_version=None,
        require_cached_build=True,
    )
    adapter._batch_accept_library = object()
    adapter._batch_accept_resources = lambda runtime: (
        owner,
        remaining_tensor,
        payload_tensor,
    )
    adapter._upload_accept_array = lambda tensor, values, runtime: None
    calls = []
    monkeypatch.setattr(
        mtp2_module,
        "dflash_accept_chain_i32_packed",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    monkeypatch.setattr(
        mtp2_module,
        "copy_device_to_host",
        lambda destination, source, nbytes, *, runtime: ctypes.memmove(
            destination,
            payload_host.ctypes.data,
            nbytes,
        ),
    )
    runtime = SimpleNamespace(device_synchronize=lambda: calls.append(("sync",)))

    summary, actual_buffers = adapter._accept_target_batch_on_device(
        batch,
        target_top1,
        remaining,
        transaction_id=7,
        runtime=runtime,
    )

    assert actual_buffers is buffers
    assert summary.accepted_counts == (1, 0)
    assert summary.accepted_tokens == ((101,), ())
    assert summary.commit_rows == expected.commit_rows
    assert summary.next_tokens == expected.next_tokens
    assert len([call for call in calls if call != ("sync",)]) == 1


def test_packed_owner_commits_selected_linear_rows_once_for_the_group(
    monkeypatch,
) -> None:
    class Buffer:
        def __init__(self, ptr, nbytes):
            self.ptr = int(ptr)
            self.nbytes = int(nbytes)

    class PackedState:
        pass

    monkeypatch.setattr(runner_mod, "_GGUFPackedTargetState", PackedState)
    monkeypatch.setattr(runner_mod, "set_decode_position_i64", lambda *args, **kwargs: None)
    copies = []
    runtime = SimpleNamespace(memcpy_async=lambda *args: None)
    owner = object.__new__(runner_mod.Qwen35GGUFResidentSession)
    weights = SimpleNamespace(
        config=SimpleNamespace(layer_types=(runner_mod.LINEAR_ATTENTION,))
    )
    owner.runner = SimpleNamespace(weights=weights, hidden_size=2)
    owner.runtime = runtime
    owner._verify_hidden_seed_buf = Buffer(0x1000, 6 * 8)
    owner._packed_verify_max_written_positions = (0, 0)
    owner._verify_linear_state_row_pair = lambda layer_id: (
        Buffer(0x2000, 6 * 16),
        Buffer(0x3000, 6 * 32),
    )
    owner._fused_linear_state_pair_copy = lambda entries, **kwargs: (
        copies.extend(entries) or True
    )
    sessions = []
    for index in range(2):
        session = SimpleNamespace(
            runner=owner.runner,
            scratch=SimpleNamespace(
                hidden_seed_fp32=Buffer(0x4000 + index * 0x100, 8),
                layer_conv_states=(Buffer(0x5000 + index * 0x100, 16),),
                layer_recurrent_states=(Buffer(0x6000 + index * 0x100, 32),),
                position_host=np.asarray([0], dtype=np.int64),
                context_host=np.asarray([1], dtype=np.int64),
                position_buf=Buffer(0x7000 + index * 0x100, 8),
                context_buf=Buffer(0x8000 + index * 0x100, 8),
            ),
            _runtime_state_library=object(),
            _verify_hidden_seed_buf=None,
            _ensure_verify_block_buffers=lambda rows, runtime, session_index=index: None,
            _verify_hidden_seed_rows_populated=0,
            _hidden_seed_fp32_populated=False,
            _position=0,
        )
        session._verify_hidden_seed_buf = Buffer(0x9000 + index * 0x100, 3 * 8)
        sessions.append(session)
    packed = PackedState()
    results = (
        SimpleNamespace(
            token_ids=[1, 2, 3],
            deferred_packed_state=SimpleNamespace(
                owner=owner,
                packed_state=packed,
                row_start=0,
                row_end=3,
                slot_index=0,
                start_position=5,
            ),
        ),
        SimpleNamespace(
            token_ids=[4, 5, 6],
            deferred_packed_state=SimpleNamespace(
                owner=owner,
                packed_state=packed,
                row_start=3,
                row_end=6,
                slot_index=1,
                start_position=8,
            ),
        ),
    )
    accept_buffers = SimpleNamespace(
        accepted_counts=Tensor.from_handle(
            0xA000, (2,), DType.INT32, Device("hip", 0)
        )
    )

    contract = owner._commit_deferred_packed_verify_states_batch(
        results,
        sessions,
        accepted_counts=(2, 0),
        accept_buffers=accept_buffers,
    )

    assert contract["requests"] == 2
    assert contract["fused_linear_state_commit"] is True
    assert len(copies) == 2
    assert copies[0][0] == 0x2000 + 2 * 16
    assert copies[0][2] == 0x3000 + 2 * 32
    assert copies[1][0] == 0x2000 + 3 * 16
    assert copies[1][2] == 0x3000 + 3 * 32
    assert [session._position for session in sessions] == [8, 9]


def test_adapter_recovers_only_precommit_failure_with_canonical_target_cursors() -> None:
    rows = {
        10: SimpleNamespace(
            slot=SimpleNamespace(seq_position=7),
            lease=SimpleNamespace(session=SimpleNamespace(position=7)),
            mtp2_recoverable_failures=0,
            mtp2_failure_reasons=[],
        ),
        20: SimpleNamespace(
            slot=SimpleNamespace(seq_position=9),
            lease=SimpleNamespace(session=SimpleNamespace(position=9)),
            mtp2_recoverable_failures=0,
            mtp2_failure_reasons=[],
        ),
    }
    adapter = object.__new__(Qwen35GGUFMTP2Adapter)
    adapter.owner = SimpleNamespace(_row=lambda request_id: rows[request_id])
    plan = SimpleNamespace(speculative_request_ids=(10, 20))

    assert adapter.recover_cycle_failure(plan, RuntimeError("injected")) is True
    assert all(row.mtp2_recoverable_failures == 1 for row in rows.values())
    assert all(
        row.mtp2_failure_reasons == ["precommit_failure_ar_fallback"]
        for row in rows.values()
    )

    rows[20].lease.session.position = 10
    assert adapter.recover_cycle_failure(plan, RuntimeError("late")) is False


@pytest.mark.parametrize(
    ("accepted", "expected_inputs", "full_tail"),
    [
        (0, [(90, 10)], False),
        (1, [(90, 10), (101, 11)], False),
        (2, [(90, 10), (101, 11), (102, 12)], False),
        (3, [], True),
    ],
)
def test_provider_repair_restores_and_replays_only_committed_prefix(
    accepted,
    expected_inputs,
    full_tail,
) -> None:
    calls = []

    class Executor:
        def restore_request_checkpoint(self, checkpoint):
            calls.append(("restore", checkpoint))

        def advance_state_only(self, request_id, token_id, position, hidden):
            calls.append(("advance", request_id, token_id, position, hidden))

    results = tuple(
        SimpleNamespace(token_id=101 + index, hidden=f"h{index}")
        for index in range(3)
    )
    provider = SimpleNamespace(
        executor=Executor(),
        last_results={7: results},
        advance_full_accept_tail=lambda request_id, accepted_count: calls.append(
            ("full", request_id, accepted_count)
        ),
    )
    state = _MTP2RequestState(
        request_id=7,
        provider=provider,
        provider_pool_key=None,
        provider_group_key=(7,),
        verifier=SimpleNamespace(),
        root_hidden_buffer=SimpleNamespace(ptr=1),
        proposal_checkpoint="checkpoint",
        proposal_context=MtpProposalContext(
            request_ids=(7,),
            root_tokens=(90,),
            root_positions=(10,),
            target_hidden=SimpleNamespace(ndim=2, shape=(1, 4)),
        ),
    )
    adapter = object.__new__(Qwen35GGUFMTP2Adapter)

    adapter._repair_provider_state(
        state,
        accepted_count=accepted,
        candidate_count=3,
    )

    if full_tail:
        assert calls == [("full", 7, 3)]
    else:
        assert calls[0] == ("restore", "checkpoint")
        actual_inputs = [(call[2], call[3]) for call in calls[1:]]
        assert actual_inputs == expected_inputs


def test_provider_batch_repair_shares_full_accept_tail_and_rejected_root(
    monkeypatch,
) -> None:
    calls = []

    class Executor:
        hidden_size = 8
        runtime = SimpleNamespace(memcpy=lambda *args: None)

        def restore_request_checkpoint(self, checkpoint):
            calls.append(("restore", checkpoint))

        def advance_state_batch_only(
            self,
            request_ids,
            token_ids,
            positions,
            target_hidden,
        ):
            calls.append(
                (
                    "batch",
                    tuple(request_ids),
                    tuple(token_ids),
                    tuple(positions),
                    target_hidden.shape,
                )
            )

    executor = Executor()
    results = {
        1: (
            SimpleNamespace(token_id=101, position=5, hidden=SimpleNamespace(ptr=1101)),
            SimpleNamespace(token_id=102, position=6, hidden=SimpleNamespace(ptr=1102)),
        ),
        2: (
            SimpleNamespace(token_id=201, position=8, hidden=SimpleNamespace(ptr=1201)),
            SimpleNamespace(token_id=202, position=9, hidden=SimpleNamespace(ptr=1202)),
        ),
    }
    provider = SimpleNamespace(executor=executor, last_results=results)
    states = tuple(
        _MTP2RequestState(
            request_id=request_id,
            provider=provider,
            provider_pool_key=None,
            provider_group_key=(1, 2),
            verifier=None,
            root_hidden_buffer=SimpleNamespace(ptr=1),
            proposal_checkpoint=f"checkpoint-{request_id}",
            proposal_context=MtpProposalContext(
                request_ids=(request_id,),
                root_tokens=((90,) if request_id == 1 else (190,)),
                root_positions=((5,) if request_id == 1 else (8,)),
                target_hidden=SimpleNamespace(
                    ptr=2000 + request_id,
                    ndim=2,
                    shape=(1, 8),
                ),
            ),
        )
        for request_id in (1, 2)
    )
    monkeypatch.setattr(
        mtp2_module,
        "malloc",
        lambda nbytes, runtime: SimpleNamespace(ptr=3000, nbytes=nbytes),
    )
    monkeypatch.setattr(mtp2_module, "free", lambda buffer, runtime: None)
    adapter = object.__new__(Qwen35GGUFMTP2Adapter)

    adapter._repair_provider_states_batch(
        states,
        accepted_counts=(2, 0),
        candidate_counts=(2, 2),
    )

    assert calls == [
        ("restore", "checkpoint-2"),
        ("batch", (1, 2), (102, 190), (7, 8), (2, 8)),
    ]


def test_k0_catchup_consumes_current_root_before_target_ar() -> None:
    calls = []
    executor = SimpleNamespace(
        advance_state_only=lambda request_id, token, position, hidden: calls.append(
            (request_id, token, position, hidden)
        )
    )
    provider = SimpleNamespace(executor=executor)
    state = _MTP2RequestState(
        request_id=7,
        provider=provider,
        provider_pool_key=None,
        provider_group_key=(7,),
        verifier=SimpleNamespace(),
        root_hidden_buffer=SimpleNamespace(ptr=1),
    )
    row = SimpleNamespace(
        first_token_emitted=True,
        lease=SimpleNamespace(
            session=SimpleNamespace(position=15, last_target_hidden="pre-root-hidden")
        ),
        slot=SimpleNamespace(generated_ids=[90]),
        mtp2_k0_catchups=0,
    )
    adapter = object.__new__(Qwen35GGUFMTP2Adapter)
    adapter._states = {7: state}
    adapter._intents = {7: 3}
    adapter._prompt_hidden_rows = {7: np.zeros((1, 4), dtype=np.float32)}
    adapter._disabled_requests = set()
    adapter.owner = SimpleNamespace(
        _row=lambda request_id: row,
        _flush_row_owner=lambda owned_row: None,
    )
    plan = SimpleNamespace(
        request_ids=(7,),
        reasons=(mtp2_module.SpecPlanReason.NO_PROVIDER,),
    )

    adapter.prepare_k0(plan, (), stream=None)

    assert calls == [(7, 90, 15, "pre-root-hidden")]
    assert row.mtp2_k0_catchups == 1


def test_context_bucket_k0_does_not_attach_or_mutate_provider() -> None:
    adapter = object.__new__(Qwen35GGUFMTP2Adapter)
    adapter._states = {}
    adapter._intents = {7: 3}
    adapter._prompt_hidden_rows = {7: np.zeros((1, 4), dtype=np.float32)}
    adapter._disabled_requests = set()
    calls = []
    adapter.owner = SimpleNamespace(
        _row=lambda request_id: SimpleNamespace(
            first_token_emitted=True,
            lease=SimpleNamespace(
                session=SimpleNamespace(position=1023, last_target_hidden="hidden")
            ),
            slot=SimpleNamespace(generated_ids=[90]),
            mtp2_k0_catchups=0,
        ),
        _flush_row_owner=lambda row: calls.append("flush"),
    )
    adapter._ensure_request_states = lambda ids: calls.append(("attach", ids))

    adapter.prepare_k0(
        SimpleNamespace(
            request_ids=(7,),
            reasons=(mtp2_module.SpecPlanReason.TARGET_GRAPH_CONTEXT_BUCKET_MISS,),
        ),
        (),
        stream=None,
    )

    assert calls == []
    assert adapter._states == {}


def test_k0_does_not_advance_provider_before_prefill_root_is_published() -> None:
    calls = []
    state = _MTP2RequestState(
        request_id=7,
        provider=SimpleNamespace(
            executor=SimpleNamespace(
                advance_state_only=lambda *args: calls.append(args)
            )
        ),
        provider_pool_key=None,
        provider_group_key=(7,),
        verifier=None,
        root_hidden_buffer=SimpleNamespace(ptr=1),
    )
    row = SimpleNamespace(
        first_token_emitted=False,
        lease=SimpleNamespace(
            session=SimpleNamespace(position=15, last_target_hidden="prefill-hidden")
        ),
        slot=SimpleNamespace(generated_ids=[90]),
        mtp2_k0_catchups=0,
    )
    adapter = object.__new__(Qwen35GGUFMTP2Adapter)
    adapter._states = {7: state}
    adapter._intents = {7: 3}
    adapter._prompt_hidden_rows = {7: np.zeros((1, 4), dtype=np.float32)}
    adapter._disabled_requests = set()
    adapter.owner = SimpleNamespace(
        _row=lambda request_id: row,
        _flush_row_owner=lambda owned_row: None,
    )

    adapter.prepare_k0(
        SimpleNamespace(
            request_ids=(7,),
            reasons=(mtp2_module.SpecPlanReason.NO_PROVIDER,),
        ),
        (),
        stream=None,
    )

    assert calls == []
    assert row.mtp2_k0_catchups == 0


def test_packed_prompt_hidden_sinks_preserve_ragged_request_offsets() -> None:
    calls: dict[int, list[tuple[str, int, int, int, int]]] = {7: [], 8: []}

    class Sink:
        hidden_size = 4

        def __init__(self, request_id: int, total_rows: int) -> None:
            self.request_id = request_id
            self.total_rows = total_rows

        def consume(self, **kwargs) -> None:
            calls[self.request_id].append(
                (
                    "consume",
                    int(kwargs["chunk_start"]),
                    int(kwargs["hidden_ptr"]),
                    int(kwargs["rows"]),
                    int(kwargs["stream"]),
                )
            )

        def finish(self, **kwargs) -> None:
            calls[self.request_id].append(
                (
                    "finish",
                    int(kwargs["total_rows"]),
                    0,
                    0,
                    int(kwargs["stream"]),
                )
            )

    runner_mod._consume_packed_target_hidden_sinks(
        sinks=(Sink(7, 9), Sink(8, 13)),
        request_ids=(7, 8),
        prompt_row_starts=(2, 5),
        packed_cu_seqlens=(0, 2, 5),
        hidden_base_ptr=0x1000,
        hidden_row_nbytes=8,
        stream=3,
        finish=False,
    )

    assert calls == {
        7: [("consume", 2, 0x1000, 2, 3)],
        8: [("consume", 5, 0x1010, 3, 3)],
    }


def test_mtp2_streaming_prompt_success_transfers_one_carried_row_per_request(
    monkeypatch,
) -> None:
    class Runtime:
        pass

    targets = {
        rid: SimpleNamespace(
            runtime=Runtime(),
            target_layout=SimpleNamespace(max_sequence_length=1024),
            _last_target_hidden_ptr=0,
        )
        for rid in (7, 8)
    }
    rows = {
        rid: SimpleNamespace(
            request_id=rid,
            prompt_ids=(11 + rid, 22 + rid),
            lease=SimpleNamespace(session=targets[rid]),
            prefix_reused_tokens=0,
            mtp2_prompt_streaming=False,
            mtp2_prompt_prime_rows=0,
            mtp2_prompt_carried_bytes=0,
            mtp2_prompt_fallback_reason=None,
        )
        for rid in (7, 8)
    }
    finish_calls: list[tuple[int, bool]] = []

    class Executor:
        hidden_size = 4
        max_requests = 4
        runtime = targets[7].runtime

        def enqueue_prompt_rows(self, *args, **kwargs) -> None:
            pass

        def finish_prompt_priming(self, request_id, *, stream, synchronize) -> None:
            finish_calls.append((int(request_id), bool(synchronize)))

    class Provider:
        executor = Executor()

        def __init__(self) -> None:
            self.reset: list[int] = []
            self.released: list[int] = []

        def reset_request(self, request_id) -> None:
            self.reset.append(int(request_id))

        def release_request(self, request_id) -> None:
            self.released.append(int(request_id))

    provider = Provider()
    released_pool: list[tuple[object, object]] = []
    generator = SimpleNamespace(
        backend="hip_gfx1151",
        execution_profile="strict",
        _acquire_dense_mtp_draft_provider=lambda *args, **kwargs: (
            provider,
            "pool",
            False,
        ),
        _release_mtp_draft_runner=lambda key, owned: released_pool.append(
            (key, owned)
        ),
    )
    owner = SimpleNamespace(
        generator=generator,
        capacity=4,
        _shared_runner=SimpleNamespace(hidden_size=4),
        _row=lambda request_id: rows[int(request_id)],
    )
    carried = {
        7: DeviceBuffer(0x7000, 8),
        8: DeviceBuffer(0x8000, 8),
    }

    class Sink:
        hidden_size = 4

        def __init__(self, *, request_id, prompt_tokens, **kwargs) -> None:
            self.request_id = int(request_id)
            self.total_rows = len(tuple(prompt_tokens))
            self.closed = False

        def take_final_pending_buffer(self):
            return carried[self.request_id]

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(
        mtp2_module,
        "_StreamingNextNPromptSink",
        Sink,
        raising=False,
    )
    adapter = Qwen35GGUFMTP2Adapter(
        owner,
        enabled=True,
        target_verify_mode="native",
        candidate_budget=2,
    )
    adapter.register_request(7, 2)
    adapter.register_request(8, 2)

    sinks = adapter.begin_prompt_streaming((7, 8), checkpoints={})
    adapter.finish_prompt_streaming((7, 8), success=True, stream=0)

    assert tuple(sink.request_id for sink in sinks) == (7, 8)
    assert provider.reset == [7, 8]
    assert finish_calls == [(7, False), (8, False)]
    assert adapter._prompt_hidden_rows == {}
    assert set(adapter._states) == {7, 8}
    assert adapter._states[7].root_hidden_buffer is carried[7]
    assert adapter._states[8].root_hidden_buffer is carried[8]
    assert adapter._states[7].provider_group_key == adapter._states[8].provider_group_key
    assert targets[7]._last_target_hidden_ptr == carried[7].ptr
    assert targets[8]._last_target_hidden_ptr == carried[8].ptr
    assert rows[7].mtp2_prompt_streaming and rows[8].mtp2_prompt_streaming
    assert rows[7].mtp2_prompt_prime_rows == 2
    assert rows[8].mtp2_prompt_carried_bytes == 8
    adapter.observe_prefill_result(7, rows[7].prompt_ids, SimpleNamespace(token_id=9))
    assert adapter._states[7].root_hidden_buffer is carried[7]
    assert released_pool == []


def test_mtp2_long_prompt_selects_k0_before_provider_streaming() -> None:
    target = SimpleNamespace(
        target_layout=SimpleNamespace(max_sequence_length=4096),
        runtime=object(),
    )
    row = SimpleNamespace(
        prompt_ids=tuple(range(1022)),
        lease=SimpleNamespace(session=target),
        prefix_reused_tokens=0,
        mtp2_candidate_budget=2,
        mtp2_prompt_fallback_reason=None,
    )
    acquired: list[str] = []
    owner = SimpleNamespace(
        generator=SimpleNamespace(
            _acquire_dense_mtp_draft_provider=lambda *args, **kwargs: acquired.append(
                "provider"
            )
        ),
        capacity=1,
        _shared_runner=SimpleNamespace(hidden_size=4),
        _row=lambda request_id: row,
    )
    adapter = Qwen35GGUFMTP2Adapter(
        owner,
        enabled=True,
        target_verify_mode="native",
        candidate_budget=2,
    )
    adapter.register_request(7, 2)

    assert adapter.begin_prompt_streaming((7,), checkpoints={}) is None
    assert row.mtp2_candidate_budget == 0
    assert row.mtp2_prompt_fallback_reason == "target_context_k0"
    assert acquired == []
    assert adapter._prompt_streaming_sinks == {}
    assert adapter._states == {}


def test_mtp2_streaming_prompt_failure_drains_provider_and_sink() -> None:
    events: list[tuple[object, ...]] = []

    class Sink:
        def close(self) -> None:
            events.append(("sink_close",))

    provider = SimpleNamespace(
        executor=SimpleNamespace(
            finish_prompt_priming=lambda request_id, *, stream, synchronize: events.append(
                ("finish", int(request_id), int(stream), bool(synchronize))
            )
        ),
        release_request=lambda request_id: events.append(
            ("release_request", int(request_id))
        ),
    )
    group = mtp2_module._MTP2ProviderGroup(
        key=(7,),
        provider=provider,
        provider_pool_key="pool",
        request_ids={7},
    )
    adapter = object.__new__(Qwen35GGUFMTP2Adapter)
    adapter._prompt_streaming_sinks = {7: Sink()}
    adapter._prompt_streaming_group_keys = {7: (7,)}
    adapter._provider_groups = {(7,): group}
    adapter.generator = SimpleNamespace(
        _release_mtp_draft_runner=lambda key, owned: events.append(
            ("release_group", key, owned)
        )
    )

    adapter.finish_prompt_streaming((7,), success=False, stream=5)

    assert events == [
        ("finish", 7, 5, True),
        ("release_request", 7),
        ("sink_close",),
        ("release_group", "pool", provider),
    ]
    assert adapter._prompt_streaming_sinks == {}
    assert adapter._prompt_streaming_group_keys == {}
    assert adapter._provider_groups == {}
