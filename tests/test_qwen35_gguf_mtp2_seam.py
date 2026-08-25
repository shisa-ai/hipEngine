from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.tensor import Tensor
from hipengine.generation.qwen35_gguf import Qwen35GGUFResidentModelRunner
import hipengine.generation.qwen35_gguf_mtp2 as mtp2_module
from hipengine.generation.qwen35_gguf_mtp2 import (
    Qwen35GGUFMTP2Adapter,
    _MTP2RequestState,
)
from hipengine.kernels.backends import backend_package_capability
from hipengine.runtime.qwen35_gguf_nextn import (
    Qwen35GGUFNextNBatchDeviceProposal,
)
from hipengine.speculative import MtpProposalContext, SpeculativeRequestSemantics


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


def test_gfx1151_package_exposes_c1_and_physical_c4_adapter_scopes() -> None:
    assert backend_package_capability(
        "hip_gfx1151", "GGUF_SPECDEC2_MTP2_C1", False
    ) is True
    assert backend_package_capability(
        "hip_gfx1151", "GGUF_SPECDEC2_MTP2_C4", False
    ) is True
    assert backend_package_capability(
        "hip_gfx1100", "GGUF_SPECDEC2_MTP2_C1", False
    ) is False
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
    assert capability.max_context_tokens == 4096

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
    plan = SimpleNamespace(request_ids=(7,))

    adapter.prepare_k0(plan, (), stream=None)

    assert calls == [(7, 90, 15, "pre-root-hidden")]
    assert row.mtp2_k0_catchups == 1


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

    adapter.prepare_k0(SimpleNamespace(request_ids=(7,)), (), stream=None)

    assert calls == []
    assert row.mtp2_k0_catchups == 0
