"""GGUF trailing-NextN executors and candidate-only DraftModel provider gates."""

from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.hip import HipMemcpyKind, get_hip_runtime
from hipengine.core.memory import DeviceBuffer, copy_device_to_host, free, host_array_ptr, malloc
from hipengine.core.tensor import Tensor
from hipengine.loading import load_gguf_index
from hipengine.loading.gguf import MissingGGUFTensorError
from hipengine.loading.qwen35_gguf_nextn import build_qwen35_gguf_nextn_tensor_map
from hipengine.runtime import qwen35_gguf_nextn as nextn_mod
from hipengine.runtime.qwen35_gguf_nextn import (
    Qwen35GGUFNextNBatchDeviceProposal,
    Qwen35GGUFNextNDeviceProposal,
    Qwen35GGUFNextNDraftProvider,
    Qwen35GGUFNextNExecutor,
    Qwen35GGUFNextNStateAdvance,
    Qwen35GGUFNextNStepResult,
    _private_slot_buffer,
)
from hipengine.speculative import MtpDraftProvider, MtpProposalContext

_MODEL = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q3_K_M.gguf")
_ORACLE = Path(__file__).parent / "fixtures" / "gguf" / "q3km_nextn_one_step_oracle.json"
_DENSE_MODEL = Path("/models/gguf/Qwen3.6-27B-Q4_K_M.gguf")
_DENSE_ORACLE = (
    Path(__file__).parent / "fixtures" / "gguf" / "qwen36_27b_q4km_nextn_one_step_oracle.json"
)


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def test_nextn_executor_backend_inherits_borrowed_target_backend() -> None:
    borrowed = {
        "token_embedding": SimpleNamespace(backend="hip_gfx1151"),
        "lm_head": SimpleNamespace(backend="hip_gfx1151"),
    }

    assert nextn_mod._resolve_nextn_executor_backend(None, borrowed) == "hip_gfx1151"
    assert (
        nextn_mod._resolve_nextn_executor_backend("auto", borrowed) == "hip_gfx1151"
    )
    assert (
        nextn_mod._resolve_nextn_executor_backend("hip_gfx1151", borrowed)
        == "hip_gfx1151"
    )
    with pytest.raises(ValueError, match="does not match borrowed fallback backend"):
        nextn_mod._resolve_nextn_executor_backend("hip_gfx1100", borrowed)
    with pytest.raises(ValueError, match="multiple backends"):
        nextn_mod._resolve_nextn_executor_backend(
            None,
            {
                "token_embedding": SimpleNamespace(backend="hip_gfx1151"),
                "lm_head": SimpleNamespace(backend="hip_gfx1100"),
            },
        )


class _FakeExecutor:
    hidden_size = 8

    def __init__(self) -> None:
        self.calls: list[tuple[int, int, int, int]] = []
        self.state_only_calls: list[tuple[int, int, int, int]] = []

    def run_step(
        self,
        request_id: int,
        token_id: int,
        position: int,
        target_hidden: Tensor,
        *,
        return_logits: bool = False,
    ) -> Qwen35GGUFNextNStepResult:
        self.calls.append((request_id, token_id, position, target_hidden.ptr))
        next_token = token_id + 1
        logits = np.asarray([[float(token_id), float(next_token)]], dtype=np.float32) if return_logits else None
        hidden = Tensor.from_handle(1000 + len(self.calls), (1, 8), DType.BF16, Device("hip", 0))
        return Qwen35GGUFNextNStepResult(
            request_id=request_id,
            input_token=token_id,
            position=position,
            token_id=next_token,
            logit=float(next_token),
            hidden=hidden,
            logits=logits,
        )

    def advance_state_only(
        self,
        request_id: int,
        token_id: int,
        position: int,
        target_hidden: Tensor,
    ) -> Qwen35GGUFNextNStateAdvance:
        self.state_only_calls.append((request_id, token_id, position, target_hidden.ptr))
        return Qwen35GGUFNextNStateAdvance(
            request_id=request_id,
            input_token=token_id,
            position=position,
        )

    def reset_request(self, request_id: int) -> None:
        del request_id

    def close(self) -> None:
        return None


def test_nextn_provider_emits_only_candidate_rows_under_locked_abi() -> None:
    executor = _FakeExecutor()
    provider = Qwen35GGUFNextNDraftProvider(executor)
    assert isinstance(provider, MtpDraftProvider)
    target_hidden = Tensor.from_handle(77, (2, 8), DType.BF16, Device("hip", 0))
    context = MtpProposalContext(
        request_ids=(41, 42),
        root_tokens=(9, 19),
        root_positions=(12, 30),
        target_hidden=target_hidden,
    )

    draft = provider.propose(context, candidate_budget=2)

    assert draft.request_ids == (41, 42)
    assert draft.candidate_tokens == (10, 11, 20, 21)
    assert draft.parent_positions == (12, 13, 30, 31)
    assert draft.draft_depths == (1, 2, 1, 2)
    assert draft.row_to_request == (41, 41, 42, 42)
    assert draft.mode == "verify_chain"
    assert executor.calls == [
        (41, 9, 12, 77),
        (41, 10, 13, 1001),
        (42, 19, 30, 93),
        (42, 20, 31, 1003),
    ]
    draft_b4 = provider.propose(context, candidate_budget=4)
    assert draft_b4.candidate_tokens == (10, 11, 12, 13, 20, 21, 22, 23)
    assert draft_b4.draft_depths == (1, 2, 3, 4, 1, 2, 3, 4)
    assert len(executor.calls) == 12
    with pytest.raises(ValueError, match="one of 1, 2, 3, 4, 5"):
        provider.propose(context, candidate_budget=6)


def test_nextn_physical_batch_publishes_consumed_positions_not_next_cursors(
    monkeypatch,
) -> None:
    slots = {
        10: SimpleNamespace(
            position_host=np.asarray([99], dtype=np.int64),
            context_host=np.asarray([100], dtype=np.int64),
            position_buf=SimpleNamespace(ptr=0x1000),
            context_buf=SimpleNamespace(ptr=0x2000),
        ),
        20: SimpleNamespace(
            position_host=np.asarray([99], dtype=np.int64),
            context_host=np.asarray([100], dtype=np.int64),
            position_buf=SimpleNamespace(ptr=0x3000),
            context_buf=SimpleNamespace(ptr=0x4000),
        ),
    }
    calls: list[tuple[int, int, int]] = []
    executor = object.__new__(Qwen35GGUFNextNExecutor)
    executor._request_slots = {10: 0, 20: 1}
    executor.scratch = SimpleNamespace(
        for_slot=lambda slot, **kwargs: slots[(10, 20)[int(slot)]]
    )
    executor._batch_session = SimpleNamespace(
        _runtime_state_library=object(),
    )
    executor.runtime = object()
    monkeypatch.setattr(
        nextn_mod,
        "set_decode_position_i64",
        lambda position_ptr, context_ptr, position, **kwargs: calls.append(
            (int(position_ptr), int(context_ptr), int(position))
        ),
    )

    executor._publish_batch_consumed_positions((10, 20), (36, 41))

    assert slots[10].position_host.tolist() == [36]
    assert slots[10].context_host.tolist() == [37]
    assert slots[20].position_host.tolist() == [41]
    assert slots[20].context_host.tolist() == [42]
    assert calls == [(0x1000, 0x2000, 36), (0x3000, 0x4000, 41)]


def test_nextn_private_slot_buffer_rebases_one_slot() -> None:
    buffer = DeviceBuffer(0x1000, 400)

    slot = _private_slot_buffer(buffer, slot=2, slot_count=4)

    assert slot.ptr == 0x1000 + 200
    assert slot.nbytes == 100
    with pytest.raises(ValueError, match="slot-major"):
        _private_slot_buffer(DeviceBuffer(0x1000, 401), slot=1, slot_count=4)


def test_nextn_singleton_block_uses_local_kv_scratch(monkeypatch) -> None:
    captured = []
    local_scratch = SimpleNamespace(
        set_full_attention_position=lambda *args: None,
        attn_out=SimpleNamespace(ptr=0xA100),
    )
    scratch_calls = []
    executor = object.__new__(Qwen35GGUFNextNExecutor)
    executor.closed = False
    executor.hidden_size = 8
    executor.vocab_size = 32
    executor._request_slots = {7: 1}
    executor._singleton_slot_scratch = (
        lambda slot: scratch_calls.append(int(slot)) or local_scratch
    )
    executor.runtime = object()
    executor._token_buf = SimpleNamespace(ptr=0x1000)
    executor._embedding_buf = SimpleNamespace(ptr=0x2000)
    executor._fusion_buf = SimpleNamespace(ptr=0x3000)
    executor._fused_buf = SimpleNamespace(ptr=0x4000)
    executor._layer_out_buf = SimpleNamespace(ptr=0x5000)
    executor._final_hidden_buf = SimpleNamespace(ptr=0x6000)

    allocation = lambda ptr: SimpleNamespace(
        allocation=lambda: SimpleNamespace(tensor=SimpleNamespace(ptr=ptr))
    )
    fallback = {"token_embedding": allocation(0x7000)}
    nextn = {
        "enorm": allocation(0x7100),
        "hnorm": allocation(0x7200),
        "eh_proj": allocation(0x7300),
    }
    executor.weights = SimpleNamespace(
        fallback=lambda name: fallback[name],
        nextn=lambda name: nextn[name],
        config=SimpleNamespace(rms_norm_eps=1.0e-6),
    )
    executor.runner = SimpleNamespace(
        _run_full_attention_layer=lambda layer, hidden, out, scratch, **kwargs: (
            captured.append(scratch)
        )
    )
    monkeypatch.setattr(nextn_mod, "launch_gguf_embedding", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        nextn_mod, "gguf_rmsnorm_bf16_f32_weight", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(nextn_mod, "launch_gguf_linear", lambda *args, **kwargs: None)

    executor._run_block(
        7,
        0,
        5,
        Tensor.from_handle(0x8000, (1, 8), DType.BF16, Device("hip", 0)),
        token_ready=True,
    )

    assert captured == [local_scratch]
    assert scratch_calls == [1]


def test_nextn_provider_keeps_physical_batch_candidates_device_resident_until_materialized() -> None:
    calls = []
    device = Qwen35GGUFNextNBatchDeviceProposal(
        request_ids=(10, 20),
        root_tokens=(100, 200),
        root_positions=(5, 8),
        candidate_counts=(1, 2),
        token_ids=Tensor.from_handle(
            0x5000,
            (3,),
            DType.INT32,
            Device("hip", 0),
        ),
        hidden_rows=(
            (Tensor.from_handle(0x6000, (1, 8), DType.BF16, Device("hip", 0)),),
            (
                Tensor.from_handle(0x7000, (1, 8), DType.BF16, Device("hip", 0)),
                Tensor.from_handle(0x8000, (1, 8), DType.BF16, Device("hip", 0)),
            ),
        ),
    )

    class DeviceExecutor:
        hidden_size = 8

        def run_batch_proposal_device(self, context, *, candidate_counts):
            calls.append(("launch", tuple(context.request_ids), tuple(candidate_counts)))
            return device

        def materialize_batch_device_proposal(self, pending):
            calls.append(("materialize", pending.token_ids.ptr))
            return (101, 201, 202)

    provider = Qwen35GGUFNextNDraftProvider(DeviceExecutor())
    context = MtpProposalContext(
        request_ids=(10, 20),
        root_tokens=(100, 200),
        root_positions=(5, 8),
        target_hidden=Tensor.from_handle(
            0x1000,
            (2, 8),
            DType.BF16,
            Device("hip", 0),
        ),
    )

    pending = provider.propose_batch_device(
        context,
        candidate_counts=(1, 2),
    )

    assert pending is device
    assert provider.last_results == {}
    assert calls == [("launch", (10, 20), (1, 2))]

    draft = provider.materialize_batch_device_proposal(pending)

    assert calls[-1] == ("materialize", 0x5000)
    assert draft.candidate_tokens == (101, 0, 201, 202)
    assert draft.active_mask == (True, False, True, True)
    assert tuple(row.token_id for row in provider.last_results[10]) == (101,)
    assert tuple(row.token_id for row in provider.last_results[20]) == (201, 202)


def test_nextn_provider_physically_batches_one_backbone_per_depth() -> None:
    calls = []

    class BatchExecutor:
        hidden_size = 8
        runtime = SimpleNamespace(memcpy=lambda *args: None)
        _batch_input_hidden = SimpleNamespace(ptr=4000)
        _final_hidden_buf = SimpleNamespace(ptr=5000)

        def run_step_batch(self, request_ids, token_ids, positions, target_hidden):
            calls.append((tuple(request_ids), tuple(token_ids), tuple(positions)))
            return tuple(
                Qwen35GGUFNextNStepResult(
                    request_id=request_id,
                    input_token=token_id,
                    position=position,
                    token_id=token_id + 1,
                    logit=1.0,
                    hidden=Tensor.from_handle(
                        6000 + row * 16,
                        (1, 8),
                        DType.BF16,
                        Device("hip", 0),
                    ),
                )
                for row, (request_id, token_id, position) in enumerate(
                    zip(request_ids, token_ids, positions, strict=True)
                )
            )

        def _preserve_proposal_hidden(self, request_id, depth, hidden):
            return hidden

    provider = Qwen35GGUFNextNDraftProvider(BatchExecutor())
    context = MtpProposalContext(
        request_ids=(10, 20, 30, 40),
        root_tokens=(100, 200, 300, 400),
        root_positions=(5, 6, 7, 8),
        target_hidden=Tensor.from_handle(
            1000,
            (4, 8),
            DType.BF16,
            Device("hip", 0),
        ),
    )

    draft = provider.propose_batch(
        context,
        candidate_counts=(2, 2, 2, 2),
    )

    assert len(calls) == 2
    assert all(len(call[0]) == 4 for call in calls)
    assert calls[0] == (
        (10, 20, 30, 40),
        (100, 200, 300, 400),
        (5, 6, 7, 8),
    )
    assert calls[1][1] == (101, 201, 301, 401)
    assert draft.request_ids == (10, 20, 30, 40)
    assert draft.draft_rows == 8
    assert draft.row_to_request == (10, 10, 20, 20, 30, 30, 40, 40)


def test_nextn_provider_batches_shared_depth_and_runs_only_ragged_tail_alone() -> None:
    calls = []

    class RaggedExecutor:
        hidden_size = 8
        runtime = SimpleNamespace(memcpy=lambda *args: None)
        _batch_input_hidden = SimpleNamespace(ptr=4000)

        def run_step_batch(self, request_ids, token_ids, positions, target_hidden):
            calls.append(("batch", tuple(request_ids), tuple(positions)))
            return tuple(
                Qwen35GGUFNextNStepResult(
                    request_id=request_id,
                    input_token=token_id,
                    position=position,
                    token_id=token_id + 1,
                    logit=1.0,
                    hidden=Tensor.from_handle(
                        6000 + row * 16,
                        (1, 8),
                        DType.BF16,
                        Device("hip", 0),
                    ),
                )
                for row, (request_id, token_id, position) in enumerate(
                    zip(request_ids, token_ids, positions, strict=True)
                )
            )

        def run_step(self, request_id, token_id, position, target_hidden, *, return_logits):
            calls.append(("single", request_id, position))
            return Qwen35GGUFNextNStepResult(
                request_id=request_id,
                input_token=token_id,
                position=position,
                token_id=token_id + 1,
                logit=1.0,
                hidden=Tensor.from_handle(
                    7000,
                    (1, 8),
                    DType.BF16,
                    Device("hip", 0),
                ),
            )

        def _preserve_proposal_hidden(self, request_id, depth, hidden):
            return hidden

    provider = Qwen35GGUFNextNDraftProvider(RaggedExecutor())
    draft = provider.propose_batch(
        MtpProposalContext(
            request_ids=(10, 20),
            root_tokens=(100, 200),
            root_positions=(5, 8),
            target_hidden=Tensor.from_handle(
                1000,
                (2, 8),
                DType.BF16,
                Device("hip", 0),
            ),
        ),
        candidate_counts=(1, 2),
    )

    assert calls == [("batch", (10, 20), (5, 8)), ("single", 20, 9)]
    assert draft.active_mask == (True, False, True, True)
    assert tuple(
        (request_id, depth, token)
        for request_id, depth, token, active in zip(
            draft.row_to_request,
            draft.draft_depths,
            draft.candidate_tokens,
            draft.active_mask,
            strict=True,
        )
        if active
    ) == ((10, 1, 101), (20, 1, 201), (20, 2, 202))


def test_nextn_provider_prefers_an_executor_chain_without_changing_draft_abi() -> None:
    executor = _FakeExecutor()
    chain_calls: list[tuple[int, int, int, int, int, bool]] = []

    def run_chain(
        request_id: int,
        token_id: int,
        position: int,
        target_hidden: Tensor,
        *,
        candidate_budget: int,
        return_logits: bool = False,
    ) -> tuple[Qwen35GGUFNextNStepResult, ...]:
        chain_calls.append(
            (
                request_id,
                token_id,
                position,
                target_hidden.ptr,
                candidate_budget,
                return_logits,
            )
        )
        rows = []
        hidden = target_hidden
        current = token_id
        for depth in range(candidate_budget):
            current += 1
            hidden = Tensor.from_handle(2000 + depth, (1, 8), DType.BF16, Device("hip", 0))
            rows.append(
                Qwen35GGUFNextNStepResult(
                    request_id=request_id,
                    input_token=current - 1,
                    position=position + depth,
                    token_id=current,
                    logit=float(current),
                    hidden=hidden,
                )
            )
        return tuple(rows)

    executor.run_chain = run_chain  # type: ignore[attr-defined]
    provider = Qwen35GGUFNextNDraftProvider(executor)
    target_hidden = Tensor.from_handle(77, (1, 8), DType.BF16, Device("hip", 0))

    draft = provider.propose(
        MtpProposalContext(
            request_ids=(41,),
            root_tokens=(9,),
            root_positions=(12,),
            target_hidden=target_hidden,
        ),
        candidate_budget=3,
    )

    assert chain_calls == [(41, 9, 12, 77, 3, False)]
    assert executor.calls == []
    assert draft.candidate_tokens == (10, 11, 12)
    assert draft.parent_positions == (12, 13, 14)
    assert draft.draft_depths == (1, 2, 3)
    assert tuple(row.input_token for row in provider.last_results[41]) == (9, 10, 11)


def test_nextn_provider_materializes_a_cached_device_proposal_after_target_retirement() -> None:
    executor = _FakeExecutor()
    hidden = Tensor.from_handle(77, (1, 8), DType.BF16, Device("hip", 0))
    proposal = Qwen35GGUFNextNDeviceProposal(
        request_id=41,
        root_token=9,
        root_position=12,
        budget=3,
        result_ptr=0x5000,
        result_nbytes=24,
        completion_event=0x6000,
        stream=0x7000,
        final_hidden=Tensor.from_handle(0x8000, (1, 8), DType.BF16, Device("hip", 0)),
    )
    launch_calls: list[tuple[int, int, int, int, int]] = []

    def launch_device(
        request_id: int,
        token_id: int,
        position: int,
        target_hidden: Tensor,
        *,
        candidate_budget: int,
    ) -> Qwen35GGUFNextNDeviceProposal:
        launch_calls.append(
            (request_id, token_id, position, target_hidden.ptr, candidate_budget)
        )
        return proposal

    def materialize_device(
        pending: Qwen35GGUFNextNDeviceProposal,
        *,
        token_ids: tuple[int, ...],
        top1_values: tuple[float, ...],
    ) -> tuple[Qwen35GGUFNextNStepResult, ...]:
        assert pending is proposal
        rows = []
        current = pending.root_token
        for depth, (token, value) in enumerate(
            zip(token_ids, top1_values, strict=True)
        ):
            rows.append(
                Qwen35GGUFNextNStepResult(
                    request_id=pending.request_id,
                    input_token=current,
                    position=pending.root_position + depth,
                    token_id=token,
                    logit=value,
                    hidden=pending.final_hidden,
                )
            )
            current = token
        return tuple(rows)

    executor.launch_cached_graph_chain_device = launch_device  # type: ignore[attr-defined]
    executor.materialize_device_proposal = materialize_device  # type: ignore[attr-defined]
    provider = Qwen35GGUFNextNDraftProvider(executor)
    context = MtpProposalContext(
        request_ids=(41,),
        root_tokens=(9,),
        root_positions=(12,),
        target_hidden=hidden,
    )

    pending = provider.launch_device_proposal(context, candidate_budget=3)

    assert pending is proposal
    assert launch_calls == [(41, 9, 12, 77, 3)]
    placeholder = provider.placeholder_device_proposal(pending)
    assert placeholder.candidate_tokens == (0, 0, 0)
    draft = provider.finish_device_proposal(
        pending,
        token_ids=(10, 11, 12),
        top1_values=(1.0, 2.0, 3.0),
    )
    assert draft.candidate_tokens == (10, 11, 12)
    assert draft.parent_positions == (12, 13, 14)
    assert tuple(row.input_token for row in provider.last_results[41]) == (9, 10, 11)
    assert tuple(row.logit for row in provider.last_results[41]) == (1.0, 2.0, 3.0)


def test_nextn_executor_chain_falls_back_to_eager_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = object.__new__(Qwen35GGUFNextNExecutor)
    executor.closed = False
    executor.weights = object()
    executor.hidden_size = 8
    executor.vocab_size = 64
    executor.scratch = SimpleNamespace(max_positions=64)
    graph_calls: list[tuple[int, int, int, int, int]] = []
    step_calls: list[tuple[int, int, int, int, bool]] = []

    def graph_chain(request_id, token_id, position, target_hidden, *, candidate_budget):
        graph_calls.append((request_id, token_id, position, target_hidden.ptr, candidate_budget))
        return None

    def run_step(request_id, token_id, position, target_hidden, *, return_logits=False):
        step_calls.append((request_id, token_id, position, target_hidden.ptr, return_logits))
        hidden = Tensor.from_handle(3000 + len(step_calls), (1, 8), DType.BF16, Device("hip", 0))
        return Qwen35GGUFNextNStepResult(
            request_id=request_id,
            input_token=token_id,
            position=position,
            token_id=token_id + 1,
            logit=float(token_id + 1),
            hidden=hidden,
        )

    monkeypatch.setattr(executor, "_run_exact_graph_chain", graph_chain, raising=False)
    monkeypatch.setattr(executor, "run_step", run_step)
    hidden = Tensor.from_handle(77, (1, 8), DType.BF16, Device("hip", 0))

    rows = executor.run_chain(41, 9, 12, hidden, candidate_budget=3)

    assert graph_calls == [(41, 9, 12, 77, 3)]
    assert step_calls == [
        (41, 9, 12, 77, False),
        (41, 10, 13, 3001, False),
        (41, 11, 14, 3002, False),
    ]
    assert tuple(row.token_id for row in rows) == (10, 11, 12)


def test_nextn_proposal_graph_keeps_split_attention_context_on_eager_path() -> None:
    executor = object.__new__(Qwen35GGUFNextNExecutor)
    executor.scratch = SimpleNamespace(max_positions=2048)
    executor._proposal_target_hidden = object()
    executor._proposal_results = object()
    executor._proposal_results_host = object()
    executor._proposal_graph_last_status = "ready"
    hidden = Tensor.from_handle(77, (1, 8), DType.BF16, Device("hip", 0))

    rows = executor._run_exact_graph_chain(
        41,
        9,
        1021,
        hidden,
        candidate_budget=3,
    )

    assert rows is None
    assert executor._proposal_graph_last_status == "eager_long_context"


def test_nextn_proposal_graph_capture_failure_is_cached_as_eager_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = object.__new__(Qwen35GGUFNextNExecutor)
    executor._proposal_graphs = {}
    executor._proposal_graph_unavailable = set()
    executor._proposal_graph_last_status = "ready"
    executor._proposal_graph_last_error = None
    executor._proposal_graph_captures = 0
    capture_calls: list[tuple[int, int, int]] = []

    def fail_capture(request_id: int, slot: int, budget: int):
        capture_calls.append((request_id, slot, budget))
        raise RuntimeError("unsupported graph")

    monkeypatch.setattr(executor, "_capture_exact_chain_graph", fail_capture)

    assert executor._proposal_graph(41, 0, 3) is None
    assert executor._proposal_graph(41, 0, 3) is None
    assert capture_calls == [(41, 0, 3)]
    assert executor._proposal_graph_unavailable == {(0, 3)}
    assert executor._proposal_graph_last_status == "capture_fallback"
    assert executor._proposal_graph_last_error == "RuntimeError: unsupported graph"


def test_nextn_provider_advances_only_a_fully_accepted_tail() -> None:
    executor = _FakeExecutor()
    provider = Qwen35GGUFNextNDraftProvider(executor)
    context = MtpProposalContext(
        request_ids=(41,),
        root_tokens=(9,),
        root_positions=(12,),
        target_hidden=Tensor.from_handle(77, (1, 8), DType.BF16, Device("hip", 0)),
    )
    provider.propose(context, candidate_budget=2)

    assert provider.advance_full_accept_tail(41, accepted_count=1) is None
    assert len(executor.calls) == 2
    update = provider.advance_full_accept_tail(41, accepted_count=2)
    assert isinstance(update, Qwen35GGUFNextNStateAdvance)
    assert len(executor.calls) == 2
    assert executor.state_only_calls == [(41, 11, 14, 1002)]
    assert update.input_token == 11
    with pytest.raises(ValueError, match="prior proposal"):
        provider.advance_full_accept_tail(42, accepted_count=0)
    with pytest.raises(ValueError, match="prior proposal budget"):
        provider.advance_full_accept_tail(41, accepted_count=3)


def test_nextn_executor_bulk_prompt_stages_hidden_rows_until_finish() -> None:
    executor = object.__new__(Qwen35GGUFNextNExecutor)
    executor.hidden_size = 8
    executor._bulk_prompt_prime = True
    executor._prompt_prime_hidden = DeviceBuffer(0x10000, 4 * 128 * 16)
    executor._prompt_prime_capacity = 128
    executor._prompt_prime_rows = {}
    executor._slot = lambda _request_id: 1
    executor.scratch = SimpleNamespace(max_positions=128)
    copies: list[tuple[int, int, int, int]] = []
    executor.runtime = SimpleNamespace(
        memcpy_async=lambda dst, src, nbytes, _kind, stream: copies.append(
            (int(dst), int(src), int(nbytes), int(stream))
        )
    )

    executor.enqueue_prompt_rows(
        7,
        (11, 22),
        position_start=3,
        target_hidden_base_ptr=0x20000,
        hidden_stride_bytes=32,
        stream=5,
    )

    assert executor._prompt_prime_rows == {7: {3: 11, 4: 22}}
    assert copies == [
        (0x10000 + (128 + 3) * 16, 0x20000, 16, 5),
        (0x10000 + (128 + 4) * 16, 0x20020, 16, 5),
    ]


def test_nextn_executor_bulk_prompt_finish_materializes_and_releases() -> None:
    executor = object.__new__(Qwen35GGUFNextNExecutor)
    executor._bulk_prompt_prime = True
    executor._prompt_prime_rows = {7: {0: 11}}
    executor._prompt_priming_staging = {7: [np.asarray([11], dtype=np.int64)]}
    calls: list[tuple[str, int, int]] = []
    executor._prime_prompt_rows_bulk = lambda request_id, stream: calls.append(
        ("prime", int(request_id), int(stream))
    )
    executor.runtime = SimpleNamespace(
        stream_synchronize=lambda stream: calls.append(("sync", 0, int(stream)))
    )

    executor.finish_prompt_priming(7, stream=5, synchronize=True)

    assert calls == [("prime", 7, 5), ("sync", 0, 5)]
    assert executor._prompt_prime_rows == {}
    assert executor._prompt_priming_staging == {}


def test_nextn_executor_enqueues_prompt_rows_on_target_stream_without_scoring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = object.__new__(Qwen35GGUFNextNExecutor)
    executor.hidden_size = 8
    executor.compiler_version = None
    executor.require_cached_build = False
    executor._proposal_graph_runtime_library = object()
    executor._prompt_priming_staging = {}
    executor._token_buf = DeviceBuffer(0x1000, DType.INT64.itemsize)
    executor._slot = lambda _request_id: 0
    executor._batch_sessions = [SimpleNamespace(_position=0)]

    slot_scratch = SimpleNamespace(
        position_host=np.zeros((1,), dtype=np.int64),
        context_host=np.zeros((1,), dtype=np.int64),
        position_buf=DeviceBuffer(0x2000, DType.INT64.itemsize),
        context_buf=DeviceBuffer(0x3000, DType.INT64.itemsize),
    )
    executor.scratch = SimpleNamespace(
        max_positions=128,
        for_slot=lambda _slot, **_kwargs: slot_scratch,
    )
    runtime_calls: list[tuple[object, ...]] = []
    executor.runtime = SimpleNamespace(
        memcpy_async=lambda dst, src, nbytes, kind, stream: runtime_calls.append(
            ("copy", int(dst), int(nbytes), int(kind), int(stream))
        ),
        stream_synchronize=lambda stream: runtime_calls.append(("sync", int(stream))),
    )
    metadata_calls: list[tuple[int, int]] = []
    monkeypatch.setattr(
        nextn_mod,
        "set_decode_position_i64",
        lambda _position, _context, value, **kwargs: metadata_calls.append(
            (int(value), int(kwargs["stream"]))
        ),
    )
    block_calls: list[tuple[int, int, int, int, int, bool, bool, bool]] = []

    def fake_run_block(
        request_id,
        token_id,
        position,
        target_hidden,
        *,
        stream,
        token_ready,
        position_ready,
        kv_write_only,
    ):
        block_calls.append(
            (
                int(request_id),
                int(token_id),
                int(position),
                int(target_hidden.ptr),
                int(stream),
                bool(token_ready),
                bool(position_ready),
                bool(kv_write_only),
            )
        )
        return 0x7000, 0x8000

    executor._run_block = fake_run_block

    executor.enqueue_prompt_rows(
        7,
        (11, 22, 33),
        position_start=9,
        target_hidden_base_ptr=0x4000,
        hidden_stride_bytes=32,
        stream=5,
    )

    assert runtime_calls == [
        ("copy", 0x1000, 8, int(HipMemcpyKind.HOST_TO_DEVICE), 5),
        ("copy", 0x1000, 8, int(HipMemcpyKind.HOST_TO_DEVICE), 5),
        ("copy", 0x1000, 8, int(HipMemcpyKind.HOST_TO_DEVICE), 5),
    ]
    assert metadata_calls == [(9, 5), (10, 5), (11, 5)]
    assert block_calls == [
        (7, 11, 9, 0x4000, 5, True, True, True),
        (7, 22, 10, 0x4020, 5, True, True, True),
        (7, 33, 11, 0x4040, 5, True, True, True),
    ]
    assert len(executor._prompt_priming_staging[7]) == 1
    assert executor._batch_sessions[0]._position == 12

    executor.finish_prompt_priming(7, stream=5, synchronize=False)
    assert 7 not in executor._prompt_priming_staging
    assert not any(call[0] == "sync" for call in runtime_calls)


def test_nextn_executor_state_only_tail_uses_kv_write_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = object.__new__(Qwen35GGUFNextNExecutor)
    runtime_calls: list[str] = []
    block_calls: list[tuple[int, int, int, int, bool]] = []
    executor.runtime = SimpleNamespace(
        device_synchronize=lambda: runtime_calls.append("synchronize")
    )
    hidden = Tensor.from_handle(0x7000, (1, 8), DType.BF16, Device("hip", 0))

    def fake_run_block(
        request_id: int,
        token_id: int,
        position: int,
        target_hidden: Tensor,
        *,
        kv_write_only: bool,
    ) -> tuple[int, int]:
        block_calls.append(
            (request_id, token_id, position, target_hidden.ptr, kv_write_only)
        )
        return 0x8000, 0x9000

    monkeypatch.setenv("HIPENGINE_GGUF_NEXTN_ACCEPT_KV_WRITE_ONLY", "1")
    monkeypatch.setattr(executor, "_run_block", fake_run_block, raising=False)
    monkeypatch.setattr(
        executor,
        "_sample_lm_head",
        lambda *args, **kwargs: pytest.fail("state-only tail must not sample lm-head"),
    )

    result = executor.advance_state_only(41, 11, 14, hidden)

    assert result == Qwen35GGUFNextNStateAdvance(
        request_id=41,
        input_token=11,
        position=14,
    )
    assert block_calls == [(41, 11, 14, 0x7000, True)]
    assert runtime_calls == ["synchronize"]


def test_nextn_executor_device_state_only_tail_uses_kv_write_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = object.__new__(Qwen35GGUFNextNExecutor)
    calls: list[tuple[object, ...]] = []
    executor.runtime = SimpleNamespace(
        device_synchronize=lambda: calls.append(("synchronize",))
    )
    executor._slot = lambda request_id: 2
    executor._token_buf = SimpleNamespace(ptr=0x1000)
    executor._batch_session = SimpleNamespace(
        _runtime_state_library="runtime-library"
    )
    executor._set_batch_session_position = (
        lambda slot, position: calls.append(("position", slot, position))
    )
    monkeypatch.setenv("HIPENGINE_GGUF_NEXTN_ACCEPT_KV_WRITE_ONLY", "1")
    monkeypatch.setattr(
        nextn_mod,
        "copy_i32_to_i64",
        lambda src, dst, count, **kwargs: calls.append(
            ("copy", src, dst, count, kwargs["library"])
        ),
    )
    executor._run_block = lambda *args, **kwargs: calls.append(
        ("block", args, kwargs)
    )
    token = Tensor.from_handle(0x2000, (1,), DType.INT32, Device("hip", 0))
    hidden = Tensor.from_handle(0x3000, (1, 8), DType.BF16, Device("hip", 0))

    executor.advance_state_only_device(41, token, 14, hidden)

    assert calls[0] == ("copy", 0x2000, 0x1010, 1, "runtime-library")
    assert calls[1] == (
        "block",
        (41, 0, 14, hidden),
        {"token_ready": True, "kv_write_only": True},
    )
    assert calls[2:] == [("position", 2, 15), ("synchronize",)]


def test_nextn_executor_batch_state_only_requests_kv_write_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = object.__new__(Qwen35GGUFNextNExecutor)
    calls: list[dict[str, object]] = []
    monkeypatch.setenv("HIPENGINE_GGUF_NEXTN_ACCEPT_KV_WRITE_ONLY", "1")

    def fake_batch(*args, **kwargs):
        calls.append(kwargs)
        return (
            Qwen35GGUFNextNStateAdvance(10, 101, 4),
            Qwen35GGUFNextNStateAdvance(20, 202, 7),
        )

    executor._run_step_batch_impl = fake_batch
    hidden = Tensor.from_handle(0x4000, (2, 8), DType.BF16, Device("hip", 0))
    result = executor.advance_state_batch_only((10, 20), (101, 202), (4, 7), hidden)
    assert len(result) == 2
    assert calls == [{"score_output": False, "kv_write_only": True}]


@pytest.mark.parametrize("quant_key", ["gguf_q6_k", "gguf_q6_k_t16_v1"])
def test_nextn_executor_prepares_exact_top1_through_quant_registry(
    monkeypatch: pytest.MonkeyPatch,
    quant_key: str,
) -> None:
    executor = object.__new__(Qwen35GGUFNextNExecutor)
    weight = SimpleNamespace(
        spec=SimpleNamespace(quant_key=quant_key),
        allocation=lambda name: SimpleNamespace(tensor=SimpleNamespace(ptr=0x2000))
        if name in {"raw", "tiles"}
        else (_ for _ in ()).throw(KeyError(name)),
    )
    executor.weights = SimpleNamespace(
        backend="hip_gfx1100",
        fallback=lambda slot: weight
        if slot == "lm_head"
        else (_ for _ in ()).throw(KeyError(slot)),
    )
    executor.hidden_size = 512
    executor.vocab_size = 1024
    executor.compiler_version = "compiler"
    executor.require_cached_build = True
    executor.runtime = object()
    executor._lm_head_top1_kernel = None
    executor._lm_head_top1_weight = None
    executor._lm_head_top1_block_values = None
    executor._lm_head_top1_block_indices = None
    executor._lm_head_top1_result = None
    executor._lm_head_top1_libraries = None
    kernel = object()
    pack8_library = object()
    t16_library = object()
    registered_keys: list[object] = []
    resolve_calls: list[dict[str, object]] = []
    malloc_calls: list[int] = []

    def fake_is_registered(key) -> bool:
        registered_keys.append(key)
        return True

    def fake_resolve(**kwargs):
        resolve_calls.append(kwargs)
        return kernel

    def fake_malloc(nbytes, *, runtime):
        assert runtime is executor.runtime
        malloc_calls.append(nbytes)
        return SimpleNamespace(ptr=0x3000 + len(malloc_calls) * 0x1000, nbytes=nbytes)

    monkeypatch.setattr(nextn_mod, "is_registered", fake_is_registered)
    monkeypatch.setattr(nextn_mod, "resolve", fake_resolve)
    monkeypatch.setattr(
        nextn_mod,
        "build_gguf_q6_k_pack8_gemv",
        lambda **kwargs: pack8_library,
    )
    monkeypatch.setattr(
        nextn_mod,
        "build_gguf_q6_k_t16_gemv",
        lambda **kwargs: t16_library,
        raising=False,
    )
    monkeypatch.setattr(nextn_mod, "malloc", fake_malloc)

    executor._prepare_exact_lm_head_top1()

    assert [(key.backend, key.layer, key.quant, key.variant) for key in registered_keys] == [
        (
            "hip_gfx1100",
            "linear+argmax",
            quant_key,
            "proposal_top1_exact_bf16",
        )
    ]
    assert resolve_calls == [
        {
            "backend": "hip_gfx1100",
            "layer": "linear+argmax",
            "quant": quant_key,
            "variant": "proposal_top1_exact_bf16",
        }
    ]
    assert malloc_calls == [512, 512, 8]
    assert executor._lm_head_top1_kernel is kernel
    assert executor._lm_head_top1_weight is weight
    assert executor._lm_head_top1_libraries == {
        "q6_pack8": pack8_library,
        "q6_t16": t16_library,
    }


def test_nextn_executor_prefers_model_bound_hot_vocab_with_full_head_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = object.__new__(Qwen35GGUFNextNExecutor)
    full_weight = SimpleNamespace(spec=SimpleNamespace(quant_key="gguf_q6_k_t16_qmicro_planar_v1"))
    hot_weight = SimpleNamespace(spec=SimpleNamespace(quant_key="gguf_q6_k_t16_qmicro_planar_v1"))
    hot_vocab = SimpleNamespace(
        lm_head=hot_weight,
        size=256,
        token_ids=SimpleNamespace(tensor=SimpleNamespace(ptr=0x9000)),
    )
    executor.weights = SimpleNamespace(
        backend="hip_gfx1100",
        hot_vocab=hot_vocab,
        fallback=lambda slot: full_weight,
    )
    executor.hidden_size = 512
    executor.vocab_size = 1024
    executor.compiler_version = "compiler"
    executor.require_cached_build = True
    executor.runtime = object()
    executor._lm_head_top1_kernel = None
    executor._lm_head_top1_weight = None
    executor._lm_head_top1_block_values = None
    executor._lm_head_top1_block_indices = None
    executor._lm_head_top1_result = None
    executor._lm_head_top1_libraries = None
    kernel = object()
    keys = []
    malloc_calls = []

    def fake_is_registered(key) -> bool:
        keys.append(key)
        return key.variant == "proposal_top1_mapped_bf16"

    monkeypatch.setattr(nextn_mod, "is_registered", fake_is_registered)
    monkeypatch.setattr(nextn_mod, "resolve", lambda **kwargs: kernel)
    monkeypatch.setattr(nextn_mod, "build_gguf_q6_k_pack8_gemv", lambda **kwargs: object())
    monkeypatch.setattr(nextn_mod, "build_gguf_q6_k_t16_gemv", lambda **kwargs: object())
    monkeypatch.setattr(
        nextn_mod,
        "malloc",
        lambda nbytes, **kwargs: (
            malloc_calls.append(nbytes)
            or SimpleNamespace(ptr=0xA000 + len(malloc_calls) * 0x100, nbytes=nbytes)
        ),
    )

    executor._prepare_exact_lm_head_top1()

    assert [key.variant for key in keys] == ["proposal_top1_mapped_bf16"]
    assert malloc_calls == [128, 128, 8]
    assert executor._lm_head_top1_kernel is kernel
    assert executor._lm_head_top1_weight is hot_weight
    assert executor._lm_head_top1_token_map_ptr == 0x9000
    assert executor._lm_head_top1_vocab_size == 256
    assert executor._lm_head_top1_mapped is True


def test_nextn_executor_exact_top1_reads_only_token_and_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = object.__new__(Qwen35GGUFNextNExecutor)
    kernel_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    sync_calls: list[str] = []
    weight = object()
    libraries = {"q6_pack8": object(), "q6_t16": object()}
    runtime = SimpleNamespace(device_synchronize=lambda: sync_calls.append("sync"))
    executor.runtime = runtime
    executor.hidden_size = 512
    executor.vocab_size = 1024
    executor._logits_buf = SimpleNamespace(ptr=0x6000)
    executor._lm_head_top1_kernel = lambda *args, **kwargs: kernel_calls.append((args, kwargs))
    executor._lm_head_top1_weight = weight
    executor._lm_head_top1_block_values = SimpleNamespace(ptr=0x3000)
    executor._lm_head_top1_block_indices = SimpleNamespace(ptr=0x4000)
    executor._lm_head_top1_result = SimpleNamespace(ptr=0x5000, nbytes=8)
    executor._lm_head_top1_libraries = libraries

    def fake_copy(host_ptr, _device, nbytes, *, runtime) -> None:
        assert nbytes == 8
        assert runtime is executor.runtime
        result = (ctypes.c_uint32 * 2).from_address(host_ptr)
        result[0] = 731
        result[1] = int(np.asarray([4.25], dtype=np.float32).view(np.uint32)[0])

    monkeypatch.setattr(nextn_mod, "copy_device_to_host", fake_copy)

    compact = executor._run_exact_lm_head_top1(0x1000, stream=7)

    assert compact == (731, 4.25)
    assert sync_calls == ["sync"]
    assert len(kernel_calls) == 1
    args, kwargs = kernel_calls[0]
    assert args == (
        weight,
        0x1000,
        0x6000,
        0x3000,
        0x4000,
        0x5000,
        0x5004,
        1,
        512,
        1024,
    )
    assert kwargs == {"stream": 7, "libraries": libraries, "runtime": runtime}


def test_nextn_executor_mapped_top1_passes_compact_and_full_vocab() -> None:
    executor = object.__new__(Qwen35GGUFNextNExecutor)
    calls = []
    executor.runtime = object()
    executor.hidden_size = 512
    executor.vocab_size = 1024
    executor._logits_buf = SimpleNamespace(ptr=0x6000)
    executor._lm_head_top1_kernel = lambda *args, **kwargs: calls.append((args, kwargs))
    executor._lm_head_top1_weight = object()
    executor._lm_head_top1_block_values = SimpleNamespace(ptr=0x3000)
    executor._lm_head_top1_block_indices = SimpleNamespace(ptr=0x4000)
    executor._lm_head_top1_libraries = {"q6_pack8": object(), "q6_t16": object()}
    executor._lm_head_top1_token_map_ptr = 0x7000
    executor._lm_head_top1_vocab_size = 256
    executor._lm_head_top1_mapped = True

    assert executor._enqueue_exact_lm_head_top1(0x1000, 0x5000, 0x5004, stream=7)

    args, kwargs = calls[0]
    assert args[1:] == (
        0x1000,
        0x6000,
        0x3000,
        0x4000,
        0x5000,
        0x5004,
        0x7000,
        1,
        512,
        256,
        1024,
    )
    assert kwargs["stream"] == 7


def test_nextn_executor_sample_prefers_compact_top1_without_logits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = object.__new__(Qwen35GGUFNextNExecutor)
    compact_calls: list[tuple[int, int]] = []
    executor._run_exact_lm_head_top1 = lambda hidden_ptr, stream=0: (
        compact_calls.append((hidden_ptr, stream)) or (17, 3.5)
    )
    monkeypatch.setattr(
        nextn_mod,
        "launch_gguf_linear",
        lambda *args, **kwargs: pytest.fail("compact scoring must not launch full logits"),
    )

    token, logit, logits = executor._sample_lm_head(0x1234, return_logits=False, stream=9)

    assert (token, logit, logits) == (17, 3.5, None)
    assert compact_calls == [(0x1234, 9)]


def test_nextn_executor_logits_diagnostic_keeps_full_scoring_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = object.__new__(Qwen35GGUFNextNExecutor)
    executor.hidden_size = 8
    executor.vocab_size = 4
    executor.weights = SimpleNamespace(fallback=lambda slot: f"weight:{slot}")
    executor._logits_buf = SimpleNamespace(ptr=0x6000, nbytes=16)
    executor._logits_host = np.empty((1, 4), dtype=np.float32)
    sync_calls: list[str] = []
    executor.runtime = SimpleNamespace(device_synchronize=lambda: sync_calls.append("sync"))
    executor._run_exact_lm_head_top1 = lambda *_args, **_kwargs: pytest.fail(
        "diagnostic logits must bypass compact scoring"
    )
    launch_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    monkeypatch.setattr(
        nextn_mod,
        "launch_gguf_linear",
        lambda *args, **kwargs: launch_calls.append((args, kwargs)),
    )

    def fake_copy(host_ptr, _device, nbytes, *, runtime) -> None:
        assert nbytes == 16
        assert runtime is executor.runtime
        out = np.ctypeslib.as_array((ctypes.c_float * 4).from_address(host_ptr))
        out[:] = (1.0, 5.0, 3.0, 2.0)

    monkeypatch.setattr(nextn_mod, "copy_device_to_host", fake_copy)

    token, logit, logits = executor._sample_lm_head(0x7000, return_logits=True, stream=11)

    assert token == 1
    assert logit == 5.0
    np.testing.assert_array_equal(logits, np.asarray([[1.0, 5.0, 3.0, 2.0]], dtype=np.float32))
    assert sync_calls == ["sync"]
    assert len(launch_calls) == 1
    assert launch_calls[0][0][0:3] == ("weight:lm_head", 0x7000, 0x6000)
    assert launch_calls[0][1]["stream"] == 11


def _device_sha256(runtime, buffers: tuple[DeviceBuffer, ...]) -> str:
    import hashlib

    digest = hashlib.sha256()
    for buffer in buffers:
        host = np.empty((buffer.nbytes,), dtype=np.uint8)
        copy_device_to_host(host_array_ptr(host), buffer, buffer.nbytes, runtime=runtime)
        digest.update(host)
    return digest.hexdigest()


def _require_real_nextn(model: Path) -> None:
    if not model.exists():
        pytest.skip(f"local GGUF fixture not found: {model}")
    if not _hip_available():
        pytest.skip("HIP runtime is not available")
    try:
        build_qwen35_gguf_nextn_tensor_map(load_gguf_index(model))
    except MissingGGUFTensorError as exc:
        pytest.skip(f"local GGUF fixture is not MTP-capable: {exc}")
    free_bytes, _ = get_hip_runtime().mem_get_info()
    if free_bytes < 3 * 1024**3:
        pytest.skip(f"GGUF NextN one-step gate needs 3 GiB free VRAM; only {free_bytes / 1024**3:.2f} GiB")


@pytest.mark.skipif(not _MODEL.exists(), reason=f"local GGUF fixture not found: {_MODEL}")
def test_real_blk40_one_step_logits_match_direct_executor_and_provider() -> None:
    _require_real_nextn(_MODEL)
    runtime = get_hip_runtime()
    hidden_buf = malloc(2048 * DType.BF16.itemsize, runtime=runtime)
    runtime.memset(hidden_buf.ptr, 0, hidden_buf.nbytes)
    hidden = Tensor.from_handle(hidden_buf.ptr, (1, 2048), DType.BF16, Device("hip", 0))
    executor = Qwen35GGUFNextNExecutor(
        _MODEL,
        max_positions=256,
        max_requests=1,
        runtime=runtime,
        require_cached_build=os.environ.get("HIPENGINE_REQUIRE_CACHED_BUILD") == "1",
    )
    try:
        direct = executor.run_step(7, 11, 0, hidden, return_logits=True)
        assert direct.logits is not None
        assert direct.logits.shape == (1, executor.vocab_size)
        assert np.all(np.isfinite(direct.logits))
        oracle = json.loads(_ORACLE.read_text())
        top_ids = np.argpartition(direct.logits[0], -10)[-10:]
        top_ids = top_ids[np.argsort(direct.logits[0, top_ids])[::-1]]
        expected_ids = np.asarray([row[0] for row in oracle["top10"]], dtype=np.int64)
        expected_values = np.asarray([row[1] for row in oracle["top10"]], dtype=np.float32)
        np.testing.assert_array_equal(top_ids, expected_ids)
        tolerance = oracle["tolerance"]
        np.testing.assert_allclose(
            direct.logits[0, top_ids],
            expected_values,
            atol=tolerance["top10_logits_atol"],
            rtol=tolerance["top10_logits_rtol"],
        )
        assert direct.token_id == oracle["token_id"]
        executor.reset_request(7)

        compact = executor.run_step(7, 11, 0, hidden, return_logits=False)
        assert compact.logits is None
        assert compact.token_id == direct.token_id
        assert compact.logit == direct.logit
        assert executor.last_lm_head_path == "exact_q6_top1"
        executor.reset_request(7)

        provider = Qwen35GGUFNextNDraftProvider(executor)
        draft = provider.propose(
            MtpProposalContext(
                request_ids=(7,),
                root_tokens=(11,),
                root_positions=(0,),
                target_hidden=hidden,
            ),
            candidate_budget=1,
            return_logits=True,
        )
        proposed = provider.last_results[7][-1]
        assert proposed.logits is not None
        np.testing.assert_array_equal(proposed.logits, direct.logits)
        assert proposed.token_id == direct.token_id
        assert proposed.logit == direct.logit
        assert draft.candidate_tokens == (direct.token_id,)
        assert draft.parent_positions == (0,)
        assert draft.draft_depths == (1,)
    finally:
        executor.close()
        free(hidden_buf, runtime=runtime)


@pytest.mark.skipif(not _DENSE_MODEL.exists(), reason=f"local GGUF fixture not found: {_DENSE_MODEL}")
def test_real_dense_blk64_one_step_logits_match_llamacpp_oracle() -> None:
    _require_real_nextn(_DENSE_MODEL)
    runtime = get_hip_runtime()
    hidden_buf = malloc(5120 * DType.BF16.itemsize, runtime=runtime)
    runtime.memset(hidden_buf.ptr, 0, hidden_buf.nbytes)
    hidden = Tensor.from_handle(hidden_buf.ptr, (1, 5120), DType.BF16, Device("hip", 0))
    executor = Qwen35GGUFNextNExecutor(
        _DENSE_MODEL,
        # Keep the allocation above the scalar-attention graph ceiling to prove
        # short live contexts still use the exact graph in a server-sized cache.
        max_positions=2048,
        max_requests=1,
        runtime=runtime,
        require_cached_build=os.environ.get("HIPENGINE_REQUIRE_CACHED_BUILD") == "1",
    )
    try:
        assert executor.weights is not None
        assert executor.weights.block_id == 64
        assert not executor.weights.config.is_moe
        direct = executor.run_step(7, 9707, 0, hidden, return_logits=True)
        assert direct.logits is not None
        assert direct.logits.shape == (1, executor.vocab_size)
        assert np.all(np.isfinite(direct.logits))
        oracle = json.loads(_DENSE_ORACLE.read_text())
        top_ids = np.argpartition(direct.logits[0], -10)[-10:]
        top_ids = top_ids[np.argsort(direct.logits[0, top_ids])[::-1]]
        expected_ids = np.asarray([row[0] for row in oracle["top10"]], dtype=np.int64)
        expected_values = np.asarray([row[1] for row in oracle["top10"]], dtype=np.float32)
        np.testing.assert_array_equal(top_ids, expected_ids)
        tolerance = oracle["tolerance"]
        np.testing.assert_allclose(
            direct.logits[0, top_ids],
            expected_values,
            atol=tolerance["top10_logits_atol"],
            rtol=tolerance["top10_logits_rtol"],
        )
        assert direct.token_id == oracle["token_id"]
        executor.reset_request(7)

        compact = executor.run_step(7, 9707, 0, hidden, return_logits=False)
        assert compact.logits is None
        assert compact.token_id == direct.token_id
        assert compact.logit == direct.logit
        assert executor.last_lm_head_path == "exact_q6_top1"
        executor.reset_request(7)

        provider = Qwen35GGUFNextNDraftProvider(executor)
        draft = provider.propose(
            MtpProposalContext(
                request_ids=(7,),
                root_tokens=(9707,),
                root_positions=(0,),
                target_hidden=hidden,
            ),
            candidate_budget=1,
            return_logits=True,
        )
        proposed = provider.last_results[7][-1]
        assert proposed.logits is not None
        np.testing.assert_array_equal(proposed.logits, direct.logits)
        assert proposed.token_id == direct.token_id
        assert proposed.logit == direct.logit
        assert draft.candidate_tokens == (direct.token_id,)
        assert draft.parent_positions == (0,)
        assert draft.draft_depths == (1,)

        executor.reset_request(7)
        current_token = 9707
        current_hidden = hidden
        eager_rows = []
        for depth in range(3):
            row = executor.run_step(7, current_token, depth, current_hidden, return_logits=False)
            eager_rows.append(row)
            current_token = row.token_id
            current_hidden = row.hidden
        key_cache, value_cache = executor.scratch.full_cache(0)
        state_buffers = (
            DeviceBuffer(executor._final_hidden_buf.ptr, executor.hidden_size * DType.BF16.itemsize),
            key_cache,
            value_cache,
        )
        eager_state = _device_sha256(runtime, state_buffers)
        executor.reset_request(7)

        graph_rows = executor.run_chain(
            7,
            9707,
            0,
            hidden,
            candidate_budget=3,
            return_logits=False,
        )
        graph_state = _device_sha256(runtime, state_buffers)
        assert tuple(row.token_id for row in graph_rows) == tuple(row.token_id for row in eager_rows)
        assert tuple(row.logit for row in graph_rows) == tuple(row.logit for row in eager_rows)
        assert graph_state == eager_state
        assert executor.proposal_graph_contract()["replays"] >= 1
        assert executor.proposal_graph_contract()["last_status"] == "replay"
    finally:
        executor.close()
        free(hidden_buf, runtime=runtime)


def test_nextn_executor_batch_kv_only_reaches_packed_model_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = object.__new__(Qwen35GGUFNextNExecutor)
    executor.hidden_size = 8
    executor.vocab_size = 16
    executor.max_requests = 2
    executor.runtime = SimpleNamespace(memcpy_async=lambda *args, **kwargs: None)
    executor._request_slots = {10: 0, 20: 1}
    executor._batch_sessions = (
        SimpleNamespace(position=4),
        SimpleNamespace(position=7),
    )
    executor._token_host = np.zeros((2,), dtype=np.int64)
    executor._token_buf = SimpleNamespace(ptr=100, nbytes=16)
    for name, ptr in (
        ("_embedding_buf", 200),
        ("_enorm_buf", 300),
        ("_hnorm_buf", 400),
        ("_fusion_buf", 500),
        ("_fused_buf", 600),
        ("_final_hidden_buf", 700),
        ("_logits_buf", 800),
    ):
        setattr(executor, name, SimpleNamespace(ptr=ptr))

    def weight():
        return SimpleNamespace(
            allocation=lambda name="raw": SimpleNamespace(
                tensor=SimpleNamespace(ptr=900)
            )
        )

    executor.weights = SimpleNamespace(
        fallback=lambda name: weight(),
        nextn=lambda name: weight(),
        config=SimpleNamespace(rms_norm_eps=1e-6),
    )
    calls = []
    executor._batch_session = SimpleNamespace(
        step_hidden_batch_native=lambda *args, **kwargs: calls.append(kwargs),
    )
    monkeypatch.setattr(nextn_mod, "copy_host_to_device", lambda *a, **k: None)
    monkeypatch.setattr(nextn_mod, "launch_gguf_embedding", lambda *a, **k: None)
    monkeypatch.setattr(nextn_mod, "gguf_rmsnorm_bf16_f32_weight", lambda *a, **k: None)
    monkeypatch.setattr(nextn_mod, "launch_gguf_linear", lambda *a, **k: None)
    monkeypatch.setattr(
        Qwen35GGUFNextNExecutor,
        "_publish_batch_consumed_positions",
        lambda *args, **kwargs: None,
    )

    results = executor._run_step_batch_impl(
        (10, 20),
        (1, 2),
        (4, 7),
        Tensor.from_handle(1100, (2, 8), DType.BF16, Device("hip", 0)),
        score_output=False,
        kv_write_only=True,
    )

    assert all(isinstance(row, Qwen35GGUFNextNStateAdvance) for row in results)
    assert calls and calls[0]["score_output"] is False
    assert calls[0]["kv_write_only"] is True


def test_device_batch_top1_keeps_model_step_enqueue_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = object.__new__(Qwen35GGUFNextNExecutor)
    executor.hidden_size = 8
    executor.vocab_size = 16
    executor.max_requests = 2
    executor.runtime = SimpleNamespace(memcpy_async=lambda *args, **kwargs: None)
    executor._request_slots = {10: 0, 20: 1}
    executor._batch_sessions = (
        SimpleNamespace(position=4),
        SimpleNamespace(position=7),
    )
    executor._token_buf = SimpleNamespace(ptr=100)
    executor._embedding_buf = SimpleNamespace(ptr=200)
    executor._enorm_buf = SimpleNamespace(ptr=300)
    executor._hnorm_buf = SimpleNamespace(ptr=400)
    executor._fusion_buf = SimpleNamespace(ptr=500)
    executor._fused_buf = SimpleNamespace(ptr=600)
    executor._final_hidden_buf = SimpleNamespace(ptr=700)
    executor._logits_buf = SimpleNamespace(ptr=800)

    def weight():
        return SimpleNamespace(
            allocation=lambda name="raw": SimpleNamespace(
                tensor=SimpleNamespace(ptr=900)
            )
        )

    executor.weights = SimpleNamespace(
        fallback=lambda name: weight(),
        nextn=lambda name: weight(),
        config=SimpleNamespace(rms_norm_eps=1e-6),
    )
    calls = []
    executor._batch_session = SimpleNamespace(
        step_hidden_batch_native=lambda *args, **kwargs: calls.append(kwargs),
    )
    monkeypatch.setattr(nextn_mod, "launch_gguf_embedding", lambda *a, **k: None)
    monkeypatch.setattr(nextn_mod, "gguf_rmsnorm_bf16_f32_weight", lambda *a, **k: None)
    monkeypatch.setattr(nextn_mod, "launch_gguf_linear", lambda *a, **k: None)
    monkeypatch.setattr(
        Qwen35GGUFNextNExecutor,
        "_device_top1_rows",
        lambda self, rows, **kwargs: Tensor.from_handle(
            1000,
            (rows,),
            DType.INT32,
            Device("hip", 0),
        ),
    )
    monkeypatch.setattr(
        Qwen35GGUFNextNExecutor,
        "_publish_batch_consumed_positions",
        lambda *args, **kwargs: None,
    )

    tokens, hidden = executor._run_step_batch_device_top1(
        (10, 20),
        (4, 7),
        Tensor.from_handle(1100, (2, 8), DType.BF16, Device("hip", 0)),
    )

    assert tokens.shape == (2,)
    assert len(hidden) == 2
    assert calls and calls[0]["synchronize"] is False
    assert calls[0]["score_weight"] is None
    assert calls[0]["score_vocab_size"] is None


def test_device_batch_top1_routes_selected_head_and_maps_full_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = object.__new__(Qwen35GGUFNextNExecutor)
    executor.hidden_size = 8
    executor.vocab_size = 32
    executor.max_requests = 2
    executor._lm_head_top1_mapped = True
    executor.runtime = SimpleNamespace(memcpy_async=lambda *args, **kwargs: None)
    executor._request_slots = {10: 0, 20: 1}
    executor._batch_sessions = (
        SimpleNamespace(position=4),
        SimpleNamespace(position=7),
    )
    for name, ptr in (
        ("_token_buf", 100),
        ("_embedding_buf", 200),
        ("_enorm_buf", 300),
        ("_hnorm_buf", 400),
        ("_fusion_buf", 500),
        ("_fused_buf", 600),
        ("_final_hidden_buf", 700),
        ("_logits_buf", 800),
    ):
        setattr(executor, name, SimpleNamespace(ptr=ptr))

    def weight():
        return SimpleNamespace(
            allocation=lambda name="raw": SimpleNamespace(
                tensor=SimpleNamespace(ptr=900)
            )
        )

    hot_weight = weight()
    executor.weights = SimpleNamespace(
        hot_vocab=SimpleNamespace(
            lm_head=hot_weight,
            size=16,
            token_ids=SimpleNamespace(tensor=SimpleNamespace(ptr=0xA000)),
        ),
        fallback=lambda name: weight(),
        nextn=lambda name: weight(),
        config=SimpleNamespace(rms_norm_eps=1e-6),
    )
    step_calls = []
    top1_calls = []
    executor._batch_session = SimpleNamespace(
        step_hidden_batch_native=lambda *args, **kwargs: step_calls.append(kwargs),
    )
    monkeypatch.setattr(nextn_mod, "launch_gguf_embedding", lambda *a, **k: None)
    monkeypatch.setattr(nextn_mod, "gguf_rmsnorm_bf16_f32_weight", lambda *a, **k: None)
    monkeypatch.setattr(nextn_mod, "launch_gguf_linear", lambda *a, **k: None)
    monkeypatch.setattr(
        Qwen35GGUFNextNExecutor,
        "_device_top1_rows",
        lambda self, rows, **kwargs: (
            top1_calls.append((rows, kwargs))
            or Tensor.from_handle(1000, (rows,), DType.INT32, Device("hip", 0))
        ),
    )
    monkeypatch.setattr(
        Qwen35GGUFNextNExecutor,
        "_publish_batch_consumed_positions",
        lambda *args, **kwargs: None,
    )

    executor._run_step_batch_device_top1(
        (10, 20),
        (4, 7),
        Tensor.from_handle(1100, (2, 8), DType.BF16, Device("hip", 0)),
    )

    assert step_calls[0]["score_weight"] is hot_weight
    assert step_calls[0]["score_vocab_size"] == 16
    assert top1_calls == [
        (2, {"score_vocab_size": 16, "token_map_i32_ptr": 0xA000})
    ]
    assert executor.last_lm_head_path == "physical_batch_selected_q6_top1"


def test_device_top1_maps_compact_batch_ids_in_place(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = object.__new__(Qwen35GGUFNextNExecutor)
    executor.vocab_size = 1024
    executor.runtime = object()
    executor._lm_head_top1_libraries = {"q6_pack8": object()}
    owner = SimpleNamespace(
        _verify_lm_block_values=SimpleNamespace(ptr=0x1000),
        _verify_lm_block_indices_i32=SimpleNamespace(ptr=0x2000),
        _verify_lm_out_indices_i32=SimpleNamespace(ptr=0x3000),
        _verify_lm_out_values=SimpleNamespace(ptr=0x4000),
        _lm_head_threads=128,
        _lm_head_library=object(),
        _ensure_verify_lm_head_buffers=lambda rows, runtime: None,
    )
    executor._batch_session = owner
    argmax_calls = []
    map_calls = []
    monkeypatch.setattr(
        nextn_mod,
        "argmax_f32_rows_i32",
        lambda *args, **kwargs: argmax_calls.append((args, kwargs)),
    )
    monkeypatch.setattr(
        nextn_mod,
        "gguf_q6_k_pack8_top1_stage2_gather_mapped_f32",
        lambda *args, **kwargs: map_calls.append((args, kwargs)),
    )
    executor._logits_buf = SimpleNamespace(ptr=0x5000)

    tokens = executor._device_top1_rows(
        6,
        score_vocab_size=256,
        token_map_i32_ptr=0x6000,
    )

    assert tokens.ptr == 0x3000
    assert argmax_calls[0][0][-2:] == (6, 256)
    assert map_calls[0][0] == (
        0x4000,
        0x3000,
        0x6000,
        0x3000,
        None,
        6,
        1,
        256,
        1024,
    )
