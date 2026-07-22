from __future__ import annotations

from types import SimpleNamespace

import pytest

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.memory import DeviceBuffer
from hipengine.core.tensor import Tensor
from hipengine.runtime import laguna_gguf_runner as runner_module
from hipengine.runtime.laguna_gguf_runner import (
    LagunaDFlashVerifyResult,
    LagunaGGUFResidentSession,
    LagunaVerifierRowsResult,
)
from hipengine.speculative.laguna_dflash import (
    LagunaDFlashDraftResult,
    LagunaDFlashResidentCycle,
    LagunaDFlashResidentDrafter,
)


def test_laguna_dflash_accept_payload_covers_reject_partial_full_and_budget() -> None:
    validate = runner_module._validate_laguna_dflash_accept_payload
    tokens = (10, 20, 30)
    positions = (5, 6, 7)

    validate(
        tokens=tokens,
        positions=positions,
        target_top1=(99, 88, 77),
        remaining_decode=None,
        payload=(0, 0, 10, 5, 99, 0, 1),
    )
    validate(
        tokens=tokens,
        positions=positions,
        target_top1=(20, 99, 77),
        remaining_decode=None,
        payload=(1, 1, 20, 6, 99, 0, 2),
    )
    validate(
        tokens=tokens,
        positions=positions,
        target_top1=(20, 30, 77),
        remaining_decode=None,
        payload=(2, 2, 30, 7, 77, 1, 3),
    )
    validate(
        tokens=tokens,
        positions=positions,
        target_top1=(20, 30, 77),
        remaining_decode=1,
        payload=(1, 1, 20, 6, -1, 0, 2),
    )
    with pytest.raises(RuntimeError, match="GPU accept summary"):
        validate(
            tokens=tokens,
            positions=positions,
            target_top1=(20, 99, 77),
            remaining_decode=None,
            payload=(2, 2, 30, 7, 77, 1, 3),
        )


class _FakeKV:
    def __init__(self) -> None:
        self.pending_positions = (511, 512, 513, 514)
        self.calls: list[tuple] = []
        self.position = 510

    def discard_rows(self) -> None:
        self.calls.append(("discard", self.pending_positions))
        self.pending_positions = ()

    def prepare_rows(self, positions) -> None:
        parsed = tuple(positions)
        self.calls.append(("prepare", parsed))
        self.pending_positions = parsed

    def append_rows(self, layer, key, value, rows, **kwargs) -> None:
        del kwargs
        self.calls.append(("append", layer, key, value, rows))

    def commit_rows(self) -> None:
        self.calls.append(("commit", self.pending_positions))
        self.position = self.pending_positions[-1]
        self.pending_positions = ()


class _FakeStaging:
    @staticmethod
    def key_ptr(layer: int) -> int:
        return 0x1000 + layer * 0x100

    @staticmethod
    def value_ptr(layer: int) -> int:
        return 0x2000 + layer * 0x100


def test_laguna_staged_verifier_commits_only_prefix_across_swa_wrap() -> None:
    session = object.__new__(LagunaGGUFResidentSession)
    session._closed = False
    session._staged_verifier_tokens = (1, 2, 3, 4)
    session.kv_cache = _FakeKV()
    session.verifier_scratch = _FakeStaging()
    session.weights = SimpleNamespace(config=SimpleNamespace(block_count=3))
    session.libraries = SimpleNamespace(kv_attention=object())
    session.position = 510

    session._commit_staged_verifier_rows(2, stream=7)

    assert session.position == 512
    assert session._staged_verifier_tokens is None
    assert session.kv_cache.calls == [
        ("discard", (511, 512, 513, 514)),
        ("prepare", (511, 512)),
        ("append", 0, 0x1000, 0x2000, 2),
        ("append", 1, 0x1100, 0x2100, 2),
        ("append", 2, 0x1200, 0x2200, 2),
        ("commit", (511, 512)),
    ]


def test_laguna_staged_verifier_cancel_leaves_committed_cursor_unchanged() -> None:
    session = object.__new__(LagunaGGUFResidentSession)
    session._closed = False
    session._staged_verifier_tokens = (1, 2, 3)
    session.kv_cache = _FakeKV()
    session.position = 510

    session._discard_staged_verifier_rows()

    assert session.position == 510
    assert session._staged_verifier_tokens is None
    assert session.kv_cache.calls == [("discard", (511, 512, 513, 514))]


def _draft_result(candidates: tuple[int, ...]) -> LagunaDFlashDraftResult:
    device = Device("hip", 0)
    rows = len(candidates)
    return LagunaDFlashDraftResult(
        candidate_token_ids=candidates,
        candidate_values=tuple(float(index) for index in range(rows)),
        topk_token_ids=tuple((token,) for token in candidates),
        topk_values=tuple((float(index),) for index in range(rows)),
        query_rows=rows + 1,
        candidate_budget=rows,
        logits=Tensor.from_handle(0x5000, (rows, 100), DType.FP32, device),
        final_hidden=Tensor.from_handle(0x6000, (rows + 1, 8), DType.BF16, device),
    )


class _FakeCapture:
    def __init__(self, rows: int, base: int) -> None:
        self.rows = rows
        self.targets = SimpleNamespace(rows=rows)
        self.buffers = tuple(SimpleNamespace(ptr=base + index * 0x100) for index in range(2))
        self.freed = False

    def prefix_tensors(self, rows: int):
        return tuple((rows, index) for index in range(2))

    def prefix_targets(self, rows: int):
        return SimpleNamespace(rows=rows)

    def free(self) -> None:
        self.freed = True


class _FakeTarget:
    def __init__(self) -> None:
        self.position = 4
        self.closed = False
        self.verify_calls: list[tuple] = []

    def verifier_address_signature(self) -> dict[str, int]:
        return {"target.rows": 0x100}

    def verify_dflash_chain(self, root, candidates, **kwargs) -> LagunaDFlashVerifyResult:
        self.verify_calls.append((root, tuple(candidates), kwargs))
        self.position = 6
        rows = LagunaVerifierRowsResult(
            start_position=5,
            input_token_ids=(10, 20, 30),
            logits=DeviceBuffer(0x7000, 3 * 100 * 4),
            final_hidden=DeviceBuffer(0x8000, 3 * 8 * 2),
            post_layer_hidden=DeviceBuffer(0x9000, 3 * 8 * 2),
            logits_row_stride=100,
        )
        return LagunaDFlashVerifyResult(
            rows_result=rows,
            target_top1_ids=(20, 99, 77),
            target_top1_values=(3.0, 2.0, 1.0),
            accepted_draft_count=1,
            accepted_token_ids=(20,),
            commit_row=1,
            commit_token_id=20,
            commit_position=6,
            next_token_id=99,
            full_accept=False,
            committed_input_ids=(10, 20),
            visible_output_ids=(20, 99),
            packed_payload=(1, 1, 20, 6, 99, 0, 2),
        )

    def close(self) -> None:
        self.closed = True


class _FakeDrafter:
    def __init__(self, target: _FakeTarget) -> None:
        self.target = target
        self._closed = False
        self.candidate_budget = 2
        self.max_append_rows = 3
        self.committed_context_tokens = 5
        self.allocations = 0
        self.append_calls: list[tuple] = []

    def allocate_captures(self, *, rows: int) -> _FakeCapture:
        self.allocations += 1
        return _FakeCapture(rows, 0xA000 + self.allocations * 0x1000)

    def append_target_hidden(self, captures, *, positions, stream=0) -> None:
        parsed = tuple(positions)
        self.append_calls.append((captures, parsed, stream))
        self.committed_context_tokens = parsed[-1] + 1

    def close(self) -> None:
        self._closed = True


def test_laguna_dflash_drafter_reset_retains_owner_and_resets_kv() -> None:
    class ResetKV:
        def __init__(self) -> None:
            self.calls = 0

        def reset(self) -> None:
            self.calls += 1

    drafter = object.__new__(LagunaDFlashResidentDrafter)
    drafter._closed = False
    drafter.kv_cache = ResetKV()

    drafter.reset_state()

    assert drafter.kv_cache.calls == 1
    assert drafter._closed is False


def test_laguna_resident_cycle_requires_b_plus_one_append_capacity() -> None:
    target = _FakeTarget()
    drafter = _FakeDrafter(target)
    drafter.max_append_rows = drafter.candidate_budget

    with pytest.raises(ValueError, match=r"B\+1"):
        LagunaDFlashResidentCycle(target, drafter)


def test_laguna_resident_cycle_normalizes_chain_and_commits_capture_prefix() -> None:
    target = _FakeTarget()
    drafter = _FakeDrafter(target)
    cycle = LagunaDFlashResidentCycle(target, drafter, request_id=17)
    proposal = _draft_result((20, 30))

    with pytest.raises(TypeError, match="remaining_decode"):
        cycle.run_cycle(10, proposal=proposal, remaining_decode=True)
    result = cycle.run_cycle(10, proposal=proposal)

    assert result.target_batch.request_ids == (17,)
    assert result.target_batch.tokens == (10, 20, 30)
    assert result.target_batch.positions == (5, 6, 7)
    assert result.target_batch.parent_rows == (-1, 0, 1)
    assert result.target_batch.draft_depths == (0, 1, 2)
    assert result.accept_summary.accepted_counts == (1,)
    assert result.accept_summary.commit_rows == (1,)
    assert result.visible_output_ids == (20, 99)
    assert result.verifier_addresses_stable
    assert result.proposal_seconds == 0.0
    assert result.target_verify_seconds >= 0.0
    assert result.draft_commit_enqueue_seconds >= 0.0
    assert result.cycle_host_seconds >= result.target_verify_seconds
    captures, positions, stream = drafter.append_calls[-1]
    assert captures == ((2, 0), (2, 1))
    assert positions == (5, 6)
    assert stream == 0
    assert target.position == 6
    assert drafter.committed_context_tokens == 7

    cycle.close()
    assert cycle.closed


def test_laguna_resident_cycle_truncates_at_accepted_eos_without_bonus() -> None:
    class StopTarget(_FakeTarget):
        def verify_dflash_chain(self, root, candidates, **kwargs) -> LagunaDFlashVerifyResult:
            assert tuple(candidates) == (20,)
            assert kwargs["remaining_decode"] == 1
            assert kwargs["captures"].rows == 2
            self.position = 6
            rows = LagunaVerifierRowsResult(
                start_position=5,
                input_token_ids=(10, 20),
                logits=DeviceBuffer(0x7000, 2 * 100 * 4),
                final_hidden=DeviceBuffer(0x8000, 2 * 8 * 2),
                post_layer_hidden=DeviceBuffer(0x9000, 2 * 8 * 2),
                logits_row_stride=100,
            )
            return LagunaDFlashVerifyResult(
                rows_result=rows,
                target_top1_ids=(20, 99),
                target_top1_values=(3.0, 2.0),
                accepted_draft_count=1,
                accepted_token_ids=(20,),
                commit_row=1,
                commit_token_id=20,
                commit_position=6,
                next_token_id=None,
                full_accept=True,
                committed_input_ids=(10, 20),
                visible_output_ids=(20,),
                packed_payload=(1, 1, 20, 6, -1, 1, 2),
            )

    target = StopTarget()
    drafter = _FakeDrafter(target)
    cycle = LagunaDFlashResidentCycle(target, drafter)

    result = cycle.run_cycle(
        10,
        proposal=_draft_result((20, 30)),
        remaining_decode=8,
        stop_token_ids=(20,),
    )

    assert result.target_batch.tokens == (10, 20)
    assert result.visible_output_ids == (20,)
    assert result.target_result.next_token_id is None
    assert drafter.append_calls[-1][1] == (5, 6)
    cycle.close()
