"""CPU lifecycle coverage for the PARO prompt KV-only priming route.

The staged PARO MTP2 provider primes from the target's own prompt: every prompt
row except the true last one appends K/V only (``_prompt_kv_only``), and the
final row runs one full step that produces the live hidden and the first
candidate.  These tests drive that contract through the real
``Qwen35ParoResidentModelRunner`` prefill path and the real
``Qwen35ParoMTP2Adapter``, with a session double that mirrors
``prefill_native_packed``'s packed token/position buffer and
``final_hidden_row_sink`` contract, and a proposer double that mirrors
``NativeMtpChainProposer``'s prompt-phase cursor behaviour.

What is checked here is the owner's own lifecycle: which row may carry the seed,
that a prompt row that fails never advances the provider cursor, that no cycle
scratch or speculative capability exists before priming completes, and that the
prompt capture buffer plus provider state are reclaimed by the owner that owns
them (``discard``/``close``) rather than by the failing call.

Depth support, C>1, and GPU behaviour are out of scope.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import hipengine.generation.qwen35_paro as paro_generation_module
from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.tensor import Tensor
from hipengine.dispatch.batch import WorkItem, WorkKind
from hipengine.generation.deadline import GenerationCancelled, GenerationCancellationToken
from hipengine.generation.qwen35_paro import (
    Qwen35ParoResidentModelRunner,
    _ParoResidentLoopRow,
)
from hipengine.generation.qwen35_paro_mtp2 import (
    Qwen35ParoMTP2Adapter,
    _ParoMTP2RequestState,
)
from hipengine.generation.registry import GenerationRequest
from hipengine.runtime.qwen35_paro_runner import Qwen35ParoAutoregressiveStepResult
from hipengine.speculative import SpeculativeRequestSemantics

_REQUEST_ID = 7
_SEED_TOKEN = 99


class _PromptProposer:
    """Prompt-phase stand-in for ``NativeMtpChainProposer``.

    Only the surface the prompt route touches is modelled, but the state it
    leaves behind matches the real proposer: a KV-only append ends with an
    unusable ``current`` (``token == -1``, exactly what ``advance`` writes for
    ``_prompt_kv_only``), and a full step ends with a consumable candidate.
    """

    closed = False

    def __init__(
        self,
        *,
        fail_append_at: int | None = None,
        fail_full_step: bool = False,
    ) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.resets = 0
        self.close_calls = 0
        self.cache_len = 0
        self.position = -1
        self.current = SimpleNamespace(token=0)
        self.max_positions = 8192
        self.max_mtp_tokens = 256
        self.scoring_head: SimpleNamespace | None = None
        self._fail_append_at = fail_append_at
        self._fail_full_step = fail_full_step
        self._append_count = 0

    def append_prompt_kv_with_target_hidden(self, **kwargs) -> None:
        if self._fail_append_at is not None and self._append_count == self._fail_append_at:
            self._append_count += 1
            raise RuntimeError("MTP prompt KV append failed")
        self._append_count += 1
        self.calls.append(("kv", kwargs))
        self.cache_len += 1
        self.position = int(kwargs["position"])
        self.current = SimpleNamespace(token=-1)

    def advance_with_target_hidden(self, **kwargs):
        if self._fail_full_step:
            raise RuntimeError("MTP final prompt step failed")
        self.calls.append(("full", kwargs))
        self.cache_len += 1
        self.position = int(kwargs["position"])
        self.current = SimpleNamespace(token=int(kwargs["input_token"]) + 1)
        return self.current

    def reset(self) -> None:
        self.resets += 1
        self.cache_len = 0
        self.position = -1
        self.current = SimpleNamespace(token=0)

    def close(self) -> None:
        self.close_calls += 1
        self.closed = True

    def save_state(self, slot: int):
        return ("snapshot", slot)

    def device_candidate_token_ids(self) -> Tensor:
        return Tensor.from_handle(0x7000, (1,), DType.INT32, Device("hip", 0))

    @property
    def kinds(self) -> list[str]:
        return [kind for kind, _ in self.calls]


class _PackedPromptSession:
    """Resident-session double for the packed prompt slab contract.

    ``prefill_native_packed`` is reproduced as the real one behaves for a single
    request: it walks ``slab.row_to_request``/``slab.positions`` in slab order,
    hands the sink one hidden row per prompt row with the absolute prompt index,
    and passes a seed token only for the last row of the slab when sampling.
    ``position_transform`` and ``seed_override`` exist so a test can inject the
    two contract violations a real session could produce.
    """

    block_size = 256
    hidden_nbytes = 16
    max_sequence_length = 4096
    tokenizer = None
    closed = False

    def __init__(
        self,
        *,
        position_transform=None,
        seed_override=None,
        fail_after_rows: int | None = None,
        serial_only: bool = False,
    ) -> None:
        self.config = SimpleNamespace(hidden_size=8)
        self.runtime = object()
        self.slabs: list = []
        self.serial_steps: list[tuple[int, int, bool, int]] = []
        self.hidden_writes: list[tuple[int, int, int]] = []
        self.scratch_reservations: list[tuple[int, str, int]] = []
        self.reset_slot_calls: list[tuple[int, ...]] = []
        self.sampler_row_configurations: list[tuple[object, object]] = []
        self._position_transform = position_transform or (lambda prompt_index: prompt_index)
        self._seed_override = seed_override
        self._fail_after_rows = fail_after_rows
        self._serial_only = serial_only

    def prefill_native_packed(self, slab, *, sample: bool = True, final_hidden_row_sink=None):
        if self._serial_only:
            raise NotImplementedError("packed prompt prefill is unavailable")
        self.slabs.append(slab)
        results = (self._step_result(),) if sample else (None,)
        final_rows = tuple(
            int(slab.cu_seqlens_q[index + 1]) - 1 for index in range(slab.request_count)
        )
        seed_tokens = {
            int(final_row): (None if result is None else int(result.token_id))
            for final_row, result in zip(final_rows, results, strict=True)
        }
        for row_index, (request_id, prompt_index) in enumerate(
            zip(slab.row_to_request, slab.positions, strict=True)
        ):
            if self._fail_after_rows is not None and row_index >= self._fail_after_rows:
                raise RuntimeError("packed prompt slab failed mid-slab")
            hidden = Tensor.from_handle(
                0x9000 + row_index * self.hidden_nbytes,
                (1, self.config.hidden_size),
                DType.FP16,
                Device("hip", 0),
            )
            seed = seed_tokens.get(row_index)
            if self._seed_override is not None:
                seed = self._seed_override(row_index, final_rows[0], seed)
            final_hidden_row_sink(
                int(request_id),
                int(self._position_transform(int(prompt_index))),
                hidden,
                seed,
            )
        return results

    @staticmethod
    def _step_result() -> Qwen35ParoAutoregressiveStepResult:
        return Qwen35ParoAutoregressiveStepResult(
            token_id=_SEED_TOKEN,
            token_text="seed",
            logit=float("nan"),
        )

    def step_with_hidden_taps(
        self,
        token_id,
        *,
        position,
        capture_layer_ids,
        capture_hidden_concat,
        capture_row,
        sample,
        capture_final_hidden_bf16,
    ):
        assert capture_layer_ids == ()
        assert capture_row == 0
        self.serial_steps.append(
            (
                int(token_id),
                int(position),
                bool(sample),
                int(capture_final_hidden_bf16.ptr),
            )
        )
        return self._step_result() if sample else None

    def _write_final_normalized_hidden_bf16(self, hidden, *, destination_ptr, stream=0) -> None:
        self.hidden_writes.append((int(hidden.ptr), int(destination_ptr), int(stream)))

    def prepare_specdec2_verify_scratch(self, *, rows, chain_attn_mode, max_context_tokens) -> None:
        self.scratch_reservations.append((int(rows), str(chain_attn_mode), int(max_context_tokens)))

    def reset_slots(self, slots) -> None:
        self.reset_slot_calls.append(tuple(int(slot) for slot in slots))

    def configure_host_sampler_rows(self, rows, plan) -> None:
        self.sampler_row_configurations.append((rows, plan))

    def close(self) -> None:
        self.closed = True


def _row(
    request_id: int,
    prompt: tuple[int, ...],
    *,
    max_tokens: int = 4,
    model_slot: int | None = 0,
    candidate_budget: int = 1,
    cancellation_token: GenerationCancellationToken | None = None,
) -> _ParoResidentLoopRow:
    request = GenerationRequest(
        prompts=(prompt,),
        max_tokens=max_tokens,
        temperature=0.0,
        top_p=1.0,
        ignore_eos=True,
        cancellation_token=cancellation_token,
    )
    return _ParoResidentLoopRow(
        request_id=request_id,
        batch_id=0,
        row_index=0,
        request=request,
        sampling_request=request,
        prompt_ids=tuple(prompt),
        sampler_plan=None,
        native_greedy=True,
        submitted_at=0.0,
        model_slot=model_slot,
        mtp2_candidate_budget=candidate_budget,
    )


class _Harness:
    def __init__(self, runner, adapter, session, proposer, row, freed, allocated):
        self.runner = runner
        self.adapter = adapter
        self.session = session
        self.proposer = proposer
        self.row = row
        self.freed = freed
        self.allocated = allocated

    def state(self, request_id: int = _REQUEST_ID) -> _ParoMTP2RequestState:
        return self.adapter._states[request_id]

    def prefill(self, chunk: tuple[int, ...], request_id: int = _REQUEST_ID) -> None:
        self.runner.prefill_batch(
            WorkItem(
                kind=WorkKind.PREFILL,
                request_ids=(request_id,),
                row_to_request=(request_id,),
                token_rows=(tuple(chunk),),
            ),
            commit=True,
        )


def _harness(
    monkeypatch,
    *,
    prompt: tuple[int, ...] = (10, 11, 12, 13, 14),
    proposer: _PromptProposer | None = None,
    session: _PackedPromptSession | None = None,
    max_tokens: int = 4,
    cancellation_token: GenerationCancellationToken | None = None,
) -> _Harness:
    allocated: list = []
    freed: list = []

    def _malloc(nbytes, runtime=None):
        buffer = SimpleNamespace(ptr=0x8000 + 0x100 * len(allocated), nbytes=int(nbytes))
        allocated.append(buffer)
        return buffer

    def _free(buffer, runtime=None):
        freed.append(buffer)

    monkeypatch.setattr(paro_generation_module, "malloc", _malloc)
    monkeypatch.setattr(paro_generation_module, "free", _free)

    session = session if session is not None else _PackedPromptSession()
    proposer = proposer if proposer is not None else _PromptProposer()
    proposer.scoring_head = SimpleNamespace(owner=session)
    row = _row(_REQUEST_ID, prompt, max_tokens=max_tokens, cancellation_token=cancellation_token)

    runner = object.__new__(Qwen35ParoResidentModelRunner)
    runner._session = session
    runner._rows = {_REQUEST_ID: row}
    runner._outputs = {}
    runner._completed_metadata = {}
    runner._available_model_slots = [1]
    runner._route_counts = {
        "admissions": 0,
        "prefill_chunks": 0,
        "native_group_calls": 0,
        "native_rows": 0,
        "serial_row_calls": 0,
        "scheduler_compactions": 0,
        "slot_resets": 0,
        "reclaims": 0,
    }
    runner._fallback_reasons = {"packed_prefill_unavailable": 0}
    runner._closed = False
    runner.generator = SimpleNamespace(
        backend="hip_gfx1100",
        execution_profile="production",
        _resident_model_runner=None,
    )

    adapter = Qwen35ParoMTP2Adapter(runner)
    adapter._intents[_REQUEST_ID] = 1
    adapter._states[_REQUEST_ID] = _ParoMTP2RequestState(_REQUEST_ID, proposer)
    runner._mtp2_adapter = adapter
    runner._mtp2_adapter_resolved = True
    runner._resolved_mtp2_adapter = lambda: adapter

    return _Harness(runner, adapter, session, proposer, row, freed, allocated)


# ---------------------------------------------------------------------------
# Ragged chunks: only the true final row seeds, and only once
# ---------------------------------------------------------------------------


def test_ragged_chunks_seed_only_the_true_final_row_and_prime_once(monkeypatch) -> None:
    prompt = (10, 11, 12, 13, 14)
    harness = _harness(monkeypatch, prompt=prompt)

    # Two ragged chunks. Neither chunk boundary is a provider boundary: the
    # first chunk is entirely KV-only even though it is the chunk's last row.
    harness.prefill((10, 11))
    assert harness.state().prompt_rows_consumed == 2
    assert harness.proposer.kinds == ["kv", "kv"]
    assert harness.session.scratch_reservations == []
    assert harness.freed == []

    harness.prefill((12, 13, 14))

    # The packed slab each chunk builds is chunk-local in tokens and absolute in
    # position, which is what makes the sink's prompt_index a valid provider
    # cursor comparison across chunk boundaries.
    first, second = harness.session.slabs
    assert (first.token_ids, first.positions, first.cu_seqlens_q) == ((10, 11), (0, 1), (0, 2))
    assert (second.token_ids, second.positions, second.cu_seqlens_q) == (
        (12, 13, 14),
        (2, 3, 4),
        (0, 3),
    )
    assert second.row_to_request == (_REQUEST_ID,) * 3
    assert second.slot_ids == (0,)

    assert harness.proposer.kinds == ["kv", "kv", "kv", "kv", "full"]
    # Each prompt row primes the *next* prompt token, and only the true final
    # row carries the sampled root instead.
    assert [kwargs["input_token"] for _, kwargs in harness.proposer.calls] == [
        11,
        12,
        13,
        14,
        _SEED_TOKEN,
    ]
    assert [kwargs["position"] for _, kwargs in harness.proposer.calls] == [1, 2, 3, 4, 5]
    full = harness.proposer.calls[-1][1]
    assert full["need_result"] is True
    assert full["read_token_id"] is False
    assert harness.proposer.cache_len == len(prompt)
    assert harness.proposer.position == len(prompt)
    assert harness.state().prompt_rows_consumed == len(prompt)

    # Exactly one full final step produced the recorded seed token.
    assert [step.token_id for step in harness.row.generated_steps] == [_SEED_TOKEN]
    # Cycle scratch is reserved only after priming completes, sized from the
    # prompt plus the request's own decode budget.
    assert harness.session.scratch_reservations == [(2, "decode_batched", len(prompt) + 4)]
    # The prompt capture buffer is freed by the owner that allocated it.
    assert harness.allocated and harness.freed == harness.allocated
    assert harness.row.mtp2_prompt_hidden_buffer is None
    assert harness.row.mtp2_prompt_prime_ms >= 0.0
    assert harness.row.native_prefill is True


# ---------------------------------------------------------------------------
# Ordering and seed placement
# ---------------------------------------------------------------------------


def test_out_of_order_prompt_rows_are_rejected_without_advancing(monkeypatch) -> None:
    prompt = (10, 11, 12, 13)
    harness = _harness(monkeypatch, prompt=prompt)
    harness.prefill((10, 11))
    assert harness.state().prompt_rows_consumed == 2

    # A session that re-reports prompt indices from zero for the second chunk is
    # the realistic way an out-of-order row reaches the adapter.
    harness.session._position_transform = lambda prompt_index: 0
    with pytest.raises(RuntimeError, match="must arrive in order"):
        harness.prefill((12, 13))

    assert harness.state().prompt_rows_consumed == 2
    assert harness.proposer.kinds == ["kv", "kv"]
    assert harness.freed == []


def test_out_of_order_and_repeated_indices_are_rejected_directly(monkeypatch) -> None:
    harness = _harness(monkeypatch, prompt=(10, 11, 12))

    with pytest.raises(RuntimeError, match="must arrive in order"):
        harness.adapter.consume_prompt_row(
            _REQUEST_ID, prompt_index=1, target_hidden_ptr=0x1000, seed_token=None
        )
    assert harness.state().prompt_rows_consumed == 0
    assert harness.proposer.calls == []

    harness.adapter.consume_prompt_row(
        _REQUEST_ID, prompt_index=0, target_hidden_ptr=0x1000, seed_token=None
    )
    with pytest.raises(RuntimeError, match="must arrive in order"):
        harness.adapter.consume_prompt_row(
            _REQUEST_ID, prompt_index=0, target_hidden_ptr=0x1000, seed_token=None
        )
    assert harness.state().prompt_rows_consumed == 1
    assert harness.proposer.kinds == ["kv"]


def test_a_seed_on_a_non_final_row_is_rejected_before_any_append(monkeypatch) -> None:
    prompt = (10, 11, 12)
    harness = _harness(
        monkeypatch,
        prompt=prompt,
        session=_PackedPromptSession(
            seed_override=lambda row_index, final_row, faithful: (
                _SEED_TOKEN if row_index == 0 else faithful
            )
        ),
    )

    with pytest.raises(RuntimeError, match="final prompt row must carry target root"):
        harness.prefill(prompt)

    # The misplaced seed is rejected on arrival, before that row's KV append.
    assert harness.state().prompt_rows_consumed == 0
    assert harness.proposer.calls == []
    assert harness.session.scratch_reservations == []
    assert harness.row.generated_steps == []


def test_a_missing_seed_on_the_final_row_leaves_the_prime_incomplete(monkeypatch) -> None:
    prompt = (10, 11, 12)
    harness = _harness(
        monkeypatch,
        prompt=prompt,
        session=_PackedPromptSession(seed_override=lambda row_index, final_row, faithful: None),
    )

    with pytest.raises(RuntimeError, match="final prompt row must carry target root"):
        harness.prefill(prompt)

    # The KV-only rows are legitimate and stay appended; the seed row did not
    # run a full step, so priming can never complete for this attempt.
    assert harness.state().prompt_rows_consumed == len(prompt) - 1
    assert harness.proposer.kinds == ["kv", "kv"]
    assert harness.proposer.current.token == -1
    assert harness.session.scratch_reservations == []
    assert harness.row.generated_steps == []
    with pytest.raises(RuntimeError, match="streaming prompt priming is incomplete"):
        harness.adapter.observe_prefill_result(_REQUEST_ID)


# ---------------------------------------------------------------------------
# No capability and no cycle scratch mid-prime
# ---------------------------------------------------------------------------


def test_no_capability_or_verify_scratch_while_priming_is_incomplete(monkeypatch) -> None:
    prompt = (10, 11, 12, 13)
    harness = _harness(monkeypatch, prompt=prompt)
    semantics = (
        SpeculativeRequestSemantics(
            request_id=_REQUEST_ID,
            sampling_mode="greedy",
            mode="verify_chain",
            context_tokens=2,
            remaining_decode=4,
        ),
    )

    harness.prefill((10, 11))
    state = harness.state()
    assert state.prompt_rows_consumed == 2
    # A KV-only row leaves no consumable prediction behind.
    assert harness.proposer.current.token == -1
    # So the row cannot be advertised as speculative-capable: no plan, and
    # therefore no candidate, can be formed from a half-primed provider.
    assert harness.adapter.capability(semantics) is None
    # And the adapter refuses to move on to the cycle phase.
    with pytest.raises(RuntimeError, match="streaming prompt priming is incomplete"):
        harness.adapter.observe_prefill_result(_REQUEST_ID)
    assert harness.session.scratch_reservations == []

    # The root is published by the first decode, not by prefill, so even a
    # completed prime is not yet a capability.
    harness.session._position_transform = lambda prompt_index: prompt_index
    harness.prefill((12, 13))
    assert state.prompt_rows_consumed == len(prompt)
    assert harness.adapter.capability(semantics) is None
    harness.row.first_token_emitted = True
    assert harness.adapter.capability(semantics) is not None


# ---------------------------------------------------------------------------
# Cancellation / drop after some KV-only rows
# ---------------------------------------------------------------------------


def test_cancelled_prompt_drop_reclaims_capture_and_recycles_provider(monkeypatch) -> None:
    prompt = (10, 11, 12, 13)
    token = GenerationCancellationToken()
    harness = _harness(monkeypatch, prompt=prompt, cancellation_token=token)
    harness.prefill((10, 11))
    capture = harness.row.mtp2_prompt_hidden_buffer
    assert capture is not None and harness.freed == []

    token.cancel()

    # The real cancellation surface the chunked prefill observes between
    # chunks: the request's own token.
    with pytest.raises(GenerationCancelled):
        harness.prefill((12, 13))
    assert harness.state().prompt_rows_consumed == 2
    assert harness.proposer.kinds == ["kv", "kv"]

    # Reclamation is the owner's job, and it is the only thing that frees the
    # prompt capture and the provider.
    harness.runner.discard((_REQUEST_ID,))

    assert harness.freed == [capture]
    assert harness.row.mtp2_prompt_hidden_buffer is None
    assert _REQUEST_ID not in harness.adapter._states
    assert _REQUEST_ID not in harness.adapter._intents
    assert harness.proposer.close_calls == 0
    assert harness.proposer.resets == 1
    assert harness.adapter._proposer_pool == [harness.proposer]
    assert harness.row.model_slot is None
    assert harness.session.reset_slot_calls == [(0,)]
    assert harness.runner._available_model_slots == [0, 1]

    # A new request reuses the recycled provider from a reset cursor.
    new_row = _row(8, prompt)
    harness.runner._rows[8] = new_row
    harness.adapter._intents[8] = 1
    harness.adapter.begin_prompt(8)
    assert harness.adapter._proposer_reuses == 1
    assert harness.adapter._proposer_builds == 0
    assert harness.adapter._states[8].proposer is harness.proposer
    assert harness.adapter._states[8].prompt_rows_consumed == 0
    assert harness.proposer.cache_len == 0
    assert harness.proposer.position == -1
    assert harness.proposer.resets == 2

    harness.prefill(prompt, request_id=8)
    assert harness.adapter._states[8].prompt_rows_consumed == len(prompt)
    # The two rows from the cancelled request are still in the call log; the
    # reused provider primed this request's four rows from a reset cursor.
    assert harness.proposer.kinds == [
        "kv",
        "kv",
        "kv",
        "kv",
        "kv",
        "full",
    ]
    assert [step.token_id for step in new_row.generated_steps] == [_SEED_TOKEN]


def test_owner_close_closes_pooled_providers_and_frees_captures(monkeypatch) -> None:
    harness = _harness(monkeypatch, prompt=(10, 11, 12))
    harness.prefill((10,))
    capture = harness.row.mtp2_prompt_hidden_buffer
    harness.runner.discard((_REQUEST_ID,))
    assert harness.freed == [capture]

    # A second request still priming when the owner closes: its capture is freed
    # and its provider is closed, including the pooled one from the first.
    harness.runner._rows[8] = _row(8, (20, 21, 22))
    harness.runner._available_model_slots.pop(0)  # admission takes slot 0
    harness.adapter._intents[8] = 1
    harness.adapter.begin_prompt(8)
    pooled = harness.proposer
    assert harness.adapter._states[8].proposer is pooled
    harness.runner._rows[8].mtp2_prompt_hidden_buffer = SimpleNamespace(ptr=0xDEAD, nbytes=16)

    harness.runner.close()

    assert harness.freed[-1].ptr == 0xDEAD
    assert harness.runner._rows == {}
    assert harness.adapter._states == {}
    assert harness.adapter._proposer_pool == []
    assert pooled.close_calls == 1
    assert harness.runner._session is None


# ---------------------------------------------------------------------------
# Failures: the cursor must not advance, and cleanup is discard-only
# ---------------------------------------------------------------------------


def test_kv_append_failure_does_not_advance_the_prompt_cursor(monkeypatch) -> None:
    prompt = (10, 11, 12, 13)
    proposer = _PromptProposer(fail_append_at=1)
    harness = _harness(monkeypatch, prompt=prompt, proposer=proposer)

    with pytest.raises(RuntimeError, match="MTP prompt KV append failed"):
        harness.prefill(prompt)

    # The row that failed is not counted, and neither is the provider advanced.
    assert harness.state().prompt_rows_consumed == 1, (
        f"the failed prompt row advanced the cursor to {harness.state().prompt_rows_consumed}"
    )
    assert proposer.kinds == ["kv"]
    assert proposer.cache_len == 1
    assert harness.session.scratch_reservations == []
    assert harness.row.generated_steps == []
    # The capture buffer outlives the failure: only the owner frees it.
    capture = harness.row.mtp2_prompt_hidden_buffer
    assert capture is not None and harness.freed == []

    # Retrying the same request is not a recovery path: prefill bookkeeping has
    # already consumed the chunk, so the retry is rejected as chunk drift.
    with pytest.raises(RuntimeError, match="prefill chunk drift"):
        harness.prefill(prompt)

    harness.runner.discard((_REQUEST_ID,))
    assert harness.freed == [capture]
    assert harness.row.mtp2_prompt_hidden_buffer is None
    assert harness.adapter._states == {}
    assert proposer.resets == 1 and proposer.close_calls == 0

    # A new request over the same prompt starts from an empty provider.
    fresh = _PromptProposer()
    harness.runner._rows[9] = _row(9, prompt)
    harness.adapter._intents[9] = 1
    harness.adapter.begin_prompt(9)
    assert harness.adapter._states[9].prompt_rows_consumed == 0
    assert harness.adapter._states[9].proposer is proposer
    assert proposer.cache_len == 0
    harness.adapter._states[9] = _ParoMTP2RequestState(9, fresh)
    harness.prefill(prompt, request_id=9)
    assert fresh.kinds == ["kv", "kv", "kv", "full"]
    assert [kwargs["input_token"] for _, kwargs in fresh.calls] == [11, 12, 13, _SEED_TOKEN]
    assert harness.adapter._states[9].prompt_rows_consumed == len(prompt)


def test_final_full_step_failure_skips_verify_reservation_and_recovers_by_discard(
    monkeypatch,
) -> None:
    prompt = (10, 11, 12)
    proposer = _PromptProposer(fail_full_step=True)
    harness = _harness(monkeypatch, prompt=prompt, proposer=proposer)

    with pytest.raises(RuntimeError, match="MTP final prompt step failed"):
        harness.prefill(prompt)

    # Every KV-only row landed; the seed row did not, so priming is incomplete.
    assert harness.state().prompt_rows_consumed == len(prompt) - 1, (
        f"cursor {harness.state().prompt_rows_consumed} should stop at the last KV-only row"
    )
    assert proposer.kinds == ["kv", "kv"]
    assert proposer.cache_len == len(prompt) - 1
    assert proposer.current.token == -1
    # No cycle scratch was reserved and no token was published.
    assert harness.session.scratch_reservations == []
    assert harness.row.generated_steps == []
    capture = harness.row.mtp2_prompt_hidden_buffer
    assert capture is not None and harness.freed == []
    with pytest.raises(RuntimeError, match="streaming prompt priming is incomplete"):
        harness.adapter.observe_prefill_result(_REQUEST_ID)

    harness.runner.discard((_REQUEST_ID,))
    assert harness.freed == [capture]
    assert harness.adapter._states == {}
    assert harness.adapter._proposer_pool == [proposer]

    # A brand-new request initializes cleanly on the recycled provider once the
    # transient fault is gone (the provider is reused, not rebuilt).
    proposer._fail_full_step = False
    harness.runner._rows[8] = _row(8, prompt)
    harness.adapter._intents[8] = 1
    harness.adapter.begin_prompt(8)
    assert harness.adapter._proposer_reuses == 1
    harness.prefill(prompt, request_id=8)
    assert harness.adapter._states[8].prompt_rows_consumed == len(prompt)
    assert harness.adapter._states[8].proposer.cache_len == len(prompt)
    assert harness.adapter._states[8].checkpoint is None


def test_failure_inside_the_packed_slab_leaves_the_cursor_at_the_last_good_row(
    monkeypatch,
) -> None:
    prompt = (10, 11, 12, 13)
    harness = _harness(
        monkeypatch,
        prompt=prompt,
        session=_PackedPromptSession(fail_after_rows=3),
    )

    with pytest.raises(RuntimeError, match="packed prompt slab failed mid-slab"):
        harness.prefill(prompt)

    # Rows 0..2 primed; the slab died before the seed row.
    assert harness.state().prompt_rows_consumed == 3
    assert harness.proposer.kinds == ["kv", "kv", "kv"]
    assert harness.row.generated_steps == []
    assert harness.session.scratch_reservations == []

    harness.runner.discard((_REQUEST_ID,))
    assert harness.row.mtp2_prompt_hidden_buffer is None
    assert harness.freed and harness.freed[0].ptr == 0x8000
    assert harness.adapter._states == {}


def test_serial_prompt_fallback_keeps_the_same_seed_and_cursor_contract(monkeypatch) -> None:
    """The serial fallback route primes through the same adapter contract.

    ``_prefill_row_serial_with_mtp2_capture`` is the production fallback when the
    packed slab path reports ``NotImplementedError``. It feeds the adapter one
    row at a time with the capture buffer already written in place, so it must
    keep the same rule: every row but the true last is KV-only, and the last row
    runs the single full step.
    """

    prompt = (10, 11, 12, 13)
    harness = _harness(
        monkeypatch,
        prompt=prompt,
        session=_PackedPromptSession(serial_only=True),
    )

    harness.prefill((10, 11))
    harness.prefill((12, 13))

    assert harness.runner._fallback_reasons["packed_prefill_unavailable"] == 2
    assert harness.row.native_prefill is False
    # Serial steps run in chunk order with absolute positions, and only the
    # final row of the final chunk samples.
    assert [
        (token, position, sample) for token, position, sample, _ in harness.session.serial_steps
    ] == [
        (10, 0, False),
        (11, 1, False),
        (12, 2, False),
        (13, 3, True),
    ]
    assert harness.proposer.kinds == ["kv", "kv", "kv", "full"]
    assert [kwargs["input_token"] for _, kwargs in harness.proposer.calls] == [
        11,
        12,
        13,
        _SEED_TOKEN,
    ]
    assert [kwargs["position"] for _, kwargs in harness.proposer.calls] == [1, 2, 3, 4]
    assert harness.state().prompt_rows_consumed == len(prompt)
    # The in-place capture buffer is reused for every row, so every append reads
    # the same pointer.
    assert {kwargs["target_hidden_ptr"] for _, kwargs in harness.proposer.calls} == {0x8000}
    assert harness.session.scratch_reservations == [(2, "decode_batched", len(prompt) + 4)]
    assert [step.token_id for step in harness.row.generated_steps] == [_SEED_TOKEN]
    assert harness.row.mtp2_prompt_hidden_buffer is None
    assert harness.freed == harness.allocated
