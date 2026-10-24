"""CPU lifecycle coverage for the PARO prompt-priming routes.

The staged PARO MTP2 provider primes from the target's own prompt: every prompt
row except the true last one only appends the provider's historical K/V, and the
last row runs one full prediction from the target's sampled root. The packed
prefill route hands the provider one contiguous span per scheduler chunk through
a request-owned sink; the serial fallback route feeds it one row at a time from
a single capture buffer.

These tests drive that contract through the real
``Qwen35ParoResidentModelRunner`` prefill path and the real
``Qwen35ParoMTP2Adapter``, with a session double that mirrors
``prefill_native_packed``'s packed slab and ``final_hidden_chunk_sink`` contract
and a proposer double that mirrors ``NativeMtpChainProposer``'s prompt-phase
cursors.

What is checked here is the owner's own lifecycle: which row may carry the seed,
that a prompt row that fails never advances the provider cursor, that the
prompt workspace is bounded by the chunk rather than by the prompt, that no cycle
scratch or speculative capability exists before priming completes, and that the
workspace plus provider state are reclaimed by the owner that owns them
(``discard``/``close``) rather than by the failing call.

Depth, C>1, and GPU behaviour are out of scope.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import hipengine.generation.qwen35_paro as paro_generation_module
import hipengine.generation.qwen35_paro_mtp2 as paro_mtp2_module
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
    PROMPT_PRIME_CHUNK_ROWS,
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

    Only the surface the prompt routes touch is modelled, but the state left
    behind matches the real proposer: a K/V-only append ends with an unusable
    ``current`` (``token == -1``, exactly what ``advance`` writes for a
    K/V-only row), and a full step ends with a consumable candidate.
    """

    closed = False

    def __init__(
        self,
        *,
        fail_bulk_at: int | None = None,
        fail_append_at: int | None = None,
        fail_full_step: bool = False,
        hidden: int = 8,
    ) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.resets = 0
        self.close_calls = 0
        self.cache_len = 0
        self.position = -1
        self.current = SimpleNamespace(token=0)
        self.max_positions = 8192
        self.max_mtp_tokens = 256
        self.hidden = int(hidden)
        self.scoring_head: SimpleNamespace | None = None
        self._fail_bulk_at = fail_bulk_at
        self._fail_append_at = fail_append_at
        self._fail_full_step = fail_full_step
        self._bulk_count = 0
        self._append_count = 0

    # -- prompt-phase appends ---------------------------------------------

    def append_prompt_kv_bulk(self, rows, *, stream: int = 0) -> int:
        rows = tuple(rows)
        if self._fail_bulk_at is not None and self._bulk_count == self._fail_bulk_at:
            self._bulk_count += 1
            raise RuntimeError("MTP bulk prompt append failed")
        self._bulk_count += 1
        self.calls.append(("bulk", {"rows": rows, "stream": int(stream)}))
        self.cache_len += len(rows)
        if rows:
            self.position = int(rows[-1][2])
        self.current = SimpleNamespace(token=-1)
        return len(rows)

    def append_prompt_kv(self, **kwargs) -> None:
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

    # -- lifecycle --------------------------------------------------------

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

    def device_candidate_token_ids(self, rows: int = 1) -> Tensor:
        return Tensor.from_handle(0x7000, (int(rows),), DType.INT32, Device("hip", 0))

    @property
    def kinds(self) -> list[str]:
        return [kind for kind, _ in self.calls]


class _PackedPromptSession:
    """Resident-session double for the packed prompt slab contract.

    ``prefill_native_packed`` is reproduced as the real one behaves for a single
    request: it walks the slab, hands the chunk sink one contiguous span of
    hidden rows with the span's first absolute prompt index, and finishes the
    sink only on the span that ends the prompt. ``chunk_start_override`` exists
    so a test can inject the contract violation a real session could produce.
    """

    block_size = 256
    hidden_nbytes = 16
    max_sequence_length = 4096
    tokenizer = None
    closed = False

    def __init__(
        self,
        *,
        serial_only: bool = False,
        fail_before_chunk: bool = False,
        chunk_start_override: int | None = None,
    ) -> None:
        self.config = SimpleNamespace(hidden_size=8)
        self.device = Device("hip", 0)
        self.runtime = object()
        self.slabs: list = []
        self.hidden_writes: list[tuple] = []
        self.serial_steps: list[tuple] = []
        self.scratch_reservations: list[tuple[int, str, int]] = []
        self.reset_slot_calls: list[tuple[int, ...]] = []
        self.sampler_row_configurations: list[tuple] = []
        self._serial_only = serial_only
        self._fail_before_chunk = fail_before_chunk
        self._chunk_start_override = chunk_start_override

    # -- packed route -----------------------------------------------------

    def prefill_native_packed(
        self,
        slab,
        *,
        sample: bool = True,
        final_hidden_chunk_sink=None,
    ):
        if self._serial_only:
            raise NotImplementedError("packed prompt prefill is unavailable")
        self.slabs.append(slab)
        if self._fail_before_chunk:
            raise RuntimeError("packed prompt slab failed before the hidden fan-out")
        results = (self._step_result(),) if sample else (None,)
        chunk_start = (
            int(slab.positions[0])
            if self._chunk_start_override is None
            else int(self._chunk_start_override)
        )
        rows = len(slab.positions)
        base_ptr = 0x9000 + 0x1000 * (len(self.slabs) - 1)
        final_hidden_chunk_sink.consume(
            request_id=int(slab.request_ids[0]),
            chunk_start=chunk_start,
            hidden_ptr=base_ptr,
            rows=rows,
            stream=0,
        )
        if chunk_start + rows == int(final_hidden_chunk_sink.total_rows):
            final_hidden_chunk_sink.finish(
                request_id=int(slab.request_ids[0]),
                total_rows=int(final_hidden_chunk_sink.total_rows),
                stream=0,
            )
        return results

    def _write_final_normalized_hidden_bf16_rows(
        self, hidden, *, rows, destination_ptr, scratch_ptr, stream
    ) -> None:
        self.hidden_writes.append(
            (int(hidden.ptr), int(rows), int(destination_ptr), int(scratch_ptr), int(stream))
        )

    # -- serial route -----------------------------------------------------

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

    @staticmethod
    def _step_result() -> Qwen35ParoAutoregressiveStepResult:
        return Qwen35ParoAutoregressiveStepResult(
            token_id=_SEED_TOKEN,
            token_text="seed",
            logit=float("nan"),
        )

    # -- owner surface ----------------------------------------------------

    def prepare_specdec2_verify_scratch(
        self, *, rows, chain_attn_mode, max_context_tokens
    ) -> None:
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

    # The row seam allocates from the generation module and the chunk sink from
    # the adapter module, so both must observe the same allocator.
    for module in (paro_generation_module, paro_mtp2_module):
        monkeypatch.setattr(module, "malloc", _malloc)
        monkeypatch.setattr(module, "free", _free)

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


def _assert_workspace_released(harness: _Harness) -> None:
    """Assert every buffer the request allocated was freed by its owner."""

    assert harness.allocated
    assert sorted(harness.freed, key=lambda buffer: buffer.ptr) == sorted(
        harness.allocated, key=lambda buffer: buffer.ptr
    )


# ---------------------------------------------------------------------------
# Packed chunks: historical rows only, one final prediction
# ---------------------------------------------------------------------------


def test_packed_chunks_append_historical_rows_then_one_final_prediction(monkeypatch) -> None:
    prompt = (10, 11, 12, 13, 14)
    harness = _harness(monkeypatch, prompt=prompt)

    harness.prefill((10, 11))

    # The chunk is entirely historical: its last row is not the prompt's last
    # row, so it appends K/V and runs no prediction.
    assert harness.state().prompt_rows_consumed == 2
    assert harness.proposer.kinds == ["bulk"]
    assert harness.proposer.calls[0][1]["rows"] == (
        (11, harness.allocated[0].ptr + 0, 1),
        (12, harness.allocated[0].ptr + 16, 2),
    )
    assert harness.session.scratch_reservations == []
    assert harness.freed == []

    harness.prefill((12, 13, 14))

    # The slab each chunk builds is chunk-local in tokens and absolute in
    # position, which is what makes the sink's prompt index a valid provider
    # cursor across chunk boundaries.
    first, second = harness.session.slabs
    assert (first.token_ids, first.positions, first.cu_seqlens_q) == ((10, 11), (0, 1), (0, 2))
    assert (second.token_ids, second.positions, second.cu_seqlens_q) == (
        (12, 13, 14),
        (2, 3, 4),
        (0, 3),
    )
    assert second.row_to_request == (_REQUEST_ID,) * 3
    assert second.slot_ids == (0,)

    # Each historical row primes the *next* prompt token; only the true final
    # row consumes the sampled root instead.
    assert [call[1]["rows"] for call in harness.proposer.calls if call[0] == "bulk"] == [
        ((11, harness.allocated[0].ptr + 0, 1), (12, harness.allocated[0].ptr + 16, 2)),
        ((13, harness.allocated[0].ptr + 0, 3), (14, harness.allocated[0].ptr + 16, 4)),
    ]
    full = [kwargs for kind, kwargs in harness.proposer.calls if kind == "full"]
    assert len(full) == 1
    assert full[0]["input_token"] == _SEED_TOKEN
    assert full[0]["position"] == len(prompt)
    assert full[0]["need_result"] is True
    # A one-candidate cycle reads the provider's own device scalar instead.
    assert full[0]["read_token_id"] is False
    assert harness.proposer.cache_len == len(prompt)
    assert harness.proposer.position == len(prompt)
    assert harness.state().prompt_rows_consumed == len(prompt)

    # Exactly one full final step produced the recorded seed token.
    assert [step.token_id for step in harness.row.generated_steps] == [_SEED_TOKEN]
    assert harness.row.native_prefill is True
    # Cycle scratch is reserved only after priming completes, sized from the
    # prompt plus the request's own decode budget.
    assert harness.session.scratch_reservations == [
        (2, "decode_batched", len(prompt) + 4 + 1)
    ]
    # The chunk workspace is freed by the owner that allocated it.
    _assert_workspace_released(harness)
    assert harness.row.mtp2_prompt_hidden_buffer is None


def test_packed_prompt_workspace_is_bounded_by_the_chunk_not_the_prompt(monkeypatch) -> None:
    prompt = tuple(range(10, 30))
    assert len(prompt) > PROMPT_PRIME_CHUNK_ROWS
    harness = _harness(monkeypatch, prompt=prompt)

    harness.prefill(prompt[:PROMPT_PRIME_CHUNK_ROWS])
    harness.prefill(prompt[PROMPT_PRIME_CHUNK_ROWS:])

    hidden = harness.session.config.hidden_size
    # One slab per request, sized by the chunk bound rather than by prompt
    # length: a longer prompt is primed in more chunks through the same slab.
    assert [buffer.nbytes for buffer in harness.allocated] == [
        PROMPT_PRIME_CHUNK_ROWS * hidden * 2,
        PROMPT_PRIME_CHUNK_ROWS * hidden * 2,
    ]
    bulk_rows = [len(call[1]["rows"]) for call in harness.proposer.calls if call[0] == "bulk"]
    assert bulk_rows == [PROMPT_PRIME_CHUNK_ROWS, len(prompt) - PROMPT_PRIME_CHUNK_ROWS - 1]
    # Every prompt row but the last appended a historical row, and the final row
    # ran the single prediction.
    assert sum(bulk_rows) + 1 == len(prompt)
    assert harness.state().prompt_rows_consumed == len(prompt)
    assert harness.proposer.position == len(prompt)
    _assert_workspace_released(harness)


def test_packed_chunk_must_stay_contiguous(monkeypatch) -> None:
    prompt = (10, 11, 12, 13)
    harness = _harness(monkeypatch, prompt=prompt)
    harness.prefill((10, 11))
    assert harness.state().prompt_rows_consumed == 2

    # A session that re-reports prompt indices from zero for the second chunk is
    # the realistic way a non-contiguous span reaches the sink.
    harness.session._chunk_start_override = 0
    with pytest.raises(RuntimeError, match="must be contiguous"):
        harness.prefill((12, 13))

    assert harness.state().prompt_rows_consumed == 2
    assert harness.proposer.kinds == ["bulk"]
    assert harness.freed == []


def test_packed_chunk_failure_before_the_fan_out_publishes_nothing(monkeypatch) -> None:
    harness = _harness(
        monkeypatch,
        prompt=(10, 11, 12, 13),
        session=_PackedPromptSession(fail_before_chunk=True),
    )

    with pytest.raises(RuntimeError, match="failed before the hidden fan-out"):
        harness.prefill((10, 11))

    # The workspace exists, but no row was published and no prediction ran.
    assert harness.state().prompt_rows_consumed == 0
    assert harness.proposer.calls == []
    assert harness.row.generated_steps == []
    assert harness.session.scratch_reservations == []
    assert harness.allocated and harness.freed == []

    # Reclamation is the owner's job, and it frees the whole workspace.
    harness.runner.discard((_REQUEST_ID,))
    _assert_workspace_released(harness)
    assert harness.adapter._states == {}


# ---------------------------------------------------------------------------
# Serial fallback: one row at a time from one capture buffer
# ---------------------------------------------------------------------------


def test_serial_fallback_primes_one_row_at_a_time_from_one_capture(monkeypatch) -> None:
    """The serial fallback route primes through the same adapter contract.

    ``_prefill_row_serial_with_mtp2_capture`` is the production fallback when
    the packed slab path reports ``NotImplementedError``. It feeds the adapter
    one row at a time with the capture buffer already written in place, so it
    must keep the same rule: every row but the true last is K/V-only, and the
    last row runs the single full step.
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
    # Serial steps run in chunk order at absolute positions, and only the final
    # row of the final chunk samples.
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
    # the same pointer. It is allocated once for the request.
    capture_pointers = {
        kwargs["target_hidden_ptr"] for _, kwargs in harness.proposer.calls
    }
    assert len(capture_pointers) == 1
    (capture_ptr,) = capture_pointers
    assert [buffer.nbytes for buffer in harness.allocated if buffer.ptr == capture_ptr] == [
        harness.session.hidden_nbytes
    ]
    # The packed attempt runs first and creates the request's chunk sink before
    # the slab path refuses; that workspace is bounded by the chunk and released
    # with the request, so it must not survive priming either.
    chunk_slab_bytes = (
        min(PROMPT_PRIME_CHUNK_ROWS, len(prompt))
        * harness.session.config.hidden_size
        * 2
    )
    assert sorted(buffer.nbytes for buffer in harness.allocated) == sorted(
        [harness.session.hidden_nbytes, chunk_slab_bytes, chunk_slab_bytes]
    )
    assert harness.session.scratch_reservations == [
        (2, "decode_batched", len(prompt) + 4 + 1)
    ]
    assert [step.token_id for step in harness.row.generated_steps] == [_SEED_TOKEN]
    assert harness.row.mtp2_prompt_hidden_buffer is None
    _assert_workspace_released(harness)


def test_serial_seed_on_a_non_final_row_is_refused_before_any_append(monkeypatch) -> None:
    prompt = (10, 11, 12)
    harness = _harness(
        monkeypatch,
        prompt=prompt,
        session=_PackedPromptSession(serial_only=True),
    )

    with pytest.raises(RuntimeError, match="final prompt row must carry target root"):
        harness.adapter.consume_prompt_row(
            _REQUEST_ID, prompt_index=0, target_hidden_ptr=0x1000, seed_token=_SEED_TOKEN
        )

    # The misplaced seed is rejected on arrival, before that row's K/V append.
    assert harness.state().prompt_rows_consumed == 0
    assert harness.proposer.calls == []
    assert harness.row.generated_steps == []


def test_serial_missing_seed_on_the_final_row_leaves_the_prime_incomplete(monkeypatch) -> None:
    prompt = (10, 11, 12)
    harness = _harness(
        monkeypatch,
        prompt=prompt,
        session=_PackedPromptSession(serial_only=True),
    )
    harness.adapter.begin_prompt(_REQUEST_ID)

    for index in range(len(prompt) - 1):
        harness.adapter.consume_prompt_row(
            _REQUEST_ID, prompt_index=index, target_hidden_ptr=0x1000, seed_token=None
        )
    with pytest.raises(RuntimeError, match="final prompt row must carry target root"):
        harness.adapter.consume_prompt_row(
            _REQUEST_ID,
            prompt_index=len(prompt) - 1,
            target_hidden_ptr=0x1000,
            seed_token=None,
        )

    # The K/V-only rows are legitimate and stay appended; the seed row did not
    # run, so priming can never complete for this attempt.
    assert harness.state().prompt_rows_consumed == len(prompt) - 1
    assert harness.proposer.kinds == ["kv", "kv"]
    assert harness.proposer.current.token == -1
    assert harness.session.scratch_reservations == []
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
    assert harness.state().prompt_rows_consumed == 2
    # The adapter refuses to move on to the cycle phase, so no cycle scratch is
    # reserved from a half-primed provider.
    with pytest.raises(RuntimeError, match="streaming prompt priming is incomplete"):
        harness.adapter.observe_prefill_result(_REQUEST_ID)
    assert harness.session.scratch_reservations == []
    # The root is published by the first decode, not by prefill, so even a
    # completed prime is not yet a capability.
    assert harness.adapter.capability(semantics) is None

    harness.prefill((12, 13))
    assert harness.state().prompt_rows_consumed == len(prompt)
    assert harness.adapter.capability(semantics) is None
    harness.row.first_token_emitted = True
    assert harness.adapter.capability(semantics) is not None
    assert harness.session.scratch_reservations == [
        (2, "decode_batched", len(prompt) + 4 + 1)
    ]


# ---------------------------------------------------------------------------
# Cancellation and owner reclamation
# ---------------------------------------------------------------------------


def test_cancelled_prompt_drop_reclaims_the_workspace_and_recycles_the_provider(
    monkeypatch,
) -> None:
    prompt = (10, 11, 12, 13)
    token = GenerationCancellationToken()
    harness = _harness(monkeypatch, prompt=prompt, cancellation_token=token)
    harness.prefill((10, 11))
    assert harness.state().prompt_rows_consumed == 2
    assert harness.allocated and harness.freed == []

    token.cancel()

    # The real cancellation surface the chunked prefill observes between chunks:
    # the request's own token.
    with pytest.raises(GenerationCancelled):
        harness.prefill((12, 13))
    assert harness.state().prompt_rows_consumed == 2
    assert harness.proposer.kinds == ["bulk"]

    # Reclamation is the owner's job, and it is the only thing that frees the
    # prompt workspace and recycles the provider.
    harness.runner.discard((_REQUEST_ID,))

    _assert_workspace_released(harness)
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
    harness.runner._rows[8] = _row(8, prompt)
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
    # The first request's historical chunk is still in the call log; the reused
    # provider then primed this request's rows from a reset cursor, ending in
    # one final prediction.
    assert harness.proposer.kinds == ["bulk", "bulk", "full"]
    assert harness.proposer.calls[0][1]["rows"] == (
        (11, harness.allocated[0].ptr + 0, 1),
        (12, harness.allocated[0].ptr + 16, 2),
    )
    assert [step.token_id for step in harness.runner._rows[8].generated_steps] == [_SEED_TOKEN]


def test_owner_close_closes_pooled_providers_and_frees_workspaces(monkeypatch) -> None:
    harness = _harness(monkeypatch, prompt=(10, 11, 12))
    harness.prefill((10,))
    harness.runner.discard((_REQUEST_ID,))
    _assert_workspace_released(harness)

    # A second request still priming when the owner closes: its workspace is
    # freed and its provider is closed, including the pooled one.
    harness.runner._rows[8] = _row(8, (20, 21, 22))
    harness.runner._available_model_slots.pop(0)  # admission takes slot 0
    harness.adapter._intents[8] = 1
    harness.adapter.begin_prompt(8)
    pooled = harness.proposer
    assert harness.adapter._states[8].proposer is pooled
    harness.prefill((20,), request_id=8)
    assert len(harness.allocated) > len(harness.freed)

    harness.runner.close()

    _assert_workspace_released(harness)
    assert harness.runner._rows == {}
    assert harness.adapter._states == {}
    assert harness.adapter._proposer_pool == []
    assert pooled.close_calls == 1
    assert harness.session.closed is True
    assert harness.runner._session is None


# ---------------------------------------------------------------------------
# Failures: the cursor must not advance, and cleanup is discard-only
# ---------------------------------------------------------------------------


def test_bulk_append_failure_does_not_publish_the_chunk(monkeypatch) -> None:
    prompt = (10, 11, 12, 13)
    proposer = _PromptProposer(fail_bulk_at=0)
    harness = _harness(monkeypatch, prompt=prompt, proposer=proposer)

    with pytest.raises(RuntimeError, match="MTP bulk prompt append failed"):
        harness.prefill((10, 11))

    # The chunk that failed published no row, and the provider holds nothing.
    assert harness.state().prompt_rows_consumed == 0
    assert proposer.kinds == []
    assert proposer.cache_len == 0
    assert harness.session.scratch_reservations == []
    assert harness.row.generated_steps == []

    # Retrying the same request is not a recovery path: prefill bookkeeping has
    # already consumed the chunk, so the retry is rejected as chunk drift.
    with pytest.raises(RuntimeError, match="prefill chunk drift"):
        harness.prefill((10, 11))

    harness.runner.discard((_REQUEST_ID,))
    _assert_workspace_released(harness)
    assert harness.adapter._states == {}
    assert proposer.resets == 1 and proposer.close_calls == 0

    # A new request over the same prompt primes cleanly on the recycled provider.
    harness.runner._rows[9] = _row(9, prompt)
    harness.adapter._intents[9] = 1
    harness.adapter.begin_prompt(9)
    assert harness.adapter._states[9].proposer is proposer
    assert harness.adapter._states[9].prompt_rows_consumed == 0
    assert proposer.cache_len == 0
    harness.prefill(prompt, request_id=9)
    assert harness.adapter._states[9].prompt_rows_consumed == len(prompt)
    # The failed attempt recorded no append; this chunk appended its historical
    # rows and ran the one final prediction.
    assert proposer.kinds == ["bulk", "full"]
    slab_ptr = harness.allocated[-2].ptr
    assert proposer.calls[0][1]["rows"] == (
        (11, slab_ptr + 0, 1),
        (12, slab_ptr + 16, 2),
        (13, slab_ptr + 32, 3),
    )


def test_final_full_step_failure_skips_verify_reservation_and_recovers_by_discard(
    monkeypatch,
) -> None:
    prompt = (10, 11, 12)
    proposer = _PromptProposer(fail_full_step=True)
    harness = _harness(monkeypatch, prompt=prompt, proposer=proposer)

    with pytest.raises(RuntimeError, match="MTP final prompt step failed"):
        harness.prefill(prompt)

    # Every historical row landed; the seed row did not, so priming is
    # incomplete and the cycle phase must refuse.
    assert harness.state().prompt_rows_consumed == len(prompt) - 1
    assert proposer.kinds == ["bulk"]
    assert proposer.cache_len == len(prompt) - 1
    assert harness.session.scratch_reservations == []
    assert harness.row.generated_steps == []
    with pytest.raises(RuntimeError, match="streaming prompt priming is incomplete"):
        harness.adapter.observe_prefill_result(_REQUEST_ID)

    harness.runner.discard((_REQUEST_ID,))
    _assert_workspace_released(harness)
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
