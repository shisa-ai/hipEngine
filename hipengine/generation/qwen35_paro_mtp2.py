"""Staged packed-PARO MTP adapter for one request and a bounded candidate chain."""

from __future__ import annotations

from dataclasses import dataclass, replace
import time
from typing import Any, Mapping, Sequence

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.memory import DeviceBuffer, free, malloc
from hipengine.core.tensor import Tensor
from hipengine.kvcache import ClaimLifetime, ResourceClaimSet
from hipengine.runtime.qwen35_paro_runner import (
    Qwen35ParoAutoregressiveStepResult,
    _decode_token_cached,
)
from hipengine.speculative.frontier import (
    CandidateGraph,
    SpeculativeKVBinding,
    ProviderAttachment,
    ProviderCatchupMode,
    SpecRequestPlan,
    SpecTransactionMode,
    SpeculativeCapability,
    TargetFrontier,
)
from hipengine.speculative.interfaces import AcceptResult
from hipengine.speculative.mtp_native import (
    NativeMtpChainProposer,
    NativeMtpStateSnapshot,
    NativeMtpW8A16Head,
)
from hipengine.speculative.prompt_prime import (
    PROMPT_PRIME_REFUSAL_FINISHED,
    PROMPT_PRIME_REFUSAL_NOT_CONTIGUOUS,
    PromptPrimeChunk,
    PromptPrimeLedger,
    PromptPrimePlan,
    PromptPrimeSink,
)
from hipengine.speculative.provider import (
    SpeculativeRequestSemantics,
    validate_staged_target_adapter,
)
from hipengine.speculative.transaction import (
    SpecCycleResult,
    SpecCycleStage,
    SpecCycleTelemetry,
    SpecCycleTransaction,
    SpeculativeStateJournal,
)


_PROPOSER_CAPACITY_FLOOR = 256

#: Deepest candidate chain the resident verifier executes for one request. The
#: target owns ``depth + 1`` verify rows (the root plus one row per candidate) and
#: the provider owns ``depth`` draft advances per cycle, so this is a structural
#: bound on both owners rather than a tuned default.
PARO_MTP2_MAX_CANDIDATE_DEPTH = 3

#: Rows one bulk historical-prompt append stages. This bounds the request's
#: prompt-priming workspace, never the prompt: a longer prompt is primed in more
#: chunks through the same slab. It matches the proposer's default per-launch row
#: chunk, so one sink chunk is one bulk launch set, and it must stay at or below
#: that primitive's hard row cap (64).
PROMPT_PRIME_CHUNK_ROWS = 16


def _proposer_token_requirement(prompt_rows: int, max_tokens: int, depth: int) -> int:
    """Tokens one proposer must hold for a request's prompt and candidate chain.

    The provider walks the prompt, then one draft row per candidate, then the
    target's bonus row. ``max_tokens`` is already counted twice for the prompt
    tail, so the chain adds ``depth`` rows rather than a second copy of the
    context.
    """

    return int(prompt_rows) + 2 * int(max_tokens) + int(depth) + 8


def _proposer_capacity_bucket(required_tokens: int) -> int:
    required = int(required_tokens)
    if required <= 0:
        raise ValueError("PARO MTP2 proposer capacity requirement must be positive")
    return 1 << (max(_PROPOSER_CAPACITY_FLOOR, required) - 1).bit_length()


@dataclass(slots=True)
class _ParoMTP2RequestState:
    request_id: int
    proposer: NativeMtpChainProposer
    prompt_rows_consumed: int = 0
    prompt_sink: "_ParoPromptPrimeSink | None" = None
    checkpoint: NativeMtpStateSnapshot | None = None
    journal: SpeculativeStateJournal | None = None
    # One provider snapshot per drafted candidate past the accepted prefix:
    # ``draft_snapshots[k]`` is the state after consuming a prefix of ``k``
    # candidates, so accepting ``k`` restores ``draft_snapshots[k]``.
    draft_snapshots: tuple[NativeMtpStateSnapshot, ...] = ()
    last_proposal_seconds: float = 0.0
    prompt_prime_seconds: float = 0.0


class _ParoPromptPrimeSink:
    """Request-owned prompt-priming sink over the PARO MTP proposer.

    The target fans one contiguous span of a request's post-output-norm hidden
    rows into :meth:`consume`, and this sink turns them into the proposer's
    historical prompt K/V. Row accounting, chunk splitting and refusal codes
    come from the shared :mod:`hipengine.speculative.prompt_prime` plan and
    ledger, so the chunk contract is the same one the GGUF prompt sinks use;
    only the device work here is PARO's.

    Every historical prompt row consumes the *next* prompt token at position
    ``index + 1``, so the rows of one slice are one contiguous token/position/
    hidden run and go to the bulk append primitive as a single request. The last
    prompt row is different in kind: it consumes the target's own root token and
    must produce a prediction, so it is staged here and completed by
    :meth:`complete` once the target root exists.

    The workspace is two bounded slabs sized by ``PROMPT_PRIME_CHUNK_ROWS``, not
    by prompt length: a longer prompt is primed in more chunks through the same
    allocation. Nothing here is shared between requests.
    """

    def __init__(
        self,
        *,
        adapter: "Qwen35ParoMTP2Adapter",
        state: _ParoMTP2RequestState,
        request_id: int,
        prompt_ids: Sequence[int],
        session: Any,
        chunk_rows: int = PROMPT_PRIME_CHUNK_ROWS,
    ) -> None:
        proposer = state.proposer
        self.adapter = adapter
        self.state = state
        self.proposer = proposer
        self.session = session
        self.prompt_ids = tuple(int(token) for token in prompt_ids)
        self.request_id = int(request_id)
        self.hidden_size = int(session.config.hidden_size)
        self.total_rows = len(self.prompt_ids)
        self.hidden_nbytes = self.hidden_size * DType.BF16.itemsize
        self.device = getattr(session, "device", Device("hip", 0))
        if self.total_rows <= 0:
            raise ValueError("PARO MTP2 prompt sink requires a non-empty prompt")
        proposer_hidden = getattr(proposer, "hidden", None)
        if proposer_hidden is not None and int(proposer_hidden) != self.hidden_size:
            raise ValueError(
                "PARO MTP2 prompt sink width does not match the proposer hidden size: "
                f"target {self.hidden_size}, proposer {int(proposer_hidden)}"
            )
        rows = int(chunk_rows)
        if rows <= 0:
            raise ValueError("PARO MTP2 prompt sink chunk rows must be positive")
        # One chunk is one bounded slice; the slab is never resized.
        self.chunk_rows = min(rows, self.total_rows)
        self.plan = PromptPrimePlan(start_position=0, max_chunk_rows=self.chunk_rows)
        self.ledger = PromptPrimeLedger(
            request_id=self.request_id,
            total_rows=self.total_rows,
        )
        # A request may already own published rows from the row-at-a-time seam
        # (a serial chunk, or a chunk that fell back to serial prefill). Carry
        # that cursor into the ledger so one sink can take over mid-prompt.
        already_published = int(state.prompt_rows_consumed)
        if already_published:
            if already_published > self.total_rows:
                raise RuntimeError(
                    "PARO MTP2 prompt sink cannot resume past the registered prompt"
                )
            self.ledger.publish(chunk_start=0, rows=already_published)
        self.bulk_chunks = 0
        self.bulk_rows = 0
        self._stream: int | None = None
        self._final_row_ptr = 0
        self._fanout_finished = False
        self._completed = False
        self._failed = False
        self._closed = False
        self._slab: DeviceBuffer | None = None
        self._norm_scratch: DeviceBuffer | None = None
        runtime = getattr(session, "runtime", None)
        slab = malloc(self.chunk_rows * self.hidden_nbytes, runtime=runtime)
        try:
            scratch = malloc(
                self.chunk_rows * self.hidden_size * DType.FP16.itemsize,
                runtime=runtime,
            )
        except Exception:
            free(slab, runtime=runtime)
            raise
        self._slab = slab
        self._norm_scratch = scratch

    # -- shared contract ---------------------------------------------------

    @property
    def published_rows(self) -> int:
        """Prompt rows this sink has appended, from the shared ledger."""

        return self.ledger.published_rows

    @property
    def closed(self) -> bool:
        return self._closed

    def consume(
        self,
        *,
        request_id: int,
        chunk_start: int,
        hidden_ptr: int,
        rows: int,
        stream: int,
    ) -> None:
        """Prime the proposer from one contiguous target-hidden chunk."""

        self._validate_owner(int(request_id))
        if self._closed:
            raise RuntimeError("PARO MTP2 prompt sink is closed")
        if self._failed:
            raise RuntimeError("PARO MTP2 prompt sink is failed")
        refusal = self.ledger.check(chunk_start=int(chunk_start), rows=int(rows))
        if refusal == PROMPT_PRIME_REFUSAL_FINISHED:
            raise RuntimeError("PARO MTP2 prompt sink is already finished")
        if refusal == PROMPT_PRIME_REFUSAL_NOT_CONTIGUOUS:
            raise RuntimeError(
                "PARO MTP2 prompt chunks must be contiguous: "
                f"expected {self.ledger.published_rows}, got {int(chunk_start)}"
            )
        if refusal is not None:
            raise ValueError(f"PARO MTP2 prompt chunk is refused: {refusal}")
        if int(hidden_ptr) <= 0:
            raise ValueError("PARO MTP2 prompt chunk requires a device hidden pointer")
        stream = int(stream)
        if self._stream is None:
            self._stream = stream
        elif stream != self._stream:
            raise RuntimeError("PARO MTP2 prompt chunks must use one stream")
        started = time.perf_counter()
        # Row limits are chunk-shape policy, so they come from the shared plan.
        for chunk in self.plan.split(chunk_start=int(chunk_start), rows=int(rows)):
            self._consume_slice(chunk=chunk, hidden_ptr=int(hidden_ptr), stream=stream)
        self._account(time.perf_counter() - started)

    def finish(self, *, request_id: int, total_rows: int, stream: int) -> None:
        """Record that the target fanned out this request's whole prompt.

        PARO's last prompt row is not a cache-only append: it must run a full
        prediction from the target's own root token. That root is only sampled
        once the target prefill has committed, so the row is completed by
        :meth:`complete` rather than here, and this call only pins that the
        fan-out itself is done.
        """

        self._validate_owner(int(request_id))
        del stream
        if self._closed:
            raise RuntimeError("PARO MTP2 prompt sink is closed")
        if self._failed:
            raise RuntimeError("PARO MTP2 prompt sink is failed")
        if int(total_rows) != self.total_rows:
            raise ValueError(
                "PARO MTP2 prompt sink completion row count does not match the prompt: "
                f"expected {self.total_rows}, got {int(total_rows)}"
            )
        self._fanout_finished = True

    def note_row_published(self, index: int) -> None:
        """Record one row that a row-at-a-time caller appended itself."""

        if self._closed:
            raise RuntimeError("PARO MTP2 prompt sink is closed")
        self.ledger.publish(chunk_start=int(index), rows=1)
        if self.ledger.complete:
            self.ledger.mark_finished()
        self._mirror_published_rows()

    def complete(self, *, seed_token: int | None) -> None:
        """Run the final prompt row's full prediction from the target root.

        A no-op when this request's prompt has not reached its last row yet, so
        repeated scheduler chunks leave it untouched until the chunk that ends
        the prompt. The advance is refused until the target has declared its
        fan-out complete, so a mismatched completion signal fails loudly instead
        of running a prediction the target does not expect. The ledger is
        published only after the advance succeeded, so a failed advance never
        claims a completed prompt.
        """

        if self._closed or self._failed or self._completed:
            return
        if int(self._final_row_ptr) <= 0:
            return
        if not self._fanout_finished:
            raise RuntimeError(
                "PARO MTP2 final prompt row completed before the target finished the prompt"
            )
        if seed_token is None:
            raise RuntimeError(
                "PARO MTP2 final prompt row requires the target root token"
            )
        started = time.perf_counter()
        try:
            self.proposer.advance_with_target_hidden(
                input_token=int(seed_token),
                target_hidden_ptr=int(self._final_row_ptr),
                position=self.total_rows,
                need_result=True,
                # A chained cycle drafts from this row's sampled token, so the
                # host readback is required; a one-candidate cycle reads the
                # provider's own packed top-1 scalar on device instead.
                read_token_id=self.adapter.request_depth(self.request_id) > 1,
                read_expert_topk=False,
                read_lm_head_value=False,
                stream=0 if self._stream is None else int(self._stream),
            )
        except Exception:
            self._failed = True
            raise
        self.ledger.publish(chunk_start=self.total_rows - 1, rows=1)
        self.ledger.mark_finished()
        self._completed = True
        self._final_row_ptr = 0
        self._mirror_published_rows()
        self._account(time.perf_counter() - started)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        runtime = getattr(self.session, "runtime", None)
        for buffer in (self._norm_scratch, self._slab):
            if buffer is not None:
                free(buffer, runtime=runtime)
        self._norm_scratch = None
        self._slab = None

    # -- device work -------------------------------------------------------

    def _validate_owner(self, request_id: int) -> None:
        if request_id != self.request_id:
            raise RuntimeError(
                "PARO MTP2 prompt sink owner mismatch: "
                f"expected request {self.request_id}, got {request_id}"
            )

    def _consume_slice(
        self,
        *,
        chunk: PromptPrimeChunk,
        hidden_ptr: int,
        stream: int,
    ) -> None:
        """Prime one validated slice, publishing it only after it lands."""

        slab = self._slab
        scratch = self._norm_scratch
        if slab is None or scratch is None:
            raise RuntimeError("PARO MTP2 prompt sink is closed")
        start = int(chunk.chunk_start)
        count = int(chunk.rows)
        end = start + count
        if end > self.total_rows:
            raise ValueError("PARO MTP2 prompt slice runs past the registered prompt")
        # The row that ends the prompt consumes the target's own root instead of
        # the next prompt token, so it is staged rather than bulk-appended.
        historical = count if end < self.total_rows else count - 1
        # Target prompt rows are FP16 and the slab is BF16; both are two bytes
        # wide, so one row stride covers the source and the destination.
        source_ptr = int(hidden_ptr) + int(chunk.row_offset) * self.hidden_nbytes
        # One row-capable normalize+cast covers the whole slice. The kernels are
        # per-row and elementwise, so each row gets the arithmetic the single-row
        # seam gives it. Every slice refills the same slab from row zero; that is
        # safe because this sink pins one stream and the cast is enqueued ahead
        # of the append that reads it.
        self.session._write_final_normalized_hidden_bf16_rows(
            Tensor.from_handle(
                source_ptr,
                (count, self.hidden_size),
                DType.FP16,
                self.device,
            ),
            rows=count,
            destination_ptr=int(slab.ptr),
            scratch_ptr=int(scratch.ptr),
            stream=stream,
        )
        if historical > 0:
            rows = tuple(
                (
                    int(self.prompt_ids[start + offset + 1]),
                    int(slab.ptr) + offset * self.hidden_nbytes,
                    start + offset + 1,
                )
                for offset in range(historical)
            )
            try:
                appended = int(
                    self.proposer.append_prompt_kv_bulk(rows, stream=stream)
                )
            except Exception:
                # A chunk that fails mid-enqueue may already hold cache rows the
                # ledger never saw, so this request cannot resume priming.
                self._failed = True
                raise
            self.bulk_chunks += 1
            self.bulk_rows += appended
            self.adapter._note_bulk_prompt(chunks=1, rows=appended)
            row = self.adapter.owner._row(self.request_id)
            row.mtp2_bulk_prompt_chunks = int(
                getattr(row, "mtp2_bulk_prompt_chunks", 0)
            ) + 1
            row.mtp2_bulk_prompt_rows = int(
                getattr(row, "mtp2_bulk_prompt_rows", 0)
            ) + appended
        if historical < count:
            self._final_row_ptr = int(slab.ptr) + historical * self.hidden_nbytes
        # Only now, after every launch for this slice succeeded, does the row
        # count move: a slice that fails mid-enqueue must not be published. A
        # slice with no historical row (a one-token prompt) publishes nothing.
        if historical > 0:
            self.ledger.publish(chunk_start=start, rows=historical)
            self._mirror_published_rows()

    # -- accounting --------------------------------------------------------

    def _mirror_published_rows(self) -> None:
        """Keep the request's published-row cursor equal to the ledger."""

        self.state.prompt_rows_consumed = int(self.ledger.published_rows)

    def _account(self, elapsed: float) -> None:
        seconds = max(0.0, float(elapsed))
        self.state.prompt_prime_seconds += seconds
        self.adapter.owner._row(self.request_id).mtp2_prompt_prime_ms += seconds * 1000.0


class Qwen35ParoMTP2Adapter:
    """One bounded provider+target cycle on the packed PARO route.

    The adapter plans between one and ``candidate_budget`` candidates per cycle.
    Provider state is checkpointed before proposal, target state is copied into
    the spare resident slot before verify, and either both owners commit or both
    restore. A chained cycle drafts candidates sequentially and keeps one
    provider snapshot per candidate so the accepted prefix can be repaired
    exactly; a one-candidate cycle hands the target the provider's own packed
    top-1 scalar and never reads a token back to the host.
    """

    provider_name = "paro_mtp2"

    def __init__(
        self,
        owner: Any,
        *,
        enabled: bool = True,
        candidate_budget: int = 1,
    ) -> None:
        self.owner = owner
        self.generator = owner.generator
        self.enabled = bool(enabled)
        # Deepest chain this adapter may execute. The scheduler resolves it
        # through the shared serving policy and registers one intent per
        # request; both numbers come from the same resolved depth, so the
        # session's verify rows and the planned candidate count cannot drift.
        self.candidate_budget = max(
            1, min(PARO_MTP2_MAX_CANDIDATE_DEPTH, int(candidate_budget))
        )
        self._intents: dict[int, int] = {}
        self._states: dict[int, _ParoMTP2RequestState] = {}
        self._disabled_requests: set[int] = set()
        self._proposer_pool: list[NativeMtpChainProposer] = []
        self._active_claims: ResourceClaimSet | None = None
        self._transaction_sequence = 0
        self._proposer_builds = 0
        self._proposer_reuses = 0
        self._bulk_prompt_chunks = 0
        self._bulk_prompt_rows = 0
        validate_staged_target_adapter(self)

    def register_request(self, request_id: int, candidate_budget: int) -> None:
        """Record one request's desired chain depth, clamped to this adapter.

        The scheduler hands the same number to the planner as its desired
        candidate count, so the adapter never invents a depth the plan did not
        ask for and never plans one it cannot execute.
        """

        rid = int(request_id)
        budget = int(candidate_budget)
        if budget <= 0:
            raise ValueError("PARO MTP2 candidate budget must be positive")
        self._intents[rid] = min(budget, self.candidate_budget)
        self._disabled_requests.discard(rid)

    def request_depth(self, request_id: int) -> int:
        """Return the candidate depth this request may plan for."""

        return int(self._intents.get(int(request_id), 0))

    def begin_prompt(self, request_id: int) -> None:
        rid = int(request_id)
        if rid not in self._intents or rid in self._disabled_requests:
            return
        if rid in self._states:
            return
        row = self.owner._row(rid)
        session = self.owner._session
        if session is None:
            raise RuntimeError("PARO MTP2 target session is unavailable")
        started = time.perf_counter()
        required_tokens = _proposer_token_requirement(
            len(row.prompt_ids),
            int(row.request.max_tokens),
            self.candidate_budget,
        )
        proposer_capacity = _proposer_capacity_bucket(required_tokens)
        proposer = None
        while self._proposer_pool:
            candidate = self._proposer_pool.pop()
            if (
                not candidate.closed
                and int(candidate.max_positions) >= int(session.max_sequence_length)
                and int(candidate.max_mtp_tokens) >= required_tokens
                and candidate.scoring_head is not None
                and candidate.scoring_head.owner is session
            ):
                proposer = candidate
                self._proposer_reuses += 1
                break
            candidate.close()
        if proposer is None:
            scoring_head = NativeMtpW8A16Head(
                weight_int8_ptr=int(session.lm_head_weight.tensor.ptr),
                scale_f32_ptr=int(session.lm_head_scale.tensor.ptr),
                vocab_size=int(session.vocab_size),
                threads=int(session.lm_head_threads),
                owner=session,
            )
            proposer = NativeMtpChainProposer(
                self.generator.model_path,
                max_positions=int(session.max_sequence_length),
                max_mtp_tokens=proposer_capacity,
                runtime=session.runtime,
                compiler_version=session.compiler_version,
                scoring_head=scoring_head,
            )
            self._proposer_builds += 1
        proposer.reset()
        self._states[rid] = _ParoMTP2RequestState(rid, proposer)
        row.mtp2_provider_open_ms += (time.perf_counter() - started) * 1000.0

    def consume_prompt_row(
        self,
        request_id: int,
        *,
        prompt_index: int,
        target_hidden_ptr: int,
        seed_token: int | None,
    ) -> None:
        rid = int(request_id)
        self.begin_prompt(rid)
        state = self._states.get(rid)
        if state is None:
            return
        row = self.owner._row(rid)
        index = int(prompt_index)
        if index != state.prompt_rows_consumed:
            raise RuntimeError("PARO MTP2 prompt rows must arrive in order")
        final = index == len(row.prompt_ids) - 1
        if final != (seed_token is not None):
            raise RuntimeError("PARO MTP2 final prompt row must carry target root")
        input_token = (
            int(seed_token)
            if final
            else int(row.prompt_ids[index + 1])
        )
        started = time.perf_counter()
        if final:
            state.proposer.advance_with_target_hidden(
                input_token=input_token,
                target_hidden_ptr=int(target_hidden_ptr),
                position=index + 1,
                need_result=True,
                # The root's sampled token seeds the first candidate of the
                # first cycle, so a chained request needs it on the host.
                read_token_id=self.request_depth(rid) > 1,
                read_expert_topk=False,
                read_lm_head_value=False,
            )
        else:
            state.proposer.append_prompt_kv(
                input_token=input_token,
                target_hidden_ptr=int(target_hidden_ptr),
                position=index + 1,
            )
        elapsed = time.perf_counter() - started
        state.prompt_prime_seconds += elapsed
        row.mtp2_prompt_prime_ms += elapsed * 1000.0
        if state.prompt_sink is not None:
            # A chunk sink may already own this request's prompt accounting, so
            # publish the row there too: one ledger, one cursor, both seams.
            state.prompt_sink.note_row_published(index)
        else:
            state.prompt_rows_consumed += 1

    def prompt_capture_sink(self, request_id: int) -> PromptPrimeSink:
        """Return the request-owned chunk sink for packed prompt priming.

        The sink is created on first use and reused for every scheduler chunk of
        the same prompt, so its ledger cursor survives chunk boundaries. It owns
        the bounded workspace and is released by :meth:`release_prompt_capture`.
        """

        rid = int(request_id)
        self.begin_prompt(rid)
        state = self._states.get(rid)
        if state is None:
            raise RuntimeError("PARO MTP2 prompt capture has no staged request state")
        existing = state.prompt_sink
        if existing is not None and not existing.closed:
            return existing
        session = self.owner._session
        if session is None:
            raise RuntimeError("PARO MTP2 target session is unavailable")
        row = self.owner._row(rid)
        sink = _ParoPromptPrimeSink(
            adapter=self,
            state=state,
            request_id=rid,
            prompt_ids=tuple(row.prompt_ids),
            session=session,
        )
        state.prompt_sink = sink
        return sink

    def finish_prompt_capture(
        self,
        request_id: int,
        *,
        seed_token: int | None,
    ) -> None:
        """Complete a request's prompt priming after the target prefill ran.

        Historical rows were appended inside the target's hidden-chunk fan-out.
        The final prompt row needs the root the target just sampled, so it runs
        here: exactly one full prediction, and only once the target root exists.
        A chunk that does not end the prompt leaves this a no-op.
        """

        state = self._states.get(int(request_id))
        if state is None or state.prompt_sink is None:
            return
        state.prompt_sink.complete(seed_token=seed_token)

    def release_prompt_capture(self, request_id: int) -> None:
        """Free a request's prompt-priming workspace once priming is done."""

        state = self._states.get(int(request_id))
        if state is None or state.prompt_sink is None:
            return
        state.prompt_sink.close()
        state.prompt_sink = None

    def observe_prefill_result(self, request_id: int) -> None:
        rid = int(request_id)
        state = self._states.get(rid)
        if state is None:
            return
        row = self.owner._row(rid)
        if state.prompt_rows_consumed != len(row.prompt_ids):
            raise RuntimeError("PARO MTP2 streaming prompt priming is incomplete")
        session = self.owner._session
        if session is None:
            raise RuntimeError("PARO MTP2 target session is unavailable after prefill")
        depth = self.request_depth(rid)
        if depth <= 0:
            depth = 1
        # Verify writes rows at the root and one per candidate, so the highest
        # position this request's cycles touch is ``context - 1 + depth``; the
        # reservation must cover it rather than the root alone.
        session.prepare_specdec2_verify_scratch(
            rows=depth + 1,
            chain_attn_mode=(
                "c1_loop"
                if str(getattr(self.generator, "execution_profile", "production"))
                == "strict"
                else "decode_batched"
            ),
            max_context_tokens=min(
                int(session.max_sequence_length),
                len(row.prompt_ids) + int(row.request.max_tokens) + depth,
            ),
        )
        self.release_prompt_capture(rid)
        self.owner._release_mtp2_prompt_capture(row)

    def capability(
        self,
        request_semantics: Sequence[SpeculativeRequestSemantics],
    ) -> SpeculativeCapability | None:
        semantics = tuple(request_semantics)
        if not self.enabled or len(semantics) != 1:
            return None
        row_semantics = semantics[0]
        rid = int(row_semantics.request_id)
        row = self.owner._row(rid)
        session = self.owner._session
        if (
            rid not in self._intents
            or rid in self._disabled_requests
            or rid not in self._states
            or session is None
            or int(row.model_slot if row.model_slot is not None else -1) != 0
            or not row.native_greedy
            or not row.first_token_emitted
        ):
            return None
        profile = str(
            getattr(self.generator, "execution_profile", None) or "production"
        )
        target_variant = (
            "b1_graph_off_strict_exact"
            if profile == "strict"
            else "b1_graph_off_fast_d64_candidate"
        )
        depth = self.request_depth(rid)
        if depth <= 0:
            return None
        return SpeculativeCapability(
            capability_key=(
                f"paro_mtp2_c{depth}:{self.generator.backend}:w4_paro:{profile}:{target_variant}"
            ),
            target_key="qwen_paro_w4",
            provider_key="qwen_paro_mtp_bf16",
            method_key="mtp2",
            policy_fingerprint=f"paro-mtp2:c{depth}:k1:{profile}:{target_variant}",
            execution_profile=profile,
            kv_backend_key="paged_bf16_kv_live_spans",
            attachment=ProviderAttachment.TARGET_ATTACHED,
            catchup_mode=ProviderCatchupMode.ONE_WAY_AR,
            supported_modes=("verify_chain",),
            supported_sampling_modes=("greedy",),
            max_requests=1,
            max_candidates_per_request=depth,
            max_frontier_rows=depth + 1,
            proposal_widths=(1,),
            target_row_buckets=tuple(range(2, depth + 2)),
            target_transaction_mode=SpecTransactionMode.PACKED_SCRATCH,
            provider_transaction_mode=SpecTransactionMode.REVERSIBLE_JOURNAL,
            # The resident target captures and replays one verifier
            # VERIFY|ACCEPT graph per row/split/mode bucket; the direct pass stays
            # the per-cycle fallback and the eager route.
            graph_supported=True,
            eager_supported=True,
            strict_fallback_key="paro_target_c1_loop_exact",
            # The shared planner computes ``context_room = max_context_tokens -
            # context_tokens - 1`` and clamps the candidate count against it, so
            # this field is the target's context capacity and nothing else; the
            # planner's own ``- 1`` is the root row's reservation.
            #
            # Bound: a cycle verifies the root at ``context_tokens - 1`` and
            # candidate ``k`` at ``context_tokens - 1 + k``, so the highest
            # position it may touch is ``context_tokens - 1 + count``, and
            # ``_check_position`` requires that to be below
            # ``max_sequence_length``. With ``count <= max_sequence_length -
            # context_tokens - 1`` the highest touched position is at most
            # ``max_sequence_length - 2``. Subtracting the depth here would
            # reserve the whole chain a second time and refuse shallower tails
            # near capacity. This matches the dense GGUF declaration, which
            # publishes the target's capacity for the same reason.
            max_context_tokens=int(session.max_sequence_length),
            kv_binding=SpeculativeKVBinding.for_resident_target(
                target=session,
                storage_view_key="paged_bf16_kv_live_spans",
                topology_key="paro_paged_bf16",
                transaction_mode=SpecTransactionMode.PACKED_SCRATCH,
            ),
        )

    def claims_fit(self, plan: SpecRequestPlan) -> bool:
        if (
            not self.enabled
            or self._active_claims is not None
            or plan.request_ids != plan.speculative_request_ids
            or len(plan.request_ids) != 1
            or int(plan.request_ids[0]) not in self._states
        ):
            return False
        counts = tuple(int(count) for count in plan.candidate_counts)
        if len(counts) != 1:
            return False
        depth = self.request_depth(int(plan.request_ids[0]))
        return bool(1 <= counts[0] <= depth)

    def component_claims(self, plan: SpecRequestPlan) -> Mapping[str, ResourceClaimSet]:
        rid = int(plan.request_ids[0])
        # The target owns one verify row per candidate plus the root row.
        target_rows = int(plan.candidate_counts[0]) + 1
        return {
            "provider": ResourceClaimSet.from_mapping(
                f"{plan.operation_id}:provider",
                {"paro_mtp2.provider_rows": 1},
                request_id=rid,
                lifetime=ClaimLifetime.TRANSACTION,
            ),
            "target": ResourceClaimSet.from_mapping(
                f"{plan.operation_id}:target",
                {"paro_mtp2.target_rows": target_rows},
                request_id=rid,
                lifetime=ClaimLifetime.TRANSACTION,
            ),
            "result": ResourceClaimSet.from_mapping(
                f"{plan.operation_id}:result",
                {"paro_mtp2.result_rows": 1},
                request_id=rid,
                lifetime=ClaimLifetime.TRANSACTION,
            ),
        }

    def reserve_claims(self, claims: ResourceClaimSet) -> ResourceClaimSet:
        if self._active_claims is not None:
            raise RuntimeError("PARO MTP2 already owns an open claim")
        self._active_claims = claims
        return claims

    def release_claims(self, reservation: ResourceClaimSet) -> None:
        if self._active_claims != reservation:
            raise RuntimeError("PARO MTP2 reservation does not own active claims")
        self._active_claims = None

    def prepare_k0(
        self,
        plan: SpecRequestPlan,
        request_semantics: Sequence[SpeculativeRequestSemantics],
        *,
        stream: int | None = None,
    ) -> None:
        del request_semantics, stream
        for request_id in plan.request_ids:
            rid = int(request_id)
            if rid not in self._states:
                continue
            row = self.owner._row(rid)
            # The first scheduler decode transition only publishes the root
            # sampled by target prefill. The provider is already primed from
            # that prompt/root and must remain live for the following cycle.
            if not bool(row.first_token_emitted):
                continue
            self._drop_request(rid, disable=True, recycle=True)

    def prepare_requests(
        self,
        plan: SpecRequestPlan,
        request_semantics: Sequence[SpeculativeRequestSemantics],
        *,
        stream: int | None = None,
    ) -> None:
        del request_semantics, stream
        counts = tuple(int(count) for count in plan.candidate_counts)
        depth = (
            self.request_depth(int(plan.request_ids[0]))
            if len(plan.request_ids) == 1
            else 0
        )
        if (
            self._active_claims is None
            or len(plan.request_ids) != 1
            or plan.request_ids != plan.speculative_request_ids
            or len(counts) != 1
            or not 1 <= counts[0] <= depth
            or int(plan.request_ids[0]) not in self._states
        ):
            raise RuntimeError("PARO MTP2 plan is not prepared/claim-fit")

    def propose_batch(
        self,
        plan: SpecRequestPlan,
        request_semantics: Sequence[SpeculativeRequestSemantics],
        *,
        stream: int | None = None,
    ) -> CandidateGraph:
        del stream
        semantics = tuple(request_semantics)
        if len(semantics) != 1:
            raise ValueError("PARO MTP2 proposal requires one request semantic row")
        rid = int(plan.request_ids[0])
        row = self.owner._row(rid)
        state = self._states[rid]
        count = int(plan.candidate_counts[0])
        depth = self.request_depth(rid)
        if not 1 <= count <= depth:
            raise RuntimeError(
                "PARO MTP2 proposal depth exceeds the request's candidate budget"
            )
        started = time.perf_counter()
        if state.journal is not None:
            raise RuntimeError("PARO MTP2 previous cycle still owns a state journal")
        state.journal = SpeculativeStateJournal()
        state.checkpoint = state.journal.capture(
            "provider", lambda: state.proposer.save_state(0),
            state.proposer.restore_state,
        )
        snapshots: list[NativeMtpStateSnapshot] = []
        candidate_token_ids: Tensor | None = None
        if depth == 1:
            # A one-candidate cycle hands the target the provider's own packed
            # top-1 scalar and keeps the candidate token off the host.
            candidate_tokens: tuple[int, ...] = (0,)
            candidate_token_ids = state.proposer.device_candidate_token_ids()
            handoff = "device_i32"
            row.mtp2_candidate_device_handoffs += 1
        else:
            candidate_tokens = (self._host_candidate_token(state.proposer),)
            if count > 1:
                # One snapshot per candidate the cycle may have to undo. The
                # pre-draft state repairs a zero-length accepted prefix, and the
                # live draft state already repairs ``count - 1``, so slots
                # ``1 .. count-1`` cover every shorter prefix.
                snapshots.append(state.proposer.save_state(1))
                for step in range(1, count):
                    state.proposer.advance_with_previous_hidden(
                        input_token=candidate_tokens[-1],
                        position=state.proposer.position + 1,
                        need_result=True,
                        read_token_id=True,
                        read_expert_topk=False,
                        read_lm_head_value=False,
                    )
                    candidate_tokens += (self._host_candidate_token(state.proposer),)
                    if step < count - 1:
                        snapshots.append(state.proposer.save_state(1 + step))
            handoff = "host_i32"
        state.draft_snapshots = tuple(snapshots)
        state.last_proposal_seconds = time.perf_counter() - started
        return CandidateGraph(
            provider_key=plan.provider_key or "qwen_paro_mtp_bf16",
            method_key="mtp2",
            policy_fingerprint=f"paro-mtp2:c{depth}:k1",
            cycle_id=plan.cycle_id,
            transaction_id=self._next_transaction_id(),
            request_ids=plan.request_ids,
            resident_slots=plan.resident_slots,
            root_positions=(int(semantics[0].context_tokens) - 1,),
            row_offsets=(0, count),
            row_to_request=(rid,) * count,
            parent_candidate_rows=tuple(
                -1 if step == 0 else step - 1 for step in range(count)
            ),
            draft_depths=tuple(range(1, count + 1)),
            active_mask=(True,) * count,
            candidate_tokens=candidate_tokens,
            token_ids=candidate_token_ids,
            mode="verify_chain",
            provider_metadata=(("candidate_handoff", handoff),),
        )

    @staticmethod
    def _host_candidate_token(proposer: NativeMtpChainProposer) -> int:
        """Return the provider's host-visible next candidate token.

        A device-only result leaves ``current.token`` at its sentinel, so a
        chained proposal refuses instead of drafting from an invalid id.
        """

        token = int(proposer.current.token)
        if token < 0:
            raise RuntimeError(
                "PARO MTP2 chained proposal requires a host candidate token"
            )
        return token

    def kv_live_spans_owner(self, plan: SpecRequestPlan) -> str:
        return f"paro-resident:{id(self.owner)}:{plan.operation_id}"

    @staticmethod
    def _verify_target(
        session: Any,
        batch: Any,
        *,
        chain_attn_mode: str,
        graph_mode: str,
        candidate_token_ids_i32: Tensor | None,
    ) -> Any:
        rows = int(batch.rows)
        return session.verify_chain_bulk_and_commit(
            batch,
            base_slot=0,
            capture_layer_ids=(),
            # A zero-width capture is the persistent PARO MTP contract, but the
            # row count must equal this cycle's verify rows: the verifier refuses
            # a capture buffer shorter than the cycle, and the graph-commit
            # bucket keys on the same shape.
            capture_hidden_concat=Tensor.from_handle(
                0, (rows, 0), DType.BF16, Device("hip", 0)
            ),
            capture_row_start=0,
            chain_attn_mode=str(chain_attn_mode),
            graph_mode=str(graph_mode),
            canonicalize_after=False,
            synchronize_after_commit=False,
            candidate_token_ids_i32=candidate_token_ids_i32,
        )

    @staticmethod
    def _target_graph_mode(plan: SpecRequestPlan) -> str:
        """Honor the planned variant without disabling graph capability.

        One-candidate public requests currently select eager verification:
        graph replay lost 3.23% decode throughput, confirmed at 3.06%, in the
        2026-10-24 gfx1151 comparison. Graph plans still execute capture/replay.
        See docs/REFACTOR.md for the comparison that clears this preference.
        """

        return "auto" if plan.execution_route == "graph" else "off"

    @staticmethod
    def _executed_graph_route(verify: Any) -> str:
        """Return the route the target actually executed for one verify cycle.

        ``replayed`` is the truthful signal: it is true exactly when the graph
        was launched and its outputs are the ones the caller reads. That covers
        a cached replay, a ``validate`` capture that executed its fresh graph,
        and a ``#107`` recapture cycle, none of which are the eager route. A
        capture that never launched its graph leaves the direct-pass outputs
        live and reports eager. The status string is only a fallback for a
        provider record that carries no ``replayed`` field.
        """

        graph = getattr(verify, "graph", None)
        if not isinstance(graph, Mapping):
            return "eager"
        executed = graph.get("replayed")
        if isinstance(executed, bool):
            return "graph" if executed else "eager"
        return "graph" if str(graph.get("status")) == "replayed" else "eager"

    def execute_target_frontier(
        self,
        plan: SpecRequestPlan,
        frontier: TargetFrontier,
        complete_claims: ResourceClaimSet,
        *,
        commit: bool,
        cancelled_request_ids: Any,
    ) -> SpecCycleResult:
        if not commit:
            raise ValueError("PARO MTP2 target execution requires commit=True")
        if frontier.target_batch is None:
            raise RuntimeError("PARO MTP2 C1 requires host-visible target batch")
        rid = int(plan.request_ids[0])
        row = self.owner._row(rid)
        state = self._states[rid]
        session = self.owner._session
        if session is None or row.model_slot != 0:
            raise RuntimeError("PARO MTP2 C1 requires resident target slot 0")
        transaction = SpecCycleTransaction(
            operation_id=plan.operation_id,
            transaction_id=int(frontier.provider_transaction_id or 0),
            cycle_id=plan.cycle_id,
            request_ids=plan.request_ids,
            reserved_claims=complete_claims,
            pre_target_cursors=(int(session.position_arr[0]),),
            pre_rng_counters=(0,),
            target_transaction_mode=plan.target_transaction_mode,
            target_owner=f"paro-target:{id(session)}",
            target_checkpoint_ids=(f"target-slot1:{plan.cycle_id}",),
            pre_provider_cursors=(int(state.proposer.cache_len),),
            provider_transaction_mode=plan.provider_transaction_mode,
            provider_owner=f"paro-provider:{id(state.proposer)}",
            provider_checkpoint_ids=(f"provider-snapshot0:{plan.cycle_id}",),
            target_open=True,
            provider_open=True,
        )
        self._checkpoint_target(state, session)
        if tuple(int(value) for value in cancelled_request_ids()):
            return self._cancelled_result(transaction, state, session)
        target_started = time.perf_counter()
        verify = self._verify_target(
            session,
            frontier.target_batch,
            chain_attn_mode=(
                "c1_loop"
                if str(getattr(self.generator, "execution_profile", "production"))
                == "strict"
                else "decode_batched"
            ),
            graph_mode=self._target_graph_mode(plan),
            candidate_token_ids_i32=(
                frontier.candidate_graph.token_ids
                if frontier.candidate_graph is not None
                else None
            ),
        )
        actual_route = self._executed_graph_route(verify)
        target_seconds = time.perf_counter() - target_started
        candidate_graph = frontier.candidate_graph
        count = (
            int(candidate_graph.candidate_counts[0])
            if candidate_graph is not None
            else 1
        )
        if candidate_graph is not None and candidate_graph.token_ids is not None:
            row.mtp2_candidate_d2h_after_target += int(candidate_graph.candidate_rows)
        accepted = int(verify.accepted_count)
        bonus = int(
            verify.next_token
            if verify.next_token is not None
            else verify.target_top1[accepted]
        )
        depth = max(1, self.request_depth(rid))
        update_started = time.perf_counter()
        try:
            self._repair_provider_after_verify(
                state,
                accepted=accepted,
                count=count,
                accepted_tokens=tuple(int(token) for token in verify.accepted_tokens),
                bonus=bonus,
                target_hidden_ptr=int(verify.selected_target_hidden_ptr),
                # The next cycle's first candidate comes from this row's host
                # readback, which only a chained cycle needs.
                read_token_id=depth > 1,
            )
        except Exception:
            self._restore_target_and_provider(state, session)
            raise
        provider_update_seconds = time.perf_counter() - update_started
        cancelled = tuple(int(value) for value in cancelled_request_ids())
        if cancelled:
            return self._cancelled_result(transaction, state, session)
        if state.journal is not None:
            state.journal.commit()
            state.journal = None
        state.checkpoint = None
        output_ids = (
            *tuple(int(token) for token in verify.accepted_tokens),
            bonus,
        )
        for token in output_ids:
            self.owner._record_step(
                row,
                Qwen35ParoAutoregressiveStepResult(
                    token_id=token,
                    token_text=_decode_token_cached(session.tokenizer, token),
                    logit=float("nan"),
                ),
            )
        row.mtp2_cycles += 1
        row.mtp2_candidate_counts.append(count)
        row.mtp2_accepted_counts.append(accepted)
        row.mtp2_execution_routes.append(actual_route)
        accept = AcceptResult(
            request_ids=plan.request_ids,
            accepted_counts=(accepted,),
            accepted_tokens=(tuple(int(token) for token in verify.accepted_tokens),),
            transaction_id=transaction.transaction_id,
            selected_candidate_rows=(int(verify.commit_row),),
            correction_or_bonus_tokens=(bonus,),
            target_cursor_deltas=(len(output_ids),),
            provider_cursor_deltas=(accepted,),
            finish_reasons=(None,),
        )
        telemetry = SpecCycleTelemetry(
            operation_id=plan.operation_id,
            request_ids=plan.request_ids,
            candidate_counts=plan.candidate_counts,
            plan_reasons=plan.reasons,
            k0_classes=plan.k0_classes,
            proposal_widths=plan.proposal_widths,
            target_row_decomposition=plan.target_row_decomposition,
            execution_route=actual_route,
            proposal_seconds=max(0.0, state.last_proposal_seconds),
            target_seconds=target_seconds,
            provider_update_seconds=provider_update_seconds,
            weight_sweeps=1,
        )
        committed = replace(
            transaction,
            target_committed=True,
            provider_committed=True,
        )
        return SpecCycleResult.committed(committed, accept, telemetry=telemetry)

    def rollback_cycle(
        self,
        plan: SpecRequestPlan,
        candidate_graph: CandidateGraph | None,
        error: BaseException,
    ) -> None:
        del candidate_graph, error
        rid = int(plan.request_ids[0])
        state = self._states.get(rid)
        session = self.owner._session
        if state is not None and session is not None:
            self._restore_target_and_provider(state, session)

    def recover_cycle_failure(self, plan: SpecRequestPlan, error: BaseException) -> bool:
        del error
        for request_id in plan.request_ids:
            self._drop_request(int(request_id), disable=True)
        return True

    def close_requests(self, request_ids: Sequence[int]) -> None:
        for request_id in request_ids:
            self._drop_request(int(request_id), disable=False, recycle=True)
            self._intents.pop(int(request_id), None)
            self._disabled_requests.discard(int(request_id))

    @staticmethod
    def _repair_provider_after_verify(
        state: _ParoMTP2RequestState,
        *,
        accepted: int,
        count: int,
        accepted_tokens: tuple[int, ...],
        bonus: int,
        target_hidden_ptr: int,
        read_token_id: bool,
    ) -> None:
        """Move the provider onto the accepted prefix, then take the bonus row.

        The proposal left one snapshot per candidate it drafted past the
        accepted prefix. Accepting ``count - 1`` needs no restore because the
        live draft state already sits after that candidate; full acceptance
        consumes the last candidate from the live state; a shorter prefix
        restores the snapshot taken before the first rejected candidate. A
        one-candidate cycle is the ``count == 1`` case of the same table, so its
        two provider calls are unchanged.
        """

        proposer = state.proposer
        if accepted >= count:
            proposer.advance_with_previous_hidden(
                input_token=int(accepted_tokens[-1]),
                position=proposer.position + 1,
                need_result=False,
                read_expert_topk=False,
                read_lm_head_value=False,
            )
        elif accepted < count - 1:
            snapshots = state.draft_snapshots
            if accepted >= len(snapshots):
                raise RuntimeError(
                    "PARO MTP2 accepted prefix has no provider snapshot"
                )
            proposer.restore_state(snapshots[accepted])
        proposer.advance_with_target_hidden(
            input_token=bonus,
            target_hidden_ptr=int(target_hidden_ptr),
            position=proposer.position + 1,
            read_token_id=bool(read_token_id),
            read_expert_topk=False,
            read_lm_head_value=False,
        )
        state.draft_snapshots = ()

    def _checkpoint_target(
        self, state: _ParoMTP2RequestState, session: Any,
    ) -> None:
        if state.journal is None:
            raise RuntimeError("PARO MTP2 target checkpoint requires a provider journal")

        def capture() -> int:
            position = int(session.position_arr[0])
            # Verify appends only at/after this cursor. The spare slot owns
            # mutable recurrent/hidden state, not another copy of the KV prefix.
            session.copy_slot_state(0, 1, kv_rows=0)
            session._record_slot_position_host(position, slot=1)
            return position

        def restore(position: int) -> None:
            session.copy_slot_state(1, 0, kv_rows=0)
            session._record_slot_position_host(position, slot=0)

        state.journal.capture("target", capture, restore)

    def _restore_target_and_provider(
        self,
        state: _ParoMTP2RequestState,
        session: Any,
    ) -> None:
        if state.journal is None:
            return
        try:
            state.journal.rollback()
        finally:
            session.runtime.device_synchronize()
        state.checkpoint = None
        state.journal = None
        state.draft_snapshots = ()

    def _cancelled_result(
        self,
        transaction: SpecCycleTransaction,
        state: _ParoMTP2RequestState,
        session: Any,
    ) -> SpecCycleResult:
        self._restore_target_and_provider(state, session)
        return SpecCycleResult(
            stage=SpecCycleStage.CANCELLED,
            transaction=replace(
                transaction,
                target_open=False,
                provider_open=False,
                rolled_back=True,
            ),
            cancelled_request_ids=transaction.request_ids,
        )

    def _drop_request(
        self,
        request_id: int,
        *,
        disable: bool,
        recycle: bool = False,
    ) -> None:
        state = self._states.pop(int(request_id), None)
        if state is not None:
            if state.prompt_sink is not None:
                state.prompt_sink.close()
                state.prompt_sink = None
            if recycle and not state.proposer.closed and state.checkpoint is None and state.journal is None:
                state.proposer.reset()
                self._proposer_pool.append(state.proposer)
            else:
                state.proposer.close()
        if disable:
            self._disabled_requests.add(int(request_id))

    def _next_transaction_id(self) -> int:
        self._transaction_sequence += 1
        return self._transaction_sequence

    def _note_bulk_prompt(self, *, chunks: int, rows: int) -> None:
        """Count bulk prompt-priming work that actually ran on the device."""

        self._bulk_prompt_chunks += int(chunks)
        self._bulk_prompt_rows += int(rows)

    def observability_snapshot(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "intents": len(self._intents),
            "states": len(self._states),
            "pooled_proposers": len(self._proposer_pool),
            "proposer_builds": self._proposer_builds,
            "proposer_reuses": self._proposer_reuses,
            "active_claim": self._active_claims is not None,
            "disabled_requests": sorted(self._disabled_requests),
            "bulk_prompt_chunks": self._bulk_prompt_chunks,
            "bulk_prompt_rows": self._bulk_prompt_rows,
        }

    def close(self) -> None:
        for request_id in tuple(self._states):
            self._drop_request(request_id, disable=False)
        for proposer in self._proposer_pool:
            proposer.close()
        self._proposer_pool.clear()
        self._intents.clear()
        self._disabled_requests.clear()
        self._active_claims = None


__all__ = ["PROMPT_PRIME_CHUNK_ROWS", "Qwen35ParoMTP2Adapter"]
