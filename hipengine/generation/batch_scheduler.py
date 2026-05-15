"""Batch-friendly generation scheduler shell.

The scheduler is model-agnostic: it owns request ids, pending/admitted queues,
physical batch slots, prefill/decode work items, and completion routing.  Model
runners consume the emitted ``WorkItem`` metadata and report generated tokens
back through ``record_generated``.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from hipengine.dispatch import ActiveBatch, BatchShapeKey, RequestState, WorkItem, WorkKind
from hipengine.kvcache import KVTransaction
from hipengine.speculative import DraftBatch, TargetAcceptSummary, TargetCommitPlan, TargetStateCommitBuffers, TargetVerifyBatch, TargetVerifyBuffers


@dataclass(frozen=True, slots=True)
class BatchGenerateRequest:
    request_id: int
    prompt_tokens: tuple[int, ...]
    max_new_tokens: int

    @classmethod
    def from_tokens(cls, request_id: int, prompt_tokens: Iterable[int], *, max_new_tokens: int) -> "BatchGenerateRequest":
        return cls(request_id=int(request_id), prompt_tokens=tuple(int(token) for token in prompt_tokens), max_new_tokens=int(max_new_tokens))


@dataclass(frozen=True, slots=True)
class GeneratedToken:
    request_id: int
    token_id: int
    finished: bool = False


@dataclass(frozen=True, slots=True)
class CompletedRequest:
    request_id: int
    prompt_tokens: tuple[int, ...]
    generated_tokens: tuple[int, ...]
    finished: bool


@dataclass(frozen=True, slots=True)
class SpeculativeVerifyWork:
    target_batch: TargetVerifyBatch
    work_item: WorkItem


@dataclass(frozen=True, slots=True)
class SpeculativeVerifyPlan:
    target_batch: TargetVerifyBatch
    work_item: WorkItem
    transaction: KVTransaction
    shape_key: BatchShapeKey
    graph: object


@dataclass(frozen=True, slots=True)
class SpeculativeVerifyBufferPlan:
    plan: SpeculativeVerifyPlan
    buffers: TargetVerifyBuffers


@dataclass(frozen=True, slots=True)
class SpeculativeCommitPlan:
    verify_plan: SpeculativeVerifyBufferPlan
    summary: TargetAcceptSummary
    commit_plan: TargetCommitPlan


@dataclass(frozen=True, slots=True)
class SpeculativeStateCommitPlan:
    commit_plan: SpeculativeCommitPlan
    buffers: TargetStateCommitBuffers


@dataclass(frozen=True, slots=True)
class GraphBucketStats:
    entries: int
    hits: int
    misses: int


class GraphBucketCache:
    """Tiny graph bucket cache keyed by full batch/specdecode shape."""

    def __init__(self) -> None:
        self._cache: dict[BatchShapeKey, object] = {}
        self._hits = 0
        self._misses = 0

    @property
    def stats(self) -> GraphBucketStats:
        return GraphBucketStats(entries=len(self._cache), hits=self._hits, misses=self._misses)

    def get(self, key: BatchShapeKey) -> object | None:
        if key in self._cache:
            self._hits += 1
            return self._cache[key]
        self._misses += 1
        return None

    def put(self, key: BatchShapeKey, graph: object) -> None:
        self._cache[key] = graph

    def get_or_create(self, key: BatchShapeKey, factory) -> object:
        cached = self.get(key)
        if cached is not None:
            return cached
        graph = factory(key)
        self.put(key, graph)
        return graph

    def clear(self) -> None:
        self._cache.clear()
        self._hits = 0
        self._misses = 0


class ResidentBatchScheduler:
    """Continuous-batching scheduler shell for resident decode runners."""

    def __init__(self, *, capacity: int, context_bucket_size: int = 256) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if context_bucket_size <= 0:
            raise ValueError("context_bucket_size must be positive")
        self.capacity = int(capacity)
        self.context_bucket_size = int(context_bucket_size)
        self.active_batch = ActiveBatch(self.capacity)
        self.graph_buckets = GraphBucketCache()
        self._pending: deque[RequestState] = deque()
        self._completed: dict[int, CompletedRequest] = {}
        self._next_request_id = 0

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    @property
    def active_count(self) -> int:
        return self.active_batch.active_count

    @property
    def completed(self) -> Mapping[int, CompletedRequest]:
        return self._completed

    def submit(self, prompt_tokens: Iterable[int], *, max_new_tokens: int, request_id: int | None = None) -> int:
        rid = self._allocate_request_id() if request_id is None else int(request_id)
        if rid in self.active_batch.requests or any(req.request_id == rid for req in self._pending) or rid in self._completed:
            raise ValueError(f"request_id {rid} already exists")
        self._pending.append(RequestState.from_tokens(rid, prompt_tokens, max_new_tokens=max_new_tokens))
        return rid

    def admit_pending(self) -> tuple[int, ...]:
        """Fill free slots from the pending queue and return admitted request ids."""

        admitted: list[int] = []
        while self._pending and self.active_batch.active_count < self.capacity:
            request = self._pending.popleft()
            self.active_batch.admit(request)
            admitted.append(request.request_id)
        return tuple(admitted)

    def compact(self, order: Sequence[int] | None = None):
        return self.active_batch.compact(order=order)

    def next_prefill_work(self, *, chunk_size: int) -> WorkItem | None:
        """Emit one prefill chunk and advance the request's prompt cursor."""

        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        for request_id in self.active_batch.active_request_ids:
            request = self.active_batch.requests[request_id]
            if request.remaining_prefill <= 0:
                continue
            updated, chunk = request.take_prefill(chunk_size)
            self.active_batch.update_request(updated)
            return WorkItem(
                kind=WorkKind.PREFILL,
                request_ids=(request_id,),
                row_to_request=(request_id,),
                token_rows=(chunk,),
            )
        return None

    def next_decode_work(self) -> WorkItem | None:
        """Emit one decode step over active requests with completed prefill."""

        request_ids = tuple(
            request_id
            for request_id in self.active_batch.active_request_ids
            if self.active_batch.requests[request_id].remaining_prefill == 0
            and self.active_batch.requests[request_id].remaining_decode > 0
            and not self.active_batch.requests[request_id].finished
        )
        if not request_ids:
            return None
        return WorkItem(kind=WorkKind.DECODE, request_ids=request_ids, row_to_request=request_ids)

    def next_speculative_verify_work(
        self,
        draft: DraftBatch,
        *,
        root_tokens: Sequence[int],
        root_positions: Sequence[int],
    ) -> SpeculativeVerifyWork:
        """Emit scheduler metadata for a target verification batch.

        This only validates scheduler ownership/readiness and materializes the
        root+candidate row layout.  It does not run a draft model, target
        verifier, or commit accepted state.
        """

        for request_id in draft.request_ids:
            if request_id not in self.active_batch.requests:
                raise KeyError(request_id)
            request = self.active_batch.requests[request_id]
            if request.remaining_prefill != 0:
                raise ValueError("speculative verification requires completed prefill")
            if request.remaining_decode <= 0 or request.finished:
                raise ValueError("speculative verification requires an active decode request")
        target = TargetVerifyBatch.from_draft(draft, root_tokens=root_tokens, root_positions=root_positions)
        return SpeculativeVerifyWork(target_batch=target, work_item=target.to_work_item())

    def begin_speculative_verify_transaction(self, kv_policy, work: SpeculativeVerifyWork):
        """Begin the KV transaction for scheduler-owned target verification."""

        if work.work_item.request_ids != work.target_batch.request_ids:
            raise ValueError("speculative work request_ids must match target batch")
        seqs = tuple(self.active_batch.requests[request_id] for request_id in work.target_batch.request_ids)
        return kv_policy.begin_transaction(seqs, work.target_batch)

    def record_generated(self, tokens: Sequence[GeneratedToken | tuple[int, int] | tuple[int, int, bool]]) -> tuple[CompletedRequest, ...]:
        """Record generated tokens and reclaim newly completed requests."""

        completed: list[CompletedRequest] = []
        for item in tokens:
            token = _coerce_generated_token(item)
            done = self._append_generated_token(token)
            if done is not None:
                completed.append(done)
        return tuple(completed)

    def record_speculative_accept(self, summary: TargetAcceptSummary) -> tuple[CompletedRequest, ...]:
        """Record accepted speculative tokens from a target verifier summary."""

        for request_id, tokens in zip(summary.request_ids, summary.accepted_tokens, strict=True):
            if request_id not in self.active_batch.requests:
                raise KeyError(request_id)
            request = self.active_batch.requests[request_id]
            if len(tokens) > request.remaining_decode:
                raise ValueError("accepted speculative tokens exceed remaining decode budget")
        completed: list[CompletedRequest] = []
        for request_id, tokens in zip(summary.request_ids, summary.accepted_tokens, strict=True):
            for token_id in tokens:
                done = self._append_generated_token(GeneratedToken(request_id, token_id))
                if done is not None:
                    completed.append(done)
                    break
        return tuple(completed)

    def speculative_verify_shape_key(
        self,
        work: SpeculativeVerifyWork,
        *,
        top_k: int = 0,
        experts_per_token: int = 0,
        replay_steps: int = 1,
    ) -> BatchShapeKey:
        """Return the graph bucket key for scheduler-owned verify work."""

        return work.target_batch.shape_key(
            self.active_batch,
            context_bucket_size=self.context_bucket_size,
            top_k=top_k,
            experts_per_token=experts_per_token,
            replay_steps=replay_steps,
        )

    def get_or_create_speculative_verify_graph(
        self,
        work: SpeculativeVerifyWork,
        factory,
        *,
        top_k: int = 0,
        experts_per_token: int = 0,
        replay_steps: int = 1,
    ) -> object:
        """Cache graph/replay objects for scheduler-owned verify work."""

        key = self.speculative_verify_shape_key(
            work,
            top_k=top_k,
            experts_per_token=experts_per_token,
            replay_steps=replay_steps,
        )
        return self.graph_buckets.get_or_create(key, factory)

    def plan_speculative_verify(
        self,
        kv_policy,
        work: SpeculativeVerifyWork,
        factory,
        *,
        top_k: int = 0,
        experts_per_token: int = 0,
        replay_steps: int = 1,
    ) -> SpeculativeVerifyPlan:
        """Bundle scheduler metadata for one native target-verifier replay."""

        transaction = self.begin_speculative_verify_transaction(kv_policy, work)
        if transaction.request_ids != work.target_batch.request_ids:
            raise ValueError("speculative transaction request_ids must match target batch")
        if transaction.draft_rows != work.target_batch.candidate_count:
            raise ValueError("speculative transaction rows must match target candidate rows")
        if transaction.candidate_counts is not None and transaction.candidate_counts != work.target_batch.candidate_counts:
            raise ValueError("speculative transaction candidate counts must match target batch")
        key = self.speculative_verify_shape_key(
            work,
            top_k=top_k,
            experts_per_token=experts_per_token,
            replay_steps=replay_steps,
        )
        graph = self.graph_buckets.get_or_create(key, factory)
        return SpeculativeVerifyPlan(
            target_batch=work.target_batch,
            work_item=work.work_item,
            transaction=transaction,
            shape_key=key,
            graph=graph,
        )

    def bind_speculative_verify_buffers(
        self,
        plan: SpeculativeVerifyPlan,
        buffers: TargetVerifyBuffers,
    ) -> SpeculativeVerifyBufferPlan:
        """Bind target-verifier device buffers to a scheduler plan."""

        if buffers.request_ids != plan.target_batch.request_ids:
            raise ValueError("target verify buffers request_ids must match speculative plan")
        if buffers.rows != plan.target_batch.rows:
            raise ValueError("target verify buffer rows must match speculative plan")
        if buffers.candidate_rows != plan.target_batch.candidate_count:
            raise ValueError("target verify candidate rows must match speculative plan")
        if buffers.mode != plan.target_batch.mode:
            raise ValueError("target verify buffer mode must match speculative plan")
        return SpeculativeVerifyBufferPlan(plan=plan, buffers=buffers)

    def plan_speculative_commit(
        self,
        verify_plan: SpeculativeVerifyBufferPlan,
        summary: TargetAcceptSummary,
    ) -> SpeculativeCommitPlan:
        """Build the scheduler-owned commit plan for accepted verify rows."""

        target = verify_plan.plan.target_batch
        if summary.request_ids != target.request_ids:
            raise ValueError("accept summary request_ids must match speculative plan")
        if summary.mode != target.mode:
            raise ValueError("accept summary mode must match speculative plan")
        root_rows = set(target.root_rows)
        candidate_rows = set(target.candidate_rows)
        for request_id, count, row, token, position in zip(
            summary.request_ids,
            summary.accepted_counts,
            summary.commit_rows,
            summary.commit_tokens,
            summary.commit_positions,
            strict=True,
        ):
            if row < 0 or row >= target.rows:
                raise ValueError("accept summary commit row must be in target batch")
            if target.row_to_request[row] != request_id:
                raise ValueError("accept summary commit row must belong to its request")
            if count == 0 and row not in root_rows:
                raise ValueError("zero accepted candidates must commit the request root row")
            if count > 0 and row not in candidate_rows:
                raise ValueError("accepted candidates must commit a candidate row")
            if target.draft_depths[row] != count:
                raise ValueError("accept summary commit row depth must match accepted count")
            if target.tokens[row] != token or target.positions[row] != position:
                raise ValueError("accept summary commit token/position must match target row")
        commit = TargetCommitPlan.from_summary(summary, verify_plan.plan.transaction)
        return SpeculativeCommitPlan(verify_plan=verify_plan, summary=summary, commit_plan=commit)

    def bind_speculative_commit_buffers(
        self,
        plan: SpeculativeCommitPlan,
        buffers: TargetStateCommitBuffers,
    ) -> SpeculativeStateCommitPlan:
        """Bind verified state/KV commit buffers to a scheduler commit plan."""

        commit = plan.commit_plan
        if buffers.request_ids != commit.request_ids:
            raise ValueError("state commit buffers request_ids must match speculative commit plan")
        if buffers.mode != commit.mode:
            raise ValueError("state commit buffers mode must match speculative commit plan")
        if not buffers.has_linear_state and not buffers.has_kv_rows:
            raise ValueError("state commit buffers must include linear state or KV rows")
        return SpeculativeStateCommitPlan(commit_plan=plan, buffers=buffers)

    def commit_speculative_kv_transaction(self, kv_policy, plan: SpeculativeStateCommitPlan) -> KVTransaction:
        """Mark the scheduler-owned speculative KV transaction committed."""

        commit = plan.commit_plan.commit_plan
        transaction = plan.commit_plan.verify_plan.plan.transaction
        if commit.transaction_id != transaction.transaction_id:
            raise ValueError("speculative commit plan transaction_id must match KV transaction")
        if commit.request_ids != transaction.request_ids:
            raise ValueError("speculative commit plan request_ids must match KV transaction")
        return kv_policy.commit(transaction, commit.kv_accept_counts)

    def rollback_speculative_kv_transaction(self, kv_policy, plan: SpeculativeVerifyPlan) -> KVTransaction:
        """Rollback a scheduler-owned speculative KV transaction."""

        transaction = plan.transaction
        if transaction.request_ids != plan.target_batch.request_ids:
            raise ValueError("speculative transaction request_ids must match target batch")
        if transaction.draft_rows != plan.target_batch.candidate_count:
            raise ValueError("speculative transaction rows must match target candidate rows")
        return kv_policy.rollback(transaction)

    def shape_key(self, *, mode: WorkKind | str, top_k: int = 0, experts_per_token: int = 0, replay_steps: int = 1) -> BatchShapeKey:
        return self.active_batch.shape_key(
            mode=mode,
            context_bucket_size=self.context_bucket_size,
            top_k=top_k,
            experts_per_token=experts_per_token,
            replay_steps=replay_steps,
        )

    def _allocate_request_id(self) -> int:
        rid = self._next_request_id
        self._next_request_id += 1
        return rid

    def _append_generated_token(self, token: GeneratedToken) -> CompletedRequest | None:
        request = self.active_batch.requests[token.request_id]
        updated = request.append_generated(token.token_id, finished=token.finished)
        self.active_batch.update_request(updated)
        if not updated.finished:
            return None
        self.active_batch.finish(updated.request_id)
        reclaimed = self.active_batch.reclaim(updated.request_id)
        done = CompletedRequest(
            request_id=reclaimed.request_id,
            prompt_tokens=reclaimed.prompt_tokens,
            generated_tokens=reclaimed.generated_tokens,
            finished=reclaimed.finished,
        )
        self._completed[done.request_id] = done
        return done


def _coerce_generated_token(item: GeneratedToken | tuple[int, int] | tuple[int, int, bool]) -> GeneratedToken:
    if isinstance(item, GeneratedToken):
        return item
    if len(item) == 2:
        request_id, token_id = item
        return GeneratedToken(int(request_id), int(token_id), False)
    request_id, token_id, finished = item
    return GeneratedToken(int(request_id), int(token_id), bool(finished))


__all__ = [
    "BatchGenerateRequest",
    "CompletedRequest",
    "GeneratedToken",
    "GraphBucketCache",
    "GraphBucketStats",
    "ResidentBatchScheduler",
    "SpeculativeCommitPlan",
    "SpeculativeStateCommitPlan",
    "SpeculativeVerifyBufferPlan",
    "SpeculativeVerifyPlan",
    "SpeculativeVerifyWork",
]
