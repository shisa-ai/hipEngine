"""Gemma request-owned MTP quanta on the shared resident scheduler.

Device work packs ready requests. Each request keeps its own KV, sampler frame,
and unpublished commit tail. The scheduler publishes one canonical token at a
time, so cancellation discards only the cancelled request's private tail.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from hipengine.generation.batch_scheduler import GeneratedToken
from hipengine.generation.registry import GenerationOutput, GenerationStreamChunk
from hipengine.generation.sampling import row_seed_for_index
from hipengine.runtime.gemma4 import Gemma4Runner


@dataclass
class _Row:
    request: Any
    prompt_ids: tuple[int, ...]
    runner: Any = None
    frame: Any = None
    operation: Any = None
    prefilled: int = 0
    speculative: bool = False
    published: int = 0
    published_text: str = ""
    latest: GenerationOutput | None = None
    target_widths: list[int] = field(default_factory=list)
    proposal_widths: list[int] = field(default_factory=list)


# Provider diagnostics that describe drafting and verification; a plain row
# served on the same owners reports none of them.
_MTP_ONLY_DIAGNOSTICS = (
    "candidate_budget", "draft_acceptance", "tokens_per_cycle", "cycles",
    "target_verify", "draft_rollback", "accept_rule", "draft_law",
    "verify_equivalence",
)


class Gemma4ResidentModelRunner:
    """Physical target/proposal packing with stable request-local cache owners."""

    supports_buffered_speculative_decode = True
    supports_prefill_decode_same_round = True
    supports_multiple_prefill_quanta_per_round = False

    def __init__(self, generator, *, capacity: int | None = None):
        self.generator = generator
        self.provider = generator._speculative_provider
        self.capacity = 4 if capacity is None else int(capacity)
        if self.capacity < 1:
            raise ValueError("Gemma resident capacity must be positive")
        self.server_mtp_batch_max_active_requests = min(4, self.capacity)
        # Independent KV owners permit one scheduler prefill quantum to cover
        # several prompts even though their device prefill calls remain separate.
        self.packed_prefill_max_rows = min(4, self.capacity)
        self._rows: dict[int, _Row] = {}
        self._outputs: dict[int, GenerationOutput] = {}
        self._available: list[Any] = []
        self._owners: list[Any] = []
        self._workspace = None
        self._closed = False
        self._target_widths = deque(maxlen=256)
        self._proposal_widths = deque(maxlen=256)

    def _new_runner(self, source, *, logits_rows, storage="bf16", scale_dtype="fp16"):
        return Gemma4Runner(
            weights=source.weights, capacity=max(source.capacity, logits_rows),
            max_block=max(source.max_block, logits_rows),
            max_logits_rows=logits_rows,
            sliding_reserve=source.sliding_reserve,
            # The resident verifier uses the registered row-exact fallback.
            # Packing requests changes ownership/launch shape, not target math.
            # Prompt prefill still uses the selected profile's variants.
            verify_dense_exact=True,
            prefill_attention_variants=source.prefill_attention_variants,
            decode_attention_variants=source.decode_attention_variants,
            exact_prefill_attention_variants=source.exact_prefill_attention_variants,
            kv_storage=storage, kv_scale_dtype=scale_dtype,
        )

    def prepare(self, *, max_sequence_length=None):
        if max_sequence_length is not None and int(max_sequence_length) > self.generator.context_length:
            raise ValueError("Gemma resident prepare exceeds the configured context capacity")
        if self._workspace is not None:
            return
        self.provider._ensure_drafter()
        source = self.generator._ensure_runner()
        width = self.provider.candidate_budget + 1
        try:
            self._workspace = self._new_runner(source, logits_rows=min(4, self.capacity) * width)
            for _ in range(self.capacity):
                self._owners.append(self._new_runner(source, logits_rows=width))
            self._available.extend(self._owners)
        except BaseException:
            self.close()
            raise

    @property
    def active_request_ids(self):
        return tuple(self._rows)

    def prompt_tokens(self, prompt):
        return tuple(self.generator.tokenizer.encode(prompt) if isinstance(prompt, str) else prompt)

    def scheduler_max_new_tokens(self, request):
        return int(request.max_tokens)

    def speculative_desired_candidate_count(self, request):
        return self.provider.candidate_budget

    def register_batch(self, request_ids, request, *, prompt_rows):
        if len(request_ids) != len(prompt_rows) or len(request_ids) != len(request.prompts):
            raise ValueError("Gemma request IDs, prompts and token rows must align")
        self.generator.validate_request(request)
        for index, (rid, prompt) in enumerate(zip(request_ids, prompt_rows, strict=True)):
            if rid in self._rows or rid in self._outputs:
                raise ValueError("Gemma request ID is already registered")
            if len(prompt) + request.max_tokens > self.generator.context_length:
                raise ValueError("Gemma prompt and output budget exceed the configured context")
            seed = (row_seed_for_index(request, index),)
            child = replace(request, prompts=(tuple(prompt),), row_seeds=seed,
                            cancellation_token=None, deadline_at=None)
            self._rows[int(rid)] = _Row(child, tuple(prompt))

    def register_speculative_request(self, request_id, desired, *, static_eligibility=None):
        del desired, static_eligibility
        row = self._rows[int(request_id)]
        if row.request.kv_storage not in {"auto", "bf16"}:
            raise NotImplementedError("Gemma MTP requires BF16 shared KV; INT8 assistant reads are not implemented")
        row.speculative = True

    def reserve_admission(self, request):
        row = self._rows[request.request_id]
        if row.runner is not None:
            return
        if not self._available:
            raise MemoryError("Gemma resident request slots are exhausted")
        row.runner = self._available.pop()
        from hipengine.generation.gemma4_gguf import _resolve_kv_storage

        storage, scale_dtype, _ = _resolve_kv_storage(row.request)
        current_storage = getattr(row.runner, "kv_storage_resolved", "bf16")
        current_scale = getattr(row.runner, "kv_scale_dtype", "fp16")
        if (current_storage, current_scale) != (storage, scale_dtype):
            old = row.runner
            # AR INT8 remains executable, but cannot join a BF16 packed verify.
            # Allocate before releasing the old owner so failure is reversible.
            try:
                row.runner = self._new_runner(old, logits_rows=self.provider.candidate_budget + 1,
                                              storage=storage, scale_dtype=scale_dtype)
            except BaseException:
                row.runner = None
                self._available.append(old)
                raise
            owner_index = next(i for i, owner in enumerate(self._owners) if owner is old)
            self._owners[owner_index] = row.runner
            old.close()
        row.frame = self.provider.generation_steps(
            row.prompt_ids, row.request, runner=row.runner,
            candidate_budget=self.provider.candidate_budget if row.speculative else 0,
        )
        try:
            row.operation = next(row.frame)
        except StopIteration as done:
            row.latest = done.value

    def rollback_admission(self, request):
        row = self._rows.get(request.request_id)
        if row is not None:
            self._release(row)

    def prefill_batch(self, work, *, commit):
        if not commit:
            raise ValueError("Gemma prefill requires commit=True")
        for rid, tokens in zip(work.request_ids, work.token_rows, strict=True):
            row = self._rows[int(rid)]
            tokens = tuple(tokens)
            if tokens != row.prompt_ids[row.prefilled:row.prefilled + len(tokens)]:
                raise ValueError("Gemma prefill chunk differs from its request prompt")
            if row.request.max_tokens == 0:
                row.prefilled += len(tokens)
                continue
            logits = row.runner.forward(tokens)
            row.prefilled += len(tokens)
            if row.prefilled == len(row.prompt_ids):
                self._advance(row, logits)
                if row.operation.kind != "commit":
                    raise RuntimeError("Gemma prefill did not produce its initial commit")
                row.latest = row.operation.output

    def _advance(self, row, result=None):
        try:
            row.operation = row.frame.send(result)
        except StopIteration as done:
            row.latest = done.value
            row.operation = None

    def _draft(self, rows):
        head = self.provider._ensure_drafter().head
        operations = [row.operation for row in rows]
        states = head.prime_batch([op.runner.hidden_state(row=op.hidden_row) for op in operations])
        shared = [{index: op.runner.shared_kv(index) for index in range(op.runner.layer_count)}
                  for op in operations]
        tokens = [op.token for op in operations]
        drafts = [[] for _ in rows]
        for depth in range(max(op.budget for op in operations)):
            active = [index for index, op in enumerate(operations) if depth < op.budget]
            logits, next_states = head.forward_batch(
                [tokens[index] for index in active],
                positions=[operations[index].runner.position for index in active],
                shared_kv=[shared[index] for index in active],
                recurrent_states=np.ascontiguousarray(states[active]),
            )
            self._proposal_widths.append(len(active))
            for offset, index in enumerate(active):
                row = rows[index]
                row.proposal_widths.append(len(active))
                states[index] = next_states[offset]
                tokens[index] = int(operations[index].select(logits[offset]))
                drafts[index].append(tokens[index])
        for row, proposed in zip(rows, drafts, strict=True):
            self._advance(row, proposed)

    def _compute(self, rows):
        for row in rows:
            if row.operation.kind == "commit":
                self._advance(row)
        drafting = [row for row in rows if row.operation is not None and row.operation.kind == "draft"]
        if drafting:
            self._draft(drafting)
        verifying = [row for row in rows if row.operation is not None and row.operation.kind == "verify"]
        unpacked = [row for row in verifying if row.request.kv_storage == "int8_per_token_head"]
        for row in unpacked:
            result = row.runner.forward(row.operation.tokens)
            row.target_widths.append(1)
            self._target_widths.append(1)
            self._advance(row, np.asarray(result).reshape(1, -1))
            row.latest = row.operation.output
        verifying = [row for row in verifying if row.request.kv_storage != "int8_per_token_head"]
        if verifying:
            logits = self._workspace.forward_batch(
                [row.runner for row in verifying], [row.operation.tokens for row in verifying],
                verification=True,
            )
            self._target_widths.append(len(verifying))
            for row, result in zip(verifying, logits, strict=True):
                row.target_widths.append(len(verifying))
                self._advance(row, result)
                if row.operation is None or row.operation.kind != "commit":
                    raise RuntimeError("Gemma verification did not produce a commit")
                row.latest = row.operation.output

    def decode_batch(self, work, *, commit):
        if not commit:
            raise ValueError("Gemma decode requires commit=True")
        rows = [self._rows[int(rid)] for rid in work.request_ids]
        ready = [row for row in rows if row.latest is not None
                 and row.published == len(row.latest.generated_token_ids)]
        if ready:
            # Different acceptance lengths otherwise keep live requests out of
            # phase forever. Allow one bounded cycle of queued output when a
            # peer needs work; never advance a terminal request. Each queue is
            # bounded by this allowance plus one cycle's committed output.
            allowance = self.provider.candidate_budget + 1
            ready = [row for row in rows if row.latest is not None
                     and row.operation is not None and not row.operation.terminal
                     and len(row.latest.generated_token_ids) - row.published <= allowance]
            for start in range(0, len(ready), 4):
                self._compute(ready[start:start + 4])
        events = []
        for rid, row in zip(work.request_ids, rows, strict=True):
            output = row.latest
            if output is None or row.published >= len(output.generated_token_ids):
                raise RuntimeError("Gemma decode has no committed token to publish")
            token = int(output.generated_token_ids[row.published])
            row.published += 1
            terminal = bool(row.operation.terminal and row.published == len(output.generated_token_ids))
            # A byte-fallback token can leave an incomplete UTF-8 suffix. Retain
            # that suffix until a later token completes it, or terminal flush.
            text = self.generator.tokenizer.decode(output.generated_token_ids[:row.published], skip_special=False)
            if not terminal:
                text = text.rstrip("\ufffd")
            if not text.startswith(row.published_text):
                raise RuntimeError("Gemma tokenizer changed an already published prefix")
            delta = text[len(row.published_text):]
            row.published_text = text
            records = output.token_logprobs[row.published - 1:row.published]
            telemetry = self._telemetry(row)
            events.append(GeneratedToken(int(rid), token, finished=terminal,
                stream_chunk=GenerationStreamChunk(
                    text=delta, token_logprobs=records,
                    finish_details=self._finish_details(row, output.finish_details) if terminal else None,
                    telemetry=telemetry,
                )))
        return tuple(events)

    def _telemetry(self, row):
        telemetry = row.latest.telemetry if row.latest else None
        if telemetry is None:
            return None
        diagnostics = dict(telemetry.diagnostics or {})
        diagnostics.update({
            "physical_target_request_widths": tuple(row.target_widths),
            "physical_proposal_request_widths": tuple(row.proposal_widths),
            "request_state_owner": "independent_kv_and_sampler",
            "publication_mode": "resident_token_queue",
            "phase_census_clock": "request_wall_including_scheduler_wait",
        })
        if not row.speculative:
            diagnostics["speculative_provider"] = None
            # A plain row runs no draft and no verifier; it must not claim one.
            for key in _MTP_ONLY_DIAGNOSTICS:
                diagnostics.pop(key, None)
        state = telemetry.decode_state
        if state is not None:
            state = replace(state, generated_tokens=row.published)
            if not row.speculative:
                state = replace(state, execution_path="gemma4_resident_ar",
                                sampler_mode=self._plain_sampler_mode(row))
        return replace(telemetry, decode_state=state, diagnostics=diagnostics)

    def _plain_sampler_mode(self, row):
        """The sampler mode the plain Gemma 4 route reports for this row.

        Resident rows select from host logits, so in that route's terms a
        greedy row is ``processed_argmax`` and a sampled one
        ``host_logits_sample``.
        """
        from hipengine.generation.gemma4_gguf import _ar_sampler_mode
        from hipengine.generation.gemma4_sampling import Gemma4Sampler

        return _ar_sampler_mode(True, Gemma4Sampler(row.request, list(row.prompt_ids)))

    def _finish_details(self, row, finish):
        if finish is None or row.speculative:
            return finish
        return replace(finish, sampler_mode=self._plain_sampler_mode(row))

    def compact_batch(self, moves):
        # Scheduler slot numbers never address a request's device cache.
        del moves

    def reclaim(self, completed):
        row = self._rows.pop(completed.request_id, None)
        if row is None:
            return
        ids = tuple(completed.generated_tokens)
        output = row.latest
        if output is None:
            output = GenerationOutput(text="", generated_token_ids=())
        fully_published = ids == tuple(output.generated_token_ids or ())
        finish = output.finish_details if fully_published and row.operation and row.operation.terminal else completed.finish_details
        visible = len(ids) - len(finish.stop_sequence or ()) if finish else len(ids)
        self._outputs[completed.request_id] = replace(
            output, generated_token_ids=ids,
            text=self.generator.tokenizer.decode(ids[:visible], skip_special=False),
            token_logprobs=output.token_logprobs[:visible],
            finish_details=self._finish_details(row, finish), telemetry=self._telemetry(row),
        )
        self._release(row)

    def _release(self, row):
        if row.frame is not None:
            row.frame.close()
            row.frame = None
        if row.runner is not None:
            row.runner.reset()
            self._available.append(row.runner)
            row.runner = None

    def has_outputs(self, request_ids):
        return all(int(rid) in self._outputs for rid in request_ids)

    def missing_outputs(self, request_ids):
        return [int(rid) for rid in request_ids if int(rid) not in self._outputs]

    def take_outputs(self, request_ids):
        return [self._outputs.pop(int(rid)) for rid in request_ids]

    def discard(self, request_ids):
        for rid in request_ids:
            row = self._rows.pop(int(rid), None)
            if row is not None:
                self._release(row)
            self._outputs.pop(int(rid), None)

    def finalize_batch(self, request, request_ids, outputs):
        del request, request_ids
        self.generator.last_generation_outputs = tuple(outputs)

    def observability_snapshot(self):
        return {"active_requests": len(self._rows), "available_slots": len(self._available),
                "physical_target_request_widths": tuple(self._target_widths),
                "physical_proposal_request_widths": tuple(self._proposal_widths)}

    def close(self):
        if self._closed:
            return
        self._closed = True
        for row in self._rows.values():
            if row.frame is not None:
                row.frame.close()
        self._rows.clear()
        self._outputs.clear()
        for runner in self._owners:
            runner.close()
        self._owners.clear()
        self._available.clear()
        if self._workspace is not None:
            self._workspace.close()
            self._workspace = None
