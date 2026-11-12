"""Request ownership and incremental publication for feature-based GGUF drafters."""
from __future__ import annotations

import threading
import time

from hipengine.generation.deadline import raise_if_generation_deadline_expired
from hipengine.generation.registry import FinishDetails, GenerationOutput, GenerationStreamChunk
from hipengine.kernels.backends import hip_target_arch_environment
from hipengine.speculative.streaming import trim_speculative_output


class FeatureDraftTextProvider:
    """Share immutable weights; lease mutable target and drafter state per request."""

    def _init_request_owner(self):
        self._lock = threading.RLock()
        self._active_children = set()

    def create_child(self, request):
        # Admission does not tokenize, allocate, or execute model work.
        with self._lock:
            if self._closed:
                raise RuntimeError("speculative provider is closed")
            child = FeatureDraftChild(self, request)
            self._active_children.add(child)
            return child

    def close(self):
        with self._lock:
            self._closed = True
            error = None
            for child in tuple(self._active_children):
                try:
                    child.close(reason="shutdown")
                except BaseException as exc:
                    error = error or exc
            if self._native is not None:
                native, self._native = self._native, None
                try:
                    native.close()
                except BaseException as exc:
                    error = error or exc
            if error is not None:
                raise error

    def generate_detailed(self, request):
        child = self.create_child(request)
        try:
            output = None
            while output is None:
                output = child.advance()
                child.drain_token_chunks()
            return [output]
        finally:
            child.close(reason="completed" if child.output is not None else "error")

    def stream_detailed(self, request):
        child = self.create_child(request)
        try:
            while child.output is None:
                child.advance()
                chunks = child.drain_token_chunks()
                for _, chunk in chunks:
                    yield chunk
                if child.output is not None and not chunks:
                    yield GenerationStreamChunk(
                        text="", generated_token_ids=child.output.generated_token_ids,
                        finish_details=child.output.finish_details, telemetry=child.output.telemetry,
                    )
        finally:
            child.close(reason="completed" if child.output is not None else "disconnect")


class FeatureDraftChild:
    """Advance one prompt prefill or verified cycle, never a whole decode loop."""

    def __init__(self, provider, request):
        self.provider = provider
        self.request = request
        self.native = None
        self.output = None
        self._scope = None
        self._steps = None
        self._closed = False
        self._events = []
        self._ids = []
        self._pending = []
        self._text = []
        self._decoder = None
        self._prefill_ms = 0.0

    def advance(self):
        owner = self.provider
        with owner._lock, hip_target_arch_environment(owner.target.target_arch):
            if self._closed:
                raise RuntimeError("speculative child is closed")
            if self.output is not None:
                return self.output
            try:
                raise_if_generation_deadline_expired(self.request)
                if self._steps is None:
                    self._started = time.perf_counter()
                    self._prompt = owner._prepare_request(self.request)
                    if self.request.max_tokens == 0:
                        self.output = GenerationOutput(
                            text="", generated_token_ids=(), prompt_tokens=len(self._prompt),
                            finish_details=FinishDetails(reason="length", length_limit=0, sampler_mode="greedy"),
                        )
                        return self.output
                    owner.target._prepare_kv_policy(self.request)
                    owner._ensure_loaded()
                    required = len(self._prompt) + self.request.max_tokens + owner._workspace_tokens()
                    maximum = getattr(owner.target, "_prepared_max_sequence_length", None) or required
                    if required > maximum:
                        raise ValueError("request and draft workspace exceed target context capacity")
                    scope = owner.target._resident_session_scope(
                        shared_runner=owner.target._get_shared_runner(),
                        pool_name=f"{owner.provider_name}_target", max_sequence_length=maximum,
                    )
                    session, self._reused = scope.__enter__()
                    self._scope = scope
                    self.native, owner._native = owner._native, None
                    if self.native is not None and self.native.max_context_len < maximum:
                        native, self.native = self.native, None
                        native.close()
                    if self.native is None:
                        self.native = owner._make_native(session, maximum)
                    self._steps = owner._cycle_steps(session, self.native, self._prompt, self.request)
                    tokenizer = owner.target.tokenizer
                    factory = getattr(tokenizer, "incremental_decoder", None)
                    self._decoder = factory(skip_special=False) if callable(factory) else None
                    self._eos = tokenizer.eos_token_id if self.request.eos_token_id is None else self.request.eos_token_id
                    self._stops = set(self.request.stop_token_ids)
                    if not self.request.ignore_eos:
                        self._stops.update(getattr(tokenizer, "stop_token_ids", ()))
                    self._prefixes = tuple(
                        tuple(sequence[:width]) for sequence in self.request.stop_token_sequences
                        for width in range(1, len(sequence))
                    )
                raise_if_generation_deadline_expired(self.request)
                started = time.perf_counter()
                update = next(self._steps)
                if update is None:
                    self._prefill_ms += (time.perf_counter() - started) * 1000
                    return None
                return self._publish(update)
            except BaseException:
                self.close(reason="error")
                raise

    def _publish(self, update):
        request = self.request
        tail = trim_speculative_output(
            update["output_ids"], max_tokens=request.max_tokens, min_tokens=request.min_tokens,
            eos_token_id=self._eos, stop_token_ids=self._stops,
            stop_token_sequences=request.stop_token_sequences, ignore_eos=request.ignore_eos,
        )
        finished = tail.finish_reason is not None or update["finished"]
        details = None
        if finished:
            reason = tail.finish_reason or "length"
            details = FinishDetails(
                reason=reason, eos_token_id=self._eos if reason == "eos" else None,
                stop_sequence=tail.matched_stop_sequence,
                length_limit=request.max_tokens if reason == "length" else None,
                sampler_mode="greedy", phase="decode",
            )
        new_ids = tail.token_ids[len(self._ids):]
        for index, token in enumerate(new_ids):
            self._ids.append(token)
            self._pending.append(token)
            final = finished and index == len(new_ids) - 1
            if final:
                suppressed = len(tail.matched_stop_sequence) or (1 if details.reason == "eos" else 0)
                visible = self._pending[:-suppressed] if suppressed else self._pending
                self._pending = []
            else:
                retained = max((len(p) for p in self._prefixes if tuple(self._pending[-len(p):]) == p), default=0)
                count = len(self._pending) - retained
                visible, self._pending = self._pending[:count], self._pending[count:]
            text = (self._decoder.feed(visible, final=final) if self._decoder is not None
                    else self.provider.target.tokenizer.decode(visible, skip_special=False))
            self._text.append(text)
            self._events.append((token, GenerationStreamChunk(text=text)))
        if not finished:
            return None
        self.provider.last_decode_seconds = update["elapsed_s"]
        self.output = GenerationOutput(
            text="".join(self._text), generated_token_ids=tuple(self._ids),
            prompt_tokens=len(self._prompt), finish_details=details,
            telemetry={
                "execution_path": self.provider.execution_path,
                "prompt_tokens": len(self._prompt), "generated_tokens": len(self._ids),
                "diagnostics": {
                    "target_session_reused": self._reused,
                    **{key: update[key] for key in ("draft_forwards", "graph_verify_cycles", "mean_acceptance")},
                },
                "timing": {"prefill_ms": self._prefill_ms, "decode_ms": update["elapsed_s"] * 1000,
                           "total_ms": (time.perf_counter() - self._started) * 1000},
            },
        )
        if self._events:
            token, chunk = self._events[-1]
            self._events[-1] = (token, GenerationStreamChunk(
                text=chunk.text, finish_details=details, generated_token_ids=tuple(self._ids),
                telemetry=self.output.telemetry,
            ))
        return self.output

    def drain_token_chunks(self):
        events, self._events = tuple(self._events), []
        return events

    def close(self, *, reason):
        with self.provider._lock:
            if self._closed:
                return
            self._closed = True
            try:
                if self._steps is not None:
                    self._steps.close()
            finally:
                try:
                    if self._scope is not None:
                        scope, self._scope = self._scope, None
                        if reason == "completed":
                            try:
                                scope.__exit__(None, None, None)
                            except BaseException:
                                reason = "error"
                                raise
                        else:
                            scope.__exit__(GeneratorExit, GeneratorExit(), None)
                finally:
                    try:
                        if self.native is not None:
                            native, self.native = self.native, None
                            if reason == "completed" and not self.provider._closed and self.provider._native is None:
                                self.provider._native = native
                            else:
                                native.close()
                    finally:
                        self.provider._active_children.discard(self)
