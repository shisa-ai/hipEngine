"""Public generation for Gemma 4 GGUF artifacts.

This is the adapter between ``LLM.generate()`` and
:class:`hipengine.runtime.gemma4.Gemma4Runner`. It owns the request protocol -
tokenization, the decode loop, stop conditions, and the output shape - and
delegates every forward pass to the runner.

A greedy request with no logit processors takes the device argmax and never
brings a logits row to the host. A request with a temperature, top-k, top-p,
min-p, a repetition, presence or frequency penalty, a logit bias, suppressed
tokens, a ``min_tokens`` floor, a thinking budget, or a logprob request copies
each step's row home and draws from it through
:class:`hipengine.generation.gemma4_sampling.Gemma4Sampler`. Multi-token stop
sequences are matched against the emitted history on both this route and the
speculative one. Every other control is named by ``_validate_request`` rather
than silently ignored, so a request that asks for one fails loudly instead of
quietly dropping it - and ``validate_request`` exposes that same answer to a
serving front end before it schedules the request.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

from hipengine.generation.deadline import raise_if_generation_deadline_expired
from hipengine.generation.finish import finish_details_with_sampling_state
from hipengine.generation.gemma4_sampling import Gemma4Sampler
from hipengine.generation.registry import (
    FinishDetails,
    GenerationOutput,
    GenerationRequest,
    GenerationStreamChunk,
    TokenLogprob,
    register_text_generator,
)
from hipengine.generation.sampling import validate_sampling_params
from hipengine.kernels.backends import hip_target_arch_for_backend, resolve_backend
from hipengine.loading.gguf import GGUFReader
from hipengine.loading.safetensors import WeightIndex
from hipengine.runtime.gemma4 import (
    Gemma4DeviceWeights,
    Gemma4Runner,
    gemma4_require_context_capacity,
    gemma4_text_config_from_reader,
    load_gemma4_device_weights,
)
from hipengine.tokenization.gguf import Gemma4GGUFTokenizer

_GEMMA4_QUANT = "gguf_q4_k_m"
_GEMMA4_DEFAULT_CONTEXT = 8_192


def _gemma4_decode_graph_enabled() -> bool:
    """Whether the whole-step decode graph replays Gemma 4 decode steps.

    The graph replaces the ~500 per-step host submissions with one
    ``hipGraphLaunch``; on the shallow-to-mid depths where the launched step
    leaves the device idle between kernels it measures 1.09-1.16x launched
    decode. Depth windows that reach the decode split route are not
    capture-exact, so the session itself falls back to the launched step
    there and the graph stays a pure win on the default path. Set
    ``HIPENGINE_GEMMA4_DECODE_GRAPH=0`` to restore the launched decode.
    """

    return os.environ.get("HIPENGINE_GEMMA4_DECODE_GRAPH", "1") != "0"


def gemma4_token_logprob(
    tokenizer: Any,
    token_id: int,
    logprob: float | None,
    top_logprobs: tuple[tuple[int, float], ...],
) -> TokenLogprob:
    """One generated token's host-visible logprob metadata.

    ``logprob`` and ``top_logprobs`` come from
    :class:`hipengine.generation.gemma4_sampling.Gemma4Sampler`, which reports
    them the way the shared host sampler does; this turns token ids into the text
    the OpenAI schema carries beside them.
    """

    return TokenLogprob(
        token_id=int(token_id),
        token_text=tokenizer.decode([int(token_id)], skip_special=False),
        logprob=None if logprob is None else float(logprob),
        top_logprobs=tuple(
            (
                int(top_id),
                tokenizer.decode([int(top_id)], skip_special=False),
                float(top_logprob),
            )
            for top_id, top_logprob in top_logprobs
        ),
    )


def _resolve_kv_storage(request: GenerationRequest) -> tuple[str, str, str]:
    """Map a request's KV storage controls onto the runner's constructor args.

    ``auto`` is the BF16 comparison path. ``int8_per_token_head`` is the only
    INT8 layout the Gemma 4 writer and consumers implement: per-token/head
    scales in fp16 or fp32. The request validator rejects the unsupported
    combinations before this runs, so the failure here is defensive.
    """

    storage = str(request.kv_storage or "auto")
    if storage == "auto":
        storage = "bf16"
    if storage == "bf16":
        return "bf16", "fp16", "per_token_head"
    if storage != "int8_per_token_head":
        raise NotImplementedError(
            f"Gemma 4 KV storage {request.kv_storage!r} is not implemented "
            "(bf16 or int8_per_token_head)"
        )
    granularity = str(request.kv_scale_granularity or "per_token_head")
    if granularity != "per_token_head":
        raise NotImplementedError(
            "Gemma 4 INT8 KV storage implements per_token_head scale granularity only"
        )
    scale_dtype = str(request.kv_scale_dtype or "fp16")
    if scale_dtype not in {"fp16", "fp32"}:
        raise NotImplementedError("Gemma 4 INT8 KV storage implements fp16 or fp32 scales only")
    return storage, scale_dtype, granularity


def _ar_sampler_mode(host_sampler: bool, sampler: Gemma4Sampler) -> str:
    """The sampler mode the decode loop reports for one row.

    ``greedy`` is the device argmax route, which never processes a row on the
    host; the two host modes split on whether the request samples at all.
    """

    if not host_sampler:
        return "greedy"
    return "processed_argmax" if sampler.greedy else "host_logits_sample"


@dataclass
class Gemma4GGUFGenerator:
    """Greedy generator over a resident Gemma 4 GGUF artifact."""

    model_path: str | Path
    weight_index: WeightIndex
    model_plugin: Any
    backend: str = "hip_gfx1100"
    context_length: int = _GEMMA4_DEFAULT_CONTEXT
    # The prefill-attention variants this run requests, or ``None`` for the
    # strict kernel. The execution profile supplies them before the runner is
    # built; the layer still matches each against its own layer's head geometry
    # and keeps the strict kernel on a capability miss.
    prefill_attention_variants: tuple[str, ...] | None = None
    # The decode-attention variants this run requests, or ``None`` for the strict
    # native kernel. The execution profile's binder supplies them before the
    # runner is built; the layer applies them only at its native/exact attention
    # call site and keeps every other route unchanged.
    decode_attention_variants: tuple[str, ...] | None = None
    # The exact-prefill attention variants this run requests, or ``None`` for the
    # strict native kernel. The execution profile's binder supplies them before
    # the runner is built; the layer applies them only at its native/exact
    # attention call site and only for a multi-row sliding block.
    exact_prefill_attention_variants: tuple[str, ...] | None = None
    last_generation_outputs: tuple[GenerationOutput, ...] = field(
        default=(), init=False, repr=False
    )
    last_generation_seconds: float | None = field(default=None, init=False, repr=False)
    _reader: GGUFReader | None = field(default=None, init=False, repr=False)
    _weights: Gemma4DeviceWeights | None = field(default=None, init=False, repr=False)
    _runner: Gemma4Runner | None = field(default=None, init=False, repr=False)
    _tokenizer: Gemma4GGUFTokenizer | None = field(default=None, init=False, repr=False)
    _load_seconds: float | None = field(default=None, init=False, repr=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)
    _decode_graph_session: Any | None = field(default=None, init=False, repr=False)
    _decode_graph_runner: Any | None = field(default=None, init=False, repr=False)
    _speculative_provider: Any | None = field(default=None, init=False, repr=False)
    _speculative_max_logits_rows: int = field(default=0, init=False, repr=False)

    # No ``supports_speculative_mtp`` here. This generator does not implement the
    # legacy ``generate_speculative_mtp_detailed`` protocol, but the engine also
    # accepts a *staged* protocol implemented on the runner
    # (``speculative_capability`` plus ``execute_target_frontier`` or
    # ``execute_speculative_cycle``). Declaring the legacy miss as a class
    # attribute made ``engine_loop`` return False before it ever consulted those
    # staged hooks, so the runner could not opt in. Leaving the attribute off is
    # accurate: ``engine_loop`` treats a missing attribute as "no legacy
    # method" and still routes through the staged check, which reports False
    # until the runner grows the hooks.
    #
    # ``attach_speculative_provider`` is the *public* route and it is the one
    # this generator implements: ``LLM`` resolves a provider from the
    # speculative registry and attaches it here before materialization.
    supports_stream_many = False
    supports_resident_session_kv = False
    # Both routes publish one buffered chunk that carries the whole published
    # run, logprobs included: this generator's stream is the engine loop's
    # ``generate_detailed`` fallback chunk, and the MTP provider's
    # ``stream_detailed`` builds its chunk from the same output. The server maps
    # those chunk logprobs through the same helpers the blocking response uses,
    # so a streamed ``logprobs`` request reports what the blocking one does.
    supports_stream_logprobs = True
    chat_template_family = "gemma"
    reasoning_parser_name = "gemma_tags"
    tool_parser_name = "unsupported"

    @property
    def engine_loop_config_defaults(self):
        # Protect-decode drains an active request before prefilling arrivals,
        # preventing independent HTTP requests from reaching packed decode.
        return {"prefill_decode_policy": "fair"}

    @property
    def server_mtp_batch_max_active_requests(self) -> int | None:
        """Maximum physical request width of the attached resident MTP owner."""

        return 4

    def validate_request(self, request: GenerationRequest) -> None:
        """Public pre-admission validation for this generator's requests.

        ``SubmitPollTextGenerator`` calls this before it schedules a request, so a
        control this generator cannot execute is refused with a 4xx instead of
        failing inside a decode step. The single source of truth stays
        :meth:`_validate_request`, which :meth:`generate_detailed` also runs, so
        the pre-admission answer and the in-generation answer cannot drift.
        """

        self._validate_request(request)

    def __post_init__(self) -> None:
        self.model_path = Path(self.model_path).expanduser().resolve()
        self.backend = resolve_backend(self.backend)
        self.context_length = int(self.context_length)
        if self.context_length <= 0:
            raise ValueError("context_length must be positive")

    # --- tokenizer ------------------------------------------------------

    @property
    def reader(self) -> GGUFReader:
        if self._reader is None:
            self._reader = GGUFReader(self.model_path)
        return self._reader

    @property
    def tokenizer(self) -> Gemma4GGUFTokenizer:
        if self._tokenizer is None:
            self._tokenizer = Gemma4GGUFTokenizer.from_gguf_info(self.reader.info)
        return self._tokenizer

    def tokenize(self, text: str) -> tuple[int, ...]:
        """Tokenize preformatted text; use ``tokenize_chat`` for a plain message."""

        return tuple(self.tokenizer.encode(str(text)))

    def render_chat_prompt(
        self,
        messages,
        *,
        tools=None,
        enable_thinking: bool = True,
        add_generation_prompt: bool = True,
    ) -> str:
        """Render the embedded template through the server's model protocol."""

        from hipengine.chat.gemma4 import render_gemma4_chat

        if tools:
            raise NotImplementedError("Gemma 4 server tool-call parsing is not implemented")
        return render_gemma4_chat(
            self.tokenizer.chat_template,
            messages,
            tools=tools,
            enable_thinking=enable_thinking,
            add_generation_prompt=add_generation_prompt,
        )

    def tokenize_chat(
        self,
        user: str,
        *,
        system: str | None = None,
        enable_thinking: bool = True,
    ) -> tuple[int, ...]:
        """Tokenize one user turn through the artifact's chat template.

        ``enable_thinking`` defaults to True. The Gemma 4 template treats it as a
        bare variable defaulting to false, and the false branch emits an empty
        thought channel that suppresses reasoning. Passing it explicitly keeps
        the reasoning behaviour a caller gets from depending on a decision
        rather than on a Jinja default.

        The rendered template is a complete prompt, so it is encoded without
        adding special tokens. The canonical template emits ``{{ bos_token }}``
        itself; adding BOS here would insert a second one, which is a prompt
        neither llama.cpp nor HuggingFace ever produces.
        """

        template = self.tokenizer.chat_template
        if not template:
            raise ValueError(
                "the artifact carries no tokenizer.chat_template, so a chat prompt "
                "cannot be formatted; pass preformatted text instead"
            )
        messages = []
        if system:
            messages.append({"role": "system", "content": str(system)})
        messages.append({"role": "user", "content": str(user)})
        rendered = self.render_chat_prompt(messages, enable_thinking=enable_thinking)
        return tuple(self.tokenizer.encode(rendered, add_special_tokens=False))

    def count_tokens(self, text: str) -> int:
        return len(self.tokenize(text))

    # --- generation -----------------------------------------------------

    def generate(self, request: GenerationRequest) -> list[str]:
        return [output.text for output in self.generate_detailed(request)]

    def generate_detailed(self, request: GenerationRequest) -> tuple[GenerationOutput, ...]:
        self._validate_request(request)
        storage, scale_dtype, granularity = _resolve_kv_storage(request)
        if request.max_tokens == 0:
            # A zero-token request decodes nothing, so it costs no device work:
            # no weight load, no runner reset, no prefill, no selection. The MTP
            # route returns the same empty output before it touches the device,
            # and a request that asks for no tokens must not pay for a prefill
            # on one route and not the other. The context ceiling is a KV-capacity
            # bound and this path allocates no KV, so it is not consulted.
            with self._lock:
                self._require_open()
                outputs = tuple(
                    self._empty_output(request, row_index)
                    for row_index in range(len(request.prompts))
                )
                self.last_generation_outputs = outputs
            return outputs
        with self._lock:
            self._require_open()
            runner = self._ensure_runner(
                storage=storage, scale_dtype=scale_dtype, granularity=granularity
            )
            outputs: list[GenerationOutput] = []
            for row_index in range(len(request.prompts)):
                raise_if_generation_deadline_expired(request)
                prompt_ids = request.prompt_token_ids(row_index, self.tokenize)
                if not prompt_ids:
                    raise ValueError("Gemma 4 prompt produced no token IDs")
                if len(prompt_ids) + request.max_tokens > self.context_length:
                    raise ValueError(
                        f"Gemma 4 prompt ({len(prompt_ids)} tokens) plus max_tokens "
                        f"({request.max_tokens}) exceeds context_length {self.context_length}"
                    )
                started = time.perf_counter()
                runner.reset()
                graph_session = self._decode_graph_session_for(runner)
                stop_ids = set(request.stop_token_ids)
                configured_eos = self._configured_eos_ids(request)
                sampler = self._row_sampler(request, prompt_ids, row_index)
                host_sampler = sampler.processes_logits
                wants_logprobs = sampler.wants_logprobs
                if host_sampler:
                    selection = sampler.select_row(runner.forward(list(prompt_ids)))
                    token_id = selection.token_id
                else:
                    # The device argmax route (D10) brings home the token
                    # directly; identical tokens to np.argmax(forward(...)),
                    # pinned by the argmax battery and the campaign bench's
                    # public/instrumented path parity. The token is still
                    # observed: the stop matcher and the row's history advance
                    # on either route.
                    selection = None
                    token_id = runner.forward_argmax(list(prompt_ids))
                    sampler.observe(token_id)
                generated: list[int] = []
                token_logprobs: list[TokenLogprob] = []
                finish_reason = "length"
                eos_id: int | None = None
                stop_sequence: tuple[int, ...] = ()
                for step in range(request.max_tokens):
                    raise_if_generation_deadline_expired(request)
                    generated.append(token_id)
                    sampler.discard_undo_history()
                    if wants_logprobs:
                        token_logprobs.append(
                            gemma4_token_logprob(
                                self.tokenizer,
                                token_id,
                                None if selection is None else selection.logprob,
                                () if selection is None else selection.top_logprobs,
                            )
                        )
                    matched = sampler.stop_sequence_matched
                    if matched:
                        # A multi-token stop is matched against the emitted
                        # history, so it can complete on any step; the sequence
                        # itself is not part of the reply.
                        finish_reason = "stop"
                        stop_sequence = matched
                        break
                    if token_id in stop_ids:
                        finish_reason = "stop"
                        stop_sequence = (token_id,)
                        break
                    if (
                        not request.ignore_eos
                        and token_id in configured_eos
                    ):
                        finish_reason = "stop"
                        eos_id = token_id
                        break
                    if step + 1 < request.max_tokens:
                        # The decode graph replays the step's launches.
                        # Iteration 0 stays launched: the capture needs one
                        # normal single-token step first so every kernel it
                        # records is already loaded, and after a reset the
                        # first step is that warm step.
                        replay = graph_session is not None and step >= 1
                        if host_sampler:
                            selection = sampler.select_row(
                                graph_session.step(token_id)
                                if replay
                                else runner.forward([token_id])
                            )
                            token_id = selection.token_id
                        elif replay:
                            selection = None
                            token_id = graph_session.step_argmax(token_id)
                            sampler.observe(token_id)
                        else:
                            selection = None
                            token_id = runner.forward_argmax([token_id])
                            sampler.observe(token_id)
                # The matched stop sequence is held back from the reply; the
                # generated ids keep the whole emitted run, which is what the
                # shared stop-sequence handling reports too.
                visible = (
                    len(generated) - len(stop_sequence) if stop_sequence else len(generated)
                )
                outputs.append(
                    GenerationOutput(
                        text=self.tokenizer.decode(generated[:visible], skip_special=False),
                        token_logprobs=tuple(token_logprobs[:visible]),
                        generated_token_ids=tuple(generated),
                        finish_details=finish_details_with_sampling_state(
                            FinishDetails(
                                reason=finish_reason,
                                eos_token_id=eos_id,
                                stop_sequence=stop_sequence,
                                length_limit=(
                                    request.max_tokens if finish_reason == "length" else None
                                ),
                                sampler_mode=_ar_sampler_mode(host_sampler, sampler),
                                phase="decode" if sampler.thinking_budget is None else None,
                            ),
                            sampler,
                        ),
                    )
                )
                self.last_generation_seconds = time.perf_counter() - started
            self.last_generation_outputs = tuple(outputs)
            return self.last_generation_outputs

    def _configured_eos_ids(self, request: GenerationRequest) -> set[int]:
        """The end-of-generation ids this request stops on.

        Gemma 4 stops on a set rather than one id, so an unset ``eos_token_id``
        means the tokenizer's own set. Both routes resolve it here so the
        sampler's ``min_tokens`` floor and thinking-budget suppression hold back
        the same ids on either route.
        """

        return (
            set(self.tokenizer.stop_token_ids)
            if request.eos_token_id is None
            else {int(request.eos_token_id)}
        )

    def _row_sampler(
        self,
        request: GenerationRequest,
        prompt_ids: Sequence[int],
        row_index: int,
    ) -> Gemma4Sampler:
        """One row's sampler, with the request's whole stop set resolved.

        The decode loop and the zero-token path build their sampler here, so the
        sampler mode a request reports does not depend on which path answered it.
        """

        return Gemma4Sampler(
            request,
            prompt_ids,
            row_index=row_index,
            eos_token_ids=tuple(sorted(self._configured_eos_ids(request))),
        )

    def _empty_output(
        self, request: GenerationRequest, row_index: int
    ) -> GenerationOutput:
        """The reply to a request that asks for no tokens.

        The shape is the MTP route's zero-token reply: no text, no generated ids,
        no logprobs, and a ``length`` finish whose limit is the requested zero.
        The sampler is built only to name the sampler mode the same request would
        report from the decode loop, which is host-side work with no device call.
        """

        prompt_ids = request.prompt_token_ids(row_index, self.tokenize)
        if not prompt_ids:
            raise ValueError("Gemma 4 prompt produced no token IDs")
        sampler = self._row_sampler(request, prompt_ids, row_index)
        return GenerationOutput(
            text="",
            generated_token_ids=(),
            finish_details=FinishDetails(
                reason="length",
                length_limit=0,
                sampler_mode=_ar_sampler_mode(sampler.processes_logits, sampler),
            ),
        )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            # The provider's assistant head borrows the backbone embedding, so
            # it closes before the weights it reads are freed.
            provider = self.__dict__.get("_speculative_provider")
            if provider is not None:
                provider.close()
            if self._runner is not None:
                self._runner.close()
                self._runner = None
            if self._decode_graph_session is not None:
                self._decode_graph_session.close()
                self._decode_graph_session = None
                self._decode_graph_runner = None
            if self._weights is not None:
                self._weights.free()
                self._weights = None

    def _decode_graph_session_for(self, runner: "Gemma4Runner"):
        """The runner's whole-step decode graph session, or None to stay launched.

        One session per generator, keyed to the runner instance it captured
        (a KV-storage change rebuilds the runner, so the stale session is
        closed first). Returns None when the graph is disabled by env or when
        the runner's INT8 KV storage cannot be captured — the launched decode
        the runner already runs is the whole step in that case.
        """

        if not _gemma4_decode_graph_enabled():
            return None
        if self._decode_graph_runner is not runner:
            if self._decode_graph_session is not None:
                self._decode_graph_session.close()
            self._decode_graph_session = None
            self._decode_graph_runner = None
        if getattr(runner, "uses_int8_kv", False):
            # INT8 KV storage needs a checked device-to-host readback per
            # step, which a graph capture cannot hold. The launched path runs
            # the requested storage exactly; refusing capture here keeps the
            # explicit storage request loud instead of silently downgrading
            # it.
            return None
        if self._decode_graph_session is None:
            from hipengine.runtime.gemma4_decode_graph import Gemma4DecodeGraphSession

            self._decode_graph_session = Gemma4DecodeGraphSession(runner)
            self._decode_graph_runner = runner
        return self._decode_graph_session

    @property
    def verify_dense_exact(self) -> bool:
        """Whether a speculative verify block must equal one-token decode bitwise.

        The production profile verifies on the WMMA dense kernels, which agree
        with one-token decode within that profile's numerical envelope and cost
        less per verify block. Every other resolution -- strict, or a generator
        no profile was bound to -- keeps the verify rows exact.
        """

        return getattr(self, "execution_profile", None) != "production"

    def _resolve_prefill_attention_variants(self) -> tuple[str, ...] | None:
        """The prefill-attention variants the execution profile selects, if any.

        Resolved once, before the runner exists, so a profile decision cannot
        change between two prefill blocks of the same request. An explicit
        attribute set by a caller or by a profile binder wins.
        """

        if self.prefill_attention_variants is not None:
            return self.prefill_attention_variants
        from hipengine.generation.gemma4_gguf_profiles import (
            resolve_gemma4_prefill_attention_variants,
        )

        return resolve_gemma4_prefill_attention_variants(backend=self.backend) or None

    def _ensure_runner(
        self,
        *,
        storage: str = "bf16",
        scale_dtype: str = "fp16",
        granularity: str = "per_token_head",
    ) -> Gemma4Runner:
        if self._runner is not None:
            # Storage is fixed for a runner's lifetime. A request that changes
            # it rebuilds rather than silently reusing the previous mode, which
            # would give the caller a different cache than it asked for.
            if (
                self._runner.kv_storage_resolved == storage
                and (
                    storage != "int8_per_token_head"
                    or scale_dtype == self._runner.kv_scale_dtype_resolved.value
                )
            ):
                return self._runner
            self._runner.close()
            self._runner = None
        # Refuse an unservable context before loading the artifact. The attention
        # geometry comes from metadata alone, and the ceiling depends only on that
        # geometry, on context_length, on the selected storage and on the selected
        # attention paths, so discovering it after a multi-gigabyte load would
        # charge the user for the refusal. Gemma4Runner repeats the check when it
        # is constructed; this is the same check, earlier, and it names the
        # selected INT8 bound when INT8 is asked for rather than the BF16 one.
        gemma4_require_context_capacity(
            gemma4_text_config_from_reader(self.reader),
            self.context_length,
            kv_storage=storage,
            variants=self._resolve_prefill_attention_variants(),
        )
        started = time.perf_counter()
        weights = self._weights
        loaded_here = False
        if weights is None:
            weights = load_gemma4_device_weights(self.reader, backend=self.backend)
            loaded_here = True
        try:
            # A speculative verify reads one logits row per verified position, so
            # the projection buffer is sized for the attached provider's verifier
            # rows. The default of one row is what keeps a wide prefill from
            # sizing its logits scratch for the whole block, so this only grows
            # when a provider actually asked for it.
            runner = Gemma4Runner(
                weights=weights,
                capacity=self.context_length,
                max_logits_rows=max(1, int(self._speculative_max_logits_rows)),
                prefill_attention_variants=self._resolve_prefill_attention_variants(),
                decode_attention_variants=self.decode_attention_variants,
                exact_prefill_attention_variants=self.exact_prefill_attention_variants,
                # The production profile verifies a speculative block on the
                # WMMA dense kernels; every other profile keeps verify rows
                # bit-identical to one-token decode.
                verify_dense_exact=self.verify_dense_exact,
                kv_storage=storage,
                kv_scale_dtype=scale_dtype,
                kv_scale_granularity=granularity,
            )
        except BaseException:
            # A failure while building the runner releases weights loaded for
            # this call. Weights already cached by a prior successful load are
            # left for a later retry; only the new allocations are rolled back.
            if loaded_here:
                weights.free()
            raise
        self._weights = weights
        self._runner = runner
        self._load_seconds = time.perf_counter() - started
        return self._runner

    # --- speculative provider -------------------------------------------

    @property
    def supports_speculative(self) -> bool:
        return self._speculative_provider is not None

    def __getattr__(self, name: str) -> Any:
        """Materialise ``supports_speculative_mtp`` only when a provider is attached.

        ``SubmitPollTextGenerator.supports_speculative_mtp`` ends in
        ``bool(supports) and callable(...)``, so a provider-backed generator has
        to *declare* the attribute truthy for the legacy route to be reachable.
        But a class-level declaration is what
        ``tests/test_unit_gemma4_speculative_wiring.py`` pins against, and it is
        right to: the wrapper returns ``False`` outright when the attribute is
        present and falsy, so a declaration that reads ``False`` while no
        provider is attached would make the runner's staged hooks unreachable.

        Answering only when a provider is attached satisfies both. With none
        attached this raises ``AttributeError``, ``getattr(..., None)`` yields
        ``None``, and the wrapper falls through to the staged check exactly as it
        did before this generator grew a provider. With one attached it reads
        ``True`` and the legacy route is selected.
        """

        if name == "create_resident_model_runner" and callable(
            getattr(self.__dict__.get("_speculative_provider"), "generation_steps", None)
        ):
            from hipengine.generation.gemma4_resident import Gemma4ResidentModelRunner

            return lambda *, capacity=None: Gemma4ResidentModelRunner(self, capacity=capacity)
        if name == "supports_speculative_mtp":
            if self.__dict__.get("_speculative_provider") is not None:
                return True
        raise AttributeError(name)

    def attach_speculative_provider(self, provider: Any) -> None:
        """Attach one registry-resolved provider before target materialization.

        The provider is built against this generator, and it needs the backbone's
        embedding and output norm to construct the assistant head, so it must
        attach before the weights exist rather than after. What it may set here
        is the verifier row bound the runner is later constructed with.
        """

        if provider is None:
            raise TypeError("speculative provider must not be None")
        for name in ("generate_detailed", "stream_detailed", "capabilities", "close"):
            if not callable(getattr(provider, name, None)):
                raise TypeError(f"speculative provider must implement {name}()")
        with self._lock:
            if self._closed:
                raise RuntimeError("Gemma 4 generator is closed")
            if self._weights is not None:
                raise RuntimeError(
                    "speculative provider must attach before target materialization"
                )
            if self._speculative_provider is not None:
                raise RuntimeError("a speculative provider is already attached")
            declared = provider.capabilities().get("max_verifier_rows", 1)
            try:
                rows = int(declared)
            except (TypeError, ValueError) as error:
                raise TypeError(
                    "speculative provider max_verifier_rows must be an integer"
                ) from error
            if rows < 1:
                raise ValueError(
                    "speculative provider max_verifier_rows must be positive"
                )
            self._speculative_max_logits_rows = rows
            self._speculative_provider = provider

    def speculative_capabilities(self) -> dict[str, Any]:
        provider = self._speculative_provider
        return {} if provider is None else dict(provider.capabilities())

    @property
    def target_arch(self) -> str:
        """The gfx target the resolved backend runs on."""

        return hip_target_arch_for_backend(self.backend)

    def resolve_speculative_mtp_serving_plan(
        self,
        *,
        realized_group_rows: int,
        resident_capacity: int,
        candidate_budget: int,
        sampling_mode: str,
        kv_storage: str,
        memory_fit: bool,
        request_mode: str = "automatic",
    ):
        """Resolve the assistant-head serving scope before mutation.

        The key is mechanical request and runtime identity only: backend, gfx
        target, KV storage, realized group width, resident capacity, candidate
        depth, sampling mode, and memory fit. The model plugin owns the
        admission decision, which for Gemma 4 is a capability declaration rather
        than a retained evidence row. There is no artifact hash or model-name
        axis, so a newly resolvable artifact that fits the declaration runs.

        ``auto`` storage resolves to the provider's BF16 contract, the way the
        request validator resolves it before the runner is built. Any other
        storage keeps its own value so the declaration refuses it by name rather
        than silently routing to BF16.
        """

        from hipengine.speculative.serving import (
            SpeculativeMTPServingKey,
            resolve_speculative_mtp_serving_plan,
        )

        requested_kv = str(kv_storage or "auto")
        effective_kv = "bf16" if requested_kv == "auto" else requested_kv
        key = SpeculativeMTPServingKey(
            artifact_sha256=None,
            artifact_size_bytes=None,
            content_verified=False,
            backend=str(self.backend),
            target_arch=str(self.target_arch),
            weight_quant=str(
                getattr(self.model_plugin, "default_quant", _GEMMA4_QUANT)
                or _GEMMA4_QUANT
            ),
            kv_storage=effective_kv,
            kv_layout="uniform",
            realized_group_rows=int(realized_group_rows),
            resident_capacity=int(resident_capacity),
            candidate_budget=int(candidate_budget),
            sampling_mode=str(sampling_mode),
            memory_fit=bool(memory_fit),
        )
        resolver = getattr(
            self.model_plugin, "resolve_speculative_mtp_serving_plan", None
        )
        if callable(resolver):
            return resolver(key=key, request_mode=str(request_mode))
        return resolve_speculative_mtp_serving_plan(
            (),
            key=key,
            request_mode=str(request_mode),
        )

    def generate_speculative_detailed(
        self,
        request: GenerationRequest,
    ) -> list[GenerationOutput]:
        provider = self._speculative_provider
        if provider is None:
            raise NotImplementedError("Gemma 4 speculative provider is not configured")
        return list(provider.generate_detailed(request))

    def generate_speculative_mtp_detailed(
        self,
        request: GenerationRequest,
    ) -> list[GenerationOutput]:
        """The name ``engine_loop`` routes speculative MTP through.

        ``generate_speculative_detailed`` is the name the provider protocol and
        the ``LLM`` attach path use; this is the one the engine loop's capability
        probe looks for. Both delegate to the same provider.
        """

        return self.generate_speculative_detailed(request)

    def stream_speculative_detailed(
        self,
        request: GenerationRequest,
    ) -> Iterator[GenerationStreamChunk]:
        provider = self._speculative_provider
        if provider is None:
            raise NotImplementedError("Gemma 4 speculative provider is not configured")
        for chunk in provider.stream_detailed(request):
            yield GenerationStreamChunk.from_value(chunk)

    def stream_speculative_mtp_detailed(
        self,
        request: GenerationRequest,
    ) -> Iterator[GenerationStreamChunk]:
        """The name the server's speculative-MTP streaming route calls.

        The provider commits a whole draft/verify cycle at a time, so its
        streaming output is the same buffered chunk the blocking route returns;
        the streaming surface exists so a streamed MTP request reaches the
        provider instead of failing with a missing-method error. The provider
        itself refuses more than one prompt, which is the single-request
        constraint the route carries.
        """

        return self.stream_speculative_detailed(request)

    @staticmethod
    def _validate_request(request: GenerationRequest) -> None:
        """Refuse a control this generator cannot execute, before any mutation.

        Every refusal here is a capability miss, not a missing measurement, and
        the check runs before the runner is built or reset, so a request that
        asks for one costs no device work and leaves no state behind. The
        sampler's own field preconditions are validated here too: they are part
        of the request, and discovering a bad one after a multi-gigabyte weight
        load would charge the user for the refusal.
        """

        validate_sampling_params(request)
        blockers: list[str] = []
        if request.forced_tokens_pending or request.post_thinking_forced_tokens_pending:
            blockers.append("forced-token queues are not implemented")
        if (
            request.tool_call_constraint is not None
            or request.json_object_close_forcing
            or request.grammar is not None
        ):
            blockers.append("structured constraints are not implemented")
        if request.force_sequence_completion_token_sequences:
            blockers.append("forced sequence completion is not implemented")
        if request.kv_storage not in {"auto", "bf16", "int8_per_token_head"}:
            blockers.append(
                f"KV storage {request.kv_storage!r} is not implemented "
                "(bf16 or int8_per_token_head)"
            )
        elif request.kv_storage == "int8_per_token_head":
            if request.kv_scale_granularity != "per_token_head":
                blockers.append(
                    "INT8 KV storage implements per_token_head scale granularity only"
                )
            if request.kv_scale_dtype not in {"fp16", "fp32"}:
                blockers.append("INT8 KV storage implements fp16 or fp32 scales only")
        if blockers:
            raise NotImplementedError("Gemma 4 basic runner: " + "; ".join(blockers))

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("Gemma 4 generator is closed")


def make_gemma4_generator(
    *,
    backend: str,
    model_path: str | Path,
    weight_index: WeightIndex,
    model_plugin: Any,
    max_sequence_length: int | None = None,
) -> Gemma4GGUFGenerator:
    """Create the Gemma 4 generator for one concrete HIP backend.

    The runner drives arch-native kernels: the build takes its ``--offload-arch``
    from ``HIPENGINE_HIP_ARCH`` or the host device, and the backend key only
    selects the registry keys the weight materializer and the selected-expert
    dispatch resolve. gfx1100 and gfx1151 share the same gfx11 source lineage,
    so one factory serves both and the backend decides which registrations are
    looked up.

    ``max_sequence_length`` is the public ``LLM`` context limit and sizes the
    runner's KV cache. Prefill scratch is bounded separately by its block size.
    An omitted limit uses the generator's 8192-token default.
    """

    context_length = _GEMMA4_DEFAULT_CONTEXT
    if max_sequence_length is not None:
        context_length = int(max_sequence_length)
    return Gemma4GGUFGenerator(
        model_path=model_path,
        weight_index=weight_index,
        model_plugin=model_plugin,
        backend=backend,
        context_length=context_length,
    )


def make_gemma4_generator_gfx1100(
    *,
    model_path: str | Path,
    weight_index: WeightIndex,
    model_plugin: Any,
    max_sequence_length: int | None = None,
) -> Gemma4GGUFGenerator:
    """Create the gfx1100 Gemma 4 generator."""

    return make_gemma4_generator(
        backend="hip_gfx1100",
        model_path=model_path,
        weight_index=weight_index,
        model_plugin=model_plugin,
        max_sequence_length=max_sequence_length,
    )


def make_gemma4_generator_gfx1151(
    *,
    model_path: str | Path,
    weight_index: WeightIndex,
    model_plugin: Any,
    max_sequence_length: int | None = None,
) -> Gemma4GGUFGenerator:
    """Create the gfx1151 (Strix Halo) Gemma 4 generator.

    The gfx1151 kernel package aliases the proven gfx1100 gfx11 bodies and
    compiles them for gfx1151, so the same runner runs on Strix Halo.
    """

    return make_gemma4_generator(
        backend="hip_gfx1151",
        model_path=model_path,
        weight_index=weight_index,
        model_plugin=model_plugin,
        max_sequence_length=max_sequence_length,
    )


for _backend, _factory in (
    ("hip_gfx1100", make_gemma4_generator_gfx1100),
    ("hip_gfx1151", make_gemma4_generator_gfx1151),
):
    register_text_generator(
        model="gemma4_gguf",
        backend=_backend,
        quant=_GEMMA4_QUANT,
        factory=_factory,
    )

# Register the execution-profile plans with the generators they belong to.
# ``LLM`` resolves a profile before it constructs a generator, so a plan that is
# registered any later than this import is a plan the engine never sees. Doing
# it here also keeps the two in step: whoever can build the generator can
# resolve its profile. Idempotent, and it registers nothing for a combination
# that has no plan.
from hipengine.generation.gemma4_gguf_profiles import (  # noqa: E402
    register_gemma4_gguf_profiles as _register_gemma4_gguf_profiles,
)
from hipengine.generation.gemma4_profiles import (  # noqa: E402
    register_gemma4_gguf_profiles as _register_gemma4_gfx1100_profiles,
)

_register_gemma4_gguf_profiles()
_register_gemma4_gfx1100_profiles()


__all__ = [
    "Gemma4GGUFGenerator",
    "make_gemma4_generator",
    "make_gemma4_generator_gfx1100",
    "make_gemma4_generator_gfx1151",
]
