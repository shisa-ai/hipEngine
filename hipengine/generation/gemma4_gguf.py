"""Public greedy generation for Gemma 4 GGUF artifacts.

This is the adapter between ``LLM.generate()`` and
:class:`hipengine.runtime.gemma4.Gemma4Runner`. It owns the request protocol -
tokenization, the decode loop, stop conditions, and the output shape - and
delegates every forward pass to the runner.

Only the greedy path is implemented. ``_validate_request`` names each unimplemented
sampling feature rather than silently ignoring it, so a request that asks for
temperature or top-p fails loudly instead of quietly returning greedy output.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from hipengine.generation.deadline import raise_if_generation_deadline_expired
from hipengine.generation.registry import (
    FinishDetails,
    GenerationOutput,
    GenerationRequest,
    GenerationStreamChunk,
    register_text_generator,
)
from hipengine.kernels.backends import resolve_backend
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
    supports_stream_logprobs = False
    chat_template_family = "gemma"
    reasoning_parser_name = "gemma_tags"
    tool_parser_name = "unsupported"

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
        with self._lock:
            self._require_open()
            runner = self._ensure_runner()
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
                logits = runner.forward(list(prompt_ids))
                generated: list[int] = []
                finish_reason = "length"
                eos_id: int | None = None
                stop_ids = set(request.stop_token_ids)
                configured_eos = (
                    set(self.tokenizer.stop_token_ids)
                    if request.eos_token_id is None
                    else {int(request.eos_token_id)}
                )
                for step in range(request.max_tokens):
                    raise_if_generation_deadline_expired(request)
                    token_id = runner.next_token(logits)
                    generated.append(token_id)
                    if token_id in stop_ids:
                        finish_reason = "stop"
                        eos_id = token_id
                        break
                    if (
                        not request.ignore_eos
                        and token_id in configured_eos
                    ):
                        finish_reason = "stop"
                        eos_id = token_id
                        break
                    if step + 1 < request.max_tokens:
                        logits = runner.forward([token_id])
                outputs.append(
                    GenerationOutput(
                        text=self.tokenizer.decode(generated, skip_special=False),
                        generated_token_ids=tuple(generated),
                        finish_details=FinishDetails(
                            reason=finish_reason,
                            eos_token_id=eos_id,
                            length_limit=(
                                request.max_tokens if finish_reason == "length" else None
                            ),
                            sampler_mode="greedy",
                            phase="decode",
                        ),
                    )
                )
                self.last_generation_seconds = time.perf_counter() - started
            self.last_generation_outputs = tuple(outputs)
            return self.last_generation_outputs

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._runner is not None:
                self._runner.close()
                self._runner = None
            if self._weights is not None:
                self._weights.free()
                self._weights = None

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

    def _ensure_runner(self) -> Gemma4Runner:
        if self._runner is not None:
            return self._runner
        # Refuse an unservable context before loading the artifact. The attention
        # geometry comes from metadata alone, and the ceiling depends only on that
        # geometry and on context_length, so discovering it after a multi-gigabyte
        # load would charge the user for the refusal. Gemma4Runner repeats the
        # check when it is constructed; this is the same check, earlier.
        gemma4_require_context_capacity(
            gemma4_text_config_from_reader(self.reader), self.context_length
        )
        started = time.perf_counter()
        weights = load_gemma4_device_weights(self.reader, backend=self.backend)
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
            )
        except BaseException:
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

    @staticmethod
    def _validate_request(request: GenerationRequest, *, greedy_top_k: bool = False) -> None:
        blockers: list[str] = []
        if request.temperature != 0.0:
            blockers.append("temperature must be 0")
        allowed_top_k = (0, 1) if greedy_top_k else (0,)
        if request.top_p != 1.0 or request.top_k not in allowed_top_k or request.min_p != 0.0:
            blockers.append("top-p/top-k/min-p sampling is not implemented")
        if (
            request.repetition_penalty != 1.0
            or request.presence_penalty != 0.0
            or request.frequency_penalty != 0.0
        ):
            blockers.append("logit penalties are not implemented")
        if request.logit_bias or request.suppress_token_ids:
            blockers.append("logit bias/suppression is not implemented")
        if request.stop_token_sequences:
            blockers.append("multi-token stop sequences are not implemented")
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
        if request.thinking_hard_token_cap is not None or request.thinking_soft_close_window:
            blockers.append("thinking budget controls are not implemented")
        if request.min_tokens or request.logprobs or request.top_logprobs:
            blockers.append("min_tokens/logprobs are not implemented")
        if request.kv_storage not in {"auto", "bf16"}:
            blockers.append("only BF16 KV storage is implemented")
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

_register_gemma4_gguf_profiles()


__all__ = [
    "Gemma4GGUFGenerator",
    "make_gemma4_generator",
    "make_gemma4_generator_gfx1100",
    "make_gemma4_generator_gfx1151",
]
