"""Top-level user API scaffolding.

The public API stays torch-free. Model-specific generation implementations are resolved
through a registry at call time so backend/quant choices do not become engine branches.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SamplingParams:
    """Sampling parameter container for the public API surface."""

    max_tokens: int = 16
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    logit_bias: Any = ()
    suppress_token_ids: tuple[int, ...] = ()
    min_tokens: int = 0
    eos_token_id: int | None = None
    stop_token_ids: tuple[int, ...] = ()
    stop_token_sequences: tuple[tuple[int, ...], ...] = ()
    forced_tokens_pending: tuple[int, ...] = ()
    forced_token_reason: str | None = None
    post_thinking_forced_tokens_pending: tuple[int, ...] = ()
    post_thinking_forced_token_reason: str | None = None
    force_sequence_completion_token_sequences: tuple[tuple[int, ...], ...] = ()
    force_sequence_completion_reason: str | None = None
    json_object_close_forcing: bool = False
    thinking_close_token_ids: tuple[int, ...] = ()
    thinking_hard_token_cap: int | None = None
    thinking_soft_close_window: int = 0
    ignore_eos: bool = False
    kv_storage: str = "auto"
    kv_scale_dtype: str = "fp16"
    kv_scale_granularity: str = "per_token_head"
    seed: int | None = None
    row_seeds: tuple[int, ...] = ()
    deadline_at: float | None = None
    cancellation_token: Any | None = field(default=None, compare=False, repr=False)
    logprobs: bool = False
    top_logprobs: int = 0

    def __post_init__(self) -> None:
        from hipengine.generation.sampling import normalize_logit_bias_pairs, normalize_stop_token_sequences, validate_sampling_params

        object.__setattr__(self, "max_tokens", int(self.max_tokens))
        object.__setattr__(self, "temperature", float(self.temperature))
        object.__setattr__(self, "top_p", float(self.top_p))
        object.__setattr__(self, "top_k", int(self.top_k))
        object.__setattr__(self, "min_p", float(self.min_p))
        object.__setattr__(self, "repetition_penalty", float(self.repetition_penalty))
        object.__setattr__(self, "presence_penalty", float(self.presence_penalty))
        object.__setattr__(self, "frequency_penalty", float(self.frequency_penalty))
        object.__setattr__(self, "logit_bias", normalize_logit_bias_pairs(self.logit_bias))
        object.__setattr__(self, "suppress_token_ids", tuple(int(token) for token in self.suppress_token_ids))
        object.__setattr__(self, "min_tokens", int(self.min_tokens))
        object.__setattr__(self, "eos_token_id", None if self.eos_token_id is None else int(self.eos_token_id))
        object.__setattr__(self, "stop_token_ids", tuple(int(token) for token in self.stop_token_ids))
        object.__setattr__(self, "stop_token_sequences", normalize_stop_token_sequences(self.stop_token_sequences))
        object.__setattr__(self, "forced_tokens_pending", tuple(int(token) for token in self.forced_tokens_pending))
        object.__setattr__(self, "forced_token_reason", None if self.forced_token_reason is None else str(self.forced_token_reason))
        object.__setattr__(
            self,
            "post_thinking_forced_tokens_pending",
            tuple(int(token) for token in self.post_thinking_forced_tokens_pending),
        )
        object.__setattr__(
            self,
            "post_thinking_forced_token_reason",
            None if self.post_thinking_forced_token_reason is None else str(self.post_thinking_forced_token_reason),
        )
        object.__setattr__(
            self,
            "force_sequence_completion_token_sequences",
            normalize_stop_token_sequences(self.force_sequence_completion_token_sequences),
        )
        object.__setattr__(
            self,
            "force_sequence_completion_reason",
            None if self.force_sequence_completion_reason is None else str(self.force_sequence_completion_reason),
        )
        object.__setattr__(self, "json_object_close_forcing", bool(self.json_object_close_forcing))
        object.__setattr__(
            self,
            "thinking_close_token_ids",
            tuple(int(token) for token in self.thinking_close_token_ids),
        )
        object.__setattr__(
            self,
            "thinking_hard_token_cap",
            None if self.thinking_hard_token_cap is None else int(self.thinking_hard_token_cap),
        )
        object.__setattr__(self, "thinking_soft_close_window", int(self.thinking_soft_close_window))
        object.__setattr__(self, "ignore_eos", bool(self.ignore_eos))
        object.__setattr__(self, "kv_storage", str(self.kv_storage))
        object.__setattr__(self, "kv_scale_dtype", str(self.kv_scale_dtype))
        object.__setattr__(self, "kv_scale_granularity", str(self.kv_scale_granularity))
        object.__setattr__(self, "seed", None if self.seed is None else int(self.seed))
        object.__setattr__(self, "row_seeds", tuple(int(seed) for seed in self.row_seeds))
        object.__setattr__(self, "deadline_at", None if self.deadline_at is None else float(self.deadline_at))
        object.__setattr__(self, "cancellation_token", self.cancellation_token)
        object.__setattr__(self, "logprobs", bool(self.logprobs))
        object.__setattr__(self, "top_logprobs", int(self.top_logprobs))
        validate_sampling_params(self)


class LLM:
    """Minimal public LLM API.

    Phase-0 generation currently resolves to narrow bring-up implementations registered by
    model/backend/quant. The default ``backend="auto"`` is resolved once to a concrete
    backend before registry lookup; unsupported keys fail explicitly instead of adding
    engine-level backend or quant conditionals.
    """

    def __init__(self, model: str, *, backend: str = "auto", quant: str = "fp16"):
        self.model = model
        self.backend = backend
        self.quant = quant
        self._resolved_backend: str | None = None
        self._weight_index: Any | None = None
        self._model_plugin: Any | None = None
        self._text_generator: Any | None = None

    def generate(
        self,
        prompts: str | Iterable[str],
        sampling_params: SamplingParams | None = None,
    ) -> list[str]:
        prompt_tuple = _normalize_prompts(prompts)
        if not prompt_tuple:
            return []
        return [output.text for output in self.generate_detailed(prompt_tuple, sampling_params)]

    def generate_detailed(
        self,
        prompts: str | Iterable[str],
        sampling_params: SamplingParams | None = None,
    ):
        """Return generated text plus optional per-token metadata."""

        from hipengine.generation import GenerationOutput

        prompt_tuple = _normalize_prompts(prompts)
        if not prompt_tuple:
            return []
        generator = self._get_text_generator()
        request = _generation_request(prompt_tuple, sampling_params or SamplingParams())
        detailed = getattr(generator, "generate_detailed", None)
        if callable(detailed):
            outputs = list(detailed(request))
        else:
            outputs = [GenerationOutput(text=str(item)) for item in generator.generate(request)]
        if len(outputs) != len(prompt_tuple):
            raise RuntimeError(f"generator returned {len(outputs)} outputs for {len(prompt_tuple)} prompts")
        return [output if isinstance(output, GenerationOutput) else GenerationOutput(text=str(output)) for output in outputs]

    def stream(
        self,
        prompt: str,
        sampling_params: SamplingParams | None = None,
    ) -> Iterator[str]:
        """Yield generated text chunks for a single prompt when supported."""

        for chunk in self.stream_detailed(prompt, sampling_params):
            yield str(chunk)

    def stream_detailed(
        self,
        prompt: str,
        sampling_params: SamplingParams | None = None,
    ):
        """Yield generated text chunks plus optional backend telemetry."""

        generator = self._get_text_generator()
        request = _generation_request((str(prompt),), sampling_params or SamplingParams())
        from hipengine.generation import GenerationStreamChunk

        detailed_streamer = getattr(generator, "stream_detailed", None)
        if callable(detailed_streamer):
            for chunk in detailed_streamer(request):
                yield GenerationStreamChunk.from_value(chunk)
            return
        streamer = getattr(generator, "stream", None)
        if callable(streamer):
            for chunk in streamer(request):
                yield GenerationStreamChunk.from_value(chunk)
            return

        for text in generator.generate(request):
            yield GenerationStreamChunk(text=str(text))

    @property
    def supports_stream_many(self) -> bool:
        """Whether the resolved generator advertises public multi-row streaming."""

        generator = self._text_generator
        if generator is None:
            return False
        return bool(
            getattr(generator, "supports_stream_many", False)
            or getattr(generator, "supports_stream_many_detailed", False)
        )

    def stream_many_detailed(
        self,
        prompts: str | Iterable[str],
        sampling_params: SamplingParams | None = None,
    ):
        """Yield row-indexed stream chunks for multiple prompts when supported."""

        prompt_tuple = _normalize_prompts(prompts)
        if not prompt_tuple:
            return
        generator = self._get_text_generator()
        detailed_streamer = getattr(generator, "stream_many_detailed", None)
        if not callable(detailed_streamer):
            raise NotImplementedError("multi-row streaming is not supported by this generator")
        request = _generation_request(prompt_tuple, sampling_params or SamplingParams())
        from hipengine.generation import GenerationStreamChunk

        for chunk in detailed_streamer(request):
            yield GenerationStreamChunk.from_value(chunk)

    def prepare(
        self,
        *,
        max_sequence_length: int | None = None,
        sampling_params: SamplingParams | None = None,
    ) -> int | None:
        """Eagerly prepare a resident session when the generator supports it.

        Passing ``max_sequence_length=None`` lets generators choose the largest
        context they can preallocate for the selected model/KV policy.
        """

        generator = self._get_text_generator()
        preparer = getattr(generator, "prepare", None)
        if not callable(preparer):
            return None
        return preparer(
            max_sequence_length=None if max_sequence_length is None else int(max_sequence_length),
            sampling_params=sampling_params or SamplingParams(),
        )

    def prepare_request_scratch(
        self,
        *,
        max_prompt_tokens: int,
        max_new_tokens: int = 0,
        sampling_params: SamplingParams | None = None,
        max_batch_size: int = 1,
        release_after_probe: bool = True,
    ) -> dict[str, Any] | None:
        """Ask the backend to allocate serving scratch for an admitted request shape.

        Backends that keep lazy prompt/decode workspaces may implement this hook
        so server startup can prove the selected resident context has enough
        transient headroom without decoding to the output limit.
        """

        generator = self._get_text_generator()
        preparer = getattr(generator, "prepare_request_scratch", None)
        if not callable(preparer):
            return None
        return preparer(
            max_prompt_tokens=max_prompt_tokens,
            max_new_tokens=max_new_tokens,
            sampling_params=sampling_params or SamplingParams(),
            max_batch_size=max_batch_size,
            release_after_probe=release_after_probe,
        )

    def count_tokens(self, text: str) -> int:
        """Return tokenizer token count when the resolved generator exposes one."""

        generator = self._get_text_generator()
        counter = getattr(generator, "count_tokens", None)
        if not callable(counter):
            raise NotImplementedError("token counting is not supported by this generator")
        return int(counter(str(text)))

    def tokenize(self, text: str) -> tuple[int, ...]:
        """Return tokenizer token ids when the resolved generator exposes them."""

        generator = self._get_text_generator()
        tokenizer = getattr(generator, "tokenize", None)
        if not callable(tokenizer):
            raise NotImplementedError("tokenization is not supported by this generator")
        return tuple(int(token) for token in tokenizer(str(text)))

    def detokenize(self, token_ids: Iterable[int], *, skip_special: bool = False) -> str:
        """Return text for token ids when the resolved generator exposes decoding."""

        ids = tuple(int(token) for token in token_ids)
        generator = self._get_text_generator()
        detokenizer = getattr(generator, "detokenize", None)
        if callable(detokenizer):
            try:
                return str(detokenizer(ids, skip_special=bool(skip_special)))
            except TypeError:
                return str(detokenizer(ids))
        tokenizer = getattr(generator, "tokenizer", None)
        decode = getattr(tokenizer, "decode", None)
        if callable(decode):
            try:
                return str(decode(ids, skip_special=bool(skip_special)))
            except TypeError:
                return str(decode(ids))
        raise NotImplementedError("detokenization is not supported by this generator")

    def _get_text_generator(self) -> Any:
        if self._text_generator is not None:
            return self._text_generator

        from hipengine.generation import SubmitPollTextGenerator, register_builtin_generators, resolve_text_generator

        register_builtin_generators()
        weight_index, model_plugin = self._load_model_metadata()
        backend = self._resolve_backend()
        factory = resolve_text_generator(
            model=model_plugin.name,
            backend=backend,
            quant=self.quant,
        )
        self._text_generator = SubmitPollTextGenerator(
            factory(
                model_path=self.model,
                weight_index=weight_index,
                model_plugin=model_plugin,
            )
        )
        return self._text_generator

    def _resolve_backend(self) -> str:
        if self._resolved_backend is not None:
            return self._resolved_backend

        from hipengine.kernels.backends import resolve_backend

        self._resolved_backend = resolve_backend(self.backend)
        return self._resolved_backend

    def _load_model_metadata(self) -> tuple[Any, Any]:
        if self._weight_index is not None and self._model_plugin is not None:
            return self._weight_index, self._model_plugin

        from hipengine.loading import discover_gguf_files, load_gguf_index, load_weight_index, resolve_model_path
        from hipengine.models import resolve_model

        model_path = resolve_model_path(self.model)
        if _looks_like_gguf_path(model_path):
            index = load_gguf_index(discover_gguf_files(model_path)[0])
            self.model = str(index.path)
            plugin = resolve_model(index.architecture or "")
        else:
            index = load_weight_index(self.model)
            # Store resolved filesystem path so downstream code (tokenizer, runner) gets a
            # real directory instead of an HF model ID string.
            self.model = str(index.model_path)
            plugin = resolve_model(_primary_architecture(index.config))
        self._weight_index = index
        self._model_plugin = plugin
        return index, plugin


def _generation_request(prompt_tuple: tuple[str, ...], params: SamplingParams):
    from hipengine.generation import GenerationRequest

    return GenerationRequest(
        prompts=prompt_tuple,
        max_tokens=params.max_tokens,
        temperature=params.temperature,
        top_p=params.top_p,
        top_k=params.top_k,
        min_p=params.min_p,
        repetition_penalty=params.repetition_penalty,
        presence_penalty=params.presence_penalty,
        frequency_penalty=params.frequency_penalty,
        logit_bias=params.logit_bias,
        suppress_token_ids=params.suppress_token_ids,
        min_tokens=params.min_tokens,
        eos_token_id=params.eos_token_id,
        stop_token_ids=params.stop_token_ids,
        stop_token_sequences=params.stop_token_sequences,
        forced_tokens_pending=params.forced_tokens_pending,
        forced_token_reason=params.forced_token_reason,
        post_thinking_forced_tokens_pending=params.post_thinking_forced_tokens_pending,
        post_thinking_forced_token_reason=params.post_thinking_forced_token_reason,
        force_sequence_completion_token_sequences=params.force_sequence_completion_token_sequences,
        force_sequence_completion_reason=params.force_sequence_completion_reason,
        json_object_close_forcing=params.json_object_close_forcing,
        thinking_close_token_ids=params.thinking_close_token_ids,
        thinking_hard_token_cap=params.thinking_hard_token_cap,
        thinking_soft_close_window=params.thinking_soft_close_window,
        ignore_eos=params.ignore_eos,
        kv_storage=params.kv_storage,
        kv_scale_dtype=params.kv_scale_dtype,
        kv_scale_granularity=params.kv_scale_granularity,
        seed=params.seed,
        row_seeds=params.row_seeds,
        deadline_at=params.deadline_at,
        cancellation_token=params.cancellation_token,
        logprobs=params.logprobs,
        top_logprobs=params.top_logprobs,
    )


def _looks_like_gguf_path(path: Path) -> bool:
    if path.is_file():
        return path.suffix.lower() == ".gguf"
    if path.is_dir():
        return any(path.glob("*.gguf"))
    return path.suffix.lower() == ".gguf"


def _normalize_prompts(prompts: str | Iterable[str]) -> tuple[str, ...]:
    if isinstance(prompts, str):
        return (prompts,)
    return tuple(str(prompt) for prompt in prompts)


def _primary_architecture(config: dict[str, Any]) -> str:
    text = config.get("text_config") if isinstance(config.get("text_config"), dict) else config
    architectures = config.get("architectures") or text.get("architectures") or ()
    if architectures:
        return str(architectures[0])
    model_type = str(text.get("model_type", config.get("model_type", "")))
    raise ValueError(f"checkpoint config for model_type {model_type!r} does not declare an architecture")
