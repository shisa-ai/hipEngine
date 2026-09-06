"""Top-level user API scaffolding.

The public API stays torch-free. Model-specific generation implementations are resolved
through a registry at call time so backend/quant choices do not become engine branches.
"""

from __future__ import annotations

import os
import inspect
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field, replace
from numbers import Integral
from pathlib import Path
from typing import Any

AUTO_QUANT = "auto"


def _factory_capacity_kwargs(factory, *, max_sequence_length, resident_capacity):
    """Forward configured limits only to factories that explicitly declare them."""
    try:
        parameters = inspect.signature(factory).parameters
    except (TypeError, ValueError):
        return {}
    return {
        name: value for name,value in (
            ("max_sequence_length",max_sequence_length),("resident_capacity",resident_capacity))
        if value is not None and name in parameters and parameters[name].kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,inspect.Parameter.KEYWORD_ONLY)
    }

_ENGINE_LOOP_GENERATOR_DEFAULT_ENVS = {
    "prefill_decode_policy": "HIPENGINE_PREFILL_DECODE_POLICY",
    "max_prefill_chunk_tokens": "HIPENGINE_MAX_PREFILL_CHUNK_TOKENS",
    "fair_prefill_burst_chunks": "HIPENGINE_FAIR_PREFILL_BURST_CHUNKS",
}


def _server_plain_ar_capacity(
    generator: Any,
    *,
    max_sequence_length: int | None,
) -> int | None:
    """Resolve registry-owned plain-AR residency for one serving context."""

    raw_default = getattr(generator, "server_plain_ar_max_active_requests", None)
    default = None if raw_default is None else int(raw_default)
    if default is not None and default < 1:
        raise ValueError("server_plain_ar_max_active_requests must be positive")
    raw_limits = getattr(
        generator,
        "server_plain_ar_max_active_requests_by_max_sequence_length",
        None,
    )
    if not raw_limits or max_sequence_length is None:
        return default
    if not isinstance(raw_limits, Mapping):
        raise TypeError(
            "server_plain_ar_max_active_requests_by_max_sequence_length must be a mapping"
        )
    requested = int(max_sequence_length)
    if requested <= 0:
        raise ValueError("max_sequence_length must be positive when set")
    limits: list[tuple[int, int]] = []
    for raw_context, raw_capacity in raw_limits.items():
        context = int(raw_context)
        capacity = int(raw_capacity)
        if context <= 0 or capacity <= 0:
            raise ValueError("plain-AR context limits and capacities must be positive")
        limits.append((context, capacity))
    for context, capacity in sorted(limits):
        if requested <= context:
            return capacity
    return default


def _engine_loop_config_with_generator_defaults(
    config: Any,
    generator: Any,
    *,
    environ: Mapping[str, str] | None = None,
):
    """Apply registry-selected defaults without overriding explicit env knobs."""

    defaults = getattr(generator, "engine_loop_config_defaults", None)
    if not defaults:
        return config
    if not isinstance(defaults, Mapping):
        raise TypeError("engine_loop_config_defaults must be a mapping")
    unknown = set(defaults) - set(_ENGINE_LOOP_GENERATOR_DEFAULT_ENVS)
    if unknown:
        raise ValueError(f"unsupported generator engine-loop defaults: {sorted(unknown)!r}")
    env = os.environ if environ is None else environ
    overrides = {
        name: value
        for name, value in defaults.items()
        if not str(env.get(_ENGINE_LOOP_GENERATOR_DEFAULT_ENVS[name], "")).strip()
    }
    return config if not overrides else replace(config, **overrides)


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
    tool_call_constraint: Any | None = None
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
    resident_session_key: str | None = field(default=None, compare=False, repr=False)
    resident_session_cache_action: str | None = field(
        default=None,
        compare=False,
        repr=False,
    )
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
        if self.tool_call_constraint is not None:
            from hipengine.generation.constraints import ToolCallConstraintSpec

            constraint = self.tool_call_constraint
            if not isinstance(constraint, ToolCallConstraintSpec):
                if not isinstance(constraint, Mapping):
                    raise TypeError("tool_call_constraint must be ToolCallConstraintSpec or a mapping")
                constraint = ToolCallConstraintSpec(**constraint)
            object.__setattr__(self, "tool_call_constraint", constraint)
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
        object.__setattr__(
            self,
            "resident_session_key",
            None if self.resident_session_key is None else str(self.resident_session_key),
        )
        object.__setattr__(
            self,
            "resident_session_cache_action",
            None
            if self.resident_session_cache_action is None
            else str(self.resident_session_cache_action),
        )
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

    def __init__(
        self,
        model: str,
        *,
        backend: str = "auto",
        quant: str = AUTO_QUANT,
        execution_profile: str | None = None,
        max_active_requests: int | None = None,
        max_sequence_length: int | None = None,
        prefix_cache: str | None = None,
        speculative_provider: str | None = None,
        draft_model: str | None = None,
        speculative_candidate_budget: int = 4,
        vision_model: str | None = None,
    ) -> None:
        if max_active_requests is not None and int(max_active_requests) <= 0:
            raise ValueError("max_active_requests must be positive when set")
        if max_sequence_length is not None and int(max_sequence_length) <= 0:
            raise ValueError("max_sequence_length must be positive when set")
        provider = (
            None
            if speculative_provider is None
            else str(speculative_provider).strip()
        )
        drafter = None if draft_model is None else str(draft_model).strip()
        if provider == "":
            raise ValueError("speculative_provider must be non-empty when set")
        if drafter == "":
            raise ValueError("draft_model must be non-empty when set")
        if provider is None and drafter is not None:
            raise ValueError("draft_model requires speculative_provider")
        if provider is not None and drafter is None:
            raise ValueError("speculative_provider requires draft_model")
        candidate_budget = int(speculative_candidate_budget)
        if candidate_budget <= 0:
            raise ValueError("speculative_candidate_budget must be positive")
        from hipengine.execution_profiles import resolve_requested_execution_profile

        requested_profile = resolve_requested_execution_profile(execution_profile)
        self.model = model
        self.backend = backend
        self.quant = quant
        self.execution_profile = (
            None if requested_profile is None else requested_profile.value
        )
        self.max_active_requests = (
            None if max_active_requests is None else int(max_active_requests)
        )
        self.max_sequence_length = (
            None if max_sequence_length is None else int(max_sequence_length)
        )
        self.speculative_provider = provider
        self.draft_model = drafter
        self.speculative_candidate_budget = candidate_budget
        self.vision_model = (
            None if vision_model is None else str(vision_model).strip()
        )
        if self.vision_model == "":
            raise ValueError("vision_model must be non-empty when set")
        if prefix_cache is None:
            self.prefix_cache = None
        else:
            from hipengine.kvcache import resolve_prefix_cache_mode

            self.prefix_cache = resolve_prefix_cache_mode(prefix_cache)
        self._resolved_backend: str | None = None
        self._resolved_quant: str | None = None
        self._resolved_execution_profile: Any | None = None
        self._weight_index: Any | None = None
        self._model_plugin: Any | None = None
        self._text_generator: Any | None = None

    def generate(
        self,
        prompts: Any,
        sampling_params: SamplingParams | None = None,
    ) -> list[str]:
        prompt_tuple = _normalize_prompts(prompts)
        if not prompt_tuple:
            return []
        return [output.text for output in self.generate_detailed(prompt_tuple, sampling_params)]

    def generate_detailed(
        self,
        prompts: Any,
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

    def generate_multimodal_detailed(
        self,
        prompt: str,
        image: Any,
        sampling_params: SamplingParams | None = None,
    ):
        """Generate one basic image+text request through a model-owned vision path."""

        from hipengine.generation import GenerationOutput

        generator = self._get_text_generator()
        detailed = getattr(generator, "generate_multimodal_detailed", None)
        if not callable(detailed):
            raise NotImplementedError("multimodal generation is not supported")
        params = sampling_params or SamplingParams()
        request = _generation_request((str(prompt),), params)
        output = detailed(str(prompt), image, request)
        return output if isinstance(output, GenerationOutput) else GenerationOutput(text=str(output))

    @property
    def supports_vision(self) -> bool:
        generator = self._text_generator
        return bool(generator is not None and getattr(generator, "supports_vision", False))

    def generate_speculative_mtp_detailed(
        self,
        prompts: Any,
        sampling_params: SamplingParams | None = None,
    ):
        """Return generated text through a model-owned speculative MTP route."""

        from hipengine.generation import GenerationOutput

        prompt_tuple = _normalize_prompts(prompts)
        if not prompt_tuple:
            return []
        generator = self._get_text_generator()
        supports = getattr(generator, "supports_speculative_mtp", None)
        if supports is not None and not bool(supports):
            raise NotImplementedError("speculative MTP generation is not supported by this generator")
        detailed = getattr(generator, "generate_speculative_mtp_detailed", None)
        if not callable(detailed):
            raise NotImplementedError("speculative MTP generation is not supported by this generator")
        request = _generation_request(prompt_tuple, sampling_params or SamplingParams())
        outputs = list(detailed(request))
        if len(outputs) != len(prompt_tuple):
            raise RuntimeError(f"generator returned {len(outputs)} MTP outputs for {len(prompt_tuple)} prompts")
        return [output if isinstance(output, GenerationOutput) else GenerationOutput(text=str(output)) for output in outputs]

    def stream_speculative_mtp_detailed(
        self,
        prompts: Any,
        sampling_params: SamplingParams | None = None,
    ):
        """Stream committed speculative output through the EngineService path."""

        prompt_tuple = _normalize_prompts(prompts)
        if len(prompt_tuple) != 1:
            raise ValueError("speculative streaming requires exactly one prompt")
        generator = self._get_text_generator()
        stream = getattr(generator, "stream_speculative_mtp_detailed", None)
        if not callable(stream):
            raise NotImplementedError("speculative streaming is not supported by this generator")
        request = _generation_request(prompt_tuple, sampling_params or SamplingParams())
        yield from stream(request)

    @property
    def supports_speculative_mtp(self) -> bool:
        """Whether the resolved generator exposes public speculative MTP generation."""

        generator = self._text_generator
        if generator is None:
            return False
        supports = getattr(generator, "supports_speculative_mtp", None)
        if supports is not None and not bool(supports):
            return False
        return callable(getattr(generator, "generate_speculative_mtp_detailed", None))

    def resolve_speculative_mtp_serving_plan(
        self,
        *,
        realized_group_rows: int,
        sampling_mode: str,
        context_tokens: int,
        output_horizon_tokens: int,
        kv_storage: str = "auto",
        memory_fit: bool = True,
    ):
        """Resolve one model-plugin serving plan before request mutation."""

        generator = self._get_text_generator()
        resolver = getattr(generator, "resolve_speculative_mtp_serving_plan", None)
        if not callable(resolver):
            return None
        manifest_sha256 = self.execution_profile_manifest_sha256 or ("0" * 64)
        resident_capacity = int(
            getattr(
                generator,
                "resident_capacity",
                self.max_active_requests if self.max_active_requests is not None else 32,
            )
        )
        return resolver(
            execution_profile_manifest_sha256=manifest_sha256,
            realized_group_rows=int(realized_group_rows),
            resident_capacity=resident_capacity,
            candidate_budget=int(self.speculative_candidate_budget),
            sampling_mode=str(sampling_mode),
            max_sequence_length=int(self.max_sequence_length or 1),
            context_tokens=int(context_tokens),
            output_horizon_tokens=int(output_horizon_tokens),
            kv_storage=str(kv_storage or "auto"),
            memory_fit=bool(memory_fit),
        )

    @property
    def speculative_mtp_serving_capability(self):
        """Resolve the loaded artifact against its model-plugin evidence scope."""

        _weight_index, model_plugin = self._load_model_metadata()
        evidence = tuple(
            getattr(model_plugin, "speculative_mtp_serving_evidence", ()) or ()
        )
        if not evidence:
            return None
        row = evidence[0]
        return self.resolve_speculative_mtp_serving_plan(
            realized_group_rows=int(row.realized_group_rows),
            sampling_mode=str(row.sampling_modes[0]),
            context_tokens=int(row.max_context_tokens),
            output_horizon_tokens=int(row.max_output_horizon_tokens),
            kv_storage=str(row.kv_storage),
            memory_fit=True,
        )

    @property
    def supports_default_mtp(self) -> bool:
        """Whether default-on MTP serving is safe for the resolved generator.

        True only for dense Qwen models whose MTP route is validated for
        serving (native verify by default, or the token-exact serial_exact
        rollback control); MoE and unsupported models report False so the
        server can keep them on plain AR unless a request explicitly opts in.
        """

        generator = self._text_generator
        if generator is None:
            return False
        return bool(getattr(generator, "supports_default_mtp", False))

    def generate_speculative_detailed(
        self,
        prompts: Any,
        sampling_params: SamplingParams | None = None,
    ):
        """Return output through the explicitly configured speculative provider."""

        from hipengine.generation import GenerationOutput

        prompt_tuple = _normalize_prompts(prompts)
        if not prompt_tuple:
            return []
        generator = self._get_text_generator()
        supports = getattr(generator, "supports_speculative", None)
        detailed = getattr(generator, "generate_speculative_detailed", None)
        if supports is not None and not bool(supports):
            raise NotImplementedError(
                "speculative generation is not supported by this generator"
            )
        if not callable(detailed):
            raise NotImplementedError(
                "speculative generation is not supported by this generator"
            )
        request = _generation_request(
            prompt_tuple,
            sampling_params or SamplingParams(),
        )
        outputs = list(detailed(request))
        if len(outputs) != len(prompt_tuple):
            raise RuntimeError(
                f"generator returned {len(outputs)} speculative outputs for "
                f"{len(prompt_tuple)} prompts"
            )
        return [
            output
            if isinstance(output, GenerationOutput)
            else GenerationOutput(text=str(output))
            for output in outputs
        ]

    def stream_speculative_detailed(
        self,
        prompt: Any,
        sampling_params: SamplingParams | None = None,
    ):
        """Yield chunks through the explicitly configured speculative provider."""

        from hipengine.generation import GenerationStreamChunk
        from hipengine.generation.registry import normalize_prompt_input

        generator = self._get_text_generator()
        supports = getattr(generator, "supports_speculative", None)
        streamer = getattr(generator, "stream_speculative_detailed", None)
        if supports is not None and not bool(supports):
            raise NotImplementedError(
                "speculative streaming is not supported by this generator"
            )
        if not callable(streamer):
            raise NotImplementedError(
                "speculative streaming is not supported by this generator"
            )
        request = _generation_request(
            (normalize_prompt_input(prompt),),
            sampling_params or SamplingParams(),
        )
        for chunk in streamer(request):
            yield GenerationStreamChunk.from_value(chunk)

    @property
    def supports_speculative(self) -> bool:
        """Whether an explicit public speculative provider is attached."""

        if self.speculative_provider is None:
            return False
        generator = self._get_text_generator()
        supports = getattr(generator, "supports_speculative", None)
        if supports is not None and not bool(supports):
            return False
        return callable(getattr(generator, "generate_speculative_detailed", None))

    @property
    def speculative_capabilities(self) -> dict[str, Any]:
        """Return truthful provider metadata without loading model weights."""

        if self.speculative_provider is None:
            return {}
        generator = self._get_text_generator()
        capabilities = getattr(generator, "speculative_capabilities", None)
        if not callable(capabilities):
            return {}
        payload = capabilities()
        if not isinstance(payload, Mapping):
            raise TypeError("speculative_capabilities must return a mapping")
        return dict(payload)

    def stream(
        self,
        prompt: Any,
        sampling_params: SamplingParams | None = None,
    ) -> Iterator[str]:
        """Yield generated text chunks for a single prompt when supported."""

        for chunk in self.stream_detailed(prompt, sampling_params):
            yield str(chunk)

    def stream_detailed(
        self,
        prompt: Any,
        sampling_params: SamplingParams | None = None,
    ):
        """Yield generated text chunks plus optional backend telemetry."""

        generator = self._get_text_generator()
        from hipengine.generation.registry import normalize_prompt_input

        request = _generation_request((normalize_prompt_input(prompt),), sampling_params or SamplingParams())
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

    def live_loop_snapshot(self) -> dict[str, object] | None:
        """Return live resident-loop observability without forcing model load."""

        generator = self._text_generator
        if generator is None:
            return None
        snapshot = getattr(generator, "live_loop_snapshot", None)
        if not callable(snapshot):
            return None
        payload = snapshot()
        return dict(payload) if isinstance(payload, dict) else None

    def drain_generation_cancellations(self) -> int:
        """Acknowledge queued resident cancellations without forcing model load."""

        generator = self._text_generator
        drainer = None if generator is None else getattr(generator, "drain_cancellations", None)
        if not callable(drainer):
            return 0
        return int(drainer())

    @property
    def server_plain_ar_max_active_requests(self) -> int | None:
        """Return the registry-selected plain-AR HTTP grouping capability."""

        generator = self._text_generator
        if generator is None:
            return None
        return _server_plain_ar_capacity(
            generator,
            max_sequence_length=self.max_sequence_length,
        )

    @property
    def supports_independent_generation(self) -> bool:
        """Whether one sole model service owns independently completing children."""

        generator = self._text_generator
        return bool(
            generator is not None
            and getattr(generator, "supports_independent_generation", False)
        )

    @property
    def supports_controlled_streaming(self) -> bool:
        """Whether streaming is driven by the shared submit/poll model loop."""

        generator = self._text_generator
        return bool(
            generator is not None
            and getattr(generator, "supports_controlled_streaming", False)
        )

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
        prompts: Any,
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

    def close(self) -> None:
        """Release the resolved generator's long-lived model resources."""

        generator = self._text_generator
        closer = None if generator is None else getattr(generator, "close", None)
        if callable(closer):
            closer()

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

        requested = (
            self.max_sequence_length
            if max_sequence_length is None
            else int(max_sequence_length)
        )
        if requested is not None and requested <= 0:
            raise ValueError("max_sequence_length must be positive when set")
        if self.max_sequence_length is not None and (
            requested is not None and requested > self.max_sequence_length
        ):
            raise ValueError(
                "prepare max_sequence_length exceeds the configured serving context"
            )
        if (
            self.max_sequence_length is None
            and self._text_generator is None
            and requested is not None
        ):
            self.max_sequence_length = requested
        generator = self._get_text_generator()
        preparer = getattr(generator, "prepare", None)
        if not callable(preparer):
            return None
        return preparer(
            max_sequence_length=requested,
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

        from hipengine.generation import (
            EngineService,
            SubmitPollTextGenerator,
            engine_loop_config_from_env,
            register_builtin_generators,
            resolve_text_generator,
        )

        register_builtin_generators()
        weight_index, model_plugin = self._load_model_metadata()
        backend = self._resolve_backend()
        quant = self._resolve_quant(model_plugin)
        factory = resolve_text_generator(
            model=model_plugin.name,
            backend=backend,
            quant=quant,
        )
        profile_resolution = self._resolve_execution_profile(
            model_plugin=model_plugin,
            backend=backend,
            quant=quant,
        )
        base_loop_config = engine_loop_config_from_env()
        factory_kwargs = {
            "model_path": self.model,
            "weight_index": weight_index,
            "model_plugin": model_plugin,
        }
        if self.vision_model is not None:
            factory_kwargs["vision_model_path"] = self.vision_model
        effective_factory = factory if profile_resolution is None else (profile_resolution.factory or factory)
        factory_kwargs.update(_factory_capacity_kwargs(
            effective_factory,max_sequence_length=self.max_sequence_length,
            resident_capacity=(self.max_active_requests if self.max_active_requests is not None
                               else base_loop_config.max_active_requests)))
        generator = (
            factory(**factory_kwargs)
            if profile_resolution is None
            else profile_resolution.construct_generator(factory, **factory_kwargs)
        )
        if self.speculative_provider is not None:
            from hipengine.speculative.registry import (
                SpeculativeProviderConfig,
                register_builtin_speculative_providers,
                resolve_speculative_provider,
            )

            register_builtin_speculative_providers()
            provider_factory = resolve_speculative_provider(
                provider=self.speculative_provider,
                target_model=model_plugin.name,
                backend=backend,
                quant=quant,
            )
            provider = provider_factory(
                target_generator=generator,
                config=SpeculativeProviderConfig(
                    provider=self.speculative_provider,
                    draft_model=self.draft_model or "",
                    candidate_budget=self.speculative_candidate_budget,
                ),
            )
            attach = getattr(generator, "attach_speculative_provider", None)
            if not callable(attach):
                closer = getattr(provider, "close", None)
                if callable(closer):
                    closer()
                raise TypeError(
                    "registered target generator cannot attach a speculative provider"
                )
            attach(provider)
        loop_config = _engine_loop_config_with_generator_defaults(
            engine_loop_config_from_env(),
            generator,
        )
        resident_capacity = self.max_active_requests
        registered_plain_ar_capacity = _server_plain_ar_capacity(
            generator,
            max_sequence_length=self.max_sequence_length,
        )
        has_resident_runner = callable(
            getattr(generator, "create_resident_model_runner", None)
        )
        if registered_plain_ar_capacity is not None and not has_resident_runner:
            # Non-resident compatibility generators still execute one static
            # call. Native resident services separate logical residency from
            # their registered physical kernel widths.
            resident_capacity = (
                registered_plain_ar_capacity
                if resident_capacity is None
                else min(resident_capacity, registered_plain_ar_capacity)
            )
        if resident_capacity is not None:
            loop_config = replace(
                loop_config,
                max_active_requests=resident_capacity,
            )
        if self.prefix_cache is not None:
            loop_config = replace(loop_config, prefix_cache=self.prefix_cache)
        resident_driver = SubmitPollTextGenerator(
            generator,
            config=loop_config,
        )
        self._text_generator = (
            EngineService(resident_driver, idle_wait_seconds=0.0)
            if resident_driver.supports_controlled_streaming
            else resident_driver
        )
        return self._text_generator

    def _resolve_backend(self) -> str:
        if self._resolved_backend is not None:
            return self._resolved_backend

        from hipengine.kernels.backends import resolve_backend

        self._resolved_backend = resolve_backend(self.backend)
        return self._resolved_backend

    @property
    def resolved_backend(self) -> str:
        """Return the concrete backend key selected for this process."""

        return self._resolve_backend()

    def _resolve_quant(self, model_plugin: Any) -> str:
        if self._resolved_quant is not None:
            return self._resolved_quant
        requested = str(self.quant or AUTO_QUANT).strip() or AUTO_QUANT
        if requested != AUTO_QUANT:
            self._resolved_quant = requested
            return requested
        default_quant = str(getattr(model_plugin, "default_quant", "") or "").strip()
        if not default_quant or default_quant == AUTO_QUANT:
            raise RuntimeError(
                f"model plugin {getattr(model_plugin, 'name', '<unknown>')!r} does not "
                "declare a concrete default_quant; pass quant= explicitly"
            )
        self._resolved_quant = default_quant
        return default_quant

    @property
    def resolved_quant(self) -> str:
        """Return the concrete quant key selected for this model."""

        if self._resolved_quant is None:
            _weight_index, model_plugin = self._load_model_metadata()
            self._resolve_quant(model_plugin)
        assert self._resolved_quant is not None
        return self._resolved_quant

    def _resolve_execution_profile(
        self,
        *,
        model_plugin: Any | None = None,
        backend: str | None = None,
        quant: str | None = None,
    ) -> Any | None:
        if self.execution_profile is None:
            return None
        if self._resolved_execution_profile is not None:
            return self._resolved_execution_profile
        if model_plugin is None:
            _weight_index, model_plugin = self._load_model_metadata()
        concrete_backend = self._resolve_backend() if backend is None else backend
        concrete_quant = self._resolve_quant(model_plugin) if quant is None else quant
        from hipengine.execution_profiles import resolve_runtime_profile

        self._resolved_execution_profile = resolve_runtime_profile(
            model=model_plugin.name,
            backend=concrete_backend,
            quant=concrete_quant,
            profile=self.execution_profile,
        )
        return self._resolved_execution_profile

    @property
    def resolved_execution_profile(self) -> str | None:
        """Return the explicit resolved profile, or ``None`` for legacy migration."""

        resolution = self._resolve_execution_profile()
        return None if resolution is None else resolution.profile.value

    @property
    def execution_profile_manifest(self) -> dict[str, Any] | None:
        """Return a plain copy of the immutable resolved variant manifest."""

        resolution = self._resolve_execution_profile()
        if resolution is None:
            return None
        from hipengine.execution_profiles import validate_variant_manifest

        return validate_variant_manifest(resolution.manifest)

    @property
    def execution_profile_manifest_sha256(self) -> str | None:
        """Return the resolved manifest hash without constructing a generator."""

        resolution = self._resolve_execution_profile()
        return None if resolution is None else resolution.manifest_sha256

    @property
    def execution_profile_strict_manifest_sha256(self) -> str | None:
        """Return the strict manifest hash for the resolved profile.

        Direct and server paths report both the selected and the strict
        immutable manifest hashes so provenance can distinguish a candidate
        route from its strict fallback baseline even after resolution.
        """

        resolution = self._resolve_execution_profile()
        return None if resolution is None else resolution.strict_manifest_sha256

    @property
    def execution_profile_fell_back_to_strict(self) -> bool | None:
        """Report whether any selected scope came from the strict plan."""

        resolution = self._resolve_execution_profile()
        return None if resolution is None else bool(resolution.fell_back_to_strict)

    def _load_model_metadata(self) -> tuple[Any, Any]:
        if self._weight_index is not None and self._model_plugin is not None:
            return self._weight_index, self._model_plugin

        from hipengine.loading import discover_gguf_files, load_gguf_index, load_weight_index, resolve_model_path
        from hipengine.models import resolve_model

        model_path = resolve_model_path(self.model)
        if _looks_like_gguf_path(model_path):
            gguf_files = discover_gguf_files(model_path)
            index = load_gguf_index(gguf_files[0])
            self.model = str(model_path if len(gguf_files) > 1 else index.path)
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


def _generation_request(prompt_tuple: tuple[Any, ...], params: SamplingParams):
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
        tool_call_constraint=params.tool_call_constraint,
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
        resident_session_key=params.resident_session_key,
        resident_session_cache_action=params.resident_session_cache_action,
        logprobs=params.logprobs,
        top_logprobs=params.top_logprobs,
    )


def _looks_like_gguf_path(path: Path) -> bool:
    if path.is_file():
        return path.suffix.lower() == ".gguf"
    if path.is_dir():
        return any(path.glob("*.gguf"))
    return path.suffix.lower() == ".gguf"


def _normalize_prompts(prompts: Any) -> tuple[Any, ...]:
    from hipengine.generation.registry import normalize_prompt_input

    if isinstance(prompts, str):
        return (prompts,)
    values = tuple(prompts)
    if values and all(isinstance(token, Integral) and not isinstance(token, bool) for token in values):
        return (normalize_prompt_input(values),)
    return tuple(normalize_prompt_input(prompt) for prompt in values)


def _primary_architecture(config: dict[str, Any]) -> str:
    text = config.get("text_config") if isinstance(config.get("text_config"), dict) else config
    architectures = config.get("architectures") or text.get("architectures") or ()
    if architectures:
        return str(architectures[0])
    model_type = str(text.get("model_type", config.get("model_type", "")))
    raise ValueError(f"checkpoint config for model_type {model_type!r} does not declare an architecture")
