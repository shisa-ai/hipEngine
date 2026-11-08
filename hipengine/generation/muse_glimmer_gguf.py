"""Public generation for Muse Glimmer GGUF text checkpoints.

Prefill-then-decode on the resident eager session
(:class:`hipengine.runtime.muse_glimmer.MuseGlimmerResidentSession`). Token
selection is the shared :func:`hipengine.generation.sampling.select_token`, so
temperature, top-k/top-p/min-p, logit bias, and the history penalties mean here
what they mean for every other generator. Requests this path cannot serve are
refused with a named capability instead of being quietly ignored.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from hipengine.generation.deadline import raise_if_generation_deadline_expired
from hipengine.generation.registry import (
    FinishDetails,
    GenerationOutput,
    GenerationRequest,
    register_text_generator,
)
from hipengine.generation.sampling import RowSamplingState, row_seed_for_index, select_token
from hipengine.kernels.backends import resolve_backend
from hipengine.loading.gguf import GGUFModelInfo, GGUFReader
from hipengine.loading.muse_glimmer_gguf import (
    MuseGlimmerLoadedWeights,
    config_from_gguf_info,
    materialize_muse_glimmer_weights,
)
from hipengine.runtime.muse_glimmer import MuseGlimmerResidentSession
from hipengine.tokenization.gguf import MuseGlimmerGGUFTokenizer

_MUSE_GLIMMER_QUANT = "gguf_q4_k_m"
_MUSE_GLIMMER_RESIDENT_PATH = "muse_glimmer_gguf_resident_eager"
#: The production llama.cpp launcher serves this model at 8192 positions; the
#: artifact declares 131072 and the cache admits any positive bound up to it.
_DEFAULT_CONTEXT = 8192
_LOG = logging.getLogger(__name__)


@dataclass
class MuseGlimmerGGUFGenerator:
    """Muse Glimmer dense decoder with resident weights and a per-request KV."""

    model_path: str | Path
    weight_index: GGUFModelInfo
    model_plugin: Any
    backend: str = "hip_gfx1151"
    context_length: int | None = None
    execution_profile: str | None = None
    vision_model_path: str | Path | None = None
    tokenizer: MuseGlimmerGGUFTokenizer = field(init=False)
    max_prompt_tokens: int = field(init=False)
    _loaded: MuseGlimmerLoadedWeights | None = field(default=None, init=False, repr=False)
    _session: MuseGlimmerResidentSession | None = field(default=None, init=False, repr=False)
    _vision_runner: Any = field(default=None, init=False, repr=False)
    _vision_reader: Any = field(default=None, init=False, repr=False)
    _speculative_provider: Any = field(default=None, init=False, repr=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)
    _load_seconds: float | None = field(default=None, init=False, repr=False)
    last_generation_seconds: float | None = field(default=None, init=False, repr=False)
    last_prefill_seconds: float | None = field(default=None, init=False, repr=False)
    last_decode_seconds: float | None = field(default=None, init=False, repr=False)

    supports_speculative_mtp = False
    supports_stream_many = False
    supports_resident_session_kv = False
    supports_stream_logprobs = False
    server_plain_ar_max_active_requests = 1
    chat_template_family = "muse-glimmer"
    execution_path = _MUSE_GLIMMER_RESIDENT_PATH

    def __post_init__(self) -> None:
        self.model_path = Path(self.weight_index.path).expanduser().resolve()
        self.backend = resolve_backend(self.backend)
        self.config = config_from_gguf_info(self.weight_index)
        self.execution_profile = self._validate_execution_profile(self.execution_profile)
        self.context_length = (
            _DEFAULT_CONTEXT if self.context_length is None
            else self._validate_context(self.context_length)
        )
        self.tokenizer = MuseGlimmerGGUFTokenizer.from_gguf_info(self.weight_index)
        self.max_prompt_tokens = self.context_length
        if self.vision_model_path is not None:
            self.vision_model_path = Path(self.vision_model_path).expanduser().resolve()
            if not self.vision_model_path.exists():
                raise FileNotFoundError(f"vision model not found: {self.vision_model_path}")

    @property
    def supports_vision(self) -> bool:
        return self.vision_model_path is not None

    # -- speculative attachment --------------------------------------------
    @property
    def supports_speculative(self) -> bool:
        return self._speculative_provider is not None

    def attach_speculative_provider(self, provider: Any) -> None:
        """Attach one registry-resolved provider before target materialization."""

        if provider is None:
            raise TypeError("speculative provider must not be None")
        for name in ("generate_detailed", "stream_detailed", "capabilities", "close"):
            if not callable(getattr(provider, name, None)):
                raise TypeError(f"speculative provider must implement {name}()")
        with self._lock:
            if self._closed:
                raise RuntimeError("the Muse Glimmer generator is closed")
            if self._session is not None:
                raise RuntimeError("speculative provider must attach before target materialization")
            if self._speculative_provider is not None:
                raise RuntimeError("a speculative provider is already attached")
            self._speculative_provider = provider

    def speculative_capabilities(self) -> dict[str, Any]:
        provider = self._speculative_provider
        return {} if provider is None else dict(provider.capabilities())

    def generate_speculative_detailed(self, request) -> list:
        provider = self._speculative_provider
        if provider is None:
            raise NotImplementedError("Muse Glimmer speculative provider is not configured")
        return list(provider.generate_detailed(request))

    def stream_speculative_detailed(self, request):
        from hipengine.generation.registry import GenerationStreamChunk

        provider = self._speculative_provider
        if provider is None:
            raise NotImplementedError("Muse Glimmer speculative provider is not configured")
        for chunk in provider.stream_detailed(request):
            yield GenerationStreamChunk.from_value(chunk)

    @staticmethod
    def _validate_execution_profile(value: str | None) -> str | None:
        if value is None:
            return None
        from hipengine.execution_profiles import resolve_requested_execution_profile

        return resolve_requested_execution_profile(value).value

    def _validate_context(self, tokens: int) -> int:
        context = int(tokens)
        if context <= 0:
            raise ValueError("context_length must be positive")
        # The artifact's declared context is its trained length, not a kernel
        # bound: RoPE runs only on the window-2048 SWA layers (relative
        # distance never exceeds the window), the global layers are NoPE, and
        # the rope tables are sized to the requested context. Memory is the
        # capability limit and the cache allocation surfaces it. A request
        # beyond the trained length runs; output quality there is unmeasured.
        declared = int(self.weight_index.metadata.get(
            "muse-glimmer.context_length", 131072
        ))
        if context > declared:
            _LOG.warning(
                "Muse Glimmer context_length %d exceeds the artifact's trained %d "
                "positions; running without position scaling (NoPE global layers, "
                "window-bounded RoPE)", context, declared,
            )
        return context

    # -- serving hooks -----------------------------------------------------
    def declare_max_sequence_length(self, tokens: int) -> None:
        requested = self._validate_context(tokens)
        if self._session is not None:
            if requested != self.context_length:
                raise RuntimeError(
                    "the Muse Glimmer context cannot change after the resident cache exists"
                )
            return
        self.context_length = requested
        self.max_prompt_tokens = requested

    def declare_serving_capacity(self, capacity: int | None) -> None:
        if capacity is not None and int(capacity) > 1:
            raise NotImplementedError(
                f"the Muse Glimmer bring-up path serves 1 resident request, not "
                f"{int(capacity)}; batched decode needs the paged attention path"
            )

    # -- text --------------------------------------------------------------
    def tokenize(self, text: str) -> tuple[int, ...]:
        return tuple(self.tokenizer.encode(str(text), add_special_tokens=True))

    def count_tokens(self, text: str) -> int:
        return len(self.tokenize(text))

    def decode(self, token_ids, *, skip_special: bool = False) -> str:
        return self.tokenizer.decode(token_ids, skip_special=skip_special)

    # -- lifecycle ---------------------------------------------------------
    def _ensure_session(self) -> MuseGlimmerResidentSession:
        if self._session is not None:
            return self._session
        started = time.perf_counter()
        reader = GGUFReader(self.model_path)
        loaded = materialize_muse_glimmer_weights(reader, backend=self.backend)
        try:
            session = MuseGlimmerResidentSession(
                loaded, max_context=self.context_length
            )
        except Exception:
            loaded.free()
            raise
        self._loaded = loaded
        self._session = session
        self._load_seconds = time.perf_counter() - started
        return session

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._speculative_provider is not None:
                self._speculative_provider.close()
                self._speculative_provider = None
            if self._vision_runner is not None:
                self._vision_runner.close()
                self._vision_runner = None
            self._vision_reader = None
            if self._session is not None:
                self._session.close()
                self._session = None
            if self._loaded is not None:
                self._loaded.free()
                self._loaded = None

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("the Muse Glimmer generator is closed")

    # -- generation --------------------------------------------------------
    def generate(self, request: GenerationRequest) -> list[str]:
        return [output.text for output in self.generate_detailed(request)]

    def generate_detailed(self, request: GenerationRequest) -> tuple[GenerationOutput, ...]:
        self._validate_request(request)
        with self._lock:
            self._require_open()
            prompt_ids = tuple(
                self._prompt_ids(request, row_index)
                for row_index in range(len(request.prompts))
            )
            for ids in prompt_ids:
                self._validate_prompt_length(len(ids), request.max_tokens)
            session = self._ensure_session()
            outputs: list[GenerationOutput] = []
            for row_index in range(len(request.prompts)):
                raise_if_generation_deadline_expired(request)
                outputs.append(
                    self._generate_row(session, request, row_index, prompt_ids[row_index])
                )
            return tuple(outputs)

    def _prompt_ids(self, request: GenerationRequest, row_index: int) -> Sequence[int]:
        prompt_ids = request.prompt_token_ids(row_index, self.tokenize)
        if not prompt_ids:
            raise ValueError("the prompt produced no token ids")
        return prompt_ids

    def _validate_prompt_length(self, prompt_tokens: int, max_tokens: int) -> None:
        if prompt_tokens > self.max_prompt_tokens:
            raise ValueError(
                f"prompt of {prompt_tokens} tokens exceeds the {self.max_prompt_tokens}-token "
                "serving context"
            )
        if prompt_tokens + max_tokens > self.context_length:
            raise ValueError(
                f"prompt of {prompt_tokens} tokens plus {max_tokens} generated "
                f"tokens exceeds the {self.context_length}-token context"
            )

    def _generate_row(
        self,
        session: MuseGlimmerResidentSession,
        request: GenerationRequest,
        row_index: int,
        prompt_ids: Sequence[int],
    ) -> GenerationOutput:
        started = time.perf_counter()
        session.reset()
        row_logits = np.asarray(session.prefill(prompt_ids), dtype=np.float32)

        state = RowSamplingState(
            prompt_tokens=prompt_ids,
            seed=row_seed_for_index(request, row_index),
            row_index=row_index,
        )
        generated: list[int] = []
        finish_reason = "length"
        eos_token_id: int | None = None
        stop_ids = set(request.stop_token_ids) | set(self.tokenizer.stop_token_ids)
        configured_eos = (
            self.tokenizer.eos_token_id if request.eos_token_id is None
            else int(request.eos_token_id)
        )
        sampler_mode = "greedy"

        decode_started = time.perf_counter()
        self.last_prefill_seconds = decode_started - started
        for step in range(request.max_tokens):
            raise_if_generation_deadline_expired(request)
            selection = select_token(
                row_logits,
                request,
                state,
                token_text_for_id=lambda token_id: self.tokenizer.decode([int(token_id)]),
            )
            sampler_mode = "greedy" if request.temperature <= 0.0 else str(selection.mode.value)
            token_id = int(selection.token_id)
            generated.append(token_id)
            if token_id in stop_ids:
                finish_reason = "stop"
                eos_token_id = token_id
                break
            if not request.ignore_eos and configured_eos is not None and token_id == configured_eos:
                finish_reason = "stop"
                eos_token_id = token_id
                break
            if step + 1 >= request.max_tokens:
                break
            row_logits = np.asarray(session.decode_step(token_id), dtype=np.float32)

        text = self.tokenizer.decode(generated, skip_special=False)
        self.last_decode_seconds = time.perf_counter() - decode_started
        self.last_generation_seconds = time.perf_counter() - started
        return GenerationOutput(
            text=text,
            generated_token_ids=tuple(generated),
            prompt_tokens=len(prompt_ids),
            finish_details=FinishDetails(
                reason=finish_reason,
                eos_token_id=eos_token_id,
                length_limit=request.max_tokens if finish_reason == "length" else None,
                sampler_mode=sampler_mode,
                phase="decode",
            ),
        )

    # -- multimodal --------------------------------------------------------
    def _ensure_vision_runner(self):
        if self._vision_runner is not None:
            return self._vision_runner
        if self.vision_model_path is None:
            raise NotImplementedError(
                "Muse Glimmer images require the mmproj GGUF via vision_model="
            )
        from hipengine.kernels.cpu_reference.muse_glimmer_vision import MuseGlimmerVisionConfig
        from hipengine.runtime.muse_glimmer_vision import MuseGlimmerVisionRunner

        self._vision_reader = GGUFReader(self.vision_model_path)
        config = MuseGlimmerVisionConfig.from_gguf_info(self._vision_reader.info)
        self._vision_runner = MuseGlimmerVisionRunner(
            self._vision_reader, config, backend=self.backend
        )
        return self._vision_runner

    def _load_image_rgb(self, value) -> np.ndarray:
        if isinstance(value, np.ndarray):
            array = np.asarray(value)
        else:
            from PIL import Image

            if isinstance(value, (str, Path)):
                image = Image.open(value)
            elif isinstance(value, Image.Image):
                image = value
            else:
                raise NotImplementedError(
                    "Muse Glimmer images accept an RGB array, PIL image, or file path"
                )
            array = np.asarray(image.convert("RGB"))
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError("Muse Glimmer images must be HWC RGB")
        return array.astype(np.uint8)

    def generate_multimodal_detailed(self, prompt, image, request):
        """Encode ordered images and run the normal sampler over image-aware KV."""

        from dataclasses import replace

        from hipengine.generation.muse_glimmer_multimodal import (
            muse_glimmer_embedding_overrides,
            muse_glimmer_image_prompt,
            muse_glimmer_media_images,
        )

        self._validate_request(request)
        if len(request.prompts) != 1:
            raise ValueError("Muse Glimmer image generation requires one prompt per request")
        with self._lock:
            self._require_open()
            images = [self._load_image_rgb(v) for v in muse_glimmer_media_images(image)]
            vision = self._ensure_vision_runner()
            features = [vision.encode(img, request=request) for img in images]
            ids, spans = muse_glimmer_image_prompt(
                prompt, [len(f) for f in features], self.tokenizer
            )
            if len(ids) + request.max_tokens > self.context_length:
                raise ValueError("Muse Glimmer image prompt plus max_tokens exceeds context_length")
            prepared = replace(request, prompts=(ids,))
            session = self._ensure_session()
            overrides = muse_glimmer_embedding_overrides(features, spans)
            with session.image_input(overrides):
                result = self.generate_detailed(prepared)[0]
            return replace(result, prompt_tokens=len(ids))

    # -- refusals ----------------------------------------------------------
    def _validate_request(self, request: GenerationRequest) -> None:
        blockers: list[str] = []
        if request.grammar is not None:
            blockers.append("grammar-constrained decoding")
        if request.tool_call_constraint is not None:
            blockers.append("tool-call constraints")
        if request.json_object_close_forcing:
            blockers.append("json_object close forcing")
        if request.thinking_hard_token_cap is not None or request.thinking_close_token_ids:
            blockers.append("thinking budgets")
        if request.forced_tokens_pending or request.post_thinking_forced_tokens_pending:
            blockers.append("forced-token queues")
        if request.stop_token_sequences:
            blockers.append("multi-token stop sequences")
        if request.force_sequence_completion_token_sequences:
            blockers.append("forced sequence completion")
        if request.logprobs or request.top_logprobs:
            blockers.append("logprobs")
        if request.resident_session_key is not None:
            blockers.append("resident session KV")
        if request.speculative_mtp_static_eligibility is not None or request.speculative_mtp_automatic:
            blockers.append("speculative MTP")
        if request.kv_storage not in {"auto", "bf16"}:
            blockers.append(f"kv_storage={request.kv_storage!r} (this path stores BF16)")
        if blockers:
            raise NotImplementedError(
                "the Muse Glimmer bring-up path does not implement: " + ", ".join(blockers)
            )


def make_muse_glimmer_gguf_generator_gfx1151(
    *,
    model_path: str | Path,
    weight_index: GGUFModelInfo,
    model_plugin: Any,
    max_sequence_length: int | None = None,
    execution_profile: str | None = None,
    vision_model_path: str | Path | None = None,
) -> MuseGlimmerGGUFGenerator:
    generator = MuseGlimmerGGUFGenerator(
        model_path=model_path,
        weight_index=weight_index,
        model_plugin=model_plugin,
        backend="hip_gfx1151",
        execution_profile=execution_profile,
        vision_model_path=vision_model_path,
    )
    if max_sequence_length is not None:
        generator.declare_max_sequence_length(int(max_sequence_length))
    return generator


register_text_generator(
    model="muse_glimmer_gguf",
    backend="hip_gfx1151",
    quant=_MUSE_GLIMMER_QUANT,
    factory=make_muse_glimmer_gguf_generator_gfx1151,
)


__all__ = [
    "MuseGlimmerGGUFGenerator",
    "make_muse_glimmer_gguf_generator_gfx1151",
]