"""Surya OCR 2 text generator (cpu_reference backend, torch-free).

Implements the TextGenerator protocol for text-only prompts and the
multimodal detailed path (``generate_multimodal_detailed``) for OCR
requests, over the fp32 NumPy CPU reference in
``hipengine.kernels.cpu_reference.surya``.

Greedy decoding only: Surya OCR is a deterministic transcription model.
Supported request controls are ``max_tokens``, ``ignore_eos``,
``eos_token_id``, ``stop_token_ids``, ``deadline_at``, and
``cancellation_token``; any other sampling/constraint control must stay at
its neutral default or the request is rejected (see
``hipengine.generation.surya_contract``).

Registered under ``(surya_ocr2, cpu_reference, fp32)`` through
``register_builtin_generators()``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from hipengine.generation.deadline import raise_if_generation_deadline_expired
from hipengine.generation.registry import register_text_generator
from hipengine.generation.surya_contract import (
    greedy_decode_tokens,
    resolve_surya_greedy_settings,
)
from hipengine.kernels.cpu_reference.surya import (
    SuryaSpec,
    SuryaWeights,
    text_decode_step,
    text_prefill,
)
from hipengine.loading.surya import (
    SuryaTokenizer,
    compute_mrope_positions,
    load_surya_spec,
    load_surya_weights,
    preprocess_image_surya,
    render_chat_prompt,
    resolve_surya_path,
)

_QUANT = "fp32"


class SuryaOCRGenerator:
    """Greedy OCR generator over the torch-free CPU reference path."""

    def __init__(
        self,
        *,
        model_path: str | Path,
        weight_index: Any = None,
        model_plugin: Any = None,
        vision_model_path: str | Path | None = None,
    ) -> None:
        self.model_dir = resolve_surya_path(model_path)
        self.spec: SuryaSpec = load_surya_spec(self.model_dir)
        self.weights: SuryaWeights = load_surya_weights(self.model_dir)
        self.tokenizer = SuryaTokenizer(self.model_dir)
        self.model_plugin = model_plugin
        self.supports_vision = True
        self.speculative_candidate_budget = 0

    # -- TextGenerator protocol (text-only) --------------------------------

    def generate(self, request: Any) -> list[str]:
        settings = resolve_surya_greedy_settings(request, self.spec)
        return [
            self._generate_text_only(str(prompt), request, settings)
            for prompt in request.prompts
        ]

    def close(self) -> None:  # engine-loop lifecycle hook
        return None

    def detokenize(self, token_ids: Any, *, skip_special: bool = True) -> str:
        return self.tokenizer.decode(list(token_ids), skip_special=skip_special)

    # -- multimodal OCR path -----------------------------------------------

    def generate_multimodal_detailed(
        self, prompt: str, image: Any, request: Any
    ) -> Any:
        from hipengine.generation.registry import FinishDetails, GenerationOutput

        settings = resolve_surya_greedy_settings(request, self.spec)
        token_ids, finish_reason = self._generate_ocr(
            str(prompt), image, request, settings
        )
        return GenerationOutput(
            text=self.tokenizer.decode(token_ids, skip_special=True),
            finish_details=FinishDetails(
                reason=finish_reason,
                eos_token_id=next(iter(settings.eos_token_ids), None),
                length_limit=settings.max_tokens,
            ),
            generated_token_ids=tuple(token_ids),
        )

    # -- internals ----------------------------------------------------------

    def _decode_greedy(
        self,
        input_ids: np.ndarray,
        position_ids: np.ndarray,
        visual_features: np.ndarray | None,
        request: Any,
        settings: Any,
    ) -> tuple[list[int], str]:
        # An abandoned request must not pay for the prefill. The caller already
        # checked before vision, and the decode loop checks before every step.
        raise_if_generation_deadline_expired(request)
        hidden, state = text_prefill(
            self.weights,
            self.spec,
            input_ids,
            position_ids,
            visual_features=visual_features,
        )
        emb = self.weights["model.language_model.embed_tokens.weight"]
        logits = hidden[:, -1] @ emb.T
        pos = int(position_ids[:, -1].max())

        def step_fn(token_id: int, step: int) -> np.ndarray:
            return text_decode_step(
                self.weights, self.spec, token_id, state, pos + 1 + step
            )

        return greedy_decode_tokens(logits, settings, step_fn, request)

    def _generate_text_only(
        self, prompt: str, request: Any, settings: Any
    ) -> str:
        input_ids, mm = render_chat_prompt(self.tokenizer, prompt)
        position_ids = compute_mrope_positions(mm, None)
        ids, _ = self._decode_greedy(
            np.array([input_ids], dtype=np.int64),
            position_ids,
            None,
            request,
            settings,
        )
        return self.tokenizer.decode(ids, skip_special=True)

    def _generate_ocr(
        self, prompt: str, image: Any, request: Any, settings: Any
    ) -> tuple[list[int], str]:
        from hipengine.kernels.cpu_reference.surya import vision_forward

        # Fail before any work when the request is already abandoned, then
        # again before the vision tower. The prefill is checked inside
        # `_decode_greedy`.
        raise_if_generation_deadline_expired(request)
        pixel_rows, grid = preprocess_image_surya(image)
        _, _, merged = vision_forward(self.weights, self.spec, pixel_rows, [grid])
        n_image_tokens = (grid[1] // 2) * (grid[2] // 2)
        input_ids, mm = render_chat_prompt(
            self.tokenizer, prompt, n_image_tokens
        )
        position_ids = compute_mrope_positions(
            mm, grid, self.spec.vision_spatial_merge_size
        )
        raise_if_generation_deadline_expired(request)
        return self._decode_greedy(
            np.array([input_ids], dtype=np.int64),
            position_ids,
            merged[None],
            request,
            settings,
        )


def make_surya_generator_cpu(
    *,
    model_path: str | Path,
    weight_index: Any = None,
    model_plugin: Any = None,
    **_kwargs: Any,
) -> SuryaOCRGenerator:
    return SuryaOCRGenerator(
        model_path=model_path,
        weight_index=weight_index,
        model_plugin=model_plugin,
    )


register_text_generator(
    model="surya_ocr2",
    backend="cpu_reference",
    quant=_QUANT,
    factory=make_surya_generator_cpu,
)
