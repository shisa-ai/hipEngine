"""Surya OCR 2 text generator (hip_gfx1151 backend, torch-free).

End-to-end HIP execution of the Surya OCR pipeline on gfx1100-class
devices: the vision tower, feature injection, prefill, and decode all run
through ``hipengine.runtime.surya.SuryaGpuRunner``. Preprocessing (resize,
normalize, patchify, mRoPE tables) stays on the host — it is data
preparation, not model math, and keeps the runtime torch-free.

Greedy decoding only: Surya OCR is a deterministic transcription model.
Supported request controls are ``max_tokens``, ``ignore_eos``,
``eos_token_id``, ``stop_token_ids``, ``deadline_at``, and
``cancellation_token``; any other sampling/constraint control must stay at
its neutral default or the request is rejected (see
``hipengine.generation.surya_contract``). Prompt plus output capacity is
validated against the runner context before any device work runs.

Registered under ``(surya_ocr2, hip_gfx1151, fp32)`` through
``register_builtin_generators()``. Correctness contract: greedy IDs must
match the CPU reference path, which is itself gated against the torch
fp32 oracle fixtures.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from hipengine.generation.registry import register_text_generator
from hipengine.generation.surya_contract import (
    check_prompt_capacity,
    greedy_decode_tokens,
    resolve_surya_greedy_settings,
)
from hipengine.kernels.cpu_reference.surya import SuryaSpec, SuryaWeights
from hipengine.loading.surya import (
    SuryaTokenizer,
    compute_mrope_positions,
    load_surya_spec,
    load_surya_weights,
    preprocess_image_surya,
    render_chat_prompt,
    resolve_surya_path,
)
from hipengine.runtime.surya import SuryaGpuRunner

_QUANT = "fp32"


class SuryaOCRGeneratorGPU:
    """Greedy OCR generator running vision + text on the HIP device."""

    def __init__(
        self,
        *,
        model_path: str | Path,
        weight_index: Any = None,
        model_plugin: Any = None,
        vision_model_path: str | Path | None = None,
        max_seq: int = 2048,
    ) -> None:
        self.model_dir = resolve_surya_path(model_path)
        self.spec: SuryaSpec = load_surya_spec(self.model_dir)
        weights = load_surya_weights(self.model_dir)
        self.tokenizer = SuryaTokenizer(self.model_dir)
        self.runner = SuryaGpuRunner(weights, self.spec, max_seq=max_seq)
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

    def close(self) -> None:
        self.runner.close()

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
        check_prompt_capacity(
            int(input_ids.shape[-1]), settings.max_tokens, self.runner.max_seq
        )
        logits = self.runner.prefill(
            input_ids[0], position_ids, visual_features=visual_features
        )
        pos = int(position_ids[:, -1].max())

        def step_fn(token_id: int, step: int) -> np.ndarray:
            return self.runner.decode_step(token_id, pos + 1 + step)

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
        pixel_rows, grid = preprocess_image_surya(image)
        merged = self.runner.vision_forward(pixel_rows, [grid])
        n_image_tokens = (grid[1] // 2) * (grid[2] // 2)
        input_ids, mm = render_chat_prompt(
            self.tokenizer, prompt, n_image_tokens
        )
        position_ids = compute_mrope_positions(
            mm, grid, self.spec.vision_spatial_merge_size
        )
        return self._decode_greedy(
            np.array([input_ids], dtype=np.int64),
            position_ids,
            merged,
            request,
            settings,
        )


def make_surya_generator_gpu(
    *,
    model_path: str | Path,
    weight_index: Any = None,
    model_plugin: Any = None,
    **_kwargs: Any,
) -> SuryaOCRGeneratorGPU:
    return SuryaOCRGeneratorGPU(
        model_path=model_path,
        weight_index=weight_index,
        model_plugin=model_plugin,
    )


register_text_generator(
    model="surya_ocr2",
    backend="hip_gfx1151",
    quant=_QUANT,
    factory=make_surya_generator_gpu,
)
