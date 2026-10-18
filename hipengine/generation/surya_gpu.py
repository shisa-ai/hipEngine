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
``hipengine.generation.surya_contract``).

Four budgets are explicit, validated, and advertised on the generator: the
text context (``max_seq``, from ``LLM(max_sequence_length=...)``), the
per-request output budget (``max_tokens``), and the two attention score tiles
(``max_vision_scratch_bytes`` for the vision tower,
``max_prefill_scratch_bytes`` for the text prefill). Prompt plus output
capacity is validated before any device work runs, including before the
vision tower.

Registered under ``(surya_ocr2, hip_gfx1151, fp32)`` through
``register_builtin_generators()``. Correctness contract: greedy IDs must
match the CPU reference path, which is itself gated against the torch
fp32 oracle fixtures.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from hipengine.generation.deadline import (
    GenerationDeadlineExceeded,
    generation_deadline_remaining,
    raise_if_generation_deadline_expired,
)
from hipengine.generation.registry import register_text_generator
from hipengine.generation.surya_contract import (
    check_prompt_capacity,
    greedy_decode_tokens,
    resolve_surya_greedy_settings,
)
from hipengine.generation.surya_protocol import FULL_PAGE_HTML_PROMPT
from hipengine.kernels.cpu_reference.surya import SuryaSpec, SuryaWeights
from hipengine.loading.surya import (
    SURYA_MAX_PIXELS,
    SuryaTokenizer,
    compute_mrope_positions,
    load_surya_spec,
    load_surya_weights,
    preprocess_image_surya,
    render_chat_prompt,
    resolve_surya_path,
)
from hipengine.runtime.surya import DEFAULT_MAX_SEQ, SuryaGpuRunner

_QUANT = "fp32"


class SuryaOCRGeneratorGPU:
    """Greedy OCR generator running vision + text on the HIP device.

    Five budgets are explicit and advertised so a caller can admit a request
    before submitting it:

    - ``max_seq`` — text context (prompt plus output). Defaults to
      ``DEFAULT_MAX_SEQ``, which admits a 300-DPI A4 page's 8580 image tokens;
      set it from ``LLM(max_sequence_length=...)``.
    - ``max_tokens`` — the per-request output budget, validated against
      ``max_seq`` before any device work.
    - ``max_vision_scratch_bytes`` — the peak vision-attention score tile.
      ``None`` means the runner default; pass a value to trade GEMM width for
      memory. The vision path never allocates the quadratic score matrix.
    - ``max_vision_seconds`` — the wall-clock budget for one vision forward.
      ``None`` means the runner default (``DEFAULT_MAX_VISION_SECONDS``, which
      admits every page the benchmark suite measures and rejects the 65536-
      patch ``vision_max_pixels`` ceiling); ``math.inf`` disables it. This is
      what actually bounds a page-scale grid: 65536 patches is 1.7e14 FLOPs of
      vision attention and about 2.7 minutes on gfx1151, and the byte budget
      only sees the 502 MB score tile.
    - ``max_prefill_scratch_bytes`` — the peak text-prefill causal score tile.
      ``None`` means the runner default; pass a value to trade GEMM width for
      memory. The prefill never allocates the quadratic score matrix.

    Vision-input capabilities for a serving front end:

    - ``vision_max_pixels`` — the checkpoint's own preprocessor ceiling, so an
      HTTP layer can bound a decoded page without guessing.
    - ``vision_media_input = "image_array"`` — one RGB array per request.
    - ``vision_default_prompt`` — the checkpoint's full-page transcription
      prompt, which is the task. Surya is prompt-driven, so a caller that sends
      no instruction wants this rather than an empty prompt.
    """

    vision_max_pixels = SURYA_MAX_PIXELS
    vision_media_input = "image_array"
    vision_default_prompt = FULL_PAGE_HTML_PROMPT
    # render_chat_prompt derives the image pad span from the patch grid,
    # so the prompt is the bare text with no inline placeholder.
    vision_prompt_marker = ""

    def __init__(
        self,
        *,
        model_path: str | Path,
        weight_index: Any = None,
        model_plugin: Any = None,
        vision_model_path: str | Path | None = None,
        max_seq: int | None = None,
        max_vision_scratch_bytes: int | None = None,
        max_vision_seconds: float | None = None,
        max_prefill_scratch_bytes: int | None = None,
    ) -> None:
        self.model_dir = resolve_surya_path(model_path)
        self.spec: SuryaSpec = load_surya_spec(self.model_dir)
        weights = load_surya_weights(self.model_dir)
        self.tokenizer = SuryaTokenizer(self.model_dir)
        resolved_max_seq = DEFAULT_MAX_SEQ if max_seq is None else int(max_seq)
        if resolved_max_seq <= 0:
            raise ValueError("max_seq must be positive")
        if max_vision_scratch_bytes is not None and int(max_vision_scratch_bytes) <= 0:
            raise ValueError(
                "max_vision_scratch_bytes must be positive when set"
            )
        if max_vision_seconds is not None and not float(max_vision_seconds) > 0:
            raise ValueError("max_vision_seconds must be positive when set")
        if max_prefill_scratch_bytes is not None and int(max_prefill_scratch_bytes) <= 0:
            raise ValueError(
                "max_prefill_scratch_bytes must be positive when set"
            )
        # `None` at this layer means "use the runner default"; only an explicit
        # value is forwarded, because the runner reads `None` as "no budget".
        # An unbounded vision time budget is `math.inf`, not `None`.
        capacity_kwargs: dict[str, Any] = {}
        if max_vision_scratch_bytes is not None:
            capacity_kwargs["max_vision_scratch_bytes"] = int(max_vision_scratch_bytes)
        if max_vision_seconds is not None:
            capacity_kwargs["max_vision_seconds"] = float(max_vision_seconds)
        if max_prefill_scratch_bytes is not None:
            capacity_kwargs["max_prefill_scratch_bytes"] = int(max_prefill_scratch_bytes)
        self.runner = SuryaGpuRunner(
            weights, self.spec, max_seq=resolved_max_seq, **capacity_kwargs
        )
        self.model_plugin = model_plugin
        self.supports_vision = True
        self.speculative_candidate_budget = 0
        # Advertised so a caller can admit a request before submitting it.
        self.max_seq = self.runner.max_seq
        self.max_vision_scratch_bytes = self.runner.max_vision_scratch_bytes
        self.max_vision_seconds = self.runner.max_vision_seconds
        self.max_prefill_scratch_bytes = self.runner.max_prefill_scratch_bytes

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
        token_ids, finish_reason, prompt_tokens = self._generate_ocr(
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
            prompt_tokens=prompt_tokens,
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
        # An abandoned request must not pay for the prefill. This is the second
        # of three checks: the caller already checked before vision, and the
        # decode loop checks before every step.
        raise_if_generation_deadline_expired(request)
        check_prompt_capacity(
            int(input_ids.shape[-1]),
            settings.max_tokens,
            self.runner.max_seq,
            hint="raise LLM(max_sequence_length=...)",
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
    ) -> tuple[list[int], str, int]:
        # Fail before any work when the request is already abandoned, then
        # again before the two expensive stages. Preprocessing is host-side and
        # cheap but not free at 16.7 MP, vision is the first device stage, and
        # the prefill is checked inside `_decode_greedy`.
        raise_if_generation_deadline_expired(request)
        pixel_rows, grid = preprocess_image_surya(image)
        n_image_tokens = (grid[1] // 2) * (grid[2] // 2)
        input_ids, mm = render_chat_prompt(
            self.tokenizer, prompt, n_image_tokens
        )
        position_ids = compute_mrope_positions(
            mm, grid, self.spec.vision_spatial_merge_size
        )
        # Admit before vision, not after. Preprocessing, the prompt and the
        # mRoPE tables are host-side and cheap; vision is the expensive step and
        # allocates scratch quadratic in the patch count, so an over-capacity or
        # over-budget page must be rejected before it runs. `_decode_greedy`
        # repeats the context check for its own callers; it is O(1).
        check_prompt_capacity(
            len(input_ids),
            settings.max_tokens,
            self.runner.max_seq,
            hint="raise LLM(max_sequence_length=...)",
        )
        raise_if_generation_deadline_expired(request)
        self.runner.check_vision_capacity([grid])
        # The deadline is otherwise only observed at phase boundaries, so a page
        # whose vision forward cannot finish inside the remaining budget would
        # still run to completion before the client's timeout was noticed.
        # Reject it now, as the timeout it is, rather than after the wait.
        deadline_at = getattr(request, "deadline_at", None)
        remaining = generation_deadline_remaining(deadline_at)
        if remaining is not None and self.runner.vision_time_seconds([grid]) > remaining:
            raise GenerationDeadlineExceeded(deadline_at=deadline_at)
        merged = self.runner.vision_forward(pixel_rows, [grid])
        token_ids, finish_reason = self._decode_greedy(
            np.array([input_ids], dtype=np.int64),
            position_ids,
            merged,
            request,
            settings,
        )
        # The prompt is text *plus* image tokens, so its length is only known
        # here. A serving front end reports it as usage.prompt_tokens.
        return token_ids, finish_reason, len(input_ids)


def make_surya_generator_gpu(
    *,
    model_path: str | Path,
    weight_index: Any = None,
    model_plugin: Any = None,
    vision_model_path: str | Path | None = None,
    max_sequence_length: int | None = None,
    vision_max_scratch_bytes: int | None = None,
    prefill_max_scratch_bytes: int | None = None,
    vision_max_seconds: float | None = None,
) -> SuryaOCRGeneratorGPU:
    """Registered ``(surya_ocr2, hip_gfx1151, fp32)`` factory.

    The capacity parameters are declared by name because
    ``LLM._factory_capacity_kwargs`` forwards a limit only to a factory that
    accepts it: an earlier ``**_kwargs`` signature silently dropped
    ``max_sequence_length``, leaving the runner at its low-level 2048 default.
    """

    return SuryaOCRGeneratorGPU(
        model_path=model_path,
        weight_index=weight_index,
        model_plugin=model_plugin,
        vision_model_path=vision_model_path,
        max_seq=max_sequence_length,
        max_vision_scratch_bytes=vision_max_scratch_bytes,
        max_vision_seconds=vision_max_seconds,
        max_prefill_scratch_bytes=prefill_max_scratch_bytes,
    )


register_text_generator(
    model="surya_ocr2",
    backend="hip_gfx1151",
    quant=_QUANT,
    factory=make_surya_generator_gpu,
)
