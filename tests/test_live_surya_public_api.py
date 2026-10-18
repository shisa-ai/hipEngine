from __future__ import annotations

import json
from pathlib import Path

import pytest

from hipengine import LLM, SamplingParams

FIXTURES = Path("tests/fixtures/surya")


@pytest.fixture(scope="module")
def llm() -> LLM | None:
    from hipengine.loading.surya import resolve_surya_path

    try:
        path = resolve_surya_path("datalab-to/surya-ocr-2")
    except FileNotFoundError:
        pytest.skip("datalab-to/surya-ocr-2 not in local HF cache")
    return LLM(str(path), backend="cpu_reference")


def test_plugin_and_generator_registered(llm: LLM) -> None:
    from hipengine.generation import register_builtin_generators
    from hipengine.generation.registry import resolve_text_generator
    from hipengine.models.registry import resolve_model

    register_builtin_generators()

    plugin = resolve_model("Qwen3_5ForConditionalGeneration")
    assert plugin.name == "surya_ocr2"
    factory = resolve_text_generator(
        model="surya_ocr2", backend="cpu_reference", quant="fp32"
    )
    assert callable(factory)


def test_public_text_only_generation(llm: LLM) -> None:
    out = llm.generate(["Transcribe this page."], SamplingParams(max_tokens=8))
    assert len(out) == 1
    assert isinstance(out[0], str) and out[0] != ""


def test_public_multimodal_ocr_matches_torch_reference(llm: LLM) -> None:
    ref_path = FIXTURES / "oracle_greedy.json"
    if not ref_path.exists():
        pytest.skip("oracle_greedy.json not present; capture with transformers")
    ref = json.loads(ref_path.read_text())
    res = llm.generate_multimodal_detailed(
        "Transcribe this page.",
        str(FIXTURES / "page_small.png"),
        SamplingParams(max_tokens=64),
    )
    # torch-free public path reproduces the torch fp32 greedy output exactly
    assert res.text == ref["text"]
    assert llm.supports_vision


def test_run_surya_ocr_defaults_to_the_full_page_prompt() -> None:
    """The public OCR entry point must drive the checkpoint's real task.

    ``run_surya_ocr`` used to default to the ad-hoc ``"Transcribe this
    page."`` prompt, whose continuation is layout JSON or a degenerate
    repeated list. Its default is now the full-page transcription prompt, and
    the call must go through the registered generator so this entry point and
    ``LLM(...)`` share one implementation.
    """

    from PIL import Image

    from hipengine.generation.registry import GenerationRequest, resolve_text_generator
    from hipengine.generation.surya_protocol import FULL_PAGE_HTML_PROMPT
    from hipengine.loading.surya import run_surya_ocr, resolve_surya_path

    page = FIXTURES / "page_small.png"
    if not page.exists():
        pytest.skip("page_small.png fixture not present")
    try:
        model_dir = resolve_surya_path("datalab-to/surya-ocr-2")
    except FileNotFoundError:
        pytest.skip("datalab-to/surya-ocr-2 not in local HF cache")

    image = Image.open(page).convert("RGB")
    default = run_surya_ocr(model_dir, image, max_new_tokens=32)

    factory = resolve_text_generator(
        model="surya_ocr2", backend="cpu_reference", quant="fp32"
    )
    generator = factory(model_path=model_dir)
    try:
        explicit = generator.generate_multimodal_detailed(
            FULL_PAGE_HTML_PROMPT,
            image,
            GenerationRequest(
                prompts=[FULL_PAGE_HTML_PROMPT],
                max_tokens=32,
                temperature=0.0,
                top_p=1.0,
                ignore_eos=False,
            ),
        )
    finally:
        generator.close()

    assert default.token_ids == list(explicit.generated_token_ids)
    assert default.text == explicit.text
    assert default.text.startswith("<div data-bbox="), (
        f"the default prompt did not produce the transcription protocol: "
        f"{default.text[:120]!r}"
    )


def test_run_surya_ocr_propagates_cancellation_and_deadlines() -> None:
    from PIL import Image

    from hipengine.generation.deadline import (
        GenerationCancelled,
        GenerationCancellationToken,
        GenerationDeadlineExceeded,
    )
    from hipengine.loading.surya import run_surya_ocr, resolve_surya_path

    page = FIXTURES / "page_small.png"
    if not page.exists():
        pytest.skip("page_small.png fixture not present")
    try:
        model_dir = resolve_surya_path("datalab-to/surya-ocr-2")
    except FileNotFoundError:
        pytest.skip("datalab-to/surya-ocr-2 not in local HF cache")

    image = Image.open(page).convert("RGB")
    token = GenerationCancellationToken()
    token.cancel()
    with pytest.raises(GenerationCancelled):
        run_surya_ocr(model_dir, image, max_new_tokens=8, cancellation_token=token)

    with pytest.raises(GenerationDeadlineExceeded):
        run_surya_ocr(model_dir, image, max_new_tokens=8, deadline_at=0.0)
