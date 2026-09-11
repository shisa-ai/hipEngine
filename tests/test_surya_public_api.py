from __future__ import annotations

import json
from pathlib import Path

import pytest

from hipengine import LLM, SamplingParams

FIXTURES = Path("tests/fixtures/surya")


@pytest.fixture(scope="module")
def llm() -> LLM | None:
    try:
        return LLM("datalab-to/surya-ocr-2", backend="cpu_reference")
    except FileNotFoundError:
        pytest.skip("datalab-to/surya-ocr-2 not in local HF cache")


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
