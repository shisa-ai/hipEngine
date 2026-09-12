"""Surya OCR over the OpenAI-compatible HTTP API.

The multimodal HTTP branch was Qwen4Exp-shaped in three ways: it capped decoded
images at 1024 px a side (an A4 300-DPI page is 2480x3508), it handed the engine
a ``{"items": [...]}`` media mapping, and its error strings named Qwen4Exp. These
tests pin the model-neutral behaviour and the Surya path through it, including
that the committed A4 fixture is admitted.
"""

from __future__ import annotations

import base64
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from hipengine.generation import FinishDetails, GenerationOutput  # noqa: E402
from hipengine.generation.surya import SuryaOCRGenerator  # noqa: E402
from hipengine.generation.surya_protocol import FULL_PAGE_HTML_PROMPT  # noqa: E402
from hipengine.server import ServerConfig, create_app  # noqa: E402
from hipengine.server.__main__ import build_parser  # noqa: E402
from hipengine.server.multimodal import (  # noqa: E402
    DEFAULT_VISION_MAX_BYTES,
    DEFAULT_VISION_MAX_PIXELS,
    decode_bounded_png_data_url,
    media_for_engine,
    resolve_vision_http_limits,
    vision_max_side,
)

FIXTURES = Path("tests/fixtures/surya")


def _png_data_url(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


class _VisionGenerator:
    """The generation surface the server's multimodal branch calls."""

    def __init__(self, text: str = "<div data-bbox=\"1 2 3 4\">ok</div>") -> None:
        self.calls: list[tuple] = []
        self._text = text

    def count_tokens(self, text: object) -> int:
        return len(str(text).split())

    def generate_multimodal_detailed(self, prompt, image, sampling):
        self.calls.append((prompt, image, sampling))
        return GenerationOutput(
            text=self._text,
            generated_token_ids=(7, 8, 9),
            finish_details=FinishDetails(
                reason="eos", eos_token_id=2, length_limit=3, sampler_mode="greedy"
            ),
        )

    def close(self) -> None:
        pass


class _SuryaLikeLLM(_VisionGenerator):
    """Surya's declarations, read off the real generator class so this fake
    cannot drift from what ships."""

    supports_vision = True
    vision_max_pixels = SuryaOCRGenerator.vision_max_pixels
    vision_media_input = SuryaOCRGenerator.vision_media_input
    vision_default_prompt = SuryaOCRGenerator.vision_default_prompt
    vision_prompt_marker = SuryaOCRGenerator.vision_prompt_marker
    max_sequence_length = 16384


class _Qwen4ExpLikeLLM(_VisionGenerator):
    """An engine that declares no vision bounds keeps the old scope."""

    supports_vision = True
    max_sequence_length = 16384


def _client(llm: object, model: str = "surya-ocr-2", **config_kwargs) -> TestClient:
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name=model,
            eager_load=False,
            startup_chat_smoke=False,
            startup_scratch_probe=False,
            **config_kwargs,
        ),
        llm=llm,
    )
    return TestClient(app)


def _post_page(client: TestClient, page: Path, *, model: str, text: str | None,
               max_tokens: int = 8):
    parts: list[dict] = []
    if text is not None:
        parts.append({"type": "text", "text": text})
    parts.append({"type": "image_url", "image_url": {"url": _png_data_url(page)}})
    return client.post(
        "/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": parts}],
            "max_tokens": max_tokens,
            "temperature": 0,
        },
    )


# -- limits -----------------------------------------------------------------


def test_http_vision_limits_default_to_the_qwen4exp_scope() -> None:
    pixels, max_bytes = resolve_vision_http_limits(_Qwen4ExpLikeLLM())
    assert pixels == DEFAULT_VISION_MAX_PIXELS
    assert vision_max_side(pixels) == 1024
    assert max_bytes == DEFAULT_VISION_MAX_BYTES


def test_http_vision_limits_follow_a_page_scale_engine_declaration() -> None:
    pixels, max_bytes = resolve_vision_http_limits(_SuryaLikeLLM())
    assert pixels == 16_777_216
    assert vision_max_side(pixels) == 4096
    # widened with the area, because a 300-DPI document does not compress like
    # a thumbnail
    assert max_bytes > DEFAULT_VISION_MAX_BYTES


def test_explicit_http_vision_limits_win() -> None:
    pixels, max_bytes = resolve_vision_http_limits(
        _SuryaLikeLLM(), max_pixels=2_000_000, max_bytes=1_000_000
    )
    assert (pixels, max_bytes) == (2_000_000, 1_000_000)


def test_server_cli_exposes_the_vision_limits() -> None:
    args = build_parser().parse_args(
        ["--model", "m", "--vision-max-pixels", "16777216",
         "--vision-max-image-bytes", "33554432"]
    )
    assert args.vision_max_pixels == 16_777_216
    assert args.vision_max_image_bytes == 33_554_432
    config = ServerConfig(
        model="m", vision_max_pixels=16_777_216, vision_max_image_bytes=33_554_432
    )
    assert config.vision_max_pixels == 16_777_216
    with pytest.raises(ValueError, match="vision_max_pixels"):
        ServerConfig(model="m", vision_max_pixels=0)


# -- media adaptation -------------------------------------------------------


def test_media_for_engine_unwraps_one_image_for_an_image_array_engine() -> None:
    image = np.zeros((4, 6, 3), dtype=np.uint8)
    media = {"items": [{"type": "image", "data": image}]}
    adapted = media_for_engine(_SuryaLikeLLM(), media)
    assert isinstance(adapted, np.ndarray)
    np.testing.assert_array_equal(adapted, image)


def test_media_for_engine_keeps_the_items_mapping_for_the_default_form() -> None:
    media = {"items": [{"type": "image", "data": np.zeros((4, 6, 3), dtype=np.uint8)}]}
    assert media_for_engine(_Qwen4ExpLikeLLM(), media) is media


def test_media_for_engine_rejects_more_than_one_image_for_a_single_image_model() -> None:
    image = np.zeros((4, 6, 3), dtype=np.uint8)
    media = {"items": [{"type": "image", "data": image}, {"type": "image", "data": image}]}
    with pytest.raises(ValueError, match="exactly one image"):
        media_for_engine(_SuryaLikeLLM(), media)


def test_request_has_media_is_structural() -> None:
    from hipengine.server.multimodal import request_has_media

    assert request_has_media([{"role": "user", "content": "plain text"}]) is False
    assert request_has_media([{"role": "user", "content": [{"type": "text", "text": "a"}]}]) is False
    assert request_has_media(
        [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}]
    ) is True
    # ... and does not decode, so an unusable URL is still "has media".
    assert request_has_media(
        [{"role": "user", "content": [{"type": "input_image", "image": "https://x"}]}]
    ) is True


def test_a_lazily_loaded_server_resolves_the_engine_for_a_media_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """eager_load=False leaves no attached engine; a media request must load it.

    Otherwise an engine-declared vision model (Surya) is invisible on a lazy
    server and every page is rejected as "does not support vision".
    """

    import hipengine.server.api as api_module

    page = FIXTURES / "page_small.png"
    if not page.exists():
        pytest.skip("page_small.png is not present")

    llm = _SuryaLikeLLM()
    built: list[str] = []

    def fake_llm(model, **kwargs):
        built.append(model)
        return llm

    monkeypatch.setattr(api_module, "LLM", fake_llm)
    app = create_app(
        ServerConfig(
            model="datalab-to/surya-ocr-2",
            served_model_name="surya-ocr-2",
            eager_load=False,
            startup_chat_smoke=False,
            startup_scratch_probe=False,
        )
    )
    assert app.state.hipengine_llm is None
    response = _post_page(
        TestClient(app), page, model="surya-ocr-2", text=FULL_PAGE_HTML_PROMPT
    )
    assert response.status_code == 200, response.text
    assert built == ["datalab-to/surya-ocr-2"]
    assert len(llm.calls) == 1
    assert isinstance(llm.calls[0][1], np.ndarray)


def test_vision_prompt_markers_follow_the_engine_declaration() -> None:
    from hipengine.server.multimodal import vision_prompt_markers

    # Qwen4Exp splices features at an inline marker ...
    assert vision_prompt_markers(_Qwen4ExpLikeLLM())[0] == (
        "<|vision_start|><|image_pad|><|vision_end|>"
    )
    # ... Surya renders the placeholder from the patch grid, so bare text.
    assert vision_prompt_markers(_SuryaLikeLLM()) == ("", "")


# -- the A4 page is admitted -------------------------------------------------


def test_the_committed_a4_page_decodes_under_the_surya_bound() -> None:
    page = FIXTURES / "page_a4.png"
    if not page.exists():
        pytest.skip("page_a4.png is not present")
    pixels, max_bytes = resolve_vision_http_limits(_SuryaLikeLLM())
    decoded = decode_bounded_png_data_url(
        _png_data_url(page),
        max_bytes=max_bytes,
        max_side=vision_max_side(pixels),
        max_pixels=pixels,
    )
    assert decoded.shape == (3508, 2480, 3)
    # ... and is rejected by the Qwen4Exp scope it used to be limited to.
    with pytest.raises(ValueError, match="unsupported PNG geometry"):
        decode_bounded_png_data_url(
            _png_data_url(page),
            max_bytes=DEFAULT_VISION_MAX_BYTES,
            max_side=vision_max_side(DEFAULT_VISION_MAX_PIXELS),
            max_pixels=DEFAULT_VISION_MAX_PIXELS,
        )


# -- end to end over HTTP ----------------------------------------------------


def test_surya_chat_completions_transcribes_a_page() -> None:
    page = FIXTURES / "page_small.png"
    if not page.exists():
        pytest.skip("page_small.png is not present")
    llm = _SuryaLikeLLM()
    response = _post_page(
        _client(llm), page, model="surya-ocr-2", text=FULL_PAGE_HTML_PROMPT
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["choices"][0]["message"]["content"].startswith("<div data-bbox=")
    assert payload["choices"][0]["finish_reason"] == "stop"
    assert payload["hipengine"]["multimodal"] is True
    assert payload["usage"]["completion_tokens"] == 3

    assert len(llm.calls) == 1
    prompt, image, sampling = llm.calls[0]
    assert prompt == FULL_PAGE_HTML_PROMPT
    assert isinstance(image, np.ndarray), "the engine must receive the RGB array"
    assert image.ndim == 3 and image.shape[2] == 3


def test_surya_chat_completions_falls_back_to_the_model_prompt() -> None:
    """An image-only request means "do the model's task", not "empty prompt"."""

    page = FIXTURES / "page_small.png"
    if not page.exists():
        pytest.skip("page_small.png is not present")
    llm = _SuryaLikeLLM()
    response = _post_page(_client(llm), page, model="surya-ocr-2", text=None)
    assert response.status_code == 200, response.text
    assert llm.calls[0][0] == FULL_PAGE_HTML_PROMPT


def test_surya_chat_completions_keeps_an_explicit_prompt() -> None:
    page = FIXTURES / "page_small.png"
    if not page.exists():
        pytest.skip("page_small.png is not present")
    llm = _SuryaLikeLLM()
    response = _post_page(_client(llm), page, model="surya-ocr-2", text="LAYOUT ONLY")
    assert response.status_code == 200, response.text
    assert llm.calls[0][0] == "LAYOUT ONLY"


def test_surya_chat_completions_rejects_a_page_above_a_configured_bound() -> None:
    page = FIXTURES / "page_a4.png"
    if not page.exists():
        pytest.skip("page_a4.png is not present")
    llm = _SuryaLikeLLM()
    response = _post_page(
        _client(llm, vision_max_pixels=1_048_576),
        page,
        model="surya-ocr-2",
        text=FULL_PAGE_HTML_PROMPT,
    )
    assert response.status_code == 400, response.text
    assert "unsupported PNG geometry" in response.text
    assert llm.calls == [], "the engine ran for a rejected image"


def test_surya_generators_declare_their_vision_capabilities() -> None:
    """A serving front end needs the bounds and the media form, not guesses."""

    from hipengine.generation.surya import SuryaOCRGenerator
    from hipengine.generation.surya_gpu import SuryaOCRGeneratorGPU

    for generator in (SuryaOCRGenerator, SuryaOCRGeneratorGPU):
        assert generator.vision_media_input == "image_array"
        assert generator.vision_max_pixels == 16_777_216
        assert generator.vision_default_prompt == FULL_PAGE_HTML_PROMPT
        # render_chat_prompt renders the image placeholder itself
        assert generator.vision_prompt_marker == ""
