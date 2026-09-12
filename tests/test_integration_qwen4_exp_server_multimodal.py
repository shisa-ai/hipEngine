from __future__ import annotations

import base64
import struct
import zlib

import numpy as np
import pytest
from fastapi.testclient import TestClient

from hipengine import SamplingParams
from hipengine.generation import FinishDetails, GenerationOutput
from hipengine.server import ServerConfig, create_app
from hipengine.server.__main__ import build_parser
from hipengine.server.multimodal import (
    decode_bounded_png_data_url,
    extract_qwen4_exp_chat_media,
)


def _png_data_url(image: np.ndarray) -> str:
    values = np.ascontiguousarray(image, dtype=np.uint8)
    height, width, channels = values.shape
    assert channels == 3

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    scanlines = b"".join(b"\0" + values[row].tobytes() for row in range(height))
    payload = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(scanlines))
        + chunk(b"IEND", b"")
    )
    return "data:image/png;base64," + base64.b64encode(payload).decode()


def test_qwen4_exp_server_cli_accepts_vision_model() -> None:
    args = build_parser().parse_args(
        ["--model", "target.gguf", "--vision-model", "mmproj.gguf"]
    )
    assert args.vision_model == "mmproj.gguf"
    assert ServerConfig(model="target.gguf", vision_model="mmproj.gguf").vision_model == "mmproj.gguf"


def test_qwen4_exp_bounded_png_data_url_round_trips_rgb() -> None:
    image = np.arange(32 * 64 * 3, dtype=np.uint8).reshape(32, 64, 3)
    np.testing.assert_array_equal(decode_bounded_png_data_url(_png_data_url(image)), image)


def test_qwen4_exp_chat_media_extracts_image_and_video_in_order() -> None:
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    other = image.copy()
    other[..., 1] = 255
    prompt, media = extract_qwen4_exp_chat_media(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Compare "},
                    {"type": "image_url", "image_url": {"url": _png_data_url(image)}},
                    {"type": "text", "text": " with video "},
                    {"type": "video_frames", "frames": [_png_data_url(image), _png_data_url(other)]},
                ],
            }
        ]
    )
    assert prompt.count("<|image_pad|>") == 1
    assert prompt.count("<|video_pad|>") == 1
    assert [item["type"] for item in media["items"]] == ["image", "video"]
    assert media["items"][0]["data"].shape == (32, 32, 3)
    assert media["items"][1]["data"].shape == (2, 32, 32, 3)


def test_qwen4_exp_chat_completions_accepts_bounded_png_multimodal_input() -> None:
    class VisionLLM:
        max_sequence_length = 1024
        supports_vision = True

        def __init__(self):
            self.calls = []

        def count_tokens(self, text):
            return len(str(text).split())

        def generate_multimodal_detailed(self, prompt, media, sampling):
            assert isinstance(sampling, SamplingParams)
            self.calls.append((prompt, media, sampling))
            return GenerationOutput(
                text="vision-ok",
                generated_token_ids=(7, 8),
                finish_details=FinishDetails(
                    reason="length", length_limit=2, sampler_mode="greedy"
                ),
            )

        def close(self):
            pass

    image = np.zeros((32, 32, 3), dtype=np.uint8)
    fake = VisionLLM()
    app = create_app(
        ServerConfig(
            model="fake-path", served_model_name="qwen4exp", eager_load=False,
            startup_chat_smoke=False, startup_scratch_probe=False,
        ),
        llm=fake,
    )
    response = TestClient(app).post(
        "/v1/chat/completions",
        json={
            "model": "qwen4exp",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe "},
                        {
                            "type": "image_url",
                            "image_url": {"url": _png_data_url(image)},
                        },
                    ],
                }
            ],
            "max_tokens": 2,
            "temperature": 0,
        },
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["choices"][0]["message"]["content"] == "vision-ok"
    assert payload["hipengine"]["multimodal"] is True
    assert len(fake.calls) == 1
    assert fake.calls[0][1]["items"][0]["data"].shape == (32, 32, 3)


@pytest.mark.parametrize("value", ["https://example.com/a.png", "data:image/jpeg;base64,AAAA"])
def test_qwen4_exp_http_vision_rejects_unbounded_or_unsupported_urls(value) -> None:
    with pytest.raises(ValueError, match="image/png"):
        decode_bounded_png_data_url(value)
