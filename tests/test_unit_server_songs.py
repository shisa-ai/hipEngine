"""The song endpoint and the surface routing around it.

A model declares what it generates, and the server routes on that declaration:
a song model serves ``POST /v1/audio/songs`` and refuses the text endpoints, a
text model does the reverse. These tests drive the app with a fake engine, so
they pin the HTTP contract rather than the pipeline.
"""

from __future__ import annotations

import asyncio
import base64
import io
import wave
from types import SimpleNamespace
from typing import Any

import httpx
import numpy as np
import pytest

from hipengine.generation.yue2 import SongRequest
from hipengine.server import ServerConfig, create_app


def _song(**overrides) -> SimpleNamespace:
    audio = np.linspace(-0.5, 0.5, 2 * 64, dtype=np.float32).reshape(2, 64)
    request = SongRequest(style="warm pop", lyrics="[verse]\nhi", cot="full", seed=11)
    values: dict[str, Any] = {
        "audio": audio,
        "sample_rate": 48000,
        "frames": 64,
        "duration_seconds": float(audio.shape[-1]) / 48000.0,
        "truncation": {"abc": False, "semantic": True},
        "semantic": SimpleNamespace(
            plan=SimpleNamespace(abc=None, abc_ids=(), request=request),
            tokens=(1, 2, 3),
        ),
        "latent_identity": "a" * 64,
        "audio_identity": "b" * 64,
        "config": {"ode_steps": 32},
        "weights": {"model": {"path": "/tmp/model"}, "vae": {"path": "/tmp/vae"}},
        "timing": {"e2e_seconds": 1.5},
        "request_id": "c" * 64,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class FakeSongGenerator:
    def __init__(self, *, song=None, error: BaseException | None = None) -> None:
        self.calls: list[tuple[Any, dict]] = []
        self._song = _song() if song is None else song
        self._error = error
        self.loaded = True

    def song_capabilities(self) -> dict[str, Any]:
        return {
            "surface": "song",
            "loaded": self.loaded,
            "sample_rate": 48000,
            "channels": 2,
            "cot_modes": ["full", "melody", "off"],
            "ode_steps": {"default": 32},
        }

    def generate_song(self, request, **kwargs):
        self.calls.append((request, kwargs))
        if self._error is not None:
            raise self._error
        return self._song


class FakeSongEngine:
    generation_surfaces = ("song",)

    def __init__(self, generator: FakeSongGenerator | None = None) -> None:
        self._text_generator = generator or FakeSongGenerator()

    @property
    def generator(self) -> FakeSongGenerator:
        return self._text_generator

    def song_generator(self):
        return self._text_generator


class FakeTextEngine:
    generation_surfaces = ("text",)

    def song_generator(self):
        raise NotImplementedError("song generation is not supported by this model")


def _app(engine: Any, **config_overrides) -> Any:
    config = ServerConfig(
        model="fake",
        served_model_name="fake-model",
        eager_load=False,
        **config_overrides,
    )
    return create_app(config, llm=engine)


def _post(
    app: Any, path: str, payload: dict | None = None, *, raise_app_exceptions: bool = True
) -> httpx.Response:
    async def run() -> httpx.Response:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(
                app=app, raise_app_exceptions=raise_app_exceptions
            ),
            base_url="http://test",
        ) as client:
            return await client.post(path, json=payload)

    return asyncio.run(run())


def _get(app: Any, path: str) -> httpx.Response:
    async def run() -> httpx.Response:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            return await client.get(path)

    return asyncio.run(run())


# ---------------------------------------------------------------------------
# the wav response
# ---------------------------------------------------------------------------


def test_the_song_endpoint_returns_wav_for_a_song_model() -> None:
    engine = FakeSongEngine()
    app = _app(engine)

    response = _post(
        app, "/v1/audio/songs", {"style": "warm pop", "lyrics": "[verse]\nhi"}
    )

    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "audio/wav"
    assert response.headers["content-disposition"] == 'attachment; filename="song.wav"'
    assert response.headers["x-hipengine-request-id"] == "c" * 64
    assert response.headers["x-hipengine-frames"] == "64"
    assert response.headers["x-hipengine-latent-identity"] == "a" * 64
    with wave.open(io.BytesIO(response.content), "rb") as handle:
        assert handle.getnchannels() == 2
        assert handle.getsampwidth() == 2
        assert handle.getframerate() == 48000
        assert handle.getnframes() == 64

    request, kwargs = engine.generator.calls[-1]
    assert request.style == "warm pop"
    assert request.cot == "full"
    assert kwargs["steps"] is None
    # Cancellation is wired: the pipeline polls this callable.
    assert callable(kwargs["cancelled"])


def test_the_request_fields_reach_the_model_request() -> None:
    engine = FakeSongEngine()
    app = _app(engine)

    response = _post(
        app,
        "/v1/audio/songs",
        {
            "style": "warm pop",
            "lyrics": "[verse]\nhi",
            "cot": "melody",
            "seed": 99,
            "abc": "X:1\nK:C\nC D E",
            "cfg_scale": 1.5,
            "steps": 4,
            "request_id": "song-7",
        },
    )

    assert response.status_code == 200, response.text
    request, kwargs = engine.generator.calls[-1]
    assert (request.cot, request.seed, request.abc) == ("melody", 99, "X:1\nK:C\nC D E")
    assert request.cfg_scale == 1.5
    assert request.id == "song-7"
    assert kwargs["steps"] == 4
    assert response.headers["content-disposition"] == 'attachment; filename="song-7.wav"'


def test_the_json_form_reports_provenance_and_the_wav_body() -> None:
    app = _app(FakeSongEngine())

    response = _post(
        app,
        "/v1/audio/songs",
        {"style": "s", "lyrics": "l", "response_format": "json"},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["object"] == "hipengine.song"
    assert body["model"] == "fake-model"
    assert body["sample_rate"] == 48000
    assert (body["channels"], body["samples"], body["frames"]) == (2, 64, 64)
    assert body["semantic_tokens"] == 3
    assert body["truncation"] == {"abc": False, "semantic": True}
    assert body["latent_identity"] == "a" * 64
    assert body["weights"]["vae"]["path"] == "/tmp/vae"
    assert body["config"] == {"ode_steps": 32}
    with wave.open(io.BytesIO(base64.b64decode(body["audio_wav_base64"])), "rb") as handle:
        assert handle.getframerate() == 48000
        assert handle.getnframes() == 64


# ---------------------------------------------------------------------------
# surface routing
# ---------------------------------------------------------------------------


def test_a_text_model_refuses_the_song_endpoint() -> None:
    app = _app(FakeTextEngine())

    response = _post(app, "/v1/audio/songs", {"style": "s", "lyrics": "l"})

    assert response.status_code == 501, response.text
    body = response.json()
    assert body["error"]["code"] == "unsupported_feature"
    assert body["error"]["hipengine"]["code"] == "unsupported_feature"
    assert "song generation" in body["error"]["message"]


def test_a_song_model_refuses_the_text_endpoints() -> None:
    app = _app(FakeSongEngine())

    chat = _post(
        app,
        "/v1/chat/completions",
        {"model": "fake-model", "messages": [{"role": "user", "content": "hi"}]},
    )
    completion = _post(app, "/v1/completions", {"model": "fake-model", "prompt": "hi"})

    for response, endpoint in ((chat, "/v1/chat/completions"), (completion, "/v1/completions")):
        assert response.status_code == 501, response.text
        body = response.json()
        assert body["error"]["code"] == "unsupported_feature"
        assert endpoint in body["error"]["message"]
        assert "POST /v1/audio/songs" in body["error"]["message"]
        assert "song" in body["error"]["message"]


def test_a_text_model_keeps_the_text_endpoints() -> None:
    """The guard is a routing decision, not a new refusal for existing models."""

    app = _app(FakeTextEngine())
    # Reaching request validation (rather than the surface guard) is the point:
    # an empty body is refused as a validation error, not as an unsupported surface.
    response = _post(app, "/v1/chat/completions", {})
    assert response.status_code in (400, 422), response.text
    assert "unsupported_feature" not in response.text


def test_the_wrong_model_id_is_a_404() -> None:
    app = _app(FakeSongEngine())
    response = _post(
        app, "/v1/audio/songs", {"model": "other", "style": "s", "lyrics": "l"}
    )
    assert response.status_code == 404, response.text
    assert response.json()["error"]["code"] == "model_not_found"


# ---------------------------------------------------------------------------
# request validation
# ---------------------------------------------------------------------------


def test_an_unknown_mode_is_refused_with_its_parameter() -> None:
    app = _app(FakeSongEngine())
    response = _post(app, "/v1/audio/songs", {"style": "s", "lyrics": "l", "cot": "sometimes"})
    assert response.status_code == 400, response.text
    body = response.json()
    assert body["error"]["code"] == "invalid_request"
    assert body["error"]["param"] == "cot"
    assert "off" in body["error"]["message"]


def test_an_unknown_response_format_is_refused_with_its_parameter() -> None:
    app = _app(FakeSongEngine())
    response = _post(
        app, "/v1/audio/songs", {"style": "s", "lyrics": "l", "response_format": "mp3"}
    )
    assert response.status_code == 400, response.text
    body = response.json()
    assert body["error"]["code"] == "unsupported_parameter"
    assert body["error"]["param"] == "response_format"


def test_an_external_score_requires_a_symbolic_mode() -> None:
    app = _app(FakeSongEngine())
    response = _post(
        app, "/v1/audio/songs", {"style": "s", "lyrics": "l", "cot": "off", "abc": "X:1"}
    )
    assert response.status_code == 400, response.text
    assert response.json()["error"]["param"] == "abc"


def test_a_filename_unsafe_request_id_is_refused() -> None:
    app = _app(FakeSongEngine())
    response = _post(
        app, "/v1/audio/songs", {"style": "s", "lyrics": "l", "request_id": "../escape"}
    )
    assert response.status_code == 400, response.text
    assert response.json()["error"]["param"] == "request_id"


@pytest.mark.parametrize(
    "payload",
    [
        {"lyrics": "l"},
        {"style": "", "lyrics": "l"},
        {"style": "s", "lyrics": ""},
        {"style": "s", "lyrics": "l", "steps": 0},
        {"style": "s", "lyrics": "l", "seed": -1},
        {"style": "s", "lyrics": "l", "cfg_scale": 50.0},
    ],
)
def test_a_malformed_body_is_a_validation_error(payload: dict) -> None:
    app = _app(FakeSongEngine())
    response = _post(app, "/v1/audio/songs", payload)
    assert response.status_code in (400, 422), response.text
    assert "error" in response.json()


# ---------------------------------------------------------------------------
# busy and cancelled
# ---------------------------------------------------------------------------


def test_a_busy_pipeline_is_a_retryable_429() -> None:
    from hipengine.runtime.yue2_session import Yue2SessionBusy

    engine = FakeSongEngine(FakeSongGenerator(error=Yue2SessionBusy("already running")))
    app = _app(engine, queue_retry_after_seconds=3)

    response = _post(app, "/v1/audio/songs", {"style": "s", "lyrics": "l"})

    assert response.status_code == 429, response.text
    body = response.json()
    assert body["error"]["code"] == "engine_busy"
    assert body["error"]["hipengine"]["retryable"] is True
    assert response.headers["retry-after"] == "3"


def test_a_cancelled_request_reports_499() -> None:
    engine = FakeSongEngine(FakeSongGenerator(error=InterruptedError("cancelled")))
    app = _app(engine)

    response = _post(app, "/v1/audio/songs", {"style": "s", "lyrics": "l"})

    assert response.status_code == 499, response.text
    body = response.json()
    assert body["error"]["code"] == "cancelled"
    assert body["error"]["hipengine"]["retryable"] is True


def test_an_unencodable_pipeline_result_is_a_server_error() -> None:
    """A non-finite sample is the pipeline's bug, not the request's."""

    audio = np.full((2, 8), np.nan, dtype=np.float32)
    engine = FakeSongEngine(FakeSongGenerator(song=_song(audio=audio)))
    response = _post(
        _app(engine),
        "/v1/audio/songs",
        {"style": "s", "lyrics": "l"},
        raise_app_exceptions=False,
    )
    assert response.status_code == 500, response.text
    body = response.json()
    assert body["error"]["code"] == "internal_error"
    assert "could not be encoded" in body["error"]["message"]


def test_a_pipeline_failure_is_not_reported_as_success() -> None:
    engine = FakeSongEngine(FakeSongGenerator(error=RuntimeError("vae decode failed")))
    app = _app(engine)
    response = _post(
        app,
        "/v1/audio/songs",
        {"style": "s", "lyrics": "l"},
        raise_app_exceptions=False,
    )
    assert response.status_code == 500, response.text
    assert response.json()["error"]["hipengine"]["exception_type"] == "RuntimeError"


# ---------------------------------------------------------------------------
# advertised capability
# ---------------------------------------------------------------------------


def test_capabilities_advertise_the_song_surface_and_endpoint() -> None:
    app = _app(FakeSongEngine())
    response = _get(app, "/v1/hipengine/capabilities")
    assert response.status_code == 200, response.text
    block = response.json()["song_generation"]
    assert block["enabled"] is True
    assert block["endpoint"] == "/v1/audio/songs"
    assert block["response_formats"] == ["wav", "json"]
    assert block["detail"]["sample_rate"] == 48000


def test_capabilities_report_the_decoder_checkpoint() -> None:
    app = _app(FakeSongEngine(), vae_model="/tmp/vae")
    block = _get(app, "/v1/hipengine/capabilities").json()["song_generation"]
    assert block["checkpoints"] == {"model": "fake", "decoder": "/tmp/vae"}


def test_a_text_model_reports_no_song_surface() -> None:
    app = _app(FakeTextEngine())
    body = _get(app, "/v1/hipengine/capabilities").json()
    assert body["song_generation"]["enabled"] is False
    assert body["song_generation"]["reason"]


def test_the_model_listing_reports_surfaces_and_disabled_text_routes() -> None:
    app = _app(FakeSongEngine())
    body = _get(app, "/v1/models").json()
    hipengine = body["data"][0]["hipengine"]
    assert hipengine["capabilities"]["surfaces"] == ["song"]
    assert hipengine["capabilities"]["song_generation"] is True
    assert hipengine["capabilities"]["chat_completions"] is False
    assert hipengine["song_generation"]["endpoint"] == "/v1/audio/songs"


def test_a_text_model_listing_is_unchanged() -> None:
    app = _app(FakeTextEngine())
    hipengine = _get(app, "/v1/models").json()["data"][0]["hipengine"]
    assert hipengine["capabilities"]["surfaces"] == ["text"]
    assert hipengine["capabilities"]["chat_completions"] is True
    assert hipengine["capabilities"]["song_generation"] is False
