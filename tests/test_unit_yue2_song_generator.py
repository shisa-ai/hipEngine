"""The YuE2 song surface: registration, the companion decoder path, and the
request shape that ``LLM.generate_song`` and the server both reach.

These tests never build a real session: the pipeline's own numerics are covered
by the other ``test_unit_yue2_*`` modules and the GPU gates. What is checked here
is the wiring around it -- which directory the decoder comes from, that nothing
loads until a request arrives, and that a request reaches the session with the
arguments it was given.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.generation.yue2_song import (
    DEFAULT_AR_CONTEXT,
    VAE_ENV,
    VAE_REPO_DIRNAME,
    YuE2SongGenerator,
    make_yue2_song_generator,
    resolve_vae_path,
)


# ---------------------------------------------------------------------------
# registration
# ---------------------------------------------------------------------------


def test_registration_covers_both_amd_backends() -> None:
    from hipengine.generation import (
        register_builtin_generators,
        registered_text_generators,
        resolve_text_generator,
    )

    register_builtin_generators()
    keys = {
        (key.model, key.backend, key.quant)
        for key in registered_text_generators()
        if key.model == "yue2"
    }
    assert keys == {
        ("yue2", "hip_gfx1100", "bf16"),
        ("yue2", "hip_gfx1151", "bf16"),
    }
    for backend in ("hip_gfx1100", "hip_gfx1151"):
        factory = resolve_text_generator(model="yue2", backend=backend, quant="bf16")
        generator = factory(
            model_path=_model_dir(),
            weight_index=None,
            model_plugin=None,
            vae_model_path=_vae_dir(),
            backend=backend,
        )
        assert isinstance(generator, YuE2SongGenerator)
        assert generator.backend == backend


def test_the_declared_surface_is_song() -> None:
    generator = YuE2SongGenerator(_model_dir(), vae_path=_vae_dir())
    assert generator.generation_surfaces == ("song",)
    assert generator.loaded is False


# ---------------------------------------------------------------------------
# the companion decoder path
# ---------------------------------------------------------------------------


def test_explicit_vae_directory_wins_over_the_environment_and_the_cache(tmp_path) -> None:
    explicit = tmp_path / "explicit"
    explicit.mkdir()
    from_env = tmp_path / "from-env"
    from_env.mkdir()
    resolved = resolve_vae_path(
        explicit,
        environ={VAE_ENV: str(from_env)},
        cache_root=tmp_path / "hub",
    )
    assert resolved == explicit


def test_vae_directory_falls_back_to_the_environment(tmp_path) -> None:
    from_env = tmp_path / "from-env"
    from_env.mkdir()
    assert resolve_vae_path(None, environ={VAE_ENV: str(from_env)}, cache_root=tmp_path) == from_env


def test_vae_directory_follows_the_cache_ref_not_a_snapshot_name(tmp_path) -> None:
    repo = tmp_path / "hub" / VAE_REPO_DIRNAME
    wanted = repo / "snapshots" / "0ref"
    wanted.mkdir(parents=True)
    # A lexicographically later snapshot that the ref does not point at.
    (repo / "snapshots" / "zzzz").mkdir()
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text("0ref\n")
    assert resolve_vae_path(None, environ={}, cache_root=tmp_path / "hub") == wanted


def test_a_missing_companion_is_reported_by_name(tmp_path) -> None:
    with pytest.raises(FileNotFoundError) as excinfo:
        resolve_vae_path(None, environ={}, cache_root=tmp_path / "hub")
    message = str(excinfo.value)
    assert "YuE2 VAE" in message
    assert VAE_ENV in message
    assert "vae_model=" in message


def test_an_explicit_directory_that_does_not_exist_is_reported(tmp_path) -> None:
    missing = tmp_path / "not-here"
    with pytest.raises(FileNotFoundError) as excinfo:
        resolve_vae_path(missing, environ={}, cache_root=tmp_path)
    assert str(missing) in str(excinfo.value)


def test_several_cached_snapshots_are_refused_rather_than_guessed(tmp_path) -> None:
    repo = tmp_path / "hub" / VAE_REPO_DIRNAME
    (repo / "snapshots" / "aaaa").mkdir(parents=True)
    (repo / "snapshots" / "bbbb").mkdir()
    with pytest.raises(ValueError) as excinfo:
        resolve_vae_path(None, environ={}, cache_root=tmp_path / "hub")
    message = str(excinfo.value)
    assert "2 snapshots" in message
    assert "aaaa" in message and "bbbb" in message


# ---------------------------------------------------------------------------
# lifecycle
# ---------------------------------------------------------------------------


def _model_dir(tmp_path=None) -> Path:
    """A directory the generator accepts as a model path."""

    return Path("/tmp") if tmp_path is None else tmp_path


def _vae_dir(tmp_path=None) -> Path:
    return Path("/tmp") if tmp_path is None else tmp_path


class FakeSession:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple, dict]] = []
        self.reset_calls = 0
        self.close_calls = 0

    def generate(self, request, **kwargs):
        self.calls.append(("generate", (request,), kwargs))
        return SimpleNamespace(request=request, kwargs=kwargs, frames=3)

    def plan(self, request, **kwargs):
        self.calls.append(("plan", (request,), kwargs))
        return SimpleNamespace(request=request, kwargs=kwargs)

    def generate_semantic(self, plan, **kwargs):
        self.calls.append(("generate_semantic", (plan,), kwargs))
        return SimpleNamespace(plan=plan, kwargs=kwargs)

    def synthesize(self, semantic, **kwargs):
        self.calls.append(("synthesize", (semantic,), kwargs))
        return np.zeros((4, 64), dtype=np.float32)

    def decode(self, latents, **kwargs):
        self.calls.append(("decode", (latents,), kwargs))
        return np.zeros((2, 16), dtype=np.float32)

    def reset(self) -> None:
        self.reset_calls += 1

    def close(self) -> None:
        self.close_calls += 1


def _generator(tmp_path, *, session: FakeSession | None = None, builds: list | None = None, **kwargs):
    (tmp_path / "config.json").write_text("{}")
    generator = YuE2SongGenerator(tmp_path, vae_path=tmp_path, **kwargs)
    built = session if session is not None else FakeSession()

    def build():
        if builds is not None:
            builds.append(True)
        return built

    generator._build = build
    return generator, built


def test_the_session_is_built_once_on_first_use(tmp_path) -> None:
    builds: list = []
    generator, session = _generator(tmp_path, builds=builds)
    assert builds == []
    assert generator.loaded is False
    assert generator.session() is session
    assert generator.session() is session
    assert builds == [True]
    assert generator.loaded is True


def test_reset_and_close_ignore_an_unbuilt_session(tmp_path) -> None:
    generator, session = _generator(tmp_path)
    generator.reset()
    generator.close()
    assert session.reset_calls == 0
    assert session.close_calls == 0


def test_close_is_idempotent_and_reset_reaches_a_built_session(tmp_path) -> None:
    generator, session = _generator(tmp_path)
    generator.session()
    generator.reset()
    assert session.reset_calls == 1
    generator.close()
    generator.close()
    assert session.close_calls == 1
    assert generator.loaded is False


def test_the_build_uses_the_product_configuration(tmp_path, monkeypatch) -> None:
    """Two CFG branches, the declared AR context, and both checkpoints by path."""

    import hipengine.loading.yue2 as loading
    import hipengine.runtime.yue2_ar as ar_module
    import hipengine.runtime.yue2_nar as nar_module
    import hipengine.runtime.yue2_session as session_module
    import hipengine.runtime.yue2_vae as vae_module
    import hipengine.tokenization.yue2 as tokenization

    seen: dict = {}

    class FakeAr:
        def __init__(self, weights, **kwargs):
            seen["ar"] = kwargs
            seen["ar_weights"] = weights

    class FakeNar:
        def __init__(self, weights, ar):
            seen["nar"] = (weights, ar)

    class FakeVae:
        def __init__(self, weights):
            seen["vae_weights"] = weights
            self.sample_rate = 48000

    class FakeTokenizer:
        def __init__(self, path):
            seen["tokenizer_path"] = path

        def encode(self, text):
            return [1]

        def decode(self, ids):
            return "x"

    class FakeArSession:
        def __init__(self, runtime, *, encode, decode):
            seen["ar_session"] = (runtime, encode, decode)

    class FakeSession:
        def __init__(self, ar_session, nar, vae, **kwargs):
            seen["session"] = (ar_session, nar, vae, kwargs)

    monkeypatch.setattr(loading, "load_yue2_weights", lambda directory: ("weights", directory))
    monkeypatch.setattr(loading, "load_yue2_vae_decoder", lambda directory: ("vae", directory))
    monkeypatch.setattr(ar_module, "Yue2ArRuntime", FakeAr)
    monkeypatch.setattr(nar_module, "Yue2NarRuntime", FakeNar)
    monkeypatch.setattr(vae_module, "Yue2VaeRuntime", FakeVae)
    monkeypatch.setattr(session_module, "Yue2ArSession", FakeArSession)
    monkeypatch.setattr(session_module, "Yue2Session", FakeSession)
    monkeypatch.setattr(tokenization, "YuE2TextTokenizer", FakeTokenizer)

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "qwen.tiktoken").write_text("asset")
    vae_dir = tmp_path / "vae"
    vae_dir.mkdir()

    generator = YuE2SongGenerator(
        model_dir,
        vae_path=vae_dir,
        backend="hip_gfx1151",
        max_context=2048,
        vae_core_frames=8,
        vae_halo_frames=2,
    )
    generator.session()

    assert seen["ar"] == {"branches": 2, "max_context": 2048, "backend": "hip_gfx1151"}
    assert seen["ar_weights"] == ("weights", model_dir)
    assert seen["vae_weights"] == ("vae", vae_dir)
    assert seen["nar"][0] == ("weights", model_dir)
    assert seen["tokenizer_path"] == model_dir / "qwen.tiktoken"
    assert seen["session"][3] == {"vae_core_frames": 8, "vae_halo_frames": 2}


def test_a_missing_tokenizer_asset_is_reported(tmp_path) -> None:
    generator = YuE2SongGenerator(tmp_path, vae_path=tmp_path)
    with pytest.raises(FileNotFoundError) as excinfo:
        generator.session()
    assert "qwen.tiktoken" in str(excinfo.value)


# ---------------------------------------------------------------------------
# requests
# ---------------------------------------------------------------------------


def test_generate_song_assembles_the_request_and_forwards_the_overrides(tmp_path) -> None:
    generator, session = _generator(tmp_path)
    sentinel = object()
    result = generator.generate_song(
        style="warm pop",
        lyrics="[verse]\nhello",
        cot="melody",
        seed=7,
        abc="X:1",
        cfg_scale=1.5,
        request_id="song-a",
        steps=4,
        context=1024,
        tiled=False,
        cancelled=sentinel,
        on_token=sentinel,
        abc_sampling={"max_tokens": 8},
        semantic_sampling={"max_tokens": 16},
    )

    name, args, kwargs = session.calls[-1]
    request = args[0]
    assert name == "generate"
    assert (request.style, request.lyrics, request.cot, request.seed) == ("warm pop", "[verse]\nhello", "melody", 7)
    assert (request.abc, request.cfg_scale, request.id) == ("X:1", 1.5, "song-a")
    assert kwargs == {
        "abc_sampling": {"max_tokens": 8},
        "semantic_sampling": {"max_tokens": 16},
        "steps": 4,
        "context": 1024,
        "tiled": False,
        "cancelled": sentinel,
        "on_token": sentinel,
    }
    assert result.frames == 3


def test_omitted_fields_keep_the_model_defaults(tmp_path) -> None:
    from hipengine.generation.yue2 import SongRequest

    generator, session = _generator(tmp_path)
    generator.generate_song(style="s", lyrics="l")
    request = session.calls[-1][1][0]
    assert request == SongRequest(style="s", lyrics="l")


def test_generate_song_accepts_a_request_object_or_a_mapping(tmp_path) -> None:
    from hipengine.generation.yue2 import SongRequest

    generator, session = _generator(tmp_path)
    generator.generate_song(SongRequest(style="s", lyrics="l", cot="off"))
    assert session.calls[-1][1][0].cot == "off"
    generator.generate_song({"style": "s2", "lyrics": "l2", "cot": "melody"})
    assert session.calls[-1][1][0] == SongRequest(style="s2", lyrics="l2", cot="melody")


def test_generate_song_refuses_mixed_request_and_fields(tmp_path) -> None:
    generator, session = _generator(tmp_path)
    with pytest.raises(TypeError) as excinfo:
        generator.generate_song({"style": "s", "lyrics": "l"}, style="other")
    assert "style" in str(excinfo.value)
    assert session.calls == []


def test_generate_song_requires_style_and_lyrics(tmp_path) -> None:
    generator, session = _generator(tmp_path)
    with pytest.raises(ValueError) as excinfo:
        generator.generate_song(style="only a style")
    assert "style and lyrics" in str(excinfo.value)
    assert session.calls == []


def test_generate_song_rejects_an_unsupported_request_type(tmp_path) -> None:
    generator, _session = _generator(tmp_path)
    with pytest.raises(TypeError) as excinfo:
        generator.generate_song(42)
    assert "SongRequest" in str(excinfo.value)


def test_the_staged_methods_reach_the_session(tmp_path) -> None:
    generator, session = _generator(tmp_path)
    plan = generator.plan(style="s", lyrics="l", cot="full", abc_sampling={"max_tokens": 4})
    semantic = generator.generate_semantic(plan, semantic_sampling={"max_tokens": 8})
    latents = generator.synthesize(semantic, steps=2)
    audio = generator.decode(latents, tiled=False)

    assert [name for name, _args, _kwargs in session.calls] == [
        "plan",
        "generate_semantic",
        "synthesize",
        "decode",
    ]
    # The phase-named override becomes the stage's own ``sampling`` argument.
    assert session.calls[0][2] == {"sampling": {"max_tokens": 4}}
    assert session.calls[1][2] == {"sampling": {"max_tokens": 8}}
    assert session.calls[2][2] == {"steps": 2}
    assert session.calls[3][2] == {"tiled": False}
    assert audio.shape == (2, 16)


def test_the_staged_methods_keep_the_session_spelling_and_refuse_both(tmp_path) -> None:
    generator, session = _generator(tmp_path)
    generator.plan(style="s", lyrics="l", sampling={"max_tokens": 4})
    assert session.calls[-1][2] == {"sampling": {"max_tokens": 4}}
    with pytest.raises(TypeError) as excinfo:
        generator.plan(style="s", lyrics="l", abc_sampling={"max_tokens": 4}, sampling={"max_tokens": 8})
    assert "not both" in str(excinfo.value)
    with pytest.raises(TypeError):
        generator.generate_semantic(object(), semantic_sampling={}, sampling={})


# ---------------------------------------------------------------------------
# declared context and capabilities
# ---------------------------------------------------------------------------


def test_the_declared_resident_context_sizes_the_ar_context(tmp_path) -> None:
    generator, _session = _generator(tmp_path)
    assert generator.max_context == DEFAULT_AR_CONTEXT
    generator.declare_max_sequence_length(8192)
    assert generator.max_context == 8192


def test_a_context_beyond_the_span_attention_bound_is_refused(tmp_path) -> None:
    from hipengine.runtime.yue2_ar import SPAN_ATTENTION_MAX_CONTEXT

    generator, _session = _generator(tmp_path)
    with pytest.raises(ValueError) as excinfo:
        generator.declare_max_sequence_length(SPAN_ATTENTION_MAX_CONTEXT + 1)
    assert str(SPAN_ATTENTION_MAX_CONTEXT) in str(excinfo.value)
    with pytest.raises(ValueError):
        YuE2SongGenerator(tmp_path, vae_path=tmp_path, max_context=0)


def test_song_capabilities_report_the_surface_and_the_checkpoints(tmp_path) -> None:
    generator, _session = _generator(tmp_path)
    capabilities = generator.song_capabilities()
    assert capabilities["surface"] == "song"
    assert capabilities["sample_rate"] == 48000
    assert capabilities["channels"] == 2
    assert capabilities["cot_modes"] == ["full", "melody", "off"]
    assert capabilities["context"]["ar_span_attention"] == DEFAULT_AR_CONTEXT
    assert capabilities["checkpoints"] == {"model": str(tmp_path), "vae": str(tmp_path)}
    assert capabilities["loaded"] is False
    generator.session()
    assert generator.song_capabilities()["loaded"] is True


def test_the_factory_takes_the_declared_context_from_either_spelling(tmp_path) -> None:
    (tmp_path / "config.json").write_text("{}")
    assert make_yue2_song_generator(
        model_path=tmp_path, vae_model_path=tmp_path, max_sequence_length=3072
    ).max_context == 3072
    assert make_yue2_song_generator(
        model_path=tmp_path, vae_model_path=tmp_path, max_context=512, max_sequence_length=3072
    ).max_context == 512
    assert make_yue2_song_generator(model_path=tmp_path, vae_model_path=tmp_path).max_context == (
        DEFAULT_AR_CONTEXT
    )


# ---------------------------------------------------------------------------
# the LLM entry point
# ---------------------------------------------------------------------------


def _install_fake_model(monkeypatch, plugin, *, factory, model_path="/tmp/fake-model"):
    import hipengine.generation as generation
    import hipengine.loading as loading
    import hipengine.models as models

    from hipengine.generation import register_text_generator

    fake_index = SimpleNamespace(
        config={"architectures": list(plugin.architectures)}, model_path=model_path
    )
    monkeypatch.setattr(generation, "register_builtin_generators", lambda: None)
    monkeypatch.setattr(loading, "load_weight_index", lambda model: fake_index)
    monkeypatch.setattr(models, "resolve_model", lambda architecture: plugin)
    register_text_generator(
        model=plugin.name,
        backend="fake_backend",
        quant="fake_quant",
        factory=factory,
        replace=True,
    )
    return fake_index


class FakeSongGenerator:
    generation_surfaces = ("song",)

    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.declared: list[int] = []

    def declare_max_sequence_length(self, value: int) -> None:
        self.declared.append(int(value))

    def generate_song(self, request, **kwargs):
        self.calls.append((request, kwargs))
        return SimpleNamespace(request=request, kwargs=kwargs)


def test_llm_generate_song_dispatches_through_the_registry(monkeypatch) -> None:
    from hipengine import LLM

    generator = FakeSongGenerator()
    plugin = SimpleNamespace(
        name="fake_song",
        architectures=("FakeSongForCausalLM",),
        generation_surfaces=("song",),
    )
    _install_fake_model(monkeypatch, plugin, factory=lambda **kwargs: generator)

    llm = LLM("/tmp/fake-model", backend="fake_backend", quant="fake_quant", max_sequence_length=3072)
    result = llm.generate_song("warm pop", "[verse]\nhi", cot="off", seed=5, steps=2, tiled=False)

    assert result.kwargs["style"] == "warm pop"
    assert result.kwargs["lyrics"] == "[verse]\nhi"
    assert result.kwargs["cot"] == "off"
    assert result.kwargs["seed"] == 5
    assert result.kwargs["steps"] == 2
    assert result.kwargs["tiled"] is False
    assert result.kwargs["abc_sampling"] is None
    # The declared resident context reached the generator that allocates for it.
    assert generator.declared == [3072]
    assert llm.supports_song_generation is True
    assert llm.song_generator() is llm._text_generator


def test_llm_forwards_the_companion_decoder_to_the_factory(monkeypatch) -> None:
    from hipengine import LLM

    seen: dict = {}

    def factory(**kwargs):
        seen.update(kwargs)
        return FakeSongGenerator()

    plugin = SimpleNamespace(
        name="fake_song",
        architectures=("FakeSongForCausalLM",),
        generation_surfaces=("song",),
    )
    _install_fake_model(monkeypatch, plugin, factory=factory)

    llm = LLM("/tmp/fake-model", backend="fake_backend", quant="fake_quant", vae_model="/tmp/vae")
    llm.song_generator()
    assert seen["vae_model_path"] == "/tmp/vae"
    assert seen["model_path"] == "/tmp/fake-model"

    with pytest.raises(ValueError):
        LLM("/tmp/fake-model", vae_model="")


def test_llm_refuses_song_generation_without_the_surface(monkeypatch) -> None:
    from hipengine import LLM

    class FakeTextGenerator:
        def generate(self, request):
            return ["x"]

    plugin = SimpleNamespace(name="fake_text", architectures=("FakeForCausalLM",))
    _install_fake_model(monkeypatch, plugin, factory=lambda **kwargs: FakeTextGenerator())

    llm = LLM("/tmp/fake-model", backend="fake_backend", quant="fake_quant")
    assert llm.generation_surfaces == ("text",)
    assert llm.supports_song_generation is False
    with pytest.raises(NotImplementedError) as excinfo:
        llm.generate_song("style", "lyrics")
    assert "song generation" in str(excinfo.value)
    with pytest.raises(NotImplementedError):
        llm.song_generator()


def test_generation_surfaces_helper_reads_the_declaration() -> None:
    from hipengine.models import DEFAULT_GENERATION_SURFACES, generation_surfaces

    assert DEFAULT_GENERATION_SURFACES == ("text",)
    assert generation_surfaces(SimpleNamespace()) == ("text",)
    assert generation_surfaces(SimpleNamespace(generation_surfaces=("song",))) == ("song",)
    with pytest.raises(TypeError):
        generation_surfaces(SimpleNamespace(generation_surfaces="song"))
    with pytest.raises(ValueError):
        generation_surfaces(SimpleNamespace(generation_surfaces=()))
    with pytest.raises(ValueError):
        generation_surfaces(SimpleNamespace(generation_surfaces=("song", " ")))


def test_the_yue2_plugin_declares_song_and_others_declare_text() -> None:
    from hipengine.models import generation_surfaces, registered_models

    by_name = {plugin.name: plugin for plugin in registered_models()}
    assert generation_surfaces(by_name["yue2"]) == ("song",)
    assert generation_surfaces(by_name["qwen3_5_gguf"]) == ("text",)
