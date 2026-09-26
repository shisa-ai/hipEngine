"""M6 product-session contracts: lifecycle, serialization, identities, stages.

These run against deterministic stand-ins for the three runtimes, so they need no
GPU and no checkpoints. The GPU end-to-end gate is
``scripts/yue2_e2e_gate.py``, which replays the recorded 12-case prefix/semantic
fixtures through the real NAR and VAE runtimes and runs one live greedy request.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import json

import numpy as np
import pytest

from hipengine.generation.yue2 import GenerationConfig, Sampling, SongRequest
from tests._torch_absence import run_in_clean_interpreter
from hipengine.runtime.yue2_session import (
    SemanticResult,
    SongResult,
    SymbolicPlan,
    Yue2Session,
    Yue2SessionBusy,
    canonical_identity,
    tensor_identity,
)


class _Weights:
    latent_dim = 64

    def __init__(self, name: str):
        self.identity = {"name": name, "sha256": name * 4}


class _ArRuntime:
    def __init__(self):
        self.weights = _Weights("model")


class FakeArSession:
    """Stand-in for ``Yue2ArSession`` that emits a fixed token trajectory."""

    def __init__(self, tokens=(11, 12, 13)):
        self.runtime = _ArRuntime()
        self.config = GenerationConfig()
        self.encode = lambda text: [1, 2, 3]
        self.decode = lambda ids: "abc"
        self._tokens = tuple(tokens)
        self.resets = 0
        self.closed = False
        self.plan_calls = 0
        self.abc_sampling = None
        self.semantic_sampling = None

    def plan(self, request, *, sampling=None, **kwargs) -> SymbolicPlan:
        from hipengine.generation.yue2 import token_prefixes

        self.plan_calls += 1
        self.abc_sampling = sampling
        return SymbolicPlan(
            request=request,
            abc=None,
            abc_ids=(),
            prefix=tuple(token_prefixes(request, self.encode)),
            timing={"seconds": 0.0, "output_tokens": 0},
            truncated=False,
        )

    def generate_semantic(self, plan, *, sampling=None, **kwargs) -> SemanticResult:
        # Apply the budget the way the real loop does, so a session whose config never
        # reaches generation cannot pass a cap-shaped test.
        self.semantic_sampling = sampling
        budget = self.config.semantic.max_tokens if sampling is None else sampling.max_tokens
        tokens = self._tokens[:budget]
        return SemanticResult(
            plan=plan, tokens=tokens,
            timing={"seconds": 0.0, "output_tokens": len(tokens)}, truncated=False,
        )

    def reset(self) -> None:
        self.resets += 1

    def close(self) -> None:
        self.closed = True


class FakeNarRuntime:
    def __init__(self, frames: int = 4):
        self.frames = frames
        self.conditions = 0
        self.solved = 0
        self.closed = False

    def condition(self, chunk) -> None:
        self.conditions += 1

    def solve(self, steps: int) -> np.ndarray:
        self.solved += 1
        # Deterministic in (frames, steps): the session must be reproducible
        # whenever its runtimes are.
        return np.full((self.frames, 64), float(steps), dtype=np.float32)

    def close(self) -> None:
        self.closed = True


class FakeVaeRuntime:
    sample_rate = 48000

    def __init__(self):
        self.weights = _Weights("vae")
        self.calls: list[str] = []
        self.closed = False

    def decode(self, latent):
        self.calls.append("full")
        assert latent.shape[1] == 64, "decoder receives channel-first latents"
        return np.ones((1, 2, latent.shape[-1] * 1920 - 64), dtype=np.float32)

    def decode_tiled(self, latent, *, core_frames, halo_frames):
        self.calls.append(f"tiled:{core_frames}/{halo_frames}")
        assert latent.shape[1] == 64, "decoder receives channel-first latents"
        return np.ones((1, 2, latent.shape[-1] * 1920 - 64), dtype=np.float32)

    def close(self) -> None:
        self.closed = True


@pytest.fixture()
def session():
    instance = Yue2Session(FakeArSession(), FakeNarRuntime(), FakeVaeRuntime())
    yield instance
    instance.close()


def _request() -> SongRequest:
    return SongRequest(style="pop", lyrics="la", cot="off", seed=7, id="unit")


# -- identities ---------------------------------------------------------

def test_tensor_identity_is_content_addressed():
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    assert tensor_identity(a) == tensor_identity(a.copy())
    assert tensor_identity(a) != tensor_identity(a + 1)
    # Shape is part of the identity, not just the bytes; the hash is over FP32
    # values, so a same-valued wider dtype is the same tensor.
    assert tensor_identity(a) != tensor_identity(a.reshape(3, 2))
    assert tensor_identity(a) == tensor_identity(a.astype(np.float64))


def test_canonical_identity_ignores_key_order():
    assert canonical_identity({"a": 1, "b": [1, 2]}) == canonical_identity({"b": [1, 2], "a": 1})
    assert canonical_identity({"a": 1}) != canonical_identity({"a": 2})


# -- lifecycle ----------------------------------------------------------

def test_close_is_idempotent_and_closes_every_runtime(session):
    session.close()
    session.close()
    assert session.nar.closed and session.vae.closed and session.ar.closed


def test_closed_session_refuses_work(session):
    session.close()
    with pytest.raises(RuntimeError, match="session is closed"):
        session.plan(_request())


def test_a_running_request_serializes_the_session(session):
    session._enter()
    try:
        with pytest.raises(Yue2SessionBusy):
            session.generate(_request())
    finally:
        session._leave()
    # The guard is released, so the session is usable again.
    assert session.plan(_request()).prefix == tuple(
        __import__("hipengine.generation.yue2", fromlist=["x"]).token_prefixes(
            _request(), session.ar.encode
        )
    )


def test_reset_keeps_the_session_usable(session):
    session.reset()
    assert session.ar.resets == 1


# -- stages -------------------------------------------------------------

def test_synthesize_uses_the_plans_own_prefix(session):
    semantic = session.generate_semantic(session.plan(_request()))
    latents = session.synthesize(semantic)
    assert latents.shape == (4, 64)
    assert session.nar.conditions == 1 and session.nar.solved == 1


def test_synthesize_rejects_a_semantic_result_whose_prefix_drifted(session):
    semantic = session.generate_semantic(session.plan(_request()))
    tampered = SemanticResult(
        plan=SymbolicPlan(
            request=semantic.plan.request,
            abc=None,
            abc_ids=(),
            prefix=(999,),
            timing={},
            truncated=False,
        ),
        tokens=semantic.tokens,
        timing={},
        truncated=False,
    )
    with pytest.raises(ValueError, match="exact prefix"):
        session.synthesize(tampered)


def test_synthesize_honours_cancellation(session):
    semantic = session.generate_semantic(session.plan(_request()))
    with pytest.raises(InterruptedError, match="Cancelled before acoustic prefill"):
        session.synthesize(semantic, cancelled=lambda: True)
    assert session.nar.conditions == 0


def test_decode_normalizes_the_latent_layout(session):
    """The solver emits [frames, latent_dim]; the decoder wants [1, latent_dim, frames]."""

    latents = np.zeros((4, 64), dtype=np.float32)
    audio = session.decode(latents, tiled=True, core_frames=2, halo_frames=16)
    assert audio.shape == (2, 4 * 1920 - 64)
    assert session.vae.calls == ["tiled:2/16"]
    session.decode(latents, tiled=False)
    assert session.vae.calls[-1] == "full"
    # An already batched tensor is accepted in either layout.
    session.decode(np.zeros((1, 64, 4), dtype=np.float32), tiled=False)
    session.decode(np.zeros((1, 4, 64), dtype=np.float32), tiled=False)
    assert session.vae.calls[-2:] == ["full", "full"]


def test_effective_config_records_resolved_settings(session):
    config = session.effective_config(_request())
    assert config["ode_steps"] == 32 and config["ode_method"] == "midpoint"
    assert config["vae_halo_frames"] == 16 and config["vae_tiled"] is True
    assert config["cot"] == "off" and config["guidance"] == 1.01
    assert set(config["semantic"]) == {
        "temperature", "top_p", "top_k", "repetition_penalty", "penalty_window",
        "min_tokens", "max_tokens",
    }


def test_a_session_config_is_what_runs_not_just_what_is_recorded():
    """The recorded settings and the executed settings have to be the same object.

    The session's own ``GenerationConfig`` is what ``effective_config`` reports, so a
    request that never passes an override has to generate with it too. Resolving the
    default in the AR session instead let a session record a budget it did not use: a
    harness capped the semantic phase and generation ran to the AR session's own
    default.
    """

    ar = FakeArSession(tokens=tuple(range(1000, 1040)))
    config = GenerationConfig(
        abc=Sampling(max_tokens=4, min_tokens=0),
        semantic=Sampling(max_tokens=6, min_tokens=0),
    )
    instance = Yue2Session(ar, FakeNarRuntime(), FakeVaeRuntime(), config=config)
    try:
        result = instance.generate(_request())
        assert len(result.semantic.tokens) == 6, "the session's semantic budget must cap generation"
        assert ar.semantic_sampling is config.semantic
        assert ar.abc_sampling is config.abc
        recorded = instance.effective_config(_request())
        assert recorded["semantic"]["max_tokens"] == 6
        assert recorded["abc"]["max_tokens"] == 4
    finally:
        instance.close()


def test_an_explicit_override_still_beats_the_session_config():
    ar = FakeArSession(tokens=tuple(range(1000, 1040)))
    instance = Yue2Session(ar, FakeNarRuntime(), FakeVaeRuntime(),
                           config=GenerationConfig(semantic=Sampling(max_tokens=6, min_tokens=0)))
    try:
        result = instance.generate(_request(), semantic_sampling=Sampling(max_tokens=3, min_tokens=0))
        assert len(result.semantic.tokens) == 3
        assert instance.effective_config(
            _request(), semantic_sampling=Sampling(max_tokens=3, min_tokens=0)
        )["semantic"]["max_tokens"] == 3
    finally:
        instance.close()


def test_staged_generation_applies_the_session_config_too():
    ar = FakeArSession(tokens=tuple(range(1000, 1040)))
    instance = Yue2Session(ar, FakeNarRuntime(), FakeVaeRuntime(),
                           config=GenerationConfig(semantic=Sampling(max_tokens=5, min_tokens=0)))
    try:
        plan = instance.plan(_request())
        semantic = instance.generate_semantic(plan)
        assert len(semantic.tokens) == 5
        assert ar.semantic_sampling is instance.config.semantic
    finally:
        instance.close()


# -- end to end (stand-ins) ---------------------------------------------

def test_generate_returns_a_result_with_identities(session):
    result = session.generate(_request())
    assert isinstance(result, SongResult)
    assert result.sample_rate == 48000
    assert result.frames == 4
    assert result.audio.shape == (2, 4 * 1920 - 64)
    assert result.latent_identity == tensor_identity(result.latents)
    assert result.audio_identity == tensor_identity(result.audio)
    assert result.truncation == {"abc": False, "semantic": False}
    assert result.timing["e2e_seconds"] >= 0
    payload = result.to_dict()
    assert payload["request_id"] == result.request_id
    assert payload["weights"]["model"]["name"] == "model"
    assert payload["weights"]["vae"]["name"] == "vae"


def test_generate_is_serialized_and_usable_afterwards(session):
    first = session.generate(_request())
    second = session.generate(_request())
    assert first.request_id == second.request_id
    assert first.latent_identity == second.latent_identity


def test_result_save_writes_data_not_pickle(session, tmp_path):
    result = session.generate(_request())
    path = result.save(tmp_path / "out")
    payload = json.loads(path.read_text())
    assert payload["files"]["latents.npy"] == result.latent_identity
    assert payload["files"]["audio.npy"] == result.audio_identity
    assert (tmp_path / "out" / "plan.json").is_file()
    reloaded = np.load(tmp_path / "out" / "latents.npy")
    assert np.array_equal(reloaded, result.latents)
    # Round-trip the stages so a saved result can be re-synthesized or re-decoded
    # without the AR path; the stage loader re-validates tokens and prefix.
    semantic = SemanticResult.load(tmp_path / "out")
    assert semantic.tokens == result.semantic.tokens
    assert semantic.plan.prefix == result.semantic.plan.prefix
    assert semantic.plan.abc_ids == result.semantic.plan.abc_ids


# -- torch-free product path --------------------------------------------

def test_the_product_path_never_imports_torch(session, request):
    """The whole staged path must stay torch-free.

    Everything the session reaches is imported here - the generation protocol,
    the AR session, the NAR kernel wrappers, the VAE runtime, the model plugin -
    and torch must not appear. A silently imported torch would invalidate the
    architectural claim rather than fail loudly.

    Runs in a clean child interpreter: a sibling unit test that imports torch
    must not decide this claim through suite order.
    """

    if not run_in_clean_interpreter(request.node.nodeid):
        return

    import sys

    import hipengine.models  # noqa: F401  (registers the YuE2 plugin)
    from hipengine.models import resolve_model
    from hipengine.runtime.yue2_nar import song_chunks  # noqa: F401
    from hipengine.runtime.yue2_vae import Yue2VaeRuntime  # noqa: F401

    assert resolve_model("yue2").name == "yue2"
    assert resolve_model("YuE2ForCausalLM").name == "yue2"
    result = session.generate(_request())
    assert result.audio_identity == tensor_identity(result.audio)
    assert "torch" not in sys.modules
