"""YuE2 song generation as a registered generation surface.

``LLM.generate_song`` and the server's song route both reach this object, so one
resident process builds the pipeline once and reuses it across requests. Nothing
loads weights at import time or at construction: the session is built on the
first request, which is what lets a server start, report readiness, and only
then pay the load.

The pipeline itself belongs to the model plugin and lives in
``hipengine/runtime/yue2_*.py``; this module owns registration, the companion
decoder path, and the request shape.
"""

from __future__ import annotations

import glob
import os
from collections.abc import Mapping
from functools import partial
from pathlib import Path
from typing import Any

from hipengine.generation.registry import register_text_generator

#: The companion decoder checkpoint. The pipeline cannot produce audio without it.
VAE_ENV = "HIPENGINE_YUE2_VAE_DIR"
VAE_REPO_DIRNAME = "models--m-a-p--YuE2-Vae"

#: The AR span-attention route allocates its K/V for this many positions, and the
#: runtime rejects a longer prefill. It is the pipeline's real context bound even
#: though the checkpoint itself declares 24576 positions.
DEFAULT_AR_CONTEXT = 4096
MODEL_CONTEXT = 24576

SAMPLE_RATE = 48000
CHANNELS = 2
DEFAULT_ODE_STEPS = 32


def _snapshot_dir(repo_dir: Path) -> Path:
    """Pick the checkpoint revision a Hugging Face cache directory points at.

    A ref is a statement about which revision the user fetched; a directory name
    is not. Choosing among several snapshots by name would silently pick a
    different checkpoint, so more than one candidate is an error rather than a
    guess.
    """

    ref = repo_dir / "refs" / "main"
    if ref.is_file():
        revision = ref.read_text().strip()
        if revision:
            snapshot = repo_dir / "snapshots" / revision
            if snapshot.is_dir():
                return snapshot
    snapshots = sorted(
        Path(candidate)
        for candidate in glob.glob(str(repo_dir / "snapshots" / "*"))
        if Path(candidate).is_dir()
    )
    if len(snapshots) == 1:
        return snapshots[0]
    if not snapshots:
        raise FileNotFoundError(
            f"no snapshot under {repo_dir / 'snapshots'}; the cached checkpoint is incomplete"
        )
    listed = ", ".join(str(candidate) for candidate in snapshots)
    raise ValueError(
        f"{repo_dir} holds {len(snapshots)} snapshots ({listed}); pass the checkpoint "
        "directory explicitly instead of relying on the cache"
    )


def resolve_vae_path(
    vae_path: str | Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    cache_root: str | Path | None = None,
) -> Path:
    """Resolve the ``m-a-p/YuE2-Vae`` directory the pipeline decodes with.

    Explicit path first, then :data:`VAE_ENV`, then the Hugging Face cache. A
    missing companion is reported by name, because a song request that fails at
    decode has already spent minutes of generation.
    """

    env = os.environ if environ is None else environ
    candidate = vae_path if vae_path is not None else env.get(VAE_ENV)
    if candidate is not None and str(candidate).strip():
        directory = Path(str(candidate)).expanduser()
        if not directory.is_dir():
            raise FileNotFoundError(f"YuE2 VAE directory does not exist: {directory}")
        return directory
    root = Path(cache_root) if cache_root is not None else Path.home() / ".cache/huggingface/hub"
    repo_dir = root / VAE_REPO_DIRNAME
    if not repo_dir.is_dir():
        raise FileNotFoundError(
            f"the YuE2 VAE decoder checkpoint was not found: pass its directory with "
            f"vae_model= (or set {VAE_ENV}); looked in {repo_dir}"
        )
    return _snapshot_dir(repo_dir)


class YuE2SongGenerator:
    """One resident YuE2 song pipeline, built on first use."""

    #: What this generator produces, for capability reporting and routing.
    generation_surfaces: tuple[str, ...] = ("song",)

    def __init__(
        self,
        model_path: str | Path,
        *,
        vae_path: str | Path | None = None,
        backend: str = "auto",
        max_context: int = DEFAULT_AR_CONTEXT,
        vae_core_frames: int = 1024,
        vae_halo_frames: int = 16,
        environ: Mapping[str, str] | None = None,
        cache_root: str | Path | None = None,
    ) -> None:
        directory = Path(str(model_path)).expanduser()
        if not directory.is_dir():
            raise FileNotFoundError(f"YuE2 model directory does not exist: {directory}")
        self.model_path = directory
        self.vae_path = resolve_vae_path(vae_path, environ=environ, cache_root=cache_root)
        self.backend = str(backend)
        self.max_context = self._check_context(max_context)
        self.vae_core_frames = int(vae_core_frames)
        self.vae_halo_frames = int(vae_halo_frames)
        self._session: Any | None = None

    # -- lifecycle ------------------------------------------------------
    @staticmethod
    def _check_context(value: int) -> int:
        from hipengine.runtime.yue2_ar import SPAN_ATTENTION_MAX_CONTEXT

        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("max_context must be an integer")
        if not 0 < int(value) <= SPAN_ATTENTION_MAX_CONTEXT:
            raise ValueError(
                f"max_context must be in [1, {SPAN_ATTENTION_MAX_CONTEXT}] for the "
                "AR span-attention route"
            )
        return int(value)

    def declare_max_sequence_length(self, value: int) -> None:
        """Size the AR context from the caller's declared resident context.

        This is the hook ``LLM(max_sequence_length=...)`` publishes through, so a
        declared bound reaches the runtime that actually allocates for it instead
        of being dropped.
        """

        self.max_context = self._check_context(int(value))

    @property
    def loaded(self) -> bool:
        return self._session is not None

    def session(self):
        """Return the resident session, building it on first use."""

        if self._session is None:
            self._session = self._build()
        return self._session

    def _build(self):
        from hipengine.loading.yue2 import load_yue2_vae_decoder, load_yue2_weights
        from hipengine.runtime.yue2_ar import Yue2ArRuntime
        from hipengine.runtime.yue2_nar import Yue2NarRuntime
        from hipengine.runtime.yue2_session import Yue2ArSession, Yue2Session
        from hipengine.runtime.yue2_vae import Yue2VaeRuntime
        from hipengine.tokenization.yue2 import YuE2TextTokenizer

        tokenizer_path = self.model_path / "qwen.tiktoken"
        if not tokenizer_path.is_file():
            raise FileNotFoundError(
                f"the YuE2 tokenizer asset is missing: {tokenizer_path}"
            )
        weights = load_yue2_weights(self.model_path)
        vae_weights = load_yue2_vae_decoder(self.vae_path)
        tokenizer = YuE2TextTokenizer(tokenizer_path)
        # The product configuration: the semantic phase uses classifier-free
        # guidance, which needs both branches. The NAR reads branch 0 only, so the
        # second branch costs AR time and nothing else.
        ar = Yue2ArRuntime(
            weights,
            branches=2,
            max_context=self.max_context,
            backend=self.backend,
        )
        nar = Yue2NarRuntime(weights, ar)
        vae = Yue2VaeRuntime(vae_weights)
        return Yue2Session(
            Yue2ArSession(ar, encode=tokenizer.encode, decode=tokenizer.decode),
            nar,
            vae,
            vae_core_frames=self.vae_core_frames,
            vae_halo_frames=self.vae_halo_frames,
        )

    def reset(self) -> None:
        """Drop request-local state; the resident pipeline stays usable."""

        if self._session is not None:
            self._session.reset()

    def close(self) -> None:
        """Release the resident pipeline. Idempotent."""

        session, self._session = self._session, None
        if session is not None:
            session.close()

    def __enter__(self) -> "YuE2SongGenerator":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- requests -------------------------------------------------------
    @staticmethod
    def _fields(**values) -> dict:
        """The request fields the caller actually supplied."""

        return {name: value for name, value in values.items() if value is not None}

    def _request(self, request: Any, fields: Mapping[str, Any]):
        """Resolve the stage's request from an object, a mapping, or fields."""

        from hipengine.generation.yue2 import SongRequest

        if request is not None:
            if fields:
                raise TypeError(
                    "pass either request or the individual request fields, not both: "
                    + ", ".join(sorted(fields))
                )
            if isinstance(request, SongRequest):
                return request
            if isinstance(request, Mapping):
                return SongRequest(**dict(request))
            raise TypeError("request must be a SongRequest or a mapping of its fields")
        if fields.get("style") is None or fields.get("lyrics") is None:
            raise ValueError("style and lyrics are required")
        return SongRequest(**fields)

    def generate_song(
        self,
        request: Any = None,
        *,
        style: Any = None,
        lyrics: Any = None,
        cot: Any = None,
        seed: Any = None,
        abc: Any = None,
        cfg_scale: Any = None,
        request_id: Any = None,
        steps: int | None = None,
        context: int | None = None,
        tiled: bool = True,
        cancelled: Any = None,
        on_token: Any = None,
        abc_sampling: Any = None,
        semantic_sampling: Any = None,
    ):
        """Plan, generate, solve, and decode one song end to end."""

        song = self._request(
            request,
            self._fields(
                style=style,
                lyrics=lyrics,
                cot=cot,
                seed=seed,
                abc=abc,
                cfg_scale=cfg_scale,
                id=request_id,
            ),
        )
        return self.session().generate(
            song,
            abc_sampling=abc_sampling,
            semantic_sampling=semantic_sampling,
            steps=steps,
            context=context,
            tiled=tiled,
            cancelled=cancelled,
            on_token=on_token,
        )

    def plan(
        self,
        request: Any = None,
        *,
        style: Any = None,
        lyrics: Any = None,
        cot: Any = None,
        seed: Any = None,
        abc: Any = None,
        cfg_scale: Any = None,
        request_id: Any = None,
        abc_sampling: Any = None,
        **kwargs,
    ):
        """Run the symbolic (ABC) stage only.

        ``abc_sampling`` names this phase's sampling, as it does on
        :meth:`generate_song`; the session's own ``sampling`` spelling also works.
        Other ``kwargs`` (``cancelled``, ``on_token``) go to the stage.
        """

        song = self._request(
            request,
            self._fields(
                style=style,
                lyrics=lyrics,
                cot=cot,
                seed=seed,
                abc=abc,
                cfg_scale=cfg_scale,
                id=request_id,
            ),
        )
        return self.session().plan(song, **self._stage_sampling(kwargs, abc_sampling))

    @staticmethod
    def _stage_sampling(kwargs: dict, sampling: Any) -> dict:
        """Fold a phase-named sampling override into the stage's ``sampling``."""

        if sampling is None:
            return kwargs
        if "sampling" in kwargs:
            raise TypeError("pass either sampling or the phase-named override, not both")
        return {**kwargs, "sampling": sampling}

    def generate_semantic(self, plan, *, semantic_sampling: Any = None, **kwargs):
        """Run the semantic phase over an existing plan."""

        return self.session().generate_semantic(
            plan, **self._stage_sampling(kwargs, semantic_sampling)
        )

    def synthesize(self, semantic, **kwargs):
        """Solve the acoustic flow-matching ODE for a semantic result."""

        return self.session().synthesize(semantic, **kwargs)

    def decode(self, latents, **kwargs):
        """Decode latents to stereo PCM."""

        return self.session().decode(latents, **kwargs)

    # -- capabilities ---------------------------------------------------
    def song_capabilities(self) -> dict[str, Any]:
        """Describe this surface for ``/v1/hipengine/capabilities`` and clients."""

        from hipengine.generation.yue2 import INSTRUCTIONS

        return {
            "surface": "song",
            "loaded": self.loaded,
            "sample_rate": SAMPLE_RATE,
            "channels": CHANNELS,
            "cot_modes": sorted(INSTRUCTIONS),
            "ode_steps": {"default": DEFAULT_ODE_STEPS},
            "context": {"ar_span_attention": self.max_context, "model_limit": MODEL_CONTEXT},
            "vae": {
                "tiled": True,
                "core_frames": self.vae_core_frames,
                "halo_frames": self.vae_halo_frames,
            },
            "torch_free": True,
            "serialized_requests": True,
            "backend": self.backend,
            "checkpoints": {"model": str(self.model_path), "vae": str(self.vae_path)},
        }


def make_yue2_song_generator(
    *,
    model_path,
    weight_index=None,
    model_plugin=None,
    vae_model_path=None,
    backend="auto",
    max_sequence_length=None,
    max_context=None,
):
    """Build the generator a registry key resolves to."""

    if max_context is None and max_sequence_length is not None:
        max_context = int(max_sequence_length)
    return YuE2SongGenerator(
        model_path,
        vae_path=vae_model_path,
        backend=backend,
        max_context=DEFAULT_AR_CONTEXT if max_context is None else int(max_context),
    )


for _backend in ("hip_gfx1100", "hip_gfx1151"):
    register_text_generator(
        model="yue2",
        backend=_backend,
        quant="bf16",
        factory=partial(make_yue2_song_generator, backend=_backend),
    )
