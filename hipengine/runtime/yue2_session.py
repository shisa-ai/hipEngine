"""Torch-free YuE2 AR generation: staged ``plan`` and ``generate_semantic`` loops.

The reference pipeline separates symbolic planning from semantic generation and
resets its request-local random stream at each phase. This module keeps that
structure over :class:`~hipengine.runtime.yue2_ar.Yue2ArRuntime`: the sampler,
the masks, the penalties and the CFG arithmetic are the pure functions in
:mod:`hipengine.generation.yue2`, and every emitted token is one runtime decode
step per branch.

Nothing here imports torch. Seeded identity is promised only for
:class:`~hipengine.generation.yue2.YuE2Random` (``numpy-pcg64-v1``); the
reference draws from PyTorch's generator, so cross-implementation seeded
equality is never claimed - temperature-zero requests are the ones that can be
compared token-for-token.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

from hipengine.generation.yue2 import (
    ABC_END,
    ABC_START,
    CODEC_OFFSET,
    CODEC_SIZE,
    CONTEXT,
    EOD,
    GenerationConfig,
    Sampling,
    SongRequest,
    YuE2Random,
    combine_cfg,
    distribution,
    distribution_windowed,
    MUSIC_START,
    VOCAB_SIZE,
    negative_prefix,
    phase_end_token,
    phase_window,
    resolve_sampling,
    softmax_f32,
    token_prefixes,
)
from hipengine.runtime.yue2_ar import Yue2ArRuntime, bf16_bits_to_f32
from hipengine.runtime.yue2_nar import song_chunks

Cancelled = Callable[[], bool] | None
TokenCallback = Callable[[str, int], None] | None


@dataclass(frozen=True)
class SymbolicPlan:
    """The ABC stage's output: exact IDs plus the assembled semantic prefix."""

    request: SongRequest
    abc: str | None
    abc_ids: tuple[int, ...]
    prefix: tuple[int, ...]
    timing: dict = field(default_factory=dict)
    truncated: bool = False

    def to_dict(self) -> dict:
        return {
            "protocol": "yue2-native-v1",
            "request": self.request.to_dict(),
            "abc": self.abc,
            "abc_ids": list(self.abc_ids),
            "prefix": list(self.prefix),
            "timing": self.timing,
            "truncated": self.truncated,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SymbolicPlan":
        plan = cls(
            request=SongRequest.from_dict(data["request"]),
            abc=data["abc"],
            abc_ids=tuple(int(token) for token in data["abc_ids"]),
            prefix=tuple(int(token) for token in data["prefix"]),
            timing=dict(data.get("timing") or {}),
            truncated=bool(data.get("truncated", False)),
        )
        plan.validate()
        return plan

    def validate(self) -> None:
        """Domain and structural checks applied to every plan, loaded or fresh."""
        _validate_ids(self.abc_ids, EOD, "ABC IDs must be ordinary text tokens")
        _validate_ids(self.prefix, VOCAB_SIZE, "prefix IDs must be vocabulary tokens")
        if self.request.cot == "off":
            if self.abc is not None or self.abc_ids:
                raise ValueError("The off mode has no symbolic stage")
            if self.prefix[-3:] != (ABC_START, ABC_END, MUSIC_START):
                raise ValueError("Off-mode prefix does not end with the empty ABC block")
            return
        if not self.abc_ids:
            raise ValueError("A symbolic plan must carry ABC IDs")
        if self.abc is None or not self.abc.strip():
            raise ValueError("A symbolic plan must carry ABC text")
        tail = self.prefix[-len(self.abc_ids) - 2 :]
        if tail[:-2] != self.abc_ids or tail[-2:] != (ABC_END, MUSIC_START):
            raise ValueError("Prefix does not embed the plan's exact ABC IDs")

    def save(self, directory: str | Path) -> Path:
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        path = target / "plan.json"
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")
        return path

    @classmethod
    def load(cls, directory: str | Path) -> "SymbolicPlan":
        return cls.from_dict(json.loads((Path(directory) / "plan.json").read_text()))


@dataclass(frozen=True)
class SemanticResult:
    """The semantic stage's output: raw codec IDs (offset removed)."""

    plan: SymbolicPlan
    tokens: tuple[int, ...]
    timing: dict = field(default_factory=dict)
    truncated: bool = False

    def to_dict(self) -> dict:
        return {
            "protocol": "yue2-native-v1",
            "tokens": list(self.tokens),
            "timing": self.timing,
            "truncated": self.truncated,
        }

    @classmethod
    def from_dict(cls, data: dict, plan: SymbolicPlan) -> "SemanticResult":
        result = cls(
            plan=plan,
            tokens=tuple(int(token) for token in data["tokens"]),
            timing=dict(data.get("timing") or {}),
            truncated=bool(data.get("truncated", False)),
        )
        _validate_ids(result.tokens, CODEC_SIZE, "semantic codes must be raw codec IDs")
        return result

    def save(self, directory: str | Path) -> Path:
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        self.plan.save(target)
        path = target / "semantic.json"
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")
        return path

    @classmethod
    def load(cls, directory: str | Path) -> "SemanticResult":
        target = Path(directory)
        plan = SymbolicPlan.load(target)
        return cls.from_dict(json.loads((target / "semantic.json").read_text()), plan)


def _validate_ids(ids: Sequence[int], limit: int, message: str) -> None:
    for token in ids:
        if not 0 <= int(token) < limit:
            raise ValueError(f"{message}; got {token}")


def generate_tokens(
    runtime: Yue2ArRuntime,
    prefix: Sequence[int],
    sampling: Sampling,
    seed: int,
    phase: str,
    *,
    negative: Sequence[int] | None = None,
    cfg_scale: float = 1.0,
    legacy_off: bool = False,
    cancelled: Cancelled = None,
    on_token: TokenCallback = None,
) -> tuple[list[int], dict, bool]:
    """Reference AR loop over the HIP runtime.

    Returns ``(content_ids, timing, truncated)``. ``content_ids`` excludes the
    phase's end token; ``truncated`` is true when the budget ran out before the
    end token arrived, which is the reference's distinction between an EOS exit
    and a budget exit.
    """

    prefix = [int(token) for token in prefix]
    if not prefix:
        raise ValueError("empty prefix")
    if len(prefix) + sampling.max_tokens > CONTEXT:
        raise ValueError("Prefix + requested generation budget exceeds 24576; no implicit truncation")
    if cfg_scale != 1 and negative is None:
        raise ValueError("CFG requires a negative prefix")
    if negative is not None:
        negative = [int(token) for token in negative]
        if len(negative) + sampling.max_tokens > CONTEXT:
            raise ValueError("Negative prefix + generation budget exceeds context")
        if runtime.branches < 2:
            raise ValueError("this runtime was built with branches=1; CFG needs two branches")
    if cancelled is not None and cancelled():
        raise InterruptedError("Cancelled before prefill")

    if phase not in ("abc", "semantic"):
        raise ValueError("phase must be abc or semantic")
    end = phase_end_token(phase)

    generator = YuE2Random(seed)
    runtime.reset()
    start = time.perf_counter()
    runtime.prefill_host_rows([runtime.embed_row(token) for token in prefix], branch=0, start_pos=0)
    if negative is not None:
        runtime.prefill_host_rows([runtime.embed_row(token) for token in negative], branch=1, start_pos=0)
    prefill_seconds = time.perf_counter() - start

    history: list[int] = []
    first: float | None = None
    eos = False
    # A phase can only select rows inside its own window, so the head projects that slice
    # of the weight and the sampler works in window coordinates. Every stage is the same
    # arithmetic the full-row path runs over the rows the phase can select; see
    # `distribution_windowed`.
    window = phase_window(phase)
    low, high = window
    for step in range(sampling.max_tokens):
        if cancelled is not None and cancelled():
            raise InterruptedError(f"Cancelled during {phase}")
        conditional = bf16_bits_to_f32(runtime.logits(0, domain=window)[low:high])
        if negative is None:
            logits = conditional
        else:
            unconditional = bf16_bits_to_f32(runtime.logits(1, domain=window)[low:high])
            logits = combine_cfg(conditional, unconditional, cfg_scale)
        scores = distribution_windowed(
            logits, sampling, history, step, phase, window, legacy_off=legacy_off
        )
        if sampling.temperature == 0:
            token = scores.argmax_token()
        else:
            token = scores.offset + generator.sample_categorical(softmax_f32(scores.values))
        if first is None:
            first = time.perf_counter() - start
        if on_token is not None:
            on_token(phase, token)
        if token == end:
            eos = True
            break
        history.append(token)
        if step + 1 < sampling.max_tokens:
            row = runtime.embed_row(token)
            position = runtime.context_length(0)
            runtime.push_token(row, position, branch=0)
            runtime.forward_layers(position, branch=0)
            if negative is not None:
                negative_position = runtime.context_length(1)
                runtime.push_token(row, negative_position, branch=1)
                runtime.forward_layers(negative_position, branch=1)
    seconds = time.perf_counter() - start
    count = len(history) + int(eos)
    timing = {
        "seconds": seconds,
        "prefill_seconds": prefill_seconds,
        "ttft_seconds": first,
        "output_tokens": count,
        "content_tokens": len(history),
        "output_tps": (count / seconds) if seconds > 0 else None,
        "prefix_tokens": len(prefix),
        "negative_prefix_tokens": len(negative) if negative is not None else 0,
        "cfg_branches": 1 if negative is None else 2,
        "execution": "eager",
        "attention": "spans",
        "random": generator.state(),
        # The live PCG64 position, not just the algorithm and seed: two runs that
        # consumed the same stream share it, and equal digests mean every later draw
        # agrees too.
        "random_state_digest": generator.state_digest(),
    }
    return history, timing, not eos


class Yue2ArSession:
    """Staged YuE2 AR generation over one resident runtime.

    ``plan`` and ``generate_semantic`` mirror the reference's staged API. Each
    call resets the runtime's per-branch contexts first, so a cancelled or
    failed request cannot leak into the next one, and the two phases draw from
    independent request-local random streams seeded with the request's seed.
    """

    def __init__(
        self,
        runtime: Yue2ArRuntime,
        *,
        encode: Callable[[str], list[int]],
        decode: Callable[[Iterable[int]], str],
        config: GenerationConfig | None = None,
    ) -> None:
        self.runtime = runtime
        self.encode = encode
        self.decode = decode
        self.config = config or GenerationConfig()
        self.last_fallback_reason: str | None = None

    # -- staged API ----------------------------------------------------
    def plan(
        self,
        request: SongRequest,
        *,
        sampling: Sampling | dict | None = None,
        cancelled: Cancelled = None,
        on_token: TokenCallback = None,
    ) -> SymbolicPlan:
        if request.cot == "off":
            return SymbolicPlan(
                request=request,
                abc=None,
                abc_ids=(),
                prefix=tuple(token_prefixes(request, self.encode)),
                timing={"seconds": 0.0, "output_tokens": 0},
                truncated=False,
            )
        if request.abc is not None:
            ids = self.encode(request.abc)
            return SymbolicPlan(
                request=request,
                abc=request.abc,
                abc_ids=tuple(ids),
                prefix=tuple(token_prefixes(request, self.encode, ids)),
                timing={"seconds": 0.0, "output_tokens": 0, "external_prefix_tokens": len(ids)},
                truncated=False,
            )
        resolved = resolve_sampling(sampling, self.config.abc)
        prefix = token_prefixes(request, self.encode)
        ids, timing, truncated = generate_tokens(
            self.runtime,
            prefix,
            resolved,
            request.seed,
            "abc",
            cancelled=cancelled,
            on_token=on_token,
        )
        self.last_fallback_reason = self.runtime.prefill_fallback_reason
        return SymbolicPlan(
            request=request,
            abc=self.decode(ids),
            abc_ids=tuple(ids),
            prefix=tuple(token_prefixes(request, self.encode, ids)),
            timing=timing,
            truncated=truncated,
        )

    def generate_semantic(
        self,
        plan: SymbolicPlan,
        *,
        sampling: Sampling | dict | None = None,
        cancelled: Cancelled = None,
        on_token: TokenCallback = None,
    ) -> SemanticResult:
        if not isinstance(plan, SymbolicPlan):
            raise TypeError("Pass the SymbolicPlan returned by plan()")
        request = plan.request
        expected = tuple(token_prefixes(request, self.encode, plan.abc_ids))
        if expected != tuple(plan.prefix):
            raise ValueError("Plan prefix disagrees with request/exact ABC IDs")
        resolved = resolve_sampling(sampling, self.config.semantic)
        negative = (
            negative_prefix(request, self.encode, plan.abc_ids)
            if request.needs_negative_branch
            else None
        )
        ids, timing, truncated = generate_tokens(
            self.runtime,
            plan.prefix,
            resolved,
            request.seed,
            "semantic",
            negative=negative,
            cfg_scale=request.guidance,
            legacy_off=request.cot == "off",
            cancelled=cancelled,
            on_token=on_token,
        )
        self.last_fallback_reason = self.runtime.prefill_fallback_reason
        return SemanticResult(
            plan=plan,
            tokens=tuple(int(token) - CODEC_OFFSET for token in ids),
            timing=timing,
            truncated=truncated,
        )

    def run(
        self,
        request: SongRequest,
        *,
        abc_sampling: Sampling | dict | None = None,
        semantic_sampling: Sampling | dict | None = None,
        cancelled: Cancelled = None,
        on_token: TokenCallback = None,
    ) -> SemanticResult:
        plan = self.plan(request, sampling=abc_sampling, cancelled=cancelled, on_token=on_token)
        return self.generate_semantic(
            plan, sampling=semantic_sampling, cancelled=cancelled, on_token=on_token
        )

    def reset(self) -> None:
        self.runtime.reset()

    def close(self) -> None:
        self.runtime.close()


# ---------------------------------------------------------------------------
# product session
# ---------------------------------------------------------------------------

def canonical_identity(payload: dict) -> str:
    """Stable SHA256 of a JSON-canonical payload (sorted keys, no whitespace)."""

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def tensor_identity(array: np.ndarray) -> str:
    """Content hash of a tensor: dtype, shape, then FP32-contiguous bytes."""

    host = np.ascontiguousarray(np.asarray(array, dtype=np.float32))
    digest = hashlib.sha256()
    digest.update(f"{host.dtype.str}{host.shape}".encode("ascii"))
    digest.update(host.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class SongResult:
    """One finished request: audio plus the identities of everything it produced.

    ``audio`` is ``[channels, samples]`` FP32 at ``sample_rate``, unclipped, as
    the reference decoder returns it. ``semantic`` retains the plan and the raw
    semantic codes, so a saved result can be re-synthesized or re-decoded without
    regenerating anything.
    """

    audio: np.ndarray
    sample_rate: int
    semantic: SemanticResult
    latents: np.ndarray
    config: dict
    weights: dict
    timing: dict
    request_id: str
    latent_identity: str
    audio_identity: str

    @property
    def frames(self) -> int:
        return int(self.latents.shape[0])

    @property
    def duration_seconds(self) -> float:
        return float(self.audio.shape[-1]) / float(self.sample_rate)

    @property
    def truncation(self) -> dict:
        return {
            "abc": bool(self.semantic.plan.truncated),
            "semantic": bool(self.semantic.truncated),
        }

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "sample_rate": self.sample_rate,
            "frames": self.frames,
            "samples": int(self.audio.shape[-1]),
            "channels": int(self.audio.shape[0]),
            "duration_seconds": self.duration_seconds,
            "latent_identity": self.latent_identity,
            "audio_identity": self.audio_identity,
            "abc_ids": list(self.semantic.plan.abc_ids),
            "abc": self.semantic.plan.abc,
            "semantic_tokens": list(self.semantic.tokens),
            "truncation": self.truncation,
            "config": self.config,
            "weights": self.weights,
            "timing": self.timing,
            "request": self.semantic.plan.request.to_dict(),
        }

    def save(self, directory: str | Path) -> Path:
        """Write ``result.json``, ``latents.npy`` and ``audio.npy``.

        Tensors are plain ``.npy`` data, never pickle, so a saved result is data
        rather than an executable object.
        """

        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        # The stages keep their own validated serialization (``plan.json`` and
        # ``semantic.json``), which re-checks hashes, configuration, shapes and
        # token domains on reload.
        self.semantic.save(target)
        np.save(target / "latents.npy", np.ascontiguousarray(self.latents, dtype=np.float32))
        np.save(target / "audio.npy", np.ascontiguousarray(self.audio, dtype=np.float32))
        payload = self.to_dict()
        payload["files"] = {
            "plan.json": "plan",
            "semantic.json": "semantic",
            "latents.npy": tensor_identity(self.latents),
            "audio.npy": tensor_identity(self.audio),
        }
        path = target / "result.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return path


class Yue2SessionBusy(RuntimeError):
    """Raised when a second request is issued against a session that is running."""


class Yue2Session:
    """The complete torch-free YuE2 product path.

    Stages compose the three runtimes: the AR session plans and generates
    semantic codes, the NAR runtime conditions on the plan's exact prefix and
    solves the flow-matching ODE, and the VAE runtime decodes the latents to
    audio. Each stage is also usable on its own, and a saved plan or semantic
    result can be replayed without regenerating it.

    Requests are serialized per session explicitly: one request at a time, and
    any stage raises :class:`Yue2SessionBusy` rather than interleaving two. State
    is request-local (the AR session resets its per-branch contexts at the start
    of every phase, and the NAR conditions freshly per chunk), so a cancelled or
    failed request cannot leak into the next one. ``close`` is idempotent.
    """

    def __init__(
        self,
        ar_session: Yue2ArSession,
        nar_runtime,
        vae_runtime,
        *,
        config: GenerationConfig | None = None,
        vae_core_frames: int = 1024,
        vae_halo_frames: int = 16,
        on_progress: Callable[[str, int, int], None] | None = None,
    ) -> None:
        self.ar = ar_session
        self.nar = nar_runtime
        self.vae = vae_runtime
        self.config = config or ar_session.config
        self.vae_core_frames = int(vae_core_frames)
        self.vae_halo_frames = int(vae_halo_frames)
        self.on_progress = on_progress
        self._busy = False
        self._closed = False

    # -- lifecycle ------------------------------------------------------
    def _enter(self) -> None:
        if self._closed:
            raise RuntimeError("session is closed")
        if self._busy:
            raise Yue2SessionBusy("this session is already running a request")
        self._busy = True

    def _leave(self) -> None:
        self._busy = False

    def reset(self) -> None:
        """Drop request-local AR state; the session stays usable."""

        self.ar.reset()

    def close(self) -> None:
        """Release every runtime. Idempotent."""

        if self._closed:
            return
        self._closed = True
        self._busy = False
        self.nar.close()
        self.vae.close()
        self.ar.close()

    def __enter__(self) -> "Yue2Session":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- stages ---------------------------------------------------------
    def _phase_sampling(self, phase: str, override: Sampling | dict | None) -> Sampling:
        """The sampling a phase runs with: the caller's override, else this config.

        ``effective_config`` reports this session's ``GenerationConfig``, so generation
        has to resolve against the same object; leaving the default to the AR session
        instead would let a session run with the settings the AR session was built with
        while recording the ones this session was built with.
        """

        return resolve_sampling(override, getattr(self.config, phase))

    def plan(self, request: SongRequest, **kwargs) -> SymbolicPlan:
        kwargs["sampling"] = self._phase_sampling("abc", kwargs.get("sampling"))
        with self._serialized():
            return self.ar.plan(request, **kwargs)

    def generate_semantic(self, plan: SymbolicPlan, **kwargs) -> SemanticResult:
        kwargs["sampling"] = self._phase_sampling("semantic", kwargs.get("sampling"))
        with self._serialized():
            return self.ar.generate_semantic(plan, **kwargs)

    @contextlib.contextmanager
    def _serialized(self):
        self._enter()
        try:
            yield
        finally:
            self._leave()

    def _check_cancelled(self, cancelled: Cancelled, message: str) -> None:
        if cancelled is not None and cancelled():
            raise InterruptedError(message)

    def synthesize(
        self,
        semantic: SemanticResult,
        *,
        steps: int | None = None,
        context: int | None = None,
        noise: np.ndarray | None = None,
        nar_cond_end: int = 0,
        cancelled: Cancelled = None,
        on_chunk: Callable[[int, int], None] | None = None,
    ) -> np.ndarray:
        """Solve the acoustic flow-matching ODE for one semantic result.

        The conditioning prefix is the plan's own positive prefix, so a
        divergence cannot be read as a prompt-assembly difference. Cancellation
        is checked between chunks and before each chunk's solve; the solver
        itself is one device call per chunk.
        """

        if not isinstance(semantic, SemanticResult):
            raise TypeError("Pass the SemanticResult returned by generate_semantic()")
        request = semantic.plan.request
        if tuple(token_prefixes(request, self.ar.encode, semantic.plan.abc_ids)) != tuple(
            semantic.plan.prefix
        ):
            raise ValueError("Semantic result does not retain the request's exact prefix")
        resolved_steps = int(self.config.ode_steps if steps is None else steps)
        resolved_context = int(self.config.context if context is None else context)
        chunks = song_chunks(
            semantic.plan.prefix,
            semantic.tokens,
            request.seed,
            context=resolved_context,
            noise=noise,
            nar_cond_end=nar_cond_end,
        )
        self._check_cancelled(cancelled, "Cancelled before acoustic prefill")
        latents = []
        for index, chunk in enumerate(chunks):
            self._check_cancelled(cancelled, "Cancelled before acoustic prefill")
            self.nar.condition(chunk)
            latents.append(self.nar.solve(resolved_steps))
            if on_chunk is not None:
                on_chunk(index + 1, len(chunks))
        return latents[0] if len(latents) == 1 else np.concatenate(latents, axis=0)

    def decode(
        self,
        latents: np.ndarray,
        *,
        tiled: bool = True,
        core_frames: int | None = None,
        halo_frames: int | None = None,
    ) -> np.ndarray:
        """Decode latents to unclipped FP32 audio ``[channels, samples]``.

        Accepts the solver's own ``[frames, latent_dim]`` layout or an already
        batched ``[1, latent_dim, frames]`` tensor, and normalizes to the
        decoder's channel-first layout.
        """

        values = np.asarray(latents, dtype=np.float32)
        if values.ndim == 2:
            values = values.T[None, ...]
        elif values.ndim == 3 and values.shape[1] != self.vae.weights.latent_dim:
            # A caller that batched the solver's layout gets the same treatment
            # rather than a silently mis-shaped decode.
            values = np.transpose(values, (0, 2, 1))
        if tiled:
            audio = self.vae.decode_tiled(
                values,
                core_frames=self.vae_core_frames if core_frames is None else int(core_frames),
                halo_frames=self.vae_halo_frames if halo_frames is None else int(halo_frames),
            )
        else:
            audio = self.vae.decode(values)
        return audio[0]

    # -- end to end -----------------------------------------------------
    def effective_config(
        self,
        request: SongRequest,
        abc_sampling: Sampling | dict | None = None,
        semantic_sampling: Sampling | dict | None = None,
        *,
        steps: int | None = None,
        context: int | None = None,
    ) -> dict:
        """The settings a request actually ran with, recorded in its result."""

        return {
            "abc": self._phase_sampling("abc", abc_sampling).to_dict(),
            "semantic": self._phase_sampling("semantic", semantic_sampling).to_dict(),
            "ode_steps": int(self.config.ode_steps if steps is None else steps),
            "ode_method": self.config.ode_method,
            "context": int(self.config.context if context is None else context),
            "vae_core_frames": self.vae_core_frames,
            "vae_halo_frames": self.vae_halo_frames,
            "vae_tiled": True,
            "guidance": request.guidance,
            "cot": request.cot,
        }

    def generate(
        self,
        request: SongRequest,
        *,
        abc_sampling: Sampling | dict | None = None,
        semantic_sampling: Sampling | dict | None = None,
        steps: int | None = None,
        context: int | None = None,
        tiled: bool = True,
        cancelled: Cancelled = None,
        on_token: TokenCallback = None,
    ) -> SongResult:
        """Plan, generate, synthesize and decode one request end to end."""

        with self._serialized():
            return self._generate_locked(
                request,
                abc_sampling=abc_sampling,
                semantic_sampling=semantic_sampling,
                steps=steps,
                context=context,
                tiled=tiled,
                cancelled=cancelled,
                on_token=on_token,
            )

    def _generate_locked(
        self,
        request: SongRequest,
        *,
        abc_sampling,
        semantic_sampling,
        steps,
        context,
        tiled,
        cancelled,
        on_token,
    ) -> SongResult:
        config = self.effective_config(
            request, abc_sampling, semantic_sampling, steps=steps, context=context
        )
        weights = {
            "model": dict(self.ar.runtime.weights.identity),
            "vae": dict(self.vae.weights.identity),
        }
        request_id = canonical_identity(
            {"request": request.to_dict(), "config": config, "weights": weights}
        )
        started = time.perf_counter()
        plan = self.ar.plan(
            request,
            sampling=self._phase_sampling("abc", abc_sampling),
            cancelled=cancelled,
            on_token=on_token,
        )
        semantic = self.ar.generate_semantic(
            plan,
            sampling=self._phase_sampling("semantic", semantic_sampling),
            cancelled=cancelled,
            on_token=on_token,
        )
        self._check_cancelled(cancelled, "Cancelled before acoustic prefill")
        nar_started = time.perf_counter()
        latents = self.synthesize(
            semantic, steps=steps, context=context, cancelled=cancelled
        )
        nar_seconds = time.perf_counter() - nar_started
        self._check_cancelled(cancelled, "Cancelled before audio decode")
        vae_started = time.perf_counter()
        audio = self.decode(latents, tiled=tiled)
        vae_seconds = time.perf_counter() - vae_started
        timing = {
            "abc": plan.timing,
            "semantic": semantic.timing,
            "nar_seconds": nar_seconds,
            "vae_seconds": vae_seconds,
            "e2e_seconds": time.perf_counter() - started,
        }
        return SongResult(
            audio=audio,
            sample_rate=int(self.vae.sample_rate),
            semantic=semantic,
            latents=np.asarray(latents, dtype=np.float32),
            config=config,
            weights=weights,
            timing=timing,
            request_id=request_id,
            latent_identity=tensor_identity(latents),
            audio_identity=tensor_identity(audio),
        )
