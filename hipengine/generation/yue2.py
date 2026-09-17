"""YuE2 request protocol, sampling order, flow schedule, and request-local RNG.

Torch-free by construction: this module is importable and testable without a GPU
and without ``torch``. It is the normative description of the checkpoint-native
interface in ``docs/MODEL-YUE2.md``:

* token domains and instruction strings,
* the assembled prefix / negative-prefix IDs for ``off`` / ``melody`` / ``full``,
* the acoustic chunk ranges,
* the reference sampling order (phase mask, minimum-length EOS mask, window
  repetition penalty, temperature, top-k, top-p),
* the FP64 midpoint schedule and the BF16 sigmoid time shift,
* a versioned request-local RNG.

The sampler's arithmetic follows the released implementation. Two deliberate,
documented differences exist and are gated by the production task gate rather
than by bit equality:

* the legacy ``off`` path's BF16 score arithmetic is reproduced on request via
  ``legacy_off=True``; the production path promotes scores to FP32 before the
  penalty/top-k/top-p stages,
* draws come from a versioned NumPy generator rather than from PyTorch's
  generator, so seeded HIP and seeded torch runs are not token-identical.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, replace
from numbers import Integral

import numpy as np

EOD = 151643
ABC_START, ABC_END = 151847, 151848
MUSIC_START, MUSIC_END = 151851, 151852
CODEC_OFFSET, CODEC_SIZE = 151853, 32768
LATENT_START, LATENT_END, LATENT_PAD = 184621, 184622, 184623
VOCAB_SIZE, CONTEXT = 184704, 24576
PROTOCOL_VERSION = "yue2-native-v1"

INSTRUCTIONS = {
    "off": "Generate music with codec tokens from the given conditions.",
    "melody": (
        "Generate a melody-only ABC transcription without chord symbols, then generate "
        "music with codec tokens from the given conditions."
    ),
    "full": (
        "Generate a chord-annotated ABC transcription, then generate music with codec "
        "tokens from the given conditions."
    ),
}

#: Latent-frame state dimension of the acoustic (flow matching) path.
LATENT_DIM = 64
#: Decoder stride product: PCM samples per latent frame.
DOWNSAMPLING_RATIO = 1920
SAMPLE_RATE = 48000
#: Natural decoder output length for ``frames`` latent frames.
NATURAL_LENGTH_TAIL = 64


def natural_output_length(frames: int) -> int:
    """Pinned decoder length: ``1920 * frames - 64``."""
    if isinstance(frames, bool) or not isinstance(frames, Integral) or frames < 1:
        raise ValueError("frames must be a positive integer")
    return DOWNSAMPLING_RATIO * int(frames) - NATURAL_LENGTH_TAIL


@dataclass(frozen=True)
class Sampling:
    """Per-phase sampling settings, validated exactly like the released preset."""

    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 100
    repetition_penalty: float = 1.2
    penalty_window: int = 50
    min_tokens: int = 200
    max_tokens: int = 9000

    def __post_init__(self):
        if any(
            type(value) is not int
            for value in (self.top_k, self.penalty_window, self.min_tokens, self.max_tokens)
        ):
            raise ValueError("Sampling counts must be integers")
        if not all(
            math.isfinite(value)
            for value in (self.temperature, self.top_p, self.repetition_penalty)
        ):
            raise ValueError("Sampling numbers must be finite")
        if not 0 <= self.temperature <= 5 or not 0 < self.top_p <= 1 or self.top_k < 1:
            raise ValueError("Invalid sampling temperature/top_p/top_k")
        if self.repetition_penalty <= 0 or not 1 <= self.penalty_window <= 100:
            raise ValueError("Invalid repetition penalty/window")
        if not 0 <= self.min_tokens <= self.max_tokens or self.max_tokens < 1:
            raise ValueError("Require 0 <= min_tokens <= max_tokens")

    def to_dict(self) -> dict:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "penalty_window": self.penalty_window,
            "min_tokens": self.min_tokens,
            "max_tokens": self.max_tokens,
        }


DEFAULT_ABC_SAMPLING = Sampling(0.7, 0.9, 30, 1.005, 100, 32, 4096)
DEFAULT_SEMANTIC_SAMPLING = Sampling()


@dataclass(frozen=True)
class GenerationConfig:
    abc: Sampling = field(default_factory=lambda: DEFAULT_ABC_SAMPLING)
    semantic: Sampling = field(default_factory=lambda: DEFAULT_SEMANTIC_SAMPLING)
    ode_steps: int = 32
    ode_method: str = "midpoint"
    context: int = CONTEXT
    version: str = PROTOCOL_VERSION

    def __post_init__(self):
        if (
            self.context != CONTEXT
            or self.ode_method != "midpoint"
            or type(self.ode_steps) is not int
            or self.ode_steps < 1
        ):
            raise ValueError("Require context=24576 and midpoint with positive integer steps")

    def to_dict(self) -> dict:
        return {
            "abc": self.abc.to_dict(),
            "semantic": self.semantic.to_dict(),
            "ode_steps": self.ode_steps,
            "ode_method": self.ode_method,
            "context": self.context,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data) -> "GenerationConfig":
        value = dict(data)
        defaults = cls()
        for key in ("abc", "semantic"):
            if key in value and isinstance(value[key], dict):
                base = getattr(defaults, key).to_dict()
                value[key] = Sampling(**{**base, **value[key]})
        return cls(**value)

    def sampling_for(self, phase: str) -> Sampling:
        if phase == "abc":
            return self.abc
        if phase == "semantic":
            return self.semantic
        raise ValueError("phase must be abc or semantic")


def resolve_sampling(value, default: Sampling) -> Sampling:
    """Reference ``resolve_sampling``: ``None`` -> default, dict -> overrides."""
    if value is None:
        return default
    if isinstance(value, Sampling):
        return value
    if isinstance(value, dict):
        return Sampling(**{**default.to_dict(), **value})
    raise TypeError("Sampling must be a Sampling object or a dictionary of overrides")


_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,179}")


@dataclass(frozen=True)
class SongRequest:
    style: str
    lyrics: str
    cot: str = "full"
    seed: int = 831001
    abc: str | None = None
    cfg_scale: float | None = None
    id: str = "song"

    def __post_init__(self):
        if self.cot not in INSTRUCTIONS:
            raise ValueError("cot must be off, melody or full")
        if not isinstance(self.style, str) or not isinstance(self.lyrics, str):
            raise TypeError("style and lyrics must be strings")
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ValueError("seed must be an integer in [0, 2**63)")
        if not _ID_PATTERN.fullmatch(self.id) or self.id in {".", ".."}:
            raise ValueError("id must be a filename-safe identifier")
        if self.abc is not None and (
            self.cot == "off" or not isinstance(self.abc, str) or not self.abc.strip()
        ):
            raise ValueError("External ABC requires nonempty text and cot=melody/full")
        if self.cfg_scale is not None and (
            not math.isfinite(self.cfg_scale) or not 0 <= self.cfg_scale <= 20
        ):
            raise ValueError("cfg_scale must be finite and in [0,20]")

    @property
    def guidance(self) -> float:
        """Semantic CFG scale: 1.01 for ``off``, 1.0 for symbolic modes."""
        if self.cfg_scale is not None:
            return float(self.cfg_scale)
        return 1.01 if self.cot == "off" else 1.0

    @property
    def needs_negative_branch(self) -> bool:
        return self.guidance != 1.0

    def text(self) -> str:
        return f"{INSTRUCTIONS[self.cot]}\n[Tags]\n{self.style}\n[Lyrics]\n{self.lyrics}\n"

    def to_dict(self) -> dict:
        return {
            "style": self.style,
            "lyrics": self.lyrics,
            "cot": self.cot,
            "seed": self.seed,
            "abc": self.abc,
            "cfg_scale": self.cfg_scale,
            "id": self.id,
        }

    @classmethod
    def from_dict(cls, data) -> "SongRequest":
        return cls(**dict(data))


def _validate_abc_ids(abc_ids) -> list[int]:
    values = list(abc_ids)
    if any(type(token) is not int or not 0 <= token < EOD for token in values):
        raise ValueError("ABC IDs must remain inside the ordinary text vocabulary")
    return values


def token_prefixes(request: SongRequest, encode, abc_ids=None) -> list[int]:
    """Assembled positive-branch prefix. ``encode`` maps text -> token IDs."""
    base = [EOD] + list(encode(request.text()))
    if request.cot == "off":
        return base + [ABC_START, ABC_END, MUSIC_START]
    if abc_ids is None:
        if request.abc is None:
            return base + [ABC_START]
        abc_ids = encode(request.abc)
    abc_ids = _validate_abc_ids(abc_ids)
    return base + [ABC_START] + abc_ids + [ABC_END, MUSIC_START]


def negative_prefix(request: SongRequest, encode, abc_ids=None) -> list[int]:
    """Negative CFG branch: instruction-only text, same exact ABC IDs."""
    base = [EOD] + list(encode(INSTRUCTIONS[request.cot]))
    if request.cot == "off":
        return base + [MUSIC_START]
    if abc_ids is None:
        raise ValueError("Symbolic CFG must retain the exact positive-branch ABC IDs")
    abc_ids = _validate_abc_ids(abc_ids)
    return base + [ABC_START] + abc_ids + [ABC_END, MUSIC_START]


def chunk_ranges(frames: int, prefix_tokens: int, context: int = CONTEXT):
    """Original upstream chunk boundaries for a semantic sequence."""
    size = min((context - prefix_tokens - 3) // 2, context)
    if frames < 1 or size < 1:
        raise ValueError("Empty codec or prefix leaves no acoustic context")
    return [(start, min(start + size, frames)) for start in range(0, frames, size)]


def phase_domain(phase: str) -> tuple[int, int]:
    """Inclusive-exclusive output support of a phase, excluding its end token."""
    if phase == "abc":
        return 0, EOD
    if phase == "semantic":
        return CODEC_OFFSET, CODEC_OFFSET + CODEC_SIZE
    raise ValueError("phase must be abc or semantic")


def phase_end_token(phase: str) -> int:
    if phase == "abc":
        return ABC_END
    if phase == "semantic":
        return MUSIC_END
    raise ValueError("phase must be abc or semantic")


def phase_window(phase: str) -> tuple[int, int]:
    """Projection window covering a phase's domain and its end token.

    :func:`distribution` masks every row outside :func:`phase_domain` and the phase's
    end token to ``-inf``, so those are the only rows a phase can ever select. The
    window is the smallest contiguous range holding them, which costs at most the few
    hundred masked rows between a domain's end and its end token: 32 769 rows for
    ``semantic`` (the codec range plus ``MUSIC_END``) and 151 849 for ``abc``.
    """

    low, high = phase_domain(phase)
    end = phase_end_token(phase)
    return min(low, end), max(high, end + 1)


# ---------------------------------------------------------------------------
# sampling
# ---------------------------------------------------------------------------


def window_penalty(scores: np.ndarray, recent_ids, penalty: float, *, bf16_round: bool = False) -> np.ndarray:
    """Frequency-based, sign-dependent repetition penalty over the recent window.

    ``bf16_round`` reproduces the legacy ``off`` path, whose scores and penalty
    arithmetic stay in BF16 (each product/quotient rounds before the select).
    """
    if penalty == 1.0 or len(recent_ids) == 0:
        return scores
    recent = np.asarray(recent_ids, dtype=np.int64).reshape(-1)
    counts = np.bincount(recent, minlength=scores.shape[-1]).astype(scores.dtype)
    if counts.shape[0] > scores.shape[-1]:
        raise ValueError("history contains an out-of-range token ID")
    if bf16_round:
        alpha = bf16(np.power(np.float32(penalty), counts.astype(np.float32)))
        negative = bf16(bf16(scores) * alpha)
        positive = bf16(bf16(scores) / alpha)
        return np.where(scores < 0, negative, positive)
    alpha = np.power(penalty, counts).astype(scores.dtype)
    return np.where(scores < 0, scores * alpha, scores / alpha)


def _distribution_legacy_off(logits, sampling: Sampling, history, step: int, phase: str) -> np.ndarray:
    """Historical ``off`` arithmetic: BF16 scores, BF16 softmax, three survivors."""
    scores = bf16(np.asarray(logits, dtype=np.float32).reshape(-1))
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    allowed = np.full_like(scores, -np.inf)
    allowed[CODEC_OFFSET : CODEC_OFFSET + CODEC_SIZE] = 0.0
    allowed[phase_end_token(phase)] = 0.0
    scores = bf16(scores + allowed)
    if step < sampling.min_tokens:
        scores[phase_end_token(phase)] = -np.inf
    scores = window_penalty(
        scores, list(history)[-sampling.penalty_window :], sampling.repetition_penalty, bf16_round=True
    )
    if sampling.temperature == 0:
        return scores
    if sampling.temperature != 1:
        scores = bf16(scores / bf16(np.asarray([sampling.temperature], dtype=np.float32))[0])
    top_k = min(sampling.top_k, scores.shape[-1])
    if top_k < scores.shape[-1]:
        threshold = np.partition(scores, -top_k)[-top_k]
        scores = np.where(scores < threshold, -np.inf, scores)
    if sampling.top_p < 1:
        order = np.argsort(-scores, kind="stable")
        values = scores[order]
        finite = np.isfinite(values)
        maximum = values[0] if finite[0] else -np.inf
        probabilities = bf16(np.exp(bf16(values - maximum)))
        total = bf16(probabilities.sum())
        probabilities = bf16(probabilities / total) if total > 0 else probabilities
        cumulative = bf16(np.cumsum(probabilities))
        removed = bf16(cumulative - probabilities) > bf16(np.asarray([sampling.top_p], dtype=np.float32))[0]
        removed[:3] = False
        removed |= ~finite
        values = np.where(removed, -np.inf, values)
        scores = np.empty_like(scores)
        scores[order] = values
    return scores


def distribution(
    logits: np.ndarray,
    sampling: Sampling,
    history,
    step: int,
    phase: str,
    legacy_off: bool = False,
) -> np.ndarray:
    """Reference sampling order. ``logits`` is a 1-D BF16-valued score row.

    ``legacy_off`` reproduces the historical ``off`` arithmetic (BF16 scores and
    at least three surviving top-p candidates). The production path uses FP32
    scores for every mode and retains at least one candidate.
    """
    if legacy_off:
        return _distribution_legacy_off(logits, sampling, history, step, phase)
    scores = np.asarray(logits, dtype=np.float32).reshape(-1).copy()
    # A windowed head row carries -inf outside its projection window, and the CFG
    # combination turns `-inf - -inf` into NaN. Everything non-finite is masked below
    # either way, but a NaN threshold silently disables the top-k stage (`NaN < t` is
    # False), so fold every non-finite value to -inf first. This is the identity on
    # every row the full projection produces.
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    allowed = np.full_like(scores, -np.inf)
    if phase == "abc":
        allowed[:EOD] = 0.0
    else:
        allowed[CODEC_OFFSET : CODEC_OFFSET + CODEC_SIZE] = 0.0
    allowed[phase_end_token(phase)] = 0.0
    scores = scores + allowed
    if step < sampling.min_tokens:
        scores[phase_end_token(phase)] = -np.inf
    scores = window_penalty(scores, list(history)[-sampling.penalty_window :], sampling.repetition_penalty)
    if sampling.temperature == 0:
        return scores
    if sampling.temperature != 1:
        scores = scores / np.float32(sampling.temperature)
    top_k = min(sampling.top_k, scores.shape[-1])
    if top_k < scores.shape[-1]:
        threshold = np.partition(scores, -top_k)[-top_k]
        scores = np.where(scores < threshold, -np.inf, scores)
    if sampling.top_p < 1:
        order = np.argsort(-scores, kind="stable")
        values = scores[order]
        finite = np.isfinite(values)
        shifted = np.where(finite, values, -np.inf)
        maximum = shifted[0] if finite[0] else -np.inf
        probabilities = np.exp(shifted - maximum)
        total = probabilities.sum()
        probabilities = probabilities / total if total > 0 else probabilities
        cumulative = np.cumsum(probabilities) - probabilities
        removed = cumulative > sampling.top_p
        removed[: 3 if legacy_off else 1] = False
        removed |= ~finite
        values = np.where(removed, -np.inf, values)
        scores = np.empty_like(scores)
        scores[order] = values
    return scores


def _round_bf16(values: np.ndarray) -> np.ndarray:
    """Round-to-nearest-even BF16 (matches a BF16 tensor's stored values)."""
    return bf16(values)


def bf16(values) -> np.ndarray:
    """Round-to-nearest-even BF16, keeping FP32 storage."""
    array = np.asarray(values, dtype=np.float32)
    bits = array.view(np.uint32)
    rounding = ((bits >> 16) & np.uint32(1)) + np.uint32(0x7FFF)
    rounded = ((bits + rounding) & np.uint32(0xFFFF0000)).view(np.float32)
    return np.where(np.isnan(array), array, rounded).astype(np.float32)


def combine_cfg(conditional: np.ndarray, unconditional: np.ndarray, scale: float) -> np.ndarray:
    """Reference BF16 CFG combination: ``neg + scale * (pos - neg)``.

    Rows that a phase cannot select may be ``-inf`` (a windowed head projection is, see
    :func:`hipengine.runtime.yue2_ar.Yue2ArRuntime.logits`), and ``-inf - -inf`` is NaN,
    so the difference is taken only where both sides are finite. That is the identity on
    every row the full projection produces.
    """

    conditional = _round_bf16(np.asarray(conditional, dtype=np.float32))
    unconditional = _round_bf16(np.asarray(unconditional, dtype=np.float32))
    if scale == 1.0:
        return conditional
    both = np.isfinite(conditional) & np.isfinite(unconditional)
    # Mask the operands before subtracting so the masked rows never evaluate
    # `-inf - -inf` (which would raise a warning as well as produce NaN).
    safe_conditional = np.where(both, conditional, 0.0)
    safe_unconditional = np.where(both, unconditional, 0.0)
    delta = np.where(both, _round_bf16(safe_conditional - safe_unconditional), -np.inf)
    scaled = np.where(both, _round_bf16(np.float32(scale) * delta), -np.inf)
    return np.where(both, _round_bf16(safe_unconditional + scaled), -np.inf)


# ---------------------------------------------------------------------------
# flow-matching schedule
# ---------------------------------------------------------------------------


def midpoint_schedule(steps: int) -> list[tuple[float, float, float, float]]:
    """FP64 host schedule: ``(t, raw_t, t_mid, raw_t_mid)`` per solver step."""
    if isinstance(steps, bool) or not isinstance(steps, Integral) or steps < 1:
        raise ValueError("steps must be a positive integer")
    step_size = np.float64(1.0) / np.float64(steps)
    schedule = []
    with np.errstate(divide="ignore", invalid="ignore"):
        for step in range(int(steps)):
            t = np.float64(1.0) - np.float64(step) * step_size
            t_mid = t - step_size / np.float64(2.0)
            schedule.append(
                (
                    float(t),
                    float(np.clip(np.log(t / (1 - t)), -20, 20)),
                    float(t_mid),
                    float(np.clip(np.log(t_mid / (1 - t_mid)), -20, 20)),
                )
            )
    return schedule


def time_shift(raw_t: float, shift: float = 1.0) -> float:
    """Sigmoid time shift applied in model (BF16) precision."""
    sigmoid = _round_bf16(np.asarray([1.0 / (1.0 + math.exp(-raw_t))], dtype=np.float32))[0]
    value = np.float32(shift) * sigmoid
    denominator = np.float32(1.0) + np.float32(shift - 1) * sigmoid
    return float(_round_bf16(np.asarray([value / denominator], dtype=np.float32))[0])


# ---------------------------------------------------------------------------
# request-local RNG
# ---------------------------------------------------------------------------


class YuE2Random:
    """Versioned, request-local, torch-free random source.

    ``ALGORITHM`` is recorded in result identities. Seeded identity is only
    promised for this algorithm: the reference implementation draws from
    PyTorch's generator, so cross-implementation seeded equality is not claimed.
    """

    ALGORITHM = "numpy-pcg64-v1"

    def __init__(self, seed: int):
        if isinstance(seed, bool) or not isinstance(seed, Integral):
            raise TypeError("seed must be an integer")
        self.seed = int(seed)
        self._generator = np.random.Generator(np.random.PCG64(self.seed))

    def standard_normal(self, shape, dtype=np.float32) -> np.ndarray:
        return np.asarray(self._generator.standard_normal(shape), dtype=dtype)

    def uniform(self) -> float:
        return float(self._generator.random())

    def sample_categorical(self, probabilities: np.ndarray) -> int:
        """Exact categorical draw by inverse CDF over the (already masked) scores."""
        cumulative = np.cumsum(np.asarray(probabilities, dtype=np.float64))
        total = cumulative[-1] if cumulative.size else 0.0
        if not math.isfinite(total) or total <= 0:
            raise ValueError("sampling distribution has no probability mass")
        draw = self.uniform() * total
        index = int(np.searchsorted(cumulative, draw, side="right"))
        return min(index, cumulative.size - 1)

    def state(self) -> dict:
        return {"algorithm": self.ALGORITHM, "seed": self.seed}

    def reset(self) -> None:
        self._generator = np.random.Generator(np.random.PCG64(self.seed))


def softmax_f32(scores: np.ndarray) -> np.ndarray:
    """FP32 softmax over a masked score row; ``-inf`` entries contribute zero."""
    values = np.asarray(scores, dtype=np.float32)
    finite = np.isfinite(values)
    if not finite.any():
        raise ValueError("score row has no finite entry")
    maximum = values[finite].max()
    exponential = np.where(finite, np.exp(values - maximum), 0.0)
    total = exponential.sum()
    return (exponential / total).astype(np.float32)


def replace_sampling(config: GenerationConfig, phase: str, **overrides) -> GenerationConfig:
    return replace(config, **{phase: replace(config.sampling_for(phase), **overrides)})
