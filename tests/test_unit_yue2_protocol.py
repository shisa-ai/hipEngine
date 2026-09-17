"""M1 protocol gates: token domains, prefixes, defaults, schedule, RNG.

Fixtures under ``tests/fixtures/yue2`` are produced by ``scripts/yue2_oracle.py``
from the pinned release; this module never imports torch.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hipengine.generation.yue2 import (
    ABC_END,
    INSTRUCTIONS,
    ABC_START,
    CODEC_OFFSET,
    CODEC_SIZE,
    CONTEXT,
    EOD,
    LATENT_END,
    LATENT_PAD,
    LATENT_START,
    MUSIC_END,
    MUSIC_START,
    VOCAB_SIZE,
    GenerationConfig,
    Sampling,
    SongRequest,
    YuE2Random,
    chunk_ranges,
    combine_cfg,
    distribution,
    midpoint_schedule,
    natural_output_length,
    negative_prefix,
    phase_domain,
    phase_end_token,
    phase_window,
    resolve_sampling,
    softmax_f32,
    time_shift,
    token_prefixes,
    window_penalty,
)
from hipengine.kernels.cpu_reference.yue2 import bf16, bf16_bits_to_f32

FIXTURES = Path(__file__).parent / "fixtures/yue2"


class _FakeTokenizer:
    """Deterministic stand-in: one ID per character, offset into the text range."""

    def __init__(self, offset: int = 1000):
        self.offset = offset
        self.calls: list[str] = []

    def encode(self, text: str) -> list[int]:
        self.calls.append(text)
        return [self.offset + index for index in range(len(text))]


def test_token_domain_constants():
    assert (EOD, ABC_START, ABC_END) == (151643, 151847, 151848)
    assert (MUSIC_START, MUSIC_END) == (151851, 151852)
    assert (CODEC_OFFSET, CODEC_SIZE) == (151853, 32768)
    assert (LATENT_START, LATENT_END, LATENT_PAD) == (184621, 184622, 184623)
    assert (VOCAB_SIZE, CONTEXT) == (184704, 24576)
    assert CODEC_OFFSET + CODEC_SIZE == LATENT_START


def test_sampling_defaults_match_release():
    config = GenerationConfig()
    assert config.abc == Sampling(0.7, 0.9, 30, 1.005, 100, 32, 4096)
    assert config.semantic == Sampling(1.0, 0.95, 100, 1.2, 50, 200, 9000)
    assert config.ode_steps == 32 and config.ode_method == "midpoint"
    assert GenerationConfig.from_dict(config.to_dict()) == config
    # Partial override keeps the other fields.
    merged = GenerationConfig.from_dict({"semantic": {"top_p": 0.5}})
    assert merged.semantic.top_p == 0.5 and merged.semantic.top_k == 100


@pytest.mark.parametrize(
    "kwargs",
    [
        {"top_k": 0},
        {"penalty_window": 0},
        {"penalty_window": 101},
        {"min_tokens": 10, "max_tokens": 5},
        {"temperature": -0.1},
        {"temperature": 5.1},
        {"top_p": 0.0},
        {"top_p": 1.5},
        {"repetition_penalty": 0.0},
        {"top_k": 1.5},
        {"temperature": float("nan")},
    ],
)
def test_sampling_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        Sampling(**kwargs)


def test_sampling_window_bounds_are_inclusive():
    assert Sampling(penalty_window=1).penalty_window == 1
    assert Sampling(penalty_window=100).penalty_window == 100


@pytest.mark.parametrize(
    "kwargs",
    [
        {"context": 4096},
        {"ode_steps": 0},
        {"ode_steps": 32.0},
        {"ode_method": "euler"},
    ],
)
def test_generation_config_rejects_non_pinned_values(kwargs):
    with pytest.raises(ValueError):
        GenerationConfig(**kwargs)


def test_request_guidance_and_text():
    off = SongRequest(style="pop", lyrics="la", cot="off")
    assert off.guidance == 1.01
    assert SongRequest(style="pop", lyrics="la", cot="full").guidance == 1.0
    assert SongRequest(style="pop", lyrics="la", cot="full", cfg_scale=1.5).guidance == 1.5
    assert off.text().startswith("Generate music with codec tokens")
    assert off.text().endswith("\n")
    assert "[Tags]\npop\n[Lyrics]\nla\n" in off.text()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"cot": "none"},
        {"seed": -1},
        {"seed": 2**63},
        {"seed": True},
        {"id": "../escape"},
        {"id": ""},
        {"cfg_scale": float("inf")},
        {"cfg_scale": 21.0},
        {"abc": "X:1", "cot": "off"},
        {"abc": "   ", "cot": "full"},
    ],
)
def test_request_rejects_invalid_values(kwargs):
    base = {"style": "pop", "lyrics": "la"}
    base.update(kwargs)
    with pytest.raises((ValueError, TypeError)):
        SongRequest(**base)


def test_token_prefixes_off_and_symbolic():
    tokenizer = _FakeTokenizer()
    off = SongRequest(style="pop", lyrics="la", cot="off")
    prefix = token_prefixes(off, tokenizer.encode)
    assert prefix[0] == EOD
    assert prefix[-3:] == [ABC_START, ABC_END, MUSIC_START]
    assert prefix[1:-3] == tokenizer.encode(off.text())

    full = SongRequest(style="pop", lyrics="la", cot="full")
    pending = token_prefixes(full, tokenizer.encode)
    assert pending[-1] == ABC_START
    assert pending[1:-1] == tokenizer.encode(full.text())
    abc_ids = [11, 12, 13]
    planned = token_prefixes(full, tokenizer.encode, abc_ids)
    assert planned == pending + abc_ids + [ABC_END, MUSIC_START]

    # An external score is encoded with the same callback and keeps its exact IDs.
    external = SongRequest(style="pop", lyrics="la", cot="melody", abc="X:1\n")
    encoded = token_prefixes(external, lambda text: [5, 6])
    assert encoded[-5:] == [ABC_START, 5, 6, ABC_END, MUSIC_START]


def test_negative_prefix_shape():
    tokenizer = _FakeTokenizer()
    off = SongRequest(style="pop", lyrics="la", cot="off")
    negative = negative_prefix(off, tokenizer.encode)
    # The negative branch carries the instruction only: no style, no lyrics.
    assert negative == [EOD] + tokenizer.encode(INSTRUCTIONS["off"]) + [MUSIC_START]
    assert negative[-1] == MUSIC_START and ABC_START not in negative
    # The negative branch never contains style/lyrics for symbolic modes.
    full = SongRequest(style="pop", lyrics="la", cot="full")
    with pytest.raises(ValueError):
        negative_prefix(full, tokenizer.encode)
    negative = negative_prefix(full, tokenizer.encode, [11, 12])
    assert negative[-5:] == [ABC_START, 11, 12, ABC_END, MUSIC_START]
    assert len(negative) < len(token_prefixes(full, tokenizer.encode, [11, 12]))


def test_token_prefixes_reject_out_of_range_abc_ids():
    full = SongRequest(style="pop", lyrics="la", cot="full")
    for bad in ([EOD], [-1], [1.0]):
        with pytest.raises(ValueError):
            token_prefixes(full, lambda text: [], bad)


def test_chunk_ranges_preserve_upstream_boundaries():
    assert chunk_ranges(10, 100) == [(0, 10)]
    size = (CONTEXT - 100 - 3) // 2
    ranges = chunk_ranges(size * 2 + 5, 100)
    assert ranges == [(0, size), (size, size * 2), (size * 2, size * 2 + 5)]
    with pytest.raises(ValueError):
        chunk_ranges(0, 100)
    with pytest.raises(ValueError):
        chunk_ranges(10, CONTEXT)


def test_chunk_ranges_match_reference_formula():
    # capacity = min((24576 - prefix - 3) // 2, 24576)
    capacity = min((CONTEXT - 512 - 3) // 2, CONTEXT)
    ranges = chunk_ranges(capacity + 1, 512)
    assert ranges == [(0, capacity), (capacity, capacity + 1)]
    assert chunk_ranges(capacity, 512) == [(0, capacity)]


def test_natural_output_length():
    assert natural_output_length(1) == 1920 - 64
    assert natural_output_length(64) == 1920 * 64 - 64
    with pytest.raises(ValueError):
        natural_output_length(0)


def test_resolve_sampling():
    default = GenerationConfig().semantic
    assert resolve_sampling(None, default) is default
    assert resolve_sampling({"top_k": 5}, default).top_k == 5
    with pytest.raises(TypeError):
        resolve_sampling("0.9", default)


# ---------------------------------------------------------------------------
# fixtures: schedule, time shift, sampling arithmetic
# ---------------------------------------------------------------------------


def _load(name: str):
    path = FIXTURES / name
    if not path.is_file():
        pytest.skip(f"missing oracle fixture {path}")
    return np.load(path)


def test_midpoint_schedule_matches_reference_fixture():
    data = _load("operators/midpoint_schedule.npz")
    expected = data["schedule"]
    got = midpoint_schedule(expected.shape[0])
    assert len(got) == expected.shape[0]
    for row, (t, raw, t_mid, raw_mid) in zip(expected, got):
        assert t == row[1] and t_mid == row[3]
        assert raw == pytest.approx(row[2], abs=0, rel=0)
        assert raw_mid == pytest.approx(row[4], abs=0, rel=0)
    # FP64 host schedule: t=1 -> raw 20, and the schedule is monotone.
    assert got[0][1] == 20.0
    assert all(got[index][1] < got[index - 1][1] for index in range(1, len(got)))


def test_time_shift_matches_reference_fixture():
    data = _load("operators/timestep_shift.npz")
    for raw, shifted in zip(data["raw"], data["shifted"]):
        assert time_shift(float(raw)) == pytest.approx(float(shifted), abs=1e-7)


def test_sampling_distribution_matches_reference_fixture():
    manifest = FIXTURES / "sampling/manifest.json"
    if not manifest.is_file():
        pytest.skip("missing sampling fixtures")
    cases = json.loads(manifest.read_text())
    checked = 0
    for case in cases:
        if case.get("kind") == "window_penalty":
            continue
        data = np.load(FIXTURES / "sampling" / case["file"])
        logits = bf16_bits_to_f32(data["logits"]).reshape(-1)
        settings = case["sampling"]
        sampling = Sampling(
            settings["temperature"],
            settings["top_p"],
            settings["top_k"],
            settings["repetition_penalty"],
            settings["penalty_window"],
            settings["min_tokens"],
            settings["max_tokens"],
        )
        got = distribution(
            logits,
            sampling,
            data["history"].tolist(),
            case["step"],
            case["phase"],
            legacy_off=case["legacy_off"],
        )
        expected = data["scores"].reshape(-1)
        candidates = distribution(
            logits,
            Sampling(
                sampling.temperature,
                1.0,
                sampling.top_k,
                sampling.repetition_penalty,
                sampling.penalty_window,
                sampling.min_tokens,
                sampling.max_tokens,
            ),
            data["history"].tolist(),
            case["step"],
            case["phase"],
            legacy_off=case["legacy_off"],
        )
        _assert_tie_aware_equal(
            expected, got, candidates, f"{case['name']} step={case['step']}"
        )
        checked += 1
    assert checked >= 24


def test_window_penalty_matches_reference_fixture():
    manifest = json.loads((FIXTURES / "sampling/manifest.json").read_text())
    checked = 0
    for case in manifest:
        if case.get("kind") != "window_penalty":
            continue
        data = np.load(FIXTURES / "sampling" / case["file"])
        logits = bf16_bits_to_f32(data["logits"]).reshape(-1).astype(np.float32)
        history = data["history"].tolist()[-case["penalty_window"] :]
        got = window_penalty(logits, history, case["repetition_penalty"])
        np.testing.assert_allclose(got, data["scores"].reshape(-1), rtol=0, atol=0)
        checked += 1
    assert checked >= 4


def _assert_tie_aware_equal(expected, got, candidates, label):
    """Exact equality up to the arbitrary order of equally scored cutoff ties.

    The reference sorts with ``torch.sort``, which is not stable, so BF16 score
    ties at the top-p cutoff boundary survive in an arbitrary order. The
    *distribution* is what the sampler consumes, so the gate requires the
    surviving probability multiset to match exactly and any index that differs
    to be part of a duplicated score value inside the pre-top-p candidate set.
    """
    reference_finite = np.isfinite(expected)
    got_finite = np.isfinite(got)
    np.testing.assert_array_equal(
        np.sort(expected[reference_finite]), np.sort(got[got_finite]), err_msg=label
    )
    differing = reference_finite != got_finite
    if differing.any():
        candidate_values = np.asarray(candidates)[np.isfinite(candidates)]
        values, counts = np.unique(candidate_values, return_counts=True)
        duplicated = set(values[counts > 1].tolist())
        offenders = [
            float(value)
            for value in np.asarray(candidates)[differing]
            if value not in duplicated
        ]
        assert not offenders, f"{label}: non-tied survivor differs: {offenders[:4]}"
    # The sampled distribution is the softmax of the surviving scores; equal
    # score multisets make it equal up to FP32 exp/sum rounding.
    reference_mass = np.sort(softmax_f32(np.where(reference_finite, expected, -np.inf))[reference_finite])
    got_mass = np.sort(softmax_f32(np.where(got_finite, got, -np.inf))[got_finite])
    np.testing.assert_allclose(reference_mass, got_mass, rtol=1e-6, atol=1e-12, err_msg=label)


def test_window_penalty_is_sign_dependent():
    scores = np.asarray([2.0, -2.0, 0.5], dtype=np.float32)
    penalized = window_penalty(scores, [0, 0, 1], 1.5)
    assert penalized[0] == pytest.approx(2.0 / 1.5**2)
    assert penalized[1] == pytest.approx(-2.0 * 1.5)
    assert penalized[2] == pytest.approx(0.5)


def test_distribution_masks_phase_domain_and_min_length():
    # top_k = vocab disables the top-k stage so only the domain mask is visible.
    wide = Sampling(top_k=VOCAB_SIZE, top_p=1.0, min_tokens=5)
    logits = np.zeros(VOCAB_SIZE, dtype=np.float32)
    logits[MUSIC_END] = 100.0
    masked = distribution(logits, wide, [], 0, "semantic")
    assert masked[MUSIC_END] == -np.inf
    assert np.isfinite(masked[CODEC_OFFSET]) and np.isfinite(masked[CODEC_OFFSET + CODEC_SIZE - 1])
    assert masked[0] == -np.inf and masked[EOD] == -np.inf
    assert masked[CODEC_OFFSET + CODEC_SIZE] == -np.inf
    allowed = distribution(logits, wide, [], 5, "semantic")
    assert np.isfinite(allowed[MUSIC_END])

    abc = distribution(np.zeros(VOCAB_SIZE, dtype=np.float32), wide, [], 5, "abc")
    assert np.isfinite(abc[0]) and np.isfinite(abc[EOD - 1]) and np.isfinite(abc[ABC_END])
    assert abc[EOD] == -np.inf and abc[ABC_END + 1] == -np.inf
    assert abc[CODEC_OFFSET] == -np.inf


def test_distribution_temperature_zero_is_argmax_input():
    logits = np.full(VOCAB_SIZE, -5.0, dtype=np.float32)
    logits[CODEC_OFFSET + 7] = 3.0
    scores = distribution(logits, Sampling(temperature=0.0), [], 0, "semantic")
    assert int(np.argmax(scores)) == CODEC_OFFSET + 7


def test_distribution_top_k_and_top_p_shrink_support():
    rng = np.random.default_rng(0)
    logits = rng.standard_normal(VOCAB_SIZE).astype(np.float32)
    scores = distribution(logits, Sampling(top_k=10, top_p=1.0), [], 0, "semantic")
    assert np.isfinite(scores).sum() == 10
    scores = distribution(logits, Sampling(top_k=100, top_p=0.5), [], 0, "semantic")
    finite = np.isfinite(scores)
    assert 1 <= finite.sum() < 100
    probabilities = softmax_f32(scores)
    assert probabilities[finite].sum() == pytest.approx(1.0, abs=1e-5)


def test_legacy_off_keeps_three_candidates():
    rng = np.random.default_rng(1)
    logits = rng.standard_normal(VOCAB_SIZE).astype(np.float32)
    scores = distribution(logits, Sampling(top_p=1e-9), [], 0, "semantic", legacy_off=True)
    assert np.isfinite(scores).sum() >= 3


def test_combine_cfg_matches_bf16_reference_order():
    rng = np.random.default_rng(2)
    conditional = rng.standard_normal(4096).astype(np.float32)
    unconditional = rng.standard_normal(4096).astype(np.float32)
    got = combine_cfg(conditional, unconditional, 1.01)
    expected = bf16(bf16(unconditional) + bf16(np.float32(1.01) * bf16(bf16(conditional) - bf16(unconditional))))
    np.testing.assert_array_equal(got, expected)
    np.testing.assert_array_equal(combine_cfg(conditional, unconditional, 1.0), bf16(conditional))


def test_softmax_rejects_all_masked_rows():
    with pytest.raises(ValueError):
        softmax_f32(np.full(4, -np.inf, dtype=np.float32))


# ---------------------------------------------------------------------------
# RNG
# ---------------------------------------------------------------------------


def test_rng_reproducible_and_resettable():
    first = YuE2Random(1234)
    noise = first.standard_normal((8, 64))
    assert noise.shape == (8, 64) and noise.dtype == np.float32
    second = YuE2Random(1234)
    np.testing.assert_array_equal(noise, second.standard_normal((8, 64)))
    first.reset()
    np.testing.assert_array_equal(noise, first.standard_normal((8, 64)))
    assert YuE2Random(1234).state() == {"algorithm": "numpy-pcg64-v1", "seed": 1234}


def test_rng_categorical_is_exact_and_bounded():
    rng = YuE2Random(7)
    probabilities = np.zeros(1000, dtype=np.float32)
    probabilities[[3, 17, 999]] = [0.5, 0.25, 0.25]
    counts = np.zeros(1000)
    for _ in range(2000):
        counts[rng.sample_categorical(probabilities)] += 1
    assert counts[3] + counts[17] + counts[999] == 2000
    assert counts[3] / 2000 == pytest.approx(0.5, abs=0.04)
    assert counts[17] / 2000 == pytest.approx(0.25, abs=0.04)
    with pytest.raises(ValueError):
        rng.sample_categorical(np.zeros(4, dtype=np.float32))


def test_rng_standard_normal_distribution_is_sane():
    rng = YuE2Random(99)
    sample = rng.standard_normal(20000)
    assert abs(float(sample.mean())) < 0.05
    assert abs(float(sample.std()) - 1.0) < 0.05
    assert np.isfinite(sample).all()


def test_phase_window_contains_exactly_the_selectable_rows():
    """The head's projection window must cover every row a phase can select.

    `distribution` is the only consumer of the head's row in the product path, and it
    masks everything outside `phase_domain(phase)` and the phase's end token to -inf
    (see `test_distribution_masks_phase_domain_and_min_length`). A windowed projection
    is therefore equivalent exactly when the window holds both sets, which is what the
    runtime's `logits(domain=...)` relies on.
    """

    wide = Sampling(top_k=VOCAB_SIZE, top_p=1.0, min_tokens=0)
    for phase in ("semantic", "abc"):
        low, high = phase_window(phase)
        assert (low, high) == (min(phase_domain(phase)[0], phase_end_token(phase)),
                               max(phase_domain(phase)[1], phase_end_token(phase) + 1))
        # After min_tokens the end token is allowed too, so nothing outside the window
        # may survive the mask.
        logits = np.zeros(VOCAB_SIZE, dtype=np.float32)
        scores = distribution(logits, wide, [], wide.min_tokens, phase)
        finite = np.flatnonzero(np.isfinite(scores))
        assert finite.size > 0
        assert finite.min() >= low and finite.max() < high, phase
        # Inside the window, only the masked filler between a domain's end and its end
        # token is ever non-finite, and only for `abc`.
        inside = np.flatnonzero(~np.isfinite(scores[low:high])) + low
        if phase == "abc":
            assert np.array_equal(inside, np.arange(phase_domain("abc")[1], ABC_END))
        else:
            assert inside.size == 0
        # The semantic window is the codec range plus its end token, which is what makes
        # it worth projecting separately.
        if phase == "semantic":
            assert (low, high) == (MUSIC_END, CODEC_OFFSET + CODEC_SIZE)
            assert (high - low) < VOCAB_SIZE / 5


def test_phase_window_rejects_unknown_phases():
    with pytest.raises(ValueError):
        phase_window("chorus")


def test_combine_cfg_keeps_masked_rows_masked():
    """A windowed row is -inf where a phase cannot select, so CFG must not make NaN.

    `-inf - -inf` is NaN, and a NaN score disables the sampler's top-k stage
    (`NaN < threshold` is False), so the combination has to leave non-finite rows
    non-finite. On rows where both sides are finite the arithmetic is unchanged.
    """

    conditional = np.asarray([1.5, -2.25, 3.0], dtype=np.float32)
    unconditional = np.asarray([0.5, -1.0, -np.inf], dtype=np.float32)
    combined = combine_cfg(conditional, unconditional, 1.5)
    assert combined[2] == -np.inf and not np.isnan(combined[2])
    # The finite rows still follow `neg + scale * (pos - neg)` in BF16.
    for index in (0, 1):
        expected = bf16(
            bf16(unconditional[index])
            + bf16(np.float32(1.5) * bf16(bf16(conditional[index]) - bf16(unconditional[index])))
        )
        assert combined[index] == expected
    # Scale 1.0 returns the conditional row untouched, as the reference does.
    assert np.array_equal(combine_cfg(conditional, unconditional, 1.0), bf16(conditional))


def test_distribution_rejects_non_finite_inside_the_allowed_domain():
    """A NaN or +inf where the phase can select is a failure, not something to mask.

    A windowed head row is -inf outside its window on purpose, and folding that to -inf
    is right. Inside the domain the same treatment would hide a broken head or a broken
    CFG combination, so it raises instead.
    """

    wide = Sampling(top_k=VOCAB_SIZE, top_p=1.0, min_tokens=0)
    clean = np.zeros(VOCAB_SIZE, dtype=np.float32)
    # Intentional masking outside the domain, including NaN, is fine. For `semantic` the
    # allowed set is the codec range plus MUSIC_END, which sits just below CODEC_OFFSET,
    # so the masked prefix stops there.
    outside = clean.copy()
    outside[:MUSIC_END] = np.nan
    assert np.isfinite(distribution(outside, wide, [], 0, "semantic")[CODEC_OFFSET])
    for bad_value in (np.nan, np.inf, -np.inf):
        broken = clean.copy()
        broken[CODEC_OFFSET + 3] = bad_value
        with pytest.raises(FloatingPointError):
            distribution(broken, wide, [], 0, "semantic")
        with pytest.raises(FloatingPointError):
            distribution(broken, wide, [], 0, "semantic", legacy_off=True)
    # The end token is selectable too, so it is covered by the same guard.
    broken = clean.copy()
    broken[MUSIC_END] = np.nan
    with pytest.raises(FloatingPointError):
        distribution(broken, wide, [], 5, "semantic")
    # And abc's own holes are still just masked, not errors.
    holes = clean.copy()
    holes[EOD + 5] = np.nan
    assert np.isfinite(distribution(holes, wide, [], 5, "abc")[ABC_END])


def test_random_state_digest_tracks_the_stream_position():
    """`state()` records the identity; the digest is what proves identical consumption."""

    first = YuE2Random(1234)
    second = YuE2Random(1234)
    assert first.state() == second.state() == {"algorithm": "numpy-pcg64-v1", "seed": 1234}
    assert first.state_digest() == second.state_digest()
    first.uniform()
    assert first.state_digest() != second.state_digest()
    assert first.state() == second.state(), "identity is unchanged by consumption"
    second.uniform()
    assert first.state_digest() == second.state_digest()
    # Equal digests mean every later draw agrees, which is the point of comparing them.
    assert [first.uniform() for _ in range(4)] == [second.uniform() for _ in range(4)]
