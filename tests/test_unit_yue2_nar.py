"""M4 protocol gate: YuE2 NAR chunking, noise slicing, and solver schedule.

These are the CPU-side halves of the acoustic flow-matching stage. The device
half (NAR attention, velocity, midpoint loop) is gated separately against the
recorded oracle fixture in ``tests/fixtures/yue2/nar/``.

Expected values come from the pinned upstream implementation
(``yue2/nar.py``, ``yue2/modeling_yue2.py``) evaluated with torch on the same
host that produced the oracle fixtures; they are recorded here as literals so
the contract is checked without torch installed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hipengine.generation.yue2 import CODEC_OFFSET, MUSIC_END, YuE2Random
from hipengine.runtime.yue2_nar import (
    LATENT_DIM,
    from_bf16_bits,
    audio_position_rows,
    clamp_logit,
    shift_t_value,
    solver_schedule,
    song_chunks,
    timestep_sinusoid,
    to_bf16_bits,
    validate_chunk_context,
    visible_ar_length,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "yue2" / "nar"

#: ``shift_t_value`` outputs for the 32-step schedule, as BF16 storage bits.
SHIFTED_BITS_32 = [
    0x3F80, 0x3F78, 0x3F70, 0x3F68, 0x3F60, 0x3F58, 0x3F50, 0x3F48,
    0x3F40, 0x3F38, 0x3F30, 0x3F28, 0x3F20, 0x3F18, 0x3F10, 0x3F08,
    0x3F00, 0x3EF0, 0x3EE0, 0x3ED0, 0x3EC0, 0x3EB0, 0x3EA0, 0x3E90,
    0x3E7F, 0x3E60, 0x3E40, 0x3E20, 0x3E00, 0x3DC1, 0x3D81, 0x3CFF,
]


def test_solver_schedule_matches_the_reference_fp64_values():
    schedule = solver_schedule(32)
    assert len(schedule) == 32
    assert schedule[0] == (20.0, 4.143134726391533)
    assert schedule[1][0] == 3.4339872044851463
    assert schedule[1][1] == 3.0122615755052013
    assert schedule[2] == (2.70805020110221, 2.468099531471619)
    assert schedule[-1] == (-3.4339872044851463, -4.143134726391533)


def test_clamp_logit_is_fp64_and_clamped():
    assert clamp_logit(1.0) == 20.0
    assert clamp_logit(0.0) == -20.0
    assert clamp_logit(0.5) == 0.0
    assert clamp_logit(0.96875) == 3.4339872044851463
    with pytest.raises(ValueError):
        clamp_logit(1.5)
    with pytest.raises(ValueError):
        solver_schedule(0)


def test_shift_t_value_matches_recorded_bf16_bits():
    bits = [
        int(shift_t_value(raw)[0]) for raw, _ in solver_schedule(32)
    ]
    assert bits == SHIFTED_BITS_32


def test_shift_t_value_is_identity_when_shift_is_one():
    # timestep_shift is 1.0 for this checkpoint, so the shifted value is the
    # BF16 sigmoid of the BF16-rounded raw timestep.
    raw = 3.4339872044851463
    assert int(shift_t_value(raw)[0]) == 0x3F78


def test_shift_t_value_applies_a_non_unit_shift_in_bf16():
    # shift=2: 2*s/(1+s) for s = sigmoid(0) = 0.5 -> 0.6667 -> bf16 0x3F2B.
    bits = int(shift_t_value(0.0, shift=2.0)[0])
    assert bits == int(to_bf16_bits(np.asarray([2.0 * 0.5 / (1.0 + 1.0 * 0.5)], dtype=np.float32))[0])


def test_timestep_sinusoid_matches_the_reference_at_bf16_resolution():
    """The MLP sees BF16, so the sinusoid is gated at BF16 resolution.

    Measured against torch on this host: the FP32 sinusoid can differ by up to
    7 ulp (libm/SLEEF differences in ``exp``/``sin``/``cos``), and all 256
    entries still round to the same BF16 value, which is the only thing the
    ``TimestepEmbedder`` MLP consumes.
    """

    shifted = float(from_bf16_bits(np.asarray([SHIFTED_BITS_32[0]], dtype=np.uint16))[0])
    emb = timestep_sinusoid(np.float32(shifted))
    assert emb.shape == (256,)
    bits = to_bf16_bits(emb)
    assert int(bits[0]) == 0x3F0A
    assert int(bits[128]) == 0x3F57


def test_audio_position_rows_clamp_to_the_table():
    assert audio_position_rows([0, 1, 2], 8).tolist() == [0, 1, 2]
    assert audio_position_rows([7, 8, 9], 8).tolist() == [7, 7, 7]
    with pytest.raises(ValueError):
        audio_position_rows([-1], 8)


def test_song_chunks_reproduce_the_recorded_oracle_fixture():
    with np.load(FIXTURES / "chunk0.npz") as data:
        prefix = data["prefix"].tolist()
        codec = data["codec"].tolist()
        noise = data["noise"]
    chunks = song_chunks(prefix, codec, 1234, noise=noise)
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.ar_tokens == tuple(prefix + [value + CODEC_OFFSET for value in codec] + [MUSIC_END])
    assert chunk.nar_length == len(codec) + 2
    assert chunk.ar_length == len(prefix) + len(codec) + 1
    assert np.array_equal(chunk.noise, noise)
    assert chunk.noise.dtype == np.float32
    assert visible_ar_length(chunk) == chunk.ar_length
    validate_chunk_context(chunk, 24576)


def test_song_chunks_draws_once_for_the_whole_song():
    # A chunk's noise is a view of the song-level draw, so the second chunk of a
    # small-context song equals the corresponding slice of the full draw.
    frames = 25
    codec = list(range(frames))
    chunks = song_chunks([1, 2, 3], codec, 7, context=32)
    assert [chunk.frame_range for chunk in chunks] == [(0, 13), (13, 25)]
    rng = YuE2Random(7)
    full = rng.standard_normal((frames, LATENT_DIM), dtype=np.float32)
    for chunk in chunks:
        start, stop = chunk.frame_range
        assert np.array_equal(chunk.noise, full[start:stop])
    # And the whole-song draw is deterministic across calls.
    again = song_chunks([1, 2, 3], codec, 7, context=32)
    assert all(np.array_equal(a.noise, b.noise) for a, b in zip(chunks, again))


def test_song_chunks_chunk_boundaries_match_the_reference_rule():
    # size = min((context - prefix - 3) // 2, context)
    chunks = song_chunks([1] * 8, list(range(32)), 1, context=32)
    assert [chunk.frame_range for chunk in chunks] == [(0, 10), (10, 20), (20, 30), (30, 32)]
    assert [len(chunk.ar_tokens) for chunk in chunks] == [8 + 10 + 1, 8 + 10 + 1, 8 + 10 + 1, 8 + 2 + 1]
    # A single chunk when the context is large enough.
    single = song_chunks([1] * 8, list(range(32)), 1)
    assert [chunk.frame_range for chunk in single] == [(0, 32)]


def test_song_chunks_rejects_bad_input():
    with pytest.raises(ValueError):
        song_chunks([], [1, 2], 1)
    with pytest.raises(ValueError):
        song_chunks([1], [], 1)
    with pytest.raises(ValueError):
        song_chunks([1], [0], 1, noise=np.zeros((2, LATENT_DIM), dtype=np.float32))
    with pytest.raises(ValueError):
        song_chunks([1], [0], 1, noise=np.full((1, LATENT_DIM), np.nan, dtype=np.float32))
    with pytest.raises(ValueError):
        song_chunks([1], [CODEC_OFFSET + 32768], 1)
    with pytest.raises(ValueError):
        song_chunks([1], [0], True)
    with pytest.raises(ValueError):
        song_chunks([1], [0], 1, nar_cond_end=-1)
    with pytest.raises(ValueError):
        song_chunks([1], list(range(32)), 1, context=4)


def test_nar_cond_end_restricts_ar_visibility():
    chunks = song_chunks([1] * 512, list(range(4)), 1, nar_cond_end=256)
    chunk = chunks[0]
    assert chunk.nar_cond_end == 256
    assert visible_ar_length(chunk) == 256
    # A value past the end is a no-op; 0 means "see everything". The AR prefix
    # includes the codec tokens and the music-end token.
    full = song_chunks([1] * 8, [0], 1)[0]
    assert full.ar_length == 10
    assert visible_ar_length(song_chunks([1] * 8, [0], 1, nar_cond_end=999)[0]) == 10
    assert visible_ar_length(full) == 10


def test_validate_chunk_context_matches_the_reference_bound():
    chunk = song_chunks([1] * 8, list(range(4)), 1)[0]
    bound = chunk.ar_length + chunk.nar_length
    validate_chunk_context(chunk, bound)
    with pytest.raises(ValueError):
        validate_chunk_context(chunk, bound - 1)


def test_nar_fixture_manifest_is_consistent():
    manifest = json.loads((FIXTURES / "manifest.json").read_text())
    with np.load(FIXTURES / "chunk0.npz") as data:
        assert manifest["prefix_length"] == data["prefix"].shape[0]
        assert manifest["frames"] == data["codec"].shape[0] == data["noise"].shape[0]
        assert data["noise"].shape[1] == LATENT_DIM
        assert data["velocities"].shape[0] == data["states"].shape[0]
        assert abs(float(np.linalg.norm(data["latents"])) - manifest["latent_norm"]) < 1e-3
        assert abs(float(np.linalg.norm(data["velocities"][0])) - manifest["velocity_norm_step0"]) < 1e-3
