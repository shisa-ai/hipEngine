"""WAV transport conversion: clip, scale, interleave, wrap.

The song pipeline returns unclipped FP32 ``[channels, samples]``. This is the
only place that becomes a file body, so the tests pin the header fields, the
interleaving order, and the clipping contract.
"""

from __future__ import annotations

import io
import wave
from typing import Any

import numpy as np
import pytest

from hipengine.util.wav import INT16_FULL_SCALE, encode_wav_bytes


def _read(payload: bytes) -> tuple[Any, np.ndarray]:
    with wave.open(io.BytesIO(payload), "rb") as handle:
        params = handle.getparams()
        frames = handle.readframes(handle.getnframes())
    width = params.sampwidth
    dtype = {2: "<i2", 4: "<f4"}[width]
    return params, np.frombuffer(frames, dtype=dtype)


def test_the_header_describes_the_stream() -> None:
    audio = np.zeros((2, 480), dtype=np.float32)
    params, samples = _read(encode_wav_bytes(audio, 48000))

    assert (params.nchannels, params.sampwidth, params.framerate) == (2, 2, 48000)
    assert params.nframes == 480
    assert samples.size == 960


def test_channels_are_interleaved_by_frame() -> None:
    audio = np.array([[1.0, 0.5, -1.0], [-1.0, 0.0, 0.25]], dtype=np.float32)
    _, samples = _read(encode_wav_bytes(audio, 24000))

    assert samples.tolist() == [
        INT16_FULL_SCALE,
        -INT16_FULL_SCALE,
        round(0.5 * INT16_FULL_SCALE),
        0,
        -INT16_FULL_SCALE,
        round(0.25 * INT16_FULL_SCALE),
    ]


def test_out_of_range_samples_clip_rather_than_wrap() -> None:
    audio = np.array([[4.0, -4.0, 1.0, -1.0]], dtype=np.float32)
    _, samples = _read(encode_wav_bytes(audio, 48000))

    assert samples.tolist() == [INT16_FULL_SCALE, -INT16_FULL_SCALE, INT16_FULL_SCALE, -INT16_FULL_SCALE]


def test_a_mono_signal_needs_no_reshape() -> None:
    params, samples = _read(encode_wav_bytes(np.zeros(16, dtype=np.float32), 16000))
    assert params.nchannels == 1
    assert samples.size == 16


def test_the_float_form_keeps_the_original_values() -> None:
    audio = np.array([[0.5, -0.25]], dtype=np.float32)
    params, samples = _read(encode_wav_bytes(audio, 48000, bit_depth=32))

    assert params.sampwidth == 4
    assert samples.tolist() == [0.5, -0.25]


def test_a_float32_input_is_not_reinterpreted_as_pcm() -> None:
    """The default is 16-bit PCM even though the pipeline's array is float32."""

    params, _ = _read(encode_wav_bytes(np.zeros((1, 4), dtype=np.float32), 48000))
    assert params.sampwidth == 2


@pytest.mark.parametrize(
    "audio, sample_rate, bit_depth",
    [
        (np.zeros((0, 4), dtype=np.float32), 48000, 16),
        (np.zeros((1, 2, 2), dtype=np.float32), 48000, 16),
        (np.zeros((1, 4), dtype=np.float32), 0, 16),
        (np.zeros((1, 4), dtype=np.float32), 48000, 24),
        (np.array([[np.nan, 0.0]], dtype=np.float32), 48000, 16),
        (np.array([[np.inf, 0.0]], dtype=np.float32), 48000, 16),
    ],
)
def test_an_unencodable_array_is_refused(audio: np.ndarray, sample_rate: int, bit_depth: int) -> None:
    with pytest.raises(ValueError):
        encode_wav_bytes(audio, sample_rate, bit_depth=bit_depth)
