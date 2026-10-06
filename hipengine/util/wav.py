"""PCM to WAV, so an audio response needs no third-party encoder.

The pipeline returns unclipped FP32 ``[channels, samples]``. WAV carries that as
interleaved samples, and the 16-bit form is what players and browsers expect, so
this is the whole transport conversion: clip, scale, interleave, wrap.
"""

from __future__ import annotations

import io
import wave

import numpy as np

#: Full-scale magnitude of a signed 16-bit sample.
INT16_FULL_SCALE = 32767


def encode_wav_bytes(
    audio: np.ndarray,
    sample_rate: int,
    *,
    bit_depth: int = 16,
) -> bytes:
    """Encode ``[channels, samples]`` audio as a WAV file body.

    ``bit_depth`` selects 16-bit PCM (the default) or 32-bit float samples.
    Values outside ``[-1, 1]`` are clipped rather than wrapped, because the
    pipeline's output is unclipped by contract and a wrap would be audible
    distortion rather than the loud sample the model produced.
    """

    values = np.asarray(audio, dtype=np.float32)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2:
        raise ValueError("audio must be [channels, samples] or a single channel")
    channels, frames = (int(values.shape[0]), int(values.shape[1]))
    if channels < 1:
        raise ValueError("audio must carry at least one channel")
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
        raise ValueError("sample_rate must be a positive integer")
    if not np.isfinite(values).all():
        raise ValueError("audio must be finite to encode")

    # WAV is interleaved by frame, so a channel-first array is transposed first.
    interleaved = values.T
    if bit_depth == 16:
        scaled = np.clip(interleaved, -1.0, 1.0) * INT16_FULL_SCALE
        payload = np.rint(scaled).astype("<i2").tobytes()
        sample_width = 2
    elif bit_depth == 32:
        payload = np.ascontiguousarray(interleaved, dtype="<f4").tobytes()
        sample_width = 4
    else:
        raise ValueError("bit_depth must be 16 (PCM) or 32 (float)")

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(sample_width)
        handle.setframerate(sample_rate)
        handle.writeframes(payload)
    return buffer.getvalue()
