"""GPU tests for the device-resident VibeVoice codec decoder (gfx1100).

The decoder module (hipengine/models/vibevoice_codec.py) is geometry-
parametric, so these tests run a small synthetic geometry with random weights
against the CPU reference (hipengine/kernels/cpu_reference/vibevoice_codec.py)
on identical inputs. The pinned-checkpoint end-to-end validation lives in
scripts/vibevoice_codec_gpu_reference_check.py (oracle venv). Continuous fp32
outputs across different accumulation orders are gated with calibrated
tolerances; streaming-vs-full on device is expected to be bit-identical
because every output element is computed by the same kernel code path over
the same effective window. Skips without a ROCm device.
"""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

from hipengine.kernels.cpu_reference.vibevoice_codec import (
    DecoderGeometry,
    expected_decoder_weight_shapes,
)
from hipengine.kernels.cpu_reference import vibevoice_codec as ref

hip_runtime = pytest.importorskip("hipengine.core.hip").get_hip_runtime()

if hip_runtime is None:  # pragma: no cover - CI without ROCm
    pytest.skip("no HIP runtime available", allow_module_level=True)

ctypes.CDLL("libamdhip64.so")

from hipengine.models.vibevoice_codec import VibeVoiceCodecDecoderDevice  # noqa: E402

SYNTHETIC_GEOMETRY = DecoderGeometry(
    latent_dim=8,
    n_filters=8,
    ratios=(3,),
    depths=(2, 2),
    kernel_size=3,
    last_kernel_size=3,
)

TOLERANCE = 1e-5  # measured drift for this geometry; see test module docstring


def _random_weights(geometry: DecoderGeometry, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    weights = {}
    for name, shape in expected_decoder_weight_shapes(geometry).items():
        weights[name] = (rng.standard_normal(shape) * 0.1).astype(np.float32)
    return weights


def _make_decoder(seed: int = 7, frames: int = 6) -> VibeVoiceCodecDecoderDevice:
    return VibeVoiceCodecDecoderDevice(
        _random_weights(SYNTHETIC_GEOMETRY, seed),
        geometry=SYNTHETIC_GEOMETRY,
        max_chunk_frames=frames,
    )


def _latents(frames: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(1000 + seed)
    return (rng.standard_normal((frames, SYNTHETIC_GEOMETRY.latent_dim)) * 0.5).astype(
        np.float32
    )


class TestDeviceDecoderAgainstReference:
    def test_full_decode_matches_reference(self):
        decoder = _make_decoder()
        try:
            latent = _latents(5, 1)
            got = decoder.decode_full(latent)
            expected = ref.decode_full_sequence(
                latent, _random_weights(SYNTHETIC_GEOMETRY, 7), SYNTHETIC_GEOMETRY
            )
            assert got.shape == expected.shape
            drift = np.abs(got - expected).max()
            assert drift < TOLERANCE, f"full decode drift {drift}"
        finally:
            decoder.close()

    def test_streaming_decode_matches_reference_frame_by_frame(self):
        weights = _random_weights(SYNTHETIC_GEOMETRY, 7)
        decoder = _make_decoder()
        try:
            latent = _latents(5, 2)
            got_chunks = []
            for frame in range(latent.shape[0]):
                got_chunks.append(decoder.decode_chunk(latent[frame : frame + 1]))
            got = np.concatenate(got_chunks)

            streams = ref.DecoderStreams()
            expected_chunks = [
                ref.decode_latent_frames(
                    latent[i : i + 1], weights, streams, SYNTHETIC_GEOMETRY
                )
                for i in range(latent.shape[0])
            ]
            expected = np.concatenate(expected_chunks)
            assert got.shape == expected.shape
            drift = np.abs(got - expected).max()
            assert drift < TOLERANCE, f"streaming drift {drift}"
        finally:
            decoder.close()

    def test_streaming_matches_full_on_device(self):
        decoder = _make_decoder()
        try:
            latent = _latents(5, 3)
            streamed = np.concatenate(
                [
                    decoder.decode_chunk(latent[i : i + 1])
                    for i in range(latent.shape[0])
                ]
            )
            decoder.reset()
            full = decoder.decode_full(latent)
            assert streamed.shape == full.shape
            # Same windows, same kernel code path per element: bit-identical.
            assert np.array_equal(streamed, full), (
                f"streaming vs full drift {np.abs(streamed - full).max()}"
            )
        finally:
            decoder.close()

    def test_reset_restores_initial_state(self):
        decoder = _make_decoder()
        try:
            latent = _latents(4, 4)
            first = decoder.decode_chunk(latent)
            # Two more utterances advance the caches.
            decoder.decode_chunk(_latents(4, 5))
            decoder.decode_chunk(_latents(4, 6))
            decoder.reset()
            again = decoder.decode_chunk(latent)
            assert np.array_equal(first, again), "reset() did not restore state"
        finally:
            decoder.close()

    def test_chunked_streaming_matches_reference(self):
        weights = _random_weights(SYNTHETIC_GEOMETRY, 7)
        decoder = _make_decoder()
        try:
            latent = _latents(6, 7)
            # Two-frame chunks exercise multi-frame window assembly.
            got = np.concatenate(
                [
                    decoder.decode_chunk(latent[i : i + 2])
                    for i in range(0, latent.shape[0], 2)
                ]
            )
            streams = ref.DecoderStreams()
            expected = ref.decode_latent_frames(latent, weights, streams,
                                                SYNTHETIC_GEOMETRY)
            assert got.shape == expected.shape
            drift = np.abs(got - expected).max()
            assert drift < TOLERANCE, f"chunked streaming drift {drift}"
        finally:
            decoder.close()

    def test_full_decode_leaves_caches_untouched(self):
        # decoder A interleaves a full decode between two streaming calls;
        # decoder B runs the same two streaming calls without it. The full
        # decode must be invisible: A's second call matches B's bit-for-bit.
        decoder_a = _make_decoder()
        decoder_b = _make_decoder()
        try:
            latent = _latents(4, 8)
            filler = _latents(4, 9)
            a1 = decoder_a.decode_chunk(latent)
            b1 = decoder_b.decode_chunk(latent)
            assert np.array_equal(a1, b1)
            decoder_a.decode_full(filler)  # must not disturb A's caches
            a2 = decoder_a.decode_chunk(latent)
            b2 = decoder_b.decode_chunk(latent)
            assert np.array_equal(a2, b2), (
                "decode_full advanced the streaming caches"
            )
        finally:
            decoder_a.close()
            decoder_b.close()
